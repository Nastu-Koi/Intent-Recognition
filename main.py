"""
Intent-Recognition Main Service — FastAPI 入口。

核心职责:
  1. HTTP 端点: /chat (阻塞式对话), /health, /agents, /roles
  2. RBAC 验证与 Agent Card 过滤
  3. A2A Agent Card 远程发现
  4. 文件上传 (本地存储)
  5. 构建初始 State 并执行 LangGraph 图
  6. 多轮对话记忆 (通过 session_id + MemorySaver)
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Any, Optional, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from engine.llm_factory import load_env_file
from engine.a2a import discover_a2a_agent_cards
from engine.rbac import RoleBasedAccessControl
from engine.logging_config import get_logger
from orchestrator.graph import build_graph

logger = get_logger(__name__)

# ─── 初始化 ───
load_env_file(".env")

app = FastAPI(
    title="Intent-Recognition Service",
    description="LangGraph 多智能体编排系统 — Planner-Evaluator 架构",
    version="1.0.0",
)

# 静态文件 & 模板
PROJECT_ROOT = Path(__file__).parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "web" / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "web" / "templates"))

# RBAC
RBAC = RoleBasedAccessControl()

# Agent Card 缓存
_CACHED_CARDS = None


def _get_agent_cards():
    """获取远程 A2A Agent Cards (带缓存)。"""
    global _CACHED_CARDS
    if _CACHED_CARDS is None:
        _CACHED_CARDS = discover_a2a_agent_cards()
        logger.info(f"Discovered {len(_CACHED_CARDS)} remote A2A agents")
    return _CACHED_CARDS


def _card_to_prompt_dict(card) -> dict:
    """将 AgentCard 转换为 Planner 可用的描述字典 (完整版)。"""
    a2a_meta = card.custom_attributes.get("a2a", {})
    return {
        "agent_id": card.metadata.agent_id,
        "name": card.metadata.name,
        "description": card.metadata.description,
        "skills": card.capabilities.skills,
        "keywords": card.capabilities.keywords,
        "intent_patterns": card.capabilities.intent_patterns,
        "scope": card.custom_attributes.get("scope", []),
        "examples": card.custom_attributes.get("examples", []),
        "a2a_url": a2a_meta.get("url", ""),
    }


# ─── LangGraph 图 (延迟初始化) ───
_GRAPH = None


def _get_graph():
    """延迟构建 LangGraph 图。"""
    global _GRAPH
    if _GRAPH is None:
        logger.info("Building LangGraph orchestrator...")
        _GRAPH = build_graph()
        logger.info("LangGraph orchestrator ready.")
    return _GRAPH


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入文本")
    role: Optional[str] = Field(default=None, description="用户角色 (RBAC)")
    session_id: Optional[str] = Field(default=None, description="会话 ID (多轮对话)")


class ChatResponse(BaseModel):
    answer: str = Field(description="最终回复文本")
    session_id: str = Field(description="会话 ID")
    iterations: int = Field(description="迭代轮次")
    plan_rationale: str = Field(default="", description="Planner 规划思路")
    eval_action: str = Field(default="", description="Evaluator 最终决策")
    eval_thought: str = Field(default="", description="Evaluator 思考过程")
    agent_results: dict = Field(default_factory=dict, description="各 Agent 执行结果")
    thinking_chain: List[dict] = Field(default_factory=list, description="完整思维链历史")


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Web UI 入口页面。"""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health():
    """健康检查。"""
    cards = _get_agent_cards()
    return {
        "status": "ok",
        "agents_discovered": len(cards),
        "rbac_roles": len(RBAC.roles),
    }


@app.get("/agents")
def list_agents():
    """列出所有已发现的 A2A Agents。"""
    cards = _get_agent_cards()
    return {
        "agents": [
            {
                "agent_id": c.metadata.agent_id,
                "name": c.metadata.name,
                "description": c.metadata.description,
                "skills": c.capabilities.skills,
                "keywords": c.capabilities.keywords,
            }
            for c in cards
        ]
    }


@app.get("/roles")
def list_roles():
    """列出所有可用角色。"""
    return {
        "roles": RBAC.list_all_roles(),
        "default_role": RBAC.default_role,
    }


@app.post("/refresh-agents")
def refresh_agents():
    """手动刷新 Agent Card 发现缓存。"""
    global _CACHED_CARDS
    _CACHED_CARDS = None
    cards = _get_agent_cards()
    return {
        "status": "ok",
        "agents_discovered": len(cards),
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    文件上传端点 — 存储到本地 uploads/ 目录。

    返回文件元信息，由前端在后续 /chat 请求中引用。
    """
    file_id = str(uuid.uuid4())
    ext = Path(file.filename or "file").suffix
    save_name = f"{file_id}{ext}"
    save_path = UPLOAD_DIR / save_name

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 判断文件类型
    ext_lower = ext.lower().lstrip(".")
    file_type = "image" if ext_lower in ("png", "jpg", "jpeg", "bmp", "webp", "gif") else "document"

    logger.info(f"File uploaded: {file.filename} -> {save_path} (type={file_type})")

    return {
        "file_id": file_id,
        "file_name": file.filename,
        "file_type": file_type,
        "save_path": str(save_path),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    主聊天端点 — 阻塞式返回。

    流程:
    1. RBAC 验证角色
    2. 发现并过滤 Agent Cards
    3. 构建初始 LangGraph State
    4. 执行 LangGraph 图
    5. 返回最终结果
    """
    # 生成或复用 session_id
    session_id = request.session_id or str(uuid.uuid4())

    # ─── Step 1: RBAC 验证 ───
    accessible = RBAC.get_accessible_agents(request.role)
    if accessible is None:
        return JSONResponse(
            status_code=403,
            content={
                "error": f"Invalid or missing role: {request.role}",
                "hint": "Use /roles to see available roles",
            },
        )

    # ─── Step 2: 发现并过滤 Agent Cards ───
    all_cards = _get_agent_cards()
    filtered_cards = RBAC.filter_cards(all_cards, request.role)

    # 转换为 Planner 可用的描述字典 (完整版 with a2a_url)
    available_agents = [_card_to_prompt_dict(c) for c in filtered_cards]

    logger.info(
        f"[Chat] session={session_id} | role={request.role} | "
        f"agents={[a['agent_id'] for a in available_agents]} | "
        f"query={request.query[:100]}..."
    )

    # ─── Step 3: 构建初始 State ───
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "query": request.query,
        "file_ctx": None,  # 文件通过 /upload 上传后通过 chat_with_files 传入
        "role": request.role or "",
        "available_agents": available_agents,
        "plan": {},
        "results": {},
        "iter": 0,
        "feedback_history": [],
        "eval_action": "",
        "eval_thought": "",
        "final_text": "",
        "thinking_chain": [],
    }

    # ─── Step 4: 执行 LangGraph 图 ───
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)

        return ChatResponse(
            answer=result.get("final_text", "未能生成回复"),
            session_id=session_id,
            iterations=result.get("iter", 0),
            plan_rationale=(result.get("plan") or {}).get("rationale", ""),
            eval_action=result.get("eval_action", ""),
            eval_thought=result.get("eval_thought", ""),
            agent_results={
                k: str(v)[:500]
                for k, v in (result.get("results") or {}).items()
                if not k.startswith("_")
            },
            thinking_chain=result.get("thinking_chain", []),
        )
    except Exception as e:
        logger.error(f"[Chat] Error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "session_id": session_id},
        )


@app.post("/chat-with-files")
async def chat_with_files(
    query: str = Form(...),
    role: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
):
    """
    带文件上传的聊天端点 (multipart/form-data)。

    文件存储到本地 uploads/ 目录，文件信息注入到 file_ctx。
    """
    sid = session_id or str(uuid.uuid4())

    # 处理文件上传
    file_ctx = None
    if files:
        images = []
        documents = []
        for f in files:
            if f.filename:
                file_id = str(uuid.uuid4())
                ext = Path(f.filename).suffix
                save_name = f"{file_id}{ext}"
                save_path = UPLOAD_DIR / save_name

                with open(save_path, "wb") as out:
                    content = await f.read()
                    out.write(content)

                ext_lower = ext.lower().lstrip(".")
                file_info = {
                    "file_id": file_id,
                    "file_name": f.filename,
                    "file_path": str(save_path),
                }

                if ext_lower in ("png", "jpg", "jpeg", "bmp", "webp", "gif"):
                    file_info["file_type"] = "image"
                    images.append(file_info)
                else:
                    file_info["file_type"] = "document"
                    documents.append(file_info)

        if images or documents:
            file_ctx = {}
            if images:
                file_ctx["images"] = images
            if documents:
                file_ctx["documents"] = documents

    # RBAC 验证
    accessible = RBAC.get_accessible_agents(role)
    if accessible is None:
        return JSONResponse(
            status_code=403,
            content={"error": f"Invalid or missing role: {role}"},
        )

    # 发现并过滤 Agent Cards
    all_cards = _get_agent_cards()
    filtered_cards = RBAC.filter_cards(all_cards, role)
    available_agents = [_card_to_prompt_dict(c) for c in filtered_cards]

    # 构建初始 State
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "file_ctx": file_ctx,
        "role": role or "",
        "available_agents": available_agents,
        "plan": {},
        "results": {},
        "iter": 0,
        "feedback_history": [],
        "eval_action": "",
        "eval_thought": "",
        "final_text": "",
        "thinking_chain": [],
    }

    # 执行 LangGraph 图
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": sid},
        "recursion_limit": 25,
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)

        return {
            "answer": result.get("final_text", "未能生成回复"),
            "session_id": sid,
            "iterations": result.get("iter", 0),
            "plan_rationale": (result.get("plan") or {}).get("rationale", ""),
            "eval_action": result.get("eval_action", ""),
            "eval_thought": result.get("eval_thought", ""),
            "agent_results": {
                k: str(v)[:500]
                for k, v in (result.get("results") or {}).items()
                if not k.startswith("_")
            },
            "thinking_chain": result.get("thinking_chain", []),
        }
    except Exception as e:
        logger.error(f"[ChatWithFiles] Error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "session_id": sid},
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
