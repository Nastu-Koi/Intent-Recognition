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
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Optional, List, Dict

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool

from engine.llm_factory import load_env_file
from engine.a2a import discover_a2a_agent_cards
from engine.rbac import RoleBasedAccessControl
from engine.logging_config import get_logger, setup_logging
from engine.streaming import stream_orchestrator_graph
from orchestrator.graph import build_graph
from db.store import ConversationStore

app = FastAPI(
    title="Intent-Recognition Service",
    description="LangGraph 多智能体编排系统 — Planner-Evaluator 架构",
    version="1.0.0",
)

# ─── CORS 配置 ───
# 暴露必要的响应头，使前端能够访问 X-Session-Id
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],  # 允许前端访问 X-Session-Id 响应头
)

# ─── 初始化 ───
load_env_file(".env")

# ─── 初始化日志系统 ───
setup_logging()
logger = get_logger(__name__)

# ─── 数据库与持久化 ───
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
_CHECKPOINTER = None
_STORE = None
_DB_POOL = None


def _enable_memory_checkpointer(reason: str):
    """Use an in-process checkpointer when Postgres is unavailable."""
    global _CHECKPOINTER, _STORE
    _CHECKPOINTER = MemorySaver()
    _STORE = None
    logger.warning(
        "%s Falling back to in-memory checkpointing. "
        "Conversation context is kept until the service restarts.",
        reason,
    )


async def _init_persistence():
    global _CHECKPOINTER, _STORE, _DB_POOL
    if not DATABASE_URL:
        _enable_memory_checkpointer("DATABASE_URL not set.")
        return

    try:
        logger.info(f"Connecting to database: {DATABASE_URL}")

        # Step 1: 用一个独立的 autocommit 连接执行 setup()
        #         因为 setup() 内部有 CREATE INDEX CONCURRENTLY，不能在事务内运行
        from psycopg import AsyncConnection as PsycopgAsyncConnection
        async with await PsycopgAsyncConnection.connect(
            DATABASE_URL, autocommit=True
        ) as setup_conn:
            temp_saver = AsyncPostgresSaver(setup_conn)
            await temp_saver.setup()
            logger.info("Checkpoint tables migration complete.")

        # Step 2: 创建全局连接池 (用于运行时)
        _DB_POOL = AsyncConnectionPool(conninfo=DATABASE_URL, max_size=20, open=False)
        await _DB_POOL.open()

        # Step 3: 初始化元数据存储
        _STORE = ConversationStore(DATABASE_URL)
        _STORE.pool = _DB_POOL  # 复用连接池
        await _STORE.init_db()

        # Step 4: 创建运行时 Checkpointer (使用连接池)
        _CHECKPOINTER = AsyncPostgresSaver(_DB_POOL)
        
        logger.info("Persistence layer (PostgreSQL) ready.")
    except Exception as e:
        logger.error(f"Failed to initialize persistence: {e}")
        _enable_memory_checkpointer("PostgreSQL persistence initialization failed.")

@app.on_event("startup")
async def startup_event():
    await _init_persistence()
    # 在持久化初始化完成后立即构建图，确保 checkpointer 已就绪
    _get_graph()

@app.on_event("shutdown")
async def shutdown_event():
    if _DB_POOL:
        await _DB_POOL.close()



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


async def _save_uploaded_file(upload: UploadFile, *, current_upload: bool = False) -> Dict[str, Any]:
    """Persist an UploadFile with a unique server filename and return file metadata."""
    file_id = str(uuid.uuid4())
    original_name = Path(upload.filename or f"{file_id}.bin").name
    save_name = f"{file_id}_{original_name}"
    save_path = UPLOAD_DIR / save_name

    with open(save_path, "wb") as out:
        content = await upload.read()
        out.write(content)

    ext_lower = Path(original_name).suffix.lower().lstrip(".")
    file_info: Dict[str, Any] = {
        "file_id": file_id,
        "file_name": original_name,
        "file_path": str(save_path),
        "stored_file_name": save_name,
        "is_current_upload": current_upload,
    }
    if ext_lower in ("png", "jpg", "jpeg", "bmp", "webp", "gif"):
        file_info["file_type"] = "image"
    else:
        file_info["file_type"] = "document"
    return file_info


def _without_current_upload_marks(file_ctx: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return file_ctx with persisted current-upload markers cleared."""
    if not file_ctx:
        return file_ctx
    cleaned: Dict[str, Any] = {}
    for key in ("images", "documents"):
        items = []
        for item in file_ctx.get(key) or []:
            if isinstance(item, dict):
                next_item = item.copy()
                next_item["is_current_upload"] = False
                items.append(next_item)
            else:
                items.append(item)
        if items:
            cleaned[key] = items
    return cleaned or None


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


def _normalize_skill_item(item: Any) -> Optional[dict]:
    """Normalize eyc-skills output into the fields used by the UI."""
    if isinstance(item, str):
        name = item.strip()
        return {"name": name, "description": "", "path": ""} if name else None
    if not isinstance(item, dict):
        return None

    name = (
        item.get("name")
        or item.get("id")
        or item.get("slug")
        or item.get("skill")
        or item.get("path")
    )
    if not name:
        return None

    description = item.get("description") or item.get("summary") or item.get("desc") or ""
    return {
        "name": str(name),
        "description": str(description) if description is not None else "",
        "path": str(item.get("path") or ""),
    }


def _read_skill_description(skill_path: str) -> str:
    """Read description from a skill's SKILL.md front matter when CLI omits it."""
    if not skill_path:
        return ""

    skill_md = Path(skill_path) / "SKILL.md"
    if not skill_md.exists():
        return ""

    try:
        lines = skill_md.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "---":
                break
            if stripped.startswith("description:"):
                return stripped.split(":", 1)[1].strip().strip('"').strip("'")

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped != "---":
            return stripped[:240]
    return ""


def _load_skill_context(skill_name: Optional[str]) -> Optional[dict]:
    """Load the selected skill's full SKILL.md for system prompt injection."""
    if not skill_name:
        return None

    normalized = skill_name.strip()
    if not normalized or normalized != Path(normalized).name:
        return None

    skill_dir = PROJECT_ROOT / "skills" / ".eyc" / "skills" / normalized
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        logger.warning("[Skills] Selected skill not found: %s", normalized)
        return None

    try:
        instruction = skill_md.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("[Skills] Failed to read %s: %s", skill_md, exc)
        return None

    return {
        "name": normalized,
        "path": str(skill_dir),
        "description": _read_skill_description(str(skill_dir)),
        "instruction": instruction,
    }


def _extract_skills_from_payload(payload: Any) -> list[dict]:
    """Handle common JSON shapes returned by skill CLIs."""
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = (
            payload.get("skills")
            or payload.get("items")
            or payload.get("data")
            or payload.get("results")
            or []
        )
    else:
        candidates = []

    if isinstance(candidates, dict):
        candidates = candidates.values()

    skills = []
    seen = set()
    for item in candidates:
        skill = _normalize_skill_item(item)
        if not skill:
            continue
        if not skill.get("description"):
            skill["description"] = _read_skill_description(skill.get("path", ""))
        key = skill["name"]
        if key in seen:
            continue
        seen.add(key)
        skills.append(skill)
    return skills


# ─── LangGraph 图 (延迟初始化) ───
_GRAPH = None


def _get_graph():
    """延迟构建 LangGraph 图。"""
    global _GRAPH
    if _GRAPH is None:
        logger.info("Building LangGraph orchestrator...")
        # 传入数据库 Checkpointer 实现持久化
        _GRAPH = build_graph(checkpointer=_CHECKPOINTER)
        logger.info("LangGraph orchestrator ready.")
    return _GRAPH


async def _get_checkpoint_file_ctx(session_id: str) -> Optional[Dict[str, Any]]:
    """
    从已有 checkpoint 中加载 file_ctx，用于多轮对话中保留文件上下文。

    当文本端点（无新文件上传）被调用时，从上一轮 checkpoint 中恢复 file_ctx，
    避免因 initial_state 中 file_ctx=None 而丢失多轮文件信息。
    """
    if not _CHECKPOINTER or not session_id:
        return None
    try:
        checkpoint_tuple = await _CHECKPOINTER.aget_tuple(
            {"configurable": {"thread_id": session_id}}
        )
        if checkpoint_tuple:
            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            return _without_current_upload_marks(channel_values.get("file_ctx"))
    except Exception:
        pass
    return None


async def _get_checkpoint_messages(session_id: str) -> List[Any]:
    """
    从已有 checkpoint 中加载完整的消息历史，用于多轮对话中保留上下文。
    
    这样可以让 Conversation Router 看到完整的对话历史，正确判断当前输入与前文的关系。
    """
    if not _CHECKPOINTER or not session_id:
        return []
    try:
        checkpoint_tuple = await _CHECKPOINTER.aget_tuple(
            {"configurable": {"thread_id": session_id}}
        )
        if checkpoint_tuple:
            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])
            logger.debug(f"[_get_checkpoint_messages] session={session_id} restored {len(messages)} messages")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                msg_content = msg.content if hasattr(msg, "content") else str(msg)[:100]
                logger.debug(f"  [{i}] {msg_type}: {msg_content}")
            return messages
        else:
            logger.debug(f"[_get_checkpoint_messages] session={session_id} no checkpoint found")
    except Exception as e:
        logger.error(f"[_get_checkpoint_messages] session={session_id} error: {e}")
    return []


async def _get_checkpoint_state(session_id: str) -> Dict[str, Any]:
    """
    从 checkpoint 中恢复完整的 orchestrator state，包括上一轮的思维链、评估结果等。
    
    这样下一轮 Conversation Router 和 Planner 就能看到上一轮的所有思考过程。
    """
    if not _CHECKPOINTER or not session_id:
        logger.debug(f"[_get_checkpoint_state] no checkpointer or session_id for {session_id}")
        return {}
    
    try:
        checkpoint_tuple = await _CHECKPOINTER.aget_tuple(
            {"configurable": {"thread_id": session_id}}
        )
        if checkpoint_tuple:
            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            restored_state = {
                "messages": channel_values.get("messages", []),
                "thinking_chain": channel_values.get("thinking_chain", []),
                "eval_action": channel_values.get("eval_action", ""),
                "eval_thought": channel_values.get("eval_thought", ""),
                "final_text": channel_values.get("final_text", ""),
                "plan": channel_values.get("plan", {}),
                "results": channel_values.get("results", {}),
                "_agent_outputs": channel_values.get("_agent_outputs", {}),
                "feedback_history": channel_values.get("feedback_history", []),
                "conversation_route": channel_values.get("conversation_route", {}),
                "human_gate_response": channel_values.get("human_gate_response", {}),
                "iter": channel_values.get("iter", 0),
            }
            logger.info(f"[_get_checkpoint_state] session={session_id} restored state:")
            logger.info(f"  messages={len(restored_state['messages'])}, "
                       f"thinking_chain={len(restored_state['thinking_chain'])}, "
                       f"eval_action={bool(restored_state['eval_action'])}, "
                       f"iter={restored_state['iter']}")
            for i, msg in enumerate(restored_state["messages"]):
                msg_type = type(msg).__name__
                msg_content = msg.content if hasattr(msg, "content") else str(msg)[:100]
                logger.info(f"  restored_msg[{i}] {msg_type}: {msg_content}")
            return restored_state
        else:
            logger.info(f"[_get_checkpoint_state] session={session_id} no checkpoint found")
    except Exception as e:
        logger.error(f"[_get_checkpoint_state] session={session_id} error: {e}")
    
    return {}


def _extract_dify_conversation_id(state: Dict[str, Any]) -> str:
    """Find a Dify conversation_id returned by a sub-agent."""
    agent_outputs = state.get("_agent_outputs") or {}
    for output in agent_outputs.values():
        if isinstance(output, dict) and output.get("conversation_id"):
            return str(output["conversation_id"])
    return ""


def _interrupt_human_gate(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract human_gate payload from a LangGraph interrupt result."""
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return {}
    value = getattr(interrupts[0], "value", {}) or {}
    if not isinstance(value, dict):
        return {}
    return value.get("human_gate") or {}


def _human_gate_answer(human_gate: Dict[str, Any]) -> str:
    reason = str(human_gate.get("reason") or "继续执行前需要你确认一点信息。").strip()
    questions = [str(q).strip() for q in (human_gate.get("questions") or []) if str(q).strip()]
    proposed_plan = [str(s).strip() for s in (human_gate.get("proposed_plan") or []) if str(s).strip()]
    lines = [reason]
    if questions:
        lines.append("")
        lines.append("我需要你确认：")
        lines.extend(f"{idx}. {question}" for idx, question in enumerate(questions, start=1))
    if proposed_plan:
        lines.append("")
        lines.append("确认后我会按这个方向继续：")
        lines.extend(f"{idx}. {step}" for idx, step in enumerate(proposed_plan, start=1))
    return "\n".join(lines)


def _chat_response_from_result(result: Dict[str, Any], session_id: str) -> dict:
    return {
        "answer": result.get("final_text", "未能生成回复"),
        "session_id": session_id,
        "iterations": result.get("iter", 0),
        "plan_rationale": (result.get("plan") or {}).get("rationale", ""),
        "human_gate": (result.get("plan") or {}).get("human_gate", {}),
        "eval_action": result.get("eval_action", ""),
        "eval_thought": result.get("eval_thought", ""),
        "agent_results": {
            k: str(v)
            for k, v in (result.get("results") or {}).items()
            if not k.startswith("_")
        },
        "thinking_chain": result.get("thinking_chain", []),
    }


async def _persist_conversation_metadata(
    *,
    session_id: str,
    title: str,
    role: str,
    result: Dict[str, Any],
) -> None:
    if not _STORE:
        return
    dify_conversation_id = _extract_dify_conversation_id(result)
    if dify_conversation_id:
        await _STORE.set_conversation_id(session_id, dify_conversation_id)
    messages = result.get("messages", [])
    final_text = result.get("final_text", "")
    asyncio.create_task(_STORE.upsert_conversation(
        session_id=session_id,
        title=title[:40] if len(title) > 40 else title,
        role=role or "default",
        message_count=len(messages),
        last_reply=final_text[:200] if final_text else "",
    ))


async def _checkpoint_role(session_id: str) -> str:
    try:
        checkpoint_tuple = await _CHECKPOINTER.aget_tuple(
            {"configurable": {"thread_id": session_id}}
        ) if _CHECKPOINTER else None
        if checkpoint_tuple:
            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            return channel_values.get("role", "") or ""
    except Exception as e:
        logger.warning("[Resume] Failed to restore role from checkpoint: %s", e)
    return ""


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入文本")
    role: Optional[str] = Field(default=None, description="用户角色 (RBAC)")
    session_id: Optional[str] = Field(default=None, description="会话 ID (多轮对话)")
    selected_skill: Optional[str] = Field(default=None, description="前端选中的 skill 名称")


class ChatResponse(BaseModel):
    answer: str = Field(description="最终回复文本")
    session_id: str = Field(description="会话 ID")
    iterations: int = Field(description="迭代轮次")
    plan_rationale: str = Field(default="", description="Planner 规划思路")
    human_gate: dict = Field(default_factory=dict, description="Planner human-in-the-loop gate 决策")
    eval_action: str = Field(default="", description="Evaluator 最终决策")
    eval_thought: str = Field(default="", description="Evaluator 思考过程")
    agent_results: dict = Field(default_factory=dict, description="各 Agent 执行结果")
    thinking_chain: List[dict] = Field(default_factory=list, description="完整思维链历史")


class HumanGateResumeRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="需要 resume 的会话 ID")
    action: str = Field(..., description="用户动作: approve / deny / supplement")
    message: str = Field(default="", description="用户确认文本或补充信息")


class FinalEvalRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="会话 ID")
    action: str = Field(..., description="最终答案评价动作: accepted / revise / replan")
    message: str = Field(default="", description="修改意见或重新规划原因")


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


@app.get("/skills")
def list_installed_skills():
    """List installed EYC skills for the frontend selector."""
    env = os.environ.copy()
    node22_bin = Path("/opt/homebrew/opt/node@22/bin")
    if node22_bin.exists():
        env["PATH"] = f"{node22_bin}{os.pathsep}{env.get('PATH', '')}"
    skills_cwd = PROJECT_ROOT / "skills"
    command_cwd = skills_cwd if (skills_cwd / ".eyc" / "skills").exists() else PROJECT_ROOT

    try:
        result = subprocess.run(
            ["eyc-skills", "list", "-a", "eyc", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=env,
            cwd=command_cwd,
        )
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            content={
                "skills": [],
                "error": "eyc-skills command not found",
            },
        )
    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=504,
            content={
                "skills": [],
                "error": "eyc-skills list timed out",
            },
        )

    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "skills": [],
                "error": (result.stderr or result.stdout or "eyc-skills list failed").strip(),
            },
        )

    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "skills": [],
                "error": f"Invalid eyc-skills JSON output: {exc}",
            },
        )

    return {"skills": _extract_skills_from_payload(payload)}


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
    file_info = await _save_uploaded_file(file, current_upload=True)

    logger.info(
        "File uploaded: %s -> %s (type=%s)",
        file_info["file_name"],
        file_info["file_path"],
        file_info["file_type"],
    )

    return {
        "file_id": file_info["file_id"],
        "file_name": file_info["file_name"],
        "file_type": file_info["file_type"],
        "save_path": file_info["file_path"],
        "file_path": file_info["file_path"],
        "stored_file_name": file_info["stored_file_name"],
    }


@app.get("/uploads/{filename}")
async def download_upload(filename: str):
    """Download files generated or uploaded under uploads/."""
    safe_name = Path(filename).name
    file_path = UPLOAD_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse(status_code=404, content={"error": "file not found"})
    return FileResponse(path=str(file_path), filename=safe_name)


# ─── 会话历史 API ───

@app.get("/conversations")
async def list_conversations():
    """获取所有历史会话列表。"""
    if not _STORE:
        return {"conversations": []}
    conversations = await _STORE.list_conversations()
    return {"conversations": conversations}


@app.get("/conversations/{session_id}/messages")
async def get_conversation_history(session_id: str):
    """获取特定会话的消息历史与思考链。"""
    if not _CHECKPOINTER:
        return {"messages": [], "thinking_chain": []}
    
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        checkpoint_tuple = await _CHECKPOINTER.aget_tuple(config)
    except Exception as e:
        logger.error(f"[History] Error fetching checkpoint: {e}")
        return {"messages": [], "thinking_chain": []}
    
    if not checkpoint_tuple:
        return {"messages": [], "thinking_chain": []}
    
    # CheckpointTuple.checkpoint 包含 channel_values
    checkpoint = checkpoint_tuple.checkpoint
    channel_values = checkpoint.get("channel_values", {})
    
    messages = channel_values.get("messages", [])
    thinking_chain = channel_values.get("thinking_chain", [])
    
    # 格式化消息历史
    formatted = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(msg, HumanMessage) and (
            "【Conversation Router】" in str(content)
            or "【Human Gate】" in str(content)
        ):
            continue
        role = "user" if isinstance(msg, HumanMessage) else "agent"
        formatted.append({"role": role, "content": content})
        
    return {
        "messages": formatted,
        "thinking_chain": thinking_chain
    }


@app.post("/conversations/{session_id}/save-partial-reply")
async def save_partial_reply(session_id: str, request: Request):
    """
    保存前端收集的部分回答文本（当流被中止时调用）。
    
    Body: { "partial_reply": "..." }
    """
    if not _STORE:
        return JSONResponse(status_code=400, content={"error": "Persistence not enabled"})
    
    try:
        body = await request.json()
        partial_reply = body.get("partial_reply", "")
        
        logger.info(f"[SavePartialReply] Received partial reply for {session_id}: {len(partial_reply)} chars")
        
        # 直接更新 conversation_metadata 中的 last_reply
        async with _STORE.pool.connection() as conn:
            async with conn.cursor() as cur:
                # 只更新 last_reply 和 updated_at；如果 metadata 尚未创建，则先补一行。
                await cur.execute("""
                    INSERT INTO conversation_metadata (session_id, title, last_reply, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id) DO UPDATE SET
                        last_reply = EXCLUDED.last_reply,
                        updated_at = CURRENT_TIMESTAMP;
                """, (session_id, partial_reply[:40] or "中断的会话", partial_reply[:200]))
                await conn.commit()
                
                # 验证更新是否成功
                await cur.execute("SELECT last_reply FROM conversation_metadata WHERE session_id = %s;", (session_id,))
                result = await cur.fetchone()
                if result:
                    logger.info(f"[SavePartialReply] Verified: saved {len(result[0] or '')} chars")
        
        return {"status": "success", "saved_length": len(partial_reply)}
    except Exception as e:
        logger.error(f"[SavePartialReply] Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/conversations/{session_id}/last-reply")
async def get_last_reply(session_id: str):
    """查询保存的 last_reply (用于调试)。"""
    if not _STORE:
        return {"last_reply": ""}
    
    try:
        async with _STORE.pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute("SELECT last_reply FROM conversation_metadata WHERE session_id = %s;", (session_id,))
                result = await cur.fetchone()
                return {"last_reply": result["last_reply"] if result else ""}
    except Exception as e:
        logger.error(f"[GetLastReply] Error: {e}")
        return {"last_reply": ""}


@app.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """彻底删除会话。"""
    if not _STORE:
        return JSONResponse(status_code=400, content={"error": "Persistence not enabled"})
    
    # 1. 删除 LangGraph checkpoint 数据
    if _CHECKPOINTER:
        try:
            await _CHECKPOINTER.adelete_thread(session_id)
        except Exception as e:
            logger.warning(f"[Delete] Failed to delete checkpoints: {e}")
    
    # 2. 删除元数据
    await _STORE.delete_metadata(session_id)
    return {"status": "ok"}


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
    # 多轮对话：从 checkpoint 恢复完整状态，包括思维链、评估结果等
    file_ctx = await _get_checkpoint_file_ctx(session_id)
    checkpoint_state = await _get_checkpoint_state(session_id)
    conversation_id = ""
    if _STORE:
        conversation_id = await _STORE.get_conversation_id(session_id) or ""
    conversation_id = conversation_id or _extract_dify_conversation_id(checkpoint_state)
    skill_context = _load_skill_context(request.selected_skill)
    
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "query": request.query,
        "file_ctx": file_ctx,
        "role": request.role or "",
        "available_agents": available_agents,
        "skill_context": skill_context,
        "plan": checkpoint_state.get("plan", {}),
        "results": checkpoint_state.get("results", {}),
        "_agent_outputs": checkpoint_state.get("_agent_outputs", {}),
        "iter": checkpoint_state.get("iter", 0),
        "feedback_history": checkpoint_state.get("feedback_history", []),
        "eval_action": checkpoint_state.get("eval_action", ""),
        "eval_thought": checkpoint_state.get("eval_thought", ""),
        "final_text": checkpoint_state.get("final_text", ""),
        "thinking_chain": checkpoint_state.get("thinking_chain", []),
        "conversation_route": checkpoint_state.get("conversation_route", {}),
        "conversation_id": conversation_id,
        "human_gate_response": {},
    }

    # ─── Step 4: 执行 LangGraph 图 ───
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)
        interrupted_gate = _interrupt_human_gate(result)
        if interrupted_gate:
            return ChatResponse(
                answer=_human_gate_answer(interrupted_gate),
                session_id=session_id,
                iterations=result.get("iter", 0),
                plan_rationale=(result.get("plan") or {}).get("rationale", ""),
                human_gate=interrupted_gate,
                eval_action="NEEDS_HUMAN_INPUT",
                eval_thought=interrupted_gate.get("reason", ""),
                agent_results={},
                thinking_chain=result.get("thinking_chain", []),
            )
        
        await _persist_conversation_metadata(
            session_id=session_id,
            title=request.query,
            role=request.role or "default",
            result=result,
        )

        return ChatResponse(**_chat_response_from_result(result, session_id))
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
    selected_skill: Optional[str] = Form(None),
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
                file_info = await _save_uploaded_file(f, current_upload=True)
                if file_info["file_type"] == "image":
                    images.append(file_info)
                else:
                    documents.append(file_info)

        if images or documents:
            file_ctx = {}
            if images:
                file_ctx["images"] = images
            if documents:
                file_ctx["documents"] = documents

    # 多轮对话：若本轮未上传文件，从 checkpoint 恢复 file_ctx
    if file_ctx is None and session_id:
        file_ctx = await _get_checkpoint_file_ctx(session_id)

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

    # 构建初始 State - 从 checkpoint 恢复完整状态
    checkpoint_state = await _get_checkpoint_state(sid)
    conversation_id = ""
    if _STORE:
        conversation_id = await _STORE.get_conversation_id(sid) or ""
    conversation_id = conversation_id or _extract_dify_conversation_id(checkpoint_state)
    skill_context = _load_skill_context(selected_skill)
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "file_ctx": file_ctx,
        "role": role or "",
        "available_agents": available_agents,
        "skill_context": skill_context,
        "plan": checkpoint_state.get("plan", {}),
        "results": checkpoint_state.get("results", {}),
        "_agent_outputs": checkpoint_state.get("_agent_outputs", {}),
        "iter": checkpoint_state.get("iter", 0),
        "feedback_history": checkpoint_state.get("feedback_history", []),
        "eval_action": checkpoint_state.get("eval_action", ""),
        "eval_thought": checkpoint_state.get("eval_thought", ""),
        "final_text": checkpoint_state.get("final_text", ""),
        "thinking_chain": checkpoint_state.get("thinking_chain", []),
        "conversation_route": checkpoint_state.get("conversation_route", {}),
        "conversation_id": conversation_id,
        "human_gate_response": {},
    }

    # 执行 LangGraph 图
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": sid},
        "recursion_limit": 25,
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)
        interrupted_gate = _interrupt_human_gate(result)
        if interrupted_gate:
            return {
                "answer": _human_gate_answer(interrupted_gate),
                "session_id": sid,
                "iterations": result.get("iter", 0),
                "plan_rationale": (result.get("plan") or {}).get("rationale", ""),
                "human_gate": interrupted_gate,
                "eval_action": "NEEDS_HUMAN_INPUT",
                "eval_thought": interrupted_gate.get("reason", ""),
                "agent_results": {},
                "thinking_chain": result.get("thinking_chain", []),
            }

        await _persist_conversation_metadata(
            session_id=sid,
            title=query,
            role=role or "default",
            result=result,
        )

        return _chat_response_from_result(result, sid)
    except Exception as e:
        logger.error(f"[ChatWithFiles] Error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "session_id": sid},
        )


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天端点 — 使用 Server-Sent Events (SSE)。

    流程:
    1. RBAC 验证角色
    2. 发现并过滤 Agent Cards
    3. 构建初始 LangGraph State
    4. 使用 astream 流式执行图
    5. 实时发送思维链的每个阶段

    响应格式: text/event-stream
    事件类型:
    - start: 开始执行
    - planner: 规划节点完成，包含生成的任务
    - agent_result: 单个 Agent 执行结果
    - dispatcher: 所有 Agent 执行完成
    - evaluator: 评估节点完成，包含决策
    - final_reply: 最终回复生成
    - done: 执行完成
    - error: 错误发生
    """
    # 生成或复用 session_id
    session_id = request.session_id or str(uuid.uuid4())

    # ─── Step 1: RBAC 验证 ───
    accessible = RBAC.get_accessible_agents(request.role)
    if accessible is None:
        error_msg = f"Invalid or missing role: {request.role}"
        logger.warning(f"[ChatStream] RBAC error: {error_msg}")
        return JSONResponse(
            status_code=403,
            content={
                "error": error_msg,
                "hint": "Use /roles to see available roles",
            },
        )

    # ─── Step 2: 发现并过滤 Agent Cards ───
    all_cards = _get_agent_cards()
    filtered_cards = RBAC.filter_cards(all_cards, request.role)
    available_agents = [_card_to_prompt_dict(c) for c in filtered_cards]

    logger.info(
        f"[ChatStream] session={session_id} | role={request.role} | "
        f"client_session={request.session_id or '<new>'} | "
        f"agents={[a['agent_id'] for a in available_agents]} | "
        f"query={request.query[:100]}..."
    )

    # ─── Step 3: 构建初始 State ───
    # 多轮对话：从 checkpoint 恢复完整状态，包括思维链、评估结果等
    file_ctx = await _get_checkpoint_file_ctx(session_id)
    checkpoint_state = await _get_checkpoint_state(session_id)
    conversation_id = ""
    if _STORE:
        conversation_id = await _STORE.get_conversation_id(session_id) or ""
    conversation_id = conversation_id or _extract_dify_conversation_id(checkpoint_state)
    skill_context = _load_skill_context(request.selected_skill)
    logger.info(
        f"[ChatStream] Conversation mapping: session_id={session_id} | "
        f"conversation_id={conversation_id or '<new>'} | "
        f"client_session_id={request.session_id} | "
        f"client_provided={'YES' if request.session_id else 'NO'}"
    )
    
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "query": request.query,
        "file_ctx": file_ctx,
        "role": request.role or "",
        "available_agents": available_agents,
        "skill_context": skill_context,
        "plan": checkpoint_state.get("plan", {}),
        "results": checkpoint_state.get("results", {}),
        "_agent_outputs": checkpoint_state.get("_agent_outputs", {}),
        "iter": checkpoint_state.get("iter", 0),
        "feedback_history": checkpoint_state.get("feedback_history", []),
        "eval_action": checkpoint_state.get("eval_action", ""),
        "eval_thought": checkpoint_state.get("eval_thought", ""),
        "final_text": checkpoint_state.get("final_text", ""),
        "thinking_chain": checkpoint_state.get("thinking_chain", []),
        "conversation_route": checkpoint_state.get("conversation_route", {}),
        "conversation_id": conversation_id,
        "human_gate_response": {},
    }

    # ─── Step 4: 使用 astream 流式执行 ───
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }

    async def event_generator():
        """生成 SSE 事件流。"""
        try:
            # 流式执行图，实时发送事件
            async for event_data in stream_orchestrator_graph(graph, initial_state, config):
                yield event_data
        except Exception as e:
            logger.error(f"[ChatStream] Error during streaming: {e}", exc_info=True)
            from engine.streaming import format_sse_response
            yield format_sse_response("error", {"message": str(e), "status": "error"})
        finally:
            # 无论是正常完成还是异常中断，都要保存对话元数据
            # （实际的回答内容由前端通过 /save-partial-reply 端点保存）
            if _STORE:
                final_state = await _get_checkpoint_state(session_id)
                dify_conversation_id = _extract_dify_conversation_id(final_state)
                if dify_conversation_id:
                    await _STORE.set_conversation_id(session_id, dify_conversation_id)
                title = request.query[:40] if len(request.query) > 40 else request.query
                asyncio.create_task(_STORE.upsert_conversation(
                    session_id=session_id,
                    title=title,
                    role=request.role or "default",
                    message_count=1,
                    last_reply=""  # 由前端负责更新
                ))
                logger.info(f"[ChatStream] Conversation metadata saved (session_id={session_id})")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        }
    )


@app.post("/chat-resume", response_model=ChatResponse)
async def chat_resume(request: HumanGateResumeRequest):
    """Resume a paused LangGraph human gate and return a blocking response."""
    session_id = request.session_id
    checkpoint_state = await _get_checkpoint_state(session_id)
    if not checkpoint_state:
        return JSONResponse(
            status_code=404,
            content={"error": f"No paused conversation found for session_id={session_id}"},
        )

    role = await _checkpoint_role(session_id)
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }
    command = Command(
        resume={
            "action": request.action,
            "message": request.message,
        }
    )

    try:
        result = await graph.ainvoke(command, config=config)
        interrupted_gate = _interrupt_human_gate(result)
        if interrupted_gate:
            return ChatResponse(
                answer=_human_gate_answer(interrupted_gate),
                session_id=session_id,
                iterations=result.get("iter", 0),
                plan_rationale=(result.get("plan") or {}).get("rationale", ""),
                human_gate=interrupted_gate,
                eval_action="NEEDS_HUMAN_INPUT",
                eval_thought=interrupted_gate.get("reason", ""),
                agent_results={},
                thinking_chain=result.get("thinking_chain", []),
            )

        await _persist_conversation_metadata(
            session_id=session_id,
            title=request.message or checkpoint_state.get("query", "继续会话"),
            role=role or "default",
            result=result,
        )
        return ChatResponse(**_chat_response_from_result(result, session_id))
    except Exception as e:
        logger.error(f"[ChatResumeBlocking] Error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "session_id": session_id},
        )


@app.post("/chat-final-eval")
async def chat_final_eval(request: FinalEvalRequest):
    """Record or convert final-answer human evaluation into a follow-up instruction."""
    action = request.action.strip().lower()
    if action not in {"accepted", "revise", "replan"}:
        return JSONResponse(status_code=400, content={"error": "invalid final evaluation action"})
    if action == "accepted":
        logger.info("[FinalEval] accepted session=%s", request.session_id)
        return {"status": "accepted", "session_id": request.session_id}

    feedback = request.message.strip()
    if not feedback:
        return JSONResponse(status_code=400, content={"error": "message is required"})
    prefix = "请根据这条人工评价修改上一条最终回答，不需要重新调用工具或 Agent：" if action == "revise" else "请根据这条人工评价重新规划并必要时重新调用 Agent："
    return {
        "status": "follow_up_required",
        "session_id": request.session_id,
        "query": f"{prefix}{feedback}",
        "mode": action,
    }


@app.post("/chat-resume-stream")
async def chat_resume_stream(request: HumanGateResumeRequest):
    """Resume a paused LangGraph human gate and continue streaming the same run."""
    session_id = request.session_id
    checkpoint_state = await _get_checkpoint_state(session_id)
    if not checkpoint_state:
        return JSONResponse(
            status_code=404,
            content={"error": f"No paused conversation found for session_id={session_id}"},
        )

    all_cards = _get_agent_cards()
    role = await _checkpoint_role(session_id)
    filtered_cards = RBAC.filter_cards(all_cards, role or None)
    available_agents = [_card_to_prompt_dict(c) for c in filtered_cards]
    source_state = {"available_agents": available_agents}

    graph = _get_graph()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }
    command = Command(
        resume={
            "action": request.action,
            "message": request.message,
        }
    )

    async def event_generator():
        try:
            async for event_data in stream_orchestrator_graph(
                graph,
                command,
                config,
                agent_source_state=source_state,
            ):
                yield event_data
        except Exception as e:
            logger.error(f"[ChatResume] Error during resume streaming: {e}", exc_info=True)
            from engine.streaming import format_sse_response
            yield format_sse_response("error", {"message": str(e), "status": "error"})
        finally:
            if _STORE:
                final_state = await _get_checkpoint_state(session_id)
                dify_conversation_id = _extract_dify_conversation_id(final_state)
                if dify_conversation_id:
                    await _STORE.set_conversation_id(session_id, dify_conversation_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        }
    )


@app.post("/chat-with-files-stream")
async def chat_with_files_stream(
    query: str = Form(...),
    role: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    selected_skill: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
):
    """
    带文件的流式聊天端点 (multipart/form-data)。

    响应格式: text/event-stream (同 /chat-stream)
    """
    sid = session_id or str(uuid.uuid4())

    # 处理文件上传
    file_ctx = None
    if files:
        images = []
        documents = []
        for f in files:
            if f.filename:
                file_info = await _save_uploaded_file(f, current_upload=True)
                if file_info["file_type"] == "image":
                    images.append(file_info)
                else:
                    documents.append(file_info)

        if images or documents:
            file_ctx = {}
            if images:
                file_ctx["images"] = images
            if documents:
                file_ctx["documents"] = documents

    # 多轮对话：若本轮未上传文件，从 checkpoint 恢复 file_ctx
    if file_ctx is None and session_id:
        file_ctx = await _get_checkpoint_file_ctx(session_id)

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

    logger.info(
        f"[ChatWithFilesStream] session={sid} | role={role} | "
        f"client_session={session_id or '<new>'} | "
        f"selected_skill={selected_skill or '<none>'} | "
        f"files={len(files or [])} | file_ctx={file_ctx} | query={query[:100]}..."
    )

    # 构建初始 State - 从 checkpoint 恢复完整状态
    checkpoint_state = await _get_checkpoint_state(sid)
    conversation_id = ""
    if _STORE:
        conversation_id = await _STORE.get_conversation_id(sid) or ""
    conversation_id = conversation_id or _extract_dify_conversation_id(checkpoint_state)
    skill_context = _load_skill_context(selected_skill)
    logger.info(
        f"[ChatWithFilesStream] Conversation mapping: session_id={sid} | "
        f"conversation_id={conversation_id or '<new>'} | "
        f"client_session_id={session_id} | "
        f"client_provided={'YES' if session_id else 'NO'}"
    )
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "file_ctx": file_ctx,
        "role": role or "",
        "available_agents": available_agents,
        "skill_context": skill_context,
        "plan": checkpoint_state.get("plan", {}),
        "results": checkpoint_state.get("results", {}),
        "_agent_outputs": checkpoint_state.get("_agent_outputs", {}),
        "iter": checkpoint_state.get("iter", 0),
        "feedback_history": checkpoint_state.get("feedback_history", []),
        "eval_action": checkpoint_state.get("eval_action", ""),
        "eval_thought": checkpoint_state.get("eval_thought", ""),
        "final_text": checkpoint_state.get("final_text", ""),
        "thinking_chain": checkpoint_state.get("thinking_chain", []),
        "conversation_route": checkpoint_state.get("conversation_route", {}),
        "conversation_id": conversation_id,
        "human_gate_response": {},
    }

    # 使用 astream 流式执行
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": sid},
        "recursion_limit": 25,
    }

    async def event_generator():
        """生成 SSE 事件流。"""
        try:
            async for event_data in stream_orchestrator_graph(graph, initial_state, config):
                yield event_data
        except Exception as e:
            logger.error(f"[ChatWithFilesStream] Error: {e}", exc_info=True)
            from engine.streaming import format_sse_response
            yield format_sse_response("error", {"message": str(e), "status": "error"})
        finally:
            # 无论是正常完成还是异常中断，都要保存对话元数据
            # （实际的回答内容由前端通过 /save-partial-reply 端点保存）
            if _STORE:
                final_state = await _get_checkpoint_state(sid)
                dify_conversation_id = _extract_dify_conversation_id(final_state)
                if dify_conversation_id:
                    await _STORE.set_conversation_id(sid, dify_conversation_id)
                title = query[:40] if len(query) > 40 else query
                asyncio.create_task(_STORE.upsert_conversation(
                    session_id=sid,
                    title=title,
                    role=role or "default",
                    message_count=1,
                    last_reply=""  # 由前端负责更新
                ))
                logger.info(f"[ChatWithFilesStream] Conversation metadata saved (session_id={sid})")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": sid,
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
