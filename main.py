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
from pathlib import Path
from typing import Any, Optional, List, Dict

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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
            return channel_values.get("file_ctx")
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
                "feedback_history": channel_values.get("feedback_history", []),
                "conversation_route": channel_values.get("conversation_route", {}),
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
    save_name = file.filename or f"{file_id}.bin"
    save_path = UPLOAD_DIR / save_name

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 判断文件类型
    ext_lower = Path(save_name).suffix.lower().lstrip(".")
    file_type = "image" if ext_lower in ("png", "jpg", "jpeg", "bmp", "webp", "gif") else "document"

    logger.info(f"File uploaded: {file.filename} -> {save_path} (type={file_type})")

    return {
        "file_id": file_id,
        "file_name": file.filename,
        "file_type": file_type,
        "save_path": str(save_path),
    }


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
        if isinstance(msg, HumanMessage) and "【Conversation Router】" in str(content):
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
    
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "query": request.query,
        "file_ctx": file_ctx,
        "role": request.role or "",
        "available_agents": available_agents,
        "plan": checkpoint_state.get("plan", {}),
        "results": checkpoint_state.get("results", {}),
        "iter": checkpoint_state.get("iter", 0),
        "feedback_history": checkpoint_state.get("feedback_history", []),
        "eval_action": checkpoint_state.get("eval_action", ""),
        "eval_thought": checkpoint_state.get("eval_thought", ""),
        "final_text": checkpoint_state.get("final_text", ""),
        "thinking_chain": checkpoint_state.get("thinking_chain", []),
        "conversation_route": checkpoint_state.get("conversation_route", {}),
    }

    # ─── Step 4: 执行 LangGraph 图 ───
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)
        
        # 更新会话元数据 (异步不阻塞回复)
        if _STORE:
            # 摘要取用户输入前 40 字
            title = request.query[:40] if len(request.query) > 40 else request.query
            messages = result.get("messages", [])
            final_text = result.get("final_text", "")
            last_reply = final_text[:200] if final_text else ""
            asyncio.create_task(_STORE.upsert_conversation(
                session_id=session_id,
                title=title,
                role=request.role or "default",
                message_count=len(messages),
                last_reply=last_reply
            ))

        return ChatResponse(
            answer=result.get("final_text", "未能生成回复"),
            session_id=session_id,
            iterations=result.get("iter", 0),
            plan_rationale=(result.get("plan") or {}).get("rationale", ""),
            eval_action=result.get("eval_action", ""),
            eval_thought=result.get("eval_thought", ""),
            agent_results={
                k: str(v)
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
                save_name = f.filename or f"{file_id}.bin"
                save_path = UPLOAD_DIR / save_name

                with open(save_path, "wb") as out:
                    content = await f.read()
                    out.write(content)

                ext_lower = Path(save_name).suffix.lower().lstrip(".")
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
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "file_ctx": file_ctx,
        "role": role or "",
        "available_agents": available_agents,
        "plan": checkpoint_state.get("plan", {}),
        "results": checkpoint_state.get("results", {}),
        "iter": checkpoint_state.get("iter", 0),
        "feedback_history": checkpoint_state.get("feedback_history", []),
        "eval_action": checkpoint_state.get("eval_action", ""),
        "eval_thought": checkpoint_state.get("eval_thought", ""),
        "final_text": checkpoint_state.get("final_text", ""),
        "thinking_chain": checkpoint_state.get("thinking_chain", []),
        "conversation_route": checkpoint_state.get("conversation_route", {}),
    }

    # 执行 LangGraph 图
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": sid},
        "recursion_limit": 25,
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)

        # 更新会话元数据
        if _STORE:
            title = query[:40] if len(query) > 40 else query
            messages = result.get("messages", [])
            final_text = result.get("final_text", "")
            last_reply = final_text[:200] if final_text else ""
            asyncio.create_task(_STORE.upsert_conversation(
                session_id=sid,
                title=title,
                role=role or "default",
                message_count=len(messages),
                last_reply=last_reply
            ))

        return {
            "answer": result.get("final_text", "未能生成回复"),
            "session_id": sid,
            "iterations": result.get("iter", 0),
            "plan_rationale": (result.get("plan") or {}).get("rationale", ""),
            "eval_action": result.get("eval_action", ""),
            "eval_thought": result.get("eval_thought", ""),
            "agent_results": {
                k: str(v)
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
    
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "query": request.query,
        "file_ctx": file_ctx,
        "role": request.role or "",
        "available_agents": available_agents,
        "plan": checkpoint_state.get("plan", {}),
        "results": checkpoint_state.get("results", {}),
        "iter": checkpoint_state.get("iter", 0),
        "feedback_history": checkpoint_state.get("feedback_history", []),
        "eval_action": checkpoint_state.get("eval_action", ""),
        "eval_thought": checkpoint_state.get("eval_thought", ""),
        "final_text": checkpoint_state.get("final_text", ""),
        "thinking_chain": checkpoint_state.get("thinking_chain", []),
        "conversation_route": checkpoint_state.get("conversation_route", {}),
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


@app.post("/chat-with-files-stream")
async def chat_with_files_stream(
    query: str = Form(...),
    role: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
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
                file_id = str(uuid.uuid4())
                save_name = f.filename or f"{file_id}.bin"
                save_path = UPLOAD_DIR / save_name

                with open(save_path, "wb") as out:
                    content = await f.read()
                    out.write(content)

                ext_lower = Path(save_name).suffix.lower().lstrip(".")
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
        f"client_session={session_id or '<new>'} | query={query[:100]}..."
    )

    # 构建初始 State - 从 checkpoint 恢复完整状态
    checkpoint_state = await _get_checkpoint_state(sid)
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "file_ctx": file_ctx,
        "role": role or "",
        "available_agents": available_agents,
        "plan": checkpoint_state.get("plan", {}),
        "results": checkpoint_state.get("results", {}),
        "iter": checkpoint_state.get("iter", 0),
        "feedback_history": checkpoint_state.get("feedback_history", []),
        "eval_action": checkpoint_state.get("eval_action", ""),
        "eval_thought": checkpoint_state.get("eval_thought", ""),
        "final_text": checkpoint_state.get("final_text", ""),
        "thinking_chain": checkpoint_state.get("thinking_chain", []),
        "conversation_route": checkpoint_state.get("conversation_route", {}),
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
