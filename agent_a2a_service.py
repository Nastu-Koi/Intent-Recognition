"""
Standalone A2A server for one business agent.

Run one process per agent, for example:
    A2A_AGENT_ID=dify_file_uploader A2A_PORT=8101 python agent_a2a_service.py
    A2A_AGENT_ID=dify_doc_summary A2A_PORT=8102 python agent_a2a_service.py
    A2A_AGENT_ID=dify_knowledge_qa A2A_PORT=8103 python agent_a2a_service.py
    A2A_AGENT_ID=dify_vision A2A_PORT=8104 python agent_a2a_service.py
"""

from __future__ import annotations

import os
import uuid
import importlib
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from engine.a2a import local_card_to_a2a
from engine.agent_card import AgentCard
from engine.llm_factory import load_env_file


load_env_file(".env")
AGENT_ID = os.getenv("A2A_AGENT_ID", "dify_knowledge_qa")
HOST = os.getenv("A2A_HOST", "127.0.0.1")
PORT = int(os.getenv("A2A_PORT", "8103"))
PUBLIC_BASE_URL = os.getenv("A2A_PUBLIC_BASE_URL", f"http://{HOST}:{PORT}")
PROJECT_ROOT = Path(__file__).parent


def _load_local_agent(agent_id: str) -> tuple[AgentCard | None, object | None, str | None]:
    card_path = PROJECT_ROOT / "agents" / agent_id / "agent_card.yaml"
    if not card_path.exists():
        return None, None, f"agent card not found: {card_path}"

    try:
        card = AgentCard.from_yaml(str(card_path))
        if card.execution is None:
            return card, None, f"agent execution config is missing: {agent_id}"

        module = importlib.import_module(card.execution.module)
        subagent_class = getattr(module, card.execution.class_name)
        return card, subagent_class(), None
    except Exception as exc:
        return None, None, str(exc)


LOCAL_CARD, SUBAGENT, LOAD_ERROR = _load_local_agent(AGENT_ID)

app = FastAPI(title=f"A2A Agent Service - {AGENT_ID}", version="0.1.0")
TASKS: dict[str, dict[str, Any]] = {}


class JSONRPCRequest(BaseModel):
    jsonrpc: str = Field(default="2.0")
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


def _jsonrpc_result(request_id: str | int | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(
    request_id: str | int | None,
    code: int,
    message: str,
    data: Any | None = None,
) -> dict[str, Any]:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _message_text(message: dict[str, Any]) -> str:
    texts = []
    for part in message.get("parts") or []:
        if isinstance(part, dict) and part.get("kind") == "text":
            texts.append(str(part.get("text") or ""))
    return "\n".join(text for text in texts if text)


def _task_for_output(
    *,
    input_message: dict[str, Any],
    result_text: str,
    status_state: str = "completed",
    error: str | None = None,
    structured_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = str(input_message.get("taskId") or uuid.uuid4())
    context_id = str(input_message.get("contextId") or uuid.uuid4())
    agent_message = {
        "role": "agent",
        "parts": [{"kind": "text", "text": result_text}],
        "messageId": str(uuid.uuid4()),
        "taskId": task_id,
        "contextId": context_id,
    }
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": status_state,
            "message": agent_message,
        },
        "artifacts": [
            {
                "artifactId": str(uuid.uuid4()),
                "name": "response",
                "parts": [{"kind": "text", "text": result_text}],
            }
        ],
        "history": [
            {
                **input_message,
                "taskId": task_id,
                "contextId": context_id,
            }
        ],
        "kind": "task",
        "metadata": {
            "agent_id": AGENT_ID,
            "error": error,
            **structured_output,
        },
    }
    TASKS[task_id] = task
    return task


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if LOCAL_CARD is not None and SUBAGENT is not None else "error",
        "agent_id": AGENT_ID,
        "card_loaded": LOCAL_CARD is not None,
        "subagent_loaded": SUBAGENT is not None,
        "error": LOAD_ERROR,
    }


@app.get("/.well-known/agent-card.json")
def agent_card() -> dict[str, Any]:
    if LOCAL_CARD is None:
        return {"error": f"agent not found: {AGENT_ID}"}
    return local_card_to_a2a(LOCAL_CARD, base_url=PUBLIC_BASE_URL)


@app.post("/a2a/{agent_id}")
def a2a_endpoint(agent_id: str, request: JSONRPCRequest) -> dict[str, Any]:
    if agent_id != AGENT_ID:
        return _jsonrpc_error(request.id, -32602, f"agent_id mismatch: {agent_id}")

    if request.method == "tasks/get":
        task_id = str(request.params.get("id") or request.params.get("taskId") or "")
        task = TASKS.get(task_id)
        if task is None:
            return _jsonrpc_error(request.id, -32001, f"task not found: {task_id}")
        return _jsonrpc_result(request.id, task)

    if request.method != "message/send":
        return _jsonrpc_error(request.id, -32601, f"unsupported method: {request.method}")

    if SUBAGENT is None:
        return _jsonrpc_error(request.id, -32000, f"subagent not loaded: {AGENT_ID}")

    message = request.params.get("message") or {}
    query = _message_text(message)
    if not query:
        return _jsonrpc_error(request.id, -32602, "message text is required")

    # 从 A2A metadata 提取文件上下文和上游结果
    metadata = request.params.get("metadata") or {}
    file_ctx = metadata.get("file_ctx") or {}
    prior_results = metadata.get("prior_results") or {}

    # 构建 subagent context（合并文件路径、file_id 等信息）
    context: dict[str, Any] = {
        "a2a": True,
        "metadata": metadata,
        "agent_id": AGENT_ID,
    }

    # 注入文件路径 (供 dify_file_uploader 使用)
    file_paths = []
    for category in ("images", "documents"):
        for f in file_ctx.get(category, []):
            if isinstance(f, dict) and f.get("file_path"):
                file_paths.append(f["file_path"])
    if file_paths:
        context["file_paths"] = file_paths

    # 注入上游结果 (纯文本，供参考)
    if prior_results:
        context["prior_results"] = prior_results

    # 注入上游结构化数据 (含 uploaded_files / file_id 等)
    prior_structured = metadata.get("prior_structured") or {}
    if prior_structured:
        context["prior_structured"] = prior_structured

        # 自动提取 file_ids: 从任意上游 agent 的 uploaded_files 中提取
        all_file_ids = []
        all_uploaded_files = []
        for agent_id_key, agent_output in prior_structured.items():
            if isinstance(agent_output, dict):
                # uploaded_files 列表 (dify_file_uploader 返回格式)
                for uf in agent_output.get("uploaded_files", []):
                    if isinstance(uf, dict):
                        fid = uf.get("file_id") or uf.get("id")
                        if fid:
                            all_file_ids.append(fid)
                            all_uploaded_files.append(uf)
                # 单个 file_id
                if agent_output.get("file_id"):
                    all_file_ids.append(agent_output["file_id"])

        if all_file_ids:
            context["file_ids"] = all_file_ids
        if all_uploaded_files:
            context["uploaded_files"] = all_uploaded_files

    output = SUBAGENT(
        {
            "query": query,
            "context": context,
        }
    )

    # 提取结构化字段 (uploaded_files, file_id 等)，保留在 A2A metadata 中
    structured_output = {
        k: v for k, v in output.items()
        if k not in ("status", "result", "error", "query", "agent")
    }

    if output.get("status") == "success":
        task = _task_for_output(
            input_message=message,
            result_text=str(output.get("result", "")),
            structured_output=structured_output,
        )
    else:
        error_text = str(output.get("error") or "agent execution failed")
        task = _task_for_output(
            input_message=message,
            result_text=error_text,
            status_state="failed",
            error=error_text,
            structured_output=structured_output,
        )
    return _jsonrpc_result(request.id, task)


if __name__ == "__main__":
    uvicorn.run("agent_a2a_service:app", host=HOST, port=PORT, reload=False)
