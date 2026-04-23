"""
Workers Node — A2A JSON-RPC 并发执行节点。

职责:
  - 解析 Planner 的 plan
  - 通过 A2A 协议并发调用远程 SubAgent
  - 利用 operator.ior 自动增量合并结果
"""

import asyncio
import aiohttp
import uuid
from typing import Dict, Any

from orchestrator.state import OrchestratorState
from engine.logging_config import get_logger

logger = get_logger(__name__)


async def _a2a_send_message(
    agent_info: Dict[str, Any],
    instruction: str,
    original_query: str,
    *,
    file_ctx: Dict[str, Any] | None = None,
    prior_results: Dict[str, Any] | None = None,
    prior_structured: Dict[str, Any] | None = None,
    timeout: int = 60,
) -> tuple[str, str, Dict[str, Any]]:
    """
    通过 A2A JSON-RPC 协议调用远程 SubAgent (异步)。

    Args:
        agent_info: Agent 描述信息 (含 a2a_url)
        instruction: Planner 分派的具体执行指令
        original_query: 用户原始问题
        file_ctx: 文件上下文 (含 file_path 等)
        prior_results: 之前 Worker 的执行结果 (供依赖链使用)
        prior_structured: 之前 Worker 的结构化输出 (含 file_id 等)
        timeout: 请求超时时间

    Returns:
        (agent_id, result_text, structured_output)
    """
    agent_id = agent_info.get("agent_id", "unknown")
    a2a_url = agent_info.get("a2a_url", "")

    if not a2a_url:
        return (agent_id, f"[{agent_id}] Error: A2A URL is missing", {})

    request_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": instruction}],
                "messageId": message_id,
            },
            "metadata": {
                "agent_id": agent_id,
                "original_query": original_query,
                "file_ctx": file_ctx,
                "prior_results": prior_results,
                "prior_structured": prior_structured,
            },
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                a2a_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        if data.get("error"):
            error = data["error"]
            return (agent_id, f"[{agent_id}] A2A error: {error.get('message', str(error))}", {})

        result = data.get("result", {})
        text = _extract_text_from_a2a_result(result)
        structured = _extract_structured_from_a2a_result(result)
        return (agent_id, text, structured)

    except aiohttp.ClientError as e:
        logger.error(f"[Workers] A2A HTTP error for {agent_id}: {e}")
        return (agent_id, f"[{agent_id}] HTTP 调用异常: {e}", {})
    except asyncio.TimeoutError:
        logger.error(f"[Workers] A2A timeout for {agent_id}")
        return (agent_id, f"[{agent_id}] 调用超时", {})
    except Exception as e:
        logger.error(f"[Workers] Unexpected error for {agent_id}: {e}")
        return (agent_id, f"[{agent_id}] 调用异常: {e}", {})


def _extract_text_from_a2a_result(result: dict) -> str:
    """从 A2A JSON-RPC 结果中提取文本内容。"""
    if not isinstance(result, dict):
        return str(result)

    # 从 artifacts 提取
    for artifact in result.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        text = _extract_text_from_parts(artifact.get("parts") or [])
        if text:
            return text

    # 从 status.message 提取
    status = result.get("status") or {}
    if isinstance(status, dict):
        message = status.get("message") or {}
        text = _extract_text_from_parts(message.get("parts") or [])
        if text:
            return text

    # 从顶层 parts 提取
    text = _extract_text_from_parts(result.get("parts") or [])
    if text:
        return text

    return str(result)


def _extract_text_from_parts(parts: list) -> str:
    """从 A2A parts 数组中提取文本。"""
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("kind") == "text" and part.get("text"):
            texts.append(str(part["text"]))
    return "\n".join(texts)


def _extract_structured_from_a2a_result(result: dict) -> Dict[str, Any]:
    """从 A2A task result 的 metadata 中提取结构化数据，排除 agent_id 和 error。"""
    if not isinstance(result, dict):
        return {}
    metadata = result.get("metadata") or {}
    return {k: v for k, v in metadata.items() if k not in ("agent_id", "error")}


async def workers_node(state: OrchestratorState) -> dict:
    """
    Workers 节点: 通过 A2A JSON-RPC 异步并发调用远程 SubAgents。

    - 从 state.plan 获取任务列表
    - 从 state.available_agents 获取 Agent A2A 端点信息
    - asyncio.gather() 并发执行
    - 利用 operator.ior 自动增量合并结果
    """
    plan_data = state.get("plan") or {}
    query = state.get("query", "")
    available_agents = state.get("available_agents", [])
    file_ctx = state.get("file_ctx")
    current_results = state.get("results") or {}
    current_structured = state.get("_agent_outputs") or {}

    tasks = plan_data.get("tasks", [])
    if not tasks:
        return {"results": {"_empty": "Planner 未生成任何任务。"}}

    # 构建 agent_id → agent_info 映射
    agent_by_id = {agent["agent_id"]: agent for agent in available_agents}

    # 顺序执行任务，累积结构化输出以支持依赖
    results = {}
    agent_outputs = current_structured.copy()  # 初始化为上一次的累积输出
    for task in tasks:
        target = task.get("target", "")
        instruction = task.get("instruction", "")
        agent_info = agent_by_id.get(target)

        if agent_info is None:
            logger.warning(f"[Workers] Unknown target '{target}', skipping")
            continue

        # 传递当前累积的 agent_outputs 作为 prior_structured
        result = await _a2a_send_message(
            agent_info=agent_info,
            instruction=instruction,
            original_query=query,
            file_ctx=file_ctx,
            prior_results=results,  # 当前累积的文本结果
            prior_structured=agent_outputs,  # 当前累积的结构化输出
        )

        target_name, result_text, structured = result
        results[target_name] = result_text
        if structured:
            agent_outputs[target_name] = structured

    logger.info(
        f"[Workers] 任务执行完毕。结果 keys: {list(results.keys())} | "
        f"结构化 keys: {list(agent_outputs.keys())}"
    )

    return {"results": results, "_agent_outputs": agent_outputs}
