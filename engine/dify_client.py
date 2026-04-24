"""
Dify API client.

Secrets are loaded from environment variables, not from agent_card.yaml:
- DIFY_API_BASE_URL or DIFY_BASE_URL
- DIFY_API_KEY
- DIFY_<AGENT_ID>_API_KEY, for example DIFY_FACILITY_SERVICE_API_KEY
"""

import os
from dataclasses import dataclass
from typing import Any

import requests

from engine.logging_config import get_logger

logger = get_logger(__name__)


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


def _env_key_for_agent(agent_id: str) -> str:
    normalized = agent_id.upper().replace("-", "_")
    return f"DIFY_{normalized}_API_KEY"


def _legacy_env_key_for_agent(agent_id: str) -> str:
    normalized = agent_id.upper().replace("-", "_")
    return f"{normalized}_API_KEY"


@dataclass
class DifyClient:
    api_base_url: str
    api_key: str
    timeout: int = 30

    @classmethod
    def from_env(cls, agent_id: str | None = None) -> "DifyClient":
        _load_dotenv_if_available()

        api_base_url = (
            os.getenv("DIFY_API_BASE_URL")
            or os.getenv("DIFY_BASE_URL")
            or "https://api.dify.ai/v1"
        ).rstrip("/")

        api_key = ""
        if agent_id:
            api_key = (
                os.getenv(_env_key_for_agent(agent_id), "")
                or os.getenv(_legacy_env_key_for_agent(agent_id), "")
            )
        api_key = api_key or os.getenv("DIFY_API_KEY", "")

        if not api_key:
            key_hint = _env_key_for_agent(agent_id) if agent_id else "DIFY_API_KEY"
            raise ValueError(f"Dify API key is not configured. Set {key_hint} or DIFY_API_KEY in .env")

        timeout = int(os.getenv("DIFY_TIMEOUT", "30"))

        return cls(api_base_url=api_base_url, api_key=api_key, timeout=timeout)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.api_base_url}{path}",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def chat(self, query: str, inputs: dict[str, Any] | None = None, user: str = "intent-recognition") -> str:
        # 从 inputs 中提取 file_ids 并构建 files 参数
        files = []
        processed_inputs = dict(inputs or {})
        
        # 支持 file_ids (列表) 或 file_id (单个)
        file_ids = processed_inputs.pop("file_ids", []) or []
        if processed_inputs.get("file_id") and "file_id" not in ["file_ids"]:
            single_id = processed_inputs.pop("file_id", None)
            if single_id and single_id not in file_ids:
                file_ids.insert(0, single_id)
        
        # 将 file_ids 转换为 Dify 期望的 files 格式
        if file_ids:
            for fid in file_ids:
                if isinstance(fid, str):
                    # 默认当作 document，但更好的做法是让 dify_file_uploader 在 inputs 中标明类型
                    files.append({
                        "type": "document",
                        "transfer_method": "local_file",
                        "upload_file_id": fid,
                    })
        
        data = self._post(
            "/chat-messages",
            {
                "inputs": processed_inputs,
                "query": query,
                "response_mode": "blocking",
                "conversation_id": "",
                "user": user,
                "files": files,
            },
        )
        return data.get("answer") or data.get("text") or str(data)

    def workflow(self, query: str, inputs: dict[str, Any] | None = None, user: str = "intent-recognition") -> str:
        workflow_inputs = dict(inputs or {})
        workflow_inputs.setdefault("query", query)
        
        # 从 workflow_inputs 中提取 file_ids 并构建 files 参数
        files = []
        file_ids = workflow_inputs.pop("file_ids", []) or []
        if workflow_inputs.get("file_id") and "file_id" not in ["file_ids"]:
            single_id = workflow_inputs.pop("file_id", None)
            if single_id and single_id not in file_ids:
                file_ids.insert(0, single_id)
        
        # 将 file_ids 转换为 Dify 期望的 files 格式
        if file_ids:
            for fid in file_ids:
                if isinstance(fid, str):
                    files.append({
                        "type": "document",
                        "transfer_method": "local_file",
                        "upload_file_id": fid,
                    })
        
        payload = {
            "inputs": workflow_inputs,
            "response_mode": "blocking",
            "user": user,
        }
        if files:
            payload["files"] = files
        
        data = self._post(
            "/workflows/run",
            payload,
        )
        outputs = (data.get("data") or {}).get("outputs") or data.get("outputs") or {}
        if isinstance(outputs, dict):
            for key in ("answer", "result", "text", "output"):
                if key in outputs:
                    return str(outputs[key])
        return str(outputs or data)

    def retrieve(self, dataset_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        data = self._post(
            f"/datasets/{dataset_id}/retrieve",
            {
                "query": query,
                "retrieval_model": {
                    "search_method": "semantic_search",
                    "reranking_enable": False,
                    "top_k": top_k,
                    "score_threshold_enabled": False,
                },
            },
        )
        return data.get("records", [])


def format_dify_records(records: list[dict[str, Any]], empty_message: str = "未检索到相关知识库内容") -> str:
    if not records:
        return empty_message

    lines = []
    for index, record in enumerate(records, 1):
        segment = record.get("segment") or {}
        document = record.get("document") or {}
        content = segment.get("content") or record.get("content") or ""
        source = document.get("name") or segment.get("document_id") or "Dify"
        score = record.get("score")

        prefix = f"{index}. 来源: {source}"
        if score is not None:
            prefix += f" | score: {score}"

        lines.append(prefix)
        lines.append(content.strip() if content else "无内容")

    return "\n".join(lines)


def query_dify_app(
    *,
    agent_id: str,
    query: str,
    inputs: dict[str, Any] | None = None,
    user: str = "intent-recognition",
) -> str:
    client = DifyClient.from_env(agent_id=agent_id)
    app_type = os.getenv(f"DIFY_{agent_id.upper()}_APP_TYPE", os.getenv("DIFY_APP_TYPE", "chat"))
    app_type = app_type.lower()

    if app_type == "workflow":
        return client.workflow(query=query, inputs=inputs, user=user)
    return client.chat(query=query, inputs=inputs, user=user)


def query_dify_dataset(
    *,
    agent_id: str,
    dataset_id: str,
    query: str,
    top_k: int = 3,
) -> str:
    client = DifyClient.from_env(agent_id=agent_id)
    records = client.retrieve(dataset_id=dataset_id, query=query, top_k=top_k)
    return format_dify_records(records)
