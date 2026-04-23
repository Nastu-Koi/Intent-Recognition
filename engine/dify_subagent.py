"""
Dify SubAgent 轻量基类 (不依赖 LangGraph SubAgent 重型基础设施)。

直接通过 DifyClient 调用 Dify API，适合简单的 Dify App 封装。
"""

from typing import Dict, Any
from engine.dify_client import DifyClient, query_dify_app
from engine.logging_config import get_logger

logger = get_logger(__name__)


class DifySubAgent:
    """
    Dify SubAgent 基类。

    子类只需要实现 `execute(input_data)` 即可，
    或者直接使用 `DifySubAgent` 并传入 agent_id 和 app_type。
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.execute(input_data)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Execute error: {e}")
            return {"status": "error", "error": str(e), "agent": self.agent_id}
