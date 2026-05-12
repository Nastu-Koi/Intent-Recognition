"""
SubAgent 轻量基类 (不依赖 LangGraph SubAgent 重型基础设施)。

所有 Agent 执行器继承此类，实现 `execute(input_data)` 即可。
"""

from typing import Dict, Any
from engine.dify_client import DifyClient, query_dify_app
from engine.logging_config import get_logger

logger = get_logger(__name__)


class SubAgent:
    """
    SubAgent 基类 — 所有 Agent 执行器的公共父类。

    子类只需要实现 `execute(input_data)` 即可。
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
