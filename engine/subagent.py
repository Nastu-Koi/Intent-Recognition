"""
SubAgent 轻量基类 (不依赖 LangGraph SubAgent 重型基础设施)。

所有 Agent 执行器继承此类，实现 `execute(input_data)` 即可。
支持 async 场景: 可选覆写 `aexecute(input_data)` 以获得原生异步支持。
"""

import asyncio
from typing import Dict, Any
from engine.dify_client import DifyClient, query_dify_app
from engine.logging_config import get_logger

logger = get_logger(__name__)


class SubAgent:
    """
    SubAgent 基类 — 所有 Agent 执行器的公共父类。

    子类只需要实现 `execute(input_data)` 即可。
    如果子类有原生 async 实现，可覆写 `aexecute(input_data)`。
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def aexecute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行入口。默认将 sync execute() 包装到线程池中运行。
        
        子类可覆写此方法以实现原生异步执行，保留 contextvars 传播。
        """
        return await asyncio.to_thread(self.execute, input_data)

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.execute(input_data)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Execute error: {e}")
            return {"status": "error", "error": str(e), "agent": self.agent_id}

    async def acall(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步调用入口（带错误处理），等同于 __call__ 的 async 版本。"""
        try:
            return await self.aexecute(input_data)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Async execute error: {e}")
            return {"status": "error", "error": str(e), "agent": self.agent_id}

