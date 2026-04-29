"""
Dify 报销助手 SubAgent

调用 Dify Workflow App 提供财务报销政策咨询、流程指导等服务。

配置环境变量:
  DIFY_DIFY_EXPENSE_ASSISTANT_APP_TYPE   - workflow 或 chat (默认 workflow)
  DIFY_DIFY_EXPENSE_ASSISTANT_API_KEY 或 DIFY_API_KEY
  DIFY_API_BASE_URL
"""

import os
from typing import Dict, Any, List

from engine.dify_client import query_dify_app
from engine.dify_subagent import DifySubAgent
from engine.logging_config import get_logger

logger = get_logger(__name__)

AGENT_ID = "dify_expense_assistant"


class DifyExpenseAssistantAgent(DifySubAgent):
    """
    调用 Dify Workflow/Chat App 提供报销助手服务。

    支持：
    - 财务报销政策咨询
    - 发票要求查询
    - 报销流程指导
    - 项目Code选择帮助
    - 差旅、交通、餐饮等报销标准
    """

    def __init__(self):
        super().__init__(agent_id=AGENT_ID)
        self.app_type = os.getenv("DIFY_DIFY_EXPENSE_ASSISTANT_APP_TYPE", "workflow")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行报销助手查询。

        input_data 格式:
          {
            "query": "请解答报销相关问题: ...",
            "context": {
              "file_id": "...",           # 相关文件 file_id (可选)
              "file_ids": [...],          # 多个 file_id (可选)
              "metadata": {...}           # 其他上下文信息
            }
          }
        """
        query = input_data.get("query", "")
        context = input_data.get("context", {})

        # 构建 Dify inputs
        inputs: Dict[str, Any] = {}

        # 收集 file_ids（如果有相关文件）
        file_ids: List[str] = []

        # 从多种来源提取 file_id
        if context.get("file_id"):
            file_ids.append(context["file_id"])

        if context.get("file_ids"):
            file_ids.extend([fid for fid in context["file_ids"] if fid not in file_ids])

        # 从上传文件中提取
        for uf in context.get("uploaded_files", []):
            if isinstance(uf, dict):
                fid = uf.get("file_id") or uf.get("id", "")
                if fid and fid not in file_ids:
                    file_ids.append(fid)

        # 如果有 file_ids，注入到 inputs
        if file_ids:
            inputs["file_ids"] = file_ids
            if len(file_ids) == 1:
                inputs["file_id"] = file_ids[0]

        logger.info(
            f"[DifyExpenseAssistant] query={query[:100]}... | file_ids={file_ids}"
        )

        try:
            result = query_dify_app(
                agent_id=AGENT_ID,
                query=query,
                inputs=inputs if inputs else None,
                user="intent-recognition",
            )
            return {
                "status": "success",
                "result": result,
                "agent": self.agent_id,
            }
        except Exception as e:
            logger.error(f"[DifyExpenseAssistant] Error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.agent_id,
            }
