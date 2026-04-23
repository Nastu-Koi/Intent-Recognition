"""
Dify 知识库问答 SubAgent

调用 Dify Chat App（绑定了知识库）进行企业内部知识的精准问答。

配置环境变量:
  DIFY_DIFY_KNOWLEDGE_QA_API_KEY 或 DIFY_API_KEY
  DIFY_KNOWLEDGE_QA_APP_TYPE - chat 或 workflow (默认 chat)
  DIFY_API_BASE_URL
"""

import os
from typing import Dict, Any

from engine.dify_client import query_dify_app
from engine.dify_subagent import DifySubAgent
from engine.logging_config import get_logger

logger = get_logger(__name__)

AGENT_ID = "dify_knowledge_qa"


class DifyKnowledgeQAAgent(DifySubAgent):
    """
    基于 Dify 知识库的企业内部问答 Agent。

    支持:
    - HR 政策、假期、福利、招聘流程
    - 财务报销制度与流程
    - IT 设备申请与使用规范
    - 行政服务与办公设施管理
    - 公司规章制度通用问答
    """

    def __init__(self):
        super().__init__(agent_id=AGENT_ID)
        self.app_type = os.getenv("DIFY_KNOWLEDGE_QA_APP_TYPE", "chat")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行知识库问答。

        input_data 格式:
          {
            "query": "年假有多少天？",
            "context": {
              "role": "employee",         # 用户角色 (可注入到 Dify inputs)
              "domain": "hr",             # 问答领域提示 (可选)
              "metadata": {...}
            }
          }
        """
        query = input_data.get("query", "")
        context = input_data.get("context", {})

        # 构建 inputs（透传角色信息，让 Dify App 可以按角色过滤知识）
        inputs: Dict[str, Any] = {}
        if context.get("role"):
            inputs["user_role"] = context["role"]
        if context.get("domain"):
            inputs["domain"] = context["domain"]

        logger.info(f"[DifyKnowledgeQA] query={query[:100]}... | role={context.get('role', 'unknown')}")

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
            logger.error(f"[DifyKnowledgeQA] Error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.agent_id,
            }
