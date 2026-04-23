"""
Dify 文档总结 SubAgent

调用 Dify Workflow App 对文档或长文本内容进行智能总结。

配置环境变量:
  DIFY_DOC_SUMMARY_APP_TYPE   - workflow 或 chat (默认 workflow)
  DIFY_DIFY_DOC_SUMMARY_API_KEY 或 DIFY_API_KEY
  DIFY_API_BASE_URL
"""

import os
from typing import Dict, Any, List

from engine.dify_client import query_dify_app
from engine.dify_subagent import DifySubAgent
from engine.logging_config import get_logger

logger = get_logger(__name__)

AGENT_ID = "dify_doc_summary"


class DifyDocSummaryAgent(DifySubAgent):
    """
    调用 Dify Workflow/Chat App 对文档进行总结摘要。

    支持：
    - 纯文本内容总结 (query 中包含内容)
    - 已上传文件的总结 (通过 context.file_id 传入 Dify file_id)
    - 长文档核心要点提炼
    """

    def __init__(self):
        super().__init__(agent_id=AGENT_ID)
        self.app_type = os.getenv("DIFY_DOC_SUMMARY_APP_TYPE", "workflow")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行文档总结。

        input_data 格式:
          {
            "query": "请总结以下文档: ...",
            "context": {
              "file_id": "...",           # Dify file_id (可选，由 dify_file_uploader 提供)
              "file_ids": [...],          # 多个 file_id (可选)
              "content": "文档内容...",    # 直接提供文本内容 (可选)
              "metadata": {...}
            }
          }
        """
        query = input_data.get("query", "")
        context = input_data.get("context", {})

        # 构建 Dify inputs
        inputs: Dict[str, Any] = {}

        # 如果 context 中有文档内容，注入到 inputs
        content = context.get("content", "")
        if content:
            inputs["document_content"] = content
            inputs["content"] = content

        # 收集 file_ids
        file_ids: List[str] = []

        # 从多种来源提取 file_id
        if context.get("file_id"):
            file_ids.append(context["file_id"])

        if context.get("file_ids"):
            file_ids.extend([fid for fid in context["file_ids"] if fid not in file_ids])

        # 从 dify_file_uploader 输出中提取
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
            f"[DifyDocSummary] query={query[:100]}... | "
            f"has_content={bool(content)} | file_ids={file_ids}"
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
            logger.error(f"[DifyDocSummary] Error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.agent_id,
            }
