"""
Dify 图片识别 SubAgent

调用 Dify Vision App 对图片进行内容识别、OCR文字提取、发票识别等。

重要依赖:
  使用前需通过 dify_file_uploader 将图片上传到 Dify 并获取 file_id。
  Planner 应确保此 Agent 在 dify_file_uploader 之后调用。

配置环境变量:
  DIFY_DIFY_VISION_API_KEY 或 DIFY_API_KEY
  DIFY_VISION_APP_TYPE - chat 或 workflow (默认 chat，Vision 通常是 Chat App)
  DIFY_API_BASE_URL
"""

import os
from typing import Dict, Any, List

import requests

from engine.dify_subagent import DifySubAgent
from engine.logging_config import get_logger

logger = get_logger(__name__)

AGENT_ID = "dify_vision"


def _get_dify_base_url() -> str:
    return (
        os.getenv("DIFY_API_BASE_URL")
        or os.getenv("DIFY_BASE_URL")
        or "https://api.dify.ai/v1"
    ).rstrip("/")


def _get_dify_api_key() -> str:
    return (
        os.getenv("DIFY_DIFY_VISION_API_KEY")
        or os.getenv("DIFY_VISION_API_KEY")
        or os.getenv("DIFY_API_KEY")
        or ""
    )


def call_dify_vision(
    query: str,
    file_ids: List[str],
    user: str = "intent-recognition",
    app_type: str = "chat",
) -> str:
    """
    调用 Dify Vision App（Chat 模式，支持图片文件附件）。

    Args:
        query: 识别指令
        file_ids: Dify file_id 列表（由 dify_file_uploader 上传后得到）
        user: 用户标识
        app_type: chat 或 workflow

    Returns:
        识别结果文本
    """
    api_key = _get_dify_api_key()
    if not api_key:
        raise ValueError("DIFY_API_KEY is not set. Please configure DIFY_DIFY_VISION_API_KEY or DIFY_API_KEY.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 构建 files 参数（Dify Chat API vision 格式）
    files_payload = [
        {
            "type": "image",
            "transfer_method": "local_file",
            "upload_file_id": fid,
        }
        for fid in file_ids
    ]

    if app_type == "workflow":
        url = f"{_get_dify_base_url()}/workflows/run"
        payload = {
            "inputs": {"query": query},
            "response_mode": "blocking",
            "user": user,
            "files": files_payload,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        outputs = (data.get("data") or {}).get("outputs") or data.get("outputs") or {}
        if isinstance(outputs, dict):
            for key in ("answer", "result", "text", "output"):
                if key in outputs:
                    return str(outputs[key])
        return str(outputs or data)

    else:  # chat
        url = f"{_get_dify_base_url()}/chat-messages"
        payload = {
            "inputs": {},
            "query": query,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": user,
            "files": files_payload,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer") or data.get("text") or str(data)


class DifyVisionAgent(DifySubAgent):
    """
    Dify 图片识别 Agent。

    调用 Dify Vision App 完成:
    - 发票/票据 OCR 识别
    - 图片内容描述与分析
    - 截图文字提取
    - 场景理解与分析

    注意: 需要图片已通过 dify_file_uploader 上传到 Dify 并有 file_id。
    如果直接传入本地路径而没有 file_id，此 Agent 会返回错误并建议先运行 dify_file_uploader。
    """

    def __init__(self):
        super().__init__(agent_id=AGENT_ID)
        self.app_type = os.getenv("DIFY_VISION_APP_TYPE", "chat")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行图片识别。

        input_data 格式:
          {
            "query": "请识别这张发票上的金额和抬头",
            "context": {
              "file_id": "...",            # 单个 Dify file_id
              "file_ids": ["...", "..."],  # 多个 Dify file_id
              "uploaded_files": [          # dify_file_uploader 的完整输出
                {"file_id": "...", "file_name": "invoice.jpg", ...}
              ],
              "metadata": {...}
            }
          }
        """
        query = input_data.get("query", "请描述这张图片的内容")
        context = input_data.get("context", {})

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

        # 诊断日志：跟踪上游文件 ID 是否正常传递
        prior_structured_keys = list(context.get("prior_structured", {}).keys())
        logger.info(
            f"[DifyVision] 接收到的 context:\n"
            f"  file_ids: {file_ids}\n"
            f"  uploaded_files 数量: {len(context.get('uploaded_files', []))}\n"
            f"  prior_structured keys: {prior_structured_keys}"
        )

        if not file_ids:
            return {
                "status": "error",
                "error": (
                    "未提供 Dify file_id。图片识别需要先通过 dify_file_uploader 上传图片到 Dify 获取 file_id。"
                    "请 Planner 在调度 dify_vision 前先调度 dify_file_uploader。"
                ),
                "agent": self.agent_id,
                "hint": "需要先执行 dify_file_uploader",
            }

        logger.info(f"[DifyVision] query={query[:100]}... | file_ids={file_ids}")

        try:
            result = call_dify_vision(
                query=query,
                file_ids=file_ids,
                user="intent-recognition",
                app_type=self.app_type,
            )
            return {
                "status": "success",
                "result": result,
                "agent": self.agent_id,
            }
        except Exception as e:
            logger.error(f"[DifyVision] Error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.agent_id,
            }
