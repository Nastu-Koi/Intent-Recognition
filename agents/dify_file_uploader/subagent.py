"""
Dify 文件上传 SubAgent

职责:
  - 将本地文件（图片/文档）上传到 Dify
  - 返回 Dify file_id，供 dify_doc_summary / dify_vision 等后续 Agent 使用
  - 是所有需要文件 ID 任务的前置依赖 Agent

配置环境变量:
  DIFY_API_BASE_URL  - Dify API 地址 (默认 https://api.dify.ai/v1)
  DIFY_API_KEY       - Dify API Key (全局)
  DIFY_UPLOADER_USER - 上传用户标识 (默认 intent-recognition-uploader)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

import requests

from engine.dify_subagent import DifySubAgent
from engine.logging_config import get_logger

logger = get_logger(__name__)


def _get_dify_base_url() -> str:
    return (
        os.getenv("DIFY_API_BASE_URL")
        or os.getenv("DIFY_BASE_URL")
        or "https://api.dify.ai/v1"
    ).rstrip("/")


def _get_dify_api_key() -> str:
    return (
        os.getenv("DIFY_DIFY_FILE_UPLOADER_API_KEY")
        or os.getenv("DIFY_API_KEY")
        or ""
    )


def upload_file_to_dify(file_path: str, user: str = "intent-recognition-uploader") -> Dict[str, Any]:
    """
    上传文件到 Dify，返回 file_id 等元信息。

    Args:
        file_path: 本地文件路径
        user: 上传用户标识

    Returns:
        {file_id, name, size, mime_type, ...}
    """
    api_key = _get_dify_api_key()
    if not api_key:
        raise ValueError("DIFY_API_KEY is not set in .env")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    url = f"{_get_dify_base_url()}/files/upload"
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(path, "rb") as f:
        files = {"file": (path.name, f, _guess_mime(path))}
        data = {"user": user}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        resp.raise_for_status()
        return resp.json()


def _guess_mime(path: Path) -> str:
    """根据扩展名猜测 MIME 类型。"""
    ext = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
    }
    return mime_map.get(ext, "application/octet-stream")


class DifyFileUploaderAgent(DifySubAgent):
    """
    将本地文件上传到 Dify 并返回 file_id。

    这是需要文件处理的流程中的「前置必执行」Agent。
    Planner 如果检测到用户有文件且后续需要 Dify file_id，应优先调度此 Agent。
    """

    def __init__(self):
        super().__init__(agent_id="dify_file_uploader")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行文件上传。

        input_data 格式:
          {
            "query": "...",           # 用户指令 (通常包含文件路径信息)
            "context": {
              "file_paths": [...],    # 要上传的文件路径列表 (优先使用)
              "file_path": "...",     # 单个文件路径
              "metadata": {...}       # A2A 上下文
            }
          }
        """
        context = input_data.get("context", {})
        query = input_data.get("query", "")

        # 从 context 中提取文件路径
        file_paths = context.get("file_paths", [])
        if not file_paths and context.get("file_path"):
            file_paths = [context["file_path"]]

        # 尝试从 query 中解析路径 (降级：Planner 会在 instruction 中注入)
        if not file_paths:
            import re
            # 匹配引号内或空格分隔的路径
            paths = re.findall(r"['\"]([^'\"]+\.\w+)['\"]|(\S+\.\w+)", query)
            for p1, p2 in paths:
                candidate = p1 or p2
                if Path(candidate).exists():
                    file_paths.append(candidate)

        if not file_paths:
            return {
                "status": "error",
                "error": "没有找到需要上传的文件路径。请在 context.file_paths 中提供本地文件路径。",
                "agent": self.agent_id,
            }

        user = os.getenv("DIFY_UPLOADER_USER", "intent-recognition-uploader")
        uploaded = []
        errors = []

        for fp in file_paths:
            try:
                result = upload_file_to_dify(fp, user=user)
                file_id = result.get("id") or result.get("file_id", "")
                file_name = result.get("name", Path(fp).name)
                file_type_desc = "图片" if result.get("mime_type", "").startswith('image/') else "文档"
                uploaded.append({
                    "file_path": fp,
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_type": file_type_desc,
                    "size": result.get("size", 0),
                    "mime_type": result.get("mime_type", ""),
                    "raw": result,
                })
                logger.info(f"[DifyFileUploader] Uploaded {fp} -> file_id={file_id}")
            except Exception as e:
                logger.error(f"[DifyFileUploader] Failed to upload {fp}: {e}")
                errors.append({"file_path": fp, "error": str(e)})

        if not uploaded and errors:
            return {
                "status": "error",
                "error": f"所有文件上传失败: {errors}",
                "agent": self.agent_id,
            }

        result_text = "文件上传完成。以下是 Dify file_id 映射：\n"
        for u in uploaded:
            file_type_desc = "图片" if u['mime_type'].startswith('image/') else "文档"
            result_text += f"  - {u['file_name']} ({file_type_desc}, size: {u['size']} bytes) -> file_id: {u['file_id']}\n"
        if errors:
            result_text += f"\n以下文件上传失败：\n"
            for e in errors:
                result_text += f"  - {e['file_path']}: {e['error']}\n"

        return {
            "status": "success",
            "result": result_text,
            "agent": self.agent_id,
            "uploaded_files": uploaded,
            "errors": errors,
        }
