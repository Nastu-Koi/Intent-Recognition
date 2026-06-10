"""
General Chat 工具集 — 图像识别与文档总结。

每个工具内部完成「上传文件到 Dify → 调用 Dify API」的完整流程，
调用方无需关心 Dify file_id 的获取。

工具列表:
  1. image_recognition  — 图片 OCR / 场景分析 / 发票识别
  2. document_summary   — 文档总结 / 要点提炼
"""

import os
import time
from io import BytesIO
from pathlib import Path
from typing import List

import requests
from langchain_core.tools import tool

from engine.logging_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Dify 文件上传（内部共享函数）
# ──────────────────────────────────────────────

def _get_dify_base_url() -> str:
    return (
        os.getenv("DIFY_API_BASE_URL")
        or os.getenv("DIFY_BASE_URL")
        or "https://api.dify.ai/v1"
    ).rstrip("/")


def _get_dify_api_key(agent_id: str = "") -> str:
    """
    按优先级获取 Dify API Key:
      1. DIFY_<AGENT_ID>_API_KEY  (例: DIFY_VISION_API_KEY)
      2. DIFY_API_KEY             (全局)
    """
    if agent_id:
        normalized = agent_id.upper().replace("-", "_")
        key = os.getenv(f"DIFY_{normalized}_API_KEY", "")
        if key:
            return key
    return os.getenv("DIFY_API_KEY", "")


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


# 统一的 Dify 用户标识 (上传与调用必须一致)
DIFY_USER = os.getenv("DIFY_USER", "intent-recognition")


def _upload_file_to_dify(file_path: str, api_key: str = "") -> str:
    """
    上传本地文件到 Dify，返回 file_id。

    重要: api_key 必须是目标 Dify App 的 API Key，否则上传的 file_id
    在后续调用该 App 的 chat/workflow 时会找不到文件。

    Args:
        file_path: 本地文件绝对路径
        api_key: 目标 Dify App 的 API Key (必须与后续调用 App 的 key 一致)

    Returns:
        Dify file_id 字符串

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: API Key 未配置
        requests.HTTPError: 上传失败
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    key = api_key or _get_dify_api_key()
    if not key:
        raise ValueError("DIFY_API_KEY 未配置，无法上传文件。")

    url = f"{_get_dify_base_url()}/files/upload"
    headers = {"Authorization": f"Bearer {key}"}

    with open(path, "rb") as f:
        files = {"file": (path.name, f, _guess_mime(path))}
        data = {"user": DIFY_USER}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        resp.raise_for_status()
        result = resp.json()

    file_id = result.get("id") or result.get("file_id", "")
    logger.info(f"[Tools] 文件上传成功: {path.name} -> file_id={file_id}")
    return file_id


# ──────────────────────────────────────────────
# Dify Vision 调用（内部函数）
# ──────────────────────────────────────────────

def _call_dify_vision(
    query: str,
    file_ids: List[str],
    app_type: str = "chat",
    conversation_id: str = "",
) -> str:
    """
    调用 Dify Vision App（支持 chat / workflow 模式）。

    Args:
        query: 识别指令
        file_ids: Dify file_id 列表
        app_type: chat 或 workflow
        conversation_id: Dify 会话 ID（用于多轮对话）

    Returns:
        识别结果文本
    """
    api_key = _get_dify_api_key("VISION")
    if not api_key:
        raise ValueError("DIFY_VISION_API_KEY 或 DIFY_API_KEY 未配置。")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

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
            "user": DIFY_USER,
            "files": files_payload,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
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
            "conversation_id": conversation_id,
            "user": DIFY_USER,
            "files": files_payload,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer") or data.get("text") or str(data)


# ──────────────────────────────────────────────
# Dify Doc Summary 调用（内部函数）
# ──────────────────────────────────────────────

def _call_dify_doc_summary(
    query: str,
    file_ids: List[str],
    app_type: str = "chat",
    conversation_id: str = "",
) -> str:
    """
    调用 Dify 文档总结 App。

    Args:
        query: 总结指令
        file_ids: Dify file_id 列表
        app_type: chat 或 workflow
        conversation_id: Dify 会话 ID（用于多轮对话）

    Returns:
        总结结果文本
    """
    api_key = _get_dify_api_key("DOC_SUMMARY")
    if not api_key:
        raise ValueError("DIFY_DOC_SUMMARY_API_KEY 或 DIFY_API_KEY 未配置。")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 构建 files 参数
    files_payload = [
        {
            "type": "document",
            "transfer_method": "local_file",
            "upload_file_id": fid,
        }
        for fid in file_ids
    ]

    if app_type == "workflow":
        input_variable = os.getenv("DIFY_DOC_SUMMARY_INPUT_VARIABLE", "query")
        url = f"{_get_dify_base_url()}/workflows/run"
        payload = {
            "inputs": {input_variable: query},
            "response_mode": "blocking",
            "user": DIFY_USER,
            "files": files_payload,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        outputs = (data.get("data") or {}).get("outputs") or data.get("outputs") or {}
        if isinstance(outputs, dict):
            for key in ("answer", "result", "text", "output", "summary", "summarized"):
                if key in outputs:
                    return str(outputs[key])
        return str(outputs or data)
    else:  # chat
        url = f"{_get_dify_base_url()}/chat-messages"
        payload = {
            "inputs": {},
            "query": query,
            "response_mode": "blocking",
            "conversation_id": conversation_id,
            "user": DIFY_USER,
            "files": files_payload,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer") or data.get("text") or str(data)


# ══════════════════════════════════════════════
# LangChain Tool 定义
# ══════════════════════════════════════════════

# 支持的图片扩展名
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
# 支持的文档扩展名
_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt", ".md", ".csv"}


@tool
def image_recognition(file_path: str, instruction: str = "请描述这张图片的内容") -> str:
    """识别和分析图片内容。支持 OCR 文字识别、发票识别、场景分析、图片内容描述等视觉任务。

    Args:
        file_path: 图片文件的本地路径（必须是服务器上的绝对路径）
        instruction: 识别指令，例如「请识别这张发票上的金额」、「描述图片内容」、「提取图中的文字」
    """
    logger.info(f"[Tool:image_recognition] file={file_path} | instruction={instruction[:100]}")

    # 文件类型校验：仅接受图片文件
    ext = Path(file_path).suffix.lower()
    if ext not in _IMAGE_EXTENSIONS:
        return (
            f"错误: 文件 {Path(file_path).name} 不是支持的图片格式 "
            f"(支持: {', '.join(_IMAGE_EXTENSIONS)})。"
            f"如需处理文档，请使用 document_summary 工具。"
        )

    try:
        # 获取目标 App 的 API Key（上传和调用必须使用同一个 key）
        vision_api_key = _get_dify_api_key("VISION")

        # Step 1: 使用 Vision App 的 key 上传图片到 Dify
        file_id = _upload_file_to_dify(file_path, api_key=vision_api_key)

        # Step 2: 调用 Dify Vision API
        app_type = os.getenv("DIFY_VISION_APP_TYPE", "chat")
        result = _call_dify_vision(
            query=instruction,
            file_ids=[file_id],
            app_type=app_type,
        )

        logger.info(f"[Tool:image_recognition] 识别完成，结果长度={len(result)}")
        return result

    except FileNotFoundError as e:
        return f"错误: {e}"
    except Exception as e:
        logger.error(f"[Tool:image_recognition] 执行失败: {e}")
        return f"图片识别失败: {e}"


@tool
def document_summary(file_path: str, instruction: str = "请总结这份文档的核心内容") -> str:
    """对文档进行智能总结和要点提炼。支持 PDF、Word、Excel、TXT、Markdown 等格式。

    Args:
        file_path: 文档文件的本地路径（必须是服务器上的绝对路径）
        instruction: 总结指令，例如「总结核心要点」、「提炼关键条款」、「分析报告结论」
    """
    logger.info(f"[Tool:document_summary] file={file_path} | instruction={instruction[:100]}")

    # 文件类型校验：仅接受文档文件
    ext = Path(file_path).suffix.lower()
    if ext not in _DOCUMENT_EXTENSIONS:
        return (
            f"错误: 文件 {Path(file_path).name} 不是支持的文档格式 "
            f"(支持: {', '.join(_DOCUMENT_EXTENSIONS)})。"
            f"如需处理图片，请使用 image_recognition 工具。"
        )

    try:
        # 获取目标 App 的 API Key（上传和调用必须使用同一个 key）
        doc_summary_api_key = _get_dify_api_key("DOC_SUMMARY")

        # Step 1: 使用 Doc Summary App 的 key 上传文档到 Dify
        file_id = _upload_file_to_dify(file_path, api_key=doc_summary_api_key)

        # Step 2: 调用 Dify 文档总结 API
        app_type = os.getenv("DIFY_DOC_SUMMARY_APP_TYPE", "chat")
        result = _call_dify_doc_summary(
            query=instruction,
            file_ids=[file_id],
            app_type=app_type,
        )

        logger.info(f"[Tool:document_summary] 总结完成，结果长度={len(result)}")
        return result

    except FileNotFoundError as e:
        return f"错误: {e}"
    except Exception as e:
        logger.error(f"[Tool:document_summary] 执行失败: {e}")
        return f"文档总结失败: {e}"


@tool
def pdf_add_watermark(
    file_path: str,
    watermark_text: str = "WATERMARK",
    opacity: float = 0.18,
    font_size: int = 48,
    rotation: int = 45,
) -> str:
    """给 PDF 的每一页添加文本水印，并生成新的 PDF 文件。

    Args:
        file_path: PDF 文件的本地路径（必须是服务器上的绝对路径）
        watermark_text: 要添加的水印文字，例如「CONFIDENTIAL」或「仅供内部使用」
        opacity: 水印透明度，0 到 1 之间，默认 0.18
        font_size: 水印字号，默认 48
        rotation: 水印旋转角度，默认 45
    """
    logger.info(
        "[Tool:pdf_add_watermark] file=%s | text=%s | opacity=%s | font_size=%s | rotation=%s",
        file_path,
        watermark_text,
        opacity,
        font_size,
        rotation,
    )

    path = Path(file_path)
    if not path.exists():
        return f"错误: 文件不存在: {file_path}"
    if path.suffix.lower() != ".pdf":
        return f"错误: 文件 {path.name} 不是 PDF 文件，无法添加 PDF 水印。"

    try:
        from pypdf import PdfReader, PdfWriter
        from reportlab.pdfgen import canvas
        from reportlab.lib.colors import Color
    except ImportError as e:
        return (
            "PDF 加水印工具缺少依赖，请先安装 pypdf 和 reportlab："
            "pip install pypdf reportlab。"
            f"具体错误: {e}"
        )

    try:
        safe_opacity = max(0.0, min(float(opacity), 1.0))
        safe_font_size = max(8, min(int(font_size), 240))
        safe_rotation = int(rotation)
        text = str(watermark_text or "WATERMARK")

        reader = PdfReader(str(path))
        writer = PdfWriter()

        for page in reader.pages:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)

            packet = BytesIO()
            overlay = canvas.Canvas(packet, pagesize=(width, height))
            overlay.saveState()
            if hasattr(overlay, "setFillAlpha"):
                overlay.setFillAlpha(safe_opacity)
            overlay.setFillColor(Color(0.45, 0.45, 0.45, alpha=safe_opacity))
            overlay.setFont("Helvetica-Bold", safe_font_size)
            overlay.translate(width / 2, height / 2)
            overlay.rotate(safe_rotation)
            overlay.drawCentredString(0, 0, text)
            overlay.restoreState()
            overlay.save()

            packet.seek(0)
            watermark_page = PdfReader(packet).pages[0]
            page.merge_page(watermark_page)
            writer.add_page(page)

        if reader.metadata:
            writer.add_metadata(dict(reader.metadata))

        output_name = f"{path.stem}_watermarked_{int(time.time())}.pdf"
        output_path = path.with_name(output_name)
        with open(output_path, "wb") as output:
            writer.write(output)

        return (
            f"已成功添加水印「{text}」。\n"
            f"输出文件: {output_path}\n"
            f"下载链接: /uploads/{output_name}"
        )
    except Exception as e:
        logger.error(f"[Tool:pdf_add_watermark] 执行失败: {e}", exc_info=True)
        return f"PDF 加水印失败: {e}"


# 导出工具列表
GENERAL_CHAT_TOOLS = [image_recognition, document_summary, pdf_add_watermark]
