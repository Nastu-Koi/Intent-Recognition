"""
日志配置模块
支持将工作流处理过程输出到 log.txt
"""

import logging
import os
from pathlib import Path
from typing import Optional


def _parse_bool(value: Optional[str], default: bool) -> bool:
    """解析环境变量中的布尔值。"""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_log_level(value: Optional[str], default: int) -> int:
    """解析日志级别字符串。"""
    if not value:
        return default
    return getattr(logging, value.strip().upper(), default)


def setup_logging(
    log_file: Optional[str] = None,
    enable_console: Optional[bool] = None,
    enable_file: Optional[bool] = None
) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        log_file: 日志文件路径（显式传入时优先于环境变量）
        enable_console: 是否同时输出到控制台（显式传入时优先于环境变量）
        enable_file: 是否输出到日志文件（显式传入时优先于环境变量）
    
    Returns:
        配置好的 Logger 实例
    """
    resolved_log_file = log_file if log_file is not None else os.getenv("LOG_FILE", "log.txt")
    resolved_enable_console = (
        enable_console
        if enable_console is not None
        else _parse_bool(os.getenv("LOG_ENABLE_CONSOLE"), True)
    )
    resolved_enable_file = (
        enable_file
        if enable_file is not None
        else _parse_bool(os.getenv("LOG_ENABLE_FILE"), True)
    )
    # 默认保持安静，仅在显式配置 LOG_LEVEL 时输出更详细日志。
    root_level = _parse_log_level(os.getenv("LOG_LEVEL"), logging.WARNING)
    file_level = _parse_log_level(os.getenv("LOG_FILE_LEVEL"), root_level)
    console_level = _parse_log_level(os.getenv("LOG_CONSOLE_LEVEL"), root_level)

    # 配置根 logger，确保 api_service/agent_factory 等命名 logger 都能继承处理器
    logger = logging.getLogger()
    logger.setLevel(root_level)

    # 清空现有的处理器
    logger.handlers.clear()

    # 日志格式
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器（可选）
    if resolved_enable_file and resolved_log_file:
        log_path = Path(resolved_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(resolved_log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 控制台处理器（可选）
    if resolved_enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logging.getLogger("intent_recognition")


def get_logger(name: str = "intent_recognition") -> logging.Logger:
    """
    获取指定名称的 Logger（需要先调用 setup_logging 初始化）
    
    Args:
        name: Logger 名称
    
    Returns:
        Logger 实例
    """
    return logging.getLogger(name)


def clear_log(log_file: str = "log.txt"):
    """
    清空日志文件
    
    Args:
        log_file: 日志文件路径
    """
    Path(log_file).write_text("")
