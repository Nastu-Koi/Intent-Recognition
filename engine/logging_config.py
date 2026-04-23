"""
日志配置模块
支持将工作流处理过程输出到 log.txt
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = "log.txt",
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        log_file: 日志文件路径（相对于项目根目录）
        enable_console: 是否同时输出到控制台
        enable_file: 是否输出到日志文件
    
    Returns:
        配置好的 Logger 实例
    """

    # 配置根 logger，确保 api_service/agent_factory 等命名 logger 都能继承处理器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 清空现有的处理器
    logger.handlers.clear()

    # 日志格式
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器（可选）
    if enable_file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 控制台处理器（可选）
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
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
