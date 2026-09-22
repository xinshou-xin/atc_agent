"""
日志工具
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from config.settings import LOG


def setup_logger(name: str = None) -> logging.Logger:
    """
    设置并返回 logger

    Args:
        name: logger 名称

    Returns:
        logger 实例
    """
    logger = logging.getLogger(name or "drug_agent")

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG["level"].upper(), logging.INFO))

    # 格式化
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    log_file = LOG["file"]
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOG["max_bytes"],
            backupCount=LOG["backup_count"],
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"日志文件初始化失败: {e}")

    return logger
