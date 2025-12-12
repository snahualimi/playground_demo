import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os

def setup_logger(logger_name: str='playground', log_dir: str = "/home/ubuntu/playground/logs") -> tuple[logging.Logger, str]:
    """
    配置并返回一个日志记录器（Logger）和日志文件路径。
    """

    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{logger_name}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # 创建日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # 设置最低日志级别
    logger.propagate = False
    # 避免重复添加处理器（防止多次调用时重复日志）
    if not logger.handlers:
        # 1. 控制台处理器（输出到终端）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 控制台只显示 INFO 及以上级别
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 2. 文件处理器（输出到文件，自动轮转）
        file_handler = RotatingFileHandler(
            log_filepath,
            maxBytes=10 * 1024 * 1024,  # 10MB 后轮转
            backupCount=5,  # 保留 5 个备份
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有 DEBUG 及以上级别
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger, log_filepath