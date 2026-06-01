"""
logger.py - 日志记录模块
"""
import logging
import sys
from datetime import datetime

def setup_logger(log_file='modeling_log.txt', level=logging.INFO):
    """
    配置日志：同时输出到控制台和文件
    """
    # 清除已有的 handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_model_results(logger, section, metrics):
    """
    通用记录模型结果
    section: 字符串，如 '问题一'
    metrics: dict, 包含指标名和值
    """
    logger.info(f"========== {section} ==========")
    for name, value in metrics.items():
        logger.info(f"{name}: {value}")
    logger.info("")