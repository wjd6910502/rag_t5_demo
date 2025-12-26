"""
日志配置模块 - 统一管理日志配置
支持DEBUG和INFO级别，可配置控制台和文件日志级别
"""
import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    log_file: Optional[str] = None,
    log_dir: str = 'logs',
    console_level: str = 'INFO',
    file_level: str = 'DEBUG',
    log_format: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    配置并返回日志记录器
    
    Args:
        log_file: 日志文件路径（如果为None，则自动生成）
        log_dir: 日志文件目录（默认：logs）
        console_level: 控制台日志级别（默认：INFO）
        file_level: 文件日志级别（默认：DEBUG）
        log_format: 日志格式（如果为None，使用默认格式）
        date_format: 日期格式（如果为None，使用默认格式）
    
    Returns:
        配置好的日志记录器
    """
    # 默认日志格式
    if log_format is None:
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
    if date_format is None:
        date_format = '%Y-%m-%d %H:%M:%S'
    
    # 解析日志级别
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    console_log_level = log_level_map.get(console_level.upper(), logging.INFO)
    file_log_level = log_level_map.get(file_level.upper(), logging.DEBUG)
    
    # 创建日志文件路径
    if log_file is None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    else:
        # 确保目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 设置为DEBUG以支持所有级别
    
    # 清除已有的处理器（避免重复）
    root_logger.handlers = []
    
    # 创建文件处理器（记录所有级别）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_log_level)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台处理器（根据参数设置级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 获取模块日志记录器
    logger = logging.getLogger(__name__)
    
    # 记录日志配置信息
    logger.info(f"📄 日志文件: {log_file}")
    logger.debug(f"日志级别: 控制台={console_level}, 文件={file_level}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称（如果为None，返回根记录器）
    
    Returns:
        日志记录器
    """
    if name:
        return logging.getLogger(name)
    return logging.getLogger()

