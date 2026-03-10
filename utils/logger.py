# 统一日志类
import logging
from config import Config

class Logger:
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取日志记录器"""
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(Config.LOG_LEVEL)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(Config.LOG_LEVEL)
            
            # 格式化器
            formatter = logging.Formatter(Config.LOG_FORMAT)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            logger.addHandler(console_handler)
            cls._loggers[name] = logger
        
        return cls._loggers[name]