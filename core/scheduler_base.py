# 调度器基类
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.task import Task
from utils.logger import Logger
from utils.exception_handler import ExceptionHandler

class BaseScheduler(ABC):
    """调度器基类"""
    
    def __init__(self):
        self.logger = Logger.get_logger(self.__class__.__name__)
    
    def validate_task(self, task: Task) -> bool:
        """验证任务
        Args:
            task: 任务对象
        Returns:
            bool: 验证是否通过
        """
        if not task.task_id:
            self.logger.error("Task ID is required")
            return False
        
        if not task.plugin_id:
            self.logger.error("Plugin ID is required")
            return False
        
        return True
    
    def parse_task_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """解析任务参数
        Args:
            params: 原始参数
        Returns:
            Dict[str, Any]: 解析后的参数
        """
        # 这里可以添加参数解析逻辑
        return params
    
    @abstractmethod
    def submit_task(self, task: Task) -> Dict[str, Any]:
        """提交任务
        Args:
            task: 任务对象
        Returns:
            Dict[str, Any]: 提交结果
        """
        pass
    
    def handle_task_exception(self, e: Exception) -> Dict[str, Any]:
        """处理任务异常
        Args:
            e: 异常对象
        Returns:
            Dict[str, Any]: 异常处理结果
        """
        return ExceptionHandler.handle_exception(e)