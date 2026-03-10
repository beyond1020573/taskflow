# Task/TaskMode 定义
from enum import Enum
from typing import Dict, Any, Optional

class TaskMode(Enum):
    """任务模式"""
    SINGLE = "SINGLE"  # 单次任务
    LONG = "LONG"      # 长时任务

class Task:
    """任务载体"""
    
    def __init__(self,
                 task_id: str,
                 plugin_id: str,
                 mode: TaskMode,
                 params: Dict[str, Any],
                 timeout: Optional[int] = None):
        """
        Args:
            task_id: 任务 ID
            plugin_id: 插件 ID
            mode: 任务模式
            params: 任务参数
            timeout: 超时时间（秒）
        """
        self.task_id = task_id
        self.plugin_id = plugin_id
        self.mode = mode
        self.params = params
        self.timeout = timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'plugin_id': self.plugin_id,
            'mode': self.mode.value,
            'params': self.params,
            'timeout': self.timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建 Task"""
        return cls(
            task_id=data['task_id'],
            plugin_id=data['plugin_id'],
            mode=TaskMode(data['mode']),
            params=data['params'],
            timeout=data.get('timeout')
        )