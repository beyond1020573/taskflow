# Plugin 抽象接口
from abc import ABC, abstractmethod
from typing import Dict, Any

class Plugin(ABC):
    """插件抽象接口"""
    
    @abstractmethod
    def pre_init(self, config: Dict[str, Any]) -> bool:
        """预初始化
        Args:
            config: 插件配置
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def execute(self, task_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行单次任务
        Args:
            task_id: 任务 ID
            params: 任务参数
        Returns:
            Dict[str, Any]: 执行结果
        """
        pass
    
    @abstractmethod
    def start_long_task(self, task_id: str, params: Dict[str, Any]) -> bool:
        """启动长时任务
        Args:
            task_id: 任务 ID
            params: 任务参数
        Returns:
            bool: 启动是否成功
        """
        pass
    
    @abstractmethod
    def stop_long_task(self, task_id: str) -> bool:
        """停止长时任务
        Args:
            task_id: 任务 ID
        Returns:
            bool: 停止是否成功
        """
        pass
    
    @abstractmethod
    def destroy(self) -> None:
        """资源销毁"""
        pass