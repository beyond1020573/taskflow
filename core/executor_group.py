# 执行器组
from typing import List, Optional
from core.executor import Executor, ExecutorStatus
from core.task import TaskMode
from utils.logger import Logger

class ExecutorGroup:
    """执行器组"""
    
    def __init__(self, plugin_id: str, max_executors: int):
        """
        Args:
            plugin_id: 插件 ID
            max_executors: 最大执行器数量
        """
        self.plugin_id = plugin_id
        self.max_executors = max_executors
        self.executors: List[Executor] = []
        self.logger = Logger.get_logger(f"executor_group_{plugin_id}")
    
    def add_executor(self, executor: Executor) -> None:
        """添加执行器"""
        if len(self.executors) < self.max_executors:
            self.executors.append(executor)
            self.logger.info(f"Added executor for plugin {self.plugin_id}")
        else:
            self.logger.warning(f"Executor group for {self.plugin_id} is full")
    
    def get_ready_executor(self, mode: TaskMode) -> Optional[Executor]:
        """获取就绪的执行器
        Args:
            mode: 任务模式
        Returns:
            Optional[Executor]: 就绪的执行器
        """
        # 清理失效执行器
        self._cleanup_dead_executors()
        
        # 查找就绪执行器
        for executor in self.executors:
            if executor.status == ExecutorStatus.READY:
                return executor
        
        return None
    
    def _cleanup_dead_executors(self) -> None:
        """清理失效执行器"""
        alive_executors = []
        for executor in self.executors:
            if executor.is_alive():
                alive_executors.append(executor)
            else:
                self.logger.info(f"Cleaning up dead executor for plugin {self.plugin_id}")
                executor.destroy()
        
        self.executors = alive_executors
    
    def get_executor_count(self) -> int:
        """获取执行器数量"""
        self._cleanup_dead_executors()
        return len(self.executors)
    
    def destroy(self) -> None:
        """销毁执行器组"""
        for executor in self.executors:
            executor.destroy()
        self.executors = []
        self.logger.info(f"Destroyed executor group for plugin {self.plugin_id}")