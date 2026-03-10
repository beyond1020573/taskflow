# Executor/ExecutorStatus 定义
from enum import Enum
import multiprocessing
import time
from typing import Dict, Any, Optional
from core.plugin import Plugin, TaskResponse
from utils.logger import Logger
from config import Config

class ExecutorStatus(Enum):
    """执行器状态"""
    READY = "READY"              # 就绪
    BUSY_SINGLE = "BUSY_SINGLE"  # 执行单次任务中
    BUSY_LONG = "BUSY_LONG"      # 执行长时任务中
    ERROR = "ERROR"              # 错误

class Executor:
    """执行器"""
    
    def __init__(self, plugin: Plugin):
        """
        Args:
            plugin: 插件实例
        """
        self.plugin = plugin
        self.status = ExecutorStatus.READY
        self.last_heartbeat = time.time()
        self.process: Optional[multiprocessing.Process] = None
        self.logger = Logger.get_logger(f"executor_{plugin.plugin_id}")
    
    def heartbeat(self) -> None:
        """更新心跳时间"""
        self.last_heartbeat = time.time()
    
    def is_alive(self) -> bool:
        """检查执行器是否存活"""
        return time.time() - self.last_heartbeat < Config.EXECUTOR_HEARTBEAT_TIMEOUT
    
    def execute(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        """执行单次任务"""
        if self.status != ExecutorStatus.READY:
            return TaskResponse(
                success=False,
                code="executor/busy",
                message="Executor is not ready"
            )
        
        try:
            self.status = ExecutorStatus.BUSY_SINGLE
            self.heartbeat()
            result = self.plugin.execute(task_id, params)
            self.status = ExecutorStatus.READY
            self.heartbeat()
            return result
        except Exception as e:
            self.status = ExecutorStatus.ERROR
            self.logger.error(f"Execute task failed: {e}")
            return TaskResponse(
                success=False,
                code="executor/error",
                message=f"Execute task failed: {str(e)}"
            )
    
    def start_long_task(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        """启动长时任务"""
        if self.status != ExecutorStatus.READY:
            return TaskResponse(
                success=False,
                code="executor/busy",
                message="Executor is not ready"
            )
        
        try:
            self.status = ExecutorStatus.BUSY_LONG
            self.heartbeat()
            result = self.plugin.start_long_task(task_id, params)
            if not result.success:
                self.status = ExecutorStatus.READY
            self.heartbeat()
            return result
        except Exception as e:
            self.status = ExecutorStatus.ERROR
            self.logger.error(f"Start long task failed: {e}")
            return TaskResponse(
                success=False,
                code="executor/error",
                message=f"Start long task failed: {str(e)}"
            )
    
    def stop_long_task(self, task_id: str) -> TaskResponse:
        """停止长时任务"""
        if self.status != ExecutorStatus.BUSY_LONG:
            return TaskResponse(
                success=False,
                code="executor/not-busy",
                message="Executor is not running a long task"
            )
        
        try:
            result = self.plugin.stop_long_task(task_id)
            if result.success:
                self.status = ExecutorStatus.READY
            self.heartbeat()
            return result
        except Exception as e:
            self.logger.error(f"Stop long task failed: {e}")
            return TaskResponse(
                success=False,
                code="executor/error",
                message=f"Stop long task failed: {str(e)}"
            )
    
    def destroy(self) -> None:
        """销毁执行器"""
        try:
            self.plugin.destroy()
            if self.process and self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        except Exception as e:
            self.logger.error(f"Destroy executor failed: {e}")