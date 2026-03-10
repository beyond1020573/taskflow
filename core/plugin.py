from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

# 全框架通用的标准化响应类
@dataclass
class TaskResponse:
    """标准化任务响应结构体（全框架复用）"""
    success: bool          # 核心标识：操作是否成功
    code: str              # 错误码：成功为"success"，失败为分层错误码（如param/missing、resource/limit）
    message: str           # 人类可读的提示/错误信息
    data: Optional[Any] = None  # 可选附加数据（成功返回任务信息，失败返回详情）

class Plugin(ABC):
    """插件抽象接口（用户仅需实现此类）"""
    
    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """插件类型唯一标识"""
        pass
    
    @abstractmethod
    def pre_init(self, config: Dict[str, Any]) -> TaskResponse:
        """插件预初始化（执行器启动时调用一次）
        Args:
            config: 插件配置字典
        Returns:
            TaskResponse: 初始化结果
        """
        pass
    
    @abstractmethod
    def execute(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        """执行单次任务（非阻塞）
        Args:
            task_id: 唯一任务ID
            params: 任务参数字典
        Returns:
            TaskResponse: 执行结果（data字段返回任务执行结果）
        """
        pass
    
    @abstractmethod
    def start_long_task(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        """启动长时任务（阻塞运行，独占执行器）
        Args:
            task_id: 唯一任务ID
            params: 任务参数字典
        Returns:
            TaskResponse: 启动结果（成功时data可返回任务PID/预估时长等）
        """
        pass
    
    @abstractmethod
    def stop_long_task(self, task_id: str) -> TaskResponse:
        """停止长时任务
        Args:
            task_id: 唯一任务ID
        Returns:
            TaskResponse: 停止结果
        """
        pass
    
    @abstractmethod
    def destroy(self) -> None:
        """插件资源销毁（执行器退出时调用）"""
        pass