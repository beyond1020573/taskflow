# 会话管理
from typing import Dict, Any, Optional
from utils.logger import Logger

class SessionManager:
    """会话管理"""
    
    def __init__(self):
        # 客户端 ID 到 WebSocket 连接的映射
        self.client_websocket_map: Dict[str, Any] = {}
        # 任务 ID 到客户端 ID 的映射
        self.task_client_map: Dict[str, str] = {}
        # WebSocket 连接到客户端 ID 的映射
        self.websocket_client_map: Dict[Any, str] = {}
        self.logger = Logger.get_logger(__name__)
    
    def register_session(self, client_id: str, websocket: Any) -> None:
        """注册会话
        Args:
            client_id: 客户端 ID
            websocket: WebSocket 连接
        """
        self.client_websocket_map[client_id] = websocket
        self.websocket_client_map[websocket] = client_id
        self.logger.info(f"Session registered for client {client_id}")
    
    def bind_task_to_client(self, task_id: str, client_id: str) -> None:
        """绑定任务到客户端
        Args:
            task_id: 任务 ID
            client_id: 客户端 ID
        """
        self.task_client_map[task_id] = client_id
        self.logger.info(f"Task {task_id} bound to client {client_id}")
    
    def get_client_id_by_task_id(self, task_id: str) -> Optional[str]:
        """根据任务 ID 获取客户端 ID
        Args:
            task_id: 任务 ID
        Returns:
            Optional[str]: 客户端 ID
        """
        return self.task_client_map.get(task_id)
    
    def get_websocket_by_client_id(self, client_id: str) -> Optional[Any]:
        """根据客户端 ID 获取 WebSocket 连接
        Args:
            client_id: 客户端 ID
        Returns:
            Optional[Any]: WebSocket 连接
        """
        return self.client_websocket_map.get(client_id)
    
    def remove_session_by_client_id(self, client_id: str) -> None:
        """根据客户端 ID 移除会话
        Args:
            client_id: 客户端 ID
        """
        if client_id in self.client_websocket_map:
            websocket = self.client_websocket_map.pop(client_id)
            if websocket in self.websocket_client_map:
                self.websocket_client_map.pop(websocket)
            self.logger.info(f"Session removed for client {client_id}")
    
    def remove_session_by_websocket(self, websocket: Any) -> None:
        """根据 WebSocket 连接移除会话
        Args:
            websocket: WebSocket 连接
        """
        if websocket in self.websocket_client_map:
            client_id = self.websocket_client_map.pop(websocket)
            if client_id in self.client_websocket_map:
                self.client_websocket_map.pop(client_id)
            self.logger.info(f"Session removed for client {client_id}")
    
    def remove_task_binding(self, task_id: str) -> None:
        """移除任务绑定
        Args:
            task_id: 任务 ID
        """
        if task_id in self.task_client_map:
            self.task_client_map.pop(task_id)
            self.logger.info(f"Task binding removed for {task_id}")
    
    def close(self) -> None:
        """关闭会话管理器"""
        # 清理所有映射
        self.client_websocket_map.clear()
        self.task_client_map.clear()
        self.websocket_client_map.clear()
        self.logger.info("Session manager closed")