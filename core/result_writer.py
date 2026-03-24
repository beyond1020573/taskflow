from abc import ABC, abstractmethod
from typing import Dict, Any
from utils.logger import Logger


class ResultWriter(ABC):
    """结果写入器抽象基类"""
    
    @abstractmethod
    def write(self, task_id: str, result: Dict[str, Any]) -> bool:
        """写入任务结果
        
        Args:
            task_id: 任务 ID
            result: 任务结果
            
        Returns:
            bool: 写入是否成功
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭写入器，释放资源"""
        pass


class PrintResultWriter(ResultWriter):
    """控制台打印结果写入器（用于测试和调试）"""
    
    def __init__(self):
        self.logger = Logger.get_logger(__name__)
    
    def write(self, task_id: str, result: Dict[str, Any]) -> bool:
        """打印结果到控制台
        
        Args:
            task_id: 任务 ID
            result: 任务结果
            
        Returns:
            bool: 始终返回 True
        """
        import json
        result_str = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        print(f"[ResultWriter] task_id={task_id}, result={result_str}")
        self.logger.info(f"Result written for task {task_id}")
        return True
    
    def close(self) -> None:
        """关闭写入器"""
        self.logger.info("PrintResultWriter closed")


class MessageQueueResultWriter(ResultWriter):
    """消息队列结果写入器"""
    
    def __init__(self, queue_name: str = "task_results"):
        from distributed.message_queue import MessageQueueClient
        self.message_queue = MessageQueueClient()
        self.queue_name = queue_name
        self.logger = Logger.get_logger(__name__)
    
    def write(self, task_id: str, result: Dict[str, Any]) -> bool:
        """写入结果到消息队列
        
        Args:
            task_id: 任务 ID
            result: 任务结果
            
        Returns:
            bool: 写入是否成功
        """
        try:
            message = {
                'task_id': task_id,
                'result': result,
                'timestamp': self.message_queue.get_timestamp()
            }
            
            success = self.message_queue.send_message(self.queue_name, message)
            if success:
                self.logger.info(f"Result written for task {task_id}")
            else:
                self.logger.error(f"Failed to write result for task {task_id}")
            
            return success
        except Exception as e:
            self.logger.error(f"Error writing result: {e}")
            return False
    
    def close(self) -> None:
        """关闭写入器"""
        self.message_queue.close()
        self.logger.info("MessageQueueResultWriter closed")
