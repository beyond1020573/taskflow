# 结果写入器
from typing import Dict, Any
from distributed.message_queue import MessageQueueClient
from utils.logger import Logger

class ResultWriter:
    """结果写入器"""
    
    def __init__(self):
        self.message_queue = MessageQueueClient()
        self.logger = Logger.get_logger(__name__)
    
    def write_result(self, task_id: str, result: Dict[str, Any]) -> bool:
        """写入任务结果
        Args:
            task_id: 任务 ID
            result: 任务结果
        Returns:
            bool: 写入是否成功
        """
        try:
            # 构建消息
            message = {
                'task_id': task_id,
                'result': result,
                'timestamp': self.message_queue.get_timestamp()
            }
            
            # 发送到消息队列
            success = self.message_queue.send_message('task_results', message)
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
        self.logger.info("Result writer closed")