# 消息队列客户端
import json
import time
from typing import Dict, Any
import redis
from kafka import KafkaProducer, KafkaConsumer
from config import Config
from utils.logger import Logger

class MessageQueueClient:
    """消息队列客户端"""
    
    def __init__(self):
        self.type = Config.MESSAGE_QUEUE_TYPE
        self.logger = Logger.get_logger(__name__)
        
        if self.type == 'redis':
            # 初始化 Redis 客户端
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB
            )
        elif self.type == 'kafka':
            # 初始化 Kafka 客户端
            self.producer = KafkaProducer(
                bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        else:
            raise ValueError(f"Unsupported message queue type: {self.type}")
    
    def send_message(self, topic: str, message: Dict[str, Any]) -> bool:
        """发送消息
        Args:
            topic: 主题
            message: 消息内容
        Returns:
            bool: 发送是否成功
        """
        try:
            if self.type == 'redis':
                # 使用 Redis List 作为消息队列
                self.redis_client.lpush(topic, json.dumps(message))
            elif self.type == 'kafka':
                # 使用 Kafka 发送消息
                self.producer.send(topic, message)
                self.producer.flush()
            
            self.logger.info(f"Message sent to {topic}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def receive_message(self, topic: str, timeout: int = 1) -> Dict[str, Any]:
        """接收消息
        Args:
            topic: 主题
            timeout: 超时时间（秒）
        Returns:
            Dict[str, Any]: 消息内容
        """
        try:
            if self.type == 'redis':
                # 从 Redis List 中获取消息
                message = self.redis_client.brpop(topic, timeout=timeout)
                if message:
                    return json.loads(message[1])
            elif self.type == 'kafka':
                # 从 Kafka 中获取消息
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
                    auto_offset_reset='earliest',
                    group_id='taskflow_consumer'
                )
                for message in consumer:
                    return message.value
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None
    
    def get_timestamp(self) -> float:
        """获取当前时间戳"""
        return time.time()
    
    def close(self) -> None:
        """关闭客户端"""
        try:
            if self.type == 'kafka' and hasattr(self, 'producer'):
                self.producer.close()
            elif self.type == 'redis' and hasattr(self, 'redis_client'):
                self.redis_client.close()
            self.logger.info("Message queue client closed")
        except Exception as e:
            self.logger.error(f"Error closing message queue client: {e}")