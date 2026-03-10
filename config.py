# 全局配置
import os
from typing import Dict, Any

class Config:
    # 执行器配置
    EXECUTOR_HEARTBEAT_INTERVAL = 5  # 心跳间隔（秒）
    EXECUTOR_HEARTBEAT_TIMEOUT = 15  # 心跳超时（秒）
    
    # 任务配置
    TASK_DEFAULT_TIMEOUT = 300  # 默认任务超时时间（秒）
    
    # 注册中心配置
    REGISTRY_ETCD_HOSTS = ['localhost:2379']  # ETCD 服务地址
    REGISTRY_HEARTBEAT_INTERVAL = 10  # 注册中心心跳间隔（秒）
    
    # 消息队列配置
    MESSAGE_QUEUE_TYPE = 'redis'  # redis 或 kafka
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
    
    # 推送服务配置
    PUSH_SERVICE_HOST = '0.0.0.0'
    PUSH_SERVICE_PORT = 8765
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典加载配置"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
        return cls
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """从文件加载配置"""
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)