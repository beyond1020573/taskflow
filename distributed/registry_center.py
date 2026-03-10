# 注册中心客户端
import etcd3
import json
import time
from typing import List, Dict, Any
from config import Config
from utils.logger import Logger

class RegistryCenter:
    """注册中心客户端"""
    
    def __init__(self):
        self.client = etcd3.client(host='localhost', port=2379)
        self.logger = Logger.get_logger(__name__)
    
    def register_local_scheduler(self, scheduler_id: str, plugins: List[str]) -> bool:
        """注册本地调度器
        Args:
            scheduler_id: 调度器 ID
            plugins: 支持的插件列表
        Returns:
            bool: 注册是否成功
        """
        try:
            # 构建注册信息
            register_info = {
                'scheduler_id': scheduler_id,
                'plugins': plugins,
                'timestamp': time.time()
            }
            
            # 写入 ETCD
            key = f'/taskflow/schedulers/{scheduler_id}'
            self.client.put(key, json.dumps(register_info))
            
            # 设置过期时间
            self.client.put(key, json.dumps(register_info), lease=self.client.lease(Config.REGISTRY_HEARTBEAT_INTERVAL * 2))
            
            self.logger.info(f"Registered local scheduler {scheduler_id} with plugins {plugins}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register local scheduler: {e}")
            return False
    
    def heartbeat(self, scheduler_id: str) -> bool:
        """发送心跳
        Args:
            scheduler_id: 调度器 ID
        Returns:
            bool: 心跳是否成功
        """
        try:
            key = f'/taskflow/schedulers/{scheduler_id}'
            # 获取当前注册信息
            value, _ = self.client.get(key)
            if value:
                register_info = json.loads(value)
                register_info['timestamp'] = time.time()
                # 更新注册信息并续期
                self.client.put(key, json.dumps(register_info), lease=self.client.lease(Config.REGISTRY_HEARTBEAT_INTERVAL * 2))
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat: {e}")
            return False
    
    def get_local_schedulers(self, plugin_id: str) -> List[str]:
        """获取支持指定插件的本地调度器
        Args:
            plugin_id: 插件 ID
        Returns:
            List[str]: 调度器 ID 列表
        """
        try:
            schedulers = []
            # 扫描所有调度器
            for key, value in self.client.get_prefix('/taskflow/schedulers/'):
                if value:
                    register_info = json.loads(value)
                    # 检查是否支持指定插件
                    if plugin_id in register_info.get('plugins', []):
                        # 检查是否过期
                        if time.time() - register_info.get('timestamp', 0) < Config.REGISTRY_HEARTBEAT_INTERVAL * 2:
                            schedulers.append(register_info['scheduler_id'])
            
            self.logger.info(f"Found {len(schedulers)} local schedulers for plugin {plugin_id}")
            return schedulers
        except Exception as e:
            self.logger.error(f"Failed to get local schedulers: {e}")
            return []
    
    def unregister_local_scheduler(self, scheduler_id: str) -> bool:
        """注销本地调度器
        Args:
            scheduler_id: 调度器 ID
        Returns:
            bool: 注销是否成功
        """
        try:
            key = f'/taskflow/schedulers/{scheduler_id}'
            self.client.delete(key)
            self.logger.info(f"Unregistered local scheduler {scheduler_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to unregister local scheduler: {e}")
            return False
    
    def close(self) -> None:
        """关闭客户端"""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            self.logger.info("Registry center client closed")
        except Exception as e:
            self.logger.error(f"Error closing registry center client: {e}")