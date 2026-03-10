# 网关调度器
from typing import Dict, Any
from core.scheduler_base import BaseScheduler
from core.task import Task
from distributed.registry_center import RegistryCenter
from utils.logger import Logger

class GatewayScheduler(BaseScheduler):
    """网关调度器"""
    
    def __init__(self):
        super().__init__()
        self.registry_center = RegistryCenter()
    
    def submit_task(self, task: Task) -> Dict[str, Any]:
        """提交任务
        Args:
            task: 任务对象
        Returns:
            Dict[str, Any]: 提交结果
        """
        try:
            # 验证任务
            if not self.validate_task(task):
                return {'success': False, 'error_message': 'Invalid task'}
            
            # 解析参数
            params = self.parse_task_params(task.params)
            
            # 查询注册中心，获取可用的 LocalScheduler
            local_schedulers = self.registry_center.get_local_schedulers(task.plugin_id)
            if not local_schedulers:
                return {'success': False, 'error_message': f'No local scheduler available for plugin {task.plugin_id}'}
            
            # 简单负载均衡：选择第一个可用的 LocalScheduler
            local_scheduler = local_schedulers[0]
            self.logger.info(f"Routing task {task.task_id} to local scheduler {local_scheduler}")
            
            # 这里应该通过网络请求将任务发送到 LocalScheduler
            # 由于是示例，这里返回模拟结果
            return {'success': True, 'message': f'Task {task.task_id} routed to {local_scheduler}'}
        except Exception as e:
            return self.handle_task_exception(e)
    
    def shutdown(self) -> None:
        """关闭调度器"""
        self.registry_center.close()
        self.logger.info("Gateway scheduler shutdown")