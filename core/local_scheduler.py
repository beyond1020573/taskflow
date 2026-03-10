# 本地调度器
from typing import Dict, Any
from core.scheduler_base import BaseScheduler
from core.task import Task, TaskMode
from core.executor_group import ExecutorGroup
from core.executor import Executor, ExecutorStatus
from core.plugin import Plugin, TaskResponse
from utils.logger import Logger

class LocalScheduler(BaseScheduler):
    """本地调度器"""
    
    def __init__(self):
        super().__init__()
        self.executor_groups: Dict[str, ExecutorGroup] = {}
    
    def register_plugin(self, plugin_id: str, plugin_class: type, config: Dict[str, Any], max_executors: int = 1):
        """注册插件
        Args:
            plugin_id: 插件 ID
            plugin_class: 插件类
            config: 插件配置
            max_executors: 最大执行器数量
        """
        if plugin_id in self.executor_groups:
            self.logger.warning(f"Plugin {plugin_id} already registered")
            return
        
        # 创建执行器组
        executor_group = ExecutorGroup(plugin_id, max_executors)
        
        # 预启动执行器
        for i in range(max_executors):
            try:
                plugin = plugin_class()
                pre_init_result = plugin.pre_init(config)
                if pre_init_result.success:
                    executor = Executor(plugin)
                    executor_group.add_executor(executor)
                else:
                    self.logger.error(f"Failed to pre_init plugin {plugin_id}: {pre_init_result.message}")
            except Exception as e:
                self.logger.error(f"Failed to create executor for plugin {plugin_id}: {e}")
        
        self.executor_groups[plugin_id] = executor_group
        self.logger.info(f"Registered plugin {plugin_id} with {max_executors} executors")
    
    def submit_task(self, task: Task) -> TaskResponse:
        """提交任务
        Args:
            task: 任务对象
        Returns:
            TaskResponse: 提交结果
        """
        try:
            # 验证任务
            if not self.validate_task(task):
                return TaskResponse(
                    success=False,
                    code="task/invalid",
                    message="无效的任务"
                )
            
            # 解析参数
            params = self.parse_task_params(task.params)
            
            # 查找执行器组
            if task.plugin_id not in self.executor_groups:
                return TaskResponse(
                    success=False,
                    code="plugin/not_registered",
                    message=f"插件 {task.plugin_id} 未注册"
                )
            
            executor_group = self.executor_groups[task.plugin_id]
            
            # 获取就绪执行器
            executor = executor_group.get_ready_executor(task.mode)
            if not executor:
                return TaskResponse(
                    success=False,
                    code="executor/unavailable",
                    message="无可用执行器"
                )
            
            # 执行任务
            if task.mode == TaskMode.SINGLE:
                return executor.execute(task.task_id, params)
            elif task.mode == TaskMode.LONG:
                return executor.start_long_task(task.task_id, params)
            else:
                return TaskResponse(
                    success=False,
                    code="task/invalid_mode",
                    message="无效的任务模式"
                )
        except Exception as e:
            return TaskResponse(
                success=False,
                code="system/error",
                message=f"系统错误：{str(e)}"
            )
    
    def stop_long_task(self, task_id: str, plugin_id: str) -> TaskResponse:
        """停止长时任务
        Args:
            task_id: 任务 ID
            plugin_id: 插件 ID
        Returns:
            TaskResponse: 停止结果
        """
        try:
            if plugin_id not in self.executor_groups:
                return TaskResponse(
                    success=False,
                    code="plugin/not_registered",
                    message=f"插件 {plugin_id} 未注册"
                )
            
            executor_group = self.executor_groups[plugin_id]
            
            # 清理失效执行器
            executor_group._cleanup_dead_executors()
            
            # 查找执行长时任务的执行器
            for executor in executor_group.executors:
                if executor.status == ExecutorStatus.BUSY_LONG:
                    return executor.stop_long_task(task_id)
            
            return TaskResponse(
                success=False,
                code="task/not_found",
                message="未找到运行中的长时任务"
            )
        except Exception as e:
            return TaskResponse(
                success=False,
                code="system/error",
                message=f"系统错误：{str(e)}"
            )
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """注销插件
        Args:
            plugin_id: 插件 ID
        Returns:
            bool: 注销是否成功
        """
        if plugin_id in self.executor_groups:
            executor_group = self.executor_groups.pop(plugin_id)
            executor_group.destroy()
            self.logger.info(f"Unregistered plugin {plugin_id}")
            return True
        return False
    
    def shutdown(self) -> None:
        """关闭调度器"""
        for plugin_id, executor_group in self.executor_groups.items():
            executor_group.destroy()
        self.executor_groups.clear()
        self.logger.info("Local scheduler shutdown")