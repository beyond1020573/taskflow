# 测试框架基本功能
import sys
import os

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from core.local_scheduler import LocalScheduler
from core.task import Task, TaskMode
from core.plugin import Plugin, TaskResponse
from typing import Dict, Any

class TestPlugin(Plugin):
    """测试插件"""
    
    def pre_init(self, config: Dict[str, Any]) -> TaskResponse:
        return TaskResponse(
            success=True,
            code="success",
            message="Plugin initialized successfully"
        )
    
    def execute(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        return TaskResponse(
            success=True,
            code="success",
            message="Task executed successfully",
            data={'result': f'Hello, {params.get("name", "World")}!', 'task_id': task_id}
        )
    
    def start_long_task(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        return TaskResponse(
            success=True,
            code="success",
            message="Long task started successfully",
            data={'task_id': task_id, 'pid': os.getpid()}
        )
    
    def stop_long_task(self, task_id: str) -> TaskResponse:
        return TaskResponse(
            success=True,
            code="success",
            message="Long task stopped successfully"
        )
    
    def destroy(self) -> None:
        pass

class TestTaskFlow:
    """测试 TaskFlow 框架"""
    
    def setup_method(self):
        """设置测试环境"""
        self.scheduler = LocalScheduler()
        # 注册测试插件
        self.scheduler.register_plugin('test_plugin', TestPlugin, {}, max_executors=2)
    
    def teardown_method(self):
        """清理测试环境"""
        self.scheduler.shutdown()
    
    def test_submit_single_task(self):
        """测试提交单次任务"""
        task = Task(
            task_id='test_task_1',
            plugin_id='test_plugin',
            mode=TaskMode.SINGLE,
            params={'name': 'TaskFlow'}
        )
        result = self.scheduler.submit_task(task)
        assert result.success is True
        assert result.code == 'success'
        assert 'Hello, TaskFlow!' in result.data['result']
    
    def test_submit_long_task(self):
        """测试提交长时任务"""
        task = Task(
            task_id='test_task_2',
            plugin_id='test_plugin',
            mode=TaskMode.LONG,
            params={'name': 'LongTask'}
        )
        result = self.scheduler.submit_task(task)
        assert result.success is True
        assert result.code == 'success'
    
    def test_stop_long_task(self):
        """测试停止长时任务"""
        # 先启动长时任务
        task = Task(
            task_id='test_task_3',
            plugin_id='test_plugin',
            mode=TaskMode.LONG,
            params={'name': 'LongTask'}
        )
        start_result = self.scheduler.submit_task(task)
        assert start_result.success is True
        assert start_result.code == 'success'
        
        # 停止长时任务
        stop_result = self.scheduler.stop_long_task('test_task_3', 'test_plugin')
        assert stop_result.success is True
        assert stop_result.code == 'success'
    
    def test_invalid_plugin(self):
        """测试无效插件"""
        task = Task(
            task_id='test_task_4',
            plugin_id='invalid_plugin',
            mode=TaskMode.SINGLE,
            params={'name': 'Test'}
        )
        result = self.scheduler.submit_task(task)
        assert result.success is False
        assert result.code == 'plugin/not_registered'
        assert '未注册' in result.message