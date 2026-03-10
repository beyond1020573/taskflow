# TaskFlow 框架

轻量级任务调度框架，专为智能应用插件化开发设计，核心目标：极简、稳定、可维护，支持单机 / 分布式无缝切换，程序员仅需关注业务插件开发，无需关心调度逻辑。

## 核心架构分层

1. **核心内核层**（SDK 类库，用户开发插件时引用）
   - Plugin（插件）：抽象接口，用户实现业务逻辑
   - Task（任务）：统一任务载体
   - Executor（执行器）：子进程管理类
   - ExecutorGroup（执行器组）：按插件分组管理执行器
   - LocalScheduler（本地调度器）：管理本机所有执行器
   - BaseScheduler（调度器基类）：封装公共逻辑
   - ResultWriter（结果写入器）：执行器输出结果到消息队列

2. **分布式扩展层**（独立部署组件，不侵入内核）
   - GatewayScheduler（网关调度器）：接收客户端请求、路由任务
   - RegistryCenter（注册中心）：基于 ETCD 实现服务发现
   - MessageQueueClient（消息队列客户端）：基于 Redis/Kafka 实现
   - PushService（推送服务）：独立 WebSocket 服务
   - SessionManager（会话管理）：维护 task_id 与客户端 WebSocket 连接的映射

3. **辅助工具层**（全框架共用）
   - Logger：统一日志类
   - ExceptionHandler：全局异常处理类
   - ConfigLoader：统一配置加载类

## 技术选型

- Python 版本：3.12
- 进程管理：multiprocessing（Python 内置）
- 消息队列：redis>=5.0.1 / kafka-python>=2.0.2
- 注册中心：etcd3>=1.1.0
- WebSocket 推送：websockets>=12.0
- 配置加载：pyyaml>=6.0.1
- 测试框架：pytest>=7.4.0
- 类型注解增强：typing-extensions>=4.8.0

## 项目结构

```
taskflow/
├── core/                  # 内核层核心代码
│   ├── plugin.py          # Plugin 抽象接口
│   ├── task.py            # Task/TaskMode 定义
│   ├── executor.py        # Executor/ExecutorStatus 定义
│   ├── scheduler_base.py  # 调度器基类
│   ├── local_scheduler.py # 本地调度器
│   ├── gateway_scheduler.py # 网关调度器
│   ├── executor_group.py  # 执行器组
│   └── result_writer.py   # 结果写入器
├── distributed/           # 分布式扩展层
│   ├── registry_center.py # 注册中心客户端
│   ├── message_queue.py   # 消息队列客户端
│   ├── push_service.py    # 推送服务
│   └── session_manager.py # 会话管理
├── utils/                 # 辅助工具层
│   ├── logger.py          # 统一日志
│   ├── exception_handler.py # 全局异常处理
│   └── config_loader.py   # 配置加载
├── config.py              # 全局配置
├── README.md              # 项目说明
├── requirements.txt       # 依赖清单
└── tests/                 # 测试目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 开发插件

```python
from core.plugin import Plugin, TaskResponse
from typing import Dict, Any

class MyPlugin(Plugin):
    @property
    def plugin_id(self) -> str:
        """插件类型唯一标识"""
        return "my_plugin"
    
    def pre_init(self, config: Dict[str, Any]) -> TaskResponse:
        """初始化插件"""
        try:
            # 模拟加载配置/资源
            self.model_path = config.get("model_path")
            if not self.model_path:
                return TaskResponse(
                    success=False,
                    code="param/missing",
                    message="插件配置缺失model_path参数",
                    data={"required_params": ["model_path"]}
                )
            return TaskResponse(
                success=True,
                code="success",
                message="插件初始化成功",
                data={"model_path": self.model_path}
            )
        except Exception as e:
            return TaskResponse(
                success=False,
                code="plugin/init_failed",
                message=f"插件初始化失败：{str(e)}",
                data={"error_detail": str(e)}
            )
    
    def execute(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        """执行单次任务"""
        try:
            # 模拟业务逻辑
            result = {"task_id": task_id, "result": "Success", "params": params}
            return TaskResponse(
                success=True,
                code="success",
                message="单次任务执行成功",
                data=result
            )
        except Exception as e:
            return TaskResponse(
                success=False,
                code="plugin/exec_error",
                message=f"单次任务执行失败：{str(e)}",
                data={"task_id": task_id}
            )
    
    def start_long_task(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        """启动长时任务"""
        try:
            # 模拟检查任务冲突
            if hasattr(self, "running_long_task") and self.running_long_task:
                return TaskResponse(
                    success=False,
                    code="task/conflict",
                    message=f"已有长时任务运行：{self.running_long_task}",
                    data={"current_task_id": self.running_long_task, "new_task_id": task_id}
                )
            # 模拟启动长时任务
            self.running_long_task = task_id
            return TaskResponse(
                success=True,
                code="success",
                message="长时任务启动成功",
                data={"task_id": task_id, "estimated_time": "1h"}
            )
        except Exception as e:
            return TaskResponse(
                success=False,
                code="plugin/long_task_failed",
                message=f"长时任务启动失败：{str(e)}",
                data={"task_id": task_id}
            )
    
    def stop_long_task(self, task_id: str) -> TaskResponse:
        """停止长时任务"""
        try:
            if not hasattr(self, "running_long_task") or self.running_long_task != task_id:
                return TaskResponse(
                    success=False,
                    code="task/not_found",
                    message=f"未找到运行中的长时任务：{task_id}",
                    data={"task_id": task_id}
                )
            # 模拟停止长时任务
            self.running_long_task = None
            return TaskResponse(
                success=True,
                code="success",
                message="长时任务停止成功",
                data={"task_id": task_id}
            )
        except Exception as e:
            return TaskResponse(
                success=False,
                code="plugin/stop_failed",
                message=f"长时任务停止失败：{str(e)}",
                data={"task_id": task_id}
            )
    
    def destroy(self) -> None:
        """资源销毁"""
        # 模拟清理资源
        self.running_long_task = None
        self.model_path = None
```

### 3. 使用本地调度器

```python
from core.local_scheduler import LocalScheduler
from core.task import Task, TaskMode

# 创建调度器
scheduler = LocalScheduler()

# 注册插件（带配置）
plugin_config = {
    "model_path": "/path/to/model"
}
scheduler.register_plugin('my_plugin', MyPlugin, plugin_config, max_executors=2)

# 提交任务
task = Task(
    task_id='task_1',
    plugin_id='my_plugin',
    mode=TaskMode.SINGLE,
    params={'key': 'value'}
)

# 提交单次任务并处理响应
result = scheduler.submit_task(task)
if result.success:
    print(f"任务执行成功：{result.data}")
else:
    print(f"任务执行失败：{result.code} - {result.message}")

# 提交长时任务
long_task = Task(
    task_id='long_task_1',
    plugin_id='my_plugin',
    mode=TaskMode.LONG,
    params={'duration': '1h'}
)

# 提交长时任务并处理响应
start_result = scheduler.submit_task(long_task)
if start_result.success:
    print(f"长时任务启动成功：{start_result.data}")
else:
    print(f"长时任务启动失败：{start_result.code} - {start_result.message}")

# 尝试启动另一个长时任务（应该失败，因为已有任务在运行）
another_long_task = Task(
    task_id='long_task_2',
    plugin_id='my_plugin',
    mode=TaskMode.LONG,
    params={'duration': '30min'}
)

conflict_result = scheduler.submit_task(another_long_task)
if conflict_result.success:
    print(f"长时任务启动成功：{conflict_result.data}")
else:
    print(f"长时任务启动失败：{conflict_result.code} - {conflict_result.message}")

# 停止长时任务并处理响应
stop_result = scheduler.stop_long_task('long_task_1', 'my_plugin')
if stop_result.success:
    print(f"长时任务停止成功：{stop_result.data}")
else:
    print(f"长时任务停止失败：{stop_result.code} - {stop_result.message}")

# 尝试停止不存在的任务
stop_nonexistent_result = scheduler.stop_long_task('nonexistent_task', 'my_plugin')
if stop_nonexistent_result.success:
    print(f"长时任务停止成功：{stop_nonexistent_result.data}")
else:
    print(f"长时任务停止失败：{stop_nonexistent_result.code} - {stop_nonexistent_result.message}")

# 关闭调度器
scheduler.shutdown()
```

### 4. 运行测试

```bash
pytest tests/test_basic.py
```

## 分布式部署

1. 启动 ETCD 服务
2. 启动 Redis/Kafka 服务
3. 启动本地调度器（注册到 ETCD）
4. 启动网关调度器（查询 ETCD）
5. 启动推送服务（WebSocket）

## 核心特性

- **极简设计**：核心内核保持极简，无冗余逻辑
- **插件化**：用户仅需实现 Plugin 接口即可开发业务插件
- **分布式支持**：支持单机/分布式无缝切换
- **状态管理**：所有状态管理仅通过状态机实现
- **可靠性**：执行器与插件严格 1:1 绑定，长时/单次任务状态严格隔离

## 注意事项

- 框架内核（Plugin/Executor/LocalScheduler）必须保持极简，无冗余逻辑
- 分布式组件与内核层解耦，内核层不感知分布式存在，可独立运行
- 代码兼容 Python3.12 语法，避免使用专属新特性（保证向下兼容 3.8+）
- 执行器组启动时全量预启动最大数执行器，无动态扩容/缩容，无客户端等待逻辑
- 执行器状态仅通过心跳保活，超时自动清理，无额外僵尸标记/释放逻辑