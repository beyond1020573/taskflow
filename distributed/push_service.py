# 推送服务
import asyncio
import websockets
import json
from typing import Dict, Any
from distributed.message_queue import MessageQueueClient
from distributed.session_manager import SessionManager
from config import Config
from utils.logger import Logger

class PushService:
    """推送服务"""
    
    def __init__(self):
        self.message_queue = MessageQueueClient()
        self.session_manager = SessionManager()
        self.logger = Logger.get_logger(__name__)
    
    async def handle_connection(self, websocket, path):
        """处理 WebSocket 连接"""
        try:
            # 接收客户端注册信息
            register_message = await websocket.recv()
            register_data = json.loads(register_message)
            client_id = register_data.get('client_id')
            
            if client_id:
                # 注册会话
                self.session_manager.register_session(client_id, websocket)
                self.logger.info(f"Client {client_id} connected")
                
                # 保持连接
                while True:
                    # 接收客户端消息（如果需要）
                    await websocket.recv()
        except websockets.ConnectionClosed:
            # 连接关闭，移除会话
            self.session_manager.remove_session_by_websocket(websocket)
            self.logger.info("Client disconnected")
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
            self.session_manager.remove_session_by_websocket(websocket)
    
    async def consume_messages(self):
        """消费消息队列"""
        while True:
            try:
                # 从消息队列接收消息
                message = self.message_queue.receive_message('task_results', timeout=1)
                if message:
                    task_id = message.get('task_id')
                    result = message.get('result')
                    
                    # 查找对应的客户端
                    client_id = self.session_manager.get_client_id_by_task_id(task_id)
                    if client_id:
                        # 获取 WebSocket 连接
                        websocket = self.session_manager.get_websocket_by_client_id(client_id)
                        if websocket:
                            try:
                                # 推送结果
                                await websocket.send(json.dumps({
                                    'task_id': task_id,
                                    'result': result
                                }))
                                self.logger.info(f"Result pushed for task {task_id}")
                            except Exception as e:
                                self.logger.error(f"Failed to push result: {e}")
            except Exception as e:
                self.logger.error(f"Error consuming messages: {e}")
            
            # 短暂休眠
            await asyncio.sleep(0.1)
    
    async def start(self):
        """启动推送服务"""
        # 启动 WebSocket 服务
        server = await websockets.serve(
            self.handle_connection,
            Config.PUSH_SERVICE_HOST,
            Config.PUSH_SERVICE_PORT
        )
        
        self.logger.info(f"Push service started at ws://{Config.PUSH_SERVICE_HOST}:{Config.PUSH_SERVICE_PORT}")
        
        # 启动消息消费
        consume_task = asyncio.create_task(self.consume_messages())
        
        # 等待服务结束
        await server.wait_closed()
        await consume_task
    
    def run(self):
        """运行推送服务"""
        asyncio.run(self.start())
    
    def stop(self):
        """停止推送服务"""
        self.message_queue.close()
        self.session_manager.close()
        self.logger.info("Push service stopped")