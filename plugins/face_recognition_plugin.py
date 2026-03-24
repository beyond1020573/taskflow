from typing import Dict, Any, Tuple, Optional
import threading
import time
import subprocess
import json
import numpy as np
from core.plugin import Plugin, TaskResponse
from core.result_writer import ResultWriter, PrintResultWriter


class FaceRecognitionPlugin(Plugin):
    """人脸识别插件（基于 InsightFace）"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None
        self.providers = None
        self._long_tasks: Dict[str, threading.Thread] = {}
        self._stop_flags: Dict[str, bool] = {}
        self._result_writers: Dict[str, ResultWriter] = {}
    
    @property
    def plugin_id(self) -> str:
        """插件类型唯一标识"""
        return "face_recognition"
    
    def pre_init(self, config: Dict[str, Any]) -> TaskResponse:
        """初始化插件，加载 InsightFace 人脸模型
        
        Args:
            config: 插件配置字典
                - model_name: 模型名称（默认 buffalo_l）
                - device: 设备类型，支持 "cuda"(强制GPU), "auto"(自动降级), "cpu"(强制CPU)
                - ctx_id: 上下文 ID（默认 0）
        
        Returns:
            TaskResponse: 初始化结果
        """
        try:
            from insightface.app import FaceAnalysis
            
            model_name = config.get("model_name", "buffalo_l")
            device = config.get("device", "auto")
            ctx_id = config.get("ctx_id", 0)
            
            if device == "cuda":
                providers = ['CUDAExecutionProvider']
            elif device == "auto":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif device == "cpu":
                providers = ['CPUExecutionProvider']
            else:
                return TaskResponse(
                    success=False,
                    code="plugin/invalid_config",
                    message=f"无效的 device 参数：{device}，支持的选项为 'cuda', 'auto', 'cpu'",
                    data={"invalid_device": device, "supported_devices": ["cuda", "auto", "cpu"]}
                )
            
            self.model = FaceAnalysis(
                name=model_name,
                providers=providers
            )
            self.model.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            self.model_name = model_name
            self.device = device
            self.providers = providers
            
            return TaskResponse(
                success=True,
                code="success",
                message="人脸模型加载成功",
                data={
                    "model_name": model_name,
                    "device": device,
                    "providers": providers,
                    "ctx_id": ctx_id
                }
            )
        except ImportError as e:
            return TaskResponse(
                success=False,
                code="plugin/dependency_missing",
                message=f"缺少依赖库：{str(e)}，请安装 insightface",
                data={"error_detail": str(e)}
            )
        except Exception as e:
            return TaskResponse(
                success=False,
                code="plugin/init_failed",
                message=f"人脸模型加载失败：{str(e)}",
                data={"error_detail": str(e)}
            )
    
    def execute(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        """执行单次人脸识别任务"""
        try:
            image_data = params.get("image")
            if image_data is None:
                return TaskResponse(
                    success=False,
                    code="param/missing",
                    message="缺少必需参数：image",
                    data={"required_params": ["image"]}
                )
                
            image = None

            if isinstance(image_data, str):
                import cv2
                image = cv2.imread(image_data)
                if image is None:
                    return TaskResponse(
                        success=False,
                        code="param/invalid",
                        message=f"无法读取图像文件：{image_data}",
                        data={"image_path": image_data}
                    )
            elif isinstance(image_data, np.ndarray):
                image = image_data
            else:
                return TaskResponse(
                    success=False,
                    code="param/invalid",
                    message="图像数据格式无效，支持文件路径或 numpy 数组",
                    data={"image_type": type(image_data).__name__}
                )
            
            faces = self.model.get(image)
            
            face_results = []
            for idx, face in enumerate(faces):
                face_info = {
                    "index": idx,
                    "bbox": face.bbox.tolist(),
                    "kps": face.kps.tolist() if face.kps is not None else None,
                    "det_score": float(face.det_score),
                    "landmark_3d_68": face.landmark_3d_68.tolist() if face.landmark_3d_68 is not None else None,
                    "pose": face.pose.tolist() if face.pose is not None else None,
                    "embedding": face.embedding.tolist() if face.embedding is not None else None
                }
                face_results.append(face_info)
            
            return TaskResponse(
                success=True,
                code="success",
                message=f"检测到 {len(faces)} 张人脸",
                data={
                    "task_id": task_id,
                    "face_count": len(faces),
                    "faces": face_results
                }
            )
        except Exception as e:
            return TaskResponse(
                success=False,
                code="plugin/exec_error",
                message=f"人脸识别执行失败：{str(e)}",
                data={"task_id": task_id}
            )
    
    def _get_stream_resolution(self, stream_url: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """使用 ffprobe 获取视频流分辨率
        
        Args:
            stream_url: 视频流地址
            
        Returns:
            Tuple[width, height, error_message]: 宽度、高度、错误信息（成功时为 None）
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-select_streams', 'v:0',
                stream_url
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return None, None, f"ffprobe 执行失败：{result.stderr}"
            
            probe_data = json.loads(result.stdout)
            
            if not probe_data.get('streams'):
                return None, None, "未找到视频流"
            
            stream = probe_data['streams'][0]
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))
            
            if width <= 0 or height <= 0:
                return None, None, f"无效的分辨率：{width}x{height}"
            
            return width, height, None
            
        except FileNotFoundError:
            return None, None, "ffprobe 未安装或不在 PATH 中，请先安装 FFmpeg"
        except subprocess.TimeoutExpired:
            return None, None, "ffprobe 执行超时"
        except json.JSONDecodeError as e:
            return None, None, f"ffprobe 输出解析失败：{str(e)}"
        except Exception as e:
            return None, None, f"获取视频流分辨率失败：{str(e)}"
    
    def start_long_task(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        """启动长时任务（视频流人脸检测）
        
        Args:
            task_id: 任务 ID
            params: 任务参数
                - stream_url: 视频流地址（RTSP/RTMP/HTTP-FLV/本地文件路径）
                - result_writer: 结果写入器实例（可选，默认使用 PrintResultWriter）
                - frame_interval: 帧间隔（可选，默认 1，即每帧都处理）
                - detect_interval: 检测间隔秒数（可选，默认 0，即每帧都检测）
        
        Returns:
            TaskResponse: 启动结果
        """
        if self.model is None:
            return TaskResponse(
                success=False,
                code="plugin/not_initialized",
                message="插件未初始化，请先调用 pre_init",
                data={"task_id": task_id}
            )
        
        stream_url = params.get("stream_url")
        if stream_url is None:
            return TaskResponse(
                success=False,
                code="param/missing",
                message="缺少必需参数：stream_url",
                data={"task_id": task_id, "required_params": ["stream_url"]}
            )
        
        if task_id in self._long_tasks:
            return TaskResponse(
                success=False,
                code="task/already_running",
                message=f"任务 {task_id} 已在运行中",
                data={"task_id": task_id}
            )
        
        result_writer = params.get("result_writer", PrintResultWriter())
        frame_interval = params.get("frame_interval", 1)
        detect_interval = params.get("detect_interval", 0)
        
        width, height, error = self._get_stream_resolution(stream_url)
        if error:
            return TaskResponse(
                success=False,
                code="stream/resolution_error",
                message=f"无法获取视频流分辨率：{error}",
                data={"task_id": task_id, "stream_url": stream_url}
            )
        
        self._stop_flags[task_id] = False
        self._result_writers[task_id] = result_writer
        
        thread = threading.Thread(
            target=self._run_stream_detection,
            args=(task_id, stream_url, result_writer, frame_interval, detect_interval, width, height),
            daemon=True
        )
        self._long_tasks[task_id] = thread
        thread.start()
        
        return TaskResponse(
            success=True,
            code="success",
            message=f"长时任务 {task_id} 已启动",
            data={
                "task_id": task_id,
                "stream_url": stream_url,
                "frame_interval": frame_interval,
                "detect_interval": detect_interval,
                "width": width,
                "height": height
            }
        )
    
    def stop_long_task(self, task_id: str) -> TaskResponse:
        """停止长时任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            TaskResponse: 停止结果
        """
        if task_id not in self._long_tasks:
            return TaskResponse(
                success=False,
                code="task/not_found",
                message=f"任务 {task_id} 不存在",
                data={"task_id": task_id}
            )
        
        self._stop_flags[task_id] = True
        
        thread = self._long_tasks[task_id]
        thread.join(timeout=5.0)
        
        if task_id in self._result_writers:
            self._result_writers[task_id].close()
            del self._result_writers[task_id]
        
        del self._long_tasks[task_id]
        del self._stop_flags[task_id]
        
        return TaskResponse(
            success=True,
            code="success",
            message=f"长时任务 {task_id} 已停止",
            data={"task_id": task_id}
        )
    
    def _build_ffmpeg_command(self, stream_url: str, width: int, height: int) -> list:
        """构建 FFmpeg 命令
        
        Args:
            stream_url: 视频流地址
            width: 输出宽度
            height: 输出高度
            
        Returns:
            list: FFmpeg 命令参数列表
        """
        cmd = [
            'ffmpeg',
            '-i', stream_url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-'
        ]
        return cmd
    
    def _run_stream_detection(
        self,
        task_id: str,
        stream_url: str,
        result_writer: ResultWriter,
        frame_interval: int,
        detect_interval: float,
        width: int,
        height: int
    ) -> None:
        """运行视频流人脸检测（后台线程，使用 FFmpeg）
        
        Args:
            task_id: 任务 ID
            stream_url: 视频流地址
            result_writer: 结果写入器
            frame_interval: 帧间隔
            detect_interval: 检测间隔秒数
            width: 视频宽度
            height: 视频高度
        """
        process = None
        has_error = False
        frame_size = width * height * 3
        
        try:
            cmd = self._build_ffmpeg_command(stream_url, width, height)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_size * 2
            )
            
            frame_count = 0
            last_detect_time = 0
            
            while not self._stop_flags.get(task_id, False):
                raw_frame = process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    if len(raw_frame) == 0:
                        has_error = True
                        result_writer.write(task_id, {
                            "status": "error",
                            "message": "视频流读取失败或已结束"
                        })
                    else:
                        has_error = True
                        result_writer.write(task_id, {
                            "status": "error",
                            "message": f"视频帧数据不完整：期望 {frame_size} 字节，实际 {len(raw_frame)} 字节"
                        })
                    break
                
                frame_count += 1
                
                if frame_count % frame_interval != 0:
                    continue
                
                current_time = time.time()
                if detect_interval > 0 and (current_time - last_detect_time) < detect_interval:
                    continue
                
                last_detect_time = current_time
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                
                try:
                    faces = self.model.get(frame)
                    
                    face_results = []
                    for idx, face in enumerate(faces):
                        face_info = {
                            "index": idx,
                            "bbox": face.bbox.tolist(),
                            "det_score": float(face.det_score)
                        }
                        face_results.append(face_info)
                    
                    result = {
                        "status": "success",
                        "frame_count": frame_count,
                        "timestamp": current_time,
                        "face_count": len(faces),
                        "faces": face_results
                    }
                    result_writer.write(task_id, result)
                    
                except Exception as e:
                    result_writer.write(task_id, {
                        "status": "error",
                        "frame_count": frame_count,
                        "message": f"人脸检测失败：{str(e)}"
                    })
                    
        except FileNotFoundError:
            has_error = True
            result_writer.write(task_id, {
                "status": "error",
                "message": "FFmpeg 未安装或不在 PATH 中，请先安装 FFmpeg"
            })
        except Exception as e:
            has_error = True
            result_writer.write(task_id, {
                "status": "error",
                "message": f"视频流处理异常：{str(e)}"
            })
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            if not has_error and task_id in self._stop_flags and self._stop_flags[task_id]:
                result_writer.write(task_id, {
                    "status": "stopped",
                    "message": "视频流处理已停止"
                })
    
    def destroy(self) -> None:
        """资源销毁"""
        for task_id in list(self._long_tasks.keys()):
            self.stop_long_task(task_id)
        
        if hasattr(self, "model") and self.model is not None:
            del self.model
        self.model = None
        self.model_name = None
        self.device = None
        self.providers = None
