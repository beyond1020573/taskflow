from typing import Dict, Any
import numpy as np
from core.plugin import Plugin, TaskResponse


class FaceRecognitionPlugin(Plugin):
    """人脸识别插件（基于 InsightFace）"""
    
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
            
            # 获取模型配置
            model_name = config.get("model_name", "buffalo_l")
            device = config.get("device", "auto")
            ctx_id = config.get("ctx_id", 0)
            
            # 根据 device 参数构建 providers 列表
            if device == "cuda":
                # 强制 GPU 模式：仅使用 CUDA，如果不可用则抛出异常
                providers = ['CUDAExecutionProvider']
            elif device == "auto":
                # 智能自动模式：优先 GPU，自动回退到 CPU
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif device == "cpu":
                # 强制 CPU 模式：仅使用 CPU
                providers = ['CPUExecutionProvider']
            else:
                # 无效的 device 参数
                return TaskResponse(
                    success=False,
                    code="plugin/invalid_config",
                    message=f"无效的 device 参数：{device}，支持的选项为 'cuda', 'auto', 'cpu'",
                    data={"invalid_device": device, "supported_devices": ["cuda", "auto", "cpu"]}
                )
            
            # 加载模型
            self.model = FaceAnalysis(
                name=model_name,
                providers=providers
            )
            self.model.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            # 保存配置
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
            # 获取输入图像
            image_data = params.get("image")
            if image_data is None:
                return TaskResponse(
                    success=False,
                    code="param/missing",
                    message="缺少必需参数：image",
                    data={"required_params": ["image"]}
                )
                
            # 初始化 image 变量，避免静态检查警告
            image = None

            # 转换图像数据
            if isinstance(image_data, str):
                # 如果是文件路径，读取图像
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
                # 如果已经是 numpy 数组，直接使用
                image = image_data
            else:
                return TaskResponse(
                    success=False,
                    code="param/invalid",
                    message="图像数据格式无效，支持文件路径或 numpy 数组",
                    data={"image_type": type(image_data).__name__}
                )
            
            # 执行人脸检测
            faces = self.model.get(image)
            
            # 提取人脸信息
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
    
    def start_long_task(self, task_id: str, params: Dict[str, Any]) -> TaskResponse:
        """启动长时任务（人脸识别插件不支持）"""
        return TaskResponse(
            success=False,
            code="task/unsupported",
            message="人脸识别插件不支持长时任务",
            data={"task_id": task_id}
        )
    
    def stop_long_task(self, task_id: str) -> TaskResponse:
        """停止长时任务（人脸识别插件不支持）"""
        return TaskResponse(
            success=False,
            code="task/unsupported",
            message="人脸识别插件不支持长时任务",
            data={"task_id": task_id}
        )
    
    def destroy(self) -> None:
        """资源销毁"""
        if hasattr(self, "model"):
            del self.model
        self.model_name = None
        self.device = None
        self.providers = None
