import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plugins.face_recognition_plugin import FaceRecognitionPlugin


class TestFaceRecognitionPluginPreInit:
    """FaceRecognitionPlugin.pre_init 单元测试"""
    
    def test_plugin_id(self):
        """测试 plugin_id 属性"""
        plugin = FaceRecognitionPlugin()
        assert plugin.plugin_id == "face_recognition"
    
    def test_pre_init_force_cuda(self):
        """测试 pre_init 强制 GPU 模式 (device='cuda')"""
        plugin = FaceRecognitionPlugin()
        
        mock_model = Mock()
        mock_face_analysis = Mock(return_value=mock_model)
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=mock_face_analysis)}):
            config = {
                "model_name": "buffalo_l",
                "device": "cuda",
                "ctx_id": 0
            }
            
            result = plugin.pre_init(config)
            
            assert result.success is True
            assert result.code == "success"
            assert result.data["device"] == "cuda"
            assert result.data["providers"] == ['CUDAExecutionProvider']
            
            mock_face_analysis.assert_called_once_with(
                name="buffalo_l",
                providers=['CUDAExecutionProvider']
            )
    
    def test_pre_init_auto_fallback(self):
        """测试 pre_init 智能自动模式 (device='auto')"""
        plugin = FaceRecognitionPlugin()
        
        mock_model = Mock()
        mock_face_analysis = Mock(return_value=mock_model)
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=mock_face_analysis)}):
            config = {
                "model_name": "buffalo_l",
                "device": "auto",
                "ctx_id": 0
            }
            
            result = plugin.pre_init(config)
            
            assert result.success is True
            assert result.code == "success"
            assert result.data["device"] == "auto"
            assert result.data["providers"] == ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            mock_face_analysis.assert_called_once_with(
                name="buffalo_l",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
    
    def test_pre_init_force_cpu(self):
        """测试 pre_init 强制 CPU 模式 (device='cpu')"""
        plugin = FaceRecognitionPlugin()
        
        mock_model = Mock()
        mock_face_analysis = Mock(return_value=mock_model)
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=mock_face_analysis)}):
            config = {
                "model_name": "buffalo_l",
                "device": "cpu",
                "ctx_id": 0
            }
            
            result = plugin.pre_init(config)
            
            assert result.success is True
            assert result.code == "success"
            assert result.data["device"] == "cpu"
            assert result.data["providers"] == ['CPUExecutionProvider']
            
            mock_face_analysis.assert_called_once_with(
                name="buffalo_l",
                providers=['CPUExecutionProvider']
            )
    
    def test_pre_init_invalid_device(self):
        """测试 pre_init 无效的 device 参数"""
        plugin = FaceRecognitionPlugin()
        
        mock_face_analysis = Mock()
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=mock_face_analysis)}):
            config = {
                "model_name": "buffalo_l",
                "device": "tpu",
                "ctx_id": 0
            }
            
            result = plugin.pre_init(config)
            
            assert result.success is False
            assert result.code == "plugin/invalid_config"
            assert "无效的 device 参数" in result.message
            assert "tpu" in result.message
            assert result.data["invalid_device"] == "tpu"
            assert result.data["supported_devices"] == ["cuda", "auto", "cpu"]
            
            mock_face_analysis.assert_not_called()
    
    def test_pre_init_with_default_config(self):
        """测试 pre_init 使用默认配置（默认 device='auto'）"""
        plugin = FaceRecognitionPlugin()
        
        mock_model = Mock()
        mock_face_analysis = Mock(return_value=mock_model)
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=mock_face_analysis)}):
            config = {}
            
            result = plugin.pre_init(config)
            
            assert result.success is True
            assert result.data["model_name"] == "buffalo_l"
            assert result.data["device"] == "auto"
            assert result.data["providers"] == ['CUDAExecutionProvider', 'CPUExecutionProvider']
            assert result.data["ctx_id"] == 0
    
    def test_pre_init_with_custom_config(self):
        """测试 pre_init 使用自定义配置"""
        plugin = FaceRecognitionPlugin()
        
        mock_model = Mock()
        mock_face_analysis = Mock(return_value=mock_model)
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=mock_face_analysis)}):
            config = {
                "model_name": "buffalo_m",
                "device": "cpu",
                "ctx_id": -1
            }
            
            result = plugin.pre_init(config)
            
            assert result.success is True
            assert result.data["model_name"] == "buffalo_m"
            assert result.data["device"] == "cpu"
            assert result.data["providers"] == ['CPUExecutionProvider']
            assert result.data["ctx_id"] == -1
    
    def test_pre_init_import_error(self):
        """测试 pre_init 缺少依赖库"""
        plugin = FaceRecognitionPlugin()
        
        def raise_import_error(*args, **kwargs):
            raise ImportError("No module named 'insightface'")
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=raise_import_error)}):
            config = {"device": "auto"}
            
            result = plugin.pre_init(config)
            
            assert result.success is False
            assert result.code == "plugin/dependency_missing"
            assert "缺少依赖库" in result.message
            assert "insightface" in result.message
            assert "error_detail" in result.data
    
    def test_pre_init_model_load_error(self):
        """测试 pre_init 模型加载失败"""
        plugin = FaceRecognitionPlugin()
        
        def raise_runtime_error(*args, **kwargs):
            raise RuntimeError("Model file not found")
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=raise_runtime_error)}):
            config = {"device": "auto"}
            
            result = plugin.pre_init(config)
            
            assert result.success is False
            assert result.code == "plugin/init_failed"
            assert "人脸模型加载失败" in result.message
            assert "Model file not found" in result.message
            assert "error_detail" in result.data
    
    def test_pre_init_prepare_error(self):
        """测试 pre_init 模型 prepare 失败"""
        plugin = FaceRecognitionPlugin()
        
        mock_model = Mock()
        mock_model.prepare.side_effect = RuntimeError("CUDA out of memory")
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=Mock(return_value=mock_model))}):
            config = {"device": "cuda"}
            
            result = plugin.pre_init(config)
            
            assert result.success is False
            assert result.code == "plugin/init_failed"
            assert "人脸模型加载失败" in result.message
            assert "CUDA out of memory" in result.message
    
    def test_pre_init_cuda_unavailable_error(self):
        """测试 pre_init 强制 CUDA 模式但 CUDA 不可用"""
        plugin = FaceRecognitionPlugin()
        
        def raise_cuda_error(*args, **kwargs):
            raise RuntimeError("CUDA execution provider is not available")
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=raise_cuda_error)}):
            config = {"device": "cuda"}
            
            result = plugin.pre_init(config)
            
            assert result.success is False
            assert result.code == "plugin/init_failed"
            assert "CUDA execution provider is not available" in result.message
    
    def test_destroy(self):
        """测试 destroy 方法"""
        plugin = FaceRecognitionPlugin()
        
        mock_model = Mock()
        
        with patch.dict('sys.modules', {'insightface.app': Mock(FaceAnalysis=Mock(return_value=mock_model))}):
            config = {"device": "auto"}
            plugin.pre_init(config)
            
            plugin.destroy()
            
            assert not hasattr(plugin, 'model') or plugin.model is None
            assert plugin.model_name is None
            assert plugin.device is None
            assert plugin.providers is None


class TestFaceRecognitionPluginExecute:
    """FaceRecognitionPlugin.execute 单元测试"""
    
    def _create_mock_face(self, bbox=None, kps=None, det_score=0.95, embedding=None):
        """创建模拟的人脸对象"""
        mock_face = Mock()
        mock_face.bbox = np.array(bbox) if bbox else np.array([10, 10, 100, 100])
        mock_face.kps = np.array(kps) if kps else np.array([[20, 20], [80, 20], [50, 50], [30, 80], [70, 80]])
        mock_face.det_score = det_score
        mock_face.landmark_3d_68 = None
        mock_face.pose = None
        mock_face.embedding = np.array(embedding) if embedding else np.random.rand(512)
        return mock_face
    
    def _setup_plugin(self):
        """设置已初始化的插件"""
        plugin = FaceRecognitionPlugin()
        mock_model = Mock()
        plugin.model = mock_model
        plugin.model_name = "buffalo_l"
        plugin.device = "auto"
        plugin.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return plugin, mock_model
    
    def test_execute_missing_image_param(self):
        """测试 execute 缺少 image 参数"""
        plugin, _ = self._setup_plugin()
        
        result = plugin.execute("task_001", {})
        
        assert result.success is False
        assert result.code == "param/missing"
        assert "缺少必需参数" in result.message
        assert "image" in result.message
        assert result.data["required_params"] == ["image"]
    
    def test_execute_with_file_path_success(self):
        """测试 execute 使用文件路径成功"""
        plugin, mock_model = self._setup_plugin()
        
        mock_face = self._create_mock_face()
        mock_model.get.return_value = [mock_face]
        
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            result = plugin.execute("task_001", {"image": "/path/to/image.jpg"})
            
            assert result.success is True
            assert result.code == "success"
            assert "检测到 1 张人脸" in result.message
            assert result.data["task_id"] == "task_001"
            assert result.data["face_count"] == 1
            assert len(result.data["faces"]) == 1
            
            mock_imread.assert_called_once_with("/path/to/image.jpg")
            mock_model.get.assert_called_once()
    
    def test_execute_with_numpy_array_success(self):
        """测试 execute 使用 numpy 数组成功"""
        plugin, mock_model = self._setup_plugin()
        
        mock_face = self._create_mock_face()
        mock_model.get.return_value = [mock_face]
        
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = plugin.execute("task_001", {"image": image_array})
        
        assert result.success is True
        assert result.code == "success"
        assert result.data["face_count"] == 1
        mock_model.get.assert_called_once()
    
    def test_execute_multiple_faces(self):
        """测试 execute 检测多张人脸"""
        plugin, mock_model = self._setup_plugin()
        
        mock_faces = [
            self._create_mock_face(bbox=[10, 10, 50, 50], det_score=0.95),
            self._create_mock_face(bbox=[60, 60, 100, 100], det_score=0.88),
            self._create_mock_face(bbox=[110, 110, 150, 150], det_score=0.92)
        ]
        mock_model.get.return_value = mock_faces
        
        image_array = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = plugin.execute("task_001", {"image": image_array})
        
        assert result.success is True
        assert result.data["face_count"] == 3
        assert len(result.data["faces"]) == 3
        
        for i, face_data in enumerate(result.data["faces"]):
            assert face_data["index"] == i
            assert "bbox" in face_data
            assert "det_score" in face_data
    
    def test_execute_no_faces_detected(self):
        """测试 execute 未检测到人脸"""
        plugin, mock_model = self._setup_plugin()
        
        mock_model.get.return_value = []
        
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = plugin.execute("task_001", {"image": image_array})
        
        assert result.success is True
        assert result.code == "success"
        assert "检测到 0 张人脸" in result.message
        assert result.data["face_count"] == 0
        assert result.data["faces"] == []
    
    def test_execute_invalid_file_path(self):
        """测试 execute 无效的文件路径"""
        plugin, _ = self._setup_plugin()
        
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = None
            
            result = plugin.execute("task_001", {"image": "/invalid/path.jpg"})
            
            assert result.success is False
            assert result.code == "param/invalid"
            assert "无法读取图像文件" in result.message
            assert result.data["image_path"] == "/invalid/path.jpg"
    
    def test_execute_invalid_image_type(self):
        """测试 execute 无效的图像类型"""
        plugin, _ = self._setup_plugin()
        
        result = plugin.execute("task_001", {"image": 12345})
        
        assert result.success is False
        assert result.code == "param/invalid"
        assert "图像数据格式无效" in result.message
        assert result.data["image_type"] == "int"
    
    def test_execute_model_error(self):
        """测试 execute 模型执行错误"""
        plugin, mock_model = self._setup_plugin()
        
        mock_model.get.side_effect = RuntimeError("Model inference failed")
        
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = plugin.execute("task_001", {"image": image_array})
        
        assert result.success is False
        assert result.code == "plugin/exec_error"
        assert "人脸识别执行失败" in result.message
        assert "Model inference failed" in result.message
        assert result.data["task_id"] == "task_001"
    
    def test_execute_face_with_all_attributes(self):
        """测试 execute 人脸包含所有属性"""
        plugin, mock_model = self._setup_plugin()
        
        mock_face = Mock()
        mock_face.bbox = np.array([10, 10, 100, 100])
        mock_face.kps = np.array([[20, 20], [80, 20], [50, 50], [30, 80], [70, 80]])
        mock_face.det_score = 0.98
        mock_face.landmark_3d_68 = np.random.rand(68, 3)
        mock_face.pose = np.array([0.1, 0.2, 0.3])
        mock_face.embedding = np.random.rand(512)
        
        mock_model.get.return_value = [mock_face]
        
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = plugin.execute("task_001", {"image": image_array})
        
        assert result.success is True
        face_data = result.data["faces"][0]
        
        assert "bbox" in face_data
        assert "kps" in face_data
        assert face_data["det_score"] == 0.98
        assert "landmark_3d_68" in face_data
        assert "pose" in face_data
        assert "embedding" in face_data
