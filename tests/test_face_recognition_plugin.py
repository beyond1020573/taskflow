import sys
import os
import pytest
import time
import subprocess
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plugins.face_recognition_plugin import FaceRecognitionPlugin
from core.result_writer import ResultWriter, PrintResultWriter


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


class TestFaceRecognitionPluginLongTask:
    """FaceRecognitionPlugin 长时任务单元测试（使用 FFmpeg）"""
    
    def _create_mock_face(self, bbox=None, det_score=0.95):
        """创建模拟的人脸对象"""
        mock_face = Mock()
        mock_face.bbox = np.array(bbox) if bbox else np.array([10, 10, 100, 100])
        mock_face.det_score = det_score
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
    
    def _create_mock_process(self, width=640, height=480, num_frames=3):
        """创建模拟的 FFmpeg 进程"""
        mock_process = Mock()
        frame_size = width * height * 3
        frames = [np.zeros((height, width, 3), dtype=np.uint8).tobytes() for _ in range(num_frames)]
        
        read_index = [0]
        
        def mock_read(size):
            if read_index[0] < len(frames):
                data = frames[read_index[0]]
                read_index[0] += 1
                return data
            return b''
        
        mock_process.stdout.read = mock_read
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        mock_process.kill = Mock()
        return mock_process
    
    def _create_mock_ffprobe_result(self, width=1920, height=1080):
        """创建模拟的 ffprobe 结果"""
        return json.dumps({
            "streams": [{
                "width": width,
                "height": height,
                "codec_type": "video"
            }]
        })
    
    def test_get_stream_resolution_success(self):
        """测试 ffprobe 获取视频流分辨率成功"""
        plugin, _ = self._setup_plugin()
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = self._create_mock_ffprobe_result(1920, 1080)
        
        with patch('subprocess.run', return_value=mock_result):
            width, height, error = plugin._get_stream_resolution("rtsp://test")
            
            assert width == 1920
            assert height == 1080
            assert error is None
    
    def test_get_stream_resolution_ffprobe_error(self):
        """测试 ffprobe 执行失败"""
        plugin, _ = self._setup_plugin()
        
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "ffprobe error"
        
        with patch('subprocess.run', return_value=mock_result):
            width, height, error = plugin._get_stream_resolution("rtsp://invalid")
            
            assert width is None
            assert height is None
            assert "ffprobe 执行失败" in error
    
    def test_get_stream_resolution_no_streams(self):
        """测试视频流中无视频轨道"""
        plugin, _ = self._setup_plugin()
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"streams": []})
        
        with patch('subprocess.run', return_value=mock_result):
            width, height, error = plugin._get_stream_resolution("rtsp://test")
            
            assert width is None
            assert height is None
            assert "未找到视频流" in error
    
    def test_get_stream_resolution_ffprobe_not_found(self):
        """测试 ffprobe 未安装"""
        plugin, _ = self._setup_plugin()
        
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            width, height, error = plugin._get_stream_resolution("rtsp://test")
            
            assert width is None
            assert height is None
            assert "ffprobe 未安装" in error
    
    def test_get_stream_resolution_timeout(self):
        """测试 ffprobe 执行超时"""
        plugin, _ = self._setup_plugin()
        
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=10)):
            width, height, error = plugin._get_stream_resolution("rtsp://test")
            
            assert width is None
            assert height is None
            assert "超时" in error
    
    def test_start_long_task_auto_resolution(self):
        """测试 start_long_task 自动获取分辨率"""
        plugin, mock_model = self._setup_plugin()
        
        mock_writer = Mock(spec=ResultWriter)
        mock_process = self._create_mock_process(width=1920, height=1080, num_frames=2)
        mock_ffprobe_result = Mock()
        mock_ffprobe_result.returncode = 0
        mock_ffprobe_result.stdout = self._create_mock_ffprobe_result(1920, 1080)
        
        with patch('subprocess.run', return_value=mock_ffprobe_result):
            with patch('subprocess.Popen', return_value=mock_process):
                mock_model.get.return_value = [self._create_mock_face()]
                
                result = plugin.start_long_task("task_001", {
                    "stream_url": "rtsp://test",
                    "result_writer": mock_writer
                })
                
                assert result.success is True
                assert result.data["width"] == 1920
                assert result.data["height"] == 1080
                
                time.sleep(0.2)
                plugin.stop_long_task("task_001")
    
    def test_start_long_task_resolution_error(self):
        """测试 start_long_task 获取分辨率失败"""
        plugin, mock_model = self._setup_plugin()
        
        mock_writer = Mock(spec=ResultWriter)
        mock_ffprobe_result = Mock()
        mock_ffprobe_result.returncode = 1
        mock_ffprobe_result.stderr = "error"
        
        with patch('subprocess.run', return_value=mock_ffprobe_result):
            result = plugin.start_long_task("task_001", {
                "stream_url": "rtsp://invalid",
                "result_writer": mock_writer
            })
            
            assert result.success is False
            assert result.code == "stream/resolution_error"
            assert "无法获取视频流分辨率" in result.message
    
    def test_start_long_task_missing_stream_url(self):
        """测试 start_long_task 缺少 stream_url 参数"""
        plugin, _ = self._setup_plugin()
        
        result = plugin.start_long_task("task_001", {})
        
        assert result.success is False
        assert result.code == "param/missing"
        assert "缺少必需参数：stream_url" in result.message
        assert result.data["required_params"] == ["stream_url"]
    
    def test_start_long_task_plugin_not_initialized(self):
        """测试 start_long_task 插件未初始化"""
        plugin = FaceRecognitionPlugin()
        
        result = plugin.start_long_task("task_001", {"stream_url": "rtsp://test"})
        
        assert result.success is False
        assert result.code == "plugin/not_initialized"
        assert "插件未初始化" in result.message
    
    def test_start_long_task_already_running(self):
        """测试 start_long_task 任务已在运行"""
        plugin, mock_model = self._setup_plugin()
        
        mock_writer = Mock(spec=ResultWriter)
        mock_process = self._create_mock_process()
        mock_ffprobe_result = Mock()
        mock_ffprobe_result.returncode = 0
        mock_ffprobe_result.stdout = self._create_mock_ffprobe_result()
        
        with patch('subprocess.run', return_value=mock_ffprobe_result):
            with patch('subprocess.Popen', return_value=mock_process):
                mock_model.get.return_value = [self._create_mock_face()]
                
                result1 = plugin.start_long_task("task_001", {
                    "stream_url": "rtsp://test",
                    "result_writer": mock_writer
                })
                
                assert result1.success is True
                
                result2 = plugin.start_long_task("task_001", {
                    "stream_url": "rtsp://test",
                    "result_writer": mock_writer
                })
                
                assert result2.success is False
                assert result2.code == "task/already_running"
                
                plugin.stop_long_task("task_001")
    
    def test_stop_long_task_not_found(self):
        """测试 stop_long_task 任务不存在"""
        plugin, _ = self._setup_plugin()
        
        result = plugin.stop_long_task("task_001")
        
        assert result.success is False
        assert result.code == "task/not_found"
        assert "任务 task_001 不存在" in result.message
    
    def test_stop_long_task_success(self):
        """测试 stop_long_task 成功停止"""
        plugin, mock_model = self._setup_plugin()
        
        mock_writer = Mock(spec=ResultWriter)
        mock_process = self._create_mock_process(num_frames=10)
        mock_ffprobe_result = Mock()
        mock_ffprobe_result.returncode = 0
        mock_ffprobe_result.stdout = self._create_mock_ffprobe_result()
        
        with patch('subprocess.run', return_value=mock_ffprobe_result):
            with patch('subprocess.Popen', return_value=mock_process):
                mock_model.get.return_value = [self._create_mock_face()]
                
                plugin.start_long_task("task_001", {
                    "stream_url": "rtsp://test",
                    "result_writer": mock_writer
                })
                
                time.sleep(0.1)
                
                result = plugin.stop_long_task("task_001")
                
                assert result.success is True
                assert result.code == "success"
                assert "长时任务 task_001 已停止" in result.message
    
    def test_long_task_result_writer_called(self):
        """测试长时任务调用 result_writer"""
        plugin, mock_model = self._setup_plugin()
        
        mock_writer = Mock(spec=ResultWriter)
        mock_process = self._create_mock_process(num_frames=3)
        mock_ffprobe_result = Mock()
        mock_ffprobe_result.returncode = 0
        mock_ffprobe_result.stdout = self._create_mock_ffprobe_result()
        
        with patch('subprocess.run', return_value=mock_ffprobe_result):
            with patch('subprocess.Popen', return_value=mock_process):
                mock_model.get.return_value = [self._create_mock_face()]
                
                plugin.start_long_task("task_001", {
                    "stream_url": "rtsp://test",
                    "result_writer": mock_writer
                })
                
                time.sleep(0.3)
                
                assert mock_writer.write.called
                
                plugin.stop_long_task("task_001")
    
    def test_long_task_ffmpeg_not_found(self):
        """测试长时任务 FFmpeg 未安装"""
        plugin, mock_model = self._setup_plugin()
        
        mock_writer = Mock(spec=ResultWriter)
        mock_ffprobe_result = Mock()
        mock_ffprobe_result.returncode = 0
        mock_ffprobe_result.stdout = self._create_mock_ffprobe_result()
        
        with patch('subprocess.run', return_value=mock_ffprobe_result):
            with patch('subprocess.Popen', side_effect=FileNotFoundError("ffmpeg not found")):
                plugin.start_long_task("task_001", {
                    "stream_url": "rtsp://invalid",
                    "result_writer": mock_writer
                })
                
                time.sleep(0.2)
                
                assert mock_writer.write.called
                call_args = mock_writer.write.call_args
                assert call_args[0][0] == "task_001"
                assert call_args[0][1]["status"] == "error"
                assert "FFmpeg 未安装" in call_args[0][1]["message"]
    
    def test_destroy_stops_all_long_tasks(self):
        """测试 destroy 停止所有长时任务"""
        plugin, mock_model = self._setup_plugin()
        
        mock_writer = Mock(spec=ResultWriter)
        mock_process = self._create_mock_process(num_frames=10)
        mock_ffprobe_result = Mock()
        mock_ffprobe_result.returncode = 0
        mock_ffprobe_result.stdout = self._create_mock_ffprobe_result()
        
        with patch('subprocess.run', return_value=mock_ffprobe_result):
            with patch('subprocess.Popen', return_value=mock_process):
                mock_model.get.return_value = [self._create_mock_face()]
                
                plugin.start_long_task("task_001", {
                    "stream_url": "rtsp://test1",
                    "result_writer": mock_writer
                })
                plugin.start_long_task("task_002", {
                    "stream_url": "rtsp://test2",
                    "result_writer": mock_writer
                })
                
                time.sleep(0.1)
                
                assert len(plugin._long_tasks) == 2
                
                plugin.destroy()
                
                assert len(plugin._long_tasks) == 0
    
    def test_build_ffmpeg_command(self):
        """测试 FFmpeg 命令构建"""
        plugin, _ = self._setup_plugin()
        
        cmd = plugin._build_ffmpeg_command("rtsp://test", 1920, 1080)
        
        assert 'ffmpeg' in cmd
        assert '-i' in cmd
        assert 'rtsp://test' in cmd
        assert '-f' in cmd
        assert 'rawvideo' in cmd
        assert '-pix_fmt' in cmd
        assert 'bgr24' in cmd
        assert '-s' in cmd
        assert '1920x1080' in cmd


class TestResultWriter:
    """ResultWriter 单元测试"""
    
    def test_print_result_writer(self):
        """测试 PrintResultWriter"""
        from core.result_writer import PrintResultWriter
        
        writer = PrintResultWriter()
        result = writer.write("task_001", {"status": "success", "face_count": 1})
        
        assert result is True
        
        writer.close()
    
    def test_result_writer_abstract(self):
        """测试 ResultWriter 抽象类"""
        from core.result_writer import ResultWriter
        
        with pytest.raises(TypeError):
            ResultWriter()
