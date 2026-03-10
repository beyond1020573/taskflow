# 全局异常处理类
import traceback
from utils.logger import Logger

class TaskFlowException(Exception):
    """TaskFlow 基础异常"""
    pass

class PluginException(TaskFlowException):
    """插件异常"""
    pass

class ExecutorException(TaskFlowException):
    """执行器异常"""
    pass

class SchedulerException(TaskFlowException):
    """调度器异常"""
    pass

class DistributedException(TaskFlowException):
    """分布式组件异常"""
    pass

class ExceptionHandler:
    @staticmethod
    def handle_exception(e: Exception) -> dict:
        """处理异常并返回标准化结果"""
        logger = Logger.get_logger(__name__)
        
        # 记录异常信息
        logger.error(f"Exception occurred: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 标准化异常返回
        if isinstance(e, TaskFlowException):
            return {
                'success': False,
                'error_code': type(e).__name__,
                'error_message': str(e)
            }
        else:
            return {
                'success': False,
                'error_code': 'UnknownError',
                'error_message': str(e)
            }