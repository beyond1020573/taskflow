# 统一配置加载类
import json
import yaml
from typing import Dict, Any
from utils.logger import Logger

class ConfigLoader:
    @staticmethod
    def load(file_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        logger = Logger.get_logger(__name__)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")
            
            logger.info(f"Config loaded from {file_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise