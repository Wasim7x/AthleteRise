import yaml
import os
from typing import Dict, Any

class ConfigReader:
    """Configuration file reader utility"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration file
        """
        try:
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")
