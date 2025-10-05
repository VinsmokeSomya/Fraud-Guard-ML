"""
Configuration settings for the fraud detection system.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config files."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Paths
    project_root: Path = Path(__file__).parent.parent
    config_dir: Path = project_root / "config"
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"
    
    # Database (if needed in future)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Model Settings
    model_retrain_interval_hours: int = Field(default=24, env="MODEL_RETRAIN_INTERVAL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def get_data_path(filename: str) -> Path:
    """Get full path to data file."""
    settings = Settings()
    return settings.data_dir / "raw" / filename


def get_model_path(model_name: str) -> Path:
    """Get full path to model file."""
    settings = Settings()
    return settings.models_dir / f"{model_name}.joblib"


def get_log_path(log_name: str) -> Path:
    """Get full path to log file."""
    settings = Settings()
    return settings.logs_dir / f"{log_name}.log"


# Global settings instance
settings = Settings()
config = load_config()