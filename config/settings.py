"""
Configuration settings for the fraud detection system.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import Field
import logging

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
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_correlation_ids: bool = Field(default=True, env="ENABLE_CORRELATION_IDS")
    
    # Monitoring Settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_interval: int = Field(default=60, env="METRICS_INTERVAL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config(config_path: Optional[str] = None, environment: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment-specific overrides.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        environment: Environment name (development, staging, production)
        
    Returns:
        Dictionary containing configuration parameters
    """
    # Load base configuration
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load environment-specific configuration if specified
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    env_config_path = Path(__file__).parent / "environments" / f"{environment}.yaml"
    if env_config_path.exists():
        with open(env_config_path, 'r') as file:
            env_config = yaml.safe_load(file)
            # Merge environment config with base config
            config = merge_configs(config, env_config)
    
    # Substitute environment variables
    config = substitute_env_vars(config)
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def substitute_env_vars(config: Union[Dict, str, Any]) -> Union[Dict, str, Any]:
    """
    Recursively substitute environment variables in configuration.
    
    Args:
        config: Configuration value (dict, string, or other)
        
    Returns:
        Configuration with environment variables substituted
    """
    if isinstance(config, dict):
        return {key: substitute_env_vars(value) for key, value in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    else:
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
config = load_config(environment=settings.environment)