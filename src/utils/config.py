"""
Utility functions for configuration management.
"""
from pathlib import Path
from typing import Dict, Any
from config.settings import settings, config, load_config
from config.logging_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


def get_config() -> Dict[str, Any]:
    """Get the global configuration dictionary."""
    return config


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a specific setting from the configuration.
    
    Args:
        key: Configuration key (supports dot notation, e.g., 'models.xgboost.n_estimators')
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    try:
        keys = key.split('.')
        value = config
        
        for k in keys:
            value = value[k]
        
        return value
    except (KeyError, TypeError):
        logger.warning(f"Configuration key '{key}' not found, using default: {default}")
        return default


def validate_paths() -> bool:
    """
    Validate that all required directories exist.
    
    Returns:
        True if all paths are valid, False otherwise
    """
    required_dirs = [
        settings.data_dir / "raw",
        settings.data_dir / "processed", 
        settings.models_dir,
        settings.logs_dir
    ]
    
    all_valid = True
    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            all_valid = False
        else:
            logger.debug(f"Directory exists: {dir_path}")
    
    return all_valid


def create_directories() -> None:
    """Create all required directories if they don't exist."""
    required_dirs = [
        settings.data_dir / "raw",
        settings.data_dir / "processed",
        settings.models_dir,
        settings.logs_dir
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")


# Initialize directories on import
create_directories()