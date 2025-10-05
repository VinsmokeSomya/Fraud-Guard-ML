"""
Logging configuration for the fraud detection system.
"""
import os
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration using loguru.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, uses default location
        rotation: Log rotation policy
        retention: Log retention policy
        format_string: Custom format string for log messages
    """
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler if log_file is specified
    if log_file is None:
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "fraud_detection.log"
    
    logger.add(
        log_file,
        level=log_level,
        format=format_string,
        rotation=rotation,
        retention=retention,
        backtrace=True,
        diagnose=True,
        enqueue=True  # Thread-safe logging
    )
    
    logger.info(f"Logging initialized with level: {log_level}")
    logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module/component
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Initialize logging with default settings
def init_default_logging():
    """Initialize logging with default settings."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level=log_level)


# Auto-initialize logging when module is imported
init_default_logging()