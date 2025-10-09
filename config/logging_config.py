"""
Logging configuration for the fraud detection system.
"""
import os
import sys
import json
import uuid
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any
from contextvars import ContextVar
from datetime import datetime


# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id.set(cid)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id.get()


def structured_formatter(record: Dict[str, Any]) -> str:
    """
    Format log record as structured JSON.
    
    Args:
        record: Log record dictionary
        
    Returns:
        JSON formatted log string
    """
    # Extract basic information
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "correlation_id": get_correlation_id(),
        "process_id": record["process"].id,
        "thread_id": record["thread"].id,
    }
    
    # Add extra fields if present
    if "extra" in record:
        log_entry.update(record["extra"])
    
    # Add exception information if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback
        }
    
    return json.dumps(log_entry, default=str)


def human_readable_formatter(record: Dict[str, Any]) -> str:
    """
    Format log record in human-readable format.
    
    Args:
        record: Log record dictionary
        
    Returns:
        Human-readable log string
    """
    cid = get_correlation_id()
    cid_part = f" | CID:{cid[:8]}" if cid else ""
    
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        f"{cid_part} | "
        "<level>{message}</level>"
    )


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    format_type: str = "human",
    enable_correlation_ids: bool = True,
    log_to_console: bool = True,
    log_to_file: bool = True,
    enable_performance_logging: bool = False
) -> None:
    """
    Setup logging configuration using loguru.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, uses default location
        rotation: Log rotation policy
        retention: Log retention policy
        format_type: Format type ('human', 'structured', 'detailed')
        enable_correlation_ids: Whether to enable correlation ID tracking
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        enable_performance_logging: Whether to enable performance logging
    """
    # Remove default handler
    logger.remove()
    
    # Choose formatter based on format type
    if format_type == "structured":
        formatter = structured_formatter
        colorize = False
    elif format_type == "detailed":
        formatter = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{process.id}</cyan>:<cyan>{thread.id}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "CID:{extra[correlation_id]} | "
            "<level>{message}</level> | "
            "{extra}"
        )
        colorize = True
    else:  # human
        formatter = human_readable_formatter
        colorize = True
    
    # Add console handler
    if log_to_console:
        logger.add(
            sys.stdout,
            level=log_level,
            format=formatter,
            colorize=colorize,
            backtrace=True,
            diagnose=True,
            filter=lambda record: record["level"].name != "PERFORMANCE" or enable_performance_logging
        )
    
    # Add file handler
    if log_to_file:
        if log_file is None:
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "fraud_detection.log"
        
        logger.add(
            log_file,
            level=log_level,
            format=structured_formatter if format_type == "structured" else formatter,
            rotation=rotation,
            retention=retention,
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Thread-safe logging
            filter=lambda record: record["level"].name != "PERFORMANCE" or enable_performance_logging
        )
    
    # Add performance log file if enabled
    if enable_performance_logging:
        perf_log_file = Path(log_file).parent / "performance.log" if log_file else Path(__file__).parent.parent / "logs" / "performance.log"
        logger.add(
            perf_log_file,
            level="DEBUG",
            format=structured_formatter,
            rotation="50 MB",
            retention="7 days",
            enqueue=True,
            filter=lambda record: record["level"].name == "PERFORMANCE"
        )
    
    logger.info(f"Logging initialized with level: {log_level}")
    logger.info(f"Format type: {format_type}")
    logger.info(f"Correlation IDs enabled: {enable_correlation_ids}")
    if log_to_file:
        logger.info(f"Log file: {log_file}")
    if enable_performance_logging:
        logger.info("Performance logging enabled")


def get_logger(name: str) -> logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module/component
        
    Returns:
        Logger instance with correlation ID binding
    """
    return logger.bind(name=name, correlation_id=get_correlation_id())


def performance_logger():
    """Get a performance logger instance."""
    return logger.bind(level="PERFORMANCE", correlation_id=get_correlation_id())


# Initialize logging with default settings
def init_default_logging():
    """Initialize logging with default settings from environment and config."""
    from config.settings import settings, config
    
    # Get logging configuration
    logging_config = config.get("logging", {})
    
    setup_logging(
        log_level=logging_config.get("level", settings.log_level),
        format_type=logging_config.get("format", "human"),
        enable_correlation_ids=logging_config.get("enable_correlation_ids", settings.enable_correlation_ids),
        log_to_console=logging_config.get("log_to_console", True),
        log_to_file=logging_config.get("log_to_file", True),
        rotation=logging_config.get("rotation", "10 MB"),
        retention=logging_config.get("retention", "30 days"),
        enable_performance_logging=logging_config.get("enable_performance_logging", False)
    )


# Auto-initialize logging when module is imported
init_default_logging()