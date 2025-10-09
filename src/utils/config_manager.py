"""
Configuration manager for the fraud detection system.

This module provides centralized configuration management with environment-specific
overrides, validation, and runtime configuration updates.
"""

import os
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import yaml
from dataclasses import dataclass
from enum import Enum

from config.settings import Settings, load_config, merge_configs, substitute_env_vars
from config.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class Environment(Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    timeout: int = 30
    enable_cors: bool = False
    cors_origins: List[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "human"
    enable_correlation_ids: bool = True
    enable_performance_logging: bool = False
    log_to_file: bool = True
    log_to_console: bool = True
    rotation: str = "10 MB"
    retention: str = "30 days"


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_metrics_collection: bool = True
    metrics_interval: int = 60
    enable_profiling: bool = False
    enable_tracing: bool = False
    trace_sampling_rate: float = 0.1
    metrics_retention_days: int = 30


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_rate_limiting: bool = False
    rate_limit_per_minute: int = 1000
    enable_authentication: bool = False
    enable_encryption: bool = False
    jwt_secret_key: Optional[str] = None
    jwt_expiration_hours: int = 24


class ConfigManager:
    """
    Centralized configuration manager.
    
    Provides access to all configuration settings with environment-specific
    overrides, validation, and runtime updates.
    """
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            environment: Environment name (development, staging, production)
        """
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self._settings = Settings()
        self._config = load_config(environment=self.environment)
        self._initialized = False
        
        # Initialize logging based on configuration
        self._setup_logging()
        
        logger.info(f"ConfigManager initialized for environment: {self.environment}")
        self._initialized = True
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        logging_config = self.get_logging_config()
        
        setup_logging(
            log_level=logging_config.level,
            format_type=logging_config.format,
            enable_correlation_ids=logging_config.enable_correlation_ids,
            log_to_console=logging_config.log_to_console,
            log_to_file=logging_config.log_to_file,
            rotation=logging_config.rotation,
            retention=logging_config.retention,
            enable_performance_logging=logging_config.enable_performance_logging
        )
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT.value
    
    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.environment == Environment.STAGING.value
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION.value
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration setting by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'api.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        db_config = self._config.get("database", {})
        
        return DatabaseConfig(
            url=db_config.get("url", "sqlite:///./fraud_detection.db"),
            echo=db_config.get("echo", False),
            pool_size=db_config.get("pool_size", 5),
            max_overflow=db_config.get("max_overflow", 10),
            pool_timeout=db_config.get("pool_timeout", 30)
        )
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        api_config = self._config.get("api", {})
        
        return APIConfig(
            host=api_config.get("host", self._settings.api_host),
            port=api_config.get("port", self._settings.api_port),
            debug=api_config.get("debug", self._settings.debug),
            workers=api_config.get("workers", 1),
            timeout=api_config.get("timeout", 30),
            enable_cors=api_config.get("enable_cors", False),
            cors_origins=api_config.get("cors_origins", [])
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        logging_config = self._config.get("logging", {})
        
        return LoggingConfig(
            level=logging_config.get("level", self._settings.log_level),
            format=logging_config.get("format", "human"),
            enable_correlation_ids=logging_config.get("enable_correlation_ids", True),
            enable_performance_logging=logging_config.get("enable_performance_logging", False),
            log_to_file=logging_config.get("log_to_file", True),
            log_to_console=logging_config.get("log_to_console", True),
            rotation=logging_config.get("rotation", "10 MB"),
            retention=logging_config.get("retention", "30 days")
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        monitoring_config = self._config.get("monitoring", {})
        
        return MonitoringConfig(
            enable_metrics_collection=monitoring_config.get("enable_metrics_collection", True),
            metrics_interval=monitoring_config.get("metrics_interval", 60),
            enable_profiling=monitoring_config.get("enable_profiling", False),
            enable_tracing=monitoring_config.get("enable_tracing", False),
            trace_sampling_rate=monitoring_config.get("trace_sampling_rate", 0.1),
            metrics_retention_days=monitoring_config.get("metrics_retention_days", 30)
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        security_config = self._config.get("security", {})
        
        return SecurityConfig(
            enable_rate_limiting=security_config.get("enable_rate_limiting", False),
            rate_limit_per_minute=security_config.get("rate_limit_per_minute", 1000),
            enable_authentication=security_config.get("enable_authentication", False),
            enable_encryption=security_config.get("enable_encryption", False),
            jwt_secret_key=security_config.get("jwt_secret_key"),
            jwt_expiration_hours=security_config.get("jwt_expiration_hours", 24)
        )
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        models_config = self._config.get("models", {})
        return models_config.get(model_name, {})
    
    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """
        Get a feature flag value.
        
        Args:
            flag_name: Name of the feature flag
            default: Default value if flag not found
            
        Returns:
            Feature flag value
        """
        features = self._config.get("features", {})
        return features.get(flag_name, default)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration at runtime.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self._config = merge_configs(self._config, updates)
        logger.info(f"Configuration updated: {list(updates.keys())}")
    
    def reload_config(self) -> None:
        """Reload configuration from files."""
        self._config = load_config(environment=self.environment)
        self._setup_logging()
        logger.info("Configuration reloaded")
    
    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate required settings
        required_settings = [
            ("api.host", str),
            ("api.port", int),
            ("logging.level", str)
        ]
        
        for setting_path, expected_type in required_settings:
            value = self.get_setting(setting_path)
            if value is None:
                errors.append(f"Missing required setting: {setting_path}")
            elif not isinstance(value, expected_type):
                errors.append(f"Invalid type for {setting_path}: expected {expected_type.__name__}, got {type(value).__name__}")
        
        # Validate environment-specific requirements
        if self.is_production:
            # Production-specific validations
            if not self.get_setting("security.enable_authentication", False):
                errors.append("Authentication must be enabled in production")
            
            if self.get_setting("logging.level") == "DEBUG":
                errors.append("DEBUG logging should not be used in production")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Configuration summary dictionary
        """
        return {
            "environment": self.environment,
            "debug": self._settings.debug,
            "api": {
                "host": self.get_setting("api.host"),
                "port": self.get_setting("api.port"),
                "workers": self.get_setting("api.workers")
            },
            "logging": {
                "level": self.get_setting("logging.level"),
                "format": self.get_setting("logging.format"),
                "correlation_ids": self.get_setting("logging.enable_correlation_ids")
            },
            "monitoring": {
                "metrics_enabled": self.get_setting("monitoring.enable_metrics_collection"),
                "profiling_enabled": self.get_setting("monitoring.enable_profiling")
            },
            "security": {
                "rate_limiting": self.get_setting("security.enable_rate_limiting"),
                "authentication": self.get_setting("security.enable_authentication")
            }
        }


# Global configuration manager instance
config_manager = ConfigManager()


# Convenience functions
def get_setting(key: str, default: Any = None) -> Any:
    """Get a configuration setting using the global config manager."""
    return config_manager.get_setting(key, default)


def get_feature_flag(flag_name: str, default: bool = False) -> bool:
    """Get a feature flag using the global config manager."""
    return config_manager.get_feature_flag(flag_name, default)


def is_development() -> bool:
    """Check if running in development environment."""
    return config_manager.is_development


def is_production() -> bool:
    """Check if running in production environment."""
    return config_manager.is_production