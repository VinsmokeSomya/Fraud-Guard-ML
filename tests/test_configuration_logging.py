"""
Tests for the configuration management and logging system.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.config_manager import ConfigManager, get_setting, get_feature_flag
from config.logging_config import (
    generate_correlation_id, 
    set_correlation_id, 
    get_correlation_id,
    structured_formatter
)
from src.utils.performance_metrics import (
    MetricsCollector,
    record_metric,
    increment_counter,
    time_operation
)


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager(environment="development")
        assert config_manager.environment == "development"
        assert config_manager.is_development
        assert not config_manager.is_production
    
    def test_get_setting_with_dot_notation(self):
        """Test getting settings with dot notation."""
        config_manager = ConfigManager(environment="development")
        
        # Test existing setting
        api_host = config_manager.get_setting("api.host", "localhost")
        assert api_host is not None
        
        # Test non-existing setting with default
        non_existing = config_manager.get_setting("non.existing.setting", "default_value")
        assert non_existing == "default_value"
    
    def test_feature_flags(self):
        """Test feature flag functionality."""
        config_manager = ConfigManager(environment="development")
        
        # Test existing feature flag
        batch_processing = config_manager.get_feature_flag("enable_batch_processing", False)
        assert isinstance(batch_processing, bool)
        
        # Test non-existing feature flag with default
        non_existing_flag = config_manager.get_feature_flag("non_existing_flag", True)
        assert non_existing_flag is True
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager(environment="development")
        
        # Validation should return a list (empty if no errors)
        validation_errors = config_manager.validate_config()
        assert isinstance(validation_errors, list)
    
    def test_runtime_config_update(self):
        """Test runtime configuration updates."""
        config_manager = ConfigManager(environment="development")
        
        # Update configuration
        config_manager.update_config({
            "test": {
                "runtime_setting": "test_value"
            }
        })
        
        # Verify update
        test_value = config_manager.get_setting("test.runtime_setting")
        assert test_value == "test_value"


class TestCorrelationIds:
    """Test cases for correlation ID functionality."""
    
    def test_correlation_id_generation(self):
        """Test correlation ID generation."""
        correlation_id = generate_correlation_id()
        assert correlation_id is not None
        assert len(correlation_id) > 0
        assert isinstance(correlation_id, str)
    
    def test_correlation_id_context(self):
        """Test correlation ID context management."""
        # Generate and set correlation ID
        test_correlation_id = generate_correlation_id()
        set_correlation_id(test_correlation_id)
        
        # Verify it's set correctly
        retrieved_id = get_correlation_id()
        assert retrieved_id == test_correlation_id
    
    def test_structured_formatter(self):
        """Test structured log formatter."""
        # Mock log record
        mock_record = {
            "time": MagicMock(),
            "level": MagicMock(),
            "name": "test_logger",
            "module": "test_module",
            "function": "test_function",
            "line": 123,
            "message": "Test message",
            "process": MagicMock(),
            "thread": MagicMock(),
            "exception": None,
            "extra": {"custom_field": "custom_value"}
        }
        
        # Mock time and level
        mock_record["time"].isoformat.return_value = "2023-01-01T12:00:00"
        mock_record["level"].name = "INFO"
        mock_record["process"].id = 12345
        mock_record["thread"].id = 67890
        
        # Test formatter
        formatted = structured_formatter(mock_record)
        assert isinstance(formatted, str)
        assert "Test message" in formatted
        assert "custom_field" in formatted


class TestPerformanceMetrics:
    """Test cases for performance metrics collection."""
    
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector(max_metrics_in_memory=1000)
        assert collector.max_metrics_in_memory == 1000
    
    def test_record_metric(self):
        """Test recording individual metrics."""
        collector = MetricsCollector()
        
        # Record a metric
        collector.record_metric("test_metric", 42.5, "units")
        
        # Verify it was recorded
        summary = collector.get_metrics_summary(1)
        assert "test_metric" in summary
    
    def test_increment_counter(self):
        """Test counter increment functionality."""
        collector = MetricsCollector()
        
        # Increment counter
        collector.increment_counter("test_counter", 5)
        collector.increment_counter("test_counter", 3)
        
        # Verify counter value
        summary = collector.get_metrics_summary(1)
        assert "counters" in summary
        assert summary["counters"]["test_counter"] == 8
    
    def test_set_gauge(self):
        """Test gauge setting functionality."""
        collector = MetricsCollector()
        
        # Set gauge value
        collector.set_gauge("test_gauge", 123.45)
        
        # Verify gauge value
        summary = collector.get_metrics_summary(1)
        assert "gauges" in summary
        assert summary["gauges"]["test_gauge"] == 123.45
    
    def test_time_operation_context_manager(self):
        """Test timing operation context manager."""
        collector = MetricsCollector()
        
        # Time an operation
        with collector.time_operation("test_operation"):
            # Simulate some work
            pass
        
        # Verify timing was recorded
        summary = collector.get_metrics_summary(1)
        assert "test_operation_duration_ms" in summary
    
    def test_request_tracking(self):
        """Test request-level metrics tracking."""
        collector = MetricsCollector()
        
        # Start request tracking
        correlation_id = collector.start_request_tracking("/api/test", "GET")
        assert correlation_id is not None
        
        # End request tracking
        collector.end_request_tracking(correlation_id, 200, None, {"custom": 123})
        
        # Verify request metrics were recorded
        summary = collector.get_metrics_summary(1)
        assert "request_duration_ms" in summary
        assert "requests_total" in summary["counters"]
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.Process')
    def test_system_metrics_collection(self, mock_process, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0, available=8000000000)
        mock_disk.return_value = MagicMock(percent=70.0, free=100000000000)
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(rss=500000000)
        mock_process_instance.cpu_percent.return_value = 25.0
        mock_process.return_value = mock_process_instance
        
        collector = MetricsCollector(enable_system_metrics=True)
        
        # Collect system metrics
        metrics = collector.collect_system_metrics()
        
        # Verify metrics were collected
        assert "system_cpu_percent" in metrics
        assert "system_memory_percent" in metrics
        assert "process_memory_mb" in metrics
        assert metrics["system_cpu_percent"] == 50.0
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_metric("export_test", 100, "count")
        collector.increment_counter("export_counter", 5)
        
        # Export metrics to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            export_path = collector.export_metrics(tmp_file.name)
            
            # Verify export file was created
            assert Path(export_path).exists()
            
            # Clean up
            os.unlink(export_path)


class TestGlobalFunctions:
    """Test cases for global convenience functions."""
    
    def test_global_get_setting(self):
        """Test global get_setting function."""
        # This should work without errors
        value = get_setting("non.existing.setting", "default")
        assert value == "default"
    
    def test_global_get_feature_flag(self):
        """Test global get_feature_flag function."""
        # This should work without errors
        flag = get_feature_flag("non_existing_flag", True)
        assert flag is True
    
    def test_global_metrics_functions(self):
        """Test global metrics functions."""
        # These should work without errors
        record_metric("global_test", 42, "units")
        increment_counter("global_counter", 1)
        
        # Test time_operation context manager
        with time_operation("global_operation"):
            pass


if __name__ == "__main__":
    pytest.main([__file__])