"""
Example demonstrating the enhanced configuration management and logging system.

This example shows how to use the new configuration management, structured logging
with correlation IDs, and performance metrics collection features.
"""

import asyncio
import time
from typing import Dict, Any
from datetime import datetime

# Import the new configuration and logging utilities
from src.utils.config_manager import config_manager, get_setting, get_feature_flag
from config.logging_config import (
    get_logger, 
    set_correlation_id, 
    generate_correlation_id,
    performance_logger
)
from src.utils.performance_metrics import (
    metrics_collector,
    time_operation,
    record_metric,
    increment_counter,
    set_gauge
)
from src.utils.correlation_middleware import CorrelationIdContext, with_correlation_id

# Initialize logger
logger = get_logger(__name__)
perf_logger = performance_logger()


def demonstrate_configuration_management():
    """Demonstrate the configuration management features."""
    logger.info("=== Configuration Management Demo ===")
    
    # Show current environment
    logger.info(f"Current environment: {config_manager.environment}")
    logger.info(f"Is development: {config_manager.is_development}")
    logger.info(f"Is production: {config_manager.is_production}")
    
    # Get various configuration settings
    api_config = config_manager.get_api_config()
    logger.info(f"API Configuration: host={api_config.host}, port={api_config.port}")
    
    logging_config = config_manager.get_logging_config()
    logger.info(f"Logging Configuration: level={logging_config.level}, correlation_ids={logging_config.enable_correlation_ids}")
    
    monitoring_config = config_manager.get_monitoring_config()
    logger.info(f"Monitoring Configuration: metrics_enabled={monitoring_config.enable_metrics_collection}")
    
    # Get specific settings using dot notation
    fraud_threshold = get_setting("alerts.fraud_threshold", 0.8)
    logger.info(f"Fraud threshold: {fraud_threshold}")
    
    # Check feature flags
    batch_processing_enabled = get_feature_flag("enable_batch_processing", False)
    logger.info(f"Batch processing enabled: {batch_processing_enabled}")
    
    # Get model-specific configuration
    xgboost_config = config_manager.get_model_config("xgboost")
    logger.info(f"XGBoost config: {xgboost_config}")
    
    # Show configuration summary
    config_summary = config_manager.get_config_summary()
    logger.info(f"Configuration summary: {config_summary}")


def demonstrate_correlation_ids():
    """Demonstrate correlation ID tracking."""
    logger.info("=== Correlation ID Demo ===")
    
    # Generate and set a correlation ID
    correlation_id = generate_correlation_id()
    set_correlation_id(correlation_id)
    
    logger.info("This log message will include the correlation ID")
    logger.info("All subsequent logs in this context will have the same correlation ID")
    
    # Demonstrate nested operations with the same correlation ID
    simulate_nested_operations()
    
    # Demonstrate using correlation ID context manager
    with CorrelationIdContext() as new_correlation_id:
        logger.info(f"Inside context manager with new correlation ID: {new_correlation_id}")
        simulate_database_operation()
        simulate_external_api_call()
    
    logger.info("Back to original correlation ID context")


def simulate_nested_operations():
    """Simulate nested operations that maintain correlation ID."""
    logger.info("Starting nested operation 1")
    time.sleep(0.1)  # Simulate work
    
    logger.info("Starting nested operation 2")
    time.sleep(0.05)  # Simulate work
    
    logger.info("Nested operations completed")


def simulate_database_operation():
    """Simulate a database operation with logging."""
    logger.info("Starting database operation")
    time.sleep(0.2)  # Simulate database query
    logger.info("Database operation completed successfully")


def simulate_external_api_call():
    """Simulate an external API call with logging."""
    logger.info("Making external API call")
    time.sleep(0.3)  # Simulate API call
    logger.info("External API call completed")


def demonstrate_performance_metrics():
    """Demonstrate performance metrics collection."""
    logger.info("=== Performance Metrics Demo ===")
    
    # Record simple metrics
    record_metric("demo_counter", 1, "count")
    record_metric("demo_gauge", 42.5, "units")
    
    # Increment counters
    increment_counter("demo_requests", 1, {"endpoint": "/api/fraud/score"})
    increment_counter("demo_requests", 1, {"endpoint": "/api/fraud/batch"})
    
    # Set gauge values
    set_gauge("demo_cpu_usage", 65.2, {"host": "fraud-detector-1"})
    set_gauge("demo_memory_usage", 1024.5, {"host": "fraud-detector-1"})
    
    # Demonstrate timing operations
    with time_operation("demo_fraud_scoring"):
        simulate_fraud_scoring()
    
    with time_operation("demo_batch_processing", {"batch_size": "100"}):
        simulate_batch_processing()
    
    # Demonstrate request tracking
    correlation_id = metrics_collector.start_request_tracking("/api/fraud/score", "POST")
    time.sleep(0.1)  # Simulate request processing
    metrics_collector.end_request_tracking(
        correlation_id, 
        200, 
        None, 
        {"custom_metric": 123.45}
    )
    
    # Collect system metrics
    system_metrics = metrics_collector.collect_system_metrics()
    logger.info(f"System metrics collected: {len(system_metrics)} metrics")
    
    # Get metrics summary
    summary = metrics_collector.get_metrics_summary(5)  # Last 5 minutes
    logger.info(f"Metrics summary: {len(summary)} metric types")


def simulate_fraud_scoring():
    """Simulate fraud scoring operation."""
    logger.info("Performing fraud scoring")
    time.sleep(0.05)  # Simulate model inference
    
    # Record fraud-specific metrics
    fraud_score = 0.75
    record_metric("fraud_score", fraud_score, "probability")
    
    if fraud_score > 0.5:
        increment_counter("fraud_detected", 1)
    
    logger.info(f"Fraud scoring completed with score: {fraud_score}")


def simulate_batch_processing():
    """Simulate batch processing operation."""
    logger.info("Starting batch processing")
    
    batch_size = 100
    for i in range(batch_size):
        if i % 20 == 0:  # Log progress every 20 items
            logger.debug(f"Processed {i}/{batch_size} items")
        time.sleep(0.001)  # Simulate processing time
    
    increment_counter("batch_items_processed", batch_size)
    logger.info(f"Batch processing completed: {batch_size} items")


@with_correlation_id()
def demonstrate_decorator():
    """Demonstrate the correlation ID decorator."""
    logger.info("=== Correlation ID Decorator Demo ===")
    logger.info("This function is decorated with @with_correlation_id()")
    logger.info("A correlation ID was automatically generated and set")
    
    # Simulate some work
    with time_operation("decorated_function_work"):
        time.sleep(0.1)
    
    logger.info("Decorated function completed")


def demonstrate_structured_logging():
    """Demonstrate structured logging with extra fields."""
    logger.info("=== Structured Logging Demo ===")
    
    # Log with extra structured data
    logger.info(
        "Processing fraud detection request",
        extra={
            "transaction_id": "txn_123456789",
            "customer_id": "cust_987654321",
            "amount": 1500.00,
            "currency": "USD",
            "merchant": "Online Store XYZ",
            "risk_factors": {
                "high_amount": True,
                "new_merchant": False,
                "unusual_time": False
            }
        }
    )
    
    # Log an error with context
    try:
        # Simulate an error
        raise ValueError("Simulated error for demonstration")
    except Exception as e:
        logger.error(
            "Error occurred during fraud detection",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "transaction_id": "txn_123456789",
                "recovery_action": "fallback_to_rule_based_detection"
            }
        )
    
    # Performance logging
    perf_logger.info(
        "Performance measurement",
        extra={
            "operation": "model_inference",
            "duration_ms": 45.2,
            "cpu_usage": 23.5,
            "memory_mb": 512.3,
            "model_name": "xgboost_v2.1"
        }
    )


def demonstrate_configuration_validation():
    """Demonstrate configuration validation."""
    logger.info("=== Configuration Validation Demo ===")
    
    # Validate current configuration
    validation_errors = config_manager.validate_config()
    
    if validation_errors:
        logger.warning(f"Configuration validation found {len(validation_errors)} errors:")
        for error in validation_errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("Configuration validation passed - no errors found")
    
    # Demonstrate runtime configuration updates
    logger.info("Demonstrating runtime configuration update")
    config_manager.update_config({
        "demo": {
            "runtime_setting": "updated_value",
            "timestamp": datetime.now().isoformat()
        }
    })
    
    # Verify the update
    updated_value = get_setting("demo.runtime_setting")
    logger.info(f"Updated configuration value: {updated_value}")


async def demonstrate_async_correlation_ids():
    """Demonstrate correlation ID tracking in async operations."""
    logger.info("=== Async Correlation ID Demo ===")
    
    # Set correlation ID for async context
    correlation_id = generate_correlation_id()
    set_correlation_id(correlation_id)
    
    logger.info("Starting async operations")
    
    # Simulate concurrent async operations
    tasks = [
        simulate_async_operation("operation_1", 0.1),
        simulate_async_operation("operation_2", 0.15),
        simulate_async_operation("operation_3", 0.08)
    ]
    
    results = await asyncio.gather(*tasks)
    logger.info(f"All async operations completed: {results}")


async def simulate_async_operation(operation_name: str, delay: float) -> str:
    """Simulate an async operation with logging."""
    logger.info(f"Starting {operation_name}")
    await asyncio.sleep(delay)
    logger.info(f"Completed {operation_name}")
    return f"{operation_name}_result"


def main():
    """Main demonstration function."""
    logger.info("Starting Configuration and Logging System Demonstration")
    
    # Demonstrate all features
    demonstrate_configuration_management()
    demonstrate_correlation_ids()
    demonstrate_performance_metrics()
    demonstrate_decorator()
    demonstrate_structured_logging()
    demonstrate_configuration_validation()
    
    # Demonstrate async features
    asyncio.run(demonstrate_async_correlation_ids())
    
    # Export metrics at the end
    export_path = metrics_collector.export_metrics()
    logger.info(f"Metrics exported to: {export_path}")
    
    logger.info("Configuration and Logging System Demonstration completed")


if __name__ == "__main__":
    main()