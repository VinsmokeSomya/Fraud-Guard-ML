# Configuration Management and Logging System

This document describes the enhanced configuration management and logging system implemented for the fraud detection system.

## Overview

The system provides:
- **Environment-specific configuration management** with automatic environment detection
- **Structured logging with correlation IDs** for request tracking across services
- **Performance metrics collection and monitoring** with automatic aggregation
- **Runtime configuration updates** and validation
- **Thread-safe operations** for concurrent environments

## Configuration Management

### Environment-Specific Configurations

The system supports three environments with automatic configuration loading:

- **Development** (`config/environments/development.yaml`)
- **Staging** (`config/environments/staging.yaml`) 
- **Production** (`config/environments/production.yaml`)

Environment is determined by the `ENVIRONMENT` environment variable (defaults to `development`).

### Configuration Structure

```yaml
# Example configuration structure
environment: development
debug: true

logging:
  level: DEBUG
  format: "detailed"
  enable_correlation_ids: true
  enable_performance_logging: true

api:
  host: "0.0.0.0"
  port: 8000
  workers: 1

monitoring:
  enable_metrics_collection: true
  metrics_interval: 60

security:
  enable_rate_limiting: false
  enable_authentication: false

features:
  enable_batch_processing: true
  enable_real_time_scoring: true
```

### Using Configuration

```python
from src.utils.config_manager import config_manager, get_setting, get_feature_flag

# Get configuration objects
api_config = config_manager.get_api_config()
logging_config = config_manager.get_logging_config()

# Get specific settings with dot notation
fraud_threshold = get_setting("alerts.fraud_threshold", 0.8)

# Check feature flags
if get_feature_flag("enable_batch_processing"):
    # Feature is enabled
    pass

# Runtime configuration updates
config_manager.update_config({
    "api": {"timeout": 60}
})
```

## Structured Logging with Correlation IDs

### Correlation ID Tracking

Correlation IDs automatically track requests across the entire system:

```python
from config.logging_config import get_logger, set_correlation_id, generate_correlation_id
from src.utils.correlation_middleware import CorrelationIdContext

logger = get_logger(__name__)

# Manual correlation ID management
correlation_id = generate_correlation_id()
set_correlation_id(correlation_id)
logger.info("This log will include the correlation ID")

# Using context manager
with CorrelationIdContext() as cid:
    logger.info(f"New correlation ID: {cid}")
    # All logs in this context will have the same correlation ID
```

### Structured Logging

The system supports multiple log formats:

- **Human-readable** (development): Colored console output with correlation IDs
- **Structured** (production): JSON format for log aggregation systems
- **Detailed** (debugging): Extended format with process/thread information

```python
# Structured logging with extra fields
logger.info(
    "Processing fraud detection request",
    extra={
        "transaction_id": "txn_123456789",
        "customer_id": "cust_987654321",
        "amount": 1500.00,
        "risk_factors": {
            "high_amount": True,
            "new_merchant": False
        }
    }
)
```

### FastAPI Middleware Integration

```python
from fastapi import FastAPI
from src.utils.correlation_middleware import CorrelationIdMiddleware

app = FastAPI()

# Add correlation ID middleware
app.add_middleware(
    CorrelationIdMiddleware,
    header_name="X-Correlation-ID",
    generate_if_missing=True,
    enable_performance_tracking=True
)
```

## Performance Metrics Collection

### Automatic Metrics Collection

The system automatically collects:
- Request duration and throughput
- System resource usage (CPU, memory, disk)
- Custom business metrics
- Error rates and patterns

```python
from src.utils.performance_metrics import (
    time_operation, 
    record_metric, 
    increment_counter,
    metrics_collector
)

# Time operations automatically
with time_operation("fraud_scoring"):
    fraud_score = model.predict(transaction)

# Record custom metrics
record_metric("fraud_score", fraud_score, "probability")
increment_counter("transactions_processed", 1, {"type": "transfer"})

# Request-level tracking
correlation_id = metrics_collector.start_request_tracking("/api/fraud/score", "POST")
# ... process request ...
metrics_collector.end_request_tracking(correlation_id, 200, None, {"custom_metric": 123})
```

### Metrics Export and Monitoring

```python
# Get metrics summary
summary = metrics_collector.get_metrics_summary(window_minutes=5)

# Export metrics to file
export_path = metrics_collector.export_metrics()

# Collect system metrics
system_metrics = metrics_collector.collect_system_metrics()
```

## Environment-Specific Features

### Development Environment
- Debug logging enabled
- Auto-reload for code changes
- Detailed error messages
- Performance profiling enabled
- Mock external services

### Staging Environment
- Info-level logging
- Rate limiting enabled
- Authentication required
- Monitoring and alerting
- Load balancing configuration

### Production Environment
- Warning-level logging only
- Full security features enabled
- High availability configuration
- Compliance logging
- Performance optimization

## Configuration Validation

The system includes automatic configuration validation:

```python
# Validate current configuration
validation_errors = config_manager.validate_config()

if validation_errors:
    for error in validation_errors:
        logger.error(f"Configuration error: {error}")
```

Common validation checks:
- Required settings presence
- Data type validation
- Environment-specific requirements
- Security configuration in production
- Resource limit validation

## Best Practices

### Logging Best Practices

1. **Use correlation IDs consistently**:
   ```python
   # Always use correlation ID context for background tasks
   with CorrelationIdContext():
       process_background_task()
   ```

2. **Include relevant context in logs**:
   ```python
   logger.info(
       "Transaction processed",
       extra={
           "transaction_id": transaction_id,
           "processing_time_ms": duration,
           "result": "approved"
       }
   )
   ```

3. **Use appropriate log levels**:
   - `DEBUG`: Detailed diagnostic information
   - `INFO`: General operational messages
   - `WARNING`: Potentially harmful situations
   - `ERROR`: Error events that might still allow the application to continue
   - `CRITICAL`: Serious error events

### Configuration Best Practices

1. **Use environment variables for secrets**:
   ```yaml
   database:
     url: "${DATABASE_URL}"
     password: "${DB_PASSWORD}"
   ```

2. **Validate configuration on startup**:
   ```python
   errors = config_manager.validate_config()
   if errors:
       raise ConfigurationError(f"Invalid configuration: {errors}")
   ```

3. **Use feature flags for gradual rollouts**:
   ```python
   if get_feature_flag("enable_new_fraud_algorithm"):
       result = new_fraud_algorithm(transaction)
   else:
       result = legacy_fraud_algorithm(transaction)
   ```

### Performance Monitoring Best Practices

1. **Monitor key business metrics**:
   ```python
   # Track fraud detection accuracy
   record_metric("fraud_detection_accuracy", accuracy, "percentage")
   
   # Track processing latency
   with time_operation("fraud_detection"):
       result = detect_fraud(transaction)
   ```

2. **Use tags for metric segmentation**:
   ```python
   increment_counter(
       "transactions_processed", 
       1, 
       {"type": transaction_type, "region": user_region}
   )
   ```

3. **Set up alerting thresholds**:
   ```yaml
   monitoring:
     alert_thresholds:
       error_rate: 0.01
       response_time_p95: 500
       cpu_utilization: 70
   ```

## Troubleshooting

### Common Issues

1. **Correlation IDs not appearing in logs**:
   - Ensure `enable_correlation_ids: true` in configuration
   - Check that correlation ID is set before logging
   - Verify middleware is properly configured

2. **Performance metrics not collected**:
   - Check `enable_metrics_collection: true` in configuration
   - Ensure proper import of metrics utilities
   - Verify system permissions for resource monitoring

3. **Configuration not loading**:
   - Check `ENVIRONMENT` environment variable
   - Verify configuration file syntax (YAML)
   - Check file permissions and paths

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export DEBUG=true
```

This will enable:
- Detailed logging output
- Configuration validation messages
- Performance profiling information
- Error stack traces

## Integration Examples

See `examples/configuration_logging_example.py` for comprehensive usage examples demonstrating all features of the configuration and logging system.