# Model Deployment and Persistence Guide

This guide covers the model persistence, versioning, and deployment utilities for the fraud detection system.

## Overview

The fraud detection system provides comprehensive model lifecycle management through:

1. **Model Registry**: Centralized model versioning and metadata management
2. **Secure Persistence**: Enhanced model serialization with compression and encryption
3. **Deployment Management**: Automated deployment to multiple environments
4. **CLI Tools**: Command-line interfaces for model and deployment management

## Model Registry

### Features

- **Versioning**: Automatic version management for models
- **Metadata Tracking**: Comprehensive model metadata and performance metrics
- **Status Management**: Model lifecycle status tracking
- **Integrity Verification**: Checksum-based integrity checking
- **Import/Export**: Model portability between environments

### Usage

```python
from src.services.model_registry import ModelRegistry, ModelStatus

# Initialize registry
registry = ModelRegistry()

# Register a trained model
model_id = registry.register_model(
    model=trained_model,
    model_name="fraud_detector_v1",
    description="XGBoost model trained on Q4 2024 data",
    tags=["production", "xgboost", "q4-2024"],
    author="data-science-team",
    performance_metrics={
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90
    }
)

# List models
models = registry.list_models(status=ModelStatus.TRAINED)

# Load a model
model_obj, metadata = registry.load_model(model_id)

# Update model status
registry.update_model_status(model_id, ModelStatus.DEPLOYED)
```

## Secure Model Persistence

### Features

- **Multiple Formats**: Support for joblib, pickle serialization
- **Compression**: GZIP, BZ2, LZMA compression algorithms
- **Encryption**: Optional AES encryption for sensitive models
- **Integrity Checking**: SHA-256 checksums for data verification
- **Format Conversion**: Convert between different serialization formats

### Usage

```python
from src.utils.model_persistence import SecureModelPersistence, CompressionType

# Initialize with encryption
persistence = SecureModelPersistence(
    encryption_key=SecureModelPersistence.generate_encryption_key(),
    default_compression=CompressionType.GZIP
)

# Save model securely
save_info = persistence.save_model_secure(
    model=trained_model,
    filepath="models/fraud_model_encrypted.joblib",
    compression=CompressionType.LZMA,
    encrypt=True
)

# Load model securely
model, metadata = persistence.load_model_secure(
    filepath="models/fraud_model_encrypted.joblib",
    verify_checksum=True
)

# Convert model format
conversion_report = persistence.convert_model_format(
    input_path="models/old_model.pkl",
    output_path="models/new_model.joblib",
    target_format="joblib",
    target_compression=CompressionType.GZIP
)
```

## Model Deployment

### Features

- **Multi-Environment**: Support for development, staging, production
- **Configuration Management**: Environment-specific deployment configurations
- **Health Monitoring**: Automated health checks and monitoring
- **Rollback Support**: Automated rollback capabilities
- **Container Support**: Docker-based deployments

### Deployment Configuration

Create environment-specific configuration files:

```yaml
# config/deployment_templates/production.yaml
environment: production
deployment_name: fraud-detection-prod
api_endpoint: https://api.fraud-detection.com

resource_limits:
  cpu: "2000m"
  memory: "4Gi"
  gpu: 1

scaling_config:
  min_replicas: 3
  max_replicas: 10
  target_cpu_utilization: 60

monitoring_config:
  health_check_interval: 15
  metrics_collection: true
  alert_thresholds:
    error_rate: 0.01
    response_time_p95: 500
```

### Usage

```python
from src.services.model_deployment import ModelDeployment, DeploymentEnvironment

# Initialize deployment manager
deployment = ModelDeployment()

# Create deployment configuration
config = deployment.create_deployment_config(
    model_id="fraud_detector_20241009_123456",
    environment=DeploymentEnvironment.PRODUCTION,
    deployment_name="fraud-detection-prod"
)

# Deploy model
deployment_id = deployment.deploy_model(config)

# Check deployment health
health_status = deployment.check_deployment_health(deployment_id)

# Rollback if needed
rollback_id = deployment.rollback_deployment("production")
```

## Command Line Tools

### Model Management CLI

The `scripts/manage_models.py` script provides comprehensive model management:

```bash
# Register a new model
python scripts/manage_models.py register \
    models/xgboost_model.joblib \
    fraud_detector_v2 \
    XGBoost \
    --description "Improved XGBoost model" \
    --tags "production,xgboost,v2" \
    --metrics '{"accuracy": 0.96, "f1_score": 0.91}'

# List all models
python scripts/manage_models.py list

# Get detailed model information
python scripts/manage_models.py info fraud_detector_XGBoost_20241009_123456

# Update model status
python scripts/manage_models.py update-status fraud_detector_XGBoost_20241009_123456 deployed

# Export model
python scripts/manage_models.py export fraud_detector_XGBoost_20241009_123456 /path/to/export

# Show registry statistics
python scripts/manage_models.py stats

# Clean up old models
python scripts/manage_models.py cleanup models/ --keep-latest 5 --dry-run
```

### Deployment CLI

The `scripts/deploy_model.py` script handles model deployments:

```bash
# Deploy model to production
python scripts/deploy_model.py deploy \
    fraud_detector_XGBoost_20241009_123456 \
    production \
    --config config/deployment_templates/production.yaml \
    --health-check

# List available models
python scripts/deploy_model.py list-models --status trained

# List deployments
python scripts/deploy_model.py list-deployments --environment production

# Check deployment health
python scripts/deploy_model.py health deploy_fraud_detector_production_20241009_140000

# Rollback deployment
python scripts/deploy_model.py rollback production
```

## Environment Setup

### Development Environment

```bash
# Install all dependencies (includes deployment dependencies)
pip install -r requirements.txt

# Set up development configuration
cp config/deployment_templates/development.yaml config/deployment_config.yaml

# Initialize model registry
python -c "from src.services.model_registry import ModelRegistry; ModelRegistry()"
```

### Production Environment

```bash
# Install all dependencies
pip install -r requirements.txt

# Set up production configuration
cp config/deployment_templates/production.yaml config/deployment_config.yaml

# Set environment variables
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export API_HOST=0.0.0.0
export API_PORT=8000

# Initialize services
python scripts/deploy_model.py list-models
```

## Docker Deployment

### Building Container

```bash
# Build Docker image
docker build -t fraud-detection:latest .

# Run container
docker run -d \
    --name fraud-detection-prod \
    -p 8000:8000 \
    -e ENVIRONMENT=production \
    -e MODEL_ID=fraud_detector_XGBoost_20241009_123456 \
    fraud-detection:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  fraud-detection-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - MODEL_ID=fraud_detector_XGBoost_20241009_123456
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Monitoring and Observability

### Health Checks

The deployment system provides comprehensive health monitoring:

- **Container Health**: Docker container status monitoring
- **API Health**: HTTP endpoint health checks
- **Model Health**: Model loading and prediction capability checks
- **Resource Health**: CPU, memory, and disk utilization monitoring

### Metrics Collection

Key metrics collected during deployment:

- **Performance Metrics**: Response time, throughput, error rates
- **Resource Metrics**: CPU usage, memory consumption, disk I/O
- **Business Metrics**: Prediction accuracy, fraud detection rates
- **System Metrics**: Container restarts, deployment success rates

### Alerting

Configurable alert thresholds for:

- High error rates (>1% in production)
- Slow response times (>500ms P95 in production)
- Resource exhaustion (>80% CPU/memory)
- Model performance degradation

## Security Considerations

### Model Security

- **Encryption**: Models can be encrypted at rest using AES encryption
- **Access Control**: Role-based access to model registry and deployments
- **Audit Logging**: Complete audit trail of model operations
- **Integrity Verification**: Checksum verification for model files

### Deployment Security

- **Container Security**: Minimal base images and security scanning
- **Network Security**: HTTPS-only communication in production
- **API Security**: API key authentication and rate limiting
- **Secrets Management**: Secure handling of encryption keys and credentials

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   # Verify model integrity
   python scripts/manage_models.py info MODEL_ID
   
   # Check model file permissions
   ls -la models/
   ```

2. **Deployment Failures**
   ```bash
   # Check deployment logs
   python scripts/deploy_model.py health DEPLOYMENT_ID
   
   # Verify Docker container status
   docker ps -a
   docker logs CONTAINER_NAME
   ```

3. **Performance Issues**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Check API response times
   curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"
   ```

### Log Analysis

Key log locations:
- Application logs: `logs/fraud_detection.log`
- Deployment logs: `deployments/*/logs/`
- Container logs: `docker logs CONTAINER_NAME`

## Best Practices

### Model Management

1. **Version Control**: Always version models with meaningful descriptions
2. **Testing**: Validate models before deployment using staging environment
3. **Backup**: Regular backups of model registry and critical models
4. **Cleanup**: Regular cleanup of old model versions to save storage

### Deployment

1. **Blue-Green Deployment**: Use staging environment for validation
2. **Gradual Rollout**: Use canary deployments for production releases
3. **Monitoring**: Comprehensive monitoring and alerting setup
4. **Rollback Plan**: Always have a tested rollback procedure

### Security

1. **Encryption**: Encrypt sensitive models and use secure communication
2. **Access Control**: Implement proper authentication and authorization
3. **Audit Trail**: Maintain complete audit logs for compliance
4. **Regular Updates**: Keep dependencies and base images updated

## Integration Examples

### CI/CD Pipeline Integration

```yaml
# .github/workflows/deploy.yml
name: Deploy Model
on:
  push:
    tags:
      - 'model-v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Register Model
        run: |
          python scripts/manage_models.py register \
            models/latest_model.joblib \
            fraud_detector_${{ github.ref_name }} \
            XGBoost \
            --tags "ci-cd,${{ github.ref_name }}"
      
      - name: Deploy to Staging
        run: |
          python scripts/deploy_model.py deploy \
            $MODEL_ID \
            staging \
            --config config/deployment_templates/staging.yaml
      
      - name: Run Tests
        run: |
          python -m pytest tests/integration/
      
      - name: Deploy to Production
        if: success()
        run: |
          python scripts/deploy_model.py deploy \
            $MODEL_ID \
            production \
            --config config/deployment_templates/production.yaml
```

This comprehensive model deployment and persistence system provides enterprise-grade capabilities for managing fraud detection models throughout their lifecycle.