# Fraud Detection System - Docker Deployment Guide

This guide covers the containerized deployment of the Fraud Detection System using Docker and Docker Compose.

## Quick Start

### Development Environment

```bash
# Start development environment
make dev

# Or manually
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Production Environment

```bash
# Start production environment
make prod

# Or manually
docker-compose --env-file docker/env/production.env up -d
```

## Architecture Overview

The system consists of the following services:

- **fraud-api**: FastAPI service for fraud detection
- **fraud-dashboard**: Streamlit dashboard for analysis
- **redis**: Caching and session management
- **nginx**: Reverse proxy (optional)
- **prometheus**: Metrics collection (optional)
- **grafana**: Monitoring dashboard (optional)

## Service Configuration

### Environment Variables

#### Core Settings
- `ENVIRONMENT`: deployment environment (development/production)
- `DEBUG`: enable debug mode (true/false)
- `LOG_LEVEL`: logging level (DEBUG/INFO/WARNING/ERROR)

#### API Settings
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `MODEL_PATH`: path to trained model file
- `RISK_THRESHOLD`: fraud classification threshold (0-1)
- `HIGH_RISK_THRESHOLD`: high-risk alert threshold (0-1)

#### Alert Settings
- `ENABLE_ALERTS`: enable alert system (true/false)
- `ALERT_EMAIL_*`: email notification configuration

### Volume Mounts

- `./data`: Transaction data files (read-only)
- `./models`: Trained model files (read-only)
- `./logs`: Application logs (read-write)
- `./config`: Configuration files (read-only)
- `./monitoring_data`: Monitoring data (read-write)
- `./reports`: Generated reports (read-write)

## Deployment Scenarios

### 1. Development Setup

```bash
# Build and start development environment
make dev-build

# Start Jupyter notebook for experimentation
make jupyter

# View logs
make logs

# Run tests
make test
```

**Services Available:**
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Jupyter: http://localhost:8888
- Redis: localhost:6379

### 2. Production Setup

```bash
# Build and start production environment
make prod-build

# Check service health
make health

# View production logs
make logs
```

**Services Available:**
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Redis: localhost:6379

### 3. Full Monitoring Stack

```bash
# Start with monitoring
make monitoring

# Start with reverse proxy
make nginx
```

**Additional Services:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)
- Nginx: http://localhost:80

## Configuration Files

### Environment-Specific Configs

- `docker/env/development.env`: Development settings
- `docker/env/production.env`: Production settings

### Service Configs

- `docker/nginx/nginx.conf`: Nginx reverse proxy configuration
- `docker/prometheus/prometheus.yml`: Prometheus monitoring configuration
- `docker/grafana/provisioning/`: Grafana dashboard provisioning

## Health Checks

All services include health checks:

```bash
# Check all service health
docker-compose ps

# Check API health specifically
curl http://localhost:8000/health

# Check dashboard health
curl http://localhost:8501/_stcore/health
```

## Scaling

### Horizontal Scaling

```bash
# Scale API service to 3 replicas
docker-compose up -d --scale fraud-api=3

# Scale with load balancer
docker-compose --profile nginx up -d --scale fraud-api=3
```

### Resource Limits

Add resource limits in docker-compose.yml:

```yaml
services:
  fraud-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## Security Considerations

### Network Security

- Services communicate through isolated Docker network
- Only necessary ports are exposed to host
- Nginx provides additional security layer

### Data Security

- Sensitive data mounted as read-only volumes
- Environment variables for secrets (use Docker secrets in production)
- Non-root user in containers

### Production Hardening

```bash
# Use production environment file
cp docker/env/production.env .env

# Update secrets
vim .env  # Update SECRET_KEY, JWT_SECRET_KEY, passwords

# Use Docker secrets (recommended)
echo "your-secret-key" | docker secret create fraud_secret_key -
```

## Monitoring and Logging

### Application Logs

```bash
# View all logs
make logs

# View specific service logs
make logs-api
make logs-dashboard

# Follow logs in real-time
docker-compose logs -f fraud-api
```

### Metrics and Monitoring

With monitoring stack enabled:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Alert Rules**: Automated alerting for issues

### Log Aggregation

For production, consider external log aggregation:

```yaml
services:
  fraud-api:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: "localhost:24224"
        tag: "fraud-api"
```

## Backup and Recovery

### Data Backup

```bash
# Backup Redis data
make backup

# Backup models and configuration
tar -czf backup-$(date +%Y%m%d).tar.gz models/ config/ docker/env/
```

### Disaster Recovery

```bash
# Restore from backup
make restore BACKUP_FILE=redis-backup-20231009-143022.tar.gz

# Rebuild and restart services
make clean
make prod-build
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Use different ports
   docker-compose up -d -p 8001:8000
   ```

2. **Memory Issues**
   ```bash
   # Check container resource usage
   docker stats
   
   # Increase memory limits
   # Edit docker-compose.yml deploy.resources.limits.memory
   ```

3. **Model Loading Failures**
   ```bash
   # Check model file exists
   docker-compose exec fraud-api ls -la /app/models/
   
   # Check model path in environment
   docker-compose exec fraud-api env | grep MODEL_PATH
   ```

### Debug Mode

```bash
# Start in debug mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access container shell
make shell

# Check service status
make health
```

### Performance Tuning

1. **API Performance**
   - Adjust worker count in production
   - Enable Redis caching
   - Optimize model loading

2. **Dashboard Performance**
   - Limit data visualization size
   - Use data sampling for large datasets
   - Enable caching

3. **Database Performance**
   - Add database connection pooling
   - Optimize queries
   - Use read replicas

## Production Deployment Checklist

- [ ] Update environment variables in production.env
- [ ] Configure proper secrets management
- [ ] Set up SSL certificates for HTTPS
- [ ] Configure log aggregation
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategy
- [ ] Test disaster recovery procedures
- [ ] Set up CI/CD pipeline
- [ ] Configure resource limits
- [ ] Enable security scanning

## Support

For issues and questions:

1. Check service logs: `make logs`
2. Verify service health: `make health`
3. Review configuration files
4. Check Docker and Docker Compose versions
5. Consult main project documentation