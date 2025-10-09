# Fraud Detection System Docker Management

.PHONY: help build up down logs clean test dev prod monitoring

# Default target
help:
	@echo "Fraud Detection System Docker Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development environment"
	@echo "  make dev-build    - Build and start development environment"
	@echo "  make jupyter      - Start Jupyter notebook service"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start production environment"
	@echo "  make prod-build   - Build and start production environment"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitoring   - Start with monitoring stack (Prometheus + Grafana)"
	@echo "  make nginx        - Start with Nginx reverse proxy"
	@echo ""
	@echo "Management:"
	@echo "  make build        - Build all images"
	@echo "  make up           - Start services"
	@echo "  make down         - Stop services"
	@echo "  make restart      - Restart services"
	@echo "  make logs         - Show logs"
	@echo "  make clean        - Clean up containers and volumes"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run tests in container"
	@echo "  make lint         - Run linting"
	@echo ""
	@echo "Utilities:"
	@echo "  make shell        - Open shell in API container"
	@echo "  make health       - Check service health"

# Development environment
dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

dev-build:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

jupyter:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d jupyter

# Production environment
prod:
	docker-compose --env-file docker/env/production.env up -d

prod-build:
	docker-compose --env-file docker/env/production.env up -d --build

# Monitoring stack
monitoring:
	docker-compose --profile monitoring up -d

# Nginx reverse proxy
nginx:
	docker-compose --profile nginx up -d

# Basic operations
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

# Specific service logs
logs-api:
	docker-compose logs -f fraud-api

logs-dashboard:
	docker-compose logs -f fraud-dashboard

logs-redis:
	docker-compose logs -f redis

# Testing
test:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm fraud-api pytest tests/ -v

lint:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm fraud-api black --check src/
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm fraud-api flake8 src/

# Utilities
shell:
	docker-compose exec fraud-api /bin/bash

shell-dashboard:
	docker-compose exec fraud-dashboard /bin/bash

health:
	@echo "Checking API health..."
	@curl -s http://localhost:8000/health | jq '.' || echo "API not responding"
	@echo ""
	@echo "Checking Dashboard health..."
	@curl -s http://localhost:8501/_stcore/health || echo "Dashboard not responding"

# Cleanup
clean:
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-all:
	docker-compose down -v --remove-orphans
	docker system prune -af
	docker volume prune -f

# Database operations (if needed)
db-migrate:
	docker-compose exec fraud-api alembic upgrade head

db-reset:
	docker-compose exec fraud-api alembic downgrade base
	docker-compose exec fraud-api alembic upgrade head

# Model management
model-train:
	docker-compose exec fraud-api python scripts/manage_models.py train

model-deploy:
	docker-compose exec fraud-api python scripts/deploy_model.py

# Backup and restore
backup:
	mkdir -p backups
	docker run --rm -v fraud-detection_redis_data:/data -v $(PWD)/backups:/backup alpine tar czf /backup/redis-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .

restore:
	@echo "Usage: make restore BACKUP_FILE=redis-backup-YYYYMMDD-HHMMSS.tar.gz"
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Please specify BACKUP_FILE"; exit 1; fi
	docker run --rm -v fraud-detection_redis_data:/data -v $(PWD)/backups:/backup alpine tar xzf /backup/$(BACKUP_FILE) -C /data

# Security scan
security-scan:
	docker run --rm -v $(PWD):/app -w /app securecodewarrior/docker-security-scanner

# Performance testing
load-test:
	docker run --rm -i loadimpact/k6 run --vus 10 --duration 30s - < tests/load_test.js