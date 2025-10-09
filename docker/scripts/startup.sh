#!/bin/bash

# Startup script for fraud detection containers

set -e

echo "🚀 Starting Fraud Detection System..."

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for Redis
if [ -n "${REDIS_URL}" ]; then
    echo "Waiting for Redis..."
    while ! redis-cli -u "${REDIS_URL}" ping > /dev/null 2>&1; do
        echo "Redis is unavailable - sleeping"
        sleep 2
    done
    echo "✓ Redis is ready"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /app/logs /app/data/processed /app/models /app/reports /app/monitoring_data

# Set permissions
chown -R appuser:appuser /app/logs /app/data/processed /app/reports /app/monitoring_data

# Check if model exists
if [ -n "${MODEL_PATH}" ] && [ -f "${MODEL_PATH}" ]; then
    echo "✓ Model found at ${MODEL_PATH}"
else
    echo "⚠️  No model found at ${MODEL_PATH:-/app/models/}"
    echo "   The service will start but predictions may fail"
fi

# Run database migrations if needed
if [ -n "${DATABASE_URL}" ]; then
    echo "🗄️  Running database migrations..."
    alembic upgrade head || echo "⚠️  Migration failed or not configured"
fi

echo "✅ Startup complete!"

# Execute the main command
exec "$@"