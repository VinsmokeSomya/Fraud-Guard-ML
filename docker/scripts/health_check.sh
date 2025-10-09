#!/bin/bash

# Health check script for fraud detection services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:8000"
DASHBOARD_URL="http://localhost:8501"
REDIS_HOST="localhost"
REDIS_PORT="6379"

echo "🔍 Fraud Detection System Health Check"
echo "======================================"

# Check API Health
echo -n "API Service: "
if curl -s -f "${API_URL}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
    API_STATUS=$(curl -s "${API_URL}/health" | jq -r '.status' 2>/dev/null || echo "unknown")
    echo "  Status: ${API_STATUS}"
else
    echo -e "${RED}✗ Unhealthy${NC}"
    API_HEALTHY=false
fi

# Check Dashboard Health
echo -n "Dashboard Service: "
if curl -s -f "${DASHBOARD_URL}/_stcore/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${RED}✗ Unhealthy${NC}"
    DASHBOARD_HEALTHY=false
fi

# Check Redis Health
echo -n "Redis Service: "
if command -v redis-cli > /dev/null 2>&1; then
    if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
        REDIS_INFO=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
        echo "  Memory Usage: ${REDIS_INFO}"
    else
        echo -e "${RED}✗ Unhealthy${NC}"
        REDIS_HEALTHY=false
    fi
else
    echo -e "${YELLOW}? Redis CLI not available${NC}"
fi

# Check Docker Services
echo ""
echo "Docker Services Status:"
echo "======================"
docker-compose ps --format "table {{.Name}}\t{{.State}}\t{{.Status}}"

# Check Model Status
echo ""
echo "Model Information:"
echo "=================="
MODEL_INFO=$(curl -s "${API_URL}/status" 2>/dev/null | jq -r '.model_info // "No model info available"' 2>/dev/null || echo "API not available")
echo "Model Status: ${MODEL_INFO}"

# Check Disk Space
echo ""
echo "System Resources:"
echo "================="
echo "Disk Usage:"
df -h | grep -E "(Filesystem|/dev/)"

echo ""
echo "Memory Usage:"
free -h

# Summary
echo ""
echo "Health Check Summary:"
echo "===================="

if [[ "${API_HEALTHY}" != "false" && "${DASHBOARD_HEALTHY}" != "false" && "${REDIS_HEALTHY}" != "false" ]]; then
    echo -e "${GREEN}✓ All services are healthy${NC}"
    exit 0
else
    echo -e "${RED}✗ Some services are unhealthy${NC}"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check service logs: make logs"
    echo "2. Restart services: make restart"
    echo "3. Check configuration files"
    echo "4. Verify model files are present"
    exit 1
fi