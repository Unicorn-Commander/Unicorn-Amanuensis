#!/bin/bash
# Health check for Unicorn-Amanuensis
# Week 6 Production Deployment

set -e

SERVICE_PORT="${SERVICE_PORT:-9050}"
SERVICE_HOST="${SERVICE_HOST:-localhost}"

echo "========================================="
echo " Unicorn-Amanuensis Health Check"
echo "========================================="
echo

# Check if service port is listening
echo "[1/3] Checking service port ${SERVICE_PORT}..."
if timeout 2 bash -c "</dev/tcp/${SERVICE_HOST}/${SERVICE_PORT}" 2>/dev/null; then
    echo "  ✅ Service port is open"
else
    echo "  ❌ Service port not accessible"
    exit 1
fi

# Check health endpoint
echo
echo "[2/3] Checking /health endpoint..."
response=$(curl -s -o /dev/null -w "%{http_code}" "http://${SERVICE_HOST}:${SERVICE_PORT}/health" 2>/dev/null || echo "000")
if [ "$response" = "200" ]; then
    echo "  ✅ Health endpoint OK (HTTP 200)"
else
    echo "  ❌ Health endpoint failed: HTTP $response"
    exit 1
fi

# Get service info
echo
echo "[3/3] Fetching service information..."
service_info=$(curl -s "http://${SERVICE_HOST}:${SERVICE_PORT}/health" 2>/dev/null)
if [ -n "$service_info" ]; then
    echo "  ✅ Service info retrieved"
    echo
    echo "$service_info" | python3 -m json.tool 2>/dev/null || echo "$service_info"
else
    echo "  ⚠️  Could not fetch service info (but service is running)"
fi

echo
echo "========================================="
echo " ✅ All health checks passed"
echo "========================================="
exit 0
