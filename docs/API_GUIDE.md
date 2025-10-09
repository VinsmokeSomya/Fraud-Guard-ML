# Fraud Detection API Guide

## Overview

The Fraud Detection API provides a comprehensive REST interface for real-time fraud detection and risk assessment of financial transactions. Built with FastAPI, it offers high performance, automatic documentation, and robust error handling.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Formats](#requestresponse-formats)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Examples](#examples)
8. [SDKs and Libraries](#sdks-and-libraries)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### Starting the API Server

```bash
# Basic startup
python run_api.py

# With custom configuration
python run_api.py --host 0.0.0.0 --port 8080 --model-path models/fraud_model.joblib

# With custom thresholds
python run_api.py --risk-threshold 0.6 --high-risk-threshold 0.85
```

### API Base Information

- **Base URL**: `http://localhost:8000` (default)
- **Content Type**: `application/json`
- **Documentation**: Available at `/docs` (Swagger) and `/redoc`
- **Health Check**: Available at `/health`

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Server host address |
| `--port` | 8000 | Server port |
| `--model-path` | None | Path to trained fraud detection model |
| `--risk-threshold` | 0.5 | Risk classification threshold (0-1) |
| `--high-risk-threshold` | 0.8 | High-risk alert threshold (0-1) |
| `--no-explanations` | False | Disable detailed fraud explanations |
| `--reload` | False | Enable auto-reload for development |
| `--log-level` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Authentication

**Current Version**: No authentication required (suitable for internal use)

**Production Considerations**: For production deployment, consider implementing:
- API key authentication
- JWT tokens
- OAuth2 integration
- Rate limiting per user/API key

## API Endpoints

### Core Endpoints

#### `GET /` - Root Information
Returns basic API information and available endpoints.

**Response:**
```json
{
  "service": "Fraud Detection API",
  "version": "1.0.0", 
  "status": "active",
  "docs": "/docs",
  "health": "/health"
}
```

#### `GET /health` - Health Check
Check service status and model availability.

**Response:**
```json
{
  "service_name": "FraudDetector",
  "status": "active",
  "model_loaded": true,
  "risk_threshold": 0.5,
  "high_risk_threshold": 0.8,
  "explanations_enabled": true,
  "model_info": {
    "model_type": "XGBoostModel",
    "training_date": "2024-01-01T12:00:00",
    "feature_count": 15
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Prediction Endpoints

#### `POST /predict` - Single Transaction Scoring

Score a single transaction for fraud risk.

**Request Body:**
```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 181.0,
  "nameOrig": "C1231006815",
  "oldbalanceOrg": 181.0,
  "newbalanceOrig": 0.0,
  "nameDest": "C1666544295",
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0
}
```

**Response:**
```json
{
  "fraud_score": 0.85,
  "risk_level": "HIGH",
  "is_fraud_prediction": true,
  "confidence": 0.7,
  "processed_at": "2024-01-01T12:00:00"
}
```

#### `POST /predict/explain` - Detailed Fraud Analysis

Get comprehensive fraud analysis with explanations and risk factors.

**Request Body:** Same as `/predict`

**Response:**
```json
{
  "fraud_score": 0.85,
  "risk_level": "HIGH", 
  "is_fraud_prediction": true,
  "confidence": 0.7,
  "risk_factors": {
    "high_risk_factors": [
      "Complete balance depletion detected",
      "High-value transfer transaction"
    ],
    "medium_risk_factors": [
      "High-risk transaction type: TRANSFER"
    ],
    "low_risk_factors": [],
    "risk_scores": {
      "transaction_type": 0.7,
      "amount": 0.6,
      "balance_patterns": 0.9
    }
  },
  "feature_importance": {
    "amount": 0.25,
    "balance_change_orig": 0.20,
    "type_TRANSFER": 0.18,
    "oldbalanceOrg": 0.15
  },
  "explanation_text": "HIGH RISK: This transaction has a 85.0% probability of being fraudulent. Key risk factors include complete balance depletion and high-value transfer pattern.",
  "recommendations": [
    "IMMEDIATE ACTION REQUIRED: Block transaction and investigate",
    "Contact customer to verify transaction legitimacy",
    "Review recent account activity for suspicious patterns"
  ],
  "processed_at": "2024-01-01T12:00:00"
}
```

#### `POST /predict/batch` - Batch Processing

Process multiple transactions simultaneously (max 1000 per request).

**Request Body:**
```json
{
  "transactions": [
    {
      "step": 1,
      "type": "PAYMENT",
      "amount": 9839.64,
      "nameOrig": "C1231006815",
      "oldbalanceOrg": 170136.0,
      "newbalanceOrig": 160296.36,
      "nameDest": "M1979787155",
      "oldbalanceDest": 0.0,
      "newbalanceDest": 0.0
    },
    {
      "step": 1,
      "type": "TRANSFER", 
      "amount": 181.0,
      "nameOrig": "C1231006815",
      "oldbalanceOrg": 181.0,
      "newbalanceOrig": 0.0,
      "nameDest": "C1666544295",
      "oldbalanceDest": 0.0,
      "newbalanceDest": 0.0
    }
  ]
}
```

**Response:**
```json
{
  "total_transactions": 2,
  "fraud_detected": 1,
  "high_risk_count": 1,
  "medium_risk_count": 0,
  "low_risk_count": 1,
  "results": [
    {
      "fraud_score": 0.15,
      "risk_level": "LOW",
      "is_fraud_prediction": false,
      "confidence": 0.8,
      "processed_at": "2024-01-01T12:00:00"
    },
    {
      "fraud_score": 0.85,
      "risk_level": "HIGH",
      "is_fraud_prediction": true,
      "confidence": 0.7,
      "processed_at": "2024-01-01T12:00:00"
    }
  ],
  "processed_at": "2024-01-01T12:00:00"
}
```

### Configuration Endpoints

#### `PUT /config/thresholds` - Update Detection Thresholds

Adjust fraud detection sensitivity by updating thresholds.

**Request Body:**
```json
{
  "risk_threshold": 0.6,
  "high_risk_threshold": 0.85
}
```

**Response:**
```json
{
  "message": "Thresholds updated successfully",
  "risk_threshold": 0.6,
  "high_risk_threshold": 0.85,
  "updated_at": "2024-01-01T12:00:00"
}
```

#### `GET /status` - Detailed Service Status

Get comprehensive service status and configuration.

**Response:**
```json
{
  "service_name": "FraudDetector",
  "status": "active",
  "model_loaded": true,
  "risk_threshold": 0.5,
  "high_risk_threshold": 0.8,
  "explanations_enabled": true,
  "model_info": {
    "model_type": "XGBoostModel",
    "training_date": "2024-01-01T12:00:00",
    "feature_count": 15,
    "accuracy": 0.95
  },
  "api_version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "predict_explain": "/predict/explain",
    "batch_predict": "/predict/batch",
    "update_thresholds": "/config/thresholds",
    "status": "/status"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Alert Management Endpoints

#### `GET /alerts` - Get Active Alerts

Retrieve list of active fraud alerts.

**Query Parameters:**
- `severity` (optional): Filter by severity (LOW, MEDIUM, HIGH, CRITICAL)

**Response:**
```json
[
  {
    "alert_id": "alert_20240101_001",
    "fraud_score": 0.92,
    "severity": "HIGH",
    "risk_level": "HIGH",
    "transaction_data": {
      "step": 1,
      "type": "TRANSFER",
      "amount": 500000.0
    },
    "explanation": "High-risk transfer with complete balance depletion",
    "recommendations": [
      "Block transaction immediately",
      "Contact customer for verification"
    ],
    "created_at": "2024-01-01T12:00:00",
    "status": "ACTIVE",
    "acknowledged_by": null
  }
]
```

#### `GET /alerts/history` - Get Alert History

Retrieve alert history for specified time period.

**Query Parameters:**
- `hours_back` (default: 24): Number of hours to look back (1-168)
- `severity` (optional): Filter by severity level

#### `POST /alerts/{alert_id}/acknowledge` - Acknowledge Alert

Acknowledge a specific fraud alert.

**Request Body:**
```json
{
  "acknowledged_by": "fraud_analyst_001"
}
```

#### `GET /alerts/statistics` - Get Alert Statistics

Retrieve alert statistics and metrics.

**Response:**
```json
{
  "total_alerts": 150,
  "active_alerts": 5,
  "acknowledged_alerts": 145,
  "alerts_by_severity": {
    "LOW": 50,
    "MEDIUM": 75,
    "HIGH": 20,
    "CRITICAL": 5
  },
  "notifications_sent": 148,
  "notifications_failed": 2,
  "notification_success_rate": 0.987
}
```

## Request/Response Formats

### Transaction Data Format

All transaction requests must include these fields:

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `step` | int | Yes | Time step (0-744) | 1 |
| `type` | string | Yes | Transaction type | "TRANSFER" |
| `amount` | float | Yes | Transaction amount (≥0) | 181.0 |
| `nameOrig` | string | Yes | Origin customer ID | "C1231006815" |
| `oldbalanceOrg` | float | Yes | Origin balance before (≥0) | 181.0 |
| `newbalanceOrig` | float | Yes | Origin balance after (≥0) | 0.0 |
| `nameDest` | string | Yes | Destination customer ID | "C1666544295" |
| `oldbalanceDest` | float | Yes | Destination balance before (≥0) | 0.0 |
| `newbalanceDest` | float | Yes | Destination balance after (≥0) | 0.0 |

### Valid Transaction Types

- `CASH-IN`: Cash deposit to account
- `CASH-OUT`: Cash withdrawal from account  
- `DEBIT`: Debit card transaction
- `PAYMENT`: Payment to merchant
- `TRANSFER`: Transfer between accounts

### Risk Levels

Transactions are categorized into three risk levels:

- **LOW**: `fraud_score < risk_threshold` (default: < 0.5)
- **MEDIUM**: `risk_threshold ≤ fraud_score < high_risk_threshold` (default: 0.5-0.8)  
- **HIGH**: `fraud_score ≥ high_risk_threshold` (default: ≥ 0.8)

### Response Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid input data or parameters |
| 404 | Not Found | Endpoint or resource not found |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server-side processing error |
| 503 | Service Unavailable | Service not initialized or model not loaded |

## Error Handling

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Common Error Scenarios

#### Invalid Transaction Type
```json
{
  "detail": "Invalid transaction data: Transaction type must be one of: ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']"
}
```

#### Missing Required Fields
```json
{
  "detail": "Field required: amount"
}
```

#### Service Not Initialized
```json
{
  "detail": "Fraud detection service is not initialized. Please check service health."
}
```

#### Model Not Loaded
```json
{
  "detail": "Fraud detection model is not loaded. Please check model configuration."
}
```

## Rate Limiting

**Current Version**: No rate limiting implemented

**Production Recommendations**:
- Implement rate limiting per IP/API key
- Suggested limits:
  - Single predictions: 1000 requests/minute
  - Batch predictions: 100 requests/minute  
  - Configuration changes: 10 requests/minute

## Examples

### Python SDK Example

```python
import requests
from typing import Dict, List

class FraudDetectionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self) -> Dict:
        """Check service health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_single(self, transaction: Dict) -> Dict:
        """Score a single transaction."""
        response = requests.post(
            f"{self.base_url}/predict",
            json=transaction
        )
        response.raise_for_status()
        return response.json()
    
    def predict_with_explanation(self, transaction: Dict) -> Dict:
        """Get detailed fraud analysis."""
        response = requests.post(
            f"{self.base_url}/predict/explain",
            json=transaction
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, transactions: List[Dict]) -> Dict:
        """Process multiple transactions."""
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json={"transactions": transactions}
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = FraudDetectionClient()

# Check service health
health = client.health_check()
print(f"Service Status: {health['status']}")

# Score a transaction
transaction = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "nameOrig": "C1231006815", 
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C1666544295",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}

result = client.predict_single(transaction)
print(f"Fraud Score: {result['fraud_score']}")
print(f"Risk Level: {result['risk_level']}")

# Get detailed explanation
explanation = client.predict_with_explanation(transaction)
print(f"Risk Factors: {explanation['risk_factors']}")
print(f"Recommendations: {explanation['recommendations']}")
```

### cURL Examples

#### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C1666544295",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
  }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "nameDest": "M1979787155",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
      }
    ]
  }'
```

#### Update Thresholds
```bash
curl -X PUT "http://localhost:8000/config/thresholds" \
  -H "Content-Type: application/json" \
  -d '{
    "risk_threshold": 0.6,
    "high_risk_threshold": 0.85
  }'
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

class FraudDetectionClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }
    
    async healthCheck() {
        const response = await this.client.get('/health');
        return response.data;
    }
    
    async predictSingle(transaction) {
        const response = await this.client.post('/predict', transaction);
        return response.data;
    }
    
    async predictWithExplanation(transaction) {
        const response = await this.client.post('/predict/explain', transaction);
        return response.data;
    }
    
    async predictBatch(transactions) {
        const response = await this.client.post('/predict/batch', {
            transactions: transactions
        });
        return response.data;
    }
}

// Usage
const client = new FraudDetectionClient();

const transaction = {
    step: 1,
    type: 'TRANSFER',
    amount: 181.0,
    nameOrig: 'C1231006815',
    oldbalanceOrg: 181.0,
    newbalanceOrig: 0.0,
    nameDest: 'C1666544295',
    oldbalanceDest: 0.0,
    newbalanceDest: 0.0
};

client.predictSingle(transaction)
    .then(result => {
        console.log(`Fraud Score: ${result.fraud_score}`);
        console.log(`Risk Level: ${result.risk_level}`);
    })
    .catch(error => {
        console.error('Error:', error.response.data);
    });
```

## SDKs and Libraries

### Official SDKs
- **Python SDK**: Available in `examples/python_sdk.py`
- **JavaScript SDK**: Available in `examples/javascript_sdk.js`

### Third-Party Libraries
- **requests** (Python): HTTP client library
- **axios** (JavaScript): Promise-based HTTP client
- **curl**: Command-line HTTP client
- **Postman**: API testing and development

## Troubleshooting

### Common Issues

#### Service Unavailable (503)
**Cause**: Fraud detector not initialized or model not loaded
**Solution**: 
1. Check service health: `GET /health`
2. Verify model file exists and is accessible
3. Restart service with correct model path

#### Validation Errors (400/422)
**Cause**: Invalid input data or missing required fields
**Solution**:
1. Verify all required fields are present
2. Check transaction type is valid
3. Ensure numeric fields are non-negative
4. Validate data types match expected format

#### Server Won't Start
**Cause**: Port conflict, missing dependencies, or configuration issues
**Solution**:
1. Check if port is already in use: `netstat -an | grep 8000`
2. Verify all dependencies installed: `pip list`
3. Check log output for specific errors
4. Try different port: `python run_api.py --port 8080`

#### Slow Response Times
**Cause**: Large batch sizes, model complexity, or resource constraints
**Solution**:
1. Reduce batch size (max 1000 transactions)
2. Use single predictions for real-time requirements
3. Monitor system resources (CPU, memory)
4. Consider model optimization or caching

#### Memory Errors
**Cause**: Insufficient system memory for model or large batches
**Solution**:
1. Increase system memory or virtual memory
2. Reduce batch sizes
3. Use model optimization techniques
4. Monitor memory usage during processing

### Debug Mode

Enable debug logging for troubleshooting:

```bash
python run_api.py --log-level DEBUG --reload
```

### Performance Monitoring

Monitor API performance:

```bash
# Check service status
curl http://localhost:8000/status

# Monitor system resources
top -p $(pgrep -f "run_api.py")

# Check API response times
time curl -X POST "http://localhost:8000/predict" -d @sample_transaction.json
```

### Log Analysis

Check application logs for errors:

```bash
# View recent logs
tail -f logs/fraud_detection.log

# Search for errors
grep -i error logs/fraud_detection.log

# Filter by timestamp
grep "2024-01-01" logs/fraud_detection.log
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
export FRAUD_API_HOST=0.0.0.0
export FRAUD_API_PORT=8000
export FRAUD_MODEL_PATH=/app/models/fraud_model.joblib
export FRAUD_RISK_THRESHOLD=0.5
export FRAUD_HIGH_RISK_THRESHOLD=0.8
export FRAUD_LOG_LEVEL=INFO
```

### Load Balancing

Use nginx for load balancing multiple API instances:

```nginx
upstream fraud_api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://fraud_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Monitoring and Alerting

Set up monitoring for production:

1. **Health Checks**: Regular `/health` endpoint monitoring
2. **Performance Metrics**: Response time and throughput monitoring
3. **Error Tracking**: Log aggregation and error alerting
4. **Resource Monitoring**: CPU, memory, and disk usage
5. **Business Metrics**: Fraud detection rates and accuracy

---

**Document Version**: 1.0  
**Last Updated**: October 2024  
**API Version**: 1.0.0