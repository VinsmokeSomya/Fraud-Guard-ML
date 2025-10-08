# Fraud Detection API

A FastAPI-based REST API for real-time fraud detection and risk assessment of financial transactions.

## Features

- **Single Transaction Scoring**: Score individual transactions for fraud risk
- **Batch Processing**: Process multiple transactions simultaneously
- **Detailed Explanations**: Get comprehensive fraud analysis with risk factors
- **Health Monitoring**: Service health checks and status monitoring
- **Configurable Thresholds**: Adjust fraud detection sensitivity
- **Interactive Documentation**: Auto-generated API docs with Swagger UI

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Basic startup (without trained model)
python run_api.py

# With trained model
python run_api.py --model-path models/fraud_model.joblib

# Custom configuration
python run_api.py --host 0.0.0.0 --port 8080 --risk-threshold 0.6
```

### 3. Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Core Endpoints

#### `GET /` - Root Information
Returns basic API information and available endpoints.

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
    "high_risk_factors": ["Complete balance depletion detected"],
    "medium_risk_factors": ["High-risk transaction type: TRANSFER"],
    "low_risk_factors": [],
    "risk_scores": {
      "transaction_type": 0.7,
      "amount": 0.3,
      "balance_patterns": 0.9
    }
  },
  "explanation_text": "HIGH RISK: This transaction has a 85.0% probability of being fraudulent...",
  "recommendations": [
    "IMMEDIATE ACTION REQUIRED: Block transaction and investigate",
    "Contact customer to verify transaction legitimacy"
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
      "confidence": 0.7,
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

## Transaction Data Format

All transaction requests must include the following fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `step` | int | Time step (0-744) | 1 |
| `type` | string | Transaction type | "TRANSFER" |
| `amount` | float | Transaction amount | 181.0 |
| `nameOrig` | string | Origin customer ID | "C1231006815" |
| `oldbalanceOrg` | float | Origin balance before | 181.0 |
| `newbalanceOrig` | float | Origin balance after | 0.0 |
| `nameDest` | string | Destination customer ID | "C1666544295" |
| `oldbalanceDest` | float | Destination balance before | 0.0 |
| `newbalanceDest` | float | Destination balance after | 0.0 |

### Valid Transaction Types
- `CASH-IN`: Cash deposit
- `CASH-OUT`: Cash withdrawal
- `DEBIT`: Debit transaction
- `PAYMENT`: Payment to merchant
- `TRANSFER`: Transfer between accounts

## Risk Levels

The API categorizes transactions into three risk levels:

- **LOW**: `fraud_score < risk_threshold` (default: < 0.5)
- **MEDIUM**: `risk_threshold ≤ fraud_score < high_risk_threshold` (default: 0.5-0.8)
- **HIGH**: `fraud_score ≥ high_risk_threshold` (default: ≥ 0.8)

## Testing the API

### Using the Test Script

```bash
# Start the API server in one terminal
python run_api.py

# Run tests in another terminal
python test_api.py

# Test against different URL
python test_api.py --url http://localhost:8080
```

### Using curl

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Single prediction
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

### Using Python requests

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
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
)

result = response.json()
print(f"Fraud Score: {result['fraud_score']}")
print(f"Risk Level: {result['risk_level']}")
```

## Configuration Options

### Command Line Arguments

```bash
python run_api.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | 0.0.0.0 | Server host address |
| `--port` | 8000 | Server port |
| `--model-path` | None | Path to trained model |
| `--risk-threshold` | 0.5 | Risk classification threshold |
| `--high-risk-threshold` | 0.8 | High-risk alert threshold |
| `--no-explanations` | False | Disable detailed explanations |
| `--reload` | False | Enable auto-reload (development) |
| `--log-level` | INFO | Logging level |

### Environment Variables

You can also configure the API using environment variables:

```bash
export FRAUD_API_HOST=0.0.0.0
export FRAUD_API_PORT=8000
export FRAUD_MODEL_PATH=/path/to/model.joblib
export FRAUD_RISK_THRESHOLD=0.5
export FRAUD_HIGH_RISK_THRESHOLD=0.8
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- **400 Bad Request**: Invalid input data or parameters
- **404 Not Found**: Endpoint not found
- **500 Internal Server Error**: Server-side processing error
- **503 Service Unavailable**: Service not initialized or model not loaded

Example error response:
```json
{
  "detail": "Invalid transaction data: Transaction type must be one of: ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']"
}
```

## Performance Considerations

- **Single Predictions**: ~10-50ms per transaction
- **Batch Processing**: More efficient for multiple transactions
- **Concurrent Requests**: API supports multiple simultaneous requests
- **Memory Usage**: Scales with batch size and model complexity

## Security Notes

- The API runs without authentication by default (suitable for internal use)
- For production deployment, consider adding:
  - API key authentication
  - Rate limiting
  - HTTPS/TLS encryption
  - Input validation and sanitization
  - Audit logging

## Deployment

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

### Production Deployment

For production use, consider:

1. **Load Balancing**: Use nginx or similar for load balancing
2. **Process Management**: Use gunicorn or uvicorn workers
3. **Monitoring**: Add health checks and metrics collection
4. **Logging**: Configure structured logging and log aggregation
5. **Scaling**: Use container orchestration (Kubernetes, Docker Swarm)

## Troubleshooting

### Common Issues

1. **Service Unavailable (503)**
   - Check if the fraud detector is properly initialized
   - Verify model file exists and is accessible

2. **Validation Errors (400)**
   - Ensure all required fields are present
   - Check transaction type is valid
   - Verify numeric fields are non-negative

3. **Server Won't Start**
   - Check if port is already in use
   - Verify all dependencies are installed
   - Check log output for specific errors

### Debug Mode

Run with debug logging for troubleshooting:

```bash
python run_api.py --log-level DEBUG --reload
```

## Support

For issues and questions:
1. Check the interactive API documentation at `/docs`
2. Review the logs for error details
3. Run the test script to verify functionality
4. Check the main project README for general setup instructions