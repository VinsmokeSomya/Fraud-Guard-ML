# Fraud Guard ML 🛡️

A comprehensive machine learning-powered fraud detection system for financial transactions. This system analyzes transaction patterns to identify fraudulent activities using multiple ML algorithms and provides real-time fraud scoring capabilities.

## 🎯 Features

- **Multi-Algorithm Detection**: Logistic Regression, Random Forest, and XGBoost models
- **Real-time Scoring**: FastAPI-based service for instant fraud detection
- **Interactive Dashboard**: Streamlit-based web interface for analysis
- **Comprehensive Analytics**: Detailed fraud pattern analysis and reporting
- **Production Ready**: Docker containerization and monitoring capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (recommended for large datasets)
- 2GB+ free disk space

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd fraud-guard-ml

# Run the automated setup script (creates virtual environment and installs dependencies)
python setup_env.py

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Unix/Linux/MacOS:
source .venv/bin/activate

# Verify installation
python -c "import pandas, sklearn, xgboost, streamlit, fastapi; print('All dependencies installed successfully!')"
```

### 2. Data Preparation

```bash
# Place your fraud dataset in the data directory
# The system expects a CSV file with transaction data
cp /path/to/your/Fraud.csv data/raw/

# For demo purposes, you can download the sample dataset:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Or use the synthetic fraud dataset generator (see examples/)

# Update configuration (optional)
cp .env.example .env
# Edit .env with your specific settings if needed
```

### 3. Run the System

Choose your preferred interface:

#### Option A: Interactive Dashboard (Recommended for beginners)
```bash
# Start the Streamlit dashboard
streamlit run run_dashboard.py

# Open browser to: http://localhost:8501
# Follow the guided workflow in the dashboard
```

#### Option B: API Service (For integration and production)
```bash
# Start the FastAPI service
python run_api.py

# API available at: http://localhost:8000
# Interactive docs at: http://localhost:8000/docs
```

#### Option C: Jupyter Notebooks (For data scientists)
```bash
# Start Jupyter for exploration and experimentation
jupyter notebook

# Navigate to examples/ folder for sample notebooks
```

### 4. First Steps

1. **Load Data**: Use the dashboard to upload your fraud dataset
2. **Explore Data**: Review transaction patterns and fraud statistics  
3. **Train Models**: Train multiple ML models (Logistic Regression, Random Forest, XGBoost)
4. **Evaluate Performance**: Compare model accuracy and performance metrics
5. **Detect Fraud**: Use real-time fraud detection on new transactions

## 📊 Dataset

The system works with financial transaction data containing:
- **Transaction Types**: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER
- **Customer Information**: Origin and destination account details
- **Balance Information**: Before and after transaction balances
- **Fraud Labels**: Ground truth fraud indicators
- **Time Series**: 30 days of transaction data (744 time steps)

## 🏗️ Project Structure

```
fraud-guard-ml/
├── .kiro/specs/fraud-detection-system/  # Project specifications
│   ├── requirements.md                  # Detailed requirements
│   ├── design.md                       # System design document
│   └── tasks.md                        # Implementation tasks
├── data/
│   ├── raw/                            # Original datasets
│   └── processed/                      # Cleaned and processed data
├── src/
│   ├── data/                           # Data processing modules
│   ├── models/                         # ML model implementations
│   ├── api/                            # FastAPI service
│   ├── dashboard/                      # Streamlit dashboard
│   └── utils/                          # Utility functions
├── models/                             # Trained model artifacts
├── notebooks/                          # Jupyter notebooks
├── tests/                              # Test suite
├── logs/                               # Application logs
└── reports/                            # Generated reports
```

## 🛠️ Development

### Implementation Tasks

Follow the detailed implementation plan in `.kiro/specs/fraud-detection-system/tasks.md`:

1. **Project Setup** - Dependencies and configuration
2. **Data Processing** - Loading, cleaning, feature engineering
3. **Model Development** - ML algorithms and training
4. **Evaluation** - Performance metrics and comparison
5. **Visualization** - Charts and analysis tools
6. **API Service** - Real-time fraud detection
7. **Dashboard** - Interactive web interface
8. **Deployment** - Production deployment setup

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

## 📈 Model Performance

The system implements multiple algorithms optimized for fraud detection:

- **Logistic Regression**: Fast, interpretable baseline model
- **Random Forest**: Robust ensemble method with feature importance
- **XGBoost**: High-performance gradient boosting with SHAP explanations

Performance metrics include:
- Precision, Recall, F1-Score
- AUC-ROC curves
- Confusion matrices
- Feature importance analysis

## 🔧 Configuration

Key configuration options in `.env`:

```env
# Data paths
DATA_PATH=./data/raw/Fraud.csv
MODEL_PATH=./models/

# Model parameters
RANDOM_STATE=42
TEST_SIZE=0.2
FRAUD_THRESHOLD=0.7

# API settings
API_HOST=localhost
API_PORT=8000
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build container
docker build -t fraud-guard-ml .

# Run service
docker run -p 8000:8000 fraud-guard-ml
```

### Production Considerations

- Model versioning and registry
- Performance monitoring
- Alert management
- Data privacy and security
- Scalability and load balancing

## 📝 API Documentation

The system provides a comprehensive REST API for fraud detection integration.

### API Access
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/health

### Quick API Examples

#### Single Transaction Scoring
```python
import requests

# Score a single transaction
response = requests.post("http://localhost:8000/predict", json={
    "step": 1,
    "type": "TRANSFER", 
    "amount": 250000.0,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 500000.0,
    "newbalanceOrig": 250000.0,
    "nameDest": "C1666544295", 
    "oldbalanceDest": 0.0,
    "newbalanceDest": 250000.0
})

result = response.json()
print(f"Fraud Score: {result['fraud_score']}")
print(f"Risk Level: {result['risk_level']}")
```

#### Detailed Fraud Analysis
```python
# Get detailed explanation
response = requests.post("http://localhost:8000/predict/explain", json={
    # ... same transaction data ...
})

explanation = response.json()
print(f"Risk Factors: {explanation['risk_factors']}")
print(f"Recommendations: {explanation['recommendations']}")
```

#### Batch Processing
```python
# Process multiple transactions
response = requests.post("http://localhost:8000/predict/batch", json={
    "transactions": [
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 9839.64,
            # ... other fields ...
        },
        {
            "step": 2, 
            "type": "TRANSFER",
            "amount": 181.0,
            # ... other fields ...
        }
    ]
})

batch_results = response.json()
print(f"Total: {batch_results['total_transactions']}")
print(f"Fraud Detected: {batch_results['fraud_detected']}")
```

📖 **Complete Documentation Available:**
- **[API Guide](docs/API_GUIDE.md)**: Comprehensive REST API documentation
- **[User Guide](docs/USER_GUIDE.md)**: Complete dashboard and reporting guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the implementation tasks in `tasks.md`
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Documentation

### 🎯 **[Complete Documentation Hub](docs/README.md)** ← **START HERE**

### Essential Guides

| Guide | Purpose | Audience |
|-------|---------|----------|
| **[📖 User Guide](docs/USER_GUIDE.md)** | Dashboard, training, reporting | All Users |
| **[🔧 API Guide](docs/API_GUIDE.md)** | REST API integration | Developers |
| **[⚙️ Configuration](docs/CONFIGURATION_AND_LOGGING.md)** | System setup | Administrators |
| **[🚀 Deployment](docs/MODEL_DEPLOYMENT.md)** | Production deployment | DevOps |

### Interactive Documentation

- **API Documentation**: http://localhost:8000/docs (when API is running)
- **Alternative API Docs**: http://localhost:8000/redoc
- **Test API Documentation**: `python test_api_docs.py`

### Project Specifications

- **[📋 Requirements](.kiro/specs/fraud-detection-system/requirements.md)**: Detailed system requirements
- **[🏗️ Design](.kiro/specs/fraud-detection-system/design.md)**: System architecture and design  
- **[✅ Tasks](.kiro/specs/fraud-detection-system/tasks.md)**: Implementation task list

## 🆘 Support

For questions or issues:
1. **Check Documentation**: Review the comprehensive guides above
2. **API Issues**: See [API Guide](docs/API_GUIDE.md) troubleshooting section
3. **Dashboard Issues**: See [User Guide](docs/USER_GUIDE.md) troubleshooting section
4. **System Architecture**: Review the [design document](.kiro/specs/fraud-detection-system/design.md)
5. **Implementation Details**: Check [tasks](.kiro/specs/fraud-detection-system/tasks.md) and requirements

---

**Built with ❤️ for financial security and fraud prevention**