# Fraud Guard ML 🛡️

A comprehensive machine learning-powered fraud detection system for financial transactions. This system analyzes transaction patterns to identify fraudulent activities using multiple ML algorithms and provides real-time fraud scoring capabilities.

## 🎯 Features

- **Multi-Algorithm Detection**: Logistic Regression, Random Forest, and XGBoost models
- **Real-time Scoring**: FastAPI-based service for instant fraud detection
- **Interactive Dashboard**: Streamlit-based web interface for analysis
- **Comprehensive Analytics**: Detailed fraud pattern analysis and reporting
- **Production Ready**: Docker containerization and monitoring capabilities

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/VinsmokeSomya/Fraud-Guard-ML.git
cd Fraud-Guard-ML

# Run the automated setup script
python setup_env.py

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Unix/Linux/MacOS:
source .venv/bin/activate
```

### 2. Data Preparation

```bash
# Place your fraud dataset in the data directory
cp /path/to/your/Fraud.csv data/raw/

# Update configuration
cp .env.example .env
# Edit .env with your specific settings
```

### 3. Run the System

```bash
# Start Jupyter for exploration
jupyter notebook

# Or run the Streamlit dashboard
streamlit run src/dashboard/app.py

# Or start the API service
uvicorn src.api.main:app --reload
```

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

Once the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example API Usage

```python
import requests

# Score a single transaction
response = requests.post("http://localhost:8000/predict", json={
    "step": 1,
    "type": "TRANSFER",
    "amount": 250000.0,
    "oldbalanceOrg": 500000.0,
    "newbalanceOrig": 250000.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 250000.0
})

fraud_score = response.json()["fraud_probability"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the implementation tasks in `tasks.md`
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:
1. Check the implementation tasks in `.kiro/specs/fraud-detection-system/`
2. Review the design document for architecture details
3. Open an issue on GitHub

---

**Built with ❤️ for financial security and fraud prevention**