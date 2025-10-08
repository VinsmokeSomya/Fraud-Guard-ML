# Fraud Detection Dashboard

A comprehensive Streamlit-based web dashboard for fraud detection analysis, model training, and real-time fraud scoring.

## Features

### 1. Data Upload & Exploration
- Upload CSV files with transaction data
- Load sample/synthetic data for demonstration
- Interactive data exploration with visualizations
- Data quality assessment and missing value analysis
- Transaction distribution analysis by type, amount, and time
- Fraud pattern visualization

### 2. Model Training & Evaluation
- Automated data preprocessing (cleaning, feature engineering, encoding)
- Train multiple models: Logistic Regression, Random Forest, XGBoost
- Comprehensive model evaluation with metrics comparison
- Interactive performance visualizations (ROC curves, confusion matrices)
- Model comparison and selection

### 3. Real-time Fraud Detection
- Single transaction analysis with detailed explanations
- Batch prediction for multiple transactions
- Risk factor identification and scoring
- Alert management and configuration
- Downloadable analysis results

## Quick Start

### Prerequisites
```bash
pip install -r dashboard_requirements.txt
```

### Running the Dashboard
```bash
streamlit run run_dashboard.py
```

The dashboard will open in your web browser at `http://localhost:8501`

## Usage Guide

### 1. Data Upload
1. Navigate to "Data Upload & Exploration" page
2. Either upload your CSV file or click "Load Sample Data"
3. Explore the data using the interactive visualizations
4. Review data quality metrics

### 2. Model Training
1. Go to "Model Training & Evaluation" page
2. Click "Preprocess Data" to prepare the data
3. Select which models to train (LR, RF, XGBoost)
4. Adjust training parameters if needed
5. Click "Train Selected Models"
6. Review model performance comparisons

### 3. Fraud Detection
1. Navigate to "Real-time Fraud Detection" page
2. For single transactions:
   - Enter transaction details in the form
   - Click "Analyze Transaction"
   - Review fraud score, risk level, and explanations
3. For batch analysis:
   - Upload CSV file with transactions
   - Click "Run Batch Analysis"
   - Download results

## Data Format

The dashboard expects CSV files with the following columns:
- `step`: Time step (1-744)
- `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
- `amount`: Transaction amount
- `nameOrig`: Origin account ID
- `oldbalanceOrg`: Origin account balance before transaction
- `newbalanceOrig`: Origin account balance after transaction
- `nameDest`: Destination account ID
- `oldbalanceDest`: Destination account balance before transaction
- `newbalanceDest`: Destination account balance after transaction
- `isFraud`: Fraud label (0/1) - optional for prediction data
- `isFlaggedFraud`: Business rule flag (0/1) - optional

## Features Overview

### Interactive Visualizations
- Transaction type distribution with fraud breakdown
- Amount distribution with log scaling
- Time-based transaction patterns
- Correlation heatmaps
- ROC curves and confusion matrices

### Model Capabilities
- Automated feature engineering
- Class imbalance handling
- Cross-validation
- Statistical significance testing
- Feature importance analysis

### Fraud Detection
- Real-time scoring with explanations
- Risk factor identification
- Business rule integration
- Alert management
- Batch processing capabilities

## Technical Architecture

The dashboard is built using:
- **Streamlit**: Web framework for the dashboard interface
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models and evaluation
- **XGBoost**: Gradient boosting model
- **Pandas**: Data manipulation and analysis

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r dashboard_requirements.txt
   ```

2. **Data Loading Issues**: Check that your CSV file matches the expected format

3. **Model Training Errors**: Ensure data contains the required columns and fraud labels

4. **Performance Issues**: For large datasets, consider using the sample data feature first

### Support

For issues or questions, please refer to the main project documentation or create an issue in the project repository.