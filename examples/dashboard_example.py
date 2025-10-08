#!/usr/bin/env python3
"""
Example script demonstrating how to use the Fraud Detection Dashboard.

This script shows how to programmatically interact with dashboard components
and provides examples of the main functionality.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from dashboard.fraud_dashboard import FraudDashboard
from data.data_loader import DataLoader
from models.logistic_regression_model import LogisticRegressionModel
from services.fraud_detector import FraudDetector

def main():
    """Demonstrate dashboard functionality."""
    print("=== Fraud Detection Dashboard Example ===\n")
    
    # 1. Initialize dashboard components
    print("1. Initializing dashboard components...")
    dashboard = FraudDashboard()
    
    # 2. Generate sample data
    print("2. Generating sample data...")
    sample_data = dashboard._generate_sample_data()
    print(f"   Generated {len(sample_data)} transactions")
    print(f"   Fraud rate: {(sample_data['isFraud'].sum() / len(sample_data)) * 100:.3f}%")
    
    # 3. Data exploration example
    print("\n3. Data exploration example...")
    print("   Dataset shape:", sample_data.shape)
    print("   Transaction types:", sample_data['type'].value_counts().to_dict())
    print("   Amount statistics:")
    print(f"     Mean: ${sample_data['amount'].mean():,.2f}")
    print(f"     Median: ${sample_data['amount'].median():,.2f}")
    print(f"     Max: ${sample_data['amount'].max():,.2f}")
    
    # 4. Data preprocessing example
    print("\n4. Data preprocessing example...")
    
    # Clean data
    cleaned_data = dashboard.data_cleaner.clean_data(sample_data)
    print(f"   Data cleaning completed: {len(cleaned_data)} rows")
    
    # Feature engineering
    engineered_data = dashboard.feature_engineering.engineer_features(cleaned_data)
    print(f"   Feature engineering completed: {len(engineered_data.columns)} features")
    
    # Encoding
    categorical_features = ['type']
    encoded_data = dashboard.data_encoder.encode_categorical_features(
        engineered_data, categorical_features
    )
    print(f"   Encoding completed: {len(encoded_data.columns)} total features")
    
    # 5. Model training example
    print("\n5. Model training example...")
    
    # Prepare data for training
    exclude_cols = ['isFraud', 'nameOrig', 'nameDest']
    feature_cols = [col for col in encoded_data.columns if col not in exclude_cols]
    X = encoded_data[feature_cols].select_dtypes(include=[np.number])
    y = encoded_data['isFraud']
    
    print(f"   Training features: {len(X.columns)}")
    print(f"   Training samples: {len(X)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    print("   Model training completed!")
    
    # Evaluate model
    results = dashboard.model_evaluator.evaluate_model(model, X_test, y_test, "Example Model")
    print(f"   Model accuracy: {results['accuracy']:.4f}")
    print(f"   Model F1-score: {results['f1_score']:.4f}")
    
    # 6. Fraud detection example
    print("\n6. Fraud detection example...")
    
    # Initialize fraud detector
    fraud_detector = FraudDetector(model=model)
    
    # Example transaction
    transaction = {
        'step': 100,
        'type': 'TRANSFER',
        'amount': 150000.0,
        'nameOrig': 'C123456789',
        'oldbalanceOrg': 200000.0,
        'newbalanceOrig': 50000.0,
        'nameDest': 'C987654321',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 150000.0
    }
    
    # Analyze transaction
    fraud_score = fraud_detector.score_transaction(transaction)
    explanation = fraud_detector.get_fraud_explanation(transaction)
    
    print(f"   Transaction fraud score: {fraud_score:.4f}")
    print(f"   Risk level: {explanation['risk_level']}")
    print(f"   Prediction: {'FRAUD' if explanation['is_fraud_prediction'] else 'LEGITIMATE'}")
    
    # Show risk factors
    risk_factors = explanation['risk_factors']
    if risk_factors['high_risk_factors']:
        print("   High risk factors:")
        for factor in risk_factors['high_risk_factors']:
            print(f"     • {factor}")
    
    # 7. Batch prediction example
    print("\n7. Batch prediction example...")
    
    # Create a small batch of test transactions
    batch_data = sample_data.head(10).copy()
    batch_results = fraud_detector.batch_predict(batch_data)
    
    fraud_predictions = (batch_results['fraud_prediction'] == 1).sum()
    high_risk_count = (batch_results['risk_level'] == 'HIGH').sum()
    
    print(f"   Analyzed {len(batch_results)} transactions")
    print(f"   Predicted fraud cases: {fraud_predictions}")
    print(f"   High risk transactions: {high_risk_count}")
    
    print("\n=== Dashboard Example Completed Successfully! ===")
    print("\nTo run the full interactive dashboard:")
    print("   streamlit run run_dashboard.py")

if __name__ == "__main__":
    main()