#!/usr/bin/env python3
"""
Test script for the fraud detection dashboard components.

This script tests the basic functionality of dashboard components
without running the full Streamlit interface.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_dashboard_imports():
    """Test that all dashboard components can be imported."""
    print("Testing dashboard imports...")
    
    try:
        from dashboard.fraud_dashboard import FraudDashboard
        print("✓ FraudDashboard imported successfully")
        
        from data.data_loader import DataLoader
        print("✓ DataLoader imported successfully")
        
        from models.logistic_regression_model import LogisticRegressionModel
        print("✓ LogisticRegressionModel imported successfully")
        
        from services.fraud_detector import FraudDetector
        print("✓ FraudDetector imported successfully")
        
        from visualization.data_visualizer import DataVisualizer
        print("✓ DataVisualizer imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_sample_data_generation():
    """Test sample data generation functionality."""
    print("\nTesting sample data generation...")
    
    try:
        from dashboard.fraud_dashboard import FraudDashboard
        
        dashboard = FraudDashboard()
        sample_data = dashboard._generate_sample_data()
        
        print(f"✓ Generated sample data with {len(sample_data)} rows")
        print(f"✓ Sample data columns: {list(sample_data.columns)}")
        print(f"✓ Fraud rate: {(sample_data['isFraud'].sum() / len(sample_data)) * 100:.3f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data generation error: {e}")
        return False

def test_basic_model_training():
    """Test basic model training functionality."""
    print("\nTesting basic model training...")
    
    try:
        from models.logistic_regression_model import LogisticRegressionModel
        from data.data_loader import DataLoader
        from data.feature_engineering import FeatureEngineering
        from data.data_encoder import DataEncoder
        
        # Generate small sample data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'step': np.random.randint(1, 745, n_samples),
            'type': np.random.choice(['PAYMENT', 'TRANSFER'], n_samples),
            'amount': np.random.lognormal(8, 1, n_samples),
            'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
            'oldbalanceOrg': np.random.lognormal(10, 1, n_samples),
            'newbalanceOrig': np.random.lognormal(10, 1, n_samples),
            'nameDest': [f'C{i:09d}' for i in range(n_samples)],
            'oldbalanceDest': np.random.lognormal(9, 1, n_samples),
            'newbalanceDest': np.random.lognormal(9, 1, n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Basic feature engineering
        feature_eng = FeatureEngineering()
        df_features = feature_eng.engineer_features(df)
        
        # Encoding
        encoder = DataEncoder()
        categorical_features = ['type']
        df_encoded = encoder.encode_categorical_features(df_features, categorical_features)
        
        # Prepare for training - select only numerical features
        # Remove string columns that shouldn't be used for training
        exclude_cols = ['isFraud', 'nameOrig', 'nameDest']
        feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
        
        # Select only numerical columns
        X = df_encoded[feature_cols].select_dtypes(include=[np.number])
        y = df_encoded['isFraud']
        
        # Train model
        model = LogisticRegressionModel()
        model.train(X, y)
        
        # Test prediction
        predictions = model.predict(X[:5])
        probabilities = model.predict_proba(X[:5])
        
        print(f"✓ Model trained successfully")
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"✓ Probabilities shape: {probabilities.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model training error: {e}")
        return False

def test_fraud_detector():
    """Test fraud detector functionality."""
    print("\nTesting fraud detector...")
    
    try:
        from services.fraud_detector import FraudDetector
        
        # Create a simple mock model for testing
        class MockModel:
            def __init__(self):
                self.is_trained = True
            
            def predict(self, X):
                return np.random.choice([0, 1], len(X))
            
            def predict_proba(self, X):
                return np.random.random(len(X))
        
        mock_model = MockModel()
        detector = FraudDetector(model=mock_model)
        
        # Test single transaction
        transaction = {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 50000.0,
            'nameOrig': 'C123456789',
            'oldbalanceOrg': 100000.0,
            'newbalanceOrig': 50000.0,
            'nameDest': 'C987654321',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 50000.0
        }
        
        fraud_score = detector.score_transaction(transaction)
        explanation = detector.get_fraud_explanation(transaction)
        
        print(f"✓ Fraud score calculated: {fraud_score:.4f}")
        print(f"✓ Explanation generated with keys: {list(explanation.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Fraud detector error: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Fraud Detection Dashboard Component Tests ===\n")
    
    tests = [
        test_dashboard_imports,
        test_sample_data_generation,
        test_basic_model_training,
        test_fraud_detector
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Dashboard components are working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())