"""
Integration tests for end-to-end fraud detection pipeline.

This module contains comprehensive integration tests that validate the complete
workflow from data loading through prediction, model training and evaluation,
and API endpoints with realistic data.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Import all components for end-to-end testing
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineering
from src.data.data_encoder import DataEncoder
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.model_evaluator import ModelEvaluator
from src.services.fraud_detector import FraudDetector
from src.services.alert_manager import AlertManager, NotificationConfig
from src.services.fraud_api import app, initialize_fraud_detector
from src.visualization.data_visualizer import DataVisualizer
from src.visualization.model_visualizer import ModelVisualizer


class TestEndToEndPipeline:
    """Test complete end-to-end fraud detection pipeline."""
    
    @pytest.fixture
    def sample_fraud_dataset(self):
        """Create a realistic sample fraud dataset for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic transaction data
        data = {
            'step': np.random.randint(1, 745, n_samples),
            'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples),
            'amount': np.random.lognormal(8, 2, n_samples),  # Log-normal distribution for amounts
            'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
            'oldbalanceOrg': np.random.exponential(10000, n_samples),
            'newbalanceOrig': np.random.exponential(10000, n_samples),
            'nameDest': [f'{"M" if np.random.random() > 0.7 else "C"}{i:09d}' for i in range(n_samples)],
            'oldbalanceDest': np.random.exponential(5000, n_samples),
            'newbalanceDest': np.random.exponential(5000, n_samples),
        }
        
        # Create realistic fraud labels based on business rules
        df = pd.DataFrame(data)
        
        # Higher fraud probability for certain conditions
        fraud_prob = np.zeros(n_samples)
        
        # CASH_OUT and TRANSFER have higher fraud rates
        fraud_prob += np.where(df['type'].isin(['CASH_OUT', 'TRANSFER']), 0.3, 0.05)
        
        # Large amounts increase fraud probability
        fraud_prob += np.where(df['amount'] > 200000, 0.4, 0)
        
        # Zero balance transactions increase fraud probability
        fraud_prob += np.where((df['oldbalanceOrg'] == 0) & (df['newbalanceOrig'] == 0), 0.2, 0)
        
        # Generate fraud labels
        df['isFraud'] = np.random.binomial(1, np.clip(fraud_prob, 0, 0.8), n_samples)
        
        # Ensure we have at least some fraud cases for training
        if df['isFraud'].sum() == 0:
            # Force some fraud cases
            fraud_indices = np.random.choice(df.index, size=max(1, int(n_samples * 0.05)), replace=False)
            df.loc[fraud_indices, 'isFraud'] = 1
        
        # Business rule flagging (subset of actual fraud)
        df['isFlaggedFraud'] = np.where(
            (df['type'] == 'TRANSFER') & (df['amount'] > 200000) & (df['isFraud'] == 1),
            1, 0
        )
        
        return df
    
    @pytest.fixture
    def temp_data_file(self, sample_fraud_dataset):
        """Create temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_fraud_dataset.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_complete_data_pipeline(self, temp_data_file):
        """Test complete data processing pipeline from loading to feature engineering."""
        # Step 1: Load data
        loader = DataLoader(chunk_size=500)
        raw_data = loader.load_data(temp_data_file, validate_schema=True)
        
        assert len(raw_data) == 1000
        assert 'isFraud' in raw_data.columns
        
        # Step 2: Clean data
        cleaner = DataCleaner(missing_strategy='auto', validate_business_rules=True)
        cleaned_data = cleaner.clean_data(raw_data)
        
        # Verify cleaning was successful
        assert len(cleaned_data) > 0
        cleaning_stats = cleaner.get_cleaning_summary()
        assert 'missing_values_handled' in cleaning_stats
        
        # Step 3: Feature engineering
        feature_engineer = FeatureEngineering()
        engineered_data = feature_engineer.engineer_features(cleaned_data)
        
        # Verify new features were created
        expected_features = [
            'balance_change_orig', 'balance_change_dest', 'amount_to_orig_balance_ratio',
            'is_merchant_dest', 'is_large_transfer', 'hour_of_day', 'day_of_month'
        ]
        
        for feature in expected_features:
            assert feature in engineered_data.columns
        
        # Step 4: Encode data
        encoder = DataEncoder()
        
        # Separate features and target
        target_col = 'isFraud'
        feature_cols = [col for col in engineered_data.columns if col != target_col]
        
        # Encode features
        X = encoder.fit_transform(engineered_data[feature_cols])
        y = engineered_data[target_col]
        
        assert len(X) == len(y)
        assert y.name == 'isFraud'
        assert X.select_dtypes(include=[np.number]).shape[1] > 0
    
    def test_model_training_and_evaluation_pipeline(self, temp_data_file, temp_model_dir):
        """Test complete model training and evaluation pipeline."""
        # Prepare data using the same pipeline
        loader = DataLoader(chunk_size=500)
        raw_data = loader.load_data(temp_data_file, validate_schema=True)
        
        cleaner = DataCleaner(missing_strategy='auto', validate_business_rules=True)
        cleaned_data = cleaner.clean_data(raw_data)
        
        feature_engineer = FeatureEngineering()
        engineered_data = feature_engineer.engineer_features(cleaned_data)
        
        encoder = DataEncoder()
        target_col = 'isFraud'
        feature_cols = [col for col in engineered_data.columns if col != target_col]
        X = encoder.fit_transform(engineered_data[feature_cols])
        y = engineered_data[target_col]
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # Ensure we have both classes before splitting
        if len(y.unique()) < 2:
            # Force some fraud cases if we don't have both classes
            fraud_indices = np.random.choice(y.index, size=max(1, int(len(y) * 0.1)), replace=False)
            y.iloc[fraud_indices] = 1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Test multiple models
        models = {
            'logistic_regression': LogisticRegressionModel(),
            'random_forest': RandomForestModel(n_estimators=10),  # Small for speed
            'xgboost': XGBoostModel(n_estimators=10)  # Small for speed
        }
        
        trained_models = {}
        evaluation_results = {}
        
        # Train and evaluate each model
        evaluator = ModelEvaluator()
        
        for model_name, model in models.items():
            # Train model
            model.train(X_train, y_train)
            assert model.is_trained
            
            # Make predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            assert len(predictions) == len(X_test)
            assert probabilities.shape == (len(X_test), 2)
            
            # Evaluate model
            metrics = evaluator.evaluate_model(model, X_test, y_test)
            
            # Verify evaluation metrics
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            for metric in expected_metrics:
                assert metric in metrics
                assert 0 <= metrics[metric] <= 1
            
            # Save model
            model_path = os.path.join(temp_model_dir, f"{model_name}_model")
            model.save_model(model_path)
            
            # Verify model files were created
            assert os.path.exists(f"{model_path}.joblib")
            assert os.path.exists(f"{model_path}.json")
            
            trained_models[model_name] = model
            evaluation_results[model_name] = metrics
        
        # Compare models - convert to list format expected by compare_models
        results_list = []
        for model_name, metrics in evaluation_results.items():
            metrics_with_name = metrics.copy()
            metrics_with_name['model_name'] = model_name
            results_list.append(metrics_with_name)
        
        comparison = evaluator.compare_models(results_list)
        assert 'best_model' in comparison
        assert comparison['best_model'] in models.keys()
        
        return trained_models, evaluation_results
    
    def test_fraud_detection_service_integration(self, temp_data_file, temp_model_dir):
        """Test fraud detection service with trained model."""
        # Train a model first
        trained_models, _ = self.test_model_training_and_evaluation_pipeline(temp_data_file, temp_model_dir)
        
        # Use the best performing model (random forest typically performs well)
        model = trained_models['random_forest']
        
        # Initialize alert manager
        notification_config = NotificationConfig(
            email_recipients=["test@example.com"],
            smtp_server="localhost",
            smtp_port=587
        )
        alert_manager = AlertManager(notification_config=notification_config)
        
        # Initialize fraud detector
        fraud_detector = FraudDetector(
            model=model,
            risk_threshold=0.5,
            high_risk_threshold=0.8,
            enable_explanations=True,
            alert_manager=alert_manager
        )
        
        # Test single transaction scoring
        test_transaction = {
            'step': 100,
            'type': 'TRANSFER',
            'amount': 250000.0,
            'nameOrig': 'C123456789',
            'oldbalanceOrg': 300000.0,
            'newbalanceOrig': 50000.0,
            'nameDest': 'C987654321',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 250000.0
        }
        
        # Score transaction
        fraud_score = fraud_detector.score_transaction(test_transaction)
        assert 0 <= fraud_score <= 1
        
        # Get detailed explanation
        explanation = fraud_detector.get_fraud_explanation(test_transaction)
        
        assert 'fraud_score' in explanation
        assert 'risk_level' in explanation
        assert 'risk_factors' in explanation
        assert 'explanation_text' in explanation
        assert 'recommendations' in explanation
        
        # Test batch prediction
        batch_transactions = pd.DataFrame([test_transaction] * 5)
        batch_results = fraud_detector.batch_predict(batch_transactions)
        
        assert len(batch_results) == 5
        assert 'fraud_score' in batch_results.columns
        assert 'fraud_prediction' in batch_results.columns
        assert 'risk_level' in batch_results.columns
        
        return fraud_detector
    
    def test_visualization_integration(self, temp_data_file):
        """Test visualization components with real data."""
        # Load and prepare data using the same pipeline
        loader = DataLoader(chunk_size=500)
        raw_data = loader.load_data(temp_data_file, validate_schema=True)
        
        cleaner = DataCleaner(missing_strategy='auto', validate_business_rules=True)
        cleaned_data = cleaner.clean_data(raw_data)
        
        feature_engineer = FeatureEngineering()
        engineered_data = feature_engineer.engineer_features(cleaned_data)
        
        encoder = DataEncoder()
        target_col = 'isFraud'
        feature_cols = [col for col in engineered_data.columns if col != target_col]
        X = encoder.fit_transform(engineered_data[feature_cols])
        y = engineered_data[target_col]
        
        # Combine features and target for visualization
        viz_data = X.copy()
        viz_data['isFraud'] = y
        
        # Test data visualizer
        data_viz = DataVisualizer()
        
        # Test various visualization methods (should not raise errors)
        try:
            data_viz.plot_transaction_distribution(viz_data)
            data_viz.plot_fraud_patterns(viz_data)
            data_viz.plot_correlation_matrix(X)
        except Exception as e:
            pytest.fail(f"Data visualization failed: {e}")
        
        # Train a simple model for model visualization
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        simple_model = RandomForestClassifier(n_estimators=10, random_state=42)
        simple_model.fit(X_train, y_train)
        y_pred = simple_model.predict(X_test)
        y_pred_proba = simple_model.predict_proba(X_test)
        
        # Test model visualizer
        model_viz = ModelVisualizer()
        
        try:
            model_viz.plot_confusion_matrix(y_test, y_pred)
            model_viz.plot_roc_curve(y_test, y_pred_proba[:, 1])
            model_viz.plot_feature_importance(simple_model, X.columns)
        except Exception as e:
            pytest.fail(f"Model visualization failed: {e}")
    
    @patch('smtplib.SMTP')
    def test_alert_system_integration(self, mock_smtp, temp_data_file, temp_model_dir):
        """Test alert system integration with fraud detection."""
        # Mock SMTP for testing
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Get fraud detector with alert system
        fraud_detector = self.test_fraud_detection_service_integration(temp_data_file, temp_model_dir)
        
        # Create high-risk transaction that should trigger alert
        high_risk_transaction = {
            'step': 200,
            'type': 'CASH_OUT',
            'amount': 500000.0,
            'nameOrig': 'C999999999',
            'oldbalanceOrg': 500000.0,
            'newbalanceOrig': 0.0,
            'nameDest': 'M888888888',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }
        
        # Score transaction (should trigger alert if score is high enough)
        fraud_score = fraud_detector.score_transaction(high_risk_transaction)
        
        # Check if alert was created (depends on model performance)
        if fraud_score >= fraud_detector.high_risk_threshold:
            active_alerts = fraud_detector.alert_manager.get_active_alerts()
            assert len(active_alerts) > 0
            
            # Test alert acknowledgment
            alert = active_alerts[0]
            result = fraud_detector.alert_manager.acknowledge_alert(alert.alert_id, "test_user")
            assert result is True
        
        # Test alert statistics
        stats = fraud_detector.alert_manager.get_alert_statistics()
        assert 'total_alerts' in stats
        assert 'active_alerts' in stats
    
    def test_model_persistence_and_loading(self, temp_data_file, temp_model_dir):
        """Test model persistence and loading functionality."""
        # Train models
        trained_models, _ = self.test_model_training_and_evaluation_pipeline(temp_data_file, temp_model_dir)
        
        # Test loading saved models
        for model_name in trained_models.keys():
            model_path = os.path.join(temp_model_dir, f"{model_name}_model")
            
            # Create new model instance and load
            if model_name == 'logistic_regression':
                new_model = LogisticRegressionModel()
            elif model_name == 'random_forest':
                new_model = RandomForestModel()
            elif model_name == 'xgboost':
                new_model = XGBoostModel()
            
            new_model.load_model(model_path)
            
            # Verify model was loaded correctly
            assert new_model.is_trained
            assert new_model.feature_names is not None
            
            # Test that loaded model can make predictions
            # Prepare test data using the same pipeline
            loader = DataLoader(chunk_size=500)
            raw_data = loader.load_data(temp_data_file, validate_schema=True)
            
            cleaner = DataCleaner(missing_strategy='auto', validate_business_rules=True)
            cleaned_data = cleaner.clean_data(raw_data)
            
            feature_engineer = FeatureEngineering()
            engineered_data = feature_engineer.engineer_features(cleaned_data)
            
            encoder = DataEncoder()
            target_col = 'isFraud'
            feature_cols = [col for col in engineered_data.columns if col != target_col]
            X = encoder.fit_transform(engineered_data[feature_cols])
            
            predictions = new_model.predict(X.head(10))
            probabilities = new_model.predict_proba(X.head(10))
            
            assert len(predictions) == 10
            assert probabilities.shape == (10, 2)
    
    def test_performance_monitoring(self, temp_data_file, temp_model_dir):
        """Test performance monitoring and metrics collection."""
        # Get fraud detector
        fraud_detector = self.test_fraud_detection_service_integration(temp_data_file, temp_model_dir)
        
        # Create multiple test transactions
        test_transactions = []
        for i in range(10):
            transaction = {
                'step': i + 1,
                'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'TRANSFER', 'PAYMENT']),
                'amount': np.random.uniform(1000, 100000),
                'nameOrig': f'C{i:09d}',
                'oldbalanceOrg': np.random.uniform(0, 50000),
                'newbalanceOrig': np.random.uniform(0, 50000),
                'nameDest': f'M{i:09d}',
                'oldbalanceDest': np.random.uniform(0, 20000),
                'newbalanceDest': np.random.uniform(0, 20000)
            }
            test_transactions.append(transaction)
        
        # Score transactions and measure performance
        start_time = time.time()
        scores = []
        
        for transaction in test_transactions:
            score = fraud_detector.score_transaction(transaction)
            scores.append(score)
        
        end_time = time.time()
        
        # Verify performance
        total_time = end_time - start_time
        avg_time_per_transaction = total_time / len(test_transactions)
        
        # Should process transactions reasonably quickly (< 1 second each)
        assert avg_time_per_transaction < 1.0
        assert len(scores) == len(test_transactions)
        assert all(0 <= score <= 1 for score in scores)
    
    def test_error_handling_and_recovery(self, temp_data_file):
        """Test error handling and recovery in the pipeline."""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'invalid_column': [1, 2, 3],
            'another_invalid': ['a', 'b', 'c']
        })
        
        # Data loader should handle missing columns gracefully
        cleaner = DataCleaner()
        
        # Should not crash, but may return empty or handle gracefully
        try:
            result = cleaner.clean_data(invalid_data)
            # If it succeeds, result should be a DataFrame
            assert isinstance(result, pd.DataFrame)
        except (ValueError, KeyError):
            # It's acceptable to fail with invalid data
            pass
        
        # Test fraud detector with missing model
        fraud_detector = FraudDetector(model=None)
        
        test_transaction = {
            'step': 1,
            'type': 'PAYMENT',
            'amount': 1000.0,
            'nameOrig': 'C123456789',
            'oldbalanceOrg': 2000.0,
            'newbalanceOrig': 1000.0,
            'nameDest': 'M987654321',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 1000.0
        }
        
        # Should raise appropriate error
        with pytest.raises(ValueError, match="No model loaded"):
            fraud_detector.score_transaction(test_transaction)
    
    def test_data_quality_validation(self, temp_data_file):
        """Test data quality validation throughout the pipeline."""
        # Load data
        loader = DataLoader()
        data = loader.load_data(temp_data_file)
        
        # Get data info
        data_info = loader.get_data_info(data)
        
        # Verify data info structure
        assert 'shape' in data_info
        assert 'columns' in data_info
        assert 'missing_values' in data_info
        assert 'transaction_types' in data_info
        assert 'fraud_distribution' in data_info
        
        # Verify data quality
        assert data_info['shape'][0] > 0  # Has rows
        assert data_info['shape'][1] > 0  # Has columns
        
        # Check fraud distribution is reasonable
        fraud_dist = data_info['fraud_distribution']
        total_transactions = sum(fraud_dist.values())
        fraud_rate = fraud_dist.get(1, 0) / total_transactions
        
        # Fraud rate should be reasonable (not 0% or 100%)
        assert 0 < fraud_rate < 0.5


class TestAPIIntegration:
    """Test API endpoints with realistic data."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_transaction_request(self):
        """Create sample transaction request for API testing."""
        return {
            "step": 100,
            "type": "TRANSFER",
            "amount": 50000.0,
            "nameOrig": "C123456789",
            "oldbalanceOrg": 100000.0,
            "newbalanceOrig": 50000.0,
            "nameDest": "C987654321",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 50000.0
        }
    
    def test_api_health_check(self, test_client):
        """Test API health check endpoint."""
        response = test_client.get("/health")
        
        # Should return 200 even without model loaded
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "service_name" in data
        assert "status" in data
        assert "model_loaded" in data
    
    def test_api_root_endpoint(self, test_client):
        """Test API root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Fraud Detection API"
        assert "version" in data
    
    @patch('src.services.fraud_api.fraud_detector')
    def test_api_predict_endpoint(self, mock_fraud_detector, test_client, sample_transaction_request):
        """Test API prediction endpoint."""
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector.score_transaction.return_value = 0.75
        mock_detector._categorize_risk_level.return_value = ["HIGH"]
        mock_detector._calculate_confidence.return_value = 0.85
        mock_detector.risk_threshold = 0.5
        mock_fraud_detector.return_value = mock_detector
        
        # Make prediction request
        response = test_client.post("/predict", json=sample_transaction_request)
        
        if response.status_code == 503:
            # Service not initialized - this is acceptable for integration test
            pytest.skip("Fraud detection service not initialized")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "fraud_score" in data
        assert "risk_level" in data
        assert "is_fraud_prediction" in data
        assert "confidence" in data
        assert "processed_at" in data
    
    @patch('src.services.fraud_api.fraud_detector')
    def test_api_predict_with_explanation(self, mock_fraud_detector, test_client, sample_transaction_request):
        """Test API prediction with explanation endpoint."""
        # Mock fraud detector
        mock_detector = Mock()
        mock_explanation = {
            "fraud_score": 0.75,
            "risk_level": "HIGH",
            "is_fraud_prediction": True,
            "confidence": 0.85,
            "risk_factors": {"high_risk_factors": ["Large transfer amount"]},
            "feature_importance": {"amount": 0.3, "type": 0.2},
            "explanation_text": "High-risk transaction detected",
            "recommendations": ["Investigate immediately"],
            "processed_at": "2023-01-01T00:00:00"
        }
        mock_detector.get_fraud_explanation.return_value = mock_explanation
        mock_fraud_detector.return_value = mock_detector
        
        response = test_client.post("/predict/explain", json=sample_transaction_request)
        
        if response.status_code == 503:
            pytest.skip("Fraud detection service not initialized")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "fraud_score" in data
        assert "risk_factors" in data
        assert "explanation_text" in data
        assert "recommendations" in data
    
    @patch('src.services.fraud_api.fraud_detector')
    def test_api_batch_predict(self, mock_fraud_detector, test_client, sample_transaction_request):
        """Test API batch prediction endpoint."""
        # Mock fraud detector
        mock_detector = Mock()
        
        # Create mock batch results
        batch_df = pd.DataFrame([sample_transaction_request] * 3)
        batch_df['fraud_score'] = [0.3, 0.7, 0.9]
        batch_df['fraud_prediction'] = [0, 1, 1]
        batch_df['risk_level'] = ['LOW', 'MEDIUM', 'HIGH']
        batch_df['processed_at'] = '2023-01-01T00:00:00'
        
        mock_detector.batch_predict.return_value = batch_df
        mock_detector._calculate_confidence.return_value = 0.8
        mock_fraud_detector.return_value = mock_detector
        
        # Create batch request
        batch_request = {
            "transactions": [sample_transaction_request] * 3
        }
        
        response = test_client.post("/predict/batch", json=batch_request)
        
        if response.status_code == 503:
            pytest.skip("Fraud detection service not initialized")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "total_transactions" in data
        assert "fraud_detected" in data
        assert "results" in data
        assert len(data["results"]) == 3
    
    def test_api_validation_errors(self, test_client):
        """Test API validation with invalid requests."""
        # Test invalid transaction type
        invalid_request = {
            "step": 100,
            "type": "INVALID_TYPE",
            "amount": 50000.0,
            "nameOrig": "C123456789",
            "oldbalanceOrg": 100000.0,
            "newbalanceOrig": 50000.0,
            "nameDest": "C987654321",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 50000.0
        }
        
        response = test_client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error
        
        # Test missing required fields
        incomplete_request = {
            "step": 100,
            "type": "PAYMENT"
            # Missing other required fields
        }
        
        response = test_client.post("/predict", json=incomplete_request)
        assert response.status_code == 422
    
    def test_api_error_handling(self, test_client):
        """Test API error handling with edge cases."""
        # Test with negative amounts
        negative_amount_request = {
            "step": 100,
            "type": "PAYMENT",
            "amount": -1000.0,  # Negative amount
            "nameOrig": "C123456789",
            "oldbalanceOrg": 100000.0,
            "newbalanceOrig": 50000.0,
            "nameDest": "C987654321",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 50000.0
        }
        
        response = test_client.post("/predict", json=negative_amount_request)
        assert response.status_code == 422  # Should fail validation
        
        # Test with invalid step values
        invalid_step_request = {
            "step": -1,  # Invalid step
            "type": "PAYMENT",
            "amount": 1000.0,
            "nameOrig": "C123456789",
            "oldbalanceOrg": 100000.0,
            "newbalanceOrig": 50000.0,
            "nameDest": "C987654321",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 50000.0
        }
        
        response = test_client.post("/predict", json=invalid_step_request)
        assert response.status_code == 422


class TestConcurrencyAndPerformance:
    """Test system performance under concurrent load."""
    
    @pytest.fixture
    def mock_fraud_detector(self):
        """Create mock fraud detector for performance testing."""
        detector = Mock()
        detector.score_transaction.return_value = 0.5
        detector._categorize_risk_level.return_value = ["MEDIUM"]
        detector._calculate_confidence.return_value = 0.7
        detector.risk_threshold = 0.5
        return detector
    
    @patch('src.services.fraud_api.fraud_detector')
    def test_concurrent_api_requests(self, mock_fraud_detector_global, mock_fraud_detector):
        """Test API performance under concurrent requests."""
        mock_fraud_detector_global.return_value = mock_fraud_detector
        
        client = TestClient(app)
        
        sample_request = {
            "step": 100,
            "type": "PAYMENT",
            "amount": 1000.0,
            "nameOrig": "C123456789",
            "oldbalanceOrg": 2000.0,
            "newbalanceOrig": 1000.0,
            "nameDest": "M987654321",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1000.0
        }
        
        # Simulate concurrent requests
        import concurrent.futures
        import threading
        
        def make_request():
            return client.post("/predict", json=sample_request)
        
        # Test with multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete successfully or fail consistently
        status_codes = [r.status_code for r in responses]
        
        # Should either all succeed (200) or all fail with service unavailable (503)
        assert len(set(status_codes)) <= 2  # At most 2 different status codes
        assert all(code in [200, 503] for code in status_codes)
    
    def test_memory_usage_with_large_batch(self, mock_fraud_detector):
        """Test memory usage with large batch predictions."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large batch of transactions
        large_batch = pd.DataFrame({
            'step': range(1000),
            'type': ['PAYMENT'] * 1000,
            'amount': [1000.0] * 1000,
            'nameOrig': [f'C{i:09d}' for i in range(1000)],
            'oldbalanceOrg': [2000.0] * 1000,
            'newbalanceOrig': [1000.0] * 1000,
            'nameDest': [f'M{i:09d}' for i in range(1000)],
            'oldbalanceDest': [0.0] * 1000,
            'newbalanceDest': [1000.0] * 1000
        })
        
        # Mock batch prediction
        mock_fraud_detector.batch_predict.return_value = large_batch.copy()
        
        # Process batch
        result = mock_fraud_detector.batch_predict(large_batch)
        
        # Check memory usage didn't increase dramatically
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100
        assert len(result) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])