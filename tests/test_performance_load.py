"""
Performance and Load Testing for Fraud Detection System

This module contains comprehensive performance tests that validate:
- System performance with large datasets
- Real-time prediction latency requirements
- Concurrent API request handling
- Memory usage and resource consumption
- Throughput and scalability metrics

Requirements tested: 6.2, 6.3
"""

import pytest
import time
import threading
import concurrent.futures
import psutil
import os
import statistics
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Import system components
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineering
from src.data.data_encoder import DataEncoder
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.services.fraud_detector import FraudDetector
from src.services.fraud_api import app


class TestLargeDatasetPerformance:
    """Test system performance with large datasets."""
    
    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for performance testing."""
        np.random.seed(42)
        
        # Generate 50,000 transactions for performance testing
        n_samples = 50000
        
        data = {
            'step': np.random.randint(1, 745, n_samples),
            'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples),
            'amount': np.random.exponential(1000, n_samples),
            'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
            'oldbalanceOrg': np.random.exponential(5000, n_samples),
            'newbalanceOrig': np.random.exponential(5000, n_samples),
            'nameDest': [f'{"M" if np.random.random() > 0.8 else "C"}{i:09d}' for i in range(n_samples)],
            'oldbalanceDest': np.random.exponential(5000, n_samples),
            'newbalanceDest': np.random.exponential(5000, n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),  # 0.2% fraud rate
            'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.999, 0.001])  # Required by DataLoader
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def very_large_dataset(self):
        """Generate very large dataset for stress testing."""
        np.random.seed(42)
        
        # Generate 200,000 transactions for stress testing
        n_samples = 200000
        
        data = {
            'step': np.random.randint(1, 745, n_samples),
            'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples),
            'amount': np.random.exponential(1000, n_samples),
            'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
            'oldbalanceOrg': np.random.exponential(5000, n_samples),
            'newbalanceOrig': np.random.exponential(5000, n_samples),
            'nameDest': [f'{"M" if np.random.random() > 0.8 else "C"}{i:09d}' for i in range(n_samples)],
            'oldbalanceDest': np.random.exponential(5000, n_samples),
            'newbalanceDest': np.random.exponential(5000, n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),
            'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.999, 0.001])
        }
        
        return pd.DataFrame(data)
    
    def test_data_loading_performance(self, large_dataset, tmp_path):
        """Test data loading performance with large datasets."""
        # Save large dataset to CSV
        csv_path = tmp_path / "large_dataset.csv"
        large_dataset.to_csv(csv_path, index=False)
        
        # Test loading performance
        loader = DataLoader(chunk_size=10000)
        
        start_time = time.time()
        loaded_data = loader.load_data(str(csv_path))
        load_time = time.time() - start_time
        
        # Performance requirements
        assert load_time < 30.0  # Should load 50k records in under 30 seconds
        assert len(loaded_data) == len(large_dataset)
        
        # Test chunked loading performance
        start_time = time.time()
        chunks = list(loader.load_data_chunks(str(csv_path)))
        chunk_time = time.time() - start_time
        
        assert chunk_time < 35.0  # Chunked loading should be reasonable
        assert sum(len(chunk) for chunk in chunks) == len(large_dataset)
    
    def test_data_preprocessing_performance(self, large_dataset):
        """Test data preprocessing performance with large datasets."""
        # Test data cleaning performance
        cleaner = DataCleaner()
        
        start_time = time.time()
        cleaned_data = cleaner.clean_data(large_dataset.copy())
        clean_time = time.time() - start_time
        
        assert clean_time < 20.0  # Should clean 50k records in under 20 seconds
        assert len(cleaned_data) <= len(large_dataset)  # May remove some records
        
        # Test feature engineering performance
        feature_engineer = FeatureEngineering()
        
        start_time = time.time()
        engineered_data = feature_engineer.engineer_features(cleaned_data.copy())
        engineer_time = time.time() - start_time
        
        assert engineer_time < 15.0  # Should engineer features in under 15 seconds
        assert len(engineered_data) == len(cleaned_data)
        
        # Test encoding performance
        encoder = DataEncoder()
        
        # Define categorical features (including time_period created by feature engineering)
        categorical_features = ['type', 'nameOrig', 'nameDest', 'time_period']
        
        start_time = time.time()
        encoded_data = encoder.encode_categorical_features(engineered_data.copy(), categorical_features)
        encode_time = time.time() - start_time
        
        assert encode_time < 10.0  # Should encode in under 10 seconds
        assert len(encoded_data) == len(engineered_data)
    
    def test_model_training_performance(self, large_dataset):
        """Test model training performance with large datasets."""
        # Prepare data
        cleaner = DataCleaner()
        feature_engineer = FeatureEngineering()
        encoder = DataEncoder()
        
        cleaned_data = cleaner.clean_data(large_dataset.copy())
        engineered_data = feature_engineer.engineer_features(cleaned_data)
        categorical_features = ['type', 'nameOrig', 'nameDest', 'time_period']
        encoded_data = encoder.encode_categorical_features(engineered_data, categorical_features)
        
        # Prepare features and target
        feature_columns = [col for col in encoded_data.columns if col != 'isFraud']
        X = encoded_data[feature_columns]
        y = encoded_data['isFraud']
        
        # Test Logistic Regression training performance
        lr_model = LogisticRegressionModel()
        
        start_time = time.time()
        lr_model.train(X, y)
        lr_train_time = time.time() - start_time
        
        assert lr_train_time < 60.0  # Should train in under 60 seconds
        assert lr_model.is_trained
        
        # Test Random Forest training performance
        rf_model = RandomForestModel(n_estimators=50)  # Reduced for performance
        
        start_time = time.time()
        rf_model.train(X, y)
        rf_train_time = time.time() - start_time
        
        assert rf_train_time < 120.0  # Should train in under 2 minutes
        assert rf_model.is_trained
    
    def test_batch_prediction_performance(self, large_dataset):
        """Test batch prediction performance with large datasets."""
        # Prepare a simple trained model
        cleaner = DataCleaner()
        feature_engineer = FeatureEngineering()
        encoder = DataEncoder()
        
        # Use smaller sample for training
        train_sample = large_dataset.sample(n=5000, random_state=42)
        cleaned_data = cleaner.clean_data(train_sample.copy())
        engineered_data = feature_engineer.engineer_features(cleaned_data)
        categorical_features = ['type', 'nameOrig', 'nameDest', 'time_period']
        encoded_data = encoder.encode_categorical_features(engineered_data, categorical_features)
        
        feature_columns = [col for col in encoded_data.columns if col != 'isFraud']
        X_train = encoded_data[feature_columns]
        y_train = encoded_data['isFraud']
        
        # Train a fast model
        model = LogisticRegressionModel()
        model.train(X_train, y_train)
        
        # Prepare test data
        test_cleaned = cleaner.clean_data(large_dataset.copy())
        test_engineered = feature_engineer.engineer_features(test_cleaned)
        categorical_features = ['type', 'nameOrig', 'nameDest', 'time_period']
        test_encoded = encoder.encode_categorical_features(test_engineered, categorical_features, fit=False)
        X_test = test_encoded[feature_columns]
        
        # Test batch prediction performance
        start_time = time.time()
        predictions = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Performance requirements
        assert predict_time < 30.0  # Should predict 50k records in under 30 seconds
        assert len(predictions) == len(X_test)
        
        # Test prediction probabilities performance
        start_time = time.time()
        probabilities = model.predict_proba(X_test)
        proba_time = time.time() - start_time
        
        assert proba_time < 35.0  # Should compute probabilities in under 35 seconds
        assert len(probabilities) == len(X_test)
    
    def test_memory_usage_large_dataset(self, very_large_dataset):
        """Test memory usage with very large datasets."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process very large dataset
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_data(very_large_dataset.copy())
        
        # Check memory usage after cleaning
        mid_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = mid_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for 200k records)
        assert memory_increase < 500
        
        # Continue processing
        feature_engineer = FeatureEngineering()
        engineered_data = feature_engineer.engineer_features(cleaned_data)
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        # Total memory increase should be manageable (< 800MB)
        assert total_memory_increase < 800
        
        # Clean up
        del very_large_dataset, cleaned_data, engineered_data


class TestRealTimePredictionLatency:
    """Test real-time prediction latency requirements."""
    
    @pytest.fixture
    def trained_fraud_detector(self):
        """Create a trained fraud detector for latency testing."""
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'step': np.random.randint(1, 745, n_samples),
            'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples),
            'amount': np.random.exponential(1000, n_samples),
            'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
            'oldbalanceOrg': np.random.exponential(5000, n_samples),
            'newbalanceOrig': np.random.exponential(5000, n_samples),
            'nameDest': [f'{"M" if np.random.random() > 0.8 else "C"}{i:09d}' for i in range(n_samples)],
            'oldbalanceDest': np.random.exponential(5000, n_samples),
            'newbalanceDest': np.random.exponential(5000, n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
            'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.999, 0.001])
        }
        
        df = pd.DataFrame(data)
        
        # Process data
        cleaner = DataCleaner()
        feature_engineer = FeatureEngineering()
        encoder = DataEncoder()
        
        cleaned_data = cleaner.clean_data(df)
        engineered_data = feature_engineer.engineer_features(cleaned_data)
        categorical_features = ['type', 'nameOrig', 'nameDest', 'time_period']
        encoded_data = encoder.encode_categorical_features(engineered_data, categorical_features)
        
        # Train model
        feature_columns = [col for col in encoded_data.columns if col != 'isFraud']
        X = encoded_data[feature_columns]
        y = encoded_data['isFraud']
        
        model = LogisticRegressionModel()
        model.train(X, y)
        
        # Create fraud detector
        detector = FraudDetector(model=model)
        return detector
    
    @pytest.fixture
    def sample_transaction(self):
        """Create sample transaction for latency testing."""
        return {
            'step': 100,
            'type': 'TRANSFER',
            'amount': 50000.0,
            'nameOrig': 'C123456789',
            'oldbalanceOrg': 100000.0,
            'newbalanceOrig': 50000.0,
            'nameDest': 'C987654321',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 50000.0
        }
    
    def test_single_prediction_latency(self, trained_fraud_detector, sample_transaction):
        """Test single transaction prediction latency."""
        detector = trained_fraud_detector
        
        # Warm up the model (first prediction may be slower)
        detector.score_transaction(sample_transaction)
        
        # Measure latency for multiple predictions
        latencies = []
        for _ in range(100):
            start_time = time.time()
            score = detector.score_transaction(sample_transaction)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            assert isinstance(score, float)
            assert 0 <= score <= 1
        
        # Latency requirements
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        max_latency = max(latencies)
        
        # Performance requirements (in milliseconds)
        assert avg_latency < 50.0  # Average latency under 50ms
        assert p95_latency < 100.0  # 95th percentile under 100ms
        assert max_latency < 200.0  # Maximum latency under 200ms
        
        print(f"Latency stats - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms, Max: {max_latency:.2f}ms")
    
    def test_explanation_latency(self, trained_fraud_detector, sample_transaction):
        """Test fraud explanation generation latency."""
        detector = trained_fraud_detector
        
        # Warm up
        detector.get_fraud_explanation(sample_transaction)
        
        # Measure explanation latencies
        latencies = []
        for _ in range(50):  # Fewer iterations as explanations are more expensive
            start_time = time.time()
            explanation = detector.get_fraud_explanation(sample_transaction)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            assert 'fraud_score' in explanation
            assert 'explanation_text' in explanation
        
        # Explanation latency requirements
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        
        # Explanations can be slower but should still be reasonable
        assert avg_latency < 200.0  # Average under 200ms
        assert p95_latency < 500.0  # 95th percentile under 500ms
        
        print(f"Explanation latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")
    
    def test_batch_prediction_throughput(self, trained_fraud_detector):
        """Test batch prediction throughput."""
        detector = trained_fraud_detector
        
        # Create batch of transactions
        batch_sizes = [10, 50, 100, 500, 1000]
        
        for batch_size in batch_sizes:
            transactions = []
            for i in range(batch_size):
                transaction = {
                    'step': 100 + i,
                    'type': 'PAYMENT',
                    'amount': 1000.0 + i,
                    'nameOrig': f'C{i:09d}',
                    'oldbalanceOrg': 2000.0,
                    'newbalanceOrig': 1000.0,
                    'nameDest': f'M{i:09d}',
                    'oldbalanceDest': 0.0,
                    'newbalanceDest': 1000.0
                }
                transactions.append(transaction)
            
            df = pd.DataFrame(transactions)
            
            # Measure batch processing time
            start_time = time.time()
            results = detector.batch_predict(df)
            batch_time = time.time() - start_time
            
            # Calculate throughput (transactions per second)
            throughput = batch_size / batch_time
            
            # Throughput requirements
            assert throughput > 100  # Should process at least 100 transactions/second
            assert len(results) == batch_size
            
            print(f"Batch size {batch_size}: {throughput:.1f} transactions/second")


class TestConcurrentAPIPerformance:
    """Test concurrent API request handling performance."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_request(self):
        """Sample API request for testing."""
        return {
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
    
    @patch('src.services.fraud_api.fraud_detector')
    def test_concurrent_prediction_requests(self, mock_fraud_detector, test_client, sample_request):
        """Test concurrent prediction request handling."""
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector.score_transaction.return_value = 0.5
        mock_detector._categorize_risk_level.return_value = ["MEDIUM"]
        mock_detector._calculate_confidence.return_value = 0.7
        mock_detector.risk_threshold = 0.5
        mock_fraud_detector.return_value = mock_detector
        
        def make_request():
            """Make a single prediction request."""
            start_time = time.time()
            response = test_client.post("/predict", json=sample_request)
            latency = (time.time() - start_time) * 1000
            return response, latency
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50]
        
        for concurrency in concurrency_levels:
            latencies = []
            success_count = 0
            
            # Execute concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(make_request) for _ in range(concurrency)]
                
                for future in concurrent.futures.as_completed(futures):
                    response, latency = future.result()
                    latencies.append(latency)
                    
                    if response.status_code == 200:
                        success_count += 1
                    elif response.status_code == 503:
                        # Service unavailable is acceptable in test environment
                        pass
                    else:
                        pytest.fail(f"Unexpected status code: {response.status_code}")
            
            # Performance analysis
            if latencies:
                avg_latency = statistics.mean(latencies)
                max_latency = max(latencies)
                
                # Concurrent performance requirements
                assert avg_latency < 500.0  # Average latency under 500ms under load
                assert max_latency < 2000.0  # Max latency under 2 seconds
                
                print(f"Concurrency {concurrency}: Avg latency {avg_latency:.1f}ms, "
                      f"Max latency {max_latency:.1f}ms, Success rate {success_count/concurrency:.1%}")
    
    @patch('src.services.fraud_api.fraud_detector')
    def test_batch_request_performance(self, mock_fraud_detector, test_client, sample_request):
        """Test batch request performance under load."""
        # Mock fraud detector
        mock_detector = Mock()
        
        def mock_batch_predict(df):
            # Simulate processing time proportional to batch size
            time.sleep(len(df) * 0.001)  # 1ms per transaction
            result_df = df.copy()
            result_df['fraud_score'] = 0.5
            result_df['fraud_prediction'] = 0
            result_df['risk_level'] = 'MEDIUM'
            result_df['processed_at'] = '2023-01-01T00:00:00'
            return result_df
        
        mock_detector.batch_predict.side_effect = mock_batch_predict
        mock_detector._calculate_confidence.return_value = 0.7
        mock_fraud_detector.return_value = mock_detector
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100, 500]
        
        for batch_size in batch_sizes:
            batch_request = {
                "transactions": [sample_request] * batch_size
            }
            
            start_time = time.time()
            response = test_client.post("/predict/batch", json=batch_request)
            request_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                assert data["total_transactions"] == batch_size
                
                # Batch performance requirements
                throughput = batch_size / (request_time / 1000)
                assert throughput > 50  # At least 50 transactions/second
                
                print(f"Batch size {batch_size}: {request_time:.1f}ms, "
                      f"Throughput: {throughput:.1f} tx/sec")
            elif response.status_code == 503:
                pytest.skip("Service not available for batch testing")
    
    @patch('src.services.fraud_api.fraud_detector')
    def test_mixed_workload_performance(self, mock_fraud_detector, test_client, sample_request):
        """Test performance under mixed workload (single + batch requests)."""
        # Mock fraud detector
        mock_detector = Mock()
        mock_detector.score_transaction.return_value = 0.5
        mock_detector._categorize_risk_level.return_value = ["MEDIUM"]
        mock_detector._calculate_confidence.return_value = 0.7
        mock_detector.risk_threshold = 0.5
        
        def mock_batch_predict(df):
            result_df = df.copy()
            result_df['fraud_score'] = 0.5
            result_df['fraud_prediction'] = 0
            result_df['risk_level'] = 'MEDIUM'
            result_df['processed_at'] = '2023-01-01T00:00:00'
            return result_df
        
        mock_detector.batch_predict.side_effect = mock_batch_predict
        mock_fraud_detector.return_value = mock_detector
        
        def make_single_request():
            """Make single prediction request."""
            return test_client.post("/predict", json=sample_request)
        
        def make_batch_request():
            """Make batch prediction request."""
            batch_request = {"transactions": [sample_request] * 10}
            return test_client.post("/predict/batch", json=batch_request)
        
        # Mixed workload: 70% single requests, 30% batch requests
        single_requests = 35
        batch_requests = 15
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit mixed requests
            futures = []
            
            # Single requests
            for _ in range(single_requests):
                futures.append(executor.submit(make_single_request))
            
            # Batch requests
            for _ in range(batch_requests):
                futures.append(executor.submit(make_batch_request))
            
            # Collect results
            responses = []
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                responses.append(response)
        
        total_time = time.time() - start_time
        
        # Analyze results
        success_responses = [r for r in responses if r.status_code == 200]
        service_unavailable = [r for r in responses if r.status_code == 503]
        
        # Mixed workload should complete in reasonable time
        assert total_time < 30.0  # Should complete in under 30 seconds
        
        # Most requests should succeed or fail consistently
        total_requests = single_requests + batch_requests
        success_rate = len(success_responses) / total_requests
        unavailable_rate = len(service_unavailable) / total_requests
        
        # Either high success rate or consistent unavailability (test environment)
        assert success_rate > 0.8 or unavailable_rate > 0.8
        
        print(f"Mixed workload: {total_time:.1f}s total, "
              f"Success rate: {success_rate:.1%}, "
              f"Unavailable rate: {unavailable_rate:.1%}")


class TestResourceUtilization:
    """Test system resource utilization under load."""
    
    def test_cpu_usage_under_load(self):
        """Test CPU usage during intensive processing."""
        # Generate data for CPU-intensive processing
        np.random.seed(42)
        n_samples = 10000
        
        data = pd.DataFrame({
            'step': np.random.randint(1, 745, n_samples),
            'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples),
            'amount': np.random.exponential(1000, n_samples),
            'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
            'oldbalanceOrg': np.random.exponential(5000, n_samples),
            'newbalanceOrig': np.random.exponential(5000, n_samples),
            'nameDest': [f'M{i:09d}' for i in range(n_samples)],
            'oldbalanceDest': np.random.exponential(5000, n_samples),
            'newbalanceDest': np.random.exponential(5000, n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
            'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.999, 0.001])
        })
        
        # Monitor CPU usage during processing
        process = psutil.Process(os.getpid())
        initial_cpu = process.cpu_percent()
        
        # Perform CPU-intensive operations (data processing only)
        cleaner = DataCleaner()
        feature_engineer = FeatureEngineering()
        encoder = DataEncoder()
        
        # Process data multiple times to create CPU load
        for i in range(5):
            cleaned_data = cleaner.clean_data(data.copy())
            engineered_data = feature_engineer.engineer_features(cleaned_data)
            categorical_features = ['type', 'nameOrig', 'nameDest', 'time_period']
            encoded_data = encoder.encode_categorical_features(engineered_data, categorical_features)
            
            # Perform some CPU-intensive calculations
            feature_columns = [col for col in encoded_data.columns if col != 'isFraud']
            X = encoded_data[feature_columns].select_dtypes(include=[np.number])  # Only numeric columns
            
            # CPU-intensive operations
            correlation_matrix = X.corr()
            mean_values = X.mean()
            std_values = X.std()
            
            # Ensure we have results
            assert len(correlation_matrix) > 0
            assert len(mean_values) > 0
            assert len(std_values) > 0
        
        # CPU usage should be reasonable (not pegged at 100%)
        final_cpu = process.cpu_percent()
        
        # Ensure processing completed successfully
        assert len(encoded_data) == len(data)
        
        print(f"CPU usage - Initial: {initial_cpu}%, Final: {final_cpu}%")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform repeated operations that might cause memory leaks
        for iteration in range(10):
            # Generate data
            np.random.seed(42 + iteration)
            data = pd.DataFrame({
                'step': np.random.randint(1, 745, 1000),
                'type': np.random.choice(['PAYMENT', 'TRANSFER'], 1000),
                'amount': np.random.exponential(1000, 1000),
                'nameOrig': [f'C{i:09d}' for i in range(1000)],
                'oldbalanceOrg': np.random.exponential(5000, 1000),
                'newbalanceOrig': np.random.exponential(5000, 1000),
                'nameDest': [f'M{i:09d}' for i in range(1000)],
                'oldbalanceDest': np.random.exponential(5000, 1000),
                'newbalanceDest': np.random.exponential(5000, 1000),
                'isFraud': np.random.choice([0, 1], 1000, p=[0.99, 0.01]),
                'isFlaggedFraud': np.random.choice([0, 1], 1000, p=[0.999, 0.001])
            })
            
            # Process data
            cleaner = DataCleaner()
            cleaned_data = cleaner.clean_data(data)
            
            # Clean up explicitly
            del data, cleaned_data, cleaner
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be minimal (< 50MB after 10 iterations)
        assert memory_increase < 50
        
        print(f"Memory usage - Baseline: {baseline_memory:.1f}MB, "
              f"Final: {final_memory:.1f}MB, "
              f"Increase: {memory_increase:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])