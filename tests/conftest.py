"""
Pytest configuration and shared fixtures for fraud detection tests.

This module provides common test configuration, fixtures, and utilities
that are shared across all test modules.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import warnings

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="session")
def sample_fraud_data():
    """Create sample fraud detection dataset for testing."""
    np.random.seed(42)
    n_samples = 500
    
    # Generate realistic transaction data
    data = {
        'step': np.random.randint(1, 745, n_samples),
        'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples),
        'amount': np.random.lognormal(7, 1.5, n_samples),
        'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
        'oldbalanceOrg': np.random.exponential(5000, n_samples),
        'newbalanceOrig': np.random.exponential(5000, n_samples),
        'nameDest': [f'{"M" if np.random.random() > 0.7 else "C"}{i:09d}' for i in range(n_samples)],
        'oldbalanceDest': np.random.exponential(3000, n_samples),
        'newbalanceDest': np.random.exponential(3000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic fraud labels
    fraud_prob = np.zeros(n_samples)
    fraud_prob += np.where(df['type'].isin(['CASH_OUT', 'TRANSFER']), 0.2, 0.02)
    fraud_prob += np.where(df['amount'] > 100000, 0.3, 0)
    fraud_prob += np.where((df['oldbalanceOrg'] == 0) & (df['newbalanceOrig'] == 0), 0.15, 0)
    
    df['isFraud'] = np.random.binomial(1, np.clip(fraud_prob, 0, 0.6), n_samples)
    
    # Ensure we have at least some fraud cases for training
    if df['isFraud'].sum() == 0:
        # Force some fraud cases
        fraud_indices = np.random.choice(df.index, size=max(1, int(n_samples * 0.05)), replace=False)
        df.loc[fraud_indices, 'isFraud'] = 1
    df['isFlaggedFraud'] = np.where(
        (df['type'] == 'TRANSFER') & (df['amount'] > 200000) & (df['isFraud'] == 1),
        1, 0
    )
    
    return df


@pytest.fixture(scope="session")
def sample_fraud_csv(sample_fraud_data, test_data_dir):
    """Create CSV file with sample fraud data."""
    csv_path = os.path.join(test_data_dir, "sample_fraud_data.csv")
    sample_fraud_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def clean_transaction_data():
    """Create clean transaction data without missing values."""
    return pd.DataFrame({
        'step': [1, 2, 3, 4, 5],
        'type': ['CASH_IN', 'CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT'],
        'amount': [1000.0, 500.0, 2000.0, 100.0, 50.0],
        'nameOrig': ['C123456789', 'C987654321', 'C111111111', 'C222222222', 'C333333333'],
        'oldbalanceOrg': [2000.0, 1000.0, 3000.0, 500.0, 200.0],
        'newbalanceOrig': [3000.0, 500.0, 1000.0, 400.0, 150.0],
        'nameDest': ['M123456789', 'C111111111', 'M987654321', 'M222222222', 'M333333333'],
        'oldbalanceDest': [5000.0, 0.0, 10000.0, 0.0, 0.0],
        'newbalanceDest': [6000.0, 500.0, 12000.0, 100.0, 50.0],
        'isFraud': [0, 0, 1, 0, 0],
        'isFlaggedFraud': [0, 0, 1, 0, 0]
    })


@pytest.fixture
def sample_transaction():
    """Create a single sample transaction for testing."""
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


@pytest.fixture
def high_risk_transaction():
    """Create a high-risk transaction for testing alerts."""
    return {
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


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid or "e2e" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark API tests
        if "api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        
        # Mark slow tests
        if any(keyword in item.nodeid for keyword in ["batch", "large", "performance", "concurrent"]):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set random seeds for reproducible tests
    np.random.seed(42)
    
    # Suppress matplotlib GUI warnings
    import matplotlib
    matplotlib.use('Agg')
    
    yield
    
    # Cleanup after test if needed
    pass


# Custom assertion helpers
def assert_valid_fraud_score(score):
    """Assert that a fraud score is valid."""
    assert isinstance(score, (int, float, np.number))
    assert 0 <= score <= 1


def assert_valid_dataframe(df, min_rows=1, required_columns=None):
    """Assert that a DataFrame is valid."""
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= min_rows
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        assert not missing_cols, f"Missing required columns: {missing_cols}"


def assert_model_trained(model):
    """Assert that a model is properly trained."""
    assert hasattr(model, 'is_trained')
    assert model.is_trained is True
    assert hasattr(model, 'feature_names')
    assert model.feature_names is not None


# Make assertion helpers available to all tests
pytest.assert_valid_fraud_score = assert_valid_fraud_score
pytest.assert_valid_dataframe = assert_valid_dataframe
pytest.assert_model_trained = assert_model_trained