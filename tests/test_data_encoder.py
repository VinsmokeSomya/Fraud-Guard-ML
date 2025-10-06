"""
Unit tests for DataEncoder class.

This module contains comprehensive tests for encoding and scaling transformations
to validate the correctness of categorical encoding and numerical scaling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from src.data.data_encoder import DataEncoder, StandardScalerWrapper, MinMaxScalerWrapper


class TestDataEncoder:
    """Test class for DataEncoder functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing encoding and scaling."""
        return pd.DataFrame({
            'categorical_col': ['A', 'B', 'C', 'A', 'B', 'C', 'A'],
            'numerical_col1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'numerical_col2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            'target': [0, 1, 0, 1, 0, 1, 0]
        })
    
    @pytest.fixture
    def transaction_data(self):
        """Create transaction-like data for testing."""
        return pd.DataFrame({
            'step': [1, 2, 3, 4, 5],
            'type': ['CASH_IN', 'CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT'],
            'amount': [1000.0, 500.0, 2000.0, 100.0, 50.0],
            'nameOrig': ['C123', 'C456', 'C789', 'C012', 'C345'],
            'oldbalanceOrg': [2000.0, 1000.0, 3000.0, 500.0, 200.0],
            'newbalanceOrig': [3000.0, 500.0, 1000.0, 400.0, 150.0],
            'isFraud': [0, 0, 1, 0, 0]
        })
    
    @pytest.fixture
    def data_encoder(self):
        """Create DataEncoder instance for testing."""
        return DataEncoder()
    
    def test_initialization_default_parameters(self):
        """Test DataEncoder initialization with default parameters."""
        encoder = DataEncoder()
        
        assert encoder.categorical_encoding == 'label'
        assert encoder.numerical_scaling == 'standard'
        assert encoder.handle_unknown == 'ignore'
        assert encoder._fitted is False
        assert encoder._label_encoders == {}
        assert encoder._onehot_encoders == {}
        assert encoder._scalers == {}
    
    def test_initialization_custom_parameters(self):
        """Test DataEncoder initialization with custom parameters."""
        encoder = DataEncoder(
            categorical_encoding='onehot',
            numerical_scaling='minmax',
            handle_unknown='error'
        )
        
        assert encoder.categorical_encoding == 'onehot'
        assert encoder.numerical_scaling == 'minmax'
        assert encoder.handle_unknown == 'error'
    
    def test_auto_detect_feature_types(self, data_encoder, sample_data):
        """Test automatic feature type detection."""
        categorical_features, numerical_features = data_encoder._auto_detect_feature_types(
            sample_data, target_column='target'
        )
        
        assert 'categorical_col' in categorical_features
        assert 'numerical_col1' in numerical_features
        assert 'numerical_col2' in numerical_features
        assert 'target' not in categorical_features
        assert 'target' not in numerical_features
    
    def test_label_encoding_fit_transform(self, data_encoder, sample_data):
        """Test label encoding with fit_transform."""
        data_encoder.categorical_encoding = 'label'
        
        result = data_encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        # Verify encoding was applied
        assert result['categorical_col'].dtype in ['int64', 'int32']
        assert set(result['categorical_col'].unique()) == {0, 1, 2}
        
        # Verify encoder was stored
        assert 'categorical_col' in data_encoder._label_encoders
        
        # Verify original values can be mapped back
        encoder = data_encoder._label_encoders['categorical_col']
        assert 'A' in encoder.classes_
        assert 'B' in encoder.classes_
        assert 'C' in encoder.classes_
    
    def test_onehot_encoding_fit_transform(self, sample_data):
        """Test one-hot encoding with fit_transform."""
        encoder = DataEncoder(categorical_encoding='onehot')
        
        result = encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        # Verify one-hot encoding was applied
        assert 'categorical_col' not in result.columns  # Original column should be removed
        
        # Check for one-hot encoded columns
        onehot_cols = [col for col in result.columns if col.startswith('categorical_col_')]
        assert len(onehot_cols) == 3  # A, B, C
        
        # Verify encoder was stored
        assert 'categorical_col' in encoder._onehot_encoders
    
    def test_standard_scaling_fit_transform(self, data_encoder, sample_data):
        """Test standard scaling with fit_transform."""
        data_encoder.numerical_scaling = 'standard'
        
        result = data_encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        # Verify scaling was applied (mean should be close to 0, std close to 1)
        assert abs(result['numerical_col1'].mean()) < 1e-10
        assert abs(result['numerical_col1'].std() - 1.0) < 0.2  # Allow some tolerance for small datasets
        assert abs(result['numerical_col2'].mean()) < 1e-10
        assert abs(result['numerical_col2'].std() - 1.0) < 0.2
        
        # Verify scalers were stored
        assert 'numerical_col1' in data_encoder._scalers
        assert 'numerical_col2' in data_encoder._scalers
    
    def test_minmax_scaling_fit_transform(self, sample_data):
        """Test min-max scaling with fit_transform."""
        encoder = DataEncoder(numerical_scaling='minmax')
        
        result = encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        # Verify scaling was applied (values should be between 0 and 1)
        assert abs(result['numerical_col1'].min() - 0.0) < 1e-10
        assert abs(result['numerical_col1'].max() - 1.0) < 1e-10
        assert abs(result['numerical_col2'].min() - 0.0) < 1e-10
        assert abs(result['numerical_col2'].max() - 1.0) < 1e-10
        
        # Verify scalers were stored
        assert 'numerical_col1' in encoder._scalers
        assert 'numerical_col2' in encoder._scalers
    
    def test_no_scaling_option(self, sample_data):
        """Test with scaling disabled."""
        encoder = DataEncoder(numerical_scaling='none')
        
        original_values = sample_data[['numerical_col1', 'numerical_col2']].copy()
        
        result = encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        # Verify numerical columns are unchanged
        pd.testing.assert_frame_equal(
            result[['numerical_col1', 'numerical_col2']], 
            original_values
        )
        
        # Verify no scalers were created
        assert len(encoder._scalers) == 0
    
    def test_transform_after_fit(self, data_encoder, sample_data):
        """Test transform method after fitting."""
        # First fit the encoder
        data_encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        # Create new data to transform
        new_data = pd.DataFrame({
            'categorical_col': ['A', 'B', 'C'],
            'numerical_col1': [8.0, 9.0, 10.0],
            'numerical_col2': [80.0, 90.0, 100.0],
            'target': [1, 0, 1]
        })
        
        # Transform new data
        result = data_encoder.transform(new_data)
        
        # Verify transformations were applied consistently
        assert result['categorical_col'].dtype in ['int64', 'int32']
        assert len(result) == 3
        
        # Verify numerical columns were scaled using fitted scalers
        assert 'numerical_col1' in result.columns
        assert 'numerical_col2' in result.columns
    
    def test_transform_without_fit_raises_error(self, data_encoder, sample_data):
        """Test that transform raises error when called before fit."""
        with pytest.raises(ValueError, match="Encoders must be fitted before transforming data"):
            data_encoder.transform(sample_data)
    
    def test_stratified_train_test_split(self, data_encoder, sample_data):
        """Test stratified train-test split functionality."""
        X_train, X_test, y_train, y_test = data_encoder.stratified_train_test_split(
            sample_data,
            target_column='target',
            test_size=0.3,
            random_state=42
        )
        
        # Verify split proportions
        total_samples = len(sample_data)
        expected_train_size = int(total_samples * 0.7)
        expected_test_size = total_samples - expected_train_size
        
        assert len(X_train) == expected_train_size
        assert len(X_test) == expected_test_size
        assert len(y_train) == expected_train_size
        assert len(y_test) == expected_test_size
        
        # Verify stratification (fraud ratios should be similar)
        train_fraud_ratio = y_train.mean()
        test_fraud_ratio = y_test.mean()
        original_fraud_ratio = sample_data['target'].mean()
        
        # Allow some tolerance for small datasets
        assert abs(train_fraud_ratio - original_fraud_ratio) < 0.2
        assert abs(test_fraud_ratio - original_fraud_ratio) < 0.2
    
    def test_handle_unknown_categories_ignore(self, data_encoder, sample_data):
        """Test handling of unknown categories with 'ignore' strategy."""
        data_encoder.handle_unknown = 'ignore'
        
        # Fit on original data
        data_encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        # Create data with unknown category
        new_data = pd.DataFrame({
            'categorical_col': ['A', 'B', 'D'],  # 'D' is unknown
            'numerical_col1': [8.0, 9.0, 10.0],
            'numerical_col2': [80.0, 90.0, 100.0],
            'target': [1, 0, 1]
        })
        
        # Should not raise error
        result = data_encoder.transform(new_data)
        assert len(result) == 3
    
    def test_handle_missing_values_in_numerical_features(self, sample_data):
        """Test handling of missing values in numerical features."""
        # Add missing values
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.loc[0, 'numerical_col1'] = np.nan
        sample_data_with_nan.loc[1, 'numerical_col2'] = np.nan
        
        encoder = DataEncoder()
        
        # Should handle NaN values gracefully
        result = encoder.fit_transform(
            sample_data_with_nan,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        # Verify no NaN values remain
        assert not result['numerical_col1'].isna().any()
        assert not result['numerical_col2'].isna().any()
    
    def test_save_and_load_transformers(self, data_encoder, sample_data):
        """Test saving and loading fitted transformers."""
        # Fit the encoder
        data_encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        # Save transformers
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            data_encoder.save_transformers(tmp_path)
            
            # Create new encoder and load transformers
            new_encoder = DataEncoder()
            new_encoder.load_transformers(tmp_path)
            
            # Verify transformers were loaded
            assert new_encoder._fitted is True
            assert len(new_encoder._label_encoders) == len(data_encoder._label_encoders)
            assert len(new_encoder._scalers) == len(data_encoder._scalers)
            
            # Verify they produce same results
            test_data = sample_data.iloc[:3].copy()
            result1 = data_encoder.transform(test_data)
            result2 = new_encoder.transform(test_data)
            
            pd.testing.assert_frame_equal(result1, result2)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_transformers_without_fit_raises_error(self, data_encoder):
        """Test that saving transformers raises error when not fitted."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(ValueError, match="No fitted transformers to save"):
                data_encoder.save_transformers(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_get_feature_names_label_encoding(self, data_encoder, sample_data):
        """Test getting feature names after label encoding."""
        data_encoder.categorical_encoding = 'label'
        
        data_encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        feature_names = data_encoder.get_feature_names()
        
        assert 'categorical_col' in feature_names
        assert 'numerical_col1' in feature_names
        assert 'numerical_col2' in feature_names
        assert len(feature_names) == 3
    
    def test_get_feature_names_onehot_encoding(self, sample_data):
        """Test getting feature names after one-hot encoding."""
        encoder = DataEncoder(categorical_encoding='onehot')
        
        encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        feature_names = encoder.get_feature_names()
        
        # Should have one-hot encoded features
        onehot_features = [name for name in feature_names if name.startswith('categorical_col_')]
        assert len(onehot_features) == 3  # A, B, C
        
        assert 'numerical_col1' in feature_names
        assert 'numerical_col2' in feature_names
    
    def test_get_encoding_summary(self, data_encoder, sample_data):
        """Test getting encoding summary."""
        data_encoder.fit_transform(
            sample_data,
            categorical_features=['categorical_col'],
            numerical_features=['numerical_col1', 'numerical_col2'],
            target_column='target'
        )
        
        summary = data_encoder.get_encoding_summary()
        
        assert summary['categorical_encoding'] == 'label'
        assert summary['numerical_scaling'] == 'standard'
        assert summary['fitted'] is True
        assert 'categorical_col' in summary['categorical_features']
        assert 'numerical_col1' in summary['numerical_features']
        assert 'numerical_col2' in summary['numerical_features']
        assert summary['original_shape'] == (7, 4)
        assert summary['encoded_shape'] == (7, 4)  # Same shape for label encoding
    
    def test_encode_categorical_features_separately(self, data_encoder, sample_data):
        """Test encoding categorical features separately."""
        result = data_encoder.encode_categorical_features(
            sample_data,
            categorical_features=['categorical_col'],
            fit=True
        )
        
        # Verify encoding was applied
        assert result['categorical_col'].dtype in ['int64', 'int32']
        assert 'categorical_col' in data_encoder._label_encoders
    
    def test_scale_numerical_features_separately(self, data_encoder, sample_data):
        """Test scaling numerical features separately."""
        result = data_encoder.scale_numerical_features(
            sample_data,
            numerical_features=['numerical_col1', 'numerical_col2'],
            fit=True
        )
        
        # Verify scaling was applied
        assert abs(result['numerical_col1'].mean()) < 1e-10
        assert abs(result['numerical_col1'].std() - 1.0) < 0.2  # Allow tolerance for small datasets
        assert 'numerical_col1' in data_encoder._scalers
        assert 'numerical_col2' in data_encoder._scalers
    
    def test_transaction_data_encoding(self, data_encoder, transaction_data):
        """Test encoding with transaction-like data."""
        result = data_encoder.fit_transform(
            transaction_data,
            categorical_features=['type', 'nameOrig'],
            numerical_features=['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig'],
            target_column='isFraud'
        )
        
        # Verify categorical encoding
        assert result['type'].dtype in ['int64', 'int32']
        assert result['nameOrig'].dtype in ['int64', 'int32']
        
        # Verify numerical scaling
        for col in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig']:
            assert abs(result[col].mean()) < 1e-10
            assert abs(result[col].std() - 1.0) < 0.2  # Allow tolerance for small datasets
        
        # Verify target column is preserved
        assert 'isFraud' in result.columns
        pd.testing.assert_series_equal(result['isFraud'], transaction_data['isFraud'])


class TestStandardScalerWrapper:
    """Test class for StandardScalerWrapper functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
    
    def test_fit_transform_dataframe(self, sample_data):
        """Test fit_transform with DataFrame input."""
        scaler = StandardScalerWrapper()
        result = scaler.fit_transform(sample_data)
        
        # Verify result is DataFrame
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(sample_data.columns)
        assert list(result.index) == list(sample_data.index)
        
        # Verify scaling (mean ~0, std ~1)
        assert abs(result['col1'].mean()) < 1e-10
        assert abs(result['col1'].std() - 1.0) < 0.2  # Allow tolerance for small datasets
        assert abs(result['col2'].mean()) < 1e-10
        assert abs(result['col2'].std() - 1.0) < 0.2
    
    def test_fit_transform_numpy_array(self):
        """Test fit_transform with numpy array input."""
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
        scaler = StandardScalerWrapper()
        result = scaler.fit_transform(data)
        
        # Verify result is numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        
        # Verify scaling
        assert abs(result[:, 0].mean()) < 1e-10
        assert abs(result[:, 0].std() - 1.0) < 1e-10
    
    def test_fit_then_transform(self, sample_data):
        """Test separate fit and transform calls."""
        scaler = StandardScalerWrapper()
        scaler.fit(sample_data)
        result = scaler.transform(sample_data)
        
        # Verify scaling
        assert abs(result['col1'].mean()) < 1e-10
        assert abs(result['col1'].std() - 1.0) < 0.2  # Allow tolerance for small datasets
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform raises error when called before fit."""
        scaler = StandardScalerWrapper()
        
        with pytest.raises(ValueError, match="Scaler must be fitted before transforming data"):
            scaler.transform(sample_data)
    
    def test_inverse_transform(self, sample_data):
        """Test inverse transformation."""
        scaler = StandardScalerWrapper()
        scaled_data = scaler.fit_transform(sample_data)
        reconstructed_data = scaler.inverse_transform(scaled_data)
        
        # Verify reconstruction is close to original
        pd.testing.assert_frame_equal(reconstructed_data, sample_data, check_dtype=False, atol=1e-10)


class TestMinMaxScalerWrapper:
    """Test class for MinMaxScalerWrapper functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
    
    def test_fit_transform_dataframe(self, sample_data):
        """Test fit_transform with DataFrame input."""
        scaler = MinMaxScalerWrapper()
        result = scaler.fit_transform(sample_data)
        
        # Verify result is DataFrame
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(sample_data.columns)
        
        # Verify scaling (min=0, max=1)
        assert result['col1'].min() == 0.0
        assert result['col1'].max() == 1.0
        assert result['col2'].min() == 0.0
        assert result['col2'].max() == 1.0
    
    def test_custom_feature_range(self, sample_data):
        """Test MinMaxScaler with custom feature range."""
        scaler = MinMaxScalerWrapper(feature_range=(-1, 1))
        result = scaler.fit_transform(sample_data)
        
        # Verify custom range
        assert result['col1'].min() == -1.0
        assert result['col1'].max() == 1.0
        assert result['col2'].min() == -1.0
        assert result['col2'].max() == 1.0
    
    def test_inverse_transform(self, sample_data):
        """Test inverse transformation."""
        scaler = MinMaxScalerWrapper()
        scaled_data = scaler.fit_transform(sample_data)
        reconstructed_data = scaler.inverse_transform(scaled_data)
        
        # Verify reconstruction is close to original
        pd.testing.assert_frame_equal(reconstructed_data, sample_data, check_dtype=False, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__])