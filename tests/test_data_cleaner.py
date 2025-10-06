"""
Unit tests for DataCleaner class.

This module contains comprehensive tests for data cleaning functionality
including missing value handling, outlier detection, and data validation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from src.data.data_cleaner import DataCleaner


class TestDataCleaner:
    """Test class for DataCleaner functionality."""
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing."""
        return pd.DataFrame({
            'step': [1, 2, 3, 4, 5, 6, 7],
            'type': ['CASH_IN', 'CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT', 'TRANSFER', 'CASH_OUT'],
            'amount': [1000.0, 500.0, 2000.0, 100.0, 50.0, 3000.0, 1500.0],
            'nameOrig': ['C123456789', 'C987654321', 'C111111111', 'C222222222', 'C333333333', 'C444444444', 'C555555555'],
            'oldbalanceOrg': [2000.0, 1000.0, 3000.0, 500.0, 200.0, 4000.0, 2000.0],
            'newbalanceOrig': [3000.0, 500.0, 1000.0, 400.0, 150.0, 1000.0, 500.0],
            'nameDest': ['M123456789', 'C111111111', 'M987654321', 'M222222222', 'M333333333', 'C666666666', 'M777777777'],
            'oldbalanceDest': [5000.0, 0.0, 10000.0, 0.0, 0.0, 5000.0, 10000.0],
            'newbalanceDest': [6000.0, 500.0, 12000.0, 100.0, 50.0, 8000.0, 11500.0],
            'isFraud': [0, 0, 1, 0, 0, 1, 0],
            'isFlaggedFraud': [0, 0, 1, 0, 0, 1, 0]
        })
    
    @pytest.fixture
    def data_with_missing_values(self):
        """Create data with missing values for testing."""
        return pd.DataFrame({
            'step': [1, 2, np.nan, 4, 5],
            'type': ['CASH_IN', 'CASH_OUT', None, 'PAYMENT', 'DEBIT'],
            'amount': [1000.0, np.nan, 2000.0, 100.0, 50.0],
            'oldbalanceOrg': [2000.0, 1000.0, 3000.0, np.nan, 200.0],
            'newbalanceOrig': [3000.0, 500.0, 1000.0, 400.0, 150.0],
            'isFraud': [0, 0, 1, 0, 0]
        })
    
    @pytest.fixture
    def data_with_outliers(self):
        """Create data with outliers for testing."""
        return pd.DataFrame({
            'amount': [100.0, 200.0, 300.0, 400.0, 500.0, 10000.0, 600.0, 700.0],  # 10000 is outlier
            'balance': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 100000.0, 8000.0],  # 100000 is outlier
            'step': [1, 2, 3, 4, 5, 6, 7, 8]
        })
    
    @pytest.fixture
    def data_cleaner(self):
        """Create DataCleaner instance for testing."""
        return DataCleaner()
    
    def test_initialization_default_parameters(self):
        """Test DataCleaner initialization with default parameters."""
        cleaner = DataCleaner()
        
        assert cleaner.missing_strategy == 'auto'
        assert cleaner.outlier_method == 'iqr'
        assert cleaner.outlier_threshold == 1.5
        assert cleaner.validate_business_rules is True
        assert cleaner._imputers == {}
        assert cleaner._label_encoders == {}
    
    def test_initialization_custom_parameters(self):
        """Test DataCleaner initialization with custom parameters."""
        cleaner = DataCleaner(
            missing_strategy='median',
            outlier_method='zscore',
            outlier_threshold=2.0,
            validate_business_rules=False
        )
        
        assert cleaner.missing_strategy == 'median'
        assert cleaner.outlier_method == 'zscore'
        assert cleaner.outlier_threshold == 2.0
        assert cleaner.validate_business_rules is False
    
    def test_convert_data_types(self, data_cleaner, sample_transaction_data):
        """Test data type conversion functionality."""
        # Convert some columns to wrong types first
        test_data = sample_transaction_data.copy()
        test_data['step'] = test_data['step'].astype(str)
        test_data['amount'] = test_data['amount'].astype(str)
        
        result = data_cleaner.convert_data_types(test_data)
        
        # Verify conversions
        assert result['step'].dtype in ['int64', 'Int64']
        assert result['amount'].dtype in ['float64']
        assert result['type'].dtype == 'object'
        
        # Verify values are preserved
        assert result['step'].iloc[0] == 1
        assert result['amount'].iloc[0] == 1000.0
    
    def test_handle_missing_values_mean_strategy(self, data_cleaner, data_with_missing_values):
        """Test missing value handling with mean strategy."""
        # Test only on numerical columns
        numerical_data = data_with_missing_values[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']].copy()
        result = data_cleaner.handle_missing_values(numerical_data, strategy='mean')
        
        # Verify no missing values remain in numerical columns
        assert not result['amount'].isna().any()
        assert not result['oldbalanceOrg'].isna().any()
        
        # Verify mean imputation was applied correctly
        original_mean = numerical_data['amount'].mean()
        assert result.loc[1, 'amount'] == original_mean  # Missing value should be replaced with mean
    
    def test_handle_missing_values_median_strategy(self, data_cleaner, data_with_missing_values):
        """Test missing value handling with median strategy."""
        # Test only on numerical columns
        numerical_data = data_with_missing_values[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']].copy()
        result = data_cleaner.handle_missing_values(numerical_data, strategy='median')
        
        # Verify no missing values remain in numerical columns
        assert not result['amount'].isna().any()
        assert not result['oldbalanceOrg'].isna().any()
        
        # Verify median imputation was applied correctly
        original_median = numerical_data['amount'].median()
        assert result.loc[1, 'amount'] == original_median
    
    def test_handle_missing_values_mode_strategy(self, data_cleaner, data_with_missing_values):
        """Test missing value handling with mode strategy."""
        # Test only on categorical columns, but convert None to np.nan first
        categorical_data = data_with_missing_values[['type']].copy()
        categorical_data['type'] = categorical_data['type'].replace({None: np.nan})
        
        result = data_cleaner.handle_missing_values(categorical_data, strategy='mode')
        
        # Verify no missing values remain in categorical columns
        assert not result['type'].isna().any()
        
        # Verify mode imputation was applied correctly
        original_mode = categorical_data['type'].mode().iloc[0]
        assert result.loc[2, 'type'] == original_mode
    
    def test_handle_missing_values_drop_strategy(self, data_cleaner, data_with_missing_values):
        """Test missing value handling with drop strategy."""
        original_length = len(data_with_missing_values)
        result = data_cleaner.handle_missing_values(data_with_missing_values, strategy='drop')
        
        # Verify rows with missing values were dropped
        assert len(result) < original_length
        assert not result.isna().any().any()
    
    def test_auto_strategy_selection(self, data_cleaner, data_with_missing_values):
        """Test automatic strategy selection based on column characteristics."""
        # Test with different missing percentages and column types
        
        # Numerical column with low missing percentage should use mean or median
        strategy = data_cleaner._select_auto_strategy(data_with_missing_values, 'amount', 5.0)
        assert strategy in ['mean', 'median']  # Both are acceptable for low missing percentage
        
        # Numerical column with medium missing percentage should use median
        strategy = data_cleaner._select_auto_strategy(data_with_missing_values, 'amount', 15.0)
        assert strategy == 'median'
        
        # Numerical column with high missing percentage should use knn
        strategy = data_cleaner._select_auto_strategy(data_with_missing_values, 'amount', 25.0)
        assert strategy == 'knn'
        
        # Categorical column should use mode
        strategy = data_cleaner._select_auto_strategy(data_with_missing_values, 'type', 10.0)
        assert strategy == 'mode'
        
        # Very high missing percentage should drop
        strategy = data_cleaner._select_auto_strategy(data_with_missing_values, 'amount', 60.0)
        assert strategy == 'drop'
    
    def test_detect_outliers_iqr_method(self, data_cleaner, data_with_outliers):
        """Test outlier detection using IQR method."""
        data_cleaner.outlier_method = 'iqr'
        outliers = data_cleaner.detect_outliers(data_with_outliers, columns=['amount', 'balance'])
        
        # Verify outliers were detected
        assert 'amount' in outliers
        assert 'balance' in outliers
        
        # Verify specific outliers
        assert outliers['amount'][5] == True  # 10000 should be detected as outlier
        assert outliers['balance'][6] == True  # 100000 should be detected as outlier
        
        # Verify normal values are not flagged
        assert outliers['amount'][0] == False  # 100 should not be outlier
        assert outliers['balance'][0] == False  # 1000 should not be outlier
    
    def test_detect_outliers_zscore_method(self, data_with_outliers):
        """Test outlier detection using Z-score method."""
        cleaner = DataCleaner(outlier_method='zscore', outlier_threshold=2.0)
        outliers = cleaner.detect_outliers(data_with_outliers, columns=['amount', 'balance'])
        
        # Verify outliers were detected
        assert 'amount' in outliers
        assert 'balance' in outliers
        
        # Verify extreme values are detected
        assert outliers['amount'][5] == True  # 10000 should be detected
        assert outliers['balance'][6] == True  # 100000 should be detected
    
    def test_treat_outliers_clip_method(self, data_cleaner, data_with_outliers):
        """Test outlier treatment using clipping method."""
        outliers = data_cleaner.detect_outliers(data_with_outliers, columns=['amount'])
        result = data_cleaner.treat_outliers(data_with_outliers, outliers, treatment='clip')
        
        # Verify outliers were clipped
        original_max = data_with_outliers['amount'].max()
        result_max = result['amount'].max()
        assert result_max < original_max
        
        # Verify normal values are unchanged
        assert result.loc[0, 'amount'] == data_with_outliers.loc[0, 'amount']
    
    def test_treat_outliers_remove_method(self, data_cleaner, data_with_outliers):
        """Test outlier treatment using removal method."""
        outliers = data_cleaner.detect_outliers(data_with_outliers, columns=['amount'])
        result = data_cleaner.treat_outliers(data_with_outliers, outliers, treatment='remove')
        
        # Verify outliers were removed
        assert len(result) < len(data_with_outliers)
        assert result['amount'].max() < data_with_outliers['amount'].max()
    
    def test_validate_data_consistency(self, data_cleaner, sample_transaction_data):
        """Test data consistency validation."""
        validation_results = data_cleaner.validate_data_consistency(sample_transaction_data)
        
        # Verify validation structure
        assert 'total_rows' in validation_results
        assert 'issues_found' in validation_results
        assert 'warnings' in validation_results
        assert 'business_rule_violations' in validation_results
        
        # Verify total rows count
        assert validation_results['total_rows'] == len(sample_transaction_data)
        
        # Should be lists
        assert isinstance(validation_results['issues_found'], list)
        assert isinstance(validation_results['warnings'], list)
    
    def test_validate_balance_calculations(self, data_cleaner):
        """Test balance calculation validation."""
        # Create data with known balance inconsistencies
        inconsistent_data = pd.DataFrame({
            'type': ['CASH_IN', 'CASH_OUT', 'TRANSFER'],
            'amount': [1000.0, 500.0, 2000.0],
            'oldbalanceOrg': [2000.0, 1000.0, 3000.0],
            'newbalanceOrig': [2500.0, 1000.0, 1000.0]  # Inconsistent values
        })
        
        violations = data_cleaner._validate_balance_calculations(inconsistent_data)
        
        # Should detect violations
        assert violations > 0
    
    def test_fix_balance_inconsistencies(self, data_cleaner):
        """Test fixing balance calculation inconsistencies."""
        # Create data with known inconsistencies
        inconsistent_data = pd.DataFrame({
            'type': ['CASH_IN', 'CASH_OUT'],
            'amount': [1000.0, 500.0],
            'oldbalanceOrg': [2000.0, 1000.0],
            'newbalanceOrig': [2500.0, 1000.0]  # Should be 3000 and 500
        })
        
        result = data_cleaner._fix_balance_inconsistencies(inconsistent_data)
        
        # Verify fixes
        assert result.loc[0, 'newbalanceOrig'] == 3000.0  # 2000 + 1000
        assert result.loc[1, 'newbalanceOrig'] == 500.0   # 1000 - 500
    
    def test_clean_data_full_pipeline(self, data_cleaner, data_with_missing_values):
        """Test the complete data cleaning pipeline."""
        result = data_cleaner.clean_data(data_with_missing_values)
        
        # Verify most missing values are handled (some categorical might remain with None)
        # Check that numerical columns don't have missing values
        numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']
        for col in numerical_cols:
            if col in result.columns:
                assert not result[col].isna().any(), f"Column {col} still has missing values"
        
        # Verify data types are correct
        if 'step' in result.columns:
            assert result['step'].dtype in ['int64', 'Int64', 'float64']
        if 'amount' in result.columns:
            assert result['amount'].dtype == 'float64'
        
        # Verify cleaning stats were updated
        stats = data_cleaner.get_cleaning_summary()
        assert stats['missing_values_handled'] > 0
    
    def test_clean_data_with_business_rules_disabled(self, data_with_missing_values):
        """Test data cleaning with business rule validation disabled."""
        cleaner = DataCleaner(validate_business_rules=False)
        result = cleaner.clean_data(data_with_missing_values)
        
        # Should still clean missing values and outliers for numerical columns
        numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']
        for col in numerical_cols:
            if col in result.columns:
                assert not result[col].isna().any(), f"Column {col} still has missing values"
        
        # Business rule violations should be 0
        stats = cleaner.get_cleaning_summary()
        assert stats['business_rule_violations'] == 0
    
    def test_validate_fraud_flags(self, data_cleaner):
        """Test fraud flag consistency validation."""
        # Create data with inconsistent fraud flags
        inconsistent_data = pd.DataFrame({
            'isFraud': [0, 1, 0],
            'isFlaggedFraud': [1, 1, 0]  # First row: flagged but not fraud
        })
        
        violations = data_cleaner._validate_fraud_flags(inconsistent_data)
        
        # Should detect one violation
        assert violations == 1
    
    def test_handle_invalid_transaction_types(self, data_cleaner):
        """Test handling of invalid transaction types."""
        invalid_data = pd.DataFrame({
            'type': ['CASH_IN', 'INVALID_TYPE', 'TRANSFER'],
            'amount': [1000.0, 500.0, 2000.0]
        })
        
        validation_results = data_cleaner.validate_data_consistency(invalid_data)
        
        # Should detect invalid transaction type
        issues = validation_results['issues_found']
        assert any('Invalid transaction types' in issue for issue in issues)
    
    def test_handle_negative_amounts(self, data_cleaner):
        """Test handling of negative amounts."""
        negative_data = pd.DataFrame({
            'amount': [1000.0, -500.0, 2000.0],  # Negative amount
            'type': ['CASH_IN', 'CASH_OUT', 'TRANSFER']
        })
        
        validation_results = data_cleaner.validate_data_consistency(negative_data)
        
        # Should warn about negative amounts
        warnings = validation_results['warnings']
        assert any('negative amounts' in warning for warning in warnings)
    
    def test_handle_invalid_step_values(self, data_cleaner):
        """Test handling of invalid step values."""
        invalid_steps_data = pd.DataFrame({
            'step': [0, 500, 1000],  # 0 and 1000 are invalid (should be 1-744)
            'amount': [1000.0, 500.0, 2000.0]
        })
        
        validation_results = data_cleaner.validate_data_consistency(invalid_steps_data)
        
        # Should detect invalid step values
        issues = validation_results['issues_found']
        assert any('invalid step values' in issue for issue in issues)
    
    def test_get_cleaning_summary(self, data_cleaner, data_with_missing_values):
        """Test getting cleaning summary."""
        data_cleaner.clean_data(data_with_missing_values)
        summary = data_cleaner.get_cleaning_summary()
        
        # Verify summary structure
        expected_keys = [
            'missing_values_handled', 'outliers_detected', 'outliers_treated',
            'data_type_conversions', 'business_rule_violations'
        ]
        
        for key in expected_keys:
            assert key in summary
            assert isinstance(summary[key], (int, np.integer))
    
    def test_handle_empty_dataframe(self, data_cleaner):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = data_cleaner.clean_data(empty_df)
        
        # Should return empty DataFrame without errors
        assert len(result) == 0
        assert len(result.columns) == 0
    
    def test_handle_dataframe_with_all_missing_values(self, data_cleaner):
        """Test handling of DataFrame with all missing values."""
        all_missing_df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [None, None, None]
        })
        
        # Should handle gracefully - with auto strategy, it should drop columns with too many missing values
        try:
            result = data_cleaner.clean_data(all_missing_df)
            # If it succeeds, should be a DataFrame (might be empty)
            assert isinstance(result, pd.DataFrame)
        except ValueError:
            # It's acceptable to fail with all missing data
            pass
    
    def test_knn_imputation_with_categorical_data(self, data_cleaner):
        """Test KNN imputation with categorical data."""
        mixed_data = pd.DataFrame({
            'numerical': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical': ['A', 'B', None, 'A', 'B']
        })
        
        result = data_cleaner.handle_missing_values(mixed_data, strategy='knn')
        
        # Should handle both numerical and categorical missing values
        assert not result['numerical'].isna().any()
        assert not result['categorical'].isna().any()
    
    def test_outlier_detection_with_isolation_forest(self, data_with_outliers):
        """Test outlier detection using Isolation Forest."""
        cleaner = DataCleaner(outlier_method='isolation')
        outliers = cleaner.detect_outliers(data_with_outliers, columns=['amount'])
        
        # Should detect outliers
        assert 'amount' in outliers
        assert outliers['amount'].any()  # At least some outliers should be detected
    
    def test_transform_outliers_log_transformation(self, data_cleaner, data_with_outliers):
        """Test outlier treatment using log transformation."""
        outliers = data_cleaner.detect_outliers(data_with_outliers, columns=['amount'])
        result = data_cleaner.treat_outliers(data_with_outliers, outliers, treatment='transform')
        
        # Verify transformation was applied
        # Log transformation should reduce the impact of outliers
        original_std = data_with_outliers['amount'].std()
        transformed_std = result['amount'].std()
        assert transformed_std < original_std
    
    def test_cap_outliers_treatment(self, data_cleaner, data_with_outliers):
        """Test outlier treatment using capping method."""
        outliers = data_cleaner.detect_outliers(data_with_outliers, columns=['amount'])
        result = data_cleaner.treat_outliers(data_with_outliers, outliers, treatment='cap')
        
        # Verify capping was applied
        original_max = data_with_outliers['amount'].max()
        capped_max = result['amount'].max()
        assert capped_max <= original_max
        
        # Verify normal values are unchanged
        assert result.loc[0, 'amount'] == data_with_outliers.loc[0, 'amount']


if __name__ == '__main__':
    pytest.main([__file__])