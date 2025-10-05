"""
DataCleaner module for handling data quality issues in fraud detection datasets.

This module provides the DataCleaner class for handling missing values, data type
conversions, validation, and outlier detection and treatment.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings


logger = logging.getLogger(__name__)


class DataCleaner:
    """
    DataCleaner class for handling data quality issues in fraud detection datasets.
    
    This class provides comprehensive data cleaning functionality including:
    - Missing value imputation with multiple strategies
    - Data type conversions and validation
    - Outlier detection and treatment
    - Data consistency checks and corrections
    """
    
    # Expected data types for fraud detection dataset
    EXPECTED_DTYPES = {
        'step': 'int64',
        'type': 'object',
        'amount': 'float64',
        'nameOrig': 'object',
        'oldbalanceOrg': 'float64',
        'newbalanceOrig': 'float64',
        'nameDest': 'object',
        'oldbalanceDest': 'float64',
        'newbalanceDest': 'float64',
        'isFraud': 'int64',
        'isFlaggedFraud': 'int64'
    }
    
    # Valid transaction types
    VALID_TRANSACTION_TYPES = {'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'}
    
    def __init__(self, 
                 missing_strategy: str = 'auto',
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 1.5,
                 validate_business_rules: bool = True):
        """
        Initialize DataCleaner with specified parameters.
        
        Args:
            missing_strategy: Strategy for handling missing values 
                            ('auto', 'mean', 'median', 'mode', 'knn', 'drop')
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            outlier_threshold: Threshold for outlier detection
            validate_business_rules: Whether to validate business logic rules
        """
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.validate_business_rules = validate_business_rules
        
        # Store imputers for consistent transformation
        self._imputers = {}
        self._label_encoders = {}
        
        # Track cleaning statistics
        self.cleaning_stats = {
            'missing_values_handled': 0,
            'outliers_detected': 0,
            'outliers_treated': 0,
            'data_type_conversions': 0,
            'business_rule_violations': 0
        }
    
    def clean_data(self, df: pd.DataFrame, fit_transformers: bool = True) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning on the dataset.
        
        Args:
            df: DataFrame to clean
            fit_transformers: Whether to fit new transformers or use existing ones
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        logger.info("Starting comprehensive data cleaning process")
        
        # Create a copy to avoid modifying the original
        df_cleaned = df.copy()
        
        # Reset cleaning statistics
        self.cleaning_stats = {key: 0 for key in self.cleaning_stats}
        
        # Step 1: Handle data type conversions and validation
        df_cleaned = self._convert_and_validate_dtypes(df_cleaned)
        
        # Step 2: Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned, fit_transformers)
        
        # Step 3: Detect and treat outliers
        df_cleaned = self._handle_outliers(df_cleaned)
        
        # Step 4: Validate business rules and fix inconsistencies
        if self.validate_business_rules:
            df_cleaned = self._validate_and_fix_business_rules(df_cleaned)
        
        # Step 5: Final validation
        self._perform_final_validation(df_cleaned)
        
        # Log cleaning summary
        self._log_cleaning_summary(df, df_cleaned)
        
        logger.info("Data cleaning process completed successfully")
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Optional[str] = None,
                            fit_transformers: bool = True) -> pd.DataFrame:
        """
        Handle missing values using specified strategy.
        
        Args:
            df: DataFrame with missing values
            strategy: Imputation strategy (overrides default if provided)
            fit_transformers: Whether to fit new transformers
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        strategy = strategy or self.missing_strategy
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        df_imputed = df.copy()
        
        # Get columns with missing values
        missing_cols = df_imputed.columns[df_imputed.isnull().any()].tolist()
        
        if not missing_cols:
            logger.info("No missing values found")
            return df_imputed
        
        logger.info(f"Found missing values in columns: {missing_cols}")
        
        for col in missing_cols:
            missing_count = df_imputed[col].isnull().sum()
            missing_pct = (missing_count / len(df_imputed)) * 100
            
            logger.info(f"Handling {missing_count} missing values ({missing_pct:.2f}%) in column '{col}'")
            
            if strategy == 'auto':
                # Use automatic strategy selection based on column type and missing percentage
                col_strategy = self._select_auto_strategy(df_imputed, col, missing_pct)
            else:
                col_strategy = strategy
            
            df_imputed = self._apply_imputation_strategy(df_imputed, col, col_strategy, fit_transformers)
            self.cleaning_stats['missing_values_handled'] += missing_count
        
        return df_imputed
    
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Detect outliers in numerical columns.
        
        Args:
            df: DataFrame to analyze
            columns: Specific columns to check (default: all numerical columns)
            
        Returns:
            Dict mapping column names to boolean arrays indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Detecting outliers in columns: {columns}")
        
        outliers = {}
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            if self.outlier_method == 'iqr':
                outlier_mask = self._detect_outliers_iqr(df[col])
            elif self.outlier_method == 'zscore':
                outlier_mask = self._detect_outliers_zscore(df[col])
            elif self.outlier_method == 'isolation':
                outlier_mask = self._detect_outliers_isolation(df[[col]])
            else:
                raise ValueError(f"Unknown outlier detection method: {self.outlier_method}")
            
            outliers[col] = outlier_mask
            outlier_count = outlier_mask.sum()
            outlier_pct = (outlier_count / len(df)) * 100
            
            logger.info(f"Detected {outlier_count} outliers ({outlier_pct:.2f}%) in column '{col}'")
            self.cleaning_stats['outliers_detected'] += outlier_count
        
        return outliers
    
    def treat_outliers(self, df: pd.DataFrame, 
                      outliers: Dict[str, np.ndarray],
                      treatment: str = 'clip') -> pd.DataFrame:
        """
        Treat detected outliers using specified method.
        
        Args:
            df: DataFrame containing outliers
            outliers: Dict mapping column names to outlier masks
            treatment: Treatment method ('clip', 'remove', 'transform', 'cap')
            
        Returns:
            pd.DataFrame: DataFrame with treated outliers
        """
        logger.info(f"Treating outliers using method: {treatment}")
        
        df_treated = df.copy()
        
        for col, outlier_mask in outliers.items():
            if not outlier_mask.any():
                continue
            
            outlier_count = outlier_mask.sum()
            logger.info(f"Treating {outlier_count} outliers in column '{col}'")
            
            if treatment == 'clip':
                df_treated = self._clip_outliers(df_treated, col, outlier_mask)
            elif treatment == 'remove':
                df_treated = df_treated[~outlier_mask]
            elif treatment == 'transform':
                df_treated = self._transform_outliers(df_treated, col)
            elif treatment == 'cap':
                df_treated = self._cap_outliers(df_treated, col, outlier_mask)
            else:
                raise ValueError(f"Unknown outlier treatment method: {treatment}")
            
            self.cleaning_stats['outliers_treated'] += outlier_count
        
        return df_treated
    
    def convert_data_types(self, df: pd.DataFrame, 
                          type_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Convert data types with proper error handling.
        
        Args:
            df: DataFrame to convert
            type_mapping: Custom type mapping (default: use expected dtypes)
            
        Returns:
            pd.DataFrame: DataFrame with converted types
        """
        type_mapping = type_mapping or self.EXPECTED_DTYPES
        logger.info("Converting data types")
        
        df_converted = df.copy()
        
        for col, target_dtype in type_mapping.items():
            if col not in df_converted.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            current_dtype = str(df_converted[col].dtype)
            
            if current_dtype == target_dtype:
                continue
            
            try:
                logger.debug(f"Converting column '{col}' from {current_dtype} to {target_dtype}")
                
                if target_dtype in ['int64', 'int32']:
                    # Handle integer conversion with NaN values
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('Int64')
                elif target_dtype in ['float64', 'float32']:
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                elif target_dtype == 'object':
                    df_converted[col] = df_converted[col].astype(str)
                elif target_dtype == 'category':
                    df_converted[col] = df_converted[col].astype('category')
                else:
                    df_converted[col] = df_converted[col].astype(target_dtype)
                
                self.cleaning_stats['data_type_conversions'] += 1
                logger.debug(f"Successfully converted column '{col}' to {target_dtype}")
                
            except Exception as e:
                logger.warning(f"Failed to convert column '{col}' to {target_dtype}: {e}")
        
        return df_converted
    
    def validate_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data consistency and business rules.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict containing validation results
        """
        logger.info("Validating data consistency and business rules")
        
        validation_results = {
            'total_rows': len(df),
            'issues_found': [],
            'warnings': [],
            'business_rule_violations': 0
        }
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_results['issues_found'].append(f"Found {duplicates} duplicate rows")
        
        # Validate transaction types
        if 'type' in df.columns:
            invalid_types = set(df['type'].unique()) - self.VALID_TRANSACTION_TYPES
            if invalid_types:
                validation_results['issues_found'].append(f"Invalid transaction types: {invalid_types}")
        
        # Validate balance calculations
        if all(col in df.columns for col in ['oldbalanceOrg', 'newbalanceOrig', 'amount', 'type']):
            balance_issues = self._validate_balance_calculations(df)
            validation_results['business_rule_violations'] += balance_issues
            if balance_issues > 0:
                validation_results['issues_found'].append(f"Found {balance_issues} balance calculation inconsistencies")
        
        # Validate amount ranges
        if 'amount' in df.columns:
            negative_amounts = (df['amount'] < 0).sum()
            if negative_amounts > 0:
                validation_results['warnings'].append(f"Found {negative_amounts} negative amounts")
        
        # Validate step ranges
        if 'step' in df.columns:
            invalid_steps = ((df['step'] < 1) | (df['step'] > 744)).sum()
            if invalid_steps > 0:
                validation_results['issues_found'].append(f"Found {invalid_steps} invalid step values")
        
        # Validate fraud flags consistency
        if all(col in df.columns for col in ['isFraud', 'isFlaggedFraud']):
            flag_issues = self._validate_fraud_flags(df)
            if flag_issues > 0:
                validation_results['warnings'].append(f"Found {flag_issues} fraud flag inconsistencies")
        
        logger.info(f"Validation completed. Found {len(validation_results['issues_found'])} issues and {len(validation_results['warnings'])} warnings")
        
        return validation_results
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get summary of cleaning operations performed.
        
        Returns:
            Dict containing cleaning statistics
        """
        return self.cleaning_stats.copy()
    
    # Private helper methods
    
    def _convert_and_validate_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert and validate data types."""
        logger.debug("Converting and validating data types")
        return self.convert_data_types(df)
    
    def _handle_missing_values(self, df: pd.DataFrame, fit_transformers: bool) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        return self.handle_missing_values(df, fit_transformers=fit_transformers)
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers."""
        outliers = self.detect_outliers(df)
        if any(mask.any() for mask in outliers.values()):
            return self.treat_outliers(df, outliers, treatment='clip')
        return df
    
    def _validate_and_fix_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix business rule violations."""
        logger.debug("Validating and fixing business rules")
        
        df_fixed = df.copy()
        
        # Fix balance calculation inconsistencies for CASH_IN and CASH_OUT
        if all(col in df_fixed.columns for col in ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']):
            df_fixed = self._fix_balance_inconsistencies(df_fixed)
        
        return df_fixed
    
    def _perform_final_validation(self, df: pd.DataFrame) -> None:
        """Perform final validation checks."""
        logger.debug("Performing final validation")
        
        # Check for remaining missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Still have {missing_values} missing values after cleaning")
        
        # Check data types
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != expected_dtype and not (expected_dtype == 'int64' and actual_dtype == 'Int64'):
                    logger.warning(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
    
    def _select_auto_strategy(self, df: pd.DataFrame, col: str, missing_pct: float) -> str:
        """Automatically select imputation strategy based on column characteristics."""
        if missing_pct > 50:
            return 'drop'
        
        if df[col].dtype in ['object', 'category']:
            return 'mode'
        elif df[col].dtype in ['int64', 'Int64', 'float64']:
            if missing_pct < 5:
                return 'mean'
            elif missing_pct < 20:
                return 'median'
            else:
                return 'knn'
        else:
            return 'mode'
    
    def _apply_imputation_strategy(self, df: pd.DataFrame, col: str, strategy: str, fit_transformers: bool) -> pd.DataFrame:
        """Apply specific imputation strategy to a column."""
        df_imputed = df.copy()
        
        if strategy == 'drop':
            df_imputed = df_imputed.dropna(subset=[col])
        elif strategy == 'mean':
            if fit_transformers or col not in self._imputers:
                self._imputers[col] = SimpleImputer(strategy='mean')
                df_imputed[col] = self._imputers[col].fit_transform(df_imputed[[col]]).ravel()
            else:
                df_imputed[col] = self._imputers[col].transform(df_imputed[[col]]).ravel()
        elif strategy == 'median':
            if fit_transformers or col not in self._imputers:
                self._imputers[col] = SimpleImputer(strategy='median')
                df_imputed[col] = self._imputers[col].fit_transform(df_imputed[[col]]).ravel()
            else:
                df_imputed[col] = self._imputers[col].transform(df_imputed[[col]]).ravel()
        elif strategy == 'mode':
            if fit_transformers or col not in self._imputers:
                self._imputers[col] = SimpleImputer(strategy='most_frequent')
                df_imputed[col] = self._imputers[col].fit_transform(df_imputed[[col]]).ravel()
            else:
                df_imputed[col] = self._imputers[col].transform(df_imputed[[col]]).ravel()
        elif strategy == 'knn':
            if fit_transformers or col not in self._imputers:
                # For KNN imputation, we need to handle categorical variables
                if df_imputed[col].dtype == 'object':
                    # Use mode for categorical variables
                    mode_value = df_imputed[col].mode().iloc[0] if not df_imputed[col].mode().empty else 'Unknown'
                    df_imputed[col] = df_imputed[col].fillna(mode_value)
                else:
                    self._imputers[col] = KNNImputer(n_neighbors=5)
                    df_imputed[col] = self._imputers[col].fit_transform(df_imputed[[col]]).ravel()
            else:
                if col in self._imputers:
                    df_imputed[col] = self._imputers[col].transform(df_imputed[[col]]).ravel()
        
        return df_imputed
    
    def _detect_outliers_iqr(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > self.outlier_threshold
    
    def _detect_outliers_isolation(self, df: pd.DataFrame) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(df)
        return outlier_labels == -1
    
    def _clip_outliers(self, df: pd.DataFrame, col: str, outlier_mask: np.ndarray) -> pd.DataFrame:
        """Clip outliers to reasonable bounds."""
        df_clipped = df.copy()
        
        Q1 = df_clipped[col].quantile(0.25)
        Q3 = df_clipped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        df_clipped[col] = df_clipped[col].clip(lower=lower_bound, upper=upper_bound)
        return df_clipped
    
    def _transform_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Transform outliers using log transformation."""
        df_transformed = df.copy()
        
        # Apply log transformation (add 1 to handle zero values)
        if (df_transformed[col] >= 0).all():
            df_transformed[col] = np.log1p(df_transformed[col])
        else:
            logger.warning(f"Cannot apply log transformation to column '{col}' with negative values")
        
        return df_transformed
    
    def _cap_outliers(self, df: pd.DataFrame, col: str, outlier_mask: np.ndarray) -> pd.DataFrame:
        """Cap outliers at percentile values."""
        df_capped = df.copy()
        
        lower_cap = df_capped[col].quantile(0.01)
        upper_cap = df_capped[col].quantile(0.99)
        
        df_capped.loc[outlier_mask, col] = df_capped.loc[outlier_mask, col].clip(
            lower=lower_cap, upper=upper_cap
        )
        
        return df_capped
    
    def _validate_balance_calculations(self, df: pd.DataFrame) -> int:
        """Validate balance calculations for consistency."""
        violations = 0
        
        # For CASH_IN transactions: newbalanceOrig should equal oldbalanceOrg + amount
        cash_in_mask = df['type'] == 'CASH_IN'
        if cash_in_mask.any():
            expected_balance = df.loc[cash_in_mask, 'oldbalanceOrg'] + df.loc[cash_in_mask, 'amount']
            actual_balance = df.loc[cash_in_mask, 'newbalanceOrig']
            violations += (abs(expected_balance - actual_balance) > 0.01).sum()
        
        # For CASH_OUT transactions: newbalanceOrig should equal oldbalanceOrg - amount
        cash_out_mask = df['type'] == 'CASH_OUT'
        if cash_out_mask.any():
            expected_balance = df.loc[cash_out_mask, 'oldbalanceOrg'] - df.loc[cash_out_mask, 'amount']
            actual_balance = df.loc[cash_out_mask, 'newbalanceOrig']
            violations += (abs(expected_balance - actual_balance) > 0.01).sum()
        
        return violations
    
    def _validate_fraud_flags(self, df: pd.DataFrame) -> int:
        """Validate fraud flag consistency."""
        # Check if flagged fraud transactions are actually fraudulent
        flagged_but_not_fraud = ((df['isFlaggedFraud'] == 1) & (df['isFraud'] == 0)).sum()
        return flagged_but_not_fraud
    
    def _fix_balance_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix balance calculation inconsistencies."""
        df_fixed = df.copy()
        
        # Fix CASH_IN balance inconsistencies
        cash_in_mask = df_fixed['type'] == 'CASH_IN'
        if cash_in_mask.any():
            expected_balance = df_fixed.loc[cash_in_mask, 'oldbalanceOrg'] + df_fixed.loc[cash_in_mask, 'amount']
            inconsistent_mask = cash_in_mask & (abs(df_fixed['newbalanceOrig'] - expected_balance) > 0.01)
            
            if inconsistent_mask.any():
                df_fixed.loc[inconsistent_mask, 'newbalanceOrig'] = expected_balance[inconsistent_mask]
                self.cleaning_stats['business_rule_violations'] += inconsistent_mask.sum()
        
        # Fix CASH_OUT balance inconsistencies
        cash_out_mask = df_fixed['type'] == 'CASH_OUT'
        if cash_out_mask.any():
            expected_balance = df_fixed.loc[cash_out_mask, 'oldbalanceOrg'] - df_fixed.loc[cash_out_mask, 'amount']
            inconsistent_mask = cash_out_mask & (abs(df_fixed['newbalanceOrig'] - expected_balance) > 0.01)
            
            if inconsistent_mask.any():
                df_fixed.loc[inconsistent_mask, 'newbalanceOrig'] = expected_balance[inconsistent_mask]
                self.cleaning_stats['business_rule_violations'] += inconsistent_mask.sum()
        
        return df_fixed
    
    def _log_cleaning_summary(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> None:
        """Log summary of cleaning operations."""
        logger.info("=" * 60)
        logger.info("DATA CLEANING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Original dataset shape: {df_original.shape}")
        logger.info(f"Cleaned dataset shape: {df_cleaned.shape}")
        logger.info(f"Rows removed: {len(df_original) - len(df_cleaned)}")
        
        for stat_name, count in self.cleaning_stats.items():
            if count > 0:
                logger.info(f"{stat_name.replace('_', ' ').title()}: {count:,}")
        
        # Log remaining data quality issues
        remaining_missing = df_cleaned.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"Remaining missing values: {remaining_missing}")
        
        logger.info("=" * 60)