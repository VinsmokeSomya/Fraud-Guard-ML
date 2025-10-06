"""
FeatureEngineering module for creating derived features in fraud detection datasets.

This module provides the FeatureEngineering class for creating derived features
from the original transaction data to improve model performance.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings


logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    FeatureEngineering class for creating derived features from transaction data.
    
    This class implements feature engineering techniques specifically designed for
    fraud detection, including:
    - Balance change calculations
    - Amount-to-balance ratios
    - Merchant identification flags
    - Time-based features from step column
    - Large transfer flags based on business rules
    """
    
    # Business rule thresholds
    LARGE_TRANSFER_THRESHOLD = 200000.0
    MERCHANT_PREFIX = 'M'
    
    def __init__(self, 
                 enable_time_features: bool = True,
                 enable_balance_features: bool = True,
                 enable_ratio_features: bool = True,
                 enable_merchant_features: bool = True,
                 enable_business_rule_features: bool = True):
        """
        Initialize FeatureEngineering with feature creation options.
        
        Args:
            enable_time_features: Whether to create time-based features
            enable_balance_features: Whether to create balance change features
            enable_ratio_features: Whether to create ratio features
            enable_merchant_features: Whether to create merchant identification features
            enable_business_rule_features: Whether to create business rule features
        """
        self.enable_time_features = enable_time_features
        self.enable_balance_features = enable_balance_features
        self.enable_ratio_features = enable_ratio_features
        self.enable_merchant_features = enable_merchant_features
        self.enable_business_rule_features = enable_business_rule_features
        
        # Store encoders and scalers for consistent transformation
        self._label_encoders = {}
        self._scalers = {}
        
        # Track feature engineering statistics
        self.feature_stats = {
            'original_features': 0,
            'engineered_features': 0,
            'total_features': 0,
            'features_created': []
        }
    
    def engineer_features(self, df: pd.DataFrame, fit_transformers: bool = True) -> pd.DataFrame:
        """
        Create derived features from the original transaction data.
        
        Args:
            df: DataFrame with original transaction data
            fit_transformers: Whether to fit new transformers or use existing ones
            
        Returns:
            pd.DataFrame: DataFrame with original and engineered features
        """
        logger.info("Starting feature engineering process")
        
        # Create a copy to avoid modifying the original
        df_engineered = df.copy()
        
        # Reset feature statistics
        self.feature_stats['original_features'] = len(df_engineered.columns)
        self.feature_stats['features_created'] = []
        
        # Step 1: Create balance change features
        if self.enable_balance_features:
            df_engineered = self.create_balance_features(df_engineered)
        
        # Step 2: Create ratio features
        if self.enable_ratio_features:
            df_engineered = self.create_ratio_features(df_engineered)
        
        # Step 3: Create merchant identification features
        if self.enable_merchant_features:
            df_engineered = self.create_merchant_features(df_engineered)
        
        # Step 4: Create time-based features
        if self.enable_time_features:
            df_engineered = self.create_time_features(df_engineered)
        
        # Step 5: Create business rule features
        if self.enable_business_rule_features:
            df_engineered = self.create_business_rule_features(df_engineered)
        
        # Update feature statistics
        self.feature_stats['engineered_features'] = len(df_engineered.columns) - self.feature_stats['original_features']
        self.feature_stats['total_features'] = len(df_engineered.columns)
        
        # Log feature engineering summary
        self._log_feature_summary(df, df_engineered)
        
        logger.info("Feature engineering process completed successfully")
        return df_engineered
    
    def create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create balance change features from transaction data.
        
        Args:
            df: DataFrame with balance columns
            
        Returns:
            pd.DataFrame: DataFrame with balance change features added
        """
        logger.info("Creating balance change features")
        
        df_balance = df.copy()
        
        # Calculate balance changes for origin account
        if all(col in df_balance.columns for col in ['newbalanceOrig', 'oldbalanceOrg']):
            df_balance['balance_change_orig'] = df_balance['newbalanceOrig'] - df_balance['oldbalanceOrg']
            self.feature_stats['features_created'].append('balance_change_orig')
            logger.debug("Created feature: balance_change_orig")
        
        # Calculate balance changes for destination account
        if all(col in df_balance.columns for col in ['newbalanceDest', 'oldbalanceDest']):
            df_balance['balance_change_dest'] = df_balance['newbalanceDest'] - df_balance['oldbalanceDest']
            self.feature_stats['features_created'].append('balance_change_dest')
            logger.debug("Created feature: balance_change_dest")
        
        # Calculate absolute balance changes
        if 'balance_change_orig' in df_balance.columns:
            df_balance['abs_balance_change_orig'] = df_balance['balance_change_orig'].abs()
            self.feature_stats['features_created'].append('abs_balance_change_orig')
            logger.debug("Created feature: abs_balance_change_orig")
        
        if 'balance_change_dest' in df_balance.columns:
            df_balance['abs_balance_change_dest'] = df_balance['balance_change_dest'].abs()
            self.feature_stats['features_created'].append('abs_balance_change_dest')
            logger.debug("Created feature: abs_balance_change_dest")
        
        # Calculate total balance change (sum of origin and destination changes)
        if all(col in df_balance.columns for col in ['balance_change_orig', 'balance_change_dest']):
            df_balance['total_balance_change'] = df_balance['balance_change_orig'] + df_balance['balance_change_dest']
            self.feature_stats['features_created'].append('total_balance_change')
            logger.debug("Created feature: total_balance_change")
        
        logger.info(f"Created {len([f for f in self.feature_stats['features_created'] if 'balance' in f])} balance features")
        return df_balance
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-to-balance ratio features.
        
        Args:
            df: DataFrame with amount and balance columns
            
        Returns:
            pd.DataFrame: DataFrame with ratio features added
        """
        logger.info("Creating ratio features")
        
        df_ratio = df.copy()
        
        # Amount to origin balance ratio
        if all(col in df_ratio.columns for col in ['amount', 'oldbalanceOrg']):
            # Avoid division by zero
            df_ratio['amount_to_orig_balance_ratio'] = np.where(
                df_ratio['oldbalanceOrg'] > 0,
                df_ratio['amount'] / df_ratio['oldbalanceOrg'],
                0
            )
            self.feature_stats['features_created'].append('amount_to_orig_balance_ratio')
            logger.debug("Created feature: amount_to_orig_balance_ratio")
        
        # Amount to destination balance ratio
        if all(col in df_ratio.columns for col in ['amount', 'oldbalanceDest']):
            df_ratio['amount_to_dest_balance_ratio'] = np.where(
                df_ratio['oldbalanceDest'] > 0,
                df_ratio['amount'] / df_ratio['oldbalanceDest'],
                0
            )
            self.feature_stats['features_created'].append('amount_to_dest_balance_ratio')
            logger.debug("Created feature: amount_to_dest_balance_ratio")
        
        # Amount to new origin balance ratio
        if all(col in df_ratio.columns for col in ['amount', 'newbalanceOrig']):
            df_ratio['amount_to_new_orig_balance_ratio'] = np.where(
                df_ratio['newbalanceOrig'] > 0,
                df_ratio['amount'] / df_ratio['newbalanceOrig'],
                0
            )
            self.feature_stats['features_created'].append('amount_to_new_orig_balance_ratio')
            logger.debug("Created feature: amount_to_new_orig_balance_ratio")
        
        # Amount to new destination balance ratio
        if all(col in df_ratio.columns for col in ['amount', 'newbalanceDest']):
            df_ratio['amount_to_new_dest_balance_ratio'] = np.where(
                df_ratio['newbalanceDest'] > 0,
                df_ratio['amount'] / df_ratio['newbalanceDest'],
                0
            )
            self.feature_stats['features_created'].append('amount_to_new_dest_balance_ratio')
            logger.debug("Created feature: amount_to_new_dest_balance_ratio")
        
        # Balance ratio (origin to destination)
        if all(col in df_ratio.columns for col in ['oldbalanceOrg', 'oldbalanceDest']):
            df_ratio['orig_to_dest_balance_ratio'] = np.where(
                df_ratio['oldbalanceDest'] > 0,
                df_ratio['oldbalanceOrg'] / df_ratio['oldbalanceDest'],
                0
            )
            self.feature_stats['features_created'].append('orig_to_dest_balance_ratio')
            logger.debug("Created feature: orig_to_dest_balance_ratio")
        
        logger.info(f"Created {len([f for f in self.feature_stats['features_created'] if 'ratio' in f])} ratio features")
        return df_ratio
    
    def create_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create merchant identification features.
        
        Args:
            df: DataFrame with customer name columns
            
        Returns:
            pd.DataFrame: DataFrame with merchant features added
        """
        logger.info("Creating merchant identification features")
        
        df_merchant = df.copy()
        
        # Check if destination is a merchant (starts with 'M')
        if 'nameDest' in df_merchant.columns:
            df_merchant['is_merchant_dest'] = df_merchant['nameDest'].str.startswith(self.MERCHANT_PREFIX).astype(int)
            self.feature_stats['features_created'].append('is_merchant_dest')
            logger.debug("Created feature: is_merchant_dest")
        
        # Check if origin is a merchant (starts with 'M')
        if 'nameOrig' in df_merchant.columns:
            df_merchant['is_merchant_orig'] = df_merchant['nameOrig'].str.startswith(self.MERCHANT_PREFIX).astype(int)
            self.feature_stats['features_created'].append('is_merchant_orig')
            logger.debug("Created feature: is_merchant_orig")
        
        # Check if transaction is between merchants
        if all(col in df_merchant.columns for col in ['is_merchant_orig', 'is_merchant_dest']):
            df_merchant['is_merchant_to_merchant'] = (
                (df_merchant['is_merchant_orig'] == 1) & 
                (df_merchant['is_merchant_dest'] == 1)
            ).astype(int)
            self.feature_stats['features_created'].append('is_merchant_to_merchant')
            logger.debug("Created feature: is_merchant_to_merchant")
        
        # Check if transaction involves any merchant
        if all(col in df_merchant.columns for col in ['is_merchant_orig', 'is_merchant_dest']):
            df_merchant['involves_merchant'] = (
                (df_merchant['is_merchant_orig'] == 1) | 
                (df_merchant['is_merchant_dest'] == 1)
            ).astype(int)
            self.feature_stats['features_created'].append('involves_merchant')
            logger.debug("Created feature: involves_merchant")
        
        logger.info(f"Created {len([f for f in self.feature_stats['features_created'] if 'merchant' in f])} merchant features")
        return df_merchant
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from the step column.
        
        Args:
            df: DataFrame with step column
            
        Returns:
            pd.DataFrame: DataFrame with time features added
        """
        logger.info("Creating time-based features")
        
        df_time = df.copy()
        
        if 'step' not in df_time.columns:
            logger.warning("Step column not found, skipping time feature creation")
            return df_time
        
        # Hour of day (assuming each step represents 1 hour)
        df_time['hour_of_day'] = df_time['step'] % 24
        self.feature_stats['features_created'].append('hour_of_day')
        logger.debug("Created feature: hour_of_day")
        
        # Day of month (assuming 24 steps per day)
        df_time['day_of_month'] = (df_time['step'] - 1) // 24 + 1
        self.feature_stats['features_created'].append('day_of_month')
        logger.debug("Created feature: day_of_month")
        
        # Week of month
        df_time['week_of_month'] = ((df_time['day_of_month'] - 1) // 7) + 1
        self.feature_stats['features_created'].append('week_of_month')
        logger.debug("Created feature: week_of_month")
        
        # Is weekend (assuming Saturday=6, Sunday=0 in day_of_week)
        df_time['day_of_week'] = ((df_time['step'] - 1) // 24) % 7
        df_time['is_weekend'] = ((df_time['day_of_week'] == 0) | (df_time['day_of_week'] == 6)).astype(int)
        self.feature_stats['features_created'].extend(['day_of_week', 'is_weekend'])
        logger.debug("Created features: day_of_week, is_weekend")
        
        # Time period categories
        df_time['time_period'] = pd.cut(
            df_time['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        self.feature_stats['features_created'].append('time_period')
        logger.debug("Created feature: time_period")
        
        # Business hours flag (9 AM to 5 PM)
        df_time['is_business_hours'] = (
            (df_time['hour_of_day'] >= 9) & 
            (df_time['hour_of_day'] <= 17)
        ).astype(int)
        self.feature_stats['features_created'].append('is_business_hours')
        logger.debug("Created feature: is_business_hours")
        
        logger.info(f"Created {len([f for f in self.feature_stats['features_created'] if any(t in f for t in ['hour', 'day', 'week', 'time', 'business'])])} time features")
        return df_time
    
    def create_business_rule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on business rules for fraud detection.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            pd.DataFrame: DataFrame with business rule features added
        """
        logger.info("Creating business rule features")
        
        df_business = df.copy()
        
        # Large transfer flag (amount > 200,000)
        if 'amount' in df_business.columns:
            df_business['is_large_transfer'] = (df_business['amount'] > self.LARGE_TRANSFER_THRESHOLD).astype(int)
            self.feature_stats['features_created'].append('is_large_transfer')
            logger.debug(f"Created feature: is_large_transfer (threshold: {self.LARGE_TRANSFER_THRESHOLD})")
        
        # Large transfer and TRANSFER type combination
        if all(col in df_business.columns for col in ['amount', 'type']):
            df_business['is_large_transfer_type'] = (
                (df_business['amount'] > self.LARGE_TRANSFER_THRESHOLD) & 
                (df_business['type'] == 'TRANSFER')
            ).astype(int)
            self.feature_stats['features_created'].append('is_large_transfer_type')
            logger.debug("Created feature: is_large_transfer_type")
        
        # Zero balance after transaction (potential account emptying)
        if 'newbalanceOrig' in df_business.columns:
            df_business['is_zero_balance_orig'] = (df_business['newbalanceOrig'] == 0).astype(int)
            self.feature_stats['features_created'].append('is_zero_balance_orig')
            logger.debug("Created feature: is_zero_balance_orig")
        
        if 'newbalanceDest' in df_business.columns:
            df_business['is_zero_balance_dest'] = (df_business['newbalanceDest'] == 0).astype(int)
            self.feature_stats['features_created'].append('is_zero_balance_dest')
            logger.debug("Created feature: is_zero_balance_dest")
        
        # Round amount transactions (potentially suspicious)
        if 'amount' in df_business.columns:
            df_business['is_round_amount'] = (df_business['amount'] % 1000 == 0).astype(int)
            self.feature_stats['features_created'].append('is_round_amount')
            logger.debug("Created feature: is_round_amount")
        
        # High-risk transaction types (TRANSFER and CASH_OUT are more prone to fraud)
        if 'type' in df_business.columns:
            high_risk_types = ['TRANSFER', 'CASH_OUT']
            df_business['is_high_risk_type'] = df_business['type'].isin(high_risk_types).astype(int)
            self.feature_stats['features_created'].append('is_high_risk_type')
            logger.debug("Created feature: is_high_risk_type")
        
        # Amount equals old balance (potential account emptying)
        if all(col in df_business.columns for col in ['amount', 'oldbalanceOrg']):
            df_business['amount_equals_old_balance'] = (
                abs(df_business['amount'] - df_business['oldbalanceOrg']) < 0.01
            ).astype(int)
            self.feature_stats['features_created'].append('amount_equals_old_balance')
            logger.debug("Created feature: amount_equals_old_balance")
        
        # Inconsistent balance flag (newbalance doesn't match expected calculation)
        if all(col in df_business.columns for col in ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']):
            df_business['has_balance_inconsistency'] = self._detect_balance_inconsistencies(df_business)
            self.feature_stats['features_created'].append('has_balance_inconsistency')
            logger.debug("Created feature: has_balance_inconsistency")
        
        logger.info(f"Created {len([f for f in self.feature_stats['features_created'] if any(b in f for b in ['large', 'zero', 'round', 'risk', 'equals', 'inconsistency'])])} business rule features")
        return df_business
    
    def get_feature_importance_by_category(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features by their type for analysis.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dict mapping feature categories to feature lists
        """
        feature_categories = {
            'original': [],
            'balance': [],
            'ratio': [],
            'merchant': [],
            'time': [],
            'business_rule': []
        }
        
        original_features = [
            'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
            'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud'
        ]
        
        for col in df.columns:
            if col in original_features:
                feature_categories['original'].append(col)
            elif 'balance' in col:
                feature_categories['balance'].append(col)
            elif 'ratio' in col:
                feature_categories['ratio'].append(col)
            elif 'merchant' in col:
                feature_categories['merchant'].append(col)
            elif any(t in col for t in ['hour', 'day', 'week', 'time', 'business_hours']):
                feature_categories['time'].append(col)
            elif any(b in col for b in ['large', 'zero', 'round', 'risk', 'equals', 'inconsistency']):
                feature_categories['business_rule'].append(col)
        
        return feature_categories
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature engineering operations.
        
        Returns:
            Dict containing feature engineering statistics
        """
        return self.feature_stats.copy()
    
    # Private helper methods
    
    def _create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create balance-related features."""
        return self.create_balance_features(df)
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-related features."""
        return self.create_ratio_features(df)
    
    def _create_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create merchant-related features."""
        return self.create_merchant_features(df)
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-related features."""
        return self.create_time_features(df)
    
    def _create_business_rule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create business rule features."""
        return self.create_business_rule_features(df)
    
    def _detect_balance_inconsistencies(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect balance calculation inconsistencies.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            pd.Series: Boolean series indicating inconsistencies
        """
        inconsistencies = pd.Series(False, index=df.index)
        
        # Check CASH_IN transactions
        cash_in_mask = df['type'] == 'CASH_IN'
        if cash_in_mask.any():
            expected_balance = df.loc[cash_in_mask, 'oldbalanceOrg'] + df.loc[cash_in_mask, 'amount']
            actual_balance = df.loc[cash_in_mask, 'newbalanceOrig']
            inconsistencies.loc[cash_in_mask] = abs(expected_balance - actual_balance) > 0.01
        
        # Check CASH_OUT transactions
        cash_out_mask = df['type'] == 'CASH_OUT'
        if cash_out_mask.any():
            expected_balance = df.loc[cash_out_mask, 'oldbalanceOrg'] - df.loc[cash_out_mask, 'amount']
            actual_balance = df.loc[cash_out_mask, 'newbalanceOrig']
            inconsistencies.loc[cash_out_mask] = abs(expected_balance - actual_balance) > 0.01
        
        # Check PAYMENT transactions (balance should decrease)
        payment_mask = df['type'] == 'PAYMENT'
        if payment_mask.any():
            expected_balance = df.loc[payment_mask, 'oldbalanceOrg'] - df.loc[payment_mask, 'amount']
            actual_balance = df.loc[payment_mask, 'newbalanceOrig']
            inconsistencies.loc[payment_mask] = abs(expected_balance - actual_balance) > 0.01
        
        # Check DEBIT transactions (balance should decrease)
        debit_mask = df['type'] == 'DEBIT'
        if debit_mask.any():
            expected_balance = df.loc[debit_mask, 'oldbalanceOrg'] - df.loc[debit_mask, 'amount']
            actual_balance = df.loc[debit_mask, 'newbalanceOrig']
            inconsistencies.loc[debit_mask] = abs(expected_balance - actual_balance) > 0.01
        
        # Check TRANSFER transactions (origin balance should decrease)
        transfer_mask = df['type'] == 'TRANSFER'
        if transfer_mask.any():
            expected_balance = df.loc[transfer_mask, 'oldbalanceOrg'] - df.loc[transfer_mask, 'amount']
            actual_balance = df.loc[transfer_mask, 'newbalanceOrig']
            inconsistencies.loc[transfer_mask] = abs(expected_balance - actual_balance) > 0.01
        
        return inconsistencies.astype(int)
    
    def _log_feature_summary(self, df_original: pd.DataFrame, df_engineered: pd.DataFrame) -> None:
        """Log summary of feature engineering operations."""
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Original features: {self.feature_stats['original_features']}")
        logger.info(f"Engineered features: {self.feature_stats['engineered_features']}")
        logger.info(f"Total features: {self.feature_stats['total_features']}")
        
        if self.feature_stats['features_created']:
            logger.info("Features created:")
            for feature in self.feature_stats['features_created']:
                logger.info(f"  - {feature}")
        
        # Log feature categories
        feature_categories = self.get_feature_importance_by_category(df_engineered)
        for category, features in feature_categories.items():
            if features:
                logger.info(f"{category.title()} features ({len(features)}): {', '.join(features[:5])}")
                if len(features) > 5:
                    logger.info(f"  ... and {len(features) - 5} more")
        
        logger.info("=" * 60)