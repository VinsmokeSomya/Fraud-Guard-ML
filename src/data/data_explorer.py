"""
Data exploration utilities for fraud detection dataset analysis.

This module provides comprehensive data exploration functions to understand
transaction patterns, fraud distribution, and data quality issues.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


logger = logging.getLogger(__name__)


class DataExplorer:
    """
    Data exploration utilities for fraud detection dataset analysis.
    
    This class provides methods to analyze transaction patterns, fraud distribution,
    and data quality issues in the fraud detection dataset.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize DataExplorer with default figure size.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def display_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Display comprehensive dataset statistics and information.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing comprehensive dataset information
        """
        logger.info("Generating comprehensive dataset information")
        
        # Basic dataset information
        info = {
            'basic_info': {
                'shape': df.shape,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'columns': list(df.columns)
            },
            'data_types': df.dtypes.to_dict(),
            'missing_values': self._analyze_missing_values(df),
            'numerical_summary': self._get_numerical_summary(df),
            'categorical_summary': self._get_categorical_summary(df)
        }
        
        # Print formatted information
        self._print_dataset_info(info)
        
        return info
    
    def analyze_transaction_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze transaction type distribution and patterns.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            Dict containing transaction type analysis
        """
        logger.info("Analyzing transaction type distribution")
        
        if 'type' not in df.columns:
            raise ValueError("Column 'type' not found in dataset")
        
        # Transaction type counts and percentages
        type_counts = df['type'].value_counts()
        type_percentages = df['type'].value_counts(normalize=True) * 100
        
        # Transaction type statistics
        type_analysis = {
            'counts': type_counts.to_dict(),
            'percentages': type_percentages.to_dict(),
            'unique_types': df['type'].nunique(),
            'most_common': type_counts.index[0],
            'least_common': type_counts.index[-1]
        }
        
        # Analyze transaction amounts by type
        if 'amount' in df.columns:
            amount_by_type = df.groupby('type')['amount'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max', 'sum'
            ]).round(2)
            type_analysis['amount_statistics'] = amount_by_type.to_dict()
        
        # Analyze fraud by transaction type
        if 'isFraud' in df.columns:
            fraud_by_type = df.groupby('type')['isFraud'].agg([
                'count', 'sum', 'mean'
            ])
            fraud_by_type['fraud_rate_percent'] = fraud_by_type['mean'] * 100
            type_analysis['fraud_by_type'] = fraud_by_type.to_dict()
        
        # Print analysis
        self._print_transaction_type_analysis(type_analysis)
        
        return type_analysis
    
    def analyze_amount_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze transaction amount distribution and patterns.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            Dict containing amount distribution analysis
        """
        logger.info("Analyzing transaction amount distribution")
        
        if 'amount' not in df.columns:
            raise ValueError("Column 'amount' not found in dataset")
        
        # Basic amount statistics
        amount_stats = df['amount'].describe()
        
        # Additional statistics
        amount_analysis = {
            'basic_stats': amount_stats.to_dict(),
            'zero_amounts': (df['amount'] == 0).sum(),
            'negative_amounts': (df['amount'] < 0).sum(),
            'large_transactions': (df['amount'] > 200000).sum(),  # Business rule threshold
            'percentiles': {
                '90th': df['amount'].quantile(0.90),
                '95th': df['amount'].quantile(0.95),
                '99th': df['amount'].quantile(0.99),
                '99.9th': df['amount'].quantile(0.999)
            }
        }
        
        # Amount distribution by fraud status
        if 'isFraud' in df.columns:
            fraud_amount_stats = df.groupby('isFraud')['amount'].describe()
            amount_analysis['amount_by_fraud_status'] = fraud_amount_stats.to_dict()
        
        # Print analysis
        self._print_amount_analysis(amount_analysis)
        
        return amount_analysis
    
    def calculate_fraud_ratio(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate fraud ratios and related statistics.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            Dict containing fraud ratio analysis
        """
        logger.info("Calculating fraud ratios and statistics")
        
        if 'isFraud' not in df.columns:
            raise ValueError("Column 'isFraud' not found in dataset")
        
        # Basic fraud statistics
        total_transactions = len(df)
        fraud_transactions = df['isFraud'].sum()
        legitimate_transactions = total_transactions - fraud_transactions
        fraud_rate = fraud_transactions / total_transactions * 100
        
        fraud_analysis = {
            'total_transactions': total_transactions,
            'fraud_transactions': fraud_transactions,
            'legitimate_transactions': legitimate_transactions,
            'fraud_rate_percent': fraud_rate,
            'class_imbalance_ratio': legitimate_transactions / fraud_transactions if fraud_transactions > 0 else float('inf')
        }
        
        # Fraud by transaction type
        if 'type' in df.columns:
            fraud_by_type = df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean'])
            fraud_by_type['fraud_rate_percent'] = fraud_by_type['mean'] * 100
            fraud_analysis['fraud_by_transaction_type'] = fraud_by_type.to_dict()
        
        # Fraud by time (if step column exists)
        if 'step' in df.columns:
            # Group by time periods (e.g., every 24 steps = 1 day)
            df_temp = df.copy()
            df_temp['day'] = df_temp['step'] // 24 + 1
            fraud_by_day = df_temp.groupby('day')['isFraud'].agg(['count', 'sum', 'mean'])
            fraud_by_day['fraud_rate_percent'] = fraud_by_day['mean'] * 100
            fraud_analysis['fraud_by_day'] = {
                'daily_stats': fraud_by_day.describe().to_dict(),
                'highest_fraud_day': fraud_by_day['sum'].idxmax(),
                'lowest_fraud_day': fraud_by_day['sum'].idxmin()
            }
        
        # Business rule flag analysis
        if 'isFlaggedFraud' in df.columns:
            flagged_analysis = self._analyze_flagged_fraud(df)
            fraud_analysis['flagged_fraud_analysis'] = flagged_analysis
        
        # Print analysis
        self._print_fraud_analysis(fraud_analysis)
        
        return fraud_analysis
    
    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive missing value detection and analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing missing value analysis
        """
        logger.info("Detecting and analyzing missing values")
        
        # Calculate missing values
        missing_counts = df.isnull().sum()
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        
        # Identify columns with missing values
        columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
        
        missing_analysis = {
            'total_missing_values': missing_counts.sum(),
            'columns_with_missing': len(columns_with_missing),
            'missing_by_column': {
                'counts': missing_counts.to_dict(),
                'percentages': missing_percentages.to_dict()
            },
            'columns_with_missing_list': columns_with_missing,
            'complete_rows': len(df) - df.isnull().any(axis=1).sum(),
            'rows_with_missing': df.isnull().any(axis=1).sum()
        }
        
        # Missing value patterns
        if columns_with_missing:
            # Analyze missing value patterns
            missing_patterns = df[columns_with_missing].isnull().value_counts()
            missing_analysis['missing_patterns'] = missing_patterns.head(10).to_dict()
        
        # Print analysis
        self._print_missing_value_analysis(missing_analysis)
        
        return missing_analysis
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing complete data quality analysis
        """
        logger.info("Generating comprehensive data quality report")
        
        quality_report = {
            'dataset_info': self.display_dataset_info(df),
            'transaction_types': self.analyze_transaction_types(df),
            'amount_distribution': self.analyze_amount_distribution(df),
            'fraud_analysis': self.calculate_fraud_ratio(df),
            'missing_values': self.detect_missing_values(df),
            'data_quality_issues': self._identify_data_quality_issues(df)
        }
        
        return quality_report
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': (missing_counts > 0).sum(),
            'by_column': {
                'counts': missing_counts.to_dict(),
                'percentages': missing_percentages.to_dict()
            }
        }
    
    def _get_numerical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            return {}
        
        return {
            'columns': numerical_cols,
            'statistics': df[numerical_cols].describe().to_dict()
        }
    
    def _get_categorical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            return {}
        
        summary = {}
        for col in categorical_cols:
            summary[col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'value_counts': df[col].value_counts().head(10).to_dict()
            }
        
        return {
            'columns': categorical_cols,
            'summary': summary
        }
    
    def _analyze_flagged_fraud(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze business rule flagged fraud vs actual fraud."""
        # Compare isFlaggedFraud with isFraud
        confusion_matrix = pd.crosstab(df['isFlaggedFraud'], df['isFraud'], margins=True)
        
        # Calculate metrics
        true_positives = confusion_matrix.loc[1, 1] if 1 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
        false_positives = confusion_matrix.loc[1, 0] if 1 in confusion_matrix.index and 0 in confusion_matrix.columns else 0
        false_negatives = confusion_matrix.loc[0, 1] if 0 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
        true_negatives = confusion_matrix.loc[0, 0] if 0 in confusion_matrix.index and 0 in confusion_matrix.columns else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return {
            'confusion_matrix': confusion_matrix.to_dict(),
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives
        }
    
    def _identify_data_quality_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify potential data quality issues."""
        issues = {
            'duplicate_rows': df.duplicated().sum(),
            'empty_strings': {},
            'outliers': {},
            'inconsistencies': []
        }
        
        # Check for empty strings in object columns
        for col in df.select_dtypes(include=['object']).columns:
            empty_strings = (df[col] == '').sum()
            if empty_strings > 0:
                issues['empty_strings'][col] = empty_strings
        
        # Check for outliers in numerical columns using IQR method
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                issues['outliers'][col] = outliers
        
        # Check for business logic inconsistencies
        if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns and 'amount' in df.columns:
            # Check if balance changes match transaction amounts
            expected_balance = df['oldbalanceOrg'] - df['amount']
            balance_mismatch = (abs(df['newbalanceOrig'] - expected_balance) > 0.01).sum()
            if balance_mismatch > 0:
                issues['inconsistencies'].append(f"Balance calculation mismatch: {balance_mismatch} transactions")
        
        return issues
    
    def _print_dataset_info(self, info: Dict[str, Any]) -> None:
        """Print formatted dataset information."""
        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        
        basic = info['basic_info']
        print(f"Shape: {basic['shape']}")
        print(f"Total Rows: {basic['total_rows']:,}")
        print(f"Total Columns: {basic['total_columns']}")
        print(f"Memory Usage: {basic['memory_usage_mb']:.2f} MB")
        print(f"Columns: {', '.join(basic['columns'])}")
        
        print("\nDATA TYPES:")
        for col, dtype in info['data_types'].items():
            print(f"  {col}: {dtype}")
        
        if info['missing_values']['total_missing'] > 0:
            print(f"\nMISSING VALUES: {info['missing_values']['total_missing']:,} total")
            for col, count in info['missing_values']['by_column']['counts'].items():
                if count > 0:
                    pct = info['missing_values']['by_column']['percentages'][col]
                    print(f"  {col}: {count:,} ({pct:.2f}%)")
    
    def _print_transaction_type_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print formatted transaction type analysis."""
        print("\n" + "=" * 60)
        print("TRANSACTION TYPE ANALYSIS")
        print("=" * 60)
        
        print(f"Unique Transaction Types: {analysis['unique_types']}")
        print(f"Most Common: {analysis['most_common']}")
        print(f"Least Common: {analysis['least_common']}")
        
        print("\nTRANSACTION TYPE DISTRIBUTION:")
        for trans_type, count in analysis['counts'].items():
            pct = analysis['percentages'][trans_type]
            print(f"  {trans_type}: {count:,} ({pct:.2f}%)")
        
        if 'fraud_by_type' in analysis:
            print("\nFRAUD BY TRANSACTION TYPE:")
            for trans_type in analysis['counts'].keys():
                if trans_type in analysis['fraud_by_type']['fraud_rate_percent']:
                    fraud_rate = analysis['fraud_by_type']['fraud_rate_percent'][trans_type]
                    fraud_count = analysis['fraud_by_type']['sum'][trans_type]
                    print(f"  {trans_type}: {fraud_count:,} fraudulent ({fraud_rate:.4f}%)")
    
    def _print_amount_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print formatted amount distribution analysis."""
        print("\n" + "=" * 60)
        print("AMOUNT DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        stats = analysis['basic_stats']
        print(f"Mean: ${stats['mean']:,.2f}")
        print(f"Median: ${stats['50%']:,.2f}")
        print(f"Std Dev: ${stats['std']:,.2f}")
        print(f"Min: ${stats['min']:,.2f}")
        print(f"Max: ${stats['max']:,.2f}")
        
        print(f"\nZero Amount Transactions: {analysis['zero_amounts']:,}")
        print(f"Negative Amount Transactions: {analysis['negative_amounts']:,}")
        print(f"Large Transactions (>$200K): {analysis['large_transactions']:,}")
        
        print("\nPERCENTILES:")
        for pct, value in analysis['percentiles'].items():
            print(f"  {pct}: ${value:,.2f}")
    
    def _print_fraud_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print formatted fraud analysis."""
        print("\n" + "=" * 60)
        print("FRAUD ANALYSIS")
        print("=" * 60)
        
        print(f"Total Transactions: {analysis['total_transactions']:,}")
        print(f"Fraudulent Transactions: {analysis['fraud_transactions']:,}")
        print(f"Legitimate Transactions: {analysis['legitimate_transactions']:,}")
        print(f"Fraud Rate: {analysis['fraud_rate_percent']:.4f}%")
        
        if analysis['class_imbalance_ratio'] != float('inf'):
            print(f"Class Imbalance Ratio: {analysis['class_imbalance_ratio']:.1f}:1")
        else:
            print("Class Imbalance Ratio: No fraudulent transactions found")
    
    def _print_missing_value_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print formatted missing value analysis."""
        print("\n" + "=" * 60)
        print("MISSING VALUE ANALYSIS")
        print("=" * 60)
        
        print(f"Total Missing Values: {analysis['total_missing_values']:,}")
        print(f"Columns with Missing Values: {analysis['columns_with_missing']}")
        print(f"Complete Rows: {analysis['complete_rows']:,}")
        print(f"Rows with Missing Values: {analysis['rows_with_missing']:,}")
        
        if analysis['columns_with_missing_list']:
            print("\nMISSING VALUES BY COLUMN:")
            for col in analysis['columns_with_missing_list']:
                count = analysis['missing_by_column']['counts'][col]
                pct = analysis['missing_by_column']['percentages'][col]
                print(f"  {col}: {count:,} ({pct:.2f}%)")