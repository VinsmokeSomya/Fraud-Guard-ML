#!/usr/bin/env python3
"""
Example script demonstrating the DataVisualizer class for exploratory data analysis.

This script shows how to use the DataVisualizer to create comprehensive visualizations
of fraud detection data, including transaction distributions, fraud patterns, and
correlation analysis.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from visualization.data_visualizer import DataVisualizer
from data.data_loader import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Create sample fraud detection data for demonstration.
    
    Args:
        n_samples: Number of sample transactions to generate
        
    Returns:
        pd.DataFrame: Sample transaction data
    """
    logger.info(f"Creating sample dataset with {n_samples:,} transactions")
    
    np.random.seed(42)  # For reproducible results
    
    # Transaction types with different probabilities
    transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    type_probs = [0.15, 0.20, 0.05, 0.45, 0.15]
    
    data = {
        'step': np.random.randint(1, 745, n_samples),
        'type': np.random.choice(transaction_types, n_samples, p=type_probs),
        'amount': np.random.lognormal(mean=8, sigma=2, size=n_samples),
        'nameOrig': [f'C{i:010d}' for i in np.random.randint(1, 1000000, n_samples)],
        'oldbalanceOrg': np.random.exponential(scale=50000, size=n_samples),
        'newbalanceOrig': np.random.exponential(scale=50000, size=n_samples),
        'nameDest': [f'{"M" if np.random.random() < 0.3 else "C"}{i:010d}' 
                    for i in np.random.randint(1, 1000000, n_samples)],
        'oldbalanceDest': np.random.exponential(scale=30000, size=n_samples),
        'newbalanceDest': np.random.exponential(scale=30000, size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create fraud labels with higher probability for certain conditions
    fraud_prob = np.where(
        (df['type'].isin(['TRANSFER', 'CASH_OUT'])) & 
        (df['amount'] > 200000), 
        0.8,  # High fraud probability for large transfers/cash-outs
        0.001  # Low base fraud rate
    )
    
    df['isFraud'] = np.random.binomial(1, fraud_prob)
    df['isFlaggedFraud'] = np.where(
        (df['type'] == 'TRANSFER') & (df['amount'] > 200000), 1, 0
    )
    
    logger.info(f"Sample data created with {df['isFraud'].sum():,} fraudulent transactions "
               f"({df['isFraud'].mean()*100:.4f}% fraud rate)")
    
    return df


def demonstrate_static_visualizations(visualizer: DataVisualizer, df: pd.DataFrame) -> None:
    """
    Demonstrate static matplotlib/seaborn visualizations.
    
    Args:
        visualizer: DataVisualizer instance
        df: Transaction data
    """
    logger.info("Creating static visualizations...")
    
    print("\n" + "="*60)
    print("STATIC VISUALIZATIONS (matplotlib/seaborn)")
    print("="*60)
    
    # Print data summary
    visualizer.print_data_summary(df)
    
    # Transaction distribution by type
    print("\n1. Transaction Distribution by Type")
    visualizer.plot_transaction_distribution_by_type(df, show_fraud=True, interactive=False)
    
    # Amount distribution
    print("\n2. Transaction Amount Distribution")
    visualizer.plot_transaction_distribution_by_amount(df, bins=30, log_scale=True, 
                                                      show_fraud=True, interactive=False)
    
    # Time distribution
    print("\n3. Transaction Distribution by Hour")
    visualizer.plot_transaction_distribution_by_time(df, time_unit='hour', 
                                                    show_fraud=True, interactive=False)
    
    # Fraud patterns
    print("\n4. Fraud Patterns by Transaction Type")
    visualizer.plot_fraud_patterns_by_type(df, interactive=False)
    
    # Correlation heatmap
    print("\n5. Feature Correlation Heatmap")
    numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                         'oldbalanceDest', 'newbalanceDest']
    visualizer.plot_correlation_heatmap(df, features=numerical_features, interactive=False)
    
    # Feature distributions
    print("\n6. Feature Distributions")
    key_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
    visualizer.plot_feature_distributions(df, features=key_features, 
                                        show_fraud_comparison=True, interactive=False)


def demonstrate_interactive_visualizations(visualizer: DataVisualizer, df: pd.DataFrame) -> None:
    """
    Demonstrate interactive plotly visualizations.
    
    Args:
        visualizer: DataVisualizer instance
        df: Transaction data
    """
    logger.info("Creating interactive visualizations...")
    
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZATIONS (plotly)")
    print("="*60)
    
    # Transaction distribution by type
    print("\n1. Interactive Transaction Distribution by Type")
    visualizer.plot_transaction_distribution_by_type(df, show_fraud=True, interactive=True)
    
    # Amount distribution
    print("\n2. Interactive Transaction Amount Distribution")
    visualizer.plot_transaction_distribution_by_amount(df, bins=30, log_scale=True, 
                                                      show_fraud=True, interactive=True)
    
    # Time distribution
    print("\n3. Interactive Transaction Distribution by Day")
    visualizer.plot_transaction_distribution_by_time(df, time_unit='day', 
                                                    show_fraud=True, interactive=True)
    
    # Fraud patterns
    print("\n4. Interactive Fraud Patterns by Transaction Type")
    visualizer.plot_fraud_patterns_by_type(df, interactive=True)
    
    # Correlation heatmap
    print("\n5. Interactive Feature Correlation Heatmap")
    numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                         'oldbalanceDest', 'newbalanceDest']
    visualizer.plot_correlation_heatmap(df, features=numerical_features, interactive=True)
    
    # Feature distributions
    print("\n6. Interactive Feature Distributions")
    key_features = ['amount', 'oldbalanceOrg']
    visualizer.plot_feature_distributions(df, features=key_features, 
                                        show_fraud_comparison=True, interactive=True)


def demonstrate_comprehensive_report(visualizer: DataVisualizer, df: pd.DataFrame) -> None:
    """
    Demonstrate comprehensive EDA report generation.
    
    Args:
        visualizer: DataVisualizer instance
        df: Transaction data
    """
    logger.info("Creating comprehensive EDA report...")
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EDA REPORT")
    print("="*60)
    
    # Create comprehensive static report
    print("\nGenerating comprehensive static EDA report...")
    visualizer.create_comprehensive_eda_report(df, interactive=False)
    
    # Optionally save the report
    # visualizer.create_comprehensive_eda_report(df, save_path='fraud_eda_report.png', interactive=False)


def main():
    """Main function to demonstrate DataVisualizer functionality."""
    logger.info("Starting DataVisualizer demonstration")
    
    try:
        # Create sample data
        df = create_sample_data(n_samples=50000)
        
        # Initialize visualizer
        visualizer = DataVisualizer(
            style='whitegrid',
            palette='Set2',
            figure_size=(12, 8),
            dpi=100
        )
        
        # Demonstrate different visualization types
        demonstrate_static_visualizations(visualizer, df)
        
        # Ask user if they want to see interactive plots
        response = input("\nWould you like to see interactive visualizations? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            demonstrate_interactive_visualizations(visualizer, df)
        
        # Ask user if they want to see comprehensive report
        response = input("\nWould you like to see the comprehensive EDA report? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            demonstrate_comprehensive_report(visualizer, df)
        
        logger.info("DataVisualizer demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()