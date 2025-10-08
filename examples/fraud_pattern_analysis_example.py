"""
Example script demonstrating fraud pattern analysis utilities.

This script shows how to use the FraudPatternAnalyzer class to analyze
fraud patterns including time series analysis, customer behavior analysis,
and risk factor identification.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from visualization.fraud_pattern_analyzer import FraudPatternAnalyzer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_fraud_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Create sample fraud detection data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample transaction data
    """
    np.random.seed(42)
    
    # Generate basic transaction data
    data = {
        'step': np.random.randint(1, 745, n_samples),  # 744 time steps (30 days * 24 hours + 1)
        'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], 
                                n_samples, p=[0.2, 0.15, 0.1, 0.4, 0.15]),
        'amount': np.random.lognormal(8, 2, n_samples),  # Log-normal distribution for amounts
        'nameOrig': [f'C{i}' if np.random.random() > 0.1 else f'M{i}' 
                    for i in range(n_samples)],  # 90% customers, 10% merchants
        'nameDest': [f'C{i+n_samples}' if np.random.random() > 0.15 else f'M{i+n_samples}' 
                    for i in range(n_samples)]  # 85% customers, 15% merchants
    }
    
    # Generate balance data
    data['oldbalanceOrg'] = np.random.exponential(50000, n_samples)
    data['oldbalanceDest'] = np.random.exponential(30000, n_samples)
    
    # Calculate new balances based on transaction type and amount
    data['newbalanceOrig'] = []
    data['newbalanceDest'] = []
    
    for i in range(n_samples):
        if data['type'][i] in ['CASH_OUT', 'TRANSFER', 'PAYMENT']:
            # Outgoing transaction
            new_orig = max(0, data['oldbalanceOrg'][i] - data['amount'][i])
        else:
            # Incoming transaction
            new_orig = data['oldbalanceOrg'][i] + data['amount'][i]
        
        if data['type'][i] in ['CASH_IN', 'TRANSFER']:
            # Incoming transaction to destination
            new_dest = data['oldbalanceDest'][i] + data['amount'][i]
        else:
            # No change to destination for other transaction types
            new_dest = data['oldbalanceDest'][i]
        
        data['newbalanceOrig'].append(new_orig)
        data['newbalanceDest'].append(new_dest)
    
    df = pd.DataFrame(data)
    
    # Generate fraud labels with realistic patterns
    fraud_prob = np.zeros(n_samples)
    
    # Higher fraud probability for certain patterns
    fraud_prob += (df['type'] == 'TRANSFER') * 0.05  # Transfers more likely to be fraud
    fraud_prob += (df['type'] == 'CASH_OUT') * 0.03  # Cash-outs somewhat likely
    fraud_prob += (df['amount'] > 200000) * 0.15     # Large amounts more likely
    fraud_prob += (df['newbalanceOrig'] == 0) * 0.1  # Complete balance depletion
    fraud_prob += (df['oldbalanceOrg'] == 0) * 0.02  # Zero starting balance
    
    # Time-based patterns (higher fraud at certain hours)
    hour = df['step'] % 24
    fraud_prob += ((hour >= 22) | (hour <= 4)) * 0.02  # Late night/early morning
    
    # Merchant destination patterns
    is_merchant_dest = df['nameDest'].str.startswith('M')
    fraud_prob += (~is_merchant_dest) * 0.01  # Customer-to-customer slightly higher risk
    
    # Generate fraud labels
    df['isFraud'] = np.random.binomial(1, np.clip(fraud_prob, 0, 0.3), n_samples)
    
    # Add flagged fraud column (business rule: large transfers)
    df['isFlaggedFraud'] = ((df['type'] == 'TRANSFER') & (df['amount'] > 200000)).astype(int)
    
    logger.info(f"Generated {n_samples} transactions with {df['isFraud'].sum()} fraud cases "
                f"({df['isFraud'].mean()*100:.2f}% fraud rate)")
    
    return df


def demonstrate_fraud_pattern_analysis():
    """Demonstrate fraud pattern analysis capabilities."""
    print("=== Fraud Pattern Analysis Demonstration ===\n")
    
    # Create sample data
    print("1. Creating sample fraud detection dataset...")
    df = create_sample_fraud_data(n_samples=50000)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Fraud rate: {df['isFraud'].mean()*100:.2f}%")
    print(f"   Transaction types: {df['type'].value_counts().to_dict()}")
    
    # Initialize analyzer
    print("\n2. Initializing FraudPatternAnalyzer...")
    analyzer = FraudPatternAnalyzer(figure_size=(12, 8))
    
    # Time series analysis
    print("\n3. Analyzing fraud time series patterns...")
    print("   3.1 Hourly fraud count analysis...")
    hourly_results = analyzer.plot_fraud_time_series(
        df, time_unit='hour', aggregation='count', interactive=False
    )
    
    print("   3.2 Daily fraud rate analysis...")
    daily_results = analyzer.plot_fraud_time_series(
        df, time_unit='day', aggregation='rate', interactive=False
    )
    
    # Customer behavior analysis
    print("\n4. Analyzing customer behavior patterns...")
    behavior_results = analyzer.analyze_customer_behavior(df, interactive=False)
    
    if 'unusual_balance_fraud_rate' in behavior_results:
        print(f"   Unusual balance change fraud rate: {behavior_results['unusual_balance_fraud_rate']*100:.2f}%")
        print(f"   Merchant destination fraud rate: {behavior_results['merchant_dest_fraud_rate']*100:.2f}%")
        print(f"   Customer destination fraud rate: {behavior_results['customer_dest_fraud_rate']*100:.2f}%")
    
    # Risk factor identification
    print("\n5. Identifying and ranking risk factors...")
    risk_results = analyzer.identify_risk_factors(df, top_n=15)
    
    if 'ranked_factors' in risk_results:
        print("   Top 10 Risk Factors:")
        for i, factor in enumerate(risk_results['ranked_factors'][:10], 1):
            print(f"   {i:2d}. {factor['factor_name']:<30} "
                  f"Fraud Rate: {factor['fraud_rate']*100:6.2f}% "
                  f"({factor['fraud_count']}/{factor['total_count']})")
    
    if 'summary_stats' in risk_results:
        summary = risk_results['summary_stats']
        print(f"\n   Risk Factor Summary:")
        print(f"   - Total risk factors identified: {summary['total_risk_factors']}")
        print(f"   - Highest risk category: {summary['highest_risk_factor']}")
        print(f"   - Average fraud rate across factors: {summary['average_fraud_rate']*100:.2f}%")
    
    # Comprehensive analysis
    print("\n6. Creating comprehensive fraud pattern analysis...")
    comprehensive_results = analyzer.create_comprehensive_fraud_analysis(
        df, interactive=False, save_path=None
    )
    
    print("\n=== Analysis Complete ===")
    print("All fraud pattern analysis utilities have been demonstrated successfully!")
    
    return {
        'hourly_results': hourly_results,
        'daily_results': daily_results,
        'behavior_results': behavior_results,
        'risk_results': risk_results,
        'comprehensive_results': comprehensive_results
    }


def demonstrate_interactive_analysis():
    """Demonstrate interactive fraud pattern analysis."""
    print("\n=== Interactive Analysis Demonstration ===\n")
    
    # Create smaller dataset for interactive demo
    print("Creating smaller dataset for interactive demonstration...")
    df = create_sample_fraud_data(n_samples=10000)
    
    analyzer = FraudPatternAnalyzer()
    
    print("Running interactive fraud pattern analysis...")
    print("(Note: Interactive plots will open in browser/notebook)")
    
    # Interactive time series
    analyzer.plot_fraud_time_series(df, time_unit='hour', interactive=True)
    
    # Interactive customer behavior
    analyzer.analyze_customer_behavior(df, interactive=True)
    
    print("Interactive analysis complete!")


if __name__ == "__main__":
    # Run static analysis demonstration
    results = demonstrate_fraud_pattern_analysis()
    
    # Optionally run interactive demonstration
    # Uncomment the line below to see interactive plots
    # demonstrate_interactive_analysis()
    
    print(f"\nExample completed successfully!")
    print(f"Generated analysis results with {len(results)} components.")