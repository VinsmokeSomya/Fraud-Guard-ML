#!/usr/bin/env python3
"""
Test data generator for fraud detection integration tests.

This script generates realistic test datasets for use in integration testing.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def generate_fraud_dataset(n_samples=1000, fraud_rate=0.1, seed=42):
    """
    Generate a realistic fraud detection dataset.
    
    Args:
        n_samples: Number of transactions to generate
        fraud_rate: Target fraud rate (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated fraud dataset
    """
    np.random.seed(seed)
    
    print(f"Generating {n_samples:,} transactions with {fraud_rate:.1%} fraud rate...")
    
    # Generate basic transaction features
    data = {
        'step': np.random.randint(1, 745, n_samples),
        'type': np.random.choice(
            ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], 
            n_samples,
            p=[0.15, 0.20, 0.05, 0.45, 0.15]  # Realistic distribution
        ),
        'amount': np.random.lognormal(8, 2, n_samples),  # Log-normal for realistic amounts
        'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
        'oldbalanceOrg': np.random.exponential(10000, n_samples),
        'newbalanceOrig': np.random.exponential(10000, n_samples),
        'nameDest': [],
        'oldbalanceDest': np.random.exponential(5000, n_samples),
        'newbalanceDest': np.random.exponential(5000, n_samples),
    }
    
    # Generate destination names (mix of customers and merchants)
    for i in range(n_samples):
        if np.random.random() > 0.3:  # 70% merchants
            data['nameDest'].append(f'M{i:09d}')
        else:  # 30% customers
            data['nameDest'].append(f'C{i+n_samples:09d}')
    
    df = pd.DataFrame(data)
    
    # Generate realistic fraud labels based on business rules
    fraud_prob = np.zeros(n_samples)
    
    # Higher fraud probability for certain transaction types
    fraud_prob += np.where(df['type'] == 'CASH-OUT', 0.4, 0)
    fraud_prob += np.where(df['type'] == 'TRANSFER', 0.3, 0)
    fraud_prob += np.where(df['type'] == 'PAYMENT', 0.02, 0)
    fraud_prob += np.where(df['type'] == 'CASH-IN', 0.01, 0)
    fraud_prob += np.where(df['type'] == 'DEBIT', 0.01, 0)
    
    # Large amounts increase fraud probability
    fraud_prob += np.where(df['amount'] > 200000, 0.5, 0)
    fraud_prob += np.where(df['amount'] > 100000, 0.2, 0)
    fraud_prob += np.where(df['amount'] > 50000, 0.1, 0)
    
    # Zero balance patterns (common in fraud)
    zero_balance_orig = (df['oldbalanceOrg'] == 0) & (df['newbalanceOrig'] == 0)
    fraud_prob += np.where(zero_balance_orig, 0.3, 0)
    
    # Complete balance depletion
    balance_depletion = (df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)
    fraud_prob += np.where(balance_depletion, 0.4, 0)
    
    # Customer-to-customer transfers (higher risk)
    c2c_transfer = (df['type'] == 'TRANSFER') & (~df['nameDest'].str.startswith('M'))
    fraud_prob += np.where(c2c_transfer, 0.2, 0)
    
    # Clip probabilities and adjust to target fraud rate
    fraud_prob = np.clip(fraud_prob, 0, 0.9)
    
    # Scale to achieve target fraud rate
    current_expected_fraud = fraud_prob.mean()
    if current_expected_fraud > 0:
        scaling_factor = fraud_rate / current_expected_fraud
        fraud_prob *= scaling_factor
        fraud_prob = np.clip(fraud_prob, 0, 0.9)
    
    # Generate fraud labels
    df['isFraud'] = np.random.binomial(1, fraud_prob, n_samples)
    
    # Business rule flagging (subset of actual fraud for large transfers)
    df['isFlaggedFraud'] = np.where(
        (df['type'] == 'TRANSFER') & 
        (df['amount'] > 200000) & 
        (df['isFraud'] == 1),
        1, 0
    )
    
    # Adjust balances to be more realistic
    for idx in df.index:
        if df.loc[idx, 'type'] in ['CASH-OUT', 'PAYMENT', 'DEBIT']:
            # Outgoing transactions should reduce origin balance
            if df.loc[idx, 'oldbalanceOrg'] >= df.loc[idx, 'amount']:
                df.loc[idx, 'newbalanceOrig'] = df.loc[idx, 'oldbalanceOrg'] - df.loc[idx, 'amount']
        elif df.loc[idx, 'type'] == 'CASH-IN':
            # Incoming transactions should increase origin balance
            df.loc[idx, 'newbalanceOrig'] = df.loc[idx, 'oldbalanceOrg'] + df.loc[idx, 'amount']
        elif df.loc[idx, 'type'] == 'TRANSFER':
            # Transfers affect both origin and destination
            if df.loc[idx, 'oldbalanceOrg'] >= df.loc[idx, 'amount']:
                df.loc[idx, 'newbalanceOrig'] = df.loc[idx, 'oldbalanceOrg'] - df.loc[idx, 'amount']
                df.loc[idx, 'newbalanceDest'] = df.loc[idx, 'oldbalanceDest'] + df.loc[idx, 'amount']
    
    # Final statistics
    actual_fraud_rate = df['isFraud'].mean()
    flagged_fraud_rate = df['isFlaggedFraud'].mean()
    
    print(f"✅ Generated dataset:")
    print(f"   - Total transactions: {len(df):,}")
    print(f"   - Fraud rate: {actual_fraud_rate:.3%}")
    print(f"   - Flagged fraud rate: {flagged_fraud_rate:.3%}")
    print(f"   - Transaction types: {df['type'].value_counts().to_dict()}")
    print(f"   - Amount range: ${df['amount'].min():.2f} - ${df['amount'].max():,.2f}")
    
    return df


def generate_small_test_dataset(n_samples=100):
    """Generate a small dataset for quick testing."""
    return generate_fraud_dataset(n_samples=n_samples, fraud_rate=0.15, seed=42)


def generate_large_test_dataset(n_samples=10000):
    """Generate a large dataset for performance testing."""
    return generate_fraud_dataset(n_samples=n_samples, fraud_rate=0.08, seed=123)


def generate_edge_case_dataset():
    """Generate dataset with edge cases for robustness testing."""
    np.random.seed(999)
    
    # Create various edge cases
    edge_cases = []
    
    # Case 1: Zero amounts
    edge_cases.append({
        'step': 1, 'type': 'PAYMENT', 'amount': 0.0,
        'nameOrig': 'C000000001', 'oldbalanceOrg': 1000.0, 'newbalanceOrig': 1000.0,
        'nameDest': 'M000000001', 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0,
        'isFraud': 0, 'isFlaggedFraud': 0
    })
    
    # Case 2: Very large amounts
    edge_cases.append({
        'step': 2, 'type': 'TRANSFER', 'amount': 10000000.0,
        'nameOrig': 'C000000002', 'oldbalanceOrg': 10000000.0, 'newbalanceOrig': 0.0,
        'nameDest': 'C000000003', 'oldbalanceDest': 0.0, 'newbalanceDest': 10000000.0,
        'isFraud': 1, 'isFlaggedFraud': 1
    })
    
    # Case 3: Zero balances throughout
    edge_cases.append({
        'step': 3, 'type': 'CASH-OUT', 'amount': 5000.0,
        'nameOrig': 'C000000004', 'oldbalanceOrg': 0.0, 'newbalanceOrig': 0.0,
        'nameDest': 'M000000002', 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0,
        'isFraud': 1, 'isFlaggedFraud': 0
    })
    
    # Case 4: Negative balance (data quality issue)
    edge_cases.append({
        'step': 4, 'type': 'PAYMENT', 'amount': 1000.0,
        'nameOrig': 'C000000005', 'oldbalanceOrg': 500.0, 'newbalanceOrig': -500.0,
        'nameDest': 'M000000003', 'oldbalanceDest': 1000.0, 'newbalanceDest': 2000.0,
        'isFraud': 0, 'isFlaggedFraud': 0
    })
    
    # Case 5: Merchant to customer (unusual)
    edge_cases.append({
        'step': 5, 'type': 'TRANSFER', 'amount': 2000.0,
        'nameOrig': 'M000000004', 'oldbalanceOrg': 5000.0, 'newbalanceOrig': 3000.0,
        'nameDest': 'C000000006', 'oldbalanceDest': 1000.0, 'newbalanceDest': 3000.0,
        'isFraud': 0, 'isFlaggedFraud': 0
    })
    
    # Add some normal transactions
    normal_data = generate_fraud_dataset(n_samples=95, fraud_rate=0.1, seed=888)
    
    # Combine edge cases with normal data
    edge_df = pd.DataFrame(edge_cases)
    combined_df = pd.concat([edge_df, normal_data], ignore_index=True)
    
    print(f"✅ Generated edge case dataset with {len(combined_df)} transactions")
    
    return combined_df


def main():
    """Main function to generate test datasets."""
    parser = argparse.ArgumentParser(description="Generate test datasets for fraud detection")
    
    parser.add_argument(
        "--type",
        choices=["small", "large", "edge", "all"],
        default="small",
        help="Type of dataset to generate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/test"),
        help="Output directory for generated datasets"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples to generate (overrides type defaults)"
    )
    
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.1,
        help="Target fraud rate (0.0 to 1.0)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🏭 Fraud Detection Test Data Generator")
    print("=" * 50)
    
    if args.type == "small" or args.type == "all":
        print("\n📊 Generating small test dataset...")
        n_samples = args.samples if args.samples else 100
        df = generate_fraud_dataset(n_samples=n_samples, fraud_rate=args.fraud_rate)
        output_path = args.output_dir / "small_test_data.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved to: {output_path}")
    
    if args.type == "large" or args.type == "all":
        print("\n📊 Generating large test dataset...")
        n_samples = args.samples if args.samples else 10000
        df = generate_fraud_dataset(n_samples=n_samples, fraud_rate=args.fraud_rate)
        output_path = args.output_dir / "large_test_data.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved to: {output_path}")
    
    if args.type == "edge" or args.type == "all":
        print("\n📊 Generating edge case dataset...")
        df = generate_edge_case_dataset()
        output_path = args.output_dir / "edge_case_data.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved to: {output_path}")
    
    print("\n✅ Test data generation completed!")


if __name__ == "__main__":
    main()