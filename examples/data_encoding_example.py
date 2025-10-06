"""
Example script demonstrating data encoding and scaling utilities.

This script shows how to use the DataEncoder class and scaler wrappers
for preprocessing fraud detection data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data import DataEncoder, StandardScalerWrapper, MinMaxScalerWrapper

def main():
    """Demonstrate data encoding and scaling functionality."""
    print("Data Encoding and Scaling Example")
    print("=" * 50)
    
    # Create sample fraud detection data
    np.random.seed(42)
    sample_data = {
        'step': np.random.randint(1, 745, 100),
        'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], 100),
        'amount': np.random.exponential(1000, 100),
        'nameOrig': [f'C{i}' for i in np.random.randint(1, 20, 100)],
        'nameDest': [f'{"M" if np.random.random() > 0.7 else "C"}{i}' for i in np.random.randint(1, 20, 100)],
        'oldbalanceOrg': np.random.exponential(5000, 100),
        'newbalanceOrig': np.random.exponential(5000, 100),
        'oldbalanceDest': np.random.exponential(5000, 100),
        'newbalanceDest': np.random.exponential(5000, 100),
        'isFraud': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Sample data shape: {df.shape}")
    print(f"Fraud ratio: {df['isFraud'].mean():.2%}")
    print("\nOriginal data types:")
    print(df.dtypes)
    
    # Example 1: Label encoding with standard scaling
    print("\n" + "=" * 50)
    print("Example 1: Label Encoding + Standard Scaling")
    print("=" * 50)
    
    encoder = DataEncoder(
        categorical_encoding='label',
        numerical_scaling='standard'
    )
    
    # Define feature types
    categorical_features = ['type', 'nameOrig', 'nameDest']
    numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                         'oldbalanceDest', 'newbalanceDest']
    
    # Fit and transform data
    df_encoded = encoder.fit_transform(
        df,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_column='isFraud'
    )
    
    print(f"Encoded data shape: {df_encoded.shape}")
    print("\nEncoded data types:")
    print(df_encoded.dtypes)
    print("\nEncoding summary:")
    summary = encoder.get_encoding_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Example 2: Stratified train-test split
    print("\n" + "=" * 50)
    print("Example 2: Stratified Train-Test Split")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = encoder.stratified_train_test_split(
        df,
        target_column='isFraud',
        test_size=0.3,
        random_state=42,
        encode_before_split=True
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train fraud ratio: {y_train.mean():.2%}")
    print(f"Test fraud ratio: {y_test.mean():.2%}")
    
    # Example 3: One-hot encoding with min-max scaling
    print("\n" + "=" * 50)
    print("Example 3: One-Hot Encoding + Min-Max Scaling")
    print("=" * 50)
    
    encoder_onehot = DataEncoder(
        categorical_encoding='onehot',
        numerical_scaling='minmax'
    )
    
    # Only encode transaction type to avoid too many features
    df_onehot = encoder_onehot.fit_transform(
        df,
        categorical_features=['type'],
        numerical_features=numerical_features,
        target_column='isFraud'
    )
    
    print(f"One-hot encoded data shape: {df_onehot.shape}")
    print("Feature names after one-hot encoding:")
    feature_names = encoder_onehot.get_feature_names()
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name}")
    
    # Example 4: Using scaler wrappers directly
    print("\n" + "=" * 50)
    print("Example 4: Direct Scaler Usage")
    print("=" * 50)
    
    # Select numerical columns
    numerical_data = df[numerical_features]
    
    # Standard scaling
    std_scaler = StandardScalerWrapper()
    scaled_std = std_scaler.fit_transform(numerical_data)
    
    print("Standard Scaling Results:")
    print(f"  Original mean: {numerical_data.mean().mean():.2f}")
    print(f"  Scaled mean: {scaled_std.mean().mean():.6f}")
    print(f"  Original std: {numerical_data.std().mean():.2f}")
    print(f"  Scaled std: {scaled_std.std().mean():.6f}")
    
    # Min-max scaling
    minmax_scaler = MinMaxScalerWrapper(feature_range=(0, 1))
    scaled_minmax = minmax_scaler.fit_transform(numerical_data)
    
    print("\nMin-Max Scaling Results:")
    print(f"  Original min: {numerical_data.min().min():.2f}")
    print(f"  Scaled min: {scaled_minmax.min().min():.6f}")
    print(f"  Original max: {numerical_data.max().max():.2f}")
    print(f"  Scaled max: {scaled_minmax.max().max():.6f}")
    
    # Example 5: Handling unknown categories
    print("\n" + "=" * 50)
    print("Example 5: Handling Unknown Categories")
    print("=" * 50)
    
    # Create new data with unknown transaction type
    new_data = df.copy()
    new_data.loc[0, 'type'] = 'UNKNOWN_TYPE'
    
    try:
        # Transform with fitted encoder (should handle unknown gracefully)
        new_encoded = encoder.transform(new_data.drop(columns=['isFraud']))
        print("Successfully handled unknown category")
        print(f"New data encoded shape: {new_encoded.shape}")
    except Exception as e:
        print(f"Error handling unknown category: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()