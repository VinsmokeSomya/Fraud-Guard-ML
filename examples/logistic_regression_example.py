"""
Example usage of LogisticRegressionModel for fraud detection.

This example demonstrates how to use the LogisticRegressionModel class
with sample data, including training, prediction, and feature importance analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models import LogisticRegressionModel


def create_sample_fraud_data():
    """Create sample fraud detection dataset."""
    print("Creating sample fraud detection dataset...")
    
    # Generate imbalanced binary classification data similar to fraud detection
    X, y = make_classification(
        n_samples=5000,
        n_features=15,
        n_informative=12,
        n_redundant=3,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],  # 5% fraud rate
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [
        'transaction_amount', 'account_balance', 'days_since_last_transaction',
        'transaction_hour', 'is_weekend', 'merchant_risk_score',
        'customer_age', 'account_age_days', 'previous_fraud_count',
        'transaction_velocity', 'amount_to_balance_ratio', 'is_high_value',
        'location_risk_score', 'payment_method_risk', 'time_since_last_login'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='is_fraud')
    
    print(f"Dataset created: {len(X_df)} samples, {len(feature_names)} features")
    print(f"Fraud rate: {y_series.mean():.2%}")
    
    return X_df, y_series


def demonstrate_basic_usage():
    """Demonstrate basic LogisticRegressionModel usage."""
    print("\n" + "="*60)
    print("BASIC LOGISTIC REGRESSION MODEL USAGE")
    print("="*60)
    
    # Create sample data
    X, y = create_sample_fraud_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize model without hyperparameter tuning for speed
    print("\nInitializing LogisticRegressionModel...")
    model = LogisticRegressionModel(
        tune_hyperparameters=False,
        class_weight='balanced',
        random_state=42
    )
    
    # Train the model
    print("Training model...")
    model.train(X_train, y_train)
    
    # Get training info
    training_info = model.get_training_info()
    print(f"Training completed in {training_info['training_time']:.2f} seconds")
    print(f"Training samples: {training_info['training_samples']}")
    print(f"Features: {training_info['feature_count']}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    fraud_probs = model.predict_fraud_probability(X_test)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Fraud probabilities range: {fraud_probs.min():.3f} - {fraud_probs.max():.3f}")
    
    # Calculate basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f"\nModel Performance:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    
    return model, X_test, y_test


def demonstrate_feature_importance(model):
    """Demonstrate feature importance analysis."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance
    importance = model.get_feature_importance()
    coefficients = model.get_coefficients()
    intercept = model.get_intercept()
    
    print(f"Model intercept: {intercept:.3f}")
    print("\nTop 10 Most Important Features:")
    print("-" * 50)
    
    for i, (feature, importance_score) in enumerate(list(importance.items())[:10], 1):
        coef = coefficients[feature]
        direction = "↑" if coef > 0 else "↓"
        print(f"{i:2d}. {feature:<25} {importance_score:8.3f} {direction} ({coef:+.3f})")
    
    return importance


def demonstrate_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning capabilities."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING DEMONSTRATION")
    print("="*60)
    
    # Create smaller dataset for faster tuning
    X, y = create_sample_fraud_data()
    X_small = X.sample(n=1000, random_state=42)
    y_small = y.loc[X_small.index]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.2, random_state=42, stratify=y_small
    )
    
    # Initialize model with hyperparameter tuning
    print("Initializing model with hyperparameter tuning...")
    model = LogisticRegressionModel(
        tune_hyperparameters=True,
        cv_folds=3,  # Reduced for speed
        scoring='f1',
        random_state=42
    )
    
    # Override parameter grid for faster demonstration
    def get_demo_param_grid():
        return [
            {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['liblinear']
            },
            {
                'C': [0.1, 1.0],
                'penalty': ['l1'],
                'solver': ['liblinear']
            }
        ]
    
    model._get_param_grid = get_demo_param_grid
    
    # Train with tuning
    print("Training with hyperparameter tuning...")
    model.train(X_train, y_train)
    
    # Get tuning results
    tuning_results = model.get_tuning_results()
    if tuning_results:
        print(f"\nBest parameters: {tuning_results['best_params']}")
        print(f"Best CV score: {tuning_results['best_score']:.3f}")
    
    # Get model summary
    summary = model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"Model type: {summary['model_type']}")
    print(f"Feature count: {summary['feature_count']}")
    print(f"Hyperparameter tuning: {summary['hyperparameter_tuning']}")
    
    return model


def demonstrate_model_persistence(model):
    """Demonstrate model saving and loading."""
    print("\n" + "="*60)
    print("MODEL PERSISTENCE DEMONSTRATION")
    print("="*60)
    
    # Save the model
    model_path = "models/demo_logistic_regression"
    print(f"Saving model to {model_path}...")
    model.save_model(model_path)
    
    # Load the model
    print("Loading model from disk...")
    loaded_model = LogisticRegressionModel()
    loaded_model.load_model(model_path)
    
    print("Model loaded successfully!")
    print(f"Loaded model is trained: {loaded_model.is_trained}")
    print(f"Feature count: {len(loaded_model.feature_names)}")
    
    return loaded_model


def main():
    """Run all demonstrations."""
    print("LogisticRegressionModel Demonstration")
    print("=" * 60)
    
    try:
        # Basic usage
        model, X_test, y_test = demonstrate_basic_usage()
        
        # Feature importance
        demonstrate_feature_importance(model)
        
        # Hyperparameter tuning
        tuned_model = demonstrate_hyperparameter_tuning()
        
        # Model persistence
        demonstrate_model_persistence(model)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()