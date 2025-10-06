"""
Example demonstrating the ModelEvaluator class usage.

This example shows how to use the ModelEvaluator to evaluate fraud detection models,
perform cross-validation, and compare multiple models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models import ModelEvaluator, LogisticRegressionModel, RandomForestModel


def create_sample_fraud_data():
    """Create sample fraud detection data for demonstration."""
    print("Creating sample fraud detection data...")
    
    # Generate synthetic fraud data
    X, y = make_classification(
        n_samples=5000,
        n_features=15,
        n_informative=12,
        n_redundant=3,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],  # Highly imbalanced like real fraud data
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [
        'transaction_amount', 'account_balance', 'days_since_last_transaction',
        'transaction_hour', 'is_weekend', 'merchant_risk_score',
        'customer_age', 'account_age_days', 'previous_fraud_count',
        'transaction_frequency', 'amount_deviation', 'location_risk',
        'payment_method_risk', 'customer_risk_score', 'velocity_score'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='is_fraud')
    
    print(f"Dataset created: {len(X_df)} samples, {len(feature_names)} features")
    print(f"Fraud ratio: {y_series.mean():.3f}")
    
    return X_df, y_series


def demonstrate_model_evaluation():
    """Demonstrate basic model evaluation functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL EVALUATION")
    print("="*60)
    
    # Create data
    X, y = create_sample_fraud_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(random_state=42)
    
    # Train and evaluate Logistic Regression
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegressionModel(random_state=42)
    lr_model.train(X_train, y_train)
    
    print("Evaluating Logistic Regression model...")
    lr_results = evaluator.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Print key metrics
    print(f"\nLogistic Regression Results:")
    print(f"  Accuracy:  {lr_results['accuracy']:.4f}")
    print(f"  Precision: {lr_results['precision']:.4f}")
    print(f"  Recall:    {lr_results['recall']:.4f}")
    print(f"  F1-Score:  {lr_results['f1_score']:.4f}")
    print(f"  AUC-ROC:   {lr_results['auc_roc']:.4f}")
    
    # Show confusion matrix
    cm = lr_results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    return evaluator, X_train, y_train, X_test, y_test


def demonstrate_cross_validation():
    """Demonstrate cross-validation functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING CROSS-VALIDATION")
    print("="*60)
    
    # Create data
    X, y = create_sample_fraud_data()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(random_state=42)
    
    # Perform cross-validation on Random Forest
    print("\nPerforming 3-fold cross-validation on Random Forest...")
    rf_model = RandomForestModel(n_estimators=10, random_state=42)  # Smaller for demo
    
    cv_results = evaluator.cross_validate_model(
        rf_model, X, y, cv_folds=5
    )
    
    # Print cross-validation results
    print(f"\nCross-Validation Results:")
    summary = cv_results['cv_summary']
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        mean_score = summary.get(f'{metric}_mean')
        std_score = summary.get(f'{metric}_std')
        if mean_score is not None:
            print(f"  {metric.upper()}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    print(f"\nTotal samples: {cv_results['total_samples']}")
    print(f"Positive samples: {cv_results['positive_samples']}")
    print(f"CV folds: {cv_results['cv_folds']}")


def demonstrate_model_comparison():
    """Demonstrate model comparison functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL COMPARISON")
    print("="*60)
    
    # Create data
    X, y = create_sample_fraud_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(random_state=42)
    
    # Train and evaluate multiple models
    models_to_compare = []
    
    print("\nTraining and evaluating multiple models...")
    
    # Logistic Regression
    print("  - Logistic Regression")
    lr_model = LogisticRegressionModel(random_state=42)
    lr_model.train(X_train, y_train)
    lr_results = evaluator.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    models_to_compare.append(lr_results)
    
    # Random Forest
    print("  - Random Forest")
    rf_model = RandomForestModel(n_estimators=50, random_state=42)
    rf_model.train(X_train, y_train)
    rf_results = evaluator.evaluate_model(rf_model, X_test, y_test, "Random Forest")
    models_to_compare.append(rf_results)
    
    # Compare models
    print("\nComparing models...")
    comparison = evaluator.compare_models(models_to_compare, primary_metric='f1_score')
    
    # Print comparison results
    print(f"\nModel Comparison (ranked by F1-Score):")
    print("-" * 80)
    print(f"{'Rank':<4} {'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-" * 80)
    
    for model_data in comparison['comparison_table']:
        print(f"{model_data['rank']:<4} {model_data['model_name']:<20} "
              f"{model_data['accuracy']:<10.4f} {model_data['precision']:<10.4f} "
              f"{model_data['recall']:<10.4f} {model_data['f1_score']:<10.4f} "
              f"{model_data['auc_roc']:<10.4f}")
    
    # Show best model
    best_model = comparison['best_model']
    print(f"\nBest Model: {best_model['model_name']}")
    print(f"Best F1-Score: {best_model['f1_score']:.4f}")


def demonstrate_evaluation_history():
    """Demonstrate evaluation history tracking."""
    print("\n" + "="*60)
    print("DEMONSTRATING EVALUATION HISTORY")
    print("="*60)
    
    # Use the evaluator from previous demonstrations
    evaluator = ModelEvaluator(random_state=42)
    
    # Create some dummy evaluations
    X, y = create_sample_fraud_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Evaluate a few models
    lr_model = LogisticRegressionModel(random_state=42)
    lr_model.train(X_train, y_train)
    evaluator.evaluate_model(lr_model, X_test, y_test, "LR_Test1")
    evaluator.evaluate_model(lr_model, X_test, y_test, "LR_Test2")
    
    rf_model = RandomForestModel(n_estimators=30, random_state=42)
    rf_model.train(X_train, y_train)
    evaluator.evaluate_model(rf_model, X_test, y_test, "RF_Test1")
    
    # Show evaluation history
    history_summary = evaluator.get_evaluation_summary()
    
    print(f"\nEvaluation History Summary:")
    print(f"Total evaluations: {history_summary['total_evaluations']}")
    print(f"Unique models: {history_summary['unique_models']}")
    
    print(f"\nAll evaluations:")
    for i, eval_data in enumerate(history_summary['evaluations_summary'], 1):
        print(f"  {i}. {eval_data['model_name']}: F1={eval_data['f1_score']:.4f}, "
              f"AUC={eval_data['auc_roc']:.4f}")


def main():
    """Run all demonstration examples."""
    print("ModelEvaluator Demonstration")
    print("="*60)
    
    try:
        # Run demonstrations
        demonstrate_model_evaluation()
        demonstrate_cross_validation()
        demonstrate_model_comparison()
        demonstrate_evaluation_history()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()