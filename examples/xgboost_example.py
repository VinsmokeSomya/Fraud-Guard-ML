"""
XGBoost Model Example for Fraud Detection

This example demonstrates how to use the XGBoostModel class for fraud detection,
including training, prediction, feature importance analysis, and SHAP interpretability.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import the XGBoost model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.xgboost_model import XGBoostModel


def create_sample_fraud_data():
    """Create a sample fraud detection dataset."""
    print("Creating sample fraud detection dataset...")
    
    # Generate synthetic fraud data
    X, y = make_classification(
        n_samples=5000,
        n_features=15,
        n_informative=12,
        n_redundant=3,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],  # 5% fraud rate (realistic)
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [
        'transaction_amount', 'account_balance', 'days_since_last_transaction',
        'transaction_hour', 'is_weekend', 'merchant_risk_score',
        'customer_age', 'account_age_days', 'previous_fraud_count',
        'transaction_frequency', 'amount_to_balance_ratio', 'location_risk',
        'payment_method_risk', 'time_since_account_creation', 'velocity_score'
    ]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='is_fraud')
    
    print(f"Dataset created: {len(X_df)} samples, {len(feature_names)} features")
    print(f"Fraud rate: {y_series.mean():.1%}")
    
    return X_df, y_series


def demonstrate_xgboost_model():
    """Demonstrate XGBoost model capabilities."""
    print("=" * 60)
    print("XGBoost Fraud Detection Model Example")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_fraud_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Example 1: Basic XGBoost model (fast training)
    print("\n" + "="*50)
    print("1. Basic XGBoost Model (No Hyperparameter Tuning)")
    print("="*50)
    
    basic_model = XGBoostModel(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        tune_hyperparameters=False,
        early_stopping_rounds=10,
        random_state=42
    )
    
    print("Training basic model...")
    basic_model.train(X_train, y_train)
    
    # Make predictions
    y_pred_basic = basic_model.predict(X_test)
    y_prob_basic = basic_model.predict_fraud_probability(X_test)
    
    print("\nBasic Model Performance:")
    print(classification_report(y_test, y_pred_basic, target_names=['Legitimate', 'Fraud']))
    
    # Example 2: XGBoost with different parameters
    print("\n" + "="*50)
    print("2. XGBoost with Custom Parameters")
    print("="*50)
    
    tuned_model = XGBoostModel(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tune_hyperparameters=False,  # Disabled for demo speed
        early_stopping_rounds=0,  # Disabled for demo
        random_state=42
    )
    
    print("Training custom model...")
    tuned_model.train(X_train, y_train)
    
    # Make predictions
    y_pred_tuned = tuned_model.predict(X_test)
    y_prob_tuned = tuned_model.predict_fraud_probability(X_test)
    
    print("\nCustom Model Performance:")
    print(classification_report(y_test, y_pred_tuned, target_names=['Legitimate', 'Fraud']))
    
    # Example 3: Feature importance analysis
    print("\n" + "="*50)
    print("3. Feature Importance Analysis")
    print("="*50)
    
    # Get feature importance from custom model
    feature_importance = tuned_model.get_feature_importance()
    
    print("Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"{i:2d}. {feature:<30} {importance:.4f}")
    
    # Get different types of XGBoost feature importance
    print("\nXGBoost Feature Importance by Type:")
    for importance_type in ['weight', 'gain', 'cover']:
        importance_by_type = tuned_model.get_feature_importance_by_type(importance_type)
        top_feature = list(importance_by_type.items())[0]
        print(f"{importance_type.capitalize():<8}: {top_feature[0]} ({top_feature[1]:.4f})")
    
    # Example 4: SHAP interpretability
    print("\n" + "="*50)
    print("4. SHAP Model Interpretability")
    print("="*50)
    
    try:
        # Initialize SHAP explainer
        print("Initializing SHAP explainer...")
        tuned_model.initialize_shap_explainer(X_train.sample(100, random_state=42))
        
        # Get SHAP feature importance
        shap_importance = tuned_model.get_shap_feature_importance(X_test.iloc[:100])
        
        print("Top 5 Features by SHAP Importance:")
        for i, (feature, importance) in enumerate(list(shap_importance.items())[:5], 1):
            print(f"{i}. {feature:<30} {importance:.4f}")
        
        # Explain a specific prediction
        print("\nExplaining a specific fraud prediction:")
        fraud_indices = np.where(y_test == 1)[0]
        if len(fraud_indices) > 0:
            fraud_idx = fraud_indices[0]
            explanation = tuned_model.explain_prediction(X_test, fraud_idx)
            
            print(f"Sample {fraud_idx}:")
            print(f"  Prediction: {'Fraud' if explanation['prediction'] else 'Legitimate'}")
            print(f"  Fraud Probability: {explanation['fraud_probability']:.4f}")
            print(f"  Base Value: {explanation['base_value']:.4f}")
            
            print("  Top 3 Contributing Features:")
            shap_values = explanation['shap_values']
            sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, shap_val) in enumerate(sorted_shap[:3], 1):
                feature_val = explanation['feature_values'][feature]
                print(f"    {i}. {feature}: {shap_val:+.4f} (value: {feature_val:.4f})")
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("This might be due to environment limitations.")
    
    # Example 5: Cross-validation
    print("\n" + "="*50)
    print("5. Cross-Validation Results")
    print("="*50)
    
    cv_results = tuned_model.cross_validate_model(X_train, y_train)
    print(f"Cross-validation scores: {[f'{score:.4f}' for score in cv_results['cv_scores']]}")
    print(f"Mean CV score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    # Example 6: Model summary and comparison
    print("\n" + "="*50)
    print("6. Model Summary and Comparison")
    print("="*50)
    
    basic_summary = basic_model.get_model_summary()
    custom_summary = tuned_model.get_model_summary()
    
    print("Basic Model Summary:")
    print(f"  Model Type: {basic_summary['model_type']}")
    print(f"  N Estimators: {basic_summary['n_estimators']}")
    print(f"  Feature Count: {basic_summary['feature_count']}")
    print(f"  GPU Enabled: {basic_summary['use_gpu']}")
    
    print("\nCustom Model Summary:")
    print(f"  Model Type: {custom_summary['model_type']}")
    print(f"  N Estimators: {custom_summary['n_estimators']}")
    print(f"  Feature Count: {custom_summary['feature_count']}")
    print(f"  GPU Enabled: {custom_summary['use_gpu']}")
    
    # Example 7: Model persistence
    print("\n" + "="*50)
    print("7. Model Persistence")
    print("="*50)
    
    # Save the custom model
    model_path = "models/demo_xgboost_model"
    tuned_model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Load and verify
    loaded_model = XGBoostModel()
    loaded_model.load_model(model_path)
    
    # Test loaded model
    loaded_predictions = loaded_model.predict(X_test.iloc[:10])
    original_predictions = tuned_model.predict(X_test.iloc[:10])
    
    print(f"Loaded model works correctly: {np.array_equal(loaded_predictions, original_predictions)}")
    
    print("\n" + "="*60)
    print("XGBoost Model Example Complete!")
    print("="*60)


if __name__ == "__main__":
    demonstrate_xgboost_model()