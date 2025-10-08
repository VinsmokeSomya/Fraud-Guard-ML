#!/usr/bin/env python3
"""
Example script demonstrating the ModelVisualizer class for model performance visualization.

This script shows how to use the ModelVisualizer to create comprehensive visualizations
of fraud detection model performance, including ROC curves, precision-recall curves,
confusion matrices, feature importance plots, and SHAP analysis.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from visualization.model_visualizer import ModelVisualizer
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.model_evaluator import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_fraud_data(n_samples: int = 10000):
    """
    Create sample fraud detection data for demonstration.
    
    Args:
        n_samples: Number of sample transactions to generate
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    logger.info(f"Creating sample fraud dataset with {n_samples:,} transactions")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate features
    data = {
        'amount': np.random.lognormal(mean=8, sigma=2, size=n_samples),
        'oldbalanceOrg': np.random.exponential(scale=50000, size=n_samples),
        'newbalanceOrig': np.random.exponential(scale=50000, size=n_samples),
        'oldbalanceDest': np.random.exponential(scale=30000, size=n_samples),
        'newbalanceDest': np.random.exponential(scale=30000, size=n_samples),
        'step': np.random.randint(1, 745, n_samples),
    }
    
    # Transaction types
    transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    type_probs = [0.15, 0.20, 0.05, 0.45, 0.15]
    data['type'] = np.random.choice(transaction_types, n_samples, p=type_probs)
    
    # Encode transaction types
    le = LabelEncoder()
    data['type_encoded'] = le.fit_transform(data['type'])
    
    # Create derived features
    data['balance_change_orig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
    data['balance_change_dest'] = data['newbalanceDest'] - data['oldbalanceDest']
    data['amount_to_balance_ratio'] = np.where(
        data['oldbalanceOrg'] > 0, 
        data['amount'] / data['oldbalanceOrg'], 
        0
    )
    data['is_large_transfer'] = (
        (data['amount'] > 200000) & 
        (np.array([t == 'TRANSFER' for t in data['type']]))
    ).astype(int)
    
    df = pd.DataFrame(data)
    
    # Create fraud labels with realistic patterns
    fraud_prob = np.where(
        (df['type'].isin(['TRANSFER', 'CASH_OUT'])) & 
        (df['amount'] > 200000), 
        0.8,  # High fraud probability for large transfers/cash-outs
        np.where(
            df['amount_to_balance_ratio'] > 0.9,
            0.3,  # Medium fraud probability for high ratio transactions
            0.001  # Low base fraud rate
        )
    )
    
    y = np.random.binomial(1, fraud_prob)
    
    # Select numerical features for modeling
    feature_columns = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 
        'newbalanceDest', 'step', 'type_encoded', 'balance_change_orig',
        'balance_change_dest', 'amount_to_balance_ratio', 'is_large_transfer'
    ]
    
    X = df[feature_columns]
    y = pd.Series(y, name='isFraud')
    
    logger.info(f"Sample data created with {y.sum():,} fraudulent transactions "
               f"({y.mean()*100:.4f}% fraud rate)")
    
    return X, y


def demonstrate_single_model_visualization(visualizer: ModelVisualizer, 
                                         model, X_test: pd.DataFrame, 
                                         y_test: pd.Series, model_name: str) -> None:
    """
    Demonstrate visualization for a single model.
    
    Args:
        visualizer: ModelVisualizer instance
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    """
    logger.info(f"Creating visualizations for {model_name}")
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATIONS FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Handle different probability output formats
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
        y_pred_proba_positive = y_pred_proba[:, 1]
    else:
        y_pred_proba_positive = y_pred_proba
    
    # Get feature importance
    try:
        feature_importance = model.get_feature_importance()
    except (AttributeError, NotImplementedError):
        feature_importance = None
    
    # Individual plots
    print("\n1. ROC Curve:")
    roc_result = visualizer.plot_roc_curve(y_test, y_pred_proba_positive, model_name)
    
    print("\n2. Precision-Recall Curve:")
    pr_result = visualizer.plot_precision_recall_curve(y_test, y_pred_proba_positive, model_name)
    
    print("\n3. Confusion Matrix:")
    cm_result = visualizer.plot_confusion_matrix(y_test, y_pred, model_name)
    
    if feature_importance:
        print("\n4. Feature Importance:")
        fi_result = visualizer.plot_feature_importance(feature_importance, model_name, top_n=15)
    
    # Comprehensive dashboard
    print("\n5. Comprehensive Performance Dashboard:")
    visualizer.create_model_performance_dashboard(
        y_test, y_pred, y_pred_proba_positive, feature_importance, model_name
    )
    
    # SHAP analysis (if available)
    if hasattr(model, 'predict_proba'):
        print("\n6. SHAP Analysis:")
        # Use a small sample for SHAP analysis
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
        shap_result = visualizer.plot_shap_summary(model, X_sample, model_name, plot_type="summary")
        
        if 'error' not in shap_result:
            print("   SHAP summary plot created successfully")
        else:
            print(f"   SHAP analysis failed: {shap_result['error']}")


def demonstrate_interactive_visualization(visualizer: ModelVisualizer, 
                                        model, X_test: pd.DataFrame, 
                                        y_test: pd.Series, model_name: str) -> None:
    """
    Demonstrate interactive visualizations for a single model.
    
    Args:
        visualizer: ModelVisualizer instance
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    """
    logger.info(f"Creating interactive visualizations for {model_name}")
    
    print(f"\n{'='*60}")
    print(f"INTERACTIVE VISUALIZATIONS FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Handle different probability output formats
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
        y_pred_proba_positive = y_pred_proba[:, 1]
    else:
        y_pred_proba_positive = y_pred_proba
    
    # Get feature importance
    try:
        feature_importance = model.get_feature_importance()
    except (AttributeError, NotImplementedError):
        feature_importance = None
    
    # Interactive dashboard
    visualizer.create_model_performance_dashboard(
        y_test, y_pred, y_pred_proba_positive, feature_importance, model_name, interactive=True
    )


def demonstrate_model_comparison(visualizer: ModelVisualizer, 
                               models: List, model_names: List[str],
                               X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Demonstrate model comparison visualizations.
    
    Args:
        visualizer: ModelVisualizer instance
        models: List of trained models
        model_names: List of model names
        X_test: Test features
        y_test: Test labels
    """
    logger.info("Creating model comparison visualizations")
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Evaluate all models
    evaluator = ModelEvaluator()
    model_results = []
    
    for model, name in zip(models, model_names):
        result = evaluator.evaluate_model(model, X_test, y_test, name)
        model_results.append(result)
    
    # Create comparison visualizations
    print("\nStatic Model Comparison:")
    visualizer.compare_models_performance(model_results, interactive=False)
    
    print("\nInteractive Model Comparison:")
    visualizer.compare_models_performance(model_results, interactive=True)


def main():
    """Main function to demonstrate ModelVisualizer functionality."""
    logger.info("Starting ModelVisualizer demonstration")
    
    try:
        # Create sample data
        X, y = create_sample_fraud_data(n_samples=20000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        # Initialize visualizer
        visualizer = ModelVisualizer(
            style='whitegrid',
            palette='Set2',
            figure_size=(10, 8),
            dpi=100
        )
        
        # Train models
        print("Training models...")
        
        # Logistic Regression
        lr_model = LogisticRegressionModel(random_state=42)
        lr_model.train(X_train_scaled, y_train)
        
        # Random Forest
        rf_model = RandomForestModel(n_estimators=100, random_state=42)
        rf_model.train(X_train, y_train)  # Tree models don't need scaling
        
        # XGBoost
        xgb_model = XGBoostModel(random_state=42)
        xgb_model.train(X_train, y_train)
        
        models = [lr_model, rf_model, xgb_model]
        model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
        
        # Demonstrate single model visualizations
        for model, name in zip(models, model_names):
            test_data = X_test_scaled if name == "Logistic Regression" else X_test
            demonstrate_single_model_visualization(visualizer, model, test_data, y_test, name)
        
        # Ask user for interactive visualizations
        response = input("\nWould you like to see interactive visualizations? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            # Show interactive for one model as example
            demonstrate_interactive_visualization(
                visualizer, rf_model, X_test, y_test, "Random Forest"
            )
        
        # Ask user for model comparison
        response = input("\nWould you like to see model comparison visualizations? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            test_datasets = [X_test_scaled, X_test, X_test]  # Scaled for LR, unscaled for tree models
            demonstrate_model_comparison(visualizer, models, model_names, X_test, y_test)
        
        # Ask user about saving plots
        response = input("\nWould you like to save plots to files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            save_path = input("Enter save directory path (or press Enter for 'model_plots'): ").strip()
            if not save_path:
                save_path = "model_plots"
            
            for model, name in zip(models, model_names):
                test_data = X_test_scaled if name == "Logistic Regression" else X_test
                y_pred = model.predict(test_data)
                y_pred_proba = model.predict_proba(test_data)
                
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    y_pred_proba_positive = y_pred_proba[:, 1]
                else:
                    y_pred_proba_positive = y_pred_proba
                
                try:
                    feature_importance = model.get_feature_importance()
                except (AttributeError, NotImplementedError):
                    feature_importance = None
                
                visualizer.save_plots_to_file(
                    y_test, y_pred, y_pred_proba_positive, 
                    feature_importance, name, save_path
                )
            
            print(f"Plots saved to {save_path}")
        
        logger.info("ModelVisualizer demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()