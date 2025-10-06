"""
Random Forest Model Example for Fraud Detection.

This example demonstrates how to use the RandomForestModel class
for fraud detection, including training, evaluation, and feature importance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.random_forest_model import RandomForestModel


def create_sample_fraud_data():
    """Create sample fraud detection dataset."""
    print("Creating sample fraud detection dataset...")
    
    # Generate synthetic fraud detection data
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
        'transaction_frequency', 'amount_deviation', 'location_risk',
        'payment_method_risk', 'velocity_score', 'behavioral_score'
    ]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='is_fraud')
    
    print(f"Dataset created: {len(X_df)} samples, {len(feature_names)} features")
    print(f"Fraud rate: {y_series.mean():.2%}")
    
    return X_df, y_series


def train_random_forest_models(X_train, y_train):
    """Train Random Forest models with and without hyperparameter tuning."""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODELS")
    print("="*60)
    
    # Model 1: Basic Random Forest without tuning
    print("\n1. Training Random Forest without hyperparameter tuning...")
    rf_basic = RandomForestModel(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        tune_hyperparameters=False,
        oob_score=True
    )
    rf_basic.train(X_train, y_train)
    
    print(f"   Training completed in {rf_basic.metadata['training_time']:.2f} seconds")
    print(f"   OOB Score: {rf_basic.get_oob_score():.4f}")
    
    # Model 2: Random Forest with hyperparameter tuning
    print("\n2. Training Random Forest with hyperparameter tuning...")
    rf_tuned = RandomForestModel(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        tune_hyperparameters=True,
        cv_folds=3,
        scoring='f1'
    )
    rf_tuned.train(X_train, y_train)
    
    print(f"   Training completed in {rf_tuned.metadata['training_time']:.2f} seconds")
    print(f"   Best CV Score: {rf_tuned.best_score_:.4f}")
    print(f"   Best Parameters: {rf_tuned.best_params_}")
    print(f"   OOB Score: {rf_tuned.get_oob_score():.4f}")
    
    return rf_basic, rf_tuned


def evaluate_models(models, X_test, y_test):
    """Evaluate the trained models."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name} Results:")
        print("-" * 40)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_fraud_probability(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    return results


def analyze_feature_importance(models, feature_names):
    """Analyze and visualize feature importance."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{name} - Top 10 Most Important Features:")
        print("-" * 50)
        
        # Get feature importance
        importance = model.get_feature_importance()
        importance_std = model.get_feature_importance_std()
        
        # Display top 10 features
        for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
            std = importance_std[feature]
            print(f"{i:2d}. {feature:<25} {score:.4f} (±{std:.4f})")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        features = list(importance.keys())[:10]
        scores = list(importance.values())[:10]
        stds = [importance_std[f] for f in features]
        
        bars = plt.bar(range(len(features)), scores, yerr=stds, capsize=5)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(f'{name} - Feature Importance (Top 10)')
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def demonstrate_individual_predictions(model, X_test, y_test):
    """Demonstrate individual tree predictions and consensus."""
    print("\n" + "="*60)
    print("INDIVIDUAL TREE ANALYSIS")
    print("="*60)
    
    # Get a few test samples
    sample_indices = [0, 1, 2, 3, 4]
    X_sample = X_test.iloc[sample_indices]
    y_sample = y_test.iloc[sample_indices]
    
    # Get individual tree predictions
    tree_predictions = model.get_individual_tree_predictions(X_sample)
    ensemble_predictions = model.predict(X_sample)
    fraud_probabilities = model.predict_fraud_probability(X_sample)
    
    print("Sample predictions from individual trees:")
    print("(Showing first 10 trees for brevity)")
    print("-" * 60)
    
    for i, (idx, true_label) in enumerate(zip(sample_indices, y_sample)):
        individual_preds = tree_predictions[i, :10]  # First 10 trees
        ensemble_pred = ensemble_predictions[i]
        fraud_prob = fraud_probabilities[i]
        
        print(f"\nSample {idx} (True: {'Fraud' if true_label else 'Legit'}):")
        print(f"  Individual trees: {individual_preds}")
        print(f"  Ensemble pred:    {ensemble_pred} ({'Fraud' if ensemble_pred else 'Legit'})")
        print(f"  Fraud probability: {fraud_prob:.4f}")


def main():
    """Main function to run the Random Forest example."""
    print("Random Forest Model Example for Fraud Detection")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_fraud_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    rf_basic, rf_tuned = train_random_forest_models(X_train, y_train)
    
    # Evaluate models
    models = {
        'Random Forest (Basic)': rf_basic,
        'Random Forest (Tuned)': rf_tuned
    }
    
    results = evaluate_models(models, X_test, y_test)
    
    # Analyze feature importance
    analyze_feature_importance(models, X.columns)
    
    # Demonstrate individual tree predictions
    demonstrate_individual_predictions(rf_tuned, X_test, y_test)
    
    # Model summaries
    print("\n" + "="*60)
    print("MODEL SUMMARIES")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{name}:")
        summary = model.get_model_summary()
        for key, value in summary.items():
            if key != 'top_features':
                print(f"  {key}: {value}")
        
        print("  Top 3 features:")
        for feature, importance in list(summary['top_features'].items())[:3]:
            print(f"    {feature}: {importance:.4f}")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()