"""
Model evaluation utilities for fraud detection models.

This module provides comprehensive evaluation capabilities including
performance metrics calculation, cross-validation, and model comparison.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
import warnings
from .base_model import FraudModelInterface


class ModelEvaluator:
    """
    Comprehensive model evaluation class for fraud detection models.
    
    Provides methods to evaluate model performance using various metrics,
    perform cross-validation, and compare multiple models.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelEvaluator.
        
        Args:
            random_state: Random state for reproducible results
        """
        self.random_state = random_state
        self.evaluation_history = []
    
    def evaluate_model(self, 
                      model: FraudModelInterface, 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series,
                      model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained fraud detection model
            X_test: Test features
            y_test: Test labels
            model_name: Optional name for the model
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if not hasattr(model, 'is_trained') or not model.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Handle case where predict_proba returns single column
        if y_pred_proba.ndim == 1:
            y_pred_proba_positive = y_pred_proba
        else:
            y_pred_proba_positive = y_pred_proba[:, 1]
        
        # Calculate basic metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba_positive)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        # Calculate ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba_positive)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        }
        
        # Calculate Precision-Recall curve data
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_test, y_pred_proba_positive
        )
        metrics['precision_recall_curve'] = {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
        
        # Get feature importance if available
        try:
            feature_importance = model.get_feature_importance()
            metrics['feature_importance'] = feature_importance
        except (AttributeError, NotImplementedError):
            metrics['feature_importance'] = None
        
        # Add metadata
        metrics['model_name'] = model_name or getattr(model, 'model_name', 'Unknown')
        metrics['test_samples'] = len(X_test)
        metrics['positive_samples'] = int(y_test.sum())
        metrics['negative_samples'] = int(len(y_test) - y_test.sum())
        
        # Store in evaluation history
        self.evaluation_history.append(metrics.copy())
        
        return metrics
    
    def _calculate_metrics(self, 
                          y_true: pd.Series, 
                          y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate core performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        # AUC-ROC (handle edge cases)
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError as e:
            warnings.warn(f"Could not calculate AUC-ROC: {e}")
            metrics['auc_roc'] = None
        
        # Additional metrics for fraud detection
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        # False Positive Rate
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate
        metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def cross_validate_model(self, 
                           model: FraudModelInterface,
                           X: pd.DataFrame, 
                           y: pd.Series,
                           cv_folds: int = 5,
                           scoring: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform stratified cross-validation on a model.
        
        Args:
            model: Fraud detection model (must be untrained or retrained for each fold)
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            scoring: List of scoring metrics to use
            
        Returns:
            Dictionary containing cross-validation results
        """
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Create stratified k-fold cross-validator
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Initialize results storage
        cv_results = {metric: [] for metric in scoring}
        fold_details = []
        
        # Perform cross-validation manually to get detailed results
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Create a fresh model instance for this fold
            # Note: This assumes the model can be reset or a new instance created
            try:
                # Reset model state if possible
                if hasattr(model, '_reset_model'):
                    model._reset_model()
                elif hasattr(model, 'model'):
                    model.model = None
                    model.is_trained = False
            except:
                pass
            
            # Train model on fold
            model.train(X_train_fold, y_train_fold)
            
            # Evaluate on validation set
            fold_metrics = self.evaluate_model(model, X_val_fold, y_val_fold, 
                                             f"CV_Fold_{fold_idx + 1}")
            
            # Store results for each scoring metric
            for metric in scoring:
                if metric in fold_metrics:
                    cv_results[metric].append(fold_metrics[metric])
            
            fold_details.append({
                'fold': fold_idx + 1,
                'train_size': len(X_train_fold),
                'val_size': len(X_val_fold),
                'metrics': {k: v for k, v in fold_metrics.items() 
                           if k in scoring and v is not None}
            })
        
        # Calculate summary statistics
        cv_summary = {}
        for metric in scoring:
            if cv_results[metric] and all(v is not None for v in cv_results[metric]):
                scores = np.array(cv_results[metric])
                cv_summary[f'{metric}_mean'] = float(np.mean(scores))
                cv_summary[f'{metric}_std'] = float(np.std(scores))
                cv_summary[f'{metric}_scores'] = scores.tolist()
            else:
                cv_summary[f'{metric}_mean'] = None
                cv_summary[f'{metric}_std'] = None
                cv_summary[f'{metric}_scores'] = []
        
        return {
            'cv_summary': cv_summary,
            'fold_details': fold_details,
            'cv_folds': cv_folds,
            'total_samples': len(X),
            'positive_samples': int(y.sum()),
            'negative_samples': int(len(y) - y.sum())
        }
    
    def compare_models(self, 
                      model_results: List[Dict[str, Any]],
                      primary_metric: str = 'f1_score') -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: List of evaluation results from evaluate_model()
            primary_metric: Primary metric to use for ranking models
            
        Returns:
            Dictionary containing model comparison results
        """
        if not model_results:
            raise ValueError("No model results provided for comparison")
        
        # Create comparison table
        comparison_data = []
        for result in model_results:
            model_data = {
                'model_name': result.get('model_name', 'Unknown'),
                'accuracy': result.get('accuracy'),
                'precision': result.get('precision'),
                'recall': result.get('recall'),
                'f1_score': result.get('f1_score'),
                'auc_roc': result.get('auc_roc'),
                'test_samples': result.get('test_samples'),
                'positive_samples': result.get('positive_samples')
            }
            comparison_data.append(model_data)
        
        # Convert to DataFrame for easier manipulation
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by primary metric (descending order)
        if primary_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(
                primary_metric, ascending=False, na_position='last'
            )
            comparison_df['rank'] = range(1, len(comparison_df) + 1)
        
        # Find best model
        best_model_idx = 0
        best_model = comparison_data[best_model_idx] if comparison_data else None
        
        # Calculate performance differences
        performance_gaps = {}
        if len(comparison_data) > 1 and best_model:
            for i, model in enumerate(comparison_data[1:], 1):
                gaps = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                    if (best_model.get(metric) is not None and 
                        model.get(metric) is not None):
                        gaps[metric] = best_model[metric] - model[metric]
                performance_gaps[model['model_name']] = gaps
        
        return {
            'comparison_table': comparison_df.to_dict('records'),
            'best_model': best_model,
            'primary_metric': primary_metric,
            'performance_gaps': performance_gaps,
            'model_count': len(model_results)
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all evaluations performed.
        
        Returns:
            Dictionary containing evaluation history summary
        """
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        # Extract key metrics from all evaluations
        summary_data = []
        for eval_result in self.evaluation_history:
            summary_data.append({
                'model_name': eval_result.get('model_name'),
                'accuracy': eval_result.get('accuracy'),
                'precision': eval_result.get('precision'),
                'recall': eval_result.get('recall'),
                'f1_score': eval_result.get('f1_score'),
                'auc_roc': eval_result.get('auc_roc'),
                'test_samples': eval_result.get('test_samples')
            })
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'evaluations_summary': summary_data,
            'unique_models': len(set(eval_result.get('model_name', 'Unknown') 
                                   for eval_result in self.evaluation_history))
        }
    
    def clear_history(self) -> None:
        """Clear the evaluation history."""
        self.evaluation_history.clear()