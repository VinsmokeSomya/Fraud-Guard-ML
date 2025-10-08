"""
Model evaluation utilities for fraud detection models.

This module provides comprehensive evaluation capabilities including
performance metrics calculation, cross-validation, and model comparison.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
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
    
    def create_comparison_table(self, 
                              model_results: List[Dict[str, Any]],
                              metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a comprehensive model performance comparison table.
        
        Args:
            model_results: List of evaluation results from evaluate_model()
            metrics: List of metrics to include in comparison
            
        Returns:
            DataFrame with model comparison data
        """
        if not model_results:
            raise ValueError("No model results provided for comparison")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 
                      'specificity', 'false_positive_rate', 'false_negative_rate']
        
        # Create comparison data
        comparison_data = []
        for result in model_results:
            model_data = {'model_name': result.get('model_name', 'Unknown')}
            
            # Add requested metrics
            for metric in metrics:
                model_data[metric] = result.get(metric)
            
            # Add additional useful information
            model_data.update({
                'test_samples': result.get('test_samples'),
                'positive_samples': result.get('positive_samples'),
                'negative_samples': result.get('negative_samples'),
                'true_positives': result.get('true_positives'),
                'true_negatives': result.get('true_negatives'),
                'false_positives': result.get('false_positives'),
                'false_negatives': result.get('false_negatives')
            })
            
            comparison_data.append(model_data)
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add ranking columns for key metrics
        ranking_metrics = ['f1_score', 'auc_roc', 'precision', 'recall']
        for metric in ranking_metrics:
            if metric in comparison_df.columns:
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(
                    ascending=False, method='min', na_option='bottom'
                )
        
        return comparison_df
    
    def statistical_significance_test(self,
                                    model1_cv_scores: List[float],
                                    model2_cv_scores: List[float],
                                    test_type: str = 'paired_ttest',
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance testing between two models.
        
        Args:
            model1_cv_scores: Cross-validation scores for model 1
            model2_cv_scores: Cross-validation scores for model 2
            test_type: Type of statistical test ('paired_ttest', 'wilcoxon', 'mannwhitney')
            alpha: Significance level
            
        Returns:
            Dictionary containing test results
        """
        if len(model1_cv_scores) != len(model2_cv_scores):
            if test_type in ['paired_ttest', 'wilcoxon']:
                raise ValueError("Paired tests require equal number of scores")
        
        # Convert to numpy arrays
        scores1 = np.array(model1_cv_scores)
        scores2 = np.array(model2_cv_scores)
        
        # Remove any NaN values
        valid_mask1 = ~np.isnan(scores1)
        valid_mask2 = ~np.isnan(scores2)
        
        if test_type in ['paired_ttest', 'wilcoxon']:
            valid_mask = valid_mask1 & valid_mask2
            scores1 = scores1[valid_mask]
            scores2 = scores2[valid_mask]
        else:
            scores1 = scores1[valid_mask1]
            scores2 = scores2[valid_mask2]
        
        if len(scores1) < 2 or len(scores2) < 2:
            return {
                'test_type': test_type,
                'statistic': None,
                'p_value': None,
                'significant': None,
                'error': 'Insufficient valid scores for testing'
            }
        
        # Perform the appropriate statistical test
        try:
            if test_type == 'paired_ttest':
                statistic, p_value = ttest_rel(scores1, scores2)
            elif test_type == 'wilcoxon':
                statistic, p_value = wilcoxon(scores1, scores2)
            elif test_type == 'mannwhitney':
                statistic, p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Determine significance
            significant = p_value < alpha
            
            # Calculate effect size (Cohen's d for t-test)
            effect_size = None
            if test_type == 'paired_ttest':
                diff = scores1 - scores2
                effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            
            return {
                'test_type': test_type,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': significant,
                'alpha': alpha,
                'effect_size': effect_size,
                'mean_diff': float(np.mean(scores1) - np.mean(scores2)),
                'model1_mean': float(np.mean(scores1)),
                'model2_mean': float(np.mean(scores2)),
                'model1_std': float(np.std(scores1, ddof=1)),
                'model2_std': float(np.std(scores2, ddof=1)),
                'sample_size': len(scores1)
            }
            
        except Exception as e:
            return {
                'test_type': test_type,
                'statistic': None,
                'p_value': None,
                'significant': None,
                'error': str(e)
            }
    
    def compare_models_with_significance(self,
                                       cv_results: List[Dict[str, Any]],
                                       metric: str = 'f1_score',
                                       test_type: str = 'paired_ttest',
                                       alpha: float = 0.05) -> Dict[str, Any]:
        """
        Compare multiple models with statistical significance testing.
        
        Args:
            cv_results: List of cross-validation results from cross_validate_model()
            metric: Metric to use for comparison
            test_type: Statistical test type
            alpha: Significance level
            
        Returns:
            Dictionary containing pairwise comparison results
        """
        if len(cv_results) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        # Extract model names and scores
        model_names = []
        model_scores = []
        
        for cv_result in cv_results:
            # Try to extract model name from fold details or use index
            if cv_result.get('fold_details'):
                model_name = cv_result['fold_details'][0].get('metrics', {}).get('model_name', f"Model_{len(model_names)}")
            else:
                model_name = f"Model_{len(model_names)}"
            
            model_names.append(model_name)
            
            # Extract scores for the specified metric
            scores_key = f'{metric}_scores'
            if scores_key in cv_result.get('cv_summary', {}):
                scores = cv_result['cv_summary'][scores_key]
                model_scores.append(scores)
            else:
                # Try to extract from fold details
                scores = []
                for fold in cv_result.get('fold_details', []):
                    if metric in fold.get('metrics', {}):
                        scores.append(fold['metrics'][metric])
                model_scores.append(scores)
        
        # Perform pairwise comparisons
        pairwise_results = {}
        comparison_matrix = pd.DataFrame(index=model_names, columns=model_names)
        
        for i, (name1, scores1) in enumerate(zip(model_names, model_scores)):
            for j, (name2, scores2) in enumerate(zip(model_names, model_scores)):
                if i != j:
                    comparison_key = f"{name1}_vs_{name2}"
                    test_result = self.statistical_significance_test(
                        scores1, scores2, test_type, alpha
                    )
                    pairwise_results[comparison_key] = test_result
                    
                    # Fill comparison matrix with p-values
                    comparison_matrix.loc[name1, name2] = test_result.get('p_value')
                else:
                    comparison_matrix.loc[name1, name2] = 1.0  # Same model
        
        # Create summary statistics
        summary_stats = []
        for name, scores in zip(model_names, model_scores):
            if scores:
                summary_stats.append({
                    'model_name': name,
                    f'{metric}_mean': np.mean(scores),
                    f'{metric}_std': np.std(scores, ddof=1),
                    f'{metric}_min': np.min(scores),
                    f'{metric}_max': np.max(scores),
                    'cv_folds': len(scores)
                })
        
        return {
            'pairwise_comparisons': pairwise_results,
            'comparison_matrix': comparison_matrix.astype(float),
            'summary_statistics': summary_stats,
            'test_parameters': {
                'metric': metric,
                'test_type': test_type,
                'alpha': alpha
            }
        }
    
    def select_best_model(self,
                         model_results: List[Dict[str, Any]],
                         business_metrics: Optional[Dict[str, float]] = None,
                         primary_metric: str = 'f1_score',
                         min_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Select the best model based on business metrics and constraints.
        
        Args:
            model_results: List of evaluation results from evaluate_model()
            business_metrics: Weights for different metrics (e.g., {'precision': 0.4, 'recall': 0.6})
            primary_metric: Primary metric for ranking if business_metrics not provided
            min_thresholds: Minimum acceptable values for metrics
            
        Returns:
            Dictionary containing best model selection results
        """
        if not model_results:
            raise ValueError("No model results provided")
        
        # Default business metrics for fraud detection
        if business_metrics is None:
            business_metrics = {
                'precision': 0.3,  # Important to minimize false positives
                'recall': 0.4,     # Critical to catch fraud
                'f1_score': 0.2,   # Balance between precision and recall
                'auc_roc': 0.1     # Overall discriminative ability
            }
        
        # Default minimum thresholds for fraud detection
        if min_thresholds is None:
            min_thresholds = {
                'precision': 0.5,  # At least 50% precision
                'recall': 0.6,     # At least 60% recall
                'f1_score': 0.5    # At least 50% F1-score
            }
        
        # Filter models that meet minimum thresholds
        qualified_models = []
        disqualified_models = []
        
        for result in model_results:
            model_name = result.get('model_name', 'Unknown')
            meets_thresholds = True
            failed_thresholds = []
            
            for metric, threshold in min_thresholds.items():
                model_value = result.get(metric)
                if model_value is None or model_value < threshold:
                    meets_thresholds = False
                    failed_thresholds.append(f"{metric}: {model_value} < {threshold}")
            
            if meets_thresholds:
                qualified_models.append(result)
            else:
                disqualified_models.append({
                    'model_name': model_name,
                    'failed_thresholds': failed_thresholds
                })
        
        if not qualified_models:
            return {
                'best_model': None,
                'selection_reason': 'No models met minimum thresholds',
                'disqualified_models': disqualified_models,
                'business_metrics': business_metrics,
                'min_thresholds': min_thresholds
            }
        
        # Calculate business scores for qualified models
        model_scores = []
        for result in qualified_models:
            business_score = 0.0
            metric_contributions = {}
            
            for metric, weight in business_metrics.items():
                value = result.get(metric, 0)
                contribution = value * weight
                business_score += contribution
                metric_contributions[metric] = {
                    'value': value,
                    'weight': weight,
                    'contribution': contribution
                }
            
            model_scores.append({
                'model_result': result,
                'business_score': business_score,
                'metric_contributions': metric_contributions
            })
        
        # Sort by business score (descending)
        model_scores.sort(key=lambda x: x['business_score'], reverse=True)
        
        # Select best model
        best_model_info = model_scores[0]
        best_model = best_model_info['model_result']
        
        # Create ranking of all qualified models
        model_ranking = []
        for i, model_info in enumerate(model_scores):
            model_ranking.append({
                'rank': i + 1,
                'model_name': model_info['model_result'].get('model_name'),
                'business_score': model_info['business_score'],
                'metric_contributions': model_info['metric_contributions']
            })
        
        # Calculate performance advantages of best model
        performance_advantages = {}
        if len(model_scores) > 1:
            second_best = model_scores[1]['model_result']
            for metric in business_metrics.keys():
                best_value = best_model.get(metric, 0)
                second_value = second_best.get(metric, 0)
                if second_value > 0:
                    improvement = ((best_value - second_value) / second_value) * 100
                    performance_advantages[metric] = {
                        'best_model_value': best_value,
                        'second_best_value': second_value,
                        'improvement_percent': improvement
                    }
        
        return {
            'best_model': best_model,
            'best_model_name': best_model.get('model_name'),
            'business_score': best_model_info['business_score'],
            'selection_reason': 'Highest business score among qualified models',
            'model_ranking': model_ranking,
            'performance_advantages': performance_advantages,
            'qualified_models_count': len(qualified_models),
            'disqualified_models': disqualified_models,
            'business_metrics': business_metrics,
            'min_thresholds': min_thresholds
        }
    
    def generate_model_selection_report(self,
                                      model_results: List[Dict[str, Any]],
                                      cv_results: Optional[List[Dict[str, Any]]] = None,
                                      business_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive model selection report.
        
        Args:
            model_results: List of evaluation results from evaluate_model()
            cv_results: Optional cross-validation results for significance testing
            business_metrics: Business metric weights for selection
            
        Returns:
            Comprehensive model selection report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': len(model_results)
        }
        
        # Create comparison table
        comparison_table = self.create_comparison_table(model_results)
        report['comparison_table'] = comparison_table.to_dict('records')
        
        # Select best model
        selection_result = self.select_best_model(model_results, business_metrics)
        report['model_selection'] = selection_result
        
        # Add statistical significance testing if CV results provided
        if cv_results and len(cv_results) >= 2:
            try:
                significance_results = self.compare_models_with_significance(cv_results)
                report['statistical_significance'] = significance_results
            except Exception as e:
                report['statistical_significance'] = {
                    'error': f"Could not perform significance testing: {str(e)}"
                }
        
        # Add summary insights
        insights = []
        
        if selection_result['best_model']:
            best_name = selection_result['best_model_name']
            best_score = selection_result['business_score']
            insights.append(f"Best model: {best_name} with business score of {best_score:.3f}")
            
            # Check for performance advantages
            if selection_result['performance_advantages']:
                for metric, advantage in selection_result['performance_advantages'].items():
                    if advantage['improvement_percent'] > 5:  # Significant improvement
                        insights.append(
                            f"{best_name} shows {advantage['improvement_percent']:.1f}% "
                            f"improvement in {metric} over second-best model"
                        )
        
        # Check for disqualified models
        if selection_result['disqualified_models']:
            disqualified_count = len(selection_result['disqualified_models'])
            insights.append(f"{disqualified_count} models failed to meet minimum thresholds")
        
        # Performance distribution insights
        if len(model_results) > 1:
            f1_scores = [r.get('f1_score', 0) for r in model_results if r.get('f1_score') is not None]
            if f1_scores:
                f1_range = max(f1_scores) - min(f1_scores)
                if f1_range > 0.1:
                    insights.append(f"Large performance variation: F1-score range of {f1_range:.3f}")
                else:
                    insights.append("Models show similar performance levels")
        
        report['insights'] = insights
        
        return report