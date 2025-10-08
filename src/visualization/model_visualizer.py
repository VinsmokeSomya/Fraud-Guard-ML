"""
ModelVisualizer module for fraud detection model performance visualization.

This module provides the ModelVisualizer class for creating comprehensive visualizations
of model performance including ROC curves, precision-recall curves, confusion matrices,
feature importance plots, and SHAP analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    roc_auc_score, average_precision_score
)

# Import SHAP with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. SHAP plots will be disabled.")

try:
    from ..models.base_model import FraudModelInterface
except ImportError:
    # Handle case when running tests or standalone
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.base_model import FraudModelInterface

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    ModelVisualizer class for creating model performance visualizations.
    
    This class provides comprehensive visualization capabilities for fraud detection
    model performance, including ROC curves, precision-recall curves, confusion matrices,
    feature importance plots, and SHAP analysis.
    """
    
    def __init__(self, 
                 style: str = 'whitegrid',
                 palette: str = 'Set2',
                 figure_size: Tuple[int, int] = (10, 8),
                 dpi: int = 100):
        """
        Initialize ModelVisualizer with styling options.
        
        Args:
            style: Seaborn style for matplotlib plots
            palette: Color palette for visualizations
            figure_size: Default figure size for matplotlib plots
            dpi: DPI for matplotlib figures
        """
        self.style = style
        self.palette = palette
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Set up plotting style
        sns.set_style(self.style)
        sns.set_palette(self.palette)
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['figure.dpi'] = self.dpi
        
        logger.info(f"ModelVisualizer initialized with style: {style}, palette: {palette}")
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_pred_proba: np.ndarray,
                      model_name: str = "Model",
                      interactive: bool = False,
                      ax: Optional[plt.Axes] = None) -> Dict[str, Any]:
        """
        Create ROC curve visualization.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model for labeling
            interactive: Whether to create interactive plotly chart
            ax: Matplotlib axes object (for static plots)
            
        Returns:
            Dictionary containing ROC curve data and metrics
        """
        logger.info(f"Creating ROC curve for {model_name}")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        if interactive:
            return self._plot_roc_curve_interactive(fpr, tpr, auc_score, model_name)
        else:
            return self._plot_roc_curve_static(fpr, tpr, auc_score, model_name, ax)
    
    def plot_precision_recall_curve(self, 
                                   y_true: np.ndarray, 
                                   y_pred_proba: np.ndarray,
                                   model_name: str = "Model",
                                   interactive: bool = False,
                                   ax: Optional[plt.Axes] = None) -> Dict[str, Any]:
        """
        Create Precision-Recall curve visualization.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model for labeling
            interactive: Whether to create interactive plotly chart
            ax: Matplotlib axes object (for static plots)
            
        Returns:
            Dictionary containing PR curve data and metrics
        """
        logger.info(f"Creating Precision-Recall curve for {model_name}")
        
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        if interactive:
            return self._plot_pr_curve_interactive(precision, recall, avg_precision, model_name)
        else:
            return self._plot_pr_curve_static(precision, recall, avg_precision, model_name, ax)
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             model_name: str = "Model",
                             normalize: Optional[str] = None,
                             interactive: bool = False,
                             ax: Optional[plt.Axes] = None) -> Dict[str, Any]:
        """
        Create confusion matrix heatmap with annotations.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            model_name: Name of the model for labeling
            normalize: Normalization method ('true', 'pred', 'all', or None)
            interactive: Whether to create interactive plotly chart
            ax: Matplotlib axes object (for static plots)
            
        Returns:
            Dictionary containing confusion matrix data
        """
        logger.info(f"Creating confusion matrix for {model_name}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        
        if interactive:
            return self._plot_confusion_matrix_interactive(cm, model_name, normalize)
        else:
            return self._plot_confusion_matrix_static(cm, model_name, normalize, ax)
    
    def plot_feature_importance(self, 
                               feature_importance: Dict[str, float],
                               model_name: str = "Model",
                               top_n: int = 20,
                               interactive: bool = False,
                               ax: Optional[plt.Axes] = None) -> Dict[str, Any]:
        """
        Create feature importance bar chart.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            model_name: Name of the model for labeling
            top_n: Number of top features to display
            interactive: Whether to create interactive plotly chart
            ax: Matplotlib axes object (for static plots)
            
        Returns:
            Dictionary containing feature importance data
        """
        logger.info(f"Creating feature importance plot for {model_name}")
        
        if not feature_importance:
            logger.warning("No feature importance data available")
            return {'error': 'No feature importance data available'}
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        if interactive:
            return self._plot_feature_importance_interactive(features, importances, model_name)
        else:
            return self._plot_feature_importance_static(features, importances, model_name, ax)
    
    def plot_shap_summary(self, 
                         model: FraudModelInterface,
                         X_sample: pd.DataFrame,
                         model_name: str = "Model",
                         plot_type: str = "summary",
                         max_display: int = 20) -> Dict[str, Any]:
        """
        Create SHAP summary plots for model interpretability.
        
        Args:
            model: Trained fraud detection model
            X_sample: Sample data for SHAP analysis
            model_name: Name of the model for labeling
            plot_type: Type of SHAP plot ('summary', 'waterfall', 'force')
            max_display: Maximum number of features to display
            
        Returns:
            Dictionary containing SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Cannot create SHAP plots.")
            return {'error': 'SHAP not available'}
        
        logger.info(f"Creating SHAP {plot_type} plot for {model_name}")
        
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model.predict_proba, X_sample)
            shap_values = explainer(X_sample)
            
            # Create the appropriate SHAP plot
            if plot_type == "summary":
                return self._plot_shap_summary(shap_values, model_name, max_display)
            elif plot_type == "waterfall":
                return self._plot_shap_waterfall(shap_values, model_name, max_display)
            elif plot_type == "force":
                return self._plot_shap_force(shap_values, model_name)
            else:
                logger.warning(f"Unknown SHAP plot type: {plot_type}")
                return {'error': f'Unknown SHAP plot type: {plot_type}'}
                
        except Exception as e:
            logger.error(f"Error creating SHAP plot: {e}")
            return {'error': f'Error creating SHAP plot: {str(e)}'}
    
    def create_model_performance_dashboard(self,
                                         y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         y_pred_proba: np.ndarray,
                                         feature_importance: Optional[Dict[str, float]] = None,
                                         model_name: str = "Model",
                                         interactive: bool = False) -> None:
        """
        Create a comprehensive model performance dashboard.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities for positive class
            feature_importance: Dictionary of feature importance scores
            model_name: Name of the model for labeling
            interactive: Whether to create interactive plotly charts
        """
        logger.info(f"Creating comprehensive performance dashboard for {model_name}")
        
        if interactive:
            self._create_interactive_dashboard(y_true, y_pred, y_pred_proba, 
                                             feature_importance, model_name)
        else:
            self._create_static_dashboard(y_true, y_pred, y_pred_proba, 
                                        feature_importance, model_name)
    
    # Static plotting methods (matplotlib/seaborn)
    def _plot_roc_curve_static(self, fpr: np.ndarray, tpr: np.ndarray, 
                              auc_score: float, model_name: str, 
                              ax: Optional[plt.Axes] = None) -> Dict[str, Any]:
        """Create static ROC curve plot."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc_score': auc_score,
            'model_name': model_name
        }
    
    def _plot_pr_curve_static(self, precision: np.ndarray, recall: np.ndarray,
                             avg_precision: float, model_name: str,
                             ax: Optional[plt.Axes] = None) -> Dict[str, Any]:
        """Create static Precision-Recall curve plot."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot PR curve
        ax.plot(recall, precision, linewidth=2, 
               label=f'{model_name} (AP = {avg_precision:.3f})')
        
        # Add baseline (random classifier)
        baseline = np.sum(precision > 0) / len(precision)  # Approximate positive class ratio
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                  label=f'Random Classifier (AP = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'avg_precision': avg_precision,
            'model_name': model_name
        }
    
    def _plot_confusion_matrix_static(self, cm: np.ndarray, model_name: str,
                                     normalize: Optional[str] = None,
                                     ax: Optional[plt.Axes] = None) -> Dict[str, Any]:
        """Create static confusion matrix heatmap."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', ax=ax, cbar_kws={'shrink': 0.8})
        
        # Set labels
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xticklabels(['Legitimate', 'Fraud'])
        ax.set_yticklabels(['Legitimate', 'Fraud'])
        
        plt.tight_layout()
        plt.show()
        
        return {
            'confusion_matrix': cm.tolist(),
            'model_name': model_name,
            'normalize': normalize
        }
    
    def _plot_feature_importance_static(self, features: Tuple, importances: Tuple,
                                       model_name: str, ax: Optional[plt.Axes] = None) -> Dict[str, Any]:
        """Create static feature importance bar chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance - {model_name}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'features': list(features),
            'importances': list(importances),
            'model_name': model_name
        }    
   
 # Interactive plotting methods (plotly)
    def _plot_roc_curve_interactive(self, fpr: np.ndarray, tpr: np.ndarray,
                                   auc_score: float, model_name: str) -> Dict[str, Any]:
        """Create interactive ROC curve plot."""
        fig = go.Figure()
        
        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.3f})',
            line=dict(width=3)
        ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600,
            hovermode='x unified'
        )
        
        fig.show()
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc_score': auc_score,
            'model_name': model_name
        }
    
    def _plot_pr_curve_interactive(self, precision: np.ndarray, recall: np.ndarray,
                                  avg_precision: float, model_name: str) -> Dict[str, Any]:
        """Create interactive Precision-Recall curve plot."""
        fig = go.Figure()
        
        # Add PR curve
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{model_name} (AP = {avg_precision:.3f})',
            line=dict(width=3)
        ))
        
        # Add baseline
        baseline = np.sum(precision > 0) / len(precision)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            name=f'Random Classifier (AP = {baseline:.3f})',
            line=dict(dash='dash', color='gray')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Precision-Recall Curve - {model_name}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=800, height=600,
            hovermode='x unified'
        )
        
        fig.show()
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'avg_precision': avg_precision,
            'model_name': model_name
        }
    
    def _plot_confusion_matrix_interactive(self, cm: np.ndarray, model_name: str,
                                          normalize: Optional[str] = None) -> Dict[str, Any]:
        """Create interactive confusion matrix heatmap."""
        labels = ['Legitimate', 'Fraud']
        
        # Create annotations
        annotations = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=str(cm[i, j]) if normalize is None else f'{cm[i, j]:.2f}',
                        showarrow=False,
                        font=dict(color="white" if cm[i, j] > cm.max()/2 else "black", size=16)
                    )
                )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            annotations=annotations,
            width=600, height=500
        )
        
        fig.show()
        
        return {
            'confusion_matrix': cm.tolist(),
            'model_name': model_name,
            'normalize': normalize
        }
    
    def _plot_feature_importance_interactive(self, features: Tuple, importances: Tuple,
                                            model_name: str) -> Dict[str, Any]:
        """Create interactive feature importance bar chart."""
        fig = go.Figure(data=go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker=dict(color='lightblue', opacity=0.7)
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Feature Importance - {model_name}',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            width=900, height=max(400, len(features) * 25),
            yaxis=dict(autorange='reversed')  # Top feature at the top
        )
        
        fig.show()
        
        return {
            'features': list(features),
            'importances': list(importances),
            'model_name': model_name
        }
    
    # SHAP plotting methods
    def _plot_shap_summary(self, shap_values, model_name: str, max_display: int) -> Dict[str, Any]:
        """Create SHAP summary plot."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create SHAP summary plot
            if hasattr(shap_values, 'values') and shap_values.values.ndim == 3:
                # Multi-class case - use positive class (index 1)
                shap.summary_plot(shap_values.values[:, :, 1], shap_values.data, 
                                max_display=max_display, show=False)
            else:
                # Binary case
                shap.summary_plot(shap_values, max_display=max_display, show=False)
            
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()
            
            return {
                'plot_type': 'shap_summary',
                'model_name': model_name,
                'max_display': max_display
            }
            
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            return {'error': f'Error creating SHAP summary plot: {str(e)}'}
    
    def _plot_shap_waterfall(self, shap_values, model_name: str, max_display: int) -> Dict[str, Any]:
        """Create SHAP waterfall plot for first instance."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create waterfall plot for first instance
            if hasattr(shap_values, 'values') and shap_values.values.ndim == 3:
                # Multi-class case - use positive class (index 1)
                shap.waterfall_plot(shap_values[0, :, 1], max_display=max_display, show=False)
            else:
                # Binary case
                shap.waterfall_plot(shap_values[0], max_display=max_display, show=False)
            
            plt.title(f'SHAP Waterfall Plot - {model_name} (First Instance)')
            plt.tight_layout()
            plt.show()
            
            return {
                'plot_type': 'shap_waterfall',
                'model_name': model_name,
                'max_display': max_display
            }
            
        except Exception as e:
            logger.error(f"Error creating SHAP waterfall plot: {e}")
            return {'error': f'Error creating SHAP waterfall plot: {str(e)}'}
    
    def _plot_shap_force(self, shap_values, model_name: str) -> Dict[str, Any]:
        """Create SHAP force plot for first instance."""
        try:
            # Force plots are typically displayed in Jupyter notebooks
            # For regular Python scripts, we'll create a simplified version
            logger.info(f"SHAP force plot for {model_name} - best viewed in Jupyter notebook")
            
            if hasattr(shap_values, 'values') and shap_values.values.ndim == 3:
                # Multi-class case - use positive class (index 1)
                force_plot = shap.force_plot(shap_values.base_values[0, 1], 
                                           shap_values.values[0, :, 1], 
                                           shap_values.data[0])
            else:
                # Binary case
                force_plot = shap.force_plot(shap_values.base_values[0], 
                                           shap_values.values[0], 
                                           shap_values.data[0])
            
            return {
                'plot_type': 'shap_force',
                'model_name': model_name,
                'force_plot': force_plot
            }
            
        except Exception as e:
            logger.error(f"Error creating SHAP force plot: {e}")
            return {'error': f'Error creating SHAP force plot: {str(e)}'}
    
    # Dashboard creation methods
    def _create_static_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: np.ndarray, feature_importance: Optional[Dict[str, float]],
                                model_name: str) -> None:
        """Create static matplotlib dashboard."""
        # Determine subplot layout
        n_plots = 4 if feature_importance else 3
        fig = plt.figure(figsize=(16, 12))
        
        # ROC Curve
        ax1 = plt.subplot(2, 2, 1)
        self.plot_roc_curve(y_true, y_pred_proba, model_name, interactive=False, ax=ax1)
        
        # Precision-Recall Curve
        ax2 = plt.subplot(2, 2, 2)
        self.plot_precision_recall_curve(y_true, y_pred_proba, model_name, interactive=False, ax=ax2)
        
        # Confusion Matrix
        ax3 = plt.subplot(2, 2, 3)
        self.plot_confusion_matrix(y_true, y_pred, model_name, interactive=False, ax=ax3)
        
        # Feature Importance (if available)
        if feature_importance:
            ax4 = plt.subplot(2, 2, 4)
            self.plot_feature_importance(feature_importance, model_name, top_n=10, 
                                       interactive=False, ax=ax4)
        
        plt.suptitle(f'Model Performance Dashboard - {model_name}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
    
    def _create_interactive_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: np.ndarray, feature_importance: Optional[Dict[str, float]],
                                    model_name: str) -> None:
        """Create interactive plotly dashboard."""
        # Create individual plots
        print(f"\n=== Interactive Performance Dashboard for {model_name} ===\n")
        
        print("1. ROC Curve:")
        self.plot_roc_curve(y_true, y_pred_proba, model_name, interactive=True)
        
        print("\n2. Precision-Recall Curve:")
        self.plot_precision_recall_curve(y_true, y_pred_proba, model_name, interactive=True)
        
        print("\n3. Confusion Matrix:")
        self.plot_confusion_matrix(y_true, y_pred, model_name, interactive=True)
        
        if feature_importance:
            print("\n4. Feature Importance:")
            self.plot_feature_importance(feature_importance, model_name, top_n=15, interactive=True)
    
    def compare_models_performance(self,
                                 model_results: List[Dict[str, Any]],
                                 interactive: bool = False) -> None:
        """
        Create comparative visualizations for multiple models.
        
        Args:
            model_results: List of dictionaries containing model evaluation results
            interactive: Whether to create interactive plotly charts
        """
        logger.info(f"Creating model comparison visualizations for {len(model_results)} models")
        
        if len(model_results) < 2:
            logger.warning("Need at least 2 models for comparison")
            return
        
        if interactive:
            self._create_interactive_comparison(model_results)
        else:
            self._create_static_comparison(model_results)
    
    def _create_static_comparison(self, model_results: List[Dict[str, Any]]) -> None:
        """Create static comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ROC Curves comparison
        ax1 = axes[0, 0]
        for result in model_results:
            if 'roc_curve' in result:
                roc_data = result['roc_curve']
                ax1.plot(roc_data['fpr'], roc_data['tpr'], 
                        linewidth=2, label=f"{result['model_name']} (AUC = {result['auc_roc']:.3f})")
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR Curves comparison
        ax2 = axes[0, 1]
        for result in model_results:
            if 'precision_recall_curve' in result:
                pr_data = result['precision_recall_curve']
                ax2.plot(pr_data['recall'], pr_data['precision'], 
                        linewidth=2, label=f"{result['model_name']} (AP = {result.get('avg_precision', 'N/A')})")
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Metrics comparison
        ax3 = axes[1, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = [result['model_name'] for result in model_results]
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [result.get(metric, 0) for result in model_results]
            ax3.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x + width * 2)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Feature importance comparison (if available)
        ax4 = axes[1, 1]
        feature_importance_available = any('feature_importance' in result and result['feature_importance'] 
                                         for result in model_results)
        
        if feature_importance_available:
            # Get common top features across models
            all_features = set()
            for result in model_results:
                if 'feature_importance' in result and result['feature_importance']:
                    all_features.update(result['feature_importance'].keys())
            
            common_features = list(all_features)[:10]  # Top 10 common features
            
            x = np.arange(len(common_features))
            width = 0.8 / len(model_results)
            
            for i, result in enumerate(model_results):
                if 'feature_importance' in result and result['feature_importance']:
                    importances = [result['feature_importance'].get(feat, 0) for feat in common_features]
                    ax4.bar(x + i * width, importances, width, 
                           label=result['model_name'], alpha=0.8)
            
            ax4.set_xlabel('Features')
            ax4.set_ylabel('Importance')
            ax4.set_title('Feature Importance Comparison')
            ax4.set_xticks(x + width * (len(model_results) - 1) / 2)
            ax4.set_xticklabels(common_features, rotation=45)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Feature Importance Comparison')
        
        plt.suptitle('Model Performance Comparison Dashboard', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
    
    def _create_interactive_comparison(self, model_results: List[Dict[str, Any]]) -> None:
        """Create interactive comparison plots."""
        print("\n=== Interactive Model Comparison Dashboard ===\n")
        
        # ROC Curves comparison
        print("1. ROC Curves Comparison:")
        fig_roc = go.Figure()
        
        for result in model_results:
            if 'roc_curve' in result:
                roc_data = result['roc_curve']
                fig_roc.add_trace(go.Scatter(
                    x=roc_data['fpr'], y=roc_data['tpr'],
                    mode='lines', name=f"{result['model_name']} (AUC = {result['auc_roc']:.3f})",
                    line=dict(width=3)
                ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600
        )
        fig_roc.show()
        
        # Metrics comparison
        print("\n2. Performance Metrics Comparison:")
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = [result['model_name'] for result in model_results]
        
        fig_metrics = go.Figure()
        
        for metric in metrics:
            values = [result.get(metric, 0) for result in model_results]
            fig_metrics.add_trace(go.Bar(
                x=model_names, y=values, name=metric, opacity=0.8
            ))
        
        fig_metrics.update_layout(
            title='Performance Metrics Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            width=900, height=600
        )
        fig_metrics.show()
    
    def save_plots_to_file(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          y_pred_proba: np.ndarray,
                          feature_importance: Optional[Dict[str, float]],
                          model_name: str,
                          save_path: str) -> None:
        """
        Save all performance plots to files.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities for positive class
            feature_importance: Dictionary of feature importance scores
            model_name: Name of the model for labeling
            save_path: Directory path to save plots
        """
        import os
        from pathlib import Path
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving performance plots for {model_name} to {save_path}")
        
        # Save individual plots
        plots_to_save = [
            ('roc_curve', lambda: self.plot_roc_curve(y_true, y_pred_proba, model_name)),
            ('pr_curve', lambda: self.plot_precision_recall_curve(y_true, y_pred_proba, model_name)),
            ('confusion_matrix', lambda: self.plot_confusion_matrix(y_true, y_pred, model_name)),
        ]
        
        if feature_importance:
            plots_to_save.append(
                ('feature_importance', lambda: self.plot_feature_importance(feature_importance, model_name))
            )
        
        for plot_name, plot_func in plots_to_save:
            plt.figure(figsize=self.figure_size)
            plot_func()
            plt.savefig(save_dir / f"{model_name}_{plot_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save dashboard
        self._create_static_dashboard(y_true, y_pred, y_pred_proba, feature_importance, model_name)
        plt.savefig(save_dir / f"{model_name}_dashboard.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"All plots saved to {save_path}")