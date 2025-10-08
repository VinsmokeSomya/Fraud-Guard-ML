"""
Unit tests for ModelVisualizer class.

This module contains tests for the ModelVisualizer class functionality
including ROC curves, precision-recall curves, confusion matrices,
and feature importance visualizations.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from visualization.model_visualizer import ModelVisualizer
from models.base_model import FraudModelInterface


class TestModelVisualizer(unittest.TestCase):
    """Test cases for ModelVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = ModelVisualizer()
        
        # Create sample data
        np.random.seed(42)
        self.n_samples = 1000
        
        # Generate binary labels with some fraud cases
        self.y_true = np.random.binomial(1, 0.1, self.n_samples)  # 10% fraud rate
        
        # Generate predicted probabilities (higher for fraud cases)
        self.y_pred_proba = np.where(
            self.y_true == 1,
            np.random.beta(7, 3, np.sum(self.y_true == 1)),  # Higher proba for fraud
            np.random.beta(2, 8, np.sum(self.y_true == 0))   # Lower proba for legitimate
        )
        
        # Generate binary predictions based on threshold
        self.y_pred = (self.y_pred_proba > 0.5).astype(int)
        
        # Sample feature importance
        self.feature_importance = {
            'amount': 0.25,
            'balance_change': 0.20,
            'transaction_type': 0.15,
            'time_of_day': 0.12,
            'account_age': 0.10,
            'previous_transactions': 0.08,
            'location': 0.06,
            'device_type': 0.04
        }
    
    def test_initialization(self):
        """Test ModelVisualizer initialization."""
        visualizer = ModelVisualizer(
            style='darkgrid',
            palette='viridis',
            figure_size=(12, 10),
            dpi=150
        )
        
        self.assertEqual(visualizer.style, 'darkgrid')
        self.assertEqual(visualizer.palette, 'viridis')
        self.assertEqual(visualizer.figure_size, (12, 10))
        self.assertEqual(visualizer.dpi, 150)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_roc_curve_static(self, mock_show):
        """Test ROC curve plotting (static)."""
        result = self.visualizer.plot_roc_curve(
            self.y_true, self.y_pred_proba, "Test Model", interactive=False
        )
        
        # Check return values
        self.assertIn('fpr', result)
        self.assertIn('tpr', result)
        self.assertIn('auc_score', result)
        self.assertIn('model_name', result)
        self.assertEqual(result['model_name'], "Test Model")
        
        # Check that AUC is reasonable
        self.assertGreater(result['auc_score'], 0.5)  # Should be better than random
        self.assertLessEqual(result['auc_score'], 1.0)
        
        # Verify plot was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_precision_recall_curve_static(self, mock_show):
        """Test Precision-Recall curve plotting (static)."""
        result = self.visualizer.plot_precision_recall_curve(
            self.y_true, self.y_pred_proba, "Test Model", interactive=False
        )
        
        # Check return values
        self.assertIn('precision', result)
        self.assertIn('recall', result)
        self.assertIn('avg_precision', result)
        self.assertIn('model_name', result)
        self.assertEqual(result['model_name'], "Test Model")
        
        # Check that average precision is reasonable
        self.assertGreater(result['avg_precision'], 0.0)
        self.assertLessEqual(result['avg_precision'], 1.0)
        
        # Verify plot was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix_static(self, mock_show):
        """Test confusion matrix plotting (static)."""
        result = self.visualizer.plot_confusion_matrix(
            self.y_true, self.y_pred, "Test Model", interactive=False
        )
        
        # Check return values
        self.assertIn('confusion_matrix', result)
        self.assertIn('model_name', result)
        self.assertEqual(result['model_name'], "Test Model")
        
        # Check confusion matrix shape
        cm = np.array(result['confusion_matrix'])
        self.assertEqual(cm.shape, (2, 2))
        
        # Check that matrix values sum to total samples
        self.assertEqual(cm.sum(), len(self.y_true))
        
        # Verify plot was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_feature_importance_static(self, mock_show):
        """Test feature importance plotting (static)."""
        result = self.visualizer.plot_feature_importance(
            self.feature_importance, "Test Model", top_n=5, interactive=False
        )
        
        # Check return values
        self.assertIn('features', result)
        self.assertIn('importances', result)
        self.assertIn('model_name', result)
        self.assertEqual(result['model_name'], "Test Model")
        
        # Check that top 5 features are returned
        self.assertEqual(len(result['features']), 5)
        self.assertEqual(len(result['importances']), 5)
        
        # Check that features are sorted by importance (descending)
        importances = result['importances']
        self.assertTrue(all(importances[i] >= importances[i+1] for i in range(len(importances)-1)))
        
        # Verify plot was called
        mock_show.assert_called_once()
    
    def test_plot_feature_importance_empty(self):
        """Test feature importance plotting with empty data."""
        result = self.visualizer.plot_feature_importance({}, "Test Model")
        
        # Should return error
        self.assertIn('error', result)
    
    @patch('matplotlib.pyplot.show')
    def test_create_model_performance_dashboard_static(self, mock_show):
        """Test comprehensive dashboard creation (static)."""
        self.visualizer.create_model_performance_dashboard(
            self.y_true, self.y_pred, self.y_pred_proba, 
            self.feature_importance, "Test Model", interactive=False
        )
        
        # Verify plot was called (dashboard creates one combined plot)
        mock_show.assert_called_once()
    
    def test_compare_models_performance_insufficient_models(self):
        """Test model comparison with insufficient models."""
        # Create mock model result
        model_result = {
            'model_name': 'Test Model',
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'auc_roc': 0.82
        }
        
        # Test with only one model (should log warning)
        with patch('src.visualization.model_visualizer.logger') as mock_logger:
            self.visualizer.compare_models_performance([model_result])
            mock_logger.warning.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_compare_models_performance_static(self, mock_show):
        """Test model comparison visualization (static)."""
        # Create mock model results
        model_results = [
            {
                'model_name': 'Model A',
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.75,
                'f1_score': 0.77,
                'auc_roc': 0.82,
                'roc_curve': {
                    'fpr': [0, 0.2, 0.4, 1],
                    'tpr': [0, 0.6, 0.8, 1]
                },
                'precision_recall_curve': {
                    'precision': [1, 0.8, 0.6, 0.1],
                    'recall': [0, 0.4, 0.8, 1]
                }
            },
            {
                'model_name': 'Model B',
                'accuracy': 0.88,
                'precision': 0.82,
                'recall': 0.78,
                'f1_score': 0.80,
                'auc_roc': 0.85,
                'roc_curve': {
                    'fpr': [0, 0.15, 0.35, 1],
                    'tpr': [0, 0.65, 0.85, 1]
                },
                'precision_recall_curve': {
                    'precision': [1, 0.85, 0.7, 0.15],
                    'recall': [0, 0.45, 0.85, 1]
                }
            }
        ]
        
        self.visualizer.compare_models_performance(model_results, interactive=False)
        
        # Verify plot was called
        mock_show.assert_called_once()
    
    def test_shap_not_available(self):
        """Test SHAP plotting when SHAP is not available."""
        # Mock a model
        mock_model = Mock(spec=FraudModelInterface)
        mock_model.predict_proba.return_value = np.random.random((100, 2))
        
        # Create sample data
        X_sample = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100)
        })
        
        # Test with SHAP not available
        with patch('src.visualization.model_visualizer.SHAP_AVAILABLE', False):
            result = self.visualizer.plot_shap_summary(mock_model, X_sample, "Test Model")
            
            # Should return error
            self.assertIn('error', result)
            self.assertIn('SHAP not available', result['error'])
    
    @patch('src.visualization.model_visualizer.SHAP_AVAILABLE', True)
    @patch('src.visualization.model_visualizer.shap')
    def test_shap_summary_plot(self, mock_shap):
        """Test SHAP summary plot creation."""
        # Mock SHAP components
        mock_explainer = Mock()
        mock_shap_values = Mock()
        mock_shap_values.values = np.random.random((100, 5, 2))  # Multi-class format
        mock_shap_values.data = np.random.random((100, 5))
        
        mock_shap.Explainer.return_value = mock_explainer
        mock_explainer.return_value = mock_shap_values
        
        # Mock model
        mock_model = Mock(spec=FraudModelInterface)
        mock_model.predict_proba.return_value = np.random.random((100, 2))
        
        # Create sample data
        X_sample = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'feature3': np.random.random(100),
            'feature4': np.random.random(100),
            'feature5': np.random.random(100)
        })
        
        with patch('matplotlib.pyplot.show'):
            result = self.visualizer.plot_shap_summary(mock_model, X_sample, "Test Model")
        
        # Check that SHAP explainer was called
        mock_shap.Explainer.assert_called_once()
        
        # Should return success result
        self.assertNotIn('error', result)
        self.assertEqual(result['plot_type'], 'shap_summary')
        self.assertEqual(result['model_name'], 'Test Model')
    
    def test_save_plots_to_file(self):
        """Test saving plots to files."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig') as mock_savefig:
                self.visualizer.save_plots_to_file(
                    self.y_true, self.y_pred, self.y_pred_proba,
                    self.feature_importance, "Test Model", temp_dir
                )
                
                # Check that savefig was called multiple times (for different plots)
                self.assertGreater(mock_savefig.call_count, 3)


class TestModelVisualizerIntegration(unittest.TestCase):
    """Integration tests for ModelVisualizer with real models."""
    
    def setUp(self):
        """Set up test fixtures with real model."""
        self.visualizer = ModelVisualizer()
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        
        # Generate features
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        
        # Generate labels with some correlation to features
        y = ((X['feature1'] + X['feature2'] * 0.5 + np.random.normal(0, 0.5, n_samples)) > 0).astype(int)
        
        self.X = X
        self.y = pd.Series(y)
    
    @patch('matplotlib.pyplot.show')
    def test_with_logistic_regression(self, mock_show):
        """Test ModelVisualizer with actual LogisticRegression model."""
        from models.logistic_regression_model import LogisticRegressionModel
        
        # Train model
        model = LogisticRegressionModel(random_state=42)
        model.train(self.X, self.y)
        
        # Get predictions
        y_pred = model.predict(self.X)
        y_pred_proba = model.predict_proba(self.X)
        
        # Handle probability format
        if y_pred_proba.ndim == 2:
            y_pred_proba_positive = y_pred_proba[:, 1]
        else:
            y_pred_proba_positive = y_pred_proba
        
        # Test ROC curve
        result = self.visualizer.plot_roc_curve(
            self.y, y_pred_proba_positive, "Logistic Regression", interactive=False
        )
        
        self.assertIn('auc_score', result)
        self.assertGreater(result['auc_score'], 0.5)  # Should be better than random
        
        # Test feature importance
        feature_importance = model.get_feature_importance()
        result = self.visualizer.plot_feature_importance(
            feature_importance, "Logistic Regression", interactive=False
        )
        
        self.assertIn('features', result)
        self.assertIn('importances', result)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)