"""
Tests for the ModelEvaluator class.

This module contains unit tests for the model evaluation functionality,
including metrics calculation, cross-validation, and model comparison.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.model_evaluator import ModelEvaluator
from src.models.base_model import FraudModelInterface


class MockFraudModel(FraudModelInterface):
    """Mock fraud model for testing purposes."""
    
    def __init__(self, model_name="MockModel"):
        self.model_name = model_name
        self.is_trained = False
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Mock training method."""
        self.is_trained = True
        self.feature_names = list(X_train.columns)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Mock prediction method - returns random predictions."""
        np.random.seed(42)  # For reproducible tests
        return np.random.randint(0, 2, size=len(X))
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Mock probability prediction method."""
        np.random.seed(42)  # For reproducible tests
        proba_positive = np.random.random(len(X))
        proba_negative = 1 - proba_positive
        return np.column_stack([proba_negative, proba_positive])
        
    def get_feature_importance(self) -> dict:
        """Mock feature importance method."""
        if self.feature_names:
            return {name: np.random.random() for name in self.feature_names}
        return {}


@pytest.fixture
def sample_data():
    """Create sample fraud detection data for testing."""
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # Imbalanced like fraud data
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='is_fraud')
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42, stratify=y_series
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_mock_model(sample_data):
    """Create a trained mock model for testing."""
    X_train, _, y_train, _ = sample_data
    model = MockFraudModel("TestModel")
    model.train(X_train, y_train)
    return model


@pytest.fixture
def model_evaluator():
    """Create a ModelEvaluator instance for testing."""
    return ModelEvaluator(random_state=42)


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def test_initialization(self, model_evaluator):
        """Test ModelEvaluator initialization."""
        assert model_evaluator.random_state == 42
        assert model_evaluator.evaluation_history == []
    
    def test_evaluate_model_basic_metrics(self, model_evaluator, trained_mock_model, sample_data):
        """Test basic model evaluation metrics calculation."""
        _, X_test, _, y_test = sample_data
        
        results = model_evaluator.evaluate_model(trained_mock_model, X_test, y_test)
        
        # Check that all required metrics are present
        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
            'confusion_matrix', 'classification_report'
        ]
        
        for metric in required_metrics:
            assert metric in results
            
        # Check that metrics are reasonable values
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1_score'] <= 1
        
        if results['auc_roc'] is not None:
            assert 0 <= results['auc_roc'] <= 1
    
    def test_evaluate_model_confusion_matrix(self, model_evaluator, trained_mock_model, sample_data):
        """Test confusion matrix generation."""
        _, X_test, _, y_test = sample_data
        
        results = model_evaluator.evaluate_model(trained_mock_model, X_test, y_test)
        
        cm = results['confusion_matrix']
        assert isinstance(cm, list)
        assert len(cm) == 2  # Binary classification
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2
        
        # Check that confusion matrix values sum to total test samples
        total_predictions = sum(sum(row) for row in cm)
        assert total_predictions == len(X_test)
    
    def test_evaluate_model_metadata(self, model_evaluator, trained_mock_model, sample_data):
        """Test that evaluation includes proper metadata."""
        _, X_test, _, y_test = sample_data
        
        results = model_evaluator.evaluate_model(
            trained_mock_model, X_test, y_test, model_name="CustomName"
        )
        
        assert results['model_name'] == "CustomName"
        assert results['test_samples'] == len(X_test)
        assert results['positive_samples'] == int(y_test.sum())
        assert results['negative_samples'] == int(len(y_test) - y_test.sum())
    
    def test_evaluate_untrained_model_raises_error(self, model_evaluator, sample_data):
        """Test that evaluating an untrained model raises an error."""
        _, X_test, _, y_test = sample_data
        untrained_model = MockFraudModel()
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model_evaluator.evaluate_model(untrained_model, X_test, y_test)
    
    def test_cross_validate_model(self, model_evaluator, sample_data):
        """Test cross-validation functionality."""
        X_train, _, y_train, _ = sample_data
        model = MockFraudModel("CVTestModel")
        
        cv_results = model_evaluator.cross_validate_model(
            model, X_train, y_train, cv_folds=3
        )
        
        # Check structure of results
        assert 'cv_summary' in cv_results
        assert 'fold_details' in cv_results
        assert cv_results['cv_folds'] == 3
        assert len(cv_results['fold_details']) == 3
        
        # Check that summary contains expected metrics
        summary = cv_results['cv_summary']
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in expected_metrics:
            assert f'{metric}_mean' in summary
            assert f'{metric}_std' in summary
            assert f'{metric}_scores' in summary
    
    def test_compare_models(self, model_evaluator, sample_data):
        """Test model comparison functionality."""
        _, X_test, _, y_test = sample_data
        
        # Create and evaluate multiple models
        model1 = MockFraudModel("Model1")
        model1.train(sample_data[0], sample_data[2])
        results1 = model_evaluator.evaluate_model(model1, X_test, y_test)
        
        model2 = MockFraudModel("Model2")
        model2.train(sample_data[0], sample_data[2])
        results2 = model_evaluator.evaluate_model(model2, X_test, y_test)
        
        # Compare models
        comparison = model_evaluator.compare_models([results1, results2])
        
        assert 'comparison_table' in comparison
        assert 'best_model' in comparison
        assert 'primary_metric' in comparison
        assert comparison['model_count'] == 2
        
        # Check comparison table structure
        table = comparison['comparison_table']
        assert len(table) == 2
        assert all('model_name' in row for row in table)
        assert all('rank' in row for row in table)
    
    def test_evaluation_history(self, model_evaluator, trained_mock_model, sample_data):
        """Test evaluation history tracking."""
        _, X_test, _, y_test = sample_data
        
        # Perform multiple evaluations
        model_evaluator.evaluate_model(trained_mock_model, X_test, y_test, "Test1")
        model_evaluator.evaluate_model(trained_mock_model, X_test, y_test, "Test2")
        
        # Check history
        assert len(model_evaluator.evaluation_history) == 2
        
        summary = model_evaluator.get_evaluation_summary()
        assert summary['total_evaluations'] == 2
        assert summary['unique_models'] == 2
        
        # Clear history
        model_evaluator.clear_history()
        assert len(model_evaluator.evaluation_history) == 0
    
    def test_calculate_metrics_edge_cases(self, model_evaluator):
        """Test metrics calculation with edge cases."""
        # Test with perfect predictions
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])
        
        metrics = model_evaluator._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        
        # Test with all negative predictions
        y_pred_all_neg = np.array([0, 0, 0, 0])
        y_pred_proba_all_neg = np.array([0.1, 0.2, 0.3, 0.4])
        
        metrics_neg = model_evaluator._calculate_metrics(
            y_true, y_pred_all_neg, y_pred_proba_all_neg
        )
        
        assert metrics_neg['precision'] == 0.0  # No true positives
        assert metrics_neg['recall'] == 0.0     # No true positives
        assert metrics_neg['f1_score'] == 0.0   # No true positives


    def test_create_comparison_table(self, model_evaluator):
        """Test comparison table creation functionality."""
        # Create mock evaluation results
        model_results = [
            {
                'model_name': 'LogisticRegression',
                'accuracy': 0.85, 'precision': 0.75, 'recall': 0.80, 'f1_score': 0.77,
                'auc_roc': 0.88, 'test_samples': 1000, 'positive_samples': 200
            },
            {
                'model_name': 'RandomForest',
                'accuracy': 0.88, 'precision': 0.82, 'recall': 0.78, 'f1_score': 0.80,
                'auc_roc': 0.91, 'test_samples': 1000, 'positive_samples': 200
            }
        ]
        
        comparison_df = model_evaluator.create_comparison_table(model_results)
        
        assert len(comparison_df) == 2
        assert 'model_name' in comparison_df.columns
        assert 'f1_score_rank' in comparison_df.columns
        assert 'auc_roc_rank' in comparison_df.columns
        
        # Check ranking (RandomForest should rank higher in F1 and AUC)
        rf_row = comparison_df[comparison_df['model_name'] == 'RandomForest'].iloc[0]
        lr_row = comparison_df[comparison_df['model_name'] == 'LogisticRegression'].iloc[0]
        
        assert rf_row['f1_score_rank'] < lr_row['f1_score_rank']
        assert rf_row['auc_roc_rank'] < lr_row['auc_roc_rank']
    
    def test_statistical_significance_test(self, model_evaluator):
        """Test statistical significance testing functionality."""
        # Test with clearly different scores
        model1_scores = [0.70, 0.72, 0.71, 0.73, 0.69]
        model2_scores = [0.80, 0.82, 0.81, 0.83, 0.79]
        
        # Test paired t-test
        result = model_evaluator.statistical_significance_test(
            model1_scores, model2_scores, test_type='paired_ttest'
        )
        
        assert result['test_type'] == 'paired_ttest'
        assert result['p_value'] is not None
        assert result['significant'] is not None
        assert result['mean_diff'] is not None
        assert result['effect_size'] is not None
        
        # Should be significant difference
        assert result['significant'] == True
        assert result['mean_diff'] < 0  # Model 1 should be worse
        
        # Test Wilcoxon test
        wilcoxon_result = model_evaluator.statistical_significance_test(
            model1_scores, model2_scores, test_type='wilcoxon'
        )
        
        assert wilcoxon_result['test_type'] == 'wilcoxon'
        assert wilcoxon_result['p_value'] is not None
        
        # Test with identical scores (should not be significant)
        identical_scores = [0.75, 0.75, 0.75, 0.75, 0.75]
        identical_result = model_evaluator.statistical_significance_test(
            identical_scores, identical_scores, test_type='paired_ttest'
        )
        
        assert identical_result['p_value'] == 1.0 or np.isnan(identical_result['p_value'])
        assert identical_result['mean_diff'] == 0.0
    
    def test_statistical_significance_edge_cases(self, model_evaluator):
        """Test statistical significance testing with edge cases."""
        # Test with insufficient data
        short_scores1 = [0.7]
        short_scores2 = [0.8]
        
        result = model_evaluator.statistical_significance_test(
            short_scores1, short_scores2, test_type='paired_ttest'
        )
        
        assert 'error' in result
        assert result['p_value'] is None
        
        # Test with NaN values
        nan_scores1 = [0.7, np.nan, 0.8, 0.75, 0.72]
        nan_scores2 = [0.8, 0.82, np.nan, 0.85, 0.79]
        
        nan_result = model_evaluator.statistical_significance_test(
            nan_scores1, nan_scores2, test_type='paired_ttest'
        )
        
        # Should handle NaN values gracefully
        assert nan_result['p_value'] is not None or 'error' in nan_result
    
    def test_compare_models_with_significance(self, model_evaluator):
        """Test comprehensive model comparison with significance testing."""
        # Create mock CV results
        cv_results = [
            {
                'cv_summary': {
                    'f1_score_scores': [0.75, 0.78, 0.76, 0.79, 0.77]
                },
                'fold_details': [
                    {'metrics': {'model_name': 'LogisticRegression'}}
                ]
            },
            {
                'cv_summary': {
                    'f1_score_scores': [0.82, 0.84, 0.81, 0.85, 0.83]
                },
                'fold_details': [
                    {'metrics': {'model_name': 'RandomForest'}}
                ]
            }
        ]
        
        comparison_result = model_evaluator.compare_models_with_significance(
            cv_results, metric='f1_score'
        )
        
        assert 'pairwise_comparisons' in comparison_result
        assert 'comparison_matrix' in comparison_result
        assert 'summary_statistics' in comparison_result
        
        # Check pairwise comparisons
        pairwise = comparison_result['pairwise_comparisons']
        assert len(pairwise) == 2  # 2 models = 2 pairwise comparisons
        
        # Check summary statistics
        summary_stats = comparison_result['summary_statistics']
        assert len(summary_stats) == 2
        
        for stat in summary_stats:
            assert 'model_name' in stat
            assert 'f1_score_mean' in stat
            assert 'f1_score_std' in stat
    
    def test_select_best_model_default(self, model_evaluator):
        """Test best model selection with default parameters."""
        model_results = [
            {
                'model_name': 'LogisticRegression',
                'precision': 0.75, 'recall': 0.80, 'f1_score': 0.77, 'auc_roc': 0.88
            },
            {
                'model_name': 'RandomForest',
                'precision': 0.82, 'recall': 0.78, 'f1_score': 0.80, 'auc_roc': 0.91
            },
            {
                'model_name': 'XGBoost',
                'precision': 0.85, 'recall': 0.82, 'f1_score': 0.83, 'auc_roc': 0.93
            }
        ]
        
        selection_result = model_evaluator.select_best_model(model_results)
        
        assert selection_result['best_model'] is not None
        assert selection_result['best_model_name'] == 'XGBoost'  # Should be best overall
        assert 'business_score' in selection_result
        assert 'model_ranking' in selection_result
        assert len(selection_result['model_ranking']) == 3
        
        # Check ranking order
        ranking = selection_result['model_ranking']
        assert ranking[0]['model_name'] == 'XGBoost'
        assert ranking[0]['rank'] == 1
    
    def test_select_best_model_custom_metrics(self, model_evaluator):
        """Test best model selection with custom business metrics."""
        model_results = [
            {
                'model_name': 'HighPrecision',
                'precision': 0.95, 'recall': 0.60, 'f1_score': 0.74, 'auc_roc': 0.85
            },
            {
                'model_name': 'HighRecall',
                'precision': 0.70, 'recall': 0.90, 'f1_score': 0.79, 'auc_roc': 0.88
            }
        ]
        
        # Prioritize precision
        precision_focused = {
            'precision': 0.7,
            'recall': 0.2,
            'f1_score': 0.1
        }
        
        precision_result = model_evaluator.select_best_model(
            model_results, business_metrics=precision_focused
        )
        
        assert precision_result['best_model_name'] == 'HighPrecision'
        
        # Prioritize recall
        recall_focused = {
            'precision': 0.2,
            'recall': 0.7,
            'f1_score': 0.1
        }
        
        recall_result = model_evaluator.select_best_model(
            model_results, business_metrics=recall_focused
        )
        
        assert recall_result['best_model_name'] == 'HighRecall'
    
    def test_select_best_model_thresholds(self, model_evaluator):
        """Test best model selection with minimum thresholds."""
        model_results = [
            {
                'model_name': 'LowPerformance',
                'precision': 0.40, 'recall': 0.45, 'f1_score': 0.42, 'auc_roc': 0.65
            },
            {
                'model_name': 'GoodPerformance',
                'precision': 0.80, 'recall': 0.75, 'f1_score': 0.77, 'auc_roc': 0.88
            }
        ]
        
        # Set high thresholds
        high_thresholds = {
            'precision': 0.70,
            'recall': 0.70,
            'f1_score': 0.70
        }
        
        selection_result = model_evaluator.select_best_model(
            model_results, min_thresholds=high_thresholds
        )
        
        assert selection_result['best_model_name'] == 'GoodPerformance'
        assert selection_result['qualified_models_count'] == 1
        assert len(selection_result['disqualified_models']) == 1
        assert selection_result['disqualified_models'][0]['model_name'] == 'LowPerformance'
        
        # Test with impossible thresholds
        impossible_thresholds = {
            'precision': 0.99,
            'recall': 0.99,
            'f1_score': 0.99
        }
        
        impossible_result = model_evaluator.select_best_model(
            model_results, min_thresholds=impossible_thresholds
        )
        
        assert impossible_result['best_model'] is None
        assert 'No models met minimum thresholds' in impossible_result['selection_reason']
        assert len(impossible_result['disqualified_models']) == 2
    
    def test_generate_model_selection_report(self, model_evaluator):
        """Test comprehensive model selection report generation."""
        model_results = [
            {
                'model_name': 'Model1',
                'accuracy': 0.85, 'precision': 0.75, 'recall': 0.80, 'f1_score': 0.77,
                'auc_roc': 0.88, 'test_samples': 1000, 'positive_samples': 200
            },
            {
                'model_name': 'Model2',
                'accuracy': 0.88, 'precision': 0.82, 'recall': 0.78, 'f1_score': 0.80,
                'auc_roc': 0.91, 'test_samples': 1000, 'positive_samples': 200
            }
        ]
        
        cv_results = [
            {
                'cv_summary': {'f1_score_scores': [0.75, 0.78, 0.76, 0.79, 0.77]},
                'fold_details': [{'metrics': {'model_name': 'Model1'}}]
            },
            {
                'cv_summary': {'f1_score_scores': [0.79, 0.81, 0.78, 0.82, 0.80]},
                'fold_details': [{'metrics': {'model_name': 'Model2'}}]
            }
        ]
        
        report = model_evaluator.generate_model_selection_report(
            model_results, cv_results
        )
        
        # Check report structure
        assert 'timestamp' in report
        assert 'models_evaluated' in report
        assert 'comparison_table' in report
        assert 'model_selection' in report
        assert 'statistical_significance' in report
        assert 'insights' in report
        
        assert report['models_evaluated'] == 2
        assert len(report['comparison_table']) == 2
        assert isinstance(report['insights'], list)
        
        # Check that insights are generated
        assert len(report['insights']) > 0
        
        # Test report without CV results
        report_no_cv = model_evaluator.generate_model_selection_report(model_results)
        
        assert 'comparison_table' in report_no_cv
        assert 'model_selection' in report_no_cv
        # Should not have statistical significance without CV results
        assert 'statistical_significance' not in report_no_cv or 'error' in report_no_cv.get('statistical_significance', {})


if __name__ == "__main__":
    pytest.main([__file__])