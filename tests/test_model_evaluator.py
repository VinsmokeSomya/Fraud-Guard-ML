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


if __name__ == "__main__":
    pytest.main([__file__])