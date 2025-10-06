"""
Tests for LogisticRegressionModel implementation.

This module contains unit tests for the LogisticRegressionModel class,
testing training, prediction, hyperparameter tuning, and class balancing.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.logistic_regression_model import LogisticRegressionModel


class TestLogisticRegressionModel:
    """Test suite for LogisticRegressionModel."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample fraud detection data for testing."""
        # Generate imbalanced binary classification data
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            weights=[0.9, 0.1],  # Imbalanced classes (10% fraud)
            random_state=42
        )
        
        # Convert to DataFrame with meaningful feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='is_fraud')
        
        return train_test_split(X_df, y_series, test_size=0.3, random_state=42, stratify=y)
    
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = LogisticRegressionModel()
        
        assert model.model_name == "LogisticRegression"
        assert model.random_state == 42
        assert model.max_iter == 1000
        assert model.class_weight == 'balanced'
        assert model.tune_hyperparameters is True
        assert model.cv_folds == 5
        assert model.scoring == 'f1'
        assert model.is_trained is False
    
    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = LogisticRegressionModel(
            random_state=123,
            max_iter=500,
            class_weight=None,
            tune_hyperparameters=False,
            cv_folds=3,
            scoring='accuracy'
        )
        
        assert model.random_state == 123
        assert model.max_iter == 500
        assert model.class_weight is None
        assert model.tune_hyperparameters is False
        assert model.cv_folds == 3
        assert model.scoring == 'accuracy'
    
    def test_training_without_tuning(self, sample_data):
        """Test model training without hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False)
        model.train(X_train, y_train)
        
        assert model.is_trained is True
        assert model.model is not None
        assert model.feature_names == list(X_train.columns)
        assert model.best_params_ is None  # No tuning performed
    
    def test_training_with_tuning(self, sample_data):
        """Test model training with hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Use smaller parameter grid for faster testing
        model = LogisticRegressionModel(
            tune_hyperparameters=True,
            cv_folds=3  # Reduce CV folds for faster testing
        )
        
        # Override the parameter grid method for faster testing
        def get_small_param_grid():
            return [{
                'C': [0.1, 1.0],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }]
        
        model._get_param_grid = get_small_param_grid
        model.train(X_train, y_train)
        
        assert model.is_trained is True
        assert model.model is not None
        assert model.best_params_ is not None
        assert model.best_score_ is not None
        assert 'hyperparameter_tuning' in model.metadata
    
    def test_predictions(self, sample_data):
        """Test model predictions."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False)
        model.train(X_train, y_train)
        
        # Test binary predictions
        predictions = model.predict(X_test)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})
        
        # Test probability predictions
        probabilities = model.predict_proba(X_test)
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        
        # Test fraud probability method
        fraud_probs = model.predict_fraud_probability(X_test)
        assert isinstance(fraud_probs, np.ndarray)
        assert len(fraud_probs) == len(X_test)
        assert np.allclose(fraud_probs, probabilities[:, 1])
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False)
        model.train(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(X_train.columns)
        assert all(isinstance(v, (int, float)) for v in importance.values())
        assert all(v >= 0 for v in importance.values())  # Absolute values
        
        # Check that features are sorted by importance
        importance_values = list(importance.values())
        assert importance_values == sorted(importance_values, reverse=True)
    
    def test_coefficients_and_intercept(self, sample_data):
        """Test coefficient and intercept extraction."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = LogisticRegressionModel(tune_hyperparameters=False)
        model.train(X_train, y_train)
        
        # Test coefficients
        coefficients = model.get_coefficients()
        assert isinstance(coefficients, dict)
        assert len(coefficients) == len(X_train.columns)
        assert all(isinstance(v, (int, float)) for v in coefficients.values())
        
        # Test intercept
        intercept = model.get_intercept()
        assert isinstance(intercept, float)
    
    def test_model_summary(self, sample_data):
        """Test model summary generation."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Test untrained model
        model = LogisticRegressionModel()
        summary = model.get_model_summary()
        assert summary["status"] == "not_trained"
        
        # Test trained model
        model.train(X_train, y_train)
        summary = model.get_model_summary()
        
        assert summary["model_type"] == "LogisticRegression"
        assert summary["is_trained"] is True
        assert summary["feature_count"] == len(X_train.columns)
        assert "intercept" in summary
        assert "top_features" in summary
        assert len(summary["top_features"]) <= 5
    
    def test_class_balancing(self, sample_data):
        """Test class balancing functionality."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Test with balanced class weights
        model_balanced = LogisticRegressionModel(
            class_weight='balanced',
            tune_hyperparameters=False
        )
        model_balanced.train(X_train, y_train)
        
        # Test with no class balancing
        model_unbalanced = LogisticRegressionModel(
            class_weight=None,
            tune_hyperparameters=False
        )
        model_unbalanced.train(X_train, y_train)
        
        # Both models should train successfully
        assert model_balanced.is_trained
        assert model_unbalanced.is_trained
        
        # Predictions should be different due to class balancing
        pred_balanced = model_balanced.predict_fraud_probability(X_test)
        pred_unbalanced = model_unbalanced.predict_fraud_probability(X_test)
        
        # They shouldn't be identical (though this is probabilistic)
        assert not np.allclose(pred_balanced, pred_unbalanced, rtol=0.01)
    
    def test_error_handling(self, sample_data):
        """Test error handling for invalid inputs."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = LogisticRegressionModel()
        
        # Test prediction before training
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X_test)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict_proba(X_test)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.get_feature_importance()
        
        # Test with mismatched training data
        with pytest.raises(ValueError, match="same length"):
            model.train(X_train, y_train[:-10])  # Different lengths
        
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="empty"):
            model.train(pd.DataFrame(), pd.Series(dtype=int))
    
    def test_model_persistence(self, sample_data, tmp_path):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train and save model
        model = LogisticRegressionModel(tune_hyperparameters=False)
        model.train(X_train, y_train)
        
        model_path = tmp_path / "test_model"
        model.save_model(str(model_path))
        
        # Load model and test
        new_model = LogisticRegressionModel()
        new_model.load_model(str(model_path))
        
        assert new_model.is_trained
        assert new_model.feature_names == model.feature_names
        
        # Test that predictions are the same
        original_pred = model.predict(X_test)
        loaded_pred = new_model.predict(X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)