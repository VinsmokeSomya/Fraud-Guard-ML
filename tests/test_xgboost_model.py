"""
Tests for XGBoostModel implementation.

This module contains unit tests for the XGBoostModel class,
testing training, prediction, feature importance, SHAP values, and GPU support.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from unittest.mock import patch, MagicMock

from src.models.xgboost_model import XGBoostModel


class TestXGBoostModel:
    """Test cases for XGBoostModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample fraud detection data for testing."""
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            weights=[0.9, 0.1],  # Imbalanced classes like fraud detection
            random_state=42
        )
        
        # Convert to DataFrame with feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='is_fraud')
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_model_initialization(self):
        """Test XGBoostModel initialization."""
        model = XGBoostModel(
            n_estimators=100,
            random_state=42,
            learning_rate=0.1,
            max_depth=6,
            use_gpu=False
        )
        
        assert model.model_name == "XGBoost"
        assert model.n_estimators == 100
        assert model.random_state == 42
        assert model.learning_rate == 0.1
        assert model.max_depth == 6
        assert model.use_gpu is False
        assert not model.is_trained
        assert model.early_stopping_rounds == 10  # Default value
    
    def test_scale_pos_weight_calculation(self, sample_data):
        """Test automatic scale_pos_weight calculation."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(random_state=42)
        scale_pos_weight = model._calculate_scale_pos_weight(y_train)
        
        # Should be ratio of negative to positive samples
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        expected_weight = neg_count / pos_count
        
        assert scale_pos_weight == expected_weight
        assert scale_pos_weight > 1  # Since we have imbalanced data
    
    def test_model_training_without_tuning(self, sample_data):
        """Test model training without hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        
        # Train the model
        model.train(X_train, y_train)
        
        assert model.is_trained
        assert model.model is not None
        assert model.feature_names == list(X_train.columns)
        assert model.best_params_ is None  # No tuning performed
    
    def test_predictions(self, sample_data):
        """Test model predictions."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        # Test binary predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probability predictions
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Test fraud probability
        fraud_probs = model.predict_fraud_probability(X_test)
        assert len(fraud_probs) == len(X_test)
        assert np.allclose(fraud_probs, probabilities[:, 1])
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        # Test default feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X_train.columns)
        # XGBoost returns numpy numeric types, not regular Python floats
        assert all(isinstance(score, np.number) for score in importance.values())
        assert all(score >= 0 for score in importance.values())
        
        # Test feature importance by type
        importance_gain = model.get_feature_importance_by_type('gain')
        assert len(importance_gain) == len(X_train.columns)
        assert all(isinstance(score, (float, np.number)) for score in importance_gain.values())
        
        importance_cover = model.get_feature_importance_by_type('cover')
        assert len(importance_cover) == len(X_train.columns)
        assert all(isinstance(score, (float, np.number)) for score in importance_cover.values())
    
    def test_model_summary(self, sample_data):
        """Test model summary generation."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        
        # Test summary before training
        summary = model.get_model_summary()
        assert summary["status"] == "not_trained"
        
        # Train and test summary after training
        model.train(X_train, y_train)
        summary = model.get_model_summary()
        
        assert summary["model_type"] == "XGBoost"
        assert summary["is_trained"] is True
        assert summary["n_estimators"] == 50
        assert summary["feature_count"] == len(X_train.columns)
        assert summary["use_gpu"] is False
        assert "top_features" in summary
        assert len(summary["top_features"]) <= 5
    
    def test_error_handling(self, sample_data):
        """Test error handling for various scenarios."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(random_state=42, use_gpu=False)
        
        # Test predictions before training
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X_test)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict_proba(X_test)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.get_feature_importance()
        
        # Train the model
        model.train(X_train, y_train)
        
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            model.predict(pd.DataFrame())
    
    def test_model_persistence(self, sample_data, tmp_path):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train a model
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        # Save the model
        model_path = tmp_path / "test_xgb_model"
        model.save_model(str(model_path))
        
        # Create a new model and load
        new_model = XGBoostModel()
        new_model.load_model(str(model_path))
        
        # Test that loaded model works
        assert new_model.is_trained
        assert new_model.feature_names == model.feature_names
        
        # Test predictions are the same
        original_predictions = model.predict(X_test)
        loaded_predictions = new_model.predict(X_test)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_zero_positive_samples_edge_case(self):
        """Test edge case with zero positive samples."""
        model = XGBoostModel(random_state=42)
        
        # Create data with no positive samples
        y_all_negative = pd.Series([0, 0, 0, 0, 0])
        
        scale_pos_weight = model._calculate_scale_pos_weight(y_all_negative)
        assert scale_pos_weight == 1.0  # Should default to 1.0
    
    def test_parameter_grid_structure(self):
        """Test that parameter grid has expected structure."""
        model = XGBoostModel(random_state=42)
        param_grid = model._get_param_grid()
        
        assert isinstance(param_grid, dict)
        assert 'n_estimators' in param_grid
        assert 'learning_rate' in param_grid
        assert 'max_depth' in param_grid
        assert all(isinstance(values, list) for values in param_grid.values())