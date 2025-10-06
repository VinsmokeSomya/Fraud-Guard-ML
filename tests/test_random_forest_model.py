"""
Tests for RandomForestModel implementation.

This module contains unit tests for the RandomForestModel class,
testing training, prediction, feature importance, and OOB scoring functionality.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.random_forest_model import RandomForestModel


class TestRandomForestModel:
    """Test cases for RandomForestModel class."""
    
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
        """Test RandomForestModel initialization."""
        model = RandomForestModel(
            n_estimators=50,
            random_state=42,
            class_weight='balanced'
        )
        
        assert model.model_name == "RandomForest"
        assert model.n_estimators == 50
        assert model.random_state == 42
        assert model.class_weight == 'balanced'
        assert not model.is_trained
        assert model.oob_score is True  # Default value
    
    def test_model_training_without_tuning(self, sample_data):
        """Test model training without hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False
        )
        
        # Train the model
        model.train(X_train, y_train)
        
        assert model.is_trained
        assert model.model is not None
        assert model.feature_names == list(X_train.columns)
        assert model.oob_score_ is not None  # Should have OOB score
    
    def test_model_training_with_tuning(self, sample_data):
        """Test model training with hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=True,
            cv_folds=3  # Reduced for faster testing
        )
        
        # Train the model
        model.train(X_train, y_train)
        
        assert model.is_trained
        assert model.model is not None
        assert model.best_params_ is not None
        assert model.best_score_ is not None
        assert model.feature_names == list(X_train.columns)
    
    def test_predictions(self, sample_data):
        """Test model predictions."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False
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
        
        model = RandomForestModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False
        )
        model.train(X_train, y_train)
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X_train.columns)
        assert all(isinstance(score, float) for score in importance.values())
        assert all(score >= 0 for score in importance.values())
        
        # Test feature importance ranking
        ranking = model.get_feature_importance_ranking()
        assert len(ranking) == len(X_train.columns)
        assert set(ranking.values()) == set(range(1, len(X_train.columns) + 1))
        
        # Test feature importance standard deviation
        importance_std = model.get_feature_importance_std()
        assert len(importance_std) == len(X_train.columns)
        assert all(isinstance(std, float) for std in importance_std.values())
        assert all(std >= 0 for std in importance_std.values())
    
    def test_oob_score(self, sample_data):
        """Test out-of-bag scoring."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(
            n_estimators=50,
            random_state=42,
            oob_score=True,
            tune_hyperparameters=False
        )
        model.train(X_train, y_train)
        
        oob_score = model.get_oob_score()
        assert oob_score is not None
        assert isinstance(oob_score, float)
        assert 0 <= oob_score <= 1
    
    def test_oob_score_disabled(self, sample_data):
        """Test when OOB scoring is disabled."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(
            n_estimators=50,
            random_state=42,
            oob_score=False,
            tune_hyperparameters=False
        )
        model.train(X_train, y_train)
        
        oob_score = model.get_oob_score()
        assert oob_score is None
    
    def test_individual_tree_predictions(self, sample_data):
        """Test individual tree predictions."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(
            n_estimators=10,  # Small number for testing
            random_state=42,
            tune_hyperparameters=False
        )
        model.train(X_train, y_train)
        
        tree_predictions = model.get_individual_tree_predictions(X_test)
        assert tree_predictions.shape == (len(X_test), 10)
        assert all(pred in [0, 1] for pred in tree_predictions.flatten())
    
    def test_model_summary(self, sample_data):
        """Test model summary generation."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False
        )
        
        # Test summary before training
        summary = model.get_model_summary()
        assert summary["status"] == "not_trained"
        
        # Train and test summary after training
        model.train(X_train, y_train)
        summary = model.get_model_summary()
        
        assert summary["model_type"] == "RandomForest"
        assert summary["is_trained"] is True
        assert summary["n_estimators"] == 50
        assert summary["feature_count"] == len(X_train.columns)
        assert "top_features" in summary
        assert len(summary["top_features"]) <= 5
    
    def test_tuning_results(self, sample_data):
        """Test hyperparameter tuning results."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Test with tuning enabled
        model_with_tuning = RandomForestModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=True,
            cv_folds=3
        )
        model_with_tuning.train(X_train, y_train)
        
        tuning_results = model_with_tuning.get_tuning_results()
        assert tuning_results is not None
        assert "best_params" in tuning_results
        assert "best_score" in tuning_results
        
        # Test with tuning disabled
        model_without_tuning = RandomForestModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False
        )
        model_without_tuning.train(X_train, y_train)
        
        tuning_results = model_without_tuning.get_tuning_results()
        assert tuning_results is None
    
    def test_error_handling(self, sample_data):
        """Test error handling for various scenarios."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestModel(random_state=42)
        
        # Test predictions before training
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X_test)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict_proba(X_test)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.get_feature_importance()
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.get_oob_score()
        
        # Train the model
        model.train(X_train, y_train)
        
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            model.predict(pd.DataFrame())
    
    def test_model_persistence(self, sample_data, tmp_path):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train a model
        model = RandomForestModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False
        )
        model.train(X_train, y_train)
        
        # Save the model
        model_path = tmp_path / "test_rf_model"
        model.save_model(str(model_path))
        
        # Create a new model and load
        new_model = RandomForestModel()
        new_model.load_model(str(model_path))
        
        # Test that loaded model works
        assert new_model.is_trained
        assert new_model.feature_names == model.feature_names
        
        # Test predictions are the same
        original_predictions = model.predict(X_test)
        loaded_predictions = new_model.predict(X_test)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)