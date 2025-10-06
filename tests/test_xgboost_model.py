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
    
    def test_model_initialization_with_gpu(self):
        """Test XGBoostModel initialization with GPU enabled."""
        model = XGBoostModel(
            n_estimators=50,
            use_gpu=True,
            random_state=42
        )
        
        assert model.use_gpu is True
        assert model.metadata['parameters']['use_gpu'] is True
    
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
    
    def test_scale_pos_weight_manual(self, sample_data):
        """Test manual scale_pos_weight setting."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(scale_pos_weight=2.0, random_state=42)
        scale_pos_weight = model._calculate_scale_pos_weight(y_train)
        
        assert scale_pos_weight == 2.0
    
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
        assert model.early_stopping_results_ is not None  # Early stopping used
    
    def test_model_training_with_tuning(self, sample_data):
        """Test model training with hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=True,
            cv_folds=3,  # Reduced for faster testing
            use_gpu=False
        )
        
        # Override parameter grid for faster testing
        def get_small_param_grid():
            return {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 6]
            }
        
        model._get_param_grid = get_small_param_grid
        
        # Train the model
        model.train(X_train, y_train)
        
        assert model.is_trained
        assert model.model is not None
        assert model.best_params_ is not None
        assert model.best_score_ is not None
        assert model.feature_names == list(X_train.columns)
        assert 'hyperparameter_tuning' in model.metadata
    
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
        assert all(isinstance(score, float) for score in importance.values())
        assert all(score >= 0 for score in importance.values())
        
        # Test feature importance by type
        importance_gain = model.get_feature_importance_by_type('gain')
        assert len(importance_gain) == len(X_train.columns)
        assert all(isinstance(score, float) for score in importance_gain.values())
        
        importance_cover = model.get_feature_importance_by_type('cover')
        assert len(importance_cover) == len(X_train.columns)
        assert all(isinstance(score, float) for score in importance_cover.values())
    
    @patch('shap.TreeExplainer')
    def test_shap_initialization(self, mock_tree_explainer, sample_data):
        """Test SHAP explainer initialization."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        # Mock the SHAP explainer
        mock_explainer = MagicMock()
        mock_tree_explainer.return_value = mock_explainer
        
        # Initialize SHAP explainer
        model.initialize_shap_explainer(X_train.head(10))
        
        assert model.shap_explainer_ is not None
        assert model.metadata.get('shap_initialized') is True
        mock_tree_explainer.assert_called_once()
    
    @patch('shap.TreeExplainer')
    def test_shap_values_calculation(self, mock_tree_explainer, sample_data):
        """Test SHAP values calculation."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        # Mock the SHAP explainer and values
        mock_explainer = MagicMock()
        mock_shap_values = np.random.rand(len(X_test), len(X_train.columns))
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_explainer.expected_value = 0.1
        mock_tree_explainer.return_value = mock_explainer
        
        # Get SHAP values
        shap_values = model.get_shap_values(X_test)
        
        assert shap_values is not None
        assert shap_values.shape == (len(X_test), len(X_train.columns))
        assert model.shap_values_cache_ is not None
    
    @patch('shap.TreeExplainer')
    def test_shap_feature_importance(self, mock_tree_explainer, sample_data):
        """Test SHAP-based feature importance."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        # Mock the SHAP explainer and values
        mock_explainer = MagicMock()
        mock_shap_values = np.random.rand(len(X_test), len(X_train.columns))
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_tree_explainer.return_value = mock_explainer
        
        # Get SHAP feature importance
        shap_importance = model.get_shap_feature_importance(X_test)
        
        assert len(shap_importance) == len(X_train.columns)
        assert all(isinstance(score, float) for score in shap_importance.values())
        assert all(score >= 0 for score in shap_importance.values())
    
    @patch('shap.TreeExplainer')
    def test_explain_prediction(self, mock_tree_explainer, sample_data):
        """Test single prediction explanation."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        # Mock the SHAP explainer and values
        mock_explainer = MagicMock()
        mock_shap_values = np.random.rand(1, len(X_train.columns))
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_explainer.expected_value = 0.1
        mock_tree_explainer.return_value = mock_explainer
        
        # Explain a prediction
        explanation = model.explain_prediction(X_test, sample_idx=0)
        
        assert 'prediction' in explanation
        assert 'fraud_probability' in explanation
        assert 'base_value' in explanation
        assert 'shap_values' in explanation
        assert 'feature_values' in explanation
        assert len(explanation['shap_values']) == len(X_train.columns)
        assert len(explanation['feature_values']) == len(X_train.columns)
    
    def test_cross_validation(self, sample_data):
        """Test cross-validation functionality."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            cv_folds=3,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        # Perform cross-validation
        cv_results = model.cross_validate_model(X_train, y_train)
        
        assert 'cv_scores' in cv_results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert len(cv_results['cv_scores']) == 3
        assert isinstance(cv_results['mean_score'], float)
        assert isinstance(cv_results['std_score'], float)
    
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
    
    def test_booster_info(self, sample_data):
        """Test booster information extraction."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        booster_info = model.get_booster_info()
        
        assert 'num_boosted_rounds' in booster_info
        assert 'num_features' in booster_info
        assert 'feature_names' in booster_info
        assert 'feature_types' in booster_info
        assert isinstance(booster_info['num_boosted_rounds'], int)
        assert isinstance(booster_info['num_features'], int)
    
    def test_tuning_results(self, sample_data):
        """Test hyperparameter tuning results."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Test with tuning disabled
        model_without_tuning = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=False,
            use_gpu=False
        )
        model_without_tuning.train(X_train, y_train)
        
        tuning_results = model_without_tuning.get_tuning_results()
        assert tuning_results is None
        
        # Test with tuning enabled
        model_with_tuning = XGBoostModel(
            n_estimators=50,
            random_state=42,
            tune_hyperparameters=True,
            cv_folds=3,
            use_gpu=False
        )
        
        # Override parameter grid for faster testing
        def get_small_param_grid():
            return {
                'n_estimators': [50],
                'learning_rate': [0.1],
                'max_depth': [3]
            }
        
        model_with_tuning._get_param_grid = get_small_param_grid
        model_with_tuning.train(X_train, y_train)
        
        tuning_results = model_with_tuning.get_tuning_results()
        assert tuning_results is not None
        assert "best_params" in tuning_results
        assert "best_score" in tuning_results
    
    def test_early_stopping_results(self, sample_data):
        """Test early stopping results."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostModel(
            n_estimators=100,
            random_state=42,
            tune_hyperparameters=False,
            early_stopping_rounds=10,
            use_gpu=False
        )
        model.train(X_train, y_train)
        
        early_stopping_results = model.get_early_stopping_results()
        
        # Early stopping results should be available
        if early_stopping_results is not None:
            assert 'best_iteration' in early_stopping_results
            assert 'best_score' in early_stopping_results
            assert isinstance(early_stopping_results['best_iteration'], int)
    
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
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.cross_validate_model(X_train, y_train)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.get_booster_info()
        
        # Train the model
        model.train(X_train, y_train)
        
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            model.predict(pd.DataFrame())
        
        # Test explain_prediction with invalid index
        with pytest.raises(ValueError, match="Sample index .* out of range"):
            model.explain_prediction(X_test, sample_idx=len(X_test))
    
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