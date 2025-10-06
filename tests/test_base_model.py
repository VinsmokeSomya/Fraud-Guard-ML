"""
Tests for the base fraud model interface and abstract class.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock
from src.models.base_model import FraudModel, FraudModelInterface


class SimpleMockModel:
    """Simple mock model that can be serialized."""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, X, y):
        self.fitted = True


class MockFraudModel(FraudModel):
    """Mock implementation of FraudModel for testing."""
    
    def __init__(self, model_name: str = "mock_model", **kwargs):
        super().__init__(model_name, **kwargs)
    
    def _fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Mock training implementation."""
        self.model = SimpleMockModel()
        self.model.fit(X_train, y_train)
    
    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Mock prediction implementation."""
        np.random.seed(42)  # For reproducible tests
        return np.random.randint(0, 2, size=len(X))
    
    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Mock probability prediction implementation."""
        np.random.seed(42)  # For reproducible tests
        probs = np.random.rand(len(X), 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    
    def get_feature_importance(self) -> dict:
        """Mock feature importance implementation."""
        if not self.is_trained or not self.feature_names:
            return {}
        np.random.seed(42)  # For reproducible tests
        return {name: np.random.rand() for name in self.feature_names}


class TestFraudModelInterface:
    """Test the FraudModelInterface abstract class."""
    
    def test_cannot_instantiate_interface(self):
        """Test that the interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FraudModelInterface()


class TestFraudModel:
    """Test the FraudModel base class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50]
        })
        y = pd.Series([0, 1, 0, 1, 0])
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained mock model."""
        X, y = sample_data
        model = MockFraudModel("test_model", param1="value1")
        model.train(X, y)
        return model
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = MockFraudModel("test_model", param1="value1", param2=42)
        
        assert model.model_name == "test_model"
        assert not model.is_trained
        assert model.feature_names is None
        assert model.metadata['model_name'] == "test_model"
        assert model.metadata['parameters'] == {'param1': 'value1', 'param2': 42}
    
    def test_training(self, sample_data):
        """Test model training."""
        X, y = sample_data
        model = MockFraudModel("test_model")
        
        model.train(X, y)
        
        assert model.is_trained
        assert model.feature_names == ['feature1', 'feature2', 'feature3']
        assert model.metadata['training_samples'] == 5
        assert model.metadata['feature_count'] == 3
        assert model.metadata['training_time'] is not None
        assert model.metadata['class_distribution'] == {0: 3, 1: 2}
    
    def test_training_validation(self):
        """Test training input validation."""
        model = MockFraudModel("test_model")
        
        # Test with non-DataFrame input
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            model.train([[1, 2, 3]], pd.Series([0, 1]))
        
        # Test with mismatched lengths
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="X_train and y_train must have the same length"):
            model.train(X, y)
    
    def test_prediction_before_training(self, sample_data):
        """Test that prediction fails before training."""
        X, _ = sample_data
        model = MockFraudModel("test_model")
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model.predict(X)
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model.predict_proba(X)
    
    def test_prediction_after_training(self, trained_model, sample_data):
        """Test prediction after training."""
        X, _ = sample_data
        
        predictions = trained_model.predict(X)
        probabilities = trained_model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert predictions.dtype in [np.int32, np.int64]
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_feature_importance(self, trained_model):
        """Test feature importance extraction."""
        importance = trained_model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert all(name in importance for name in ['feature1', 'feature2', 'feature3'])
    
    def test_model_serialization(self, trained_model):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model")
            
            # Save the model
            trained_model.save_model(model_path)
            
            # Check that files were created
            assert os.path.exists(f"{model_path}.joblib")
            assert os.path.exists(f"{model_path}.json")
            assert os.path.exists(f"{model_path}_features.json")
            
            # Load the model into a new instance
            new_model = MockFraudModel("loaded_model")
            new_model.load_model(model_path)
            
            assert new_model.is_trained
            assert new_model.feature_names == trained_model.feature_names
            assert new_model.metadata['training_samples'] == trained_model.metadata['training_samples']
    
    def test_save_untrained_model(self):
        """Test that saving an untrained model raises an error."""
        model = MockFraudModel("test_model")
        
        with pytest.raises(ValueError, match="Cannot save untrained model"):
            model.save_model("test_path")
    
    def test_load_nonexistent_model(self):
        """Test loading a non-existent model."""
        model = MockFraudModel("test_model")
        
        with pytest.raises(FileNotFoundError):
            model.load_model("nonexistent_path")
    
    def test_metadata_access(self, trained_model):
        """Test metadata access methods."""
        metadata = trained_model.get_metadata()
        training_info = trained_model.get_training_info()
        
        assert isinstance(metadata, dict)
        assert metadata['model_name'] == "test_model"
        
        assert training_info['status'] == "trained"
        assert 'training_time' in training_info
        assert 'training_samples' in training_info
    
    def test_untrained_model_info(self):
        """Test training info for untrained model."""
        model = MockFraudModel("test_model")
        training_info = model.get_training_info()
        
        assert training_info['status'] == "not_trained"
    
    def test_input_validation_with_missing_features(self, trained_model):
        """Test input validation with missing features."""
        # Create data with missing features
        X_missing = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required features"):
            trained_model.predict(X_missing)
    
    def test_empty_dataframe_validation(self, trained_model):
        """Test validation with empty DataFrame."""
        X_empty = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            trained_model.predict(X_empty)