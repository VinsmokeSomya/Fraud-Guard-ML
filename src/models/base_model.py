"""
Base fraud detection model interface and abstract class.

This module defines the common interface that all fraud detection models must implement,
including training, prediction, serialization, and metadata tracking functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime
from pathlib import Path


class FraudModelInterface(ABC):
    """Abstract interface for fraud detection models."""
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the fraud detection model.
        
        Args:
            X_train: Training features
            y_train: Training labels (0 for legitimate, 1 for fraud)
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0 for legitimate, 1 for fraud)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass


class FraudModel(FraudModelInterface):
    """
    Base abstract class for fraud detection models with common functionality.
    
    Provides model serialization, metadata tracking, and common utilities
    that all fraud detection models can inherit.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the fraud model.
        
        Args:
            model_name: Name identifier for the model
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.metadata = {
            'model_name': model_name,
            'created_at': datetime.now().isoformat(),
            'training_time': None,
            'parameters': kwargs,
            'training_samples': None,
            'feature_count': None,
            'class_distribution': None
        }
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input data format and features.
        
        Args:
            X: Input features to validate
            
        Raises:
            ValueError: If input validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            extra_features = set(X.columns) - set(self.feature_names)
            if extra_features:
                # Log warning but don't fail - select only required features
                X = X[self.feature_names]
    
    def _update_training_metadata(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                training_time: float) -> None:
        """
        Update metadata after training.
        
        Args:
            X_train: Training features
            y_train: Training labels
            training_time: Time taken for training in seconds
        """
        self.metadata.update({
            'training_time': training_time,
            'training_samples': len(X_train),
            'feature_count': len(X_train.columns),
            'class_distribution': y_train.value_counts().to_dict(),
            'trained_at': datetime.now().isoformat()
        })
        self.feature_names = list(X_train.columns)
        self.is_trained = True
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model with timing and metadata tracking.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self._validate_input(X_train)
        
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")
        
        start_time = time.time()
        self._fit_model(X_train, y_train)
        training_time = time.time() - start_time
        
        self._update_training_metadata(X_train, y_train, training_time)
    
    @abstractmethod
    def _fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Internal method to fit the specific model implementation.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions with validation.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self._validate_input(X)
        return self._predict_implementation(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities with validation.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self._validate_input(X)
        return self._predict_proba_implementation(X)
    
    @abstractmethod
    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Internal prediction implementation."""
        pass
    
    @abstractmethod
    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Internal probability prediction implementation."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and metadata to disk.
        
        Args:
            filepath: Path to save the model (without extension)
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model object
        model_path = filepath.with_suffix('.joblib')
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save feature names
        features_path = filepath.with_name(f"{filepath.name}_features.json")
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and metadata from disk.
        
        Args:
            filepath: Path to the saved model (without extension)
        """
        filepath = Path(filepath)
        
        # Load the model object
        model_path = filepath.with_suffix('.joblib')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load feature names
        features_path = filepath.with_name(f"{filepath.name}_features.json")
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
        
        self.is_trained = True
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary containing model metadata
        """
        return self.metadata.copy()
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get training-specific information.
        
        Returns:
            Dictionary with training details
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "training_time": self.metadata.get('training_time'),
            "training_samples": self.metadata.get('training_samples'),
            "feature_count": self.metadata.get('feature_count'),
            "class_distribution": self.metadata.get('class_distribution'),
            "trained_at": self.metadata.get('trained_at')
        }