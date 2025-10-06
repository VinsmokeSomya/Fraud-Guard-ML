"""
Logistic Regression implementation for fraud detection.

This module implements a logistic regression model with hyperparameter tuning
and class balancing capabilities for fraud detection tasks.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import warnings

from .base_model import FraudModel


class LogisticRegressionModel(FraudModel):
    """
    Logistic Regression model for fraud detection with hyperparameter tuning.
    
    This class wraps scikit-learn's LogisticRegression with automatic hyperparameter
    tuning using GridSearchCV and built-in class balancing capabilities.
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 max_iter: int = 1000,
                 class_weight: Optional[str] = 'balanced',
                 tune_hyperparameters: bool = True,
                 cv_folds: int = 5,
                 scoring: str = 'f1',
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize the Logistic Regression model.
        
        Args:
            random_state: Random state for reproducibility
            max_iter: Maximum number of iterations for convergence
            class_weight: Strategy for handling class imbalance ('balanced', None, or dict)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds for tuning
            scoring: Scoring metric for hyperparameter tuning
            n_jobs: Number of parallel jobs for grid search
            **kwargs: Additional parameters passed to LogisticRegression
        """
        super().__init__(model_name="LogisticRegression", **kwargs)
        
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.additional_params = kwargs
        
        # Store best parameters from tuning
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        
        # Update metadata with model-specific parameters
        self.metadata['parameters'].update({
            'random_state': random_state,
            'max_iter': max_iter,
            'class_weight': class_weight,
            'tune_hyperparameters': tune_hyperparameters,
            'cv_folds': cv_folds,
            'scoring': scoring,
            'n_jobs': n_jobs,
            **kwargs
        })
    
    def _get_param_grid(self) -> Dict[str, list]:
        """
        Define the parameter grid for hyperparameter tuning.
        
        Returns:
            List of parameter dictionaries to avoid incompatible combinations
        """
        # Create separate parameter grids for different penalty-solver combinations
        param_grids = [
            # L2 penalty with liblinear and saga solvers
            {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'saga']
            },
            # L1 penalty with liblinear and saga solvers
            {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1'],
                'solver': ['liblinear', 'saga']
            },
            # Elasticnet penalty only with saga solver
            {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['elasticnet'],
                'solver': ['saga'],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
        ]
        
        return param_grids
    
    def _create_base_model(self) -> LogisticRegression:
        """
        Create a base LogisticRegression model with default parameters.
        
        Returns:
            Configured LogisticRegression instance
        """
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            **self.additional_params
        )
    
    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best LogisticRegression model from grid search
        """
        base_model = self._create_base_model()
        param_grids = self._get_param_grid()
        
        # Create stratified k-fold for cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Set up scoring
        if self.scoring == 'f1':
            scorer = make_scorer(f1_score, average='binary')
        else:
            scorer = self.scoring
        
        # Suppress convergence warnings during grid search
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Perform grid search with list of parameter grids
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grids,
                cv=cv,
                scoring=scorer,
                n_jobs=self.n_jobs,
                verbose=0,
                error_score='raise'
            )
            
            grid_search.fit(X_train, y_train)
        
        # Store results
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.cv_results_ = grid_search.cv_results_
        
        # Update metadata with tuning results
        self.metadata['hyperparameter_tuning'] = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_folds': self.cv_folds,
            'scoring': self.scoring
        }
        
        return grid_search.best_estimator_
    
    def _fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the logistic regression model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.tune_hyperparameters:
            self.model = self._tune_hyperparameters(X_train, y_train)
        else:
            self.model = self._create_base_model()
            self.model.fit(X_train, y_train)
    
    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """
        Internal implementation for binary predictions.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0 or 1)
        """
        # Select only the features used during training
        if self.feature_names:
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """
        Internal implementation for probability predictions.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities with shape (n_samples, 2)
        """
        # Select only the features used during training
        if self.feature_names:
            X = X[self.feature_names]
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on logistic regression coefficients.
        
        Returns:
            Dictionary mapping feature names to absolute coefficient values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        # Get absolute values of coefficients as importance scores
        coefficients = np.abs(self.model.coef_[0])
        
        # Create feature importance dictionary
        feature_importance = dict(zip(self.feature_names, coefficients))
        
        # Sort by importance (descending)
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Get the raw logistic regression coefficients (with sign).
        
        Returns:
            Dictionary mapping feature names to coefficient values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting coefficients")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        coefficients = self.model.coef_[0]
        return dict(zip(self.feature_names, coefficients))
    
    def get_intercept(self) -> float:
        """
        Get the logistic regression intercept.
        
        Returns:
            Intercept value
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting intercept")
        
        return float(self.model.intercept_[0])
    
    def get_tuning_results(self) -> Optional[Dict[str, Any]]:
        """
        Get hyperparameter tuning results if tuning was performed.
        
        Returns:
            Dictionary with tuning results or None if no tuning was done
        """
        if not self.tune_hyperparameters or self.best_params_ is None:
            return None
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': self.cv_results_
        }
    
    def predict_fraud_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get the probability of fraud (class 1) for input transactions.
        
        Args:
            X: Input features
            
        Returns:
            Array of fraud probabilities
        """
        probabilities = self.predict_proba(X)
        return probabilities[:, 1]  # Return probability of positive class (fraud)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the trained model.
        
        Returns:
            Dictionary with model summary information
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        summary = {
            "model_type": "LogisticRegression",
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "intercept": self.get_intercept(),
            "class_weight": self.class_weight,
            "hyperparameter_tuning": self.tune_hyperparameters
        }
        
        if self.tune_hyperparameters and self.best_params_:
            summary["best_parameters"] = self.best_params_
            summary["best_cv_score"] = self.best_score_
        
        # Add top 5 most important features
        feature_importance = self.get_feature_importance()
        summary["top_features"] = dict(list(feature_importance.items())[:5])
        
        return summary