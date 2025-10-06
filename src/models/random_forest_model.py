"""
Random Forest implementation for fraud detection.

This module implements a Random Forest model with hyperparameter tuning,
feature importance extraction, and out-of-bag scoring for fraud detection tasks.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import warnings

from .base_model import FraudModel


class RandomForestModel(FraudModel):
    """
    Random Forest model for fraud detection with hyperparameter tuning.
    
    This class wraps scikit-learn's RandomForestClassifier with automatic hyperparameter
    tuning using GridSearchCV, feature importance extraction, and out-of-bag scoring.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 random_state: int = 42,
                 class_weight: Optional[str] = 'balanced',
                 oob_score: bool = True,
                 tune_hyperparameters: bool = True,
                 cv_folds: int = 5,
                 scoring: str = 'f1',
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            random_state: Random state for reproducibility
            class_weight: Strategy for handling class imbalance ('balanced', None, or dict)
            oob_score: Whether to use out-of-bag samples for validation
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds for tuning
            scoring: Scoring metric for hyperparameter tuning
            n_jobs: Number of parallel jobs for training and grid search
            **kwargs: Additional parameters passed to RandomForestClassifier
        """
        super().__init__(model_name="RandomForest", **kwargs)
        
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.class_weight = class_weight
        self.oob_score = oob_score
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.additional_params = kwargs
        
        # Store best parameters from tuning
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.oob_score_ = None
        
        # Update metadata with model-specific parameters
        self.metadata['parameters'].update({
            'n_estimators': n_estimators,
            'random_state': random_state,
            'class_weight': class_weight,
            'oob_score': oob_score,
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
            Dictionary of parameters to search over
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True]  # Keep bootstrap=True for OOB scoring
        }
        
        return param_grid
    
    def _create_base_model(self) -> RandomForestClassifier:
        """
        Create a base RandomForestClassifier model with default parameters.
        
        Returns:
            Configured RandomForestClassifier instance
        """
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight=self.class_weight,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            **self.additional_params
        )
    
    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best RandomForestClassifier model from grid search
        """
        base_model = self._create_base_model()
        param_grid = self._get_param_grid()
        
        # Create stratified k-fold for cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Set up scoring
        if self.scoring == 'f1':
            scorer = make_scorer(f1_score, average='binary')
        else:
            scorer = self.scoring
        
        # Suppress warnings during grid search
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
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
        Fit the random forest model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.tune_hyperparameters:
            self.model = self._tune_hyperparameters(X_train, y_train)
        else:
            self.model = self._create_base_model()
            self.model.fit(X_train, y_train)
        
        # Store OOB score if available
        if self.oob_score and hasattr(self.model, 'oob_score_'):
            self.oob_score_ = self.model.oob_score_
            self.metadata['oob_score'] = self.oob_score_
    
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
        Get feature importance based on Random Forest feature importances.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        # Get feature importances from the trained model
        importances = self.model.feature_importances_
        
        # Create feature importance dictionary
        feature_importance = dict(zip(self.feature_names, importances))
        
        # Sort by importance (descending)
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def get_feature_importance_ranking(self) -> Dict[str, int]:
        """
        Get feature importance ranking (1 = most important).
        
        Returns:
            Dictionary mapping feature names to their importance rank
        """
        feature_importance = self.get_feature_importance()
        
        # Create ranking dictionary
        ranking = {}
        for rank, (feature, _) in enumerate(feature_importance.items(), 1):
            ranking[feature] = rank
        
        return ranking
    
    def get_oob_score(self) -> Optional[float]:
        """
        Get the out-of-bag score if available.
        
        Returns:
            OOB score or None if not available
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting OOB score")
        
        return self.oob_score_
    
    def get_tree_count(self) -> int:
        """
        Get the number of trees in the forest.
        
        Returns:
            Number of estimators in the trained model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting tree count")
        
        return self.model.n_estimators
    
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
    
    def get_individual_tree_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from individual trees in the forest.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, n_estimators) with individual tree predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting tree predictions")
        
        # Select only the features used during training
        if self.feature_names:
            X = X[self.feature_names]
        
        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ]).T
        
        return tree_predictions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the trained model.
        
        Returns:
            Dictionary with model summary information
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        summary = {
            "model_type": "RandomForest",
            "is_trained": self.is_trained,
            "n_estimators": self.get_tree_count(),
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "class_weight": self.class_weight,
            "hyperparameter_tuning": self.tune_hyperparameters,
            "oob_score_enabled": self.oob_score
        }
        
        if self.oob_score_:
            summary["oob_score"] = self.oob_score_
        
        if self.tune_hyperparameters and self.best_params_:
            summary["best_parameters"] = self.best_params_
            summary["best_cv_score"] = self.best_score_
        
        # Add top 5 most important features
        feature_importance = self.get_feature_importance()
        summary["top_features"] = dict(list(feature_importance.items())[:5])
        
        return summary
    
    def get_feature_importance_std(self) -> Dict[str, float]:
        """
        Get standard deviation of feature importances across trees.
        
        Returns:
            Dictionary mapping feature names to importance standard deviations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance std")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        # Get feature importances from all trees
        tree_importances = np.array([
            tree.feature_importances_ for tree in self.model.estimators_
        ])
        
        # Calculate standard deviation across trees
        importances_std = np.std(tree_importances, axis=0)
        
        # Create dictionary
        importance_std = dict(zip(self.feature_names, importances_std))
        
        return importance_std