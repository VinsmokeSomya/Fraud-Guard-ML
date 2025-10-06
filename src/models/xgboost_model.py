"""
XGBoost implementation for fraud detection.

This module implements an XGBoost model with GPU support, early stopping,
cross-validation, and SHAP values for model interpretability in fraud detection tasks.
"""

from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
import warnings
import shap

from .base_model import FraudModel


class XGBoostModel(FraudModel):
    """
    XGBoost model for fraud detection with GPU support, early stopping, and SHAP interpretability.
    
    This class wraps XGBoost's XGBClassifier with automatic hyperparameter tuning,
    early stopping, cross-validation, and SHAP values for model interpretability.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 random_state: int = 42,
                 learning_rate: float = 0.1,
                 max_depth: int = 6,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 scale_pos_weight: Optional[float] = None,
                 use_gpu: bool = False,
                 early_stopping_rounds: int = 10,
                 tune_hyperparameters: bool = True,
                 cv_folds: int = 5,
                 scoring: str = 'f1',
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize the XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            random_state: Random state for reproducibility
            learning_rate: Boosting learning rate
            max_depth: Maximum depth of trees
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            scale_pos_weight: Balancing of positive and negative weights (auto-calculated if None)
            use_gpu: Whether to use GPU acceleration
            early_stopping_rounds: Number of rounds for early stopping
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds for tuning
            scoring: Scoring metric for hyperparameter tuning
            n_jobs: Number of parallel jobs for training and grid search
            **kwargs: Additional parameters passed to XGBClassifier
        """
        super().__init__(model_name="XGBoost", **kwargs)
        
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.use_gpu = use_gpu
        self.early_stopping_rounds = early_stopping_rounds
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.additional_params = kwargs
        
        # Store best parameters from tuning
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.early_stopping_results_ = None
        
        # SHAP explainer for interpretability
        self.shap_explainer_ = None
        self.shap_values_cache_ = None
        
        # Update metadata with model-specific parameters
        self.metadata['parameters'].update({
            'n_estimators': n_estimators,
            'random_state': random_state,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'scale_pos_weight': scale_pos_weight,
            'use_gpu': use_gpu,
            'early_stopping_rounds': early_stopping_rounds,
            'tune_hyperparameters': tune_hyperparameters,
            'cv_folds': cv_folds,
            'scoring': scoring,
            'n_jobs': n_jobs,
            **kwargs
        })
    
    def _calculate_scale_pos_weight(self, y_train: pd.Series) -> float:
        """
        Calculate scale_pos_weight for class imbalance handling.
        
        Args:
            y_train: Training labels
            
        Returns:
            Calculated scale_pos_weight value
        """
        if self.scale_pos_weight is not None:
            return self.scale_pos_weight
        
        # Calculate ratio of negative to positive samples
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        
        if pos_count == 0:
            return 1.0
        
        return neg_count / pos_count
    
    def _get_param_grid(self) -> Dict[str, list]:
        """
        Define the parameter grid for hyperparameter tuning.
        
        Returns:
            Dictionary of parameters to search over
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        
        return param_grid
    
    def _create_base_model(self, scale_pos_weight: float = None) -> xgb.XGBClassifier:
        """
        Create a base XGBClassifier model with default parameters.
        
        Args:
            scale_pos_weight: Weight for positive class balancing
            
        Returns:
            Configured XGBClassifier instance
        """
        # Set tree method based on GPU usage
        tree_method = 'gpu_hist' if self.use_gpu else 'hist'
        
        # Set device based on GPU usage
        device = 'cuda' if self.use_gpu else 'cpu'
        
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight or self.scale_pos_weight,
            tree_method=tree_method,
            device=device,
            n_jobs=self.n_jobs,
            eval_metric='logloss',
            **self.additional_params
        )
    
    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best XGBClassifier model from grid search
        """
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        
        # Create base model without early stopping for grid search
        base_model = xgb.XGBClassifier(
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            tree_method='gpu_hist' if self.use_gpu else 'hist',
            device='cuda' if self.use_gpu else 'cpu',
            n_jobs=self.n_jobs,
            eval_metric='logloss',
            **self.additional_params
        )
        
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
                n_jobs=1,  # XGBoost handles parallelization internally
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
            'scoring': self.scoring,
            'scale_pos_weight': scale_pos_weight
        }
        
        return grid_search.best_estimator_
    
    def _fit_with_early_stopping(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame = None, y_val: pd.Series = None) -> xgb.XGBClassifier:
        """
        Fit model with early stopping using validation set.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Trained XGBClassifier with early stopping
        """
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        
        model = self._create_base_model(scale_pos_weight)
        
        # If no validation set provided, use a portion of training data
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
            )
        else:
            X_train_split, X_val_split = X_train, X_val
            y_train_split, y_val_split = y_train, y_val
        
        # Fit with early stopping
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            verbose=False
        )
        
        # Store early stopping results
        if hasattr(model, 'best_iteration'):
            self.early_stopping_results_ = {
                'best_iteration': model.best_iteration,
                'best_score': model.best_score
            }
            self.metadata['early_stopping'] = self.early_stopping_results_
        
        return model
    
    def _fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the XGBoost model with optional hyperparameter tuning and early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.tune_hyperparameters:
            self.model = self._tune_hyperparameters(X_train, y_train)
        else:
            self.model = self._fit_with_early_stopping(X_train, y_train)
    
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
        Get feature importance based on XGBoost feature importances.
        
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
    
    def get_feature_importance_by_type(self, importance_type: str = 'weight') -> Dict[str, float]:
        """
        Get feature importance by specific XGBoost importance type.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        # Get feature importances by type
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Ensure all features are included (XGBoost may exclude zero-importance features)
        feature_importance = {}
        for feature in self.feature_names:
            feature_importance[feature] = importance_dict.get(feature, 0.0)
        
        # Sort by importance (descending)
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def initialize_shap_explainer(self, X_background: pd.DataFrame = None) -> None:
        """
        Initialize SHAP explainer for model interpretability.
        
        Args:
            X_background: Background dataset for SHAP explainer (optional)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before initializing SHAP explainer")
        
        if X_background is not None and self.feature_names:
            X_background = X_background[self.feature_names]
        
        # Initialize TreeExplainer for XGBoost
        self.shap_explainer_ = shap.TreeExplainer(self.model, X_background)
        
        self.metadata['shap_initialized'] = True
    
    def get_shap_values(self, X: pd.DataFrame, check_additivity: bool = False) -> np.ndarray:
        """
        Calculate SHAP values for input data.
        
        Args:
            X: Input features
            check_additivity: Whether to check additivity of SHAP values
            
        Returns:
            SHAP values array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating SHAP values")
        
        if self.shap_explainer_ is None:
            # Initialize with default background (use training data subset if available)
            self.initialize_shap_explainer()
        
        # Select only the features used during training
        if self.feature_names:
            X = X[self.feature_names]
        
        # Calculate SHAP values
        shap_values = self.shap_explainer_.shap_values(X, check_additivity=check_additivity)
        
        # Cache for potential reuse
        self.shap_values_cache_ = shap_values
        
        return shap_values
    
    def get_shap_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Args:
            X: Input features for SHAP calculation
            
        Returns:
            Dictionary mapping feature names to SHAP-based importance scores
        """
        shap_values = self.get_shap_values(X)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature importance dictionary
        feature_importance = dict(zip(self.feature_names, mean_abs_shap))
        
        # Sort by importance (descending)
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            X: Input features
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with prediction explanation
        """
        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} out of range for dataset of size {len(X)}")
        
        # Get prediction and probability
        prediction = self.predict(X.iloc[[sample_idx]])[0]
        probability = self.predict_proba(X.iloc[[sample_idx]])[0]
        
        # Get SHAP values
        shap_values = self.get_shap_values(X.iloc[[sample_idx]])
        
        # Create explanation
        explanation = {
            'prediction': int(prediction),
            'fraud_probability': float(probability[1]),
            'base_value': float(self.shap_explainer_.expected_value),
            'shap_values': dict(zip(self.feature_names, shap_values[0])),
            'feature_values': dict(zip(self.feature_names, X.iloc[sample_idx].values))
        }
        
        return explanation
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary with cross-validation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")
        
        # Create stratified k-fold
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Set up scoring
        if self.scoring == 'f1':
            scorer = make_scorer(f1_score, average='binary')
        else:
            scorer = self.scoring
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scorer, n_jobs=1)
        
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(cv_scores.mean()),
            'std_score': float(cv_scores.std()),
            'cv_folds': self.cv_folds,
            'scoring': self.scoring
        }
        
        return cv_results
    
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
    
    def get_early_stopping_results(self) -> Optional[Dict[str, Any]]:
        """
        Get early stopping results if early stopping was used.
        
        Returns:
            Dictionary with early stopping results or None if not used
        """
        return self.early_stopping_results_
    
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
            "model_type": "XGBoost",
            "is_trained": self.is_trained,
            "n_estimators": self.n_estimators,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "use_gpu": self.use_gpu,
            "hyperparameter_tuning": self.tune_hyperparameters,
            "early_stopping_enabled": self.early_stopping_rounds > 0,
            "shap_initialized": self.shap_explainer_ is not None
        }
        
        if self.early_stopping_results_:
            summary["early_stopping_results"] = self.early_stopping_results_
        
        if self.tune_hyperparameters and self.best_params_:
            summary["best_parameters"] = self.best_params_
            summary["best_cv_score"] = self.best_score_
        
        # Add top 5 most important features
        feature_importance = self.get_feature_importance()
        summary["top_features"] = dict(list(feature_importance.items())[:5])
        
        return summary
    
    def get_booster_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying XGBoost booster.
        
        Returns:
            Dictionary with booster information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting booster info")
        
        booster = self.model.get_booster()
        
        return {
            "num_boosted_rounds": booster.num_boosted_rounds(),
            "num_features": booster.num_features(),
            "feature_names": booster.feature_names,
            "feature_types": booster.feature_types
        }