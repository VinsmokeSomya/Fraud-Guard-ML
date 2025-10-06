"""
Fraud detection models package.

This package contains the base model interface and implementations
for various machine learning algorithms used in fraud detection.
"""

from .base_model import FraudModel, FraudModelInterface
from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

__all__ = ['FraudModel', 'FraudModelInterface', 'LogisticRegressionModel', 'RandomForestModel', 'XGBoostModel']