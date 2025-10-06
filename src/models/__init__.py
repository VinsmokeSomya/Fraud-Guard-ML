"""
Fraud detection models package.

This package contains the base model interface and implementations
for various machine learning algorithms used in fraud detection.
"""

from .base_model import FraudModel, FraudModelInterface
from .logistic_regression_model import LogisticRegressionModel

__all__ = ['FraudModel', 'FraudModelInterface', 'LogisticRegressionModel']