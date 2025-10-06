"""
Fraud detection models package.

This package contains the base model interface and implementations
for various machine learning algorithms used in fraud detection.
"""

from .base_model import FraudModel, FraudModelInterface

__all__ = ['FraudModel', 'FraudModelInterface']