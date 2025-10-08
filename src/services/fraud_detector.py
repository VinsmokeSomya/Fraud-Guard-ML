"""
FraudDetector service class for real-time fraud detection and risk assessment.

This module provides the main FraudDetector service that orchestrates fraud detection
by combining trained models with risk factor analysis to provide comprehensive
fraud scoring, batch predictions, and detailed fraud explanations.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from src.models.base_model import FraudModelInterface
from src.visualization.fraud_pattern_analyzer import FraudPatternAnalyzer
from src.utils.config import get_setting, get_logger

logger = get_logger(__name__)


class FraudDetector:
    """
    Main fraud detection service class.
    
    Provides real-time transaction scoring, batch prediction functionality,
    and comprehensive fraud explanation with risk factor identification.
    """
    
    def __init__(self, 
                 model: Optional[FraudModelInterface] = None,
                 risk_threshold: float = 0.5,
                 high_risk_threshold: float = 0.8,
                 enable_explanations: bool = True,
                 alert_manager: Optional['AlertManager'] = None):
        """
        Initialize the FraudDetector service.
        
        Args:
            model: Trained fraud detection model
            risk_threshold: Threshold for classifying transactions as fraudulent
            high_risk_threshold: Threshold for high-risk alerts
            enable_explanations: Whether to generate detailed fraud explanations
            alert_manager: AlertManager instance for handling alerts
        """
        self.model = model
        self.risk_threshold = risk_threshold
        self.high_risk_threshold = high_risk_threshold
        self.enable_explanations = enable_explanations
        
        # Initialize AlertManager if provided
        self.alert_manager = alert_manager
        
        # Initialize fraud pattern analyzer for risk factor identification
        self.pattern_analyzer = FraudPatternAnalyzer()
        
        # Risk factor weights for explanation scoring
        self.risk_factor_weights = {
            'transaction_type': 0.25,
            'amount_based': 0.20,
            'balance_patterns': 0.20,
            'account_patterns': 0.15,
            'time_patterns': 0.10,
            'model_confidence': 0.10
        }
        
        # Cache for risk factor analysis
        self._risk_factor_cache = {}
        self._cache_timestamp = None
        
        logger.info(f"FraudDetector initialized with risk_threshold={risk_threshold}, "
                   f"high_risk_threshold={high_risk_threshold}, "
                   f"alert_manager={'enabled' if alert_manager else 'disabled'}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained fraud detection model.
        
        Args:
            model_path: Path to the saved model file
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # This would need to be implemented based on the specific model type
        # For now, we'll assume the model is already loaded
        logger.info(f"Model loading from {model_path} - implementation needed")
    
    def score_transaction(self, transaction: Dict[str, Any]) -> float:
        """
        Score a single transaction for fraud risk.
        
        Args:
            transaction: Dictionary containing transaction data
            
        Returns:
            Fraud risk score between 0 and 1
        """
        if self.model is None:
            raise ValueError("No model loaded for fraud detection")
        
        # Convert transaction to DataFrame format expected by model
        transaction_df = self._prepare_transaction_data([transaction])
        
        # Get model prediction probability
        try:
            fraud_probabilities = self.model.predict_proba(transaction_df)
            
            # Handle different probability output formats
            if fraud_probabilities.ndim == 1:
                fraud_score = fraud_probabilities[0]
            else:
                fraud_score = fraud_probabilities[0, 1]  # Probability of fraud class
            
            logger.debug(f"Transaction scored with fraud probability: {fraud_score}")
            
            # Check for alert conditions if AlertManager is available
            if self.alert_manager and fraud_score >= self.risk_threshold:
                try:
                    # Get detailed explanation for alert
                    explanation_data = self.get_fraud_explanation(transaction)
                    
                    # Create alert if conditions are met
                    alert = self.alert_manager.check_alert_conditions(
                        fraud_score=fraud_score,
                        transaction_data=transaction,
                        risk_factors=explanation_data.get('risk_factors', {}),
                        explanation=explanation_data.get('explanation_text', ''),
                        recommendations=explanation_data.get('recommendations', [])
                    )
                    
                    if alert:
                        logger.info(f"Alert created for transaction: {alert.alert_id}")
                        
                except Exception as e:
                    logger.error(f"Error creating alert for transaction: {e}")
            
            return float(fraud_score)
            
        except Exception as e:
            logger.error(f"Error scoring transaction: {e}")
            raise ValueError(f"Failed to score transaction: {e}")
    
    def batch_predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Perform batch prediction on multiple transactions.
        
        Args:
            transactions: DataFrame containing multiple transactions
            
        Returns:
            DataFrame with original data plus fraud scores and predictions
        """
        if self.model is None:
            raise ValueError("No model loaded for fraud detection")
        
        if transactions.empty:
            logger.warning("Empty DataFrame provided for batch prediction")
            return transactions.copy()
        
        logger.info(f"Processing batch prediction for {len(transactions)} transactions")
        
        # Prepare data for model
        prepared_data = self._prepare_transaction_data(transactions.to_dict('records'))
        
        try:
            # Get predictions and probabilities
            predictions = self.model.predict(prepared_data)
            probabilities = self.model.predict_proba(prepared_data)
            
            # Handle different probability output formats
            if probabilities.ndim == 1:
                fraud_scores = probabilities
            else:
                fraud_scores = probabilities[:, 1]
            
            # Create result DataFrame
            result_df = transactions.copy()
            result_df['fraud_score'] = fraud_scores
            result_df['fraud_prediction'] = predictions
            result_df['risk_level'] = self._categorize_risk_level(fraud_scores)
            
            # Add batch processing timestamp
            result_df['processed_at'] = datetime.now().isoformat()
            
            logger.info(f"Batch prediction completed. Found {sum(predictions)} potential fraud cases")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise ValueError(f"Failed to process batch prediction: {e}")
    
    def get_fraud_explanation(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed fraud explanation and risk factor identification.
        
        Args:
            transaction: Dictionary containing transaction data
            
        Returns:
            Dictionary containing fraud explanation and risk factors
        """
        if not self.enable_explanations:
            return {"explanation": "Fraud explanations are disabled"}
        
        # Get fraud score
        fraud_score = self.score_transaction(transaction)
        
        # Analyze risk factors
        risk_factors = self._analyze_transaction_risk_factors(transaction)
        
        # Get model feature importance if available
        feature_importance = self._get_feature_importance_explanation(transaction)
        
        # Generate human-readable explanation
        explanation_text = self._generate_explanation_text(fraud_score, risk_factors)
        
        # Compile comprehensive explanation
        explanation = {
            'fraud_score': fraud_score,
            'risk_level': self._categorize_risk_level([fraud_score])[0],
            'is_fraud_prediction': fraud_score >= self.risk_threshold,
            'confidence': self._calculate_confidence(fraud_score),
            'risk_factors': risk_factors,
            'feature_importance': feature_importance,
            'explanation_text': explanation_text,
            'recommendations': self._generate_recommendations(fraud_score, risk_factors),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"Generated fraud explanation for transaction with score: {fraud_score}")
        
        return explanation
    
    def _prepare_transaction_data(self, transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare transaction data for model input.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            DataFrame formatted for model input
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Ensure required columns exist with default values
        required_columns = {
            'step': 0,
            'type': 'PAYMENT',
            'amount': 0.0,
            'nameOrig': 'C000000000',
            'oldbalanceOrg': 0.0,
            'newbalanceOrig': 0.0,
            'nameDest': 'C000000000',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }
        
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
                logger.warning(f"Missing column '{col}', using default value: {default_val}")
        
        # Apply feature engineering (this would typically use the FeatureEngineering class)
        df = self._apply_feature_engineering(df)
        
        return df
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to transaction data.
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_processed = df.copy()
        
        # Calculate balance changes
        df_processed['balance_change_orig'] = df_processed['newbalanceOrig'] - df_processed['oldbalanceOrg']
        df_processed['balance_change_dest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
        
        # Amount to balance ratios
        df_processed['amount_to_balance_ratio'] = np.where(
            df_processed['oldbalanceOrg'] > 0,
            df_processed['amount'] / df_processed['oldbalanceOrg'],
            0
        )
        
        # Merchant flags
        df_processed['is_merchant_dest'] = df_processed['nameDest'].str.startswith('M')
        df_processed['is_merchant_orig'] = df_processed['nameOrig'].str.startswith('M')
        
        # Large transfer flags (business rule: >200,000)
        df_processed['is_large_transfer'] = (df_processed['amount'] > 200000) & (df_processed['type'] == 'TRANSFER')
        
        # Time-based features
        df_processed['hour_of_day'] = df_processed['step'] % 24
        df_processed['day_of_month'] = df_processed['step'] // 24
        
        # Encode transaction types (simple label encoding for now)
        type_mapping = {'CASH-IN': 0, 'CASH-OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
        df_processed['type_encoded'] = df_processed['type'].map(type_mapping).fillna(3)
        
        return df_processed
    
    def _analyze_transaction_risk_factors(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze risk factors for a single transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Dictionary of identified risk factors
        """
        risk_factors = {
            'high_risk_factors': [],
            'medium_risk_factors': [],
            'low_risk_factors': [],
            'risk_scores': {}
        }
        
        # Transaction type risk
        transaction_type = transaction.get('type', 'PAYMENT')
        type_risk_score = self._get_transaction_type_risk(transaction_type)
        risk_factors['risk_scores']['transaction_type'] = type_risk_score
        
        if type_risk_score > 0.7:
            risk_factors['high_risk_factors'].append(f"High-risk transaction type: {transaction_type}")
        elif type_risk_score > 0.4:
            risk_factors['medium_risk_factors'].append(f"Medium-risk transaction type: {transaction_type}")
        
        # Amount-based risk
        amount = transaction.get('amount', 0)
        amount_risk_score = self._get_amount_risk(amount)
        risk_factors['risk_scores']['amount'] = amount_risk_score
        
        if amount > 200000:
            risk_factors['high_risk_factors'].append(f"Large transaction amount: {amount:,.2f}")
        elif amount > 50000:
            risk_factors['medium_risk_factors'].append(f"Moderate transaction amount: {amount:,.2f}")
        
        # Balance pattern risk
        balance_risk_score = self._get_balance_pattern_risk(transaction)
        risk_factors['risk_scores']['balance_patterns'] = balance_risk_score
        
        old_balance = transaction.get('oldbalanceOrg', 0)
        new_balance = transaction.get('newbalanceOrig', 0)
        
        if old_balance > 0 and new_balance == 0:
            risk_factors['high_risk_factors'].append("Complete balance depletion detected")
        
        if old_balance == 0 and new_balance == 0:
            risk_factors['medium_risk_factors'].append("Zero balance transaction")
        
        # Account pattern risk
        name_dest = transaction.get('nameDest', '')
        name_orig = transaction.get('nameOrig', '')
        
        if not name_dest.startswith('M') and not name_orig.startswith('M'):
            risk_factors['low_risk_factors'].append("Customer-to-customer transaction")
        elif name_dest.startswith('M'):
            risk_factors['low_risk_factors'].append("Transaction to merchant account")
        
        # Time pattern risk (if step is available)
        step = transaction.get('step', 0)
        if step > 0:
            hour = step % 24
            time_risk_score = self._get_time_pattern_risk(hour)
            risk_factors['risk_scores']['time_patterns'] = time_risk_score
            
            if time_risk_score > 0.6:
                risk_factors['medium_risk_factors'].append(f"Transaction at high-risk hour: {hour}")
        
        return risk_factors
    
    def _get_transaction_type_risk(self, transaction_type: str) -> float:
        """Get risk score for transaction type based on historical patterns."""
        # These would typically be learned from historical data
        type_risk_scores = {
            'CASH-OUT': 0.8,
            'TRANSFER': 0.7,
            'CASH-IN': 0.3,
            'DEBIT': 0.2,
            'PAYMENT': 0.1
        }
        return type_risk_scores.get(transaction_type, 0.5)
    
    def _get_amount_risk(self, amount: float) -> float:
        """Get risk score based on transaction amount."""
        if amount > 200000:
            return 0.9
        elif amount > 100000:
            return 0.7
        elif amount > 50000:
            return 0.5
        elif amount > 10000:
            return 0.3
        else:
            return 0.1
    
    def _get_balance_pattern_risk(self, transaction: Dict[str, Any]) -> float:
        """Get risk score based on balance patterns."""
        old_balance = transaction.get('oldbalanceOrg', 0)
        new_balance = transaction.get('newbalanceOrig', 0)
        amount = transaction.get('amount', 0)
        
        # Complete balance depletion
        if old_balance > 0 and new_balance == 0:
            return 0.9
        
        # Zero balance transactions
        if old_balance == 0 and new_balance == 0:
            return 0.6
        
        # Large percentage of balance
        if old_balance > 0 and amount / old_balance > 0.8:
            return 0.7
        
        return 0.2
    
    def _get_time_pattern_risk(self, hour: int) -> float:
        """Get risk score based on time patterns."""
        # These would typically be learned from historical data
        # Assuming higher risk during off-hours
        if hour >= 22 or hour <= 6:  # Night hours
            return 0.7
        elif hour >= 12 and hour <= 14:  # Lunch hours
            return 0.4
        else:
            return 0.2
    
    def _get_feature_importance_explanation(self, transaction: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Get feature importance explanation from the model.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Dictionary of feature importance scores or None if not available
        """
        if self.model is None or not hasattr(self.model, 'get_feature_importance'):
            return None
        
        try:
            feature_importance = self.model.get_feature_importance()
            
            # Filter to top 10 most important features for explanation
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:10])
            
            return top_features
            
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return None
    
    def _generate_explanation_text(self, fraud_score: float, risk_factors: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            fraud_score: Calculated fraud score
            risk_factors: Identified risk factors
            
        Returns:
            Human-readable explanation string
        """
        explanation_parts = []
        
        # Overall assessment
        if fraud_score >= self.high_risk_threshold:
            explanation_parts.append(f"HIGH RISK: This transaction has a {fraud_score:.1%} probability of being fraudulent.")
        elif fraud_score >= self.risk_threshold:
            explanation_parts.append(f"MEDIUM RISK: This transaction has a {fraud_score:.1%} probability of being fraudulent.")
        else:
            explanation_parts.append(f"LOW RISK: This transaction has a {fraud_score:.1%} probability of being fraudulent.")
        
        # High risk factors
        if risk_factors['high_risk_factors']:
            explanation_parts.append("Critical risk factors identified:")
            for factor in risk_factors['high_risk_factors']:
                explanation_parts.append(f"• {factor}")
        
        # Medium risk factors
        if risk_factors['medium_risk_factors']:
            explanation_parts.append("Moderate risk factors identified:")
            for factor in risk_factors['medium_risk_factors']:
                explanation_parts.append(f"• {factor}")
        
        # Low risk factors
        if risk_factors['low_risk_factors'] and fraud_score < self.risk_threshold:
            explanation_parts.append("Low risk indicators:")
            for factor in risk_factors['low_risk_factors'][:2]:  # Limit to 2 for brevity
                explanation_parts.append(f"• {factor}")
        
        return "\n".join(explanation_parts)
    
    def _generate_recommendations(self, fraud_score: float, risk_factors: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on fraud score and risk factors.
        
        Args:
            fraud_score: Calculated fraud score
            risk_factors: Identified risk factors
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if fraud_score >= self.high_risk_threshold:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Block transaction and investigate",
                "Contact customer to verify transaction legitimacy",
                "Review account for other suspicious activities",
                "Consider temporary account restrictions"
            ])
        elif fraud_score >= self.risk_threshold:
            recommendations.extend([
                "Enhanced verification recommended",
                "Request additional authentication from customer",
                "Monitor account for follow-up transactions",
                "Consider manual review by fraud analyst"
            ])
        else:
            recommendations.extend([
                "Transaction appears legitimate",
                "Continue normal processing",
                "Maintain standard monitoring"
            ])
        
        # Add specific recommendations based on risk factors
        if any("Large transaction" in factor for factor in risk_factors['high_risk_factors']):
            recommendations.append("Verify large amount with customer via secure channel")
        
        if any("balance depletion" in factor for factor in risk_factors['high_risk_factors']):
            recommendations.append("Investigate potential account takeover")
        
        return recommendations
    
    def _categorize_risk_level(self, fraud_scores: List[float]) -> List[str]:
        """
        Categorize fraud scores into risk levels.
        
        Args:
            fraud_scores: List of fraud probability scores
            
        Returns:
            List of risk level strings
        """
        risk_levels = []
        for score in fraud_scores:
            if score >= self.high_risk_threshold:
                risk_levels.append("HIGH")
            elif score >= self.risk_threshold:
                risk_levels.append("MEDIUM")
            else:
                risk_levels.append("LOW")
        
        return risk_levels
    
    def _calculate_confidence(self, fraud_score: float) -> float:
        """
        Calculate confidence level for the fraud prediction.
        
        Args:
            fraud_score: Fraud probability score
            
        Returns:
            Confidence score between 0 and 1
        """
        # Confidence is higher when score is closer to 0 or 1 (more decisive)
        distance_from_middle = abs(fraud_score - 0.5)
        confidence = distance_from_middle * 2  # Scale to 0-1 range
        
        return min(confidence, 1.0)
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the fraud detection service.
        
        Returns:
            Dictionary containing service status information
        """
        status = {
            'service_name': 'FraudDetector',
            'status': 'active' if self.model is not None else 'inactive',
            'model_loaded': self.model is not None,
            'risk_threshold': self.risk_threshold,
            'high_risk_threshold': self.high_risk_threshold,
            'explanations_enabled': self.enable_explanations,
            'alert_manager_enabled': self.alert_manager is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add alert statistics if AlertManager is available
        if self.alert_manager:
            alert_stats = self.alert_manager.get_alert_statistics()
            status['alert_statistics'] = alert_stats
        
        if self.model is not None:
            try:
                model_info = self.model.get_metadata() if hasattr(self.model, 'get_metadata') else {}
                status['model_info'] = model_info
            except Exception as e:
                logger.warning(f"Could not get model metadata: {e}")
                status['model_info'] = {'error': str(e)}
        
        return status
    
    def update_thresholds(self, risk_threshold: Optional[float] = None, 
                         high_risk_threshold: Optional[float] = None) -> None:
        """
        Update fraud detection thresholds.
        
        Args:
            risk_threshold: New risk threshold for fraud classification
            high_risk_threshold: New high-risk threshold for alerts
        """
        if risk_threshold is not None:
            if not 0 <= risk_threshold <= 1:
                raise ValueError("Risk threshold must be between 0 and 1")
            self.risk_threshold = risk_threshold
            logger.info(f"Updated risk threshold to {risk_threshold}")
        
        if high_risk_threshold is not None:
            if not 0 <= high_risk_threshold <= 1:
                raise ValueError("High risk threshold must be between 0 and 1")
            self.high_risk_threshold = high_risk_threshold
            logger.info(f"Updated high risk threshold to {high_risk_threshold}")
        
        if (risk_threshold is not None and high_risk_threshold is not None and 
            risk_threshold >= high_risk_threshold):
            logger.warning("Risk threshold should be lower than high risk threshold")
    
    def set_alert_manager(self, alert_manager: 'AlertManager') -> None:
        """
        Set the AlertManager instance for handling fraud alerts.
        
        Args:
            alert_manager: AlertManager instance
        """
        self.alert_manager = alert_manager
        logger.info("AlertManager has been configured for FraudDetector")
    
    def get_alert_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get alert statistics from the AlertManager.
        
        Returns:
            Alert statistics dictionary or None if AlertManager not configured
        """
        if self.alert_manager:
            return self.alert_manager.get_alert_statistics()
        return None