"""
Model monitoring service for tracking performance and detecting drift.

This module provides comprehensive monitoring capabilities for fraud detection models,
including performance tracking over time, data drift detection, and model performance drift detection.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import warnings

from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from src.models.base_model import FraudModelInterface
from src.utils.config import get_setting, get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    timestamp: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_predictions: int
    fraud_rate: float
    model_version: Optional[str] = None
    data_source: Optional[str] = None


@dataclass
class DriftMetrics:
    """Data class for storing drift detection metrics."""
    timestamp: str
    feature_name: str
    drift_score: float
    p_value: float
    is_drift_detected: bool
    drift_method: str
    reference_period: str
    current_period: str
    threshold: float


@dataclass
class ModelHealthMetrics:
    """Data class for storing overall model health metrics."""
    timestamp: str
    model_version: str
    health_score: float
    performance_trend: str  # 'improving', 'stable', 'degrading'
    drift_alerts: int
    prediction_volume: int
    avg_response_time: float
    error_rate: float


class ModelMonitor:
    """
    Comprehensive model monitoring service.
    
    Tracks model performance over time, detects data and performance drift,
    and provides alerting capabilities for model degradation.
    """
    
    def __init__(self,
                 model: Optional[FraudModelInterface] = None,
                 monitoring_window_days: int = 30,
                 drift_detection_threshold: float = 0.05,
                 performance_degradation_threshold: float = 0.1,
                 min_samples_for_drift: int = 100,
                 storage_path: Optional[str] = None):
        """
        Initialize the ModelMonitor.
        
        Args:
            model: The fraud detection model to monitor
            monitoring_window_days: Number of days to keep in monitoring window
            drift_detection_threshold: P-value threshold for drift detection
            performance_degradation_threshold: Threshold for performance degradation alerts
            min_samples_for_drift: Minimum samples required for drift detection
            storage_path: Path to store monitoring data
        """
        self.model = model
        self.monitoring_window_days = monitoring_window_days
        self.drift_threshold = drift_detection_threshold
        self.performance_threshold = performance_degradation_threshold
        self.min_samples_for_drift = min_samples_for_drift
        
        # Set up storage
        if storage_path is None:
            storage_path = Path("monitoring_data")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize monitoring data structures
        self.performance_history: List[PerformanceMetrics] = []
        self.drift_history: List[DriftMetrics] = []
        self.health_history: List[ModelHealthMetrics] = []
        
        # Reference data for drift detection
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_labels: Optional[pd.Series] = None
        self.reference_timestamp: Optional[str] = None
        
        # Performance tracking
        self.baseline_performance: Optional[PerformanceMetrics] = None
        self.recent_predictions = deque(maxlen=10000)  # Store recent predictions for analysis
        
        # Load existing monitoring data
        self._load_monitoring_data()
        
        logger.info(f"ModelMonitor initialized with {len(self.performance_history)} "
                   f"historical performance records")
    
    def set_reference_data(self, X_reference: pd.DataFrame, y_reference: pd.Series) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            X_reference: Reference feature data (typically training data)
            y_reference: Reference labels
        """
        self.reference_data = X_reference.copy()
        self.reference_labels = y_reference.copy()
        self.reference_timestamp = datetime.now().isoformat()
        
        # Calculate baseline performance if model is available
        if self.model is not None:
            try:
                y_pred = self.model.predict(X_reference)
                y_pred_proba = self.model.predict_proba(X_reference)
                
                # Handle different probability output formats
                if y_pred_proba.ndim == 1:
                    y_pred_proba_positive = y_pred_proba
                else:
                    y_pred_proba_positive = y_pred_proba[:, 1]
                
                self.baseline_performance = self._calculate_performance_metrics(
                    y_reference, y_pred, y_pred_proba_positive,
                    timestamp=self.reference_timestamp,
                    data_source="reference_data"
                )
                
                logger.info(f"Baseline performance established: F1={self.baseline_performance.f1_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error calculating baseline performance: {e}")
        
        # Save reference data
        self._save_reference_data()
        logger.info(f"Reference data set with {len(X_reference)} samples")
    
    def log_predictions(self, 
                       X_data: pd.DataFrame, 
                       y_true: Optional[pd.Series] = None,
                       y_pred: Optional[np.ndarray] = None,
                       y_pred_proba: Optional[np.ndarray] = None,
                       prediction_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log predictions for monitoring and drift detection.
        
        Args:
            X_data: Feature data for predictions
            y_true: True labels (if available)
            y_pred: Predicted labels (if not provided, will use model to predict)
            y_pred_proba: Predicted probabilities (if not provided, will use model)
            prediction_metadata: Additional metadata about predictions
        """
        timestamp = datetime.now().isoformat()
        
        # Get predictions if not provided
        if self.model is not None and (y_pred is None or y_pred_proba is None):
            try:
                if y_pred is None:
                    y_pred = self.model.predict(X_data)
                if y_pred_proba is None:
                    y_pred_proba = self.model.predict_proba(X_data)
            except Exception as e:
                logger.error(f"Error getting model predictions: {e}")
                return
        
        # Store prediction data for analysis
        prediction_record = {
            'timestamp': timestamp,
            'features': X_data.copy(),
            'y_true': y_true.copy() if y_true is not None else None,
            'y_pred': y_pred.copy() if y_pred is not None else None,
            'y_pred_proba': y_pred_proba.copy() if y_pred_proba is not None else None,
            'metadata': prediction_metadata or {}
        }
        
        self.recent_predictions.append(prediction_record)
        
        # Calculate performance metrics if true labels are available
        if y_true is not None and y_pred is not None:
            # Handle different probability output formats
            if y_pred_proba is not None:
                if y_pred_proba.ndim == 1:
                    y_pred_proba_positive = y_pred_proba
                else:
                    y_pred_proba_positive = y_pred_proba[:, 1]
            else:
                y_pred_proba_positive = None
            
            performance_metrics = self._calculate_performance_metrics(
                y_true, y_pred, y_pred_proba_positive, timestamp
            )
            
            self.performance_history.append(performance_metrics)
            
            # Check for performance degradation
            self._check_performance_degradation(performance_metrics)
        
        # Perform drift detection if we have enough data
        if len(self.recent_predictions) >= self.min_samples_for_drift:
            self._perform_drift_detection(X_data, timestamp)
        
        # Update model health metrics
        self._update_model_health(timestamp)
        
        # Cleanup old data
        self._cleanup_old_data()
        
        # Save monitoring data periodically
        if len(self.performance_history) % 100 == 0:  # Save every 100 predictions
            self._save_monitoring_data()
        
        logger.debug(f"Logged {len(X_data)} predictions at {timestamp}")
    
    def _calculate_performance_metrics(self,
                                     y_true: pd.Series,
                                     y_pred: np.ndarray,
                                     y_pred_proba: Optional[np.ndarray] = None,
                                     timestamp: Optional[str] = None,
                                     data_source: Optional[str] = None) -> PerformanceMetrics:
        """Calculate performance metrics for given predictions."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate AUC-ROC if probabilities are available
        auc_roc = None
        if y_pred_proba is not None:
            try:
                auc_roc = roc_auc_score(y_true, y_pred_proba)
            except ValueError as e:
                logger.warning(f"Could not calculate AUC-ROC: {e}")
        
        # Calculate confusion matrix components
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate fraud rate
        fraud_rate = y_true.sum() / len(y_true)
        
        return PerformanceMetrics(
            timestamp=timestamp,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            total_predictions=len(y_true),
            fraud_rate=fraud_rate,
            data_source=data_source
        )
    
    def _perform_drift_detection(self, current_data: pd.DataFrame, timestamp: str) -> None:
        """
        Perform drift detection comparing current data to reference data.
        
        Args:
            current_data: Current batch of feature data
            timestamp: Timestamp for the drift detection
        """
        if self.reference_data is None:
            logger.warning("No reference data available for drift detection")
            return
        
        # Get recent data for comparison
        recent_data = self._get_recent_feature_data()
        if recent_data is None or len(recent_data) < self.min_samples_for_drift:
            logger.debug("Insufficient recent data for drift detection")
            return
        
        # Perform drift detection for each feature
        for feature in self.reference_data.columns:
            if feature in recent_data.columns:
                drift_result = self._detect_feature_drift(
                    self.reference_data[feature],
                    recent_data[feature],
                    feature,
                    timestamp
                )
                
                if drift_result is not None:
                    self.drift_history.append(drift_result)
                    
                    if drift_result.is_drift_detected:
                        logger.warning(f"Drift detected in feature '{feature}': "
                                     f"p-value={drift_result.p_value:.4f}")
    
    def _detect_feature_drift(self,
                            reference_data: pd.Series,
                            current_data: pd.Series,
                            feature_name: str,
                            timestamp: str) -> Optional[DriftMetrics]:
        """
        Detect drift for a single feature using Kolmogorov-Smirnov test.
        
        Args:
            reference_data: Reference feature values
            current_data: Current feature values
            feature_name: Name of the feature
            timestamp: Timestamp for the detection
            
        Returns:
            DriftMetrics object or None if detection failed
        """
        try:
            # Remove NaN values
            ref_clean = reference_data.dropna()
            curr_clean = current_data.dropna()
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                logger.warning(f"No valid data for drift detection in feature '{feature_name}'")
                return None
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(ref_clean, curr_clean)
            
            # Determine if drift is detected
            is_drift_detected = p_value < self.drift_threshold
            
            return DriftMetrics(
                timestamp=timestamp,
                feature_name=feature_name,
                drift_score=ks_statistic,
                p_value=p_value,
                is_drift_detected=is_drift_detected,
                drift_method="kolmogorov_smirnov",
                reference_period=self.reference_timestamp or "unknown",
                current_period=timestamp,
                threshold=self.drift_threshold
            )
            
        except Exception as e:
            logger.error(f"Error detecting drift for feature '{feature_name}': {e}")
            return None
    
    def _get_recent_feature_data(self, hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Get recent feature data from the prediction log.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            DataFrame with recent feature data or None
        """
        if not self.recent_predictions:
            return None
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_features = []
        
        for prediction in self.recent_predictions:
            pred_time = datetime.fromisoformat(prediction['timestamp'])
            if pred_time >= cutoff_time:
                recent_features.append(prediction['features'])
        
        if not recent_features:
            return None
        
        # Concatenate all recent feature data
        try:
            combined_data = pd.concat(recent_features, ignore_index=True)
            return combined_data
        except Exception as e:
            logger.error(f"Error combining recent feature data: {e}")
            return None
    
    def _check_performance_degradation(self, current_metrics: PerformanceMetrics) -> None:
        """
        Check if model performance has degraded significantly.
        
        Args:
            current_metrics: Current performance metrics
        """
        if self.baseline_performance is None:
            return
        
        # Calculate performance degradation
        f1_degradation = self.baseline_performance.f1_score - current_metrics.f1_score
        precision_degradation = self.baseline_performance.precision - current_metrics.precision
        recall_degradation = self.baseline_performance.recall - current_metrics.recall
        
        # Check if degradation exceeds threshold
        significant_degradation = (
            f1_degradation > self.performance_threshold or
            precision_degradation > self.performance_threshold or
            recall_degradation > self.performance_threshold
        )
        
        if significant_degradation:
            logger.warning(f"Significant performance degradation detected: "
                         f"F1 drop: {f1_degradation:.3f}, "
                         f"Precision drop: {precision_degradation:.3f}, "
                         f"Recall drop: {recall_degradation:.3f}")
            
            # Could trigger alerts here
            self._trigger_performance_alert(current_metrics, {
                'f1_degradation': f1_degradation,
                'precision_degradation': precision_degradation,
                'recall_degradation': recall_degradation
            })
    
    def _trigger_performance_alert(self, 
                                 current_metrics: PerformanceMetrics,
                                 degradation_info: Dict[str, float]) -> None:
        """
        Trigger performance degradation alert.
        
        Args:
            current_metrics: Current performance metrics
            degradation_info: Information about performance degradation
        """
        alert_data = {
            'alert_type': 'performance_degradation',
            'timestamp': current_metrics.timestamp,
            'current_f1': current_metrics.f1_score,
            'baseline_f1': self.baseline_performance.f1_score if self.baseline_performance else None,
            'degradation_info': degradation_info,
            'severity': 'high' if max(degradation_info.values()) > 0.2 else 'medium'
        }
        
        # Log the alert
        logger.error(f"PERFORMANCE ALERT: {alert_data}")
        
        # Save alert to file
        alert_file = self.storage_path / "performance_alerts.jsonl"
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert_data) + '\n')
    
    def _update_model_health(self, timestamp: str) -> None:
        """
        Update overall model health metrics.
        
        Args:
            timestamp: Current timestamp
        """
        # Calculate health score based on recent performance and drift
        health_score = self._calculate_health_score()
        
        # Determine performance trend
        performance_trend = self._calculate_performance_trend()
        
        # Count recent drift alerts
        recent_drift_alerts = self._count_recent_drift_alerts(hours=24)
        
        # Calculate prediction volume and response time
        prediction_volume = len([p for p in self.recent_predictions 
                               if datetime.fromisoformat(p['timestamp']) > 
                               datetime.now() - timedelta(hours=1)])
        
        # Estimate average response time (placeholder - would need actual timing data)
        avg_response_time = 0.1  # seconds
        
        # Calculate error rate (placeholder - would need actual error tracking)
        error_rate = 0.0
        
        health_metrics = ModelHealthMetrics(
            timestamp=timestamp,
            model_version=getattr(self.model, 'model_version', 'unknown') if self.model else 'unknown',
            health_score=health_score,
            performance_trend=performance_trend,
            drift_alerts=recent_drift_alerts,
            prediction_volume=prediction_volume,
            avg_response_time=avg_response_time,
            error_rate=error_rate
        )
        
        self.health_history.append(health_metrics)
        
        # Log health status
        if health_score < 0.7:
            logger.warning(f"Model health score is low: {health_score:.3f}")
        
        logger.debug(f"Model health updated: score={health_score:.3f}, "
                    f"trend={performance_trend}, drift_alerts={recent_drift_alerts}")
    
    def _calculate_health_score(self) -> float:
        """
        Calculate overall model health score (0-1).
        
        Returns:
            Health score between 0 and 1
        """
        if not self.performance_history:
            return 1.0
        
        # Get recent performance metrics
        recent_performance = self.performance_history[-10:]  # Last 10 records
        
        # Calculate average F1 score
        avg_f1 = np.mean([p.f1_score for p in recent_performance])
        
        # Penalize for drift alerts
        recent_drift_count = self._count_recent_drift_alerts(hours=24)
        drift_penalty = min(recent_drift_count * 0.1, 0.3)  # Max 30% penalty
        
        # Penalize for performance degradation
        degradation_penalty = 0.0
        if self.baseline_performance:
            f1_drop = max(0, self.baseline_performance.f1_score - avg_f1)
            degradation_penalty = min(f1_drop * 2, 0.4)  # Max 40% penalty
        
        # Calculate final health score
        health_score = max(0.0, avg_f1 - drift_penalty - degradation_penalty)
        
        return health_score
    
    def _calculate_performance_trend(self) -> str:
        """
        Calculate performance trend based on recent history.
        
        Returns:
            Trend string: 'improving', 'stable', or 'degrading'
        """
        if len(self.performance_history) < 5:
            return 'stable'
        
        # Get recent F1 scores
        recent_f1_scores = [p.f1_score for p in self.performance_history[-10:]]
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_f1_scores))
        slope, _, _, p_value, _ = stats.linregress(x, recent_f1_scores)
        
        # Determine trend based on slope and significance
        if p_value < 0.05:  # Statistically significant trend
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'degrading'
        
        return 'stable'
    
    def _count_recent_drift_alerts(self, hours: int = 24) -> int:
        """
        Count drift alerts in the recent time window.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Number of drift alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        count = 0
        for drift_metric in self.drift_history:
            drift_time = datetime.fromisoformat(drift_metric.timestamp)
            if drift_time >= cutoff_time and drift_metric.is_drift_detected:
                count += 1
        
        return count
    
    def _cleanup_old_data(self) -> None:
        """Remove old monitoring data beyond the retention window."""
        cutoff_time = datetime.now() - timedelta(days=self.monitoring_window_days)
        
        # Clean performance history
        self.performance_history = [
            p for p in self.performance_history
            if datetime.fromisoformat(p.timestamp) >= cutoff_time
        ]
        
        # Clean drift history
        self.drift_history = [
            d for d in self.drift_history
            if datetime.fromisoformat(d.timestamp) >= cutoff_time
        ]
        
        # Clean health history
        self.health_history = [
            h for h in self.health_history
            if datetime.fromisoformat(h.timestamp) >= cutoff_time
        ]
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive monitoring summary.
        
        Returns:
            Dictionary containing monitoring summary
        """
        summary = {
            'monitoring_status': {
                'total_performance_records': len(self.performance_history),
                'total_drift_records': len(self.drift_history),
                'total_health_records': len(self.health_history),
                'monitoring_window_days': self.monitoring_window_days,
                'reference_data_available': self.reference_data is not None,
                'baseline_performance_available': self.baseline_performance is not None
            }
        }
        
        # Recent performance summary
        if self.performance_history:
            recent_performance = self.performance_history[-10:]
            summary['recent_performance'] = {
                'avg_f1_score': np.mean([p.f1_score for p in recent_performance]),
                'avg_precision': np.mean([p.precision for p in recent_performance]),
                'avg_recall': np.mean([p.recall for p in recent_performance]),
                'total_predictions': sum([p.total_predictions for p in recent_performance]),
                'avg_fraud_rate': np.mean([p.fraud_rate for p in recent_performance])
            }
        
        # Drift summary
        recent_drift_alerts = self._count_recent_drift_alerts(hours=24)
        summary['drift_status'] = {
            'recent_drift_alerts_24h': recent_drift_alerts,
            'total_drift_detections': len([d for d in self.drift_history if d.is_drift_detected]),
            'features_monitored': len(set([d.feature_name for d in self.drift_history]))
        }
        
        # Model health summary
        if self.health_history:
            latest_health = self.health_history[-1]
            summary['model_health'] = {
                'current_health_score': latest_health.health_score,
                'performance_trend': latest_health.performance_trend,
                'prediction_volume_1h': latest_health.prediction_volume,
                'avg_response_time': latest_health.avg_response_time
            }
        
        return summary
    
    def get_performance_trend_data(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance trend data for visualization.
        
        Args:
            days: Number of days to include in trend
            
        Returns:
            Dictionary containing trend data
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Filter performance data
        recent_performance = [
            p for p in self.performance_history
            if datetime.fromisoformat(p.timestamp) >= cutoff_time
        ]
        
        if not recent_performance:
            return {'message': 'No performance data available for the specified period'}
        
        # Extract trend data
        timestamps = [p.timestamp for p in recent_performance]
        f1_scores = [p.f1_score for p in recent_performance]
        precision_scores = [p.precision for p in recent_performance]
        recall_scores = [p.recall for p in recent_performance]
        fraud_rates = [p.fraud_rate for p in recent_performance]
        
        return {
            'timestamps': timestamps,
            'f1_scores': f1_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'fraud_rates': fraud_rates,
            'period_days': days,
            'total_records': len(recent_performance)
        }
    
    def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get drift detection summary for the specified period.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Dictionary containing drift summary
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Filter drift data
        recent_drift = [
            d for d in self.drift_history
            if datetime.fromisoformat(d.timestamp) >= cutoff_time
        ]
        
        if not recent_drift:
            return {'message': 'No drift data available for the specified period'}
        
        # Summarize by feature
        feature_drift_summary = defaultdict(list)
        for drift in recent_drift:
            feature_drift_summary[drift.feature_name].append(drift)
        
        # Create summary
        drift_summary = {}
        for feature, drift_records in feature_drift_summary.items():
            drift_detections = [d for d in drift_records if d.is_drift_detected]
            drift_summary[feature] = {
                'total_checks': len(drift_records),
                'drift_detections': len(drift_detections),
                'drift_rate': len(drift_detections) / len(drift_records) if drift_records else 0,
                'latest_drift_score': drift_records[-1].drift_score if drift_records else None,
                'latest_p_value': drift_records[-1].p_value if drift_records else None
            }
        
        return {
            'period_days': days,
            'total_drift_checks': len(recent_drift),
            'total_drift_detections': len([d for d in recent_drift if d.is_drift_detected]),
            'features_with_drift': len([f for f, s in drift_summary.items() if s['drift_detections'] > 0]),
            'feature_summary': drift_summary
        }
    
    def _save_monitoring_data(self) -> None:
        """Save monitoring data to disk."""
        try:
            # Save performance history
            perf_file = self.storage_path / "performance_history.json"
            with open(perf_file, 'w') as f:
                perf_data = [self._convert_to_json_serializable(asdict(p)) for p in self.performance_history]
                json.dump(perf_data, f, indent=2)
            
            # Save drift history
            drift_file = self.storage_path / "drift_history.json"
            with open(drift_file, 'w') as f:
                drift_data = [self._convert_to_json_serializable(asdict(d)) for d in self.drift_history]
                json.dump(drift_data, f, indent=2)
            
            # Save health history
            health_file = self.storage_path / "health_history.json"
            with open(health_file, 'w') as f:
                health_data = [self._convert_to_json_serializable(asdict(h)) for h in self.health_history]
                json.dump(health_data, f, indent=2)
            
            logger.debug("Monitoring data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy/pandas types to JSON serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _load_monitoring_data(self) -> None:
        """Load existing monitoring data from disk."""
        try:
            # Load performance history
            perf_file = self.storage_path / "performance_history.json"
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    perf_data = json.load(f)
                    self.performance_history = [PerformanceMetrics(**p) for p in perf_data]
            
            # Load drift history
            drift_file = self.storage_path / "drift_history.json"
            if drift_file.exists():
                with open(drift_file, 'r') as f:
                    drift_data = json.load(f)
                    self.drift_history = [DriftMetrics(**d) for d in drift_data]
            
            # Load health history
            health_file = self.storage_path / "health_history.json"
            if health_file.exists():
                with open(health_file, 'r') as f:
                    health_data = json.load(f)
                    self.health_history = [ModelHealthMetrics(**h) for h in health_data]
            
            logger.debug("Monitoring data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
    
    def _save_reference_data(self) -> None:
        """Save reference data to disk."""
        try:
            if self.reference_data is not None:
                ref_file = self.storage_path / "reference_data.pkl"
                with open(ref_file, 'wb') as f:
                    pickle.dump({
                        'features': self.reference_data,
                        'labels': self.reference_labels,
                        'timestamp': self.reference_timestamp
                    }, f)
                
                logger.debug("Reference data saved successfully")
                
        except Exception as e:
            logger.error(f"Error saving reference data: {e}")
    
    def _load_reference_data(self) -> None:
        """Load reference data from disk."""
        try:
            ref_file = self.storage_path / "reference_data.pkl"
            if ref_file.exists():
                with open(ref_file, 'rb') as f:
                    ref_data = pickle.load(f)
                    self.reference_data = ref_data['features']
                    self.reference_labels = ref_data['labels']
                    self.reference_timestamp = ref_data['timestamp']
                
                logger.debug("Reference data loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
    
    def export_monitoring_report(self, output_path: str, days: int = 30) -> None:
        """
        Export comprehensive monitoring report.
        
        Args:
            output_path: Path to save the report
            days: Number of days to include in report
        """
        try:
            report_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'period_days': days,
                    'model_version': getattr(self.model, 'model_version', 'unknown') if self.model else 'unknown'
                },
                'monitoring_summary': self.get_monitoring_summary(),
                'performance_trend': self.get_performance_trend_data(days),
                'drift_summary': self.get_drift_summary(days)
            }
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Monitoring report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting monitoring report: {e}")