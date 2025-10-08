"""
AlertManager service for managing fraud detection alerts and notifications.

This module provides alert management functionality including threshold-based alerting,
notification delivery (email/SMS), and comprehensive audit logging for high-risk transactions.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty

from src.utils.config import get_setting, get_logger, settings

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(Enum):
    """Alert status tracking."""
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    ACKNOWLEDGED = "ACKNOWLEDGED"


class NotificationType(Enum):
    """Notification delivery methods."""
    EMAIL = "EMAIL"
    SMS = "SMS"
    WEBHOOK = "WEBHOOK"
    LOG_ONLY = "LOG_ONLY"


@dataclass
class AlertThreshold:
    """Configuration for alert thresholds."""
    name: str
    fraud_score_min: float
    fraud_score_max: float = 1.0
    severity: AlertSeverity = AlertSeverity.MEDIUM
    notification_types: List[NotificationType] = None
    cooldown_minutes: int = 5
    enabled: bool = True
    
    def __post_init__(self):
        if self.notification_types is None:
            self.notification_types = [NotificationType.EMAIL, NotificationType.LOG_ONLY]


@dataclass
class FraudAlert:
    """Fraud alert data structure."""
    alert_id: str
    transaction_id: Optional[str]
    fraud_score: float
    risk_level: str
    severity: AlertSeverity
    transaction_data: Dict[str, Any]
    risk_factors: Dict[str, Any]
    explanation: str
    recommendations: List[str]
    created_at: datetime
    threshold_name: str
    status: AlertStatus = AlertStatus.PENDING
    notification_attempts: int = 0
    last_notification_attempt: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        if self.last_notification_attempt:
            data['last_notification_attempt'] = self.last_notification_attempt.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        # Convert enums to strings
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        return data


@dataclass
class NotificationConfig:
    """Configuration for notification delivery."""
    # Email configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_recipients: List[str] = None
    
    # SMS configuration (placeholder for future implementation)
    sms_provider: str = ""
    sms_api_key: str = ""
    sms_recipients: List[str] = None
    
    # Webhook configuration
    webhook_urls: List[str] = None
    webhook_timeout: int = 30
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []
        if self.sms_recipients is None:
            self.sms_recipients = []
        if self.webhook_urls is None:
            self.webhook_urls = []


class AlertManager:
    """
    Main alert management service for fraud detection alerts.
    
    Manages alert thresholds, notification delivery, and audit logging
    for high-risk fraud detection events.
    """
    
    def __init__(self, 
                 notification_config: Optional[NotificationConfig] = None,
                 audit_log_path: Optional[str] = None,
                 max_notification_retries: int = 3,
                 enable_async_notifications: bool = True):
        """
        Initialize the AlertManager.
        
        Args:
            notification_config: Configuration for notification delivery
            audit_log_path: Path to audit log file
            max_notification_retries: Maximum retry attempts for failed notifications
            enable_async_notifications: Whether to send notifications asynchronously
        """
        self.notification_config = notification_config or NotificationConfig()
        self.max_notification_retries = max_notification_retries
        self.enable_async_notifications = enable_async_notifications
        
        # Alert thresholds configuration
        self.alert_thresholds: List[AlertThreshold] = []
        self._setup_default_thresholds()
        
        # Alert storage and tracking
        self.active_alerts: Dict[str, FraudAlert] = {}
        self.alert_history: List[FraudAlert] = []
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Audit logging
        if audit_log_path:
            self.audit_log_path = Path(audit_log_path)
        else:
            self.audit_log_path = settings.logs_dir / "fraud_alerts_audit.log"
        self._setup_audit_logging()
        
        # Notification queue and processing
        self.notification_queue: Queue = Queue()
        self.notification_executor = ThreadPoolExecutor(max_workers=3)
        self._notification_thread_running = False
        
        # Statistics tracking
        self.stats = {
            'total_alerts': 0,
            'alerts_by_severity': {severity.value: 0 for severity in AlertSeverity},
            'notifications_sent': 0,
            'notifications_failed': 0,
            'last_alert_time': None
        }
        
        logger.info("AlertManager initialized successfully")
    
    def _setup_default_thresholds(self) -> None:
        """Setup default alert thresholds."""
        default_thresholds = [
            AlertThreshold(
                name="critical_fraud",
                fraud_score_min=0.9,
                severity=AlertSeverity.CRITICAL,
                notification_types=[NotificationType.EMAIL, NotificationType.SMS, NotificationType.LOG_ONLY],
                cooldown_minutes=1
            ),
            AlertThreshold(
                name="high_risk_fraud",
                fraud_score_min=0.8,
                fraud_score_max=0.9,
                severity=AlertSeverity.HIGH,
                notification_types=[NotificationType.EMAIL, NotificationType.LOG_ONLY],
                cooldown_minutes=5
            ),
            AlertThreshold(
                name="medium_risk_fraud",
                fraud_score_min=0.6,
                fraud_score_max=0.8,
                severity=AlertSeverity.MEDIUM,
                notification_types=[NotificationType.LOG_ONLY],
                cooldown_minutes=15
            )
        ]
        
        self.alert_thresholds = default_thresholds
        logger.info(f"Configured {len(default_thresholds)} default alert thresholds")
    
    def _setup_audit_logging(self) -> None:
        """Setup audit logging for alerts."""
        # Ensure audit log directory exists
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create audit logger
        self.audit_logger = get_logger(f"{__name__}.audit")
        logger.info(f"Audit logging configured: {self.audit_log_path}")
    
    def add_threshold(self, threshold: AlertThreshold) -> None:
        """
        Add a new alert threshold.
        
        Args:
            threshold: AlertThreshold configuration
        """
        # Validate threshold
        if not 0 <= threshold.fraud_score_min <= 1:
            raise ValueError("fraud_score_min must be between 0 and 1")
        if not 0 <= threshold.fraud_score_max <= 1:
            raise ValueError("fraud_score_max must be between 0 and 1")
        if threshold.fraud_score_min >= threshold.fraud_score_max:
            raise ValueError("fraud_score_min must be less than fraud_score_max")
        
        # Check for duplicate names
        existing_names = [t.name for t in self.alert_thresholds]
        if threshold.name in existing_names:
            raise ValueError(f"Threshold with name '{threshold.name}' already exists")
        
        self.alert_thresholds.append(threshold)
        logger.info(f"Added alert threshold: {threshold.name}")
    
    def remove_threshold(self, threshold_name: str) -> bool:
        """
        Remove an alert threshold by name.
        
        Args:
            threshold_name: Name of threshold to remove
            
        Returns:
            True if threshold was removed, False if not found
        """
        for i, threshold in enumerate(self.alert_thresholds):
            if threshold.name == threshold_name:
                del self.alert_thresholds[i]
                logger.info(f"Removed alert threshold: {threshold_name}")
                return True
        
        logger.warning(f"Threshold not found: {threshold_name}")
        return False
    
    def update_threshold(self, threshold_name: str, **kwargs) -> bool:
        """
        Update an existing alert threshold.
        
        Args:
            threshold_name: Name of threshold to update
            **kwargs: Threshold attributes to update
            
        Returns:
            True if threshold was updated, False if not found
        """
        for threshold in self.alert_thresholds:
            if threshold.name == threshold_name:
                for key, value in kwargs.items():
                    if hasattr(threshold, key):
                        setattr(threshold, key, value)
                        logger.info(f"Updated threshold {threshold_name}.{key} = {value}")
                    else:
                        logger.warning(f"Invalid threshold attribute: {key}")
                return True
        
        logger.warning(f"Threshold not found: {threshold_name}")
        return False
    
    def check_alert_conditions(self, 
                             fraud_score: float,
                             transaction_data: Dict[str, Any],
                             risk_factors: Optional[Dict[str, Any]] = None,
                             explanation: Optional[str] = None,
                             recommendations: Optional[List[str]] = None) -> Optional[FraudAlert]:
        """
        Check if fraud score meets alert conditions and create alert if needed.
        
        Args:
            fraud_score: Fraud probability score (0-1)
            transaction_data: Transaction data dictionary
            risk_factors: Risk factors analysis
            explanation: Fraud explanation text
            recommendations: Recommended actions
            
        Returns:
            FraudAlert if conditions are met, None otherwise
        """
        # Find matching threshold
        matching_threshold = None
        for threshold in self.alert_thresholds:
            if (threshold.enabled and 
                threshold.fraud_score_min <= fraud_score < threshold.fraud_score_max):
                matching_threshold = threshold
                break
        
        if not matching_threshold:
            return None
        
        # Check cooldown period
        cooldown_key = f"{matching_threshold.name}_{transaction_data.get('nameOrig', 'unknown')}"
        if self._is_in_cooldown(cooldown_key, matching_threshold.cooldown_minutes):
            logger.debug(f"Alert suppressed due to cooldown: {cooldown_key}")
            return None
        
        # Create alert
        alert = self._create_alert(
            fraud_score=fraud_score,
            transaction_data=transaction_data,
            threshold=matching_threshold,
            risk_factors=risk_factors or {},
            explanation=explanation or "High fraud risk detected",
            recommendations=recommendations or []
        )
        
        # Update cooldown tracker
        self.cooldown_tracker[cooldown_key] = datetime.now()
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Update statistics
        self._update_stats(alert)
        
        # Log audit entry
        self._log_audit_entry("ALERT_CREATED", alert)
        
        # Queue notification
        if self.enable_async_notifications:
            self._queue_notification(alert)
        else:
            self._send_notification_sync(alert)
        
        logger.info(f"Alert created: {alert.alert_id} (severity: {alert.severity.value})")
        
        return alert
    
    def _is_in_cooldown(self, cooldown_key: str, cooldown_minutes: int) -> bool:
        """Check if alert is in cooldown period."""
        if cooldown_key not in self.cooldown_tracker:
            return False
        
        last_alert_time = self.cooldown_tracker[cooldown_key]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - last_alert_time < cooldown_period
    
    def _create_alert(self,
                     fraud_score: float,
                     transaction_data: Dict[str, Any],
                     threshold: AlertThreshold,
                     risk_factors: Dict[str, Any],
                     explanation: str,
                     recommendations: List[str]) -> FraudAlert:
        """Create a new fraud alert."""
        alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(transaction_data)) % 10000:04d}"
        
        # Determine risk level
        if fraud_score >= 0.9:
            risk_level = "CRITICAL"
        elif fraud_score >= 0.8:
            risk_level = "HIGH"
        elif fraud_score >= 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        alert = FraudAlert(
            alert_id=alert_id,
            transaction_id=transaction_data.get('transaction_id'),
            fraud_score=fraud_score,
            risk_level=risk_level,
            severity=threshold.severity,
            transaction_data=transaction_data,
            risk_factors=risk_factors,
            explanation=explanation,
            recommendations=recommendations,
            created_at=datetime.now(),
            threshold_name=threshold.name
        )
        
        return alert
    
    def _queue_notification(self, alert: FraudAlert) -> None:
        """Queue alert notification for asynchronous processing."""
        self.notification_queue.put(alert)
        
        # Start notification processing thread if not running
        if not self._notification_thread_running:
            self._start_notification_thread()
    
    def _start_notification_thread(self) -> None:
        """Start the notification processing thread."""
        def process_notifications():
            self._notification_thread_running = True
            logger.info("Notification processing thread started")
            
            while True:
                try:
                    # Get alert from queue (blocking with timeout)
                    alert = self.notification_queue.get(timeout=5)
                    
                    # Process notification
                    self._send_notification_sync(alert)
                    
                    # Mark task as done
                    self.notification_queue.task_done()
                    
                except Empty:
                    # No alerts in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Error processing notification: {e}")
        
        # Start thread
        notification_thread = threading.Thread(target=process_notifications, daemon=True)
        notification_thread.start()
    
    def _send_notification_sync(self, alert: FraudAlert) -> None:
        """Send notification synchronously."""
        threshold = self._get_threshold_by_name(alert.threshold_name)
        if not threshold:
            logger.error(f"Threshold not found for alert: {alert.threshold_name}")
            return
        
        success_count = 0
        total_attempts = 0
        
        for notification_type in threshold.notification_types:
            total_attempts += 1
            
            try:
                if notification_type == NotificationType.EMAIL:
                    self._send_email_notification(alert)
                elif notification_type == NotificationType.SMS:
                    self._send_sms_notification(alert)
                elif notification_type == NotificationType.WEBHOOK:
                    self._send_webhook_notification(alert)
                elif notification_type == NotificationType.LOG_ONLY:
                    self._send_log_notification(alert)
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to send {notification_type.value} notification for alert {alert.alert_id}: {e}")
        
        # Update alert status
        if success_count > 0:
            alert.status = AlertStatus.SENT
            self.stats['notifications_sent'] += success_count
        else:
            alert.status = AlertStatus.FAILED
            self.stats['notifications_failed'] += total_attempts
        
        alert.notification_attempts += 1
        alert.last_notification_attempt = datetime.now()
        
        # Log audit entry
        self._log_audit_entry("NOTIFICATION_SENT" if success_count > 0 else "NOTIFICATION_FAILED", alert)
    
    def _send_email_notification(self, alert: FraudAlert) -> None:
        """Send email notification for alert."""
        if not self.notification_config.email_recipients:
            logger.warning("No email recipients configured")
            return
        
        # Create email message
        subject = f"Fraud Alert - {alert.severity.value} Risk Transaction Detected"
        body = self._create_email_body(alert)
        
        msg = MIMEMultipart()
        msg['From'] = self.notification_config.email_from
        msg['To'] = ", ".join(self.notification_config.email_recipients)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP(self.notification_config.smtp_server, self.notification_config.smtp_port) as server:
            server.starttls(context=context)
            if self.notification_config.smtp_username and self.notification_config.smtp_password:
                server.login(self.notification_config.smtp_username, self.notification_config.smtp_password)
            
            text = msg.as_string()
            server.sendmail(self.notification_config.email_from, self.notification_config.email_recipients, text)
        
        logger.info(f"Email notification sent for alert: {alert.alert_id}")
    
    def _send_sms_notification(self, alert: FraudAlert) -> None:
        """Send SMS notification for alert (placeholder implementation)."""
        # This would integrate with SMS providers like Twilio, AWS SNS, etc.
        logger.info(f"SMS notification would be sent for alert: {alert.alert_id}")
        # Placeholder - actual implementation would depend on SMS provider
        pass
    
    def _send_webhook_notification(self, alert: FraudAlert) -> None:
        """Send webhook notification for alert (placeholder implementation)."""
        # This would send HTTP POST requests to configured webhook URLs
        logger.info(f"Webhook notification would be sent for alert: {alert.alert_id}")
        # Placeholder - actual implementation would use requests library
        pass
    
    def _send_log_notification(self, alert: FraudAlert) -> None:
        """Send log-only notification for alert."""
        log_message = (
            f"FRAUD ALERT: {alert.severity.value} | "
            f"Score: {alert.fraud_score:.3f} | "
            f"Transaction: {alert.transaction_data.get('type', 'UNKNOWN')} "
            f"${alert.transaction_data.get('amount', 0):,.2f} | "
            f"Account: {alert.transaction_data.get('nameOrig', 'UNKNOWN')}"
        )
        
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(log_message)
        elif alert.severity == AlertSeverity.HIGH:
            logger.error(log_message)
        elif alert.severity == AlertSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _create_email_body(self, alert: FraudAlert) -> str:
        """Create HTML email body for alert notification."""
        severity_colors = {
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.HIGH: "#fd7e14", 
            AlertSeverity.MEDIUM: "#ffc107",
            AlertSeverity.LOW: "#28a745"
        }
        
        color = severity_colors.get(alert.severity, "#6c757d")
        
        html_body = f"""
        <html>
        <body>
            <h2 style="color: {color};">Fraud Detection Alert - {alert.severity.value} Risk</h2>
            
            <h3>Alert Details</h3>
            <ul>
                <li><strong>Alert ID:</strong> {alert.alert_id}</li>
                <li><strong>Fraud Score:</strong> {alert.fraud_score:.1%}</li>
                <li><strong>Risk Level:</strong> {alert.risk_level}</li>
                <li><strong>Created:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>
            
            <h3>Transaction Information</h3>
            <ul>
                <li><strong>Type:</strong> {alert.transaction_data.get('type', 'N/A')}</li>
                <li><strong>Amount:</strong> ${alert.transaction_data.get('amount', 0):,.2f}</li>
                <li><strong>Origin Account:</strong> {alert.transaction_data.get('nameOrig', 'N/A')}</li>
                <li><strong>Destination Account:</strong> {alert.transaction_data.get('nameDest', 'N/A')}</li>
            </ul>
            
            <h3>Risk Analysis</h3>
            <p>{alert.explanation}</p>
            
            <h3>Recommended Actions</h3>
            <ul>
        """
        
        for recommendation in alert.recommendations:
            html_body += f"<li>{recommendation}</li>"
        
        html_body += """
            </ul>
            
            <hr>
            <p><small>This is an automated fraud detection alert. Please investigate immediately.</small></p>
        </body>
        </html>
        """
        
        return html_body
    
    def _get_threshold_by_name(self, name: str) -> Optional[AlertThreshold]:
        """Get threshold configuration by name."""
        for threshold in self.alert_thresholds:
            if threshold.name == name:
                return threshold
        return None
    
    def _update_stats(self, alert: FraudAlert) -> None:
        """Update alert statistics."""
        self.stats['total_alerts'] += 1
        self.stats['alerts_by_severity'][alert.severity.value] += 1
        self.stats['last_alert_time'] = alert.created_at.isoformat()
    
    def _log_audit_entry(self, action: str, alert: FraudAlert, additional_data: Optional[Dict] = None) -> None:
        """Log audit entry for alert action."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'alert_id': alert.alert_id,
            'fraud_score': alert.fraud_score,
            'severity': alert.severity.value,
            'transaction_data': alert.transaction_data,
            'additional_data': additional_data or {}
        }
        
        # Write to audit log file
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
        
        # Also log to application logger
        self.audit_logger.info(f"{action}: {alert.alert_id}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if alert was acknowledged, False if not found
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            
            self._log_audit_entry("ALERT_ACKNOWLEDGED", alert, {'acknowledged_by': acknowledged_by})
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            
            return True
        
        logger.warning(f"Alert not found for acknowledgment: {alert_id}")
        return False
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[FraudAlert]:
        """
        Get list of active (unacknowledged) alerts.
        
        Args:
            severity_filter: Optional severity filter
            
        Returns:
            List of active alerts
        """
        active_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.status != AlertStatus.ACKNOWLEDGED
        ]
        
        if severity_filter:
            active_alerts = [
                alert for alert in active_alerts
                if alert.severity == severity_filter
            ]
        
        return sorted(active_alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_history(self, 
                         hours_back: int = 24,
                         severity_filter: Optional[AlertSeverity] = None) -> List[FraudAlert]:
        """
        Get alert history for specified time period.
        
        Args:
            hours_back: Number of hours to look back
            severity_filter: Optional severity filter
            
        Returns:
            List of historical alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_alerts = [
            alert for alert in self.alert_history
            if alert.created_at >= cutoff_time
        ]
        
        if severity_filter:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.severity == severity_filter
            ]
        
        return sorted(filtered_alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics and metrics.
        
        Returns:
            Dictionary containing alert statistics
        """
        active_count = len(self.get_active_alerts())
        
        stats = self.stats.copy()
        stats.update({
            'active_alerts': active_count,
            'acknowledged_alerts': len([a for a in self.active_alerts.values() if a.status == AlertStatus.ACKNOWLEDGED]),
            'failed_alerts': len([a for a in self.active_alerts.values() if a.status == AlertStatus.FAILED]),
            'thresholds_configured': len(self.alert_thresholds),
            'notification_success_rate': (
                self.stats['notifications_sent'] / 
                (self.stats['notifications_sent'] + self.stats['notifications_failed'])
                if (self.stats['notifications_sent'] + self.stats['notifications_failed']) > 0 else 0
            )
        })
        
        return stats
    
    def update_notification_config(self, config: NotificationConfig) -> None:
        """
        Update notification configuration.
        
        Args:
            config: New notification configuration
        """
        self.notification_config = config
        logger.info("Notification configuration updated")
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current AlertManager configuration.
        
        Returns:
            Dictionary containing current configuration
        """
        return {
            'thresholds': [asdict(threshold) for threshold in self.alert_thresholds],
            'notification_config': {
                'smtp_server': self.notification_config.smtp_server,
                'smtp_port': self.notification_config.smtp_port,
                'email_recipients_count': len(self.notification_config.email_recipients),
                'sms_recipients_count': len(self.notification_config.sms_recipients),
                'webhook_urls_count': len(self.notification_config.webhook_urls)
            },
            'settings': {
                'max_notification_retries': self.max_notification_retries,
                'enable_async_notifications': self.enable_async_notifications,
                'audit_log_path': str(self.audit_log_path)
            }
        }
    
    def cleanup_old_alerts(self, days_to_keep: int = 30) -> int:
        """
        Clean up old alerts from memory and optionally archive them.
        
        Args:
            days_to_keep: Number of days of alerts to keep in memory
            
        Returns:
            Number of alerts cleaned up
        """
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        # Remove old alerts from active alerts
        old_alert_ids = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.created_at < cutoff_time
        ]
        
        for alert_id in old_alert_ids:
            del self.active_alerts[alert_id]
        
        # Remove old alerts from history
        original_count = len(self.alert_history)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.created_at >= cutoff_time
        ]
        
        cleaned_count = original_count - len(self.alert_history) + len(old_alert_ids)
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old alerts (older than {days_to_keep} days)")
        
        return cleaned_count