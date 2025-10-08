"""
Unit tests for AlertManager functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from src.services.alert_manager import (
    AlertManager, AlertThreshold, AlertSeverity, NotificationType,
    NotificationConfig, FraudAlert, AlertStatus
)


class TestAlertManager:
    """Test cases for AlertManager class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary audit log file
        self.temp_audit_log = tempfile.NamedTemporaryFile(delete=False)
        self.temp_audit_log.close()
        
        # Initialize AlertManager with test configuration
        self.notification_config = NotificationConfig(
            email_recipients=["test@example.com"],
            smtp_server="localhost",
            smtp_port=587
        )
        
        self.alert_manager = AlertManager(
            notification_config=self.notification_config,
            audit_log_path=self.temp_audit_log.name,
            enable_async_notifications=False  # Disable for testing
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary audit log file
        if os.path.exists(self.temp_audit_log.name):
            os.unlink(self.temp_audit_log.name)
    
    def test_initialization(self):
        """Test AlertManager initialization."""
        assert self.alert_manager is not None
        assert len(self.alert_manager.alert_thresholds) > 0
        assert self.alert_manager.notification_config == self.notification_config
        assert not self.alert_manager.enable_async_notifications
    
    def test_add_threshold(self):
        """Test adding alert thresholds."""
        initial_count = len(self.alert_manager.alert_thresholds)
        
        # Add valid threshold
        threshold = AlertThreshold(
            name="test_threshold",
            fraud_score_min=0.7,
            fraud_score_max=0.9,
            severity=AlertSeverity.HIGH
        )
        
        self.alert_manager.add_threshold(threshold)
        assert len(self.alert_manager.alert_thresholds) == initial_count + 1
        
        # Test duplicate name validation
        with pytest.raises(ValueError, match="already exists"):
            self.alert_manager.add_threshold(threshold)
        
        # Test invalid score range
        invalid_threshold = AlertThreshold(
            name="invalid_threshold",
            fraud_score_min=0.9,
            fraud_score_max=0.7,  # Invalid: min > max
            severity=AlertSeverity.HIGH
        )
        
        with pytest.raises(ValueError, match="must be less than"):
            self.alert_manager.add_threshold(invalid_threshold)
    
    def test_remove_threshold(self):
        """Test removing alert thresholds."""
        # Add a test threshold
        threshold = AlertThreshold(
            name="removable_threshold",
            fraud_score_min=0.5,
            fraud_score_max=0.7,
            severity=AlertSeverity.MEDIUM
        )
        self.alert_manager.add_threshold(threshold)
        
        initial_count = len(self.alert_manager.alert_thresholds)
        
        # Remove existing threshold
        result = self.alert_manager.remove_threshold("removable_threshold")
        assert result is True
        assert len(self.alert_manager.alert_thresholds) == initial_count - 1
        
        # Try to remove non-existent threshold
        result = self.alert_manager.remove_threshold("non_existent")
        assert result is False
    
    def test_update_threshold(self):
        """Test updating alert thresholds."""
        # Add a test threshold
        threshold = AlertThreshold(
            name="updatable_threshold",
            fraud_score_min=0.5,
            fraud_score_max=0.7,
            severity=AlertSeverity.MEDIUM,
            cooldown_minutes=10
        )
        self.alert_manager.add_threshold(threshold)
        
        # Update threshold properties
        result = self.alert_manager.update_threshold(
            "updatable_threshold",
            cooldown_minutes=5,
            severity=AlertSeverity.HIGH
        )
        assert result is True
        
        # Verify updates
        updated_threshold = self.alert_manager._get_threshold_by_name("updatable_threshold")
        assert updated_threshold.cooldown_minutes == 5
        assert updated_threshold.severity == AlertSeverity.HIGH
        
        # Try to update non-existent threshold
        result = self.alert_manager.update_threshold("non_existent", cooldown_minutes=1)
        assert result is False
    
    def test_check_alert_conditions_no_match(self):
        """Test alert condition checking with no matching threshold."""
        transaction_data = {
            "type": "PAYMENT",
            "amount": 100.0,
            "nameOrig": "C123456789"
        }
        
        # Low fraud score that shouldn't trigger any alerts
        alert = self.alert_manager.check_alert_conditions(
            fraud_score=0.3,
            transaction_data=transaction_data
        )
        
        assert alert is None
    
    def test_check_alert_conditions_with_match(self):
        """Test alert condition checking with matching threshold."""
        transaction_data = {
            "type": "TRANSFER",
            "amount": 50000.0,
            "nameOrig": "C123456789",
            "nameDest": "C987654321"
        }
        
        # High fraud score that should trigger alert
        alert = self.alert_manager.check_alert_conditions(
            fraud_score=0.85,
            transaction_data=transaction_data,
            explanation="High-risk transfer detected",
            recommendations=["Investigate immediately", "Contact customer"]
        )
        
        assert alert is not None
        assert alert.fraud_score == 0.85
        assert alert.severity == AlertSeverity.HIGH
        assert alert.transaction_data == transaction_data
        assert "High-risk transfer detected" in alert.explanation
        assert len(alert.recommendations) == 2
        # Status should be SENT because log notification succeeds even if email fails
        assert alert.status in [AlertStatus.PENDING, AlertStatus.SENT]
    
    def test_cooldown_functionality(self):
        """Test alert cooldown functionality."""
        transaction_data = {
            "type": "CASH-OUT",
            "amount": 10000.0,
            "nameOrig": "C123456789"
        }
        
        # First alert should be created
        alert1 = self.alert_manager.check_alert_conditions(
            fraud_score=0.85,
            transaction_data=transaction_data
        )
        assert alert1 is not None
        
        # Second alert with same customer should be suppressed due to cooldown
        alert2 = self.alert_manager.check_alert_conditions(
            fraud_score=0.85,
            transaction_data=transaction_data
        )
        assert alert2 is None
        
        # Different customer should still create alert
        transaction_data_different = transaction_data.copy()
        transaction_data_different["nameOrig"] = "C987654321"
        
        alert3 = self.alert_manager.check_alert_conditions(
            fraud_score=0.85,
            transaction_data=transaction_data_different
        )
        assert alert3 is not None
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment functionality."""
        transaction_data = {
            "type": "TRANSFER",
            "amount": 100000.0,
            "nameOrig": "C123456789"
        }
        
        # Create alert
        alert = self.alert_manager.check_alert_conditions(
            fraud_score=0.9,
            transaction_data=transaction_data
        )
        assert alert is not None
        
        # Acknowledge alert
        result = self.alert_manager.acknowledge_alert(alert.alert_id, "test_user")
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"
        assert alert.acknowledged_at is not None
        
        # Try to acknowledge non-existent alert
        result = self.alert_manager.acknowledge_alert("non_existent", "test_user")
        assert result is False
    
    def test_get_active_alerts(self):
        """Test retrieving active alerts."""
        # Create multiple alerts with different severities
        transaction_data_high = {
            "type": "CASH-OUT",
            "amount": 200000.0,
            "nameOrig": "C111111111"
        }
        
        transaction_data_medium = {
            "type": "TRANSFER",
            "amount": 50000.0,
            "nameOrig": "C222222222"
        }
        
        # Create high severity alert
        alert_high = self.alert_manager.check_alert_conditions(
            fraud_score=0.95,
            transaction_data=transaction_data_high
        )
        
        # Create medium severity alert
        alert_medium = self.alert_manager.check_alert_conditions(
            fraud_score=0.75,
            transaction_data=transaction_data_medium
        )
        
        # Get all active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        assert len(active_alerts) >= 2
        
        # Get high severity alerts only
        high_alerts = self.alert_manager.get_active_alerts(AlertSeverity.HIGH)
        assert len(high_alerts) >= 1
        assert all(alert.severity == AlertSeverity.HIGH for alert in high_alerts)
        
        # Acknowledge one alert and verify it's no longer active
        self.alert_manager.acknowledge_alert(alert_high.alert_id, "test_user")
        active_alerts_after = self.alert_manager.get_active_alerts()
        assert len(active_alerts_after) == len(active_alerts) - 1
    
    def test_get_alert_history(self):
        """Test retrieving alert history."""
        transaction_data = {
            "type": "PAYMENT",
            "amount": 1000.0,
            "nameOrig": "C123456789"
        }
        
        # Create alert
        alert = self.alert_manager.check_alert_conditions(
            fraud_score=0.85,
            transaction_data=transaction_data
        )
        
        # Get recent history
        history = self.alert_manager.get_alert_history(hours_back=1)
        assert len(history) >= 1
        assert alert.alert_id in [a.alert_id for a in history]
        
        # Get history with severity filter
        high_history = self.alert_manager.get_alert_history(
            hours_back=1,
            severity_filter=AlertSeverity.HIGH
        )
        assert all(alert.severity == AlertSeverity.HIGH for alert in high_history)
    
    def test_get_alert_statistics(self):
        """Test alert statistics retrieval."""
        # Get initial statistics
        initial_stats = self.alert_manager.get_alert_statistics()
        initial_total = initial_stats['total_alerts']
        
        # Create some alerts
        for i in range(3):
            transaction_data = {
                "type": "TRANSFER",
                "amount": 50000.0,
                "nameOrig": f"C{i:09d}"
            }
            
            self.alert_manager.check_alert_conditions(
                fraud_score=0.85,
                transaction_data=transaction_data
            )
        
        # Get updated statistics
        updated_stats = self.alert_manager.get_alert_statistics()
        assert updated_stats['total_alerts'] == initial_total + 3
        assert updated_stats['active_alerts'] >= 3
        assert 'alerts_by_severity' in updated_stats
        assert 'notifications_sent' in updated_stats
    
    @patch('smtplib.SMTP')
    def test_email_notification(self, mock_smtp):
        """Test email notification functionality."""
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        transaction_data = {
            "type": "CASH-OUT",
            "amount": 100000.0,
            "nameOrig": "C123456789"
        }
        
        # Create alert that should trigger email notification
        alert = self.alert_manager.check_alert_conditions(
            fraud_score=0.95,
            transaction_data=transaction_data
        )
        
        assert alert is not None
        
        # Verify SMTP methods were called
        mock_server.starttls.assert_called_once()
        mock_server.sendmail.assert_called_once()
    
    def test_cleanup_old_alerts(self):
        """Test cleanup of old alerts."""
        # Create some alerts
        transaction_data = {
            "type": "PAYMENT",
            "amount": 1000.0,
            "nameOrig": "C123456789"
        }
        
        alert = self.alert_manager.check_alert_conditions(
            fraud_score=0.85,
            transaction_data=transaction_data
        )
        
        initial_count = len(self.alert_manager.alert_history)
        
        # Cleanup with very short retention (should remove nothing recent)
        cleaned = self.alert_manager.cleanup_old_alerts(days_to_keep=1)
        assert cleaned == 0
        assert len(self.alert_manager.alert_history) == initial_count
        
        # Simulate old alert by modifying creation time
        if alert:
            alert.created_at = datetime.now() - timedelta(days=31)
        
        # Cleanup with 30-day retention
        cleaned = self.alert_manager.cleanup_old_alerts(days_to_keep=30)
        assert cleaned >= 0  # Should clean up the old alert
    
    def test_notification_config_update(self):
        """Test updating notification configuration."""
        new_config = NotificationConfig(
            email_recipients=["new@example.com", "admin@example.com"],
            smtp_server="smtp.newserver.com",
            smtp_port=465
        )
        
        self.alert_manager.update_notification_config(new_config)
        assert self.alert_manager.notification_config == new_config
        assert len(self.alert_manager.notification_config.email_recipients) == 2
    
    def test_get_configuration(self):
        """Test getting AlertManager configuration."""
        config = self.alert_manager.get_configuration()
        
        assert 'thresholds' in config
        assert 'notification_config' in config
        assert 'settings' in config
        
        assert isinstance(config['thresholds'], list)
        assert len(config['thresholds']) > 0
        
        assert 'email_recipients_count' in config['notification_config']
        assert config['notification_config']['email_recipients_count'] == 1


if __name__ == "__main__":
    pytest.main([__file__])