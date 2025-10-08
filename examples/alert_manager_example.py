"""
Example demonstrating AlertManager functionality for fraud detection alerts.

This example shows how to:
1. Configure and initialize the AlertManager
2. Set up alert thresholds and notification settings
3. Process transactions and generate alerts
4. Handle alert acknowledgments and monitoring
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.alert_manager import (
    AlertManager, AlertThreshold, AlertSeverity, NotificationType,
    NotificationConfig
)
from src.services.fraud_detector import FraudDetector
from src.utils.config import get_logger

logger = get_logger(__name__)


def setup_notification_config():
    """Setup notification configuration for alerts."""
    # Configure email notifications
    # Note: In production, these would come from environment variables or config files
    notification_config = NotificationConfig(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        smtp_username="your-email@gmail.com",  # Replace with actual email
        smtp_password="your-app-password",     # Replace with actual app password
        email_from="fraud-alerts@yourcompany.com",
        email_recipients=[
            "fraud-team@yourcompany.com",
            "security@yourcompany.com"
        ],
        # SMS configuration (placeholder)
        sms_provider="twilio",
        sms_api_key="your-sms-api-key",
        sms_recipients=["+1234567890"],
        # Webhook configuration
        webhook_urls=["https://your-webhook-endpoint.com/fraud-alerts"],
        webhook_timeout=30
    )
    
    return notification_config


def setup_custom_alert_thresholds(alert_manager):
    """Setup custom alert thresholds for different risk scenarios."""
    
    # Critical fraud threshold - immediate action required
    critical_threshold = AlertThreshold(
        name="critical_fraud_immediate",
        fraud_score_min=0.95,
        fraud_score_max=1.0,
        severity=AlertSeverity.CRITICAL,
        notification_types=[
            NotificationType.EMAIL,
            NotificationType.SMS,
            NotificationType.WEBHOOK,
            NotificationType.LOG_ONLY
        ],
        cooldown_minutes=0,  # No cooldown for critical alerts
        enabled=True
    )
    
    # Large amount transfer threshold
    large_transfer_threshold = AlertThreshold(
        name="large_transfer_alert",
        fraud_score_min=0.7,
        fraud_score_max=1.0,
        severity=AlertSeverity.HIGH,
        notification_types=[NotificationType.EMAIL, NotificationType.LOG_ONLY],
        cooldown_minutes=2,
        enabled=True
    )
    
    # Suspicious pattern threshold
    suspicious_pattern_threshold = AlertThreshold(
        name="suspicious_pattern",
        fraud_score_min=0.6,
        fraud_score_max=0.8,
        severity=AlertSeverity.MEDIUM,
        notification_types=[NotificationType.LOG_ONLY],
        cooldown_minutes=10,
        enabled=True
    )
    
    # Add thresholds to alert manager
    try:
        alert_manager.add_threshold(critical_threshold)
        alert_manager.add_threshold(large_transfer_threshold)
        alert_manager.add_threshold(suspicious_pattern_threshold)
        
        logger.info("Custom alert thresholds configured successfully")
        
    except Exception as e:
        logger.error(f"Error configuring alert thresholds: {e}")


def simulate_fraud_transactions():
    """Generate sample fraud transactions for testing alerts."""
    
    # High-risk transactions that should trigger alerts
    high_risk_transactions = [
        {
            "transaction_id": "TXN_001",
            "step": 100,
            "type": "CASH-OUT",
            "amount": 250000.0,
            "nameOrig": "C1234567890",
            "oldbalanceOrg": 300000.0,
            "newbalanceOrig": 50000.0,
            "nameDest": "C9876543210",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 250000.0
        },
        {
            "transaction_id": "TXN_002",
            "step": 150,
            "type": "TRANSFER",
            "amount": 500000.0,
            "nameOrig": "C2345678901",
            "oldbalanceOrg": 500000.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C8765432109",
            "oldbalanceDest": 100000.0,
            "newbalanceDest": 600000.0
        },
        {
            "transaction_id": "TXN_003",
            "step": 200,
            "type": "CASH-OUT",
            "amount": 75000.0,
            "nameOrig": "C3456789012",
            "oldbalanceOrg": 75000.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C7654321098",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 75000.0
        }
    ]
    
    # Medium-risk transactions
    medium_risk_transactions = [
        {
            "transaction_id": "TXN_004",
            "step": 250,
            "type": "TRANSFER",
            "amount": 25000.0,
            "nameOrig": "C4567890123",
            "oldbalanceOrg": 50000.0,
            "newbalanceOrig": 25000.0,
            "nameDest": "C6543210987",
            "oldbalanceDest": 10000.0,
            "newbalanceDest": 35000.0
        }
    ]
    
    return high_risk_transactions + medium_risk_transactions


def demonstrate_alert_processing():
    """Demonstrate complete alert processing workflow."""
    
    print("=" * 60)
    print("FRAUD DETECTION ALERT MANAGER DEMONSTRATION")
    print("=" * 60)
    
    # 1. Setup notification configuration
    print("\n1. Setting up notification configuration...")
    notification_config = setup_notification_config()
    print(f"   - Email recipients: {len(notification_config.email_recipients)}")
    print(f"   - SMS recipients: {len(notification_config.sms_recipients)}")
    print(f"   - Webhook URLs: {len(notification_config.webhook_urls)}")
    
    # 2. Initialize AlertManager
    print("\n2. Initializing AlertManager...")
    alert_manager = AlertManager(
        notification_config=notification_config,
        enable_async_notifications=False  # Synchronous for demo
    )
    
    # 3. Setup custom alert thresholds
    print("\n3. Configuring custom alert thresholds...")
    setup_custom_alert_thresholds(alert_manager)
    
    # Display configured thresholds
    print(f"   - Total thresholds configured: {len(alert_manager.alert_thresholds)}")
    for threshold in alert_manager.alert_thresholds:
        print(f"     * {threshold.name}: {threshold.fraud_score_min:.2f}-{threshold.fraud_score_max:.2f} "
              f"({threshold.severity.value})")
    
    # 4. Initialize FraudDetector with AlertManager
    print("\n4. Initializing FraudDetector with AlertManager...")
    fraud_detector = FraudDetector(
        model=None,  # No model for demo
        risk_threshold=0.6,
        high_risk_threshold=0.8,
        enable_explanations=True,
        alert_manager=alert_manager
    )
    
    # 5. Process sample transactions
    print("\n5. Processing sample transactions...")
    transactions = simulate_fraud_transactions()
    
    # Simulate fraud scores for transactions
    fraud_scores = [0.98, 0.92, 0.85, 0.65]  # High to medium risk scores
    
    alerts_created = []
    
    for i, transaction in enumerate(transactions):
        fraud_score = fraud_scores[i] if i < len(fraud_scores) else 0.5
        
        print(f"\n   Processing Transaction {transaction['transaction_id']}:")
        print(f"   - Type: {transaction['type']}")
        print(f"   - Amount: ${transaction['amount']:,.2f}")
        print(f"   - Fraud Score: {fraud_score:.2%}")
        
        # Check for alert conditions
        alert = alert_manager.check_alert_conditions(
            fraud_score=fraud_score,
            transaction_data=transaction,
            risk_factors={
                'high_risk_factors': ['Large amount transaction', 'Complete balance depletion'],
                'medium_risk_factors': ['High-risk transaction type'],
                'risk_scores': {
                    'transaction_type': 0.8,
                    'amount': 0.9,
                    'balance_patterns': 0.7
                }
            },
            explanation=f"Transaction shows {fraud_score:.1%} probability of fraud based on amount, "
                       f"transaction type, and balance patterns.",
            recommendations=[
                "Investigate transaction immediately",
                "Contact customer for verification",
                "Review account for other suspicious activities"
            ]
        )
        
        if alert:
            alerts_created.append(alert)
            print(f"   ✓ ALERT CREATED: {alert.alert_id} (Severity: {alert.severity.value})")
        else:
            print(f"   - No alert triggered (below threshold or in cooldown)")
    
    # 6. Display alert statistics
    print(f"\n6. Alert Statistics:")
    stats = alert_manager.get_alert_statistics()
    print(f"   - Total alerts created: {stats['total_alerts']}")
    print(f"   - Active alerts: {stats['active_alerts']}")
    print(f"   - Notifications sent: {stats['notifications_sent']}")
    print(f"   - Notification success rate: {stats['notification_success_rate']:.1%}")
    
    # Display alerts by severity
    print(f"\n   Alerts by severity:")
    for severity, count in stats['alerts_by_severity'].items():
        if count > 0:
            print(f"   - {severity}: {count}")
    
    # 7. Demonstrate alert management
    print(f"\n7. Alert Management:")
    
    # Get active alerts
    active_alerts = alert_manager.get_active_alerts()
    print(f"   - Active alerts: {len(active_alerts)}")
    
    if active_alerts:
        # Show details of first alert
        first_alert = active_alerts[0]
        print(f"\n   Sample Alert Details:")
        print(f"   - Alert ID: {first_alert.alert_id}")
        print(f"   - Fraud Score: {first_alert.fraud_score:.1%}")
        print(f"   - Severity: {first_alert.severity.value}")
        print(f"   - Status: {first_alert.status.value}")
        print(f"   - Created: {first_alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Acknowledge the alert
        print(f"\n   Acknowledging alert {first_alert.alert_id}...")
        success = alert_manager.acknowledge_alert(first_alert.alert_id, "fraud_analyst_demo")
        if success:
            print(f"   ✓ Alert acknowledged successfully")
            print(f"   - Status: {first_alert.status.value}")
            print(f"   - Acknowledged by: {first_alert.acknowledged_by}")
        
        # Get updated active alerts count
        updated_active = alert_manager.get_active_alerts()
        print(f"   - Active alerts after acknowledgment: {len(updated_active)}")
    
    # 8. Alert history
    print(f"\n8. Alert History (last 24 hours):")
    history = alert_manager.get_alert_history(hours_back=24)
    print(f"   - Total alerts in history: {len(history)}")
    
    for alert in history[:3]:  # Show first 3 alerts
        print(f"   - {alert.alert_id}: {alert.severity.value} "
              f"(Score: {alert.fraud_score:.1%}, Status: {alert.status.value})")
    
    # 9. Configuration summary
    print(f"\n9. AlertManager Configuration:")
    config = alert_manager.get_configuration()
    print(f"   - Thresholds configured: {len(config['thresholds'])}")
    print(f"   - Email recipients: {config['notification_config']['email_recipients_count']}")
    print(f"   - Async notifications: {config['settings']['enable_async_notifications']}")
    print(f"   - Audit log: {config['settings']['audit_log_path']}")
    
    print(f"\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


def demonstrate_threshold_management():
    """Demonstrate alert threshold management."""
    
    print("\n" + "=" * 50)
    print("ALERT THRESHOLD MANAGEMENT DEMONSTRATION")
    print("=" * 50)
    
    # Initialize AlertManager
    alert_manager = AlertManager()
    
    print(f"\nInitial thresholds: {len(alert_manager.alert_thresholds)}")
    
    # Add custom threshold
    custom_threshold = AlertThreshold(
        name="weekend_high_risk",
        fraud_score_min=0.75,
        fraud_score_max=0.95,
        severity=AlertSeverity.HIGH,
        notification_types=[NotificationType.EMAIL],
        cooldown_minutes=3,
        enabled=True
    )
    
    alert_manager.add_threshold(custom_threshold)
    print(f"Added custom threshold: {custom_threshold.name}")
    
    # Update threshold
    alert_manager.update_threshold(
        "weekend_high_risk",
        cooldown_minutes=5,
        enabled=False
    )
    print(f"Updated threshold cooldown and disabled it")
    
    # Remove threshold
    removed = alert_manager.remove_threshold("weekend_high_risk")
    print(f"Removed threshold: {removed}")
    
    print(f"Final thresholds: {len(alert_manager.alert_thresholds)}")


if __name__ == "__main__":
    try:
        # Run main demonstration
        demonstrate_alert_processing()
        
        # Run threshold management demo
        demonstrate_threshold_management()
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        print(f"\nError occurred: {e}")
        print("Note: Some features (like email notifications) require proper configuration.")