"""
Example demonstrating model monitoring and performance tracking functionality.

This example shows how to use the monitoring and decision logging components
to track model performance, detect drift, and log fraud detection decisions.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.services.fraud_detector import FraudDetector
from src.services.model_monitor import ModelMonitor
from src.services.decision_logger import DecisionLogger
from src.utils.monitoring_utils import MonitoringDashboard, create_monitoring_alerts
from src.models.logistic_regression_model import LogisticRegressionModel
from src.utils.config import get_logger

logger = get_logger(__name__)


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample transaction data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic transaction data
    data = {
        'step': np.random.randint(1, 745, n_samples),
        'type': np.random.choice(['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples),
        'amount': np.random.lognormal(8, 2, n_samples),
        'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
        'oldbalanceOrg': np.random.lognormal(10, 2, n_samples),
        'newbalanceOrig': np.random.lognormal(10, 2, n_samples),
        'nameDest': [f'C{i+1000000:09d}' for i in range(n_samples)],
        'oldbalanceDest': np.random.lognormal(9, 2, n_samples),
        'newbalanceDest': np.random.lognormal(9, 2, n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)


def demonstrate_monitoring():
    """Demonstrate monitoring and decision logging functionality."""
    print("=== Fraud Detection Monitoring Example ===\n")
    
    # Create sample data
    print("1. Creating sample transaction data...")
    train_data = create_sample_data(5000)
    test_data = create_sample_data(1000)
    
    # Prepare features - use simple feature engineering for demo
    print("2. Preparing features...")
    
    def simple_feature_engineering(df):
        """Simple feature engineering for demo purposes."""
        df_processed = df.copy()
        
        # Encode transaction types
        type_mapping = {'CASH-IN': 0, 'CASH-OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
        df_processed['type_encoded'] = df_processed['type'].map(type_mapping).fillna(3)
        
        # Calculate balance changes
        df_processed['balance_change_orig'] = df_processed['newbalanceOrig'] - df_processed['oldbalanceOrg']
        df_processed['balance_change_dest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
        
        # Amount to balance ratios
        df_processed['amount_to_balance_ratio'] = np.where(
            df_processed['oldbalanceOrg'] > 0,
            df_processed['amount'] / df_processed['oldbalanceOrg'],
            0
        )
        
        # Select numerical features for model
        feature_columns = [
            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
            'oldbalanceDest', 'newbalanceDest', 'type_encoded',
            'balance_change_orig', 'balance_change_dest', 'amount_to_balance_ratio'
        ]
        
        return df_processed[feature_columns]
    
    # Process training data
    X_train = simple_feature_engineering(train_data.drop('isFraud', axis=1))
    y_train = train_data['isFraud']
    
    # Process test data
    X_test = simple_feature_engineering(test_data.drop('isFraud', axis=1))
    y_test = test_data['isFraud']
    
    # Train a simple model
    print("3. Training fraud detection model...")
    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    
    # Initialize FraudDetector with monitoring enabled
    print("4. Initializing FraudDetector with monitoring...")
    fraud_detector = FraudDetector(
        model=model,
        risk_threshold=0.5,
        high_risk_threshold=0.8,
        enable_monitoring=True,
        enable_decision_logging=True
    )
    
    # Set reference data for drift detection
    print("5. Setting reference data for drift detection...")
    fraud_detector.set_reference_data_for_monitoring(X_train, y_train)
    
    # Simulate real-time predictions with monitoring
    print("6. Simulating real-time predictions...")
    
    for i in range(100):
        # Get a random transaction
        transaction_idx = np.random.randint(0, len(test_data))
        transaction = test_data.iloc[transaction_idx].to_dict()
        
        # Add some context for logging
        context = {
            'user_id': f'user_{i % 10}',
            'session_id': f'session_{i // 10}',
            'ip_address': f'192.168.1.{i % 255}',
            'transaction_id': f'txn_{i:06d}'
        }
        
        # Score the transaction (this will automatically log the decision and monitor performance)
        try:
            fraud_score = fraud_detector.score_transaction(transaction, context)
        except Exception as e:
            # Handle any scoring errors gracefully for demo
            logger.warning(f"Error scoring transaction {i}: {e}")
            fraud_score = 0.0
        
        # Simulate getting actual outcome later (for some transactions)
        if i % 10 == 0 and fraud_detector.decision_logger:  # Update outcome for every 10th transaction
            actual_outcome = test_data.iloc[transaction_idx]['isFraud']
            # This would normally be done later when investigation is complete
            # For demo, we'll use the true label
            
        if i % 20 == 0:
            print(f"   Processed {i+1} transactions, latest fraud score: {fraud_score:.3f}")
    
    # Simulate batch prediction
    print("7. Simulating batch prediction...")
    batch_data = test_data.sample(50)
    batch_results = fraud_detector.batch_predict(batch_data, data_source="daily_batch")
    
    print(f"   Batch processed {len(batch_results)} transactions")
    print(f"   Found {batch_results['fraud_prediction'].sum()} potential fraud cases")
    
    # Get monitoring summaries
    print("\n8. Retrieving monitoring summaries...")
    
    # Get monitoring summary
    monitoring_summary = fraud_detector.get_monitoring_summary()
    if monitoring_summary:
        print("   Monitoring Summary:")
        print(f"   - Total performance records: {monitoring_summary['monitoring_status']['total_performance_records']}")
        print(f"   - Total drift records: {monitoring_summary['monitoring_status']['total_drift_records']}")
        
        if 'recent_performance' in monitoring_summary:
            perf = monitoring_summary['recent_performance']
            print(f"   - Average F1 Score: {perf.get('avg_f1_score', 0):.3f}")
            print(f"   - Average Fraud Rate: {perf.get('avg_fraud_rate', 0):.3f}")
    
    # Get decision statistics
    decision_stats = fraud_detector.get_decision_statistics(hours=1)
    if decision_stats:
        print("   Decision Statistics (last hour):")
        print(f"   - Total decisions: {decision_stats['total_decisions']}")
        print(f"   - Fraud predictions: {decision_stats['fraud_predictions']}")
        print(f"   - Human reviews required: {decision_stats['human_reviews_required']}")
        print(f"   - Alerts generated: {decision_stats['alerts_generated']}")
    
    # Get performance trend
    performance_trend = fraud_detector.get_performance_trend(days=1)
    if performance_trend and 'f1_scores' in performance_trend:
        print(f"   Performance Trend: {len(performance_trend['f1_scores'])} data points")
    
    # Get drift summary
    drift_summary = fraud_detector.get_drift_summary(days=1)
    if drift_summary and 'feature_summary' in drift_summary:
        print(f"   Drift Summary: {len(drift_summary['feature_summary'])} features monitored")
        drift_detections = drift_summary.get('total_drift_detections', 0)
        print(f"   - Total drift detections: {drift_detections}")
    
    # Create monitoring alerts
    print("\n9. Checking for monitoring alerts...")
    alerts = create_monitoring_alerts(monitoring_summary or {})
    
    if alerts:
        print(f"   Found {len(alerts)} alerts:")
        for alert in alerts:
            print(f"   - {alert['severity'].upper()}: {alert['message']}")
    else:
        print("   No alerts detected - system is healthy")
    
    # Generate monitoring dashboard
    print("\n10. Generating monitoring visualizations...")
    dashboard = MonitoringDashboard()
    
    try:
        # Create performance trend chart
        if performance_trend and 'timestamps' in performance_trend:
            perf_chart = dashboard.create_performance_trend_chart(performance_trend)
            print("   - Performance trend chart created")
        
        # Create drift detection chart
        if drift_summary and 'feature_summary' in drift_summary:
            drift_chart = dashboard.create_drift_detection_chart(drift_summary)
            print("   - Drift detection chart created")
        
        # Create decision statistics chart
        if decision_stats:
            decision_chart = dashboard.create_decision_statistics_chart(decision_stats)
            print("   - Decision statistics chart created")
        
        # Create model health dashboard
        if monitoring_summary:
            health_chart = dashboard.create_model_health_dashboard(monitoring_summary)
            print("   - Model health dashboard created")
        
        print("   All visualizations created successfully!")
        
    except Exception as e:
        print(f"   Error creating visualizations: {e}")
        print("   (This is expected if plotly is not installed)")
    
    # Export monitoring data
    print("\n11. Exporting monitoring data...")
    
    try:
        # Export monitoring report
        report_path = "monitoring_report.json"
        fraud_detector.export_monitoring_report(report_path, days=1)
        print(f"   - Monitoring report exported to {report_path}")
        
        # Export decision data
        decisions_path = "decisions_export.csv"
        fraud_detector.export_decisions_for_analysis(decisions_path, days=1)
        print(f"   - Decision data exported to {decisions_path}")
        
    except Exception as e:
        print(f"   Error exporting data: {e}")
    
    # Demonstrate outcome feedback
    print("\n12. Demonstrating outcome feedback...")
    
    if fraud_detector.decision_logger and fraud_detector.decision_logger.recent_decisions:
        # Update outcome for the first few decisions
        for i, decision in enumerate(fraud_detector.decision_logger.recent_decisions[:5]):
            # Simulate investigation outcome
            actual_outcome = np.random.choice([0, 1])
            investigation_notes = f"Investigation completed for decision {decision.decision_id}"
            action_taken = "account_flagged" if actual_outcome == 1 else "no_action"
            
            success = fraud_detector.update_decision_outcome(
                decision.decision_id,
                actual_outcome,
                investigation_notes,
                action_taken
            )
            
            if success:
                print(f"   - Updated outcome for decision {decision.decision_id}: {actual_outcome}")
    
    # Get performance feedback with outcomes
    performance_feedback = fraud_detector.get_model_performance_feedback(days=1)
    if performance_feedback and 'total_decisions_with_outcomes' in performance_feedback:
        print("   Performance Feedback (with outcomes):")
        print(f"   - Decisions with outcomes: {performance_feedback['total_decisions_with_outcomes']}")
        if performance_feedback['total_decisions_with_outcomes'] > 0:
            print(f"   - Accuracy: {performance_feedback['accuracy']:.3f}")
            print(f"   - Precision: {performance_feedback['precision']:.3f}")
            print(f"   - Recall: {performance_feedback['recall']:.3f}")
            print(f"   - F1 Score: {performance_feedback['f1_score']:.3f}")
    else:
        print("   Performance Feedback: No decisions with outcomes available yet")
    
    print("\n=== Monitoring Example Complete ===")
    print("\nKey Features Demonstrated:")
    print("✓ Real-time decision logging with full audit trail")
    print("✓ Model performance monitoring over time")
    print("✓ Data drift detection using statistical tests")
    print("✓ Comprehensive monitoring dashboards and visualizations")
    print("✓ Automated alert generation for model degradation")
    print("✓ Outcome feedback loop for continuous improvement")
    print("✓ Export capabilities for compliance and analysis")
    
    return fraud_detector


if __name__ == "__main__":
    try:
        fraud_detector = demonstrate_monitoring()
        print("\nMonitoring example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in monitoring example: {e}")
        print(f"Error: {e}")
        sys.exit(1)