#!/usr/bin/env python3
"""
Report Generator Example

This example demonstrates how to use the ReportGenerator service to create
compliance reports, transaction analysis reports, and automated report scheduling.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.services.report_generator import (
    ReportGenerator, ReportConfig, ReportData, ReportType, ReportFormat
)
from src.utils.config import get_logger

logger = get_logger(__name__)


def create_sample_data():
    """Create sample data for report generation."""
    # Generate sample transaction data
    np.random.seed(42)
    n_transactions = 1000
    
    transaction_data = pd.DataFrame({
        'step': np.random.randint(1, 745, n_transactions),
        'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_transactions),
        'amount': np.random.exponential(1000, n_transactions),
        'nameOrig': [f'C{i:08d}' for i in range(n_transactions)],
        'oldbalanceOrg': np.random.exponential(5000, n_transactions),
        'newbalanceOrig': np.random.exponential(5000, n_transactions),
        'nameDest': [f'C{i:08d}' for i in np.random.randint(0, n_transactions, n_transactions)],
        'oldbalanceDest': np.random.exponential(5000, n_transactions),
        'newbalanceDest': np.random.exponential(5000, n_transactions),
        'isFraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05]),
        'fraud_prediction': np.random.choice([0, 1], n_transactions, p=[0.93, 0.07])
    })
    
    # Sample fraud statistics
    fraud_statistics = {
        'total_transactions': n_transactions,
        'fraud_detected': int(transaction_data['isFraud'].sum()),
        'fraud_rate': transaction_data['isFraud'].mean(),
        'accuracy': 0.952,
        'precision': 0.847,
        'recall': 0.723,
        'f1_score': 0.780,
        'false_positive_rate': 0.048,
        'false_negative_rate': 0.277
    }
    
    # Sample model metrics
    model_metrics = {
        'accuracy': 0.952,
        'precision': 0.847,
        'recall': 0.723,
        'f1_score': 0.780,
        'auc_roc': 0.891,
        'training_time': 45.2,
        'prediction_time': 0.0023,
        'feature_importance': {
            'amount': 0.234,
            'oldbalanceOrg': 0.187,
            'newbalanceOrig': 0.156,
            'type_TRANSFER': 0.143,
            'type_CASH_OUT': 0.098,
            'balance_change_orig': 0.087,
            'oldbalanceDest': 0.065,
            'newbalanceDest': 0.030
        }
    }
    
    # Sample alert data
    alert_data = [
        {'severity': 'critical', 'timestamp': datetime.now() - timedelta(hours=2)},
        {'severity': 'high', 'timestamp': datetime.now() - timedelta(hours=5)},
        {'severity': 'high', 'timestamp': datetime.now() - timedelta(hours=8)},
        {'severity': 'medium', 'timestamp': datetime.now() - timedelta(hours=12)},
        {'severity': 'medium', 'timestamp': datetime.now() - timedelta(hours=18)},
        {'severity': 'low', 'timestamp': datetime.now() - timedelta(days=1)},
        {'severity': 'low', 'timestamp': datetime.now() - timedelta(days=2)}
    ]
    
    return ReportData(
        fraud_statistics=fraud_statistics,
        transaction_data=transaction_data,
        model_metrics=model_metrics,
        alert_data=alert_data
    )


def example_fraud_summary_pdf():
    """Generate a fraud summary PDF report."""
    print("\n=== Generating Fraud Summary PDF Report ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Create report configuration
    config = ReportConfig(
        report_type=ReportType.FRAUD_SUMMARY,
        report_format=ReportFormat.PDF,
        title="Fraud Detection Summary Report",
        description="Comprehensive fraud detection performance summary",
        include_charts=True,
        date_range_days=30
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Generate report
    try:
        filepath = generator.generate_fraud_summary_report(data, config)
        print(f"✓ Fraud summary PDF report generated: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error generating fraud summary PDF: {str(e)}")
        return None


def example_fraud_summary_excel():
    """Generate a fraud summary Excel report."""
    print("\n=== Generating Fraud Summary Excel Report ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Create report configuration
    config = ReportConfig(
        report_type=ReportType.FRAUD_SUMMARY,
        report_format=ReportFormat.EXCEL,
        title="Fraud Detection Summary Report - Excel",
        description="Excel format fraud detection summary with detailed sheets",
        include_raw_data=True,
        date_range_days=30
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Generate report
    try:
        filepath = generator.generate_fraud_summary_report(data, config)
        print(f"✓ Fraud summary Excel report generated: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error generating fraud summary Excel: {str(e)}")
        return None


def example_transaction_analysis_excel():
    """Generate a transaction analysis Excel report."""
    print("\n=== Generating Transaction Analysis Excel Report ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Create report configuration
    config = ReportConfig(
        report_type=ReportType.TRANSACTION_ANALYSIS,
        report_format=ReportFormat.EXCEL,
        title="Detailed Transaction Analysis Report",
        description="Comprehensive transaction pattern analysis",
        include_raw_data=True,
        date_range_days=30
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Generate report
    try:
        filepath = generator.generate_transaction_analysis_report(data, config)
        print(f"✓ Transaction analysis Excel report generated: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error generating transaction analysis Excel: {str(e)}")
        return None


def example_transaction_analysis_csv():
    """Generate a transaction analysis CSV report."""
    print("\n=== Generating Transaction Analysis CSV Report ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Create report configuration
    config = ReportConfig(
        report_type=ReportType.TRANSACTION_ANALYSIS,
        report_format=ReportFormat.CSV,
        title="Transaction Data Export",
        description="Raw transaction data export for analysis"
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Generate report
    try:
        filepath = generator.generate_transaction_analysis_report(data, config)
        print(f"✓ Transaction analysis CSV report generated: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error generating transaction analysis CSV: {str(e)}")
        return None


def example_model_performance_pdf():
    """Generate a model performance PDF report."""
    print("\n=== Generating Model Performance PDF Report ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Create report configuration
    config = ReportConfig(
        report_type=ReportType.MODEL_PERFORMANCE,
        report_format=ReportFormat.PDF,
        title="Model Performance Analysis Report",
        description="Detailed analysis of fraud detection model performance",
        include_charts=True
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Generate report
    try:
        filepath = generator.generate_model_performance_report(data, config)
        print(f"✓ Model performance PDF report generated: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error generating model performance PDF: {str(e)}")
        return None


def example_model_performance_excel():
    """Generate a model performance Excel report."""
    print("\n=== Generating Model Performance Excel Report ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Create report configuration
    config = ReportConfig(
        report_type=ReportType.MODEL_PERFORMANCE,
        report_format=ReportFormat.EXCEL,
        title="Model Performance Metrics - Excel",
        description="Excel format model performance analysis with feature importance"
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Generate report
    try:
        filepath = generator.generate_model_performance_report(data, config)
        print(f"✓ Model performance Excel report generated: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error generating model performance Excel: {str(e)}")
        return None


def example_compliance_audit_pdf():
    """Generate a compliance audit PDF report."""
    print("\n=== Generating Compliance Audit PDF Report ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Create report configuration
    config = ReportConfig(
        report_type=ReportType.COMPLIANCE_AUDIT,
        report_format=ReportFormat.PDF,
        title="Compliance Audit Report",
        description="Regulatory compliance audit for fraud detection system",
        date_range_days=30
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Generate report
    try:
        filepath = generator.generate_compliance_audit_report(data, config)
        print(f"✓ Compliance audit PDF report generated: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error generating compliance audit PDF: {str(e)}")
        return None


def example_compliance_audit_excel():
    """Generate a compliance audit Excel report."""
    print("\n=== Generating Compliance Audit Excel Report ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Create report configuration
    config = ReportConfig(
        report_type=ReportType.COMPLIANCE_AUDIT,
        report_format=ReportFormat.EXCEL,
        title="Compliance Audit Report - Excel",
        description="Excel format compliance audit with detailed metrics"
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Generate report
    try:
        filepath = generator.generate_compliance_audit_report(data, config)
        print(f"✓ Compliance audit Excel report generated: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error generating compliance audit Excel: {str(e)}")
        return None


def example_custom_report():
    """Generate a custom report using a template."""
    print("\n=== Generating Custom Report ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Create report configuration
    config = ReportConfig(
        report_type=ReportType.CUSTOM,
        report_format=ReportFormat.JSON,  # Will be HTML output
        title="Custom Fraud Analysis Report",
        description="Custom report using Jinja2 template"
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Custom HTML template
    template_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ config.title }}</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .metric { margin: 10px 0; }
            .metric-value { font-weight: bold; color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{{ config.title }}</h1>
            <p>{{ config.description }}</p>
            <p><strong>Generated:</strong> {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        </div>
        
        <h2>Fraud Detection Summary</h2>
        <div class="metric">
            Total Transactions: <span class="metric-value">{{ fraud_statistics.total_transactions | default(0) | int | format(',') }}</span>
        </div>
        <div class="metric">
            Fraud Detected: <span class="metric-value">{{ fraud_statistics.fraud_detected | default(0) | int | format(',') }}</span>
        </div>
        <div class="metric">
            Fraud Rate: <span class="metric-value">{{ (fraud_statistics.fraud_rate | default(0) * 100) | round(2) }}%</span>
        </div>
        <div class="metric">
            Model Accuracy: <span class="metric-value">{{ (fraud_statistics.accuracy | default(0) * 100) | round(1) }}%</span>
        </div>
        
        <h2>Model Performance</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Accuracy</td><td>{{ (model_metrics.accuracy | default(0) * 100) | round(1) }}%</td></tr>
            <tr><td>Precision</td><td>{{ (model_metrics.precision | default(0) * 100) | round(1) }}%</td></tr>
            <tr><td>Recall</td><td>{{ (model_metrics.recall | default(0) * 100) | round(1) }}%</td></tr>
            <tr><td>F1-Score</td><td>{{ (model_metrics.f1_score | default(0) * 100) | round(1) }}%</td></tr>
            <tr><td>AUC-ROC</td><td>{{ model_metrics.auc_roc | default(0) | round(3) }}</td></tr>
        </table>
        
        {% if model_metrics.feature_importance %}
        <h2>Top Features</h2>
        <table>
            <tr><th>Feature</th><th>Importance</th></tr>
            {% for feature, importance in (model_metrics.feature_importance.items() | list)[:5] %}
            <tr><td>{{ feature }}</td><td>{{ importance | round(3) }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
    </body>
    </html>
    """
    
    # Generate report
    try:
        filepath = generator.generate_custom_report(data, config, template_content)
        print(f"✓ Custom HTML report generated: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error generating custom report: {str(e)}")
        return None


def example_scheduled_reports():
    """Demonstrate automated report scheduling."""
    print("\n=== Setting Up Scheduled Reports ===")
    
    # Create report generator with scheduling enabled
    generator = ReportGenerator(enable_scheduling=True)
    
    # Schedule daily fraud summary
    daily_config = ReportConfig(
        report_type=ReportType.FRAUD_SUMMARY,
        report_format=ReportFormat.PDF,
        title="Daily Fraud Summary",
        description="Automated daily fraud detection summary",
        auto_schedule=True,
        schedule_frequency="daily",
        recipients=["fraud-team@company.com", "compliance@company.com"]
    )
    
    # Schedule weekly compliance audit
    weekly_config = ReportConfig(
        report_type=ReportType.COMPLIANCE_AUDIT,
        report_format=ReportFormat.EXCEL,
        title="Weekly Compliance Audit",
        description="Automated weekly compliance audit report",
        auto_schedule=True,
        schedule_frequency="weekly",
        recipients=["compliance@company.com", "audit@company.com"],
        date_range_days=7
    )
    
    # Schedule monthly transaction analysis
    monthly_config = ReportConfig(
        report_type=ReportType.TRANSACTION_ANALYSIS,
        report_format=ReportFormat.EXCEL,
        title="Monthly Transaction Analysis",
        description="Automated monthly transaction pattern analysis",
        auto_schedule=True,
        schedule_frequency="monthly",
        recipients=["analytics@company.com"],
        date_range_days=30,
        include_raw_data=True
    )
    
    try:
        # Schedule the reports
        generator.schedule_report(daily_config)
        generator.schedule_report(weekly_config)
        generator.schedule_report(monthly_config)
        
        print("✓ Scheduled reports configured:")
        print(f"  - Daily fraud summary (PDF)")
        print(f"  - Weekly compliance audit (Excel)")
        print(f"  - Monthly transaction analysis (Excel)")
        print(f"  - Total scheduled reports: {len(generator.scheduled_reports)}")
        
        # Note: In a real application, the scheduler would run continuously
        # For this example, we'll stop it immediately
        generator.stop_scheduler()
        
        return True
    except Exception as e:
        print(f"✗ Error setting up scheduled reports: {str(e)}")
        return False


def example_report_history():
    """Demonstrate report history tracking."""
    print("\n=== Report History Tracking ===")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Generate a few reports to create history
    data = create_sample_data()
    
    configs = [
        ReportConfig(ReportType.FRAUD_SUMMARY, ReportFormat.PDF, "Test PDF Report"),
        ReportConfig(ReportType.TRANSACTION_ANALYSIS, ReportFormat.EXCEL, "Test Excel Report"),
        ReportConfig(ReportType.MODEL_PERFORMANCE, ReportFormat.PDF, "Test Model Report")
    ]
    
    generated_reports = []
    for config in configs:
        try:
            if config.report_type == ReportType.FRAUD_SUMMARY:
                filepath = generator.generate_fraud_summary_report(data, config)
            elif config.report_type == ReportType.TRANSACTION_ANALYSIS:
                filepath = generator.generate_transaction_analysis_report(data, config)
            elif config.report_type == ReportType.MODEL_PERFORMANCE:
                filepath = generator.generate_model_performance_report(data, config)
            
            generated_reports.append(filepath)
        except Exception as e:
            print(f"Error generating {config.title}: {str(e)}")
    
    # Display report history
    history = generator.get_report_history()
    print(f"✓ Generated {len(generated_reports)} reports")
    print(f"✓ Report history contains {len(history)} entries")
    
    for i, record in enumerate(history, 1):
        print(f"  {i}. {record['title']} ({record['report_type']}) - {record['timestamp']}")
    
    return history


def main():
    """Run all report generation examples."""
    print("🚀 Report Generator Examples")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    # Run examples
    examples = [
        ("Fraud Summary PDF", example_fraud_summary_pdf),
        ("Fraud Summary Excel", example_fraud_summary_excel),
        ("Transaction Analysis Excel", example_transaction_analysis_excel),
        ("Transaction Analysis CSV", example_transaction_analysis_csv),
        ("Model Performance PDF", example_model_performance_pdf),
        ("Model Performance Excel", example_model_performance_excel),
        ("Compliance Audit PDF", example_compliance_audit_pdf),
        ("Compliance Audit Excel", example_compliance_audit_excel),
        ("Custom Report", example_custom_report),
        ("Scheduled Reports", example_scheduled_reports),
        ("Report History", example_report_history)
    ]
    
    results = {}
    for name, example_func in examples:
        try:
            result = example_func()
            results[name] = "✓ Success" if result else "✗ Failed"
        except Exception as e:
            results[name] = f"✗ Error: {str(e)}"
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Example Results Summary")
    print("=" * 50)
    
    for name, status in results.items():
        print(f"{name:30} {status}")
    
    successful = sum(1 for status in results.values() if status.startswith("✓"))
    total = len(results)
    print(f"\nSuccess Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    print(f"\n📁 Generated reports are saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()