#!/usr/bin/env python3
"""
Tests for ReportGenerator service.

This module contains unit tests for the ReportGenerator class and its
report generation capabilities.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.services.report_generator import (
    ReportGenerator, ReportConfig, ReportData, ReportType, ReportFormat
)


class TestReportGenerator:
    """Test cases for ReportGenerator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def report_generator(self, temp_dir):
        """Create a ReportGenerator instance for testing."""
        return ReportGenerator(output_directory=temp_dir, enable_scheduling=False)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Generate sample transaction data
        np.random.seed(42)
        n_transactions = 100
        
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
                'type_CASH_OUT': 0.098
            }
        }
        
        alert_data = [
            {'severity': 'critical', 'timestamp': datetime.now() - timedelta(hours=2)},
            {'severity': 'high', 'timestamp': datetime.now() - timedelta(hours=5)},
            {'severity': 'medium', 'timestamp': datetime.now() - timedelta(hours=12)},
            {'severity': 'low', 'timestamp': datetime.now() - timedelta(days=1)}
        ]
        
        return ReportData(
            fraud_statistics=fraud_statistics,
            transaction_data=transaction_data,
            model_metrics=model_metrics,
            alert_data=alert_data
        )
    
    def test_report_generator_initialization(self, temp_dir):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator(output_directory=temp_dir)
        
        assert generator.output_directory == Path(temp_dir)
        assert generator.enable_scheduling is True
        assert generator.scheduled_reports == []
        assert generator.report_history == []
    
    def test_fraud_summary_pdf_generation(self, report_generator, sample_data):
        """Test fraud summary PDF report generation."""
        config = ReportConfig(
            report_type=ReportType.FRAUD_SUMMARY,
            report_format=ReportFormat.PDF,
            title="Test Fraud Summary PDF",
            description="Test PDF report"
        )
        
        filepath = report_generator.generate_fraud_summary_report(sample_data, config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.pdf')
        assert 'fraud_summary' in filepath
        
        # Check report history
        history = report_generator.get_report_history()
        assert len(history) == 1
        assert history[0]['report_type'] == 'fraud_summary'
        assert history[0]['report_format'] == 'pdf'
    
    def test_fraud_summary_excel_generation(self, report_generator, sample_data):
        """Test fraud summary Excel report generation."""
        config = ReportConfig(
            report_type=ReportType.FRAUD_SUMMARY,
            report_format=ReportFormat.EXCEL,
            title="Test Fraud Summary Excel",
            include_raw_data=True
        )
        
        filepath = report_generator.generate_fraud_summary_report(sample_data, config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.xlsx')
        assert 'fraud_summary' in filepath
    
    def test_transaction_analysis_excel_generation(self, report_generator, sample_data):
        """Test transaction analysis Excel report generation."""
        config = ReportConfig(
            report_type=ReportType.TRANSACTION_ANALYSIS,
            report_format=ReportFormat.EXCEL,
            title="Test Transaction Analysis Excel"
        )
        
        filepath = report_generator.generate_transaction_analysis_report(sample_data, config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.xlsx')
        assert 'transaction_analysis' in filepath
    
    def test_transaction_analysis_csv_generation(self, report_generator, sample_data):
        """Test transaction analysis CSV report generation."""
        config = ReportConfig(
            report_type=ReportType.TRANSACTION_ANALYSIS,
            report_format=ReportFormat.CSV,
            title="Test Transaction Analysis CSV"
        )
        
        filepath = report_generator.generate_transaction_analysis_report(sample_data, config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.csv')
        assert 'transaction_analysis' in filepath
    
    def test_model_performance_pdf_generation(self, report_generator, sample_data):
        """Test model performance PDF report generation."""
        config = ReportConfig(
            report_type=ReportType.MODEL_PERFORMANCE,
            report_format=ReportFormat.PDF,
            title="Test Model Performance PDF"
        )
        
        filepath = report_generator.generate_model_performance_report(sample_data, config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.pdf')
        assert 'model_performance' in filepath
    
    def test_model_performance_excel_generation(self, report_generator, sample_data):
        """Test model performance Excel report generation."""
        config = ReportConfig(
            report_type=ReportType.MODEL_PERFORMANCE,
            report_format=ReportFormat.EXCEL,
            title="Test Model Performance Excel"
        )
        
        filepath = report_generator.generate_model_performance_report(sample_data, config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.xlsx')
        assert 'model_performance' in filepath
    
    def test_compliance_audit_pdf_generation(self, report_generator, sample_data):
        """Test compliance audit PDF report generation."""
        config = ReportConfig(
            report_type=ReportType.COMPLIANCE_AUDIT,
            report_format=ReportFormat.PDF,
            title="Test Compliance Audit PDF"
        )
        
        filepath = report_generator.generate_compliance_audit_report(sample_data, config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.pdf')
        assert 'compliance_audit' in filepath
    
    def test_compliance_audit_excel_generation(self, report_generator, sample_data):
        """Test compliance audit Excel report generation."""
        config = ReportConfig(
            report_type=ReportType.COMPLIANCE_AUDIT,
            report_format=ReportFormat.EXCEL,
            title="Test Compliance Audit Excel"
        )
        
        filepath = report_generator.generate_compliance_audit_report(sample_data, config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.xlsx')
        assert 'compliance_audit' in filepath
    
    def test_custom_report_generation(self, report_generator, sample_data):
        """Test custom report generation with template."""
        config = ReportConfig(
            report_type=ReportType.CUSTOM,
            report_format=ReportFormat.JSON,  # Will be HTML output
            title="Test Custom Report"
        )
        
        template_content = """
        <html>
        <body>
            <h1>{{ config.title }}</h1>
            <p>Total Transactions: {{ fraud_statistics.total_transactions }}</p>
            <p>Fraud Rate: {{ fraud_statistics.fraud_rate }}</p>
        </body>
        </html>
        """
        
        filepath = report_generator.generate_custom_report(sample_data, config, template_content)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.html')
        assert 'custom_report' in filepath
        
        # Check content
        content = Path(filepath).read_text()
        assert 'Test Custom Report' in content
        assert 'Total Transactions:' in content
    
    def test_transaction_analysis_without_data_raises_error(self, report_generator):
        """Test that transaction analysis without data raises an error."""
        config = ReportConfig(
            report_type=ReportType.TRANSACTION_ANALYSIS,
            report_format=ReportFormat.EXCEL,
            title="Test Without Data"
        )
        
        empty_data = ReportData()  # No transaction data
        
        with pytest.raises(ValueError, match="Transaction data is required"):
            report_generator.generate_transaction_analysis_report(empty_data, config)
    
    def test_custom_report_without_template_raises_error(self, report_generator, sample_data):
        """Test that custom report without template raises an error."""
        config = ReportConfig(
            report_type=ReportType.CUSTOM,
            report_format=ReportFormat.JSON,
            title="Test Without Template"
        )
        
        with pytest.raises(ValueError, match="Template content or template path must be provided"):
            report_generator.generate_custom_report(sample_data, config)
    
    def test_unsupported_format_raises_error(self, report_generator, sample_data):
        """Test that unsupported format raises an error."""
        config = ReportConfig(
            report_type=ReportType.FRAUD_SUMMARY,
            report_format=ReportFormat.CSV,  # Not supported for fraud summary
            title="Test Unsupported Format"
        )
        
        with pytest.raises(ValueError, match="Unsupported format"):
            report_generator.generate_fraud_summary_report(sample_data, config)
    
    def test_alert_data_analysis(self, report_generator):
        """Test alert data analysis functionality."""
        alert_data = [
            {'severity': 'critical'},
            {'severity': 'critical'},
            {'severity': 'high'},
            {'severity': 'medium'},
            {'severity': 'low'},
            {'severity': 'low'},
            {'severity': 'low'}
        ]
        
        result = report_generator._analyze_alert_data(alert_data)
        
        assert result['critical'] == 2
        assert result['high'] == 1
        assert result['medium'] == 1
        assert result['low'] == 3
        assert result['critical_pct'] == 2/7
        assert result['low_pct'] == 3/7
    
    def test_empty_alert_data_analysis(self, report_generator):
        """Test alert data analysis with empty data."""
        result = report_generator._analyze_alert_data([])
        assert result == {}
    
    def test_report_history_tracking(self, report_generator, sample_data):
        """Test report history tracking functionality."""
        config1 = ReportConfig(
            report_type=ReportType.FRAUD_SUMMARY,
            report_format=ReportFormat.PDF,
            title="Test Report 1"
        )
        
        config2 = ReportConfig(
            report_type=ReportType.MODEL_PERFORMANCE,
            report_format=ReportFormat.EXCEL,
            title="Test Report 2"
        )
        
        # Generate reports
        report_generator.generate_fraud_summary_report(sample_data, config1)
        report_generator.generate_model_performance_report(sample_data, config2)
        
        # Check history
        history = report_generator.get_report_history()
        assert len(history) == 2
        
        assert history[0]['report_type'] == 'fraud_summary'
        assert history[0]['title'] == 'Test Report 1'
        assert history[1]['report_type'] == 'model_performance'
        assert history[1]['title'] == 'Test Report 2'
    
    @patch('src.services.report_generator.schedule')
    def test_report_scheduling(self, mock_schedule, temp_dir):
        """Test report scheduling functionality."""
        generator = ReportGenerator(output_directory=temp_dir, enable_scheduling=True)
        
        config = ReportConfig(
            report_type=ReportType.FRAUD_SUMMARY,
            report_format=ReportFormat.PDF,
            title="Scheduled Report",
            auto_schedule=True,
            schedule_frequency="daily",
            recipients=["test@example.com"]
        )
        
        generator.schedule_report(config)
        
        assert len(generator.scheduled_reports) == 1
        assert generator.scheduled_reports[0] == config
        mock_schedule.every.assert_called_once()
    
    def test_scheduling_disabled_warning(self, temp_dir, caplog):
        """Test warning when scheduling is disabled."""
        generator = ReportGenerator(output_directory=temp_dir, enable_scheduling=False)
        
        config = ReportConfig(
            report_type=ReportType.FRAUD_SUMMARY,
            report_format=ReportFormat.PDF,
            title="Test Report",
            auto_schedule=True
        )
        
        generator.schedule_report(config)
        
        assert "Report scheduling is disabled" in caplog.text
    
    def test_auto_schedule_disabled_warning(self, temp_dir, caplog):
        """Test warning when auto-schedule is disabled."""
        generator = ReportGenerator(output_directory=temp_dir, enable_scheduling=True)
        
        config = ReportConfig(
            report_type=ReportType.FRAUD_SUMMARY,
            report_format=ReportFormat.PDF,
            title="Test Report",
            auto_schedule=False
        )
        
        generator.schedule_report(config)
        
        assert "Auto-scheduling not enabled" in caplog.text


class TestReportConfig:
    """Test cases for ReportConfig dataclass."""
    
    def test_report_config_creation(self):
        """Test ReportConfig creation with required fields."""
        config = ReportConfig(
            report_type=ReportType.FRAUD_SUMMARY,
            report_format=ReportFormat.PDF,
            title="Test Report"
        )
        
        assert config.report_type == ReportType.FRAUD_SUMMARY
        assert config.report_format == ReportFormat.PDF
        assert config.title == "Test Report"
        assert config.description == ""
        assert config.include_charts is True
        assert config.include_raw_data is False
        assert config.date_range_days == 30
        assert config.recipients == []
    
    def test_report_config_with_all_fields(self):
        """Test ReportConfig creation with all fields."""
        config = ReportConfig(
            report_type=ReportType.TRANSACTION_ANALYSIS,
            report_format=ReportFormat.EXCEL,
            title="Full Config Test",
            description="Test description",
            include_charts=False,
            include_raw_data=True,
            date_range_days=7,
            output_directory="/custom/path",
            template_path="/templates/custom.html",
            auto_schedule=True,
            schedule_frequency="weekly",
            recipients=["user1@example.com", "user2@example.com"]
        )
        
        assert config.report_type == ReportType.TRANSACTION_ANALYSIS
        assert config.report_format == ReportFormat.EXCEL
        assert config.title == "Full Config Test"
        assert config.description == "Test description"
        assert config.include_charts is False
        assert config.include_raw_data is True
        assert config.date_range_days == 7
        assert config.output_directory == "/custom/path"
        assert config.template_path == "/templates/custom.html"
        assert config.auto_schedule is True
        assert config.schedule_frequency == "weekly"
        assert len(config.recipients) == 2


class TestReportData:
    """Test cases for ReportData dataclass."""
    
    def test_report_data_creation_empty(self):
        """Test ReportData creation with default values."""
        data = ReportData()
        
        assert data.fraud_statistics == {}
        assert data.transaction_data is None
        assert data.model_metrics is None
        assert data.alert_data == []
        assert data.time_series_data == {}
        assert data.custom_data == {}
    
    def test_report_data_creation_with_data(self):
        """Test ReportData creation with actual data."""
        fraud_stats = {'total_transactions': 1000, 'fraud_rate': 0.05}
        transaction_df = pd.DataFrame({'amount': [100, 200, 300]})
        model_metrics = {'accuracy': 0.95, 'precision': 0.87}
        alerts = [{'severity': 'high', 'timestamp': datetime.now()}]
        
        data = ReportData(
            fraud_statistics=fraud_stats,
            transaction_data=transaction_df,
            model_metrics=model_metrics,
            alert_data=alerts
        )
        
        assert data.fraud_statistics == fraud_stats
        assert data.transaction_data.equals(transaction_df)
        assert data.model_metrics == model_metrics
        assert data.alert_data == alerts


if __name__ == "__main__":
    pytest.main([__file__])