"""
Unit tests for FraudPatternAnalyzer module.

This module contains tests for the fraud pattern analysis utilities including
time series analysis, customer behavior analysis, and risk factor identification.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from visualization.fraud_pattern_analyzer import FraudPatternAnalyzer


class TestFraudPatternAnalyzer(unittest.TestCase):
    """Test cases for FraudPatternAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FraudPatternAnalyzer()
        
        # Create sample test data
        np.random.seed(42)
        n_samples = 1000
        
        self.test_data = pd.DataFrame({
            'step': np.random.randint(1, 745, n_samples),
            'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'TRANSFER'], n_samples),
            'amount': np.random.lognormal(8, 1, n_samples),
            'nameOrig': [f'C{i}' for i in range(n_samples)],
            'nameDest': [f'C{i+n_samples}' if i % 10 != 0 else f'M{i+n_samples}' 
                        for i in range(n_samples)],
            'oldbalanceOrg': np.random.exponential(10000, n_samples),
            'newbalanceOrig': np.random.exponential(8000, n_samples),
            'oldbalanceDest': np.random.exponential(5000, n_samples),
            'newbalanceDest': np.random.exponential(6000, n_samples),
            'isFraud': np.random.binomial(1, 0.05, n_samples)  # 5% fraud rate
        })
    
    def test_initialization(self):
        """Test FraudPatternAnalyzer initialization."""
        analyzer = FraudPatternAnalyzer(style='darkgrid', palette='viridis')
        self.assertEqual(analyzer.style, 'darkgrid')
        self.assertEqual(analyzer.palette, 'viridis')
    
    def test_plot_fraud_time_series_hourly(self):
        """Test fraud time series plotting with hourly aggregation."""
        result = self.analyzer.plot_fraud_time_series(
            self.test_data, time_unit='hour', aggregation='count', interactive=False
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('time_data', result)
        self.assertIn('fraud_data', result)
        self.assertIn('xlabel', result)
        self.assertIn('ylabel', result)
        self.assertEqual(len(result['time_data']), 24)  # 24 hours
        self.assertEqual(result['xlabel'], 'Hour of Day')
    
    def test_plot_fraud_time_series_daily(self):
        """Test fraud time series plotting with daily aggregation."""
        result = self.analyzer.plot_fraud_time_series(
            self.test_data, time_unit='day', aggregation='rate', interactive=False
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('time_data', result)
        self.assertIn('fraud_data', result)
        self.assertEqual(result['xlabel'], 'Day')
        self.assertTrue(all(rate >= 0 for rate in result['fraud_data']))  # Rates should be non-negative
    
    def test_plot_fraud_time_series_missing_columns(self):
        """Test fraud time series with missing required columns."""
        # Test missing isFraud column
        df_no_fraud = self.test_data.drop('isFraud', axis=1)
        result = self.analyzer.plot_fraud_time_series(df_no_fraud, interactive=False)
        self.assertIn('error', result)
        
        # Test missing step column
        df_no_step = self.test_data.drop('step', axis=1)
        result = self.analyzer.plot_fraud_time_series(df_no_step, interactive=False)
        self.assertIn('error', result)
    
    def test_analyze_customer_behavior(self):
        """Test customer behavior analysis."""
        result = self.analyzer.analyze_customer_behavior(self.test_data, interactive=False)
        
        self.assertIsInstance(result, dict)
        self.assertIn('unusual_balance_fraud_rate', result)
        self.assertIn('merchant_dest_fraud_rate', result)
        self.assertIn('customer_dest_fraud_rate', result)
        self.assertIn('balance_change_stats', result)
        
        # Check that fraud rates are between 0 and 1
        self.assertTrue(0 <= result['unusual_balance_fraud_rate'] <= 1)
        self.assertTrue(0 <= result['merchant_dest_fraud_rate'] <= 1)
        self.assertTrue(0 <= result['customer_dest_fraud_rate'] <= 1)
    
    def test_analyze_customer_behavior_missing_columns(self):
        """Test customer behavior analysis with missing columns."""
        # Remove required columns
        df_incomplete = self.test_data.drop(['nameOrig', 'nameDest'], axis=1)
        result = self.analyzer.analyze_customer_behavior(df_incomplete, interactive=False)
        
        self.assertIn('error', result)
    
    def test_identify_risk_factors(self):
        """Test risk factor identification and ranking."""
        result = self.analyzer.identify_risk_factors(self.test_data, top_n=10)
        
        self.assertIsInstance(result, dict)
        self.assertIn('risk_factors', result)
        self.assertIn('ranked_factors', result)
        self.assertIn('summary_stats', result)
        
        # Check ranked factors structure
        ranked_factors = result['ranked_factors']
        self.assertIsInstance(ranked_factors, list)
        self.assertTrue(len(ranked_factors) <= 10)  # Should respect top_n limit
        
        if ranked_factors:
            # Check that factors are sorted by fraud rate (descending)
            fraud_rates = [factor['fraud_rate'] for factor in ranked_factors]
            self.assertEqual(fraud_rates, sorted(fraud_rates, reverse=True))
            
            # Check factor structure
            first_factor = ranked_factors[0]
            required_keys = ['factor_type', 'factor_name', 'fraud_rate', 'fraud_count', 'total_count']
            for key in required_keys:
                self.assertIn(key, first_factor)
        
        # Check summary stats
        summary = result['summary_stats']
        self.assertIn('total_risk_factors', summary)
        self.assertIn('average_fraud_rate', summary)
        self.assertIsInstance(summary['total_risk_factors'], int)
        self.assertTrue(0 <= summary['average_fraud_rate'] <= 1)
    
    def test_identify_risk_factors_missing_fraud_column(self):
        """Test risk factor identification with missing fraud column."""
        df_no_fraud = self.test_data.drop('isFraud', axis=1)
        result = self.analyzer.identify_risk_factors(df_no_fraud)
        
        self.assertIn('error', result)
    
    def test_create_comprehensive_fraud_analysis(self):
        """Test comprehensive fraud analysis."""
        result = self.analyzer.create_comprehensive_fraud_analysis(
            self.test_data, interactive=False, save_path=None
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('time_series_hourly', result)
        self.assertIn('time_series_daily', result)
        self.assertIn('customer_behavior', result)
        self.assertIn('risk_factors', result)
        
        # Verify each component has expected structure
        self.assertIn('time_data', result['time_series_hourly'])
        self.assertIn('fraud_data', result['time_series_hourly'])
        
        self.assertIn('unusual_balance_fraud_rate', result['customer_behavior'])
        
        self.assertIn('ranked_factors', result['risk_factors'])
        self.assertIn('summary_stats', result['risk_factors'])
    
    def test_large_transfer_risk_identification(self):
        """Test identification of large transfer risk (business rule >200,000)."""
        # Create data with some large transfers
        large_transfer_data = self.test_data.copy()
        large_transfer_data.loc[:50, 'amount'] = 250000  # Make first 51 transactions large
        large_transfer_data.loc[:50, 'type'] = 'TRANSFER'
        large_transfer_data.loc[:25, 'isFraud'] = 1  # Make half of large transfers fraudulent
        
        result = self.analyzer.identify_risk_factors(large_transfer_data, top_n=15)
        
        # Check if large transfers are identified as a risk factor
        risk_factors = result['risk_factors']
        self.assertIn('large_transfers', risk_factors)
        
        large_transfer_risk = risk_factors['large_transfers']
        self.assertEqual(large_transfer_risk['threshold'], 200000)
        self.assertGreater(large_transfer_risk['fraud_rate'], 0)
        self.assertGreater(large_transfer_risk['total_count'], 0)
    
    def test_merchant_vs_customer_pattern_analysis(self):
        """Test merchant vs customer pattern analysis."""
        result = self.analyzer.analyze_customer_behavior(self.test_data, interactive=False)
        
        # Should have both merchant and customer destination fraud rates
        self.assertIn('merchant_dest_fraud_rate', result)
        self.assertIn('customer_dest_fraud_rate', result)
        
        # Both should be valid probabilities
        self.assertTrue(0 <= result['merchant_dest_fraud_rate'] <= 1)
        self.assertTrue(0 <= result['customer_dest_fraud_rate'] <= 1)
    
    def test_time_based_risk_patterns(self):
        """Test time-based risk pattern identification."""
        result = self.analyzer.identify_risk_factors(self.test_data, top_n=20)
        
        risk_factors = result['risk_factors']
        
        # Should identify hourly patterns
        if 'hourly_patterns' in risk_factors:
            hourly_patterns = risk_factors['hourly_patterns']
            self.assertIsInstance(hourly_patterns, list)
            
            if hourly_patterns:
                # Check structure of hourly pattern
                first_pattern = hourly_patterns[0]
                self.assertIn('hour', first_pattern)
                self.assertIn('fraud_rate', first_pattern)
                self.assertTrue(0 <= first_pattern['hour'] <= 23)
                self.assertTrue(0 <= first_pattern['fraud_rate'] <= 1)


if __name__ == '__main__':
    unittest.main()