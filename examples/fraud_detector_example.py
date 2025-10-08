"""
Example usage of the FraudDetector service class.

This script demonstrates how to use the FraudDetector for real-time transaction scoring,
batch predictions, and fraud explanation generation.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.services.fraud_detector import FraudDetector


def create_sample_transactions():
    """Create sample transaction data for demonstration."""
    transactions = [
        {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 250000.0,
            'nameOrig': 'C1234567890',
            'oldbalanceOrg': 300000.0,
            'newbalanceOrig': 50000.0,
            'nameDest': 'C9876543210',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        },
        {
            'step': 2,
            'type': 'PAYMENT',
            'amount': 1500.0,
            'nameOrig': 'C1111111111',
            'oldbalanceOrg': 5000.0,
            'newbalanceOrig': 3500.0,
            'nameDest': 'M2222222222',
            'oldbalanceDest': 10000.0,
            'newbalanceDest': 11500.0
        },
        {
            'step': 3,
            'type': 'CASH_OUT',
            'amount': 50000.0,
            'nameOrig': 'C3333333333',
            'oldbalanceOrg': 50000.0,
            'newbalanceOrig': 0.0,
            'nameDest': 'C4444444444',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }
    ]
    
    return transactions


def demonstrate_real_time_scoring():
    """Demonstrate real-time transaction scoring."""
    print("=== Real-time Transaction Scoring Demo ===")
    
    # Initialize FraudDetector (without a model for demo purposes)
    detector = FraudDetector(
        model=None,  # In real usage, you'd load a trained model
        risk_threshold=0.5,
        high_risk_threshold=0.8,
        enable_explanations=True
    )
    
    # Create sample transactions
    transactions = create_sample_transactions()
    
    print(f"Processing {len(transactions)} transactions...\n")
    
    for i, transaction in enumerate(transactions, 1):
        print(f"Transaction {i}:")
        print(f"  Type: {transaction['type']}")
        print(f"  Amount: ${transaction['amount']:,.2f}")
        print(f"  Origin Balance: ${transaction['oldbalanceOrg']:,.2f} -> ${transaction['newbalanceOrig']:,.2f}")
        
        try:
            # Note: This will fail without a loaded model, but shows the interface
            fraud_score = detector.score_transaction(transaction)
            print(f"  Fraud Score: {fraud_score:.3f}")
        except ValueError as e:
            print(f"  Fraud Score: Cannot calculate (no model loaded)")
        
        print()


def demonstrate_batch_prediction():
    """Demonstrate batch prediction functionality."""
    print("=== Batch Prediction Demo ===")
    
    # Initialize FraudDetector
    detector = FraudDetector(
        model=None,  # In real usage, you'd load a trained model
        risk_threshold=0.5,
        high_risk_threshold=0.8
    )
    
    # Create sample transaction DataFrame
    transactions = create_sample_transactions()
    df = pd.DataFrame(transactions)
    
    print(f"Processing batch of {len(df)} transactions...\n")
    print("Input DataFrame:")
    print(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']].to_string())
    print()
    
    try:
        # Note: This will fail without a loaded model, but shows the interface
        result_df = detector.batch_predict(df)
        print("Results with fraud scores:")
        print(result_df[['type', 'amount', 'fraud_score', 'fraud_prediction', 'risk_level']].to_string())
    except ValueError as e:
        print(f"Batch prediction failed: {e}")
        print("(This is expected without a loaded model)")
    
    print()


def demonstrate_fraud_explanation():
    """Demonstrate fraud explanation functionality."""
    print("=== Fraud Explanation Demo ===")
    
    # Initialize FraudDetector with explanations enabled
    detector = FraudDetector(
        model=None,  # In real usage, you'd load a trained model
        risk_threshold=0.5,
        high_risk_threshold=0.8,
        enable_explanations=True
    )
    
    # Create a high-risk transaction for demonstration
    high_risk_transaction = {
        'step': 1440,  # Late night hour (24 * 60 = 1440 minutes, hour 0)
        'type': 'TRANSFER',
        'amount': 500000.0,  # Large amount
        'nameOrig': 'C1234567890',
        'oldbalanceOrg': 500000.0,
        'newbalanceOrig': 0.0,  # Complete balance depletion
        'nameDest': 'C9876543210',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0  # Suspicious destination balance pattern
    }
    
    print("Analyzing high-risk transaction:")
    print(f"  Type: {high_risk_transaction['type']}")
    print(f"  Amount: ${high_risk_transaction['amount']:,.2f}")
    print(f"  Origin Balance: ${high_risk_transaction['oldbalanceOrg']:,.2f} -> ${high_risk_transaction['newbalanceOrig']:,.2f}")
    print(f"  Time Step: {high_risk_transaction['step']} (Hour: {high_risk_transaction['step'] % 24})")
    print()
    
    try:
        # Note: This will fail at scoring but will show risk factor analysis
        explanation = detector.get_fraud_explanation(high_risk_transaction)
        print("Fraud Explanation:")
        print(f"  Risk Level: {explanation.get('risk_level', 'Unknown')}")
        print(f"  Confidence: {explanation.get('confidence', 0):.3f}")
        print()
        print("Risk Factors:")
        risk_factors = explanation.get('risk_factors', {})
        
        if risk_factors.get('high_risk_factors'):
            print("  High Risk Factors:")
            for factor in risk_factors['high_risk_factors']:
                print(f"    • {factor}")
        
        if risk_factors.get('medium_risk_factors'):
            print("  Medium Risk Factors:")
            for factor in risk_factors['medium_risk_factors']:
                print(f"    • {factor}")
        
        if risk_factors.get('low_risk_factors'):
            print("  Low Risk Factors:")
            for factor in risk_factors['low_risk_factors']:
                print(f"    • {factor}")
        
        print()
        print("Recommendations:")
        recommendations = explanation.get('recommendations', [])
        for rec in recommendations:
            print(f"  • {rec}")
            
    except ValueError as e:
        print(f"Explanation generation failed at scoring step: {e}")
        print("(This is expected without a loaded model)")
        
        # Show risk factor analysis that works without model
        risk_factors = detector._analyze_transaction_risk_factors(high_risk_transaction)
        print("\nRisk Factor Analysis (model-independent):")
        
        if risk_factors.get('high_risk_factors'):
            print("  High Risk Factors:")
            for factor in risk_factors['high_risk_factors']:
                print(f"    • {factor}")
        
        if risk_factors.get('medium_risk_factors'):
            print("  Medium Risk Factors:")
            for factor in risk_factors['medium_risk_factors']:
                print(f"    • {factor}")
        
        print(f"\n  Risk Scores:")
        for risk_type, score in risk_factors.get('risk_scores', {}).items():
            print(f"    {risk_type}: {score:.3f}")
    
    print()


def demonstrate_service_status():
    """Demonstrate service status functionality."""
    print("=== Service Status Demo ===")
    
    # Initialize FraudDetector
    detector = FraudDetector(
        model=None,
        risk_threshold=0.6,
        high_risk_threshold=0.85,
        enable_explanations=True
    )
    
    # Get service status
    status = detector.get_service_status()
    
    print("FraudDetector Service Status:")
    for key, value in status.items():
        if key != 'model_info':
            print(f"  {key}: {value}")
    
    print()
    
    # Demonstrate threshold updates
    print("Updating thresholds...")
    detector.update_thresholds(risk_threshold=0.4, high_risk_threshold=0.7)
    
    updated_status = detector.get_service_status()
    print("Updated thresholds:")
    print(f"  risk_threshold: {updated_status['risk_threshold']}")
    print(f"  high_risk_threshold: {updated_status['high_risk_threshold']}")
    
    print()


def main():
    """Run all demonstration examples."""
    print("FraudDetector Service Demonstration")
    print("=" * 50)
    print()
    
    # Note: These demos show the interface and functionality
    # In real usage, you would load a trained model first
    
    demonstrate_real_time_scoring()
    demonstrate_batch_prediction()
    demonstrate_fraud_explanation()
    demonstrate_service_status()
    
    print("Demo completed!")
    print("\nNote: To use with real fraud detection, load a trained model:")
    print("  detector = FraudDetector()")
    print("  detector.load_model('path/to/trained/model')")
    print("  # or pass model directly:")
    print("  detector = FraudDetector(model=your_trained_model)")


if __name__ == "__main__":
    main()