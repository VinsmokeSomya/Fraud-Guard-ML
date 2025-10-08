#!/usr/bin/env python3
"""
Test script for the Fraud Detection API endpoints.

This script tests all the API endpoints to ensure they work correctly.
"""

import requests
import json
import time
from typing import Dict, Any


class FraudAPITester:
    """Test client for the Fraud Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester."""
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self) -> Dict[str, Any]:
        """Test the health check endpoint."""
        print("Testing health check endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ Health check passed: {result['status']}")
            return result
            
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return {}
    
    def test_root_endpoint(self) -> Dict[str, Any]:
        """Test the root endpoint."""
        print("Testing root endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ Root endpoint passed: {result['service']}")
            return result
            
        except Exception as e:
            print(f"✗ Root endpoint failed: {e}")
            return {}
    
    def test_single_prediction(self) -> Dict[str, Any]:
        """Test single transaction prediction."""
        print("Testing single transaction prediction...")
        
        # Sample transaction data
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "nameOrig": "C1231006815",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C1666544295",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=transaction
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ Single prediction passed: fraud_score={result['fraud_score']:.3f}, "
                  f"risk_level={result['risk_level']}")
            return result
            
        except Exception as e:
            print(f"✗ Single prediction failed: {e}")
            return {}
    
    def test_prediction_with_explanation(self) -> Dict[str, Any]:
        """Test single transaction prediction with explanation."""
        print("Testing prediction with explanation...")
        
        # Sample high-risk transaction
        transaction = {
            "step": 1,
            "type": "CASH-OUT",
            "amount": 229133.94,
            "nameOrig": "C905080434",
            "oldbalanceOrg": 15325.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C476402209",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/explain",
                json=transaction
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ Explanation passed: fraud_score={result['fraud_score']:.3f}, "
                  f"risk_factors={len(result['risk_factors']['high_risk_factors'])} high-risk factors")
            return result
            
        except Exception as e:
            print(f"✗ Explanation failed: {e}")
            return {}
    
    def test_batch_prediction(self) -> Dict[str, Any]:
        """Test batch transaction prediction."""
        print("Testing batch prediction...")
        
        # Sample batch of transactions
        transactions = [
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 9839.64,
                "nameOrig": "C1231006815",
                "oldbalanceOrg": 170136.0,
                "newbalanceOrig": 160296.36,
                "nameDest": "M1979787155",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0
            },
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "nameOrig": "C1231006815",
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C1666544295",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0
            },
            {
                "step": 1,
                "type": "CASH-OUT",
                "amount": 229133.94,
                "nameOrig": "C905080434",
                "oldbalanceOrg": 15325.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C476402209",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0
            }
        ]
        
        batch_request = {"transactions": transactions}
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_request
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ Batch prediction passed: {result['total_transactions']} transactions, "
                  f"{result['fraud_detected']} fraud detected")
            return result
            
        except Exception as e:
            print(f"✗ Batch prediction failed: {e}")
            return {}
    
    def test_threshold_update(self) -> Dict[str, Any]:
        """Test threshold update endpoint."""
        print("Testing threshold update...")
        
        # Update thresholds
        threshold_update = {
            "risk_threshold": 0.6,
            "high_risk_threshold": 0.85
        }
        
        try:
            response = self.session.put(
                f"{self.base_url}/config/thresholds",
                json=threshold_update
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ Threshold update passed: risk={result['risk_threshold']}, "
                  f"high_risk={result['high_risk_threshold']}")
            
            # Reset to original values
            reset_thresholds = {
                "risk_threshold": 0.5,
                "high_risk_threshold": 0.8
            }
            self.session.put(
                f"{self.base_url}/config/thresholds",
                json=reset_thresholds
            )
            
            return result
            
        except Exception as e:
            print(f"✗ Threshold update failed: {e}")
            return {}
    
    def test_service_status(self) -> Dict[str, Any]:
        """Test service status endpoint."""
        print("Testing service status...")
        
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ Service status passed: {result['service_name']}")
            return result
            
        except Exception as e:
            print(f"✗ Service status failed: {e}")
            return {}
    
    def run_all_tests(self) -> None:
        """Run all API tests."""
        print("=" * 60)
        print("FRAUD DETECTION API TESTS")
        print("=" * 60)
        
        # Wait for server to be ready
        print("Waiting for server to be ready...")
        max_retries = 10
        for i in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    break
            except:
                pass
            
            if i == max_retries - 1:
                print("✗ Server is not responding. Please start the API server first.")
                return
            
            time.sleep(1)
        
        print("Server is ready. Running tests...\n")
        
        # Run tests
        tests = [
            self.test_root_endpoint,
            self.test_health_check,
            self.test_service_status,
            self.test_single_prediction,
            self.test_prediction_with_explanation,
            self.test_batch_prediction,
            self.test_threshold_update
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                result = test()
                if result:
                    passed += 1
                print()
            except Exception as e:
                print(f"✗ Test failed with exception: {e}\n")
        
        print("=" * 60)
        print(f"RESULTS: {passed}/{total} tests passed")
        print("=" * 60)
        
        if passed == total:
            print("🎉 All tests passed! The API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")


def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Fraud Detection API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000", 
        help="API base URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    tester = FraudAPITester(args.url)
    tester.run_all_tests()


if __name__ == "__main__":
    main()