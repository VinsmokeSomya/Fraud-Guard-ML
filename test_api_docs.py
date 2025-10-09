#!/usr/bin/env python3
"""
Test script to verify API documentation and endpoints.

This script tests the fraud detection API endpoints and verifies
that the documentation is properly generated.
"""

import requests
import json
import sys
from pathlib import Path

def test_api_documentation(base_url="http://localhost:8000"):
    """Test API documentation endpoints."""
    
    print("🔍 Testing Fraud Detection API Documentation...")
    print(f"Base URL: {base_url}")
    print("-" * 50)
    
    # Test endpoints to verify
    endpoints = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/docs", "Swagger UI documentation"),
        ("/redoc", "ReDoc documentation"),
        ("/openapi.json", "OpenAPI schema")
    ]
    
    results = []
    
    for endpoint, description in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ {description}: {url} - OK")
                results.append(True)
            else:
                print(f"❌ {description}: {url} - Status {response.status_code}")
                results.append(False)
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {description}: {url} - Connection failed (API not running?)")
            results.append(False)
        except Exception as e:
            print(f"❌ {description}: {url} - Error: {e}")
            results.append(False)
    
    print("-" * 50)
    
    # Summary
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print(f"🎉 All {total_count} endpoints working correctly!")
        print("\n📖 Access the documentation at:")
        print(f"   • Swagger UI: {base_url}/docs")
        print(f"   • ReDoc: {base_url}/redoc")
        return True
    else:
        print(f"⚠️  {success_count}/{total_count} endpoints working")
        print("\n💡 To start the API server:")
        print("   python run_api.py")
        return False

def test_sample_prediction(base_url="http://localhost:8000"):
    """Test a sample fraud prediction."""
    
    print("\n🧪 Testing Sample Fraud Prediction...")
    print("-" * 50)
    
    # Sample transaction data
    sample_transaction = {
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
        # Test prediction endpoint
        url = f"{base_url}/predict"
        response = requests.post(url, json=sample_transaction, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Fraud Score: {result.get('fraud_score', 'N/A')}")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
            print(f"   Is Fraud: {result.get('is_fraud_prediction', 'N/A')}")
            return True
        elif response.status_code == 503:
            print(f"⚠️  Service unavailable - Model not loaded")
            print(f"   Start API with: python run_api.py --model-path models/your_model.joblib")
            return False
        else:
            print(f"❌ Prediction failed - Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed - API not running")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function."""
    
    print("🚀 Fraud Detection API Documentation Test")
    print("=" * 60)
    
    # Check if API is running
    base_url = "http://localhost:8000"
    
    # Test documentation endpoints
    docs_ok = test_api_documentation(base_url)
    
    # Test sample prediction if docs are working
    if docs_ok:
        test_sample_prediction(base_url)
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    
    if docs_ok:
        print("✅ API documentation is working correctly")
        print("✅ All endpoints are accessible")
        print("\n🎯 Next Steps:")
        print("1. Visit the interactive documentation:")
        print(f"   • Swagger UI: {base_url}/docs")
        print(f"   • ReDoc: {base_url}/redoc")
        print("2. Try the API endpoints using the interactive interface")
        print("3. Review the comprehensive API guide: docs/API_GUIDE.md")
    else:
        print("❌ Some issues found with API documentation")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure the API server is running:")
        print("   python run_api.py")
        print("2. Check if the port 8000 is available")
        print("3. Review the API guide: docs/API_GUIDE.md")

if __name__ == "__main__":
    main()