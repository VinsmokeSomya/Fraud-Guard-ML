#!/usr/bin/env python3
"""
Performance Test Runner for Fraud Detection System

This script runs comprehensive performance and load tests including:
- Large dataset processing performance
- Real-time prediction latency validation
- Concurrent API request handling
- Resource utilization monitoring
- K6 load testing (if available)

Usage:
    python run_performance_tests.py [options]

Requirements tested: 6.2, 6.3
"""

import argparse
import subprocess
import sys
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceTestRunner:
    """Orchestrates performance testing for the fraud detection system."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.test_results = {}
        self.start_time = time.time()
        
    def run_python_performance_tests(self, test_categories: List[str] = None) -> Dict:
        """Run Python-based performance tests."""
        logger.info("Starting Python performance tests...")
        
        # Default test categories
        if test_categories is None:
            test_categories = [
                "TestLargeDatasetPerformance",
                "TestRealTimePredictionLatency", 
                "TestConcurrentAPIPerformance",
                "TestResourceUtilization"
            ]
        
        results = {}
        
        for category in test_categories:
            logger.info(f"Running test category: {category}")
            
            try:
                # Run specific test class
                cmd = [
                    sys.executable, "-m", "pytest", 
                    f"tests/test_performance_load.py::{category}",
                    "-v", "-s", "--tb=short",
                    "--durations=10"  # Show 10 slowest tests
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=1800  # 30 minute timeout
                )
                
                results[category] = {
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "success": result.returncode == 0
                }
                
                if result.returncode == 0:
                    logger.info(f"✓ {category} completed successfully")
                else:
                    logger.error(f"✗ {category} failed with return code {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"✗ {category} timed out after 30 minutes")
                results[category] = {
                    "return_code": -1,
                    "error": "Test timed out",
                    "success": False
                }
            except Exception as e:
                logger.error(f"✗ {category} failed with exception: {e}")
                results[category] = {
                    "return_code": -1,
                    "error": str(e),
                    "success": False
                }
        
        return results
    
    def run_k6_load_tests(self, test_scenarios: List[str] = None) -> Dict:
        """Run K6 load tests if K6 is available."""
        logger.info("Starting K6 load tests...")
        
        # Check if K6 is available
        try:
            subprocess.run(["k6", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("K6 not found. Skipping load tests. Install K6 from https://k6.io/")
            return {"k6_available": False, "message": "K6 not installed"}
        
        # Default scenarios
        if test_scenarios is None:
            test_scenarios = ["light_load", "medium_load", "heavy_load", "spike_test"]
        
        results = {}
        test_script = Path("tests/load_test.js")
        
        if not test_script.exists():
            logger.error("Load test script not found: tests/load_test.js")
            return {"error": "Load test script not found"}
        
        for scenario in test_scenarios:
            logger.info(f"Running K6 scenario: {scenario}")
            
            try:
                cmd = [
                    "k6", "run",
                    "--env", f"API_URL={self.api_url}",
                    "--env", f"EXEC_SCENARIO={scenario}",
                    "--out", f"json=k6_results_{scenario}.json",
                    str(test_script)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                results[scenario] = {
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "success": result.returncode == 0
                }
                
                # Try to parse K6 results
                results_file = f"k6_results_{scenario}.json"
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r') as f:
                            k6_data = [json.loads(line) for line in f if line.strip()]
                        results[scenario]["k6_metrics"] = self._parse_k6_metrics(k6_data)
                    except Exception as e:
                        logger.warning(f"Could not parse K6 results for {scenario}: {e}")
                
                if result.returncode == 0:
                    logger.info(f"✓ K6 scenario {scenario} completed successfully")
                else:
                    logger.error(f"✗ K6 scenario {scenario} failed")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"✗ K6 scenario {scenario} timed out")
                results[scenario] = {
                    "return_code": -1,
                    "error": "Test timed out",
                    "success": False
                }
            except Exception as e:
                logger.error(f"✗ K6 scenario {scenario} failed: {e}")
                results[scenario] = {
                    "return_code": -1,
                    "error": str(e),
                    "success": False
                }
        
        return results
    
    def _parse_k6_metrics(self, k6_data: List[Dict]) -> Dict:
        """Parse K6 JSON output to extract key metrics."""
        metrics = {}
        
        for entry in k6_data:
            if entry.get("type") == "Point" and "data" in entry:
                metric_name = entry["data"].get("name")
                value = entry["data"].get("value")
                
                if metric_name and value is not None:
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(value)
        
        # Calculate summary statistics
        summary = {}
        for metric, values in metrics.items():
            if values:
                summary[metric] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0]
                }
        
        return summary
    
    def check_api_availability(self) -> bool:
        """Check if the fraud detection API is running."""
        try:
            import requests
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code in [200, 503]
        except Exception as e:
            logger.warning(f"API not available at {self.api_url}: {e}")
            return False
    
    def generate_performance_report(self) -> Dict:
        """Generate a comprehensive performance test report."""
        total_time = time.time() - self.start_time
        
        report = {
            "test_summary": {
                "total_duration_seconds": round(total_time, 2),
                "api_url": self.api_url,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        # Check Python test results
        python_results = self.test_results.get("python_tests", {})
        failed_tests = [name for name, result in python_results.items() if not result.get("success", False)]
        
        if failed_tests:
            recommendations.append(f"Address failing performance tests: {', '.join(failed_tests)}")
        
        # Check K6 results
        k6_results = self.test_results.get("k6_tests", {})
        if k6_results.get("k6_available", True):
            failed_scenarios = [name for name, result in k6_results.items() 
                              if isinstance(result, dict) and not result.get("success", False)]
            
            if failed_scenarios:
                recommendations.append(f"Investigate load test failures: {', '.join(failed_scenarios)}")
        
        # General recommendations
        recommendations.extend([
            "Monitor API response times during peak usage",
            "Consider implementing caching for frequently requested predictions",
            "Set up automated performance monitoring in production",
            "Review resource allocation if tests show high latency"
        ])
        
        return recommendations
    
    def save_report(self, filename: str = None) -> str:
        """Save the performance report to a file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_report_{timestamp}.json"
        
        report = self.generate_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to: {filename}")
        return filename


def main():
    """Main function to run performance tests."""
    parser = argparse.ArgumentParser(description="Run fraud detection system performance tests")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="API URL to test (default: http://localhost:8000)")
    parser.add_argument("--python-only", action="store_true", 
                       help="Run only Python performance tests")
    parser.add_argument("--k6-only", action="store_true", 
                       help="Run only K6 load tests")
    parser.add_argument("--scenarios", nargs="+", 
                       help="Specific K6 scenarios to run")
    parser.add_argument("--test-categories", nargs="+",
                       help="Specific Python test categories to run")
    parser.add_argument("--skip-api-check", action="store_true",
                       help="Skip API availability check")
    parser.add_argument("--output", help="Output file for test report")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = PerformanceTestRunner(api_url=args.api_url)
    
    # Check API availability unless skipped
    if not args.skip_api_check:
        logger.info(f"Checking API availability at {args.api_url}...")
        if not runner.check_api_availability():
            logger.error("API is not available. Start the fraud detection service first.")
            logger.info("Run: python run_api.py")
            return 1
        logger.info("✓ API is available")
    
    # Run tests based on arguments
    try:
        if not args.k6_only:
            logger.info("=" * 60)
            logger.info("RUNNING PYTHON PERFORMANCE TESTS")
            logger.info("=" * 60)
            python_results = runner.run_python_performance_tests(args.test_categories)
            runner.test_results["python_tests"] = python_results
        
        if not args.python_only:
            logger.info("=" * 60)
            logger.info("RUNNING K6 LOAD TESTS")
            logger.info("=" * 60)
            k6_results = runner.run_k6_load_tests(args.scenarios)
            runner.test_results["k6_tests"] = k6_results
        
        # Generate and save report
        logger.info("=" * 60)
        logger.info("GENERATING PERFORMANCE REPORT")
        logger.info("=" * 60)
        
        report_file = runner.save_report(args.output)
        report = runner.generate_performance_report()
        
        # Print summary
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        print(f"Total Duration: {report['test_summary']['total_duration_seconds']} seconds")
        print(f"API URL: {report['test_summary']['api_url']}")
        print(f"Report saved to: {report_file}")
        
        # Print recommendations
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        # Determine exit code based on results
        all_success = True
        
        for test_type, results in runner.test_results.items():
            if isinstance(results, dict):
                for test_name, result in results.items():
                    if isinstance(result, dict) and not result.get("success", False):
                        all_success = False
                        break
        
        return 0 if all_success else 1
        
    except KeyboardInterrupt:
        logger.info("Performance tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Performance tests failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)