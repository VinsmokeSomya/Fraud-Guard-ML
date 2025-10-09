#!/usr/bin/env python3
"""
Integration test runner for fraud detection system.

This script runs the comprehensive integration tests and provides
detailed reporting on test results and coverage.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_tests(test_type="all", verbose=False, coverage=False, parallel=False):
    """
    Run integration tests with specified options.
    
    Args:
        test_type: Type of tests to run ("all", "e2e", "api", "performance")
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        parallel: Run tests in parallel
    """
    print("🚀 Starting Fraud Detection Integration Tests")
    print("=" * 60)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test file
    test_file = Path(__file__).parent / "test_integration_e2e.py"
    cmd.append(str(test_file))
    
    # Add markers based on test type
    if test_type == "e2e":
        cmd.extend(["-m", "e2e"])
    elif test_type == "api":
        cmd.extend(["-m", "api"])
    elif test_type == "performance":
        cmd.extend(["-m", "slow"])
    elif test_type != "all":
        print(f"❌ Unknown test type: {test_type}")
        return False
    
    # Add verbose output
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution
    if parallel:
        try:
            import pytest_xdist
            cmd.extend(["-n", "auto"])
        except ImportError:
            print("⚠️  pytest-xdist not installed, running sequentially")
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings"  # Disable warnings for cleaner output
    ])
    
    print(f"📋 Running command: {' '.join(cmd)}")
    print()
    
    # Run tests
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        end_time = time.time()
        
        print()
        print("=" * 60)
        print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            print("✅ All integration tests passed!")
            
            if coverage:
                print("📊 Coverage report generated in htmlcov/index.html")
            
            return True
        else:
            print("❌ Some integration tests failed!")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "pytest",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "fastapi",
        "uvicorn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed")
    return True


def setup_test_environment():
    """Set up test environment and create necessary directories."""
    print("🛠️  Setting up test environment...")
    
    # Create necessary directories
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Create logs directory if it doesn't exist
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create models directory if it doesn't exist
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create reports directory if it doesn't exist
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    print("✅ Test environment ready")
    return True


def main():
    """Main function to run integration tests."""
    parser = argparse.ArgumentParser(
        description="Run fraud detection integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_integration_tests.py                    # Run all tests
  python run_integration_tests.py --type e2e         # Run only end-to-end tests
  python run_integration_tests.py --type api -v      # Run API tests with verbose output
  python run_integration_tests.py --coverage         # Run with coverage reporting
  python run_integration_tests.py --parallel         # Run tests in parallel
        """
    )
    
    parser.add_argument(
        "--type",
        choices=["all", "e2e", "api", "performance"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency checking"
    )
    
    args = parser.parse_args()
    
    print("🧪 Fraud Detection System - Integration Test Runner")
    print("=" * 60)
    
    # Check dependencies
    if not args.skip_deps and not check_dependencies():
        sys.exit(1)
    
    # Set up test environment
    if not setup_test_environment():
        sys.exit(1)
    
    # Run tests
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )
    
    if success:
        print("\n🎉 Integration tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Integration tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()