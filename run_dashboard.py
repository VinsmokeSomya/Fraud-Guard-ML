#!/usr/bin/env python3
"""
Main entry point for the Fraud Detection Dashboard.

This script launches the Streamlit dashboard for fraud detection analysis.
Run with: streamlit run run_dashboard.py
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the dashboard
from dashboard.fraud_dashboard import main

if __name__ == "__main__":
    main()