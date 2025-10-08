#!/usr/bin/env python3
"""
Startup script for the Fraud Detection API server.

This script initializes the fraud detection service and starts the FastAPI server
with proper configuration and model loading.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.fraud_api import initialize_fraud_detector, run_server
from src.utils.config import get_logger

logger = get_logger(__name__)


def main():
    """Main function to start the fraud detection API server."""
    parser = argparse.ArgumentParser(description="Fraud Detection API Server")
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Server host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        help="Path to trained fraud detection model"
    )
    parser.add_argument(
        "--risk-threshold", 
        type=float, 
        default=0.5, 
        help="Risk threshold for fraud classification (default: 0.5)"
    )
    parser.add_argument(
        "--high-risk-threshold", 
        type=float, 
        default=0.8, 
        help="High-risk threshold for alerts (default: 0.8)"
    )
    parser.add_argument(
        "--no-explanations", 
        action="store_true", 
        help="Disable detailed fraud explanations"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO", 
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Validate thresholds
        if not 0 <= args.risk_threshold <= 1:
            raise ValueError("Risk threshold must be between 0 and 1")
        if not 0 <= args.high_risk_threshold <= 1:
            raise ValueError("High-risk threshold must be between 0 and 1")
        if args.risk_threshold >= args.high_risk_threshold:
            logger.warning("Risk threshold should be lower than high-risk threshold")
        
        # Validate model path if provided
        if args.model_path and not Path(args.model_path).exists():
            logger.error(f"Model file not found: {args.model_path}")
            sys.exit(1)
        
        # Initialize fraud detector
        logger.info("Initializing fraud detection service...")
        initialize_fraud_detector(
            model_path=args.model_path,
            risk_threshold=args.risk_threshold,
            high_risk_threshold=args.high_risk_threshold,
            enable_explanations=not args.no_explanations
        )
        
        # Start server
        logger.info(f"Starting API server on {args.host}:{args.port}")
        logger.info(f"API documentation available at: http://{args.host}:{args.port}/docs")
        
        run_server(
            host=args.host,
            port=args.port,
            reload=args.reload
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()