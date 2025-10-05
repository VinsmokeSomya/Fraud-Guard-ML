"""
Main entry point for the fraud detection system.
"""
from config.logging_config import get_logger
from src.utils.config import validate_paths, get_config

logger = get_logger(__name__)


def main():
    """Main function to initialize and verify the system setup."""
    logger.info("Starting Fraud Detection System")
    
    # Validate configuration
    config = get_config()
    logger.info("Configuration loaded successfully")
    
    # Validate paths
    if validate_paths():
        logger.info("All required directories are present")
    else:
        logger.warning("Some directories are missing - they will be created automatically")
    
    logger.info("System initialization complete")


if __name__ == "__main__":
    main()