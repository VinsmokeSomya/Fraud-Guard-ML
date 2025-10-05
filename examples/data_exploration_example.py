"""
Example usage of DataExplorer for fraud detection dataset analysis.

This example demonstrates how to use the DataExplorer class to analyze
fraud detection datasets and generate comprehensive reports.
"""

from src.data import DataLoader, DataExplorer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Example usage of DataExplorer."""
    
    # Initialize components
    data_loader = DataLoader()
    explorer = DataExplorer()
    
    # Load your fraud detection dataset
    # Replace 'path/to/your/dataset.csv' with actual path
    try:
        # df = data_loader.load_data('data/raw/fraud_dataset.csv')
        print("To use this example:")
        print("1. Place your fraud detection CSV file in the data/raw/ directory")
        print("2. Uncomment the line above and update the file path")
        print("3. Run this script to generate comprehensive data exploration reports")
        print()
        print("The DataExplorer provides the following analysis functions:")
        print("- display_dataset_info(): Basic dataset statistics and information")
        print("- analyze_transaction_types(): Transaction type distribution and patterns")
        print("- analyze_amount_distribution(): Transaction amount analysis")
        print("- calculate_fraud_ratio(): Fraud rate and class imbalance analysis")
        print("- detect_missing_values(): Missing value detection and patterns")
        print("- generate_data_quality_report(): Comprehensive quality report")
        
        # Example usage (uncomment when you have data):
        """
        # Generate comprehensive data quality report
        quality_report = explorer.generate_data_quality_report(df)
        
        # Or use individual analysis functions
        dataset_info = explorer.display_dataset_info(df)
        transaction_analysis = explorer.analyze_transaction_types(df)
        amount_analysis = explorer.analyze_amount_distribution(df)
        fraud_analysis = explorer.calculate_fraud_ratio(df)
        missing_analysis = explorer.detect_missing_values(df)
        """
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your dataset file exists and has the correct format.")

if __name__ == "__main__":
    main()