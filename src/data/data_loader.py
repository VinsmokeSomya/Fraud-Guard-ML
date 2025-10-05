"""
DataLoader module for loading and validating fraud detection datasets.

This module provides the DataLoader class for efficiently reading large CSV files
with proper validation and error handling.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader class for reading CSV files efficiently with validation and error handling.
    
    This class handles large datasets by supporting chunked reading, validates data schema,
    and provides robust error handling for file operations.
    """
    
    # Expected schema for fraud detection dataset
    EXPECTED_SCHEMA = {
        'step': 'int64',
        'type': 'object',
        'amount': 'float64',
        'nameOrig': 'object',
        'oldbalanceOrg': 'float64',
        'newbalanceOrig': 'float64',
        'nameDest': 'object',
        'oldbalanceDest': 'float64',
        'newbalanceDest': 'float64',
        'isFraud': 'int64',
        'isFlaggedFraud': 'int64'
    }
    
    EXPECTED_TRANSACTION_TYPES = {'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'}
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize DataLoader with specified chunk size.
        
        Args:
            chunk_size: Number of rows to read per chunk for large files
        """
        self.chunk_size = chunk_size
        
    def load_data(self, file_path: str, validate_schema: bool = True) -> pd.DataFrame:
        """
        Load data from CSV file with validation and error handling.
        
        Args:
            file_path: Path to the CSV file
            validate_schema: Whether to validate the data schema
            
        Returns:
            pd.DataFrame: Loaded and validated dataset
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data doesn't match expected schema
            pd.errors.EmptyDataError: If the file is empty
            pd.errors.ParserError: If the file cannot be parsed
        """
        resolved_path = self._resolve_file_path(file_path)
        
        try:
            logger.info(f"Loading data from {resolved_path}")
            
            # Check file size to determine loading strategy
            file_size_mb = os.path.getsize(resolved_path) / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
            if file_size_mb > 100:  # Use chunked reading for files > 100MB
                logger.info("Using chunked reading for large file")
                df = self._load_data_chunked(resolved_path)
            else:
                logger.info("Loading entire file into memory")
                df = pd.read_csv(resolved_path)
            
            logger.info(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
            
            if validate_schema:
                self._validate_schema(df)
                self._validate_data_quality(df)
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {resolved_path}")
            raise FileNotFoundError(f"Data file not found: {resolved_path}")
        except pd.errors.EmptyDataError:
            logger.error(f"File is empty: {resolved_path}")
            raise pd.errors.EmptyDataError(f"Data file is empty: {resolved_path}")
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise pd.errors.ParserError(f"Error parsing CSV file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            raise RuntimeError(f"Unexpected error loading data: {e}")
    
    def load_data_chunked(self, file_path: str, validate_schema: bool = True) -> Iterator[pd.DataFrame]:
        """
        Load data in chunks for memory-efficient processing.
        
        Args:
            file_path: Path to the CSV file
            validate_schema: Whether to validate the schema of each chunk
            
        Yields:
            pd.DataFrame: Data chunks
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data doesn't match expected schema
        """
        resolved_path = self._resolve_file_path(file_path)
        
        try:
            logger.info(f"Loading data in chunks from {resolved_path}")
            
            chunk_reader = pd.read_csv(resolved_path, chunksize=self.chunk_size)
            
            for i, chunk in enumerate(chunk_reader):
                logger.debug(f"Processing chunk {i + 1} with {len(chunk)} rows")
                
                if validate_schema and i == 0:  # Validate schema on first chunk
                    self._validate_schema(chunk)
                
                yield chunk
                
        except FileNotFoundError:
            logger.error(f"File not found: {resolved_path}")
            raise FileNotFoundError(f"Data file not found: {resolved_path}")
        except Exception as e:
            logger.error(f"Error reading data chunks: {e}")
            raise RuntimeError(f"Error reading data chunks: {e}")
    
    def _load_data_chunked(self, file_path: str) -> pd.DataFrame:
        """
        Load large file using chunked reading and concatenate results.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Complete dataset
        """
        chunks = []
        
        try:
            chunk_reader = pd.read_csv(file_path, chunksize=self.chunk_size)
            
            for i, chunk in enumerate(chunk_reader):
                logger.debug(f"Loading chunk {i + 1} with {len(chunk)} rows")
                chunks.append(chunk)
                
                # Log progress for very large files
                if (i + 1) % 100 == 0:
                    total_rows = (i + 1) * self.chunk_size
                    logger.info(f"Loaded {total_rows:,} rows so far...")
            
            logger.info(f"Concatenating {len(chunks)} chunks")
            return pd.concat(chunks, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error during chunked loading: {e}")
            raise RuntimeError(f"Error during chunked loading: {e}")
    
    def _resolve_file_path(self, file_path: str) -> str:
        """
        Resolve file path, handling relative paths and path validation.
        
        Args:
            file_path: Input file path
            
        Returns:
            str: Resolved absolute file path
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        # Convert to Path object for better handling
        path = Path(file_path)
        
        # If relative path, try to resolve from project root
        if not path.is_absolute():
            # Try relative to current working directory first
            if path.exists():
                resolved_path = path.resolve()
            else:
                # Try relative to project root (assuming we're in src/data)
                project_root = Path(__file__).parent.parent.parent
                potential_path = project_root / path
                if potential_path.exists():
                    resolved_path = potential_path.resolve()
                else:
                    raise FileNotFoundError(f"File not found: {file_path}")
        else:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            resolved_path = path.resolve()
        
        logger.debug(f"Resolved file path: {resolved_path}")
        return str(resolved_path)
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validate that the DataFrame matches the expected schema.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If schema validation fails
        """
        logger.debug("Validating data schema")
        
        # Check if all expected columns are present
        missing_columns = set(self.EXPECTED_SCHEMA.keys()) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for unexpected columns
        extra_columns = set(df.columns) - set(self.EXPECTED_SCHEMA.keys())
        if extra_columns:
            logger.warning(f"Found unexpected columns: {extra_columns}")
        
        # Validate transaction types
        if 'type' in df.columns:
            invalid_types = set(df['type'].unique()) - self.EXPECTED_TRANSACTION_TYPES
            if invalid_types:
                raise ValueError(f"Invalid transaction types found: {invalid_types}")
        
        # Validate data types (basic validation)
        for col, expected_dtype in self.EXPECTED_SCHEMA.items():
            if col in df.columns:
                try:
                    if expected_dtype == 'int64':
                        # Check if values can be converted to int (allowing for NaN)
                        pd.to_numeric(df[col], errors='coerce')
                    elif expected_dtype == 'float64':
                        # Check if values can be converted to float
                        pd.to_numeric(df[col], errors='coerce')
                    # object type (string) doesn't need validation
                except Exception as e:
                    logger.warning(f"Data type validation warning for column {col}: {e}")
        
        logger.debug("Schema validation completed successfully")
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """
        Perform basic data quality checks.
        
        Args:
            df: DataFrame to validate
        """
        logger.debug("Performing data quality checks")
        
        # Check for completely empty DataFrame
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Log basic statistics
        total_rows = len(df)
        logger.info(f"Dataset contains {total_rows:,} transactions")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning("Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                percentage = (count / total_rows) * 100
                logger.warning(f"  {col}: {count:,} ({percentage:.2f}%)")
        
        # Validate business rules
        if 'step' in df.columns:
            step_range = df['step'].agg(['min', 'max'])
            logger.info(f"Time step range: {step_range['min']} to {step_range['max']}")
            
            if step_range['min'] < 1 or step_range['max'] > 744:
                logger.warning("Step values outside expected range (1-744)")
        
        if 'amount' in df.columns:
            negative_amounts = (df['amount'] < 0).sum()
            if negative_amounts > 0:
                logger.warning(f"Found {negative_amounts:,} transactions with negative amounts")
        
        # Check fraud distribution
        if 'isFraud' in df.columns:
            fraud_counts = df['isFraud'].value_counts()
            fraud_rate = fraud_counts.get(1, 0) / total_rows * 100
            logger.info(f"Fraud rate: {fraud_rate:.4f}% ({fraud_counts.get(1, 0):,} fraudulent transactions)")
        
        logger.debug("Data quality checks completed")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Add transaction-specific information
        if 'type' in df.columns:
            info['transaction_types'] = df['type'].value_counts().to_dict()
        
        if 'isFraud' in df.columns:
            info['fraud_distribution'] = df['isFraud'].value_counts().to_dict()
        
        if 'amount' in df.columns:
            info['amount_stats'] = df['amount'].describe().to_dict()
        
        return info