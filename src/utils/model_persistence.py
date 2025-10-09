"""
Model persistence utilities for fraud detection models.

This module provides enhanced model serialization, loading, and persistence utilities
that extend the base model functionality with additional features like compression,
encryption, and format conversion.
"""

import json
import joblib
import pickle
import gzip
import bz2
import lzma
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, BinaryIO
from io import BytesIO
import hashlib
import base64
import pandas as pd
import numpy as np

# Optional cryptography imports (graceful fallback if not available)
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None

from config.settings import settings
from config.logging_config import get_logger
from ..models.base_model import FraudModel

logger = get_logger(__name__)


class CompressionType:
    """Compression type constants."""
    NONE = "none"
    GZIP = "gzip"
    BZ2 = "bz2"
    LZMA = "lzma"


class SerializationFormat:
    """Serialization format constants."""
    JOBLIB = "joblib"
    PICKLE = "pickle"
    JSON = "json"  # For metadata only


class ModelPersistenceError(Exception):
    """Custom exception for model persistence errors."""
    pass


class SecureModelPersistence:
    """
    Secure model persistence with encryption, compression, and integrity checking.
    
    Provides enhanced model serialization capabilities including:
    - Multiple compression algorithms
    - Encryption for sensitive models
    - Integrity verification with checksums
    - Metadata preservation
    - Format conversion utilities
    """
    
    def __init__(self, 
                 encryption_key: Optional[bytes] = None,
                 default_compression: str = CompressionType.GZIP,
                 default_format: str = SerializationFormat.JOBLIB):
        """
        Initialize secure model persistence.
        
        Args:
            encryption_key: Encryption key for secure storage (optional)
            default_compression: Default compression algorithm
            default_format: Default serialization format
        """
        self.encryption_key = encryption_key
        self.default_compression = default_compression
        self.default_format = default_format
        
        # Initialize encryption if key provided and cryptography is available
        self.cipher_suite = None
        if encryption_key and CRYPTOGRAPHY_AVAILABLE:
            self.cipher_suite = Fernet(encryption_key)
        elif encryption_key and not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Encryption requested but cryptography library not available")
        
        logger.info("Secure model persistence initialized")
    
    @staticmethod
    def generate_encryption_key() -> bytes:
        """
        Generate a new encryption key.
        
        Returns:
            Encryption key bytes
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ModelPersistenceError("Cryptography library not available for encryption")
        return Fernet.generate_key()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password.
        
        Args:
            password: Password string
            salt: Salt bytes (generated if not provided)
            
        Returns:
            Tuple of (key, salt)
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ModelPersistenceError("Cryptography library not available for key derivation")
            
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()
    
    def _compress_data(self, data: bytes, compression: str) -> bytes:
        """Compress data using specified algorithm."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression == CompressionType.BZ2:
            return bz2.compress(data)
        elif compression == CompressionType.LZMA:
            return lzma.compress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
    
    def _decompress_data(self, data: bytes, compression: str) -> bytes:
        """Decompress data using specified algorithm."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.BZ2:
            return bz2.decompress(data)
        elif compression == CompressionType.LZMA:
            return lzma.decompress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if encryption is enabled."""
        if self.cipher_suite is None:
            return data
        return self.cipher_suite.encrypt(data)
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if encryption is enabled."""
        if self.cipher_suite is None:
            return data
        return self.cipher_suite.decrypt(data)
    
    def _serialize_model(self, model: Any, format_type: str) -> bytes:
        """Serialize model using specified format."""
        if format_type == SerializationFormat.JOBLIB:
            return joblib.dump(model, None, compress=False)[0]  # Get bytes
        elif format_type == SerializationFormat.PICKLE:
            return pickle.dumps(model)
        else:
            raise ValueError(f"Unsupported serialization format: {format_type}")
    
    def _deserialize_model(self, data: bytes, format_type: str) -> Any:
        """Deserialize model using specified format."""
        if format_type == SerializationFormat.JOBLIB:
            return joblib.load(BytesIO(data))
        elif format_type == SerializationFormat.PICKLE:
            return pickle.loads(data)
        else:
            raise ValueError(f"Unsupported serialization format: {format_type}")
    
    def save_model_secure(self,
                         model: FraudModel,
                         filepath: Union[str, Path],
                         compression: Optional[str] = None,
                         format_type: Optional[str] = None,
                         encrypt: bool = None,
                         include_metadata: bool = True) -> Dict[str, Any]:
        """
        Save model with enhanced security and compression.
        
        Args:
            model: Trained fraud detection model
            filepath: Path to save the model
            compression: Compression algorithm to use
            format_type: Serialization format
            encrypt: Whether to encrypt the model
            include_metadata: Whether to save metadata separately
            
        Returns:
            Dictionary with save operation details
        """
        if not model.is_trained:
            raise ModelPersistenceError("Cannot save untrained model")
        
        filepath = Path(filepath)
        compression = compression or self.default_compression
        format_type = format_type or self.default_format
        encrypt = encrypt if encrypt is not None else (self.cipher_suite is not None)
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize model
        model_data = self._serialize_model(model.model, format_type)
        
        # Compress data
        compressed_data = self._compress_data(model_data, compression)
        
        # Encrypt if requested
        if encrypt and self.cipher_suite is None:
            raise ModelPersistenceError("Encryption requested but no encryption key provided")
        
        final_data = self._encrypt_data(compressed_data) if encrypt else compressed_data
        
        # Calculate checksums
        original_checksum = self._calculate_checksum(model_data)
        final_checksum = self._calculate_checksum(final_data)
        
        # Create save metadata
        save_metadata = {
            "saved_at": datetime.now().isoformat(),
            "format": format_type,
            "compression": compression,
            "encrypted": encrypt,
            "original_size": len(model_data),
            "compressed_size": len(compressed_data),
            "final_size": len(final_data),
            "compression_ratio": len(compressed_data) / len(model_data),
            "original_checksum": original_checksum,
            "final_checksum": final_checksum,
            "model_metadata": model.get_metadata() if include_metadata else {}
        }
        
        # Save model file
        with open(filepath, 'wb') as f:
            f.write(final_data)
        
        # Save metadata file
        if include_metadata:
            metadata_file = filepath.with_suffix('.metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(save_metadata, f, indent=2)
        
        # Save feature names
        if model.feature_names:
            features_file = filepath.with_suffix('.features.json')
            with open(features_file, 'w') as f:
                json.dump(model.feature_names, f, indent=2)
        
        logger.info(f"Saved model securely to {filepath}")
        logger.info(f"Compression ratio: {save_metadata['compression_ratio']:.2f}")
        
        return save_metadata
    
    def load_model_secure(self,
                         filepath: Union[str, Path],
                         model_class: Optional[type] = None,
                         verify_checksum: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model with security verification.
        
        Args:
            filepath: Path to the saved model
            model_class: Model class to wrap the loaded model (optional)
            verify_checksum: Whether to verify data integrity
            
        Returns:
            Tuple of (model, metadata)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ModelPersistenceError(f"Model file not found: {filepath}")
        
        # Load metadata
        metadata_file = filepath.with_suffix('.metadata.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                save_metadata = json.load(f)
        else:
            # Try to infer format and compression
            save_metadata = {
                "format": self.default_format,
                "compression": self.default_compression,
                "encrypted": False
            }
        
        # Read model file
        with open(filepath, 'rb') as f:
            final_data = f.read()
        
        # Verify final checksum if available
        if verify_checksum and "final_checksum" in save_metadata:
            current_checksum = self._calculate_checksum(final_data)
            if current_checksum != save_metadata["final_checksum"]:
                raise ModelPersistenceError("Model file checksum verification failed")
        
        # Decrypt if needed
        if save_metadata.get("encrypted", False):
            if self.cipher_suite is None:
                raise ModelPersistenceError("Model is encrypted but no decryption key provided")
            compressed_data = self._decrypt_data(final_data)
        else:
            compressed_data = final_data
        
        # Decompress data
        model_data = self._decompress_data(
            compressed_data, 
            save_metadata.get("compression", CompressionType.NONE)
        )
        
        # Verify original checksum if available
        if verify_checksum and "original_checksum" in save_metadata:
            current_checksum = self._calculate_checksum(model_data)
            if current_checksum != save_metadata["original_checksum"]:
                raise ModelPersistenceError("Model data checksum verification failed")
        
        # Deserialize model
        model = self._deserialize_model(
            model_data, 
            save_metadata.get("format", SerializationFormat.JOBLIB)
        )
        
        # Wrap in model class if provided
        if model_class and issubclass(model_class, FraudModel):
            wrapper = model_class()
            wrapper.model = model
            wrapper.is_trained = True
            
            # Load feature names if available
            features_file = filepath.with_suffix('.features.json')
            if features_file.exists():
                with open(features_file, 'r') as f:
                    wrapper.feature_names = json.load(f)
            
            # Load model metadata if available
            if "model_metadata" in save_metadata:
                wrapper.metadata = save_metadata["model_metadata"]
            
            model = wrapper
        
        logger.info(f"Loaded model securely from {filepath}")
        return model, save_metadata
    
    def convert_model_format(self,
                           input_path: Union[str, Path],
                           output_path: Union[str, Path],
                           target_format: str,
                           target_compression: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert model between different formats and compression schemes.
        
        Args:
            input_path: Path to input model
            output_path: Path for converted model
            target_format: Target serialization format
            target_compression: Target compression algorithm
            
        Returns:
            Conversion details dictionary
        """
        # Load original model
        model, original_metadata = self.load_model_secure(input_path, verify_checksum=True)
        
        # Extract the actual model object if wrapped
        if isinstance(model, FraudModel):
            actual_model = model.model
            wrapper_metadata = model.get_metadata()
            feature_names = model.feature_names
        else:
            actual_model = model
            wrapper_metadata = {}
            feature_names = None
        
        # Save in new format
        save_metadata = self.save_model_secure(
            model if isinstance(model, FraudModel) else actual_model,
            output_path,
            compression=target_compression,
            format_type=target_format,
            encrypt=original_metadata.get("encrypted", False)
        )
        
        # Create conversion report
        conversion_report = {
            "converted_at": datetime.now().isoformat(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "original_format": original_metadata.get("format", "unknown"),
            "target_format": target_format,
            "original_compression": original_metadata.get("compression", "unknown"),
            "target_compression": target_compression or self.default_compression,
            "original_size": original_metadata.get("final_size", 0),
            "new_size": save_metadata["final_size"],
            "size_change_ratio": (save_metadata["final_size"] / original_metadata.get("final_size", 1)),
            "conversion_successful": True
        }
        
        logger.info(f"Converted model from {input_path} to {output_path}")
        logger.info(f"Size change: {conversion_report['size_change_ratio']:.2f}x")
        
        return conversion_report
    
    def batch_convert_models(self,
                           input_dir: Union[str, Path],
                           output_dir: Union[str, Path],
                           target_format: str,
                           target_compression: Optional[str] = None,
                           pattern: str = "*.joblib") -> List[Dict[str, Any]]:
        """
        Convert multiple models in batch.
        
        Args:
            input_dir: Directory containing input models
            output_dir: Directory for converted models
            target_format: Target serialization format
            target_compression: Target compression algorithm
            pattern: File pattern to match
            
        Returns:
            List of conversion reports
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_files = list(input_dir.glob(pattern))
        conversion_reports = []
        
        for model_file in model_files:
            try:
                output_file = output_dir / model_file.name
                
                conversion_report = self.convert_model_format(
                    model_file,
                    output_file,
                    target_format,
                    target_compression
                )
                
                conversion_reports.append(conversion_report)
                
            except Exception as e:
                error_report = {
                    "input_path": str(model_file),
                    "error": str(e),
                    "conversion_successful": False
                }
                conversion_reports.append(error_report)
                logger.error(f"Failed to convert {model_file}: {e}")
        
        logger.info(f"Batch conversion completed: {len(conversion_reports)} models processed")
        return conversion_reports
    
    def verify_model_integrity(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Verify the integrity of a saved model.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Integrity verification report
        """
        filepath = Path(filepath)
        
        verification_report = {
            "verified_at": datetime.now().isoformat(),
            "file_path": str(filepath),
            "file_exists": filepath.exists(),
            "metadata_exists": False,
            "features_exists": False,
            "checksum_valid": False,
            "loadable": False,
            "errors": []
        }
        
        if not filepath.exists():
            verification_report["errors"].append("Model file does not exist")
            return verification_report
        
        # Check metadata file
        metadata_file = filepath.with_suffix('.metadata.json')
        verification_report["metadata_exists"] = metadata_file.exists()
        
        # Check features file
        features_file = filepath.with_suffix('.features.json')
        verification_report["features_exists"] = features_file.exists()
        
        try:
            # Try to load the model
            model, metadata = self.load_model_secure(filepath, verify_checksum=True)
            verification_report["loadable"] = True
            verification_report["checksum_valid"] = True
            
            # Additional checks if model is wrapped
            if isinstance(model, FraudModel):
                verification_report["model_trained"] = model.is_trained
                verification_report["feature_count"] = len(model.feature_names) if model.feature_names else 0
            
        except Exception as e:
            verification_report["errors"].append(f"Failed to load model: {str(e)}")
        
        verification_report["integrity_score"] = sum([
            verification_report["file_exists"],
            verification_report["metadata_exists"],
            verification_report["features_exists"],
            verification_report["checksum_valid"],
            verification_report["loadable"]
        ]) / 5.0
        
        return verification_report
    
    def cleanup_old_models(self,
                          directory: Union[str, Path],
                          keep_latest: int = 5,
                          dry_run: bool = True) -> List[str]:
        """
        Clean up old model files, keeping only the latest versions.
        
        Args:
            directory: Directory containing models
            keep_latest: Number of latest models to keep
            dry_run: If True, only report what would be deleted
            
        Returns:
            List of files that were (or would be) deleted
        """
        directory = Path(directory)
        
        # Find all model files
        model_files = []
        for pattern in ["*.joblib", "*.pkl", "*.pickle"]:
            model_files.extend(directory.glob(pattern))
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Identify files to delete
        files_to_delete = model_files[keep_latest:]
        deleted_files = []
        
        for model_file in files_to_delete:
            # Also delete associated metadata and features files
            associated_files = [
                model_file,
                model_file.with_suffix('.metadata.json'),
                model_file.with_suffix('.features.json')
            ]
            
            for file_path in associated_files:
                if file_path.exists():
                    if not dry_run:
                        file_path.unlink()
                    deleted_files.append(str(file_path))
        
        if dry_run:
            logger.info(f"Dry run: Would delete {len(deleted_files)} files")
        else:
            logger.info(f"Deleted {len(deleted_files)} old model files")
        
        return deleted_files


# Convenience functions for common operations
def save_model_compressed(model: FraudModel, 
                         filepath: Union[str, Path],
                         compression: str = CompressionType.GZIP) -> Dict[str, Any]:
    """
    Save model with compression (convenience function).
    
    Args:
        model: Trained fraud detection model
        filepath: Path to save the model
        compression: Compression algorithm
        
    Returns:
        Save operation details
    """
    persistence = SecureModelPersistence(default_compression=compression)
    return persistence.save_model_secure(model, filepath)


def load_model_compressed(filepath: Union[str, Path],
                         model_class: Optional[type] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Load compressed model (convenience function).
    
    Args:
        filepath: Path to the saved model
        model_class: Model class to wrap the loaded model
        
    Returns:
        Tuple of (model, metadata)
    """
    persistence = SecureModelPersistence()
    return persistence.load_model_secure(filepath, model_class)


def create_model_backup(model: FraudModel,
                       backup_dir: Union[str, Path],
                       include_timestamp: bool = True) -> Path:
    """
    Create a backup of a trained model.
    
    Args:
        model: Trained fraud detection model
        backup_dir: Directory to store backup
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        Path to backup file
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate backup filename
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{model.model_name}_backup_{timestamp}.joblib"
    else:
        backup_filename = f"{model.model_name}_backup.joblib"
    
    backup_path = backup_dir / backup_filename
    
    # Save backup
    persistence = SecureModelPersistence()
    persistence.save_model_secure(model, backup_path)
    
    logger.info(f"Created model backup at {backup_path}")
    return backup_path