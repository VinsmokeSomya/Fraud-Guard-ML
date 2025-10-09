"""
Model Registry service for fraud detection models.

This module provides comprehensive model persistence, versioning, and registry functionality
for managing trained fraud detection models throughout their lifecycle.
"""

import json
import joblib
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pandas as pd
import numpy as np

from config.settings import settings
from config.logging_config import get_logger
from ..models.base_model import FraudModel

logger = get_logger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Model version metadata."""
    model_id: str
    version: str
    model_name: str
    model_type: str
    status: ModelStatus
    created_at: str
    updated_at: str
    file_path: str
    metadata_path: str
    features_path: str
    checksum: str
    size_bytes: int
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    feature_names: List[str]
    tags: List[str]
    description: str
    author: str


class ModelRegistry:
    """
    Comprehensive model registry for managing fraud detection models.
    
    Provides functionality for model versioning, persistence, loading,
    and lifecycle management with metadata tracking.
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to the registry directory. If None, uses default from settings.
        """
        self.registry_path = registry_path or settings.models_dir
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry metadata file
        self.registry_file = self.registry_path / "model_registry.json"
        
        # Initialize registry if it doesn't exist
        if not self.registry_file.exists():
            self._initialize_registry()
        
        logger.info(f"Model registry initialized at: {self.registry_path}")
    
    def _initialize_registry(self) -> None:
        """Initialize empty model registry."""
        registry_data = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "models": {},
            "metadata": {
                "version": "1.0.0",
                "description": "Fraud Detection Model Registry"
            }
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info("Initialized new model registry")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry data from file."""
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading registry: {e}")
            self._initialize_registry()
            return self._load_registry()
    
    def _save_registry(self, registry_data: Dict[str, Any]) -> None:
        """Save registry data to file."""
        registry_data["updated_at"] = datetime.now().isoformat()
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _generate_model_id(self, model_name: str, model_type: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{model_type}_{timestamp}"
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number for a model."""
        registry_data = self._load_registry()
        
        # Find existing versions for this model
        existing_versions = []
        for model_id, model_info in registry_data["models"].items():
            if model_info["model_name"] == model_name:
                version = model_info["version"]
                try:
                    # Extract version number (assuming format like "v1.0.0")
                    version_num = float(version.replace("v", "").split(".")[0])
                    existing_versions.append(version_num)
                except ValueError:
                    continue
        
        # Generate next version
        if existing_versions:
            next_version = max(existing_versions) + 1
        else:
            next_version = 1
        
        return f"v{int(next_version)}.0.0"
    
    def register_model(self,
                      model: FraudModel,
                      model_name: str,
                      description: str = "",
                      tags: List[str] = None,
                      author: str = "system",
                      performance_metrics: Dict[str, float] = None) -> str:
        """
        Register a trained model in the registry.
        
        Args:
            model: Trained fraud detection model
            model_name: Name for the model
            description: Model description
            tags: List of tags for categorization
            author: Model author/creator
            performance_metrics: Model performance metrics
            
        Returns:
            Model ID of the registered model
        """
        if not model.is_trained:
            raise ValueError("Cannot register untrained model")
        
        # Generate model metadata
        model_id = self._generate_model_id(model_name, model.model_name)
        version = self._generate_version(model_name)
        
        # Create model directory
        model_dir = self.registry_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model files
        model_file = model_dir / "model.joblib"
        metadata_file = model_dir / "metadata.json"
        features_file = model_dir / "features.json"
        
        # Save the model
        joblib.dump(model.model, model_file)
        
        # Save model metadata
        model_metadata = model.get_metadata()
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Save feature names
        with open(features_file, 'w') as f:
            json.dump(model.feature_names or [], f, indent=2)
        
        # Calculate file properties
        checksum = self._calculate_checksum(model_file)
        size_bytes = model_file.stat().st_size
        
        # Create model version entry
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_name=model_name,
            model_type=model.model_name,
            status=ModelStatus.TRAINED,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            file_path=str(model_file.relative_to(self.registry_path)),
            metadata_path=str(metadata_file.relative_to(self.registry_path)),
            features_path=str(features_file.relative_to(self.registry_path)),
            checksum=checksum,
            size_bytes=size_bytes,
            performance_metrics=performance_metrics or {},
            training_config=model_metadata.get("parameters", {}),
            feature_names=model.feature_names or [],
            tags=tags or [],
            description=description,
            author=author
        )
        
        # Update registry
        registry_data = self._load_registry()
        registry_data["models"][model_id] = asdict(model_version)
        self._save_registry(registry_data)
        
        logger.info(f"Registered model: {model_id} (version: {version})")
        return model_id
    
    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model from the registry.
        
        Args:
            model_id: Model ID to load
            
        Returns:
            Tuple of (model_object, model_metadata)
        """
        registry_data = self._load_registry()
        
        if model_id not in registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = registry_data["models"][model_id]
        model_file = self.registry_path / model_info["file_path"]
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Verify checksum
        current_checksum = self._calculate_checksum(model_file)
        if current_checksum != model_info["checksum"]:
            logger.warning(f"Checksum mismatch for model {model_id}")
        
        # Load model
        model = joblib.load(model_file)
        
        logger.info(f"Loaded model: {model_id}")
        return model, model_info
    
    def load_model_with_wrapper(self, model_id: str, model_class: type) -> FraudModel:
        """
        Load a model and wrap it in the appropriate FraudModel class.
        
        Args:
            model_id: Model ID to load
            model_class: FraudModel subclass to wrap the model in
            
        Returns:
            Wrapped FraudModel instance
        """
        model_obj, model_info = self.load_model(model_id)
        
        # Create wrapper instance
        wrapper = model_class()
        wrapper.model = model_obj
        wrapper.is_trained = True
        wrapper.feature_names = model_info["feature_names"]
        wrapper.metadata = model_info.get("training_config", {})
        
        return wrapper
    
    def list_models(self, 
                   model_name: Optional[str] = None,
                   model_type: Optional[str] = None,
                   status: Optional[ModelStatus] = None,
                   tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List models in the registry with optional filtering.
        
        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            status: Filter by model status
            tags: Filter by tags (models must have all specified tags)
            
        Returns:
            List of model information dictionaries
        """
        registry_data = self._load_registry()
        models = []
        
        for model_id, model_info in registry_data["models"].items():
            # Apply filters
            if model_name and model_info["model_name"] != model_name:
                continue
            if model_type and model_info["model_type"] != model_type:
                continue
            if status and model_info["status"] != status.value:
                continue
            if tags and not all(tag in model_info["tags"] for tag in tags):
                continue
            
            models.append(model_info)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x["created_at"], reverse=True)
        
        return models
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information dictionary
        """
        registry_data = self._load_registry()
        
        if model_id not in registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        return registry_data["models"][model_id]
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> None:
        """
        Update the status of a model.
        
        Args:
            model_id: Model ID
            status: New status
        """
        registry_data = self._load_registry()
        
        if model_id not in registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        registry_data["models"][model_id]["status"] = status.value
        registry_data["models"][model_id]["updated_at"] = datetime.now().isoformat()
        
        self._save_registry(registry_data)
        logger.info(f"Updated model {model_id} status to {status.value}")
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a model.
        
        Args:
            model_id: Model ID
            metrics: Performance metrics dictionary
        """
        registry_data = self._load_registry()
        
        if model_id not in registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        registry_data["models"][model_id]["performance_metrics"].update(metrics)
        registry_data["models"][model_id]["updated_at"] = datetime.now().isoformat()
        
        self._save_registry(registry_data)
        logger.info(f"Updated metrics for model {model_id}")
    
    def add_model_tags(self, model_id: str, tags: List[str]) -> None:
        """
        Add tags to a model.
        
        Args:
            model_id: Model ID
            tags: List of tags to add
        """
        registry_data = self._load_registry()
        
        if model_id not in registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        current_tags = set(registry_data["models"][model_id]["tags"])
        current_tags.update(tags)
        registry_data["models"][model_id]["tags"] = list(current_tags)
        registry_data["models"][model_id]["updated_at"] = datetime.now().isoformat()
        
        self._save_registry(registry_data)
        logger.info(f"Added tags {tags} to model {model_id}")
    
    def delete_model(self, model_id: str, force: bool = False) -> None:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model ID to delete
            force: If True, delete even if model is deployed
        """
        registry_data = self._load_registry()
        
        if model_id not in registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = registry_data["models"][model_id]
        
        # Check if model is deployed
        if model_info["status"] == ModelStatus.DEPLOYED.value and not force:
            raise ValueError(f"Cannot delete deployed model {model_id}. Use force=True to override.")
        
        # Delete model files
        model_dir = self.registry_path / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del registry_data["models"][model_id]
        self._save_registry(registry_data)
        
        logger.info(f"Deleted model {model_id}")
    
    def get_latest_model(self, 
                        model_name: str,
                        status: Optional[ModelStatus] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            status: Optional status filter
            
        Returns:
            Latest model info or None if not found
        """
        models = self.list_models(model_name=model_name, status=status)
        
        if not models:
            return None
        
        # Models are already sorted by creation date (newest first)
        return models[0]
    
    def export_model(self, model_id: str, export_path: Path) -> None:
        """
        Export a model to a specified path.
        
        Args:
            model_id: Model ID to export
            export_path: Path to export the model to
        """
        registry_data = self._load_registry()
        
        if model_id not in registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = registry_data["models"][model_id]
        model_dir = self.registry_path / model_id
        
        # Create export directory
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        shutil.copytree(model_dir, export_path / model_id, dirs_exist_ok=True)
        
        # Create export manifest
        manifest = {
            "exported_at": datetime.now().isoformat(),
            "model_info": model_info,
            "export_path": str(export_path)
        }
        
        with open(export_path / f"{model_id}_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Exported model {model_id} to {export_path}")
    
    def import_model(self, import_path: Path) -> str:
        """
        Import a model from an exported path.
        
        Args:
            import_path: Path containing exported model
            
        Returns:
            Model ID of imported model
        """
        # Find manifest file
        manifest_files = list(import_path.glob("*_manifest.json"))
        if not manifest_files:
            raise ValueError("No manifest file found in import path")
        
        manifest_file = manifest_files[0]
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        model_info = manifest["model_info"]
        model_id = model_info["model_id"]
        
        # Copy model directory
        source_dir = import_path / model_id
        target_dir = self.registry_path / model_id
        
        if target_dir.exists():
            raise ValueError(f"Model {model_id} already exists in registry")
        
        shutil.copytree(source_dir, target_dir)
        
        # Update registry
        registry_data = self._load_registry()
        registry_data["models"][model_id] = model_info
        self._save_registry(registry_data)
        
        logger.info(f"Imported model {model_id}")
        return model_id
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model registry.
        
        Returns:
            Dictionary with registry statistics
        """
        registry_data = self._load_registry()
        models = registry_data["models"]
        
        # Count by status
        status_counts = {}
        for status in ModelStatus:
            status_counts[status.value] = sum(
                1 for model in models.values() 
                if model["status"] == status.value
            )
        
        # Count by model type
        type_counts = {}
        for model in models.values():
            model_type = model["model_type"]
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
        
        # Calculate total size
        total_size = sum(model["size_bytes"] for model in models.values())
        
        return {
            "total_models": len(models),
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "registry_created": registry_data["created_at"],
            "last_updated": registry_data["updated_at"]
        }