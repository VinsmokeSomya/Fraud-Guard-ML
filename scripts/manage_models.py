#!/usr/bin/env python3
"""
Model management script for fraud detection models.

This script provides a command-line interface for managing the model registry,
including registration, versioning, and lifecycle management.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.model_registry import ModelRegistry, ModelStatus
from utils.model_persistence import SecureModelPersistence, save_model_compressed, load_model_compressed
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from config.logging_config import get_logger

logger = get_logger(__name__)

# Model class mapping
MODEL_CLASSES = {
    "LogisticRegression": LogisticRegressionModel,
    "RandomForest": RandomForestModel,
    "XGBoost": XGBoostModel
}


def register_model_command(args) -> None:
    """Execute model registration command."""
    try:
        registry = ModelRegistry()
        
        # Load the model
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)
        
        # Determine model class
        model_class = MODEL_CLASSES.get(args.model_type)
        if not model_class:
            logger.error(f"Unsupported model type: {args.model_type}")
            logger.info(f"Supported types: {list(MODEL_CLASSES.keys())}")
            sys.exit(1)
        
        # Load model with wrapper
        model, metadata = load_model_compressed(model_path, model_class)
        
        # Parse performance metrics if provided
        performance_metrics = {}
        if args.metrics:
            try:
                performance_metrics = json.loads(args.metrics)
            except json.JSONDecodeError:
                logger.error("Invalid JSON format for metrics")
                sys.exit(1)
        
        # Parse tags
        tags = args.tags.split(',') if args.tags else []
        
        # Register model
        model_id = registry.register_model(
            model=model,
            model_name=args.model_name,
            description=args.description or "",
            tags=tags,
            author=args.author or "system",
            performance_metrics=performance_metrics
        )
        
        logger.info(f"Model registered successfully with ID: {model_id}")
        
        # Display model info
        model_info = registry.get_model_info(model_id)
        print(f"\nRegistered Model Details:")
        print(f"Model ID: {model_info['model_id']}")
        print(f"Name: {model_info['model_name']}")
        print(f"Type: {model_info['model_type']}")
        print(f"Version: {model_info['version']}")
        print(f"Status: {model_info['status']}")
        
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        sys.exit(1)


def list_models_command(args) -> None:
    """Execute list models command."""
    try:
        registry = ModelRegistry()
        
        # Apply filters
        status_filter = ModelStatus(args.status) if args.status else None
        tags_filter = args.tags.split(',') if args.tags else None
        
        models = registry.list_models(
            model_name=args.model_name,
            model_type=args.model_type,
            status=status_filter,
            tags=tags_filter
        )
        
        if not models:
            logger.info("No models found matching criteria")
            return
        
        # Display models
        print(f"\nFound {len(models)} models:")
        print("=" * 100)
        
        for model in models:
            print(f"Model ID: {model['model_id']}")
            print(f"Name: {model['model_name']} (v{model['version']})")
            print(f"Type: {model['model_type']}")
            print(f"Status: {model['status']}")
            print(f"Created: {model['created_at']}")
            print(f"Size: {model['size_bytes']} bytes")
            
            if model.get('description'):
                print(f"Description: {model['description']}")
            
            if model.get('tags'):
                print(f"Tags: {', '.join(model['tags'])}")
            
            if model.get('performance_metrics'):
                print(f"Metrics: {model['performance_metrics']}")
            
            print("-" * 100)
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        sys.exit(1)


def model_info_command(args) -> None:
    """Execute model info command."""
    try:
        registry = ModelRegistry()
        
        model_info = registry.get_model_info(args.model_id)
        
        print(f"\nDetailed Model Information:")
        print("=" * 50)
        
        # Basic information
        print(f"Model ID: {model_info['model_id']}")
        print(f"Name: {model_info['model_name']}")
        print(f"Type: {model_info['model_type']}")
        print(f"Version: {model_info['version']}")
        print(f"Status: {model_info['status']}")
        print(f"Author: {model_info['author']}")
        print(f"Description: {model_info['description']}")
        
        # Timestamps
        print(f"\nTimestamps:")
        print(f"Created: {model_info['created_at']}")
        print(f"Updated: {model_info['updated_at']}")
        
        # File information
        print(f"\nFile Information:")
        print(f"File Path: {model_info['file_path']}")
        print(f"Size: {model_info['size_bytes']} bytes")
        print(f"Checksum: {model_info['checksum']}")
        
        # Features
        if model_info.get('feature_names'):
            print(f"\nFeatures ({len(model_info['feature_names'])}):")
            for i, feature in enumerate(model_info['feature_names'][:10]):  # Show first 10
                print(f"  {i+1}. {feature}")
            if len(model_info['feature_names']) > 10:
                print(f"  ... and {len(model_info['feature_names']) - 10} more")
        
        # Performance metrics
        if model_info.get('performance_metrics'):
            print(f"\nPerformance Metrics:")
            for metric, value in model_info['performance_metrics'].items():
                print(f"  {metric}: {value}")
        
        # Training configuration
        if model_info.get('training_config'):
            print(f"\nTraining Configuration:")
            for param, value in model_info['training_config'].items():
                print(f"  {param}: {value}")
        
        # Tags
        if model_info.get('tags'):
            print(f"\nTags: {', '.join(model_info['tags'])}")
        
    except ValueError as e:
        logger.error(f"Model not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        sys.exit(1)


def update_status_command(args) -> None:
    """Execute update model status command."""
    try:
        registry = ModelRegistry()
        
        status = ModelStatus(args.status)
        registry.update_model_status(args.model_id, status)
        
        logger.info(f"Updated model {args.model_id} status to {status.value}")
        
    except ValueError as e:
        logger.error(f"Invalid status or model ID: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to update model status: {e}")
        sys.exit(1)


def add_tags_command(args) -> None:
    """Execute add tags command."""
    try:
        registry = ModelRegistry()
        
        tags = args.tags.split(',')
        registry.add_model_tags(args.model_id, tags)
        
        logger.info(f"Added tags {tags} to model {args.model_id}")
        
    except ValueError as e:
        logger.error(f"Model not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to add tags: {e}")
        sys.exit(1)


def export_model_command(args) -> None:
    """Execute export model command."""
    try:
        registry = ModelRegistry()
        
        export_path = Path(args.export_path)
        registry.export_model(args.model_id, export_path)
        
        logger.info(f"Exported model {args.model_id} to {export_path}")
        
    except ValueError as e:
        logger.error(f"Model not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        sys.exit(1)


def import_model_command(args) -> None:
    """Execute import model command."""
    try:
        registry = ModelRegistry()
        
        import_path = Path(args.import_path)
        model_id = registry.import_model(import_path)
        
        logger.info(f"Imported model with ID: {model_id}")
        
    except ValueError as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to import model: {e}")
        sys.exit(1)


def delete_model_command(args) -> None:
    """Execute delete model command."""
    try:
        registry = ModelRegistry()
        
        # Confirm deletion unless force flag is used
        if not args.force:
            model_info = registry.get_model_info(args.model_id)
            print(f"Are you sure you want to delete model '{model_info['model_name']}' ({args.model_id})?")
            confirmation = input("Type 'yes' to confirm: ")
            if confirmation.lower() != 'yes':
                logger.info("Deletion cancelled")
                return
        
        registry.delete_model(args.model_id, force=args.force)
        logger.info(f"Deleted model {args.model_id}")
        
    except ValueError as e:
        logger.error(f"Model not found or cannot be deleted: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        sys.exit(1)


def registry_stats_command(args) -> None:
    """Execute registry statistics command."""
    try:
        registry = ModelRegistry()
        
        stats = registry.get_registry_stats()
        
        print(f"\nModel Registry Statistics:")
        print("=" * 40)
        print(f"Total Models: {stats['total_models']}")
        print(f"Total Size: {stats['total_size_mb']} MB")
        print(f"Registry Created: {stats['registry_created']}")
        print(f"Last Updated: {stats['last_updated']}")
        
        print(f"\nStatus Distribution:")
        for status, count in stats['status_distribution'].items():
            print(f"  {status}: {count}")
        
        print(f"\nModel Type Distribution:")
        for model_type, count in stats['type_distribution'].items():
            print(f"  {model_type}: {count}")
        
    except Exception as e:
        logger.error(f"Failed to get registry stats: {e}")
        sys.exit(1)


def cleanup_command(args) -> None:
    """Execute cleanup command."""
    try:
        persistence = SecureModelPersistence()
        
        deleted_files = persistence.cleanup_old_models(
            directory=args.directory,
            keep_latest=args.keep_latest,
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            print(f"Dry run: Would delete {len(deleted_files)} files")
            for file_path in deleted_files:
                print(f"  {file_path}")
        else:
            print(f"Deleted {len(deleted_files)} old model files")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for the model management script."""
    parser = argparse.ArgumentParser(
        description="Manage fraud detection models in the registry",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new model')
    register_parser.add_argument('model_path', help='Path to the model file')
    register_parser.add_argument('model_name', help='Name for the model')
    register_parser.add_argument('model_type', choices=list(MODEL_CLASSES.keys()),
                                help='Type of the model')
    register_parser.add_argument('--description', help='Model description')
    register_parser.add_argument('--tags', help='Comma-separated tags')
    register_parser.add_argument('--author', help='Model author')
    register_parser.add_argument('--metrics', help='Performance metrics as JSON string')
    register_parser.set_defaults(func=register_model_command)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List models in registry')
    list_parser.add_argument('--model-name', help='Filter by model name')
    list_parser.add_argument('--model-type', help='Filter by model type')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--tags', help='Filter by tags (comma-separated)')
    list_parser.set_defaults(func=list_models_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get detailed model information')
    info_parser.add_argument('model_id', help='Model ID')
    info_parser.set_defaults(func=model_info_command)
    
    # Update status command
    status_parser = subparsers.add_parser('update-status', help='Update model status')
    status_parser.add_argument('model_id', help='Model ID')
    status_parser.add_argument('status', choices=[s.value for s in ModelStatus],
                              help='New status')
    status_parser.set_defaults(func=update_status_command)
    
    # Add tags command
    tags_parser = subparsers.add_parser('add-tags', help='Add tags to model')
    tags_parser.add_argument('model_id', help='Model ID')
    tags_parser.add_argument('tags', help='Comma-separated tags to add')
    tags_parser.set_defaults(func=add_tags_command)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('model_id', help='Model ID to export')
    export_parser.add_argument('export_path', help='Export directory path')
    export_parser.set_defaults(func=export_model_command)
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import model')
    import_parser.add_argument('import_path', help='Import directory path')
    import_parser.set_defaults(func=import_model_command)
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete model')
    delete_parser.add_argument('model_id', help='Model ID to delete')
    delete_parser.add_argument('--force', action='store_true', 
                              help='Force deletion without confirmation')
    delete_parser.set_defaults(func=delete_model_command)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show registry statistics')
    stats_parser.set_defaults(func=registry_stats_command)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old model files')
    cleanup_parser.add_argument('directory', help='Directory to clean up')
    cleanup_parser.add_argument('--keep-latest', type=int, default=5,
                               help='Number of latest models to keep')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be deleted without actually deleting')
    cleanup_parser.set_defaults(func=cleanup_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()