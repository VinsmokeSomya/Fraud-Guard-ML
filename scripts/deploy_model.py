#!/usr/bin/env python3
"""
Model deployment script for fraud detection models.

This script provides a command-line interface for deploying trained models
to different environments with various configuration options.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.model_registry import ModelRegistry, ModelStatus
from services.model_deployment import ModelDeployment, DeploymentEnvironment, DeploymentConfig
from config.logging_config import get_logger

logger = get_logger(__name__)


def load_deployment_config(config_path: Path) -> Dict[str, Any]:
    """Load deployment configuration from file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            return json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            import yaml
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def create_default_config(environment: str) -> Dict[str, Any]:
    """Create default deployment configuration."""
    return {
        "environment": environment,
        "deployment_name": f"fraud-detection-{environment}",
        "api_endpoint": "http://localhost:8000",
        "resource_limits": {
            "cpu": "1000m",
            "memory": "2Gi",
            "gpu": 0
        },
        "environment_variables": {
            "ENVIRONMENT": environment,
            "LOG_LEVEL": "INFO",
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000"
        },
        "scaling_config": {
            "min_replicas": 1,
            "max_replicas": 3,
            "target_cpu_utilization": 70
        }
    }


def deploy_model_command(args) -> None:
    """Execute model deployment command."""
    try:
        # Initialize services
        registry = ModelRegistry()
        deployment = ModelDeployment(registry)
        
        # Verify model exists
        try:
            model_info = registry.get_model_info(args.model_id)
            logger.info(f"Found model: {args.model_id} ({model_info['model_name']})")
        except ValueError as e:
            logger.error(f"Model not found: {e}")
            sys.exit(1)
        
        # Load or create deployment configuration
        if args.config:
            config_data = load_deployment_config(Path(args.config))
        else:
            config_data = create_default_config(args.environment)
        
        # Override config with command line arguments
        if args.deployment_name:
            config_data["deployment_name"] = args.deployment_name
        if args.api_endpoint:
            config_data["api_endpoint"] = args.api_endpoint
        
        # Create deployment configuration
        deployment_config = DeploymentConfig(
            model_id=args.model_id,
            environment=DeploymentEnvironment(args.environment),
            deployment_name=config_data["deployment_name"],
            api_endpoint=config_data["api_endpoint"],
            health_check_endpoint=f"{config_data['api_endpoint']}/health",
            resource_limits=config_data.get("resource_limits", {}),
            environment_variables=config_data.get("environment_variables", {}),
            scaling_config=config_data.get("scaling_config", {}),
            monitoring_config=config_data.get("monitoring_config", {}),
            rollback_config=config_data.get("rollback_config", {})
        )
        
        # Execute deployment
        if args.dry_run:
            logger.info("Performing dry run deployment...")
            deployment_id = deployment.deploy_model(deployment_config, dry_run=True)
            logger.info(f"Dry run completed successfully. Deployment ID: {deployment_id}")
        else:
            logger.info("Starting model deployment...")
            deployment_id = deployment.deploy_model(deployment_config, dry_run=False)
            logger.info(f"Model deployed successfully. Deployment ID: {deployment_id}")
            
            # Check deployment health
            if args.health_check:
                logger.info("Checking deployment health...")
                health_status = deployment.check_deployment_health(deployment_id)
                if health_status["healthy"]:
                    logger.info("Deployment is healthy")
                else:
                    logger.warning("Deployment health check failed")
                    logger.info(f"Health status: {health_status}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


def list_models_command(args) -> None:
    """Execute list models command."""
    try:
        registry = ModelRegistry()
        
        # Apply filters
        status_filter = ModelStatus(args.status) if args.status else None
        
        models = registry.list_models(
            model_name=args.model_name,
            model_type=args.model_type,
            status=status_filter
        )
        
        if not models:
            logger.info("No models found matching criteria")
            return
        
        # Display models
        print(f"\nFound {len(models)} models:")
        print("-" * 80)
        
        for model in models:
            print(f"Model ID: {model['model_id']}")
            print(f"Name: {model['model_name']}")
            print(f"Type: {model['model_type']}")
            print(f"Version: {model['version']}")
            print(f"Status: {model['status']}")
            print(f"Created: {model['created_at']}")
            
            if model.get('performance_metrics'):
                print(f"Metrics: {model['performance_metrics']}")
            
            print("-" * 80)
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        sys.exit(1)


def list_deployments_command(args) -> None:
    """Execute list deployments command."""
    try:
        deployment = ModelDeployment()
        
        # Apply filters
        deployments = deployment.list_deployments(
            environment=args.environment,
            status=args.status
        )
        
        if not deployments:
            logger.info("No deployments found matching criteria")
            return
        
        # Display deployments
        print(f"\nFound {len(deployments)} deployments:")
        print("-" * 80)
        
        for deploy in deployments:
            print(f"Deployment ID: {deploy['deployment_id']}")
            print(f"Model ID: {deploy['model_id']}")
            print(f"Environment: {deploy['environment']}")
            print(f"Name: {deploy['deployment_name']}")
            print(f"Status: {deploy['status']}")
            print(f"Created: {deploy['created_at']}")
            
            if deploy.get('deployed_at'):
                print(f"Deployed: {deploy['deployed_at']}")
            
            print("-" * 80)
        
    except Exception as e:
        logger.error(f"Failed to list deployments: {e}")
        sys.exit(1)


def rollback_command(args) -> None:
    """Execute rollback command."""
    try:
        deployment = ModelDeployment()
        
        logger.info(f"Rolling back environment: {args.environment}")
        
        rollback_deployment_id = deployment.rollback_deployment(
            environment=args.environment,
            target_deployment_id=args.target_deployment
        )
        
        logger.info(f"Rollback completed. Active deployment: {rollback_deployment_id}")
        
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        sys.exit(1)


def health_check_command(args) -> None:
    """Execute health check command."""
    try:
        deployment = ModelDeployment()
        
        health_status = deployment.check_deployment_health(args.deployment_id)
        
        print(f"\nHealth Check Results for {args.deployment_id}:")
        print("-" * 50)
        print(f"Overall Status: {'HEALTHY' if health_status['healthy'] else 'UNHEALTHY'}")
        print(f"Last Check: {health_status['last_check']}")
        
        if health_status.get('checks'):
            print("\nDetailed Checks:")
            for check_name, check_result in health_status['checks'].items():
                status = check_result.get('status', 'unknown').upper()
                message = check_result.get('message', 'No message')
                print(f"  {check_name}: {status} - {message}")
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for the deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy and manage fraud detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy a model')
    deploy_parser.add_argument('model_id', help='Model ID to deploy')
    deploy_parser.add_argument('environment', choices=['development', 'staging', 'production'],
                              help='Target environment')
    deploy_parser.add_argument('--config', '-c', help='Deployment configuration file')
    deploy_parser.add_argument('--deployment-name', help='Custom deployment name')
    deploy_parser.add_argument('--api-endpoint', help='API endpoint URL')
    deploy_parser.add_argument('--dry-run', action='store_true', help='Perform dry run without actual deployment')
    deploy_parser.add_argument('--health-check', action='store_true', help='Check deployment health after deployment')
    deploy_parser.set_defaults(func=deploy_model_command)
    
    # List models command
    list_models_parser = subparsers.add_parser('list-models', help='List available models')
    list_models_parser.add_argument('--model-name', help='Filter by model name')
    list_models_parser.add_argument('--model-type', help='Filter by model type')
    list_models_parser.add_argument('--status', help='Filter by status')
    list_models_parser.set_defaults(func=list_models_command)
    
    # List deployments command
    list_deployments_parser = subparsers.add_parser('list-deployments', help='List deployments')
    list_deployments_parser.add_argument('--environment', help='Filter by environment')
    list_deployments_parser.add_argument('--status', help='Filter by status')
    list_deployments_parser.set_defaults(func=list_deployments_command)
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback deployment')
    rollback_parser.add_argument('environment', choices=['development', 'staging', 'production'],
                                help='Environment to rollback')
    rollback_parser.add_argument('--target-deployment', help='Specific deployment to rollback to')
    rollback_parser.set_defaults(func=rollback_command)
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check deployment health')
    health_parser.add_argument('deployment_id', help='Deployment ID to check')
    health_parser.set_defaults(func=health_check_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()