"""
Model deployment utilities for fraud detection models.

This module provides functionality for deploying trained models to production,
including configuration management, health checks, and deployment validation.
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

from config.settings import settings
from config.logging_config import get_logger
from .model_registry import ModelRegistry, ModelStatus

logger = get_logger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLBACK = "rollback"
    STOPPED = "stopped"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    model_id: str
    environment: DeploymentEnvironment
    deployment_name: str
    api_endpoint: str
    health_check_endpoint: str
    resource_limits: Dict[str, Any]
    environment_variables: Dict[str, str]
    scaling_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    rollback_config: Dict[str, Any]


@dataclass
class DeploymentRecord:
    """Deployment record for tracking deployments."""
    deployment_id: str
    model_id: str
    environment: str
    deployment_name: str
    status: str
    created_at: str
    updated_at: str
    deployed_at: Optional[str]
    config: Dict[str, Any]
    health_status: Dict[str, Any]
    metrics: Dict[str, Any]
    logs: List[str]


class ModelDeployment:
    """
    Model deployment manager for fraud detection models.
    
    Handles deployment of trained models to different environments with
    configuration management, health monitoring, and rollback capabilities.
    """
    
    def __init__(self, 
                 registry: Optional[ModelRegistry] = None,
                 deployment_dir: Optional[Path] = None):
        """
        Initialize the model deployment manager.
        
        Args:
            registry: Model registry instance
            deployment_dir: Directory for deployment artifacts
        """
        self.registry = registry or ModelRegistry()
        self.deployment_dir = deployment_dir or (settings.project_root / "deployments")
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Deployment tracking file
        self.deployments_file = self.deployment_dir / "deployments.json"
        
        # Initialize deployments tracking
        if not self.deployments_file.exists():
            self._initialize_deployments()
        
        logger.info(f"Model deployment manager initialized at: {self.deployment_dir}")
    
    def _initialize_deployments(self) -> None:
        """Initialize empty deployments tracking."""
        deployments_data = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "deployments": {},
            "active_deployments": {},
            "metadata": {
                "version": "1.0.0",
                "description": "Model Deployment Tracking"
            }
        }
        
        with open(self.deployments_file, 'w') as f:
            json.dump(deployments_data, f, indent=2)
        
        logger.info("Initialized deployments tracking")
    
    def _load_deployments(self) -> Dict[str, Any]:
        """Load deployments data from file."""
        try:
            with open(self.deployments_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading deployments: {e}")
            self._initialize_deployments()
            return self._load_deployments()
    
    def _save_deployments(self, deployments_data: Dict[str, Any]) -> None:
        """Save deployments data to file."""
        deployments_data["updated_at"] = datetime.now().isoformat()
        
        with open(self.deployments_file, 'w') as f:
            json.dump(deployments_data, f, indent=2)
    
    def _generate_deployment_id(self, model_id: str, environment: str) -> str:
        """Generate unique deployment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"deploy_{model_id}_{environment}_{timestamp}"
    
    def create_deployment_config(self,
                               model_id: str,
                               environment: DeploymentEnvironment,
                               deployment_name: str,
                               api_endpoint: str = "http://localhost:8000",
                               resource_limits: Dict[str, Any] = None,
                               environment_variables: Dict[str, str] = None,
                               scaling_config: Dict[str, Any] = None) -> DeploymentConfig:
        """
        Create a deployment configuration.
        
        Args:
            model_id: Model ID to deploy
            environment: Target environment
            deployment_name: Name for the deployment
            api_endpoint: API endpoint URL
            resource_limits: Resource limits configuration
            environment_variables: Environment variables
            scaling_config: Scaling configuration
            
        Returns:
            DeploymentConfig instance
        """
        # Default configurations
        default_resource_limits = {
            "cpu": "1000m",
            "memory": "2Gi",
            "gpu": 0
        }
        
        default_scaling_config = {
            "min_replicas": 1,
            "max_replicas": 3,
            "target_cpu_utilization": 70
        }
        
        default_monitoring_config = {
            "health_check_interval": 30,
            "metrics_collection": True,
            "log_level": "INFO"
        }
        
        default_rollback_config = {
            "enabled": True,
            "failure_threshold": 3,
            "rollback_timeout": 300
        }
        
        config = DeploymentConfig(
            model_id=model_id,
            environment=environment,
            deployment_name=deployment_name,
            api_endpoint=api_endpoint,
            health_check_endpoint=f"{api_endpoint}/health",
            resource_limits=resource_limits or default_resource_limits,
            environment_variables=environment_variables or {},
            scaling_config=scaling_config or default_scaling_config,
            monitoring_config=default_monitoring_config,
            rollback_config=default_rollback_config
        )
        
        return config
    
    def prepare_deployment_artifacts(self, 
                                   deployment_config: DeploymentConfig) -> Path:
        """
        Prepare deployment artifacts for a model.
        
        Args:
            deployment_config: Deployment configuration
            
        Returns:
            Path to deployment artifacts directory
        """
        model_id = deployment_config.model_id
        
        # Verify model exists and is trained
        model_info = self.registry.get_model_info(model_id)
        if model_info["status"] not in [ModelStatus.TRAINED.value, ModelStatus.VALIDATED.value]:
            raise ValueError(f"Model {model_id} is not ready for deployment (status: {model_info['status']})")
        
        # Create deployment directory
        deployment_id = self._generate_deployment_id(model_id, deployment_config.environment.value)
        artifact_dir = self.deployment_dir / deployment_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_dir = self.registry.registry_path / model_id
        model_artifact_dir = artifact_dir / "model"
        shutil.copytree(model_dir, model_artifact_dir)
        
        # Create deployment configuration file
        config_file = artifact_dir / "deployment_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(asdict(deployment_config), f, default_flow_style=False)
        
        # Create Docker configuration
        self._create_docker_config(artifact_dir, deployment_config)
        
        # Create API service configuration
        self._create_api_config(artifact_dir, deployment_config)
        
        # Create monitoring configuration
        self._create_monitoring_config(artifact_dir, deployment_config)
        
        logger.info(f"Prepared deployment artifacts at: {artifact_dir}")
        return artifact_dir
    
    def _create_docker_config(self, 
                            artifact_dir: Path, 
                            config: DeploymentConfig) -> None:
        """Create Docker configuration files."""
        # Dockerfile
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV MODEL_ID={config.model_id}
ENV ENVIRONMENT={config.environment.value}
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f {config.health_check_endpoint} || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "src.services.fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open(artifact_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile_content.strip())
        
        # Docker Compose
        compose_content = {
            "version": "3.8",
            "services": {
                "fraud-detection-api": {
                    "build": ".",
                    "ports": ["8000:8000"],
                    "environment": config.environment_variables,
                    "deploy": {
                        "resources": {
                            "limits": config.resource_limits,
                            "reservations": {
                                "cpus": "0.5",
                                "memory": "1Gi"
                            }
                        }
                    },
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", config.health_check_endpoint],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3,
                        "start_period": "60s"
                    }
                }
            }
        }
        
        with open(artifact_dir / "docker-compose.yml", 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False)
    
    def _create_api_config(self, 
                          artifact_dir: Path, 
                          config: DeploymentConfig) -> None:
        """Create API service configuration."""
        api_config = {
            "model_id": config.model_id,
            "environment": config.environment.value,
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": config.scaling_config.get("min_replicas", 1),
                "timeout": 30
            },
            "model": {
                "prediction_timeout": 5,
                "batch_size_limit": 1000,
                "cache_predictions": True
            },
            "monitoring": config.monitoring_config,
            "logging": {
                "level": config.monitoring_config.get("log_level", "INFO"),
                "format": "json"
            }
        }
        
        with open(artifact_dir / "api_config.yaml", 'w') as f:
            yaml.dump(api_config, f, default_flow_style=False)
    
    def _create_monitoring_config(self, 
                                artifact_dir: Path, 
                                config: DeploymentConfig) -> None:
        """Create monitoring configuration."""
        monitoring_config = {
            "metrics": {
                "enabled": config.monitoring_config.get("metrics_collection", True),
                "endpoint": "/metrics",
                "collection_interval": 60
            },
            "health_checks": {
                "enabled": True,
                "endpoint": "/health",
                "interval": config.monitoring_config.get("health_check_interval", 30),
                "timeout": 10
            },
            "alerts": {
                "enabled": True,
                "thresholds": {
                    "error_rate": 0.05,
                    "response_time_p95": 1000,
                    "cpu_utilization": 80,
                    "memory_utilization": 85
                }
            },
            "logging": {
                "level": config.monitoring_config.get("log_level", "INFO"),
                "structured": True,
                "correlation_id": True
            }
        }
        
        with open(artifact_dir / "monitoring_config.yaml", 'w') as f:
            yaml.dump(monitoring_config, f, default_flow_style=False)
    
    def deploy_model(self, 
                    deployment_config: DeploymentConfig,
                    dry_run: bool = False) -> str:
        """
        Deploy a model to the specified environment.
        
        Args:
            deployment_config: Deployment configuration
            dry_run: If True, validate deployment without actually deploying
            
        Returns:
            Deployment ID
        """
        model_id = deployment_config.model_id
        environment = deployment_config.environment.value
        
        # Generate deployment ID
        deployment_id = self._generate_deployment_id(model_id, environment)
        
        # Prepare artifacts
        artifact_dir = self.prepare_deployment_artifacts(deployment_config)
        
        # Create deployment record
        deployment_record = DeploymentRecord(
            deployment_id=deployment_id,
            model_id=model_id,
            environment=environment,
            deployment_name=deployment_config.deployment_name,
            status=DeploymentStatus.PENDING.value,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            deployed_at=None,
            config=asdict(deployment_config),
            health_status={},
            metrics={},
            logs=[]
        )
        
        # Save deployment record
        deployments_data = self._load_deployments()
        deployments_data["deployments"][deployment_id] = asdict(deployment_record)
        self._save_deployments(deployments_data)
        
        if dry_run:
            logger.info(f"Dry run completed for deployment {deployment_id}")
            return deployment_id
        
        try:
            # Update status to deploying
            self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYING)
            
            # Execute deployment
            self._execute_deployment(deployment_id, artifact_dir, deployment_config)
            
            # Update model status in registry
            self.registry.update_model_status(model_id, ModelStatus.DEPLOYED)
            
            # Update deployment status
            self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYED)
            
            # Update active deployments
            deployments_data = self._load_deployments()
            deployments_data["active_deployments"][environment] = deployment_id
            deployments_data["deployments"][deployment_id]["deployed_at"] = datetime.now().isoformat()
            self._save_deployments(deployments_data)
            
            logger.info(f"Successfully deployed model {model_id} as {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self._update_deployment_status(deployment_id, DeploymentStatus.FAILED)
            self._add_deployment_log(deployment_id, f"Deployment failed: {str(e)}")
            raise
    
    def _execute_deployment(self, 
                          deployment_id: str,
                          artifact_dir: Path,
                          config: DeploymentConfig) -> None:
        """Execute the actual deployment."""
        # For this implementation, we'll create a simple deployment script
        # In a real production environment, this would integrate with container orchestration
        
        deployment_script = artifact_dir / "deploy.sh"
        script_content = f"""#!/bin/bash
set -e

echo "Starting deployment {deployment_id}"

# Build Docker image
docker build -t fraud-detection-{config.model_id}:latest .

# Stop existing container if running
docker stop fraud-detection-{config.environment.value} || true
docker rm fraud-detection-{config.environment.value} || true

# Run new container
docker run -d \\
    --name fraud-detection-{config.environment.value} \\
    -p 8000:8000 \\
    --restart unless-stopped \\
    fraud-detection-{config.model_id}:latest

echo "Deployment {deployment_id} completed"
"""
        
        with open(deployment_script, 'w') as f:
            f.write(script_content)
        
        deployment_script.chmod(0o755)
        
        # Execute deployment script
        result = subprocess.run(
            [str(deployment_script)],
            cwd=artifact_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Deployment script failed: {result.stderr}")
        
        self._add_deployment_log(deployment_id, f"Deployment executed successfully: {result.stdout}")
    
    def _update_deployment_status(self, 
                                deployment_id: str, 
                                status: DeploymentStatus) -> None:
        """Update deployment status."""
        deployments_data = self._load_deployments()
        
        if deployment_id not in deployments_data["deployments"]:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployments_data["deployments"][deployment_id]["status"] = status.value
        deployments_data["deployments"][deployment_id]["updated_at"] = datetime.now().isoformat()
        
        self._save_deployments(deployments_data)
        logger.info(f"Updated deployment {deployment_id} status to {status.value}")
    
    def _add_deployment_log(self, deployment_id: str, message: str) -> None:
        """Add log entry to deployment."""
        deployments_data = self._load_deployments()
        
        if deployment_id not in deployments_data["deployments"]:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        log_entry = f"{datetime.now().isoformat()}: {message}"
        deployments_data["deployments"][deployment_id]["logs"].append(log_entry)
        
        self._save_deployments(deployments_data)
    
    def check_deployment_health(self, deployment_id: str) -> Dict[str, Any]:
        """
        Check the health of a deployment.
        
        Args:
            deployment_id: Deployment ID to check
            
        Returns:
            Health status dictionary
        """
        deployments_data = self._load_deployments()
        
        if deployment_id not in deployments_data["deployments"]:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = deployments_data["deployments"][deployment_id]
        config = deployment["config"]
        
        health_status = {
            "deployment_id": deployment_id,
            "status": deployment["status"],
            "last_check": datetime.now().isoformat(),
            "healthy": False,
            "checks": {}
        }
        
        try:
            # Check if container is running (simplified check)
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name=fraud-detection-{config['environment']}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and "Up" in result.stdout:
                health_status["checks"]["container"] = {"status": "healthy", "message": "Container is running"}
                health_status["healthy"] = True
            else:
                health_status["checks"]["container"] = {"status": "unhealthy", "message": "Container not running"}
            
        except Exception as e:
            health_status["checks"]["container"] = {"status": "error", "message": str(e)}
        
        # Update deployment health status
        deployments_data["deployments"][deployment_id]["health_status"] = health_status
        self._save_deployments(deployments_data)
        
        return health_status
    
    def rollback_deployment(self, 
                          environment: str,
                          target_deployment_id: Optional[str] = None) -> str:
        """
        Rollback to a previous deployment.
        
        Args:
            environment: Environment to rollback
            target_deployment_id: Specific deployment to rollback to (optional)
            
        Returns:
            Rollback deployment ID
        """
        deployments_data = self._load_deployments()
        
        # Get current active deployment
        current_deployment_id = deployments_data["active_deployments"].get(environment)
        if not current_deployment_id:
            raise ValueError(f"No active deployment found for environment {environment}")
        
        # Find target deployment
        if target_deployment_id:
            if target_deployment_id not in deployments_data["deployments"]:
                raise ValueError(f"Target deployment {target_deployment_id} not found")
            target_deployment = deployments_data["deployments"][target_deployment_id]
        else:
            # Find previous successful deployment
            env_deployments = [
                (dep_id, dep_info) for dep_id, dep_info in deployments_data["deployments"].items()
                if dep_info["environment"] == environment and dep_info["status"] == DeploymentStatus.DEPLOYED.value
                and dep_id != current_deployment_id
            ]
            
            if not env_deployments:
                raise ValueError(f"No previous deployment found for rollback in environment {environment}")
            
            # Sort by creation date and get the most recent
            env_deployments.sort(key=lambda x: x[1]["created_at"], reverse=True)
            target_deployment_id, target_deployment = env_deployments[0]
        
        # Execute rollback
        logger.info(f"Rolling back environment {environment} from {current_deployment_id} to {target_deployment_id}")
        
        # Update current deployment status
        self._update_deployment_status(current_deployment_id, DeploymentStatus.ROLLBACK)
        
        # Reactivate target deployment
        deployments_data["active_deployments"][environment] = target_deployment_id
        self._save_deployments(deployments_data)
        
        # Log rollback
        self._add_deployment_log(current_deployment_id, f"Rolled back to deployment {target_deployment_id}")
        self._add_deployment_log(target_deployment_id, f"Reactivated via rollback from {current_deployment_id}")
        
        return target_deployment_id
    
    def list_deployments(self, 
                        environment: Optional[str] = None,
                        status: Optional[DeploymentStatus] = None) -> List[Dict[str, Any]]:
        """
        List deployments with optional filtering.
        
        Args:
            environment: Filter by environment
            status: Filter by status
            
        Returns:
            List of deployment information
        """
        deployments_data = self._load_deployments()
        deployments = []
        
        for deployment_id, deployment_info in deployments_data["deployments"].items():
            # Apply filters
            if environment and deployment_info["environment"] != environment:
                continue
            if status and deployment_info["status"] != status.value:
                continue
            
            deployments.append(deployment_info)
        
        # Sort by creation date (newest first)
        deployments.sort(key=lambda x: x["created_at"], reverse=True)
        
        return deployments
    
    def get_active_deployments(self) -> Dict[str, str]:
        """
        Get currently active deployments by environment.
        
        Returns:
            Dictionary mapping environment to deployment ID
        """
        deployments_data = self._load_deployments()
        return deployments_data["active_deployments"].copy()
    
    def stop_deployment(self, deployment_id: str) -> None:
        """
        Stop a running deployment.
        
        Args:
            deployment_id: Deployment ID to stop
        """
        deployments_data = self._load_deployments()
        
        if deployment_id not in deployments_data["deployments"]:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = deployments_data["deployments"][deployment_id]
        
        try:
            # Stop container
            result = subprocess.run(
                ["docker", "stop", f"fraud-detection-{deployment['environment']}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self._update_deployment_status(deployment_id, DeploymentStatus.STOPPED)
                self._add_deployment_log(deployment_id, "Deployment stopped successfully")
                
                # Remove from active deployments
                if deployment["environment"] in deployments_data["active_deployments"]:
                    if deployments_data["active_deployments"][deployment["environment"]] == deployment_id:
                        del deployments_data["active_deployments"][deployment["environment"]]
                        self._save_deployments(deployments_data)
                
                logger.info(f"Stopped deployment {deployment_id}")
            else:
                raise RuntimeError(f"Failed to stop deployment: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error stopping deployment {deployment_id}: {e}")
            self._add_deployment_log(deployment_id, f"Error stopping deployment: {str(e)}")
            raise