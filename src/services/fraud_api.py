"""
FastAPI endpoints for fraud detection service.

This module provides REST API endpoints for real-time fraud detection,
batch processing, and service health monitoring.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import pandas as pd
import uvicorn

from src.services.fraud_detector import FraudDetector
from src.utils.config import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="REST API for real-time fraud detection and risk assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global fraud detector instance
fraud_detector: Optional[FraudDetector] = None


# Pydantic models for request/response validation
class TransactionRequest(BaseModel):
    """Single transaction request model."""
    step: int = Field(..., ge=0, description="Time step (0-744)")
    type: str = Field(..., description="Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)")
    amount: float = Field(..., ge=0, description="Transaction amount")
    nameOrig: str = Field(..., description="Origin customer ID")
    oldbalanceOrg: float = Field(..., ge=0, description="Origin balance before transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Origin balance after transaction")
    nameDest: str = Field(..., description="Destination customer ID")
    oldbalanceDest: float = Field(..., ge=0, description="Destination balance before transaction")
    newbalanceDest: float = Field(..., ge=0, description="Destination balance after transaction")
    
    @validator('type')
    def validate_transaction_type(cls, v):
        valid_types = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        if v not in valid_types:
            raise ValueError(f'Transaction type must be one of: {valid_types}')
        return v


class BatchTransactionRequest(BaseModel):
    """Batch transaction request model."""
    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=1000, 
                                                  description="List of transactions (max 1000)")


class FraudScoreResponse(BaseModel):
    """Single transaction fraud score response."""
    transaction_id: Optional[str] = Field(None, description="Transaction identifier")
    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability score (0-1)")
    risk_level: str = Field(..., description="Risk level (LOW, MEDIUM, HIGH)")
    is_fraud_prediction: bool = Field(..., description="Binary fraud prediction")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    processed_at: str = Field(..., description="Processing timestamp")


class FraudExplanationResponse(BaseModel):
    """Detailed fraud explanation response."""
    transaction_id: Optional[str] = Field(None, description="Transaction identifier")
    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability score (0-1)")
    risk_level: str = Field(..., description="Risk level (LOW, MEDIUM, HIGH)")
    is_fraud_prediction: bool = Field(..., description="Binary fraud prediction")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    risk_factors: Dict[str, Any] = Field(..., description="Identified risk factors")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Model feature importance")
    explanation_text: str = Field(..., description="Human-readable explanation")
    recommendations: List[str] = Field(..., description="Recommended actions")
    processed_at: str = Field(..., description="Processing timestamp")


class BatchFraudResponse(BaseModel):
    """Batch fraud detection response."""
    total_transactions: int = Field(..., description="Total number of transactions processed")
    fraud_detected: int = Field(..., description="Number of fraudulent transactions detected")
    high_risk_count: int = Field(..., description="Number of high-risk transactions")
    medium_risk_count: int = Field(..., description="Number of medium-risk transactions")
    low_risk_count: int = Field(..., description="Number of low-risk transactions")
    results: List[FraudScoreResponse] = Field(..., description="Individual transaction results")
    processed_at: str = Field(..., description="Processing timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    service_name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status (active, inactive, error)")
    model_loaded: bool = Field(..., description="Whether fraud detection model is loaded")
    risk_threshold: float = Field(..., description="Current risk threshold")
    high_risk_threshold: float = Field(..., description="Current high-risk threshold")
    explanations_enabled: bool = Field(..., description="Whether explanations are enabled")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata")
    timestamp: str = Field(..., description="Status check timestamp")


class ThresholdUpdateRequest(BaseModel):
    """Request model for updating fraud detection thresholds."""
    risk_threshold: Optional[float] = Field(None, ge=0, le=1, description="Risk threshold (0-1)")
    high_risk_threshold: Optional[float] = Field(None, ge=0, le=1, description="High-risk threshold (0-1)")


# Dependency to get fraud detector instance
def get_fraud_detector() -> FraudDetector:
    """Get the fraud detector instance."""
    global fraud_detector
    if fraud_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud detection service is not initialized. Please check service health."
        )
    return fraud_detector


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify service status and model availability.
    
    Returns:
        HealthCheckResponse: Service health status and configuration
    """
    try:
        global fraud_detector
        
        if fraud_detector is None:
            return HealthCheckResponse(
                service_name="FraudDetector",
                status="inactive",
                model_loaded=False,
                risk_threshold=0.5,
                high_risk_threshold=0.8,
                explanations_enabled=True,
                timestamp=datetime.now().isoformat()
            )
        
        # Get detailed status from fraud detector
        status_info = fraud_detector.get_service_status()
        
        return HealthCheckResponse(**status_info)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/predict", response_model=FraudScoreResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """
    Score a single transaction for fraud risk.
    
    Args:
        transaction: Transaction data to analyze
        detector: Fraud detector service instance
        
    Returns:
        FraudScoreResponse: Fraud score and risk assessment
    """
    try:
        # Convert Pydantic model to dictionary
        transaction_dict = transaction.dict()
        
        # Get fraud score
        fraud_score = detector.score_transaction(transaction_dict)
        
        # Categorize risk level
        risk_levels = detector._categorize_risk_level([fraud_score])
        risk_level = risk_levels[0]
        
        # Calculate confidence
        confidence = detector._calculate_confidence(fraud_score)
        
        # Determine fraud prediction
        is_fraud = fraud_score >= detector.risk_threshold
        
        response = FraudScoreResponse(
            fraud_score=fraud_score,
            risk_level=risk_level,
            is_fraud_prediction=is_fraud,
            confidence=confidence,
            processed_at=datetime.now().isoformat()
        )
        
        logger.info(f"Single transaction scored: fraud_score={fraud_score:.3f}, risk_level={risk_level}")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in fraud prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transaction data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in fraud prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fraud prediction failed: {str(e)}"
        )


@app.post("/predict/explain", response_model=FraudExplanationResponse)
async def predict_fraud_with_explanation(
    transaction: TransactionRequest,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """
    Score a single transaction with detailed fraud explanation.
    
    Args:
        transaction: Transaction data to analyze
        detector: Fraud detector service instance
        
    Returns:
        FraudExplanationResponse: Detailed fraud analysis with explanations
    """
    try:
        # Convert Pydantic model to dictionary
        transaction_dict = transaction.dict()
        
        # Get detailed fraud explanation
        explanation = detector.get_fraud_explanation(transaction_dict)
        
        response = FraudExplanationResponse(**explanation)
        
        logger.info(f"Transaction explained: fraud_score={explanation['fraud_score']:.3f}, "
                   f"risk_level={explanation['risk_level']}")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in fraud explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transaction data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in fraud explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fraud explanation failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchFraudResponse)
async def predict_fraud_batch(
    batch_request: BatchTransactionRequest,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """
    Perform batch fraud prediction on multiple transactions.
    
    Args:
        batch_request: Batch of transactions to analyze
        detector: Fraud detector service instance
        
    Returns:
        BatchFraudResponse: Batch prediction results with summary statistics
    """
    try:
        # Convert transactions to DataFrame
        transactions_data = [tx.dict() for tx in batch_request.transactions]
        transactions_df = pd.DataFrame(transactions_data)
        
        # Perform batch prediction
        results_df = detector.batch_predict(transactions_df)
        
        # Create individual response objects
        individual_results = []
        for _, row in results_df.iterrows():
            result = FraudScoreResponse(
                fraud_score=float(row['fraud_score']),
                risk_level=row['risk_level'],
                is_fraud_prediction=bool(row['fraud_prediction']),
                confidence=detector._calculate_confidence(row['fraud_score']),
                processed_at=row['processed_at']
            )
            individual_results.append(result)
        
        # Calculate summary statistics
        total_transactions = len(results_df)
        fraud_detected = int(results_df['fraud_prediction'].sum())
        high_risk_count = int((results_df['risk_level'] == 'HIGH').sum())
        medium_risk_count = int((results_df['risk_level'] == 'MEDIUM').sum())
        low_risk_count = int((results_df['risk_level'] == 'LOW').sum())
        
        response = BatchFraudResponse(
            total_transactions=total_transactions,
            fraud_detected=fraud_detected,
            high_risk_count=high_risk_count,
            medium_risk_count=medium_risk_count,
            low_risk_count=low_risk_count,
            results=individual_results,
            processed_at=datetime.now().isoformat()
        )
        
        logger.info(f"Batch prediction completed: {total_transactions} transactions, "
                   f"{fraud_detected} fraud detected, {high_risk_count} high-risk")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid batch data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.put("/config/thresholds", response_model=Dict[str, Any])
async def update_thresholds(
    threshold_request: ThresholdUpdateRequest,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """
    Update fraud detection thresholds.
    
    Args:
        threshold_request: New threshold values
        detector: Fraud detector service instance
        
    Returns:
        Dict: Updated configuration
    """
    try:
        # Update thresholds
        detector.update_thresholds(
            risk_threshold=threshold_request.risk_threshold,
            high_risk_threshold=threshold_request.high_risk_threshold
        )
        
        # Return updated configuration
        response = {
            "message": "Thresholds updated successfully",
            "risk_threshold": detector.risk_threshold,
            "high_risk_threshold": detector.high_risk_threshold,
            "updated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Thresholds updated: risk={detector.risk_threshold}, "
                   f"high_risk={detector.high_risk_threshold}")
        
        return response
        
    except ValueError as e:
        logger.error(f"Invalid threshold values: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid threshold values: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error updating thresholds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threshold update failed: {str(e)}"
        )


@app.get("/status", response_model=Dict[str, Any])
async def get_service_status(detector: FraudDetector = Depends(get_fraud_detector)):
    """
    Get detailed service status and configuration.
    
    Args:
        detector: Fraud detector service instance
        
    Returns:
        Dict: Detailed service status
    """
    try:
        status_info = detector.get_service_status()
        
        # Add API-specific information
        status_info.update({
            "api_version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "predict_explain": "/predict/explain",
                "batch_predict": "/predict/batch",
                "update_thresholds": "/config/thresholds",
                "status": "/status"
            }
        })
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status retrieval failed: {str(e)}"
        )


# Service initialization and management functions

def initialize_fraud_detector(model_path: Optional[str] = None,
                            risk_threshold: float = 0.5,
                            high_risk_threshold: float = 0.8,
                            enable_explanations: bool = True) -> None:
    """
    Initialize the fraud detector service.
    
    Args:
        model_path: Path to trained model file
        risk_threshold: Fraud classification threshold
        high_risk_threshold: High-risk alert threshold
        enable_explanations: Whether to enable detailed explanations
    """
    global fraud_detector
    
    try:
        # Initialize fraud detector
        fraud_detector = FraudDetector(
            model=None,  # Model will be loaded separately
            risk_threshold=risk_threshold,
            high_risk_threshold=high_risk_threshold,
            enable_explanations=enable_explanations
        )
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            fraud_detector.load_model(model_path)
            logger.info(f"Fraud detector initialized with model: {model_path}")
        else:
            logger.warning("Fraud detector initialized without model - predictions will fail")
        
    except Exception as e:
        logger.error(f"Failed to initialize fraud detector: {e}")
        raise


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Server host address
        port: Server port
        reload: Enable auto-reload for development
    """
    logger.info(f"Starting Fraud Detection API server on {host}:{port}")
    
    uvicorn.run(
        "src.services.fraud_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Initialize service (for development/testing)
    initialize_fraud_detector()
    
    # Run server
    run_server(reload=True)