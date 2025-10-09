"""
Correlation ID middleware for tracking requests across the system.

This module provides middleware for FastAPI and other frameworks to automatically
generate and track correlation IDs for all requests.
"""

import uuid
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time

from config.logging_config import set_correlation_id, get_correlation_id, get_logger
from src.utils.performance_metrics import start_request_tracking, end_request_tracking

logger = get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle correlation ID tracking for HTTP requests.
    
    Automatically generates correlation IDs for requests that don't have them,
    and ensures the correlation ID is available throughout the request lifecycle.
    """
    
    def __init__(self, 
                 app,
                 header_name: str = "X-Correlation-ID",
                 generate_if_missing: bool = True,
                 enable_performance_tracking: bool = True):
        """
        Initialize the correlation ID middleware.
        
        Args:
            app: FastAPI application instance
            header_name: HTTP header name for correlation ID
            generate_if_missing: Whether to generate correlation ID if not provided
            enable_performance_tracking: Whether to track performance metrics
        """
        super().__init__(app)
        self.header_name = header_name
        self.generate_if_missing = generate_if_missing
        self.enable_performance_tracking = enable_performance_tracking
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and response with correlation ID tracking.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler in the chain
            
        Returns:
            HTTP response with correlation ID header
        """
        # Extract or generate correlation ID
        correlation_id = request.headers.get(self.header_name)
        
        if not correlation_id and self.generate_if_missing:
            correlation_id = str(uuid.uuid4())
        
        # Set correlation ID in context
        if correlation_id:
            set_correlation_id(correlation_id)
        
        # Start performance tracking if enabled
        if self.enable_performance_tracking and correlation_id:
            endpoint = f"{request.method} {request.url.path}"
            start_request_tracking(endpoint, request.method)
        
        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        start_time = time.time()
        error_message = None
        status_code = 200
        
        try:
            # Process the request
            response = await call_next(request)
            status_code = response.status_code
            
            # Add correlation ID to response headers
            if correlation_id:
                response.headers[self.header_name] = correlation_id
            
            return response
            
        except Exception as e:
            error_message = str(e)
            status_code = 500
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "error": error_message,
                    "exception_type": type(e).__name__
                }
            )
            raise
            
        finally:
            # Calculate request duration
            duration_ms = (time.time() - start_time) * 1000
            
            # End performance tracking
            if self.enable_performance_tracking and correlation_id:
                end_request_tracking(
                    correlation_id,
                    status_code,
                    error_message,
                    {"duration_ms": duration_ms}
                )
            
            # Log request completion
            logger.info(
                f"Request completed: {request.method} {request.url.path}",
                extra={
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "error": error_message
                }
            )


def get_or_generate_correlation_id(request: Request, header_name: str = "X-Correlation-ID") -> str:
    """
    Get correlation ID from request or generate a new one.
    
    Args:
        request: HTTP request
        header_name: HTTP header name for correlation ID
        
    Returns:
        Correlation ID string
    """
    correlation_id = request.headers.get(header_name)
    
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    
    return correlation_id


def add_correlation_id_to_response(response: Response, correlation_id: str, header_name: str = "X-Correlation-ID") -> None:
    """
    Add correlation ID to response headers.
    
    Args:
        response: HTTP response
        correlation_id: Correlation ID to add
        header_name: HTTP header name for correlation ID
    """
    response.headers[header_name] = correlation_id


class CorrelationIdContext:
    """
    Context manager for setting correlation ID in non-HTTP contexts.
    
    Useful for background tasks, scheduled jobs, or other non-request contexts
    where you want to maintain correlation ID tracking.
    """
    
    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize the correlation ID context.
        
        Args:
            correlation_id: Correlation ID to set. If None, generates a new one.
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.previous_correlation_id = None
    
    def __enter__(self) -> str:
        """
        Enter the context and set the correlation ID.
        
        Returns:
            The correlation ID being set
        """
        self.previous_correlation_id = get_correlation_id()
        set_correlation_id(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context and restore the previous correlation ID.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        if self.previous_correlation_id:
            set_correlation_id(self.previous_correlation_id)


def with_correlation_id(correlation_id: Optional[str] = None):
    """
    Decorator to run a function with a correlation ID context.
    
    Args:
        correlation_id: Correlation ID to use. If None, generates a new one.
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with CorrelationIdContext(correlation_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def with_correlation_id_async(correlation_id: Optional[str] = None):
    """
    Async decorator to run a function with a correlation ID context.
    
    Args:
        correlation_id: Correlation ID to use. If None, generates a new one.
        
    Returns:
        Async decorator function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with CorrelationIdContext(correlation_id):
                return await func(*args, **kwargs)
        return wrapper
    return decorator