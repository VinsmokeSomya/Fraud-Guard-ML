"""
Performance metrics collection and monitoring utilities.

This module provides comprehensive performance monitoring capabilities including
request timing, resource usage tracking, and custom metrics collection.
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
import json
from pathlib import Path

from config.logging_config import get_logger, performance_logger, get_correlation_id

logger = get_logger(__name__)
perf_logger = performance_logger()


@dataclass
class PerformanceMetric:
    """Data class for storing performance metrics."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    correlation_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class RequestMetrics:
    """Data class for storing request-level metrics."""
    correlation_id: str
    endpoint: str
    method: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


class MetricsCollector:
    """
    Centralized metrics collection and aggregation system.
    
    Provides thread-safe collection of performance metrics with
    automatic aggregation and export capabilities.
    """
    
    def __init__(self, 
                 max_metrics_in_memory: int = 10000,
                 aggregation_window_minutes: int = 5,
                 enable_system_metrics: bool = True):
        """
        Initialize the MetricsCollector.
        
        Args:
            max_metrics_in_memory: Maximum number of metrics to keep in memory
            aggregation_window_minutes: Window size for metric aggregation
            enable_system_metrics: Whether to collect system-level metrics
        """
        self.max_metrics_in_memory = max_metrics_in_memory
        self.aggregation_window = timedelta(minutes=aggregation_window_minutes)
        self.enable_system_metrics = enable_system_metrics
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._metrics: deque = deque(maxlen=max_metrics_in_memory)
        self._request_metrics: Dict[str, RequestMetrics] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        
        # System metrics tracking
        self._system_metrics_enabled = enable_system_metrics
        self._last_system_metrics_time = datetime.now()
        
        logger.info("MetricsCollector initialized")
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     unit: str = "count",
                     tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a single metric value.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Optional tags for the metric
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            correlation_id=get_correlation_id(),
            tags=tags or {}
        )
        
        with self._lock:
            self._metrics.append(metric)
        
        # Log performance metric
        perf_logger.info(
            f"Metric recorded: {name}={value}{unit}",
            extra={
                "metric_name": name,
                "metric_value": value,
                "metric_unit": unit,
                "metric_tags": tags
            }
        )
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Increment value
            tags: Optional tags
        """
        with self._lock:
            self._counters[name] += value
        
        self.record_metric(name, value, "count", tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Gauge value
            tags: Optional tags
        """
        with self._lock:
            self._gauges[name] = value
        
        self.record_metric(name, value, "gauge", tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a value in a histogram.
        
        Args:
            name: Histogram name
            value: Value to record
            tags: Optional tags
        """
        with self._lock:
            self._histograms[name].append(value)
            # Keep only recent values to prevent memory bloat
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
        
        self.record_metric(name, value, "histogram", tags)
    
    @contextmanager
    def time_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            tags: Optional tags
        """
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Record timing
            self.record_histogram(f"{operation_name}_duration_ms", duration_ms, tags)
            
            # Record resource usage if system metrics are enabled
            if self._system_metrics_enabled:
                end_cpu = psutil.cpu_percent()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                self.record_metric(f"{operation_name}_cpu_usage", end_cpu - start_cpu, "percent", tags)
                self.record_metric(f"{operation_name}_memory_delta", end_memory - start_memory, "MB", tags)
    
    def start_request_tracking(self, endpoint: str, method: str = "POST") -> str:
        """
        Start tracking metrics for a request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            
        Returns:
            Correlation ID for the request
        """
        correlation_id = get_correlation_id() or f"req_{int(time.time() * 1000)}"
        
        request_metrics = RequestMetrics(
            correlation_id=correlation_id,
            endpoint=endpoint,
            method=method,
            start_time=datetime.now()
        )
        
        with self._lock:
            self._request_metrics[correlation_id] = request_metrics
        
        return correlation_id
    
    def end_request_tracking(self, 
                           correlation_id: str, 
                           status_code: int = 200, 
                           error: Optional[str] = None,
                           custom_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        End tracking metrics for a request.
        
        Args:
            correlation_id: Request correlation ID
            status_code: HTTP status code
            error: Error message if any
            custom_metrics: Additional custom metrics
        """
        with self._lock:
            if correlation_id not in self._request_metrics:
                logger.warning(f"Request metrics not found for correlation_id: {correlation_id}")
                return
            
            request_metrics = self._request_metrics[correlation_id]
            request_metrics.end_time = datetime.now()
            request_metrics.duration_ms = (request_metrics.end_time - request_metrics.start_time).total_seconds() * 1000
            request_metrics.status_code = status_code
            request_metrics.error = error
            request_metrics.custom_metrics = custom_metrics or {}
            
            # Record system metrics if enabled
            if self._system_metrics_enabled:
                request_metrics.cpu_usage_percent = psutil.cpu_percent()
                process = psutil.Process()
                request_metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            
            # Record aggregated metrics
            tags = {
                "endpoint": request_metrics.endpoint,
                "method": request_metrics.method,
                "status_code": str(status_code)
            }
            
            self.record_histogram("request_duration_ms", request_metrics.duration_ms, tags)
            self.increment_counter("requests_total", 1, tags)
            
            if error:
                self.increment_counter("requests_errors_total", 1, tags)
            
            # Log request completion
            perf_logger.info(
                f"Request completed: {request_metrics.method} {request_metrics.endpoint}",
                extra=asdict(request_metrics)
            )
            
            # Clean up
            del self._request_metrics[correlation_id]
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """
        Collect current system metrics.
        
        Returns:
            Dictionary of system metrics
        """
        if not self._system_metrics_enabled:
            return {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / 1024 / 1024 / 1024
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / 1024 / 1024 / 1024
            
            # Process metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / 1024 / 1024
            process_cpu_percent = process.cpu_percent()
            
            metrics = {
                "system_cpu_percent": cpu_percent,
                "system_cpu_count": cpu_count,
                "system_memory_percent": memory_percent,
                "system_memory_available_gb": memory_available_gb,
                "system_disk_percent": disk_percent,
                "system_disk_free_gb": disk_free_gb,
                "process_memory_mb": process_memory_mb,
                "process_cpu_percent": process_cpu_percent
            }
            
            # Record metrics
            for name, value in metrics.items():
                self.set_gauge(name, value)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get a summary of metrics from the specified time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary containing metrics summary
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            # Filter metrics by time window
            recent_metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
            
            # Group metrics by name
            metrics_by_name = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_name[metric.name].append(metric.value)
            
            # Calculate summary statistics
            summary = {}
            for name, values in metrics_by_name.items():
                if values:
                    summary[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "sum": sum(values)
                    }
            
            # Add current counters and gauges
            summary["counters"] = dict(self._counters)
            summary["gauges"] = dict(self._gauges)
            
            return summary
    
    def export_metrics(self, file_path: Optional[str] = None) -> str:
        """
        Export metrics to a JSON file.
        
        Args:
            file_path: Path to export file. If None, uses default location
            
        Returns:
            Path to the exported file
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"logs/metrics_export_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics_summary": self.get_metrics_summary(60),  # Last hour
            "system_metrics": self.collect_system_metrics()
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to: {file_path}")
        return file_path


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Convenience functions
def record_metric(name: str, value: float, unit: str = "count", tags: Optional[Dict[str, str]] = None) -> None:
    """Record a metric using the global collector."""
    metrics_collector.record_metric(name, value, unit, tags)


def increment_counter(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter using the global collector."""
    metrics_collector.increment_counter(name, value, tags)


def set_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Set a gauge using the global collector."""
    metrics_collector.set_gauge(name, value, tags)


def time_operation(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Time an operation using the global collector."""
    return metrics_collector.time_operation(operation_name, tags)


def start_request_tracking(endpoint: str, method: str = "POST") -> str:
    """Start request tracking using the global collector."""
    return metrics_collector.start_request_tracking(endpoint, method)


def end_request_tracking(correlation_id: str, status_code: int = 200, error: Optional[str] = None, custom_metrics: Optional[Dict[str, float]] = None) -> None:
    """End request tracking using the global collector."""
    metrics_collector.end_request_tracking(correlation_id, status_code, error, custom_metrics)