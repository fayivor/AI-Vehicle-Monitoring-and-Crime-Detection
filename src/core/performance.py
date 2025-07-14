"""Performance optimization and monitoring for the RAG pipeline."""

import time
import asyncio
import psutil
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import threading
from functools import wraps

from ..utils.logging import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    active_connections: int
    query_rate: float
    avg_response_time: float
    error_rate: float
    cache_hit_rate: float


class PerformanceMonitor:
    """
    Performance monitoring system for real-time metrics collection.
    
    Tracks:
    - Response times (target: < 2 seconds for full RAG pipeline)
    - Vector search times (target: < 100ms)
    - Context update times (target: < 500ms)
    - System resource usage
    - Error rates and availability
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.snapshots: deque = deque(maxlen=window_size)
        self.start_time = datetime.utcnow()
        self.lock = threading.Lock()
        
        # Initialize metric queues
        self._init_metric_queues()
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _init_metric_queues(self) -> None:
        """Initialize metric queues."""
        metric_names = [
            'response_time',
            'vector_search_time',
            'context_update_time',
            'llm_processing_time',
            'query_count',
            'error_count',
            'cache_hits',
            'cache_misses'
        ]
        
        for name in metric_names:
            self.metrics[name] = deque(maxlen=self.window_size)
    
    def _start_background_monitoring(self) -> None:
        """Start background system monitoring."""
        def monitor_system():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    logger.error("System monitoring failed", error=str(e))
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Calculate derived metrics
            query_rate = self._calculate_query_rate()
            avg_response_time = self._calculate_avg_response_time()
            error_rate = self._calculate_error_rate()
            cache_hit_rate = self._calculate_cache_hit_rate()
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                active_connections=0,  # TODO: Track actual connections
                query_rate=query_rate,
                avg_response_time=avg_response_time,
                error_rate=error_rate,
                cache_hit_rate=cache_hit_rate
            )
            
            with self.lock:
                self.snapshots.append(snapshot)
                
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    def record_metric(self, name: str, value: float, metadata: Dict[str, Any] = None) -> None:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=self._get_metric_unit(name),
            metadata=metadata or {}
        )
        
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.window_size)
            self.metrics[name].append(metric)
    
    def _get_metric_unit(self, name: str) -> str:
        """Get unit for metric name."""
        time_metrics = ['response_time', 'vector_search_time', 'context_update_time', 'llm_processing_time']
        if name in time_metrics:
            return 'ms'
        elif 'rate' in name:
            return 'per_second'
        elif 'percent' in name:
            return 'percent'
        else:
            return 'count'
    
    def _calculate_query_rate(self) -> float:
        """Calculate queries per second over last minute."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=1)
        
        with self.lock:
            if 'query_count' not in self.metrics:
                return 0.0
            
            recent_queries = [
                m for m in self.metrics['query_count']
                if m.timestamp >= cutoff_time
            ]
            
            return len(recent_queries) / 60.0  # Per second
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time over last 100 requests."""
        with self.lock:
            if 'response_time' not in self.metrics or not self.metrics['response_time']:
                return 0.0
            
            recent_times = list(self.metrics['response_time'])[-100:]
            return sum(m.value for m in recent_times) / len(recent_times)
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate over last 100 requests."""
        with self.lock:
            query_count = len(self.metrics.get('query_count', []))
            error_count = len(self.metrics.get('error_count', []))
            
            if query_count == 0:
                return 0.0
            
            return error_count / query_count
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        with self.lock:
            hits = len(self.metrics.get('cache_hits', []))
            misses = len(self.metrics.get('cache_misses', []))
            total = hits + misses
            
            if total == 0:
                return 0.0
            
            return hits / total
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            latest_snapshot = self.snapshots[-1] if self.snapshots else None
            
            metrics = {
                'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
                'query_rate': self._calculate_query_rate(),
                'avg_response_time': self._calculate_avg_response_time(),
                'error_rate': self._calculate_error_rate(),
                'cache_hit_rate': self._calculate_cache_hit_rate(),
            }
            
            if latest_snapshot:
                metrics.update({
                    'cpu_percent': latest_snapshot.cpu_percent,
                    'memory_percent': latest_snapshot.memory_percent,
                    'memory_used_mb': latest_snapshot.memory_used_mb,
                })
            
            return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_metrics = self.get_current_metrics()
        
        # Check against targets from guidelines
        targets = {
            'vector_search_time': 100,      # < 100ms
            'response_time': 2000,          # < 2 seconds
            'context_update_time': 500,     # < 500ms
            'availability': 99.5,           # 99.5% uptime
        }
        
        # Calculate target compliance
        compliance = {}
        with self.lock:
            if 'vector_search_time' in self.metrics and self.metrics['vector_search_time']:
                avg_vector_time = sum(m.value for m in self.metrics['vector_search_time']) / len(self.metrics['vector_search_time'])
                compliance['vector_search'] = avg_vector_time <= targets['vector_search_time']
            
            if 'response_time' in self.metrics and self.metrics['response_time']:
                avg_response_time = sum(m.value for m in self.metrics['response_time']) / len(self.metrics['response_time'])
                compliance['response_time'] = avg_response_time <= targets['response_time']
            
            if 'context_update_time' in self.metrics and self.metrics['context_update_time']:
                avg_context_time = sum(m.value for m in self.metrics['context_update_time']) / len(self.metrics['context_update_time'])
                compliance['context_update'] = avg_context_time <= targets['context_update_time']
        
        # Calculate availability
        error_rate = current_metrics.get('error_rate', 0.0)
        availability = (1.0 - error_rate) * 100
        compliance['availability'] = availability >= targets['availability']
        
        return {
            'current_metrics': current_metrics,
            'target_compliance': compliance,
            'targets': targets,
            'overall_health': all(compliance.values()) if compliance else False,
            'timestamp': datetime.utcnow().isoformat()
        }


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    @staticmethod
    def async_timeout(timeout_seconds: float):
        """Decorator for async function timeout."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Function timeout",
                        function=func.__name__,
                        timeout=timeout_seconds
                    )
                    raise
            return wrapper
        return decorator
    
    @staticmethod
    def measure_time(metric_name: str, monitor: PerformanceMonitor):
        """Decorator to measure function execution time."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000  # Convert to ms
                    monitor.record_metric(metric_name, execution_time)
                    return result
                except Exception as e:
                    monitor.record_metric('error_count', 1)
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000  # Convert to ms
                    monitor.record_metric(metric_name, execution_time)
                    return result
                except Exception as e:
                    monitor.record_metric('error_count', 1)
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    @staticmethod
    def connection_pool_optimizer(max_connections: int = 100):
        """Connection pool optimization decorator."""
        def decorator(func: Callable):
            # TODO: Implement connection pooling logic
            return func
        return decorator
    
    @staticmethod
    def cache_result(ttl_seconds: int = 300):
        """Simple result caching decorator."""
        cache = {}
        
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # Check cache
                if cache_key in cache:
                    result, timestamp = cache[cache_key]
                    if (datetime.utcnow() - timestamp).total_seconds() < ttl_seconds:
                        return result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                cache[cache_key] = (result, datetime.utcnow())
                
                # Clean old entries
                current_time = datetime.utcnow()
                expired_keys = [
                    key for key, (_, timestamp) in cache.items()
                    if (current_time - timestamp).total_seconds() >= ttl_seconds
                ]
                for key in expired_keys:
                    del cache[key]
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Similar logic for sync functions
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                
                if cache_key in cache:
                    result, timestamp = cache[cache_key]
                    if (datetime.utcnow() - timestamp).total_seconds() < ttl_seconds:
                        return result
                
                result = func(*args, **kwargs)
                cache[cache_key] = (result, datetime.utcnow())
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator


class GracefulDegradation:
    """Graceful degradation for high load scenarios."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.degradation_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time': 5000.0,  # 5 seconds
            'error_rate': 0.1  # 10%
        }
        self.degraded_mode = False
    
    def check_degradation_needed(self) -> bool:
        """Check if system should enter degraded mode."""
        metrics = self.monitor.get_current_metrics()
        
        for metric, threshold in self.degradation_thresholds.items():
            if metrics.get(metric, 0) > threshold:
                logger.warning(
                    "Degradation threshold exceeded",
                    metric=metric,
                    value=metrics.get(metric),
                    threshold=threshold
                )
                return True
        
        return False
    
    def enter_degraded_mode(self) -> None:
        """Enter degraded performance mode."""
        if not self.degraded_mode:
            self.degraded_mode = True
            logger.warning("Entering degraded performance mode")
    
    def exit_degraded_mode(self) -> None:
        """Exit degraded performance mode."""
        if self.degraded_mode:
            self.degraded_mode = False
            logger.info("Exiting degraded performance mode")
    
    def get_degraded_config(self) -> Dict[str, Any]:
        """Get configuration for degraded mode."""
        return {
            'max_results': 5,           # Reduce from 10
            'vector_search_timeout': 50,  # Reduce from 100ms
            'llm_max_tokens': 1024,     # Reduce from 2048
            'context_window': 2048,     # Reduce from 4096
            'disable_mcp': True,        # Disable MCP processing
            'cache_ttl': 600           # Increase cache TTL
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
graceful_degradation = GracefulDegradation(performance_monitor)
