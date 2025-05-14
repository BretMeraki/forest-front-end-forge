"""
Circuit Breaker Pattern for Forest App

This module implements the circuit breaker pattern to ensure robust
interactions with external services. It provides graceful degradation
and fallback mechanisms to maintain the sanctuary experience when
external dependencies are unstable.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, Union, TypeVar, Generic, List, Awaitable
from datetime import datetime, timedelta
from enum import Enum
import functools
import pybreaker
import random

logger = logging.getLogger(__name__)

# Type variables for function signatures
T = TypeVar('T')
R = TypeVar('R')

class CircuitState(Enum):
    """States for the circuit breaker."""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"      # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back online

class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exceptions: List[type] = None,
        fallback_function: Optional[Callable] = None,
        name: str = "default",
        timeout: Optional[float] = None
    ):
        """
        Initialize circuit breaker configuration.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open state
            expected_exceptions: List of exception types that trigger the breaker
            fallback_function: Function to call when circuit is open
            name: Name for this circuit breaker (for logging)
            timeout: Optional timeout for the protected function
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions or [Exception]
        self.fallback_function = fallback_function
        self.name = name
        self.timeout = timeout

class CircuitBreaker:
    """
    Circuit breaker implementation for protecting external service calls.
    
    This implementation supports both synchronous and asynchronous functions,
    with configurable failure thresholds, timeouts, and fallback mechanisms.
    """
    
    _instances: Dict[str, 'CircuitBreaker'] = {}
    
    @classmethod
    def get_instance(cls, name: str = "default") -> 'CircuitBreaker':
        """Get a named circuit breaker instance (singleton pattern)."""
        if name not in cls._instances:
            cls._instances[name] = CircuitBreaker(config=CircuitBreakerConfig(name=name))
        return cls._instances[name]
    
    def __init__(self, config: CircuitBreakerConfig):
        """
        Initialize a new circuit breaker.
        
        Args:
            config: Configuration for this circuit breaker
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        # Create pybreaker instance
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=config.failure_threshold,
            reset_timeout=config.recovery_timeout,
            exclude=config.expected_exceptions,
            name=config.name
        )
        
        # Set up pybreaker-compatible listener
        class _PyBreakerListener(pybreaker.CircuitBreakerListener):
            def __init__(listener_self, parent):
                listener_self.parent = parent
            def state_change(listener_self, cb, old_state, new_state):
                if new_state == pybreaker.STATE_OPEN:
                    parent._on_open(cb)
                elif new_state == pybreaker.STATE_CLOSED:
                    parent._on_close(cb)
                elif new_state == pybreaker.STATE_HALF_OPEN:
                    parent._on_half_open(cb)
        parent = self
        self._breaker.add_listener(_PyBreakerListener(parent))
        
        logger.info(f"Circuit breaker '{config.name}' initialized with failure threshold "
                   f"{config.failure_threshold}, recovery timeout {config.recovery_timeout}s")
    
    def _on_open(self, breaker):
        """Called when circuit transitions to open state."""
        self.state = CircuitState.OPEN
        logger.warning(f"Circuit '{self.config.name}' OPENED after {self.failure_count} failures")
    
    def _on_close(self, breaker):
        """Called when circuit transitions to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_success_time = datetime.now()
        logger.info(f"Circuit '{self.config.name}' CLOSED, service appears healthy")
    
    def _on_half_open(self, breaker):
        """Called when circuit transitions to half-open state."""
        self.state = CircuitState.HALF_OPEN
        logger.info(f"Circuit '{self.config.name}' HALF-OPEN, testing service health")
    
    def _record_success(self):
        """Record a successful call."""
        self.last_success_time = datetime.now()
        # Let pybreaker handle state transitions
    
    def _record_failure(self, exception):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        # Let pybreaker handle state transitions
        logger.debug(f"Circuit '{self.config.name}' recorded failure ({self.failure_count}/{self.config.failure_threshold}): {exception}")
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call a synchronous function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            pybreaker.CircuitBreakerError: If circuit is open
            Exception: If function raises an exception and no fallback is available
        """
        try:
            # Use pybreaker to manage the circuit state
            result = self._breaker.call(func, *args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            
            # Try fallback if available
            if self.config.fallback_function:
                logger.info(f"Using fallback for circuit '{self.config.name}'")
                return self.config.fallback_function(*args, **kwargs)
            
            # Re-raise the exception
            raise
    
    async def call_async(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Call an asynchronous function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the async function
            
        Raises:
            pybreaker.CircuitBreakerError: If circuit is open
            Exception: If function raises an exception and no fallback is available
        """
        try:
            # Add timeout if configured
            if self.config.timeout:
                # Wrap function with timeout
                async def func_with_timeout():
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
                
                # Use pybreaker to manage the circuit state for the timeout-wrapped function
                result = await self._breaker.call(func_with_timeout)
            else:
                # Use pybreaker directly
                result = await self._breaker.call(func, *args, **kwargs)
                
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure(e)
            
            # Try fallback if available
            if self.config.fallback_function:
                if asyncio.iscoroutinefunction(self.config.fallback_function):
                    logger.info(f"Using async fallback for circuit '{self.config.name}'")
                    return await self.config.fallback_function(*args, **kwargs)
                else:
                    logger.info(f"Using sync fallback for circuit '{self.config.name}'")
                    return self.config.fallback_function(*args, **kwargs)
            
            # Re-raise the exception
            raise

def circuit_protected(
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    expected_exceptions: List[type] = None,
    fallback_function: Optional[Callable] = None,
    name: Optional[str] = None,
    timeout: Optional[float] = None
):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying half-open state
        expected_exceptions: List of exception types that trigger the breaker
        fallback_function: Function to call when circuit is open
        name: Optional name for this circuit breaker
        timeout: Optional timeout for the function
        
    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func):
        # Generate circuit name from function if not provided
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        
        # Create config
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exceptions=expected_exceptions,
            fallback_function=fallback_function,
            name=circuit_name,
            timeout=timeout
        )
        
        # Create or get circuit breaker
        breaker = CircuitBreaker(config)
        
        # Determine if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)
        
        return wrapper
    
    return decorator
