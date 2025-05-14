"""
Distributed Caching Service for Forest App

This module implements a distributed caching system that can scale horizontally
while maintaining the intimate, personal experience for each user. It supports
both local memory caching and Redis-based distributed caching.
"""

import asyncio
import logging
import json
import hashlib
import time
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TypeVar, Generic, Callable
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)

# Type variable for generic cache methods
T = TypeVar('T')

class CacheBackend(Enum):
    """Supported cache backend types."""
    MEMORY = "memory"  # Local in-memory cache
    REDIS = "redis"    # Redis distributed cache
    NONE = "none"      # No caching (for testing/debugging)

class CacheConfig:
    """Configuration for the cache service."""
    
    def __init__(
        self,
        backend: CacheBackend = CacheBackend.MEMORY,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,  # 1 hour
        namespace: str = "forest:",
        serializer: Optional[Callable] = None,
        deserializer: Optional[Callable] = None
    ):
        """
        Initialize cache configuration.
        
        Args:
            backend: The cache backend to use
            redis_url: Redis connection URL (required for REDIS backend)
            default_ttl: Default time-to-live for cache entries in seconds
            namespace: Prefix for all cache keys
            serializer: Custom serializer function (default: pickle)
            deserializer: Custom deserializer function (default: pickle)
        """
        self.backend = backend
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.namespace = namespace
        self.serializer = serializer or pickle.dumps
        self.deserializer = deserializer or pickle.loads

class MemoryCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self, config: CacheConfig):
        """
        Initialize memory cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache: Dict[str, Tuple[Any, float]] = {}  # (value, expiry)
        self.namespace = config.namespace
        self.lock = asyncio.Lock()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())
        
        logger.info("Memory cache initialized")
    
    async def _cleanup_task(self):
        """Background task to clean up expired cache entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self.cleanup()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    async def cleanup(self):
        """Remove expired entries from cache."""
        now = time.time()
        expired_keys = []
        
        async with self.lock:
            # Find expired keys
            for key, (_, expiry) in self.cache.items():
                if expiry < now:
                    expired_keys.append(key)
            
            # Remove expired keys
            for key in expired_keys:
                del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired cache entries")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        full_key = f"{self.namespace}{key}"
        now = time.time()
        
        async with self.lock:
            if full_key in self.cache:
                value, expiry = self.cache[full_key]
                
                # Check if expired
                if expiry < now:
                    del self.cache[full_key]
                    return None
                
                # Return cached value
                try:
                    return self.config.deserializer(value)
                except Exception as e:
                    logger.error(f"Error deserializing cached value: {e}")
                    return None
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        full_key = f"{self.namespace}{key}"
        ttl = ttl if ttl is not None else self.config.default_ttl
        expiry = time.time() + ttl
        
        try:
            # Serialize value
            serialized_value = self.config.serializer(value)
            
            # Store in cache
            async with self.lock:
                self.cache[full_key] = (serialized_value, expiry)
            
            return True
        except Exception as e:
            logger.error(f"Error setting cache value: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        full_key = f"{self.namespace}{key}"
        
        async with self.lock:
            if full_key in self.cache:
                del self.cache[full_key]
                return True
        
        return False
    
    async def flush(self) -> bool:
        """
        Clear the entire cache.
        
        Returns:
            True if successful
        """
        async with self.lock:
            self.cache.clear()
        
        logger.info("Memory cache flushed")
        return True

class RedisCache:
    """Redis-based distributed cache implementation."""
    
    def __init__(self, config: CacheConfig):
        """
        Initialize Redis cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.namespace = config.namespace
        self.redis = None
        self.lock = asyncio.Lock()
        
        # Import Redis here to avoid dependency if not used
        try:
            import redis.asyncio as aioredis
            self.redis = aioredis.from_url(config.redis_url)
            logger.info("Redis cache initialized with URL: " + config.redis_url.split("@")[-1])  # Hide credentials
        except ImportError:
            logger.error("Redis package not installed. Please install 'redis' package.")
            raise
        except Exception as e:
            logger.error(f"Error initializing Redis connection: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.redis:
            return None
        
        full_key = f"{self.namespace}{key}"
        
        try:
            # Get value from Redis
            value = await self.redis.get(full_key)
            
            if value is None:
                return None
            
            # Deserialize value
            return self.config.deserializer(value)
        except Exception as e:
            logger.error(f"Error getting value from Redis: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False
        
        full_key = f"{self.namespace}{key}"
        ttl = ttl if ttl is not None else self.config.default_ttl
        
        try:
            # Serialize value
            serialized_value = self.config.serializer(value)
            
            # Store in Redis
            await self.redis.set(full_key, serialized_value, ex=ttl)
            
            return True
        except Exception as e:
            logger.error(f"Error setting value in Redis: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if not self.redis:
            return False
        
        full_key = f"{self.namespace}{key}"
        
        try:
            # Delete from Redis
            result = await self.redis.delete(full_key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting value from Redis: {e}")
            return False
    
    async def flush(self) -> bool:
        """
        Clear all cache entries with this namespace.
        
        Returns:
            True if successful
        """
        if not self.redis:
            return False
        
        try:
            # Find all keys with this namespace
            pattern = f"{self.namespace}*"
            keys = []
            
            # Scan for keys in batches to avoid blocking Redis
            cursor = 0
            while True:
                cursor, batch = await self.redis.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                
                if cursor == 0:
                    break
            
            # Delete keys if found
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Flushed {len(keys)} keys from Redis cache")
            
            return True
        except Exception as e:
            logger.error(f"Error flushing Redis cache: {e}")
            return False

class CacheService:
    """
    Distributed caching service for improving performance and scalability.
    
    This service provides a unified interface for caching data, whether using
    local memory or Redis, making it easy to scale horizontally while maintaining
    the intimate, personal experience for each user.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, config: Optional[CacheConfig] = None) -> 'CacheService':
        """Get the singleton instance of the CacheService."""
        if cls._instance is None:
            cls._instance = CacheService(config or CacheConfig())
        elif config is not None:
            logger.warning("Cache already initialized, ignoring new config")
        return cls._instance
    
    def __init__(self, config: CacheConfig):
        """
        Initialize the cache service.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        
        # Initialize backend
        if config.backend == CacheBackend.MEMORY:
            self.backend = MemoryCache(config)
        elif config.backend == CacheBackend.REDIS:
            if not config.redis_url:
                raise ValueError("Redis URL is required for Redis backend")
            self.backend = RedisCache(config)
        elif config.backend == CacheBackend.NONE:
            self.backend = None
            logger.warning("Cache disabled (NONE backend)")
        else:
            raise ValueError(f"Unsupported cache backend: {config.backend}")
        
        logger.info(f"Cache service initialized with {config.backend.value} backend")
    
    async def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.backend:
            return None
        
        value = await self.backend.get(key)
        logger.debug(f"Cache {'hit' if value is not None else 'miss'} for key: {key}")
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.backend:
            return False
        
        success = await self.backend.set(key, value, ttl)
        if success:
            logger.debug(f"Cached value for key: {key} (TTL: {ttl or self.config.default_ttl}s)")
        return success
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if not self.backend:
            return False
        
        success = await self.backend.delete(key)
        if success:
            logger.debug(f"Deleted cached value for key: {key}")
        return success
    
    async def flush(self) -> bool:
        """
        Clear the entire cache.
        
        Returns:
            True if successful
        """
        if not self.backend:
            return False
        
        return await self.backend.flush()

# Decorator for cacheable functions
def cacheable(key_pattern: str, ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        key_pattern: Pattern for cache key, using {arg_name} for arg values
                    For positional args, use {0}, {1}, etc.
        ttl: Time-to-live in seconds (uses default if None)
        
    Returns:
        Decorated function with caching
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache service
            cache = CacheService.get_instance()
            
            # Skip if cache is disabled
            if not cache.backend:
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Build cache key from pattern
            key_context = kwargs.copy()
            # Add positional args to context
            for i, arg in enumerate(args):
                # Skip self/cls for methods
                if i == 0 and func.__name__ == func.__qualname__.split('.')[-1]:
                    continue
                key_context[str(i)] = arg
            
            try:
                # Format key pattern with args
                cache_key = key_pattern.format(**key_context)
                
                # Hash long or complex keys
                if len(cache_key) > 100:
                    cache_key = hashlib.md5(cache_key.encode()).hexdigest()
                
                # Add function name prefix
                cache_key = f"{func.__module__}.{func.__name__}:{cache_key}"
                
                # Check cache first
                cached_value = await cache.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Call function if cache miss
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await cache.set(cache_key, result, ttl)
                
                return result
                
            except Exception as e:
                logger.warning(f"Error in cache logic: {e}, falling back to uncached function")
                # Fall back to uncached function call
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        
        return wrapper
    
    return decorator
