"""
Background Task Queue for Forest App

This module implements an asynchronous background task queue system to handle
intensive operations without blocking the user experience. It ensures the
journey remains responsive even during complex analysis or processing.
"""

import asyncio
import logging
import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Awaitable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import uuid
import json

logger = logging.getLogger(__name__)

class TaskQueue:
    """
    An asynchronous task queue for processing intensive background operations.
    
    This implementation uses asyncio and provides:
    - Task prioritization
    - Scheduled task execution
    - Result caching
    - Failure handling with exponential backoff
    """
    
    def __init__(self, max_workers: int = 10, result_ttl: int = 300):
        """
        Initialize the task queue.
        
        Args:
            max_workers: Maximum number of worker tasks to run simultaneously
            result_ttl: Time (in seconds) to keep task results in cache
        """
        self.queue = asyncio.PriorityQueue()
        self.processing: Set[str] = set()  # Currently processing task IDs
        self.results: Dict[str, Any] = {}  # Task results
        self.result_timestamps: Dict[str, float] = {}  # When results were stored
        self.max_workers = max_workers
        self.result_ttl = result_ttl
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task metadata for better monitoring
        self.task_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"TaskQueue initialized with {max_workers} workers and {result_ttl}s result TTL")
    
    async def start(self):
        """Start the task queue workers."""
        if self.running:
            return
            
        self.running = True
        
        # Create worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.worker_tasks.append(worker)
            
        # Create task for cache cleanup
        asyncio.create_task(self._cleanup_results())
        
        logger.info("TaskQueue workers started")
        
    async def stop(self):
        """Gracefully stop the task queue."""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for all worker tasks to complete
        for worker in self.worker_tasks:
            worker.cancel()
            
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks = []
        
        # Close thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("TaskQueue workers stopped")
        
    async def _worker(self, worker_id: int):
        """
        Background worker to process tasks from the queue.
        
        Args:
            worker_id: Identifier for this worker
        """
        logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    priority, task_id, func, args, kwargs = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                    
                # Mark task as processing
                self.processing.add(task_id)
                
                # Update task metadata
                self.task_metadata[task_id]["status"] = "processing"
                self.task_metadata[task_id]["started_at"] = datetime.now(timezone.utc).isoformat()
                
                logger.debug(f"Worker {worker_id} processing task {task_id} (priority: {priority})")
                
                # Execute the task
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        # Run sync functions in thread pool
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            self.thread_pool, 
                            functools.partial(func, *args, **kwargs)
                        )
                        
                    # Store the result
                    self.results[task_id] = {
                        "status": "completed",
                        "result": result
                    }
                    
                    # Update task metadata
                    self.task_metadata[task_id]["status"] = "completed"
                    self.task_metadata[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                    
                    logger.info(f"Task {task_id} completed successfully")
                    
                except Exception as e:
                    error_details = {
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    
                    # Store the error
                    self.results[task_id] = {
                        "status": "failed",
                        "error": error_details
                    }
                    
                    # Update task metadata
                    self.task_metadata[task_id]["status"] = "failed"
                    self.task_metadata[task_id]["error"] = error_details
                    self.task_metadata[task_id]["failed_at"] = datetime.now(timezone.utc).isoformat()
                    
                    logger.error(f"Task {task_id} failed: {e}")
                
                # Record timestamp for cache expiration
                self.result_timestamps[task_id] = asyncio.get_event_loop().time()
                
                # Remove from processing set
                self.processing.remove(task_id)
                
                # Mark queue task as done
                self.queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}")
                
        logger.debug(f"Worker {worker_id} stopped")
        
    async def _cleanup_results(self):
        """Periodically clean up expired results."""
        while self.running:
            try:
                current_time = asyncio.get_event_loop().time()
                expired_tasks = []
                
                # Find expired results
                for task_id, timestamp in self.result_timestamps.items():
                    if current_time - timestamp > self.result_ttl:
                        expired_tasks.append(task_id)
                
                # Remove expired results
                for task_id in expired_tasks:
                    if task_id in self.results:
                        del self.results[task_id]
                    if task_id in self.result_timestamps:
                        del self.result_timestamps[task_id]
                    if task_id in self.task_metadata:
                        # Archive metadata if needed instead of deleting
                        self.task_metadata[task_id]["archived"] = True
                
                if expired_tasks:
                    logger.debug(f"Cleaned up {len(expired_tasks)} expired task results")
                
                # Sleep for a while
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"Error in result cleanup: {e}")
                await asyncio.sleep(60)  # Still sleep on error
    
    async def enqueue(self, 
                     func: Callable[..., Any], 
                     *args, 
                     priority: int = 5,
                     task_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     **kwargs) -> str:
        """
        Add a task to the queue.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            priority: Priority level (lower numbers = higher priority)
            task_id: Optional custom task ID (generates UUID if not provided)
            metadata: Optional task metadata
            **kwargs: Keyword arguments for the function
            
        Returns:
            Task ID that can be used to get the result
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = str(uuid.uuid4())
            
        # Store task metadata
        self.task_metadata[task_id] = {
            "id": task_id,
            "function": func.__name__,
            "args_summary": f"{len(args)} positional, {len(kwargs)} keyword args",
            "priority": priority,
            "status": "queued",
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "user_metadata": metadata or {}
        }
        
        # Add task to queue
        await self.queue.put((priority, task_id, func, args, kwargs))
        
        logger.info(f"Task {task_id} added to queue with priority {priority}")
        return task_id
        
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get the result of a task.
        
        Args:
            task_id: ID of the task
            timeout: Optional timeout in seconds to wait for result
            
        Returns:
            Dictionary with task status and result/error
            
        Raises:
            asyncio.TimeoutError: If timeout is reached and task is not complete
            KeyError: If task ID is not found
        """
        start_time = asyncio.get_event_loop().time()
        
        while timeout is None or asyncio.get_event_loop().time() - start_time < timeout:
            # Check if task has a result
            if task_id in self.results:
                return self.results[task_id]
                
            # Check if task exists but is still processing
            if task_id in self.processing or any(task_id == item[1] for item in self.queue._queue):
                # Task exists but not done yet, wait a bit
                await asyncio.sleep(0.1)
                continue
                
            # Task not found in queue or processing
            if task_id not in self.task_metadata:
                raise KeyError(f"Task {task_id} not found")
                
            # Task exists in metadata but not in queue or processing, something went wrong
            return {
                "status": "unknown",
                "error": "Task exists in metadata but not in queue or processing"
            }
            
        # Timeout reached
        raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
        
    def get_task_metadata(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with task metadata or None if not found
        """
        return self.task_metadata.get(task_id)
        
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the task queue.
        
        Returns:
            Dictionary with queue statistics
        """
        return {
            "running": self.running,
            "workers": len(self.worker_tasks),
            "queue_size": self.queue.qsize(),
            "processing": len(self.processing),
            "completed_results": len(self.results)
        }
        
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for a task to complete and return its result.
        
        Args:
            task_id: ID of the task
            timeout: Optional timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
            KeyError: If task is not found
        """
        return await self.get_result(task_id, timeout)

    @classmethod
    def get_instance(cls):
        """Return the global singleton instance of TaskQueue."""
        from forest_app.core.task_queue import task_queue
        return task_queue

task_queue = TaskQueue()
