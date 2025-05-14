"""
Structured logging middleware for FastAPI
"""

import logging
import time
from typing import Callable, Dict, Any
from uuid import UUID

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pythonjsonlogger import jsonlogger


# Configure JSON logger
logger = logging.getLogger("api")
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(trace_id)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z"
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured logging of HTTP requests and responses.
    
    Includes trace_id from RequestContext in all logs for request tracing.
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        start_time = time.time()
        trace_id = None
        
        # Extract trace_id from headers if present
        if "x-trace-id" in request.headers:
            try:
                trace_id = UUID(request.headers["x-trace-id"])
            except (ValueError, TypeError):
                pass
        
        # If no trace_id in header, try to get from RequestContext later
        
        # Process the request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            status_code = response.status_code
            
            # Log the request details with structured logging
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "processing_time_ms": round(process_time * 1000, 2),
                "trace_id": str(trace_id) if trace_id else None
            }
            
            if 200 <= status_code < 400:
                logger.info("Request processed", extra=log_data)
            elif 400 <= status_code < 500:
                logger.warning("Request resulted in client error", extra=log_data)
            else:
                logger.error("Request resulted in server error", extra=log_data)
                
            # Add X-Process-Time header to response
            response.headers["X-Process-Time"] = str(process_time)
            
            # If we have a trace_id, add it to the response headers
            if trace_id:
                response.headers["X-Trace-ID"] = str(trace_id)
                
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "processing_time_ms": round(process_time * 1000, 2),
                "trace_id": str(trace_id) if trace_id else None
            }
            logger.error("Request failed with exception", extra=log_data)
            raise
