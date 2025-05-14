"""
LLM Service Abstraction Layer for Forest OS.

This module implements the [MCP-LLM Vision - Arch] PRD requirement by providing
a clean abstraction layer for different LLM providers. The main component is
the BaseLLMService abstract base class, which defines a standard interface for
interacting with LLMs regardless of the specific provider.

For the MVP, Google Gemini is used as the default LLM provider via the
GoogleGeminiService concrete implementation.
"""

from abc import ABC, abstractmethod
import asyncio
import backoff
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Type, TypeVar, Union, List, Generic, Callable, Awaitable

import aiohttp
from pydantic import BaseModel, Field

# Import auxiliary services
try:
    from forest_app.integrations.context_trimmer import ContextTrimmer
    from forest_app.integrations.prompt_augmentation import PromptAugmentationService
    aux_services_import_ok = True
except ImportError:
    logging.getLogger(__name__).warning(
        "Failed to import auxiliary services. Some features may be unavailable."
    )
    aux_services_import_ok = False

# Ensure we try to import Google Generative AI library
try:
    import google.generativeai as genai
    from google.generativeai.types import (
        ContentDict, GenerationConfig, GenerateContentResponse,
        HarmBlockThreshold, HarmCategory,
    )
    from google.generativeai import protos
    from google.api_core import exceptions as google_api_exceptions
    google_import_ok = True
except ImportError:
    logging.getLogger(__name__).critical(
        "Failed to import google.generativeai or related components. "
        "Install with: pip install google-generativeai"
    )
    google_import_ok = False
    # We'll define dummy classes for type checking

# Import configurations
try:
    from forest_app.config.settings import settings
    settings_import_ok = True
except ImportError:
    logging.getLogger(__name__).warning(
        "Failed to import settings. Using default configuration values."
    )
    settings_import_ok = False

# Set up logging
logger = logging.getLogger(__name__)

# Type for Pydantic model that can be used for response validation
T = TypeVar('T', bound=BaseModel)

# Detailed request logging model
class LLMRequestLog(BaseModel):
    """Log entry for an LLM request."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    service: str
    model: str
    operation: str
    prompt_length: int
    duration_ms: Optional[int] = None
    token_count: Optional[int] = None
    success: bool = False
    retry_count: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    response_length: Optional[int] = None
    
    def complete(self, response: str) -> None:
        """Mark the request as completed successfully."""
        self.success = True
        self.duration_ms = int((datetime.now() - self.timestamp).total_seconds() * 1000)
        if response:
            self.response_length = len(response)
    
    def record_error(self, error: Exception) -> None:
        """Record an error that occurred during the request."""
        self.success = False
        self.error_type = type(error).__name__
        self.error_message = str(error)

# Exception classes for the LLM service layer
class LLMServiceError(Exception):
    """Base exception class for LLM service errors."""
    pass

class LLMConfigError(LLMServiceError):
    """Error in LLM service configuration."""
    pass

class LLMRequestError(LLMServiceError):
    """Error making a request to the LLM service."""
    pass

class LLMResponseError(LLMServiceError):
    """Error processing a response from the LLM service."""
    pass

class LLMTimeoutError(LLMRequestError):
    """Request to the LLM service timed out."""
    pass

class LLMTokenLimitError(LLMRequestError):
    """Request exceeds token limit for the LLM service."""
    pass

class LLMRateLimitError(LLMRequestError):
    """Request was rate limited by the LLM service."""
    pass

class LLMAuthenticationError(LLMConfigError):
    """Authentication to the LLM service failed."""
    pass


class BaseLLMService(ABC, Generic[T]):
    """
    Abstract base class defining the interface for LLM services.
    
    This class provides a standard interface for interacting with different LLM 
    providers. Concrete subclasses must implement the abstract methods to interact 
    with specific LLM providers.
    
    Features:
    - Fully async operation for non-blocking API calls
    - Robust retry with exponential backoff for transient errors
    - Timeout controls to prevent hanging requests
    - Fallback service support for high availability
    - Token tracking and management
    - Comprehensive audit logging
    - Lightweight caching for identical, repeatable calls
    """
    
    def __init__(
        self,
        service_name: str,
        default_model: str,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        enable_logging: bool = True,
        context_trimmer: Optional['ContextTrimmer'] = None,
        prompt_augmentation: Optional['PromptAugmentationService'] = None
    ):
        """
        Initialize the BaseLLMService.
        
        Args:
            service_name: Name of the LLM service provider
            default_model: Default model to use
            max_retries: Maximum number of retries for failed requests
            timeout_seconds: Timeout in seconds for requests
            enable_logging: Whether to enable comprehensive request logging
            context_trimmer: Optional ContextTrimmer instance
            prompt_augmentation: Optional PromptAugmentationService instance
        """
        self.service_name = service_name
        self.default_model = default_model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.enable_logging = enable_logging
        
        # Initialize auxiliary services if not provided
        self.context_trimmer = context_trimmer
        if not self.context_trimmer and aux_services_import_ok:
            try:
                self.context_trimmer = ContextTrimmer()
                logger.info(f"Created default ContextTrimmer for {service_name}")
            except Exception as e:
                logger.warning(f"Failed to create default ContextTrimmer: {e}")
            
        self.prompt_augmentation = prompt_augmentation
        if not self.prompt_augmentation and aux_services_import_ok:
            try:
                self.prompt_augmentation = PromptAugmentationService()
                logger.info(f"Created default PromptAugmentationService for {service_name}")
            except Exception as e:
                logger.warning(f"Failed to create default PromptAugmentationService: {e}")
        
        # For tracking requests
        self.request_logs: List[LLMRequestLog] = []
        
        # Set up fallback chains
        self.fallback_services: List['BaseLLMService'] = []
        
        # Simple cache for identical, small, repeatable calls
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_enabled = True
        self._cache_max_size = 100  # Maximum number of items to cache
        
        logger.info(f"Initialized {service_name} LLM service with default model {default_model}")
    
    def add_fallback(self, service: 'BaseLLMService') -> None:
        """
        Add a fallback service to use if this service fails.
        
        Args:
            service: Another LLM service to use as fallback
        """
        self.fallback_services.append(service)
        logger.info(f"Added {service.service_name} as fallback for {self.service_name}")
    
    def _create_request_log(self, operation: str, model: str, prompt: str) -> LLMRequestLog:
        """Create a request log entry."""
        return LLMRequestLog(
            service=self.service_name,
            model=model,
            operation=operation,
            prompt_length=len(prompt)
        )
    
    def _record_metrics(self, log: LLMRequestLog) -> None:
        """Record metrics for the request log."""
        if self.enable_logging:
            self.request_logs.append(log)
            # Log summary to logger
            status = "success" if log.success else f"error: {log.error_type}"
            logger.info(
                f"LLM request {log.request_id[:8]} to {log.service}:{log.model} "
                f"completed in {log.duration_ms}ms ({status})"
            )
            
            # Future: Add metrics sending to a monitoring system
            # if settings_import_ok and hasattr(settings, "METRICS_ENABLED") and settings.METRICS_ENABLED:
            #     # Send metrics to monitoring system
            #     pass
    
    def _cache_key(self, operation: str, prompt: str, **kwargs) -> str:
        """Generate a cache key for a request."""
        # Only cache if prompt is relatively small
        if not self._cache_enabled or len(prompt) > 500:
            return None
            
        # Create a deterministic cache key from the operation, prompt, and relevant kwargs
        key_parts = [operation, prompt]
        
        # Add relevant parameters to the cache key
        for k, v in sorted(kwargs.items()):
            if k in ("temperature", "max_tokens"):
                key_parts.append(f"{k}={v}")
        
        return "|".join(key_parts)
    
    async def _with_retry_and_fallback(
        self,
        operation: str,
        model: str,
        func: Callable[[], Awaitable[Any]],
        prompt: str,
        cache_key: Optional[str] = None,
        log: Optional[LLMRequestLog] = None
    ) -> Any:
        """
        Execute an LLM operation with retry, timeout, fallback, and caching.
        
        Args:
            operation: Name of the operation (for logging)
            model: Name of the model being used
            func: Async function to execute
            prompt: The prompt being sent to the LLM
            cache_key: Optional cache key for the request
            log: Optional existing log entry to update
            
        Returns:
            The result of the operation
            
        Raises:
            LLMServiceError: If all attempts and fallbacks fail
        """
        # Check cache first if a cache key is provided
        if cache_key and cache_key in self._cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for {operation} ({self._cache_hits} hits, {self._cache_misses} misses)")
            return self._cache[cache_key]
        elif cache_key:
            self._cache_misses += 1
        
        if log is None:
            log = self._create_request_log(operation, model, prompt)
        
        start_time = time.time()
        
        # Define which exceptions should trigger retry
        retry_exceptions = (
            LLMRequestError,
            aiohttp.ClientError,
            asyncio.TimeoutError,
        )
        
        # Define which exceptions should be considered permanent and not retried
        permanent_exceptions = (
            LLMConfigError,
            LLMResponseError,
            LLMTokenLimitError,
            ValueError,
            KeyError
        )
        
        # Use exponential backoff for retries
        @backoff.on_exception(
            backoff.expo,
            retry_exceptions,
            max_tries=self.max_retries + 1,  # +1 because first try is not a retry
            giveup=lambda e: isinstance(e, permanent_exceptions),
            on_backoff=lambda details: setattr(log, 'retry_count', details.get('tries', 0))
        )
        async def execute_with_retry():
            try:
                # Set timeout for the operation
                return await asyncio.wait_for(func(), self.timeout_seconds)
            except asyncio.TimeoutError:
                raise LLMTimeoutError(f"Request to {self.service_name} timed out after {self.timeout_seconds}s")
        
        try:
            result = await execute_with_retry()
            log.complete(str(result) if isinstance(result, (str, dict)) else "<non-string result>")
            self._record_metrics(log)
            
            # Cache the result if appropriate
            if cache_key and self._cache_enabled:
                if len(self._cache) >= self._cache_max_size:
                    # Simple cache eviction - remove a random item
                    if self._cache:
                        self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = result
                
            return result
        except Exception as e:
            log.record_error(e)
            log.duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(
                f"LLM request to {self.service_name} failed after {log.retry_count} "
                f"retries: {type(e).__name__}: {str(e)}"
            )
            
            # Try fallback services if available
            if self.fallback_services:
                for fallback in self.fallback_services:
                    logger.info(f"Trying fallback service: {fallback.service_name}")
                    try:
                        if operation == "generate_text":
                            return await fallback.generate_text(prompt)
                        elif operation == "generate_json":
                            # This is incomplete - in a real implementation we would pass all params
                            raise NotImplementedError("Fallback for generate_json not fully implemented")
                        elif operation == "generate_structured_output":
                            # This is incomplete - in a real implementation we would pass all params
                            raise NotImplementedError("Fallback for generate_structured_output not fully implemented")
                        else:
                            raise ValueError(f"Unknown operation: {operation}")
                    except Exception as fallback_error:
                        logger.warning(
                            f"Fallback service {fallback.service_name} also failed: "
                            f"{type(fallback_error).__name__}: {str(fallback_error)}"
                        )
            
            # If we get here, all attempts have failed
            self._record_metrics(log)
            raise
    
    def trim_prompt_if_needed(self, prompt: str, max_tokens: int) -> str:
        """
        Trim a prompt to fit within token limits if needed.
        
        Args:
            prompt: The prompt to trim
            max_tokens: Maximum tokens allowed
            
        Returns:
            The trimmed prompt
        """
        if not self.context_trimmer:
            logger.warning("No context trimmer available, prompt will not be trimmed")
            return prompt
            
        trimmed, token_count = self.context_trimmer.trim_content(
            prompt, max_tokens=max_tokens
        )
        
        if token_count < len(prompt.split()):
            logger.info(f"Prompt trimmed from approximately {len(prompt.split())} to {token_count} tokens")
            
        return trimmed
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
        retry_count: Optional[int] = None
    ) -> str:
        """
        Generate text from the LLM based on a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: The temperature parameter (creativity)
            max_tokens: The maximum number of tokens to generate
            timeout: Custom timeout for this specific request (in seconds)
            retry_count: Custom retry count for this specific request
            
        Returns:
            The generated text as a string
            
        Raises:
            LLMServiceError: If there's an error generating the text
        """
        pass
    
    @abstractmethod
    async def generate_json(
        self, 
        prompt: str, 
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        schema: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retry_count: Optional[int] = None
    ) -> T:
        """
        Generate JSON from the LLM based on a prompt and validate it against a Pydantic model.
        
        Args:
            prompt: The prompt to send to the LLM
            response_model: A Pydantic model class that the response should conform to
            temperature: The temperature parameter (creativity)
            max_tokens: The maximum number of tokens to generate
            schema: Optional JSON schema to guide the LLM's response format
            timeout: Custom timeout for this specific request (in seconds)
            retry_count: Custom retry count for this specific request
            
        Returns:
            A validated instance of the response_model Pydantic class
            
        Raises:
            LLMServiceError: If there's an error generating the JSON or it doesn't match the schema
        """
        pass
    
    @abstractmethod
    async def generate_structured_output(
        self, 
        prompt: str, 
        structure_name: str,
        structure_description: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
        retry_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate structured output from the LLM based on a prompt and a description of the structure.
        
        Args:
            prompt: The prompt to send to the LLM
            structure_name: A name for the structure (e.g., "TaskList")
            structure_description: A description of the fields in the structure
            temperature: The temperature parameter (creativity)
            max_tokens: The maximum number of tokens to generate
            timeout: Custom timeout for this specific request (in seconds)
            retry_count: Custom retry count for this specific request
            
        Returns:
            A dictionary representing the structured output
            
        Raises:
            LLMServiceError: If there's an error generating the structured output
        """
        pass
        
    async def generate_text_with_template(
        self,
        template_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **template_params
    ) -> str:
        """
        Generate text using a predefined prompt template.
        
        Args:
            template_name: Name of the template to use
            temperature: The temperature parameter (creativity)
            max_tokens: The maximum number of tokens to generate
            **template_params: Parameters to fill into the template
            
        Returns:
            The generated text
            
        Raises:
            LLMServiceError: If there's an error generating the text
            ValueError: If the template doesn't exist
        """
        if not self.prompt_augmentation:
            raise ValueError("Prompt augmentation service not available")
            
        messages = self.prompt_augmentation.format_with_template(template_name, **template_params)
        
        # Convert the chat messages to a single prompt for now
        # In a real implementation, we would use a different method that accepts messages directly
        prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        return await self.generate_text(prompt, temperature, max_tokens)


class GoogleGeminiService(BaseLLMService):
    """
    Google Gemini implementation of the BaseLLMService interface.
    
    This class uses the Google Generative AI library to interact with the Gemini models.
    Features include:
    - Fully async API using Google's generate_content_async
    - Token limit enforcement
    - Advanced and standard model support
    - Safety settings configuration
    - All BaseLLMService features (retry, timeout, etc.)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        advanced_model_name: Optional[str] = None,
        safety_settings: Optional[Dict] = None,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        enable_logging: bool = True,
        context_trimmer: Optional['ContextTrimmer'] = None,
        prompt_augmentation: Optional['PromptAugmentationService'] = None
    ):
        """
        Initialize the GoogleGeminiService.
        
        Args:
            api_key: The API key for Google Generative AI. If None, it will be loaded from settings.
            model_name: The name of the standard Gemini model to use
            advanced_model_name: The name of the advanced Gemini model for complex tasks
            safety_settings: Optional safety settings for the model
            max_retries: Maximum number of retries for failed requests
            timeout_seconds: Timeout in seconds for requests
            enable_logging: Whether to enable comprehensive request logging
            context_trimmer: Optional ContextTrimmer instance
            prompt_augmentation: Optional PromptAugmentationService instance
            
        Raises:
            LLMConfigError: If the API key is missing or there's an error configuring the library
        """
        if not google_import_ok:
            raise ImportError("google.generativeai library is required but not found.")
            
        # Get the API key from settings if not provided
        self.api_key = api_key
        if not self.api_key:
            if settings_import_ok and hasattr(settings, "GOOGLE_API_KEY"):
                self.api_key = settings.GOOGLE_API_KEY
            else:
                raise LLMConfigError("Google API key is required but was not provided.")
        
        # Default model names from settings or use provided values
        self.model_name = model_name
        if not self.model_name and settings_import_ok and hasattr(settings, "GEMINI_MODEL_NAME"):
            self.model_name = settings.GEMINI_MODEL_NAME
        elif not self.model_name:
            self.model_name = "gemini-1.5-flash-latest"
            
        self.advanced_model_name = advanced_model_name
        if not self.advanced_model_name and settings_import_ok and hasattr(settings, "GEMINI_ADVANCED_MODEL_NAME"):
            self.advanced_model_name = settings.GEMINI_ADVANCED_MODEL_NAME
        elif not self.advanced_model_name:
            self.advanced_model_name = "gemini-1.5-pro-latest"
            
        # Initialize the parent class
        super().__init__(
            service_name="Google Gemini",
            default_model=self.model_name,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            enable_logging=enable_logging,
            context_trimmer=context_trimmer,
            prompt_augmentation=prompt_augmentation
        )
            
        # Configure the Google Generative AI library
        try:
            genai.configure(api_key=self.api_key)
            logger.info(f"Google Gemini configured with model: {self.model_name}")
        except Exception as e:
            raise LLMConfigError(f"Failed to configure Google Generative AI: {e}")
            
        # Set up safety settings
        self.safety_settings = safety_settings or {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        } if google_import_ok else {}
    
    def _get_model(self, use_advanced: bool = False):
        """Get the Gemini model instance, optionally using the advanced model."""
        try:
            model_name = self.advanced_model_name if use_advanced else self.model_name
            return genai.GenerativeModel(model_name)
        except Exception as e:
            raise LLMConfigError(f"Failed to create Gemini model instance: {e}")
    
    def _create_generation_config(
        self, 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> GenerationConfig:
        """Create a GenerationConfig object for the Gemini model."""
        return GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=1.0,
            top_k=32,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
    
    async def _process_response_async(self, response: GenerateContentResponse) -> str:
        """
        Process the raw response from the Gemini API asynchronously.
        
        Args:
            response: The raw response from the Gemini API
            
        Returns:
            The text content of the response
            
        Raises:
            LLMResponseError: If the response is blocked or empty
        """
        if not response.candidates:
            raise LLMResponseError("No response candidates returned from Gemini API.")
            
        candidate = response.candidates[0]
        
        # Check for blocking due to safety concerns
        if hasattr(candidate, 'finish_reason'):
            if candidate.finish_reason == protos.Candidate.FinishReason.SAFETY:
                raise LLMResponseError("Response was blocked due to safety concerns.")
            elif candidate.finish_reason == protos.Candidate.FinishReason.RECITATION:
                logger.warning("Response may contain recitation/copying from training data.")
                
        # Extract the text content
        if not candidate.content.parts:
            raise LLMResponseError("Response content is empty.")
            
        text = candidate.content.parts[0].text
        if not text or text.strip() == "":
            raise LLMResponseError("Response text is empty or only whitespace.")
            
        return text
        
    def _process_response(self, response: GenerateContentResponse) -> str:
        """
        Process the raw response from the Gemini API (synchronous version).
        
        Args:
            response: The raw response from the Gemini API
            
        Returns:
            The text content of the response
            
        Raises:
            LLMResponseError: If the response is blocked or empty
        """
        # This is kept for backward compatibility
        return self._process_response_async(response)
    
    async def generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
        retry_count: Optional[int] = None,
        use_advanced_model: bool = False
    ) -> str:
        """
        Generate text from the Gemini model asynchronously.
        
        Args:
            prompt: The prompt to send to the model
            temperature: The temperature parameter (creativity)
            max_tokens: The maximum number of tokens to generate
            timeout: Custom timeout for this specific request (in seconds)
            retry_count: Custom retry count for this specific request
            use_advanced_model: Whether to use the advanced model for this request
            
        Returns:
            The generated text as a string
            
        Raises:
            LLMServiceError: If there's an error generating the text
        """
        # Apply custom retry/timeout settings if provided
        original_timeout = self.timeout_seconds
        original_retries = self.max_retries
        if timeout is not None:
            self.timeout_seconds = timeout
        if retry_count is not None:
            self.max_retries = retry_count
            
        # Check if we need to trim the prompt
        if self.context_trimmer:
            prompt = self.trim_prompt_if_needed(prompt, max_tokens=8000)  # Adjust based on model limits
            
        # Create a cache key for this request
        cache_key = self._cache_key(
            "generate_text", 
            prompt, 
            temperature=temperature, 
            max_tokens=max_tokens,
            advanced=use_advanced_model
        )
        
        # Prepare our async operation to retry
        async def execute_llm_call():
            model = self._get_model(use_advanced=use_advanced_model)
            generation_config = self._create_generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=False
            )
            
            # Use the async version of generate_content
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            return await self._process_response_async(response)
            
        try:
            # Execute with retry, timeout, and fallback
            model_name = self.advanced_model_name if use_advanced_model else self.model_name
            result = await self._with_retry_and_fallback(
                operation="generate_text",
                model=model_name,
                func=execute_llm_call,
                prompt=prompt,
                cache_key=cache_key
            )
            
            return result
        except google_api_exceptions.ResourceExhausted as e:
            raise LLMTokenLimitError(f"Token limit exceeded: {e}")
        except Exception as e:
            if isinstance(e, LLMServiceError):
                raise
            raise LLMRequestError(f"Error generating text from Gemini: {e}")
        finally:
            # Restore original settings
            if timeout is not None:
                self.timeout_seconds = original_timeout
            if retry_count is not None:
                self.max_retries = original_retries
    
    async def generate_json(
        self, 
        prompt: str, 
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        schema: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retry_count: Optional[int] = None,
        use_advanced_model: bool = True  # Default to advanced model for structured tasks
    ) -> T:
        """
        Generate JSON from the Gemini model and validate it against a Pydantic model.
        
        Args:
            prompt: The prompt to send to the model
            response_model: A Pydantic model class that the response should conform to
            temperature: The temperature parameter (creativity)
            max_tokens: The maximum number of tokens to generate
            schema: Optional JSON schema to guide the model's response format
            timeout: Custom timeout for this specific request (in seconds)
            retry_count: Custom retry count for this specific request
            use_advanced_model: Whether to use the advanced model for this request
            
        Returns:
            A validated instance of the response_model Pydantic class
            
        Raises:
            LLMServiceError: If there's an error generating the JSON or it doesn't match the schema
        """
        # Apply custom retry/timeout settings if provided
        original_timeout = self.timeout_seconds
        original_retries = self.max_retries
        if timeout is not None:
            self.timeout_seconds = timeout
        if retry_count is not None:
            self.max_retries = retry_count
            
        # Append schema information to the prompt
        if schema:
            schema_json = json.dumps(schema, indent=2)
            schema_prompt = f"\n\nOutput should follow this JSON schema:\n{schema_json}"
        else:
            # Use Pydantic model to generate schema
            model_schema = response_model.model_schema()
            schema_json = json.dumps(model_schema, indent=2)
            schema_prompt = f"\n\nOutput should follow this JSON schema:\n{schema_json}"
            
        augmented_prompt = f"{prompt}{schema_prompt}\n\nOutput ONLY valid JSON, no markdown, no other text."
        
        # Check if we need to trim the prompt
        if self.context_trimmer:
            augmented_prompt = self.trim_prompt_if_needed(augmented_prompt, max_tokens=8000)  # Adjust based on model limits
            
        # We don't cache JSON generation due to its complexity and likelihood of unique requests
        
        # Prepare our async operation to retry
        async def execute_llm_call():
            model = self._get_model(use_advanced=use_advanced_model)
            generation_config = self._create_generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True
            )
            
            # Use the async version of generate_content
            response = await model.generate_content_async(
                augmented_prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            text = await self._process_response_async(response)
            
            # Try to parse JSON from the response
            # Clean up the response to handle potential non-JSON prefix/suffix
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = text[json_start:json_end]
                data = json.loads(json_text)
            else:
                raise LLMResponseError("No JSON object found in response")
            
            # Validate against Pydantic model
            return response_model.model_validate(data)
            
        try:
            # Execute with retry, timeout, and fallback
            model_name = self.advanced_model_name if use_advanced_model else self.model_name
            result = await self._with_retry_and_fallback(
                operation="generate_json",
                model=model_name,
                func=execute_llm_call,
                prompt=augmented_prompt,
                cache_key=None  # No caching for JSON generation
            )
            
            return result
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Failed to parse JSON from response: {e}")
        except google_api_exceptions.ResourceExhausted as e:
            raise LLMTokenLimitError(f"Token limit exceeded: {e}")
        except Exception as e:
            if isinstance(e, LLMServiceError):
                raise
            if isinstance(e, ValueError) and "model_validate" in str(e):
                raise LLMResponseError(f"Response does not match schema: {e}")
            raise LLMRequestError(f"Error generating JSON from Gemini: {e}")
        finally:
            # Restore original settings
            if timeout is not None:
                self.timeout_seconds = original_timeout
            if retry_count is not None:
                self.max_retries = original_retries
    
    async def generate_structured_output(
        self, 
        prompt: str, 
        structure_name: str,
        structure_description: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
        retry_count: Optional[int] = None,
        use_advanced_model: bool = True  # Default to advanced model for structured tasks
    ) -> Dict[str, Any]:
        """
        Generate structured output from the Gemini model.
        
        Args:
            prompt: The prompt to send to the model
            structure_name: A name for the structure (e.g., "TaskList")
            structure_description: A description of the fields in the structure
            temperature: The temperature parameter (creativity)
            max_tokens: The maximum number of tokens to generate
            timeout: Custom timeout for this specific request (in seconds)
            retry_count: Custom retry count for this specific request
            use_advanced_model: Whether to use the advanced model for this request
            
        Returns:
            A dictionary representing the structured output
            
        Raises:
            LLMServiceError: If there's an error generating the structured output
        """
        # Apply custom retry/timeout settings if provided
        original_timeout = self.timeout_seconds
        original_retries = self.max_retries
        if timeout is not None:
            self.timeout_seconds = timeout
        if retry_count is not None:
            self.max_retries = retry_count
            
        # Create a prompt that describes the expected structure
        structured_prompt = (
            f"{prompt}\n\n"
            f"Please return a JSON object representing a {structure_name} with the following structure:\n"
            f"{structure_description}\n\n"
            f"Return only valid JSON without any explanation or additional text."
        )
        
        # Check if we need to trim the prompt
        if self.context_trimmer:
            structured_prompt = self.trim_prompt_if_needed(structured_prompt, max_tokens=8000)  # Adjust based on model limits
        
        # Prepare our async operation to retry
        async def execute_llm_call():
            model = self._get_model(use_advanced=use_advanced_model)
            generation_config = self._create_generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True
            )
            
            # Use the async version of generate_content
            response = await model.generate_content_async(
                structured_prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            text = await self._process_response_async(response)
            
            # Try to parse JSON from the response
            # Clean up the response to handle potential non-JSON prefix/suffix
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = text[json_start:json_end]
                return json.loads(json_text)
            else:
                raise LLMResponseError("No JSON object found in response")
                
        try:
            # Execute with retry, timeout, and fallback
            model_name = self.advanced_model_name if use_advanced_model else self.model_name
            return await self._with_retry_and_fallback(
                operation="generate_structured_output",
                model=model_name,
                func=execute_llm_call,
                prompt=structured_prompt,
                cache_key=None  # No caching for structured output generation
            )
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Failed to parse JSON from response: {e}")
        except google_api_exceptions.ResourceExhausted as e:
            raise LLMTokenLimitError(f"Token limit exceeded: {e}")
        except Exception as e:
            if isinstance(e, LLMServiceError):
                raise
            raise LLMRequestError(f"Error generating structured output from Gemini: {e}")
        finally:
            # Restore original settings
            if timeout is not None:
                self.timeout_seconds = original_timeout
            if retry_count is not None:
                self.max_retries = original_retries


# Factory function to create the appropriate LLM service based on configuration
def create_llm_service(
    provider: str = "gemini", 
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    advanced_model_name: Optional[str] = None,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
    enable_logging: bool = True,
    context_trimmer: Optional['ContextTrimmer'] = None,
    prompt_augmentation: Optional['PromptAugmentationService'] = None,
    **kwargs
) -> BaseLLMService:
    """
    Create an LLM service instance based on the specified provider for dependency injection.
    
    This factory function creates the appropriate LLM service based on configuration,
    making it easy to inject the service into other components.
    
    Args:
        provider: The LLM provider to use ("gemini" for Google Gemini)
        api_key: Optional API key for the LLM provider
        model_name: Optional model name for the provider
        advanced_model_name: Optional advanced model name for more complex tasks
        max_retries: Maximum number of retries for failed requests
        timeout_seconds: Timeout in seconds for requests
        enable_logging: Whether to enable comprehensive request logging
        context_trimmer: Optional ContextTrimmer instance
        prompt_augmentation: Optional PromptAugmentationService instance
        **kwargs: Additional configuration parameters for the service
        
    Returns:
        An instance of a BaseLLMService implementation
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider.lower() == "gemini":
        return GoogleGeminiService(
            api_key=api_key,
            model_name=model_name,
            advanced_model_name=advanced_model_name,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            enable_logging=enable_logging,
            context_trimmer=context_trimmer,
            prompt_augmentation=prompt_augmentation,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
        
# Convenience function to get a configured LLM service for use in other services
def get_llm_service() -> BaseLLMService:
    """
    Get a configured LLM service instance based on application settings.
    
    This is a convenience function for use in dependency injection.
    
    Returns:
        A configured BaseLLMService implementation
    """
    provider = "gemini"
    if settings_import_ok and hasattr(settings, "LLM_PROVIDER"):
        provider = settings.LLM_PROVIDER
        
    api_key = None
    if settings_import_ok:
        if provider.lower() == "gemini" and hasattr(settings, "GOOGLE_API_KEY"):
            api_key = settings.GOOGLE_API_KEY
            
    return create_llm_service(
        provider=provider,
        api_key=api_key
    )
