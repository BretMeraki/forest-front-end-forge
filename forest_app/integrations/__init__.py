"""
Forest App Integrations Package

This package contains external service integrations and clients.
"""

from forest_app.integrations.llm import (
    LLMClient,
    LLMError,
    LLMValidationError,
    HTAEvolveResponse,
    DistilledReflectionResponse,
    generate_response,
    LLMResponseModel
)

from forest_app.integrations.llm_service import (
    BaseLLMService,
    GoogleGeminiService,
    create_llm_service,
    LLMServiceError,
    LLMConfigError,
    LLMRequestError,
    LLMResponseError
)

__all__ = [
    # LLMClient and related
    'LLMClient',
    'LLMError',
    'LLMValidationError',
    'HTAEvolveResponse',
    'DistilledReflectionResponse',
    'generate_response',
    'LLMResponseModel',
    
    # LLM Service Abstraction Layer
    'BaseLLMService',
    'GoogleGeminiService',
    'create_llm_service',
    'LLMServiceError',
    'LLMConfigError',
    'LLMRequestError',
    'LLMResponseError'
]
