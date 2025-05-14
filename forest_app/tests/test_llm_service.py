"""
Tests for the LLM service implementation.

These tests verify that the LLM service functionality works correctly,
including async operation, retry logic, timeouts, fallbacks, and token controls.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from forest_app.integrations.llm_service import (
    BaseLLMService,
    GoogleGeminiService,
    create_llm_service,
    get_llm_service,
    LLMRequestError,
    LLMResponseError,
    LLMTimeoutError,
    LLMTokenLimitError
)

from forest_app.integrations.context_trimmer import ContextTrimmer
from forest_app.integrations.prompt_augmentation import PromptAugmentationService

# Setup logging for tests
logging.basicConfig(level=logging.INFO)

class TestLLMService:
    """Test the LLM service implementations."""

    @pytest.mark.asyncio
    async def test_create_llm_service(self):
        """Test that the factory function creates the correct service."""
        service = create_llm_service(provider="gemini")
        assert isinstance(service, GoogleGeminiService)
        assert service.service_name == "Google Gemini"
        
    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_generate_text_basic(self, mock_generative_model):
        """Test basic text generation functionality."""
        # Setup mock response
        mock_model_instance = AsyncMock()
        mock_model_instance.generate_content_async.return_value = MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(
                        parts=[MagicMock(text="This is a test response")]
                    )
                )
            ]
        )
        mock_generative_model.return_value = mock_model_instance
        
        # Create service and test
        service = GoogleGeminiService(
            api_key="test_key",
            enable_logging=True
        )
        
        result = await service.generate_text("Test prompt", temperature=0.7, max_tokens=100)
        
        # Assertions
        assert result == "This is a test response"
        mock_model_instance.generate_content_async.assert_called_once()
        assert len(service.request_logs) == 1
        assert service.request_logs[0].success == True
    
    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_generate_text_with_retry(self, mock_generative_model):
        """Test retry logic for failed requests."""
        # Setup mock to fail once then succeed
        mock_model_instance = AsyncMock()
        
        # First call raises an error, second succeeds
        mock_model_instance.generate_content_async.side_effect = [
            LLMRequestError("Test error"),
            MagicMock(
                candidates=[
                    MagicMock(
                        content=MagicMock(
                            parts=[MagicMock(text="Retry succeeded")]
                        )
                    )
                ]
            )
        ]
        mock_generative_model.return_value = mock_model_instance
        
        # Create service and test
        service = GoogleGeminiService(
            api_key="test_key",
            max_retries=2,
            enable_logging=True
        )
        
        result = await service.generate_text("Test prompt")
        
        # Assertions
        assert result == "Retry succeeded"
        assert mock_model_instance.generate_content_async.call_count == 2
        assert len(service.request_logs) == 1
        assert service.request_logs[0].retry_count == 1
        assert service.request_logs[0].success == True
    
    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_generate_json(self, mock_generative_model):
        """Test JSON generation and validation."""
        # Create a simple Pydantic model for testing
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            name: str
            value: int
        
        # Setup mock response with JSON
        mock_model_instance = AsyncMock()
        mock_model_instance.generate_content_async.return_value = MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(
                        parts=[MagicMock(text='{"name": "test", "value": 42}')]
                    )
                )
            ]
        )
        mock_generative_model.return_value = mock_model_instance
        
        # Create service and test
        service = GoogleGeminiService(
            api_key="test_key",
            enable_logging=True
        )
        
        result = await service.generate_json("Test JSON prompt", TestModel)
        
        # Assertions
        assert isinstance(result, TestModel)
        assert result.name == "test"
        assert result.value == 42
        mock_model_instance.generate_content_async.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_timeout_handling(self, mock_generative_model):
        """Test timeout handling."""
        # Setup mock to simulate a timeout
        mock_model_instance = AsyncMock()
        
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.2)  # Simulate slow response
            return MagicMock(
                candidates=[
                    MagicMock(
                        content=MagicMock(
                            parts=[MagicMock(text="Slow response")]
                        )
                    )
                ]
            )
            
        mock_model_instance.generate_content_async.side_effect = slow_response
        mock_generative_model.return_value = mock_model_instance
        
        # Create service with very short timeout
        service = GoogleGeminiService(
            api_key="test_key",
            timeout_seconds=0.1,  # Very short timeout
            enable_logging=True
        )
        
        # Test that it raises a timeout error
        with pytest.raises(LLMTimeoutError):
            await service.generate_text("Test prompt")
    
    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_context_trimmer(self, mock_generative_model):
        """Test context trimming functionality."""
        # Setup mock response
        mock_model_instance = AsyncMock()
        mock_model_instance.generate_content_async.return_value = MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(
                        parts=[MagicMock(text="Trimmed response")]
                    )
                )
            ]
        )
        mock_generative_model.return_value = mock_model_instance
        
        # Create a real ContextTrimmer
        trimmer = ContextTrimmer()
        
        # Create service with the trimmer
        service = GoogleGeminiService(
            api_key="test_key",
            context_trimmer=trimmer,
            enable_logging=True
        )
        
        # Create a very long prompt
        long_prompt = "Test " * 1000  # Approximately 5000 characters
        
        # Generate text with the long prompt
        result = await service.generate_text(long_prompt)
        
        # Assertions
        assert result == "Trimmed response"
        
        # Verify the prompt sent to the model is shorter than original
        args, kwargs = mock_model_instance.generate_content_async.call_args
        sent_prompt = args[0]
        assert len(sent_prompt) < len(long_prompt)
    
    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_fallback(self, mock_generative_model):
        """Test fallback service mechanism."""
        # Setup primary service to fail
        mock_primary_model = AsyncMock()
        mock_primary_model.generate_content_async.side_effect = LLMRequestError("Primary service error")
        
        # Setup fallback service to succeed
        mock_fallback_model = AsyncMock()
        mock_fallback_model.generate_content_async.return_value = MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(
                        parts=[MagicMock(text="Fallback service response")]
                    )
                )
            ]
        )
        
        # Use side_effect to return different instances for different calls
        mock_generative_model.side_effect = [mock_primary_model, mock_fallback_model]
        
        # Create primary and fallback services
        primary_service = GoogleGeminiService(
            api_key="primary_key",
            model_name="primary-model",
            enable_logging=True
        )
        
        fallback_service = GoogleGeminiService(
            api_key="fallback_key",
            model_name="fallback-model",
            enable_logging=True
        )
        
        # Add fallback to primary
        primary_service.add_fallback(fallback_service)
        
        # Test that fallback works
        result = await primary_service.generate_text("Test prompt")
        
        # Assertions
        assert result == "Fallback service response"
        mock_primary_model.generate_content_async.assert_called_once()
        # Note: Our current fallback implementation is not fully wired up for all params

    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_prompt_augmentation(self, mock_generative_model):
        """Test prompt augmentation integration."""
        # Setup mock response
        mock_model_instance = AsyncMock()
        mock_model_instance.generate_content_async.return_value = MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(
                        parts=[MagicMock(text="Augmented response")]
                    )
                )
            ]
        )
        mock_generative_model.return_value = mock_model_instance
        
        # Create a real PromptAugmentationService
        augmentation = PromptAugmentationService()
        
        # Create service with the augmentation service
        service = GoogleGeminiService(
            api_key="test_key",
            prompt_augmentation=augmentation,
            enable_logging=True
        )
        
        # Test template-based text generation
        # This will fail since we defined the template methods but don't have actual data in the test
        # Let's mock the format_with_template method
        
        with patch.object(augmentation, 'format_with_template', return_value=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Test template prompt"}
        ]):
            result = await service.generate_text_with_template(
                "json_generation",
                schema_description="A test schema",
                input_data="Some test data",
                requirements="No special requirements"
            )
        
        # Assertions
        assert result == "Augmented response"
        # Verify the right template was used in the augmentation service
        augmentation.format_with_template.assert_called_once_with(
            "json_generation",
            schema_description="A test schema",
            input_data="Some test data",
            requirements="No special requirements"
        )
        
    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_caching(self, mock_generative_model):
        """Test caching for identical requests."""
        # Setup mock response
        mock_model_instance = AsyncMock()
        mock_model_instance.generate_content_async.return_value = MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(
                        parts=[MagicMock(text="Cached response")]
                    )
                )
            ]
        )
        mock_generative_model.return_value = mock_model_instance
        
        # Create service with caching enabled
        service = GoogleGeminiService(
            api_key="test_key",
            enable_logging=True
        )
        service._cache_enabled = True
        
        # Make the same request twice
        prompt = "Test prompt for caching"
        result1 = await service.generate_text(prompt, temperature=0.7, max_tokens=100)
        result2 = await service.generate_text(prompt, temperature=0.7, max_tokens=100)
        
        # Assertions
        assert result1 == "Cached response"
        assert result2 == "Cached response"
        
        # Verify the model was only called once
        mock_model_instance.generate_content_async.assert_called_once()
        
        # Verify cache stats
        assert service._cache_hits == 1
        assert service._cache_misses == 1  # First miss, then a hit
