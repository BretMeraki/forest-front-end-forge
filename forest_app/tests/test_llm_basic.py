"""
Basic tests for the LLM service implementation.
"""

import pytest
from unittest.mock import MagicMock, patch

from forest_app.integrations.llm_service import (
    GoogleGeminiService,
    create_llm_service
)

def test_create_llm_service():
    """Test that the factory function can create a service without errors."""
    with patch("google.generativeai.configure") as mock_configure:
        with patch("google.generativeai.GenerativeModel") as mock_generative_model:
            mock_model_instance = MagicMock()
            mock_generative_model.return_value = mock_model_instance
            
            # Create the service
            service = create_llm_service(
                provider="gemini",
                api_key="test_key"
            )
            
            # Basic assertions
            assert service is not None
            assert isinstance(service, GoogleGeminiService)
            mock_configure.assert_called_once()
