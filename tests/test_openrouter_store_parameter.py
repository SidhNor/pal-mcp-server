"""Tests for OpenRouter store parameter handling in responses endpoint.

Regression tests for GitHub Issue #348: OpenAI "store" parameter validation error
for certain models via OpenRouter.

OpenRouter's /responses endpoint rejects store:true via Zod validation. This is an
endpoint-level limitation, not model-specific. These tests verify that:
- OpenRouter provider omits the store parameter
- Direct OpenAI provider includes store: true
"""

import unittest
from unittest.mock import Mock, patch

from providers.openai_compatible import OpenAICompatibleProvider
from providers.shared import ProviderType


class MockOpenRouterProvider(OpenAICompatibleProvider):
    """Mock provider that simulates OpenRouter behavior."""

    FRIENDLY_NAME = "OpenRouter Test"

    def get_provider_type(self):
        return ProviderType.OPENROUTER

    def get_capabilities(self, model_name):
        mock_caps = Mock()
        mock_caps.default_reasoning_effort = "high"
        return mock_caps

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["openai/gpt-5-pro", "openai/gpt-5.1-codex"]


class MockOpenAIProvider(OpenAICompatibleProvider):
    """Mock provider that simulates direct OpenAI behavior."""

    FRIENDLY_NAME = "OpenAI Test"

    def get_provider_type(self):
        return ProviderType.OPENAI

    def get_capabilities(self, model_name):
        mock_caps = Mock()
        mock_caps.default_reasoning_effort = "high"
        return mock_caps

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["gpt-5-pro", "gpt-5.1-codex"]


class MockXAIProvider(OpenAICompatibleProvider):
    """Mock provider that simulates xAI Responses API models without default reasoning effort."""

    FRIENDLY_NAME = "XAI Test"

    def get_provider_type(self):
        return ProviderType.XAI

    def get_capabilities(self, model_name):
        mock_caps = Mock()
        mock_caps.default_reasoning_effort = None
        return mock_caps

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["grok-4.20-reasoning"]


class TestStoreParameterHandling(unittest.TestCase):
    """Test store parameter is conditionally included based on provider type.

    **Feature: openrouter-store-parameter-fix, Property 1: OpenRouter requests omit store parameter**
    **Feature: openrouter-store-parameter-fix, Property 2: Direct OpenAI requests include store parameter**
    """

    def test_openrouter_responses_omits_store_parameter(self):
        """Test that OpenRouter provider omits store parameter from responses endpoint.

        **Feature: openrouter-store-parameter-fix, Property 1: OpenRouter requests omit store parameter**
        **Validates: Requirements 1.1, 2.1**

        OpenRouter's /responses endpoint rejects store:true via Zod validation (Issue #348).
        The store parameter should be omitted entirely for OpenRouter requests.
        """
        # Capture the completion_params passed to the API
        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            # Return a mock response
            mock_response = Mock()
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with patch.object(
            MockOpenRouterProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
        ):
            provider = MockOpenRouterProvider("test-key")

            # Call the method that builds completion_params
            provider._generate_with_responses_endpoint(
                model_name="openai/gpt-5-pro",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
            )

        # Verify store parameter is NOT in the request
        self.assertNotIn("store", captured_params, "OpenRouter requests should NOT include 'store' parameter")

    def test_openai_responses_includes_store_parameter(self):
        """Test that direct OpenAI provider includes store parameter in responses endpoint.

        **Feature: openrouter-store-parameter-fix, Property 2: Direct OpenAI requests include store parameter**
        **Validates: Requirements 1.2, 2.2**

        Direct OpenAI API supports the store parameter for stored completions.
        The store parameter should be included with value True for OpenAI requests.
        """
        # Capture the completion_params passed to the API
        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            # Return a mock response
            mock_response = Mock()
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with patch.object(
            MockOpenAIProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
        ):
            provider = MockOpenAIProvider("test-key")

            # Call the method that builds completion_params
            provider._generate_with_responses_endpoint(
                model_name="gpt-5-pro",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
            )

        # Verify store parameter IS in the request with value True
        self.assertIn("store", captured_params, "OpenAI requests should include 'store' parameter")
        self.assertTrue(captured_params["store"], "OpenAI requests should have store=True")

    def test_responses_include_reasoning_only_when_default_effort_is_configured(self):
        """Responses API requests should include reasoning only when capabilities specify a default."""
        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_response = Mock()
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with patch.object(
            MockOpenAIProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
        ):
            provider = MockOpenAIProvider("test-key")
            provider._generate_with_responses_endpoint(
                model_name="gpt-5-pro",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                capabilities=provider.get_capabilities("gpt-5-pro"),
            )

        self.assertEqual(captured_params.get("reasoning"), {"effort": "high"})

    def test_responses_omit_reasoning_when_no_default_effort_is_configured(self):
        """Responses API requests should not inject reasoning without explicit config."""
        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_response = Mock()
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with patch.object(
            MockXAIProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
        ):
            provider = MockXAIProvider("test-key")
            provider._generate_with_responses_endpoint(
                model_name="grok-4.20-reasoning",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                capabilities=provider.get_capabilities("grok-4.20-reasoning"),
            )

        self.assertNotIn("reasoning", captured_params, "Responses requests should omit reasoning by default")

    def test_responses_use_max_output_tokens_parameter(self):
        """Responses API requests should use max_output_tokens for output limits."""
        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_response = Mock()
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with patch.object(
            MockXAIProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
        ):
            provider = MockXAIProvider("test-key")
            provider._generate_with_responses_endpoint(
                model_name="grok-4.3",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                max_output_tokens=123,
                capabilities=provider.get_capabilities("grok-4.3"),
            )

        self.assertEqual(captured_params.get("max_output_tokens"), 123)
        self.assertNotIn("max_completion_tokens", captured_params)


if __name__ == "__main__":
    unittest.main()
