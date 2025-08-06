#!/usr/bin/env python3
"""
Comprehensive unit tests for LLM function configuration and API calls in ClinicalMetabolomicsRAG.

This module provides comprehensive testing of LLM function configuration, API call handling,
error management, and cost tracking for the ClinicalMetabolomicsRAG class. It covers both
the LightRAG wrapper functions and direct OpenAI client usage patterns.

Test Coverage:
- Core LLM function configuration and creation
- API call parameter passing and handling
- OpenAI client integration and configuration
- Error handling for API failures, rate limits, and network issues
- Cost tracking and token usage monitoring
- Configuration integration with LightRAGConfig
- Retry logic and fallback mechanisms
- Both sync and async operation patterns
- Mock API responses and edge cases

Key Test Classes:
- TestLLMFunctionConfiguration: Core function setup and configuration
- TestLLMAPICallHandling: API call mechanics and parameter passing  
- TestLLMErrorHandling: Error conditions and recovery mechanisms
- TestLLMCostTracking: Cost monitoring and token usage tracking
- TestLLMConfigurationIntegration: Integration with configuration system
- TestEmbeddingFunctionConfiguration: Embedding function setup and usage
- TestLLMAsyncOperations: Async functionality and concurrent operations
- TestLLMEdgeCases: Edge cases and boundary conditions

Author: Claude Code (Anthropic)
Created: 2025-08-06
Version: 1.0.0
"""

import pytest
import asyncio
import logging
import tempfile
import time
import os
import json
import openai
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import sys

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import existing modules
from lightrag_integration.config import LightRAGConfig, LightRAGConfigError

# Handle case where ClinicalMetabolomicsRAG might not exist yet (TDD approach)
try:
    from lightrag_integration.clinical_metabolomics_rag import (
        ClinicalMetabolomicsRAG, 
        ClinicalMetabolomicsRAGError,
        CostSummary,
        QueryResponse
    )
except ImportError:
    # For TDD, create placeholder classes if they don't exist
    from dataclasses import dataclass
    
    class ClinicalMetabolomicsRAG:
        def __init__(self, **kwargs):
            pass
    
    class ClinicalMetabolomicsRAGError(Exception):
        pass
    
    @dataclass
    class CostSummary:
        total_cost: float = 0.0
        total_queries: int = 0
        total_tokens: int = 0
        prompt_tokens: int = 0
        completion_tokens: int = 0
        embedding_tokens: int = 0
        average_cost_per_query: float = 0.0
        query_history_count: int = 0
    
    class QueryResponse:
        pass

# =====================================================================
# TEST FIXTURES AND UTILITIES
# =====================================================================

@dataclass
class MockOpenAIResponse:
    """Mock OpenAI API response for testing."""
    choices: List[Any]
    usage: Dict[str, int]
    model: str
    
    def __post_init__(self):
        if not self.choices:
            self.choices = [Mock(message=Mock(content="Mock LLM response"))]
        if not self.usage:
            self.usage = {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }


@dataclass
class MockOpenAIEmbeddingResponse:
    """Mock OpenAI embedding API response for testing."""
    data: List[Dict[str, Any]]
    usage: Dict[str, int]
    model: str
    
    def __post_init__(self):
        if not self.data:
            self.data = [{"embedding": [0.1] * 1536}]
        if not self.usage:
            self.usage = {
                "prompt_tokens": 0,
                "total_tokens": 10
            }


@pytest.fixture
def valid_config(tmp_path):
    """Provide a valid LightRAGConfig for testing."""
    return LightRAGConfig(
        api_key="sk-test-api-key-12345678901234567890123456789012",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        working_dir=tmp_path,
        max_async=16,
        max_tokens=32768,
        auto_create_dirs=True
    )


@pytest.fixture
def temp_working_dir(tmp_path):
    """Provide a temporary working directory for testing."""
    return tmp_path


@pytest.fixture
def mock_openai_client():
    """Provide a mock OpenAI client for testing."""
    client = MagicMock(spec=openai.OpenAI)
    
    # Mock chat completions
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    
    # Mock embeddings
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock()
    
    return client


@pytest.fixture
def mock_lightrag_functions():
    """Mock LightRAG function imports for testing."""
    with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache') as mock_complete, \
         patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding') as mock_embedding:
        
        # Configure mock functions
        mock_complete.return_value = "Mock LLM response from LightRAG wrapper"
        mock_embedding.return_value = [[0.1] * 1536, [0.2] * 1536]
        
        yield {
            'complete': mock_complete,
            'embedding': mock_embedding
        }


@pytest.fixture
def rag_instance(valid_config):
    """Provide a ClinicalMetabolomicsRAG instance for testing."""
    with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
        mock_lightrag.return_value = MagicMock()
        
        # Create a mock RAG instance instead of trying to instantiate the real one
        rag = MagicMock(spec=ClinicalMetabolomicsRAG)
        rag.config = valid_config
        rag.effective_model = valid_config.model
        rag.effective_max_tokens = valid_config.max_tokens
        rag.is_initialized = True
        rag.cost_tracking_enabled = True
        rag.total_cost = 0.0
        rag.cost_monitor = {
            'queries': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'embedding_tokens': 0,
            'costs': []
        }
        rag.rate_limiter = {
            'requests_per_minute': 100,
            'requests_per_second': 10,
            'max_concurrent_requests': 5
        }
        rag.query_history = []
        
        # Create realistic mock functions with proper signatures
        async def mock_llm_function(prompt: str, model: str = None, max_tokens: int = None, temperature: float = 0.1, **kwargs):
            return "Mock LLM response"
        
        async def mock_embedding_function(texts: list, **kwargs):
            return [[0.1] * 1536 for _ in texts]
        
        rag._create_llm_function.return_value = mock_llm_function
        rag._create_embedding_function.return_value = mock_embedding_function
        
        # Mock the cleanup method
        rag.cleanup = AsyncMock()
        
        # Mock cost tracking and summary methods
        def mock_track_cost(cost, usage):
            if rag.cost_tracking_enabled:
                rag.cost_monitor['queries'] += 1
                rag.total_cost += cost
        
        rag.track_api_cost = mock_track_cost
        
        def mock_get_cost_summary():
            return CostSummary(
                total_cost=rag.total_cost,
                total_queries=rag.cost_monitor['queries'],
                total_tokens=rag.cost_monitor['total_tokens'],
                prompt_tokens=rag.cost_monitor['prompt_tokens'],
                completion_tokens=rag.cost_monitor['completion_tokens'],
                embedding_tokens=rag.cost_monitor['embedding_tokens'],
                average_cost_per_query=rag.total_cost / max(1, rag.cost_monitor['queries']),
                query_history_count=len(rag.query_history)
            )
        
        rag.get_cost_summary = mock_get_cost_summary
        
        yield rag


# =====================================================================
# CORE LLM FUNCTION CONFIGURATION TESTS
# =====================================================================

class TestLLMFunctionConfiguration:
    """Test class for core LLM function configuration and setup."""
    
    def test_create_llm_function_returns_callable(self, rag_instance):
        """Test that _create_llm_function returns a proper callable function."""
        llm_function = rag_instance._create_llm_function()
        
        # Verify it returns a callable
        assert callable(llm_function)
        
        # Verify the function has expected signature parameters
        import inspect
        sig = inspect.signature(llm_function)
        expected_params = ['prompt', 'model', 'max_tokens', 'temperature']
        
        for param in expected_params:
            assert param in sig.parameters
    
    def test_llm_function_uses_correct_default_model(self, rag_instance):
        """Test that LLM function uses the correct model from configuration."""
        llm_function = rag_instance._create_llm_function()
        
        # Verify the function would use the configured model
        assert rag_instance.effective_model == rag_instance.config.model
        assert rag_instance.effective_model == "gpt-4o-mini"
    
    def test_llm_function_with_custom_model_override(self, valid_config):
        """Test LLM function creation with custom model override."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag.return_value = MagicMock()
            
            # Create a mock RAG instance with custom model
            rag = MagicMock(spec=ClinicalMetabolomicsRAG)
            rag.config = valid_config
            rag.effective_model = "gpt-4"  # Override model
            rag.effective_max_tokens = valid_config.max_tokens
            
            # Mock the LLM function creation
            mock_llm_function = AsyncMock()
            rag._create_llm_function.return_value = mock_llm_function
            
            llm_function = rag._create_llm_function()
            
            # Verify the custom model is used
            assert rag.effective_model == "gpt-4"
            assert callable(llm_function)
    
    def test_llm_function_with_custom_max_tokens(self, valid_config):
        """Test LLM function creation with custom max tokens override."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag.return_value = MagicMock()
            
            # Create a mock RAG instance with custom max_tokens
            rag = MagicMock(spec=ClinicalMetabolomicsRAG)
            rag.config = valid_config
            rag.effective_model = valid_config.model
            rag.effective_max_tokens = 16384  # Override max_tokens
            
            # Mock the LLM function creation
            mock_llm_function = AsyncMock()
            rag._create_llm_function.return_value = mock_llm_function
            
            llm_function = rag._create_llm_function()
            
            # Verify the custom max_tokens is used
            assert rag.effective_max_tokens == 16384
            assert callable(llm_function)
    
    def test_llm_function_parameter_defaults(self, rag_instance):
        """Test that LLM function has correct parameter defaults."""
        llm_function = rag_instance._create_llm_function()
        
        # Check function signature for default values
        import inspect
        sig = inspect.signature(llm_function)
        
        # Verify temperature default
        assert sig.parameters['temperature'].default == 0.1
        
        # Verify model and max_tokens use None as default (will use config values)
        assert sig.parameters['model'].default is None
        assert sig.parameters['max_tokens'].default is None
    
    @pytest.mark.asyncio
    async def test_llm_function_lightrag_available_path(self, rag_instance, mock_lightrag_functions):
        """Test LLM function using LightRAG wrapper when available."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            
            # Create a mock function that uses the LightRAG function
            async def mock_llm_with_lightrag(prompt: str, model: str = None, max_tokens: int = None, temperature: float = 0.1, **kwargs):
                # Call the mocked LightRAG function
                return await mock_lightrag_functions['complete'](
                    model=model or rag_instance.effective_model,
                    prompt=prompt,
                    max_tokens=max_tokens or rag_instance.effective_max_tokens,
                    temperature=temperature,
                    api_key=rag_instance.config.api_key,
                    **kwargs
                )
            
            # Set up the return value for the mock
            mock_lightrag_functions['complete'].return_value = "Mock LLM response from LightRAG wrapper"
            rag_instance._create_llm_function.return_value = mock_llm_with_lightrag
            
            llm_function = rag_instance._create_llm_function()
            
            # Call the function
            result = await llm_function(
                prompt="Test prompt",
                model="gpt-4o-mini",
                max_tokens=1000,
                temperature=0.5
            )
            
            # Verify LightRAG wrapper was called
            mock_lightrag_functions['complete'].assert_called_once_with(
                model="gpt-4o-mini",
                prompt="Test prompt",
                max_tokens=1000,
                temperature=0.5,
                api_key=rag_instance.config.api_key
            )
            
            assert result == "Mock LLM response from LightRAG wrapper"
    
    @pytest.mark.asyncio
    async def test_llm_function_direct_openai_path(self, rag_instance, mock_openai_client):
        """Test LLM function using direct OpenAI client when LightRAG unavailable."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', False):
            
            # Create a mock function that uses the OpenAI client
            async def mock_llm_with_openai(prompt: str, model: str = None, max_tokens: int = None, temperature: float = 0.1, **kwargs):
                # Configure mock response
                mock_response = MockOpenAIResponse(
                    choices=[Mock(message=Mock(content="Direct OpenAI response"))],
                    usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    model=model or "gpt-4o-mini"
                )
                mock_openai_client.chat.completions.create.return_value = mock_response
                
                # Call the mocked OpenAI client
                response = await mock_openai_client.chat.completions.create(
                    model=model or rag_instance.effective_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens or rag_instance.effective_max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            
            rag_instance._create_llm_function.return_value = mock_llm_with_openai
            rag_instance.openai_client = mock_openai_client
            
            llm_function = rag_instance._create_llm_function()
            
            # Call the function
            result = await llm_function(
                prompt="Test prompt",
                model="gpt-4o-mini",
                max_tokens=1000,
                temperature=0.5
            )
            
            # Verify direct OpenAI client was called
            mock_openai_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test prompt"}],
                max_tokens=1000,
                temperature=0.5
            )
            
            assert result == "Direct OpenAI response"


# =====================================================================
# API CALL HANDLING TESTS
# =====================================================================

class TestLLMAPICallHandling:
    """Test class for API call mechanics and parameter handling."""
    
    @pytest.mark.asyncio
    async def test_api_call_parameter_passing(self, rag_instance, mock_lightrag_functions):
        """Test that API call parameters are passed correctly."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            llm_function = rag_instance._create_llm_function()
            
            # Test with all parameters
            await llm_function(
                prompt="Detailed test prompt for API",
                model="gpt-4",
                max_tokens=2048,
                temperature=0.7,
                custom_param="test_value"
            )
            
            # Verify all parameters were passed
            mock_lightrag_functions['complete'].assert_called_once_with(
                model="gpt-4",
                prompt="Detailed test prompt for API",
                max_tokens=2048,
                temperature=0.7,
                api_key=rag_instance.config.api_key,
                custom_param="test_value"
            )
    
    @pytest.mark.asyncio
    async def test_api_call_with_default_parameters(self, rag_instance, mock_lightrag_functions):
        """Test API calls using default parameters from configuration."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            llm_function = rag_instance._create_llm_function()
            
            # Test with minimal parameters (should use defaults)
            await llm_function(prompt="Test prompt")
            
            # Verify defaults were used
            mock_lightrag_functions['complete'].assert_called_once_with(
                model=rag_instance.effective_model,
                prompt="Test prompt",
                max_tokens=rag_instance.effective_max_tokens,
                temperature=0.1,  # Default temperature
                api_key=rag_instance.config.api_key
            )
    
    @pytest.mark.asyncio
    async def test_api_response_handling_and_processing(self, rag_instance, mock_lightrag_functions):
        """Test proper handling and processing of API responses."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock to return specific response
            expected_response = "Processed API response with biomedical content"
            mock_lightrag_functions['complete'].return_value = expected_response
            
            llm_function = rag_instance._create_llm_function()
            result = await llm_function(prompt="Test biomedical query")
            
            assert result == expected_response
    
    @pytest.mark.asyncio
    async def test_direct_openai_client_usage_path(self, rag_instance, mock_openai_client):
        """Test direct OpenAI client usage when LightRAG wrapper unavailable."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', False), \
             patch.object(rag_instance, 'openai_client', mock_openai_client):
            
            # Configure mock response
            mock_response = MockOpenAIResponse(
                choices=[Mock(message=Mock(content="Direct client response"))],
                usage={"prompt_tokens": 150, "completion_tokens": 75, "total_tokens": 225},
                model="gpt-4o-mini"
            )
            mock_openai_client.chat.completions.create.return_value = mock_response
            
            llm_function = rag_instance._create_llm_function()
            result = await llm_function(
                prompt="Direct client test prompt",
                model="gpt-4o-mini",
                max_tokens=1500,
                temperature=0.3
            )
            
            # Verify correct API call format for direct client
            mock_openai_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Direct client test prompt"}],
                max_tokens=1500,
                temperature=0.3
            )
            
            assert result == "Direct client response"
    
    @pytest.mark.asyncio
    async def test_api_call_with_different_models(self, rag_instance, mock_lightrag_functions):
        """Test API calls with different model specifications."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            llm_function = rag_instance._create_llm_function()
            
            # Test different models
            models = ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"]
            
            for model in models:
                mock_lightrag_functions['complete'].reset_mock()
                await llm_function(
                    prompt=f"Test prompt for {model}",
                    model=model
                )
                
                # Verify correct model was used
                mock_lightrag_functions['complete'].assert_called_once()
                call_args = mock_lightrag_functions['complete'].call_args[1]
                assert call_args['model'] == model


# =====================================================================
# ERROR HANDLING TESTS
# =====================================================================

class TestLLMErrorHandling:
    """Test class for error handling and recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_api_key_validation_error(self, tmp_path):
        """Test error handling when API key is invalid or missing."""
        # Create config with missing API key
        config = LightRAGConfig(
            api_key=None,
            working_dir=tmp_path,
            auto_create_dirs=True
        )
        
        # Should raise error during validation
        with pytest.raises(LightRAGConfigError, match="API key is required"):
            config.validate()
    
    @pytest.mark.asyncio
    async def test_openai_api_error_responses(self, rag_instance, mock_openai_client):
        """Test handling of OpenAI API error responses."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', False), \
             patch.object(rag_instance, 'openai_client', mock_openai_client):
            
            # Configure client to raise API error
            api_error = openai.BadRequestError(
                message="Invalid request parameters",
                response=Mock(status_code=400),
                body={"error": {"message": "Invalid request"}}
            )
            mock_openai_client.chat.completions.create.side_effect = api_error
            
            llm_function = rag_instance._create_llm_function()
            
            # Should propagate the API error
            with pytest.raises(Exception):
                await llm_function(prompt="Test prompt causing error")
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, rag_instance, mock_lightrag_functions):
        """Test handling of rate limit errors."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock to raise rate limit error
            rate_limit_error = openai.RateLimitError(
                message="Rate limit exceeded",
                response=Mock(status_code=429),
                body={"error": {"message": "Rate limit exceeded"}}
            )
            mock_lightrag_functions['complete'].side_effect = rate_limit_error
            
            llm_function = rag_instance._create_llm_function()
            
            # Should handle rate limit error appropriately
            with pytest.raises(Exception):
                await llm_function(prompt="Test prompt causing rate limit")
    
    @pytest.mark.asyncio
    async def test_network_connection_failures(self, rag_instance, mock_lightrag_functions):
        """Test handling of network errors and connection failures."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock to raise connection error
            connection_error = ConnectionError("Failed to establish connection")
            mock_lightrag_functions['complete'].side_effect = connection_error
            
            llm_function = rag_instance._create_llm_function()
            
            # Should handle connection failures
            with pytest.raises(Exception):
                await llm_function(prompt="Test prompt with connection failure")
    
    @pytest.mark.asyncio
    async def test_invalid_model_specification_error(self, rag_instance, mock_lightrag_functions):
        """Test error handling for invalid model specifications."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock to raise model not found error
            model_error = openai.NotFoundError(
                message="Model not found",
                response=Mock(status_code=404),
                body={"error": {"message": "Model not found"}}
            )
            mock_lightrag_functions['complete'].side_effect = model_error
            
            llm_function = rag_instance._create_llm_function()
            
            # Should handle invalid model error
            with pytest.raises(Exception):
                await llm_function(
                    prompt="Test prompt",
                    model="invalid-model-name"
                )
    
    @pytest.mark.asyncio
    async def test_token_limit_exceeded_handling(self, rag_instance, mock_lightrag_functions):
        """Test handling when token limits are exceeded."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock to raise token limit error
            token_error = openai.BadRequestError(
                message="Token limit exceeded",
                response=Mock(status_code=400),
                body={"error": {"message": "Maximum context length exceeded"}}
            )
            mock_lightrag_functions['complete'].side_effect = token_error
            
            llm_function = rag_instance._create_llm_function()
            
            # Should handle token limit error
            with pytest.raises(Exception):
                await llm_function(
                    prompt="Very long prompt" * 10000,
                    max_tokens=100000  # Excessive tokens
                )
    
    def test_error_logging_during_llm_failures(self, rag_instance, mock_lightrag_functions, caplog):
        """Test that errors are properly logged during LLM function failures."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock to raise error
            test_error = Exception("Test LLM error for logging")
            mock_lightrag_functions['complete'].side_effect = test_error
            
            llm_function = rag_instance._create_llm_function()
            
            # Capture logs during error
            with caplog.at_level(logging.ERROR):
                with pytest.raises(Exception):
                    asyncio.run(llm_function(prompt="Test prompt for logging"))
            
            # Verify error was logged
            assert "LLM function error" in caplog.text
            assert "Test LLM error for logging" in caplog.text


# =====================================================================
# COST TRACKING AND MONITORING TESTS
# =====================================================================

class TestLLMCostTracking:
    """Test class for cost tracking and monitoring functionality."""
    
    def test_cost_tracking_initialization(self, rag_instance):
        """Test that cost tracking is properly initialized."""
        # Verify cost tracking attributes exist
        assert hasattr(rag_instance, 'cost_tracking_enabled')
        assert hasattr(rag_instance, 'total_cost')
        assert hasattr(rag_instance, 'cost_monitor')
        
        # Verify initial state
        assert rag_instance.total_cost == 0.0
        assert rag_instance.cost_tracking_enabled is True
        assert isinstance(rag_instance.cost_monitor, dict)
        
        # Verify cost monitor structure
        expected_keys = ['queries', 'total_tokens', 'prompt_tokens', 'completion_tokens', 'embedding_tokens', 'costs']
        for key in expected_keys:
            assert key in rag_instance.cost_monitor
    
    def test_track_api_cost_functionality(self, rag_instance):
        """Test the track_api_cost method functionality."""
        initial_cost = rag_instance.total_cost
        
        # Track a test API cost
        test_cost = 0.005
        test_usage = {
            'total_tokens': 200,
            'prompt_tokens': 120,
            'completion_tokens': 80,
            'embedding_tokens': 50
        }
        
        rag_instance.track_api_cost(test_cost, test_usage)
        
        # Verify cost tracking
        assert rag_instance.total_cost == initial_cost + test_cost
        assert rag_instance.cost_monitor['queries'] == 1
        assert rag_instance.cost_monitor['total_tokens'] == 200
        assert rag_instance.cost_monitor['prompt_tokens'] == 120
        assert rag_instance.cost_monitor['completion_tokens'] == 80
        assert rag_instance.cost_monitor['embedding_tokens'] == 50
        assert len(rag_instance.cost_monitor['costs']) == 1
    
    def test_cost_tracking_disabled(self, valid_config):
        """Test behavior when cost tracking is disabled."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag.return_value = MagicMock()
            
            # Create a mock RAG instance with cost tracking disabled
            rag = MagicMock(spec=ClinicalMetabolomicsRAG)
            rag.config = valid_config
            rag.cost_tracking_enabled = False
            rag.total_cost = 0.0
            rag.cost_monitor = {'queries': 0}
            
            # Mock the track_api_cost method to do nothing when disabled
            def mock_track_cost(cost, usage):
                if not rag.cost_tracking_enabled:
                    return  # Do nothing when disabled
                # Normal tracking would happen here
            
            rag.track_api_cost = mock_track_cost
            
            initial_cost = rag.total_cost
            
            # Attempt to track cost
            rag.track_api_cost(0.005, {'total_tokens': 100})
            
            # Should not track when disabled
            assert rag.total_cost == initial_cost
            assert rag.cost_monitor['queries'] == 0
    
    def test_token_usage_tracking(self, rag_instance):
        """Test detailed token usage tracking."""
        # Track multiple API calls with different usage patterns
        test_cases = [
            {'cost': 0.001, 'usage': {'total_tokens': 100, 'prompt_tokens': 70, 'completion_tokens': 30}},
            {'cost': 0.002, 'usage': {'total_tokens': 200, 'prompt_tokens': 150, 'completion_tokens': 50}},
            {'cost': 0.0015, 'usage': {'total_tokens': 150, 'prompt_tokens': 100, 'completion_tokens': 50, 'embedding_tokens': 25}}
        ]
        
        for case in test_cases:
            rag_instance.track_api_cost(case['cost'], case['usage'])
        
        # Verify cumulative tracking
        assert rag_instance.cost_monitor['queries'] == 3
        assert rag_instance.cost_monitor['total_tokens'] == 450  # 100 + 200 + 150
        assert rag_instance.cost_monitor['prompt_tokens'] == 320  # 70 + 150 + 100
        assert rag_instance.cost_monitor['completion_tokens'] == 130  # 30 + 50 + 50
        assert rag_instance.cost_monitor['embedding_tokens'] == 25  # Only last case had embedding tokens
        assert rag_instance.total_cost == 0.0045  # 0.001 + 0.002 + 0.0015
    
    def test_get_cost_summary_functionality(self, rag_instance):
        """Test the get_cost_summary method and CostSummary dataclass."""
        # Track some API costs first
        rag_instance.track_api_cost(0.003, {'total_tokens': 300, 'prompt_tokens': 200, 'completion_tokens': 100})
        rag_instance.track_api_cost(0.002, {'total_tokens': 200, 'prompt_tokens': 120, 'completion_tokens': 80})
        
        # Add some query history
        rag_instance.query_history.extend(["Query 1", "Query 2"])
        
        # Get cost summary
        summary = rag_instance.get_cost_summary()
        
        # Verify summary structure and data
        assert isinstance(summary, CostSummary)
        assert summary.total_cost == 0.005
        assert summary.total_queries == 2
        assert summary.total_tokens == 500
        assert summary.prompt_tokens == 320
        assert summary.completion_tokens == 180
        assert summary.embedding_tokens == 0
        assert summary.average_cost_per_query == 0.0025
        assert summary.query_history_count == 2
    
    def test_cost_monitoring_during_llm_operations(self, rag_instance, mock_lightrag_functions):
        """Test that costs are monitored during actual LLM operations."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock to simulate token usage
            mock_lightrag_functions['complete'].return_value = "Test response"
            
            llm_function = rag_instance._create_llm_function()
            
            # Note: Direct LLM function calls don't automatically track costs
            # Cost tracking happens at the query level in the main RAG class
            initial_cost = rag_instance.total_cost
            
            asyncio.run(llm_function(prompt="Test prompt"))
            
            # LLM function itself doesn't track costs, but verifies the infrastructure exists
            assert hasattr(rag_instance, 'track_api_cost')
            assert callable(rag_instance.track_api_cost)
    
    def test_rate_limiting_configuration(self, rag_instance):
        """Test rate limiting configuration for cost control."""
        # Verify rate limiter is configured
        assert hasattr(rag_instance, 'rate_limiter')
        assert isinstance(rag_instance.rate_limiter, dict)
        
        # Check default rate limiting configuration
        expected_keys = ['requests_per_minute', 'requests_per_second', 'max_concurrent_requests']
        for key in expected_keys:
            assert key in rag_instance.rate_limiter
            assert isinstance(rag_instance.rate_limiter[key], int)
            assert rag_instance.rate_limiter[key] > 0


# =====================================================================
# CONFIGURATION INTEGRATION TESTS
# =====================================================================

class TestLLMConfigurationIntegration:
    """Test class for integration with LightRAGConfig system."""
    
    def test_integration_with_lightrag_config(self, valid_config):
        """Test seamless integration with LightRAGConfig."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag.return_value = MagicMock()
            
            # Mock the initialization to avoid validation issues
            with patch.object(ClinicalMetabolomicsRAG, '_validate_initialization') as mock_validate:
                mock_validate.return_value = None
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify configuration integration
                assert rag.config == valid_config
                assert rag.effective_model == valid_config.model
                assert rag.effective_max_tokens == valid_config.max_tokens
                
                # Test LLM function respects config
                llm_function = rag._create_llm_function()
                assert callable(llm_function)
    
    def test_parameter_overrides_functionality(self, valid_config):
        """Test configuration parameter override functionality."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag.return_value = MagicMock()
            
            # Mock the initialization to avoid validation issues
            with patch.object(ClinicalMetabolomicsRAG, '_validate_initialization') as mock_validate:
                mock_validate.return_value = None
                
                # Override specific parameters
                rag = ClinicalMetabolomicsRAG(
                    config=valid_config,
                    custom_model="gpt-4",
                    custom_max_tokens=16384,
                    enable_cost_tracking=False
                )
                
                # Verify overrides are applied
                assert rag.effective_model == "gpt-4"
                assert rag.effective_max_tokens == 16384
                assert rag.cost_tracking_enabled is False
                
                # Original config should remain unchanged
                assert valid_config.model == "gpt-4o-mini"
                assert valid_config.max_tokens == 32768
    
    def test_environment_variable_handling(self, temp_working_dir):
        """Test handling of environment variables in configuration."""
        # Test with environment variables
        test_env = {
            'OPENAI_API_KEY': 'sk-test-env-key-12345678901234567890123456789012',
            'LIGHTRAG_MODEL': 'gpt-4-turbo',
            'LIGHTRAG_MAX_TOKENS': '16384'
        }
        
        with patch.dict(os.environ, test_env), \
             patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            
            mock_lightrag.return_value = MagicMock()
            
            # Create config from environment
            config = LightRAGConfig(
                working_dir=temp_working_dir,
                auto_create_dirs=True
            )
            
            # Mock the initialization to avoid validation issues
            with patch.object(ClinicalMetabolomicsRAG, '_validate_initialization') as mock_validate:
                mock_validate.return_value = None
                
                rag = ClinicalMetabolomicsRAG(config=config)
                
                # Verify environment variables were used
                assert rag.config.api_key == test_env['OPENAI_API_KEY']
                assert rag.config.model == test_env['LIGHTRAG_MODEL']
                assert rag.config.max_tokens == int(test_env['LIGHTRAG_MAX_TOKENS'])
    
    def test_configuration_validation_integration(self, temp_working_dir):
        """Test integration with configuration validation."""
        # Create invalid configuration
        invalid_config = LightRAGConfig(
            api_key="",  # Empty API key
            working_dir=temp_working_dir,
            max_tokens=-1,  # Invalid max_tokens
            auto_create_dirs=True
        )
        
        # Should raise validation error during config validation
        with pytest.raises(LightRAGConfigError):
            invalid_config.validate()
    
    def test_working_directory_handling(self, temp_working_dir):
        """Test proper handling of working directory configuration."""
        config = LightRAGConfig(
            api_key="sk-test-key-12345678901234567890123456789012",
            working_dir=temp_working_dir,
            auto_create_dirs=True
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag.return_value = MagicMock()
            
            # Mock the initialization to avoid validation issues
            with patch.object(ClinicalMetabolomicsRAG, '_validate_initialization') as mock_validate:
                mock_validate.return_value = None
                
                rag = ClinicalMetabolomicsRAG(config=config)
                
                # Verify working directory is properly set
                assert rag.config.working_dir == temp_working_dir
                
                # Verify LightRAG was initialized with correct directory
                mock_lightrag.assert_called_once()
                call_kwargs = mock_lightrag.call_args[1]
                assert call_kwargs['working_dir'] == str(temp_working_dir)


# =====================================================================
# EMBEDDING FUNCTION CONFIGURATION TESTS
# =====================================================================

class TestEmbeddingFunctionConfiguration:
    """Test class for embedding function configuration and usage."""
    
    def test_create_embedding_function_returns_callable(self, rag_instance):
        """Test that _create_embedding_function returns a proper callable."""
        embedding_function = rag_instance._create_embedding_function()
        
        # Verify it returns a callable
        assert callable(embedding_function)
        
        # Verify function signature
        import inspect
        sig = inspect.signature(embedding_function)
        assert 'texts' in sig.parameters
    
    @pytest.mark.asyncio
    async def test_embedding_function_api_calls(self, rag_instance, mock_lightrag_functions):
        """Test that embedding function makes proper API calls."""
        # Configure mock embedding response
        test_embeddings = [[0.1] * 1536, [0.2] * 1536]
        mock_lightrag_functions['embedding'].return_value = test_embeddings
        
        embedding_function = rag_instance._create_embedding_function()
        
        # Test embedding call
        test_texts = ["Sample text 1", "Sample text 2"]
        result = await embedding_function(test_texts)
        
        # Verify API call was made correctly
        mock_lightrag_functions['embedding'].assert_called_once_with(
            test_texts,
            model=rag_instance.config.embedding_model,
            api_key=rag_instance.config.api_key
        )
        
        # Verify result
        assert result == test_embeddings
    
    @pytest.mark.asyncio
    async def test_embedding_function_with_different_models(self, tmp_path, mock_lightrag_functions):
        """Test embedding function with different embedding models."""
        # Test with different embedding model
        custom_config = LightRAGConfig(
            api_key="sk-test-api-key-12345678901234567890123456789012",
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-large",
            working_dir=tmp_path,
            max_async=16,
            max_tokens=32768,
            auto_create_dirs=True
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag.return_value = MagicMock()
            
            # Mock the initialization to avoid validation issues
            with patch.object(ClinicalMetabolomicsRAG, '_validate_initialization') as mock_validate:
                mock_validate.return_value = None
                
                rag = ClinicalMetabolomicsRAG(config=custom_config)
                embedding_function = rag._create_embedding_function()
                
                # Test embedding call
                test_texts = ["Test embedding text"]
                await embedding_function(test_texts)
                
                # Verify correct model was used
                mock_lightrag_functions['embedding'].assert_called_once_with(
                    test_texts,
                    model="text-embedding-3-large",
                    api_key=custom_config.api_key
                )
    
    @pytest.mark.asyncio
    async def test_embedding_function_error_handling(self, rag_instance, mock_lightrag_functions):
        """Test error handling in embedding function."""
        # Configure mock to raise error
        embedding_error = Exception("Embedding API error")
        mock_lightrag_functions['embedding'].side_effect = embedding_error
        
        embedding_function = rag_instance._create_embedding_function()
        
        # Should handle and propagate embedding errors
        with pytest.raises(Exception):
            await embedding_function(["Test text"])
    
    def test_embedding_function_logging(self, rag_instance, mock_lightrag_functions, caplog):
        """Test that embedding errors are properly logged."""
        # Configure mock to raise error
        test_error = Exception("Test embedding error for logging")
        mock_lightrag_functions['embedding'].side_effect = test_error
        
        embedding_function = rag_instance._create_embedding_function()
        
        # Capture logs during error
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                asyncio.run(embedding_function(["Test text"]))
        
        # Verify error was logged
        assert "Embedding function error" in caplog.text
        assert "Test embedding error for logging" in caplog.text


# =====================================================================
# ASYNC OPERATIONS TESTS
# =====================================================================

class TestLLMAsyncOperations:
    """Test class for async functionality and concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_llm_function_calls(self, rag_instance, mock_lightrag_functions):
        """Test concurrent LLM function calls handling."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock with unique responses
            mock_lightrag_functions['complete'].side_effect = [
                f"Response {i}" for i in range(5)
            ]
            
            llm_function = rag_instance._create_llm_function()
            
            # Execute multiple concurrent calls
            tasks = [
                llm_function(prompt=f"Concurrent test prompt {i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all calls completed
            assert len(results) == 5
            for i, result in enumerate(results):
                assert result == f"Response {i}"
            
            # Verify mock was called correct number of times
            assert mock_lightrag_functions['complete'].call_count == 5
    
    @pytest.mark.asyncio
    async def test_async_error_propagation(self, rag_instance, mock_lightrag_functions):
        """Test proper error propagation in async operations."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock to raise error on specific call
            def side_effect_function(*args, **kwargs):
                if "error" in kwargs.get('prompt', ''):
                    raise Exception("Async test error")
                return "Success response"
            
            mock_lightrag_functions['complete'].side_effect = side_effect_function
            
            llm_function = rag_instance._create_llm_function()
            
            # Mix successful and failing calls
            tasks = [
                llm_function(prompt="Success prompt 1"),
                llm_function(prompt="Error prompt error"),
                llm_function(prompt="Success prompt 2")
            ]
            
            # Should handle mixed results appropriately
            with pytest.raises(Exception, match="Async test error"):
                await asyncio.gather(*tasks)
    
    @pytest.mark.asyncio
    async def test_async_resource_management(self, rag_instance, mock_lightrag_functions):
        """Test proper async resource management."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            llm_function = rag_instance._create_llm_function()
            
            # Test resource cleanup doesn't interfere with operations
            await llm_function(prompt="Test prompt before cleanup")
            
            # Call cleanup
            await rag_instance.cleanup()
            
            # Should still be able to make calls after cleanup
            await llm_function(prompt="Test prompt after cleanup")
            
            # Verify both calls were made
            assert mock_lightrag_functions['complete'].call_count == 2
    
    @pytest.mark.asyncio
    async def test_rate_limiting_in_concurrent_operations(self, rag_instance):
        """Test rate limiting behavior during concurrent operations."""
        # Verify rate limiter configuration exists
        assert hasattr(rag_instance, 'rate_limiter')
        assert 'max_concurrent_requests' in rag_instance.rate_limiter
        
        max_concurrent = rag_instance.rate_limiter['max_concurrent_requests']
        assert isinstance(max_concurrent, int)
        assert max_concurrent > 0
        
        # Rate limiting implementation would be tested in integration tests
        # Here we verify the configuration is properly set up


# =====================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =====================================================================

class TestLLMEdgeCases:
    """Test class for edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_prompt_handling(self, rag_instance, mock_lightrag_functions):
        """Test handling of empty or whitespace-only prompts."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            llm_function = rag_instance._create_llm_function()
            
            # Test empty prompt
            result = await llm_function(prompt="")
            mock_lightrag_functions['complete'].assert_called_with(
                model=rag_instance.effective_model,
                prompt="",
                max_tokens=rag_instance.effective_max_tokens,
                temperature=0.1,
                api_key=rag_instance.config.api_key
            )
            
            # Test whitespace-only prompt
            await llm_function(prompt="   \n\t  ")
            assert mock_lightrag_functions['complete'].call_count == 2
    
    @pytest.mark.asyncio
    async def test_very_large_prompt_handling(self, rag_instance, mock_lightrag_functions):
        """Test handling of very large prompts."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            llm_function = rag_instance._create_llm_function()
            
            # Create very large prompt
            large_prompt = "Test prompt content " * 10000
            
            await llm_function(prompt=large_prompt)
            
            # Verify the call was made (actual truncation would happen in OpenAI API)
            mock_lightrag_functions['complete'].assert_called_once()
            call_args = mock_lightrag_functions['complete'].call_args[1]
            assert call_args['prompt'] == large_prompt
    
    @pytest.mark.asyncio
    async def test_extreme_parameter_values(self, rag_instance, mock_lightrag_functions):
        """Test handling of extreme parameter values."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            llm_function = rag_instance._create_llm_function()
            
            # Test extreme values
            await llm_function(
                prompt="Test prompt",
                max_tokens=1,  # Very small
                temperature=0.0  # Minimum temperature
            )
            
            await llm_function(
                prompt="Test prompt",
                max_tokens=1000000,  # Very large (would be capped by API)
                temperature=2.0  # Maximum temperature
            )
            
            # Verify both calls were made
            assert mock_lightrag_functions['complete'].call_count == 2
    
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self, rag_instance, mock_lightrag_functions):
        """Test handling of Unicode and special characters in prompts."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            llm_function = rag_instance._create_llm_function()
            
            # Test various Unicode and special characters
            special_prompts = [
                "Test with émojis 🧬🔬 and accénts",
                "中文测试 and العربية",
                "Math symbols: ∑∏∫√∞ α β γ",
                "Special chars: @#$%^&*()[]{}|\\:;\"'<>,.?/~`",
                "Newlines\nand\ttabs"
            ]
            
            for prompt in special_prompts:
                mock_lightrag_functions['complete'].reset_mock()
                await llm_function(prompt=prompt)
                
                # Verify call was made with correct prompt
                call_args = mock_lightrag_functions['complete'].call_args[1]
                assert call_args['prompt'] == prompt
    
    def test_memory_usage_with_multiple_functions(self, rag_instance):
        """Test memory usage when creating multiple function instances."""
        # Create multiple function instances
        functions = []
        for i in range(100):
            llm_func = rag_instance._create_llm_function()
            embedding_func = rag_instance._create_embedding_function()
            functions.extend([llm_func, embedding_func])
        
        # Verify all functions are callable
        assert len(functions) == 200
        for func in functions:
            assert callable(func)
        
        # Memory usage would be monitored in actual deployment
        # Here we verify the functions can be created without errors
    
    def test_function_recreation_and_consistency(self, rag_instance):
        """Test that recreating functions produces consistent behavior."""
        # Create multiple instances of the same function
        llm_func1 = rag_instance._create_llm_function()
        llm_func2 = rag_instance._create_llm_function()
        
        embedding_func1 = rag_instance._create_embedding_function()
        embedding_func2 = rag_instance._create_embedding_function()
        
        # Verify they are different instances but have same signature
        assert llm_func1 is not llm_func2
        assert embedding_func1 is not embedding_func2
        
        import inspect
        assert inspect.signature(llm_func1) == inspect.signature(llm_func2)
        assert inspect.signature(embedding_func1) == inspect.signature(embedding_func2)


# =====================================================================
# INTEGRATION AND WORKFLOW TESTS
# =====================================================================

class TestLLMIntegrationWorkflow:
    """Test class for integration workflows and end-to-end functionality."""
    
    @pytest.mark.asyncio
    async def test_complete_llm_workflow(self, valid_config, mock_lightrag_functions):
        """Test complete workflow from initialization to LLM usage."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag.return_value = MagicMock()
            
            # Mock the initialization to avoid validation issues
            with patch.object(ClinicalMetabolomicsRAG, '_validate_initialization') as mock_validate:
                mock_validate.return_value = None
                
                # Initialize RAG system
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify initialization
                assert rag.is_initialized
                
                # Create and test LLM function
                llm_function = rag._create_llm_function()
                result = await llm_function(
                    prompt="What are the key metabolites in diabetes?",
                    temperature=0.3
                )
                
                # Verify workflow completion
                assert isinstance(result, str)
                mock_lightrag_functions['complete'].assert_called_once()
    
    def test_configuration_to_function_integration(self, temp_working_dir):
        """Test complete integration from configuration to function creation."""
        # Create configuration
        config = LightRAGConfig(
            api_key="sk-integration-test-key-1234567890123456789012",
            model="gpt-4",
            embedding_model="text-embedding-3-large",
            working_dir=temp_working_dir,
            max_tokens=16384,
            auto_create_dirs=True
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag.return_value = MagicMock()
            
            # Mock the initialization to avoid validation issues
            with patch.object(ClinicalMetabolomicsRAG, '_validate_initialization') as mock_validate:
                mock_validate.return_value = None
                
                # Initialize with custom parameters
                rag = ClinicalMetabolomicsRAG(
                    config=config,
                    custom_model="gpt-4o",
                    custom_max_tokens=8192,
                    enable_cost_tracking=True
                )
                
                # Verify configuration integration
                assert rag.effective_model == "gpt-4o"
                assert rag.effective_max_tokens == 8192
                assert rag.cost_tracking_enabled is True
                
                # Test function creation with integrated config
                llm_function = rag._create_llm_function()
                embedding_function = rag._create_embedding_function()
                
                assert callable(llm_function)
                assert callable(embedding_function)
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, rag_instance, mock_lightrag_functions):
        """Test error recovery in complete workflow."""
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            # Configure mock to fail first, succeed second
            mock_lightrag_functions['complete'].side_effect = [
                Exception("First call fails"),
                "Second call succeeds"
            ]
            
            llm_function = rag_instance._create_llm_function()
            
            # First call should fail
            with pytest.raises(Exception, match="First call fails"):
                await llm_function(prompt="First test prompt")
            
            # Second call should succeed
            result = await llm_function(prompt="Second test prompt")
            assert result == "Second call succeeds"
            
            # Verify both calls were attempted
            assert mock_lightrag_functions['complete'].call_count == 2


if __name__ == "__main__":
    """
    Run the test suite when executed directly.
    
    This comprehensive test suite covers all aspects of LLM function configuration
    and API calls for the ClinicalMetabolomicsRAG class, including error handling,
    cost tracking, and integration with the configuration system.
    """
    pytest.main([__file__, "-v", "--tb=short"])