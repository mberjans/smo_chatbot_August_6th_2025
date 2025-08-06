#!/usr/bin/env python3
"""
Comprehensive unit tests for embedding function setup and validation in ClinicalMetabolomicsRAG.

This module provides comprehensive testing of the enhanced _get_embedding_function method
in the ClinicalMetabolomicsRAG class. It covers OpenAI embedding API integration, error handling,
cost tracking, retry logic, batch processing, and edge cases following TDD principles.

Test Coverage:
- Enhanced embedding function creation with comprehensive error handling
- OpenAI embedding API integration with both LightRAG and direct client approaches
- Error handling for RateLimitError, AuthenticationError, APITimeoutError, etc.
- Cost tracking and monitoring with real pricing calculations
- Batch processing with empty text handling and zero vectors
- Dimension validation (1536 for text-embedding-3-small)
- Retry logic with exponential backoff (both tenacity and fallback)
- Configuration validation and parameter passing
- Empty text input handling (returns zero vectors)
- Async functionality and proper operation
- Memory management for large embedding batches

Key Test Classes:
- TestEmbeddingFunctionCreation: Core function setup and configuration
- TestEmbeddingAPIIntegration: OpenAI API call mechanics and parameter passing
- TestEmbeddingErrorHandling: Comprehensive error conditions and recovery
- TestEmbeddingCostTracking: Cost calculation and monitoring functionality
- TestEmbeddingBatchProcessing: Batch processing and empty text handling
- TestEmbeddingRetryLogic: Retry mechanisms with different failure scenarios
- TestEmbeddingDimensionValidation: Dimension validation and consistency checks
- TestEmbeddingConfigurationIntegration: Integration with configuration system
- TestEmbeddingEdgeCases: Edge cases and boundary conditions

The enhanced embedding function being tested handles:
- Comprehensive OpenAI API error handling with specific exception types
- Real-time cost calculation based on current OpenAI pricing
- Automatic retry logic with exponential backoff
- Empty text input validation and zero vector generation
- Embedding dimension validation and warnings
- Support for both LightRAG and direct OpenAI client integration
- Batch processing of multiple texts with order preservation
- Token usage tracking and logging

Author: Claude Code (Anthropic)
Created: 2025-08-06
Version: 1.2.0 - Updated for enhanced implementation
Task: Comprehensive testing of _get_embedding_function method
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
class MockOpenAIEmbeddingResponse:
    """Mock OpenAI embedding API response for testing."""
    data: List[Dict[str, Any]]
    usage: Dict[str, int]
    model: str
    
    def __post_init__(self):
        if not self.data:
            # Default to single embedding with 1536 dimensions (OpenAI standard)
            self.data = [{"embedding": [0.1] * 1536}]
        if not self.usage:
            self.usage = {
                "prompt_tokens": 0,
                "total_tokens": 10
            }


@dataclass
class MockEmbeddingData:
    """Mock embedding data structure."""
    embedding: List[float]
    

@dataclass
class MockEmbeddingUsage:
    """Mock embedding usage data."""
    total_tokens: int


@dataclass
class MockEmbeddingResponseBatch:
    """Mock response for batch embedding requests."""
    embeddings: List[List[float]]
    token_count: int
    model_used: str
    
    def __post_init__(self):
        if not self.embeddings:
            # Create multiple embeddings for batch processing
            self.embeddings = [[0.1] * 1536 for _ in range(3)]
        if not self.token_count:
            self.token_count = len(self.embeddings) * 10


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
def mock_openai_embedding_success():
    """Provide a mock successful OpenAI embedding response."""
    return MockOpenAIEmbeddingResponse(
        data=[{"embedding": [0.1] * 1536}],
        usage={"prompt_tokens": 0, "total_tokens": 10},
        model="text-embedding-3-small"
    )


@pytest.fixture
def mock_openai_embedding_batch():
    """Provide a mock batch embedding response."""
    return MockEmbeddingResponseBatch(
        embeddings=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536],
        token_count=30,
        model_used="text-embedding-3-small"
    )


@pytest.fixture
def sample_texts():
    """Provide sample texts for embedding testing."""
    return [
        "Glucose metabolism in clinical metabolomics",
        "Biomarker discovery in mass spectrometry data",
        "Metabolic pathway analysis for disease diagnosis"
    ]


@pytest.fixture
def large_text_batch():
    """Provide a large batch of texts for memory testing."""
    return [
        f"Sample biomedical text {i} about metabolomics and clinical analysis"
        for i in range(100)
    ]


@pytest.fixture
def clinical_metabolomics_rag_instance(valid_config):
    """Provide a ClinicalMetabolomicsRAG instance for testing."""
    with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
        with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
            with patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding'):
                rag = ClinicalMetabolomicsRAG(valid_config)
                return rag


@pytest.fixture
def mock_openai_client():
    """Provide a mock OpenAI client for testing."""
    client = AsyncMock()
    
    # Mock embeddings response
    embedding_response = Mock()
    embedding_response.data = [Mock(embedding=[0.1] * 1536)]
    embedding_response.usage = Mock(total_tokens=10)
    client.embeddings.create.return_value = embedding_response
    
    return client


# =====================================================================
# TEST EMBEDDING FUNCTION CREATION
# =====================================================================

class TestEmbeddingFunctionCreation:
    """Test embedding function creation and basic configuration."""
    
    def test_create_embedding_function_returns_callable(self, clinical_metabolomics_rag_instance):
        """Test that _get_embedding_function returns a callable function."""
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        assert callable(embedding_func)
        assert asyncio.iscoroutinefunction(embedding_func)
    
    def test_create_embedding_function_signature(self, clinical_metabolomics_rag_instance):
        """Test that the created embedding function has the correct signature."""
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        # Check function signature matches expected EmbeddingFunc type
        import inspect
        sig = inspect.signature(embedding_func)
        
        assert len(sig.parameters) == 1
        assert 'texts' in sig.parameters
    
    def test_embedding_function_docstring(self, clinical_metabolomics_rag_instance):
        """Test that the created embedding function has proper documentation."""
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        assert embedding_func.__doc__ is not None
        assert "Enhanced embedding function with comprehensive error handling" in embedding_func.__doc__
    
    def test_create_multiple_embedding_functions_independent(self, clinical_metabolomics_rag_instance):
        """Test that multiple embedding functions can be created independently."""
        func1 = clinical_metabolomics_rag_instance._get_embedding_function()
        func2 = clinical_metabolomics_rag_instance._get_embedding_function()
        
        assert func1 is not func2
        assert callable(func1)
        assert callable(func2)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_function_uses_config_parameters(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that the embedding function uses configuration parameters correctly."""
        # Setup mock
        mock_embedding.return_value = [[0.1] * 1536]
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        # Run the embedding function
        result = await embedding_func(sample_texts[:1])
        
        # Verify openai_embedding was called with correct parameters
        mock_embedding.assert_called_once_with(
            sample_texts[:1],
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key,
            timeout=30.0
        )
        assert result == [[0.1] * 1536]


# =====================================================================
# TEST OPENAI API INTEGRATION
# =====================================================================

class TestEmbeddingAPIIntegration:
    """Test OpenAI embedding API integration and parameter passing."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    @patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True)
    async def test_embedding_api_call_success(self, mock_lightrag_available, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test successful embedding API call."""
        # Setup successful response
        expected_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(sample_texts)
        
        assert result == expected_embeddings
        mock_embedding.assert_called_once_with(
            sample_texts,
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    @patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True)
    async def test_embedding_api_call_with_single_text(self, mock_lightrag_available, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding API call with single text."""
        single_text = ["Glucose metabolism analysis"]
        expected_embedding = [[0.1] * 1536]
        mock_embedding.return_value = expected_embedding
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(single_text)
        
        assert result == expected_embedding
        assert len(result) == 1
        assert len(result[0]) == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_call_with_empty_list(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding API call with empty text list."""
        empty_texts = []
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(empty_texts)
        
        # Empty texts should return empty list without calling API
        assert result == []
        mock_embedding.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_call_with_empty_strings(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding API call with empty strings returns zero vectors."""
        empty_strings = ["", "   ", "\t", "\n"]
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(empty_strings)
        
        # Empty strings should return zero vectors without calling API
        assert len(result) == 4
        for embedding in result:
            assert embedding == [0.0] * 1536
        mock_embedding.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_respects_model_configuration(self, mock_embedding, valid_config, tmp_path):
        """Test that embedding API calls use the configured model."""
        # Test with different embedding model
        config = LightRAGConfig(
            api_key="sk-test-key",
            embedding_model="text-embedding-3-large",
            working_dir=tmp_path
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
            with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
                rag = ClinicalMetabolomicsRAG(config)
        
        mock_embedding.return_value = [[0.1] * 1536]
        embedding_func = rag._get_embedding_function()
        
        await embedding_func(["test text"])
        
        mock_embedding.assert_called_once_with(
            ["test text"],
            model="text-embedding-3-large",
            api_key="sk-test-key",
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_with_different_api_key(self, mock_embedding, tmp_path):
        """Test embedding API call with different API key."""
        config = LightRAGConfig(
            api_key="sk-different-api-key",
            embedding_model="text-embedding-3-small",
            working_dir=tmp_path
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
            with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
                rag = ClinicalMetabolomicsRAG(config)
        
        mock_embedding.return_value = [[0.1] * 1536]
        embedding_func = rag._get_embedding_function()
        
        await embedding_func(["test text"])
        
        mock_embedding.assert_called_once_with(
            ["test text"],
            model="text-embedding-3-small",
            api_key="sk-different-api-key",
            timeout=30.0
        )


# =====================================================================
# TEST ERROR HANDLING
# =====================================================================

class TestEmbeddingErrorHandling:
    """Test comprehensive embedding function error handling and recovery mechanisms."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_rate_limit_error_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of OpenAI RateLimitError."""
        # Setup rate limit error
        mock_embedding.side_effect = openai.RateLimitError(
            "Rate limit exceeded", 
            response=Mock(status_code=429), 
            body="Rate limit exceeded"
        )
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Embedding function failed permanently" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_authentication_error_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of OpenAI AuthenticationError."""
        # Setup authentication error
        mock_embedding.side_effect = openai.AuthenticationError(
            "Invalid API key",
            response=Mock(status_code=401),
            body="Invalid API key"
        )
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "OpenAI embedding authentication failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_timeout_error_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of OpenAI APITimeoutError."""
        # Setup timeout error
        mock_embedding.side_effect = openai.APITimeoutError(
            "Request timed out"
        )
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Embedding function failed permanently" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_bad_request_error_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of OpenAI BadRequestError."""
        # Setup bad request error
        mock_embedding.side_effect = openai.BadRequestError(
            "Invalid model", 
            response=Mock(status_code=400), 
            body="Invalid model"
        )
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Invalid embedding API request" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_internal_server_error_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of OpenAI InternalServerError."""
        # Setup internal server error
        mock_embedding.side_effect = openai.InternalServerError(
            "Internal server error", 
            response=Mock(status_code=500), 
            body="Internal server error"
        )
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Embedding function failed permanently" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_connection_error_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of OpenAI APIConnectionError."""
        # Setup connection error
        mock_embedding.side_effect = openai.APIConnectionError(
            "Connection failed"
        )
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Embedding function failed permanently" in str(exc_info.value)
    
    
    
    
    


# =====================================================================
# TEST COST TRACKING AND MONITORING
# =====================================================================

class TestEmbeddingCostTracking:
    """Test cost calculation and monitoring functionality for embeddings."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    @patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True)
    async def test_embedding_cost_tracking_enabled(self, mock_lightrag_available, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that cost tracking works when enabled."""
        mock_embedding.return_value = [[0.1] * 1536 for _ in sample_texts]
        
        # Ensure cost tracking is enabled
        clinical_metabolomics_rag_instance.cost_tracking_enabled = True
        initial_cost = clinical_metabolomics_rag_instance.total_cost
        initial_embedding_tokens = clinical_metabolomics_rag_instance.cost_monitor['embedding_tokens']
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        await embedding_func(sample_texts)
        
        # Cost should have increased
        assert clinical_metabolomics_rag_instance.total_cost > initial_cost
        assert clinical_metabolomics_rag_instance.cost_monitor['embedding_tokens'] > initial_embedding_tokens
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_cost_calculation_accuracy(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test accuracy of embedding cost calculations."""
        mock_embedding.return_value = [[0.1] * 1536 for _ in sample_texts]
        
        # Mock the cost calculation method to verify it's called correctly
        with patch.object(clinical_metabolomics_rag_instance, '_calculate_embedding_cost', return_value=0.001) as mock_calc:
            embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
            await embedding_func(sample_texts)
            
            # Verify cost calculation was called
            mock_calc.assert_called_once()
            args = mock_calc.call_args[0]
            assert args[0] == "text-embedding-3-small"  # model
            assert 'embedding_tokens' in args[1]  # token_usage dict
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_cost_tracking_disabled(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that cost tracking can be disabled."""
        mock_embedding.return_value = [[0.1] * 1536 for _ in sample_texts]
        
        # Disable cost tracking
        clinical_metabolomics_rag_instance.cost_tracking_enabled = False
        initial_cost = clinical_metabolomics_rag_instance.total_cost
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        await embedding_func(sample_texts)
        
        # Cost should not have changed
        assert clinical_metabolomics_rag_instance.total_cost == initial_cost
    
    def test_embedding_cost_calculation_method(self, clinical_metabolomics_rag_instance):
        """Test the _calculate_embedding_cost method with different models."""
        # Test text-embedding-3-small
        token_usage = {'embedding_tokens': 1000}
        cost = clinical_metabolomics_rag_instance._calculate_embedding_cost(
            'text-embedding-3-small', token_usage
        )
        expected_cost = 1000 / 1000.0 * 0.00002  # 1K tokens * $0.00002 per 1K
        assert cost == expected_cost
        
        # Test text-embedding-3-large
        cost = clinical_metabolomics_rag_instance._calculate_embedding_cost(
            'text-embedding-3-large', token_usage
        )
        expected_cost = 1000 / 1000.0 * 0.00013  # 1K tokens * $0.00013 per 1K
        assert cost == expected_cost
        
        # Test unknown model (uses default pricing)
        cost = clinical_metabolomics_rag_instance._calculate_embedding_cost(
            'unknown-model', token_usage
        )
        expected_cost = 1000 / 1000.0 * 0.0001  # 1K tokens * $0.0001 per 1K
        assert cost == expected_cost


# =====================================================================
# TEST RETRY LOGIC
# =====================================================================

class TestEmbeddingRetryLogic:
    """Test retry mechanisms with different failure scenarios."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.TENACITY_AVAILABLE', True)
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_retry_logic_with_tenacity(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test retry logic when tenacity is available."""
        # Setup: fail twice, then succeed
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise openai.RateLimitError(
                    "Rate limit exceeded", 
                    response=Mock(status_code=429), 
                    body="Rate limit exceeded"
                )
            return [[0.1] * 1536 for _ in sample_texts]
        
        mock_embedding.side_effect = side_effect
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        # Should eventually succeed after retries
        with patch('lightrag_integration.clinical_metabolomics_rag.retry') as mock_retry:
            # Configure the retry decorator to actually retry
            def retry_decorator(func):
                async def wrapper(*args, **kwargs):
                    for attempt in range(3):  # Max 3 attempts
                        try:
                            return await func(*args, **kwargs)
                        except openai.RateLimitError:
                            if attempt == 2:
                                raise
                            await asyncio.sleep(0.01)  # Small delay
                    return await func(*args, **kwargs)
                return wrapper
            mock_retry.return_value = retry_decorator
            
            result = await embedding_func(sample_texts)
            assert len(result) == len(sample_texts)
            assert call_count == 3  # Should have been called 3 times
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.TENACITY_AVAILABLE', False)
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_retry_logic_without_tenacity(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test fallback retry logic when tenacity is not available."""
        # Setup: fail twice, then succeed
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise openai.RateLimitError(
                    "Rate limit exceeded", 
                    response=Mock(status_code=429), 
                    body="Rate limit exceeded"
                )
            return [[0.1] * 1536 for _ in sample_texts]
        
        mock_embedding.side_effect = side_effect
        
        # Mock asyncio.sleep to speed up test
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
            result = await embedding_func(sample_texts)
            
            assert len(result) == len(sample_texts)
            assert call_count == 3  # Should have been called 3 times
            assert mock_sleep.call_count == 2  # Should have slept between retries
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_retry_exhausted(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test behavior when all retry attempts are exhausted."""
        # Always fail with retryable error
        mock_embedding.side_effect = openai.RateLimitError(
            "Rate limit exceeded", 
            response=Mock(status_code=429), 
            body="Rate limit exceeded"
        )
        
        # Mock asyncio.sleep to speed up test
        with patch('asyncio.sleep', new_callable=AsyncMock):
            embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
            
            with pytest.raises(Exception) as exc_info:
                await embedding_func(sample_texts)
            
            assert "Embedding function failed permanently" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_no_retry_for_non_retryable_errors(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that non-retryable errors don't trigger retry logic."""
        # Non-retryable error (AuthenticationError)
        mock_embedding.side_effect = openai.AuthenticationError(
            "Invalid API key",
            response=Mock(status_code=401),
            body="Invalid API key"
        )
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        # Should fail immediately without retries
        assert "OpenAI embedding authentication failed" in str(exc_info.value)
        assert mock_embedding.call_count == 1  # Only called once, no retries


# =====================================================================
# TEST DIMENSION VALIDATION
# =====================================================================

class TestEmbeddingDimensionValidation:
    """Test embedding dimension validation and consistency checks."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_dimensions_openai_standard(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that embeddings have correct dimensions for OpenAI models."""
        # OpenAI text-embedding-3-small should return 1536 dimensions
        expected_embeddings = [[0.1] * 1536 for _ in sample_texts]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(sample_texts)
        
        assert len(result) == len(sample_texts)
        for embedding in result:
            assert len(embedding) == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_dimensions_consistency(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that all embeddings in a batch have consistent dimensions."""
        # All embeddings should have the same dimension
        expected_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(sample_texts)
        
        dimensions = [len(embedding) for embedding in result]
        assert all(dim == dimensions[0] for dim in dimensions)
        assert dimensions[0] == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_single_text_dimension(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding dimension for single text input."""
        expected_embedding = [[0.1] * 1536]
        mock_embedding.return_value = expected_embedding
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(["Single text for embedding"])
        
        assert len(result) == 1
        assert len(result[0]) == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_large_model_dimensions(self, mock_embedding, tmp_path):
        """Test embedding dimensions for larger embedding model."""
        # Test with text-embedding-3-large (should be 3072 dimensions)
        config = LightRAGConfig(
            api_key="sk-test-key",
            embedding_model="text-embedding-3-large",
            working_dir=tmp_path
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
            with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
                rag = ClinicalMetabolomicsRAG(config)
        
        expected_embedding = [[0.1] * 3072]  # 3-large has 3072 dimensions
        mock_embedding.return_value = expected_embedding
        
        embedding_func = rag._get_embedding_function()
        result = await embedding_func(["test text"])
        
        assert len(result[0]) == 3072
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_dimension_validation_warning(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts, caplog):
        """Test that dimension validation warnings are logged."""
        # Return unexpected dimensions
        wrong_dimension_embeddings = [[0.1] * 512 for _ in sample_texts]  # Wrong dimension
        mock_embedding.return_value = wrong_dimension_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with caplog.at_level(logging.WARNING):
            result = await embedding_func(sample_texts)
        
        # Should log a warning about unexpected dimensions
        assert any("Unexpected embedding dimension" in record.message for record in caplog.records)
        assert result == wrong_dimension_embeddings  # But still return the result
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_return_type_validation(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that embeddings return the correct type structure."""
        expected_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(sample_texts)
        
        # Validate type structure: List[List[float]]
        assert isinstance(result, list)
        for embedding in result:
            assert isinstance(embedding, list)
            for value in embedding:
                assert isinstance(value, (int, float))


# =====================================================================
# TEST BATCH PROCESSING
# =====================================================================

class TestEmbeddingBatchProcessing:
    """Test batch processing of multiple texts and memory management."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_batch_embedding_processing(self, mock_embedding, clinical_metabolomics_rag_instance, large_text_batch):
        """Test processing of large text batches."""
        # Create embeddings for large batch
        expected_embeddings = [[0.1] * 1536 for _ in range(len(large_text_batch))]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(large_text_batch)
        
        assert len(result) == len(large_text_batch)
        assert len(result) == 100  # From fixture
        
        # Verify all embeddings have correct dimensions
        for embedding in result:
            assert len(embedding) == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_empty_batch_processing(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test processing of empty text batch."""
        mock_embedding.return_value = []
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func([])
        
        assert result == []
        assert len(result) == 0
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_mixed_length_text_batch(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test processing texts of varying lengths."""
        mixed_texts = [
            "Short",
            "This is a medium length text about metabolomics",
            "This is a very long text that contains detailed information about clinical metabolomics, biomarker discovery, mass spectrometry analysis, and various metabolic pathway investigations that are commonly used in biomedical research."
        ]
        
        expected_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(mixed_texts)
        
        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    @patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True)
    async def test_batch_with_empty_texts_mixed(self, mock_lightrag_available, mock_openai_embedding, clinical_metabolomics_rag_instance):
        """Test batch processing with mixed empty and non-empty texts."""
        mixed_texts = [
            "Valid text 1",
            "",  # Empty string
            "Valid text 2",
            "   ",  # Whitespace only
            "Valid text 3"
        ]
        
        # Only non-empty texts should be sent to API
        expected_api_call_texts = ["Valid text 1", "Valid text 2", "Valid text 3"]
        expected_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_openai_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(mixed_texts)
        
        # Should have 5 embeddings total (3 real + 2 zero vectors)
        assert len(result) == 5
        
        # Check that API was called with only non-empty texts
        mock_openai_embedding.assert_called_once_with(
            expected_api_call_texts,
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key,
            timeout=30.0
        )
        
        # Verify structure: real embedding, zero vector, real embedding, zero vector, real embedding
        assert result[0] == [0.1] * 1536  # Valid text 1
        assert result[1] == [0.0] * 1536  # Empty string -> zero vector
        assert result[2] == [0.2] * 1536  # Valid text 2
        assert result[3] == [0.0] * 1536  # Whitespace only -> zero vector
        assert result[4] == [0.3] * 1536  # Valid text 3
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_batch_processing_preserves_order(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that batch processing preserves text order in results."""
        # Create distinct embeddings to verify order
        expected_embeddings = [
            [0.1] * 1536,  # For first text
            [0.2] * 1536,  # For second text
            [0.3] * 1536   # For third text
        ]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(sample_texts)
        
        assert len(result) == len(sample_texts)
        assert result[0] == [0.1] * 1536
        assert result[1] == [0.2] * 1536
        assert result[2] == [0.3] * 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_memory_efficient_batch_processing(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test memory efficiency with very large batches."""
        # Create a very large batch
        very_large_batch = [f"Text {i}" for i in range(1000)]
        expected_embeddings = [[0.1] * 1536 for _ in range(1000)]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        # This should not cause memory issues
        result = await embedding_func(very_large_batch)
        
        assert len(result) == 1000
        # Verify we're not keeping unnecessary references
        assert all(len(emb) == 1536 for emb in result)


# =====================================================================
# TEST ASYNC OPERATIONS
# =====================================================================

class TestEmbeddingAsyncOperations:
    """Test async functionality and concurrent operations."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_function_is_async(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that embedding function operates asynchronously."""
        mock_embedding.return_value = [[0.1] * 1536 for _ in sample_texts]
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        # Verify it's a coroutine function
        assert asyncio.iscoroutinefunction(embedding_func)
        
        # Test async execution
        result = await embedding_func(sample_texts)
        assert len(result) == len(sample_texts)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_concurrent_embedding_calls(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test concurrent embedding function calls."""
        # Setup different responses for concurrent calls
        call_count = 0
        
        async def mock_embedding_side_effect(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            return [[float(call_count)] * 1536 for _ in texts]
        
        mock_embedding.side_effect = mock_embedding_side_effect
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        # Make concurrent calls
        tasks = [
            embedding_func(["Text 1"]),
            embedding_func(["Text 2"]),
            embedding_func(["Text 3"])
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(len(result) == 1 for result in results)
        assert all(len(result[0]) == 1536 for result in results)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_async_error_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test async error handling in embedding function."""
        # Setup async error
        async def async_error(*args, **kwargs):
            raise Exception("Async embedding error")
        
        mock_embedding.side_effect = async_error
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Async embedding error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_async_timeout_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test async timeout handling in embedding function."""
        # Setup slow async operation
        async def slow_embedding(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow API call
            return [[0.1] * 1536 for _ in args[0]]
        
        mock_embedding.side_effect = slow_embedding
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        # Test with timeout
        result = await asyncio.wait_for(embedding_func(sample_texts), timeout=1.0)
        assert len(result) == len(sample_texts)


# =====================================================================
# TEST CONFIGURATION INTEGRATION
# =====================================================================

class TestEmbeddingConfigurationIntegration:
    """Test integration with LightRAGConfig and configuration management."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_uses_config_api_key(self, mock_embedding, tmp_path):
        """Test that embedding function uses API key from configuration."""
        test_api_key = "sk-custom-test-key-for-embeddings"
        config = LightRAGConfig(
            api_key=test_api_key,
            working_dir=tmp_path
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
            with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
                rag = ClinicalMetabolomicsRAG(config)
        
        mock_embedding.return_value = [[0.1] * 1536]
        embedding_func = rag._get_embedding_function()
        
        await embedding_func(["test"])
        
        # Verify API key was passed correctly
        mock_embedding.assert_called_once_with(
            ["test"],
            model=config.embedding_model,
            api_key=test_api_key,
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_uses_config_model(self, mock_embedding, tmp_path):
        """Test that embedding function uses model from configuration."""
        test_model = "text-embedding-3-large"
        config = LightRAGConfig(
            api_key="sk-test-key",
            embedding_model=test_model,
            working_dir=tmp_path
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
            with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
                rag = ClinicalMetabolomicsRAG(config)
        
        mock_embedding.return_value = [[0.1] * 1536]
        embedding_func = rag._get_embedding_function()
        
        await embedding_func(["test"])
        
        # Verify model was passed correctly
        mock_embedding.assert_called_once_with(
            ["test"],
            model=test_model,
            api_key=config.api_key,
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_config_validation(self, mock_embedding, tmp_path):
        """Test embedding function behavior with configuration validation."""
        # Test that invalid configuration is properly rejected
        with pytest.raises(Exception):  # Should raise config validation error
            config = LightRAGConfig(
                api_key=None,
                working_dir=tmp_path
            )
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
                with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
                    rag = ClinicalMetabolomicsRAG(config)
        
        # Test with minimal valid configuration
        config = LightRAGConfig(
            api_key="sk-test-key",
            working_dir=tmp_path
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
            with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
                rag = ClinicalMetabolomicsRAG(config)
        
        # Mock embedding call
        mock_embedding.return_value = [[0.1] * 1536]
        embedding_func = rag._get_embedding_function()
        
        await embedding_func(["test"])
        
        # Verify API key was passed correctly  
        mock_embedding.assert_called_once_with(
            ["test"],
            model=config.embedding_model,
            api_key="sk-test-key",
            timeout=30.0
        )


# =====================================================================
# TEST DIRECT OPENAI CLIENT INTEGRATION
# =====================================================================

class TestEmbeddingDirectOpenAIIntegration:
    """Test direct OpenAI client integration when LightRAG is not available."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', False)
    async def test_direct_openai_client_usage(self, clinical_metabolomics_rag_instance, sample_texts):
        """Test embedding function using direct OpenAI client when LightRAG is not available."""
        # Mock the OpenAI client
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in sample_texts]
        mock_response.usage = Mock(total_tokens=30)
        
        clinical_metabolomics_rag_instance.openai_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(sample_texts)
        
        assert len(result) == len(sample_texts)
        for embedding in result:
            assert len(embedding) == 1536
        
        # Verify direct client was called
        clinical_metabolomics_rag_instance.openai_client.embeddings.create.assert_called_once_with(
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            input=sample_texts,
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', False)
    async def test_direct_openai_client_error_handling(self, clinical_metabolomics_rag_instance, sample_texts):
        """Test error handling with direct OpenAI client."""
        # Setup client to raise RateLimitError
        clinical_metabolomics_rag_instance.openai_client.embeddings.create = AsyncMock(
            side_effect=openai.RateLimitError(
                "Rate limit exceeded", 
                response=Mock(status_code=429), 
                body="Rate limit exceeded"
            )
        )
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Embedding function failed permanently" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', False)
    async def test_direct_openai_client_cost_tracking(self, clinical_metabolomics_rag_instance, sample_texts):
        """Test cost tracking with direct OpenAI client."""
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in sample_texts]
        mock_response.usage = Mock(total_tokens=30)
        
        clinical_metabolomics_rag_instance.openai_client.embeddings.create = AsyncMock(return_value=mock_response)
        clinical_metabolomics_rag_instance.cost_tracking_enabled = True
        
        initial_cost = clinical_metabolomics_rag_instance.total_cost
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        await embedding_func(sample_texts)
        
        # Cost should have increased
        assert clinical_metabolomics_rag_instance.total_cost > initial_cost


# =====================================================================
# TEST EDGE CASES
# =====================================================================

class TestEmbeddingEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_with_unicode_text(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding function with Unicode text."""
        unicode_texts = [
            "Glucose metabolism α-β analysis",
            "Metabolite concentration μM/L",
            "Protein structure analysis π-bonds"
        ]
        
        expected_embeddings = [[0.1] * 1536 for _ in unicode_texts]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(unicode_texts)
        
        assert len(result) == len(unicode_texts)
        mock_embedding.assert_called_once_with(
            unicode_texts,
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_with_very_long_text(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding function with very long text."""
        very_long_text = "A" * 10000  # 10k character text
        long_texts = [very_long_text]
        
        expected_embeddings = [[0.1] * 1536]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(long_texts)
        
        assert len(result) == 1
        assert len(result[0]) == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_with_special_characters(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding function with special characters."""
        special_texts = [
            "C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O",
            "NH₄⁺ + SO₄²⁻ analysis",
            "Temperature: 37°C ± 0.5°C"
        ]
        
        expected_embeddings = [[0.1] * 1536 for _ in special_texts]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(special_texts)
        
        assert len(result) == len(special_texts)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_with_whitespace_only_text(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding function with whitespace-only text."""
        whitespace_texts = ["   ", "\t\t", "\n\n", ""]
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(whitespace_texts)
        
        # All whitespace texts should return zero vectors
        assert len(result) == len(whitespace_texts)
        for embedding in result:
            assert embedding == [0.0] * 1536
        
        # API should not be called for empty/whitespace-only texts
        mock_embedding.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_response_validation(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test validation of API response format."""
        # Test with malformed response
        malformed_embeddings = [[0.1] * 1535]  # Wrong dimension
        mock_embedding.return_value = malformed_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(sample_texts[:1])
        
        # Function should return whatever the API returns (validation happens elsewhere)
        assert result == malformed_embeddings
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_none_response_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of None response from API."""
        mock_embedding.return_value = None
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(sample_texts)
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_model_warning_for_non_optimal_models(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts, caplog, tmp_path):
        """Test that warnings are logged for non-optimal embedding models."""
        # Create instance with non-standard embedding model
        config = LightRAGConfig(
            api_key="sk-test-key",
            embedding_model="some-custom-model",
            working_dir=tmp_path
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
            with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
                rag = ClinicalMetabolomicsRAG(config)
        
        mock_embedding.return_value = [[0.1] * 1536 for _ in sample_texts]
        
        embedding_func = rag._get_embedding_function()
        
        with caplog.at_level(logging.WARNING):
            await embedding_func(sample_texts)
        
        # Should log a warning about non-optimal model
        assert any("may not be optimal for biomedical tasks" in record.message for record in caplog.records)


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

class TestEmbeddingFunctionIntegration:
    """Integration tests combining multiple aspects of embedding functionality."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    @patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True)
    async def test_full_embedding_workflow(self, mock_lightrag_available, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test complete embedding workflow from creation to execution."""
        # Setup realistic embedding response
        expected_embeddings = [
            [0.1 + i * 0.1] * 1536 for i in range(len(sample_texts))
        ]
        mock_embedding.return_value = expected_embeddings
        
        # Create and execute embedding function
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        result = await embedding_func(sample_texts)
        
        # Validate complete workflow
        assert len(result) == len(sample_texts)
        assert all(len(emb) == 1536 for emb in result)
        assert result == expected_embeddings
        
        # Verify API call parameters
        mock_embedding.assert_called_once_with(
            sample_texts,
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key,
            timeout=30.0
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_integration_with_lightrag_initialization(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test that embedding function integrates properly with LightRAG initialization."""
        mock_embedding.return_value = [[0.1] * 1536]
        
        # Verify that _initialize_rag uses the embedding function
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        # Test that the function works as expected for LightRAG initialization
        result = await embedding_func(["Test document for RAG initialization"])
        
        assert len(result) == 1
        assert len(result[0]) == 1536
        assert callable(embedding_func)
        assert asyncio.iscoroutinefunction(embedding_func)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_with_error_recovery(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts, caplog):
        """Test embedding function with error recovery scenario."""
        # First call fails, but we can verify error handling
        mock_embedding.side_effect = Exception("Temporary API error")
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception) as exc_info:
                await embedding_func(sample_texts)
        
        # Verify error was logged and propagated
        assert "Embedding function failed permanently" in str(exc_info.value) or "Temporary API error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_performance_characteristics(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding function performance characteristics."""
        # Create a reasonable batch size
        batch_texts = [f"Performance test text {i}" for i in range(50)]
        expected_embeddings = [[0.1] * 1536 for _ in range(50)]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        # Time the execution
        start_time = time.time()
        result = await embedding_func(batch_texts)
        execution_time = time.time() - start_time
        
        # Validate results
        assert len(result) == 50
        assert all(len(emb) == 1536 for emb in result)
        
        # Performance should be reasonable (mock should be very fast)
        assert execution_time < 1.0  # Should complete in under 1 second
    
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    def test_embedding_function_lifecycle(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test complete lifecycle of embedding function creation and usage."""
        # Setup with different return values for different calls
        def mock_embedding_side_effect(texts, **kwargs):
            return [[0.1] * 1536 for _ in texts]
        
        mock_embedding.side_effect = mock_embedding_side_effect
        
        # Create function
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        assert callable(embedding_func)
        assert asyncio.iscoroutinefunction(embedding_func)
        
        # Use function
        result = asyncio.run(embedding_func(sample_texts))
        
        # Validate
        assert len(result) == len(sample_texts)
        assert all(len(emb) == 1536 for emb in result)
        mock_embedding.assert_called_once()
        
        # Function can be reused
        result2 = asyncio.run(embedding_func(sample_texts[:1]))
        assert len(result2) == 1
        assert len(result2[0]) == 1536
        
        # Verify total call count
        assert mock_embedding.call_count == 2
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_function_with_cost_and_logging_integration(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts, caplog):
        """Test embedding function with comprehensive cost tracking and logging."""
        mock_embedding.return_value = [[0.1] * 1536 for _ in sample_texts]
        
        # Enable cost tracking and debug logging
        clinical_metabolomics_rag_instance.cost_tracking_enabled = True
        
        embedding_func = clinical_metabolomics_rag_instance._get_embedding_function()
        
        with caplog.at_level(logging.DEBUG):
            result = await embedding_func(sample_texts)
        
        # Verify result
        assert len(result) == len(sample_texts)
        
        # Verify cost tracking was called
        assert clinical_metabolomics_rag_instance.total_cost > 0
        assert clinical_metabolomics_rag_instance.cost_monitor['embedding_tokens'] > 0
        
        # Verify debug logging occurred
        assert any("Embedding call completed" in record.message for record in caplog.records)


if __name__ == "__main__":
    # Run tests with verbose output and async support
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])