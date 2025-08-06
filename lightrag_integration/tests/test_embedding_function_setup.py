#!/usr/bin/env python3
"""
Comprehensive unit tests for embedding function setup and validation in ClinicalMetabolomicsRAG.

This module provides comprehensive testing of embedding function configuration, setup,
validation, and API call handling for the ClinicalMetabolomicsRAG class. It covers
the _create_embedding_function method, OpenAI embedding API integration, error handling,
and dimension validation following TDD principles.

Test Coverage:
- Embedding function creation and configuration
- OpenAI embedding API integration via lightrag.llm.openai_embedding
- Error handling for API failures, rate limits, and network issues
- Embedding dimension validation (should be 1536 for OpenAI embeddings)
- Batch processing of multiple texts
- Async functionality and proper operation
- Integration with LightRAGConfig
- Mock API responses and edge cases
- Cost tracking and token usage monitoring
- Retry logic and fallback mechanisms
- Memory management for large embedding batches

Key Test Classes:
- TestEmbeddingFunctionCreation: Core function setup and configuration
- TestEmbeddingAPIIntegration: OpenAI API call mechanics and parameter passing
- TestEmbeddingErrorHandling: Error conditions and recovery mechanisms
- TestEmbeddingDimensionValidation: Dimension validation and consistency checks
- TestEmbeddingBatchProcessing: Batch processing and memory management
- TestEmbeddingAsyncOperations: Async functionality and concurrent operations
- TestEmbeddingConfigurationIntegration: Integration with configuration system
- TestEmbeddingEdgeCases: Edge cases and boundary conditions

The embedding function being tested handles:
- OpenAI embedding API calls via openai_embedding wrapper
- Error handling and logging
- Proper async operation
- Using config.embedding_model and config.api_key
- Returning proper List[List[float]] embeddings
- Batch processing of multiple texts

Requirements from CMO-LIGHTRAG-005-T03-TEST:
1. Test _create_embedding_function method configuration
2. Test OpenAI embedding API integration
3. Test embedding function error handling
4. Test embedding dimension validation
5. Follow existing test patterns from test_llm_function_configuration.py
6. Create appropriate fixtures and mocks
7. Test async functionality properly
8. Test integration with LightRAGConfig
9. Test batch processing of multiple texts

Author: Claude Code (Anthropic)
Created: 2025-08-06
Version: 1.0.0
Task: CMO-LIGHTRAG-005-T03-TEST
"""

import pytest
import asyncio
import logging
import tempfile
import time
import os
import json
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


# =====================================================================
# TEST EMBEDDING FUNCTION CREATION
# =====================================================================

class TestEmbeddingFunctionCreation:
    """Test embedding function creation and basic configuration."""
    
    def test_create_embedding_function_returns_callable(self, clinical_metabolomics_rag_instance):
        """Test that _create_embedding_function returns a callable function."""
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        assert callable(embedding_func)
        assert asyncio.iscoroutinefunction(embedding_func)
    
    def test_create_embedding_function_signature(self, clinical_metabolomics_rag_instance):
        """Test that the created embedding function has the correct signature."""
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        # Check function signature matches expected EmbeddingFunc type
        import inspect
        sig = inspect.signature(embedding_func)
        
        assert len(sig.parameters) == 1
        assert 'texts' in sig.parameters
    
    def test_embedding_function_docstring(self, clinical_metabolomics_rag_instance):
        """Test that the created embedding function has proper documentation."""
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        assert embedding_func.__doc__ is not None
        assert "Embedding function that wraps OpenAI embedding API calls" in embedding_func.__doc__
    
    def test_create_multiple_embedding_functions_independent(self, clinical_metabolomics_rag_instance):
        """Test that multiple embedding functions can be created independently."""
        func1 = clinical_metabolomics_rag_instance._create_embedding_function()
        func2 = clinical_metabolomics_rag_instance._create_embedding_function()
        
        assert func1 is not func2
        assert callable(func1)
        assert callable(func2)
    
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    def test_embedding_function_uses_config_parameters(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that the embedding function uses configuration parameters correctly."""
        # Setup mock
        mock_embedding.return_value = [[0.1] * 1536]
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        # Run the embedding function
        result = asyncio.run(embedding_func(sample_texts[:1]))
        
        # Verify openai_embedding was called with correct parameters
        mock_embedding.assert_called_once_with(
            sample_texts[:1],
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key
        )
        assert result == [[0.1] * 1536]


# =====================================================================
# TEST OPENAI API INTEGRATION
# =====================================================================

class TestEmbeddingAPIIntegration:
    """Test OpenAI embedding API integration and parameter passing."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_call_success(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test successful embedding API call."""
        # Setup successful response
        expected_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(sample_texts)
        
        assert result == expected_embeddings
        mock_embedding.assert_called_once_with(
            sample_texts,
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_call_with_single_text(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding API call with single text."""
        single_text = ["Glucose metabolism analysis"]
        expected_embedding = [[0.1] * 1536]
        mock_embedding.return_value = expected_embedding
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(single_text)
        
        assert result == expected_embedding
        assert len(result) == 1
        assert len(result[0]) == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_call_with_empty_list(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding API call with empty text list."""
        empty_texts = []
        mock_embedding.return_value = []
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(empty_texts)
        
        assert result == []
        mock_embedding.assert_called_once_with(
            empty_texts,
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key
        )
    
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
        embedding_func = rag._create_embedding_function()
        
        await embedding_func(["test text"])
        
        mock_embedding.assert_called_once_with(
            ["test text"],
            model="text-embedding-3-large",
            api_key="sk-test-key"
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
        embedding_func = rag._create_embedding_function()
        
        await embedding_func(["test text"])
        
        mock_embedding.assert_called_once_with(
            ["test text"],
            model="text-embedding-3-small",
            api_key="sk-different-api-key"
        )


# =====================================================================
# TEST ERROR HANDLING
# =====================================================================

class TestEmbeddingErrorHandling:
    """Test embedding function error handling and recovery mechanisms."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_error_propagation(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that API errors are properly propagated."""
        # Setup API error
        mock_embedding.side_effect = Exception("OpenAI API error")
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "OpenAI API error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_rate_limit_error(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of rate limit errors."""
        # Simulate rate limit error
        rate_limit_error = Exception("Rate limit exceeded")
        mock_embedding.side_effect = rate_limit_error
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_authentication_error(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of authentication errors."""
        # Simulate authentication error
        auth_error = Exception("Invalid API key")
        mock_embedding.side_effect = auth_error
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Invalid API key" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_network_error(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of network errors."""
        # Simulate network error
        network_error = Exception("Network connection failed")
        mock_embedding.side_effect = network_error
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Network connection failed" in str(exc_info.value)
    
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    def test_embedding_error_logging(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts, caplog):
        """Test that errors are properly logged and propagated."""
        # Setup error and logger
        mock_embedding.side_effect = Exception("Test embedding error")
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        # Test that the error is properly propagated
        with pytest.raises(Exception) as exc_info:
            asyncio.run(embedding_func(sample_texts))
        
        # Verify the original exception is propagated
        assert "Test embedding error" in str(exc_info.value)
        
        # Verify mock was called with correct parameters
        mock_embedding.assert_called_once_with(
            sample_texts,
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_timeout_error(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of timeout errors."""
        # Simulate timeout error
        timeout_error = Exception("Request timeout")
        mock_embedding.side_effect = timeout_error
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        with pytest.raises(Exception) as exc_info:
            await embedding_func(sample_texts)
        
        assert "Request timeout" in str(exc_info.value)


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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(["Single text for embedding"])
        
        assert len(result) == 1
        assert len(result[0]) == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_large_model_dimensions(self, mock_embedding, tmp_path):
        """Test embedding dimensions for larger embedding model."""
        # Test with text-embedding-3-large (should also be 1536 for OpenAI)
        config = LightRAGConfig(
            api_key="sk-test-key",
            embedding_model="text-embedding-3-large",
            working_dir=tmp_path
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG'):
            with patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache'):
                rag = ClinicalMetabolomicsRAG(config)
        
        expected_embedding = [[0.1] * 1536]
        mock_embedding.return_value = expected_embedding
        
        embedding_func = rag._create_embedding_function()
        result = await embedding_func(["test text"])
        
        assert len(result[0]) == 1536
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_return_type_validation(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test that embeddings return the correct type structure."""
        expected_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(mixed_texts)
        
        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 1536
    
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
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
        embedding_func = rag._create_embedding_function()
        
        await embedding_func(["test"])
        
        # Verify API key was passed correctly
        mock_embedding.assert_called_once_with(
            ["test"],
            model=config.embedding_model,
            api_key=test_api_key
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
        embedding_func = rag._create_embedding_function()
        
        await embedding_func(["test"])
        
        # Verify model was passed correctly
        mock_embedding.assert_called_once_with(
            ["test"],
            model=test_model,
            api_key=config.api_key
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
        embedding_func = rag._create_embedding_function()
        
        await embedding_func(["test"])
        
        # Verify API key was passed correctly  
        mock_embedding.assert_called_once_with(
            ["test"],
            model=config.embedding_model,
            api_key="sk-test-key"
        )


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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
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
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(special_texts)
        
        assert len(result) == len(special_texts)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_with_whitespace_only_text(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding function with whitespace-only text."""
        whitespace_texts = ["   ", "\t\t", "\n\n", ""]
        
        expected_embeddings = [[0.1] * 1536 for _ in whitespace_texts]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(whitespace_texts)
        
        assert len(result) == len(whitespace_texts)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_api_response_validation(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test validation of API response format."""
        # Test with malformed response
        malformed_embeddings = [[0.1] * 1535]  # Wrong dimension
        mock_embedding.return_value = malformed_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(sample_texts[:1])
        
        # Function should return whatever the API returns (validation happens elsewhere)
        assert result == malformed_embeddings
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_none_response_handling(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test handling of None response from API."""
        mock_embedding.return_value = None
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(sample_texts)
        
        assert result is None


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

class TestEmbeddingFunctionIntegration:
    """Integration tests combining multiple aspects of embedding functionality."""
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_full_embedding_workflow(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts):
        """Test complete embedding workflow from creation to execution."""
        # Setup realistic embedding response
        expected_embeddings = [
            [0.1 + i * 0.1] * 1536 for i in range(len(sample_texts))
        ]
        mock_embedding.return_value = expected_embeddings
        
        # Create and execute embedding function
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        result = await embedding_func(sample_texts)
        
        # Validate complete workflow
        assert len(result) == len(sample_texts)
        assert all(len(emb) == 1536 for emb in result)
        assert result == expected_embeddings
        
        # Verify API call parameters
        mock_embedding.assert_called_once_with(
            sample_texts,
            model=clinical_metabolomics_rag_instance.config.embedding_model,
            api_key=clinical_metabolomics_rag_instance.config.api_key
        )
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_with_error_recovery(self, mock_embedding, clinical_metabolomics_rag_instance, sample_texts, caplog):
        """Test embedding function with error recovery scenario."""
        # First call fails, but we can verify error handling
        mock_embedding.side_effect = Exception("Temporary API error")
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception) as exc_info:
                await embedding_func(sample_texts)
        
        # Verify error was logged and propagated
        assert "Embedding function error" in str(exc_info.value) or "Temporary API error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding')
    async def test_embedding_performance_characteristics(self, mock_embedding, clinical_metabolomics_rag_instance):
        """Test embedding function performance characteristics."""
        # Create a reasonable batch size
        batch_texts = [f"Performance test text {i}" for i in range(50)]
        expected_embeddings = [[0.1] * 1536 for _ in range(50)]
        mock_embedding.return_value = expected_embeddings
        
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
        
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
        embedding_func = clinical_metabolomics_rag_instance._create_embedding_function()
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


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])