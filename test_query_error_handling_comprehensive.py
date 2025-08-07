#!/usr/bin/env python3
"""
Comprehensive test suite for query error handling in Clinical Metabolomics RAG system.

Tests all error scenarios and recovery mechanisms:
- Query validation errors
- Network and timeout errors  
- API rate limiting and quota errors
- LightRAG internal errors
- Response validation errors
- Retry mechanism with exponential backoff
- Error classification and context preservation

Author: Claude Code (Anthropic)
Created: 2025-08-07  
"""

import asyncio
import pytest
import time
import logging
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional

# Import the main classes and new exception types
from lightrag_integration.clinical_metabolomics_rag import (
    ClinicalMetabolomicsRAG,
    QueryError,
    QueryValidationError,
    QueryRetryableError,
    QueryNonRetryableError,
    QueryNetworkError,
    QueryAPIError,
    QueryLightRAGError,
    QueryResponseError,
    exponential_backoff_retry,
    classify_query_exception
)
from lightrag_integration.config import LightRAGConfig


class TestQueryErrorClassification:
    """Test query exception classification logic."""
    
    def test_classify_network_error(self):
        """Test classification of network-related errors."""
        # Timeout errors
        timeout_error = Exception("Connection timeout after 30 seconds")
        classified = classify_query_exception(timeout_error, query="test query", query_mode="hybrid")
        
        assert isinstance(classified, QueryNetworkError)
        assert classified.query == "test query"
        assert classified.query_mode == "hybrid"
        assert classified.timeout_seconds == 30.0
        assert classified.retry_after == 5
        
        # Connection errors
        connection_error = Exception("Connection refused by server")
        classified = classify_query_exception(connection_error)
        assert isinstance(classified, QueryNetworkError)
    
    def test_classify_api_rate_limit_error(self):
        """Test classification of API rate limiting errors."""
        rate_limit_error = Exception("Rate limit exceeded. Try again in 120 seconds. Status: 429")
        classified = classify_query_exception(rate_limit_error, query="test query")
        
        assert isinstance(classified, QueryAPIError)
        assert classified.status_code == 429
        assert classified.retry_after == 120
        assert "API rate limit" in str(classified)  # Updated to match actual error message
    
    def test_classify_auth_error(self):
        """Test classification of authentication errors (non-retryable)."""
        auth_error = Exception("Unauthorized: Invalid API key")
        classified = classify_query_exception(auth_error)
        
        assert isinstance(classified, QueryNonRetryableError)
        assert classified.error_code == 'AUTH_ERROR'
        assert "Authentication/authorization error" in str(classified)
    
    def test_classify_validation_error(self):
        """Test classification of parameter validation errors."""
        validation_error = ValueError("Invalid parameter: top_k must be positive")
        classified = classify_query_exception(validation_error)
        
        assert isinstance(classified, QueryValidationError)
        assert classified.error_code == 'VALIDATION_ERROR'
    
    def test_classify_lightrag_error(self):
        """Test classification of LightRAG-specific errors."""
        lightrag_error = Exception("Graph embedding failed during retrieval")
        classified = classify_query_exception(lightrag_error)
        
        assert isinstance(classified, QueryLightRAGError)
        assert classified.lightrag_error_type == 'graph_error'  # "graph" is checked first
        assert classified.retry_after == 10
        
        # Test a pure retrieval error
        retrieval_error = Exception("Document retrieval failed")
        classified_retrieval = classify_query_exception(retrieval_error)
        assert isinstance(classified_retrieval, QueryLightRAGError)
        assert classified_retrieval.lightrag_error_type == 'retrieval_error'
    
    def test_classify_response_error(self):
        """Test classification of response validation errors."""
        response_error = Exception("Empty response returned from API")
        classified = classify_query_exception(response_error)
        
        assert isinstance(classified, QueryResponseError)
        assert classified.error_code == 'RESPONSE_ERROR'
    
    def test_classify_unknown_error(self):
        """Test classification of unknown errors (default to retryable)."""
        unknown_error = Exception("Something unexpected happened")
        classified = classify_query_exception(unknown_error)
        
        assert isinstance(classified, QueryRetryableError)
        assert classified.retry_after == 30


class TestExponentialBackoffRetry:
    """Test exponential backoff retry mechanism."""
    
    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self):
        """Test that successful operations don't trigger retry."""
        call_count = 0
        
        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return {"result": "success"}
        
        result = await exponential_backoff_retry(
            operation=successful_operation,
            max_retries=3
        )
        
        assert result == {"result": "success"}
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        """Test retry logic for retryable errors."""
        call_count = 0
        
        async def failing_then_succeeding_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise QueryNetworkError("Network timeout", retry_after=0.1)
            return {"result": "success_after_retries"}
        
        start_time = time.time()
        result = await exponential_backoff_retry(
            operation=failing_then_succeeding_operation,
            max_retries=3,
            base_delay=0.1,
            backoff_factor=2.0,
            jitter=False
        )
        duration = time.time() - start_time
        
        assert result == {"result": "success_after_retries"}
        assert call_count == 3
        # Should have waited at least base_delay + base_delay*backoff_factor
        assert duration >= 0.1 + 0.2
    
    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Test that non-retryable errors don't trigger retry."""
        call_count = 0
        
        async def non_retryable_operation():
            nonlocal call_count
            call_count += 1
            raise QueryValidationError("Invalid parameter")
        
        with pytest.raises(QueryValidationError):
            await exponential_backoff_retry(
                operation=non_retryable_operation,
                max_retries=3,
                retryable_exceptions=(QueryRetryableError,)
            )
        
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test behavior when max retries are exhausted."""
        call_count = 0
        
        async def always_failing_operation():
            nonlocal call_count
            call_count += 1
            raise QueryNetworkError("Always fails")
        
        with pytest.raises(QueryNetworkError) as exc_info:
            await exponential_backoff_retry(
                operation=always_failing_operation,
                max_retries=2,
                base_delay=0.01,
                retryable_exceptions=(QueryNetworkError,)
            )
        
        assert call_count == 3  # Initial attempt + 2 retries
        assert "Always fails" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_retry_after_hint_respected(self):
        """Test that retry_after hint from exception is respected."""
        call_count = 0
        
        async def operation_with_retry_hint():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = QueryAPIError("Rate limited")
                error.retry_after = 0.5  # Override retry_after
                raise error
            return {"result": "success"}
        
        start_time = time.time()
        result = await exponential_backoff_retry(
            operation=operation_with_retry_hint,
            max_retries=1,
            base_delay=0.1
        )
        duration = time.time() - start_time
        
        assert result == {"result": "success"}
        assert duration >= 0.5  # Should wait at least the retry_after time


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for testing."""
    # Create proper LightRAGConfig
    config = LightRAGConfig(
        working_dir="/tmp/test_lightrag",
        model="gpt-4",
        embedding_model="text-embedding-3-small",
        max_tokens=8000,
        api_key="test-key"
    )
    
    # Create RAG instance with mocked LightRAG
    with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag_class:
            mock_lightrag_instance = AsyncMock()
            mock_lightrag_class.return_value = mock_lightrag_instance
            
            rag = ClinicalMetabolomicsRAG(config)
            rag.is_initialized = True
            rag.lightrag_instance = mock_lightrag_instance
            
            yield rag, mock_lightrag_instance


class TestQueryValidationErrors:
    """Test query validation error scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_query_validation(self, mock_rag_system):
        """Test validation of empty queries."""
        rag, _ = mock_rag_system
        
        # Test empty string
        with pytest.raises(QueryValidationError) as exc_info:
            await rag.query("")
        
        assert exc_info.value.parameter_name == "query"
        assert exc_info.value.error_code == "EMPTY_QUERY"
        
        # Test whitespace-only string
        with pytest.raises(QueryValidationError):
            await rag.query("   ")
        
        # Test None
        with pytest.raises(QueryValidationError):
            await rag.query(None)
    
    @pytest.mark.asyncio
    async def test_invalid_query_parameters(self, mock_rag_system):
        """Test validation of invalid query parameters."""
        rag, _ = mock_rag_system
        
        # Test invalid top_k
        with pytest.raises(QueryValidationError) as exc_info:
            await rag.query("test query", top_k=-1)
        
        assert "parameter validation failed" in str(exc_info.value).lower()
        assert exc_info.value.error_code == 'VALIDATION_ERROR'  # Updated to match actual error code
        
        # Test invalid mode
        with pytest.raises(QueryValidationError):
            await rag.query("test query", mode="invalid_mode")
        
        # Test invalid max_total_tokens
        with pytest.raises(QueryValidationError):
            await rag.query("test query", max_total_tokens=0)
    
    @pytest.mark.asyncio
    async def test_uninitialized_system_error(self):
        """Test error when RAG system is not initialized."""
        config = LightRAGConfig(
            working_dir="/tmp/test_lightrag",
            model="gpt-4",
            embedding_model="text-embedding-3-small",
            max_tokens=8000,
            api_key="test-key"
        )
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', True):
            rag = ClinicalMetabolomicsRAG(config)
            # Explicitly set system as NOT initialized
            rag.is_initialized = False
            
            with pytest.raises(QueryNonRetryableError) as exc_info:
                await rag.query("test query")
            
            assert exc_info.value.error_code == "NOT_INITIALIZED"
            assert "not initialized" in str(exc_info.value).lower()


class TestQueryProcessingErrors:
    """Test query processing error scenarios."""
    
    @pytest.mark.asyncio
    async def test_lightrag_network_timeout(self, mock_rag_system):
        """Test handling of LightRAG network timeouts."""
        rag, mock_lightrag = mock_rag_system
        
        # Mock network timeout
        mock_lightrag.aquery.side_effect = Exception("Connection timeout after 30 seconds")
        
        with pytest.raises(QueryNetworkError) as exc_info:
            await rag.query("test query")
        
        assert exc_info.value.timeout_seconds == 30.0
        assert exc_info.value.query == "test query"
        assert exc_info.value.query_mode == "hybrid"
    
    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self, mock_rag_system):
        """Test handling of API rate limits."""
        rag, mock_lightrag = mock_rag_system
        
        # Mock rate limit error
        mock_lightrag.aquery.side_effect = Exception("Rate limit exceeded. Status: 429. Retry after 60 seconds")
        
        with pytest.raises(QueryAPIError) as exc_info:
            await rag.query("test query")
        
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 60
        assert exc_info.value.query == "test query"
    
    @pytest.mark.asyncio
    async def test_lightrag_internal_error(self, mock_rag_system):
        """Test handling of LightRAG internal errors."""
        rag, mock_lightrag = mock_rag_system
        
        # Mock LightRAG graph error
        mock_lightrag.aquery.side_effect = Exception("Graph embedding failed during processing")
        
        with pytest.raises(QueryLightRAGError) as exc_info:
            await rag.query("test query")
        
        assert exc_info.value.lightrag_error_type == "graph_error"  # "graph" is checked first in error message
        assert exc_info.value.retry_after == 10
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self, mock_rag_system):
        """Test handling of empty/invalid responses."""
        rag, mock_lightrag = mock_rag_system
        
        # Test None response
        mock_lightrag.aquery.return_value = None
        with pytest.raises(QueryResponseError) as exc_info:
            await rag.query("test query")
        
        assert exc_info.value.error_code == "NULL_RESPONSE"
        
        # Test empty string response
        mock_lightrag.aquery.return_value = ""
        with pytest.raises(QueryResponseError) as exc_info:
            await rag.query("test query")
        
        assert exc_info.value.error_code == "EMPTY_RESPONSE"
        
        # Test whitespace-only response
        mock_lightrag.aquery.return_value = "   "
        with pytest.raises(QueryResponseError):
            await rag.query("test query")
    
    @pytest.mark.asyncio
    async def test_error_response_detection(self, mock_rag_system):
        """Test detection of error responses from LightRAG."""
        rag, mock_lightrag = mock_rag_system
        
        error_responses = [
            "An error occurred during processing",
            "Failed to retrieve documents",
            "Service unavailable at this time",
            "Internal error in graph retrieval"
        ]
        
        for error_response in error_responses:
            mock_lightrag.aquery.return_value = error_response
            with pytest.raises(QueryResponseError) as exc_info:
                await rag.query("test query")
            
            assert exc_info.value.error_code == "ERROR_RESPONSE"
            assert exc_info.value.response_content == error_response


class TestQueryRetryMechanism:
    """Test query retry mechanism."""
    
    @pytest.mark.asyncio
    async def test_query_with_retry_success_after_failures(self, mock_rag_system):
        """Test successful retry after transient failures."""
        rag, mock_lightrag = mock_rag_system
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Network timeout")  # Transient error
            return "Successful response after retries"
        
        mock_lightrag.aquery.side_effect = side_effect
        
        result = await rag.query_with_retry(
            "test query",
            max_retries=3,
            retry_config={'base_delay': 0.01, 'backoff_factor': 1.5}
        )
        
        assert result['content'] == "Successful response after retries"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_query_with_retry_non_retryable_error(self, mock_rag_system):
        """Test that non-retryable errors are not retried."""
        rag, mock_lightrag = mock_rag_system
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Unauthorized: Invalid API key")  # Non-retryable
        
        mock_lightrag.aquery.side_effect = side_effect
        
        with pytest.raises(QueryNonRetryableError):
            await rag.query_with_retry("test query", max_retries=3)
        
        assert call_count == 1  # Should not retry
    
    @pytest.mark.asyncio
    async def test_query_with_retry_max_attempts_exceeded(self, mock_rag_system):
        """Test behavior when max retry attempts are exceeded."""
        rag, mock_lightrag = mock_rag_system
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Rate limit exceeded")  # Always retryable
        
        mock_lightrag.aquery.side_effect = side_effect
        
        with pytest.raises(QueryAPIError):
            await rag.query_with_retry(
                "test query",
                max_retries=2,
                retry_config={'base_delay': 0.01}
            )
        
        assert call_count == 3  # Initial + 2 retries


class TestQueryErrorLoggingAndContext:
    """Test error logging and context preservation."""
    
    @pytest.mark.asyncio
    async def test_error_logging_with_context(self, mock_rag_system, caplog):
        """Test that errors are logged with proper context."""
        rag, mock_lightrag = mock_rag_system
        
        mock_lightrag.aquery.side_effect = Exception("Network timeout after 30 seconds")
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(QueryNetworkError):
                await rag.query("What is glucose metabolism?")
        
        # Check that error was logged with context
        error_log = caplog.records[-1]
        assert error_log.levelname == "ERROR"
        assert "QueryNetworkError" in error_log.message
        assert hasattr(error_log, 'query')
        assert hasattr(error_log, 'query_mode')
        assert hasattr(error_log, 'error_type')
        assert hasattr(error_log, 'retryable')
    
    @pytest.mark.asyncio
    async def test_cost_tracking_for_failed_queries(self, mock_rag_system):
        """Test that failed queries are tracked in cost metrics."""
        rag, mock_lightrag = mock_rag_system
        rag.cost_tracking_enabled = True
        
        # Mock the track_api_cost method to verify it's called
        rag.track_api_cost = Mock()
        
        mock_lightrag.aquery.side_effect = Exception("API quota exceeded")
        
        with pytest.raises(QueryAPIError):
            await rag.query("test query")
        
        # Verify cost tracking was called for failed query
        rag.track_api_cost.assert_called_once()
        call_args = rag.track_api_cost.call_args
        assert call_args[1]['success'] is False
        assert call_args[1]['error_type'] == 'QueryAPIError'
        assert call_args[1]['cost'] == 0.0  # Failed queries should have 0 cost


class TestQueryErrorRecovery:
    """Test error recovery scenarios."""
    
    @pytest.mark.asyncio 
    async def test_partial_success_handling(self, mock_rag_system):
        """Test handling of partial successes and graceful degradation."""
        rag, mock_lightrag = mock_rag_system
        
        # Mock response formatting failure but query success
        mock_lightrag.aquery.return_value = "Valid response content"
        
        # Mock formatter to raise exception
        if hasattr(rag, 'response_formatter') and rag.response_formatter:
            rag.response_formatter.format_response = Mock(side_effect=Exception("Formatting failed"))
        
        # Query should succeed despite formatting failure
        result = await rag.query("test query")
        
        assert result['content'] == "Valid response content"
        assert result['formatted_response'] is None  # Formatting failed gracefully
    
    @pytest.mark.asyncio
    async def test_validation_failure_graceful_handling(self, mock_rag_system):
        """Test graceful handling of response validation failures."""
        rag, mock_lightrag = mock_rag_system
        
        mock_lightrag.aquery.return_value = "Valid response content"
        
        # Mock validator to raise exception
        if hasattr(rag, 'response_validator') and rag.response_validator:
            rag.response_validator.validate_response = AsyncMock(side_effect=Exception("Validation failed"))
        
        # Query should succeed despite validation failure
        result = await rag.query("test query")
        
        assert result['content'] == "Valid response content"
        assert result['validation'] is None  # Validation failed gracefully


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])