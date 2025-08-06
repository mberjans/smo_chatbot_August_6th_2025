"""
Comprehensive tests for enhanced API error handling in ClinicalMetabolomicsRAG.

This test suite covers all error handling scenarios including:
- Rate limiting protection
- Circuit breaker pattern
- Retry logic with jitter
- Request queuing
- Comprehensive monitoring
- API failures (timeouts, rate limits, auth errors)
- Recovery scenarios

Author: Claude Code (Anthropic)
Created: 2025-08-06
"""

import asyncio
import pytest
import time
import unittest.mock
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import tempfile
import os

# Mock openai module before importing our classes
import sys
sys.modules['openai'] = Mock()
sys.modules['openai.RateLimitError'] = type('RateLimitError', (Exception,), {})
sys.modules['openai.APITimeoutError'] = type('APITimeoutError', (Exception,), {})
sys.modules['openai.AuthenticationError'] = type('AuthenticationError', (Exception,), {})
sys.modules['openai.BadRequestError'] = type('BadRequestError', (Exception,), {})
sys.modules['openai.InternalServerError'] = type('InternalServerError', (Exception,), {})
sys.modules['openai.APIConnectionError'] = type('APIConnectionError', (Exception,), {})

import openai

from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import (
    ClinicalMetabolomicsRAG,
    ClinicalMetabolomicsRAGError,
    CircuitBreaker,
    CircuitBreakerError,
    RateLimiter,
    RequestQueue,
    add_jitter
)


@pytest.fixture
def temp_working_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_config(temp_working_dir):
    """Create mock LightRAG configuration."""
    config = LightRAGConfig(
        working_dir=temp_working_dir,
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        max_tokens=4096,
        api_key="test-api-key"
    )
    return config


class TestCircuitBreaker:
    """Test circuit breaker pattern implementation."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes with correct parameters."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30.0
        assert cb.failure_count == 0
        assert cb.state == 'closed'
        assert cb.last_failure_time is None
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success_flow(self):
        """Test circuit breaker with successful calls."""
        cb = CircuitBreaker(failure_threshold=2)
        
        async def successful_func():
            return "success"
        
        result = await cb.call(successful_func)
        assert result == "success"
        assert cb.state == 'closed'
        assert cb.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_and_open(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=2, expected_exception=ValueError)
        
        async def failing_func():
            raise ValueError("API error")
        
        # First failure
        with pytest.raises(ValueError):
            await cb.call(failing_func)
        assert cb.state == 'closed'
        assert cb.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await cb.call(failing_func)
        assert cb.state == 'open'
        assert cb.failure_count == 2
        
        # Circuit should now be open and reject calls
        with pytest.raises(CircuitBreakerError):
            await cb.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1, 
                          expected_exception=ValueError)
        
        # Fail once to open circuit
        async def failing_func():
            raise ValueError("API error")
        
        with pytest.raises(ValueError):
            await cb.call(failing_func)
        assert cb.state == 'open'
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Next call should move to half-open
        async def successful_func():
            return "recovered"
        
        result = await cb.call(successful_func)
        assert result == "recovered"
        assert cb.state == 'closed'
        assert cb.failure_count == 0


class TestRateLimiter:
    """Test rate limiter implementation."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(max_requests=10, time_window=60.0)
        
        assert limiter.max_requests == 10
        assert limiter.time_window == 60.0
        assert limiter.tokens == 10
    
    @pytest.mark.asyncio
    async def test_rate_limiter_token_acquisition(self):
        """Test rate limiter token acquisition."""
        limiter = RateLimiter(max_requests=2, time_window=60.0)
        
        # Should acquire first token
        assert await limiter.acquire() is True
        assert limiter.tokens < 2
        
        # Should acquire second token
        assert await limiter.acquire() is True
        
        # Should not acquire third token (rate limited)
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_token_refill(self):
        """Test rate limiter token refill over time."""
        limiter = RateLimiter(max_requests=1, time_window=0.1)  # 10 tokens per second
        
        # Use up all tokens
        await limiter.acquire()
        assert await limiter.acquire() is False
        
        # Wait for refill
        await asyncio.sleep(0.2)
        
        # Should be able to acquire again
        assert await limiter.acquire() is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_wait_for_token(self):
        """Test waiting for rate limiter token."""
        limiter = RateLimiter(max_requests=1, time_window=0.1)
        
        # Use up token
        await limiter.acquire()
        
        # Wait for token should succeed after refill
        start_time = time.time()
        await limiter.wait_for_token()
        elapsed = time.time() - start_time
        
        # Should have waited for refill
        assert elapsed >= 0.1


class TestRequestQueue:
    """Test request queue implementation."""
    
    @pytest.mark.asyncio
    async def test_request_queue_initialization(self):
        """Test request queue initializes correctly."""
        queue = RequestQueue(max_concurrent=3)
        
        assert queue.max_concurrent == 3
        assert queue.active_requests == 0
    
    @pytest.mark.asyncio
    async def test_request_queue_concurrency_control(self):
        """Test request queue limits concurrent requests."""
        queue = RequestQueue(max_concurrent=2)
        
        async def slow_function():
            await asyncio.sleep(0.1)
            return "done"
        
        # Start 3 requests concurrently
        tasks = [
            asyncio.create_task(queue.execute(slow_function)),
            asyncio.create_task(queue.execute(slow_function)),
            asyncio.create_task(queue.execute(slow_function))
        ]
        
        # Wait a bit and check active count
        await asyncio.sleep(0.05)
        assert queue.active_requests <= 2
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all(r == "done" for r in results)
        assert queue.active_requests == 0


class TestJitterFunction:
    """Test jitter function for retry delays."""
    
    def test_add_jitter_basic(self):
        """Test basic jitter functionality."""
        base_time = 5.0
        jittered_time = add_jitter(base_time, jitter_factor=0.1)
        
        # Should be within jitter range
        assert 4.5 <= jittered_time <= 5.5
        assert jittered_time != base_time  # Should be different (high probability)
    
    def test_add_jitter_minimum_time(self):
        """Test jitter respects minimum wait time."""
        base_time = 0.05
        jittered_time = add_jitter(base_time, jitter_factor=1.0)
        
        # Should not go below 0.1 seconds
        assert jittered_time >= 0.1
    
    def test_add_jitter_zero_factor(self):
        """Test jitter with zero factor returns original time."""
        base_time = 5.0
        jittered_time = add_jitter(base_time, jitter_factor=0.0)
        
        assert jittered_time == base_time


class TestClinicalMetabolomicsRAGErrorHandling:
    """Test comprehensive error handling in ClinicalMetabolomicsRAG."""
    
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for testing."""
        # Mock OpenAI client
        self.mock_openai_client = Mock()
        self.mock_completion_response = Mock()
        self.mock_completion_response.choices = [Mock()]
        self.mock_completion_response.choices[0].message.content = "Test response"
        self.mock_completion_response.usage.total_tokens = 150
        self.mock_completion_response.usage.prompt_tokens = 100
        self.mock_completion_response.usage.completion_tokens = 50
        
        self.mock_embedding_response = Mock()
        self.mock_embedding_response.data = [Mock()]
        self.mock_embedding_response.data[0].embedding = [0.1] * 1536
        self.mock_embedding_response.usage.total_tokens = 100
        
        # Mock LightRAG functions
        self.mock_openai_complete = AsyncMock(return_value="Mock LLM response")
        self.mock_openai_embedding = AsyncMock(return_value=[[0.1] * 1536])
    
    @pytest.mark.asyncio
    async def test_rag_initialization_with_error_handling(self, mock_config):
        """Test RAG initialization with error handling components."""
        with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
            mock_openai.return_value = self.mock_openai_client
            
            rag = ClinicalMetabolomicsRAG(
                config=mock_config,
                circuit_breaker={'failure_threshold': 3, 'recovery_timeout': 30.0},
                rate_limiter={'requests_per_minute': 30}
            )
            
            # Check error handling components are initialized
            assert rag.rate_limiter is not None
            assert rag.request_queue is not None
            assert rag.llm_circuit_breaker is not None
            assert rag.embedding_circuit_breaker is not None
            assert rag.error_metrics is not None
            
            # Check configuration
            assert rag.llm_circuit_breaker.failure_threshold == 3
            assert rag.rate_limiter.max_requests == 30
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, mock_config):
        """Test handling of rate limit errors."""
        with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai, \
             patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache') as mock_complete:
            
            mock_openai.return_value = self.mock_openai_client
            
            # Mock rate limit error followed by success
            rate_limit_error = openai.RateLimitError("Rate limited", response=None, body=None)
            mock_complete.side_effect = [rate_limit_error, "Success response"]
            
            rag = ClinicalMetabolomicsRAG(config=mock_config)
            llm_func = rag._get_llm_function()
            
            # Should eventually succeed after retry
            response = await llm_func("test prompt")
            assert response == "Success response"
            
            # Check metrics were updated
            metrics = rag.get_error_metrics()
            assert metrics['error_counts']['rate_limit_events'] > 0
            assert metrics['error_counts']['retry_attempts'] > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_config):
        """Test circuit breaker integration with API calls."""
        with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai, \
             patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache') as mock_complete:
            
            mock_openai.return_value = self.mock_openai_client
            
            # Configure for quick circuit breaker tripping
            rag = ClinicalMetabolomicsRAG(
                config=mock_config,
                circuit_breaker={'failure_threshold': 2, 'recovery_timeout': 0.1}
            )
            
            # Mock consistent failures
            timeout_error = openai.APITimeoutError("Request timed out")
            mock_complete.side_effect = timeout_error
            
            llm_func = rag._get_llm_function()
            
            # First few calls should fail with timeout
            with pytest.raises(ClinicalMetabolomicsRAGError):
                await llm_func("test prompt")
            
            # After enough failures, should get circuit breaker error
            try:
                await llm_func("test prompt")
                assert False, "Should have raised CircuitBreakerError"
            except ClinicalMetabolomicsRAGError as e:
                assert "temporarily unavailable" in str(e)
            
            # Check circuit breaker metrics
            metrics = rag.get_error_metrics()
            assert metrics['circuit_breaker_status']['llm_circuit_state'] == 'open'
    
    @pytest.mark.asyncio
    async def test_embedding_error_handling(self, mock_config):
        """Test embedding function error handling."""
        with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai, \
             patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding') as mock_embedding:
            
            mock_openai.return_value = self.mock_openai_client
            self.mock_openai_client.embeddings.create = AsyncMock(return_value=self.mock_embedding_response)
            
            # Mock timeout error followed by success
            timeout_error = openai.APITimeoutError("Request timed out")
            mock_embedding.side_effect = [timeout_error, [[0.1] * 1536]]
            
            rag = ClinicalMetabolomicsRAG(config=mock_config)
            embedding_func = rag._get_embedding_function()
            
            # Should eventually succeed after retry
            result = await embedding_func(["test text"])
            assert len(result) == 1
            assert len(result[0]) == 1536
            
            # Check metrics were updated
            metrics = rag.get_error_metrics()
            assert metrics['error_counts']['retry_attempts'] > 0
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, mock_config):
        """Test handling of authentication errors."""
        with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai, \
             patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache') as mock_complete:
            
            mock_openai.return_value = self.mock_openai_client
            
            # Mock authentication error (should not be retried)
            auth_error = openai.AuthenticationError("Invalid API key", response=None, body=None)
            mock_complete.side_effect = auth_error
            
            rag = ClinicalMetabolomicsRAG(config=mock_config)
            llm_func = rag._get_llm_function()
            
            # Should fail immediately without retries
            with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
                await llm_func("test prompt")
            
            assert "authentication failed" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_empty_text_handling_in_embeddings(self, mock_config):
        """Test handling of empty texts in embedding function."""
        with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
            mock_openai.return_value = self.mock_openai_client
            
            rag = ClinicalMetabolomicsRAG(config=mock_config)
            embedding_func = rag._get_embedding_function()
            
            # Test with empty list
            result = await embedding_func([])
            assert result == []
            
            # Test with mix of empty and valid texts
            result = await embedding_func(["", "valid text", None, "  ", "another valid"])
            assert len(result) == 5
            # Empty texts should get zero vectors
            assert all(val == 0.0 for val in result[0])  # Empty string
            assert all(val == 0.0 for val in result[2])  # None
            assert all(val == 0.0 for val in result[3])  # Whitespace only
    
    def test_error_metrics_tracking(self, mock_config):
        """Test comprehensive error metrics tracking."""
        with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
            mock_openai.return_value = self.mock_openai_client
            
            rag = ClinicalMetabolomicsRAG(config=mock_config)
            
            # Test initial metrics
            metrics = rag.get_error_metrics()
            assert metrics['error_counts']['rate_limit_events'] == 0
            assert metrics['error_counts']['circuit_breaker_trips'] == 0
            assert metrics['api_performance']['total_calls'] == 0
            assert metrics['health_indicators']['is_healthy'] is True
            
            # Simulate some metrics updates
            rag.error_metrics['rate_limit_events'] = 3
            rag.error_metrics['circuit_breaker_trips'] = 1
            rag.error_metrics['api_call_stats']['total_calls'] = 10
            rag.error_metrics['api_call_stats']['successful_calls'] = 8
            rag.error_metrics['api_call_stats']['failed_calls'] = 2
            
            metrics = rag.get_error_metrics()
            assert metrics['error_counts']['rate_limit_events'] == 3
            assert metrics['error_counts']['circuit_breaker_trips'] == 1
            assert metrics['api_performance']['success_rate'] == 0.8
    
    def test_health_assessment(self, mock_config):
        """Test system health assessment."""
        with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
            mock_openai.return_value = self.mock_openai_client
            
            rag = ClinicalMetabolomicsRAG(config=mock_config)
            
            # Healthy system
            assert rag._assess_system_health() is True
            
            # Unhealthy due to low success rate
            rag.error_metrics['api_call_stats']['total_calls'] = 20
            rag.error_metrics['api_call_stats']['successful_calls'] = 10  # 50% success rate
            assert rag._assess_system_health() is False
            
            # Reset and test circuit breaker open
            rag.reset_error_metrics()
            rag.llm_circuit_breaker.state = 'open'
            assert rag._assess_system_health() is False
            
            # Reset and test recent rate limits
            rag.reset_error_metrics()
            rag.error_metrics['last_rate_limit'] = time.time()
            rag.error_metrics['rate_limit_events'] = 10
            assert rag._assess_system_health() is False
    
    def test_metrics_reset(self, mock_config):
        """Test error metrics reset functionality."""
        with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
            mock_openai.return_value = self.mock_openai_client
            
            rag = ClinicalMetabolomicsRAG(config=mock_config)
            
            # Set some metrics
            rag.error_metrics['rate_limit_events'] = 5
            rag.error_metrics['retry_attempts'] = 10
            rag.llm_circuit_breaker.failure_count = 3
            rag.llm_circuit_breaker.state = 'open'
            
            # Reset
            rag.reset_error_metrics()
            
            # Check everything is reset
            metrics = rag.get_error_metrics()
            assert metrics['error_counts']['rate_limit_events'] == 0
            assert metrics['error_counts']['retry_attempts'] == 0
            assert metrics['circuit_breaker_status']['llm_failure_count'] == 0
            assert metrics['circuit_breaker_status']['llm_circuit_state'] == 'closed'


@pytest.mark.asyncio
async def test_integration_scenario(mock_config):
    """Test realistic integration scenario with multiple error types."""
    with patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai, \
         patch('lightrag_integration.clinical_metabolomics_rag.openai_complete_if_cache') as mock_complete, \
         patch('lightrag_integration.clinical_metabolomics_rag.openai_embedding') as mock_embedding:
        
        mock_openai_client = Mock()
        mock_openai.return_value = mock_openai_client
        
        # Simulate realistic failure patterns
        rate_limit_error = openai.RateLimitError("Rate limited", response=None, body=None)
        timeout_error = openai.APITimeoutError("Timeout")
        success_response = "Successful response"
        
        # Pattern: rate limit, timeout, success
        mock_complete.side_effect = [rate_limit_error, timeout_error, success_response]
        mock_embedding.side_effect = [[[0.1] * 1536]]
        
        rag = ClinicalMetabolomicsRAG(config=mock_config)
        
        # Test LLM function
        llm_func = rag._get_llm_function()
        response = await llm_func("test prompt")
        assert response == success_response
        
        # Test embedding function
        embedding_func = rag._get_embedding_function()
        embeddings = await embedding_func(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
        
        # Check comprehensive metrics
        metrics = rag.get_error_metrics()
        assert metrics['error_counts']['rate_limit_events'] > 0
        assert metrics['error_counts']['retry_attempts'] > 0
        assert metrics['api_performance']['total_calls'] > 0
        assert metrics['api_performance']['successful_calls'] > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])