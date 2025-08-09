"""
Comprehensive Tests for Enhanced Circuit Breaker System
======================================================

This module provides comprehensive unit and integration tests for the enhanced circuit breaker
system, covering all major components including service-specific breakers, orchestrator,
failure analyzer, and progressive degradation manager.

Test Coverage:
1. Base Enhanced Circuit Breaker functionality
2. Service-specific circuit breaker behaviors
3. Circuit Breaker Orchestrator coordination
4. Failure Correlation Analyzer pattern detection
5. Progressive Degradation Manager strategies
6. Integration layer compatibility
7. Real-world scenario testing

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Enhanced Circuit Breaker Tests
Version: 1.0.0
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging
import uuid
import statistics

# Import the enhanced circuit breaker system
from lightrag_integration.enhanced_circuit_breaker_system import (
    # Core classes
    BaseEnhancedCircuitBreaker,
    EnhancedCircuitBreakerState,
    ServiceType,
    FailureType,
    AlertLevel,
    
    # Service-specific breakers
    OpenAICircuitBreaker,
    PerplexityCircuitBreaker,
    LightRAGCircuitBreaker,
    CacheCircuitBreaker,
    
    # Orchestration and analysis
    CircuitBreakerOrchestrator,
    FailureCorrelationAnalyzer,
    ProgressiveDegradationManager,
    
    # Integration and configuration
    EnhancedCircuitBreakerIntegration,
    CircuitBreakerConfig,
    ServiceMetrics,
    FailureEvent,
    
    # Factory functions
    create_enhanced_circuit_breaker_system,
    create_service_specific_circuit_breaker
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def logger():
    """Provide a test logger."""
    return logging.getLogger("test_enhanced_circuit_breaker")


@pytest.fixture
def basic_config():
    """Provide basic circuit breaker configuration."""
    return CircuitBreakerConfig(
        service_type=ServiceType.OPENAI_API,
        failure_threshold=3,
        recovery_timeout=30.0,
        enable_adaptive_thresholds=True
    )


@pytest.fixture
def mock_operation():
    """Provide a mock operation for testing."""
    mock_op = Mock()
    mock_op.return_value = "success"
    return mock_op


@pytest.fixture
def failing_operation():
    """Provide a failing operation for testing."""
    def fail():
        raise Exception("Test failure")
    return fail


@pytest.fixture
def orchestrator(logger):
    """Provide a circuit breaker orchestrator."""
    return CircuitBreakerOrchestrator(logger)


@pytest.fixture
def openai_breaker(basic_config, logger):
    """Provide an OpenAI circuit breaker."""
    config = CircuitBreakerConfig(
        service_type=ServiceType.OPENAI_API,
        failure_threshold=3,
        recovery_timeout=30.0
    )
    return OpenAICircuitBreaker("test_openai_cb", config, logger)


@pytest.fixture
def enhanced_integration(logger):
    """Provide enhanced circuit breaker integration."""
    integration = EnhancedCircuitBreakerIntegration(logger=logger)
    integration.initialize_service_breakers()
    return integration


# ============================================================================
# Base Circuit Breaker Tests
# ============================================================================

class TestBaseEnhancedCircuitBreaker:
    """Tests for base enhanced circuit breaker functionality."""
    
    def test_initial_state(self, openai_breaker):
        """Test circuit breaker starts in CLOSED state."""
        assert openai_breaker.state == EnhancedCircuitBreakerState.CLOSED
        assert openai_breaker.metrics.consecutive_failures == 0
        assert openai_breaker.metrics.consecutive_successes == 0
    
    def test_successful_operation(self, openai_breaker, mock_operation):
        """Test successful operation execution."""
        result = openai_breaker.call(mock_operation)
        
        assert result == "success"
        assert openai_breaker.metrics.successful_requests == 1
        assert openai_breaker.metrics.consecutive_successes == 1
        assert openai_breaker.state == EnhancedCircuitBreakerState.CLOSED
    
    def test_failed_operation(self, openai_breaker, failing_operation):
        """Test failed operation handling."""
        with pytest.raises(Exception, match="Test failure"):
            openai_breaker.call(failing_operation)
        
        assert openai_breaker.metrics.failed_requests == 1
        assert openai_breaker.metrics.consecutive_failures == 1
        assert len(openai_breaker.failure_events) == 1
    
    def test_failure_threshold_opening(self, openai_breaker, failing_operation):
        """Test circuit breaker opens after failure threshold."""
        # Execute failures up to threshold
        for _ in range(3):  # failure_threshold = 3
            with pytest.raises(Exception):
                openai_breaker.call(failing_operation)
        
        # Circuit breaker should be open
        assert openai_breaker.state == EnhancedCircuitBreakerState.OPEN
    
    def test_operation_blocked_when_open(self, openai_breaker, mock_operation, failing_operation):
        """Test operations are blocked when circuit breaker is open."""
        # Force circuit breaker to open
        for _ in range(3):
            with pytest.raises(Exception):
                openai_breaker.call(failing_operation)
        
        # Now successful operations should be blocked
        from lightrag_integration.clinical_metabolomics_rag import CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            openai_breaker.call(mock_operation)
    
    def test_half_open_transition(self, openai_breaker, failing_operation):
        """Test transition from OPEN to HALF_OPEN after timeout."""
        # Open the circuit breaker
        for _ in range(3):
            with pytest.raises(Exception):
                openai_breaker.call(failing_operation)
        
        assert openai_breaker.state == EnhancedCircuitBreakerState.OPEN
        
        # Simulate timeout passage
        openai_breaker.last_failure_time = time.time() - 31  # recovery_timeout = 30
        openai_breaker._update_state()
        
        assert openai_breaker.state == EnhancedCircuitBreakerState.HALF_OPEN
    
    def test_recovery_to_closed(self, openai_breaker, mock_operation, failing_operation):
        """Test recovery from HALF_OPEN to CLOSED."""
        # Open the circuit breaker
        for _ in range(3):
            with pytest.raises(Exception):
                openai_breaker.call(failing_operation)
        
        # Force to HALF_OPEN
        openai_breaker.force_state(EnhancedCircuitBreakerState.HALF_OPEN)
        
        # Successful operations should allow recovery
        for _ in range(3):  # half_open_max_calls = 3
            openai_breaker.call(mock_operation)
        
        assert openai_breaker.state == EnhancedCircuitBreakerState.CLOSED
    
    def test_adaptive_threshold_adjustment(self, basic_config, logger):
        """Test adaptive threshold adjustment based on performance."""
        config = basic_config
        config.enable_adaptive_thresholds = True
        config.min_failure_threshold = 2
        config.max_failure_threshold = 10
        
        breaker = OpenAICircuitBreaker("adaptive_test", config, logger)
        
        # Simulate low error rate scenario
        for _ in range(20):  # Successful operations
            breaker._record_success(0.1)
        
        # Update metrics
        breaker.metrics.total_requests = 20
        breaker.metrics.successful_requests = 20
        breaker.metrics.failed_requests = 0
        
        # Force threshold adjustment check
        breaker.last_threshold_adjustment = time.time() - 301  # Force adjustment
        breaker._adjust_adaptive_thresholds()
        
        # Threshold should be lowered for good performance
        assert breaker.current_failure_threshold <= config.failure_threshold
    
    def test_status_reporting(self, openai_breaker):
        """Test comprehensive status reporting."""
        status = openai_breaker.get_status()
        
        assert 'name' in status
        assert 'service_type' in status
        assert 'state' in status
        assert 'metrics' in status
        assert 'config' in status
        assert 'timestamp' in status
        
        assert status['state'] == EnhancedCircuitBreakerState.CLOSED.value
        assert status['service_type'] == ServiceType.OPENAI_API.value


# ============================================================================
# Service-Specific Circuit Breaker Tests
# ============================================================================

class TestOpenAICircuitBreaker:
    """Tests for OpenAI-specific circuit breaker functionality."""
    
    def test_model_health_tracking(self, openai_breaker):
        """Test model-specific health tracking."""
        # Update health for different models
        openai_breaker.update_model_health("gpt-4o", True, {
            'response_time': 1.5,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
        })
        
        openai_breaker.update_model_health("gpt-4o-mini", False)
        
        # Check model health data
        assert "gpt-4o" in openai_breaker.model_health
        assert "gpt-4o-mini" in openai_breaker.model_health
        
        assert openai_breaker.model_health["gpt-4o"]['consecutive_successes'] == 1
        assert openai_breaker.model_health["gpt-4o-mini"]['consecutive_failures'] == 1
        
        # Check token usage tracking
        assert openai_breaker.token_usage_stats['input_tokens'] == 100
        assert openai_breaker.token_usage_stats['output_tokens'] == 50
    
    def test_rate_limit_status_update(self, openai_breaker):
        """Test rate limit status tracking."""
        headers = {
            'x-ratelimit-remaining-requests': '50',
            'x-ratelimit-limit-requests': '1000',
            'x-ratelimit-remaining-tokens': '5000',
            'x-ratelimit-limit-tokens': '100000'
        }
        
        openai_breaker.update_rate_limit_status(headers)
        
        assert openai_breaker.rate_limit_status['requests_remaining'] == 50
        assert openai_breaker.rate_limit_status['requests_limit'] == 1000
        assert openai_breaker.metrics.rate_limit_remaining == 5.0  # 50/1000 * 100
    
    def test_service_health_check(self, openai_breaker):
        """Test OpenAI service health checking logic."""
        # Initially healthy
        assert openai_breaker._check_service_health() == True
        
        # Simulate low rate limit
        openai_breaker.rate_limit_status = {'requests_remaining': 5}
        assert openai_breaker._check_service_health() == False
        
        # Reset and test high token usage
        openai_breaker.rate_limit_status = {'requests_remaining': 100}
        openai_breaker.token_usage_stats = {'total_tokens': 150000}
        assert openai_breaker._check_service_health() == False


class TestPerplexityCircuitBreaker:
    """Tests for Perplexity-specific circuit breaker functionality."""
    
    def test_query_complexity_tracking(self, logger):
        """Test query complexity tracking."""
        config = CircuitBreakerConfig(service_type=ServiceType.PERPLEXITY_API)
        breaker = PerplexityCircuitBreaker("test_perplexity", config, logger)
        
        # Add complexity data
        breaker.update_query_complexity("research", 0.7)
        breaker.update_query_complexity("research", 0.8)
        breaker.update_query_complexity("simple", 0.3)
        
        assert len(breaker.query_complexity_stats["research"]) == 2
        assert len(breaker.query_complexity_stats["simple"]) == 1
        
        # Test health check with high complexity
        breaker.query_complexity_stats["research"] = [0.9] * 20
        assert breaker._check_service_health() == False
    
    def test_search_quality_tracking(self, logger):
        """Test search quality tracking."""
        config = CircuitBreakerConfig(service_type=ServiceType.PERPLEXITY_API)
        breaker = PerplexityCircuitBreaker("test_perplexity", config, logger)
        
        # Add quality scores
        for i in range(10):
            breaker.update_search_quality(f"query_{i}", 0.8)
        
        assert len(breaker.search_quality_metrics) == 10
        
        # Test health check with low quality
        for i in range(10):
            breaker.update_search_quality(f"low_query_{i}", 0.5)
        
        assert breaker._check_service_health() == False
    
    def test_quota_management(self, logger):
        """Test API quota management."""
        config = CircuitBreakerConfig(service_type=ServiceType.PERPLEXITY_API)
        breaker = PerplexityCircuitBreaker("test_perplexity", config, logger)
        
        quota_info = {
            'requests_used': 900,
            'requests_limit': 1000,
            'percentage_used': 90.0
        }
        
        breaker.update_quota_status(quota_info)
        
        assert breaker.quota_usage['percentage_used'] == 90.0
        assert breaker.metrics.quota_usage_percentage == 90.0
        
        # High quota usage should fail health check
        assert breaker._check_service_health() == False


class TestLightRAGCircuitBreaker:
    """Tests for LightRAG-specific circuit breaker functionality."""
    
    def test_knowledge_base_health_monitoring(self, logger):
        """Test knowledge base health monitoring."""
        config = CircuitBreakerConfig(service_type=ServiceType.LIGHTRAG)
        breaker = LightRAGCircuitBreaker("test_lightrag", config, logger)
        
        # Update knowledge base health
        health_info = {
            'index_accessible': True,
            'document_count': 1000,
            'index_size_mb': 500
        }
        
        breaker.update_knowledge_base_health(health_info)
        
        assert breaker.knowledge_base_health['index_accessible'] == True
        assert breaker.knowledge_base_health['document_count'] == 1000
        
        # Test health check with inaccessible index
        breaker.knowledge_base_health['index_accessible'] = False
        assert breaker._check_service_health() == False
    
    def test_retrieval_quality_tracking(self, logger):
        """Test retrieval quality tracking."""
        config = CircuitBreakerConfig(service_type=ServiceType.LIGHTRAG)
        breaker = LightRAGCircuitBreaker("test_lightrag", config, logger)
        
        # Add quality scores
        for score in [0.8, 0.9, 0.7, 0.85]:
            breaker.update_retrieval_quality(score)
        
        assert len(breaker.retrieval_quality_scores) == 4
        
        # Test health check with low quality
        for _ in range(20):
            breaker.update_retrieval_quality(0.4)  # Below 0.6 threshold
        
        assert breaker._check_service_health() == False
    
    def test_embedding_service_monitoring(self, logger):
        """Test embedding service health monitoring."""
        config = CircuitBreakerConfig(service_type=ServiceType.LIGHTRAG)
        breaker = LightRAGCircuitBreaker("test_lightrag", config, logger)
        
        # Record successful embedding operations
        breaker.update_embedding_service_status(True, 0.5)
        breaker.update_embedding_service_status(True, 0.6)
        
        assert breaker.embedding_service_status['consecutive_failures'] == 0
        assert breaker.embedding_service_status['successful_requests'] == 2
        
        # Record failures
        for _ in range(5):
            breaker.update_embedding_service_status(False)
        
        # Should fail health check
        assert breaker._check_service_health() == False


class TestCacheCircuitBreaker:
    """Tests for Cache-specific circuit breaker functionality."""
    
    def test_cache_operation_tracking(self, logger):
        """Test cache operation statistics tracking."""
        config = CircuitBreakerConfig(service_type=ServiceType.CACHE)
        breaker = CacheCircuitBreaker("test_cache", config, logger)
        
        # Record cache operations
        breaker.record_cache_operation('hits', True)
        breaker.record_cache_operation('hits', True)
        breaker.record_cache_operation('misses', True)
        
        assert breaker.cache_stats['hits'] == 2
        assert breaker.cache_stats['misses'] == 1
        assert breaker.metrics.cache_hit_rate == 2/3  # 2 hits out of 3 total
    
    def test_memory_pressure_monitoring(self, logger):
        """Test memory pressure alert tracking."""
        config = CircuitBreakerConfig(service_type=ServiceType.CACHE)
        breaker = CacheCircuitBreaker("test_cache", config, logger)
        
        # Record memory pressure alerts
        breaker.record_memory_pressure_alert(3000, 'medium')
        breaker.record_memory_pressure_alert(4500, 'high')
        
        assert len(breaker.memory_pressure_alerts) == 2
        assert breaker.metrics.memory_usage_mb == 4500
        
        # High memory should fail health check
        assert breaker._check_service_health() == False
    
    def test_storage_backend_health(self, logger):
        """Test storage backend health monitoring."""
        config = CircuitBreakerConfig(service_type=ServiceType.CACHE)
        breaker = CacheCircuitBreaker("test_cache", config, logger)
        
        # Record successful operations
        breaker.update_storage_backend_health(True, 'read')
        breaker.update_storage_backend_health(True, 'write')
        
        assert breaker.storage_backend_health['consecutive_failures'] == 0
        
        # Record failures
        for _ in range(6):  # Above threshold
            breaker.update_storage_backend_health(False, 'read')
        
        # Should fail health check
        assert breaker._check_service_health() == False


# ============================================================================
# Orchestrator Tests
# ============================================================================

class TestCircuitBreakerOrchestrator:
    """Tests for circuit breaker orchestrator functionality."""
    
    def test_circuit_breaker_registration(self, orchestrator, openai_breaker):
        """Test circuit breaker registration."""
        orchestrator.register_circuit_breaker(openai_breaker)
        
        assert openai_breaker.name in orchestrator._circuit_breakers
        assert orchestrator._circuit_breakers[openai_breaker.name] == openai_breaker
    
    def test_state_change_handling(self, orchestrator, openai_breaker, logger):
        """Test state change event handling."""
        orchestrator.register_circuit_breaker(openai_breaker)
        
        # Trigger state change
        openai_breaker.force_state(EnhancedCircuitBreakerState.OPEN, "Test")
        
        # Check that alert was created
        assert len(orchestrator._system_alerts) > 0
        alert = orchestrator._system_alerts[-1]
        assert alert['breaker_name'] == openai_breaker.name
        assert alert['new_state'] == EnhancedCircuitBreakerState.OPEN.value
    
    def test_coordination_rules(self, orchestrator, logger):
        """Test coordination rule application."""
        # Create multiple circuit breakers
        breakers = []
        for i in range(3):
            config = CircuitBreakerConfig(
                service_type=ServiceType.OPENAI_API,
                failure_threshold=2
            )
            breaker = OpenAICircuitBreaker(f"test_breaker_{i}", config, logger)
            orchestrator.register_circuit_breaker(breaker)
            breakers.append(breaker)
        
        # Force multiple breakers to fail
        breakers[0].force_state(EnhancedCircuitBreakerState.OPEN, "Test failure 1")
        breakers[1].force_state(EnhancedCircuitBreakerState.OPEN, "Test failure 2")
        
        # System state should be critical
        status = orchestrator.get_system_status()
        assert status['system_state'] == 'critical'
    
    def test_dependency_handling(self, orchestrator, logger):
        """Test service dependency handling."""
        # Create OpenAI breaker
        openai_config = CircuitBreakerConfig(service_type=ServiceType.OPENAI_API)
        openai_breaker = OpenAICircuitBreaker("openai_test", openai_config, logger)
        
        # Create LightRAG breaker with OpenAI dependency
        lightrag_config = CircuitBreakerConfig(service_type=ServiceType.LIGHTRAG)
        lightrag_breaker = LightRAGCircuitBreaker("lightrag_test", lightrag_config, logger)
        
        orchestrator.register_circuit_breaker(openai_breaker)
        orchestrator.register_circuit_breaker(
            lightrag_breaker, 
            dependencies=[ServiceType.OPENAI_API]
        )
        
        # Fail OpenAI service
        openai_breaker.force_state(EnhancedCircuitBreakerState.OPEN, "OpenAI failure")
        
        # LightRAG should be preemptively degraded
        assert lightrag_breaker.state == EnhancedCircuitBreakerState.DEGRADED
    
    def test_system_recovery(self, orchestrator, openai_breaker):
        """Test system-wide recovery."""
        orchestrator.register_circuit_breaker(openai_breaker)
        
        # Force breaker to open
        openai_breaker.force_state(EnhancedCircuitBreakerState.OPEN, "Test")
        
        # Force system recovery
        orchestrator.force_system_recovery("Test recovery")
        
        assert openai_breaker.state == EnhancedCircuitBreakerState.CLOSED
        assert orchestrator._system_state == "healthy"


# ============================================================================
# Failure Analyzer Tests
# ============================================================================

class TestFailureCorrelationAnalyzer:
    """Tests for failure correlation analyzer."""
    
    def test_failure_pattern_analysis(self, orchestrator, logger):
        """Test failure pattern detection."""
        analyzer = FailureCorrelationAnalyzer(orchestrator, logger)
        
        # Create circuit breakers with failure events
        openai_breaker = create_service_specific_circuit_breaker(
            ServiceType.OPENAI_API, logger=logger
        )
        perplexity_breaker = create_service_specific_circuit_breaker(
            ServiceType.PERPLEXITY_API, logger=logger
        )
        
        orchestrator.register_circuit_breaker(openai_breaker)
        orchestrator.register_circuit_breaker(perplexity_breaker)
        
        # Create correlated failure events
        now = time.time()
        for i in range(5):
            # OpenAI failures
            failure_event = FailureEvent(
                service_type=ServiceType.OPENAI_API,
                failure_type=FailureType.RATE_LIMIT,
                timestamp=now + i * 60  # 1 minute apart
            )
            openai_breaker.failure_events.append(failure_event)
            
            # Perplexity failures shortly after
            failure_event = FailureEvent(
                service_type=ServiceType.PERPLEXITY_API,
                failure_type=FailureType.SERVICE_UNAVAILABLE,
                timestamp=now + i * 60 + 30  # 30 seconds after OpenAI
            )
            perplexity_breaker.failure_events.append(failure_event)
        
        # Analyze patterns
        analysis = analyzer.analyze_failure_patterns()
        
        assert 'correlations_detected' in analysis
        assert 'failure_patterns' in analysis
        assert 'recommendations' in analysis
        
        # Should detect correlation between services
        assert len(analysis['correlations_detected']) > 0
    
    def test_burst_failure_detection(self, orchestrator, logger):
        """Test burst failure pattern detection."""
        analyzer = FailureCorrelationAnalyzer(orchestrator, logger)
        
        # Create circuit breaker
        breaker = create_service_specific_circuit_breaker(
            ServiceType.OPENAI_API, logger=logger
        )
        orchestrator.register_circuit_breaker(breaker)
        
        # Create burst of failures
        now = time.time()
        for i in range(8):  # 8 failures in short time
            failure_event = FailureEvent(
                service_type=ServiceType.OPENAI_API,
                failure_type=FailureType.TIMEOUT,
                timestamp=now + i * 30  # 30 seconds apart
            )
            breaker.failure_events.append(failure_event)
        
        # Analyze patterns
        analysis = analyzer.analyze_failure_patterns()
        
        # Should detect burst pattern
        burst_patterns = analysis['failure_patterns']['burst_failures']
        assert len(burst_patterns) > 0
        assert burst_patterns[0]['failure_count'] >= 5
    
    def test_recommendation_generation(self, orchestrator, logger):
        """Test failure analysis recommendation generation."""
        analyzer = FailureCorrelationAnalyzer(orchestrator, logger)
        
        # Create multiple breakers
        for service_type in [ServiceType.OPENAI_API, ServiceType.PERPLEXITY_API]:
            breaker = create_service_specific_circuit_breaker(service_type, logger=logger)
            orchestrator.register_circuit_breaker(breaker)
        
        # Create high correlation scenario
        correlations = [{
            'service_a': 'openai_api',
            'service_b': 'perplexity_api',
            'correlation_strength': 0.9,
            'failure_count_a': 10,
            'failure_count_b': 8
        }]
        
        recommendations = analyzer._generate_recommendations(correlations, {}, [])
        
        assert len(recommendations) > 0
        assert any('dependency_review' == rec['type'] for rec in recommendations)


# ============================================================================
# Degradation Manager Tests
# ============================================================================

class TestProgressiveDegradationManager:
    """Tests for progressive degradation manager."""
    
    def test_degradation_strategy_initialization(self, orchestrator, logger):
        """Test degradation strategy initialization."""
        manager = ProgressiveDegradationManager(orchestrator, logger)
        
        # Check that strategies are initialized for all service types
        assert ServiceType.OPENAI_API in manager.degradation_strategies
        assert ServiceType.PERPLEXITY_API in manager.degradation_strategies
        assert ServiceType.LIGHTRAG in manager.degradation_strategies
        assert ServiceType.CACHE in manager.degradation_strategies
        
        # Check strategy levels
        openai_strategy = manager.degradation_strategies[ServiceType.OPENAI_API]
        assert 'level_1' in openai_strategy
        assert 'level_2' in openai_strategy
        assert 'level_3' in openai_strategy
    
    def test_apply_degradation(self, orchestrator, logger):
        """Test applying progressive degradation."""
        manager = ProgressiveDegradationManager(orchestrator, logger)
        
        # Create and register circuit breaker
        breaker = create_service_specific_circuit_breaker(
            ServiceType.OPENAI_API, logger=logger
        )
        orchestrator.register_circuit_breaker(breaker)
        
        # Apply degradation
        result = manager.apply_degradation(
            target_services=[ServiceType.OPENAI_API],
            degradation_level=1,
            reason="Test degradation"
        )
        
        assert result['affected_services'] == [ServiceType.OPENAI_API.value]
        assert result['performance_impact'] > 0
        assert result['cost_savings'] > 0
        assert len(result['applied_degradations']) == 1
        
        # Breaker should be in degraded state
        assert breaker.state == EnhancedCircuitBreakerState.DEGRADED
    
    def test_degradation_action_execution(self, orchestrator, logger):
        """Test specific degradation action execution."""
        manager = ProgressiveDegradationManager(orchestrator, logger)
        
        # Create OpenAI circuit breaker
        openai_breaker = create_service_specific_circuit_breaker(
            ServiceType.OPENAI_API, logger=logger
        )
        orchestrator.register_circuit_breaker(openai_breaker)
        
        # Execute specific degradation actions
        success = manager._execute_degradation_action(
            ServiceType.OPENAI_API, 'switch_to_smaller_model'
        )
        
        assert success == True
        # Check that configuration was updated
        config = openai_breaker.config.service_specific_config
        assert config.get('preferred_model') == 'gpt-4o-mini'
        assert config.get('max_tokens') == 2000
    
    def test_recovery_from_degradation(self, orchestrator, logger):
        """Test recovery from degradation."""
        manager = ProgressiveDegradationManager(orchestrator, logger)
        
        # Create and register circuit breaker
        breaker = create_service_specific_circuit_breaker(
            ServiceType.OPENAI_API, logger=logger
        )
        orchestrator.register_circuit_breaker(breaker)
        
        # Apply degradation first
        manager.apply_degradation(
            target_services=[ServiceType.OPENAI_API],
            degradation_level=2
        )
        
        # Recover from degradation
        result = manager.recover_from_degradation(
            target_services=[ServiceType.OPENAI_API],
            recovery_level=2
        )
        
        assert len(result['recovered_services']) == 1
        assert result['recovered_services'][0]['success'] == True
        
        # Breaker should be in half-open for testing
        assert breaker.state == EnhancedCircuitBreakerState.HALF_OPEN
        
        # Configuration should be reset
        config = breaker.config.service_specific_config
        assert config.get('max_tokens') == 4000
        assert 'preferred_model' not in config
    
    def test_degradation_status_reporting(self, orchestrator, logger):
        """Test degradation status reporting."""
        manager = ProgressiveDegradationManager(orchestrator, logger)
        
        # Apply some degradation
        breaker = create_service_specific_circuit_breaker(
            ServiceType.OPENAI_API, logger=logger
        )
        orchestrator.register_circuit_breaker(breaker)
        
        manager.apply_degradation([ServiceType.OPENAI_API], 1)
        
        # Get status
        status = manager.get_degradation_status()
        
        assert status['current_degradation_level'] == 1
        assert ServiceType.OPENAI_API.value in status['degraded_services']
        assert len(status['recent_degradations']) > 0
        assert 'available_strategies' in status


# ============================================================================
# Integration Tests
# ============================================================================

class TestEnhancedCircuitBreakerIntegration:
    """Tests for enhanced circuit breaker integration."""
    
    def test_integration_initialization(self, enhanced_integration):
        """Test integration layer initialization."""
        assert enhanced_integration.integration_active == True
        assert len(enhanced_integration.service_breakers) > 0
        assert 'openai' in enhanced_integration.service_breakers
        assert 'perplexity' in enhanced_integration.service_breakers
        assert 'lightrag' in enhanced_integration.service_breakers
        assert 'cache' in enhanced_integration.service_breakers
    
    def test_execute_with_enhanced_protection(self, enhanced_integration, mock_operation):
        """Test operation execution with enhanced protection."""
        result = enhanced_integration.execute_with_enhanced_protection(
            'openai', mock_operation
        )
        
        assert result == "success"
        assert enhanced_integration.integration_metrics['successful_calls'] == 1
    
    def test_failure_handling_with_degradation(self, enhanced_integration, failing_operation):
        """Test failure handling with automatic degradation."""
        # Execute multiple failures to trigger degradation
        for _ in range(5):
            try:
                enhanced_integration.execute_with_enhanced_protection(
                    'openai', failing_operation
                )
            except Exception:
                pass  # Expected failures
        
        # Check that system took protective measures
        status = enhanced_integration.get_comprehensive_status()
        assert 'degradation_status' in status
        
        # At least one service should be degraded
        degradation_status = status['degradation_status']
        assert len(degradation_status['degraded_services']) > 0
    
    def test_service_metric_updates(self, enhanced_integration):
        """Test service-specific metric updates."""
        # Create mock operation with response info
        def mock_openai_operation():
            result = Mock()
            result.response_info = {
                'model': 'gpt-4o',
                'response_time': 1.2,
                'usage': {
                    'prompt_tokens': 100,
                    'completion_tokens': 50,
                    'total_tokens': 150
                },
                'headers': {
                    'x-ratelimit-remaining-requests': '900',
                    'x-ratelimit-limit-requests': '1000'
                }
            }
            return result
        
        result = enhanced_integration.execute_with_enhanced_protection(
            'openai', mock_openai_operation
        )
        
        # Check that OpenAI-specific metrics were updated
        openai_breaker = enhanced_integration.service_breakers['openai']
        assert 'gpt-4o' in openai_breaker.model_health
        assert openai_breaker.token_usage_stats['total_tokens'] > 0
        assert openai_breaker.rate_limit_status['requests_remaining'] == 900
    
    def test_comprehensive_status_reporting(self, enhanced_integration):
        """Test comprehensive system status reporting."""
        status = enhanced_integration.get_comprehensive_status()
        
        required_keys = [
            'integration_active',
            'integration_metrics',
            'orchestrator_status',
            'degradation_status',
            'failure_analysis',
            'service_breakers',
            'timestamp'
        ]
        
        for key in required_keys:
            assert key in status
        
        # Check service breaker statuses
        assert len(status['service_breakers']) > 0
        for service_name, service_status in status['service_breakers'].items():
            assert 'name' in service_status
            assert 'state' in service_status
            assert 'metrics' in service_status
    
    def test_system_shutdown(self, enhanced_integration):
        """Test graceful system shutdown."""
        # Ensure system is active
        assert enhanced_integration.integration_active == True
        
        # Shutdown
        enhanced_integration.shutdown()
        
        assert enhanced_integration.integration_active == False
        
        # All breakers should be reset
        for breaker in enhanced_integration.service_breakers.values():
            assert breaker.state == EnhancedCircuitBreakerState.CLOSED
            assert breaker.metrics.consecutive_failures == 0


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_enhanced_circuit_breaker_system(self, logger):
        """Test system factory function."""
        integration = create_enhanced_circuit_breaker_system(logger=logger)
        
        assert isinstance(integration, EnhancedCircuitBreakerIntegration)
        assert integration.integration_active == True
        assert len(integration.service_breakers) > 0
    
    def test_create_service_specific_circuit_breaker(self, logger):
        """Test service-specific breaker factory."""
        # Test all service types
        service_types = [
            ServiceType.OPENAI_API,
            ServiceType.PERPLEXITY_API,
            ServiceType.LIGHTRAG,
            ServiceType.CACHE
        ]
        
        for service_type in service_types:
            breaker = create_service_specific_circuit_breaker(
                service_type=service_type,
                logger=logger
            )
            
            assert breaker.config.service_type == service_type
            assert breaker.state == EnhancedCircuitBreakerState.CLOSED
    
    def test_create_with_custom_config(self, logger):
        """Test factory with custom service configuration."""
        custom_config = {
            'openai': {
                'failure_threshold': 2,
                'recovery_timeout': 15.0
            },
            'perplexity': {
                'failure_threshold': 6,
                'rate_limit_window': 600.0
            }
        }
        
        integration = create_enhanced_circuit_breaker_system(
            services_config=custom_config,
            logger=logger
        )
        
        # Check that custom configuration was applied
        openai_breaker = integration.service_breakers['openai']
        assert openai_breaker.config.failure_threshold == 2
        assert openai_breaker.config.recovery_timeout == 15.0
        
        perplexity_breaker = integration.service_breakers['perplexity']
        assert perplexity_breaker.config.failure_threshold == 6
        assert perplexity_breaker.config.rate_limit_window == 600.0


# ============================================================================
# Real-World Scenario Tests
# ============================================================================

class TestRealWorldScenarios:
    """Tests for real-world usage scenarios."""
    
    def test_cascading_failure_prevention(self, logger):
        """Test prevention of cascading failures."""
        integration = create_enhanced_circuit_breaker_system(logger=logger)
        
        # Simulate OpenAI API failure
        openai_breaker = integration.service_breakers['openai']
        for _ in range(3):  # Force to open
            openai_breaker._record_failure(FailureType.RATE_LIMIT, "Rate limit exceeded", 1.0)
        
        # Update state
        openai_breaker._update_state()
        
        # LightRAG should be automatically degraded due to dependency
        lightrag_breaker = integration.service_breakers['lightrag']
        
        # Give the orchestrator time to react
        time.sleep(0.1)
        
        # System should have taken protective measures
        status = integration.get_comprehensive_status()
        system_state = status['orchestrator_status']['system_state']
        assert system_state in ['degraded', 'critical']
    
    def test_recovery_coordination(self, logger):
        """Test coordinated recovery across services."""
        integration = create_enhanced_circuit_breaker_system(logger=logger)
        
        # Force multiple services to fail
        for service_name in ['openai', 'perplexity']:
            breaker = integration.service_breakers[service_name]
            breaker.force_state(EnhancedCircuitBreakerState.OPEN, "Test failure")
        
        # Force system recovery
        integration.orchestrator.force_system_recovery("Test recovery")
        
        # All services should be recovered
        for breaker in integration.service_breakers.values():
            assert breaker.state == EnhancedCircuitBreakerState.CLOSED
    
    def test_adaptive_performance_optimization(self, logger):
        """Test adaptive performance optimization."""
        integration = create_enhanced_circuit_breaker_system(logger=logger)
        
        # Simulate varying performance patterns
        openai_breaker = integration.service_breakers['openai']
        
        # Simulate good performance period
        for _ in range(50):
            openai_breaker._record_success(0.1)  # Fast responses
        
        # Update metrics
        openai_breaker.metrics.total_requests = 50
        openai_breaker.metrics.successful_requests = 50
        openai_breaker.metrics.failed_requests = 0
        
        # Force adaptive threshold adjustment
        original_threshold = openai_breaker.current_failure_threshold
        openai_breaker.last_threshold_adjustment = time.time() - 301
        openai_breaker._adjust_adaptive_thresholds()
        
        # Threshold should be optimized
        assert openai_breaker.current_failure_threshold != original_threshold
    
    def test_comprehensive_monitoring_integration(self, logger):
        """Test comprehensive monitoring and alerting integration."""
        integration = create_enhanced_circuit_breaker_system(logger=logger)
        
        # Start monitoring (would typically run in background)
        from lightrag_integration.enhanced_circuit_breaker_system import monitor_circuit_breaker_health
        
        # The function should start without errors
        try:
            monitor_circuit_breaker_health(integration, check_interval=0.1)
            time.sleep(0.2)  # Let it run briefly
        except Exception as e:
            pytest.fail(f"Monitoring failed: {e}")
        
        # System should still be active
        assert integration.integration_active == True
        
        # Status should be available
        status = integration.get_comprehensive_status()
        assert 'timestamp' in status


# ============================================================================
# Performance and Stress Tests
# ============================================================================

class TestPerformanceAndStress:
    """Performance and stress tests for circuit breaker system."""
    
    def test_high_throughput_operations(self, enhanced_integration):
        """Test system performance under high throughput."""
        import threading
        import concurrent.futures
        
        def execute_operation():
            try:
                return enhanced_integration.execute_with_enhanced_protection(
                    'openai', lambda: "success"
                )
            except Exception:
                return None
        
        # Execute many operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(execute_operation) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Most operations should succeed
        successful = [r for r in results if r == "success"]
        assert len(successful) > 80  # At least 80% success rate
    
    def test_memory_usage_under_load(self, enhanced_integration):
        """Test memory usage doesn't grow excessively under load."""
        import gc
        import sys
        
        initial_objects = len(gc.get_objects())
        
        # Execute many operations
        for _ in range(1000):
            try:
                enhanced_integration.execute_with_enhanced_protection(
                    'cache', lambda: "cached_result"
                )
            except Exception:
                pass
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Object growth should be reasonable (less than 50% increase)
        growth_ratio = (final_objects - initial_objects) / initial_objects
        assert growth_ratio < 0.5
    
    def test_state_consistency_under_concurrent_access(self, logger):
        """Test state consistency under concurrent access."""
        import threading
        import concurrent.futures
        
        breaker = create_service_specific_circuit_breaker(
            ServiceType.OPENAI_API, logger=logger
        )
        
        def concurrent_operation():
            try:
                return breaker.call(lambda: "success")
            except Exception:
                return None
        
        def concurrent_failure():
            try:
                breaker.call(lambda: exec('raise Exception("test")'))
            except Exception:
                return None
        
        # Mix successful and failing operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(100):
                if i % 10 == 0:  # 10% failures
                    futures.append(executor.submit(concurrent_failure))
                else:
                    futures.append(executor.submit(concurrent_operation))
            
            # Wait for all to complete
            for f in concurrent.futures.as_completed(futures):
                f.result()
        
        # State should be consistent
        status = breaker.get_status()
        assert status['metrics']['total_requests'] == 100
        assert status['metrics']['successful_requests'] + status['metrics']['failed_requests'] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])