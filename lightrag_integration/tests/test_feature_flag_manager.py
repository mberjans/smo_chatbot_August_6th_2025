#!/usr/bin/env python3
"""
Comprehensive Unit Tests for FeatureFlagManager.

This module provides extensive test coverage for the FeatureFlagManager class,
including routing logic, hash-based assignment, circuit breaker functionality,
A/B testing capabilities, and performance monitoring.

Test Coverage Areas:
- Routing decision logic and consistency
- Hash-based user assignment for rollout percentages
- Circuit breaker behavior and recovery
- A/B testing cohort assignment and metrics
- Conditional routing rules evaluation
- Quality threshold validation
- Performance metrics tracking
- Caching and optimization features
- Error handling and edge cases
- Thread safety and concurrent operations

Author: Claude Code (Anthropic)
Created: 2025-08-08
"""

import pytest
import pytest_asyncio
import asyncio
import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional

# Import the components under test
from lightrag_integration.feature_flag_manager import (
    FeatureFlagManager,
    RoutingContext,
    RoutingResult,
    RoutingDecision,
    RoutingReason,
    UserCohort,
    CircuitBreakerState,
    PerformanceMetrics
)
from lightrag_integration.config import LightRAGConfig


@pytest.fixture
def mock_config():
    """Create a real LightRAGConfig for testing."""
    # Set environment variables for test configuration
    import os
    original_env = dict(os.environ)
    
    # Set test environment variables
    test_env = {
        "OPENAI_API_KEY": "test_key_for_feature_flags",
        "LIGHTRAG_INTEGRATION_ENABLED": "true",
        "LIGHTRAG_ROLLOUT_PERCENTAGE": "50.0",
        "LIGHTRAG_USER_HASH_SALT": "test_salt_2025",
        "LIGHTRAG_ENABLE_AB_TESTING": "false",
        "LIGHTRAG_FALLBACK_TO_PERPLEXITY": "true",
        "LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS": "30.0",
        "LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON": "false",
        "LIGHTRAG_ENABLE_QUALITY_METRICS": "true",
        "LIGHTRAG_MIN_QUALITY_THRESHOLD": "0.7",
        "LIGHTRAG_ENABLE_CIRCUIT_BREAKER": "true",
        "LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "3",
        "LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "300.0",
        "LIGHTRAG_ENABLE_CONDITIONAL_ROUTING": "false",
        "LIGHTRAG_ROUTING_RULES": "{}"
    }
    
    # Update environment
    os.environ.update(test_env)
    
    try:
        # Create configuration instance
        config = LightRAGConfig()
        return config
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def feature_manager(mock_config, mock_logger):
    """Create a FeatureFlagManager instance for testing."""
    return FeatureFlagManager(config=mock_config, logger=mock_logger)


@pytest.fixture
def routing_context():
    """Create a basic routing context for testing."""
    return RoutingContext(
        user_id="test_user_123",
        session_id="test_session_456",
        query_text="What are the key metabolites in diabetes?",
        query_type="metabolite_identification",
        query_complexity=0.6,
        metadata={"source": "test"}
    )


class TestFeatureFlagManager:
    """Comprehensive test suite for FeatureFlagManager."""


class TestFeatureFlagManagerInitialization:
    """Test FeatureFlagManager initialization and configuration."""
    
    def test_initialization_with_valid_config(self, mock_config, mock_logger):
        """Test successful initialization with valid configuration."""
        manager = FeatureFlagManager(config=mock_config, logger=mock_logger)
        
        assert manager.config == mock_config
        assert manager.logger == mock_logger
        assert isinstance(manager.circuit_breaker_state, CircuitBreakerState)
        assert isinstance(manager.performance_metrics, PerformanceMetrics)
        assert isinstance(manager._routing_cache, dict)
        assert isinstance(manager._cohort_cache, dict)
    
    def test_initialization_with_invalid_config(self):
        """Test initialization fails with invalid config."""
        with pytest.raises(ValueError, match="config must be a LightRAGConfig instance"):
            FeatureFlagManager(config="invalid_config")
    
    def test_initialization_creates_default_logger(self, mock_config):
        """Test that default logger is created when none provided."""
        manager = FeatureFlagManager(config=mock_config)
        
        assert manager.logger is not None
        assert isinstance(manager.logger, logging.Logger)
    
    def test_routing_rules_parsing(self, mock_config, mock_logger):
        """Test routing rules are parsed correctly from configuration."""
        # Set up routing rules in config
        mock_config.lightrag_routing_rules = {
            "long_query_rule": {
                "type": "query_length",
                "min_length": 100,
                "max_length": 500
            },
            "complexity_rule": {
                "type": "query_complexity", 
                "min_complexity": 0.5,
                "max_complexity": 1.0
            },
            "type_rule": {
                "type": "query_type",
                "allowed_types": ["metabolite_identification", "pathway_analysis"]
            }
        }
        
        manager = FeatureFlagManager(config=mock_config, logger=mock_logger)
        
        assert len(manager.routing_rules) == 3
        assert "long_query_rule" in manager.routing_rules
        assert "complexity_rule" in manager.routing_rules
        assert "type_rule" in manager.routing_rules


class TestHashBasedUserAssignment:
    """Test hash-based consistent user assignment for rollout."""
    
    def test_calculate_user_hash_consistency(self, feature_manager):
        """Test that user hash calculation is consistent."""
        user_id = "test_user_123"
        
        hash1 = feature_manager._calculate_user_hash(user_id)
        hash2 = feature_manager._calculate_user_hash(user_id)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex string length
    
    def test_calculate_user_hash_different_users(self, feature_manager):
        """Test that different users get different hashes."""
        hash1 = feature_manager._calculate_user_hash("user1")
        hash2 = feature_manager._calculate_user_hash("user2")
        
        assert hash1 != hash2
    
    def test_calculate_user_hash_uses_salt(self, feature_manager):
        """Test that hash calculation uses the configured salt."""
        user_id = "test_user"
        expected_input = f"{user_id}:{feature_manager.config.lightrag_user_hash_salt}"
        expected_hash = hashlib.sha256(expected_input.encode()).hexdigest()
        
        actual_hash = feature_manager._calculate_user_hash(user_id)
        
        assert actual_hash == expected_hash
    
    def test_get_rollout_percentage_from_hash(self, feature_manager):
        """Test rollout percentage calculation from hash."""
        # Test with known hash suffix
        test_hash = "a" * 56 + "80000000"  # Last 8 chars: 80000000 (hex)
        percentage = feature_manager._get_rollout_percentage_from_hash(test_hash)
        
        # 0x80000000 / 0xFFFFFFFF ≈ 50%
        assert 50.0 <= percentage <= 51.0
    
    def test_rollout_percentage_distribution(self, feature_manager):
        """Test that rollout percentages are evenly distributed."""
        percentages = []
        
        for i in range(1000):
            user_hash = feature_manager._calculate_user_hash(f"user_{i}")
            percentage = feature_manager._get_rollout_percentage_from_hash(user_hash)
            percentages.append(percentage)
        
        # Check distribution is roughly uniform (between 0-100)
        assert min(percentages) >= 0.0
        assert max(percentages) <= 100.0
        
        # Average should be around 50
        avg_percentage = sum(percentages) / len(percentages)
        assert 45.0 <= avg_percentage <= 55.0


class TestUserCohortAssignment:
    """Test user cohort assignment for A/B testing."""
    
    def test_assign_user_cohort_simple_rollout(self, feature_manager):
        """Test user cohort assignment with simple rollout (no A/B testing)."""
        feature_manager.config.lightrag_enable_ab_testing = False
        feature_manager.config.lightrag_rollout_percentage = 50.0
        
        # Test users that should be in LightRAG cohort (low hash percentage)
        with patch.object(feature_manager, '_get_rollout_percentage_from_hash', return_value=25.0):
            cohort = feature_manager._assign_user_cohort("user1", "test_hash")
            assert cohort == UserCohort.LIGHTRAG
        
        # Test users that should be in Perplexity cohort (high hash percentage)
        with patch.object(feature_manager, '_get_rollout_percentage_from_hash', return_value=75.0):
            cohort = feature_manager._assign_user_cohort("user2", "test_hash")
            assert cohort == UserCohort.PERPLEXITY
    
    def test_assign_user_cohort_ab_testing(self, feature_manager):
        """Test user cohort assignment with A/B testing enabled."""
        feature_manager.config.lightrag_enable_ab_testing = True
        feature_manager.config.lightrag_rollout_percentage = 60.0
        
        # Test within rollout percentage - first half should be LightRAG
        with patch.object(feature_manager, '_get_rollout_percentage_from_hash', return_value=15.0):
            cohort = feature_manager._assign_user_cohort("user1", "test_hash")
            assert cohort == UserCohort.LIGHTRAG
        
        # Test within rollout percentage - second half should be Perplexity
        with patch.object(feature_manager, '_get_rollout_percentage_from_hash', return_value=45.0):
            cohort = feature_manager._assign_user_cohort("user2", "test_hash")
            assert cohort == UserCohort.PERPLEXITY
        
        # Test outside rollout percentage - should be Control
        with patch.object(feature_manager, '_get_rollout_percentage_from_hash', return_value=80.0):
            cohort = feature_manager._assign_user_cohort("user3", "test_hash")
            assert cohort == UserCohort.CONTROL
    
    def test_user_cohort_assignment_caching(self, feature_manager):
        """Test that user cohort assignments are cached."""
        user_id = "test_user"
        user_hash = "a" * 56 + "12345678"  # Valid hex hash
        
        # First assignment
        cohort1 = feature_manager._assign_user_cohort(user_id, user_hash)
        
        # Second assignment should return cached result
        cohort2 = feature_manager._assign_user_cohort(user_id, user_hash)
        
        assert cohort1 == cohort2
        assert user_id in feature_manager._cohort_cache
    
    def test_user_cohort_assignment_consistency(self, feature_manager):
        """Test that cohort assignment is consistent across multiple calls."""
        user_id = "consistent_user"
        
        cohorts = []
        for _ in range(10):
            user_hash = feature_manager._calculate_user_hash(user_id)
            cohort = feature_manager._assign_user_cohort(user_id, user_hash)
            cohorts.append(cohort)
        
        # All assignments should be the same
        assert len(set(cohorts)) == 1


class TestCircuitBreakerFunctionality:
    """Test circuit breaker functionality and behavior."""
    
    def test_circuit_breaker_initially_closed(self, feature_manager):
        """Test that circuit breaker is initially closed."""
        is_open = feature_manager._check_circuit_breaker()
        assert is_open is False
        assert feature_manager.circuit_breaker_state.is_open is False
    
    def test_circuit_breaker_disabled(self, feature_manager):
        """Test circuit breaker when disabled in configuration."""
        feature_manager.config.lightrag_enable_circuit_breaker = False
        
        # Even with failures, should not open when disabled
        feature_manager.circuit_breaker_state.failure_count = 10
        is_open = feature_manager._check_circuit_breaker()
        
        assert is_open is False
    
    def test_circuit_breaker_opens_on_failures(self, feature_manager):
        """Test that circuit breaker opens after threshold failures."""
        threshold = feature_manager.config.lightrag_circuit_breaker_failure_threshold
        
        # Set failure count to threshold
        feature_manager.circuit_breaker_state.failure_count = threshold
        
        is_open = feature_manager._check_circuit_breaker()
        
        assert is_open is True
        assert feature_manager.circuit_breaker_state.is_open is True
        assert feature_manager.circuit_breaker_state.last_failure_time is not None
    
    def test_circuit_breaker_recovery_timeout(self, feature_manager):
        """Test circuit breaker recovery after timeout."""
        # Open circuit breaker
        feature_manager.circuit_breaker_state.is_open = True
        feature_manager.circuit_breaker_state.last_failure_time = datetime.now() - timedelta(seconds=400)
        
        # Recovery timeout is 300 seconds, so should attempt recovery
        is_open = feature_manager._check_circuit_breaker()
        
        assert is_open is False
        assert feature_manager.circuit_breaker_state.is_open is False
    
    def test_circuit_breaker_recovery_not_ready(self, feature_manager):
        """Test circuit breaker stays open if recovery timeout not reached."""
        # Open circuit breaker recently
        feature_manager.circuit_breaker_state.is_open = True
        feature_manager.circuit_breaker_state.last_failure_time = datetime.now() - timedelta(seconds=100)
        
        # Recovery timeout is 300 seconds, so should stay open
        is_open = feature_manager._check_circuit_breaker()
        
        assert is_open is True
        assert feature_manager.circuit_breaker_state.is_open is True


class TestConditionalRoutingRules:
    """Test conditional routing rules evaluation."""
    
    def test_conditional_routing_disabled(self, feature_manager, routing_context):
        """Test conditional routing when disabled."""
        feature_manager.config.lightrag_enable_conditional_routing = False
        
        should_use, rule_name = feature_manager._evaluate_conditional_rules(routing_context)
        
        assert should_use is True
        assert rule_name == "no_rules"
    
    def test_query_length_rule(self, feature_manager, routing_context):
        """Test query length conditional routing rule."""
        feature_manager.config.lightrag_enable_conditional_routing = True
        
        # Set up query length rule
        def length_rule(context):
            return 50 <= len(context.query_text) <= 200
        
        feature_manager.routing_rules = {"length_rule": length_rule}
        
        # Test with matching query length
        routing_context.query_text = "This is a query with appropriate length for testing the rule."
        should_use, rule_name = feature_manager._evaluate_conditional_rules(routing_context)
        
        assert should_use is True
        assert rule_name == "length_rule"
        
        # Test with non-matching query length
        routing_context.query_text = "Short"
        should_use, rule_name = feature_manager._evaluate_conditional_rules(routing_context)
        
        assert should_use is False
        assert rule_name == "no_matching_rules"
    
    def test_query_complexity_rule(self, feature_manager, routing_context):
        """Test query complexity conditional routing rule."""
        feature_manager.config.lightrag_enable_conditional_routing = True
        
        # Set up complexity rule
        def complexity_rule(context):
            return context.query_complexity and 0.5 <= context.query_complexity <= 0.8
        
        feature_manager.routing_rules = {"complexity_rule": complexity_rule}
        
        # Test with matching complexity
        routing_context.query_complexity = 0.6
        should_use, rule_name = feature_manager._evaluate_conditional_rules(routing_context)
        
        assert should_use is True
        assert rule_name == "complexity_rule"
        
        # Test with non-matching complexity
        routing_context.query_complexity = 0.9
        should_use, rule_name = feature_manager._evaluate_conditional_rules(routing_context)
        
        assert should_use is False
        assert rule_name == "no_matching_rules"
    
    def test_query_type_rule(self, feature_manager, routing_context):
        """Test query type conditional routing rule."""
        feature_manager.config.lightrag_enable_conditional_routing = True
        
        # Set up query type rule
        allowed_types = {"metabolite_identification", "pathway_analysis"}
        def type_rule(context):
            return context.query_type in allowed_types
        
        feature_manager.routing_rules = {"type_rule": type_rule}
        
        # Test with allowed query type
        routing_context.query_type = "metabolite_identification"
        should_use, rule_name = feature_manager._evaluate_conditional_rules(routing_context)
        
        assert should_use is True
        assert rule_name == "type_rule"
        
        # Test with non-allowed query type
        routing_context.query_type = "drug_discovery"
        should_use, rule_name = feature_manager._evaluate_conditional_rules(routing_context)
        
        assert should_use is False
        assert rule_name == "no_matching_rules"
    
    def test_rule_evaluation_error_handling(self, feature_manager, routing_context):
        """Test error handling in rule evaluation."""
        feature_manager.config.lightrag_enable_conditional_routing = True
        
        # Set up rule that raises exception
        def failing_rule(context):
            raise ValueError("Rule evaluation failed")
        
        feature_manager.routing_rules = {"failing_rule": failing_rule}
        
        should_use, rule_name = feature_manager._evaluate_conditional_rules(routing_context)
        
        assert should_use is False
        assert rule_name == "no_matching_rules"


class TestQualityThresholdValidation:
    """Test quality threshold validation functionality."""
    
    def test_quality_threshold_disabled(self, feature_manager):
        """Test quality threshold when disabled."""
        feature_manager.config.lightrag_enable_quality_metrics = False
        
        is_acceptable = feature_manager._check_quality_threshold()
        
        assert is_acceptable is True
    
    def test_quality_threshold_no_data(self, feature_manager):
        """Test quality threshold with no quality data."""
        feature_manager.config.lightrag_enable_quality_metrics = True
        feature_manager.performance_metrics.lightrag_quality_scores = []
        
        is_acceptable = feature_manager._check_quality_threshold()
        
        assert is_acceptable is True
    
    def test_quality_threshold_above_minimum(self, feature_manager):
        """Test quality threshold when above minimum."""
        feature_manager.config.lightrag_enable_quality_metrics = True
        feature_manager.config.lightrag_min_quality_threshold = 0.7
        feature_manager.performance_metrics.lightrag_quality_scores = [0.8, 0.9, 0.75]
        
        is_acceptable = feature_manager._check_quality_threshold()
        
        assert is_acceptable is True
    
    def test_quality_threshold_below_minimum(self, feature_manager):
        """Test quality threshold when below minimum."""
        feature_manager.config.lightrag_enable_quality_metrics = True
        feature_manager.config.lightrag_min_quality_threshold = 0.7
        feature_manager.performance_metrics.lightrag_quality_scores = [0.6, 0.5, 0.4]
        
        is_acceptable = feature_manager._check_quality_threshold()
        
        assert is_acceptable is False


class TestRoutingDecisionLogic:
    """Test main routing decision logic."""
    
    def test_routing_integration_disabled(self, feature_manager, routing_context):
        """Test routing when integration is disabled."""
        feature_manager.config.lightrag_integration_enabled = False
        
        result = feature_manager.should_use_lightrag(routing_context)
        
        assert result.decision == RoutingDecision.PERPLEXITY
        assert result.reason == RoutingReason.FEATURE_DISABLED
        assert result.confidence == 1.0
    
    def test_routing_forced_cohort(self, feature_manager, routing_context):
        """Test routing with forced user cohort."""
        feature_manager.config.lightrag_force_user_cohort = "lightrag"
        
        result = feature_manager.should_use_lightrag(routing_context)
        
        assert result.decision == RoutingDecision.LIGHTRAG
        assert result.reason == RoutingReason.FORCED_COHORT
        assert result.user_cohort == UserCohort.LIGHTRAG
        assert result.confidence == 1.0
    
    def test_routing_circuit_breaker_open(self, feature_manager, routing_context):
        """Test routing when circuit breaker is open."""
        # Open circuit breaker
        feature_manager.circuit_breaker_state.is_open = True
        feature_manager.circuit_breaker_state.failure_count = 5
        feature_manager.circuit_breaker_state.last_failure_time = datetime.now()
        
        result = feature_manager.should_use_lightrag(routing_context)
        
        assert result.decision == RoutingDecision.PERPLEXITY
        assert result.reason == RoutingReason.CIRCUIT_BREAKER_OPEN
        assert result.circuit_breaker_state == "open"
        assert result.confidence == 1.0
    
    def test_routing_quality_threshold_failed(self, feature_manager, routing_context):
        """Test routing when quality threshold is not met."""
        feature_manager.config.lightrag_enable_quality_metrics = True
        feature_manager.config.lightrag_min_quality_threshold = 0.8
        feature_manager.performance_metrics.lightrag_quality_scores = [0.5, 0.6, 0.4]
        
        result = feature_manager.should_use_lightrag(routing_context)
        
        assert result.decision == RoutingDecision.PERPLEXITY
        assert result.reason == RoutingReason.QUALITY_THRESHOLD
        assert result.confidence == 0.8
    
    def test_routing_conditional_rule_failed(self, feature_manager, routing_context):
        """Test routing when conditional rules fail."""
        feature_manager.config.lightrag_enable_conditional_routing = True
        
        # Set up failing rule
        def failing_rule(context):
            return False
        
        feature_manager.routing_rules = {"test_rule": failing_rule}
        
        result = feature_manager.should_use_lightrag(routing_context)
        
        assert result.decision == RoutingDecision.PERPLEXITY
        assert result.reason == RoutingReason.CONDITIONAL_RULE
        assert result.confidence == 0.9
        assert result.metadata.get("failed_rule") == "no_matching_rules"
    
    def test_routing_successful_lightrag(self, feature_manager, routing_context):
        """Test successful routing to LightRAG."""
        # Set up for LightRAG routing
        feature_manager.config.lightrag_rollout_percentage = 100.0  # All users get LightRAG
        
        result = feature_manager.should_use_lightrag(routing_context)
        
        assert result.decision == RoutingDecision.LIGHTRAG
        assert result.reason == RoutingReason.ROLLOUT_PERCENTAGE
        assert result.confidence == 0.95
        assert result.rollout_hash is not None
        assert result.circuit_breaker_state == "closed"
    
    def test_routing_successful_perplexity(self, feature_manager, routing_context):
        """Test successful routing to Perplexity."""
        # Set up for Perplexity routing
        feature_manager.config.lightrag_rollout_percentage = 0.0  # No users get LightRAG
        
        result = feature_manager.should_use_lightrag(routing_context)
        
        assert result.decision == RoutingDecision.PERPLEXITY
        assert result.reason == RoutingReason.ROLLOUT_PERCENTAGE
        assert result.confidence == 0.95
    
    def test_routing_with_ab_testing(self, feature_manager, routing_context):
        """Test routing with A/B testing enabled."""
        feature_manager.config.lightrag_enable_ab_testing = True
        feature_manager.config.lightrag_rollout_percentage = 100.0
        
        result = feature_manager.should_use_lightrag(routing_context)
        
        assert result.reason == RoutingReason.USER_COHORT_ASSIGNMENT
        assert result.user_cohort in [UserCohort.LIGHTRAG, UserCohort.PERPLEXITY]


class TestPerformanceMetricsTracking:
    """Test performance metrics tracking and management."""
    
    def test_record_success_lightrag(self, feature_manager):
        """Test recording successful LightRAG operation."""
        initial_count = feature_manager.performance_metrics.lightrag_success_count
        initial_response_times = len(feature_manager.performance_metrics.lightrag_response_times)
        
        feature_manager.record_success("lightrag", 1.5, 0.8)
        
        assert feature_manager.performance_metrics.lightrag_success_count == initial_count + 1
        assert len(feature_manager.performance_metrics.lightrag_response_times) == initial_response_times + 1
        assert feature_manager.performance_metrics.lightrag_response_times[-1] == 1.5
        assert 0.8 in feature_manager.performance_metrics.lightrag_quality_scores
    
    def test_record_success_perplexity(self, feature_manager):
        """Test recording successful Perplexity operation."""
        initial_count = feature_manager.performance_metrics.perplexity_success_count
        
        feature_manager.record_success("perplexity", 2.0, 0.9)
        
        assert feature_manager.performance_metrics.perplexity_success_count == initial_count + 1
        assert 2.0 in feature_manager.performance_metrics.perplexity_response_times
        assert 0.9 in feature_manager.performance_metrics.perplexity_quality_scores
    
    def test_record_failure_lightrag(self, feature_manager):
        """Test recording failed LightRAG operation."""
        initial_error_count = feature_manager.performance_metrics.lightrag_error_count
        initial_failure_count = feature_manager.circuit_breaker_state.failure_count
        
        feature_manager.record_failure("lightrag", "Connection timeout")
        
        assert feature_manager.performance_metrics.lightrag_error_count == initial_error_count + 1
        assert feature_manager.circuit_breaker_state.failure_count == initial_failure_count + 1
        assert feature_manager.circuit_breaker_state.last_failure_time is not None
    
    def test_record_failure_perplexity(self, feature_manager):
        """Test recording failed Perplexity operation."""
        initial_error_count = feature_manager.performance_metrics.perplexity_error_count
        
        feature_manager.record_failure("perplexity", "API error")
        
        assert feature_manager.performance_metrics.perplexity_error_count == initial_error_count + 1
    
    def test_circuit_breaker_failure_recovery(self, feature_manager):
        """Test circuit breaker failure count recovery on success."""
        # Set some failures
        feature_manager.circuit_breaker_state.failure_count = 3
        
        # Record success should reduce failure count
        feature_manager.record_success("lightrag", 1.0)
        
        assert feature_manager.circuit_breaker_state.failure_count == 2
    
    def test_metrics_memory_management(self, feature_manager):
        """Test that metrics arrays don't grow indefinitely."""
        # Fill arrays beyond limit
        for i in range(1200):  # More than the 1000 limit
            feature_manager.record_success("lightrag", 1.0, 0.8)
        
        # Should be capped at 1000
        assert len(feature_manager.performance_metrics.lightrag_response_times) == 1000
        assert len(feature_manager.performance_metrics.lightrag_quality_scores) == 1000


class TestCachingAndOptimization:
    """Test caching mechanisms and performance optimizations."""
    
    def test_routing_result_caching(self, feature_manager, routing_context):
        """Test that routing results are cached."""
        # First call
        result1 = feature_manager.should_use_lightrag(routing_context)
        
        # Second call should return cached result
        result2 = feature_manager.should_use_lightrag(routing_context)
        
        assert result1.decision == result2.decision
        assert result1.reason == result2.reason
        assert len(feature_manager._routing_cache) > 0
    
    def test_routing_cache_expiration(self, feature_manager, routing_context):
        """Test that cached routing results expire."""
        # Mock expired cache entry
        cache_key = f"{routing_context.user_id or 'anonymous'}:{hash(routing_context.query_text or '')}"
        expired_time = datetime.now() - timedelta(minutes=10)
        feature_manager._routing_cache[cache_key] = (Mock(), expired_time)
        
        # Should not return cached result due to expiration
        result = feature_manager.should_use_lightrag(routing_context)
        
        assert result is not None  # Got fresh result
        # Cache entry should be removed or replaced with fresh data
        if cache_key in feature_manager._routing_cache:
            cached_result, cached_time = feature_manager._routing_cache[cache_key]
            # If still cached, it should be a fresh entry (not the expired one we set)
            assert cached_time != expired_time
    
    def test_cache_size_management(self, feature_manager):
        """Test that cache size is managed to prevent memory issues."""
        # Fill cache beyond limit
        for i in range(1100):  # More than 1000 limit
            cache_key = f"test_key_{i}"
            feature_manager._cache_routing_result(cache_key, Mock())
        
        # Should be capped
        assert len(feature_manager._routing_cache) <= 1000
    
    def test_cohort_cache_consistency(self, feature_manager):
        """Test that cohort cache maintains consistency."""
        user_id = "test_user"
        user_hash = "a" * 56 + "87654321"  # Valid hex hash
        
        # First assignment
        cohort1 = feature_manager._assign_user_cohort(user_id, user_hash)
        
        # Cache should contain the assignment
        assert user_id in feature_manager._cohort_cache
        assert feature_manager._cohort_cache[user_id] == cohort1
        
        # Second assignment should use cache
        cohort2 = feature_manager._assign_user_cohort(user_id, user_hash)
        assert cohort1 == cohort2


class TestPerformanceSummaryAndReporting:
    """Test performance summary generation and reporting."""
    
    def test_get_performance_summary_structure(self, feature_manager):
        """Test that performance summary has correct structure."""
        summary = feature_manager.get_performance_summary()
        
        assert "circuit_breaker" in summary
        assert "performance" in summary
        assert "configuration" in summary
        assert "cache_stats" in summary
        
        # Circuit breaker section
        cb = summary["circuit_breaker"]
        assert "is_open" in cb
        assert "failure_count" in cb
        assert "failure_rate" in cb
        assert "success_rate" in cb
        
        # Performance section
        perf = summary["performance"]
        assert "lightrag" in perf
        assert "perplexity" in perf
        assert "last_updated" in perf
        
        # Configuration section
        config = summary["configuration"]
        assert "integration_enabled" in config
        assert "rollout_percentage" in config
        
        # Cache stats section
        cache = summary["cache_stats"]
        assert "routing_cache_size" in cache
        assert "cohort_cache_size" in cache
    
    def test_performance_metrics_calculation(self, feature_manager):
        """Test performance metrics calculations."""
        # Add some test data
        feature_manager.performance_metrics.lightrag_response_times = [1.0, 2.0, 3.0]
        feature_manager.performance_metrics.lightrag_quality_scores = [0.8, 0.9, 0.7]
        
        assert feature_manager.performance_metrics.get_lightrag_avg_response_time() == 2.0
        assert abs(feature_manager.performance_metrics.get_lightrag_avg_quality() - 0.8) < 0.001  # Use tolerance for float comparison
    
    def test_circuit_breaker_metrics(self, feature_manager):
        """Test circuit breaker metrics calculation."""
        # Set up circuit breaker state
        feature_manager.circuit_breaker_state.total_requests = 100
        feature_manager.circuit_breaker_state.successful_requests = 80
        
        assert feature_manager.circuit_breaker_state.failure_rate == 0.2
        assert feature_manager.circuit_breaker_state.success_rate == 0.8


class TestUtilityMethods:
    """Test utility methods and management functions."""
    
    def test_reset_circuit_breaker(self, feature_manager):
        """Test manual circuit breaker reset."""
        # Set circuit breaker state
        feature_manager.circuit_breaker_state.is_open = True
        feature_manager.circuit_breaker_state.failure_count = 5
        feature_manager.circuit_breaker_state.last_failure_time = datetime.now()
        
        feature_manager.reset_circuit_breaker()
        
        assert feature_manager.circuit_breaker_state.is_open is False
        assert feature_manager.circuit_breaker_state.failure_count == 0
        assert feature_manager.circuit_breaker_state.last_failure_time is None
    
    def test_clear_caches(self, feature_manager):
        """Test cache clearing functionality."""
        # Add some cached data
        feature_manager._routing_cache["test"] = (Mock(), datetime.now())
        feature_manager._cohort_cache["user"] = UserCohort.LIGHTRAG
        
        feature_manager.clear_caches()
        
        assert len(feature_manager._routing_cache) == 0
        assert len(feature_manager._cohort_cache) == 0
    
    def test_update_rollout_percentage_valid(self, feature_manager):
        """Test updating rollout percentage with valid values."""
        original_percentage = feature_manager.config.lightrag_rollout_percentage
        
        feature_manager.update_rollout_percentage(75.0)
        
        assert feature_manager.config.lightrag_rollout_percentage == 75.0
        assert len(feature_manager._routing_cache) == 0  # Cache should be cleared
    
    def test_update_rollout_percentage_invalid(self, feature_manager):
        """Test updating rollout percentage with invalid values."""
        with pytest.raises(ValueError, match="Rollout percentage must be between 0 and 100"):
            feature_manager.update_rollout_percentage(-10.0)
        
        with pytest.raises(ValueError, match="Rollout percentage must be between 0 and 100"):
            feature_manager.update_rollout_percentage(150.0)


class TestThreadSafetyAndConcurrency:
    """Test thread safety and concurrent operations."""
    
    def test_concurrent_routing_decisions(self, feature_manager):
        """Test concurrent routing decisions for thread safety."""
        import concurrent.futures
        import threading
        
        routing_contexts = [
            RoutingContext(user_id=f"user_{i}", query_text=f"Query {i}")
            for i in range(10)
        ]
        
        def make_routing_decision(context):
            return feature_manager.should_use_lightrag(context)
        
        # Execute concurrent routing decisions using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_routing_decision, ctx) for ctx in routing_contexts]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete successfully
        assert len(results) == 10
        assert all(isinstance(result, RoutingResult) for result in results)
    
    def test_thread_safe_metrics_recording(self, feature_manager):
        """Test thread-safe metrics recording."""
        import threading
        
        def record_metrics():
            for i in range(100):
                feature_manager.record_success("lightrag", 1.0, 0.8)
        
        # Start multiple threads
        threads = [threading.Thread(target=record_metrics) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should have 500 total successes
        assert feature_manager.performance_metrics.lightrag_success_count == 500
    
    def test_thread_safe_cache_operations(self, feature_manager):
        """Test thread-safe cache operations."""
        import threading
        import random
        
        def cache_operations():
            for i in range(50):
                context = RoutingContext(
                    user_id=f"user_{random.randint(1, 10)}",
                    query_text=f"Query {i}"
                )
                feature_manager.should_use_lightrag(context)
        
        # Start multiple threads
        threads = [threading.Thread(target=cache_operations) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Cache should be populated without errors
        assert len(feature_manager._routing_cache) > 0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_routing_with_none_user_id(self, feature_manager):
        """Test routing with None user ID."""
        context = RoutingContext(
            user_id=None,
            session_id=None,
            query_text="Test query"
        )
        
        result = feature_manager.should_use_lightrag(context)
        
        assert result is not None
        assert isinstance(result, RoutingResult)
    
    def test_routing_with_empty_query(self, feature_manager):
        """Test routing with empty query text."""
        context = RoutingContext(
            user_id="test_user",
            query_text=""
        )
        
        result = feature_manager.should_use_lightrag(context)
        
        assert result is not None
        assert isinstance(result, RoutingResult)
    
    def test_routing_with_exception_handling(self, feature_manager):
        """Test routing decision handles exceptions gracefully."""
        # Mock a method to raise exception
        with patch.object(feature_manager, '_calculate_user_hash', side_effect=Exception("Test error")):
            context = RoutingContext(user_id="test_user", query_text="test")
            result = feature_manager.should_use_lightrag(context)
            
            assert result.decision == RoutingDecision.PERPLEXITY
            assert result.reason == RoutingReason.PERFORMANCE_DEGRADATION
            assert result.confidence == 0.5
            assert "error" in result.metadata
    
    def test_performance_metrics_with_zero_data(self, feature_manager):
        """Test performance metrics calculations with zero data."""
        # Clear all metrics data
        feature_manager.performance_metrics.lightrag_response_times = []
        feature_manager.performance_metrics.lightrag_quality_scores = []
        
        assert feature_manager.performance_metrics.get_lightrag_avg_response_time() == 0.0
        assert feature_manager.performance_metrics.get_lightrag_avg_quality() == 0.0
    
    def test_circuit_breaker_state_with_zero_requests(self, feature_manager):
        """Test circuit breaker state calculations with zero requests."""
        # Reset to zero
        feature_manager.circuit_breaker_state.total_requests = 0
        feature_manager.circuit_breaker_state.successful_requests = 0
        
        assert feature_manager.circuit_breaker_state.failure_rate == 0.0
        assert feature_manager.circuit_breaker_state.success_rate == 1.0


class TestRoutingResultSerialization:
    """Test RoutingResult serialization and data conversion."""
    
    def test_routing_result_to_dict(self, feature_manager, routing_context):
        """Test RoutingResult to_dict conversion."""
        result = feature_manager.should_use_lightrag(routing_context)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "decision" in result_dict
        assert "reason" in result_dict
        assert "confidence" in result_dict
        assert "processing_time_ms" in result_dict
        assert "metadata" in result_dict
        
        # Check enum values are serialized as strings
        assert isinstance(result_dict["decision"], str)
        assert isinstance(result_dict["reason"], str)
    
    def test_routing_result_serialization_completeness(self):
        """Test RoutingResult serialization includes all fields."""
        result = RoutingResult(
            decision=RoutingDecision.LIGHTRAG,
            reason=RoutingReason.USER_COHORT_ASSIGNMENT,
            user_cohort=UserCohort.LIGHTRAG,
            confidence=0.95,
            rollout_hash="abcd1234",
            circuit_breaker_state="closed",
            quality_score=0.85,
            processing_time_ms=150.0,
            metadata={"test": "value"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["decision"] == "lightrag"
        assert result_dict["reason"] == "user_cohort_assignment"
        assert result_dict["user_cohort"] == "lightrag"
        assert result_dict["confidence"] == 0.95
        assert result_dict["rollout_hash"] == "abcd1234"
        assert result_dict["circuit_breaker_state"] == "closed"
        assert result_dict["quality_score"] == 0.85
        assert result_dict["processing_time_ms"] == 150.0
        assert result_dict["metadata"] == {"test": "value"}


# Mark the end of the first test file
if __name__ == "__main__":
    pytest.main([__file__])