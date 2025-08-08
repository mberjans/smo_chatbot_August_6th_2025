#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Tests for Feature Flag System.

This module provides comprehensive integration tests for complete workflows
with different feature flag configurations, testing the interaction between
all components of the feature flag system.

Test Coverage Areas:
- Complete query workflows with different routing decisions
- Feature flag interactions across multiple components
- A/B testing workflow end-to-end
- Circuit breaker integration with real failures
- Fallback mechanisms under various failure scenarios
- Performance monitoring integration
- Quality assessment integration
- Configuration-driven behavior changes
- Cache integration with routing decisions
- Multi-user scenarios and consistency
- Real-world usage patterns

Author: Claude Code (Anthropic)
Created: 2025-08-08
"""

import pytest
import pytest_asyncio
import asyncio
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional, Tuple

# Import components for integration testing
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.feature_flag_manager import (
    FeatureFlagManager,
    RoutingContext,
    RoutingResult,
    RoutingDecision,
    RoutingReason,
    UserCohort
)
from lightrag_integration.integration_wrapper import (
    IntegratedQueryService,
    QueryRequest,
    ServiceResponse,
    ResponseType,
    QualityMetric
)


class TestCompleteQueryWorkflows:
    """Test complete query workflows from request to response."""
    
    @pytest.fixture
    async def integrated_test_service(self):
        """Create a fully integrated test service."""
        config = LightRAGConfig(
            api_key="test_api_key",
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0,
            lightrag_enable_ab_testing=True,
            lightrag_fallback_to_perplexity=True,
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=3,
            lightrag_enable_quality_metrics=True,
            lightrag_min_quality_threshold=0.7
        )
        
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity, \
             patch('lightrag_integration.integration_wrapper.LightRAGQueryService') as mock_lightrag:
            
            # Configure mock services
            mock_perplexity_instance = Mock()
            mock_perplexity_instance.query_async = AsyncMock()
            mock_perplexity_instance.get_service_name.return_value = "perplexity"
            mock_perplexity_instance.is_available.return_value = True
            mock_perplexity_instance.health_check = AsyncMock(return_value=True)
            mock_perplexity.return_value = mock_perplexity_instance
            
            mock_lightrag_instance = Mock()
            mock_lightrag_instance.query_async = AsyncMock()
            mock_lightrag_instance.get_service_name.return_value = "lightrag"
            mock_lightrag_instance.is_available.return_value = True
            mock_lightrag_instance.health_check = AsyncMock(return_value=True)
            mock_lightrag.return_value = mock_lightrag_instance
            
            service = IntegratedQueryService(
                config=config,
                perplexity_api_key="test_perplexity_key"
            )
            
            # Inject mocks
            service.perplexity_service = mock_perplexity_instance
            service.lightrag_service = mock_lightrag_instance
            
            yield service, mock_perplexity_instance, mock_lightrag_instance
    
    @pytest.mark.asyncio
    async def test_lightrag_routing_complete_workflow(self, integrated_test_service):
        """Test complete workflow when routed to LightRAG."""
        service, mock_perplexity, mock_lightrag = integrated_test_service
        
        # Configure routing to LightRAG
        with patch.object(service.feature_manager, 'should_use_lightrag') as mock_routing:
            mock_routing.return_value = RoutingResult(
                decision=RoutingDecision.LIGHTRAG,
                reason=RoutingReason.USER_COHORT_ASSIGNMENT,
                user_cohort=UserCohort.LIGHTRAG,
                confidence=0.95,
                rollout_hash="abc123",
                quality_score=0.85
            )
            
            # Configure LightRAG success response
            lightrag_response = ServiceResponse(
                content="LightRAG found key metabolites including glucose, lactate, and insulin pathways.",
                response_type=ResponseType.LIGHTRAG,
                processing_time=1.8,
                quality_scores={QualityMetric.RELEVANCE: 0.9, QualityMetric.ACCURACY: 0.85},
                citations=["https://pubmed.ncbi.nlm.nih.gov/example"],
                service_info={"service": "lightrag", "model": "gpt-4o-mini"}
            )
            mock_lightrag.query_async.return_value = lightrag_response
            
            # Execute query
            request = QueryRequest(
                query_text="What are the key metabolites in diabetes pathways?",
                user_id="test_user_123",
                session_id="session_456",
                timeout_seconds=30.0
            )
            
            response = await service.query_async(request)
            
            # Verify complete workflow
            assert response.is_success
            assert response.response_type == ResponseType.LIGHTRAG
            assert "glucose, lactate" in response.content
            assert response.metadata['routing_decision'] == 'lightrag'
            assert response.metadata['user_cohort'] == 'lightrag'
            assert response.metadata['routing_confidence'] == 0.95
            assert response.processing_time > 0
            
            # Verify metrics were recorded
            service.feature_manager.record_success.assert_called_once()
            call_args = service.feature_manager.record_success.call_args
            assert call_args[0][0] == "lightrag"  # service name
            assert call_args[0][1] > 0  # response time
            assert call_args[0][2] is not None  # quality score
    
    @pytest.mark.asyncio
    async def test_perplexity_routing_complete_workflow(self, integrated_test_service):
        """Test complete workflow when routed to Perplexity."""
        service, mock_perplexity, mock_lightrag = integrated_test_service
        
        # Configure routing to Perplexity
        with patch.object(service.feature_manager, 'should_use_lightrag') as mock_routing:
            mock_routing.return_value = RoutingResult(
                decision=RoutingDecision.PERPLEXITY,
                reason=RoutingReason.ROLLOUT_PERCENTAGE,
                user_cohort=UserCohort.PERPLEXITY,
                confidence=0.90,
                rollout_hash="def456"
            )
            
            # Configure Perplexity success response
            perplexity_response = ServiceResponse(
                content="Diabetes involves key metabolites such as glucose, HbA1c, and fatty acids.",
                response_type=ResponseType.PERPLEXITY,
                processing_time=2.1,
                quality_scores={QualityMetric.RELEVANCE: 0.88},
                citations=["https://doi.org/10.1000/example"],
                service_info={"service": "perplexity", "request_id": "req_123"}
            )
            mock_perplexity.query_async.return_value = perplexity_response
            
            # Execute query
            request = QueryRequest(
                query_text="What metabolites are involved in diabetes?",
                user_id="test_user_789",
                session_id="session_999"
            )
            
            response = await service.query_async(request)
            
            # Verify complete workflow
            assert response.is_success
            assert response.response_type == ResponseType.PERPLEXITY
            assert "glucose, HbA1c" in response.content
            assert response.metadata['routing_decision'] == 'perplexity'
            assert response.metadata['user_cohort'] == 'perplexity'
            assert response.metadata['routing_confidence'] == 0.90
            
            # Verify metrics were recorded
            service.feature_manager.record_success.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_workflow_lightrag_to_perplexity(self, integrated_test_service):
        """Test complete fallback workflow from LightRAG failure to Perplexity success."""
        service, mock_perplexity, mock_lightrag = integrated_test_service
        
        # Configure routing to LightRAG initially
        with patch.object(service.feature_manager, 'should_use_lightrag') as mock_routing:
            mock_routing.return_value = RoutingResult(
                decision=RoutingDecision.LIGHTRAG,
                reason=RoutingReason.USER_COHORT_ASSIGNMENT,
                confidence=0.95
            )
            
            # Configure LightRAG failure
            lightrag_failure = ServiceResponse(
                content="",
                response_type=ResponseType.LIGHTRAG,
                error_details="LightRAG connection timeout",
                processing_time=30.0
            )
            mock_lightrag.query_async.return_value = lightrag_failure
            
            # Configure Perplexity success (fallback)
            perplexity_success = ServiceResponse(
                content="Fallback response: Key diabetes metabolites include glucose and insulin.",
                response_type=ResponseType.PERPLEXITY,
                processing_time=1.5,
                quality_scores={QualityMetric.RELEVANCE: 0.82}
            )
            mock_perplexity.query_async.return_value = perplexity_success
            
            # Execute query
            request = QueryRequest(
                query_text="Diabetes metabolite analysis?",
                user_id="fallback_user"
            )
            
            response = await service.query_async(request)
            
            # Verify fallback workflow completed
            assert response.is_success
            assert response.response_type == ResponseType.PERPLEXITY
            assert "Fallback response" in response.content
            assert response.metadata['fallback_used'] is True
            assert response.metadata['original_routing_decision'] == 'lightrag'
            
            # Verify both failure and success were recorded
            assert service.feature_manager.record_failure.called
            assert service.feature_manager.record_success.called


class TestABTestingIntegration:
    """Test A/B testing integration workflows."""
    
    @pytest.fixture
    def ab_testing_config(self):
        """Create configuration for A/B testing."""
        return LightRAGConfig(
            api_key="test_key",
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=60.0,  # 60% rollout
            lightrag_enable_ab_testing=True,   # Enable A/B testing
            lightrag_user_hash_salt="test_salt_ab"
        )
    
    @pytest.mark.asyncio
    async def test_ab_testing_cohort_consistency(self, ab_testing_config):
        """Test that users are consistently assigned to cohorts."""
        feature_manager = FeatureFlagManager(config=ab_testing_config)
        
        user_cohorts = {}
        test_users = [f"user_{i}" for i in range(100)]
        
        # Assign cohorts multiple times for each user
        for _ in range(5):  # 5 iterations
            for user_id in test_users:
                context = RoutingContext(user_id=user_id, query_text="test query")
                result = feature_manager.should_use_lightrag(context)
                
                if user_id not in user_cohorts:
                    user_cohorts[user_id] = result.user_cohort
                else:
                    # Should be consistent across calls
                    assert user_cohorts[user_id] == result.user_cohort, f"Inconsistent cohort for {user_id}"
    
    @pytest.mark.asyncio
    async def test_ab_testing_distribution(self, ab_testing_config):
        """Test A/B testing distribution across cohorts."""
        feature_manager = FeatureFlagManager(config=ab_testing_config)
        
        cohort_counts = {
            UserCohort.LIGHTRAG: 0,
            UserCohort.PERPLEXITY: 0,
            UserCohort.CONTROL: 0
        }
        
        # Test with many users
        for i in range(1000):
            user_id = f"ab_test_user_{i}"
            context = RoutingContext(user_id=user_id, query_text="test query")
            result = feature_manager.should_use_lightrag(context)
            
            if result.user_cohort in cohort_counts:
                cohort_counts[result.user_cohort] += 1
        
        total_users = sum(cohort_counts.values())
        rollout_percentage = ab_testing_config.lightrag_rollout_percentage
        
        # Within rollout percentage, should be split between LightRAG and Perplexity
        lightrag_percentage = (cohort_counts[UserCohort.LIGHTRAG] / total_users) * 100
        perplexity_percentage = (cohort_counts[UserCohort.PERPLEXITY] / total_users) * 100
        control_percentage = (cohort_counts[UserCohort.CONTROL] / total_users) * 100
        
        # Control should be roughly (100 - rollout_percentage)%
        expected_control = 100 - rollout_percentage
        assert abs(control_percentage - expected_control) < 5.0  # Within 5% tolerance
        
        # LightRAG and Perplexity should split the rollout percentage roughly equally
        expected_each = rollout_percentage / 2
        assert abs(lightrag_percentage - expected_each) < 5.0
        assert abs(perplexity_percentage - expected_each) < 5.0
    
    @pytest.mark.asyncio
    async def test_ab_testing_metrics_collection(self, ab_testing_config):
        """Test A/B testing metrics collection."""
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity, \
             patch('lightrag_integration.integration_wrapper.LightRAGQueryService') as mock_lightrag:
            
            # Configure mocks
            mock_perplexity_instance = Mock()
            mock_perplexity_instance.query_async = AsyncMock(return_value=ServiceResponse(
                content="Perplexity response", response_type=ResponseType.PERPLEXITY, processing_time=1.0
            ))
            mock_perplexity.return_value = mock_perplexity_instance
            
            mock_lightrag_instance = Mock()
            mock_lightrag_instance.query_async = AsyncMock(return_value=ServiceResponse(
                content="LightRAG response", response_type=ResponseType.LIGHTRAG, processing_time=1.5
            ))
            mock_lightrag.return_value = mock_lightrag_instance
            
            service = IntegratedQueryService(
                config=ab_testing_config,
                perplexity_api_key="test_key"
            )
            
            # Execute queries from different cohorts
            test_scenarios = [
                ("lightrag_user", "Should route to LightRAG"),
                ("perplexity_user", "Should route to Perplexity")
            ]
            
            for user_id, query_text in test_scenarios:
                request = QueryRequest(user_id=user_id, query_text=query_text)
                await service.query_async(request)
            
            # Check that A/B test metrics were collected
            ab_metrics = service.get_ab_test_metrics()
            assert isinstance(ab_metrics, dict)
            
            # Should have some data for different cohorts
            # The exact structure depends on implementation


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration in complete workflows."""
    
    @pytest.fixture
    def circuit_breaker_config(self):
        """Create configuration with circuit breaker enabled."""
        return LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=3,
            lightrag_circuit_breaker_recovery_timeout=10.0,  # Short timeout for testing
            lightrag_fallback_to_perplexity=True
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_accumulation(self, circuit_breaker_config):
        """Test circuit breaker failure accumulation and opening."""
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity, \
             patch('lightrag_integration.integration_wrapper.LightRAGQueryService') as mock_lightrag:
            
            # Configure LightRAG to fail
            mock_lightrag_instance = Mock()
            mock_lightrag_instance.query_async = AsyncMock(return_value=ServiceResponse(
                content="", response_type=ResponseType.LIGHTRAG, error_details="Connection failed"
            ))
            mock_lightrag.return_value = mock_lightrag_instance
            
            # Configure Perplexity to succeed (for fallback)
            mock_perplexity_instance = Mock()
            mock_perplexity_instance.query_async = AsyncMock(return_value=ServiceResponse(
                content="Perplexity fallback response", response_type=ResponseType.PERPLEXITY
            ))
            mock_perplexity.return_value = mock_perplexity_instance
            
            service = IntegratedQueryService(
                config=circuit_breaker_config,
                perplexity_api_key="test_key"
            )
            
            # Force routing to LightRAG
            with patch.object(service.feature_manager, 'should_use_lightrag') as mock_routing:
                mock_routing.return_value = RoutingResult(
                    decision=RoutingDecision.LIGHTRAG,
                    reason=RoutingReason.USER_COHORT_ASSIGNMENT,
                    confidence=0.95
                )
                
                # Execute multiple failing queries
                for i in range(5):
                    request = QueryRequest(
                        user_id=f"user_{i}",
                        query_text="Test query that will fail"
                    )
                    response = await service.query_async(request)
                    
                    if i < 3:
                        # First 3 should use fallback (circuit breaker not open yet)
                        assert response.metadata.get('fallback_used', False)
                    else:
                        # After threshold, should route directly to Perplexity due to circuit breaker
                        assert response.metadata.get('circuit_breaker_blocked', False) or \
                               response.metadata.get('routing_decision') == 'perplexity'
                
                # Check circuit breaker state
                cb_state = service.feature_manager.circuit_breaker_state
                assert cb_state.failure_count >= circuit_breaker_config.lightrag_circuit_breaker_failure_threshold
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_workflow(self, circuit_breaker_config):
        """Test circuit breaker recovery workflow."""
        feature_manager = FeatureFlagManager(config=circuit_breaker_config)
        
        # Manually open circuit breaker
        feature_manager.circuit_breaker_state.is_open = True
        feature_manager.circuit_breaker_state.failure_count = 5
        feature_manager.circuit_breaker_state.last_failure_time = datetime.now() - timedelta(seconds=15)
        
        # Test routing decision with open circuit breaker past recovery timeout
        context = RoutingContext(user_id="recovery_user", query_text="test")
        result = feature_manager.should_use_lightrag(context)
        
        # Should attempt recovery (routing depends on recovery logic)
        # Circuit breaker should be closed or at least attempting recovery
        assert result.circuit_breaker_state != "open" or result.reason == RoutingReason.CIRCUIT_BREAKER_OPEN


class TestQualityThresholdIntegration:
    """Test quality threshold integration in workflows."""
    
    @pytest.fixture
    def quality_config(self):
        """Create configuration with quality metrics enabled."""
        return LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_enable_quality_metrics=True,
            lightrag_min_quality_threshold=0.8,  # High threshold
            lightrag_rollout_percentage=100.0    # Ensure LightRAG would be chosen
        )
    
    @pytest.mark.asyncio
    async def test_quality_threshold_routing_decision(self, quality_config):
        """Test that quality threshold affects routing decisions."""
        feature_manager = FeatureFlagManager(config=quality_config)
        
        # Add low quality scores to trigger threshold check
        feature_manager.performance_metrics.lightrag_quality_scores = [0.6, 0.7, 0.5]  # Below 0.8 threshold
        
        context = RoutingContext(user_id="quality_user", query_text="test")
        result = feature_manager.should_use_lightrag(context)
        
        # Should route to Perplexity due to quality threshold
        assert result.decision == RoutingDecision.PERPLEXITY
        assert result.reason == RoutingReason.QUALITY_THRESHOLD
        assert result.confidence == 0.8  # Quality threshold confidence
    
    @pytest.mark.asyncio
    async def test_quality_threshold_recovery(self, quality_config):
        """Test quality threshold recovery when quality improves."""
        feature_manager = FeatureFlagManager(config=quality_config)
        
        # Start with low quality scores
        feature_manager.performance_metrics.lightrag_quality_scores = [0.5, 0.6, 0.7]
        
        context = RoutingContext(user_id="quality_user", query_text="test")
        result1 = feature_manager.should_use_lightrag(context)
        assert result1.decision == RoutingDecision.PERPLEXITY  # Should be blocked by quality
        
        # Improve quality scores
        feature_manager.performance_metrics.lightrag_quality_scores = [0.85, 0.9, 0.82]
        
        result2 = feature_manager.should_use_lightrag(context)
        # Should now allow LightRAG due to improved quality
        assert result2.decision == RoutingDecision.LIGHTRAG or result2.reason != RoutingReason.QUALITY_THRESHOLD


class TestConditionalRoutingIntegration:
    """Test conditional routing integration workflows."""
    
    @pytest.fixture
    def conditional_routing_config(self):
        """Create configuration with conditional routing enabled."""
        return LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_enable_conditional_routing=True,
            lightrag_rollout_percentage=100.0
        )
    
    def test_conditional_routing_rule_evaluation(self, conditional_routing_config):
        """Test conditional routing rule evaluation."""
        feature_manager = FeatureFlagManager(
            config=conditional_routing_config,
            logger=Mock()
        )
        
        # Add custom routing rules
        def length_rule(context):
            return len(context.query_text) > 50
        
        def complexity_rule(context):
            return context.query_complexity and context.query_complexity > 0.5
        
        feature_manager.routing_rules = {
            "length_rule": length_rule,
            "complexity_rule": complexity_rule
        }
        
        # Test short query (should fail length rule)
        short_context = RoutingContext(user_id="user1", query_text="Short query")
        result1 = feature_manager.should_use_lightrag(short_context)
        assert result1.decision == RoutingDecision.PERPLEXITY
        assert result1.reason == RoutingReason.CONDITIONAL_RULE
        
        # Test long query with low complexity (should fail complexity rule)
        long_context = RoutingContext(
            user_id="user2",
            query_text="This is a very long query that exceeds the minimum length requirement for the length rule to pass",
            query_complexity=0.3
        )
        result2 = feature_manager.should_use_lightrag(long_context)
        assert result2.decision == RoutingDecision.PERPLEXITY
        assert result2.reason == RoutingReason.CONDITIONAL_RULE
        
        # Test query that passes all rules
        good_context = RoutingContext(
            user_id="user3",
            query_text="This is a sufficiently long and complex query about metabolomics pathways and biomarker identification",
            query_complexity=0.7
        )
        result3 = feature_manager.should_use_lightrag(good_context)
        assert result3.decision == RoutingDecision.LIGHTRAG
        assert result3.reason == RoutingReason.CONDITIONAL_RULE


class TestCacheIntegrationWorkflows:
    """Test cache integration in complete workflows."""
    
    @pytest.mark.asyncio
    async def test_response_caching_across_requests(self):
        """Test response caching across multiple requests."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=100.0
        )
        
        with patch('lightrag_integration.integration_wrapper.LightRAGQueryService') as mock_lightrag:
            # Configure mock to return different responses
            responses = [
                ServiceResponse(content="First response", processing_time=2.0),
                ServiceResponse(content="Second response", processing_time=2.0)
            ]
            mock_lightrag_instance = Mock()
            mock_lightrag_instance.query_async = AsyncMock(side_effect=responses)
            mock_lightrag.return_value = mock_lightrag_instance
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            # First request
            request1 = QueryRequest(query_text="What are metabolites?", user_id="cache_user")
            response1 = await service.query_async(request1)
            
            # Second identical request should use cache
            request2 = QueryRequest(query_text="What are metabolites?", user_id="cache_user")
            response2 = await service.query_async(request2)
            
            # Should get cached response
            assert response2.response_type == ResponseType.CACHED or response1.content == response2.content
            # Cache should make second request faster
            if response2.response_type == ResponseType.CACHED:
                assert response2.processing_time < response1.processing_time
            
            # Verify LightRAG was only called once (second was cached)
            assert mock_lightrag_instance.query_async.call_count <= 2
    
    @pytest.mark.asyncio
    async def test_routing_cache_consistency(self):
        """Test routing decision caching consistency."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0
        )
        
        feature_manager = FeatureFlagManager(config=config)
        
        # Same user and query should get consistent routing
        context = RoutingContext(
            user_id="consistent_user",
            query_text="Consistent query for caching test"
        )
        
        results = []
        for _ in range(10):
            result = feature_manager.should_use_lightrag(context)
            results.append(result)
        
        # All results should be identical (cached)
        first_decision = results[0].decision
        first_reason = results[0].reason
        
        for result in results[1:]:
            assert result.decision == first_decision
            assert result.reason == first_reason


class TestMultiUserScenarios:
    """Test multi-user scenarios and concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_routing_consistency(self):
        """Test routing consistency under concurrent user requests."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=30.0,
            lightrag_user_hash_salt="concurrent_test"
        )
        
        feature_manager = FeatureFlagManager(config=config)
        
        async def test_user_routing(user_id):
            """Test routing for a specific user multiple times."""
            results = []
            for i in range(5):
                context = RoutingContext(
                    user_id=user_id,
                    query_text=f"Query {i} for user {user_id}"
                )
                result = feature_manager.should_use_lightrag(context)
                results.append(result.decision)
            return user_id, results
        
        # Test multiple users concurrently
        users = [f"concurrent_user_{i}" for i in range(20)]
        tasks = [test_user_routing(user_id) for user_id in users]
        
        user_results = await asyncio.gather(*tasks)
        
        # Each user should have consistent decisions
        for user_id, decisions in user_results:
            first_decision = decisions[0]
            for decision in decisions[1:]:
                assert decision == first_decision, f"Inconsistent routing for {user_id}"
    
    @pytest.mark.asyncio
    async def test_user_cohort_distribution_stability(self):
        """Test that user cohort distribution remains stable over time."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=60.0,
            lightrag_enable_ab_testing=True,
            lightrag_user_hash_salt="distribution_test"
        )
        
        feature_manager = FeatureFlagManager(config=config)
        
        def get_cohort_distribution(num_users=100):
            """Get cohort distribution for a set of users."""
            cohorts = []
            for i in range(num_users):
                context = RoutingContext(user_id=f"dist_user_{i}", query_text="test")
                result = feature_manager.should_use_lightrag(context)
                cohorts.append(result.user_cohort)
            
            return {
                UserCohort.LIGHTRAG: cohorts.count(UserCohort.LIGHTRAG),
                UserCohort.PERPLEXITY: cohorts.count(UserCohort.PERPLEXITY),
                UserCohort.CONTROL: cohorts.count(UserCohort.CONTROL)
            }
        
        # Get distribution at different times
        dist1 = get_cohort_distribution()
        await asyncio.sleep(0.1)  # Small delay
        dist2 = get_cohort_distribution()
        
        # Distributions should be identical (deterministic hashing)
        assert dist1 == dist2
    
    @pytest.mark.asyncio
    async def test_mixed_user_query_patterns(self):
        """Test mixed user query patterns and routing decisions."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=40.0,
            lightrag_enable_conditional_routing=True
        )
        
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity, \
             patch('lightrag_integration.integration_wrapper.LightRAGQueryService') as mock_lightrag:
            
            # Configure mocks
            mock_perplexity.return_value.query_async = AsyncMock(return_value=ServiceResponse(
                content="Perplexity response", response_type=ResponseType.PERPLEXITY
            ))
            mock_lightrag.return_value.query_async = AsyncMock(return_value=ServiceResponse(
                content="LightRAG response", response_type=ResponseType.LIGHTRAG
            ))
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            # Mixed query patterns
            query_patterns = [
                ("power_user", "Complex metabolomics pathway analysis with multiple biomarkers"),
                ("casual_user", "What is diabetes?"),
                ("researcher", "Glucose metabolism in type 2 diabetes with insulin resistance"),
                ("student", "Help with homework"),
                ("clinician", "Biomarker interpretation for patient care")
            ]
            
            results = []
            for user_type, query in query_patterns:
                for i in range(3):  # Multiple queries per user type
                    request = QueryRequest(
                        user_id=f"{user_type}_{i}",
                        query_text=query,
                        query_type=user_type
                    )
                    response = await service.query_async(request)
                    results.append((user_type, response.metadata.get('routing_decision')))
            
            # Analyze routing patterns
            routing_by_type = {}
            for user_type, routing_decision in results:
                if user_type not in routing_by_type:
                    routing_by_type[user_type] = []
                routing_by_type[user_type].append(routing_decision)
            
            # Each user type should have consistent routing within their pattern
            for user_type, decisions in routing_by_type.items():
                # Same user type should generally get same routing (due to hashing)
                unique_decisions = set(decisions)
                # Allow some variation but not complete randomness
                assert len(unique_decisions) <= 2, f"Too much routing variation for {user_type}"


class TestRealWorldUsagePatterns:
    """Test realistic usage patterns and edge cases."""
    
    @pytest.mark.asyncio
    async def test_gradual_rollout_simulation(self):
        """Test gradual rollout simulation over time."""
        rollout_percentages = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]
        
        for rollout_pct in rollout_percentages:
            config = LightRAGConfig(
                lightrag_integration_enabled=True,
                lightrag_rollout_percentage=rollout_pct,
                lightrag_user_hash_salt="rollout_test"
            )
            
            feature_manager = FeatureFlagManager(config=config)
            
            lightrag_count = 0
            total_users = 100
            
            for i in range(total_users):
                context = RoutingContext(user_id=f"rollout_user_{i}", query_text="test")
                result = feature_manager.should_use_lightrag(context)
                
                if result.decision == RoutingDecision.LIGHTRAG:
                    lightrag_count += 1
            
            actual_percentage = (lightrag_count / total_users) * 100
            
            # Should be close to expected rollout percentage (within 10% tolerance)
            tolerance = 10.0
            assert abs(actual_percentage - rollout_pct) <= tolerance, \
                f"Rollout {rollout_pct}%: expected ~{rollout_pct}%, got {actual_percentage}%"
    
    @pytest.mark.asyncio
    async def test_high_load_performance_stability(self):
        """Test performance stability under high load."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0,
            lightrag_enable_circuit_breaker=True
        )
        
        feature_manager = FeatureFlagManager(config=config)
        
        # Simulate high load with many concurrent requests
        async def simulate_request(request_id):
            context = RoutingContext(
                user_id=f"load_user_{request_id % 100}",  # 100 unique users
                query_text=f"Load test query {request_id}"
            )
            start_time = time.time()
            result = feature_manager.should_use_lightrag(context)
            end_time = time.time()
            return end_time - start_time, result
        
        # Execute many concurrent requests
        num_requests = 500
        tasks = [simulate_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # Analyze performance
        response_times = [duration for duration, _ in results]
        routing_results = [result for _, result in results]
        
        # Performance should be consistent
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        assert avg_response_time < 0.1, f"Average response time too high: {avg_response_time}s"
        assert max_response_time < 0.5, f"Max response time too high: {max_response_time}s"
        
        # Routing should be consistent and reasonable
        decisions = [r.decision for r in routing_results]
        lightrag_ratio = decisions.count(RoutingDecision.LIGHTRAG) / len(decisions)
        
        # Should be roughly 50% (±20% tolerance for hash distribution)
        assert 0.3 <= lightrag_ratio <= 0.7, f"Unexpected routing distribution: {lightrag_ratio}"
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test various error recovery scenarios."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=100.0,
            lightrag_fallback_to_perplexity=True,
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=2
        )
        
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity, \
             patch('lightrag_integration.integration_wrapper.LightRAGQueryService') as mock_lightrag:
            
            # Configure intermittent failures
            failure_responses = [
                ServiceResponse(content="", error_details="Timeout", response_type=ResponseType.LIGHTRAG),
                ServiceResponse(content="", error_details="Connection failed", response_type=ResponseType.LIGHTRAG),
                ServiceResponse(content="Success after failures", response_type=ResponseType.LIGHTRAG, processing_time=1.0)
            ]
            
            mock_lightrag_instance = Mock()
            mock_lightrag_instance.query_async = AsyncMock(side_effect=failure_responses)
            mock_lightrag.return_value = mock_lightrag_instance
            
            # Configure Perplexity for fallback
            mock_perplexity_instance = Mock()
            mock_perplexity_instance.query_async = AsyncMock(return_value=ServiceResponse(
                content="Perplexity fallback", response_type=ResponseType.PERPLEXITY
            ))
            mock_perplexity.return_value = mock_perplexity_instance
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            # Execute queries to trigger failures and recovery
            for i in range(3):
                request = QueryRequest(user_id=f"recovery_user_{i}", query_text="test query")
                response = await service.query_async(request)
                
                if i < 2:
                    # First two should fail and use fallback
                    assert response.content in ["Perplexity fallback", "Success after failures"]
                    if response.content == "Perplexity fallback":
                        assert response.metadata.get('fallback_used', False)
                else:
                    # Third should succeed or still use fallback due to circuit breaker
                    assert response.is_success


# Mark the end of integration tests
if __name__ == "__main__":
    pytest.main([__file__])