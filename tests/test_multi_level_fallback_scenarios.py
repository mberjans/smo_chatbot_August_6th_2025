"""
Comprehensive Tests for Multi-Level Fallback Scenarios
======================================================

This test suite validates the multi-level fallback system implementation for the 
Clinical Metabolomics Oracle project, focusing specifically on the required
LightRAG → Perplexity → Cache fallback chain.

Test Categories:
- Multi-Level Fallback Chain Tests (LightRAG → Perplexity → Cache)
- Failure Simulation and Recovery Tests
- Performance and Latency Tests
- Edge Cases and Boundary Conditions
- Integration with Production Load Balancer
- Monitoring and Analytics Validation
- Error Propagation and Logging Tests

Author: Claude Code (Anthropic)
Task: CMO-LIGHTRAG-014-T01-TEST - Write tests for multi-level fallback scenarios
Created: August 9, 2025
"""

import pytest
import asyncio
import time
import threading
import json
import uuid
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

# Import the comprehensive fallback system components
try:
    from lightrag_integration.comprehensive_fallback_system import (
        FallbackOrchestrator,
        FallbackResult,
        FallbackLevel,
        FailureType,
        create_comprehensive_fallback_system
    )
    
    from lightrag_integration.enhanced_query_router_with_fallback import (
        EnhancedBiomedicalQueryRouter,
        FallbackIntegrationConfig,
        create_production_ready_enhanced_router
    )
    
    from lightrag_integration.production_intelligent_query_router import (
        ProductionIntelligentQueryRouter,
        DeploymentMode,
        ProductionFeatureFlags
    )
    
    from lightrag_integration.query_router import (
        BiomedicalQueryRouter,
        RoutingDecision,
        RoutingPrediction,
        ConfidenceMetrics
    )
    
    from lightrag_integration.research_categorizer import CategoryPrediction
    from lightrag_integration.cost_persistence import ResearchCategory
    
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


# ============================================================================
# TEST FIXTURES AND UTILITIES
# ============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_lightrag_backend():
    """Mock LightRAG backend that can simulate failures."""
    backend = Mock()
    backend.query = Mock()
    backend.is_healthy = Mock(return_value=True)
    backend.name = "lightrag"
    backend.failure_count = 0
    backend.response_time_ms = 800
    
    def mock_query(query_text, context=None):
        """Mock query that can simulate various failure conditions."""
        if backend.failure_count > 0:
            backend.failure_count -= 1
            if "timeout" in query_text.lower():
                time.sleep(2.0)  # Simulate timeout
                raise TimeoutError("LightRAG query timeout")
            elif "error" in query_text.lower():
                raise ConnectionError("LightRAG connection failed")
            elif "slow" in query_text.lower():
                time.sleep(1.5)
                backend.response_time_ms = 1500
        
        backend.response_time_ms = 800
        return create_mock_routing_prediction(
            confidence=0.85,
            routing=RoutingDecision.LIGHTRAG,
            backend="lightrag"
        )
    
    backend.query.side_effect = mock_query
    return backend


@pytest.fixture
def mock_perplexity_backend():
    """Mock Perplexity backend that can simulate failures."""
    backend = Mock()
    backend.query = Mock()
    backend.is_healthy = Mock(return_value=True)
    backend.name = "perplexity"
    backend.failure_count = 0
    backend.response_time_ms = 1200
    
    def mock_query(query_text, context=None):
        """Mock query that can simulate various failure conditions."""
        if backend.failure_count > 0:
            backend.failure_count -= 1
            if "timeout" in query_text.lower():
                time.sleep(3.0)  # Simulate timeout
                raise TimeoutError("Perplexity API timeout")
            elif "rate_limit" in query_text.lower():
                raise Exception("Rate limit exceeded")
            elif "error" in query_text.lower():
                raise Exception("Perplexity API error")
        
        backend.response_time_ms = 1200
        return create_mock_routing_prediction(
            confidence=0.75,
            routing=RoutingDecision.PERPLEXITY,
            backend="perplexity"
        )
    
    backend.query.side_effect = mock_query
    return backend


@pytest.fixture
def mock_cache_backend():
    """Mock cache backend that can simulate cache hits/misses."""
    backend = Mock()
    backend.get = Mock()
    backend.set = Mock()
    backend.is_available = Mock(return_value=True)
    backend.name = "cache"
    backend.cache_data = {}
    backend.response_time_ms = 50
    
    def mock_get(query_key):
        """Mock cache get with configurable hit/miss behavior."""
        # Simulate cache hit for common patterns
        common_patterns = [
            "what is metabolomics",
            "clinical metabolomics",
            "biomarker discovery",
            "pathway analysis"
        ]
        
        query_lower = query_key.lower()
        if any(pattern in query_lower for pattern in common_patterns):
            backend.response_time_ms = 50
            return create_mock_routing_prediction(
                confidence=0.30,  # Lower confidence from cache
                routing=RoutingDecision.EITHER,
                backend="cache"
            )
        return None  # Cache miss
    
    backend.get.side_effect = mock_get
    return backend


@pytest.fixture
def fallback_orchestrator(temp_cache_dir, mock_lightrag_backend, 
                         mock_perplexity_backend, mock_cache_backend):
    """Create fallback orchestrator with mocked backends."""
    config = {
        'emergency_cache_file': str(Path(temp_cache_dir) / "test_cache.pkl"),
        'lightrag_backend': mock_lightrag_backend,
        'perplexity_backend': mock_perplexity_backend,
        'cache_backend': mock_cache_backend,
        'performance_targets': {
            'max_response_time_ms': 2000,
            'min_confidence': 0.1,
            'target_success_rate': 0.99
        }
    }
    
    orchestrator = FallbackOrchestrator(config=config)
    
    # Integrate mock backends
    orchestrator.lightrag_backend = mock_lightrag_backend
    orchestrator.perplexity_backend = mock_perplexity_backend
    orchestrator.cache_backend = mock_cache_backend
    
    return orchestrator


@pytest.fixture
def enhanced_router(temp_cache_dir):
    """Create enhanced router for integration testing."""
    config = FallbackIntegrationConfig(
        emergency_cache_file=str(Path(temp_cache_dir) / "test_cache.pkl"),
        enable_monitoring=False,  # Disable for testing
        enable_fallback_system=True,
        max_response_time_ms=2000,
        confidence_threshold=0.5
    )
    
    return EnhancedBiomedicalQueryRouter(fallback_config=config)


def create_mock_routing_prediction(confidence=0.8, routing=RoutingDecision.EITHER, backend="mock"):
    """Create a mock routing prediction for testing."""
    confidence_metrics = ConfidenceMetrics(
        overall_confidence=confidence,
        research_category_confidence=confidence,
        temporal_analysis_confidence=confidence - 0.1,
        signal_strength_confidence=confidence - 0.05,
        context_coherence_confidence=confidence,
        keyword_density=0.4,
        pattern_match_strength=0.45,
        biomedical_entity_count=2,
        ambiguity_score=1.0 - confidence,
        conflict_score=0.3,
        alternative_interpretations=[
            (routing, confidence),
            (RoutingDecision.EITHER, 0.3)
        ],
        calculation_time_ms=25.0
    )
    
    prediction = RoutingPrediction(
        routing_decision=routing,
        confidence=confidence,
        reasoning=[f"Mock routing prediction from {backend}"],
        research_category=ResearchCategory.GENERAL_QUERY,
        confidence_metrics=confidence_metrics,
        temporal_indicators=[],
        knowledge_indicators=[],
        metadata={'backend': backend, 'mock': True}
    )
    
    return prediction


# ============================================================================
# MULTI-LEVEL FALLBACK CHAIN TESTS
# ============================================================================

class TestMultiLevelFallbackChain:
    """Test the complete LightRAG → Perplexity → Cache fallback chain."""
    
    def test_successful_lightrag_primary_route(self, fallback_orchestrator,
                                             mock_lightrag_backend):
        """Test successful processing through primary LightRAG route."""
        # Ensure LightRAG is healthy and responsive
        mock_lightrag_backend.failure_count = 0
        
        result = fallback_orchestrator.process_query_with_comprehensive_fallback(
            query_text="What are the metabolic pathways in diabetes?",
            context={'priority': 'normal'},
            priority='normal'
        )
        
        # Should succeed with primary LightRAG
        assert result.success is True
        assert result.fallback_level_used == FallbackLevel.FULL_LLM_WITH_CONFIDENCE
        assert result.routing_prediction.metadata['backend'] == 'lightrag'
        assert result.total_processing_time_ms < 1500
        assert result.confidence_degradation == 0.0  # No degradation on primary
        
        # Verify LightRAG was called
        mock_lightrag_backend.query.assert_called()
    
    def test_lightrag_failure_perplexity_fallback(self, fallback_orchestrator,
                                                 mock_lightrag_backend,
                                                 mock_perplexity_backend):
        """Test fallback to Perplexity when LightRAG fails."""
        # Simulate LightRAG failure
        mock_lightrag_backend.failure_count = 1
        
        result = fallback_orchestrator.process_query_with_comprehensive_fallback(
            query_text="Recent advances in metabolomics error testing",
            context={'priority': 'normal'},
            priority='normal'
        )
        
        # Should succeed with Perplexity fallback
        assert result.success is True
        assert result.fallback_level_used in [FallbackLevel.SIMPLIFIED_LLM, FallbackLevel.KEYWORD_BASED_ONLY]
        assert result.routing_prediction.metadata.get('backend') in ['perplexity', 'cache']
        assert len(result.attempted_levels) >= 2  # Attempted multiple levels
        assert FailureType.API_ERROR in result.failure_reasons or \
               FailureType.SERVICE_UNAVAILABLE in result.failure_reasons
        
        # Should have some confidence degradation but still be useful
        assert 0.0 < result.confidence_degradation <= 0.5
    
    def test_lightrag_perplexity_both_fail_cache_fallback(self, fallback_orchestrator,
                                                        mock_lightrag_backend,
                                                        mock_perplexity_backend,
                                                        mock_cache_backend):
        """Test fallback to cache when both LightRAG and Perplexity fail."""
        # Simulate both primary backends failing
        mock_lightrag_backend.failure_count = 1
        mock_perplexity_backend.failure_count = 1
        
        # Use a query that should be in cache
        result = fallback_orchestrator.process_query_with_comprehensive_fallback(
            query_text="What is metabolomics?",  # Common pattern, should be cached
            context={'priority': 'normal'},
            priority='normal'
        )
        
        # Should succeed with cache fallback
        assert result.success is True
        assert result.fallback_level_used == FallbackLevel.EMERGENCY_CACHE
        assert len(result.attempted_levels) >= 3  # Attempted all three levels
        
        # Should have moderate confidence degradation but still be functional
        assert 0.3 <= result.confidence_degradation <= 0.7
        assert result.routing_prediction.confidence >= 0.1  # Still has some confidence
        
        # Should have cached the failure reasons
        failure_types = result.failure_reasons
        assert len(failure_types) >= 2  # Multiple failures recorded
    
    def test_complete_fallback_chain_failure_default_routing(self, fallback_orchestrator,
                                                           mock_lightrag_backend,
                                                           mock_perplexity_backend,
                                                           mock_cache_backend):
        """Test default routing when entire fallback chain fails."""
        # Simulate all backends failing
        mock_lightrag_backend.failure_count = 1
        mock_perplexity_backend.failure_count = 1
        mock_cache_backend.is_available.return_value = False
        
        # Use an uncommon query that won't be in cache
        result = fallback_orchestrator.process_query_with_comprehensive_fallback(
            query_text="Extremely specific rare metabolite error timeout analysis",
            context={'priority': 'normal'},
            priority='normal'
        )
        
        # Should succeed with default routing (system always provides a response)
        assert result.success is True
        assert result.fallback_level_used == FallbackLevel.DEFAULT_ROUTING
        assert len(result.attempted_levels) >= 4  # Attempted all levels
        
        # Should have significant confidence degradation but still functional
        assert result.confidence_degradation >= 0.5
        assert result.routing_prediction.confidence >= 0.05  # Minimal but non-zero confidence
        
        # Should record all failure types
        assert len(result.failure_reasons) >= 3
        
        # Performance should still be reasonable even in worst case
        assert result.total_processing_time_ms < 5000
    
    def test_fallback_chain_with_timeouts(self, fallback_orchestrator,
                                        mock_lightrag_backend,
                                        mock_perplexity_backend):
        """Test fallback chain behavior with timeout conditions."""
        # Set up timeout scenarios
        mock_lightrag_backend.failure_count = 1  # Will timeout
        mock_perplexity_backend.failure_count = 1  # Will timeout
        
        start_time = time.time()
        result = fallback_orchestrator.process_query_with_comprehensive_fallback(
            query_text="Metabolomics timeout analysis",
            context={'priority': 'high', 'max_wait_time': 3000},
            priority='high'
        )
        total_time = (time.time() - start_time) * 1000
        
        # Should handle timeouts gracefully
        assert result.success is True
        assert FailureType.API_TIMEOUT in result.failure_reasons
        
        # Should not take too long despite timeouts
        assert total_time < 4000  # Should have reasonable timeout handling
        
        # Should eventually succeed with cache or default routing
        assert result.fallback_level_used in [
            FallbackLevel.EMERGENCY_CACHE,
            FallbackLevel.DEFAULT_ROUTING
        ]
    
    def test_fallback_performance_characteristics(self, fallback_orchestrator):
        """Test performance characteristics of different fallback levels."""
        test_queries = [
            {
                'query': 'Primary route success test',
                'expected_max_time': 1000,
                'lightrag_failures': 0,
                'perplexity_failures': 0
            },
            {
                'query': 'Secondary route error fallback test',
                'expected_max_time': 2000,
                'lightrag_failures': 1,
                'perplexity_failures': 0
            },
            {
                'query': 'What is metabolomics error testing?',  # Cache hit
                'expected_max_time': 1500,
                'lightrag_failures': 1,
                'perplexity_failures': 1
            }
        ]
        
        performance_results = []
        
        for test_case in test_queries:
            # Set up failure conditions
            fallback_orchestrator.lightrag_backend.failure_count = test_case['lightrag_failures']
            fallback_orchestrator.perplexity_backend.failure_count = test_case['perplexity_failures']
            
            start_time = time.time()
            result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=test_case['query'],
                context={'performance_test': True},
                priority='normal'
            )
            processing_time = (time.time() - start_time) * 1000
            
            performance_results.append({
                'query': test_case['query'],
                'processing_time_ms': processing_time,
                'expected_max_time': test_case['expected_max_time'],
                'success': result.success,
                'fallback_level': result.fallback_level_used,
                'confidence_degradation': result.confidence_degradation
            })
            
            # Performance assertions
            assert result.success is True, f"Query failed: {test_case['query']}"
            assert processing_time <= test_case['expected_max_time'], \
                f"Too slow ({processing_time}ms > {test_case['expected_max_time']}ms): {test_case['query']}"
        
        # Verify performance scaling is reasonable
        primary_time = next(r['processing_time_ms'] for r in performance_results 
                          if 'Primary' in r['query'])
        fallback_time = next(r['processing_time_ms'] for r in performance_results 
                           if 'Secondary' in r['query'])
        
        # Fallback shouldn't be more than 3x slower than primary
        assert fallback_time <= primary_time * 3, \
            f"Fallback too slow compared to primary: {fallback_time} vs {primary_time}"


# ============================================================================
# FAILURE SIMULATION AND RECOVERY TESTS
# ============================================================================

class TestFailureSimulationAndRecovery:
    """Test various failure scenarios and recovery mechanisms."""
    
    def test_intermittent_failure_recovery(self, fallback_orchestrator,
                                         mock_lightrag_backend):
        """Test recovery from intermittent failures."""
        results = []
        
        # Simulate intermittent failures
        failure_pattern = [1, 0, 1, 0, 0, 1, 0, 0, 0]  # 1 = fail, 0 = succeed
        
        for i, should_fail in enumerate(failure_pattern):
            mock_lightrag_backend.failure_count = should_fail
            
            result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=f"Intermittent test query {i+1}",
                context={'test_sequence': i},
                priority='normal'
            )
            
            results.append({
                'index': i,
                'should_fail_primary': should_fail,
                'success': result.success,
                'fallback_level': result.fallback_level_used,
                'confidence_degradation': result.confidence_degradation
            })
        
        # All queries should ultimately succeed
        assert all(r['success'] for r in results)
        
        # Should use primary route when available
        primary_successes = [r for r in results if not r['should_fail_primary']]
        fallback_uses = [r for r in results if r['should_fail_primary']]
        
        # When primary should work, should mostly use primary level
        primary_level_uses = [r for r in primary_successes 
                            if r['fallback_level'] == FallbackLevel.FULL_LLM_WITH_CONFIDENCE]
        assert len(primary_level_uses) >= len(primary_successes) * 0.7  # Most should succeed
        
        # When primary fails, should use fallback levels
        fallback_level_uses = [r for r in fallback_uses 
                             if r['fallback_level'] != FallbackLevel.FULL_LLM_WITH_CONFIDENCE]
        assert len(fallback_level_uses) >= len(fallback_uses) * 0.8  # Most should fallback
    
    def test_cascading_failure_scenarios(self, fallback_orchestrator,
                                       mock_lightrag_backend,
                                       mock_perplexity_backend):
        """Test cascading failure scenarios across multiple backends."""
        # Progressive failure scenario
        cascading_scenarios = [
            {'lightrag_fail': 0, 'perplexity_fail': 0, 'expected_level': FallbackLevel.FULL_LLM_WITH_CONFIDENCE},
            {'lightrag_fail': 1, 'perplexity_fail': 0, 'expected_min_level': FallbackLevel.SIMPLIFIED_LLM},
            {'lightrag_fail': 1, 'perplexity_fail': 1, 'expected_min_level': FallbackLevel.KEYWORD_BASED_ONLY},
        ]
        
        for i, scenario in enumerate(cascading_scenarios):
            mock_lightrag_backend.failure_count = scenario['lightrag_fail']
            mock_perplexity_backend.failure_count = scenario['perplexity_fail']
            
            result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=f"Cascading failure scenario {i+1}: pathway analysis",
                context={'scenario': i, 'cascading_test': True},
                priority='normal'
            )
            
            assert result.success is True, f"Scenario {i+1} failed"
            
            # Check expected fallback level
            if 'expected_level' in scenario:
                assert result.fallback_level_used == scenario['expected_level'], \
                    f"Scenario {i+1}: expected {scenario['expected_level']}, got {result.fallback_level_used}"
            elif 'expected_min_level' in scenario:
                assert result.fallback_level_used.value >= scenario['expected_min_level'].value, \
                    f"Scenario {i+1}: expected >= {scenario['expected_min_level']}, got {result.fallback_level_used}"
            
            # Confidence should degrade appropriately with failures
            expected_degradation = (scenario['lightrag_fail'] + scenario['perplexity_fail']) * 0.2
            assert result.confidence_degradation >= expected_degradation, \
                f"Insufficient degradation for scenario {i+1}: {result.confidence_degradation} < {expected_degradation}"
    
    def test_rapid_successive_failures(self, fallback_orchestrator):
        """Test system behavior under rapid successive failures."""
        num_rapid_queries = 15
        max_total_time = 10.0  # 10 seconds for all queries
        
        # Set up high failure rate
        fallback_orchestrator.lightrag_backend.failure_count = 5
        fallback_orchestrator.perplexity_backend.failure_count = 5
        
        start_time = time.time()
        results = []
        
        for i in range(num_rapid_queries):
            result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=f"Rapid failure test {i+1} error analysis",
                context={'rapid_test': i, 'batch_processing': True},
                priority='high' if i % 3 == 0 else 'normal'  # Vary priority
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Performance constraints
        assert total_time <= max_total_time, \
            f"Rapid failure test took too long: {total_time}s > {max_total_time}s"
        
        # All queries should eventually succeed
        assert all(r.success for r in results), "Some rapid failure queries failed"
        
        # System should adapt to failures
        later_results = results[num_rapid_queries//2:]  # Second half
        fallback_levels = [r.fallback_level_used for r in later_results]
        
        # Should predominantly use fallback levels after learning about failures
        emergency_or_default = [level for level in fallback_levels 
                              if level in [FallbackLevel.EMERGENCY_CACHE, FallbackLevel.DEFAULT_ROUTING]]
        assert len(emergency_or_default) >= len(later_results) * 0.5, \
            "Should adapt to use fallback levels more frequently"
        
        # Average response time should be reasonable
        avg_response_time = sum(r.total_processing_time_ms for r in results) / len(results)
        assert avg_response_time <= 800, f"Average response time too high: {avg_response_time}ms"
    
    def test_recovery_after_extended_outage(self, fallback_orchestrator,
                                          mock_lightrag_backend):
        """Test recovery behavior after extended backend outage."""
        # Simulate extended outage
        mock_lightrag_backend.failure_count = 10  # Many failures
        mock_lightrag_backend.is_healthy.return_value = False
        
        # Process queries during outage
        outage_results = []
        for i in range(5):
            result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=f"Outage query {i+1} during extended failure",
                context={'outage_test': True},
                priority='normal'
            )
            outage_results.append(result)
        
        # All should succeed via fallback
        assert all(r.success for r in outage_results)
        fallback_levels = [r.fallback_level_used for r in outage_results]
        assert all(level != FallbackLevel.FULL_LLM_WITH_CONFIDENCE for level in fallback_levels)
        
        # Simulate recovery
        mock_lightrag_backend.failure_count = 0
        mock_lightrag_backend.is_healthy.return_value = True
        
        # Test queries after recovery
        recovery_results = []
        for i in range(5):
            result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=f"Recovery query {i+1} after outage",
                context={'recovery_test': True},
                priority='normal'
            )
            recovery_results.append(result)
        
        # Should gradually return to primary route
        primary_uses = [r for r in recovery_results 
                       if r.fallback_level_used == FallbackLevel.FULL_LLM_WITH_CONFIDENCE]
        
        # At least some queries should return to primary (gradual recovery)
        assert len(primary_uses) >= 2, "Should show recovery to primary route"
        
        # Later queries should show better recovery
        later_recovery = recovery_results[2:]  # Last 3 queries
        later_primary = [r for r in later_recovery 
                        if r.fallback_level_used == FallbackLevel.FULL_LLM_WITH_CONFIDENCE]
        assert len(later_primary) >= 1, "Later queries should show recovery"


# ============================================================================
# INTEGRATION TESTS WITH PRODUCTION COMPONENTS
# ============================================================================

class TestProductionIntegration:
    """Test integration with production load balancer and routing components."""
    
    def test_enhanced_router_fallback_integration(self, enhanced_router):
        """Test integration of fallback system with enhanced router."""
        # Test various query types with different confidence levels
        test_queries = [
            {
                'query': 'What is clinical metabolomics?',
                'expected_success': True,
                'max_time_ms': 1000
            },
            {
                'query': 'Complex pathway interaction analysis with failures',
                'expected_success': True,
                'max_time_ms': 2000
            },
            {
                'query': 'Recent advances in metabolomics research',
                'expected_success': True,
                'max_time_ms': 1500
            }
        ]
        
        for test_case in test_queries:
            start_time = time.time()
            
            # Use enhanced router's route_query method
            result = enhanced_router.route_query(
                query_text=test_case['query'],
                context={'integration_test': True}
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Verify integration success
            assert result is not None, f"No result for query: {test_case['query']}"
            assert processing_time <= test_case['max_time_ms'], \
                f"Too slow: {processing_time}ms > {test_case['max_time_ms']}ms"
            
            # Check for fallback metadata
            if result.metadata:
                fallback_info = result.metadata.get('fallback_system_available', False)
                assert fallback_info is not False, "Fallback system should be available"
                
                # If fallback was used, should have additional metadata
                if result.metadata.get('fallback_system_used', False):
                    assert 'fallback_level_used' in result.metadata
                    assert 'total_fallback_time_ms' in result.metadata
    
    def test_production_router_fallback_compatibility(self, temp_cache_dir):
        """Test compatibility with production intelligent query router."""
        # Create production router with fallback features
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.CANARY,
            production_traffic_percentage=50.0,
            enable_automatic_failback=True
        )
        
        # Mock existing router for base functionality
        mock_base_router = Mock(spec=BiomedicalQueryRouter)
        mock_base_router.route_query.return_value = create_mock_routing_prediction(0.8)
        
        production_router = ProductionIntelligentQueryRouter(
            base_router=mock_base_router,
            feature_flags=feature_flags
        )
        
        # Test fallback scenarios
        test_scenarios = [
            {
                'query': 'Production fallback test query',
                'context': {'production_test': True},
                'expected_success': True
            },
            {
                'query': 'Production error handling test',
                'context': {'error_simulation': True},
                'expected_success': True
            }
        ]
        
        for scenario in test_scenarios:
            try:
                result = asyncio.run(production_router.route_query(
                    query_text=scenario['query'],
                    context=scenario['context']
                ))
                
                assert result is not None, f"No result for: {scenario['query']}"
                assert hasattr(result, 'routing_decision'), "Should have routing decision"
                assert hasattr(result, 'confidence'), "Should have confidence"
                
                # Should have load balancer metrics if available
                if hasattr(result, 'load_balancer_metrics'):
                    metrics = result.load_balancer_metrics
                    assert isinstance(metrics, dict), "Metrics should be dict"
                
            except Exception as e:
                # Some failures are acceptable in production testing
                # as long as they're handled gracefully
                assert "timeout" in str(e).lower() or "connection" in str(e).lower(), \
                    f"Unexpected error type: {e}"
    
    def test_fallback_analytics_and_monitoring(self, enhanced_router):
        """Test fallback system analytics and monitoring integration."""
        # Process multiple queries to generate analytics data
        queries = [
            "Analytics test query 1",
            "Analytics test query 2 with potential errors",
            "What is metabolomics analytics test?",  # Cache hit
            "Analytics test query 4",
            "Analytics test query 5 with complexity"
        ]
        
        for query in queries:
            result = enhanced_router.route_query(query, {'analytics_test': True})
            assert result is not None
        
        # Get enhanced statistics
        stats = enhanced_router.get_enhanced_routing_statistics()
        
        # Verify analytics structure
        assert 'enhanced_router_stats' in stats
        assert 'fallback_system_enabled' in stats
        assert stats['fallback_system_enabled'] is True
        
        enhanced_stats = stats['enhanced_router_stats']
        assert 'total_enhanced_queries' in enhanced_stats
        assert enhanced_stats['total_enhanced_queries'] >= len(queries)
        
        # Check for fallback-specific metrics
        if 'fallback_system_stats' in stats:
            fallback_stats = stats['fallback_system_stats']
            assert isinstance(fallback_stats, dict), "Fallback stats should be dict"
            
            # Should have comprehensive metrics
            if 'comprehensive_metrics' in fallback_stats:
                comprehensive = fallback_stats['comprehensive_metrics']
                assert 'integration_effectiveness' in comprehensive or \
                       'fallback_orchestrator' in comprehensive
        
        # Get system health report
        health_report = enhanced_router.get_system_health_report()
        assert 'enhanced_router_operational' in health_report
        assert health_report['enhanced_router_operational'] is True
        assert 'fallback_system_status' in health_report


# ============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================

class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for fallback scenarios."""
    
    def test_extreme_load_conditions(self, fallback_orchestrator):
        """Test fallback behavior under extreme load conditions."""
        num_concurrent_queries = 20
        max_concurrent_time = 5.0  # 5 seconds for all concurrent queries
        
        # Set up challenging conditions
        fallback_orchestrator.lightrag_backend.failure_count = 3
        fallback_orchestrator.perplexity_backend.failure_count = 2
        
        def process_query(query_id):
            """Process a single query with timing."""
            start_time = time.time()
            result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=f"Extreme load test query {query_id}",
                context={'extreme_load_test': True, 'query_id': query_id},
                priority='normal'
            )
            processing_time = time.time() - start_time
            return {
                'query_id': query_id,
                'success': result.success,
                'processing_time': processing_time,
                'fallback_level': result.fallback_level_used,
                'confidence_degradation': result.confidence_degradation
            }
        
        # Run concurrent queries
        import concurrent.futures
        
        overall_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_query = {
                executor.submit(process_query, i): i 
                for i in range(num_concurrent_queries)
            }
            
            results = []
            for future in concurrent.futures.as_completed(future_to_query, timeout=max_concurrent_time):
                query_id = future_to_query[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'query_id': query_id,
                        'success': False,
                        'error': str(e),
                        'processing_time': max_concurrent_time
                    })
        
        overall_time = time.time() - overall_start
        
        # Verify extreme load handling
        assert len(results) == num_concurrent_queries, "Should process all queries"
        assert overall_time <= max_concurrent_time + 1, "Should complete within time limit"
        
        # Most queries should succeed even under extreme load
        successful_queries = [r for r in results if r.get('success', False)]
        success_rate = len(successful_queries) / len(results)
        assert success_rate >= 0.8, f"Success rate too low under load: {success_rate:.2%}"
        
        # Average processing time should be reasonable
        processing_times = [r['processing_time'] for r in successful_queries]
        avg_time = sum(processing_times) / len(processing_times)
        assert avg_time <= 2.0, f"Average processing time too high: {avg_time}s"
    
    def test_memory_pressure_conditions(self, enhanced_router):
        """Test fallback behavior under memory pressure."""
        # Process many queries to simulate memory pressure
        memory_test_queries = 50
        
        # Monitor memory usage (simplified)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = []
        for i in range(memory_test_queries):
            query_text = f"Memory pressure test query {i+1} " * 10  # Longer queries
            result = enhanced_router.route_query(
                query_text=query_text,
                context={'memory_test': True, 'query_id': i}
            )
            results.append(result)
            
            # Check memory growth periodically
            if i % 10 == 9:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable (under 100MB)
                assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"
        
        # All queries should succeed
        assert all(r is not None for r in results), "Some queries failed under memory pressure"
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        assert total_growth < 150, f"Total memory growth too high: {total_growth:.1f}MB"
    
    def test_boundary_confidence_scenarios(self, fallback_orchestrator):
        """Test fallback behavior at confidence boundaries."""
        # Test queries at various confidence boundaries
        boundary_scenarios = [
            {'confidence': 0.99, 'description': 'near_perfect'},
            {'confidence': 0.90, 'description': 'high_confidence'},
            {'confidence': 0.70, 'description': 'medium_high'},
            {'confidence': 0.50, 'description': 'boundary_threshold'},
            {'confidence': 0.30, 'description': 'low_confidence'},
            {'confidence': 0.10, 'description': 'very_low'},
            {'confidence': 0.01, 'description': 'minimal_confidence'}
        ]
        
        for scenario in boundary_scenarios:
            # Create a result with controlled confidence
            with patch('lightrag_integration.comprehensive_fallback_system.FallbackOrchestrator') as mock_orchestrator:
                mock_result = Mock()
                mock_result.success = True
                mock_result.routing_prediction = create_mock_routing_prediction(scenario['confidence'])
                mock_result.fallback_level_used = FallbackLevel.FULL_LLM_WITH_CONFIDENCE
                mock_result.total_processing_time_ms = 500
                mock_result.confidence_degradation = 0.0
                mock_result.failure_reasons = []
                
                mock_orchestrator.return_value.process_query_with_comprehensive_fallback.return_value = mock_result
                
                result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                    query_text=f"Boundary test query with {scenario['description']} confidence",
                    context={'boundary_test': scenario['description']},
                    priority='normal'
                )
                
                # Should handle all confidence levels appropriately
                assert result.success is True, f"Failed for {scenario['description']}"
                
                # Very low confidence should trigger fallback considerations
                if scenario['confidence'] <= 0.3:
                    # System may choose to use fallback levels for safety
                    assert result.routing_prediction.confidence > 0, \
                        f"Should maintain some confidence for {scenario['description']}"
    
    def test_rapid_backend_state_changes(self, fallback_orchestrator,
                                       mock_lightrag_backend,
                                       mock_perplexity_backend):
        """Test handling of rapid backend state changes."""
        # Simulate rapidly changing backend states
        state_sequence = [
            {'lightrag_healthy': True, 'perplexity_healthy': True},
            {'lightrag_healthy': False, 'perplexity_healthy': True},
            {'lightrag_healthy': False, 'perplexity_healthy': False},
            {'lightrag_healthy': True, 'perplexity_healthy': False},
            {'lightrag_healthy': True, 'perplexity_healthy': True},
        ]
        
        results = []
        
        for i, state in enumerate(state_sequence):
            # Configure backend states
            mock_lightrag_backend.is_healthy.return_value = state['lightrag_healthy']
            mock_lightrag_backend.failure_count = 0 if state['lightrag_healthy'] else 1
            
            mock_perplexity_backend.is_healthy.return_value = state['perplexity_healthy']
            mock_perplexity_backend.failure_count = 0 if state['perplexity_healthy'] else 1
            
            result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=f"Rapid state change test {i+1}",
                context={'state_test': i, 'state_config': state},
                priority='normal'
            )
            
            results.append({
                'state_index': i,
                'state': state,
                'success': result.success,
                'fallback_level': result.fallback_level_used,
                'confidence_degradation': result.confidence_degradation,
                'processing_time_ms': result.total_processing_time_ms
            })
        
        # All queries should succeed despite rapid state changes
        assert all(r['success'] for r in results), "Some queries failed during rapid state changes"
        
        # Should adapt to state changes appropriately
        healthy_states = [r for r in results if r['state']['lightrag_healthy'] and r['state']['perplexity_healthy']]
        unhealthy_states = [r for r in results if not (r['state']['lightrag_healthy'] or r['state']['perplexity_healthy'])]
        
        if healthy_states and unhealthy_states:
            # Healthy states should generally have lower degradation
            avg_healthy_degradation = sum(r['confidence_degradation'] for r in healthy_states) / len(healthy_states)
            avg_unhealthy_degradation = sum(r['confidence_degradation'] for r in unhealthy_states) / len(unhealthy_states)
            
            assert avg_unhealthy_degradation >= avg_healthy_degradation, \
                "Unhealthy states should have higher degradation"
        
        # Processing times should remain reasonable despite state changes
        max_processing_time = max(r['processing_time_ms'] for r in results)
        assert max_processing_time < 2000, f"Processing time too high during state changes: {max_processing_time}ms"


# ============================================================================
# MONITORING AND ANALYTICS VALIDATION
# ============================================================================

class TestMonitoringAndAnalyticsValidation:
    """Test monitoring and analytics functionality for fallback scenarios."""
    
    def test_fallback_decision_logging(self, enhanced_router):
        """Test that fallback decisions are properly logged and tracked."""
        # Process queries to generate decision logs
        test_queries = [
            "Logging test query 1 - standard processing",
            "Logging test query 2 - potential error scenario",
            "What is metabolomics?",  # Should hit cache
            "Logging test query 4 - complex analysis"
        ]
        
        with patch('logging.Logger') as mock_logger:
            for query in test_queries:
                result = enhanced_router.route_query(
                    query,
                    context={'decision_logging_test': True}
                )
                assert result is not None
            
            # Verify logging occurred
            assert mock_logger.called or any(
                call.args for call in mock_logger.return_value.info.call_args_list
                if 'fallback' in str(call.args).lower()
            ), "Fallback decisions should be logged"
    
    def test_performance_metrics_collection(self, fallback_orchestrator):
        """Test collection of performance metrics during fallback scenarios."""
        # Generate various performance scenarios
        performance_scenarios = [
            {'query': 'Fast query test', 'expected_fast': True},
            {'query': 'Slow query test with processing delay', 'expected_fast': False},
            {'query': 'What is metabolomics performance test?', 'expected_fast': True}  # Cache hit
        ]
        
        for scenario in performance_scenarios:
            start_time = time.time()
            result = fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=scenario['query'],
                context={'performance_metrics_test': True},
                priority='normal'
            )
            actual_time = (time.time() - start_time) * 1000
            
            # Verify timing information is captured
            assert result.total_processing_time_ms > 0, "Should capture processing time"
            assert abs(result.total_processing_time_ms - actual_time) < 100, \
                "Captured time should be close to actual time"
            
            # Verify level-specific timing
            if hasattr(result, 'level_processing_times'):
                level_times = result.level_processing_times
                assert len(level_times) > 0, "Should capture level-specific timings"
                total_level_time = sum(level_times.values())
                assert total_level_time <= result.total_processing_time_ms + 50, \
                    "Level times should sum to less than or equal to total time"
    
    def test_failure_pattern_analysis(self, enhanced_router):
        """Test analysis of failure patterns and trends."""
        # Simulate various failure patterns
        failure_patterns = [
            # Intermittent failures
            [True, False, True, False, False],
            # Cascading failures
            [True, True, True, False, False],
            # Recovery pattern
            [True, True, False, False, False]
        ]
        
        pattern_results = []
        
        for pattern_idx, failure_pattern in enumerate(failure_patterns):
            pattern_start = time.time()
            
            for query_idx, should_simulate_issues in enumerate(failure_pattern):
                query_suffix = " error test" if should_simulate_issues else " success test"
                query_text = f"Pattern {pattern_idx+1} query {query_idx+1}{query_suffix}"
                
                result = enhanced_router.route_query(
                    query_text,
                    context={
                        'pattern_analysis_test': True,
                        'pattern_id': pattern_idx,
                        'query_in_pattern': query_idx,
                        'simulated_issue': should_simulate_issues
                    }
                )
                
                pattern_results.append({
                    'pattern_id': pattern_idx,
                    'query_id': query_idx,
                    'simulated_issue': should_simulate_issues,
                    'success': result is not None and hasattr(result, 'routing_decision'),
                    'confidence': getattr(result, 'confidence', 0) if result else 0
                })
        
        # Analyze patterns
        for pattern_idx in range(len(failure_patterns)):
            pattern_queries = [r for r in pattern_results if r['pattern_id'] == pattern_idx]
            
            # All queries should ultimately succeed
            success_rate = sum(1 for r in pattern_queries if r['success']) / len(pattern_queries)
            assert success_rate >= 0.8, f"Pattern {pattern_idx+1} success rate too low: {success_rate:.2%}"
            
            # Confidence should adapt to issues
            issue_queries = [r for r in pattern_queries if r['simulated_issue']]
            normal_queries = [r for r in pattern_queries if not r['simulated_issue']]
            
            if issue_queries and normal_queries:
                avg_issue_confidence = sum(r['confidence'] for r in issue_queries) / len(issue_queries)
                avg_normal_confidence = sum(r['confidence'] for r in normal_queries) / len(normal_queries)
                
                # Issues may result in different confidence patterns but both should be reasonable
                assert avg_issue_confidence >= 0.1, "Issue queries should maintain minimal confidence"
                assert avg_normal_confidence >= 0.1, "Normal queries should maintain minimal confidence"
    
    def test_system_health_reporting(self, enhanced_router):
        """Test comprehensive system health reporting."""
        # Process queries to establish baseline
        baseline_queries = [f"Health baseline query {i+1}" for i in range(5)]
        
        for query in baseline_queries:
            result = enhanced_router.route_query(
                query,
                context={'health_reporting_test': True}
            )
            assert result is not None
        
        # Get initial health report
        initial_health = enhanced_router.get_system_health_report()
        
        # Verify health report structure
        assert 'enhanced_router_operational' in initial_health
        assert initial_health['enhanced_router_operational'] is True
        assert 'fallback_system_status' in initial_health
        assert 'timestamp' in initial_health
        
        # Health status should be meaningful
        if 'system_health_score' in initial_health:
            health_score = initial_health['system_health_score']
            assert isinstance(health_score, (int, float)), "Health score should be numeric"
            assert 0 <= health_score <= 1, "Health score should be between 0 and 1"
        
        # Process queries with potential issues to change health
        issue_queries = [f"Health issue query {i+1} error test" for i in range(3)]
        
        for query in issue_queries:
            result = enhanced_router.route_query(
                query,
                context={'health_issue_test': True}
            )
            assert result is not None
        
        # Get updated health report
        updated_health = enhanced_router.get_system_health_report()
        
        # Should still be operational
        assert updated_health['enhanced_router_operational'] is True
        
        # Timestamp should be updated
        assert updated_health['timestamp'] > initial_health['timestamp']
        
        # Should have activity metrics
        if 'fallback_activations' in updated_health:
            activations = updated_health['fallback_activations']
            assert isinstance(activations, int), "Fallback activations should be integer"
            assert activations >= 0, "Activations should be non-negative"


if __name__ == '__main__':
    # Configure pytest to run with detailed output
    pytest.main([__file__, '-v', '--tb=short', '--durations=10'])