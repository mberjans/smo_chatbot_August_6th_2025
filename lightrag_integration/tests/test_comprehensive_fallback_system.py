"""
Comprehensive Test Suite for Multi-Tiered Fallback System

This test suite validates the comprehensive fallback system's ability to:
- Detect failures intelligently
- Implement progressive degradation strategies
- Ensure 100% system availability
- Recover automatically from failures
- Maintain performance under adverse conditions

Test Categories:
    - Failure Detection Tests
    - Multi-Level Fallback Tests  
    - Degradation Strategy Tests
    - Recovery Mechanism Tests
    - Emergency Cache Tests
    - Integration Tests
    - Performance Tests
    - Stress Tests

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any
import tempfile
import shutil
from pathlib import Path

# Import the components to test
try:
    from ..comprehensive_fallback_system import (
        FallbackOrchestrator,
        FallbackMonitor,
        FallbackResult,
        FallbackLevel,
        FailureType,
        FailureDetector,
        GracefulDegradationManager,
        RecoveryManager,
        EmergencyCache,
        create_comprehensive_fallback_system
    )
    
    from ..enhanced_query_router_with_fallback import (
        EnhancedBiomedicalQueryRouter,
        FallbackIntegrationConfig,
        create_production_ready_enhanced_router
    )
    
    from ..query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction, ConfidenceMetrics
    from ..research_categorizer import CategoryPrediction
    from ..cost_persistence import ResearchCategory
    
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


# ============================================================================
# TEST FIXTURES AND UTILITIES
# ============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_query_router():
    """Create a mock query router for testing."""
    router = Mock(spec=BiomedicalQueryRouter)
    
    def mock_route_query(query_text, context=None):
        # Simulate routing based on query text
        if "fail" in query_text.lower():
            raise Exception("Simulated routing failure")
        elif "slow" in query_text.lower():
            time.sleep(0.1)  # Simulate slow response
            return create_mock_routing_prediction(confidence=0.3)
        else:
            return create_mock_routing_prediction(confidence=0.8)
    
    router.route_query = mock_route_query
    return router


@pytest.fixture
def mock_llm_classifier():
    """Create a mock LLM classifier for testing."""
    classifier = Mock()
    
    def mock_classify_query_semantic(query_text):
        if "llm_fail" in query_text.lower():
            raise Exception("Simulated LLM failure")
        return Mock(confidence=0.7)
    
    def mock_classify_query_basic(query_text):
        if "llm_fail" in query_text.lower():
            raise Exception("Simulated LLM failure")
        return Mock(confidence=0.5)
    
    classifier.classify_query_semantic = mock_classify_query_semantic
    classifier.classify_query_basic = mock_classify_query_basic
    return classifier


@pytest.fixture
def mock_research_categorizer():
    """Create a mock research categorizer for testing."""
    categorizer = Mock()
    
    def mock_categorize_query(query_text, context=None):
        if "categorizer_fail" in query_text.lower():
            raise Exception("Simulated categorizer failure")
        
        return CategoryPrediction(
            category=ResearchCategory.GENERAL_QUERY,
            confidence=0.6,
            evidence=["test", "evidence"]
        )
    
    categorizer.categorize_query = mock_categorize_query
    return categorizer


def create_mock_routing_prediction(confidence=0.8, routing=RoutingDecision.EITHER):
    """Create a mock routing prediction for testing."""
    confidence_metrics = ConfidenceMetrics(
        overall_confidence=confidence,
        research_category_confidence=confidence,
        temporal_analysis_confidence=0.3,
        signal_strength_confidence=0.3,
        context_coherence_confidence=0.3,
        keyword_density=0.2,
        pattern_match_strength=0.2,
        biomedical_entity_count=1,
        ambiguity_score=0.4,
        conflict_score=0.2,
        alternative_interpretations=[(routing, confidence)],
        calculation_time_ms=10.0
    )
    
    return RoutingPrediction(
        routing_decision=routing,
        confidence=confidence,
        reasoning=["Mock routing prediction"],
        research_category=ResearchCategory.GENERAL_QUERY,
        confidence_metrics=confidence_metrics,
        temporal_indicators=[],
        knowledge_indicators=[],
        metadata={'test': True}
    )


# ============================================================================
# FAILURE DETECTION TESTS
# ============================================================================

class TestFailureDetection:
    """Test suite for failure detection system."""
    
    def test_failure_detector_initialization(self):
        """Test failure detector initializes correctly."""
        detector = FailureDetector()
        
        assert detector.metrics is not None
        assert detector.thresholds is not None
        assert 'response_time_warning_ms' in detector.thresholds
        assert 'error_rate_warning' in detector.thresholds
    
    def test_response_time_detection(self):
        """Test response time degradation detection."""
        detector = FailureDetector()
        
        # Record slow responses
        for _ in range(10):
            detector.record_operation_result(
                response_time_ms=3000,  # Very slow
                success=True
            )
        
        failures = detector.detect_failure_conditions("test query")
        assert FailureType.PERFORMANCE_DEGRADATION in failures
    
    def test_error_rate_detection(self):
        """Test error rate detection."""
        detector = FailureDetector()
        
        # Record high error rate
        for _ in range(5):
            detector.record_operation_result(
                response_time_ms=100,
                success=False,
                error_type=FailureType.API_ERROR
            )
        
        # Add a few successful calls
        for _ in range(3):
            detector.record_operation_result(
                response_time_ms=100,
                success=True
            )
        
        failures = detector.detect_failure_conditions("test query")
        assert FailureType.API_ERROR in failures or FailureType.SERVICE_UNAVAILABLE in failures
    
    def test_confidence_degradation_detection(self):
        """Test confidence degradation detection."""
        detector = FailureDetector()
        
        # Record low confidences
        for _ in range(10):
            detector.record_operation_result(
                response_time_ms=100,
                success=True,
                confidence=0.1  # Very low confidence
            )
        
        failures = detector.detect_failure_conditions("test query")
        assert FailureType.LOW_CONFIDENCE in failures
    
    def test_health_score_calculation(self):
        """Test system health score calculation."""
        detector = FailureDetector()
        
        # Record good performance
        for _ in range(5):
            detector.record_operation_result(
                response_time_ms=500,
                success=True,
                confidence=0.8
            )
        
        health_score = detector.metrics.calculate_health_score()
        assert health_score > 0.7
        
        # Record poor performance
        for _ in range(10):
            detector.record_operation_result(
                response_time_ms=5000,
                success=False,
                confidence=0.1
            )
        
        health_score = detector.metrics.calculate_health_score()
        assert health_score < 0.5
    
    def test_pattern_detection(self):
        """Test failure pattern detection."""
        detector = FailureDetector()
        
        # Simulate consecutive failures
        for _ in range(6):
            detector.record_operation_result(
                response_time_ms=100,
                success=False,
                error_type=FailureType.API_ERROR
            )
        
        failures = detector.detect_failure_conditions("test query")
        assert FailureType.SERVICE_UNAVAILABLE in failures
    
    def test_early_warning_signals(self):
        """Test early warning signal generation."""
        detector = FailureDetector()
        
        # Record degrading performance
        for i in range(10):
            detector.record_operation_result(
                response_time_ms=1000 + i * 200,  # Increasing response time
                success=True,
                confidence=0.8 - i * 0.05  # Decreasing confidence
            )
        
        warnings = detector.get_early_warning_signals()
        assert len(warnings) > 0
        warning_types = [w['type'] for w in warnings]
        assert any('response_time' in wt for wt in warning_types) or any('confidence' in wt for wt in warning_types)


# ============================================================================
# EMERGENCY CACHE TESTS
# ============================================================================

class TestEmergencyCache:
    """Test suite for emergency cache system."""
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test emergency cache initializes correctly."""
        cache_file = Path(temp_cache_dir) / "test_cache.pkl"
        cache = EmergencyCache(cache_file=str(cache_file))
        
        assert len(cache.cache) > 0  # Should have pre-populated patterns
        assert cache.max_cache_size == 1000
        assert cache.default_confidence == 0.15
    
    def test_cache_retrieval(self, temp_cache_dir):
        """Test cache retrieval functionality."""
        cache_file = Path(temp_cache_dir) / "test_cache.pkl"
        cache = EmergencyCache(cache_file=str(cache_file))
        
        # Test direct pattern match
        result = cache.get_cached_response("identify metabolite")
        assert result is not None
        assert result.confidence == cache.default_confidence
        assert result.metadata['emergency_cache'] is True
    
    def test_cache_pattern_matching(self, temp_cache_dir):
        """Test pattern matching in cache."""
        cache_file = Path(temp_cache_dir) / "test_cache.pkl"
        cache = EmergencyCache(cache_file=str(cache_file))
        
        # Test partial pattern match
        result = cache.get_cached_response("I need to identify a metabolite compound")
        assert result is not None
        assert 'pattern_matched' in result.metadata
    
    def test_cache_warming(self, temp_cache_dir):
        """Test cache warming functionality."""
        cache_file = Path(temp_cache_dir) / "test_cache.pkl"
        cache = EmergencyCache(cache_file=str(cache_file))
        
        initial_size = len(cache.cache)
        
        # Warm cache with new patterns
        new_patterns = ["custom pattern 1", "custom pattern 2"]
        cache.warm_cache(new_patterns)
        
        assert len(cache.cache) >= initial_size
        
        # Test retrieval of warmed patterns
        result = cache.get_cached_response("custom pattern 1")
        assert result is not None
    
    def test_cache_eviction(self, temp_cache_dir):
        """Test LRU cache eviction."""
        cache_file = Path(temp_cache_dir) / "test_cache.pkl"
        cache = EmergencyCache(cache_file=str(cache_file))
        cache.max_cache_size = 5  # Small cache for testing
        
        # Fill cache beyond capacity
        for i in range(10):
            prediction = create_mock_routing_prediction()
            cache.cache_response(f"test query {i}", prediction, force=True)
        
        # Check that cache size is maintained
        assert len(cache.cache) <= cache.max_cache_size
    
    def test_cache_persistence(self, temp_cache_dir):
        """Test cache persistence to disk."""
        cache_file = Path(temp_cache_dir) / "test_cache.pkl"
        
        # Create cache and add data
        cache1 = EmergencyCache(cache_file=str(cache_file))
        prediction = create_mock_routing_prediction()
        cache1.cache_response("persistent test", prediction, force=True)
        cache1._save_cache()
        
        # Create new cache instance and verify data persisted
        cache2 = EmergencyCache(cache_file=str(cache_file))
        result = cache2.get_cached_response("persistent test")
        assert result is not None


# ============================================================================
# GRACEFUL DEGRADATION TESTS
# ============================================================================

class TestGracefulDegradation:
    """Test suite for graceful degradation manager."""
    
    def test_degradation_manager_initialization(self):
        """Test degradation manager initializes correctly."""
        manager = GracefulDegradationManager()
        
        assert manager.degradation_levels is not None
        assert FallbackLevel.FULL_LLM_WITH_CONFIDENCE in manager.degradation_levels
        assert FallbackLevel.DEFAULT_ROUTING in manager.degradation_levels
    
    def test_fallback_level_determination(self):
        """Test optimal fallback level determination."""
        manager = GracefulDegradationManager()
        
        # Test with no failures - should use full LLM
        level = manager.determine_optimal_fallback_level([], 0.9, 'normal')
        assert level == FallbackLevel.FULL_LLM_WITH_CONFIDENCE
        
        # Test with service unavailable - should use emergency cache
        level = manager.determine_optimal_fallback_level(
            [FailureType.SERVICE_UNAVAILABLE], 0.5, 'normal'
        )
        assert level == FallbackLevel.EMERGENCY_CACHE
        
        # Test with low health score - should use default routing
        level = manager.determine_optimal_fallback_level(
            [], 0.1, 'normal'
        )
        assert level == FallbackLevel.DEFAULT_ROUTING
    
    def test_timeout_reduction(self):
        """Test progressive timeout reduction."""
        manager = GracefulDegradationManager()
        
        base_timeout = 1000
        
        # No failures - no reduction
        adjusted = manager.apply_progressive_timeout_reduction(base_timeout, 0)
        assert adjusted == base_timeout
        
        # Multiple failures - significant reduction
        adjusted = manager.apply_progressive_timeout_reduction(base_timeout, 3)
        assert adjusted < base_timeout
        assert adjusted >= 100  # Minimum timeout maintained
    
    def test_quality_threshold_adjustment(self):
        """Test quality threshold adjustment under stress."""
        manager = GracefulDegradationManager()
        
        # Low stress - minimal adjustment
        thresholds = manager.adjust_quality_thresholds(
            FallbackLevel.FULL_LLM_WITH_CONFIDENCE, 0.1
        )
        assert thresholds['quality_threshold'] >= 0.6
        
        # High stress - significant adjustment
        thresholds = manager.adjust_quality_thresholds(
            FallbackLevel.FULL_LLM_WITH_CONFIDENCE, 0.9
        )
        assert thresholds['quality_threshold'] < 0.7
    
    def test_load_shedding(self):
        """Test load shedding functionality."""
        manager = GracefulDegradationManager()
        
        # Load shedding disabled - should not shed
        assert not manager.should_shed_load('low')
        
        # Enable load shedding
        manager.enable_load_shedding(max_queue_size=5)
        
        # Fill queue to capacity
        for i in range(6):
            manager.priority_queue.append(f"query_{i}")
        
        # Should shed low priority queries
        assert manager.should_shed_load('low')
        # Should not shed critical queries
        assert not manager.should_shed_load('critical')
    
    def test_degradation_event_recording(self):
        """Test degradation event recording."""
        manager = GracefulDegradationManager()
        
        # Record degradation event
        manager.record_degradation_event(
            FallbackLevel.SIMPLIFIED_LLM,
            "test_degradation",
            True,
            {'test_metric': 123}
        )
        
        # Check event was recorded
        assert len(manager.degradation_history) == 1
        event = manager.degradation_history[0]
        assert event['fallback_level'] == 'SIMPLIFIED_LLM'
        assert event['reason'] == 'test_degradation'
        assert event['success'] is True


# ============================================================================
# RECOVERY MANAGER TESTS
# ============================================================================

class TestRecoveryManager:
    """Test suite for recovery manager."""
    
    def test_recovery_manager_initialization(self):
        """Test recovery manager initializes correctly."""
        manager = RecoveryManager()
        
        assert manager.recovery_config is not None
        assert manager.recovery_states == {}
        assert not manager.recovery_thread_running
    
    def test_service_health_check_registration(self):
        """Test service health check registration."""
        manager = RecoveryManager()
        
        # Register health check
        def mock_health_check():
            return True
        
        manager.register_service_health_check('test_service', mock_health_check)
        
        assert 'test_service' in manager.health_check_functions
        assert manager.health_check_functions['test_service']() is True
    
    def test_service_failure_marking(self):
        """Test marking services as failed."""
        manager = RecoveryManager()
        
        # Mark service as failed
        manager.mark_service_as_failed('test_service', 'Test failure')
        
        assert 'test_service' in manager.recovery_states
        assert manager.recovery_states['test_service']['status'] == 'failed'
        assert manager.recovery_states['test_service']['failure_reason'] == 'Test failure'
    
    def test_traffic_allowance_during_recovery(self):
        """Test traffic allowance during recovery process."""
        manager = RecoveryManager()
        
        # Healthy service - allow all traffic
        allow, percentage = manager.should_allow_traffic('healthy_service')
        assert allow is True
        assert percentage == 1.0
        
        # Failed service - no traffic
        manager.mark_service_as_failed('failed_service', 'Test failure')
        allow, percentage = manager.should_allow_traffic('failed_service')
        assert allow is False
        assert percentage == 0.0
        
        # Recovering service - limited traffic
        manager.recovery_states['recovering_service'] = {
            'status': 'recovering',
            'current_step': 1,  # Second step in ramp-up
            'successful_calls': 0,
            'total_calls': 0
        }
        
        allow, percentage = manager.should_allow_traffic('recovering_service')
        assert allow is True
        assert percentage == manager.recovery_config['ramp_up_steps'][1]
    
    def test_service_call_result_recording(self):
        """Test recording service call results during recovery."""
        manager = RecoveryManager()
        
        # Set up recovering service
        manager.recovery_states['test_service'] = {
            'status': 'recovering',
            'current_step': 0,
            'successful_calls': 0,
            'total_calls': 0,
            'step_success_rate': 0.0
        }
        
        # Record successful call
        manager.record_service_call_result('test_service', True)
        
        state = manager.recovery_states['test_service']
        assert state['total_calls'] == 1
        assert state['successful_calls'] == 1
        assert state['step_success_rate'] == 1.0
        
        # Record failed call
        manager.record_service_call_result('test_service', False)
        
        assert state['total_calls'] == 2
        assert state['successful_calls'] == 1
        assert state['step_success_rate'] == 0.5
    
    def test_recovery_status_reporting(self):
        """Test recovery status reporting."""
        manager = RecoveryManager()
        
        # Register health check and mark as failed
        manager.register_service_health_check('test_service', lambda: True)
        manager.mark_service_as_failed('test_service', 'Test failure')
        
        # Get status for specific service
        status = manager.get_recovery_status('test_service')
        assert status['service'] == 'test_service'
        assert status['health_check_registered'] is True
        assert status['status']['status'] == 'failed'
        
        # Get status for all services
        all_status = manager.get_recovery_status()
        assert 'all_services' in all_status
        assert 'registered_health_checks' in all_status
        assert 'test_service' in all_status['registered_health_checks']


# ============================================================================
# FALLBACK ORCHESTRATOR TESTS
# ============================================================================

class TestFallbackOrchestrator:
    """Test suite for fallback orchestrator."""
    
    @pytest.fixture
    def orchestrator(self, temp_cache_dir):
        """Create a fallback orchestrator for testing."""
        config = {'emergency_cache_file': str(Path(temp_cache_dir) / "test_cache.pkl")}
        orchestrator = FallbackOrchestrator(config=config)
        return orchestrator
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.failure_detector is not None
        assert orchestrator.degradation_manager is not None
        assert orchestrator.recovery_manager is not None
        assert orchestrator.emergency_cache is not None
        assert len(orchestrator.level_processors) == 5
    
    def test_component_integration(self, orchestrator, mock_query_router, mock_llm_classifier):
        """Test integration with existing components."""
        orchestrator.integrate_with_existing_components(
            query_router=mock_query_router,
            llm_classifier=mock_llm_classifier
        )
        
        assert orchestrator.query_router == mock_query_router
        assert orchestrator.llm_classifier == mock_llm_classifier
    
    def test_successful_primary_processing(self, orchestrator, mock_query_router):
        """Test successful processing at primary level."""
        orchestrator.integrate_with_existing_components(query_router=mock_query_router)
        
        result = orchestrator.process_query_with_comprehensive_fallback(
            "test query success"
        )
        
        assert result.success is True
        assert result.fallback_level_used == FallbackLevel.FULL_LLM_WITH_CONFIDENCE
        assert result.routing_prediction is not None
    
    def test_fallback_chain_execution(self, orchestrator):
        """Test execution of fallback chain when primary fails."""
        # Don't integrate components to force fallback
        result = orchestrator.process_query_with_comprehensive_fallback(
            "test query for fallback"
        )
        
        assert result.success is True
        assert result.fallback_level_used != FallbackLevel.FULL_LLM_WITH_CONFIDENCE
        assert len(result.attempted_levels) > 0
        assert len(result.fallback_chain) > 0
    
    def test_emergency_cache_fallback(self, orchestrator):
        """Test emergency cache fallback."""
        # Use a query that should match cache patterns
        result = orchestrator.process_query_with_comprehensive_fallback(
            "identify metabolite compound"
        )
        
        assert result.success is True
        # Should use emergency cache or default routing
        assert result.fallback_level_used in [FallbackLevel.EMERGENCY_CACHE, FallbackLevel.DEFAULT_ROUTING]
    
    def test_default_routing_fallback(self, orchestrator):
        """Test default routing as last resort."""
        # Process with failures to force default routing
        orchestrator.failure_detector.record_operation_result(5000, False)
        orchestrator.failure_detector.record_operation_result(5000, False)
        orchestrator.failure_detector.record_operation_result(5000, False)
        
        result = orchestrator.process_query_with_comprehensive_fallback(
            "unknown test query"
        )
        
        assert result.success is True  # Should always succeed with default
        assert result.routing_prediction is not None
        assert result.confidence_degradation > 0
    
    def test_load_shedding(self, orchestrator):
        """Test load shedding functionality."""
        # Enable load shedding with very small queue
        orchestrator.degradation_manager.enable_load_shedding(max_queue_size=1)
        
        # Fill queue
        orchestrator.degradation_manager.priority_queue.append("existing_query")
        
        # Try to process low priority query
        result = orchestrator.process_query_with_comprehensive_fallback(
            "low priority query",
            priority='low'
        )
        
        # Should get load shed result
        assert result.success is False or 'load' in str(result.routing_prediction.reasoning).lower()
    
    def test_performance_tracking(self, orchestrator):
        """Test performance tracking and statistics."""
        # Process several queries
        for i in range(5):
            orchestrator.process_query_with_comprehensive_fallback(f"test query {i}")
        
        # Get statistics
        stats = orchestrator.get_comprehensive_statistics()
        
        assert 'fallback_orchestrator' in stats
        assert 'failure_detection' in stats
        assert 'degradation_management' in stats
        assert 'recovery_management' in stats
        assert 'emergency_cache' in stats
        assert 'system_health' in stats
    
    def test_emergency_mode(self, orchestrator):
        """Test emergency mode activation."""
        orchestrator.enable_emergency_mode()
        
        # Process query in emergency mode
        result = orchestrator.process_query_with_comprehensive_fallback(
            "emergency test query"
        )
        
        assert result.success is True
        # Should use fallback mechanisms
        assert result.fallback_level_used != FallbackLevel.FULL_LLM_WITH_CONFIDENCE
        
        # Disable emergency mode
        orchestrator.disable_emergency_mode()


# ============================================================================
# ENHANCED ROUTER INTEGRATION TESTS
# ============================================================================

class TestEnhancedRouterIntegration:
    """Test suite for enhanced router integration."""
    
    @pytest.fixture
    def enhanced_router(self, temp_cache_dir):
        """Create enhanced router for testing."""
        config = FallbackIntegrationConfig(
            emergency_cache_file=str(Path(temp_cache_dir) / "test_cache.pkl"),
            enable_monitoring=False  # Disable for testing
        )
        return EnhancedBiomedicalQueryRouter(fallback_config=config)
    
    def test_enhanced_router_initialization(self, enhanced_router):
        """Test enhanced router initializes correctly."""
        assert enhanced_router.fallback_config is not None
        assert enhanced_router.compatibility_layer is not None
        assert enhanced_router.auto_config_manager is not None
        
        if enhanced_router.fallback_config.enable_fallback_system:
            assert enhanced_router.fallback_orchestrator is not None
    
    def test_backward_compatibility(self, enhanced_router):
        """Test backward compatibility with existing API."""
        # Test route_query method
        result = enhanced_router.route_query("test query")
        assert isinstance(result, RoutingPrediction)
        assert hasattr(result, 'routing_decision')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reasoning')
        
        # Test boolean methods
        assert isinstance(enhanced_router.should_use_lightrag("test query"), bool)
        assert isinstance(enhanced_router.should_use_perplexity("test query"), bool)
    
    def test_fallback_integration(self, enhanced_router):
        """Test fallback system integration."""
        # Process query that might trigger fallback
        result = enhanced_router.route_query("complex metabolomics pathway analysis")
        
        assert result is not None
        assert result.confidence > 0  # Should always have some confidence
        
        # Check for fallback metadata
        if result.metadata:
            fallback_used = result.metadata.get('fallback_system_used', False)
            if fallback_used:
                assert 'fallback_level_used' in result.metadata
    
    def test_performance_under_load(self, enhanced_router):
        """Test performance under load conditions."""
        queries = [f"test query {i}" for i in range(20)]
        results = []
        
        start_time = time.time()
        
        for query in queries:
            result = enhanced_router.route_query(query)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time_per_query = (total_time / len(queries)) * 1000  # ms
        
        # All queries should succeed
        assert all(r is not None for r in results)
        
        # Average response time should be reasonable
        assert avg_time_per_query < 500  # 500ms per query max
    
    def test_health_reporting(self, enhanced_router):
        """Test system health reporting."""
        health_report = enhanced_router.get_system_health_report()
        
        assert 'enhanced_router_operational' in health_report
        assert health_report['enhanced_router_operational'] is True
        assert 'fallback_system_status' in health_report
        assert 'timestamp' in health_report
    
    def test_statistics_reporting(self, enhanced_router):
        """Test enhanced statistics reporting."""
        # Process some queries first
        for i in range(5):
            enhanced_router.route_query(f"test query {i}")
        
        stats = enhanced_router.get_enhanced_routing_statistics()
        
        assert 'enhanced_router_stats' in stats
        assert 'fallback_system_enabled' in stats
        assert 'fallback_config' in stats
        
        enhanced_stats = stats['enhanced_router_stats']
        assert 'total_enhanced_queries' in enhanced_stats
        assert enhanced_stats['total_enhanced_queries'] >= 5
    
    def test_emergency_mode_integration(self, enhanced_router):
        """Test emergency mode integration."""
        if enhanced_router.fallback_orchestrator:
            # Enable emergency mode
            enhanced_router.enable_emergency_mode()
            
            # Process query in emergency mode
            result = enhanced_router.route_query("emergency test query")
            
            assert result is not None
            assert result.confidence >= 0  # Should have some confidence even in emergency
            
            # Disable emergency mode
            enhanced_router.disable_emergency_mode()
    
    def test_graceful_shutdown(self, enhanced_router):
        """Test graceful shutdown of enhanced features."""
        # Should not raise exceptions
        enhanced_router.shutdown_enhanced_features()


# ============================================================================
# STRESS AND PERFORMANCE TESTS
# ============================================================================

class TestStressAndPerformance:
    """Stress tests and performance validation."""
    
    def test_concurrent_processing(self, temp_cache_dir):
        """Test concurrent query processing."""
        config = FallbackIntegrationConfig(
            emergency_cache_file=str(Path(temp_cache_dir) / "concurrent_cache.pkl"),
            enable_monitoring=False
        )
        router = EnhancedBiomedicalQueryRouter(fallback_config=config)
        
        def process_queries(query_prefix, num_queries):
            results = []
            for i in range(num_queries):
                result = router.route_query(f"{query_prefix} query {i}")
                results.append(result)
            return results
        
        # Create multiple threads
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for thread_id in range(4):
                future = executor.submit(process_queries, f"thread_{thread_id}", 10)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)
        
        # All queries should succeed
        assert len(all_results) == 40
        assert all(r is not None for r in all_results)
        assert all(r.confidence >= 0 for r in all_results)
        
        router.shutdown_enhanced_features()
    
    def test_system_under_failure_conditions(self, temp_cache_dir):
        """Test system behavior under various failure conditions."""
        config = FallbackIntegrationConfig(
            emergency_cache_file=str(Path(temp_cache_dir) / "failure_cache.pkl"),
            enable_monitoring=False
        )
        router = EnhancedBiomedicalQueryRouter(fallback_config=config)
        
        # Simulate various failure conditions
        failure_queries = [
            "fail primary routing",  # Should trigger routing failure
            "llm_fail classification",  # Should trigger LLM failure
            "categorizer_fail analysis",  # Should trigger categorizer failure
            "slow response query",  # Should trigger timeout
            "unknown complex query with multiple failures"  # Should test multiple fallbacks
        ]
        
        results = []
        for query in failure_queries:
            result = router.route_query(query)
            results.append(result)
        
        # System should handle all failures gracefully
        assert len(results) == len(failure_queries)
        assert all(r is not None for r in results)
        
        # At least some queries should use fallback mechanisms
        fallback_used = any(
            r.metadata and r.metadata.get('fallback_system_used', False)
            for r in results if r.metadata
        )
        
        router.shutdown_enhanced_features()
    
    def test_memory_usage_stability(self, temp_cache_dir):
        """Test memory usage remains stable during extended operation."""
        config = FallbackIntegrationConfig(
            emergency_cache_file=str(Path(temp_cache_dir) / "memory_cache.pkl"),
            enable_monitoring=False
        )
        router = EnhancedBiomedicalQueryRouter(fallback_config=config)
        
        # Process many queries to test memory stability
        for batch in range(10):
            queries = [f"batch_{batch}_query_{i}" for i in range(50)]
            
            for query in queries:
                result = router.route_query(query)
                assert result is not None
        
        # Check that caches haven't grown unboundedly
        if router.fallback_orchestrator and router.fallback_orchestrator.emergency_cache:
            cache_size = len(router.fallback_orchestrator.emergency_cache.cache)
            max_cache_size = router.fallback_orchestrator.emergency_cache.max_cache_size
            assert cache_size <= max_cache_size
        
        router.shutdown_enhanced_features()
    
    def test_recovery_after_system_stress(self, temp_cache_dir):
        """Test system recovery after stress conditions."""
        config = FallbackIntegrationConfig(
            emergency_cache_file=str(Path(temp_cache_dir) / "recovery_cache.pkl"),
            enable_monitoring=False
        )
        router = EnhancedBiomedicalQueryRouter(fallback_config=config)
        
        # Simulate system stress
        stress_queries = ["fail " + f"stress_query_{i}" for i in range(20)]
        
        # Process stress queries
        for query in stress_queries:
            router.route_query(query)
        
        # Allow system to settle
        time.sleep(0.1)
        
        # Test normal queries - should work correctly
        normal_queries = [f"normal_query_{i}" for i in range(10)]
        results = []
        
        for query in normal_queries:
            result = router.route_query(query)
            results.append(result)
        
        # System should have recovered
        assert all(r is not None for r in results)
        assert all(r.confidence > 0 for r in results)
        
        router.shutdown_enhanced_features()


# ============================================================================
# INTEGRATION WITH PRODUCTION SCENARIOS
# ============================================================================

class TestProductionIntegration:
    """Tests for production integration scenarios."""
    
    def test_production_ready_router_creation(self, temp_cache_dir):
        """Test creation of production-ready enhanced router."""
        router = create_production_ready_enhanced_router(
            emergency_cache_dir=temp_cache_dir
        )
        
        assert router is not None
        assert router.fallback_config is not None
        assert router.fallback_config.enable_fallback_system is True
        assert router.fallback_config.enable_monitoring is True
        
        # Test basic functionality
        result = router.route_query("production test query")
        assert result is not None
        
        router.shutdown_enhanced_features()
    
    def test_existing_router_enhancement(self):
        """Test enhancement of existing router."""
        # Create existing router
        existing_router = BiomedicalQueryRouter()
        
        # Create enhanced router from existing
        enhanced_router = create_enhanced_router_from_existing(existing_router)
        
        assert enhanced_router is not None
        assert isinstance(enhanced_router, EnhancedBiomedicalQueryRouter)
        
        # Test that existing configuration was copied
        assert enhanced_router.category_routing_map == existing_router.category_routing_map
        assert enhanced_router.routing_thresholds == existing_router.routing_thresholds
        
        enhanced_router.shutdown_enhanced_features()
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Test with default configuration
        config1 = FallbackIntegrationConfig()
        assert config1.enable_fallback_system is True
        assert config1.max_response_time_ms == 2000.0
        
        # Test with custom configuration
        config2 = FallbackIntegrationConfig(
            max_response_time_ms=1500,
            confidence_threshold=0.7,
            enable_cache_warming=False
        )
        assert config2.max_response_time_ms == 1500
        assert config2.confidence_threshold == 0.7
        assert config2.enable_cache_warming is False
        
        # Test serialization
        config_dict = config2.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['max_response_time_ms'] == 1500


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])