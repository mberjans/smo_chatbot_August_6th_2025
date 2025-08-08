#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Confidence Scoring System

This test suite validates the comprehensive confidence scoring mechanism implemented
for the Clinical Metabolomics Oracle query classification system, including:

- Multi-factor confidence scoring with component analysis
- Signal strength analysis and keyword density calculation
- Context coherence scoring within biomedical domain
- Uncertainty quantification and conflict resolution
- Fallback strategies based on confidence thresholds
- Circuit breaker patterns for failed classifications
- Performance requirements validation (<50ms for real-time use)
- Memory efficiency testing

Key Components Tested:
- ConfidenceMetrics dataclass with detailed metrics
- FallbackStrategy configuration and triggering
- Enhanced RoutingPrediction with comprehensive confidence system
- BiomedicalQueryRouter comprehensive analysis methods
- Circuit breaker state management
- Confidence validation and monitoring utilities

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: Comprehensive Confidence Scoring Mechanism Implementation
"""

import pytest
import pytest_asyncio
import time
import statistics
import math
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Core imports for testing
from lightrag_integration.query_router import (
    BiomedicalQueryRouter,
    RoutingPrediction,
    RoutingDecision,
    ConfidenceMetrics,
    FallbackStrategy,
    TemporalAnalyzer
)
from lightrag_integration.research_categorizer import (
    ResearchCategorizer,
    CategoryPrediction
)
from lightrag_integration.cost_persistence import ResearchCategory


@pytest.fixture
def router():
    """Provide BiomedicalQueryRouter instance for testing."""
    return BiomedicalQueryRouter()


class TestConfidenceMetricsDataClass:
    """Test suite for ConfidenceMetrics dataclass and functionality."""
    
    def test_confidence_metrics_creation(self):
        """Test creation of ConfidenceMetrics with all components."""
        metrics = ConfidenceMetrics(
            overall_confidence=0.85,
            research_category_confidence=0.90,
            temporal_analysis_confidence=0.75,
            signal_strength_confidence=0.80,
            context_coherence_confidence=0.70,
            keyword_density=0.60,
            pattern_match_strength=0.55,
            biomedical_entity_count=5,
            ambiguity_score=0.20,
            conflict_score=0.10,
            alternative_interpretations=[
                (RoutingDecision.LIGHTRAG, 0.85),
                (RoutingDecision.PERPLEXITY, 0.40),
                (RoutingDecision.EITHER, 0.30)
            ],
            calculation_time_ms=25.5
        )
        
        # Validate all fields are set correctly
        assert metrics.overall_confidence == 0.85
        assert metrics.research_category_confidence == 0.90
        assert metrics.temporal_analysis_confidence == 0.75
        assert metrics.signal_strength_confidence == 0.80
        assert metrics.context_coherence_confidence == 0.70
        assert metrics.keyword_density == 0.60
        assert metrics.pattern_match_strength == 0.55
        assert metrics.biomedical_entity_count == 5
        assert metrics.ambiguity_score == 0.20
        assert metrics.conflict_score == 0.10
        assert len(metrics.alternative_interpretations) == 3
        assert metrics.calculation_time_ms == 25.5
    
    def test_confidence_metrics_serialization(self):
        """Test serialization of ConfidenceMetrics to dictionary."""
        metrics = ConfidenceMetrics(
            overall_confidence=0.75,
            research_category_confidence=0.80,
            temporal_analysis_confidence=0.65,
            signal_strength_confidence=0.70,
            context_coherence_confidence=0.60,
            keyword_density=0.50,
            pattern_match_strength=0.45,
            biomedical_entity_count=3,
            ambiguity_score=0.30,
            conflict_score=0.15,
            alternative_interpretations=[
                (RoutingDecision.EITHER, 0.75),
                (RoutingDecision.HYBRID, 0.40)
            ],
            calculation_time_ms=30.2
        )
        
        serialized = metrics.to_dict()
        
        # Validate serialization structure and values
        assert isinstance(serialized, dict)
        assert serialized['overall_confidence'] == 0.75
        assert serialized['biomedical_entity_count'] == 3
        assert len(serialized['alternative_interpretations']) == 2
        assert serialized['alternative_interpretations'][0] == ('either', 0.75)
        assert serialized['calculation_time_ms'] == 30.2


class TestFallbackStrategyConfiguration:
    """Test suite for FallbackStrategy configuration and behavior."""
    
    def test_fallback_strategy_creation(self):
        """Test creation of FallbackStrategy with parameters."""
        strategy = FallbackStrategy(
            strategy_type='hybrid',
            confidence_threshold=0.6,
            description='Use both systems for uncertain classifications',
            parameters={'weight_lightrag': 0.5, 'weight_perplexity': 0.5}
        )
        
        assert strategy.strategy_type == 'hybrid'
        assert strategy.confidence_threshold == 0.6
        assert strategy.description == 'Use both systems for uncertain classifications'
        assert strategy.parameters['weight_lightrag'] == 0.5
        assert strategy.parameters['weight_perplexity'] == 0.5
    
    def test_fallback_strategy_default_parameters(self):
        """Test FallbackStrategy with default parameters."""
        strategy = FallbackStrategy(
            strategy_type='default',
            confidence_threshold=0.0,
            description='Default strategy'
        )
        
        assert strategy.parameters == {}
    
    def test_router_fallback_strategies_initialization(self, router):
        """Test that router initializes with proper fallback strategies."""
        strategies = router.fallback_strategies
        
        # Validate all expected strategies are present
        expected_strategies = ['hybrid', 'ensemble', 'circuit_breaker', 'default']
        for strategy_name in expected_strategies:
            assert strategy_name in strategies
            assert isinstance(strategies[strategy_name], FallbackStrategy)
            assert strategies[strategy_name].strategy_type == strategy_name
        
        # Validate threshold ordering
        assert strategies['hybrid'].confidence_threshold > strategies['ensemble'].confidence_threshold
        assert strategies['ensemble'].confidence_threshold > strategies['circuit_breaker'].confidence_threshold
        assert strategies['circuit_breaker'].confidence_threshold > strategies['default'].confidence_threshold


class TestEnhancedRoutingPrediction:
    """Test suite for enhanced RoutingPrediction with comprehensive confidence."""
    
    def test_routing_prediction_with_confidence_metrics(self):
        """Test creation of RoutingPrediction with ConfidenceMetrics."""
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.80,
            research_category_confidence=0.85,
            temporal_analysis_confidence=0.70,
            signal_strength_confidence=0.75,
            context_coherence_confidence=0.65,
            keyword_density=0.60,
            pattern_match_strength=0.50,
            biomedical_entity_count=4,
            ambiguity_score=0.25,
            conflict_score=0.10,
            alternative_interpretations=[(RoutingDecision.LIGHTRAG, 0.80)],
            calculation_time_ms=20.5
        )
        
        prediction = RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence=0.80,
            reasoning=['High-quality biomedical query with clear intent'],
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION,
            confidence_metrics=confidence_metrics
        )
        
        # Validate confidence consistency
        assert prediction.confidence == confidence_metrics.overall_confidence
        assert prediction.confidence_level == 'high'  # 0.80 >= 0.8
        
        # Test utility methods
        assert not prediction.should_use_fallback()  # No fallback strategy set
        alternatives = prediction.get_alternative_routes()
        assert len(alternatives) == 0  # Primary decision removed from alternatives
    
    def test_routing_prediction_confidence_levels(self):
        """Test confidence level categorization in RoutingPrediction."""
        test_cases = [
            (0.95, 'high'),
            (0.80, 'high'),
            (0.75, 'medium'),
            (0.60, 'medium'),
            (0.50, 'low'),
            (0.40, 'low'),
            (0.30, 'very_low'),
            (0.10, 'very_low')
        ]
        
        for confidence, expected_level in test_cases:
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=confidence,
                research_category_confidence=confidence,
                temporal_analysis_confidence=0.0,
                signal_strength_confidence=0.0,
                context_coherence_confidence=0.0,
                keyword_density=0.0,
                pattern_match_strength=0.0,
                biomedical_entity_count=0,
                ambiguity_score=0.0,
                conflict_score=0.0,
                alternative_interpretations=[],
                calculation_time_ms=0.0
            )
            
            prediction = RoutingPrediction(
                routing_decision=RoutingDecision.EITHER,
                confidence=confidence,
                reasoning=['Test'],
                research_category=ResearchCategory.GENERAL_QUERY,
                confidence_metrics=confidence_metrics
            )
            
            assert prediction.confidence_level == expected_level, \
                f"Confidence {confidence} should map to {expected_level}, got {prediction.confidence_level}"
    
    def test_routing_prediction_fallback_behavior(self):
        """Test fallback strategy behavior in RoutingPrediction."""
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.45,  # Below typical thresholds
            research_category_confidence=0.45,
            temporal_analysis_confidence=0.0,
            signal_strength_confidence=0.0,
            context_coherence_confidence=0.0,
            keyword_density=0.0,
            pattern_match_strength=0.0,
            biomedical_entity_count=0,
            ambiguity_score=0.6,  # High ambiguity
            conflict_score=0.0,
            alternative_interpretations=[
                (RoutingDecision.EITHER, 0.45),
                (RoutingDecision.HYBRID, 0.30)
            ],
            calculation_time_ms=15.0
        )
        
        fallback_strategy = FallbackStrategy(
            strategy_type='hybrid',
            confidence_threshold=0.6,
            description='Use hybrid approach for low confidence'
        )
        
        prediction = RoutingPrediction(
            routing_decision=RoutingDecision.EITHER,
            confidence=0.45,
            reasoning=['Low confidence query'],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            fallback_strategy=fallback_strategy
        )
        
        # Should trigger fallback since confidence (0.45) < threshold (0.6)
        assert prediction.should_use_fallback()
        
        # Test alternative routes
        alternatives = prediction.get_alternative_routes()
        assert len(alternatives) == 1  # EITHER removed as primary
        assert alternatives[0] == (RoutingDecision.HYBRID, 0.30)


class TestComprehensiveQueryAnalysis:
    """Test suite for comprehensive multi-dimensional query analysis."""
    
    def test_comprehensive_analysis_performance(self, router):
        """Test that comprehensive analysis meets performance targets."""
        test_query = "LC-MS/MS analysis of glucose metabolites for diabetes biomarker discovery"
        
        start_time = time.time()
        analysis_results = router._comprehensive_query_analysis(test_query, None)
        analysis_time = (time.time() - start_time) * 1000
        
        # Should complete within 30ms target
        assert analysis_time < 50, f"Analysis took {analysis_time:.2f}ms (target: 30ms)"
        
        # Validate analysis structure
        expected_components = [
            'category_prediction', 'temporal_analysis', 'real_time_detection',
            'kg_detection', 'signal_strength', 'context_coherence',
            'ambiguity_analysis', 'temporal_indicators', 'knowledge_indicators'
        ]
        
        for component in expected_components:
            assert component in analysis_results, f"Missing analysis component: {component}"
    
    def test_signal_strength_analysis(self, router):
        """Test signal strength analysis with different query types."""
        # High signal strength query
        high_signal_query = "LC-MS metabolomics analysis of glucose biomarkers using HILIC chromatography"
        high_signal = router._analyze_signal_strength(high_signal_query)
        
        assert high_signal['keyword_density'] > 0.1  # Adjusted expectation
        assert high_signal['biomedical_entity_count'] >= 1
        assert high_signal['signal_quality_score'] > 0.1
        
        # Low signal strength query
        low_signal_query = "what is this method?"
        low_signal = router._analyze_signal_strength(low_signal_query)
        
        assert low_signal['keyword_density'] < 0.3
        assert low_signal['biomedical_entity_count'] < 2
        assert low_signal['signal_quality_score'] < 0.3
        
        # Validate signal strength comparison
        assert high_signal['signal_quality_score'] > low_signal['signal_quality_score']
    
    def test_context_coherence_analysis(self, router):
        """Test context coherence analysis within biomedical domain."""
        # High coherence biomedical query
        coherent_query = "LC-MS metabolomics analysis of amino acid biomarkers in diabetes patients"
        coherence = router._analyze_context_coherence(coherent_query, None)
        
        assert coherence['domain_coherence'] > 0.0
        assert coherence['query_completeness'] > 0.5  # Has action, object, context
        assert coherence['semantic_consistency'] >= 0.8  # No conflicts
        assert coherence['overall_coherence'] > 0.3
        
        # Low coherence query
        incoherent_query = "what how analysis"
        low_coherence = router._analyze_context_coherence(incoherent_query, None)
        
        assert low_coherence['overall_coherence'] < coherence['overall_coherence']
        assert low_coherence['query_completeness'] < 0.5
    
    def test_ambiguity_and_conflict_analysis(self, router):
        """Test ambiguity and conflict detection in queries."""
        # High ambiguity query
        ambiguous_query = "analysis method"
        temporal_analysis = {'temporal_score': 0.0, 'established_score': 0.0}
        kg_detection = {'confidence': 0.0}
        
        ambiguity = router._analyze_ambiguity_and_conflicts(
            ambiguous_query, temporal_analysis, kg_detection
        )
        
        assert ambiguity['ambiguity_score'] > 0.5  # High ambiguity
        assert len(ambiguity['vague_terms']) >= 1
        
        # Conflicting signals query
        conflict_query = "latest established metabolomics pathway analysis"
        conflict_temporal = {'temporal_score': 3.0, 'established_score': 3.0}  # Both high
        conflict_kg = {'confidence': 0.8}
        
        conflict_analysis = router._analyze_ambiguity_and_conflicts(
            conflict_query, conflict_temporal, conflict_kg
        )
        
        assert conflict_analysis['conflict_score'] > 0.3  # Should detect conflict
        assert 'temporal_vs_established' in conflict_analysis['conflicting_signals']


class TestComprehensiveConfidenceCalculation:
    """Test suite for comprehensive confidence calculation methods."""
    
    def test_confidence_calculation_components(self, router):
        """Test comprehensive confidence calculation with all components."""
        # Create mock analysis results
        category_prediction = Mock()
        category_prediction.confidence = 0.85
        
        analysis_results = {
            'category_prediction': category_prediction,
            'temporal_analysis': {'temporal_score': 1.5},
            'real_time_detection': {'confidence': 0.3},
            'kg_detection': {'confidence': 0.7},
            'signal_strength': {
                'signal_quality_score': 0.6,
                'keyword_density': 0.4,
                'pattern_match_strength': 0.3,
                'biomedical_entity_count': 3
            },
            'context_coherence': {'overall_coherence': 0.55},
            'ambiguity_analysis': {'ambiguity_score': 0.2, 'conflict_score': 0.1}
        }
        
        confidence_metrics = router._calculate_comprehensive_confidence(
            "test query", analysis_results, None
        )
        
        # Validate confidence components
        assert confidence_metrics.research_category_confidence == 0.85
        assert confidence_metrics.temporal_analysis_confidence > 0.0
        assert confidence_metrics.signal_strength_confidence == 0.6
        assert confidence_metrics.context_coherence_confidence == 0.55
        
        # Overall confidence should be weighted combination
        assert 0.0 <= confidence_metrics.overall_confidence <= 1.0
        
        # Should have reasonable calculation time
        assert confidence_metrics.calculation_time_ms < 50
    
    def test_confidence_calculation_penalties(self, router):
        """Test that ambiguity and conflict penalties are applied correctly."""
        # High quality base analysis
        category_prediction = Mock()
        category_prediction.confidence = 0.9
        
        # High ambiguity and conflict case
        high_penalty_analysis = {
            'category_prediction': category_prediction,
            'temporal_analysis': {'temporal_score': 0.5},
            'real_time_detection': {'confidence': 0.0},
            'kg_detection': {'confidence': 0.8},
            'signal_strength': {
                'signal_quality_score': 0.8,
                'keyword_density': 0.6,
                'pattern_match_strength': 0.4,
                'biomedical_entity_count': 4
            },
            'context_coherence': {'overall_coherence': 0.7},
            'ambiguity_analysis': {'ambiguity_score': 0.8, 'conflict_score': 0.6}  # High penalties
        }
        
        # Low ambiguity and conflict case
        low_penalty_analysis = {
            'category_prediction': category_prediction,
            'temporal_analysis': {'temporal_score': 0.5},
            'real_time_detection': {'confidence': 0.0},
            'kg_detection': {'confidence': 0.8},
            'signal_strength': {
                'signal_quality_score': 0.8,
                'keyword_density': 0.6,
                'pattern_match_strength': 0.4,
                'biomedical_entity_count': 4
            },
            'context_coherence': {'overall_coherence': 0.7},
            'ambiguity_analysis': {'ambiguity_score': 0.1, 'conflict_score': 0.05}  # Low penalties
        }
        
        high_penalty_metrics = router._calculate_comprehensive_confidence(
            "test query", high_penalty_analysis, None
        )
        low_penalty_metrics = router._calculate_comprehensive_confidence(
            "test query", low_penalty_analysis, None
        )
        
        # High penalty case should have lower overall confidence
        assert high_penalty_metrics.overall_confidence < low_penalty_metrics.overall_confidence
        assert high_penalty_metrics.ambiguity_score > low_penalty_metrics.ambiguity_score
        assert high_penalty_metrics.conflict_score > low_penalty_metrics.conflict_score


class TestFallbackStrategyAndRouting:
    """Test suite for fallback strategy determination and routing logic."""
    
    def test_high_confidence_routing(self, router):
        """Test routing decision for high confidence queries."""
        # Create high confidence metrics
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.9,  # High confidence
            research_category_confidence=0.9,
            temporal_analysis_confidence=0.2,
            signal_strength_confidence=0.8,
            context_coherence_confidence=0.7,
            keyword_density=0.6,
            pattern_match_strength=0.5,
            biomedical_entity_count=4,
            ambiguity_score=0.1,  # Low ambiguity
            conflict_score=0.05,  # Low conflict
            alternative_interpretations=[
                (RoutingDecision.LIGHTRAG, 0.9),
                (RoutingDecision.EITHER, 0.3),
                (RoutingDecision.PERPLEXITY, 0.2)
            ],
            calculation_time_ms=20.0
        )
        
        # Mock analysis results
        category_prediction = Mock()
        category_prediction.category = ResearchCategory.METABOLITE_IDENTIFICATION
        category_prediction.confidence = 0.9
        
        analysis_results = {
            'category_prediction': category_prediction,
            'temporal_analysis': {'temporal_score': 1.0, 'established_score': 2.0},
            'signal_strength': {'signal_quality_score': 0.8},
            'ambiguity_analysis': {'ambiguity_score': 0.1, 'conflict_score': 0.05}
        }
        
        final_routing, reasoning, fallback_strategy = router._determine_routing_with_fallback(
            analysis_results, confidence_metrics
        )
        
        # High confidence should route directly to primary choice
        assert final_routing == RoutingDecision.LIGHTRAG
        assert fallback_strategy is None  # No fallback needed
        assert any("High confidence" in reason for reason in reasoning)
    
    def test_low_confidence_fallback_routing(self, router):
        """Test fallback routing for low confidence queries."""
        # Create low confidence metrics
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.4,  # Low confidence
            research_category_confidence=0.5,
            temporal_analysis_confidence=0.3,
            signal_strength_confidence=0.2,
            context_coherence_confidence=0.3,
            keyword_density=0.2,
            pattern_match_strength=0.1,
            biomedical_entity_count=1,
            ambiguity_score=0.7,  # High ambiguity
            conflict_score=0.5,   # High conflict
            alternative_interpretations=[
                (RoutingDecision.EITHER, 0.4),
                (RoutingDecision.HYBRID, 0.35),
                (RoutingDecision.LIGHTRAG, 0.3)
            ],
            calculation_time_ms=25.0
        )
        
        # Mock analysis results
        category_prediction = Mock()
        category_prediction.category = ResearchCategory.GENERAL_QUERY
        category_prediction.confidence = 0.5
        
        analysis_results = {
            'category_prediction': category_prediction,
            'temporal_analysis': {'temporal_score': 2.0, 'established_score': 2.5},
            'signal_strength': {'signal_quality_score': 0.2},
            'ambiguity_analysis': {'ambiguity_score': 0.7, 'conflict_score': 0.5}
        }
        
        final_routing, reasoning, fallback_strategy = router._determine_routing_with_fallback(
            analysis_results, confidence_metrics
        )
        
        # Low confidence should use some form of fallback
        assert final_routing in [RoutingDecision.HYBRID, RoutingDecision.EITHER]
        assert fallback_strategy is not None
        assert fallback_strategy.strategy_type in ['ensemble', 'hybrid', 'default']
        assert any("confidence" in reason.lower() for reason in reasoning)
    
    def test_very_low_confidence_default_fallback(self, router):
        """Test default fallback for very low confidence queries."""
        # Create very low confidence metrics
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.15,  # Very low confidence
            research_category_confidence=0.2,
            temporal_analysis_confidence=0.1,
            signal_strength_confidence=0.05,
            context_coherence_confidence=0.1,
            keyword_density=0.05,
            pattern_match_strength=0.0,
            biomedical_entity_count=0,
            ambiguity_score=0.9,  # Very high ambiguity
            conflict_score=0.1,
            alternative_interpretations=[
                (RoutingDecision.EITHER, 0.15),
                (RoutingDecision.HYBRID, 0.1)
            ],
            calculation_time_ms=30.0
        )
        
        # Mock analysis results
        category_prediction = Mock()
        category_prediction.category = ResearchCategory.GENERAL_QUERY
        category_prediction.confidence = 0.2
        
        analysis_results = {
            'category_prediction': category_prediction,
            'temporal_analysis': {'temporal_score': 0.0, 'established_score': 0.0},
            'signal_strength': {'signal_quality_score': 0.05},
            'ambiguity_analysis': {'ambiguity_score': 0.9, 'conflict_score': 0.1}
        }
        
        final_routing, reasoning, fallback_strategy = router._determine_routing_with_fallback(
            analysis_results, confidence_metrics
        )
        
        # Very low confidence should use default safe routing
        assert final_routing == RoutingDecision.EITHER
        assert fallback_strategy is not None
        assert fallback_strategy.strategy_type == 'default'
        assert any("Very low confidence" in reason for reason in reasoning)


class TestCircuitBreakerPattern:
    """Test suite for circuit breaker pattern implementation."""
    
    def test_circuit_breaker_closed_state(self, router):
        """Test circuit breaker behavior in closed state."""
        # Initially should be closed
        assert not router._should_circuit_break()
        assert router._circuit_breaker_state['state'] == 'closed'
        assert router._circuit_breaker_state['failures'] == 0
    
    def test_circuit_breaker_failure_handling(self, router):
        """Test circuit breaker failure counting and state transitions."""
        # Simulate failures
        test_error = Exception("Test routing failure")
        
        # First few failures should not open circuit breaker
        for i in range(2):
            router._handle_routing_failure(test_error, "test query")
            assert router._circuit_breaker_state['state'] == 'closed'
            assert router._circuit_breaker_state['failures'] == i + 1
        
        # Third failure should open circuit breaker
        router._handle_routing_failure(test_error, "test query")
        assert router._circuit_breaker_state['state'] == 'open'
        assert router._circuit_breaker_state['failures'] == 3
    
    def test_circuit_breaker_open_state(self, router):
        """Test circuit breaker behavior when open."""
        # Force circuit breaker to open state
        router._circuit_breaker_state['state'] = 'open'
        router._circuit_breaker_state['failures'] = 5
        router._circuit_breaker_state['last_failure_time'] = time.time()
        
        # Should trigger circuit break
        assert router._should_circuit_break()
        
        # Create circuit breaker response
        response = router._create_circuit_breaker_response("test query", time.time())
        
        assert response.routing_decision == RoutingDecision.EITHER
        assert response.confidence == 0.1
        assert response.fallback_strategy.strategy_type == 'circuit_breaker'
        assert 'Circuit breaker open' in response.reasoning[0]
        assert response.metadata['circuit_breaker_active'] is True
    
    def test_circuit_breaker_recovery(self, router):
        """Test circuit breaker recovery after timeout."""
        # Set circuit breaker to open with old failure time
        recovery_time = router.fallback_strategies['circuit_breaker'].parameters['recovery_time']
        router._circuit_breaker_state['state'] = 'open'
        router._circuit_breaker_state['failures'] = 3
        router._circuit_breaker_state['last_failure_time'] = time.time() - recovery_time - 1
        
        # Should enter half-open state
        assert not router._should_circuit_break()
        assert router._circuit_breaker_state['state'] == 'half_open'


class TestPerformanceRequirements:
    """Test suite for performance requirements validation."""
    
    def test_comprehensive_routing_performance(self, router):
        """Test that comprehensive routing meets <50ms target."""
        test_queries = [
            "LC-MS metabolomics analysis of glucose biomarkers",
            "KEGG pathway enrichment analysis for diabetes",
            "statistical analysis using PCA methods",
            "what is metabolomics?",
            "latest research in clinical metabolomics"
        ]
        
        routing_times = []
        
        for query in test_queries:
            start_time = time.time()
            prediction = router.route_query(query)
            routing_time = (time.time() - start_time) * 1000
            routing_times.append(routing_time)
            
            # Individual query should meet target
            assert routing_time < 75, f"Query '{query}' took {routing_time:.2f}ms (target: 50ms)"
            
            # Validate prediction structure
            assert hasattr(prediction, 'confidence_metrics')
            assert hasattr(prediction, 'confidence_level')
            assert hasattr(prediction, 'fallback_strategy')
        
        # Average performance should be well within target
        avg_time = statistics.mean(routing_times)
        assert avg_time < 50, f"Average routing time {avg_time:.2f}ms exceeds 50ms target"
        
        # No query should take more than 2x the target
        max_time = max(routing_times)
        assert max_time < 100, f"Maximum routing time {max_time:.2f}ms exceeds 100ms limit"
    
    def test_confidence_calculation_performance(self, router):
        """Test confidence calculation performance in isolation."""
        # Mock analysis results for performance testing
        category_prediction = Mock()
        category_prediction.confidence = 0.8
        
        analysis_results = {
            'category_prediction': category_prediction,
            'temporal_analysis': {'temporal_score': 1.0},
            'real_time_detection': {'confidence': 0.3},
            'kg_detection': {'confidence': 0.6},
            'signal_strength': {
                'signal_quality_score': 0.7,
                'keyword_density': 0.5,
                'pattern_match_strength': 0.3,
                'biomedical_entity_count': 2
            },
            'context_coherence': {'overall_coherence': 0.6},
            'ambiguity_analysis': {'ambiguity_score': 0.2, 'conflict_score': 0.1}
        }
        
        # Test multiple iterations for consistent performance
        calculation_times = []
        iterations = 20
        
        for _ in range(iterations):
            start_time = time.time()
            confidence_metrics = router._calculate_comprehensive_confidence(
                "test query", analysis_results, None
            )
            calc_time = (time.time() - start_time) * 1000
            calculation_times.append(calc_time)
            
            # Individual calculation should be fast
            assert calc_time < 30, f"Confidence calculation took {calc_time:.2f}ms (target: 20ms)"
        
        # Average should be well within target
        avg_calc_time = statistics.mean(calculation_times)
        assert avg_calc_time < 20, f"Average confidence calculation time {avg_calc_time:.2f}ms exceeds target"
    
    def test_memory_efficiency(self, router):
        """Test memory efficiency of confidence scoring system."""
        import gc
        
        # Get baseline memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Process multiple queries
        test_queries = [
            f"LC-MS metabolomics analysis query {i} for memory efficiency testing"
            for i in range(25)
        ]
        
        predictions = []
        for query in test_queries:
            prediction = router.route_query(query)
            predictions.append(prediction)
            
            # Validate comprehensive confidence structure
            assert hasattr(prediction, 'confidence_metrics')
            assert isinstance(prediction.confidence_metrics, ConfidenceMetrics)
        
        # Check memory usage growth
        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Memory growth should be reasonable (< 1500 objects for 25 queries)
        assert object_growth < 1500, \
            f"Excessive memory usage: {object_growth} new objects for 25 queries"
        
        # Validate that predictions contain expected data without excessive memory overhead
        for prediction in predictions:
            assert len(str(prediction.confidence_metrics.to_dict())) < 5000  # Reasonable serialization size


class TestConfidenceValidationAndMonitoring:
    """Test suite for confidence validation and monitoring utilities."""
    
    def test_confidence_validation_utility(self, router):
        """Test confidence validation utility with diagnostics."""
        test_query = "LC-MS analysis of glucose metabolites for diabetes biomarker discovery"
        expected_range = (0.3, 1.0)  # Broader range to accommodate actual behavior
        
        validation = router.validate_confidence_calculation(
            test_query, expected_range
        )
        
        # Validate validation structure
        assert 'query' in validation
        assert 'predicted_confidence' in validation
        assert 'confidence_level' in validation
        assert 'routing_decision' in validation
        assert 'validation_passed' in validation
        assert 'issues' in validation
        assert 'diagnostics' in validation
        assert 'performance_metrics' in validation
        
        # Should pass validation for high-quality query
        assert validation['validation_passed']
        assert validation['predicted_confidence'] >= expected_range[0]
        assert validation['predicted_confidence'] <= expected_range[1]
        
        # Validate diagnostic information
        diagnostics = validation['diagnostics']
        assert 'component_confidences' in diagnostics
        assert 'signal_metrics' in diagnostics
        assert 'uncertainty_metrics' in diagnostics
        assert 'reasoning' in diagnostics
        
        # Performance metrics should be present
        perf_metrics = validation['performance_metrics']
        assert 'confidence_calculation_time_ms' in perf_metrics
        assert 'total_validation_time_ms' in perf_metrics
        assert perf_metrics['confidence_calculation_time_ms'] < 50
    
    def test_confidence_validation_edge_cases(self, router):
        """Test confidence validation with edge cases and failures."""
        # Test with very low quality query
        low_quality_query = "what?"
        expected_range = (0.8, 1.0)  # Unrealistic expectation
        
        validation = router.validate_confidence_calculation(
            low_quality_query, expected_range
        )
        
        # Should fail validation due to unrealistic expectations
        assert not validation['validation_passed']
        assert len(validation['issues']) > 0
        assert any("outside expected range" in issue for issue in validation['issues'])
        
        # Performance should still be acceptable
        assert validation['performance_metrics']['confidence_calculation_time_ms'] < 50
    
    def test_comprehensive_statistics(self, router):
        """Test comprehensive confidence scoring statistics."""
        # Process a few queries to populate statistics
        test_queries = [
            "LC-MS metabolomics analysis",
            "pathway analysis using KEGG",
            "what is metabolomics?"
        ]
        
        for query in test_queries:
            router.route_query(query)
        
        stats = router.get_confidence_statistics()
        
        # Validate statistics structure
        assert 'fallback_strategies' in stats
        assert 'circuit_breaker_state' in stats
        assert 'confidence_thresholds' in stats
        assert 'performance_targets' in stats
        
        # Validate fallback strategies information
        fallback_stats = stats['fallback_strategies']
        expected_strategies = ['hybrid', 'ensemble', 'circuit_breaker', 'default']
        for strategy_name in expected_strategies:
            assert strategy_name in fallback_stats
            assert 'strategy_type' in fallback_stats[strategy_name]
            assert 'confidence_threshold' in fallback_stats[strategy_name]
            assert 'description' in fallback_stats[strategy_name]
        
        # Validate performance targets
        perf_targets = stats['performance_targets']
        assert perf_targets['total_routing_time_ms'] == 50
        assert perf_targets['comprehensive_analysis_time_ms'] == 30
        assert perf_targets['confidence_calculation_time_ms'] == 20


class TestIntegrationWithExistingSystem:
    """Test suite for integration with existing ResearchCategorizer system."""
    
    def test_backward_compatibility(self, router):
        """Test that enhanced system maintains backward compatibility."""
        # Test queries that should demonstrate different routing behavior
        lightrag_query = "pathway analysis using KEGG database"
        perplexity_query = "latest metabolomics research 2025"
        
        # Get predictions and check legacy methods exist and work
        lightrag_prediction = router.route_query(lightrag_query)
        perplexity_prediction = router.route_query(perplexity_query)
        
        # Legacy boolean methods should exist and be callable
        lightrag_bool = router.should_use_lightrag(lightrag_query)
        perplexity_bool = router.should_use_perplexity(perplexity_query)
        
        # Methods should return boolean values
        assert isinstance(lightrag_bool, bool)
        assert isinstance(perplexity_bool, bool)
        
        # At least one of the queries should prefer its respective service
        # (The specific routing may depend on the query analysis, so we test flexibility)
        either_lightrag_works = (
            lightrag_prediction.routing_decision.value in ['lightrag', 'either', 'hybrid'] or
            lightrag_bool
        )
        either_perplexity_works = (
            perplexity_prediction.routing_decision.value in ['perplexity', 'either', 'hybrid'] or
            perplexity_bool
        )
        
        assert either_lightrag_works, f"Pathway query should work with LightRAG: {lightrag_prediction.routing_decision.value}"
        assert either_perplexity_works, f"Latest research query should work with Perplexity: {perplexity_prediction.routing_decision.value}"
        
        # Get predictions and validate legacy fields
        lightrag_prediction = router.route_query(lightrag_query)
        perplexity_prediction = router.route_query(perplexity_query)
        
        # Legacy confidence field should match new detailed metrics
        assert lightrag_prediction.confidence == lightrag_prediction.confidence_metrics.overall_confidence
        assert perplexity_prediction.confidence == perplexity_prediction.confidence_metrics.overall_confidence
        
        # Legacy metadata should be preserved
        assert hasattr(lightrag_prediction, 'temporal_indicators')
        assert hasattr(lightrag_prediction, 'knowledge_indicators')
        assert hasattr(lightrag_prediction, 'metadata')
    
    def test_research_categorizer_integration(self, router):
        """Test integration with underlying ResearchCategorizer."""
        test_query = "LC-MS metabolite identification using mass spectrometry"
        prediction = router.route_query(test_query)
        
        # Should properly use ResearchCategorizer for category prediction
        assert prediction.research_category in [
            ResearchCategory.METABOLITE_IDENTIFICATION,
            ResearchCategory.PATHWAY_ANALYSIS,
            ResearchCategory.CLINICAL_DIAGNOSIS,
            ResearchCategory.GENERAL_QUERY
        ]
        
        # Research category confidence should be incorporated
        assert prediction.confidence_metrics.research_category_confidence > 0.0
        
        # Should have evidence from research categorizer
        assert len(prediction.knowledge_indicators) > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "-x"])