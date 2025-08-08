#!/usr/bin/env python3
"""
Test Suite for Enhanced Confidence Scoring in Classification Results

This module provides comprehensive testing for the enhanced confidence scoring
functionality integrated with the query classification system.

Test Coverage:
- EnhancedClassificationResult creation and serialization
- Integration with HybridConfidenceResult
- EnhancedQueryClassificationEngine functionality
- Confidence calibration and validation
- Batch processing capabilities
- Routing integration compatibility
- Performance and accuracy validation

Author: Claude Code (Anthropic)
Created: 2025-08-08
Task: CMO-LIGHTRAG-012-T06 - Add confidence scoring for classification results
"""

import asyncio
import pytest
import logging
import time
import json
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock

# Configure test logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Import components to test
try:
    from query_classification_system import (
        QueryClassificationCategories,
        ClassificationResult,
        EnhancedClassificationResult,
        EnhancedQueryClassificationEngine,
        create_enhanced_classification_engine,
        classify_with_enhanced_confidence,
        integrate_enhanced_classification_with_routing
    )
    
    # Import comprehensive confidence scoring components
    from comprehensive_confidence_scorer import (
        HybridConfidenceResult,
        LLMConfidenceAnalysis,
        KeywordConfidenceAnalysis,
        HybridConfidenceScorer,
        ConfidenceSource
    )
    
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestEnhancedClassificationResult:
    """Test suite for EnhancedClassificationResult class."""
    
    def test_enhanced_result_creation(self):
        """Test creating EnhancedClassificationResult with all fields."""
        
        result = EnhancedClassificationResult(
            category=QueryClassificationCategories.KNOWLEDGE_GRAPH,
            confidence=0.85,
            reasoning=["High biomedical content", "Clear pathway terminology"],
            keyword_match_confidence=0.9,
            pattern_match_confidence=0.8,
            semantic_confidence=0.85,
            temporal_confidence=0.1,
            matched_keywords=["pathway", "biomarker"],
            matched_patterns=["metabolic pathway"],
            biomedical_entities=["glucose", "insulin"],
            temporal_indicators=[],
            alternative_classifications=[
                (QueryClassificationCategories.KNOWLEDGE_GRAPH, 0.85),
                (QueryClassificationCategories.GENERAL, 0.15)
            ],
            classification_time_ms=150.0,
            ambiguity_score=0.2,
            conflict_score=0.1,
            confidence_interval=(0.75, 0.95),
            epistemic_uncertainty=0.15,
            aleatoric_uncertainty=0.10,
            total_uncertainty=0.20,
            confidence_reliability=0.85,
            evidence_strength=0.90
        )
        
        assert result.category == QueryClassificationCategories.KNOWLEDGE_GRAPH
        assert result.confidence == 0.85
        assert result.confidence_interval == (0.75, 0.95)
        assert result.epistemic_uncertainty == 0.15
        assert result.confidence_reliability == 0.85
        assert result.evidence_strength == 0.90
    
    def test_enhanced_result_from_basic_classification(self):
        """Test creating EnhancedClassificationResult from basic ClassificationResult."""
        
        # Create basic classification result
        basic_result = ClassificationResult(
            category=QueryClassificationCategories.REAL_TIME,
            confidence=0.7,
            reasoning=["Temporal indicators found"],
            keyword_match_confidence=0.8,
            pattern_match_confidence=0.6,
            semantic_confidence=0.7,
            temporal_confidence=0.9,
            matched_keywords=["latest", "2025"],
            matched_patterns=["latest.*research"],
            biomedical_entities=["metabolomics"],
            temporal_indicators=["latest", "2025"],
            alternative_classifications=[
                (QueryClassificationCategories.REAL_TIME, 0.7),
                (QueryClassificationCategories.GENERAL, 0.3)
            ],
            classification_time_ms=120.0,
            ambiguity_score=0.3,
            conflict_score=0.2
        )
        
        # Convert to enhanced result
        enhanced_result = EnhancedClassificationResult.from_basic_classification(basic_result)
        
        # Verify basic fields are preserved
        assert enhanced_result.category == basic_result.category
        assert enhanced_result.confidence == basic_result.confidence
        assert enhanced_result.reasoning == basic_result.reasoning
        assert enhanced_result.matched_keywords == basic_result.matched_keywords
        assert enhanced_result.classification_time_ms == basic_result.classification_time_ms
        
        # Verify enhanced fields have defaults
        assert enhanced_result.confidence_interval is not None
        assert enhanced_result.llm_weight == 0.5
        assert enhanced_result.keyword_weight == 0.5
        assert enhanced_result.confidence_reliability == 0.5
    
    def test_enhanced_result_with_hybrid_confidence(self):
        """Test EnhancedClassificationResult integration with HybridConfidenceResult."""
        
        # Create mock hybrid confidence result
        mock_llm_analysis = LLMConfidenceAnalysis(
            raw_confidence=0.8,
            calibrated_confidence=0.75,
            reasoning_quality_score=0.9,
            consistency_score=0.85,
            response_length=150,
            reasoning_depth=3,
            uncertainty_indicators=[],
            confidence_expressions=["high confidence"]
        )
        
        mock_keyword_analysis = KeywordConfidenceAnalysis(
            pattern_match_confidence=0.8,
            keyword_density_confidence=0.9,
            biomedical_entity_confidence=0.85,
            domain_specificity_confidence=0.8,
            total_biomedical_signals=5,
            strong_signals=3,
            weak_signals=2,
            conflicting_signals=0,
            semantic_coherence_score=0.85,
            domain_alignment_score=0.9,
            query_completeness_score=0.8
        )
        
        mock_hybrid_result = HybridConfidenceResult(
            overall_confidence=0.82,
            confidence_interval=(0.72, 0.92),
            llm_confidence=mock_llm_analysis,
            keyword_confidence=mock_keyword_analysis,
            llm_weight=0.6,
            keyword_weight=0.4,
            calibration_adjustment=0.02,
            epistemic_uncertainty=0.18,
            aleatoric_uncertainty=0.15,
            total_uncertainty=0.25,
            confidence_reliability=0.88,
            evidence_strength=0.85,
            alternative_confidences=[("llm_only", 0.75), ("keyword_only", 0.90)],
            calculation_time_ms=45.0,
            calibration_version="1.0"
        )
        
        # Create basic result
        basic_result = ClassificationResult(
            category=QueryClassificationCategories.KNOWLEDGE_GRAPH,
            confidence=0.7,  # Will be overridden
            reasoning=["Test reasoning"],
            keyword_match_confidence=0.8,
            pattern_match_confidence=0.7,
            semantic_confidence=0.75,
            temporal_confidence=0.1,
            matched_keywords=["pathway"],
            matched_patterns=["metabolic"],
            biomedical_entities=["glucose"],
            temporal_indicators=[],
            alternative_classifications=[],
            classification_time_ms=100.0,
            ambiguity_score=0.2,
            conflict_score=0.1
        )
        
        # Create enhanced result with hybrid confidence
        enhanced_result = EnhancedClassificationResult.from_basic_classification(
            basic_result, mock_hybrid_result
        )
        
        # Verify hybrid confidence integration
        assert enhanced_result.confidence == mock_hybrid_result.overall_confidence
        assert enhanced_result.confidence_interval == mock_hybrid_result.confidence_interval
        assert enhanced_result.llm_confidence_analysis == mock_llm_analysis
        assert enhanced_result.keyword_confidence_analysis == mock_keyword_analysis
        assert enhanced_result.llm_weight == 0.6
        assert enhanced_result.keyword_weight == 0.4
        assert enhanced_result.calibration_adjustment == 0.02
        assert enhanced_result.confidence_reliability == 0.88
        assert enhanced_result.evidence_strength == 0.85
    
    def test_enhanced_result_to_dict_serialization(self):
        """Test serialization of EnhancedClassificationResult to dictionary."""
        
        result = EnhancedClassificationResult(
            category=QueryClassificationCategories.GENERAL,
            confidence=0.6,
            reasoning=["Basic query"],
            keyword_match_confidence=0.5,
            pattern_match_confidence=0.4,
            semantic_confidence=0.6,
            temporal_confidence=0.0,
            matched_keywords=["define"],
            matched_patterns=["what.*is"],
            biomedical_entities=["metabolomics"],
            temporal_indicators=[],
            alternative_classifications=[
                (QueryClassificationCategories.GENERAL, 0.6),
                (QueryClassificationCategories.KNOWLEDGE_GRAPH, 0.4)
            ],
            classification_time_ms=80.0,
            ambiguity_score=0.4,
            conflict_score=0.0,
            confidence_interval=(0.5, 0.7),
            alternative_confidence_scenarios=[("conservative", 0.5), ("optimistic", 0.75)]
        )
        
        result_dict = result.to_dict()
        
        # Verify basic structure
        assert result_dict['category'] == 'general'
        assert result_dict['confidence'] == 0.6
        assert 'enhanced_confidence_scoring' in result_dict
        
        # Verify enhanced confidence structure
        enhanced = result_dict['enhanced_confidence_scoring']
        assert 'confidence_interval' in enhanced
        assert enhanced['confidence_interval']['lower_bound'] == 0.5
        assert enhanced['confidence_interval']['upper_bound'] == 0.7
        assert 'uncertainty_metrics' in enhanced
        assert 'quality_indicators' in enhanced
        assert 'alternative_scenarios' in enhanced
    
    def test_enhanced_result_backwards_compatibility(self):
        """Test that EnhancedClassificationResult can convert back to basic format."""
        
        enhanced_result = EnhancedClassificationResult(
            category=QueryClassificationCategories.KNOWLEDGE_GRAPH,
            confidence=0.85,
            reasoning=["High confidence classification"],
            keyword_match_confidence=0.9,
            pattern_match_confidence=0.8,
            semantic_confidence=0.85,
            temporal_confidence=0.1,
            matched_keywords=["pathway", "mechanism"],
            matched_patterns=["metabolic pathway"],
            biomedical_entities=["glucose", "insulin"],
            temporal_indicators=[],
            alternative_classifications=[
                (QueryClassificationCategories.KNOWLEDGE_GRAPH, 0.85),
                (QueryClassificationCategories.GENERAL, 0.15)
            ],
            classification_time_ms=150.0,
            ambiguity_score=0.15,
            conflict_score=0.05,
            confidence_interval=(0.8, 0.9)
        )
        
        # Convert back to basic format
        basic_result = enhanced_result.to_basic_classification()
        
        # Verify all basic fields are preserved
        assert basic_result.category == enhanced_result.category
        assert basic_result.confidence == enhanced_result.confidence
        assert basic_result.reasoning == enhanced_result.reasoning
        assert basic_result.keyword_match_confidence == enhanced_result.keyword_match_confidence
        assert basic_result.matched_keywords == enhanced_result.matched_keywords
        assert basic_result.classification_time_ms == enhanced_result.classification_time_ms
        assert basic_result.ambiguity_score == enhanced_result.ambiguity_score
        
        # Verify it's the correct type
        assert isinstance(basic_result, ClassificationResult)
        assert not isinstance(basic_result, EnhancedClassificationResult)
    
    def test_confidence_summary_and_recommendations(self):
        """Test confidence summary and recommendation methods."""
        
        # High confidence result
        high_conf_result = EnhancedClassificationResult(
            category=QueryClassificationCategories.KNOWLEDGE_GRAPH,
            confidence=0.9,
            reasoning=["Very high confidence"],
            keyword_match_confidence=0.9, pattern_match_confidence=0.85,
            semantic_confidence=0.9, temporal_confidence=0.1,
            matched_keywords=[], matched_patterns=[], biomedical_entities=[],
            temporal_indicators=[], alternative_classifications=[],
            classification_time_ms=100.0, ambiguity_score=0.1, conflict_score=0.05,
            confidence_interval=(0.85, 0.95),
            confidence_reliability=0.9, evidence_strength=0.85,
            total_uncertainty=0.15
        )
        
        summary = high_conf_result.get_confidence_summary()
        assert summary['overall_confidence'] == 0.9
        assert summary['reliability'] == 0.9
        assert summary['uncertainty'] == 0.15
        
        assert high_conf_result.is_high_confidence()
        
        recommendation = high_conf_result.get_recommendation()
        assert recommendation['confidence_level'] == 'high'
        assert 'high confidence' in recommendation['recommendation'].lower()
        
        # Low confidence result
        low_conf_result = EnhancedClassificationResult(
            category=QueryClassificationCategories.GENERAL,
            confidence=0.3,
            reasoning=["Uncertain classification"],
            keyword_match_confidence=0.2, pattern_match_confidence=0.3,
            semantic_confidence=0.4, temporal_confidence=0.0,
            matched_keywords=[], matched_patterns=[], biomedical_entities=[],
            temporal_indicators=[], alternative_classifications=[],
            classification_time_ms=200.0, ambiguity_score=0.7, conflict_score=0.5,
            confidence_interval=(0.2, 0.4),
            confidence_reliability=0.4, evidence_strength=0.3,
            total_uncertainty=0.8
        )
        
        assert not low_conf_result.is_high_confidence()
        
        low_recommendation = low_conf_result.get_recommendation()
        assert low_recommendation['confidence_level'] in ['low', 'uncertain']


class TestEnhancedQueryClassificationEngine:
    """Test suite for EnhancedQueryClassificationEngine class."""
    
    @pytest.fixture
    async def mock_engine(self):
        """Create a mock enhanced classification engine for testing."""
        
        engine = EnhancedQueryClassificationEngine(
            logger=logger,
            enable_hybrid_confidence=False  # Disable for simpler testing
        )
        
        return engine
    
    @pytest.fixture
    async def mock_hybrid_engine(self):
        """Create a mock enhanced classification engine with hybrid confidence."""
        
        with patch('query_classification_system.create_hybrid_confidence_scorer') as mock_scorer:
            # Mock the confidence scorer
            mock_scorer_instance = Mock()
            mock_scorer_instance.calculate_comprehensive_confidence = AsyncMock(return_value=Mock(
                overall_confidence=0.8,
                confidence_interval=(0.7, 0.9),
                llm_confidence=Mock(raw_confidence=0.75, calibrated_confidence=0.8),
                keyword_confidence=Mock(pattern_match_confidence=0.85),
                llm_weight=0.6, keyword_weight=0.4,
                calibration_adjustment=0.05,
                epistemic_uncertainty=0.2, aleatoric_uncertainty=0.15, total_uncertainty=0.25,
                confidence_reliability=0.8, evidence_strength=0.75,
                alternative_confidences=[("llm_only", 0.75)],
                calculation_time_ms=50.0, calibration_version="1.0"
            ))
            
            mock_scorer.return_value = mock_scorer_instance
            
            engine = EnhancedQueryClassificationEngine(
                logger=logger,
                enable_hybrid_confidence=True
            )
            
            yield engine
    
    @pytest.mark.asyncio
    async def test_enhanced_engine_basic_classification(self, mock_engine):
        """Test basic classification functionality."""
        
        query = "What is clinical metabolomics?"
        
        # Test basic classification (should work without enhanced confidence)
        basic_result = mock_engine.classify_query(query)
        
        assert isinstance(basic_result, ClassificationResult)
        assert basic_result.category in QueryClassificationCategories
        assert 0.0 <= basic_result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_enhanced_classification_without_hybrid(self, mock_engine):
        """Test enhanced classification without hybrid confidence scoring."""
        
        query = "What are metabolic pathways in glucose metabolism?"
        
        enhanced_result = await mock_engine.classify_query_enhanced(query)
        
        assert isinstance(enhanced_result, EnhancedClassificationResult)
        assert enhanced_result.category in QueryClassificationCategories
        assert 0.0 <= enhanced_result.confidence <= 1.0
        assert enhanced_result.confidence_interval is not None
        assert enhanced_result.classification_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_classification_with_hybrid(self, mock_hybrid_engine):
        """Test enhanced classification with hybrid confidence scoring."""
        
        query = "Latest metabolomics research developments in 2025"
        
        enhanced_result = await mock_hybrid_engine.classify_query_enhanced(query)
        
        assert isinstance(enhanced_result, EnhancedClassificationResult)
        assert enhanced_result.category in QueryClassificationCategories
        assert 0.0 <= enhanced_result.confidence <= 1.0
        assert enhanced_result.confidence_interval is not None
        assert enhanced_result.llm_confidence_analysis is not None
        assert enhanced_result.confidence_calculation_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_batch_classification(self, mock_engine):
        """Test batch classification functionality."""
        
        queries = [
            "What is metabolomics?",
            "Latest LC-MS developments 2025",
            "Glucose metabolic pathways mechanisms"
        ]
        
        batch_results = await mock_engine.batch_classify_enhanced(queries)
        
        assert len(batch_results) == len(queries)
        
        for result in batch_results:
            assert isinstance(result, EnhancedClassificationResult)
            assert result.category in QueryClassificationCategories
            assert 0.0 <= result.confidence <= 1.0
    
    def test_confidence_validation(self, mock_engine):
        """Test confidence validation functionality."""
        
        # Create a mock enhanced result
        enhanced_result = EnhancedClassificationResult(
            category=QueryClassificationCategories.KNOWLEDGE_GRAPH,
            confidence=0.8,
            reasoning=["High biomedical content"],
            keyword_match_confidence=0.8, pattern_match_confidence=0.7,
            semantic_confidence=0.8, temporal_confidence=0.1,
            matched_keywords=[], matched_patterns=[], biomedical_entities=[],
            temporal_indicators=[], alternative_classifications=[],
            classification_time_ms=100.0, ambiguity_score=0.2, conflict_score=0.1,
            confidence_interval=(0.7, 0.9)
        )
        
        validation_result = mock_engine.validate_confidence_accuracy(
            query_text="Test query about metabolic pathways",
            predicted_result=enhanced_result,
            actual_category=QueryClassificationCategories.KNOWLEDGE_GRAPH,
            actual_routing_success=True
        )
        
        assert 'query' in validation_result
        assert 'predicted_category' in validation_result
        assert 'actual_category' in validation_result
        assert 'category_correct' in validation_result
        assert 'confidence_error' in validation_result
        assert validation_result['category_correct'] is True
    
    def test_system_statistics(self, mock_engine):
        """Test system statistics functionality."""
        
        stats = mock_engine.get_confidence_calibration_stats()
        
        assert 'confidence_scoring_enabled' in stats
        # With mock engine, hybrid confidence should be disabled
        assert stats['confidence_scoring_enabled'] is False


class TestRoutingIntegration:
    """Test suite for routing integration functionality."""
    
    def test_routing_integration_basic(self):
        """Test basic routing integration."""
        
        enhanced_result = EnhancedClassificationResult(
            category=QueryClassificationCategories.REAL_TIME,
            confidence=0.75,
            reasoning=["Temporal indicators detected"],
            keyword_match_confidence=0.8, pattern_match_confidence=0.7,
            semantic_confidence=0.7, temporal_confidence=0.9,
            matched_keywords=["latest", "2025"], matched_patterns=[],
            biomedical_entities=["metabolomics"], temporal_indicators=["latest"],
            alternative_classifications=[
                (QueryClassificationCategories.REAL_TIME, 0.75),
                (QueryClassificationCategories.GENERAL, 0.25)
            ],
            classification_time_ms=120.0, ambiguity_score=0.25, conflict_score=0.1,
            confidence_interval=(0.65, 0.85),
            confidence_reliability=0.75, evidence_strength=0.8,
            total_uncertainty=0.3
        )
        
        routing_info = integrate_enhanced_classification_with_routing(enhanced_result)
        
        assert 'routing_decision' in routing_info
        assert routing_info['routing_decision']['primary_route'] == 'perplexity'
        assert routing_info['routing_decision']['category'] == 'real_time'
        assert 'confidence_metrics' in routing_info
        assert 'recommendation' in routing_info
        assert 'fallback_routes' in routing_info
    
    def test_routing_integration_high_confidence(self):
        """Test routing integration with high confidence result."""
        
        high_conf_result = EnhancedClassificationResult(
            category=QueryClassificationCategories.KNOWLEDGE_GRAPH,
            confidence=0.92,
            reasoning=["Clear pathway terminology", "High biomedical content"],
            keyword_match_confidence=0.9, pattern_match_confidence=0.85,
            semantic_confidence=0.95, temporal_confidence=0.1,
            matched_keywords=["pathway", "mechanism"], matched_patterns=["metabolic"],
            biomedical_entities=["glucose", "insulin"], temporal_indicators=[],
            alternative_classifications=[
                (QueryClassificationCategories.KNOWLEDGE_GRAPH, 0.92),
                (QueryClassificationCategories.GENERAL, 0.08)
            ],
            classification_time_ms=100.0, ambiguity_score=0.08, conflict_score=0.02,
            confidence_interval=(0.88, 0.96),
            confidence_reliability=0.9, evidence_strength=0.88,
            total_uncertainty=0.12
        )
        
        routing_info = integrate_enhanced_classification_with_routing(high_conf_result)
        
        assert routing_info['routing_decision']['primary_route'] == 'lightrag'
        assert routing_info['should_use_hybrid'] is False  # Low uncertainty
        assert routing_info['requires_clarification'] is False  # High confidence
    
    def test_routing_integration_uncertain(self):
        """Test routing integration with uncertain result."""
        
        uncertain_result = EnhancedClassificationResult(
            category=QueryClassificationCategories.GENERAL,
            confidence=0.35,
            reasoning=["Ambiguous query content"],
            keyword_match_confidence=0.3, pattern_match_confidence=0.2,
            semantic_confidence=0.4, temporal_confidence=0.0,
            matched_keywords=[], matched_patterns=[], biomedical_entities=[],
            temporal_indicators=[], alternative_classifications=[
                (QueryClassificationCategories.GENERAL, 0.35),
                (QueryClassificationCategories.KNOWLEDGE_GRAPH, 0.33),
                (QueryClassificationCategories.REAL_TIME, 0.32)
            ],
            classification_time_ms=180.0, ambiguity_score=0.8, conflict_score=0.6,
            confidence_interval=(0.25, 0.45),
            confidence_reliability=0.4, evidence_strength=0.3,
            total_uncertainty=0.75
        )
        
        routing_info = integrate_enhanced_classification_with_routing(uncertain_result)
        
        assert routing_info['should_use_hybrid'] is True  # High uncertainty
        assert routing_info['requires_clarification'] is True  # Low confidence
        
        recommendation = routing_info['recommendation']
        assert recommendation['confidence_level'] in ['low', 'uncertain']


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_create_enhanced_classification_engine(self):
        """Test enhanced classification engine factory function."""
        
        # Test with hybrid confidence disabled
        engine = await create_enhanced_classification_engine(
            logger=logger,
            enable_hybrid_confidence=False
        )
        
        assert isinstance(engine, EnhancedQueryClassificationEngine)
        assert engine.enable_hybrid_confidence is False
    
    @pytest.mark.asyncio
    async def test_classify_with_enhanced_confidence_function(self):
        """Test convenience classification function."""
        
        query = "What is metabolomics analysis?"
        
        # Create a mock engine for the test
        mock_engine = Mock()
        mock_result = EnhancedClassificationResult(
            category=QueryClassificationCategories.GENERAL,
            confidence=0.6,
            reasoning=["General query"],
            keyword_match_confidence=0.5, pattern_match_confidence=0.4,
            semantic_confidence=0.6, temporal_confidence=0.0,
            matched_keywords=[], matched_patterns=[], biomedical_entities=[],
            temporal_indicators=[], alternative_classifications=[],
            classification_time_ms=100.0, ambiguity_score=0.4, conflict_score=0.1,
            confidence_interval=(0.5, 0.7)
        )
        
        mock_engine.classify_query_enhanced = AsyncMock(return_value=mock_result)
        
        result = await classify_with_enhanced_confidence(query, engine=mock_engine)
        
        assert isinstance(result, EnhancedClassificationResult)
        assert result.category == QueryClassificationCategories.GENERAL
        assert result.confidence == 0.6


class TestPerformanceAndAccuracy:
    """Test suite for performance and accuracy validation."""
    
    @pytest.mark.asyncio
    async def test_classification_performance(self):
        """Test classification performance meets requirements."""
        
        engine = EnhancedQueryClassificationEngine(
            logger=logger,
            enable_hybrid_confidence=False  # Disable for performance testing
        )
        
        test_queries = [
            "What is clinical metabolomics?",
            "Latest metabolomics research 2025",
            "Glucose metabolic pathway mechanisms",
            "LC-MS analysis methods",
            "Biomarker discovery techniques"
        ]
        
        total_time = 0
        results = []
        
        for query in test_queries:
            start_time = time.time()
            result = await engine.classify_query_enhanced(query)
            elapsed_time = (time.time() - start_time) * 1000
            
            total_time += elapsed_time
            results.append(result)
            
            # Individual query should be fast
            assert elapsed_time < 2000  # Less than 2 seconds
        
        # Average time should be reasonable
        avg_time = total_time / len(test_queries)
        assert avg_time < 1000  # Average less than 1 second
        
        # All results should be valid
        for result in results:
            assert isinstance(result, EnhancedClassificationResult)
            assert result.category in QueryClassificationCategories
            assert 0.0 <= result.confidence <= 1.0
            assert result.classification_time_ms > 0
    
    def test_confidence_score_validity(self):
        """Test that confidence scores are valid and reasonable."""
        
        # Test various confidence scenarios
        test_scenarios = [
            {
                'confidence': 0.95,
                'reliability': 0.9,
                'uncertainty': 0.1,
                'expected_high_confidence': True
            },
            {
                'confidence': 0.45,
                'reliability': 0.5,
                'uncertainty': 0.6,
                'expected_high_confidence': False
            },
            {
                'confidence': 0.2,
                'reliability': 0.3,
                'uncertainty': 0.8,
                'expected_high_confidence': False
            }
        ]
        
        for scenario in test_scenarios:
            result = EnhancedClassificationResult(
                category=QueryClassificationCategories.GENERAL,
                confidence=scenario['confidence'],
                reasoning=["Test reasoning"],
                keyword_match_confidence=0.5, pattern_match_confidence=0.5,
                semantic_confidence=0.5, temporal_confidence=0.0,
                matched_keywords=[], matched_patterns=[], biomedical_entities=[],
                temporal_indicators=[], alternative_classifications=[],
                classification_time_ms=100.0, ambiguity_score=0.3, conflict_score=0.2,
                confidence_interval=(scenario['confidence'] - 0.1, scenario['confidence'] + 0.1),
                confidence_reliability=scenario['reliability'],
                evidence_strength=0.5,
                total_uncertainty=scenario['uncertainty']
            )
            
            # Validate confidence bounds
            assert 0.0 <= result.confidence <= 1.0
            assert 0.0 <= result.confidence_reliability <= 1.0
            assert 0.0 <= result.total_uncertainty <= 1.0
            
            # Validate interval consistency
            assert result.confidence_interval[0] <= result.confidence <= result.confidence_interval[1]
            
            # Validate high confidence detection
            assert result.is_high_confidence() == scenario['expected_high_confidence']


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])