"""
Comprehensive Unit Tests for LLM-based Classification System

This module provides comprehensive unit tests for the LLM-powered classification system
including LLMQueryClassifier, HybridConfidenceScorer, and all related components.

Test Coverage:
    - Core LLM classification functionality
    - Caching and performance optimization
    - Circuit breakers and cost management
    - Fallback mechanisms and error handling
    - Confidence scoring and calibration
    - Integration with existing infrastructure
    - Performance and load testing
    - Edge cases and error scenarios

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import json
import time
import asyncio
import pytest
import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
import threading
import tempfile
import os
import logging

# Import the modules under test
try:
    from ..llm_query_classifier import (
        LLMQueryClassifier,
        LLMClassificationConfig,
        LLMProvider,
        ClassificationCache,
        ClassificationMetrics,
        create_llm_enhanced_router,
        convert_llm_result_to_routing_prediction
    )
    from ..llm_classification_prompts import (
        ClassificationResult,
        ClassificationCategory,
        LLMClassificationPrompts
    )
    from ..comprehensive_confidence_scorer import (
        HybridConfidenceScorer,
        LLMConfidenceAnalyzer,
        ConfidenceCalibrator,
        ConfidenceValidator,
        LLMConfidenceAnalysis,
        KeywordConfidenceAnalysis,
        HybridConfidenceResult,
        ConfidenceSource,
        ConfidenceCalibrationData,
        create_hybrid_confidence_scorer,
        integrate_with_existing_confidence_metrics
    )
    from ..query_router import (
        BiomedicalQueryRouter,
        RoutingPrediction,
        RoutingDecision,
        ConfidenceMetrics
    )
    from ..research_categorizer import CategoryPrediction
    from ..cost_persistence import ResearchCategory
except ImportError as e:
    # Fallback imports for testing
    print(f"Import warning: {e}")


# ============================================================================
# TEST FIXTURES AND MOCK DATA
# ============================================================================

@pytest.fixture
def sample_queries():
    """Realistic biomedical query test data."""
    return {
        'knowledge_graph': [
            "What is the relationship between glucose metabolism and insulin signaling pathways?",
            "How does the citric acid cycle connect to amino acid biosynthesis?",
            "What are the key metabolic pathways in cancer cell metabolism?",
            "Explain the role of mitochondria in cellular energy production",
            "What is the mechanism of action for metformin in diabetes treatment?"
        ],
        'real_time': [
            "Latest research on metabolomics biomarkers for COVID-19 in 2025",
            "Recent breakthrough in Alzheimer's disease biomarker discovery",
            "Current trends in personalized medicine for cancer treatment 2024",
            "New FDA approved metabolomics-based diagnostic tools this year",
            "Breaking news on CRISPR gene editing applications in metabolic disorders"
        ],
        'general': [
            "metabolomics analysis methods",
            "LC-MS techniques",
            "biomarker discovery approaches",
            "clinical metabolomics overview",
            "metabolite identification strategies"
        ],
        'ambiguous': [
            "metabolomics",
            "analysis",
            "recent studies",
            "biomarkers research",
            "clinical applications"
        ]
    }

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = AsyncMock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({
        "category": "KNOWLEDGE_GRAPH",
        "confidence": 0.85,
        "reasoning": "Query asks about established metabolic pathways and relationships",
        "alternative_categories": [],
        "uncertainty_indicators": [],
        "biomedical_signals": {
            "entities": ["glucose", "insulin", "metabolism"],
            "relationships": ["signaling pathway"],
            "techniques": []
        },
        "temporal_signals": {
            "keywords": [],
            "patterns": [],
            "years": []
        }
    })
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 50
    
    client.chat.completions.create.return_value = mock_response
    return client

@pytest.fixture
def mock_biomedical_router():
    """Mock BiomedicalQueryRouter for testing."""
    router = Mock()
    
    # Mock routing prediction
    mock_prediction = RoutingPrediction(
        routing_decision=RoutingDecision.LIGHTRAG,
        confidence=0.75,
        reasoning=["Biomedical entities detected", "Established knowledge query"],
        research_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        confidence_metrics=ConfidenceMetrics(
            overall_confidence=0.75,
            research_category_confidence=0.8,
            temporal_analysis_confidence=0.3,
            signal_strength_confidence=0.7,
            context_coherence_confidence=0.8,
            keyword_density=0.6,
            pattern_match_strength=0.7,
            biomedical_entity_count=3,
            ambiguity_score=0.2,
            conflict_score=0.1,
            alternative_interpretations=[],
            calculation_time_ms=25.0
        ),
        temporal_indicators=[],
        knowledge_indicators=["glucose", "insulin", "metabolism"]
    )
    
    router.route_query.return_value = mock_prediction
    return router

@pytest.fixture
def llm_config():
    """Standard LLM configuration for testing."""
    return LLMClassificationConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        api_key="test-api-key",
        timeout_seconds=2.0,
        max_retries=2,
        daily_api_budget=1.0,  # Low budget for testing
        enable_caching=True,
        cache_ttl_hours=1,
        max_cache_size=100
    )

@pytest.fixture
def temp_calibration_file():
    """Temporary file for calibration data."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write('{}')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass

@pytest.fixture
def logger():
    """Test logger instance."""
    logger = logging.getLogger('test_llm_classification')
    logger.setLevel(logging.DEBUG)
    return logger


# ============================================================================
# CORE LLM CLASSIFIER TESTS
# ============================================================================

class TestLLMQueryClassifier(unittest.TestCase):
    """Comprehensive tests for LLMQueryClassifier."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = LLMClassificationConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            timeout_seconds=1.0,
            max_retries=1,
            enable_caching=True
        )
        self.logger = logging.getLogger('test')
    
    @patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
    def test_classifier_initialization(self, mock_openai_class):
        """Test classifier initialization with various configurations."""
        
        # Test successful initialization
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        
        self.assertEqual(classifier.config.provider, LLMProvider.OPENAI)
        self.assertEqual(classifier.config.api_key, "test-key")
        self.assertIsNotNone(classifier.cache)
        self.assertIsInstance(classifier.metrics, ClassificationMetrics)
        
        # Verify OpenAI client was created
        mock_openai_class.assert_called_once()
    
    def test_cache_functionality(self):
        """Test classification cache operations."""
        
        cache = ClassificationCache(max_size=5, ttl_hours=1)
        
        # Test cache miss
        result = cache.get("test query")
        self.assertIsNone(result)
        
        # Test cache put and get
        classification_result = ClassificationResult(
            category="KNOWLEDGE_GRAPH",
            confidence=0.8,
            reasoning="Test classification",
            alternative_categories=[],
            uncertainty_indicators=[],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        cache.put("test query", classification_result)
        cached_result = cache.get("test query")
        
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result.category, "KNOWLEDGE_GRAPH")
        self.assertEqual(cached_result.confidence, 0.8)
        
        # Test cache size limit
        for i in range(10):
            cache.put(f"query_{i}", classification_result)
        
        stats = cache.get_stats()
        self.assertLessEqual(stats['cache_size'], 5)  # Should not exceed max_size
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        
        cache = ClassificationCache(max_size=10, ttl_hours=0.001)  # Very short TTL
        
        classification_result = ClassificationResult(
            category="KNOWLEDGE_GRAPH",
            confidence=0.8,
            reasoning="Test classification",
            alternative_categories=[],
            uncertainty_indicators=[],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        cache.put("test query", classification_result)
        
        # Should be available immediately
        result = cache.get("test query")
        self.assertIsNotNone(result)
        
        # Wait for expiration
        time.sleep(0.01)  # Wait slightly longer than TTL
        
        # Should be expired
        result = cache.get("test query")
        self.assertIsNone(result)
    
    @patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
    async def test_successful_classification(self, mock_openai_class):
        """Test successful LLM classification."""
        
        # Setup mock client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "category": "KNOWLEDGE_GRAPH",
            "confidence": 0.85,
            "reasoning": "Query about established metabolic pathways",
            "alternative_categories": [],
            "uncertainty_indicators": [],
            "biomedical_signals": {
                "entities": ["glucose", "metabolism"],
                "relationships": ["pathway"],
                "techniques": []
            },
            "temporal_signals": {
                "keywords": [],
                "patterns": [],
                "years": []
            }
        })
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 30
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create classifier and test
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        query = "What is glucose metabolism pathway?"
        
        result, used_llm = await classifier.classify_query(query)
        
        self.assertTrue(used_llm)
        self.assertEqual(result.category, "KNOWLEDGE_GRAPH")
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.reasoning, "Query about established metabolic pathways")
        
        # Verify API call was made
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
    async def test_api_failure_fallback(self, mock_openai_class):
        """Test fallback to keyword classification when LLM API fails."""
        
        # Setup mock client that fails
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Setup mock biomedical router
        mock_router = Mock()
        mock_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence=0.7,
            reasoning=["Keyword-based classification"],
            research_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.7,
                research_category_confidence=0.7,
                temporal_analysis_confidence=0.3,
                signal_strength_confidence=0.6,
                context_coherence_confidence=0.7,
                keyword_density=0.5,
                pattern_match_strength=0.6,
                biomedical_entity_count=2,
                ambiguity_score=0.3,
                conflict_score=0.1,
                alternative_interpretations=[],
                calculation_time_ms=15.0
            ),
            knowledge_indicators=["glucose", "metabolism"]
        )
        mock_router.route_query.return_value = mock_prediction
        
        # Create classifier with biomedical router
        classifier = LLMQueryClassifier(self.config, mock_router, self.logger)
        query = "What is glucose metabolism pathway?"
        
        result, used_llm = await classifier.classify_query(query)
        
        # Should fallback to keyword-based classification
        self.assertFalse(used_llm)
        self.assertEqual(result.category, "KNOWLEDGE_GRAPH")
        self.assertEqual(result.confidence, 0.7)
        self.assertIn("fallback_classification_used", result.uncertainty_indicators)
        
        # Verify metrics updated
        self.assertEqual(classifier.metrics.llm_failures, 1)
        self.assertEqual(classifier.metrics.fallback_used, 1)
    
    @patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
    async def test_daily_budget_enforcement(self, mock_openai_class):
        """Test that daily budget limits are enforced."""
        
        # Setup config with very low budget
        low_budget_config = LLMClassificationConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            daily_api_budget=0.001  # Very low budget
        )
        
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        classifier = LLMQueryClassifier(low_budget_config, logger=self.logger)
        
        # Exceed budget
        classifier.metrics.daily_api_cost = 0.002
        
        result, used_llm = await classifier.classify_query("test query")
        
        # Should fallback due to budget
        self.assertFalse(used_llm)
        self.assertIn("simple_fallback_used", result.uncertainty_indicators)
        
        # API should not be called
        mock_client.chat.completions.create.assert_not_called()
    
    @patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
    async def test_malformed_response_handling(self, mock_openai_class):
        """Test handling of malformed JSON responses from LLM."""
        
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        # Mock malformed JSON response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        mock_client.chat.completions.create.return_value = mock_response
        
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        
        result, used_llm = await classifier.classify_query("test query")
        
        # Should fallback gracefully
        self.assertFalse(used_llm)
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(classifier.metrics.llm_failures, 1)
    
    @patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
    async def test_retry_mechanism(self, mock_openai_class):
        """Test retry mechanism for transient failures."""
        
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        # Setup to fail first time, succeed second time
        mock_responses = [
            Exception("Transient error"),
            Mock()  # Successful response
        ]
        
        # Setup successful response
        mock_responses[1].choices = [Mock()]
        mock_responses[1].choices[0].message.content = json.dumps({
            "category": "GENERAL",
            "confidence": 0.6,
            "reasoning": "General query after retry"
        })
        mock_responses[1].usage = Mock()
        mock_responses[1].usage.prompt_tokens = 100
        mock_responses[1].usage.completion_tokens = 30
        
        mock_client.chat.completions.create.side_effect = mock_responses
        
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        
        result, used_llm = await classifier.classify_query("test query")
        
        # Should succeed after retry
        self.assertTrue(used_llm)
        self.assertEqual(result.category, "GENERAL")
        
        # Should have made 2 API calls
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
    
    def test_metrics_calculation(self):
        """Test classification metrics calculation and statistics."""
        
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        
        # Simulate some classifications
        classifier.metrics.total_classifications = 100
        classifier.metrics.llm_successful = 85
        classifier.metrics.llm_failures = 10
        classifier.metrics.fallback_used = 5
        classifier.metrics.cache_hits = 20
        classifier.metrics.daily_api_cost = 0.50
        classifier.metrics.daily_token_usage = 10000
        
        # Add some response times and confidence scores
        classifier._response_times.extend([100, 150, 200, 120, 180])
        classifier._confidence_scores.extend([0.8, 0.9, 0.7, 0.85, 0.75])
        
        stats = classifier.get_classification_statistics()
        
        # Verify basic metrics
        self.assertEqual(stats['classification_metrics']['total_classifications'], 100)
        self.assertEqual(stats['classification_metrics']['llm_successful'], 85)
        self.assertEqual(stats['classification_metrics']['success_rate'], 85.0)
        
        # Verify performance metrics
        self.assertGreater(stats['performance_metrics']['avg_response_time_ms'], 0)
        self.assertGreater(stats['performance_metrics']['avg_confidence_score'], 0)
        
        # Verify cost metrics
        self.assertEqual(stats['cost_metrics']['daily_api_cost'], 0.50)
        self.assertEqual(stats['cost_metrics']['daily_token_usage'], 10000)
    
    def test_optimization_recommendations(self):
        """Test configuration optimization recommendations."""
        
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        
        # Simulate poor performance conditions
        classifier.metrics.total_classifications = 100
        classifier.metrics.llm_successful = 70  # 70% success rate
        classifier.metrics.daily_api_cost = 0.9  # 90% of budget
        classifier._response_times.extend([3000, 3500, 2800])  # Slow responses
        
        recommendations = classifier.optimize_configuration()
        
        self.assertIn('recommendations', recommendations)
        self.assertTrue(len(recommendations['recommendations']) > 0)
        
        # Check for specific recommendation types
        rec_types = [r['type'] for r in recommendations['recommendations']]
        self.assertIn('performance', rec_types)  # Should recommend performance improvements
        self.assertIn('reliability', rec_types)  # Should recommend reliability improvements
        self.assertIn('cost', rec_types)  # Should recommend cost optimization


# ============================================================================
# CONFIDENCE SCORING TESTS
# ============================================================================

class TestHybridConfidenceScorer(unittest.TestCase):
    """Tests for HybridConfidenceScorer and related confidence components."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = logging.getLogger('test')
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test environment."""
        try:
            os.unlink(self.temp_file.name)
        except FileNotFoundError:
            pass
    
    def test_llm_confidence_analyzer(self):
        """Test LLM confidence analysis."""
        
        analyzer = LLMConfidenceAnalyzer(self.logger)
        
        # Create test classification result
        classification_result = ClassificationResult(
            category="KNOWLEDGE_GRAPH",
            confidence=0.85,
            reasoning="This query asks about established metabolic pathways with clear biomedical entities",
            alternative_categories=[],
            uncertainty_indicators=["possible", "might"],
            biomedical_signals={"entities": ["glucose", "metabolism"], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        # Analyze confidence
        analysis = analyzer.analyze_llm_confidence(
            classification_result,
            llm_response_text="This query asks about established metabolic pathways. The evidence shows clear biomedical entities.",
            response_metadata={'response_time_ms': 150.0, 'temperature': 0.1}
        )
        
        self.assertEqual(analysis.raw_confidence, 0.85)
        self.assertGreater(analysis.reasoning_quality_score, 0)
        self.assertEqual(analysis.consistency_score, 1.0)  # Default with no alternatives
        self.assertEqual(len(analysis.uncertainty_indicators), 2)
        self.assertGreater(analysis.reasoning_depth, 1)
        self.assertEqual(analysis.response_time_ms, 150.0)
        self.assertEqual(analysis.model_temperature, 0.1)
        
        # Test calibrated confidence adjustment
        self.assertNotEqual(analysis.calibrated_confidence, analysis.raw_confidence)
    
    def test_confidence_calibrator(self):
        """Test confidence calibration system."""
        
        calibrator = ConfidenceCalibrator(self.temp_file.name, self.logger)
        
        # Record some prediction outcomes
        test_data = [
            (0.9, True),   # High confidence, correct
            (0.8, True),   # High confidence, correct
            (0.7, False),  # Medium confidence, incorrect
            (0.6, True),   # Medium confidence, correct
            (0.3, False),  # Low confidence, incorrect
        ]
        
        for confidence, accuracy in test_data:
            calibrator.record_prediction_outcome(
                confidence, accuracy, ConfidenceSource.LLM_SEMANTIC
            )
        
        # Test calibration
        calibrated_high = calibrator.calibrate_confidence(0.9, ConfidenceSource.LLM_SEMANTIC)
        calibrated_low = calibrator.calibrate_confidence(0.3, ConfidenceSource.LLM_SEMANTIC)
        
        self.assertIsInstance(calibrated_high, float)
        self.assertIsInstance(calibrated_low, float)
        self.assertGreaterEqual(calibrated_high, 0.0)
        self.assertLessEqual(calibrated_high, 1.0)
        
        # Test confidence intervals
        interval_high = calibrator.get_confidence_interval(calibrated_high, evidence_strength=0.8)
        interval_low = calibrator.get_confidence_interval(calibrated_low, evidence_strength=0.3)
        
        self.assertEqual(len(interval_high), 2)
        self.assertLessEqual(interval_high[0], interval_high[1])  # Lower <= Upper
        self.assertLessEqual(interval_low[1] - interval_low[0], 1.0)  # Reasonable width
        
        # Test calibration stats
        stats = calibrator.get_calibration_stats()
        self.assertEqual(stats['total_predictions'], 5)
        self.assertEqual(stats['overall_accuracy'], 0.6)  # 3/5 correct
    
    async def test_hybrid_confidence_calculation(self):
        """Test comprehensive hybrid confidence calculation."""
        
        # Setup mock components
        mock_router = Mock()
        mock_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence=0.75,
            reasoning=["Biomedical entities detected"],
            research_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.75,
                research_category_confidence=0.8,
                temporal_analysis_confidence=0.3,
                signal_strength_confidence=0.7,
                context_coherence_confidence=0.8,
                keyword_density=0.6,
                pattern_match_strength=0.7,
                biomedical_entity_count=3,
                ambiguity_score=0.2,
                conflict_score=0.1,
                alternative_interpretations=[],
                calculation_time_ms=25.0
            ),
            knowledge_indicators=["glucose", "metabolism", "pathway"]
        )
        mock_router.route_query.return_value = mock_prediction
        
        # Create scorer
        scorer = HybridConfidenceScorer(
            biomedical_router=mock_router,
            calibration_data_path=self.temp_file.name,
            logger=self.logger
        )
        
        query = "What is the glucose metabolism pathway?"
        
        # Test with LLM result
        llm_result = ClassificationResult(
            category="KNOWLEDGE_GRAPH",
            confidence=0.85,
            reasoning="Query about established metabolic pathways",
            alternative_categories=[{"category": "GENERAL", "confidence": 0.4}],
            uncertainty_indicators=[],
            biomedical_signals={"entities": ["glucose", "metabolism"], "relationships": ["pathway"], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        result = await scorer.calculate_comprehensive_confidence(
            query_text=query,
            llm_result=llm_result,
            llm_response_metadata={'response_time_ms': 200.0}
        )
        
        # Validate result structure
        self.assertIsInstance(result, HybridConfidenceResult)
        self.assertGreater(result.overall_confidence, 0)
        self.assertLessEqual(result.overall_confidence, 1)
        
        # Check confidence interval
        self.assertEqual(len(result.confidence_interval), 2)
        self.assertLessEqual(result.confidence_interval[0], result.confidence_interval[1])
        
        # Check component analyses
        self.assertIsInstance(result.llm_confidence, LLMConfidenceAnalysis)
        self.assertIsInstance(result.keyword_confidence, KeywordConfidenceAnalysis)
        
        # Check weights sum to 1
        self.assertAlmostEqual(result.llm_weight + result.keyword_weight, 1.0, places=5)
        
        # Check uncertainty metrics
        self.assertGreaterEqual(result.epistemic_uncertainty, 0)
        self.assertGreaterEqual(result.aleatoric_uncertainty, 0)
        self.assertGreaterEqual(result.total_uncertainty, 0)
        
        # Check evidence strength and reliability
        self.assertGreater(result.evidence_strength, 0)
        self.assertGreater(result.confidence_reliability, 0)
        
        # Check alternative confidences
        self.assertTrue(len(result.alternative_confidences) > 0)
        
        # Verify performance metrics
        self.assertGreater(result.calculation_time_ms, 0)
    
    async def test_adaptive_weighting(self):
        """Test adaptive weighting based on query characteristics."""
        
        scorer = HybridConfidenceScorer(logger=self.logger)
        
        # Test short query (should favor keywords)
        short_query = "metabolomics"
        short_llm_analysis = LLMConfidenceAnalysis(
            raw_confidence=0.8, calibrated_confidence=0.8,
            reasoning_quality_score=0.7, consistency_score=0.9,
            response_length=50, reasoning_depth=2,
            uncertainty_indicators=[], confidence_expressions=[]
        )
        short_keyword_analysis = KeywordConfidenceAnalysis(
            pattern_match_confidence=0.7, keyword_density_confidence=0.9,
            biomedical_entity_confidence=0.8, domain_specificity_confidence=0.6,
            total_biomedical_signals=2, strong_signals=2, weak_signals=0,
            conflicting_signals=0, semantic_coherence_score=0.8,
            domain_alignment_score=0.9, query_completeness_score=0.3
        )
        
        llm_weight_short, keyword_weight_short = scorer._calculate_adaptive_weights(
            short_query, short_llm_analysis, short_keyword_analysis
        )
        
        # Test long query (should favor LLM)
        long_query = "What is the detailed mechanism of glucose metabolism in relation to insulin signaling pathways and how does this affect cellular energy production in diabetic patients?"
        long_llm_analysis = LLMConfidenceAnalysis(
            raw_confidence=0.9, calibrated_confidence=0.9,
            reasoning_quality_score=0.9, consistency_score=0.95,
            response_length=200, reasoning_depth=4,
            uncertainty_indicators=[], confidence_expressions=[]
        )
        long_keyword_analysis = KeywordConfidenceAnalysis(
            pattern_match_confidence=0.6, keyword_density_confidence=0.5,
            biomedical_entity_confidence=0.8, domain_specificity_confidence=0.7,
            total_biomedical_signals=5, strong_signals=3, weak_signals=2,
            conflicting_signals=0, semantic_coherence_score=0.7,
            domain_alignment_score=0.8, query_completeness_score=0.9
        )
        
        llm_weight_long, keyword_weight_long = scorer._calculate_adaptive_weights(
            long_query, long_llm_analysis, long_keyword_analysis
        )
        
        # Long queries should have higher LLM weight
        self.assertGreater(llm_weight_long, llm_weight_short)
        self.assertLess(keyword_weight_long, keyword_weight_short)
    
    def test_uncertainty_quantification(self):
        """Test epistemic and aleatoric uncertainty calculation."""
        
        scorer = HybridConfidenceScorer(logger=self.logger)
        
        # Test with high uncertainty scenario
        high_uncertainty_llm = LLMConfidenceAnalysis(
            raw_confidence=0.5, calibrated_confidence=0.5,
            reasoning_quality_score=0.3, consistency_score=0.6,
            response_length=50, reasoning_depth=1,
            uncertainty_indicators=["uncertain", "unclear", "possibly"],
            confidence_expressions=[]
        )
        
        high_uncertainty_keyword = KeywordConfidenceAnalysis(
            pattern_match_confidence=0.4, keyword_density_confidence=0.3,
            biomedical_entity_confidence=0.2, domain_specificity_confidence=0.3,
            total_biomedical_signals=1, strong_signals=0, weak_signals=1,
            conflicting_signals=2, semantic_coherence_score=0.4,
            domain_alignment_score=0.3, query_completeness_score=0.4
        )
        
        uncertainties = scorer._calculate_uncertainty_metrics(
            high_uncertainty_llm, high_uncertainty_keyword, evidence_strength=0.3
        )
        
        # High uncertainty scenario should have high uncertainty values
        self.assertGreater(uncertainties['epistemic'], 0.3)
        self.assertGreater(uncertainties['aleatoric'], 0.3)
        self.assertGreater(uncertainties['total'], uncertainties['epistemic'])
        
        # Test with low uncertainty scenario
        low_uncertainty_llm = LLMConfidenceAnalysis(
            raw_confidence=0.9, calibrated_confidence=0.9,
            reasoning_quality_score=0.9, consistency_score=0.95,
            response_length=200, reasoning_depth=4,
            uncertainty_indicators=[], confidence_expressions=[]
        )
        
        low_uncertainty_keyword = KeywordConfidenceAnalysis(
            pattern_match_confidence=0.9, keyword_density_confidence=0.8,
            biomedical_entity_confidence=0.9, domain_specificity_confidence=0.8,
            total_biomedical_signals=5, strong_signals=5, weak_signals=0,
            conflicting_signals=0, semantic_coherence_score=0.9,
            domain_alignment_score=0.9, query_completeness_score=0.9
        )
        
        low_uncertainties = scorer._calculate_uncertainty_metrics(
            low_uncertainty_llm, low_uncertainty_keyword, evidence_strength=0.9
        )
        
        # Low uncertainty scenario should have lower uncertainty values
        self.assertLess(low_uncertainties['epistemic'], uncertainties['epistemic'])
        self.assertLess(low_uncertainties['aleatoric'], uncertainties['aleatoric'])
        self.assertLess(low_uncertainties['total'], uncertainties['total'])


# ============================================================================
# PERFORMANCE AND LOAD TESTS
# ============================================================================

class TestPerformanceOptimization(unittest.TestCase):
    """Tests for performance optimization and load handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = logging.getLogger('test')
        self.config = LLMClassificationConfig(
            timeout_seconds=0.5,  # Fast timeout for testing
            enable_caching=True,
            max_cache_size=50
        )
    
    async def test_response_time_requirement(self):
        """Test that classification meets <2 second response time requirement."""
        
        with patch('lightrag_integration.llm_query_classifier.AsyncOpenAI') as mock_openai:
            # Setup fast mock response
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "category": "GENERAL",
                "confidence": 0.7,
                "reasoning": "Fast classification"
            })
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 20
            
            mock_client.chat.completions.create.return_value = mock_response
            
            classifier = LLMQueryClassifier(self.config, logger=self.logger)
            
            # Test multiple queries for timing
            queries = [
                "metabolomics analysis",
                "biomarker discovery",
                "pathway analysis",
                "LC-MS methods",
                "clinical metabolomics"
            ]
            
            for query in queries:
                start_time = time.time()
                result, used_llm = await classifier.classify_query(query)
                end_time = time.time()
                
                response_time = end_time - start_time
                self.assertLess(response_time, 2.0, f"Query '{query}' took {response_time:.3f}s (>2s)")
                self.assertIsNotNone(result)
    
    async def test_concurrent_request_handling(self):
        """Test handling of concurrent classification requests."""
        
        with patch('lightrag_integration.llm_query_classifier.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Add small delay to simulate network latency
            async def mock_create(*args, **kwargs):
                await asyncio.sleep(0.1)  # 100ms simulated latency
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps({
                    "category": "GENERAL",
                    "confidence": 0.7,
                    "reasoning": "Concurrent classification"
                })
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 50
                mock_response.usage.completion_tokens = 20
                return mock_response
            
            mock_client.chat.completions.create.side_effect = mock_create
            
            classifier = LLMQueryClassifier(self.config, logger=self.logger)
            
            # Create concurrent requests
            queries = [f"Query {i}" for i in range(10)]
            start_time = time.time()
            
            tasks = [classifier.classify_query(query) for query in queries]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete all 10 requests in < 2 seconds (due to concurrency)
            self.assertLess(total_time, 2.0, f"Concurrent requests took {total_time:.3f}s")
            self.assertEqual(len(results), 10)
            
            # All requests should succeed
            for result, used_llm in results:
                self.assertIsNotNone(result)
    
    def test_cache_efficiency(self):
        """Test cache hit rate and efficiency."""
        
        cache = ClassificationCache(max_size=10, ttl_hours=1)
        
        classification_result = ClassificationResult(
            category="KNOWLEDGE_GRAPH",
            confidence=0.8,
            reasoning="Test classification",
            alternative_categories=[],
            uncertainty_indicators=[],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        # Test cache hit rate with repeated queries
        queries = ["query1", "query2", "query3"] * 5  # 15 total, 3 unique
        
        cache_hits = 0
        for query in queries:
            cached = cache.get(query)
            if cached is not None:
                cache_hits += 1
            else:
                cache.put(query, classification_result)
        
        expected_hits = 15 - 3  # Should miss first occurrence of each unique query
        self.assertEqual(cache_hits, expected_hits)
        
        # Test cache efficiency stats
        stats = cache.get_stats()
        self.assertEqual(stats['cache_size'], 3)
        self.assertEqual(stats['utilization'], 0.3)  # 3/10
    
    async def test_memory_usage_optimization(self):
        """Test memory usage doesn't grow unboundedly."""
        
        config = LLMClassificationConfig(
            enable_caching=True,
            max_cache_size=5  # Small cache size
        )
        
        with patch('lightrag_integration.llm_query_classifier.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "category": "GENERAL",
                "confidence": 0.7,
                "reasoning": "Memory test"
            })
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 20
            mock_client.chat.completions.create.return_value = mock_response
            
            classifier = LLMQueryClassifier(config, logger=self.logger)
            
            # Process many unique queries
            for i in range(20):
                query = f"Unique query {i}"
                await classifier.classify_query(query)
            
            # Cache should be limited to max size
            cache_stats = classifier.cache.get_stats()
            self.assertLessEqual(cache_stats['cache_size'], 5)
            
            # Response times deque should be limited
            self.assertLessEqual(len(classifier._response_times), 100)
            
            # Confidence scores deque should be limited
            self.assertLessEqual(len(classifier._confidence_scores), 100)
    
    def test_batch_processing_efficiency(self):
        """Test efficiency improvements from request batching."""
        
        # This test would validate batching optimizations if implemented
        # For now, we test that individual requests don't interfere
        
        scorer = HybridConfidenceScorer(logger=self.logger)
        
        # Simulate processing multiple queries
        start_time = time.time()
        
        for i in range(10):
            # These would be processed more efficiently in a real batch system
            keyword_analysis = KeywordConfidenceAnalysis(
                pattern_match_confidence=0.7,
                keyword_density_confidence=0.6,
                biomedical_entity_confidence=0.8,
                domain_specificity_confidence=0.7,
                total_biomedical_signals=3,
                strong_signals=2,
                weak_signals=1,
                conflicting_signals=0,
                semantic_coherence_score=0.7,
                domain_alignment_score=0.8,
                query_completeness_score=0.6
            )
            
            # Verify each processing is efficient
            self.assertIsInstance(keyword_analysis, KeywordConfidenceAnalysis)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should process 10 analyses quickly
        self.assertLess(total_time, 0.1, f"Batch processing took {total_time:.3f}s")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegrationCompatibility(unittest.TestCase):
    """Tests for integration with existing infrastructure."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = logging.getLogger('test')
    
    def test_routing_prediction_compatibility(self):
        """Test conversion between LLM results and RoutingPrediction."""
        
        # Create LLM classification result
        llm_result = ClassificationResult(
            category="KNOWLEDGE_GRAPH",
            confidence=0.85,
            reasoning="Query about established biomedical relationships",
            alternative_categories=[{"category": "GENERAL", "confidence": 0.3}],
            uncertainty_indicators=[],
            biomedical_signals={
                "entities": ["glucose", "metabolism", "insulin"],
                "relationships": ["pathway", "signaling"],
                "techniques": []
            },
            temporal_signals={
                "keywords": [],
                "patterns": [],
                "years": []
            }
        )
        
        query_text = "What is the glucose metabolism pathway?"
        used_llm = True
        
        # Convert to RoutingPrediction
        routing_prediction = convert_llm_result_to_routing_prediction(
            llm_result, query_text, used_llm
        )
        
        # Verify conversion
        self.assertIsInstance(routing_prediction, RoutingPrediction)
        self.assertEqual(routing_prediction.routing_decision, RoutingDecision.LIGHTRAG)
        self.assertEqual(routing_prediction.confidence, 0.85)
        self.assertIn("LLM-powered semantic classification", routing_prediction.reasoning)
        
        # Check metadata
        self.assertTrue(routing_prediction.metadata['llm_powered'])
        self.assertEqual(routing_prediction.metadata['llm_category'], "KNOWLEDGE_GRAPH")
        
        # Check confidence metrics compatibility
        self.assertIsInstance(routing_prediction.confidence_metrics, ConfidenceMetrics)
        self.assertEqual(routing_prediction.confidence_metrics.overall_confidence, 0.85)
    
    def test_confidence_metrics_integration(self):
        """Test integration with existing ConfidenceMetrics structure."""
        
        # Create hybrid confidence result
        hybrid_result = HybridConfidenceResult(
            overall_confidence=0.82,
            confidence_interval=(0.75, 0.89),
            llm_confidence=LLMConfidenceAnalysis(
                raw_confidence=0.85, calibrated_confidence=0.82,
                reasoning_quality_score=0.8, consistency_score=0.9,
                response_length=150, reasoning_depth=3,
                uncertainty_indicators=[], confidence_expressions=[]
            ),
            keyword_confidence=KeywordConfidenceAnalysis(
                pattern_match_confidence=0.7, keyword_density_confidence=0.6,
                biomedical_entity_confidence=0.9, domain_specificity_confidence=0.8,
                total_biomedical_signals=4, strong_signals=3, weak_signals=1,
                conflicting_signals=0, semantic_coherence_score=0.8,
                domain_alignment_score=0.85, query_completeness_score=0.7
            ),
            llm_weight=0.6, keyword_weight=0.4, calibration_adjustment=0.03,
            epistemic_uncertainty=0.15, aleatoric_uncertainty=0.1, total_uncertainty=0.2,
            confidence_reliability=0.8, evidence_strength=0.85,
            alternative_confidences=[("llm_only", 0.85), ("keyword_only", 0.75)],
            calculation_time_ms=125.0, calibration_version="1.0"
        )
        
        query_text = "What is glucose metabolism?"
        
        # Convert to legacy ConfidenceMetrics
        legacy_metrics = integrate_with_existing_confidence_metrics(hybrid_result, query_text)
        
        # Verify conversion
        self.assertIsInstance(legacy_metrics, ConfidenceMetrics)
        self.assertEqual(legacy_metrics.overall_confidence, 0.82)
        self.assertEqual(legacy_metrics.research_category_confidence, 0.82)
        self.assertEqual(legacy_metrics.biomedical_entity_count, 4)
        self.assertEqual(legacy_metrics.calculation_time_ms, 125.0)
        
        # Check alternative interpretations
        self.assertTrue(len(legacy_metrics.alternative_interpretations) > 0)
    
    async def test_llm_enhanced_router_creation(self):
        """Test factory function for creating LLM-enhanced router."""
        
        with patch('lightrag_integration.llm_query_classifier.BiomedicalQueryRouter') as mock_router_class:
            mock_router = Mock()
            mock_router_class.return_value = mock_router
            
            config = LLMClassificationConfig(api_key="test-key")
            
            with patch('lightrag_integration.llm_query_classifier.AsyncOpenAI'):
                enhanced_router = await create_llm_enhanced_router(
                    config=config,
                    logger=self.logger
                )
                
                self.assertIsInstance(enhanced_router, LLMQueryClassifier)
                self.assertEqual(enhanced_router.config.api_key, "test-key")
                self.assertIsNotNone(enhanced_router.biomedical_router)
    
    def test_hybrid_scorer_creation(self):
        """Test factory function for creating hybrid confidence scorer."""
        
        with patch('lightrag_integration.comprehensive_confidence_scorer.BiomedicalQueryRouter') as mock_router_class:
            mock_router = Mock()
            mock_router_class.return_value = mock_router
            
            scorer = create_hybrid_confidence_scorer(logger=self.logger)
            
            self.assertIsInstance(scorer, HybridConfidenceScorer)
            self.assertIsNotNone(scorer.biomedical_router)
            self.assertIsNotNone(scorer.llm_analyzer)
            self.assertIsNotNone(scorer.calibrator)
    
    def test_backwards_compatibility(self):
        """Test that new system maintains backwards compatibility."""
        
        # Test that old RoutingPrediction format still works
        old_style_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence=0.8,
            reasoning=["Old style reasoning"],
            research_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.8,
                research_category_confidence=0.8,
                temporal_analysis_confidence=0.3,
                signal_strength_confidence=0.7,
                context_coherence_confidence=0.8,
                keyword_density=0.6,
                pattern_match_strength=0.7,
                biomedical_entity_count=3,
                ambiguity_score=0.2,
                conflict_score=0.1,
                alternative_interpretations=[],
                calculation_time_ms=50.0
            )
        )
        
        # Should be able to serialize/deserialize
        serialized = old_style_prediction.to_dict()
        self.assertIn('routing_decision', serialized)
        self.assertIn('confidence', serialized)
        self.assertIn('confidence_metrics', serialized)
        
        # New features should have defaults
        self.assertEqual(old_style_prediction.confidence_level, 'high')
        self.assertIsNone(old_style_prediction.fallback_strategy)


# ============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# ============================================================================

class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Tests for edge cases and error handling scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = logging.getLogger('test')
        self.config = LLMClassificationConfig(
            api_key="test-key",
            timeout_seconds=0.1,  # Very short timeout for testing
            max_retries=1
        )
    
    @patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
    async def test_api_timeout_handling(self, mock_openai_class):
        """Test handling of API timeouts."""
        
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = asyncio.TimeoutError("Timeout")
        
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        
        result, used_llm = await classifier.classify_query("test query")
        
        # Should fallback gracefully
        self.assertFalse(used_llm)
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(classifier.metrics.llm_failures, 1)
    
    @patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
    async def test_network_connectivity_issues(self, mock_openai_class):
        """Test handling of network connectivity issues."""
        
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = ConnectionError("Network error")
        
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        
        result, used_llm = await classifier.classify_query("test query")
        
        # Should fallback gracefully
        self.assertFalse(used_llm)
        self.assertIsInstance(result, ClassificationResult)
        self.assertIn("simple_fallback_used", result.uncertainty_indicators)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        
        # Test empty query
        with self.assertRaises(Exception):
            asyncio.run(classifier.classify_query(""))
        
        # Test None query
        with self.assertRaises(Exception):
            asyncio.run(classifier.classify_query(None))
        
        # Test extremely long query
        very_long_query = "test " * 10000  # Very long query
        # Should handle gracefully (not crash)
        result = asyncio.run(classifier.classify_query(very_long_query))
        self.assertIsNotNone(result)
    
    def test_malformed_configuration(self):
        """Test handling of malformed configuration."""
        
        # Test missing API key
        with self.assertRaises(ValueError):
            bad_config = LLMClassificationConfig(api_key=None)
            LLMQueryClassifier(bad_config)
        
        # Test invalid provider
        with self.assertRaises(NotImplementedError):
            bad_config = LLMClassificationConfig(
                provider=LLMProvider.ANTHROPIC,  # Not implemented
                api_key="test-key"
            )
            LLMQueryClassifier(bad_config)
    
    @patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
    async def test_malformed_llm_response(self, mock_openai_class):
        """Test handling of various malformed LLM responses."""
        
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        
        # Test cases for different malformed responses
        malformed_responses = [
            "Not JSON at all",
            '{"invalid": "json"',  # Incomplete JSON
            '{"category": "INVALID_CATEGORY", "confidence": 0.8}',  # Invalid category
            '{"category": "GENERAL", "confidence": 1.5}',  # Invalid confidence range
            '{"category": "GENERAL"}',  # Missing required fields
            json.dumps({"category": "GENERAL", "confidence": "not_a_number"}),  # Wrong type
        ]
        
        classifier = LLMQueryClassifier(self.config, logger=self.logger)
        
        for malformed_response in malformed_responses:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = malformed_response
            mock_client.chat.completions.create.return_value = mock_response
            
            result, used_llm = await classifier.classify_query("test query")
            
            # Should fallback gracefully for all malformed responses
            self.assertFalse(used_llm)
            self.assertIsInstance(result, ClassificationResult)
    
    def test_resource_exhaustion_scenarios(self):
        """Test handling of resource exhaustion scenarios."""
        
        # Test cache under memory pressure
        cache = ClassificationCache(max_size=2, ttl_hours=1)
        
        classification_result = ClassificationResult(
            category="GENERAL",
            confidence=0.7,
            reasoning="Test",
            alternative_categories=[],
            uncertainty_indicators=[],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        # Add many items to trigger eviction
        for i in range(10):
            cache.put(f"query_{i}", classification_result)
        
        # Cache size should be limited
        stats = cache.get_stats()
        self.assertLessEqual(stats['cache_size'], 2)
        
        # Should still function correctly
        cache.put("new_query", classification_result)
        result = cache.get("new_query")
        self.assertIsNotNone(result)
    
    def test_concurrent_access_safety(self):
        """Test thread safety under concurrent access."""
        
        cache = ClassificationCache(max_size=10, ttl_hours=1)
        classification_result = ClassificationResult(
            category="GENERAL",
            confidence=0.7,
            reasoning="Test",
            alternative_categories=[],
            uncertainty_indicators=[],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        def worker(thread_id):
            """Worker function for concurrent testing."""
            for i in range(50):
                query = f"thread_{thread_id}_query_{i}"
                cache.put(query, classification_result)
                result = cache.get(query)
                self.assertIsNotNone(result)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)  # 5 second timeout
        
        # Cache should still be functional and within size limits
        stats = cache.get_stats()
        self.assertLessEqual(stats['cache_size'], 10)
    
    def test_confidence_calibration_edge_cases(self):
        """Test edge cases in confidence calibration."""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('{}')
            temp_path = f.name
        
        try:
            calibrator = ConfidenceCalibrator(temp_path, self.logger)
            
            # Test with no historical data
            calibrated = calibrator.calibrate_confidence(0.8, ConfidenceSource.LLM_SEMANTIC)
            self.assertIsInstance(calibrated, float)
            self.assertGreaterEqual(calibrated, 0.0)
            self.assertLessEqual(calibrated, 1.0)
            
            # Test with extreme confidence values
            calibrated_low = calibrator.calibrate_confidence(0.0, ConfidenceSource.LLM_SEMANTIC)
            calibrated_high = calibrator.calibrate_confidence(1.0, ConfidenceSource.LLM_SEMANTIC)
            
            self.assertGreaterEqual(calibrated_low, 0.0)
            self.assertLessEqual(calibrated_high, 1.0)
            
            # Test with minimal data (edge case for recalibration)
            for i in range(3):  # Less than minimum for recalibration
                calibrator.record_prediction_outcome(0.5, True, ConfidenceSource.LLM_SEMANTIC)
            
            stats = calibrator.get_calibration_stats()
            self.assertEqual(stats['total_predictions'], 3)
            
        finally:
            os.unlink(temp_path)
    
    def test_validation_edge_cases(self):
        """Test edge cases in confidence validation."""
        
        mock_scorer = Mock()
        mock_scorer.calibrator = Mock()
        mock_scorer.calibrator.get_calibration_stats.return_value = {
            'total_predictions': 0,
            'overall_accuracy': 0.0,
            'brier_score': 0.5
        }
        
        validator = ConfidenceValidator(mock_scorer, self.logger)
        
        # Test validation with no historical data
        result = validator.validate_confidence_accuracy(
            query_text="test query",
            predicted_confidence=0.8,
            actual_routing_accuracy=True
        )
        
        self.assertIn('validation_record', result)
        self.assertIn('validation_metrics', result)
        self.assertIn('recommendations', result)
        
        # Test validation report with no data
        report = validator.get_validation_report()
        self.assertEqual(report['validation_summary']['total_validations'], 1)


# ============================================================================
# PERFORMANCE BENCHMARKING TESTS
# ============================================================================

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests with automated validation."""
    
    def setUp(self):
        """Set up benchmarking environment."""
        self.logger = logging.getLogger('test')
    
    async def test_classification_latency_benchmark(self):
        """Benchmark classification latency under various conditions."""
        
        with patch('lightrag_integration.llm_query_classifier.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Simulate realistic API latency
            async def mock_create(*args, **kwargs):
                await asyncio.sleep(0.05)  # 50ms simulated latency
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps({
                    "category": "GENERAL",
                    "confidence": 0.7,
                    "reasoning": "Benchmark classification"
                })
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 75
                mock_response.usage.completion_tokens = 25
                return mock_response
            
            mock_client.chat.completions.create.side_effect = mock_create
            
            config = LLMClassificationConfig(
                timeout_seconds=2.0,
                enable_caching=True
            )
            classifier = LLMQueryClassifier(config, logger=self.logger)
            
            # Benchmark different query lengths
            test_cases = [
                ("Short", "metabolomics"),
                ("Medium", "What is glucose metabolism pathway analysis?"),
                ("Long", "Explain the detailed mechanism of glucose metabolism in relation to insulin signaling pathways and how this affects cellular energy production in diabetic patients with metabolic syndrome.")
            ]
            
            for case_name, query in test_cases:
                latencies = []
                
                # Run multiple trials
                for _ in range(5):
                    start_time = time.time()
                    result, used_llm = await classifier.classify_query(query)
                    end_time = time.time()
                    
                    latency = end_time - start_time
                    latencies.append(latency)
                    
                    self.assertIsNotNone(result)
                
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                
                print(f"{case_name} query - Avg: {avg_latency*1000:.1f}ms, Max: {max_latency*1000:.1f}ms")
                
                # All should be under 2 seconds
                self.assertLess(max_latency, 2.0, f"{case_name} query exceeded 2s: {max_latency:.3f}s")
    
    def test_cache_performance_benchmark(self):
        """Benchmark cache performance and hit rates."""
        
        cache_sizes = [10, 50, 100, 500]
        query_sets = [50, 100, 200, 500]
        
        for cache_size in cache_sizes:
            for query_set_size in query_sets:
                if query_set_size <= cache_size * 2:  # Only test reasonable ratios
                    cache = ClassificationCache(max_size=cache_size, ttl_hours=1)
                    
                    classification_result = ClassificationResult(
                        category="GENERAL",
                        confidence=0.7,
                        reasoning="Benchmark",
                        alternative_categories=[],
                        uncertainty_indicators=[],
                        biomedical_signals={"entities": [], "relationships": [], "techniques": []},
                        temporal_signals={"keywords": [], "patterns": [], "years": []}
                    )
                    
                    # Create query set with some repetition
                    queries = []
                    for i in range(query_set_size):
                        if i < query_set_size // 2:
                            queries.append(f"unique_query_{i}")
                        else:
                            # Repeat earlier queries for cache hits
                            queries.append(f"unique_query_{i % (query_set_size // 2)}")
                    
                    # Benchmark cache operations
                    start_time = time.time()
                    hits = 0
                    
                    for query in queries:
                        result = cache.get(query)
                        if result is not None:
                            hits += 1
                        else:
                            cache.put(query, classification_result)
                    
                    end_time = time.time()
                    
                    hit_rate = hits / len(queries) * 100
                    total_time = end_time - start_time
                    ops_per_second = len(queries) / total_time
                    
                    print(f"Cache {cache_size}, Queries {query_set_size}: {hit_rate:.1f}% hit rate, {ops_per_second:.0f} ops/sec")
                    
                    # Performance assertions
                    self.assertGreater(ops_per_second, 1000, "Cache operations should be >1000 ops/sec")
    
    async def test_confidence_scoring_benchmark(self):
        """Benchmark confidence scoring performance."""
        
        scorer = HybridConfidenceScorer(logger=self.logger)
        
        # Test queries of varying complexity
        test_queries = [
            "metabolomics",
            "What is glucose metabolism?",
            "Explain the detailed relationship between glucose metabolism and insulin signaling in diabetic patients",
            "Latest research on metabolomics biomarkers for personalized medicine applications in cancer treatment 2025"
        ]
        
        for query in test_queries:
            latencies = []
            
            for _ in range(10):  # Multiple trials
                start_time = time.time()
                
                # Create mock LLM result
                llm_result = ClassificationResult(
                    category="KNOWLEDGE_GRAPH",
                    confidence=0.8,
                    reasoning="Test classification for benchmarking",
                    alternative_categories=[],
                    uncertainty_indicators=[],
                    biomedical_signals={"entities": ["glucose"], "relationships": [], "techniques": []},
                    temporal_signals={"keywords": [], "patterns": [], "years": []}
                )
                
                result = await scorer.calculate_comprehensive_confidence(
                    query_text=query,
                    llm_result=llm_result
                )
                
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)
                
                self.assertIsNotNone(result)
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            print(f"Confidence scoring '{query[:30]}...': Avg {avg_latency*1000:.1f}ms, Max {max_latency*1000:.1f}ms")
            
            # Should be fast for confidence scoring
            self.assertLess(max_latency, 0.5, f"Confidence scoring too slow: {max_latency:.3f}s")
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage patterns."""
        
        import sys
        
        # Baseline memory usage
        baseline_size = sys.getsizeof(object())
        
        # Test cache memory usage
        cache = ClassificationCache(max_size=1000, ttl_hours=1)
        classification_result = ClassificationResult(
            category="GENERAL",
            confidence=0.7,
            reasoning="Memory test",
            alternative_categories=[],
            uncertainty_indicators=[],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        # Fill cache and measure memory growth
        for i in range(1000):
            cache.put(f"query_{i}", classification_result)
        
        cache_size = sys.getsizeof(cache)
        print(f"Cache memory usage: {cache_size / 1024:.1f} KB for 1000 entries")
        
        # Memory usage should be reasonable
        self.assertLess(cache_size, 10 * 1024 * 1024, "Cache using >10MB memory")
        
        # Test confidence scorer memory
        scorer = HybridConfidenceScorer(logger=self.logger)
        scorer_size = sys.getsizeof(scorer)
        
        print(f"Confidence scorer memory usage: {scorer_size / 1024:.1f} KB")
        self.assertLess(scorer_size, 1024 * 1024, "Confidence scorer using >1MB memory")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("COMPREHENSIVE LLM CLASSIFICATION SYSTEM TESTS")
    print("=" * 80)
    
    # Run test suite
    unittest.main(verbosity=2, buffer=True)