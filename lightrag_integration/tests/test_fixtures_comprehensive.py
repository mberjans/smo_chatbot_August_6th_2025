"""
Comprehensive Test Fixtures and Mock Data for LLM Classification System

This module provides realistic test fixtures, mock data, and utilities for testing
the LLM-based classification system with comprehensive coverage of biomedical queries,
edge cases, and performance scenarios.

Features:
    - Realistic biomedical query datasets
    - Mock API responses for different scenarios
    - Performance test data generators
    - Edge case scenario builders
    - Confidence calibration test data
    - Integration test helpers

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import json
import random
import time
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import pytest
import logging

# Import the types we need to mock
try:
    from ..llm_classification_prompts import ClassificationResult
    from ..query_router import RoutingPrediction, RoutingDecision, ConfidenceMetrics
    from ..research_categorizer import CategoryPrediction
    from ..cost_persistence import ResearchCategory
    from ..comprehensive_confidence_scorer import (
        LLMConfidenceAnalysis, KeywordConfidenceAnalysis, HybridConfidenceResult,
        ConfidenceSource, ConfidenceCalibrationData
    )
except ImportError:
    # Fallback for testing environments where imports might not work
    pass


# ============================================================================
# REALISTIC BIOMEDICAL QUERY DATASETS
# ============================================================================

@dataclass
class QueryTestCase:
    """Test case for a biomedical query with expected outcomes."""
    
    query_text: str
    expected_category: str
    expected_confidence_range: Tuple[float, float]
    expected_routing: str
    biomedical_entities: List[str]
    temporal_indicators: List[str]
    complexity_level: str  # 'simple', 'medium', 'complex'
    domain_specificity: str  # 'general', 'specific', 'highly_specific'
    uncertainty_level: str  # 'low', 'medium', 'high'
    description: str


class BiomedicalQueryDatasets:
    """Comprehensive datasets of biomedical queries for testing."""
    
    @staticmethod
    def get_knowledge_graph_queries() -> List[QueryTestCase]:
        """Queries that should route to knowledge graph (LightRAG)."""
        return [
            QueryTestCase(
                query_text="What is the relationship between glucose metabolism and insulin signaling pathways?",
                expected_category="KNOWLEDGE_GRAPH",
                expected_confidence_range=(0.8, 0.95),
                expected_routing="lightrag",
                biomedical_entities=["glucose", "metabolism", "insulin", "signaling", "pathway"],
                temporal_indicators=[],
                complexity_level="medium",
                domain_specificity="specific",
                uncertainty_level="low",
                description="Classic pathway relationship query"
            ),
            QueryTestCase(
                query_text="How does mitochondrial dysfunction affect cellular energy production in metabolic disorders?",
                expected_category="KNOWLEDGE_GRAPH",
                expected_confidence_range=(0.85, 0.95),
                expected_routing="lightrag",
                biomedical_entities=["mitochondrial", "dysfunction", "cellular", "energy", "metabolic", "disorders"],
                temporal_indicators=[],
                complexity_level="complex",
                domain_specificity="highly_specific",
                uncertainty_level="low",
                description="Complex cellular mechanism query"
            ),
            QueryTestCase(
                query_text="Explain the citric acid cycle and its connection to amino acid biosynthesis",
                expected_category="KNOWLEDGE_GRAPH",
                expected_confidence_range=(0.9, 0.98),
                expected_routing="lightrag",
                biomedical_entities=["citric acid cycle", "amino acid", "biosynthesis"],
                temporal_indicators=[],
                complexity_level="medium",
                domain_specificity="highly_specific",
                uncertainty_level="low",
                description="Fundamental biochemistry query"
            ),
            QueryTestCase(
                query_text="What are the key metabolic pathways involved in cancer cell metabolism?",
                expected_category="KNOWLEDGE_GRAPH",
                expected_confidence_range=(0.8, 0.92),
                expected_routing="lightrag",
                biomedical_entities=["metabolic", "pathways", "cancer", "cell", "metabolism"],
                temporal_indicators=[],
                complexity_level="complex",
                domain_specificity="specific",
                uncertainty_level="low",
                description="Cancer metabolism pathways"
            ),
            QueryTestCase(
                query_text="Describe the mechanism of action for metformin in diabetes treatment",
                expected_category="KNOWLEDGE_GRAPH",
                expected_confidence_range=(0.85, 0.95),
                expected_routing="lightrag",
                biomedical_entities=["mechanism", "metformin", "diabetes", "treatment"],
                temporal_indicators=[],
                complexity_level="medium",
                domain_specificity="highly_specific",
                uncertainty_level="low",
                description="Drug mechanism query"
            )
        ]
    
    @staticmethod
    def get_real_time_queries() -> List[QueryTestCase]:
        """Queries that should route to real-time search (Perplexity)."""
        return [
            QueryTestCase(
                query_text="Latest research on metabolomics biomarkers for COVID-19 in 2025",
                expected_category="REAL_TIME",
                expected_confidence_range=(0.85, 0.95),
                expected_routing="perplexity",
                biomedical_entities=["metabolomics", "biomarkers", "COVID-19"],
                temporal_indicators=["latest", "2025"],
                complexity_level="medium",
                domain_specificity="specific",
                uncertainty_level="low",
                description="Current research query with specific year"
            ),
            QueryTestCase(
                query_text="Recent breakthrough in Alzheimer's disease biomarker discovery",
                expected_category="REAL_TIME",
                expected_confidence_range=(0.8, 0.9),
                expected_routing="perplexity",
                biomedical_entities=["Alzheimer's disease", "biomarker", "discovery"],
                temporal_indicators=["recent", "breakthrough"],
                complexity_level="medium",
                domain_specificity="specific",
                uncertainty_level="medium",
                description="Recent breakthrough query"
            ),
            QueryTestCase(
                query_text="Current trends in personalized medicine for cancer treatment 2024",
                expected_category="REAL_TIME",
                expected_confidence_range=(0.8, 0.92),
                expected_routing="perplexity",
                biomedical_entities=["personalized medicine", "cancer", "treatment"],
                temporal_indicators=["current", "trends", "2024"],
                complexity_level="complex",
                domain_specificity="specific",
                uncertainty_level="medium",
                description="Current trends query"
            ),
            QueryTestCase(
                query_text="New FDA approved metabolomics-based diagnostic tools this year",
                expected_category="REAL_TIME",
                expected_confidence_range=(0.85, 0.95),
                expected_routing="perplexity",
                biomedical_entities=["FDA", "metabolomics", "diagnostic", "tools"],
                temporal_indicators=["new", "this year"],
                complexity_level="medium",
                domain_specificity="highly_specific",
                uncertainty_level="low",
                description="Regulatory approval query"
            ),
            QueryTestCase(
                query_text="Breaking news on CRISPR gene editing applications in metabolic disorders",
                expected_category="REAL_TIME",
                expected_confidence_range=(0.9, 0.98),
                expected_routing="perplexity",
                biomedical_entities=["CRISPR", "gene editing", "metabolic disorders"],
                temporal_indicators=["breaking news"],
                complexity_level="complex",
                domain_specificity="highly_specific",
                uncertainty_level="low",
                description="Breaking news query"
            )
        ]
    
    @staticmethod
    def get_general_queries() -> List[QueryTestCase]:
        """General queries that could go to either service."""
        return [
            QueryTestCase(
                query_text="metabolomics analysis methods",
                expected_category="GENERAL",
                expected_confidence_range=(0.6, 0.8),
                expected_routing="either",
                biomedical_entities=["metabolomics", "analysis", "methods"],
                temporal_indicators=[],
                complexity_level="simple",
                domain_specificity="general",
                uncertainty_level="medium",
                description="General methods query"
            ),
            QueryTestCase(
                query_text="LC-MS techniques for metabolite identification",
                expected_category="GENERAL",
                expected_confidence_range=(0.7, 0.85),
                expected_routing="either",
                biomedical_entities=["LC-MS", "metabolite", "identification"],
                temporal_indicators=[],
                complexity_level="medium",
                domain_specificity="specific",
                uncertainty_level="low",
                description="Analytical technique query"
            ),
            QueryTestCase(
                query_text="biomarker discovery approaches in clinical studies",
                expected_category="GENERAL",
                expected_confidence_range=(0.65, 0.8),
                expected_routing="either",
                biomedical_entities=["biomarker", "discovery", "clinical", "studies"],
                temporal_indicators=[],
                complexity_level="medium",
                domain_specificity="general",
                uncertainty_level="medium",
                description="Discovery approaches query"
            ),
            QueryTestCase(
                query_text="clinical metabolomics applications overview",
                expected_category="GENERAL",
                expected_confidence_range=(0.6, 0.75),
                expected_routing="either",
                biomedical_entities=["clinical", "metabolomics", "applications"],
                temporal_indicators=[],
                complexity_level="simple",
                domain_specificity="general",
                uncertainty_level="medium",
                description="Overview query"
            )
        ]
    
    @staticmethod
    def get_ambiguous_queries() -> List[QueryTestCase]:
        """Ambiguous queries with high uncertainty."""
        return [
            QueryTestCase(
                query_text="metabolomics",
                expected_category="GENERAL",
                expected_confidence_range=(0.3, 0.6),
                expected_routing="either",
                biomedical_entities=["metabolomics"],
                temporal_indicators=[],
                complexity_level="simple",
                domain_specificity="general",
                uncertainty_level="high",
                description="Single term query"
            ),
            QueryTestCase(
                query_text="recent studies",
                expected_category="REAL_TIME",
                expected_confidence_range=(0.4, 0.7),
                expected_routing="perplexity",
                biomedical_entities=[],
                temporal_indicators=["recent"],
                complexity_level="simple",
                domain_specificity="general",
                uncertainty_level="high",
                description="Vague temporal query"
            ),
            QueryTestCase(
                query_text="biomarkers research applications",
                expected_category="GENERAL",
                expected_confidence_range=(0.4, 0.7),
                expected_routing="either",
                biomedical_entities=["biomarkers", "research"],
                temporal_indicators=[],
                complexity_level="simple",
                domain_specificity="general",
                uncertainty_level="high",
                description="Generic research query"
            ),
            QueryTestCase(
                query_text="analysis methods current approaches",
                expected_category="GENERAL",
                expected_confidence_range=(0.35, 0.65),
                expected_routing="either",
                biomedical_entities=["analysis", "methods"],
                temporal_indicators=["current"],
                complexity_level="simple",
                domain_specificity="general",
                uncertainty_level="high",
                description="Mixed signals query"
            )
        ]
    
    @staticmethod
    def get_edge_case_queries() -> List[QueryTestCase]:
        """Edge case queries for robustness testing."""
        return [
            QueryTestCase(
                query_text="",
                expected_category="GENERAL",
                expected_confidence_range=(0.0, 0.3),
                expected_routing="either",
                biomedical_entities=[],
                temporal_indicators=[],
                complexity_level="simple",
                domain_specificity="general",
                uncertainty_level="high",
                description="Empty query"
            ),
            QueryTestCase(
                query_text="a",
                expected_category="GENERAL",
                expected_confidence_range=(0.1, 0.4),
                expected_routing="either",
                biomedical_entities=[],
                temporal_indicators=[],
                complexity_level="simple",
                domain_specificity="general",
                uncertainty_level="high",
                description="Single character query"
            ),
            QueryTestCase(
                query_text="What is the extremely complex and detailed molecular mechanism underlying the intricate relationships between glucose metabolism, insulin signaling pathways, mitochondrial biogenesis, cellular energy homeostasis, and the pathophysiological processes that lead to the development of type 2 diabetes mellitus in genetically predisposed individuals with metabolic syndrome?" * 5,
                expected_category="KNOWLEDGE_GRAPH",
                expected_confidence_range=(0.6, 0.8),
                expected_routing="lightrag",
                biomedical_entities=["glucose", "metabolism", "insulin", "signaling", "mitochondrial", "diabetes"],
                temporal_indicators=[],
                complexity_level="complex",
                domain_specificity="highly_specific",
                uncertainty_level="medium",
                description="Extremely long query"
            ),
            QueryTestCase(
                query_text="1234567890 !@#$%^&*() metabolomics analysis 2025 recent breakthrough",
                expected_category="REAL_TIME",
                expected_confidence_range=(0.4, 0.7),
                expected_routing="perplexity",
                biomedical_entities=["metabolomics", "analysis"],
                temporal_indicators=["2025", "recent", "breakthrough"],
                complexity_level="simple",
                domain_specificity="specific",
                uncertainty_level="high",
                description="Query with special characters and numbers"
            ),
            QueryTestCase(
                query_text="Latest breakthrough in 2025 metabolomics but also established pathways and mechanisms from traditional biochemistry",
                expected_category="GENERAL",  # Mixed signals
                expected_confidence_range=(0.4, 0.7),
                expected_routing="either",
                biomedical_entities=["metabolomics", "pathways", "mechanisms", "biochemistry"],
                temporal_indicators=["latest", "breakthrough", "2025"],
                complexity_level="complex",
                domain_specificity="specific",
                uncertainty_level="high",
                description="Conflicting temporal and knowledge signals"
            )
        ]
    
    @staticmethod
    def get_all_test_cases() -> List[QueryTestCase]:
        """Get all test cases combined."""
        return (
            BiomedicalQueryDatasets.get_knowledge_graph_queries() +
            BiomedicalQueryDatasets.get_real_time_queries() +
            BiomedicalQueryDatasets.get_general_queries() +
            BiomedicalQueryDatasets.get_ambiguous_queries() +
            BiomedicalQueryDatasets.get_edge_case_queries()
        )


# ============================================================================
# MOCK API RESPONSES AND SCENARIOS
# ============================================================================

class MockLLMResponses:
    """Mock LLM responses for different scenarios."""
    
    @staticmethod
    def get_successful_response(category: str = "KNOWLEDGE_GRAPH", confidence: float = 0.85) -> str:
        """Generate a successful LLM response."""
        responses = {
            "KNOWLEDGE_GRAPH": {
                "category": category,
                "confidence": confidence,
                "reasoning": "Query asks about established biomedical relationships and mechanisms",
                "alternative_categories": [
                    {"category": "GENERAL", "confidence": 0.3}
                ],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["glucose", "metabolism", "insulin", "pathway"],
                    "relationships": ["signaling pathway", "metabolic relationship"],
                    "techniques": []
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            },
            "REAL_TIME": {
                "category": category,
                "confidence": confidence,
                "reasoning": "Query contains temporal indicators requesting current information",
                "alternative_categories": [
                    {"category": "GENERAL", "confidence": 0.4}
                ],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["biomarker", "research"],
                    "relationships": [],
                    "techniques": ["LC-MS", "metabolomics"]
                },
                "temporal_signals": {
                    "keywords": ["latest", "recent", "2025"],
                    "patterns": ["current research"],
                    "years": ["2025"]
                }
            },
            "GENERAL": {
                "category": category,
                "confidence": confidence,
                "reasoning": "General query that could be addressed by multiple approaches",
                "alternative_categories": [
                    {"category": "KNOWLEDGE_GRAPH", "confidence": 0.4},
                    {"category": "REAL_TIME", "confidence": 0.3}
                ],
                "uncertainty_indicators": ["general scope", "multiple approaches"],
                "biomedical_signals": {
                    "entities": ["metabolomics", "analysis"],
                    "relationships": [],
                    "techniques": ["analysis methods"]
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        }
        
        return json.dumps(responses.get(category, responses["GENERAL"]))
    
    @staticmethod
    def get_uncertain_response(confidence: float = 0.4) -> str:
        """Generate an uncertain LLM response."""
        return json.dumps({
            "category": "GENERAL",
            "confidence": confidence,
            "reasoning": "Query is ambiguous and could be interpreted in multiple ways",
            "alternative_categories": [
                {"category": "KNOWLEDGE_GRAPH", "confidence": 0.35},
                {"category": "REAL_TIME", "confidence": 0.25}
            ],
            "uncertainty_indicators": [
                "ambiguous phrasing",
                "multiple interpretations possible",
                "unclear intent"
            ],
            "biomedical_signals": {
                "entities": ["metabolomics"],
                "relationships": [],
                "techniques": []
            },
            "temporal_signals": {
                "keywords": [],
                "patterns": [],
                "years": []
            }
        })
    
    @staticmethod
    def get_malformed_responses() -> List[str]:
        """Get various malformed response examples."""
        return [
            "Not JSON at all",
            '{"invalid": "json"',  # Incomplete JSON
            '{"category": "INVALID_CATEGORY", "confidence": 0.8}',  # Invalid category
            '{"category": "GENERAL", "confidence": 1.5}',  # Invalid confidence range
            '{"category": "GENERAL"}',  # Missing required fields
            json.dumps({"category": "GENERAL", "confidence": "not_a_number"}),  # Wrong type
            '{}',  # Empty object
            json.dumps({"category": None, "confidence": 0.8, "reasoning": "Invalid category"}),  # None category
        ]
    
    @staticmethod
    def create_mock_openai_client(scenario: str = "success") -> AsyncMock:
        """Create a mock OpenAI client for different scenarios."""
        client = AsyncMock()
        
        if scenario == "success":
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = MockLLMResponses.get_successful_response()
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            client.chat.completions.create.return_value = mock_response
            
        elif scenario == "timeout":
            import asyncio
            client.chat.completions.create.side_effect = asyncio.TimeoutError("Request timeout")
            
        elif scenario == "network_error":
            client.chat.completions.create.side_effect = ConnectionError("Network error")
            
        elif scenario == "api_error":
            client.chat.completions.create.side_effect = Exception("API Error")
            
        elif scenario == "malformed":
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Invalid JSON response"
            client.chat.completions.create.return_value = mock_response
            
        elif scenario == "rate_limit":
            from openai import RateLimitError
            client.chat.completions.create.side_effect = RateLimitError("Rate limit exceeded")
            
        return client


# ============================================================================
# PERFORMANCE TEST DATA GENERATORS
# ============================================================================

class PerformanceTestDataGenerator:
    """Generate test data for performance testing."""
    
    @staticmethod
    def generate_query_batches(batch_sizes: List[int], query_patterns: List[str]) -> Dict[int, List[str]]:
        """Generate batches of queries for load testing."""
        batches = {}
        
        for batch_size in batch_sizes:
            batch = []
            for i in range(batch_size):
                pattern = random.choice(query_patterns)
                query = pattern.format(
                    entity=random.choice(["glucose", "insulin", "metabolite", "biomarker"]),
                    process=random.choice(["metabolism", "signaling", "biosynthesis", "regulation"]),
                    condition=random.choice(["diabetes", "cancer", "cardiovascular", "obesity"]),
                    year=random.choice(["2024", "2025"]),
                    number=i
                )
                batch.append(query)
            batches[batch_size] = batch
        
        return batches
    
    @staticmethod
    def generate_concurrent_queries(count: int) -> List[str]:
        """Generate queries for concurrent testing."""
        base_queries = [
            "What is {entity} {process} in {condition}?",
            "Latest research on {entity} {process} {year}",
            "How does {entity} affect {process} pathways?",
            "Recent {entity} discoveries in {condition} treatment",
            "Mechanism of {entity} in {process} regulation"
        ]
        
        queries = []
        for i in range(count):
            template = random.choice(base_queries)
            query = template.format(
                entity=random.choice(["glucose", "insulin", "metabolite", "biomarker", "protein"]),
                process=random.choice(["metabolism", "signaling", "transport", "regulation", "synthesis"]),
                condition=random.choice(["diabetes", "cancer", "obesity", "hypertension", "depression"]),
                year=random.choice(["2024", "2025"])
            )
            queries.append(f"{query}_{i}")  # Add unique identifier
        
        return queries
    
    @staticmethod
    def generate_cache_test_queries(unique_count: int, total_count: int, repetition_pattern: str = "zipf") -> List[str]:
        """Generate queries for cache testing with realistic access patterns."""
        unique_queries = [f"metabolomics query {i}" for i in range(unique_count)]
        
        if repetition_pattern == "zipf":
            # Zipf distribution - some queries much more popular
            weights = [1.0 / (i + 1) for i in range(unique_count)]
        elif repetition_pattern == "uniform":
            # Uniform distribution
            weights = [1.0] * unique_count
        else:
            # Heavy tail - most queries accessed once, few accessed many times
            weights = [1.0] * (unique_count // 2) + [10.0] * (unique_count // 2)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Generate query sequence
        queries = []
        for _ in range(total_count):
            query = random.choices(unique_queries, weights=weights)[0]
            queries.append(query)
        
        return queries


# ============================================================================
# CONFIDENCE CALIBRATION TEST DATA
# ============================================================================

class ConfidenceCalibrationTestData:
    """Generate test data for confidence calibration testing."""
    
    @staticmethod
    def generate_calibration_history(
        num_predictions: int,
        accuracy_by_confidence: Dict[float, float] = None
    ) -> List[Tuple[float, bool, str]]:
        """Generate historical prediction data for calibration testing."""
        
        if accuracy_by_confidence is None:
            # Default calibration curve - well calibrated
            accuracy_by_confidence = {
                0.1: 0.15,  # Slightly overconfident at low confidence
                0.3: 0.35,
                0.5: 0.55,
                0.7: 0.75,
                0.9: 0.85   # Slightly underconfident at high confidence
            }
        
        history = []
        confidence_levels = list(accuracy_by_confidence.keys())
        
        for i in range(num_predictions):
            confidence = random.choice(confidence_levels)
            expected_accuracy = accuracy_by_confidence[confidence]
            
            # Add some noise
            noise = random.uniform(-0.1, 0.1)
            actual_accuracy_prob = min(1.0, max(0.0, expected_accuracy + noise))
            
            is_accurate = random.random() < actual_accuracy_prob
            query_type = random.choice(["knowledge_graph", "real_time", "general"])
            
            history.append((confidence, is_accurate, query_type))
        
        return history
    
    @staticmethod
    def generate_poorly_calibrated_data(num_predictions: int) -> List[Tuple[float, bool, str]]:
        """Generate data for a poorly calibrated system."""
        
        # Overconfident system
        accuracy_by_confidence = {
            0.1: 0.1,   # Accurate at low confidence
            0.3: 0.2,   # Overconfident
            0.5: 0.3,   # Very overconfident  
            0.7: 0.4,   # Extremely overconfident
            0.9: 0.5    # Extremely overconfident
        }
        
        return ConfidenceCalibrationTestData.generate_calibration_history(
            num_predictions, accuracy_by_confidence
        )
    
    @staticmethod
    def generate_confidence_scenarios() -> Dict[str, Dict[str, Any]]:
        """Generate different confidence calibration scenarios."""
        
        scenarios = {
            "well_calibrated": {
                "history": ConfidenceCalibrationTestData.generate_calibration_history(200),
                "expected_brier_score": (0.1, 0.3),
                "expected_calibration_slope": (0.8, 1.2),
                "description": "Well-calibrated system with good accuracy"
            },
            "overconfident": {
                "history": ConfidenceCalibrationTestData.generate_poorly_calibrated_data(200),
                "expected_brier_score": (0.3, 0.6),
                "expected_calibration_slope": (0.3, 0.7),
                "description": "Overconfident system with poor calibration"
            },
            "underconfident": {
                "history": [(conf * 0.7, acc, qtype) for conf, acc, qtype in 
                           ConfidenceCalibrationTestData.generate_calibration_history(200)],
                "expected_brier_score": (0.2, 0.4),
                "expected_calibration_slope": (1.2, 2.0),
                "description": "Underconfident system"
            },
            "insufficient_data": {
                "history": ConfidenceCalibrationTestData.generate_calibration_history(5),
                "expected_brier_score": (0.0, 1.0),  # Wide range due to insufficient data
                "expected_calibration_slope": (0.1, 3.0),
                "description": "System with insufficient calibration data"
            }
        }
        
        return scenarios


# ============================================================================
# INTEGRATION TEST HELPERS
# ============================================================================

class IntegrationTestHelpers:
    """Helpers for integration testing."""
    
    @staticmethod
    def create_mock_biomedical_router() -> Mock:
        """Create a mock BiomedicalQueryRouter with realistic responses."""
        router = Mock()
        
        def mock_route_query(query_text: str, context: Optional[Dict] = None) -> Mock:
            """Mock routing logic based on query content."""
            query_lower = query_text.lower()
            
            # Determine routing based on content
            if any(temporal in query_lower for temporal in ["latest", "recent", "2024", "2025", "current"]):
                routing_decision = RoutingDecision.PERPLEXITY
                confidence = 0.8
                knowledge_indicators = []
                temporal_indicators = [word for word in ["latest", "recent", "2024", "2025", "current"] 
                                     if word in query_lower]
            elif any(relationship in query_lower for relationship in ["pathway", "mechanism", "relationship"]):
                routing_decision = RoutingDecision.LIGHTRAG
                confidence = 0.85
                knowledge_indicators = ["pathway", "mechanism", "relationship"]
                temporal_indicators = []
            else:
                routing_decision = RoutingDecision.EITHER
                confidence = 0.6
                knowledge_indicators = []
                temporal_indicators = []
            
            # Create confidence metrics
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=confidence,
                research_category_confidence=confidence,
                temporal_analysis_confidence=0.8 if temporal_indicators else 0.3,
                signal_strength_confidence=confidence,
                context_coherence_confidence=confidence,
                keyword_density=len(query_text.split()) / 100,  # Simple approximation
                pattern_match_strength=confidence,
                biomedical_entity_count=len([w for w in query_lower.split() 
                                            if w in ["glucose", "insulin", "metabolite", "biomarker"]]),
                ambiguity_score=0.3 if confidence < 0.7 else 0.1,
                conflict_score=0.1,
                alternative_interpretations=[],
                calculation_time_ms=25.0
            )
            
            return RoutingPrediction(
                routing_decision=routing_decision,
                confidence=confidence,
                reasoning=[f"Mock routing based on content analysis"],
                research_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
                confidence_metrics=confidence_metrics,
                temporal_indicators=temporal_indicators,
                knowledge_indicators=knowledge_indicators,
                metadata={"mock": True}
            )
        
        router.route_query.side_effect = mock_route_query
        return router
    
    @staticmethod
    def create_test_classification_result(
        category: str = "KNOWLEDGE_GRAPH",
        confidence: float = 0.85,
        entities: List[str] = None,
        temporal_keywords: List[str] = None
    ) -> 'ClassificationResult':
        """Create a test classification result."""
        
        if entities is None:
            entities = ["glucose", "metabolism", "insulin"]
        if temporal_keywords is None:
            temporal_keywords = []
        
        return ClassificationResult(
            category=category,
            confidence=confidence,
            reasoning=f"Test classification result for {category} with confidence {confidence}",
            alternative_categories=[
                {"category": "GENERAL", "confidence": confidence * 0.4}
            ] if confidence > 0.7 else [],
            uncertainty_indicators=["test_scenario"] if confidence < 0.6 else [],
            biomedical_signals={
                "entities": entities,
                "relationships": ["pathway", "signaling"] if category == "KNOWLEDGE_GRAPH" else [],
                "techniques": ["LC-MS", "metabolomics"] if "metabolomics" in str(entities) else []
            },
            temporal_signals={
                "keywords": temporal_keywords,
                "patterns": ["recent research"] if temporal_keywords else [],
                "years": [kw for kw in temporal_keywords if kw.isdigit()]
            }
        )
    
    @staticmethod
    def create_test_confidence_analyses(scenario: str = "normal") -> Tuple[LLMConfidenceAnalysis, KeywordConfidenceAnalysis]:
        """Create test confidence analyses for different scenarios."""
        
        scenarios = {
            "high_confidence": {
                "llm": LLMConfidenceAnalysis(
                    raw_confidence=0.9, calibrated_confidence=0.88,
                    reasoning_quality_score=0.9, consistency_score=0.95,
                    response_length=200, reasoning_depth=4,
                    uncertainty_indicators=[], confidence_expressions=["high confidence", "clear evidence"]
                ),
                "keyword": KeywordConfidenceAnalysis(
                    pattern_match_confidence=0.9, keyword_density_confidence=0.85,
                    biomedical_entity_confidence=0.95, domain_specificity_confidence=0.9,
                    total_biomedical_signals=8, strong_signals=7, weak_signals=1,
                    conflicting_signals=0, semantic_coherence_score=0.9,
                    domain_alignment_score=0.95, query_completeness_score=0.85
                )
            },
            "low_confidence": {
                "llm": LLMConfidenceAnalysis(
                    raw_confidence=0.4, calibrated_confidence=0.35,
                    reasoning_quality_score=0.3, consistency_score=0.6,
                    response_length=50, reasoning_depth=1,
                    uncertainty_indicators=["uncertain", "unclear", "ambiguous"],
                    confidence_expressions=["possibly", "might be"]
                ),
                "keyword": KeywordConfidenceAnalysis(
                    pattern_match_confidence=0.3, keyword_density_confidence=0.2,
                    biomedical_entity_confidence=0.4, domain_specificity_confidence=0.3,
                    total_biomedical_signals=2, strong_signals=0, weak_signals=2,
                    conflicting_signals=1, semantic_coherence_score=0.4,
                    domain_alignment_score=0.3, query_completeness_score=0.2
                )
            },
            "normal": {
                "llm": LLMConfidenceAnalysis(
                    raw_confidence=0.75, calibrated_confidence=0.73,
                    reasoning_quality_score=0.7, consistency_score=0.8,
                    response_length=120, reasoning_depth=3,
                    uncertainty_indicators=[], confidence_expressions=["likely", "suggests"]
                ),
                "keyword": KeywordConfidenceAnalysis(
                    pattern_match_confidence=0.7, keyword_density_confidence=0.6,
                    biomedical_entity_confidence=0.8, domain_specificity_confidence=0.7,
                    total_biomedical_signals=4, strong_signals=3, weak_signals=1,
                    conflicting_signals=0, semantic_coherence_score=0.75,
                    domain_alignment_score=0.8, query_completeness_score=0.7
                )
            }
        }
        
        scenario_data = scenarios.get(scenario, scenarios["normal"])
        return scenario_data["llm"], scenario_data["keyword"]


# ============================================================================
# TEST UTILITIES AND FIXTURES
# ============================================================================

class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_confidence_range(confidence: float, expected_range: Tuple[float, float], message: str = ""):
        """Assert that confidence is within expected range."""
        lower, upper = expected_range
        assert lower <= confidence <= upper, f"Confidence {confidence} not in range [{lower}, {upper}]. {message}"
    
    @staticmethod
    def assert_response_time(start_time: float, end_time: float, max_time: float, message: str = ""):
        """Assert that response time is within acceptable limits."""
        elapsed = end_time - start_time
        assert elapsed <= max_time, f"Response time {elapsed:.3f}s exceeded {max_time}s. {message}"
    
    @staticmethod
    def generate_random_biomedical_query(complexity: str = "medium") -> str:
        """Generate a random biomedical query for testing."""
        
        entities = ["glucose", "insulin", "metabolite", "biomarker", "protein", "pathway"]
        processes = ["metabolism", "signaling", "regulation", "synthesis", "transport"]
        conditions = ["diabetes", "cancer", "obesity", "cardiovascular", "neurological"]
        techniques = ["LC-MS", "NMR", "metabolomics", "proteomics", "genomics"]
        
        if complexity == "simple":
            return f"{random.choice(entities)} {random.choice(processes)}"
        elif complexity == "complex":
            return (f"What is the detailed mechanism of {random.choice(entities)} "
                   f"{random.choice(processes)} in {random.choice(conditions)} patients "
                   f"using {random.choice(techniques)} analysis?")
        else:  # medium
            return (f"How does {random.choice(entities)} affect {random.choice(processes)} "
                   f"in {random.choice(conditions)}?")
    
    @staticmethod
    def create_test_logger(name: str = "test", level: int = logging.INFO) -> logging.Logger:
        """Create a test logger with appropriate configuration."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def biomedical_queries():
    """Fixture providing biomedical query test cases."""
    return BiomedicalQueryDatasets.get_all_test_cases()

@pytest.fixture
def mock_llm_client():
    """Fixture providing mock OpenAI client."""
    return MockLLMResponses.create_mock_openai_client("success")

@pytest.fixture
def performance_test_data():
    """Fixture providing performance test data."""
    return PerformanceTestDataGenerator.generate_concurrent_queries(20)

@pytest.fixture
def calibration_test_data():
    """Fixture providing confidence calibration test data."""
    return ConfidenceCalibrationTestData.generate_confidence_scenarios()

@pytest.fixture
def integration_helpers():
    """Fixture providing integration test helpers."""
    return IntegrationTestHelpers()

@pytest.fixture
def test_logger():
    """Fixture providing test logger."""
    return TestUtilities.create_test_logger()


if __name__ == "__main__":
    # Demo of test fixtures
    print("=== Biomedical Query Test Fixtures Demo ===")
    
    # Show sample queries
    knowledge_queries = BiomedicalQueryDatasets.get_knowledge_graph_queries()
    print(f"\nKnowledge Graph Queries: {len(knowledge_queries)}")
    for query in knowledge_queries[:2]:
        print(f"  - {query.query_text}")
        print(f"    Expected: {query.expected_category} ({query.expected_confidence_range})")
    
    real_time_queries = BiomedicalQueryDatasets.get_real_time_queries()
    print(f"\nReal-Time Queries: {len(real_time_queries)}")
    for query in real_time_queries[:2]:
        print(f"  - {query.query_text}")
        print(f"    Expected: {query.expected_category} ({query.expected_confidence_range})")
    
    # Show performance test data
    concurrent_queries = PerformanceTestDataGenerator.generate_concurrent_queries(5)
    print(f"\nSample Concurrent Queries: {len(concurrent_queries)}")
    for query in concurrent_queries:
        print(f"  - {query}")
    
    # Show calibration scenarios
    scenarios = ConfidenceCalibrationTestData.generate_confidence_scenarios()
    print(f"\nCalibration Scenarios: {list(scenarios.keys())}")
    for name, scenario in scenarios.items():
        print(f"  - {name}: {scenario['description']}")
    
    print("\n=== Test fixtures ready for comprehensive testing ===")