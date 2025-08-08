"""
Unit tests for the Biomedical Query Router module.

This test suite validates the query routing functionality including:
- Routing decision classification
- Temporal analysis
- Research category mapping
- Confidence scoring
- Integration with existing ResearchCategorizer
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from ..query_router import (
    BiomedicalQueryRouter, 
    RoutingDecision, 
    RoutingPrediction, 
    TemporalAnalyzer
)
from ..cost_persistence import ResearchCategory


class TestTemporalAnalyzer:
    """Test suite for TemporalAnalyzer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TemporalAnalyzer()
    
    def test_temporal_analyzer_initialization(self):
        """Test that TemporalAnalyzer initializes correctly."""
        assert len(self.analyzer.temporal_keywords) > 20
        assert len(self.analyzer.temporal_patterns) > 10
        assert len(self.analyzer.established_patterns) > 5
        
        # Check for key temporal keywords
        assert 'latest' in self.analyzer.temporal_keywords
        assert 'recent' in self.analyzer.temporal_keywords
        assert '2024' in self.analyzer.temporal_keywords
        assert '2025' in self.analyzer.temporal_keywords
    
    @pytest.mark.parametrize("query_text,expected_temporal", [
        ("What are the latest metabolomics research findings?", True),
        ("Recent advances in clinical metabolomics", True),
        ("Current trends in biomarker discovery", True),
        ("What is published in 2024 about metabolites?", True),
        ("Breaking news in drug discovery", True),
        ("What is metabolomics?", False),
        ("Explain the pathway analysis methodology", False),
        ("Define biomarkers in clinical diagnosis", False),
        ("Mechanism of metabolite identification", False)
    ])
    def test_temporal_keyword_detection(self, query_text: str, expected_temporal: bool):
        """Test detection of temporal keywords in queries."""
        analysis = self.analyzer.analyze_temporal_content(query_text)
        
        if expected_temporal:
            assert analysis['has_temporal_keywords'] == True
            assert len(analysis['temporal_keywords_found']) > 0
            assert analysis['temporal_score'] > 0
        else:
            # May have some temporal words but should be low score
            if analysis['has_temporal_keywords']:
                assert analysis['temporal_score'] <= 2.0  # Allow for minor temporal presence
    
    @pytest.mark.parametrize("query_text,expected_patterns", [
        ("What are the latest research developments?", True),
        ("Recent studies published in 2024", True),
        ("Current clinical trials in progress", True),
        ("Just published metabolomics findings", True),
        ("What's new in biomarker discovery?", True),
        ("Standard metabolite identification methods", False),
        ("Traditional pathway analysis approach", False),
        ("What is the definition of metabolomics?", False)
    ])
    def test_temporal_pattern_detection(self, query_text: str, expected_patterns: bool):
        """Test detection of temporal patterns in queries."""
        analysis = self.analyzer.analyze_temporal_content(query_text)
        
        if expected_patterns:
            assert analysis['has_temporal_patterns'] == True
            assert len(analysis['temporal_patterns_found']) > 0
        else:
            assert analysis['has_temporal_patterns'] == False or len(analysis['temporal_patterns_found']) == 0
    
    def test_established_knowledge_detection(self):
        """Test detection of established knowledge patterns."""
        queries_with_established = [
            "What is clinical metabolomics?",
            "Define biomarker discovery",
            "Explain the mechanism of action",
            "Describe pathway analysis methods",
            "What are the fundamental principles?",
            "History of metabolite identification"
        ]
        
        for query in queries_with_established:
            analysis = self.analyzer.analyze_temporal_content(query)
            assert analysis['has_established_patterns'] == True, f"Failed for: {query}"
            assert analysis['established_score'] > 0, f"No established score for: {query}"
    
    def test_year_mention_detection(self):
        """Test detection of specific year mentions."""
        test_cases = [
            ("Research published in 2024", ['2024']),
            ("Studies from 2025 and 2026", ['2025', '2026']),
            ("Data from the year 2030", ['2030']),
            ("Historical data from 1990", []),  # Too old, should not match
            ("No year mentions here", [])
        ]
        
        for query, expected_years in test_cases:
            analysis = self.analyzer.analyze_temporal_content(query)
            assert analysis['year_mentions'] == expected_years, f"Failed for: {query}"


class TestBiomedicalQueryRouter:
    """Test suite for BiomedicalQueryRouter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock logger to avoid logging setup in tests
        mock_logger = Mock()
        self.router = BiomedicalQueryRouter(logger=mock_logger)
    
    def test_router_initialization(self):
        """Test that BiomedicalQueryRouter initializes correctly."""
        assert self.router.temporal_analyzer is not None
        assert isinstance(self.router.category_routing_map, dict)
        assert len(self.router.category_routing_map) > 10
        assert isinstance(self.router.routing_thresholds, dict)
        
        # Check that all research categories are mapped
        for category in ResearchCategory:
            assert category in self.router.category_routing_map
    
    def test_category_routing_map_validity(self):
        """Test that category routing map contains valid routing decisions."""
        for category, routing in self.router.category_routing_map.items():
            assert isinstance(category, ResearchCategory)
            assert isinstance(routing, RoutingDecision)
        
        # Check specific mappings from requirements
        assert self.router.category_routing_map[ResearchCategory.PATHWAY_ANALYSIS] == RoutingDecision.LIGHTRAG
        assert self.router.category_routing_map[ResearchCategory.METABOLITE_IDENTIFICATION] == RoutingDecision.LIGHTRAG
        assert self.router.category_routing_map[ResearchCategory.LITERATURE_SEARCH] == RoutingDecision.PERPLEXITY
    
    @pytest.mark.parametrize("query_text,expected_routing_group,min_confidence", [
        # Knowledge graph queries (relationships, pathways, mechanisms)
        ("What are the metabolic pathways involved in diabetes?", "knowledge_preferred", 0.4),
        ("Show me the relationship between metabolites and disease", "knowledge_preferred", 0.4),
        ("How do biomarkers connect to clinical diagnosis?", "knowledge_preferred", 0.4),
        ("Mechanism of metabolite identification in mass spectrometry", "knowledge_preferred", 0.4),
        
        # Real-time queries (latest, recent, current)
        ("What are the latest metabolomics research findings?", "temporal_preferred", 0.4),
        ("Recent advances in clinical metabolomics 2024", "temporal_preferred", 0.4),
        ("Current trends in biomarker discovery", "temporal_preferred", 0.4),
        ("Breaking news in drug discovery", "temporal_preferred", 0.4),
        
        # General queries (what is, define, explain) - flexible routing acceptable
        ("What is clinical metabolomics?", "flexible", 0.3),
        ("Define biomarker discovery", "flexible", 0.3),
        ("Explain pathway analysis", "knowledge_preferred", 0.4)  # This should go to LightRAG due to pathway
    ])
    def test_query_routing_decisions(self, query_text: str, expected_routing_group: str, min_confidence: float):
        """Test routing decisions for various query types."""
        prediction = self.router.route_query(query_text)
        
        assert isinstance(prediction, RoutingPrediction)
        
        # Check routing based on expected group
        if expected_routing_group == "knowledge_preferred":
            # Should route to LightRAG, EITHER, or HYBRID (knowledge graph friendly options)
            assert prediction.routing_decision in [
                RoutingDecision.LIGHTRAG, 
                RoutingDecision.EITHER, 
                RoutingDecision.HYBRID
            ], f"Expected knowledge-preferred routing, got {prediction.routing_decision.value} for: {query_text}"
        elif expected_routing_group == "temporal_preferred":
            # Should route to Perplexity, EITHER, or HYBRID (real-time friendly options)
            assert prediction.routing_decision in [
                RoutingDecision.PERPLEXITY, 
                RoutingDecision.EITHER, 
                RoutingDecision.HYBRID
            ], f"Expected temporal-preferred routing, got {prediction.routing_decision.value} for: {query_text}"
        elif expected_routing_group == "flexible":
            # Any routing decision is acceptable for general queries
            assert prediction.routing_decision in [
                RoutingDecision.LIGHTRAG, 
                RoutingDecision.PERPLEXITY, 
                RoutingDecision.EITHER, 
                RoutingDecision.HYBRID
            ], f"Expected valid routing decision, got {prediction.routing_decision.value} for: {query_text}"
        
        assert prediction.confidence >= min_confidence, f"Confidence {prediction.confidence} below minimum {min_confidence} for: {query_text}"
        assert len(prediction.reasoning) > 0
        assert isinstance(prediction.research_category, ResearchCategory)
    
    def test_routing_prediction_structure(self):
        """Test that RoutingPrediction contains all required fields."""
        query = "What are metabolic pathways in diabetes?"
        prediction = self.router.route_query(query)
        
        # Check all required fields
        assert hasattr(prediction, 'routing_decision')
        assert hasattr(prediction, 'confidence')
        assert hasattr(prediction, 'reasoning')
        assert hasattr(prediction, 'research_category')
        assert hasattr(prediction, 'temporal_indicators')
        assert hasattr(prediction, 'knowledge_indicators')
        assert hasattr(prediction, 'metadata')
        
        # Check types
        assert isinstance(prediction.routing_decision, RoutingDecision)
        assert isinstance(prediction.confidence, float)
        assert isinstance(prediction.reasoning, list)
        assert isinstance(prediction.research_category, ResearchCategory)
        assert 0.0 <= prediction.confidence <= 1.0
    
    def test_routing_prediction_serialization(self):
        """Test that RoutingPrediction can be serialized to dictionary."""
        query = "Latest research in metabolomics"
        prediction = self.router.route_query(query)
        
        prediction_dict = prediction.to_dict()
        
        # Check required keys
        required_keys = [
            'routing_decision', 'confidence', 'reasoning', 
            'research_category', 'temporal_indicators', 
            'knowledge_indicators', 'metadata'
        ]
        
        for key in required_keys:
            assert key in prediction_dict, f"Missing key: {key}"
        
        # Check value types
        assert isinstance(prediction_dict['routing_decision'], str)
        assert isinstance(prediction_dict['confidence'], float)
        assert isinstance(prediction_dict['reasoning'], list)
        assert isinstance(prediction_dict['research_category'], str)
    
    def test_temporal_routing_override(self):
        """Test that strong temporal indicators override category preferences."""
        # Pathway analysis normally goes to LightRAG, but with temporal indicators should go to Perplexity
        query = "What are the latest advances in pathway analysis published in 2024?"
        prediction = self.router.route_query(query)
        
        # Should route to Perplexity due to strong temporal indicators
        assert prediction.routing_decision in [RoutingDecision.PERPLEXITY, RoutingDecision.HYBRID]
        assert any('temporal' in reason.lower() for reason in prediction.reasoning)
    
    def test_knowledge_graph_routing_preference(self):
        """Test that knowledge graph indicators strengthen LightRAG routing."""
        knowledge_queries = [
            "What is the relationship between metabolites and diabetes?",
            "Show connections between biomarkers and disease pathways",
            "How do metabolic mechanisms relate to clinical outcomes?"
        ]
        
        for query in knowledge_queries:
            prediction = self.router.route_query(query)
            # Should prefer LightRAG for relationship/connection queries
            assert prediction.routing_decision in [RoutingDecision.LIGHTRAG, RoutingDecision.EITHER, RoutingDecision.HYBRID]
    
    def test_should_use_lightrag_helper(self):
        """Test the boolean helper method for LightRAG usage."""
        lightrag_queries = [
            "What are metabolic pathways?",
            "Show me biomarker relationships",
            "Mechanism of metabolite identification"
        ]
        
        for query in lightrag_queries:
            should_use = self.router.should_use_lightrag(query)
            assert isinstance(should_use, bool)
            
            if should_use:
                prediction = self.router.route_query(query)
                assert prediction.routing_decision in [RoutingDecision.LIGHTRAG, RoutingDecision.HYBRID]
    
    def test_should_use_perplexity_helper(self):
        """Test the boolean helper method for Perplexity usage."""
        perplexity_queries = [
            "Latest metabolomics research findings",
            "Recent advances in 2024",
            "What's new in biomarker discovery?"
        ]
        
        for query in perplexity_queries:
            should_use = self.router.should_use_perplexity(query)
            assert isinstance(should_use, bool)
            
            if should_use:
                prediction = self.router.route_query(query)
                assert prediction.routing_decision in [
                    RoutingDecision.PERPLEXITY, 
                    RoutingDecision.EITHER, 
                    RoutingDecision.HYBRID
                ]
    
    def test_confidence_scoring_consistency(self):
        """Test that confidence scoring is consistent and reasonable."""
        test_queries = [
            "What is metabolomics?",  # Should have moderate confidence
            "Latest metabolomics research in 2024",  # Should have good confidence
            "Complex multi-part query about relationships between metabolic pathways and disease mechanisms in clinical settings with recent advances",  # Complex query
            "xyz",  # Very simple query
        ]
        
        for query in test_queries:
            prediction = self.router.route_query(query)
            
            # Confidence should be between 0 and 1
            assert 0.0 <= prediction.confidence <= 1.0
            
            # Reasoning should be provided
            assert len(prediction.reasoning) > 0
            
            # Metadata should contain analysis details
            assert 'category_prediction' in prediction.metadata
            assert 'temporal_analysis' in prediction.metadata
            assert 'routing_scores' in prediction.metadata
    
    def test_context_integration(self):
        """Test that context information is properly integrated."""
        query = "What is metabolomics?"
        context = {
            'previous_categories': ['pathway_analysis'],
            'user_research_areas': ['clinical_diagnosis'],
            'project_type': 'clinical_study'
        }
        
        prediction_with_context = self.router.route_query(query, context)
        prediction_without_context = self.router.route_query(query)
        
        # Both should return valid predictions
        assert isinstance(prediction_with_context, RoutingPrediction)
        assert isinstance(prediction_without_context, RoutingPrediction)
        
        # Context may influence routing, but both should be reasonable
        assert prediction_with_context.confidence > 0
        assert prediction_without_context.confidence > 0
    
    def test_fallback_routing(self):
        """Test that fallback routing works for edge cases."""
        edge_case_queries = [
            "",  # Empty query
            "a",  # Single character
            "   ",  # Whitespace only
            "12345",  # Numbers only
        ]
        
        for query in edge_case_queries:
            prediction = self.router.route_query(query)
            
            # Should still return a valid prediction
            assert isinstance(prediction, RoutingPrediction)
            assert isinstance(prediction.routing_decision, RoutingDecision)
            
            # Low-confidence queries should often route to EITHER
            if prediction.confidence < 0.3:
                assert prediction.routing_decision in [RoutingDecision.EITHER, RoutingDecision.HYBRID]
    
    def test_routing_statistics(self):
        """Test that routing statistics are properly calculated."""
        stats = self.router.get_routing_statistics()
        
        # Should contain routing-specific statistics
        assert 'routing_thresholds' in stats
        assert 'category_routing_map' in stats
        assert 'temporal_keywords_count' in stats
        assert 'temporal_patterns_count' in stats
        
        # Should also contain base categorization stats
        assert 'total_predictions' in stats
        assert 'confidence_distribution' in stats
        
        # Verify counts
        assert stats['temporal_keywords_count'] > 20
        assert stats['temporal_patterns_count'] > 10
    
    def test_inheritance_from_research_categorizer(self):
        """Test that BiomedicalQueryRouter properly inherits from ResearchCategorizer."""
        # Should have all parent methods
        assert hasattr(self.router, 'categorize_query')
        assert hasattr(self.router, 'get_category_statistics')
        assert hasattr(self.router, 'update_from_feedback')
        
        # Should be able to use parent functionality
        query = "What is metabolomics?"
        category_prediction = self.router.categorize_query(query)
        
        assert hasattr(category_prediction, 'category')
        assert hasattr(category_prediction, 'confidence')
        assert isinstance(category_prediction.category, ResearchCategory)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        mock_logger = Mock()
        self.router = BiomedicalQueryRouter(logger=mock_logger)
    
    def test_mixed_temporal_knowledge_query(self):
        """Test queries that have both temporal and knowledge indicators."""
        query = "What are the latest advances in metabolic pathway analysis published this year?"
        prediction = self.router.route_query(query)
        
        # Should handle the conflict between temporal (Perplexity) and pathway (LightRAG)
        assert prediction.routing_decision in [
            RoutingDecision.PERPLEXITY,  # Temporal wins
            RoutingDecision.HYBRID,      # Use both
            RoutingDecision.EITHER       # Flexible
        ]
        
        # Should have reasonable confidence
        assert prediction.confidence > 0.3
        
        # Should explain the reasoning
        assert len(prediction.reasoning) > 1  # Multiple factors considered
    
    def test_clinical_workflow_sequence(self):
        """Test a sequence of queries that might occur in a clinical workflow."""
        clinical_queries = [
            ("What is clinical metabolomics?", RoutingDecision.EITHER),
            ("Show me metabolic pathways in diabetes", RoutingDecision.LIGHTRAG),
            ("What are the latest diabetes biomarkers discovered in 2024?", RoutingDecision.PERPLEXITY),
            ("How do these biomarkers relate to metabolic pathways?", RoutingDecision.LIGHTRAG),
        ]
        
        # Simulate context building through the sequence
        context = {'previous_categories': []}
        
        for query, expected_routing_type in clinical_queries:
            prediction = self.router.route_query(query, context)
            
            # Should route to appropriate service or flexible option
            if expected_routing_type == RoutingDecision.LIGHTRAG:
                assert prediction.routing_decision in [
                    RoutingDecision.LIGHTRAG, 
                    RoutingDecision.EITHER, 
                    RoutingDecision.HYBRID
                ]
            elif expected_routing_type == RoutingDecision.PERPLEXITY:
                assert prediction.routing_decision in [
                    RoutingDecision.PERPLEXITY, 
                    RoutingDecision.EITHER, 
                    RoutingDecision.HYBRID
                ]
            
            # Update context for next query
            context['previous_categories'].append(prediction.research_category.value)
    
    def test_performance_requirements(self):
        """Test that routing decisions are made quickly."""
        import time
        
        query = "What are the metabolic pathways involved in drug metabolism?"
        
        start_time = time.time()
        prediction = self.router.route_query(query)
        end_time = time.time()
        
        # Routing should be fast (less than 100ms for unit test)
        routing_time = end_time - start_time
        assert routing_time < 0.1, f"Routing took {routing_time:.3f}s, expected < 0.1s"
        
        # Should still produce valid result
        assert isinstance(prediction, RoutingPrediction)
        assert prediction.confidence > 0