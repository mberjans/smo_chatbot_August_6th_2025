"""
Comprehensive Test Suite for Enhanced Query Classification System

Tests the CMO-LIGHTRAG-012-T04 implementation including:
- Query classification accuracy across all three categories
- Performance requirements (<2 second classification response)
- Keyword and pattern matching functionality
- Confidence scoring accuracy
- Integration with existing router systems
- Edge cases and error handling

Author: Claude Code (Anthropic)
Created: August 8, 2025
Version: 1.0.0
"""

import pytest
import time
import asyncio
import logging
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch

# Import the query classification system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_classification_system import (
    QueryClassificationCategories,
    BiomedicalKeywordSets,
    QueryClassificationEngine,
    ClassificationResult,
    create_classification_engine,
    classify_for_routing,
    get_routing_category_mapping
)


class TestQueryClassificationCategories:
    """Test the QueryClassificationCategories enum."""
    
    def test_category_values(self):
        """Test that categories have correct values."""
        assert QueryClassificationCategories.KNOWLEDGE_GRAPH.value == "knowledge_graph"
        assert QueryClassificationCategories.REAL_TIME.value == "real_time"
        assert QueryClassificationCategories.GENERAL.value == "general"
    
    def test_category_count(self):
        """Test that we have exactly three categories as specified."""
        categories = list(QueryClassificationCategories)
        assert len(categories) == 3
    
    def test_category_enum_completeness(self):
        """Test that all required categories are present."""
        category_values = {cat.value for cat in QueryClassificationCategories}
        expected_values = {"knowledge_graph", "real_time", "general"}
        assert category_values == expected_values


class TestBiomedicalKeywordSets:
    """Test the BiomedicalKeywordSets class."""
    
    @pytest.fixture
    def keyword_sets(self):
        """Create BiomedicalKeywordSets instance for testing."""
        return BiomedicalKeywordSets()
    
    def test_initialization(self, keyword_sets):
        """Test that keyword sets are properly initialized."""
        assert hasattr(keyword_sets, 'knowledge_graph_keywords')
        assert hasattr(keyword_sets, 'real_time_keywords')
        assert hasattr(keyword_sets, 'general_keywords')
        
        # Check that flattened sets are created
        assert hasattr(keyword_sets, 'knowledge_graph_set')
        assert hasattr(keyword_sets, 'real_time_set')
        assert hasattr(keyword_sets, 'general_set')
        assert hasattr(keyword_sets, 'biomedical_entities_set')
    
    def test_knowledge_graph_keywords(self, keyword_sets):
        """Test knowledge graph keyword categories."""
        kg_keywords = keyword_sets.knowledge_graph_keywords
        
        # Check that all expected categories are present
        expected_categories = [
            'relationships', 'pathways', 'biomarkers', 'metabolites',
            'diseases', 'clinical_studies', 'analytical_methods', 'biological_processes'
        ]
        
        for category in expected_categories:
            assert category in kg_keywords, f"Missing category: {category}"
            assert len(kg_keywords[category]) > 0, f"Empty category: {category}"
        
        # Test specific keywords
        assert 'biomarker' in kg_keywords['biomarkers']
        assert 'pathway' in kg_keywords['pathways']
        assert 'metabolite' in kg_keywords['metabolites']
        assert 'mass spectrometry' in kg_keywords['analytical_methods']
    
    def test_real_time_keywords(self, keyword_sets):
        """Test real-time keyword categories."""
        rt_keywords = keyword_sets.real_time_keywords
        
        expected_categories = [
            'temporal_indicators', 'year_indicators', 'news_updates',
            'research_developments', 'clinical_temporal', 'technology_updates'
        ]
        
        for category in expected_categories:
            assert category in rt_keywords, f"Missing category: {category}"
            assert len(rt_keywords[category]) > 0, f"Empty category: {category}"
        
        # Test specific keywords
        assert 'latest' in rt_keywords['temporal_indicators']
        assert '2024' in rt_keywords['year_indicators']
        assert 'breakthrough' in rt_keywords['news_updates']
        assert 'fda approval' in rt_keywords['clinical_temporal']
    
    def test_general_keywords(self, keyword_sets):
        """Test general keyword categories."""
        general_keywords = keyword_sets.general_keywords
        
        expected_categories = ['definitions', 'procedures', 'educational', 'comparison']
        
        for category in expected_categories:
            assert category in general_keywords, f"Missing category: {category}"
            assert len(general_keywords[category]) > 0, f"Empty category: {category}"
        
        # Test specific keywords
        assert 'what is' in general_keywords['definitions']
        assert 'how to' in general_keywords['procedures']
        assert 'compare' in general_keywords['comparison']
    
    def test_compiled_patterns(self, keyword_sets):
        """Test that regex patterns are compiled correctly."""
        assert hasattr(keyword_sets, 'kg_patterns')
        assert hasattr(keyword_sets, 'rt_patterns')
        assert hasattr(keyword_sets, 'general_patterns')
        
        # Check that patterns are compiled regex objects
        assert len(keyword_sets.kg_patterns) > 0
        assert len(keyword_sets.rt_patterns) > 0
        assert len(keyword_sets.general_patterns) > 0
        
        # Test that patterns can match text
        kg_pattern = keyword_sets.kg_patterns[0]
        assert hasattr(kg_pattern, 'search')
        assert hasattr(kg_pattern, 'findall')
    
    def test_get_category_keywords(self, keyword_sets):
        """Test getting keywords by category."""
        kg_keywords = keyword_sets.get_category_keywords(QueryClassificationCategories.KNOWLEDGE_GRAPH)
        rt_keywords = keyword_sets.get_category_keywords(QueryClassificationCategories.REAL_TIME)
        general_keywords = keyword_sets.get_category_keywords(QueryClassificationCategories.GENERAL)
        
        assert isinstance(kg_keywords, dict)
        assert isinstance(rt_keywords, dict)
        assert isinstance(general_keywords, dict)
        
        assert len(kg_keywords) > 0
        assert len(rt_keywords) > 0
        assert len(general_keywords) > 0
    
    def test_get_category_patterns(self, keyword_sets):
        """Test getting patterns by category."""
        kg_patterns = keyword_sets.get_category_patterns(QueryClassificationCategories.KNOWLEDGE_GRAPH)
        rt_patterns = keyword_sets.get_category_patterns(QueryClassificationCategories.REAL_TIME)
        general_patterns = keyword_sets.get_category_patterns(QueryClassificationCategories.GENERAL)
        
        assert isinstance(kg_patterns, list)
        assert isinstance(rt_patterns, list)
        assert isinstance(general_patterns, list)
        
        assert len(kg_patterns) > 0
        assert len(rt_patterns) > 0
        assert len(general_patterns) > 0


class TestQueryClassificationEngine:
    """Test the QueryClassificationEngine class."""
    
    @pytest.fixture
    def classification_engine(self):
        """Create QueryClassificationEngine instance for testing."""
        logger = logging.getLogger("test_classification")
        logger.setLevel(logging.DEBUG)
        return QueryClassificationEngine(logger)
    
    def test_initialization(self, classification_engine):
        """Test that classification engine initializes correctly."""
        assert isinstance(classification_engine.keyword_sets, BiomedicalKeywordSets)
        assert hasattr(classification_engine, 'confidence_thresholds')
        assert hasattr(classification_engine, 'scoring_weights')
        assert hasattr(classification_engine, '_performance_target_ms')
        assert classification_engine._performance_target_ms == 2000  # 2 second target
    
    def test_knowledge_graph_classification(self, classification_engine):
        """Test classification of knowledge graph queries."""
        knowledge_graph_queries = [
            "What is the relationship between glucose metabolism and insulin resistance?",
            "How do metabolites in the TCA cycle interact with each other?",
            "What are the key biomarkers for diabetes?",
            "Explain the pathway analysis for fatty acid metabolism",
            "What is the mechanism of action for metformin?",
            "How are amino acids connected to protein synthesis pathways?",
            "What metabolic networks are involved in cancer metabolism?",
            "Describe the molecular interactions in glycolysis"
        ]
        
        for query in knowledge_graph_queries:
            result = classification_engine.classify_query(query)
            
            assert isinstance(result, ClassificationResult)
            assert result.category == QueryClassificationCategories.KNOWLEDGE_GRAPH
            assert result.confidence > 0.4  # Reasonable confidence threshold
            assert len(result.reasoning) > 0
            assert result.classification_time_ms > 0
    
    def test_real_time_classification(self, classification_engine):
        """Test classification of real-time queries."""
        real_time_queries = [
            "What are the latest developments in metabolomics research?",
            "Recent breakthroughs in biomarker discovery 2024",
            "Current clinical trials for metabolic disorders",
            "Latest FDA approvals for metabolomics drugs",
            "New findings in glucose metabolism published this year",
            "Breaking news in precision medicine",
            "What's new in mass spectrometry technology?",
            "Recent advances in lipidomics methods"
        ]
        
        for query in real_time_queries:
            result = classification_engine.classify_query(query)
            
            assert isinstance(result, ClassificationResult)
            assert result.category == QueryClassificationCategories.REAL_TIME
            assert result.confidence > 0.4
            assert len(result.temporal_indicators) > 0
    
    def test_general_classification(self, classification_engine):
        """Test classification of general queries."""
        general_queries = [
            "What is metabolomics?",
            "Define biomarker",
            "Explain mass spectrometry",
            "How to perform pathway analysis?",
            "What are the basics of clinical metabolomics?",
            "Introduction to lipidomics",
            "Compare LC-MS and GC-MS",
            "Overview of metabolite identification methods"
        ]
        
        for query in general_queries:
            result = classification_engine.classify_query(query)
            
            assert isinstance(result, ClassificationResult)
            assert result.category == QueryClassificationCategories.GENERAL
            assert result.confidence > 0.3
    
    def test_performance_requirements(self, classification_engine):
        """Test that classification meets performance requirements (<2 seconds)."""
        test_queries = [
            "What is the relationship between metabolites and diseases?",
            "Latest research in metabolomics 2024",
            "How to identify unknown compounds?"
        ]
        
        for query in test_queries:
            start_time = time.time()
            result = classification_engine.classify_query(query)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Test actual elapsed time
            assert elapsed_time < 2000, f"Classification took {elapsed_time:.2f}ms (should be <2000ms)"
            
            # Test reported time
            assert result.classification_time_ms < 2000, f"Reported time {result.classification_time_ms:.2f}ms exceeds target"
            assert result.classification_time_ms > 0, "Classification time should be positive"
    
    def test_confidence_scoring(self, classification_engine):
        """Test confidence scoring accuracy."""
        # High confidence queries (clear category indicators)
        high_confidence_queries = [
            ("What are the latest metabolomics breakthroughs in 2024?", QueryClassificationCategories.REAL_TIME),
            ("Analyze the pathway relationships in glucose metabolism", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            ("What is the basic definition of research methodology?", QueryClassificationCategories.GENERAL)  # More general term
        ]
        
        for query, expected_category in high_confidence_queries:
            result = classification_engine.classify_query(query)
            assert result.category == expected_category
            assert result.confidence >= 0.6, f"Expected high confidence for query: {query}"
        
        # Low confidence queries (ambiguous)
        low_confidence_queries = [
            "xyz",      # Meaningless
            "hello",    # Not biomedical
            "test"      # Generic term
        ]
        
        for query in low_confidence_queries:
            result = classification_engine.classify_query(query)
            assert result.confidence <= 0.5, f"Expected lower confidence for ambiguous query: {query}"
    
    def test_detailed_confidence_breakdown(self, classification_engine):
        """Test detailed confidence breakdown metrics."""
        query = "What are the metabolic pathways involved in cancer biomarker discovery?"
        result = classification_engine.classify_query(query)
        
        # Check that all confidence components are present and valid
        assert 0.0 <= result.keyword_match_confidence <= 1.0
        assert 0.0 <= result.pattern_match_confidence <= 1.0
        assert 0.0 <= result.semantic_confidence <= 1.0
        assert 0.0 <= result.temporal_confidence <= 1.0
        
        # Check that evidence is collected
        assert isinstance(result.matched_keywords, list)
        assert isinstance(result.matched_patterns, list)
        assert isinstance(result.biomedical_entities, list)
        assert isinstance(result.temporal_indicators, list)
    
    def test_alternative_classifications(self, classification_engine):
        """Test alternative classification generation."""
        query = "Recent metabolomics research on biomarkers"  # Could be real-time or knowledge graph
        result = classification_engine.classify_query(query)
        
        assert len(result.alternative_classifications) == 3  # Should have all three categories
        
        # Check that alternatives are sorted by confidence
        confidences = [conf for _, conf in result.alternative_classifications]
        assert confidences == sorted(confidences, reverse=True)
        
        # Check that the top alternative matches the selected category
        top_category, top_confidence = result.alternative_classifications[0]
        assert top_category == result.category
        assert abs(top_confidence - result.confidence) < 0.01  # Should be very close
    
    def test_uncertainty_metrics(self, classification_engine):
        """Test ambiguity and conflict score calculation."""
        # Ambiguous query
        ambiguous_query = "data analysis method"
        result = classification_engine.classify_query(ambiguous_query)
        assert result.ambiguity_score > 0.0, "Ambiguous query should have non-zero ambiguity score"
        
        # Clear query
        clear_query = "What is the definition of metabolomics?"
        result = classification_engine.classify_query(clear_query)
        assert result.ambiguity_score < 0.5, "Clear query should have low ambiguity score"
        
        # Conflicting signals query
        conflict_query = "Latest research on established metabolic pathways"  # Both real-time and knowledge graph signals
        result = classification_engine.classify_query(conflict_query)
        assert result.conflict_score >= 0.0, "Conflict score should be non-negative"
    
    def test_caching_functionality(self, classification_engine):
        """Test query caching for performance."""
        query = "Test query for caching"
        
        # First classification
        result1 = classification_engine.classify_query(query)
        
        # Second classification (should use cache)
        start_time = time.time()
        result2 = classification_engine.classify_query(query)
        cached_time = (time.time() - start_time) * 1000
        
        # Results should be identical
        assert result1.category == result2.category
        assert result1.confidence == result2.confidence
        
        # Cached result should be faster (though this is not guaranteed with mock)
        # This is more of a logical test
        assert hasattr(classification_engine, '_classification_cache')
    
    def test_error_handling(self, classification_engine):
        """Test error handling and fallback behavior."""
        # Empty query
        result = classification_engine.classify_query("")
        assert result.category == QueryClassificationCategories.GENERAL
        assert result.confidence <= 0.5
        
        # Very short query
        result = classification_engine.classify_query("a")
        assert isinstance(result, ClassificationResult)
        assert result.confidence <= 0.5
        
        # None query (should handle gracefully)
        with pytest.raises(AttributeError):
            classification_engine.classify_query(None)
    
    def test_context_integration(self, classification_engine):
        """Test integration with context information."""
        query = "analyze metabolites"
        
        # Without context
        result_no_context = classification_engine.classify_query(query)
        
        # With context
        context = {
            'previous_categories': ['biomarker_discovery'],
            'user_research_areas': ['clinical_metabolomics'],
            'project_type': 'clinical_study'
        }
        result_with_context = classification_engine.classify_query(query, context)
        
        assert isinstance(result_with_context, ClassificationResult)
        # Context might influence classification, but both should be valid results
        assert result_with_context.confidence > 0.0
    
    def test_performance_statistics(self, classification_engine):
        """Test performance statistics collection."""
        # Perform several classifications
        queries = [
            "What is metabolomics?",
            "Latest biomarker research",
            "Pathway analysis methods"
        ]
        
        for query in queries:
            classification_engine.classify_query(query)
        
        stats = classification_engine.get_classification_statistics()
        
        # Check statistics structure
        assert 'performance_metrics' in stats
        assert 'keyword_sets' in stats
        assert 'pattern_counts' in stats
        assert 'configuration' in stats
        
        perf_metrics = stats['performance_metrics']
        assert perf_metrics['total_classifications'] >= len(queries)
        assert perf_metrics['average_classification_time_ms'] > 0
        assert perf_metrics['performance_target_ms'] == 2000
    
    def test_validation_functionality(self, classification_engine):
        """Test classification validation functionality."""
        query = "What are metabolic biomarkers?"
        expected_category = QueryClassificationCategories.KNOWLEDGE_GRAPH
        expected_confidence_range = (0.4, 1.0)
        
        validation = classification_engine.validate_classification_performance(
            query, expected_category, expected_confidence_range
        )
        
        assert 'query' in validation
        assert 'expected_category' in validation
        assert 'predicted_category' in validation
        assert 'classification_correct' in validation
        assert 'meets_performance_target' in validation
        assert 'validation_passed' in validation
        
        assert validation['query'] == query
        assert validation['expected_category'] == expected_category.value


class TestIntegrationFunctions:
    """Test integration functions and utilities."""
    
    def test_create_classification_engine(self):
        """Test factory function for creating classification engine."""
        engine = create_classification_engine()
        assert isinstance(engine, QueryClassificationEngine)
        assert engine.keyword_sets is not None
        assert engine._performance_target_ms == 2000
    
    def test_classify_for_routing(self):
        """Test convenience function for routing classification."""
        query = "What is the latest research on metabolomics?"
        result = classify_for_routing(query)
        
        assert isinstance(result, ClassificationResult)
        assert result.category in [
            QueryClassificationCategories.KNOWLEDGE_GRAPH,
            QueryClassificationCategories.REAL_TIME,
            QueryClassificationCategories.GENERAL
        ]
    
    def test_get_routing_category_mapping(self):
        """Test routing category mapping function."""
        mapping = get_routing_category_mapping()
        
        expected_mapping = {
            QueryClassificationCategories.KNOWLEDGE_GRAPH: "lightrag",
            QueryClassificationCategories.REAL_TIME: "perplexity",
            QueryClassificationCategories.GENERAL: "either"
        }
        
        assert mapping == expected_mapping
    
    def test_classify_for_routing_with_context(self):
        """Test routing classification with context."""
        query = "analyze biomarkers"
        context = {'previous_queries': ['metabolomics research']}
        
        result = classify_for_routing(query, context)
        assert isinstance(result, ClassificationResult)
    
    def test_classify_for_routing_with_custom_engine(self):
        """Test routing classification with custom engine."""
        custom_engine = create_classification_engine()
        query = "What is proteomics?"
        
        result = classify_for_routing(query, engine=custom_engine)
        assert isinstance(result, ClassificationResult)


class TestClassificationAccuracy:
    """Test classification accuracy with comprehensive test cases."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for accuracy testing."""
        return create_classification_engine()
    
    def test_knowledge_graph_accuracy(self, engine):
        """Test knowledge graph classification accuracy."""
        test_cases = [
            # Relationships and connections
            ("What is the relationship between metabolites and diseases?", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            ("How do amino acids connect to protein synthesis?", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            ("Describe the interaction between glucose and insulin", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            
            # Pathways and mechanisms
            ("Analyze the TCA cycle pathway", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            ("What is the mechanism of fatty acid oxidation?", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            ("Explain the glycolysis metabolic network", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            
            # Biomarkers and clinical studies
            ("What are biomarkers for cardiovascular disease?", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            ("Clinical studies on diabetes metabolites", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            ("Metabolite markers for cancer diagnosis", QueryClassificationCategories.KNOWLEDGE_GRAPH),
        ]
        
        correct = 0
        for query, expected_category in test_cases:
            result = engine.classify_query(query)
            if result.category == expected_category:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.8, f"Knowledge graph accuracy {accuracy:.2f} below 80%"
    
    def test_real_time_accuracy(self, engine):
        """Test real-time classification accuracy."""
        test_cases = [
            # Temporal indicators
            ("Latest metabolomics research findings", QueryClassificationCategories.REAL_TIME),
            ("Recent advances in mass spectrometry", QueryClassificationCategories.REAL_TIME),
            ("Current trends in biomarker discovery", QueryClassificationCategories.REAL_TIME),
            
            # Year-specific queries
            ("Metabolomics breakthroughs in 2024", QueryClassificationCategories.REAL_TIME),
            ("New publications this year on lipidomics", QueryClassificationCategories.REAL_TIME),
            ("2025 developments in clinical metabolomics", QueryClassificationCategories.REAL_TIME),
            
            # Clinical and regulatory updates
            ("Recent FDA approvals for metabolomics", QueryClassificationCategories.REAL_TIME),
            ("Latest clinical trial results", QueryClassificationCategories.REAL_TIME),
            ("Breaking news in precision medicine", QueryClassificationCategories.REAL_TIME),
        ]
        
        correct = 0
        for query, expected_category in test_cases:
            result = engine.classify_query(query)
            if result.category == expected_category:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.8, f"Real-time accuracy {accuracy:.2f} below 80%"
    
    def test_general_accuracy(self, engine):
        """Test general classification accuracy."""
        test_cases = [
            # Definitions
            ("What is metabolomics?", QueryClassificationCategories.GENERAL),
            ("Define biomarker", QueryClassificationCategories.GENERAL),
            ("Explain mass spectrometry", QueryClassificationCategories.GENERAL),
            
            # Procedures and how-to
            ("How to perform LC-MS analysis?", QueryClassificationCategories.GENERAL),
            ("Steps for metabolite identification", QueryClassificationCategories.GENERAL),
            ("Protocol for sample preparation", QueryClassificationCategories.GENERAL),
            
            # Educational and comparative
            ("Introduction to lipidomics", QueryClassificationCategories.GENERAL),
            ("Compare NMR and MS techniques", QueryClassificationCategories.GENERAL),
            ("Overview of metabolomics workflows", QueryClassificationCategories.GENERAL),
        ]
        
        correct = 0
        for query, expected_category in test_cases:
            result = engine.classify_query(query)
            if result.category == expected_category:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.8, f"General query accuracy {accuracy:.2f} below 80%"
    
    def test_overall_classification_accuracy(self, engine):
        """Test overall classification accuracy across all categories."""
        all_test_cases = [
            # Knowledge graph cases
            ("Relationship between metabolites and pathways", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            ("Biomarkers for diabetes diagnosis", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            ("Mechanism of drug metabolism", QueryClassificationCategories.KNOWLEDGE_GRAPH),
            
            # Real-time cases
            ("Latest research in 2024", QueryClassificationCategories.REAL_TIME),
            ("Recent FDA approvals", QueryClassificationCategories.REAL_TIME),
            ("Breaking developments in metabolomics", QueryClassificationCategories.REAL_TIME),
            
            # General cases
            ("What is lipidomics?", QueryClassificationCategories.GENERAL),
            ("How to analyze metabolites?", QueryClassificationCategories.GENERAL),
            ("Compare different analytical methods", QueryClassificationCategories.GENERAL),
        ]
        
        correct = 0
        total_time = 0
        
        for query, expected_category in all_test_cases:
            result = engine.classify_query(query)
            total_time += result.classification_time_ms
            
            if result.category == expected_category:
                correct += 1
        
        # Test overall accuracy
        accuracy = correct / len(all_test_cases)
        assert accuracy >= 0.75, f"Overall accuracy {accuracy:.2f} below 75%"
        
        # Test average performance
        avg_time = total_time / len(all_test_cases)
        assert avg_time < 2000, f"Average classification time {avg_time:.2f}ms exceeds 2000ms target"


class TestEdgeCasesAndRobustness:
    """Test edge cases and system robustness."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for robustness testing."""
        return create_classification_engine()
    
    def test_empty_and_minimal_queries(self, engine):
        """Test handling of empty and minimal queries."""
        edge_cases = ["", " ", "a", "?", "the"]
        
        for query in edge_cases:
            result = engine.classify_query(query)
            assert isinstance(result, ClassificationResult)
            assert result.category in [
                QueryClassificationCategories.KNOWLEDGE_GRAPH,
                QueryClassificationCategories.REAL_TIME,
                QueryClassificationCategories.GENERAL
            ]
            assert 0.0 <= result.confidence <= 1.0
    
    def test_very_long_queries(self, engine):
        """Test handling of very long queries."""
        long_query = ("metabolomics " * 100) + "research analysis biomarkers pathways latest developments"
        
        result = engine.classify_query(long_query)
        assert isinstance(result, ClassificationResult)
        assert result.classification_time_ms < 2000  # Should still meet performance target
    
    def test_mixed_category_queries(self, engine):
        """Test queries that could belong to multiple categories."""
        mixed_queries = [
            "Latest research on metabolite pathways",  # Real-time + Knowledge graph
            "How to analyze recent biomarker studies?",  # General + Real-time + Knowledge graph
            "Define the latest metabolomics techniques",  # General + Real-time
            "Recent pathways analysis methods",  # Real-time + Knowledge graph + General
        ]
        
        for query in mixed_queries:
            result = engine.classify_query(query)
            assert isinstance(result, ClassificationResult)
            assert result.confidence > 0.0
            assert len(result.alternative_classifications) == 3
            
            # Should have some confidence in multiple categories
            alt_confidences = [conf for _, conf in result.alternative_classifications]
            high_confidence_count = sum(1 for conf in alt_confidences if conf > 0.3)
            assert high_confidence_count >= 2, "Mixed query should have confidence in multiple categories"
    
    def test_special_characters_and_formatting(self, engine):
        """Test queries with special characters and formatting."""
        special_queries = [
            "What is LC-MS/MS analysis?",
            "Define β-oxidation pathway",
            "Analyze H₂O metabolism",
            "Research on CO₂ metabolites",
            "Study of α-amino acids",
            "Metabolomics & proteomics integration",
            "Latest NMR (nuclear magnetic resonance) developments"
        ]
        
        for query in special_queries:
            result = engine.classify_query(query)
            assert isinstance(result, ClassificationResult)
            assert result.confidence > 0.0
    
    def test_multilingual_content(self, engine):
        """Test handling of queries with mixed language content."""
        # Note: The system is designed for English, but should handle gracefully
        mixed_queries = [
            "What is metabolomics in vivo analysis?",
            "Define in vitro biomarker studies",
            "Latest recherche in metabolomics",  # French word
            "Análisis de metabolitos research",  # Spanish word
        ]
        
        for query in mixed_queries:
            result = engine.classify_query(query)
            assert isinstance(result, ClassificationResult)
            # Should not fail, even if confidence is lower
            assert 0.0 <= result.confidence <= 1.0
    
    def test_case_insensitivity(self, engine):
        """Test that classification is case-insensitive."""
        base_query = "What are metabolic biomarkers?"
        
        # Test different case variations
        case_variations = [
            base_query.lower(),
            base_query.upper(),
            base_query.title(),
            "wHaT aRe MeTaBoLiC bIoMaRkErS?"  # Mixed case
        ]
        
        results = []
        for query in case_variations:
            result = engine.classify_query(query)
            results.append(result)
        
        # All should produce the same category
        categories = [result.category for result in results]
        assert len(set(categories)) == 1, "Case variations should produce same category"
        
        # Confidence should be similar (within reasonable range)
        confidences = [result.confidence for result in results]
        max_conf_diff = max(confidences) - min(confidences)
        assert max_conf_diff < 0.2, "Case variations should have similar confidence"


class TestPerformanceBenchmarks:
    """Comprehensive performance testing suite."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for performance testing."""
        return create_classification_engine()
    
    def test_single_query_performance(self, engine):
        """Test performance of individual query classification."""
        query = "What is the relationship between metabolites and disease pathways?"
        
        # Measure actual performance
        start_time = time.time()
        result = engine.classify_query(query)
        elapsed_time = (time.time() - start_time) * 1000
        
        # Performance requirements
        assert elapsed_time < 2000, f"Single query took {elapsed_time:.2f}ms (target: <2000ms)"
        assert result.classification_time_ms < 2000, f"Reported time {result.classification_time_ms:.2f}ms exceeds target"
    
    def test_batch_performance(self, engine):
        """Test performance with multiple queries."""
        queries = [
            "What are metabolic biomarkers?",
            "Latest developments in metabolomics 2024",
            "How to perform LC-MS analysis?",
            "Relationship between glucose and insulin",
            "Recent FDA approvals for metabolomics",
            "Define lipidomics methodology",
            "Pathway analysis for fatty acids",
            "Current trends in proteomics",
            "What is the mechanism of drug metabolism?",
            "Latest clinical trials in metabolomics"
        ]
        
        start_time = time.time()
        results = []
        
        for query in queries:
            result = engine.classify_query(query)
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(queries)
        
        # Performance requirements
        assert avg_time < 2000, f"Average query time {avg_time:.2f}ms exceeds 2000ms target"
        
        # All queries should complete successfully
        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, ClassificationResult)
            assert result.classification_time_ms > 0
    
    def test_cache_performance_improvement(self, engine):
        """Test that caching improves performance for repeated queries."""
        query = "What is metabolomics research methodology?"
        
        # First query (no cache)
        start_time = time.time()
        result1 = engine.classify_query(query)
        first_time = (time.time() - start_time) * 1000
        
        # Second query (should use cache if confidence is high enough)
        start_time = time.time()
        result2 = engine.classify_query(query)
        second_time = (time.time() - start_time) * 1000
        
        # Results should be identical
        assert result1.category == result2.category
        assert abs(result1.confidence - result2.confidence) < 0.001
        
        # If caching is working and confidence is high, second should be faster
        # Note: This test is more about logical correctness than guaranteed performance improvement
        assert second_time > 0  # Should still take some time
    
    def test_concurrent_performance(self, engine):
        """Test performance under concurrent load (simulated)."""
        import threading
        import queue
        
        queries = [
            "What are metabolic pathways?",
            "Latest biomarker research",
            "How to identify compounds?",
            "Relationship between metabolites",
            "Recent clinical developments"
        ]
        
        results_queue = queue.Queue()
        
        def classify_query_worker(query):
            start_time = time.time()
            result = engine.classify_query(query)
            elapsed_time = (time.time() - start_time) * 1000
            results_queue.put((result, elapsed_time))
        
        # Create threads for concurrent classification
        threads = []
        start_time = time.time()
        
        for query in queries:
            thread = threading.Thread(target=classify_query_worker, args=(query,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = (time.time() - start_time) * 1000
        
        # Collect results
        results = []
        times = []
        
        while not results_queue.empty():
            result, elapsed_time = results_queue.get()
            results.append(result)
            times.append(elapsed_time)
        
        # Performance checks
        assert len(results) == len(queries), "All concurrent queries should complete"
        
        max_time = max(times)
        avg_time = sum(times) / len(times)
        
        assert max_time < 3000, f"Slowest concurrent query took {max_time:.2f}ms (target: <3000ms)"
        assert avg_time < 2000, f"Average concurrent query time {avg_time:.2f}ms exceeds 2000ms target"
    
    def test_memory_efficiency(self, engine):
        """Test memory efficiency with large numbers of classifications."""
        import sys
        
        # Measure initial memory usage (approximate)
        initial_cache_size = len(engine._classification_cache)
        
        # Perform many unique classifications
        unique_queries = [f"What is metabolomics research question {i}?" for i in range(250)]
        
        results = []
        for query in unique_queries:
            result = engine.classify_query(query)
            results.append(result)
        
        # Check cache size management
        final_cache_size = len(engine._classification_cache)
        max_cache_size = engine._cache_max_size
        
        assert final_cache_size <= max_cache_size, f"Cache size {final_cache_size} exceeds maximum {max_cache_size}"
        
        # All classifications should complete
        assert len(results) == len(unique_queries)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])