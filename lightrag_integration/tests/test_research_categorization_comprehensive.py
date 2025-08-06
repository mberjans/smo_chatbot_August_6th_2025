#!/usr/bin/env python3
"""
Comprehensive test suite for Research Categorization System.

This test suite provides complete coverage of the research categorization components including:
- CategoryPrediction data model and confidence scoring
- CategoryMetrics tracking and accuracy calculation
- QueryAnalyzer pattern matching and feature extraction
- ResearchCategorizer main categorization logic
- Confidence thresholds and evidence weighting
- Context-aware categorization and user feedback integration
- Performance under various query types and edge cases

Author: Claude Code (Anthropic)
Created: August 6, 2025
"""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional

# Test imports
from lightrag_integration.research_categorizer import (
    ResearchCategorizer,
    QueryAnalyzer,
    CategoryPrediction,
    CategoryMetrics
)
from lightrag_integration.cost_persistence import ResearchCategory


class TestCategoryPrediction:
    """Comprehensive tests for CategoryPrediction data model."""
    
    def test_category_prediction_basic_creation(self):
        """Test basic CategoryPrediction creation."""
        prediction = CategoryPrediction(
            category=ResearchCategory.METABOLITE_IDENTIFICATION,
            confidence=0.85,
            evidence=["keyword:metabolite", "pattern:mass spectrum"]
        )
        
        assert prediction.category == ResearchCategory.METABOLITE_IDENTIFICATION
        assert prediction.confidence == 0.85
        assert len(prediction.evidence) == 2
        assert "keyword:metabolite" in prediction.evidence
        assert "pattern:mass spectrum" in prediction.evidence
    
    def test_category_prediction_with_optional_fields(self):
        """Test CategoryPrediction with optional fields."""
        metadata = {
            "confidence_breakdown": {"keyword_score": 0.6, "pattern_score": 0.3},
            "alternative_categories": ["pathway_analysis", "biomarker_discovery"],
            "query_complexity": "high"
        }
        
        prediction = CategoryPrediction(
            category=ResearchCategory.BIOMARKER_DISCOVERY,
            confidence=0.92,
            evidence=["keyword:biomarker", "keyword:diagnostic", "context_match"],
            subject_area="clinical",
            query_type="scientific_inquiry",
            metadata=metadata
        )
        
        assert prediction.subject_area == "clinical"
        assert prediction.query_type == "scientific_inquiry"
        assert prediction.metadata == metadata
        assert prediction.metadata["query_complexity"] == "high"
    
    def test_category_prediction_serialization(self):
        """Test CategoryPrediction serialization to dictionary."""
        prediction = CategoryPrediction(
            category=ResearchCategory.PATHWAY_ANALYSIS,
            confidence=0.78,
            evidence=["pattern:kegg", "keyword:metabolism", "technical_terms"],
            subject_area="biochemistry",
            query_type="analysis",
            metadata={"processing_time": 0.05, "model_version": "v2.1"}
        )
        
        prediction_dict = prediction.to_dict()
        
        assert isinstance(prediction_dict, dict)
        assert prediction_dict['category'] == ResearchCategory.PATHWAY_ANALYSIS.value
        assert prediction_dict['confidence'] == 0.78
        assert prediction_dict['evidence'] == ["pattern:kegg", "keyword:metabolism", "technical_terms"]
        assert prediction_dict['subject_area'] == "biochemistry"
        assert prediction_dict['query_type'] == "analysis"
        assert prediction_dict['metadata']['processing_time'] == 0.05
    
    def test_category_prediction_edge_cases(self):
        """Test CategoryPrediction with edge case values."""
        # Minimum confidence
        prediction_min = CategoryPrediction(
            category=ResearchCategory.GENERAL_QUERY,
            confidence=0.0,
            evidence=[]
        )
        assert prediction_min.confidence == 0.0
        assert len(prediction_min.evidence) == 0
        
        # Maximum confidence
        prediction_max = CategoryPrediction(
            category=ResearchCategory.DRUG_DISCOVERY,
            confidence=1.0,
            evidence=["high_confidence_match"]
        )
        assert prediction_max.confidence == 1.0
        
        # Empty metadata
        prediction_empty_meta = CategoryPrediction(
            category=ResearchCategory.CLINICAL_DIAGNOSIS,
            confidence=0.5,
            evidence=["basic_match"],
            metadata={}
        )
        assert prediction_empty_meta.metadata == {}


class TestCategoryMetrics:
    """Comprehensive tests for CategoryMetrics tracking."""
    
    def test_category_metrics_initialization(self):
        """Test CategoryMetrics initialization."""
        metrics = CategoryMetrics()
        
        assert metrics.total_predictions == 0
        assert metrics.correct_predictions == 0
        assert metrics.average_confidence == 0.0
        assert len(metrics.category_counts) == 0
        assert len(metrics.confidence_distribution) == 0
        assert metrics.accuracy == 0.0
    
    def test_category_metrics_update(self):
        """Test CategoryMetrics update functionality."""
        metrics = CategoryMetrics()
        
        # Add predictions
        prediction1 = CategoryPrediction(
            category=ResearchCategory.METABOLITE_IDENTIFICATION,
            confidence=0.85,
            evidence=["test_evidence"]
        )
        
        prediction2 = CategoryPrediction(
            category=ResearchCategory.PATHWAY_ANALYSIS,
            confidence=0.72,
            evidence=["test_evidence"]
        )
        
        prediction3 = CategoryPrediction(
            category=ResearchCategory.METABOLITE_IDENTIFICATION,
            confidence=0.91,
            evidence=["test_evidence"]
        )
        
        # Update with feedback
        metrics.update(prediction1, is_correct=True)
        metrics.update(prediction2, is_correct=False)
        metrics.update(prediction3, is_correct=True)
        
        assert metrics.total_predictions == 3
        assert metrics.correct_predictions == 2
        assert metrics.accuracy == pytest.approx(2/3, abs=1e-10)
        
        # Check category counts
        assert metrics.category_counts[ResearchCategory.METABOLITE_IDENTIFICATION.value] == 2
        assert metrics.category_counts[ResearchCategory.PATHWAY_ANALYSIS.value] == 1
        
        # Check confidence distribution
        assert len(metrics.confidence_distribution) == 3
        assert 0.85 in metrics.confidence_distribution
        assert 0.72 in metrics.confidence_distribution
        assert 0.91 in metrics.confidence_distribution
        
        # Check average confidence
        expected_avg = (0.85 + 0.72 + 0.91) / 3
        assert metrics.average_confidence == pytest.approx(expected_avg, abs=1e-10)
    
    def test_category_metrics_update_without_feedback(self):
        """Test CategoryMetrics update without correctness feedback."""
        metrics = CategoryMetrics()
        
        prediction = CategoryPrediction(
            category=ResearchCategory.BIOMARKER_DISCOVERY,
            confidence=0.66,
            evidence=["test_evidence"]
        )
        
        # Update without feedback
        metrics.update(prediction)
        
        assert metrics.total_predictions == 1
        assert metrics.correct_predictions == 0  # No feedback provided
        assert metrics.accuracy == 0.0  # Can't calculate without feedback
        assert metrics.category_counts[ResearchCategory.BIOMARKER_DISCOVERY.value] == 1
    
    def test_category_metrics_accuracy_calculation(self):
        """Test accuracy calculation with various feedback scenarios."""
        metrics = CategoryMetrics()
        
        # All correct predictions
        for i in range(5):
            prediction = CategoryPrediction(
                category=ResearchCategory.GENERAL_QUERY,
                confidence=0.8,
                evidence=["test"]
            )
            metrics.update(prediction, is_correct=True)
        
        assert metrics.accuracy == 1.0
        
        # All incorrect predictions
        for i in range(3):
            prediction = CategoryPrediction(
                category=ResearchCategory.GENERAL_QUERY,
                confidence=0.7,
                evidence=["test"]
            )
            metrics.update(prediction, is_correct=False)
        
        assert metrics.accuracy == 5/8  # 5 correct out of 8 total


class TestQueryAnalyzer:
    """Comprehensive tests for QueryAnalyzer functionality."""
    
    @pytest.fixture
    def query_analyzer(self):
        """Create a QueryAnalyzer instance for testing."""
        return QueryAnalyzer()
    
    def test_query_analyzer_initialization(self, query_analyzer):
        """Test QueryAnalyzer initialization."""
        assert hasattr(query_analyzer, 'category_patterns')
        assert hasattr(query_analyzer, 'query_type_patterns')
        assert hasattr(query_analyzer, 'subject_area_patterns')
        
        # Verify patterns are loaded for all categories
        expected_categories = [
            ResearchCategory.METABOLITE_IDENTIFICATION,
            ResearchCategory.PATHWAY_ANALYSIS,
            ResearchCategory.BIOMARKER_DISCOVERY,
            ResearchCategory.DRUG_DISCOVERY,
            ResearchCategory.CLINICAL_DIAGNOSIS
        ]
        
        for category in expected_categories:
            assert category in query_analyzer.category_patterns
            assert 'keywords' in query_analyzer.category_patterns[category]
            assert 'patterns' in query_analyzer.category_patterns[category]
    
    def test_metabolite_identification_analysis(self, query_analyzer):
        """Test analysis of metabolite identification queries."""
        test_queries = [
            "What is the molecular structure of this unknown metabolite?",
            "Can you help me identify this compound with mass spectrum data?",
            "I need to determine the chemical formula of this peak",
            "Mass spectrometry fragmentation pattern analysis needed",
            "NMR spectroscopy data interpretation for compound identification"
        ]
        
        for query in test_queries:
            analysis = query_analyzer.analyze_query(query)
            
            assert ResearchCategory.METABOLITE_IDENTIFICATION in analysis['matched_keywords']
            assert len(analysis['matched_keywords'][ResearchCategory.METABOLITE_IDENTIFICATION]) > 0
            
            # Check for expected keywords
            metabolite_keywords = analysis['matched_keywords'][ResearchCategory.METABOLITE_IDENTIFICATION]
            expected_keywords = ['metabolite', 'compound', 'identify', 'mass spectrum', 'molecular', 'nmr']
            assert any(keyword in ' '.join(metabolite_keywords) for keyword in expected_keywords)
    
    def test_pathway_analysis_queries(self, query_analyzer):
        """Test analysis of pathway analysis queries."""
        test_queries = [
            "How does the glycolysis pathway work?",
            "KEGG pathway enrichment analysis for my metabolite list",
            "What are the key enzymes in fatty acid metabolism?",
            "Metabolic network reconstruction from omics data",
            "Systems biology approach to study metabolic regulation"
        ]
        
        for query in test_queries:
            analysis = query_analyzer.analyze_query(query)
            
            # Should detect pathway-related terms
            if ResearchCategory.PATHWAY_ANALYSIS in analysis['matched_keywords']:
                pathway_keywords = analysis['matched_keywords'][ResearchCategory.PATHWAY_ANALYSIS]
                pathway_terms = ['pathway', 'kegg', 'metabolism', 'network', 'systems biology']
                assert any(term in ' '.join(pathway_keywords) for term in pathway_terms)
    
    def test_biomarker_discovery_queries(self, query_analyzer):
        """Test analysis of biomarker discovery queries."""
        test_queries = [
            "Can you help me find biomarkers for diabetes?",
            "Diagnostic metabolites for cancer detection",
            "Prognostic signature discovery using metabolomics",
            "What are the best predictive markers for this disease?",
            "Personalized medicine approach using metabolite profiles"
        ]
        
        for query in test_queries:
            analysis = query_analyzer.analyze_query(query)
            
            if ResearchCategory.BIOMARKER_DISCOVERY in analysis['matched_keywords']:
                biomarker_keywords = analysis['matched_keywords'][ResearchCategory.BIOMARKER_DISCOVERY]
                biomarker_terms = ['biomarker', 'diagnostic', 'prognostic', 'predictive', 'personalized medicine']
                assert any(term in ' '.join(biomarker_keywords) for term in biomarker_terms)
    
    def test_query_type_detection(self, query_analyzer):
        """Test query type detection functionality."""
        test_cases = [
            ("What is metabolomics?", "question"),
            ("How do I analyze LC-MS data?", "question"),
            ("Find biomarkers for cancer", "search"),
            ("Search for pathway databases", "search"),
            ("Analyze this metabolite dataset", "analysis"),
            ("Calculate the molecular weight", "analysis"),
            ("Compare these two metabolic profiles", "comparison"),
            ("Explain the citric acid cycle", "explanation"),
            ("Tell me about mass spectrometry", "explanation"),
            ("How to perform pathway enrichment", "procedure")
        ]
        
        for query, expected_type in test_cases:
            analysis = query_analyzer.analyze_query(query)
            assert analysis['query_type'] == expected_type
    
    def test_subject_area_detection(self, query_analyzer):
        """Test subject area detection functionality."""
        test_cases = [
            ("Lipid metabolism in adipose tissue", "lipidomics"),
            ("Protein expression levels in cancer", "proteomics"),
            ("Gene regulation in metabolic pathways", "genomics"),
            ("Clinical diagnosis of diabetes", "clinical"),
            ("Plant metabolite profiling", "plant"),
            ("Bacterial fermentation products", "microbial"),
            ("Environmental pollutant analysis", "environmental"),
            ("Food metabolomics study", "food")
        ]
        
        for query, expected_area in test_cases:
            analysis = query_analyzer.analyze_query(query)
            assert analysis['subject_area'] == expected_area
    
    def test_technical_terms_detection(self, query_analyzer):
        """Test technical terms detection."""
        technical_queries = [
            "LC-MS metabolomics analysis of serum samples",
            "NMR-based metabolite identification using HMDB",
            "Mass spectrometry data processing pipeline",
            "PCA analysis of metabolomic datasets"
        ]
        
        non_technical_queries = [
            "What is the best diet for health?",
            "How do I lose weight?",
            "General information about nutrition"
        ]
        
        for query in technical_queries:
            analysis = query_analyzer.analyze_query(query)
            assert analysis['has_technical_terms'] is True
        
        for query in non_technical_queries:
            analysis = query_analyzer.analyze_query(query)
            assert analysis['has_technical_terms'] is False
    
    def test_complex_query_analysis(self, query_analyzer):
        """Test analysis of complex, multi-faceted queries."""
        complex_query = """
        I need help with LC-MS metabolomics data analysis for biomarker discovery 
        in a clinical study of Type 2 diabetes patients. Specifically, I want to 
        identify metabolites that are significantly different between cases and 
        controls, perform pathway enrichment analysis using KEGG database, and 
        validate the findings using machine learning approaches for diagnostic 
        marker development.
        """
        
        analysis = query_analyzer.analyze_query(complex_query)
        
        # Should detect multiple categories
        assert len(analysis['matched_keywords']) >= 3
        
        # Should detect technical terms
        assert analysis['has_technical_terms'] is True
        
        # Should detect clinical subject area
        assert analysis['subject_area'] == "clinical"
        
        # Should have reasonable word count
        assert analysis['word_count'] > 20
        
        # Should detect multiple research categories
        expected_categories = [
            ResearchCategory.BIOMARKER_DISCOVERY,
            ResearchCategory.PATHWAY_ANALYSIS,
            ResearchCategory.CLINICAL_DIAGNOSIS,
            ResearchCategory.DATA_PREPROCESSING,
            ResearchCategory.STATISTICAL_ANALYSIS
        ]
        
        found_categories = list(analysis['matched_keywords'].keys())
        overlap = set(expected_categories) & set(found_categories)
        assert len(overlap) >= 2  # Should find at least 2 relevant categories
    
    def test_edge_case_queries(self, query_analyzer):
        """Test analysis of edge case queries."""
        edge_cases = [
            "",  # Empty query
            "a",  # Single character
            "The the the the",  # Repetitive words
            "12345 67890",  # Numbers only
            "!@#$%^&*()",  # Special characters only
            "x" * 1000  # Very long query
        ]
        
        for query in edge_cases:
            analysis = query_analyzer.analyze_query(query)
            
            # Should not crash and return valid structure
            assert 'original_query' in analysis
            assert 'matched_keywords' in analysis
            assert 'matched_patterns' in analysis
            assert 'query_length' in analysis
            assert 'word_count' in analysis
            assert analysis['original_query'] == query


class TestResearchCategorizer:
    """Comprehensive tests for ResearchCategorizer main functionality."""
    
    @pytest.fixture
    def research_categorizer(self):
        """Create a ResearchCategorizer instance for testing."""
        return ResearchCategorizer()
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)
    
    def test_research_categorizer_initialization(self, research_categorizer):
        """Test ResearchCategorizer initialization."""
        assert hasattr(research_categorizer, 'query_analyzer')
        assert hasattr(research_categorizer, 'metrics')
        assert hasattr(research_categorizer, 'confidence_thresholds')
        assert hasattr(research_categorizer, 'evidence_weights')
        
        # Verify confidence thresholds
        assert research_categorizer.confidence_thresholds['high'] == 0.8
        assert research_categorizer.confidence_thresholds['medium'] == 0.6
        assert research_categorizer.confidence_thresholds['low'] == 0.4
    
    def test_categorize_metabolite_identification(self, research_categorizer):
        """Test categorization of metabolite identification queries."""
        test_queries = [
            "What is the structure of this unknown metabolite with m/z 180.06?",
            "Help me identify this compound using MS/MS fragmentation data",
            "I need to determine the molecular formula from exact mass",
            "Unknown peak identification in LC-MS experiment"
        ]
        
        for query in test_queries:
            prediction = research_categorizer.categorize_query(query)
            
            assert isinstance(prediction, CategoryPrediction)
            assert prediction.category == ResearchCategory.METABOLITE_IDENTIFICATION
            assert prediction.confidence > 0.5  # Should be reasonably confident
            assert len(prediction.evidence) > 0
            
            # Should have metabolite-related evidence
            evidence_text = ' '.join(prediction.evidence)
            metabolite_terms = ['metabolite', 'compound', 'identify', 'structure', 'mass']
            assert any(term in evidence_text for term in metabolite_terms)
    
    def test_categorize_pathway_analysis(self, research_categorizer):
        """Test categorization of pathway analysis queries."""
        test_queries = [
            "Perform KEGG pathway enrichment analysis on my metabolite list",
            "How does the TCA cycle regulate cellular metabolism?",
            "Systems biology approach to metabolic network reconstruction",
            "Enzyme kinetics in glycolysis pathway"
        ]
        
        for query in test_queries:
            prediction = research_categorizer.categorize_query(query)
            
            assert prediction.category == ResearchCategory.PATHWAY_ANALYSIS
            assert prediction.confidence > 0.4
            
            # Should have pathway-related evidence
            evidence_text = ' '.join(prediction.evidence)
            pathway_terms = ['kegg', 'pathway', 'metabolism', 'network', 'enzyme']
            assert any(term in evidence_text for term in pathway_terms)
    
    def test_categorize_biomarker_discovery(self, research_categorizer):
        """Test categorization of biomarker discovery queries."""
        test_queries = [
            "Find diagnostic biomarkers for early cancer detection",
            "Prognostic metabolite signature for disease outcome",
            "Predictive markers for drug response in precision medicine",
            "Therapeutic targets identification using metabolomics"
        ]
        
        for query in test_queries:
            prediction = research_categorizer.categorize_query(query)
            
            assert prediction.category == ResearchCategory.BIOMARKER_DISCOVERY
            assert prediction.confidence > 0.3
            
            # Should have biomarker-related evidence
            evidence_text = ' '.join(prediction.evidence)
            biomarker_terms = ['biomarker', 'diagnostic', 'prognostic', 'predictive', 'therapeutic']
            assert any(term in evidence_text for term in biomarker_terms)
    
    def test_categorize_with_context(self, research_categorizer):
        """Test categorization with context information."""
        query = "Analyze the metabolite profiles"
        
        # Without context - should be general
        prediction_no_context = research_categorizer.categorize_query(query)
        
        # With context - should influence categorization
        context = {
            'previous_categories': ['biomarker_discovery'],
            'user_research_areas': ['biomarker_discovery', 'clinical_diagnosis'],
            'project_type': 'clinical_study'
        }
        
        prediction_with_context = research_categorizer.categorize_query(query, context)
        
        # Context should potentially influence the result
        assert isinstance(prediction_with_context, CategoryPrediction)
        # May increase confidence or change category based on context
    
    def test_confidence_scoring(self, research_categorizer):
        """Test confidence scoring accuracy."""
        # High confidence query
        high_conf_query = "LC-MS metabolite identification using MS/MS fragmentation patterns and HMDB database search for unknown compound structure determination"
        prediction_high = research_categorizer.categorize_query(high_conf_query)
        
        # Low confidence query
        low_conf_query = "Help me with data analysis"
        prediction_low = research_categorizer.categorize_query(low_conf_query)
        
        # High confidence query should have higher confidence
        assert prediction_high.confidence > prediction_low.confidence
        
        # Check confidence levels
        assert prediction_high.metadata['confidence_level'] in ['high', 'medium']
        assert prediction_low.metadata['confidence_level'] in ['low', 'very_low']
    
    def test_default_categorization(self, research_categorizer):
        """Test default categorization for unclear queries."""
        ambiguous_queries = [
            "Hello",
            "What should I do?",
            "General question about science",
            "Random text with no specific meaning"
        ]
        
        for query in ambiguous_queries:
            prediction = research_categorizer.categorize_query(query)
            
            assert prediction.category == ResearchCategory.GENERAL_QUERY
            assert prediction.confidence <= 0.5  # Should have low confidence
            assert 'no_specific_patterns_found' in prediction.evidence or len(prediction.evidence) == 0
    
    def test_evidence_collection(self, research_categorizer):
        """Test evidence collection and weighting."""
        query = "Metabolite identification using mass spectrometry and NMR spectroscopy for biomarker discovery"
        prediction = research_categorizer.categorize_query(query)
        
        assert len(prediction.evidence) > 0
        
        # Should have different types of evidence
        evidence_types = set()
        for evidence in prediction.evidence:
            if ':' in evidence:
                evidence_type = evidence.split(':')[0]
                evidence_types.add(evidence_type)
        
        # Should have keyword and/or pattern matches
        expected_types = {'keyword', 'pattern', 'technical_terms', 'subject_area'}
        overlap = evidence_types & expected_types
        assert len(overlap) > 0
    
    def test_metadata_information(self, research_categorizer):
        """Test metadata information in predictions."""
        query = "Comprehensive metabolomic analysis for pathway enrichment and biomarker discovery"
        prediction = research_categorizer.categorize_query(query)
        
        assert prediction.metadata is not None
        assert 'all_scores' in prediction.metadata
        assert 'analysis_details' in prediction.metadata
        assert 'confidence_level' in prediction.metadata
        
        # All scores should contain other categories considered
        all_scores = prediction.metadata['all_scores']
        assert len(all_scores) >= 1
        
        # Analysis details should contain query analysis
        analysis_details = prediction.metadata['analysis_details']
        assert 'matched_keywords' in analysis_details
        assert 'word_count' in analysis_details
    
    def test_user_feedback_integration(self, research_categorizer):
        """Test user feedback integration."""
        query = "Find metabolite biomarkers"
        
        # Make prediction
        prediction = research_categorizer.categorize_query(query)
        original_accuracy = research_categorizer.metrics.accuracy
        
        # Provide feedback
        research_categorizer.update_from_feedback(
            query_text=query,
            predicted_category=prediction.category,
            actual_category=ResearchCategory.BIOMARKER_DISCOVERY,
            confidence=prediction.confidence
        )
        
        # Metrics should be updated
        assert research_categorizer.metrics.total_predictions > 0
    
    def test_category_statistics(self, research_categorizer):
        """Test category statistics generation."""
        # Make several predictions
        test_queries = [
            "Identify this metabolite",
            "Analyze metabolic pathways",
            "Find disease biomarkers",
            "Clinical diagnosis help",
            "Drug discovery research"
        ]
        
        for query in test_queries:
            research_categorizer.categorize_query(query)
        
        stats = research_categorizer.get_category_statistics()
        
        assert 'total_predictions' in stats
        assert 'accuracy' in stats
        assert 'average_confidence' in stats
        assert 'category_distribution' in stats
        assert 'confidence_distribution' in stats
        assert 'thresholds' in stats
        
        assert stats['total_predictions'] == len(test_queries)
        assert isinstance(stats['category_distribution'], dict)
        assert isinstance(stats['confidence_distribution'], dict)
    
    def test_reset_metrics(self, research_categorizer):
        """Test metrics reset functionality."""
        # Make some predictions
        for i in range(3):
            research_categorizer.categorize_query(f"Test query {i}")
        
        assert research_categorizer.metrics.total_predictions == 3
        
        # Reset metrics
        research_categorizer.reset_metrics()
        
        assert research_categorizer.metrics.total_predictions == 0
        assert research_categorizer.metrics.correct_predictions == 0
        assert len(research_categorizer.metrics.category_counts) == 0
        assert len(research_categorizer.metrics.confidence_distribution) == 0
    
    def test_concurrent_categorization(self, research_categorizer):
        """Test thread safety of categorization."""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        test_queries = [
            "Metabolite identification query",
            "Pathway analysis request",
            "Biomarker discovery study",
            "Clinical diagnosis help",
            "Drug discovery research"
        ] * 10  # 50 queries total
        
        results = []
        
        def categorize_worker(query):
            return research_categorizer.categorize_query(query)
        
        # Run concurrent categorizations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(categorize_worker, query) for query in test_queries]
            for future in futures:
                results.append(future.result())
        
        # Verify all predictions completed
        assert len(results) == len(test_queries)
        
        # Verify all results are valid CategoryPrediction objects
        for result in results:
            assert isinstance(result, CategoryPrediction)
            assert hasattr(result, 'category')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'evidence')
        
        # Verify metrics were updated correctly
        assert research_categorizer.metrics.total_predictions == len(test_queries)
    
    def test_performance_with_long_queries(self, research_categorizer):
        """Test performance with very long queries."""
        # Create very long query
        long_query = """
        This is an extremely long query about metabolomics research that includes
        many different aspects of the field including metabolite identification,
        pathway analysis, biomarker discovery, drug discovery, clinical diagnosis,
        data preprocessing, statistical analysis, literature search, knowledge
        extraction, database integration, and experimental validation. The query
        contains multiple technical terms such as mass spectrometry, NMR spectroscopy,
        LC-MS, GC-MS, KEGG, HMDB, PubChem, ChEBI, MetLin, MassBank, and many others.
        It also discusses various research categories and methodologies used in
        metabolomics studies for understanding biological systems and disease
        mechanisms. The comprehensive nature of this query should test the
        categorization system's ability to handle complex, multi-faceted research
        questions that span multiple domains within metabolomics research.
        """ * 10  # Make it even longer
        
        # Should handle long query without issues
        prediction = research_categorizer.categorize_query(long_query)
        
        assert isinstance(prediction, CategoryPrediction)
        assert prediction.confidence > 0  # Should find some relevant patterns
        assert len(prediction.evidence) > 0
        
        # Should detect technical terms
        analysis_details = prediction.metadata['analysis_details']
        assert analysis_details['has_technical_terms'] is True
        assert analysis_details['word_count'] > 100


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])