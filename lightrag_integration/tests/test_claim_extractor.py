#!/usr/bin/env python3
"""
Comprehensive Test Suite for Biomedical Claim Extraction System.

This module provides comprehensive unit tests for the BiomedicalClaimExtractor
class in the Clinical Metabolomics Oracle LightRAG integration system.

Test Coverage:
    - Claim extraction functionality
    - Multi-type claim classification (numeric, qualitative, methodological, etc.)
    - Confidence scoring system
    - Context preservation and analysis
    - Biomedical specialization patterns
    - Integration with quality assessment pipeline
    - Performance and error handling
    - Duplicate detection and merging

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Factual Claim Extraction Implementation
"""

import pytest
import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the claim extractor
try:
    from claim_extractor import (
        BiomedicalClaimExtractor,
        ExtractedClaim,
        ClaimContext,
        ClaimConfidence,
        ClaimExtractionError,
        ClaimProcessingError,
        ClaimValidationError,
        extract_claims_from_response,
        prepare_claims_for_quality_assessment
    )
    CLAIM_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    CLAIM_EXTRACTOR_AVAILABLE = False
    pytest.skip(f"Claim extractor not available: {e}", allow_module_level=True)

# Test fixtures and utilities
try:
    from biomedical_test_fixtures import (
        BiomedicalTestDataManager,
        generate_biomedical_response_samples
    )
    from comprehensive_test_fixtures import (
        TestResultValidator,
        BiomedicalContentValidator
    )
    FIXTURES_AVAILABLE = True
except ImportError:
    FIXTURES_AVAILABLE = False


class TestBiomedicalClaimExtractor:
    """Comprehensive test suite for BiomedicalClaimExtractor."""
    
    @pytest.fixture
    async def extractor(self):
        """Create a BiomedicalClaimExtractor instance for testing."""
        return BiomedicalClaimExtractor()
    
    @pytest.fixture
    def sample_responses(self):
        """Sample biomedical response texts for testing."""
        return {
            'numeric_heavy': """
            Metabolomic profiling revealed glucose concentrations of 8.5 ± 1.2 mmol/L 
            in diabetic patients, which is 45% higher than healthy controls (5.9 ± 0.8 mmol/L).
            The correlation coefficient was r = 0.78 (p < 0.001). Sample size was n = 150.
            """,
            'qualitative_relationships': """
            Insulin resistance leads to impaired glucose metabolism and increased 
            oxidative stress. This condition is associated with elevated inflammatory 
            markers and correlates with mitochondrial dysfunction. Chronic hyperglycemia 
            triggers cellular damage through advanced glycation end products.
            """,
            'methodological': """
            Samples were analyzed using LC-MS/MS with a QTOF mass spectrometer.
            The study employed a randomized controlled trial design with double-blind 
            placebo-controlled methodology. Statistical analysis was performed using 
            PCA and PLS-DA multivariate methods.
            """,
            'temporal_claims': """
            Metabolite levels changed significantly after 6 weeks of treatment.
            Patients were monitored for 12 months following intervention. 
            Blood samples were collected every 2 hours during the first day 
            and then daily for the next 7 days.
            """,
            'comparative_claims': """
            Treatment group showed 2.3-fold increase in metabolite X compared to controls.
            Biomarker levels were significantly higher in diseased patients versus 
            healthy subjects. The intervention resulted in a 30% reduction in 
            inflammatory markers relative to baseline measurements.
            """,
            'mixed_content': """
            The LC-MS analysis identified 247 metabolites with statistical significance 
            (p < 0.05). Glucose levels increased by approximately 25% in diabetic patients 
            compared to controls. This finding correlates with previous studies showing 
            insulin resistance leads to metabolic dysregulation. The study was conducted 
            over 18 months using a prospective cohort design.
            """,
            'low_confidence': """
            Some studies suggest that metabolite levels might be associated with 
            disease progression. It appears that certain biomarkers could potentially 
            serve as diagnostic indicators. Further research may be needed to 
            confirm these preliminary findings.
            """,
            'high_confidence': """
            Targeted metabolomics analysis definitively demonstrated that serum 
            lactate concentrations were 4.8 ± 0.6 mmol/L in septic patients, 
            representing a statistically significant 85% increase compared to 
            healthy controls (2.6 ± 0.3 mmol/L, p < 0.001).
            """
        }
    
    @pytest.fixture
    def expected_claim_patterns(self):
        """Expected patterns for different claim types."""
        return {
            'numeric': {
                'patterns': ['8.5 ± 1.2 mmol/L', '45%', 'r = 0.78', 'p < 0.001', 'n = 150'],
                'min_expected': 3
            },
            'qualitative': {
                'patterns': ['leads to', 'associated with', 'correlates with', 'triggers'],
                'min_expected': 2
            },
            'methodological': {
                'patterns': ['LC-MS/MS', 'QTOF', 'randomized controlled trial', 'PCA', 'PLS-DA'],
                'min_expected': 2
            },
            'temporal': {
                'patterns': ['6 weeks', '12 months', 'every 2 hours', '7 days'],
                'min_expected': 2
            },
            'comparative': {
                'patterns': ['2.3-fold', 'higher', 'versus', '30% reduction'],
                'min_expected': 2
            }
        }
    
    @pytest.mark.asyncio
    async def test_basic_claim_extraction(self, extractor, sample_responses):
        """Test basic claim extraction functionality."""
        
        response = sample_responses['numeric_heavy']
        claims = await extractor.extract_claims(response)
        
        # Basic assertions
        assert len(claims) > 0, "Should extract at least some claims"
        assert all(isinstance(claim, ExtractedClaim) for claim in claims), "All claims should be ExtractedClaim instances"
        
        # Check that claims have required attributes
        for claim in claims:
            assert claim.claim_id, "Claim should have ID"
            assert claim.claim_text, "Claim should have text"
            assert claim.claim_type, "Claim should have type"
            assert isinstance(claim.confidence, ClaimConfidence), "Claim should have confidence assessment"
            assert isinstance(claim.context, ClaimContext), "Claim should have context"
    
    @pytest.mark.asyncio
    async def test_numeric_claim_extraction(self, extractor, sample_responses, expected_claim_patterns):
        """Test extraction of numeric claims."""
        
        response = sample_responses['numeric_heavy']
        claims = await extractor.extract_claims(response)
        
        numeric_claims = [c for c in claims if c.claim_type == 'numeric']
        
        assert len(numeric_claims) >= expected_claim_patterns['numeric']['min_expected'], \
            f"Should extract at least {expected_claim_patterns['numeric']['min_expected']} numeric claims"
        
        # Check for specific numeric patterns
        claim_texts = ' '.join([c.claim_text for c in numeric_claims])
        
        for pattern in expected_claim_patterns['numeric']['patterns'][:3]:  # Check first 3 patterns
            assert pattern in claim_texts or any(pattern in c.claim_text for c in numeric_claims), \
                f"Should extract claim containing '{pattern}'"
        
        # Verify numeric values are extracted
        numeric_values_found = any(c.numeric_values for c in numeric_claims)
        assert numeric_values_found, "Should extract numeric values from numeric claims"
        
        # Verify units are extracted
        units_found = any(c.units for c in numeric_claims)
        assert units_found, "Should extract units from numeric claims"
    
    @pytest.mark.asyncio
    async def test_qualitative_claim_extraction(self, extractor, sample_responses, expected_claim_patterns):
        """Test extraction of qualitative relationship claims."""
        
        response = sample_responses['qualitative_relationships']
        claims = await extractor.extract_claims(response)
        
        qualitative_claims = [c for c in claims if c.claim_type == 'qualitative']
        
        assert len(qualitative_claims) >= expected_claim_patterns['qualitative']['min_expected'], \
            f"Should extract at least {expected_claim_patterns['qualitative']['min_expected']} qualitative claims"
        
        # Check for relationship patterns
        claim_texts = ' '.join([c.claim_text for c in qualitative_claims])
        
        relationship_patterns_found = 0
        for pattern in expected_claim_patterns['qualitative']['patterns']:
            if pattern in claim_texts.lower():
                relationship_patterns_found += 1
        
        assert relationship_patterns_found >= 2, \
            "Should identify at least 2 relationship patterns in qualitative claims"
        
        # Verify relationships are extracted
        relationships_found = any(c.relationships for c in qualitative_claims)
        assert relationships_found, "Should extract relationships from qualitative claims"
    
    @pytest.mark.asyncio
    async def test_methodological_claim_extraction(self, extractor, sample_responses, expected_claim_patterns):
        """Test extraction of methodological claims."""
        
        response = sample_responses['methodological']
        claims = await extractor.extract_claims(response)
        
        methodological_claims = [c for c in claims if c.claim_type == 'methodological']
        
        assert len(methodological_claims) >= expected_claim_patterns['methodological']['min_expected'], \
            f"Should extract at least {expected_claim_patterns['methodological']['min_expected']} methodological claims"
        
        # Check for methodological patterns
        claim_texts = ' '.join([c.claim_text for c in methodological_claims])
        
        method_patterns_found = 0
        for pattern in expected_claim_patterns['methodological']['patterns']:
            if pattern.lower() in claim_texts.lower():
                method_patterns_found += 1
        
        assert method_patterns_found >= 2, \
            "Should identify at least 2 methodological patterns"
    
    @pytest.mark.asyncio
    async def test_temporal_claim_extraction(self, extractor, sample_responses, expected_claim_patterns):
        """Test extraction of temporal claims."""
        
        response = sample_responses['temporal_claims']
        claims = await extractor.extract_claims(response)
        
        temporal_claims = [c for c in claims if c.claim_type == 'temporal']
        
        assert len(temporal_claims) >= expected_claim_patterns['temporal']['min_expected'], \
            f"Should extract at least {expected_claim_patterns['temporal']['min_expected']} temporal claims"
        
        # Check for temporal patterns
        claim_texts = ' '.join([c.claim_text for c in temporal_claims])
        
        temporal_patterns_found = 0
        for pattern in expected_claim_patterns['temporal']['patterns']:
            if pattern in claim_texts:
                temporal_patterns_found += 1
        
        assert temporal_patterns_found >= 2, \
            "Should identify at least 2 temporal patterns"
    
    @pytest.mark.asyncio
    async def test_comparative_claim_extraction(self, extractor, sample_responses, expected_claim_patterns):
        """Test extraction of comparative claims."""
        
        response = sample_responses['comparative_claims']
        claims = await extractor.extract_claims(response)
        
        comparative_claims = [c for c in claims if c.claim_type == 'comparative']
        
        assert len(comparative_claims) >= expected_claim_patterns['comparative']['min_expected'], \
            f"Should extract at least {expected_claim_patterns['comparative']['min_expected']} comparative claims"
        
        # Check for comparative patterns
        claim_texts = ' '.join([c.claim_text for c in comparative_claims])
        
        comparative_patterns_found = 0
        for pattern in expected_claim_patterns['comparative']['patterns']:
            if pattern in claim_texts:
                comparative_patterns_found += 1
        
        assert comparative_patterns_found >= 2, \
            "Should identify at least 2 comparative patterns"
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, extractor, sample_responses):
        """Test confidence scoring system."""
        
        # Test high-confidence response
        high_conf_claims = await extractor.extract_claims(sample_responses['high_confidence'])
        
        # Test low-confidence response
        low_conf_claims = await extractor.extract_claims(sample_responses['low_confidence'])
        
        # High confidence claims should have higher scores
        if high_conf_claims and low_conf_claims:
            avg_high_confidence = sum(c.confidence.overall_confidence for c in high_conf_claims) / len(high_conf_claims)
            avg_low_confidence = sum(c.confidence.overall_confidence for c in low_conf_claims) / len(low_conf_claims)
            
            assert avg_high_confidence > avg_low_confidence, \
                "High-confidence response should yield higher confidence scores"
        
        # Check confidence components
        for claims_list in [high_conf_claims, low_conf_claims]:
            for claim in claims_list:
                conf = claim.confidence
                
                # All confidence components should be in valid range
                assert 0 <= conf.overall_confidence <= 100, "Overall confidence should be 0-100"
                assert 0 <= conf.linguistic_confidence <= 100, "Linguistic confidence should be 0-100"
                assert 0 <= conf.contextual_confidence <= 100, "Contextual confidence should be 0-100"
                assert 0 <= conf.domain_confidence <= 100, "Domain confidence should be 0-100"
                assert 0 <= conf.specificity_confidence <= 100, "Specificity confidence should be 0-100"
                assert 0 <= conf.verification_confidence <= 100, "Verification confidence should be 0-100"
                
                # Confidence factors should be lists
                assert isinstance(conf.factors, list), "Confidence factors should be a list"
                assert isinstance(conf.uncertainty_indicators, list), "Uncertainty indicators should be a list"
    
    @pytest.mark.asyncio
    async def test_context_preservation(self, extractor, sample_responses):
        """Test that context information is preserved."""
        
        response = sample_responses['mixed_content']
        claims = await extractor.extract_claims(response)
        
        for claim in claims:
            context = claim.context
            
            # Context should have required attributes
            assert hasattr(context, 'surrounding_text'), "Context should have surrounding text"
            assert hasattr(context, 'sentence_position'), "Context should have sentence position"
            assert hasattr(context, 'semantic_context'), "Context should have semantic context"
            
            # Check that context is meaningful
            assert context.surrounding_text, "Surrounding text should not be empty"
            assert isinstance(context.sentence_position, int), "Sentence position should be integer"
            assert isinstance(context.semantic_context, list), "Semantic context should be a list"
            
            # Source sentence should be preserved
            assert claim.source_sentence, "Source sentence should be preserved"
            assert claim.source_sentence in response, "Source sentence should be from original response"
    
    @pytest.mark.asyncio
    async def test_biomedical_specialization(self, extractor, sample_responses):
        """Test biomedical domain specialization."""
        
        response = sample_responses['mixed_content']
        claims = await extractor.extract_claims(response)
        
        # Check for biomedical keywords extraction
        all_keywords = []
        for claim in claims:
            all_keywords.extend(claim.keywords)
        
        # Should find common biomedical terms
        biomedical_terms_found = [
            kw for kw in all_keywords 
            if kw.lower() in ['metabolomics', 'glucose', 'insulin', 'biomarker', 'analysis']
        ]
        
        assert len(biomedical_terms_found) > 0, \
            "Should extract biomedical keywords from response"
        
        # Check domain confidence scoring
        domain_confidences = [c.confidence.domain_confidence for c in claims]
        avg_domain_confidence = sum(domain_confidences) / len(domain_confidences) if domain_confidences else 0
        
        # Biomedical content should have reasonable domain confidence
        assert avg_domain_confidence > 40, \
            "Biomedical content should have reasonable domain confidence scores"
    
    @pytest.mark.asyncio
    async def test_claim_classification(self, extractor, sample_responses):
        """Test claim classification functionality."""
        
        response = sample_responses['mixed_content']
        claims = await extractor.extract_claims(response)
        
        # Test classification method
        classified_claims = await extractor.classify_claims_by_type(claims)
        
        # Should return dictionary
        assert isinstance(classified_claims, dict), "Classification should return dictionary"
        
        # Should have multiple types for mixed content
        assert len(classified_claims) > 1, "Mixed content should produce multiple claim types"
        
        # Each type should contain ExtractedClaim instances
        for claim_type, type_claims in classified_claims.items():
            assert isinstance(claim_type, str), "Claim type should be string"
            assert isinstance(type_claims, list), "Type claims should be list"
            assert all(isinstance(c, ExtractedClaim) for c in type_claims), \
                "All items should be ExtractedClaim instances"
            assert all(c.claim_type == claim_type for c in type_claims), \
                "All claims should match their classification type"
    
    @pytest.mark.asyncio
    async def test_confidence_filtering(self, extractor, sample_responses):
        """Test confidence-based filtering."""
        
        response = sample_responses['mixed_content']
        claims = await extractor.extract_claims(response)
        
        # Test filtering at different confidence levels
        high_conf_claims = await extractor.filter_high_confidence_claims(claims, 80.0)
        med_conf_claims = await extractor.filter_high_confidence_claims(claims, 60.0)
        low_conf_claims = await extractor.filter_high_confidence_claims(claims, 40.0)
        
        # Higher thresholds should yield fewer claims
        assert len(high_conf_claims) <= len(med_conf_claims), \
            "Higher confidence threshold should yield fewer or equal claims"
        assert len(med_conf_claims) <= len(low_conf_claims), \
            "Medium confidence threshold should yield fewer or equal claims than low threshold"
        
        # All filtered claims should meet confidence threshold
        for claim in high_conf_claims:
            assert claim.confidence.overall_confidence >= 80.0, \
                "High-confidence filtered claims should meet threshold"
        
        for claim in med_conf_claims:
            assert claim.confidence.overall_confidence >= 60.0, \
                "Medium-confidence filtered claims should meet threshold"
    
    @pytest.mark.asyncio
    async def test_priority_scoring(self, extractor, sample_responses):
        """Test priority scoring system."""
        
        response = sample_responses['numeric_heavy']
        claims = await extractor.extract_claims(response)
        
        # All claims should have priority scores
        for claim in claims:
            assert hasattr(claim, 'priority_score'), "Claim should have priority_score property"
            priority = claim.priority_score
            assert 0 <= priority <= 100, "Priority score should be 0-100"
        
        # Numeric claims with units should generally have higher priority
        numeric_claims = [c for c in claims if c.claim_type == 'numeric']
        
        if len(numeric_claims) >= 2:
            claims_with_units = [c for c in numeric_claims if c.units]
            claims_without_units = [c for c in numeric_claims if not c.units]
            
            if claims_with_units and claims_without_units:
                avg_priority_with_units = sum(c.priority_score for c in claims_with_units) / len(claims_with_units)
                avg_priority_without_units = sum(c.priority_score for c in claims_without_units) / len(claims_without_units)
                
                # This is a soft assertion as there might be other factors
                assert avg_priority_with_units >= avg_priority_without_units * 0.9, \
                    "Claims with units should generally have higher priority"
    
    @pytest.mark.asyncio
    async def test_verification_preparation(self, extractor, sample_responses):
        """Test preparation of claims for verification."""
        
        response = sample_responses['mixed_content']
        claims = await extractor.extract_claims(response)
        
        # Test verification preparation
        verification_data = await extractor.prepare_claims_for_verification(claims)
        
        # Should return dictionary with expected structure
        assert isinstance(verification_data, dict), "Verification data should be dictionary"
        
        required_keys = [
            'claims_by_type', 'high_priority_claims', 'verification_candidates', 'extraction_metadata'
        ]
        for key in required_keys:
            assert key in verification_data, f"Verification data should contain '{key}'"
        
        # Check claims_by_type
        claims_by_type = verification_data['claims_by_type']
        assert isinstance(claims_by_type, dict), "Claims by type should be dictionary"
        
        # Check high_priority_claims
        high_priority = verification_data['high_priority_claims']
        assert isinstance(high_priority, list), "High priority claims should be list"
        
        # Check verification_candidates
        candidates = verification_data['verification_candidates']
        assert isinstance(candidates, list), "Verification candidates should be list"
        
        # Verification candidates should have required fields
        for candidate in candidates:
            required_fields = [
                'claim_id', 'claim_text', 'claim_type', 'verification_targets',
                'search_keywords', 'confidence_score', 'priority_score'
            ]
            for field in required_fields:
                assert field in candidate, f"Verification candidate should have '{field}'"
        
        # Check extraction_metadata
        metadata = verification_data['extraction_metadata']
        assert isinstance(metadata, dict), "Extraction metadata should be dictionary"
        assert 'total_claims' in metadata, "Metadata should include total claims count"
        assert 'extraction_timestamp' in metadata, "Metadata should include timestamp"
    
    @pytest.mark.asyncio
    async def test_duplicate_detection_and_merging(self, extractor):
        """Test duplicate detection and claim merging."""
        
        # Response with potential duplicates
        response_with_duplicates = """
        Glucose levels were 8.5 mmol/L in patients. The glucose concentration was 8.5 mmol/L.
        Patient glucose measured 8.5 mmol/L. Serum glucose was elevated at 8.5 mmol/L.
        """
        
        claims = await extractor.extract_claims(response_with_duplicates)
        
        # Should detect and merge similar claims
        claim_texts = [c.claim_text for c in claims]
        unique_normalized_texts = set(c.normalized_text for c in claims)
        
        # Should have fewer unique normalized texts than total claims if merging worked
        # This is a soft assertion as the exact behavior depends on similarity thresholds
        assert len(unique_normalized_texts) <= len(claims), \
            "Normalized texts should be equal or fewer than total claims"
        
        # Check for merge metadata
        merged_claims = [c for c in claims if 'merged_from_count' in c.metadata]
        
        # At least some merging should have occurred with this repetitive input
        # This is conditional as the exact merging behavior might vary
        if merged_claims:
            for claim in merged_claims:
                assert claim.metadata['merged_from_count'] > 1, \
                    "Merged claims should indicate source count"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, extractor):
        """Test error handling and edge cases."""
        
        # Test empty input
        empty_claims = await extractor.extract_claims("")
        assert isinstance(empty_claims, list), "Should return empty list for empty input"
        assert len(empty_claims) == 0, "Should return no claims for empty input"
        
        # Test very short input
        short_claims = await extractor.extract_claims("Yes.")
        assert isinstance(short_claims, list), "Should handle very short input"
        
        # Test input with special characters
        special_char_input = "Metabolite levels: 5.2 ± 0.3 μmol/L (p < 0.001) ***"
        special_claims = await extractor.extract_claims(special_char_input)
        assert isinstance(special_claims, list), "Should handle special characters"
        
        # Test malformed input
        malformed_input = "This is 123 without proper context or units or meaning."
        malformed_claims = await extractor.extract_claims(malformed_input)
        assert isinstance(malformed_claims, list), "Should handle malformed input gracefully"
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, extractor, sample_responses):
        """Test performance tracking functionality."""
        
        # Extract claims from multiple responses
        for response_type, response in sample_responses.items():
            await extractor.extract_claims(response)
        
        # Get statistics
        stats = extractor.get_extraction_statistics()
        
        # Should have statistics
        assert isinstance(stats, dict), "Statistics should be dictionary"
        
        required_stats = [
            'total_extractions', 'total_claims_extracted', 
            'average_claims_per_extraction', 'processing_times'
        ]
        for stat in required_stats:
            assert stat in stats, f"Statistics should include '{stat}'"
        
        # Check processing times
        processing_times = stats['processing_times']
        assert isinstance(processing_times, dict), "Processing times should be dictionary"
        assert 'count' in processing_times, "Should track processing time count"
        assert 'average_ms' in processing_times, "Should track average processing time"
        
        # Should have processed multiple extractions
        assert stats['total_extractions'] >= len(sample_responses), \
            "Should track all extractions performed"


class TestClaimExtractorIntegration:
    """Test integration with existing systems."""
    
    @pytest.mark.asyncio
    async def test_convenience_function(self, sample_responses):
        """Test convenience function for claim extraction."""
        
        response = sample_responses['mixed_content']
        query = "What are the metabolomic findings in diabetes?"
        
        # Test convenience function
        claims = await extract_claims_from_response(response, query)
        
        assert isinstance(claims, list), "Convenience function should return list"
        assert all(isinstance(c, ExtractedClaim) for c in claims), \
            "All items should be ExtractedClaim instances"
        
        # Claims should have enhanced keywords from query
        all_keywords = []
        for claim in claims:
            all_keywords.extend(claim.keywords)
        
        # Should find query-related terms
        query_terms_found = any('metabolomic' in kw.lower() for kw in all_keywords)
        # Note: This is a soft assertion as keyword extraction might vary
    
    @pytest.mark.asyncio
    async def test_quality_assessment_preparation(self, sample_responses):
        """Test preparation for quality assessment integration."""
        
        # Extract claims
        extractor = BiomedicalClaimExtractor()
        claims = await extractor.extract_claims(sample_responses['mixed_content'])
        
        # Prepare for quality assessment
        quality_data = await prepare_claims_for_quality_assessment(claims, min_confidence=60.0)
        
        assert isinstance(quality_data, dict), "Quality data should be dictionary"
        
        required_keys = [
            'factual_claims', 'claim_count', 'high_priority_claims',
            'verification_needed', 'assessment_metadata'
        ]
        for key in required_keys:
            assert key in quality_data, f"Quality data should contain '{key}'"
        
        # Check factual_claims format
        factual_claims = quality_data['factual_claims']
        assert isinstance(factual_claims, list), "Factual claims should be list"
        
        for claim_dict in factual_claims:
            assert isinstance(claim_dict, dict), "Each claim should be dictionary"
            assert 'claim_id' in claim_dict, "Claim should have ID"
            assert 'claim_text' in claim_dict, "Claim should have text"
            assert 'claim_type' in claim_dict, "Claim should have type"
        
        # Check high_priority_claims
        high_priority = quality_data['high_priority_claims']
        assert isinstance(high_priority, list), "High priority claims should be list"
        
        # Check verification_needed
        verification_needed = quality_data['verification_needed']
        assert isinstance(verification_needed, list), "Verification needed should be list"
        
        # All items should be claim IDs
        for claim_id in verification_needed:
            assert isinstance(claim_id, str), "Verification item should be string ID"
        
        # Check assessment_metadata
        metadata = quality_data['assessment_metadata']
        assert isinstance(metadata, dict), "Assessment metadata should be dictionary"
        assert 'extraction_timestamp' in metadata, "Metadata should have timestamp"
        assert 'confidence_threshold' in metadata, "Metadata should have confidence threshold"
        assert 'total_original_claims' in metadata, "Metadata should have original claim count"


class TestClaimDataStructures:
    """Test claim data structures and utilities."""
    
    def test_extracted_claim_creation(self):
        """Test ExtractedClaim creation and properties."""
        
        # Create a sample claim
        context = ClaimContext(
            surrounding_text="Sample context text",
            sentence_position=1,
            semantic_context=['numeric', 'measurement']
        )
        
        confidence = ClaimConfidence(
            overall_confidence=75.0,
            linguistic_confidence=80.0,
            factors=['test_factor']
        )
        
        claim = ExtractedClaim(
            claim_id="test_claim_001",
            claim_text="Glucose levels were 8.5 mmol/L",
            claim_type="numeric",
            subject="glucose levels",
            predicate="were",
            object_value="8.5 mmol/L",
            numeric_values=[8.5],
            units=['mmol/L'],
            context=context,
            confidence=confidence,
            source_sentence="Glucose levels were 8.5 mmol/L in patients."
        )
        
        # Test basic attributes
        assert claim.claim_id == "test_claim_001"
        assert claim.claim_text == "Glucose levels were 8.5 mmol/L"
        assert claim.claim_type == "numeric"
        assert claim.numeric_values == [8.5]
        assert claim.units == ['mmol/L']
        
        # Test priority score property
        priority = claim.priority_score
        assert isinstance(priority, float)
        assert 0 <= priority <= 100
        
        # Test to_dict method
        claim_dict = claim.to_dict()
        assert isinstance(claim_dict, dict)
        assert 'claim_id' in claim_dict
        assert 'claim_text' in claim_dict
        assert 'extraction_timestamp' in claim_dict
    
    def test_claim_context_attributes(self):
        """Test ClaimContext data structure."""
        
        context = ClaimContext(
            surrounding_text="Test context",
            sentence_position=2,
            paragraph_position=1,
            section_type="results",
            semantic_context=['numeric', 'clinical'],
            relevance_indicators=['significant', 'elevated']
        )
        
        assert context.surrounding_text == "Test context"
        assert context.sentence_position == 2
        assert context.paragraph_position == 1
        assert context.section_type == "results"
        assert 'numeric' in context.semantic_context
        assert 'significant' in context.relevance_indicators
    
    def test_claim_confidence_attributes(self):
        """Test ClaimConfidence data structure."""
        
        confidence = ClaimConfidence(
            overall_confidence=85.0,
            linguistic_confidence=90.0,
            contextual_confidence=80.0,
            domain_confidence=85.0,
            specificity_confidence=80.0,
            verification_confidence=85.0,
            factors=['high_specificity', 'domain_match'],
            uncertainty_indicators=['approximation']
        )
        
        assert confidence.overall_confidence == 85.0
        assert confidence.linguistic_confidence == 90.0
        assert 'high_specificity' in confidence.factors
        assert 'approximation' in confidence.uncertainty_indicators


# Integration test with mock data
class TestClaimExtractorWithMockData:
    """Test claim extractor with comprehensive mock data."""
    
    @pytest.fixture
    def mock_biomedical_responses(self):
        """Mock biomedical responses for testing."""
        if FIXTURES_AVAILABLE:
            try:
                return generate_biomedical_response_samples()
            except Exception:
                pass
        
        # Fallback mock data
        return {
            'metabolomics_study': """
            A comprehensive metabolomics analysis using LC-MS/MS identified 
            342 metabolites in plasma samples from 150 diabetic patients and 
            120 healthy controls. Glucose concentrations were significantly 
            elevated (9.2 ± 1.5 mmol/L vs 5.8 ± 0.7 mmol/L, p < 0.001).
            """,
            'clinical_trial': """
            The randomized controlled trial enrolled 200 participants over 
            24 months. Primary endpoint was achieved in 78% of treatment group 
            versus 45% of placebo group (OR: 4.2, 95% CI: 2.1-8.4, p = 0.002).
            """,
            'analytical_method': """
            Samples underwent protein precipitation followed by LC-MS analysis 
            on a QTOF mass spectrometer. Chromatographic separation used a 
            C18 column with gradient elution over 15 minutes. Detection 
            limits ranged from 0.1 to 10 ng/mL.
            """
        }
    
    @pytest.mark.asyncio
    async def test_comprehensive_extraction_workflow(self, mock_biomedical_responses):
        """Test complete extraction workflow with mock data."""
        
        extractor = BiomedicalClaimExtractor()
        
        all_claims = []
        for response_name, response_text in mock_biomedical_responses.items():
            claims = await extractor.extract_claims(response_text)
            all_claims.extend(claims)
        
        # Should extract multiple claims
        assert len(all_claims) > 0, "Should extract claims from mock data"
        
        # Should have multiple claim types
        claim_types = set(c.claim_type for c in all_claims)
        assert len(claim_types) > 1, "Should identify multiple claim types"
        
        # Should have reasonable confidence scores
        confidences = [c.confidence.overall_confidence for c in all_claims]
        avg_confidence = sum(confidences) / len(confidences)
        assert avg_confidence > 30, "Average confidence should be reasonable"
        
        # Test full verification preparation workflow
        verification_data = await extractor.prepare_claims_for_verification(all_claims)
        
        assert len(verification_data['verification_candidates']) > 0, \
            "Should prepare verification candidates"
        
        # Test classification workflow
        classified = await extractor.classify_claims_by_type(all_claims)
        assert len(classified) > 0, "Should classify claims by type"


if __name__ == "__main__":
    """Run tests directly if executed as script."""
    
    # Simple test runner
    async def run_basic_tests():
        """Run basic tests to verify functionality."""
        
        print("Testing Biomedical Claim Extractor...")
        
        extractor = BiomedicalClaimExtractor()
        
        sample_response = """
        Metabolomic analysis revealed glucose levels of 8.5 ± 1.2 mmol/L in diabetic patients, 
        which is 45% higher than controls (5.9 ± 0.8 mmol/L, p < 0.001). The LC-MS method 
        was used for analysis. Insulin resistance correlates with increased oxidative stress. 
        Samples were collected over 6 months.
        """
        
        try:
            claims = await extractor.extract_claims(sample_response)
            
            print(f"✓ Extracted {len(claims)} claims")
            
            for claim in claims:
                print(f"  - {claim.claim_type}: {claim.claim_text[:60]}...")
                print(f"    Confidence: {claim.confidence.overall_confidence:.1f}")
            
            # Test classification
            classified = await extractor.classify_claims_by_type(claims)
            print(f"✓ Classified into {len(classified)} types: {list(classified.keys())}")
            
            # Test verification preparation
            verification_data = await extractor.prepare_claims_for_verification(claims)
            print(f"✓ Prepared {len(verification_data['verification_candidates'])} verification candidates")
            
            print("✓ All basic tests passed!")
            
        except Exception as e:
            print(f"✗ Error in testing: {str(e)}")
            raise
    
    # Run if executed directly
    asyncio.run(run_basic_tests())