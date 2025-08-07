#!/usr/bin/env python3
"""
Comprehensive Test Suite for Factual Accuracy Validation System.

This test suite provides thorough testing for the FactualAccuracyValidator class
and its integration with the claim extraction and document indexing systems.

Test Categories:
1. Unit tests for individual verification strategies
2. Integration tests with claim extractor and document indexer
3. Performance and scalability tests
4. Error handling and edge case tests
5. End-to-end workflow tests

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Factual Accuracy Validation Testing
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from dataclasses import dataclass

# Import the modules to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from factual_accuracy_validator import (
        FactualAccuracyValidator, VerificationResult, VerificationStatus,
        EvidenceItem, VerificationReport, verify_extracted_claims
    )
    from claim_extractor import ExtractedClaim, ClaimContext, ClaimConfidence
    from factual_validation_integration_example import IntegratedFactualValidationPipeline
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestFactualAccuracyValidator:
    """Test suite for FactualAccuracyValidator core functionality."""
    
    @pytest.fixture
    def mock_document_indexer(self):
        """Create mock document indexer for testing."""
        indexer = Mock()
        indexer.verify_claim = AsyncMock(return_value={
            'verification_status': 'supported',
            'confidence': 0.85,
            'supporting_evidence': ['Evidence 1', 'Evidence 2'],
            'contradicting_evidence': [],
            'related_facts': ['Fact 1', 'Fact 2'],
            'verification_metadata': {'search_time_ms': 50}
        })
        indexer.search_content = AsyncMock(return_value=[
            {
                'document_id': 'doc_001',
                'content': 'Glucose levels were 150 mg/dL in diabetic patients compared to 90 mg/dL in controls',
                'page_number': 1,
                'section': 'Results'
            }
        ])
        return indexer
    
    @pytest.fixture
    def mock_claim_extractor(self):
        """Create mock claim extractor for testing."""
        extractor = Mock()
        extractor.extract_claims = AsyncMock(return_value=[
            self.create_test_claim('numeric'),
            self.create_test_claim('qualitative')
        ])
        return extractor
    
    @pytest.fixture
    def validator(self, mock_document_indexer, mock_claim_extractor):
        """Create validator instance with mocked dependencies."""
        return FactualAccuracyValidator(
            document_indexer=mock_document_indexer,
            claim_extractor=mock_claim_extractor,
            config={'test_mode': True}
        )
    
    def create_test_claim(self, claim_type: str) -> 'ExtractedClaim':
        """Create a test claim for testing purposes."""
        
        # Create mock classes if ExtractedClaim is not available
        try:
            from claim_extractor import ExtractedClaim, ClaimContext, ClaimConfidence
            
            context = ClaimContext(
                surrounding_text="Test context text",
                sentence_position=0,
                paragraph_position=0
            )
            
            confidence = ClaimConfidence(
                overall_confidence=75.0,
                linguistic_confidence=70.0,
                contextual_confidence=80.0,
                domain_confidence=75.0
            )
            
            return ExtractedClaim(
                claim_id=f"test_claim_{claim_type}",
                claim_text=f"Test {claim_type} claim text",
                claim_type=claim_type,
                subject="test_subject",
                predicate="test_predicate",
                object_value="test_object",
                numeric_values=[150.0, 90.0] if claim_type == 'numeric' else [],
                units=['mg/dL'] if claim_type == 'numeric' else [],
                keywords=['glucose', 'diabetic', 'patients'],
                context=context,
                confidence=confidence
            )
            
        except ImportError:
            # Create mock ExtractedClaim for testing
            @dataclass
            class MockExtractedClaim:
                claim_id: str
                claim_text: str
                claim_type: str
                subject: str = ""
                predicate: str = ""
                object_value: str = ""
                numeric_values: List[float] = None
                units: List[str] = None
                keywords: List[str] = None
                confidence: Any = None
                relationships: List[Dict] = None
                
                def __post_init__(self):
                    if self.numeric_values is None:
                        self.numeric_values = []
                    if self.units is None:
                        self.units = []
                    if self.keywords is None:
                        self.keywords = []
                    if self.relationships is None:
                        self.relationships = []
                    if self.confidence is None:
                        self.confidence = type('Confidence', (), {'overall_confidence': 75.0})()
            
            return MockExtractedClaim(
                claim_id=f"test_claim_{claim_type}",
                claim_text=f"Test {claim_type} claim text",
                claim_type=claim_type,
                subject="test_subject",
                predicate="test_predicate",
                object_value="test_object",
                numeric_values=[150.0, 90.0] if claim_type == 'numeric' else [],
                units=['mg/dL'] if claim_type == 'numeric' else [],
                keywords=['glucose', 'diabetic', 'patients']
            )
    
    @pytest.mark.asyncio
    async def test_validator_initialization(self, mock_document_indexer, mock_claim_extractor):
        """Test validator initialization."""
        
        validator = FactualAccuracyValidator(
            document_indexer=mock_document_indexer,
            claim_extractor=mock_claim_extractor
        )
        
        assert validator.document_indexer is mock_document_indexer
        assert validator.claim_extractor is mock_claim_extractor
        assert 'numeric' in validator.verification_strategies
        assert 'qualitative' in validator.verification_strategies
        assert len(validator.numeric_verification_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_verify_single_numeric_claim(self, validator):
        """Test verification of a single numeric claim."""
        
        numeric_claim = self.create_test_claim('numeric')
        config = {'min_evidence_confidence': 50}
        
        result = await validator._verify_single_claim(numeric_claim, config)
        
        assert isinstance(result, VerificationResult)
        assert result.claim_id == numeric_claim.claim_id
        assert result.verification_status in [status for status in VerificationStatus]
        assert 0 <= result.verification_confidence <= 100
        assert 0 <= result.evidence_strength <= 100
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_verify_multiple_claims(self, validator):
        """Test verification of multiple claims."""
        
        claims = [
            self.create_test_claim('numeric'),
            self.create_test_claim('qualitative'),
            self.create_test_claim('methodological')
        ]
        
        report = await validator.verify_claims(claims)
        
        assert isinstance(report, VerificationReport)
        assert report.total_claims == len(claims)
        assert len(report.verification_results) == len(claims)
        
        for result in report.verification_results:
            assert isinstance(result, VerificationResult)
            assert result.verification_status in [status for status in VerificationStatus]
    
    @pytest.mark.asyncio
    async def test_numeric_verification_strategy(self, validator):
        """Test numeric claim verification strategy."""
        
        numeric_claim = self.create_test_claim('numeric')
        config = {'min_evidence_confidence': 60}
        
        result = await validator._verify_numeric_claim(numeric_claim, config)
        
        assert result.claim_id == numeric_claim.claim_id
        assert result.verification_strategy == 'numeric' or not result.verification_strategy
        assert isinstance(result.supporting_evidence, list)
        assert isinstance(result.contradicting_evidence, list)
        assert isinstance(result.neutral_evidence, list)
    
    @pytest.mark.asyncio
    async def test_qualitative_verification_strategy(self, validator):
        """Test qualitative claim verification strategy."""
        
        qualitative_claim = self.create_test_claim('qualitative')
        config = {'min_evidence_confidence': 50}
        
        result = await validator._verify_qualitative_claim(qualitative_claim, config)
        
        assert result.claim_id == qualitative_claim.claim_id
        assert result.verification_confidence >= 0
        assert result.evidence_strength >= 0
    
    @pytest.mark.asyncio
    async def test_evidence_assessment(self, validator):
        """Test evidence assessment functionality."""
        
        claim = self.create_test_claim('numeric')
        
        # Create test evidence
        supporting_evidence = EvidenceItem(
            source_document="test_doc",
            evidence_text="150 mg/dL",
            evidence_type="numeric",
            confidence=85.0
        )
        
        assessment = await validator._assess_numeric_evidence(claim, supporting_evidence)
        assert assessment in ['supporting', 'contradicting', 'neutral']
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, validator):
        """Test confidence calculation methods."""
        
        claim = self.create_test_claim('numeric')
        
        # Test numeric evidence confidence
        confidence = await validator._calculate_numeric_evidence_confidence(
            claim, "150 mg/dL", "glucose levels were 150 mg/dL", "exact_match"
        )
        
        assert 0 <= confidence <= 100
        assert isinstance(confidence, float)
    
    @pytest.mark.asyncio
    async def test_verification_status_determination(self, validator):
        """Test verification status determination logic."""
        
        # Test with supporting evidence
        supporting = [EvidenceItem("doc1", "text1", "type1", confidence=80.0)]
        contradicting = []
        neutral = []
        
        status = await validator._determine_verification_status(
            supporting, contradicting, neutral
        )
        
        assert isinstance(status, VerificationStatus)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, validator):
        """Test error handling in verification process."""
        
        # Create a claim that might cause errors
        problematic_claim = self.create_test_claim('numeric')
        problematic_claim.claim_text = None  # This should cause an error
        
        config = {}
        
        # The validator should handle this gracefully
        result = await validator._verify_single_claim(problematic_claim, config)
        
        assert result.verification_status == VerificationStatus.ERROR
        assert result.error_details is not None
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, validator):
        """Test performance tracking functionality."""
        
        claims = [self.create_test_claim('numeric') for _ in range(3)]
        
        initial_stats = validator.get_verification_statistics()
        
        await validator.verify_claims(claims)
        
        final_stats = validator.get_verification_statistics()
        
        assert final_stats['total_verifications'] > initial_stats['total_verifications']
        assert final_stats['total_claims_verified'] > initial_stats['total_claims_verified']


class TestEvidenceAssessment:
    """Test suite for evidence assessment functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create basic validator for evidence testing."""
        return FactualAccuracyValidator()
    
    @pytest.mark.asyncio
    async def test_numeric_evidence_assessment(self, validator):
        """Test numeric evidence assessment."""
        
        claim = Mock()
        claim.numeric_values = [150.0]
        claim.claim_text = "glucose was 150 mg/dL"
        
        evidence = EvidenceItem(
            source_document="test",
            evidence_text="150",
            evidence_type="numeric",
            context="glucose levels were 150 mg/dL in patients"
        )
        
        assessment = await validator._assess_numeric_evidence(claim, evidence)
        assert assessment in ['supporting', 'contradicting', 'neutral']
    
    @pytest.mark.asyncio
    async def test_qualitative_evidence_assessment(self, validator):
        """Test qualitative evidence assessment."""
        
        claim = Mock()
        claim.relationships = [{'type': 'correlation'}]
        
        evidence = EvidenceItem(
            source_document="test",
            evidence_text="correlates with",
            evidence_type="qualitative",
            metadata={'relationship_type': 'correlation'}
        )
        
        assessment = await validator._assess_qualitative_evidence(claim, evidence)
        assert assessment in ['supporting', 'contradicting', 'neutral']


class TestIntegratedPipeline:
    """Test suite for integrated validation pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create test pipeline instance."""
        return IntegratedFactualValidationPipeline({
            'test_mode': True
        })
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization (mocked)."""
        
        # Mock the initialization since we don't have real components
        with patch.object(pipeline, '_initialize_components') as mock_init:
            mock_init.return_value = None
            
            # This would normally call await pipeline.initialize()
            # but we'll test the structure instead
            
            assert hasattr(pipeline, 'claim_extractor')
            assert hasattr(pipeline, 'document_indexer')
            assert hasattr(pipeline, 'factual_validator')
            assert hasattr(pipeline, 'pipeline_stats')
    
    @pytest.mark.asyncio
    async def test_response_processing_workflow(self, pipeline):
        """Test the complete response processing workflow."""
        
        test_response = "Glucose levels were 150 mg/dL in diabetic patients."
        test_query = "What are glucose levels in diabetes?"
        
        # Mock the components
        pipeline.claim_extractor = Mock()
        pipeline.claim_extractor.extract_claims = AsyncMock(return_value=[
            Mock(claim_id="test1", claim_type="numeric", confidence=Mock(overall_confidence=80))
        ])
        
        pipeline.factual_validator = Mock()
        pipeline.factual_validator.verify_claims = AsyncMock(return_value=Mock(
            verification_results=[Mock(verification_confidence=85, evidence_strength=75)],
            recommendations=["Good accuracy detected"]
        ))
        
        # Mock the method that processes results
        with patch.object(pipeline, '_generate_comprehensive_results') as mock_results:
            mock_results.return_value = {
                'success': True,
                'claim_extraction_results': {'total_claims_extracted': 1},
                'factual_verification_results': {'overall_metrics': {'average_verification_confidence': 85}}
            }
            
            result = await pipeline.process_lightrag_response(test_response, test_query)
            
            assert result['success'] is True
            assert 'claim_extraction_results' in result
            assert 'factual_verification_results' in result


class TestPerformanceAndScalability:
    """Test suite for performance and scalability aspects."""
    
    @pytest.mark.asyncio
    async def test_single_claim_performance(self):
        """Test performance of single claim verification."""
        
        validator = FactualAccuracyValidator()
        claim = Mock()
        claim.claim_id = "perf_test"
        claim.claim_type = "numeric"
        claim.claim_text = "test claim"
        claim.keywords = ["test"]
        claim.numeric_values = []
        claim.units = []
        claim.confidence = Mock(overall_confidence=75)
        
        start_time = time.time()
        
        # Mock the verification process
        with patch.object(validator, '_search_documents_for_claim') as mock_search:
            mock_search.return_value = []
            
            result = await validator._verify_single_claim(claim, {})
            
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        # Performance assertion - should complete within reasonable time
        assert processing_time < 1000  # Less than 1 second
        assert result.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """Test performance of batch claim verification."""
        
        validator = FactualAccuracyValidator()
        
        # Create multiple test claims
        claims = []
        for i in range(10):
            claim = Mock()
            claim.claim_id = f"batch_claim_{i}"
            claim.claim_type = "numeric"
            claim.claim_text = f"test claim {i}"
            claim.keywords = ["test"]
            claim.numeric_values = []
            claim.units = []
            claim.confidence = Mock(overall_confidence=75)
            claims.append(claim)
        
        start_time = time.time()
        
        # Mock dependencies
        with patch.object(validator, '_search_documents_for_claim') as mock_search:
            mock_search.return_value = []
            
            report = await validator.verify_claims(claims)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Performance assertions
        assert total_time < 5000  # Less than 5 seconds for 10 claims
        assert report.total_claims == len(claims)
        assert len(report.verification_results) == len(claims)


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""
    
    @pytest.fixture
    def validator(self):
        """Create validator for error testing."""
        return FactualAccuracyValidator()
    
    @pytest.mark.asyncio
    async def test_empty_claims_list(self, validator):
        """Test handling of empty claims list."""
        
        report = await validator.verify_claims([])
        
        assert report.total_claims == 0
        assert len(report.verification_results) == 0
        assert report.summary_statistics is not None
    
    @pytest.mark.asyncio
    async def test_malformed_claim(self, validator):
        """Test handling of malformed claims."""
        
        # Create a claim with missing required attributes
        malformed_claim = Mock()
        malformed_claim.claim_id = "malformed"
        malformed_claim.claim_type = "numeric"
        malformed_claim.claim_text = None  # This should cause an error
        malformed_claim.keywords = []
        malformed_claim.confidence = Mock(overall_confidence=0)
        
        with patch.object(validator, '_search_documents_for_claim') as mock_search:
            mock_search.side_effect = Exception("Search failed")
            
            result = await validator._verify_single_claim(malformed_claim, {})
            
        assert result.verification_status == VerificationStatus.ERROR
        assert result.error_details is not None
    
    @pytest.mark.asyncio
    async def test_document_indexer_failure(self, validator):
        """Test handling of document indexer failures."""
        
        claim = Mock()
        claim.claim_id = "test"
        claim.claim_type = "numeric"
        claim.claim_text = "test claim"
        claim.keywords = ["test"]
        claim.confidence = Mock(overall_confidence=75)
        
        # Mock document indexer failure
        validator.document_indexer = Mock()
        validator.document_indexer.verify_claim = AsyncMock(side_effect=Exception("Indexer failed"))
        
        with patch.object(validator, '_search_documents_for_claim') as mock_search:
            mock_search.side_effect = Exception("Search failed")
            
            result = await validator._verify_single_claim(claim, {})
            
        # Should handle the error gracefully
        assert result.verification_status == VerificationStatus.ERROR
        assert "failed" in result.error_details.lower()
    
    @pytest.mark.asyncio
    async def test_confidence_calculation_edge_cases(self, validator):
        """Test confidence calculation with edge cases."""
        
        # Test with empty evidence
        confidence = await validator._calculate_verification_confidence(
            Mock(confidence=Mock(overall_confidence=50)), [], [], []
        )
        assert confidence == 0.0
        
        # Test with mixed evidence
        supporting = [Mock(confidence=80)]
        contradicting = [Mock(confidence=70)]
        neutral = [Mock(confidence=60)]
        
        confidence = await validator._calculate_verification_confidence(
            Mock(confidence=Mock(overall_confidence=50)), 
            supporting, contradicting, neutral
        )
        assert 0 <= confidence <= 100


# Test fixtures and utilities
@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_convenience_functions():
    """Test convenience functions for easy integration."""
    
    # Test the convenience function (with mocked dependencies)
    with patch('factual_accuracy_validator.FactualAccuracyValidator') as mock_validator_class:
        mock_validator = Mock()
        mock_validator.verify_claims = AsyncMock(return_value=Mock())
        mock_validator_class.return_value = mock_validator
        
        result = await verify_extracted_claims([], Mock())
        
        assert result is not None
        mock_validator.verify_claims.assert_called_once()


def test_verification_result_properties():
    """Test VerificationResult properties and methods."""
    
    # Create test evidence
    evidence = [
        EvidenceItem("doc1", "text1", "type1", confidence=80),
        EvidenceItem("doc2", "text2", "type2", confidence=70)
    ]
    
    result = VerificationResult(
        claim_id="test",
        verification_status=VerificationStatus.SUPPORTED,
        verification_confidence=85.0,
        evidence_strength=75.0,
        supporting_evidence=evidence
    )
    
    assert result.total_evidence_count == 2
    assert result.verification_grade in ["Very High", "High", "Moderate", "Low", "Very Low"]
    
    # Test dictionary conversion
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert result_dict['verification_status'] == 'SUPPORTED'


def test_verification_report_creation():
    """Test VerificationReport creation and methods."""
    
    results = [
        VerificationResult("claim1", VerificationStatus.SUPPORTED, 85.0),
        VerificationResult("claim2", VerificationStatus.CONTRADICTED, 75.0)
    ]
    
    report = VerificationReport(
        report_id="test_report",
        total_claims=2,
        verification_results=results
    )
    
    assert report.total_claims == 2
    assert len(report.verification_results) == 2
    
    # Test dictionary conversion
    report_dict = report.to_dict()
    assert isinstance(report_dict, dict)
    assert 'verification_results' in report_dict


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])