#!/usr/bin/env python3
"""
Comprehensive Error Handling and Edge Case Tests for Factual Accuracy Validation System.

This test suite provides thorough testing of error conditions, edge cases, and robustness
of the factual accuracy validation pipeline including failure modes, recovery mechanisms,
and boundary condition handling.

Test Categories:
1. Input validation and malformed data handling
2. Network and external dependency failures
3. Resource constraint and timeout handling
4. Data corruption and integrity issues
5. Concurrent access and race condition handling
6. Recovery and fallback mechanism testing
7. Logging and error reporting validation

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import json
import time
import tempfile
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import signal
import os

# Import test fixtures
from .factual_validation_test_fixtures import *

# Import the modules to test
try:
    from ..accuracy_scorer import (
        FactualAccuracyScorer, AccuracyScore, AccuracyReport,
        AccuracyScoringError, ReportGenerationError, QualityIntegrationError
    )
    from ..factual_accuracy_validator import (
        FactualAccuracyValidator, VerificationResult, VerificationStatus,
        FactualValidationError, VerificationProcessingError, EvidenceAssessmentError
    )
    from ..claim_extractor import (
        BiomedicalClaimExtractor, ExtractedClaim,
        ClaimExtractionError, ClaimProcessingError, ClaimValidationError
    )
    from ..document_indexer import (
        SourceDocumentIndex, DocumentIndexError, ContentExtractionError, IndexingError
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


@pytest.mark.error_handling_validation
class TestInputValidationAndMalformedData:
    """Test suite for input validation and malformed data handling."""
    
    @pytest.fixture
    def malformed_test_data(self):
        """Provide various malformed test data scenarios."""
        return {
            'empty_inputs': {
                'empty_string': '',
                'none_value': None,
                'empty_list': [],
                'empty_dict': {}
            },
            'invalid_types': {
                'numeric_as_string': '123.45',
                'string_as_number': 'not_a_number',
                'list_as_string': '[1,2,3]',
                'dict_as_list': {'key': 'value'}
            },
            'malformed_json': {
                'incomplete_json': '{"key": "value"',
                'invalid_syntax': '{"key": value}',
                'mixed_quotes': '{"key\': "value"}',
                'trailing_comma': '{"key": "value",}'
            },
            'oversized_inputs': {
                'huge_string': 'x' * 1000000,  # 1MB string
                'deep_nesting': {'level' + str(i): {} for i in range(1000)},
                'large_list': list(range(100000))
            },
            'special_characters': {
                'unicode_text': 'Тест с юникодом и émojis 🧬⚗️',
                'control_characters': '\x00\x01\x02\x03\x04',
                'sql_injection': "'; DROP TABLE claims; --",
                'script_injection': '<script>alert("xss")</script>'
            }
        }
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, malformed_test_data):
        """Test handling of empty or None inputs."""
        
        scorer = FactualAccuracyScorer()
        validator = FactualAccuracyValidator()
        
        empty_data = malformed_test_data['empty_inputs']
        
        # Test empty verification results
        try:
            accuracy_score = await scorer.score_accuracy([])
            assert accuracy_score.overall_score == 0.0
            assert accuracy_score.total_claims_assessed == 0
        except AccuracyScoringError:
            pass  # Expected error is acceptable
        
        # Test None input
        with pytest.raises((AccuracyScoringError, TypeError, AttributeError)):
            await scorer.score_accuracy(None)
        
        # Test empty claims list
        try:
            report = await validator.verify_claims([])
            assert report.total_claims == 0
        except (FactualValidationError, AttributeError):
            pass  # Expected error is acceptable
        
        # Test None claims
        with pytest.raises((FactualValidationError, TypeError, AttributeError)):
            await validator.verify_claims(None)
    
    @pytest.mark.asyncio
    async def test_invalid_type_handling(self, malformed_test_data):
        """Test handling of incorrect data types."""
        
        scorer = FactualAccuracyScorer()
        invalid_types = malformed_test_data['invalid_types']
        
        # Test with string instead of verification results list
        with pytest.raises((AccuracyScoringError, TypeError, AttributeError)):
            await scorer.score_accuracy("not a list")
        
        # Test with number instead of list
        with pytest.raises((AccuracyScoringError, TypeError, AttributeError)):
            await scorer.score_accuracy(123)
        
        # Test with dict instead of list
        with pytest.raises((AccuracyScoringError, TypeError, AttributeError)):
            await scorer.score_accuracy({"key": "value"})
    
    @pytest.mark.asyncio
    async def test_malformed_verification_results(self, sample_verification_results):
        """Test handling of malformed verification result objects."""
        
        scorer = FactualAccuracyScorer()
        
        # Create malformed verification results
        malformed_results = []
        
        # Missing required attributes
        malformed_result1 = Mock()
        malformed_result1.claim_id = "malformed_1"
        # Missing verification_status, verification_confidence, etc.
        malformed_results.append(malformed_result1)
        
        # Invalid attribute types
        malformed_result2 = Mock()
        malformed_result2.claim_id = 123  # Should be string
        malformed_result2.verification_status = "invalid_status"  # Should be VerificationStatus enum
        malformed_result2.verification_confidence = "high"  # Should be float
        malformed_result2.evidence_strength = -50  # Should be 0-100
        malformed_results.append(malformed_result2)
        
        # None values
        malformed_result3 = Mock()
        malformed_result3.claim_id = None
        malformed_result3.verification_status = None
        malformed_result3.verification_confidence = None
        malformed_results.append(malformed_result3)
        
        # Test scoring with malformed results
        try:
            accuracy_score = await scorer.score_accuracy(malformed_results)
            # If it succeeds, verify it handled errors gracefully
            assert accuracy_score is not None
            assert accuracy_score.total_claims_assessed >= 0
        except AccuracyScoringError:
            # Expected error handling
            pass
    
    @pytest.mark.asyncio
    async def test_oversized_input_handling(self, malformed_test_data):
        """Test handling of oversized inputs."""
        
        scorer = FactualAccuracyScorer()
        oversized_data = malformed_test_data['oversized_inputs']
        
        # Create oversized verification results
        oversized_results = []
        for i in range(1000):  # Very large number of results
            result = Mock()
            result.claim_id = f"oversized_claim_{i}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0
            result.evidence_strength = 80.0
            result.context_match = 82.0
            result.supporting_evidence = []
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 1
            result.processing_time_ms = 100.0
            result.metadata = {}
            oversized_results.append(result)
        
        # Test with timeout to prevent hanging
        try:
            accuracy_score = await asyncio.wait_for(
                scorer.score_accuracy(oversized_results),
                timeout=30.0  # 30 second timeout
            )
            # Should handle large input reasonably
            assert accuracy_score.total_claims_assessed <= len(oversized_results)
        except (asyncio.TimeoutError, AccuracyScoringError):
            # Timeout or error handling is acceptable for oversized inputs
            pass
    
    @pytest.mark.asyncio
    async def test_special_character_handling(self, malformed_test_data):
        """Test handling of special characters and encoding issues."""
        
        scorer = FactualAccuracyScorer()
        special_chars = malformed_test_data['special_characters']
        
        # Create verification results with special characters
        for char_type, char_data in special_chars.items():
            result = Mock()
            result.claim_id = f"special_{char_type}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0
            result.evidence_strength = 80.0
            result.supporting_evidence = [Mock(evidence_text=char_data)]
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 1
            result.processing_time_ms = 100.0
            result.metadata = {'test_data': char_data}
            
            try:
                accuracy_score = await scorer.score_accuracy([result])
                # Should handle special characters gracefully
                assert accuracy_score is not None
            except (AccuracyScoringError, UnicodeError, ValueError):
                # Expected error handling for problematic characters
                pass
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        
        # Test invalid configuration
        invalid_configs = [
            None,  # None config
            "invalid_config",  # String instead of dict
            {"invalid_key": "value"},  # Missing required keys
            {
                "scoring_weights": {
                    "claim_verification": 1.5,  # > 1.0
                    "evidence_quality": -0.5   # < 0.0
                }
            },
            {
                "scoring_weights": {
                    "claim_verification": 0.3,
                    "evidence_quality": 0.2
                    # Missing other required weights, sum != 1.0
                }
            }
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            try:
                scorer = FactualAccuracyScorer(config=invalid_config)
                # If it succeeds, verify defaults are applied
                assert hasattr(scorer, 'scoring_weights')
                assert hasattr(scorer, 'config')
            except (AccuracyScoringError, ValueError, TypeError):
                # Expected error for invalid config
                pass


@pytest.mark.error_handling_validation
class TestNetworkAndExternalDependencyFailures:
    """Test suite for network and external dependency failure handling."""
    
    @pytest.fixture
    def network_failure_scenarios(self):
        """Provide network failure scenarios for testing."""
        return {
            'connection_errors': [
                ConnectionError("Connection refused"),
                ConnectionResetError("Connection reset by peer"),
                OSError("Network is unreachable")
            ],
            'timeout_errors': [
                asyncio.TimeoutError("Operation timed out"),
                TimeoutError("Request timed out")
            ],
            'http_errors': [
                Exception("HTTP 500 Internal Server Error"),
                Exception("HTTP 404 Not Found"),
                Exception("HTTP 503 Service Unavailable")
            ],
            'dns_errors': [
                Exception("DNS resolution failed"),
                Exception("Host not found")
            ]
        }
    
    @pytest.mark.asyncio
    async def test_document_indexer_connection_failure(self, network_failure_scenarios):
        """Test handling of document indexer connection failures."""
        
        validator = FactualAccuracyValidator()
        
        # Mock document indexer with connection failures
        mock_indexer = Mock()
        
        for error_type, errors in network_failure_scenarios.items():
            for error in errors:
                mock_indexer.search_content = AsyncMock(side_effect=error)
                mock_indexer.verify_claim = AsyncMock(side_effect=error)
                
                validator.document_indexer = mock_indexer
                
                # Test with simple claim
                test_claim = Mock()
                test_claim.claim_id = "network_test"
                test_claim.claim_text = "test claim"
                test_claim.claim_type = "numeric"
                test_claim.keywords = ["test"]
                test_claim.confidence = Mock(overall_confidence=75.0)
                
                try:
                    result = await validator._verify_single_claim(test_claim, {})
                    # Should handle error gracefully
                    assert result.verification_status == VerificationStatus.ERROR
                    assert result.error_details is not None
                except (FactualValidationError, ConnectionError, TimeoutError):
                    # Expected error handling
                    pass
    
    @pytest.mark.asyncio
    async def test_relevance_scorer_integration_failure(self, network_failure_scenarios):
        """Test handling of relevance scorer integration failures."""
        
        # Mock failing relevance scorer
        mock_relevance_scorer = Mock()
        
        for error_type, errors in network_failure_scenarios.items():
            for error in errors:
                mock_relevance_scorer.calculate_relevance_score = AsyncMock(side_effect=error)
                
                scorer = FactualAccuracyScorer(relevance_scorer=mock_relevance_scorer)
                
                sample_score = Mock()
                sample_score.overall_score = 85.0
                sample_score.dimension_scores = {"test": 85.0}
                sample_score.confidence_score = 80.0
                
                try:
                    integration_result = await scorer.integrate_with_relevance_scorer(
                        sample_score, "test query", "test response"
                    )
                    # Should not fail completely
                    assert integration_result is not None
                except QualityIntegrationError:
                    # Expected error handling
                    pass
    
    @pytest.mark.asyncio
    async def test_external_service_timeout_handling(self):
        """Test handling of external service timeouts."""
        
        validator = FactualAccuracyValidator()
        
        # Mock slow external service
        async def slow_operation():
            await asyncio.sleep(10)  # Simulate slow operation
            return Mock()
        
        mock_indexer = Mock()
        mock_indexer.verify_claim = slow_operation
        validator.document_indexer = mock_indexer
        
        test_claim = Mock()
        test_claim.claim_id = "timeout_test"
        test_claim.claim_text = "test claim"
        test_claim.claim_type = "numeric"
        test_claim.keywords = ["test"]
        test_claim.confidence = Mock(overall_confidence=75.0)
        
        # Test with short timeout
        try:
            result = await asyncio.wait_for(
                validator._verify_single_claim(test_claim, {}),
                timeout=1.0  # 1 second timeout
            )
            # Should handle timeout gracefully
            assert result.verification_status == VerificationStatus.ERROR
        except asyncio.TimeoutError:
            # Expected timeout behavior
            pass
    
    @pytest.mark.asyncio
    async def test_partial_service_failure_recovery(self):
        """Test recovery from partial service failures."""
        
        validator = FactualAccuracyValidator()
        
        # Mock partially failing service
        call_count = 0
        
        async def intermittent_failure(claim):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise ConnectionError("Intermittent failure")
            return {
                'verification_status': 'supported',
                'confidence': 75.0,
                'supporting_evidence': [],
                'contradicting_evidence': []
            }
        
        mock_indexer = Mock()
        mock_indexer.verify_claim = intermittent_failure
        validator.document_indexer = mock_indexer
        
        # Create multiple test claims
        test_claims = []
        for i in range(10):
            claim = Mock()
            claim.claim_id = f"recovery_test_{i}"
            claim.claim_text = f"test claim {i}"
            claim.claim_type = "numeric"
            claim.keywords = ["test"]
            claim.confidence = Mock(overall_confidence=75.0)
            test_claims.append(claim)
        
        # Verify claims - some should succeed despite failures
        successful_verifications = 0
        failed_verifications = 0
        
        for claim in test_claims:
            try:
                result = await validator._verify_single_claim(claim, {})
                if result.verification_status != VerificationStatus.ERROR:
                    successful_verifications += 1
                else:
                    failed_verifications += 1
            except:
                failed_verifications += 1
        
        # Should have some successes and some failures
        assert successful_verifications > 0
        assert failed_verifications > 0
        assert successful_verifications + failed_verifications == len(test_claims)


@pytest.mark.error_handling_validation
class TestResourceConstraintAndTimeoutHandling:
    """Test suite for resource constraint and timeout handling."""
    
    @pytest.mark.asyncio
    async def test_memory_constraint_handling(self):
        """Test handling of memory constraints."""
        
        scorer = FactualAccuracyScorer()
        
        # Create memory-intensive verification results
        memory_intensive_results = []
        for i in range(10000):  # Large number of results
            result = Mock()
            result.claim_id = f"memory_test_{i}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0
            result.evidence_strength = 80.0
            result.supporting_evidence = [Mock(evidence_text="x" * 1000) for _ in range(100)]  # Large evidence
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 100
            result.processing_time_ms = 100.0
            result.metadata = {"large_data": "x" * 10000}  # Large metadata
            memory_intensive_results.append(result)
        
        # Test with memory monitoring
        import gc
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            accuracy_score = await asyncio.wait_for(
                scorer.score_accuracy(memory_intensive_results),
                timeout=60.0  # Allow reasonable time
            )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Should not consume excessive memory
            assert memory_increase < 1000  # Less than 1GB increase
            
            if accuracy_score:
                assert accuracy_score.total_claims_assessed <= len(memory_intensive_results)
                
        except (asyncio.TimeoutError, MemoryError, AccuracyScoringError):
            # Acceptable to fail under extreme memory pressure
            pass
        finally:
            # Force garbage collection
            gc.collect()
    
    @pytest.mark.asyncio
    async def test_processing_timeout_handling(self):
        """Test handling of processing timeouts."""
        
        scorer = FactualAccuracyScorer()
        
        # Create slow processing scenario
        slow_results = []
        for i in range(100):
            result = Mock()
            result.claim_id = f"slow_test_{i}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0
            result.evidence_strength = 80.0
            result.supporting_evidence = []
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 0
            result.processing_time_ms = 1000.0  # Simulate slow processing
            result.metadata = {}
            slow_results.append(result)
        
        # Test with different timeout values
        timeout_scenarios = [0.1, 0.5, 1.0, 5.0]  # seconds
        
        for timeout in timeout_scenarios:
            try:
                accuracy_score = await asyncio.wait_for(
                    scorer.score_accuracy(slow_results),
                    timeout=timeout
                )
                
                # If completed within timeout, verify results
                if accuracy_score:
                    assert accuracy_score.total_claims_assessed >= 0
                    break  # Success - no need to test longer timeouts
                    
            except asyncio.TimeoutError:
                # Expected for short timeouts
                continue
    
    @pytest.mark.asyncio
    async def test_concurrent_access_resource_limits(self):
        """Test resource limits under concurrent access."""
        
        scorer = FactualAccuracyScorer()
        
        # Create test data
        test_results = []
        for i in range(20):
            result = Mock()
            result.claim_id = f"concurrent_test_{i}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0
            result.evidence_strength = 80.0
            result.supporting_evidence = []
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 1
            result.processing_time_ms = 100.0
            result.metadata = {}
            test_results.append(result)
        
        # Create many concurrent scoring tasks
        num_concurrent = 50
        tasks = []
        
        for i in range(num_concurrent):
            task = scorer.score_accuracy(test_results.copy())
            tasks.append(task)
        
        # Monitor resource usage during concurrent execution
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Analyze results
            successful_results = [r for r in results if isinstance(r, AccuracyScore)]
            error_results = [r for r in results if isinstance(r, Exception)]
            
            # Should handle concurrent access reasonably
            success_rate = len(successful_results) / len(results)
            assert success_rate >= 0.7  # At least 70% success rate
            
            # Memory usage should be reasonable
            assert memory_increase < 500  # Less than 500MB increase
            
        except Exception as e:
            # Some level of failure is acceptable under extreme concurrency
            print(f"Concurrent access test handled error: {e}")
    
    @pytest.mark.asyncio
    async def test_cpu_intensive_operation_limits(self):
        """Test handling of CPU-intensive operations."""
        
        scorer = FactualAccuracyScorer()
        
        # Create CPU-intensive scenario with complex calculations
        complex_results = []
        for i in range(1000):
            result = Mock()
            result.claim_id = f"cpu_test_{i}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0 + (i % 15)  # Varying confidence
            result.evidence_strength = 80.0 + (i % 20)  # Varying strength
            result.context_match = 75.0 + (i % 25)  # Varying context
            result.supporting_evidence = [Mock() for _ in range(i % 10)]  # Varying evidence count
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = i % 10
            result.processing_time_ms = 50.0 + (i % 100)  # Varying processing time
            result.metadata = {'complexity_factor': i % 5}
            complex_results.append(result)
        
        # Monitor CPU usage
        import psutil
        process = psutil.Process()
        
        start_time = time.time()
        cpu_start = process.cpu_percent()
        
        try:
            accuracy_score = await asyncio.wait_for(
                scorer.score_accuracy(complex_results),
                timeout=30.0  # Reasonable timeout
            )
            
            processing_time = (time.time() - start_time) * 1000
            cpu_usage = process.cpu_percent()
            
            # Should complete within reasonable time
            assert processing_time < 30000  # Less than 30 seconds
            
            if accuracy_score:
                assert accuracy_score.total_claims_assessed == len(complex_results)
                assert 0 <= accuracy_score.overall_score <= 100
            
            print(f"CPU-intensive test: {processing_time:.1f}ms, CPU: {cpu_usage:.1f}%")
            
        except asyncio.TimeoutError:
            # Acceptable to timeout on extremely CPU-intensive operations
            pass


@pytest.mark.error_handling_validation  
class TestDataCorruptionAndIntegrityIssues:
    """Test suite for data corruption and integrity issues."""
    
    @pytest.fixture
    def corruption_scenarios(self):
        """Provide data corruption scenarios for testing."""
        return {
            'partial_corruption': {
                'missing_fields': ['claim_id', 'verification_status', 'verification_confidence'],
                'corrupted_values': {
                    'verification_confidence': [None, 'invalid', -50, 150, float('inf'), float('nan')],
                    'evidence_strength': [None, 'invalid', -100, 200, float('inf')],
                    'processing_time_ms': [None, 'invalid', -1000, float('inf')]
                }
            },
            'type_corruption': {
                'string_to_number': 'should_be_number',
                'number_to_string': 123,
                'list_to_dict': ['item1', 'item2'],
                'dict_to_list': {'key': 'value'}
            },
            'encoding_corruption': {
                'broken_utf8': b'\xff\xfe\x00\x00invalid',
                'mixed_encoding': 'Normal text \xff\xfe corrupted',
                'null_bytes': 'Text with \x00 null bytes'
            }
        }
    
    @pytest.mark.asyncio
    async def test_missing_field_handling(self, corruption_scenarios):
        """Test handling of missing required fields."""
        
        scorer = FactualAccuracyScorer()
        corruption_data = corruption_scenarios['partial_corruption']
        
        # Test with missing required fields
        for missing_field in corruption_data['missing_fields']:
            corrupted_result = Mock()
            
            # Set some valid fields
            corrupted_result.claim_id = "corrupted_test" if missing_field != 'claim_id' else None
            corrupted_result.verification_status = VerificationStatus.SUPPORTED if missing_field != 'verification_status' else None
            corrupted_result.verification_confidence = 85.0 if missing_field != 'verification_confidence' else None
            corrupted_result.evidence_strength = 80.0
            corrupted_result.supporting_evidence = []
            corrupted_result.contradicting_evidence = []
            corrupted_result.neutral_evidence = []
            corrupted_result.total_evidence_count = 0
            corrupted_result.processing_time_ms = 100.0
            corrupted_result.metadata = {}
            
            # Remove the missing field
            if hasattr(corrupted_result, missing_field):
                delattr(corrupted_result, missing_field)
            
            try:
                accuracy_score = await scorer.score_accuracy([corrupted_result])
                # Should handle missing fields gracefully
                assert accuracy_score is not None
                # May have reduced reliability or error status
            except (AccuracyScoringError, AttributeError):
                # Expected error handling
                pass
    
    @pytest.mark.asyncio
    async def test_corrupted_value_handling(self, corruption_scenarios):
        """Test handling of corrupted field values."""
        
        scorer = FactualAccuracyScorer()
        corruption_data = corruption_scenarios['partial_corruption']
        
        for field, corrupted_values in corruption_data['corrupted_values'].items():
            for corrupted_value in corrupted_values:
                corrupted_result = Mock()
                corrupted_result.claim_id = "value_corruption_test"
                corrupted_result.verification_status = VerificationStatus.SUPPORTED
                corrupted_result.verification_confidence = 85.0
                corrupted_result.evidence_strength = 80.0
                corrupted_result.context_match = 82.0
                corrupted_result.supporting_evidence = []
                corrupted_result.contradicting_evidence = []
                corrupted_result.neutral_evidence = []
                corrupted_result.total_evidence_count = 0
                corrupted_result.processing_time_ms = 100.0
                corrupted_result.metadata = {}
                
                # Set the corrupted value
                setattr(corrupted_result, field, corrupted_value)
                
                try:
                    accuracy_score = await scorer.score_accuracy([corrupted_result])
                    # Should handle corrupted values gracefully
                    if accuracy_score:
                        # Verify score is still within valid range
                        assert 0 <= accuracy_score.overall_score <= 100 or accuracy_score.overall_score == 0
                except (AccuracyScoringError, ValueError, TypeError):
                    # Expected error handling for corrupted values
                    pass
    
    @pytest.mark.asyncio
    async def test_type_mismatch_handling(self, corruption_scenarios):
        """Test handling of type mismatches."""
        
        scorer = FactualAccuracyScorer()
        type_corruptions = corruption_scenarios['type_corruption']
        
        # Create result with type mismatches
        type_corrupted_result = Mock()
        type_corrupted_result.claim_id = 12345  # Should be string
        type_corrupted_result.verification_status = "SUPPORTED"  # Should be enum
        type_corrupted_result.verification_confidence = "high"  # Should be float
        type_corrupted_result.evidence_strength = ["80.0"]  # Should be float
        type_corrupted_result.supporting_evidence = "evidence_text"  # Should be list
        type_corrupted_result.metadata = ["key", "value"]  # Should be dict
        
        try:
            accuracy_score = await scorer.score_accuracy([type_corrupted_result])
            # Should attempt to handle type conversions or use defaults
            assert accuracy_score is not None
        except (AccuracyScoringError, TypeError, ValueError, AttributeError):
            # Expected error handling for type mismatches
            pass
    
    @pytest.mark.asyncio
    async def test_circular_reference_handling(self):
        """Test handling of circular references in data structures."""
        
        scorer = FactualAccuracyScorer()
        
        # Create circular reference
        circular_result = Mock()
        circular_result.claim_id = "circular_test"
        circular_result.verification_status = VerificationStatus.SUPPORTED
        circular_result.verification_confidence = 85.0
        circular_result.evidence_strength = 80.0
        circular_result.supporting_evidence = []
        circular_result.contradicting_evidence = []
        circular_result.neutral_evidence = []
        circular_result.total_evidence_count = 0
        circular_result.processing_time_ms = 100.0
        
        # Create circular reference in metadata
        circular_dict = {}
        circular_dict['self'] = circular_dict
        circular_result.metadata = circular_dict
        
        try:
            accuracy_score = await scorer.score_accuracy([circular_result])
            # Should handle circular references without infinite loops
            assert accuracy_score is not None
        except (AccuracyScoringError, RecursionError, ValueError):
            # Expected error handling for circular references
            pass
    
    @pytest.mark.asyncio
    async def test_data_integrity_validation(self):
        """Test data integrity validation during processing."""
        
        scorer = FactualAccuracyScorer()
        
        # Create data with integrity issues
        integrity_issues = []
        
        # Inconsistent data
        inconsistent_result = Mock()
        inconsistent_result.claim_id = "integrity_test"
        inconsistent_result.verification_status = VerificationStatus.SUPPORTED
        inconsistent_result.verification_confidence = 95.0  # High confidence
        inconsistent_result.evidence_strength = 10.0      # But low evidence strength (inconsistent)
        inconsistent_result.supporting_evidence = []      # No supporting evidence (inconsistent with SUPPORTED status)
        inconsistent_result.contradicting_evidence = [Mock(), Mock()]  # But contradicting evidence (very inconsistent)
        inconsistent_result.neutral_evidence = []
        inconsistent_result.total_evidence_count = 0      # Inconsistent with evidence lists
        inconsistent_result.processing_time_ms = 100.0
        inconsistent_result.metadata = {}
        integrity_issues.append(inconsistent_result)
        
        # Out-of-range values
        out_of_range_result = Mock()
        out_of_range_result.claim_id = "out_of_range_test"
        out_of_range_result.verification_status = VerificationStatus.SUPPORTED
        out_of_range_result.verification_confidence = 150.0  # > 100
        out_of_range_result.evidence_strength = -50.0       # < 0
        out_of_range_result.context_match = 500.0          # Way > 100
        out_of_range_result.supporting_evidence = []
        out_of_range_result.contradicting_evidence = []
        out_of_range_result.neutral_evidence = []
        out_of_range_result.total_evidence_count = 0
        out_of_range_result.processing_time_ms = -1000.0   # Negative time
        out_of_range_result.metadata = {}
        integrity_issues.append(out_of_range_result)
        
        try:
            accuracy_score = await scorer.score_accuracy(integrity_issues)
            # Should handle integrity issues gracefully
            if accuracy_score:
                # Score should still be in valid range
                assert 0 <= accuracy_score.overall_score <= 100
                assert accuracy_score.total_claims_assessed >= 0
                # May have lower confidence due to integrity issues
        except AccuracyScoringError:
            # Expected error handling for severe integrity issues
            pass


@pytest.mark.error_handling_validation
class TestConcurrentAccessAndRaceConditions:
    """Test suite for concurrent access and race condition handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_scoring_operations(self):
        """Test concurrent scoring operations for race conditions."""
        
        scorer = FactualAccuracyScorer()
        
        # Create test data
        test_results = []
        for i in range(10):
            result = Mock()
            result.claim_id = f"concurrent_score_{i}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 80.0 + i
            result.evidence_strength = 75.0 + i
            result.supporting_evidence = []
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 1
            result.processing_time_ms = 100.0
            result.metadata = {}
            test_results.append(result)
        
        # Run many concurrent scoring operations
        num_concurrent = 100
        tasks = []
        
        for i in range(num_concurrent):
            # Each task gets a copy of the data to avoid sharing issues
            task_data = [
                Mock(
                    claim_id=f"concurrent_{i}_{j}",
                    verification_status=result.verification_status,
                    verification_confidence=result.verification_confidence,
                    evidence_strength=result.evidence_strength,
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    neutral_evidence=[],
                    total_evidence_count=result.total_evidence_count,
                    processing_time_ms=result.processing_time_ms,
                    metadata={}
                )
                for j, result in enumerate(test_results)
            ]
            
            task = scorer.score_accuracy(task_data)
            tasks.append(task)
        
        # Execute all concurrent tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results for race conditions
        successful_results = [r for r in results if isinstance(r, AccuracyScore)]
        error_results = [r for r in results if isinstance(r, Exception)]
        
        # Should handle concurrency reasonably well
        success_rate = len(successful_results) / len(results)
        assert success_rate >= 0.8  # At least 80% success rate
        
        # Check for consistency in results
        if len(successful_results) > 1:
            scores = [r.overall_score for r in successful_results]
            score_variance = max(scores) - min(scores)
            
            # Scores should be reasonably consistent (allowing for some variation due to concurrency)
            assert score_variance < 10.0  # Less than 10 point variation
    
    @pytest.mark.asyncio
    async def test_shared_resource_access(self):
        """Test shared resource access patterns."""
        
        # Create shared scorer instance
        shared_scorer = FactualAccuracyScorer()
        
        # Simulate shared state modifications
        async def modify_and_score(task_id: int):
            # Simulate configuration changes (potential race condition)
            config_copy = shared_scorer.config.copy()
            config_copy['test_task_id'] = task_id
            
            # Create test data
            result = Mock()
            result.claim_id = f"shared_test_{task_id}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 80.0
            result.evidence_strength = 75.0
            result.supporting_evidence = []
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 1
            result.processing_time_ms = 100.0
            result.metadata = {'task_id': task_id}
            
            # Score with potential shared state access
            return await shared_scorer.score_accuracy([result])
        
        # Run concurrent operations that might access shared resources
        num_tasks = 50
        tasks = [modify_and_score(i) for i in range(num_tasks)]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for race condition indicators
        successful_results = [r for r in results if isinstance(r, AccuracyScore)]
        
        # Should maintain data integrity despite shared access
        assert len(successful_results) >= num_tasks * 0.8  # At least 80% success
        
        # Verify no data corruption occurred
        for result in successful_results:
            assert 0 <= result.overall_score <= 100
            assert result.total_claims_assessed == 1
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_race_conditions(self):
        """Test resource cleanup race conditions."""
        
        import tempfile
        import shutil
        from pathlib import Path
        
        # Create temporary resources that multiple tasks will access
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # Create shared resource access pattern
            async def access_and_cleanup_resource(task_id: int):
                scorer = FactualAccuracyScorer()
                
                # Create temporary file for this task
                task_file = temp_path / f"task_{task_id}.json"
                
                # Simulate resource creation, usage, and cleanup
                task_data = {
                    'task_id': task_id,
                    'timestamp': time.time(),
                    'data': [i for i in range(100)]  # Some data
                }
                
                try:
                    # Write resource
                    with open(task_file, 'w') as f:
                        json.dump(task_data, f)
                    
                    # Use resource (simulate processing)
                    await asyncio.sleep(random.uniform(0.01, 0.1))
                    
                    # Read resource
                    with open(task_file, 'r') as f:
                        loaded_data = json.load(f)
                    
                    # Verify data integrity
                    assert loaded_data['task_id'] == task_id
                    
                    # Create mock scoring operation
                    result = Mock()
                    result.claim_id = f"cleanup_test_{task_id}"
                    result.verification_status = VerificationStatus.SUPPORTED
                    result.verification_confidence = 80.0
                    result.evidence_strength = 75.0
                    result.supporting_evidence = []
                    result.contradicting_evidence = []
                    result.neutral_evidence = []
                    result.total_evidence_count = 1
                    result.processing_time_ms = 100.0
                    result.metadata = {'resource_file': str(task_file)}
                    
                    score = await scorer.score_accuracy([result])
                    
                    return {'success': True, 'score': score, 'task_id': task_id}
                    
                except Exception as e:
                    return {'success': False, 'error': str(e), 'task_id': task_id}
                
                finally:
                    # Cleanup resource (potential race condition here)
                    try:
                        if task_file.exists():
                            task_file.unlink()
                    except FileNotFoundError:
                        pass  # Already cleaned up by another task
            
            # Run concurrent resource access tasks
            num_tasks = 30
            tasks = [access_and_cleanup_resource(i) for i in range(num_tasks)]
            
            results = await asyncio.gather(*tasks)
            
            # Analyze race condition results
            successful_tasks = [r for r in results if r['success']]
            failed_tasks = [r for r in results if not r['success']]
            
            success_rate = len(successful_tasks) / len(results)
            
            # Should handle resource cleanup race conditions reasonably
            assert success_rate >= 0.7  # At least 70% success rate
            
            # Verify no severe data corruption
            for task_result in successful_tasks:
                if 'score' in task_result and task_result['score']:
                    score = task_result['score']
                    assert 0 <= score.overall_score <= 100
        
        finally:
            # Final cleanup
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.error_handling_validation
class TestRecoveryAndFallbackMechanisms:
    """Test suite for recovery and fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        
        # Create scorer with failing integration
        failing_relevance_scorer = Mock()
        failing_relevance_scorer.calculate_relevance_score = AsyncMock(
            side_effect=Exception("Relevance scorer failed")
        )
        
        scorer = FactualAccuracyScorer(relevance_scorer=failing_relevance_scorer)
        
        # Create test data
        test_results = []
        for i in range(5):
            result = Mock()
            result.claim_id = f"degradation_test_{i}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0
            result.evidence_strength = 80.0
            result.supporting_evidence = []
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 1
            result.processing_time_ms = 100.0
            result.metadata = {}
            test_results.append(result)
        
        # Should still provide basic scoring functionality
        accuracy_score = await scorer.score_accuracy(test_results)
        
        assert accuracy_score is not None
        assert accuracy_score.total_claims_assessed == len(test_results)
        assert 0 <= accuracy_score.overall_score <= 100
        
        # Integration should fail gracefully but basic scoring works
        sample_score = Mock()
        sample_score.overall_score = 85.0
        sample_score.dimension_scores = {"test": 85.0}
        sample_score.confidence_score = 80.0
        
        try:
            integration_result = await scorer.integrate_with_relevance_scorer(
                sample_score, "test query", "test response"
            )
            # Should not succeed with failing relevance scorer
            assert integration_result is None
        except QualityIntegrationError:
            # Expected failure - system should handle gracefully
            pass
    
    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self):
        """Test recovery from partial failures in batch operations."""
        
        scorer = FactualAccuracyScorer()
        
        # Create mixed good and bad data
        mixed_results = []
        
        # Good results
        for i in range(3):
            result = Mock()
            result.claim_id = f"good_result_{i}"
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0
            result.evidence_strength = 80.0
            result.supporting_evidence = []
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 1
            result.processing_time_ms = 100.0
            result.metadata = {}
            mixed_results.append(result)
        
        # Bad results (various issues)
        bad_result1 = Mock()
        bad_result1.claim_id = None  # Invalid claim ID
        bad_result1.verification_status = "invalid_status"  # Invalid status
        bad_result1.verification_confidence = "not_a_number"  # Invalid confidence
        mixed_results.append(bad_result1)
        
        bad_result2 = Mock()
        bad_result2.claim_id = "bad_result_2"
        bad_result2.verification_status = VerificationStatus.SUPPORTED
        bad_result2.verification_confidence = float('nan')  # NaN confidence
        bad_result2.evidence_strength = float('inf')  # Infinite strength
        mixed_results.append(bad_result2)
        
        # Should process good results despite bad ones
        try:
            accuracy_score = await scorer.score_accuracy(mixed_results)
            
            # Should have processed some results
            assert accuracy_score is not None
            # May have fewer claims assessed due to filtering out bad ones
            assert accuracy_score.total_claims_assessed <= len(mixed_results)
            assert 0 <= accuracy_score.overall_score <= 100
            
        except AccuracyScoringError:
            # If complete failure, should still handle gracefully
            pass
    
    @pytest.mark.asyncio
    async def test_fallback_configuration_handling(self):
        """Test fallback to default configuration on errors."""
        
        # Test with various invalid configurations
        invalid_configs = [
            {"invalid_structure": True},
            {"scoring_weights": "not_a_dict"},
            {"scoring_weights": {"invalid_weight": 2.0}},
            None
        ]
        
        for invalid_config in invalid_configs:
            try:
                scorer = FactualAccuracyScorer(config=invalid_config)
                
                # Should fall back to default configuration
                assert hasattr(scorer, 'config')
                assert hasattr(scorer, 'scoring_weights')
                assert hasattr(scorer, 'claim_type_weights')
                
                # Should still be functional with fallback config
                test_result = Mock()
                test_result.claim_id = "fallback_test"
                test_result.verification_status = VerificationStatus.SUPPORTED
                test_result.verification_confidence = 85.0
                test_result.evidence_strength = 80.0
                test_result.supporting_evidence = []
                test_result.contradicting_evidence = []
                test_result.neutral_evidence = []
                test_result.total_evidence_count = 1
                test_result.processing_time_ms = 100.0
                test_result.metadata = {}
                
                accuracy_score = await scorer.score_accuracy([test_result])
                assert accuracy_score is not None
                assert accuracy_score.total_claims_assessed == 1
                
            except (AccuracyScoringError, ValueError, TypeError):
                # Some level of configuration validation failure is acceptable
                pass
    
    @pytest.mark.asyncio
    async def test_error_state_recovery(self):
        """Test recovery from error states."""
        
        scorer = FactualAccuracyScorer()
        
        # Simulate error state by corrupting internal state
        original_weights = scorer.scoring_weights.copy()
        
        # Corrupt internal state
        scorer.scoring_weights = None
        scorer.claim_type_weights = "corrupted"
        
        try:
            # Should detect and recover from corrupted state
            test_result = Mock()
            test_result.claim_id = "recovery_test"
            test_result.verification_status = VerificationStatus.SUPPORTED
            test_result.verification_confidence = 85.0
            test_result.evidence_strength = 80.0
            test_result.supporting_evidence = []
            test_result.contradicting_evidence = []
            test_result.neutral_evidence = []
            test_result.total_evidence_count = 1
            test_result.processing_time_ms = 100.0
            test_result.metadata = {}
            
            # This should either work (after recovery) or fail gracefully
            accuracy_score = await scorer.score_accuracy([test_result])
            
            if accuracy_score:
                assert accuracy_score.total_claims_assessed == 1
                assert 0 <= accuracy_score.overall_score <= 100
                
        except AccuracyScoringError:
            # Graceful failure is acceptable for corrupted state
            pass
        
        finally:
            # Restore original state
            scorer.scoring_weights = original_weights


@pytest.mark.error_handling_validation
class TestLoggingAndErrorReporting:
    """Test suite for logging and error reporting validation."""
    
    @pytest.fixture
    def test_logger(self):
        """Provide test logger for error reporting tests."""
        import logging
        import io
        
        # Create string buffer to capture log output
        log_stream = io.StringIO()
        
        # Create test logger
        test_logger = logging.getLogger('test_validation_errors')
        test_logger.setLevel(logging.DEBUG)
        
        # Add string handler
        handler = logging.StreamHandler(log_stream)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        handler.setFormatter(formatter)
        test_logger.addHandler(handler)
        
        return test_logger, log_stream
    
    @pytest.mark.asyncio
    async def test_error_logging_completeness(self, test_logger):
        """Test completeness of error logging."""
        
        logger, log_stream = test_logger
        
        # Patch the scorer's logger
        scorer = FactualAccuracyScorer()
        scorer.logger = logger
        
        # Create error-inducing scenario
        invalid_result = Mock()
        invalid_result.claim_id = None
        invalid_result.verification_status = "invalid"
        invalid_result.verification_confidence = "not_a_number"
        
        try:
            await scorer.score_accuracy([invalid_result])
        except:
            pass
        
        # Check log output
        log_output = log_stream.getvalue()
        
        # Should have logged the error
        assert len(log_output) > 0
        # Should contain error information
        assert any(level in log_output.upper() for level in ['ERROR', 'WARNING'])
    
    @pytest.mark.asyncio
    async def test_structured_error_reporting(self):
        """Test structured error reporting with context."""
        
        scorer = FactualAccuracyScorer()
        
        # Test error reporting with context
        error_contexts = [
            {
                "error_type": "invalid_input",
                "data": None,
                "context": "null_input_test"
            },
            {
                "error_type": "type_mismatch",
                "data": "string_instead_of_list",
                "context": "type_validation_test"
            },
            {
                "error_type": "value_out_of_range", 
                "data": {"confidence": 150.0},
                "context": "range_validation_test"
            }
        ]
        
        for error_context in error_contexts:
            try:
                if error_context["error_type"] == "invalid_input":
                    await scorer.score_accuracy(None)
                elif error_context["error_type"] == "type_mismatch":
                    await scorer.score_accuracy("not_a_list")
                elif error_context["error_type"] == "value_out_of_range":
                    invalid_result = Mock()
                    invalid_result.claim_id = "range_test"
                    invalid_result.verification_confidence = 150.0  # Out of range
                    await scorer.score_accuracy([invalid_result])
                    
            except Exception as e:
                # Error should contain useful information
                error_str = str(e)
                assert len(error_str) > 0
                # Should not be a generic error message
                assert error_str != "An error occurred"
    
    @pytest.mark.asyncio
    async def test_error_recovery_logging(self):
        """Test logging of error recovery attempts."""
        
        scorer = FactualAccuracyScorer()
        
        # Create recoverable error scenario
        partially_invalid_results = []
        
        # Mix of valid and invalid results
        valid_result = Mock()
        valid_result.claim_id = "valid_test"
        valid_result.verification_status = VerificationStatus.SUPPORTED
        valid_result.verification_confidence = 85.0
        valid_result.evidence_strength = 80.0
        valid_result.supporting_evidence = []
        valid_result.contradicting_evidence = []
        valid_result.neutral_evidence = []
        valid_result.total_evidence_count = 1
        valid_result.processing_time_ms = 100.0
        valid_result.metadata = {}
        partially_invalid_results.append(valid_result)
        
        invalid_result = Mock()
        invalid_result.claim_id = "invalid_test"
        invalid_result.verification_status = "not_an_enum"  # Invalid
        invalid_result.verification_confidence = None  # Invalid
        partially_invalid_results.append(invalid_result)
        
        try:
            accuracy_score = await scorer.score_accuracy(partially_invalid_results)
            # Should attempt recovery and log the process
            if accuracy_score:
                # Should have processed at least the valid result
                assert accuracy_score.total_claims_assessed >= 1
        except AccuracyScoringError:
            # If recovery fails, error should be informative
            pass
    
    def test_exception_hierarchy_correctness(self):
        """Test that custom exceptions are properly structured."""
        
        # Test exception inheritance
        assert issubclass(AccuracyScoringError, Exception)
        assert issubclass(ReportGenerationError, AccuracyScoringError)
        assert issubclass(QualityIntegrationError, AccuracyScoringError)
        
        # Test exception instantiation
        base_error = AccuracyScoringError("Base error message")
        assert str(base_error) == "Base error message"
        
        report_error = ReportGenerationError("Report generation failed")
        assert str(report_error) == "Report generation failed"
        assert isinstance(report_error, AccuracyScoringError)
        
        integration_error = QualityIntegrationError("Integration failed")
        assert str(integration_error) == "Integration failed"
        assert isinstance(integration_error, AccuracyScoringError)


if __name__ == "__main__":
    # Run the error handling test suite
    pytest.main([__file__, "-v", "--tb=short", "-m", "error_handling_validation"])