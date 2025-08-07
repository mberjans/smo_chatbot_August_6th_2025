#!/usr/bin/env python3
"""
Comprehensive Test Fixtures for Factual Accuracy Validation System.

This module provides comprehensive test fixtures, mock objects, and test data
for testing the entire factual accuracy validation pipeline including:
- AccuracyScorer testing
- FactualAccuracyValidator testing
- ClaimExtractor testing
- DocumentIndexer testing
- Integration testing
- Performance testing

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import json
import time
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import random
import string


# Import the modules we're testing
try:
    from ..accuracy_scorer import (
        FactualAccuracyScorer, AccuracyScore, AccuracyReport, AccuracyMetrics,
        AccuracyGrade, score_verification_results, generate_accuracy_report
    )
    from ..factual_accuracy_validator import (
        FactualAccuracyValidator, VerificationResult, VerificationStatus,
        EvidenceItem, VerificationReport, FactualValidationError
    )
    from ..claim_extractor import (
        BiomedicalClaimExtractor, ExtractedClaim, ClaimContext, ClaimConfidence
    )
    from ..document_indexer import (
        SourceDocumentIndex, IndexedContent, NumericFact, ScientificStatement
    )
except ImportError:
    # Create mock classes for testing when imports fail
    class MockAccuracyScorer:
        pass
    class MockValidator:
        pass


# Test Data Constants
SAMPLE_BIOMEDICAL_RESPONSES = [
    "Glucose levels were significantly elevated at 150 mg/dL in diabetic patients compared to 90 mg/dL in healthy controls (p<0.05).",
    "The metabolomics analysis revealed increased levels of branched-chain amino acids in patients with insulin resistance.",
    "LC-MS/MS analysis was performed using an Agilent 6495 triple quadrupole mass spectrometer with electrospray ionization.",
    "Serum samples were collected after 12-hour fasting and stored at -80°C until analysis.",
    "Principal component analysis showed clear separation between diabetic and control groups (R² = 0.85)."
]

SAMPLE_CLAIMS_DATA = [
    {
        "claim_type": "numeric",
        "text": "Glucose levels were 150 mg/dL in diabetic patients",
        "numeric_values": [150.0],
        "units": ["mg/dL"],
        "confidence": 85.0
    },
    {
        "claim_type": "qualitative", 
        "text": "Metabolomics analysis revealed increased levels of amino acids",
        "relationships": [{"type": "increase", "subject": "amino acids", "object": "insulin resistance"}],
        "confidence": 75.0
    },
    {
        "claim_type": "methodological",
        "text": "LC-MS/MS analysis was performed using Agilent 6495",
        "methodology": "LC-MS/MS",
        "equipment": "Agilent 6495",
        "confidence": 90.0
    }
]

SAMPLE_EVIDENCE_DATA = [
    {
        "source_document": "diabetes_study_2024.pdf",
        "evidence_text": "Mean glucose concentration was 148.3 ± 12.5 mg/dL in the diabetic cohort",
        "evidence_type": "numeric",
        "confidence": 88.0,
        "page_number": 15,
        "section": "Results"
    },
    {
        "source_document": "metabolomics_review_2024.pdf", 
        "evidence_text": "Elevated branched-chain amino acids are associated with insulin resistance",
        "evidence_type": "qualitative",
        "confidence": 82.0,
        "page_number": 23,
        "section": "Discussion"
    }
]


@dataclass
class TestConfiguration:
    """Test configuration settings."""
    enable_performance_tests: bool = True
    enable_integration_tests: bool = True
    enable_mock_tests: bool = True
    test_timeout_seconds: int = 30
    performance_threshold_ms: float = 1000.0
    min_coverage_percentage: float = 90.0
    test_data_directory: Optional[Path] = None
    mock_api_responses: bool = True
    generate_test_reports: bool = True


class ValidationTestFixtures:
    """Main test fixtures provider for validation system testing."""
    
    def __init__(self, config: Optional[TestConfiguration] = None):
        """Initialize test fixtures with configuration."""
        self.config = config or TestConfiguration()
        self.temp_dir = None
        self.mock_registry = {}
        self.test_data_cache = {}
        
    def setup_test_environment(self):
        """Set up test environment with temporary directories and mock objects."""
        self.temp_dir = tempfile.mkdtemp(prefix="validation_test_")
        self.test_data_path = Path(self.temp_dir) / "test_data"
        self.test_data_path.mkdir(exist_ok=True)
        
        # Create test directories
        (self.test_data_path / "pdfs").mkdir(exist_ok=True)
        (self.test_data_path / "indexes").mkdir(exist_ok=True)
        (self.test_data_path / "reports").mkdir(exist_ok=True)
        
        return self.temp_dir
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        self.mock_registry.clear()
        self.test_data_cache.clear()


# Core Test Fixtures
@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration for all tests."""
    return TestConfiguration()


@pytest.fixture(scope="session") 
def fixtures_provider(test_config):
    """Provide main fixtures provider."""
    provider = ValidationTestFixtures(test_config)
    provider.setup_test_environment()
    yield provider
    provider.cleanup_test_environment()


@pytest.fixture
def temp_test_dir(fixtures_provider):
    """Provide temporary test directory."""
    return fixtures_provider.temp_dir


# Mock Component Fixtures
@pytest.fixture
def mock_claim_extractor():
    """Mock BiomedicalClaimExtractor for testing."""
    extractor = Mock(spec=BiomedicalClaimExtractor)
    
    # Mock extracted claims
    mock_claims = []
    for i, claim_data in enumerate(SAMPLE_CLAIMS_DATA):
        claim = Mock(spec=ExtractedClaim)
        claim.claim_id = f"test_claim_{i+1}"
        claim.claim_text = claim_data["text"]
        claim.claim_type = claim_data["claim_type"]
        claim.numeric_values = claim_data.get("numeric_values", [])
        claim.units = claim_data.get("units", [])
        claim.keywords = ["test", "metabolomics", "glucose"]
        claim.confidence = Mock(overall_confidence=claim_data["confidence"])
        claim.subject = "glucose" if "glucose" in claim_data["text"] else "metabolite"
        claim.predicate = "were" if "were" in claim_data["text"] else "revealed"
        claim.object_value = "150 mg/dL" if "150" in claim_data["text"] else "increased"
        claim.relationships = claim_data.get("relationships", [])
        mock_claims.append(claim)
    
    extractor.extract_claims = AsyncMock(return_value=mock_claims)
    extractor.get_extraction_statistics = Mock(return_value={
        "total_extractions": 5,
        "total_claims_extracted": 15,
        "average_claims_per_extraction": 3.0,
        "processing_times": {"average_ms": 150.0}
    })
    
    return extractor


@pytest.fixture
def mock_document_indexer():
    """Mock SourceDocumentIndex for testing."""
    indexer = Mock(spec=SourceDocumentIndex)
    
    # Mock document search results
    mock_search_results = []
    for evidence_data in SAMPLE_EVIDENCE_DATA:
        result = Mock()
        result.document_id = evidence_data["source_document"]
        result.content = evidence_data["evidence_text"]
        result.page_number = evidence_data["page_number"]
        result.section = evidence_data["section"]
        result.confidence = evidence_data["confidence"]
        mock_search_results.append(result)
    
    indexer.search_content = AsyncMock(return_value=mock_search_results)
    
    # Mock claim verification
    indexer.verify_claim = AsyncMock(return_value={
        "verification_status": "supported",
        "confidence": 85.0,
        "supporting_evidence": SAMPLE_EVIDENCE_DATA[:1],
        "contradicting_evidence": [],
        "related_facts": ["glucose", "diabetes", "mg/dL"],
        "verification_metadata": {"search_time_ms": 45.0}
    })
    
    indexer.get_indexing_statistics = Mock(return_value={
        "total_documents": 25,
        "total_indexed_facts": 150,
        "index_size_mb": 12.5
    })
    
    return indexer


@pytest.fixture
def mock_factual_validator(mock_document_indexer, mock_claim_extractor):
    """Mock FactualAccuracyValidator for testing."""
    validator = Mock(spec=FactualAccuracyValidator)
    validator.document_indexer = mock_document_indexer
    validator.claim_extractor = mock_claim_extractor
    
    # Mock verification results
    mock_results = []
    for i, claim_data in enumerate(SAMPLE_CLAIMS_DATA):
        evidence_items = []
        for j, evidence_data in enumerate(SAMPLE_EVIDENCE_DATA[:2]):  # Max 2 evidence per claim
            evidence = Mock(spec=EvidenceItem)
            evidence.source_document = evidence_data["source_document"]
            evidence.evidence_text = evidence_data["evidence_text"]
            evidence.evidence_type = evidence_data["evidence_type"]
            evidence.confidence = evidence_data["confidence"]
            evidence.metadata = {"page": evidence_data["page_number"]}
            evidence_items.append(evidence)
        
        result = Mock(spec=VerificationResult)
        result.claim_id = f"test_claim_{i+1}"
        result.verification_status = VerificationStatus.SUPPORTED if i % 3 != 2 else VerificationStatus.NEUTRAL
        result.verification_confidence = random.uniform(70.0, 95.0)
        result.evidence_strength = random.uniform(65.0, 90.0)
        result.context_match = random.uniform(75.0, 95.0)
        result.supporting_evidence = evidence_items if result.verification_status == VerificationStatus.SUPPORTED else []
        result.contradicting_evidence = []
        result.neutral_evidence = evidence_items if result.verification_status == VerificationStatus.NEUTRAL else []
        result.total_evidence_count = len(evidence_items)
        result.processing_time_ms = random.uniform(50.0, 200.0)
        result.verification_strategy = claim_data["claim_type"]
        result.verification_grade = "High"
        result.error_details = None
        result.metadata = {"claim_type": claim_data["claim_type"]}
        mock_results.append(result)
    
    # Mock verification report
    mock_report = Mock(spec=VerificationReport)
    mock_report.report_id = "test_verification_report"
    mock_report.total_claims = len(mock_results)
    mock_report.verification_results = mock_results
    mock_report.summary_statistics = {
        "supported_claims": len([r for r in mock_results if r.verification_status == VerificationStatus.SUPPORTED]),
        "contradicted_claims": 0,
        "neutral_claims": len([r for r in mock_results if r.verification_status == VerificationStatus.NEUTRAL]),
        "error_claims": 0,
        "average_confidence": sum(r.verification_confidence for r in mock_results) / len(mock_results)
    }
    mock_report.processing_time_ms = sum(r.processing_time_ms for r in mock_results)
    mock_report.created_timestamp = datetime.now()
    mock_report.recommendations = ["Good verification coverage", "Consider adding more sources"]
    
    validator.verify_claims = AsyncMock(return_value=mock_report)
    validator._verify_single_claim = AsyncMock(side_effect=lambda claim, config: mock_results[0])
    
    validator.get_verification_statistics = Mock(return_value={
        "total_verifications": 10,
        "total_claims_verified": 30,
        "average_claims_per_verification": 3.0,
        "processing_times": {
            "average_ms": 125.0,
            "median_ms": 110.0,
            "min_ms": 45.0,
            "max_ms": 250.0
        }
    })
    
    return validator


@pytest.fixture
def mock_accuracy_scorer():
    """Mock FactualAccuracyScorer for testing."""
    scorer = Mock(spec=FactualAccuracyScorer)
    
    # Mock accuracy score
    mock_score = Mock(spec=AccuracyScore)
    mock_score.overall_score = 82.5
    mock_score.dimension_scores = {
        "claim_verification": 85.0,
        "evidence_quality": 80.0,
        "coverage_assessment": 83.0,
        "consistency_analysis": 81.0,
        "confidence_factor": 78.0
    }
    mock_score.claim_type_scores = {
        "numeric": 88.0,
        "qualitative": 78.0,
        "methodological": 85.0
    }
    mock_score.evidence_quality_score = 80.0
    mock_score.coverage_score = 83.0
    mock_score.consistency_score = 81.0
    mock_score.confidence_score = 78.0
    mock_score.grade = AccuracyGrade.GOOD
    mock_score.total_claims_assessed = 15
    mock_score.processing_time_ms = 185.7
    mock_score.is_reliable = True
    mock_score.accuracy_percentage = "82.5%"
    mock_score.metadata = {"scoring_method": "comprehensive_weighted"}
    
    # Mock accuracy metrics
    mock_metrics = Mock(spec=AccuracyMetrics)
    mock_metrics.verification_performance = {"avg_verification_time_ms": 120.0}
    mock_metrics.scoring_performance = {"total_scoring_time_ms": 185.7}
    mock_metrics.quality_indicators = {"error_rate": 0.02, "avg_confidence": 82.5}
    mock_metrics.system_health = {"memory_efficient": True, "error_rate_acceptable": True}
    
    # Mock accuracy report
    mock_report = Mock(spec=AccuracyReport)
    mock_report.report_id = "FACR_test_report_001"
    mock_report.accuracy_score = mock_score
    mock_report.detailed_breakdown = {
        "status_distribution": {"SUPPORTED": 12, "NEUTRAL": 3, "CONTRADICTED": 0, "ERROR": 0},
        "evidence_statistics": {"total_evidence_items": 30, "avg_evidence_per_claim": 2.0},
        "confidence_distribution": {"mean": 82.5, "median": 85.0, "std_dev": 8.2}
    }
    mock_report.summary_statistics = {
        "total_claims": 15,
        "verification_rate": 1.0,
        "support_rate": 0.8,
        "contradiction_rate": 0.0,
        "high_confidence_rate": 0.73
    }
    mock_report.performance_metrics = mock_metrics
    mock_report.quality_recommendations = [
        "Overall accuracy is good - maintain current standards",
        "Consider expanding evidence sources for better coverage"
    ]
    mock_report.claims_analysis = []
    mock_report.evidence_analysis = {"total_evidence_items": 30, "unique_sources": 8}
    mock_report.coverage_analysis = {"overall_coverage_rate": 0.87, "claims_with_evidence": 13}
    mock_report.created_timestamp = datetime.now()
    mock_report.report_summary = f"Accuracy Report {mock_report.report_id} - Overall: 82.5% (Good)"
    
    scorer.score_accuracy = AsyncMock(return_value=mock_score)
    scorer.generate_comprehensive_report = AsyncMock(return_value=mock_report)
    scorer.integrate_with_relevance_scorer = AsyncMock(return_value={
        "factual_accuracy": {"overall_score": 82.5, "grade": "Good"},
        "integrated_quality": {"combined_score": 84.2, "quality_grade": "Good"}
    })
    
    scorer.get_scoring_statistics = Mock(return_value={
        "total_scorings": 5,
        "total_claims_scored": 75,
        "average_claims_per_scoring": 15.0,
        "processing_times": {"average_ms": 185.0, "median_ms": 175.0}
    })
    
    return scorer


# Test Data Fixtures
@pytest.fixture
def sample_extracted_claims():
    """Provide sample ExtractedClaim objects for testing."""
    claims = []
    
    for i, claim_data in enumerate(SAMPLE_CLAIMS_DATA):
        # Create mock claim context
        context = Mock(spec=ClaimContext)
        context.surrounding_text = f"Context for {claim_data['text']}"
        context.sentence_position = i
        context.paragraph_position = 0
        context.topic_context = "biomedical research"
        
        # Create mock claim confidence
        confidence = Mock(spec=ClaimConfidence)
        confidence.overall_confidence = claim_data["confidence"]
        confidence.linguistic_confidence = claim_data["confidence"] - 5.0
        confidence.contextual_confidence = claim_data["confidence"] + 3.0
        confidence.domain_confidence = claim_data["confidence"] + 2.0
        
        # Create mock claim
        claim = Mock(spec=ExtractedClaim)
        claim.claim_id = f"sample_claim_{i+1}"
        claim.claim_text = claim_data["text"]
        claim.claim_type = claim_data["claim_type"]
        claim.subject = "glucose" if "glucose" in claim_data["text"] else "metabolite"
        claim.predicate = "were" if "were" in claim_data["text"] else "revealed"
        claim.object_value = "150 mg/dL" if "150" in claim_data["text"] else "increased"
        claim.numeric_values = claim_data.get("numeric_values", [])
        claim.units = claim_data.get("units", [])
        claim.keywords = ["metabolomics", "glucose", "analysis"]
        claim.relationships = claim_data.get("relationships", [])
        claim.context = context
        claim.confidence = confidence
        claim.priority_score = random.uniform(70.0, 95.0)
        claim.extraction_metadata = {"source": "test", "method": "pattern_based"}
        
        claims.append(claim)
    
    return claims


@pytest.fixture 
def sample_verification_results(sample_extracted_claims):
    """Provide sample VerificationResult objects for testing."""
    results = []
    
    for i, claim in enumerate(sample_extracted_claims):
        # Create evidence items
        evidence_items = []
        for j, evidence_data in enumerate(SAMPLE_EVIDENCE_DATA):
            if j >= 2:  # Limit to 2 evidence items per claim
                break
                
            evidence = Mock(spec=EvidenceItem)
            evidence.source_document = evidence_data["source_document"]
            evidence.evidence_text = evidence_data["evidence_text"]
            evidence.evidence_type = evidence_data["evidence_type"]
            evidence.confidence = evidence_data["confidence"]
            evidence.context = f"Context for evidence {j+1}"
            evidence.metadata = {
                "page_number": evidence_data["page_number"],
                "section": evidence_data["section"]
            }
            evidence_items.append(evidence)
        
        # Create verification result
        result = Mock(spec=VerificationResult)
        result.claim_id = claim.claim_id
        result.verification_status = VerificationStatus.SUPPORTED if i % 3 != 2 else VerificationStatus.NEUTRAL
        result.verification_confidence = random.uniform(70.0, 95.0)
        result.evidence_strength = random.uniform(65.0, 90.0)
        result.context_match = random.uniform(75.0, 95.0)
        result.supporting_evidence = evidence_items if result.verification_status == VerificationStatus.SUPPORTED else []
        result.contradicting_evidence = []
        result.neutral_evidence = evidence_items if result.verification_status == VerificationStatus.NEUTRAL else []
        result.total_evidence_count = len(evidence_items)
        result.processing_time_ms = random.uniform(50.0, 200.0)
        result.verification_strategy = claim.claim_type
        result.verification_grade = "High" if result.verification_confidence >= 80 else "Moderate"
        result.error_details = None
        result.metadata = {"claim_type": claim.claim_type}
        
        results.append(result)
    
    return results


@pytest.fixture
def sample_accuracy_score():
    """Provide sample AccuracyScore for testing."""
    score = Mock(spec=AccuracyScore)
    score.overall_score = 84.7
    score.dimension_scores = {
        "claim_verification": 87.0,
        "evidence_quality": 82.0,
        "coverage_assessment": 85.0,
        "consistency_analysis": 83.0,
        "confidence_factor": 81.0
    }
    score.claim_type_scores = {
        "numeric": 89.0,
        "qualitative": 80.0,
        "methodological": 86.0
    }
    score.evidence_quality_score = 82.0
    score.coverage_score = 85.0
    score.consistency_score = 83.0
    score.confidence_score = 81.0
    score.grade = AccuracyGrade.GOOD
    score.total_claims_assessed = 12
    score.processing_time_ms = 156.3
    score.is_reliable = True
    score.accuracy_percentage = "84.7%"
    score.metadata = {"scoring_method": "comprehensive_weighted", "config_version": "1.0.0"}
    
    return score


# Performance Test Fixtures
@pytest.fixture
def performance_test_data():
    """Provide data for performance testing."""
    # Generate larger datasets for performance testing
    large_response = " ".join(SAMPLE_BIOMEDICAL_RESPONSES * 10)
    
    # Generate many claims for batch testing
    many_claims = []
    for i in range(50):  # 50 claims for performance testing
        claim_data = random.choice(SAMPLE_CLAIMS_DATA)
        claim = Mock()
        claim.claim_id = f"perf_claim_{i+1}"
        claim.claim_text = f"{claim_data['text']} (variant {i+1})"
        claim.claim_type = claim_data["claim_type"]
        claim.numeric_values = claim_data.get("numeric_values", [])
        claim.confidence = Mock(overall_confidence=random.uniform(60.0, 95.0))
        many_claims.append(claim)
    
    return {
        "large_response": large_response,
        "many_claims": many_claims,
        "performance_thresholds": {
            "single_claim_verification_ms": 500,
            "batch_verification_ms": 5000,
            "accuracy_scoring_ms": 1000,
            "report_generation_ms": 2000
        }
    }


# Error Testing Fixtures  
@pytest.fixture
def error_test_scenarios():
    """Provide error scenarios for testing."""
    return {
        "malformed_claims": [
            Mock(claim_id="bad1", claim_text=None, claim_type="numeric"),
            Mock(claim_id="bad2", claim_text="", claim_type=""),
            Mock(claim_id="bad3", claim_text="test", claim_type=None)
        ],
        "network_errors": [
            ConnectionError("Network connection failed"),
            TimeoutError("Request timed out"),
            Exception("Unexpected error")
        ],
        "data_corruption_scenarios": [
            {"corrupted_json": '{"invalid": json}'},
            {"missing_fields": {"claim_text": "test"}},  # Missing required fields
            {"wrong_types": {"claim_id": 123, "confidence": "high"}}  # Wrong data types
        ],
        "resource_constraints": {
            "high_memory_usage": True,
            "processing_timeout": 0.1,  # Very short timeout
            "concurrent_requests": 100
        }
    }


# Integration Test Fixtures
@pytest.fixture
def integration_test_environment(temp_test_dir):
    """Set up complete integration test environment."""
    env = {
        "temp_dir": temp_test_dir,
        "test_pdfs": [],
        "test_index": None,
        "test_config": {
            "test_mode": True,
            "enable_caching": False,
            "performance_tracking": True,
            "detailed_logging": True
        }
    }
    
    # Create some test PDF content
    test_pdf_content = [
        "Diabetes mellitus is characterized by elevated glucose levels above 126 mg/dL.",
        "LC-MS/MS analysis provides accurate quantification of metabolites in biological samples.",
        "Statistical analysis was performed using R software with significance set at p<0.05."
    ]
    
    for i, content in enumerate(test_pdf_content):
        pdf_path = Path(temp_test_dir) / f"test_document_{i+1}.txt"
        pdf_path.write_text(content)
        env["test_pdfs"].append(str(pdf_path))
    
    return env


# Async Test Utilities
@pytest.fixture(scope="session")
def event_loop():
    """Provide event loop for async testing."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Benchmarking Fixtures
@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities."""
    
    class PerformanceMonitor:
        def __init__(self):
            self.measurements = {}
            self.thresholds = {}
        
        def start_measurement(self, operation_name: str):
            self.measurements[operation_name] = {"start": time.time()}
        
        def end_measurement(self, operation_name: str):
            if operation_name in self.measurements:
                end_time = time.time()
                start_time = self.measurements[operation_name]["start"]
                self.measurements[operation_name]["duration"] = (end_time - start_time) * 1000
                return self.measurements[operation_name]["duration"]
            return 0
        
        def set_threshold(self, operation_name: str, threshold_ms: float):
            self.thresholds[operation_name] = threshold_ms
        
        def check_performance(self, operation_name: str) -> bool:
            if operation_name not in self.measurements or operation_name not in self.thresholds:
                return True
            
            duration = self.measurements[operation_name]["duration"]
            threshold = self.thresholds[operation_name]
            return duration <= threshold
        
        def get_report(self) -> Dict[str, Any]:
            report = {"measurements": {}, "threshold_violations": []}
            
            for op_name, measurement in self.measurements.items():
                report["measurements"][op_name] = measurement.get("duration", 0)
                
                if op_name in self.thresholds:
                    duration = measurement.get("duration", 0)
                    threshold = self.thresholds[op_name]
                    if duration > threshold:
                        report["threshold_violations"].append({
                            "operation": op_name,
                            "duration_ms": duration,
                            "threshold_ms": threshold,
                            "violation_ms": duration - threshold
                        })
            
            return report
    
    monitor = PerformanceMonitor()
    
    # Set default thresholds
    monitor.set_threshold("claim_extraction", 500.0)
    monitor.set_threshold("claim_verification", 1000.0)
    monitor.set_threshold("accuracy_scoring", 750.0)
    monitor.set_threshold("report_generation", 1500.0)
    monitor.set_threshold("integration_test", 5000.0)
    
    return monitor


# Test Data Generation Utilities
@pytest.fixture
def test_data_generator():
    """Provide utilities for generating test data."""
    
    class TestDataGenerator:
        
        @staticmethod
        def generate_random_claim(claim_type: str = None) -> Dict[str, Any]:
            """Generate a random claim for testing."""
            types = ["numeric", "qualitative", "methodological", "temporal", "comparative"]
            selected_type = claim_type or random.choice(types)
            
            templates = {
                "numeric": "The concentration was {value} {unit} in {condition} subjects",
                "qualitative": "There was a {relationship} between {subject} and {object}",
                "methodological": "Analysis was performed using {method} with {equipment}",
                "temporal": "The measurement was taken {time} after {event}",
                "comparative": "{subject1} showed {comparison} levels compared to {subject2}"
            }
            
            # Generate random values based on type
            if selected_type == "numeric":
                value = round(random.uniform(50.0, 200.0), 1)
                unit = random.choice(["mg/dL", "µM", "mM", "ng/mL"])
                condition = random.choice(["diabetic", "healthy", "treated", "control"])
                text = templates[selected_type].format(value=value, unit=unit, condition=condition)
                numeric_values = [value]
                units = [unit]
            else:
                text = templates[selected_type].format(
                    relationship=random.choice(["positive correlation", "negative correlation", "association"]),
                    subject=random.choice(["glucose", "insulin", "metabolite"]),
                    object=random.choice(["disease", "treatment", "outcome"]),
                    method=random.choice(["LC-MS/MS", "GC-MS", "NMR"]),
                    equipment=random.choice(["Agilent", "Thermo", "Waters"]),
                    time=random.choice(["1 hour", "24 hours", "1 week"]),
                    event=random.choice(["treatment", "fasting", "exercise"]),
                    subject1=random.choice(["patients", "controls", "treated group"]),
                    comparison=random.choice(["higher", "lower", "similar"]),
                    subject2=random.choice(["controls", "baseline", "placebo group"])
                )
                numeric_values = []
                units = []
            
            return {
                "claim_id": f"generated_{hashlib.md5(text.encode()).hexdigest()[:8]}",
                "claim_text": text,
                "claim_type": selected_type,
                "numeric_values": numeric_values,
                "units": units,
                "confidence": random.uniform(65.0, 95.0),
                "keywords": text.lower().split()[:5]
            }
        
        @staticmethod
        def generate_verification_results(claims: List[Any], 
                                        success_rate: float = 0.8) -> List[Dict[str, Any]]:
            """Generate verification results for claims."""
            results = []
            
            for claim in claims:
                # Determine verification status based on success rate
                if random.random() < success_rate:
                    status = random.choice([VerificationStatus.SUPPORTED, VerificationStatus.NEUTRAL])
                else:
                    status = random.choice([VerificationStatus.CONTRADICTED, VerificationStatus.NOT_FOUND])
                
                result = {
                    "claim_id": getattr(claim, 'claim_id', f"unknown_{len(results)}"),
                    "verification_status": status,
                    "verification_confidence": random.uniform(60.0, 95.0),
                    "evidence_strength": random.uniform(55.0, 90.0),
                    "context_match": random.uniform(70.0, 95.0),
                    "processing_time_ms": random.uniform(50.0, 300.0),
                    "total_evidence_count": random.randint(0, 5),
                    "verification_strategy": getattr(claim, 'claim_type', 'general')
                }
                
                results.append(result)
            
            return results
        
        @staticmethod
        def generate_large_dataset(num_claims: int = 100) -> Dict[str, Any]:
            """Generate large dataset for performance testing."""
            claims = []
            for i in range(num_claims):
                claim_data = TestDataGenerator.generate_random_claim()
                claims.append(claim_data)
            
            return {
                "claims": claims,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "num_claims": num_claims,
                    "claim_types": list(set(c["claim_type"] for c in claims))
                }
            }
    
    return TestDataGenerator()


# Custom Pytest Markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "validation: mark test as validation system test")
    config.addinivalue_line("markers", "accuracy_scorer: mark test as accuracy scorer test")
    config.addinivalue_line("markers", "integration_validation: mark test as integration validation test")
    config.addinivalue_line("markers", "performance_validation: mark test as performance validation test")
    config.addinivalue_line("markers", "mock_validation: mark test as mock-based validation test")
    config.addinivalue_line("markers", "error_handling_validation: mark test as error handling validation test")


if __name__ == "__main__":
    # Test the fixtures
    print("Factual Validation Test Fixtures initialized successfully!")
    print("Available fixtures:")
    print("- mock_claim_extractor")
    print("- mock_document_indexer") 
    print("- mock_factual_validator")
    print("- mock_accuracy_scorer")
    print("- sample_extracted_claims")
    print("- sample_verification_results")
    print("- sample_accuracy_score")
    print("- performance_test_data")
    print("- error_test_scenarios")
    print("- integration_test_environment")
    print("- performance_monitor")
    print("- test_data_generator")