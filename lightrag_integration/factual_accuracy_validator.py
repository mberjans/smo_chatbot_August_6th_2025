#!/usr/bin/env python3
"""
Factual Accuracy Validation System for Clinical Metabolomics Oracle.

This module provides the FactualAccuracyValidator class for comprehensive verification
of extracted factual claims against indexed source documents in the Clinical Metabolomics
Oracle LightRAG integration project.

Classes:
    - FactualValidationError: Base custom exception for factual validation errors
    - VerificationProcessingError: Exception for verification processing failures
    - EvidenceAssessmentError: Exception for evidence assessment failures
    - VerificationResult: Data class for structured verification results
    - EvidenceItem: Data class for evidence items found in documents
    - VerificationReport: Data class for comprehensive verification reports
    - FactualAccuracyValidator: Main class for factual accuracy validation

The validator handles:
    - Multi-strategy claim verification against source documents
    - Evidence assessment with support/contradict/neutral classifications
    - Confidence scoring for verification results
    - Detailed verification reports with supporting evidence
    - Integration with existing claim extraction and document indexing systems
    - High-performance async processing for large-scale verification
    - Comprehensive error handling and recovery mechanisms

Key Features:
    - Multiple verification strategies for different claim types
    - Evidence strength assessment and confidence scoring
    - Context matching between claims and document evidence
    - Detailed verification reports for debugging and analysis
    - Integration with BiomedicalClaimExtractor and SourceDocumentIndex
    - Performance tracking and optimization
    - Comprehensive error handling and logging

Verification Strategies:
    - Numeric Verification: Match numeric values, ranges, and measurements
    - Qualitative Verification: Assess relationships and qualitative statements
    - Methodological Verification: Validate methods and procedures
    - Temporal Verification: Verify time-based claims and sequences
    - Comparative Verification: Validate comparisons and statistical claims

Evidence Assessment Levels:
    - SUPPORTED: Document provides evidence supporting the claim
    - CONTRADICTED: Document provides evidence contradicting the claim
    - NEUTRAL: Document mentions related concepts but doesn't support/contradict
    - NOT_FOUND: No relevant information found in documents

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Factual Accuracy Validation Implementation
"""

import asyncio
import json
import logging
import re
import time
import hashlib
import math
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, Counter
from contextlib import asynccontextmanager
from enum import Enum

# Enhanced logging imports
try:
    from .enhanced_logging import (
        EnhancedLogger, correlation_manager, performance_logged, PerformanceTracker
    )
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    # Fallback for when enhanced logging is not available
    ENHANCED_LOGGING_AVAILABLE = False
    
    def performance_logged(description="", track_memory=True):
        """Fallback performance logging decorator."""
        def decorator(func):
            return func
        return decorator

# Import related modules
try:
    from .claim_extractor import ExtractedClaim, BiomedicalClaimExtractor
    from .document_indexer import SourceDocumentIndex
except ImportError:
    # Handle import errors gracefully
    logging.warning("Could not import claim_extractor or document_indexer - some features may be limited")

# Configure logging
logger = logging.getLogger(__name__)


class FactualValidationError(Exception):
    """Base custom exception for factual validation errors."""
    pass


class VerificationProcessingError(FactualValidationError):
    """Exception raised when verification processing fails."""
    pass


class EvidenceAssessmentError(FactualValidationError):
    """Exception raised when evidence assessment fails."""
    pass


class VerificationStatus(Enum):
    """Enumeration of verification status values."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NEUTRAL = "NEUTRAL"
    NOT_FOUND = "NOT_FOUND"
    ERROR = "ERROR"


@dataclass
class EvidenceItem:
    """
    Evidence item found in source documents.
    
    Attributes:
        source_document: Identifier for the source document
        evidence_text: Text excerpt containing the evidence
        evidence_type: Type of evidence (numeric, qualitative, etc.)
        context: Surrounding context for the evidence
        confidence: Confidence in the evidence relevance (0-100)
        page_number: Page number in source document
        section: Section or subsection where evidence was found
        metadata: Additional metadata about the evidence
    """
    source_document: str
    evidence_text: str
    evidence_type: str
    context: str = ""
    confidence: float = 0.0
    page_number: Optional[int] = None
    section: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """
    Comprehensive verification result for a factual claim.
    
    Attributes:
        claim_id: Unique identifier for the verified claim
        verification_status: Status of verification (SUPPORTED, CONTRADICTED, etc.)
        verification_confidence: Overall confidence in verification result (0-100)
        evidence_strength: Strength of supporting/contradicting evidence (0-100)
        context_match: How well the claim context matches document context (0-100)
        supporting_evidence: List of evidence items supporting the claim
        contradicting_evidence: List of evidence items contradicting the claim
        neutral_evidence: List of related but neutral evidence items
        verification_strategy: Strategy used for verification
        processing_time_ms: Time taken for verification in milliseconds
        error_details: Details of any errors encountered during verification
        metadata: Additional verification metadata
    """
    claim_id: str
    verification_status: VerificationStatus
    verification_confidence: float
    evidence_strength: float = 0.0
    context_match: float = 0.0
    supporting_evidence: List[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = field(default_factory=list)
    neutral_evidence: List[EvidenceItem] = field(default_factory=list)
    verification_strategy: str = ""
    processing_time_ms: float = 0.0
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_evidence_count(self) -> int:
        """Total number of evidence items found."""
        return (len(self.supporting_evidence) + 
                len(self.contradicting_evidence) + 
                len(self.neutral_evidence))
    
    @property
    def verification_grade(self) -> str:
        """Convert verification confidence to human-readable grade."""
        if self.verification_confidence >= 90:
            return "Very High"
        elif self.verification_confidence >= 75:
            return "High"
        elif self.verification_confidence >= 60:
            return "Moderate"
        elif self.verification_confidence >= 40:
            return "Low"
        else:
            return "Very Low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert verification result to dictionary representation."""
        result = asdict(self)
        result['verification_status'] = self.verification_status.value
        return result


@dataclass
class VerificationReport:
    """
    Comprehensive verification report for a set of claims.
    
    Attributes:
        report_id: Unique identifier for the report
        total_claims: Total number of claims processed
        verification_results: List of all verification results
        summary_statistics: Summary statistics for the verification process
        processing_metadata: Metadata about the verification process
        recommendations: Recommendations based on verification results
        created_timestamp: When the report was created
    """
    report_id: str
    total_claims: int
    verification_results: List[VerificationResult] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert verification report to dictionary representation."""
        result = asdict(self)
        result['created_timestamp'] = self.created_timestamp.isoformat()
        result['verification_results'] = [vr.to_dict() for vr in self.verification_results]
        return result


class FactualAccuracyValidator:
    """
    Main class for comprehensive factual accuracy validation.
    
    Provides comprehensive claim verification capabilities including:
    - Multi-strategy verification for different claim types
    - Evidence assessment and confidence scoring
    - Integration with claim extraction and document indexing systems
    - Performance tracking and optimization
    - Comprehensive error handling and recovery
    """
    
    def __init__(self, 
                 document_indexer: Optional['SourceDocumentIndex'] = None,
                 claim_extractor: Optional['BiomedicalClaimExtractor'] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FactualAccuracyValidator.
        
        Args:
            document_indexer: Optional SourceDocumentIndex instance
            claim_extractor: Optional BiomedicalClaimExtractor instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logger
        self.document_indexer = document_indexer
        self.claim_extractor = claim_extractor
        
        # Initialize verification strategies
        self._initialize_verification_strategies()
        
        # Initialize confidence assessment parameters
        self._initialize_confidence_parameters()
        
        # Initialize performance tracking
        self.verification_stats = defaultdict(int)
        self.processing_times = []
        
        self.logger.info("FactualAccuracyValidator initialized successfully")
    
    def _initialize_verification_strategies(self):
        """Initialize verification strategies for different claim types."""
        
        self.verification_strategies = {
            'numeric': self._verify_numeric_claim,
            'qualitative': self._verify_qualitative_claim,
            'methodological': self._verify_methodological_claim,
            'temporal': self._verify_temporal_claim,
            'comparative': self._verify_comparative_claim
        }
        
        # Numeric verification patterns
        self.numeric_verification_patterns = {
            'exact_match': re.compile(r'(\d+(?:\.\d+)?)', re.IGNORECASE),
            'range_match': re.compile(
                r'(?:between|from|range\s+of)?\s*'
                r'(\d+(?:\.\d+)?)\s*(?:to|and|-|–|—)\s*'
                r'(\d+(?:\.\d+)?)',
                re.IGNORECASE
            ),
            'percentage_match': re.compile(
                r'(\d+(?:\.\d+)?)\s*(?:%|percent)',
                re.IGNORECASE
            ),
            'statistical_match': re.compile(
                r'(?:p-value|p\s*[=<>]\s*|significance\s*[=<>]\s*)'
                r'(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)',
                re.IGNORECASE
            )
        }
        
        # Qualitative verification keywords
        self.qualitative_keywords = {
            'causation': ['causes', 'leads to', 'results in', 'triggers', 'induces'],
            'correlation': ['correlates with', 'associated with', 'linked to', 'related to'],
            'comparison': ['higher than', 'lower than', 'greater than', 'less than', 'compared to'],
            'temporal': ['before', 'after', 'during', 'while', 'when', 'since']
        }
        
        # Methodological verification terms
        self.methodological_terms = [
            'LC-MS', 'GC-MS', 'UPLC', 'HPLC', 'NMR', 'mass spectrometry',
            'chromatography', 'randomized controlled trial', 'RCT', 'case-control',
            'cohort study', 'cross-sectional', 'longitudinal'
        ]
    
    def _initialize_confidence_parameters(self):
        """Initialize parameters for confidence assessment."""
        
        self.confidence_weights = {
            'evidence_quality': 0.35,
            'context_alignment': 0.25,
            'source_credibility': 0.20,
            'consistency': 0.20
        }
        
        self.evidence_quality_factors = {
            'exact_match': 1.0,
            'close_match': 0.8,
            'contextual_match': 0.6,
            'weak_match': 0.3
        }
        
        self.context_alignment_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    @performance_logged("Verify factual claims")
    async def verify_claims(self,
                           claims: List['ExtractedClaim'],
                           verification_config: Optional[Dict[str, Any]] = None) -> VerificationReport:
        """
        Verify a list of factual claims against source documents.
        
        Args:
            claims: List of ExtractedClaim objects to verify
            verification_config: Optional configuration for verification process
            
        Returns:
            VerificationReport with comprehensive verification results
            
        Raises:
            VerificationProcessingError: If verification process fails
        """
        start_time = time.time()
        
        try:
            if not self.document_indexer:
                raise VerificationProcessingError(
                    "Document indexer not available for claim verification"
                )
            
            self.logger.info(f"Starting verification of {len(claims)} claims")
            
            # Process verification configuration
            config = self._merge_verification_config(verification_config)
            
            # Verify each claim
            verification_results = []
            for claim in claims:
                try:
                    result = await self._verify_single_claim(claim, config)
                    verification_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error verifying claim {claim.claim_id}: {str(e)}")
                    # Create error result
                    error_result = VerificationResult(
                        claim_id=claim.claim_id,
                        verification_status=VerificationStatus.ERROR,
                        verification_confidence=0.0,
                        error_details=str(e)
                    )
                    verification_results.append(error_result)
            
            # Generate comprehensive report
            report = await self._generate_verification_report(verification_results, config)
            
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.verification_stats['total_verifications'] += 1
            self.verification_stats['total_claims_verified'] += len(claims)
            
            self.logger.info(
                f"Completed verification of {len(claims)} claims in {processing_time:.2f}ms"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in claims verification: {str(e)}")
            raise VerificationProcessingError(f"Failed to verify claims: {str(e)}") from e
    
    async def _verify_single_claim(self,
                                  claim: 'ExtractedClaim',
                                  config: Dict[str, Any]) -> VerificationResult:
        """
        Verify a single factual claim against source documents.
        
        Args:
            claim: ExtractedClaim object to verify
            config: Verification configuration
            
        Returns:
            VerificationResult for the claim
        """
        start_time = time.time()
        
        try:
            # Select appropriate verification strategy
            verification_strategy = self.verification_strategies.get(
                claim.claim_type, 
                self._verify_general_claim
            )
            
            # Execute verification
            result = await verification_strategy(claim, config)
            
            # Calculate processing time
            result.processing_time_ms = (time.time() - start_time) * 1000
            result.verification_strategy = claim.claim_type
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single claim verification: {str(e)}")
            return VerificationResult(
                claim_id=claim.claim_id,
                verification_status=VerificationStatus.ERROR,
                verification_confidence=0.0,
                error_details=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _verify_numeric_claim(self,
                                   claim: 'ExtractedClaim',
                                   config: Dict[str, Any]) -> VerificationResult:
        """
        Verify numeric claims using specialized numeric matching strategies.
        
        Args:
            claim: Numeric claim to verify
            config: Verification configuration
            
        Returns:
            VerificationResult for the numeric claim
        """
        try:
            # Search for relevant documents containing numeric information
            search_results = await self._search_documents_for_claim(claim, config)
            
            supporting_evidence = []
            contradicting_evidence = []
            neutral_evidence = []
            
            # Process each search result
            for doc_result in search_results:
                evidence_items = await self._extract_numeric_evidence(
                    claim, doc_result, config
                )
                
                for evidence in evidence_items:
                    if evidence.confidence >= config.get('min_evidence_confidence', 70):
                        # Assess evidence against claim
                        assessment = await self._assess_numeric_evidence(claim, evidence)
                        
                        if assessment == 'supporting':
                            supporting_evidence.append(evidence)
                        elif assessment == 'contradicting':
                            contradicting_evidence.append(evidence)
                        else:
                            neutral_evidence.append(evidence)
            
            # Determine verification status
            status = await self._determine_verification_status(
                supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            # Calculate confidence scores
            verification_confidence = await self._calculate_verification_confidence(
                claim, supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            evidence_strength = await self._calculate_evidence_strength(
                supporting_evidence, contradicting_evidence
            )
            
            context_match = await self._calculate_context_match(
                claim, supporting_evidence + contradicting_evidence + neutral_evidence
            )
            
            return VerificationResult(
                claim_id=claim.claim_id,
                verification_status=status,
                verification_confidence=verification_confidence,
                evidence_strength=evidence_strength,
                context_match=context_match,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                neutral_evidence=neutral_evidence,
                metadata={
                    'claim_type': 'numeric',
                    'numeric_values': claim.numeric_values,
                    'units': claim.units,
                    'search_results_count': len(search_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in numeric claim verification: {str(e)}")
            raise EvidenceAssessmentError(f"Failed to verify numeric claim: {str(e)}") from e
    
    async def _verify_qualitative_claim(self,
                                       claim: 'ExtractedClaim',
                                       config: Dict[str, Any]) -> VerificationResult:
        """
        Verify qualitative relationship claims.
        
        Args:
            claim: Qualitative claim to verify
            config: Verification configuration
            
        Returns:
            VerificationResult for the qualitative claim
        """
        try:
            # Search for documents containing relationship information
            search_results = await self._search_documents_for_claim(claim, config)
            
            supporting_evidence = []
            contradicting_evidence = []
            neutral_evidence = []
            
            # Process search results for relationship evidence
            for doc_result in search_results:
                evidence_items = await self._extract_qualitative_evidence(
                    claim, doc_result, config
                )
                
                for evidence in evidence_items:
                    if evidence.confidence >= config.get('min_evidence_confidence', 60):
                        # Assess relationship evidence
                        assessment = await self._assess_qualitative_evidence(claim, evidence)
                        
                        if assessment == 'supporting':
                            supporting_evidence.append(evidence)
                        elif assessment == 'contradicting':
                            contradicting_evidence.append(evidence)
                        else:
                            neutral_evidence.append(evidence)
            
            # Determine verification status and confidence
            status = await self._determine_verification_status(
                supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            verification_confidence = await self._calculate_verification_confidence(
                claim, supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            evidence_strength = await self._calculate_evidence_strength(
                supporting_evidence, contradicting_evidence
            )
            
            context_match = await self._calculate_context_match(
                claim, supporting_evidence + contradicting_evidence + neutral_evidence
            )
            
            return VerificationResult(
                claim_id=claim.claim_id,
                verification_status=status,
                verification_confidence=verification_confidence,
                evidence_strength=evidence_strength,
                context_match=context_match,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                neutral_evidence=neutral_evidence,
                metadata={
                    'claim_type': 'qualitative',
                    'relationships': claim.relationships,
                    'search_results_count': len(search_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in qualitative claim verification: {str(e)}")
            raise EvidenceAssessmentError(f"Failed to verify qualitative claim: {str(e)}") from e
    
    async def _verify_methodological_claim(self,
                                          claim: 'ExtractedClaim',
                                          config: Dict[str, Any]) -> VerificationResult:
        """
        Verify methodological claims about techniques and procedures.
        
        Args:
            claim: Methodological claim to verify
            config: Verification configuration
            
        Returns:
            VerificationResult for the methodological claim
        """
        try:
            # Search for documents containing methodological information
            search_results = await self._search_documents_for_claim(claim, config)
            
            supporting_evidence = []
            contradicting_evidence = []
            neutral_evidence = []
            
            # Process methodological evidence
            for doc_result in search_results:
                evidence_items = await self._extract_methodological_evidence(
                    claim, doc_result, config
                )
                
                for evidence in evidence_items:
                    if evidence.confidence >= config.get('min_evidence_confidence', 65):
                        # Assess methodological evidence
                        assessment = await self._assess_methodological_evidence(claim, evidence)
                        
                        if assessment == 'supporting':
                            supporting_evidence.append(evidence)
                        elif assessment == 'contradicting':
                            contradicting_evidence.append(evidence)
                        else:
                            neutral_evidence.append(evidence)
            
            # Calculate verification metrics
            status = await self._determine_verification_status(
                supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            verification_confidence = await self._calculate_verification_confidence(
                claim, supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            evidence_strength = await self._calculate_evidence_strength(
                supporting_evidence, contradicting_evidence
            )
            
            context_match = await self._calculate_context_match(
                claim, supporting_evidence + contradicting_evidence + neutral_evidence
            )
            
            return VerificationResult(
                claim_id=claim.claim_id,
                verification_status=status,
                verification_confidence=verification_confidence,
                evidence_strength=evidence_strength,
                context_match=context_match,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                neutral_evidence=neutral_evidence,
                metadata={
                    'claim_type': 'methodological',
                    'methods_mentioned': self._extract_methods_from_claim(claim),
                    'search_results_count': len(search_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in methodological claim verification: {str(e)}")
            raise EvidenceAssessmentError(f"Failed to verify methodological claim: {str(e)}") from e
    
    async def _verify_temporal_claim(self,
                                    claim: 'ExtractedClaim',
                                    config: Dict[str, Any]) -> VerificationResult:
        """
        Verify temporal claims about time-based relationships and sequences.
        
        Args:
            claim: Temporal claim to verify
            config: Verification configuration
            
        Returns:
            VerificationResult for the temporal claim
        """
        try:
            # Search for documents with temporal information
            search_results = await self._search_documents_for_claim(claim, config)
            
            supporting_evidence = []
            contradicting_evidence = []
            neutral_evidence = []
            
            # Extract temporal evidence
            for doc_result in search_results:
                evidence_items = await self._extract_temporal_evidence(
                    claim, doc_result, config
                )
                
                for evidence in evidence_items:
                    if evidence.confidence >= config.get('min_evidence_confidence', 60):
                        # Assess temporal evidence
                        assessment = await self._assess_temporal_evidence(claim, evidence)
                        
                        if assessment == 'supporting':
                            supporting_evidence.append(evidence)
                        elif assessment == 'contradicting':
                            contradicting_evidence.append(evidence)
                        else:
                            neutral_evidence.append(evidence)
            
            # Calculate verification results
            status = await self._determine_verification_status(
                supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            verification_confidence = await self._calculate_verification_confidence(
                claim, supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            evidence_strength = await self._calculate_evidence_strength(
                supporting_evidence, contradicting_evidence
            )
            
            context_match = await self._calculate_context_match(
                claim, supporting_evidence + contradicting_evidence + neutral_evidence
            )
            
            return VerificationResult(
                claim_id=claim.claim_id,
                verification_status=status,
                verification_confidence=verification_confidence,
                evidence_strength=evidence_strength,
                context_match=context_match,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                neutral_evidence=neutral_evidence,
                metadata={
                    'claim_type': 'temporal',
                    'temporal_expressions': self._extract_temporal_expressions_from_claim(claim),
                    'search_results_count': len(search_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in temporal claim verification: {str(e)}")
            raise EvidenceAssessmentError(f"Failed to verify temporal claim: {str(e)}") from e
    
    async def _verify_comparative_claim(self,
                                       claim: 'ExtractedClaim',
                                       config: Dict[str, Any]) -> VerificationResult:
        """
        Verify comparative claims about differences and statistical comparisons.
        
        Args:
            claim: Comparative claim to verify
            config: Verification configuration
            
        Returns:
            VerificationResult for the comparative claim
        """
        try:
            # Search for documents with comparative data
            search_results = await self._search_documents_for_claim(claim, config)
            
            supporting_evidence = []
            contradicting_evidence = []
            neutral_evidence = []
            
            # Extract comparative evidence
            for doc_result in search_results:
                evidence_items = await self._extract_comparative_evidence(
                    claim, doc_result, config
                )
                
                for evidence in evidence_items:
                    if evidence.confidence >= config.get('min_evidence_confidence', 70):
                        # Assess comparative evidence
                        assessment = await self._assess_comparative_evidence(claim, evidence)
                        
                        if assessment == 'supporting':
                            supporting_evidence.append(evidence)
                        elif assessment == 'contradicting':
                            contradicting_evidence.append(evidence)
                        else:
                            neutral_evidence.append(evidence)
            
            # Calculate verification metrics
            status = await self._determine_verification_status(
                supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            verification_confidence = await self._calculate_verification_confidence(
                claim, supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            evidence_strength = await self._calculate_evidence_strength(
                supporting_evidence, contradicting_evidence
            )
            
            context_match = await self._calculate_context_match(
                claim, supporting_evidence + contradicting_evidence + neutral_evidence
            )
            
            return VerificationResult(
                claim_id=claim.claim_id,
                verification_status=status,
                verification_confidence=verification_confidence,
                evidence_strength=evidence_strength,
                context_match=context_match,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                neutral_evidence=neutral_evidence,
                metadata={
                    'claim_type': 'comparative',
                    'comparative_expressions': self._extract_comparative_expressions_from_claim(claim),
                    'search_results_count': len(search_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in comparative claim verification: {str(e)}")
            raise EvidenceAssessmentError(f"Failed to verify comparative claim: {str(e)}") from e
    
    async def _verify_general_claim(self,
                                   claim: 'ExtractedClaim',
                                   config: Dict[str, Any]) -> VerificationResult:
        """
        Verify general claims using a comprehensive approach.
        
        Args:
            claim: General claim to verify
            config: Verification configuration
            
        Returns:
            VerificationResult for the general claim
        """
        try:
            # Use multiple verification strategies
            search_results = await self._search_documents_for_claim(claim, config)
            
            supporting_evidence = []
            contradicting_evidence = []
            neutral_evidence = []
            
            # Apply general evidence extraction
            for doc_result in search_results:
                evidence_items = await self._extract_general_evidence(
                    claim, doc_result, config
                )
                
                for evidence in evidence_items:
                    if evidence.confidence >= config.get('min_evidence_confidence', 50):
                        # General evidence assessment
                        assessment = await self._assess_general_evidence(claim, evidence)
                        
                        if assessment == 'supporting':
                            supporting_evidence.append(evidence)
                        elif assessment == 'contradicting':
                            contradicting_evidence.append(evidence)
                        else:
                            neutral_evidence.append(evidence)
            
            # Calculate verification results
            status = await self._determine_verification_status(
                supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            verification_confidence = await self._calculate_verification_confidence(
                claim, supporting_evidence, contradicting_evidence, neutral_evidence
            )
            
            evidence_strength = await self._calculate_evidence_strength(
                supporting_evidence, contradicting_evidence
            )
            
            context_match = await self._calculate_context_match(
                claim, supporting_evidence + contradicting_evidence + neutral_evidence
            )
            
            return VerificationResult(
                claim_id=claim.claim_id,
                verification_status=status,
                verification_confidence=verification_confidence,
                evidence_strength=evidence_strength,
                context_match=context_match,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                neutral_evidence=neutral_evidence,
                metadata={
                    'claim_type': 'general',
                    'search_results_count': len(search_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in general claim verification: {str(e)}")
            raise EvidenceAssessmentError(f"Failed to verify general claim: {str(e)}") from e
    
    async def _search_documents_for_claim(self,
                                         claim: 'ExtractedClaim',
                                         config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search indexed documents for content related to the claim.
        
        Args:
            claim: Claim to search for
            config: Search configuration
            
        Returns:
            List of document search results
        """
        try:
            # Prepare search query from claim
            search_query = await self._prepare_search_query_from_claim(claim)
            
            # Use document indexer to search
            if hasattr(self.document_indexer, 'search_content'):
                search_results = await self.document_indexer.search_content(
                    search_query, 
                    max_results=config.get('max_search_results', 50)
                )
            else:
                # Fallback to basic claim verification
                verification_result = await self.document_indexer.verify_claim(
                    claim.claim_text, 
                    config.get('verification_config', {})
                )
                
                # Convert to expected format
                search_results = [{
                    'content': claim.claim_text,
                    'metadata': verification_result.get('verification_metadata', {}),
                    'supporting_facts': verification_result.get('supporting_evidence', []),
                    'contradicting_facts': verification_result.get('contradicting_evidence', []),
                    'related_facts': verification_result.get('related_facts', [])
                }]
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error searching documents for claim: {str(e)}")
            return []
    
    async def _prepare_search_query_from_claim(self, claim: 'ExtractedClaim') -> str:
        """
        Prepare search query from claim information.
        
        Args:
            claim: Claim to create search query for
            
        Returns:
            Search query string
        """
        query_parts = []
        
        # Add claim keywords
        if claim.keywords:
            query_parts.extend(claim.keywords)
        
        # Add subject and object
        if claim.subject:
            query_parts.append(claim.subject)
        if claim.object_value:
            query_parts.append(claim.object_value)
        
        # Add numeric values as search terms
        if claim.numeric_values:
            for value in claim.numeric_values:
                query_parts.append(str(value))
        
        # Add units
        if claim.units:
            query_parts.extend(claim.units)
        
        # Create search query
        search_query = ' '.join(query_parts[:10])  # Limit to top 10 terms
        return search_query
    
    async def _extract_numeric_evidence(self,
                                       claim: 'ExtractedClaim',
                                       doc_result: Dict[str, Any],
                                       config: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Extract numeric evidence from document result.
        
        Args:
            claim: Claim being verified
            doc_result: Document search result
            config: Configuration
            
        Returns:
            List of numeric evidence items
        """
        evidence_items = []
        
        try:
            content = doc_result.get('content', '')
            
            # Search for numeric patterns in content
            for pattern_name, pattern in self.numeric_verification_patterns.items():
                matches = pattern.finditer(content)
                
                for match in matches:
                    # Extract surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]
                    
                    # Calculate confidence based on match type and context
                    confidence = await self._calculate_numeric_evidence_confidence(
                        claim, match.group(), context, pattern_name
                    )
                    
                    if confidence >= config.get('min_match_confidence', 30):
                        evidence = EvidenceItem(
                            source_document=doc_result.get('document_id', 'unknown'),
                            evidence_text=match.group(),
                            evidence_type='numeric',
                            context=context,
                            confidence=confidence,
                            page_number=doc_result.get('page_number'),
                            section=doc_result.get('section', ''),
                            metadata={
                                'pattern_type': pattern_name,
                                'match_start': match.start(),
                                'match_end': match.end()
                            }
                        )
                        evidence_items.append(evidence)
            
        except Exception as e:
            self.logger.error(f"Error extracting numeric evidence: {str(e)}")
        
        return evidence_items
    
    async def _extract_qualitative_evidence(self,
                                           claim: 'ExtractedClaim',
                                           doc_result: Dict[str, Any],
                                           config: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Extract qualitative evidence from document result.
        
        Args:
            claim: Claim being verified
            doc_result: Document search result
            config: Configuration
            
        Returns:
            List of qualitative evidence items
        """
        evidence_items = []
        
        try:
            content = doc_result.get('content', '')
            
            # Search for qualitative relationship keywords
            for relationship_type, keywords in self.qualitative_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in content.lower():
                        # Find all occurrences
                        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                        matches = pattern.finditer(content)
                        
                        for match in matches:
                            # Extract context
                            start = max(0, match.start() - 150)
                            end = min(len(content), match.end() + 150)
                            context = content[start:end]
                            
                            # Calculate confidence
                            confidence = await self._calculate_qualitative_evidence_confidence(
                                claim, keyword, context, relationship_type
                            )
                            
                            if confidence >= config.get('min_match_confidence', 40):
                                evidence = EvidenceItem(
                                    source_document=doc_result.get('document_id', 'unknown'),
                                    evidence_text=keyword,
                                    evidence_type='qualitative',
                                    context=context,
                                    confidence=confidence,
                                    page_number=doc_result.get('page_number'),
                                    section=doc_result.get('section', ''),
                                    metadata={
                                        'relationship_type': relationship_type,
                                        'keyword': keyword
                                    }
                                )
                                evidence_items.append(evidence)
            
        except Exception as e:
            self.logger.error(f"Error extracting qualitative evidence: {str(e)}")
        
        return evidence_items
    
    async def _extract_methodological_evidence(self,
                                              claim: 'ExtractedClaim',
                                              doc_result: Dict[str, Any],
                                              config: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Extract methodological evidence from document result.
        
        Args:
            claim: Claim being verified
            doc_result: Document search result
            config: Configuration
            
        Returns:
            List of methodological evidence items
        """
        evidence_items = []
        
        try:
            content = doc_result.get('content', '')
            
            # Search for methodological terms
            for method_term in self.methodological_terms:
                if method_term.lower() in content.lower():
                    pattern = re.compile(re.escape(method_term), re.IGNORECASE)
                    matches = pattern.finditer(content)
                    
                    for match in matches:
                        # Extract context
                        start = max(0, match.start() - 120)
                        end = min(len(content), match.end() + 120)
                        context = content[start:end]
                        
                        # Calculate confidence
                        confidence = await self._calculate_methodological_evidence_confidence(
                            claim, method_term, context
                        )
                        
                        if confidence >= config.get('min_match_confidence', 50):
                            evidence = EvidenceItem(
                                source_document=doc_result.get('document_id', 'unknown'),
                                evidence_text=method_term,
                                evidence_type='methodological',
                                context=context,
                                confidence=confidence,
                                page_number=doc_result.get('page_number'),
                                section=doc_result.get('section', ''),
                                metadata={
                                    'method_term': method_term
                                }
                            )
                            evidence_items.append(evidence)
            
        except Exception as e:
            self.logger.error(f"Error extracting methodological evidence: {str(e)}")
        
        return evidence_items
    
    async def _extract_temporal_evidence(self,
                                        claim: 'ExtractedClaim',
                                        doc_result: Dict[str, Any],
                                        config: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Extract temporal evidence from document result.
        
        Args:
            claim: Claim being verified
            doc_result: Document search result
            config: Configuration
            
        Returns:
            List of temporal evidence items
        """
        evidence_items = []
        
        try:
            content = doc_result.get('content', '')
            
            # Temporal patterns
            temporal_patterns = [
                r'\b(?:before|after|during|while|when|since|until|following)\b',
                r'\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?)',
                r'\b(?:first|second|third|initially|subsequently|finally)\b'
            ]
            
            for pattern_str in temporal_patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                matches = pattern.finditer(content)
                
                for match in matches:
                    # Extract context
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]
                    
                    # Calculate confidence
                    confidence = await self._calculate_temporal_evidence_confidence(
                        claim, match.group(), context
                    )
                    
                    if confidence >= config.get('min_match_confidence', 35):
                        evidence = EvidenceItem(
                            source_document=doc_result.get('document_id', 'unknown'),
                            evidence_text=match.group(),
                            evidence_type='temporal',
                            context=context,
                            confidence=confidence,
                            page_number=doc_result.get('page_number'),
                            section=doc_result.get('section', ''),
                            metadata={
                                'temporal_expression': match.group()
                            }
                        )
                        evidence_items.append(evidence)
            
        except Exception as e:
            self.logger.error(f"Error extracting temporal evidence: {str(e)}")
        
        return evidence_items
    
    async def _extract_comparative_evidence(self,
                                           claim: 'ExtractedClaim',
                                           doc_result: Dict[str, Any],
                                           config: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Extract comparative evidence from document result.
        
        Args:
            claim: Claim being verified
            doc_result: Document search result
            config: Configuration
            
        Returns:
            List of comparative evidence items
        """
        evidence_items = []
        
        try:
            content = doc_result.get('content', '')
            
            # Comparative patterns
            comparative_patterns = [
                r'\b(?:higher|lower|greater|less|increased|decreased|elevated|reduced)\b',
                r'\d+(?:\.\d+)?\s*(?:-|\s*)?fold\s*(?:increase|decrease|change)',
                r'\b(?:compared\s+to|versus|vs\.?|relative\s+to)\b',
                r'\b(?:significantly|statistically\s+significant)\b'
            ]
            
            for pattern_str in comparative_patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                matches = pattern.finditer(content)
                
                for match in matches:
                    # Extract context
                    start = max(0, match.start() - 120)
                    end = min(len(content), match.end() + 120)
                    context = content[start:end]
                    
                    # Calculate confidence
                    confidence = await self._calculate_comparative_evidence_confidence(
                        claim, match.group(), context
                    )
                    
                    if confidence >= config.get('min_match_confidence', 45):
                        evidence = EvidenceItem(
                            source_document=doc_result.get('document_id', 'unknown'),
                            evidence_text=match.group(),
                            evidence_type='comparative',
                            context=context,
                            confidence=confidence,
                            page_number=doc_result.get('page_number'),
                            section=doc_result.get('section', ''),
                            metadata={
                                'comparative_expression': match.group()
                            }
                        )
                        evidence_items.append(evidence)
            
        except Exception as e:
            self.logger.error(f"Error extracting comparative evidence: {str(e)}")
        
        return evidence_items
    
    async def _extract_general_evidence(self,
                                       claim: 'ExtractedClaim',
                                       doc_result: Dict[str, Any],
                                       config: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Extract general evidence from document result.
        
        Args:
            claim: Claim being verified
            doc_result: Document search result
            config: Configuration
            
        Returns:
            List of general evidence items
        """
        evidence_items = []
        
        try:
            content = doc_result.get('content', '')
            
            # Search for claim keywords in content
            for keyword in claim.keywords[:5]:  # Limit to top 5 keywords
                if len(keyword) > 3 and keyword.lower() in content.lower():
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    matches = pattern.finditer(content)
                    
                    for match in matches:
                        # Extract context
                        start = max(0, match.start() - 100)
                        end = min(len(content), match.end() + 100)
                        context = content[start:end]
                        
                        # Calculate confidence
                        confidence = await self._calculate_general_evidence_confidence(
                            claim, keyword, context
                        )
                        
                        if confidence >= config.get('min_match_confidence', 25):
                            evidence = EvidenceItem(
                                source_document=doc_result.get('document_id', 'unknown'),
                                evidence_text=keyword,
                                evidence_type='general',
                                context=context,
                                confidence=confidence,
                                page_number=doc_result.get('page_number'),
                                section=doc_result.get('section', ''),
                                metadata={
                                    'keyword': keyword
                                }
                            )
                            evidence_items.append(evidence)
            
        except Exception as e:
            self.logger.error(f"Error extracting general evidence: {str(e)}")
        
        return evidence_items
    
    # Evidence assessment methods
    async def _assess_numeric_evidence(self, claim: 'ExtractedClaim', evidence: EvidenceItem) -> str:
        """Assess whether numeric evidence supports, contradicts, or is neutral to claim."""
        try:
            # Extract numeric values from evidence
            evidence_numbers = re.findall(r'\d+(?:\.\d+)?', evidence.evidence_text)
            
            if not evidence_numbers or not claim.numeric_values:
                return 'neutral'
            
            evidence_values = [float(num) for num in evidence_numbers]
            
            # Compare with claim values
            for claim_value in claim.numeric_values:
                for evidence_value in evidence_values:
                    # Check for exact match or close match (within 10%)
                    if abs(claim_value - evidence_value) / max(claim_value, evidence_value) < 0.1:
                        return 'supporting'
                    
                    # Check for significant difference (might indicate contradiction)
                    if abs(claim_value - evidence_value) / max(claim_value, evidence_value) > 0.5:
                        # Further context analysis needed
                        if self._analyze_contradiction_context(claim, evidence):
                            return 'contradicting'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Error assessing numeric evidence: {str(e)}")
            return 'neutral'
    
    async def _assess_qualitative_evidence(self, claim: 'ExtractedClaim', evidence: EvidenceItem) -> str:
        """Assess whether qualitative evidence supports, contradicts, or is neutral to claim."""
        try:
            # Analyze relationship alignment
            claim_relationships = [rel.get('type', '') for rel in claim.relationships]
            evidence_type = evidence.metadata.get('relationship_type', '')
            
            if evidence_type in claim_relationships:
                return 'supporting'
            
            # Check for contradictory relationships
            contradictory_pairs = {
                'causation': ['correlation'],
                'correlation': ['causation'],
                'positive': ['negative'],
                'increase': ['decrease']
            }
            
            for claim_rel in claim_relationships:
                if evidence_type in contradictory_pairs.get(claim_rel, []):
                    return 'contradicting'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Error assessing qualitative evidence: {str(e)}")
            return 'neutral'
    
    async def _assess_methodological_evidence(self, claim: 'ExtractedClaim', evidence: EvidenceItem) -> str:
        """Assess whether methodological evidence supports, contradicts, or is neutral to claim."""
        try:
            claim_methods = self._extract_methods_from_claim(claim)
            evidence_method = evidence.metadata.get('method_term', '')
            
            # Direct method match
            if evidence_method.lower() in [method.lower() for method in claim_methods]:
                return 'supporting'
            
            # Check for alternative methods for same purpose
            method_families = {
                'mass_spectrometry': ['LC-MS', 'GC-MS', 'UPLC', 'MS/MS', 'QTOF'],
                'chromatography': ['HPLC', 'UPLC', 'GC', 'LC'],
                'nmr': ['NMR', '1H-NMR', '13C-NMR'],
                'clinical_study': ['RCT', 'randomized controlled trial', 'clinical trial']
            }
            
            for family, methods in method_families.items():
                claim_in_family = any(method.upper() in [m.upper() for m in methods] for method in claim_methods)
                evidence_in_family = evidence_method.upper() in [m.upper() for m in methods]
                
                if claim_in_family and evidence_in_family:
                    return 'supporting'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Error assessing methodological evidence: {str(e)}")
            return 'neutral'
    
    async def _assess_temporal_evidence(self, claim: 'ExtractedClaim', evidence: EvidenceItem) -> str:
        """Assess whether temporal evidence supports, contradicts, or is neutral to claim."""
        try:
            claim_temporal = self._extract_temporal_expressions_from_claim(claim)
            evidence_temporal = evidence.metadata.get('temporal_expression', '')
            
            # Direct temporal expression match
            if evidence_temporal.lower() in [expr.lower() for expr in claim_temporal]:
                return 'supporting'
            
            # Check for contradictory temporal relationships
            contradictory_temporal = {
                'before': ['after'],
                'after': ['before'],
                'increase': ['decrease'],
                'first': ['last', 'final']
            }
            
            for claim_expr in claim_temporal:
                for evidence_word in evidence_temporal.split():
                    if evidence_word.lower() in contradictory_temporal.get(claim_expr.lower(), []):
                        return 'contradicting'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Error assessing temporal evidence: {str(e)}")
            return 'neutral'
    
    async def _assess_comparative_evidence(self, claim: 'ExtractedClaim', evidence: EvidenceItem) -> str:
        """Assess whether comparative evidence supports, contradicts, or is neutral to claim."""
        try:
            claim_comparatives = self._extract_comparative_expressions_from_claim(claim)
            evidence_comparative = evidence.metadata.get('comparative_expression', '')
            
            # Direct comparative match
            if evidence_comparative.lower() in [expr.lower() for expr in claim_comparatives]:
                return 'supporting'
            
            # Check for contradictory comparatives
            contradictory_comparatives = {
                'higher': ['lower'],
                'lower': ['higher'],
                'increased': ['decreased'],
                'decreased': ['increased'],
                'greater': ['less'],
                'less': ['greater']
            }
            
            for claim_expr in claim_comparatives:
                for evidence_word in evidence_comparative.split():
                    if evidence_word.lower() in contradictory_comparatives.get(claim_expr.lower(), []):
                        return 'contradicting'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Error assessing comparative evidence: {str(e)}")
            return 'neutral'
    
    async def _assess_general_evidence(self, claim: 'ExtractedClaim', evidence: EvidenceItem) -> str:
        """Assess whether general evidence supports, contradicts, or is neutral to claim."""
        try:
            # Simple keyword-based assessment
            keyword = evidence.metadata.get('keyword', '')
            
            # If keyword appears in claim text, it's generally supporting
            if keyword.lower() in claim.claim_text.lower():
                return 'supporting'
            
            # Check context for contradictory indicators
            context = evidence.context.lower()
            contradictory_indicators = ['not', 'no', 'never', 'without', 'except', 'however', 'but']
            
            keyword_index = context.find(keyword.lower())
            if keyword_index != -1:
                # Look for contradictory words near the keyword
                nearby_text = context[max(0, keyword_index-50):keyword_index+50]
                if any(indicator in nearby_text for indicator in contradictory_indicators):
                    return 'contradicting'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Error assessing general evidence: {str(e)}")
            return 'neutral'
    
    # Confidence calculation methods
    async def _calculate_numeric_evidence_confidence(self,
                                                    claim: 'ExtractedClaim',
                                                    evidence_text: str,
                                                    context: str,
                                                    pattern_type: str) -> float:
        """Calculate confidence for numeric evidence."""
        base_confidence = 50.0
        
        try:
            # Pattern type bonuses
            pattern_bonuses = {
                'exact_match': 30,
                'range_match': 25,
                'percentage_match': 20,
                'statistical_match': 35
            }
            
            base_confidence += pattern_bonuses.get(pattern_type, 0)
            
            # Context quality bonus
            if any(unit in context.lower() for unit in claim.units):
                base_confidence += 15
            
            # Keyword alignment bonus
            matching_keywords = sum(1 for keyword in claim.keywords if keyword.lower() in context.lower())
            base_confidence += min(matching_keywords * 5, 20)
            
            return min(100.0, max(0.0, base_confidence))
            
        except Exception:
            return 50.0
    
    async def _calculate_qualitative_evidence_confidence(self,
                                                        claim: 'ExtractedClaim',
                                                        keyword: str,
                                                        context: str,
                                                        relationship_type: str) -> float:
        """Calculate confidence for qualitative evidence."""
        base_confidence = 40.0
        
        try:
            # Relationship type bonus
            if relationship_type in ['causation', 'correlation']:
                base_confidence += 20
            
            # Subject/object alignment
            if claim.subject and claim.subject.lower() in context.lower():
                base_confidence += 15
            if claim.object_value and claim.object_value.lower() in context.lower():
                base_confidence += 15
            
            # Context strength
            strong_indicators = ['demonstrated', 'shown', 'observed', 'found', 'confirmed']
            if any(indicator in context.lower() for indicator in strong_indicators):
                base_confidence += 10
            
            return min(100.0, max(0.0, base_confidence))
            
        except Exception:
            return 40.0
    
    async def _calculate_methodological_evidence_confidence(self,
                                                           claim: 'ExtractedClaim',
                                                           method_term: str,
                                                           context: str) -> float:
        """Calculate confidence for methodological evidence."""
        base_confidence = 55.0
        
        try:
            # Method specificity bonus
            specific_methods = ['LC-MS', 'GC-MS', 'UPLC-MS', 'NMR', 'QTOF']
            if method_term in specific_methods:
                base_confidence += 20
            
            # Context quality
            methodological_indicators = ['analysis', 'performed', 'using', 'method', 'technique']
            matching_indicators = sum(1 for indicator in methodological_indicators 
                                    if indicator in context.lower())
            base_confidence += matching_indicators * 3
            
            return min(100.0, max(0.0, base_confidence))
            
        except Exception:
            return 55.0
    
    async def _calculate_temporal_evidence_confidence(self,
                                                     claim: 'ExtractedClaim',
                                                     temporal_expr: str,
                                                     context: str) -> float:
        """Calculate confidence for temporal evidence."""
        base_confidence = 35.0
        
        try:
            # Temporal specificity bonus
            if re.search(r'\d+', temporal_expr):  # Has numbers
                base_confidence += 15
            
            # Context alignment
            if any(keyword.lower() in context.lower() for keyword in claim.keywords[:3]):
                base_confidence += 10
            
            # Time unit specificity
            time_units = ['minutes', 'hours', 'days', 'weeks', 'months', 'years']
            if any(unit in temporal_expr.lower() for unit in time_units):
                base_confidence += 10
            
            return min(100.0, max(0.0, base_confidence))
            
        except Exception:
            return 35.0
    
    async def _calculate_comparative_evidence_confidence(self,
                                                        claim: 'ExtractedClaim',
                                                        comparative_expr: str,
                                                        context: str) -> float:
        """Calculate confidence for comparative evidence."""
        base_confidence = 45.0
        
        try:
            # Statistical significance bonus
            if 'significant' in comparative_expr.lower():
                base_confidence += 20
            
            # Numeric specificity bonus
            if re.search(r'\d+', comparative_expr):
                base_confidence += 10
            
            # Context quality
            statistical_terms = ['p-value', 'confidence', 'analysis', 'study', 'trial']
            if any(term in context.lower() for term in statistical_terms):
                base_confidence += 10
            
            return min(100.0, max(0.0, base_confidence))
            
        except Exception:
            return 45.0
    
    async def _calculate_general_evidence_confidence(self,
                                                    claim: 'ExtractedClaim',
                                                    keyword: str,
                                                    context: str) -> float:
        """Calculate confidence for general evidence."""
        base_confidence = 25.0
        
        try:
            # Keyword importance (longer keywords generally more specific)
            base_confidence += min(len(keyword), 10)
            
            # Context relevance
            relevant_keywords = sum(1 for kw in claim.keywords[:5] if kw.lower() in context.lower())
            base_confidence += relevant_keywords * 3
            
            # Biomedical domain indicators
            biomedical_terms = ['patients', 'clinical', 'study', 'analysis', 'treatment', 'disease']
            if any(term in context.lower() for term in biomedical_terms):
                base_confidence += 10
            
            return min(100.0, max(0.0, base_confidence))
            
        except Exception:
            return 25.0
    
    # Utility methods for verification assessment
    async def _determine_verification_status(self,
                                           supporting: List[EvidenceItem],
                                           contradicting: List[EvidenceItem],
                                           neutral: List[EvidenceItem]) -> VerificationStatus:
        """Determine overall verification status based on evidence."""
        
        # Calculate evidence strengths
        support_strength = sum(evidence.confidence for evidence in supporting)
        contradict_strength = sum(evidence.confidence for evidence in contradicting)
        
        # Decision thresholds
        strong_support_threshold = 150
        strong_contradict_threshold = 100
        
        if support_strength >= strong_support_threshold and support_strength > contradict_strength * 1.5:
            return VerificationStatus.SUPPORTED
        elif contradict_strength >= strong_contradict_threshold and contradict_strength > support_strength * 1.5:
            return VerificationStatus.CONTRADICTED
        elif supporting or contradicting or neutral:
            return VerificationStatus.NEUTRAL
        else:
            return VerificationStatus.NOT_FOUND
    
    async def _calculate_verification_confidence(self,
                                               claim: 'ExtractedClaim',
                                               supporting: List[EvidenceItem],
                                               contradicting: List[EvidenceItem],
                                               neutral: List[EvidenceItem]) -> float:
        """Calculate overall verification confidence."""
        
        try:
            total_evidence = len(supporting) + len(contradicting) + len(neutral)
            
            if total_evidence == 0:
                return 0.0
            
            # Base confidence from evidence quantity
            base_confidence = min(total_evidence * 10, 60)
            
            # Quality bonus from high-confidence evidence
            high_quality_evidence = [
                e for e in (supporting + contradicting + neutral)
                if e.confidence >= 70
            ]
            base_confidence += len(high_quality_evidence) * 5
            
            # Consistency bonus/penalty
            if supporting and not contradicting:
                base_confidence += 20  # Consistent support
            elif contradicting and not supporting:
                base_confidence += 15  # Consistent contradiction
            elif supporting and contradicting:
                base_confidence -= 10  # Mixed evidence
            
            # Claim confidence factor
            base_confidence += claim.confidence.overall_confidence * 0.2
            
            return min(100.0, max(0.0, base_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating verification confidence: {str(e)}")
            return 50.0
    
    async def _calculate_evidence_strength(self,
                                         supporting: List[EvidenceItem],
                                         contradicting: List[EvidenceItem]) -> float:
        """Calculate strength of evidence."""
        
        try:
            if not supporting and not contradicting:
                return 0.0
            
            support_strength = sum(evidence.confidence for evidence in supporting)
            contradict_strength = sum(evidence.confidence for evidence in contradicting)
            total_strength = support_strength + contradict_strength
            
            if total_strength == 0:
                return 0.0
            
            # Normalize to 0-100 scale
            max_possible = len(supporting + contradicting) * 100
            strength_score = (total_strength / max_possible) * 100 if max_possible > 0 else 0
            
            return min(100.0, strength_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating evidence strength: {str(e)}")
            return 0.0
    
    async def _calculate_context_match(self,
                                     claim: 'ExtractedClaim',
                                     all_evidence: List[EvidenceItem]) -> float:
        """Calculate context match between claim and evidence."""
        
        try:
            if not all_evidence:
                return 0.0
            
            total_match = 0.0
            
            for evidence in all_evidence:
                match_score = 0.0
                
                # Keyword overlap
                evidence_words = set(evidence.context.lower().split())
                claim_words = set(claim.claim_text.lower().split())
                overlap = len(evidence_words.intersection(claim_words))
                match_score += (overlap / max(len(claim_words), 1)) * 50
                
                # Subject/object alignment
                if claim.subject and claim.subject.lower() in evidence.context.lower():
                    match_score += 25
                if claim.object_value and claim.object_value.lower() in evidence.context.lower():
                    match_score += 25
                
                total_match += min(100.0, match_score)
            
            average_match = total_match / len(all_evidence)
            return min(100.0, average_match)
            
        except Exception as e:
            self.logger.error(f"Error calculating context match: {str(e)}")
            return 0.0
    
    async def _generate_verification_report(self,
                                          verification_results: List[VerificationResult],
                                          config: Dict[str, Any]) -> VerificationReport:
        """Generate comprehensive verification report."""
        
        try:
            # Create report ID
            report_id = hashlib.md5(
                f"{datetime.now().isoformat()}_{len(verification_results)}".encode()
            ).hexdigest()[:12]
            
            # Calculate summary statistics
            summary_stats = await self._calculate_verification_statistics(verification_results)
            
            # Generate recommendations
            recommendations = await self._generate_verification_recommendations(
                verification_results, summary_stats
            )
            
            # Create processing metadata
            processing_metadata = {
                'config_used': config,
                'processing_timestamp': datetime.now().isoformat(),
                'total_processing_time_ms': sum(vr.processing_time_ms for vr in verification_results),
                'average_processing_time_ms': statistics.mean([vr.processing_time_ms for vr in verification_results]) if verification_results else 0,
                'verification_strategies_used': list(set(vr.verification_strategy for vr in verification_results))
            }
            
            return VerificationReport(
                report_id=report_id,
                total_claims=len(verification_results),
                verification_results=verification_results,
                summary_statistics=summary_stats,
                processing_metadata=processing_metadata,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error generating verification report: {str(e)}")
            raise VerificationProcessingError(f"Failed to generate verification report: {str(e)}") from e
    
    async def _calculate_verification_statistics(self,
                                               verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """Calculate summary statistics for verification results."""
        
        if not verification_results:
            return {}
        
        # Status distribution
        status_counts = Counter(vr.verification_status for vr in verification_results)
        
        # Confidence statistics
        confidences = [vr.verification_confidence for vr in verification_results]
        
        # Evidence statistics
        total_evidence_items = sum(vr.total_evidence_count for vr in verification_results)
        
        return {
            'status_distribution': {status.value: count for status, count in status_counts.items()},
            'confidence_statistics': {
                'mean': statistics.mean(confidences),
                'median': statistics.median(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'std_dev': statistics.stdev(confidences) if len(confidences) > 1 else 0
            },
            'evidence_statistics': {
                'total_evidence_items': total_evidence_items,
                'average_evidence_per_claim': total_evidence_items / len(verification_results),
                'claims_with_supporting_evidence': sum(1 for vr in verification_results if vr.supporting_evidence),
                'claims_with_contradicting_evidence': sum(1 for vr in verification_results if vr.contradicting_evidence)
            },
            'performance_statistics': {
                'total_processing_time_ms': sum(vr.processing_time_ms for vr in verification_results),
                'average_processing_time_ms': statistics.mean([vr.processing_time_ms for vr in verification_results])
            }
        }
    
    async def _generate_verification_recommendations(self,
                                                   verification_results: List[VerificationResult],
                                                   summary_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on verification results."""
        
        recommendations = []
        
        try:
            # High contradiction rate
            contradict_rate = summary_stats.get('status_distribution', {}).get('CONTRADICTED', 0) / len(verification_results)
            if contradict_rate > 0.2:
                recommendations.append(
                    "High contradiction rate detected. Review source documents and claim extraction accuracy."
                )
            
            # Low evidence rate
            not_found_rate = summary_stats.get('status_distribution', {}).get('NOT_FOUND', 0) / len(verification_results)
            if not_found_rate > 0.3:
                recommendations.append(
                    "Many claims lack supporting evidence. Consider expanding document index or improving search strategies."
                )
            
            # Low confidence
            avg_confidence = summary_stats.get('confidence_statistics', {}).get('mean', 0)
            if avg_confidence < 60:
                recommendations.append(
                    "Low average verification confidence. Consider refining evidence assessment algorithms."
                )
            
            # Performance issues
            avg_processing_time = summary_stats.get('performance_statistics', {}).get('average_processing_time_ms', 0)
            if avg_processing_time > 1000:
                recommendations.append(
                    "High processing times detected. Consider optimizing search and verification algorithms."
                )
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Error generating recommendations - manual review recommended.")
        
        return recommendations
    
    # Utility helper methods
    def _merge_verification_config(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge user configuration with defaults."""
        
        default_config = {
            'max_search_results': 50,
            'min_evidence_confidence': 50,
            'min_match_confidence': 30,
            'enable_context_analysis': True,
            'evidence_quality_threshold': 0.6,
            'consistency_weight': 0.3
        }
        
        if user_config:
            default_config.update(user_config)
        
        return default_config
    
    def _analyze_contradiction_context(self, claim: 'ExtractedClaim', evidence: EvidenceItem) -> bool:
        """Analyze context to determine if evidence contradicts claim."""
        
        contradiction_indicators = [
            'however', 'but', 'although', 'nevertheless', 'nonetheless',
            'in contrast', 'on the contrary', 'different from', 'unlike'
        ]
        
        context = evidence.context.lower()
        return any(indicator in context for indicator in contradiction_indicators)
    
    def _extract_methods_from_claim(self, claim: 'ExtractedClaim') -> List[str]:
        """Extract methodological terms from claim."""
        
        methods = []
        claim_text = claim.claim_text.lower()
        
        for method_term in self.methodological_terms:
            if method_term.lower() in claim_text:
                methods.append(method_term)
        
        return methods
    
    def _extract_temporal_expressions_from_claim(self, claim: 'ExtractedClaim') -> List[str]:
        """Extract temporal expressions from claim."""
        
        temporal_patterns = [
            r'\b(?:before|after|during|while|when|since|until|following)\b',
            r'\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?)',
            r'\b(?:first|second|third|initially|subsequently|finally)\b'
        ]
        
        expressions = []
        claim_text = claim.claim_text.lower()
        
        for pattern_str in temporal_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(claim_text)
            expressions.extend(matches)
        
        return expressions
    
    def _extract_comparative_expressions_from_claim(self, claim: 'ExtractedClaim') -> List[str]:
        """Extract comparative expressions from claim."""
        
        comparative_patterns = [
            r'\b(?:higher|lower|greater|less|increased|decreased|elevated|reduced)\b',
            r'\d+(?:\.\d+)?\s*(?:-|\s*)?fold\s*(?:increase|decrease|change)',
            r'\b(?:compared\s+to|versus|vs\.?|relative\s+to)\b'
        ]
        
        expressions = []
        claim_text = claim.claim_text.lower()
        
        for pattern_str in comparative_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(claim_text)
            expressions.extend(matches)
        
        return expressions
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """Get statistics about verification performance."""
        
        stats = {
            'total_verifications': self.verification_stats['total_verifications'],
            'total_claims_verified': self.verification_stats['total_claims_verified'],
            'average_claims_per_verification': (
                self.verification_stats['total_claims_verified'] / 
                max(1, self.verification_stats['total_verifications'])
            ),
            'processing_times': {
                'count': len(self.processing_times),
                'average_ms': statistics.mean(self.processing_times) if self.processing_times else 0,
                'median_ms': statistics.median(self.processing_times) if self.processing_times else 0,
                'min_ms': min(self.processing_times) if self.processing_times else 0,
                'max_ms': max(self.processing_times) if self.processing_times else 0
            }
        }
        
        return stats


# Convenience functions for integration
async def verify_extracted_claims(
    claims: List['ExtractedClaim'],
    document_indexer: 'SourceDocumentIndex',
    config: Optional[Dict[str, Any]] = None
) -> VerificationReport:
    """
    Convenience function for verifying extracted claims.
    
    Args:
        claims: List of ExtractedClaim objects to verify
        document_indexer: SourceDocumentIndex instance for document lookup
        config: Optional verification configuration
        
    Returns:
        VerificationReport with comprehensive results
    """
    
    validator = FactualAccuracyValidator(document_indexer=document_indexer, config=config)
    return await validator.verify_claims(claims, config)


async def verify_claim_against_documents(
    claim_text: str,
    document_indexer: 'SourceDocumentIndex',
    claim_extractor: Optional['BiomedicalClaimExtractor'] = None,
    config: Optional[Dict[str, Any]] = None
) -> VerificationReport:
    """
    Convenience function for verifying a single claim text.
    
    Args:
        claim_text: Text of claim to verify
        document_indexer: SourceDocumentIndex instance
        claim_extractor: Optional BiomedicalClaimExtractor instance
        config: Optional verification configuration
        
    Returns:
        VerificationReport with verification results
    """
    
    # Extract claim if extractor provided
    if claim_extractor:
        extracted_claims = await claim_extractor.extract_claims(claim_text)
    else:
        # Create a basic ExtractedClaim
        from datetime import datetime
        basic_claim = type('ExtractedClaim', (), {
            'claim_id': hashlib.md5(claim_text.encode()).hexdigest()[:12],
            'claim_text': claim_text,
            'claim_type': 'general',
            'subject': '',
            'predicate': '',
            'object_value': '',
            'numeric_values': [],
            'units': [],
            'qualifiers': [],
            'keywords': claim_text.split()[:5],
            'relationships': [],
            'confidence': type('ClaimConfidence', (), {'overall_confidence': 50.0})()
        })()
        extracted_claims = [basic_claim]
    
    # Verify claims
    validator = FactualAccuracyValidator(
        document_indexer=document_indexer,
        claim_extractor=claim_extractor,
        config=config
    )
    
    return await validator.verify_claims(extracted_claims, config)


if __name__ == "__main__":
    # Simple test example
    async def test_factual_accuracy_validation():
        """Test the factual accuracy validation system."""
        
        # This would require actual document indexer and claim extractor instances
        print("Factual Accuracy Validator initialized successfully!")
        print("For full testing, integrate with SourceDocumentIndex and BiomedicalClaimExtractor")
        
        # Example of creating test verification result
        test_evidence = EvidenceItem(
            source_document="test_doc_001",
            evidence_text="glucose levels were 150 mg/dL",
            evidence_type="numeric",
            context="In diabetic patients, glucose levels were 150 mg/dL compared to 90 mg/dL in controls",
            confidence=85.0
        )
        
        test_result = VerificationResult(
            claim_id="test_claim_001",
            verification_status=VerificationStatus.SUPPORTED,
            verification_confidence=85.0,
            evidence_strength=75.0,
            context_match=80.0,
            supporting_evidence=[test_evidence],
            verification_strategy="numeric"
        )
        
        print(f"Test verification result: {test_result.verification_status.value}")
        print(f"Confidence: {test_result.verification_confidence}")
        print(f"Evidence grade: {test_result.verification_grade}")
        
    # Run test if executed directly
    asyncio.run(test_factual_accuracy_validation())