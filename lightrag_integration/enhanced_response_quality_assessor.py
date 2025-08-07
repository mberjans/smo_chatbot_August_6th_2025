#!/usr/bin/env python3
"""
Enhanced Response Quality Assessor with Factual Accuracy Integration.

This module provides an enhanced ResponseQualityAssessor class that integrates
with the factual accuracy validation pipeline to provide comprehensive quality
assessment for Clinical Metabolomics Oracle LightRAG responses.

Classes:
    - QualityAssessmentError: Base custom exception for quality assessment errors
    - ResponseQualityMetrics: Enhanced data class with factual validation results
    - EnhancedResponseQualityAssessor: Main class for integrated quality assessment

Key Features:
    - Integration with factual accuracy validation pipeline
    - Enhanced quality metrics with factual validation dimensions
    - Backwards compatibility with existing quality assessment workflows
    - Comprehensive error handling and fallback mechanisms
    - Performance optimization with async processing
    - Configurable validation components

Integration Points:
    - BiomedicalClaimExtractor for claim extraction
    - FactualAccuracyValidator for claim verification
    - FactualAccuracyScorer for comprehensive scoring
    - DocumentIndexer for source document analysis

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Quality Assessment Enhancement
"""

import asyncio
import json
import logging
import re
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, Counter

# Configure logging
logger = logging.getLogger(__name__)


class QualityAssessmentError(Exception):
    """Base custom exception for quality assessment errors."""
    pass


@dataclass
class ResponseQualityMetrics:
    """
    Enhanced response quality metrics with factual accuracy integration.
    
    Attributes:
        # Core quality scores
        relevance_score: Query-response relevance (0-100)
        accuracy_score: Overall accuracy assessment (0-100) 
        completeness_score: Completeness of response (0-100)
        clarity_score: Clarity and readability (0-100)
        biomedical_terminology_score: Biomedical terminology appropriateness (0-100)
        source_citation_score: Source citation quality (0-100)
        consistency_score: Internal consistency (0-100)
        
        # Enhanced factual accuracy scores
        factual_accuracy_score: Factual accuracy from validation pipeline (0-100)
        claim_verification_score: Individual claim verification quality (0-100)
        evidence_quality_score: Quality of supporting evidence (0-100)
        hallucination_score: Hallucination risk assessment (0-100)
        
        # Overall quality
        overall_quality_score: Weighted overall quality (0-100)
        
        # Assessment details
        key_concepts_covered: List of key concepts found in response
        missing_concepts: List of expected concepts not found
        biomedical_terms_found: List of biomedical terms identified
        citations_extracted: List of citations found
        quality_flags: List of quality issues identified
        assessment_details: Additional assessment metadata
        
        # Factual accuracy details
        factual_validation_results: Results from factual accuracy validation
        verified_claims_count: Number of claims successfully verified
        contradicted_claims_count: Number of claims contradicted by evidence
        claims_without_evidence_count: Number of claims lacking evidence
        factual_confidence_score: Confidence in factual assessment (0-100)
    """
    # Core quality scores
    relevance_score: float
    accuracy_score: float
    completeness_score: float
    clarity_score: float
    biomedical_terminology_score: float
    source_citation_score: float
    consistency_score: float
    
    # Enhanced factual accuracy scores
    factual_accuracy_score: float
    hallucination_score: float
    overall_quality_score: float
    claim_verification_score: float = 0.0
    evidence_quality_score: float = 0.0
    
    # Assessment details
    key_concepts_covered: List[str] = field(default_factory=list)
    missing_concepts: List[str] = field(default_factory=list)
    biomedical_terms_found: List[str] = field(default_factory=list)
    citations_extracted: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    assessment_details: Dict[str, Any] = field(default_factory=dict)
    
    # Factual accuracy details
    factual_validation_results: Dict[str, Any] = field(default_factory=dict)
    verified_claims_count: int = 0
    contradicted_claims_count: int = 0
    claims_without_evidence_count: int = 0
    factual_confidence_score: float = 0.0
    
    @property
    def quality_grade(self) -> str:
        """Convert overall quality score to human-readable grade."""
        if self.overall_quality_score >= 90:
            return "Excellent"
        elif self.overall_quality_score >= 80:
            return "Good"
        elif self.overall_quality_score >= 70:
            return "Acceptable"
        elif self.overall_quality_score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"
    
    @property
    def factual_reliability_grade(self) -> str:
        """Convert factual accuracy score to reliability grade."""
        if self.factual_accuracy_score >= 90:
            return "Highly Reliable"
        elif self.factual_accuracy_score >= 80:
            return "Reliable"
        elif self.factual_accuracy_score >= 70:
            return "Moderately Reliable"
        elif self.factual_accuracy_score >= 60:
            return "Questionable"
        else:
            return "Unreliable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return asdict(self)


class EnhancedResponseQualityAssessor:
    """
    Enhanced response quality assessor with factual accuracy integration.
    
    Provides comprehensive quality assessment by combining traditional
    quality metrics with advanced factual accuracy validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced quality assessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize biomedical keywords for assessment
        self.biomedical_keywords = {
            'metabolomics_core': [
                'metabolomics', 'metabolite', 'metabolism', 'biomarker',
                'mass spectrometry', 'NMR', 'chromatography', 'metabolic pathway'
            ],
            'clinical_terms': [
                'clinical', 'patient', 'disease', 'diagnosis', 'therapeutic',
                'biomedical', 'pathology', 'phenotype', 'precision medicine'
            ],
            'analytical_methods': [
                'LC-MS', 'GC-MS', 'UPLC', 'HILIC', 'targeted analysis',
                'untargeted analysis', 'quantitative', 'qualitative'
            ],
            'research_concepts': [
                'study design', 'statistical analysis', 'p-value',
                'effect size', 'confidence interval', 'validation'
            ]
        }
        
        # Quality assessment weights
        self.quality_weights = {
            'relevance': 0.20,
            'accuracy': 0.15,
            'completeness': 0.15,
            'clarity': 0.10,
            'biomedical_terminology': 0.08,
            'source_citation': 0.07,
            'factual_accuracy': 0.25  # New factual accuracy dimension
        }
        
        # Initialize factual accuracy components
        self._claim_extractor = None
        self._factual_validator = None
        self._accuracy_scorer = None
        self._initialize_factual_components()
        
        logger.info("EnhancedResponseQualityAssessor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for quality assessment."""
        return {
            'factual_validation_enabled': True,
            'fallback_on_error': True,
            'minimum_claims_for_reliable_score': 3,
            'performance_timeout_seconds': 10.0,
            'enable_caching': True,
            'detailed_reporting': True
        }
    
    def _initialize_factual_components(self):
        """Initialize factual accuracy validation components."""
        try:
            if self.config.get('factual_validation_enabled', True):
                # Try to import and initialize components
                try:
                    from .claim_extractor import BiomedicalClaimExtractor
                    self._claim_extractor = BiomedicalClaimExtractor()
                    logger.info("BiomedicalClaimExtractor initialized in quality assessor")
                except ImportError:
                    logger.warning("BiomedicalClaimExtractor not available - using fallback")
                
                try:
                    from .factual_accuracy_validator import FactualAccuracyValidator
                    self._factual_validator = FactualAccuracyValidator()
                    logger.info("FactualAccuracyValidator initialized in quality assessor")
                except ImportError:
                    logger.warning("FactualAccuracyValidator not available - using fallback")
                
                try:
                    from .accuracy_scorer import FactualAccuracyScorer
                    self._accuracy_scorer = FactualAccuracyScorer()
                    logger.info("FactualAccuracyScorer initialized in quality assessor")
                except ImportError:
                    logger.warning("FactualAccuracyScorer not available - using fallback")
            else:
                logger.info("Factual validation disabled in configuration")
        except Exception as e:
            logger.error(f"Error initializing factual accuracy components: {str(e)}")
    
    def enable_factual_validation(self, 
                                 claim_extractor=None,
                                 factual_validator=None,
                                 accuracy_scorer=None):
        """
        Enable factual validation with external components.
        
        Args:
            claim_extractor: BiomedicalClaimExtractor instance
            factual_validator: FactualAccuracyValidator instance
            accuracy_scorer: FactualAccuracyScorer instance
        """
        if claim_extractor:
            self._claim_extractor = claim_extractor
        if factual_validator:
            self._factual_validator = factual_validator
        if accuracy_scorer:
            self._accuracy_scorer = accuracy_scorer
        
        self.config['factual_validation_enabled'] = True
        logger.info("Factual validation components enabled")
    
    async def assess_response_quality(self,
                                    query: str,
                                    response: str,
                                    source_documents: Optional[List[str]] = None,
                                    expected_concepts: Optional[List[str]] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> ResponseQualityMetrics:
        """
        Perform comprehensive quality assessment with factual accuracy integration.
        
        Args:
            query: Original user query
            response: System response to assess
            source_documents: Optional source documents for validation
            expected_concepts: Optional list of expected concepts
            metadata: Optional metadata for assessment context
            
        Returns:
            ResponseQualityMetrics with comprehensive assessment
            
        Raises:
            QualityAssessmentError: If assessment fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting comprehensive quality assessment for response ({len(response)} chars)")
            
            # Input validation
            if not query or not response:
                raise QualityAssessmentError("Query and response are required")
            
            # Set defaults
            source_documents = source_documents or []
            expected_concepts = expected_concepts or []
            metadata = metadata or {}
            
            # Calculate core quality metrics
            core_metrics = await self._calculate_core_quality_metrics(
                query, response, source_documents, expected_concepts
            )
            
            # Calculate factual accuracy metrics
            factual_metrics = await self._calculate_factual_accuracy_metrics(
                query, response, source_documents, metadata
            )
            
            # Extract detailed information
            assessment_details = self._extract_assessment_details(query, response)
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_quality_score(core_metrics, factual_metrics)
            
            # Create comprehensive metrics
            quality_metrics = ResponseQualityMetrics(
                # Core scores
                relevance_score=core_metrics['relevance'],
                accuracy_score=core_metrics['accuracy'], 
                completeness_score=core_metrics['completeness'],
                clarity_score=core_metrics['clarity'],
                biomedical_terminology_score=core_metrics['biomedical_terminology'],
                source_citation_score=core_metrics['source_citation'],
                consistency_score=core_metrics['consistency'],
                
                # Factual accuracy scores
                factual_accuracy_score=factual_metrics['factual_accuracy'],
                claim_verification_score=factual_metrics['claim_verification'],
                evidence_quality_score=factual_metrics['evidence_quality'], 
                hallucination_score=factual_metrics['hallucination'],
                
                # Overall quality
                overall_quality_score=overall_score,
                
                # Details
                key_concepts_covered=assessment_details['key_concepts'],
                missing_concepts=assessment_details['missing_concepts'],
                biomedical_terms_found=assessment_details['biomedical_terms'],
                citations_extracted=assessment_details['citations'],
                quality_flags=assessment_details['quality_flags'],
                assessment_details=assessment_details['details'],
                
                # Factual details
                factual_validation_results=factual_metrics['validation_results'],
                verified_claims_count=factual_metrics['verified_claims'],
                contradicted_claims_count=factual_metrics['contradicted_claims'],
                claims_without_evidence_count=factual_metrics['claims_without_evidence'],
                factual_confidence_score=factual_metrics['confidence']
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"Quality assessment completed in {processing_time:.2f}ms: "
                f"Overall {overall_score:.1f}/100, Factual {factual_metrics['factual_accuracy']:.1f}/100"
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error during quality assessment: {str(e)}")
            raise QualityAssessmentError(f"Quality assessment failed: {str(e)}") from e
    
    async def _calculate_core_quality_metrics(self,
                                            query: str,
                                            response: str,
                                            source_documents: List[str],
                                            expected_concepts: List[str]) -> Dict[str, float]:
        """Calculate core quality metrics (non-factual)."""
        return {
            'relevance': self._assess_relevance(query, response),
            'accuracy': self._assess_accuracy(response, source_documents),
            'completeness': self._assess_completeness(response, expected_concepts),
            'clarity': self._assess_clarity(response),
            'biomedical_terminology': self._assess_biomedical_terminology(response),
            'source_citation': self._assess_source_citation(response),
            'consistency': await self._assess_consistency(query, response)
        }
    
    async def _calculate_factual_accuracy_metrics(self,
                                                query: str, 
                                                response: str,
                                                source_documents: List[str],
                                                metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate factual accuracy metrics using validation pipeline."""
        try:
            # Check if we have factual validation components
            if not self._has_factual_components():
                return await self._calculate_fallback_factual_metrics(response, source_documents)
            
            # Extract claims from response
            claims = await self._claim_extractor.extract_claims(response)
            if not claims:
                logger.info("No claims extracted - using fallback factual assessment")
                return await self._calculate_fallback_factual_metrics(response, source_documents)
            
            # Verify claims against documents
            verification_report = await self._factual_validator.verify_claims(claims)
            
            # Score the verification results
            accuracy_score = await self._accuracy_scorer.score_accuracy(
                verification_report.verification_results, claims
            )
            
            # Process results into metrics
            return self._process_factual_validation_results(
                accuracy_score, verification_report, claims
            )
            
        except Exception as e:
            logger.warning(f"Error in factual accuracy calculation: {str(e)}")
            if self.config.get('fallback_on_error', True):
                return await self._calculate_fallback_factual_metrics(response, source_documents)
            else:
                raise
    
    def _has_factual_components(self) -> bool:
        """Check if factual validation components are available."""
        return (self._claim_extractor is not None and 
                self._factual_validator is not None and
                self._accuracy_scorer is not None)
    
    async def _calculate_fallback_factual_metrics(self,
                                                response: str,
                                                source_documents: List[str]) -> Dict[str, float]:
        """Calculate fallback factual metrics without full pipeline."""
        # Basic factual accuracy assessment
        factual_accuracy = self._assess_factual_accuracy_basic(response, source_documents)
        
        # Basic claim verification assessment
        claim_verification = self._assess_claim_patterns(response)
        
        # Basic evidence quality assessment
        evidence_quality = self._assess_evidence_indicators(response)
        
        # Hallucination risk assessment
        hallucination = self._assess_hallucination_risk(response, source_documents)
        
        return {
            'factual_accuracy': factual_accuracy,
            'claim_verification': claim_verification,
            'evidence_quality': evidence_quality,
            'hallucination': hallucination,
            'validation_results': {'method': 'fallback'},
            'verified_claims': 0,
            'contradicted_claims': 0, 
            'claims_without_evidence': 0,
            'confidence': 60.0  # Lower confidence for fallback
        }
    
    def _process_factual_validation_results(self,
                                          accuracy_score,
                                          verification_report,
                                          claims) -> Dict[str, float]:
        """Process factual validation results into metrics."""
        # Count claim verification outcomes
        verified_claims = sum(
            1 for result in verification_report.verification_results
            if result.verification_status.value == 'SUPPORTED'
        )
        
        contradicted_claims = sum(
            1 for result in verification_report.verification_results  
            if result.verification_status.value == 'CONTRADICTED'
        )
        
        claims_without_evidence = sum(
            1 for result in verification_report.verification_results
            if result.verification_status.value == 'NOT_FOUND'
        )
        
        return {
            'factual_accuracy': accuracy_score.overall_score,
            'claim_verification': accuracy_score.dimension_scores.get('claim_verification', 0.0),
            'evidence_quality': accuracy_score.evidence_quality_score,
            'hallucination': 100.0 - (contradicted_claims / max(len(claims), 1) * 50),
            'validation_results': {
                'method': 'full_pipeline',
                'accuracy_score': accuracy_score.to_dict(),
                'verification_report': verification_report.to_dict()
            },
            'verified_claims': verified_claims,
            'contradicted_claims': contradicted_claims,
            'claims_without_evidence': claims_without_evidence,
            'confidence': accuracy_score.confidence_score
        }
    
    # Core quality assessment methods (existing implementations enhanced)
    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess query-response relevance."""
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        query_terms -= common_words
        response_terms -= common_words
        
        if not query_terms:
            return 50.0
        
        overlap = len(query_terms.intersection(response_terms))
        relevance_ratio = overlap / len(query_terms)
        
        # Biomedical context bonus
        biomedical_bonus = 0
        if 'clinical' in query.lower() or 'metabolomics' in query.lower():
            biomedical_terms = [term for term_list in self.biomedical_keywords.values() 
                              for term in term_list]
            biomedical_bonus = min(20, sum(2 for term in biomedical_terms 
                                         if term in response.lower()))
        
        return min(100, (relevance_ratio * 80) + biomedical_bonus)
    
    def _assess_accuracy(self, response: str, source_documents: List[str]) -> float:
        """Assess general accuracy indicators."""
        accuracy_score = 75.0
        
        # Evidence indicators
        evidence_indicators = [
            'studies show', 'research indicates', 'according to',
            'evidence suggests', 'data demonstrates', 'findings reveal'
        ]
        
        for indicator in evidence_indicators:
            if indicator in response.lower():
                accuracy_score += 5
        
        # Penalize absolute claims
        absolute_claims = ['always', 'never', 'all', 'none', 'every', 'completely']
        for claim in absolute_claims:
            if claim in response.lower():
                accuracy_score -= 3
        
        return min(100, max(0, accuracy_score))
    
    def _assess_completeness(self, response: str, expected_concepts: List[str]) -> float:
        """Assess response completeness."""
        if not expected_concepts:
            return 80.0
        
        concepts_covered = sum(1 for concept in expected_concepts 
                             if concept.lower() in response.lower())
        completeness_ratio = concepts_covered / len(expected_concepts)
        
        # Length-based adjustment
        if len(response) < 100:
            length_penalty = 20
        elif len(response) < 200:
            length_penalty = 10
        else:
            length_penalty = 0
        
        return min(100, (completeness_ratio * 80) + 20 - length_penalty)
    
    def _assess_clarity(self, response: str) -> float:
        """Assess response clarity and readability."""
        words = response.split()
        sentences = re.findall(r'[.!?]+', response)
        
        if not words or not sentences:
            return 20.0
        
        # Sentence length assessment
        avg_sentence_length = len(words) / len(sentences)
        if 15 <= avg_sentence_length <= 25:
            length_score = 40
        elif 10 <= avg_sentence_length <= 30:
            length_score = 30
        else:
            length_score = 20
        
        # Structure indicators
        structure_indicators = ['first', 'second', 'furthermore', 'moreover', 
                               'however', 'therefore', 'in conclusion']
        structure_score = min(30, sum(5 for indicator in structure_indicators 
                                    if indicator in response.lower()))
        
        # Technical term balance
        technical_terms = sum(1 for term_list in self.biomedical_keywords.values() 
                            for term in term_list if term in response.lower())
        jargon_ratio = technical_terms / len(words) * 100
        
        if 2 <= jargon_ratio <= 8:
            jargon_score = 30
        elif 1 <= jargon_ratio <= 10:
            jargon_score = 20
        else:
            jargon_score = 10
        
        return length_score + structure_score + jargon_score
    
    def _assess_biomedical_terminology(self, response: str) -> float:
        """Assess biomedical terminology usage."""
        response_lower = response.lower()
        total_terms = 0
        found_terms = 0
        
        for category, terms in self.biomedical_keywords.items():
            for term in terms:
                total_terms += 1
                if term in response_lower:
                    found_terms += 1
        
        if total_terms == 0:
            return 50.0
        
        terminology_ratio = found_terms / total_terms
        
        # Category diversity bonus
        categories_used = sum(1 for category, terms in self.biomedical_keywords.items()
                            if any(term in response_lower for term in terms))
        diversity_bonus = categories_used * 5
        
        return min(100, (terminology_ratio * 70) + diversity_bonus + 20)
    
    def _assess_source_citation(self, response: str) -> float:
        """Assess source citation quality."""
        citation_patterns = [
            r'\[[0-9]+\]',
            r'\([A-Za-z]+.*?\d{4}\)',
            r'et al\.',
            r'according to',
            r'study by',
            r'research from'
        ]
        
        citations_found = sum(len(re.findall(pattern, response, re.IGNORECASE)) 
                            for pattern in citation_patterns)
        
        if citations_found > 0:
            return 60 + min(40, citations_found * 10)
        else:
            # Check for evidence indicators
            evidence_indicators = ['studies show', 'research indicates', 'data suggests']
            if any(indicator in response.lower() for indicator in evidence_indicators):
                return 40
            return 20
    
    async def _assess_consistency(self, query: str, response: str) -> float:
        """Assess response consistency."""
        consistency_indicators = [
            len(response) > 100,
            'metabolomics' in response.lower() if 'metabolomics' in query.lower() else True,
            not any(contradiction in response.lower() for contradiction in ['however', 'but', 'although']),
        ]
        
        consistency_score = sum(20 for indicator in consistency_indicators if indicator) + 40
        return min(100, consistency_score)
    
    # Enhanced factual accuracy assessment methods
    def _assess_factual_accuracy_basic(self, response: str, source_documents: List[str]) -> float:
        """Basic factual accuracy assessment."""
        base_score = 70.0
        
        # Evidence support indicators
        evidence_patterns = ['studies show', 'research demonstrates', 'data indicates',
                           'clinical trials', 'meta-analysis', 'systematic review']
        evidence_count = sum(1 for pattern in evidence_patterns if pattern in response.lower())
        evidence_bonus = min(15.0, evidence_count * 5.0)
        
        # Uncertainty acknowledgment (positive)
        uncertainty_patterns = ['may', 'might', 'suggests', 'preliminary', 'limited evidence']
        uncertainty_count = sum(1 for pattern in uncertainty_patterns if pattern in response.lower())
        uncertainty_bonus = min(10.0, uncertainty_count * 3.0)
        
        # Overconfident claims (negative)
        overconfident_patterns = ['always', 'never', 'proven', 'guaranteed', 'definitely']
        overconfident_count = sum(1 for pattern in overconfident_patterns if pattern in response.lower())
        overconfident_penalty = min(20.0, overconfident_count * 4.0)
        
        return min(100.0, max(0.0, base_score + evidence_bonus + uncertainty_bonus - overconfident_penalty))
    
    def _assess_claim_patterns(self, response: str) -> float:
        """Assess factual claim patterns in response."""
        # Look for specific claims that can be verified
        numeric_claims = len(re.findall(r'\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:mg|kg|ml|µM|nM|fold)', response))
        comparative_claims = len(re.findall(r'(?:higher|lower|increased|decreased|significant)', response, re.IGNORECASE))
        causal_claims = len(re.findall(r'(?:causes?|leads? to|results? in|due to)', response, re.IGNORECASE))
        
        total_claims = numeric_claims + comparative_claims + causal_claims
        
        if total_claims == 0:
            return 60.0  # Neutral for no specific claims
        
        # More claims require higher verification standards
        if total_claims <= 3:
            return 80.0
        elif total_claims <= 6:
            return 75.0
        else:
            return 70.0  # Many claims need careful verification
    
    def _assess_evidence_indicators(self, response: str) -> float:
        """Assess evidence quality indicators."""
        high_quality_indicators = [
            'peer-reviewed', 'systematic review', 'meta-analysis',
            'randomized controlled trial', 'clinical trial'
        ]
        medium_quality_indicators = [
            'study', 'research', 'investigation', 'analysis', 'data'
        ]
        
        high_count = sum(1 for indicator in high_quality_indicators 
                        if indicator in response.lower())
        medium_count = sum(1 for indicator in medium_quality_indicators 
                          if indicator in response.lower())
        
        evidence_score = (high_count * 20) + (medium_count * 10)
        return min(100.0, evidence_score + 50)  # Base score of 50
    
    def _assess_hallucination_risk(self, response: str, source_documents: List[str]) -> float:
        """Assess hallucination risk."""
        high_risk_indicators = [
            'breakthrough', 'revolutionary', 'miracle', 'unprecedented',
            'i believe', 'i think', 'definitely', 'absolutely certain'
        ]
        
        risk_count = sum(1 for indicator in high_risk_indicators 
                        if indicator in response.lower())
        
        # Base hallucination score (higher is better)
        hallucination_score = 90.0 - (risk_count * 10)
        
        # Bonus for evidence indicators
        evidence_indicators = ['study', 'research', 'data', 'evidence', 'analysis']
        evidence_count = sum(1 for indicator in evidence_indicators 
                           if indicator in response.lower())
        evidence_bonus = min(10.0, evidence_count * 2.0)
        
        return min(100.0, max(10.0, hallucination_score + evidence_bonus))
    
    def _extract_assessment_details(self, query: str, response: str) -> Dict[str, Any]:
        """Extract detailed assessment information."""
        # Extract key concepts
        key_concepts = []
        for term_list in self.biomedical_keywords.values():
            for term in term_list:
                if term in response.lower():
                    key_concepts.append(term)
        
        # Extract biomedical terms
        biomedical_terms = []
        for category, terms in self.biomedical_keywords.items():
            for term in terms:
                if term in response.lower():
                    biomedical_terms.append(term)
        
        # Extract citations
        citation_patterns = [
            r'\[[0-9]+\]',
            r'\([A-Za-z]+.*?\d{4}\)',
            r'[A-Za-z]+ et al\. \(\d{4}\)'
        ]
        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, response))
        
        # Identify quality flags
        quality_flags = []
        if len(response) < 50:
            quality_flags.append("response_too_short")
        if len(response) > 2000:
            quality_flags.append("response_very_long")
        if response.count('?') > 3:
            quality_flags.append("too_many_questions")
        if not biomedical_terms:
            quality_flags.append("lacks_biomedical_terminology")
        
        uncertainty_indicators = ['maybe', 'perhaps', 'possibly', 'might', 'could be']
        if sum(1 for indicator in uncertainty_indicators if indicator in response.lower()) > 2:
            quality_flags.append("high_uncertainty")
        
        # Calculate technical density
        words = response.lower().split()
        if words:
            technical_words = sum(1 for word in words
                                for term_list in self.biomedical_keywords.values()
                                for term in term_list if term in word)
            technical_density = technical_words / len(words) * 100
        else:
            technical_density = 0.0
        
        return {
            'key_concepts': list(set(key_concepts)),
            'missing_concepts': [],  # Would need expected concepts to calculate
            'biomedical_terms': list(set(biomedical_terms)),
            'citations': citations,
            'quality_flags': quality_flags,
            'details': {
                'response_length': len(response),
                'word_count': len(words) if words else 0,
                'sentence_count': len(re.findall(r'[.!?]+', response)),
                'paragraph_count': len(response.split('\n\n')),
                'technical_density': technical_density,
                'query_length': len(query)
            }
        }
    
    def _calculate_overall_quality_score(self, 
                                       core_metrics: Dict[str, float],
                                       factual_metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        # Combine core and factual metrics
        all_metrics = {**core_metrics, **factual_metrics}
        
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in self.quality_weights.items():
            if dimension in all_metrics:
                total_score += all_metrics[dimension] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    async def batch_assess_quality(self, 
                                 assessments: List[Tuple[str, str, List[str], List[str]]]) -> List[ResponseQualityMetrics]:
        """
        Perform batch quality assessment for multiple query-response pairs.
        
        Args:
            assessments: List of (query, response, source_docs, expected_concepts) tuples
            
        Returns:
            List of ResponseQualityMetrics for each assessment
        """
        results = []
        
        for query, response, source_docs, expected_concepts in assessments:
            try:
                result = await self.assess_response_quality(
                    query, response, source_docs, expected_concepts
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch assessment: {str(e)}")
                # Create fallback result
                fallback_result = ResponseQualityMetrics(
                    relevance_score=0.0, accuracy_score=0.0, completeness_score=0.0,
                    clarity_score=0.0, biomedical_terminology_score=0.0, source_citation_score=0.0,
                    consistency_score=0.0, factual_accuracy_score=0.0, hallucination_score=0.0,
                    overall_quality_score=0.0,
                    quality_flags=["assessment_error"]
                )
                results.append(fallback_result)
        
        return results


# Convenience functions for easy integration
async def assess_response_quality(query: str, 
                                response: str,
                                source_documents: Optional[List[str]] = None,
                                expected_concepts: Optional[List[str]] = None,
                                config: Optional[Dict[str, Any]] = None) -> ResponseQualityMetrics:
    """
    Convenience function for response quality assessment.
    
    Args:
        query: Original user query
        response: System response to assess
        source_documents: Optional source documents
        expected_concepts: Optional expected concepts
        config: Optional configuration
        
    Returns:
        ResponseQualityMetrics with comprehensive assessment
    """
    assessor = EnhancedResponseQualityAssessor(config)
    return await assessor.assess_response_quality(
        query, response, source_documents, expected_concepts
    )


if __name__ == "__main__":
    # Simple test example
    async def test_enhanced_quality_assessment():
        """Test the enhanced quality assessment system."""
        
        print("Enhanced Response Quality Assessor Test")
        print("=" * 50)
        
        assessor = EnhancedResponseQualityAssessor()
        
        # Test query and response
        query = "What is metabolomics and how is it used in clinical applications?"
        response = """Metabolomics is the comprehensive study of small molecules (metabolites) in biological systems. 
        In clinical applications, metabolomics enables biomarker discovery for disease diagnosis and treatment monitoring. 
        LC-MS and GC-MS are commonly used analytical platforms for metabolite analysis. Research indicates that metabolomics 
        shows promise for precision medicine approaches."""
        
        # Perform assessment
        result = await assessor.assess_response_quality(
            query=query,
            response=response,
            source_documents=[],
            expected_concepts=["metabolomics", "clinical", "biomarker"]
        )
        
        print(f"Overall Quality Score: {result.overall_quality_score:.2f}/100")
        print(f"Quality Grade: {result.quality_grade}")
        print(f"Factual Accuracy Score: {result.factual_accuracy_score:.2f}/100")
        print(f"Factual Reliability Grade: {result.factual_reliability_grade}")
        print(f"\nCore Metrics:")
        print(f"  Relevance: {result.relevance_score:.1f}/100")
        print(f"  Clarity: {result.clarity_score:.1f}/100")
        print(f"  Biomedical Terminology: {result.biomedical_terminology_score:.1f}/100")
        print(f"\nFactual Metrics:")
        print(f"  Hallucination Risk: {100-result.hallucination_score:.1f}/100")
        print(f"  Evidence Quality: {result.evidence_quality_score:.1f}/100")
        print(f"\nBiomedical Terms Found: {result.biomedical_terms_found}")
        print(f"Quality Flags: {result.quality_flags}")
        
    # Run test
    asyncio.run(test_enhanced_quality_assessment())