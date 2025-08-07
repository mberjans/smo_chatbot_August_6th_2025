#!/usr/bin/env python3
"""
Factual Accuracy Scoring and Reporting System for Clinical Metabolomics Oracle.

This module provides the FactualAccuracyScorer class for comprehensive scoring and reporting
of factual accuracy validation results in the Clinical Metabolomics Oracle LightRAG integration
project. It takes verification results from the FactualAccuracyValidator and generates detailed
scores and reports for integration with existing quality assessment systems.

Classes:
    - AccuracyScoringError: Base custom exception for accuracy scoring errors
    - ReportGenerationError: Exception for report generation failures
    - QualityIntegrationError: Exception for quality system integration failures
    - AccuracyScore: Data class for structured accuracy scoring results
    - AccuracyReport: Data class for comprehensive accuracy reports
    - AccuracyMetrics: Data class for performance and quality metrics
    - FactualAccuracyScorer: Main class for accuracy scoring and reporting

The scorer handles:
    - Multi-dimensional accuracy scoring from verification results
    - Comprehensive report generation with detailed breakdowns
    - Integration with existing quality assessment pipeline (ClinicalMetabolomicsRelevanceScorer)
    - Performance metrics and system health monitoring
    - Configurable scoring weights and thresholds
    - Quality recommendations for accuracy improvement
    - Standardized output formats for system integration

Key Features:
    - Overall factual accuracy score calculation (0-100)
    - Claim type-specific scoring (numeric, qualitative, methodological, etc.)
    - Evidence quality assessment and scoring
    - Coverage analysis of claims vs source documents
    - Consistency scoring across multiple claims
    - Performance tracking and optimization
    - Integration data for existing quality systems
    - Comprehensive reporting with actionable insights
    - Configuration management for flexible scoring
    - Error handling and recovery mechanisms

Scoring Dimensions:
    - Overall Accuracy: Weighted aggregate of all claim verifications
    - Claim Type Scores: Separate scores for different claim types
    - Evidence Quality: Assessment of supporting evidence strength
    - Coverage Score: How well claims are covered by source documents
    - Consistency Score: Internal consistency across claims
    - Verification Confidence: Confidence in the verification process
    - Processing Performance: Speed and efficiency metrics

Integration Features:
    - ClinicalMetabolomicsRelevanceScorer compatibility
    - Standard JSON output formats
    - Quality pipeline data structures
    - Performance monitoring integration
    - Configuration inheritance from existing systems

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Factual Accuracy Scoring Implementation
"""

import asyncio
import json
import logging
import time
import statistics
import hashlib
import math
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
    from .factual_accuracy_validator import (
        FactualAccuracyValidator, VerificationResult, VerificationReport, VerificationStatus
    )
    from .claim_extractor import ExtractedClaim
    from .relevance_scorer import ClinicalMetabolomicsRelevanceScorer, RelevanceScore
except ImportError:
    # Handle import errors gracefully
    logging.warning("Could not import validation components - some features may be limited")

# Configure logging
logger = logging.getLogger(__name__)


class AccuracyScoringError(Exception):
    """Base custom exception for accuracy scoring errors."""
    pass


class ReportGenerationError(AccuracyScoringError):
    """Exception raised when report generation fails."""
    pass


class QualityIntegrationError(AccuracyScoringError):
    """Exception raised when quality system integration fails."""
    pass


class AccuracyGrade(Enum):
    """Enumeration of accuracy grade values."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    ACCEPTABLE = "Acceptable"
    MARGINAL = "Marginal"
    POOR = "Poor"


@dataclass
class AccuracyScore:
    """
    Comprehensive factual accuracy scoring results.
    
    Attributes:
        overall_score: Overall factual accuracy score (0-100)
        dimension_scores: Scores for each accuracy dimension
        claim_type_scores: Scores broken down by claim type
        evidence_quality_score: Overall evidence quality assessment
        coverage_score: Coverage of claims by source documents
        consistency_score: Internal consistency across claims
        confidence_score: Confidence in the accuracy assessment
        grade: Human-readable accuracy grade
        total_claims_assessed: Number of claims included in scoring
        processing_time_ms: Time taken for scoring in milliseconds
        metadata: Additional scoring metadata
    """
    overall_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    claim_type_scores: Dict[str, float] = field(default_factory=dict)
    evidence_quality_score: float = 0.0
    coverage_score: float = 0.0
    consistency_score: float = 0.0
    confidence_score: float = 0.0
    grade: AccuracyGrade = AccuracyGrade.POOR
    total_claims_assessed: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def accuracy_percentage(self) -> str:
        """Return accuracy as formatted percentage."""
        return f"{self.overall_score:.1f}%"
    
    @property
    def is_reliable(self) -> bool:
        """Check if accuracy is considered reliable (>= 70%)."""
        return self.overall_score >= 70.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert accuracy score to dictionary representation."""
        result = asdict(self)
        result['grade'] = self.grade.value
        return result


@dataclass
class AccuracyMetrics:
    """
    Performance and quality metrics for accuracy assessment.
    
    Attributes:
        verification_performance: Performance metrics from verification process
        scoring_performance: Performance metrics from scoring process
        quality_indicators: Quality indicators for the assessment process
        system_health: System health metrics
        resource_usage: Resource usage statistics
        error_rates: Error rate tracking
        coverage_statistics: Coverage analysis statistics
        recommendation_counts: Count of different recommendation types
    """
    verification_performance: Dict[str, float] = field(default_factory=dict)
    scoring_performance: Dict[str, float] = field(default_factory=dict)
    quality_indicators: Dict[str, float] = field(default_factory=dict)
    system_health: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    coverage_statistics: Dict[str, int] = field(default_factory=dict)
    recommendation_counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return asdict(self)


@dataclass
class AccuracyReport:
    """
    Comprehensive factual accuracy report.
    
    Attributes:
        report_id: Unique identifier for the report
        accuracy_score: Overall accuracy scoring results
        detailed_breakdown: Detailed breakdown by claim and verification
        summary_statistics: Summary statistics for the assessment
        performance_metrics: Performance and system metrics
        quality_recommendations: Recommendations for improving accuracy
        integration_data: Data for integration with quality systems
        claims_analysis: Detailed analysis of individual claims
        evidence_analysis: Analysis of evidence quality and sources
        coverage_analysis: Analysis of claim coverage by source documents
        created_timestamp: When the report was created
        configuration_used: Configuration used for scoring
    """
    report_id: str
    accuracy_score: AccuracyScore
    detailed_breakdown: Dict[str, Any] = field(default_factory=dict)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    quality_recommendations: List[str] = field(default_factory=list)
    integration_data: Dict[str, Any] = field(default_factory=dict)
    claims_analysis: List[Dict[str, Any]] = field(default_factory=list)
    evidence_analysis: Dict[str, Any] = field(default_factory=dict)
    coverage_analysis: Dict[str, Any] = field(default_factory=dict)
    created_timestamp: datetime = field(default_factory=datetime.now)
    configuration_used: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        result = asdict(self)
        result['created_timestamp'] = self.created_timestamp.isoformat()
        result['accuracy_score'] = self.accuracy_score.to_dict()
        result['performance_metrics'] = self.performance_metrics.to_dict()
        return result
    
    @property
    def report_summary(self) -> str:
        """Generate brief report summary."""
        return (
            f"Factual Accuracy Report {self.report_id}\n"
            f"Overall Accuracy: {self.accuracy_score.accuracy_percentage} ({self.accuracy_score.grade.value})\n"
            f"Claims Assessed: {self.accuracy_score.total_claims_assessed}\n"
            f"Evidence Quality: {self.accuracy_score.evidence_quality_score:.1f}/100\n"
            f"Coverage Score: {self.accuracy_score.coverage_score:.1f}/100\n"
            f"Generated: {self.created_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )


class FactualAccuracyScorer:
    """
    Main class for comprehensive factual accuracy scoring and reporting.
    
    Provides comprehensive scoring capabilities including:
    - Multi-dimensional accuracy scoring
    - Claim type-specific assessment
    - Evidence quality evaluation
    - Coverage and consistency analysis
    - Performance metrics tracking
    - Integration with existing quality systems
    - Comprehensive reporting and recommendations
    """
    
    def __init__(self, 
                 relevance_scorer: Optional['ClinicalMetabolomicsRelevanceScorer'] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FactualAccuracyScorer.
        
        Args:
            relevance_scorer: Optional ClinicalMetabolomicsRelevanceScorer instance
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.relevance_scorer = relevance_scorer
        self.logger = logger
        
        # Initialize scoring weights and parameters
        self._initialize_scoring_parameters()
        
        # Initialize grading thresholds
        self._initialize_grading_thresholds()
        
        # Initialize integration mappings
        self._initialize_integration_mappings()
        
        # Performance tracking
        self.scoring_stats = defaultdict(int)
        self.processing_times = []
        
        self.logger.info("FactualAccuracyScorer initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for accuracy scoring."""
        return {
            'scoring_weights': {
                'claim_verification': 0.35,
                'evidence_quality': 0.25,
                'coverage_assessment': 0.20,
                'consistency_analysis': 0.15,
                'confidence_factor': 0.05
            },
            'claim_type_weights': {
                'numeric': 1.2,        # Higher weight for numeric claims
                'qualitative': 1.0,    # Standard weight
                'methodological': 1.1, # Slightly higher for methodological
                'temporal': 0.9,       # Slightly lower for temporal
                'comparative': 1.1,    # Higher for comparative
                'general': 0.8         # Lower for general claims
            },
            'evidence_quality_thresholds': {
                'high_quality': 80.0,
                'medium_quality': 60.0,
                'low_quality': 40.0
            },
            'coverage_requirements': {
                'excellent_coverage': 0.9,
                'good_coverage': 0.7,
                'acceptable_coverage': 0.5
            },
            'consistency_thresholds': {
                'high_consistency': 0.85,
                'medium_consistency': 0.65,
                'low_consistency': 0.45
            },
            'performance_targets': {
                'max_processing_time_ms': 5000,
                'min_claims_for_reliable_score': 3,
                'max_error_rate': 0.05
            },
            'integration_settings': {
                'enable_relevance_integration': True,
                'quality_system_compatibility': True,
                'generate_integration_data': True
            }
        }
    
    def _initialize_scoring_parameters(self):
        """Initialize scoring parameters and weights."""
        
        # Extract weights from config
        self.scoring_weights = self.config['scoring_weights']
        self.claim_type_weights = self.config['claim_type_weights']
        
        # Evidence assessment parameters
        self.evidence_quality_factors = {
            'source_credibility': 0.30,
            'evidence_strength': 0.25,
            'context_alignment': 0.20,
            'verification_confidence': 0.15,
            'evidence_completeness': 0.10
        }
        
        # Coverage assessment parameters
        self.coverage_factors = {
            'claim_coverage_ratio': 0.40,
            'evidence_density': 0.25,
            'source_diversity': 0.20,
            'coverage_quality': 0.15
        }
        
        # Consistency assessment parameters
        self.consistency_factors = {
            'internal_consistency': 0.35,
            'cross_claim_consistency': 0.30,
            'temporal_consistency': 0.20,
            'logical_consistency': 0.15
        }
    
    def _initialize_grading_thresholds(self):
        """Initialize thresholds for accuracy grading."""
        
        self.grading_thresholds = {
            AccuracyGrade.EXCELLENT: 90.0,
            AccuracyGrade.GOOD: 80.0,
            AccuracyGrade.ACCEPTABLE: 70.0,
            AccuracyGrade.MARGINAL: 60.0,
            AccuracyGrade.POOR: 0.0
        }
    
    def _initialize_integration_mappings(self):
        """Initialize mappings for quality system integration."""
        
        # Mapping between accuracy dimensions and relevance scorer dimensions
        self.dimension_mappings = {
            'claim_verification': 'scientific_rigor',
            'evidence_quality': 'biomedical_context_depth',
            'coverage_assessment': 'query_alignment',
            'consistency_analysis': 'metabolomics_relevance',
            'confidence_factor': 'clinical_applicability'
        }
        
        # Quality system compatibility parameters
        self.integration_parameters = {
            'score_normalization_factor': 1.0,
            'confidence_adjustment_factor': 0.9,
            'quality_boost_threshold': 85.0,
            'integration_weight': 0.15
        }
    
    @performance_logged("Score factual accuracy")
    async def score_accuracy(self,
                           verification_results: List['VerificationResult'],
                           claims: Optional[List['ExtractedClaim']] = None,
                           context: Optional[Dict[str, Any]] = None) -> AccuracyScore:
        """
        Calculate comprehensive factual accuracy score from verification results.
        
        Args:
            verification_results: List of VerificationResult objects from validator
            claims: Optional list of original ExtractedClaim objects
            context: Optional context information for scoring
            
        Returns:
            AccuracyScore with comprehensive accuracy assessment
            
        Raises:
            AccuracyScoringError: If scoring process fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting accuracy scoring for {len(verification_results)} verification results")
            
            if not verification_results:
                return AccuracyScore(
                    overall_score=0.0,
                    grade=AccuracyGrade.POOR,
                    total_claims_assessed=0,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Calculate dimension scores
            dimension_scores = await self._calculate_dimension_scores(
                verification_results, claims, context
            )
            
            # Calculate claim type scores
            claim_type_scores = await self._calculate_claim_type_scores(
                verification_results, claims
            )
            
            # Calculate overall score
            overall_score = await self._calculate_overall_score(
                dimension_scores, claim_type_scores, verification_results
            )
            
            # Determine accuracy grade
            grade = self._determine_accuracy_grade(overall_score)
            
            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(
                verification_results, dimension_scores
            )
            
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.scoring_stats['total_scorings'] += 1
            self.scoring_stats['total_claims_scored'] += len(verification_results)
            
            # Create accuracy score
            accuracy_score = AccuracyScore(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                claim_type_scores=claim_type_scores,
                evidence_quality_score=dimension_scores.get('evidence_quality', 0.0),
                coverage_score=dimension_scores.get('coverage_assessment', 0.0),
                consistency_score=dimension_scores.get('consistency_analysis', 0.0),
                confidence_score=confidence_score,
                grade=grade,
                total_claims_assessed=len(verification_results),
                processing_time_ms=processing_time,
                metadata={
                    'scoring_method': 'comprehensive_weighted',
                    'config_version': '1.0.0',
                    'has_claims_context': claims is not None,
                    'has_additional_context': context is not None
                }
            )
            
            self.logger.info(
                f"Accuracy scoring completed: {overall_score:.1f}/100 ({grade.value}) "
                f"in {processing_time:.2f}ms"
            )
            
            return accuracy_score
            
        except Exception as e:
            self.logger.error(f"Error in accuracy scoring: {str(e)}")
            raise AccuracyScoringError(f"Failed to score accuracy: {str(e)}") from e
    
    async def _calculate_dimension_scores(self,
                                        verification_results: List['VerificationResult'],
                                        claims: Optional[List['ExtractedClaim']] = None,
                                        context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate scores for each accuracy dimension."""
        
        try:
            # Calculate claim verification score
            claim_verification_score = await self._calculate_claim_verification_score(
                verification_results
            )
            
            # Calculate evidence quality score
            evidence_quality_score = await self._calculate_evidence_quality_score(
                verification_results
            )
            
            # Calculate coverage assessment score
            coverage_score = await self._calculate_coverage_score(
                verification_results, claims
            )
            
            # Calculate consistency analysis score
            consistency_score = await self._calculate_consistency_score(
                verification_results, claims
            )
            
            # Calculate confidence factor
            confidence_factor = await self._calculate_confidence_factor(
                verification_results
            )
            
            return {
                'claim_verification': claim_verification_score,
                'evidence_quality': evidence_quality_score,
                'coverage_assessment': coverage_score,
                'consistency_analysis': consistency_score,
                'confidence_factor': confidence_factor
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating dimension scores: {str(e)}")
            # Return default scores on error
            return {
                'claim_verification': 0.0,
                'evidence_quality': 0.0,
                'coverage_assessment': 0.0,
                'consistency_analysis': 0.0,
                'confidence_factor': 0.0
            }
    
    async def _calculate_claim_verification_score(self,
                                                verification_results: List['VerificationResult']) -> float:
        """Calculate claim verification dimension score."""
        
        if not verification_results:
            return 0.0
        
        # Calculate verification status distribution
        status_scores = {
            VerificationStatus.SUPPORTED: 100.0,
            VerificationStatus.NEUTRAL: 60.0,
            VerificationStatus.NOT_FOUND: 40.0,
            VerificationStatus.CONTRADICTED: 0.0,
            VerificationStatus.ERROR: 0.0
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in verification_results:
            # Get base score from verification status
            base_score = status_scores.get(result.verification_status, 0.0)
            
            # Weight by verification confidence
            weight = result.verification_confidence / 100.0
            
            # Adjust score by evidence strength
            evidence_adjustment = result.evidence_strength / 100.0
            adjusted_score = base_score * (0.7 + 0.3 * evidence_adjustment)
            
            total_score += adjusted_score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 1.0)
    
    async def _calculate_evidence_quality_score(self,
                                              verification_results: List['VerificationResult']) -> float:
        """Calculate evidence quality dimension score."""
        
        if not verification_results:
            return 0.0
        
        quality_scores = []
        
        for result in verification_results:
            # Calculate evidence quality based on multiple factors
            evidence_count = result.total_evidence_count
            evidence_strength = result.evidence_strength
            context_match = result.context_match
            
            # Base score from evidence strength
            base_quality = evidence_strength
            
            # Bonus for multiple evidence items
            evidence_bonus = min(20.0, evidence_count * 5.0)
            
            # Context alignment bonus
            context_bonus = context_match * 0.15
            
            # Supporting vs contradicting evidence ratio
            supporting_count = len(result.supporting_evidence)
            contradicting_count = len(result.contradicting_evidence)
            
            if supporting_count + contradicting_count > 0:
                support_ratio = supporting_count / (supporting_count + contradicting_count)
                ratio_bonus = support_ratio * 10.0
            else:
                ratio_bonus = 0.0
            
            # Calculate final quality score
            quality_score = min(100.0, base_quality + evidence_bonus + context_bonus + ratio_bonus)
            quality_scores.append(quality_score)
        
        return statistics.mean(quality_scores) if quality_scores else 0.0
    
    async def _calculate_coverage_score(self,
                                      verification_results: List['VerificationResult'],
                                      claims: Optional[List['ExtractedClaim']] = None) -> float:
        """Calculate coverage assessment dimension score."""
        
        if not verification_results:
            return 0.0
        
        # Calculate claim coverage (claims with any evidence)
        claims_with_evidence = sum(
            1 for result in verification_results 
            if result.total_evidence_count > 0
        )
        
        claim_coverage_ratio = claims_with_evidence / len(verification_results)
        
        # Calculate evidence density (average evidence per claim)
        total_evidence = sum(result.total_evidence_count for result in verification_results)
        evidence_density = total_evidence / len(verification_results)
        
        # Normalize evidence density (assume 3 pieces of evidence per claim is optimal)
        normalized_density = min(1.0, evidence_density / 3.0)
        
        # Calculate source diversity
        all_sources = set()
        for result in verification_results:
            for evidence in (result.supporting_evidence + 
                           result.contradicting_evidence + 
                           result.neutral_evidence):
                all_sources.add(evidence.source_document)
        
        source_diversity = min(1.0, len(all_sources) / max(1, len(verification_results)))
        
        # Calculate coverage quality (average context match)
        avg_context_match = statistics.mean(
            [result.context_match for result in verification_results]
        ) / 100.0
        
        # Weighted coverage score
        coverage_score = (
            claim_coverage_ratio * self.coverage_factors['claim_coverage_ratio'] +
            normalized_density * self.coverage_factors['evidence_density'] +
            source_diversity * self.coverage_factors['source_diversity'] +
            avg_context_match * self.coverage_factors['coverage_quality']
        ) * 100.0
        
        return min(100.0, max(0.0, coverage_score))
    
    async def _calculate_consistency_score(self,
                                         verification_results: List['VerificationResult'],
                                         claims: Optional[List['ExtractedClaim']] = None) -> float:
        """Calculate consistency analysis dimension score."""
        
        if len(verification_results) < 2:
            return 75.0  # Neutral score for single claims
        
        # Internal consistency (individual claim coherence)
        internal_scores = []
        for result in verification_results:
            # Consistency between verification status and evidence
            supporting_count = len(result.supporting_evidence)
            contradicting_count = len(result.contradicting_evidence)
            
            if result.verification_status == VerificationStatus.SUPPORTED:
                if supporting_count > contradicting_count:
                    internal_scores.append(100.0)
                elif supporting_count == contradicting_count:
                    internal_scores.append(60.0)
                else:
                    internal_scores.append(20.0)
            elif result.verification_status == VerificationStatus.CONTRADICTED:
                if contradicting_count > supporting_count:
                    internal_scores.append(100.0)
                elif contradicting_count == supporting_count:
                    internal_scores.append(60.0)
                else:
                    internal_scores.append(20.0)
            else:
                internal_scores.append(80.0)  # Neutral cases are consistent
        
        internal_consistency = statistics.mean(internal_scores) if internal_scores else 50.0
        
        # Cross-claim consistency (claims don't contradict each other)
        supported_claims = [r for r in verification_results if r.verification_status == VerificationStatus.SUPPORTED]
        contradicted_claims = [r for r in verification_results if r.verification_status == VerificationStatus.CONTRADICTED]
        
        # Simple heuristic: high contradiction rate suggests inconsistency
        total_claims = len(verification_results)
        contradiction_rate = len(contradicted_claims) / total_claims
        cross_consistency = max(0.0, 100.0 - (contradiction_rate * 100.0))
        
        # Temporal consistency (time-based claims are logically consistent)
        temporal_consistency = 85.0  # Default for now - can be enhanced
        
        # Logical consistency (no direct contradictions)
        logical_consistency = 90.0  # Default for now - can be enhanced
        
        # Weighted consistency score
        consistency_score = (
            internal_consistency * self.consistency_factors['internal_consistency'] +
            cross_consistency * self.consistency_factors['cross_claim_consistency'] +
            temporal_consistency * self.consistency_factors['temporal_consistency'] +
            logical_consistency * self.consistency_factors['logical_consistency']
        )
        
        return min(100.0, max(0.0, consistency_score))
    
    async def _calculate_confidence_factor(self,
                                         verification_results: List['VerificationResult']) -> float:
        """Calculate confidence factor dimension score."""
        
        if not verification_results:
            return 0.0
        
        # Average verification confidence
        avg_confidence = statistics.mean([r.verification_confidence for r in verification_results])
        
        # Processing quality (low processing times suggest efficient verification)
        avg_processing_time = statistics.mean([r.processing_time_ms for r in verification_results])
        processing_quality = max(0.0, 100.0 - (avg_processing_time / 100.0))  # Normalize to 100
        
        # Error rate (claims with ERROR status)
        error_count = sum(1 for r in verification_results if r.verification_status == VerificationStatus.ERROR)
        error_rate = error_count / len(verification_results)
        error_penalty = error_rate * 50.0
        
        # Coverage completeness (claims with evidence found)
        coverage_completeness = sum(
            1 for r in verification_results 
            if r.verification_status != VerificationStatus.NOT_FOUND
        ) / len(verification_results) * 100.0
        
        # Weighted confidence factor
        confidence_factor = (
            avg_confidence * 0.4 +
            min(100.0, processing_quality) * 0.2 +
            coverage_completeness * 0.3 +
            max(0.0, 100.0 - error_penalty) * 0.1
        )
        
        return min(100.0, max(0.0, confidence_factor))
    
    async def _calculate_claim_type_scores(self,
                                         verification_results: List['VerificationResult'],
                                         claims: Optional[List['ExtractedClaim']] = None) -> Dict[str, float]:
        """Calculate scores broken down by claim type."""
        
        # Group verification results by claim type (from metadata)
        type_groups = defaultdict(list)
        
        for result in verification_results:
            claim_type = result.metadata.get('claim_type', 'general')
            type_groups[claim_type].append(result)
        
        type_scores = {}
        
        for claim_type, results in type_groups.items():
            # Calculate type-specific score
            type_score = await self._calculate_type_specific_score(results, claim_type)
            
            # Apply claim type weight
            weight = self.claim_type_weights.get(claim_type, 1.0)
            weighted_score = type_score * weight
            
            type_scores[claim_type] = min(100.0, max(0.0, weighted_score))
        
        return type_scores
    
    async def _calculate_type_specific_score(self,
                                           results: List['VerificationResult'],
                                           claim_type: str) -> float:
        """Calculate score specific to a claim type."""
        
        if not results:
            return 0.0
        
        # Type-specific scoring strategies
        if claim_type == 'numeric':
            return await self._score_numeric_claims(results)
        elif claim_type == 'qualitative':
            return await self._score_qualitative_claims(results)
        elif claim_type == 'methodological':
            return await self._score_methodological_claims(results)
        elif claim_type == 'temporal':
            return await self._score_temporal_claims(results)
        elif claim_type == 'comparative':
            return await self._score_comparative_claims(results)
        else:
            return await self._score_general_claims(results)
    
    async def _score_numeric_claims(self, results: List['VerificationResult']) -> float:
        """Score numeric claims with emphasis on precision and evidence strength."""
        
        scores = []
        
        for result in results:
            # Base score from verification status
            if result.verification_status == VerificationStatus.SUPPORTED:
                base_score = 90.0
            elif result.verification_status == VerificationStatus.NEUTRAL:
                base_score = 60.0
            elif result.verification_status == VerificationStatus.NOT_FOUND:
                base_score = 30.0
            else:
                base_score = 0.0
            
            # Bonus for high evidence strength (important for numeric claims)
            evidence_bonus = result.evidence_strength * 0.15
            
            # Bonus for multiple supporting evidence
            support_bonus = min(15.0, len(result.supporting_evidence) * 5.0)
            
            # Context match bonus (precision in numeric context)
            context_bonus = result.context_match * 0.1
            
            total_score = min(100.0, base_score + evidence_bonus + support_bonus + context_bonus)
            scores.append(total_score)
        
        return statistics.mean(scores)
    
    async def _score_qualitative_claims(self, results: List['VerificationResult']) -> float:
        """Score qualitative claims with emphasis on context and relationships."""
        
        scores = []
        
        for result in results:
            # Base score from verification status
            if result.verification_status == VerificationStatus.SUPPORTED:
                base_score = 85.0
            elif result.verification_status == VerificationStatus.NEUTRAL:
                base_score = 70.0
            elif result.verification_status == VerificationStatus.NOT_FOUND:
                base_score = 40.0
            else:
                base_score = 0.0
            
            # Context match is crucial for qualitative claims
            context_bonus = result.context_match * 0.2
            
            # Evidence diversity bonus
            evidence_diversity = len(set([e.evidence_type for e in 
                                        result.supporting_evidence + result.contradicting_evidence]))
            diversity_bonus = min(10.0, evidence_diversity * 3.0)
            
            total_score = min(100.0, base_score + context_bonus + diversity_bonus)
            scores.append(total_score)
        
        return statistics.mean(scores)
    
    async def _score_methodological_claims(self, results: List['VerificationResult']) -> float:
        """Score methodological claims with emphasis on technical accuracy."""
        
        scores = []
        
        for result in results:
            # Base score from verification status
            if result.verification_status == VerificationStatus.SUPPORTED:
                base_score = 95.0  # Higher base for methodological accuracy
            elif result.verification_status == VerificationStatus.NEUTRAL:
                base_score = 65.0
            elif result.verification_status == VerificationStatus.NOT_FOUND:
                base_score = 35.0
            else:
                base_score = 0.0
            
            # Technical precision bonus (high evidence strength)
            if result.evidence_strength >= 80.0:
                precision_bonus = 10.0
            elif result.evidence_strength >= 60.0:
                precision_bonus = 5.0
            else:
                precision_bonus = 0.0
            
            total_score = min(100.0, base_score + precision_bonus)
            scores.append(total_score)
        
        return statistics.mean(scores)
    
    async def _score_temporal_claims(self, results: List['VerificationResult']) -> float:
        """Score temporal claims with standard weighting."""
        
        return await self._score_general_claims(results)
    
    async def _score_comparative_claims(self, results: List['VerificationResult']) -> float:
        """Score comparative claims with emphasis on evidence strength."""
        
        scores = []
        
        for result in results:
            # Base score from verification status
            if result.verification_status == VerificationStatus.SUPPORTED:
                base_score = 88.0
            elif result.verification_status == VerificationStatus.NEUTRAL:
                base_score = 65.0
            elif result.verification_status == VerificationStatus.NOT_FOUND:
                base_score = 35.0
            else:
                base_score = 0.0
            
            # Evidence strength is crucial for comparative claims
            evidence_bonus = result.evidence_strength * 0.12
            
            total_score = min(100.0, base_score + evidence_bonus)
            scores.append(total_score)
        
        return statistics.mean(scores)
    
    async def _score_general_claims(self, results: List['VerificationResult']) -> float:
        """Score general claims with standard methodology."""
        
        scores = []
        
        for result in results:
            # Base score from verification status
            if result.verification_status == VerificationStatus.SUPPORTED:
                base_score = 80.0
            elif result.verification_status == VerificationStatus.NEUTRAL:
                base_score = 60.0
            elif result.verification_status == VerificationStatus.NOT_FOUND:
                base_score = 40.0
            else:
                base_score = 0.0
            
            # Standard bonuses
            evidence_bonus = result.evidence_strength * 0.1
            context_bonus = result.context_match * 0.08
            
            total_score = min(100.0, base_score + evidence_bonus + context_bonus)
            scores.append(total_score)
        
        return statistics.mean(scores)
    
    async def _calculate_overall_score(self,
                                     dimension_scores: Dict[str, float],
                                     claim_type_scores: Dict[str, float],
                                     verification_results: List['VerificationResult']) -> float:
        """Calculate weighted overall accuracy score."""
        
        # Calculate dimension-weighted score
        dimension_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.scoring_weights.get(dimension, 0.0)
            dimension_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            dimension_score = dimension_score / total_weight
        else:
            dimension_score = 0.0
        
        # Calculate claim type average (unweighted for balance)
        if claim_type_scores:
            type_score = statistics.mean(claim_type_scores.values())
        else:
            type_score = 0.0
        
        # Combine dimension and type scores
        overall_score = (dimension_score * 0.75) + (type_score * 0.25)
        
        # Apply minimum claims penalty
        min_claims = self.config['performance_targets']['min_claims_for_reliable_score']
        if len(verification_results) < min_claims:
            penalty_factor = len(verification_results) / min_claims
            overall_score *= penalty_factor
        
        return min(100.0, max(0.0, overall_score))
    
    def _determine_accuracy_grade(self, score: float) -> AccuracyGrade:
        """Determine accuracy grade from overall score."""
        
        for grade, threshold in self.grading_thresholds.items():
            if score >= threshold:
                return grade
        
        return AccuracyGrade.POOR
    
    async def _calculate_confidence_score(self,
                                        verification_results: List['VerificationResult'],
                                        dimension_scores: Dict[str, float]) -> float:
        """Calculate confidence in the accuracy assessment."""
        
        if not verification_results or not dimension_scores:
            return 0.0
        
        # Base confidence from verification results
        avg_verification_confidence = statistics.mean(
            [r.verification_confidence for r in verification_results]
        )
        
        # Consistency bonus (low variance in dimension scores)
        if len(dimension_scores) > 1:
            score_variance = statistics.variance(dimension_scores.values())
            consistency_factor = max(0.0, 1.0 - (score_variance / 1000.0))  # Normalize variance
        else:
            consistency_factor = 0.5
        
        # Evidence availability factor
        evidence_factor = min(1.0, sum(r.total_evidence_count for r in verification_results) / 
                             (len(verification_results) * 2))  # Assume 2 evidence items per claim is good
        
        # Processing quality factor (no errors, reasonable processing times)
        error_count = sum(1 for r in verification_results if r.verification_status == VerificationStatus.ERROR)
        error_factor = max(0.0, 1.0 - (error_count / len(verification_results)))
        
        # Combined confidence score
        confidence_score = (
            avg_verification_confidence * 0.4 +
            consistency_factor * 100 * 0.25 +
            evidence_factor * 100 * 0.25 +
            error_factor * 100 * 0.1
        )
        
        return min(100.0, max(0.0, confidence_score))
    
    @performance_logged("Generate comprehensive accuracy report")
    async def generate_comprehensive_report(self,
                                          verification_results: List['VerificationResult'],
                                          claims: Optional[List['ExtractedClaim']] = None,
                                          query: Optional[str] = None,
                                          response: Optional[str] = None,
                                          context: Optional[Dict[str, Any]] = None) -> AccuracyReport:
        """
        Generate comprehensive factual accuracy report.
        
        Args:
            verification_results: List of VerificationResult objects
            claims: Optional list of original ExtractedClaim objects
            query: Optional original query for context
            response: Optional original response for integration
            context: Optional additional context
            
        Returns:
            AccuracyReport with comprehensive analysis and recommendations
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating comprehensive accuracy report for {len(verification_results)} results")
            
            # Generate unique report ID
            report_id = self._generate_report_id(verification_results, claims)
            
            # Calculate accuracy score
            accuracy_score = await self.score_accuracy(verification_results, claims, context)
            
            # Generate detailed breakdown
            detailed_breakdown = await self._generate_detailed_breakdown(
                verification_results, claims, accuracy_score
            )
            
            # Calculate summary statistics
            summary_statistics = await self._generate_summary_statistics(
                verification_results, accuracy_score
            )
            
            # Generate performance metrics
            performance_metrics = await self._generate_performance_metrics(
                verification_results, start_time
            )
            
            # Generate quality recommendations
            recommendations = await self._generate_quality_recommendations(
                accuracy_score, verification_results, claims
            )
            
            # Generate integration data for quality systems
            integration_data = await self._generate_integration_data(
                accuracy_score, query, response, context
            )
            
            # Generate individual claim analysis
            claims_analysis = await self._generate_claims_analysis(verification_results, claims)
            
            # Generate evidence analysis
            evidence_analysis = await self._generate_evidence_analysis(verification_results)
            
            # Generate coverage analysis
            coverage_analysis = await self._generate_coverage_analysis(
                verification_results, claims
            )
            
            # Create comprehensive report
            report = AccuracyReport(
                report_id=report_id,
                accuracy_score=accuracy_score,
                detailed_breakdown=detailed_breakdown,
                summary_statistics=summary_statistics,
                performance_metrics=performance_metrics,
                quality_recommendations=recommendations,
                integration_data=integration_data,
                claims_analysis=claims_analysis,
                evidence_analysis=evidence_analysis,
                coverage_analysis=coverage_analysis,
                configuration_used=self.config
            )
            
            processing_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"Comprehensive report generated in {processing_time:.2f}ms: "
                f"{accuracy_score.accuracy_percentage} ({accuracy_score.grade.value})"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            raise ReportGenerationError(f"Failed to generate report: {str(e)}") from e
    
    def _generate_report_id(self,
                           verification_results: List['VerificationResult'],
                           claims: Optional[List['ExtractedClaim']] = None) -> str:
        """Generate unique report ID."""
        
        # Create hash from key components
        content_hash = hashlib.md5()
        content_hash.update(str(len(verification_results)).encode())
        content_hash.update(datetime.now().isoformat().encode())
        
        if claims:
            content_hash.update(str(len(claims)).encode())
        
        return f"FACR_{content_hash.hexdigest()[:12]}"
    
    async def _generate_detailed_breakdown(self,
                                         verification_results: List['VerificationResult'],
                                         claims: Optional[List['ExtractedClaim']],
                                         accuracy_score: AccuracyScore) -> Dict[str, Any]:
        """Generate detailed breakdown of accuracy assessment."""
        
        # Verification status distribution
        status_distribution = Counter(r.verification_status for r in verification_results)
        
        # Evidence statistics
        evidence_stats = {
            'total_evidence_items': sum(r.total_evidence_count for r in verification_results),
            'avg_evidence_per_claim': statistics.mean([r.total_evidence_count for r in verification_results]),
            'claims_with_supporting_evidence': sum(1 for r in verification_results if r.supporting_evidence),
            'claims_with_contradicting_evidence': sum(1 for r in verification_results if r.contradicting_evidence),
            'claims_with_neutral_evidence': sum(1 for r in verification_results if r.neutral_evidence)
        }
        
        # Confidence distribution
        confidence_scores = [r.verification_confidence for r in verification_results]
        confidence_distribution = {
            'mean': statistics.mean(confidence_scores),
            'median': statistics.median(confidence_scores),
            'std_dev': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
            'min': min(confidence_scores),
            'max': max(confidence_scores)
        }
        
        # Processing performance
        processing_times = [r.processing_time_ms for r in verification_results]
        processing_stats = {
            'total_processing_time_ms': sum(processing_times),
            'avg_processing_time_ms': statistics.mean(processing_times),
            'max_processing_time_ms': max(processing_times),
            'min_processing_time_ms': min(processing_times)
        }
        
        return {
            'status_distribution': {status.value: count for status, count in status_distribution.items()},
            'evidence_statistics': evidence_stats,
            'confidence_distribution': confidence_distribution,
            'processing_statistics': processing_stats,
            'dimension_breakdown': accuracy_score.dimension_scores,
            'claim_type_breakdown': accuracy_score.claim_type_scores
        }
    
    async def _generate_summary_statistics(self,
                                         verification_results: List['VerificationResult'],
                                         accuracy_score: AccuracyScore) -> Dict[str, Any]:
        """Generate summary statistics for the accuracy assessment."""
        
        # Basic counts
        total_claims = len(verification_results)
        verified_claims = sum(1 for r in verification_results 
                            if r.verification_status in [VerificationStatus.SUPPORTED, VerificationStatus.CONTRADICTED])
        
        # Success metrics
        supported_claims = sum(1 for r in verification_results if r.verification_status == VerificationStatus.SUPPORTED)
        contradicted_claims = sum(1 for r in verification_results if r.verification_status == VerificationStatus.CONTRADICTED)
        
        # Quality metrics
        high_confidence_claims = sum(1 for r in verification_results if r.verification_confidence >= 80)
        high_evidence_claims = sum(1 for r in verification_results if r.evidence_strength >= 70)
        
        return {
            'total_claims': total_claims,
            'verified_claims': verified_claims,
            'verification_rate': verified_claims / total_claims if total_claims > 0 else 0,
            'support_rate': supported_claims / total_claims if total_claims > 0 else 0,
            'contradiction_rate': contradicted_claims / total_claims if total_claims > 0 else 0,
            'high_confidence_rate': high_confidence_claims / total_claims if total_claims > 0 else 0,
            'high_evidence_rate': high_evidence_claims / total_claims if total_claims > 0 else 0,
            'overall_accuracy_score': accuracy_score.overall_score,
            'accuracy_grade': accuracy_score.grade.value,
            'reliability_indicator': accuracy_score.is_reliable
        }
    
    async def _generate_performance_metrics(self,
                                          verification_results: List['VerificationResult'],
                                          start_time: float) -> AccuracyMetrics:
        """Generate performance metrics for the accuracy assessment."""
        
        current_time = time.time()
        total_processing_time = (current_time - start_time) * 1000
        
        # Verification performance
        verification_times = [r.processing_time_ms for r in verification_results]
        verification_performance = {
            'total_verification_time_ms': sum(verification_times),
            'avg_verification_time_ms': statistics.mean(verification_times),
            'verification_throughput': len(verification_results) / (sum(verification_times) / 1000) if sum(verification_times) > 0 else 0
        }
        
        # Scoring performance
        scoring_performance = {
            'total_scoring_time_ms': total_processing_time,
            'scoring_throughput': len(verification_results) / (total_processing_time / 1000) if total_processing_time > 0 else 0
        }
        
        # Quality indicators
        error_count = sum(1 for r in verification_results if r.verification_status == VerificationStatus.ERROR)
        quality_indicators = {
            'error_rate': error_count / len(verification_results) if verification_results else 0,
            'avg_confidence': statistics.mean([r.verification_confidence for r in verification_results]) if verification_results else 0,
            'evidence_coverage_rate': sum(1 for r in verification_results if r.total_evidence_count > 0) / len(verification_results) if verification_results else 0
        }
        
        # System health
        system_health = {
            'memory_efficient': total_processing_time < self.config['performance_targets']['max_processing_time_ms'],
            'error_rate_acceptable': quality_indicators['error_rate'] <= self.config['performance_targets']['max_error_rate'],
            'sufficient_claims': len(verification_results) >= self.config['performance_targets']['min_claims_for_reliable_score']
        }
        
        return AccuracyMetrics(
            verification_performance=verification_performance,
            scoring_performance=scoring_performance,
            quality_indicators=quality_indicators,
            system_health=system_health
        )
    
    async def _generate_quality_recommendations(self,
                                              accuracy_score: AccuracyScore,
                                              verification_results: List['VerificationResult'],
                                              claims: Optional[List['ExtractedClaim']]) -> List[str]:
        """Generate quality improvement recommendations."""
        
        recommendations = []
        
        # Overall accuracy recommendations
        if accuracy_score.overall_score < 60:
            recommendations.append("Overall accuracy is low - review claim extraction and verification processes")
        elif accuracy_score.overall_score < 80:
            recommendations.append("Accuracy is acceptable but could be improved with better evidence sourcing")
        
        # Evidence quality recommendations
        if accuracy_score.evidence_quality_score < 70:
            recommendations.append("Evidence quality is low - expand document index and improve search strategies")
        
        # Coverage recommendations
        if accuracy_score.coverage_score < 60:
            recommendations.append("Poor claim coverage - consider adding more diverse source documents")
        elif accuracy_score.coverage_score < 80:
            recommendations.append("Coverage can be improved with additional authoritative sources")
        
        # Consistency recommendations
        if accuracy_score.consistency_score < 70:
            recommendations.append("Consistency issues detected - review for contradictory claims and evidence")
        
        # Claim type specific recommendations
        for claim_type, score in accuracy_score.claim_type_scores.items():
            if score < 60:
                recommendations.append(f"Low accuracy for {claim_type} claims - improve verification methods for this type")
        
        # Confidence recommendations
        if accuracy_score.confidence_score < 70:
            recommendations.append("Low confidence in assessment - increase evidence requirements and verification rigor")
        
        # Processing performance recommendations
        if accuracy_score.total_claims_assessed < 5:
            recommendations.append("Few claims assessed - results may not be reliable without more claims")
        
        # Evidence distribution recommendations
        not_found_count = sum(1 for r in verification_results if r.verification_status == VerificationStatus.NOT_FOUND)
        if not_found_count > len(verification_results) * 0.3:
            recommendations.append("Many claims lack evidence - expand document collection and indexing")
        
        # Error rate recommendations
        error_count = sum(1 for r in verification_results if r.verification_status == VerificationStatus.ERROR)
        if error_count > 0:
            recommendations.append("Processing errors detected - review system configuration and error handling")
        
        # Default recommendation if all looks good
        if not recommendations:
            recommendations.append("Accuracy assessment is performing well - maintain current standards")
        
        return recommendations
    
    async def _generate_integration_data(self,
                                       accuracy_score: AccuracyScore,
                                       query: Optional[str] = None,
                                       response: Optional[str] = None,
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate data for integration with existing quality systems."""
        
        integration_data = {
            'factual_accuracy_score': accuracy_score.overall_score,
            'accuracy_grade': accuracy_score.grade.value,
            'reliability_indicator': accuracy_score.is_reliable,
            'dimension_scores': accuracy_score.dimension_scores,
            'integration_weights': self.integration_parameters,
            'quality_boost_eligible': accuracy_score.overall_score >= self.integration_parameters['quality_boost_threshold']
        }
        
        # Generate data compatible with ClinicalMetabolomicsRelevanceScorer
        if self.config['integration_settings']['enable_relevance_integration']:
            relevance_compatible_scores = {}
            
            for accuracy_dim, relevance_dim in self.dimension_mappings.items():
                if accuracy_dim in accuracy_score.dimension_scores:
                    # Convert accuracy score to relevance score format with adjustment
                    adjusted_score = accuracy_score.dimension_scores[accuracy_dim] * self.integration_parameters['confidence_adjustment_factor']
                    relevance_compatible_scores[relevance_dim] = adjusted_score
            
            integration_data['relevance_scorer_compatibility'] = {
                'dimension_scores': relevance_compatible_scores,
                'overall_adjustment_factor': self.integration_parameters['score_normalization_factor'],
                'integration_weight': self.integration_parameters['integration_weight']
            }
        
        # Add contextual information for quality assessment
        if query and response:
            integration_data['contextual_assessment'] = {
                'query_provided': True,
                'response_provided': True,
                'query_length': len(query),
                'response_length': len(response),
                'assessment_scope': 'full_context'
            }
        
        # Performance integration data
        integration_data['performance_indicators'] = {
            'processing_time_ms': accuracy_score.processing_time_ms,
            'claims_assessed': accuracy_score.total_claims_assessed,
            'confidence_score': accuracy_score.confidence_score,
            'metadata': accuracy_score.metadata
        }
        
        return integration_data
    
    async def _generate_claims_analysis(self,
                                      verification_results: List['VerificationResult'],
                                      claims: Optional[List['ExtractedClaim']]) -> List[Dict[str, Any]]:
        """Generate detailed analysis of individual claims."""
        
        claims_analysis = []
        
        for result in verification_results:
            claim_analysis = {
                'claim_id': result.claim_id,
                'verification_status': result.verification_status.value,
                'verification_confidence': result.verification_confidence,
                'evidence_strength': result.evidence_strength,
                'context_match': result.context_match,
                'processing_time_ms': result.processing_time_ms,
                'evidence_summary': {
                    'supporting_count': len(result.supporting_evidence),
                    'contradicting_count': len(result.contradicting_evidence),
                    'neutral_count': len(result.neutral_evidence),
                    'total_evidence': result.total_evidence_count
                },
                'verification_strategy': result.verification_strategy,
                'confidence_grade': result.verification_grade,
                'error_details': result.error_details
            }
            
            # Add evidence details
            if result.supporting_evidence:
                claim_analysis['supporting_evidence'] = [
                    {
                        'source': evidence.source_document,
                        'text': evidence.evidence_text[:100] + '...' if len(evidence.evidence_text) > 100 else evidence.evidence_text,
                        'confidence': evidence.confidence,
                        'type': evidence.evidence_type
                    }
                    for evidence in result.supporting_evidence[:3]  # Top 3 evidence items
                ]
            
            if result.contradicting_evidence:
                claim_analysis['contradicting_evidence'] = [
                    {
                        'source': evidence.source_document,
                        'text': evidence.evidence_text[:100] + '...' if len(evidence.evidence_text) > 100 else evidence.evidence_text,
                        'confidence': evidence.confidence,
                        'type': evidence.evidence_type
                    }
                    for evidence in result.contradicting_evidence[:3]  # Top 3 evidence items
                ]
            
            claims_analysis.append(claim_analysis)
        
        return claims_analysis
    
    async def _generate_evidence_analysis(self,
                                        verification_results: List['VerificationResult']) -> Dict[str, Any]:
        """Generate comprehensive evidence analysis."""
        
        # Collect all evidence
        all_evidence = []
        for result in verification_results:
            all_evidence.extend(result.supporting_evidence)
            all_evidence.extend(result.contradicting_evidence)
            all_evidence.extend(result.neutral_evidence)
        
        if not all_evidence:
            return {
                'total_evidence_items': 0,
                'message': 'No evidence found for analysis'
            }
        
        # Source analysis
        sources = defaultdict(int)
        evidence_types = defaultdict(int)
        confidence_scores = []
        
        for evidence in all_evidence:
            sources[evidence.source_document] += 1
            evidence_types[evidence.evidence_type] += 1
            confidence_scores.append(evidence.confidence)
        
        # Quality analysis
        high_quality_evidence = sum(1 for e in all_evidence if e.confidence >= 80)
        medium_quality_evidence = sum(1 for e in all_evidence if 60 <= e.confidence < 80)
        low_quality_evidence = sum(1 for e in all_evidence if e.confidence < 60)
        
        return {
            'total_evidence_items': len(all_evidence),
            'unique_sources': len(sources),
            'source_distribution': dict(sources),
            'evidence_type_distribution': dict(evidence_types),
            'quality_distribution': {
                'high_quality': high_quality_evidence,
                'medium_quality': medium_quality_evidence,
                'low_quality': low_quality_evidence
            },
            'confidence_statistics': {
                'mean': statistics.mean(confidence_scores),
                'median': statistics.median(confidence_scores),
                'std_dev': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
                'min': min(confidence_scores),
                'max': max(confidence_scores)
            },
            'average_evidence_per_claim': len(all_evidence) / len(verification_results),
            'top_sources': sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    async def _generate_coverage_analysis(self,
                                        verification_results: List['VerificationResult'],
                                        claims: Optional[List['ExtractedClaim']]) -> Dict[str, Any]:
        """Generate coverage analysis of claims vs source documents."""
        
        # Claims coverage
        claims_with_evidence = sum(1 for r in verification_results if r.total_evidence_count > 0)
        claims_with_support = sum(1 for r in verification_results if r.supporting_evidence)
        claims_without_evidence = sum(1 for r in verification_results if r.verification_status == VerificationStatus.NOT_FOUND)
        
        # Evidence coverage by claim type
        coverage_by_type = {}
        type_groups = defaultdict(list)
        
        for result in verification_results:
            claim_type = result.metadata.get('claim_type', 'general')
            type_groups[claim_type].append(result)
        
        for claim_type, results in type_groups.items():
            type_coverage = sum(1 for r in results if r.total_evidence_count > 0) / len(results)
            coverage_by_type[claim_type] = type_coverage
        
        # Source utilization
        all_sources = set()
        for result in verification_results:
            for evidence in (result.supporting_evidence + result.contradicting_evidence + result.neutral_evidence):
                all_sources.add(evidence.source_document)
        
        coverage_analysis = {
            'total_claims': len(verification_results),
            'claims_with_evidence': claims_with_evidence,
            'claims_with_support': claims_with_support,
            'claims_without_evidence': claims_without_evidence,
            'overall_coverage_rate': claims_with_evidence / len(verification_results) if verification_results else 0,
            'support_coverage_rate': claims_with_support / len(verification_results) if verification_results else 0,
            'coverage_by_claim_type': coverage_by_type,
            'sources_utilized': len(all_sources),
            'coverage_quality': {
                'excellent': sum(1 for r in verification_results if r.total_evidence_count >= 3),
                'good': sum(1 for r in verification_results if r.total_evidence_count == 2),
                'minimal': sum(1 for r in verification_results if r.total_evidence_count == 1),
                'none': claims_without_evidence
            }
        }
        
        # Coverage recommendations
        recommendations = []
        if coverage_analysis['overall_coverage_rate'] < 0.6:
            recommendations.append("Low overall coverage - expand document collection")
        if coverage_analysis['support_coverage_rate'] < 0.4:
            recommendations.append("Few claims have supporting evidence - review claim extraction accuracy")
        if len(all_sources) < 3:
            recommendations.append("Limited source diversity - add more authoritative documents")
        
        coverage_analysis['recommendations'] = recommendations
        
        return coverage_analysis
    
    async def integrate_with_relevance_scorer(self,
                                            accuracy_score: AccuracyScore,
                                            query: str,
                                            response: str,
                                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Integrate accuracy scoring with ClinicalMetabolomicsRelevanceScorer.
        
        Args:
            accuracy_score: Factual accuracy score to integrate
            query: Original query
            response: Original response  
            context: Optional additional context
            
        Returns:
            Dictionary with integrated quality assessment
            
        Raises:
            QualityIntegrationError: If integration fails
        """
        try:
            self.logger.info("Integrating accuracy score with relevance scorer")
            
            if not self.relevance_scorer:
                if self.config['integration_settings']['enable_relevance_integration']:
                    # Create relevance scorer instance
                    from .relevance_scorer import ClinicalMetabolomicsRelevanceScorer
                    self.relevance_scorer = ClinicalMetabolomicsRelevanceScorer()
                else:
                    raise QualityIntegrationError("Relevance scorer integration disabled")
            
            # Calculate relevance score
            relevance_score = await self.relevance_scorer.calculate_relevance_score(
                query, response, context
            )
            
            # Create integrated assessment
            integrated_assessment = {
                'factual_accuracy': {
                    'overall_score': accuracy_score.overall_score,
                    'grade': accuracy_score.grade.value,
                    'dimension_scores': accuracy_score.dimension_scores,
                    'claim_type_scores': accuracy_score.claim_type_scores,
                    'confidence': accuracy_score.confidence_score
                },
                'relevance_assessment': {
                    'overall_score': relevance_score.overall_score,
                    'grade': relevance_score.relevance_grade,
                    'dimension_scores': relevance_score.dimension_scores,
                    'query_type': relevance_score.query_type,
                    'confidence': relevance_score.confidence_score
                },
                'integrated_quality': {
                    'combined_score': self._calculate_combined_quality_score(
                        accuracy_score, relevance_score
                    ),
                    'quality_grade': None,  # Will be set below
                    'strength_areas': [],
                    'improvement_areas': [],
                    'overall_assessment': None  # Will be set below
                }
            }
            
            # Calculate combined quality metrics
            combined_score = integrated_assessment['integrated_quality']['combined_score']
            integrated_assessment['integrated_quality']['quality_grade'] = self._get_combined_quality_grade(combined_score)
            
            # Identify strengths and areas for improvement
            strengths, improvements = self._analyze_quality_dimensions(accuracy_score, relevance_score)
            integrated_assessment['integrated_quality']['strength_areas'] = strengths
            integrated_assessment['integrated_quality']['improvement_areas'] = improvements
            
            # Overall assessment summary
            integrated_assessment['integrated_quality']['overall_assessment'] = self._generate_overall_assessment(
                accuracy_score, relevance_score, combined_score
            )
            
            # Integration metadata
            integrated_assessment['integration_metadata'] = {
                'integration_timestamp': datetime.now().isoformat(),
                'accuracy_weight': self.integration_parameters['integration_weight'],
                'relevance_weight': 1.0 - self.integration_parameters['integration_weight'],
                'normalization_applied': True,
                'confidence_adjustment_applied': True
            }
            
            self.logger.info(
                f"Quality integration completed: Combined score {combined_score:.1f}/100"
            )
            
            return integrated_assessment
            
        except Exception as e:
            self.logger.error(f"Error integrating with relevance scorer: {str(e)}")
            raise QualityIntegrationError(f"Failed to integrate quality assessments: {str(e)}") from e
    
    def _calculate_combined_quality_score(self,
                                        accuracy_score: AccuracyScore,
                                        relevance_score: 'RelevanceScore') -> float:
        """Calculate combined quality score from accuracy and relevance."""
        
        # Weight configuration
        accuracy_weight = self.integration_parameters['integration_weight']
        relevance_weight = 1.0 - accuracy_weight
        
        # Normalize scores if needed
        normalized_accuracy = accuracy_score.overall_score * self.integration_parameters['score_normalization_factor']
        normalized_relevance = relevance_score.overall_score * self.integration_parameters['score_normalization_factor']
        
        # Apply confidence adjustments
        accuracy_confidence_adj = accuracy_score.confidence_score / 100.0 * self.integration_parameters['confidence_adjustment_factor']
        relevance_confidence_adj = relevance_score.confidence_score / 100.0 * self.integration_parameters['confidence_adjustment_factor']
        
        # Calculate weighted combination
        combined_score = (
            normalized_accuracy * accuracy_weight * (0.8 + 0.2 * accuracy_confidence_adj) +
            normalized_relevance * relevance_weight * (0.8 + 0.2 * relevance_confidence_adj)
        )
        
        # Quality boost for high-performing systems
        if (normalized_accuracy >= self.integration_parameters['quality_boost_threshold'] and
            normalized_relevance >= self.integration_parameters['quality_boost_threshold']):
            combined_score *= 1.05  # 5% boost for dual high performance
        
        return min(100.0, max(0.0, combined_score))
    
    def _get_combined_quality_grade(self, combined_score: float) -> str:
        """Get quality grade for combined score."""
        
        if combined_score >= 90:
            return "Excellent"
        elif combined_score >= 80:
            return "Good"
        elif combined_score >= 70:
            return "Acceptable"
        elif combined_score >= 60:
            return "Marginal"
        else:
            return "Poor"
    
    def _analyze_quality_dimensions(self,
                                  accuracy_score: AccuracyScore,
                                  relevance_score: 'RelevanceScore') -> Tuple[List[str], List[str]]:
        """Analyze dimensions to identify strengths and improvement areas."""
        
        strengths = []
        improvements = []
        
        # Analyze accuracy dimensions
        for dimension, score in accuracy_score.dimension_scores.items():
            if score >= 85:
                strengths.append(f"Excellent {dimension.replace('_', ' ')}")
            elif score < 60:
                improvements.append(f"Improve {dimension.replace('_', ' ')}")
        
        # Analyze relevance dimensions
        for dimension, score in relevance_score.dimension_scores.items():
            if score >= 85:
                strengths.append(f"Excellent {dimension.replace('_', ' ')}")
            elif score < 60:
                improvements.append(f"Improve {dimension.replace('_', ' ')}")
        
        # Analyze claim type performance
        for claim_type, score in accuracy_score.claim_type_scores.items():
            if score >= 85:
                strengths.append(f"Strong {claim_type} claim accuracy")
            elif score < 60:
                improvements.append(f"Improve {claim_type} claim verification")
        
        # Overall performance analysis
        if accuracy_score.overall_score >= 85 and relevance_score.overall_score >= 85:
            strengths.append("Excellent overall quality performance")
        elif accuracy_score.overall_score < 60 or relevance_score.overall_score < 60:
            improvements.append("Overall quality needs significant improvement")
        
        return strengths[:5], improvements[:5]  # Limit to top 5 each
    
    def _generate_overall_assessment(self,
                                   accuracy_score: AccuracyScore,
                                   relevance_score: 'RelevanceScore',
                                   combined_score: float) -> str:
        """Generate overall quality assessment summary."""
        
        assessment_parts = []
        
        # Combined performance assessment
        if combined_score >= 90:
            assessment_parts.append("Excellent overall quality with strong factual accuracy and relevance.")
        elif combined_score >= 80:
            assessment_parts.append("Good quality performance with solid factual and relevance scores.")
        elif combined_score >= 70:
            assessment_parts.append("Acceptable quality with room for improvement in accuracy or relevance.")
        elif combined_score >= 60:
            assessment_parts.append("Marginal quality requiring attention to both accuracy and relevance.")
        else:
            assessment_parts.append("Poor quality requiring significant improvements across all dimensions.")
        
        # Specific performance highlights
        if accuracy_score.overall_score > relevance_score.overall_score + 10:
            assessment_parts.append("Factual accuracy is stronger than relevance.")
        elif relevance_score.overall_score > accuracy_score.overall_score + 10:
            assessment_parts.append("Relevance is stronger than factual accuracy.")
        else:
            assessment_parts.append("Balanced performance between accuracy and relevance.")
        
        # Confidence assessment
        avg_confidence = (accuracy_score.confidence_score + relevance_score.confidence_score) / 2
        if avg_confidence >= 80:
            assessment_parts.append("High confidence in quality assessment.")
        elif avg_confidence >= 60:
            assessment_parts.append("Moderate confidence in quality assessment.")
        else:
            assessment_parts.append("Low confidence suggests need for more evidence or claims.")
        
        # Reliability indicator
        if accuracy_score.is_reliable and relevance_score.overall_score >= 70:
            assessment_parts.append("Results are considered reliable for production use.")
        else:
            assessment_parts.append("Results require additional validation before production use.")
        
        return " ".join(assessment_parts)
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get statistics about accuracy scoring performance."""
        
        stats = {
            'total_scorings': self.scoring_stats['total_scorings'],
            'total_claims_scored': self.scoring_stats['total_claims_scored'],
            'average_claims_per_scoring': (
                self.scoring_stats['total_claims_scored'] / 
                max(1, self.scoring_stats['total_scorings'])
            ),
            'processing_times': {
                'count': len(self.processing_times),
                'average_ms': statistics.mean(self.processing_times) if self.processing_times else 0,
                'median_ms': statistics.median(self.processing_times) if self.processing_times else 0,
                'min_ms': min(self.processing_times) if self.processing_times else 0,
                'max_ms': max(self.processing_times) if self.processing_times else 0
            },
            'configuration': {
                'scoring_weights': self.scoring_weights,
                'claim_type_weights': self.claim_type_weights,
                'integration_enabled': self.config['integration_settings']['enable_relevance_integration']
            }
        }
        
        return stats


# Convenience functions for integration
async def score_verification_results(
    verification_results: List['VerificationResult'],
    claims: Optional[List['ExtractedClaim']] = None,
    config: Optional[Dict[str, Any]] = None
) -> AccuracyScore:
    """
    Convenience function for scoring verification results.
    
    Args:
        verification_results: List of VerificationResult objects
        claims: Optional list of ExtractedClaim objects
        config: Optional configuration
        
    Returns:
        AccuracyScore with comprehensive assessment
    """
    
    scorer = FactualAccuracyScorer(config=config)
    return await scorer.score_accuracy(verification_results, claims)


async def generate_accuracy_report(
    verification_results: List['VerificationResult'],
    claims: Optional[List['ExtractedClaim']] = None,
    query: Optional[str] = None,
    response: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> AccuracyReport:
    """
    Convenience function for generating comprehensive accuracy report.
    
    Args:
        verification_results: List of VerificationResult objects
        claims: Optional list of ExtractedClaim objects
        query: Optional original query
        response: Optional original response
        config: Optional configuration
        
    Returns:
        AccuracyReport with comprehensive analysis
    """
    
    scorer = FactualAccuracyScorer(config=config)
    return await scorer.generate_comprehensive_report(
        verification_results, claims, query, response
    )


async def integrate_quality_assessment(
    verification_results: List['VerificationResult'],
    query: str,
    response: str,
    claims: Optional[List['ExtractedClaim']] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for integrated quality assessment.
    
    Args:
        verification_results: List of VerificationResult objects  
        query: Original query
        response: Original response
        claims: Optional list of ExtractedClaim objects
        config: Optional configuration
        
    Returns:
        Dictionary with integrated quality assessment
    """
    
    scorer = FactualAccuracyScorer(config=config)
    
    # Calculate accuracy score
    accuracy_score = await scorer.score_accuracy(verification_results, claims)
    
    # Integrate with relevance scorer
    return await scorer.integrate_with_relevance_scorer(
        accuracy_score, query, response
    )


if __name__ == "__main__":
    # Simple test example
    async def test_accuracy_scoring():
        """Test the accuracy scoring system."""
        
        print("Factual Accuracy Scorer initialized successfully!")
        print("For full testing, integrate with FactualAccuracyValidator results")
        
        # Example of creating test accuracy score
        test_score = AccuracyScore(
            overall_score=85.5,
            dimension_scores={
                'claim_verification': 88.0,
                'evidence_quality': 82.0,
                'coverage_assessment': 87.0,
                'consistency_analysis': 84.0,
                'confidence_factor': 86.0
            },
            claim_type_scores={
                'numeric': 90.0,
                'qualitative': 85.0,
                'methodological': 88.0
            },
            evidence_quality_score=82.0,
            coverage_score=87.0,
            consistency_score=84.0,
            confidence_score=86.0,
            grade=AccuracyGrade.GOOD,
            total_claims_assessed=15,
            processing_time_ms=245.7
        )
        
        print(f"\nTest Accuracy Score: {test_score.accuracy_percentage}")
        print(f"Grade: {test_score.grade.value}")
        print(f"Reliability: {'Reliable' if test_score.is_reliable else 'Needs Improvement'}")
        print(f"Claims Assessed: {test_score.total_claims_assessed}")
        print(f"Processing Time: {test_score.processing_time_ms:.1f}ms")
    
    # Run test if executed directly
    asyncio.run(test_accuracy_scoring())