#!/usr/bin/env python3
"""
Integrated Quality Assessment Workflow for Clinical Metabolomics Oracle.

This module provides integrated workflows that combine relevance scoring,
response quality assessment, and factual accuracy validation into comprehensive
quality evaluation pipelines for the Clinical Metabolomics Oracle LightRAG system.

Classes:
    - IntegratedQualityWorkflowError: Base exception for workflow errors
    - QualityAssessmentResult: Comprehensive quality assessment results
    - IntegratedQualityWorkflow: Main workflow orchestrator

Key Features:
    - Seamless integration of all quality assessment components
    - Parallel processing for performance optimization
    - Comprehensive error handling and fallback mechanisms
    - Backwards compatibility with existing workflows
    - Configurable assessment pipelines
    - Detailed reporting and analytics

Integration Components:
    - ClinicalMetabolomicsRelevanceScorer: Multi-dimensional relevance scoring
    - EnhancedResponseQualityAssessor: Quality assessment with factual validation
    - BiomedicalClaimExtractor: Factual claim extraction
    - FactualAccuracyValidator: Claim verification against documents
    - FactualAccuracyScorer: Comprehensive accuracy scoring

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Integrated Quality Assessment Implementation
"""

import asyncio
import json
import logging
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Configure logging
logger = logging.getLogger(__name__)


class IntegratedQualityWorkflowError(Exception):
    """Base custom exception for integrated quality workflow errors."""
    pass


@dataclass
class QualityAssessmentResult:
    """
    Comprehensive quality assessment result combining all assessment dimensions.
    
    Attributes:
        # Overall assessment
        overall_quality_score: Combined overall quality score (0-100)
        quality_grade: Human-readable quality grade
        assessment_confidence: Confidence in the assessment (0-100)
        
        # Component scores
        relevance_assessment: Results from ClinicalMetabolomicsRelevanceScorer
        quality_metrics: Results from EnhancedResponseQualityAssessor  
        factual_accuracy_results: Results from factual accuracy validation
        
        # Processing metadata
        processing_time_ms: Total processing time in milliseconds
        components_used: List of assessment components used
        error_details: Any errors encountered during assessment
        
        # Integration analysis
        consistency_analysis: Cross-component consistency analysis
        strength_areas: List of identified strength areas
        improvement_areas: List of areas needing improvement
        actionable_recommendations: List of specific recommendations
        
        # Performance metrics
        performance_metrics: Performance and efficiency metrics
        resource_usage: Resource usage statistics
        
        # Validation metadata
        validation_timestamp: When the assessment was performed
        configuration_used: Configuration parameters used
    """
    # Overall assessment
    overall_quality_score: float
    quality_grade: str
    assessment_confidence: float
    
    # Component results
    relevance_assessment: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    factual_accuracy_results: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    processing_time_ms: float = 0.0
    components_used: List[str] = field(default_factory=list)
    error_details: List[str] = field(default_factory=list)
    
    # Integration analysis
    consistency_analysis: Dict[str, Any] = field(default_factory=dict)
    strength_areas: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    actionable_recommendations: List[str] = field(default_factory=list)
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    # Validation metadata  
    validation_timestamp: datetime = field(default_factory=datetime.now)
    configuration_used: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        result = asdict(self)
        result['validation_timestamp'] = self.validation_timestamp.isoformat()
        return result
    
    @property
    def summary(self) -> str:
        """Generate brief assessment summary."""
        return (
            f"Quality Assessment Summary\n"
            f"Overall Quality: {self.overall_quality_score:.1f}/100 ({self.quality_grade})\n"
            f"Assessment Confidence: {self.assessment_confidence:.1f}/100\n"
            f"Components Used: {', '.join(self.components_used)}\n"
            f"Processing Time: {self.processing_time_ms:.2f}ms\n"
            f"Key Strengths: {', '.join(self.strength_areas[:3])}\n"
            f"Areas for Improvement: {', '.join(self.improvement_areas[:3])}"
        )


class IntegratedQualityWorkflow:
    """
    Integrated quality assessment workflow orchestrator.
    
    Coordinates multiple quality assessment components to provide
    comprehensive quality evaluation with integrated analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integrated quality workflow.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize assessment components
        self._relevance_scorer = None
        self._quality_assessor = None
        self._claim_extractor = None
        self._factual_validator = None
        self._accuracy_scorer = None
        
        # Performance tracking
        self._performance_metrics = defaultdict(list)
        self._assessment_history = []
        
        # Initialize components
        self._initialize_components()
        
        logger.info("IntegratedQualityWorkflow initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for integrated workflow."""
        return {
            'enable_parallel_processing': True,
            'enable_factual_validation': True,
            'enable_relevance_scoring': True,
            'enable_quality_assessment': True,
            'fallback_on_component_failure': True,
            'max_processing_time_seconds': 30.0,
            'confidence_threshold': 70.0,
            'detailed_reporting': True,
            'enable_caching': True,
            'component_weights': {
                'relevance_score': 0.35,
                'quality_metrics': 0.35,
                'factual_accuracy': 0.30
            },
            'performance_optimization': {
                'use_async_components': True,
                'max_concurrent_assessments': 5,
                'timeout_per_component': 10.0
            }
        }
    
    def _initialize_components(self):
        """Initialize all quality assessment components."""
        try:
            # Initialize ClinicalMetabolomicsRelevanceScorer
            if self.config.get('enable_relevance_scoring', True):
                try:
                    from .relevance_scorer import ClinicalMetabolomicsRelevanceScorer
                    self._relevance_scorer = ClinicalMetabolomicsRelevanceScorer(
                        config=self.config.get('relevance_scorer_config', {})
                    )
                    logger.info("ClinicalMetabolomicsRelevanceScorer initialized")
                except ImportError:
                    logger.warning("ClinicalMetabolomicsRelevanceScorer not available")
            
            # Initialize EnhancedResponseQualityAssessor
            if self.config.get('enable_quality_assessment', True):
                try:
                    from .enhanced_response_quality_assessor import EnhancedResponseQualityAssessor
                    self._quality_assessor = EnhancedResponseQualityAssessor(
                        config=self.config.get('quality_assessor_config', {})
                    )
                    logger.info("EnhancedResponseQualityAssessor initialized")
                except ImportError:
                    logger.warning("EnhancedResponseQualityAssessor not available")
            
            # Initialize factual accuracy components
            if self.config.get('enable_factual_validation', True):
                try:
                    from .claim_extractor import BiomedicalClaimExtractor
                    self._claim_extractor = BiomedicalClaimExtractor()
                    logger.info("BiomedicalClaimExtractor initialized")
                except ImportError:
                    logger.warning("BiomedicalClaimExtractor not available")
                
                try:
                    from .factual_accuracy_validator import FactualAccuracyValidator
                    self._factual_validator = FactualAccuracyValidator()
                    logger.info("FactualAccuracyValidator initialized")
                except ImportError:
                    logger.warning("FactualAccuracyValidator not available")
                
                try:
                    from .accuracy_scorer import FactualAccuracyScorer
                    self._accuracy_scorer = FactualAccuracyScorer()
                    logger.info("FactualAccuracyScorer initialized")
                except ImportError:
                    logger.warning("FactualAccuracyScorer not available")
            
            # Configure cross-component integration
            self._configure_component_integration()
            
        except Exception as e:
            logger.error(f"Error initializing workflow components: {str(e)}")
            if not self.config.get('fallback_on_component_failure', True):
                raise IntegratedQualityWorkflowError(f"Component initialization failed: {str(e)}")
    
    def _configure_component_integration(self):
        """Configure integration between components."""
        try:
            # Enable factual accuracy in relevance scorer if available
            if (self._relevance_scorer and self._claim_extractor and 
                self._factual_validator and hasattr(self._relevance_scorer, 'enable_factual_accuracy_validation')):
                self._relevance_scorer.enable_factual_accuracy_validation(
                    self._claim_extractor, self._factual_validator
                )
                logger.info("Factual accuracy enabled in relevance scorer")
            
            # Enable factual validation in quality assessor if available
            if (self._quality_assessor and self._claim_extractor and 
                self._factual_validator and self._accuracy_scorer):
                self._quality_assessor.enable_factual_validation(
                    self._claim_extractor, self._factual_validator, self._accuracy_scorer
                )
                logger.info("Factual validation enabled in quality assessor")
        except Exception as e:
            logger.warning(f"Error configuring component integration: {str(e)}")
    
    async def assess_comprehensive_quality(self,
                                         query: str,
                                         response: str,
                                         source_documents: Optional[List[str]] = None,
                                         expected_concepts: Optional[List[str]] = None,
                                         metadata: Optional[Dict[str, Any]] = None) -> QualityAssessmentResult:
        """
        Perform comprehensive quality assessment using all available components.
        
        Args:
            query: Original user query
            response: System response to assess
            source_documents: Optional source documents for validation
            expected_concepts: Optional list of expected concepts
            metadata: Optional metadata for assessment context
            
        Returns:
            QualityAssessmentResult with comprehensive assessment
            
        Raises:
            IntegratedQualityWorkflowError: If assessment fails
        """
        start_time = time.time()
        components_used = []
        error_details = []
        
        try:
            logger.info(f"Starting comprehensive quality assessment for query: {query[:50]}...")
            
            # Validate inputs
            if not query or not response:
                raise IntegratedQualityWorkflowError("Query and response are required")
            
            # Set defaults
            source_documents = source_documents or []
            expected_concepts = expected_concepts or []
            metadata = metadata or {}
            
            # Run component assessments in parallel if enabled
            if self.config.get('enable_parallel_processing', True):
                results = await self._run_parallel_assessments(
                    query, response, source_documents, expected_concepts, metadata
                )
            else:
                results = await self._run_sequential_assessments(
                    query, response, source_documents, expected_concepts, metadata
                )
            
            relevance_result, quality_result, factual_result = results
            
            # Track which components were used
            if relevance_result:
                components_used.append('ClinicalMetabolomicsRelevanceScorer')
            if quality_result:
                components_used.append('EnhancedResponseQualityAssessor')
            if factual_result:
                components_used.append('FactualAccuracyValidation')
            
            # Calculate integrated overall score
            overall_score, assessment_confidence = self._calculate_integrated_scores(
                relevance_result, quality_result, factual_result
            )
            
            # Determine quality grade
            quality_grade = self._determine_quality_grade(overall_score)
            
            # Perform cross-component analysis
            consistency_analysis = self._analyze_component_consistency(
                relevance_result, quality_result, factual_result
            )
            
            # Identify strengths and improvement areas
            strength_areas, improvement_areas = self._identify_quality_dimensions(
                relevance_result, quality_result, factual_result
            )
            
            # Generate actionable recommendations
            recommendations = self._generate_recommendations(
                relevance_result, quality_result, factual_result, 
                strength_areas, improvement_areas
            )
            
            # Calculate performance metrics
            processing_time = (time.time() - start_time) * 1000
            performance_metrics = self._calculate_performance_metrics(
                processing_time, components_used
            )
            
            # Create comprehensive result
            result = QualityAssessmentResult(
                overall_quality_score=overall_score,
                quality_grade=quality_grade,
                assessment_confidence=assessment_confidence,
                relevance_assessment=relevance_result.to_dict() if relevance_result else None,
                quality_metrics=quality_result.to_dict() if quality_result else None,
                factual_accuracy_results=factual_result,
                processing_time_ms=processing_time,
                components_used=components_used,
                error_details=error_details,
                consistency_analysis=consistency_analysis,
                strength_areas=strength_areas,
                improvement_areas=improvement_areas,
                actionable_recommendations=recommendations,
                performance_metrics=performance_metrics,
                configuration_used=self.config
            )
            
            # Update performance tracking
            self._update_performance_tracking(result)
            
            logger.info(
                f"Comprehensive assessment completed in {processing_time:.2f}ms: "
                f"{overall_score:.1f}/100 ({quality_grade})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive quality assessment: {str(e)}")
            logger.error(traceback.format_exc())
            raise IntegratedQualityWorkflowError(f"Assessment failed: {str(e)}") from e
    
    async def _run_parallel_assessments(self,
                                      query: str,
                                      response: str,
                                      source_documents: List[str],
                                      expected_concepts: List[str],
                                      metadata: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        """Run component assessments in parallel."""
        tasks = []
        
        # Create assessment tasks
        if self._relevance_scorer:
            tasks.append(self._safe_relevance_assessment(query, response, metadata))
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
        
        if self._quality_assessor:
            tasks.append(self._safe_quality_assessment(
                query, response, source_documents, expected_concepts, metadata
            ))
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
        
        if self._has_factual_components():
            tasks.append(self._safe_factual_assessment(query, response, source_documents))
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
        
        # Run tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.get('max_processing_time_seconds', 30.0)
            )
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Component {i} failed: {str(result)}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return tuple(processed_results)
            
        except asyncio.TimeoutError:
            logger.warning("Assessment timeout - returning partial results")
            return None, None, None
    
    async def _run_sequential_assessments(self,
                                        query: str,
                                        response: str,
                                        source_documents: List[str],
                                        expected_concepts: List[str],
                                        metadata: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        """Run component assessments sequentially."""
        relevance_result = None
        quality_result = None
        factual_result = None
        
        # Relevance assessment
        if self._relevance_scorer:
            try:
                relevance_result = await self._safe_relevance_assessment(query, response, metadata)
            except Exception as e:
                logger.warning(f"Relevance assessment failed: {str(e)}")
        
        # Quality assessment
        if self._quality_assessor:
            try:
                quality_result = await self._safe_quality_assessment(
                    query, response, source_documents, expected_concepts, metadata
                )
            except Exception as e:
                logger.warning(f"Quality assessment failed: {str(e)}")
        
        # Factual accuracy assessment
        if self._has_factual_components():
            try:
                factual_result = await self._safe_factual_assessment(query, response, source_documents)
            except Exception as e:
                logger.warning(f"Factual assessment failed: {str(e)}")
        
        return relevance_result, quality_result, factual_result
    
    async def _safe_relevance_assessment(self, query: str, response: str, metadata: Dict[str, Any]):
        """Safely run relevance assessment with error handling."""
        try:
            return await asyncio.wait_for(
                self._relevance_scorer.calculate_relevance_score(query, response, metadata),
                timeout=self.config['performance_optimization']['timeout_per_component']
            )
        except Exception as e:
            logger.warning(f"Relevance assessment error: {str(e)}")
            if self.config.get('fallback_on_component_failure', True):
                return None
            raise
    
    async def _safe_quality_assessment(self, query: str, response: str, 
                                     source_documents: List[str], expected_concepts: List[str],
                                     metadata: Dict[str, Any]):
        """Safely run quality assessment with error handling."""
        try:
            return await asyncio.wait_for(
                self._quality_assessor.assess_response_quality(
                    query, response, source_documents, expected_concepts, metadata
                ),
                timeout=self.config['performance_optimization']['timeout_per_component']
            )
        except Exception as e:
            logger.warning(f"Quality assessment error: {str(e)}")
            if self.config.get('fallback_on_component_failure', True):
                return None
            raise
    
    async def _safe_factual_assessment(self, query: str, response: str, source_documents: List[str]):
        """Safely run factual accuracy assessment with error handling."""
        try:
            # Extract claims
            claims = await self._claim_extractor.extract_claims(response)
            if not claims:
                return {'method': 'no_claims', 'overall_score': 75.0}
            
            # Verify claims
            verification_report = await self._factual_validator.verify_claims(claims)
            
            # Score results
            accuracy_score = await self._accuracy_scorer.score_accuracy(
                verification_report.verification_results, claims
            )
            
            return {
                'method': 'full_pipeline',
                'overall_score': accuracy_score.overall_score,
                'accuracy_score': accuracy_score.to_dict(),
                'verification_report': verification_report.to_dict(),
                'claims_count': len(claims)
            }
            
        except Exception as e:
            logger.warning(f"Factual assessment error: {str(e)}")
            if self.config.get('fallback_on_component_failure', True):
                return {'method': 'fallback', 'overall_score': 60.0, 'error': str(e)}
            raise
    
    def _has_factual_components(self) -> bool:
        """Check if factual accuracy components are available."""
        return (self._claim_extractor is not None and 
                self._factual_validator is not None and
                self._accuracy_scorer is not None)
    
    def _calculate_integrated_scores(self, 
                                   relevance_result, 
                                   quality_result, 
                                   factual_result) -> Tuple[float, float]:
        """Calculate integrated overall score and confidence."""
        scores = []
        weights = []
        confidences = []
        
        # Collect scores from available components
        if relevance_result:
            scores.append(relevance_result.overall_score)
            weights.append(self.config['component_weights']['relevance_score'])
            confidences.append(getattr(relevance_result, 'confidence_score', 75.0))
        
        if quality_result:
            scores.append(quality_result.overall_quality_score)
            weights.append(self.config['component_weights']['quality_metrics'])
            confidences.append(getattr(quality_result, 'factual_confidence_score', 75.0))
        
        if factual_result and factual_result.get('overall_score'):
            scores.append(factual_result['overall_score'])
            weights.append(self.config['component_weights']['factual_accuracy'])
            confidences.append(75.0)  # Default confidence
        
        # Calculate weighted average
        if scores:
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                overall_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
            else:
                overall_score = statistics.mean(scores)
            
            # Calculate confidence
            assessment_confidence = statistics.mean(confidences) if confidences else 50.0
        else:
            overall_score = 0.0
            assessment_confidence = 0.0
        
        return overall_score, assessment_confidence
    
    def _determine_quality_grade(self, overall_score: float) -> str:
        """Determine quality grade from overall score."""
        if overall_score >= 90:
            return "Excellent"
        elif overall_score >= 80:
            return "Good"
        elif overall_score >= 70:
            return "Acceptable"
        elif overall_score >= 60:
            return "Marginal"
        else:
            return "Poor"
    
    def _analyze_component_consistency(self, relevance_result, quality_result, factual_result) -> Dict[str, Any]:
        """Analyze consistency across component assessments."""
        consistency_analysis = {
            'component_score_variance': 0.0,
            'score_agreement_level': 'unknown',
            'conflicting_assessments': [],
            'consistent_strengths': [],
            'analysis_notes': []
        }
        
        try:
            # Collect scores for variance analysis
            scores = []
            if relevance_result:
                scores.append(relevance_result.overall_score)
            if quality_result:
                scores.append(quality_result.overall_quality_score)
            if factual_result and factual_result.get('overall_score'):
                scores.append(factual_result['overall_score'])
            
            if len(scores) >= 2:
                # Calculate variance
                consistency_analysis['component_score_variance'] = statistics.variance(scores)
                
                # Determine agreement level
                score_range = max(scores) - min(scores)
                if score_range <= 10:
                    consistency_analysis['score_agreement_level'] = 'high'
                elif score_range <= 20:
                    consistency_analysis['score_agreement_level'] = 'moderate'
                else:
                    consistency_analysis['score_agreement_level'] = 'low'
                
                # Identify conflicts and consistencies
                avg_score = statistics.mean(scores)
                for i, score in enumerate(scores):
                    component_names = ['relevance', 'quality', 'factual'][i]
                    if abs(score - avg_score) > 15:
                        consistency_analysis['conflicting_assessments'].append(
                            f"{component_names} score ({score:.1f}) differs significantly from average ({avg_score:.1f})"
                        )
                
                # Analysis notes
                if consistency_analysis['score_agreement_level'] == 'high':
                    consistency_analysis['analysis_notes'].append("High consistency across components")
                elif consistency_analysis['score_agreement_level'] == 'low':
                    consistency_analysis['analysis_notes'].append("Low consistency - investigate component differences")
        
        except Exception as e:
            logger.warning(f"Error in consistency analysis: {str(e)}")
            consistency_analysis['analysis_notes'].append(f"Analysis error: {str(e)}")
        
        return consistency_analysis
    
    def _identify_quality_dimensions(self, 
                                   relevance_result, 
                                   quality_result, 
                                   factual_result) -> Tuple[List[str], List[str]]:
        """Identify strength areas and improvement areas across components."""
        strengths = []
        improvements = []
        
        # Analyze relevance results
        if relevance_result:
            for dimension, score in relevance_result.dimension_scores.items():
                if score >= 85:
                    strengths.append(f"Excellent {dimension.replace('_', ' ')}")
                elif score < 60:
                    improvements.append(f"Improve {dimension.replace('_', ' ')}")
        
        # Analyze quality results
        if quality_result:
            if quality_result.factual_accuracy_score >= 85:
                strengths.append("High factual accuracy")
            elif quality_result.factual_accuracy_score < 60:
                improvements.append("Improve factual accuracy")
            
            if quality_result.clarity_score >= 85:
                strengths.append("Excellent clarity")
            elif quality_result.clarity_score < 60:
                improvements.append("Improve response clarity")
            
            if quality_result.biomedical_terminology_score >= 85:
                strengths.append("Appropriate biomedical terminology")
            elif quality_result.biomedical_terminology_score < 60:
                improvements.append("Enhance biomedical terminology usage")
        
        # Analyze factual results
        if factual_result and factual_result.get('overall_score'):
            score = factual_result['overall_score']
            if score >= 85:
                strengths.append("Strong factual validation")
            elif score < 60:
                improvements.append("Enhance factual verification")
        
        return strengths[:5], improvements[:5]  # Limit to top 5 each
    
    def _generate_recommendations(self,
                                relevance_result,
                                quality_result,
                                factual_result,
                                strength_areas: List[str],
                                improvement_areas: List[str]) -> List[str]:
        """Generate actionable recommendations based on assessment results."""
        recommendations = []
        
        # Overall performance recommendations
        all_scores = []
        if relevance_result:
            all_scores.append(relevance_result.overall_score)
        if quality_result:
            all_scores.append(quality_result.overall_quality_score)
        if factual_result and factual_result.get('overall_score'):
            all_scores.append(factual_result['overall_score'])
        
        if all_scores:
            avg_score = statistics.mean(all_scores)
            if avg_score >= 90:
                recommendations.append("Excellent overall performance - maintain current quality standards")
            elif avg_score >= 80:
                recommendations.append("Good performance - focus on addressing specific improvement areas")
            elif avg_score >= 70:
                recommendations.append("Acceptable performance - systematic improvements needed")
            else:
                recommendations.append("Significant quality improvements required across all dimensions")
        
        # Component-specific recommendations
        if relevance_result and relevance_result.overall_score < 70:
            recommendations.append("Improve query-response alignment and biomedical context depth")
        
        if quality_result and quality_result.overall_quality_score < 70:
            recommendations.append("Enhance response structure, clarity, and completeness")
        
        if factual_result and factual_result.get('overall_score', 0) < 70:
            recommendations.append("Strengthen factual accuracy through better evidence integration")
        
        # Specific improvement recommendations
        if "Improve factual accuracy" in improvement_areas:
            recommendations.append("Implement more rigorous fact-checking and source validation")
        
        if "Improve response clarity" in improvement_areas:
            recommendations.append("Simplify language while maintaining technical accuracy")
        
        if "Enhance biomedical terminology usage" in improvement_areas:
            recommendations.append("Incorporate more domain-specific terminology appropriately")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _calculate_performance_metrics(self, 
                                     processing_time: float, 
                                     components_used: List[str]) -> Dict[str, Any]:
        """Calculate performance and efficiency metrics."""
        return {
            'total_processing_time_ms': processing_time,
            'components_used_count': len(components_used),
            'avg_component_time_ms': processing_time / max(len(components_used), 1),
            'performance_grade': self._get_performance_grade(processing_time),
            'efficiency_score': self._calculate_efficiency_score(processing_time, components_used),
            'throughput_estimate': 1000 / processing_time if processing_time > 0 else 0  # Assessments per second
        }
    
    def _get_performance_grade(self, processing_time: float) -> str:
        """Get performance grade based on processing time."""
        if processing_time <= 1000:  # <= 1 second
            return "Excellent"
        elif processing_time <= 3000:  # <= 3 seconds
            return "Good"
        elif processing_time <= 5000:  # <= 5 seconds
            return "Acceptable"
        elif processing_time <= 10000:  # <= 10 seconds
            return "Slow"
        else:
            return "Very Slow"
    
    def _calculate_efficiency_score(self, processing_time: float, components_used: List[str]) -> float:
        """Calculate efficiency score based on time and components."""
        base_score = 100.0
        
        # Time penalty
        time_penalty = min(50.0, processing_time / 100.0)  # 1 point per 100ms
        
        # Component complexity bonus (more components = more work)
        complexity_bonus = len(components_used) * 2.0
        
        efficiency_score = base_score - time_penalty + complexity_bonus
        return min(100.0, max(0.0, efficiency_score))
    
    def _update_performance_tracking(self, result: QualityAssessmentResult):
        """Update performance tracking metrics."""
        self._performance_metrics['processing_times'].append(result.processing_time_ms)
        self._performance_metrics['quality_scores'].append(result.overall_quality_score)
        self._performance_metrics['confidence_scores'].append(result.assessment_confidence)
        self._assessment_history.append(result)
        
        # Keep only recent history
        if len(self._assessment_history) > 100:
            self._assessment_history = self._assessment_history[-100:]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the workflow."""
        if not self._performance_metrics['processing_times']:
            return {'status': 'no_data'}
        
        processing_times = self._performance_metrics['processing_times']
        quality_scores = self._performance_metrics['quality_scores']
        
        return {
            'total_assessments': len(processing_times),
            'avg_processing_time_ms': statistics.mean(processing_times),
            'median_processing_time_ms': statistics.median(processing_times),
            'min_processing_time_ms': min(processing_times),
            'max_processing_time_ms': max(processing_times),
            'avg_quality_score': statistics.mean(quality_scores),
            'quality_score_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            'assessments_per_minute': len(processing_times) / (sum(processing_times) / 60000) if sum(processing_times) > 0 else 0
        }
    
    async def batch_assess_quality(self, 
                                 assessments: List[Tuple[str, str, List[str], List[str]]]) -> List[QualityAssessmentResult]:
        """
        Perform batch quality assessment for multiple query-response pairs.
        
        Args:
            assessments: List of (query, response, source_docs, expected_concepts) tuples
            
        Returns:
            List of QualityAssessmentResult for each assessment
        """
        results = []
        max_concurrent = self.config['performance_optimization'].get('max_concurrent_assessments', 5)
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(assessments), max_concurrent):
            batch = assessments[i:i + max_concurrent]
            
            # Create tasks for batch
            tasks = [
                self.assess_comprehensive_quality(query, response, source_docs, expected_concepts)
                for query, response, source_docs, expected_concepts in batch
            ]
            
            # Run batch
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch assessment error: {str(result)}")
                        # Create fallback result
                        fallback_result = QualityAssessmentResult(
                            overall_quality_score=0.0,
                            quality_grade="Error",
                            assessment_confidence=0.0,
                            error_details=[str(result)]
                        )
                        results.append(fallback_result)
                    else:
                        results.append(result)
            
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                # Add error results for the batch
                for _ in batch:
                    error_result = QualityAssessmentResult(
                        overall_quality_score=0.0,
                        quality_grade="Error", 
                        assessment_confidence=0.0,
                        error_details=[str(e)]
                    )
                    results.append(error_result)
        
        return results


# Convenience functions for integration
async def assess_response_comprehensive_quality(query: str,
                                              response: str,
                                              source_documents: Optional[List[str]] = None,
                                              expected_concepts: Optional[List[str]] = None,
                                              config: Optional[Dict[str, Any]] = None) -> QualityAssessmentResult:
    """
    Convenience function for comprehensive quality assessment.
    
    Args:
        query: Original user query
        response: System response to assess
        source_documents: Optional source documents
        expected_concepts: Optional expected concepts
        config: Optional configuration
        
    Returns:
        QualityAssessmentResult with comprehensive assessment
    """
    workflow = IntegratedQualityWorkflow(config)
    return await workflow.assess_comprehensive_quality(
        query, response, source_documents, expected_concepts
    )


if __name__ == "__main__":
    # Comprehensive test example
    async def test_integrated_workflow():
        """Test the integrated quality assessment workflow."""
        
        print("Integrated Quality Assessment Workflow Test")
        print("=" * 60)
        
        workflow = IntegratedQualityWorkflow()
        
        # Test data
        query = "What are the clinical applications of metabolomics in personalized medicine?"
        response = """Metabolomics has several important clinical applications in personalized medicine. 
        First, it enables biomarker discovery for disease diagnosis and prognosis. LC-MS and GC-MS platforms 
        are used to analyze metabolite profiles in patient samples. Studies show that metabolomic signatures 
        can predict treatment responses and identify patients who may benefit from specific therapies. 
        Research indicates that metabolomics-based approaches show promise for precision medicine applications 
        in cancer, cardiovascular disease, and metabolic disorders."""
        
        source_documents = ["Metabolomics research paper 1", "Clinical study on biomarkers"]
        expected_concepts = ["metabolomics", "personalized medicine", "biomarker", "clinical"]
        
        # Perform assessment
        result = await workflow.assess_comprehensive_quality(
            query=query,
            response=response,
            source_documents=source_documents,
            expected_concepts=expected_concepts
        )
        
        # Display results
        print(result.summary)
        print(f"\nDetailed Results:")
        print(f"Components Used: {result.components_used}")
        print(f"Processing Time: {result.processing_time_ms:.2f}ms")
        print(f"Assessment Confidence: {result.assessment_confidence:.1f}/100")
        
        if result.consistency_analysis:
            print(f"\nConsistency Analysis:")
            print(f"Score Agreement: {result.consistency_analysis.get('score_agreement_level', 'unknown')}")
            print(f"Variance: {result.consistency_analysis.get('component_score_variance', 0):.2f}")
        
        if result.actionable_recommendations:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(result.actionable_recommendations[:3], 1):
                print(f"{i}. {rec}")
        
        # Performance statistics
        perf_stats = workflow.get_performance_statistics()
        if perf_stats.get('status') != 'no_data':
            print(f"\nPerformance Statistics:")
            print(f"Total Assessments: {perf_stats['total_assessments']}")
            print(f"Average Processing Time: {perf_stats['avg_processing_time_ms']:.2f}ms")
    
    # Run comprehensive test
    asyncio.run(test_integrated_workflow())