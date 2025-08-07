#!/usr/bin/env python3
"""
Clinical Metabolomics Response Relevance Scoring System.

This module implements the ClinicalMetabolomicsRelevanceScorer class which provides
specialized relevance scoring for clinical metabolomics query-response pairs.

The scorer builds upon existing ResponseQualityAssessor infrastructure and provides:
- Multi-dimensional relevance scoring
- Query-type adaptive weighting
- Semantic similarity assessment
- Domain expertise validation
- Real-time performance optimization
- Comprehensive response length validation
- Response structure quality assessment
- Readability and clarity evaluation
- Completeness checking
- Response formatting quality analysis

Key Features:
- Validates response length appropriateness for different query types
- Assesses structure quality including formatting, organization, and coherence
- Provides detailed quality recommendations for response improvement
- Supports batch processing of multiple query-response pairs
- Offers comprehensive quality validation beyond just relevance scoring

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 2.0.0 - Enhanced with comprehensive quality checks
Related to: CMO-LIGHTRAG-009-T02 - Clinical Metabolomics Relevance Scoring
"""

import asyncio
import statistics
import time
import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging
import math

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RelevanceScore:
    """
    Comprehensive relevance scoring results for clinical metabolomics responses.
    
    Attributes:
        overall_score: Overall relevance score (0-100)
        dimension_scores: Scores for each relevance dimension
        query_type: Classified query type
        weights_used: Weights applied for scoring
        explanation: Human-readable explanation of the scoring
        confidence_score: Confidence in the relevance assessment (0-100)
        processing_time_ms: Time taken for scoring in milliseconds
        metadata: Additional scoring metadata
    """
    overall_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    query_type: str = "general"
    weights_used: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def relevance_grade(self) -> str:
        """Convert overall score to human-readable grade."""
        if self.overall_score >= 90:
            return "Excellent"
        elif self.overall_score >= 80:
            return "Good"
        elif self.overall_score >= 70:
            return "Acceptable"
        elif self.overall_score >= 60:
            return "Marginal"
        else:
            return "Poor"


class QueryTypeClassifier:
    """Classifies queries into clinical metabolomics query types."""
    
    def __init__(self):
        self.classification_keywords = {
            'basic_definition': [
                'what is', 'define', 'definition', 'explain', 'basics', 
                'introduction', 'overview', 'meaning', 'concept'
            ],
            'clinical_application': [
                'clinical', 'patient', 'diagnosis', 'treatment', 'medical',
                'therapeutic', 'diagnostic', 'healthcare', 'therapy',
                'biomarker', 'precision medicine', 'personalized medicine'
            ],
            'analytical_method': [
                'LC-MS', 'GC-MS', 'NMR', 'method', 'analysis', 'protocol',
                'technique', 'instrumentation', 'mass spectrometry',
                'chromatography', 'UPLC', 'HILIC', 'sample preparation'
            ],
            'research_design': [
                'study design', 'statistics', 'statistical analysis', 'methodology',
                'experimental design', 'sample size', 'power analysis',
                'validation', 'reproducibility', 'quality control'
            ],
            'disease_specific': [
                'disease', 'cancer', 'diabetes', 'alzheimer', 'cardiovascular',
                'obesity', 'metabolic disorder', 'pathology', 'syndrome',
                'condition', 'illness', 'disorder'
            ]
        }
    
    def classify_query(self, query: str) -> str:
        """
        Classify query into one of the defined types.
        
        Args:
            query: Query text to classify
            
        Returns:
            Query type string
        """
        scores = {}
        query_lower = query.lower()
        
        for query_type, keywords in self.classification_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            # Weight exact matches higher
            exact_matches = sum(2 for keyword in keywords if f" {keyword} " in f" {query_lower} ")
            scores[query_type] = score + exact_matches
            
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return 'general'


class SemanticSimilarityEngine:
    """Handles semantic similarity calculations for queries and responses."""
    
    def __init__(self):
        # Initialize with simple text-based similarity for now
        # BioBERT integration can be added later
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'what', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }
    
    async def calculate_similarity(self, query: str, response: str) -> float:
        """
        Calculate semantic similarity between query and response.
        
        Args:
            query: Original query text
            response: Response text to evaluate
            
        Returns:
            Similarity score (0-100)
        """
        # Simple Jaccard similarity with biomedical term weighting
        query_terms = self._extract_meaningful_terms(query)
        response_terms = self._extract_meaningful_terms(response)
        
        if not query_terms:
            return 0.0
        
        intersection = len(query_terms.intersection(response_terms))
        union = len(query_terms.union(response_terms))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # Boost score for biomedical term matches
        biomedical_boost = self._calculate_biomedical_term_boost(query, response)
        
        similarity_score = (jaccard_similarity * 70) + biomedical_boost
        return min(100.0, similarity_score)
    
    def _extract_meaningful_terms(self, text: str) -> set:
        """Extract meaningful terms from text, excluding stopwords."""
        words = re.findall(r'\b\w+\b', text.lower())
        return {word for word in words if word not in self.stopwords and len(word) > 2}
    
    def _calculate_biomedical_term_boost(self, query: str, response: str) -> float:
        """Calculate boost score for biomedical term alignment."""
        biomedical_terms = [
            'metabolomics', 'metabolite', 'metabolism', 'biomarker',
            'clinical', 'diagnostic', 'therapeutic', 'LC-MS', 'GC-MS', 'NMR'
        ]
        
        query_bio_terms = {term for term in biomedical_terms if term in query.lower()}
        response_bio_terms = {term for term in biomedical_terms if term in response.lower()}
        
        if not query_bio_terms:
            return 0.0
        
        overlap = len(query_bio_terms.intersection(response_bio_terms))
        return min(30.0, overlap * 7.5)  # Up to 30 point boost


class WeightingSchemeManager:
    """Manages weighting schemes for different query types."""
    
    def __init__(self):
        self.weighting_schemes = {
            'basic_definition': {
                'metabolomics_relevance': 0.30,
                'query_alignment': 0.20,
                'scientific_rigor': 0.15,
                'clinical_applicability': 0.12,
                'biomedical_context_depth': 0.05,
                'response_length_quality': 0.08,
                'response_structure_quality': 0.10
            },
            'clinical_application': {
                'clinical_applicability': 0.25,
                'metabolomics_relevance': 0.20,
                'query_alignment': 0.18,
                'scientific_rigor': 0.12,
                'biomedical_context_depth': 0.08,
                'response_length_quality': 0.07,
                'response_structure_quality': 0.10
            },
            'analytical_method': {
                'metabolomics_relevance': 0.35,
                'query_alignment': 0.20,
                'scientific_rigor': 0.18,
                'biomedical_context_depth': 0.08,
                'clinical_applicability': 0.04,
                'response_length_quality': 0.08,
                'response_structure_quality': 0.07
            },
            'research_design': {
                'scientific_rigor': 0.25,
                'metabolomics_relevance': 0.20,
                'query_alignment': 0.18,
                'biomedical_context_depth': 0.12,
                'clinical_applicability': 0.08,
                'response_length_quality': 0.07,
                'response_structure_quality': 0.10
            },
            'disease_specific': {
                'clinical_applicability': 0.25,
                'biomedical_context_depth': 0.20,
                'metabolomics_relevance': 0.18,
                'query_alignment': 0.12,
                'scientific_rigor': 0.08,
                'response_length_quality': 0.07,
                'response_structure_quality': 0.10
            },
            'general': {
                'query_alignment': 0.20,
                'metabolomics_relevance': 0.20,
                'clinical_applicability': 0.18,
                'scientific_rigor': 0.12,
                'biomedical_context_depth': 0.12,
                'response_length_quality': 0.08,
                'response_structure_quality': 0.10
            }
        }
    
    def get_weights(self, query_type: str) -> Dict[str, float]:
        """Get weighting scheme for query type."""
        return self.weighting_schemes.get(query_type, self.weighting_schemes['general'])


class DomainExpertiseValidator:
    """Validates domain expertise and factual consistency."""
    
    def __init__(self):
        self.expertise_rules = {
            'analytical_method_compatibility': {
                'polar_metabolites': ['HILIC', 'negative mode', 'hydrophilic'],
                'lipids': ['C18 positive mode', 'lipid column', 'reverse phase'],
                'volatile_compounds': ['GC-MS', 'headspace', 'derivatization']
            },
            'statistical_appropriateness': {
                'univariate': ['t-test', 'ANOVA', 'fold change', 'mann-whitney'],
                'multivariate': ['PCA', 'PLS-DA', 'OPLS-DA', 'random forest'],
                'pathway_analysis': ['GSEA', 'pathway enrichment', 'MetaboAnalyst']
            },
            'clinical_validity': {
                'biomarker_criteria': ['sensitivity', 'specificity', 'reproducibility', 'ROC', 'AUC'],
                'study_requirements': ['sample size', 'validation cohort', 'clinical relevance']
            }
        }
        
        self.common_errors = [
            'always accurate', 'never fails', 'completely reliable',
            'revolutionary breakthrough', 'miracle solution', 'unprecedented results'
        ]
    
    async def validate_domain_expertise(self, response: str) -> float:
        """
        Validate domain expertise demonstrated in response.
        
        Args:
            response: Response text to validate
            
        Returns:
            Expertise score (0-100)
        """
        expertise_score = 70.0  # Base score
        response_lower = response.lower()
        
        # Check for appropriate terminology usage
        terminology_score = self._assess_terminology_usage(response_lower)
        
        # Check for methodological accuracy
        methodology_score = self._assess_methodology(response_lower)
        
        # Penalize for common errors or overstatements
        error_penalty = self._assess_error_penalty(response_lower)
        
        # Reward evidence-based statements
        evidence_bonus = self._assess_evidence_quality(response_lower)
        
        final_score = expertise_score + (terminology_score * 0.3) + (methodology_score * 0.4) - error_penalty + (evidence_bonus * 0.3)
        
        return max(0.0, min(100.0, final_score))
    
    def _assess_terminology_usage(self, response: str) -> float:
        """Assess appropriate use of technical terminology."""
        correct_usage = 0
        total_checks = 0
        
        for category, terms in self.expertise_rules.items():
            for subcategory, appropriate_terms in terms.items():
                for term in appropriate_terms:
                    total_checks += 1
                    if term in response:
                        correct_usage += 1
        
        return (correct_usage / max(total_checks, 1)) * 20  # Up to 20 points
    
    def _assess_methodology(self, response: str) -> float:
        """Assess methodological soundness."""
        methodology_indicators = [
            'validation', 'quality control', 'reproducibility',
            'statistical significance', 'p-value', 'confidence interval',
            'sample size', 'study design'
        ]
        
        found_indicators = sum(1 for indicator in methodology_indicators if indicator in response)
        return min(15.0, found_indicators * 2.5)  # Up to 15 points
    
    def _assess_error_penalty(self, response: str) -> float:
        """Assess penalty for common errors or overstatements."""
        penalty = 0
        for error in self.common_errors:
            if error in response:
                penalty += 10
        
        return min(penalty, 30.0)  # Max 30 point penalty
    
    def _assess_evidence_quality(self, response: str) -> float:
        """Assess quality of evidence presentation."""
        evidence_indicators = [
            'studies show', 'research indicates', 'data demonstrates',
            'according to', 'evidence suggests', 'meta-analysis'
        ]
        
        found_evidence = sum(1 for indicator in evidence_indicators if indicator in response)
        return min(10.0, found_evidence * 3.0)  # Up to 10 point bonus


class ClinicalMetabolomicsRelevanceScorer:
    """
    Main relevance scorer for clinical metabolomics query-response pairs.
    
    Provides comprehensive relevance scoring across multiple dimensions:
    - Metabolomics relevance
    - Clinical applicability  
    - Query alignment
    - Scientific rigor
    - Biomedical context depth
    - Response length quality
    - Response structure quality
    - Readability and clarity
    - Completeness assessment
    - Formatting quality
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the relevance scorer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.query_classifier = QueryTypeClassifier()
        self.semantic_engine = SemanticSimilarityEngine()
        self.weighting_manager = WeightingSchemeManager()
        self.domain_validator = DomainExpertiseValidator()
        
        # Biomedical keywords for relevance assessment
        self.biomedical_keywords = {
            'metabolomics_core': [
                'metabolomics', 'metabolite', 'metabolism', 'biomarker',
                'mass spectrometry', 'NMR', 'chromatography', 'metabolic pathway',
                'metabolome', 'small molecules', 'biochemical profiling'
            ],
            'analytical_methods': [
                'LC-MS', 'GC-MS', 'UPLC', 'HILIC', 'targeted analysis',
                'untargeted analysis', 'quantitative', 'qualitative',
                'sample preparation', 'derivatization', 'extraction'
            ],
            'clinical_terms': [
                'clinical', 'patient', 'disease', 'diagnosis', 'therapeutic',
                'biomedical', 'pathology', 'phenotype', 'precision medicine',
                'personalized medicine', 'treatment monitoring'
            ],
            'research_concepts': [
                'study design', 'statistical analysis', 'p-value',
                'effect size', 'confidence interval', 'validation',
                'reproducibility', 'quality control', 'standardization'
            ]
        }
        
        # Response length and structure assessment configuration
        self.length_criteria = {
            'basic_definition': {'min': 50, 'optimal_min': 100, 'optimal_max': 400, 'max': 800},
            'clinical_application': {'min': 80, 'optimal_min': 150, 'optimal_max': 600, 'max': 1200},
            'analytical_method': {'min': 100, 'optimal_min': 200, 'optimal_max': 800, 'max': 1500},
            'research_design': {'min': 120, 'optimal_min': 250, 'optimal_max': 1000, 'max': 2000},
            'disease_specific': {'min': 80, 'optimal_min': 150, 'optimal_max': 700, 'max': 1400},
            'general': {'min': 60, 'optimal_min': 120, 'optimal_max': 500, 'max': 1000}
        }
        
        # Structure quality indicators
        self.structure_indicators = {
            'formatting': ['**', '*', '##', '-', '•', '1.', '2.', '3.'],
            'citations': ['(', ')', '[', ']', 'et al', 'study', 'research'],
            'sections': ['introduction', 'background', 'method', 'result', 'conclusion', 'summary'],
            'coherence': ['first', 'second', 'third', 'finally', 'however', 'therefore', 'moreover']
        }
        
        # Performance monitoring
        self._start_time = None
        self._performance_metrics = defaultdict(list)
        
        logger.info("ClinicalMetabolomicsRelevanceScorer initialized with enhanced structure quality checks")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enable_caching': True,
            'cache_ttl_seconds': 3600,
            'parallel_processing': True,
            'confidence_threshold': 70.0,
            'minimum_relevance_threshold': 50.0
        }
    
    async def calculate_relevance_score(self,
                                     query: str,
                                     response: str,
                                     metadata: Optional[Dict[str, Any]] = None) -> RelevanceScore:
        """
        Calculate comprehensive relevance score for clinical metabolomics response.
        
        Args:
            query: Original user query
            response: System response to evaluate
            metadata: Optional metadata about the query/response context
            
        Returns:
            RelevanceScore: Comprehensive scoring results
        """
        start_time = time.time()
        
        try:
            # Step 1: Classify query type
            query_type = self.query_classifier.classify_query(query)
            logger.debug(f"Classified query as: {query_type}")
            
            # Step 2: Get appropriate weighting scheme
            weights = self.weighting_manager.get_weights(query_type)
            
            # Step 3: Calculate dimension scores (including new quality dimensions)
            dimension_scores = await self._calculate_all_dimensions(query, response, metadata)
            
            # Step 4: Calculate weighted overall score
            overall_score = self._calculate_weighted_score(dimension_scores, weights)
            
            # Step 5: Calculate confidence score
            confidence_score = self._calculate_confidence(dimension_scores, weights)
            
            # Step 6: Generate explanation
            explanation = self._generate_explanation(dimension_scores, weights, query_type)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create result
            result = RelevanceScore(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                query_type=query_type,
                weights_used=weights,
                explanation=explanation,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                metadata={
                    'query_length': len(query),
                    'response_length': len(response),
                    'word_count': len(response.split()),
                    'biomedical_terms_found': self._count_biomedical_terms(response)
                }
            )
            
            logger.debug(f"Relevance scoring completed in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            # Return fallback score
            return RelevanceScore(
                overall_score=0.0,
                explanation=f"Error during scoring: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _calculate_all_dimensions(self, query: str, response: str, metadata: Optional[Dict]) -> Dict[str, float]:
        """Calculate all relevance dimensions efficiently."""
        if self.config.get('parallel_processing', True):
            # Run dimension calculations concurrently
            tasks = [
                self._calculate_metabolomics_relevance(query, response),
                self._calculate_clinical_applicability(query, response),
                self._calculate_query_alignment(query, response),
                self._calculate_scientific_rigor(response),
                self._calculate_biomedical_context_depth(response),
                self._calculate_response_length_quality(query, response),
                self._calculate_response_structure_quality(response)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            dimension_names = [
                'metabolomics_relevance', 'clinical_applicability', 'query_alignment',
                'scientific_rigor', 'biomedical_context_depth',
                'response_length_quality', 'response_structure_quality'
            ]
            
            dimension_scores = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Error calculating {dimension_names[i]}: {result}")
                    dimension_scores[dimension_names[i]] = 0.0
                else:
                    dimension_scores[dimension_names[i]] = result
            
            return dimension_scores
        else:
            # Sequential calculation
            return {
                'metabolomics_relevance': await self._calculate_metabolomics_relevance(query, response),
                'clinical_applicability': await self._calculate_clinical_applicability(query, response),
                'query_alignment': await self._calculate_query_alignment(query, response),
                'scientific_rigor': await self._calculate_scientific_rigor(response),
                'biomedical_context_depth': await self._calculate_biomedical_context_depth(response),
                'response_length_quality': await self._calculate_response_length_quality(query, response),
                'response_structure_quality': await self._calculate_response_structure_quality(response)
            }
    
    async def _calculate_metabolomics_relevance(self, query: str, response: str) -> float:
        """
        Calculate metabolomics-specific relevance score.
        
        Assesses:
        - Analytical method relevance (30%)
        - Metabolite specificity (25%)
        - Research context (20%)
        - Technical accuracy (25%)
        """
        analytical_score = self._assess_analytical_methods(response)
        metabolite_score = self._assess_metabolite_coverage(query, response)
        research_score = self._assess_research_context(response)
        technical_score = await self._assess_technical_accuracy(response)
        
        weighted_score = (
            analytical_score * 0.30 +
            metabolite_score * 0.25 +
            research_score * 0.20 +
            technical_score * 0.25
        )
        
        return min(100.0, max(0.0, weighted_score))
    
    async def _calculate_clinical_applicability(self, query: str, response: str) -> float:
        """
        Calculate clinical applicability score.
        
        Assesses:
        - Disease relevance (35%)
        - Diagnostic utility (25%)
        - Therapeutic relevance (25%)
        - Clinical workflow (15%)
        """
        disease_score = self._assess_disease_relevance(response)
        diagnostic_score = self._assess_diagnostic_utility(response)
        therapeutic_score = self._assess_therapeutic_relevance(response)
        workflow_score = self._assess_clinical_workflow(response)
        
        weighted_score = (
            disease_score * 0.35 +
            diagnostic_score * 0.25 +
            therapeutic_score * 0.25 +
            workflow_score * 0.15
        )
        
        return min(100.0, max(0.0, weighted_score))
    
    async def _calculate_query_alignment(self, query: str, response: str) -> float:
        """
        Calculate query alignment score.
        
        Assesses:
        - Semantic similarity (40%)
        - Keyword overlap (25%)
        - Intent matching (20%)
        - Context preservation (15%)
        """
        semantic_score = await self.semantic_engine.calculate_similarity(query, response)
        keyword_score = self._assess_keyword_overlap(query, response)
        intent_score = self._assess_intent_matching(query, response)
        context_score = self._assess_context_preservation(query, response)
        
        weighted_score = (
            semantic_score * 0.40 +
            keyword_score * 0.25 +
            intent_score * 0.20 +
            context_score * 0.15
        )
        
        return min(100.0, max(0.0, weighted_score))
    
    async def _calculate_scientific_rigor(self, response: str) -> float:
        """
        Calculate scientific rigor score.
        
        Assesses:
        - Evidence quality (30%)
        - Statistical appropriateness (25%)
        - Methodological soundness (25%)
        - Uncertainty acknowledgment (20%)
        """
        evidence_score = self._assess_evidence_quality(response)
        statistical_score = self._assess_statistical_appropriateness(response)
        methodological_score = self._assess_methodological_soundness(response)
        uncertainty_score = self._assess_uncertainty_acknowledgment(response)
        
        weighted_score = (
            evidence_score * 0.30 +
            statistical_score * 0.25 +
            methodological_score * 0.25 +
            uncertainty_score * 0.20
        )
        
        return min(100.0, max(0.0, weighted_score))
    
    async def _calculate_biomedical_context_depth(self, response: str) -> float:
        """
        Calculate biomedical context depth score.
        
        Assesses:
        - Biological pathway integration (30%)
        - Physiological relevance (25%)
        - Multi-omics integration (25%)
        - Translational context (20%)
        """
        pathway_score = self._assess_pathway_integration(response)
        physiological_score = self._assess_physiological_relevance(response)
        omics_score = self._assess_multi_omics_integration(response)
        translational_score = self._assess_translational_context(response)
        
        weighted_score = (
            pathway_score * 0.30 +
            physiological_score * 0.25 +
            omics_score * 0.25 +
            translational_score * 0.20
        )
        
        return min(100.0, max(0.0, weighted_score))
    
    # Individual assessment methods
    
    def _assess_analytical_methods(self, response: str) -> float:
        """Assess analytical methods coverage."""
        analytical_terms = self.biomedical_keywords['analytical_methods']
        response_lower = response.lower()
        
        found_terms = sum(1 for term in analytical_terms if term in response_lower)
        max_terms = len(analytical_terms)
        
        base_score = (found_terms / max_terms) * 70 if max_terms > 0 else 0
        
        # Bonus for method-specific details
        detail_bonus = 0
        if 'sample preparation' in response_lower:
            detail_bonus += 10
        if 'quality control' in response_lower:
            detail_bonus += 10
        if 'validation' in response_lower:
            detail_bonus += 10
        
        return min(100.0, base_score + detail_bonus)
    
    def _assess_metabolite_coverage(self, query: str, response: str) -> float:
        """Assess metabolite-specific coverage."""
        metabolite_indicators = [
            'metabolite', 'compound', 'molecule', 'biomarker',
            'concentration', 'abundance', 'level', 'pathway'
        ]
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Check if query is metabolite-specific
        query_metabolite_focus = sum(1 for indicator in metabolite_indicators if indicator in query_lower)
        
        if query_metabolite_focus == 0:
            return 75.0  # Neutral score for non-metabolite queries
        
        response_coverage = sum(1 for indicator in metabolite_indicators if indicator in response_lower)
        
        coverage_score = (response_coverage / max(query_metabolite_focus, 1)) * 80
        
        # Bonus for specific metabolite names or pathways
        specific_bonus = 0
        if re.search(r'\b[A-Z][a-z]+-\d+', response):  # Metabolite naming pattern
            specific_bonus += 20
        
        return min(100.0, coverage_score + specific_bonus)
    
    def _assess_research_context(self, response: str) -> float:
        """Assess research context and methodology."""
        research_terms = self.biomedical_keywords['research_concepts']
        response_lower = response.lower()
        
        found_terms = sum(1 for term in research_terms if term in response_lower)
        base_score = min(80.0, found_terms * 15)
        
        # Bonus for comprehensive methodology discussion
        if 'study design' in response_lower and 'statistical' in response_lower:
            base_score += 20
        
        return min(100.0, base_score)
    
    async def _assess_technical_accuracy(self, response: str) -> float:
        """Assess technical accuracy and appropriate terminology."""
        return await self.domain_validator.validate_domain_expertise(response)
    
    def _assess_disease_relevance(self, response: str) -> float:
        """Assess disease-related relevance."""
        disease_terms = [
            'disease', 'disorder', 'syndrome', 'condition', 'pathology',
            'cancer', 'diabetes', 'cardiovascular', 'neurological',
            'metabolic disorder', 'biomarker', 'diagnostic'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in disease_terms if term in response_lower)
        
        return min(100.0, found_terms * 12 + 40)  # Base score of 40
    
    def _assess_diagnostic_utility(self, response: str) -> float:
        """Assess diagnostic utility discussion."""
        diagnostic_terms = [
            'diagnosis', 'diagnostic', 'biomarker', 'screening',
            'detection', 'sensitivity', 'specificity', 'accuracy',
            'ROC', 'AUC', 'predictive value'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in diagnostic_terms if term in response_lower)
        
        return min(100.0, found_terms * 10 + 30)
    
    def _assess_therapeutic_relevance(self, response: str) -> float:
        """Assess therapeutic relevance discussion."""
        therapeutic_terms = [
            'treatment', 'therapy', 'therapeutic', 'drug', 'medication',
            'intervention', 'monitoring', 'response', 'efficacy',
            'personalized medicine', 'precision medicine'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in therapeutic_terms if term in response_lower)
        
        return min(100.0, found_terms * 12 + 35)
    
    def _assess_clinical_workflow(self, response: str) -> float:
        """Assess clinical workflow integration."""
        workflow_terms = [
            'clinical practice', 'workflow', 'implementation',
            'healthcare', 'clinician', 'physician', 'routine',
            'standard of care', 'guidelines', 'protocol'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in workflow_terms if term in response_lower)
        
        return min(100.0, found_terms * 15 + 25)
    
    def _assess_keyword_overlap(self, query: str, response: str) -> float:
        """Assess keyword overlap between query and response."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Remove common stopwords
        stopwords = self.semantic_engine.stopwords
        query_words -= stopwords
        response_words -= stopwords
        
        if not query_words:
            return 50.0
        
        overlap = len(query_words.intersection(response_words))
        return min(100.0, (overlap / len(query_words)) * 100)
    
    def _assess_intent_matching(self, query: str, response: str) -> float:
        """Assess intent matching between query and response."""
        # Simple intent analysis based on question words and response structure
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        query_lower = query.lower()
        
        has_question_word = any(word in query_lower for word in question_words)
        
        if has_question_word:
            # Expect informative response
            response_length = len(response.split())
            if response_length < 20:
                return 40.0  # Too short for informative response
            elif response_length > 100:
                return 90.0  # Comprehensive response
            else:
                return 70.0  # Adequate response
        else:
            # Command or statement - different evaluation
            return 75.0
    
    def _assess_context_preservation(self, query: str, response: str) -> float:
        """Assess context preservation throughout response."""
        # Check if key concepts from query appear throughout response
        query_terms = re.findall(r'\b\w+\b', query.lower())
        important_terms = [term for term in query_terms if len(term) > 4]
        
        if not important_terms:
            return 75.0
        
        response_sentences = response.split('.')
        context_maintained = 0
        
        for term in important_terms[:3]:  # Check top 3 important terms
            sentences_with_term = sum(1 for sentence in response_sentences if term in sentence.lower())
            if sentences_with_term > 1:  # Term appears in multiple sentences
                context_maintained += 1
        
        return min(100.0, (context_maintained / min(len(important_terms), 3)) * 100)
    
    def _assess_evidence_quality(self, response: str) -> float:
        """Assess quality of evidence presented."""
        evidence_indicators = [
            'study', 'research', 'data', 'evidence', 'findings',
            'according to', 'demonstrated', 'showed', 'indicated',
            'meta-analysis', 'systematic review', 'clinical trial'
        ]
        
        response_lower = response.lower()
        found_indicators = sum(1 for indicator in evidence_indicators if indicator in response_lower)
        
        # Penalty for unsupported claims
        claim_words = ['always', 'never', 'all', 'none', 'completely', 'absolutely']
        unsupported_claims = sum(1 for claim in claim_words if claim in response_lower)
        
        base_score = min(80.0, found_indicators * 12)
        penalty = min(30.0, unsupported_claims * 10)
        
        return max(20.0, min(100.0, base_score - penalty + 20))
    
    def _assess_statistical_appropriateness(self, response: str) -> float:
        """Assess statistical appropriateness."""
        statistical_terms = [
            'p-value', 'significance', 'confidence interval', 'correlation',
            'regression', 'analysis', 'test', 'statistical', 'significant'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in statistical_terms if term in response_lower)
        
        return min(100.0, found_terms * 10 + 50)
    
    def _assess_methodological_soundness(self, response: str) -> float:
        """Assess methodological soundness."""
        methodology_terms = [
            'method', 'methodology', 'approach', 'procedure',
            'protocol', 'validation', 'reproducibility', 'standardization',
            'quality control', 'control group', 'randomized'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in methodology_terms if term in response_lower)
        
        return min(100.0, found_terms * 8 + 45)
    
    def _assess_uncertainty_acknowledgment(self, response: str) -> float:
        """Assess appropriate acknowledgment of uncertainty."""
        uncertainty_phrases = [
            'may', 'might', 'could', 'possibly', 'likely', 'potentially',
            'suggests', 'indicates', 'appears', 'seems', 'preliminary',
            'limited', 'further research', 'more studies needed'
        ]
        
        response_lower = response.lower()
        found_phrases = sum(1 for phrase in uncertainty_phrases if phrase in response_lower)
        
        # Balance - some uncertainty is good, too much is bad
        if found_phrases == 0:
            return 60.0  # No uncertainty acknowledgment
        elif found_phrases <= 3:
            return 85.0  # Appropriate uncertainty
        else:
            return 70.0  # Too much uncertainty
    
    def _assess_pathway_integration(self, response: str) -> float:
        """Assess biological pathway integration."""
        pathway_terms = [
            'pathway', 'network', 'cascade', 'regulation', 'signaling',
            'metabolic network', 'biochemical pathway', 'KEGG', 'reactome'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in pathway_terms if term in response_lower)
        
        return min(100.0, found_terms * 15 + 30)
    
    def _assess_physiological_relevance(self, response: str) -> float:
        """Assess physiological relevance."""
        physiological_terms = [
            'physiological', 'biological', 'cellular', 'molecular',
            'organ', 'tissue', 'system', 'function', 'mechanism',
            'homeostasis', 'regulation', 'metabolism'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in physiological_terms if term in response_lower)
        
        return min(100.0, found_terms * 10 + 40)
    
    def _assess_multi_omics_integration(self, response: str) -> float:
        """Assess multi-omics integration discussion."""
        omics_terms = [
            'omics', 'genomics', 'transcriptomics', 'proteomics',
            'metabolomics', 'multi-omics', 'integration', 'systems biology',
            'bioinformatics', 'data integration'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in omics_terms if term in response_lower)
        
        return min(100.0, found_terms * 12 + 35)
    
    def _assess_translational_context(self, response: str) -> float:
        """Assess translational context (bench-to-bedside)."""
        translational_terms = [
            'translational', 'clinical application', 'bench to bedside',
            'clinical implementation', 'real-world', 'practical',
            'clinical utility', 'clinical significance', 'patient care'
        ]
        
        response_lower = response.lower()
        found_terms = sum(1 for term in translational_terms if term in response_lower)
        
        return min(100.0, found_terms * 18 + 25)
    
    async def _calculate_response_length_quality(self, query: str, response: str) -> float:
        """
        Calculate response length quality score.
        
        Evaluates whether response length is appropriate for query complexity:
        - Too short responses (40% penalty)
        - Optimal length range (100% score)
        - Slightly over/under optimal (90% score)
        - Excessively long responses (60% score)
        
        Args:
            query: Original query
            response: Response to evaluate
            
        Returns:
            Length quality score (0-100)
        """
        query_type = self.query_classifier.classify_query(query)
        criteria = self.length_criteria.get(query_type, self.length_criteria['general'])
        
        word_count = len(response.split())
        
        # Calculate base score based on length appropriateness
        if word_count < criteria['min']:
            # Too short - significant penalty
            shortage_ratio = word_count / criteria['min']
            base_score = 30.0 + (shortage_ratio * 30.0)  # 30-60 range
        elif word_count >= criteria['optimal_min'] and word_count <= criteria['optimal_max']:
            # Optimal range - full score
            base_score = 95.0
        elif word_count < criteria['optimal_min']:
            # Slightly short - minor penalty
            ratio = (word_count - criteria['min']) / (criteria['optimal_min'] - criteria['min'])
            base_score = 70.0 + (ratio * 25.0)  # 70-95 range
        elif word_count <= criteria['max']:
            # Slightly long - minor penalty
            ratio = (criteria['max'] - word_count) / (criteria['max'] - criteria['optimal_max'])
            base_score = 70.0 + (ratio * 25.0)  # 70-95 range
        else:
            # Excessively long - moderate penalty
            excess_ratio = min(2.0, word_count / criteria['max'])  # Cap at 2x
            base_score = max(20.0, 60.0 - (excess_ratio - 1.0) * 40.0)  # 20-60 range
        
        # Adjust score based on query complexity indicators
        complexity_bonus = self._assess_query_complexity_bonus(query)
        
        # Assess response density (information per word)
        density_score = self._assess_response_density(response)
        
        final_score = base_score + complexity_bonus + density_score
        
        return min(100.0, max(0.0, final_score))
    
    def _assess_query_complexity_bonus(self, query: str) -> float:
        """
        Assess query complexity and provide length tolerance bonus.
        
        Complex queries may warrant longer responses.
        """
        complexity_indicators = [
            'explain', 'describe', 'compare', 'analyze', 'evaluate',
            'discuss', 'overview', 'comprehensive', 'detail', 'mechanism',
            'multiple', 'various', 'different', 'relationship', 'interaction'
        ]
        
        query_lower = query.lower()
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        
        # Multiple questions or parts
        question_marks = query.count('?')
        and_count = query_lower.count(' and ')
        or_count = query_lower.count(' or ')
        
        complexity_score = (
            complexity_count * 2 +
            max(0, question_marks - 1) * 3 +  # Multiple questions
            and_count * 1.5 +
            or_count * 1
        )
        
        return min(5.0, complexity_score)  # Up to 5 point bonus
    
    def _assess_response_density(self, response: str) -> float:
        """
        Assess information density of response.
        
        Higher density means more information per word, which can justify length.
        """
        words = response.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Count information-rich elements
        technical_terms = sum(1 for word in words if len(word) > 6)  # Longer technical terms
        numbers = len(re.findall(r'\d+', response))  # Numerical data
        citations = response.count('(') + response.count('[')  # Citation indicators
        
        # Calculate density score
        density_ratio = (technical_terms + numbers * 0.5 + citations * 0.3) / word_count
        
        # Convert to score (0-5 range)
        density_score = min(5.0, density_ratio * 50)
        
        return density_score
    
    async def _calculate_response_structure_quality(self, response: str) -> float:
        """
        Calculate response structure quality score.
        
        Evaluates:
        - Formatting quality (25%)
        - Logical organization (30%)
        - Coherence and flow (25%)
        - Readability (20%)
        
        Returns:
            Structure quality score (0-100)
        """
        formatting_score = self._assess_formatting_quality(response)
        organization_score = self._assess_logical_organization(response)
        coherence_score = self._assess_coherence_flow(response)
        readability_score = self._assess_readability(response)
        
        weighted_score = (
            formatting_score * 0.25 +
            organization_score * 0.30 +
            coherence_score * 0.25 +
            readability_score * 0.20
        )
        
        return min(100.0, max(0.0, weighted_score))
    
    def _assess_formatting_quality(self, response: str) -> float:
        """
        Assess formatting quality of response.
        
        Checks for:
        - Use of markdown formatting
        - Bullet points or numbered lists
        - Proper paragraph structure
        - Emphasis markers
        """
        formatting_score = 50.0  # Base score
        
        # Check for markdown formatting
        markdown_indicators = self.structure_indicators['formatting']
        found_formatting = sum(1 for indicator in markdown_indicators if indicator in response)
        
        # Bonus for appropriate formatting use
        if found_formatting > 0:
            formatting_score += min(20.0, found_formatting * 5.0)
        
        # Check paragraph structure (not just wall of text)
        paragraphs = response.split('\n\n')
        if len(paragraphs) > 1:
            formatting_score += 15.0
        
        # Check for list structures
        list_patterns = [r'\n\s*[-•*]\s+', r'\n\s*\d+\.\s+', r'\n\s*[a-zA-Z]\)\s+']
        has_lists = any(re.search(pattern, response) for pattern in list_patterns)
        if has_lists:
            formatting_score += 15.0
        
        return min(100.0, formatting_score)
    
    def _assess_logical_organization(self, response: str) -> float:
        """
        Assess logical organization of response content.
        
        Checks for:
        - Clear introduction/conclusion
        - Logical flow of ideas
        - Section headers or clear transitions
        - Information hierarchy
        """
        organization_score = 60.0  # Base score
        
        # Check for section indicators
        section_words = self.structure_indicators['sections']
        found_sections = sum(1 for word in section_words if word.lower() in response.lower())
        
        if found_sections > 0:
            organization_score += min(20.0, found_sections * 4.0)
        
        # Check for logical flow indicators
        flow_indicators = ['first', 'second', 'next', 'then', 'finally', 'in conclusion']
        found_flow = sum(1 for indicator in flow_indicators if indicator.lower() in response.lower())
        
        if found_flow > 0:
            organization_score += min(15.0, found_flow * 3.0)
        
        # Check for appropriate response structure (intro -> body -> conclusion pattern)
        sentences = response.split('.')
        if len(sentences) >= 3:
            # Simple heuristic: first sentence introduces, last sentence concludes
            first_sentence = sentences[0].lower()
            last_sentence = sentences[-1].lower()
            
            intro_words = ['is', 'are', 'refers', 'involves', 'includes']
            conclusion_words = ['therefore', 'thus', 'overall', 'in summary', 'important']
            
            has_intro = any(word in first_sentence for word in intro_words)
            has_conclusion = any(word in last_sentence for word in conclusion_words)
            
            if has_intro:
                organization_score += 5.0
            if has_conclusion:
                organization_score += 5.0
        
        return min(100.0, organization_score)
    
    def _assess_coherence_flow(self, response: str) -> float:
        """
        Assess coherence and flow of response.
        
        Checks for:
        - Transition words and phrases
        - Consistent terminology
        - Logical connections between ideas
        - Avoidance of contradictions
        """
        coherence_score = 55.0  # Base score
        
        # Check for transition indicators
        transition_words = self.structure_indicators['coherence']
        found_transitions = sum(1 for word in transition_words if word.lower() in response.lower())
        
        if found_transitions > 0:
            coherence_score += min(25.0, found_transitions * 4.0)
        
        # Check for consistent terminology (repeated key terms)
        words = re.findall(r'\b\w{4,}\b', response.lower())  # Words 4+ chars
        if words:
            word_freq = {}
            for word in words:
                if word not in self.semantic_engine.stopwords:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Bonus for consistent use of key terms (appearing 2+ times)
            consistent_terms = sum(1 for freq in word_freq.values() if freq >= 2)
            coherence_score += min(15.0, consistent_terms * 2.0)
        
        # Penalty for contradictory language
        contradictions = [
            ('always', 'never'), ('all', 'none'), ('completely', 'partially'),
            ('definitely', 'possibly'), ('certain', 'uncertain')
        ]
        
        response_lower = response.lower()
        contradiction_penalty = 0
        for word1, word2 in contradictions:
            if word1 in response_lower and word2 in response_lower:
                contradiction_penalty += 5.0
        
        coherence_score -= min(15.0, contradiction_penalty)
        
        return min(100.0, max(30.0, coherence_score))
    
    def _assess_readability(self, response: str) -> float:
        """
        Assess readability of response.
        
        Uses simplified metrics:
        - Average sentence length
        - Use of complex terminology (balanced)
        - Paragraph length variation
        - Clarity indicators
        """
        readability_score = 60.0  # Base score
        
        # Calculate average sentence length
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            # Optimal sentence length is 15-25 words
            if 15 <= avg_sentence_length <= 25:
                readability_score += 15.0
            elif 10 <= avg_sentence_length < 15 or 25 < avg_sentence_length <= 35:
                readability_score += 10.0
            elif avg_sentence_length < 10:
                readability_score += 5.0  # Too choppy
            else:
                readability_score -= 10.0  # Too complex
        
        # Check for clarity indicators
        clarity_phrases = [
            'for example', 'such as', 'in other words', 'specifically',
            'that is', 'namely', 'this means', 'put simply'
        ]
        
        found_clarity = sum(1 for phrase in clarity_phrases if phrase in response.lower())
        readability_score += min(15.0, found_clarity * 5.0)
        
        # Balance of technical vs. accessible language
        words = response.split()
        if words:
            long_words = sum(1 for word in words if len(word) > 8)
            long_word_ratio = long_words / len(words)
            
            # Optimal ratio is 5-15% long words for technical content
            if 0.05 <= long_word_ratio <= 0.15:
                readability_score += 10.0
            elif 0.15 < long_word_ratio <= 0.25:
                readability_score += 5.0
            else:
                readability_score -= 5.0
        
        return min(100.0, max(30.0, readability_score))
    
    def _assess_completeness(self, query: str, response: str) -> float:
        """
        Assess whether response fully addresses the query.
        
        This is a utility method that can be used for additional validation.
        
        Returns:
            Completeness score (0-100)
        """
        # Extract key concepts from query
        query_concepts = self._extract_key_concepts(query)
        
        if not query_concepts:
            return 75.0  # Neutral score for unclear queries
        
        # Check coverage of key concepts in response
        response_lower = response.lower()
        covered_concepts = sum(1 for concept in query_concepts if concept in response_lower)
        
        coverage_ratio = covered_concepts / len(query_concepts)
        base_completeness = coverage_ratio * 80  # Up to 80 points for coverage
        
        # Bonus for depth of coverage (multiple mentions)
        depth_bonus = 0
        for concept in query_concepts:
            mentions = response_lower.count(concept)
            if mentions > 1:
                depth_bonus += min(3.0, mentions)  # Up to 3 points per concept
        
        total_score = base_completeness + min(20.0, depth_bonus)
        
        return min(100.0, max(20.0, total_score))
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """
        Extract key concepts from query for completeness assessment.
        
        Returns:
            List of key concepts (normalized to lowercase)
        """
        # Remove question words and common terms
        question_words = {'what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are', 'can', 'does'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter for meaningful terms (>3 chars, not stopwords)
        key_concepts = [
            word for word in words 
            if len(word) > 3 and 
            word not in question_words and 
            word not in self.semantic_engine.stopwords
        ]
        
        # Also extract multi-word technical terms
        technical_phrases = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+\b', query.lower())  # hyphenated terms
        key_concepts.extend(technical_phrases)
        
        return list(set(key_concepts))  # Remove duplicates
    
    def validate_response_quality(self, query: str, response: str) -> Dict[str, Any]:
        """
        Comprehensive response quality validation.
        
        This method provides a complete quality assessment including:
        - Length appropriateness
        - Structure quality  
        - Completeness
        - Readability
        - Formatting
        
        Args:
            query: Original query
            response: Response to validate
            
        Returns:
            Dictionary with detailed quality assessment
        """
        # Get query type for context
        query_type = self.query_classifier.classify_query(query)
        
        # Length assessment
        length_assessment = self._get_length_assessment(query, response, query_type)
        
        # Structure assessment
        structure_assessment = {
            'formatting_quality': self._assess_formatting_quality(response),
            'organization_quality': self._assess_logical_organization(response),
            'coherence_quality': self._assess_coherence_flow(response),
            'readability_quality': self._assess_readability(response)
        }
        
        # Completeness assessment  
        completeness_score = self._assess_completeness(query, response)
        
        # Overall quality grade
        avg_score = (
            length_assessment['score'] + 
            sum(structure_assessment.values()) / len(structure_assessment) + 
            completeness_score
        ) / 3
        
        quality_grade = self._get_quality_grade(avg_score)
        
        return {
            'query_type': query_type,
            'length_assessment': length_assessment,
            'structure_assessment': structure_assessment,
            'completeness_score': completeness_score,
            'overall_quality_score': avg_score,
            'quality_grade': quality_grade,
            'recommendations': self._generate_quality_recommendations(
                length_assessment, structure_assessment, completeness_score
            )
        }
    
    def _get_length_assessment(self, query: str, response: str, query_type: str) -> Dict[str, Any]:
        """Get detailed length assessment."""
        criteria = self.length_criteria.get(query_type, self.length_criteria['general'])
        word_count = len(response.split())
        
        # Determine length category
        if word_count < criteria['min']:
            category = 'too_short'
            score = 40.0
            message = f"Response is too short ({word_count} words). Minimum recommended: {criteria['min']} words."
        elif word_count >= criteria['optimal_min'] and word_count <= criteria['optimal_max']:
            category = 'optimal'
            score = 95.0
            message = f"Response length is optimal ({word_count} words)."
        elif word_count < criteria['optimal_min']:
            category = 'slightly_short'
            score = 75.0
            message = f"Response could be more comprehensive ({word_count} words). Optimal range: {criteria['optimal_min']}-{criteria['optimal_max']} words."
        elif word_count <= criteria['max']:
            category = 'slightly_long'
            score = 75.0
            message = f"Response is somewhat verbose ({word_count} words). Consider condensing to {criteria['optimal_min']}-{criteria['optimal_max']} words."
        else:
            category = 'too_long'
            score = 50.0
            message = f"Response is excessively long ({word_count} words). Maximum recommended: {criteria['max']} words."
        
        return {
            'word_count': word_count,
            'category': category,
            'score': score,
            'message': message,
            'criteria': criteria
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_quality_recommendations(self, length_assess: Dict, structure_assess: Dict, completeness: float) -> List[str]:
        """Generate specific recommendations for improving response quality."""
        recommendations = []
        
        # Length recommendations
        if length_assess['category'] == 'too_short':
            recommendations.append("Expand response with more detailed explanations and examples")
        elif length_assess['category'] == 'too_long':
            recommendations.append("Condense response by removing redundant information")
        elif length_assess['category'] in ['slightly_short', 'slightly_long']:
            recommendations.append(f"Adjust length to optimal range: {length_assess['criteria']['optimal_min']}-{length_assess['criteria']['optimal_max']} words")
        
        # Structure recommendations
        if structure_assess['formatting_quality'] < 70:
            recommendations.append("Improve formatting with bullet points, headers, or emphasis markers")
        
        if structure_assess['organization_quality'] < 70:
            recommendations.append("Enhance organization with clearer introduction, body, and conclusion structure")
        
        if structure_assess['coherence_quality'] < 70:
            recommendations.append("Add transition words and ensure consistent terminology throughout")
        
        if structure_assess['readability_quality'] < 70:
            recommendations.append("Improve readability with shorter sentences and clearer explanations")
        
        # Completeness recommendations  
        if completeness < 70:
            recommendations.append("Address all aspects of the query more thoroughly")
        
        if not recommendations:
            recommendations.append("Response quality is excellent - maintain current standards")
        
        return recommendations
    
    def _calculate_weighted_score(self, dimension_scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted overall score from dimension scores."""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in weights.items():
            if dimension in dimension_scores:
                total_score += dimension_scores[dimension] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def _calculate_confidence(self, dimension_scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate confidence score based on dimension consistency."""
        if not dimension_scores:
            return 0.0
        
        scores = list(dimension_scores.values())
        
        if len(scores) < 2:
            return 50.0  # Low confidence with limited data
        
        # Calculate variance - lower variance means higher confidence
        score_variance = statistics.variance(scores)
        
        # Normalize variance to confidence score
        # High variance (>400) -> Low confidence (0-40)
        # Medium variance (100-400) -> Medium confidence (40-70)
        # Low variance (0-100) -> High confidence (70-100)
        
        if score_variance > 400:
            confidence = max(0, 40 - (score_variance - 400) / 20)
        elif score_variance > 100:
            confidence = 40 + ((400 - score_variance) / 300) * 30
        else:
            confidence = 70 + ((100 - score_variance) / 100) * 30
        
        return min(100.0, max(0.0, confidence))
    
    def _generate_explanation(self, dimension_scores: Dict[str, float], weights: Dict[str, float], query_type: str) -> str:
        """Generate human-readable explanation of the scoring."""
        explanation_parts = [
            f"Query classified as: {query_type.replace('_', ' ').title()}"
        ]
        
        # Sort dimensions by their weighted contribution
        weighted_contributions = [
            (dim, score * weights.get(dim, 0), weights.get(dim, 0))
            for dim, score in dimension_scores.items()
        ]
        weighted_contributions.sort(key=lambda x: x[1], reverse=True)
        
        explanation_parts.append("\nDimension Scores (weighted contribution):")
        
        for dimension, weighted_score, weight in weighted_contributions:
            dimension_name = dimension.replace('_', ' ').title()
            raw_score = dimension_scores[dimension]
            explanation_parts.append(
                f"• {dimension_name}: {raw_score:.1f}/100 (weight: {weight:.2f}, contribution: {weighted_score:.1f})"
            )
        
        # Add insights based on scores
        insights = []
        for dimension, score in dimension_scores.items():
            if score >= 90:
                insights.append(f"Excellent {dimension.replace('_', ' ')}")
            elif score < 60:
                insights.append(f"Low {dimension.replace('_', ' ')}")
        
        if insights:
            explanation_parts.append(f"\nKey Insights: {', '.join(insights)}")
        
        return '\n'.join(explanation_parts)
    
    def _count_biomedical_terms(self, response: str) -> int:
        """Count biomedical terms found in response."""
        response_lower = response.lower()
        total_terms = 0
        
        for category_terms in self.biomedical_keywords.values():
            total_terms += sum(1 for term in category_terms if term in response_lower)
        
        return total_terms


# Utility functions for integration and testing

async def quick_relevance_check(query: str, response: str) -> float:
    """Quick relevance check for testing purposes."""
    scorer = ClinicalMetabolomicsRelevanceScorer()
    result = await scorer.calculate_relevance_score(query, response)
    return result.overall_score


async def batch_relevance_scoring(query_response_pairs: List[Tuple[str, str]]) -> List[RelevanceScore]:
    """Score multiple query-response pairs in batch."""
    scorer = ClinicalMetabolomicsRelevanceScorer()
    
    tasks = [
        scorer.calculate_relevance_score(query, response)
        for query, response in query_response_pairs
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = [result for result in results if isinstance(result, RelevanceScore)]
    
    return valid_results


if __name__ == "__main__":
    # Example usage and demonstrations
    async def demo():
        scorer = ClinicalMetabolomicsRelevanceScorer()
        
        # Example 1: Well-structured response
        print("=== EXAMPLE 1: Well-structured Response ===")
        query1 = "What is metabolomics and how is it used in clinical applications?"
        response1 = """# Metabolomics in Clinical Applications

## Definition
Metabolomics is the comprehensive study of small molecules called metabolites in biological systems. This field focuses on analyzing the complete set of metabolites present in cells, tissues, or biological fluids.

## Clinical Applications

### Biomarker Discovery
- Identification of disease-specific metabolic signatures
- Early detection of pathological conditions
- Monitoring disease progression

### Diagnostic Applications
- Non-invasive diagnostic tests using blood, urine, or tissue samples
- Improved sensitivity and specificity compared to traditional markers
- Personalized medicine approaches

### Treatment Monitoring
- Assessment of drug efficacy and toxicity
- Real-time monitoring of therapeutic responses
- Optimization of treatment protocols

## Analytical Methods
The most commonly used analytical platforms include:
- **LC-MS (Liquid Chromatography-Mass Spectrometry)**: Ideal for polar metabolites
- **GC-MS (Gas Chromatography-Mass Spectrometry)**: Suitable for volatile compounds
- **NMR (Nuclear Magnetic Resonance)**: Provides structural information

## Challenges and Future Directions
Current challenges include standardization of protocols, quality control, and data integration. However, advances in analytical technology and bioinformatics are addressing these limitations, making metabolomics increasingly valuable for precision medicine."""
        
        result1 = await scorer.calculate_relevance_score(query1, response1)
        
        print(f"Overall Relevance Score: {result1.overall_score:.2f}/100")
        print(f"Query Type: {result1.query_type}")
        print(f"Relevance Grade: {result1.relevance_grade}")
        print(f"Processing Time: {result1.processing_time_ms:.2f}ms")
        print("\nDimension Scores:")
        for dimension, score in result1.dimension_scores.items():
            print(f"  {dimension.replace('_', ' ').title()}: {score:.2f}/100")
        
        # Quality validation
        quality_assessment = scorer.validate_response_quality(query1, response1)
        print(f"\nQuality Assessment:")
        print(f"  Length Category: {quality_assessment['length_assessment']['category']}")
        print(f"  Word Count: {quality_assessment['length_assessment']['word_count']}")
        print(f"  Overall Quality Grade: {quality_assessment['quality_grade']}")
        
        print("\n" + "="*50)
        
        # Example 2: Poor quality response
        print("\n=== EXAMPLE 2: Poor Quality Response ===")
        query2 = "Explain the role of LC-MS in metabolomics research and clinical applications."
        response2 = "LC-MS is good for metabolomics. It works well and gives results."
        
        result2 = await scorer.calculate_relevance_score(query2, response2)
        
        print(f"Overall Relevance Score: {result2.overall_score:.2f}/100")
        print(f"Relevance Grade: {result2.relevance_grade}")
        
        quality_assessment2 = scorer.validate_response_quality(query2, response2)
        print(f"\nQuality Assessment:")
        print(f"  Length Category: {quality_assessment2['length_assessment']['category']}")
        print(f"  Quality Grade: {quality_assessment2['quality_grade']}")
        print(f"  Message: {quality_assessment2['length_assessment']['message']}")
        print("\nRecommendations:")
        for rec in quality_assessment2['recommendations']:
            print(f"  - {rec}")
        
        print("\n" + "="*50)
        
        # Example 3: Batch scoring
        print("\n=== EXAMPLE 3: Batch Scoring ===")
        test_pairs = [
            ("What are biomarkers?", "Biomarkers are measurable biological indicators of disease states."),
            ("How does GC-MS work?", "GC-MS separates compounds using gas chromatography and then identifies them using mass spectrometry."),
            ("Define precision medicine", "Precision medicine uses individual patient data to customize treatment.")
        ]
        
        batch_results = await batch_relevance_scoring(test_pairs)
        
        print(f"Processed {len(batch_results)} query-response pairs:")
        for i, result in enumerate(batch_results, 1):
            print(f"  Pair {i}: {result.overall_score:.1f}/100 ({result.relevance_grade})")
    
    # Run comprehensive demo
    asyncio.run(demo())