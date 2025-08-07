#!/usr/bin/env python3
"""
Biomedical Factual Claim Extraction System for Clinical Metabolomics Oracle.

This module provides the BiomedicalClaimExtractor class for parsing LightRAG responses
and extracting verifiable factual claims for accuracy validation against source documents
in the Clinical Metabolomics Oracle LightRAG integration project.

Classes:
    - ClaimExtractionError: Base custom exception for claim extraction errors
    - ClaimProcessingError: Exception for claim processing failures
    - ClaimValidationError: Exception for claim validation failures
    - ExtractedClaim: Data class for structured claim representation
    - ClaimContext: Data class for claim context information
    - ClaimConfidence: Data class for confidence assessment
    - BiomedicalClaimExtractor: Main class for factual claim extraction

The extractor handles:
    - Parsing LightRAG responses and identifying factual claims
    - Classifying claims by type (numeric, qualitative, methodological, etc.)
    - Extracting context information for each claim
    - Providing structured claim data for verification
    - Integration with existing quality assessment pipeline
    - Async support for performance optimization

Key Features:
    - Specialized biomedical terminology patterns
    - Multiple claim classification types
    - Confidence scoring for extracted claims
    - Context preservation for verification
    - Integration with existing document indexing systems
    - Comprehensive error handling and logging
    - High-performance async processing
    - Duplicate detection and merging
    - Claim priority scoring system

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Factual Claim Extraction Implementation
"""

import asyncio
import json
import logging
import re
import hashlib
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable, Pattern
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, Counter
from contextlib import asynccontextmanager
import math

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

# Configure logging
logger = logging.getLogger(__name__)


class ClaimExtractionError(Exception):
    """Base custom exception for claim extraction errors."""
    pass


class ClaimProcessingError(ClaimExtractionError):
    """Exception raised when claim processing fails."""
    pass


class ClaimValidationError(ClaimExtractionError):
    """Exception raised when claim validation fails."""
    pass


@dataclass
class ClaimContext:
    """
    Context information for extracted claims.
    
    Attributes:
        surrounding_text: Text context surrounding the claim
        sentence_position: Position of claim sentence in response
        paragraph_position: Position of claim paragraph in response
        section_type: Type of section containing the claim
        preceding_context: Text immediately before the claim
        following_context: Text immediately after the claim
        semantic_context: Semantic context indicators
        relevance_indicators: Indicators of claim relevance
    """
    surrounding_text: str = ""
    sentence_position: int = 0
    paragraph_position: int = 0
    section_type: str = "general"
    preceding_context: str = ""
    following_context: str = ""
    semantic_context: List[str] = field(default_factory=list)
    relevance_indicators: List[str] = field(default_factory=list)


@dataclass
class ClaimConfidence:
    """
    Confidence assessment for extracted claims.
    
    Attributes:
        overall_confidence: Overall confidence score (0-100)
        linguistic_confidence: Confidence based on linguistic patterns
        contextual_confidence: Confidence based on contextual clues
        domain_confidence: Confidence based on domain-specific patterns
        specificity_confidence: Confidence based on claim specificity
        verification_confidence: Confidence in claim verifiability
        factors: Factors contributing to confidence assessment
        uncertainty_indicators: Indicators of uncertainty in the claim
    """
    overall_confidence: float = 0.0
    linguistic_confidence: float = 0.0
    contextual_confidence: float = 0.0
    domain_confidence: float = 0.0
    specificity_confidence: float = 0.0
    verification_confidence: float = 0.0
    factors: List[str] = field(default_factory=list)
    uncertainty_indicators: List[str] = field(default_factory=list)


@dataclass
class ExtractedClaim:
    """
    Structured representation of extracted factual claims.
    
    Attributes:
        claim_id: Unique identifier for the claim
        claim_text: Original text of the claim
        claim_type: Type classification of the claim
        subject: Main subject of the claim
        predicate: Action or relationship in the claim
        object_value: Object or value of the claim
        numeric_values: Extracted numeric values
        units: Associated units for numeric values
        qualifiers: Qualifying terms or conditions
        context: Context information
        confidence: Confidence assessment
        source_sentence: Original sentence containing the claim
        normalized_text: Normalized version of claim text
        keywords: Key terms extracted from the claim
        relationships: Relationships identified in the claim
        metadata: Additional metadata
        extraction_timestamp: When the claim was extracted
        verification_status: Status of claim verification
    """
    claim_id: str
    claim_text: str
    claim_type: str
    subject: str = ""
    predicate: str = ""
    object_value: str = ""
    numeric_values: List[float] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    qualifiers: List[str] = field(default_factory=list)
    context: ClaimContext = field(default_factory=ClaimContext)
    confidence: ClaimConfidence = field(default_factory=ClaimConfidence)
    source_sentence: str = ""
    normalized_text: str = ""
    keywords: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    verification_status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to dictionary representation."""
        result = asdict(self)
        result['extraction_timestamp'] = self.extraction_timestamp.isoformat()
        return result
    
    @property
    def priority_score(self) -> float:
        """Calculate priority score for claim verification."""
        base_score = self.confidence.overall_confidence
        
        # Boost score for numeric claims
        if self.claim_type in ['numeric', 'statistical', 'measurement']:
            base_score *= 1.2
        
        # Boost score for claims with specific units
        if self.units:
            base_score *= 1.1
        
        # Boost score for claims with multiple numeric values
        if len(self.numeric_values) > 1:
            base_score *= 1.15
        
        # Reduce score for highly qualified claims (indicating uncertainty)
        qualifier_penalty = max(0, 1.0 - (len(self.qualifiers) * 0.1))
        base_score *= qualifier_penalty
        
        return min(100.0, base_score)


class BiomedicalClaimExtractor:
    """
    Main class for extracting factual claims from biomedical LightRAG responses.
    
    Provides comprehensive claim extraction capabilities including:
    - Multi-type claim classification
    - Context-aware extraction
    - Confidence scoring
    - Biomedical specialization
    - Integration with quality assessment systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BiomedicalClaimExtractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logger
        
        # Initialize extraction patterns
        self._initialize_extraction_patterns()
        
        # Initialize biomedical terminology
        self._initialize_biomedical_terms()
        
        # Initialize confidence assessment
        self._initialize_confidence_factors()
        
        # Performance tracking
        self.extraction_stats = defaultdict(int)
        self.processing_times = []
        
        self.logger.info("BiomedicalClaimExtractor initialized successfully")
    
    def _initialize_extraction_patterns(self):
        """Initialize regex patterns for claim extraction."""
        
        # Numeric claim patterns
        self.numeric_patterns = {
            'percentage': re.compile(
                r'(?i)(?:approximately|about|roughly|around|\~)?\s*'
                r'(\d+(?:\.\d+)?)\s*(?:%|percent|percentage)',
                re.IGNORECASE
            ),
            'measurement': re.compile(
                r'(?i)(\d+(?:\.\d+)?)\s*(?:mg|g|kg|ml|l|μl|μg|nm|μm|mm|cm|m|'
                r'mol|mmol|μmol|nmol|ppm|ppb|°c|°f|k|hz|khz|mhz|ghz|'
                r'min|hr|h|sec|s|day|days|week|weeks|month|months|year|years)',
                re.IGNORECASE
            ),
            'statistical': re.compile(
                r'(?i)(?:p-value|p\s*[=<>]\s*|significance\s*[=<>]\s*|'
                r'correlation\s*[=<>]\s*|r\s*[=<>]\s*|'
                r'confidence\s+interval|ci\s*[=<>]\s*)'
                r'(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)',
                re.IGNORECASE
            ),
            'range': re.compile(
                r'(?i)(?:between|from|range\s+of)\s*'
                r'(\d+(?:\.\d+)?)\s*(?:to|and|-|–|—)\s*'
                r'(\d+(?:\.\d+)?)',
                re.IGNORECASE
            ),
            'concentration': re.compile(
                r'(?i)(\d+(?:\.\d+)?)\s*(?:mg/ml|μg/ml|ng/ml|g/l|mg/l|μg/l|'
                r'mmol/l|μmol/l|nmol/l|m|mm|μm|nm)',
                re.IGNORECASE
            )
        }
        
        # Qualitative claim patterns
        self.qualitative_patterns = {
            'causation': re.compile(
                r'(?i)\b(?:causes?|leads?\s+to|results?\s+in|triggers?|'
                r'induces?|produces?|generates?|creates?|brings?\s+about)\b',
                re.IGNORECASE
            ),
            'correlation': re.compile(
                r'(?i)\b(?:correlates?\s+with|associated\s+with|linked\s+to|'
                r'related\s+to|connected\s+to|corresponds?\s+to)\b',
                re.IGNORECASE
            ),
            'comparison': re.compile(
                r'(?i)\b(?:higher\s+than|lower\s+than|greater\s+than|'
                r'less\s+than|compared\s+to|versus|vs\.?|relative\s+to|'
                r'in\s+contrast\s+to|differs?\s+from)\b',
                re.IGNORECASE
            ),
            'temporal': re.compile(
                r'(?i)\b(?:before|after|during|while|when|since|until|'
                r'following|preceding|simultaneously|concurrently)\b',
                re.IGNORECASE
            ),
            'conditional': re.compile(
                r'(?i)\b(?:if|when|unless|provided\s+that|given\s+that|'
                r'in\s+case|depending\s+on|contingent\s+on)\b',
                re.IGNORECASE
            )
        }
        
        # Methodological claim patterns
        self.methodological_patterns = {
            'analytical_method': re.compile(
                r'(?i)\b(?:LC-MS|GC-MS|UPLC|HPLC|NMR|MS/MS|QTOF|'
                r'mass\s+spectrometry|chromatography|spectroscopy)\b',
                re.IGNORECASE
            ),
            'study_design': re.compile(
                r'(?i)\b(?:randomized\s+controlled\s+trial|RCT|'
                r'case-control\s+study|cohort\s+study|cross-sectional|'
                r'longitudinal|prospective|retrospective|'
                r'double-blind|single-blind|placebo-controlled)\b',
                re.IGNORECASE
            ),
            'sample_processing': re.compile(
                r'(?i)\b(?:extraction|derivatization|protein\s+precipitation|'
                r'solid\s+phase\s+extraction|SPE|liquid-liquid\s+extraction|'
                r'centrifugation|filtration|dilution)\b',
                re.IGNORECASE
            ),
            'statistical_method': re.compile(
                r'(?i)\b(?:t-test|ANOVA|regression|PCA|PLS-DA|'
                r'principal\s+component\s+analysis|partial\s+least\s+squares|'
                r'multivariate\s+analysis|univariate\s+analysis)\b',
                re.IGNORECASE
            )
        }
        
        # Temporal claim patterns
        self.temporal_patterns = {
            'duration': re.compile(
                r'(?i)(?:for|during|over|within|after)\s*'
                r'(\d+)\s*(?:minutes?|hours?|days?|weeks?|months?|years?)',
                re.IGNORECASE
            ),
            'frequency': re.compile(
                r'(?i)(?:every|each|per|once|twice|thrice|\d+\s+times)\s*'
                r'(?:per|/)?\s*(?:day|week|month|year|hour|minute)',
                re.IGNORECASE
            ),
            'sequence': re.compile(
                r'(?i)\b(?:first|second|third|initially|subsequently|'
                r'finally|then|next|afterwards|previously)\b',
                re.IGNORECASE
            )
        }
        
        # Comparative claim patterns
        self.comparative_patterns = {
            'increase_decrease': re.compile(
                r'(?i)\b(?:increased?|decreased?|elevated?|reduced?|'
                r'upregulated?|downregulated?|enhanced?|diminished?)\s+'
                r'(?:by|to)?\s*(\d+(?:\.\d+)?)\s*(?:fold|times|%|percent)?',
                re.IGNORECASE
            ),
            'fold_change': re.compile(
                r'(?i)(\d+(?:\.\d+)?)\s*(?:-|\s*)?fold\s*(?:increase|decrease|'
                r'change|higher|lower|up|down)',
                re.IGNORECASE
            ),
            'significance_level': re.compile(
                r'(?i)\b(?:significantly|statistically\s+significant|'
                r'non-significantly?|marginally\s+significant)\b',
                re.IGNORECASE
            )
        }
        
        # Uncertainty and qualifier patterns
        self.uncertainty_patterns = {
            'hedging': re.compile(
                r'(?i)\b(?:may|might|could|possibly|potentially|likely|'
                r'probably|perhaps|appears?\s+to|seems?\s+to|'
                r'suggests?|indicates?|implies?)\b',
                re.IGNORECASE
            ),
            'approximation': re.compile(
                r'(?i)\b(?:approximately|roughly|about|around|nearly|'
                r'close\s+to|in\s+the\s+range\s+of|on\s+the\s+order\s+of)\b',
                re.IGNORECASE
            ),
            'conditionality': re.compile(
                r'(?i)\b(?:under\s+certain\s+conditions|in\s+some\s+cases|'
                r'depending\s+on|subject\s+to|provided\s+that)\b',
                re.IGNORECASE
            )
        }
    
    def _initialize_biomedical_terms(self):
        """Initialize biomedical terminology and concepts."""
        
        self.biomedical_terms = {
            'metabolomics_core': {
                'metabolomics', 'metabolite', 'metabolome', 'metabonomics',
                'small molecule', 'endogenous', 'exogenous', 'metabolic profile',
                'metabolic signature', 'metabolic fingerprint', 'metabolic pathway',
                'metabolic network', 'flux analysis', 'isotope labeling'
            },
            'analytical_techniques': {
                'mass spectrometry', 'MS', 'LC-MS', 'GC-MS', 'UPLC-MS',
                'QTOF', 'QQQ', 'orbitrap', 'ion trap', 'NMR', 'nuclear magnetic resonance',
                'chromatography', 'liquid chromatography', 'gas chromatography',
                'HILIC', 'reverse phase', 'ion exchange', 'size exclusion'
            },
            'clinical_contexts': {
                'biomarker', 'diagnostic', 'prognostic', 'therapeutic',
                'precision medicine', 'personalized medicine', 'pharmacogenomics',
                'drug metabolism', 'toxicology', 'adverse drug reaction',
                'disease progression', 'therapeutic monitoring', 'clinical trial'
            },
            'biological_systems': {
                'plasma', 'serum', 'urine', 'saliva', 'cerebrospinal fluid',
                'tissue', 'cell culture', 'mitochondria', 'cytoplasm',
                'membrane', 'organelle', 'enzyme', 'protein', 'gene expression'
            },
            'pathological_conditions': {
                'diabetes', 'cancer', 'cardiovascular disease', 'neurological disorder',
                'inflammatory disease', 'autoimmune', 'metabolic syndrome',
                'obesity', 'hypertension', 'alzheimer', 'parkinson'
            },
            'statistical_concepts': {
                'p-value', 'false discovery rate', 'FDR', 'multiple testing correction',
                'principal component analysis', 'PCA', 'partial least squares',
                'PLS-DA', 'OPLS-DA', 'multivariate analysis', 'univariate analysis',
                'fold change', 'effect size', 'confidence interval', 'statistical power'
            }
        }
        
        # Flatten all terms for quick lookup
        self.all_biomedical_terms = set()
        for category in self.biomedical_terms.values():
            self.all_biomedical_terms.update(category)
    
    def _initialize_confidence_factors(self):
        """Initialize factors for confidence assessment."""
        
        self.confidence_factors = {
            'linguistic': {
                'definitive_language': {
                    'patterns': [r'\bis\b', r'\bare\b', r'\bwere\b', r'\bwas\b'],
                    'boost': 10
                },
                'tentative_language': {
                    'patterns': [r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bpossibly\b'],
                    'penalty': -15
                },
                'quantified_statements': {
                    'patterns': [r'\d+(?:\.\d+)?', r'\b(?:all|most|many|some|few)\b'],
                    'boost': 5
                },
                'hedging': {
                    'patterns': [r'\bapproximately\b', r'\broughly\b', r'\babout\b'],
                    'penalty': -5
                }
            },
            'contextual': {
                'source_attribution': {
                    'patterns': [r'\baccording to\b', r'\bas reported\b', r'\bstudies show\b'],
                    'boost': 15
                },
                'methodological_context': {
                    'patterns': [r'\busing\b', r'\bvia\b', r'\bthrough\b', r'\bby means of\b'],
                    'boost': 8
                },
                'temporal_specificity': {
                    'patterns': [r'\b\d{4}\b', r'\brecent\b', r'\bcurrent\b'],
                    'boost': 5
                }
            },
            'domain': {
                'biomedical_terminology': {
                    'boost_per_term': 2,
                    'max_boost': 20
                },
                'technical_precision': {
                    'patterns': [r'\b[A-Z]{2,}-[A-Z]{2,}\b', r'\b\d+\.\d+\b'],
                    'boost': 8
                }
            },
            'specificity': {
                'numeric_precision': {
                    'patterns': [r'\d+\.\d{2,}', r'\d+(?:\.\d+)?\s*[±]\s*\d+'],
                    'boost': 12
                },
                'unit_specification': {
                    'boost_per_unit': 3,
                    'max_boost': 15
                },
                'range_specification': {
                    'patterns': [r'\d+(?:\.\d+)?\s*(?:to|-|–)\s*\d+(?:\.\d+)?'],
                    'boost': 8
                }
            }
        }
    
    @performance_logged("Extract claims from response")
    async def extract_claims(
        self,
        response_text: str,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractedClaim]:
        """
        Extract factual claims from a LightRAG response.
        
        Args:
            response_text: The response text to analyze
            query: Optional original query for context
            context: Optional additional context information
            
        Returns:
            List of extracted claims with full context and confidence scores
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting claim extraction from response of length {len(response_text)}")
            
            # Preprocess the response text
            preprocessed_text = await self._preprocess_text(response_text)
            
            # Split into sentences for analysis
            sentences = await self._split_into_sentences(preprocessed_text)
            
            # Extract claims from each sentence
            all_claims = []
            for i, sentence in enumerate(sentences):
                sentence_claims = await self._extract_claims_from_sentence(
                    sentence, i, preprocessed_text, context
                )
                all_claims.extend(sentence_claims)
            
            # Post-process claims
            processed_claims = await self._post_process_claims(all_claims, query, context)
            
            # Remove duplicates and merge similar claims
            final_claims = await self._deduplicate_and_merge_claims(processed_claims)
            
            # Calculate priority scores
            await self._calculate_priority_scores(final_claims)
            
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.extraction_stats['total_extractions'] += 1
            self.extraction_stats['total_claims_extracted'] += len(final_claims)
            
            self.logger.info(
                f"Extracted {len(final_claims)} claims in {processing_time:.2f}ms"
            )
            
            return final_claims
            
        except Exception as e:
            self.logger.error(f"Error in claim extraction: {str(e)}")
            raise ClaimExtractionError(f"Failed to extract claims: {str(e)}") from e
    
    async def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better claim extraction."""
        
        # Clean up whitespace and line breaks
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Handle common abbreviations
        abbreviations = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'and so on',
            'vs.': 'versus',
            'cf.': 'compare'
        }
        
        for abbrev, expansion in abbreviations.items():
            text = text.replace(abbrev, expansion)
        
        return text
    
    async def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for individual analysis."""
        
        # Simple sentence splitting - can be enhanced with more sophisticated NLP
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    async def _extract_claims_from_sentence(
        self,
        sentence: str,
        sentence_index: int,
        full_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractedClaim]:
        """Extract claims from a single sentence."""
        
        claims = []
        
        # Try different extraction strategies
        strategies = [
            ('numeric', self._extract_numeric_claims),
            ('qualitative', self._extract_qualitative_claims),
            ('methodological', self._extract_methodological_claims),
            ('temporal', self._extract_temporal_claims),
            ('comparative', self._extract_comparative_claims)
        ]
        
        for claim_type, extraction_method in strategies:
            try:
                type_claims = await extraction_method(
                    sentence, sentence_index, full_text, context
                )
                claims.extend(type_claims)
            except Exception as e:
                self.logger.warning(
                    f"Failed to extract {claim_type} claims from sentence: {str(e)}"
                )
        
        return claims
    
    async def _extract_numeric_claims(
        self,
        sentence: str,
        sentence_index: int,
        full_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractedClaim]:
        """Extract numeric claims from sentence."""
        
        claims = []
        
        for pattern_name, pattern in self.numeric_patterns.items():
            matches = pattern.finditer(sentence)
            
            for match in matches:
                # Create claim
                claim_text = match.group(0)
                claim_id = self._generate_claim_id(claim_text, sentence_index)
                
                # Extract numeric values
                numeric_values = []
                units = []
                
                # Extract all numbers from the match
                number_pattern = re.compile(r'\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
                numbers = number_pattern.findall(claim_text)
                numeric_values = [float(num) for num in numbers]
                
                # Extract units
                unit_pattern = re.compile(
                    r'\b(?:mg|g|kg|ml|l|μl|μg|nm|μm|mm|cm|m|mol|mmol|μmol|nmol|'
                    r'ppm|ppb|°c|°f|k|hz|khz|mhz|ghz|min|hr|h|sec|s|day|days|'
                    r'week|weeks|month|months|year|years|%|percent|fold|times)\b',
                    re.IGNORECASE
                )
                unit_matches = unit_pattern.findall(claim_text)
                units = list(set(unit_matches))
                
                # Create claim context
                claim_context = ClaimContext(
                    surrounding_text=sentence,
                    sentence_position=sentence_index,
                    section_type=pattern_name,
                    semantic_context=[pattern_name, 'numeric', 'quantitative']
                )
                
                # Assess confidence
                confidence = await self._assess_claim_confidence(
                    claim_text, sentence, 'numeric', context
                )
                
                # Create extracted claim
                claim = ExtractedClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    claim_type='numeric',
                    subject=self._extract_subject_from_sentence(sentence, match.start()),
                    predicate=self._extract_predicate_from_match(claim_text),
                    object_value=claim_text,
                    numeric_values=numeric_values,
                    units=units,
                    context=claim_context,
                    confidence=confidence,
                    source_sentence=sentence,
                    normalized_text=claim_text.lower(),
                    keywords=self._extract_keywords_from_text(sentence),
                    metadata={
                        'pattern_type': pattern_name,
                        'match_start': match.start(),
                        'match_end': match.end()
                    }
                )
                
                claims.append(claim)
        
        return claims
    
    async def _extract_qualitative_claims(
        self,
        sentence: str,
        sentence_index: int,
        full_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractedClaim]:
        """Extract qualitative relationship claims from sentence."""
        
        claims = []
        
        for pattern_name, pattern in self.qualitative_patterns.items():
            matches = pattern.finditer(sentence)
            
            for match in matches:
                claim_text = sentence  # Use full sentence for qualitative claims
                claim_id = self._generate_claim_id(claim_text, sentence_index)
                
                # Extract relationship components
                subject = self._extract_subject_from_sentence(sentence, match.start())
                predicate = match.group(0)
                object_value = self._extract_object_from_sentence(sentence, match.end())
                
                # Extract qualifiers
                qualifiers = self._extract_qualifiers_from_sentence(sentence)
                
                # Create claim context
                claim_context = ClaimContext(
                    surrounding_text=sentence,
                    sentence_position=sentence_index,
                    section_type=pattern_name,
                    semantic_context=[pattern_name, 'qualitative', 'relationship'],
                    relevance_indicators=[predicate]
                )
                
                # Assess confidence
                confidence = await self._assess_claim_confidence(
                    claim_text, sentence, 'qualitative', context
                )
                
                # Create relationship data
                relationships = [{
                    'type': pattern_name,
                    'subject': subject,
                    'predicate': predicate,
                    'object': object_value
                }]
                
                # Create extracted claim
                claim = ExtractedClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    claim_type='qualitative',
                    subject=subject,
                    predicate=predicate,
                    object_value=object_value,
                    qualifiers=qualifiers,
                    context=claim_context,
                    confidence=confidence,
                    source_sentence=sentence,
                    normalized_text=claim_text.lower(),
                    keywords=self._extract_keywords_from_text(sentence),
                    relationships=relationships,
                    metadata={
                        'relationship_type': pattern_name,
                        'match_start': match.start(),
                        'match_end': match.end()
                    }
                )
                
                claims.append(claim)
        
        return claims
    
    async def _extract_methodological_claims(
        self,
        sentence: str,
        sentence_index: int,
        full_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractedClaim]:
        """Extract methodological claims from sentence."""
        
        claims = []
        
        for pattern_name, pattern in self.methodological_patterns.items():
            matches = pattern.finditer(sentence)
            
            for match in matches:
                claim_text = sentence  # Use full sentence for methodological claims
                claim_id = self._generate_claim_id(claim_text, sentence_index)
                
                method_mentioned = match.group(0)
                
                # Create claim context
                claim_context = ClaimContext(
                    surrounding_text=sentence,
                    sentence_position=sentence_index,
                    section_type=pattern_name,
                    semantic_context=[pattern_name, 'methodological', 'technical'],
                    relevance_indicators=[method_mentioned]
                )
                
                # Assess confidence
                confidence = await self._assess_claim_confidence(
                    claim_text, sentence, 'methodological', context
                )
                
                # Create extracted claim
                claim = ExtractedClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    claim_type='methodological',
                    subject=method_mentioned,
                    predicate='method_used',
                    object_value=self._extract_object_from_sentence(sentence, match.end()),
                    context=claim_context,
                    confidence=confidence,
                    source_sentence=sentence,
                    normalized_text=claim_text.lower(),
                    keywords=self._extract_keywords_from_text(sentence),
                    metadata={
                        'method_type': pattern_name,
                        'method_name': method_mentioned,
                        'match_start': match.start(),
                        'match_end': match.end()
                    }
                )
                
                claims.append(claim)
        
        return claims
    
    async def _extract_temporal_claims(
        self,
        sentence: str,
        sentence_index: int,
        full_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractedClaim]:
        """Extract temporal claims from sentence."""
        
        claims = []
        
        for pattern_name, pattern in self.temporal_patterns.items():
            matches = pattern.finditer(sentence)
            
            for match in matches:
                claim_text = sentence
                claim_id = self._generate_claim_id(claim_text, sentence_index)
                
                temporal_expression = match.group(0)
                
                # Extract numeric values from temporal expressions
                numeric_values = []
                number_pattern = re.compile(r'\d+')
                numbers = number_pattern.findall(temporal_expression)
                if numbers:
                    numeric_values = [float(num) for num in numbers]
                
                # Create claim context
                claim_context = ClaimContext(
                    surrounding_text=sentence,
                    sentence_position=sentence_index,
                    section_type=pattern_name,
                    semantic_context=[pattern_name, 'temporal', 'chronological'],
                    relevance_indicators=[temporal_expression]
                )
                
                # Assess confidence
                confidence = await self._assess_claim_confidence(
                    claim_text, sentence, 'temporal', context
                )
                
                # Create extracted claim
                claim = ExtractedClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    claim_type='temporal',
                    subject=self._extract_subject_from_sentence(sentence, match.start()),
                    predicate=temporal_expression,
                    object_value=self._extract_object_from_sentence(sentence, match.end()),
                    numeric_values=numeric_values,
                    context=claim_context,
                    confidence=confidence,
                    source_sentence=sentence,
                    normalized_text=claim_text.lower(),
                    keywords=self._extract_keywords_from_text(sentence),
                    metadata={
                        'temporal_type': pattern_name,
                        'temporal_expression': temporal_expression,
                        'match_start': match.start(),
                        'match_end': match.end()
                    }
                )
                
                claims.append(claim)
        
        return claims
    
    async def _extract_comparative_claims(
        self,
        sentence: str,
        sentence_index: int,
        full_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractedClaim]:
        """Extract comparative claims from sentence."""
        
        claims = []
        
        for pattern_name, pattern in self.comparative_patterns.items():
            matches = pattern.finditer(sentence)
            
            for match in matches:
                claim_text = sentence
                claim_id = self._generate_claim_id(claim_text, sentence_index)
                
                comparative_expression = match.group(0)
                
                # Extract numeric values from comparative expressions
                numeric_values = []
                if match.groups():
                    for group in match.groups():
                        if group and re.match(r'\d+(?:\.\d+)?', group):
                            numeric_values.append(float(group))
                
                # Create claim context
                claim_context = ClaimContext(
                    surrounding_text=sentence,
                    sentence_position=sentence_index,
                    section_type=pattern_name,
                    semantic_context=[pattern_name, 'comparative', 'quantitative'],
                    relevance_indicators=[comparative_expression]
                )
                
                # Assess confidence
                confidence = await self._assess_claim_confidence(
                    claim_text, sentence, 'comparative', context
                )
                
                # Create extracted claim
                claim = ExtractedClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    claim_type='comparative',
                    subject=self._extract_subject_from_sentence(sentence, match.start()),
                    predicate=comparative_expression,
                    object_value=self._extract_object_from_sentence(sentence, match.end()),
                    numeric_values=numeric_values,
                    context=claim_context,
                    confidence=confidence,
                    source_sentence=sentence,
                    normalized_text=claim_text.lower(),
                    keywords=self._extract_keywords_from_text(sentence),
                    metadata={
                        'comparative_type': pattern_name,
                        'comparative_expression': comparative_expression,
                        'match_start': match.start(),
                        'match_end': match.end()
                    }
                )
                
                claims.append(claim)
        
        return claims
    
    async def _assess_claim_confidence(
        self,
        claim_text: str,
        sentence: str,
        claim_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ClaimConfidence:
        """Assess confidence in extracted claim."""
        
        confidence = ClaimConfidence()
        
        # Linguistic confidence assessment
        linguistic_score = 50.0  # Base score
        
        for factor_name, factor_data in self.confidence_factors['linguistic'].items():
            for pattern_str in factor_data['patterns']:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                if pattern.search(sentence):
                    if 'boost' in factor_data:
                        linguistic_score += factor_data['boost']
                        confidence.factors.append(f"linguistic_boost_{factor_name}")
                    elif 'penalty' in factor_data:
                        linguistic_score += factor_data['penalty']
                        confidence.uncertainty_indicators.append(factor_name)
        
        confidence.linguistic_confidence = max(0, min(100, linguistic_score))
        
        # Contextual confidence assessment
        contextual_score = 50.0
        
        for factor_name, factor_data in self.confidence_factors['contextual'].items():
            for pattern_str in factor_data['patterns']:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                if pattern.search(sentence):
                    contextual_score += factor_data['boost']
                    confidence.factors.append(f"contextual_boost_{factor_name}")
        
        confidence.contextual_confidence = max(0, min(100, contextual_score))
        
        # Domain confidence assessment
        domain_score = 50.0
        
        # Check for biomedical terminology
        biomedical_terms_found = 0
        for term in self.all_biomedical_terms:
            if term.lower() in sentence.lower():
                biomedical_terms_found += 1
        
        if biomedical_terms_found > 0:
            boost = min(
                biomedical_terms_found * self.confidence_factors['domain']['biomedical_terminology']['boost_per_term'],
                self.confidence_factors['domain']['biomedical_terminology']['max_boost']
            )
            domain_score += boost
            confidence.factors.append(f"biomedical_terms_{biomedical_terms_found}")
        
        # Check for technical precision
        for factor_name, factor_data in self.confidence_factors['domain'].items():
            if 'patterns' in factor_data:
                for pattern_str in factor_data['patterns']:
                    pattern = re.compile(pattern_str)
                    if pattern.search(claim_text):
                        domain_score += factor_data['boost']
                        confidence.factors.append(f"domain_boost_{factor_name}")
        
        confidence.domain_confidence = max(0, min(100, domain_score))
        
        # Specificity confidence assessment
        specificity_score = 50.0
        
        for factor_name, factor_data in self.confidence_factors['specificity'].items():
            if 'patterns' in factor_data:
                for pattern_str in factor_data['patterns']:
                    pattern = re.compile(pattern_str)
                    if pattern.search(claim_text):
                        specificity_score += factor_data['boost']
                        confidence.factors.append(f"specificity_boost_{factor_name}")
        
        confidence.specificity_confidence = max(0, min(100, specificity_score))
        
        # Verification confidence assessment
        verification_score = 50.0
        
        # Boost for claims with specific numeric values
        numeric_pattern = re.compile(r'\d+(?:\.\d+)?')
        numeric_matches = numeric_pattern.findall(claim_text)
        if numeric_matches:
            verification_score += len(numeric_matches) * 5
            confidence.factors.append(f"numeric_values_{len(numeric_matches)}")
        
        # Check for uncertainty indicators
        for uncertainty_type, pattern in self.uncertainty_patterns.items():
            if pattern.search(sentence):
                verification_score -= 10
                confidence.uncertainty_indicators.append(uncertainty_type)
        
        confidence.verification_confidence = max(0, min(100, verification_score))
        
        # Calculate overall confidence
        weights = {
            'linguistic': 0.25,
            'contextual': 0.20,
            'domain': 0.25,
            'specificity': 0.15,
            'verification': 0.15
        }
        
        confidence.overall_confidence = (
            confidence.linguistic_confidence * weights['linguistic'] +
            confidence.contextual_confidence * weights['contextual'] +
            confidence.domain_confidence * weights['domain'] +
            confidence.specificity_confidence * weights['specificity'] +
            confidence.verification_confidence * weights['verification']
        )
        
        return confidence
    
    def _generate_claim_id(self, claim_text: str, sentence_index: int) -> str:
        """Generate unique ID for claim."""
        combined_text = f"{claim_text}_{sentence_index}"
        return hashlib.md5(combined_text.encode()).hexdigest()[:12]
    
    def _extract_subject_from_sentence(self, sentence: str, position: int) -> str:
        """Extract subject from sentence based on position."""
        # Simple heuristic: take words before the position
        before_text = sentence[:position].strip()
        words = before_text.split()
        
        # Take last few words as potential subject
        if len(words) >= 3:
            return ' '.join(words[-3:])
        elif len(words) >= 1:
            return ' '.join(words)
        else:
            return sentence.split()[0] if sentence.split() else ""
    
    def _extract_predicate_from_match(self, match_text: str) -> str:
        """Extract predicate from match text."""
        # For numeric claims, the predicate is often implicit
        return "has_value"
    
    def _extract_object_from_sentence(self, sentence: str, position: int) -> str:
        """Extract object from sentence based on position."""
        # Simple heuristic: take words after the position
        after_text = sentence[position:].strip()
        words = after_text.split()
        
        # Take first few words as potential object
        if len(words) >= 3:
            return ' '.join(words[:3])
        elif len(words) >= 1:
            return ' '.join(words)
        else:
            return ""
    
    def _extract_qualifiers_from_sentence(self, sentence: str) -> List[str]:
        """Extract qualifying terms from sentence."""
        qualifiers = []
        
        # Look for common qualifying patterns
        qualifier_patterns = [
            r'\b(?:may|might|could|possibly|potentially)\b',
            r'\b(?:approximately|roughly|about|around)\b',
            r'\b(?:significantly|substantially|markedly)\b',
            r'\b(?:slightly|moderately|severely)\b',
            r'\b(?:under certain conditions|in some cases)\b'
        ]
        
        for pattern_str in qualifier_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(sentence)
            qualifiers.extend(matches)
        
        return list(set(qualifiers))  # Remove duplicates
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        keywords = []
        
        # Extract biomedical terms
        for term in self.all_biomedical_terms:
            if term.lower() in text.lower():
                keywords.append(term)
        
        # Extract other significant terms
        # Simple approach: words longer than 4 characters that aren't common words
        common_words = {'this', 'that', 'with', 'from', 'they', 'were', 'been',
                       'have', 'will', 'would', 'could', 'should', 'might'}
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        for word in words:
            if word not in common_words and word not in keywords:
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    async def _post_process_claims(
        self,
        claims: List[ExtractedClaim],
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractedClaim]:
        """Post-process extracted claims."""
        
        processed_claims = []
        
        for claim in claims:
            # Normalize claim text
            claim.normalized_text = self._normalize_claim_text(claim.claim_text)
            
            # Enhance keywords with context
            if query:
                query_keywords = self._extract_keywords_from_text(query)
                claim.keywords.extend(query_keywords)
                claim.keywords = list(set(claim.keywords))  # Remove duplicates
            
            # Update metadata
            claim.metadata.update({
                'processing_timestamp': datetime.now().isoformat(),
                'has_query_context': query is not None,
                'context_provided': context is not None
            })
            
            processed_claims.append(claim)
        
        return processed_claims
    
    def _normalize_claim_text(self, text: str) -> str:
        """Normalize claim text for comparison."""
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove punctuation at the end
        normalized = re.sub(r'[.!?]+$', '', normalized)
        
        return normalized
    
    async def _deduplicate_and_merge_claims(
        self,
        claims: List[ExtractedClaim]
    ) -> List[ExtractedClaim]:
        """Remove duplicate claims and merge similar ones."""
        
        if not claims:
            return []
        
        # Group claims by similarity
        claim_groups = defaultdict(list)
        
        for claim in claims:
            # Create a key based on normalized text and claim type
            similarity_key = f"{claim.claim_type}:{claim.normalized_text[:50]}"
            claim_groups[similarity_key].append(claim)
        
        # Process each group
        final_claims = []
        for group_claims in claim_groups.values():
            if len(group_claims) == 1:
                final_claims.append(group_claims[0])
            else:
                # Merge similar claims
                merged_claim = await self._merge_similar_claims(group_claims)
                final_claims.append(merged_claim)
        
        return final_claims
    
    async def _merge_similar_claims(
        self,
        similar_claims: List[ExtractedClaim]
    ) -> ExtractedClaim:
        """Merge a group of similar claims."""
        
        if len(similar_claims) == 1:
            return similar_claims[0]
        
        # Use the claim with highest confidence as base
        base_claim = max(similar_claims, key=lambda c: c.confidence.overall_confidence)
        
        # Merge information from other claims
        merged_claim = ExtractedClaim(
            claim_id=base_claim.claim_id,
            claim_text=base_claim.claim_text,
            claim_type=base_claim.claim_type,
            subject=base_claim.subject,
            predicate=base_claim.predicate,
            object_value=base_claim.object_value
        )
        
        # Merge numeric values
        all_numeric_values = []
        all_units = []
        all_qualifiers = []
        all_keywords = []
        all_relationships = []
        
        for claim in similar_claims:
            all_numeric_values.extend(claim.numeric_values)
            all_units.extend(claim.units)
            all_qualifiers.extend(claim.qualifiers)
            all_keywords.extend(claim.keywords)
            all_relationships.extend(claim.relationships)
        
        merged_claim.numeric_values = list(set(all_numeric_values))
        merged_claim.units = list(set(all_units))
        merged_claim.qualifiers = list(set(all_qualifiers))
        merged_claim.keywords = list(set(all_keywords))
        merged_claim.relationships = all_relationships
        
        # Merge confidence (use average of top scores)
        top_confidences = sorted(
            [c.confidence.overall_confidence for c in similar_claims],
            reverse=True
        )[:3]  # Top 3 scores
        
        merged_claim.confidence = ClaimConfidence(
            overall_confidence=statistics.mean(top_confidences),
            factors=[f"merged_from_{len(similar_claims)}_claims"]
        )
        
        # Copy other attributes from base claim
        merged_claim.context = base_claim.context
        merged_claim.source_sentence = base_claim.source_sentence
        merged_claim.normalized_text = base_claim.normalized_text
        merged_claim.metadata = base_claim.metadata
        merged_claim.metadata['merged_from_count'] = len(similar_claims)
        
        return merged_claim
    
    async def _calculate_priority_scores(self, claims: List[ExtractedClaim]):
        """Calculate priority scores for claims."""
        
        for claim in claims:
            # Priority score is already calculated in the property
            # Just ensure it's accessible
            _ = claim.priority_score
    
    @performance_logged("Classify claim types")
    async def classify_claims_by_type(
        self,
        claims: List[ExtractedClaim]
    ) -> Dict[str, List[ExtractedClaim]]:
        """
        Classify claims by type for targeted processing.
        
        Args:
            claims: List of extracted claims
            
        Returns:
            Dictionary mapping claim types to lists of claims
        """
        
        classified = defaultdict(list)
        
        for claim in claims:
            classified[claim.claim_type].append(claim)
        
        # Sort each type by confidence
        for claim_type in classified:
            classified[claim_type].sort(
                key=lambda c: c.confidence.overall_confidence,
                reverse=True
            )
        
        return dict(classified)
    
    @performance_logged("Filter high-confidence claims")
    async def filter_high_confidence_claims(
        self,
        claims: List[ExtractedClaim],
        min_confidence: float = 70.0
    ) -> List[ExtractedClaim]:
        """
        Filter claims based on confidence threshold.
        
        Args:
            claims: List of extracted claims
            min_confidence: Minimum confidence threshold (0-100)
            
        Returns:
            List of high-confidence claims
        """
        
        high_confidence_claims = [
            claim for claim in claims
            if claim.confidence.overall_confidence >= min_confidence
        ]
        
        # Sort by priority score
        high_confidence_claims.sort(key=lambda c: c.priority_score, reverse=True)
        
        self.logger.info(
            f"Filtered {len(high_confidence_claims)} high-confidence claims "
            f"from {len(claims)} total claims (threshold: {min_confidence})"
        )
        
        return high_confidence_claims
    
    async def prepare_claims_for_verification(
        self,
        claims: List[ExtractedClaim],
        source_documents: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare extracted claims for verification against source documents.
        
        Args:
            claims: List of extracted claims
            source_documents: Optional list of source documents
            
        Returns:
            Dictionary with prepared verification data
        """
        
        verification_data = {
            'claims_by_type': await self.classify_claims_by_type(claims),
            'high_priority_claims': await self.filter_high_confidence_claims(claims, 75.0),
            'verification_candidates': [],
            'extraction_metadata': {
                'total_claims': len(claims),
                'extraction_timestamp': datetime.now().isoformat(),
                'confidence_distribution': self._calculate_confidence_distribution(claims),
                'type_distribution': self._calculate_type_distribution(claims)
            }
        }
        
        # Prepare verification candidates
        for claim in claims:
            if claim.confidence.verification_confidence >= 60.0:
                verification_candidate = {
                    'claim_id': claim.claim_id,
                    'claim_text': claim.claim_text,
                    'claim_type': claim.claim_type,
                    'verification_targets': self._identify_verification_targets(claim),
                    'search_keywords': claim.keywords,
                    'confidence_score': claim.confidence.overall_confidence,
                    'priority_score': claim.priority_score
                }
                verification_data['verification_candidates'].append(verification_candidate)
        
        # Sort verification candidates by priority
        verification_data['verification_candidates'].sort(
            key=lambda c: c['priority_score'],
            reverse=True
        )
        
        return verification_data
    
    def _calculate_confidence_distribution(self, claims: List[ExtractedClaim]) -> Dict[str, int]:
        """Calculate distribution of confidence scores."""
        
        distribution = {
            'very_high': 0,  # 90-100
            'high': 0,       # 75-89
            'medium': 0,     # 60-74
            'low': 0,        # 45-59
            'very_low': 0    # 0-44
        }
        
        for claim in claims:
            confidence = claim.confidence.overall_confidence
            if confidence >= 90:
                distribution['very_high'] += 1
            elif confidence >= 75:
                distribution['high'] += 1
            elif confidence >= 60:
                distribution['medium'] += 1
            elif confidence >= 45:
                distribution['low'] += 1
            else:
                distribution['very_low'] += 1
        
        return distribution
    
    def _calculate_type_distribution(self, claims: List[ExtractedClaim]) -> Dict[str, int]:
        """Calculate distribution of claim types."""
        
        type_counts = Counter(claim.claim_type for claim in claims)
        return dict(type_counts)
    
    def _identify_verification_targets(self, claim: ExtractedClaim) -> List[str]:
        """Identify what aspects of the claim should be verified."""
        
        targets = []
        
        if claim.claim_type == 'numeric':
            targets.extend(['numeric_values', 'units', 'measurement_context'])
            
        elif claim.claim_type == 'qualitative':
            targets.extend(['relationships', 'causation', 'correlation'])
            
        elif claim.claim_type == 'methodological':
            targets.extend(['methods', 'procedures', 'protocols'])
            
        elif claim.claim_type == 'temporal':
            targets.extend(['timing', 'duration', 'sequence'])
            
        elif claim.claim_type == 'comparative':
            targets.extend(['comparisons', 'differences', 'statistical_significance'])
        
        # Common targets for all types
        targets.extend(['factual_accuracy', 'source_attribution'])
        
        return targets
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about claim extraction performance."""
        
        stats = {
            'total_extractions': self.extraction_stats['total_extractions'],
            'total_claims_extracted': self.extraction_stats['total_claims_extracted'],
            'average_claims_per_extraction': (
                self.extraction_stats['total_claims_extracted'] / 
                max(1, self.extraction_stats['total_extractions'])
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


# Example usage and integration helpers
async def extract_claims_from_response(
    response_text: str,
    query: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[ExtractedClaim]:
    """
    Convenience function for claim extraction.
    
    Args:
        response_text: LightRAG response text
        query: Optional original query
        config: Optional extractor configuration
        
    Returns:
        List of extracted claims
    """
    
    extractor = BiomedicalClaimExtractor(config)
    return await extractor.extract_claims(response_text, query)


async def prepare_claims_for_quality_assessment(
    claims: List[ExtractedClaim],
    min_confidence: float = 60.0
) -> Dict[str, Any]:
    """
    Prepare claims for integration with quality assessment systems.
    
    Args:
        claims: List of extracted claims
        min_confidence: Minimum confidence threshold
        
    Returns:
        Dictionary with quality assessment data
    """
    
    filtered_claims = [
        claim for claim in claims
        if claim.confidence.overall_confidence >= min_confidence
    ]
    
    return {
        'factual_claims': [claim.to_dict() for claim in filtered_claims],
        'claim_count': len(filtered_claims),
        'high_priority_claims': [
            claim.to_dict() for claim in filtered_claims
            if claim.priority_score >= 80.0
        ],
        'verification_needed': [
            claim.claim_id for claim in filtered_claims
            if claim.verification_status == 'pending'
        ],
        'assessment_metadata': {
            'extraction_timestamp': datetime.now().isoformat(),
            'confidence_threshold': min_confidence,
            'total_original_claims': len(claims)
        }
    }


if __name__ == "__main__":
    # Simple test example
    async def test_claim_extraction():
        """Test the claim extraction system."""
        
        sample_response = """
        Metabolomics analysis revealed that glucose levels were elevated by 25% 
        in diabetic patients compared to healthy controls. The LC-MS analysis 
        showed significant differences (p < 0.05) in 47 metabolites. 
        Insulin resistance correlates with increased branched-chain amino acid 
        concentrations, which were approximately 1.8-fold higher in the patient group.
        """
        
        extractor = BiomedicalClaimExtractor()
        claims = await extractor.extract_claims(sample_response)
        
        print(f"Extracted {len(claims)} claims:")
        for claim in claims:
            print(f"- {claim.claim_type}: {claim.claim_text}")
            print(f"  Confidence: {claim.confidence.overall_confidence:.1f}")
            print(f"  Priority: {claim.priority_score:.1f}")
            print()
    
    # Run test if executed directly
    asyncio.run(test_claim_extraction())