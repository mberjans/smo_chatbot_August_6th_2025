"""
Enhanced Query Classification System for Clinical Metabolomics Oracle LightRAG Integration

This module provides a comprehensive query classification system specifically designed
for the CMO-LIGHTRAG-012 requirements. It consolidates and enhances the existing
classification capabilities from research_categorizer.py and query_router.py.

Classes:
    - QueryClassificationCategories: Enum for the three main routing categories
    - BiomedicalKeywordSets: Comprehensive keyword dictionaries for classification
    - QueryClassificationEngine: Main classification engine with pattern matching
    - ClassificationResult: Detailed classification result with confidence metrics

The system supports:
    - Three-category classification (KNOWLEDGE_GRAPH, REAL_TIME, GENERAL)
    - Clinical metabolomics specific terminology and patterns
    - Performance-optimized pattern matching (<2 second response time)
    - Integration with existing LightRAG routing system
    - Comprehensive confidence scoring and uncertainty quantification

Performance Target: < 2 seconds for classification response
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Set, Any, Pattern
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from datetime import datetime
import logging
from functools import lru_cache
import hashlib


class QueryClassificationCategories(Enum):
    """
    Main routing categories for biomedical query classification.
    
    Based on docs/plan.md routing requirements:
    - KNOWLEDGE_GRAPH: relationships, connections, pathways, mechanisms, biomarkers, metabolites, diseases, clinical studies
    - REAL_TIME: latest, recent, current, new, breaking, today, this year, 2024, 2025
    - GENERAL: what is, define, explain, overview, introduction
    """
    
    KNOWLEDGE_GRAPH = "knowledge_graph"  # Route to LightRAG knowledge graph
    REAL_TIME = "real_time"              # Route to Perplexity API for current information
    GENERAL = "general"                  # Basic queries, can be handled by either system


@dataclass
class ClassificationResult:
    """
    Comprehensive classification result with detailed confidence metrics.
    """
    
    category: QueryClassificationCategories
    confidence: float  # Overall confidence score (0.0-1.0)
    reasoning: List[str]  # Explanation of classification decision
    
    # Detailed confidence breakdown
    keyword_match_confidence: float  # Confidence from keyword matching
    pattern_match_confidence: float  # Confidence from regex pattern matching
    semantic_confidence: float       # Confidence from semantic analysis
    temporal_confidence: float       # Confidence from temporal indicators
    
    # Evidence and indicators
    matched_keywords: List[str]      # Keywords that influenced classification
    matched_patterns: List[str]      # Regex patterns that matched
    biomedical_entities: List[str]   # Identified biomedical entities
    temporal_indicators: List[str]   # Temporal/real-time indicators found
    
    # Alternative classifications
    alternative_classifications: List[Tuple[QueryClassificationCategories, float]]
    
    # Performance metrics
    classification_time_ms: float    # Time taken for classification
    
    # Uncertainty quantification
    ambiguity_score: float          # How ambiguous is the query (0.0-1.0)
    conflict_score: float           # Conflicting signals between categories (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'category': self.category.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'confidence_breakdown': {
                'keyword_match_confidence': self.keyword_match_confidence,
                'pattern_match_confidence': self.pattern_match_confidence,
                'semantic_confidence': self.semantic_confidence,
                'temporal_confidence': self.temporal_confidence
            },
            'evidence': {
                'matched_keywords': self.matched_keywords,
                'matched_patterns': self.matched_patterns,
                'biomedical_entities': self.biomedical_entities,
                'temporal_indicators': self.temporal_indicators
            },
            'alternative_classifications': [
                (cat.value, conf) for cat, conf in self.alternative_classifications
            ],
            'performance': {
                'classification_time_ms': self.classification_time_ms,
                'ambiguity_score': self.ambiguity_score,
                'conflict_score': self.conflict_score
            }
        }


class BiomedicalKeywordSets:
    """
    Comprehensive keyword dictionaries for biomedical query classification.
    
    Organizes keywords by classification category with clinical metabolomics
    specific terminology and optimized for fast lookup operations.
    """
    
    def __init__(self):
        """Initialize comprehensive biomedical keyword sets."""
        
        # KNOWLEDGE_GRAPH category keywords - established biomedical knowledge
        self.knowledge_graph_keywords = {
            # Relationships and connections
            'relationships': {
                'relationship', 'relationships', 'connection', 'connections',
                'association', 'associations', 'correlation', 'correlations',
                'interaction', 'interactions', 'link', 'links', 'linkage',
                'binding', 'regulation', 'modulation', 'influence', 'effect'
            },
            
            # Pathways and mechanisms
            'pathways': {
                'pathway', 'pathways', 'network', 'networks', 'mechanism', 'mechanisms',
                'metabolic pathway', 'biochemical pathway', 'signaling pathway',
                'biosynthetic pathway', 'catabolic pathway', 'anabolic pathway',
                'metabolic network', 'regulatory network', 'gene network',
                'protein network', 'pathway analysis', 'network analysis'
            },
            
            # Biomarkers and metabolites
            'biomarkers': {
                'biomarker', 'biomarkers', 'marker', 'markers', 'indicator', 'indicators',
                'signature', 'signatures', 'metabolic signature', 'disease marker',
                'diagnostic marker', 'prognostic marker', 'therapeutic marker',
                'clinical marker', 'molecular marker', 'genetic marker',
                'protein marker', 'metabolite marker'
            },
            
            # Metabolites and compounds
            'metabolites': {
                'metabolite', 'metabolites', 'compound', 'compounds', 'molecule', 'molecules',
                'chemical', 'chemicals', 'substrate', 'substrates', 'product', 'products',
                'intermediate', 'intermediates', 'cofactor', 'cofactors',
                'small molecule', 'organic compound', 'inorganic compound',
                'natural product', 'synthetic compound'
            },
            
            # Clinical and disease entities
            'diseases': {
                'disease', 'diseases', 'disorder', 'disorders', 'syndrome', 'syndromes',
                'condition', 'conditions', 'pathology', 'pathologies', 'illness', 'illnesses',
                'cancer', 'cancers', 'tumor', 'tumors', 'diabetes', 'diabetic',
                'obesity', 'obese', 'hypertension', 'hypertensive',
                'cardiovascular', 'neurological', 'psychiatric', 'metabolic disorder'
            },
            
            # Clinical studies and research
            'clinical_studies': {
                'clinical study', 'clinical studies', 'clinical trial', 'clinical trials',
                'patient study', 'patient studies', 'cohort study', 'cohort studies',
                'case study', 'case studies', 'longitudinal study', 'cross-sectional study',
                'randomized trial', 'controlled trial', 'intervention study',
                'observational study', 'epidemiological study'
            },
            
            # Analytical techniques and methods
            'analytical_methods': {
                'mass spectrometry', 'ms', 'lc-ms', 'gc-ms', 'lc-ms/ms', 'gc-ms/ms',
                'nmr', 'nuclear magnetic resonance', 'chromatography', 'spectroscopy',
                'hplc', 'uplc', 'ce-ms', 'ion mobility', 'ftir', 'raman',
                'metabolomics', 'proteomics', 'genomics', 'lipidomics'
            },
            
            # Biological processes
            'biological_processes': {
                'metabolism', 'metabolic process', 'cellular metabolism',
                'energy metabolism', 'lipid metabolism', 'glucose metabolism',
                'amino acid metabolism', 'nucleotide metabolism',
                'glycolysis', 'gluconeogenesis', 'citric acid cycle', 'tca cycle',
                'oxidative phosphorylation', 'fatty acid synthesis',
                'beta oxidation', 'pentose phosphate pathway'
            }
        }
        
        # REAL_TIME category keywords - current and temporal information
        self.real_time_keywords = {
            # Temporal indicators
            'temporal_indicators': {
                'latest', 'recent', 'current', 'new', 'breaking', 'fresh',
                'today', 'yesterday', 'this week', 'this month', 'this year',
                'now', 'presently', 'nowadays', 'recently', 'lately',
                'up-to-date', 'contemporary', 'modern'
            },
            
            # Year-specific indicators
            'year_indicators': {
                '2024', '2025', '2026', '2027', 'this year', 'last year',
                'past year', 'recent years', 'in recent years'
            },
            
            # News and updates
            'news_updates': {
                'news', 'update', 'updates', 'announcement', 'announced',
                'breakthrough', 'discovery', 'published', 'release', 'released',
                'launched', 'unveiled', 'revealed', 'reported', 'confirmed'
            },
            
            # Research developments
            'research_developments': {
                'trend', 'trends', 'trending', 'emerging', 'evolving',
                'development', 'developments', 'advancement', 'advances',
                'progress', 'innovation', 'innovations', 'novel', 'new findings'
            },
            
            # Clinical trials and regulatory
            'clinical_temporal': {
                'clinical trial results', 'trial update', 'study results',
                'interim analysis', 'preliminary results', 'ongoing study',
                'recruiting', 'enrolling', 'phase i', 'phase ii', 'phase iii',
                'phase 1', 'phase 2', 'phase 3', 'fda approval', 'approved',
                'regulatory approval', 'market approval', 'breakthrough therapy',
                'fast track', 'priority review', 'orphan designation'
            },
            
            # Technology and methods updates
            'technology_updates': {
                'cutting-edge', 'state-of-the-art', 'next-generation',
                'innovative', 'first-in-class', 'revolutionary',
                'groundbreaking', 'pioneering', 'emerging technology',
                'new method', 'improved method', 'enhanced technique'
            }
        }
        
        # GENERAL category keywords - basic informational queries
        self.general_keywords = {
            # Definition and explanation
            'definitions': {
                'what is', 'what are', 'define', 'definition', 'definitions',
                'meaning', 'means', 'explain', 'explanation', 'describe',
                'description', 'overview', 'introduction', 'basics', 'basic',
                'fundamentals', 'principles', 'concept', 'concepts'
            },
            
            # How-to and procedural
            'procedures': {
                'how to', 'how do', 'how does', 'how can', 'procedure',
                'procedures', 'protocol', 'protocols', 'method', 'methods',
                'methodology', 'approach', 'technique', 'techniques',
                'steps', 'process', 'workflow'
            },
            
            # Educational and informational
            'educational': {
                'learn', 'learning', 'understand', 'understanding',
                'tutorial', 'guide', 'handbook', 'manual', 'reference',
                'textbook', 'educational', 'informational', 'background',
                'history', 'historical', 'context'
            },
            
            # Comparison and analysis
            'comparison': {
                'compare', 'comparison', 'versus', 'vs', 'difference',
                'differences', 'similarity', 'similarities', 'contrast',
                'advantages', 'disadvantages', 'pros', 'cons',
                'better', 'best', 'optimal', 'preferred'
            }
        }
        
        # Create flattened sets for faster lookup
        self._create_lookup_sets()
        
        # Create compiled patterns for performance
        self._compile_patterns()
    
    def _create_lookup_sets(self) -> None:
        """Create flattened keyword sets for fast lookup operations."""
        
        # Flatten keyword sets for each category
        self.knowledge_graph_set = set()
        for keyword_group in self.knowledge_graph_keywords.values():
            self.knowledge_graph_set.update(keyword_group)
        
        self.real_time_set = set()
        for keyword_group in self.real_time_keywords.values():
            self.real_time_set.update(keyword_group)
        
        self.general_set = set()
        for keyword_group in self.general_keywords.values():
            self.general_set.update(keyword_group)
        
        # Create combined biomedical entity set for entity recognition
        self.biomedical_entities_set = (
            self.knowledge_graph_keywords['biomarkers'] |
            self.knowledge_graph_keywords['metabolites'] |
            self.knowledge_graph_keywords['diseases'] |
            self.knowledge_graph_keywords['analytical_methods'] |
            self.knowledge_graph_keywords['pathways'] |
            self.knowledge_graph_keywords['biological_processes'] |
            self.knowledge_graph_keywords['relationships']
        )
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for optimized pattern matching."""
        
        # Knowledge graph patterns
        self.kg_patterns = [
            # Relationship patterns
            re.compile(r'\b(?:relationship|connection|association|correlation)\s+(?:between|of|with)', re.IGNORECASE),
            re.compile(r'\bhow\s+(?:does|do|is|are)\s+\w+\s+(?:relate|connect|associate|interact)', re.IGNORECASE),
            re.compile(r'\blink\s+between\s+\w+\s+and\s+\w+', re.IGNORECASE),
            re.compile(r'\binteraction\s+(?:between|of|with)', re.IGNORECASE),
            
            # Pathway patterns
            re.compile(r'\b(?:pathway|network|mechanism)\s+(?:of|for|in|involving)', re.IGNORECASE),
            re.compile(r'\bmetabolic\s+(?:pathway|network|route)', re.IGNORECASE),
            re.compile(r'\bbiomedical\s+pathway', re.IGNORECASE),
            re.compile(r'\bsignaling\s+(?:pathway|cascade)', re.IGNORECASE),
            
            # Mechanism patterns
            re.compile(r'\bmechanism\s+(?:of\s+action|behind|underlying)', re.IGNORECASE),
            re.compile(r'\bhow\s+does\s+\w+\s+work', re.IGNORECASE),
            re.compile(r'\bmode\s+of\s+action', re.IGNORECASE),
            re.compile(r'\bmolecular\s+mechanism', re.IGNORECASE),
            
            # Clinical study patterns
            re.compile(r'\bclinical\s+(?:study|studies|trial|trials)', re.IGNORECASE),
            re.compile(r'\bpatient\s+(?:study|studies|cohort)', re.IGNORECASE),
            re.compile(r'\b(?:randomized|controlled)\s+trial', re.IGNORECASE)
        ]
        
        # Real-time patterns
        self.rt_patterns = [
            # Temporal patterns
            re.compile(r'\b(?:latest|recent|current|new)\s+(?:research|studies|findings|developments|trials|results)', re.IGNORECASE),
            re.compile(r'\b(?:published|released)\s+(?:in\s+)?(?:2024|2025|2026|this\s+year|recently)', re.IGNORECASE),
            re.compile(r'\b(?:breaking|recent)\s+(?:news|research|discovery|breakthrough)', re.IGNORECASE),
            re.compile(r'\b(?:what\'?s\s+new|what\s+are\s+the\s+latest)', re.IGNORECASE),
            re.compile(r'\b(?:today|this\s+(?:week|month|year))', re.IGNORECASE),
            re.compile(r'\b(?:emerging|evolving|trending)\s+(?:research|field|area|therapy|treatment)', re.IGNORECASE),
            
            # Clinical and regulatory patterns
            re.compile(r'\b(?:fda\s+approval|regulatory\s+approval|market\s+approval)', re.IGNORECASE),
            re.compile(r'\bphase\s+(?:i{1,3}|[123])\s+(?:trial|study|results)', re.IGNORECASE),
            re.compile(r'\b(?:clinical\s+trial\s+results|interim\s+analysis)', re.IGNORECASE),
            re.compile(r'\b(?:breakthrough\s+therapy|fast\s+track|priority\s+review)', re.IGNORECASE),
            
            # Innovation patterns
            re.compile(r'\b(?:cutting-edge|state-of-the-art|next-generation)', re.IGNORECASE),
            re.compile(r'\b(?:novel|innovative|first-in-class)\s+(?:drug|therapy|treatment|approach)', re.IGNORECASE)
        ]
        
        # General patterns
        self.general_patterns = [
            # Definition patterns
            re.compile(r'\b(?:what\s+is|define|definition\s+of)', re.IGNORECASE),
            re.compile(r'\b(?:explain|describe|tell\s+me\s+about)', re.IGNORECASE),
            re.compile(r'\b(?:overview\s+of|introduction\s+to)', re.IGNORECASE),
            re.compile(r'\b(?:basics\s+of|fundamentals\s+of)', re.IGNORECASE),
            
            # How-to patterns
            re.compile(r'\bhow\s+to\s+\w+', re.IGNORECASE),
            re.compile(r'\b(?:procedure|protocol|method)\s+for', re.IGNORECASE),
            re.compile(r'\bsteps\s+(?:to|for)', re.IGNORECASE),
            
            # Comparison patterns
            re.compile(r'\b(?:compare|comparison|versus|vs\.?)\b', re.IGNORECASE),
            re.compile(r'\b(?:difference|differences)\s+between', re.IGNORECASE),
            re.compile(r'\b(?:advantages|disadvantages|pros|cons)\s+of', re.IGNORECASE)
        ]
    
    def get_category_keywords(self, category: QueryClassificationCategories) -> Dict[str, Set[str]]:
        """Get keyword sets for a specific category."""
        if category == QueryClassificationCategories.KNOWLEDGE_GRAPH:
            return self.knowledge_graph_keywords
        elif category == QueryClassificationCategories.REAL_TIME:
            return self.real_time_keywords
        elif category == QueryClassificationCategories.GENERAL:
            return self.general_keywords
        else:
            return {}
    
    def get_category_patterns(self, category: QueryClassificationCategories) -> List[Pattern]:
        """Get compiled patterns for a specific category."""
        if category == QueryClassificationCategories.KNOWLEDGE_GRAPH:
            return self.kg_patterns
        elif category == QueryClassificationCategories.REAL_TIME:
            return self.rt_patterns
        elif category == QueryClassificationCategories.GENERAL:
            return self.general_patterns
        else:
            return []


class QueryClassificationEngine:
    """
    Main query classification engine with pattern matching and confidence scoring.
    
    Provides comprehensive classification of biomedical queries into the three
    main categories with detailed confidence metrics and performance optimization.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the query classification engine."""
        self.logger = logger or logging.getLogger(__name__)
        self.keyword_sets = BiomedicalKeywordSets()
        
        # Classification thresholds
        self.confidence_thresholds = {
            'high': 0.7,      # High confidence classification
            'medium': 0.5,    # Medium confidence classification  
            'low': 0.3,       # Low confidence classification
            'very_low': 0.1   # Very low confidence classification
        }
        
        # Scoring weights for different types of evidence
        self.scoring_weights = {
            'keyword_match': 1.5,         # Increased weight for keyword matches
            'pattern_match': 2.5,         # Patterns weighted significantly higher
            'biomedical_entity': 1.5,     # Higher weight for biomedical entities
            'temporal_indicator': 1.8,    # Higher weight for temporal indicators
            'query_length_bonus': 0.4,    # Slightly higher length bonus
            'specificity_bonus': 0.6      # Higher specificity bonus
        }
        
        # Performance monitoring
        self._classification_times = []
        self._performance_target_ms = 2000  # 2 second target
        
        # Query caching for performance
        self._classification_cache = {}
        self._cache_max_size = 200
        
        self.logger.info("Query classification engine initialized with performance optimizations")
    
    def classify_query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify a query into one of the three main categories.
        
        Args:
            query_text: The user query text to classify
            context: Optional context information
            
        Returns:
            ClassificationResult with detailed confidence metrics
            
        Performance Target: < 2 seconds for classification
        """
        start_time = time.time()
        
        # Check cache first for performance
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        cached_result = self._get_cached_classification(query_hash)
        if cached_result and not context:  # Only use cache if no context
            return cached_result
        
        try:
            # Multi-dimensional classification analysis
            analysis_results = self._comprehensive_classification_analysis(query_text, context)
            
            # Calculate category scores
            category_scores = self._calculate_category_scores(analysis_results)
            
            # Determine final classification
            final_category, confidence, reasoning = self._determine_final_classification(
                category_scores, analysis_results
            )
            
            # Calculate detailed confidence breakdown
            confidence_breakdown = self._calculate_confidence_breakdown(
                category_scores, analysis_results
            )
            
            # Generate alternative classifications
            alternatives = self._generate_alternative_classifications(category_scores)
            
            # Calculate uncertainty metrics
            ambiguity_score, conflict_score = self._calculate_uncertainty_metrics(
                category_scores, analysis_results
            )
            
            # Create comprehensive result
            classification_time = (time.time() - start_time) * 1000
            
            result = ClassificationResult(
                category=final_category,
                confidence=confidence,
                reasoning=reasoning,
                keyword_match_confidence=confidence_breakdown['keyword_match'],
                pattern_match_confidence=confidence_breakdown['pattern_match'],
                semantic_confidence=confidence_breakdown['semantic'],
                temporal_confidence=confidence_breakdown['temporal'],
                matched_keywords=analysis_results['matched_keywords'],
                matched_patterns=analysis_results['matched_patterns'],
                biomedical_entities=analysis_results['biomedical_entities'],
                temporal_indicators=analysis_results['temporal_indicators'],
                alternative_classifications=alternatives,
                classification_time_ms=classification_time,
                ambiguity_score=ambiguity_score,
                conflict_score=conflict_score
            )
            
            # Performance tracking
            self._classification_times.append(classification_time)
            
            # Cache result for performance
            if not context and confidence >= 0.7:
                self._cache_classification_result(query_text, result)
            
            # Log performance warnings
            if classification_time > self._performance_target_ms:
                self.logger.warning(f"Classification took {classification_time:.2f}ms "
                                  f"(target: {self._performance_target_ms}ms)")
            
            # Log classification details for monitoring
            self.logger.debug(f"Classified query as {final_category.value} "
                            f"with confidence {confidence:.3f} "
                            f"in {classification_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return self._create_fallback_classification(query_text, start_time, str(e))
    
    def _comprehensive_classification_analysis(self, query_text: str, 
                                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive multi-dimensional analysis for classification.
        
        Args:
            query_text: The user query text to analyze
            context: Optional context information
            
        Returns:
            Dict containing comprehensive analysis results
        """
        query_lower = query_text.lower()
        words = query_lower.split()
        
        analysis = {
            'query_text': query_text,
            'query_lower': query_lower,
            'words': words,
            'word_count': len(words),
            'matched_keywords': [],
            'matched_patterns': [],
            'biomedical_entities': [],
            'temporal_indicators': [],
            'category_keyword_matches': defaultdict(list),
            'category_pattern_matches': defaultdict(list),
            'query_characteristics': {}
        }
        
        # Keyword matching analysis for each category
        for category in QueryClassificationCategories:
            category_keywords = self.keyword_sets.get_category_keywords(category)
            
            for keyword_group, keywords in category_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in query_lower:
                        analysis['category_keyword_matches'][category].append(keyword)
                        analysis['matched_keywords'].append(keyword)
        
        # Pattern matching analysis for each category
        for category in QueryClassificationCategories:
            patterns = self.keyword_sets.get_category_patterns(category)
            
            for pattern in patterns:
                matches = pattern.findall(query_lower)
                if matches:
                    analysis['category_pattern_matches'][category].extend(matches)
                    analysis['matched_patterns'].extend(matches)
        
        # Enhanced biomedical entity recognition
        words_set = set(words)
        biomedical_matches = words_set.intersection(self.keyword_sets.biomedical_entities_set)
        
        # Also check for partial matches and multi-word terms
        for entity in self.keyword_sets.biomedical_entities_set:
            if ' ' in entity and entity.lower() in query_lower:
                biomedical_matches.add(entity)
            elif len(entity) > 4:  # Check partial matches for longer terms
                for word in words:
                    if len(word) > 3 and (word in entity or entity in word):
                        biomedical_matches.add(entity)
                        break
        
        analysis['biomedical_entities'] = list(biomedical_matches)
        
        # Temporal indicator detection
        temporal_matches = words_set.intersection(self.keyword_sets.real_time_set)
        analysis['temporal_indicators'] = list(temporal_matches)
        
        # Query characteristics analysis
        analysis['query_characteristics'] = {
            'is_question': any(word in words for word in ['what', 'how', 'why', 'when', 'where', 'which']),
            'has_technical_terms': len(biomedical_matches) > 0,
            'has_temporal_indicators': len(temporal_matches) > 0,
            'query_complexity': len(words) + len(re.findall(r'[?.,;:]', query_text)),
            'has_comparison_terms': any(term in query_lower for term in ['compare', 'versus', 'vs', 'difference']),
            'has_definition_request': any(pattern in query_lower for pattern in ['what is', 'define', 'definition']),
            'has_procedural_request': any(pattern in query_lower for pattern in ['how to', 'procedure', 'method']),
        }
        
        return analysis
    
    def _calculate_category_scores(self, analysis_results: Dict[str, Any]) -> Dict[QueryClassificationCategories, float]:
        """Calculate scores for each classification category."""
        scores = {category: 0.0 for category in QueryClassificationCategories}
        
        # Keyword matching scores
        for category, keywords in analysis_results['category_keyword_matches'].items():
            keyword_score = len(keywords) * self.scoring_weights['keyword_match']
            scores[category] += keyword_score
        
        # Pattern matching scores (weighted higher)
        for category, patterns in analysis_results['category_pattern_matches'].items():
            pattern_score = len(patterns) * self.scoring_weights['pattern_match']
            scores[category] += pattern_score
        
        # Biomedical entity bonus for knowledge graph queries
        biomedical_count = len(analysis_results['biomedical_entities'])
        if biomedical_count > 0:
            kg_bonus = biomedical_count * self.scoring_weights['biomedical_entity']
            scores[QueryClassificationCategories.KNOWLEDGE_GRAPH] += kg_bonus
        
        # Temporal indicator bonus for real-time queries
        temporal_count = len(analysis_results['temporal_indicators'])
        if temporal_count > 0:
            rt_bonus = temporal_count * self.scoring_weights['temporal_indicator']
            scores[QueryClassificationCategories.REAL_TIME] += rt_bonus
        
        # Query characteristics bonuses
        characteristics = analysis_results['query_characteristics']
        
        # Definition requests favor general category
        if characteristics['has_definition_request']:
            scores[QueryClassificationCategories.GENERAL] += 2.0
        
        # Complex technical queries favor knowledge graph
        if characteristics['has_technical_terms'] and characteristics['query_complexity'] > 10:
            scores[QueryClassificationCategories.KNOWLEDGE_GRAPH] += 1.0
        
        # Temporal indicators strongly favor real-time
        if characteristics['has_temporal_indicators']:
            scores[QueryClassificationCategories.REAL_TIME] += 1.5
        
        # Procedural requests can favor general category
        if characteristics['has_procedural_request'] and not characteristics['has_technical_terms']:
            scores[QueryClassificationCategories.GENERAL] += 1.0
        
        # Query length and complexity bonuses
        word_count = analysis_results['word_count']
        if word_count > 10:  # Longer queries get slight boost to non-general categories
            scores[QueryClassificationCategories.KNOWLEDGE_GRAPH] += self.scoring_weights['query_length_bonus']
            scores[QueryClassificationCategories.REAL_TIME] += self.scoring_weights['query_length_bonus']
        elif word_count <= 5:  # Short queries might be general
            scores[QueryClassificationCategories.GENERAL] += self.scoring_weights['query_length_bonus']
        
        # Normalize scores to 0-1 range with improved scaling
        # Use a more realistic maximum score based on actual scoring patterns
        max_possible_score = 6.0  # Reduced for more reasonable confidence levels
        for category in scores:
            # Apply square root scaling to boost lower scores while keeping high scores reasonable
            normalized_score = scores[category] / max_possible_score
            # Apply boost function: sqrt(x) for x < 1, otherwise keep as-is
            if normalized_score < 1.0:
                scores[category] = min(normalized_score ** 0.7, 1.0)  # Gentle boost
            else:
                scores[category] = 1.0
        
        return scores
    
    def _determine_final_classification(self, category_scores: Dict[QueryClassificationCategories, float],
                                      analysis_results: Dict[str, Any]) -> Tuple[QueryClassificationCategories, float, List[str]]:
        """Determine the final classification with confidence and reasoning."""
        
        # Find the highest scoring category
        max_score = max(category_scores.values())
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        
        reasoning = []
        
        # Handle low-quality queries
        if max_score < 0.1:
            reasoning.append("No strong indicators found - defaulting to general category")
            return QueryClassificationCategories.GENERAL, 0.3, reasoning
        
        # Calculate confidence based on score difference and evidence quality
        sorted_scores = sorted(category_scores.values(), reverse=True)
        second_best_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        
        # Start with the max score and apply enhancements
        confidence = max_score
        
        # Check for very short or low-quality queries
        word_count = analysis_results.get('word_count', 0)
        if word_count <= 2 and max_score < 0.6:
            reasoning.append("Very short query with limited context")
            confidence *= 0.7  # Reduce confidence for very short queries
        
        # Boost confidence for clear category preference
        score_separation = max_score - second_best_score
        if score_separation >= 0.4:
            confidence = min(confidence * 1.2, 0.95)  # Boost for very clear decisions, cap at 0.95
            reasoning.append("Very clear category preference detected")
        elif score_separation >= 0.2:
            confidence = min(confidence * 1.1, 0.9)   # Moderate boost for clear decisions, cap at 0.9
            reasoning.append("Clear category preference detected")
        else:
            confidence *= 0.95  # Small penalty for close decisions
            reasoning.append("Close scores between categories - moderate confidence")
        
        # Add reasoning based on evidence
        matched_keywords = analysis_results.get('matched_keywords', [])
        matched_patterns = analysis_results.get('matched_patterns', [])
        
        if matched_keywords:
            reasoning.append(f"Matched {len(matched_keywords)} relevant keywords")
        
        if matched_patterns:
            reasoning.append(f"Matched {len(matched_patterns)} classification patterns")
        
        # Category-specific reasoning
        if best_category == QueryClassificationCategories.KNOWLEDGE_GRAPH:
            reasoning.append("Query focuses on established biomedical knowledge, relationships, or mechanisms")
        elif best_category == QueryClassificationCategories.REAL_TIME:
            reasoning.append("Query requires current or recent information")
        elif best_category == QueryClassificationCategories.GENERAL:
            reasoning.append("Query is a basic informational or definitional request")
        
        # Evidence quality assessment and confidence boosts
        biomedical_entities = analysis_results.get('biomedical_entities', [])
        if biomedical_entities:
            reasoning.append(f"Identified {len(biomedical_entities)} biomedical entities")
            # Boost confidence for biomedical entities (especially for knowledge graph queries)
            if best_category == QueryClassificationCategories.KNOWLEDGE_GRAPH:
                confidence = min(confidence * (1.0 + 0.1 * len(biomedical_entities)), 1.0)
        
        temporal_indicators = analysis_results.get('temporal_indicators', [])
        if temporal_indicators:
            reasoning.append(f"Detected {len(temporal_indicators)} temporal indicators")
            # Boost confidence for temporal indicators (especially for real-time queries)
            if best_category == QueryClassificationCategories.REAL_TIME:
                confidence = min(confidence * (1.0 + 0.1 * len(temporal_indicators)), 1.0)
        
        # Boost confidence based on keyword and pattern matches
        if matched_keywords and len(matched_keywords) >= 3:
            confidence = min(confidence * 1.05, 0.9)  # Small boost for multiple keyword matches
        
        if matched_patterns and len(matched_patterns) >= 2:
            confidence = min(confidence * 1.1, 0.92)  # Moderate boost for multiple pattern matches
        
        # Ensure minimum confidence for strong evidence
        if (len(matched_keywords) >= 2 or len(matched_patterns) >= 1 or len(biomedical_entities) >= 2):
            confidence = max(confidence, 0.5)  # Higher minimum confidence for decent evidence
        elif (len(matched_keywords) >= 1 or len(biomedical_entities) >= 1):
            confidence = max(confidence, 0.4)  # Basic minimum confidence
        
        return best_category, min(confidence, 1.0), reasoning
    
    def _calculate_confidence_breakdown(self, category_scores: Dict[QueryClassificationCategories, float],
                                       analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed confidence breakdown by evidence type."""
        
        # Get the best category scores for normalization
        max_score = max(category_scores.values())
        
        breakdown = {
            'keyword_match': 0.0,
            'pattern_match': 0.0,
            'semantic': 0.0,
            'temporal': 0.0
        }
        
        # Keyword match confidence
        keyword_count = len(analysis_results.get('matched_keywords', []))
        if keyword_count > 0:
            breakdown['keyword_match'] = min(keyword_count / 5.0, 1.0)  # Normalize to max 5 keywords
        
        # Pattern match confidence
        pattern_count = len(analysis_results.get('matched_patterns', []))
        if pattern_count > 0:
            breakdown['pattern_match'] = min(pattern_count / 3.0, 1.0)  # Normalize to max 3 patterns
        
        # Semantic confidence (based on biomedical entities)
        entity_count = len(analysis_results.get('biomedical_entities', []))
        if entity_count > 0:
            breakdown['semantic'] = min(entity_count / 4.0, 1.0)  # Normalize to max 4 entities
        
        # Temporal confidence (based on temporal indicators)
        temporal_count = len(analysis_results.get('temporal_indicators', []))
        if temporal_count > 0:
            breakdown['temporal'] = min(temporal_count / 3.0, 1.0)  # Normalize to max 3 temporal indicators
        
        return breakdown
    
    def _generate_alternative_classifications(self, category_scores: Dict[QueryClassificationCategories, float]) -> List[Tuple[QueryClassificationCategories, float]]:
        """Generate alternative classifications sorted by confidence."""
        alternatives = [(category, score) for category, score in category_scores.items()]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives
    
    def _calculate_uncertainty_metrics(self, category_scores: Dict[QueryClassificationCategories, float],
                                     analysis_results: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate ambiguity and conflict scores."""
        
        # Ambiguity score - how unclear is the classification
        sorted_scores = sorted(category_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            best_score = sorted_scores[0]
            second_best = sorted_scores[1]
            # High ambiguity when scores are similar
            ambiguity_score = 1.0 - (best_score - second_best)
        else:
            ambiguity_score = 1.0 if sorted_scores[0] < 0.5 else 0.0
        
        # Conflict score - contradictory evidence between categories
        conflict_score = 0.0
        
        # Check for conflicting signals
        has_temporal = len(analysis_results.get('temporal_indicators', [])) > 0
        has_kg_signals = len(analysis_results.get('biomedical_entities', [])) > 0
        has_general_patterns = analysis_results['query_characteristics']['has_definition_request']
        
        conflict_indicators = sum([has_temporal, has_kg_signals, has_general_patterns])
        if conflict_indicators > 1:
            conflict_score = min(conflict_indicators / 3.0, 1.0)
        
        return min(ambiguity_score, 1.0), min(conflict_score, 1.0)
    
    @lru_cache(maxsize=200)
    def _get_cached_classification(self, query_hash: str) -> Optional[ClassificationResult]:
        """Get cached classification result if available."""
        return self._classification_cache.get(query_hash)
    
    def _cache_classification_result(self, query_text: str, result: ClassificationResult) -> None:
        """Cache classification result for performance."""
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        
        # Limit cache size
        if len(self._classification_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._classification_cache))
            del self._classification_cache[oldest_key]
        
        self._classification_cache[query_hash] = result
    
    def _create_fallback_classification(self, query_text: str, start_time: float, error_message: str) -> ClassificationResult:
        """Create fallback classification when classification fails."""
        classification_time = (time.time() - start_time) * 1000
        
        return ClassificationResult(
            category=QueryClassificationCategories.GENERAL,
            confidence=0.1,
            reasoning=[f"Classification failed: {error_message} - using fallback"],
            keyword_match_confidence=0.0,
            pattern_match_confidence=0.0,
            semantic_confidence=0.0,
            temporal_confidence=0.0,
            matched_keywords=[],
            matched_patterns=[],
            biomedical_entities=[],
            temporal_indicators=[],
            alternative_classifications=[(QueryClassificationCategories.GENERAL, 0.1)],
            classification_time_ms=classification_time,
            ambiguity_score=1.0,
            conflict_score=0.0
        )
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive classification performance statistics."""
        
        # Calculate performance statistics
        avg_time = sum(self._classification_times) / len(self._classification_times) if self._classification_times else 0
        max_time = max(self._classification_times) if self._classification_times else 0
        
        stats = {
            'performance_metrics': {
                'total_classifications': len(self._classification_times),
                'average_classification_time_ms': avg_time,
                'max_classification_time_ms': max_time,
                'performance_target_ms': self._performance_target_ms,
                'classifications_over_target': len([t for t in self._classification_times if t > self._performance_target_ms]),
                'cache_size': len(self._classification_cache),
                'cache_max_size': self._cache_max_size,
                'cache_hit_rate': 0.0  # Would need to track hits vs misses
            },
            'keyword_sets': {
                'knowledge_graph_keywords': len(self.keyword_sets.knowledge_graph_set),
                'real_time_keywords': len(self.keyword_sets.real_time_set),
                'general_keywords': len(self.keyword_sets.general_set),
                'total_biomedical_entities': len(self.keyword_sets.biomedical_entities_set)
            },
            'pattern_counts': {
                'knowledge_graph_patterns': len(self.keyword_sets.kg_patterns),
                'real_time_patterns': len(self.keyword_sets.rt_patterns),
                'general_patterns': len(self.keyword_sets.general_patterns)
            },
            'configuration': {
                'confidence_thresholds': self.confidence_thresholds,
                'scoring_weights': self.scoring_weights
            }
        }
        
        return stats
    
    def validate_classification_performance(self, query_text: str, 
                                          expected_category: QueryClassificationCategories,
                                          expected_confidence_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Validate classification performance for a specific query.
        
        Args:
            query_text: Query to validate classification for
            expected_category: Expected classification category
            expected_confidence_range: Optional expected confidence range (min, max)
            
        Returns:
            Dict containing validation results
        """
        start_time = time.time()
        
        # Perform classification
        result = self.classify_query(query_text)
        
        validation = {
            'query': query_text,
            'expected_category': expected_category.value,
            'predicted_category': result.category.value,
            'predicted_confidence': result.confidence,
            'classification_correct': result.category == expected_category,
            'classification_time_ms': result.classification_time_ms,
            'meets_performance_target': result.classification_time_ms <= self._performance_target_ms,
            'issues': [],
            'validation_passed': True
        }
        
        # Validate category prediction
        if result.category != expected_category:
            validation['validation_passed'] = False
            validation['issues'].append(f"Category mismatch: expected {expected_category.value}, got {result.category.value}")
        
        # Validate confidence range
        if expected_confidence_range:
            min_conf, max_conf = expected_confidence_range
            if not (min_conf <= result.confidence <= max_conf):
                validation['validation_passed'] = False
                validation['issues'].append(f"Confidence {result.confidence:.3f} outside expected range [{min_conf:.3f}, {max_conf:.3f}]")
        
        # Validate performance
        if result.classification_time_ms > self._performance_target_ms:
            validation['issues'].append(f"Classification time {result.classification_time_ms:.2f}ms exceeds target {self._performance_target_ms}ms")
        
        validation['total_validation_time_ms'] = (time.time() - start_time) * 1000
        validation['detailed_result'] = result.to_dict()
        
        return validation
    
    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self._classification_cache.clear()
        self.logger.info("Classification cache cleared")
    
    def reset_performance_metrics(self) -> None:
        """Reset performance tracking metrics."""
        self._classification_times.clear()
        self.logger.info("Classification performance metrics reset")


# Integration functions for existing systems
def create_classification_engine(logger: Optional[logging.Logger] = None) -> QueryClassificationEngine:
    """Factory function to create a configured classification engine."""
    return QueryClassificationEngine(logger)


def classify_for_routing(query_text: str, 
                        context: Optional[Dict[str, Any]] = None,
                        engine: Optional[QueryClassificationEngine] = None) -> ClassificationResult:
    """
    Convenience function for query classification in routing systems.
    
    Args:
        query_text: The user query text to classify
        context: Optional context information
        engine: Optional pre-configured classification engine
        
    Returns:
        ClassificationResult with detailed metrics
    """
    if engine is None:
        engine = create_classification_engine()
    
    return engine.classify_query(query_text, context)


def get_routing_category_mapping() -> Dict[QueryClassificationCategories, str]:
    """
    Get mapping of classification categories to routing decisions.
    
    Returns:
        Dict mapping classification categories to routing system values
    """
    return {
        QueryClassificationCategories.KNOWLEDGE_GRAPH: "lightrag",
        QueryClassificationCategories.REAL_TIME: "perplexity",
        QueryClassificationCategories.GENERAL: "either"
    }


# ============================================================================
# ENHANCED CLASSIFICATION RESULT WITH COMPREHENSIVE CONFIDENCE SCORING
# ============================================================================

@dataclass
class EnhancedClassificationResult:
    """
    Enhanced classification result that integrates comprehensive confidence scoring
    from the HybridConfidenceScorer system while maintaining compatibility with
    existing ClassificationResult structure.
    
    This extends the basic ClassificationResult with:
    - Comprehensive confidence metrics from HybridConfidenceResult
    - LLM and keyword-based confidence analysis integration
    - Historical calibration and uncertainty quantification
    - Evidence strength and reliability metrics
    - Integration with existing routing infrastructure
    """
    
    # Base classification information (compatible with original)
    category: QueryClassificationCategories
    confidence: float  # Overall confidence score (0.0-1.0)
    reasoning: List[str]  # Explanation of classification decision
    
    # Original confidence breakdown (maintained for compatibility)
    keyword_match_confidence: float  # Confidence from keyword matching
    pattern_match_confidence: float  # Confidence from regex pattern matching
    semantic_confidence: float       # Confidence from semantic analysis
    temporal_confidence: float       # Confidence from temporal indicators
    
    # Evidence and indicators (maintained for compatibility)
    matched_keywords: List[str]      # Keywords that influenced classification
    matched_patterns: List[str]      # Regex patterns that matched
    biomedical_entities: List[str]   # Identified biomedical entities
    temporal_indicators: List[str]   # Temporal/real-time indicators found
    
    # Alternative classifications (maintained for compatibility)
    alternative_classifications: List[Tuple[QueryClassificationCategories, float]]
    
    # Performance metrics (maintained for compatibility)
    classification_time_ms: float    # Time taken for classification
    
    # Basic uncertainty quantification (maintained for compatibility)
    ambiguity_score: float          # How ambiguous is the query (0.0-1.0)
    conflict_score: float           # Conflicting signals between categories (0.0-1.0)
    
    # === ENHANCED CONFIDENCE SCORING INTEGRATION ===
    
    # Comprehensive confidence interval
    confidence_interval: Tuple[float, float]  # Confidence bounds (lower, upper)
    
    # Advanced confidence breakdown from HybridConfidenceResult
    llm_confidence_analysis: Optional['LLMConfidenceAnalysis'] = None
    keyword_confidence_analysis: Optional['KeywordConfidenceAnalysis'] = None
    
    # Hybrid scoring weights and calibration
    llm_weight: float = 0.5  # Weight given to LLM confidence
    keyword_weight: float = 0.5  # Weight given to keyword confidence
    calibration_adjustment: float = 0.0  # Historical calibration adjustment
    
    # Comprehensive uncertainty quantification
    epistemic_uncertainty: float = 0.0  # Model uncertainty (what we don't know)
    aleatoric_uncertainty: float = 0.0  # Data uncertainty (inherent noise)
    total_uncertainty: float = 0.0  # Combined uncertainty
    
    # Quality and reliability indicators
    confidence_reliability: float = 0.5  # How reliable this confidence estimate is
    evidence_strength: float = 0.5  # Strength of evidence for classification
    
    # Alternative confidence scenarios
    alternative_confidence_scenarios: List[Tuple[str, float]] = None
    
    # Calibration and validation metadata
    calibration_version: str = "1.0"
    confidence_calculation_time_ms: float = 0.0
    
    def __post_init__(self):
        """Initialize default values for new fields."""
        if self.alternative_confidence_scenarios is None:
            self.alternative_confidence_scenarios = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization with enhanced confidence data."""
        
        # Base dictionary from original ClassificationResult
        base_dict = {
            'category': self.category.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'confidence_breakdown': {
                'keyword_match_confidence': self.keyword_match_confidence,
                'pattern_match_confidence': self.pattern_match_confidence,
                'semantic_confidence': self.semantic_confidence,
                'temporal_confidence': self.temporal_confidence
            },
            'evidence': {
                'matched_keywords': self.matched_keywords,
                'matched_patterns': self.matched_patterns,
                'biomedical_entities': self.biomedical_entities,
                'temporal_indicators': self.temporal_indicators
            },
            'alternative_classifications': [
                (cat.value, conf) for cat, conf in self.alternative_classifications
            ],
            'performance': {
                'classification_time_ms': self.classification_time_ms,
                'ambiguity_score': self.ambiguity_score,
                'conflict_score': self.conflict_score
            }
        }
        
        # Enhanced confidence scoring data
        enhanced_confidence = {
            'confidence_interval': {
                'lower_bound': self.confidence_interval[0],
                'upper_bound': self.confidence_interval[1],
                'width': self.confidence_interval[1] - self.confidence_interval[0]
            },
            'hybrid_scoring': {
                'llm_weight': self.llm_weight,
                'keyword_weight': self.keyword_weight,
                'calibration_adjustment': self.calibration_adjustment
            },
            'uncertainty_metrics': {
                'epistemic_uncertainty': self.epistemic_uncertainty,
                'aleatoric_uncertainty': self.aleatoric_uncertainty,
                'total_uncertainty': self.total_uncertainty
            },
            'quality_indicators': {
                'confidence_reliability': self.confidence_reliability,
                'evidence_strength': self.evidence_strength
            },
            'alternative_scenarios': [
                {'scenario': scenario, 'confidence': conf} 
                for scenario, conf in self.alternative_confidence_scenarios
            ],
            'calibration_metadata': {
                'calibration_version': self.calibration_version,
                'confidence_calculation_time_ms': self.confidence_calculation_time_ms
            }
        }
        
        # LLM confidence analysis (if available)
        if self.llm_confidence_analysis:
            enhanced_confidence['llm_analysis'] = {
                'raw_confidence': self.llm_confidence_analysis.raw_confidence,
                'calibrated_confidence': self.llm_confidence_analysis.calibrated_confidence,
                'reasoning_quality_score': self.llm_confidence_analysis.reasoning_quality_score,
                'consistency_score': self.llm_confidence_analysis.consistency_score,
                'response_length': self.llm_confidence_analysis.response_length,
                'reasoning_depth': self.llm_confidence_analysis.reasoning_depth,
                'uncertainty_indicators': self.llm_confidence_analysis.uncertainty_indicators,
                'confidence_expressions': self.llm_confidence_analysis.confidence_expressions
            }
        
        # Keyword confidence analysis (if available)
        if self.keyword_confidence_analysis:
            enhanced_confidence['keyword_analysis'] = {
                'pattern_match_confidence': self.keyword_confidence_analysis.pattern_match_confidence,
                'keyword_density_confidence': self.keyword_confidence_analysis.keyword_density_confidence,
                'biomedical_entity_confidence': self.keyword_confidence_analysis.biomedical_entity_confidence,
                'domain_specificity_confidence': self.keyword_confidence_analysis.domain_specificity_confidence,
                'signal_analysis': {
                    'total_biomedical_signals': self.keyword_confidence_analysis.total_biomedical_signals,
                    'strong_signals': self.keyword_confidence_analysis.strong_signals,
                    'weak_signals': self.keyword_confidence_analysis.weak_signals,
                    'conflicting_signals': self.keyword_confidence_analysis.conflicting_signals
                },
                'coherence_metrics': {
                    'semantic_coherence_score': self.keyword_confidence_analysis.semantic_coherence_score,
                    'domain_alignment_score': self.keyword_confidence_analysis.domain_alignment_score,
                    'query_completeness_score': self.keyword_confidence_analysis.query_completeness_score
                }
            }
        
        # Combine base and enhanced data
        base_dict['enhanced_confidence_scoring'] = enhanced_confidence
        
        return base_dict
    
    @classmethod
    def from_basic_classification(cls, 
                                basic_result: ClassificationResult, 
                                hybrid_confidence_result: Optional['HybridConfidenceResult'] = None) -> 'EnhancedClassificationResult':
        """
        Create EnhancedClassificationResult from basic ClassificationResult.
        
        Args:
            basic_result: Original ClassificationResult
            hybrid_confidence_result: Optional comprehensive confidence data
            
        Returns:
            EnhancedClassificationResult with integrated confidence scoring
        """
        
        # Extract basic values
        enhanced = cls(
            category=basic_result.category,
            confidence=basic_result.confidence,
            reasoning=basic_result.reasoning,
            keyword_match_confidence=basic_result.keyword_match_confidence,
            pattern_match_confidence=basic_result.pattern_match_confidence,
            semantic_confidence=basic_result.semantic_confidence,
            temporal_confidence=basic_result.temporal_confidence,
            matched_keywords=basic_result.matched_keywords,
            matched_patterns=basic_result.matched_patterns,
            biomedical_entities=basic_result.biomedical_entities,
            temporal_indicators=basic_result.temporal_indicators,
            alternative_classifications=basic_result.alternative_classifications,
            classification_time_ms=basic_result.classification_time_ms,
            ambiguity_score=basic_result.ambiguity_score,
            conflict_score=basic_result.conflict_score,
            
            # Default confidence interval if not provided
            confidence_interval=(max(0.0, basic_result.confidence - 0.1), 
                               min(1.0, basic_result.confidence + 0.1))
        )
        
        # Integrate hybrid confidence result if available
        if hybrid_confidence_result:
            enhanced.confidence = hybrid_confidence_result.overall_confidence
            enhanced.confidence_interval = hybrid_confidence_result.confidence_interval
            enhanced.llm_confidence_analysis = hybrid_confidence_result.llm_confidence
            enhanced.keyword_confidence_analysis = hybrid_confidence_result.keyword_confidence
            enhanced.llm_weight = hybrid_confidence_result.llm_weight
            enhanced.keyword_weight = hybrid_confidence_result.keyword_weight
            enhanced.calibration_adjustment = hybrid_confidence_result.calibration_adjustment
            enhanced.epistemic_uncertainty = hybrid_confidence_result.epistemic_uncertainty
            enhanced.aleatoric_uncertainty = hybrid_confidence_result.aleatoric_uncertainty
            enhanced.total_uncertainty = hybrid_confidence_result.total_uncertainty
            enhanced.confidence_reliability = hybrid_confidence_result.confidence_reliability
            enhanced.evidence_strength = hybrid_confidence_result.evidence_strength
            enhanced.alternative_confidence_scenarios = hybrid_confidence_result.alternative_confidences
            enhanced.calibration_version = hybrid_confidence_result.calibration_version
            enhanced.confidence_calculation_time_ms = hybrid_confidence_result.calculation_time_ms
        
        return enhanced
    
    def to_basic_classification(self) -> ClassificationResult:
        """
        Convert back to basic ClassificationResult for compatibility.
        
        Returns:
            ClassificationResult with core data
        """
        return ClassificationResult(
            category=self.category,
            confidence=self.confidence,
            reasoning=self.reasoning,
            keyword_match_confidence=self.keyword_match_confidence,
            pattern_match_confidence=self.pattern_match_confidence,
            semantic_confidence=self.semantic_confidence,
            temporal_confidence=self.temporal_confidence,
            matched_keywords=self.matched_keywords,
            matched_patterns=self.matched_patterns,
            biomedical_entities=self.biomedical_entities,
            temporal_indicators=self.temporal_indicators,
            alternative_classifications=self.alternative_classifications,
            classification_time_ms=self.classification_time_ms,
            ambiguity_score=self.ambiguity_score,
            conflict_score=self.conflict_score
        )
    
    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get concise summary of confidence metrics."""
        
        return {
            'overall_confidence': self.confidence,
            'confidence_range': f"[{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]",
            'evidence_strength': self.evidence_strength,
            'reliability': self.confidence_reliability,
            'uncertainty': self.total_uncertainty,
            'calibrated': self.calibration_adjustment != 0.0,
            'llm_weight': self.llm_weight,
            'keyword_weight': self.keyword_weight
        }
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if classification meets high confidence threshold."""
        return (self.confidence >= threshold and 
                self.confidence_reliability >= 0.7 and
                self.total_uncertainty <= 0.3)
    
    def get_recommendation(self) -> Dict[str, str]:
        """Get routing recommendation based on confidence analysis."""
        
        if self.is_high_confidence():
            confidence_level = "high"
            recommendation = f"Route to {self.category.value} with high confidence"
        elif self.confidence >= 0.6 and self.evidence_strength >= 0.6:
            confidence_level = "medium"
            recommendation = f"Route to {self.category.value} with medium confidence"
        elif self.total_uncertainty > 0.7:
            confidence_level = "low"
            recommendation = "Consider hybrid routing or request clarification"
        else:
            confidence_level = "uncertain"
            recommendation = "Use fallback routing strategy"
        
        return {
            'confidence_level': confidence_level,
            'recommendation': recommendation,
            'alternative_category': self.alternative_classifications[1][0].value if len(self.alternative_classifications) > 1 else None
        }


# ============================================================================
# ENHANCED CLASSIFICATION ENGINE WITH CONFIDENCE SCORING INTEGRATION
# ============================================================================

class EnhancedQueryClassificationEngine:
    """
    Enhanced query classification engine that integrates comprehensive confidence
    scoring with the existing classification system.
    
    This engine extends the basic QueryClassificationEngine to:
    - Automatically calculate comprehensive confidence metrics
    - Integrate LLM and keyword-based confidence analysis
    - Apply historical calibration to confidence scores
    - Provide enhanced classification results with uncertainty quantification
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 enable_hybrid_confidence: bool = True,
                 confidence_scorer: Optional['HybridConfidenceScorer'] = None):
        """
        Initialize enhanced classification engine.
        
        Args:
            logger: Logger instance
            enable_hybrid_confidence: Enable comprehensive confidence scoring
            confidence_scorer: Pre-configured hybrid confidence scorer
        """
        self.logger = logger or logging.getLogger(__name__)
        self.enable_hybrid_confidence = enable_hybrid_confidence
        
        # Initialize base classification engine
        self.base_engine = QueryClassificationEngine(logger)
        
        # Initialize confidence scorer if enabled
        self.confidence_scorer = None
        if enable_hybrid_confidence:
            try:
                if confidence_scorer:
                    self.confidence_scorer = confidence_scorer
                else:
                    # Import and create confidence scorer
                    from .comprehensive_confidence_scorer import create_hybrid_confidence_scorer
                    self.confidence_scorer = create_hybrid_confidence_scorer(logger=self.logger)
                    
                self.logger.info("Enhanced classification engine initialized with comprehensive confidence scoring")
            except Exception as e:
                self.logger.warning(f"Failed to initialize confidence scorer: {e}. Using basic confidence only.")
                self.enable_hybrid_confidence = False
    
    async def classify_query_enhanced(self, 
                                    query_text: str, 
                                    context: Optional[Dict[str, Any]] = None,
                                    include_llm_analysis: bool = True) -> EnhancedClassificationResult:
        """
        Classify query with enhanced confidence scoring.
        
        Args:
            query_text: Query text to classify
            context: Optional context information
            include_llm_analysis: Whether to include LLM confidence analysis
            
        Returns:
            EnhancedClassificationResult with comprehensive confidence metrics
        """
        
        start_time = time.time()
        
        try:
            # Get basic classification
            basic_result = self.base_engine.classify_query(query_text, context)
            
            # If hybrid confidence scoring is disabled, return basic result in enhanced format
            if not self.enable_hybrid_confidence or not self.confidence_scorer:
                return EnhancedClassificationResult.from_basic_classification(basic_result)
            
            # Calculate comprehensive confidence
            hybrid_confidence_result = await self.confidence_scorer.calculate_comprehensive_confidence(
                query_text=query_text,
                context=context
            )
            
            # Create enhanced result
            enhanced_result = EnhancedClassificationResult.from_basic_classification(
                basic_result, hybrid_confidence_result
            )
            
            # Update timing
            total_time = (time.time() - start_time) * 1000
            enhanced_result.classification_time_ms = total_time
            
            self.logger.debug(f"Enhanced classification completed in {total_time:.2f}ms for query: {query_text[:50]}...")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Enhanced classification failed: {e}")
            
            # Fallback to basic classification
            try:
                basic_result = self.base_engine.classify_query(query_text, context)
                return EnhancedClassificationResult.from_basic_classification(basic_result)
            except Exception as fallback_error:
                self.logger.error(f"Fallback classification also failed: {fallback_error}")
                return self._create_fallback_enhanced_result(query_text, start_time, str(e))
    
    def classify_query(self, 
                      query_text: str, 
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify query with basic result format for compatibility.
        
        Args:
            query_text: Query text to classify
            context: Optional context information
            
        Returns:
            ClassificationResult (basic format)
        """
        return self.base_engine.classify_query(query_text, context)
    
    async def batch_classify_enhanced(self, 
                                    queries: List[str],
                                    context: Optional[Dict[str, Any]] = None) -> List[EnhancedClassificationResult]:
        """
        Batch classify multiple queries with enhanced confidence scoring.
        
        Args:
            queries: List of query texts to classify
            context: Optional shared context information
            
        Returns:
            List of EnhancedClassificationResult objects
        """
        
        results = []
        for query in queries:
            try:
                result = await self.classify_query_enhanced(query, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to classify query '{query[:50]}...': {e}")
                fallback_result = self._create_fallback_enhanced_result(query, time.time(), str(e))
                results.append(fallback_result)
        
        return results
    
    def validate_confidence_accuracy(self, 
                                   query_text: str,
                                   predicted_result: EnhancedClassificationResult,
                                   actual_category: QueryClassificationCategories,
                                   actual_routing_success: bool) -> Dict[str, Any]:
        """
        Validate confidence prediction accuracy for calibration improvement.
        
        Args:
            query_text: Original query text
            predicted_result: The classification result that was predicted
            actual_category: The actual correct category
            actual_routing_success: Whether the routing was successful
            
        Returns:
            Validation results and feedback for calibration
        """
        
        validation_results = {
            'query': query_text,
            'predicted_category': predicted_result.category.value,
            'actual_category': actual_category.value,
            'category_correct': predicted_result.category == actual_category,
            'predicted_confidence': predicted_result.confidence,
            'confidence_interval': predicted_result.confidence_interval,
            'routing_successful': actual_routing_success,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Record feedback for confidence calibration if scorer available
        if self.confidence_scorer:
            try:
                # Import confidence source enum
                from .comprehensive_confidence_scorer import ConfidenceSource
                
                self.confidence_scorer.record_prediction_feedback(
                    query_text=query_text,
                    predicted_confidence=predicted_result.confidence,
                    actual_accuracy=actual_routing_success,
                    confidence_source=ConfidenceSource.ENSEMBLE_VOTING
                )
                
                validation_results['calibration_feedback_recorded'] = True
                
            except Exception as e:
                self.logger.warning(f"Failed to record calibration feedback: {e}")
                validation_results['calibration_feedback_recorded'] = False
        
        # Calculate confidence accuracy metrics
        confidence_error = abs(predicted_result.confidence - (1.0 if actual_routing_success else 0.0))
        validation_results['confidence_error'] = confidence_error
        
        # Check if confidence interval contained the actual outcome
        if actual_routing_success:
            actual_confidence = 1.0
        else:
            actual_confidence = 0.0
            
        interval_contains_actual = (predicted_result.confidence_interval[0] <= actual_confidence <= 
                                  predicted_result.confidence_interval[1])
        validation_results['confidence_interval_accurate'] = interval_contains_actual
        
        return validation_results
    
    def get_confidence_calibration_stats(self) -> Dict[str, Any]:
        """Get confidence calibration statistics."""
        
        if not self.confidence_scorer:
            return {'confidence_scoring_enabled': False}
        
        try:
            stats = self.confidence_scorer.get_comprehensive_stats()
            stats['confidence_scoring_enabled'] = True
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get calibration stats: {e}")
            return {'confidence_scoring_enabled': False, 'error': str(e)}
    
    def _create_fallback_enhanced_result(self, 
                                       query_text: str, 
                                       start_time: float, 
                                       error_message: str) -> EnhancedClassificationResult:
        """Create fallback enhanced result when classification fails."""
        
        classification_time = (time.time() - start_time) * 1000
        
        # Simple heuristics for fallback
        query_lower = query_text.lower()
        
        # Determine category based on simple keyword presence
        if any(term in query_lower for term in ['latest', 'recent', 'current', '2024', '2025']):
            category = QueryClassificationCategories.REAL_TIME
            confidence = 0.3
        elif any(term in query_lower for term in ['pathway', 'mechanism', 'relationship']):
            category = QueryClassificationCategories.KNOWLEDGE_GRAPH
            confidence = 0.3
        else:
            category = QueryClassificationCategories.GENERAL
            confidence = 0.2
        
        return EnhancedClassificationResult(
            category=category,
            confidence=confidence,
            reasoning=[f"Fallback classification due to error: {error_message}"],
            keyword_match_confidence=0.0,
            pattern_match_confidence=0.0,
            semantic_confidence=0.0,
            temporal_confidence=0.0,
            matched_keywords=[],
            matched_patterns=[],
            biomedical_entities=[],
            temporal_indicators=[],
            alternative_classifications=[(category, confidence)],
            classification_time_ms=classification_time,
            ambiguity_score=1.0,
            conflict_score=0.0,
            confidence_interval=(max(0.0, confidence - 0.2), min(1.0, confidence + 0.1)),
            epistemic_uncertainty=0.9,
            aleatoric_uncertainty=0.7,
            total_uncertainty=1.0,
            confidence_reliability=0.1,
            evidence_strength=0.1,
            alternative_confidence_scenarios=[("fallback", confidence)],
            calibration_version="fallback"
        )


# ============================================================================
# INTEGRATION HELPER FUNCTIONS FOR ENHANCED CONFIDENCE SCORING
# ============================================================================

async def create_enhanced_classification_engine(
    logger: Optional[logging.Logger] = None,
    enable_hybrid_confidence: bool = True
) -> EnhancedQueryClassificationEngine:
    """
    Factory function to create enhanced classification engine with confidence scoring.
    
    Args:
        logger: Logger instance
        enable_hybrid_confidence: Enable comprehensive confidence scoring
        
    Returns:
        Configured EnhancedQueryClassificationEngine
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        engine = EnhancedQueryClassificationEngine(
            logger=logger,
            enable_hybrid_confidence=enable_hybrid_confidence
        )
        
        logger.info("Enhanced classification engine created successfully")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create enhanced classification engine: {e}")
        # Fallback to basic engine wrapped in enhanced interface
        return EnhancedQueryClassificationEngine(
            logger=logger,
            enable_hybrid_confidence=False
        )


async def classify_with_enhanced_confidence(
    query_text: str,
    context: Optional[Dict[str, Any]] = None,
    engine: Optional[EnhancedQueryClassificationEngine] = None
) -> EnhancedClassificationResult:
    """
    Convenience function for enhanced classification with confidence scoring.
    
    Args:
        query_text: Query to classify
        context: Optional context information
        engine: Optional pre-configured engine
        
    Returns:
        EnhancedClassificationResult with comprehensive confidence metrics
    """
    
    if engine is None:
        engine = await create_enhanced_classification_engine()
    
    return await engine.classify_query_enhanced(query_text, context)


def integrate_enhanced_classification_with_routing(
    enhanced_result: EnhancedClassificationResult
) -> Dict[str, Any]:
    """
    Convert EnhancedClassificationResult to routing decision format.
    
    Args:
        enhanced_result: Enhanced classification result
        
    Returns:
        Dict with routing information including enhanced confidence metrics
    """
    
    # Get basic routing mapping
    category_mapping = get_routing_category_mapping()
    recommended_route = category_mapping[enhanced_result.category]
    
    # Get confidence-based recommendation
    recommendation = enhanced_result.get_recommendation()
    
    return {
        'routing_decision': {
            'primary_route': recommended_route,
            'category': enhanced_result.category.value,
            'confidence_level': recommendation['confidence_level']
        },
        'confidence_metrics': enhanced_result.get_confidence_summary(),
        'recommendation': recommendation,
        'fallback_routes': [
            category_mapping[alt_cat] for alt_cat, _ in enhanced_result.alternative_classifications[:2]
        ],
        'should_use_hybrid': enhanced_result.total_uncertainty > 0.5,
        'requires_clarification': enhanced_result.confidence < 0.4,
        'detailed_analysis': enhanced_result.to_dict()
    }