"""
Biomedical Query Router for Clinical Metabolomics Oracle LightRAG Integration

This module provides intelligent query routing between LightRAG knowledge graph
and Perplexity API based on query intent, temporal requirements, and content analysis.

Classes:
    - RoutingDecision: Enum for routing destinations
    - RoutingPrediction: Result of routing analysis
    - BiomedicalQueryRouter: Main router extending ResearchCategorizer
    - TemporalAnalyzer: Specialized analyzer for real-time detection
    
The routing system supports:
    - Knowledge graph queries (relationships, pathways, established knowledge)
    - Real-time queries (latest, recent, breaking news)
    - Flexible routing with confidence scoring
    - Fallback mechanisms and hybrid approaches
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Set, Any, Pattern
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import logging
from functools import lru_cache
import hashlib

from .research_categorizer import ResearchCategorizer, CategoryPrediction
from .cost_persistence import ResearchCategory


class RoutingDecision(Enum):
    """Routing destinations for query processing."""
    
    LIGHTRAG = "lightrag"           # Route to LightRAG knowledge graph
    PERPLEXITY = "perplexity"      # Route to Perplexity API for real-time
    EITHER = "either"              # Can be handled by either service
    HYBRID = "hybrid"              # Use both services for comprehensive response


@dataclass
class ConfidenceMetrics:
    """
    Detailed confidence metrics for routing decisions.
    """
    
    overall_confidence: float  # Final confidence score (0.0-1.0)
    
    # Component confidence scores
    research_category_confidence: float  # Confidence in research category classification
    temporal_analysis_confidence: float  # Confidence in temporal vs. established detection
    signal_strength_confidence: float   # Confidence based on signal strength analysis
    context_coherence_confidence: float # Confidence in query coherence in biomedical domain
    
    # Signal strength analysis
    keyword_density: float       # Density of relevant keywords (0.0-1.0)
    pattern_match_strength: float # Strength of regex pattern matches (0.0-1.0)
    biomedical_entity_count: int # Number of recognized biomedical entities
    
    # Uncertainty quantification
    ambiguity_score: float       # How ambiguous is the query (0.0-1.0, lower is better)
    conflict_score: float        # Temporal vs. non-temporal signal conflicts (0.0-1.0, lower is better)
    alternative_interpretations: List[Tuple[RoutingDecision, float]]  # Alternative routing options
    
    # Performance metrics
    calculation_time_ms: float   # Time taken to calculate confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_confidence': self.overall_confidence,
            'research_category_confidence': self.research_category_confidence,
            'temporal_analysis_confidence': self.temporal_analysis_confidence,
            'signal_strength_confidence': self.signal_strength_confidence,
            'context_coherence_confidence': self.context_coherence_confidence,
            'keyword_density': self.keyword_density,
            'pattern_match_strength': self.pattern_match_strength,
            'biomedical_entity_count': self.biomedical_entity_count,
            'ambiguity_score': self.ambiguity_score,
            'conflict_score': self.conflict_score,
            'alternative_interpretations': [(decision.value, conf) for decision, conf in self.alternative_interpretations],
            'calculation_time_ms': self.calculation_time_ms
        }


@dataclass
class FallbackStrategy:
    """
    Fallback strategy configuration for uncertain routing decisions.
    """
    
    strategy_type: str  # 'hybrid', 'ensemble', 'circuit_breaker', 'default'
    confidence_threshold: float  # Threshold below which to use this strategy
    description: str
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class RoutingPrediction:
    """
    Represents a query routing prediction with comprehensive confidence and reasoning.
    Enhanced with detailed confidence metrics and fallback strategies.
    """
    
    routing_decision: RoutingDecision
    confidence: float  # Legacy compatibility - same as confidence_metrics.overall_confidence
    reasoning: List[str]  # Explanation of routing decision
    research_category: ResearchCategory
    
    # Enhanced confidence system
    confidence_metrics: ConfidenceMetrics
    confidence_level: str = ""  # 'high', 'medium', 'low', 'very_low' - will be set in __post_init__
    fallback_strategy: Optional[FallbackStrategy] = None
    
    # Legacy compatibility
    temporal_indicators: Optional[List[str]] = None
    knowledge_indicators: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Ensure confidence consistency and set confidence level."""
        # Ensure legacy confidence matches detailed metrics
        self.confidence = self.confidence_metrics.overall_confidence
        
        # Set confidence level based on thresholds
        if self.confidence >= 0.8:
            self.confidence_level = 'high'
        elif self.confidence >= 0.6:
            self.confidence_level = 'medium'
        elif self.confidence >= 0.4:
            self.confidence_level = 'low'
        else:
            self.confidence_level = 'very_low'
    
    def should_use_fallback(self) -> bool:
        """Determine if fallback strategy should be used."""
        return (self.fallback_strategy is not None and 
                self.confidence < self.fallback_strategy.confidence_threshold)
    
    def get_alternative_routes(self) -> List[Tuple[RoutingDecision, float]]:
        """Get alternative routing options sorted by confidence."""
        alternatives = self.confidence_metrics.alternative_interpretations.copy()
        # Remove the primary decision from alternatives
        alternatives = [(decision, conf) for decision, conf in alternatives 
                       if decision != self.routing_decision]
        return sorted(alternatives, key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'routing_decision': self.routing_decision.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'research_category': self.research_category.value,
            'confidence_metrics': self.confidence_metrics.to_dict(),
            'confidence_level': self.confidence_level,
            'temporal_indicators': self.temporal_indicators or [],
            'knowledge_indicators': self.knowledge_indicators or [],
            'metadata': self.metadata or {}
        }
        
        if self.fallback_strategy:
            result['fallback_strategy'] = {
                'strategy_type': self.fallback_strategy.strategy_type,
                'confidence_threshold': self.fallback_strategy.confidence_threshold,
                'description': self.fallback_strategy.description,
                'parameters': self.fallback_strategy.parameters
            }
        
        return result


class TemporalAnalyzer:
    """
    Specialized analyzer for detecting temporal/real-time query requirements.
    """
    
    def __init__(self):
        """Initialize temporal analysis patterns."""
        
        # Enhanced real-time temporal keywords - MUCH MORE AGGRESSIVE
        self.temporal_keywords = [
            # Temporal indicators (HIGH WEIGHT)
            'latest', 'recent', 'current', 'new', 'breaking', 'fresh',
            'today', 'yesterday', 'this week', 'this month', 'this year',
            'now', 'presently', 'nowadays', 'recently', 'lately',
            
            # Trend indicators (critical for literature search) 
            'trends', 'trending', 'trend',
            
            # Year-specific indicators (VERY HIGH WEIGHT)
            '2024', '2025', '2026', '2027',
            
            # News/update indicators (HIGH WEIGHT)
            'news', 'update', 'updates', 'announcement', 'announced',
            'breakthrough', 'discovery', 'published', 'release', 'released',
            'discoveries',  # CRITICAL: Added for "Recent biomarker discoveries"
            
            # Change indicators
            'trend', 'trends', 'trending', 'emerging', 'evolving',
            'development', 'developments', 'advancement', 'advances',
            'progress', 'innovation', 'innovations',
            
            # Real-time research indicators
            'preliminary', 'ongoing', 'in development', 'under investigation',
            'clinical trial results', 'fda approval', 'just approved',
            'phase i', 'phase ii', 'phase iii', 'phase 1', 'phase 2', 'phase 3',
            'trial update', 'study results', 'interim analysis',
            'breakthrough therapy', 'fast track', 'priority review',
            'regulatory approval', 'market approval', 'orphan designation',
            
            # Temporal research terms
            'cutting-edge', 'state-of-the-art', 'novel', 'innovative',
            'first-in-class', 'next-generation', 'modern', 'contemporary',
            
            # Additional biomedical temporal indicators
            'emerging', 'evolving', 'advancing', 'developing', 'improving',
            'updated', 'revised', 'enhanced', 'optimized', 'refined'
        ]
        
        # Enhanced real-time regex patterns with biomedical focus
        self.temporal_patterns = [
            r'\b(?:latest|recent|current|new)\s+(?:research|studies|findings|developments|trials|results)',
            r'\b(?:published|released)\s+(?:in\s+)?(?:2024|2025|2026|this\s+year|recently)',
            r'\b(?:breaking|recent)\s+(?:news|research|discovery|breakthrough)',
            r'\b(?:what\'?s\s+new|what\s+are\s+the\s+latest)',
            r'\b(?:today|this\s+(?:week|month|year))',
            r'\b(?:emerging|evolving|trending)\s+(?:research|field|area|therapy|treatment)',
            r'\b(?:current|recent)\s+trends\s+in\s+(?:clinical|research|metabolomics)',  # Critical missing pattern
            r'\b(?:recent|latest)\s+(?:advances|breakthroughs|discoveries)',
            r'\b(?:current|ongoing)\s+(?:clinical\s+trials|studies|research|investigation)',
            r'\b(?:up-to-date|cutting-edge|state-of-the-art)',
            r'\b(?:just\s+)?published',
            r'\bnow\s+available',
            r'\bcurrently\s+(?:being|under)\s+(?:investigated|studied|developed)',
            
            # Clinical and regulatory patterns
            r'\b(?:fda\s+approval|regulatory\s+approval|market\s+approval)',
            r'\bphase\s+(?:i{1,3}|[123])\s+(?:trial|study|results)',
            r'\b(?:clinical\s+trial\s+results|interim\s+analysis)',
            r'\b(?:breakthrough\s+therapy|fast\s+track|priority\s+review)',
            r'\b(?:orphan\s+designation|compassionate\s+use)',
            r'\b(?:preliminary|interim)\s+(?:results|data|findings)',
            
            # Time-sensitive biomedical terms
            r'\b(?:novel|innovative|first-in-class)\s+(?:drug|therapy|treatment|approach)',
            r'\b(?:next-generation|modern|contemporary)\s+(?:sequencing|analysis|method)',
            r'\binnovation\s+in\s+(?:metabolomics|biomarker|drug)',
            r'\brecent\s+advances\s+in\s+(?:clinical|therapeutic|diagnostic)'
        ]
        
        # Historical/established knowledge patterns (opposite of temporal)
        self.established_patterns = [
            r'\b(?:what\s+is|define|definition\s+of)',
            r'\b(?:explain|describe|overview\s+of)',
            r'\b(?:history\s+of|background\s+of)',
            r'\b(?:fundamental|basic|principles\s+of)',
            r'\b(?:established|known|traditional)',
            r'\b(?:textbook|standard|classical)',
            r'\bmechanism\s+of\s+action',
            r'\bpathway\s+(?:analysis|mapping)',
            r'\brelationship\s+between'
        ]
        
        # Compile patterns for performance
        self._compiled_temporal_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.temporal_patterns]
        self._compiled_established_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.established_patterns]
        
        # Create keyword sets for faster lookup
        self._temporal_keyword_set = set(keyword.lower() for keyword in self.temporal_keywords)
        
        # Performance monitoring
        self._analysis_times = []
    
    def analyze_temporal_content(self, query_text: str) -> Dict[str, Any]:
        """
        Analyze query for temporal/real-time indicators.
        
        Args:
            query_text: The user query to analyze
            
        Returns:
            Dict with temporal analysis results
        """
        start_time = time.time()
        query_lower = query_text.lower()
        
        analysis = {
            'has_temporal_keywords': False,
            'temporal_keywords_found': [],
            'has_temporal_patterns': False,
            'temporal_patterns_found': [],
            'has_established_patterns': False,
            'established_patterns_found': [],
            'temporal_score': 0.0,
            'established_score': 0.0,
            'year_mentions': []
        }
        
        # Check for temporal keywords with WEIGHTED SCORING
        high_weight_keywords = [
            'latest', 'recent', 'current', 'breaking', 'today', 'now',
            '2024', '2025', '2026', '2027', 'discoveries', 'breakthrough'
        ]
        
        for keyword in self.temporal_keywords:
            if keyword.lower() in query_lower:
                analysis['has_temporal_keywords'] = True
                analysis['temporal_keywords_found'].append(keyword)
                
                # Give higher weight to critical temporal keywords
                if keyword.lower() in high_weight_keywords:
                    analysis['temporal_score'] += 2.5  # Much higher weight for critical words
                else:
                    analysis['temporal_score'] += 1.0
        
        # Check for temporal patterns with ENHANCED SCORING
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                analysis['has_temporal_patterns'] = True
                analysis['temporal_patterns_found'].extend(matches)
                analysis['temporal_score'] += 3.0  # Even higher weight for patterns
        
        # Check for established knowledge patterns
        for pattern in self.established_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                analysis['has_established_patterns'] = True
                analysis['established_patterns_found'].extend(matches)
                analysis['established_score'] += 1.5
        
        # Check for specific years - HIGHEST WEIGHT
        year_pattern = r'\b(202[4-9]|20[3-9][0-9])\b'
        years = re.findall(year_pattern, query_lower)
        if years:
            analysis['year_mentions'] = years
            analysis['temporal_score'] += len(years) * 4.0  # VERY HIGH weight for years
        
        # Performance tracking
        analysis_time = (time.time() - start_time) * 1000
        self._analysis_times.append(analysis_time)
        
        if analysis_time > 50:  # Log if analysis takes too long
            logger = logging.getLogger(__name__)
            logger.warning(f"Temporal analysis took {analysis_time:.2f}ms (should be < 50ms)")
        
        return analysis


class BiomedicalQueryRouter(ResearchCategorizer):
    """
    Biomedical query router that extends ResearchCategorizer with intelligent
    routing decisions between LightRAG knowledge graph and Perplexity API.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the biomedical query router."""
        super().__init__(logger)
        self.temporal_analyzer = TemporalAnalyzer()
        
        # Define routing mappings based on research categories
        self.category_routing_map = self._initialize_category_routing_map()
        
        # Enhanced routing confidence thresholds with fallback strategies - more aggressive routing
        self.routing_thresholds = {
            'high_confidence': 0.7,      # Route directly to optimal system (lowered)
            'medium_confidence': 0.5,    # Route with monitoring (lowered)
            'low_confidence': 0.3,       # Use fallback strategies or hybrid approach (lowered)
            'fallback_threshold': 0.15   # Use fallback routing (lowered)
        }
        
        # Fallback strategies configuration
        self.fallback_strategies = {
            'hybrid': FallbackStrategy(
                strategy_type='hybrid',
                confidence_threshold=0.6,
                description='Use both systems and compare results for uncertain classifications',
                parameters={'weight_lightrag': 0.5, 'weight_perplexity': 0.5}
            ),
            'ensemble': FallbackStrategy(
                strategy_type='ensemble',
                confidence_threshold=0.4,
                description='Use ensemble voting from multiple classification approaches',
                parameters={'min_agreement': 0.7, 'voting_weight': 'confidence'}
            ),
            'circuit_breaker': FallbackStrategy(
                strategy_type='circuit_breaker',
                confidence_threshold=0.3,
                description='Use circuit breaker pattern for failed classifications',
                parameters={'failure_threshold': 3, 'recovery_time': 300}
            ),
            'default': FallbackStrategy(
                strategy_type='default',
                confidence_threshold=0.3,
                description='Default to safest routing option when all else fails',
                parameters={'default_route': 'either', 'safety_margin': 0.1}
            )
        }
        
        # Circuit breaker state tracking
        self._circuit_breaker_state = {
            'failures': 0,
            'last_failure_time': 0,
            'state': 'closed'  # 'closed', 'open', 'half_open'
        }
        
        # Compile keyword patterns for performance
        self._compile_keyword_patterns()
        
        # Query caching for performance
        self._query_cache = {}
        self._cache_max_size = 100
        
        # Performance tracking
        self._routing_times = []
        self._performance_target_ms = 100
        
        self.logger.info("Biomedical query router initialized with performance optimizations")
    
    def _initialize_category_routing_map(self) -> Dict[ResearchCategory, RoutingDecision]:
        """
        Initialize mapping of research categories to preferred routing decisions.
        
        Based on docs/plan.md routing requirements:
        - KNOWLEDGE_GRAPH: relationships, connections, pathways, mechanisms, biomarkers, metabolites, diseases, clinical studies
        - REAL_TIME: latest, recent, current, new, breaking, today, this year, 2024, 2025
        - GENERAL: what is, define, explain, overview, introduction
        """
        return {
            # Knowledge graph preferred (established relationships and mechanisms)
            ResearchCategory.METABOLITE_IDENTIFICATION: RoutingDecision.LIGHTRAG,
            ResearchCategory.PATHWAY_ANALYSIS: RoutingDecision.LIGHTRAG,
            ResearchCategory.BIOMARKER_DISCOVERY: RoutingDecision.LIGHTRAG,  # Knowledge graph better for biomarker relationships
            ResearchCategory.DRUG_DISCOVERY: RoutingDecision.LIGHTRAG,       # Knowledge graph better for drug mechanisms
            ResearchCategory.CLINICAL_DIAGNOSIS: RoutingDecision.LIGHTRAG,
            
            # Data processing - knowledge graph for established methods
            ResearchCategory.DATA_PREPROCESSING: RoutingDecision.LIGHTRAG,
            ResearchCategory.STATISTICAL_ANALYSIS: RoutingDecision.LIGHTRAG,
            ResearchCategory.KNOWLEDGE_EXTRACTION: RoutingDecision.LIGHTRAG,
            ResearchCategory.DATABASE_INTEGRATION: RoutingDecision.LIGHTRAG,
            
            # Real-time preferred (current information needed)
            ResearchCategory.LITERATURE_SEARCH: RoutingDecision.PERPLEXITY,
            ResearchCategory.EXPERIMENTAL_VALIDATION: RoutingDecision.EITHER,
            
            # General queries - flexible routing
            ResearchCategory.GENERAL_QUERY: RoutingDecision.EITHER,
            ResearchCategory.SYSTEM_MAINTENANCE: RoutingDecision.EITHER
        }
    
    def route_query(self, 
                   query_text: str,
                   context: Optional[Dict[str, Any]] = None) -> RoutingPrediction:
        """
        Route a query to the appropriate service with comprehensive confidence scoring.
        
        Args:
            query_text: The user query text to route
            context: Optional context information
            
        Returns:
            RoutingPrediction with detailed confidence metrics and fallback strategies
            
        Performance Target: < 50ms total routing time
        """
        start_time = time.time()
        
        # Check cache first for performance
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        cached_result = self._get_cached_routing(query_hash, query_text)
        if cached_result and not context:  # Only use cache if no context
            return cached_result
        
        # Check circuit breaker state
        if self._should_circuit_break():
            return self._create_circuit_breaker_response(query_text, start_time)
        
        try:
            # Multi-dimensional analysis for comprehensive confidence scoring
            analysis_results = self._comprehensive_query_analysis(query_text, context)
            
            # Calculate detailed confidence metrics
            confidence_metrics = self._calculate_comprehensive_confidence(
                query_text, analysis_results, context
            )
            
            # Determine routing decision with fallback strategies
            final_routing, reasoning, fallback_strategy = self._determine_routing_with_fallback(
                analysis_results, confidence_metrics
            )
            
            # Create enhanced routing prediction with comprehensive metrics
            prediction = RoutingPrediction(
                routing_decision=final_routing,
                confidence=confidence_metrics.overall_confidence,
                reasoning=reasoning,
                research_category=analysis_results['category_prediction'].category,
                confidence_metrics=confidence_metrics,
                fallback_strategy=fallback_strategy,
                temporal_indicators=analysis_results.get('temporal_indicators', []),
                knowledge_indicators=analysis_results.get('knowledge_indicators', []),
                metadata={
                    'analysis_results': self._serialize_analysis_results(analysis_results),
                    'routing_time_ms': 0  # Will be set below
                }
            )
            
            # Performance tracking
            total_time = (time.time() - start_time) * 1000
            self._routing_times.append(total_time)
            prediction.confidence_metrics.calculation_time_ms = total_time
            prediction.metadata['routing_time_ms'] = total_time
            
            # Cache result for performance (if no context and high confidence)
            if not context and prediction.confidence >= 0.7:
                self._cache_routing_result(query_text, prediction)
            
            # Log performance warnings
            if total_time > 50:  # Updated target to 50ms
                self.logger.warning(f"Routing took {total_time:.2f}ms (target: 50ms)")
            
            # Log confidence details for monitoring
            self.logger.debug(f"Routed query to {final_routing.value} "
                            f"with confidence {prediction.confidence:.3f} "
                            f"(level: {prediction.confidence_level}) in {total_time:.2f}ms")
            
            # Reset circuit breaker failures on success
            self._circuit_breaker_state['failures'] = 0
            
            return prediction
            
        except Exception as e:
            # Handle routing failures with circuit breaker
            self._handle_routing_failure(e, query_text)
            return self._create_fallback_response(query_text, start_time, str(e))
    
    def _calculate_routing_scores(self, 
                                query_text: str,
                                category_prediction: CategoryPrediction,
                                temporal_analysis: Dict[str, Any],
                                base_routing: RoutingDecision,
                                kg_detection: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate routing scores for each destination."""
        
        scores = {
            'lightrag': 0.0,
            'perplexity': 0.0,
            'either': 0.0,
            'hybrid': 0.0
        }
        
        # Base category routing score
        category_confidence = category_prediction.confidence
        
        if base_routing == RoutingDecision.LIGHTRAG:
            scores['lightrag'] += category_confidence * 0.8
        elif base_routing == RoutingDecision.PERPLEXITY:
            scores['perplexity'] += category_confidence * 0.8
        elif base_routing == RoutingDecision.EITHER:
            scores['either'] += category_confidence * 0.6
            scores['lightrag'] += category_confidence * 0.4
            scores['perplexity'] += category_confidence * 0.4
        
        # Temporal analysis impact
        temporal_score = temporal_analysis.get('temporal_score', 0.0)
        established_score = temporal_analysis.get('established_score', 0.0)
        
        # HYBRID DETECTION FIRST - before temporal override
        has_temporal_signals = temporal_score > 1.5
        has_kg_signals = kg_detection and kg_detection.get('confidence', 0.0) > 0.4
        
        # Multi-part complex queries with both temporal and knowledge components
        if has_temporal_signals and has_kg_signals:
            scores['hybrid'] += 0.7  # Strong hybrid boost for mixed signals
            
        # Check for specific hybrid patterns
        hybrid_patterns = [
            r'latest.*(?:and|relationship|mechanism|pathway|relate|understanding)',
            r'current.*(?:and|how.*relate|mechanism|understanding|approaches)',
            r'recent.*(?:and|impact|relationship|connection|how.*relate)',
            r'new.*(?:and|how.*affect|relate|impact|understanding)',
            r'(?:latest|current|recent).*(?:discoveries|advances).*(?:how|relate|mechanism|pathway)'
        ]
        
        is_hybrid_query = False
        for pattern in hybrid_patterns:
            if re.search(pattern, query_text.lower()):
                scores['hybrid'] += 0.8
                is_hybrid_query = True
                break
        
        # Strong temporal indicators favor Perplexity - BUT NOT FOR HYBRID QUERIES
        if temporal_score > 1.5 and not is_hybrid_query:
            # VERY STRONG temporal signals should heavily favor PERPLEXITY
            scores['perplexity'] += min(temporal_score * 0.6, 1.0)
            scores['lightrag'] = max(0, scores['lightrag'] - 0.5)
            
            # If temporal score is very high, force PERPLEXITY routing
            if temporal_score > 3.0:
                scores['perplexity'] = 0.95
                scores['lightrag'] = 0.1
                scores['either'] = 0.2
                scores['hybrid'] = 0.3
        elif temporal_score > 2.0 and not is_hybrid_query:
            scores['perplexity'] += min(temporal_score * 0.3, 0.8)
            scores['lightrag'] -= min(temporal_score * 0.2, 0.4)
        
        # Strong established knowledge indicators favor LightRAG
        if established_score > 2.0:
            scores['lightrag'] += min(established_score * 0.3, 0.8)
            scores['perplexity'] -= min(established_score * 0.2, 0.4)
        
        # Enhanced knowledge graph scoring using fast detection
        if kg_detection:
            kg_confidence = kg_detection.get('confidence', 0.0)
            if kg_confidence > 0.3:  # Lowered threshold
                scores['lightrag'] += kg_confidence * 0.7  # Increased weight
                
            # Specific knowledge graph indicators boost LightRAG
            relationship_count = len(kg_detection.get('relationship_indicators', []))
            pathway_count = len(kg_detection.get('pathway_indicators', []))
            mechanism_count = len(kg_detection.get('mechanism_indicators', []))
            
            kg_specific_score = (relationship_count * 0.3 + 
                               pathway_count * 0.3 + 
                               mechanism_count * 0.3)
            scores['lightrag'] += kg_specific_score
        
        # Real-time intent scoring - but NOT for hybrid queries
        real_time_confidence = temporal_analysis.get('confidence', 0.0)
        if real_time_confidence > 0.5 and not is_hybrid_query:
            scores['perplexity'] += real_time_confidence * 0.5  # Reduced since handled above
            scores['lightrag'] -= real_time_confidence * 0.2
        
        # Complex multi-part queries might benefit from hybrid approach - LEGACY SECTION
        query_complexity = len(query_text.split()) + len(re.findall(r'[?.]', query_text))
        
        # Additional complexity-based hybrid scoring (not already covered above)
        if query_complexity > 15 and not is_hybrid_query:  # Long, complex queries
            scores['hybrid'] += 0.3
        elif query_complexity > 20 and not is_hybrid_query:  # Very long queries
            scores['hybrid'] += 0.4
        
        # Ensure scores are non-negative
        for key in scores:
            scores[key] = max(0.0, scores[key])
        
        return scores
    
    def _determine_final_routing(self, 
                               routing_scores: Dict[str, float],
                               temporal_analysis: Dict[str, Any],
                               category_prediction: CategoryPrediction) -> Tuple[RoutingDecision, float, List[str]]:
        """Determine the final routing decision with confidence and reasoning."""
        
        reasoning = []
        
        # Find the highest scoring routing option
        max_score = max(routing_scores.values())
        best_routing = max(routing_scores.items(), key=lambda x: x[1])[0]
        
        # Convert to enum
        routing_map = {
            'lightrag': RoutingDecision.LIGHTRAG,
            'perplexity': RoutingDecision.PERPLEXITY,
            'either': RoutingDecision.EITHER,
            'hybrid': RoutingDecision.HYBRID
        }
        
        final_routing = routing_map[best_routing]
        
        # Adjust confidence based on score difference
        second_best_score = sorted(routing_scores.values(), reverse=True)[1]
        confidence = max_score
        
        # Add reasoning based on analysis
        if temporal_analysis.get('temporal_score', 0) > 2.0:
            reasoning.append("Strong temporal indicators detected - real-time information needed")
        
        if temporal_analysis.get('established_score', 0) > 2.0:
            reasoning.append("Established knowledge patterns detected - knowledge graph preferred")
        
        reasoning.append(f"Research category: {category_prediction.category.value}")
        
        if max_score - second_best_score < 0.2:
            reasoning.append("Close scores between routing options - using primary preference")
            confidence *= 0.8  # Reduce confidence for close decisions
        
        # Apply confidence thresholds and fallback logic
        if confidence < self.routing_thresholds['fallback_threshold']:
            final_routing = RoutingDecision.EITHER
            reasoning.append("Low confidence - defaulting to flexible routing")
            confidence = 0.3
        elif confidence < self.routing_thresholds['low_confidence'] and final_routing != RoutingDecision.EITHER:
            # Consider hybrid for low confidence specific routing
            if routing_scores['hybrid'] > 0.2:
                final_routing = RoutingDecision.HYBRID
                reasoning.append("Low confidence for specific routing - using hybrid approach")
        
        return final_routing, min(confidence, 1.0), reasoning
    
    def should_use_lightrag(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Simple boolean check for whether to use LightRAG.
        
        Args:
            query_text: The user query text
            context: Optional context information
            
        Returns:
            Boolean indicating whether LightRAG should be used
        """
        prediction = self.route_query(query_text, context)
        
        return prediction.routing_decision in [
            RoutingDecision.LIGHTRAG,
            RoutingDecision.HYBRID
        ] and prediction.confidence > self.routing_thresholds['low_confidence']
    
    def should_use_perplexity(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Simple boolean check for whether to use Perplexity API.
        
        Args:
            query_text: The user query text
            context: Optional context information
            
        Returns:
            Boolean indicating whether Perplexity API should be used
        """
        prediction = self.route_query(query_text, context)
        
        return prediction.routing_decision in [
            RoutingDecision.PERPLEXITY,
            RoutingDecision.EITHER,
            RoutingDecision.HYBRID
        ] and prediction.confidence > self.routing_thresholds['low_confidence']
    
    def _compile_keyword_patterns(self) -> None:
        """
        Compile keyword patterns for optimized performance.
        
        Creates compiled regex patterns and keyword sets for fast matching.
        Target: < 50ms for pattern compilation.
        """
        start_time = time.time()
        
        # Knowledge graph detection patterns (compiled for speed)
        self._knowledge_graph_patterns = {
            'relationship_patterns': [
                re.compile(r'\b(?:relationship|connection|association|correlation)\s+between', re.IGNORECASE),
                re.compile(r'\bhow\s+(?:does|do|is|are)\s+\w+\s+(?:relate|connect|associate)', re.IGNORECASE),
                re.compile(r'\blink\s+between\s+\w+\s+and\s+\w+', re.IGNORECASE),
                re.compile(r'\binteraction\s+(?:between|of|with)', re.IGNORECASE)
            ],
            'pathway_patterns': [
                re.compile(r'\b(?:pathway|network|mechanism)\s+(?:of|for|in|involving)', re.IGNORECASE),
                re.compile(r'\bmetabolic\s+(?:pathway|network|route)', re.IGNORECASE),
                re.compile(r'\bbiomedical\s+pathway', re.IGNORECASE),
                re.compile(r'\bsignaling\s+(?:pathway|cascade)', re.IGNORECASE)
            ],
            'mechanism_patterns': [
                re.compile(r'\bmechanism\s+(?:of\s+action|behind|underlying)', re.IGNORECASE),
                re.compile(r'\bhow\s+does\s+\w+\s+work', re.IGNORECASE),
                re.compile(r'\bmode\s+of\s+action', re.IGNORECASE),
                re.compile(r'\bmolecular\s+mechanism', re.IGNORECASE)
            ]
        }
        
        # Enhanced biomarker and metabolite keywords for fast lookup
        self._biomedical_keyword_sets = {
            'biomarkers': {
                'biomarker', 'biomarkers', 'marker', 'markers', 'indicator', 'indicators',
                'signature', 'signatures', 'metabolic signature', 'disease marker', 
                'diagnostic marker', 'prognostic marker', 'therapeutic marker', 
                'clinical marker', 'molecular marker', 'genetic marker'
            },
            'metabolites': {
                'metabolite', 'metabolites', 'compound', 'compounds', 'molecule', 'molecules',
                'chemical', 'chemicals', 'substrate', 'substrates', 'product', 'products',
                'intermediate', 'intermediates', 'cofactor', 'cofactors', 'enzyme', 'enzymes',
                'protein', 'proteins', 'peptide', 'peptides', 'lipid', 'lipids'
            },
            'diseases': {
                'disease', 'diseases', 'disorder', 'disorders', 'syndrome', 'syndromes',
                'condition', 'conditions', 'pathology', 'pathologies', 'cancer', 'cancers',
                'diabetes', 'diabetic', 'obesity', 'obese', 'hypertension', 'hypertensive',
                'inflammation', 'inflammatory', 'alzheimer', 'alzheimers'
            },
            'clinical_studies': {
                'clinical study', 'clinical studies', 'clinical trial', 'clinical trials',
                'patient study', 'patient studies', 'cohort study', 'cohort studies',
                'case study', 'case studies', 'longitudinal study', 'cross-sectional study',
                'randomized trial', 'controlled trial', 'phase', 'trials'
            },
            'pathways': {
                'pathway', 'pathways', 'network', 'networks', 'metabolism', 'metabolic',
                'biosynthesis', 'catabolism', 'anabolism', 'glycolysis', 'citric acid cycle',
                'fatty acid synthesis', 'lipid metabolism', 'glucose metabolism'
            },
            'relationships': {
                'relationship', 'relationships', 'connection', 'connections', 'association',
                'associations', 'correlation', 'correlations', 'interaction', 'interactions',
                'link', 'links', 'binding', 'regulation', 'modulation'
            }
        }
        
        # Enhanced general query patterns
        self._general_query_patterns = [
            re.compile(r'\b(?:what\s+is|define|definition\s+of)', re.IGNORECASE),
            re.compile(r'\b(?:explain|describe|tell\s+me\s+about)', re.IGNORECASE),
            re.compile(r'\b(?:overview\s+of|introduction\s+to)', re.IGNORECASE),
            re.compile(r'\b(?:basics\s+of|fundamentals\s+of)', re.IGNORECASE),
            re.compile(r'\b(?:what\s+are\s+the|how\s+do|how\s+does)', re.IGNORECASE),
            re.compile(r'\b(?:principles\s+of|concept\s+of)', re.IGNORECASE),
            re.compile(r'\b(?:understanding|comprehension)\s+(?:of|the)', re.IGNORECASE)
        ]
        
        compilation_time = (time.time() - start_time) * 1000
        self.logger.debug(f"Keyword patterns compiled in {compilation_time:.2f}ms")
    
    def _detect_real_time_intent(self, query_text: str) -> Dict[str, Any]:
        """
        Fast detection of real-time intent using compiled patterns.
        
        Args:
            query_text: The user query to analyze
        
        Returns:
            Dict with real-time detection results and confidence
            
        Target: < 10ms for real-time detection
        """
        start_time = time.time()
        
        query_lower = query_text.lower()
        
        detection_result = {
            'has_real_time_intent': False,
            'confidence': 0.0,
            'temporal_indicators': [],
            'real_time_patterns': [],
            'year_mentions': [],
            'clinical_temporal_indicators': [],
            'news_indicators': []
        }
        
        # Fast keyword detection using set lookup - ENHANCED WEIGHTING
        temporal_score = 0.0
        high_weight_keywords = {
            'latest', 'recent', 'current', 'breaking', 'today', 'now',
            '2024', '2025', '2026', '2027', 'discoveries', 'breakthrough',
            'news', 'advances'
        }
        
        for word in query_lower.split():
            if word in self.temporal_analyzer._temporal_keyword_set:
                detection_result['temporal_indicators'].append(word)
                
                # Higher weight for critical temporal words
                if word in high_weight_keywords:
                    temporal_score += 2.5
                else:
                    temporal_score += 1.0
        
        # Fast pattern matching with compiled patterns - ENHANCED SCORING
        pattern_score = 0.0
        for pattern in self.temporal_analyzer._compiled_temporal_patterns:
            if pattern.search(query_lower):
                match = pattern.search(query_lower)
                detection_result['real_time_patterns'].append(match.group())
                pattern_score += 3.5  # Much higher weight for patterns
        
        # Specific real-time indicators
        clinical_temporal = [
            'fda approval', 'clinical trial', 'phase', 'breakthrough',
            'regulatory', 'trial results', 'study results'
        ]
        
        for indicator in clinical_temporal:
            if indicator in query_lower:
                detection_result['clinical_temporal_indicators'].append(indicator)
                temporal_score += 1.5
        
        # News and update indicators
        news_terms = ['news', 'update', 'announcement', 'released', 'published']
        for term in news_terms:
            if term in query_lower:
                detection_result['news_indicators'].append(term)
                temporal_score += 1.2
        
        # Year detection
        year_pattern = re.compile(r'\b(202[4-9]|20[3-9][0-9])\b')
        years = year_pattern.findall(query_lower)
        if years:
            detection_result['year_mentions'] = years
            temporal_score += len(years) * 1.5
        
        # Calculate overall confidence - MORE AGGRESSIVE NORMALIZATION
        total_score = temporal_score + pattern_score
        detection_result['confidence'] = min(total_score / 6.0, 1.0)  # Lower denominator for higher confidence
        detection_result['has_real_time_intent'] = detection_result['confidence'] > 0.25  # Lower threshold
        
        detection_time = (time.time() - start_time) * 1000
        if detection_time > 10:  # Log if exceeds target
            self.logger.warning(f"Real-time detection took {detection_time:.2f}ms (target: 10ms)")
        
        return detection_result
    
    def _fast_knowledge_graph_detection(self, query_text: str) -> Dict[str, Any]:
        """
        Fast detection of knowledge graph indicators using optimized patterns.
        
        Args:
            query_text: The user query to analyze
            
        Returns:
            Dict with knowledge graph detection results
            
        Target: < 15ms for knowledge graph detection
        """
        start_time = time.time()
        
        query_lower = query_text.lower()
        
        detection_result = {
            'has_kg_intent': False,
            'confidence': 0.0,
            'relationship_indicators': [],
            'pathway_indicators': [],
            'mechanism_indicators': [],
            'biomedical_entities': [],
            'general_query_indicators': []
        }
        
        kg_score = 0.0
        
        # Fast relationship detection
        for pattern in self._knowledge_graph_patterns['relationship_patterns']:
            if pattern.search(query_lower):
                match = pattern.search(query_lower)
                detection_result['relationship_indicators'].append(match.group())
                kg_score += 2.0
        
        # Fast pathway detection
        for pattern in self._knowledge_graph_patterns['pathway_patterns']:
            if pattern.search(query_lower):
                match = pattern.search(query_lower)
                detection_result['pathway_indicators'].append(match.group())
                kg_score += 2.0
        
        # Fast mechanism detection
        for pattern in self._knowledge_graph_patterns['mechanism_patterns']:
            if pattern.search(query_lower):
                match = pattern.search(query_lower)
                detection_result['mechanism_indicators'].append(match.group())
                kg_score += 2.0
        
        # Fast biomedical entity detection using keyword sets
        words = set(query_lower.split())
        for entity_type, keywords in self._biomedical_keyword_sets.items():
            matches = words.intersection(keywords)
            if matches:
                detection_result['biomedical_entities'].extend(list(matches))
                kg_score += len(matches) * 1.0
        
        # General query pattern detection
        for pattern in self._general_query_patterns:
            if pattern.search(query_lower):
                match = pattern.search(query_lower)
                detection_result['general_query_indicators'].append(match.group())
                kg_score += 1.0
        
        # Calculate confidence (much more sensitive)
        detection_result['confidence'] = min(kg_score / 3.0, 1.0)  # More generous normalization
        detection_result['has_kg_intent'] = detection_result['confidence'] > 0.2  # Lower threshold for detection
        
        detection_time = (time.time() - start_time) * 1000
        if detection_time > 15:  # Log if exceeds target
            self.logger.warning(f"Knowledge graph detection took {detection_time:.2f}ms (target: 15ms)")
        
        return detection_result
    
    @lru_cache(maxsize=100)
    def _get_cached_routing(self, query_hash: str, query_text: str) -> Optional[RoutingPrediction]:
        """Get cached routing result if available."""
        return self._query_cache.get(query_hash)
    
    def _cache_routing_result(self, query_text: str, prediction: RoutingPrediction) -> None:
        """Cache routing result for performance."""
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        
        # Limit cache size
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[query_hash] = prediction
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions.
        
        Returns:
            Dict containing routing performance metrics
        """
        base_stats = self.get_category_statistics()
        
        # Calculate performance statistics
        avg_routing_time = sum(self._routing_times) / len(self._routing_times) if self._routing_times else 0
        max_routing_time = max(self._routing_times) if self._routing_times else 0
        
        routing_stats = {
            'routing_thresholds': self.routing_thresholds,
            'category_routing_map': {cat.value: decision.value 
                                   for cat, decision in self.category_routing_map.items()},
            'temporal_keywords_count': len(self.temporal_analyzer.temporal_keywords),
            'temporal_patterns_count': len(self.temporal_analyzer.temporal_patterns),
            'performance_metrics': {
                'cache_size': len(self._query_cache),
                'cache_max_size': self._cache_max_size,
                'average_routing_time_ms': avg_routing_time,
                'max_routing_time_ms': max_routing_time,
                'performance_target_ms': self._performance_target_ms,
                'queries_over_target': len([t for t in self._routing_times if t > self._performance_target_ms]),
                'total_queries_routed': len(self._routing_times)
            },
            'compiled_patterns': {
                'knowledge_graph_patterns': len(self._knowledge_graph_patterns['relationship_patterns'] + 
                                              self._knowledge_graph_patterns['pathway_patterns'] + 
                                              self._knowledge_graph_patterns['mechanism_patterns']),
                'general_query_patterns': len(self._general_query_patterns),
                'biomedical_keyword_sets': {k: len(v) for k, v in self._biomedical_keyword_sets.items()}
            }
        }
        
        # Merge with base categorization stats
        base_stats.update(routing_stats)
        return base_stats
    
    # ============================================================================
    # COMPREHENSIVE CONFIDENCE SCORING METHODS
    # ============================================================================
    
    def _comprehensive_query_analysis(self, query_text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive multi-dimensional analysis of query for confidence scoring.
        
        Args:
            query_text: The user query text to analyze
            context: Optional context information
            
        Returns:
            Dict containing comprehensive analysis results
            
        Performance Target: < 30ms for comprehensive analysis
        """
        start_time = time.time()
        
        # Parallel analysis components
        analysis_results = {}
        
        # 1. Research categorization analysis
        category_prediction = self.categorize_query(query_text, context)
        analysis_results['category_prediction'] = category_prediction
        
        # 2. Temporal analysis
        temporal_analysis = self.temporal_analyzer.analyze_temporal_content(query_text)
        analysis_results['temporal_analysis'] = temporal_analysis
        
        # 3. Fast real-time intent detection
        real_time_detection = self._detect_real_time_intent(query_text)
        analysis_results['real_time_detection'] = real_time_detection
        
        # 4. Knowledge graph detection
        kg_detection = self._fast_knowledge_graph_detection(query_text)
        analysis_results['kg_detection'] = kg_detection
        
        # 5. Signal strength analysis
        signal_strength = self._analyze_signal_strength(query_text)
        analysis_results['signal_strength'] = signal_strength
        
        # 6. Context coherence analysis
        context_coherence = self._analyze_context_coherence(query_text, context)
        analysis_results['context_coherence'] = context_coherence
        
        # 7. Ambiguity and conflict analysis
        ambiguity_analysis = self._analyze_ambiguity_and_conflicts(
            query_text, temporal_analysis, kg_detection
        )
        analysis_results['ambiguity_analysis'] = ambiguity_analysis
        
        # Aggregate indicators for easy access
        analysis_results['temporal_indicators'] = (
            temporal_analysis.get('temporal_keywords_found', []) + 
            real_time_detection.get('temporal_indicators', [])
        )
        analysis_results['knowledge_indicators'] = (
            category_prediction.evidence + 
            kg_detection.get('biomedical_entities', [])
        )
        
        analysis_time = (time.time() - start_time) * 1000
        analysis_results['analysis_time_ms'] = analysis_time
        
        if analysis_time > 30:
            self.logger.warning(f"Comprehensive analysis took {analysis_time:.2f}ms (target: 30ms)")
        
        return analysis_results
    
    def _analyze_signal_strength(self, query_text: str) -> Dict[str, Any]:
        """
        Analyze signal strength including keyword density and pattern matches.
        
        Args:
            query_text: The user query text to analyze
            
        Returns:
            Dict containing signal strength metrics
        """
        query_lower = query_text.lower()
        words = query_lower.split()
        word_count = len(words)
        
        signal_strength = {
            'keyword_density': 0.0,
            'pattern_match_strength': 0.0,
            'biomedical_entity_count': 0,
            'technical_term_density': 0.0,
            'signal_quality_score': 0.0
        }
        
        if word_count == 0:
            return signal_strength
        
        # Calculate keyword density
        biomedical_keywords = 0
        for entity_type, keywords in self._biomedical_keyword_sets.items():
            matches = set(words).intersection(keywords)
            biomedical_keywords += len(matches)
            signal_strength['biomedical_entity_count'] += len(matches)
        
        signal_strength['keyword_density'] = min(biomedical_keywords / word_count, 1.0)
        
        # Calculate pattern match strength
        pattern_matches = 0
        total_patterns = (
            len(self._knowledge_graph_patterns['relationship_patterns']) +
            len(self._knowledge_graph_patterns['pathway_patterns']) +
            len(self._knowledge_graph_patterns['mechanism_patterns'])
        )
        
        for pattern_group in self._knowledge_graph_patterns.values():
            for pattern in pattern_group:
                if pattern.search(query_lower):
                    pattern_matches += 1
        
        signal_strength['pattern_match_strength'] = (
            pattern_matches / total_patterns if total_patterns > 0 else 0.0
        )
        
        # Technical term density
        technical_terms = [
            'lc-ms', 'gc-ms', 'nmr', 'metabolomics', 'proteomics', 'genomics',
            'biomarker', 'pathway', 'kegg', 'hmdb', 'pubchem', 'chebi'
        ]
        tech_term_count = sum(1 for term in technical_terms if term in query_lower)
        signal_strength['technical_term_density'] = min(tech_term_count / word_count, 1.0)
        
        # Overall signal quality score (weighted combination with stronger boost)
        base_score = (
            signal_strength['keyword_density'] * 0.4 +
            signal_strength['pattern_match_strength'] * 0.3 +
            signal_strength['technical_term_density'] * 0.3
        )
        
        # Apply much stronger boosts for biomedical signals
        biomedical_boost = 0.0
        if signal_strength['biomedical_entity_count'] >= 3:
            biomedical_boost = 0.4  # Strong boost for rich biomedical content
        elif signal_strength['biomedical_entity_count'] >= 2:
            biomedical_boost = 0.3  # Good boost for decent content
        elif signal_strength['biomedical_entity_count'] >= 1:
            biomedical_boost = 0.25  # Still significant boost for any biomedical content
        
        # Additional boost for any biomedical keywords at all
        if signal_strength['keyword_density'] > 0:
            biomedical_boost += 0.1  # Base boost for any biomedical keywords
        
        # Final score with biomedical boost - ensure minimum quality for biomedical queries
        final_score = base_score + biomedical_boost
        if signal_strength['biomedical_entity_count'] > 0:
            final_score = max(final_score, 0.4)  # Minimum score for biomedical queries
        
        signal_strength['signal_quality_score'] = min(final_score, 1.0)
        
        return signal_strength
    
    def _analyze_context_coherence(self, query_text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze query coherence within biomedical domain.
        
        Args:
            query_text: The user query text to analyze
            context: Optional context information
            
        Returns:
            Dict containing context coherence metrics
        """
        coherence = {
            'domain_coherence': 0.0,
            'query_completeness': 0.0,
            'semantic_consistency': 0.0,
            'context_alignment': 0.0,
            'overall_coherence': 0.0
        }
        
        query_lower = query_text.lower()
        words = query_lower.split()
        word_count = len(words)
        
        # Domain coherence - how well does query fit biomedical domain
        biomedical_domains = [
            'metabolomics', 'proteomics', 'genomics', 'clinical', 'pharmaceutical',
            'analytical', 'statistical', 'bioinformatics', 'biochemical'
        ]
        domain_matches = sum(1 for domain in biomedical_domains if domain in query_lower)
        coherence['domain_coherence'] = min(domain_matches / len(biomedical_domains), 1.0)
        
        # Query completeness - does it have subject, action, context?
        completeness_score = 0.0
        if word_count >= 3:  # Has minimum complexity
            completeness_score += 0.3
        if any(action in query_lower for action in ['analyze', 'identify', 'determine', 'study']):
            completeness_score += 0.3  # Has action
        if any(obj in query_lower for obj in ['metabolite', 'biomarker', 'pathway', 'sample']):
            completeness_score += 0.4  # Has object
        coherence['query_completeness'] = min(completeness_score, 1.0)
        
        # Semantic consistency - conflicting or contradictory terms
        consistency_score = 1.0  # Start high, subtract for inconsistencies
        conflicting_pairs = [
            ('metabolomics', 'genomics'), ('lc-ms', 'nmr'), ('statistical', 'experimental')
        ]
        for term1, term2 in conflicting_pairs:
            if term1 in query_lower and term2 in query_lower:
                consistency_score -= 0.1
        coherence['semantic_consistency'] = max(consistency_score, 0.0)
        
        # Context alignment - how well does query align with provided context
        if context:
            alignment_score = 0.0
            if 'previous_categories' in context:
                # Check if query aligns with recent research focus
                alignment_score += 0.5
            if 'user_research_areas' in context:
                # Check if query matches user's expertise
                alignment_score += 0.3
            if 'project_type' in context:
                # Check if query fits project context
                alignment_score += 0.2
            coherence['context_alignment'] = min(alignment_score, 1.0)
        
        # Overall coherence (weighted combination)
        coherence['overall_coherence'] = (
            coherence['domain_coherence'] * 0.3 +
            coherence['query_completeness'] * 0.3 +
            coherence['semantic_consistency'] * 0.2 +
            coherence['context_alignment'] * 0.2
        )
        
        return coherence
    
    def _analyze_ambiguity_and_conflicts(self, query_text: str, 
                                       temporal_analysis: Dict[str, Any],
                                       kg_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query ambiguity and signal conflicts.
        
        Args:
            query_text: The user query text to analyze
            temporal_analysis: Results from temporal analysis
            kg_detection: Results from knowledge graph detection
            
        Returns:
            Dict containing ambiguity and conflict analysis
        """
        analysis = {
            'ambiguity_score': 0.0,
            'conflict_score': 0.0,
            'vague_terms': [],
            'conflicting_signals': [],
            'multiple_interpretations': []
        }
        
        query_lower = query_text.lower()
        words = query_lower.split()
        
        # Ambiguity analysis
        ambiguity_indicators = 0
        
        # Vague terms that increase ambiguity
        vague_terms = ['analysis', 'method', 'study', 'research', 'data', 'information']
        for term in vague_terms:
            if term in words:
                analysis['vague_terms'].append(term)
                ambiguity_indicators += 1
        
        # Very short queries are ambiguous
        if len(words) <= 2:
            ambiguity_indicators += 2
        
        # Questions without specific context are ambiguous
        question_words = ['what', 'how', 'why', 'when', 'where']
        if any(word in words for word in question_words) and len(words) <= 5:
            ambiguity_indicators += 1
        
        analysis['ambiguity_score'] = min(ambiguity_indicators / 5.0, 1.0)
        
        # Conflict analysis - temporal vs. established knowledge signals
        temporal_score = temporal_analysis.get('temporal_score', 0.0)
        established_score = temporal_analysis.get('established_score', 0.0)
        kg_confidence = kg_detection.get('confidence', 0.0)
        
        conflict_indicators = 0
        
        # Strong signals in both directions indicate conflict
        if temporal_score > 2.0 and established_score > 2.0:
            analysis['conflicting_signals'].append('temporal_vs_established')
            conflict_indicators += 1
        
        # High knowledge graph confidence with temporal indicators
        if kg_confidence > 0.6 and temporal_score > 2.0:
            analysis['conflicting_signals'].append('knowledge_graph_vs_temporal')
            conflict_indicators += 0.5
        
        analysis['conflict_score'] = min(conflict_indicators / 2.0, 1.0)
        
        # Multiple interpretation detection
        if analysis['ambiguity_score'] > 0.5:
            analysis['multiple_interpretations'].extend([
                ('general_query', 0.3),
                ('specific_research', 0.2)
            ])
        
        if analysis['conflict_score'] > 0.3:
            analysis['multiple_interpretations'].extend([
                ('temporal_focus', temporal_score / 10.0),
                ('knowledge_focus', kg_confidence)
            ])
        
        return analysis
    
    def _calculate_comprehensive_confidence(self, query_text: str, 
                                          analysis_results: Dict[str, Any],
                                          context: Optional[Dict[str, Any]]) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics from analysis results.
        
        Args:
            query_text: The user query text
            analysis_results: Results from comprehensive analysis
            context: Optional context information
            
        Returns:
            ConfidenceMetrics with detailed confidence scoring
        """
        start_time = time.time()
        
        # Extract analysis components
        category_prediction = analysis_results['category_prediction']
        temporal_analysis = analysis_results['temporal_analysis']
        real_time_detection = analysis_results['real_time_detection']
        kg_detection = analysis_results['kg_detection']
        signal_strength = analysis_results['signal_strength']
        context_coherence = analysis_results['context_coherence']
        ambiguity_analysis = analysis_results['ambiguity_analysis']
        
        # Component confidence scores
        research_category_confidence = category_prediction.confidence
        temporal_analysis_confidence = min(
            (temporal_analysis.get('temporal_score', 0.0) + 
             real_time_detection.get('confidence', 0.0)) / 2.0, 1.0
        )
        signal_strength_confidence = signal_strength['signal_quality_score']
        context_coherence_confidence = context_coherence['overall_coherence']
        
        # Calculate overall confidence using weighted combination - more optimistic scoring
        weights = {
            'research_category': 0.5,   # Increased weight for main categorization
            'temporal_analysis': 0.1,   # Reduced weight
            'signal_strength': 0.25,    # Balanced weight for signal quality
            'context_coherence': 0.15   # Reduced weight
        }
        
        # Base confidence calculation with better baseline
        base_confidence = (
            research_category_confidence * weights['research_category'] +
            temporal_analysis_confidence * weights['temporal_analysis'] +
            signal_strength_confidence * weights['signal_strength'] +
            context_coherence_confidence * weights['context_coherence']
        )
        
        # Apply much smaller ambiguity and conflict penalties
        ambiguity_penalty = ambiguity_analysis['ambiguity_score'] * 0.08  # Further reduced
        conflict_penalty = ambiguity_analysis['conflict_score'] * 0.05    # Much smaller penalty
        overall_confidence = max(0.2, base_confidence - ambiguity_penalty - conflict_penalty)  # Higher minimum
        
        # Apply stronger confidence boosts for biomedical evidence
        biomedical_entities = signal_strength['biomedical_entity_count']
        keyword_density = signal_strength['keyword_density']
        
        if biomedical_entities >= 3 or keyword_density > 0.2:
            overall_confidence = min(overall_confidence * 1.4, 0.95)  # Strong boost for clear biomedical signals
        elif biomedical_entities >= 2 or keyword_density > 0.15:
            overall_confidence = min(overall_confidence * 1.3, 0.9)   # Good boost for decent signals
        elif biomedical_entities >= 1 or keyword_density > 0.1:
            overall_confidence = min(overall_confidence * 1.2, 0.85)  # Moderate boost for basic signals
        
        # Additional boost for clear pathway/mechanism queries
        if (signal_strength['pattern_match_strength'] > 0.5 or 
            research_category_confidence > 0.7):
            overall_confidence = min(overall_confidence * 1.15, 0.95)
        
        # Generate alternative interpretations
        alternative_interpretations = self._generate_alternative_interpretations(query_text, analysis_results)
        
        # Create comprehensive confidence metrics
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=overall_confidence,
            research_category_confidence=research_category_confidence,
            temporal_analysis_confidence=temporal_analysis_confidence,
            signal_strength_confidence=signal_strength_confidence,
            context_coherence_confidence=context_coherence_confidence,
            keyword_density=signal_strength['keyword_density'],
            pattern_match_strength=signal_strength['pattern_match_strength'],
            biomedical_entity_count=signal_strength['biomedical_entity_count'],
            ambiguity_score=ambiguity_analysis['ambiguity_score'],
            conflict_score=ambiguity_analysis['conflict_score'],
            alternative_interpretations=alternative_interpretations,
            calculation_time_ms=(time.time() - start_time) * 1000
        )
        
        return confidence_metrics
    
    def _generate_alternative_interpretations(self, query_text: str, analysis_results: Dict[str, Any]) -> List[Tuple[RoutingDecision, float]]:
        """
        Generate alternative routing interpretations with confidence scores.
        
        Args:
            analysis_results: Results from comprehensive analysis
            
        Returns:
            List of (RoutingDecision, confidence) tuples
        """
        alternatives = []
        
        # Get component analysis
        category_prediction = analysis_results['category_prediction']
        temporal_analysis = analysis_results['temporal_analysis']
        kg_detection = analysis_results['kg_detection']
        
        # Base routing from category
        base_routing = self.category_routing_map.get(
            category_prediction.category, 
            RoutingDecision.EITHER
        )
        
        # Override general queries with strong KG signals to route to LIGHTRAG
        kg_detection = analysis_results.get('kg_detection', {})
        if (category_prediction.category == ResearchCategory.GENERAL_QUERY and 
            kg_detection.get('confidence', 0.0) > 0.5 and
            kg_detection.get('has_kg_intent', False)):
            base_routing = RoutingDecision.LIGHTRAG
        
        # Calculate scores for each routing option
        routing_scores = {
            RoutingDecision.LIGHTRAG: 0.0,
            RoutingDecision.PERPLEXITY: 0.0,
            RoutingDecision.EITHER: 0.3,  # Default baseline
            RoutingDecision.HYBRID: 0.0
        }
        
        # Research category influence
        category_conf = category_prediction.confidence
        if base_routing == RoutingDecision.LIGHTRAG:
            routing_scores[RoutingDecision.LIGHTRAG] += category_conf * 0.7
        elif base_routing == RoutingDecision.PERPLEXITY:
            routing_scores[RoutingDecision.PERPLEXITY] += category_conf * 0.7
        else:
            routing_scores[RoutingDecision.EITHER] += category_conf * 0.5
        
        # Get temporal analysis
        temporal_score = temporal_analysis.get('temporal_score', 0.0)
        
        # Get knowledge graph confidence first
        kg_confidence = kg_detection.get('confidence', 0.0)
        
        # HYBRID DETECTION FIRST - before temporal override
        has_temporal_signals = temporal_score > 1.5
        has_kg_signals = kg_confidence > 0.4
        
        # Multi-part complex queries with both temporal and knowledge components
        if has_temporal_signals and has_kg_signals:
            routing_scores[RoutingDecision.HYBRID] += 0.7  # Strong hybrid boost for mixed signals
            
        # Check for specific hybrid patterns
        hybrid_patterns = [
            r'latest.*(?:and|relationship|mechanism|pathway|relate|understanding)',
            r'current.*(?:and|how.*relate|mechanism|understanding|approaches)',
            r'recent.*(?:and|impact|relationship|connection|how.*relate)',
            r'new.*(?:and|how.*affect|relate|impact|understanding)',
            r'(?:latest|current|recent).*(?:discoveries|advances).*(?:how|relate|mechanism|pathway)'
        ]
        
        is_hybrid_query = False
        for pattern in hybrid_patterns:
            if re.search(pattern, query_text.lower()):
                routing_scores[RoutingDecision.HYBRID] += 0.8
                is_hybrid_query = True
                break
        
        # TEMPORAL OVERRIDE LOGIC - CRITICAL FOR ACCURACY
        if temporal_score > 1.5 and not is_hybrid_query:
            # VERY STRONG temporal signals should heavily favor PERPLEXITY regardless of category
            routing_scores[RoutingDecision.PERPLEXITY] += min(temporal_score * 0.15, 0.9)  # Strong temporal boost
            # Reduce LIGHTRAG score when temporal signals are strong
            routing_scores[RoutingDecision.LIGHTRAG] = max(0, routing_scores[RoutingDecision.LIGHTRAG] - 0.3)
            
            # If temporal score is very high, force PERPLEXITY routing
            if temporal_score > 4.0:
                routing_scores[RoutingDecision.PERPLEXITY] = 0.9
                routing_scores[RoutingDecision.LIGHTRAG] = 0.1
                routing_scores[RoutingDecision.EITHER] = 0.2
                routing_scores[RoutingDecision.HYBRID] = 0.3
        
        # Knowledge graph signals influence (kg_confidence already defined above)
        if kg_confidence > 0.4 and not is_hybrid_query:
            routing_scores[RoutingDecision.LIGHTRAG] += kg_confidence * 0.5
        
        # Complex queries might benefit from hybrid
        if len(analysis_results.get('knowledge_indicators', [])) > 5:
            routing_scores[RoutingDecision.HYBRID] += 0.4
        
        # Convert to list of alternatives
        for decision, score in routing_scores.items():
            alternatives.append((decision, min(score, 1.0)))
        
        # Sort by confidence (highest first)
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return alternatives
    
    # ============================================================================
    # FALLBACK STRATEGY AND CIRCUIT BREAKER METHODS
    # ============================================================================
    
    def _determine_routing_with_fallback(self, analysis_results: Dict[str, Any], 
                                       confidence_metrics: ConfidenceMetrics) -> Tuple[RoutingDecision, List[str], Optional[FallbackStrategy]]:
        """
        Determine routing decision with fallback strategies based on confidence levels.
        
        Args:
            analysis_results: Results from comprehensive analysis
            confidence_metrics: Calculated confidence metrics
            
        Returns:
            Tuple of (final_routing, reasoning, fallback_strategy)
        """
        reasoning = []
        fallback_strategy = None
        
        # Get primary routing recommendation from alternatives
        alternatives = confidence_metrics.alternative_interpretations
        if not alternatives:
            # Fallback to default routing
            final_routing = RoutingDecision.EITHER
            reasoning.append("No clear routing alternatives found - defaulting to flexible routing")
            fallback_strategy = self.fallback_strategies['default']
            return final_routing, reasoning, fallback_strategy
        
        primary_routing, primary_confidence = alternatives[0]
        overall_confidence = confidence_metrics.overall_confidence
        
        # Add reasoning based on analysis
        category_prediction = analysis_results['category_prediction']
        reasoning.append(f"Research category: {category_prediction.category.value} (conf: {category_prediction.confidence:.3f})")
        
        # Temporal analysis reasoning
        temporal_analysis = analysis_results['temporal_analysis']
        if temporal_analysis.get('temporal_score', 0) > 2.0:
            reasoning.append("Strong temporal indicators detected - real-time information preferred")
        if temporal_analysis.get('established_score', 0) > 2.0:
            reasoning.append("Established knowledge patterns detected - knowledge graph preferred")
        
        # Signal strength reasoning
        signal_strength = analysis_results['signal_strength']
        if signal_strength['signal_quality_score'] > 0.7:
            reasoning.append("High signal quality detected")
        elif signal_strength['signal_quality_score'] < 0.3:
            reasoning.append("Low signal quality - may need fallback support")
        
        # Ambiguity and conflict reasoning
        ambiguity_analysis = analysis_results['ambiguity_analysis']
        if ambiguity_analysis['ambiguity_score'] > 0.5:
            reasoning.append("High query ambiguity detected - reducing confidence")
        if ambiguity_analysis['conflict_score'] > 0.3:
            reasoning.append("Signal conflicts detected - may need hybrid approach")
        
        # Apply more aggressive routing strategies to meet accuracy targets
        if overall_confidence >= self.routing_thresholds['high_confidence']:
            # High confidence - use primary routing
            final_routing = primary_routing
            reasoning.append(f"High confidence ({overall_confidence:.3f}) - routing to {primary_routing.value}")
        
        elif overall_confidence >= self.routing_thresholds['medium_confidence']:
            # Medium confidence - use primary routing directly (more aggressive)
            final_routing = primary_routing
            reasoning.append(f"Medium confidence ({overall_confidence:.3f}) - routing to {primary_routing.value}")
        
        elif overall_confidence >= self.routing_thresholds['low_confidence']:
            # Low confidence - still prefer primary routing over fallbacks for better accuracy
            # Check if we have strong category preference or biomedical signals
            category_conf = analysis_results['category_prediction'].confidence
            biomedical_entities = signal_strength['biomedical_entity_count']
            
            if (category_conf > 0.5 or biomedical_entities > 0 or 
                primary_routing in [RoutingDecision.LIGHTRAG, RoutingDecision.PERPLEXITY]):
                # Use primary routing if we have reasonable signals
                final_routing = primary_routing
                reasoning.append(f"Low confidence ({overall_confidence:.3f}) but good signals - routing to {primary_routing.value}")
            else:
                # Use hybrid as fallback only when signals are very weak
                final_routing = RoutingDecision.HYBRID
                fallback_strategy = self.fallback_strategies['hybrid']
                reasoning.append(f"Low confidence ({overall_confidence:.3f}) with weak signals - using hybrid fallback")
        
        else:
            # Very low confidence - but still try to route intelligently
            category_conf = analysis_results['category_prediction'].confidence
            # Check for signals even if category confidence is low
            kg_detection = analysis_results.get('kg_detection', {})
            kg_confidence = kg_detection.get('confidence', 0.0)
            temporal_analysis = analysis_results.get('temporal_analysis', {})
            temporal_score = temporal_analysis.get('temporal_score', 0.0)
            
            if (category_conf > 0.3 or kg_confidence > 0.5 or temporal_score > 2.0):  # If we have any strong signals
                final_routing = primary_routing
                reasoning.append(f"Very low confidence ({overall_confidence:.3f}) but signals present (cat:{category_conf:.2f}, kg:{kg_confidence:.2f}, temp:{temporal_score:.1f}) - routing to {primary_routing.value}")
            else:
                # Only fall back to EITHER for truly ambiguous queries
                final_routing = RoutingDecision.EITHER
                fallback_strategy = self.fallback_strategies['default']
                reasoning.append(f"Very low confidence ({overall_confidence:.3f}) with no clear signals - using safe default routing")
        
        return final_routing, reasoning, fallback_strategy
    
    def _should_circuit_break(self) -> bool:
        """Check if circuit breaker should be triggered."""
        current_time = time.time()
        state = self._circuit_breaker_state
        
        if state['state'] == 'open':
            # Check if recovery time has passed
            if current_time - state['last_failure_time'] > self.fallback_strategies['circuit_breaker'].parameters['recovery_time']:
                state['state'] = 'half_open'
                self.logger.info("Circuit breaker entering half-open state")
                return False
            return True
        
        return False
    
    def _handle_routing_failure(self, error: Exception, query_text: str) -> None:
        """Handle routing failures and update circuit breaker state."""
        current_time = time.time()
        state = self._circuit_breaker_state
        
        state['failures'] += 1
        state['last_failure_time'] = current_time
        
        failure_threshold = self.fallback_strategies['circuit_breaker'].parameters['failure_threshold']
        
        if state['failures'] >= failure_threshold:
            state['state'] = 'open'
            self.logger.error(f"Circuit breaker opened after {state['failures']} failures. "
                             f"Last error: {str(error)} for query: {query_text[:100]}...")
        else:
            self.logger.warning(f"Routing failure ({state['failures']}/{failure_threshold}): "
                               f"{str(error)} for query: {query_text[:50]}...")
    
    def _create_circuit_breaker_response(self, query_text: str, start_time: float) -> RoutingPrediction:
        """Create response when circuit breaker is open."""
        total_time = (time.time() - start_time) * 1000
        
        # Create minimal confidence metrics for circuit breaker response
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.1,
            research_category_confidence=0.1,
            temporal_analysis_confidence=0.0,
            signal_strength_confidence=0.0,
            context_coherence_confidence=0.0,
            keyword_density=0.0,
            pattern_match_strength=0.0,
            biomedical_entity_count=0,
            ambiguity_score=1.0,
            conflict_score=0.0,
            alternative_interpretations=[(RoutingDecision.EITHER, 0.1)],
            calculation_time_ms=total_time
        )
        
        fallback_strategy = self.fallback_strategies['circuit_breaker']
        
        return RoutingPrediction(
            routing_decision=RoutingDecision.EITHER,
            confidence=0.1,
            reasoning=["Circuit breaker open - using safe default routing"],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            fallback_strategy=fallback_strategy,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={'circuit_breaker_active': True, 'routing_time_ms': total_time}
        )
    
    def _create_fallback_response(self, query_text: str, start_time: float, error_message: str) -> RoutingPrediction:
        """Create fallback response when routing fails."""
        total_time = (time.time() - start_time) * 1000
        
        # Create minimal confidence metrics for error response
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.05,
            research_category_confidence=0.05,
            temporal_analysis_confidence=0.0,
            signal_strength_confidence=0.0,
            context_coherence_confidence=0.0,
            keyword_density=0.0,
            pattern_match_strength=0.0,
            biomedical_entity_count=0,
            ambiguity_score=1.0,
            conflict_score=0.0,
            alternative_interpretations=[(RoutingDecision.EITHER, 0.05)],
            calculation_time_ms=total_time
        )
        
        fallback_strategy = self.fallback_strategies['default']
        
        return RoutingPrediction(
            routing_decision=RoutingDecision.EITHER,
            confidence=0.05,
            reasoning=[f"Routing failed: {error_message} - using emergency fallback"],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            fallback_strategy=fallback_strategy,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={'routing_error': error_message, 'routing_time_ms': total_time}
        )
    
    def _serialize_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize analysis results for metadata storage."""
        serialized = {}
        
        for key, value in analysis_results.items():
            if key == 'category_prediction':
                serialized[key] = value.to_dict()
            elif isinstance(value, dict):
                serialized[key] = value
            elif hasattr(value, 'to_dict'):
                serialized[key] = value.to_dict()
            else:
                serialized[key] = str(value)
        
        return serialized
    
    # ============================================================================
    # CONFIDENCE VALIDATION AND MONITORING METHODS
    # ============================================================================
    
    def validate_confidence_calculation(self, query_text: str, 
                                      expected_confidence_range: Optional[Tuple[float, float]] = None,
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate confidence calculation for a query with detailed diagnostics.
        
        Args:
            query_text: Query to validate confidence for
            expected_confidence_range: Optional expected confidence range (min, max)
            context: Optional context for validation
            
        Returns:
            Dict containing validation results and diagnostics
        """
        start_time = time.time()
        
        # Get routing prediction with detailed metrics
        prediction = self.route_query(query_text, context)
        
        validation = {
            'query': query_text,
            'predicted_confidence': prediction.confidence,
            'confidence_level': prediction.confidence_level,
            'routing_decision': prediction.routing_decision.value,
            'validation_passed': True,
            'issues': [],
            'diagnostics': {},
            'performance_metrics': {}
        }
        
        # Validate confidence range
        if expected_confidence_range:
            min_conf, max_conf = expected_confidence_range
            if not (min_conf <= prediction.confidence <= max_conf):
                validation['validation_passed'] = False
                validation['issues'].append(
                    f"Confidence {prediction.confidence:.3f} outside expected range [{min_conf:.3f}, {max_conf:.3f}]"
                )
        
        # Validate confidence consistency
        metrics = prediction.confidence_metrics
        component_avg = (
            metrics.research_category_confidence + 
            metrics.temporal_analysis_confidence + 
            metrics.signal_strength_confidence + 
            metrics.context_coherence_confidence
        ) / 4.0
        
        if abs(metrics.overall_confidence - component_avg) > 0.3:
            validation['issues'].append(
                f"Large discrepancy between overall confidence ({metrics.overall_confidence:.3f}) "
                f"and component average ({component_avg:.3f})"
            )
        
        # Validate performance
        if metrics.calculation_time_ms > 50:
            validation['issues'].append(
                f"Confidence calculation took {metrics.calculation_time_ms:.2f}ms (target: 50ms)"
            )
        
        # Diagnostic information
        validation['diagnostics'] = {
            'component_confidences': {
                'research_category': metrics.research_category_confidence,
                'temporal_analysis': metrics.temporal_analysis_confidence,
                'signal_strength': metrics.signal_strength_confidence,
                'context_coherence': metrics.context_coherence_confidence
            },
            'signal_metrics': {
                'keyword_density': metrics.keyword_density,
                'pattern_match_strength': metrics.pattern_match_strength,
                'biomedical_entity_count': metrics.biomedical_entity_count
            },
            'uncertainty_metrics': {
                'ambiguity_score': metrics.ambiguity_score,
                'conflict_score': metrics.conflict_score,
                'alternative_count': len(metrics.alternative_interpretations)
            },
            'reasoning': prediction.reasoning
        }
        
        # Performance metrics
        validation['performance_metrics'] = {
            'confidence_calculation_time_ms': metrics.calculation_time_ms,
            'total_validation_time_ms': (time.time() - start_time) * 1000,
            'memory_efficient': len(validation['diagnostics']) < 50  # Simple heuristic
        }
        
        return validation
    
    def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get comprehensive confidence scoring statistics."""
        base_stats = self.get_routing_statistics()
        
        # Add confidence-specific statistics
        confidence_stats = {
            'fallback_strategies': {
                strategy_name: {
                    'strategy_type': strategy.strategy_type,
                    'confidence_threshold': strategy.confidence_threshold,
                    'description': strategy.description
                }
                for strategy_name, strategy in self.fallback_strategies.items()
            },
            'circuit_breaker_state': self._circuit_breaker_state.copy(),
            'confidence_thresholds': self.routing_thresholds,
            'performance_targets': {
                'total_routing_time_ms': 50,
                'comprehensive_analysis_time_ms': 30,
                'confidence_calculation_time_ms': 20
            }
        }
        
        base_stats.update(confidence_stats)
        return base_stats