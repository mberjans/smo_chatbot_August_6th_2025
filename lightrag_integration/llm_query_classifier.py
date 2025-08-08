"""
LLM-Powered Query Classifier for Clinical Metabolomics Oracle

This module provides an LLM-based semantic classifier that integrates with the existing
query routing infrastructure while adding enhanced semantic understanding capabilities.

Key Features:
    - Seamless integration with existing BiomedicalQueryRouter
    - Fallback to keyword-based classification if LLM fails
    - Performance monitoring and adaptive prompt selection
    - Cost optimization through caching and smart prompt routing
    - Real-time confidence validation and adjustment

Classes:
    - LLMQueryClassifier: Main LLM-powered classification engine
    - ClassificationCache: Intelligent caching system for classifications
    - PerformanceMonitor: Tracks classification accuracy and response times
"""

import json
import time
import hashlib
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import openai
from openai import AsyncOpenAI

from .llm_classification_prompts import (
    LLMClassificationPrompts,
    ClassificationCategory,
    ClassificationResult,
    CLASSIFICATION_RESULT_SCHEMA
)
from .query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction
from .research_categorizer import CategoryPrediction
from .cost_persistence import ResearchCategory


class LLMProvider(Enum):
    """Supported LLM providers for classification."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMClassificationConfig:
    """Configuration for LLM-based classification."""
    
    # LLM provider settings
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o-mini"  # Fast, cost-effective model
    api_key: Optional[str] = None
    max_tokens: int = 200
    temperature: float = 0.1  # Low temperature for consistent classification
    
    # Performance settings
    timeout_seconds: float = 3.0  # Maximum time to wait for LLM response
    max_retries: int = 2
    fallback_to_keywords: bool = True
    
    # Prompt strategy
    use_examples_for_uncertain: bool = True  # Use examples when keyword confidence is low
    primary_confidence_threshold: float = 0.7  # Below this, use examples
    validation_threshold: float = 0.5  # Below this, validate classification
    
    # Caching settings
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    max_cache_size: int = 1000
    
    # Cost optimization
    daily_api_budget: float = 5.0  # Daily budget in USD
    cost_per_1k_tokens: float = 0.0005  # Approximate cost for gpt-4o-mini


@dataclass
class ClassificationMetrics:
    """Metrics for tracking classification performance."""
    
    total_classifications: int = 0
    llm_successful: int = 0
    llm_failures: int = 0
    fallback_used: int = 0
    cache_hits: int = 0
    
    avg_response_time_ms: float = 0.0
    avg_confidence_score: float = 0.0
    
    daily_api_cost: float = 0.0
    daily_token_usage: int = 0
    
    last_reset_date: Optional[str] = None


class ClassificationCache:
    """Intelligent caching system for query classifications."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._access_order = deque()
    
    def _get_cache_key(self, query_text: str) -> str:
        """Generate cache key from query text."""
        return hashlib.md5(query_text.lower().strip().encode()).hexdigest()
    
    def get(self, query_text: str) -> Optional[ClassificationResult]:
        """Get cached classification if available and not expired."""
        cache_key = self._get_cache_key(query_text)
        
        if cache_key not in self._cache:
            return None
        
        cached_data = self._cache[cache_key]
        cached_time = cached_data.get('timestamp', 0)
        
        # Check if cache entry has expired
        if time.time() - cached_time > (self.ttl_hours * 3600):
            self._remove_cache_entry(cache_key)
            return None
        
        # Update access tracking
        self._access_times[cache_key] = time.time()
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        self._access_order.append(cache_key)
        
        return ClassificationResult(**cached_data['result'])
    
    def put(self, query_text: str, result: ClassificationResult) -> None:
        """Cache a classification result."""
        cache_key = self._get_cache_key(query_text)
        
        # Ensure cache size limit
        while len(self._cache) >= self.max_size:
            self._evict_oldest_entry()
        
        self._cache[cache_key] = {
            'result': asdict(result),
            'timestamp': time.time(),
            'query_text': query_text
        }
        
        self._access_times[cache_key] = time.time()
        self._access_order.append(cache_key)
    
    def _evict_oldest_entry(self) -> None:
        """Evict the least recently used cache entry."""
        if self._access_order:
            oldest_key = self._access_order.popleft()
            self._remove_cache_entry(oldest_key)
    
    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a cache entry completely."""
        self._cache.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'max_size': self.max_size,
            'ttl_hours': self.ttl_hours,
            'utilization': len(self._cache) / self.max_size if self.max_size > 0 else 0
        }


class LLMQueryClassifier:
    """
    LLM-powered semantic query classifier for the Clinical Metabolomics Oracle.
    
    This classifier enhances the existing keyword-based system with semantic understanding
    while maintaining performance and cost efficiency through intelligent caching and
    fallback mechanisms.
    """
    
    def __init__(self, 
                 config: Optional[LLMClassificationConfig] = None,
                 biomedical_router: Optional[BiomedicalQueryRouter] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the LLM query classifier.
        
        Args:
            config: Configuration for LLM classification
            biomedical_router: Existing biomedical router for fallback
            logger: Logger instance
        """
        self.config = config or LLMClassificationConfig()
        self.biomedical_router = biomedical_router
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize LLM client
        self._init_llm_client()
        
        # Initialize caching and monitoring
        self.cache = ClassificationCache(
            max_size=self.config.max_cache_size,
            ttl_hours=self.config.cache_ttl_hours
        ) if self.config.enable_caching else None
        
        self.metrics = ClassificationMetrics()
        self._reset_daily_metrics_if_needed()
        
        # Performance tracking
        self._response_times = deque(maxlen=100)  # Last 100 response times
        self._confidence_scores = deque(maxlen=100)  # Last 100 confidence scores
        
        self.logger.info(f"LLM Query Classifier initialized with {self.config.provider.value} provider")
    
    def _init_llm_client(self) -> None:
        """Initialize the LLM client based on provider configuration."""
        if self.config.provider == LLMProvider.OPENAI:
            if not self.config.api_key:
                raise ValueError("OpenAI API key is required")
            
            self.llm_client = AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds
            )
        else:
            raise NotImplementedError(f"Provider {self.config.provider.value} not yet implemented")
    
    async def classify_query(self, 
                           query_text: str,
                           context: Optional[Dict[str, Any]] = None,
                           force_llm: bool = False) -> Tuple[ClassificationResult, bool]:
        """
        Classify a query using LLM with intelligent fallback strategies.
        
        Args:
            query_text: The query text to classify
            context: Optional context information
            force_llm: If True, skip cache and force LLM classification
            
        Returns:
            Tuple of (ClassificationResult, used_llm: bool)
        """
        start_time = time.time()
        self.metrics.total_classifications += 1
        
        try:
            # Check cache first (unless forced to use LLM)
            if not force_llm and self.cache:
                cached_result = self.cache.get(query_text)
                if cached_result:
                    self.metrics.cache_hits += 1
                    self.logger.debug(f"Cache hit for query: {query_text[:50]}...")
                    return cached_result, False
            
            # Check daily budget before making API call
            if self.metrics.daily_api_cost >= self.config.daily_api_budget:
                self.logger.warning("Daily API budget exceeded, falling back to keyword classification")
                return await self._fallback_classification(query_text, context), False
            
            # Decide which prompt strategy to use
            use_examples = False
            if self.biomedical_router and self.config.use_examples_for_uncertain:
                # Quick keyword-based confidence check
                keyword_prediction = self.biomedical_router.route_query(query_text, context)
                if keyword_prediction.confidence < self.config.primary_confidence_threshold:
                    use_examples = True
                    self.logger.debug("Using examples due to low keyword confidence")
            
            # Attempt LLM classification
            llm_result = await self._classify_with_llm(query_text, use_examples)
            
            if llm_result:
                self.metrics.llm_successful += 1
                
                # Update performance metrics
                response_time = (time.time() - start_time) * 1000
                self._response_times.append(response_time)
                self._confidence_scores.append(llm_result.confidence)
                self._update_avg_metrics()
                
                # Cache successful result
                if self.cache:
                    self.cache.put(query_text, llm_result)
                
                # Validate if confidence is low
                if llm_result.confidence < self.config.validation_threshold:
                    validated_result = await self._validate_classification(
                        query_text, llm_result
                    )
                    if validated_result:
                        llm_result = validated_result
                
                self.logger.debug(f"LLM classification successful: {llm_result.category} "
                                f"(confidence: {llm_result.confidence:.3f})")
                return llm_result, True
            
        except Exception as e:
            self.logger.error(f"LLM classification failed: {str(e)}")
            self.metrics.llm_failures += 1
        
        # Fallback to keyword-based classification
        fallback_result = await self._fallback_classification(query_text, context)
        return fallback_result, False
    
    async def _classify_with_llm(self, 
                                query_text: str,
                                use_examples: bool = False) -> Optional[ClassificationResult]:
        """Perform LLM-based classification with retry logic."""
        
        # Build appropriate prompt
        if use_examples:
            prompt = LLMClassificationPrompts.build_primary_prompt(query_text, include_examples=True)
        else:
            prompt = LLMClassificationPrompts.build_primary_prompt(query_text)
        
        # Estimate and track token usage
        token_estimate = LLMClassificationPrompts.estimate_token_usage(query_text, use_examples)
        estimated_cost = (token_estimate["primary_prompt_tokens"] + 
                         token_estimate["estimated_response_tokens"]) * self.config.cost_per_1k_tokens / 1000
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Make API call
                response = await self.llm_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout_seconds
                )
                
                # Update cost tracking
                self.metrics.daily_api_cost += estimated_cost
                self.metrics.daily_token_usage += (
                    response.usage.prompt_tokens + response.usage.completion_tokens
                    if response.usage else token_estimate["primary_prompt_tokens"] + 
                    token_estimate["estimated_response_tokens"]
                )
                
                # Parse JSON response
                response_text = response.choices[0].message.content.strip()
                result_data = json.loads(response_text)
                
                # Validate against schema and convert to ClassificationResult
                return self._validate_and_convert_result(result_data)
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Invalid JSON response on attempt {attempt + 1}: {str(e)}")
                if attempt == self.config.max_retries:
                    return None
                
            except Exception as e:
                self.logger.warning(f"LLM API error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.config.max_retries:
                    return None
                
                # Wait before retry
                await asyncio.sleep(0.5 * (attempt + 1))
        
        return None
    
    def _validate_and_convert_result(self, result_data: Dict[str, Any]) -> ClassificationResult:
        """Validate LLM response against schema and convert to ClassificationResult."""
        
        # Basic validation
        required_fields = ["category", "confidence", "reasoning"]
        for field in required_fields:
            if field not in result_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate category
        valid_categories = ["KNOWLEDGE_GRAPH", "REAL_TIME", "GENERAL"]
        if result_data["category"] not in valid_categories:
            raise ValueError(f"Invalid category: {result_data['category']}")
        
        # Validate confidence range
        confidence = result_data["confidence"]
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {confidence}")
        
        # Provide defaults for optional fields
        result_data.setdefault("alternative_categories", [])
        result_data.setdefault("uncertainty_indicators", [])
        result_data.setdefault("biomedical_signals", {
            "entities": [], "relationships": [], "techniques": []
        })
        result_data.setdefault("temporal_signals", {
            "keywords": [], "patterns": [], "years": []
        })
        
        return ClassificationResult(**result_data)
    
    async def _validate_classification(self, 
                                     query_text: str,
                                     classification: ClassificationResult) -> Optional[ClassificationResult]:
        """Validate a low-confidence classification using the validation prompt."""
        
        try:
            validation_prompt = LLMClassificationPrompts.build_validation_prompt(
                query_text,
                classification.category,
                classification.confidence,
                classification.reasoning
            )
            
            response = await self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": validation_prompt}],
                max_tokens=150,
                temperature=0.1,
                timeout=self.config.timeout_seconds
            )
            
            validation_result = json.loads(response.choices[0].message.content.strip())
            
            # Apply validation adjustments
            if validation_result.get("confidence_adjustment"):
                classification.confidence += validation_result["confidence_adjustment"]
                classification.confidence = max(0.0, min(1.0, classification.confidence))
            
            if validation_result.get("final_reasoning"):
                classification.reasoning = validation_result["final_reasoning"]
            
            self.logger.debug(f"Classification validated with score: {validation_result.get('validation_score', 0)}")
            return classification
            
        except Exception as e:
            self.logger.warning(f"Classification validation failed: {str(e)}")
            return None
    
    async def _fallback_classification(self, 
                                     query_text: str,
                                     context: Optional[Dict[str, Any]]) -> ClassificationResult:
        """Fallback to keyword-based classification when LLM fails."""
        
        self.metrics.fallback_used += 1
        
        if self.biomedical_router:
            # Use existing biomedical router
            routing_prediction = self.biomedical_router.route_query(query_text, context)
            
            # Convert RoutingDecision to ClassificationCategory
            category_mapping = {
                RoutingDecision.LIGHTRAG: "KNOWLEDGE_GRAPH",
                RoutingDecision.PERPLEXITY: "REAL_TIME",
                RoutingDecision.EITHER: "GENERAL",
                RoutingDecision.HYBRID: "GENERAL"
            }
            
            category = category_mapping.get(routing_prediction.routing_decision, "GENERAL")
            
            # Extract biomedical and temporal signals from the routing prediction
            biomedical_signals = {
                "entities": routing_prediction.knowledge_indicators or [],
                "relationships": [r for r in routing_prediction.reasoning if "relationship" in r.lower()],
                "techniques": []
            }
            
            temporal_signals = {
                "keywords": routing_prediction.temporal_indicators or [],
                "patterns": [],
                "years": []
            }
            
            return ClassificationResult(
                category=category,
                confidence=routing_prediction.confidence,
                reasoning=f"Fallback classification: {', '.join(routing_prediction.reasoning[:2])}",
                alternative_categories=[],
                uncertainty_indicators=["fallback_classification_used"],
                biomedical_signals=biomedical_signals,
                temporal_signals=temporal_signals
            )
        else:
            # Simple fallback based on basic patterns
            query_lower = query_text.lower()
            
            # Check for temporal indicators
            temporal_keywords = ["latest", "recent", "current", "2024", "2025", "new", "breaking"]
            if any(keyword in query_lower for keyword in temporal_keywords):
                category = "REAL_TIME"
                confidence = 0.6
                reasoning = "Simple fallback: temporal keywords detected"
            
            # Check for relationship patterns
            elif any(pattern in query_lower for pattern in ["relationship", "connection", "pathway", "mechanism"]):
                category = "KNOWLEDGE_GRAPH"
                confidence = 0.6
                reasoning = "Simple fallback: relationship patterns detected"
            
            # Default to general
            else:
                category = "GENERAL"
                confidence = 0.4
                reasoning = "Simple fallback: no specific patterns detected"
            
            return ClassificationResult(
                category=category,
                confidence=confidence,
                reasoning=reasoning,
                alternative_categories=[],
                uncertainty_indicators=["simple_fallback_used", "low_confidence"],
                biomedical_signals={"entities": [], "relationships": [], "techniques": []},
                temporal_signals={"keywords": [], "patterns": [], "years": []}
            )
    
    def _reset_daily_metrics_if_needed(self) -> None:
        """Reset daily metrics if it's a new day."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if self.metrics.last_reset_date != today:
            self.metrics.daily_api_cost = 0.0
            self.metrics.daily_token_usage = 0
            self.metrics.last_reset_date = today
    
    def _update_avg_metrics(self) -> None:
        """Update running average metrics."""
        if self._response_times:
            self.metrics.avg_response_time_ms = sum(self._response_times) / len(self._response_times)
        
        if self._confidence_scores:
            self.metrics.avg_confidence_score = sum(self._confidence_scores) / len(self._confidence_scores)
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive classification statistics."""
        
        self._reset_daily_metrics_if_needed()
        
        stats = {
            "classification_metrics": {
                "total_classifications": self.metrics.total_classifications,
                "llm_successful": self.metrics.llm_successful,
                "llm_failures": self.metrics.llm_failures,
                "fallback_used": self.metrics.fallback_used,
                "cache_hits": self.metrics.cache_hits,
                "success_rate": (self.metrics.llm_successful / max(1, self.metrics.total_classifications)) * 100
            },
            "performance_metrics": {
                "avg_response_time_ms": self.metrics.avg_response_time_ms,
                "avg_confidence_score": self.metrics.avg_confidence_score,
                "recent_response_times": list(self._response_times)[-10:] if self._response_times else [],
                "recent_confidence_scores": list(self._confidence_scores)[-10:] if self._confidence_scores else []
            },
            "cost_metrics": {
                "daily_api_cost": self.metrics.daily_api_cost,
                "daily_budget": self.config.daily_api_budget,
                "budget_utilization": (self.metrics.daily_api_cost / self.config.daily_api_budget) * 100,
                "daily_token_usage": self.metrics.daily_token_usage,
                "estimated_cost_per_classification": (
                    self.metrics.daily_api_cost / max(1, self.metrics.total_classifications)
                )
            },
            "cache_stats": self.cache.get_stats() if self.cache else {},
            "configuration": {
                "provider": self.config.provider.value,
                "model_name": self.config.model_name,
                "timeout_seconds": self.config.timeout_seconds,
                "enable_caching": self.config.enable_caching,
                "fallback_to_keywords": self.config.fallback_to_keywords
            }
        }
        
        return stats
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """
        Analyze performance and suggest configuration optimizations.
        
        Returns:
            Dict with optimization recommendations
        """
        stats = self.get_classification_statistics()
        recommendations = []
        
        # Check response time performance
        avg_response_time = stats["performance_metrics"]["avg_response_time_ms"]
        if avg_response_time > 2000:  # Target <2s
            recommendations.append({
                "type": "performance",
                "issue": f"Average response time ({avg_response_time:.0f}ms) exceeds 2s target",
                "suggestion": "Consider using fallback prompt more frequently or switching to faster model"
            })
        
        # Check success rate
        success_rate = stats["classification_metrics"]["success_rate"]
        if success_rate < 90:
            recommendations.append({
                "type": "reliability",
                "issue": f"LLM success rate ({success_rate:.1f}%) below 90%",
                "suggestion": "Increase timeout_seconds or max_retries, check API key and network connectivity"
            })
        
        # Check cost efficiency
        budget_utilization = stats["cost_metrics"]["budget_utilization"]
        if budget_utilization > 80:
            recommendations.append({
                "type": "cost",
                "issue": f"Daily budget utilization ({budget_utilization:.1f}%) high",
                "suggestion": "Increase cache_ttl_hours, use fallback more frequently, or increase daily_api_budget"
            })
        
        # Check cache efficiency
        if self.cache and stats["cache_stats"]["utilization"] < 0.5:
            recommendations.append({
                "type": "cache",
                "issue": f"Cache utilization ({stats['cache_stats']['utilization']*100:.1f}%) low",
                "suggestion": "Consider reducing max_cache_size or increasing cache_ttl_hours"
            })
        
        return {
            "current_performance": stats,
            "recommendations": recommendations,
            "overall_health": "good" if len(recommendations) <= 1 else "needs_attention"
        }


# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================

async def create_llm_enhanced_router(
    config: Optional[LLMClassificationConfig] = None,
    biomedical_router: Optional[BiomedicalQueryRouter] = None,
    logger: Optional[logging.Logger] = None
) -> LLMQueryClassifier:
    """
    Factory function to create an LLM-enhanced query router.
    
    Args:
        config: LLM classification configuration
        biomedical_router: Existing biomedical router for fallback
        logger: Logger instance
        
    Returns:
        Configured LLMQueryClassifier instance
    """
    if not config:
        config = LLMClassificationConfig()
    
    # Create biomedical router if not provided
    if not biomedical_router:
        biomedical_router = BiomedicalQueryRouter(logger)
    
    classifier = LLMQueryClassifier(config, biomedical_router, logger)
    
    if logger:
        logger.info("LLM-enhanced query router created successfully")
    
    return classifier


def convert_llm_result_to_routing_prediction(
    llm_result: ClassificationResult,
    query_text: str,
    used_llm: bool
) -> RoutingPrediction:
    """
    Convert LLM classification result to RoutingPrediction for compatibility
    with existing infrastructure.
    
    Args:
        llm_result: LLM classification result
        query_text: Original query text
        used_llm: Whether LLM was used for classification
        
    Returns:
        RoutingPrediction compatible with existing routing system
    """
    
    # Map LLM categories to routing decisions
    category_mapping = {
        "KNOWLEDGE_GRAPH": RoutingDecision.LIGHTRAG,
        "REAL_TIME": RoutingDecision.PERPLEXITY,
        "GENERAL": RoutingDecision.EITHER
    }
    
    routing_decision = category_mapping.get(llm_result.category, RoutingDecision.EITHER)
    
    # Create reasoning list
    reasoning = [llm_result.reasoning]
    if used_llm:
        reasoning.append("LLM-powered semantic classification")
    else:
        reasoning.append("Keyword-based fallback classification")
    
    # Add uncertainty indicators to reasoning
    if llm_result.uncertainty_indicators:
        reasoning.extend([f"Uncertainty: {indicator}" for indicator in llm_result.uncertainty_indicators[:2]])
    
    # Map to research category (best effort)
    research_category_mapping = {
        "KNOWLEDGE_GRAPH": ResearchCategory.KNOWLEDGE_EXTRACTION,
        "REAL_TIME": ResearchCategory.LITERATURE_SEARCH,
        "GENERAL": ResearchCategory.GENERAL_QUERY
    }
    
    research_category = research_category_mapping.get(llm_result.category, ResearchCategory.GENERAL_QUERY)
    
    # Create mock confidence metrics (simplified for compatibility)
    from .query_router import ConfidenceMetrics
    
    confidence_metrics = ConfidenceMetrics(
        overall_confidence=llm_result.confidence,
        research_category_confidence=llm_result.confidence,
        temporal_analysis_confidence=0.8 if llm_result.temporal_signals["keywords"] else 0.3,
        signal_strength_confidence=0.8 if llm_result.biomedical_signals["entities"] else 0.3,
        context_coherence_confidence=llm_result.confidence,
        keyword_density=len(llm_result.biomedical_signals["entities"]) / max(1, len(query_text.split())) * 10,
        pattern_match_strength=0.8 if llm_result.biomedical_signals["relationships"] else 0.3,
        biomedical_entity_count=len(llm_result.biomedical_signals["entities"]),
        ambiguity_score=len(llm_result.uncertainty_indicators) * 0.2,
        conflict_score=0.1 if len(llm_result.alternative_categories) > 1 else 0.0,
        alternative_interpretations=[
            (category_mapping.get(alt["category"], RoutingDecision.EITHER), alt["confidence"])
            for alt in llm_result.alternative_categories
        ],
        calculation_time_ms=50.0  # Placeholder
    )
    
    return RoutingPrediction(
        routing_decision=routing_decision,
        confidence=llm_result.confidence,
        reasoning=reasoning,
        research_category=research_category,
        confidence_metrics=confidence_metrics,
        temporal_indicators=llm_result.temporal_signals["keywords"],
        knowledge_indicators=llm_result.biomedical_signals["entities"],
        metadata={
            "llm_powered": used_llm,
            "llm_category": llm_result.category,
            "biomedical_signals": llm_result.biomedical_signals,
            "temporal_signals": llm_result.temporal_signals
        }
    )