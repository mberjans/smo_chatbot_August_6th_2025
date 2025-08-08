"""
Enhanced Query Routing Integration Example

This module demonstrates how to integrate the new Query Classification System 
(CMO-LIGHTRAG-012-T04) with the existing BiomedicalQueryRouter to create a 
more robust and accurate routing system for the Clinical Metabolomics Oracle.

Features:
- Dual-layer routing with both classification and router systems
- Confidence-based routing decisions
- Fallback strategies and conflict resolution
- Performance monitoring and optimization
- Real-world clinical metabolomics routing scenarios

Author: Claude Code (Anthropic)
Created: August 8, 2025
Version: 1.0.0
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

# Import the new classification system
from query_classification_system import (
    QueryClassificationCategories,
    QueryClassificationEngine,
    ClassificationResult,
    create_classification_engine,
    get_routing_category_mapping
)

# Import existing routing system
from query_router import (
    BiomedicalQueryRouter,
    RoutingDecision,
    RoutingPrediction
)


class RoutingStrategy(Enum):
    """Routing strategy options for integrated system."""
    
    CLASSIFICATION_FIRST = "classification_first"    # Use classification system first
    ROUTER_FIRST = "router_first"                   # Use existing router first
    ENSEMBLE = "ensemble"                           # Use both systems and combine
    ADAPTIVE = "adaptive"                           # Choose strategy based on query characteristics


@dataclass
class IntegratedRoutingResult:
    """Result from the integrated routing system."""
    
    # Final routing decision
    final_routing: RoutingDecision
    final_confidence: float
    strategy_used: RoutingStrategy
    
    # Classification system results
    classification_result: ClassificationResult
    classification_routing: str
    
    # Router system results
    router_prediction: RoutingPrediction
    
    # Integration analysis
    systems_agree: bool
    confidence_difference: float
    resolution_method: str
    reasoning: List[str]
    
    # Performance metrics
    total_time_ms: float
    classification_time_ms: float
    router_time_ms: float
    integration_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'final_routing': self.final_routing.value,
            'final_confidence': self.final_confidence,
            'strategy_used': self.strategy_used.value,
            'classification': {
                'category': self.classification_result.category.value,
                'confidence': self.classification_result.confidence,
                'suggested_routing': self.classification_routing,
                'time_ms': self.classification_result.classification_time_ms
            },
            'router': {
                'decision': self.router_prediction.routing_decision.value,
                'confidence': self.router_prediction.confidence,
                'category': self.router_prediction.research_category.value,
                'time_ms': self.router_prediction.metadata.get('routing_time_ms', 0) if self.router_prediction.metadata else 0
            },
            'integration': {
                'systems_agree': self.systems_agree,
                'confidence_difference': self.confidence_difference,
                'resolution_method': self.resolution_method,
                'reasoning': self.reasoning
            },
            'performance': {
                'total_time_ms': self.total_time_ms,
                'classification_time_ms': self.classification_time_ms,
                'router_time_ms': self.router_time_ms,
                'integration_time_ms': self.integration_time_ms
            }
        }


class EnhancedQueryRouter:
    """Enhanced query router that integrates classification and routing systems."""
    
    def __init__(self, 
                 classification_engine: Optional[QueryClassificationEngine] = None,
                 biomedical_router: Optional[BiomedicalQueryRouter] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the enhanced router."""
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize systems
        self.classification_engine = classification_engine or create_classification_engine(self.logger)
        self.biomedical_router = biomedical_router or BiomedicalQueryRouter(self.logger)
        
        # Get routing mappings
        self.category_mapping = get_routing_category_mapping()
        self.routing_decision_mapping = {
            "lightrag": RoutingDecision.LIGHTRAG,
            "perplexity": RoutingDecision.PERPLEXITY,
            "either": RoutingDecision.EITHER
        }
        
        # Configuration
        self.confidence_threshold = 0.7  # Threshold for high confidence routing
        self.agreement_threshold = 0.3   # Threshold for considering systems in agreement
        self.default_strategy = RoutingStrategy.ADAPTIVE
        
        # Performance tracking
        self._routing_times = []
        self._agreement_rate = []
        
        self.logger.info("Enhanced query router initialized with dual-layer routing")
    
    def route_query(self, 
                   query_text: str,
                   context: Optional[Dict[str, Any]] = None,
                   strategy: Optional[RoutingStrategy] = None) -> IntegratedRoutingResult:
        """Route a query using integrated classification and router systems."""
        
        start_time = time.time()
        strategy = strategy or self.default_strategy
        
        try:
            # Run both systems
            classification_start = time.time()
            classification_result = self.classification_engine.classify_query(query_text, context)
            classification_time = (time.time() - classification_start) * 1000
            
            router_start = time.time()
            router_prediction = self.biomedical_router.route_query(query_text, context)
            router_time = (time.time() - router_start) * 1000
            
            integration_start = time.time()
            
            # Get suggested routing from classification
            classification_routing = self.category_mapping[classification_result.category]
            
            # Analyze agreement between systems
            systems_agree, confidence_diff = self._analyze_system_agreement(
                classification_result, classification_routing, router_prediction
            )
            
            # Apply routing strategy
            final_routing, final_confidence, resolution_method, reasoning = self._apply_routing_strategy(
                strategy, classification_result, classification_routing, 
                router_prediction, systems_agree, confidence_diff
            )
            
            integration_time = (time.time() - integration_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            # Create integrated result
            result = IntegratedRoutingResult(
                final_routing=final_routing,
                final_confidence=final_confidence,
                strategy_used=strategy,
                classification_result=classification_result,
                classification_routing=classification_routing,
                router_prediction=router_prediction,
                systems_agree=systems_agree,
                confidence_difference=confidence_diff,
                resolution_method=resolution_method,
                reasoning=reasoning,
                total_time_ms=total_time,
                classification_time_ms=classification_time,
                router_time_ms=router_time,
                integration_time_ms=integration_time
            )
            
            # Track performance
            self._routing_times.append(total_time)
            self._agreement_rate.append(1.0 if systems_agree else 0.0)
            
            self.logger.debug(f"Routed query to {final_routing.value} using {strategy.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Integrated routing failed: {e}")
            return self._create_fallback_result(query_text, context, start_time, str(e))
    
    def _analyze_system_agreement(self, 
                                 classification_result: ClassificationResult,
                                 classification_routing: str,
                                 router_prediction: RoutingPrediction) -> Tuple[bool, float]:
        """Analyze agreement between classification and router systems."""
        
        # Map classification routing to router decision
        classification_decision = self.routing_decision_mapping.get(
            classification_routing, RoutingDecision.EITHER
        )
        
        # Check if decisions agree
        router_decision = router_prediction.routing_decision
        
        # Direct agreement
        direct_agreement = classification_decision == router_decision
        
        # Flexible agreement (EITHER can match with anything)
        flexible_agreement = (
            classification_decision == RoutingDecision.EITHER or
            router_decision == RoutingDecision.EITHER or
            direct_agreement
        )
        
        # Calculate confidence difference
        confidence_diff = abs(classification_result.confidence - router_prediction.confidence)
        
        # Agreement if decisions match and confidence difference is small
        systems_agree = flexible_agreement and confidence_diff <= self.agreement_threshold
        
        return systems_agree, confidence_diff
    
    def _apply_routing_strategy(self, 
                              strategy: RoutingStrategy,
                              classification_result: ClassificationResult,
                              classification_routing: str,
                              router_prediction: RoutingPrediction,
                              systems_agree: bool,
                              confidence_diff: float) -> Tuple[RoutingDecision, float, str, List[str]]:
        """Apply the specified routing strategy."""
        
        reasoning = []
        
        if strategy == RoutingStrategy.CLASSIFICATION_FIRST:
            return self._classification_first_strategy(
                classification_result, classification_routing, router_prediction, reasoning
            )
        elif strategy == RoutingStrategy.ROUTER_FIRST:
            return self._router_first_strategy(
                classification_result, router_prediction, reasoning
            )
        elif strategy == RoutingStrategy.ENSEMBLE:
            return self._ensemble_strategy(
                classification_result, classification_routing, router_prediction, 
                systems_agree, confidence_diff, reasoning
            )
        else:  # ADAPTIVE
            return self._adaptive_strategy(
                classification_result, classification_routing, router_prediction,
                systems_agree, confidence_diff, reasoning
            )
    
    def _classification_first_strategy(self, 
                                     classification_result: ClassificationResult,
                                     classification_routing: str,
                                     router_prediction: RoutingPrediction,
                                     reasoning: List[str]) -> Tuple[RoutingDecision, float, str, List[str]]:
        """Use classification system first, fallback to router if low confidence."""
        
        classification_confidence = classification_result.confidence
        classification_decision = self.routing_decision_mapping[classification_routing]
        
        if classification_confidence >= self.confidence_threshold:
            reasoning.append(f"High classification confidence ({classification_confidence:.3f})")
            return classification_decision, classification_confidence, "classification_high_confidence", reasoning
        else:
            reasoning.append(f"Low classification confidence, using router")
            return router_prediction.routing_decision, router_prediction.confidence, "fallback_to_router", reasoning
    
    def _router_first_strategy(self, 
                             classification_result: ClassificationResult,
                             router_prediction: RoutingPrediction,
                             reasoning: List[str]) -> Tuple[RoutingDecision, float, str, List[str]]:
        """Use router system first, fallback to classification if low confidence."""
        
        router_confidence = router_prediction.confidence
        
        if router_confidence >= self.confidence_threshold:
            reasoning.append(f"High router confidence ({router_confidence:.3f})")
            return router_prediction.routing_decision, router_confidence, "router_high_confidence", reasoning
        else:
            classification_decision = self.routing_decision_mapping[
                self.category_mapping[classification_result.category]
            ]
            reasoning.append(f"Low router confidence, using classification")
            return classification_decision, classification_result.confidence, "fallback_to_classification", reasoning
    
    def _ensemble_strategy(self, 
                         classification_result: ClassificationResult,
                         classification_routing: str,
                         router_prediction: RoutingPrediction,
                         systems_agree: bool,
                         confidence_diff: float,
                         reasoning: List[str]) -> Tuple[RoutingDecision, float, str, List[str]]:
        """Use ensemble approach combining both systems."""
        
        classification_decision = self.routing_decision_mapping[classification_routing]
        
        if systems_agree:
            # Systems agree - use higher confidence decision
            if classification_result.confidence >= router_prediction.confidence:
                reasoning.append("Systems agree, using classification (higher confidence)")
                return classification_decision, classification_result.confidence, "ensemble_agreement", reasoning
            else:
                reasoning.append("Systems agree, using router (higher confidence)")
                return router_prediction.routing_decision, router_prediction.confidence, "ensemble_agreement", reasoning
        else:
            # Systems disagree - use most confident or safe default
            if confidence_diff > 0.3:
                if classification_result.confidence > router_prediction.confidence:
                    reasoning.append("Large confidence difference, using classification")
                    return classification_decision, classification_result.confidence, "ensemble_highest_confidence", reasoning
                else:
                    reasoning.append("Large confidence difference, using router")
                    return router_prediction.routing_decision, router_prediction.confidence, "ensemble_highest_confidence", reasoning
            else:
                reasoning.append("Systems disagree, using safe default")
                safe_confidence = min(classification_result.confidence, router_prediction.confidence)
                return RoutingDecision.EITHER, safe_confidence, "ensemble_safe_default", reasoning
    
    def _adaptive_strategy(self, 
                         classification_result: ClassificationResult,
                         classification_routing: str,
                         router_prediction: RoutingPrediction,
                         systems_agree: bool,
                         confidence_diff: float,
                         reasoning: List[str]) -> Tuple[RoutingDecision, float, str, List[str]]:
        """Use adaptive strategy based on query characteristics."""
        
        # Analyze query characteristics
        has_temporal_signals = len(classification_result.temporal_indicators) > 0
        has_biomedical_entities = len(classification_result.biomedical_entities) > 0
        
        if has_temporal_signals and classification_result.temporal_confidence > 0.7:
            reasoning.append("Strong temporal signals, favoring classification")
            classification_decision = self.routing_decision_mapping[classification_routing]
            return classification_decision, classification_result.confidence, "adaptive_temporal", reasoning
        elif has_biomedical_entities and systems_agree:
            reasoning.append("Rich biomedical content with agreement, using ensemble")
            return self._ensemble_strategy(
                classification_result, classification_routing, router_prediction,
                systems_agree, confidence_diff, reasoning
            )
        else:
            reasoning.append("Default adaptive strategy")
            return self._classification_first_strategy(
                classification_result, classification_routing, router_prediction, reasoning
            )
    
    def _create_fallback_result(self, query_text: str, context: Optional[Dict[str, Any]], 
                              start_time: float, error_message: str) -> IntegratedRoutingResult:
        """Create fallback result when integration fails."""
        
        total_time = (time.time() - start_time) * 1000
        
        # Import here to avoid circular imports
        from cost_persistence import ResearchCategory
        
        # Create minimal fallback results
        fallback_classification = ClassificationResult(
            category=QueryClassificationCategories.GENERAL,
            confidence=0.1,
            reasoning=[f"Integration failed: {error_message}"],
            keyword_match_confidence=0.0,
            pattern_match_confidence=0.0,
            semantic_confidence=0.0,
            temporal_confidence=0.0,
            matched_keywords=[],
            matched_patterns=[],
            biomedical_entities=[],
            temporal_indicators=[],
            alternative_classifications=[(QueryClassificationCategories.GENERAL, 0.1)],
            classification_time_ms=0.0,
            ambiguity_score=1.0,
            conflict_score=0.0
        )
        
        fallback_router = RoutingPrediction(
            routing_decision=RoutingDecision.EITHER,
            confidence=0.1,
            reasoning=[f"Integration failed: {error_message}"],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=None,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={'routing_time_ms': 0.0}
        )
        
        return IntegratedRoutingResult(
            final_routing=RoutingDecision.EITHER,
            final_confidence=0.1,
            strategy_used=RoutingStrategy.ADAPTIVE,
            classification_result=fallback_classification,
            classification_routing="either",
            router_prediction=fallback_router,
            systems_agree=True,
            confidence_difference=0.0,
            resolution_method="fallback_on_error",
            reasoning=[f"Integration failed: {error_message}"],
            total_time_ms=total_time,
            classification_time_ms=0.0,
            router_time_ms=0.0,
            integration_time_ms=0.0
        )
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        
        avg_time = sum(self._routing_times) / len(self._routing_times) if self._routing_times else 0
        agreement_rate = sum(self._agreement_rate) / len(self._agreement_rate) if self._agreement_rate else 0
        
        return {
            'total_routes': len(self._routing_times),
            'average_routing_time_ms': avg_time,
            'max_routing_time_ms': max(self._routing_times) if self._routing_times else 0,
            'system_agreement_rate': agreement_rate,
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'agreement_threshold': self.agreement_threshold,
                'default_strategy': self.default_strategy.value
            }
        }


def demonstrate_integrated_routing():
    """Demonstrate the integrated routing system."""
    print("\nIntegrated Query Routing System Demonstration")
    print("=" * 60)
    
    # Create enhanced router
    logger = logging.getLogger('demo.routing')
    logger.setLevel(logging.INFO)
    
    enhanced_router = EnhancedQueryRouter(logger=logger)
    
    # Test queries
    test_queries = [
        "What is the relationship between glucose metabolism and diabetes?",
        "Latest FDA approvals for metabolomics diagnostics in 2024",
        "What is the definition of metabolomics?",
        "Metabolomics research analysis",
    ]
    
    strategies = [RoutingStrategy.ADAPTIVE, RoutingStrategy.ENSEMBLE]
    
    print(f"\nTesting {len(test_queries)} queries:\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        
        for strategy in strategies:
            result = enhanced_router.route_query(query, strategy=strategy)
            
            agreement_status = "✓ AGREE" if result.systems_agree else "✗ DISAGREE"
            
            print(f"  {strategy.value:15s}: {result.final_routing.value:10s} "
                  f"(conf: {result.final_confidence:.3f}, "
                  f"time: {result.total_time_ms:.1f}ms, "
                  f"{agreement_status})")
        
        print("-" * 60)
    
    # Show routing statistics
    stats = enhanced_router.get_routing_statistics()
    print(f"\nRouting Statistics:")
    print(f"Total Routes: {stats['total_routes']}")
    print(f"Average Time: {stats['average_routing_time_ms']:.1f}ms")
    print(f"Agreement Rate: {stats['system_agreement_rate']:.1%}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demonstrate_integrated_routing()