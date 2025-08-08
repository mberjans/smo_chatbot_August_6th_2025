#!/usr/bin/env python3
"""
Enhanced Confidence Scoring Integration Example

This script demonstrates how to integrate the new enhanced confidence scoring
system with existing Clinical Metabolomics Oracle infrastructure.

Integration Examples:
1. Updating existing routing systems to use enhanced confidence
2. Migrating from basic ClassificationResult to EnhancedClassificationResult
3. Adding confidence validation to existing workflows
4. Integration with LightRAG and Perplexity API routing
5. Backward compatibility maintenance

Author: Claude Code (Anthropic)
Created: 2025-08-08
Task: CMO-LIGHTRAG-012-T06 - Add confidence scoring for classification results
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced classification components
try:
    from query_classification_system import (
        QueryClassificationCategories,
        ClassificationResult,
        EnhancedClassificationResult,
        EnhancedQueryClassificationEngine,
        create_enhanced_classification_engine,
        integrate_enhanced_classification_with_routing
    )
    
    # Import existing routing components (if available)
    try:
        from query_router import BiomedicalQueryRouter, RoutingDecision
        ROUTING_AVAILABLE = True
    except ImportError:
        logger.warning("Legacy routing components not available")
        ROUTING_AVAILABLE = False
        
except ImportError as e:
    logger.error(f"Enhanced confidence components not available: {e}")
    exit(1)


class LegacyClassificationService:
    """
    Simulated legacy classification service to demonstrate migration.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def classify_query_legacy(self, query_text: str) -> ClassificationResult:
        """Legacy classification method returning basic ClassificationResult."""
        
        # Simulate legacy classification logic
        query_lower = query_text.lower()
        
        if any(term in query_lower for term in ['latest', 'recent', '2025', 'current']):
            category = QueryClassificationCategories.REAL_TIME
            confidence = 0.7
        elif any(term in query_lower for term in ['pathway', 'mechanism', 'relationship']):
            category = QueryClassificationCategories.KNOWLEDGE_GRAPH
            confidence = 0.8
        else:
            category = QueryClassificationCategories.GENERAL
            confidence = 0.6
        
        return ClassificationResult(
            category=category,
            confidence=confidence,
            reasoning=[f"Legacy classification: {category.value}"],
            keyword_match_confidence=confidence * 0.9,
            pattern_match_confidence=confidence * 0.8,
            semantic_confidence=confidence * 0.85,
            temporal_confidence=0.9 if category == QueryClassificationCategories.REAL_TIME else 0.1,
            matched_keywords=[],
            matched_patterns=[],
            biomedical_entities=[],
            temporal_indicators=[],
            alternative_classifications=[(category, confidence)],
            classification_time_ms=50.0,
            ambiguity_score=0.3,
            conflict_score=0.1
        )


class EnhancedClassificationService:
    """
    Enhanced classification service with comprehensive confidence scoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.engine = None
        self.legacy_service = LegacyClassificationService()
        
    async def initialize(self):
        """Initialize the enhanced classification engine."""
        
        try:
            self.engine = await create_enhanced_classification_engine(
                logger=self.logger,
                enable_hybrid_confidence=True
            )
            self.logger.info("Enhanced classification service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced service: {e}")
            return False
    
    async def classify_query_enhanced(self, query_text: str, 
                                    context: Optional[Dict[str, Any]] = None) -> EnhancedClassificationResult:
        """Enhanced classification with comprehensive confidence scoring."""
        
        if not self.engine:
            raise RuntimeError("Enhanced classification engine not initialized")
        
        return await self.engine.classify_query_enhanced(query_text, context)
    
    async def classify_with_fallback(self, query_text: str,
                                   context: Optional[Dict[str, Any]] = None) -> EnhancedClassificationResult:
        """
        Classify with enhanced confidence, falling back to legacy + basic enhancement if needed.
        """
        
        try:
            # Try enhanced classification first
            if self.engine and self.engine.enable_hybrid_confidence:
                return await self.classify_query_enhanced(query_text, context)
            
        except Exception as e:
            self.logger.warning(f"Enhanced classification failed, using fallback: {e}")
        
        # Fallback to legacy classification with basic enhancement
        legacy_result = self.legacy_service.classify_query_legacy(query_text)
        enhanced_result = EnhancedClassificationResult.from_basic_classification(legacy_result)
        
        self.logger.info("Used legacy classification with basic enhancement")
        return enhanced_result
    
    def validate_classification_accuracy(self, 
                                       query_text: str,
                                       predicted_result: EnhancedClassificationResult,
                                       actual_category: QueryClassificationCategories,
                                       routing_success: bool) -> Dict[str, Any]:
        """Validate classification accuracy and record feedback."""
        
        if self.engine:
            return self.engine.validate_confidence_accuracy(
                query_text, predicted_result, actual_category, routing_success
            )
        
        # Basic validation without calibration feedback
        return {
            'query': query_text,
            'predicted_category': predicted_result.category.value,
            'actual_category': actual_category.value,
            'category_correct': predicted_result.category == actual_category,
            'confidence_error': abs(predicted_result.confidence - (1.0 if routing_success else 0.0)),
            'calibration_feedback_recorded': False
        }


class RoutingIntegrationManager:
    """
    Manager for integrating enhanced confidence scoring with routing decisions.
    """
    
    def __init__(self, classification_service: EnhancedClassificationService):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.classification_service = classification_service
        
        # Routing thresholds based on confidence levels
        self.routing_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
    
    async def route_query_with_enhanced_confidence(self, 
                                                 query_text: str,
                                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route query using enhanced confidence scoring to make routing decisions.
        """
        
        try:
            # Classify with enhanced confidence
            classification = await self.classification_service.classify_with_fallback(query_text, context)
            
            # Get routing information
            routing_info = integrate_enhanced_classification_with_routing(classification)
            
            # Make enhanced routing decision based on confidence metrics
            routing_decision = self._make_enhanced_routing_decision(classification, routing_info)
            
            return {
                'query': query_text,
                'classification': classification.to_dict(),
                'routing_info': routing_info,
                'enhanced_routing_decision': routing_decision,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced routing failed for query '{query_text[:50]}...': {e}")
            return self._create_fallback_routing_decision(query_text, str(e))
    
    def _make_enhanced_routing_decision(self, 
                                      classification: EnhancedClassificationResult,
                                      routing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Make sophisticated routing decision based on enhanced confidence metrics."""
        
        primary_route = routing_info['routing_decision']['primary_route']
        confidence = classification.confidence
        uncertainty = classification.total_uncertainty
        reliability = classification.confidence_reliability
        
        # Determine routing strategy based on confidence analysis
        if classification.is_high_confidence():
            strategy = "direct_routing"
            use_primary = True
            require_validation = False
            
        elif confidence >= self.routing_thresholds['medium_confidence'] and reliability >= 0.7:
            strategy = "confident_routing"
            use_primary = True
            require_validation = True
            
        elif uncertainty > 0.6 or classification.conflict_score > 0.4:
            strategy = "hybrid_routing"
            use_primary = False
            require_validation = True
            
        else:
            strategy = "cautious_routing"
            use_primary = True
            require_validation = True
        
        # Build routing decision
        decision = {
            'strategy': strategy,
            'primary_route': primary_route if use_primary else "hybrid",
            'fallback_routes': routing_info.get('fallback_routes', []),
            'require_validation': require_validation,
            'confidence_level': routing_info['routing_decision']['confidence_level'],
            'recommendation': routing_info['recommendation']['recommendation'],
            'metadata': {
                'confidence': confidence,
                'uncertainty': uncertainty,
                'reliability': reliability,
                'evidence_strength': classification.evidence_strength,
                'should_use_hybrid': routing_info['should_use_hybrid'],
                'requires_clarification': routing_info['requires_clarification']
            }
        }
        
        return decision
    
    def _create_fallback_routing_decision(self, query_text: str, error_message: str) -> Dict[str, Any]:
        """Create fallback routing decision when enhanced routing fails."""
        
        return {
            'query': query_text,
            'error': error_message,
            'fallback_routing': {
                'strategy': 'fallback',
                'primary_route': 'either',
                'confidence_level': 'unknown',
                'require_validation': True
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def validate_routing_performance(self, 
                                         routing_results: List[Dict[str, Any]],
                                         actual_outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate routing performance and provide calibration feedback."""
        
        if not routing_results or not actual_outcomes or len(routing_results) != len(actual_outcomes):
            return {'error': 'Invalid validation data'}
        
        validation_results = []
        
        for routing_result, actual_outcome in zip(routing_results, actual_outcomes):
            try:
                # Extract classification from routing result
                classification_dict = routing_result.get('classification', {})
                
                # Reconstruct classification (simplified for demo)
                predicted_category = QueryClassificationCategories(
                    classification_dict.get('category', 'general')
                )
                predicted_confidence = classification_dict.get('confidence', 0.5)
                
                # Create enhanced result for validation
                enhanced_result = EnhancedClassificationResult(
                    category=predicted_category,
                    confidence=predicted_confidence,
                    reasoning=classification_dict.get('reasoning', []),
                    keyword_match_confidence=0.5, pattern_match_confidence=0.5,
                    semantic_confidence=0.5, temporal_confidence=0.0,
                    matched_keywords=[], matched_patterns=[], biomedical_entities=[],
                    temporal_indicators=[], alternative_classifications=[],
                    classification_time_ms=100.0, ambiguity_score=0.5, conflict_score=0.1,
                    confidence_interval=(predicted_confidence - 0.1, predicted_confidence + 0.1)
                )
                
                # Validate against actual outcome
                validation = self.classification_service.validate_classification_accuracy(
                    query_text=routing_result.get('query', ''),
                    predicted_result=enhanced_result,
                    actual_category=QueryClassificationCategories(actual_outcome.get('actual_category', 'general')),
                    routing_success=actual_outcome.get('routing_success', False)
                )
                
                validation_results.append(validation)
                
            except Exception as e:
                self.logger.error(f"Validation failed for routing result: {e}")
                continue
        
        # Aggregate validation results
        if validation_results:
            total_validations = len(validation_results)
            correct_categories = sum(1 for v in validation_results if v.get('category_correct', False))
            avg_confidence_error = sum(v.get('confidence_error', 0) for v in validation_results) / total_validations
            
            return {
                'total_validations': total_validations,
                'category_accuracy': correct_categories / total_validations,
                'average_confidence_error': avg_confidence_error,
                'validation_details': validation_results
            }
        
        return {'error': 'No valid validation results'}


class BackwardCompatibilityManager:
    """
    Manager to ensure backward compatibility with existing systems.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def convert_enhanced_to_legacy_format(self, 
                                        enhanced_result: EnhancedClassificationResult) -> Dict[str, Any]:
        """Convert EnhancedClassificationResult to legacy system format."""
        
        # Convert back to basic ClassificationResult
        basic_result = enhanced_result.to_basic_classification()
        
        # Legacy format expected by older systems
        legacy_format = {
            'category': basic_result.category.value,
            'confidence': basic_result.confidence,
            'reasoning': ' '.join(basic_result.reasoning),
            'classification_time_ms': basic_result.classification_time_ms,
            'evidence': {
                'keywords': basic_result.matched_keywords,
                'patterns': basic_result.matched_patterns,
                'entities': basic_result.biomedical_entities
            },
            'metadata': {
                'ambiguity_score': basic_result.ambiguity_score,
                'conflict_score': basic_result.conflict_score
            }
        }
        
        return legacy_format
    
    def enhance_legacy_result(self, legacy_result: Dict[str, Any]) -> EnhancedClassificationResult:
        """Convert legacy classification result to enhanced format."""
        
        # Convert legacy dict to ClassificationResult
        basic_result = ClassificationResult(
            category=QueryClassificationCategories(legacy_result.get('category', 'general')),
            confidence=legacy_result.get('confidence', 0.5),
            reasoning=[legacy_result.get('reasoning', 'Legacy classification')],
            keyword_match_confidence=legacy_result.get('confidence', 0.5) * 0.9,
            pattern_match_confidence=legacy_result.get('confidence', 0.5) * 0.8,
            semantic_confidence=legacy_result.get('confidence', 0.5),
            temporal_confidence=0.1,
            matched_keywords=legacy_result.get('evidence', {}).get('keywords', []),
            matched_patterns=legacy_result.get('evidence', {}).get('patterns', []),
            biomedical_entities=legacy_result.get('evidence', {}).get('entities', []),
            temporal_indicators=[],
            alternative_classifications=[],
            classification_time_ms=legacy_result.get('classification_time_ms', 100.0),
            ambiguity_score=legacy_result.get('metadata', {}).get('ambiguity_score', 0.5),
            conflict_score=legacy_result.get('metadata', {}).get('conflict_score', 0.1)
        )
        
        # Convert to enhanced format
        return EnhancedClassificationResult.from_basic_classification(basic_result)


class IntegrationDemo:
    """
    Comprehensive demonstration of enhanced confidence scoring integration.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.classification_service = EnhancedClassificationService()
        self.routing_manager = RoutingIntegrationManager(self.classification_service)
        self.compatibility_manager = BackwardCompatibilityManager()
    
    async def run_integration_demo(self):
        """Run comprehensive integration demonstration."""
        
        print("=" * 80)
        print("ENHANCED CONFIDENCE SCORING INTEGRATION DEMO")
        print("=" * 80)
        
        # Initialize services
        print("\n1. Initializing Enhanced Classification Service...")
        success = await self.classification_service.initialize()
        print(f"   Enhanced service initialized: {success}")
        
        # Test queries for different scenarios
        test_scenarios = [
            {
                'query': "What are the metabolic pathways involved in glucose metabolism?",
                'expected_category': QueryClassificationCategories.KNOWLEDGE_GRAPH,
                'description': "High-confidence knowledge graph query"
            },
            {
                'query': "Latest metabolomics research developments in 2025",
                'expected_category': QueryClassificationCategories.REAL_TIME,
                'description': "High-confidence real-time query"
            },
            {
                'query': "metabolomics analysis",
                'expected_category': QueryClassificationCategories.GENERAL,
                'description': "Ambiguous general query"
            }
        ]
        
        # Demonstrate enhanced classification
        print("\n2. Enhanced Classification with Comprehensive Confidence...")
        classification_results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n   Scenario {i}: {scenario['description']}")
            print(f"   Query: {scenario['query']}")
            
            try:
                enhanced_result = await self.classification_service.classify_with_fallback(
                    scenario['query']
                )
                
                classification_results.append({
                    'scenario': scenario,
                    'result': enhanced_result
                })
                
                print(f"   Category: {enhanced_result.category.value}")
                print(f"   Confidence: {enhanced_result.confidence:.3f}")
                print(f"   Interval: [{enhanced_result.confidence_interval[0]:.3f}, {enhanced_result.confidence_interval[1]:.3f}]")
                print(f"   Reliability: {enhanced_result.confidence_reliability:.3f}")
                print(f"   Uncertainty: {enhanced_result.total_uncertainty:.3f}")
                
                recommendation = enhanced_result.get_recommendation()
                print(f"   Recommendation: {recommendation['confidence_level']} - {recommendation['recommendation']}")
                
            except Exception as e:
                print(f"   Error: {e}")
        
        # Demonstrate routing integration
        print("\n3. Routing Integration with Enhanced Confidence...")
        routing_results = []
        
        for result_data in classification_results:
            scenario = result_data['scenario']
            print(f"\n   Routing for: {scenario['description']}")
            
            try:
                routing_result = await self.routing_manager.route_query_with_enhanced_confidence(
                    scenario['query']
                )
                
                routing_results.append(routing_result)
                
                decision = routing_result['enhanced_routing_decision']
                print(f"   Strategy: {decision['strategy']}")
                print(f"   Primary Route: {decision['primary_route']}")
                print(f"   Validation Required: {decision['require_validation']}")
                
            except Exception as e:
                print(f"   Routing Error: {e}")
        
        # Demonstrate backward compatibility
        print("\n4. Backward Compatibility Demonstration...")
        
        for result_data in classification_results:
            scenario = result_data['scenario']
            enhanced_result = result_data['result']
            
            print(f"\n   Converting: {scenario['description']}")
            
            # Convert to legacy format
            legacy_format = self.compatibility_manager.convert_enhanced_to_legacy_format(
                enhanced_result
            )
            print(f"   Legacy Format: category={legacy_format['category']}, confidence={legacy_format['confidence']:.3f}")
            
            # Convert back to enhanced
            restored_enhanced = self.compatibility_manager.enhance_legacy_result(legacy_format)
            print(f"   Restored Enhanced: category={restored_enhanced.category.value}, confidence={restored_enhanced.confidence:.3f}")
        
        # Demonstrate validation
        print("\n5. Classification Accuracy Validation...")
        
        # Simulate actual outcomes
        simulated_outcomes = [
            {'actual_category': 'knowledge_graph', 'routing_success': True},
            {'actual_category': 'real_time', 'routing_success': True},
            {'actual_category': 'general', 'routing_success': False}
        ]
        
        validation_result = await self.routing_manager.validate_routing_performance(
            routing_results[:len(simulated_outcomes)], simulated_outcomes
        )
        
        if 'error' not in validation_result:
            print(f"   Total Validations: {validation_result['total_validations']}")
            print(f"   Category Accuracy: {validation_result['category_accuracy']:.1%}")
            print(f"   Avg Confidence Error: {validation_result['average_confidence_error']:.3f}")
        else:
            print(f"   Validation Error: {validation_result['error']}")
        
        # Summary
        print("\n" + "=" * 80)
        print("INTEGRATION DEMO SUMMARY")
        print("=" * 80)
        print("✓ Enhanced classification with comprehensive confidence scoring")
        print("✓ Seamless integration with existing routing infrastructure")
        print("✓ Sophisticated routing decisions based on confidence analysis")
        print("✓ Full backward compatibility with legacy systems")
        print("✓ Real-time confidence validation and calibration feedback")
        print("✓ Graceful fallback mechanisms for system resilience")
        
        print("\nIntegration demo completed successfully!")
        
        return {
            'classification_results': [r['result'].to_dict() for r in classification_results],
            'routing_results': routing_results,
            'validation_result': validation_result
        }


async def main():
    """Main integration demo execution."""
    
    try:
        demo = IntegrationDemo()
        results = await demo.run_integration_demo()
        
        # Optionally save results
        results_file = "enhanced_confidence_integration_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nIntegration results saved to: {results_file}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
        
    except KeyboardInterrupt:
        print("\nIntegration demo interrupted by user")
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        print(f"Demo error: {e}")


if __name__ == "__main__":
    print("Starting Enhanced Confidence Scoring Integration Demo...")
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Failed to run integration demo: {e}")
        exit(1)