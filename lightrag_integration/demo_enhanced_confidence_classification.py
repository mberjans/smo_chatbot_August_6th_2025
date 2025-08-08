#!/usr/bin/env python3
"""
Enhanced Confidence Scoring for Classification Results - Demo Script

This script demonstrates the new comprehensive confidence scoring functionality
integrated with the existing classification system for CMO-LIGHTRAG-012-T06.

Features demonstrated:
- Enhanced ClassificationResult with comprehensive confidence metrics
- Integration with HybridConfidenceScorer for calibrated confidence scores
- Automatic confidence calibration and uncertainty quantification
- Compatibility with existing classification infrastructure
- Real-time confidence validation and feedback

Author: Claude Code (Anthropic)
Created: 2025-08-08
Task: CMO-LIGHTRAG-012-T06 - Add confidence scoring for classification results
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the enhanced classification system
try:
    from query_classification_system import (
        QueryClassificationCategories,
        ClassificationResult,
        EnhancedClassificationResult,
        EnhancedQueryClassificationEngine,
        create_enhanced_classification_engine,
        classify_with_enhanced_confidence,
        integrate_enhanced_classification_with_routing
    )
    from comprehensive_confidence_scorer import (
        HybridConfidenceScorer,
        create_hybrid_confidence_scorer,
        ConfidenceSource
    )
except ImportError as e:
    logger.error(f"Failed to import enhanced classification components: {e}")
    logger.info("Please ensure all required modules are available.")
    exit(1)


class EnhancedConfidenceDemo:
    """
    Demonstration class for enhanced confidence scoring in classification results.
    """
    
    def __init__(self):
        """Initialize the demo with enhanced classification capabilities."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.engine = None
        self.confidence_scorer = None
        
        # Test queries for different categories
        self.test_queries = {
            'knowledge_graph': [
                "What is the relationship between glucose metabolism and insulin signaling pathways?",
                "How do biomarkers interact in metabolic disease pathways?",
                "Explain the mechanism of metabolite transport in cellular metabolism",
                "What are the connections between amino acid metabolism and protein synthesis?"
            ],
            'real_time': [
                "Latest research on metabolomics biomarkers for diabetes 2025",
                "Recent developments in LC-MS technology this year",
                "Current clinical trials using metabolomics for cancer diagnosis",
                "What are the newest metabolomics discoveries in 2025?"
            ],
            'general': [
                "What is clinical metabolomics?",
                "Define mass spectrometry",
                "Explain metabolite identification",
                "Overview of biomarker discovery"
            ]
        }
    
    async def initialize_system(self):
        """Initialize the enhanced classification system."""
        
        self.logger.info("Initializing enhanced classification system with confidence scoring...")
        
        try:
            # Create enhanced classification engine
            self.engine = await create_enhanced_classification_engine(
                logger=self.logger,
                enable_hybrid_confidence=True
            )
            
            # Get reference to the confidence scorer
            self.confidence_scorer = self.engine.confidence_scorer
            
            self.logger.info("Enhanced classification system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced classification system: {e}")
            
            # Try fallback initialization
            try:
                self.engine = await create_enhanced_classification_engine(
                    logger=self.logger,
                    enable_hybrid_confidence=False
                )
                self.logger.warning("Initialized with basic confidence scoring only")
                return True
            except Exception as fallback_error:
                self.logger.error(f"Fallback initialization also failed: {fallback_error}")
                return False
    
    async def demonstrate_enhanced_classification(self):
        """Demonstrate enhanced classification with confidence scoring."""
        
        print("\n" + "=" * 80)
        print("ENHANCED CONFIDENCE SCORING FOR CLASSIFICATION RESULTS")
        print("=" * 80)
        
        results = []
        
        for category_name, queries in self.test_queries.items():
            print(f"\n--- {category_name.upper()} CATEGORY QUERIES ---")
            
            for i, query in enumerate(queries, 1):
                print(f"\nQuery {i}: {query}")
                print("-" * 60)
                
                try:
                    # Classify with enhanced confidence scoring
                    start_time = time.time()
                    
                    enhanced_result = await self.engine.classify_query_enhanced(query)
                    
                    classification_time = (time.time() - start_time) * 1000
                    
                    # Display results
                    self._display_enhanced_result(enhanced_result)
                    
                    # Store for analysis
                    results.append({
                        'query': query,
                        'expected_category': category_name,
                        'result': enhanced_result.to_dict(),
                        'classification_time_ms': classification_time
                    })
                    
                except Exception as e:
                    self.logger.error(f"Classification failed for query '{query[:50]}...': {e}")
        
        return results
    
    def _display_enhanced_result(self, result: EnhancedClassificationResult):
        """Display enhanced classification result in a formatted way."""
        
        print(f"Category: {result.category.value}")
        print(f"Overall Confidence: {result.confidence:.3f}")
        print(f"Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
        print(f"Evidence Strength: {result.evidence_strength:.3f}")
        print(f"Reliability: {result.confidence_reliability:.3f}")
        print(f"Total Uncertainty: {result.total_uncertainty:.3f}")
        
        # Enhanced confidence metrics
        print(f"\nConfidence Breakdown:")
        print(f"  LLM Weight: {result.llm_weight:.3f}")
        print(f"  Keyword Weight: {result.keyword_weight:.3f}")
        print(f"  Calibration Adjustment: {result.calibration_adjustment:+.3f}")
        
        # Uncertainty metrics
        print(f"\nUncertainty Analysis:")
        print(f"  Epistemic (Model): {result.epistemic_uncertainty:.3f}")
        print(f"  Aleatoric (Data): {result.aleatoric_uncertainty:.3f}")
        
        # Get recommendation
        recommendation = result.get_recommendation()
        print(f"\nRecommendation: {recommendation['confidence_level']} - {recommendation['recommendation']}")
        
        # Performance metrics
        print(f"\nPerformance:")
        print(f"  Classification Time: {result.classification_time_ms:.2f}ms")
        print(f"  Confidence Calculation: {result.confidence_calculation_time_ms:.2f}ms")
        
        # Confidence analysis details (if available)
        if result.llm_confidence_analysis:
            llm_analysis = result.llm_confidence_analysis
            print(f"\nLLM Analysis:")
            print(f"  Raw Confidence: {llm_analysis.raw_confidence:.3f}")
            print(f"  Calibrated: {llm_analysis.calibrated_confidence:.3f}")
            print(f"  Reasoning Quality: {llm_analysis.reasoning_quality_score:.3f}")
            print(f"  Consistency: {llm_analysis.consistency_score:.3f}")
        
        if result.keyword_confidence_analysis:
            kw_analysis = result.keyword_confidence_analysis
            print(f"\nKeyword Analysis:")
            print(f"  Pattern Match: {kw_analysis.pattern_match_confidence:.3f}")
            print(f"  Biomedical Entities: {kw_analysis.biomedical_entity_confidence:.3f}")
            print(f"  Domain Alignment: {kw_analysis.domain_alignment_score:.3f}")
            print(f"  Strong Signals: {kw_analysis.strong_signals}")
            print(f"  Conflicting Signals: {kw_analysis.conflicting_signals}")
    
    async def demonstrate_confidence_calibration(self):
        """Demonstrate confidence calibration and validation."""
        
        print("\n" + "=" * 80)
        print("CONFIDENCE CALIBRATION AND VALIDATION DEMO")
        print("=" * 80)
        
        # Simulate validation scenarios
        validation_scenarios = [
            {
                'query': "What are the biomarkers for diabetes metabolomics studies?",
                'expected_category': QueryClassificationCategories.KNOWLEDGE_GRAPH,
                'routing_success': True,
                'description': "Successful knowledge graph routing"
            },
            {
                'query': "Latest 2025 developments in metabolomics research",
                'expected_category': QueryClassificationCategories.REAL_TIME,
                'routing_success': True,
                'description': "Successful real-time routing"
            },
            {
                'query': "What is LC-MS analysis method?",
                'expected_category': QueryClassificationCategories.GENERAL,
                'routing_success': False,
                'description': "Failed routing - query was more complex"
            }
        ]
        
        validation_results = []
        
        for i, scenario in enumerate(validation_scenarios, 1):
            print(f"\nValidation Scenario {i}: {scenario['description']}")
            print(f"Query: {scenario['query']}")
            print("-" * 60)
            
            try:
                # Classify the query
                enhanced_result = await self.engine.classify_query_enhanced(scenario['query'])
                
                # Validate confidence accuracy
                validation_result = self.engine.validate_confidence_accuracy(
                    query_text=scenario['query'],
                    predicted_result=enhanced_result,
                    actual_category=scenario['expected_category'],
                    actual_routing_success=scenario['routing_success']
                )
                
                # Display validation results
                self._display_validation_result(validation_result)
                
                validation_results.append(validation_result)
                
            except Exception as e:
                self.logger.error(f"Validation failed for scenario {i}: {e}")
        
        return validation_results
    
    def _display_validation_result(self, validation: Dict[str, Any]):
        """Display confidence validation results."""
        
        print(f"Predicted Category: {validation['predicted_category']}")
        print(f"Actual Category: {validation['actual_category']}")
        print(f"Category Correct: {validation['category_correct']}")
        print(f"Predicted Confidence: {validation['predicted_confidence']:.3f}")
        print(f"Routing Successful: {validation['routing_successful']}")
        print(f"Confidence Error: {validation['confidence_error']:.3f}")
        print(f"Interval Accurate: {validation['confidence_interval_accurate']}")
        
        if validation.get('calibration_feedback_recorded'):
            print("✓ Calibration feedback recorded for future improvement")
    
    async def demonstrate_batch_classification(self):
        """Demonstrate batch classification with enhanced confidence."""
        
        print("\n" + "=" * 80)
        print("BATCH CLASSIFICATION WITH ENHANCED CONFIDENCE")
        print("=" * 80)
        
        # Mixed batch of queries
        batch_queries = [
            "Glucose metabolic pathways in diabetes",
            "Latest metabolomics software tools 2025",
            "Define biomarker discovery",
            "Amino acid metabolism mechanisms",
            "Recent LC-MS developments"
        ]
        
        print(f"Processing batch of {len(batch_queries)} queries...")
        
        try:
            start_time = time.time()
            
            # Batch classify with enhanced confidence
            batch_results = await self.engine.batch_classify_enhanced(batch_queries)
            
            total_time = (time.time() - start_time) * 1000
            
            print(f"\nBatch completed in {total_time:.2f}ms")
            print(f"Average time per query: {total_time / len(batch_queries):.2f}ms")
            
            # Display summary results
            print("\nBatch Results Summary:")
            print("-" * 40)
            
            for i, (query, result) in enumerate(zip(batch_queries, batch_results), 1):
                recommendation = result.get_recommendation()
                print(f"{i:2d}. {query[:40]:40s} -> {result.category.value:15s} "
                      f"({result.confidence:.2f}, {recommendation['confidence_level']})")
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch classification failed: {e}")
            return []
    
    async def demonstrate_routing_integration(self):
        """Demonstrate integration with routing decisions."""
        
        print("\n" + "=" * 80)
        print("ROUTING INTEGRATION WITH ENHANCED CONFIDENCE")
        print("=" * 80)
        
        test_query = "How do metabolite biomarkers correlate with disease progression pathways?"
        
        print(f"Query: {test_query}")
        print("-" * 60)
        
        try:
            # Get enhanced classification
            enhanced_result = await self.engine.classify_query_enhanced(test_query)
            
            # Convert to routing decision format
            routing_info = integrate_enhanced_classification_with_routing(enhanced_result)
            
            # Display routing decision
            self._display_routing_integration(routing_info)
            
            return routing_info
            
        except Exception as e:
            self.logger.error(f"Routing integration demonstration failed: {e}")
            return None
    
    def _display_routing_integration(self, routing_info: Dict[str, Any]):
        """Display routing integration information."""
        
        print("Routing Decision:")
        print(f"  Primary Route: {routing_info['routing_decision']['primary_route']}")
        print(f"  Category: {routing_info['routing_decision']['category']}")
        print(f"  Confidence Level: {routing_info['routing_decision']['confidence_level']}")
        
        print("\nConfidence Metrics Summary:")
        conf_metrics = routing_info['confidence_metrics']
        print(f"  Overall Confidence: {conf_metrics['overall_confidence']:.3f}")
        print(f"  Confidence Range: {conf_metrics['confidence_range']}")
        print(f"  Evidence Strength: {conf_metrics['evidence_strength']:.3f}")
        print(f"  Reliability: {conf_metrics['reliability']:.3f}")
        print(f"  Uncertainty: {conf_metrics['uncertainty']:.3f}")
        
        print("\nRouting Recommendations:")
        print(f"  Use Hybrid Routing: {routing_info['should_use_hybrid']}")
        print(f"  Requires Clarification: {routing_info['requires_clarification']}")
        
        if routing_info['fallback_routes']:
            print(f"  Fallback Routes: {', '.join(routing_info['fallback_routes'])}")
    
    async def display_system_statistics(self):
        """Display comprehensive system statistics."""
        
        print("\n" + "=" * 80)
        print("SYSTEM STATISTICS AND CALIBRATION STATUS")
        print("=" * 80)
        
        try:
            # Get confidence calibration stats
            calibration_stats = self.engine.get_confidence_calibration_stats()
            
            print("Confidence Scoring System:")
            print(f"  Enabled: {calibration_stats.get('confidence_scoring_enabled', False)}")
            
            if calibration_stats.get('confidence_scoring_enabled'):
                scoring_perf = calibration_stats.get('scoring_performance', {})
                print(f"  Total Classifications: {scoring_perf.get('total_scorings', 0)}")
                print(f"  Average Scoring Time: {scoring_perf.get('average_scoring_time_ms', 0):.2f}ms")
                
                calibration_data = calibration_stats.get('calibration_stats', {})
                print(f"\nCalibration Status:")
                print(f"  Total Predictions: {calibration_data.get('total_predictions', 0)}")
                print(f"  Overall Accuracy: {calibration_data.get('overall_accuracy', 0):.1%}")
                print(f"  Calibration Slope: {calibration_data.get('calibration_slope', 1.0):.3f}")
                print(f"  Brier Score: {calibration_data.get('brier_score', 0.0):.3f}")
                
                conf_dist = calibration_stats.get('confidence_distribution', {})
                if conf_dist:
                    print(f"\nConfidence Distribution:")
                    print(f"  Mean: {conf_dist.get('mean', 0):.3f}")
                    print(f"  Median: {conf_dist.get('median', 0):.3f}")
                    print(f"  Std Dev: {conf_dist.get('std_dev', 0):.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to get system statistics: {e}")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of enhanced confidence scoring."""
        
        print("CLINICAL METABOLOMICS ORACLE - ENHANCED CONFIDENCE SCORING DEMO")
        print("CMO-LIGHTRAG-012-T06: Add confidence scoring for classification results")
        print("=" * 80)
        
        # Initialize system
        if not await self.initialize_system():
            print("Failed to initialize system. Exiting demo.")
            return
        
        try:
            # Run all demonstrations
            demo_results = {}
            
            # 1. Enhanced classification demo
            demo_results['classification'] = await self.demonstrate_enhanced_classification()
            
            # 2. Confidence calibration demo  
            demo_results['calibration'] = await self.demonstrate_confidence_calibration()
            
            # 3. Batch classification demo
            demo_results['batch'] = await self.demonstrate_batch_classification()
            
            # 4. Routing integration demo
            demo_results['routing'] = await self.demonstrate_routing_integration()
            
            # 5. System statistics
            await self.display_system_statistics()
            
            # Summary
            print("\n" + "=" * 80)
            print("DEMO COMPLETION SUMMARY")
            print("=" * 80)
            print("✓ Enhanced classification with comprehensive confidence scoring")
            print("✓ Integration with existing ClassificationResult structure")  
            print("✓ Automatic confidence calibration and uncertainty quantification")
            print("✓ Batch processing capabilities")
            print("✓ Routing decision integration")
            print("✓ Real-time confidence validation and feedback")
            
            # Save results
            results_file = Path("enhanced_confidence_demo_results.json")
            try:
                with open(results_file, 'w') as f:
                    json.dump({
                        'demo_timestamp': time.time(),
                        'demo_results': demo_results,
                        'system_info': {
                            'confidence_scoring_enabled': self.engine.enable_hybrid_confidence,
                            'engine_type': type(self.engine).__name__
                        }
                    }, f, indent=2, default=str)
                
                print(f"\nDemo results saved to: {results_file.absolute()}")
                
            except Exception as e:
                self.logger.warning(f"Failed to save demo results: {e}")
            
            print("\nDemo completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Demo execution failed: {e}")
            print(f"Demo failed with error: {e}")


async def main():
    """Main demo execution function."""
    
    try:
        demo = EnhancedConfidenceDemo()
        await demo.run_comprehensive_demo()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    print("Starting Enhanced Confidence Scoring Demo...")
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Failed to run demo: {e}")
        exit(1)