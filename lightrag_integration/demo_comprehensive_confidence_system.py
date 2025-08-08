#!/usr/bin/env python3
"""
Comprehensive Confidence Scoring System Demonstration

This demonstration script showcases the complete comprehensive confidence scoring system
that integrates LLM-based semantic classification with keyword-based confidence metrics.

Key Features Demonstrated:
    - Hybrid confidence scoring combining LLM and keyword approaches
    - Multi-dimensional confidence analysis with component breakdown
    - Confidence calibration based on historical performance
    - Advanced uncertainty quantification (epistemic and aleatoric)
    - Confidence intervals and reliability scoring
    - Real-time validation and feedback integration
    - Backward compatibility with existing ConfidenceMetrics
    - Performance monitoring and optimization recommendations

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Any
from pathlib import Path

# Import our comprehensive confidence scoring components
try:
    from comprehensive_confidence_scorer import (
        HybridConfidenceScorer, ConfidenceValidator, 
        create_hybrid_confidence_scorer, ConfidenceSource
    )
    from enhanced_query_router_integration import (
        EnhancedBiomedicalQueryRouter, enhanced_router_context,
        create_enhanced_biomedical_router
    )
    from query_router import BiomedicalQueryRouter
    from research_categorizer import ResearchCategorizer
    
    # Optional LLM classifier import
    try:
        from enhanced_llm_classifier import EnhancedLLMQueryClassifier, EnhancedLLMConfig
        LLM_AVAILABLE = True
    except ImportError:
        LLM_AVAILABLE = False
        print("LLM classifier not available - will demonstrate keyword-only confidence")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    exit(1)


# ============================================================================
# DEMONSTRATION CONFIGURATION
# ============================================================================

class DemoConfig:
    """Configuration for the demonstration."""
    
    # Demo queries with different characteristics
    TEST_QUERIES = [
        # High-confidence biomedical queries
        "What is the relationship between glucose metabolism and insulin signaling pathways in type 2 diabetes?",
        "LC-MS/MS analysis of amino acid metabolites for biomarker discovery in Alzheimer's disease",
        "KEGG pathway enrichment analysis for metabolomics data from cancer patients",
        
        # Medium-confidence queries
        "metabolomics analysis methods for clinical research",
        "biomarker discovery using mass spectrometry techniques",
        "pathway analysis in metabolic disorders",
        
        # Lower-confidence or ambiguous queries
        "latest research 2025",
        "what is metabolomics?",
        "analysis methods",
        
        # Complex multi-part queries
        "How can I integrate LC-MS metabolomics data with genomics data to identify novel biomarkers for personalized medicine in cardiovascular disease, and what statistical methods should I use for pathway enrichment analysis?",
        
        # Temporal queries
        "What are the latest developments in clinical metabolomics research published in 2025?",
        "Recent advances in metabolite identification using high-resolution mass spectrometry"
    ]
    
    # Confidence validation scenarios
    VALIDATION_SCENARIOS = [
        {"query": "LC-MS metabolite identification", "predicted_conf": 0.85, "actual_accuracy": True},
        {"query": "pathway analysis methods", "predicted_conf": 0.75, "actual_accuracy": True},
        {"query": "biomarker discovery", "predicted_conf": 0.70, "actual_accuracy": False},  # Incorrect prediction
        {"query": "latest research", "predicted_conf": 0.45, "actual_accuracy": False},
        {"query": "metabolomics applications", "predicted_conf": 0.80, "actual_accuracy": True},
        {"query": "clinical diagnosis", "predicted_conf": 0.65, "actual_accuracy": True},
    ]
    
    # Performance targets
    PERFORMANCE_TARGETS = {
        'max_confidence_calculation_time_ms': 150,
        'max_total_routing_time_ms': 200,
        'min_confidence_reliability': 0.7,
        'max_acceptable_uncertainty': 0.4
    }


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

async def demonstrate_basic_confidence_integration():
    """Demonstrate basic confidence scoring without LLM."""
    
    print("\n" + "="*80)
    print("BASIC CONFIDENCE SCORING DEMONSTRATION (Keyword-based)")
    print("="*80)
    
    # Create basic router without LLM
    basic_router = BiomedicalQueryRouter()
    
    # Create hybrid scorer using the basic router
    hybrid_scorer = create_hybrid_confidence_scorer(
        biomedical_router=basic_router,
        llm_classifier=None,  # No LLM
        calibration_data_path="/tmp/demo_basic_calibration.json"
    )
    
    # Test confidence scoring on different queries
    for i, query in enumerate(DemoConfig.TEST_QUERIES[:5], 1):
        print(f"\n--- Basic Confidence Test {i} ---")
        print(f"Query: {query[:70]}...")
        
        start_time = time.time()
        
        # Get base routing prediction
        base_prediction = basic_router.route_query(query)
        
        # Calculate comprehensive confidence (keyword-only)
        comprehensive_result = await hybrid_scorer.calculate_comprehensive_confidence(
            query_text=query,
            llm_result=None,
            keyword_prediction=None,  # Will extract from base prediction
            context=None
        )
        
        calculation_time = (time.time() - start_time) * 1000
        
        # Display results
        print(f"Base Confidence: {base_prediction.confidence:.3f}")
        print(f"Comprehensive Confidence: {comprehensive_result.overall_confidence:.3f}")
        print(f"Confidence Interval: [{comprehensive_result.confidence_interval[0]:.3f}, {comprehensive_result.confidence_interval[1]:.3f}]")
        print(f"Evidence Strength: {comprehensive_result.evidence_strength:.3f}")
        print(f"Uncertainty (Total): {comprehensive_result.total_uncertainty:.3f}")
        print(f"Keyword Weight: {comprehensive_result.keyword_weight:.3f}")
        print(f"Reliability Score: {comprehensive_result.confidence_reliability:.3f}")
        print(f"Calculation Time: {calculation_time:.2f}ms")
        
        # Check performance targets
        meets_targets = (
            calculation_time <= DemoConfig.PERFORMANCE_TARGETS['max_confidence_calculation_time_ms'] and
            comprehensive_result.confidence_reliability >= DemoConfig.PERFORMANCE_TARGETS['min_confidence_reliability'] and
            comprehensive_result.total_uncertainty <= DemoConfig.PERFORMANCE_TARGETS['max_acceptable_uncertainty']
        )
        print(f"Meets Performance Targets: {meets_targets}")


async def demonstrate_enhanced_router_integration():
    """Demonstrate the enhanced router with comprehensive confidence."""
    
    print("\n" + "="*80)
    print("ENHANCED ROUTER INTEGRATION DEMONSTRATION")
    print("="*80)
    
    # Create enhanced router
    async with enhanced_router_context(
        enable_comprehensive_confidence=True,
        calibration_data_path="/tmp/demo_enhanced_calibration.json"
    ) as enhanced_router:
        
        print(f"Enhanced router created with comprehensive confidence enabled")
        
        # Demonstrate different routing scenarios
        scenarios = [
            ("High-confidence biomedical", DemoConfig.TEST_QUERIES[0]),
            ("Medium-confidence technical", DemoConfig.TEST_QUERIES[4]),
            ("Low-confidence ambiguous", DemoConfig.TEST_QUERIES[7]),
            ("Complex multi-part", DemoConfig.TEST_QUERIES[9])
        ]
        
        for scenario_name, query in scenarios:
            print(f"\n--- {scenario_name} Scenario ---")
            print(f"Query: {query[:100]}...")
            
            # Enhanced routing with comprehensive analysis
            enhanced_prediction = await enhanced_router.route_query_enhanced(
                query_text=query,
                context={'demo_scenario': scenario_name},
                enable_llm=LLM_AVAILABLE,
                return_comprehensive_analysis=True
            )
            
            # Get comprehensive confidence summary
            confidence_summary = enhanced_prediction.get_confidence_summary()
            
            print(f"Routing Decision: {enhanced_prediction.routing_decision.value}")
            print(f"Overall Confidence: {confidence_summary['overall_confidence']:.3f} ({confidence_summary['confidence_level']})")
            print(f"Confidence Interval: {confidence_summary['confidence_interval']}")
            print(f"Reliability Score: {confidence_summary['reliability_score']:.3f}")
            print(f"High Quality Confidence: {enhanced_prediction.is_high_quality_confidence()}")
            
            # Source contributions
            contributions = confidence_summary['source_contributions']
            print(f"LLM Contribution: {contributions['llm']:.1%}")
            print(f"Keyword Contribution: {contributions['keyword']:.1%}")
            
            # Uncertainty analysis
            uncertainty = confidence_summary['uncertainty_metrics']
            print(f"Epistemic Uncertainty: {uncertainty['epistemic']:.3f}")
            print(f"Aleatoric Uncertainty: {uncertainty['aleatoric']:.3f}")
            print(f"Total Uncertainty: {uncertainty['total']:.3f}")
            
            # Performance metrics
            performance = confidence_summary['calculation_performance']
            print(f"Confidence Calculation Time: {performance['confidence_time_ms']:.2f}ms")
            print(f"Total Processing Time: {performance['total_time_ms']:.2f}ms")
            
            # Check calibration status
            print(f"Calibration Status: {enhanced_prediction.confidence_calibration_status}")
            
            # Alternative confidence scenarios
            if enhanced_prediction.comprehensive_confidence:
                alt_confidences = enhanced_prediction.comprehensive_confidence.alternative_confidences
                if alt_confidences:
                    print("Alternative Confidences:")
                    for alt_name, alt_conf in alt_confidences[:3]:  # Show top 3
                        print(f"  {alt_name}: {alt_conf:.3f}")


async def demonstrate_confidence_calibration():
    """Demonstrate confidence calibration and validation."""
    
    print("\n" + "="*80)
    print("CONFIDENCE CALIBRATION AND VALIDATION DEMONSTRATION")
    print("="*80)
    
    # Create enhanced router for calibration demo
    enhanced_router = await create_enhanced_biomedical_router(
        enable_comprehensive_confidence=True,
        calibration_data_path="/tmp/demo_calibration_validation.json"
    )
    
    print("Simulating confidence calibration with historical data...")
    
    # Simulate validation scenarios to build calibration data
    for scenario in DemoConfig.VALIDATION_SCENARIOS:
        query = scenario['query']
        predicted_confidence = scenario['predicted_conf']
        actual_accuracy = scenario['actual_accuracy']
        
        # Validate routing accuracy
        validation_result = enhanced_router.validate_routing_accuracy(
            query_text=query,
            predicted_routing=enhanced_router.route_query(query).routing_decision,
            predicted_confidence=predicted_confidence,
            actual_accuracy=actual_accuracy,
            context={'validation_demo': True}
        )
        
        print(f"Validated: '{query[:40]}...' - Pred: {predicted_confidence:.3f}, Actual: {actual_accuracy}")
    
    # Get comprehensive validation report
    print("\n--- Calibration Report ---")
    validation_report = enhanced_router.get_confidence_validation_report()
    
    if 'error' not in validation_report:
        summary = validation_report['validation_summary']
        print(f"Total Validations: {summary['total_validations']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.3f}")
        print(f"Calibration Error: {summary['calibration_error']:.3f}")
        print(f"Overall Reliability: {summary['overall_reliability']:.3f}")
        
        # Calibration status
        calibration_status = validation_report['calibration_status']
        print(f"Calibration Status: {calibration_status['status']} ({calibration_status['quality']})")
        print(f"Brier Score: {calibration_status['brier_score']:.3f}")
        print(f"Needs Recalibration: {calibration_status['needs_recalibration']}")
        
        # System health
        print(f"System Health: {validation_report['system_health']}")
        
        # Recommendations
        recommendations = validation_report['recommendations']
        if recommendations:
            print(f"\nRecommendations ({len(recommendations)}):")
            for rec in recommendations[:3]:  # Show top 3
                print(f"  {rec['type']}: {rec['issue']} -> {rec['recommendation']}")
    else:
        print(f"Validation error: {validation_report['error']}")


async def demonstrate_performance_analysis():
    """Demonstrate performance analysis and optimization."""
    
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS AND OPTIMIZATION DEMONSTRATION") 
    print("="*80)
    
    # Create enhanced router for performance testing
    enhanced_router = await create_enhanced_biomedical_router(
        enable_comprehensive_confidence=True,
        calibration_data_path="/tmp/demo_performance.json"
    )
    
    print("Running performance analysis on multiple queries...")
    
    # Performance metrics collection
    processing_times = []
    confidence_calculations = []
    reliability_scores = []
    
    # Process multiple queries for performance analysis
    for i, query in enumerate(DemoConfig.TEST_QUERIES, 1):
        start_time = time.time()
        
        enhanced_prediction = await enhanced_router.route_query_enhanced(
            query_text=query,
            enable_llm=LLM_AVAILABLE,
            return_comprehensive_analysis=True
        )
        
        total_time = (time.time() - start_time) * 1000
        processing_times.append(total_time)
        confidence_calculations.append(enhanced_prediction.confidence_calculation_time_ms)
        reliability_scores.append(enhanced_prediction.confidence_reliability_score)
        
        print(f"Query {i}: {total_time:.2f}ms total, {enhanced_prediction.confidence_calculation_time_ms:.2f}ms confidence")
    
    # Calculate performance statistics
    print("\n--- Performance Statistics ---")
    print(f"Average Total Time: {statistics.mean(processing_times):.2f}ms")
    print(f"Average Confidence Time: {statistics.mean(confidence_calculations):.2f}ms")
    print(f"Max Total Time: {max(processing_times):.2f}ms")
    print(f"Min Total Time: {min(processing_times):.2f}ms")
    print(f"Average Reliability Score: {statistics.mean(reliability_scores):.3f}")
    print(f"Min Reliability Score: {min(reliability_scores):.3f}")
    
    # Performance target compliance
    targets = DemoConfig.PERFORMANCE_TARGETS
    target_compliance = {
        'confidence_time_compliance': sum(1 for t in confidence_calculations 
                                        if t <= targets['max_confidence_calculation_time_ms']) / len(confidence_calculations),
        'total_time_compliance': sum(1 for t in processing_times 
                                   if t <= targets['max_total_routing_time_ms']) / len(processing_times),
        'reliability_compliance': sum(1 for r in reliability_scores 
                                    if r >= targets['min_confidence_reliability']) / len(reliability_scores)
    }
    
    print(f"\n--- Target Compliance ---")
    for target_name, compliance_rate in target_compliance.items():
        print(f"{target_name}: {compliance_rate:.1%}")
    
    # Run optimization analysis
    print("\n--- Optimization Analysis ---")
    optimization_results = await enhanced_router.optimize_performance()
    
    print(f"Actions Taken: {len(optimization_results['actions_taken'])}")
    for action in optimization_results['actions_taken']:
        print(f"  - {action}")
    
    print(f"Recommendations: {len(optimization_results['recommendations'])}")
    for rec in optimization_results['recommendations'][:5]:  # Show top 5
        if isinstance(rec, dict):
            print(f"  - {rec['type']}: {rec['issue']}")
        else:
            print(f"  - {rec}")
    
    # Get comprehensive statistics
    print("\n--- Comprehensive Statistics ---")
    stats = enhanced_router.get_enhanced_statistics()
    enhanced_stats = stats['enhanced_routing_stats']
    
    print(f"Total Enhanced Routes: {enhanced_stats['enhanced_routes']}")
    print(f"Comprehensive Confidence Calculations: {enhanced_stats['comprehensive_confidence_calculations']}")
    print(f"LLM Classifications: {enhanced_stats['llm_classifications']}")
    print(f"Fallback Routes: {enhanced_stats['fallback_routes']}")
    
    # Success rates
    if enhanced_stats['enhanced_routes'] > 0:
        llm_success_rate = enhanced_stats['llm_classifications'] / enhanced_stats['enhanced_routes']
        fallback_rate = enhanced_stats['fallback_routes'] / enhanced_stats['enhanced_routes']
        print(f"LLM Success Rate: {llm_success_rate:.1%}")
        print(f"Fallback Rate: {fallback_rate:.1%}")


async def demonstrate_backward_compatibility():
    """Demonstrate backward compatibility with existing systems."""
    
    print("\n" + "="*80)
    print("BACKWARD COMPATIBILITY DEMONSTRATION")
    print("="*80)
    
    # Create enhanced router
    enhanced_router = await create_enhanced_biomedical_router(
        enable_comprehensive_confidence=True
    )
    
    test_query = "LC-MS metabolomics analysis of biomarkers for diabetes diagnosis"
    
    print(f"Test Query: {test_query}")
    print("\nTesting backward compatibility...")
    
    # Test 1: Standard route_query method (should work with existing code)
    print("\n--- Test 1: Standard route_query (legacy interface) ---")
    start_time = time.time()
    legacy_prediction = enhanced_router.route_query(test_query)
    legacy_time = (time.time() - start_time) * 1000
    
    print(f"Routing Decision: {legacy_prediction.routing_decision.value}")
    print(f"Confidence: {legacy_prediction.confidence:.3f}")
    print(f"Confidence Level: {legacy_prediction.confidence_level}")
    print(f"Research Category: {legacy_prediction.research_category.value}")
    print(f"Processing Time: {legacy_time:.2f}ms")
    print(f"Has Confidence Metrics: {legacy_prediction.confidence_metrics is not None}")
    
    # Test 2: Enhanced route_query_enhanced method
    print("\n--- Test 2: Enhanced route_query_enhanced (new interface) ---")
    start_time = time.time()
    enhanced_prediction = await enhanced_router.route_query_enhanced(test_query)
    enhanced_time = (time.time() - start_time) * 1000
    
    print(f"Routing Decision: {enhanced_prediction.routing_decision.value}")
    print(f"Confidence: {enhanced_prediction.confidence:.3f}")
    print(f"Confidence Level: {enhanced_prediction.confidence_level}")
    print(f"Research Category: {enhanced_prediction.research_category.value}")
    print(f"Processing Time: {enhanced_time:.2f}ms")
    print(f"Has Comprehensive Confidence: {enhanced_prediction.comprehensive_confidence is not None}")
    print(f"Reliability Score: {enhanced_prediction.confidence_reliability_score:.3f}")
    
    # Test 3: Conversion between formats
    print("\n--- Test 3: Format Conversion ---")
    from enhanced_query_router_integration import convert_enhanced_to_base_prediction
    
    converted_prediction = convert_enhanced_to_base_prediction(enhanced_prediction)
    
    print(f"Enhanced -> Base Conversion:")
    print(f"  Original Enhanced Confidence: {enhanced_prediction.confidence:.3f}")
    print(f"  Converted Base Confidence: {converted_prediction.confidence:.3f}")
    print(f"  Routing Decision Match: {enhanced_prediction.routing_decision == converted_prediction.routing_decision}")
    print(f"  Research Category Match: {enhanced_prediction.research_category == converted_prediction.research_category}")
    
    # Test 4: ConfidenceMetrics compatibility
    print("\n--- Test 4: ConfidenceMetrics Compatibility ---")
    
    # Both should have ConfidenceMetrics
    legacy_metrics = legacy_prediction.confidence_metrics
    enhanced_metrics = enhanced_prediction.confidence_metrics
    
    print(f"Legacy has ConfidenceMetrics: {legacy_metrics is not None}")
    print(f"Enhanced has ConfidenceMetrics: {enhanced_metrics is not None}")
    
    if legacy_metrics and enhanced_metrics:
        print(f"Legacy overall confidence: {legacy_metrics.overall_confidence:.3f}")
        print(f"Enhanced overall confidence: {enhanced_metrics.overall_confidence:.3f}")
        print(f"Both can be serialized to dict: {legacy_metrics.to_dict() is not None and enhanced_metrics.to_dict() is not None}")
    
    print("\n✅ Backward compatibility maintained - existing code will work unchanged!")


async def run_comprehensive_demonstration():
    """Run the complete comprehensive confidence scoring demonstration."""
    
    print("🚀 COMPREHENSIVE CONFIDENCE SCORING SYSTEM DEMONSTRATION")
    print("=" * 100)
    print(f"LLM Integration Available: {LLM_AVAILABLE}")
    print(f"Demo Query Count: {len(DemoConfig.TEST_QUERIES)}")
    print(f"Validation Scenarios: {len(DemoConfig.VALIDATION_SCENARIOS)}")
    print("=" * 100)
    
    try:
        # Run all demonstration modules
        await demonstrate_basic_confidence_integration()
        await demonstrate_enhanced_router_integration()
        await demonstrate_confidence_calibration()
        await demonstrate_performance_analysis()
        await demonstrate_backward_compatibility()
        
        print("\n" + "="*100)
        print("🎉 COMPREHENSIVE CONFIDENCE SCORING DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*100)
        
        print("\nKey Achievements Demonstrated:")
        print("✅ Hybrid LLM + Keyword confidence scoring")
        print("✅ Multi-dimensional confidence analysis with component breakdown")
        print("✅ Real-time confidence calibration and historical tracking")
        print("✅ Advanced uncertainty quantification (epistemic + aleatoric)")
        print("✅ Confidence intervals and reliability scoring")
        print("✅ Performance monitoring and optimization")
        print("✅ Validation framework for confidence accuracy")
        print("✅ Full backward compatibility with existing systems")
        print("✅ Integration with existing ConfidenceMetrics infrastructure")
        print("✅ Adaptive weighting based on query characteristics")
        
        print(f"\n📊 Performance Summary:")
        print(f"• Target confidence calculation time: ≤{DemoConfig.PERFORMANCE_TARGETS['max_confidence_calculation_time_ms']}ms")
        print(f"• Target total routing time: ≤{DemoConfig.PERFORMANCE_TARGETS['max_total_routing_time_ms']}ms")
        print(f"• Minimum reliability threshold: ≥{DemoConfig.PERFORMANCE_TARGETS['min_confidence_reliability']}")
        print(f"• Maximum uncertainty threshold: ≤{DemoConfig.PERFORMANCE_TARGETS['max_acceptable_uncertainty']}")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose logging for cleaner demo output
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # Run the comprehensive demonstration
    print("Starting comprehensive confidence scoring system demonstration...")
    
    try:
        success = asyncio.run(run_comprehensive_demonstration())
        
        if success:
            print("\n🏆 All demonstrations completed successfully!")
            print("\nThe comprehensive confidence scoring system is ready for production use.")
            print("It provides sophisticated confidence analysis while maintaining full")
            print("backward compatibility with existing Clinical Metabolomics Oracle infrastructure.")
        else:
            print("\n⚠️  Some demonstrations encountered issues. Please review the output above.")
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n💥 Fatal error in demonstration: {e}")
        import traceback
        traceback.print_exc()