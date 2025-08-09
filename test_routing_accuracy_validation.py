#!/usr/bin/env python3
"""
Comprehensive Routing Accuracy Validation Test
CMO-LIGHTRAG-013-T08: Execute routing tests and verify decision accuracy >90%

This script performs comprehensive accuracy testing of the routing decision engine
to validate that it meets the >90% accuracy requirement across all categories.
"""

import sys
import os
import json
import time
import statistics
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict

# Add the parent directory to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision
from lightrag_integration.intelligent_query_router import IntelligentQueryRouter
from lightrag_integration.cost_persistence import ResearchCategory


class RoutingAccuracyValidator:
    """Comprehensive routing accuracy validation system"""
    
    def __init__(self):
        self.router = BiomedicalQueryRouter()
        self.intelligent_router = IntelligentQueryRouter(self.router)
        self.test_results = []
        self.accuracy_metrics = {}
        
        # Comprehensive test dataset with expected routing decisions
        self.test_dataset = {
            RoutingDecision.LIGHTRAG: [
                # Knowledge graph and pathway queries
                "What are the metabolic pathways involved in glucose metabolism?",
                "How do amino acids interact in protein synthesis pathways?",
                "What is the relationship between cholesterol and cardiovascular disease?",
                "Explain the role of mitochondria in cellular respiration pathways",
                "How does insulin signaling affect glucose metabolism?",
                "What are the key enzymes in glycolysis pathway?",
                "Describe the relationship between fatty acid oxidation and ketogenesis",
                "How do neurotransmitters interact in synaptic transmission?",
                "What is the molecular mechanism of DNA replication?",
                "Explain the interaction between AMPK and mTOR signaling pathways",
                "How does the citric acid cycle connect to oxidative phosphorylation?",
                "What are the regulatory mechanisms in steroid hormone synthesis?",
                "Describe the molecular interactions in autophagy pathways",
                "How do growth factors regulate cell cycle progression?",
                "What is the relationship between inflammation and metabolic dysfunction?"
            ],
            RoutingDecision.PERPLEXITY: [
                # Current events, recent research, temporal queries
                "What are the latest clinical trials for COVID-19 treatments published this week?",
                "Recent breakthrough discoveries in metabolomics research from 2024",
                "What are the current FDA approvals for diabetes medications in 2025?",
                "Latest findings on CRISPR gene editing published this month",
                "Recent advances in cancer immunotherapy from top journals this year",
                "What are the newest biomarkers discovered for Alzheimer's disease in 2025?",
                "Current clinical trial results for obesity treatments published recently",
                "Latest research on microbiome and mental health from 2024-2025",
                "Recent developments in personalized medicine published this year",
                "What are the current trends in metabolomics research for 2025?",
                "Latest publications on artificial intelligence in drug discovery",
                "Recent breakthroughs in rare disease treatments from 2024",
                "Current research on longevity and aging published this month",
                "Latest findings on environmental toxins and health from recent studies",
                "What are the newest treatment protocols for autoimmune diseases in 2025?"
            ],
            RoutingDecision.EITHER: [
                # General biomedical queries that could go to either system
                "How does diabetes affect kidney function?",
                "What are the symptoms of metabolic syndrome?",
                "How do statins work to lower cholesterol?",
                "What is the role of inflammation in heart disease?",
                "How does exercise affect metabolism?",
                "What are the causes of insulin resistance?",
                "How do antidepressants affect brain chemistry?",
                "What is the relationship between sleep and metabolism?",
                "How does stress affect the immune system?",
                "What are the health effects of intermittent fasting?",
                "How do probiotics affect gut health?",
                "What is the role of genetics in obesity?",
                "How does alcohol affect liver metabolism?",
                "What are the benefits of omega-3 fatty acids?",
                "How does aging affect metabolic processes?"
            ],
            RoutingDecision.HYBRID: [
                # Complex queries requiring both systems
                "What are the current research trends in metabolomics and recent breakthrough discoveries?",
                "How do metabolic pathways change with age and what are the latest research findings?",
                "What is the molecular basis of diabetes and what are the newest treatment approaches?",
                "Explain cancer metabolism pathways and recent clinical trial results",
                "How do environmental toxins affect cellular metabolism and what are recent regulatory changes?",
                "What are the mechanistic pathways of neurodegeneration and current therapeutic trials?",
                "How does the gut microbiome influence metabolism and what are recent research developments?",
                "Explain the molecular basis of autoimmune diseases and latest treatment protocols",
                "What are the pathways involved in aging and current longevity research trends?",
                "How do genetic variants affect drug metabolism and current personalized medicine approaches?",
                "What are the metabolic mechanisms of obesity and recent FDA-approved treatments?",
                "Explain the molecular pathways of inflammation and current therapeutic targets",
                "How do circadian rhythms affect metabolism and what are recent research findings?",
                "What are the cellular mechanisms of cancer and latest immunotherapy developments?",
                "How do metabolic disorders affect brain function and current research in neurometabolism?"
            ]
        }
    
    def test_routing_accuracy(self) -> Dict[str, Any]:
        """Test routing accuracy across all categories"""
        print("="*80)
        print("COMPREHENSIVE ROUTING ACCURACY VALIDATION")
        print("="*80)
        
        category_results = {}
        all_results = []
        
        for expected_decision, queries in self.test_dataset.items():
            print(f"\nTesting {expected_decision.value.upper()} category ({len(queries)} queries)...")
            category_correct = 0
            category_total = len(queries)
            category_details = []
            
            for i, query in enumerate(queries, 1):
                start_time = time.time()
                
                try:
                    # Route with both routers for comparison
                    base_result = self.router.route_query(query)
                    intelligent_result = self.intelligent_router.route_query(query)
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Determine if routing decision is correct
                    base_correct = self._is_routing_correct(base_result.routing_decision, expected_decision)
                    intelligent_correct = self._is_routing_correct(intelligent_result.routing_decision, expected_decision)
                    
                    if base_correct:
                        category_correct += 1
                    
                    test_result = {
                        'query_id': i,
                        'query': query[:100] + "..." if len(query) > 100 else query,
                        'expected': expected_decision.value,
                        'base_router_decision': base_result.routing_decision.value,
                        'intelligent_router_decision': intelligent_result.routing_decision.value,
                        'base_confidence': base_result.confidence,
                        'intelligent_confidence': intelligent_result.confidence,
                        'base_correct': base_correct,
                        'intelligent_correct': intelligent_correct,
                        'processing_time_ms': processing_time
                    }
                    
                    category_details.append(test_result)
                    all_results.append(test_result)
                    
                    if i % 5 == 0:
                        print(f"  Processed {i}/{category_total} queries...")
                
                except Exception as e:
                    print(f"  Error processing query {i}: {e}")
                    category_details.append({
                        'query_id': i,
                        'query': query,
                        'error': str(e),
                        'base_correct': False,
                        'intelligent_correct': False
                    })
            
            # Calculate category accuracy
            category_accuracy = (category_correct / category_total) * 100
            
            category_results[expected_decision.value] = {
                'total_queries': category_total,
                'correct_predictions': category_correct,
                'accuracy_percentage': category_accuracy,
                'details': category_details
            }
            
            print(f"  {expected_decision.value.upper()} Accuracy: {category_accuracy:.2f}% ({category_correct}/{category_total})")
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(all_results)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': overall_metrics,
            'category_results': category_results,
            'detailed_results': all_results
        }
    
    def _is_routing_correct(self, actual: RoutingDecision, expected: RoutingDecision) -> bool:
        """Determine if a routing decision is correct"""
        if actual == expected:
            return True
        
        # Special handling for EITHER category - any decision is acceptable
        if expected == RoutingDecision.EITHER:
            return actual in [RoutingDecision.LIGHTRAG, RoutingDecision.PERPLEXITY]
        
        # HYBRID queries should route to HYBRID or be handled intelligently
        if expected == RoutingDecision.HYBRID:
            return actual == RoutingDecision.HYBRID
        
        return False
    
    def _calculate_overall_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive accuracy metrics"""
        if not results:
            return {}
        
        total_tests = len(results)
        base_correct = sum(1 for r in results if r.get('base_correct', False))
        intelligent_correct = sum(1 for r in results if r.get('intelligent_correct', False))
        
        # Processing time statistics
        processing_times = [r.get('processing_time_ms', 0) for r in results if 'processing_time_ms' in r]
        
        # Confidence statistics
        base_confidences = [r.get('base_confidence', 0) for r in results if 'base_confidence' in r]
        intelligent_confidences = [r.get('intelligent_confidence', 0) for r in results if 'intelligent_confidence' in r]
        
        metrics = {
            'total_test_queries': total_tests,
            'base_router_accuracy': (base_correct / total_tests) * 100,
            'intelligent_router_accuracy': (intelligent_correct / total_tests) * 100,
            'base_correct_predictions': base_correct,
            'intelligent_correct_predictions': intelligent_correct,
            'target_accuracy_met': (base_correct / total_tests) >= 0.90,
            'intelligent_target_accuracy_met': (intelligent_correct / total_tests) >= 0.90
        }
        
        if processing_times:
            metrics['processing_time_stats'] = {
                'mean_ms': statistics.mean(processing_times),
                'median_ms': statistics.median(processing_times),
                'min_ms': min(processing_times),
                'max_ms': max(processing_times),
                'p95_ms': sorted(processing_times)[int(len(processing_times) * 0.95)] if len(processing_times) > 20 else max(processing_times)
            }
        
        if base_confidences:
            metrics['base_confidence_stats'] = {
                'mean': statistics.mean(base_confidences),
                'median': statistics.median(base_confidences),
                'min': min(base_confidences),
                'max': max(base_confidences)
            }
        
        if intelligent_confidences:
            metrics['intelligent_confidence_stats'] = {
                'mean': statistics.mean(intelligent_confidences),
                'median': statistics.median(intelligent_confidences),
                'min': min(intelligent_confidences),
                'max': max(intelligent_confidences)
            }
        
        return metrics
    
    def test_load_balancing_accuracy(self) -> Dict[str, Any]:
        """Test load balancing distribution accuracy"""
        print("\nTesting Load Balancing Distribution...")
        
        # Test queries that should trigger EITHER routing
        either_queries = self.test_dataset[RoutingDecision.EITHER][:10]
        
        backend_counts = defaultdict(int)
        load_balancing_results = []
        
        for query in either_queries:
            for _ in range(5):  # Test each query 5 times for load balancing
                result = self.intelligent_router.route_query(query)
                selected_backend = result.metadata.get('selected_backend', 'unknown')
                backend_counts[selected_backend] += 1
                
                load_balancing_results.append({
                    'query': query[:50] + "...",
                    'selected_backend': selected_backend,
                    'routing_decision': result.routing_decision.value
                })
        
        total_requests = sum(backend_counts.values())
        distribution = {backend: (count/total_requests)*100 for backend, count in backend_counts.items()}
        
        return {
            'total_load_balanced_requests': total_requests,
            'backend_distribution': dict(backend_counts),
            'distribution_percentages': distribution,
            'balanced_routing': abs(distribution.get('lightrag', 0) - distribution.get('perplexity', 0)) < 30,  # Within 30% is considered balanced
            'details': load_balancing_results
        }
    
    def test_system_health_impact(self) -> Dict[str, Any]:
        """Test system health monitoring impact on routing"""
        print("\nTesting System Health Impact on Routing...")
        
        # Get system health status
        health_status = self.intelligent_router.get_system_health_status()
        
        # Test routing with different health conditions
        test_query = "How does insulin affect glucose metabolism?"
        
        # Normal health routing
        normal_result = self.intelligent_router.route_query(test_query)
        
        health_impact_results = {
            'system_health_status': health_status,
            'normal_routing': {
                'decision': normal_result.routing_decision.value,
                'confidence': normal_result.confidence,
                'backend': normal_result.metadata.get('selected_backend'),
                'health_impacted': normal_result.metadata.get('health_impacted_routing', False)
            }
        }
        
        return health_impact_results
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive accuracy validation report"""
        report = []
        report.append("="*80)
        report.append("ROUTING ACCURACY VALIDATION REPORT")
        report.append(f"Generated: {results['timestamp']}")
        report.append("="*80)
        
        # Overall metrics
        overall = results['overall_metrics']
        report.append(f"\nOVERALL ACCURACY METRICS:")
        report.append(f"{'='*40}")
        report.append(f"Total Test Queries: {overall['total_test_queries']}")
        report.append(f"Base Router Accuracy: {overall['base_router_accuracy']:.2f}%")
        report.append(f"Intelligent Router Accuracy: {overall['intelligent_router_accuracy']:.2f}%")
        report.append(f"Target Accuracy (>90%) Met: {'✓ YES' if overall['target_accuracy_met'] else '✗ NO'}")
        report.append(f"Intelligent Router Target Met: {'✓ YES' if overall['intelligent_target_accuracy_met'] else '✗ NO'}")
        
        if 'processing_time_stats' in overall:
            pt_stats = overall['processing_time_stats']
            report.append(f"\nPROCESSING TIME PERFORMANCE:")
            report.append(f"Mean Response Time: {pt_stats['mean_ms']:.2f}ms")
            report.append(f"95th Percentile: {pt_stats['p95_ms']:.2f}ms")
            report.append(f"Max Response Time: {pt_stats['max_ms']:.2f}ms")
        
        # Category-specific results
        report.append(f"\nCATEGORY-SPECIFIC ACCURACY:")
        report.append(f"{'='*40}")
        
        for category, cat_results in results['category_results'].items():
            accuracy = cat_results['accuracy_percentage']
            correct = cat_results['correct_predictions']
            total = cat_results['total_queries']
            
            status = "✓ PASS" if accuracy >= 90 else "✗ FAIL"
            report.append(f"{category.upper()}: {accuracy:.2f}% ({correct}/{total}) {status}")
        
        # Failure analysis
        failed_categories = [cat for cat, res in results['category_results'].items() 
                           if res['accuracy_percentage'] < 90]
        
        if failed_categories:
            report.append(f"\nFAILED CATEGORIES ANALYSIS:")
            report.append(f"{'='*40}")
            
            for category in failed_categories:
                cat_results = results['category_results'][category]
                report.append(f"\n{category.upper()} Failures:")
                
                failed_tests = [test for test in cat_results['details'] 
                              if not test.get('base_correct', False)][:5]  # Show first 5 failures
                
                for test in failed_tests:
                    if 'error' in test:
                        report.append(f"  - Query {test['query_id']}: ERROR - {test['error']}")
                    else:
                        report.append(f"  - Query {test['query_id']}: Expected {test['expected']}, Got {test['base_router_decision']}")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        report.append(f"{'='*40}")
        
        base_accuracy = overall['base_router_accuracy']
        
        if base_accuracy >= 90:
            report.append("✓ Routing accuracy meets >90% requirement")
            report.append("✓ System is ready for production deployment")
        else:
            report.append("✗ Routing accuracy below 90% requirement")
            report.append("✗ Additional training/tuning needed before production")
            
            if failed_categories:
                report.append(f"- Focus improvement efforts on: {', '.join(failed_categories)}")
        
        if 'processing_time_stats' in overall:
            mean_time = overall['processing_time_stats']['mean_ms']
            if mean_time > 100:
                report.append("- Consider optimizing routing performance (>100ms average)")
            elif mean_time > 50:
                report.append("- Monitor routing performance (>50ms average)")
        
        return "\n".join(report)


def main():
    """Main execution function"""
    print("Initializing Routing Accuracy Validator...")
    
    try:
        validator = RoutingAccuracyValidator()
        
        # Run comprehensive accuracy tests
        print("Running comprehensive accuracy validation...")
        accuracy_results = validator.test_routing_accuracy()
        
        # Run load balancing tests
        print("Running load balancing validation...")
        load_balancing_results = validator.test_load_balancing_accuracy()
        
        # Run system health tests
        print("Running system health impact validation...")
        health_results = validator.test_system_health_impact()
        
        # Combine all results
        comprehensive_results = {
            **accuracy_results,
            'load_balancing_results': load_balancing_results,
            'system_health_results': health_results
        }
        
        # Generate and display report
        report = validator.generate_comprehensive_report(comprehensive_results)
        print(report)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"routing_accuracy_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        report_file = f"routing_accuracy_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Report saved to: {report_file}")
        
        # Return success status
        overall_accuracy = comprehensive_results['overall_metrics']['base_router_accuracy']
        return overall_accuracy >= 90.0
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)