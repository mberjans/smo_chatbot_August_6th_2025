"""
CMO Metrics Integration Utilities
=================================

Integration utilities for CMO metrics with existing performance framework.
Provides comprehensive analysis and baseline comparison capabilities.
"""

import asyncio
import logging
import json
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class CMOMetricsIntegrator:
    """Integration utilities for CMO metrics with existing performance framework."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_metrics: Dict[str, 'CMOLoadMetrics'] = {}
        self.baseline_metrics: Optional['CMOLoadMetrics'] = None
        
    def create_enhanced_metrics(self, test_name: str) -> 'CMOLoadMetrics':
        """Create enhanced CMO metrics instance."""
        from .cmo_metrics_and_integration import CMOLoadMetrics
        
        metrics = CMOLoadMetrics(
            test_name=test_name,
            start_time=datetime.now()
        )
        
        self.active_metrics[test_name] = metrics
        return metrics
    
    def integrate_with_existing_framework(self, 
                                        existing_metrics: 'ConcurrentLoadMetrics') -> 'CMOLoadMetrics':
        """Convert existing metrics to enhanced CMO metrics."""
        from .cmo_metrics_and_integration import CMOLoadMetrics
        
        # Create new CMO metrics instance
        cmo_metrics = CMOLoadMetrics(
            test_name=existing_metrics.test_name,
            start_time=existing_metrics.start_time,
            end_time=existing_metrics.end_time,
            total_users=existing_metrics.total_users,
            concurrent_peak=existing_metrics.concurrent_peak,
            total_operations=existing_metrics.total_operations,
            successful_operations=existing_metrics.successful_operations,
            failed_operations=existing_metrics.failed_operations
        )
        
        # Transfer existing data
        cmo_metrics.response_times = existing_metrics.response_times[:]
        cmo_metrics.throughput_samples = existing_metrics.throughput_samples[:]
        cmo_metrics.error_rates = existing_metrics.error_rates[:]
        cmo_metrics.memory_samples = existing_metrics.memory_samples[:]
        cmo_metrics.cpu_samples = existing_metrics.cpu_samples[:]
        cmo_metrics.component_metrics = existing_metrics.component_metrics.copy()
        
        # Initialize CMO-specific metrics based on existing data
        self._initialize_cmo_metrics_from_existing(cmo_metrics, existing_metrics)
        
        return cmo_metrics
    
    def _initialize_cmo_metrics_from_existing(self, 
                                            cmo_metrics: 'CMOLoadMetrics',
                                            existing_metrics: 'ConcurrentLoadMetrics'):
        """Initialize CMO-specific metrics from existing framework data."""
        
        # Infer LightRAG metrics from component metrics
        if 'rag_queries' in existing_metrics.component_metrics:
            rag_data = existing_metrics.component_metrics['rag_queries']
            cmo_metrics.lightrag_metrics.total_queries = rag_data.get('total', 0)
            cmo_metrics.lightrag_metrics.successful_queries = rag_data.get('successful', 0)
            cmo_metrics.lightrag_metrics.failed_queries = rag_data.get('failed', 0)
        
        # Infer cache metrics from existing cache data
        cmo_metrics.cache_metrics.l1_hits = existing_metrics.cache_hits
        cmo_metrics.cache_metrics.l1_misses = existing_metrics.cache_misses
        
        # Initialize circuit breaker metrics
        cmo_metrics.circuit_breaker_metrics.total_requests = existing_metrics.total_operations
        cmo_metrics.circuit_breaker_metrics.successful_requests = existing_metrics.successful_operations
        cmo_metrics.circuit_breaker_metrics.failed_requests = existing_metrics.failed_operations
        cmo_metrics.circuit_breaker_metrics.blocked_requests = existing_metrics.circuit_breaker_blocks
    
    async def run_comparative_analysis(self, 
                                     test_results: Dict[str, 'CMOLoadMetrics']) -> Dict[str, Any]:
        """Run comparative analysis across multiple CMO test results."""
        
        if not test_results:
            return {'error': 'No test results provided for analysis'}
        
        analysis = {
            'test_comparison': {},
            'performance_ranking': [],
            'regression_analysis': {},
            'optimization_opportunities': []
        }
        
        # Compare tests
        for test_name, metrics in test_results.items():
            comprehensive_analysis = metrics.generate_comprehensive_analysis()
            analysis['test_comparison'][test_name] = comprehensive_analysis
            
            # Add to performance ranking
            grade_score = self._grade_to_score(metrics.current_grade)
            analysis['performance_ranking'].append({
                'test_name': test_name,
                'grade': metrics.current_grade.value,
                'score': grade_score,
                'success_rate': metrics.get_success_rate(),
                'efficiency_score': metrics.get_resource_efficiency_score()['overall']
            })
        
        # Sort performance ranking
        analysis['performance_ranking'].sort(key=lambda x: x['score'], reverse=True)
        
        # Regression analysis if baseline exists
        if self.baseline_metrics:
            for test_name, metrics in test_results.items():
                regression_results = metrics.detect_performance_regressions(self.baseline_metrics)
                if regression_results['regressions_detected']:
                    analysis['regression_analysis'][test_name] = regression_results
        
        # Identify optimization opportunities
        analysis['optimization_opportunities'] = self._identify_optimization_opportunities(test_results)
        
        return analysis
    
    def _grade_to_score(self, grade) -> float:
        """Convert performance grade to numerical score."""
        from .cmo_metrics_and_integration import PerformanceGrade
        
        grade_scores = {
            PerformanceGrade.A_PLUS: 100.0,
            PerformanceGrade.A: 90.0,
            PerformanceGrade.B: 80.0,
            PerformanceGrade.C: 70.0,
            PerformanceGrade.D: 60.0,
            PerformanceGrade.F: 0.0
        }
        return grade_scores.get(grade, 0.0)
    
    def _identify_optimization_opportunities(self, 
                                           test_results: Dict[str, 'CMOLoadMetrics']) -> List[Dict[str, Any]]:
        """Identify optimization opportunities across test results."""
        opportunities = []
        
        # Analyze cache performance across tests
        cache_hit_rates = []
        for test_name, metrics in test_results.items():
            hit_rate = metrics.cache_metrics.get_overall_hit_rate()
            cache_hit_rates.append((test_name, hit_rate))
        
        # Find tests with low cache performance
        cache_hit_rates.sort(key=lambda x: x[1])
        if cache_hit_rates and cache_hit_rates[0][1] < 0.7:
            opportunities.append({
                'type': 'cache_optimization',
                'priority': 'high',
                'description': f"Cache hit rate in {cache_hit_rates[0][0]} ({cache_hit_rates[0][1]:.1%}) needs improvement",
                'recommendation': 'Optimize cache key strategies and TTL settings'
            })
        
        # Analyze LightRAG performance
        lightrag_performance = []
        for test_name, metrics in test_results.items():
            success_rate = metrics.lightrag_metrics.get_success_rate()
            lightrag_performance.append((test_name, success_rate))
        
        lightrag_performance.sort(key=lambda x: x[1])
        if lightrag_performance and lightrag_performance[0][1] < 0.95:
            opportunities.append({
                'type': 'lightrag_optimization',
                'priority': 'critical',
                'description': f"LightRAG success rate in {lightrag_performance[0][0]} ({lightrag_performance[0][1]:.1%}) below target",
                'recommendation': 'Review query processing logic and error handling'
            })
        
        # Analyze resource efficiency
        efficiency_scores = []
        for test_name, metrics in test_results.items():
            efficiency = metrics.get_resource_efficiency_score()['overall']
            efficiency_scores.append((test_name, efficiency))
        
        efficiency_scores.sort(key=lambda x: x[1])
        if efficiency_scores and efficiency_scores[0][1] < 1.0:
            opportunities.append({
                'type': 'resource_optimization',
                'priority': 'medium',
                'description': f"Resource efficiency in {efficiency_scores[0][0]} ({efficiency_scores[0][1]:.2f}) could be improved",
                'recommendation': 'Optimize memory usage and CPU utilization patterns'
            })
        
        return opportunities
    
    def save_baseline_metrics(self, metrics: 'CMOLoadMetrics', filepath: str):
        """Save metrics as baseline for future regression analysis."""
        try:
            baseline_data = {
                'timestamp': datetime.now().isoformat(),
                'test_name': metrics.test_name,
                'comprehensive_analysis': metrics.generate_comprehensive_analysis(),
                'raw_metrics': {
                    'success_rate': metrics.get_success_rate(),
                    'percentiles': metrics.get_advanced_percentiles(),
                    'resource_efficiency': metrics.get_resource_efficiency_score(),
                    'component_health': metrics.get_component_health_assessment()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(baseline_data, f, indent=2, default=str)
            
            self.baseline_metrics = metrics
            self.logger.info(f"Baseline metrics saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save baseline metrics: {e}")
    
    def load_baseline_metrics(self, filepath: str) -> bool:
        """Load baseline metrics for regression analysis."""
        try:
            with open(filepath, 'r') as f:
                baseline_data = json.load(f)
            
            from .cmo_metrics_and_integration import CMOLoadMetrics
            
            # Create baseline metrics object (simplified for regression analysis)
            self.baseline_metrics = CMOLoadMetrics(
                test_name=baseline_data.get('test_name', 'baseline'),
                start_time=datetime.now()
            )
            
            # Set key metrics for comparison
            raw_metrics = baseline_data.get('raw_metrics', {})
            if 'percentiles' in raw_metrics:
                self.baseline_metrics.response_times = [raw_metrics['percentiles'].get('p50', 1000)] * 100
            
            self.logger.info(f"Baseline metrics loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load baseline metrics: {e}")
            return False


def create_cmo_metrics_suite(enable_real_time_monitoring: bool = True) -> Dict[str, Any]:
    """Create complete CMO metrics and integration suite."""
    
    integrator = CMOMetricsIntegrator()
    
    suite = {
        'integrator': integrator,
        'create_metrics': integrator.create_enhanced_metrics,
        'enable_monitoring': enable_real_time_monitoring
    }
    
    # Integration with existing performance suite
    existing_suite = create_enhanced_performance_suite(enable_monitoring=True)
    suite.update(existing_suite)
    
    return suite


async def run_comprehensive_cmo_analysis(test_results: Dict[str, Any],
                                       metrics_suite: Dict[str, Any],
                                       save_baseline: bool = False,
                                       baseline_path: Optional[str] = None) -> Dict[str, Any]:
    """Run comprehensive CMO performance analysis."""
    
    integrator = metrics_suite.get('integrator')
    if not integrator:
        return {'error': 'No metrics integrator available'}
    
    from .cmo_metrics_and_integration import CMOLoadMetrics
    from .concurrent_load_framework import ConcurrentLoadMetrics
    
    # Convert test results to CMO metrics if needed
    cmo_results = {}
    for test_name, result in test_results.items():
        if isinstance(result, CMOLoadMetrics):
            cmo_results[test_name] = result
        elif isinstance(result, ConcurrentLoadMetrics):
            cmo_results[test_name] = integrator.integrate_with_existing_framework(result)
    
    # Run comparative analysis
    analysis = await integrator.run_comparative_analysis(cmo_results)
    
    # Save baseline if requested
    if save_baseline and cmo_results:
        best_test = analysis['performance_ranking'][0]['test_name'] if analysis['performance_ranking'] else list(cmo_results.keys())[0]
        baseline_path = baseline_path or f"cmo_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        integrator.save_baseline_metrics(cmo_results[best_test], baseline_path)
        analysis['baseline_saved'] = baseline_path
    
    # Add execution metadata
    analysis['execution_metadata'] = {
        'analysis_timestamp': datetime.now().isoformat(),
        'tests_analyzed': list(cmo_results.keys()),
        'total_tests': len(cmo_results),
        'analysis_type': 'comprehensive_cmo_analysis'
    }
    
    return analysis


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting CMO metrics and integration system demo")
        
        # Create CMO metrics suite
        metrics_suite = create_cmo_metrics_suite(enable_real_time_monitoring=True)
        
        # Create sample metrics
        test_metrics = metrics_suite['create_metrics']("demo_cmo_test")
        
        # Start real-time monitoring
        await test_metrics.start_real_time_monitoring()
        
        # Simulate some performance data
        import random
        for i in range(50):
            test_metrics.add_response_time(random.uniform(200, 1500))
            test_metrics.total_operations += 1
            test_metrics.successful_operations += 1 if random.random() > 0.05 else 0
            
            # Simulate LightRAG metrics
            test_metrics.lightrag_metrics.total_queries += 1
            if random.random() > 0.03:
                test_metrics.lightrag_metrics.successful_queries += 1
                test_metrics.lightrag_metrics.hybrid_mode_queries += 1
            else:
                test_metrics.lightrag_metrics.failed_queries += 1
            
            await asyncio.sleep(0.1)  # 100ms intervals
        
        # Stop monitoring and generate analysis
        await test_metrics.stop_real_time_monitoring()
        
        # Generate comprehensive analysis
        analysis = test_metrics.generate_comprehensive_analysis()
        
        print("\nCMO Metrics Analysis Results:")
        print(f"Performance Grade: {analysis['executive_summary']['current_grade']}")
        print(f"Overall Success Rate: {analysis['executive_summary']['overall_success_rate']:.2%}")
        print(f"System Health: {analysis['executive_summary']['system_health']}")
        
        print(f"\nLightRAG Success Rate: {analysis['cmo_specific_analysis']['lightrag_performance']['success_rate']:.2%}")
        print(f"Cache Effectiveness: {analysis['cmo_specific_analysis']['cache_analysis']['effectiveness_score']:.2f}")
        
        if analysis['recommendations']:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(analysis['recommendations'][:3], 1):
                print(f"{i}. {rec}")
        
        # Test comparative analysis
        mock_results = {'demo_test': test_metrics}
        comparative_analysis = await run_comprehensive_cmo_analysis(
            mock_results, 
            metrics_suite,
            save_baseline=True
        )
        
        print(f"\nComparative Analysis completed:")
        print(f"- Tests analyzed: {len(comparative_analysis['test_comparison'])}")
        print(f"- Optimization opportunities: {len(comparative_analysis['optimization_opportunities'])}")
        
    asyncio.run(main())