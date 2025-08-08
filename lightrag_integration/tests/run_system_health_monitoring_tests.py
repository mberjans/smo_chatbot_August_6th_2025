#!/usr/bin/env python3
"""
System Health Monitoring Integration Test Runner

This script runs comprehensive system health monitoring integration tests
and generates detailed reports on the test results and system resilience.

Features:
- Comprehensive test execution with performance monitoring
- Detailed reporting and analysis
- Health monitoring validation
- Circuit breaker effectiveness assessment
- Service availability impact analysis

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: Health monitoring integration test validation
"""

import sys
import os
import time
import json
import logging
import subprocess
import statistics
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

# Add lightrag_integration to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(current_dir / 'logs' / 'health_monitoring_tests.log')
    ]
)
logger = logging.getLogger(__name__)


class HealthMonitoringTestRunner:
    """Comprehensive test runner for health monitoring integration."""
    
    def __init__(self):
        """Initialize test runner."""
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = time.time()
        self.test_file = current_dir / 'test_system_health_monitoring_integration.py'
        
        # Ensure logs directory exists
        (current_dir / 'logs').mkdir(exist_ok=True)
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive health monitoring integration tests."""
        logger.info("=" * 80)
        logger.info("STARTING SYSTEM HEALTH MONITORING INTEGRATION TESTS")
        logger.info("=" * 80)
        
        # Run test categories
        test_categories = [
            ('circuit_breaker', 'TestCircuitBreakerIntegration'),
            ('health_routing', 'TestHealthBasedRoutingDecisions'),
            ('failure_recovery', 'TestFailureDetectionAndRecovery'), 
            ('performance_monitoring', 'TestPerformanceMonitoring'),
            ('load_balancing', 'TestLoadBalancing'),
            ('service_availability', 'TestServiceAvailabilityImpact'),
            ('integration', 'TestHealthMonitoringIntegration')
        ]
        
        overall_results = {
            'test_categories': {},
            'performance_summary': {},
            'health_monitoring_effectiveness': {},
            'recommendations': []
        }
        
        for category, test_class in test_categories:
            logger.info(f"\n--- Running {category} tests ({test_class}) ---")
            category_results = self._run_test_category(category, test_class)
            overall_results['test_categories'][category] = category_results
        
        # Generate performance summary
        overall_results['performance_summary'] = self._generate_performance_summary()
        
        # Assess health monitoring effectiveness
        overall_results['health_monitoring_effectiveness'] = self._assess_health_monitoring_effectiveness()
        
        # Generate recommendations
        overall_results['recommendations'] = self._generate_recommendations(overall_results)
        
        # Calculate total execution time
        total_time = time.time() - self.start_time
        overall_results['execution_time_seconds'] = total_time
        overall_results['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"\n" + "=" * 80)
        logger.info(f"HEALTH MONITORING INTEGRATION TESTS COMPLETED")
        logger.info(f"Total Execution Time: {total_time:.2f} seconds")
        logger.info(f"=" * 80)
        
        return overall_results
    
    def _run_test_category(self, category: str, test_class: str) -> Dict[str, Any]:
        """Run a specific category of tests."""
        start_time = time.time()
        
        try:
            # Run pytest for specific test class
            cmd = [
                sys.executable, '-m', 'pytest',
                str(self.test_file),
                f'-k', test_class,
                '-v',
                '--tb=short',
                '--json-report',
                f'--json-report-file={current_dir}/logs/{category}_results.json'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(project_root)
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            try:
                with open(current_dir / 'logs' / f'{category}_results.json') as f:
                    detailed_results = json.load(f)
            except FileNotFoundError:
                detailed_results = None
            
            category_result = {
                'success': result.returncode == 0,
                'execution_time_seconds': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'detailed_results': detailed_results
            }
            
            if result.returncode == 0:
                logger.info(f"✅ {category} tests PASSED ({execution_time:.2f}s)")
            else:
                logger.error(f"❌ {category} tests FAILED ({execution_time:.2f}s)")
                logger.error(f"Error output: {result.stderr}")
            
            return category_result
            
        except Exception as e:
            logger.error(f"Exception running {category} tests: {e}")
            return {
                'success': False,
                'execution_time_seconds': time.time() - start_time,
                'error': str(e),
                'exception_type': type(e).__name__
            }
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from test results."""
        summary = {
            'total_test_categories': 0,
            'successful_categories': 0,
            'failed_categories': 0,
            'average_execution_time_seconds': 0,
            'performance_analysis': {}
        }
        
        execution_times = []
        successful_count = 0
        
        for category, results in self.test_results.items():
            summary['total_test_categories'] += 1
            
            if results.get('success', False):
                successful_count += 1
                summary['successful_categories'] += 1
            else:
                summary['failed_categories'] += 1
            
            exec_time = results.get('execution_time_seconds', 0)
            execution_times.append(exec_time)
        
        if execution_times:
            summary['average_execution_time_seconds'] = statistics.mean(execution_times)
            summary['max_execution_time_seconds'] = max(execution_times)
            summary['min_execution_time_seconds'] = min(execution_times)
        
        summary['success_rate_percentage'] = (successful_count / max(summary['total_test_categories'], 1)) * 100
        
        return summary
    
    def _assess_health_monitoring_effectiveness(self) -> Dict[str, Any]:
        """Assess the effectiveness of health monitoring integration."""
        effectiveness = {
            'circuit_breaker_effectiveness': 'unknown',
            'failure_detection_capability': 'unknown',
            'recovery_mechanisms': 'unknown',
            'performance_impact_handling': 'unknown',
            'load_balancing_quality': 'unknown',
            'service_availability_management': 'unknown',
            'overall_effectiveness_score': 0.0,
            'key_strengths': [],
            'areas_for_improvement': []
        }
        
        # Analyze circuit breaker effectiveness
        circuit_breaker_results = self.test_results.get('circuit_breaker', {})
        if circuit_breaker_results.get('success', False):
            effectiveness['circuit_breaker_effectiveness'] = 'excellent'
            effectiveness['key_strengths'].append('Circuit breaker patterns working correctly')
        else:
            effectiveness['circuit_breaker_effectiveness'] = 'needs_improvement'
            effectiveness['areas_for_improvement'].append('Circuit breaker implementation needs review')
        
        # Analyze failure detection
        failure_recovery_results = self.test_results.get('failure_recovery', {})
        if failure_recovery_results.get('success', False):
            effectiveness['failure_detection_capability'] = 'excellent'
            effectiveness['recovery_mechanisms'] = 'excellent'
            effectiveness['key_strengths'].append('Failure detection and recovery working well')
        else:
            effectiveness['failure_detection_capability'] = 'needs_improvement'
            effectiveness['recovery_mechanisms'] = 'needs_improvement'
            effectiveness['areas_for_improvement'].append('Failure detection/recovery needs enhancement')
        
        # Analyze performance monitoring
        performance_results = self.test_results.get('performance_monitoring', {})
        if performance_results.get('success', False):
            effectiveness['performance_impact_handling'] = 'excellent'
            effectiveness['key_strengths'].append('Performance monitoring integration effective')
        else:
            effectiveness['performance_impact_handling'] = 'needs_improvement'
            effectiveness['areas_for_improvement'].append('Performance monitoring integration needs work')
        
        # Analyze load balancing
        load_balancing_results = self.test_results.get('load_balancing', {})
        if load_balancing_results.get('success', False):
            effectiveness['load_balancing_quality'] = 'excellent'
            effectiveness['key_strengths'].append('Load balancing mechanisms effective')
        else:
            effectiveness['load_balancing_quality'] = 'needs_improvement'
            effectiveness['areas_for_improvement'].append('Load balancing implementation needs improvement')
        
        # Analyze service availability management
        availability_results = self.test_results.get('service_availability', {})
        if availability_results.get('success', False):
            effectiveness['service_availability_management'] = 'excellent'
            effectiveness['key_strengths'].append('Service availability impact handled well')
        else:
            effectiveness['service_availability_management'] = 'needs_improvement'
            effectiveness['areas_for_improvement'].append('Service availability management needs enhancement')
        
        # Calculate overall effectiveness score
        effectiveness_scores = []
        for key in ['circuit_breaker_effectiveness', 'failure_detection_capability', 'recovery_mechanisms', 
                   'performance_impact_handling', 'load_balancing_quality', 'service_availability_management']:
            if effectiveness[key] == 'excellent':
                effectiveness_scores.append(1.0)
            elif effectiveness[key] == 'good':
                effectiveness_scores.append(0.7)
            elif effectiveness[key] == 'needs_improvement':
                effectiveness_scores.append(0.3)
            else:
                effectiveness_scores.append(0.0)
        
        effectiveness['overall_effectiveness_score'] = statistics.mean(effectiveness_scores) if effectiveness_scores else 0.0
        
        return effectiveness
    
    def _generate_recommendations(self, overall_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Performance recommendations
        performance_summary = overall_results.get('performance_summary', {})
        success_rate = performance_summary.get('success_rate_percentage', 0)
        
        if success_rate < 80:
            recommendations.append(
                "CRITICAL: Test success rate is below 80%. Review failed test categories and fix underlying issues."
            )
        elif success_rate < 95:
            recommendations.append(
                "WARNING: Test success rate could be improved. Investigate failed tests for optimization opportunities."
            )
        
        # Execution time recommendations
        avg_time = performance_summary.get('average_execution_time_seconds', 0)
        if avg_time > 30:
            recommendations.append(
                "PERFORMANCE: Test execution time is high. Consider optimizing test setup and teardown procedures."
            )
        
        # Health monitoring effectiveness recommendations
        effectiveness = overall_results.get('health_monitoring_effectiveness', {})
        overall_score = effectiveness.get('overall_effectiveness_score', 0)
        
        if overall_score < 0.6:
            recommendations.append(
                "CRITICAL: Health monitoring effectiveness is low. Review and enhance health monitoring integration."
            )
        elif overall_score < 0.8:
            recommendations.append(
                "IMPROVEMENT: Health monitoring could be enhanced. Focus on areas identified for improvement."
            )
        
        # Specific component recommendations
        if effectiveness.get('circuit_breaker_effectiveness') == 'needs_improvement':
            recommendations.append(
                "CIRCUIT BREAKER: Enhance circuit breaker implementation to improve failure handling."
            )
        
        if effectiveness.get('failure_detection_capability') == 'needs_improvement':
            recommendations.append(
                "FAILURE DETECTION: Improve failure detection mechanisms for better system resilience."
            )
        
        if effectiveness.get('load_balancing_quality') == 'needs_improvement':
            recommendations.append(
                "LOAD BALANCING: Enhance load balancing algorithms to better distribute requests based on health."
            )
        
        # Best practices recommendations
        if overall_score > 0.8:
            recommendations.append(
                "EXCELLENT: Health monitoring integration is working well. Consider documenting best practices."
            )
        
        recommendations.append(
            "MONITORING: Continue to monitor system health metrics in production to validate test assumptions."
        )
        
        recommendations.append(
            "DOCUMENTATION: Update system documentation to reflect health monitoring integration patterns."
        )
        
        return recommendations
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("SYSTEM HEALTH MONITORING INTEGRATION TEST REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append(f"Total Execution Time: {results.get('execution_time_seconds', 0):.2f} seconds")
        report_lines.append("")
        
        # Performance Summary
        performance = results.get('performance_summary', {})
        report_lines.append("PERFORMANCE SUMMARY")
        report_lines.append("-" * 50)
        report_lines.append(f"Total Test Categories: {performance.get('total_test_categories', 0)}")
        report_lines.append(f"Successful Categories: {performance.get('successful_categories', 0)}")
        report_lines.append(f"Failed Categories: {performance.get('failed_categories', 0)}")
        report_lines.append(f"Success Rate: {performance.get('success_rate_percentage', 0):.1f}%")
        report_lines.append(f"Average Execution Time: {performance.get('average_execution_time_seconds', 0):.2f}s")
        report_lines.append("")
        
        # Test Category Results
        report_lines.append("TEST CATEGORY RESULTS")
        report_lines.append("-" * 50)
        for category, category_results in results.get('test_categories', {}).items():
            status = "PASS" if category_results.get('success', False) else "FAIL"
            exec_time = category_results.get('execution_time_seconds', 0)
            report_lines.append(f"{category:25} | {status:4} | {exec_time:6.2f}s")
        report_lines.append("")
        
        # Health Monitoring Effectiveness
        effectiveness = results.get('health_monitoring_effectiveness', {})
        report_lines.append("HEALTH MONITORING EFFECTIVENESS")
        report_lines.append("-" * 50)
        report_lines.append(f"Overall Effectiveness Score: {effectiveness.get('overall_effectiveness_score', 0):.2f}/1.0")
        report_lines.append(f"Circuit Breaker: {effectiveness.get('circuit_breaker_effectiveness', 'unknown')}")
        report_lines.append(f"Failure Detection: {effectiveness.get('failure_detection_capability', 'unknown')}")
        report_lines.append(f"Recovery Mechanisms: {effectiveness.get('recovery_mechanisms', 'unknown')}")
        report_lines.append(f"Performance Monitoring: {effectiveness.get('performance_impact_handling', 'unknown')}")
        report_lines.append(f"Load Balancing: {effectiveness.get('load_balancing_quality', 'unknown')}")
        report_lines.append(f"Service Availability: {effectiveness.get('service_availability_management', 'unknown')}")
        report_lines.append("")
        
        # Key Strengths
        strengths = effectiveness.get('key_strengths', [])
        if strengths:
            report_lines.append("KEY STRENGTHS")
            report_lines.append("-" * 50)
            for strength in strengths:
                report_lines.append(f"✅ {strength}")
            report_lines.append("")
        
        # Areas for Improvement
        improvements = effectiveness.get('areas_for_improvement', [])
        if improvements:
            report_lines.append("AREAS FOR IMPROVEMENT")
            report_lines.append("-" * 50)
            for improvement in improvements:
                report_lines.append(f"⚠️  {improvement}")
            report_lines.append("")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 50)
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        report_lines.append("=" * 100)
        
        return "\n".join(report_lines)
    
    def save_results(self, results: Dict[str, Any]) -> Tuple[str, str]:
        """Save results to JSON and text report files."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = current_dir / 'logs' / f'health_monitoring_test_results_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save text report
        report_content = self.generate_report(results)
        report_file = current_dir / 'logs' / f'health_monitoring_test_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return str(json_file), str(report_file)


def main():
    """Main test runner function."""
    try:
        # Initialize test runner
        runner = HealthMonitoringTestRunner()
        
        # Run comprehensive tests
        results = runner.run_comprehensive_tests()
        
        # Store results for analysis
        runner.test_results = results.get('test_categories', {})
        
        # Save results and generate reports
        json_file, report_file = runner.save_results(results)
        
        # Print summary
        print("\n" + "=" * 80)
        print("HEALTH MONITORING INTEGRATION TEST SUMMARY")
        print("=" * 80)
        
        performance = results.get('performance_summary', {})
        print(f"Success Rate: {performance.get('success_rate_percentage', 0):.1f}%")
        print(f"Total Categories: {performance.get('total_test_categories', 0)}")
        print(f"Successful: {performance.get('successful_categories', 0)}")
        print(f"Failed: {performance.get('failed_categories', 0)}")
        
        effectiveness = results.get('health_monitoring_effectiveness', {})
        print(f"Health Monitoring Effectiveness: {effectiveness.get('overall_effectiveness_score', 0):.2f}/1.0")
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {report_file}")
        
        # Print report to console
        report_content = runner.generate_report(results)
        print("\n" + report_content)
        
        # Return appropriate exit code
        success_rate = performance.get('success_rate_percentage', 0)
        if success_rate >= 95:
            logger.info("🎉 All health monitoring tests passed successfully!")
            return 0
        elif success_rate >= 80:
            logger.warning("⚠️ Most health monitoring tests passed, but some issues detected.")
            return 1
        else:
            logger.error("❌ Significant health monitoring test failures detected.")
            return 2
            
    except Exception as e:
        logger.error(f"Fatal error in test runner: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)