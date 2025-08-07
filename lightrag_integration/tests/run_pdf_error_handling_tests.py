#!/usr/bin/env python3
"""
Test Runner for Comprehensive PDF Processing Error Handling Test Suite.

This script provides comprehensive execution and reporting for the PDF processing
error handling tests, including:

- Test execution with proper async support
- Detailed error reporting and analysis
- Performance metrics collection
- Test result summarization
- Integration with existing test infrastructure

Usage:
    python run_pdf_error_handling_tests.py [options]

Options:
    --category <category>    Run specific test category
    --verbose                Enable verbose output
    --fast                   Run only fast tests (skip slow/stress tests)
    --report                 Generate detailed test report
    --parallel               Run tests in parallel where possible

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import sys
import argparse
import logging
import time
import json
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import subprocess

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Test execution imports
import pytest
from unittest.mock import patch


class PDFErrorHandlingTestRunner:
    """Comprehensive test runner for PDF error handling tests."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize test runner with configuration."""
        self.config = config or {}
        self.start_time = None
        self.test_results = {}
        self.system_metrics = {}
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration for test execution."""
        log_level = logging.DEBUG if self.config.get('verbose', False) else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('pdf_error_handling_test_execution.log')
            ]
        )
    
    def collect_system_metrics(self):
        """Collect system metrics before and during test execution."""
        try:
            process = psutil.Process()
            system_info = {
                'timestamp': datetime.now().isoformat(),
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent,
                    'process_rss': process.memory_info().rss
                },
                'cpu': {
                    'count': psutil.cpu_count(),
                    'percent': psutil.cpu_percent(interval=1),
                    'process_percent': process.cpu_percent()
                },
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                }
            }
            return system_info
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
            return {}
    
    def run_test_category(self, category: str) -> Dict[str, Any]:
        """Run a specific category of tests."""
        self.logger.info(f"Running test category: {category}")
        
        # Map categories to test classes
        category_mapping = {
            'individual': 'TestIndividualPDFErrorHandling',
            'batch': 'TestBatchProcessingErrorHandling', 
            'knowledge_base': 'TestKnowledgeBaseIntegrationErrors',
            'recovery': 'TestRecoveryMechanisms',
            'stability': 'TestSystemStability',
            'all': None  # Run all tests
        }
        
        test_file = Path(__file__).parent / "test_pdf_processing_error_handling_comprehensive.py"
        
        pytest_args = [
            str(test_file),
            "-v",
            "--tb=short",
            "--maxfail=10",
            "--disable-warnings"
        ]
        
        # Add category filter if specified
        if category != 'all' and category in category_mapping:
            test_class = category_mapping[category]
            pytest_args.append(f"::{test_class}")
        
        # Add additional options
        if self.config.get('fast', False):
            pytest_args.extend(["-m", "not slow"])
        
        if self.config.get('parallel', False):
            pytest_args.extend(["-n", "auto"])
        
        # Collect metrics before test execution
        pre_test_metrics = self.collect_system_metrics()
        
        # Execute tests
        start_time = time.time()
        exit_code = pytest.main(pytest_args)
        end_time = time.time()
        
        # Collect metrics after test execution
        post_test_metrics = self.collect_system_metrics()
        
        # Compile results
        result = {
            'category': category,
            'exit_code': exit_code,
            'duration': end_time - start_time,
            'success': exit_code == 0,
            'pre_test_metrics': pre_test_metrics,
            'post_test_metrics': post_test_metrics
        }
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories and compile comprehensive results."""
        self.logger.info("Starting comprehensive PDF error handling test execution")
        self.start_time = time.time()
        
        # System baseline metrics
        baseline_metrics = self.collect_system_metrics()
        
        # Categories to test
        test_categories = [
            'individual',
            'batch', 
            'knowledge_base',
            'recovery',
            'stability'
        ]
        
        results = {
            'start_time': self.start_time,
            'baseline_metrics': baseline_metrics,
            'category_results': {},
            'overall_success': True,
            'total_duration': 0,
            'summary': {}
        }
        
        # Run each category
        for category in test_categories:
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"EXECUTING: {category.upper()} ERROR HANDLING TESTS")
                self.logger.info(f"{'='*60}")
                
                category_result = self.run_test_category(category)
                results['category_results'][category] = category_result
                
                if not category_result['success']:
                    results['overall_success'] = False
                    self.logger.error(f"Category {category} failed with exit code {category_result['exit_code']}")
                else:
                    self.logger.info(f"Category {category} completed successfully in {category_result['duration']:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to execute category {category}: {e}")
                results['category_results'][category] = {
                    'success': False,
                    'error': str(e),
                    'duration': 0
                }
                results['overall_success'] = False
        
        # Calculate totals
        total_duration = sum(
            result.get('duration', 0) 
            for result in results['category_results'].values()
        )
        results['total_duration'] = total_duration
        
        # Generate summary
        results['summary'] = self._generate_test_summary(results)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PDF ERROR HANDLING TEST EXECUTION COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Overall Success: {results['overall_success']}")
        self.logger.info(f"Total Duration: {total_duration:.2f}s")
        self.logger.info(f"Categories Passed: {results['summary']['categories_passed']}/{results['summary']['total_categories']}")
        
        return results
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test execution summary."""
        category_results = results['category_results']
        
        summary = {
            'total_categories': len(category_results),
            'categories_passed': sum(1 for r in category_results.values() if r.get('success', False)),
            'categories_failed': sum(1 for r in category_results.values() if not r.get('success', False)),
            'total_duration': results['total_duration'],
            'average_duration': results['total_duration'] / len(category_results) if category_results else 0,
            'system_impact': self._calculate_system_impact(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        return summary
    
    def _calculate_system_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system resource impact during test execution."""
        baseline = results.get('baseline_metrics', {})
        category_results = results.get('category_results', {})
        
        if not baseline or not category_results:
            return {}
        
        # Calculate peak resource usage across all categories
        peak_memory_percent = baseline.get('memory', {}).get('percent', 0)
        peak_cpu_percent = baseline.get('cpu', {}).get('percent', 0)
        
        for category_result in category_results.values():
            post_metrics = category_result.get('post_test_metrics', {})
            if post_metrics:
                memory_percent = post_metrics.get('memory', {}).get('percent', 0)
                cpu_percent = post_metrics.get('cpu', {}).get('percent', 0)
                peak_memory_percent = max(peak_memory_percent, memory_percent)
                peak_cpu_percent = max(peak_cpu_percent, cpu_percent)
        
        return {
            'peak_memory_percent': peak_memory_percent,
            'peak_cpu_percent': peak_cpu_percent,
            'memory_increase_percent': peak_memory_percent - baseline.get('memory', {}).get('percent', 0),
            'cpu_increase_percent': peak_cpu_percent - baseline.get('cpu', {}).get('percent', 0),
            'impact_level': self._assess_impact_level(peak_memory_percent, peak_cpu_percent)
        }
    
    def _assess_impact_level(self, memory_percent: float, cpu_percent: float) -> str:
        """Assess the impact level of test execution on system resources."""
        if memory_percent > 90 or cpu_percent > 95:
            return "HIGH"
        elif memory_percent > 80 or cpu_percent > 85:
            return "MEDIUM"
        elif memory_percent > 70 or cpu_percent > 75:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test execution results."""
        recommendations = []
        
        # Analyze failed categories
        failed_categories = [
            category for category, result in results['category_results'].items()
            if not result.get('success', False)
        ]
        
        if failed_categories:
            recommendations.append(
                f"Review and address failures in categories: {', '.join(failed_categories)}"
            )
        
        # Analyze performance
        total_duration = results.get('total_duration', 0)
        if total_duration > 300:  # 5 minutes
            recommendations.append(
                "Consider optimizing test execution time - current duration exceeds 5 minutes"
            )
        
        # Analyze system impact
        system_impact = results.get('summary', {}).get('system_impact', {})
        impact_level = system_impact.get('impact_level', 'UNKNOWN')
        
        if impact_level == "HIGH":
            recommendations.append(
                "High system resource usage detected - consider running tests on dedicated test environment"
            )
        elif impact_level == "MEDIUM":
            recommendations.append(
                "Moderate system resource usage - monitor system performance during test execution"
            )
        
        if not recommendations:
            recommendations.append("All tests executed successfully with acceptable resource usage")
        
        return recommendations
    
    def generate_test_report(self, results: Dict[str, Any], output_file: Optional[str] = None):
        """Generate detailed test execution report."""
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'test_suite': 'PDF Processing Error Handling Comprehensive Tests',
                'version': '1.0.0',
                'execution_environment': {
                    'python_version': sys.version,
                    'system': results.get('baseline_metrics', {}),
                }
            },
            'execution_results': results,
            'detailed_analysis': {
                'category_performance': self._analyze_category_performance(results),
                'resource_utilization': self._analyze_resource_utilization(results),
                'error_patterns': self._analyze_error_patterns(results),
                'stability_assessment': self._assess_stability(results)
            }
        }
        
        # Save report
        if output_file:
            output_path = Path(output_file)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f'pdf_error_handling_test_report_{timestamp}.json')
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Test report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save test report: {e}")
            return None
    
    def _analyze_category_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance characteristics of each test category."""
        category_results = results.get('category_results', {})
        
        analysis = {}
        for category, result in category_results.items():
            duration = result.get('duration', 0)
            success = result.get('success', False)
            
            analysis[category] = {
                'duration_seconds': duration,
                'success': success,
                'performance_rating': self._rate_performance(duration, success),
                'resource_impact': self._analyze_category_resource_impact(result)
            }
        
        return analysis
    
    def _rate_performance(self, duration: float, success: bool) -> str:
        """Rate the performance of a test category."""
        if not success:
            return "FAILED"
        elif duration < 30:
            return "EXCELLENT"
        elif duration < 60:
            return "GOOD"
        elif duration < 120:
            return "ACCEPTABLE"
        else:
            return "SLOW"
    
    def _analyze_category_resource_impact(self, category_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource impact of a specific test category."""
        pre_metrics = category_result.get('pre_test_metrics', {})
        post_metrics = category_result.get('post_test_metrics', {})
        
        if not pre_metrics or not post_metrics:
            return {}
        
        memory_change = (
            post_metrics.get('memory', {}).get('percent', 0) - 
            pre_metrics.get('memory', {}).get('percent', 0)
        )
        
        cpu_change = (
            post_metrics.get('cpu', {}).get('percent', 0) - 
            pre_metrics.get('cpu', {}).get('percent', 0)
        )
        
        return {
            'memory_change_percent': memory_change,
            'cpu_change_percent': cpu_change,
            'impact_assessment': 'HIGH' if abs(memory_change) > 20 or abs(cpu_change) > 30 else 'LOW'
        }
    
    def _analyze_resource_utilization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall resource utilization during test execution."""
        return results.get('summary', {}).get('system_impact', {})
    
    def _analyze_error_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns from test execution."""
        category_results = results.get('category_results', {})
        
        error_analysis = {
            'failed_categories': [],
            'common_patterns': [],
            'severity_assessment': 'LOW'
        }
        
        for category, result in category_results.items():
            if not result.get('success', False):
                error_analysis['failed_categories'].append({
                    'category': category,
                    'error': result.get('error', 'Unknown error'),
                    'exit_code': result.get('exit_code', -1)
                })
        
        if len(error_analysis['failed_categories']) > len(category_results) / 2:
            error_analysis['severity_assessment'] = 'HIGH'
        elif len(error_analysis['failed_categories']) > 0:
            error_analysis['severity_assessment'] = 'MEDIUM'
        
        return error_analysis
    
    def _assess_stability(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system stability during test execution."""
        overall_success = results.get('overall_success', False)
        total_duration = results.get('total_duration', 0)
        system_impact = results.get('summary', {}).get('system_impact', {})
        
        stability_score = 100
        
        # Reduce score for failures
        if not overall_success:
            failed_count = len([
                r for r in results.get('category_results', {}).values() 
                if not r.get('success', False)
            ])
            stability_score -= failed_count * 20
        
        # Reduce score for high resource usage
        impact_level = system_impact.get('impact_level', 'MINIMAL')
        if impact_level == 'HIGH':
            stability_score -= 30
        elif impact_level == 'MEDIUM':
            stability_score -= 15
        elif impact_level == 'LOW':
            stability_score -= 5
        
        # Reduce score for excessive duration
        if total_duration > 600:  # 10 minutes
            stability_score -= 20
        elif total_duration > 300:  # 5 minutes
            stability_score -= 10
        
        stability_score = max(0, stability_score)
        
        return {
            'stability_score': stability_score,
            'stability_rating': self._get_stability_rating(stability_score),
            'overall_success': overall_success,
            'duration_assessment': 'ACCEPTABLE' if total_duration < 300 else 'EXCESSIVE',
            'resource_impact_assessment': impact_level
        }
    
    def _get_stability_rating(self, score: int) -> str:
        """Get stability rating based on score."""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 70:
            return "ACCEPTABLE"
        elif score >= 50:
            return "POOR"
        else:
            return "CRITICAL"


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive PDF Processing Error Handling Test Runner"
    )
    
    parser.add_argument(
        '--category',
        choices=['individual', 'batch', 'knowledge_base', 'recovery', 'stability', 'all'],
        default='all',
        help='Specific test category to run (default: all)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true', 
        help='Run only fast tests (skip slow/stress tests)'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed test report'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel where possible'
    )
    
    parser.add_argument(
        '--output',
        help='Output file for test report (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'verbose': args.verbose,
        'fast': args.fast,
        'parallel': args.parallel,
        'report': args.report,
        'output_file': args.output
    }
    
    # Initialize test runner
    runner = PDFErrorHandlingTestRunner(config)
    
    try:
        # Execute tests
        if args.category == 'all':
            results = runner.run_all_tests()
        else:
            result = runner.run_test_category(args.category)
            results = {
                'overall_success': result['success'],
                'total_duration': result['duration'],
                'category_results': {args.category: result}
            }
        
        # Generate report if requested
        if args.report:
            report_file = runner.generate_test_report(results, args.output)
            if report_file:
                print(f"\nDetailed test report saved to: {report_file}")
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_success'] else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()