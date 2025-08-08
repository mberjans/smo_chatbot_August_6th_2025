#!/usr/bin/env python3
"""
Comprehensive Test Runner for LLM Classification System

This script runs all comprehensive tests for the LLM-based classification system
with full coverage analysis, performance benchmarking, and quality reporting.

Features:
    - Comprehensive test suite execution
    - Automated coverage analysis with >95% target
    - Performance benchmarking and regression detection
    - Quality metrics and recommendations
    - Multiple report formats (HTML, JSON, text)
    - CI/CD integration support

Usage:
    python run_comprehensive_tests.py [options]
    
    # Run all tests with HTML report
    python run_comprehensive_tests.py
    
    # Run specific test categories
    python run_comprehensive_tests.py --categories core performance integration
    
    # Generate all report formats
    python run_comprehensive_tests.py --format all
    
    # CI/CD mode with JSON output
    python run_comprehensive_tests.py --ci --format json

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import test coverage infrastructure
try:
    from lightrag_integration.tests.test_coverage_config import CoverageAnalyzer, TestResults
except ImportError as e:
    print(f"Warning: Could not import coverage analyzer: {e}")
    CoverageAnalyzer = None
    TestResults = None


# ============================================================================
# TEST CATEGORIES AND CONFIGURATIONS
# ============================================================================

TEST_CATEGORIES = {
    'core': {
        'description': 'Core LLM classifier functionality tests',
        'patterns': [
            'test_llm_query_classifier',
            'test_classification_cache',
            'test_fallback_mechanisms'
        ],
        'timeout': 300,  # 5 minutes
        'critical': True
    },
    'confidence': {
        'description': 'Confidence scoring and calibration tests',
        'patterns': [
            'test_hybrid_confidence',
            'test_confidence_calibration',
            'test_uncertainty_quantification'
        ],
        'timeout': 240,  # 4 minutes
        'critical': True
    },
    'performance': {
        'description': 'Performance optimization and load tests',
        'patterns': [
            'test_response_time',
            'test_concurrent_requests',
            'test_cache_efficiency',
            'test_memory_usage'
        ],
        'timeout': 600,  # 10 minutes
        'critical': True
    },
    'integration': {
        'description': 'Integration compatibility tests',
        'patterns': [
            'test_routing_prediction_compatibility',
            'test_confidence_metrics_integration',
            'test_backwards_compatibility'
        ],
        'timeout': 180,  # 3 minutes
        'critical': True
    },
    'edge_cases': {
        'description': 'Edge cases and error handling tests',
        'patterns': [
            'test_api_failure',
            'test_malformed_response',
            'test_resource_exhaustion',
            'test_invalid_input'
        ],
        'timeout': 300,  # 5 minutes
        'critical': False
    },
    'quality': {
        'description': 'Code quality and validation tests',
        'patterns': [
            'test_fixtures_comprehensive',
            'test_mock_data_generation',
            'test_validation_accuracy'
        ],
        'timeout': 180,  # 3 minutes
        'critical': False
    }
}

QUALITY_THRESHOLDS = {
    'coverage_target': 95.0,
    'performance_target_ms': 2000,  # <2 second response time
    'test_success_rate': 98.0,
    'code_quality_min': 80.0,
    'critical_function_coverage': 98.0
}


# ============================================================================
# COMPREHENSIVE TEST RUNNER
# ============================================================================

class ComprehensiveTestRunner:
    """Comprehensive test runner with quality validation."""
    
    def __init__(self, 
                 source_dir: Optional[str] = None,
                 test_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.logger = logger or self._setup_logger()
        
        # Set directories
        current_dir = Path(__file__).parent
        self.source_dir = Path(source_dir) if source_dir else current_dir / "lightrag_integration"
        self.test_dir = Path(test_dir) if test_dir else current_dir / "lightrag_integration" / "tests"
        self.output_dir = Path(output_dir) if output_dir else current_dir / "test_reports"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize coverage analyzer if available
        if CoverageAnalyzer:
            self.coverage_analyzer = CoverageAnalyzer(
                source_dir=str(self.source_dir),
                test_dir=str(self.test_dir),
                output_dir=str(self.output_dir),
                logger=self.logger
            )
        else:
            self.coverage_analyzer = None
            self.logger.warning("Coverage analyzer not available - running basic tests only")
        
        self.logger.info(f"Test runner initialized - Source: {self.source_dir}, Tests: {self.test_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for test runner."""
        
        logger = logging.getLogger('test_runner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_category_tests(self, categories: List[str], format: str = "html") -> Dict[str, Any]:
        """Run tests for specific categories."""
        
        self.logger.info(f"Running tests for categories: {', '.join(categories)}")
        
        results = {
            'categories_run': categories,
            'start_time': datetime.now(),
            'category_results': {},
            'overall_success': True,
            'summary': {}
        }
        
        total_start_time = time.time()
        
        for category in categories:
            if category not in TEST_CATEGORIES:
                self.logger.error(f"Unknown test category: {category}")
                results['overall_success'] = False
                continue
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running {category.upper()} tests")
            self.logger.info(f"{'='*60}")
            
            category_config = TEST_CATEGORIES[category]
            category_start_time = time.time()
            
            try:
                # Run tests for this category
                if self.coverage_analyzer:
                    category_results = self.coverage_analyzer.run_tests_with_coverage(
                        category_config['patterns']
                    )
                else:
                    category_results = self._run_basic_tests(category_config['patterns'])
                
                category_end_time = time.time()
                category_results.category_execution_time = category_end_time - category_start_time
                
                # Validate results against thresholds
                validation_result = self._validate_category_results(category, category_results)
                
                results['category_results'][category] = {
                    'config': category_config,
                    'results': category_results,
                    'validation': validation_result,
                    'execution_time': category_results.category_execution_time
                }
                
                # Update overall success
                if not validation_result['meets_thresholds'] and category_config['critical']:
                    results['overall_success'] = False
                
                # Log category summary
                self._log_category_summary(category, category_results, validation_result)
                
            except Exception as e:
                self.logger.error(f"Failed to run {category} tests: {e}")
                results['category_results'][category] = {
                    'error': str(e),
                    'success': False
                }
                if category_config['critical']:
                    results['overall_success'] = False
        
        total_end_time = time.time()
        results['total_execution_time'] = total_end_time - total_start_time
        results['end_time'] = datetime.now()
        
        # Generate comprehensive summary
        results['summary'] = self._generate_test_summary(results)
        
        # Generate reports
        if format in ['html', 'json', 'text', 'all']:
            results['report_files'] = self._generate_reports(results, format)
        
        return results
    
    def _run_basic_tests(self, patterns: List[str]) -> Any:
        """Fallback method to run basic tests without coverage analysis."""
        
        self.logger.warning("Running basic tests without coverage analysis")
        
        # Create a mock TestResults object
        class MockTestResults:
            def __init__(self):
                self.total_tests = 0
                self.passed_tests = 0
                self.failed_tests = 0
                self.overall_coverage = 0.0
                self.total_execution_time = 0.0
                self.recommendations = []
        
        try:
            import subprocess
            
            # Run pytest directly
            cmd = ['python', '-m', 'pytest', str(self.test_dir), '-v']
            for pattern in patterns:
                cmd.extend(['-k', pattern])
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            end_time = time.time()
            
            mock_result = MockTestResults()
            mock_result.total_execution_time = end_time - start_time
            
            # Parse basic results from pytest output
            output_lines = result.stdout.split('\n') + result.stderr.split('\n')
            for line in output_lines:
                if 'passed' in line and 'failed' in line:
                    # Try to parse pytest summary
                    try:
                        parts = line.split()
                        if 'passed' in parts:
                            passed_idx = parts.index('passed')
                            mock_result.passed_tests = int(parts[passed_idx - 1])
                        if 'failed' in parts:
                            failed_idx = parts.index('failed')
                            mock_result.failed_tests = int(parts[failed_idx - 1])
                    except:
                        pass
            
            mock_result.total_tests = mock_result.passed_tests + mock_result.failed_tests
            
            return mock_result
            
        except Exception as e:
            self.logger.error(f"Basic test execution failed: {e}")
            return MockTestResults()
    
    def _validate_category_results(self, category: str, results: Any) -> Dict[str, Any]:
        """Validate test results against quality thresholds."""
        
        validation = {
            'category': category,
            'meets_thresholds': True,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Extract metrics
            coverage = getattr(results, 'overall_coverage', 0.0)
            execution_time = getattr(results, 'total_execution_time', 0.0)
            passed_tests = getattr(results, 'passed_tests', 0)
            total_tests = getattr(results, 'total_tests', 0)
            failed_tests = getattr(results, 'failed_tests', 0)
            
            validation['metrics'] = {
                'coverage': coverage,
                'execution_time': execution_time,
                'success_rate': (passed_tests / max(total_tests, 1)) * 100,
                'total_tests': total_tests,
                'failed_tests': failed_tests
            }
            
            # Validate coverage
            if coverage < QUALITY_THRESHOLDS['coverage_target']:
                validation['issues'].append(
                    f"Coverage {coverage:.1f}% below target {QUALITY_THRESHOLDS['coverage_target']}%"
                )
                validation['meets_thresholds'] = False
            
            # Validate performance for performance category
            if category == 'performance':
                avg_test_time = getattr(results, 'average_test_time', 0.0) * 1000  # Convert to ms
                if avg_test_time > QUALITY_THRESHOLDS['performance_target_ms']:
                    validation['issues'].append(
                        f"Average test time {avg_test_time:.0f}ms exceeds {QUALITY_THRESHOLDS['performance_target_ms']}ms"
                    )
                    validation['meets_thresholds'] = False
            
            # Validate test success rate
            success_rate = validation['metrics']['success_rate']
            if success_rate < QUALITY_THRESHOLDS['test_success_rate']:
                validation['issues'].append(
                    f"Test success rate {success_rate:.1f}% below target {QUALITY_THRESHOLDS['test_success_rate']}%"
                )
                validation['meets_thresholds'] = False
            
            # Check for failed tests
            if failed_tests > 0:
                validation['issues'].append(f"{failed_tests} tests failed")
                validation['meets_thresholds'] = False
            
        except Exception as e:
            validation['issues'].append(f"Validation error: {e}")
            validation['meets_thresholds'] = False
        
        return validation
    
    def _log_category_summary(self, category: str, results: Any, validation: Dict[str, Any]):
        """Log summary for a test category."""
        
        status = "✅ PASS" if validation['meets_thresholds'] else "❌ FAIL"
        
        self.logger.info(f"\n{category.upper()} RESULTS: {status}")
        self.logger.info("-" * 40)
        
        if hasattr(results, 'total_tests'):
            self.logger.info(f"Tests:    {results.passed_tests}/{results.total_tests} passed")
        
        if hasattr(results, 'overall_coverage'):
            self.logger.info(f"Coverage: {results.overall_coverage:.1f}%")
        
        if hasattr(results, 'total_execution_time'):
            self.logger.info(f"Time:     {results.total_execution_time:.2f}s")
        
        if validation['issues']:
            self.logger.warning("Issues:")
            for issue in validation['issues']:
                self.logger.warning(f"  • {issue}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                self.logger.warning(f"Warning: {warning}")
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall test summary."""
        
        summary = {
            'overall_success': results['overall_success'],
            'categories_run': len(results['categories_run']),
            'critical_categories_passed': 0,
            'total_tests': 0,
            'total_passed': 0,
            'total_failed': 0,
            'average_coverage': 0.0,
            'total_execution_time': results['total_execution_time'],
            'key_metrics': {},
            'recommendations': []
        }
        
        coverage_values = []
        
        for category, category_data in results['category_results'].items():
            if 'results' in category_data:
                cat_results = category_data['results']
                cat_validation = category_data['validation']
                
                # Aggregate test counts
                if hasattr(cat_results, 'total_tests'):
                    summary['total_tests'] += cat_results.total_tests
                    summary['total_passed'] += cat_results.passed_tests
                    summary['total_failed'] += cat_results.failed_tests
                
                # Collect coverage values
                if hasattr(cat_results, 'overall_coverage'):
                    coverage_values.append(cat_results.overall_coverage)
                
                # Count critical category passes
                if TEST_CATEGORIES[category]['critical'] and cat_validation['meets_thresholds']:
                    summary['critical_categories_passed'] += 1
        
        # Calculate average coverage
        if coverage_values:
            summary['average_coverage'] = sum(coverage_values) / len(coverage_values)
        
        # Key metrics
        summary['key_metrics'] = {
            'success_rate': (summary['total_passed'] / max(summary['total_tests'], 1)) * 100,
            'critical_success_rate': (summary['critical_categories_passed'] / 
                                    len([c for c in results['categories_run'] 
                                         if TEST_CATEGORIES.get(c, {}).get('critical', False)])) * 100,
            'average_coverage': summary['average_coverage']
        }
        
        # Generate recommendations
        if not results['overall_success']:
            summary['recommendations'].append("Address failing tests and quality issues")
        
        if summary['average_coverage'] < QUALITY_THRESHOLDS['coverage_target']:
            summary['recommendations'].append(f"Increase test coverage to >{QUALITY_THRESHOLDS['coverage_target']}%")
        
        if summary['total_failed'] > 0:
            summary['recommendations'].append(f"Fix {summary['total_failed']} failing tests")
        
        return summary
    
    def _generate_reports(self, results: Dict[str, Any], format: str) -> List[str]:
        """Generate comprehensive test reports."""
        
        if not self.coverage_analyzer:
            return []
        
        report_files = []
        timestamp = results['start_time'].strftime("%Y%m%d_%H%M%S")
        
        # Combine all category results into a single comprehensive result
        combined_results = self._combine_category_results(results)
        
        try:
            if format == "all":
                formats = ["html", "json", "text"]
            else:
                formats = [format]
            
            for fmt in formats:
                report_file = self.coverage_analyzer.generate_comprehensive_report(combined_results, fmt)
                report_files.append(report_file)
                self.logger.info(f"Generated {fmt.upper()} report: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
        
        return report_files
    
    def _combine_category_results(self, results: Dict[str, Any]) -> Any:
        """Combine results from all categories into a single TestResults object."""
        
        if not TestResults:
            return None
        
        combined = TestResults()
        combined.timestamp = results['start_time']
        combined.total_execution_time = results['total_execution_time']
        
        # Aggregate metrics from all categories
        for category, category_data in results['category_results'].items():
            if 'results' in category_data:
                cat_results = category_data['results']
                
                if hasattr(cat_results, 'total_tests'):
                    combined.total_tests += cat_results.total_tests
                    combined.passed_tests += cat_results.passed_tests
                    combined.failed_tests += cat_results.failed_tests
                    combined.skipped_tests += getattr(cat_results, 'skipped_tests', 0)
                    combined.error_tests += getattr(cat_results, 'error_tests', 0)
                
                # Use average coverage (could be improved)
                if hasattr(cat_results, 'overall_coverage'):
                    combined.overall_coverage = max(combined.overall_coverage, cat_results.overall_coverage)
                
                # Combine recommendations
                if hasattr(cat_results, 'recommendations'):
                    combined.recommendations.extend(cat_results.recommendations)
        
        # Calculate average test time
        if combined.total_tests > 0:
            combined.average_test_time = combined.total_execution_time / combined.total_tests
        
        return combined


# ============================================================================
# MAIN EXECUTION AND CLI INTERFACE
# ============================================================================

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for LLM Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--categories", 
        nargs="+",
        choices=list(TEST_CATEGORIES.keys()) + ['all'],
        default=['all'],
        help="Test categories to run (default: all)"
    )
    
    parser.add_argument(
        "--format",
        choices=["html", "json", "text", "all"],
        default="html",
        help="Report format (default: html)"
    )
    
    parser.add_argument(
        "--source-dir",
        help="Source code directory"
    )
    
    parser.add_argument(
        "--test-dir", 
        help="Test directory"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for reports"
    )
    
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI/CD mode - minimal output, exit codes"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Verbose output"
    )
    
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available test categories"
    )
    
    args = parser.parse_args()
    
    # List categories and exit
    if args.list_categories:
        print("Available test categories:")
        for category, config in TEST_CATEGORIES.items():
            critical = "CRITICAL" if config['critical'] else "optional"
            print(f"  {category:12} - {config['description']} ({critical})")
        return 0
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.ci else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s' if not args.ci else '%(message)s'
    )
    
    # Determine categories to run
    if 'all' in args.categories:
        categories = list(TEST_CATEGORIES.keys())
    else:
        categories = args.categories
    
    # Create test runner
    runner = ComprehensiveTestRunner(
        source_dir=args.source_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir
    )
    
    if not args.ci:
        print("=" * 80)
        print("LLM CLASSIFICATION SYSTEM - COMPREHENSIVE TEST RUNNER")
        print("=" * 80)
        print(f"Categories: {', '.join(categories)}")
        print(f"Report Format: {args.format}")
        print(f"Output Directory: {runner.output_dir}")
        print("=" * 80)
    
    # Run tests
    start_time = time.time()
    results = runner.run_category_tests(categories, args.format)
    end_time = time.time()
    
    # Print final summary
    if not args.ci:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        summary = results['summary']
        
        print(f"Overall Success:        {'✅ YES' if results['overall_success'] else '❌ NO'}")
        print(f"Categories Run:         {summary['categories_run']}")
        print(f"Total Tests:           {summary['total_tests']}")
        print(f"Tests Passed:          {summary['total_passed']}")
        print(f"Tests Failed:          {summary['total_failed']}")
        print(f"Success Rate:          {summary['key_metrics']['success_rate']:.1f}%")
        print(f"Average Coverage:      {summary['average_coverage']:.1f}%")
        print(f"Total Execution Time:  {results['total_execution_time']:.2f}s")
        
        if 'report_files' in results and results['report_files']:
            print(f"\nReports Generated:")
            for report_file in results['report_files']:
                print(f"  • {report_file}")
        
        if summary['recommendations']:
            print(f"\n💡 Key Recommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "=" * 80)
    
    else:
        # CI mode - minimal output
        summary = results['summary']
        status = "PASS" if results['overall_success'] else "FAIL"
        print(f"{status}: {summary['total_passed']}/{summary['total_tests']} tests, "
              f"{summary['average_coverage']:.1f}% coverage, {results['total_execution_time']:.1f}s")
    
    # Return exit code
    return 0 if results['overall_success'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)