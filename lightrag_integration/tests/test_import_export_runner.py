"""
Comprehensive Test Runner for Import/Export Functionality Tests

This module provides a comprehensive test runner for all import/export tests,
with detailed reporting, performance metrics, and integration with the existing
CMO testing infrastructure.

Features:
    - Automated test discovery and execution
    - Performance benchmarking and reporting
    - Error analysis and categorization
    - Integration with existing test infrastructure
    - Detailed logging and metrics collection

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import pytest


class ImportExportTestRunner:
    """
    Comprehensive test runner for import/export functionality.
    
    Orchestrates all import/export tests and provides detailed reporting
    and integration with the CMO testing infrastructure.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the test runner.
        
        Args:
            output_dir: Directory for test outputs and reports
        """
        self.output_dir = output_dir or Path(__file__).parent / "import_export_test_results"
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_modules = [
            'test_module_imports',
            'test_module_exports',
            'test_version_info',
            'test_import_export_error_handling',
            'test_import_export_performance'
        ]
        
        self.results = {}
        self.performance_metrics = {}
        self.start_time = None
        self.end_time = None
        
        # Set up logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the test runner."""
        logger = logging.getLogger('import_export_test_runner')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_dir / f"test_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all import/export tests.
        
        Returns:
            Dictionary with comprehensive test results
        """
        self.logger.info("Starting comprehensive import/export test suite")
        self.start_time = time.time()
        
        try:
            # Run each test module
            for test_module in self.test_modules:
                self.logger.info(f"Running tests from {test_module}")
                result = self._run_test_module(test_module)
                self.results[test_module] = result
            
            # Generate comprehensive report
            self.end_time = time.time()
            report = self._generate_comprehensive_report()
            
            # Save results
            self._save_results(report)
            
            self.logger.info("Import/export test suite completed successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"Test suite failed with error: {e}")
            self.end_time = time.time()
            error_report = self._generate_error_report(str(e))
            self._save_results(error_report)
            raise

    def _run_test_module(self, module_name: str) -> Dict[str, Any]:
        """Run tests from a specific module."""
        test_file = Path(__file__).parent / f"{module_name}.py"
        
        if not test_file.exists():
            self.logger.warning(f"Test file not found: {test_file}")
            return {
                'status': 'skipped',
                'reason': 'file_not_found',
                'tests_run': 0,
                'failures': 0,
                'errors': 0
            }
        
        # Run pytest on the specific module
        pytest_args = [
            str(test_file),
            '-v',
            '--tb=short',
            '--json-report',
            f'--json-report-file={self.output_dir / f"{module_name}_results.json"}',
            '--disable-warnings'
        ]
        
        start_time = time.time()
        exit_code = pytest.main(pytest_args)
        end_time = time.time()
        
        # Parse results
        result_file = self.output_dir / f"{module_name}_results.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    pytest_results = json.load(f)
                
                return {
                    'status': 'completed',
                    'exit_code': exit_code,
                    'duration': end_time - start_time,
                    'tests_run': pytest_results.get('summary', {}).get('total', 0),
                    'passed': pytest_results.get('summary', {}).get('passed', 0),
                    'failed': pytest_results.get('summary', {}).get('failed', 0),
                    'errors': pytest_results.get('summary', {}).get('error', 0),
                    'skipped': pytest_results.get('summary', {}).get('skipped', 0),
                    'detailed_results': pytest_results
                }
                
            except Exception as e:
                self.logger.error(f"Failed to parse results for {module_name}: {e}")
                
        return {
            'status': 'completed',
            'exit_code': exit_code,
            'duration': end_time - start_time,
            'tests_run': 0,
            'passed': 0,
            'failed': 0,
            'errors': 1 if exit_code != 0 else 0,
            'skipped': 0
        }

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Aggregate statistics
        total_tests = sum(result.get('tests_run', 0) for result in self.results.values())
        total_passed = sum(result.get('passed', 0) for result in self.results.values())
        total_failed = sum(result.get('failed', 0) for result in self.results.values())
        total_errors = sum(result.get('errors', 0) for result in self.results.values())
        total_skipped = sum(result.get('skipped', 0) for result in self.results.values())
        
        # Calculate success rate
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Performance analysis
        performance_analysis = self._analyze_performance_results()
        
        # Error analysis
        error_analysis = self._analyze_error_patterns()
        
        report = {
            'test_suite': 'Import/Export Functionality Tests',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': total_duration,
            'summary': {
                'total_test_modules': len(self.test_modules),
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'errors': total_errors,
                'skipped': total_skipped,
                'success_rate': success_rate
            },
            'module_results': self.results,
            'performance_analysis': performance_analysis,
            'error_analysis': error_analysis,
            'recommendations': self._generate_recommendations(),
            'environment_info': self._collect_environment_info()
        }
        
        return report

    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate error report when test suite fails."""
        return {
            'test_suite': 'Import/Export Functionality Tests',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': error_message,
            'partial_results': self.results,
            'environment_info': self._collect_environment_info()
        }

    def _analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze performance test results."""
        performance_data = {}
        
        # Extract performance data from test results
        for module_name, results in self.results.items():
            if 'performance' in module_name:
                detailed_results = results.get('detailed_results', {})
                tests = detailed_results.get('tests', [])
                
                performance_tests = []
                for test in tests:
                    if test.get('outcome') == 'passed' and 'performance' in test.get('nodeid', ''):
                        performance_tests.append({
                            'test_name': test.get('nodeid', ''),
                            'duration': test.get('duration', 0),
                            'outcome': test.get('outcome')
                        })
                
                performance_data[module_name] = performance_tests
        
        # Analyze performance metrics
        analysis = {
            'slowest_tests': [],
            'performance_concerns': [],
            'efficiency_metrics': {}
        }
        
        all_performance_tests = []
        for module_tests in performance_data.values():
            all_performance_tests.extend(module_tests)
        
        if all_performance_tests:
            # Find slowest tests
            sorted_tests = sorted(all_performance_tests, 
                                key=lambda x: x['duration'], reverse=True)
            analysis['slowest_tests'] = sorted_tests[:5]
            
            # Identify performance concerns
            slow_threshold = 5.0  # seconds
            slow_tests = [t for t in all_performance_tests if t['duration'] > slow_threshold]
            analysis['performance_concerns'] = slow_tests
            
            # Calculate efficiency metrics
            durations = [t['duration'] for t in all_performance_tests]
            analysis['efficiency_metrics'] = {
                'avg_test_duration': sum(durations) / len(durations),
                'max_test_duration': max(durations),
                'min_test_duration': min(durations),
                'total_performance_test_time': sum(durations)
            }
        
        return analysis

    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns across test results."""
        error_patterns = {
            'import_errors': [],
            'dependency_issues': [],
            'timeout_errors': [],
            'permission_errors': [],
            'other_errors': []
        }
        
        for module_name, results in self.results.items():
            if results.get('failed', 0) > 0 or results.get('errors', 0) > 0:
                detailed_results = results.get('detailed_results', {})
                tests = detailed_results.get('tests', [])
                
                for test in tests:
                    if test.get('outcome') in ['failed', 'error']:
                        call = test.get('call', {})
                        longrepr = call.get('longrepr', '')
                        
                        if isinstance(longrepr, str):
                            longrepr_lower = longrepr.lower()
                            
                            if 'importerror' in longrepr_lower or 'modulenotfounderror' in longrepr_lower:
                                error_patterns['import_errors'].append({
                                    'test': test.get('nodeid'),
                                    'error': longrepr[:200] + '...' if len(longrepr) > 200 else longrepr
                                })
                            elif 'no module named' in longrepr_lower or 'dependency' in longrepr_lower:
                                error_patterns['dependency_issues'].append({
                                    'test': test.get('nodeid'),
                                    'error': longrepr[:200] + '...' if len(longrepr) > 200 else longrepr
                                })
                            elif 'timeout' in longrepr_lower:
                                error_patterns['timeout_errors'].append({
                                    'test': test.get('nodeid'),
                                    'error': longrepr[:200] + '...' if len(longrepr) > 200 else longrepr
                                })
                            elif 'permission' in longrepr_lower:
                                error_patterns['permission_errors'].append({
                                    'test': test.get('nodeid'),
                                    'error': longrepr[:200] + '...' if len(longrepr) > 200 else longrepr
                                })
                            else:
                                error_patterns['other_errors'].append({
                                    'test': test.get('nodeid'),
                                    'error': longrepr[:200] + '...' if len(longrepr) > 200 else longrepr
                                })
        
        return error_patterns

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check overall success rate
        total_tests = sum(result.get('tests_run', 0) for result in self.results.values())
        total_passed = sum(result.get('passed', 0) for result in self.results.values())
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate < 90:
            recommendations.append(
                f"Overall success rate ({success_rate:.1f}%) is below 90%. "
                "Review failed tests and address underlying issues."
            )
        
        # Check for performance issues
        performance_analysis = self._analyze_performance_results()
        if performance_analysis.get('performance_concerns'):
            recommendations.append(
                f"Found {len(performance_analysis['performance_concerns'])} slow tests. "
                "Consider optimizing import performance or adjusting test thresholds."
            )
        
        # Check for dependency issues
        error_analysis = self._analyze_error_patterns()
        if error_analysis.get('dependency_issues'):
            recommendations.append(
                "Dependency issues detected. Ensure all required dependencies "
                "are installed and properly configured."
            )
        
        if error_analysis.get('import_errors'):
            recommendations.append(
                "Import errors detected. Review module structure and "
                "import statements for correctness."
            )
        
        # Module-specific recommendations
        for module_name, results in self.results.items():
            if results.get('failed', 0) > 0:
                recommendations.append(
                    f"Module {module_name} has failing tests. "
                    "Review detailed results for specific issues."
                )
        
        if not recommendations:
            recommendations.append(
                "All tests passed successfully! The import/export functionality "
                "appears to be working correctly."
            )
        
        return recommendations

    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information for the report."""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': str(Path.cwd()),
            'test_runner_version': '1.0.0',
            'pytest_version': pytest.__version__,
            'environment_variables': {
                key: value for key, value in os.environ.items()
                if key.startswith('LIGHTRAG_') or key in ['PYTHONPATH', 'PATH']
            }
        }

    def _save_results(self, report: Dict[str, Any]) -> None:
        """Save test results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = self.output_dir / f"import_export_test_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / f"import_export_test_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            self._write_summary_report(f, report)
        
        self.logger.info(f"Test results saved to {json_file}")
        self.logger.info(f"Test summary saved to {summary_file}")

    def _write_summary_report(self, file, report: Dict[str, Any]) -> None:
        """Write human-readable summary report."""
        file.write("LIGHTRAG INTEGRATION - IMPORT/EXPORT TEST SUITE RESULTS\n")
        file.write("=" * 60 + "\n\n")
        
        # Summary
        summary = report.get('summary', {})
        file.write(f"Test Suite: {report.get('test_suite', 'Unknown')}\n")
        file.write(f"Timestamp: {report.get('timestamp', 'Unknown')}\n")
        file.write(f"Duration: {report.get('duration_seconds', 0):.2f} seconds\n\n")
        
        file.write("SUMMARY:\n")
        file.write("-" * 20 + "\n")
        file.write(f"Total Test Modules: {summary.get('total_test_modules', 0)}\n")
        file.write(f"Total Tests: {summary.get('total_tests', 0)}\n")
        file.write(f"Passed: {summary.get('passed', 0)}\n")
        file.write(f"Failed: {summary.get('failed', 0)}\n")
        file.write(f"Errors: {summary.get('errors', 0)}\n")
        file.write(f"Skipped: {summary.get('skipped', 0)}\n")
        file.write(f"Success Rate: {summary.get('success_rate', 0):.1f}%\n\n")
        
        # Module Results
        file.write("MODULE RESULTS:\n")
        file.write("-" * 20 + "\n")
        for module_name, results in report.get('module_results', {}).items():
            file.write(f"{module_name}:\n")
            file.write(f"  Status: {results.get('status', 'unknown')}\n")
            file.write(f"  Tests Run: {results.get('tests_run', 0)}\n")
            file.write(f"  Passed: {results.get('passed', 0)}\n")
            file.write(f"  Failed: {results.get('failed', 0)}\n")
            file.write(f"  Duration: {results.get('duration', 0):.2f}s\n\n")
        
        # Performance Analysis
        perf_analysis = report.get('performance_analysis', {})
        if perf_analysis.get('slowest_tests'):
            file.write("SLOWEST TESTS:\n")
            file.write("-" * 20 + "\n")
            for test in perf_analysis['slowest_tests']:
                file.write(f"  {test.get('test_name', '')}: {test.get('duration', 0):.3f}s\n")
            file.write("\n")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            file.write("RECOMMENDATIONS:\n")
            file.write("-" * 20 + "\n")
            for i, rec in enumerate(recommendations, 1):
                file.write(f"{i}. {rec}\n\n")


def main():
    """Main function to run the import/export test suite."""
    print("LightRAG Integration - Import/Export Test Suite")
    print("=" * 50)
    
    # Create test runner
    runner = ImportExportTestRunner()
    
    try:
        # Run all tests
        report = runner.run_all_tests()
        
        # Print summary
        summary = report.get('summary', {})
        print(f"\nTest Results Summary:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"  Duration: {report.get('duration_seconds', 0):.2f} seconds")
        
        # Print recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        return 0 if summary.get('failed', 0) == 0 else 1
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())