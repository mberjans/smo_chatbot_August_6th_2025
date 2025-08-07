#!/usr/bin/env python3
"""
Test runner for all performance benchmarking unit tests.

This module provides a comprehensive test runner for all performance benchmarking
utilities, including test discovery, execution, reporting, and coverage analysis.

Usage:
    python run_all_tests.py [options]
    
Options:
    --verbose, -v: Verbose test output
    --coverage, -c: Generate coverage report
    --html-report: Generate HTML coverage report
    --integration: Include integration tests
    --performance: Include performance validation tests
    --parallel, -p: Run tests in parallel
    --output-dir: Directory for test reports (default: test_reports)

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import argparse
import sys
import os
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class PerformanceBenchmarkTestRunner:
    """Comprehensive test runner for performance benchmarking utilities."""
    
    def __init__(self, output_dir: Optional[Path] = None, verbose: bool = False):
        """
        Initialize the test runner.
        
        Args:
            output_dir: Directory for test reports and outputs
            verbose: Enable verbose output
        """
        self.output_dir = output_dir or Path("test_reports")
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test modules to run
        self.test_modules = [
            "test_quality_performance_benchmarks",
            "test_performance_correlation_engine", 
            "test_quality_aware_metrics_logger",
            "test_quality_performance_reporter"
        ]
        
        # Test results storage
        self.test_results = {}
        self.coverage_data = {}
        
    def discover_tests(self) -> Dict[str, List[str]]:
        """
        Discover all test cases in the test modules.
        
        Returns:
            Dictionary mapping module names to list of test case names
        """
        discovered_tests = {}
        
        for module in self.test_modules:
            if self.verbose:
                print(f"Discovering tests in {module}...")
            
            try:
                # Use pytest to discover tests
                result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    f"{module}.py",
                    "--collect-only", "-q"
                ], capture_output=True, text=True, cwd=Path.cwd())
                
                if result.returncode == 0:
                    # Parse pytest output to extract test names
                    test_names = []
                    for line in result.stdout.split('\n'):
                        if '::' in line and 'test_' in line:
                            test_name = line.split('::')[-1].strip()
                            if test_name.startswith('test_'):
                                test_names.append(test_name)
                    
                    discovered_tests[module] = test_names
                    if self.verbose:
                        print(f"  Found {len(test_names)} tests")
                else:
                    print(f"Warning: Failed to discover tests in {module}")
                    discovered_tests[module] = []
                    
            except Exception as e:
                print(f"Error discovering tests in {module}: {e}")
                discovered_tests[module] = []
        
        return discovered_tests
    
    def run_unit_tests(self, 
                      coverage: bool = False,
                      parallel: bool = False,
                      html_report: bool = False) -> Dict[str, Any]:
        """
        Run unit tests for all modules.
        
        Args:
            coverage: Generate coverage report
            parallel: Run tests in parallel
            html_report: Generate HTML coverage report
            
        Returns:
            Dictionary containing test results and statistics
        """
        print("Running unit tests for performance benchmarking utilities...")
        start_time = time.time()
        
        # Build pytest command
        pytest_args = [
            sys.executable, "-m", "pytest",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ]
        
        # Add coverage options
        if coverage:
            pytest_args.extend([
                "--cov=quality_performance_benchmarks",
                "--cov=performance_correlation_engine",
                "--cov=quality_aware_metrics_logger", 
                "--cov=reporting.quality_performance_reporter",
                "--cov-report=term-missing",
                f"--cov-report=json:{self.output_dir}/coverage.json"
            ])
            
            if html_report:
                pytest_args.append(f"--cov-report=html:{self.output_dir}/coverage_html")
        
        # Add parallel execution
        if parallel:
            pytest_args.extend(["-n", "auto"])  # Requires pytest-xdist
        
        # Add test modules
        pytest_args.extend([f"{module}.py" for module in self.test_modules])
        
        # Add output options
        pytest_args.extend([
            f"--junit-xml={self.output_dir}/test_results.xml",
            f"--json-report",
            f"--json-report-file={self.output_dir}/test_results.json"
        ])
        
        # Run tests
        print(f"Executing: {' '.join(pytest_args)}")
        result = subprocess.run(pytest_args, cwd=Path.cwd())
        
        execution_time = time.time() - start_time
        
        # Process results
        test_results = {
            'execution_time_seconds': execution_time,
            'exit_code': result.returncode,
            'success': result.returncode == 0,
            'modules_tested': self.test_modules,
            'timestamp': time.time()
        }
        
        # Load detailed results if available
        json_results_file = self.output_dir / "test_results.json"
        if json_results_file.exists():
            try:
                with open(json_results_file, 'r') as f:
                    detailed_results = json.load(f)
                test_results['detailed_results'] = detailed_results
                test_results['total_tests'] = detailed_results.get('summary', {}).get('total', 0)
                test_results['passed_tests'] = detailed_results.get('summary', {}).get('passed', 0)
                test_results['failed_tests'] = detailed_results.get('summary', {}).get('failed', 0)
            except Exception as e:
                print(f"Warning: Could not load detailed test results: {e}")
        
        # Load coverage results if available
        if coverage:
            coverage_file = self.output_dir / "coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    test_results['coverage_data'] = coverage_data
                    test_results['total_coverage_percent'] = coverage_data.get('totals', {}).get('percent_covered', 0)
                except Exception as e:
                    print(f"Warning: Could not load coverage data: {e}")
        
        self.test_results['unit_tests'] = test_results
        return test_results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """
        Run integration tests that test component interactions.
        
        Returns:
            Dictionary containing integration test results
        """
        print("Running integration tests...")
        start_time = time.time()
        
        # Integration test patterns - tests that end with '_integration' or are in integration test classes
        integration_patterns = [
            "*integration*",
            "*Integration*",
            "*end_to_end*",
            "*e2e*"
        ]
        
        pytest_args = [
            sys.executable, "-m", "pytest",
            "-v" if self.verbose else "-q",
            "--tb=short",
            "-k", " or ".join(integration_patterns),
            f"--junit-xml={self.output_dir}/integration_test_results.xml"
        ]
        
        # Add test modules
        pytest_args.extend([f"{module}.py" for module in self.test_modules])
        
        result = subprocess.run(pytest_args, cwd=Path.cwd())
        
        execution_time = time.time() - start_time
        
        integration_results = {
            'execution_time_seconds': execution_time,
            'exit_code': result.returncode,
            'success': result.returncode == 0,
            'timestamp': time.time()
        }
        
        self.test_results['integration_tests'] = integration_results
        return integration_results
    
    def run_performance_validation_tests(self) -> Dict[str, Any]:
        """
        Run performance validation tests to ensure test performance.
        
        Returns:
            Dictionary containing performance validation results
        """
        print("Running performance validation tests...")
        start_time = time.time()
        
        # Performance test patterns
        performance_patterns = [
            "*performance*",
            "*benchmark*",
            "*speed*",
            "*memory*",
            "*concurrent*"
        ]
        
        pytest_args = [
            sys.executable, "-m", "pytest",
            "-v" if self.verbose else "-q",
            "--tb=short",
            "-k", " or ".join(performance_patterns),
            f"--junit-xml={self.output_dir}/performance_test_results.xml"
        ]
        
        # Add test modules
        pytest_args.extend([f"{module}.py" for module in self.test_modules])
        
        result = subprocess.run(pytest_args, cwd=Path.cwd())
        
        execution_time = time.time() - start_time
        
        performance_results = {
            'execution_time_seconds': execution_time,
            'exit_code': result.returncode,
            'success': result.returncode == 0,
            'timestamp': time.time()
        }
        
        self.test_results['performance_tests'] = performance_results
        return performance_results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive test report.
        
        Returns:
            Complete test report dictionary
        """
        print("Generating comprehensive test report...")
        
        # Discover tests
        discovered_tests = self.discover_tests()
        
        report = {
            'test_execution_summary': {
                'timestamp': time.time(),
                'total_modules': len(self.test_modules),
                'discovered_tests': discovered_tests,
                'total_discovered_tests': sum(len(tests) for tests in discovered_tests.values())
            },
            'test_results': self.test_results,
            'environment_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': str(Path.cwd()),
                'output_directory': str(self.output_dir)
            }
        }
        
        # Calculate overall statistics
        overall_stats = {
            'total_execution_time': 0,
            'all_tests_passed': True,
            'total_test_suites': 0,
            'successful_test_suites': 0
        }
        
        for test_type, results in self.test_results.items():
            overall_stats['total_execution_time'] += results.get('execution_time_seconds', 0)
            overall_stats['total_test_suites'] += 1
            
            if results.get('success', False):
                overall_stats['successful_test_suites'] += 1
            else:
                overall_stats['all_tests_passed'] = False
        
        report['overall_statistics'] = overall_stats
        
        # Save report
        report_file = self.output_dir / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Test report saved to: {report_file}")
        return report
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """
        Print test execution summary.
        
        Args:
            report: Test report dictionary
        """
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARKING UNIT TEST SUMMARY")
        print("="*80)
        
        # Overall statistics
        stats = report['overall_statistics']
        print(f"Total Execution Time: {stats['total_execution_time']:.2f} seconds")
        print(f"Test Suites: {stats['successful_test_suites']}/{stats['total_test_suites']} passed")
        print(f"Overall Result: {'PASS' if stats['all_tests_passed'] else 'FAIL'}")
        
        # Module-specific results
        print(f"\nModule Test Results:")
        print("-" * 40)
        
        for test_type, results in report['test_results'].items():
            status = "PASS" if results.get('success', False) else "FAIL"
            time_taken = results.get('execution_time_seconds', 0)
            print(f"{test_type:<25}: {status:<6} ({time_taken:.2f}s)")
            
            if 'total_tests' in results:
                passed = results.get('passed_tests', 0)
                total = results.get('total_tests', 0)
                print(f"{'':25}  Tests: {passed}/{total}")
            
            if 'total_coverage_percent' in results:
                coverage = results['total_coverage_percent']
                print(f"{'':25}  Coverage: {coverage:.1f}%")
        
        # Test discovery summary
        discovery = report['test_execution_summary']
        print(f"\nTest Discovery:")
        print(f"Total Tests Discovered: {discovery['total_discovered_tests']}")
        print(f"Modules Analyzed: {discovery['total_modules']}")
        
        for module, tests in discovery['discovered_tests'].items():
            print(f"  {module}: {len(tests)} tests")
        
        print("\n" + "="*80)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive unit tests for performance benchmarking utilities"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose test output")
    parser.add_argument("--coverage", "-c", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("--html-report", action="store_true",
                       help="Generate HTML coverage report")
    parser.add_argument("--integration", action="store_true",
                       help="Include integration tests")
    parser.add_argument("--performance", action="store_true",
                       help="Include performance validation tests")
    parser.add_argument("--parallel", "-p", action="store_true",
                       help="Run tests in parallel")
    parser.add_argument("--output-dir", type=Path, default="test_reports",
                       help="Directory for test reports")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Run all test types (unit, integration, performance)")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = PerformanceBenchmarkTestRunner(
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    print("Performance Benchmarking Unit Test Runner")
    print("=" * 50)
    print(f"Output Directory: {runner.output_dir}")
    print(f"Verbose Mode: {args.verbose}")
    print()
    
    try:
        # Run unit tests (always run these)
        unit_results = runner.run_unit_tests(
            coverage=args.coverage,
            parallel=args.parallel,
            html_report=args.html_report
        )
        
        # Run integration tests if requested
        if args.integration or args.all:
            integration_results = runner.run_integration_tests()
        
        # Run performance validation tests if requested
        if args.performance or args.all:
            performance_results = runner.run_performance_validation_tests()
        
        # Generate comprehensive report
        report = runner.generate_test_report()
        
        # Print summary
        runner.print_summary(report)
        
        # Exit with appropriate code
        overall_success = report['overall_statistics']['all_tests_passed']
        return 0 if overall_success else 1
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError during test execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())