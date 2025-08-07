#!/usr/bin/env python3
"""
Comprehensive Error Handling Test Runner for Clinical Metabolomics Oracle.

This script runs the complete error handling test suite with detailed reporting,
performance metrics, and coverage analysis. It provides structured output for
CI/CD integration and comprehensive validation of all error handling scenarios.

Features:
- Automated test discovery and execution
- Performance benchmarking and metrics
- Coverage reporting with detailed breakdowns
- Error scenario validation and edge case testing
- Integration test coordination
- HTML and JSON report generation

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/error_handling_test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)


class TestSuiteRunner:
    """Comprehensive test suite runner for error handling tests."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize test runner."""
        self.base_dir = base_dir or Path(__file__).parent
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Test modules to run
        self.test_modules = [
            "test_comprehensive_error_handling.py",
            "test_storage_error_handling_comprehensive.py", 
            "test_advanced_recovery_edge_cases.py"
        ]
        
        # Test categories
        self.test_categories = {
            "unit": "Unit tests for individual error handling components",
            "integration": "Integration tests for error handling workflows",
            "performance": "Performance and stress tests for error handling",
            "edge_cases": "Edge cases and boundary condition tests"
        }
        
        # Performance benchmarks
        self.performance_benchmarks = {
            "error_classification_time": 1.0,  # Max seconds
            "recovery_strategy_time": 2.0,     # Max seconds
            "checkpoint_creation_time": 5.0,   # Max seconds
            "backoff_calculation_time": 0.1    # Max seconds
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all error handling tests."""
        logger.info("Starting comprehensive error handling test suite")
        self.start_time = time.time()
        
        overall_results = {
            "start_time": datetime.now().isoformat(),
            "test_modules": {},
            "summary": {},
            "performance_metrics": {},
            "coverage": {},
            "errors": []
        }
        
        try:
            # Run each test module
            for module in self.test_modules:
                logger.info(f"Running test module: {module}")
                module_results = self._run_test_module(module)
                overall_results["test_modules"][module] = module_results
            
            # Generate summary
            overall_results["summary"] = self._generate_summary(overall_results["test_modules"])
            
            # Run performance benchmarks
            overall_results["performance_metrics"] = self._run_performance_benchmarks()
            
            # Generate coverage report
            overall_results["coverage"] = self._generate_coverage_report()
            
        except Exception as e:
            logger.error(f"Error running test suite: {e}")
            overall_results["errors"].append(str(e))
        
        finally:
            self.end_time = time.time()
            overall_results["end_time"] = datetime.now().isoformat()
            overall_results["total_duration"] = self.end_time - self.start_time
        
        return overall_results
    
    def _run_test_module(self, module: str) -> Dict[str, Any]:
        """Run a specific test module."""
        module_path = self.base_dir / module
        
        if not module_path.exists():
            return {
                "status": "skipped",
                "reason": f"Module {module} not found",
                "tests": {},
                "duration": 0
            }
        
        start_time = time.time()
        
        try:
            # Run pytest with detailed output
            cmd = [
                sys.executable, "-m", "pytest",
                str(module_path),
                "-v",
                "--tb=short",
                "--json-report",
                f"--json-report-file={self.base_dir}/logs/{module}_report.json",
                "--durations=10",
                "--strict-markers"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.base_dir
            )
            
            duration = time.time() - start_time
            
            # Parse results
            module_results = {
                "status": "passed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "tests": self._parse_pytest_output(result.stdout)
            }
            
            # Try to load JSON report if available
            json_report_path = self.base_dir / "logs" / f"{module}_report.json"
            if json_report_path.exists():
                try:
                    with open(json_report_path) as f:
                        json_report = json.load(f)
                    module_results["detailed_report"] = json_report
                except Exception as e:
                    logger.warning(f"Could not load JSON report for {module}: {e}")
            
            return module_results
            
        except Exception as e:
            logger.error(f"Error running module {module}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time,
                "tests": {}
            }
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract test information."""
        tests = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "test_cases": []
        }
        
        lines = output.split('\n')
        
        for line in lines:
            # Look for test result lines
            if '::test_' in line:
                if 'PASSED' in line:
                    tests["passed"] += 1
                elif 'FAILED' in line:
                    tests["failed"] += 1
                elif 'SKIPPED' in line:
                    tests["skipped"] += 1
                elif 'ERROR' in line:
                    tests["errors"] += 1
                
                tests["total"] += 1
                tests["test_cases"].append(line.strip())
            
            # Look for summary line
            elif 'failed' in line and 'passed' in line:
                # Parse summary line like "2 failed, 8 passed in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "failed,":
                        tests["failed"] = int(parts[i-1])
                    elif part == "passed":
                        tests["passed"] = int(parts[i-1])
        
        return tests
    
    def _generate_summary(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall test summary."""
        summary = {
            "modules_run": len(module_results),
            "modules_passed": 0,
            "modules_failed": 0,
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "total_errors": 0,
            "overall_pass_rate": 0.0,
            "total_duration": 0.0
        }
        
        for module, results in module_results.items():
            if results.get("status") == "passed":
                summary["modules_passed"] += 1
            else:
                summary["modules_failed"] += 1
            
            summary["total_duration"] += results.get("duration", 0)
            
            tests = results.get("tests", {})
            summary["total_tests"] += tests.get("total", 0)
            summary["total_passed"] += tests.get("passed", 0)
            summary["total_failed"] += tests.get("failed", 0)
            summary["total_skipped"] += tests.get("skipped", 0)
            summary["total_errors"] += tests.get("errors", 0)
        
        # Calculate pass rate
        total_run = summary["total_tests"] - summary["total_skipped"]
        if total_run > 0:
            summary["overall_pass_rate"] = summary["total_passed"] / total_run * 100
        
        return summary
    
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks for error handling components."""
        logger.info("Running performance benchmarks")
        
        benchmarks = {}
        
        try:
            # Import test components
            from lightrag_integration.clinical_metabolomics_rag import (
                IngestionAPIError, IngestionNetworkError
            )
            from lightrag_integration.advanced_recovery_system import (
                AdaptiveBackoffCalculator, FailureType, BackoffStrategy
            )
            
            # Benchmark 1: Error classification speed
            start_time = time.time()
            for _ in range(1000):
                error = IngestionAPIError("Test error", status_code=500)
                error_type = type(error).__name__
            classification_time = time.time() - start_time
            
            benchmarks["error_classification"] = {
                "duration": classification_time,
                "operations_per_second": 1000 / classification_time,
                "passes_benchmark": classification_time < self.performance_benchmarks["error_classification_time"]
            }
            
            # Benchmark 2: Backoff calculation speed
            calculator = AdaptiveBackoffCalculator()
            start_time = time.time()
            for i in range(100):
                delay = calculator.calculate_backoff(
                    FailureType.API_ERROR,
                    i % 10 + 1,
                    BackoffStrategy.ADAPTIVE
                )
            backoff_time = time.time() - start_time
            
            benchmarks["backoff_calculation"] = {
                "duration": backoff_time,
                "operations_per_second": 100 / backoff_time,
                "passes_benchmark": backoff_time < self.performance_benchmarks["backoff_calculation_time"]
            }
            
        except Exception as e:
            logger.error(f"Error running performance benchmarks: {e}")
            benchmarks["error"] = str(e)
        
        return benchmarks
    
    def _generate_coverage_report(self) -> Dict[str, Any]:
        """Generate coverage report for error handling code."""
        logger.info("Generating coverage report")
        
        coverage = {
            "error_classes_covered": [],
            "recovery_strategies_tested": [],
            "logging_scenarios_covered": [],
            "edge_cases_tested": []
        }
        
        try:
            # Analyze test files to determine coverage
            test_files = [self.base_dir / module for module in self.test_modules]
            
            for test_file in test_files:
                if test_file.exists():
                    content = test_file.read_text()
                    
                    # Look for error class testing
                    error_classes = [
                        "IngestionError", "IngestionRetryableError", "IngestionNonRetryableError",
                        "IngestionResourceError", "IngestionNetworkError", "IngestionAPIError",
                        "StorageInitializationError", "StoragePermissionError", "StorageSpaceError",
                        "StorageDirectoryError", "StorageRetryableError"
                    ]
                    
                    for error_class in error_classes:
                        if error_class in content:
                            coverage["error_classes_covered"].append(error_class)
                    
                    # Look for recovery strategies
                    strategies = ["backoff_and_retry", "reduce_resources", "degrade_to_safe_mode"]
                    for strategy in strategies:
                        if strategy in content:
                            coverage["recovery_strategies_tested"].append(strategy)
                    
                    # Look for logging scenarios
                    logging_scenarios = ["log_error_with_context", "log_performance_metrics", "structured_logging"]
                    for scenario in logging_scenarios:
                        if scenario in content:
                            coverage["logging_scenarios_covered"].append(scenario)
                    
                    # Look for edge cases
                    edge_cases = ["concurrent", "stress", "extreme", "boundary", "corruption"]
                    for edge_case in edge_cases:
                        if edge_case in content.lower():
                            coverage["edge_cases_tested"].append(edge_case)
        
        except Exception as e:
            logger.error(f"Error generating coverage report: {e}")
            coverage["error"] = str(e)
        
        # Remove duplicates
        for key in coverage:
            if isinstance(coverage[key], list):
                coverage[key] = list(set(coverage[key]))
        
        return coverage
    
    def save_results(self, results: Dict[str, Any], output_path: Optional[Path] = None) -> None:
        """Save test results to file."""
        if output_path is None:
            output_path = self.base_dir / "logs" / f"comprehensive_error_handling_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {output_path}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print test summary to console."""
        print("\n" + "="*80)
        print("COMPREHENSIVE ERROR HANDLING TEST RESULTS")
        print("="*80)
        
        summary = results.get("summary", {})
        
        print(f"\nOverall Summary:")
        print(f"  Total Modules: {summary.get('modules_run', 0)}")
        print(f"  Modules Passed: {summary.get('modules_passed', 0)}")
        print(f"  Modules Failed: {summary.get('modules_failed', 0)}")
        print(f"  Total Duration: {summary.get('total_duration', 0):.2f} seconds")
        
        print(f"\nTest Results:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('total_passed', 0)}")
        print(f"  Failed: {summary.get('total_failed', 0)}")
        print(f"  Skipped: {summary.get('total_skipped', 0)}")
        print(f"  Errors: {summary.get('total_errors', 0)}")
        print(f"  Pass Rate: {summary.get('overall_pass_rate', 0):.1f}%")
        
        # Performance benchmarks
        perf = results.get("performance_metrics", {})
        if perf:
            print(f"\nPerformance Benchmarks:")
            for benchmark, data in perf.items():
                if isinstance(data, dict) and "passes_benchmark" in data:
                    status = "PASS" if data["passes_benchmark"] else "FAIL"
                    print(f"  {benchmark}: {status} ({data.get('duration', 0):.4f}s)")
        
        # Coverage
        coverage = results.get("coverage", {})
        if coverage:
            print(f"\nCoverage Summary:")
            print(f"  Error Classes Covered: {len(coverage.get('error_classes_covered', []))}")
            print(f"  Recovery Strategies Tested: {len(coverage.get('recovery_strategies_tested', []))}")
            print(f"  Logging Scenarios Covered: {len(coverage.get('logging_scenarios_covered', []))}")
            print(f"  Edge Cases Tested: {len(coverage.get('edge_cases_tested', []))}")
        
        # Module details
        print(f"\nModule Details:")
        for module, module_results in results.get("test_modules", {}).items():
            status = module_results.get("status", "unknown")
            duration = module_results.get("duration", 0)
            tests = module_results.get("tests", {})
            
            print(f"  {module}:")
            print(f"    Status: {status.upper()}")
            print(f"    Duration: {duration:.2f}s")
            print(f"    Tests: {tests.get('passed', 0)}/{tests.get('total', 0)} passed")
        
        # Errors
        if results.get("errors"):
            print(f"\nErrors:")
            for error in results["errors"]:
                print(f"  - {error}")
        
        print("\n" + "="*80)


def main():
    """Main entry point for test runner."""
    # Setup
    base_dir = Path(__file__).parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Run tests
    runner = TestSuiteRunner(base_dir)
    results = runner.run_all_tests()
    
    # Save and display results
    runner.save_results(results)
    runner.print_summary(results)
    
    # Exit with appropriate code
    summary = results.get("summary", {})
    if summary.get("modules_failed", 0) > 0 or summary.get("total_failed", 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()