#!/usr/bin/env python3
"""
Comprehensive Circuit Breaker Test Runner

This script provides a unified interface to run all circuit breaker tests
with different configurations and reporting options. It includes test
categorization, performance benchmarking, and detailed reporting.

Usage:
    python run_circuit_breaker_tests.py [options]
    
    --all                   Run all test suites
    --unit                 Run unit tests only
    --integration          Run integration tests only
    --performance          Run performance tests only
    --e2e                  Run end-to-end tests only
    --monitoring           Run monitoring tests only
    --failure-scenarios    Run failure scenario tests only
    --quick                Run quick smoke tests
    --benchmark            Run performance benchmarks
    --coverage             Run with coverage reporting
    --parallel             Run tests in parallel
    --verbose              Verbose output
    --report-file FILE     Generate detailed report file
    --junit-xml FILE       Generate JUnit XML report
"""

import sys
import os
import argparse
import subprocess
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test suite definitions
TEST_SUITES = {
    "unit": [
        "test_production_circuit_breaker_comprehensive.py::TestProductionCircuitBreakerCore",
        "test_production_circuit_breaker_comprehensive.py::TestAdaptiveThresholds", 
        "test_production_circuit_breaker_comprehensive.py::TestProactiveCircuitOpening",
        "test_production_circuit_breaker_comprehensive.py::TestCircuitBreakerRecovery",
        "test_production_circuit_breaker_comprehensive.py::TestCircuitBreakerMetrics",
        "test_production_circuit_breaker_comprehensive.py::TestServiceSpecificBehavior"
    ],
    
    "integration": [
        "test_production_circuit_breaker_integration.py::TestCircuitBreakerLoadBalancerIntegration",
        "test_production_circuit_breaker_integration.py::TestCascadeFailurePrevention",
        "test_production_circuit_breaker_integration.py::TestCostBasedCircuitBreakerIntegration",
        "test_production_circuit_breaker_integration.py::TestMonitoringIntegration"
    ],
    
    "failure_scenarios": [
        "test_production_circuit_breaker_failure_scenarios.py::TestAPITimeoutScenarios",
        "test_production_circuit_breaker_failure_scenarios.py::TestRateLimitScenarios",
        "test_production_circuit_breaker_failure_scenarios.py::TestServiceUnavailableScenarios",
        "test_production_circuit_breaker_failure_scenarios.py::TestCascadingFailureScenarios",
        "test_production_circuit_breaker_failure_scenarios.py::TestBudgetExhaustionScenarios",
        "test_production_circuit_breaker_failure_scenarios.py::TestMemoryPressureScenarios",
        "test_production_circuit_breaker_failure_scenarios.py::TestNetworkConnectivityScenarios",
        "test_production_circuit_breaker_failure_scenarios.py::TestComplexFailureScenarios"
    ],
    
    "performance": [
        "test_production_circuit_breaker_performance.py::TestPerformanceBaselines",
        "test_production_circuit_breaker_performance.py::TestConcurrentPerformance",
        "test_production_circuit_breaker_performance.py::TestMemoryPerformance",
        "test_production_circuit_breaker_performance.py::TestLoadConditions",
        "test_production_circuit_breaker_performance.py::TestRecoveryPerformance",
        "test_production_circuit_breaker_performance.py::TestMonitoringPerformance"
    ],
    
    "e2e": [
        "test_production_circuit_breaker_e2e.py::TestCompleteQueryProcessingWorkflows",
        "test_production_circuit_breaker_e2e.py::TestFallbackSystemCoordination",
        "test_production_circuit_breaker_e2e.py::TestRecoveryWorkflows", 
        "test_production_circuit_breaker_e2e.py::TestMultiServiceFailureRecovery"
    ],
    
    "monitoring": [
        "test_production_circuit_breaker_monitoring.py::TestMetricsCollection",
        "test_production_circuit_breaker_monitoring.py::TestAlertSystem",
        "test_production_circuit_breaker_monitoring.py::TestDashboardIntegration",
        "test_production_circuit_breaker_monitoring.py::TestLoggingSystem",
        "test_production_circuit_breaker_monitoring.py::TestHealthCheckIntegration"
    ],
    
    "quick": [
        "test_production_circuit_breaker_comprehensive.py::TestProductionCircuitBreakerCore::test_initial_state_closed",
        "test_production_circuit_breaker_comprehensive.py::TestProductionCircuitBreakerCore::test_state_transition_closed_to_open",
        "test_production_circuit_breaker_integration.py::TestCircuitBreakerLoadBalancerIntegration::test_circuit_breaker_initialization",
        "test_production_circuit_breaker_performance.py::TestPerformanceBaselines::test_single_request_processing_time"
    ]
}


def run_pytest_command(test_paths: List[str], extra_args: List[str] = None) -> subprocess.CompletedProcess:
    """Run pytest with specified test paths and arguments"""
    cmd = ["python", "-m", "pytest"] + test_paths
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running command: {' '.join(cmd)}")
    
    return subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True
    )


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"


def parse_pytest_output(output: str) -> Dict[str, Any]:
    """Parse pytest output to extract test results"""
    lines = output.split('\n')
    
    results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
        "failures": [],
        "errors_list": []
    }
    
    # Parse summary line
    for line in lines:
        if "passed" in line and ("failed" in line or "error" in line or "skipped" in line):
            # Extract numbers from summary line
            parts = line.split()
            for i, part in enumerate(parts):
                if "passed" in part and i > 0:
                    results["passed"] = int(parts[i-1])
                elif "failed" in part and i > 0:
                    results["failed"] = int(parts[i-1])
                elif "skipped" in part and i > 0:
                    results["skipped"] = int(parts[i-1])
                elif "error" in part and i > 0:
                    results["errors"] = int(parts[i-1])
        
        # Extract duration
        if "in " in line and line.strip().endswith("s"):
            try:
                duration_str = line.split("in ")[-1].rstrip("s")
                results["duration"] = float(duration_str)
            except (ValueError, IndexError):
                pass
    
    return results


def generate_report(suite_results: Dict[str, Dict], args) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    total_passed = sum(r["passed"] for r in suite_results.values())
    total_failed = sum(r["failed"] for r in suite_results.values())
    total_skipped = sum(r["skipped"] for r in suite_results.values())
    total_errors = sum(r["errors"] for r in suite_results.values())
    total_duration = sum(r["duration"] for r in suite_results.values())
    total_tests = total_passed + total_failed + total_skipped + total_errors
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_run_config": {
            "command_line_args": vars(args),
            "test_suites_run": list(suite_results.keys()),
            "total_test_files": len([t for suite in suite_results.keys() 
                                   for t in TEST_SUITES.get(suite, [])])
        },
        "summary": {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "errors": total_errors,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration,
            "average_test_time": total_duration / total_tests if total_tests > 0 else 0
        },
        "suite_results": suite_results,
        "performance_analysis": {
            "fastest_suite": min(suite_results.keys(), 
                               key=lambda k: suite_results[k]["duration"]) if suite_results else None,
            "slowest_suite": max(suite_results.keys(), 
                               key=lambda k: suite_results[k]["duration"]) if suite_results else None,
            "most_tests": max(suite_results.keys(), 
                            key=lambda k: suite_results[k]["passed"] + suite_results[k]["failed"]) if suite_results else None
        }
    }
    
    return report


def print_summary(report: Dict[str, Any]):
    """Print test run summary"""
    summary = report["summary"]
    
    print("\n" + "="*80)
    print("CIRCUIT BREAKER TEST SUITE RESULTS")
    print("="*80)
    
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ({summary['success_rate']:.1f}%)")
    print(f"Failed: {summary['failed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Errors: {summary['errors']}")
    print(f"Duration: {format_duration(summary['total_duration'])}")
    print(f"Average per test: {format_duration(summary['average_test_time'])}")
    
    print(f"\nTest Suites Run: {', '.join(report['test_run_config']['test_suites_run'])}")
    
    if report["performance_analysis"]["fastest_suite"]:
        fastest = report["performance_analysis"]["fastest_suite"]
        slowest = report["performance_analysis"]["slowest_suite"]
        print(f"Fastest Suite: {fastest} ({format_duration(report['suite_results'][fastest]['duration'])})")
        print(f"Slowest Suite: {slowest} ({format_duration(report['suite_results'][slowest]['duration'])})")
    
    # Suite breakdown
    print(f"\nSuite Breakdown:")
    print(f"{'Suite':<20} {'Tests':<8} {'Passed':<8} {'Failed':<8} {'Duration':<12}")
    print("-" * 60)
    
    for suite, results in report["suite_results"].items():
        total_suite_tests = results["passed"] + results["failed"] + results["skipped"] + results["errors"]
        print(f"{suite:<20} {total_suite_tests:<8} {results['passed']:<8} {results['failed']:<8} {format_duration(results['duration']):<12}")
    
    print("="*80)
    
    if summary["failed"] > 0 or summary["errors"] > 0:
        print("❌ SOME TESTS FAILED - See detailed output above")
        return False
    else:
        print("✅ ALL TESTS PASSED")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive circuit breaker tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Test suite selection
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--monitoring", action="store_true", help="Run monitoring tests")
    parser.add_argument("--failure-scenarios", action="store_true", help="Run failure scenario tests")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    
    # Test execution options
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    
    # Reporting options
    parser.add_argument("--report-file", help="Generate detailed JSON report file")
    parser.add_argument("--junit-xml", help="Generate JUnit XML report")
    parser.add_argument("--html-report", help="Generate HTML report")
    
    # Filtering options
    parser.add_argument("--marker", help="Run tests with specific marker")
    parser.add_argument("--keyword", help="Run tests matching keyword")
    parser.add_argument("--failed-first", action="store_true", help="Run failed tests first")
    
    args = parser.parse_args()
    
    # Determine which test suites to run
    suites_to_run = []
    
    if args.all:
        suites_to_run = ["unit", "integration", "failure_scenarios", "performance", "e2e", "monitoring"]
    else:
        if args.unit:
            suites_to_run.append("unit")
        if args.integration:
            suites_to_run.append("integration")
        if args.failure_scenarios:
            suites_to_run.append("failure_scenarios")
        if args.performance:
            suites_to_run.append("performance")
        if args.e2e:
            suites_to_run.append("e2e")
        if args.monitoring:
            suites_to_run.append("monitoring")
        if args.quick:
            suites_to_run.append("quick")
        if args.benchmark:
            # Run performance tests with benchmarking
            suites_to_run.append("performance")
    
    # Default to quick tests if nothing specified
    if not suites_to_run:
        print("No test suite specified. Running quick smoke tests...")
        suites_to_run = ["quick"]
    
    # Build pytest arguments
    pytest_args = []
    
    if args.verbose:
        pytest_args.extend(["-v", "-s"])
    elif args.quiet:
        pytest_args.append("-q")
    
    if args.coverage:
        pytest_args.extend(["--cov=lightrag_integration", "--cov-report=html", "--cov-report=term-missing"])
    
    if args.parallel:
        pytest_args.extend(["-n", "auto"])
    
    if args.marker:
        pytest_args.extend(["-m", args.marker])
    
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])
    
    if args.failed_first:
        pytest_args.append("--failed-first")
    
    if args.junit_xml:
        pytest_args.extend(["--junit-xml", args.junit_xml])
    
    if args.html_report:
        pytest_args.extend(["--html", args.html_report, "--self-contained-html"])
    
    if args.benchmark:
        pytest_args.extend(["--benchmark-only", "--benchmark-sort=mean"])
    
    # Run test suites
    print(f"Starting circuit breaker test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running test suites: {', '.join(suites_to_run)}")
    
    suite_results = {}
    overall_start_time = time.time()
    
    for suite in suites_to_run:
        if suite not in TEST_SUITES:
            print(f"Warning: Unknown test suite '{suite}', skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running {suite.upper()} test suite...")
        print(f"{'='*60}")
        
        test_paths = [f"tests/{path}" for path in TEST_SUITES[suite]]
        
        start_time = time.time()
        result = run_pytest_command(test_paths, pytest_args)
        end_time = time.time()
        
        # Parse results
        parsed_results = parse_pytest_output(result.stdout + result.stderr)
        parsed_results["duration"] = end_time - start_time
        parsed_results["return_code"] = result.returncode
        
        suite_results[suite] = parsed_results
        
        # Print immediate results
        print(f"\n{suite.upper()} Results:")
        print(f"  Tests: {parsed_results['passed'] + parsed_results['failed'] + parsed_results['skipped'] + parsed_results['errors']}")
        print(f"  Passed: {parsed_results['passed']}")
        print(f"  Failed: {parsed_results['failed']}")
        print(f"  Duration: {format_duration(parsed_results['duration'])}")
        
        if result.returncode != 0:
            print(f"  ❌ Suite failed with return code {result.returncode}")
            if not args.quiet:
                print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
                print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        else:
            print(f"  ✅ Suite passed")
    
    overall_end_time = time.time()
    
    # Generate comprehensive report
    report = generate_report(suite_results, args)
    report["summary"]["overall_duration"] = overall_end_time - overall_start_time
    
    # Print summary
    success = print_summary(report)
    
    # Save detailed report if requested
    if args.report_file:
        with open(args.report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {args.report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()