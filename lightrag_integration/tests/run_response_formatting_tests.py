#!/usr/bin/env python3
"""
Test runner for comprehensive response formatting tests.

This script runs the complete test suite for response formatting functionality
in the Clinical Metabolomics RAG system, providing detailed reporting and
coverage analysis.

Usage:
    python run_response_formatting_tests.py [options]

Options:
    --coverage          Run with coverage analysis
    --performance       Include performance tests
    --integration       Include integration tests
    --verbose           Verbose output
    --report            Generate detailed HTML report
    --benchmark         Run benchmark tests
"""

import sys
import os
import argparse
import subprocess
import time
import json
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'lightrag_integration'))

def run_tests_with_coverage():
    """Run tests with coverage analysis."""
    print("🧪 Running Response Formatting Tests with Coverage Analysis...")
    
    # Coverage command
    cmd = [
        sys.executable, "-m", "pytest",
        "test_response_formatting_comprehensive.py",
        "--cov=lightrag_integration.clinical_metabolomics_rag",
        "--cov-report=html:coverage_response_formatting",
        "--cov-report=term-missing",
        "--cov-branch",
        "-v"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Coverage tests completed successfully!")
        print("\n📊 Coverage Summary:")
        print(result.stdout)
    else:
        print("❌ Coverage tests failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def run_performance_tests():
    """Run performance-focused tests."""
    print("\n⚡ Running Performance Tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "test_response_formatting_comprehensive.py::TestPerformanceFormatting",
        "-v",
        "--tb=short"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ Performance tests completed in {elapsed_time:.2f} seconds!")
        print("Performance test results:")
        print(result.stdout)
    else:
        print(f"❌ Performance tests failed after {elapsed_time:.2f} seconds!")
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running Integration Tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "test_response_formatting_comprehensive.py::TestIntegrationFormatting",
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Integration tests completed successfully!")
        print("Integration test results:")
        print(result.stdout)
    else:
        print("❌ Integration tests failed!")
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def run_benchmark_tests():
    """Run benchmark tests to measure performance."""
    print("\n📈 Running Benchmark Tests...")
    
    try:
        # Import test modules
        from test_response_formatting_comprehensive import (
            TestDataProvider, 
            BiomedicalResponseFormatter,
            ResponseValidator
        )
        
        # Performance benchmarks
        benchmarks = {}
        sample_data = TestDataProvider()
        
        # Benchmark formatter initialization
        start_time = time.time()
        formatter = BiomedicalResponseFormatter()
        benchmarks['formatter_init'] = time.time() - start_time
        
        # Benchmark validator initialization
        start_time = time.time()
        validator = ResponseValidator()
        benchmarks['validator_init'] = time.time() - start_time
        
        # Benchmark entity extraction
        raw_response = sample_data.get_sample_biomedical_response()
        start_time = time.time()
        result = formatter.format_response(raw_response)
        benchmarks['entity_extraction'] = time.time() - start_time
        
        # Benchmark statistical formatting
        statistical_response = sample_data.get_statistical_response()
        start_time = time.time()
        stat_result = formatter.format_response(statistical_response)
        benchmarks['statistical_formatting'] = time.time() - start_time
        
        # Benchmark validation
        import asyncio
        async def benchmark_validation():
            start_time = time.time()
            validation_result = await validator.validate_response(
                raw_response, 
                "Test query"
            )
            return time.time() - start_time
        
        benchmarks['response_validation'] = asyncio.run(benchmark_validation())
        
        # Display benchmark results
        print("✅ Benchmark Tests Completed!")
        print("\n📊 Performance Benchmarks:")
        for operation, elapsed_time in benchmarks.items():
            print(f"  {operation}: {elapsed_time:.4f} seconds")
        
        # Save benchmark results
        benchmark_file = Path(__file__).parent / "benchmark_results.json"
        with open(benchmark_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'benchmarks': benchmarks
            }, f, indent=2)
        
        print(f"\n💾 Benchmark results saved to: {benchmark_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark tests failed: {e}")
        return False


def run_specific_test_class(class_name):
    """Run tests for a specific test class."""
    print(f"\n🎯 Running Tests for {class_name}...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        f"test_response_formatting_comprehensive.py::{class_name}",
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {class_name} tests completed successfully!")
        return True
    else:
        print(f"❌ {class_name} tests failed!")
        print("STDERR:", result.stderr)
        return False


def generate_html_report():
    """Generate detailed HTML test report."""
    print("\n📝 Generating HTML Test Report...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "test_response_formatting_comprehensive.py",
        "--html=response_formatting_test_report.html",
        "--self-contained-html",
        "-v"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    
    if result.returncode == 0:
        report_path = Path(__file__).parent / "response_formatting_test_report.html"
        print(f"✅ HTML report generated: {report_path}")
        return True
    else:
        print("❌ HTML report generation failed!")
        print("STDERR:", result.stderr)
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive response formatting tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_response_formatting_tests.py --coverage
    python run_response_formatting_tests.py --performance --integration
    python run_response_formatting_tests.py --benchmark --report
        """
    )
    
    parser.add_argument("--coverage", action="store_true",
                       help="Run with coverage analysis")
    parser.add_argument("--performance", action="store_true",
                       help="Include performance tests")
    parser.add_argument("--integration", action="store_true",
                       help="Include integration tests")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--report", action="store_true",
                       help="Generate detailed HTML report")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark tests")
    parser.add_argument("--class", dest="test_class",
                       help="Run specific test class")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests with full analysis")
    
    args = parser.parse_args()
    
    print("🔬 Response Formatting Comprehensive Test Suite")
    print("=" * 60)
    
    # Change to test directory
    os.chdir(Path(__file__).parent)
    
    success_count = 0
    total_tests = 0
    
    # Run specific test class if requested
    if args.test_class:
        total_tests += 1
        if run_specific_test_class(args.test_class):
            success_count += 1
    
    # Run all tests if --all flag is used
    if args.all:
        args.coverage = True
        args.performance = True
        args.integration = True
        args.benchmark = True
        args.report = True
    
    # Run coverage tests
    if args.coverage or not any([args.performance, args.integration, args.benchmark, args.test_class]):
        total_tests += 1
        if run_tests_with_coverage():
            success_count += 1
    
    # Run performance tests
    if args.performance:
        total_tests += 1
        if run_performance_tests():
            success_count += 1
    
    # Run integration tests
    if args.integration:
        total_tests += 1
        if run_integration_tests():
            success_count += 1
    
    # Run benchmark tests
    if args.benchmark:
        total_tests += 1
        if run_benchmark_tests():
            success_count += 1
    
    # Generate HTML report
    if args.report:
        generate_html_report()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📋 Test Summary: {success_count}/{total_tests} test suites passed")
    
    if success_count == total_tests:
        print("✅ All test suites completed successfully!")
        return 0
    else:
        print(f"❌ {total_tests - success_count} test suite(s) failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)