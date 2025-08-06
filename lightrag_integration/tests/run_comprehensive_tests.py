#!/usr/bin/env python3
"""
Comprehensive Test Runner for API Cost Monitoring System.

This test runner provides:
- Complete test suite execution with coverage reporting
- Test categorization and selective execution
- Performance benchmarking and reporting
- CI/CD integration support
- Test result analysis and reporting

Usage:
    python run_comprehensive_tests.py                    # Run all tests
    python run_comprehensive_tests.py --unit             # Run only unit tests
    python run_comprehensive_tests.py --integration      # Run only integration tests
    python run_comprehensive_tests.py --performance      # Run performance tests
    python run_comprehensive_tests.py --coverage         # Run with coverage report
    python run_comprehensive_tests.py --benchmark        # Run with benchmarking

Author: Claude Code (Anthropic)
Created: August 6, 2025
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


class ComprehensiveTestRunner:
    """Main test runner for the API cost monitoring system."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.results = {}
        
        # Test categories and their corresponding files
        self.test_categories = {
            'unit': [
                'test_cost_persistence_comprehensive.py',
                'test_budget_management_comprehensive.py', 
                'test_research_categorization_comprehensive.py',
                'test_audit_trail_comprehensive.py',
                'test_api_metrics_logging_comprehensive.py',
                'test_alert_system_comprehensive.py'
            ],
            'integration': [
                'test_budget_management_integration.py',
                'test_comprehensive_budget_alerting.py'  # Existing integration test
            ],
            'performance': [
                # Performance tests are marked within other files
            ]
        }
        
        # Coverage configuration
        self.coverage_config = {
            'source': [str(self.project_root)],
            'omit': [
                '*/tests/*',
                '*/test_*.py',
                '*/__pycache__/*',
                '*/venv/*',
                '*/env/*'
            ],
            'include': [
                '*/lightrag_integration/*.py'
            ]
        }
    
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                'command': ' '.join(command),
                'description': description,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': duration,
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                'command': ' '.join(command),
                'description': description,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out after 10 minutes',
                'duration': 600,
                'success': False
            }
        except Exception as e:
            return {
                'command': ' '.join(command),
                'description': description,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'duration': 0,
                'success': False
            }
    
    def run_unit_tests(self, with_coverage: bool = False) -> Dict[str, Any]:
        """Run unit tests."""
        test_files = self.test_categories['unit']
        
        base_command = ['python', '-m', 'pytest']
        
        if with_coverage:
            base_command.extend(['--cov=lightrag_integration', '--cov-report=html', '--cov-report=term'])
        
        base_command.extend([
            '-v',
            '--tb=short',
            '-m', 'not slow',  # Skip slow tests in regular unit test run
            '--durations=10'
        ])
        
        base_command.extend(test_files)
        
        return self.run_command(base_command, "Unit Tests")
    
    def run_integration_tests(self, with_coverage: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        test_files = self.test_categories['integration']
        
        base_command = ['python', '-m', 'pytest']
        
        if with_coverage:
            base_command.extend(['--cov=lightrag_integration', '--cov-append', '--cov-report=term'])
        
        base_command.extend([
            '-v',
            '--tb=short',
            '--durations=20'
        ])
        
        base_command.extend(test_files)
        
        return self.run_command(base_command, "Integration Tests")
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        base_command = [
            'python', '-m', 'pytest',
            '-v',
            '--tb=short',
            '-m', 'performance',
            '--durations=0',  # Show all durations for performance analysis
            '--benchmark-only',  # Only run benchmark tests if pytest-benchmark is available
        ]
        
        # Run across all test files but only performance-marked tests
        all_test_files = []
        for category_files in self.test_categories.values():
            all_test_files.extend(category_files)
        
        base_command.extend(all_test_files)
        
        return self.run_command(base_command, "Performance Tests")
    
    def run_concurrent_tests(self) -> Dict[str, Any]:
        """Run concurrent/threading tests."""
        base_command = [
            'python', '-m', 'pytest',
            '-v',
            '--tb=short',
            '-m', 'concurrent',
            '--durations=10',
        ]
        
        # Run across all test files but only concurrent-marked tests
        all_test_files = []
        for category_files in self.test_categories.values():
            all_test_files.extend(category_files)
        
        base_command.extend(all_test_files)
        
        return self.run_command(base_command, "Concurrent Tests")
    
    def run_all_tests(self, with_coverage: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Run all test categories."""
        results = {
            'unit': [],
            'integration': [],
            'performance': [],
            'concurrent': []
        }
        
        print("\n🚀 Running Comprehensive Test Suite for API Cost Monitoring System")
        print("=" * 80)
        
        # Run unit tests
        print("\n📊 Phase 1: Unit Tests")
        unit_result = self.run_unit_tests(with_coverage)
        results['unit'].append(unit_result)
        
        # Run integration tests
        print("\n🔗 Phase 2: Integration Tests")
        integration_result = self.run_integration_tests(with_coverage)
        results['integration'].append(integration_result)
        
        # Run performance tests
        print("\n⚡ Phase 3: Performance Tests")
        performance_result = self.run_performance_tests()
        results['performance'].append(performance_result)
        
        # Run concurrent tests
        print("\n🧵 Phase 4: Concurrent Tests")
        concurrent_result = self.run_concurrent_tests()
        results['concurrent'].append(concurrent_result)
        
        return results
    
    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate comprehensive coverage report."""
        command = [
            'python', '-m', 'pytest',
            '--cov=lightrag_integration',
            '--cov-report=html:htmlcov',
            '--cov-report=xml:coverage.xml',
            '--cov-report=term-missing',
            '--cov-branch',
            '-q'  # Quiet mode for coverage generation
        ]
        
        # Add all test files
        all_test_files = []
        for category_files in self.test_categories.values():
            all_test_files.extend(category_files)
        command.extend(all_test_files)
        
        return self.run_command(command, "Coverage Report Generation")
    
    def analyze_test_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze test results and generate summary."""
        total_tests = 0
        total_failures = 0
        total_duration = 0
        category_summary = {}
        
        for category, category_results in results.items():
            category_tests = 0
            category_failures = 0
            category_duration = 0
            
            for result in category_results:
                if not result['success']:
                    category_failures += 1
                category_duration += result['duration']
                
                # Parse pytest output to count tests (simplified)
                if result['success']:
                    stdout_lines = result['stdout'].split('\n')
                    for line in stdout_lines:
                        if ' passed' in line and ('failed' in line or 'error' in line or 'passed' == line.strip().split()[-1]):
                            # Parse pytest summary line
                            parts = line.split()
                            if 'passed' in parts:
                                passed_idx = parts.index('passed')
                                if passed_idx > 0:
                                    try:
                                        passed_count = int(parts[passed_idx - 1])
                                        category_tests += passed_count
                                    except (ValueError, IndexError):
                                        pass
                            break
            
            category_summary[category] = {
                'tests': category_tests,
                'failures': category_failures,
                'duration': category_duration,
                'success_rate': ((category_tests - category_failures) / max(category_tests, 1)) * 100
            }
            
            total_tests += category_tests
            total_failures += category_failures
            total_duration += category_duration
        
        overall_success_rate = ((total_tests - total_failures) / max(total_tests, 1)) * 100
        
        return {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_duration': total_duration,
            'overall_success_rate': overall_success_rate,
            'categories': category_summary
        }
    
    def print_summary_report(self, results: Dict[str, List[Dict[str, Any]]], analysis: Dict[str, Any]):
        """Print comprehensive summary report."""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TEST SUITE SUMMARY REPORT")
        print("="*80)
        
        print(f"\n📈 Overall Results:")
        print(f"   Total Tests: {analysis['total_tests']}")
        print(f"   Failed Tests: {analysis['total_failures']}")
        print(f"   Success Rate: {analysis['overall_success_rate']:.1f}%")
        print(f"   Total Duration: {analysis['total_duration']:.1f} seconds")
        
        print(f"\n📊 Category Breakdown:")
        for category, summary in analysis['categories'].items():
            status = "✅" if summary['failures'] == 0 else "❌"
            print(f"   {status} {category.upper():<12}: {summary['tests']:3d} tests, "
                  f"{summary['failures']:2d} failures, {summary['duration']:6.1f}s, "
                  f"{summary['success_rate']:5.1f}% success")
        
        print(f"\n🔍 Detailed Results:")
        for category, category_results in results.items():
            print(f"\n   {category.upper()} Tests:")
            for result in category_results:
                status = "✅" if result['success'] else "❌"
                print(f"     {status} {result['description']}: {result['duration']:.1f}s")
                
                if not result['success']:
                    print(f"       Error: {result['stderr'][:200]}...")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if analysis['total_failures'] == 0:
            print("   🎉 All tests passed! System is ready for production.")
        else:
            print(f"   🔧 {analysis['total_failures']} tests failed. Review failures before deployment.")
        
        if analysis['total_duration'] > 300:  # 5 minutes
            print("   ⏰ Test suite is slow. Consider optimizing test performance.")
        
        print(f"\n📁 Generated Files:")
        coverage_dir = self.test_dir / "htmlcov"
        if coverage_dir.exists():
            print(f"   📊 Coverage Report: {coverage_dir}/index.html")
        
        coverage_xml = self.test_dir / "coverage.xml"
        if coverage_xml.exists():
            print(f"   📄 Coverage XML: {coverage_xml}")
    
    def main(self):
        """Main test runner entry point."""
        parser = argparse.ArgumentParser(
            description="Comprehensive test runner for API Cost Monitoring System"
        )
        
        parser.add_argument('--unit', action='store_true', help='Run only unit tests')
        parser.add_argument('--integration', action='store_true', help='Run only integration tests')
        parser.add_argument('--performance', action='store_true', help='Run only performance tests')
        parser.add_argument('--concurrent', action='store_true', help='Run only concurrent tests')
        parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
        parser.add_argument('--benchmark', action='store_true', help='Run with benchmarking')
        parser.add_argument('--fast', action='store_true', help='Skip slow tests')
        
        args = parser.parse_args()
        
        start_time = time.time()
        
        try:
            if args.unit:
                result = self.run_unit_tests(args.coverage)
                results = {'unit': [result]}
            elif args.integration:
                result = self.run_integration_tests(args.coverage)
                results = {'integration': [result]}
            elif args.performance:
                result = self.run_performance_tests()
                results = {'performance': [result]}
            elif args.concurrent:
                result = self.run_concurrent_tests()
                results = {'concurrent': [result]}
            else:
                # Run all tests
                results = self.run_all_tests(args.coverage)
            
            # Generate coverage report if requested
            if args.coverage:
                print("\n📊 Generating Coverage Report...")
                coverage_result = self.generate_coverage_report()
                if not coverage_result['success']:
                    print(f"❌ Coverage report generation failed: {coverage_result['stderr']}")
            
            # Analyze and print results
            analysis = self.analyze_test_results(results)
            self.print_summary_report(results, analysis)
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            print(f"\n⏱️  Total Execution Time: {total_duration:.1f} seconds")
            
            # Exit with appropriate code
            if analysis['total_failures'] > 0:
                print("\n❌ Some tests failed. Check the output above for details.")
                sys.exit(1)
            else:
                print("\n✅ All tests passed successfully!")
                sys.exit(0)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Test execution interrupted by user.")
            sys.exit(130)
        except Exception as e:
            print(f"\n💥 Test runner error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    runner = ComprehensiveTestRunner()
    runner.main()