#!/usr/bin/env python3
"""
Unified Progress Tracking Test Runner

This script provides a convenient way to run the unified progress tracking
tests with various options and configurations.

Usage:
    python run_unified_progress_tests.py [options]
    
Options:
    --all                Run all tests (default)
    --core               Run core functionality tests only
    --integration        Run integration tests only
    --performance        Run performance tests only
    --edge-cases         Run edge case tests only
    --quick              Run quick tests only (excludes slow tests)
    --benchmark          Run performance benchmarks only
    --coverage           Run with coverage reporting
    --verbose            Enable verbose output
    --parallel           Run tests in parallel
    --failfast           Stop on first failure
    --help               Show this help message

Examples:
    python run_unified_progress_tests.py --core --verbose
    python run_unified_progress_tests.py --performance --coverage
    python run_unified_progress_tests.py --all --parallel --coverage

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to Python path to ensure imports work
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))


class UnifiedProgressTestRunner:
    """Test runner for unified progress tracking tests."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.test_results = {}
        
        # Test file mappings
        self.test_files = {
            'core': 'test_unified_progress_tracking.py',
            'fixtures': 'test_unified_progress_fixtures.py',
            'comprehensive': 'test_unified_progress_comprehensive.py'
        }
        
        # Test category mappings
        self.test_categories = {
            'core': [
                'TestUnifiedProgressTrackerCore',
                'TestPhaseWeightsAndProgress',
                'TestCallbackSystem',
                'TestProgressTrackingConfiguration'
            ],
            'integration': [
                'TestProgressTrackingIntegration',
                'TestKnowledgeBaseIntegration',
                'TestComprehensiveIntegration'
            ],
            'performance': [
                'TestPerformance',
                'TestStressAndPerformance'
            ],
            'edge-cases': [
                'TestErrorHandlingAndEdgeCases',
                'TestEdgeCasesAndBoundaryConditions',
                'TestThreadSafety'
            ]
        }
    
    def build_pytest_command(self, args: argparse.Namespace) -> List[str]:
        """Build pytest command based on arguments."""
        cmd = ['python', '-m', 'pytest']
        
        # Add test files/patterns based on test selection
        if args.all:
            cmd.extend([
                'test_unified_progress_tracking.py',
                'test_unified_progress_comprehensive.py'
            ])
        elif args.core:
            cmd.append('test_unified_progress_tracking.py')
            # Add specific core test classes
            for test_class in self.test_categories['core']:
                cmd.extend(['-k', test_class])
        elif args.integration:
            cmd.append('test_unified_progress_tracking.py')
            cmd.append('test_unified_progress_comprehensive.py')
            # Filter for integration tests
            integration_filter = ' or '.join(self.test_categories['integration'])
            cmd.extend(['-k', integration_filter])
        elif args.performance:
            cmd.append('test_unified_progress_tracking.py')
            cmd.append('test_unified_progress_comprehensive.py')
            # Filter for performance tests
            performance_filter = ' or '.join(self.test_categories['performance'])
            cmd.extend(['-k', performance_filter])
        elif args.edge_cases:
            cmd.append('test_unified_progress_tracking.py')
            cmd.append('test_unified_progress_comprehensive.py')
            # Filter for edge case tests
            edge_case_filter = ' or '.join(self.test_categories['edge-cases'])
            cmd.extend(['-k', edge_case_filter])
        elif args.quick:
            # Run quick tests (exclude slow/performance tests)
            cmd.extend([
                'test_unified_progress_tracking.py',
                '-k', 'not (performance or stress or large_collection)'
            ])
        else:
            # Default: run main test file
            cmd.append('test_unified_progress_tracking.py')
        
        # Add coverage options
        if args.coverage:
            cmd.extend([
                '--cov=lightrag_integration.unified_progress_tracker',
                '--cov=lightrag_integration.progress_config',
                '--cov-report=html',
                '--cov-report=term-missing'
            ])
        
        # Add verbosity options
        if args.verbose:
            cmd.append('-v')
        else:
            cmd.append('-q')
        
        # Add parallel execution
        if args.parallel:
            try:
                import pytest_xdist
                cmd.extend(['-n', 'auto'])
            except ImportError:
                print("Warning: pytest-xdist not installed, running sequentially")
        
        # Add fail fast option
        if args.failfast:
            cmd.append('-x')
        
        # Add additional pytest options
        cmd.extend([
            '--tb=short',  # Shorter tracebacks
            '--strict-markers',  # Strict marker validation
        ])
        
        return cmd
    
    def run_benchmark_tests(self) -> Dict[str, Any]:
        """Run performance benchmark tests."""
        print("🚀 Running Performance Benchmarks...")
        print("=" * 60)
        
        try:
            # Import and run benchmark
            # Change to project root to allow proper imports
            original_cwd = os.getcwd()
            os.chdir(self.project_root.parent)
            sys.path.insert(0, str(self.project_root.parent))
            from lightrag_integration.tests.test_unified_progress_comprehensive import TestComprehensiveRunner
            
            runner = TestComprehensiveRunner()
            
            print("📊 Running core functionality validation...")
            runner.test_all_core_functionality()
            print("✅ Core functionality: PASSED")
            
            print("\n📊 Running performance benchmarks...")
            performance_results = runner.test_performance_benchmarks()
            
            print("\n🎯 Performance Results:")
            for metric, value in performance_results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")
            
            return {
                'status': 'success',
                'results': performance_results
            }
        
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
        finally:
            # Restore original working directory
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
    
    def run_quick_validation(self) -> bool:
        """Run quick validation to ensure basic functionality works."""
        print("🔍 Running Quick Validation...")
        
        try:
            # Import key components
            from lightrag_integration.unified_progress_tracker import (
                KnowledgeBaseProgressTracker,
                KnowledgeBasePhase,
                PhaseWeights
            )
            from lightrag_integration.progress_config import ProgressTrackingConfig
            
            # Quick functionality test
            tracker = KnowledgeBaseProgressTracker()
            tracker.start_initialization(total_documents=1)
            
            # Test each phase quickly
            for phase in KnowledgeBasePhase:
                tracker.start_phase(phase, f"Quick test {phase.value}")
                tracker.update_phase_progress(phase, 0.5, "Halfway")
                tracker.complete_phase(phase, f"Completed {phase.value}")
            
            final_state = tracker.get_current_state()
            
            # Validate results
            assert abs(final_state.overall_progress - 1.0) < 0.001
            assert all(info.is_completed for info in final_state.phase_info.values())
            
            print("✅ Quick validation: PASSED")
            return True
        
        except Exception as e:
            print(f"❌ Quick validation failed: {e}")
            return False
    
    def run_tests(self, args: argparse.Namespace) -> bool:
        """Run the selected tests."""
        # Change to test directory
        os.chdir(self.test_dir)
        
        # Handle special cases
        if args.benchmark:
            result = self.run_benchmark_tests()
            return result['status'] == 'success'
        
        # Run quick validation first unless running all tests
        if not args.all and not args.quick:
            if not self.run_quick_validation():
                print("❌ Quick validation failed, aborting test run")
                return False
        
        # Build and execute pytest command
        cmd = self.build_pytest_command(args)
        
        print(f"🧪 Running Tests...")
        print(f"Command: {' '.join(cmd[2:])}")  # Skip 'python -m'
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            print("\n" + "=" * 60)
            print(f"⏱️  Test execution time: {execution_time:.2f} seconds")
            
            if result.returncode == 0:
                print("🎉 All tests passed!")
                
                # Show coverage report if generated
                if args.coverage:
                    coverage_dir = self.test_dir / "htmlcov"
                    if coverage_dir.exists():
                        print(f"📊 Coverage report generated: {coverage_dir}/index.html")
                
                return True
            else:
                print(f"❌ Tests failed with return code: {result.returncode}")
                return False
        
        except KeyboardInterrupt:
            print("\n⚠️  Test execution interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
    
    def print_help(self):
        """Print detailed help information."""
        help_text = """
🧪 Unified Progress Tracking Test Runner

This tool provides convenient options for running the comprehensive test suite
for the unified progress tracking system.

📋 Test Categories:

  Core Tests:
    • UnifiedProgressTracker initialization and basic operations
    • Phase weights and progress calculation
    • Callback system functionality
    • Configuration validation

  Integration Tests:
    • PDF processor integration
    • Knowledge base initialization workflow
    • Full system integration scenarios

  Performance Tests:
    • High-frequency progress updates
    • Memory usage validation
    • Concurrent operation testing
    • Stress testing scenarios

  Edge Case Tests:
    • Error handling and recovery
    • Thread safety validation
    • Boundary condition testing
    • Configuration edge cases

🚀 Quick Start:

  # Run all tests with coverage
  python run_unified_progress_tests.py --all --coverage --verbose
  
  # Quick smoke test
  python run_unified_progress_tests.py --quick
  
  # Performance benchmarking
  python run_unified_progress_tests.py --benchmark
  
  # Core functionality only
  python run_unified_progress_tests.py --core --verbose

📊 Performance Benchmarks:

The benchmark mode runs specialized performance tests and provides
detailed metrics about system performance including:
  • Progress update throughput (updates/second)
  • Callback system overhead
  • Memory usage efficiency
  • Concurrent operation support

📈 Coverage Reporting:

When --coverage is used, an HTML coverage report is generated showing:
  • Line coverage for all modules
  • Branch coverage analysis
  • Missing coverage identification
  • Coverage trends over time

For more information, see UNIFIED_PROGRESS_TESTING_README.md
        """
        print(help_text)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run unified progress tracking tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--all', action='store_true', 
                           help='Run all tests (default)')
    test_group.add_argument('--core', action='store_true',
                           help='Run core functionality tests only')
    test_group.add_argument('--integration', action='store_true',
                           help='Run integration tests only')
    test_group.add_argument('--performance', action='store_true',
                           help='Run performance tests only')
    test_group.add_argument('--edge-cases', action='store_true',
                           help='Run edge case tests only')
    test_group.add_argument('--quick', action='store_true',
                           help='Run quick tests only (excludes slow tests)')
    test_group.add_argument('--benchmark', action='store_true',
                           help='Run performance benchmarks only')
    
    # Execution options
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage reporting')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--failfast', action='store_true',
                       help='Stop on first failure')
    
    # Help option
    parser.add_argument('--help-detailed', action='store_true',
                       help='Show detailed help information')
    
    args = parser.parse_args()
    
    # Handle detailed help
    if args.help_detailed:
        runner = UnifiedProgressTestRunner()
        runner.print_help()
        return 0
    
    # Default to all tests if no specific option selected
    if not any([args.core, args.integration, args.performance, 
                args.edge_cases, args.quick, args.benchmark]):
        args.all = True
    
    # Create and run test runner
    runner = UnifiedProgressTestRunner()
    
    print("🧪 Unified Progress Tracking Test Suite")
    print("=" * 60)
    
    success = runner.run_tests(args)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())