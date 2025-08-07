#!/usr/bin/env python3
"""
Comprehensive Test Runner for Factual Accuracy Validation System.

This script provides comprehensive test execution capabilities for the factual accuracy
validation system including different test suites, coverage analysis, performance
benchmarking, and report generation.

Usage:
    python run_validation_tests.py --help
    python run_validation_tests.py --suite all
    python run_validation_tests.py --suite unit --coverage
    python run_validation_tests.py --suite performance --benchmark
    python run_validation_tests.py --suite integration --verbose

Test Suites:
    - unit: Unit tests for individual components
    - integration: End-to-end integration tests
    - performance: Performance and scalability tests
    - error_handling: Error conditions and edge cases
    - mock: Mock-based isolation tests
    - all: All test suites

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import tempfile
import shutil


class ValidationTestRunner:
    """Comprehensive test runner for validation system testing."""
    
    def __init__(self):
        """Initialize test runner."""
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        self.results_dir = self.test_dir / "validation_test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Test suite definitions
        self.test_suites = {
            'unit': {
                'description': 'Unit tests for individual components',
                'markers': ['accuracy_scorer', 'validation'],
                'files': ['test_accuracy_scorer_comprehensive.py'],
                'timeout': 300,  # 5 minutes
                'parallel': True
            },
            'integration': {
                'description': 'End-to-end integration tests',
                'markers': ['integration_validation'],
                'files': ['test_integrated_factual_validation.py'],
                'timeout': 900,  # 15 minutes
                'parallel': False
            },
            'performance': {
                'description': 'Performance and scalability tests',
                'markers': ['performance_validation'],
                'files': ['test_validation_performance.py'],
                'timeout': 1200,  # 20 minutes
                'parallel': False
            },
            'error_handling': {
                'description': 'Error conditions and edge cases',
                'markers': ['error_handling_validation'],
                'files': ['test_validation_error_handling.py'],
                'timeout': 600,  # 10 minutes
                'parallel': True
            },
            'mock': {
                'description': 'Mock-based isolation tests',
                'markers': ['mock_validation'],
                'files': ['test_validation_mocks.py'],
                'timeout': 300,  # 5 minutes
                'parallel': True
            }
        }
        
        # Configure logging
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_file = self.results_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Test runner initialized - logging to {log_file}")
    
    async def run_test_suite(self, 
                           suite_name: str,
                           coverage: bool = False,
                           benchmark: bool = False,
                           verbose: bool = False,
                           parallel: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a specific test suite.
        
        Args:
            suite_name: Name of test suite to run
            coverage: Enable coverage analysis
            benchmark: Enable performance benchmarking
            verbose: Enable verbose output
            parallel: Number of parallel processes (None for auto)
            
        Returns:
            Test results dictionary
        """
        
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite_config = self.test_suites[suite_name]
        self.logger.info(f"Running test suite: {suite_name} - {suite_config['description']}")
        
        start_time = time.time()
        
        # Prepare pytest command
        cmd = ['python', '-m', 'pytest']
        
        # Add test files or markers
        if suite_config['files']:
            for test_file in suite_config['files']:
                cmd.append(str(self.test_dir / test_file))
        else:
            # Use markers if no specific files
            for marker in suite_config['markers']:
                cmd.extend(['-m', marker])
        
        # Add coverage if requested
        if coverage:
            coverage_file = self.results_dir / f"coverage_{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
            cmd.extend([
                '--cov=lightrag_integration',
                '--cov-report=xml:' + str(coverage_file),
                '--cov-report=term-missing',
                '--cov-fail-under=80'
            ])
        
        # Add parallelization if supported and requested
        if suite_config['parallel'] and parallel:
            cmd.extend(['-n', str(parallel)])
        elif suite_config['parallel'] and parallel is None:
            cmd.extend(['-n', 'auto'])
        
        # Add timeout
        cmd.extend(['--timeout', str(suite_config['timeout'])])
        
        # Add verbose output if requested
        if verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.extend(['--tb=short'])
        
        # Add JSON reporting
        json_report = self.results_dir / f"results_{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        cmd.extend(['--json-report', '--json-report-file', str(json_report)])
        
        # Add benchmark options if requested
        if benchmark and suite_name == 'performance':
            cmd.extend([
                '--benchmark-only',
                '--benchmark-json=' + str(self.results_dir / f"benchmark_{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            ])
        
        self.logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute tests
        try:
            # Change to test directory
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            # Run pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite_config['timeout'] + 60  # Add buffer to subprocess timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            test_results = {
                'suite_name': suite_name,
                'description': suite_config['description'],
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'execution_time_seconds': execution_time,
                'return_code': result.returncode,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd),
                'configuration': {
                    'coverage_enabled': coverage,
                    'benchmark_enabled': benchmark,
                    'verbose_enabled': verbose,
                    'parallel_processes': parallel
                }
            }
            
            # Load JSON report if available
            if json_report.exists():
                try:
                    with open(json_report, 'r') as f:
                        json_data = json.load(f)
                        test_results['detailed_results'] = json_data
                except Exception as e:
                    self.logger.warning(f"Could not load JSON report: {e}")
            
            # Log results
            if test_results['success']:
                self.logger.info(f"✅ Test suite '{suite_name}' passed in {execution_time:.2f}s")
            else:
                self.logger.error(f"❌ Test suite '{suite_name}' failed in {execution_time:.2f}s")
                self.logger.error(f"STDERR: {result.stderr}")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            self.logger.error(f"⏰ Test suite '{suite_name}' timed out after {execution_time:.2f}s")
            
            return {
                'suite_name': suite_name,
                'description': suite_config['description'],
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'execution_time_seconds': execution_time,
                'return_code': -1,
                'success': False,
                'error': 'Test suite timed out',
                'timeout_seconds': suite_config['timeout'],
                'command': ' '.join(cmd)
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"💥 Test suite '{suite_name}' encountered error: {e}")
            
            return {
                'suite_name': suite_name,
                'description': suite_config['description'],
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'execution_time_seconds': execution_time,
                'return_code': -1,
                'success': False,
                'error': str(e),
                'command': ' '.join(cmd)
            }
        
        finally:
            os.chdir(original_cwd)
    
    async def run_all_suites(self,
                           coverage: bool = False,
                           benchmark: bool = False,
                           verbose: bool = False,
                           parallel: Optional[int] = None,
                           fail_fast: bool = False) -> Dict[str, Any]:
        """
        Run all test suites.
        
        Args:
            coverage: Enable coverage analysis
            benchmark: Enable performance benchmarking
            verbose: Enable verbose output
            parallel: Number of parallel processes
            fail_fast: Stop on first failure
            
        Returns:
            Combined test results
        """
        
        self.logger.info("🚀 Starting comprehensive validation test run")
        overall_start_time = time.time()
        
        all_results = {
            'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'configuration': {
                'coverage_enabled': coverage,
                'benchmark_enabled': benchmark,
                'verbose_enabled': verbose,
                'parallel_processes': parallel,
                'fail_fast_enabled': fail_fast
            },
            'suite_results': {},
            'summary': {}
        }
        
        # Run test suites in order (performance last)
        suite_order = ['unit', 'mock', 'error_handling', 'integration', 'performance']
        successful_suites = 0
        failed_suites = 0
        
        for suite_name in suite_order:
            if suite_name not in self.test_suites:
                continue
                
            self.logger.info(f"\n📋 Running test suite: {suite_name}")
            
            suite_result = await self.run_test_suite(
                suite_name=suite_name,
                coverage=coverage,
                benchmark=benchmark and suite_name == 'performance',
                verbose=verbose,
                parallel=parallel
            )
            
            all_results['suite_results'][suite_name] = suite_result
            
            if suite_result['success']:
                successful_suites += 1
                self.logger.info(f"✅ {suite_name} suite completed successfully")
            else:
                failed_suites += 1
                self.logger.error(f"❌ {suite_name} suite failed")
                
                if fail_fast:
                    self.logger.error("🛑 Stopping test run due to fail_fast option")
                    break
        
        overall_execution_time = time.time() - overall_start_time
        
        # Generate summary
        all_results['summary'] = {
            'total_execution_time_seconds': overall_execution_time,
            'total_suites': len(suite_order),
            'successful_suites': successful_suites,
            'failed_suites': failed_suites,
            'success_rate': successful_suites / len(suite_order) if suite_order else 0,
            'overall_success': failed_suites == 0,
            'end_time': datetime.now().isoformat()
        }
        
        # Save comprehensive results
        results_file = self.results_dir / f"comprehensive_results_{all_results['run_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Log final summary
        self.logger.info(f"\n🏁 Test run completed in {overall_execution_time:.2f}s")
        self.logger.info(f"📊 Summary: {successful_suites}/{len(suite_order)} suites passed")
        
        if all_results['summary']['overall_success']:
            self.logger.info("🎉 All test suites passed!")
        else:
            self.logger.error(f"💔 {failed_suites} test suite(s) failed")
        
        self.logger.info(f"📄 Detailed results saved to: {results_file}")
        
        return all_results
    
    def generate_coverage_report(self, coverage_files: List[Path]) -> Dict[str, Any]:
        """
        Generate combined coverage report from multiple coverage files.
        
        Args:
            coverage_files: List of coverage XML files
            
        Returns:
            Coverage report summary
        """
        
        # This is a simplified coverage report - in practice you might use
        # coverage.py's API or parse XML files for detailed metrics
        
        coverage_report = {
            'timestamp': datetime.now().isoformat(),
            'files_analyzed': len(coverage_files),
            'coverage_files': [str(f) for f in coverage_files],
            'summary': {
                'lines_covered': 0,
                'lines_total': 0,
                'coverage_percentage': 0.0
            }
        }
        
        # Save coverage report
        coverage_report_file = self.results_dir / f"coverage_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(coverage_report_file, 'w') as f:
            json.dump(coverage_report, f, indent=2)
        
        return coverage_report
    
    def cleanup_old_results(self, keep_days: int = 7):
        """
        Clean up old test result files.
        
        Args:
            keep_days: Number of days of results to keep
        """
        
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        cleaned_count = 0
        
        for file_path in self.results_dir.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"🧹 Cleaned up {cleaned_count} old result files")


async def main():
    """Main entry point for test runner."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for Factual Accuracy Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --suite all --coverage
  %(prog)s --suite unit --verbose
  %(prog)s --suite performance --benchmark
  %(prog)s --suite integration --parallel 4
  %(prog)s --suite error_handling --fail-fast
        """
    )
    
    parser.add_argument(
        '--suite',
        choices=['unit', 'integration', 'performance', 'error_handling', 'mock', 'all'],
        default='all',
        help='Test suite to run (default: all)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Enable coverage analysis'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Enable performance benchmarking (for performance suite)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose test output'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        help='Number of parallel test processes (default: auto)'
    )
    
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop on first test suite failure'
    )
    
    parser.add_argument(
        '--cleanup',
        type=int,
        default=7,
        help='Clean up result files older than N days (default: 7)'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = ValidationTestRunner()
    
    # Cleanup old results
    runner.cleanup_old_results(args.cleanup)
    
    try:
        if args.suite == 'all':
            # Run all test suites
            results = await runner.run_all_suites(
                coverage=args.coverage,
                benchmark=args.benchmark,
                verbose=args.verbose,
                parallel=args.parallel,
                fail_fast=args.fail_fast
            )
            
            # Exit with appropriate code
            sys.exit(0 if results['summary']['overall_success'] else 1)
        
        else:
            # Run specific test suite
            result = await runner.run_test_suite(
                suite_name=args.suite,
                coverage=args.coverage,
                benchmark=args.benchmark,
                verbose=args.verbose,
                parallel=args.parallel
            )
            
            # Exit with appropriate code
            sys.exit(0 if result['success'] else 1)
    
    except KeyboardInterrupt:
        runner.logger.info("🛑 Test run interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        runner.logger.error(f"💥 Test runner encountered error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())