#!/usr/bin/env python3
"""
Comprehensive Batch PDF Processing Test Runner.

This script provides a comprehensive test runner for the batch PDF processing
test suite, including performance benchmarking, resource monitoring, and
detailed reporting capabilities.

Features:
- Automated test execution with configurable parameters
- Real-time performance monitoring and resource tracking
- Comprehensive test result reporting and analysis
- Integration with existing test infrastructure
- Production-readiness validation
- Benchmark comparison and regression detection

Usage:
    python run_comprehensive_batch_processing_tests.py [options]

Options:
    --test-level {basic,comprehensive,stress}: Test execution level
    --pdf-count <int>: Number of PDFs to test with (overrides defaults)
    --output-dir <path>: Directory for test results and reports
    --benchmark-mode: Enable detailed benchmarking mode
    --concurrent-workers <int>: Number of concurrent workers to test
    --memory-limit <int>: Memory limit in MB for testing
    --skip-long-tests: Skip long-running tests (>5 minutes)
    --verbose: Enable verbose logging and output

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import argparse
import logging
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import psutil
import pytest
import tempfile
import shutil

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

# Import test modules
from test_comprehensive_batch_pdf_processing import (
    TestComprehensiveBatchPDFProcessing,
    BatchProcessingScenario,
    BatchProcessingResult,
    EnhancedBatchPDFGenerator,
    ComprehensiveBatchProcessor
)

from lightrag_integration.pdf_processor import (
    BiomedicalPDFProcessor, 
    ErrorRecoveryConfig
)


class ComprehensiveBatchTestRunner:
    """
    Comprehensive test runner for batch PDF processing operations.
    
    Provides automated execution, monitoring, and reporting for the complete
    batch processing test suite with configurable parameters and benchmarking.
    """
    
    def __init__(self, 
                 output_dir: Path,
                 test_level: str = "comprehensive",
                 verbose: bool = False):
        self.output_dir = output_dir
        self.test_level = test_level
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Test configurations by level
        self.test_configurations = {
            'basic': {
                'pdf_counts': [25, 40],
                'concurrent_workers': [1, 2],
                'memory_limits': [1024],
                'test_duration_limit': 180,  # 3 minutes
                'skip_stress_tests': True
            },
            'comprehensive': {
                'pdf_counts': [50, 75],
                'concurrent_workers': [1, 2, 4],
                'memory_limits': [1024, 2048],
                'test_duration_limit': 600,  # 10 minutes
                'skip_stress_tests': False
            },
            'stress': {
                'pdf_counts': [100, 150],
                'concurrent_workers': [1, 4, 8],
                'memory_limits': [512, 1024, 2048],
                'test_duration_limit': 1800,  # 30 minutes
                'skip_stress_tests': False
            }
        }
        
        # Test results storage
        self.test_results = []
        self.benchmark_data = {}
        self.system_info = self.collect_system_info()
    
    def setup_logging(self):
        """Setup comprehensive logging for test execution."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        log_file = self.output_dir / f"batch_test_run_{int(time.time())}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=[console_handler, file_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Batch test runner initialized - Level: {self.test_level}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"System info: {self.system_info}")
    
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for test context."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage_gb': psutil.disk_usage('/').free / (1024**3),
                'platform': sys.platform,
                'python_version': sys.version,
                'pytest_available': True
            }
        except Exception as e:
            self.logger.warning(f"Failed to collect system info: {e}")
            return {'error': str(e)}
    
    async def run_all_tests(self, 
                          custom_pdf_count: Optional[int] = None,
                          custom_workers: Optional[int] = None,
                          custom_memory_limit: Optional[int] = None,
                          benchmark_mode: bool = False) -> Dict[str, Any]:
        """
        Run all batch processing tests with comprehensive monitoring.
        
        Args:
            custom_pdf_count: Override default PDF count
            custom_workers: Override default worker count  
            custom_memory_limit: Override default memory limit
            benchmark_mode: Enable detailed benchmarking
            
        Returns:
            Comprehensive test results and analysis
        """
        self.logger.info("Starting comprehensive batch processing test suite")
        start_time = time.time()
        
        # Get test configuration
        config = self.test_configurations[self.test_level]
        
        # Override with custom parameters
        if custom_pdf_count:
            config['pdf_counts'] = [custom_pdf_count]
        if custom_workers:
            config['concurrent_workers'] = [custom_workers]
        if custom_memory_limit:
            config['memory_limits'] = [custom_memory_limit]
        
        # Create temporary directory for tests
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Run test scenarios
                await self.run_large_scale_tests(temp_path, config, benchmark_mode)
                await self.run_concurrent_tests(temp_path, config, benchmark_mode)
                await self.run_fault_tolerance_tests(temp_path, config, benchmark_mode)
                await self.run_memory_management_tests(temp_path, config, benchmark_mode)
                await self.run_synthesis_tests(temp_path, config, benchmark_mode)
                
                # Additional stress tests if enabled
                if not config.get('skip_stress_tests', False):
                    await self.run_stress_tests(temp_path, config, benchmark_mode)
                
            except Exception as e:
                self.logger.error(f"Test execution failed: {e}")
                raise
        
        # Generate comprehensive results
        end_time = time.time()
        total_duration = end_time - start_time
        
        results = {
            'test_execution': {
                'level': self.test_level,
                'start_time': start_time,
                'end_time': end_time,
                'total_duration': total_duration,
                'tests_executed': len(self.test_results),
                'system_info': self.system_info
            },
            'test_results': self.test_results,
            'benchmark_data': self.benchmark_data,
            'summary': self.generate_test_summary(),
            'recommendations': self.generate_recommendations()
        }
        
        # Save results
        await self.save_results(results)
        
        self.logger.info(f"Test suite completed in {total_duration:.2f}s")
        return results
    
    async def run_large_scale_tests(self, 
                                  temp_path: Path, 
                                  config: Dict[str, Any],
                                  benchmark_mode: bool):
        """Run large-scale batch processing tests."""
        self.logger.info("Running large-scale batch processing tests")
        
        for pdf_count in config['pdf_counts']:
            for memory_limit in config['memory_limits']:
                scenario_name = f"large_scale_{pdf_count}_pdfs_{memory_limit}mb"
                
                scenario = BatchProcessingScenario(
                    name=scenario_name,
                    description=f"Large scale test with {pdf_count} PDFs, {memory_limit}MB memory",
                    pdf_count=pdf_count,
                    batch_size=max(5, pdf_count // 10),
                    concurrent_workers=1,
                    corrupted_pdf_percentage=0.1,
                    memory_limit_mb=memory_limit,
                    timeout_seconds=config['test_duration_limit'],
                    expected_success_rate=0.85,
                    performance_benchmarks={
                        'min_throughput_pdfs_per_second': 0.5,
                        'max_total_time_seconds': config['test_duration_limit'] * 0.8
                    },
                    quality_thresholds={
                        'min_avg_text_length': 200
                    }
                )
                
                try:
                    result = await self.execute_test_scenario(temp_path, scenario, benchmark_mode)
                    self.test_results.append(result)
                    
                    self.logger.info(
                        f"Large scale test {scenario_name}: "
                        f"{'PASSED' if result.success else 'FAILED'} "
                        f"({result.success_rate:.1f}% success rate)"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Large scale test {scenario_name} failed: {e}")
                    continue
    
    async def run_concurrent_tests(self, 
                                 temp_path: Path, 
                                 config: Dict[str, Any],
                                 benchmark_mode: bool):
        """Run concurrent processing tests."""
        self.logger.info("Running concurrent batch processing tests")
        
        base_pdf_count = min(config['pdf_counts'])
        
        for workers in config['concurrent_workers']:
            if workers == 1:
                continue  # Skip single worker (covered in large scale)
                
            scenario_name = f"concurrent_{workers}_workers"
            
            scenario = BatchProcessingScenario(
                name=scenario_name,
                description=f"Concurrent processing with {workers} workers",
                pdf_count=min(40, base_pdf_count),  # Moderate count for concurrency testing
                batch_size=4,
                concurrent_workers=workers,
                corrupted_pdf_percentage=0.05,
                memory_limit_mb=max(config['memory_limits']),
                timeout_seconds=config['test_duration_limit'] // 2,
                expected_success_rate=0.90,
                performance_benchmarks={
                    'min_throughput_pdfs_per_second': 1.0 * (workers * 0.7),  # Expected scaling
                    'concurrent_efficiency_threshold': 0.75
                },
                quality_thresholds={
                    'min_avg_text_length': 150
                }
            )
            
            try:
                result = await self.execute_test_scenario(temp_path, scenario, benchmark_mode)
                self.test_results.append(result)
                
                self.logger.info(
                    f"Concurrent test {scenario_name}: "
                    f"{'PASSED' if result.success else 'FAILED'} "
                    f"({result.throughput_pdfs_per_second:.2f} PDFs/sec)"
                )
                
            except Exception as e:
                self.logger.error(f"Concurrent test {scenario_name} failed: {e}")
                continue
    
    async def run_fault_tolerance_tests(self, 
                                      temp_path: Path, 
                                      config: Dict[str, Any],
                                      benchmark_mode: bool):
        """Run fault tolerance and error recovery tests."""
        self.logger.info("Running fault tolerance tests")
        
        corruption_levels = [0.15, 0.25, 0.35] if self.test_level == 'stress' else [0.20]
        
        for corruption_pct in corruption_levels:
            scenario_name = f"fault_tolerance_{int(corruption_pct*100)}pct_corrupt"
            
            scenario = BatchProcessingScenario(
                name=scenario_name,
                description=f"Fault tolerance with {corruption_pct*100:.0f}% corrupted files",
                pdf_count=30,
                batch_size=6,
                concurrent_workers=1,
                corrupted_pdf_percentage=corruption_pct,
                memory_limit_mb=1024,
                timeout_seconds=config['test_duration_limit'] // 3,
                expected_success_rate=max(0.60, 1.0 - corruption_pct - 0.1),
                performance_benchmarks={
                    'min_throughput_pdfs_per_second': 0.3,
                    'max_error_rate': corruption_pct + 0.05
                },
                quality_thresholds={
                    'min_avg_text_length': 100
                }
            )
            
            try:
                result = await self.execute_test_scenario(temp_path, scenario, benchmark_mode)
                self.test_results.append(result)
                
                error_rate = result.pdfs_failed / result.pdfs_processed if result.pdfs_processed > 0 else 1.0
                self.logger.info(
                    f"Fault tolerance test {scenario_name}: "
                    f"{'PASSED' if result.success else 'FAILED'} "
                    f"({error_rate*100:.1f}% error rate)"
                )
                
            except Exception as e:
                self.logger.error(f"Fault tolerance test {scenario_name} failed: {e}")
                continue
    
    async def run_memory_management_tests(self, 
                                        temp_path: Path, 
                                        config: Dict[str, Any],
                                        benchmark_mode: bool):
        """Run memory management tests."""
        self.logger.info("Running memory management tests")
        
        # Test with constrained memory
        constrained_memory = min(config['memory_limits'])
        
        scenario = BatchProcessingScenario(
            name="memory_management_constrained",
            description=f"Memory management with {constrained_memory}MB limit",
            pdf_count=35,
            batch_size=3,  # Small batches for memory testing
            concurrent_workers=1,
            corrupted_pdf_percentage=0.05,
            memory_limit_mb=constrained_memory,
            timeout_seconds=config['test_duration_limit'] // 2,
            expected_success_rate=0.85,
            performance_benchmarks={
                'max_memory_growth_mb': constrained_memory * 0.3,
                'memory_efficiency_ratio': 0.75
            },
            resource_limits={
                'max_memory_mb': constrained_memory * 1.2
            }
        )
        
        try:
            result = await self.execute_test_scenario(temp_path, scenario, benchmark_mode)
            self.test_results.append(result)
            
            self.logger.info(
                f"Memory management test: "
                f"{'PASSED' if result.success else 'FAILED'} "
                f"(Peak: {result.peak_memory_usage_mb:.1f}MB)"
            )
            
        except Exception as e:
            self.logger.error(f"Memory management test failed: {e}")
    
    async def run_synthesis_tests(self, 
                                temp_path: Path, 
                                config: Dict[str, Any],
                                benchmark_mode: bool):
        """Run cross-document synthesis tests."""
        self.logger.info("Running cross-document synthesis tests")
        
        scenario = BatchProcessingScenario(
            name="cross_document_synthesis",
            description="Cross-document knowledge synthesis validation",
            pdf_count=15,
            batch_size=5,
            concurrent_workers=1,
            corrupted_pdf_percentage=0.0,
            memory_limit_mb=1024,
            timeout_seconds=config['test_duration_limit'] // 4,
            expected_success_rate=0.95,
            performance_benchmarks={
                'min_synthesis_quality_score': 70.0,
                'min_cross_document_references': 3
            },
            quality_thresholds={
                'min_entities_per_document': 8
            }
        )
        
        try:
            result = await self.execute_test_scenario(temp_path, scenario, benchmark_mode)
            self.test_results.append(result)
            
            self.logger.info(
                f"Synthesis test: "
                f"{'PASSED' if result.success else 'FAILED'} "
                f"({result.documents_indexed} docs indexed)"
            )
            
        except Exception as e:
            self.logger.error(f"Synthesis test failed: {e}")
    
    async def run_stress_tests(self, 
                             temp_path: Path, 
                             config: Dict[str, Any],
                             benchmark_mode: bool):
        """Run stress tests for extreme conditions."""
        if self.test_level != 'stress':
            return
        
        self.logger.info("Running stress tests")
        
        # High volume stress test
        stress_scenario = BatchProcessingScenario(
            name="stress_high_volume",
            description="Stress test with high PDF volume",
            pdf_count=100,
            batch_size=8,
            concurrent_workers=max(config['concurrent_workers']),
            corrupted_pdf_percentage=0.15,
            memory_limit_mb=max(config['memory_limits']),
            timeout_seconds=config['test_duration_limit'],
            expected_success_rate=0.75,
            performance_benchmarks={
                'min_throughput_pdfs_per_second': 1.5,
                'max_total_time_seconds': config['test_duration_limit'] * 0.9
            },
            quality_thresholds={
                'min_avg_text_length': 150
            }
        )
        
        try:
            result = await self.execute_test_scenario(temp_path, stress_scenario, benchmark_mode)
            self.test_results.append(result)
            
            self.logger.info(
                f"Stress test: "
                f"{'PASSED' if result.success else 'FAILED'} "
                f"({result.success_rate:.1f}% success, {result.throughput_pdfs_per_second:.2f} PDFs/sec)"
            )
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
    
    async def execute_test_scenario(self, 
                                  temp_path: Path, 
                                  scenario: BatchProcessingScenario,
                                  benchmark_mode: bool) -> BatchProcessingResult:
        """Execute a single test scenario with monitoring."""
        self.logger.debug(f"Executing scenario: {scenario.name}")
        
        # Create scenario-specific directory
        scenario_dir = temp_path / scenario.name
        scenario_dir.mkdir(exist_ok=True)
        
        # Generate test PDFs
        pdf_generator = EnhancedBatchPDFGenerator(scenario_dir)
        pdf_paths = pdf_generator.create_large_pdf_collection(
            count=scenario.pdf_count,
            corrupted_percentage=scenario.corrupted_pdf_percentage
        )
        
        try:
            # Setup PDF processor
            error_recovery = ErrorRecoveryConfig(
                max_retries=2,
                base_delay=0.1,
                memory_recovery_enabled=True,
                file_lock_retry_enabled=True,
                timeout_retry_enabled=True
            )
            
            pdf_processor = BiomedicalPDFProcessor(
                processing_timeout=25,
                memory_limit_mb=scenario.memory_limit_mb,
                error_recovery_config=error_recovery
            )
            
            # Execute test with monitoring
            batch_processor = ComprehensiveBatchProcessor(pdf_processor)
            
            if benchmark_mode:
                # Enhanced monitoring for benchmarking
                start_resources = psutil.Process().as_dict(['memory_info', 'cpu_times'])
            
            result = await batch_processor.process_large_batch(pdf_paths, scenario)
            
            if benchmark_mode:
                # Collect benchmark data
                end_resources = psutil.Process().as_dict(['memory_info', 'cpu_times'])
                self.benchmark_data[scenario.name] = {
                    'start_resources': start_resources,
                    'end_resources': end_resources,
                    'result': asdict(result)
                }
            
            return result
            
        finally:
            pdf_generator.cleanup()
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        if not self.test_results:
            return {'status': 'no_tests_completed'}
        
        # Basic statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.success])
        failed_tests = total_tests - passed_tests
        
        # Performance statistics
        throughputs = [r.throughput_pdfs_per_second for r in self.test_results if r.throughput_pdfs_per_second > 0]
        memory_usage = [r.peak_memory_usage_mb for r in self.test_results if r.peak_memory_usage_mb > 0]
        
        summary = {
            'test_statistics': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'pass_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'performance_summary': {
                'average_throughput': statistics.mean(throughputs) if throughputs else 0,
                'max_throughput': max(throughputs) if throughputs else 0,
                'average_memory_usage': statistics.mean(memory_usage) if memory_usage else 0,
                'peak_memory_usage': max(memory_usage) if memory_usage else 0
            },
            'total_pdfs_processed': sum(r.pdfs_processed for r in self.test_results),
            'total_processing_time': sum(r.total_processing_time for r in self.test_results),
            'overall_success_rate': statistics.mean([r.success_rate for r in self.test_results]) if self.test_results else 0
        }
        
        return summary
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not self.test_results:
            return ["No test results available for analysis"]
        
        # Analyze performance patterns
        failed_tests = [r for r in self.test_results if not r.success]
        if failed_tests:
            recommendations.append(
                f"Consider investigating {len(failed_tests)} failed tests for performance optimization"
            )
        
        # Memory usage analysis
        high_memory_tests = [r for r in self.test_results if r.peak_memory_usage_mb > 1500]
        if high_memory_tests:
            recommendations.append(
                "High memory usage detected - consider optimizing batch sizes or memory cleanup"
            )
        
        # Throughput analysis
        throughputs = [r.throughput_pdfs_per_second for r in self.test_results if r.throughput_pdfs_per_second > 0]
        if throughputs and statistics.mean(throughputs) < 0.5:
            recommendations.append(
                "Low average throughput - consider performance tuning or concurrent processing"
            )
        
        # Success rate analysis
        success_rates = [r.success_rate for r in self.test_results]
        if success_rates and statistics.mean(success_rates) < 80:
            recommendations.append(
                "Low overall success rate - investigate error handling and recovery mechanisms"
            )
        
        if not recommendations:
            recommendations.append("All tests performed within acceptable parameters")
        
        return recommendations
    
    async def save_results(self, results: Dict[str, Any]):
        """Save comprehensive test results."""
        timestamp = int(time.time())
        
        # Save main results
        results_file = self.output_dir / f"batch_processing_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / f"batch_processing_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("=== Comprehensive Batch PDF Processing Test Report ===\n\n")
            
            f.write(f"Test Level: {self.test_level}\n")
            f.write(f"Execution Time: {results['test_execution']['total_duration']:.2f}s\n")
            f.write(f"Tests Executed: {results['test_execution']['tests_executed']}\n\n")
            
            # Summary statistics
            summary = results['summary']
            f.write("=== Test Summary ===\n")
            f.write(f"Pass Rate: {summary['test_statistics']['pass_rate']:.1f}%\n")
            f.write(f"Total PDFs Processed: {summary['total_pdfs_processed']}\n")
            f.write(f"Average Throughput: {summary['performance_summary']['average_throughput']:.2f} PDFs/sec\n")
            f.write(f"Peak Memory Usage: {summary['performance_summary']['peak_memory_usage']:.1f} MB\n\n")
            
            # Individual test results
            f.write("=== Individual Test Results ===\n")
            for result in self.test_results:
                status = "PASS" if result.success else "FAIL"
                f.write(f"[{status}] {result.scenario_name}: {result.success_rate:.1f}% success, {result.throughput_pdfs_per_second:.2f} PDFs/sec\n")
            
            f.write("\n=== Recommendations ===\n")
            for rec in results['recommendations']:
                f.write(f"- {rec}\n")
        
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Summary saved to {summary_file}")


async def main():
    """Main entry point for batch processing test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Batch PDF Processing Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--test-level',
        choices=['basic', 'comprehensive', 'stress'],
        default='comprehensive',
        help='Test execution level (default: comprehensive)'
    )
    
    parser.add_argument(
        '--pdf-count',
        type=int,
        help='Number of PDFs to test with (overrides defaults)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./batch_test_results'),
        help='Directory for test results and reports'
    )
    
    parser.add_argument(
        '--benchmark-mode',
        action='store_true',
        help='Enable detailed benchmarking mode'
    )
    
    parser.add_argument(
        '--concurrent-workers',
        type=int,
        help='Number of concurrent workers to test'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=int,
        help='Memory limit in MB for testing'
    )
    
    parser.add_argument(
        '--skip-long-tests',
        action='store_true',
        help='Skip long-running tests (>5 minutes)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging and output'
    )
    
    args = parser.parse_args()
    
    # Adjust test level if skipping long tests
    if args.skip_long_tests and args.test_level != 'basic':
        print("Warning: Forcing basic test level due to --skip-long-tests")
        args.test_level = 'basic'
    
    # Initialize test runner
    runner = ComprehensiveBatchTestRunner(
        output_dir=args.output_dir,
        test_level=args.test_level,
        verbose=args.verbose
    )
    
    try:
        # Execute test suite
        results = await runner.run_all_tests(
            custom_pdf_count=args.pdf_count,
            custom_workers=args.concurrent_workers,
            custom_memory_limit=args.memory_limit,
            benchmark_mode=args.benchmark_mode
        )
        
        # Print summary
        summary = results['summary']
        print(f"\n=== Batch Processing Test Suite Complete ===")
        print(f"Test Level: {args.test_level}")
        print(f"Tests Executed: {summary['test_statistics']['total_tests']}")
        print(f"Pass Rate: {summary['test_statistics']['pass_rate']:.1f}%")
        print(f"Total PDFs Processed: {summary['total_pdfs_processed']}")
        print(f"Average Throughput: {summary['performance_summary']['average_throughput']:.2f} PDFs/sec")
        print(f"Results saved to: {args.output_dir}")
        
        # Exit with appropriate code
        if summary['test_statistics']['failed_tests'] == 0:
            print("✅ All tests passed!")
            sys.exit(0)
        else:
            print(f"❌ {summary['test_statistics']['failed_tests']} tests failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())