#!/usr/bin/env python3
"""
Performance Benchmark Infrastructure Validation Script

This script validates that the performance benchmark infrastructure is properly
set up and can execute benchmark tests for CMO-LIGHTRAG-008-T05.

Features:
- Validates import dependencies
- Tests benchmark infrastructure components
- Runs quick validation benchmarks
- Verifies reporting functionality
- Provides setup diagnostics

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkInfrastructureValidator:
    """Validates performance benchmark infrastructure."""
    
    def __init__(self):
        self.validation_results = {
            'imports': {},
            'fixtures': {},
            'benchmarks': {},
            'integration': {},
            'overall_status': 'unknown'
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run complete infrastructure validation."""
        logger.info("Starting Performance Benchmark Infrastructure Validation")
        logger.info("=" * 60)
        
        # Run validation steps
        self.validate_imports()
        self.validate_fixtures()
        asyncio.run(self.validate_benchmarks())
        self.validate_integration()
        
        # Calculate overall status
        self._calculate_overall_status()
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.validation_results
    
    def validate_imports(self):
        """Validate that all required imports are available."""
        logger.info("1. Validating imports...")
        
        required_imports = [
            ('pytest', 'pytest'),
            ('pytest_asyncio', 'pytest_asyncio'),
            ('numpy', 'numpy as np'),
            ('psutil', 'psutil'),
            ('asyncio', 'asyncio'),
            ('pathlib', 'pathlib'),
            ('dataclasses', 'dataclasses'),
            ('typing', 'typing'),
            ('unittest.mock', 'unittest.mock'),
            ('collections', 'collections'),
            ('json', 'json'),
            ('time', 'time'),
            ('logging', 'logging')
        ]
        
        import_results = {}
        
        for package_name, import_statement in required_imports:
            try:
                exec(f"import {import_statement}")
                import_results[package_name] = {'status': 'success', 'available': True}
                logger.info(f"  ✓ {package_name}: Available")
            except ImportError as e:
                import_results[package_name] = {'status': 'failed', 'error': str(e), 'available': False}
                logger.error(f"  ✗ {package_name}: Missing - {e}")
        
        # Test performance test fixtures
        try:
            from performance_test_fixtures import (
                PerformanceMetrics,
                LoadTestScenario,
                ResourceMonitor,
                PerformanceTestExecutor,
                LoadTestScenarioGenerator,
                MockOperationGenerator
            )
            import_results['performance_test_fixtures'] = {'status': 'success', 'available': True}
            logger.info("  ✓ performance_test_fixtures: Available")
        except ImportError as e:
            import_results['performance_test_fixtures'] = {'status': 'failed', 'error': str(e), 'available': False}
            logger.error(f"  ✗ performance_test_fixtures: Missing - {e}")
        
        # Test biomedical fixtures
        try:
            from biomedical_test_fixtures import (
                ClinicalMetabolomicsDataGenerator,
                MetaboliteData,
                ClinicalStudyData
            )
            import_results['biomedical_test_fixtures'] = {'status': 'success', 'available': True}
            logger.info("  ✓ biomedical_test_fixtures: Available")
        except ImportError as e:
            import_results['biomedical_test_fixtures'] = {'status': 'failed', 'error': str(e), 'available': False}
            logger.error(f"  ✗ biomedical_test_fixtures: Missing - {e}")
        
        # Test benchmark suite
        try:
            from test_performance_benchmarks import (
                PerformanceBenchmarkSuite,
                BenchmarkTarget,
                BenchmarkReportGenerator
            )
            import_results['test_performance_benchmarks'] = {'status': 'success', 'available': True}
            logger.info("  ✓ test_performance_benchmarks: Available")
        except ImportError as e:
            import_results['test_performance_benchmarks'] = {'status': 'failed', 'error': str(e), 'available': False}
            logger.error(f"  ✗ test_performance_benchmarks: Missing - {e}")
        
        self.validation_results['imports'] = import_results
        
        # Summary
        total_imports = len(import_results)
        successful_imports = sum(1 for result in import_results.values() if result['status'] == 'success')
        logger.info(f"  Imports: {successful_imports}/{total_imports} successful")
    
    def validate_fixtures(self):
        """Validate that test fixtures are working correctly."""
        logger.info("\n2. Validating test fixtures...")
        
        fixture_results = {}
        
        # Test performance test executor
        try:
            from performance_test_fixtures import PerformanceTestExecutor, ResourceMonitor
            
            executor = PerformanceTestExecutor()
            monitor = ResourceMonitor(sampling_interval=0.1)
            
            fixture_results['performance_executor'] = {'status': 'success', 'available': True}
            fixture_results['resource_monitor'] = {'status': 'success', 'available': True}
            logger.info("  ✓ PerformanceTestExecutor: Working")
            logger.info("  ✓ ResourceMonitor: Working")
        except Exception as e:
            fixture_results['performance_executor'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ PerformanceTestExecutor: Failed - {e}")
        
        # Test load test scenarios
        try:
            from performance_test_fixtures import LoadTestScenarioGenerator
            
            generator = LoadTestScenarioGenerator()
            baseline_scenario = generator.create_baseline_scenario()
            
            # Validate scenario structure
            assert baseline_scenario.scenario_name == "baseline_performance"
            assert baseline_scenario.duration_seconds > 0
            assert baseline_scenario.concurrent_users > 0
            assert len(baseline_scenario.operation_types) > 0
            
            fixture_results['load_test_scenarios'] = {'status': 'success', 'available': True}
            logger.info("  ✓ LoadTestScenarios: Working")
        except Exception as e:
            fixture_results['load_test_scenarios'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ LoadTestScenarios: Failed - {e}")
        
        # Test mock operation generator
        try:
            from performance_test_fixtures import MockOperationGenerator
            
            op_gen = MockOperationGenerator()
            query_data = op_gen.generate_query_data('medium_query')
            
            # Validate generated data
            assert 'query_text' in query_data
            assert 'operation_type' in query_data
            assert 'complexity_score' in query_data
            assert len(query_data['query_text']) > 0
            
            fixture_results['mock_operation_generator'] = {'status': 'success', 'available': True}
            logger.info("  ✓ MockOperationGenerator: Working")
        except Exception as e:
            fixture_results['mock_operation_generator'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ MockOperationGenerator: Failed - {e}")
        
        # Test biomedical data generator
        try:
            from biomedical_test_fixtures import ClinicalMetabolomicsDataGenerator
            
            bio_gen = ClinicalMetabolomicsDataGenerator()
            metabolite_data = bio_gen.generate_metabolite_data()
            
            # Validate generated data
            assert hasattr(metabolite_data, 'name')
            assert hasattr(metabolite_data, 'concentration_range')
            
            fixture_results['biomedical_data_generator'] = {'status': 'success', 'available': True}
            logger.info("  ✓ BiomedicalDataGenerator: Working")
        except Exception as e:
            fixture_results['biomedical_data_generator'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ BiomedicalDataGenerator: Failed - {e}")
        
        self.validation_results['fixtures'] = fixture_results
        
        # Summary
        total_fixtures = len(fixture_results)
        successful_fixtures = sum(1 for result in fixture_results.values() if result['status'] == 'success')
        logger.info(f"  Fixtures: {successful_fixtures}/{total_fixtures} working")
    
    async def validate_benchmarks(self):
        """Validate that benchmark tests can execute."""
        logger.info("\n3. Validating benchmark execution...")
        
        benchmark_results = {}
        
        # Test basic mock operation
        try:
            from performance_test_fixtures import mock_clinical_query_operation
            
            test_data = {
                'query_text': 'Test query for validation',
                'operation_type': 'simple_query',
                'complexity_score': 0.3,
                'expected_response_length': 200
            }
            
            start_time = time.time()
            result = await mock_clinical_query_operation(test_data)
            end_time = time.time()
            
            # Validate result structure
            assert 'query' in result
            assert 'response' in result
            assert 'processing_time' in result
            assert end_time - start_time >= result['processing_time']
            
            benchmark_results['mock_operation'] = {
                'status': 'success',
                'execution_time': end_time - start_time,
                'available': True
            }
            logger.info(f"  ✓ MockOperation: Executed in {(end_time - start_time)*1000:.1f}ms")
        except Exception as e:
            benchmark_results['mock_operation'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ MockOperation: Failed - {e}")
        
        # Test benchmark suite initialization
        try:
            from test_performance_benchmarks import PerformanceBenchmarkSuite
            
            suite = PerformanceBenchmarkSuite()
            
            # Validate benchmark targets
            assert len(suite.benchmark_targets) > 0
            assert 'simple_query_benchmark' in suite.benchmark_targets
            
            # Test target evaluation
            target = suite.benchmark_targets['simple_query_benchmark']
            assert target.max_response_time_ms > 0
            assert target.min_throughput_ops_per_sec > 0
            
            benchmark_results['benchmark_suite'] = {'status': 'success', 'available': True}
            logger.info("  ✓ BenchmarkSuite: Initialized")
        except Exception as e:
            benchmark_results['benchmark_suite'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ BenchmarkSuite: Failed - {e}")
        
        # Test quick benchmark execution
        try:
            from test_performance_benchmarks import PerformanceBenchmarkSuite
            
            suite = PerformanceBenchmarkSuite()
            
            # Run very quick benchmark (reduced duration for validation)
            start_time = time.time()
            result = await suite._run_simple_query_benchmark()
            end_time = time.time()
            
            # Validate benchmark result structure
            assert 'benchmark_type' in result
            assert 'metrics' in result
            assert 'evaluation' in result
            assert 'status' in result
            
            benchmark_results['quick_benchmark_execution'] = {
                'status': 'success',
                'execution_time': end_time - start_time,
                'benchmark_status': result['status'],
                'available': True
            }
            logger.info(f"  ✓ QuickBenchmark: Executed in {(end_time - start_time):.1f}s, Status: {result['status']}")
        except Exception as e:
            benchmark_results['quick_benchmark_execution'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ QuickBenchmark: Failed - {e}")
        
        self.validation_results['benchmarks'] = benchmark_results
        
        # Summary
        total_benchmarks = len(benchmark_results)
        successful_benchmarks = sum(1 for result in benchmark_results.values() if result['status'] == 'success')
        logger.info(f"  Benchmarks: {successful_benchmarks}/{total_benchmarks} working")
    
    def validate_integration(self):
        """Validate integration with pytest and reporting."""
        logger.info("\n4. Validating integration...")
        
        integration_results = {}
        
        # Test pytest configuration
        try:
            pytest_ini_path = Path(__file__).parent / "pytest.ini"
            if pytest_ini_path.exists():
                with open(pytest_ini_path, 'r') as f:
                    config_content = f.read()
                    assert 'performance' in config_content
                    assert 'asyncio-mode' in config_content
                
                integration_results['pytest_config'] = {'status': 'success', 'available': True}
                logger.info("  ✓ pytest.ini: Configured for performance tests")
            else:
                integration_results['pytest_config'] = {'status': 'missing', 'error': 'pytest.ini not found'}
                logger.warning("  ! pytest.ini: Not found")
        except Exception as e:
            integration_results['pytest_config'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ pytest.ini: Failed - {e}")
        
        # Test conftest.py
        try:
            conftest_path = Path(__file__).parent / "conftest.py"
            if conftest_path.exists():
                # Check for performance fixtures
                with open(conftest_path, 'r') as f:
                    conftest_content = f.read()
                    has_async_fixtures = 'async' in conftest_content
                    has_performance_fixtures = 'performance' in conftest_content
                
                integration_results['conftest'] = {
                    'status': 'success',
                    'available': True,
                    'has_async_fixtures': has_async_fixtures,
                    'has_performance_fixtures': has_performance_fixtures
                }
                logger.info("  ✓ conftest.py: Available with fixtures")
            else:
                integration_results['conftest'] = {'status': 'missing', 'error': 'conftest.py not found'}
                logger.warning("  ! conftest.py: Not found")
        except Exception as e:
            integration_results['conftest'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ conftest.py: Failed - {e}")
        
        # Test report generation
        try:
            from test_performance_benchmarks import BenchmarkReportGenerator
            
            # Create minimal test results
            test_results = {
                'benchmarks': [
                    {
                        'benchmark_type': 'validation_test',
                        'status': 'passed',
                        'metrics': {
                            'average_latency_ms': 1000.0,
                            'throughput_ops_per_sec': 2.0,
                            'memory_usage_mb': 300.0,
                            'error_rate_percent': 1.0
                        }
                    }
                ],
                'summary': {
                    'total_benchmarks': 1,
                    'passed_benchmarks': 1,
                    'failed_benchmarks': 0,
                    'success_rate_percent': 100.0,
                    'overall_grade': 'Excellent'
                }
            }
            
            # Test report generation in temporary location
            temp_output_dir = Path("/tmp/benchmark_validation_reports")
            temp_output_dir.mkdir(exist_ok=True)
            
            report_file = BenchmarkReportGenerator.generate_benchmark_report(
                test_results, temp_output_dir
            )
            
            # Validate report was created
            assert report_file.exists()
            assert report_file.suffix == '.json'
            
            integration_results['report_generation'] = {
                'status': 'success',
                'available': True,
                'report_file': str(report_file)
            }
            logger.info(f"  ✓ ReportGeneration: Working, created {report_file.name}")
            
            # Cleanup
            try:
                report_file.unlink()
                summary_file = report_file.with_name(report_file.stem + "_summary.txt")
                if summary_file.exists():
                    summary_file.unlink()
            except:
                pass  # Ignore cleanup failures
                
        except Exception as e:
            integration_results['report_generation'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ ReportGeneration: Failed - {e}")
        
        # Test runner script
        try:
            runner_path = Path(__file__).parent / "run_performance_benchmarks.py"
            if runner_path.exists():
                integration_results['runner_script'] = {'status': 'success', 'available': True}
                logger.info("  ✓ RunnerScript: Available")
            else:
                integration_results['runner_script'] = {'status': 'missing', 'error': 'Runner script not found'}
                logger.warning("  ! RunnerScript: Not found")
        except Exception as e:
            integration_results['runner_script'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"  ✗ RunnerScript: Failed - {e}")
        
        self.validation_results['integration'] = integration_results
        
        # Summary
        total_integrations = len(integration_results)
        successful_integrations = sum(1 for result in integration_results.values() if result['status'] == 'success')
        logger.info(f"  Integration: {successful_integrations}/{total_integrations} working")
    
    def _calculate_overall_status(self):
        """Calculate overall infrastructure status."""
        categories = ['imports', 'fixtures', 'benchmarks', 'integration']
        category_scores = []
        
        for category in categories:
            results = self.validation_results.get(category, {})
            if not results:
                category_scores.append(0.0)
                continue
            
            total_items = len(results)
            successful_items = sum(1 for result in results.values() if result.get('status') == 'success')
            score = successful_items / total_items if total_items > 0 else 0.0
            category_scores.append(score)
        
        overall_score = sum(category_scores) / len(category_scores) if category_scores else 0.0
        
        # Determine status
        if overall_score >= 0.9:
            status = 'excellent'
        elif overall_score >= 0.8:
            status = 'good'
        elif overall_score >= 0.7:
            status = 'satisfactory'
        elif overall_score >= 0.5:
            status = 'needs_improvement'
        else:
            status = 'poor'
        
        self.validation_results['overall_status'] = status
        self.validation_results['overall_score'] = overall_score
    
    def _generate_summary_report(self):
        """Generate validation summary report."""
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE BENCHMARK INFRASTRUCTURE VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        overall_score = self.validation_results.get('overall_score', 0.0)
        overall_status = self.validation_results.get('overall_status', 'unknown')
        
        logger.info(f"Overall Status: {overall_status.upper()}")
        logger.info(f"Overall Score: {overall_score:.1%}")
        
        # Category breakdown
        categories = [
            ('imports', 'Import Dependencies'),
            ('fixtures', 'Test Fixtures'),
            ('benchmarks', 'Benchmark Execution'),
            ('integration', 'Integration Components')
        ]
        
        for category_key, category_name in categories:
            results = self.validation_results.get(category_key, {})
            if results:
                total = len(results)
                successful = sum(1 for result in results.values() if result.get('status') == 'success')
                score = successful / total if total > 0 else 0.0
                logger.info(f"{category_name}: {successful}/{total} ({score:.1%})")
        
        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            logger.info("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")
        
        # Ready status
        ready_for_benchmarks = overall_score >= 0.8
        logger.info(f"\nReady for Performance Benchmarks: {'YES' if ready_for_benchmarks else 'NO'}")
        
        if ready_for_benchmarks:
            logger.info("\n✓ Infrastructure is ready for CMO-LIGHTRAG-008-T05 performance benchmarks")
            logger.info("  You can run: python run_performance_benchmarks.py --mode quick")
        else:
            logger.warning("\n! Infrastructure needs improvements before running full benchmarks")
            logger.warning("  Address the issues above and re-run validation")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for missing imports
        import_results = self.validation_results.get('imports', {})
        missing_imports = [name for name, result in import_results.items() 
                         if result.get('status') != 'success']
        
        if missing_imports:
            recommendations.append(
                f"Install missing dependencies: {', '.join(missing_imports)}"
            )
        
        # Check for fixture issues
        fixture_results = self.validation_results.get('fixtures', {})
        failed_fixtures = [name for name, result in fixture_results.items() 
                         if result.get('status') != 'success']
        
        if failed_fixtures:
            recommendations.append(
                f"Fix failing test fixtures: {', '.join(failed_fixtures)}"
            )
        
        # Check for benchmark execution issues
        benchmark_results = self.validation_results.get('benchmarks', {})
        failed_benchmarks = [name for name, result in benchmark_results.items() 
                           if result.get('status') != 'success']
        
        if failed_benchmarks:
            recommendations.append(
                f"Resolve benchmark execution issues: {', '.join(failed_benchmarks)}"
            )
        
        # Check for integration issues
        integration_results = self.validation_results.get('integration', {})
        missing_integration = [name for name, result in integration_results.items() 
                             if result.get('status') == 'missing']
        
        if missing_integration:
            recommendations.append(
                f"Set up missing integration components: {', '.join(missing_integration)}"
            )
        
        # General recommendations
        overall_score = self.validation_results.get('overall_score', 0.0)
        if overall_score >= 0.8:
            recommendations.append("Infrastructure is ready - consider running full benchmark suite")
        elif overall_score >= 0.6:
            recommendations.append("Infrastructure is mostly ready - run quick benchmarks first")
        else:
            recommendations.append("Infrastructure needs significant improvements before benchmarking")
        
        return recommendations


def main():
    """Main validation function."""
    print("Performance Benchmark Infrastructure Validation")
    print("Task: CMO-LIGHTRAG-008-T05")
    print("=" * 60)
    
    validator = BenchmarkInfrastructureValidator()
    results = validator.validate_all()
    
    # Save validation results
    output_dir = Path("lightrag_integration/tests/performance_test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"benchmark_infrastructure_validation_{timestamp}.json"
    
    try:
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nValidation results saved to: {results_file}")
    except Exception as e:
        print(f"\nFailed to save validation results: {e}")
    
    # Return appropriate exit code
    overall_score = results.get('overall_score', 0.0)
    success = overall_score >= 0.8
    
    print(f"\nValidation {'PASSED' if success else 'FAILED'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())