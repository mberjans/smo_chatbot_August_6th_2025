#!/usr/bin/env python3
"""
Performance Test Infrastructure Verification Script.

This script verifies that all components of the comprehensive performance
and quality testing infrastructure are properly configured and functional.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import sys
import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def verify_imports() -> Dict[str, bool]:
    """Verify all required imports are available."""
    
    import_results = {}
    
    # Core test components
    try:
        from test_comprehensive_query_performance_quality import (
            TestQueryPerformanceBenchmarks,
            TestResponseQualityValidation,
            TestScalabilityAndStress,
            ResponseQualityAssessor,
            PerformanceBenchmark,
            ResponseQualityMetrics
        )
        import_results['main_test_suite'] = True
    except Exception as e:
        print(f"Failed to import main test suite: {e}")
        import_results['main_test_suite'] = False
    
    # Performance analysis utilities
    try:
        from performance_analysis_utilities import (
            PerformanceReportGenerator,
            BenchmarkAnalyzer,
            PerformanceReport
        )
        import_results['analysis_utilities'] = True
    except Exception as e:
        print(f"Failed to import analysis utilities: {e}")
        import_results['analysis_utilities'] = False
    
    # Test runner
    try:
        from run_comprehensive_performance_quality_tests import (
            PerformanceTestRunner
        )
        import_results['test_runner'] = True
    except Exception as e:
        print(f"Failed to import test runner: {e}")
        import_results['test_runner'] = False
    
    # Test fixtures
    try:
        from performance_test_fixtures import (
            PerformanceMetrics,
            LoadTestScenario,
            ResourceMonitor,
            PerformanceTestExecutor
        )
        import_results['performance_fixtures'] = True
    except Exception as e:
        print(f"Failed to import performance fixtures: {e}")
        import_results['performance_fixtures'] = False
    
    try:
        from biomedical_test_fixtures import (
            MetaboliteData,
            ClinicalStudyData
        )
        import_results['biomedical_fixtures'] = True
    except Exception as e:
        print(f"Failed to import biomedical fixtures: {e}")
        import_results['biomedical_fixtures'] = False
    
    # Required external packages
    external_packages = {
        'pytest': 'pytest',
        'numpy': 'numpy',
        'psutil': 'psutil',
        'asyncio': 'asyncio'
    }
    
    for package_name, import_name in external_packages.items():
        try:
            __import__(import_name)
            import_results[f'package_{package_name}'] = True
        except ImportError:
            print(f"Missing required package: {package_name}")
            import_results[f'package_{package_name}'] = False
    
    return import_results


def verify_fixtures():
    """Verify test fixtures are functional."""
    
    fixture_results = {}
    
    try:
        # Test performance fixtures
        from performance_test_fixtures import (
            LoadTestScenarioGenerator,
            MockOperationGenerator,
            ResourceMonitor
        )
        
        # Create scenario generator
        generator = LoadTestScenarioGenerator()
        baseline_scenario = generator.create_baseline_scenario()
        
        assert baseline_scenario.scenario_name == "baseline_performance"
        assert baseline_scenario.target_operations_per_second == 1.0
        
        # Create mock operation generator
        op_generator = MockOperationGenerator()
        query_data = op_generator.generate_query_data('simple_query')
        
        assert 'query_text' in query_data
        assert 'operation_type' in query_data
        
        # Test resource monitor
        monitor = ResourceMonitor(sampling_interval=0.1)
        assert not monitor.monitoring
        
        fixture_results['performance_fixtures'] = True
        
    except Exception as e:
        print(f"Performance fixtures verification failed: {e}")
        fixture_results['performance_fixtures'] = False
    
    try:
        # Test quality assessor
        from test_comprehensive_query_performance_quality import ResponseQualityAssessor
        
        assessor = ResponseQualityAssessor()
        assert hasattr(assessor, 'biomedical_keywords')
        assert 'metabolomics_core' in assessor.biomedical_keywords
        
        fixture_results['quality_assessor'] = True
        
    except Exception as e:
        print(f"Quality assessor verification failed: {e}")
        fixture_results['quality_assessor'] = False
    
    return fixture_results


async def verify_async_functionality():
    """Verify async test functionality."""
    
    async_results = {}
    
    try:
        # Test basic async operation
        from test_comprehensive_query_performance_quality import ResponseQualityAssessor
        
        assessor = ResponseQualityAssessor()
        
        # Test quality assessment
        quality_metrics = await assessor.assess_response_quality(
            query="What is clinical metabolomics?",
            response="Clinical metabolomics involves the study of metabolites in clinical samples.",
            source_documents=[],
            expected_concepts=['metabolomics', 'clinical']
        )
        
        assert hasattr(quality_metrics, 'overall_quality_score')
        assert 0 <= quality_metrics.overall_quality_score <= 100
        
        async_results['quality_assessment'] = True
        
    except Exception as e:
        print(f"Async quality assessment failed: {e}")
        async_results['quality_assessment'] = False
    
    try:
        # Test performance test executor
        from performance_test_fixtures import (
            PerformanceTestExecutor,
            LoadTestScenarioGenerator,
            mock_clinical_query_operation,
            MockOperationGenerator
        )
        
        executor = PerformanceTestExecutor()
        generator = LoadTestScenarioGenerator()
        scenario = generator.create_baseline_scenario()
        scenario.duration_seconds = 2.0  # Short test
        
        op_generator = MockOperationGenerator()
        
        # Run quick load test
        metrics = await executor.execute_load_test(
            scenario=scenario,
            operation_func=mock_clinical_query_operation,
            operation_data_generator=op_generator.generate_query_data
        )
        
        assert hasattr(metrics, 'throughput_ops_per_sec')
        assert metrics.throughput_ops_per_sec >= 0
        
        async_results['load_testing'] = True
        
    except Exception as e:
        print(f"Async load testing failed: {e}")
        traceback.print_exc()
        async_results['load_testing'] = False
    
    return async_results


def verify_report_generation():
    """Verify report generation functionality."""
    
    report_results = {}
    
    try:
        from performance_analysis_utilities import PerformanceReportGenerator
        from test_comprehensive_query_performance_quality import (
            PerformanceBenchmark,
            ResponseQualityMetrics
        )
        
        # Create mock test results
        mock_benchmark = PerformanceBenchmark(
            query_type="test",
            benchmark_name="verification_test",
            target_response_time_ms=5000,
            actual_response_time_ms=4500,
            target_throughput_ops_per_sec=1.0,
            actual_throughput_ops_per_sec=1.1,
            target_memory_usage_mb=500,
            actual_memory_usage_mb=450,
            target_error_rate_percent=5.0,
            actual_error_rate_percent=2.0,
            meets_performance_targets=True,
            performance_ratio=1.1
        )
        
        mock_quality = ResponseQualityMetrics(
            relevance_score=85.0,
            accuracy_score=88.0,
            completeness_score=82.0,
            clarity_score=90.0,
            biomedical_terminology_score=78.0,
            source_citation_score=75.0,
            consistency_score=87.0,
            factual_accuracy_score=89.0,
            hallucination_score=92.0,
            overall_quality_score=85.5
        )
        
        # Generate report
        report_generator = PerformanceReportGenerator()
        report = report_generator.generate_comprehensive_report(
            benchmark_results=[mock_benchmark],
            quality_results=[mock_quality],
            scalability_results=[],
            test_suite_name="Infrastructure_Verification"
        )
        
        assert hasattr(report, 'report_id')
        assert hasattr(report, 'overall_performance_grade')
        assert hasattr(report, 'overall_quality_grade')
        
        report_results['report_generation'] = True
        
    except Exception as e:
        print(f"Report generation verification failed: {e}")
        traceback.print_exc()
        report_results['report_generation'] = False
    
    return report_results


def verify_test_runner():
    """Verify test runner functionality."""
    
    runner_results = {}
    
    try:
        from run_comprehensive_performance_quality_tests import PerformanceTestRunner
        
        # Create test runner
        test_runner = PerformanceTestRunner()
        
        assert hasattr(test_runner, 'quality_assessor')
        assert hasattr(test_runner, 'report_generator')
        assert hasattr(test_runner, 'benchmark_results')
        
        runner_results['test_runner_creation'] = True
        
    except Exception as e:
        print(f"Test runner verification failed: {e}")
        runner_results['test_runner_creation'] = False
    
    return runner_results


async def main():
    """Main verification function."""
    
    print("="*60)
    print("PERFORMANCE TEST INFRASTRUCTURE VERIFICATION")
    print("="*60)
    
    verification_results = {
        'imports': verify_imports(),
        'fixtures': verify_fixtures(),
        'async_functionality': await verify_async_functionality(),
        'report_generation': verify_report_generation(),
        'test_runner': verify_test_runner()
    }
    
    print("\nVERIFICATION RESULTS:")
    print("-" * 40)
    
    total_checks = 0
    passed_checks = 0
    
    for category, results in verification_results.items():
        print(f"\n{category.upper()}:")
        for check_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {check_name}: {status}")
            total_checks += 1
            if passed:
                passed_checks += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("✓ All verification checks PASSED - Infrastructure is ready!")
        return_code = 0
    elif passed_checks >= total_checks * 0.8:
        print("⚠ Most checks passed - Minor issues detected")
        return_code = 1
    else:
        print("✗ Multiple verification checks FAILED - Review issues above")
        return_code = 2
    
    print(f"{'='*60}")
    
    if return_code == 0:
        print("\nNext steps:")
        print("1. Run quick test: python run_comprehensive_performance_quality_tests.py --mode quick")
        print("2. Run full suite: python run_comprehensive_performance_quality_tests.py --mode comprehensive")
        print("3. Review documentation: COMPREHENSIVE_PERFORMANCE_QUALITY_TESTING_GUIDE.md")
    
    return return_code


if __name__ == "__main__":
    try:
        return_code = asyncio.run(main())
        sys.exit(return_code)
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Verification failed with error: {e}")
        traceback.print_exc()
        sys.exit(3)