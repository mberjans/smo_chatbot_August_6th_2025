#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for performance benchmarking tests.

This module provides shared pytest fixtures, configuration, and utilities
used across all test modules in the performance benchmarking test suite.

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import asyncio
import tempfile
import shutil
import time
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional, Generator
from dataclasses import asdict

# Import modules for fixture creation
try:
    from quality_performance_benchmarks import QualityValidationMetrics
    from performance_correlation_engine import PerformanceCorrelationMetrics
    from quality_aware_metrics_logger import QualityAPIMetric
    from reporting.quality_performance_reporter import PerformanceReportConfiguration
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp(prefix="perfbench_test_"))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_time():
    """Mock time.time() for consistent test results."""
    base_time = 1691404800.0  # Fixed timestamp for consistency
    
    def mock_time_func():
        return base_time
    
    return mock_time_func


@pytest.fixture
def sample_quality_metrics() -> List[Dict[str, Any]]:
    """Create sample quality validation metrics for testing."""
    base_time = time.time()
    
    metrics_data = []
    for i in range(5):
        metrics_data.append({
            'scenario_name': f'test_scenario_{i}',
            'operations_count': 10 + i,
            'average_latency_ms': 1000.0 + (i * 200),
            'throughput_ops_per_sec': 5.0 - (i * 0.2),
            'validation_accuracy_rate': 85.0 + (i * 2),
            'claims_extracted_count': 20 + (i * 5),
            'claims_validated_count': 18 + (i * 4),
            'error_rate_percent': 2.0 - (i * 0.3),
            'avg_validation_confidence': 80.0 + (i * 3),
            'claim_extraction_time_ms': 200.0 + (i * 50),
            'factual_validation_time_ms': 500.0 + (i * 100),
            'relevance_scoring_time_ms': 150.0 + (i * 30),
            'integrated_workflow_time_ms': 850.0 + (i * 180),
            'peak_validation_memory_mb': 400.0 + (i * 50),
            'avg_validation_cpu_percent': 40.0 + (i * 5),
            'start_time': base_time - (i * 3600),
            'duration_seconds': 100.0 + (i * 20),
            'timestamp': base_time - (i * 1800)
        })
    
    return metrics_data


@pytest.fixture
def sample_api_metrics() -> List[Dict[str, Any]]:
    """Create sample API metrics for testing."""
    base_time = time.time()
    
    api_metrics = []
    for i in range(4):
        api_metrics.append({
            'timestamp': base_time - (i * 1800),
            'endpoint': f'/api/test_endpoint_{i}',
            'method': 'POST',
            'response_time_ms': 800.0 + (i * 150),
            'status_code': 200,
            'cost_usd': 0.05 + (i * 0.02),
            'quality_validation_type': ['factual_accuracy', 'relevance_scoring', 'integrated_workflow', 'claim_extraction'][i],
            'validation_accuracy_score': 85.0 + (i * 3),
            'confidence_score': 0.8 + (i * 0.04),
            'claims_processed': 4 + i,
            'validation_duration_ms': 400.0 + (i * 80),
            'quality_validation_cost_usd': 0.03 + (i * 0.015),
            'quality_validation_cost_percentage': 60.0 + (i * 5),
            'error_occurred': False,
            'error_message': None
        })
    
    return api_metrics


@pytest.fixture
def sample_correlation_metrics() -> Dict[str, Any]:
    """Create sample correlation metrics for testing."""
    return {
        'quality_strictness_level': 'medium',
        'confidence_threshold': 0.8,
        'accuracy_requirement': 85.0,
        'validation_strictness_level': 1.0,
        'max_claims_per_response': 10,
        'quality_performance_correlations': {
            'quality_vs_latency': 0.75,
            'accuracy_vs_throughput': -0.68,
            'validation_vs_cost': 0.52,
            'confidence_vs_memory': 0.43,
            'claims_vs_cpu': 0.61,
            'strictness_vs_accuracy': 0.79
        },
        'resource_quality_correlations': {
            'memory_vs_quality': 0.45,
            'cpu_vs_validation_speed': 0.67,
            'cost_vs_accuracy': 0.58
        },
        'sample_size': 100,
        'analysis_timestamp': time.time(),
        'correlation_confidence': 0.95
    }


@pytest.fixture
def mock_quality_validation_components():
    """Mock quality validation components for testing."""
    # Mock extractor
    mock_extractor = Mock()
    mock_extractor.extract_claims = AsyncMock(return_value=[
        "Claim 1: Test factual claim",
        "Claim 2: Another test claim",
        "Claim 3: Third validation claim"
    ])
    
    # Mock validator
    mock_validator = Mock()
    mock_validation_results = [
        Mock(confidence_score=0.85, supported=True, evidence="Test evidence 1"),
        Mock(confidence_score=0.78, supported=True, evidence="Test evidence 2"),
        Mock(confidence_score=0.92, supported=False, evidence="Test evidence 3")
    ]
    mock_validator.validate_claims = AsyncMock(return_value=mock_validation_results)
    
    # Mock scorer
    mock_scorer = Mock()
    mock_score_result = Mock(
        overall_score=87.5,
        confidence_score=0.82,
        relevance_breakdown={'clinical': 0.9, 'technical': 0.85}
    )
    mock_scorer.score_relevance = AsyncMock(return_value=mock_score_result)
    
    # Mock integrated workflow
    mock_workflow = Mock()
    mock_workflow_result = {
        'overall_score': 84.2,
        'components_completed': 3,
        'quality_assessment': {
            'extraction_quality': 88.0,
            'validation_quality': 82.0,
            'scoring_quality': 86.0
        }
    }
    mock_workflow.assess_quality = AsyncMock(return_value=mock_workflow_result)
    
    return {
        'extractor': mock_extractor,
        'validator': mock_validator,
        'scorer': mock_scorer,
        'workflow': mock_workflow,
        'validation_results': mock_validation_results,
        'score_result': mock_score_result,
        'workflow_result': mock_workflow_result
    }


@pytest.fixture
def mock_http_responses():
    """Mock HTTP responses for API testing."""
    success_response = Mock()
    success_response.status_code = 200
    success_response.headers = {'content-type': 'application/json'}
    success_response.json.return_value = {
        'validation_result': {
            'accuracy_score': 88.5,
            'confidence_score': 0.84,
            'claims_processed': 6,
            'validation_details': {
                'factual_accuracy': 0.87,
                'relevance_score': 0.91,
                'confidence_breakdown': {'high': 4, 'medium': 2, 'low': 0}
            }
        },
        'cost_info': {
            'total_cost': 0.078,
            'validation_cost': 0.045
        },
        'processing_time_ms': 1200
    }
    success_response.text = json.dumps(success_response.json.return_value)
    
    error_response = Mock()
    error_response.status_code = 500
    error_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    error_response.text = "Internal Server Error"
    
    timeout_response = Mock()
    timeout_response.status_code = 408
    timeout_response.text = "Request Timeout"
    
    return {
        'success': success_response,
        'error': error_response,
        'timeout': timeout_response
    }


@pytest.fixture
def sample_performance_thresholds() -> Dict[str, float]:
    """Create sample performance thresholds for testing."""
    return {
        'response_time_ms_threshold': 2000.0,
        'throughput_ops_per_sec_threshold': 5.0,
        'accuracy_threshold': 85.0,
        'cost_per_operation_threshold': 0.01,
        'memory_usage_mb_threshold': 1000.0,
        'error_rate_threshold': 5.0,
        'confidence_threshold': 0.8,
        'cpu_usage_threshold': 80.0
    }


@pytest.fixture
def report_configuration(temp_dir) -> Any:
    """Create a test report configuration."""
    if not MODULES_AVAILABLE:
        # Return mock configuration if modules not available
        mock_config = Mock()
        mock_config.report_name = "Test Performance Report"
        mock_config.analysis_period_hours = 24
        mock_config.include_executive_summary = True
        mock_config.include_detailed_analysis = True
        mock_config.include_recommendations = True
        mock_config.generate_charts = False  # Skip for testing
        mock_config.output_directory = temp_dir
        mock_config.performance_thresholds = {
            'response_time_ms_threshold': 2000,
            'accuracy_threshold': 85.0,
            'error_rate_threshold': 5.0
        }
        return mock_config
    
    from reporting.quality_performance_reporter import PerformanceReportConfiguration, ReportFormat
    
    return PerformanceReportConfiguration(
        report_name="Test Performance Report",
        report_description="Test report for unit testing",
        analysis_period_hours=24,
        minimum_sample_size=5,
        include_executive_summary=True,
        include_detailed_analysis=True,
        include_recommendations=True,
        generate_charts=False,  # Skip chart generation for faster testing
        output_formats=[ReportFormat.JSON, ReportFormat.HTML],
        output_directory=temp_dir
    )


@pytest.fixture
def mock_resource_monitor():
    """Mock resource monitoring functionality."""
    mock_monitor = Mock()
    
    # Mock resource snapshots
    mock_snapshots = [
        Mock(
            timestamp=time.time() - 300,
            memory_mb=450.0,
            cpu_percent=55.0,
            disk_io_mb=12.5,
            network_io_mb=3.2
        ),
        Mock(
            timestamp=time.time() - 200,
            memory_mb=520.0,
            cpu_percent=62.0,
            disk_io_mb=15.8,
            network_io_mb=4.1
        ),
        Mock(
            timestamp=time.time() - 100,
            memory_mb=480.0,
            cpu_percent=58.0,
            disk_io_mb=11.3,
            network_io_mb=2.9
        )
    ]
    
    mock_monitor.start_monitoring = Mock()
    mock_monitor.stop_monitoring = Mock(return_value=mock_snapshots)
    mock_monitor.get_current_usage = Mock(return_value=mock_snapshots[-1])
    
    return mock_monitor


@pytest.fixture
def sample_test_scenarios() -> List[Dict[str, Any]]:
    """Create sample test scenarios for benchmarking."""
    return [
        {
            'scenario_name': 'factual_validation_test',
            'total_operations': 10,
            'concurrent_operations': 2,
            'ramp_up_time': 1.0,
            'validation_type': 'factual_accuracy',
            'expected_accuracy': 85.0
        },
        {
            'scenario_name': 'relevance_scoring_test',
            'total_operations': 15,
            'concurrent_operations': 3,
            'ramp_up_time': 2.0,
            'validation_type': 'relevance_scoring',
            'expected_accuracy': 90.0
        },
        {
            'scenario_name': 'integrated_workflow_test',
            'total_operations': 8,
            'concurrent_operations': 1,
            'ramp_up_time': 0.5,
            'validation_type': 'integrated_workflow',
            'expected_accuracy': 82.0
        }
    ]


@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing data persistence."""
    mock_db = Mock()
    mock_cursor = Mock()
    
    # Mock query results
    mock_cursor.fetchall.return_value = [
        ('test_metric_1', 1500.0, 87.5, 0.82),
        ('test_metric_2', 1300.0, 91.0, 0.88),
        ('test_metric_3', 1700.0, 84.0, 0.79)
    ]
    mock_cursor.fetchone.return_value = ('count', 3)
    
    mock_db.cursor.return_value = mock_cursor
    mock_db.execute = Mock()
    mock_db.commit = Mock()
    mock_db.rollback = Mock()
    mock_db.close = Mock()
    
    return mock_db


@pytest.fixture
def performance_baseline_data() -> Dict[str, Any]:
    """Create baseline performance data for comparison testing."""
    return {
        'baseline_timestamp': time.time() - 86400,  # 24 hours ago
        'baseline_metrics': {
            'average_response_time_ms': 1200.0,
            'average_accuracy_score': 86.5,
            'average_confidence_score': 0.83,
            'average_throughput': 4.8,
            'average_cost_per_operation': 0.065,
            'average_memory_usage_mb': 480.0,
            'average_cpu_usage_percent': 52.0,
            'error_rate_percent': 2.1
        },
        'baseline_sample_size': 150,
        'baseline_test_duration_hours': 8.0
    }


@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatically cleanup test environment after each test."""
    # Setup code (runs before each test)
    yield
    # Cleanup code (runs after each test)
    import gc
    gc.collect()  # Force garbage collection


@pytest.fixture(scope="session")
def test_configuration():
    """Global test configuration settings."""
    return {
        'test_timeout_seconds': 30,
        'max_test_data_size': 1000,
        'enable_performance_monitoring': False,
        'mock_external_services': True,
        'test_data_directory': Path(__file__).parent / 'test_data',
        'log_level': 'INFO',
        'parallel_test_workers': 2
    }


# Test markers for categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance validation tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "network: Tests that require network access"
    )
    config.addinivalue_line(
        "markers", "benchmark: Benchmark performance tests"
    )


# Custom pytest hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add performance marker to performance-related tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add integration marker to integration tests
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might take longer
        if any(keyword in item.name.lower() for keyword in ["comprehensive", "large", "concurrent"]):
            item.add_marker(pytest.mark.slow)
        
        # Add unit marker to all other tests
        if not any(marker.name in ["integration", "performance", "benchmark"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


def pytest_runtest_setup(item):
    """Setup hook that runs before each test."""
    # Skip network tests if no network available
    if "network" in item.keywords:
        pytest.importorskip("requests")
    
    # Skip performance tests in certain environments
    if "performance" in item.keywords and not item.config.getoption("--run-performance", default=False):
        pytest.skip("Performance tests disabled (use --run-performance to enable)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-performance", action="store_true", default=False,
        help="Run performance tests"
    )
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--run-all", action="store_true", default=False,
        help="Run all tests including slow ones"
    )