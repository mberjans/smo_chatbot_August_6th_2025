#!/usr/bin/env python3
"""
Test Configuration for Routing Decision Logic Testing

This module provides comprehensive configuration for all routing decision logic tests,
including performance targets, accuracy requirements, and test parameters.

Author: Claude Code (Anthropic)
Created: August 8, 2025
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TestCategory(Enum):
    """Test categories for organizing test execution"""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    ROUTING = "routing"
    THRESHOLDS = "thresholds"
    UNCERTAINTY = "uncertainty"
    INTEGRATION = "integration"
    EDGE_CASES = "edge_cases"
    STRESS = "stress"


@dataclass
class PerformanceTargets:
    """Performance targets and thresholds for routing system"""
    
    # Core performance requirements
    max_routing_time_ms: float = 50.0
    max_analysis_time_ms: float = 30.0
    max_classification_time_ms: float = 2000.0
    
    # Throughput requirements
    min_throughput_qps: float = 20.0
    max_concurrent_requests: int = 50
    
    # Resource limits
    max_memory_increase_mb: float = 50.0
    max_cpu_usage_percent: float = 80.0
    
    # Success rate requirements
    min_success_rate: float = 0.95
    min_availability: float = 0.99


@dataclass
class AccuracyTargets:
    """Accuracy targets and requirements for routing system"""
    
    # Core accuracy requirements
    overall_accuracy_target: float = 0.90
    min_category_accuracy: float = 0.85
    min_hybrid_accuracy: float = 0.75  # Lower bar for complex queries
    
    # Confidence calibration
    max_calibration_error: float = 0.15
    max_category_calibration_error: float = 0.20
    
    # Domain-specific targets
    clinical_metabolomics_accuracy: float = 0.88
    biomarker_discovery_accuracy: float = 0.90
    analytical_methods_accuracy: float = 0.85


@dataclass
class ConfidenceThresholds:
    """Confidence threshold configuration for testing"""
    
    # Primary thresholds (must match system configuration)
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6
    low_confidence_threshold: float = 0.5
    fallback_threshold: float = 0.2
    
    # Uncertainty thresholds
    high_ambiguity_threshold: float = 0.7
    high_conflict_threshold: float = 0.6
    weak_evidence_threshold: float = 0.3
    
    # Validation ranges
    high_confidence_range: Tuple[float, float] = (0.8, 0.95)
    medium_confidence_range: Tuple[float, float] = (0.6, 0.85)
    low_confidence_range: Tuple[float, float] = (0.5, 0.75)
    very_low_confidence_range: Tuple[float, float] = (0.1, 0.5)


@dataclass
class TestDataConfiguration:
    """Configuration for test data generation and validation"""
    
    # Test dataset sizes
    lightrag_test_cases: int = 100
    perplexity_test_cases: int = 100
    either_test_cases: int = 75
    hybrid_test_cases: int = 50
    
    # Performance test configurations
    performance_test_queries: int = 50
    concurrent_test_queries: int = 100
    stress_test_queries: int = 1000
    load_test_duration_seconds: int = 300
    
    # Edge case test configurations
    empty_query_variants: int = 10
    long_query_variants: int = 5
    special_char_variants: int = 15
    multilingual_variants: int = 8
    
    # Uncertainty test configurations
    low_confidence_cases: int = 25
    high_ambiguity_cases: int = 20
    conflicting_signal_cases: int = 15
    weak_evidence_cases: int = 20


@dataclass
class IntegrationTestConfiguration:
    """Configuration for integration and system-level tests"""
    
    # Component integration
    test_routing_classification_integration: bool = True
    test_threshold_cascade_integration: bool = True
    test_fallback_system_integration: bool = True
    
    # End-to-end workflow tests
    test_complete_routing_workflow: bool = True
    test_uncertainty_handling_workflow: bool = True
    test_performance_under_load: bool = True
    
    # Mock component configurations
    use_mock_components: bool = True
    mock_failure_rate: float = 0.1
    mock_response_delay_ms: float = 5.0


@dataclass
class ComprehensiveTestConfiguration:
    """Complete test configuration for routing decision logic testing"""
    
    # Component configurations
    performance_targets: PerformanceTargets = field(default_factory=PerformanceTargets)
    accuracy_targets: AccuracyTargets = field(default_factory=AccuracyTargets)
    confidence_thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    test_data_config: TestDataConfiguration = field(default_factory=TestDataConfiguration)
    integration_config: IntegrationTestConfiguration = field(default_factory=IntegrationTestConfiguration)
    
    # Test execution configuration
    parallel_execution: bool = True
    max_workers: int = 4
    test_timeout_seconds: int = 600
    verbose_output: bool = True
    generate_detailed_report: bool = True
    
    # Test categories to run
    enabled_test_categories: List[TestCategory] = field(default_factory=lambda: [
        TestCategory.ACCURACY,
        TestCategory.PERFORMANCE,
        TestCategory.ROUTING,
        TestCategory.THRESHOLDS,
        TestCategory.UNCERTAINTY,
        TestCategory.INTEGRATION,
        TestCategory.EDGE_CASES
    ])
    
    # Reporting configuration
    report_output_file: str = "routing_test_report.html"
    performance_metrics_file: str = "routing_performance_metrics.json"
    accuracy_details_file: str = "routing_accuracy_details.json"
    
    def get_pytest_markers(self) -> List[str]:
        """Get pytest markers based on enabled test categories"""
        marker_mapping = {
            TestCategory.ACCURACY: "accuracy",
            TestCategory.PERFORMANCE: "performance",
            TestCategory.ROUTING: "routing",
            TestCategory.THRESHOLDS: "thresholds",
            TestCategory.UNCERTAINTY: "uncertainty",
            TestCategory.INTEGRATION: "integration",
            TestCategory.EDGE_CASES: "edge_cases",
            TestCategory.STRESS: "stress"
        }
        
        return [marker_mapping[category] for category in self.enabled_test_categories 
                if category in marker_mapping]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'performance_targets': {
                'max_routing_time_ms': self.performance_targets.max_routing_time_ms,
                'max_analysis_time_ms': self.performance_targets.max_analysis_time_ms,
                'max_classification_time_ms': self.performance_targets.max_classification_time_ms,
                'min_throughput_qps': self.performance_targets.min_throughput_qps,
                'min_success_rate': self.performance_targets.min_success_rate
            },
            'accuracy_targets': {
                'overall_accuracy_target': self.accuracy_targets.overall_accuracy_target,
                'min_category_accuracy': self.accuracy_targets.min_category_accuracy,
                'max_calibration_error': self.accuracy_targets.max_calibration_error
            },
            'confidence_thresholds': {
                'high_confidence': self.confidence_thresholds.high_confidence_threshold,
                'medium_confidence': self.confidence_thresholds.medium_confidence_threshold,
                'low_confidence': self.confidence_thresholds.low_confidence_threshold,
                'fallback_threshold': self.confidence_thresholds.fallback_threshold
            },
            'test_data': {
                'total_test_cases': (
                    self.test_data_config.lightrag_test_cases +
                    self.test_data_config.perplexity_test_cases +
                    self.test_data_config.either_test_cases +
                    self.test_data_config.hybrid_test_cases
                ),
                'performance_tests': self.test_data_config.performance_test_queries,
                'edge_case_tests': (
                    self.test_data_config.empty_query_variants +
                    self.test_data_config.long_query_variants +
                    self.test_data_config.special_char_variants +
                    self.test_data_config.multilingual_variants
                )
            },
            'execution_config': {
                'parallel_execution': self.parallel_execution,
                'max_workers': self.max_workers,
                'test_timeout_seconds': self.test_timeout_seconds,
                'enabled_categories': [cat.value for cat in self.enabled_test_categories]
            }
        }


# Default configuration instances
DEFAULT_TEST_CONFIG = ComprehensiveTestConfiguration()

PERFORMANCE_FOCUSED_CONFIG = ComprehensiveTestConfiguration(
    enabled_test_categories=[TestCategory.PERFORMANCE, TestCategory.STRESS],
    test_data_config=TestDataConfiguration(
        performance_test_queries=100,
        concurrent_test_queries=200,
        stress_test_queries=5000,
        load_test_duration_seconds=600
    )
)

ACCURACY_FOCUSED_CONFIG = ComprehensiveTestConfiguration(
    enabled_test_categories=[TestCategory.ACCURACY, TestCategory.ROUTING],
    test_data_config=TestDataConfiguration(
        lightrag_test_cases=200,
        perplexity_test_cases=200,
        either_test_cases=150,
        hybrid_test_cases=100
    )
)

INTEGRATION_FOCUSED_CONFIG = ComprehensiveTestConfiguration(
    enabled_test_categories=[TestCategory.INTEGRATION, TestCategory.EDGE_CASES],
    integration_config=IntegrationTestConfiguration(
        test_routing_classification_integration=True,
        test_threshold_cascade_integration=True,
        test_fallback_system_integration=True,
        test_complete_routing_workflow=True,
        mock_failure_rate=0.2  # Higher failure rate for robustness testing
    )
)


def get_test_config(config_name: str = "default") -> ComprehensiveTestConfiguration:
    """
    Get test configuration by name.
    
    Args:
        config_name: Configuration name ("default", "performance", "accuracy", "integration")
        
    Returns:
        ComprehensiveTestConfiguration instance
    """
    configs = {
        "default": DEFAULT_TEST_CONFIG,
        "performance": PERFORMANCE_FOCUSED_CONFIG,
        "accuracy": ACCURACY_FOCUSED_CONFIG,
        "integration": INTEGRATION_FOCUSED_CONFIG
    }
    
    return configs.get(config_name, DEFAULT_TEST_CONFIG)


def validate_test_config(config: ComprehensiveTestConfiguration) -> Tuple[bool, List[str]]:
    """
    Validate test configuration for consistency and feasibility.
    
    Args:
        config: Test configuration to validate
        
    Returns:
        Tuple of (is_valid, list_of_validation_errors)
    """
    errors = []
    
    # Validate performance targets
    if config.performance_targets.max_routing_time_ms <= 0:
        errors.append("max_routing_time_ms must be positive")
    
    if config.performance_targets.max_routing_time_ms >= config.performance_targets.max_classification_time_ms:
        errors.append("max_routing_time_ms should be much less than max_classification_time_ms")
    
    # Validate accuracy targets
    if not (0.0 <= config.accuracy_targets.overall_accuracy_target <= 1.0):
        errors.append("overall_accuracy_target must be between 0.0 and 1.0")
    
    if config.accuracy_targets.min_category_accuracy > config.accuracy_targets.overall_accuracy_target:
        errors.append("min_category_accuracy should not exceed overall_accuracy_target")
    
    # Validate confidence thresholds
    thresholds = [
        config.confidence_thresholds.high_confidence_threshold,
        config.confidence_thresholds.medium_confidence_threshold,
        config.confidence_thresholds.low_confidence_threshold,
        config.confidence_thresholds.fallback_threshold
    ]
    
    if not all(0.0 <= t <= 1.0 for t in thresholds):
        errors.append("All confidence thresholds must be between 0.0 and 1.0")
    
    if not (thresholds[0] > thresholds[1] > thresholds[2] > thresholds[3]):
        errors.append("Confidence thresholds must be in descending order")
    
    # Validate test data configuration
    total_test_cases = (
        config.test_data_config.lightrag_test_cases +
        config.test_data_config.perplexity_test_cases +
        config.test_data_config.either_test_cases +
        config.test_data_config.hybrid_test_cases
    )
    
    if total_test_cases < 50:
        errors.append("Insufficient total test cases for reliable validation (minimum 50)")
    
    # Validate execution configuration
    if config.max_workers <= 0:
        errors.append("max_workers must be positive")
    
    if config.test_timeout_seconds <= 0:
        errors.append("test_timeout_seconds must be positive")
    
    # Validate enabled test categories
    if not config.enabled_test_categories:
        errors.append("At least one test category must be enabled")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def create_pytest_config_file(config: ComprehensiveTestConfiguration, 
                            output_file: str = "pytest.ini") -> str:
    """
    Create pytest configuration file based on test configuration.
    
    Args:
        config: Test configuration
        output_file: Output file name
        
    Returns:
        Path to created configuration file
    """
    
    markers = config.get_pytest_markers()
    marker_definitions = []
    
    marker_descriptions = {
        "accuracy": "Accuracy validation tests",
        "performance": "Performance requirement tests",
        "routing": "Core routing decision tests",
        "thresholds": "Confidence threshold tests",
        "uncertainty": "Uncertainty handling tests",
        "integration": "Component integration tests",
        "edge_cases": "Edge case and error handling tests",
        "stress": "Stress and load testing"
    }
    
    for marker in markers:
        description = marker_descriptions.get(marker, f"{marker} tests")
        marker_definitions.append(f"    {marker}: {description}")
    
    pytest_config = f"""# Pytest configuration for routing decision logic testing
# Generated automatically - do not edit manually

[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers --tb=short
testpaths = lightrag_integration/tests
timeout = {config.test_timeout_seconds}
markers =
{chr(10).join(marker_definitions)}

# Performance and parallel execution
workers = {config.max_workers if config.parallel_execution else 1}
dist = worksteal

# Output configuration
junit_family = xunit2
junit_logging = system-out

# Coverage configuration
addopts = --cov=lightrag_integration --cov-report=html --cov-report=term-missing

# Test discovery
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Warnings configuration
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::UserWarning:*lightrag*
"""
    
    with open(output_file, 'w') as f:
        f.write(pytest_config)
    
    return output_file


if __name__ == "__main__":
    # Example usage and validation
    print("Routing Test Configuration")
    print("=" * 50)
    
    # Test default configuration
    config = get_test_config("default")
    is_valid, errors = validate_test_config(config)
    
    print(f"Default configuration valid: {is_valid}")
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration details:")
        config_dict = config.to_dict()
        for section, details in config_dict.items():
            print(f"  {section}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"    {key}: {value}")
            else:
                print(f"    {details}")
    
    # Create pytest configuration
    pytest_config_file = create_pytest_config_file(config)
    print(f"\nPytest configuration created: {pytest_config_file}")