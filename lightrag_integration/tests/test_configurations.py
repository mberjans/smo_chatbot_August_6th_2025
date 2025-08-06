#!/usr/bin/env python3
"""
Test Configuration Data and Scenarios for Integration Testing.

This module provides comprehensive test configurations, parameter sets, and 
scenario definitions for integration testing between PDF processor and LightRAG 
components. It includes configurations for various test environments, failure 
scenarios, performance benchmarks, and edge cases.

Configuration Types:
- Base integration test configurations
- Performance testing configurations  
- Error injection configurations
- Resource constraint configurations
- Multi-environment test configurations
- Security and validation test configurations

Author: Claude Code (Anthropic)
Created: August 6, 2025
Version: 1.0.0
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import tempfile
import json


# =====================================================================
# BASE CONFIGURATION CLASSES
# =====================================================================

@dataclass
class IntegrationTestConfig:
    """Base configuration for integration testing."""
    
    # Test environment settings
    test_name: str
    description: str
    working_dir: Optional[Path] = None
    cleanup_after_test: bool = True
    
    # LightRAG settings
    api_key: str = "test-api-key"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    max_async: int = 4
    max_tokens: int = 8192
    chunk_size: int = 1000
    chunk_overlap: int = 200
    auto_create_dirs: bool = True
    enable_cost_tracking: bool = True
    
    # PDF processing settings
    pdf_processing_timeout: float = 30.0
    max_file_size_mb: float = 50.0
    batch_size: int = 5
    parallel_workers: int = 3
    retry_attempts: int = 2
    
    # Performance settings
    max_memory_mb: float = 1000.0
    max_processing_time: float = 120.0
    target_throughput: float = 2.0  # documents per second
    
    # Cost settings
    daily_budget_limit: float = 100.0
    monthly_budget_limit: float = 3000.0
    cost_alert_threshold: float = 0.75  # 75% of budget
    
    # Validation settings
    min_entities_per_document: int = 5
    min_relationships_per_document: int = 3
    expected_accuracy_threshold: float = 0.8
    
    def to_lightrag_config(self, working_dir: Path):
        """Convert to LightRAGConfig format."""
        from lightrag_integration.config import LightRAGConfig
        
        return LightRAGConfig(
            api_key=self.api_key,
            model=self.model,
            embedding_model=self.embedding_model,
            working_dir=working_dir,
            max_async=self.max_async,
            max_tokens=self.max_tokens,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            auto_create_dirs=self.auto_create_dirs,
            enable_cost_tracking=self.enable_cost_tracking
        )


@dataclass  
class FailureTestConfig(IntegrationTestConfig):
    """Configuration for failure scenario testing."""
    
    # Failure injection settings
    pdf_failure_rate: float = 0.0  # 0.0 to 1.0
    lightrag_failure_rate: float = 0.0
    network_timeout_rate: float = 0.0
    memory_pressure_enabled: bool = False
    disk_space_pressure_enabled: bool = False
    
    # Recovery settings
    enable_automatic_retry: bool = True
    max_retry_attempts: int = 3
    retry_backoff_multiplier: float = 2.0
    graceful_degradation_enabled: bool = True
    
    # Error handling validation
    expect_partial_success: bool = False
    require_error_logging: bool = True
    require_cleanup_on_failure: bool = True


@dataclass
class PerformanceTestConfig(IntegrationTestConfig):
    """Configuration for performance testing scenarios."""
    
    # Load settings
    document_count: int = 50
    concurrent_operations: int = 10
    query_load: int = 100
    test_duration_minutes: int = 10
    
    # Performance targets
    target_documents_per_second: float = 2.0
    target_queries_per_second: float = 10.0
    max_response_time_seconds: float = 5.0
    max_memory_usage_mb: float = 1000.0
    target_cost_per_document: float = 0.10
    
    # Monitoring settings
    resource_sampling_interval: float = 1.0
    enable_detailed_profiling: bool = False
    performance_alerts_enabled: bool = True
    
    # Stress testing
    enable_memory_stress: bool = False
    enable_cpu_stress: bool = False
    enable_io_stress: bool = False


@dataclass
class SecurityTestConfig(IntegrationTestConfig):
    """Configuration for security and validation testing."""
    
    # Input validation settings
    test_malformed_pdfs: bool = True
    test_oversized_files: bool = True
    test_invalid_encodings: bool = True
    test_suspicious_content: bool = True
    
    # API security settings
    test_invalid_api_keys: bool = True
    test_rate_limiting: bool = True
    test_input_sanitization: bool = True
    
    # Data protection settings
    test_pii_handling: bool = True
    test_data_retention: bool = True
    test_secure_cleanup: bool = True


# =====================================================================
# PREDEFINED TEST CONFIGURATIONS
# =====================================================================

class TestConfigurationLibrary:
    """Library of predefined test configurations for various scenarios."""
    
    @staticmethod
    def get_basic_integration_config() -> IntegrationTestConfig:
        """Get basic integration test configuration."""
        return IntegrationTestConfig(
            test_name="basic_integration",
            description="Basic integration test with minimal settings",
            max_async=2,
            batch_size=3,
            parallel_workers=2,
            daily_budget_limit=10.0,
            monthly_budget_limit=300.0,
            max_processing_time=60.0
        )
    
    @staticmethod
    def get_comprehensive_integration_config() -> IntegrationTestConfig:
        """Get comprehensive integration test configuration."""
        return IntegrationTestConfig(
            test_name="comprehensive_integration",
            description="Comprehensive integration test with full feature coverage",
            max_async=8,
            max_tokens=16384,
            chunk_size=1500,
            chunk_overlap=300,
            batch_size=10,
            parallel_workers=5,
            max_memory_mb=2000.0,
            daily_budget_limit=50.0,
            monthly_budget_limit=1500.0,
            max_processing_time=180.0,
            min_entities_per_document=10,
            min_relationships_per_document=6
        )
    
    @staticmethod
    def get_performance_benchmark_config() -> PerformanceTestConfig:
        """Get performance benchmarking configuration."""
        return PerformanceTestConfig(
            test_name="performance_benchmark",
            description="High-load performance testing scenario",
            document_count=100,
            concurrent_operations=20,
            query_load=500,
            test_duration_minutes=15,
            target_documents_per_second=5.0,
            target_queries_per_second=25.0,
            max_response_time_seconds=3.0,
            max_memory_usage_mb=1500.0,
            target_cost_per_document=0.05,
            enable_detailed_profiling=True,
            performance_alerts_enabled=True,
            max_async=16,
            parallel_workers=8
        )
    
    @staticmethod
    def get_memory_constrained_config() -> IntegrationTestConfig:
        """Get memory-constrained test configuration."""
        return IntegrationTestConfig(
            test_name="memory_constrained",
            description="Test under memory constraints",
            max_async=2,
            batch_size=2,
            parallel_workers=1,
            max_memory_mb=200.0,
            chunk_size=500,
            chunk_overlap=50,
            max_file_size_mb=5.0,
            max_processing_time=300.0
        )
    
    @staticmethod
    def get_high_failure_rate_config() -> FailureTestConfig:
        """Get high failure rate test configuration."""
        return FailureTestConfig(
            test_name="high_failure_rate",
            description="Test resilience with high failure rates",
            pdf_failure_rate=0.4,
            lightrag_failure_rate=0.2,
            network_timeout_rate=0.3,
            memory_pressure_enabled=True,
            max_retry_attempts=5,
            retry_backoff_multiplier=1.5,
            expect_partial_success=True,
            require_error_logging=True,
            require_cleanup_on_failure=True,
            batch_size=3,
            parallel_workers=2
        )
    
    @staticmethod
    def get_budget_constrained_config() -> IntegrationTestConfig:
        """Get budget-constrained test configuration."""
        return IntegrationTestConfig(
            test_name="budget_constrained",
            description="Test under tight budget constraints",
            daily_budget_limit=2.0,
            monthly_budget_limit=60.0,
            cost_alert_threshold=0.5,  # 50% alert threshold
            max_async=1,
            batch_size=2,
            chunk_size=800,
            target_cost_per_document=0.02
        )
    
    @staticmethod
    def get_security_validation_config() -> SecurityTestConfig:
        """Get security validation test configuration."""
        return SecurityTestConfig(
            test_name="security_validation",
            description="Comprehensive security and validation testing",
            test_malformed_pdfs=True,
            test_oversized_files=True,
            test_invalid_encodings=True,
            test_suspicious_content=True,
            test_invalid_api_keys=True,
            test_rate_limiting=True,
            test_input_sanitization=True,
            test_pii_handling=True,
            test_data_retention=True,
            test_secure_cleanup=True,
            max_file_size_mb=10.0,
            batch_size=1,  # Process one at a time for security testing
            parallel_workers=1
        )
    
    @staticmethod
    def get_scalability_test_config() -> PerformanceTestConfig:
        """Get scalability test configuration."""
        return PerformanceTestConfig(
            test_name="scalability_test",
            description="Large-scale scalability testing",
            document_count=500,
            concurrent_operations=50,
            query_load=2000,
            test_duration_minutes=30,
            target_documents_per_second=10.0,
            target_queries_per_second=50.0,
            max_response_time_seconds=2.0,
            max_memory_usage_mb=3000.0,
            target_cost_per_document=0.03,
            enable_memory_stress=True,
            enable_cpu_stress=True,
            max_async=32,
            parallel_workers=16,
            batch_size=20
        )
    
    @staticmethod
    def get_minimal_resources_config() -> IntegrationTestConfig:
        """Get minimal resources test configuration."""
        return IntegrationTestConfig(
            test_name="minimal_resources",
            description="Test with minimal system resources",
            max_async=1,
            batch_size=1,
            parallel_workers=1,
            max_memory_mb=100.0,
            chunk_size=300,
            chunk_overlap=30,
            max_file_size_mb=1.0,
            daily_budget_limit=1.0,
            max_processing_time=600.0,
            pdf_processing_timeout=60.0
        )
    
    @staticmethod
    def get_all_configurations() -> Dict[str, IntegrationTestConfig]:
        """Get all predefined configurations."""
        return {
            'basic_integration': TestConfigurationLibrary.get_basic_integration_config(),
            'comprehensive_integration': TestConfigurationLibrary.get_comprehensive_integration_config(),
            'performance_benchmark': TestConfigurationLibrary.get_performance_benchmark_config(),
            'memory_constrained': TestConfigurationLibrary.get_memory_constrained_config(),
            'high_failure_rate': TestConfigurationLibrary.get_high_failure_rate_config(),
            'budget_constrained': TestConfigurationLibrary.get_budget_constrained_config(),
            'security_validation': TestConfigurationLibrary.get_security_validation_config(),
            'scalability_test': TestConfigurationLibrary.get_scalability_test_config(),
            'minimal_resources': TestConfigurationLibrary.get_minimal_resources_config()
        }


# =====================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =====================================================================

@dataclass
class EnvironmentConfig:
    """Configuration for different test environments."""
    
    name: str
    description: str
    base_config: IntegrationTestConfig
    environment_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def get_config(self) -> IntegrationTestConfig:
        """Get configuration with environment overrides applied."""
        config = self.base_config
        
        # Apply overrides
        for key, value in self.environment_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


class EnvironmentConfigurationManager:
    """Manage environment-specific test configurations."""
    
    @staticmethod
    def get_development_config() -> EnvironmentConfig:
        """Get development environment configuration."""
        return EnvironmentConfig(
            name="development",
            description="Development environment with debugging enabled",
            base_config=TestConfigurationLibrary.get_basic_integration_config(),
            environment_overrides={
                'cleanup_after_test': False,  # Keep files for debugging
                'enable_cost_tracking': True,
                'daily_budget_limit': 5.0,
                'max_processing_time': 300.0,  # Longer timeout for debugging
                'batch_size': 2,  # Smaller batches for easier debugging
                'parallel_workers': 1  # Single worker for debugging
            }
        )
    
    @staticmethod
    def get_ci_config() -> EnvironmentConfig:
        """Get CI/CD pipeline configuration."""
        return EnvironmentConfig(
            name="ci_cd",
            description="CI/CD pipeline testing with fast execution",
            base_config=TestConfigurationLibrary.get_basic_integration_config(),
            environment_overrides={
                'cleanup_after_test': True,
                'max_processing_time': 60.0,  # Fast execution for CI
                'daily_budget_limit': 2.0,  # Low budget for CI
                'batch_size': 3,
                'parallel_workers': 2,
                'pdf_processing_timeout': 15.0,
                'enable_detailed_profiling': False
            }
        )
    
    @staticmethod
    def get_staging_config() -> EnvironmentConfig:
        """Get staging environment configuration."""
        return EnvironmentConfig(
            name="staging",
            description="Staging environment with production-like settings",
            base_config=TestConfigurationLibrary.get_comprehensive_integration_config(),
            environment_overrides={
                'cleanup_after_test': True,
                'daily_budget_limit': 20.0,
                'max_processing_time': 120.0,
                'batch_size': 8,
                'parallel_workers': 4,
                'enable_cost_tracking': True,
                'performance_alerts_enabled': True
            }
        )
    
    @staticmethod
    def get_production_validation_config() -> EnvironmentConfig:
        """Get production validation configuration."""
        return EnvironmentConfig(
            name="production_validation",
            description="Production validation with conservative settings",
            base_config=TestConfigurationLibrary.get_comprehensive_integration_config(),
            environment_overrides={
                'cleanup_after_test': True,
                'daily_budget_limit': 10.0,  # Conservative budget
                'max_processing_time': 180.0,
                'batch_size': 5,
                'parallel_workers': 3,
                'enable_cost_tracking': True,
                'cost_alert_threshold': 0.6,  # 60% alert threshold
                'require_error_logging': True,
                'require_cleanup_on_failure': True
            }
        )


# =====================================================================
# SPECIALIZED TEST SCENARIO CONFIGURATIONS
# =====================================================================

class TestScenarioConfigurations:
    """Specialized configurations for specific test scenarios."""
    
    @staticmethod
    def get_biomedical_domain_config() -> IntegrationTestConfig:
        """Get configuration optimized for biomedical content."""
        config = TestConfigurationLibrary.get_comprehensive_integration_config()
        config.test_name = "biomedical_domain"
        config.description = "Biomedical domain-specific optimization"
        config.chunk_size = 1200  # Larger chunks for biomedical papers
        config.chunk_overlap = 400  # More overlap for technical content
        config.min_entities_per_document = 15  # Expect more entities in biomedical text
        config.min_relationships_per_document = 10
        config.expected_accuracy_threshold = 0.85  # Higher accuracy for domain-specific
        return config
    
    @staticmethod
    def get_multilingual_config() -> IntegrationTestConfig:
        """Get configuration for multilingual content testing."""
        config = TestConfigurationLibrary.get_basic_integration_config()
        config.test_name = "multilingual"
        config.description = "Multilingual content processing"
        config.chunk_size = 800  # Smaller chunks for language processing
        config.max_processing_time = 240.0  # Longer for language processing
        config.expected_accuracy_threshold = 0.7  # Lower threshold for multilingual
        return config
    
    @staticmethod
    def get_real_time_processing_config() -> PerformanceTestConfig:
        """Get configuration for real-time processing scenarios."""
        config = PerformanceTestConfig(
            test_name="real_time_processing",
            description="Real-time document processing simulation",
            document_count=20,
            concurrent_operations=10,
            query_load=200,
            test_duration_minutes=5,
            target_documents_per_second=4.0,
            target_queries_per_second=15.0,
            max_response_time_seconds=2.0,
            max_memory_usage_mb=800.0,
            resource_sampling_interval=0.5,
            performance_alerts_enabled=True,
            max_async=12,
            parallel_workers=6,
            batch_size=1,  # Individual processing for real-time
            pdf_processing_timeout=10.0
        )
        return config
    
    @staticmethod
    def get_batch_processing_config() -> IntegrationTestConfig:
        """Get configuration for batch processing scenarios."""
        config = TestConfigurationLibrary.get_comprehensive_integration_config()
        config.test_name = "batch_processing"
        config.description = "Large batch processing optimization"
        config.batch_size = 25  # Large batches
        config.parallel_workers = 8
        config.max_async = 16
        config.max_processing_time = 600.0  # Longer for batch processing
        config.daily_budget_limit = 100.0  # Higher budget for batches
        config.target_throughput = 8.0  # Higher throughput expectation
        return config
    
    @staticmethod
    def get_edge_case_config() -> FailureTestConfig:
        """Get configuration for edge case testing."""
        config = FailureTestConfig(
            test_name="edge_cases",
            description="Edge case and boundary condition testing",
            pdf_failure_rate=0.1,
            lightrag_failure_rate=0.05,
            network_timeout_rate=0.1,
            memory_pressure_enabled=True,
            disk_space_pressure_enabled=True,
            max_retry_attempts=3,
            expect_partial_success=True,
            max_file_size_mb=100.0,  # Test large files
            batch_size=1,  # Process individually for edge cases
            parallel_workers=1,
            pdf_processing_timeout=120.0,
            max_processing_time=600.0
        )
        return config


# =====================================================================
# CONFIGURATION VALIDATION AND UTILITIES
# =====================================================================

class ConfigurationValidator:
    """Validate test configurations for consistency and feasibility."""
    
    @staticmethod
    def validate_config(config: IntegrationTestConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate resource constraints
        if config.max_memory_mb < 100:
            issues.append("max_memory_mb too low (< 100MB)")
        
        if config.max_processing_time < 10:
            issues.append("max_processing_time too low (< 10 seconds)")
        
        if config.batch_size > config.parallel_workers * 10:
            issues.append("batch_size may be too large relative to parallel_workers")
        
        # Validate budget settings
        if config.daily_budget_limit <= 0:
            issues.append("daily_budget_limit must be positive")
        
        if config.cost_alert_threshold >= 1.0:
            issues.append("cost_alert_threshold should be < 1.0")
        
        # Validate chunk settings
        if config.chunk_overlap >= config.chunk_size:
            issues.append("chunk_overlap should be less than chunk_size")
        
        if config.chunk_size < 100:
            issues.append("chunk_size too small (< 100)")
        
        # Validate thresholds
        if config.expected_accuracy_threshold > 1.0:
            issues.append("expected_accuracy_threshold should be <= 1.0")
        
        return issues
    
    @staticmethod
    def validate_performance_config(config: PerformanceTestConfig) -> List[str]:
        """Validate performance-specific configuration."""
        issues = ConfigurationValidator.validate_config(config)
        
        # Performance-specific validations
        if config.target_documents_per_second <= 0:
            issues.append("target_documents_per_second must be positive")
        
        if config.target_queries_per_second <= 0:
            issues.append("target_queries_per_second must be positive")
        
        if config.document_count < config.concurrent_operations:
            issues.append("document_count should be >= concurrent_operations")
        
        if config.test_duration_minutes <= 0:
            issues.append("test_duration_minutes must be positive")
        
        return issues
    
    @staticmethod
    def suggest_optimizations(config: IntegrationTestConfig) -> List[str]:
        """Suggest optimizations for the configuration."""
        suggestions = []
        
        # Memory optimizations
        if config.max_memory_mb > 2000:
            suggestions.append("Consider reducing max_memory_mb if not needed")
        
        # Performance optimizations
        if config.parallel_workers > config.max_async:
            suggestions.append("parallel_workers should not exceed max_async")
        
        # Cost optimizations
        if config.daily_budget_limit > 50.0:
            suggestions.append("Consider if high daily_budget_limit is necessary")
        
        # Batch optimizations
        if config.batch_size == 1 and config.parallel_workers > 1:
            suggestions.append("Consider increasing batch_size with multiple workers")
        
        return suggestions


# =====================================================================
# PYTEST FIXTURES FOR CONFIGURATIONS
# =====================================================================

@pytest.fixture
def basic_integration_config():
    """Provide basic integration test configuration."""
    return TestConfigurationLibrary.get_basic_integration_config()


@pytest.fixture
def comprehensive_integration_config():
    """Provide comprehensive integration test configuration."""
    return TestConfigurationLibrary.get_comprehensive_integration_config()


@pytest.fixture
def performance_test_config():
    """Provide performance test configuration."""
    return TestConfigurationLibrary.get_performance_benchmark_config()


@pytest.fixture
def failure_test_config():
    """Provide failure scenario test configuration."""
    return TestConfigurationLibrary.get_high_failure_rate_config()


@pytest.fixture
def security_test_config():
    """Provide security validation test configuration."""
    return TestConfigurationLibrary.get_security_validation_config()


@pytest.fixture
def biomedical_domain_config():
    """Provide biomedical domain-specific configuration."""
    return TestScenarioConfigurations.get_biomedical_domain_config()


@pytest.fixture
def memory_constrained_config():
    """Provide memory-constrained test configuration."""
    return TestConfigurationLibrary.get_memory_constrained_config()


@pytest.fixture
def budget_constrained_config():
    """Provide budget-constrained test configuration."""
    return TestConfigurationLibrary.get_budget_constrained_config()


@pytest.fixture
def development_environment_config():
    """Provide development environment configuration."""
    return EnvironmentConfigurationManager.get_development_config()


@pytest.fixture
def ci_environment_config():
    """Provide CI/CD environment configuration."""
    return EnvironmentConfigurationManager.get_ci_config()


@pytest.fixture
def staging_environment_config():
    """Provide staging environment configuration."""
    return EnvironmentConfigurationManager.get_staging_config()


@pytest.fixture
def all_test_configurations():
    """Provide all predefined test configurations."""
    return TestConfigurationLibrary.get_all_configurations()


@pytest.fixture
def configuration_validator():
    """Provide configuration validator utility."""
    return ConfigurationValidator()


# =====================================================================
# CONFIGURATION SELECTION UTILITIES
# =====================================================================

def select_config_for_test_type(test_type: str) -> IntegrationTestConfig:
    """Select appropriate configuration based on test type."""
    
    config_mapping = {
        'unit': TestConfigurationLibrary.get_basic_integration_config(),
        'integration': TestConfigurationLibrary.get_comprehensive_integration_config(),
        'performance': TestConfigurationLibrary.get_performance_benchmark_config(),
        'load': TestConfigurationLibrary.get_scalability_test_config(),
        'stress': TestConfigurationLibrary.get_high_failure_rate_config(),
        'security': TestConfigurationLibrary.get_security_validation_config(),
        'memory': TestConfigurationLibrary.get_memory_constrained_config(),
        'budget': TestConfigurationLibrary.get_budget_constrained_config(),
        'minimal': TestConfigurationLibrary.get_minimal_resources_config(),
        'biomedical': TestScenarioConfigurations.get_biomedical_domain_config(),
        'batch': TestScenarioConfigurations.get_batch_processing_config(),
        'realtime': TestScenarioConfigurations.get_real_time_processing_config(),
        'edge_cases': TestScenarioConfigurations.get_edge_case_config()
    }
    
    return config_mapping.get(test_type.lower(), 
                             TestConfigurationLibrary.get_basic_integration_config())


def create_custom_config(base_type: str = 'basic', **overrides) -> IntegrationTestConfig:
    """Create custom configuration with specified overrides."""
    
    base_config = select_config_for_test_type(base_type)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter '{key}' ignored")
    
    return base_config


def get_config_summary(config: IntegrationTestConfig) -> str:
    """Get human-readable summary of configuration."""
    
    summary = f"""
    Configuration: {config.test_name}
    Description: {config.description}
    
    Performance Settings:
    - Max Async: {config.max_async}
    - Batch Size: {config.batch_size}
    - Parallel Workers: {config.parallel_workers}
    - Max Memory: {config.max_memory_mb}MB
    - Max Processing Time: {config.max_processing_time}s
    
    LightRAG Settings:
    - Model: {config.model}
    - Embedding Model: {config.embedding_model}
    - Chunk Size: {config.chunk_size}
    - Chunk Overlap: {config.chunk_overlap}
    
    Budget Settings:
    - Daily Limit: ${config.daily_budget_limit}
    - Monthly Limit: ${config.monthly_budget_limit}
    - Alert Threshold: {config.cost_alert_threshold * 100}%
    
    Validation Settings:
    - Min Entities/Doc: {config.min_entities_per_document}
    - Min Relationships/Doc: {config.min_relationships_per_document}  
    - Accuracy Threshold: {config.expected_accuracy_threshold * 100}%
    """
    
    return summary.strip()