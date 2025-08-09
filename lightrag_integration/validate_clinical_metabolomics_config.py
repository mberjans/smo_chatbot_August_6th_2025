#!/usr/bin/env python3
"""
Clinical Metabolomics Configuration Validation Script
===================================================

This script validates the clinical metabolomics fallback configuration against
biomedical query requirements and performs comprehensive testing of the
configured system parameters.

Key Features:
- Configuration validation against clinical standards
- Timeout testing for complex scientific queries
- Confidence threshold validation for biomedical accuracy
- Cache pattern validation for metabolomics queries
- Fallback system testing with clinical scenarios
- Performance benchmarking with scientific content

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Validate clinical metabolomics fallback configuration
"""

import os
import sys
import time
import logging
import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from clinical_metabolomics_fallback_config import (
    ClinicalMetabolomicsFallbackConfig,
    ClinicalAccuracyLevel,
    create_clinical_metabolomics_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigurationValidationResult:
    """Results from configuration validation testing."""
    
    def __init__(self):
        self.test_results: Dict[str, bool] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.performance_metrics: Dict[str, float] = {}
        self.test_start_time = datetime.now()
        self.test_end_time: Optional[datetime] = None
    
    def add_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Add a test result."""
        self.test_results[test_name] = passed
        if not passed:
            self.errors.append(f"{test_name}: {details}")
        logger.info(f"Test '{test_name}': {'PASS' if passed else 'FAIL'}")
        if details:
            logger.info(f"  Details: {details}")
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)
    
    def add_performance_metric(self, metric_name: str, value: float):
        """Add a performance metric."""
        self.performance_metrics[metric_name] = value
        logger.info(f"Performance metric '{metric_name}': {value:.3f}")
    
    def finalize(self):
        """Finalize the validation results."""
        self.test_end_time = datetime.now()
        self.total_duration = (self.test_end_time - self.test_start_time).total_seconds()
        
        passed_tests = sum(1 for passed in self.test_results.values() if passed)
        total_tests = len(self.test_results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CLINICAL METABOLOMICS CONFIGURATION VALIDATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Errors: {len(self.errors)}")
        logger.info(f"Warnings: {len(self.warnings)}")
        logger.info(f"Total Duration: {self.total_duration:.2f} seconds")
        
        if self.errors:
            logger.error("\nERRORS:")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning("\nWARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
    
    def is_valid(self) -> bool:
        """Check if configuration validation passed."""
        return len(self.errors) == 0


class ClinicalMetabolomicsConfigValidator:
    """Validator for clinical metabolomics fallback configuration."""
    
    def __init__(self):
        self.result = ConfigurationValidationResult()
    
    def validate_complete_configuration(self, 
                                      environment: str = "production",
                                      accuracy_level: str = "research") -> ConfigurationValidationResult:
        """
        Perform complete validation of clinical metabolomics configuration.
        
        Args:
            environment: Environment to test
            accuracy_level: Accuracy level to test
            
        Returns:
            ConfigurationValidationResult: Validation results
        """
        logger.info(f"Starting clinical metabolomics configuration validation")
        logger.info(f"Environment: {environment}")
        logger.info(f"Accuracy Level: {accuracy_level}")
        
        # Create configuration
        try:
            config = create_clinical_metabolomics_config(environment, accuracy_level)
            self.result.add_test_result("config_creation", True, "Configuration created successfully")
        except Exception as e:
            self.result.add_test_result("config_creation", False, f"Failed to create configuration: {e}")
            return self.result
        
        # Run all validation tests
        self._validate_basic_configuration(config)
        self._validate_clinical_thresholds(config)
        self._validate_timeout_settings(config)
        self._validate_reliability_settings(config)
        self._validate_cache_configuration(config)
        self._validate_lightrag_optimization(config)
        self._validate_fallback_strategies(config)
        self._validate_monitoring_config(config)
        self._test_metabolomics_scenarios(config)
        self._benchmark_performance(config)
        
        # Finalize results
        self.result.finalize()
        return self.result
    
    def _validate_basic_configuration(self, config: ClinicalMetabolomicsFallbackConfig):
        """Validate basic configuration parameters."""
        logger.info("Validating basic configuration parameters...")
        
        # Test configuration validation method
        errors = config.validate_configuration()
        self.result.add_test_result(
            "basic_validation", 
            len(errors) == 0, 
            f"Configuration errors: {errors}" if errors else "No validation errors"
        )
        
        # Test environment setting
        self.result.add_test_result(
            "environment_setting",
            config.environment in ["development", "staging", "production"],
            f"Environment: {config.environment}"
        )
        
        # Test accuracy level
        self.result.add_test_result(
            "accuracy_level_setting",
            isinstance(config.accuracy_level, ClinicalAccuracyLevel),
            f"Accuracy level: {config.accuracy_level}"
        )
        
        # Test configuration serialization
        try:
            config_dict = config.to_dict()
            self.result.add_test_result(
                "config_serialization", 
                isinstance(config_dict, dict) and len(config_dict) > 0,
                f"Serialized {len(config_dict)} configuration keys"
            )
        except Exception as e:
            self.result.add_test_result("config_serialization", False, f"Serialization failed: {e}")
    
    def _validate_clinical_thresholds(self, config: ClinicalMetabolomicsFallbackConfig):
        """Validate clinical confidence thresholds."""
        logger.info("Validating clinical confidence thresholds...")
        
        thresholds = config.thresholds
        
        # Test diagnostic threshold (highest)
        self.result.add_test_result(
            "diagnostic_threshold",
            thresholds.diagnostic_confidence_threshold >= 0.85,
            f"Diagnostic threshold: {thresholds.diagnostic_confidence_threshold} (should be ≥0.85)"
        )
        
        # Test therapeutic threshold
        self.result.add_test_result(
            "therapeutic_threshold",
            thresholds.therapeutic_confidence_threshold >= 0.80,
            f"Therapeutic threshold: {thresholds.therapeutic_confidence_threshold} (should be ≥0.80)"
        )
        
        # Test research threshold
        self.result.add_test_result(
            "research_threshold",
            thresholds.research_confidence_threshold >= 0.70,
            f"Research threshold: {thresholds.research_confidence_threshold} (should be ≥0.70)"
        )
        
        # Test threshold hierarchy (diagnostic > therapeutic > research > educational > general)
        hierarchy_correct = (
            thresholds.diagnostic_confidence_threshold >= thresholds.therapeutic_confidence_threshold >= 
            thresholds.research_confidence_threshold >= thresholds.educational_confidence_threshold >= 
            thresholds.general_confidence_threshold
        )
        
        self.result.add_test_result(
            "threshold_hierarchy",
            hierarchy_correct,
            "Threshold hierarchy: diagnostic ≥ therapeutic ≥ research ≥ educational ≥ general"
        )
        
        # Test scientific accuracy minimum
        self.result.add_test_result(
            "scientific_accuracy_minimum",
            thresholds.minimum_scientific_accuracy >= 0.75,
            f"Scientific accuracy minimum: {thresholds.minimum_scientific_accuracy} (should be ≥0.75)"
        )
        
        # Test fallback thresholds
        self.result.add_test_result(
            "fallback_threshold_order",
            (thresholds.lightrag_fallback_threshold >= thresholds.perplexity_fallback_threshold >= 
             thresholds.emergency_cache_threshold),
            "Fallback threshold order: LightRAG ≥ Perplexity ≥ Emergency cache"
        )
    
    def _validate_timeout_settings(self, config: ClinicalMetabolomicsFallbackConfig):
        """Validate timeout settings for scientific queries."""
        logger.info("Validating timeout settings...")
        
        timeouts = config.timeouts
        
        # Test minimum timeouts for scientific processing
        self.result.add_test_result(
            "lightrag_primary_timeout",
            timeouts.lightrag_primary_timeout >= 30.0,
            f"LightRAG primary timeout: {timeouts.lightrag_primary_timeout}s (should be ≥30s)"
        )
        
        self.result.add_test_result(
            "complex_query_timeout",
            timeouts.lightrag_complex_query_timeout >= 45.0,
            f"Complex query timeout: {timeouts.lightrag_complex_query_timeout}s (should be ≥45s)"
        )
        
        self.result.add_test_result(
            "literature_search_timeout",
            timeouts.lightrag_literature_search_timeout >= 60.0,
            f"Literature search timeout: {timeouts.lightrag_literature_search_timeout}s (should be ≥60s)"
        )
        
        # Test timeout hierarchy
        timeout_hierarchy = (
            timeouts.lightrag_literature_search_timeout >= 
            timeouts.lightrag_complex_query_timeout >= 
            timeouts.lightrag_primary_timeout
        )
        
        self.result.add_test_result(
            "timeout_hierarchy",
            timeout_hierarchy,
            "Timeout hierarchy: Literature search ≥ Complex query ≥ Primary"
        )
        
        # Test circuit breaker timeout
        self.result.add_test_result(
            "circuit_breaker_timeout",
            timeouts.circuit_breaker_timeout >= 300.0,
            f"Circuit breaker timeout: {timeouts.circuit_breaker_timeout}s (should be ≥300s)"
        )
    
    def _validate_reliability_settings(self, config: ClinicalMetabolomicsFallbackConfig):
        """Validate reliability and safety settings."""
        logger.info("Validating reliability settings...")
        
        reliability = config.reliability
        
        # Test failure tolerance (should be stricter for clinical)
        self.result.add_test_result(
            "max_consecutive_failures",
            1 <= reliability.max_consecutive_failures <= 3,
            f"Max consecutive failures: {reliability.max_consecutive_failures} (should be 1-3)"
        )
        
        self.result.add_test_result(
            "failure_rate_threshold",
            reliability.failure_rate_threshold <= 0.1,
            f"Failure rate threshold: {reliability.failure_rate_threshold} (should be ≤0.1)"
        )
        
        # Test safety mechanisms
        safety_checks = [
            ("citation_verification", reliability.enable_citation_verification),
            ("source_attribution", reliability.require_source_attribution),
            ("fact_checking", reliability.enable_fact_checking),
            ("real_time_monitoring", reliability.enable_real_time_monitoring)
        ]
        
        for check_name, enabled in safety_checks:
            self.result.add_test_result(
                f"safety_{check_name}",
                enabled,
                f"{check_name.replace('_', ' ').title()}: {'Enabled' if enabled else 'Disabled'}"
            )
        
        # Test recovery settings
        self.result.add_test_result(
            "auto_recovery",
            reliability.auto_recovery_enabled,
            f"Auto recovery: {'Enabled' if reliability.auto_recovery_enabled else 'Disabled'}"
        )
        
        self.result.add_test_result(
            "recovery_threshold",
            reliability.recovery_success_threshold >= 0.80,
            f"Recovery success threshold: {reliability.recovery_success_threshold} (should be ≥0.80)"
        )
    
    def _validate_cache_configuration(self, config: ClinicalMetabolomicsFallbackConfig):
        """Validate cache configuration for metabolomics queries."""
        logger.info("Validating cache configuration...")
        
        cache = config.cache_patterns
        
        # Test common queries
        self.result.add_test_result(
            "cache_common_queries",
            len(cache.common_queries) >= 15,
            f"Common queries: {len(cache.common_queries)} (should be ≥15)"
        )
        
        # Test priority terms
        self.result.add_test_result(
            "cache_priority_terms",
            len(cache.priority_terms) >= 15,
            f"Priority terms: {len(cache.priority_terms)} (should be ≥15)"
        )
        
        # Test metabolomics-specific content
        metabolomics_terms = ["metabolomics", "biomarker", "LC-MS", "GC-MS"]
        has_metabolomics_terms = any(
            any(term.lower() in query.lower() for term in metabolomics_terms)
            for query in cache.common_queries
        )
        
        self.result.add_test_result(
            "metabolomics_cache_content",
            has_metabolomics_terms,
            "Cache includes metabolomics-specific queries"
        )
        
        # Test cache settings
        self.result.add_test_result(
            "cache_warming_enabled",
            cache.warm_cache_on_startup,
            f"Cache warming on startup: {'Enabled' if cache.warm_cache_on_startup else 'Disabled'}"
        )
        
        self.result.add_test_result(
            "cache_refresh_interval",
            6 <= cache.cache_refresh_interval_hours <= 48,
            f"Cache refresh interval: {cache.cache_refresh_interval_hours}h (should be 6-48h)"
        )
    
    def _validate_lightrag_optimization(self, config: ClinicalMetabolomicsFallbackConfig):
        """Validate LightRAG optimization settings."""
        logger.info("Validating LightRAG optimization...")
        
        optimization = config.lightrag_optimization
        
        # Test model selection
        self.result.add_test_result(
            "primary_model",
            optimization.primary_model in ["gpt-4o", "gpt-4", "gpt-4-turbo"],
            f"Primary model: {optimization.primary_model} (should be high-capability model)"
        )
        
        # Test token limits
        self.result.add_test_result(
            "max_tokens_primary",
            optimization.max_tokens_primary >= 16384,
            f"Max tokens (primary): {optimization.max_tokens_primary} (should be ≥16384)"
        )
        
        # Test embedding model
        self.result.add_test_result(
            "embedding_model",
            "embedding" in optimization.embedding_model.lower(),
            f"Embedding model: {optimization.embedding_model}"
        )
        
        # Test scientific processing parameters
        self.result.add_test_result(
            "chunk_size_scientific",
            1024 <= optimization.chunk_size_scientific <= 4096,
            f"Scientific chunk size: {optimization.chunk_size_scientific} (should be 1024-4096)"
        )
        
        self.result.add_test_result(
            "max_search_results",
            optimization.max_search_results >= 15,
            f"Max search results: {optimization.max_search_results} (should be ≥15)"
        )
        
        # Test scientific validation features
        scientific_features = [
            ("scientific_validation", optimization.enable_scientific_validation),
            ("citation_extraction", optimization.enable_citation_extraction),
            ("result_reranking", optimization.rerank_results)
        ]
        
        for feature_name, enabled in scientific_features:
            self.result.add_test_result(
                f"scientific_{feature_name}",
                enabled,
                f"{feature_name.replace('_', ' ').title()}: {'Enabled' if enabled else 'Disabled'}"
            )
    
    def _validate_fallback_strategies(self, config: ClinicalMetabolomicsFallbackConfig):
        """Validate fallback strategies configuration."""
        logger.info("Validating fallback strategies...")
        
        strategies = config.get_fallback_strategies()
        
        # Test number of strategies
        self.result.add_test_result(
            "fallback_strategies_count",
            len(strategies) >= 3,
            f"Number of fallback strategies: {len(strategies)} (should be ≥3)"
        )
        
        # Test strategy hierarchy
        strategy_levels = [strategy.level.value for strategy in strategies]
        has_primary = 1 in strategy_levels  # FULL_LLM_WITH_CONFIDENCE
        has_emergency = 4 in strategy_levels  # EMERGENCY_CACHE
        has_default = 5 in strategy_levels  # DEFAULT_ROUTING
        
        self.result.add_test_result(
            "fallback_strategy_coverage",
            has_primary and has_emergency and has_default,
            "Has primary, emergency, and default fallback strategies"
        )
        
        # Test strategy timeouts
        timeout_reasonable = all(
            5.0 <= strategy.timeout_seconds <= 120.0 
            for strategy in strategies
        )
        
        self.result.add_test_result(
            "strategy_timeouts",
            timeout_reasonable,
            "All strategy timeouts are reasonable (5-120s)"
        )
        
        # Test clinical naming
        clinical_strategies = [s for s in strategies if "clinical" in s.name.lower()]
        self.result.add_test_result(
            "clinical_strategy_naming",
            len(clinical_strategies) >= 1,
            f"Clinical-specific strategies: {len(clinical_strategies)}"
        )
    
    def _validate_monitoring_config(self, config: ClinicalMetabolomicsFallbackConfig):
        """Validate monitoring configuration."""
        logger.info("Validating monitoring configuration...")
        
        monitoring = config.get_monitoring_config()
        
        # Test real-time monitoring
        self.result.add_test_result(
            "real_time_monitoring",
            monitoring.get("enable_real_time_monitoring", False),
            "Real-time monitoring enabled"
        )
        
        # Test alert thresholds
        alert_thresholds = monitoring.get("alert_thresholds", {})
        
        accuracy_threshold = alert_thresholds.get("accuracy_drop", 0)
        self.result.add_test_result(
            "accuracy_alert_threshold",
            accuracy_threshold >= 0.75,
            f"Accuracy alert threshold: {accuracy_threshold} (should be ≥0.75)"
        )
        
        failure_rate_threshold = alert_thresholds.get("failure_rate", 1.0)
        self.result.add_test_result(
            "failure_rate_alert_threshold",
            failure_rate_threshold <= 0.1,
            f"Failure rate alert threshold: {failure_rate_threshold} (should be ≤0.1)"
        )
        
        # Test quality metrics
        quality_metrics = monitoring.get("quality_metrics", {})
        
        quality_features = [
            "enable_citation_verification",
            "enable_fact_checking",
            "source_attribution_required"
        ]
        
        quality_enabled = sum(1 for feature in quality_features if quality_metrics.get(feature, False))
        
        self.result.add_test_result(
            "quality_metrics_enabled",
            quality_enabled >= 2,
            f"Quality metrics enabled: {quality_enabled}/{len(quality_features)}"
        )
    
    def _test_metabolomics_scenarios(self, config: ClinicalMetabolomicsFallbackConfig):
        """Test configuration with metabolomics-specific scenarios."""
        logger.info("Testing metabolomics-specific scenarios...")
        
        # Test diagnostic scenario
        if config.accuracy_level in [ClinicalAccuracyLevel.DIAGNOSTIC, ClinicalAccuracyLevel.THERAPEUTIC]:
            diagnostic_threshold = config.thresholds.diagnostic_confidence_threshold
            self.result.add_test_result(
                "diagnostic_scenario_threshold",
                diagnostic_threshold >= 0.85,
                f"Diagnostic scenario threshold: {diagnostic_threshold}"
            )
        
        # Test research scenario
        research_queries = config.get_cache_warming_queries()
        research_focused = any(
            term in query.lower() 
            for query in research_queries 
            for term in ["research", "analysis", "biomarker", "pathway"]
        )
        
        self.result.add_test_result(
            "research_scenario_coverage",
            research_focused,
            "Configuration includes research-focused queries"
        )
        
        # Test timeout adequacy for complex queries
        literature_timeout = config.timeouts.lightrag_literature_search_timeout
        self.result.add_test_result(
            "literature_search_timeout_adequate",
            literature_timeout >= 60.0,
            f"Literature search timeout: {literature_timeout}s (adequate for complex searches)"
        )
        
        # Test emergency fallback for clinical safety
        emergency_threshold = config.thresholds.emergency_cache_threshold
        self.result.add_test_result(
            "emergency_fallback_available",
            emergency_threshold <= 0.5,
            f"Emergency fallback threshold: {emergency_threshold} (accessible when needed)"
        )
    
    def _benchmark_performance(self, config: ClinicalMetabolomicsFallbackConfig):
        """Benchmark configuration performance characteristics."""
        logger.info("Benchmarking configuration performance...")
        
        start_time = time.time()
        
        # Test configuration creation performance
        config_creation_start = time.time()
        try:
            test_config = create_clinical_metabolomics_config("production", "research")
            config_creation_time = time.time() - config_creation_start
            self.result.add_performance_metric("config_creation_time_ms", config_creation_time * 1000)
            
            self.result.add_test_result(
                "config_creation_performance",
                config_creation_time < 1.0,
                f"Configuration creation time: {config_creation_time:.3f}s"
            )
        except Exception as e:
            self.result.add_test_result("config_creation_performance", False, f"Creation failed: {e}")
        
        # Test LightRAG config generation performance
        lightrag_config_start = time.time()
        try:
            lightrag_config = config.get_lightrag_config()
            lightrag_config_time = time.time() - lightrag_config_start
            self.result.add_performance_metric("lightrag_config_time_ms", lightrag_config_time * 1000)
            
            self.result.add_test_result(
                "lightrag_config_performance",
                lightrag_config_time < 0.5,
                f"LightRAG config generation time: {lightrag_config_time:.3f}s"
            )
        except Exception as e:
            self.result.add_test_result("lightrag_config_performance", False, f"Generation failed: {e}")
        
        # Test fallback strategies generation performance
        strategies_start = time.time()
        try:
            strategies = config.get_fallback_strategies()
            strategies_time = time.time() - strategies_start
            self.result.add_performance_metric("strategies_generation_time_ms", strategies_time * 1000)
            
            self.result.add_test_result(
                "strategies_generation_performance",
                strategies_time < 0.1,
                f"Strategies generation time: {strategies_time:.3f}s"
            )
        except Exception as e:
            self.result.add_test_result("strategies_generation_performance", False, f"Generation failed: {e}")
        
        # Test serialization performance
        serialization_start = time.time()
        try:
            config_dict = config.to_dict()
            serialization_time = time.time() - serialization_start
            self.result.add_performance_metric("serialization_time_ms", serialization_time * 1000)
            
            self.result.add_test_result(
                "serialization_performance",
                serialization_time < 0.1,
                f"Serialization time: {serialization_time:.3f}s"
            )
        except Exception as e:
            self.result.add_test_result("serialization_performance", False, f"Serialization failed: {e}")
        
        total_benchmark_time = time.time() - start_time
        self.result.add_performance_metric("total_benchmark_time_ms", total_benchmark_time * 1000)


def save_validation_report(result: ConfigurationValidationResult, 
                          output_dir: Path = Path("validation_results")):
    """Save validation report to files."""
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report
    json_report = {
        "validation_timestamp": result.test_start_time.isoformat(),
        "validation_duration_seconds": result.total_duration,
        "test_results": result.test_results,
        "errors": result.errors,
        "warnings": result.warnings,
        "performance_metrics": result.performance_metrics,
        "summary": {
            "total_tests": len(result.test_results),
            "passed_tests": sum(1 for passed in result.test_results.values() if passed),
            "failed_tests": sum(1 for passed in result.test_results.values() if not passed),
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "is_valid": result.is_valid()
        }
    }
    
    json_file = output_dir / f"clinical_metabolomics_validation_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    logger.info(f"Validation report saved to: {json_file}")
    
    # Save human-readable report
    text_file = output_dir / f"clinical_metabolomics_validation_{timestamp}.txt"
    with open(text_file, 'w') as f:
        f.write("Clinical Metabolomics Configuration Validation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Validation Time: {result.test_start_time}\n")
        f.write(f"Duration: {result.total_duration:.2f} seconds\n")
        f.write(f"Overall Result: {'PASS' if result.is_valid() else 'FAIL'}\n\n")
        
        f.write("Test Results:\n")
        f.write("-" * 40 + "\n")
        for test_name, passed in result.test_results.items():
            status = "PASS" if passed else "FAIL"
            f.write(f"{test_name}: {status}\n")
        
        if result.errors:
            f.write("\nErrors:\n")
            f.write("-" * 40 + "\n")
            for error in result.errors:
                f.write(f"- {error}\n")
        
        if result.warnings:
            f.write("\nWarnings:\n")
            f.write("-" * 40 + "\n")
            for warning in result.warnings:
                f.write(f"- {warning}\n")
        
        if result.performance_metrics:
            f.write("\nPerformance Metrics:\n")
            f.write("-" * 40 + "\n")
            for metric_name, value in result.performance_metrics.items():
                f.write(f"{metric_name}: {value:.3f}\n")
    
    logger.info(f"Human-readable report saved to: {text_file}")
    
    return json_file, text_file


def main():
    """Main validation function."""
    logger.info("Starting Clinical Metabolomics Configuration Validation")
    
    # Parse command line arguments
    environment = os.getenv("VALIDATION_ENVIRONMENT", "production")
    accuracy_level = os.getenv("VALIDATION_ACCURACY_LEVEL", "research")
    
    logger.info(f"Testing environment: {environment}")
    logger.info(f"Testing accuracy level: {accuracy_level}")
    
    # Run validation
    validator = ClinicalMetabolomicsConfigValidator()
    result = validator.validate_complete_configuration(environment, accuracy_level)
    
    # Save reports
    try:
        json_file, text_file = save_validation_report(result)
        logger.info(f"Validation reports saved successfully")
    except Exception as e:
        logger.error(f"Failed to save validation reports: {e}")
    
    # Exit with appropriate code
    exit_code = 0 if result.is_valid() else 1
    logger.info(f"Validation {'completed successfully' if exit_code == 0 else 'failed'}")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)