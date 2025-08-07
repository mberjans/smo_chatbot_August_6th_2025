#!/usr/bin/env python3
"""
Configuration Management for Quality Assessment with Factual Accuracy Integration.

This module provides comprehensive configuration management for the integrated
quality assessment system in the Clinical Metabolomics Oracle LightRAG project.
It handles configuration for all quality assessment components including factual
accuracy validation, relevance scoring, and response quality assessment.

Classes:
    - ConfigurationError: Base exception for configuration errors
    - QualityAssessmentConfig: Main configuration manager
    - ComponentConfig: Individual component configuration
    - ValidationConfig: Validation-specific configuration

Key Features:
    - Centralized configuration management
    - Component-specific configuration sections
    - Dynamic configuration updates
    - Configuration validation and validation
    - Environment-based configuration loading
    - Backwards compatibility with existing configurations
    - Performance optimization settings
    - Security and access control settings

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Quality Assessment Configuration Management
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy

# Configure logging
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Base custom exception for configuration errors."""
    pass


@dataclass
class ComponentConfig:
    """
    Configuration for individual quality assessment components.
    
    Attributes:
        enabled: Whether the component is enabled
        config: Component-specific configuration dictionary
        fallback_enabled: Whether to use fallback when component fails
        timeout_seconds: Timeout for component operations
        cache_enabled: Whether to enable caching for this component
        performance_monitoring: Whether to monitor performance
    """
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    fallback_enabled: bool = True
    timeout_seconds: float = 10.0
    cache_enabled: bool = True
    performance_monitoring: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentConfig':
        """Create from dictionary representation."""
        return cls(**data)


@dataclass 
class ValidationConfig:
    """
    Configuration for validation and quality thresholds.
    
    Attributes:
        minimum_quality_threshold: Minimum acceptable quality score (0-100)
        minimum_factual_accuracy_threshold: Minimum factual accuracy (0-100)
        minimum_relevance_threshold: Minimum relevance score (0-100)
        confidence_threshold: Minimum confidence for reliable results (0-100)
        enable_strict_validation: Whether to use strict validation rules
        validation_timeout_seconds: Timeout for validation operations
    """
    minimum_quality_threshold: float = 60.0
    minimum_factual_accuracy_threshold: float = 70.0
    minimum_relevance_threshold: float = 60.0
    confidence_threshold: float = 70.0
    enable_strict_validation: bool = False
    validation_timeout_seconds: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationConfig':
        """Create from dictionary representation."""
        return cls(**data)


class QualityAssessmentConfig:
    """
    Main configuration manager for quality assessment system.
    
    Manages configuration for all components including factual accuracy
    validation, relevance scoring, and response quality assessment.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config = self._get_default_config()
        self._component_configs = {}
        self._validation_config = ValidationConfig()
        
        # Load configuration from file if provided
        if self.config_path and self.config_path.exists():
            self.load_from_file(self.config_path)
        
        # Load from environment variables
        self._load_from_environment()
        
        # Initialize component configurations
        self._initialize_component_configs()
        
        logger.info("QualityAssessmentConfig initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for quality assessment system."""
        return {
            'system': {
                'enable_quality_assessment': True,
                'enable_factual_accuracy_validation': True,
                'enable_relevance_scoring': True,
                'enable_parallel_processing': True,
                'max_concurrent_assessments': 5,
                'global_timeout_seconds': 60.0,
                'fallback_on_component_failure': True,
                'detailed_logging': True,
                'performance_monitoring': True
            },
            'integration': {
                'component_weights': {
                    'relevance_score': 0.35,
                    'quality_metrics': 0.35,
                    'factual_accuracy': 0.30
                },
                'enable_cross_component_validation': True,
                'consistency_analysis_enabled': True,
                'recommendation_generation_enabled': True,
                'integrated_scoring_method': 'weighted_average'
            },
            'performance': {
                'enable_caching': True,
                'cache_ttl_seconds': 3600,
                'enable_async_processing': True,
                'max_processing_time_ms': 30000,
                'optimization_level': 'balanced',  # 'fast', 'balanced', 'thorough'
                'memory_limit_mb': 1024,
                'enable_performance_tracking': True
            },
            'security': {
                'enable_input_sanitization': True,
                'max_input_length': 50000,
                'allowed_file_types': ['.txt', '.json', '.md'],
                'enable_audit_logging': True,
                'rate_limiting_enabled': False
            },
            'factual_accuracy': {
                'enabled': True,
                'claim_extraction_enabled': True,
                'document_verification_enabled': True,
                'comprehensive_scoring_enabled': True,
                'minimum_claims_for_reliable_score': 3,
                'evidence_quality_weight': 0.4,
                'claim_verification_weight': 0.35,
                'consistency_weight': 0.25,
                'fallback_heuristic_enabled': True
            },
            'relevance_scoring': {
                'enabled': True,
                'enable_semantic_analysis': True,
                'enable_biomedical_context_scoring': True,
                'enable_query_type_classification': True,
                'enable_domain_expertise_validation': True,
                'parallel_dimension_calculation': True,
                'confidence_threshold': 70.0
            },
            'quality_assessment': {
                'enabled': True,
                'enable_comprehensive_metrics': True,
                'enable_biomedical_terminology_analysis': True,
                'enable_citation_analysis': True,
                'enable_hallucination_detection': True,
                'clarity_analysis_enabled': True,
                'completeness_analysis_enabled': True
            },
            'reporting': {
                'enable_detailed_reports': True,
                'enable_performance_reports': True,
                'enable_trend_analysis': True,
                'report_retention_days': 30,
                'export_formats': ['json', 'html'],
                'include_raw_data': False
            }
        }
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'CMO_QUALITY_FACTUAL_ACCURACY_ENABLED': ['factual_accuracy', 'enabled'],
            'CMO_QUALITY_RELEVANCE_ENABLED': ['relevance_scoring', 'enabled'], 
            'CMO_QUALITY_PARALLEL_PROCESSING': ['system', 'enable_parallel_processing'],
            'CMO_QUALITY_TIMEOUT_SECONDS': ['system', 'global_timeout_seconds'],
            'CMO_QUALITY_CACHE_ENABLED': ['performance', 'enable_caching'],
            'CMO_QUALITY_DETAILED_LOGGING': ['system', 'detailed_logging'],
            'CMO_QUALITY_MAX_CONCURRENT': ['system', 'max_concurrent_assessments']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert string values to appropriate types
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        value = float(value)
                    
                    # Set the configuration value
                    self._set_nested_config(self._config, config_path, value)
                    logger.info(f"Loaded configuration from environment: {env_var} = {value}")
                except Exception as e:
                    logger.warning(f"Error loading environment variable {env_var}: {str(e)}")
    
    def _set_nested_config(self, config: Dict[str, Any], path: List[str], value: Any):
        """Set a nested configuration value."""
        for key in path[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[path[-1]] = value
    
    def _initialize_component_configs(self):
        """Initialize individual component configurations."""
        # ClinicalMetabolomicsRelevanceScorer configuration
        self._component_configs['relevance_scorer'] = ComponentConfig(
            enabled=self._config['relevance_scoring']['enabled'],
            config={
                'enable_caching': self._config['performance']['enable_caching'],
                'parallel_processing': self._config['system']['enable_parallel_processing'],
                'confidence_threshold': self._config['relevance_scoring']['confidence_threshold'],
                'factual_accuracy_enabled': self._config['factual_accuracy']['enabled'],
                'factual_accuracy_fallback_enabled': self._config['factual_accuracy']['fallback_heuristic_enabled']
            },
            fallback_enabled=self._config['system']['fallback_on_component_failure'],
            timeout_seconds=self._config['system']['global_timeout_seconds'] / 3,  # Split timeout
            cache_enabled=self._config['performance']['enable_caching'],
            performance_monitoring=self._config['performance']['enable_performance_tracking']
        )
        
        # EnhancedResponseQualityAssessor configuration
        self._component_configs['quality_assessor'] = ComponentConfig(
            enabled=self._config['quality_assessment']['enabled'],
            config={
                'factual_validation_enabled': self._config['factual_accuracy']['enabled'],
                'fallback_on_error': self._config['system']['fallback_on_component_failure'],
                'minimum_claims_for_reliable_score': self._config['factual_accuracy']['minimum_claims_for_reliable_score'],
                'performance_timeout_seconds': self._config['system']['global_timeout_seconds'] / 3,
                'enable_caching': self._config['performance']['enable_caching'],
                'detailed_reporting': self._config['reporting']['enable_detailed_reports']
            },
            fallback_enabled=self._config['system']['fallback_on_component_failure'],
            timeout_seconds=self._config['system']['global_timeout_seconds'] / 3,
            cache_enabled=self._config['performance']['enable_caching'],
            performance_monitoring=self._config['performance']['enable_performance_tracking']
        )
        
        # BiomedicalClaimExtractor configuration
        self._component_configs['claim_extractor'] = ComponentConfig(
            enabled=self._config['factual_accuracy']['claim_extraction_enabled'],
            config={
                'enable_advanced_patterns': True,
                'confidence_threshold': 0.7,
                'max_claims_per_response': 50,
                'enable_claim_classification': True,
                'enable_context_extraction': True
            },
            fallback_enabled=True,
            timeout_seconds=self._config['system']['global_timeout_seconds'] / 5,
            cache_enabled=self._config['performance']['enable_caching'],
            performance_monitoring=self._config['performance']['enable_performance_tracking']
        )
        
        # FactualAccuracyValidator configuration
        self._component_configs['factual_validator'] = ComponentConfig(
            enabled=self._config['factual_accuracy']['document_verification_enabled'],
            config={
                'verification_strategy': 'comprehensive',
                'similarity_threshold': 0.75,
                'max_evidence_items': 10,
                'enable_semantic_matching': True,
                'context_window_size': 200,
                'evidence_quality_threshold': 0.6
            },
            fallback_enabled=True,
            timeout_seconds=self._config['system']['global_timeout_seconds'] / 3,
            cache_enabled=self._config['performance']['enable_caching'],
            performance_monitoring=self._config['performance']['enable_performance_tracking']
        )
        
        # FactualAccuracyScorer configuration
        self._component_configs['accuracy_scorer'] = ComponentConfig(
            enabled=self._config['factual_accuracy']['comprehensive_scoring_enabled'],
            config={
                'scoring_weights': {
                    'claim_verification': self._config['factual_accuracy']['claim_verification_weight'],
                    'evidence_quality': self._config['factual_accuracy']['evidence_quality_weight'],
                    'consistency_analysis': self._config['factual_accuracy']['consistency_weight']
                },
                'integration_settings': {
                    'enable_relevance_integration': True,
                    'quality_system_compatibility': True,
                    'generate_integration_data': True
                }
            },
            fallback_enabled=True,
            timeout_seconds=self._config['system']['global_timeout_seconds'] / 5,
            cache_enabled=self._config['performance']['enable_caching'],
            performance_monitoring=self._config['performance']['enable_performance_tracking']
        )
        
        # IntegratedQualityWorkflow configuration  
        self._component_configs['integrated_workflow'] = ComponentConfig(
            enabled=True,
            config={
                'enable_parallel_processing': self._config['system']['enable_parallel_processing'],
                'enable_factual_validation': self._config['factual_accuracy']['enabled'],
                'enable_relevance_scoring': self._config['relevance_scoring']['enabled'],
                'enable_quality_assessment': self._config['quality_assessment']['enabled'],
                'fallback_on_component_failure': self._config['system']['fallback_on_component_failure'],
                'max_processing_time_seconds': self._config['system']['global_timeout_seconds'],
                'component_weights': self._config['integration']['component_weights'],
                'performance_optimization': {
                    'use_async_components': self._config['performance']['enable_async_processing'],
                    'max_concurrent_assessments': self._config['system']['max_concurrent_assessments'],
                    'timeout_per_component': self._config['system']['global_timeout_seconds'] / 3
                }
            },
            fallback_enabled=True,
            timeout_seconds=self._config['system']['global_timeout_seconds'],
            cache_enabled=self._config['performance']['enable_caching'],
            performance_monitoring=self._config['performance']['enable_performance_tracking']
        )
    
    def get_component_config(self, component_name: str) -> ComponentConfig:
        """
        Get configuration for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            ComponentConfig for the specified component
            
        Raises:
            ConfigurationError: If component not found
        """
        if component_name not in self._component_configs:
            raise ConfigurationError(f"Component '{component_name}' not found")
        
        return self._component_configs[component_name]
    
    def get_validation_config(self) -> ValidationConfig:
        """Get validation configuration."""
        return self._validation_config
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system-level configuration."""
        return copy.deepcopy(self._config['system'])
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Get integration configuration."""
        return copy.deepcopy(self._config['integration'])
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return copy.deepcopy(self._config['performance'])
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return copy.deepcopy(self._config['security'])
    
    def update_component_config(self, component_name: str, config_updates: Dict[str, Any]):
        """
        Update configuration for a specific component.
        
        Args:
            component_name: Name of the component
            config_updates: Dictionary of configuration updates
            
        Raises:
            ConfigurationError: If component not found
        """
        if component_name not in self._component_configs:
            raise ConfigurationError(f"Component '{component_name}' not found")
        
        component_config = self._component_configs[component_name]
        
        # Update component configuration
        for key, value in config_updates.items():
            if hasattr(component_config, key):
                setattr(component_config, key, value)
            elif key in component_config.config:
                component_config.config[key] = value
            else:
                logger.warning(f"Unknown configuration key '{key}' for component '{component_name}'")
        
        logger.info(f"Updated configuration for component '{component_name}'")
    
    def update_validation_thresholds(self, **thresholds):
        """
        Update validation thresholds.
        
        Args:
            **thresholds: Keyword arguments for threshold updates
        """
        for key, value in thresholds.items():
            if hasattr(self._validation_config, key):
                setattr(self._validation_config, key, value)
                logger.info(f"Updated validation threshold: {key} = {value}")
            else:
                logger.warning(f"Unknown validation threshold: {key}")
    
    def enable_factual_accuracy_validation(self, comprehensive: bool = True):
        """
        Enable factual accuracy validation across all components.
        
        Args:
            comprehensive: Whether to enable comprehensive validation
        """
        # Update system configuration
        self._config['factual_accuracy']['enabled'] = True
        self._config['factual_accuracy']['claim_extraction_enabled'] = comprehensive
        self._config['factual_accuracy']['document_verification_enabled'] = comprehensive
        self._config['factual_accuracy']['comprehensive_scoring_enabled'] = comprehensive
        
        # Update component configurations
        self._component_configs['claim_extractor'].enabled = comprehensive
        self._component_configs['factual_validator'].enabled = comprehensive
        self._component_configs['accuracy_scorer'].enabled = comprehensive
        
        # Update relevance scorer and quality assessor
        self._component_configs['relevance_scorer'].config['factual_accuracy_enabled'] = True
        self._component_configs['quality_assessor'].config['factual_validation_enabled'] = True
        
        logger.info(f"Factual accuracy validation enabled (comprehensive: {comprehensive})")
    
    def disable_factual_accuracy_validation(self):
        """Disable factual accuracy validation across all components."""
        # Update system configuration
        self._config['factual_accuracy']['enabled'] = False
        
        # Update component configurations
        self._component_configs['claim_extractor'].enabled = False
        self._component_configs['factual_validator'].enabled = False
        self._component_configs['accuracy_scorer'].enabled = False
        
        # Update relevance scorer and quality assessor
        self._component_configs['relevance_scorer'].config['factual_accuracy_enabled'] = False
        self._component_configs['quality_assessor'].config['factual_validation_enabled'] = False
        
        logger.info("Factual accuracy validation disabled")
    
    def optimize_for_performance(self, level: str = 'balanced'):
        """
        Optimize configuration for different performance levels.
        
        Args:
            level: Performance optimization level ('fast', 'balanced', 'thorough')
        """
        if level == 'fast':
            self._config['system']['enable_parallel_processing'] = True
            self._config['system']['max_concurrent_assessments'] = 8
            self._config['performance']['enable_caching'] = True
            self._config['performance']['optimization_level'] = 'fast'
            self._config['factual_accuracy']['fallback_heuristic_enabled'] = True
            
            # Reduce timeouts
            for component in self._component_configs.values():
                component.timeout_seconds *= 0.7
                
        elif level == 'balanced':
            self._config['system']['enable_parallel_processing'] = True
            self._config['system']['max_concurrent_assessments'] = 5
            self._config['performance']['enable_caching'] = True
            self._config['performance']['optimization_level'] = 'balanced'
            
        elif level == 'thorough':
            self._config['system']['enable_parallel_processing'] = False
            self._config['system']['max_concurrent_assessments'] = 3
            self._config['performance']['optimization_level'] = 'thorough'
            self._config['factual_accuracy']['comprehensive_scoring_enabled'] = True
            
            # Increase timeouts
            for component in self._component_configs.values():
                component.timeout_seconds *= 1.5
        
        else:
            raise ConfigurationError(f"Unknown optimization level: {level}")
        
        logger.info(f"Configuration optimized for {level} performance")
    
    def load_from_file(self, config_path: Union[str, Path]):
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge with existing configuration
            self._merge_config(self._config, file_config)
            
            # Re-initialize component configurations
            self._initialize_component_configs()
            
            logger.info(f"Configuration loaded from file: {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from file: {str(e)}")
            raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e
    
    def save_to_file(self, config_path: Union[str, Path]):
        """
        Save current configuration to JSON file.
        
        Args:
            config_path: Path to save configuration file
            
        Raises:
            ConfigurationError: If file cannot be saved
        """
        try:
            config_path = Path(config_path)
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare configuration for saving
            save_config = {
                'system_config': self._config,
                'component_configs': {
                    name: config.to_dict() 
                    for name, config in self._component_configs.items()
                },
                'validation_config': self._validation_config.to_dict(),
                'saved_timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            with open(config_path, 'w') as f:
                json.dump(save_config, f, indent=2)
            
            logger.info(f"Configuration saved to file: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to file: {str(e)}")
            raise ConfigurationError(f"Failed to save configuration: {str(e)}") from e
    
    def _merge_config(self, base_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def validate_configuration(self) -> List[str]:
        """
        Validate current configuration and return list of issues.
        
        Returns:
            List of validation issues (empty if no issues)
        """
        issues = []
        
        # System configuration validation
        if self._config['system']['global_timeout_seconds'] <= 0:
            issues.append("Global timeout must be positive")
        
        if self._config['system']['max_concurrent_assessments'] <= 0:
            issues.append("Max concurrent assessments must be positive")
        
        # Performance configuration validation
        if self._config['performance']['max_processing_time_ms'] <= 0:
            issues.append("Max processing time must be positive")
        
        if self._config['performance']['cache_ttl_seconds'] <= 0:
            issues.append("Cache TTL must be positive")
        
        # Component weights validation
        weights = self._config['integration']['component_weights']
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            issues.append(f"Component weights should sum to 1.0, got {total_weight}")
        
        # Validation thresholds
        thresholds = [
            self._validation_config.minimum_quality_threshold,
            self._validation_config.minimum_factual_accuracy_threshold,
            self._validation_config.minimum_relevance_threshold,
            self._validation_config.confidence_threshold
        ]
        
        for threshold in thresholds:
            if not 0 <= threshold <= 100:
                issues.append(f"Threshold {threshold} must be between 0 and 100")
        
        # Component configuration validation
        for name, config in self._component_configs.items():
            if config.timeout_seconds <= 0:
                issues.append(f"Component '{name}' timeout must be positive")
        
        return issues
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            'system_enabled': {
                'quality_assessment': self._config['system']['enable_quality_assessment'],
                'factual_accuracy': self._config['factual_accuracy']['enabled'],
                'relevance_scoring': self._config['relevance_scoring']['enabled'],
                'parallel_processing': self._config['system']['enable_parallel_processing']
            },
            'component_status': {
                name: config.enabled 
                for name, config in self._component_configs.items()
            },
            'performance_settings': {
                'optimization_level': self._config['performance']['optimization_level'],
                'caching_enabled': self._config['performance']['enable_caching'],
                'max_concurrent': self._config['system']['max_concurrent_assessments'],
                'global_timeout': self._config['system']['global_timeout_seconds']
            },
            'validation_thresholds': {
                'quality': self._validation_config.minimum_quality_threshold,
                'factual_accuracy': self._validation_config.minimum_factual_accuracy_threshold,
                'relevance': self._validation_config.minimum_relevance_threshold,
                'confidence': self._validation_config.confidence_threshold
            },
            'integration_weights': self._config['integration']['component_weights'],
            'configuration_issues': self.validate_configuration()
        }


# Convenience functions for easy configuration management
def create_default_config() -> QualityAssessmentConfig:
    """Create configuration manager with default settings."""
    return QualityAssessmentConfig()


def create_optimized_config(optimization_level: str = 'balanced') -> QualityAssessmentConfig:
    """
    Create configuration manager optimized for specific performance level.
    
    Args:
        optimization_level: Performance optimization level ('fast', 'balanced', 'thorough')
        
    Returns:
        Optimized QualityAssessmentConfig instance
    """
    config = QualityAssessmentConfig()
    config.optimize_for_performance(optimization_level)
    return config


def load_config_from_file(config_path: Union[str, Path]) -> QualityAssessmentConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        QualityAssessmentConfig loaded from file
    """
    return QualityAssessmentConfig(config_path)


if __name__ == "__main__":
    # Configuration management test
    def test_configuration_management():
        """Test configuration management functionality."""
        
        print("Quality Assessment Configuration Management Test")
        print("=" * 60)
        
        # Create default configuration
        config = QualityAssessmentConfig()
        
        # Display configuration summary
        summary = config.get_configuration_summary()
        print(f"System Status:")
        for key, value in summary['system_enabled'].items():
            print(f"  {key}: {'Enabled' if value else 'Disabled'}")
        
        print(f"\nComponent Status:")
        for name, enabled in summary['component_status'].items():
            print(f"  {name}: {'Enabled' if enabled else 'Disabled'}")
        
        print(f"\nPerformance Settings:")
        for key, value in summary['performance_settings'].items():
            print(f"  {key}: {value}")
        
        print(f"\nValidation Thresholds:")
        for key, value in summary['validation_thresholds'].items():
            print(f"  {key}: {value}")
        
        # Test optimization
        print(f"\nOptimizing for fast performance...")
        config.optimize_for_performance('fast')
        
        updated_summary = config.get_configuration_summary()
        print(f"Updated max concurrent: {updated_summary['performance_settings']['max_concurrent']}")
        
        # Test validation
        issues = config.validate_configuration()
        if issues:
            print(f"\nConfiguration Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\nConfiguration validation: All checks passed")
        
        # Test component configuration
        relevance_config = config.get_component_config('relevance_scorer')
        print(f"\nRelevance Scorer Configuration:")
        print(f"  Enabled: {relevance_config.enabled}")
        print(f"  Timeout: {relevance_config.timeout_seconds}s")
        print(f"  Caching: {relevance_config.cache_enabled}")
        
        print(f"\nConfiguration management test completed successfully")
    
    # Run test
    test_configuration_management()