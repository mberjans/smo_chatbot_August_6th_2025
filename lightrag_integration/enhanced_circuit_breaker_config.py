"""
Enhanced Circuit Breaker Configuration Support
=============================================

This module provides configuration support for the enhanced circuit breaker system
with environment variable integration and default values that work well with the
existing system architecture.

Classes:
    - EnhancedCircuitBreakerConfig: Configuration class with environment variable support
    - ConfigurationValidator: Validates configuration values
    - ConfigurationLoader: Loads configuration from various sources
    - DefaultConfigurationProvider: Provides sensible default values

Features:
- Environment variable integration with ENHANCED_CB_ prefix
- Validation of configuration values
- Integration with existing configuration patterns
- Backward compatibility with existing circuit breaker configurations
- Support for service-specific configurations

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Enhanced Circuit Breaker Configuration Support
Version: 1.0.0
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ServiceCircuitBreakerConfig:
    """Configuration for a specific service's circuit breaker."""
    
    # Basic circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    degradation_threshold: int = 3
    half_open_max_calls: int = 3
    
    # Service-specific settings
    rate_limit_threshold: int = 10
    budget_threshold_percentage: float = 90.0
    memory_threshold_gb: float = 2.0
    response_time_threshold: float = 30.0
    
    # Advanced settings
    enable_adaptive_thresholds: bool = True
    enable_progressive_degradation: bool = True
    enable_automatic_recovery: bool = True
    enable_cross_service_coordination: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'failure_threshold': self.failure_threshold,
            'recovery_timeout': self.recovery_timeout,
            'degradation_threshold': self.degradation_threshold,
            'half_open_max_calls': self.half_open_max_calls,
            'rate_limit_threshold': self.rate_limit_threshold,
            'budget_threshold_percentage': self.budget_threshold_percentage,
            'memory_threshold_gb': self.memory_threshold_gb,
            'response_time_threshold': self.response_time_threshold,
            'enable_adaptive_thresholds': self.enable_adaptive_thresholds,
            'enable_progressive_degradation': self.enable_progressive_degradation,
            'enable_automatic_recovery': self.enable_automatic_recovery,
            'enable_cross_service_coordination': self.enable_cross_service_coordination,
        }


@dataclass
class EnhancedCircuitBreakerConfig:
    """Enhanced circuit breaker system configuration with environment variable support."""
    
    # Global settings
    enabled: bool = True
    orchestrator_enabled: bool = True
    monitoring_enabled: bool = True
    metrics_collection_enabled: bool = True
    
    # Monitoring and alerting
    monitoring_interval_seconds: int = 30
    alert_enabled: bool = True
    alert_cooldown_seconds: int = 300
    
    # Service-specific configurations
    openai_api: ServiceCircuitBreakerConfig = field(default_factory=ServiceCircuitBreakerConfig)
    perplexity_api: ServiceCircuitBreakerConfig = field(default_factory=ServiceCircuitBreakerConfig)
    lightrag: ServiceCircuitBreakerConfig = field(default_factory=ServiceCircuitBreakerConfig)
    cache: ServiceCircuitBreakerConfig = field(default_factory=ServiceCircuitBreakerConfig)
    embedding_service: ServiceCircuitBreakerConfig = field(default_factory=ServiceCircuitBreakerConfig)
    knowledge_base: ServiceCircuitBreakerConfig = field(default_factory=ServiceCircuitBreakerConfig)
    
    # Integration settings
    integrate_with_cost_based: bool = True
    integrate_with_fallback_system: bool = True
    integrate_with_load_balancer: bool = True
    
    # Advanced features
    enable_failure_correlation: bool = True
    enable_cascading_failure_prevention: bool = True
    enable_system_health_assessment: bool = True
    
    def __post_init__(self):
        """Initialize service-specific configurations with appropriate defaults."""
        # Configure OpenAI API circuit breaker
        self.openai_api.rate_limit_threshold = 10
        self.openai_api.budget_threshold_percentage = 90.0
        self.openai_api.response_time_threshold = 15.0
        
        # Configure Perplexity API circuit breaker
        self.perplexity_api.rate_limit_threshold = 15
        self.perplexity_api.budget_threshold_percentage = 85.0
        self.perplexity_api.response_time_threshold = 20.0
        
        # Configure LightRAG circuit breaker
        self.lightrag.memory_threshold_gb = 2.0
        self.lightrag.response_time_threshold = 30.0
        self.lightrag.failure_threshold = 7  # Higher threshold for internal service
        
        # Configure Cache circuit breaker
        self.cache.failure_threshold = 10
        self.cache.recovery_timeout = 30.0
        self.cache.memory_threshold_gb = 1.0
        self.cache.response_time_threshold = 5.0
        
        # Configure Embedding Service circuit breaker
        self.embedding_service.rate_limit_threshold = 20
        self.embedding_service.budget_threshold_percentage = 80.0
        self.embedding_service.response_time_threshold = 10.0
        
        # Configure Knowledge Base circuit breaker
        self.knowledge_base.failure_threshold = 8
        self.knowledge_base.memory_threshold_gb = 1.5
        self.knowledge_base.response_time_threshold = 25.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'enabled': self.enabled,
            'orchestrator_enabled': self.orchestrator_enabled,
            'monitoring_enabled': self.monitoring_enabled,
            'metrics_collection_enabled': self.metrics_collection_enabled,
            'monitoring_interval_seconds': self.monitoring_interval_seconds,
            'alert_enabled': self.alert_enabled,
            'alert_cooldown_seconds': self.alert_cooldown_seconds,
            'openai_api': self.openai_api.to_dict(),
            'perplexity_api': self.perplexity_api.to_dict(),
            'lightrag': self.lightrag.to_dict(),
            'cache': self.cache.to_dict(),
            'embedding_service': self.embedding_service.to_dict(),
            'knowledge_base': self.knowledge_base.to_dict(),
            'integrate_with_cost_based': self.integrate_with_cost_based,
            'integrate_with_fallback_system': self.integrate_with_fallback_system,
            'integrate_with_load_balancer': self.integrate_with_load_balancer,
            'enable_failure_correlation': self.enable_failure_correlation,
            'enable_cascading_failure_prevention': self.enable_cascading_failure_prevention,
            'enable_system_health_assessment': self.enable_system_health_assessment,
        }


class ConfigurationValidator:
    """Validates enhanced circuit breaker configuration values."""
    
    @staticmethod
    def validate_service_config(config: ServiceCircuitBreakerConfig, service_name: str) -> bool:
        """Validate a service-specific configuration."""
        errors = []
        
        # Validate thresholds
        if config.failure_threshold <= 0:
            errors.append(f"{service_name}: failure_threshold must be positive")
        
        if config.recovery_timeout <= 0:
            errors.append(f"{service_name}: recovery_timeout must be positive")
        
        if config.degradation_threshold >= config.failure_threshold:
            errors.append(f"{service_name}: degradation_threshold must be less than failure_threshold")
        
        if not (0 < config.budget_threshold_percentage <= 100):
            errors.append(f"{service_name}: budget_threshold_percentage must be between 0 and 100")
        
        if config.memory_threshold_gb <= 0:
            errors.append(f"{service_name}: memory_threshold_gb must be positive")
        
        if config.response_time_threshold <= 0:
            errors.append(f"{service_name}: response_time_threshold must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation errors: {'; '.join(errors)}")
        
        return True
    
    @staticmethod
    def validate_config(config: EnhancedCircuitBreakerConfig) -> bool:
        """Validate the complete enhanced circuit breaker configuration."""
        # Validate global settings
        if config.monitoring_interval_seconds <= 0:
            raise ValueError("monitoring_interval_seconds must be positive")
        
        if config.alert_cooldown_seconds <= 0:
            raise ValueError("alert_cooldown_seconds must be positive")
        
        # Validate service configurations
        services = {
            'openai_api': config.openai_api,
            'perplexity_api': config.perplexity_api,
            'lightrag': config.lightrag,
            'cache': config.cache,
            'embedding_service': config.embedding_service,
            'knowledge_base': config.knowledge_base,
        }
        
        for service_name, service_config in services.items():
            ConfigurationValidator.validate_service_config(service_config, service_name)
        
        return True


class EnvironmentConfigurationLoader:
    """Loads configuration values from environment variables."""
    
    PREFIX = "ENHANCED_CB_"
    
    @classmethod
    def load_from_environment(cls) -> EnhancedCircuitBreakerConfig:
        """Load configuration from environment variables."""
        config = EnhancedCircuitBreakerConfig()
        
        # Load global settings
        config.enabled = cls._get_bool_env(f"{cls.PREFIX}ENABLED", config.enabled)
        config.orchestrator_enabled = cls._get_bool_env(f"{cls.PREFIX}ORCHESTRATOR_ENABLED", config.orchestrator_enabled)
        config.monitoring_enabled = cls._get_bool_env(f"{cls.PREFIX}MONITORING_ENABLED", config.monitoring_enabled)
        config.metrics_collection_enabled = cls._get_bool_env(f"{cls.PREFIX}METRICS_ENABLED", config.metrics_collection_enabled)
        
        # Load monitoring settings
        config.monitoring_interval_seconds = cls._get_int_env(f"{cls.PREFIX}MONITORING_INTERVAL", config.monitoring_interval_seconds)
        config.alert_enabled = cls._get_bool_env(f"{cls.PREFIX}ALERT_ENABLED", config.alert_enabled)
        config.alert_cooldown_seconds = cls._get_int_env(f"{cls.PREFIX}ALERT_COOLDOWN", config.alert_cooldown_seconds)
        
        # Load service-specific configurations
        cls._load_service_config_from_env("OPENAI", config.openai_api)
        cls._load_service_config_from_env("PERPLEXITY", config.perplexity_api)
        cls._load_service_config_from_env("LIGHTRAG", config.lightrag)
        cls._load_service_config_from_env("CACHE", config.cache)
        cls._load_service_config_from_env("EMBEDDING", config.embedding_service)
        cls._load_service_config_from_env("KNOWLEDGE_BASE", config.knowledge_base)
        
        # Load integration settings
        config.integrate_with_cost_based = cls._get_bool_env(f"{cls.PREFIX}INTEGRATE_COST_BASED", config.integrate_with_cost_based)
        config.integrate_with_fallback_system = cls._get_bool_env(f"{cls.PREFIX}INTEGRATE_FALLBACK", config.integrate_with_fallback_system)
        config.integrate_with_load_balancer = cls._get_bool_env(f"{cls.PREFIX}INTEGRATE_LOAD_BALANCER", config.integrate_with_load_balancer)
        
        # Load advanced features
        config.enable_failure_correlation = cls._get_bool_env(f"{cls.PREFIX}FAILURE_CORRELATION", config.enable_failure_correlation)
        config.enable_cascading_failure_prevention = cls._get_bool_env(f"{cls.PREFIX}CASCADING_PREVENTION", config.enable_cascading_failure_prevention)
        config.enable_system_health_assessment = cls._get_bool_env(f"{cls.PREFIX}HEALTH_ASSESSMENT", config.enable_system_health_assessment)
        
        return config
    
    @classmethod
    def _load_service_config_from_env(cls, service_prefix: str, service_config: ServiceCircuitBreakerConfig) -> None:
        """Load service-specific configuration from environment variables."""
        prefix = f"{cls.PREFIX}{service_prefix}_"
        
        service_config.failure_threshold = cls._get_int_env(f"{prefix}FAILURE_THRESHOLD", service_config.failure_threshold)
        service_config.recovery_timeout = cls._get_float_env(f"{prefix}RECOVERY_TIMEOUT", service_config.recovery_timeout)
        service_config.degradation_threshold = cls._get_int_env(f"{prefix}DEGRADATION_THRESHOLD", service_config.degradation_threshold)
        service_config.half_open_max_calls = cls._get_int_env(f"{prefix}HALF_OPEN_MAX_CALLS", service_config.half_open_max_calls)
        
        service_config.rate_limit_threshold = cls._get_int_env(f"{prefix}RATE_LIMIT_THRESHOLD", service_config.rate_limit_threshold)
        service_config.budget_threshold_percentage = cls._get_float_env(f"{prefix}BUDGET_THRESHOLD", service_config.budget_threshold_percentage)
        service_config.memory_threshold_gb = cls._get_float_env(f"{prefix}MEMORY_THRESHOLD", service_config.memory_threshold_gb)
        service_config.response_time_threshold = cls._get_float_env(f"{prefix}RESPONSE_TIME_THRESHOLD", service_config.response_time_threshold)
        
        service_config.enable_adaptive_thresholds = cls._get_bool_env(f"{prefix}ADAPTIVE_THRESHOLDS", service_config.enable_adaptive_thresholds)
        service_config.enable_progressive_degradation = cls._get_bool_env(f"{prefix}PROGRESSIVE_DEGRADATION", service_config.enable_progressive_degradation)
        service_config.enable_automatic_recovery = cls._get_bool_env(f"{prefix}AUTOMATIC_RECOVERY", service_config.enable_automatic_recovery)
        service_config.enable_cross_service_coordination = cls._get_bool_env(f"{prefix}CROSS_SERVICE_COORDINATION", service_config.enable_cross_service_coordination)
    
    @staticmethod
    def _get_bool_env(var_name: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.environ.get(var_name)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def _get_int_env(var_name: str, default: int) -> int:
        """Get integer value from environment variable."""
        value = os.environ.get(var_name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logging.warning(f"Invalid integer value for {var_name}: {value}, using default: {default}")
            return default
    
    @staticmethod
    def _get_float_env(var_name: str, default: float) -> float:
        """Get float value from environment variable."""
        value = os.environ.get(var_name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logging.warning(f"Invalid float value for {var_name}: {value}, using default: {default}")
            return default


class ConfigurationLoader:
    """Loads configuration from various sources with precedence ordering."""
    
    @staticmethod
    def load_configuration(
        config_file: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
        use_environment: bool = True
    ) -> EnhancedCircuitBreakerConfig:
        """
        Load configuration with precedence: override_config > environment > config_file > defaults.
        
        Args:
            config_file: Path to JSON configuration file
            override_config: Dictionary of configuration overrides
            use_environment: Whether to load from environment variables
            
        Returns:
            EnhancedCircuitBreakerConfig: Loaded and validated configuration
        """
        # Start with defaults
        config = EnhancedCircuitBreakerConfig()
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config = ConfigurationLoader._apply_dict_to_config(config, file_config)
                logging.info(f"Loaded enhanced circuit breaker configuration from {config_file}")
            except Exception as e:
                logging.warning(f"Failed to load config file {config_file}: {e}")
        
        # Load from environment variables
        if use_environment:
            try:
                env_config = EnvironmentConfigurationLoader.load_from_environment()
                # Merge environment config with existing config
                config = ConfigurationLoader._merge_configs(config, env_config)
                logging.info("Loaded enhanced circuit breaker configuration from environment variables")
            except Exception as e:
                logging.warning(f"Failed to load environment configuration: {e}")
        
        # Apply overrides
        if override_config:
            config = ConfigurationLoader._apply_dict_to_config(config, override_config)
            logging.info("Applied configuration overrides")
        
        # Validate final configuration
        ConfigurationValidator.validate_config(config)
        
        return config
    
    @staticmethod
    def _merge_configs(base_config: EnhancedCircuitBreakerConfig, 
                      override_config: EnhancedCircuitBreakerConfig) -> EnhancedCircuitBreakerConfig:
        """Merge two configuration objects, with override_config taking precedence."""
        # This is a simplified merge - in practice you might want more sophisticated merging
        merged_dict = base_config.to_dict()
        override_dict = override_config.to_dict()
        
        # Deep merge the dictionaries
        def deep_merge(base: Dict, override: Dict) -> Dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged_dict = deep_merge(merged_dict, override_dict)
        return ConfigurationLoader._apply_dict_to_config(EnhancedCircuitBreakerConfig(), merged_dict)
    
    @staticmethod
    def _apply_dict_to_config(config: EnhancedCircuitBreakerConfig, 
                             config_dict: Dict[str, Any]) -> EnhancedCircuitBreakerConfig:
        """Apply dictionary values to configuration object."""
        # Apply global settings
        for key in ['enabled', 'orchestrator_enabled', 'monitoring_enabled', 'metrics_collection_enabled',
                   'monitoring_interval_seconds', 'alert_enabled', 'alert_cooldown_seconds',
                   'integrate_with_cost_based', 'integrate_with_fallback_system', 'integrate_with_load_balancer',
                   'enable_failure_correlation', 'enable_cascading_failure_prevention', 'enable_system_health_assessment']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # Apply service-specific settings
        services = ['openai_api', 'perplexity_api', 'lightrag', 'cache', 'embedding_service', 'knowledge_base']
        for service in services:
            if service in config_dict and isinstance(config_dict[service], dict):
                service_config = getattr(config, service)
                for key, value in config_dict[service].items():
                    if hasattr(service_config, key):
                        setattr(service_config, key, value)
        
        return config


def create_default_enhanced_circuit_breaker_config() -> EnhancedCircuitBreakerConfig:
    """Create a default enhanced circuit breaker configuration."""
    return EnhancedCircuitBreakerConfig()


def load_enhanced_circuit_breaker_config(
    config_file: Optional[str] = None,
    environment_prefix: str = "ENHANCED_CB_"
) -> EnhancedCircuitBreakerConfig:
    """
    Load enhanced circuit breaker configuration from file and environment.
    
    This is a convenience function that provides the most common configuration loading pattern.
    """
    return ConfigurationLoader.load_configuration(
        config_file=config_file,
        use_environment=True
    )


# Example configuration for documentation/testing
def create_example_config() -> EnhancedCircuitBreakerConfig:
    """Create an example configuration showing all available options."""
    config = EnhancedCircuitBreakerConfig()
    
    # Customize some example values
    config.monitoring_interval_seconds = 30
    config.alert_cooldown_seconds = 600  # 10 minutes
    
    # Example OpenAI API configuration
    config.openai_api.failure_threshold = 3
    config.openai_api.recovery_timeout = 30.0
    config.openai_api.rate_limit_threshold = 5
    config.openai_api.budget_threshold_percentage = 85.0
    
    # Example LightRAG configuration
    config.lightrag.failure_threshold = 5
    config.lightrag.memory_threshold_gb = 1.5
    config.lightrag.response_time_threshold = 45.0
    
    return config


if __name__ == "__main__":
    # Example usage
    import pprint
    
    # Create default config
    config = create_default_enhanced_circuit_breaker_config()
    print("Default Configuration:")
    pprint.pprint(config.to_dict())
    
    # Load from environment (if variables are set)
    try:
        env_config = load_enhanced_circuit_breaker_config()
        print("\nConfiguration with Environment Variables:")
        pprint.pprint(env_config.to_dict())
    except Exception as e:
        print(f"Environment loading failed: {e}")