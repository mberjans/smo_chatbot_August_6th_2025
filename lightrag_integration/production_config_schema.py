"""
Production Configuration Schema for Load Balancer
===============================================

This module provides configuration schemas, validation, and factory functions
for the production load balancer system. It includes example configurations
for different deployment scenarios and validation utilities.

Author: Claude Code Assistant
Date: August 2025
Version: 1.0.0
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from .production_load_balancer import (
    ProductionLoadBalancingConfig,
    BackendInstanceConfig,
    BackendType,
    LoadBalancingStrategy
)


# ============================================================================
# Configuration Validation
# ============================================================================

class ConfigurationError(Exception):
    """Configuration validation error"""
    pass


class ConfigurationValidator:
    """Validates production load balancer configuration"""
    
    @staticmethod
    def validate_backend_config(config: BackendInstanceConfig) -> List[str]:
        """Validate individual backend configuration"""
        errors = []
        
        # Required fields
        if not config.id:
            errors.append("Backend ID is required")
            
        if not config.endpoint_url:
            errors.append("Endpoint URL is required")
            
        if not config.api_key:
            errors.append("API key is required")
            
        # Value validation
        if config.weight <= 0:
            errors.append("Weight must be positive")
            
        if config.cost_per_1k_tokens < 0:
            errors.append("Cost per 1k tokens cannot be negative")
            
        if config.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
            
        if not (0 <= config.quality_score <= 1.0):
            errors.append("Quality score must be between 0 and 1")
            
        if not (0 <= config.reliability_score <= 1.0):
            errors.append("Reliability score must be between 0 and 1")
            
        # Circuit breaker validation
        if config.circuit_breaker_enabled:
            if config.failure_threshold <= 0:
                errors.append("Failure threshold must be positive")
                
            if config.recovery_timeout_seconds <= 0:
                errors.append("Recovery timeout must be positive")
                
        return errors
    
    @staticmethod
    def validate_load_balancing_config(config: ProductionLoadBalancingConfig) -> List[str]:
        """Validate complete load balancing configuration"""
        errors = []
        
        # Backend instances validation
        if not config.backend_instances:
            errors.append("At least one backend instance is required")
            
        # Validate each backend
        for instance_id, backend_config in config.backend_instances.items():
            backend_errors = ConfigurationValidator.validate_backend_config(backend_config)
            for error in backend_errors:
                errors.append(f"Backend '{instance_id}': {error}")
                
        # Strategy validation
        if config.strategy not in LoadBalancingStrategy:
            errors.append(f"Invalid load balancing strategy: {config.strategy}")
            
        # Timing validation
        if config.routing_decision_timeout_ms <= 0:
            errors.append("Routing decision timeout must be positive")
            
        # Cost optimization validation
        if config.enable_cost_optimization:
            if not (0 < config.cost_optimization_target <= 1.0):
                errors.append("Cost optimization target must be between 0 and 1")
                
        # Quality validation
        if config.enable_quality_based_routing:
            if not (0 <= config.minimum_quality_threshold <= 1.0):
                errors.append("Minimum quality threshold must be between 0 and 1")
                
        # Adaptive learning validation
        if config.enable_adaptive_routing:
            if not (0 < config.learning_rate <= 1.0):
                errors.append("Learning rate must be between 0 and 1")
                
        return errors
    
    @staticmethod
    def validate_and_raise(config: ProductionLoadBalancingConfig):
        """Validate configuration and raise exception if invalid"""
        errors = ConfigurationValidator.validate_load_balancing_config(config)
        if errors:
            raise ConfigurationError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))


# ============================================================================
# Configuration Factory Functions
# ============================================================================

class ConfigurationFactory:
    """Factory for creating production configurations"""
    
    @staticmethod
    def create_development_config() -> ProductionLoadBalancingConfig:
        """Create configuration suitable for development environment"""
        
        return ProductionLoadBalancingConfig(
            strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
            
            backend_instances={
                "dev_lightrag": BackendInstanceConfig(
                    id="dev_lightrag",
                    backend_type=BackendType.LIGHTRAG,
                    endpoint_url="http://localhost:8080",
                    api_key="dev_key",
                    weight=2.0,  # Prefer local service
                    cost_per_1k_tokens=0.01,  # Very low cost for dev
                    timeout_seconds=30.0,
                    expected_response_time_ms=500.0,
                    quality_score=0.85,
                    failure_threshold=10,  # More lenient for dev
                    recovery_timeout_seconds=30
                ),
                
                "dev_perplexity": BackendInstanceConfig(
                    id="dev_perplexity", 
                    backend_type=BackendType.PERPLEXITY,
                    endpoint_url="https://api.perplexity.ai",
                    api_key=os.getenv("PERPLEXITY_API_KEY", "dev_key"),
                    weight=0.5,  # Lower weight for external API in dev
                    cost_per_1k_tokens=0.20,
                    timeout_seconds=45.0,
                    expected_response_time_ms=2000.0,
                    quality_score=0.80,
                    failure_threshold=5,
                    recovery_timeout_seconds=60
                )
            },
            
            # Development-friendly settings
            enable_adaptive_routing=False,  # Keep weights manual
            enable_cost_optimization=False,  # No cost pressure in dev
            enable_quality_based_routing=True,
            enable_real_time_monitoring=False,  # Reduce complexity
            
            routing_decision_timeout_ms=100.0,  # More generous timeout
            max_concurrent_health_checks=5,
            health_check_batch_size=2,
            
            cost_optimization_target=0.9,  # Very lenient
            minimum_quality_threshold=0.6,  # Lower bar for dev
            
            learning_rate=0.1,  # Faster learning for dev
            weight_adjustment_frequency_minutes=30  # More frequent adjustments
        )
    
    @staticmethod
    def create_staging_config() -> ProductionLoadBalancingConfig:
        """Create configuration suitable for staging environment"""
        
        return ProductionLoadBalancingConfig(
            strategy=LoadBalancingStrategy.ADAPTIVE_LEARNING,
            
            backend_instances={
                "staging_lightrag_primary": BackendInstanceConfig(
                    id="staging_lightrag_primary",
                    backend_type=BackendType.LIGHTRAG,
                    endpoint_url="http://lightrag-staging:8080",
                    api_key=os.getenv("LIGHTRAG_API_KEY", "staging_key"),
                    weight=1.5,
                    cost_per_1k_tokens=0.05,
                    timeout_seconds=25.0,
                    expected_response_time_ms=800.0,
                    quality_score=0.88,
                    failure_threshold=5,
                    recovery_timeout_seconds=45
                ),
                
                "staging_lightrag_secondary": BackendInstanceConfig(
                    id="staging_lightrag_secondary",
                    backend_type=BackendType.LIGHTRAG,
                    endpoint_url="http://lightrag-staging-2:8080",
                    api_key=os.getenv("LIGHTRAG_API_KEY", "staging_key"),
                    weight=1.2,  # Slightly lower weight
                    cost_per_1k_tokens=0.05,
                    timeout_seconds=25.0,
                    expected_response_time_ms=900.0,
                    quality_score=0.85,
                    failure_threshold=5,
                    recovery_timeout_seconds=45,
                    priority=2  # Secondary priority
                ),
                
                "staging_perplexity": BackendInstanceConfig(
                    id="staging_perplexity",
                    backend_type=BackendType.PERPLEXITY,
                    endpoint_url="https://api.perplexity.ai",
                    api_key=os.getenv("PERPLEXITY_API_KEY"),
                    weight=1.0,
                    cost_per_1k_tokens=0.20,
                    timeout_seconds=35.0,
                    expected_response_time_ms=2000.0,
                    quality_score=0.82,
                    failure_threshold=3,
                    recovery_timeout_seconds=60
                )
            },
            
            # Staging settings (closer to production)
            enable_adaptive_routing=True,
            enable_cost_optimization=True,
            enable_quality_based_routing=True,
            enable_real_time_monitoring=True,
            
            routing_decision_timeout_ms=75.0,
            max_concurrent_health_checks=8,
            health_check_batch_size=4,
            
            cost_optimization_target=0.85,
            minimum_quality_threshold=0.7,
            
            learning_rate=0.02,  # Conservative learning
            weight_adjustment_frequency_minutes=20,
            
            # Monitoring
            enable_prometheus_metrics=True,
            enable_grafana_dashboards=True
        )
    
    @staticmethod
    def create_production_config() -> ProductionLoadBalancingConfig:
        """Create configuration for production environment"""
        
        return ProductionLoadBalancingConfig(
            strategy=LoadBalancingStrategy.ADAPTIVE_LEARNING,
            
            backend_instances={
                # Primary LightRAG cluster
                "prod_lightrag_primary": BackendInstanceConfig(
                    id="prod_lightrag_primary",
                    backend_type=BackendType.LIGHTRAG,
                    endpoint_url=os.getenv("LIGHTRAG_PRIMARY_URL", "http://lightrag-prod-1:8080"),
                    api_key=os.getenv("LIGHTRAG_API_KEY"),
                    weight=2.0,  # Highest weight for primary
                    cost_per_1k_tokens=0.05,
                    max_requests_per_minute=300,
                    timeout_seconds=20.0,
                    expected_response_time_ms=600.0,
                    quality_score=0.92,
                    reliability_score=0.98,
                    priority=1,
                    failure_threshold=3,  # Strict for production
                    recovery_timeout_seconds=30
                ),
                
                "prod_lightrag_secondary": BackendInstanceConfig(
                    id="prod_lightrag_secondary",
                    backend_type=BackendType.LIGHTRAG,
                    endpoint_url=os.getenv("LIGHTRAG_SECONDARY_URL", "http://lightrag-prod-2:8080"),
                    api_key=os.getenv("LIGHTRAG_API_KEY"),
                    weight=1.8,
                    cost_per_1k_tokens=0.05,
                    max_requests_per_minute=250,
                    timeout_seconds=20.0,
                    expected_response_time_ms=700.0,
                    quality_score=0.90,
                    reliability_score=0.96,
                    priority=1,
                    failure_threshold=3,
                    recovery_timeout_seconds=30
                ),
                
                "prod_lightrag_tertiary": BackendInstanceConfig(
                    id="prod_lightrag_tertiary",
                    backend_type=BackendType.LIGHTRAG,
                    endpoint_url=os.getenv("LIGHTRAG_TERTIARY_URL", "http://lightrag-prod-3:8080"),
                    api_key=os.getenv("LIGHTRAG_API_KEY"),
                    weight=1.5,
                    cost_per_1k_tokens=0.05,
                    max_requests_per_minute=200,
                    timeout_seconds=20.0,
                    expected_response_time_ms=800.0,
                    quality_score=0.88,
                    reliability_score=0.94,
                    priority=2,  # Lower priority backup
                    failure_threshold=3,
                    recovery_timeout_seconds=30
                ),
                
                # Perplexity instances
                "prod_perplexity_primary": BackendInstanceConfig(
                    id="prod_perplexity_primary",
                    backend_type=BackendType.PERPLEXITY,
                    endpoint_url="https://api.perplexity.ai",
                    api_key=os.getenv("PERPLEXITY_API_KEY_PRIMARY"),
                    weight=1.0,
                    cost_per_1k_tokens=0.20,
                    max_requests_per_minute=100,
                    timeout_seconds=30.0,
                    expected_response_time_ms=1800.0,
                    quality_score=0.86,
                    reliability_score=0.92,
                    priority=1,
                    failure_threshold=2,  # Very strict for external API
                    recovery_timeout_seconds=60
                ),
                
                "prod_perplexity_secondary": BackendInstanceConfig(
                    id="prod_perplexity_secondary",
                    backend_type=BackendType.PERPLEXITY,
                    endpoint_url="https://api.perplexity.ai",
                    api_key=os.getenv("PERPLEXITY_API_KEY_SECONDARY"),
                    weight=0.8,  # Backup instance
                    cost_per_1k_tokens=0.20,
                    max_requests_per_minute=80,
                    timeout_seconds=30.0,
                    expected_response_time_ms=2000.0,
                    quality_score=0.86,
                    reliability_score=0.90,
                    priority=2,
                    failure_threshold=2,
                    recovery_timeout_seconds=60
                )
            },
            
            # Production settings
            enable_adaptive_routing=True,
            enable_cost_optimization=True,
            enable_quality_based_routing=True,
            enable_real_time_monitoring=True,
            
            # Performance optimization
            routing_decision_timeout_ms=50.0,  # Aggressive timeout
            max_concurrent_health_checks=10,
            health_check_batch_size=5,
            
            # Cost management
            cost_optimization_target=0.8,  # 80% efficiency target
            cost_tracking_window_hours=24,
            
            # Quality assurance
            minimum_quality_threshold=0.75,  # High quality bar
            quality_sampling_rate=0.1,  # 10% sampling
            
            # Adaptive learning
            learning_rate=0.01,  # Conservative learning
            performance_history_window_hours=168,  # 1 week
            weight_adjustment_frequency_minutes=15,  # Frequent adjustments
            
            # Monitoring and alerting
            enable_prometheus_metrics=True,
            enable_grafana_dashboards=True,
            alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
            alert_email_recipients=os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(","),
            
            # Circuit breaker global settings
            global_circuit_breaker_enabled=True,
            cascade_failure_prevention=True
        )
    
    @staticmethod
    def create_high_availability_config() -> ProductionLoadBalancingConfig:
        """Create high-availability configuration with multiple regions"""
        
        instances = {}
        
        # US East region
        for i in range(3):
            instances[f"us_east_lightrag_{i+1}"] = BackendInstanceConfig(
                id=f"us_east_lightrag_{i+1}",
                backend_type=BackendType.LIGHTRAG,
                endpoint_url=f"http://lightrag-us-east-{i+1}:8080",
                api_key=os.getenv("LIGHTRAG_API_KEY"),
                weight=2.0 - (i * 0.1),  # Decreasing weights
                cost_per_1k_tokens=0.05,
                expected_response_time_ms=600.0 + (i * 50),
                quality_score=0.92 - (i * 0.01),
                priority=i + 1
            )
            
        # US West region (backup)
        for i in range(2):
            instances[f"us_west_lightrag_{i+1}"] = BackendInstanceConfig(
                id=f"us_west_lightrag_{i+1}",
                backend_type=BackendType.LIGHTRAG,
                endpoint_url=f"http://lightrag-us-west-{i+1}:8080",
                api_key=os.getenv("LIGHTRAG_API_KEY"),
                weight=1.5 - (i * 0.1),
                cost_per_1k_tokens=0.06,  # Slightly higher for cross-region
                expected_response_time_ms=800.0 + (i * 50),
                quality_score=0.90 - (i * 0.01),
                priority=4 + i  # Lower priority than primary region
            )
            
        # Multiple Perplexity accounts for redundancy
        for i in range(3):
            instances[f"perplexity_account_{i+1}"] = BackendInstanceConfig(
                id=f"perplexity_account_{i+1}",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url="https://api.perplexity.ai",
                api_key=os.getenv(f"PERPLEXITY_API_KEY_{i+1}"),
                weight=1.0 - (i * 0.1),
                cost_per_1k_tokens=0.20,
                expected_response_time_ms=1800.0 + (i * 100),
                quality_score=0.86 - (i * 0.01),
                priority=1 + i
            )
        
        return ProductionLoadBalancingConfig(
            strategy=LoadBalancingStrategy.ADAPTIVE_LEARNING,
            backend_instances=instances,
            
            # High availability settings
            enable_adaptive_routing=True,
            enable_cost_optimization=True,
            enable_quality_based_routing=True,
            enable_real_time_monitoring=True,
            
            # Aggressive health checking for HA
            max_concurrent_health_checks=15,
            health_check_batch_size=8,
            
            # Strict quality and cost controls
            cost_optimization_target=0.75,
            minimum_quality_threshold=0.8,
            
            # Conservative learning for stability
            learning_rate=0.005,
            weight_adjustment_frequency_minutes=30,
            
            # Full monitoring
            enable_prometheus_metrics=True,
            enable_grafana_dashboards=True,
            global_circuit_breaker_enabled=True,
            cascade_failure_prevention=True
        )


# ============================================================================
# Configuration File Handlers
# ============================================================================

class ConfigurationFileHandler:
    """Handles loading and saving configuration files"""
    
    @staticmethod
    def load_from_yaml(file_path: Union[str, Path]) -> ProductionLoadBalancingConfig:
        """Load configuration from YAML file"""
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            return ConfigurationFileHandler._dict_to_config(data)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {file_path}: {e}")
    
    @staticmethod
    def load_from_json(file_path: Union[str, Path]) -> ProductionLoadBalancingConfig:
        """Load configuration from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return ConfigurationFileHandler._dict_to_config(data)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {file_path}: {e}")
    
    @staticmethod
    def save_to_yaml(config: ProductionLoadBalancingConfig, file_path: Union[str, Path]):
        """Save configuration to YAML file"""
        try:
            data = ConfigurationFileHandler._config_to_dict(config)
            
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {file_path}: {e}")
    
    @staticmethod
    def save_to_json(config: ProductionLoadBalancingConfig, file_path: Union[str, Path]):
        """Save configuration to JSON file"""
        try:
            data = ConfigurationFileHandler._config_to_dict(config)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {file_path}: {e}")
    
    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> ProductionLoadBalancingConfig:
        """Convert dictionary to configuration object"""
        
        # Parse backend instances
        backend_instances = {}
        for instance_id, instance_data in data.get('backend_instances', {}).items():
            # Convert string backend type to enum
            backend_type = BackendType(instance_data['backend_type'])
            instance_data['backend_type'] = backend_type
            
            backend_instances[instance_id] = BackendInstanceConfig(**instance_data)
        
        # Parse main configuration
        config_data = data.copy()
        config_data['backend_instances'] = backend_instances
        
        # Convert strategy to enum
        if 'strategy' in config_data:
            config_data['strategy'] = LoadBalancingStrategy(config_data['strategy'])
        
        return ProductionLoadBalancingConfig(**config_data)
    
    @staticmethod
    def _config_to_dict(config: ProductionLoadBalancingConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        
        # Convert to dict using dataclass utility
        data = asdict(config)
        
        # Convert enums to strings for serialization
        data['strategy'] = config.strategy.value
        
        for instance_id, instance_data in data['backend_instances'].items():
            instance_data['backend_type'] = instance_data['backend_type']
        
        return data


# ============================================================================
# Environment-based Configuration
# ============================================================================

class EnvironmentConfigurationBuilder:
    """Builds configuration from environment variables"""
    
    @staticmethod
    def build_from_environment() -> ProductionLoadBalancingConfig:
        """Build configuration from environment variables"""
        
        env = os.getenv('ENVIRONMENT', 'development').lower()
        
        if env == 'production':
            config = ConfigurationFactory.create_production_config()
        elif env == 'staging':
            config = ConfigurationFactory.create_staging_config()
        else:
            config = ConfigurationFactory.create_development_config()
        
        # Override with environment-specific settings
        config = EnvironmentConfigurationBuilder._apply_environment_overrides(config)
        
        return config
    
    @staticmethod
    def _apply_environment_overrides(config: ProductionLoadBalancingConfig) -> ProductionLoadBalancingConfig:
        """Apply environment variable overrides"""
        
        # Strategy override
        strategy_env = os.getenv('PRODUCTION_LB_STRATEGY')
        if strategy_env:
            try:
                config.strategy = LoadBalancingStrategy(strategy_env)
            except ValueError:
                pass  # Keep default strategy
        
        # Feature flag overrides
        if os.getenv('ENABLE_ADAPTIVE_ROUTING') is not None:
            config.enable_adaptive_routing = os.getenv('ENABLE_ADAPTIVE_ROUTING').lower() == 'true'
            
        if os.getenv('ENABLE_COST_OPTIMIZATION') is not None:
            config.enable_cost_optimization = os.getenv('ENABLE_COST_OPTIMIZATION').lower() == 'true'
            
        if os.getenv('ENABLE_QUALITY_ROUTING') is not None:
            config.enable_quality_based_routing = os.getenv('ENABLE_QUALITY_ROUTING').lower() == 'true'
        
        # Numeric overrides
        if os.getenv('ROUTING_TIMEOUT_MS'):
            config.routing_decision_timeout_ms = float(os.getenv('ROUTING_TIMEOUT_MS'))
            
        if os.getenv('COST_TARGET'):
            config.cost_optimization_target = float(os.getenv('COST_TARGET'))
            
        if os.getenv('QUALITY_THRESHOLD'):
            config.minimum_quality_threshold = float(os.getenv('QUALITY_THRESHOLD'))
            
        if os.getenv('LEARNING_RATE'):
            config.learning_rate = float(os.getenv('LEARNING_RATE'))
        
        # Alert configuration
        if os.getenv('ALERT_WEBHOOK_URL'):
            config.alert_webhook_url = os.getenv('ALERT_WEBHOOK_URL')
            
        if os.getenv('ALERT_EMAIL_RECIPIENTS'):
            config.alert_email_recipients = os.getenv('ALERT_EMAIL_RECIPIENTS').split(',')
        
        return config


# ============================================================================
# Configuration Templates
# ============================================================================

def generate_configuration_templates():
    """Generate example configuration files"""
    
    templates_dir = Path(__file__).parent / "config_templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Development template
    dev_config = ConfigurationFactory.create_development_config()
    ConfigurationFileHandler.save_to_yaml(dev_config, templates_dir / "development.yaml")
    ConfigurationFileHandler.save_to_json(dev_config, templates_dir / "development.json")
    
    # Staging template
    staging_config = ConfigurationFactory.create_staging_config()
    ConfigurationFileHandler.save_to_yaml(staging_config, templates_dir / "staging.yaml")
    
    # Production template
    prod_config = ConfigurationFactory.create_production_config()
    ConfigurationFileHandler.save_to_yaml(prod_config, templates_dir / "production.yaml")
    
    # High availability template
    ha_config = ConfigurationFactory.create_high_availability_config()
    ConfigurationFileHandler.save_to_yaml(ha_config, templates_dir / "high_availability.yaml")
    
    print(f"Configuration templates generated in: {templates_dir}")


# ============================================================================
# Configuration Management Utilities
# ============================================================================

class ConfigurationManager:
    """Manages configuration lifecycle"""
    
    def __init__(self, config: ProductionLoadBalancingConfig):
        self.config = config
        self.validator = ConfigurationValidator()
    
    def validate(self) -> bool:
        """Validate current configuration"""
        try:
            self.validator.validate_and_raise(self.config)
            return True
        except ConfigurationError:
            return False
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors without raising exception"""
        return self.validator.validate_load_balancing_config(self.config)
    
    def update_backend_config(self, instance_id: str, updates: Dict[str, Any]):
        """Update configuration for specific backend"""
        if instance_id not in self.config.backend_instances:
            raise ConfigurationError(f"Backend instance not found: {instance_id}")
        
        backend_config = self.config.backend_instances[instance_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(backend_config, key):
                setattr(backend_config, key, value)
            else:
                raise ConfigurationError(f"Invalid configuration key: {key}")
        
        # Validate updated configuration
        errors = self.validator.validate_backend_config(backend_config)
        if errors:
            raise ConfigurationError(f"Invalid backend configuration: {errors}")
    
    def add_backend_instance(self, instance_id: str, config: BackendInstanceConfig):
        """Add new backend instance"""
        if instance_id in self.config.backend_instances:
            raise ConfigurationError(f"Backend instance already exists: {instance_id}")
        
        # Validate new backend configuration
        errors = self.validator.validate_backend_config(config)
        if errors:
            raise ConfigurationError(f"Invalid backend configuration: {errors}")
        
        self.config.backend_instances[instance_id] = config
    
    def remove_backend_instance(self, instance_id: str):
        """Remove backend instance"""
        if instance_id not in self.config.backend_instances:
            raise ConfigurationError(f"Backend instance not found: {instance_id}")
        
        del self.config.backend_instances[instance_id]
        
        # Ensure we still have at least one backend
        if not self.config.backend_instances:
            raise ConfigurationError("Cannot remove last backend instance")
    
    def get_backend_summary(self) -> Dict[str, Any]:
        """Get summary of backend configuration"""
        summary = {
            'total_backends': len(self.config.backend_instances),
            'backend_types': {},
            'total_weight': 0.0,
            'average_cost_per_1k': 0.0,
            'backends': []
        }
        
        total_cost = 0.0
        
        for instance_id, backend_config in self.config.backend_instances.items():
            backend_type = backend_config.backend_type.value
            summary['backend_types'][backend_type] = summary['backend_types'].get(backend_type, 0) + 1
            summary['total_weight'] += backend_config.weight
            total_cost += backend_config.cost_per_1k_tokens
            
            summary['backends'].append({
                'id': instance_id,
                'type': backend_type,
                'weight': backend_config.weight,
                'cost_per_1k': backend_config.cost_per_1k_tokens,
                'priority': backend_config.priority,
                'quality_score': backend_config.quality_score
            })
        
        if len(self.config.backend_instances) > 0:
            summary['average_cost_per_1k'] = total_cost / len(self.config.backend_instances)
        
        return summary


# ============================================================================
# Example Usage and Testing
# ============================================================================

def main():
    """Example usage of configuration system"""
    
    print("Production Load Balancer Configuration System")
    print("=" * 50)
    
    # Create development configuration
    print("\n1. Creating development configuration...")
    dev_config = ConfigurationFactory.create_development_config()
    
    # Validate configuration
    validator = ConfigurationValidator()
    errors = validator.validate_load_balancing_config(dev_config)
    
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Development configuration is valid")
    
    # Create configuration manager
    config_manager = ConfigurationManager(dev_config)
    summary = config_manager.get_backend_summary()
    
    print(f"✓ Configuration summary:")
    print(f"  - Total backends: {summary['total_backends']}")
    print(f"  - Backend types: {summary['backend_types']}")
    print(f"  - Total weight: {summary['total_weight']}")
    print(f"  - Average cost per 1K tokens: ${summary['average_cost_per_1k']:.4f}")
    
    # Generate configuration templates
    print("\n2. Generating configuration templates...")
    generate_configuration_templates()
    
    # Test environment-based configuration
    print("\n3. Testing environment-based configuration...")
    env_config = EnvironmentConfigurationBuilder.build_from_environment()
    env_errors = validator.validate_load_balancing_config(env_config)
    
    if env_errors:
        print(f"Environment configuration errors: {env_errors}")
    else:
        print("✓ Environment configuration is valid")
    
    print("\nConfiguration system test completed successfully!")


if __name__ == "__main__":
    main()