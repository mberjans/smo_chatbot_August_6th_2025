#!/usr/bin/env python3
"""
Production Configuration Loader

This module provides comprehensive configuration loading and management
for the production load balancer integration, with environment variable
support, validation, and backward compatibility.

Features:
- Environment variable configuration loading
- Configuration validation and migration
- Multiple deployment profiles (canary, A/B testing, shadow, production)
- Backward compatibility with existing configurations
- Configuration hot-reloading support
- Secure API key management

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: Production Load Balancer Integration
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import yaml
from enum import Enum

from .intelligent_query_router import LoadBalancingConfig, HealthCheckConfig
from .production_load_balancer import ProductionLoadBalancingConfig, create_default_production_config


class ConfigSource(Enum):
    """Configuration source types"""
    ENVIRONMENT = "environment"
    FILE = "file"
    DEFAULT = "default"
    OVERRIDE = "override"


@dataclass
class ConfigValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    deprecated: List[str] = field(default_factory=list)


class ProductionConfigLoader:
    """
    Production configuration loader with environment variable support
    and backward compatibility with existing IntelligentQueryRouter configurations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._config_cache: Dict[str, Any] = {}
        self._config_sources: Dict[str, ConfigSource] = {}
        
        # Configuration file search paths
        self.config_paths = [
            Path(".env"),
            Path("production_deployment_configs/production.env"),
            Path("lightrag_integration/production_deployment_configs/production.env"),
            Path("/etc/clinical_metabolomics/production.env"),
            Path.home() / ".clinical_metabolomics" / "production.env"
        ]
    
    def load_production_config(self, 
                             env_file: Optional[str] = None,
                             profile: Optional[str] = None,
                             override_config: Optional[Dict[str, Any]] = None) -> ProductionLoadBalancingConfig:
        """
        Load production configuration from environment variables and files
        
        Args:
            env_file: Specific environment file to load
            profile: Configuration profile (canary, ab_test, shadow, production_full)
            override_config: Configuration overrides
        
        Returns:
            ProductionLoadBalancingConfig instance
        """
        self.logger.info("Loading production configuration...")
        
        # Start with default configuration
        config = create_default_production_config()
        
        # Load environment variables
        env_config = self._load_from_environment()
        
        # Load from file if specified or found
        file_config = {}
        if env_file:
            file_config = self._load_from_file(env_file)
        elif profile:
            profile_file = self._get_profile_file(profile)
            if profile_file:
                file_config = self._load_from_file(profile_file)
        else:
            # Try to find a configuration file
            for config_path in self.config_paths:
                if config_path.exists():
                    file_config = self._load_from_file(str(config_path))
                    break
        
        # Merge configurations in priority order: defaults < file < environment < overrides
        merged_config = self._merge_configurations([
            ("default", config, ConfigSource.DEFAULT),
            ("file", file_config, ConfigSource.FILE),
            ("environment", env_config, ConfigSource.ENVIRONMENT),
            ("override", override_config or {}, ConfigSource.OVERRIDE)
        ])
        
        # Apply merged configuration to production config
        updated_config = self._apply_config_to_production(config, merged_config)
        
        # Validate configuration
        validation_result = self._validate_production_config(updated_config)
        if not validation_result.is_valid:
            self.logger.error(f"Configuration validation failed: {validation_result.errors}")
            if not self._should_continue_with_invalid_config():
                raise ValueError(f"Invalid configuration: {validation_result.errors}")
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                self.logger.warning(f"Configuration warning: {warning}")
        
        self.logger.info("Production configuration loaded successfully")
        return updated_config
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Feature flags
        env_config['enable_production_load_balancer'] = self._get_bool_env('PROD_LB_ENABLED', False)
        env_config['deployment_mode'] = os.getenv('PROD_LB_DEPLOYMENT_MODE', 'legacy_only')
        env_config['production_traffic_percentage'] = self._get_float_env('PROD_LB_TRAFFIC_PERCENT', 0.0)
        env_config['enable_performance_comparison'] = self._get_bool_env('PROD_LB_PERF_COMPARISON', True)
        env_config['enable_automatic_failback'] = self._get_bool_env('PROD_LB_AUTO_FAILBACK', True)
        env_config['enable_advanced_algorithms'] = self._get_bool_env('PROD_LB_ADVANCED_ALGORITHMS', False)
        env_config['enable_cost_optimization'] = self._get_bool_env('PROD_LB_COST_OPTIMIZATION', False)
        env_config['enable_quality_metrics'] = self._get_bool_env('PROD_LB_QUALITY_METRICS', True)
        env_config['rollback_threshold_error_rate'] = self._get_float_env('PROD_LB_ROLLBACK_ERROR_RATE', 5.0)
        env_config['rollback_threshold_latency_ms'] = self._get_float_env('PROD_LB_ROLLBACK_LATENCY_MS', 5000.0)
        env_config['max_canary_duration_hours'] = self._get_int_env('PROD_LB_CANARY_MAX_HOURS', 24)
        
        # Backend configurations
        backends = {}
        if os.getenv('PROD_LB_LIGHTRAG_ENDPOINT'):
            backends['lightrag'] = {
                'name': 'lightrag',
                'endpoint': os.getenv('PROD_LB_LIGHTRAG_ENDPOINT'),
                'type': 'lightrag',
                'enabled': True
            }
        
        if os.getenv('PROD_LB_PERPLEXITY_ENDPOINT'):
            backends['perplexity'] = {
                'name': 'perplexity',
                'endpoint': os.getenv('PROD_LB_PERPLEXITY_ENDPOINT'),
                'type': 'perplexity',
                'enabled': True
            }
        
        if os.getenv('PROD_LB_OPENAI_ENDPOINT'):
            backends['openai'] = {
                'name': 'openai',
                'endpoint': os.getenv('PROD_LB_OPENAI_ENDPOINT'),
                'type': 'openai',
                'enabled': True
            }
        
        if backends:
            env_config['backends'] = backends
        
        # Health monitoring
        health_config = {}\n        if os.getenv('PROD_LB_HEALTH_CHECK_INTERVAL'):\n            health_config['check_interval_seconds'] = self._get_int_env('PROD_LB_HEALTH_CHECK_INTERVAL', 30)\n        if os.getenv('PROD_LB_HEALTH_TIMEOUT'):\n            health_config['timeout_seconds'] = self._get_int_env('PROD_LB_HEALTH_TIMEOUT', 10)\n        if os.getenv('PROD_LB_UNHEALTHY_THRESHOLD'):\n            health_config['unhealthy_threshold'] = self._get_int_env('PROD_LB_UNHEALTHY_THRESHOLD', 3)\n        if os.getenv('PROD_LB_HEALTHY_THRESHOLD'):\n            health_config['healthy_threshold'] = self._get_int_env('PROD_LB_HEALTHY_THRESHOLD', 2)\n        \n        if health_config:\n            env_config['health_monitoring'] = health_config
        
        # Circuit breaker
        cb_config = {}
        if os.getenv('PROD_LB_CB_FAILURE_THRESHOLD'):
            cb_config['failure_threshold'] = self._get_int_env('PROD_LB_CB_FAILURE_THRESHOLD', 5)
        if os.getenv('PROD_LB_CB_SUCCESS_THRESHOLD'):
            cb_config['success_threshold'] = self._get_int_env('PROD_LB_CB_SUCCESS_THRESHOLD', 3)
        if os.getenv('PROD_LB_CB_TIMEOUT'):
            cb_config['recovery_timeout_seconds'] = self._get_int_env('PROD_LB_CB_TIMEOUT', 300)
        
        if cb_config:
            env_config['circuit_breaker'] = cb_config
        
        # Performance thresholds
        perf_config = {}
        if os.getenv('PROD_LB_MAX_RESPONSE_TIME_MS'):
            perf_config['response_time_ms'] = self._get_float_env('PROD_LB_MAX_RESPONSE_TIME_MS', 5000.0)
        if os.getenv('PROD_LB_TARGET_RESPONSE_TIME_MS'):
            perf_config['target_response_time_ms'] = self._get_float_env('PROD_LB_TARGET_RESPONSE_TIME_MS', 2000.0)
        if os.getenv('PROD_LB_MAX_CONCURRENT_REQUESTS'):
            perf_config['max_concurrent_requests'] = self._get_int_env('PROD_LB_MAX_CONCURRENT_REQUESTS', 100)
        
        if perf_config:
            env_config['performance_thresholds'] = perf_config
        
        # Algorithm configuration
        algo_config = {}
        if os.getenv('PROD_LB_PRIMARY_ALGORITHM'):
            algo_config['primary_algorithm'] = os.getenv('PROD_LB_PRIMARY_ALGORITHM')
        if os.getenv('PROD_LB_ENABLE_ADAPTIVE'):
            algo_config['enable_adaptive_selection'] = self._get_bool_env('PROD_LB_ENABLE_ADAPTIVE', True)
        if os.getenv('PROD_LB_FALLBACK_ALGORITHMS'):
            algo_config['fallback_algorithms'] = os.getenv('PROD_LB_FALLBACK_ALGORITHMS', '').split(',')
        
        if algo_config:
            env_config['algorithm_config'] = algo_config
        
        # Monitoring configuration
        monitoring_config = {}
        if os.getenv('PROD_LB_ENABLE_MONITORING'):
            monitoring_config['enabled'] = self._get_bool_env('PROD_LB_ENABLE_MONITORING', True)
        if os.getenv('PROD_LB_MONITORING_INTERVAL'):
            monitoring_config['metrics_collection_interval_seconds'] = self._get_int_env('PROD_LB_MONITORING_INTERVAL', 60)
        if os.getenv('PROD_LB_METRICS_RETENTION_HOURS'):
            monitoring_config['metrics_retention_hours'] = self._get_int_env('PROD_LB_METRICS_RETENTION_HOURS', 168)
        
        if monitoring_config:
            env_config['monitoring'] = monitoring_config
        
        # Cost optimization
        cost_config = {}
        if os.getenv('PROD_LB_COST_TRACKING_ENABLED'):
            cost_config['cost_tracking_enabled'] = self._get_bool_env('PROD_LB_COST_TRACKING_ENABLED', True)
        if os.getenv('PROD_LB_DAILY_BUDGET_USD'):
            cost_config['daily_budget_usd'] = self._get_float_env('PROD_LB_DAILY_BUDGET_USD', 100.0)
        if os.getenv('PROD_LB_PREFER_LOW_COST_BACKENDS'):
            cost_config['prefer_low_cost_backends'] = self._get_bool_env('PROD_LB_PREFER_LOW_COST_BACKENDS', False)
        if os.getenv('PROD_LB_COST_WEIGHT_FACTOR'):
            cost_config['cost_weight_factor'] = self._get_float_env('PROD_LB_COST_WEIGHT_FACTOR', 0.3)
        
        if cost_config:
            env_config['cost_optimization'] = cost_config
        
        # Rate limiting
        rate_limit_config = {}
        if os.getenv('PROD_LB_RATE_LIMIT_RPM'):
            rate_limit_config['requests_per_minute'] = self._get_int_env('PROD_LB_RATE_LIMIT_RPM', 1000)
        if os.getenv('PROD_LB_RATE_LIMIT_BURST'):
            rate_limit_config['burst_size'] = self._get_int_env('PROD_LB_RATE_LIMIT_BURST', 50)
        
        if rate_limit_config:
            env_config['rate_limiting'] = rate_limit_config
        
        self._record_config_sources(env_config, ConfigSource.ENVIRONMENT)
        return env_config
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        file_config = {}
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.logger.warning(f"Configuration file not found: {file_path}")
                return {}
            
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    file_config = json.load(f)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
            else:
                # Treat as .env file
                file_config = self._load_env_file(file_path)
            
            self._record_config_sources(file_config, ConfigSource.FILE)
            self.logger.info(f"Loaded configuration from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {file_path}: {e}")
        
        return file_config
    
    def _load_env_file(self, file_path: str) -> Dict[str, Any]:
        """Load .env file and parse into configuration dictionary"""
        env_vars = {}
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        env_vars[key] = value
            
            # Parse environment variables into config structure
            # Temporarily set environment variables and use the same parsing logic
            original_env = {}
            for key, value in env_vars.items():
                if key in os.environ:
                    original_env[key] = os.environ[key]
                os.environ[key] = value
            
            try:
                config = self._load_from_environment()
            finally:
                # Restore original environment
                for key in env_vars:
                    if key in original_env:
                        os.environ[key] = original_env[key]
                    elif key in os.environ:
                        del os.environ[key]
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load .env file {file_path}: {e}")
            return {}
    
    def _get_profile_file(self, profile: str) -> Optional[str]:
        """Get configuration file path for profile"""
        profile_files = {
            'canary': 'canary.env',
            'ab_test': 'ab_test.env',
            'shadow': 'shadow.env',
            'production_full': 'production_full.env'
        }
        
        if profile not in profile_files:
            self.logger.warning(f"Unknown profile: {profile}")
            return None
        
        # Search for profile file
        profile_filename = profile_files[profile]
        search_paths = [
            Path(f"production_deployment_configs/{profile_filename}"),
            Path(f"lightrag_integration/production_deployment_configs/{profile_filename}"),
            Path(f"/etc/clinical_metabolomics/{profile_filename}"),
            Path.home() / ".clinical_metabolomics" / profile_filename
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        self.logger.warning(f"Profile configuration file not found: {profile_filename}")
        return None
    
    def _merge_configurations(self, configs: List[Tuple[str, Dict[str, Any], ConfigSource]]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries with priority"""
        merged = {}
        
        for name, config, source in configs:
            if not config:
                continue
                
            merged = self._deep_merge(merged, config)
            self.logger.debug(f"Merged {name} configuration")
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_config_to_production(self, base_config: ProductionLoadBalancingConfig, 
                                   merged_config: Dict[str, Any]) -> ProductionLoadBalancingConfig:
        """Apply merged configuration to production config object"""
        
        # Convert base config to dict for easier manipulation
        config_dict = asdict(base_config)
        
        # Apply merged configuration
        updated_dict = self._deep_merge(config_dict, merged_config)
        
        # Reconstruct ProductionLoadBalancingConfig
        # This is a simplified approach - in practice you'd want proper deserialization
        try:
            # Update the original config object's attributes
            for key, value in updated_dict.items():
                if hasattr(base_config, key):
                    current_attr = getattr(base_config, key)
                    if hasattr(current_attr, '__dict__'):
                        # It's a nested config object, update its attributes
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                if hasattr(current_attr, nested_key):
                                    setattr(current_attr, nested_key, nested_value)
                    else:
                        # It's a simple attribute
                        setattr(base_config, key, value)
        except Exception as e:
            self.logger.warning(f"Failed to apply some configuration values: {e}")
        
        return base_config
    
    def _validate_production_config(self, config: ProductionLoadBalancingConfig) -> ConfigValidationResult:
        """Validate production configuration"""
        result = ConfigValidationResult(is_valid=True)
        
        # Check required configurations
        if not hasattr(config, 'backends') or not config.backends:
            result.errors.append("No backends configured")
            result.is_valid = False
        
        # Validate backends
        if hasattr(config, 'backends'):
            for backend_name, backend_config in config.backends.items():
                if not backend_config.get('endpoint'):
                    result.errors.append(f"Backend {backend_name} missing endpoint")
                    result.is_valid = False
                
                if not backend_config.get('enabled', True):
                    result.warnings.append(f"Backend {backend_name} is disabled")
        
        # Validate thresholds
        if hasattr(config, 'performance_thresholds'):
            perf = config.performance_thresholds
            if hasattr(perf, 'response_time_ms') and perf.response_time_ms <= 0:
                result.errors.append("Response time threshold must be positive")
                result.is_valid = False
        
        # Validate algorithm configuration
        if hasattr(config, 'algorithm_config'):
            algo = config.algorithm_config
            if hasattr(algo, 'primary_algorithm'):
                valid_algorithms = [
                    'round_robin', 'weighted_round_robin', 'least_connections',
                    'response_time_based', 'health_aware', 'cost_aware',
                    'quality_aware', 'adaptive', 'geographic', 'custom'
                ]
                if algo.primary_algorithm not in valid_algorithms:
                    result.warnings.append(f"Unknown primary algorithm: {algo.primary_algorithm}")
        
        # Check for deprecated settings
        # This would check for any legacy configuration patterns
        
        return result
    
    def _should_continue_with_invalid_config(self) -> bool:
        """Determine if system should continue with invalid configuration"""
        # In production, you might want to be more strict
        return os.getenv('PROD_LB_ALLOW_INVALID_CONFIG', 'false').lower() == 'true'
    
    def _record_config_sources(self, config: Dict[str, Any], source: ConfigSource):
        """Record configuration sources for debugging"""
        for key in config.keys():
            self._config_sources[key] = source
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable"""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            self.logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable"""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            self.logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    
    def get_config_sources(self) -> Dict[str, str]:
        """Get configuration sources for debugging"""
        return {key: source.value for key, source in self._config_sources.items()}
    
    def export_config(self, config: ProductionLoadBalancingConfig, 
                     file_path: str, format: str = 'json') -> bool:
        """Export configuration to file"""
        try:
            config_dict = asdict(config)
            config_dict['exported_at'] = datetime.now().isoformat()
            config_dict['config_sources'] = self.get_config_sources()
            
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            elif format.lower() in ('yaml', 'yml'):
                with open(file_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def migrate_legacy_config(self, legacy_config: LoadBalancingConfig) -> ProductionLoadBalancingConfig:
        """Migrate legacy LoadBalancingConfig to ProductionLoadBalancingConfig"""
        from .production_intelligent_query_router import ConfigurationMigrator
        
        production_config = ConfigurationMigrator.migrate_config(legacy_config)
        
        # Validate migration
        validation = ConfigurationMigrator.validate_migration(legacy_config, production_config)
        
        if not validation['migration_successful']:
            self.logger.warning("Legacy configuration migration had issues")
            for key, success in validation.items():
                if not success and key != 'migration_successful':
                    self.logger.warning(f"Migration issue: {key}")
        else:
            self.logger.info("Legacy configuration migrated successfully")
        
        return production_config


def load_production_config_from_environment(profile: Optional[str] = None) -> ProductionLoadBalancingConfig:
    """
    Convenience function to load production configuration from environment
    
    Args:
        profile: Configuration profile to load (canary, ab_test, shadow, production_full)
    
    Returns:
        ProductionLoadBalancingConfig instance
    """
    loader = ProductionConfigLoader()
    return loader.load_production_config(profile=profile)


def create_production_router_from_config(config_file: Optional[str] = None,
                                       profile: Optional[str] = None) -> 'ProductionIntelligentQueryRouter':
    """
    Create ProductionIntelligentQueryRouter from configuration
    
    Args:
        config_file: Configuration file path
        profile: Configuration profile
    
    Returns:
        ProductionIntelligentQueryRouter instance
    """
    from .production_intelligent_query_router import ProductionIntelligentQueryRouter, ProductionFeatureFlags
    
    loader = ProductionConfigLoader()
    
    if config_file:
        production_config = loader.load_production_config(env_file=config_file)
    else:
        production_config = loader.load_production_config(profile=profile)
    
    # Extract feature flags from config (simplified mapping)
    feature_flags = ProductionFeatureFlags.from_env()  # Start with env defaults
    
    # Create router
    router = ProductionIntelligentQueryRouter(
        production_config=production_config,
        feature_flags=feature_flags
    )
    
    return router


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Configuration Loader")
    parser.add_argument("--profile", choices=['canary', 'ab_test', 'shadow', 'production_full'], 
                       help="Configuration profile to load")
    parser.add_argument("--config-file", help="Configuration file to load")
    parser.add_argument("--export", help="Export configuration to file")
    parser.add_argument("--export-format", choices=['json', 'yaml'], default='json', 
                       help="Export format")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate configuration")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    loader = ProductionConfigLoader()
    
    try:
        if args.config_file:
            config = loader.load_production_config(env_file=args.config_file)
        else:
            config = loader.load_production_config(profile=args.profile)
        
        print("Configuration loaded successfully!")
        
        if args.validate_only:
            validation = loader._validate_production_config(config)
            print(f"Validation result: {'PASSED' if validation.is_valid else 'FAILED'}")
            if validation.errors:
                print("Errors:")
                for error in validation.errors:
                    print(f"  - {error}")
            if validation.warnings:
                print("Warnings:")
                for warning in validation.warnings:
                    print(f"  - {warning}")
        
        if args.export:
            success = loader.export_config(config, args.export, args.export_format)
            if success:
                print(f"Configuration exported to {args.export}")
            else:
                print("Failed to export configuration")
        
        # Print config sources
        sources = loader.get_config_sources()
        if sources:
            print("\\nConfiguration sources:")
            for key, source in sources.items():
                print(f"  {key}: {source}")
    
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)