"""
Configuration Management for Enhanced LLM Query Classifier

This module provides configuration management utilities for the Enhanced LLM Query Classifier,
including preset configurations for different deployment scenarios, validation, and 
environment-based configuration loading.

Key Features:
    - Preset configurations for different use cases (development, production, testing)
    - Environment variable integration
    - Configuration validation and optimization recommendations
    - Dynamic configuration adjustment based on performance metrics
    - Configuration templates for easy deployment

Classes:
    - ConfigPresets: Pre-defined configurations for common scenarios
    - ConfigValidator: Configuration validation and recommendation engine
    - ConfigManager: Central configuration management with environment integration
    - ConfigTemplate: Template generation for different deployment scenarios

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

try:
    from .enhanced_llm_classifier import (
        EnhancedLLMConfig,
        LLMProvider,
        CircuitBreakerConfig,
        CacheConfig,
        CostConfig,
        PerformanceConfig
    )
except ImportError:
    # For standalone testing
    from enhanced_llm_classifier import (
        EnhancedLLMConfig,
        LLMProvider,
        CircuitBreakerConfig,
        CacheConfig,
        CostConfig,
        PerformanceConfig
    )


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"


class UseCaseType(Enum):
    """Different use case types for optimization."""
    LOW_VOLUME = "low_volume"          # <100 requests/day
    MEDIUM_VOLUME = "medium_volume"    # 100-1000 requests/day
    HIGH_VOLUME = "high_volume"        # >1000 requests/day
    BATCH_PROCESSING = "batch_processing"  # Batch operations
    REAL_TIME = "real_time"            # Real-time applications
    COST_SENSITIVE = "cost_sensitive"  # Cost optimization priority


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    estimated_cost_per_day: float
    estimated_performance_score: float


class ConfigPresets:
    """
    Pre-defined configurations for common deployment scenarios.
    Each preset is optimized for specific use cases and environments.
    """
    
    @staticmethod
    def development_config(api_key: str = None) -> EnhancedLLMConfig:
        """
        Development environment configuration.
        Optimized for: Fast iteration, comprehensive logging, moderate costs
        """
        return EnhancedLLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            timeout_seconds=3.0,  # More relaxed timeout
            max_retries=2,
            temperature=0.1,
            
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                success_threshold=2
            ),
            
            cache=CacheConfig(
                enable_caching=True,
                max_cache_size=500,
                ttl_seconds=1800,  # 30 minutes
                adaptive_ttl=True,
                performance_tracking=True
            ),
            
            cost=CostConfig(
                daily_budget=5.0,
                hourly_budget=0.5,
                enable_budget_alerts=True,
                budget_warning_threshold=0.8,
                cost_optimization=True
            ),
            
            performance=PerformanceConfig(
                target_response_time_ms=3000.0,  # Relaxed for development
                enable_monitoring=True,
                auto_optimization=False,  # Manual optimization in dev
                benchmark_frequency=20
            )
        )
    
    @staticmethod
    def production_config(api_key: str = None) -> EnhancedLLMConfig:
        """
        Production environment configuration.
        Optimized for: Reliability, performance, cost efficiency, <2s response time
        """
        return EnhancedLLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            timeout_seconds=1.5,  # Aggressive for <2s target
            max_retries=3,
            temperature=0.1,
            
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=3
            ),
            
            cache=CacheConfig(
                enable_caching=True,
                max_cache_size=2000,
                ttl_seconds=3600,  # 1 hour
                adaptive_ttl=True,
                performance_tracking=True,
                cache_warming=True
            ),
            
            cost=CostConfig(
                daily_budget=20.0,  # Higher for production volume
                hourly_budget=2.0,
                enable_budget_alerts=True,
                budget_warning_threshold=0.85,
                cost_optimization=True
            ),
            
            performance=PerformanceConfig(
                target_response_time_ms=2000.0,  # <2s requirement
                enable_monitoring=True,
                auto_optimization=True,
                benchmark_frequency=100
            )
        )
    
    @staticmethod
    def testing_config(api_key: str = None) -> EnhancedLLMConfig:
        """
        Testing environment configuration.
        Optimized for: Test reliability, controlled costs, comprehensive metrics
        """
        return EnhancedLLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            timeout_seconds=2.0,
            max_retries=1,  # Fail fast for tests
            temperature=0.0,  # Deterministic for testing
            
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=2,  # Low threshold for testing
                recovery_timeout=10.0,
                success_threshold=1
            ),
            
            cache=CacheConfig(
                enable_caching=False,  # Disabled for testing consistency
                max_cache_size=100,
                ttl_seconds=300,  # 5 minutes
                adaptive_ttl=False
            ),
            
            cost=CostConfig(
                daily_budget=1.0,  # Very low for testing
                hourly_budget=0.2,
                enable_budget_alerts=True,
                budget_warning_threshold=0.7,
                cost_optimization=False
            ),
            
            performance=PerformanceConfig(
                target_response_time_ms=2000.0,
                enable_monitoring=True,
                auto_optimization=False,
                benchmark_frequency=5
            )
        )
    
    @staticmethod
    def high_volume_config(api_key: str = None) -> EnhancedLLMConfig:
        """
        High volume deployment configuration.
        Optimized for: High throughput, aggressive caching, cost efficiency
        """
        return EnhancedLLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            timeout_seconds=1.0,  # Very aggressive
            max_retries=2,
            temperature=0.1,
            
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=10,  # Higher threshold for volume
                recovery_timeout=120.0,
                success_threshold=5
            ),
            
            cache=CacheConfig(
                enable_caching=True,
                max_cache_size=5000,  # Large cache for high volume
                ttl_seconds=7200,  # 2 hours
                adaptive_ttl=True,
                performance_tracking=True,
                cache_warming=True
            ),
            
            cost=CostConfig(
                daily_budget=50.0,  # High budget for volume
                hourly_budget=5.0,
                enable_budget_alerts=True,
                budget_warning_threshold=0.9,
                cost_optimization=True
            ),
            
            performance=PerformanceConfig(
                target_response_time_ms=1500.0,  # Very aggressive
                enable_monitoring=True,
                auto_optimization=True,
                benchmark_frequency=500
            )
        )
    
    @staticmethod
    def cost_optimized_config(api_key: str = None) -> EnhancedLLMConfig:
        """
        Cost-optimized configuration.
        Optimized for: Minimum cost, maximum cache utilization, acceptable performance
        """
        return EnhancedLLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",  # Most cost-effective
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            timeout_seconds=2.5,  # Relaxed to avoid retries
            max_retries=1,  # Minimize retry costs
            temperature=0.1,
            
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=120.0,  # Longer recovery to avoid costs
                success_threshold=2
            ),
            
            cache=CacheConfig(
                enable_caching=True,
                max_cache_size=3000,  # Large cache for cost savings
                ttl_seconds=14400,  # 4 hours - longer retention
                adaptive_ttl=True,
                performance_tracking=True
            ),
            
            cost=CostConfig(
                daily_budget=2.0,  # Very conservative
                hourly_budget=0.2,
                enable_budget_alerts=True,
                budget_warning_threshold=0.6,  # Early warning
                cost_optimization=True
            ),
            
            performance=PerformanceConfig(
                target_response_time_ms=4000.0,  # Relaxed for cost savings
                enable_monitoring=True,
                auto_optimization=True,
                benchmark_frequency=50
            )
        )
    
    @staticmethod
    def get_preset(environment: DeploymentEnvironment, 
                  use_case: UseCaseType = None,
                  api_key: str = None) -> EnhancedLLMConfig:
        """
        Get preset configuration based on environment and use case.
        
        Args:
            environment: Target deployment environment
            use_case: Specific use case for optimization
            api_key: OpenAI API key
            
        Returns:
            Optimized configuration for the specified environment and use case
        """
        
        # Base configuration by environment
        if environment == DeploymentEnvironment.DEVELOPMENT:
            config = ConfigPresets.development_config(api_key)
        elif environment == DeploymentEnvironment.PRODUCTION:
            config = ConfigPresets.production_config(api_key)
        elif environment == DeploymentEnvironment.TESTING:
            config = ConfigPresets.testing_config(api_key)
        else:
            config = ConfigPresets.production_config(api_key)  # Default to production
        
        # Adjust based on use case
        if use_case == UseCaseType.HIGH_VOLUME:
            config = ConfigPresets.high_volume_config(api_key)
        elif use_case == UseCaseType.COST_SENSITIVE:
            config = ConfigPresets.cost_optimized_config(api_key)
        elif use_case == UseCaseType.REAL_TIME:
            # Ultra-aggressive timing for real-time
            config.timeout_seconds = 0.8
            config.performance.target_response_time_ms = 1000.0
        elif use_case == UseCaseType.LOW_VOLUME:
            # Conservative settings for low volume
            config.cost.daily_budget = 1.0
            config.cache.max_cache_size = 200
        
        return config


class ConfigValidator:
    """
    Configuration validation and optimization recommendation engine.
    Analyzes configurations for potential issues and suggests improvements.
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_config(self, config: EnhancedLLMConfig) -> ConfigValidationResult:
        """
        Comprehensive configuration validation.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with errors, warnings, and recommendations
        """
        
        errors = []
        warnings = []
        recommendations = []
        
        # ========== API KEY VALIDATION ==========
        if not config.api_key:
            errors.append("API key is required but not provided")
        elif config.api_key == "your-api-key-here" or len(config.api_key) < 10:
            errors.append("Invalid API key format")
        
        # ========== TIMEOUT VALIDATION ==========
        if config.timeout_seconds < 0.5:
            warnings.append(f"Very short timeout ({config.timeout_seconds}s) may cause frequent failures")
        elif config.timeout_seconds > 5.0:
            warnings.append(f"Long timeout ({config.timeout_seconds}s) may affect user experience")
        
        if config.timeout_seconds >= config.performance.target_response_time_ms / 1000.0:
            warnings.append("Timeout should be less than target response time for realistic expectations")
        
        # ========== CIRCUIT BREAKER VALIDATION ==========
        cb = config.circuit_breaker
        if cb.failure_threshold < 2:
            warnings.append("Very low circuit breaker threshold may cause premature failures")
        elif cb.failure_threshold > 20:
            warnings.append("High circuit breaker threshold may delay failure detection")
        
        if cb.recovery_timeout < 10:
            warnings.append("Short recovery timeout may not allow sufficient time for service recovery")
        
        # ========== CACHE VALIDATION ==========
        cache = config.cache
        if cache.enable_caching:
            if cache.max_cache_size < 50:
                warnings.append("Small cache size may reduce hit rate effectiveness")
            elif cache.max_cache_size > 10000:
                warnings.append("Very large cache may consume excessive memory")
            
            if cache.ttl_seconds < 300:  # 5 minutes
                warnings.append("Short cache TTL may reduce effectiveness")
            elif cache.ttl_seconds > 86400:  # 24 hours
                warnings.append("Long cache TTL may serve stale data")
        else:
            recommendations.append("Consider enabling caching for improved performance and cost efficiency")
        
        # ========== COST VALIDATION ==========
        cost = config.cost
        if cost.daily_budget < 0.1:
            warnings.append("Very low daily budget may limit system functionality")
        elif cost.daily_budget > 100:
            warnings.append("High daily budget - ensure monitoring and alerts are properly configured")
        
        if cost.hourly_budget > cost.daily_budget / 12:
            warnings.append("Hourly budget is high relative to daily budget - may exhaust budget early")
        
        # ========== PERFORMANCE VALIDATION ==========
        perf = config.performance
        if perf.target_response_time_ms < 500:
            warnings.append("Very aggressive response time target may be difficult to achieve consistently")
        elif perf.target_response_time_ms > 10000:
            warnings.append("Long response time target may impact user experience")
        
        # ========== CROSS-COMPONENT VALIDATION ==========
        
        # Timeout vs target response time
        if config.timeout_seconds * 1000 > perf.target_response_time_ms * 0.8:
            recommendations.append("Reduce timeout to 70-80% of target response time for better performance")
        
        # Budget vs expected volume
        estimated_cost_per_request = self._estimate_cost_per_request(config)
        max_requests_per_day = cost.daily_budget / estimated_cost_per_request if estimated_cost_per_request > 0 else float('inf')
        
        if max_requests_per_day < 100:
            warnings.append(f"Current budget allows for ~{max_requests_per_day:.0f} requests/day")
        
        # Cache size vs expected volume
        if cache.enable_caching and max_requests_per_day > cache.max_cache_size * 5:
            recommendations.append("Consider increasing cache size for expected request volume")
        
        # ========== GENERATE RECOMMENDATIONS ==========
        
        # Performance recommendations
        if perf.target_response_time_ms <= 2000:
            if config.timeout_seconds > 1.5:
                recommendations.append("Reduce timeout to 1.5s for <2s response target")
            if not cache.enable_caching:
                recommendations.append("Enable caching to meet aggressive response time targets")
        
        # Cost optimization recommendations
        if cost.cost_optimization:
            if config.model_name not in ["gpt-4o-mini"]:
                recommendations.append("Consider gpt-4o-mini model for cost optimization")
        
        # Calculate scores
        estimated_cost_per_day = estimated_cost_per_request * 1000  # Assume 1000 requests/day
        performance_score = self._calculate_performance_score(config)
        
        return ConfigValidationResult(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            estimated_cost_per_day=estimated_cost_per_day,
            estimated_performance_score=performance_score
        )
    
    def _estimate_cost_per_request(self, config: EnhancedLLMConfig) -> float:
        """Estimate cost per request based on configuration."""
        
        # Base token estimates
        avg_prompt_tokens = 800  # Typical prompt size
        avg_response_tokens = 150  # Typical response size
        total_tokens = avg_prompt_tokens + avg_response_tokens
        
        # Model costs (per 1k tokens)
        model_costs = {
            "gpt-4o-mini": 0.0005,
            "gpt-4o": 0.015,
            "gpt-3.5-turbo": 0.0015
        }
        
        cost_per_1k = model_costs.get(config.model_name, 0.001)
        base_cost = (total_tokens / 1000.0) * cost_per_1k
        
        # Adjust for cache hit rate (estimated)
        if config.cache.enable_caching:
            estimated_cache_hit_rate = 0.4  # 40% hit rate assumption
            effective_cost = base_cost * (1 - estimated_cache_hit_rate)
        else:
            effective_cost = base_cost
        
        # Adjust for retries
        retry_factor = 1 + (config.max_retries * 0.1)  # 10% retry rate assumption
        
        return effective_cost * retry_factor
    
    def _calculate_performance_score(self, config: EnhancedLLMConfig) -> float:
        """Calculate expected performance score (0-100)."""
        
        score = 50.0  # Base score
        
        # Timeout score (lower is better for responsiveness)
        if config.timeout_seconds <= 1.0:
            score += 20
        elif config.timeout_seconds <= 2.0:
            score += 10
        elif config.timeout_seconds > 5.0:
            score -= 10
        
        # Cache score
        if config.cache.enable_caching:
            score += 15
            if config.cache.adaptive_ttl:
                score += 5
        
        # Circuit breaker score
        if 3 <= config.circuit_breaker.failure_threshold <= 10:
            score += 5
        
        # Model score (faster models get higher scores)
        if config.model_name == "gpt-4o-mini":
            score += 10
        elif config.model_name == "gpt-4o":
            score += 5
        
        # Target response time score
        if config.performance.target_response_time_ms <= 2000:
            score += 10
        elif config.performance.target_response_time_ms <= 5000:
            score += 5
        
        return min(100.0, max(0.0, score))


class ConfigManager:
    """
    Central configuration management with environment integration and dynamic adjustment.
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validator = ConfigValidator(logger)
        self._config_cache = {}
        
    def load_config(self, 
                   environment: DeploymentEnvironment = None,
                   use_case: UseCaseType = None,
                   config_file: str = None,
                   validate: bool = True) -> EnhancedLLMConfig:
        """
        Load configuration from various sources with validation.
        
        Args:
            environment: Target environment (auto-detected if not provided)
            use_case: Use case optimization
            config_file: Optional config file path
            validate: Whether to validate the configuration
            
        Returns:
            Loaded and validated configuration
        """
        
        # Auto-detect environment if not provided
        if environment is None:
            environment = self._detect_environment()
        
        cache_key = f"{environment.value}_{use_case.value if use_case else 'default'}"
        
        # Check cache first
        if cache_key in self._config_cache:
            self.logger.debug(f"Using cached config for {cache_key}")
            return self._config_cache[cache_key]
        
        # Load configuration
        if config_file and Path(config_file).exists():
            config = self._load_from_file(config_file)
        else:
            config = ConfigPresets.get_preset(environment, use_case)
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        # Validate if requested
        if validate:
            validation_result = self.validator.validate_config(config)
            
            if not validation_result.is_valid:
                error_msg = f"Invalid configuration: {'; '.join(validation_result.errors)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    self.logger.warning(f"Config warning: {warning}")
            
            if validation_result.recommendations:
                for rec in validation_result.recommendations:
                    self.logger.info(f"Config recommendation: {rec}")
        
        # Cache the config
        self._config_cache[cache_key] = config
        
        self.logger.info(f"Loaded configuration for {environment.value} environment")
        return config
    
    def _detect_environment(self) -> DeploymentEnvironment:
        """Auto-detect deployment environment from environment variables."""
        
        env_name = os.getenv('DEPLOYMENT_ENV', '').lower()
        
        if env_name in ['prod', 'production']:
            return DeploymentEnvironment.PRODUCTION
        elif env_name in ['dev', 'development']:
            return DeploymentEnvironment.DEVELOPMENT
        elif env_name in ['test', 'testing']:
            return DeploymentEnvironment.TESTING
        elif env_name in ['stage', 'staging']:
            return DeploymentEnvironment.STAGING
        
        # Default detection based on other environment variables
        if os.getenv('NODE_ENV') == 'production' or os.getenv('FLASK_ENV') == 'production':
            return DeploymentEnvironment.PRODUCTION
        elif os.getenv('CI') or os.getenv('PYTEST_RUNNING'):
            return DeploymentEnvironment.TESTING
        else:
            return DeploymentEnvironment.DEVELOPMENT
    
    def _load_from_file(self, config_file: str) -> EnhancedLLMConfig:
        """Load configuration from JSON file."""
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Convert nested dicts to dataclass objects
            if 'circuit_breaker' in config_data:
                config_data['circuit_breaker'] = CircuitBreakerConfig(**config_data['circuit_breaker'])
            if 'cache' in config_data:
                config_data['cache'] = CacheConfig(**config_data['cache'])
            if 'cost' in config_data:
                config_data['cost'] = CostConfig(**config_data['cost'])
            if 'performance' in config_data:
                config_data['performance'] = PerformanceConfig(**config_data['performance'])
            
            # Convert provider string to enum
            if 'provider' in config_data and isinstance(config_data['provider'], str):
                config_data['provider'] = LLMProvider(config_data['provider'])
            
            return EnhancedLLMConfig(**config_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_file}: {e}")
            raise
    
    def _apply_env_overrides(self, config: EnhancedLLMConfig) -> EnhancedLLMConfig:
        """Apply environment variable overrides to configuration."""
        
        # API Key
        if os.getenv('OPENAI_API_KEY'):
            config.api_key = os.getenv('OPENAI_API_KEY')
        
        # Model override
        if os.getenv('LLM_MODEL_NAME'):
            config.model_name = os.getenv('LLM_MODEL_NAME')
        
        # Timeout override
        if os.getenv('LLM_TIMEOUT_SECONDS'):
            try:
                config.timeout_seconds = float(os.getenv('LLM_TIMEOUT_SECONDS'))
            except ValueError:
                self.logger.warning("Invalid LLM_TIMEOUT_SECONDS value, ignoring")
        
        # Budget overrides
        if os.getenv('LLM_DAILY_BUDGET'):
            try:
                config.cost.daily_budget = float(os.getenv('LLM_DAILY_BUDGET'))
            except ValueError:
                self.logger.warning("Invalid LLM_DAILY_BUDGET value, ignoring")
        
        # Cache overrides
        if os.getenv('LLM_CACHE_SIZE'):
            try:
                config.cache.max_cache_size = int(os.getenv('LLM_CACHE_SIZE'))
            except ValueError:
                self.logger.warning("Invalid LLM_CACHE_SIZE value, ignoring")
        
        # Performance target override
        if os.getenv('LLM_TARGET_RESPONSE_TIME_MS'):
            try:
                config.performance.target_response_time_ms = float(os.getenv('LLM_TARGET_RESPONSE_TIME_MS'))
            except ValueError:
                self.logger.warning("Invalid LLM_TARGET_RESPONSE_TIME_MS value, ignoring")
        
        return config
    
    def save_config(self, config: EnhancedLLMConfig, file_path: str) -> None:
        """Save configuration to JSON file."""
        
        try:
            config_dict = asdict(config)
            
            # Convert enum to string
            config_dict['provider'] = config.provider.value
            
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {file_path}: {e}")
            raise
    
    def optimize_config_for_metrics(self, 
                                   config: EnhancedLLMConfig,
                                   performance_metrics: Dict[str, float]) -> EnhancedLLMConfig:
        """
        Dynamically optimize configuration based on performance metrics.
        
        Args:
            config: Current configuration
            performance_metrics: Performance metrics from the system
            
        Returns:
            Optimized configuration
        """
        
        optimized_config = EnhancedLLMConfig(**asdict(config))  # Deep copy
        
        # Response time optimization
        avg_response_time = performance_metrics.get('avg_response_time_ms', 0)
        target_time = config.performance.target_response_time_ms
        
        if avg_response_time > target_time * 1.2:  # 20% over target
            # Increase cache size if hit rate is low
            hit_rate = performance_metrics.get('cache_hit_rate', 0)
            if hit_rate < 0.5 and config.cache.enable_caching:
                optimized_config.cache.max_cache_size = min(
                    config.cache.max_cache_size * 1.5,
                    5000
                )
                self.logger.info("Increased cache size for better performance")
            
            # Reduce timeout to fail faster
            if config.timeout_seconds > 1.0:
                optimized_config.timeout_seconds = max(
                    config.timeout_seconds * 0.8,
                    0.8
                )
                self.logger.info("Reduced timeout for faster failure detection")
        
        # Cost optimization
        daily_cost = performance_metrics.get('daily_cost', 0)
        if daily_cost > config.cost.daily_budget * 0.9:  # 90% of budget
            # Extend cache TTL to reduce API calls
            optimized_config.cache.ttl_seconds = min(
                config.cache.ttl_seconds * 1.5,
                14400  # Max 4 hours
            )
            self.logger.info("Extended cache TTL for cost savings")
        
        # Circuit breaker optimization
        failure_rate = performance_metrics.get('failure_rate', 0)
        if failure_rate > 0.1:  # 10% failure rate
            # Lower failure threshold to open circuit faster
            optimized_config.circuit_breaker.failure_threshold = max(
                config.circuit_breaker.failure_threshold - 1,
                2
            )
            self.logger.info("Reduced circuit breaker threshold due to high failure rate")
        
        return optimized_config
    
    def generate_config_template(self, 
                               environment: DeploymentEnvironment,
                               output_format: str = "json") -> str:
        """
        Generate configuration template for deployment.
        
        Args:
            environment: Target environment
            output_format: Output format ("json", "yaml", "env")
            
        Returns:
            Configuration template as string
        """
        
        config = ConfigPresets.get_preset(environment)
        
        if output_format == "json":
            config_dict = asdict(config)
            config_dict['provider'] = config.provider.value
            config_dict['api_key'] = "your-api-key-here"
            return json.dumps(config_dict, indent=2)
        
        elif output_format == "env":
            return self._generate_env_template(config)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_env_template(self, config: EnhancedLLMConfig) -> str:
        """Generate environment variable template."""
        
        return f"""# Enhanced LLM Query Classifier Environment Configuration
# Copy this to .env file and customize values

# API Configuration
OPENAI_API_KEY=your-api-key-here
LLM_MODEL_NAME={config.model_name}
LLM_TIMEOUT_SECONDS={config.timeout_seconds}

# Budget Configuration  
LLM_DAILY_BUDGET={config.cost.daily_budget}
LLM_HOURLY_BUDGET={config.cost.hourly_budget}

# Performance Configuration
LLM_TARGET_RESPONSE_TIME_MS={config.performance.target_response_time_ms}
LLM_CACHE_SIZE={config.cache.max_cache_size}
LLM_CACHE_TTL_SECONDS={config.cache.ttl_seconds}

# Circuit Breaker Configuration
LLM_CB_FAILURE_THRESHOLD={config.circuit_breaker.failure_threshold}
LLM_CB_RECOVERY_TIMEOUT={config.circuit_breaker.recovery_timeout}

# Environment
DEPLOYMENT_ENV={DeploymentEnvironment.PRODUCTION.value}
"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_optimized_config(environment: str = None, 
                           use_case: str = None,
                           api_key: str = None) -> EnhancedLLMConfig:
    """
    Convenient function to create optimized configuration.
    
    Args:
        environment: Environment name ("development", "production", etc.)
        use_case: Use case name ("high_volume", "cost_sensitive", etc.)
        api_key: OpenAI API key
        
    Returns:
        Optimized configuration
    """
    
    # Parse environment
    env = None
    if environment:
        try:
            env = DeploymentEnvironment(environment.lower())
        except ValueError:
            pass
    
    # Parse use case
    uc = None
    if use_case:
        try:
            uc = UseCaseType(use_case.lower())
        except ValueError:
            pass
    
    return ConfigPresets.get_preset(env or DeploymentEnvironment.PRODUCTION, uc, api_key)


def validate_environment_setup() -> Dict[str, Any]:
    """
    Validate environment setup for LLM classifier.
    
    Returns:
        Validation results with setup status
    """
    
    results = {
        "api_key": False,
        "dependencies": {},
        "environment_vars": {},
        "recommendations": []
    }
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and len(api_key) > 10:
        results["api_key"] = True
    else:
        results["recommendations"].append("Set OPENAI_API_KEY environment variable")
    
    # Check dependencies
    try:
        import openai
        results["dependencies"]["openai"] = True
    except ImportError:
        results["dependencies"]["openai"] = False
        results["recommendations"].append("Install openai package: pip install openai")
    
    # Check optional environment variables
    env_vars = [
        "DEPLOYMENT_ENV", "LLM_MODEL_NAME", "LLM_DAILY_BUDGET",
        "LLM_TARGET_RESPONSE_TIME_MS", "LLM_CACHE_SIZE"
    ]
    
    for var in env_vars:
        results["environment_vars"][var] = os.getenv(var) is not None
    
    # Overall status
    results["ready"] = (
        results["api_key"] and
        all(results["dependencies"].values())
    )
    
    return results


if __name__ == "__main__":
    # Demo usage
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("Configuration Management Demo")
    print("=" * 50)
    
    # Create config manager
    manager = ConfigManager()
    
    # Load production config
    config = manager.load_config(DeploymentEnvironment.PRODUCTION, UseCaseType.HIGH_VOLUME)
    
    print(f"Loaded config for production/high_volume:")
    print(f"  Model: {config.model_name}")
    print(f"  Timeout: {config.timeout_seconds}s")
    print(f"  Daily Budget: ${config.cost.daily_budget}")
    print(f"  Cache Size: {config.cache.max_cache_size}")
    print()
    
    # Validate environment
    env_status = validate_environment_setup()
    print(f"Environment Status:")
    print(f"  API Key: {'✅' if env_status['api_key'] else '❌'}")
    print(f"  Dependencies: {'✅' if all(env_status['dependencies'].values()) else '❌'}")
    print(f"  Ready: {'✅' if env_status['ready'] else '❌'}")
    
    if env_status["recommendations"]:
        print(f"  Recommendations:")
        for rec in env_status["recommendations"]:
            print(f"    - {rec}")
    
    print()
    
    # Generate template
    template = manager.generate_config_template(DeploymentEnvironment.PRODUCTION, "env")
    print("Environment Template:")
    print(template[:200] + "..." if len(template) > 200 else template)