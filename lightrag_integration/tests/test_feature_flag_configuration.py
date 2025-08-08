#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Feature Flag Configuration.

This module provides extensive test coverage for feature flag configuration
parsing, validation, and environment variable handling in the LightRAGConfig
system.

Test Coverage Areas:
- Environment variable parsing and defaults
- Configuration validation and error handling  
- Feature flag boolean conversion
- Routing rules JSON parsing
- Configuration serialization and deserialization
- Edge cases and invalid configurations
- Environment variable precedence
- Configuration factory methods
- Dynamic configuration updates
- Validation error reporting

Author: Claude Code (Anthropic)
Created: 2025-08-08
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

# Import the components under test
from lightrag_integration.config import LightRAGConfig, LightRAGConfigError


class TestFeatureFlagEnvironmentVariables:
    """Test environment variable parsing for feature flags."""
    
    def test_default_feature_flag_values(self):
        """Test default values for feature flags when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure clean environment
            config = LightRAGConfig()
            
            # Test feature flag defaults
            assert config.lightrag_integration_enabled is False
            assert config.lightrag_rollout_percentage == 0.0
            assert config.lightrag_enable_ab_testing is False
            assert config.lightrag_fallback_to_perplexity is True
            assert config.lightrag_force_user_cohort is None
            assert config.lightrag_enable_performance_comparison is False
            assert config.lightrag_enable_quality_metrics is False
            assert config.lightrag_enable_circuit_breaker is True
            assert config.lightrag_enable_conditional_routing is False
    
    @pytest.mark.parametrize("env_value,expected", [
        ("true", True),
        ("True", True), 
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("YES", True),
        ("t", True),
        ("on", True),
        ("ON", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("NO", False),
        ("f", False),
        ("off", False),
        ("OFF", False),
        ("invalid", False),  # Invalid values should default to False
        ("", False)
    ])
    def test_boolean_environment_variable_parsing(self, env_value, expected):
        """Test boolean environment variable parsing with various formats."""
        env_vars = {
            'LIGHTRAG_INTEGRATION_ENABLED': env_value,
            'LIGHTRAG_ENABLE_AB_TESTING': env_value,
            'LIGHTRAG_FALLBACK_TO_PERPLEXITY': env_value,
            'LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON': env_value,
            'LIGHTRAG_ENABLE_QUALITY_METRICS': env_value,
            'LIGHTRAG_ENABLE_CIRCUIT_BREAKER': env_value,
            'LIGHTRAG_ENABLE_CONDITIONAL_ROUTING': env_value,
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_integration_enabled == expected
            assert config.lightrag_enable_ab_testing == expected
            assert config.lightrag_fallback_to_perplexity == expected
            assert config.lightrag_enable_performance_comparison == expected
            assert config.lightrag_enable_quality_metrics == expected
            assert config.lightrag_enable_circuit_breaker == expected
            assert config.lightrag_enable_conditional_routing == expected
    
    def test_numeric_environment_variables(self):
        """Test numeric environment variable parsing."""
        env_vars = {
            'LIGHTRAG_ROLLOUT_PERCENTAGE': '75.5',
            'LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS': '45.0',
            'LIGHTRAG_MIN_QUALITY_THRESHOLD': '0.85',
            'LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD': '5',
            'LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT': '600.0'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_rollout_percentage == 75.5
            assert config.lightrag_integration_timeout_seconds == 45.0
            assert config.lightrag_min_quality_threshold == 0.85
            assert config.lightrag_circuit_breaker_failure_threshold == 5
            assert config.lightrag_circuit_breaker_recovery_timeout == 600.0
    
    def test_string_environment_variables(self):
        """Test string environment variable parsing."""
        env_vars = {
            'LIGHTRAG_USER_HASH_SALT': 'custom_salt_2025',
            'LIGHTRAG_FORCE_USER_COHORT': 'lightrag'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_user_hash_salt == 'custom_salt_2025'
            assert config.lightrag_force_user_cohort == 'lightrag'
    
    def test_json_environment_variables(self):
        """Test JSON environment variable parsing for routing rules."""
        routing_rules = {
            "length_rule": {
                "type": "query_length",
                "min_length": 50,
                "max_length": 500
            },
            "type_rule": {
                "type": "query_type", 
                "allowed_types": ["metabolite_identification"]
            }
        }
        
        env_vars = {
            'LIGHTRAG_ROUTING_RULES': json.dumps(routing_rules)
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_routing_rules == routing_rules
            assert config.lightrag_routing_rules["length_rule"]["min_length"] == 50
            assert "metabolite_identification" in config.lightrag_routing_rules["type_rule"]["allowed_types"]
    
    def test_invalid_json_environment_variable(self):
        """Test handling of invalid JSON in environment variables."""
        env_vars = {
            'LIGHTRAG_ROUTING_RULES': '{"invalid": json}'  # Invalid JSON
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            # Should default to empty dict or None for invalid JSON
            assert config.lightrag_routing_rules is None or config.lightrag_routing_rules == {}
    
    def test_missing_environment_variables_use_defaults(self):
        """Test that missing environment variables use appropriate defaults."""
        # Clear all relevant environment variables
        env_keys_to_clear = [
            'LIGHTRAG_INTEGRATION_ENABLED',
            'LIGHTRAG_ROLLOUT_PERCENTAGE', 
            'LIGHTRAG_USER_HASH_SALT',
            'LIGHTRAG_ENABLE_AB_TESTING',
            'LIGHTRAG_ROUTING_RULES'
        ]
        
        cleared_env = {key: '' for key in env_keys_to_clear}
        
        with patch.dict(os.environ, cleared_env, clear=False):
            # Remove the keys entirely
            for key in env_keys_to_clear:
                if key in os.environ:
                    del os.environ[key]
            
            config = LightRAGConfig()
            
            # Check defaults are used
            assert config.lightrag_integration_enabled is False
            assert config.lightrag_rollout_percentage == 0.0
            assert config.lightrag_user_hash_salt == "cmo_lightrag_2025"
            assert config.lightrag_enable_ab_testing is False
            assert config.lightrag_routing_rules == {}


class TestConfigurationValidation:
    """Test configuration validation and post-processing."""
    
    def test_rollout_percentage_validation_negative(self):
        """Test rollout percentage is clamped to valid range (negative)."""
        env_vars = {'LIGHTRAG_ROLLOUT_PERCENTAGE': '-10.0'}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_rollout_percentage == 0.0
    
    def test_rollout_percentage_validation_above_100(self):
        """Test rollout percentage is clamped to valid range (above 100)."""
        env_vars = {'LIGHTRAG_ROLLOUT_PERCENTAGE': '150.0'}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_rollout_percentage == 100.0
    
    def test_rollout_percentage_validation_valid_range(self):
        """Test rollout percentage accepts valid values."""
        env_vars = {'LIGHTRAG_ROLLOUT_PERCENTAGE': '67.3'}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_rollout_percentage == 67.3
    
    def test_user_cohort_validation_valid_values(self):
        """Test user cohort validation accepts valid values."""
        valid_cohorts = ['lightrag', 'perplexity']
        
        for cohort in valid_cohorts:
            env_vars = {'LIGHTRAG_FORCE_USER_COHORT': cohort}
            
            with patch.dict(os.environ, env_vars, clear=True):
                config = LightRAGConfig()
                
                assert config.lightrag_force_user_cohort == cohort
    
    def test_user_cohort_validation_invalid_values(self):
        """Test user cohort validation rejects invalid values."""
        env_vars = {'LIGHTRAG_FORCE_USER_COHORT': 'invalid_cohort'}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_force_user_cohort is None
    
    def test_timeout_validation_positive(self):
        """Test timeout validation ensures positive values."""
        env_vars = {'LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS': '45.0'}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_integration_timeout_seconds == 45.0
    
    def test_timeout_validation_zero_or_negative(self):
        """Test timeout validation handles zero or negative values."""
        test_cases = ['-5.0', '0.0', '-1.0']
        
        for timeout_value in test_cases:
            env_vars = {'LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS': timeout_value}
            
            with patch.dict(os.environ, env_vars, clear=True):
                config = LightRAGConfig()
                
                assert config.lightrag_integration_timeout_seconds == 30.0  # Default fallback
    
    def test_quality_threshold_validation_range(self):
        """Test quality threshold is clamped to valid range [0.0, 1.0]."""
        test_cases = [
            ('-0.5', 0.0),  # Below minimum
            ('1.5', 1.0),   # Above maximum  
            ('0.75', 0.75), # Valid value
            ('0.0', 0.0),   # Minimum boundary
            ('1.0', 1.0)    # Maximum boundary
        ]
        
        for input_value, expected_value in test_cases:
            env_vars = {'LIGHTRAG_MIN_QUALITY_THRESHOLD': input_value}
            
            with patch.dict(os.environ, env_vars, clear=True):
                config = LightRAGConfig()
                
                assert config.lightrag_min_quality_threshold == expected_value
    
    def test_circuit_breaker_threshold_validation(self):
        """Test circuit breaker failure threshold validation."""
        env_vars = {'LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD': '0'}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_circuit_breaker_failure_threshold == 3  # Default fallback
    
    def test_circuit_breaker_recovery_timeout_validation(self):
        """Test circuit breaker recovery timeout validation."""
        env_vars = {'LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT': '-100.0'}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_circuit_breaker_recovery_timeout == 300.0  # Default fallback


class TestConfigurationFactoryMethods:
    """Test configuration factory methods and utilities."""
    
    def test_config_from_environment_variables(self):
        """Test creating configuration from environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'test_api_key',
            'LIGHTRAG_MODEL': 'gpt-4o',
            'LIGHTRAG_INTEGRATION_ENABLED': 'true',
            'LIGHTRAG_ROLLOUT_PERCENTAGE': '80.0',
            'LIGHTRAG_ENABLE_AB_TESTING': 'true',
            'LIGHTRAG_ENABLE_CIRCUIT_BREAKER': 'false'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.api_key == 'test_api_key'
            assert config.model == 'gpt-4o'
            assert config.lightrag_integration_enabled is True
            assert config.lightrag_rollout_percentage == 80.0
            assert config.lightrag_enable_ab_testing is True
            assert config.lightrag_enable_circuit_breaker is False
    
    def test_config_with_custom_parameters(self):
        """Test configuration with custom parameter overrides."""
        config = LightRAGConfig(
            api_key="custom_api_key",
            model="gpt-4o-mini",
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=25.0,
            lightrag_enable_ab_testing=True
        )
        
        assert config.api_key == "custom_api_key"
        assert config.model == "gpt-4o-mini"
        assert config.lightrag_integration_enabled is True
        assert config.lightrag_rollout_percentage == 25.0
        assert config.lightrag_enable_ab_testing is True
    
    def test_config_parameter_override_environment(self):
        """Test that direct parameters override environment variables."""
        env_vars = {
            'LIGHTRAG_INTEGRATION_ENABLED': 'false',
            'LIGHTRAG_ROLLOUT_PERCENTAGE': '10.0'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig(
                lightrag_integration_enabled=True,  # Override env var
                lightrag_rollout_percentage=90.0    # Override env var
            )
            
            assert config.lightrag_integration_enabled is True
            assert config.lightrag_rollout_percentage == 90.0


class TestConfigurationSerialization:
    """Test configuration serialization and data handling."""
    
    def test_config_to_dict_feature_flags_included(self):
        """Test that configuration dict includes feature flag settings."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0,
            lightrag_enable_ab_testing=True,
            lightrag_enable_circuit_breaker=False
        )
        
        # Assuming there's a to_dict method or similar
        config_dict = vars(config)
        
        assert 'lightrag_integration_enabled' in config_dict
        assert 'lightrag_rollout_percentage' in config_dict
        assert 'lightrag_enable_ab_testing' in config_dict
        assert 'lightrag_enable_circuit_breaker' in config_dict
        
        assert config_dict['lightrag_integration_enabled'] is True
        assert config_dict['lightrag_rollout_percentage'] == 50.0
        assert config_dict['lightrag_enable_ab_testing'] is True
        assert config_dict['lightrag_enable_circuit_breaker'] is False
    
    def test_config_secure_representation(self):
        """Test that sensitive data is masked in string representation."""
        config = LightRAGConfig(api_key="sensitive_api_key_12345")
        
        config_str = str(config)
        
        # API key should be masked or not fully visible
        assert "sensitive_api_key_12345" not in config_str or "*" in config_str


class TestConfigurationErrorHandling:
    """Test configuration error handling and edge cases."""
    
    def test_invalid_numeric_environment_variable(self):
        """Test handling of invalid numeric environment variables."""
        env_vars = {
            'LIGHTRAG_ROLLOUT_PERCENTAGE': 'not_a_number',
            'LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD': 'invalid'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # Should not raise exception, should use defaults
            config = LightRAGConfig()
            
            assert config.lightrag_rollout_percentage == 0.0  # Default
            assert config.lightrag_circuit_breaker_failure_threshold == 3  # Default
    
    def test_empty_environment_variables(self):
        """Test handling of empty environment variables."""
        env_vars = {
            'LIGHTRAG_USER_HASH_SALT': '',
            'LIGHTRAG_FORCE_USER_COHORT': ''
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            # Should use defaults for empty values
            assert config.lightrag_user_hash_salt == "cmo_lightrag_2025"
            assert config.lightrag_force_user_cohort is None
    
    def test_malformed_json_routing_rules(self):
        """Test handling of malformed JSON in routing rules."""
        env_vars = {
            'LIGHTRAG_ROUTING_RULES': '{"unclosed": json'  # Malformed JSON
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            # Should handle gracefully
            assert config.lightrag_routing_rules in [None, {}]
    
    def test_extremely_large_numeric_values(self):
        """Test handling of extremely large numeric values."""
        env_vars = {
            'LIGHTRAG_ROLLOUT_PERCENTAGE': '999999.0',
            'LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT': '999999999.0'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            # Rollout percentage should be clamped
            assert config.lightrag_rollout_percentage == 100.0
            # Recovery timeout might be allowed or clamped depending on implementation
            assert config.lightrag_circuit_breaker_recovery_timeout >= 0


class TestFeatureFlagInteractions:
    """Test interactions between feature flags."""
    
    def test_ab_testing_requires_integration_enabled(self):
        """Test that A/B testing behavior when integration is disabled."""
        config = LightRAGConfig(
            lightrag_integration_enabled=False,
            lightrag_enable_ab_testing=True
        )
        
        # A/B testing can be enabled in config even if integration is disabled
        # The behavior should be handled at the application level
        assert config.lightrag_integration_enabled is False
        assert config.lightrag_enable_ab_testing is True
    
    def test_circuit_breaker_with_zero_rollout(self):
        """Test circuit breaker configuration with zero rollout."""
        config = LightRAGConfig(
            lightrag_rollout_percentage=0.0,
            lightrag_enable_circuit_breaker=True
        )
        
        assert config.lightrag_rollout_percentage == 0.0
        assert config.lightrag_enable_circuit_breaker is True
    
    def test_conditional_routing_without_rules(self):
        """Test conditional routing enabled but no rules provided."""
        config = LightRAGConfig(
            lightrag_enable_conditional_routing=True,
            lightrag_routing_rules={}
        )
        
        assert config.lightrag_enable_conditional_routing is True
        assert config.lightrag_routing_rules == {}
    
    def test_quality_metrics_with_zero_threshold(self):
        """Test quality metrics with zero threshold."""
        config = LightRAGConfig(
            lightrag_enable_quality_metrics=True,
            lightrag_min_quality_threshold=0.0
        )
        
        assert config.lightrag_enable_quality_metrics is True
        assert config.lightrag_min_quality_threshold == 0.0


class TestConfigurationCompleteIntegration:
    """Test complete configuration scenarios with all feature flags."""
    
    def test_production_feature_flag_configuration(self):
        """Test a realistic production feature flag configuration."""
        env_vars = {
            'LIGHTRAG_INTEGRATION_ENABLED': 'true',
            'LIGHTRAG_ROLLOUT_PERCENTAGE': '25.0',
            'LIGHTRAG_ENABLE_AB_TESTING': 'true', 
            'LIGHTRAG_FALLBACK_TO_PERPLEXITY': 'true',
            'LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON': 'true',
            'LIGHTRAG_ENABLE_QUALITY_METRICS': 'true',
            'LIGHTRAG_MIN_QUALITY_THRESHOLD': '0.75',
            'LIGHTRAG_ENABLE_CIRCUIT_BREAKER': 'true',
            'LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD': '5',
            'LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT': '600.0',
            'LIGHTRAG_ENABLE_CONDITIONAL_ROUTING': 'false',
            'LIGHTRAG_USER_HASH_SALT': 'production_salt_2025',
            'LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS': '45.0'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            # Verify all settings
            assert config.lightrag_integration_enabled is True
            assert config.lightrag_rollout_percentage == 25.0
            assert config.lightrag_enable_ab_testing is True
            assert config.lightrag_fallback_to_perplexity is True
            assert config.lightrag_enable_performance_comparison is True
            assert config.lightrag_enable_quality_metrics is True
            assert config.lightrag_min_quality_threshold == 0.75
            assert config.lightrag_enable_circuit_breaker is True
            assert config.lightrag_circuit_breaker_failure_threshold == 5
            assert config.lightrag_circuit_breaker_recovery_timeout == 600.0
            assert config.lightrag_enable_conditional_routing is False
            assert config.lightrag_user_hash_salt == 'production_salt_2025'
            assert config.lightrag_integration_timeout_seconds == 45.0
    
    def test_development_feature_flag_configuration(self):
        """Test a development environment feature flag configuration."""
        env_vars = {
            'LIGHTRAG_INTEGRATION_ENABLED': 'true',
            'LIGHTRAG_ROLLOUT_PERCENTAGE': '100.0',  # Full rollout for dev
            'LIGHTRAG_ENABLE_AB_TESTING': 'false',   # No A/B testing in dev
            'LIGHTRAG_FALLBACK_TO_PERPLEXITY': 'true',
            'LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON': 'false',
            'LIGHTRAG_ENABLE_QUALITY_METRICS': 'true',
            'LIGHTRAG_MIN_QUALITY_THRESHOLD': '0.5',  # Lower threshold for dev
            'LIGHTRAG_ENABLE_CIRCUIT_BREAKER': 'false',  # Disabled for easier debugging
            'LIGHTRAG_ENABLE_CONDITIONAL_ROUTING': 'true',
            'LIGHTRAG_ROUTING_RULES': json.dumps({
                "dev_rule": {
                    "type": "query_length",
                    "min_length": 10,
                    "max_length": 1000
                }
            }),
            'LIGHTRAG_FORCE_USER_COHORT': 'lightrag',  # Force LightRAG for testing
            'LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS': '60.0'  # Longer timeout for dev
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            # Verify development-specific settings
            assert config.lightrag_integration_enabled is True
            assert config.lightrag_rollout_percentage == 100.0
            assert config.lightrag_enable_ab_testing is False
            assert config.lightrag_enable_circuit_breaker is False
            assert config.lightrag_enable_conditional_routing is True
            assert config.lightrag_force_user_cohort == 'lightrag'
            assert config.lightrag_integration_timeout_seconds == 60.0
            assert config.lightrag_routing_rules is not None
            assert 'dev_rule' in config.lightrag_routing_rules
    
    def test_testing_feature_flag_configuration(self):
        """Test a testing environment feature flag configuration."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=0.0,    # No rollout for testing
            lightrag_enable_ab_testing=False,
            lightrag_fallback_to_perplexity=True,
            lightrag_enable_performance_comparison=True,  # Enable for test metrics
            lightrag_enable_quality_metrics=True,
            lightrag_min_quality_threshold=0.6,
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=2,  # Lower threshold for testing
            lightrag_enable_conditional_routing=False,
            lightrag_integration_timeout_seconds=10.0    # Shorter timeout for tests
        )
        
        # Verify testing-specific settings
        assert config.lightrag_integration_enabled is True
        assert config.lightrag_rollout_percentage == 0.0
        assert config.lightrag_enable_performance_comparison is True
        assert config.lightrag_circuit_breaker_failure_threshold == 2
        assert config.lightrag_integration_timeout_seconds == 10.0


class TestConfigurationEdgeCasesAndCornerCases:
    """Test edge cases and corner cases in configuration."""
    
    def test_unicode_in_environment_variables(self):
        """Test handling of Unicode characters in environment variables."""
        env_vars = {
            'LIGHTRAG_USER_HASH_SALT': 'sält_wïth_ünïcödë_2025'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_user_hash_salt == 'sält_wïth_ünïcödë_2025'
    
    def test_very_long_environment_variables(self):
        """Test handling of very long environment variables."""
        long_salt = 'a' * 1000  # 1000 character salt
        env_vars = {
            'LIGHTRAG_USER_HASH_SALT': long_salt
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_user_hash_salt == long_salt
    
    def test_whitespace_in_environment_variables(self):
        """Test handling of whitespace in environment variables."""
        env_vars = {
            'LIGHTRAG_USER_HASH_SALT': '  salt_with_spaces  ',
            'LIGHTRAG_FORCE_USER_COHORT': ' lightrag '
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            # Should handle whitespace appropriately
            # Behavior depends on implementation - might trim or preserve
            assert isinstance(config.lightrag_user_hash_salt, str)
            assert isinstance(config.lightrag_force_user_cohort, (str, type(None)))
    
    def test_scientific_notation_in_numeric_values(self):
        """Test handling of scientific notation in numeric environment variables."""
        env_vars = {
            'LIGHTRAG_ROLLOUT_PERCENTAGE': '2.5e1',  # 25.0
            'LIGHTRAG_MIN_QUALITY_THRESHOLD': '7.5e-1',  # 0.75
            'LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT': '3.0e2'  # 300.0
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = LightRAGConfig()
            
            assert config.lightrag_rollout_percentage == 25.0
            assert config.lightrag_min_quality_threshold == 0.75
            assert config.lightrag_circuit_breaker_recovery_timeout == 300.0
    
    def test_boundary_value_analysis(self):
        """Test boundary values for numeric configurations."""
        boundary_tests = [
            # (env_var, value, expected_result)
            ('LIGHTRAG_ROLLOUT_PERCENTAGE', '0.0', 0.0),
            ('LIGHTRAG_ROLLOUT_PERCENTAGE', '100.0', 100.0),
            ('LIGHTRAG_MIN_QUALITY_THRESHOLD', '0.0', 0.0), 
            ('LIGHTRAG_MIN_QUALITY_THRESHOLD', '1.0', 1.0),
            ('LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD', '1', 1),
        ]
        
        for env_var, value, expected in boundary_tests:
            env_vars = {env_var: value}
            
            with patch.dict(os.environ, env_vars, clear=True):
                config = LightRAGConfig()
                
                if env_var == 'LIGHTRAG_ROLLOUT_PERCENTAGE':
                    assert config.lightrag_rollout_percentage == expected
                elif env_var == 'LIGHTRAG_MIN_QUALITY_THRESHOLD':
                    assert config.lightrag_min_quality_threshold == expected
                elif env_var == 'LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD':
                    assert config.lightrag_circuit_breaker_failure_threshold == expected


# Mark the end of configuration tests
if __name__ == "__main__":
    pytest.main([__file__])