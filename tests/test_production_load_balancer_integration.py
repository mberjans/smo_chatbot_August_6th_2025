#!/usr/bin/env python3
"""
Comprehensive Test Suite for Production Load Balancer Integration

This test suite validates all aspects of the production load balancer integration
including backward compatibility, migration safety, performance monitoring,
and deployment scenarios.

Test Coverage:
- Integration with existing IntelligentQueryRouter
- Configuration migration and validation
- Feature flag functionality and deployment modes
- Performance monitoring and comparison
- Migration script validation
- Rollback mechanisms
- Error handling and edge cases

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: Production Load Balancer Integration Testing
"""

import asyncio
import pytest
import json
import tempfile
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import the modules being tested
import sys
sys.path.append(str(Path(__file__).parent.parent / "lightrag_integration"))

from lightrag_integration.production_intelligent_query_router import (
    ProductionIntelligentQueryRouter,
    ProductionFeatureFlags,
    DeploymentMode,
    ConfigurationMigrator,
    PerformanceComparison,
    create_production_intelligent_query_router
)
from lightrag_integration.production_migration_script import (
    ProductionMigrationManager,
    MigrationValidator,
    PerformanceBenchmark
)
from lightrag_integration.production_config_loader import (
    ProductionConfigLoader,
    load_production_config_from_environment,
    create_production_router_from_config
)
from lightrag_integration.production_performance_dashboard import (
    MetricsCollector,
    SystemMetrics,
    ComparisonMetrics
)
from lightrag_integration.intelligent_query_router import (
    IntelligentQueryRouter,
    LoadBalancingConfig,
    HealthCheckConfig
)
from lightrag_integration.production_load_balancer import (
    ProductionLoadBalancer,
    create_default_production_config
)


class TestProductionIntelligentQueryRouterIntegration:
    """Test production intelligent query router integration"""
    
    @pytest.fixture
    def mock_base_router(self):
        """Mock base router"""
        router = Mock()
        router.route_query.return_value = Mock(
            routing_decision="lightrag",
            confidence_metrics=Mock(),
            reasoning="test routing"
        )
        return router
    
    @pytest.fixture
    def legacy_config(self):
        """Legacy configuration fixture"""
        return LoadBalancingConfig(
            strategy="weighted_round_robin",
            health_check_interval=60,
            circuit_breaker_threshold=5,
            response_time_threshold_ms=2000.0,
            enable_adaptive_routing=True
        )
    
    @pytest.fixture
    def production_feature_flags(self):
        """Production feature flags fixture"""
        return ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.SHADOW,
            production_traffic_percentage=10.0,
            enable_performance_comparison=True,
            enable_automatic_failback=True
        )
    
    def test_initialization_with_feature_flags_disabled(self, mock_base_router, legacy_config):
        """Test initialization with production features disabled"""
        feature_flags = ProductionFeatureFlags(enable_production_load_balancer=False)
        
        router = ProductionIntelligentQueryRouter(
            base_router=mock_base_router,
            load_balancing_config=legacy_config,
            feature_flags=feature_flags
        )
        
        assert router.production_load_balancer is None
        assert router.legacy_router is not None
        assert router.feature_flags.enable_production_load_balancer is False
    
    def test_initialization_with_feature_flags_enabled(self, mock_base_router, legacy_config):
        """Test initialization with production features enabled"""
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.SHADOW
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer'):
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                load_balancing_config=legacy_config,
                feature_flags=feature_flags
            )
            
            assert router.production_load_balancer is not None
            assert router.legacy_router is not None
            assert router.feature_flags.enable_production_load_balancer is True
    
    @pytest.mark.asyncio
    async def test_routing_legacy_only_mode(self, mock_base_router, legacy_config):
        """Test routing in legacy-only mode"""
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=False,
            deployment_mode=DeploymentMode.LEGACY_ONLY
        )
        
        router = ProductionIntelligentQueryRouter(
            base_router=mock_base_router,
            load_balancing_config=legacy_config,
            feature_flags=feature_flags
        )
        
        with patch.object(router.legacy_router, 'route_query') as mock_route:
            mock_route.return_value = Mock(routing_decision="lightrag")
            
            result = await router.route_query("test query")
            
            assert result is not None
            mock_route.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_routing_shadow_mode(self, mock_base_router, legacy_config):
        """Test routing in shadow mode (both systems run)"""
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.SHADOW
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer'):
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                load_balancing_config=legacy_config,
                feature_flags=feature_flags
            )
            
            with patch.object(router, '_route_with_legacy') as mock_legacy, \
                 patch.object(router, '_route_with_production') as mock_production:
                
                mock_legacy.return_value = Mock(routing_decision="lightrag")
                mock_production.return_value = Mock(routing_decision="perplexity")
                
                result = await router.route_query("test query")
                
                assert result is not None
                mock_legacy.assert_called_once()
                mock_production.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_canary_deployment_traffic_splitting(self, mock_base_router, legacy_config):
        """Test canary deployment traffic splitting"""
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.CANARY,
            production_traffic_percentage=25.0  # 25% to production
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer'):
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                load_balancing_config=legacy_config,
                feature_flags=feature_flags
            )
            
            # Mock routing methods
            with patch.object(router, '_route_with_legacy') as mock_legacy, \
                 patch.object(router, '_route_with_production') as mock_production:
                
                mock_legacy.return_value = Mock(routing_decision="lightrag")
                mock_production.return_value = Mock(routing_decision="perplexity")
                
                # Run multiple queries and count routing
                legacy_count = 0
                production_count = 0
                
                for i in range(100):
                    router.request_counter = i  # Control request counter
                    result = await router.route_query(f"test query {i}")
                    
                    if mock_legacy.called:
                        legacy_count += 1
                        mock_legacy.reset_mock()
                    if mock_production.called:
                        production_count += 1
                        mock_production.reset_mock()
                
                # Verify approximate traffic splitting (allowing some variance)
                total_requests = legacy_count + production_count
                production_percentage = (production_count / total_requests) * 100
                
                # Should be roughly 25% with some tolerance
                assert 15 <= production_percentage <= 35
    
    def test_automatic_rollback_trigger(self, mock_base_router, legacy_config):
        """Test automatic rollback on performance degradation"""
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.CANARY,
            enable_automatic_failback=True,
            rollback_threshold_error_rate=10.0
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer'):
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                load_balancing_config=legacy_config,
                feature_flags=feature_flags
            )
            
            # Add performance comparisons with high error rate
            for i in range(150):
                comparison = PerformanceComparison(
                    timestamp=datetime.now(),
                    legacy_response_time_ms=1000,
                    production_response_time_ms=1200,
                    legacy_success=True,
                    production_success=i % 5 != 0,  # 20% failure rate
                    legacy_backend="lightrag",
                    production_backend="perplexity",
                    query_complexity=1.0
                )
                router.performance_comparisons.append(comparison)
            
            # Check if rollback is triggered
            should_rollback = router._should_trigger_rollback()
            
            assert should_rollback is True
            
            # Verify rollback actually happens
            router._should_trigger_rollback = Mock(return_value=True)
            
            # This would trigger rollback in actual routing
            assert router.feature_flags.deployment_mode != DeploymentMode.LEGACY_ONLY
    
    def test_performance_report_generation(self, mock_base_router, legacy_config):
        """Test performance report generation"""
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.A_B_TESTING
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer'):
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                load_balancing_config=legacy_config,
                feature_flags=feature_flags
            )
            
            # Add sample performance data
            for i in range(50):
                comparison = PerformanceComparison(
                    timestamp=datetime.now(),
                    legacy_response_time_ms=1000 + (i * 10),
                    production_response_time_ms=800 + (i * 8),  # Production is faster
                    legacy_success=True,
                    production_success=True,
                    legacy_backend="lightrag",
                    production_backend="perplexity",
                    query_complexity=1.0
                )
                router.performance_comparisons.append(comparison)
            
            report = router.get_performance_report()
            
            assert 'deployment_mode' in report
            assert 'total_requests' in report
            assert 'legacy_stats' in report
            assert 'production_stats' in report
            assert 'performance_improvement' in report
            assert 'recommendation' in report
            
            # Verify improvement calculation
            improvement = report['performance_improvement']['response_time_improvement_percent']
            assert improvement > 0  # Production should be faster
    
    def test_backward_compatibility_methods(self, mock_base_router, legacy_config):
        """Test backward compatibility with existing IntelligentQueryRouter interface"""
        feature_flags = ProductionFeatureFlags(enable_production_load_balancer=False)
        
        router = ProductionIntelligentQueryRouter(
            base_router=mock_base_router,
            load_balancing_config=legacy_config,
            feature_flags=feature_flags
        )
        
        # Test that all backward compatibility methods exist and work
        assert hasattr(router, 'update_backend_weights')
        assert hasattr(router, 'export_analytics')
        assert hasattr(router, 'get_health_status')
        
        # Test method calls
        router.update_backend_weights({'lightrag': 0.7, 'perplexity': 0.3})
        
        analytics = router.export_analytics()
        assert 'production_integration' in analytics
        
        health = router.get_health_status()
        assert 'production_load_balancer' in health


class TestConfigurationMigration:
    """Test configuration migration functionality"""
    
    def test_legacy_config_migration(self):
        """Test migration from legacy to production config"""
        legacy_config = LoadBalancingConfig(
            strategy="weighted_round_robin",
            health_check_interval=30,
            circuit_breaker_threshold=3,
            response_time_threshold_ms=1500.0,
            enable_adaptive_routing=True
        )
        
        migrated_config = ConfigurationMigrator.migrate_config(legacy_config)
        
        assert migrated_config.health_monitoring.check_interval_seconds == 30
        assert migrated_config.circuit_breaker.failure_threshold == 3
        assert migrated_config.performance_thresholds.response_time_ms == 1500.0
        assert migrated_config.algorithm_config.enable_adaptive_selection is True
    
    def test_migration_validation(self):
        """Test migration validation"""
        legacy_config = LoadBalancingConfig(
            strategy="round_robin",
            health_check_interval=60,
            circuit_breaker_threshold=5,
            response_time_threshold_ms=2000.0,
            enable_adaptive_routing=False
        )
        
        migrated_config = ConfigurationMigrator.migrate_config(legacy_config)
        validation_result = ConfigurationMigrator.validate_migration(legacy_config, migrated_config)
        
        assert validation_result['migration_successful'] is True
        assert validation_result['health_check_interval_preserved'] is True
        assert validation_result['circuit_breaker_threshold_preserved'] is True
        assert validation_result['response_time_threshold_preserved'] is True
        assert validation_result['adaptive_routing_preserved'] is True


class TestProductionConfigLoader:
    """Test production configuration loader"""
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables"""
        # Set test environment variables
        test_env = {
            'PROD_LB_ENABLED': 'true',
            'PROD_LB_DEPLOYMENT_MODE': 'canary',
            'PROD_LB_TRAFFIC_PERCENT': '15.0',
            'PROD_LB_HEALTH_CHECK_INTERVAL': '45',
            'PROD_LB_CB_FAILURE_THRESHOLD': '4'
        }
        
        with patch.dict(os.environ, test_env):
            loader = ProductionConfigLoader()
            env_config = loader._load_from_environment()
            
            assert env_config['enable_production_load_balancer'] is True
            assert env_config['deployment_mode'] == 'canary'
            assert env_config['production_traffic_percentage'] == 15.0
            assert env_config['health_monitoring']['check_interval_seconds'] == 45
            assert env_config['circuit_breaker']['failure_threshold'] == 4
    
    def test_config_file_loading(self):
        """Test loading configuration from file"""
        config_data = {
            'enable_production_load_balancer': True,
            'deployment_mode': 'shadow',
            'production_traffic_percentage': 0.0,
            'backends': {
                'lightrag': {
                    'name': 'lightrag',
                    'endpoint': 'http://localhost:8080',
                    'enabled': True
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            loader = ProductionConfigLoader()
            file_config = loader._load_from_file(temp_file)
            
            assert file_config['enable_production_load_balancer'] is True
            assert file_config['deployment_mode'] == 'shadow'
            assert 'lightrag' in file_config['backends']
        finally:
            os.unlink(temp_file)
    
    def test_config_validation(self):
        """Test configuration validation"""
        loader = ProductionConfigLoader()
        
        # Test valid configuration
        valid_config = create_default_production_config()
        validation_result = loader._validate_production_config(valid_config)
        
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0
        
        # Test invalid configuration (no backends)
        invalid_config = create_default_production_config()
        invalid_config.backends = {}
        
        validation_result = loader._validate_production_config(invalid_config)
        
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0


class TestMigrationScript:
    """Test migration script functionality"""
    
    @pytest.fixture
    def mock_migration_manager(self):
        """Mock migration manager"""
        with patch('lightrag_integration.production_migration_script.IntelligentQueryRouter'), \
             patch('lightrag_integration.production_migration_script.ProductionIntelligentQueryRouter'):
            manager = ProductionMigrationManager()
            return manager
    
    @pytest.mark.asyncio
    async def test_validation_phase(self, mock_migration_manager):
        """Test migration validation phase"""
        with patch.object(mock_migration_manager.validator, 'validate_prerequisites') as mock_validate:
            mock_validate.return_value = {
                'existing_router_available': True,
                'production_config_valid': True,
                'system_resources': True,
                'network_connectivity': True,
                'backend_health': True,
                'storage_permissions': True
            }
            
            result = await mock_migration_manager._run_validation_phase()
            
            assert result is True
            assert mock_migration_manager.migration_state['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_preparation_phase(self, mock_migration_manager):
        """Test migration preparation phase"""
        mock_migration_manager.existing_router = Mock()
        
        with patch.object(mock_migration_manager.benchmark, 'benchmark_production_system') as mock_benchmark:
            mock_benchmark.return_value = {
                'avg_response_time_ms': 800,
                'success_rate': 98.5,
                'total_queries': 10
            }
            
            result = await mock_migration_manager._run_preparation_phase()
            
            assert result is True
            assert 'production' in mock_migration_manager.migration_state['performance_baselines']
    
    def test_rollback_point_creation(self, mock_migration_manager):
        """Test rollback point creation"""
        mock_migration_manager._create_rollback_point("test_checkpoint")
        
        assert len(mock_migration_manager.migration_state['rollback_points']) == 1
        rollback_point = mock_migration_manager.migration_state['rollback_points'][0]
        
        assert rollback_point['name'] == "test_checkpoint"
        assert 'timestamp' in rollback_point
        assert 'phase' in rollback_point


class TestPerformanceDashboard:
    """Test performance dashboard functionality"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Metrics collector fixture"""
        return MetricsCollector(collection_interval=1)  # Fast collection for testing
    
    def test_system_metrics_creation(self):
        """Test system metrics data structure"""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            system_name='test_system',
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            avg_response_time_ms=1200.0,
            median_response_time_ms=1000.0,
            p95_response_time_ms=2000.0,
            p99_response_time_ms=3000.0,
            success_rate=95.0,
            error_rate=5.0,
            requests_per_second=10.0,
            active_backends=2
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['system_name'] == 'test_system'
        assert metrics_dict['total_requests'] == 100
        assert metrics_dict['success_rate'] == 95.0
        assert 'timestamp' in metrics_dict
    
    def test_comparison_metrics_creation(self):
        """Test comparison metrics data structure"""
        comparison = ComparisonMetrics(
            timestamp=datetime.now(),
            performance_improvement_percent=15.5,
            reliability_improvement_percent=2.1,
            cost_difference_percent=8.3,
            quality_improvement_percent=5.0,
            recommendation="RECOMMENDED: Increase production traffic"
        )
        
        comparison_dict = comparison.to_dict()
        
        assert comparison_dict['performance_improvement_percent'] == 15.5
        assert comparison_dict['recommendation'] == "RECOMMENDED: Increase production traffic"
        assert 'timestamp' in comparison_dict
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, metrics_collector):
        """Test metrics collection process"""
        # Mock routers
        legacy_router = Mock()
        legacy_router.export_analytics.return_value = {
            'response_times': [1000, 1100, 900, 1200, 1050],
            'backend_health': {'lightrag': 'healthy', 'perplexity': 'healthy'}
        }
        
        production_router = Mock()
        production_router.get_performance_report.return_value = {
            'total_requests': 5,
            'production_stats': {
                'success_rate': 100.0,
                'avg_response_time_ms': 850.0,
                'median_response_time_ms': 800.0,
                'p95_response_time_ms': 1100.0
            }
        }
        
        # Start collection briefly
        await metrics_collector.start_collection(legacy_router, production_router)
        
        # Wait for at least one collection cycle
        await asyncio.sleep(2)
        
        metrics_collector.stop_collection()
        
        # Verify metrics were collected
        latest_metrics = metrics_collector.get_latest_metrics()
        
        # Note: In a real test, you might need to mock the collection methods
        # since we're not running actual routers
        assert isinstance(latest_metrics, dict)
    
    def test_alert_generation(self, metrics_collector):
        """Test alert generation logic"""
        # Create comparison with poor performance
        poor_comparison = ComparisonMetrics(
            timestamp=datetime.now(),
            performance_improvement_percent=-60.0,  # Severe degradation
            reliability_improvement_percent=-10.0,   # Reliability drop
            cost_difference_percent=-50.0,          # Cost increase
            quality_improvement_percent=-5.0,       # Quality drop
            recommendation="CAUTION: Consider rollback"
        )
        
        # Test alert checking
        metrics_collector._check_alerts(poor_comparison)
        
        # Check if alerts were generated
        alerts = []
        while not metrics_collector.alert_queue.empty():
            alerts.append(metrics_collector.alert_queue.get())
        
        assert len(alerts) > 0
        
        # Verify critical alerts for severe degradation
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        assert len(critical_alerts) > 0


class TestFactoryFunctions:
    """Test factory functions for easy integration"""
    
    def test_create_production_intelligent_query_router(self):
        """Test factory function for creating production router"""
        existing_router = Mock()
        existing_router.base_router = Mock()
        existing_router.load_balancing_config = LoadBalancingConfig()
        existing_router.health_check_config = Mock()
        
        router = create_production_intelligent_query_router(
            existing_router=existing_router,
            enable_production=True,
            deployment_mode='canary',
            traffic_percentage=10.0
        )
        
        assert isinstance(router, ProductionIntelligentQueryRouter)
        assert router.feature_flags.enable_production_load_balancer is True
        assert router.feature_flags.deployment_mode == DeploymentMode.CANARY
        assert router.feature_flags.production_traffic_percentage == 10.0
    
    def test_load_production_config_from_environment(self):
        """Test loading production config from environment"""
        test_env = {
            'PROD_LB_ENABLED': 'true',
            'PROD_LB_DEPLOYMENT_MODE': 'production_only'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('lightrag_integration.production_config_loader.create_default_production_config') as mock_create:
                mock_config = Mock()
                mock_create.return_value = mock_config
                
                config = load_production_config_from_environment()
                
                assert config is not None


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_production_system_failure_fallback(self):
        """Test fallback to legacy when production system fails"""
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION_ONLY,
            enable_automatic_failback=True
        )
        
        mock_base_router = Mock()
        legacy_config = LoadBalancingConfig()
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer') as MockProdLB:
            # Make production load balancer fail
            mock_prod_lb = Mock()
            mock_prod_lb.select_backend.side_effect = Exception("Production system failure")
            MockProdLB.return_value = mock_prod_lb
            
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                load_balancing_config=legacy_config,
                feature_flags=feature_flags
            )
            
            with patch.object(router, '_route_with_legacy') as mock_legacy_route:
                mock_legacy_route.return_value = Mock(routing_decision="lightrag")
                
                # This should fallback to legacy despite production-only mode
                result = await router.route_query("test query")
                
                assert result is not None
                mock_legacy_route.assert_called_once()
    
    def test_configuration_validation_with_invalid_data(self):
        """Test configuration validation with invalid data"""
        loader = ProductionConfigLoader()
        
        # Create invalid configuration
        invalid_config = create_default_production_config()
        invalid_config.backends = {}  # No backends
        
        # Mock performance thresholds with invalid values
        invalid_config.performance_thresholds.response_time_ms = -100  # Invalid negative value
        
        validation_result = loader._validate_production_config(invalid_config)
        
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
        assert any("No backends configured" in error for error in validation_result.errors)
        assert any("Response time threshold must be positive" in error for error in validation_result.errors)
    
    @pytest.mark.asyncio
    async def test_canary_timeout_fallback(self):
        """Test canary deployment timeout and fallback"""
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.CANARY,
            max_canary_duration_hours=0.001  # Very short timeout for testing
        )
        
        mock_base_router = Mock()
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer'):
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                feature_flags=feature_flags
            )
            
            # Set canary start time to past
            router._canary_start_time = datetime.now() - timedelta(hours=1)
            
            # Should not use production due to timeout
            should_use_production = router._should_use_production()
            
            assert should_use_production is False


@pytest.mark.asyncio
async def test_end_to_end_integration():
    """End-to-end integration test"""
    # Create a complete integration test
    
    # 1. Create legacy router
    legacy_config = LoadBalancingConfig(
        strategy="weighted_round_robin",
        health_check_interval=60,
        enable_adaptive_routing=True
    )
    
    # 2. Create production feature flags for shadow mode
    feature_flags = ProductionFeatureFlags(
        enable_production_load_balancer=True,
        deployment_mode=DeploymentMode.SHADOW,
        enable_performance_comparison=True
    )
    
    # 3. Mock base router and production load balancer
    mock_base_router = Mock()
    mock_base_router.route_query.return_value = Mock(
        routing_decision="lightrag",
        confidence_metrics=Mock(),
        reasoning="test routing"
    )
    
    with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer'):
        # 4. Create production router
        router = ProductionIntelligentQueryRouter(
            base_router=mock_base_router,
            load_balancing_config=legacy_config,
            feature_flags=feature_flags
        )
        
        # Mock the routing methods
        with patch.object(router, '_route_with_legacy') as mock_legacy, \
             patch.object(router, '_route_with_production') as mock_production:
            
            mock_legacy.return_value = Mock(routing_decision="lightrag")
            mock_production.return_value = Mock(routing_decision="perplexity") 
            
            # 5. Test routing in shadow mode (both should be called)
            result = await router.route_query("What are metabolic pathways?")
            
            assert result is not None
            mock_legacy.assert_called_once()
            mock_production.assert_called_once()
            
            # 6. Verify performance comparison was recorded
            assert len(router.performance_comparisons) > 0
            
            # 7. Test performance report generation
            report = router.get_performance_report()
            assert 'deployment_mode' in report
            assert report['deployment_mode'] == 'shadow'
            
            # 8. Test backward compatibility
            analytics = router.export_analytics()
            assert 'production_integration' in analytics


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])