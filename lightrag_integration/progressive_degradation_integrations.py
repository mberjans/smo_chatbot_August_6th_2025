"""
Progressive Service Degradation Integration Adapters
===================================================

This module provides integration adapters that allow the Progressive Service Degradation Controller
to seamlessly integrate with existing production systems without requiring major code changes.

The adapters use the adapter pattern and dependency injection to:
1. Inject timeout configurations into production load balancer backends
2. Modify query parameters for clinical RAG processing
3. Update monitoring system configurations
4. Coordinate with fallback systems

Key Features:
- Non-invasive integration through configuration injection
- Backward compatibility with existing APIs
- Graceful fallback when integration points are not available
- Runtime detection and configuration of available systems

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import logging
import inspect
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
import threading
import copy

# Import the main degradation controller
try:
    from .progressive_service_degradation_controller import (
        ProgressiveServiceDegradationController,
        SystemLoadLevel,
        DegradationConfiguration
    )
except ImportError:
    from progressive_service_degradation_controller import (
        ProgressiveServiceDegradationController,
        SystemLoadLevel,
        DegradationConfiguration
    )

# Try to import production systems for integration
INTEGRATION_AVAILABLE = {}

try:
    try:
        from .production_load_balancer import ProductionLoadBalancer, BackendInstanceConfig
    except ImportError:
        from production_load_balancer import ProductionLoadBalancer, BackendInstanceConfig
    INTEGRATION_AVAILABLE['load_balancer'] = True
except ImportError:
    INTEGRATION_AVAILABLE['load_balancer'] = False

try:
    try:
        from .clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    except ImportError:
        from clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    INTEGRATION_AVAILABLE['clinical_rag'] = True
except ImportError:
    INTEGRATION_AVAILABLE['clinical_rag'] = False

try:
    try:
        from .production_monitoring import ProductionMonitoring
    except ImportError:
        from production_monitoring import ProductionMonitoring
    INTEGRATION_AVAILABLE['monitoring'] = True
except ImportError:
    INTEGRATION_AVAILABLE['monitoring'] = False

try:
    try:
        from .comprehensive_fallback_system import FallbackOrchestrator
    except ImportError:
        from comprehensive_fallback_system import FallbackOrchestrator
    INTEGRATION_AVAILABLE['fallback'] = True
except ImportError:
    INTEGRATION_AVAILABLE['fallback'] = False

try:
    try:
        from .enhanced_load_monitoring_system import EnhancedLoadDetectionSystem
    except ImportError:
        from enhanced_load_monitoring_system import EnhancedLoadDetectionSystem
    INTEGRATION_AVAILABLE['enhanced_monitoring'] = True
except ImportError:
    INTEGRATION_AVAILABLE['enhanced_monitoring'] = False


# ============================================================================
# PRODUCTION LOAD BALANCER INTEGRATION
# ============================================================================

class LoadBalancerDegradationAdapter:
    """Adapter for integrating degradation with ProductionLoadBalancer."""
    
    def __init__(self, load_balancer: Any, controller: ProgressiveServiceDegradationController):
        self.load_balancer = load_balancer
        self.controller = controller
        self.logger = logging.getLogger(f"{__name__}.LoadBalancerAdapter")
        self.original_configs: Dict[str, Any] = {}
        self.integration_active = False
        
        # Detect integration capabilities
        self.capabilities = self._detect_capabilities()
        self.logger.info(f"Load balancer integration capabilities: {self.capabilities}")
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect what integration capabilities are available."""
        capabilities = {
            'backend_timeout_update': hasattr(self.load_balancer, 'update_backend_timeouts'),
            'circuit_breaker_update': hasattr(self.load_balancer, 'update_circuit_breaker_settings'),
            'backend_config_access': hasattr(self.load_balancer, 'backend_instances'),
            'health_check_config': hasattr(self.load_balancer, 'update_health_check_settings'),
            'direct_config_modification': hasattr(self.load_balancer, 'config')
        }
        return capabilities
    
    def integrate(self) -> bool:
        """Integrate the adapter with the load balancer."""
        try:
            if self.capabilities['backend_config_access']:
                # Store original configurations for rollback
                self._backup_original_configs()
                
                # Register with degradation controller
                self.controller.add_load_change_callback(self._on_load_level_change)
                
                self.integration_active = True
                self.logger.info("Load balancer integration activated")
                return True
            else:
                self.logger.warning("Load balancer does not support required integration points")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to integrate with load balancer: {e}")
            return False
    
    def _backup_original_configs(self):
        """Backup original configurations for rollback."""
        try:
            if hasattr(self.load_balancer, 'backend_instances'):
                self.original_configs['backends'] = {}
                for backend_id, config in self.load_balancer.backend_instances.items():
                    self.original_configs['backends'][backend_id] = {
                        'timeout_seconds': getattr(config, 'timeout_seconds', 30.0),
                        'health_check_timeout_seconds': getattr(config, 'health_check_timeout_seconds', 10.0),
                        'failure_threshold': getattr(config, 'failure_threshold', 5),
                        'recovery_timeout_seconds': getattr(config, 'recovery_timeout_seconds', 60)
                    }
        except Exception as e:
            self.logger.warning(f"Could not backup original configs: {e}")
    
    def _on_load_level_change(self, previous_level: SystemLoadLevel, new_level: SystemLoadLevel):
        """Handle load level changes by updating load balancer configuration."""
        if not self.integration_active:
            return
        
        try:
            # Update timeouts
            if self.capabilities['backend_timeout_update']:
                self._update_backend_timeouts(new_level)
            elif self.capabilities['backend_config_access']:
                self._update_backend_configs_directly(new_level)
            
            # Update circuit breaker settings
            if self.capabilities['circuit_breaker_update']:
                self._update_circuit_breaker_settings(new_level)
            
            # Update health check settings
            if self.capabilities['health_check_config']:
                self._update_health_check_settings(new_level)
            
        except Exception as e:
            self.logger.error(f"Error updating load balancer for level {new_level.name}: {e}")
    
    def _update_backend_timeouts(self, load_level: SystemLoadLevel):
        """Update backend timeouts using the load balancer's API."""
        timeouts = self.controller.timeout_manager.get_all_timeouts()
        
        # Map degradation controller timeouts to load balancer services
        timeout_mapping = {
            'lightrag': timeouts.get('lightrag', 60.0),
            'perplexity': timeouts.get('perplexity_api', 35.0),
            'openai': timeouts.get('openai_api', 45.0),
            'health_check': timeouts.get('health_check', 10.0)
        }
        
        self.load_balancer.update_backend_timeouts(timeout_mapping)
        self.logger.debug(f"Updated load balancer timeouts for level {load_level.name}")
    
    def _update_backend_configs_directly(self, load_level: SystemLoadLevel):
        """Update backend configurations directly by modifying config objects."""
        timeouts = self.controller.timeout_manager.get_all_timeouts()
        
        for backend_id, backend_config in self.load_balancer.backend_instances.items():
            # Determine appropriate timeout based on backend type
            if hasattr(backend_config, 'backend_type'):
                backend_type = str(backend_config.backend_type).lower()
                
                if 'lightrag' in backend_type:
                    new_timeout = timeouts.get('lightrag', 60.0)
                elif 'perplexity' in backend_type:
                    new_timeout = timeouts.get('perplexity_api', 35.0)
                elif 'openai' in backend_type:
                    new_timeout = timeouts.get('openai_api', 45.0)
                else:
                    new_timeout = timeouts.get('lightrag', 60.0)  # Default
                
                # Update timeout if the attribute exists
                if hasattr(backend_config, 'timeout_seconds'):
                    backend_config.timeout_seconds = new_timeout
                
                # Update health check timeout
                if hasattr(backend_config, 'health_check_timeout_seconds'):
                    backend_config.health_check_timeout_seconds = timeouts.get('health_check', 10.0)
        
        self.logger.debug(f"Updated backend configurations for level {load_level.name}")
    
    def _update_circuit_breaker_settings(self, load_level: SystemLoadLevel):
        """Update circuit breaker settings based on load level."""
        # More aggressive circuit breaker settings under high load
        settings = {
            SystemLoadLevel.NORMAL: {'failure_threshold': 5, 'recovery_timeout': 60},
            SystemLoadLevel.ELEVATED: {'failure_threshold': 4, 'recovery_timeout': 70},
            SystemLoadLevel.HIGH: {'failure_threshold': 3, 'recovery_timeout': 80},
            SystemLoadLevel.CRITICAL: {'failure_threshold': 2, 'recovery_timeout': 90},
            SystemLoadLevel.EMERGENCY: {'failure_threshold': 1, 'recovery_timeout': 120}
        }
        
        cb_settings = settings.get(load_level, settings[SystemLoadLevel.NORMAL])
        self.load_balancer.update_circuit_breaker_settings(cb_settings)
        self.logger.debug(f"Updated circuit breaker settings: {cb_settings}")
    
    def _update_health_check_settings(self, load_level: SystemLoadLevel):
        """Update health check settings based on load level."""
        # Reduce health check frequency under high load
        intervals = {
            SystemLoadLevel.NORMAL: 30,
            SystemLoadLevel.ELEVATED: 45,
            SystemLoadLevel.HIGH: 60,
            SystemLoadLevel.CRITICAL: 90,
            SystemLoadLevel.EMERGENCY: 120
        }
        
        hc_settings = {
            'interval_seconds': intervals.get(load_level, 30),
            'timeout_seconds': self.controller.timeout_manager.get_timeout('health_check')
        }
        
        self.load_balancer.update_health_check_settings(hc_settings)
        self.logger.debug(f"Updated health check settings: {hc_settings}")
    
    def rollback(self):
        """Rollback to original configurations."""
        if not self.integration_active or not self.original_configs:
            return
        
        try:
            # Restore original backend configurations
            if 'backends' in self.original_configs and hasattr(self.load_balancer, 'backend_instances'):
                for backend_id, original_config in self.original_configs['backends'].items():
                    if backend_id in self.load_balancer.backend_instances:
                        backend_config = self.load_balancer.backend_instances[backend_id]
                        
                        for attr, value in original_config.items():
                            if hasattr(backend_config, attr):
                                setattr(backend_config, attr, value)
            
            self.logger.info("Rolled back load balancer configurations to original state")
            
        except Exception as e:
            self.logger.error(f"Error rolling back load balancer configs: {e}")


# ============================================================================
# CLINICAL RAG INTEGRATION
# ============================================================================

class ClinicalRAGDegradationAdapter:
    """Adapter for integrating degradation with ClinicalMetabolomicsRAG."""
    
    def __init__(self, clinical_rag: Any, controller: ProgressiveServiceDegradationController):
        self.clinical_rag = clinical_rag
        self.controller = controller
        self.logger = logging.getLogger(f"{__name__}.ClinicalRAGAdapter")
        self.original_params: Dict[str, Any] = {}
        self.integration_active = False
        
        # Detect integration capabilities
        self.capabilities = self._detect_capabilities()
        self.logger.info(f"Clinical RAG integration capabilities: {self.capabilities}")
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect what integration capabilities are available."""
        capabilities = {
            'query_method_wrapping': hasattr(self.clinical_rag, 'query'),
            'async_query_method': hasattr(self.clinical_rag, 'aquery') or asyncio.iscoroutinefunction(getattr(self.clinical_rag, 'query', None)),
            'config_modification': hasattr(self.clinical_rag, 'config'),
            'timeout_injection': True,  # We can always try to inject timeouts
            'param_modification': True   # We can always try to modify parameters
        }
        return capabilities
    
    def integrate(self) -> bool:
        """Integrate the adapter with the clinical RAG system."""
        try:
            # Back up original configurations
            self._backup_original_params()
            
            # Wrap query methods to apply degradation
            if self.capabilities['query_method_wrapping']:
                self._wrap_query_methods()
            
            # Register with degradation controller
            self.controller.add_load_change_callback(self._on_load_level_change)
            
            self.integration_active = True
            self.logger.info("Clinical RAG integration activated")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with clinical RAG: {e}")
            return False
    
    def _backup_original_params(self):
        """Backup original parameters for rollback."""
        try:
            if hasattr(self.clinical_rag, 'config'):
                config = self.clinical_rag.config
                self.original_params = {
                    'max_tokens': getattr(config, 'max_tokens', 8000),
                    'timeout': getattr(config, 'timeout', 60.0),
                    'enable_complex_analytics': getattr(config, 'enable_complex_analytics', True),
                    'enable_detailed_logging': getattr(config, 'enable_detailed_logging', True),
                }
        except Exception as e:
            self.logger.warning(f"Could not backup original RAG params: {e}")
    
    def _wrap_query_methods(self):
        """Wrap query methods to apply degradation logic."""
        # Store original methods
        if hasattr(self.clinical_rag, 'query'):
            self.clinical_rag._original_query = self.clinical_rag.query
            self.clinical_rag.query = self._create_wrapped_query_method(self.clinical_rag._original_query)
        
        if hasattr(self.clinical_rag, 'aquery'):
            self.clinical_rag._original_aquery = self.clinical_rag.aquery
            self.clinical_rag.aquery = self._create_wrapped_async_query_method(self.clinical_rag._original_aquery)
    
    def _create_wrapped_query_method(self, original_method):
        """Create a wrapped version of the query method."""
        def wrapped_query(query: str, **kwargs):
            # Apply degradation logic
            degraded_query = self.controller.simplify_query(query)
            degraded_kwargs = self._apply_degradation_to_kwargs(kwargs)
            
            # Call original method with degraded parameters
            return original_method(degraded_query, **degraded_kwargs)
        
        return wrapped_query
    
    def _create_wrapped_async_query_method(self, original_method):
        """Create a wrapped version of the async query method."""
        async def wrapped_aquery(query: str, **kwargs):
            # Apply degradation logic
            degraded_query = self.controller.simplify_query(query)
            degraded_kwargs = self._apply_degradation_to_kwargs(kwargs)
            
            # Call original method with degraded parameters
            return await original_method(degraded_query, **degraded_kwargs)
        
        return wrapped_aquery
    
    def _apply_degradation_to_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply degradation settings to query kwargs."""
        degraded_kwargs = kwargs.copy()
        
        # Apply complexity settings
        complexity_settings = self.controller.complexity_manager.get_query_params()
        
        # Update token limits
        if 'max_tokens' not in degraded_kwargs or degraded_kwargs['max_tokens'] > complexity_settings.get('token_limit', 8000):
            degraded_kwargs['max_tokens'] = complexity_settings.get('token_limit', 8000)
        
        # Update query mode if not specified
        if 'mode' not in degraded_kwargs:
            degraded_kwargs['mode'] = complexity_settings.get('query_mode', 'hybrid')
        
        # Apply timeout settings
        timeouts = self.controller.timeout_manager.get_all_timeouts()
        if 'timeout' not in degraded_kwargs:
            degraded_kwargs['timeout'] = timeouts.get('lightrag', 60.0)
        
        # Apply feature settings
        feature_settings = self.controller.feature_manager.get_feature_settings()
        
        # Disable complex analytics if required
        if not feature_settings.get('complex_analytics', True):
            degraded_kwargs['enable_analytics'] = False
        
        # Disable detailed logging if required
        if not feature_settings.get('detailed_logging', True):
            degraded_kwargs['verbose'] = False
            degraded_kwargs['enable_logging'] = False
        
        return degraded_kwargs
    
    def _on_load_level_change(self, previous_level: SystemLoadLevel, new_level: SystemLoadLevel):
        """Handle load level changes by updating RAG configuration."""
        if not self.integration_active:
            return
        
        try:
            if hasattr(self.clinical_rag, 'config'):
                self._update_config_directly(new_level)
                
        except Exception as e:
            self.logger.error(f"Error updating clinical RAG for level {new_level.name}: {e}")
    
    def _update_config_directly(self, load_level: SystemLoadLevel):
        """Update RAG configuration directly."""
        config = self.clinical_rag.config
        
        # Apply complexity settings
        complexity_settings = self.controller.complexity_manager.get_query_params()
        if hasattr(config, 'max_tokens'):
            config.max_tokens = complexity_settings.get('token_limit', 8000)
        
        # Apply timeout settings
        timeouts = self.controller.timeout_manager.get_all_timeouts()
        if hasattr(config, 'timeout'):
            config.timeout = timeouts.get('lightrag', 60.0)
        
        # Apply feature settings
        feature_settings = self.controller.feature_manager.get_feature_settings()
        for feature, enabled in feature_settings.items():
            if hasattr(config, feature):
                setattr(config, feature, enabled)
        
        self.logger.debug(f"Updated clinical RAG config for level {load_level.name}")
    
    def rollback(self):
        """Rollback to original methods and parameters."""
        if not self.integration_active:
            return
        
        try:
            # Restore original methods
            if hasattr(self.clinical_rag, '_original_query'):
                self.clinical_rag.query = self.clinical_rag._original_query
                delattr(self.clinical_rag, '_original_query')
            
            if hasattr(self.clinical_rag, '_original_aquery'):
                self.clinical_rag.aquery = self.clinical_rag._original_aquery
                delattr(self.clinical_rag, '_original_aquery')
            
            # Restore original parameters
            if self.original_params and hasattr(self.clinical_rag, 'config'):
                config = self.clinical_rag.config
                for param, value in self.original_params.items():
                    if hasattr(config, param):
                        setattr(config, param, value)
            
            self.logger.info("Rolled back clinical RAG configurations to original state")
            
        except Exception as e:
            self.logger.error(f"Error rolling back clinical RAG: {e}")


# ============================================================================
# MONITORING INTEGRATION
# ============================================================================

class MonitoringDegradationAdapter:
    """Adapter for integrating degradation with ProductionMonitoring."""
    
    def __init__(self, monitoring: Any, controller: ProgressiveServiceDegradationController):
        self.monitoring = monitoring
        self.controller = controller
        self.logger = logging.getLogger(f"{__name__}.MonitoringAdapter")
        self.original_settings: Dict[str, Any] = {}
        self.integration_active = False
        
        self.capabilities = self._detect_capabilities()
        self.logger.info(f"Monitoring integration capabilities: {self.capabilities}")
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect monitoring integration capabilities."""
        capabilities = {
            'interval_adjustment': hasattr(self.monitoring, 'update_monitoring_interval'),
            'logging_control': hasattr(self.monitoring, 'set_detailed_logging'),
            'metric_filtering': hasattr(self.monitoring, 'set_metric_filters'),
            'alert_threshold_update': hasattr(self.monitoring, 'update_alert_thresholds')
        }
        return capabilities
    
    def integrate(self) -> bool:
        """Integrate with the monitoring system."""
        try:
            self._backup_original_settings()
            self.controller.add_load_change_callback(self._on_load_level_change)
            self.integration_active = True
            self.logger.info("Monitoring integration activated")
            return True
        except Exception as e:
            self.logger.error(f"Failed to integrate with monitoring: {e}")
            return False
    
    def _backup_original_settings(self):
        """Backup original monitoring settings."""
        try:
            if hasattr(self.monitoring, 'monitoring_interval'):
                self.original_settings['interval'] = self.monitoring.monitoring_interval
            if hasattr(self.monitoring, 'detailed_logging_enabled'):
                self.original_settings['detailed_logging'] = self.monitoring.detailed_logging_enabled
        except Exception as e:
            self.logger.warning(f"Could not backup monitoring settings: {e}")
    
    def _on_load_level_change(self, previous_level: SystemLoadLevel, new_level: SystemLoadLevel):
        """Handle load level changes for monitoring."""
        if not self.integration_active:
            return
        
        try:
            # Adjust monitoring intervals
            if self.capabilities['interval_adjustment']:
                intervals = {
                    SystemLoadLevel.NORMAL: 5.0,
                    SystemLoadLevel.ELEVATED: 7.0,
                    SystemLoadLevel.HIGH: 10.0,
                    SystemLoadLevel.CRITICAL: 15.0,
                    SystemLoadLevel.EMERGENCY: 20.0
                }
                new_interval = intervals.get(new_level, 5.0)
                self.monitoring.update_monitoring_interval(new_interval)
            
            # Control detailed logging
            if self.capabilities['logging_control']:
                detailed_logging = self.controller.feature_manager.is_feature_enabled('detailed_logging')
                self.monitoring.set_detailed_logging(detailed_logging)
            
            self.logger.debug(f"Updated monitoring settings for level {new_level.name}")
            
        except Exception as e:
            self.logger.error(f"Error updating monitoring for level {new_level.name}: {e}")
    
    def rollback(self):
        """Rollback monitoring settings."""
        if not self.integration_active or not self.original_settings:
            return
        
        try:
            if 'interval' in self.original_settings and self.capabilities['interval_adjustment']:
                self.monitoring.update_monitoring_interval(self.original_settings['interval'])
            
            if 'detailed_logging' in self.original_settings and self.capabilities['logging_control']:
                self.monitoring.set_detailed_logging(self.original_settings['detailed_logging'])
            
            self.logger.info("Rolled back monitoring settings")
            
        except Exception as e:
            self.logger.error(f"Error rolling back monitoring settings: {e}")


# ============================================================================
# INTEGRATION MANAGER
# ============================================================================

class ProgressiveDegradationIntegrationManager:
    """
    Manager for orchestrating integration of progressive degradation with all available systems.
    """
    
    def __init__(self, controller: ProgressiveServiceDegradationController):
        self.controller = controller
        self.adapters: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.IntegrationManager")
        self.integration_status: Dict[str, bool] = {}
    
    def integrate_all_available_systems(self, 
                                       production_load_balancer: Optional[Any] = None,
                                       clinical_rag: Optional[Any] = None,
                                       production_monitoring: Optional[Any] = None) -> Dict[str, bool]:
        """Integrate with all available production systems."""
        
        results = {}
        
        # Integrate with load balancer
        if production_load_balancer and INTEGRATION_AVAILABLE.get('load_balancer', False):
            adapter = LoadBalancerDegradationAdapter(production_load_balancer, self.controller)
            if adapter.integrate():
                self.adapters['load_balancer'] = adapter
                results['load_balancer'] = True
                self.logger.info("Successfully integrated with production load balancer")
            else:
                results['load_balancer'] = False
                self.logger.warning("Failed to integrate with production load balancer")
        else:
            results['load_balancer'] = False
        
        # Integrate with clinical RAG
        if clinical_rag and INTEGRATION_AVAILABLE.get('clinical_rag', False):
            adapter = ClinicalRAGDegradationAdapter(clinical_rag, self.controller)
            if adapter.integrate():
                self.adapters['clinical_rag'] = adapter
                results['clinical_rag'] = True
                self.logger.info("Successfully integrated with clinical RAG system")
            else:
                results['clinical_rag'] = False
                self.logger.warning("Failed to integrate with clinical RAG system")
        else:
            results['clinical_rag'] = False
        
        # Integrate with monitoring
        if production_monitoring and INTEGRATION_AVAILABLE.get('monitoring', False):
            adapter = MonitoringDegradationAdapter(production_monitoring, self.controller)
            if adapter.integrate():
                self.adapters['monitoring'] = adapter
                results['monitoring'] = True
                self.logger.info("Successfully integrated with production monitoring")
            else:
                results['monitoring'] = False
                self.logger.warning("Failed to integrate with production monitoring")
        else:
            results['monitoring'] = False
        
        self.integration_status = results
        
        # Log integration summary
        successful_integrations = sum(results.values())
        total_attempted = len(results)
        self.logger.info(f"Integration summary: {successful_integrations}/{total_attempted} systems integrated")
        
        return results
    
    def rollback_all_integrations(self):
        """Rollback all active integrations."""
        for system_name, adapter in self.adapters.items():
            try:
                adapter.rollback()
                self.logger.info(f"Rolled back {system_name} integration")
            except Exception as e:
                self.logger.error(f"Error rolling back {system_name} integration: {e}")
        
        self.adapters.clear()
        self.integration_status.clear()
        self.logger.info("All integrations rolled back")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            'active_integrations': list(self.adapters.keys()),
            'integration_results': self.integration_status.copy(),
            'available_systems': INTEGRATION_AVAILABLE.copy(),
            'controller_status': self.controller.get_current_status()
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_fully_integrated_degradation_system(
    production_load_balancer: Optional[Any] = None,
    clinical_rag: Optional[Any] = None,
    production_monitoring: Optional[Any] = None,
    enhanced_detector: Optional[Any] = None,
    custom_config: Optional[DegradationConfiguration] = None
) -> Tuple[ProgressiveServiceDegradationController, ProgressiveDegradationIntegrationManager]:
    """
    Create a fully integrated progressive degradation system with all available components.
    """
    
    # Create the main controller
    controller = ProgressiveServiceDegradationController(
        config=custom_config or DegradationConfiguration(),
        enhanced_detector=enhanced_detector,
        production_load_balancer=production_load_balancer,
        clinical_rag=clinical_rag,
        production_monitoring=production_monitoring
    )
    
    # Create integration manager
    integration_manager = ProgressiveDegradationIntegrationManager(controller)
    
    # Perform all available integrations
    integration_results = integration_manager.integrate_all_available_systems(
        production_load_balancer=production_load_balancer,
        clinical_rag=clinical_rag,
        production_monitoring=production_monitoring
    )
    
    logging.info(f"Created fully integrated degradation system: {integration_results}")
    
    return controller, integration_manager


def create_degradation_system_from_existing_components(
    existing_systems: Dict[str, Any],
    monitoring_interval: float = 5.0,
    custom_config: Optional[DegradationConfiguration] = None
) -> Tuple[Any, ProgressiveServiceDegradationController, ProgressiveDegradationIntegrationManager]:
    """
    Create degradation system from existing production components.
    
    Args:
        existing_systems: Dict with keys like 'load_balancer', 'clinical_rag', 'monitoring', etc.
        monitoring_interval: Monitoring interval for load detection
        custom_config: Custom degradation configuration
    
    Returns:
        Tuple of (enhanced_detector, controller, integration_manager)
    """
    
    # Create enhanced load detector if available
    enhanced_detector = None
    if INTEGRATION_AVAILABLE.get('enhanced_monitoring', False):
        from .enhanced_load_monitoring_system import create_enhanced_load_monitoring_system
        enhanced_detector = create_enhanced_load_monitoring_system(
            monitoring_interval=monitoring_interval,
            enable_trend_analysis=True,
            production_monitoring=existing_systems.get('monitoring')
        )
    
    # Create fully integrated system
    controller, integration_manager = create_fully_integrated_degradation_system(
        production_load_balancer=existing_systems.get('load_balancer'),
        clinical_rag=existing_systems.get('clinical_rag'),
        production_monitoring=existing_systems.get('monitoring'),
        enhanced_detector=enhanced_detector,
        custom_config=custom_config
    )
    
    return enhanced_detector, controller, integration_manager


# ============================================================================
# DEMONSTRATION FUNCTION
# ============================================================================

async def demonstrate_progressive_degradation_integration():
    """Demonstrate the progressive degradation integration system."""
    print("Progressive Service Degradation Integration Demonstration")
    print("=" * 70)
    
    # Show available integrations
    print("Available Integration Components:")
    for component, available in INTEGRATION_AVAILABLE.items():
        status = "✓" if available else "✗"
        print(f"  {status} {component}")
    print()
    
    # Create mock systems for demonstration if real ones aren't available
    mock_systems = {}
    
    # Create controller and integration manager
    controller = ProgressiveServiceDegradationController()
    integration_manager = ProgressiveDegradationIntegrationManager(controller)
    
    # Attempt integrations with available systems
    integration_results = integration_manager.integrate_all_available_systems(
        production_load_balancer=mock_systems.get('load_balancer'),
        clinical_rag=mock_systems.get('clinical_rag'),
        production_monitoring=mock_systems.get('monitoring')
    )
    
    print("Integration Results:")
    for system, success in integration_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {system}")
    print()
    
    # Demonstrate load level changes
    print("Demonstrating degradation across load levels...")
    for level in [SystemLoadLevel.ELEVATED, SystemLoadLevel.HIGH, 
                 SystemLoadLevel.CRITICAL, SystemLoadLevel.EMERGENCY, SystemLoadLevel.NORMAL]:
        
        print(f"\n🔄 Load Level: {level.name}")
        controller.force_load_level(level, f"Demo - {level.name}")
        
        status = controller.get_current_status()
        print(f"   Timeouts: LightRAG={status['timeouts'].get('lightrag', 0):.1f}s")
        print(f"   Complexity: Tokens={status['query_complexity'].get('token_limit', 0)}")
        print(f"   Features: {len([k for k, v in status['feature_settings'].items() if v])} enabled")
        
        await asyncio.sleep(1)
    
    # Show final integration status
    print(f"\n📋 Final Integration Status:")
    final_status = integration_manager.get_integration_status()
    print(json.dumps(final_status, indent=2, default=str))
    
    # Cleanup
    integration_manager.rollback_all_integrations()
    print(f"\n✅ Progressive degradation integration demonstration completed!")


if __name__ == "__main__":
    import json
    # Run demonstration
    asyncio.run(demonstrate_progressive_degradation_integration())