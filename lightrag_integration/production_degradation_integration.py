"""
Production Integration for Graceful Degradation System

This module provides the integration layer between the graceful degradation system
and the existing Clinical Metabolomics Oracle production components, including:
- ProductionLoadBalancer
- ComprehensiveFallbackSystem  
- ClinicalMetabolomicsRAG
- ProductionMonitoring

It implements the adapter pattern to seamlessly integrate degradation strategies
with existing systems without requiring major architectural changes.

Key Features:
- Dynamic timeout injection into existing API clients
- Feature toggle integration with RAG processing
- Load balancer configuration updates
- Monitoring system integration
- Fallback system coordination

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime

# Import graceful degradation components
from .graceful_degradation_system import (
    GracefulDegradationManager,
    SystemLoadLevel,
    SystemLoadMetrics,
    LoadThresholds,
    create_production_degradation_system
)

# Import existing production components
PRODUCTION_COMPONENTS_AVAILABLE = True
try:
    from .production_load_balancer import ProductionLoadBalancer
except ImportError:
    PRODUCTION_COMPONENTS_AVAILABLE = False
    class ProductionLoadBalancer:
        def __init__(self, *args, **kwargs):
            self.backend_instances = {}

try:
    from .comprehensive_fallback_system import FallbackOrchestrator, FallbackLevel
except ImportError:
    PRODUCTION_COMPONENTS_AVAILABLE = False
    class FallbackOrchestrator:
        def __init__(self, *args, **kwargs):
            self.config = {}

try:
    from .clinical_metabolomics_rag import ClinicalMetabolomicsRAG
except ImportError:
    PRODUCTION_COMPONENTS_AVAILABLE = False
    class ClinicalMetabolomicsRAG:
        def __init__(self, *args, **kwargs):
            self.config = {}

try:
    from .production_monitoring import ProductionMonitoring
except ImportError:
    PRODUCTION_COMPONENTS_AVAILABLE = False
    class ProductionMonitoring:
        def __init__(self, *args, **kwargs):
            pass

if not PRODUCTION_COMPONENTS_AVAILABLE:
    logging.warning("Some production components not available - using mock interfaces")


# ============================================================================
# INTEGRATION ADAPTERS
# ============================================================================

class LoadBalancerDegradationAdapter:
    """Adapts load balancer behavior based on degradation levels."""
    
    def __init__(self, load_balancer: ProductionLoadBalancer):
        self.load_balancer = load_balancer
        self.logger = logging.getLogger(__name__)
        self.original_configs: Dict[str, Any] = {}
        self.degradation_active = False
        
    def apply_degradation(self, load_level: SystemLoadLevel, timeouts: Dict[str, float]):
        """Apply degradation settings to load balancer."""
        if load_level == SystemLoadLevel.NORMAL and not self.degradation_active:
            return
        
        try:
            # Store original configs if first time applying degradation
            if not self.degradation_active:
                self._store_original_configs()
                self.degradation_active = True
            
            # Update backend timeouts
            for instance_id, instance in self.load_balancer.backend_instances.items():
                if hasattr(instance, 'config'):
                    # Update timeouts based on backend type
                    if 'lightrag' in instance_id.lower():
                        instance.config.timeout_seconds = timeouts.get('lightrag_query', 60.0)
                    elif 'perplexity' in instance_id.lower():
                        instance.config.timeout_seconds = timeouts.get('perplexity_api', 35.0)
                    elif 'openai' in instance_id.lower():
                        instance.config.timeout_seconds = timeouts.get('openai_api', 45.0)
                    else:
                        instance.config.timeout_seconds = timeouts.get('general_api', 30.0)
            
            # Adjust concurrent request limits based on load level
            if hasattr(self.load_balancer, 'max_concurrent_requests'):
                if load_level >= SystemLoadLevel.CRITICAL:
                    self.load_balancer.max_concurrent_requests = 10
                elif load_level >= SystemLoadLevel.HIGH:
                    self.load_balancer.max_concurrent_requests = 30
                elif load_level >= SystemLoadLevel.ELEVATED:
                    self.load_balancer.max_concurrent_requests = 60
            
            self.logger.info(f"Load balancer adapted for {load_level.name}")
            
        except Exception as e:
            self.logger.error(f"Error applying degradation to load balancer: {e}")
    
    def restore_original_settings(self):
        """Restore original load balancer settings."""
        if not self.degradation_active:
            return
        
        try:
            # Restore original backend configs
            for instance_id, original_config in self.original_configs.items():
                if instance_id in self.load_balancer.backend_instances:
                    instance = self.load_balancer.backend_instances[instance_id]
                    if hasattr(instance, 'config'):
                        instance.config = original_config
            
            # Restore concurrent request limits
            if hasattr(self.load_balancer, 'max_concurrent_requests'):
                self.load_balancer.max_concurrent_requests = 100  # Default value
            
            self.degradation_active = False
            self.logger.info("Load balancer settings restored to original")
            
        except Exception as e:
            self.logger.error(f"Error restoring load balancer settings: {e}")
    
    def _store_original_configs(self):
        """Store original configurations for restoration."""
        self.original_configs.clear()
        for instance_id, instance in self.load_balancer.backend_instances.items():
            if hasattr(instance, 'config'):
                # Create a copy of the config
                self.original_configs[instance_id] = instance.config


class RAGDegradationAdapter:
    """Adapts RAG processing behavior based on degradation levels."""
    
    def __init__(self, clinical_rag: ClinicalMetabolomicsRAG):
        self.clinical_rag = clinical_rag
        self.logger = logging.getLogger(__name__)
        self.original_settings: Dict[str, Any] = {}
        self.degradation_active = False
    
    def apply_degradation(self, load_level: SystemLoadLevel, 
                         simplified_params: Dict[str, Any],
                         feature_states: Dict[str, bool]):
        """Apply degradation settings to RAG processing."""
        try:
            # Store original settings if first time
            if not self.degradation_active:
                self._store_original_settings()
                self.degradation_active = True
            
            # Update configuration based on degradation level
            if hasattr(self.clinical_rag, 'config'):
                config = self.clinical_rag.config
                
                # Adjust processing settings
                config['enable_confidence_analysis'] = feature_states.get('confidence_analysis', True)
                config['enable_detailed_logging'] = feature_states.get('detailed_logging', True)
                config['enable_complex_analytics'] = feature_states.get('complex_analytics', True)
                config['enable_query_preprocessing'] = feature_states.get('query_preprocessing', True)
                
                # Update token limits and complexity
                if load_level >= SystemLoadLevel.HIGH:
                    config['max_tokens'] = min(config.get('max_tokens', 8000), simplified_params.get('max_total_tokens', 4000))
                    config['enable_parallel_processing'] = False
                
                if load_level >= SystemLoadLevel.CRITICAL:
                    config['max_tokens'] = min(config.get('max_tokens', 4000), 2000)
                    config['skip_confidence_analysis'] = True
                
                if load_level >= SystemLoadLevel.EMERGENCY:
                    config['max_tokens'] = 1000
                    config['emergency_mode'] = True
            
            # Update circuit breaker settings if available
            if hasattr(self.clinical_rag, 'circuit_breaker_config'):
                if load_level >= SystemLoadLevel.HIGH:
                    self.clinical_rag.circuit_breaker_config['failure_threshold'] = 3
                    self.clinical_rag.circuit_breaker_config['recovery_timeout'] = 30.0
                
                if load_level >= SystemLoadLevel.CRITICAL:
                    self.clinical_rag.circuit_breaker_config['failure_threshold'] = 2
                    self.clinical_rag.circuit_breaker_config['recovery_timeout'] = 15.0
            
            self.logger.info(f"RAG processing adapted for {load_level.name}")
            
        except Exception as e:
            self.logger.error(f"Error applying degradation to RAG: {e}")
    
    def restore_original_settings(self):
        """Restore original RAG settings."""
        if not self.degradation_active:
            return
        
        try:
            if hasattr(self.clinical_rag, 'config') and self.original_settings:
                # Restore original configuration
                self.clinical_rag.config.update(self.original_settings)
            
            self.degradation_active = False
            self.logger.info("RAG settings restored to original")
            
        except Exception as e:
            self.logger.error(f"Error restoring RAG settings: {e}")
    
    def _store_original_settings(self):
        """Store original RAG settings."""
        if hasattr(self.clinical_rag, 'config'):
            self.original_settings = self.clinical_rag.config.copy()


class FallbackDegradationAdapter:
    """Adapts fallback system behavior based on degradation levels."""
    
    def __init__(self, fallback_orchestrator: FallbackOrchestrator):
        self.fallback_orchestrator = fallback_orchestrator
        self.logger = logging.getLogger(__name__)
        self.original_settings: Dict[str, Any] = {}
        self.degradation_active = False
    
    def apply_degradation(self, load_level: SystemLoadLevel):
        """Apply degradation settings to fallback system."""
        try:
            # Store original settings if first time
            if not self.degradation_active:
                self._store_original_settings()
                self.degradation_active = True
            
            # Adjust fallback behavior based on load level
            if hasattr(self.fallback_orchestrator, 'config'):
                config = self.fallback_orchestrator.config
                
                if load_level >= SystemLoadLevel.HIGH:
                    # Skip lower-priority fallback levels under high load
                    config['skip_llm_fallback'] = True
                    config['prefer_cache_responses'] = True
                
                if load_level >= SystemLoadLevel.CRITICAL:
                    # Only use essential fallback levels
                    config['emergency_mode'] = True
                    config['max_fallback_attempts'] = 2
                
                if load_level >= SystemLoadLevel.EMERGENCY:
                    # Minimal fallback functionality
                    config['emergency_only'] = True
                    config['max_fallback_attempts'] = 1
            
            self.logger.info(f"Fallback system adapted for {load_level.name}")
            
        except Exception as e:
            self.logger.error(f"Error applying degradation to fallback system: {e}")
    
    def restore_original_settings(self):
        """Restore original fallback settings."""
        if not self.degradation_active:
            return
        
        try:
            if hasattr(self.fallback_orchestrator, 'config') and self.original_settings:
                self.fallback_orchestrator.config.update(self.original_settings)
            
            self.degradation_active = False
            self.logger.info("Fallback system settings restored to original")
            
        except Exception as e:
            self.logger.error(f"Error restoring fallback settings: {e}")
    
    def _store_original_settings(self):
        """Store original fallback settings."""
        if hasattr(self.fallback_orchestrator, 'config'):
            self.original_settings = self.fallback_orchestrator.config.copy()


# ============================================================================
# MAIN INTEGRATION ORCHESTRATOR
# ============================================================================

class ProductionDegradationIntegration:
    """Main integration orchestrator for production degradation system."""
    
    def __init__(self,
                 production_load_balancer: Optional[ProductionLoadBalancer] = None,
                 clinical_rag: Optional[ClinicalMetabolomicsRAG] = None,
                 fallback_orchestrator: Optional[FallbackOrchestrator] = None,
                 production_monitoring: Optional[ProductionMonitoring] = None,
                 load_thresholds: Optional[LoadThresholds] = None,
                 monitoring_interval: float = 5.0):
        
        self.logger = logging.getLogger(__name__)
        
        # Core degradation manager
        self.degradation_manager = create_production_degradation_system(
            load_thresholds=load_thresholds,
            monitoring_interval=monitoring_interval
        )
        
        # Production components
        self.production_load_balancer = production_load_balancer
        self.clinical_rag = clinical_rag
        self.fallback_orchestrator = fallback_orchestrator
        self.production_monitoring = production_monitoring
        
        # Integration adapters
        self.adapters: Dict[str, Any] = {}
        self._initialize_adapters()
        
        # State tracking
        self.integration_active = False
        self.current_load_level = SystemLoadLevel.NORMAL
        
        # Register for load changes
        self.degradation_manager.add_load_change_callback(self._on_load_change)
    
    def _initialize_adapters(self):
        """Initialize integration adapters for available components."""
        if self.production_load_balancer:
            self.adapters['load_balancer'] = LoadBalancerDegradationAdapter(
                self.production_load_balancer
            )
            self.logger.info("Load balancer adapter initialized")
        
        if self.clinical_rag:
            self.adapters['rag'] = RAGDegradationAdapter(self.clinical_rag)
            self.logger.info("RAG adapter initialized")
        
        if self.fallback_orchestrator:
            self.adapters['fallback'] = FallbackDegradationAdapter(self.fallback_orchestrator)
            self.logger.info("Fallback adapter initialized")
    
    async def start(self):
        """Start the integrated degradation system."""
        if self.integration_active:
            return
        
        try:
            # Start core degradation manager
            await self.degradation_manager.start()
            
            # Initialize production monitoring integration
            if self.production_monitoring:
                await self._integrate_monitoring()
            
            self.integration_active = True
            self.logger.info("Production degradation integration started")
            
        except Exception as e:
            self.logger.error(f"Error starting production degradation integration: {e}")
            raise
    
    async def stop(self):
        """Stop the integrated degradation system."""
        if not self.integration_active:
            return
        
        try:
            # Restore original settings in all adapters
            for adapter_name, adapter in self.adapters.items():
                try:
                    adapter.restore_original_settings()
                except Exception as e:
                    self.logger.error(f"Error restoring {adapter_name} settings: {e}")
            
            # Stop core degradation manager
            await self.degradation_manager.stop()
            
            self.integration_active = False
            self.logger.info("Production degradation integration stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping production degradation integration: {e}")
    
    def _on_load_change(self, load_level: SystemLoadLevel, metrics: SystemLoadMetrics):
        """Handle load level changes across all integrated components."""
        previous_level = self.current_load_level
        self.current_load_level = load_level
        
        try:
            # Get current settings from degradation manager
            timeouts = self.degradation_manager.timeout_manager.get_all_timeouts()
            feature_states = {
                'confidence_analysis': self.degradation_manager.is_feature_enabled('confidence_analysis'),
                'detailed_logging': self.degradation_manager.is_feature_enabled('detailed_logging'),
                'complex_analytics': self.degradation_manager.is_feature_enabled('complex_analytics'),
                'confidence_scoring': self.degradation_manager.is_feature_enabled('confidence_scoring'),
                'query_preprocessing': self.degradation_manager.is_feature_enabled('query_preprocessing')
            }
            
            # Apply degradation to all adapters
            if 'load_balancer' in self.adapters:
                if load_level == SystemLoadLevel.NORMAL:
                    self.adapters['load_balancer'].restore_original_settings()
                else:
                    self.adapters['load_balancer'].apply_degradation(load_level, timeouts)
            
            if 'rag' in self.adapters:
                if load_level == SystemLoadLevel.NORMAL:
                    self.adapters['rag'].restore_original_settings()
                else:
                    simplified_params = self.degradation_manager.simplify_query_params({
                        'max_total_tokens': 8000,
                        'top_k': 10
                    })
                    self.adapters['rag'].apply_degradation(load_level, simplified_params, feature_states)
            
            if 'fallback' in self.adapters:
                if load_level == SystemLoadLevel.NORMAL:
                    self.adapters['fallback'].restore_original_settings()
                else:
                    self.adapters['fallback'].apply_degradation(load_level)
            
            # Log the integrated change
            self.logger.info(
                f"Production system adapted to load level {load_level.name} "
                f"(CPU: {metrics.cpu_utilization:.1f}%, Memory: {metrics.memory_pressure:.1f}%, "
                f"Queue: {metrics.request_queue_depth}, Load Score: {metrics.load_score:.3f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error applying integrated degradation: {e}")
    
    async def _integrate_monitoring(self):
        """Integrate with production monitoring system."""
        if not self.production_monitoring:
            return
        
        try:
            # Add degradation metrics to monitoring
            def add_degradation_metrics():
                status = self.degradation_manager.get_current_status()
                return {
                    'degradation_load_level': status['load_level'],
                    'degradation_load_score': status['load_score'],
                    'degradation_active': status['degradation_active'],
                    'degradation_timeouts': status['current_timeouts'],
                    'degradation_limits': status['resource_limits']
                }
            
            # Register custom metrics if monitoring supports it
            if hasattr(self.production_monitoring, 'add_custom_metrics'):
                self.production_monitoring.add_custom_metrics('degradation', add_degradation_metrics)
                self.logger.info("Degradation metrics integrated with production monitoring")
            
        except Exception as e:
            self.logger.error(f"Error integrating with production monitoring: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            'integration_active': self.integration_active,
            'current_load_level': self.current_load_level.name,
            'adapters_active': list(self.adapters.keys()),
            'degradation_status': self.degradation_manager.get_current_status(),
            'components_integrated': {
                'load_balancer': self.production_load_balancer is not None,
                'clinical_rag': self.clinical_rag is not None,
                'fallback_orchestrator': self.fallback_orchestrator is not None,
                'production_monitoring': self.production_monitoring is not None
            }
        }
    
    def force_load_level(self, load_level: SystemLoadLevel):
        """Force a specific load level for testing purposes."""
        # Create mock metrics for the specified load level
        mock_metrics = SystemLoadMetrics(
            timestamp=datetime.now(),
            cpu_utilization=50.0 if load_level == SystemLoadLevel.NORMAL else 85.0,
            memory_pressure=50.0 if load_level == SystemLoadLevel.NORMAL else 80.0,
            request_queue_depth=5 if load_level == SystemLoadLevel.NORMAL else 50,
            response_time_p95=1000.0 if load_level == SystemLoadLevel.NORMAL else 4000.0,
            response_time_p99=2000.0 if load_level == SystemLoadLevel.NORMAL else 6000.0,
            error_rate=0.1 if load_level == SystemLoadLevel.NORMAL else 2.0,
            active_connections=10,
            disk_io_wait=0.0,
            load_level=load_level,
            load_score=0.2 if load_level == SystemLoadLevel.NORMAL else 0.8,
            degradation_recommended=load_level > SystemLoadLevel.NORMAL
        )
        
        self._on_load_change(load_level, mock_metrics)
        self.logger.info(f"Forced load level to {load_level.name} for testing")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_integrated_production_system(
    production_load_balancer: Optional[ProductionLoadBalancer] = None,
    clinical_rag: Optional[ClinicalMetabolomicsRAG] = None,
    fallback_orchestrator: Optional[FallbackOrchestrator] = None,
    production_monitoring: Optional[ProductionMonitoring] = None,
    load_thresholds: Optional[LoadThresholds] = None,
    monitoring_interval: float = 5.0
) -> ProductionDegradationIntegration:
    """Create a fully integrated production degradation system."""
    
    return ProductionDegradationIntegration(
        production_load_balancer=production_load_balancer,
        clinical_rag=clinical_rag,
        fallback_orchestrator=fallback_orchestrator,
        production_monitoring=production_monitoring,
        load_thresholds=load_thresholds,
        monitoring_interval=monitoring_interval
    )


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

async def demo_production_integration():
    """Demonstrate the production integration system."""
    
    # Create integration system (with mock components for demo)
    integration = create_integrated_production_system()
    
    try:
        # Start the integration
        await integration.start()
        print("Production degradation integration started")
        
        # Simulate different load levels
        print("\nTesting load level transitions:")
        
        for level in [SystemLoadLevel.NORMAL, SystemLoadLevel.HIGH, 
                     SystemLoadLevel.CRITICAL, SystemLoadLevel.NORMAL]:
            integration.force_load_level(level)
            status = integration.get_integration_status()
            print(f"Load Level: {status['current_load_level']}")
            print(f"Adapters Active: {status['adapters_active']}")
            await asyncio.sleep(2)
        
        print("\nIntegration demonstration complete")
        
    finally:
        # Stop the integration
        await integration.stop()
        print("Production degradation integration stopped")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_production_integration())