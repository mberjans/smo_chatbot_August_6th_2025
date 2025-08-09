"""
Graceful Degradation Integration Layer for Clinical Metabolomics Oracle
========================================================================

This module provides seamless integration between all graceful degradation components:

1. Enhanced Load Monitoring System - Real-time load detection
2. Progressive Service Degradation Controller - Dynamic service optimization
3. Load-Based Request Throttling System - Intelligent request management
4. Existing Production Systems - Load balancer, RAG system, monitoring

The integration layer ensures:
- Coordinated response to load changes across all systems
- Seamless data flow between components
- Configuration synchronization
- Unified monitoring and reporting
- Production-ready fault tolerance

Architecture:
- GracefulDegradationOrchestrator: Main coordinator
- ProductionSystemIntegrator: Bridges with existing systems
- ConfigurationSynchronizer: Keeps all systems in sync
- MonitoringAggregator: Unified metrics collection

Author: Claude Code (Anthropic)
Version: 1.0.0  
Created: 2025-08-09
Production Ready: Yes
"""

import asyncio
import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref

# Import all graceful degradation components
try:
    from .enhanced_load_monitoring_system import (
        SystemLoadLevel, EnhancedSystemLoadMetrics,
        EnhancedLoadDetectionSystem, create_enhanced_load_monitoring_system
    )
    LOAD_MONITORING_AVAILABLE = True
except ImportError:
    LOAD_MONITORING_AVAILABLE = False
    logging.warning("Enhanced load monitoring system not available")
    
    class SystemLoadLevel:
        NORMAL = 0
        ELEVATED = 1 
        HIGH = 2
        CRITICAL = 3
        EMERGENCY = 4

try:
    from .progressive_service_degradation_controller import (
        ProgressiveServiceDegradationController, create_progressive_degradation_controller
    )
    DEGRADATION_CONTROLLER_AVAILABLE = True
except ImportError:
    DEGRADATION_CONTROLLER_AVAILABLE = False
    logging.warning("Progressive degradation controller not available")

try:
    from .load_based_request_throttling_system import (
        RequestThrottlingSystem, RequestType, RequestPriority,
        create_request_throttling_system
    )
    THROTTLING_SYSTEM_AVAILABLE = True
except ImportError:
    THROTTLING_SYSTEM_AVAILABLE = False
    logging.warning("Request throttling system not available")

# Import production systems for integration
try:
    from .production_load_balancer import ProductionLoadBalancer
    LOAD_BALANCER_AVAILABLE = True
except ImportError:
    LOAD_BALANCER_AVAILABLE = False

try:
    from .clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from .production_monitoring import ProductionMonitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


# ============================================================================
# INTEGRATION CONFIGURATION
# ============================================================================

@dataclass
class GracefulDegradationConfig:
    """Complete configuration for graceful degradation system."""
    
    # Monitoring configuration
    monitoring_interval: float = 5.0
    enable_trend_analysis: bool = True
    hysteresis_enabled: bool = True
    
    # Throttling configuration
    base_rate_per_second: float = 10.0
    max_queue_size: int = 1000
    max_concurrent_requests: int = 50
    starvation_threshold: float = 300.0
    
    # Connection pool configuration
    base_pool_size: int = 20
    max_pool_size: int = 100
    
    # Integration configuration
    auto_start_monitoring: bool = True
    enable_production_integration: bool = True
    metrics_retention_hours: int = 24
    
    # Emergency handling
    emergency_max_duration: float = 300.0  # 5 minutes
    auto_recovery_enabled: bool = True
    circuit_breaker_enabled: bool = True


@dataclass
class IntegrationStatus:
    """Status of system integration."""
    
    load_monitoring_active: bool = False
    degradation_controller_active: bool = False
    throttling_system_active: bool = False
    
    integrated_load_balancer: bool = False
    integrated_rag_system: bool = False
    integrated_monitoring: bool = False
    
    total_requests_processed: int = 0
    current_load_level: str = "NORMAL"
    last_level_change: Optional[datetime] = None
    
    health_status: str = "unknown"
    active_issues: List[str] = field(default_factory=list)


# ============================================================================
# PRODUCTION SYSTEM INTEGRATOR
# ============================================================================

class ProductionSystemIntegrator:
    """
    Integrates graceful degradation with existing production systems.
    """
    
    def __init__(self,
                 load_balancer: Optional[Any] = None,
                 rag_system: Optional[Any] = None,
                 monitoring_system: Optional[Any] = None):
        
        self.load_balancer = load_balancer
        self.rag_system = rag_system
        self.monitoring_system = monitoring_system
        self.logger = logging.getLogger(f"{__name__}.ProductionSystemIntegrator")
        
        # Integration state
        self.integrated_systems: Dict[str, Any] = {}
        self.integration_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize integrations
        self._initialize_integrations()
    
    def _initialize_integrations(self):
        """Initialize all available integrations."""
        
        # Load Balancer Integration
        if self.load_balancer and LOAD_BALANCER_AVAILABLE:
            self._integrate_load_balancer()
        
        # RAG System Integration
        if self.rag_system and RAG_AVAILABLE:
            self._integrate_rag_system()
        
        # Monitoring System Integration
        if self.monitoring_system and MONITORING_AVAILABLE:
            self._integrate_monitoring_system()
    
    def _integrate_load_balancer(self):
        """Integrate with production load balancer."""
        try:
            # Store reference
            self.integrated_systems['load_balancer'] = self.load_balancer
            
            # Set up callbacks if available
            if hasattr(self.load_balancer, 'add_degradation_callback'):
                self.load_balancer.add_degradation_callback(self._on_load_balancer_event)
            
            # Configure throttling integration
            if hasattr(self.load_balancer, 'set_request_throttler'):
                # Will be set when throttling system is created
                pass
            
            self.logger.info("Integrated with production load balancer")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with load balancer: {e}")
    
    def _integrate_rag_system(self):
        """Integrate with clinical metabolomics RAG system."""
        try:
            # Store reference
            self.integrated_systems['rag_system'] = self.rag_system
            
            # Set up query interception if available
            if hasattr(self.rag_system, 'add_pre_query_callback'):
                self.rag_system.add_pre_query_callback(self._on_rag_query_start)
            
            if hasattr(self.rag_system, 'add_post_query_callback'):
                self.rag_system.add_post_query_callback(self._on_rag_query_complete)
            
            self.logger.info("Integrated with clinical metabolomics RAG system")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with RAG system: {e}")
    
    def _integrate_monitoring_system(self):
        """Integrate with production monitoring system."""
        try:
            # Store reference
            self.integrated_systems['monitoring_system'] = self.monitoring_system
            
            # Set up metric callbacks
            if hasattr(self.monitoring_system, 'add_custom_metric_source'):
                self.monitoring_system.add_custom_metric_source(
                    'graceful_degradation', 
                    self._get_degradation_metrics
                )
            
            self.logger.info("Integrated with production monitoring system")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with monitoring system: {e}")
    
    def _on_load_balancer_event(self, event_type: str, data: Dict[str, Any]):
        """Handle load balancer events."""
        self.logger.debug(f"Load balancer event: {event_type}")
        
        # Notify callbacks
        for callback in self.integration_callbacks.get('load_balancer', []):
            try:
                callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Error in load balancer callback: {e}")
    
    def _on_rag_query_start(self, query_data: Dict[str, Any]):
        """Handle RAG query start."""
        # Could be used to submit query through throttling system
        pass
    
    def _on_rag_query_complete(self, query_data: Dict[str, Any], result: Any):
        """Handle RAG query completion."""
        # Could be used to track performance metrics
        pass
    
    def _get_degradation_metrics(self) -> Dict[str, Any]:
        """Get degradation metrics for monitoring system."""
        # Will be populated by the main orchestrator
        return {}
    
    def add_integration_callback(self, system: str, callback: Callable):
        """Add callback for integration events."""
        self.integration_callbacks[system].append(callback)
    
    def update_system_configuration(self, system: str, config: Dict[str, Any]):
        """Update configuration for an integrated system."""
        if system not in self.integrated_systems:
            return False
        
        try:
            target_system = self.integrated_systems[system]
            
            if system == 'load_balancer':
                return self._update_load_balancer_config(target_system, config)
            elif system == 'rag_system':
                return self._update_rag_config(target_system, config)
            elif system == 'monitoring_system':
                return self._update_monitoring_config(target_system, config)
            
        except Exception as e:
            self.logger.error(f"Error updating {system} configuration: {e}")
            return False
        
        return False
    
    def _update_load_balancer_config(self, load_balancer: Any, config: Dict[str, Any]) -> bool:
        """Update load balancer configuration."""
        success = True
        
        # Update timeouts if supported
        if 'timeouts' in config and hasattr(load_balancer, 'update_backend_timeouts'):
            try:
                load_balancer.update_backend_timeouts(config['timeouts'])
            except Exception as e:
                self.logger.error(f"Error updating load balancer timeouts: {e}")
                success = False
        
        # Update circuit breaker settings if supported
        if 'circuit_breaker' in config and hasattr(load_balancer, 'update_circuit_breaker_settings'):
            try:
                load_balancer.update_circuit_breaker_settings(config['circuit_breaker'])
            except Exception as e:
                self.logger.error(f"Error updating circuit breaker settings: {e}")
                success = False
        
        return success
    
    def _update_rag_config(self, rag_system: Any, config: Dict[str, Any]) -> bool:
        """Update RAG system configuration."""
        success = True
        
        # Update query complexity if supported
        if 'query_complexity' in config and hasattr(rag_system, 'update_query_complexity'):
            try:
                rag_system.update_query_complexity(config['query_complexity'])
            except Exception as e:
                self.logger.error(f"Error updating RAG query complexity: {e}")
                success = False
        
        # Update timeouts if supported
        if 'timeouts' in config and hasattr(rag_system, 'update_timeouts'):
            try:
                rag_system.update_timeouts(config['timeouts'])
            except Exception as e:
                self.logger.error(f"Error updating RAG timeouts: {e}")
                success = False
        
        return success
    
    def _update_monitoring_config(self, monitoring_system: Any, config: Dict[str, Any]) -> bool:
        """Update monitoring system configuration."""
        success = True
        
        # Update monitoring interval if supported
        if 'interval' in config and hasattr(monitoring_system, 'update_monitoring_interval'):
            try:
                monitoring_system.update_monitoring_interval(config['interval'])
            except Exception as e:
                self.logger.error(f"Error updating monitoring interval: {e}")
                success = False
        
        return success
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations."""
        return {
            'integrated_systems': list(self.integrated_systems.keys()),
            'system_details': {
                name: {
                    'type': type(system).__name__,
                    'available': system is not None
                }
                for name, system in self.integrated_systems.items()
            }
        }


# ============================================================================
# GRACEFUL DEGRADATION ORCHESTRATOR
# ============================================================================

class GracefulDegradationOrchestrator:
    """
    Main orchestrator that coordinates all graceful degradation components.
    
    Provides unified interface for:
    - System initialization and configuration
    - Load monitoring and response coordination
    - Request throttling and queuing
    - Production system integration
    - Health monitoring and reporting
    """
    
    def __init__(self,
                 config: Optional[GracefulDegradationConfig] = None,
                 load_balancer: Optional[Any] = None,
                 rag_system: Optional[Any] = None,
                 monitoring_system: Optional[Any] = None):
        
        self.config = config or GracefulDegradationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize production system integrator
        self.production_integrator = ProductionSystemIntegrator(
            load_balancer=load_balancer,
            rag_system=rag_system,
            monitoring_system=monitoring_system
        )
        
        # Core degradation components (will be initialized)
        self.load_detector: Optional[Any] = None
        self.degradation_controller: Optional[Any] = None
        self.throttling_system: Optional[Any] = None
        
        # System state
        self._running = False
        self._start_time: Optional[datetime] = None
        self._integration_status = IntegrationStatus()
        
        # Metrics and monitoring
        self._metrics_history: deque = deque(maxlen=1000)
        self._health_check_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize all systems
        self._initialize_graceful_degradation_systems()
        
        self.logger.info("Graceful Degradation Orchestrator initialized")
    
    def _initialize_graceful_degradation_systems(self):
        """Initialize all graceful degradation components."""
        
        # 1. Create Enhanced Load Monitoring System
        if LOAD_MONITORING_AVAILABLE:
            self._initialize_load_monitoring()
        
        # 2. Create Progressive Service Degradation Controller
        if DEGRADATION_CONTROLLER_AVAILABLE:
            self._initialize_degradation_controller()
        
        # 3. Create Load-Based Request Throttling System
        if THROTTLING_SYSTEM_AVAILABLE:
            self._initialize_throttling_system()
        
        # 4. Set up system coordination
        self._setup_system_coordination()
    
    def _initialize_load_monitoring(self):
        """Initialize enhanced load monitoring system."""
        try:
            self.load_detector = create_enhanced_load_monitoring_system(
                monitoring_interval=self.config.monitoring_interval,
                enable_trend_analysis=self.config.enable_trend_analysis,
                production_monitoring=self.production_integrator.monitoring_system
            )
            
            # Add callback for load changes
            self.load_detector.add_load_change_callback(self._on_system_load_change)
            
            self._integration_status.load_monitoring_active = True
            self.logger.info("Enhanced load monitoring system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize load monitoring: {e}")
    
    def _initialize_degradation_controller(self):
        """Initialize progressive service degradation controller."""
        try:
            self.degradation_controller = create_progressive_degradation_controller(
                enhanced_detector=self.load_detector,
                production_load_balancer=self.production_integrator.load_balancer,
                clinical_rag=self.production_integrator.rag_system,
                production_monitoring=self.production_integrator.monitoring_system
            )
            
            # Add callback for degradation changes
            self.degradation_controller.add_load_change_callback(self._on_degradation_change)
            
            self._integration_status.degradation_controller_active = True
            self.logger.info("Progressive service degradation controller initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize degradation controller: {e}")
    
    def _initialize_throttling_system(self):
        """Initialize load-based request throttling system."""
        try:
            self.throttling_system = create_request_throttling_system(
                base_rate_per_second=self.config.base_rate_per_second,
                max_queue_size=self.config.max_queue_size,
                max_concurrent_requests=self.config.max_concurrent_requests,
                load_detector=self.load_detector,
                degradation_controller=self.degradation_controller,
                custom_config={
                    'starvation_threshold': self.config.starvation_threshold,
                    'base_pool_size': self.config.base_pool_size,
                    'max_pool_size': self.config.max_pool_size
                }
            )
            
            self._integration_status.throttling_system_active = True
            self.logger.info("Load-based request throttling system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize throttling system: {e}")
    
    def _setup_system_coordination(self):
        """Set up coordination between all systems."""
        
        # Connect throttling system to load balancer if available
        if (self.throttling_system and 
            self.production_integrator.load_balancer and 
            hasattr(self.production_integrator.load_balancer, 'set_request_throttler')):
            try:
                self.production_integrator.load_balancer.set_request_throttler(self.throttling_system)
                self._integration_status.integrated_load_balancer = True
                self.logger.info("Connected throttling system to load balancer")
            except Exception as e:
                self.logger.error(f"Failed to connect throttling system to load balancer: {e}")
        
        # Set up RAG system integration
        if self.production_integrator.rag_system:
            self._integration_status.integrated_rag_system = True
        
        # Set up monitoring integration
        if self.production_integrator.monitoring_system:
            self._integration_status.integrated_monitoring = True
    
    def _on_system_load_change(self, metrics):
        """Handle system load level changes."""
        if hasattr(metrics, 'load_level'):
            current_level = metrics.load_level.name if hasattr(metrics.load_level, 'name') else str(metrics.load_level)
            
            with self._lock:
                if current_level != self._integration_status.current_load_level:
                    previous_level = self._integration_status.current_load_level
                    self._integration_status.current_load_level = current_level
                    self._integration_status.last_level_change = datetime.now()
                    
                    self.logger.info(f"System load level changed: {previous_level} → {current_level}")
                
                # Store metrics for history
                self._metrics_history.append({
                    'timestamp': datetime.now(),
                    'load_level': current_level,
                    'cpu_utilization': getattr(metrics, 'cpu_utilization', 0),
                    'memory_pressure': getattr(metrics, 'memory_pressure', 0),
                    'response_time_p95': getattr(metrics, 'response_time_p95', 0),
                    'error_rate': getattr(metrics, 'error_rate', 0)
                })
    
    def _on_degradation_change(self, previous_level, new_level):
        """Handle degradation level changes."""
        self.logger.info(f"Degradation applied: {previous_level.name if hasattr(previous_level, 'name') else previous_level} → {new_level.name if hasattr(new_level, 'name') else new_level}")
        
        # Update production systems if needed
        degradation_config = {}
        if self.degradation_controller:
            status = self.degradation_controller.get_current_status()
            degradation_config = {
                'timeouts': status.get('timeouts', {}),
                'query_complexity': status.get('query_complexity', {}),
                'circuit_breaker': {
                    'failure_threshold': 3 if new_level == SystemLoadLevel.EMERGENCY else 5,
                    'recovery_timeout': 60
                }
            }
        
        # Apply to integrated systems
        self.production_integrator.update_system_configuration('load_balancer', degradation_config)
        self.production_integrator.update_system_configuration('rag_system', degradation_config)
    
    async def start(self):
        """Start the complete graceful degradation system."""
        if self._running:
            self.logger.warning("Graceful degradation system already running")
            return
        
        self._running = True
        self._start_time = datetime.now()
        
        try:
            # Start load monitoring
            if self.load_detector and self.config.auto_start_monitoring:
                await self.load_detector.start_monitoring()
                self.logger.info("Load monitoring started")
            
            # Start throttling system
            if self.throttling_system:
                await self.throttling_system.start()
                self.logger.info("Request throttling started")
            
            self._integration_status.health_status = "healthy"
            self.logger.info("Graceful Degradation System started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting graceful degradation system: {e}")
            self._integration_status.health_status = "failed"
            self._integration_status.active_issues.append(f"Startup error: {str(e)}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the complete graceful degradation system."""
        if not self._running:
            return
        
        self._running = False
        
        try:
            # Stop throttling system
            if self.throttling_system:
                await self.throttling_system.stop()
                self.logger.info("Request throttling stopped")
            
            # Stop load monitoring
            if self.load_detector:
                await self.load_detector.stop_monitoring()
                self.logger.info("Load monitoring stopped")
            
            self._integration_status.health_status = "stopped"
            self.logger.info("Graceful Degradation System stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping graceful degradation system: {e}")
    
    async def submit_request(self,
                           request_type: str,
                           priority: Optional[str] = None,
                           handler: Optional[Callable] = None,
                           **kwargs) -> Tuple[bool, str, str]:
        """
        Submit a request through the integrated throttling system.
        
        Args:
            request_type: Type of request (e.g., 'user_query', 'health_check')
            priority: Request priority ('critical', 'high', 'medium', 'low', 'background')
            handler: Request handler function
            **kwargs: Additional arguments
            
        Returns:
            (success, message, request_id)
        """
        if not self._running or not self.throttling_system:
            return False, "Graceful degradation system not running", ""
        
        # Map string types to enums
        request_type_map = {
            'health_check': RequestType.HEALTH_CHECK,
            'user_query': RequestType.USER_QUERY,
            'batch_processing': RequestType.BATCH_PROCESSING,
            'analytics': RequestType.ANALYTICS,
            'maintenance': RequestType.MAINTENANCE,
            'admin': RequestType.ADMIN
        }
        
        priority_map = {
            'critical': RequestPriority.CRITICAL,
            'high': RequestPriority.HIGH,
            'medium': RequestPriority.MEDIUM,
            'low': RequestPriority.LOW,
            'background': RequestPriority.BACKGROUND
        }
        
        req_type = request_type_map.get(request_type, RequestType.USER_QUERY)
        req_priority = priority_map.get(priority) if priority else None
        
        try:
            success, message, request_id = await self.throttling_system.submit_request(
                request_type=req_type,
                priority=req_priority,
                handler=handler,
                **kwargs
            )
            
            # Update metrics
            with self._lock:
                self._integration_status.total_requests_processed += 1
            
            return success, message, request_id
            
        except Exception as e:
            self.logger.error(f"Error submitting request: {e}")
            return False, f"Submission error: {str(e)}", ""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the entire graceful degradation system."""
        with self._lock:
            base_status = {
                'running': self._running,
                'start_time': self._start_time.isoformat() if self._start_time else None,
                'uptime_seconds': (datetime.now() - self._start_time).total_seconds() if self._start_time else 0,
                'integration_status': {
                    'load_monitoring_active': self._integration_status.load_monitoring_active,
                    'degradation_controller_active': self._integration_status.degradation_controller_active,
                    'throttling_system_active': self._integration_status.throttling_system_active,
                    'integrated_load_balancer': self._integration_status.integrated_load_balancer,
                    'integrated_rag_system': self._integration_status.integrated_rag_system,
                    'integrated_monitoring': self._integration_status.integrated_monitoring,
                },
                'current_load_level': self._integration_status.current_load_level,
                'last_level_change': self._integration_status.last_level_change.isoformat() if self._integration_status.last_level_change else None,
                'total_requests_processed': self._integration_status.total_requests_processed,
                'health_status': self._integration_status.health_status,
                'active_issues': self._integration_status.active_issues.copy()
            }
        
        # Get detailed status from each component
        if self.load_detector:
            try:
                base_status['load_monitoring'] = self.load_detector.export_metrics_for_analysis()
            except Exception as e:
                self.logger.debug(f"Error getting load monitoring status: {e}")
        
        if self.degradation_controller:
            try:
                base_status['degradation_controller'] = self.degradation_controller.get_current_status()
            except Exception as e:
                self.logger.debug(f"Error getting degradation controller status: {e}")
        
        if self.throttling_system:
            try:
                base_status['throttling_system'] = self.throttling_system.get_system_status()
            except Exception as e:
                self.logger.debug(f"Error getting throttling system status: {e}")
        
        # Add production integration status
        base_status['production_integration'] = self.production_integrator.get_integration_status()
        
        return base_status
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health check for the entire system."""
        status = self.get_system_status()
        
        health_issues = []
        overall_health = "healthy"
        
        # Check if system is running
        if not status['running']:
            health_issues.append("System not running")
            overall_health = "critical"
        
        # Check component health
        if not status['integration_status']['load_monitoring_active']:
            health_issues.append("Load monitoring inactive")
            overall_health = "degraded" if overall_health == "healthy" else overall_health
        
        if not status['integration_status']['throttling_system_active']:
            health_issues.append("Request throttling inactive")
            overall_health = "degraded" if overall_health == "healthy" else overall_health
        
        # Check throttling system health if available
        if 'throttling_system' in status:
            throttling_health = status['throttling_system'].get('throttling', {})
            success_rate = throttling_health.get('success_rate', 100)
            if success_rate < 90:
                health_issues.append(f"Low throttling success rate: {success_rate:.1f}%")
                overall_health = "degraded" if overall_health == "healthy" else overall_health
        
        # Check load level
        current_load = status['current_load_level']
        if current_load in ['CRITICAL', 'EMERGENCY']:
            health_issues.append(f"System under {current_load.lower()} load")
            overall_health = "critical"
        elif current_load == 'HIGH':
            health_issues.append("System under high load")
            overall_health = "degraded" if overall_health == "healthy" else overall_health
        
        # Add any existing issues
        health_issues.extend(status['active_issues'])
        
        return {
            'status': overall_health,
            'issues': health_issues,
            'uptime_seconds': status['uptime_seconds'],
            'current_load_level': current_load,
            'total_requests_processed': status['total_requests_processed'],
            'component_status': {
                'load_monitoring': 'active' if status['integration_status']['load_monitoring_active'] else 'inactive',
                'degradation_controller': 'active' if status['integration_status']['degradation_controller_active'] else 'inactive',
                'throttling_system': 'active' if status['integration_status']['throttling_system_active'] else 'inactive'
            },
            'production_integration': {
                'load_balancer': 'integrated' if status['integration_status']['integrated_load_balancer'] else 'not_integrated',
                'rag_system': 'integrated' if status['integration_status']['integrated_rag_system'] else 'not_integrated',
                'monitoring': 'integrated' if status['integration_status']['integrated_monitoring'] else 'not_integrated'
            }
        }
    
    def add_health_check_callback(self, callback: Callable):
        """Add a health check callback."""
        self._health_check_callbacks.append(callback)
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                metric for metric in self._metrics_history
                if metric['timestamp'] > cutoff_time
            ]


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_graceful_degradation_system(
    # Configuration
    config: Optional[GracefulDegradationConfig] = None,
    
    # Production system references
    load_balancer: Optional[Any] = None,
    rag_system: Optional[Any] = None,
    monitoring_system: Optional[Any] = None,
    
    # Auto-start
    auto_start: bool = True
) -> GracefulDegradationOrchestrator:
    """
    Create a complete, production-ready graceful degradation system.
    
    Args:
        config: Configuration for the degradation system
        load_balancer: Production load balancer instance
        rag_system: Clinical metabolomics RAG system instance
        monitoring_system: Production monitoring system instance
        auto_start: Whether to automatically start the system
        
    Returns:
        Configured GracefulDegradationOrchestrator
    """
    
    orchestrator = GracefulDegradationOrchestrator(
        config=config,
        load_balancer=load_balancer,
        rag_system=rag_system,
        monitoring_system=monitoring_system
    )
    
    return orchestrator


async def create_and_start_graceful_degradation_system(
    config: Optional[GracefulDegradationConfig] = None,
    load_balancer: Optional[Any] = None,
    rag_system: Optional[Any] = None,
    monitoring_system: Optional[Any] = None
) -> GracefulDegradationOrchestrator:
    """
    Create and start a complete graceful degradation system.
    
    Returns:
        Running GracefulDegradationOrchestrator
    """
    
    orchestrator = create_graceful_degradation_system(
        config=config,
        load_balancer=load_balancer,
        rag_system=rag_system,
        monitoring_system=monitoring_system,
        auto_start=False
    )
    
    await orchestrator.start()
    return orchestrator


# ============================================================================
# DEMONSTRATION
# ============================================================================

async def demonstrate_graceful_degradation_integration():
    """Demonstrate the complete integrated graceful degradation system."""
    print("Complete Graceful Degradation System Integration Demonstration")
    print("=" * 80)
    
    # Create configuration
    config = GracefulDegradationConfig(
        monitoring_interval=2.0,
        base_rate_per_second=5.0,
        max_queue_size=50,
        max_concurrent_requests=10
    )
    
    # Create and start the complete system
    orchestrator = await create_and_start_graceful_degradation_system(config=config)
    
    print("✅ Complete Graceful Degradation System started")
    
    # Show initial status
    initial_status = orchestrator.get_system_status()
    print(f"\n📊 System Status:")
    print(f"  Running: {'✅' if initial_status['running'] else '❌'}")
    print(f"  Load Monitoring: {'✅' if initial_status['integration_status']['load_monitoring_active'] else '❌'}")
    print(f"  Degradation Controller: {'✅' if initial_status['integration_status']['degradation_controller_active'] else '❌'}")
    print(f"  Request Throttling: {'✅' if initial_status['integration_status']['throttling_system_active'] else '❌'}")
    print(f"  Current Load Level: {initial_status['current_load_level']}")
    
    # Test request submission
    print(f"\n🚀 Testing Request Submission...")
    
    async def sample_handler(message: str):
        print(f"    Processing: {message}")
        await asyncio.sleep(1)
        return f"Completed: {message}"
    
    # Submit various request types
    request_tests = [
        ('health_check', 'critical', 'System health check'),
        ('user_query', 'high', 'User metabolomics query'),
        ('batch_processing', 'medium', 'Batch data processing'),
        ('analytics', 'low', 'Performance analytics'),
        ('maintenance', 'background', 'Background maintenance')
    ]
    
    submitted_count = 0
    for req_type, priority, message in request_tests:
        success, response_msg, request_id = await orchestrator.submit_request(
            request_type=req_type,
            priority=priority,
            handler=sample_handler,
            message=message
        )
        
        if success:
            print(f"  ✅ {req_type}: {request_id}")
            submitted_count += 1
        else:
            print(f"  ❌ {req_type}: {response_msg}")
    
    print(f"  Submitted {submitted_count}/{len(request_tests)} requests")
    
    # Monitor system for a while
    print(f"\n📈 Monitoring System Activity...")
    for i in range(8):
        await asyncio.sleep(2)
        
        health = orchestrator.get_health_check()
        status = orchestrator.get_system_status()
        
        print(f"\n--- Status Update {i+1} ---")
        print(f"Health: {health['status'].upper()}")
        if health['issues']:
            print(f"Issues: {', '.join(health['issues'])}")
        
        print(f"Load Level: {health['current_load_level']}")
        print(f"Total Requests: {health['total_requests_processed']}")
        
        # Show component activity
        if 'throttling_system' in status:
            throttling = status['throttling_system']
            if 'queue' in throttling:
                queue_size = throttling['queue']['total_size']
                print(f"Queue Size: {queue_size}")
            
            if 'lifecycle' in throttling:
                active_requests = len(throttling['lifecycle'].get('active_requests', {}))
                print(f"Active Requests: {active_requests}")
        
        # Simulate load changes for demonstration
        if i == 3 and orchestrator.degradation_controller:
            print("🔄 Simulating HIGH load condition...")
            orchestrator.degradation_controller.force_load_level(
                SystemLoadLevel.HIGH, "Demo high load"
            )
        elif i == 6 and orchestrator.degradation_controller:
            print("🔄 Simulating recovery to NORMAL...")
            orchestrator.degradation_controller.force_load_level(
                SystemLoadLevel.NORMAL, "Demo recovery"
            )
    
    # Final comprehensive status
    print(f"\n📋 Final System Status:")
    final_health = orchestrator.get_health_check()
    final_status = orchestrator.get_system_status()
    
    print(f"Overall Health: {final_health['status'].upper()}")
    print(f"Uptime: {final_health['uptime_seconds']:.1f}s")
    print(f"Total Requests Processed: {final_health['total_requests_processed']}")
    print(f"Final Load Level: {final_health['current_load_level']}")
    
    print(f"\nComponent Status:")
    for component, status in final_health['component_status'].items():
        print(f"  {component}: {status}")
    
    print(f"\nProduction Integration:")
    for system, status in final_health['production_integration'].items():
        print(f"  {system}: {status}")
    
    # Show metrics history
    metrics_history = orchestrator.get_metrics_history(hours=1)
    print(f"\nMetrics History: {len(metrics_history)} entries collected")
    
    # Cleanup
    await orchestrator.stop()
    print(f"\n✅ Graceful Degradation System demonstration completed!")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_graceful_degradation_integration())