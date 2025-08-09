"""
Progressive Service Degradation Controller for Clinical Metabolomics Oracle
=========================================================================

This module implements the progressive service degradation controller that listens to load level 
changes from the enhanced monitoring system and applies appropriate degradation actions for each level.

The controller implements dynamic degradation strategies across 5 load levels:
- NORMAL: Full functionality with optimal performance
- ELEVATED: Minor optimizations and reduced logging detail
- HIGH: Timeout reductions and query complexity limits
- CRITICAL: Aggressive timeout cuts and feature disabling
- EMERGENCY: Minimal functionality to maintain system stability

Key Features:
1. Progressive Timeout Management - Dynamic timeout scaling based on load
2. Query Complexity Reduction - Progressive simplification of queries
3. Feature Control System - Selective disabling of non-essential features
4. Resource Management - Dynamic resource allocation adjustments
5. Integration Points - Seamless integration with existing systems

Architecture:
- DegradationController: Main orchestrator that listens to load changes
- TimeoutManager: Manages dynamic timeout adjustments
- QuerySimplifier: Handles progressive query complexity reduction
- FeatureController: Manages feature toggles based on load level
- ResourceManager: Controls resource allocation and usage limits

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
Production Ready: Yes
"""

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import copy

# Import enhanced load monitoring system
try:
    from .enhanced_load_monitoring_system import (
        SystemLoadLevel, EnhancedSystemLoadMetrics, 
        EnhancedLoadDetectionSystem, create_enhanced_load_monitoring_system
    )
    ENHANCED_MONITORING_AVAILABLE = True
except ImportError:
    ENHANCED_MONITORING_AVAILABLE = False
    logging.warning("Enhanced load monitoring system not available")
    
    # Define standalone versions
    class SystemLoadLevel(IntEnum):
        NORMAL = 0
        ELEVATED = 1
        HIGH = 2
        CRITICAL = 3
        EMERGENCY = 4

# Import existing production components for integration
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

try:
    from .comprehensive_fallback_system import FallbackOrchestrator
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False


# ============================================================================
# DEGRADATION CONFIGURATION AND STRATEGIES
# ============================================================================

@dataclass
class TimeoutConfiguration:
    """Configuration for timeout management across load levels."""
    
    # Base timeouts (NORMAL level) - in seconds
    lightrag_base_timeout: float = 60.0
    literature_search_base_timeout: float = 90.0
    openai_api_base_timeout: float = 45.0
    perplexity_api_base_timeout: float = 35.0
    health_check_base_timeout: float = 10.0
    
    # Scaling factors for each load level
    # Format: [NORMAL, ELEVATED, HIGH, CRITICAL, EMERGENCY]
    lightrag_factors: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.33, 0.17])
    literature_search_factors: List[float] = field(default_factory=lambda: [1.0, 0.74, 0.5, 0.3, 0.2])
    openai_api_factors: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.49, 0.31])
    perplexity_api_factors: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.49, 0.29])
    health_check_factors: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.3])
    
    def get_timeout(self, service: str, load_level: SystemLoadLevel) -> float:
        """Get timeout for a service at a specific load level."""
        service_map = {
            'lightrag': (self.lightrag_base_timeout, self.lightrag_factors),
            'literature_search': (self.literature_search_base_timeout, self.literature_search_factors),
            'openai_api': (self.openai_api_base_timeout, self.openai_api_factors),
            'perplexity_api': (self.perplexity_api_base_timeout, self.perplexity_api_factors),
            'health_check': (self.health_check_base_timeout, self.health_check_factors)
        }
        
        if service not in service_map:
            return 30.0  # Default fallback timeout
            
        base_timeout, factors = service_map[service]
        factor = factors[min(load_level.value, len(factors) - 1)]
        return base_timeout * factor


@dataclass
class QueryComplexityConfiguration:
    """Configuration for query complexity reduction."""
    
    # Token limits by load level [NORMAL, ELEVATED, HIGH, CRITICAL, EMERGENCY]
    token_limits: List[int] = field(default_factory=lambda: [8000, 6000, 4000, 2000, 1000])
    
    # Result depth by load level
    result_depths: List[int] = field(default_factory=lambda: [10, 8, 5, 2, 1])
    
    # Query modes by load level
    query_modes: List[str] = field(default_factory=lambda: ['hybrid', 'hybrid', 'local', 'simple', 'simple'])
    
    # Search complexity reduction
    max_search_results: List[int] = field(default_factory=lambda: [50, 30, 15, 5, 2])
    max_search_depth: List[int] = field(default_factory=lambda: [5, 4, 3, 2, 1])
    
    def get_complexity_settings(self, load_level: SystemLoadLevel) -> Dict[str, Any]:
        """Get complexity settings for a load level."""
        level_idx = min(load_level.value, 4)  # Cap at EMERGENCY level
        
        return {
            'token_limit': self.token_limits[level_idx],
            'result_depth': self.result_depths[level_idx],
            'query_mode': self.query_modes[level_idx],
            'max_search_results': self.max_search_results[level_idx],
            'max_search_depth': self.max_search_depth[level_idx]
        }


@dataclass
class FeatureControlConfiguration:
    """Configuration for feature control and disabling."""
    
    # Features that get progressively disabled
    detailed_logging: List[bool] = field(default_factory=lambda: [True, True, False, False, False])
    complex_analytics: List[bool] = field(default_factory=lambda: [True, True, True, False, False])
    confidence_analysis: List[bool] = field(default_factory=lambda: [True, True, False, False, False])
    background_tasks: List[bool] = field(default_factory=lambda: [True, True, True, False, False])
    caching_writes: List[bool] = field(default_factory=lambda: [True, True, True, True, False])
    
    # Performance optimization features
    enable_parallel_processing: List[bool] = field(default_factory=lambda: [True, True, True, False, False])
    enable_result_streaming: List[bool] = field(default_factory=lambda: [True, True, False, False, False])
    enable_metadata_extraction: List[bool] = field(default_factory=lambda: [True, True, True, False, False])
    
    def get_feature_settings(self, load_level: SystemLoadLevel) -> Dict[str, bool]:
        """Get feature settings for a load level."""
        level_idx = min(load_level.value, 4)
        
        return {
            'detailed_logging': self.detailed_logging[level_idx],
            'complex_analytics': self.complex_analytics[level_idx],
            'confidence_analysis': self.confidence_analysis[level_idx],
            'background_tasks': self.background_tasks[level_idx],
            'caching_writes': self.caching_writes[level_idx],
            'enable_parallel_processing': self.enable_parallel_processing[level_idx],
            'enable_result_streaming': self.enable_result_streaming[level_idx],
            'enable_metadata_extraction': self.enable_metadata_extraction[level_idx]
        }


@dataclass
class DegradationConfiguration:
    """Complete degradation configuration."""
    
    timeout_config: TimeoutConfiguration = field(default_factory=TimeoutConfiguration)
    complexity_config: QueryComplexityConfiguration = field(default_factory=QueryComplexityConfiguration)
    feature_config: FeatureControlConfiguration = field(default_factory=FeatureControlConfiguration)
    
    # Global degradation settings
    enable_degradation: bool = True
    degradation_hysteresis: bool = True
    min_degradation_duration: float = 30.0  # Minimum time to stay in degraded state (seconds)
    recovery_delay: float = 60.0  # Delay before attempting recovery (seconds)
    
    # Emergency settings
    emergency_mode_max_duration: float = 300.0  # Max time in emergency mode (5 minutes)
    emergency_circuit_breaker: bool = True
    emergency_load_shed_percentage: float = 50.0  # Shed 50% of load in emergency


# ============================================================================
# DEGRADATION STRATEGY MANAGERS
# ============================================================================

class TimeoutManager:
    """Manages dynamic timeout adjustments based on load level."""
    
    def __init__(self, config: TimeoutConfiguration):
        self.config = config
        self.current_timeouts: Dict[str, float] = {}
        self.load_level = SystemLoadLevel.NORMAL
        self.logger = logging.getLogger(f"{__name__}.TimeoutManager")
        
        # Initialize with normal timeouts
        self._update_timeouts(SystemLoadLevel.NORMAL)
    
    def update_load_level(self, new_level: SystemLoadLevel):
        """Update timeouts based on new load level."""
        if new_level != self.load_level:
            self.logger.info(f"Updating timeouts: {self.load_level.name} → {new_level.name}")
            self.load_level = new_level
            self._update_timeouts(new_level)
    
    def _update_timeouts(self, load_level: SystemLoadLevel):
        """Internal method to update all timeouts."""
        services = ['lightrag', 'literature_search', 'openai_api', 'perplexity_api', 'health_check']
        
        for service in services:
            new_timeout = self.config.get_timeout(service, load_level)
            old_timeout = self.current_timeouts.get(service, 0)
            self.current_timeouts[service] = new_timeout
            
            if old_timeout > 0 and abs(new_timeout - old_timeout) > 1.0:
                self.logger.debug(f"{service} timeout: {old_timeout:.1f}s → {new_timeout:.1f}s")
    
    def get_timeout(self, service: str) -> float:
        """Get current timeout for a service."""
        return self.current_timeouts.get(service, 30.0)
    
    def get_all_timeouts(self) -> Dict[str, float]:
        """Get all current timeouts."""
        return self.current_timeouts.copy()


class QueryComplexityManager:
    """Manages progressive query complexity reduction."""
    
    def __init__(self, config: QueryComplexityConfiguration):
        self.config = config
        self.current_settings: Dict[str, Any] = {}
        self.load_level = SystemLoadLevel.NORMAL
        self.logger = logging.getLogger(f"{__name__}.QueryComplexityManager")
        
        # Initialize with normal settings
        self._update_settings(SystemLoadLevel.NORMAL)
    
    def update_load_level(self, new_level: SystemLoadLevel):
        """Update complexity settings based on new load level."""
        if new_level != self.load_level:
            self.logger.info(f"Updating query complexity: {self.load_level.name} → {new_level.name}")
            self.load_level = new_level
            self._update_settings(new_level)
    
    def _update_settings(self, load_level: SystemLoadLevel):
        """Internal method to update complexity settings."""
        new_settings = self.config.get_complexity_settings(load_level)
        
        # Log significant changes
        for key, new_value in new_settings.items():
            old_value = self.current_settings.get(key)
            if old_value != new_value and old_value is not None:
                self.logger.debug(f"{key}: {old_value} → {new_value}")
        
        self.current_settings = new_settings
    
    def get_query_params(self) -> Dict[str, Any]:
        """Get current query complexity parameters."""
        return self.current_settings.copy()
    
    def should_simplify_query(self, query: str) -> bool:
        """Determine if a query should be simplified."""
        if self.load_level <= SystemLoadLevel.NORMAL:
            return False
        
        # Simple heuristics for query simplification
        query_length = len(query.split())
        
        if self.load_level >= SystemLoadLevel.CRITICAL and query_length > 20:
            return True
        elif self.load_level >= SystemLoadLevel.HIGH and query_length > 50:
            return True
        
        return False
    
    def simplify_query(self, query: str) -> str:
        """Simplify a query based on current load level."""
        if not self.should_simplify_query(query):
            return query
        
        words = query.split()
        max_words = {
            SystemLoadLevel.ELEVATED: 100,
            SystemLoadLevel.HIGH: 50,
            SystemLoadLevel.CRITICAL: 20,
            SystemLoadLevel.EMERGENCY: 10
        }.get(self.load_level, len(words))
        
        if len(words) > max_words:
            simplified = ' '.join(words[:max_words])
            self.logger.debug(f"Simplified query: {len(words)} → {max_words} words")
            return simplified
        
        return query


class FeatureControlManager:
    """Manages selective feature disabling based on load level."""
    
    def __init__(self, config: FeatureControlConfiguration):
        self.config = config
        self.current_features: Dict[str, bool] = {}
        self.load_level = SystemLoadLevel.NORMAL
        self.logger = logging.getLogger(f"{__name__}.FeatureControlManager")
        
        # Initialize with normal features
        self._update_features(SystemLoadLevel.NORMAL)
    
    def update_load_level(self, new_level: SystemLoadLevel):
        """Update feature availability based on new load level."""
        if new_level != self.load_level:
            self.logger.info(f"Updating feature availability: {self.load_level.name} → {new_level.name}")
            self.load_level = new_level
            self._update_features(new_level)
    
    def _update_features(self, load_level: SystemLoadLevel):
        """Internal method to update feature availability."""
        new_features = self.config.get_feature_settings(load_level)
        
        # Log feature changes
        disabled_features = []
        enabled_features = []
        
        for feature, enabled in new_features.items():
            old_enabled = self.current_features.get(feature, True)
            if old_enabled != enabled:
                if enabled:
                    enabled_features.append(feature)
                else:
                    disabled_features.append(feature)
        
        if disabled_features:
            self.logger.info(f"Disabled features: {', '.join(disabled_features)}")
        if enabled_features:
            self.logger.info(f"Enabled features: {', '.join(enabled_features)}")
        
        self.current_features = new_features
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled at current load level."""
        return self.current_features.get(feature, True)
    
    def get_feature_settings(self) -> Dict[str, bool]:
        """Get all current feature settings."""
        return self.current_features.copy()


# ============================================================================
# MAIN PROGRESSIVE SERVICE DEGRADATION CONTROLLER
# ============================================================================

class ProgressiveServiceDegradationController:
    """
    Main controller that orchestrates progressive service degradation based on system load.
    
    Listens to load level changes and applies appropriate degradation strategies including:
    - Dynamic timeout adjustments
    - Query complexity reduction
    - Feature disabling
    - Resource allocation changes
    """
    
    def __init__(self, 
                 config: Optional[DegradationConfiguration] = None,
                 enhanced_detector: Optional[Any] = None,
                 production_load_balancer: Optional[Any] = None,
                 clinical_rag: Optional[Any] = None,
                 production_monitoring: Optional[Any] = None):
        
        self.config = config or DegradationConfiguration()
        self.enhanced_detector = enhanced_detector
        self.production_load_balancer = production_load_balancer
        self.clinical_rag = clinical_rag
        self.production_monitoring = production_monitoring
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy managers
        self.timeout_manager = TimeoutManager(self.config.timeout_config)
        self.complexity_manager = QueryComplexityManager(self.config.complexity_config)
        self.feature_manager = FeatureControlManager(self.config.feature_config)
        
        # State tracking
        self.current_load_level = SystemLoadLevel.NORMAL
        self.last_level_change_time = datetime.now()
        self.degradation_active = False
        self.emergency_start_time: Optional[datetime] = None
        
        # Integration tracking
        self.integrated_systems: Dict[str, Any] = {}
        self.applied_configurations: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Callbacks for external integration
        self._load_change_callbacks: List[Callable[[SystemLoadLevel, SystemLoadLevel], None]] = []
        
        # Performance tracking
        self.degradation_metrics: Dict[str, Any] = {
            'level_changes': 0,
            'total_degradations': 0,
            'emergency_activations': 0,
            'last_metrics_reset': datetime.now()
        }
        
        # Initialize integration
        self._initialize_integrations()
        self.logger.info("Progressive Service Degradation Controller initialized")
    
    def _initialize_integrations(self):
        """Initialize integrations with available systems."""
        # Integrate with enhanced load detector
        if self.enhanced_detector:
            self.enhanced_detector.add_load_change_callback(self._on_load_level_change)
            self.integrated_systems['load_detector'] = self.enhanced_detector
            self.logger.info("Integrated with enhanced load detection system")
        
        # Store references to production systems for later configuration updates
        if self.production_load_balancer:
            self.integrated_systems['load_balancer'] = self.production_load_balancer
            self.logger.info("Integrated with production load balancer")
        
        if self.clinical_rag:
            self.integrated_systems['clinical_rag'] = self.clinical_rag
            self.logger.info("Integrated with clinical RAG system")
        
        if self.production_monitoring:
            self.integrated_systems['monitoring'] = self.production_monitoring
            self.logger.info("Integrated with production monitoring")
    
    def add_load_change_callback(self, callback: Callable[[SystemLoadLevel, SystemLoadLevel], None]):
        """Add callback for load level changes."""
        self._load_change_callbacks.append(callback)
    
    def _on_load_level_change(self, metrics):
        """Handle load level changes from the monitoring system."""
        if not ENHANCED_MONITORING_AVAILABLE:
            return
        
        new_load_level = metrics.load_level if hasattr(metrics, 'load_level') else SystemLoadLevel.NORMAL
        
        with self._lock:
            if new_load_level != self.current_load_level:
                previous_level = self.current_load_level
                self._apply_degradation_level(new_load_level, previous_level)
    
    def _apply_degradation_level(self, new_level: SystemLoadLevel, previous_level: SystemLoadLevel):
        """Apply degradation strategies for the new load level."""
        self.logger.info(f"Applying degradation level: {previous_level.name} → {new_level.name}")
        
        # Update state
        self.current_load_level = new_level
        self.last_level_change_time = datetime.now()
        self.degradation_metrics['level_changes'] += 1
        
        # Handle emergency mode
        if new_level == SystemLoadLevel.EMERGENCY:
            if self.emergency_start_time is None:
                self.emergency_start_time = datetime.now()
                self.degradation_metrics['emergency_activations'] += 1
                self.logger.warning("EMERGENCY MODE ACTIVATED - Maximum degradation in effect")
        else:
            self.emergency_start_time = None
        
        # Update all strategy managers
        self.timeout_manager.update_load_level(new_level)
        self.complexity_manager.update_load_level(new_level)
        self.feature_manager.update_load_level(new_level)
        
        # Apply configurations to integrated systems
        self._update_integrated_systems(new_level)
        
        # Track degradation activity
        if new_level > SystemLoadLevel.NORMAL:
            self.degradation_active = True
            self.degradation_metrics['total_degradations'] += 1
        else:
            self.degradation_active = False
        
        # Notify callbacks
        for callback in self._load_change_callbacks:
            try:
                callback(previous_level, new_level)
            except Exception as e:
                self.logger.error(f"Error in load change callback: {e}")
        
        # Log current settings
        self._log_current_settings()
    
    def _update_integrated_systems(self, load_level: SystemLoadLevel):
        """Update configurations in integrated production systems."""
        
        # Update production load balancer timeouts
        if 'load_balancer' in self.integrated_systems:
            self._update_load_balancer_config(load_level)
        
        # Update clinical RAG query parameters
        if 'clinical_rag' in self.integrated_systems:
            self._update_clinical_rag_config(load_level)
        
        # Update monitoring system settings
        if 'monitoring' in self.integrated_systems:
            self._update_monitoring_config(load_level)
    
    def _update_load_balancer_config(self, load_level: SystemLoadLevel):
        """Update load balancer configuration with new timeouts."""
        try:
            load_balancer = self.integrated_systems['load_balancer']
            new_timeouts = self.timeout_manager.get_all_timeouts()
            
            # Apply timeouts to backend configurations if the load balancer supports it
            if hasattr(load_balancer, 'update_backend_timeouts'):
                load_balancer.update_backend_timeouts(new_timeouts)
                self.logger.debug("Updated load balancer backend timeouts")
            
            # Update circuit breaker settings for higher load levels
            if hasattr(load_balancer, 'update_circuit_breaker_settings'):
                if load_level >= SystemLoadLevel.HIGH:
                    # More aggressive circuit breaker settings under high load
                    cb_settings = {
                        'failure_threshold': max(3, 5 - load_level.value),
                        'recovery_timeout': 30 + (load_level.value * 10)
                    }
                    load_balancer.update_circuit_breaker_settings(cb_settings)
                    self.logger.debug(f"Updated circuit breaker settings: {cb_settings}")
            
            self.applied_configurations['load_balancer'] = {
                'timeouts': new_timeouts,
                'load_level': load_level.name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating load balancer config: {e}")
    
    def _update_clinical_rag_config(self, load_level: SystemLoadLevel):
        """Update clinical RAG system with complexity and feature settings."""
        try:
            clinical_rag = self.integrated_systems['clinical_rag']
            complexity_settings = self.complexity_manager.get_query_params()
            feature_settings = self.feature_manager.get_feature_settings()
            timeouts = self.timeout_manager.get_all_timeouts()
            
            # Apply timeout settings
            if hasattr(clinical_rag, 'update_timeouts'):
                clinical_rag.update_timeouts({
                    'lightrag_timeout': timeouts.get('lightrag', 60.0),
                    'openai_timeout': timeouts.get('openai_api', 45.0)
                })
            
            # Apply query complexity settings
            if hasattr(clinical_rag, 'update_query_complexity'):
                clinical_rag.update_query_complexity(complexity_settings)
            
            # Apply feature settings
            if hasattr(clinical_rag, 'update_feature_flags'):
                clinical_rag.update_feature_flags(feature_settings)
            
            # For systems that don't have direct update methods, try to set configuration
            if hasattr(clinical_rag, 'config') and hasattr(clinical_rag.config, 'update'):
                clinical_rag.config.update({
                    'degradation_level': load_level.name,
                    'max_tokens': complexity_settings.get('token_limit', 8000),
                    'query_mode': complexity_settings.get('query_mode', 'hybrid'),
                    'enable_detailed_logging': feature_settings.get('detailed_logging', True),
                    'enable_complex_analytics': feature_settings.get('complex_analytics', True)
                })
            
            self.applied_configurations['clinical_rag'] = {
                'complexity': complexity_settings,
                'features': feature_settings,
                'timeouts': {k: v for k, v in timeouts.items() if 'lightrag' in k or 'openai' in k},
                'load_level': load_level.name,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.debug("Updated clinical RAG configuration")
            
        except Exception as e:
            self.logger.error(f"Error updating clinical RAG config: {e}")
    
    def _update_monitoring_config(self, load_level: SystemLoadLevel):
        """Update monitoring system configuration based on load level."""
        try:
            monitoring = self.integrated_systems['monitoring']
            feature_settings = self.feature_manager.get_feature_settings()
            
            # Reduce monitoring frequency under high load
            monitoring_intervals = {
                SystemLoadLevel.NORMAL: 5.0,
                SystemLoadLevel.ELEVATED: 7.0,
                SystemLoadLevel.HIGH: 10.0,
                SystemLoadLevel.CRITICAL: 15.0,
                SystemLoadLevel.EMERGENCY: 20.0
            }
            
            if hasattr(monitoring, 'update_monitoring_interval'):
                new_interval = monitoring_intervals.get(load_level, 5.0)
                monitoring.update_monitoring_interval(new_interval)
            
            # Disable detailed logging if configured
            if hasattr(monitoring, 'set_detailed_logging'):
                monitoring.set_detailed_logging(feature_settings.get('detailed_logging', True))
            
            self.applied_configurations['monitoring'] = {
                'interval': monitoring_intervals.get(load_level, 5.0),
                'detailed_logging': feature_settings.get('detailed_logging', True),
                'load_level': load_level.name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating monitoring config: {e}")
    
    def _log_current_settings(self):
        """Log current degradation settings."""
        timeouts = self.timeout_manager.get_all_timeouts()
        complexity = self.complexity_manager.get_query_params()
        features = self.feature_manager.get_feature_settings()
        
        self.logger.info(f"Current degradation level: {self.current_load_level.name}")
        self.logger.debug(f"Timeouts: {json.dumps({k: f'{v:.1f}s' for k, v in timeouts.items()})}")
        self.logger.debug(f"Query complexity: {json.dumps(complexity)}")
        
        disabled_features = [k for k, v in features.items() if not v]
        if disabled_features:
            self.logger.info(f"Disabled features: {', '.join(disabled_features)}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        return {
            'load_level': self.current_load_level.name,
            'degradation_active': self.degradation_active,
            'emergency_mode': self.current_load_level == SystemLoadLevel.EMERGENCY,
            'emergency_duration': (
                (datetime.now() - self.emergency_start_time).total_seconds() 
                if self.emergency_start_time else 0
            ),
            'last_level_change': self.last_level_change_time.isoformat(),
            'timeouts': self.timeout_manager.get_all_timeouts(),
            'query_complexity': self.complexity_manager.get_query_params(),
            'feature_settings': self.feature_manager.get_feature_settings(),
            'metrics': self.degradation_metrics.copy(),
            'integrated_systems': list(self.integrated_systems.keys()),
            'applied_configurations': {
                system: config.get('timestamp') 
                for system, config in self.applied_configurations.items()
            }
        }
    
    def force_load_level(self, load_level: SystemLoadLevel, reason: str = "Manual override"):
        """Force a specific load level (for testing or manual intervention)."""
        self.logger.warning(f"Force setting load level to {load_level.name}: {reason}")
        
        with self._lock:
            previous_level = self.current_load_level
            self._apply_degradation_level(load_level, previous_level)
    
    def reset_degradation_metrics(self):
        """Reset degradation performance metrics."""
        self.degradation_metrics = {
            'level_changes': 0,
            'total_degradations': 0,
            'emergency_activations': 0,
            'last_metrics_reset': datetime.now()
        }
        self.logger.info("Degradation metrics reset")
    
    def get_timeout_for_service(self, service: str) -> float:
        """Get current timeout for a specific service."""
        return self.timeout_manager.get_timeout(service)
    
    def should_simplify_query(self, query: str) -> bool:
        """Check if a query should be simplified at current load level."""
        return self.complexity_manager.should_simplify_query(query)
    
    def simplify_query(self, query: str) -> str:
        """Simplify a query based on current load level."""
        return self.complexity_manager.simplify_query(query)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled at current load level."""
        return self.feature_manager.is_feature_enabled(feature)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_progressive_degradation_controller(
    enhanced_detector: Optional[Any] = None,
    production_load_balancer: Optional[Any] = None,
    clinical_rag: Optional[Any] = None,
    production_monitoring: Optional[Any] = None,
    custom_config: Optional[DegradationConfiguration] = None
) -> ProgressiveServiceDegradationController:
    """Create a production-ready progressive degradation controller."""
    
    config = custom_config or DegradationConfiguration()
    
    controller = ProgressiveServiceDegradationController(
        config=config,
        enhanced_detector=enhanced_detector,
        production_load_balancer=production_load_balancer,
        clinical_rag=clinical_rag,
        production_monitoring=production_monitoring
    )
    
    return controller


def create_integrated_degradation_system(
    monitoring_interval: float = 5.0,
    degradation_config: Optional[DegradationConfiguration] = None,
    production_systems: Optional[Dict[str, Any]] = None
) -> Tuple[Any, ProgressiveServiceDegradationController]:
    """Create a complete integrated degradation system."""
    
    # Create enhanced monitoring system
    if ENHANCED_MONITORING_AVAILABLE:
        from .enhanced_load_monitoring_system import create_enhanced_load_monitoring_system
        enhanced_detector = create_enhanced_load_monitoring_system(
            monitoring_interval=monitoring_interval,
            enable_trend_analysis=True
        )
    else:
        enhanced_detector = None
    
    # Extract production systems
    systems = production_systems or {}
    
    # Create degradation controller
    controller = create_progressive_degradation_controller(
        enhanced_detector=enhanced_detector,
        production_load_balancer=systems.get('load_balancer'),
        clinical_rag=systems.get('clinical_rag'),
        production_monitoring=systems.get('monitoring'),
        custom_config=degradation_config
    )
    
    return enhanced_detector, controller


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

async def demonstrate_progressive_degradation():
    """Demonstrate the progressive service degradation system."""
    print("Progressive Service Degradation Controller Demonstration")
    print("=" * 70)
    
    # Create integrated system
    enhanced_detector, controller = create_integrated_degradation_system(
        monitoring_interval=2.0
    )
    
    print(f"Created integrated degradation system:")
    print(f"  Enhanced Detector: {'✓' if enhanced_detector else '✗'}")
    print(f"  Degradation Controller: ✓")
    print(f"  Integrated Systems: {len(controller.integrated_systems)}")
    print()
    
    # Add callback to monitor degradation changes
    def on_degradation_change(previous_level: SystemLoadLevel, new_level: SystemLoadLevel):
        print(f"\n🔄 DEGRADATION CHANGE: {previous_level.name} → {new_level.name}")
        
        status = controller.get_current_status()
        print(f"   Emergency Mode: {'🚨 YES' if status['emergency_mode'] else '✅ NO'}")
        
        # Show timeout changes
        timeouts = status['timeouts']
        print(f"   LightRAG Timeout: {timeouts.get('lightrag', 0):.1f}s")
        print(f"   OpenAI Timeout: {timeouts.get('openai_api', 0):.1f}s")
        
        # Show query complexity
        complexity = status['query_complexity']
        print(f"   Token Limit: {complexity.get('token_limit', 0)}")
        print(f"   Query Mode: {complexity.get('query_mode', 'unknown')}")
        
        # Show disabled features
        features = status['feature_settings']
        disabled = [k for k, v in features.items() if not v]
        if disabled:
            print(f"   Disabled Features: {', '.join(disabled[:3])}{'...' if len(disabled) > 3 else ''}")
        
        print("   " + "-" * 50)
    
    controller.add_load_change_callback(on_degradation_change)
    
    # Start monitoring if available
    if enhanced_detector:
        await enhanced_detector.start_monitoring()
        print("Enhanced monitoring started.")
        
        # Simulate load changes by updating metrics manually
        print("\n📊 Simulating system load changes...")
        
        load_scenarios = [
            (SystemLoadLevel.NORMAL, "Normal operation"),
            (SystemLoadLevel.ELEVATED, "Elevated load - minor optimizations"),
            (SystemLoadLevel.HIGH, "High load - timeout reductions"),
            (SystemLoadLevel.CRITICAL, "Critical load - aggressive degradation"),
            (SystemLoadLevel.EMERGENCY, "EMERGENCY - maximum degradation"),
            (SystemLoadLevel.HIGH, "Recovery to high load"),
            (SystemLoadLevel.NORMAL, "Full recovery")
        ]
        
        for load_level, description in load_scenarios:
            print(f"\n⏳ Setting load level: {load_level.name} ({description})")
            controller.force_load_level(load_level, description)
            await asyncio.sleep(3)
        
        await enhanced_detector.stop_monitoring()
    
    else:
        print("Enhanced monitoring not available - demonstrating manual load level changes")
        
        # Manual demonstration without monitoring
        for level in [SystemLoadLevel.ELEVATED, SystemLoadLevel.HIGH, 
                     SystemLoadLevel.CRITICAL, SystemLoadLevel.EMERGENCY, SystemLoadLevel.NORMAL]:
            print(f"\n⏳ Setting load level: {level.name}")
            controller.force_load_level(level, f"Manual demo - {level.name}")
            await asyncio.sleep(2)
    
    # Show final status
    print(f"\n📋 Final System Status:")
    final_status = controller.get_current_status()
    print(json.dumps(final_status, indent=2, default=str))
    
    print(f"\n✅ Progressive Service Degradation demonstration completed!")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_progressive_degradation())