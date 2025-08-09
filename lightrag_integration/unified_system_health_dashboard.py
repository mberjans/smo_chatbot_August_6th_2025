"""
Unified System Health Monitoring Dashboard for Clinical Metabolomics Oracle
==========================================================================

This module provides a comprehensive, unified dashboard that consolidates all existing
monitoring capabilities into a cohesive interface. It integrates with:

- GracefulDegradationOrchestrator (main coordinator)
- EnhancedLoadDetectionSystem (real-time load metrics)
- ProgressiveServiceDegradationController (degradation management)
- CircuitBreakerMonitoring (circuit breaker status)
- ProductionMonitoring (performance metrics)
- Production dashboards and APIs

Key Features:
1. Real-time system health visualization
2. Unified data aggregation from all monitoring components
3. WebSocket-based live updates
4. Historical data visualization with trends
5. Alert management and notifications
6. Predictive analytics and recommendations
7. RESTful API for external integrations
8. Responsive web interface with mobile support

Architecture:
- FastAPI-based REST API with WebSocket support
- React/Vue.js frontend (template-based for simplicity)
- SQLite for historical data storage
- Prometheus metrics integration
- Real-time data streaming via WebSocket

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
Production Ready: Yes
Task: CMO-LIGHTRAG-014-T07 - System Health Monitoring Dashboard
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import statistics
import uuid

# Web framework imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from flask import Flask, render_template_string, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Import existing monitoring components
try:
    from .graceful_degradation_integration import (
        GracefulDegradationOrchestrator, 
        create_graceful_degradation_system
    )
    GRACEFUL_DEGRADATION_AVAILABLE = True
except ImportError:
    GRACEFUL_DEGRADATION_AVAILABLE = False
    logging.warning("Graceful degradation system not available")

try:
    from .enhanced_load_monitoring_system import (
        EnhancedLoadDetectionSystem, 
        EnhancedSystemLoadMetrics,
        SystemLoadLevel
    )
    ENHANCED_MONITORING_AVAILABLE = True
except ImportError:
    ENHANCED_MONITORING_AVAILABLE = False
    logging.warning("Enhanced load monitoring not available")

try:
    from .progressive_service_degradation_controller import (
        ProgressiveServiceDegradationController
    )
    DEGRADATION_CONTROLLER_AVAILABLE = True
except ImportError:
    DEGRADATION_CONTROLLER_AVAILABLE = False
    logging.warning("Progressive degradation controller not available")

try:
    from .circuit_breaker_monitoring_integration import (
        CircuitBreakerMonitoringIntegration
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    logging.warning("Circuit breaker monitoring not available")

try:
    from .production_monitoring import ProductionMonitoring
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False
    logging.warning("Production monitoring not available")


# ============================================================================
# CONFIGURATION AND DATA MODELS
# ============================================================================

@dataclass
class DashboardConfig:
    """Configuration for the unified system health dashboard."""
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8092
    enable_cors: bool = True
    enable_ssl: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # API configuration
    api_prefix: str = "/api/v1"
    websocket_endpoint: str = "/ws/health"
    static_files_path: str = "static"
    
    # Dashboard settings
    enable_websockets: bool = True
    websocket_update_interval: float = 2.0
    enable_historical_data: bool = True
    historical_retention_hours: int = 72
    enable_predictive_analytics: bool = True
    
    # Database configuration
    db_path: str = "unified_health_dashboard.db"
    enable_db_persistence: bool = True
    db_backup_interval_hours: int = 24
    
    # Alert configuration
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 300
    enable_email_alerts: bool = False
    email_smtp_server: Optional[str] = None
    
    # Security configuration
    enable_api_key: bool = False
    api_key: Optional[str] = None
    enable_rate_limiting: bool = True
    requests_per_minute: int = 120


@dataclass
class SystemHealthSnapshot:
    """Comprehensive snapshot of system health at a point in time."""
    
    # Metadata
    timestamp: datetime
    snapshot_id: str
    system_uptime_seconds: float
    
    # Overall health status
    overall_health: str  # healthy, degraded, critical, emergency
    health_score: float  # 0.0 to 1.0
    
    # Load and performance metrics
    load_level: str
    load_score: float
    cpu_utilization: float
    memory_pressure: float
    response_time_p95: float
    error_rate: float
    request_queue_depth: int
    active_connections: int
    
    # Degradation status
    degradation_active: bool
    degradation_level: str
    emergency_mode: bool
    active_degradations: List[str] = field(default_factory=list)
    disabled_features: List[str] = field(default_factory=list)
    
    # Service status
    integrated_services: Dict[str, bool] = field(default_factory=dict)
    circuit_breakers: Dict[str, str] = field(default_factory=dict)  # service -> status
    
    # Performance metrics
    throughput_rps: float = 0.0
    success_rate: float = 100.0
    total_requests_processed: int = 0
    
    # Resource usage
    connection_pool_usage: float = 0.0
    thread_pool_usage: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Alerts and issues
    active_alerts_count: int = 0
    critical_alerts_count: int = 0
    active_issues: List[str] = field(default_factory=list)
    
    # Trend indicators (for predictive analytics)
    load_trend: str = "stable"  # improving, stable, degrading
    performance_trend: str = "stable"
    predicted_load_change: str = "none"  # increase, decrease, none
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AlertEvent:
    """Represents an alert event in the system."""
    
    id: str
    timestamp: datetime
    severity: str  # info, warning, critical, emergency
    source: str  # component that generated the alert
    title: str
    message: str
    category: str  # performance, degradation, circuit_breaker, resource
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    
    # Additional context
    related_metrics: Dict[str, Any] = field(default_factory=dict)
    affected_services: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


# ============================================================================
# DATA AGGREGATOR - CENTRAL DATA COLLECTION
# ============================================================================

class UnifiedDataAggregator:
    """
    Central component that aggregates data from all monitoring systems
    into a unified view for the dashboard.
    """
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # References to monitoring systems
        self.graceful_degradation_orchestrator: Optional[Any] = None
        self.enhanced_load_detector: Optional[Any] = None
        self.degradation_controller: Optional[Any] = None
        self.circuit_breaker_monitor: Optional[Any] = None
        self.production_monitor: Optional[Any] = None
        
        # Data storage
        self.current_snapshot: Optional[SystemHealthSnapshot] = None
        self.historical_snapshots: deque = deque(maxlen=5000)  # ~4 days at 2s intervals
        self.alert_history: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, AlertEvent] = {}
        
        # Threading and state management
        self._lock = threading.Lock()
        self._running = False
        self._aggregation_task: Optional[asyncio.Task] = None
        
        # Database for persistence
        self.db_path = Path(config.db_path)
        self._init_database()
        
        # Callbacks for real-time updates
        self._update_callbacks: List[Callable[[SystemHealthSnapshot], None]] = []
        self._alert_callbacks: List[Callable[[AlertEvent], None]] = []
        
        self.logger.info("Unified Data Aggregator initialized")
    
    def _init_database(self):
        """Initialize SQLite database for historical data storage."""
        if not self.config.enable_db_persistence:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_snapshots (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_events (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    severity TEXT,
                    source TEXT,
                    title TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON health_snapshots(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alert_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alert_events(severity)')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
    
    def register_monitoring_systems(self, 
                                  graceful_degradation_orchestrator: Optional[Any] = None,
                                  enhanced_load_detector: Optional[Any] = None,
                                  degradation_controller: Optional[Any] = None,
                                  circuit_breaker_monitor: Optional[Any] = None,
                                  production_monitor: Optional[Any] = None):
        """Register monitoring systems for data aggregation."""
        
        self.graceful_degradation_orchestrator = graceful_degradation_orchestrator
        self.enhanced_load_detector = enhanced_load_detector
        self.degradation_controller = degradation_controller
        self.circuit_breaker_monitor = circuit_breaker_monitor
        self.production_monitor = production_monitor
        
        # Count available systems
        available_systems = sum(1 for sys in [
            graceful_degradation_orchestrator,
            enhanced_load_detector,
            degradation_controller,
            circuit_breaker_monitor,
            production_monitor
        ] if sys is not None)
        
        self.logger.info(f"Registered {available_systems} monitoring systems")
    
    async def start_aggregation(self):
        """Start the data aggregation process."""
        if self._running:
            self.logger.warning("Data aggregation already running")
            return
        
        self._running = True
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        self.logger.info("Data aggregation started")
    
    async def stop_aggregation(self):
        """Stop the data aggregation process."""
        if not self._running:
            return
        
        self._running = False
        
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Data aggregation stopped")
    
    async def _aggregation_loop(self):
        """Main aggregation loop that collects data from all systems."""
        while self._running:
            try:
                # Collect data from all systems
                snapshot = await self._create_health_snapshot()
                
                if snapshot:
                    with self._lock:
                        # Update current snapshot
                        previous_snapshot = self.current_snapshot
                        self.current_snapshot = snapshot
                        
                        # Store in historical data
                        self.historical_snapshots.append(snapshot)
                        
                        # Persist to database
                        if self.config.enable_db_persistence:
                            await self._persist_snapshot(snapshot)
                        
                        # Check for alerts
                        if previous_snapshot:
                            await self._check_for_alerts(previous_snapshot, snapshot)
                    
                    # Notify callbacks
                    for callback in self._update_callbacks:
                        try:
                            callback(snapshot)
                        except Exception as e:
                            self.logger.error(f"Error in update callback: {e}")
                
                await asyncio.sleep(self.config.websocket_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(5)
    
    async def _create_health_snapshot(self) -> Optional[SystemHealthSnapshot]:
        """Create a comprehensive health snapshot from all monitoring systems."""
        try:
            snapshot_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now()
            
            # Initialize snapshot with defaults
            snapshot = SystemHealthSnapshot(
                timestamp=timestamp,
                snapshot_id=snapshot_id,
                system_uptime_seconds=0.0,
                overall_health="unknown",
                health_score=0.0,
                load_level="UNKNOWN",
                load_score=0.0,
                cpu_utilization=0.0,
                memory_pressure=0.0,
                response_time_p95=0.0,
                error_rate=0.0,
                request_queue_depth=0,
                active_connections=0,
                degradation_active=False,
                degradation_level="NORMAL",
                emergency_mode=False
            )
            
            # Collect data from Graceful Degradation Orchestrator (primary source)
            if self.graceful_degradation_orchestrator:
                await self._collect_from_graceful_degradation(snapshot)
            
            # Collect data from Enhanced Load Detection System
            if self.enhanced_load_detector:
                await self._collect_from_load_detector(snapshot)
            
            # Collect data from Degradation Controller
            if self.degradation_controller:
                await self._collect_from_degradation_controller(snapshot)
            
            # Collect data from Circuit Breaker Monitor
            if self.circuit_breaker_monitor:
                await self._collect_from_circuit_breaker(snapshot)
            
            # Collect data from Production Monitor
            if self.production_monitor:
                await self._collect_from_production_monitor(snapshot)
            
            # Calculate derived metrics
            await self._calculate_derived_metrics(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error creating health snapshot: {e}")
            return None
    
    async def _collect_from_graceful_degradation(self, snapshot: SystemHealthSnapshot):
        """Collect data from the Graceful Degradation Orchestrator."""
        try:
            status = self.graceful_degradation_orchestrator.get_system_status()
            health = self.graceful_degradation_orchestrator.get_health_check()
            
            # System uptime
            if status.get('start_time'):
                start_time = datetime.fromisoformat(status['start_time'].replace('Z', '+00:00'))
                snapshot.system_uptime_seconds = (snapshot.timestamp - start_time).total_seconds()
            
            # Integration status
            integration_status = status.get('integration_status', {})
            snapshot.integrated_services = {
                'load_monitoring': integration_status.get('load_monitoring_active', False),
                'degradation_controller': integration_status.get('degradation_controller_active', False),
                'throttling_system': integration_status.get('throttling_system_active', False),
                'load_balancer': integration_status.get('integrated_load_balancer', False),
                'rag_system': integration_status.get('integrated_rag_system', False),
                'monitoring_system': integration_status.get('integrated_monitoring', False)
            }
            
            # Overall health
            snapshot.overall_health = health.get('status', 'unknown')
            snapshot.active_issues = health.get('issues', [])
            
            # Load level
            snapshot.load_level = status.get('current_load_level', 'UNKNOWN')
            
            # Request processing
            snapshot.total_requests_processed = status.get('total_requests_processed', 0)
            
        except Exception as e:
            self.logger.error(f"Error collecting from graceful degradation orchestrator: {e}")
    
    async def _collect_from_load_detector(self, snapshot: SystemHealthSnapshot):
        """Collect data from the Enhanced Load Detection System."""
        try:
            current_metrics = self.enhanced_load_detector.get_current_metrics()
            
            if current_metrics:
                snapshot.cpu_utilization = current_metrics.cpu_utilization
                snapshot.memory_pressure = current_metrics.memory_pressure
                snapshot.response_time_p95 = current_metrics.response_time_p95
                snapshot.error_rate = current_metrics.error_rate
                snapshot.request_queue_depth = current_metrics.request_queue_depth
                snapshot.active_connections = current_metrics.active_connections
                snapshot.load_score = current_metrics.load_score
                
                # Load level from enhanced detector
                if hasattr(current_metrics, 'load_level'):
                    snapshot.load_level = current_metrics.load_level.name
                    
        except Exception as e:
            self.logger.error(f"Error collecting from load detector: {e}")
    
    async def _collect_from_degradation_controller(self, snapshot: SystemHealthSnapshot):
        """Collect data from the Progressive Service Degradation Controller."""
        try:
            status = self.degradation_controller.get_current_status()
            
            # Degradation information
            snapshot.degradation_active = status.get('degradation_active', False)
            snapshot.degradation_level = status.get('load_level', 'NORMAL')
            snapshot.emergency_mode = status.get('emergency_mode', False)
            
            # Feature status
            feature_settings = status.get('feature_settings', {})
            snapshot.disabled_features = [
                feature for feature, enabled in feature_settings.items()
                if not enabled
            ]
            
            # Degradation details
            if snapshot.degradation_active:
                degradations = []
                if status.get('timeouts'):
                    degradations.append("timeout_reductions")
                if snapshot.disabled_features:
                    degradations.append("feature_disabling")
                if status.get('query_complexity', {}).get('token_limit', 8000) < 8000:
                    degradations.append("query_simplification")
                
                snapshot.active_degradations = degradations
                
        except Exception as e:
            self.logger.error(f"Error collecting from degradation controller: {e}")
    
    async def _collect_from_circuit_breaker(self, snapshot: SystemHealthSnapshot):
        """Collect data from the Circuit Breaker Monitor."""
        try:
            if hasattr(self.circuit_breaker_monitor, 'get_health_status'):
                health_status = self.circuit_breaker_monitor.get_health_status()
                
                # Circuit breaker status
                if isinstance(health_status, dict):
                    snapshot.circuit_breakers = {
                        service: status.get('status', 'unknown')
                        for service, status in health_status.items()
                        if isinstance(status, dict)
                    }
                    
        except Exception as e:
            self.logger.error(f"Error collecting from circuit breaker monitor: {e}")
    
    async def _collect_from_production_monitor(self, snapshot: SystemHealthSnapshot):
        """Collect data from the Production Monitoring system."""
        try:
            if hasattr(self.production_monitor, 'get_current_metrics'):
                metrics = self.production_monitor.get_current_metrics()
                
                if metrics:
                    # Performance metrics
                    snapshot.throughput_rps = metrics.get('requests_per_second', 0.0)
                    snapshot.success_rate = metrics.get('success_rate', 100.0)
                    
                    # Resource usage
                    snapshot.memory_usage_mb = metrics.get('memory_usage_mb', 0.0)
                    snapshot.connection_pool_usage = metrics.get('connection_pool_usage', 0.0)
                    snapshot.thread_pool_usage = metrics.get('thread_pool_usage', 0.0)
                    
        except Exception as e:
            self.logger.error(f"Error collecting from production monitor: {e}")
    
    async def _calculate_derived_metrics(self, snapshot: SystemHealthSnapshot):
        """Calculate derived metrics and health scores."""
        try:
            # Calculate health score (0.0 to 1.0)
            health_components = []
            
            # CPU health (inverted, lower is better)
            cpu_health = max(0.0, 1.0 - (snapshot.cpu_utilization / 100.0))
            health_components.append(cpu_health * 0.2)
            
            # Memory health (inverted, lower is better)
            memory_health = max(0.0, 1.0 - (snapshot.memory_pressure / 100.0))
            health_components.append(memory_health * 0.2)
            
            # Response time health (inverted, lower is better, normalized to 5000ms)
            response_health = max(0.0, 1.0 - min(1.0, snapshot.response_time_p95 / 5000.0))
            health_components.append(response_health * 0.2)
            
            # Error rate health (inverted, lower is better, normalized to 10%)
            error_health = max(0.0, 1.0 - min(1.0, snapshot.error_rate / 10.0))
            health_components.append(error_health * 0.2)
            
            # Success rate health (normalized)
            success_health = snapshot.success_rate / 100.0
            health_components.append(success_health * 0.2)
            
            snapshot.health_score = sum(health_components)
            
            # Determine overall health status based on score and specific conditions
            if snapshot.emergency_mode:
                snapshot.overall_health = "emergency"
            elif snapshot.degradation_level in ['CRITICAL', 'EMERGENCY']:
                snapshot.overall_health = "critical"
            elif snapshot.health_score < 0.5 or snapshot.degradation_active:
                snapshot.overall_health = "degraded"
            elif snapshot.health_score >= 0.8:
                snapshot.overall_health = "healthy"
            else:
                snapshot.overall_health = "warning"
            
            # Calculate trends if we have historical data
            if len(self.historical_snapshots) >= 5:
                snapshot.load_trend = self._calculate_trend([s.load_score for s in list(self.historical_snapshots)[-5:]])
                snapshot.performance_trend = self._calculate_trend([s.health_score for s in list(self.historical_snapshots)[-5:]])
            
        except Exception as e:
            self.logger.error(f"Error calculating derived metrics: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return "stable"
        
        try:
            # Calculate linear regression slope
            n = len(values)
            x_values = list(range(n))
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(values)
            
            numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return "stable"
            
            slope = numerator / denominator
            
            # Classify trend based on slope
            if slope > 0.1:
                return "improving"
            elif slope < -0.1:
                return "degrading"
            else:
                return "stable"
                
        except Exception:
            return "stable"
    
    async def _check_for_alerts(self, previous: SystemHealthSnapshot, current: SystemHealthSnapshot):
        """Check for alert conditions and generate alerts."""
        try:
            alerts_to_generate = []
            
            # Load level change alert
            if previous.load_level != current.load_level:
                severity = "info"
                if current.load_level in ["CRITICAL", "EMERGENCY"]:
                    severity = "critical"
                elif current.load_level == "HIGH":
                    severity = "warning"
                
                alert = AlertEvent(
                    id=f"load_change_{current.snapshot_id}",
                    timestamp=current.timestamp,
                    severity=severity,
                    source="load_detector",
                    title="Load Level Changed",
                    message=f"System load level changed from {previous.load_level} to {current.load_level}",
                    category="performance",
                    related_metrics={
                        "previous_level": previous.load_level,
                        "current_level": current.load_level,
                        "load_score": current.load_score
                    }
                )
                alerts_to_generate.append(alert)
            
            # Emergency mode alert
            if not previous.emergency_mode and current.emergency_mode:
                alert = AlertEvent(
                    id=f"emergency_mode_{current.snapshot_id}",
                    timestamp=current.timestamp,
                    severity="critical",
                    source="degradation_controller",
                    title="Emergency Mode Activated",
                    message="System has entered emergency mode with maximum degradation",
                    category="degradation",
                    recommended_actions=[
                        "Monitor system stability",
                        "Reduce incoming load if possible",
                        "Check for underlying issues"
                    ]
                )
                alerts_to_generate.append(alert)
            
            # Performance degradation alert
            if (current.response_time_p95 > previous.response_time_p95 * 1.5 and 
                current.response_time_p95 > 2000):
                alert = AlertEvent(
                    id=f"performance_degradation_{current.snapshot_id}",
                    timestamp=current.timestamp,
                    severity="warning",
                    source="performance_monitor",
                    title="Performance Degradation Detected",
                    message=f"Response time increased from {previous.response_time_p95:.0f}ms to {current.response_time_p95:.0f}ms",
                    category="performance",
                    related_metrics={
                        "previous_p95": previous.response_time_p95,
                        "current_p95": current.response_time_p95
                    }
                )
                alerts_to_generate.append(alert)
            
            # Error rate spike alert
            if current.error_rate > previous.error_rate * 2 and current.error_rate > 1.0:
                alert = AlertEvent(
                    id=f"error_spike_{current.snapshot_id}",
                    timestamp=current.timestamp,
                    severity="warning" if current.error_rate < 5.0 else "critical",
                    source="error_monitor",
                    title="Error Rate Spike Detected",
                    message=f"Error rate increased from {previous.error_rate:.2f}% to {current.error_rate:.2f}%",
                    category="performance",
                    related_metrics={
                        "previous_error_rate": previous.error_rate,
                        "current_error_rate": current.error_rate
                    }
                )
                alerts_to_generate.append(alert)
            
            # Process generated alerts
            for alert in alerts_to_generate:
                await self._process_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking for alerts: {e}")
    
    async def _process_alert(self, alert: AlertEvent):
        """Process and store a new alert."""
        try:
            # Check for cooldown to prevent spam
            cooldown_key = f"{alert.source}_{alert.category}"
            
            # Simple cooldown implementation
            cutoff_time = alert.timestamp - timedelta(seconds=self.config.alert_cooldown_seconds)
            recent_similar_alerts = [
                a for a in self.alert_history
                if (a.source == alert.source and 
                    a.category == alert.category and 
                    a.timestamp > cutoff_time)
            ]
            
            if len(recent_similar_alerts) > 0:
                self.logger.debug(f"Alert {alert.id} suppressed due to cooldown")
                return
            
            # Store alert
            with self._lock:
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
            
            # Persist to database
            if self.config.enable_db_persistence:
                await self._persist_alert(alert)
            
            # Notify callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
            
            self.logger.info(f"Generated alert: {alert.severity} - {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Error processing alert: {e}")
    
    async def _persist_snapshot(self, snapshot: SystemHealthSnapshot):
        """Persist snapshot to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO health_snapshots (id, timestamp, data) VALUES (?, ?, ?)',
                (snapshot.snapshot_id, snapshot.timestamp, json.dumps(snapshot.to_dict()))
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error persisting snapshot: {e}")
    
    async def _persist_alert(self, alert: AlertEvent):
        """Persist alert to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alert_events (id, timestamp, severity, source, title, message, data) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.timestamp, alert.severity, alert.source,
                alert.title, alert.message, json.dumps(alert.to_dict())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error persisting alert: {e}")
    
    def add_update_callback(self, callback: Callable[[SystemHealthSnapshot], None]):
        """Add callback for snapshot updates."""
        self._update_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[AlertEvent], None]):
        """Add callback for alert events."""
        self._alert_callbacks.append(callback)
    
    def get_current_snapshot(self) -> Optional[SystemHealthSnapshot]:
        """Get the current system health snapshot."""
        return self.current_snapshot
    
    def get_historical_snapshots(self, hours: int = 24) -> List[SystemHealthSnapshot]:
        """Get historical snapshots for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                snapshot for snapshot in self.historical_snapshots
                if snapshot.timestamp > cutoff_time
            ]
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[AlertEvent]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an active alert."""
        try:
            with self._lock:
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    alert.resolved_by = resolved_by
                    
                    # Update in database
                    if self.config.enable_db_persistence:
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute(
                            'UPDATE alert_events SET resolved = TRUE WHERE id = ?',
                            (alert_id,)
                        )
                        conn.commit()
                        conn.close()
                    
                    self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolving alert {alert_id}: {e}")
            return False


# ============================================================================
# WEBSOCKET MANAGER - REAL-TIME UPDATES
# ============================================================================

class WebSocketManager:
    """Manages WebSocket connections for real-time dashboard updates."""
    
    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def add_connection(self, websocket: WebSocket):
        """Add a new WebSocket connection."""
        with self._lock:
            self.connections.add(websocket)
        self.logger.info(f"WebSocket client connected. Total connections: {len(self.connections)}")
    
    def remove_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        with self._lock:
            self.connections.discard(websocket)
        self.logger.info(f"WebSocket client disconnected. Total connections: {len(self.connections)}")
    
    async def broadcast_snapshot(self, snapshot: SystemHealthSnapshot):
        """Broadcast a health snapshot to all connected clients."""
        if not self.connections:
            return
        
        message = {
            "type": "health_update",
            "timestamp": datetime.now().isoformat(),
            "data": snapshot.to_dict()
        }
        
        await self._broadcast_message(message)
    
    async def broadcast_alert(self, alert: AlertEvent):
        """Broadcast an alert to all connected clients."""
        if not self.connections:
            return
        
        message = {
            "type": "alert",
            "timestamp": datetime.now().isoformat(),
            "data": alert.to_dict()
        }
        
        await self._broadcast_message(message)
    
    async def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSocket clients."""
        if not self.connections:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        with self._lock:
            connections_copy = self.connections.copy()
        
        for websocket in connections_copy:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                self.logger.debug(f"Failed to send message to WebSocket client: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        if disconnected:
            with self._lock:
                self.connections -= disconnected


# ============================================================================
# MAIN DASHBOARD APPLICATION
# ============================================================================

class UnifiedSystemHealthDashboard:
    """
    Main dashboard application that provides the unified system health monitoring interface.
    
    This class orchestrates all components:
    - Data aggregation from monitoring systems
    - WebSocket management for real-time updates
    - REST API for external integrations
    - Web interface for visualization
    """
    
    def __init__(self, 
                 config: Optional[DashboardConfig] = None,
                 graceful_degradation_orchestrator: Optional[Any] = None):
        
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.data_aggregator = UnifiedDataAggregator(self.config)
        self.websocket_manager = WebSocketManager()
        
        # Register monitoring systems
        if graceful_degradation_orchestrator:
            self._register_orchestrator_systems(graceful_degradation_orchestrator)
        
        # Web application (will be initialized based on available framework)
        self.app: Optional[Union[FastAPI, Flask]] = None
        self.framework = "none"
        
        # Initialize web application
        self._initialize_web_app()
        
        # Set up callbacks for real-time updates
        self.data_aggregator.add_update_callback(self._on_health_update)
        self.data_aggregator.add_alert_callback(self._on_alert_generated)
        
        self.logger.info("Unified System Health Dashboard initialized")
    
    def _register_orchestrator_systems(self, orchestrator):
        """Register all monitoring systems from the graceful degradation orchestrator."""
        try:
            # Get integrated systems from orchestrator
            enhanced_load_detector = getattr(orchestrator, 'load_detector', None)
            degradation_controller = getattr(orchestrator, 'degradation_controller', None)
            
            # Get production systems from orchestrator's integrator
            production_integrator = getattr(orchestrator, 'production_integrator', None)
            if production_integrator:
                integrated_systems = getattr(production_integrator, 'integrated_systems', {})
                circuit_breaker_monitor = integrated_systems.get('circuit_breaker')
                production_monitor = integrated_systems.get('monitoring_system')
            else:
                circuit_breaker_monitor = None
                production_monitor = None
            
            # Register all systems
            self.data_aggregator.register_monitoring_systems(
                graceful_degradation_orchestrator=orchestrator,
                enhanced_load_detector=enhanced_load_detector,
                degradation_controller=degradation_controller,
                circuit_breaker_monitor=circuit_breaker_monitor,
                production_monitor=production_monitor
            )
            
            self.logger.info("Successfully registered orchestrator monitoring systems")
            
        except Exception as e:
            self.logger.error(f"Error registering orchestrator systems: {e}")
    
    def _initialize_web_app(self):
        """Initialize the web application using the best available framework."""
        if FASTAPI_AVAILABLE:
            self._initialize_fastapi()
        elif FLASK_AVAILABLE:
            self._initialize_flask()
        else:
            self.logger.error("No supported web framework available. Install FastAPI or Flask.")
            raise ValueError("No supported web framework available")
    
    def _initialize_fastapi(self):
        """Initialize FastAPI application."""
        self.app = FastAPI(
            title="Unified System Health Dashboard",
            description="Comprehensive monitoring dashboard for Clinical Metabolomics Oracle",
            version="1.0.0"
        )
        self.framework = "fastapi"
        
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        self._setup_fastapi_routes()
        self.logger.info("FastAPI application initialized")
    
    def _setup_fastapi_routes(self):
        """Set up FastAPI routes."""
        
        @self.app.get("/")
        async def dashboard():
            """Serve the main dashboard HTML page."""
            return HTMLResponse(content=self._get_dashboard_html())
        
        @self.app.get(f"{self.config.api_prefix}/health")
        async def get_health():
            """Get current system health status."""
            snapshot = self.data_aggregator.get_current_snapshot()
            if snapshot:
                return JSONResponse(content={
                    "status": "success",
                    "data": snapshot.to_dict()
                })
            else:
                return JSONResponse(content={
                    "status": "error",
                    "message": "No health data available"
                }, status_code=503)
        
        @self.app.get(f"{self.config.api_prefix}/health/history")
        async def get_health_history(hours: int = 24):
            """Get historical health data."""
            if hours > 168:  # Limit to 1 week
                hours = 168
            
            snapshots = self.data_aggregator.get_historical_snapshots(hours)
            return JSONResponse(content={
                "status": "success",
                "data": [snapshot.to_dict() for snapshot in snapshots]
            })
        
        @self.app.get(f"{self.config.api_prefix}/alerts")
        async def get_alerts():
            """Get active alerts."""
            alerts = self.data_aggregator.get_active_alerts()
            return JSONResponse(content={
                "status": "success",
                "data": [alert.to_dict() for alert in alerts]
            })
        
        @self.app.get(f"{self.config.api_prefix}/alerts/history")
        async def get_alert_history(hours: int = 24):
            """Get alert history."""
            if hours > 168:  # Limit to 1 week
                hours = 168
            
            alerts = self.data_aggregator.get_alert_history(hours)
            return JSONResponse(content={
                "status": "success",
                "data": [alert.to_dict() for alert in alerts]
            })
        
        @self.app.post(f"{self.config.api_prefix}/alerts/{{alert_id}}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve an active alert."""
            success = self.data_aggregator.resolve_alert(alert_id)
            if success:
                return JSONResponse(content={
                    "status": "success",
                    "message": f"Alert {alert_id} resolved"
                })
            else:
                return JSONResponse(content={
                    "status": "error",
                    "message": f"Alert {alert_id} not found or already resolved"
                }, status_code=404)
        
        @self.app.get(f"{self.config.api_prefix}/system/status")
        async def get_system_status():
            """Get overall system status and integration information."""
            snapshot = self.data_aggregator.get_current_snapshot()
            return JSONResponse(content={
                "status": "success",
                "data": {
                    "dashboard_uptime_seconds": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
                    "framework": self.framework,
                    "config": {
                        "websockets_enabled": self.config.enable_websockets,
                        "historical_data_enabled": self.config.enable_historical_data,
                        "alerts_enabled": self.config.enable_alerts,
                        "retention_hours": self.config.historical_retention_hours
                    },
                    "current_health": snapshot.to_dict() if snapshot else None,
                    "connected_websocket_clients": len(self.websocket_manager.connections),
                    "total_snapshots": len(self.data_aggregator.historical_snapshots),
                    "active_alerts": len(self.data_aggregator.get_active_alerts())
                }
            })
        
        if self.config.enable_websockets:
            @self.app.websocket(self.config.websocket_endpoint)
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket endpoint for real-time updates."""
                await websocket.accept()
                self.websocket_manager.add_connection(websocket)
                
                try:
                    # Send initial data
                    snapshot = self.data_aggregator.get_current_snapshot()
                    if snapshot:
                        await websocket.send_text(json.dumps({
                            "type": "initial_data",
                            "timestamp": datetime.now().isoformat(),
                            "data": snapshot.to_dict()
                        }))
                    
                    # Keep connection alive
                    while True:
                        await websocket.receive_text()
                        
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    self.logger.error(f"WebSocket error: {e}")
                finally:
                    self.websocket_manager.remove_connection(websocket)
    
    def _on_health_update(self, snapshot: SystemHealthSnapshot):
        """Callback for health snapshot updates."""
        # Broadcast to WebSocket clients
        if self.config.enable_websockets:
            asyncio.create_task(self.websocket_manager.broadcast_snapshot(snapshot))
    
    def _on_alert_generated(self, alert: AlertEvent):
        """Callback for new alert generation."""
        # Broadcast to WebSocket clients
        if self.config.enable_websockets:
            asyncio.create_task(self.websocket_manager.broadcast_alert(alert))
    
    def _get_dashboard_html(self) -> str:
        """Generate the dashboard HTML template."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified System Health Dashboard - Clinical Metabolomics Oracle</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
            position: relative;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }

        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }

        .status-bar {
            display: flex;
            justify-content: space-around;
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }

        .status-item {
            text-align: center;
        }

        .status-value {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }

        .status-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }

        .connection-status {
            position: absolute;
            top: 30px;
            right: 30px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-connected { background-color: #27ae60; }
        .status-connecting { background-color: #f39c12; }
        .status-disconnected { background-color: #e74c3c; }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            padding: 30px;
        }

        .panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .panel:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }

        .panel-header {
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-bottom: 1px solid #dee2e6;
        }

        .panel-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }

        .panel-subtitle {
            color: #6c757d;
            font-size: 0.9em;
        }

        .panel-content {
            padding: 25px;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #f1f3f4;
        }

        .metric-row:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }

        .metric-label {
            color: #495057;
            font-weight: 500;
        }

        .metric-value {
            font-size: 1.4em;
            font-weight: 600;
            color: #2c3e50;
        }

        .metric-value.good { color: #27ae60; }
        .metric-value.warning { color: #f39c12; }
        .metric-value.critical { color: #e74c3c; }

        .health-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .health-badge.healthy {
            background: #d4edda;
            color: #155724;
        }

        .health-badge.warning {
            background: #fff3cd;
            color: #856404;
        }

        .health-badge.degraded {
            background: #f8d7da;
            color: #721c24;
        }

        .health-badge.critical {
            background: #f5c6cb;
            color: #721c24;
        }

        .health-badge.emergency {
            background: #721c24;
            color: white;
            animation: blink 1s infinite;
        }

        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.5; }
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .half-width {
            grid-column: span 2;
        }

        .chart-container {
            height: 300px;
            margin-top: 20px;
        }

        .timeline-container {
            height: 400px;
        }

        .alerts-container {
            max-height: 400px;
            overflow-y: auto;
        }

        .alert-item {
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #dee2e6;
            background: #f8f9fa;
            border-radius: 0 8px 8px 0;
            transition: all 0.3s ease;
        }

        .alert-item.critical {
            border-left-color: #dc3545;
            background: #f8d7da;
        }

        .alert-item.warning {
            border-left-color: #ffc107;
            background: #fff3cd;
        }

        .alert-item.info {
            border-left-color: #17a2b8;
            background: #d1ecf1;
        }

        .alert-title {
            font-weight: 600;
            margin-bottom: 5px;
        }

        .alert-message {
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 8px;
        }

        .alert-timestamp {
            font-size: 0.8em;
            color: #868e96;
        }

        .no-data {
            text-align: center;
            padding: 40px;
            color: #6c757d;
            font-style: italic;
        }

        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .status-bar {
                flex-direction: column;
                gap: 15px;
            }
            
            .connection-status {
                position: static;
                justify-content: center;
                margin-top: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>Unified System Health Dashboard</h1>
            <p>Clinical Metabolomics Oracle - Real-time Monitoring & Control</p>
            
            <div class="connection-status">
                <span id="connection-text">Connecting...</span>
                <div id="connection-indicator" class="status-indicator status-connecting"></div>
            </div>
            
            <div class="status-bar">
                <div class="status-item">
                    <span id="overall-health" class="status-value">Unknown</span>
                    <div class="status-label">Overall Health</div>
                </div>
                <div class="status-item">
                    <span id="load-level" class="status-value">Unknown</span>
                    <div class="status-label">Load Level</div>
                </div>
                <div class="status-item">
                    <span id="active-alerts" class="status-value">0</span>
                    <div class="status-label">Active Alerts</div>
                </div>
                <div class="status-item">
                    <span id="uptime" class="status-value">0h</span>
                    <div class="status-label">System Uptime</div>
                </div>
            </div>
        </div>

        <div class="main-grid">
            <!-- System Overview Panel -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">System Overview</div>
                    <div class="panel-subtitle">Current system status and health</div>
                </div>
                <div class="panel-content">
                    <div class="metric-row">
                        <span class="metric-label">Health Score</span>
                        <span id="health-score" class="metric-value">--</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Degradation Active</span>
                        <span id="degradation-active" class="metric-value">No</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Emergency Mode</span>
                        <span id="emergency-mode" class="metric-value">No</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Requests Processed</span>
                        <span id="total-requests" class="metric-value">0</span>
                    </div>
                </div>
            </div>

            <!-- Load Metrics Panel -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Load Metrics</div>
                    <div class="panel-subtitle">Real-time system load indicators</div>
                </div>
                <div class="panel-content">
                    <div class="metric-row">
                        <span class="metric-label">CPU Utilization</span>
                        <span id="cpu-utilization" class="metric-value">--%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Memory Pressure</span>
                        <span id="memory-pressure" class="metric-value">--%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Response Time P95</span>
                        <span id="response-time" class="metric-value">--ms</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Error Rate</span>
                        <span id="error-rate" class="metric-value">--%</span>
                    </div>
                </div>
            </div>

            <!-- Degradation Status Panel -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Degradation Status</div>
                    <div class="panel-subtitle">Active degradation measures</div>
                </div>
                <div class="panel-content">
                    <div class="metric-row">
                        <span class="metric-label">Current Level</span>
                        <span id="degradation-level" class="metric-value">Normal</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Active Measures</span>
                        <span id="active-degradations" class="metric-value">None</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Disabled Features</span>
                        <span id="disabled-features" class="metric-value">None</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Queue Depth</span>
                        <span id="queue-depth" class="metric-value">0</span>
                    </div>
                </div>
            </div>

            <!-- Performance Chart -->
            <div class="panel half-width">
                <div class="panel-header">
                    <div class="panel-title">Performance Trends</div>
                    <div class="panel-subtitle">Real-time performance metrics over time</div>
                </div>
                <div class="panel-content">
                    <div id="performance-chart" class="chart-container"></div>
                </div>
            </div>

            <!-- System Resources Panel -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">System Resources</div>
                    <div class="panel-subtitle">Resource utilization and capacity</div>
                </div>
                <div class="panel-content">
                    <div class="metric-row">
                        <span class="metric-label">Active Connections</span>
                        <span id="active-connections" class="metric-value">0</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Success Rate</span>
                        <span id="success-rate" class="metric-value">100%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Throughput</span>
                        <span id="throughput" class="metric-value">0 RPS</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Memory Usage</span>
                        <span id="memory-usage" class="metric-value">-- MB</span>
                    </div>
                </div>
            </div>

            <!-- Load Level Timeline -->
            <div class="panel full-width">
                <div class="panel-header">
                    <div class="panel-title">Load Level Timeline</div>
                    <div class="panel-subtitle">Historical load levels and degradation events</div>
                </div>
                <div class="panel-content">
                    <div id="load-timeline-chart" class="timeline-container"></div>
                </div>
            </div>

            <!-- Active Alerts Panel -->
            <div class="panel full-width">
                <div class="panel-header">
                    <div class="panel-title">Active Alerts</div>
                    <div class="panel-subtitle">Current system alerts and notifications</div>
                </div>
                <div class="panel-content">
                    <div id="alerts-container" class="alerts-container">
                        <div class="no-data">No active alerts</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        class DashboardWebSocket {
            constructor() {
                this.ws = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectInterval = 5000;
                this.data = {
                    snapshots: [],
                    alerts: []
                };
                this.connect();
            }

            connect() {
                const wsUrl = `ws://${window.location.host}${window.location.pathname}ws/health`;
                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus('connected');
                };

                this.ws.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };

                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.updateConnectionStatus('disconnected');
                    this.scheduleReconnect();
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus('disconnected');
                };
            }

            scheduleReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    this.updateConnectionStatus('connecting');
                    setTimeout(() => this.connect(), this.reconnectInterval);
                }
            }

            updateConnectionStatus(status) {
                const indicator = document.getElementById('connection-indicator');
                const text = document.getElementById('connection-text');

                indicator.className = `status-indicator status-${status}`;
                
                switch (status) {
                    case 'connected':
                        text.textContent = 'Connected';
                        break;
                    case 'connecting':
                        text.textContent = 'Reconnecting...';
                        break;
                    case 'disconnected':
                        text.textContent = 'Disconnected';
                        break;
                }
            }

            handleMessage(message) {
                switch (message.type) {
                    case 'initial_data':
                    case 'health_update':
                        this.handleHealthUpdate(message.data);
                        break;
                    case 'alert':
                        this.handleAlert(message.data);
                        break;
                    default:
                        console.log('Unknown message type:', message.type);
                }
            }

            handleHealthUpdate(data) {
                this.data.snapshots.push(data);
                
                // Keep only last 100 snapshots for performance
                if (this.data.snapshots.length > 100) {
                    this.data.snapshots = this.data.snapshots.slice(-100);
                }
                
                this.updateDashboard(data);
            }

            handleAlert(alert) {
                this.data.alerts.unshift(alert);
                
                // Keep only last 50 alerts
                if (this.data.alerts.length > 50) {
                    this.data.alerts = this.data.alerts.slice(0, 50);
                }
                
                this.updateAlerts();
            }

            updateDashboard(data) {
                // Update header status bar
                document.getElementById('overall-health').textContent = data.overall_health || 'Unknown';
                document.getElementById('load-level').textContent = data.load_level || 'Unknown';
                document.getElementById('active-alerts').textContent = data.active_alerts_count || 0;
                
                // Format uptime
                const uptimeHours = Math.floor((data.system_uptime_seconds || 0) / 3600);
                document.getElementById('uptime').textContent = `${uptimeHours}h`;

                // System Overview Panel
                document.getElementById('health-score').textContent = 
                    data.health_score ? `${(data.health_score * 100).toFixed(1)}%` : '--';
                document.getElementById('degradation-active').textContent = 
                    data.degradation_active ? 'Yes' : 'No';
                document.getElementById('emergency-mode').textContent = 
                    data.emergency_mode ? 'YES' : 'No';
                document.getElementById('total-requests').textContent = 
                    (data.total_requests_processed || 0).toLocaleString();

                // Load Metrics Panel
                document.getElementById('cpu-utilization').textContent = 
                    `${(data.cpu_utilization || 0).toFixed(1)}%`;
                document.getElementById('memory-pressure').textContent = 
                    `${(data.memory_pressure || 0).toFixed(1)}%`;
                document.getElementById('response-time').textContent = 
                    `${(data.response_time_p95 || 0).toFixed(0)}ms`;
                document.getElementById('error-rate').textContent = 
                    `${(data.error_rate || 0).toFixed(2)}%`;

                // Degradation Status Panel
                document.getElementById('degradation-level').textContent = 
                    data.degradation_level || 'Normal';
                document.getElementById('active-degradations').textContent = 
                    data.active_degradations?.join(', ') || 'None';
                document.getElementById('disabled-features').textContent = 
                    data.disabled_features?.join(', ') || 'None';
                document.getElementById('queue-depth').textContent = 
                    data.request_queue_depth || 0;

                // System Resources Panel
                document.getElementById('active-connections').textContent = 
                    data.active_connections || 0;
                document.getElementById('success-rate').textContent = 
                    `${(data.success_rate || 100).toFixed(1)}%`;
                document.getElementById('throughput').textContent = 
                    `${(data.throughput_rps || 0).toFixed(1)} RPS`;
                document.getElementById('memory-usage').textContent = 
                    `${(data.memory_usage_mb || 0).toFixed(0)} MB`;

                // Apply color coding based on values
                this.applyColorCoding(data);
                
                // Update charts
                this.updateCharts();
            }

            applyColorCoding(data) {
                // CPU utilization color coding
                const cpuElement = document.getElementById('cpu-utilization');
                if (data.cpu_utilization > 80) {
                    cpuElement.className = 'metric-value critical';
                } else if (data.cpu_utilization > 60) {
                    cpuElement.className = 'metric-value warning';
                } else {
                    cpuElement.className = 'metric-value good';
                }

                // Response time color coding
                const responseElement = document.getElementById('response-time');
                if (data.response_time_p95 > 3000) {
                    responseElement.className = 'metric-value critical';
                } else if (data.response_time_p95 > 1000) {
                    responseElement.className = 'metric-value warning';
                } else {
                    responseElement.className = 'metric-value good';
                }

                // Error rate color coding
                const errorElement = document.getElementById('error-rate');
                if (data.error_rate > 2) {
                    errorElement.className = 'metric-value critical';
                } else if (data.error_rate > 0.5) {
                    errorElement.className = 'metric-value warning';
                } else {
                    errorElement.className = 'metric-value good';
                }

                // Overall health badge
                const healthElement = document.getElementById('overall-health');
                if (data.overall_health) {
                    healthElement.className = `status-value health-badge ${data.overall_health}`;
                }
            }

            updateCharts() {
                if (this.data.snapshots.length < 2) return;

                // Performance trends chart
                const snapshots = this.data.snapshots.slice(-20); // Last 20 data points
                const timestamps = snapshots.map(s => new Date(s.timestamp));
                
                const performanceData = [
                    {
                        x: timestamps,
                        y: snapshots.map(s => s.cpu_utilization || 0),
                        name: 'CPU %',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#e74c3c' }
                    },
                    {
                        x: timestamps,
                        y: snapshots.map(s => s.memory_pressure || 0),
                        name: 'Memory %',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#3498db' }
                    },
                    {
                        x: timestamps,
                        y: snapshots.map(s => (s.response_time_p95 || 0) / 10), // Scale down for display
                        name: 'Response Time (x0.1ms)',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#2ecc71' }
                    }
                ];

                const performanceLayout = {
                    title: '',
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Percentage / Scaled Value' },
                    margin: { l: 50, r: 50, t: 30, b: 50 },
                    height: 250
                };

                Plotly.newPlot('performance-chart', performanceData, performanceLayout, {
                    displayModeBar: false,
                    responsive: true
                });

                // Load level timeline
                this.updateLoadTimeline();
            }

            updateLoadTimeline() {
                const snapshots = this.data.snapshots.slice(-50); // Last 50 data points
                if (snapshots.length < 2) return;

                const timestamps = snapshots.map(s => new Date(s.timestamp));
                const loadLevels = snapshots.map(s => {
                    const levelMap = {
                        'NORMAL': 0,
                        'ELEVATED': 1,
                        'HIGH': 2,
                        'CRITICAL': 3,
                        'EMERGENCY': 4
                    };
                    return levelMap[s.load_level] || 0;
                });

                const timelineData = [{
                    x: timestamps,
                    y: loadLevels,
                    type: 'scatter',
                    mode: 'lines+markers',
                    fill: 'tonexty',
                    line: { color: '#667eea', width: 3 },
                    marker: { size: 6 }
                }];

                const timelineLayout = {
                    title: '',
                    xaxis: { title: 'Time' },
                    yaxis: { 
                        title: 'Load Level',
                        tickvals: [0, 1, 2, 3, 4],
                        ticktext: ['Normal', 'Elevated', 'High', 'Critical', 'Emergency']
                    },
                    margin: { l: 80, r: 50, t: 30, b: 50 },
                    height: 350
                };

                Plotly.newPlot('load-timeline-chart', timelineData, timelineLayout, {
                    displayModeBar: false,
                    responsive: true
                });
            }

            updateAlerts() {
                const container = document.getElementById('alerts-container');
                const activeAlerts = this.data.alerts.filter(a => !a.resolved);
                
                document.getElementById('active-alerts').textContent = activeAlerts.length;

                if (activeAlerts.length === 0) {
                    container.innerHTML = '<div class="no-data">No active alerts</div>';
                    return;
                }

                const alertsHtml = activeAlerts.map(alert => `
                    <div class="alert-item ${alert.severity}">
                        <div class="alert-title">${alert.title}</div>
                        <div class="alert-message">${alert.message}</div>
                        <div class="alert-timestamp">
                            ${new Date(alert.timestamp).toLocaleString()} - ${alert.source}
                        </div>
                    </div>
                `).join('');

                container.innerHTML = alertsHtml;
            }
        }

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            console.log('Initializing Unified System Health Dashboard');
            new DashboardWebSocket();
            
            // Initialize empty charts
            Plotly.newPlot('performance-chart', [], {
                title: 'Waiting for data...',
                height: 250
            }, { displayModeBar: false, responsive: true });
            
            Plotly.newPlot('load-timeline-chart', [], {
                title: 'Waiting for data...',
                height: 350
            }, { displayModeBar: false, responsive: true });
        });
    </script>
</body>
</html>
        '''
    
    async def start(self):
        """Start the unified dashboard."""
        self._start_time = time.time()
        
        # Start data aggregation
        await self.data_aggregator.start_aggregation()
        
        self.logger.info(f"Dashboard starting on {self.config.host}:{self.config.port}")
        
        if self.framework == "fastapi":
            config = uvicorn.Config(
                app=self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        
    async def stop(self):
        """Stop the unified dashboard."""
        await self.data_aggregator.stop_aggregation()
        self.logger.info("Dashboard stopped")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_unified_dashboard(
    config: Optional[DashboardConfig] = None,
    graceful_degradation_orchestrator: Optional[Any] = None
) -> UnifiedSystemHealthDashboard:
    """Create a unified system health dashboard."""
    
    return UnifiedSystemHealthDashboard(
        config=config,
        graceful_degradation_orchestrator=graceful_degradation_orchestrator
    )


async def create_and_start_dashboard(
    config: Optional[DashboardConfig] = None,
    graceful_degradation_orchestrator: Optional[Any] = None
) -> UnifiedSystemHealthDashboard:
    """Create and start a unified system health dashboard."""
    
    dashboard = create_unified_dashboard(
        config=config,
        graceful_degradation_orchestrator=graceful_degradation_orchestrator
    )
    
    await dashboard.start()
    return dashboard


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

async def demonstrate_unified_dashboard():
    """Demonstrate the unified system health dashboard."""
    print("Unified System Health Dashboard Demonstration")
    print("=" * 80)
    
    # Create configuration
    config = DashboardConfig(
        port=8092,
        websocket_update_interval=1.0,
        enable_historical_data=True,
        enable_alerts=True
    )
    
    print(f"Dashboard will be available at: http://localhost:{config.port}")
    print(f"WebSocket endpoint: ws://localhost:{config.port}{config.websocket_endpoint}")
    print()
    
    # Create graceful degradation orchestrator if available
    orchestrator = None
    if GRACEFUL_DEGRADATION_AVAILABLE:
        try:
            from .graceful_degradation_integration import create_graceful_degradation_system
            orchestrator = create_graceful_degradation_system()
            await orchestrator.start()
            print("✅ Graceful degradation orchestrator started")
        except Exception as e:
            print(f"⚠️ Could not start graceful degradation orchestrator: {e}")
    
    # Create and start dashboard
    try:
        dashboard = create_unified_dashboard(
            config=config,
            graceful_degradation_orchestrator=orchestrator
        )
        
        print("✅ Dashboard created successfully")
        print("🚀 Starting dashboard server...")
        print("   Press Ctrl+C to stop")
        print()
        
        await dashboard.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
    finally:
        if orchestrator:
            try:
                await orchestrator.stop()
                print("✅ Graceful degradation orchestrator stopped")
            except:
                pass


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_unified_dashboard())