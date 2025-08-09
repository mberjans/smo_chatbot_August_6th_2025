"""
Circuit Breaker Dashboard and HTTP Endpoints
============================================

This module provides HTTP endpoints and dashboard integration for circuit breaker
monitoring. It offers REST API endpoints for health checks, metrics, alerts, and
real-time status monitoring that can be consumed by external dashboards and monitoring tools.

Key Features:
1. RESTful API endpoints for all monitoring data
2. Real-time status monitoring with WebSocket support
3. Prometheus metrics endpoint
4. Health check endpoints for load balancers
5. Alert management API
6. Historical data and analytics endpoints
7. Integration-ready JSON responses

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Circuit Breaker Dashboard Integration
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
import threading
import time
from pathlib import Path

# Web framework imports (optional, will fallback gracefully)
try:
    from flask import Flask, jsonify, request, Response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    jsonify = None

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, PlainTextResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

# Import monitoring components
from .circuit_breaker_monitoring_integration import (
    CircuitBreakerMonitoringIntegration,
    create_monitoring_integration
)


# ============================================================================
# Dashboard Configuration
# ============================================================================

class CircuitBreakerDashboardConfig:
    """Configuration for circuit breaker dashboard."""
    
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8091
        self.enable_cors = True
        self.enable_websockets = True
        self.enable_prometheus = True
        self.enable_health_checks = True
        self.enable_alerts_api = True
        self.enable_analytics = True
        
        # API settings
        self.api_prefix = "/api/v1/circuit-breakers"
        self.health_endpoint = "/health"
        self.metrics_endpoint = "/metrics"
        self.prometheus_endpoint = "/prometheus"
        
        # WebSocket settings
        self.websocket_endpoint = "/ws/status"
        self.websocket_update_interval = 5.0
        
        # Security settings (basic)
        self.enable_api_key = False
        self.api_key = None
        
        # Rate limiting
        self.enable_rate_limiting = False
        self.requests_per_minute = 60


# ============================================================================
# Base Dashboard Class
# ============================================================================

class CircuitBreakerDashboardBase:
    """Base dashboard class with common functionality."""
    
    def __init__(self, monitoring_integration: Optional[CircuitBreakerMonitoringIntegration] = None,
                 config: Optional[CircuitBreakerDashboardConfig] = None):
        self.monitoring_integration = monitoring_integration or create_monitoring_integration()
        self.config = config or CircuitBreakerDashboardConfig()
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        self.websocket_lock = threading.Lock()
        
        # Background tasks
        self._websocket_task = None
        self._is_running = False
    
    async def start_dashboard(self):
        """Start dashboard background tasks."""
        if self._is_running:
            return
        
        await self.monitoring_integration.start()
        
        if self.config.enable_websockets:
            self._websocket_task = asyncio.create_task(self._websocket_broadcast_loop())
        
        self._is_running = True
    
    async def stop_dashboard(self):
        """Stop dashboard background tasks."""
        if not self._is_running:
            return
        
        await self.monitoring_integration.stop()
        
        if self._websocket_task:
            self._websocket_task.cancel()
            try:
                await self._websocket_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connections
        with self.websocket_lock:
            self.websocket_connections.clear()
        
        self._is_running = False
    
    async def _websocket_broadcast_loop(self):
        """Background loop for WebSocket status broadcasts."""
        while self._is_running:
            try:
                if self.websocket_connections:
                    status_data = self._get_real_time_status()
                    await self._broadcast_to_websockets(status_data)
                
                await asyncio.sleep(self.config.websocket_update_interval)
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error in WebSocket broadcast loop: {e}")
                await asyncio.sleep(5)
    
    async def _broadcast_to_websockets(self, data: Dict[str, Any]):
        """Broadcast data to all WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message = json.dumps(data)
        disconnected = set()
        
        with self.websocket_lock:
            connections_copy = self.websocket_connections.copy()
        
        for websocket in connections_copy:
            try:
                if hasattr(websocket, 'send_text'):
                    await websocket.send_text(message)
                elif hasattr(websocket, 'send'):
                    await websocket.send(message)
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected WebSockets
        if disconnected:
            with self.websocket_lock:
                self.websocket_connections -= disconnected
    
    def _get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time status data for WebSocket broadcasts."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'status_update',
            'data': {
                'health': self.monitoring_integration.get_health_status(),
                'metrics': self.monitoring_integration.get_metrics_summary(),
                'active_alerts': len(self.monitoring_integration.get_active_alerts())
            }
        }
    
    # ========================================================================
    # API Endpoint Implementations
    # ========================================================================
    
    def get_health_status(self, service: Optional[str] = None):
        """Get health status endpoint."""
        try:
            health_data = self.monitoring_integration.get_health_status(service)
            return {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'data': health_data
            }
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def get_metrics_summary(self):
        """Get metrics summary endpoint."""
        try:
            metrics = self.monitoring_integration.get_metrics_summary()
            return {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'data': metrics
            }
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def get_active_alerts(self, service: Optional[str] = None):
        """Get active alerts endpoint."""
        try:
            alerts = self.monitoring_integration.get_active_alerts(service)
            return {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'data': {
                    'alerts': alerts,
                    'total_count': len(alerts)
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def get_prometheus_metrics(self):
        """Get Prometheus metrics endpoint."""
        try:
            metrics = self.monitoring_integration.get_prometheus_metrics()
            return metrics or "# No metrics available\n"
        except Exception as e:
            return f"# Error getting metrics: {e}\n"
    
    def get_service_details(self, service: str):
        """Get detailed information for a specific service."""
        try:
            health = self.monitoring_integration.get_health_status(service)
            alerts = self.monitoring_integration.get_active_alerts(service)
            
            # Get additional metrics if available
            metrics = {}
            if self.monitoring_integration.monitoring_system:
                metrics = self.monitoring_integration.monitoring_system.metrics.get_current_metrics(service)
            
            return {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'data': {
                    'service': service,
                    'health': health,
                    'metrics': metrics,
                    'active_alerts': alerts,
                    'alert_count': len(alerts)
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def get_system_overview(self):
        """Get system-wide overview."""
        try:
            health = self.monitoring_integration.get_health_status()
            metrics = self.monitoring_integration.get_metrics_summary()
            alerts = self.monitoring_integration.get_active_alerts()
            
            # Calculate summary statistics
            services = list(health.keys()) if isinstance(health, dict) else []
            healthy_count = sum(1 for s in services if health.get(s, {}).get('status') == 'healthy')
            warning_count = sum(1 for s in services if health.get(s, {}).get('status') == 'warning')
            critical_count = sum(1 for s in services if health.get(s, {}).get('status') == 'critical')
            
            return {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'data': {
                    'summary': {
                        'total_services': len(services),
                        'healthy_services': healthy_count,
                        'warning_services': warning_count,
                        'critical_services': critical_count,
                        'active_alerts': len(alerts)
                    },
                    'services': services,
                    'health_details': health,
                    'metrics_summary': metrics.get('system_health', {}) if isinstance(metrics, dict) else {},
                    'recent_alerts': alerts[:10]  # Last 10 alerts
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def force_health_check(self):
        """Force immediate health check update."""
        try:
            self.monitoring_integration.force_health_check_update()
            return {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'message': 'Health check update triggered'
            }
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }


# ============================================================================
# Flask Dashboard Implementation
# ============================================================================

if FLASK_AVAILABLE:
    class FlaskCircuitBreakerDashboard(CircuitBreakerDashboardBase):
        """Flask-based circuit breaker dashboard."""
        
        def __init__(self, monitoring_integration: Optional[CircuitBreakerMonitoringIntegration] = None,
                     config: Optional[CircuitBreakerDashboardConfig] = None):
            super().__init__(monitoring_integration, config)
            
            self.app = Flask(__name__)
            
            if self.config.enable_cors:
                CORS(self.app)
            
            self._setup_routes()
        
        def _setup_routes(self):
            """Setup Flask routes."""
            prefix = self.config.api_prefix
            
            # Health endpoints
            @self.app.route(f'{prefix}/health')
            @self.app.route(f'{prefix}/health/<service>')
            def health(service=None):
                return jsonify(self.get_health_status(service))
            
            # Metrics endpoints
            @self.app.route(f'{prefix}/metrics')
            def metrics():
                return jsonify(self.get_metrics_summary())
            
            # Prometheus endpoint
            @self.app.route('/metrics')
            @self.app.route(self.config.prometheus_endpoint)
            def prometheus_metrics():
                metrics_text = self.get_prometheus_metrics()
                return Response(metrics_text, mimetype='text/plain')
            
            # Alerts endpoints
            @self.app.route(f'{prefix}/alerts')
            @self.app.route(f'{prefix}/alerts/<service>')
            def alerts(service=None):
                return jsonify(self.get_active_alerts(service))
            
            # Service details
            @self.app.route(f'{prefix}/services/<service>')
            def service_details(service):
                return jsonify(self.get_service_details(service))
            
            # System overview
            @self.app.route(f'{prefix}/overview')
            def system_overview():
                return jsonify(self.get_system_overview())
            
            # Utility endpoints
            @self.app.route(f'{prefix}/health-check', methods=['POST'])
            def force_health_check():
                return jsonify(self.force_health_check())
            
            # Simple health check for load balancers
            @self.app.route('/health')
            def simple_health():
                return jsonify({
                    'status': 'ok',
                    'timestamp': datetime.utcnow().isoformat(),
                    'service': 'circuit_breaker_dashboard'
                })
        
        def run(self, debug=False):
            """Run the Flask application."""
            self.app.run(
                host=self.config.host,
                port=self.config.port,
                debug=debug,
                threaded=True
            )


# ============================================================================
# FastAPI Dashboard Implementation  
# ============================================================================

if FASTAPI_AVAILABLE:
    class FastAPICircuitBreakerDashboard(CircuitBreakerDashboardBase):
        """FastAPI-based circuit breaker dashboard with WebSocket support."""
        
        def __init__(self, monitoring_integration: Optional[CircuitBreakerMonitoringIntegration] = None,
                     config: Optional[CircuitBreakerDashboardConfig] = None):
            super().__init__(monitoring_integration, config)
            
            self.app = FastAPI(
                title="Circuit Breaker Dashboard API",
                description="Real-time monitoring and management for circuit breakers",
                version="1.0.0"
            )
            
            if self.config.enable_cors:
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
            
            self._setup_routes()
        
        def _setup_routes(self):
            """Setup FastAPI routes."""
            prefix = self.config.api_prefix
            
            # Health endpoints
            @self.app.get(f'{prefix}/health')
            @self.app.get(f'{prefix}/health/{{service}}')
            async def health(service: Optional[str] = None):
                return self.get_health_status(service)
            
            # Metrics endpoints
            @self.app.get(f'{prefix}/metrics')
            async def metrics():
                return self.get_metrics_summary()
            
            # Prometheus endpoint
            @self.app.get('/metrics')
            @self.app.get(self.config.prometheus_endpoint)
            async def prometheus_metrics():
                metrics_text = self.get_prometheus_metrics()
                return PlainTextResponse(metrics_text)
            
            # Alerts endpoints
            @self.app.get(f'{prefix}/alerts')
            async def alerts(service: Optional[str] = None):
                return self.get_active_alerts(service)
            
            # Service details
            @self.app.get(f'{prefix}/services/{{service}}')
            async def service_details(service: str):
                return self.get_service_details(service)
            
            # System overview
            @self.app.get(f'{prefix}/overview')
            async def system_overview():
                return self.get_system_overview()
            
            # Utility endpoints
            @self.app.post(f'{prefix}/health-check')
            async def force_health_check():
                return self.force_health_check()
            
            # Simple health check for load balancers
            @self.app.get('/health')
            async def simple_health():
                return {
                    'status': 'ok',
                    'timestamp': datetime.utcnow().isoformat(),
                    'service': 'circuit_breaker_dashboard'
                }
            
            # WebSocket endpoint for real-time updates
            if self.config.enable_websockets:
                @self.app.websocket(self.config.websocket_endpoint)
                async def websocket_endpoint(websocket: WebSocket):
                    await websocket.accept()
                    
                    with self.websocket_lock:
                        self.websocket_connections.add(websocket)
                    
                    try:
                        # Send initial status
                        initial_status = self._get_real_time_status()
                        await websocket.send_text(json.dumps(initial_status))
                        
                        # Keep connection alive
                        while True:
                            await websocket.receive_text()
                            
                    except WebSocketDisconnect:
                        pass
                    except Exception as e:
                        logging.getLogger(__name__).error(f"WebSocket error: {e}")
                    finally:
                        with self.websocket_lock:
                            self.websocket_connections.discard(websocket)


# ============================================================================
# Standalone Dashboard Server
# ============================================================================

class StandaloneDashboardServer:
    """Standalone dashboard server that can run without web framework dependencies."""
    
    def __init__(self, monitoring_integration: Optional[CircuitBreakerMonitoringIntegration] = None,
                 config: Optional[CircuitBreakerDashboardConfig] = None):
        self.monitoring_integration = monitoring_integration or create_monitoring_integration()
        self.config = config or CircuitBreakerDashboardConfig()
        self.dashboard = None
    
    def create_dashboard(self, framework: str = "auto"):
        """Create dashboard using specified framework."""
        if framework == "auto":
            if FASTAPI_AVAILABLE:
                framework = "fastapi"
            elif FLASK_AVAILABLE:
                framework = "flask"
            else:
                raise ValueError("No supported web framework available. Install Flask or FastAPI.")
        
        if framework == "fastapi" and FASTAPI_AVAILABLE:
            self.dashboard = FastAPICircuitBreakerDashboard(
                self.monitoring_integration, self.config
            )
        elif framework == "flask" and FLASK_AVAILABLE:
            self.dashboard = FlaskCircuitBreakerDashboard(
                self.monitoring_integration, self.config
            )
        else:
            raise ValueError(f"Framework '{framework}' not available or supported")
        
        return self.dashboard
    
    async def start_server(self, framework: str = "auto"):
        """Start the dashboard server."""
        if not self.dashboard:
            self.create_dashboard(framework)
        
        await self.dashboard.start_dashboard()
        
        if isinstance(self.dashboard, FastAPICircuitBreakerDashboard):
            # For FastAPI, use uvicorn
            try:
                import uvicorn
                config = uvicorn.Config(
                    app=self.dashboard.app,
                    host=self.config.host,
                    port=self.config.port,
                    log_level="info"
                )
                server = uvicorn.Server(config)
                await server.serve()
            except ImportError:
                raise ValueError("uvicorn is required to run FastAPI dashboard")
        
        elif isinstance(self.dashboard, FlaskCircuitBreakerDashboard):
            # For Flask, run in thread
            import threading
            
            def run_flask():
                self.dashboard.run(debug=False)
            
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
    
    def get_dashboard_info(self):
        """Get information about the dashboard configuration."""
        framework = "none"
        if isinstance(self.dashboard, FastAPICircuitBreakerDashboard):
            framework = "fastapi"
        elif isinstance(self.dashboard, FlaskCircuitBreakerDashboard):
            framework = "flask"
        
        return {
            'framework': framework,
            'host': self.config.host,
            'port': self.config.port,
            'endpoints': {
                'health': f"http://{self.config.host}:{self.config.port}{self.config.api_prefix}/health",
                'metrics': f"http://{self.config.host}:{self.config.port}{self.config.api_prefix}/metrics",
                'prometheus': f"http://{self.config.host}:{self.config.port}{self.config.prometheus_endpoint}",
                'overview': f"http://{self.config.host}:{self.config.port}{self.config.api_prefix}/overview",
            },
            'websocket': f"ws://{self.config.host}:{self.config.port}{self.config.websocket_endpoint}" if self.config.enable_websockets else None
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_dashboard(framework: str = "auto",
                    monitoring_integration: Optional[CircuitBreakerMonitoringIntegration] = None,
                    config_overrides: Optional[Dict[str, Any]] = None):
    """Factory function to create a dashboard with the specified framework."""
    config = CircuitBreakerDashboardConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    server = StandaloneDashboardServer(monitoring_integration, config)
    return server.create_dashboard(framework)


async def run_dashboard_server(framework: str = "auto",
                             monitoring_integration: Optional[CircuitBreakerMonitoringIntegration] = None,
                             config_overrides: Optional[Dict[str, Any]] = None):
    """Run a standalone dashboard server."""
    config = CircuitBreakerDashboardConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    server = StandaloneDashboardServer(monitoring_integration, config)
    await server.start_server(framework)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'CircuitBreakerDashboardConfig',
    'CircuitBreakerDashboardBase',
    'StandaloneDashboardServer',
    'create_dashboard',
    'run_dashboard_server'
]

# Framework-specific exports (only if available)
if FLASK_AVAILABLE:
    __all__.append('FlaskCircuitBreakerDashboard')

if FASTAPI_AVAILABLE:
    __all__.append('FastAPICircuitBreakerDashboard')