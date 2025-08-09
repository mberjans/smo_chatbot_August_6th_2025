"""
Circuit Breaker Monitoring System Demonstration
===============================================

This demonstration script shows how to integrate and use the comprehensive
circuit breaker monitoring system with enhanced circuit breakers.

The demo covers:
1. Setting up comprehensive monitoring
2. Registering circuit breakers for monitoring
3. Simulating real-world operations and failures
4. Monitoring health status and alerts
5. Accessing monitoring data via API endpoints
6. Dashboard integration

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Circuit Breaker Monitoring Demo
Version: 1.0.0

Usage:
    python demo_circuit_breaker_monitoring.py [--enable-dashboard] [--duration 60]
"""

import asyncio
import logging
import argparse
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import random
from pathlib import Path

# Import monitoring system components
from enhanced_circuit_breaker_monitoring_integration import (
    EnhancedCircuitBreakerMonitoringManager,
    EnhancedCircuitBreakerMonitoringConfig,
    create_enhanced_monitoring_manager,
    setup_comprehensive_monitoring
)

from circuit_breaker_dashboard import (
    CircuitBreakerDashboardConfig,
    StandaloneDashboardServer,
    run_dashboard_server
)


# ============================================================================
# Mock Circuit Breaker Classes for Demo
# ============================================================================

class MockCircuitBreaker:
    """Mock circuit breaker for demonstration purposes."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
    async def call(self, operation, *args, **kwargs):
        """Mock call method that simulates success/failure."""
        operation_name = getattr(operation, '__name__', 'unknown_operation')
        
        # Simulate operation
        await asyncio.sleep(random.uniform(0.5, 2.0))  # Random response time
        
        # Simulate failure probability based on current state
        failure_probability = 0.1 if self.state == "closed" else 0.8
        
        if random.random() < failure_probability:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Change state if too many failures
            if self.failure_count >= 3 and self.state == "closed":
                self.state = "open"
            
            # Simulate different types of failures
            failure_types = [
                "Connection timeout",
                "Rate limit exceeded", 
                "Service unavailable",
                "Authentication failed",
                "HTTP 500 error"
            ]
            raise Exception(random.choice(failure_types))
        else:
            self.success_count += 1
            
            # Potentially recover if in open state
            if self.state == "open" and random.random() < 0.3:
                self.state = "half_open"
            elif self.state == "half_open" and self.success_count % 3 == 0:
                self.state = "closed"
                self.failure_count = 0
            
            return f"Success from {operation_name}"


class MockOpenAICircuitBreaker(MockCircuitBreaker):
    def __init__(self):
        super().__init__("openai_api")
        self.__class__.__name__ = "OpenAICircuitBreaker"


class MockPerplexityCircuitBreaker(MockCircuitBreaker):
    def __init__(self):
        super().__init__("perplexity_api") 
        self.__class__.__name__ = "PerplexityCircuitBreaker"


class MockLightRAGCircuitBreaker(MockCircuitBreaker):
    def __init__(self):
        super().__init__("lightrag")
        self.__class__.__name__ = "LightRAGCircuitBreaker"


class MockCacheCircuitBreaker(MockCircuitBreaker):
    def __init__(self):
        super().__init__("cache")
        self.__class__.__name__ = "CacheCircuitBreaker"


class MockCircuitBreakerOrchestrator:
    """Mock orchestrator that holds all circuit breakers."""
    
    def __init__(self):
        self.openai_cb = MockOpenAICircuitBreaker()
        self.perplexity_cb = MockPerplexityCircuitBreaker()
        self.lightrag_cb = MockLightRAGCircuitBreaker()
        self.cache_cb = MockCacheCircuitBreaker()


# ============================================================================
# Demo Configuration
# ============================================================================

class DemoConfig:
    """Configuration for the monitoring demo."""
    
    def __init__(self):
        self.demo_duration = 60  # seconds
        self.enable_dashboard = False
        self.dashboard_port = 8091
        self.operation_interval = 2.0  # seconds between operations
        self.enable_logging = True
        self.log_level = "INFO"
        
        # Create logs directory
        self.log_dir = Path("logs/demo")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitoring_log_file = str(self.log_dir / "monitoring_demo.log")
        self.alerts_file = str(self.log_dir / "demo_alerts.json")


# ============================================================================
# Demo Operations
# ============================================================================

class DemoOperations:
    """Demo operations that simulate real-world circuit breaker usage."""
    
    def __init__(self):
        self.operation_count = 0
    
    async def chat_completion(self):
        """Simulate OpenAI chat completion."""
        self.operation_count += 1
        await asyncio.sleep(random.uniform(1.0, 3.0))
        return f"Chat completion {self.operation_count}"
    
    async def perplexity_search(self):
        """Simulate Perplexity search operation."""
        self.operation_count += 1
        await asyncio.sleep(random.uniform(0.5, 2.5))
        return f"Search result {self.operation_count}"
    
    async def lightrag_query(self):
        """Simulate LightRAG query operation."""
        self.operation_count += 1
        await asyncio.sleep(random.uniform(1.5, 4.0))
        return f"RAG query {self.operation_count}"
    
    async def cache_get(self):
        """Simulate cache get operation."""
        self.operation_count += 1
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return f"Cache value {self.operation_count}"


# ============================================================================
# Demo Monitor
# ============================================================================

class DemoMonitor:
    """Monitor for displaying demo progress and statistics."""
    
    def __init__(self, monitoring_manager: EnhancedCircuitBreakerMonitoringManager):
        self.monitoring_manager = monitoring_manager
        self.start_time = None
        self.last_status_update = 0
        
    async def start_monitoring_display(self):
        """Start the monitoring display loop."""
        self.start_time = time.time()
        
        while True:
            current_time = time.time()
            
            # Update display every 10 seconds
            if current_time - self.last_status_update >= 10:
                await self.display_status()
                self.last_status_update = current_time
            
            await asyncio.sleep(1)
    
    async def display_status(self):
        """Display current monitoring status."""
        if not self.monitoring_manager._is_started:
            return
        
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print(f"DEMO MONITORING STATUS - Elapsed Time: {elapsed_time:.1f}s")
        print("=" * 80)
        
        # Get monitoring status
        status = self.monitoring_manager.get_monitoring_status()
        
        print(f"Monitoring Enabled: {status['monitoring_enabled']}")
        print(f"Monitored Services: {', '.join(status['monitored_services'])}")
        print(f"Active Alerts: {status.get('active_alerts', 0)}")
        
        # Get health status
        health = self.monitoring_manager.get_service_health()
        if isinstance(health, dict) and 'error' not in health:
            print("\nSERVICE HEALTH STATUS:")
            for service, service_health in health.items():
                if isinstance(service_health, dict):
                    status_indicator = self._get_status_indicator(service_health.get('status', 'unknown'))
                    print(f"  {service}: {status_indicator} {service_health.get('status', 'unknown')}")
        
        # Get event statistics
        if 'event_statistics' in status:
            stats = status['event_statistics']
            print(f"\nEVENT STATISTICS:")
            print(f"  Events Processed: {stats.get('events_processed', 0)}")
            print(f"  Events Failed: {stats.get('events_failed', 0)}")
            print(f"  Success Rate: {stats.get('success_rate', 0):.1f}%")
        
        # Get recent alerts
        alerts = self.monitoring_manager.monitoring_integration.get_active_alerts() if self.monitoring_manager.monitoring_integration else []
        if alerts:
            print(f"\nRECENT ALERTS ({len(alerts)} active):")
            for alert in alerts[-3:]:  # Show last 3 alerts
                level_indicator = self._get_alert_level_indicator(alert.get('level', 'info'))
                print(f"  {level_indicator} {alert.get('service', 'unknown')}: {alert.get('message', 'No message')}")
        
        print("=" * 80)
    
    def _get_status_indicator(self, status: str) -> str:
        """Get visual indicator for health status."""
        indicators = {
            'healthy': '🟢',
            'warning': '🟡',
            'critical': '🔴',
            'unknown': '⚫'
        }
        return indicators.get(status.lower(), '⚫')
    
    def _get_alert_level_indicator(self, level: str) -> str:
        """Get visual indicator for alert level."""
        indicators = {
            'info': '📘',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        return indicators.get(level.lower(), '📘')


# ============================================================================
# Demo Workload Generator
# ============================================================================

class DemoWorkloadGenerator:
    """Generates realistic workload for circuit breaker testing."""
    
    def __init__(self, monitoring_manager: EnhancedCircuitBreakerMonitoringManager,
                 demo_operations: DemoOperations):
        self.monitoring_manager = monitoring_manager
        self.demo_operations = demo_operations
        self.running = False
        
    async def start_workload(self, duration: int):
        """Start generating workload for the specified duration."""
        self.running = True
        end_time = time.time() + duration
        
        # Get monitored circuit breakers
        circuit_breakers = self.monitoring_manager.monitored_circuit_breakers
        
        # Define operations for each service
        service_operations = {
            'openai_api': self.demo_operations.chat_completion,
            'perplexity_api': self.demo_operations.perplexity_search,
            'lightrag': self.demo_operations.lightrag_query,
            'cache': self.demo_operations.cache_get
        }
        
        print("🚀 Starting workload generation...")
        
        tasks = []
        
        # Create tasks for each service
        for service_name, monitored_cb in circuit_breakers.items():
            operation = service_operations.get(service_name)
            if operation:
                task = asyncio.create_task(
                    self._generate_service_workload(
                        service_name, monitored_cb, operation, end_time
                    )
                )
                tasks.append(task)
        
        # Wait for all workload tasks to complete
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.running = False
            print("✅ Workload generation completed")
    
    async def _generate_service_workload(self, service_name: str,
                                       monitored_cb, operation: callable,
                                       end_time: float):
        """Generate workload for a specific service."""
        operation_count = 0
        
        while time.time() < end_time and self.running:
            try:
                # Execute operation through monitored circuit breaker
                result = await monitored_cb.call(operation)
                operation_count += 1
                
                print(f"✅ {service_name}: Operation {operation_count} succeeded")
                
            except Exception as e:
                operation_count += 1
                print(f"❌ {service_name}: Operation {operation_count} failed: {str(e)[:50]}...")
            
            # Wait before next operation
            await asyncio.sleep(random.uniform(1.0, 3.0))
        
        print(f"📊 {service_name}: Completed {operation_count} operations")


# ============================================================================
# Main Demo Runner
# ============================================================================

class CircuitBreakerMonitoringDemo:
    """Main demo runner that orchestrates the entire demonstration."""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.monitoring_manager = None
        self.dashboard_server = None
        
        # Setup logging
        if config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(config.monitoring_log_file)
                ]
            )
    
    async def run_demo(self):
        """Run the complete demonstration."""
        print("🎯 Starting Circuit Breaker Monitoring System Demo")
        print(f"⏰ Demo Duration: {self.config.demo_duration} seconds")
        print(f"📊 Dashboard Enabled: {self.config.enable_dashboard}")
        print(f"📝 Logs: {self.config.monitoring_log_file}")
        
        try:
            # Step 1: Setup monitoring system
            await self._setup_monitoring_system()
            
            # Step 2: Setup dashboard if enabled
            if self.config.enable_dashboard:
                await self._setup_dashboard()
            
            # Step 3: Register circuit breakers
            self._register_circuit_breakers()
            
            # Step 4: Start demo workload and monitoring
            await self._run_demo_workload()
            
            # Step 5: Display final results
            await self._display_final_results()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logging.exception("Demo execution failed")
        finally:
            await self._cleanup()
    
    async def _setup_monitoring_system(self):
        """Setup the monitoring system."""
        print("🔧 Setting up monitoring system...")
        
        config_overrides = {
            'enable_monitoring': True,
            'monitoring_log_file': self.config.monitoring_log_file,
            'enable_dashboard': self.config.enable_dashboard,
            'dashboard_port': self.config.dashboard_port,
            'integrate_with_production_monitoring': False,
            'enable_critical_alerts': True,
            'enable_performance_alerts': True
        }
        
        self.monitoring_manager = create_enhanced_monitoring_manager(config_overrides)
        await self.monitoring_manager.start()
        
        print("✅ Monitoring system started")
    
    async def _setup_dashboard(self):
        """Setup the dashboard server."""
        print(f"🌐 Setting up dashboard on port {self.config.dashboard_port}...")
        
        try:
            dashboard_config = CircuitBreakerDashboardConfig()
            dashboard_config.port = self.config.dashboard_port
            
            self.dashboard_server = StandaloneDashboardServer(
                self.monitoring_manager.monitoring_integration,
                dashboard_config
            )
            
            # Start dashboard in background
            dashboard_task = asyncio.create_task(
                self.dashboard_server.start_server()
            )
            
            # Give dashboard time to start
            await asyncio.sleep(2)
            
            dashboard_info = self.dashboard_server.get_dashboard_info()
            print(f"✅ Dashboard available at: {dashboard_info.get('endpoints', {}).get('overview', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️ Dashboard setup failed: {e}")
            self.config.enable_dashboard = False
    
    def _register_circuit_breakers(self):
        """Register circuit breakers for monitoring."""
        print("📝 Registering circuit breakers...")
        
        # Create mock orchestrator
        mock_orchestrator = MockCircuitBreakerOrchestrator()
        
        # Register each circuit breaker
        services_registered = []
        
        if hasattr(mock_orchestrator, 'openai_cb'):
            self.monitoring_manager.register_circuit_breaker(
                mock_orchestrator.openai_cb, 'openai_api'
            )
            services_registered.append('openai_api')
        
        if hasattr(mock_orchestrator, 'perplexity_cb'):
            self.monitoring_manager.register_circuit_breaker(
                mock_orchestrator.perplexity_cb, 'perplexity_api'
            )
            services_registered.append('perplexity_api')
        
        if hasattr(mock_orchestrator, 'lightrag_cb'):
            self.monitoring_manager.register_circuit_breaker(
                mock_orchestrator.lightrag_cb, 'lightrag'
            )
            services_registered.append('lightrag')
        
        if hasattr(mock_orchestrator, 'cache_cb'):
            self.monitoring_manager.register_circuit_breaker(
                mock_orchestrator.cache_cb, 'cache'
            )
            services_registered.append('cache')
        
        print(f"✅ Registered services: {', '.join(services_registered)}")
    
    async def _run_demo_workload(self):
        """Run the demo workload with monitoring."""
        print("🏃 Starting demo workload...")
        
        demo_operations = DemoOperations()
        workload_generator = DemoWorkloadGenerator(self.monitoring_manager, demo_operations)
        demo_monitor = DemoMonitor(self.monitoring_manager)
        
        # Start monitoring display
        monitor_task = asyncio.create_task(demo_monitor.start_monitoring_display())
        
        # Start workload generation
        workload_task = asyncio.create_task(
            workload_generator.start_workload(self.config.demo_duration)
        )
        
        try:
            # Wait for workload to complete
            await workload_task
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _display_final_results(self):
        """Display final demo results."""
        print("\n" + "🎉" * 20)
        print("DEMO COMPLETED - FINAL RESULTS")
        print("🎉" * 20)
        
        if self.monitoring_manager:
            status = self.monitoring_manager.get_monitoring_status()
            health = self.monitoring_manager.get_service_health()
            
            print(f"\n📊 FINAL MONITORING STATUS:")
            print(f"  Services Monitored: {len(status.get('monitored_services', []))}")
            print(f"  Total Events Processed: {status.get('event_statistics', {}).get('events_processed', 0)}")
            print(f"  Event Success Rate: {status.get('event_statistics', {}).get('success_rate', 0):.1f}%")
            print(f"  Active Alerts: {status.get('active_alerts', 0)}")
            
            # Display service health summary
            if isinstance(health, dict) and 'error' not in health:
                print(f"\n🏥 SERVICE HEALTH SUMMARY:")
                for service, service_health in health.items():
                    if isinstance(service_health, dict):
                        status_name = service_health.get('status', 'unknown')
                        health_score = service_health.get('health_score', 0)
                        print(f"  {service}: {status_name} (score: {health_score:.2f})")
            
            # Display alerts summary
            if self.monitoring_manager.monitoring_integration:
                alerts = self.monitoring_manager.monitoring_integration.get_active_alerts()
                alert_history = []
                
                print(f"\n🚨 ALERTS SUMMARY:")
                print(f"  Active Alerts: {len(alerts)}")
                
                if alerts:
                    alert_types = {}
                    for alert in alerts:
                        alert_type = alert.get('alert_type', 'unknown')
                        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                    
                    for alert_type, count in alert_types.items():
                        print(f"    {alert_type}: {count}")
            
            if self.config.enable_dashboard:
                dashboard_info = self.dashboard_server.get_dashboard_info() if self.dashboard_server else {}
                print(f"\n🌐 DASHBOARD ENDPOINTS:")
                endpoints = dashboard_info.get('endpoints', {})
                for endpoint_name, url in endpoints.items():
                    print(f"  {endpoint_name}: {url}")
        
        print(f"\n📁 DEMO FILES GENERATED:")
        print(f"  Monitoring Log: {self.config.monitoring_log_file}")
        print(f"  Alerts File: {self.config.alerts_file}")
        
        print("\n✅ Demo completed successfully!")
    
    async def _cleanup(self):
        """Cleanup resources."""
        print("🧹 Cleaning up...")
        
        if self.monitoring_manager:
            await self.monitoring_manager.stop()
        
        print("✅ Cleanup completed")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="Circuit Breaker Monitoring System Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_circuit_breaker_monitoring.py
  python demo_circuit_breaker_monitoring.py --enable-dashboard --duration 120
  python demo_circuit_breaker_monitoring.py --dashboard-port 9091
        """
    )
    
    parser.add_argument(
        '--enable-dashboard',
        action='store_true',
        help='Enable web dashboard (default: False)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Demo duration in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=8091,
        help='Dashboard port (default: 8091)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Create demo configuration
    config = DemoConfig()
    config.enable_dashboard = args.enable_dashboard
    config.demo_duration = args.duration
    config.dashboard_port = args.dashboard_port
    config.log_level = args.log_level
    
    # Run demo
    demo = CircuitBreakerMonitoringDemo(config)
    
    try:
        asyncio.run(demo.run_demo())
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logging.exception("Demo execution failed")


if __name__ == "__main__":
    main()