#!/usr/bin/env python3
"""
Advanced Cleanup Integration for Clinical Metabolomics Oracle LightRAG Integration.

This module provides seamless integration between the advanced cleanup system
and the existing pytest infrastructure, ensuring comprehensive cleanup
mechanisms work harmoniously with current fixtures and test patterns.

Key Features:
1. Seamless integration with existing conftest.py and fixtures
2. Automatic registration of resources with advanced cleanup system
3. Pytest hooks for lifecycle management
4. Error recovery and cleanup failure handling
5. Performance monitoring during tests
6. Automatic report generation
7. Integration with existing TestDataManager

Components:
- AdvancedCleanupIntegrator: Central integration coordinator
- pytest hooks: Automatic lifecycle management
- Fixture bridges: Connect existing fixtures with advanced cleanup
- Error handlers: Robust cleanup failure recovery
- Performance trackers: Monitor cleanup impact on test performance

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import logging
import os
import sys
import time
import threading
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Union, Callable, Generator
import pytest
import weakref
from datetime import datetime, timedelta

# Import advanced cleanup system components
try:
    from .advanced_cleanup_system import (
        AdvancedCleanupOrchestrator, ResourceType, CleanupStrategy, 
        CleanupScope, CleanupPolicy, ResourceThresholds
    )
    from .cleanup_validation_monitor import (
        CleanupValidationMonitor, CleanupValidator, ResourceMonitor,
        PerformanceAnalyzer, CleanupReporter, AlertSystem
    )
    from .test_data_fixtures import TestDataManager, TestDataConfig
except ImportError as e:
    # Handle import for standalone usage
    print(f"Warning: Could not import cleanup system components: {e}")


# =====================================================================
# INTEGRATION CONFIGURATION
# =====================================================================

@dataclass
class CleanupIntegrationConfig:
    """Configuration for advanced cleanup integration."""
    enabled: bool = True
    auto_register_resources: bool = True
    monitor_performance: bool = True
    generate_reports: bool = True
    validate_cleanup: bool = True
    enable_alerts: bool = False  # Disabled by default for tests
    cleanup_on_failure: bool = True
    emergency_cleanup: bool = True
    integration_scope: CleanupScope = CleanupScope.SESSION
    
    # Performance settings
    max_cleanup_time_seconds: float = 30.0
    performance_threshold_multiplier: float = 2.0
    
    # Resource thresholds (test-friendly defaults)
    memory_threshold_mb: float = 512
    file_handle_threshold: int = 200
    db_connection_threshold: int = 10
    temp_file_threshold: int = 50
    temp_size_threshold_mb: float = 100
    
    # Reporting settings
    report_dir: Optional[str] = None
    generate_session_report: bool = True
    generate_failure_reports: bool = True


# =====================================================================
# INTEGRATION COORDINATOR
# =====================================================================

class AdvancedCleanupIntegrator:
    """Central coordinator for integrating advanced cleanup with pytest."""
    
    def __init__(self, config: CleanupIntegrationConfig = None):
        self.config = config or CleanupIntegrationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components based on config
        self._setup_components()
        
        # Integration state
        self._registered_managers = weakref.WeakSet()
        self._test_sessions = {}
        self._performance_tracking = {}
        self._integration_active = False
        self._cleanup_failures = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup pytest integration
        self._setup_pytest_integration()
        
    def _setup_components(self):
        """Initialize cleanup system components based on configuration."""
        if not self.config.enabled:
            self.logger.info("Advanced cleanup integration disabled")
            return
            
        # Create cleanup policy
        policy = CleanupPolicy(
            strategy=CleanupStrategy.DEFERRED,
            scope=self.config.integration_scope,
            resource_types=set(ResourceType),
            validate_cleanup=self.config.validate_cleanup,
            report_cleanup=self.config.generate_reports,
            emergency_cleanup=self.config.emergency_cleanup,
            timeout_seconds=self.config.max_cleanup_time_seconds
        )
        
        # Create resource thresholds
        thresholds = ResourceThresholds(
            memory_mb=self.config.memory_threshold_mb,
            file_handles=self.config.file_handle_threshold,
            db_connections=self.config.db_connection_threshold,
            temp_files=self.config.temp_file_threshold,
            temp_size_mb=self.config.temp_size_threshold_mb
        )
        
        # Initialize cleanup system
        if self.config.monitor_performance or self.config.generate_reports or self.config.validate_cleanup:
            # Use full monitoring system
            report_dir = Path(self.config.report_dir) if self.config.report_dir else None
            self.cleanup_system = CleanupValidationMonitor(
                cleanup_policy=policy,
                thresholds=thresholds,
                report_dir=report_dir
            )
        else:
            # Use basic orchestrator only
            self.cleanup_system = AdvancedCleanupOrchestrator(policy, thresholds)
            
        self.logger.info("Advanced cleanup components initialized")
        
    def _setup_pytest_integration(self):
        """Setup integration with pytest hooks and fixtures."""
        if not self.config.enabled:
            return
            
        # Register pytest hooks
        self._register_pytest_hooks()
        
        # Setup performance monitoring if enabled
        if self.config.monitor_performance:
            self._setup_performance_monitoring()
            
    def _register_pytest_hooks(self):
        """Register pytest hooks for cleanup lifecycle management."""
        # This would typically be done in conftest.py, but we'll provide
        # the hook functions here for integration
        pass
        
    def _setup_performance_monitoring(self):
        """Setup performance monitoring for cleanup operations."""
        if hasattr(self.cleanup_system, 'start_monitoring'):
            try:
                self.cleanup_system.start_monitoring()
                self.logger.info("Performance monitoring started")
            except Exception as e:
                self.logger.warning(f"Could not start performance monitoring: {e}")
                
    def register_test_data_manager(self, manager: TestDataManager, test_id: str = None):
        """Register existing TestDataManager with advanced cleanup system."""
        if not self.config.enabled or not self.config.auto_register_resources:
            return
            
        with self._lock:
            self._registered_managers.add(manager)
            
            # Integrate with advanced cleanup orchestrator
            if hasattr(self.cleanup_system, 'orchestrator'):
                orchestrator = self.cleanup_system.orchestrator
            else:
                orchestrator = self.cleanup_system
                
            orchestrator.register_test_data_manager(manager)
            
            # Register existing resources from the manager
            self._register_manager_resources(manager, orchestrator)
            
            test_id = test_id or f"test_{id(manager)}"
            self.logger.debug(f"Registered TestDataManager for test: {test_id}")
            
    def _register_manager_resources(self, manager: TestDataManager, orchestrator):
        """Register resources from TestDataManager with orchestrator."""
        try:
            # Register database connections
            for conn in manager.db_connections:
                orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
                
            # Register temporary directories
            for temp_dir in manager.temp_dirs:
                orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_dir)
                
        except Exception as e:
            self.logger.warning(f"Error registering manager resources: {e}")
            
    def create_integrated_fixture_bridge(self, existing_manager: TestDataManager):
        """Create a bridge that integrates existing fixtures with advanced cleanup."""
        
        class IntegratedFixtureBridge:
            """Bridge between existing fixtures and advanced cleanup system."""
            
            def __init__(self, integrator, manager):
                self.integrator = integrator
                self.manager = manager
                self.orchestrator = (integrator.cleanup_system.orchestrator 
                                   if hasattr(integrator.cleanup_system, 'orchestrator')
                                   else integrator.cleanup_system)
                self.logger = logging.getLogger(f"{__name__}.FixtureBridge")
                
            def register_file(self, file_obj, auto_close: bool = True):
                """Register file object with both systems."""
                # Register with existing manager (if it has this capability)
                if hasattr(self.manager, 'register_file'):
                    self.manager.register_file(file_obj)
                    
                # Register with advanced cleanup
                self.orchestrator.register_resource(ResourceType.FILE_HANDLES, file_obj)
                
                if auto_close:
                    def close_file():
                        try:
                            if hasattr(file_obj, 'close') and not file_obj.closed:
                                file_obj.close()
                        except Exception as e:
                            self.logger.warning(f"Error closing file {file_obj}: {e}")
                    
                    self.manager.add_cleanup_callback(close_file)
                    
            def register_db_connection(self, conn):
                """Register database connection with both systems."""
                self.manager.register_db_connection(conn)
                self.orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
                
            def register_temp_path(self, path):
                """Register temporary path with both systems."""
                path_obj = Path(path)
                self.manager.register_temp_dir(path_obj)
                self.orchestrator.register_resource(ResourceType.TEMPORARY_FILES, path_obj)
                
            def register_process(self, process):
                """Register process with advanced cleanup."""
                self.orchestrator.register_resource(ResourceType.PROCESSES, process)
                
            def register_memory_object(self, obj):
                """Register object for memory tracking."""
                self.orchestrator.register_resource(ResourceType.MEMORY, obj)
                
            def perform_cleanup(self, force: bool = False):
                """Perform cleanup in both systems."""
                # Advanced cleanup first
                advanced_success = self.orchestrator.cleanup(force=force)
                
                # Then existing manager cleanup
                try:
                    self.manager.cleanup_all()
                    existing_success = True
                except Exception as e:
                    self.logger.error(f"Existing manager cleanup failed: {e}")
                    existing_success = False
                    
                return advanced_success and existing_success
                
            def get_resource_usage(self):
                """Get comprehensive resource usage."""
                if hasattr(self.orchestrator, 'get_resource_usage'):
                    return self.orchestrator.get_resource_usage()
                return {}
                
        return IntegratedFixtureBridge(self, existing_manager)
        
    def handle_test_start(self, test_id: str):
        """Handle test start event."""
        if not self.config.enabled:
            return
            
        with self._lock:
            start_time = datetime.now()
            self._test_sessions[test_id] = {
                'start_time': start_time,
                'resources_at_start': self._get_current_resource_usage()
            }
            
            self.logger.debug(f"Test started: {test_id}")
            
    def handle_test_finish(self, test_id: str, success: bool):
        """Handle test completion event."""
        if not self.config.enabled:
            return
            
        with self._lock:
            end_time = datetime.now()
            test_session = self._test_sessions.get(test_id, {})
            start_time = test_session.get('start_time', end_time)
            
            # Record test performance
            self._record_test_performance(test_id, start_time, end_time, success)
            
            # Perform cleanup if configured
            if self.config.cleanup_on_failure or success:
                self._perform_test_cleanup(test_id, success)
                
            # Clean up session tracking
            self._test_sessions.pop(test_id, None)
            
            self.logger.debug(f"Test finished: {test_id}, success: {success}")
            
    def _get_current_resource_usage(self):
        """Get current system resource usage."""
        if hasattr(self.cleanup_system, 'orchestrator'):
            orchestrator = self.cleanup_system.orchestrator
        else:
            orchestrator = self.cleanup_system
            
        try:
            return orchestrator.get_resource_usage()
        except Exception:
            return {}
            
    def _record_test_performance(self, test_id: str, start_time: datetime, 
                               end_time: datetime, success: bool):
        """Record test performance metrics."""
        if not self.config.monitor_performance:
            return
            
        duration = (end_time - start_time).total_seconds()
        
        self._performance_tracking[test_id] = {
            'duration': duration,
            'success': success,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'resource_usage': self._get_current_resource_usage()
        }
        
    def _perform_test_cleanup(self, test_id: str, success: bool):
        """Perform cleanup for a specific test."""
        try:
            if hasattr(self.cleanup_system, 'perform_cleanup_cycle'):
                # Use full monitoring system
                result = self.cleanup_system.perform_cleanup_cycle(force=not success)
                cleanup_success = result.get('cleanup_success', False)
            else:
                # Use basic orchestrator
                cleanup_success = self.cleanup_system.cleanup(force=not success)
                
            if not cleanup_success:
                self._cleanup_failures.append({
                    'test_id': test_id,
                    'timestamp': datetime.now().isoformat(),
                    'test_success': success
                })
                
        except Exception as e:
            self.logger.error(f"Cleanup failed for test {test_id}: {e}")
            self._cleanup_failures.append({
                'test_id': test_id,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            
    def handle_session_start(self):
        """Handle pytest session start."""
        if not self.config.enabled:
            return
            
        self._integration_active = True
        
        # Start monitoring if available
        if hasattr(self.cleanup_system, 'start_monitoring'):
            try:
                self.cleanup_system.start_monitoring()
            except Exception as e:
                self.logger.warning(f"Could not start session monitoring: {e}")
                
        self.logger.info("Advanced cleanup integration session started")
        
    def handle_session_finish(self):
        """Handle pytest session finish."""
        if not self.config.enabled or not self._integration_active:
            return
            
        try:
            # Perform final cleanup
            self._perform_final_cleanup()
            
            # Generate session report if configured
            if self.config.generate_session_report:
                self._generate_session_report()
                
            # Stop monitoring if available
            if hasattr(self.cleanup_system, 'stop_monitoring'):
                self.cleanup_system.stop_monitoring()
                
        except Exception as e:
            self.logger.error(f"Error in session finish handling: {e}")
        finally:
            self._integration_active = False
            
        self.logger.info("Advanced cleanup integration session finished")
        
    def _perform_final_cleanup(self):
        """Perform final cleanup at session end."""
        try:
            if hasattr(self.cleanup_system, 'perform_cleanup_cycle'):
                result = self.cleanup_system.perform_cleanup_cycle(force=True)
                self.logger.info(f"Final cleanup result: {result.get('cleanup_success', False)}")
            else:
                success = self.cleanup_system.cleanup(force=True)
                self.logger.info(f"Final cleanup success: {success}")
                
        except Exception as e:
            self.logger.error(f"Final cleanup failed: {e}")
            
    def _generate_session_report(self):
        """Generate comprehensive session report."""
        try:
            if hasattr(self.cleanup_system, 'generate_comprehensive_report'):
                report = self.cleanup_system.generate_comprehensive_report()
                report_id = report.get('report_id', 'unknown')
                self.logger.info(f"Session report generated: {report_id}")
                
                # Add integration-specific data
                self._add_integration_data_to_report(report)
                
                return report
        except Exception as e:
            self.logger.error(f"Error generating session report: {e}")
            return None
            
    def _add_integration_data_to_report(self, report: Dict[str, Any]):
        """Add integration-specific data to the report."""
        integration_data = {
            'integration_config': {
                'enabled': self.config.enabled,
                'auto_register_resources': self.config.auto_register_resources,
                'monitor_performance': self.config.monitor_performance,
                'validate_cleanup': self.config.validate_cleanup
            },
            'test_performance': dict(self._performance_tracking),
            'cleanup_failures': self._cleanup_failures.copy(),
            'registered_managers_count': len(self._registered_managers)
        }
        
        report['integration_data'] = integration_data
        
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        return {
            'config': {
                'enabled': self.config.enabled,
                'integration_scope': self.config.integration_scope.name,
                'monitor_performance': self.config.monitor_performance,
                'validate_cleanup': self.config.validate_cleanup
            },
            'session_stats': {
                'integration_active': self._integration_active,
                'registered_managers': len(self._registered_managers),
                'tracked_tests': len(self._performance_tracking),
                'cleanup_failures': len(self._cleanup_failures)
            },
            'performance_summary': self._get_performance_summary(),
            'cleanup_failure_summary': self._get_cleanup_failure_summary()
        }
        
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracked tests."""
        if not self._performance_tracking:
            return {'message': 'No performance data available'}
            
        durations = [data['duration'] for data in self._performance_tracking.values()]
        successes = [data['success'] for data in self._performance_tracking.values()]
        
        return {
            'total_tests': len(durations),
            'successful_tests': sum(successes),
            'success_rate': sum(successes) / len(successes) * 100,
            'average_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations)
        }
        
    def _get_cleanup_failure_summary(self) -> Dict[str, Any]:
        """Get cleanup failure summary."""
        if not self._cleanup_failures:
            return {'message': 'No cleanup failures recorded'}
            
        return {
            'total_failures': len(self._cleanup_failures),
            'recent_failures': self._cleanup_failures[-5:],  # Last 5 failures
            'failure_rate': len(self._cleanup_failures) / max(len(self._performance_tracking), 1) * 100
        }


# =====================================================================
# PYTEST FIXTURES FOR INTEGRATION
# =====================================================================

# Global integrator instance
_global_integrator = None

def get_cleanup_integrator() -> AdvancedCleanupIntegrator:
    """Get or create global cleanup integrator instance."""
    global _global_integrator
    if _global_integrator is None:
        _global_integrator = AdvancedCleanupIntegrator()
    return _global_integrator


@pytest.fixture(scope="session", autouse=True)
def advanced_cleanup_session():
    """Session-level fixture for advanced cleanup integration."""
    integrator = get_cleanup_integrator()
    
    try:
        integrator.handle_session_start()
        yield integrator
    finally:
        integrator.handle_session_finish()


@pytest.fixture(scope="function")
def advanced_cleanup_bridge(test_data_manager):
    """Function-level fixture providing integrated cleanup bridge."""
    integrator = get_cleanup_integrator()
    
    # Register the test data manager
    test_id = f"{pytest.current_test_id if hasattr(pytest, 'current_test_id') else 'unknown'}"
    integrator.register_test_data_manager(test_data_manager, test_id)
    
    # Create bridge
    bridge = integrator.create_integrated_fixture_bridge(test_data_manager)
    
    # Handle test lifecycle
    integrator.handle_test_start(test_id)
    
    try:
        yield bridge
    finally:
        integrator.handle_test_finish(test_id, True)  # Assume success if no exception


@pytest.fixture(scope="function")
def cleanup_performance_tracker():
    """Fixture for tracking cleanup performance in individual tests."""
    integrator = get_cleanup_integrator()
    
    if not integrator.config.monitor_performance:
        yield None
        return
        
    class PerformanceTracker:
        def __init__(self, integrator):
            self.integrator = integrator
            self.start_time = None
            self.checkpoints = []
            
        def start_tracking(self):
            """Start performance tracking."""
            self.start_time = datetime.now()
            
        def add_checkpoint(self, name: str):
            """Add a performance checkpoint."""
            if self.start_time:
                checkpoint_time = datetime.now()
                elapsed = (checkpoint_time - self.start_time).total_seconds()
                self.checkpoints.append({
                    'name': name,
                    'timestamp': checkpoint_time.isoformat(),
                    'elapsed_seconds': elapsed
                })
                
        def get_performance_data(self):
            """Get performance data."""
            return {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'checkpoints': self.checkpoints.copy()
            }
            
    tracker = PerformanceTracker(integrator)
    tracker.start_tracking()
    
    yield tracker


# =====================================================================
# PYTEST HOOKS
# =====================================================================

def pytest_sessionstart(session):
    """Handle pytest session start."""
    integrator = get_cleanup_integrator()
    integrator.handle_session_start()


def pytest_sessionfinish(session, exitstatus):
    """Handle pytest session finish."""
    integrator = get_cleanup_integrator()
    integrator.handle_session_finish()


def pytest_runtest_setup(item):
    """Handle test setup."""
    integrator = get_cleanup_integrator()
    test_id = f"{item.nodeid}"
    integrator.handle_test_start(test_id)


def pytest_runtest_teardown(item, nextitem):
    """Handle test teardown."""
    integrator = get_cleanup_integrator()
    test_id = f"{item.nodeid}"
    
    # Determine if test passed (this is a simplified check)
    success = True
    if hasattr(item, 'rep_call') and item.rep_call.failed:
        success = False
        
    integrator.handle_test_finish(test_id, success)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test reports available for cleanup decisions."""
    outcome = yield
    rep = outcome.get_result()
    
    # Store report in item for later use
    setattr(item, f"rep_{rep.when}", rep)


# =====================================================================
# CONTEXT MANAGERS FOR MANUAL INTEGRATION
# =====================================================================

@contextmanager
def advanced_cleanup_context(config: CleanupIntegrationConfig = None):
    """Context manager for manual advanced cleanup integration."""
    integrator = AdvancedCleanupIntegrator(config)
    
    try:
        integrator.handle_session_start()
        yield integrator
    finally:
        integrator.handle_session_finish()


@asynccontextmanager
async def async_advanced_cleanup_context(config: CleanupIntegrationConfig = None):
    """Async context manager for advanced cleanup integration."""
    integrator = AdvancedCleanupIntegrator(config)
    
    try:
        integrator.handle_session_start()
        yield integrator
    finally:
        # Run cleanup in thread to avoid blocking async context
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, integrator.handle_session_finish)


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def configure_advanced_cleanup_for_tests(
    memory_threshold_mb: float = 256,
    file_handle_threshold: int = 100,
    generate_reports: bool = False,
    monitor_performance: bool = False
) -> CleanupIntegrationConfig:
    """Create test-friendly cleanup integration configuration."""
    return CleanupIntegrationConfig(
        enabled=True,
        auto_register_resources=True,
        monitor_performance=monitor_performance,
        generate_reports=generate_reports,
        validate_cleanup=True,
        enable_alerts=False,  # Usually disabled for tests
        cleanup_on_failure=True,
        emergency_cleanup=True,
        integration_scope=CleanupScope.FUNCTION,  # More aggressive for tests
        
        # Test-friendly thresholds
        memory_threshold_mb=memory_threshold_mb,
        file_handle_threshold=file_handle_threshold,
        db_connection_threshold=5,
        temp_file_threshold=20,
        temp_size_threshold_mb=50,
        
        # Performance settings
        max_cleanup_time_seconds=10.0,  # Shorter for tests
        performance_threshold_multiplier=3.0  # More lenient for tests
    )


def get_cleanup_integration_report() -> Dict[str, Any]:
    """Get comprehensive integration report."""
    integrator = get_cleanup_integrator()
    return integrator.get_integration_statistics()


def force_cleanup_all_resources():
    """Force cleanup of all tracked resources (emergency use)."""
    integrator = get_cleanup_integrator()
    
    try:
        if hasattr(integrator.cleanup_system, 'perform_cleanup_cycle'):
            result = integrator.cleanup_system.perform_cleanup_cycle(force=True)
            return result.get('cleanup_success', False)
        else:
            return integrator.cleanup_system.cleanup(force=True)
    except Exception as e:
        logging.error(f"Emergency cleanup failed: {e}")
        return False


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Create test-friendly configuration
    config = configure_advanced_cleanup_for_tests(
        memory_threshold_mb=128,
        generate_reports=True,
        monitor_performance=True
    )
    
    # Demonstrate manual integration
    with advanced_cleanup_context(config) as integrator:
        print("Advanced cleanup integration demo")
        print(f"Configuration: {integrator.config}")
        
        # Simulate test registration
        from test_data_fixtures import TestDataManager, TestDataConfig
        test_config = TestDataConfig()
        test_manager = TestDataManager(test_config)
        
        integrator.register_test_data_manager(test_manager, "demo_test")
        
        # Create bridge
        bridge = integrator.create_integrated_fixture_bridge(test_manager)
        
        print("Integration bridge created successfully")
        
        # Get final statistics
        stats = integrator.get_integration_statistics()
        print(f"Integration statistics: {stats}")