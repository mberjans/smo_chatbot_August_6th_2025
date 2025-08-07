#!/usr/bin/env python3
"""
Configuration Test Utilities and Resource Cleanup Management - CMO-LIGHTRAG-008-T06.

This module provides the capstone utilities for the Clinical Metabolomics Oracle test framework,
implementing comprehensive configuration management and resource cleanup coordination across
all test utilities. It serves as the central integration point that ties together:

- TestEnvironmentManager for environment coordination
- MockSystemFactory for mock configuration  
- PerformanceAssertionHelper for performance configuration
- AsyncTestCoordinator for async configuration
- All validation utilities for configuration validation

Key Components:
1. ConfigurationTestHelper: Standard configuration scenarios and management
2. ResourceCleanupManager: Automated resource cleanup across all utilities
3. TestConfigurationTemplates: Predefined configurations for different scenarios
4. EnvironmentIsolationManager: Test isolation and sandboxing
5. ConfigurationValidationSuite: Configuration testing and validation

This implementation provides:
- Configuration templates for unit, integration, performance, and stress testing
- Environment variable management with automatic restoration
- Resource tracking and cleanup coordination across all test utilities
- Configuration validation with comprehensive diagnostics
- Cross-utility coordination and cleanup management
- Memory leak detection and prevention
- Process cleanup and resource management

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
License: Clinical Metabolomics Oracle Project License
"""

import pytest
import asyncio
import time
import json
import os
import sys
import shutil
import tempfile
import logging
import threading
import weakref
import psutil
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type, Set, ContextManager
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager, asynccontextmanager, ExitStack
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import atexit
import signal
import subprocess

# Import from existing test infrastructure
from test_utilities import (
    TestEnvironmentManager, MockSystemFactory, SystemComponent, 
    TestComplexity, MockBehavior, EnvironmentSpec, MockSpec, MemoryMonitor
)
from async_test_utilities import (
    AsyncTestCoordinator, AsyncTestSession, AsyncOperationSpec, 
    ConcurrencyPolicy, AsyncTestState
)
from performance_test_utilities import (
    PerformanceAssertionHelper, PerformanceBenchmarkSuite, 
    AdvancedResourceMonitor, PerformanceThreshold
)
from validation_test_utilities import (
    BiomedicalContentValidator, TestResultValidator, 
    TestValidationType, ValidationSeverity
)


# =====================================================================
# CONFIGURATION ENUMS AND DATA CLASSES
# =====================================================================

class TestScenarioType(Enum):
    """Test scenario types for configuration templates."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    STRESS_TEST = "stress_test"
    E2E_TEST = "e2e_test"
    VALIDATION_TEST = "validation_test"
    MOCK_TEST = "mock_test"
    ASYNC_TEST = "async_test"
    BIOMEDICAL_TEST = "biomedical_test"
    CLEANUP_TEST = "cleanup_test"


class ConfigurationScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    SESSION = "session"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    TEMPORARY = "temporary"


class ResourceType(Enum):
    """Resource types for cleanup management."""
    TEMPORARY_FILES = "temporary_files"
    TEMPORARY_DIRECTORIES = "temporary_directories"
    PROCESSES = "processes"
    THREADS = "threads"
    NETWORK_CONNECTIONS = "network_connections"
    DATABASE_CONNECTIONS = "database_connections"
    ASYNC_TASKS = "async_tasks"
    MOCK_OBJECTS = "mock_objects"
    MEMORY_ALLOCATIONS = "memory_allocations"
    FILE_HANDLES = "file_handles"
    SYSTEM_RESOURCES = "system_resources"


@dataclass
class ConfigurationTemplate:
    """Template for test configuration scenarios."""
    name: str
    scenario_type: TestScenarioType
    description: str
    environment_spec: EnvironmentSpec
    mock_components: List[SystemComponent]
    performance_thresholds: Dict[str, PerformanceThreshold]
    async_config: Dict[str, Any]
    validation_rules: List[str]
    cleanup_priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceTracker:
    """Tracker for resources requiring cleanup."""
    resource_id: str
    resource_type: ResourceType
    resource_data: Any
    creation_time: float
    cleanup_callback: Optional[Callable] = None
    cleanup_priority: int = 5
    cleanup_timeout: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigurationContext:
    """Context for configuration management."""
    context_id: str
    scope: ConfigurationScope
    original_config: Dict[str, Any]
    modified_config: Dict[str, Any]
    environment_vars: Dict[str, str]
    resource_trackers: List[ResourceTracker]
    cleanup_callbacks: List[Callable]
    start_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# CONFIGURATION TEST HELPER
# =====================================================================

class ConfigurationTestHelper:
    """
    Comprehensive configuration management for Clinical Metabolomics Oracle testing.
    
    Provides standard configuration scenarios, environment management, validation,
    and seamless integration with all existing test utilities.
    """
    
    # Standard configuration templates
    CONFIGURATION_TEMPLATES = {
        TestScenarioType.UNIT_TEST: ConfigurationTemplate(
            name="Standard Unit Test",
            scenario_type=TestScenarioType.UNIT_TEST,
            description="Basic unit test configuration with minimal resource usage",
            environment_spec=EnvironmentSpec(
                temp_dirs=["logs", "output"],
                required_imports=[],
                mock_components=[SystemComponent.LOGGER],
                async_context=False,
                performance_monitoring=False,
                cleanup_on_exit=True
            ),
            mock_components=[SystemComponent.LOGGER],
            performance_thresholds={
                "execution_time": PerformanceThreshold(
                    metric_name="execution_time", 
                    threshold_value=5.0, 
                    comparison_operator="lt",
                    unit="seconds"
                )
            },
            async_config={"enabled": False},
            validation_rules=["basic_validation"],
            cleanup_priority=1
        ),
        
        TestScenarioType.INTEGRATION_TEST: ConfigurationTemplate(
            name="Standard Integration Test",
            scenario_type=TestScenarioType.INTEGRATION_TEST,
            description="Integration test with multiple components and moderate resource usage",
            environment_spec=EnvironmentSpec(
                temp_dirs=["logs", "pdfs", "output", "working"],
                required_imports=[
                    "lightrag_integration.clinical_metabolomics_rag",
                    "lightrag_integration.pdf_processor"
                ],
                mock_components=[
                    SystemComponent.LIGHTRAG_SYSTEM,
                    SystemComponent.PDF_PROCESSOR,
                    SystemComponent.COST_MONITOR
                ],
                async_context=True,
                performance_monitoring=True,
                cleanup_on_exit=True
            ),
            mock_components=[
                SystemComponent.LIGHTRAG_SYSTEM,
                SystemComponent.PDF_PROCESSOR,
                SystemComponent.COST_MONITOR,
                SystemComponent.PROGRESS_TRACKER
            ],
            performance_thresholds={
                "execution_time": PerformanceThreshold(
                    metric_name="execution_time", 
                    threshold_value=30.0, 
                    comparison_operator="lt",
                    unit="seconds"
                ),
                "memory_usage": PerformanceThreshold(
                    metric_name="memory_usage", 
                    threshold_value=256.0, 
                    comparison_operator="lt",
                    unit="MB"
                )
            },
            async_config={
                "enabled": True,
                "max_concurrent_operations": 5,
                "timeout_seconds": 30.0
            },
            validation_rules=["response_quality", "integration_validation"],
            cleanup_priority=3
        ),
        
        TestScenarioType.PERFORMANCE_TEST: ConfigurationTemplate(
            name="Performance Test",
            scenario_type=TestScenarioType.PERFORMANCE_TEST,
            description="Performance test with comprehensive monitoring and validation",
            environment_spec=EnvironmentSpec(
                temp_dirs=["logs", "pdfs", "output", "working", "perf_data"],
                required_imports=[
                    "lightrag_integration.clinical_metabolomics_rag",
                    "lightrag_integration.pdf_processor",
                    "lightrag_integration.progress_tracker"
                ],
                mock_components=[SystemComponent.LIGHTRAG_SYSTEM],
                async_context=True,
                performance_monitoring=True,
                memory_limits={'performance_limit': 512},
                cleanup_on_exit=True
            ),
            mock_components=[SystemComponent.LIGHTRAG_SYSTEM],
            performance_thresholds={
                "execution_time": PerformanceThreshold(
                    metric_name="execution_time", 
                    threshold_value=60.0, 
                    comparison_operator="lt",
                    unit="seconds"
                ),
                "memory_usage": PerformanceThreshold(
                    metric_name="memory_usage", 
                    threshold_value=512.0, 
                    comparison_operator="lt",
                    unit="MB"
                ),
                "throughput": PerformanceThreshold(
                    metric_name="throughput", 
                    threshold_value=10.0, 
                    comparison_operator="gt",
                    unit="ops/sec"
                )
            },
            async_config={
                "enabled": True,
                "max_concurrent_operations": 10,
                "timeout_seconds": 60.0,
                "concurrency_policy": "adaptive"
            },
            validation_rules=["performance_validation", "statistical_consistency"],
            cleanup_priority=4
        ),
        
        TestScenarioType.BIOMEDICAL_TEST: ConfigurationTemplate(
            name="Biomedical Content Test",
            scenario_type=TestScenarioType.BIOMEDICAL_TEST,
            description="Biomedical testing with domain-specific validation",
            environment_spec=EnvironmentSpec(
                temp_dirs=["logs", "pdfs", "output", "biomedical_data"],
                required_imports=[
                    "lightrag_integration.clinical_metabolomics_rag",
                    "lightrag_integration.pdf_processor"
                ],
                mock_components=[
                    SystemComponent.LIGHTRAG_SYSTEM,
                    SystemComponent.PDF_PROCESSOR
                ],
                async_context=True,
                performance_monitoring=False,
                cleanup_on_exit=True
            ),
            mock_components=[
                SystemComponent.LIGHTRAG_SYSTEM,
                SystemComponent.PDF_PROCESSOR,
                SystemComponent.LOGGER
            ],
            performance_thresholds={
                "execution_time": PerformanceThreshold(
                    metric_name="execution_time", 
                    threshold_value=45.0, 
                    comparison_operator="lt",
                    unit="seconds"
                )
            },
            async_config={
                "enabled": True,
                "max_concurrent_operations": 3,
                "timeout_seconds": 45.0
            },
            validation_rules=[
                "response_quality", 
                "domain_accuracy", 
                "clinical_metabolomics_validation"
            ],
            cleanup_priority=3
        )
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize configuration test helper."""
        self.logger = logger or logging.getLogger(f"config_helper_{id(self)}")
        self.active_contexts: Dict[str, ConfigurationContext] = {}
        self.global_config: Dict[str, Any] = {}
        self.template_registry: Dict[str, ConfigurationTemplate] = self.CONFIGURATION_TEMPLATES.copy()
        self.environment_manager: Optional[TestEnvironmentManager] = None
        self.mock_factory: Optional[MockSystemFactory] = None
        self.async_coordinator: Optional[AsyncTestCoordinator] = None
        self.performance_helper: Optional[PerformanceAssertionHelper] = None
        self.validation_suite: Optional['ConfigurationValidationSuite'] = None
        
        # Resource cleanup management
        self.cleanup_manager: Optional['ResourceCleanupManager'] = None
        
        # Configuration validation
        self.config_validation_enabled = True
        self.validation_errors: List[str] = []
    
    def create_test_configuration(self, 
                                scenario_type: TestScenarioType,
                                custom_overrides: Optional[Dict[str, Any]] = None,
                                context_scope: ConfigurationScope = ConfigurationScope.FUNCTION) -> str:
        """
        Create comprehensive test configuration for specified scenario.
        
        Args:
            scenario_type: Type of test scenario to configure
            custom_overrides: Optional configuration overrides
            context_scope: Scope level for configuration context
            
        Returns:
            Context ID for the created configuration
        """
        context_id = f"config_{scenario_type.value}_{int(time.time() * 1000)}"
        
        if scenario_type not in self.template_registry:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        template = self.template_registry[scenario_type]
        
        # Create configuration context
        context = ConfigurationContext(
            context_id=context_id,
            scope=context_scope,
            original_config=self.global_config.copy(),
            modified_config={},
            environment_vars={},
            resource_trackers=[],
            cleanup_callbacks=[],
            start_time=time.time(),
            metadata={
                'template_name': template.name,
                'scenario_type': scenario_type.value
            }
        )
        
        # Apply template configuration
        self._apply_template_configuration(template, context, custom_overrides)
        
        # Initialize utility components
        self._initialize_utility_components(template, context)
        
        # Store context
        self.active_contexts[context_id] = context
        
        self.logger.info(f"Created test configuration '{template.name}' with context ID: {context_id}")
        
        return context_id
    
    def _apply_template_configuration(self, 
                                    template: ConfigurationTemplate,
                                    context: ConfigurationContext,
                                    custom_overrides: Optional[Dict[str, Any]]):
        """Apply configuration template to context."""
        # Set up environment manager
        env_spec = template.environment_spec
        if custom_overrides and 'environment' in custom_overrides:
            env_overrides = custom_overrides['environment']
            for key, value in env_overrides.items():
                if hasattr(env_spec, key):
                    setattr(env_spec, key, value)
        
        self.environment_manager = TestEnvironmentManager(env_spec)
        environment_data = self.environment_manager.setup_environment()
        
        context.modified_config['environment'] = environment_data
        
        # Set up mock factory
        self.mock_factory = MockSystemFactory(self.environment_manager)
        
        # Create comprehensive mock system
        if template.mock_components:
            mock_system = self.mock_factory.create_comprehensive_mock_set(
                components=template.mock_components,
                behavior=MockBehavior.SUCCESS
            )
            context.modified_config['mock_system'] = mock_system
        
        # Set up async coordinator if needed
        if template.async_config.get('enabled', False):
            self.async_coordinator = AsyncTestCoordinator(
                logger=self.logger,
                default_timeout=template.async_config.get('timeout_seconds', 30.0),
                max_concurrent_operations=template.async_config.get('max_concurrent_operations', 10)
            )
            context.modified_config['async_coordinator'] = self.async_coordinator
        
        # Set up performance helper if needed
        if template.performance_thresholds:
            self.performance_helper = PerformanceAssertionHelper()
            for threshold_name, threshold in template.performance_thresholds.items():
                self.performance_helper.register_threshold(threshold_name, threshold)
            context.modified_config['performance_helper'] = self.performance_helper
        
        # Apply custom overrides
        if custom_overrides:
            self._apply_custom_overrides(custom_overrides, context)
    
    def _apply_custom_overrides(self, overrides: Dict[str, Any], context: ConfigurationContext):
        """Apply custom configuration overrides."""
        for key, value in overrides.items():
            if key == 'environment_vars':
                # Handle environment variable overrides
                for env_var, env_value in value.items():
                    original_value = os.environ.get(env_var)
                    context.environment_vars[env_var] = original_value
                    os.environ[env_var] = str(env_value)
                    
            elif key == 'mock_behaviors':
                # Handle mock behavior overrides
                if self.mock_factory and 'mock_system' in context.modified_config:
                    mock_system = context.modified_config['mock_system']
                    for component_name, behavior in value.items():
                        if component_name in mock_system:
                            # Update mock behavior (implementation would depend on mock structure)
                            pass
            
            else:
                context.modified_config[key] = value
    
    def _initialize_utility_components(self, template: ConfigurationTemplate, context: ConfigurationContext):
        """Initialize utility components based on template."""
        # Initialize cleanup manager
        if not self.cleanup_manager:
            self.cleanup_manager = ResourceCleanupManager(
                logger=self.logger,
                cleanup_priority=template.cleanup_priority
            )
        
        # Register environment manager for cleanup
        if self.environment_manager:
            self.cleanup_manager.register_resource(
                ResourceTracker(
                    resource_id=f"env_manager_{context.context_id}",
                    resource_type=ResourceType.SYSTEM_RESOURCES,
                    resource_data=self.environment_manager,
                    creation_time=time.time(),
                    cleanup_callback=self.environment_manager.cleanup,
                    cleanup_priority=1
                )
            )
        
        # Register async coordinator for cleanup
        if self.async_coordinator:
            self.cleanup_manager.register_resource(
                ResourceTracker(
                    resource_id=f"async_coord_{context.context_id}",
                    resource_type=ResourceType.ASYNC_TASKS,
                    resource_data=self.async_coordinator,
                    creation_time=time.time(),
                    cleanup_callback=self._cleanup_async_coordinator,
                    cleanup_priority=2
                )
            )
        
        # Initialize configuration validation suite
        if self.config_validation_enabled:
            self.validation_suite = ConfigurationValidationSuite(
                environment_manager=self.environment_manager,
                mock_factory=self.mock_factory,
                async_coordinator=self.async_coordinator,
                performance_helper=self.performance_helper
            )
    
    async def _cleanup_async_coordinator(self):
        """Clean up async coordinator resources."""
        if self.async_coordinator:
            # Cancel all active sessions
            for session_id in list(self.async_coordinator.active_sessions.keys()):
                try:
                    await self.async_coordinator.cancel_session(session_id)
                except Exception as e:
                    self.logger.warning(f"Failed to cancel session {session_id}: {e}")
    
    def get_configuration_context(self, context_id: str) -> Optional[ConfigurationContext]:
        """Get configuration context by ID."""
        return self.active_contexts.get(context_id)
    
    def validate_configuration(self, context_id: str) -> List[str]:
        """Validate configuration for context."""
        if not self.validation_suite:
            return ["Configuration validation suite not initialized"]
        
        context = self.active_contexts.get(context_id)
        if not context:
            return [f"Configuration context {context_id} not found"]
        
        return self.validation_suite.validate_configuration(context)
    
    def get_integrated_test_environment(self, context_id: str) -> Dict[str, Any]:
        """
        Get fully integrated test environment with all components.
        
        Args:
            context_id: Configuration context ID
            
        Returns:
            Dictionary containing all configured test components
        """
        context = self.active_contexts.get(context_id)
        if not context:
            raise ValueError(f"Configuration context {context_id} not found")
        
        return {
            'context_id': context_id,
            'context': context,
            'environment_manager': self.environment_manager,
            'mock_factory': self.mock_factory,
            'async_coordinator': self.async_coordinator,
            'performance_helper': self.performance_helper,
            'cleanup_manager': self.cleanup_manager,
            'validation_suite': self.validation_suite,
            'config_helper': self,
            'working_dir': context.modified_config.get('environment', {}).get('working_dir'),
            'mock_system': context.modified_config.get('mock_system', {}),
            'logger': self.logger
        }
    
    def register_custom_template(self, template: ConfigurationTemplate):
        """Register custom configuration template."""
        self.template_registry[template.scenario_type] = template
        self.logger.info(f"Registered custom template: {template.name}")
    
    def cleanup_configuration(self, context_id: str, force: bool = False):
        """
        Clean up configuration context and all associated resources.
        
        Args:
            context_id: Configuration context ID to clean up
            force: Whether to force cleanup even if errors occur
        """
        context = self.active_contexts.get(context_id)
        if not context:
            self.logger.warning(f"Configuration context {context_id} not found for cleanup")
            return
        
        cleanup_start = time.time()
        
        try:
            # Restore environment variables
            for env_var, original_value in context.environment_vars.items():
                if original_value is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = original_value
            
            # Execute cleanup callbacks
            for callback in context.cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        # Handle async cleanup callbacks
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(callback())
                        finally:
                            loop.close()
                    else:
                        callback()
                except Exception as e:
                    if not force:
                        raise
                    self.logger.warning(f"Cleanup callback failed: {e}")
            
            # Use cleanup manager if available
            if self.cleanup_manager:
                context_resources = [
                    tracker for tracker in self.cleanup_manager.resource_trackers 
                    if context_id in tracker.resource_id
                ]
                for tracker in context_resources:
                    self.cleanup_manager.cleanup_resource(tracker.resource_id, force=force)
            
            # Clean up utility components
            if self.environment_manager and hasattr(self.environment_manager, 'cleanup'):
                self.environment_manager.cleanup()
            
            # Remove context
            del self.active_contexts[context_id]
            
            cleanup_duration = time.time() - cleanup_start
            self.logger.info(f"Configuration cleanup completed in {cleanup_duration:.3f}s for context: {context_id}")
            
        except Exception as e:
            if force:
                self.logger.error(f"Forced cleanup encountered errors: {e}")
                # Still remove context
                self.active_contexts.pop(context_id, None)
            else:
                self.logger.error(f"Configuration cleanup failed: {e}")
                raise
    
    def cleanup_all_configurations(self, force: bool = False):
        """Clean up all active configurations."""
        context_ids = list(self.active_contexts.keys())
        
        for context_id in context_ids:
            try:
                self.cleanup_configuration(context_id, force=force)
            except Exception as e:
                self.logger.error(f"Failed to cleanup context {context_id}: {e}")
                if not force:
                    raise
    
    def get_configuration_statistics(self) -> Dict[str, Any]:
        """Get statistics about configuration usage."""
        active_contexts = len(self.active_contexts)
        total_resources = 0
        
        if self.cleanup_manager:
            total_resources = len(self.cleanup_manager.resource_trackers)
        
        return {
            'active_contexts': active_contexts,
            'total_resources_tracked': total_resources,
            'available_templates': len(self.template_registry),
            'validation_enabled': self.config_validation_enabled,
            'validation_errors': len(self.validation_errors)
        }


# =====================================================================
# RESOURCE CLEANUP MANAGER
# =====================================================================

class ResourceCleanupManager:
    """
    Centralized resource cleanup management across all test utilities.
    
    Manages automated resource cleanup, prevents memory leaks, handles process
    cleanup, and coordinates cleanup across async operations and all test utilities.
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 cleanup_priority: int = 5,
                 cleanup_timeout: float = 30.0):
        """Initialize resource cleanup manager."""
        self.logger = logger or logging.getLogger(f"cleanup_manager_{id(self)}")
        self.default_cleanup_priority = cleanup_priority
        self.default_cleanup_timeout = cleanup_timeout
        
        # Resource tracking
        self.resource_trackers: Dict[str, ResourceTracker] = {}
        self.cleanup_order: List[str] = []  # Ordered by priority
        
        # Cleanup state
        self.cleanup_in_progress = False
        self.cleanup_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'resources_registered': 0,
            'resources_cleaned': 0,
            'cleanup_failures': 0,
            'total_cleanup_time': 0.0,
            'memory_leaks_detected': 0
        }
        
        # Process and memory tracking
        self.initial_memory = self._get_memory_usage()
        self.tracked_processes: Set[int] = set()
        self.tracked_threads: Set[threading.Thread] = set()
        self.tracked_async_tasks: Set[asyncio.Task] = set()
        
        # Register cleanup at exit
        atexit.register(self.emergency_cleanup)
        
        # Set up signal handlers for graceful cleanup
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle cleanup on signals."""
        self.logger.warning(f"Received signal {signum}, performing cleanup...")
        self.emergency_cleanup()
    
    def register_resource(self, tracker: ResourceTracker):
        """
        Register resource for cleanup tracking.
        
        Args:
            tracker: ResourceTracker containing resource information
        """
        self.resource_trackers[tracker.resource_id] = tracker
        self._update_cleanup_order()
        self.stats['resources_registered'] += 1
        
        self.logger.debug(f"Registered resource: {tracker.resource_id} ({tracker.resource_type.value})")
        
        # Special handling for different resource types
        if tracker.resource_type == ResourceType.PROCESSES:
            if hasattr(tracker.resource_data, 'pid'):
                self.tracked_processes.add(tracker.resource_data.pid)
        
        elif tracker.resource_type == ResourceType.THREADS:
            if isinstance(tracker.resource_data, threading.Thread):
                self.tracked_threads.add(tracker.resource_data)
        
        elif tracker.resource_type == ResourceType.ASYNC_TASKS:
            if isinstance(tracker.resource_data, asyncio.Task):
                self.tracked_async_tasks.add(tracker.resource_data)
    
    def _update_cleanup_order(self):
        """Update cleanup order based on priorities."""
        self.cleanup_order = sorted(
            self.resource_trackers.keys(),
            key=lambda rid: self.resource_trackers[rid].cleanup_priority,
            reverse=True  # Higher priority first
        )
    
    def register_temporary_file(self, file_path: Path, cleanup_priority: int = 3) -> str:
        """
        Register temporary file for cleanup.
        
        Args:
            file_path: Path to temporary file
            cleanup_priority: Cleanup priority (higher = cleaned up first)
            
        Returns:
            Resource ID for tracking
        """
        resource_id = f"temp_file_{int(time.time() * 1000)}_{id(file_path)}"
        
        def cleanup_file():
            try:
                if file_path.exists():
                    file_path.unlink()
                    self.logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup file {file_path}: {e}")
                raise
        
        tracker = ResourceTracker(
            resource_id=resource_id,
            resource_type=ResourceType.TEMPORARY_FILES,
            resource_data=file_path,
            creation_time=time.time(),
            cleanup_callback=cleanup_file,
            cleanup_priority=cleanup_priority
        )
        
        self.register_resource(tracker)
        return resource_id
    
    def register_temporary_directory(self, dir_path: Path, cleanup_priority: int = 2) -> str:
        """
        Register temporary directory for cleanup.
        
        Args:
            dir_path: Path to temporary directory
            cleanup_priority: Cleanup priority
            
        Returns:
            Resource ID for tracking
        """
        resource_id = f"temp_dir_{int(time.time() * 1000)}_{id(dir_path)}"
        
        def cleanup_directory():
            try:
                if dir_path.exists() and dir_path.is_dir():
                    shutil.rmtree(dir_path)
                    self.logger.debug(f"Cleaned up temporary directory: {dir_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup directory {dir_path}: {e}")
                raise
        
        tracker = ResourceTracker(
            resource_id=resource_id,
            resource_type=ResourceType.TEMPORARY_DIRECTORIES,
            resource_data=dir_path,
            creation_time=time.time(),
            cleanup_callback=cleanup_directory,
            cleanup_priority=cleanup_priority
        )
        
        self.register_resource(tracker)
        return resource_id
    
    def register_process(self, process: subprocess.Popen, cleanup_priority: int = 4) -> str:
        """
        Register process for cleanup.
        
        Args:
            process: Process to track for cleanup
            cleanup_priority: Cleanup priority
            
        Returns:
            Resource ID for tracking
        """
        resource_id = f"process_{process.pid}_{int(time.time() * 1000)}"
        
        def cleanup_process():
            try:
                if process.poll() is None:  # Process still running
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    self.logger.debug(f"Cleaned up process: {process.pid}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup process {process.pid}: {e}")
                raise
        
        tracker = ResourceTracker(
            resource_id=resource_id,
            resource_type=ResourceType.PROCESSES,
            resource_data=process,
            creation_time=time.time(),
            cleanup_callback=cleanup_process,
            cleanup_priority=cleanup_priority
        )
        
        self.register_resource(tracker)
        self.tracked_processes.add(process.pid)
        return resource_id
    
    def register_async_task(self, task: asyncio.Task, cleanup_priority: int = 3) -> str:
        """
        Register async task for cleanup.
        
        Args:
            task: Async task to track for cleanup
            cleanup_priority: Cleanup priority
            
        Returns:
            Resource ID for tracking
        """
        resource_id = f"async_task_{id(task)}_{int(time.time() * 1000)}"
        
        async def cleanup_task():
            try:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    self.logger.debug(f"Cleaned up async task: {resource_id}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup async task {resource_id}: {e}")
                raise
        
        tracker = ResourceTracker(
            resource_id=resource_id,
            resource_type=ResourceType.ASYNC_TASKS,
            resource_data=task,
            creation_time=time.time(),
            cleanup_callback=cleanup_task,
            cleanup_priority=cleanup_priority
        )
        
        self.register_resource(tracker)
        self.tracked_async_tasks.add(task)
        return resource_id
    
    def cleanup_resource(self, resource_id: str, force: bool = False):
        """
        Clean up specific resource.
        
        Args:
            resource_id: ID of resource to clean up
            force: Whether to force cleanup even if errors occur
        """
        tracker = self.resource_trackers.get(resource_id)
        if not tracker:
            self.logger.warning(f"Resource {resource_id} not found for cleanup")
            return
        
        try:
            if tracker.cleanup_callback:
                if asyncio.iscoroutinefunction(tracker.cleanup_callback):
                    # Handle async cleanup
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Create task for async cleanup
                            loop.create_task(tracker.cleanup_callback())
                        else:
                            loop.run_until_complete(tracker.cleanup_callback())
                    except RuntimeError:
                        # No event loop, create new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(tracker.cleanup_callback())
                        finally:
                            loop.close()
                else:
                    # Synchronous cleanup
                    tracker.cleanup_callback()
            
            # Remove from tracking
            del self.resource_trackers[resource_id]
            self._update_cleanup_order()
            self.stats['resources_cleaned'] += 1
            
        except Exception as e:
            self.stats['cleanup_failures'] += 1
            if force:
                self.logger.warning(f"Forced cleanup of {resource_id} failed: {e}")
                # Remove from tracking anyway
                self.resource_trackers.pop(resource_id, None)
                self._update_cleanup_order()
            else:
                self.logger.error(f"Cleanup of {resource_id} failed: {e}")
                raise
    
    def cleanup_all_resources(self, force: bool = False):
        """
        Clean up all tracked resources.
        
        Args:
            force: Whether to force cleanup even if errors occur
        """
        if self.cleanup_in_progress:
            self.logger.warning("Cleanup already in progress, skipping")
            return
        
        with self.cleanup_lock:
            self.cleanup_in_progress = True
            cleanup_start = time.time()
            
            try:
                self.logger.info(f"Starting cleanup of {len(self.resource_trackers)} resources...")
                
                # Clean up resources in priority order
                for resource_id in self.cleanup_order.copy():
                    if resource_id in self.resource_trackers:
                        try:
                            self.cleanup_resource(resource_id, force=force)
                        except Exception as e:
                            if not force:
                                raise
                            self.logger.warning(f"Forced cleanup continued after error: {e}")
                
                # Perform memory leak detection
                self._detect_memory_leaks()
                
                # Final garbage collection
                gc.collect()
                
                cleanup_duration = time.time() - cleanup_start
                self.stats['total_cleanup_time'] += cleanup_duration
                
                self.logger.info(f"Resource cleanup completed in {cleanup_duration:.3f}s")
                
            finally:
                self.cleanup_in_progress = False
    
    def _detect_memory_leaks(self):
        """Detect potential memory leaks."""
        current_memory = self._get_memory_usage()
        memory_increase = current_memory - self.initial_memory
        
        # Alert if memory increased by more than 100MB
        if memory_increase > 100:
            self.stats['memory_leaks_detected'] += 1
            self.logger.warning(f"Potential memory leak detected: {memory_increase:.1f}MB increase")
            
            # Get memory details
            memory_details = self._get_detailed_memory_info()
            self.logger.warning(f"Memory details: {memory_details}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_detailed_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'gc_objects': len(gc.get_objects()),
                'tracked_resources': len(self.resource_trackers),
                'active_threads': threading.active_count()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def emergency_cleanup(self):
        """Perform emergency cleanup (called at exit)."""
        if self.resource_trackers:
            self.logger.warning("Performing emergency cleanup...")
            self.cleanup_all_resources(force=True)
    
    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """Get cleanup statistics."""
        return {
            **self.stats,
            'active_resources': len(self.resource_trackers),
            'current_memory_mb': self._get_memory_usage(),
            'tracked_processes': len(self.tracked_processes),
            'tracked_threads': len(self.tracked_threads),
            'tracked_async_tasks': len(self.tracked_async_tasks)
        }


# =====================================================================
# CONFIGURATION VALIDATION SUITE
# =====================================================================

class ConfigurationValidationSuite:
    """Configuration validation and testing utilities."""
    
    def __init__(self,
                 environment_manager: Optional[TestEnvironmentManager] = None,
                 mock_factory: Optional[MockSystemFactory] = None,
                 async_coordinator: Optional[AsyncTestCoordinator] = None,
                 performance_helper: Optional[PerformanceAssertionHelper] = None):
        """Initialize configuration validation suite."""
        self.environment_manager = environment_manager
        self.mock_factory = mock_factory
        self.async_coordinator = async_coordinator
        self.performance_helper = performance_helper
        self.logger = logging.getLogger(f"config_validation_{id(self)}")
    
    def validate_configuration(self, context: ConfigurationContext) -> List[str]:
        """
        Validate configuration context.
        
        Args:
            context: Configuration context to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate environment setup
        if self.environment_manager:
            env_errors = self._validate_environment()
            errors.extend(env_errors)
        
        # Validate mock system
        if self.mock_factory:
            mock_errors = self._validate_mock_system(context)
            errors.extend(mock_errors)
        
        # Validate async coordinator
        if self.async_coordinator:
            async_errors = self._validate_async_setup()
            errors.extend(async_errors)
        
        # Validate performance setup
        if self.performance_helper:
            perf_errors = self._validate_performance_setup()
            errors.extend(perf_errors)
        
        return errors
    
    def _validate_environment(self) -> List[str]:
        """Validate environment manager setup."""
        errors = []
        
        try:
            health = self.environment_manager.check_system_health()
            
            # Check memory usage
            if health['memory_usage_mb'] > 1000:
                errors.append(f"High memory usage: {health['memory_usage_mb']:.1f}MB")
            
            # Check working directory
            if not self.environment_manager.spec.working_dir.exists():
                errors.append("Working directory does not exist")
            
        except Exception as e:
            errors.append(f"Environment validation failed: {e}")
        
        return errors
    
    def _validate_mock_system(self, context: ConfigurationContext) -> List[str]:
        """Validate mock system setup."""
        errors = []
        
        try:
            mock_system = context.modified_config.get('mock_system', {})
            
            if not mock_system:
                return []  # No mocks configured
            
            # Validate mock components
            for component_name, mock_obj in mock_system.items():
                if not hasattr(mock_obj, '__call__'):
                    errors.append(f"Mock {component_name} is not callable")
            
        except Exception as e:
            errors.append(f"Mock system validation failed: {e}")
        
        return errors
    
    def _validate_async_setup(self) -> List[str]:
        """Validate async coordinator setup."""
        errors = []
        
        try:
            if not hasattr(self.async_coordinator, 'create_session'):
                errors.append("Async coordinator missing create_session method")
            
        except Exception as e:
            errors.append(f"Async coordinator validation failed: {e}")
        
        return errors
    
    def _validate_performance_setup(self) -> List[str]:
        """Validate performance helper setup."""
        errors = []
        
        try:
            if not hasattr(self.performance_helper, 'register_threshold'):
                errors.append("Performance helper missing register_threshold method")
            
        except Exception as e:
            errors.append(f"Performance helper validation failed: {e}")
        
        return errors


# =====================================================================
# ENVIRONMENT ISOLATION MANAGER
# =====================================================================

class EnvironmentIsolationManager:
    """Manager for test environment isolation and sandboxing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize environment isolation manager."""
        self.logger = logger or logging.getLogger(f"isolation_manager_{id(self)}")
        self.original_environment = os.environ.copy()
        self.original_sys_path = sys.path.copy()
        self.original_cwd = os.getcwd()
        self.isolation_contexts: Dict[str, Dict[str, Any]] = {}
    
    @contextmanager
    def isolated_environment(self, 
                           environment_vars: Optional[Dict[str, str]] = None,
                           working_directory: Optional[Path] = None,
                           sys_path_additions: Optional[List[str]] = None):
        """
        Context manager for isolated environment.
        
        Args:
            environment_vars: Environment variables to set
            working_directory: Working directory to change to
            sys_path_additions: Additional paths to add to sys.path
        """
        isolation_id = f"isolation_{int(time.time() * 1000)}"
        
        # Store original state
        original_env = os.environ.copy()
        original_path = sys.path.copy()
        original_cwd = os.getcwd()
        
        try:
            # Apply environment variables
            if environment_vars:
                os.environ.update(environment_vars)
            
            # Change working directory
            if working_directory:
                os.chdir(working_directory)
            
            # Add to sys.path
            if sys_path_additions:
                for path in sys_path_additions:
                    if path not in sys.path:
                        sys.path.insert(0, path)
            
            self.isolation_contexts[isolation_id] = {
                'environment_vars': environment_vars or {},
                'working_directory': working_directory,
                'sys_path_additions': sys_path_additions or []
            }
            
            yield isolation_id
            
        finally:
            # Restore original state
            os.environ.clear()
            os.environ.update(original_env)
            sys.path[:] = original_path
            os.chdir(original_cwd)
            
            self.isolation_contexts.pop(isolation_id, None)


# =====================================================================
# PYTEST FIXTURES AND INTEGRATION
# =====================================================================

@pytest.fixture
def configuration_test_helper():
    """Provide ConfigurationTestHelper for tests."""
    helper = ConfigurationTestHelper()
    yield helper
    helper.cleanup_all_configurations(force=True)


@pytest.fixture
def resource_cleanup_manager():
    """Provide ResourceCleanupManager for tests."""
    manager = ResourceCleanupManager()
    yield manager
    manager.cleanup_all_resources(force=True)


@pytest.fixture
def standard_unit_test_config(configuration_test_helper):
    """Standard unit test configuration."""
    context_id = configuration_test_helper.create_test_configuration(
        TestScenarioType.UNIT_TEST
    )
    yield configuration_test_helper.get_integrated_test_environment(context_id)
    configuration_test_helper.cleanup_configuration(context_id)


@pytest.fixture
def standard_integration_test_config(configuration_test_helper):
    """Standard integration test configuration."""
    context_id = configuration_test_helper.create_test_configuration(
        TestScenarioType.INTEGRATION_TEST
    )
    yield configuration_test_helper.get_integrated_test_environment(context_id)
    configuration_test_helper.cleanup_configuration(context_id)


@pytest.fixture
def standard_performance_test_config(configuration_test_helper):
    """Standard performance test configuration."""
    context_id = configuration_test_helper.create_test_configuration(
        TestScenarioType.PERFORMANCE_TEST
    )
    yield configuration_test_helper.get_integrated_test_environment(context_id)
    configuration_test_helper.cleanup_configuration(context_id)


@pytest.fixture
def biomedical_test_config(configuration_test_helper):
    """Biomedical test configuration."""
    context_id = configuration_test_helper.create_test_configuration(
        TestScenarioType.BIOMEDICAL_TEST
    )
    yield configuration_test_helper.get_integrated_test_environment(context_id)
    configuration_test_helper.cleanup_configuration(context_id)


@pytest.fixture
def environment_isolation_manager():
    """Provide EnvironmentIsolationManager for tests."""
    return EnvironmentIsolationManager()


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def create_complete_test_environment(scenario_type: TestScenarioType,
                                   custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create complete test environment with all utilities integrated.
    
    Args:
        scenario_type: Type of test scenario
        custom_overrides: Optional configuration overrides
        
    Returns:
        Dictionary containing all test utilities and configuration
    """
    helper = ConfigurationTestHelper()
    
    try:
        context_id = helper.create_test_configuration(scenario_type, custom_overrides)
        return helper.get_integrated_test_environment(context_id)
    except Exception:
        helper.cleanup_all_configurations(force=True)
        raise


@asynccontextmanager
async def managed_test_environment(scenario_type: TestScenarioType,
                                 custom_overrides: Optional[Dict[str, Any]] = None):
    """
    Async context manager for managed test environment.
    
    Args:
        scenario_type: Type of test scenario  
        custom_overrides: Optional configuration overrides
    """
    helper = ConfigurationTestHelper()
    context_id = None
    
    try:
        context_id = helper.create_test_configuration(scenario_type, custom_overrides)
        test_env = helper.get_integrated_test_environment(context_id)
        
        yield test_env
        
    finally:
        if context_id:
            helper.cleanup_configuration(context_id, force=True)


def validate_test_configuration(test_environment: Dict[str, Any]) -> List[str]:
    """
    Validate test configuration environment.
    
    Args:
        test_environment: Test environment from create_complete_test_environment
        
    Returns:
        List of validation errors (empty if valid)
    """
    validation_suite = test_environment.get('validation_suite')
    context = test_environment.get('context')
    
    if not validation_suite or not context:
        return ["Validation suite or context not available"]
    
    return validation_suite.validate_configuration(context)


# Make key classes available at module level
__all__ = [
    'ConfigurationTestHelper',
    'ResourceCleanupManager', 
    'ConfigurationValidationSuite',
    'EnvironmentIsolationManager',
    'TestScenarioType',
    'ConfigurationScope',
    'ResourceType',
    'ConfigurationTemplate',
    'ResourceTracker',
    'ConfigurationContext',
    'create_complete_test_environment',
    'managed_test_environment',
    'validate_test_configuration'
]