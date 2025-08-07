#!/usr/bin/env python3
"""
Async Test Utilities and Coordination for Clinical Metabolomics Oracle Testing.

This module provides comprehensive async testing utilities that integrate with the existing
test infrastructure to standardize async testing patterns and provide robust coordination
for complex async test scenarios.

Key Components:
1. AsyncTestCoordinator: Centralized async test execution management and coordination
2. Async context managers: Environment setup, resource management, error injection, monitoring
3. Async utilities: Data generation, operation batching, result aggregation, retry mechanisms
4. Seamless integration with existing TestEnvironmentManager, MockSystemFactory, and fixtures

Features:
- Concurrency control and resource management for async operations
- Timeout management and async operation sequencing
- Dependency management and coordination between async operations  
- Async error injection and recovery testing capabilities
- Async performance monitoring during test execution
- Integration with pytest-asyncio and existing async testing framework

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import pytest_asyncio
import asyncio
import time
import logging
import json
import threading
import functools
import warnings
import gc
import random
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Type, AsyncGenerator, Awaitable
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from enum import Enum
import weakref
import psutil

# Import from existing test infrastructure
from test_utilities import (
    TestEnvironmentManager, MockSystemFactory, SystemComponent, 
    TestComplexity, MockBehavior, EnvironmentSpec, MockSpec, MemoryMonitor
)
from performance_test_utilities import (
    PerformanceAssertionHelper, PerformanceBenchmarkSuite, 
    AdvancedResourceMonitor, PerformanceThreshold
)


# =====================================================================
# ASYNC TEST COORDINATION ENUMS AND DATA CLASSES
# =====================================================================

class AsyncTestState(Enum):
    """States for async test execution."""
    INITIALIZING = "initializing"
    RUNNING = "running" 
    WAITING_FOR_DEPENDENCIES = "waiting_for_dependencies"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ConcurrencyPolicy(Enum):
    """Concurrency control policies."""
    UNLIMITED = "unlimited"
    LIMITED = "limited"
    SEQUENTIAL = "sequential"
    BATCH_PROCESSING = "batch_processing"
    ADAPTIVE = "adaptive"


@dataclass
class AsyncOperationSpec:
    """Specification for async operation execution."""
    operation_id: str
    operation_func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class AsyncOperationResult:
    """Result of async operation execution."""
    operation_id: str
    state: AsyncTestState
    start_time: float
    end_time: Optional[float] = None
    result: Any = None
    exception: Optional[Exception] = None
    retry_count: int = 0
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def succeeded(self) -> bool:
        """Check if operation succeeded."""
        return self.state == AsyncTestState.COMPLETED and self.exception is None
    
    @property
    def failed(self) -> bool:
        """Check if operation failed."""
        return self.state == AsyncTestState.FAILED or self.exception is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AsyncTestSession:
    """Session data for async test coordination."""
    session_id: str
    start_time: float
    operations: Dict[str, AsyncOperationSpec] = field(default_factory=dict)
    results: Dict[str, AsyncOperationResult] = field(default_factory=dict)
    active_tasks: Dict[str, asyncio.Task] = field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    concurrency_semaphore: Optional[asyncio.Semaphore] = None
    cleanup_callbacks: List[Callable] = field(default_factory=list)
    resource_monitor: Optional[AdvancedResourceMonitor] = None
    performance_helper: Optional[PerformanceAssertionHelper] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# ASYNC TEST COORDINATOR
# =====================================================================

class AsyncTestCoordinator:
    """
    Centralized coordinator for async test execution with concurrency control,
    dependency management, timeout handling, and resource cleanup coordination.
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 default_timeout: float = 30.0,
                 max_concurrent_operations: int = 10,
                 concurrency_policy: ConcurrencyPolicy = ConcurrencyPolicy.LIMITED):
        """Initialize async test coordinator."""
        self.logger = logger or logging.getLogger(f"async_coordinator_{id(self)}")
        self.default_timeout = default_timeout
        self.max_concurrent_operations = max_concurrent_operations
        self.concurrency_policy = concurrency_policy
        
        # Session management
        self.active_sessions: Dict[str, AsyncTestSession] = {}
        self.session_counter = 0
        
        # Global coordination
        self.global_shutdown_event = asyncio.Event()
        self.coordination_lock = asyncio.Lock()
        
        # Statistics and monitoring
        self.stats = {
            'total_sessions': 0,
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'cancelled_operations': 0,
            'total_retries': 0
        }
        
    async def create_session(self, 
                           session_id: Optional[str] = None,
                           enable_resource_monitoring: bool = True,
                           enable_performance_monitoring: bool = True) -> str:
        """
        Create new async test session.
        
        Args:
            session_id: Optional custom session ID
            enable_resource_monitoring: Whether to enable resource monitoring
            enable_performance_monitoring: Whether to enable performance monitoring
            
        Returns:
            Session ID for the created session
        """
        async with self.coordination_lock:
            if session_id is None:
                self.session_counter += 1
                session_id = f"async_session_{self.session_counter}"
            
            if session_id in self.active_sessions:
                raise ValueError(f"Session {session_id} already exists")
            
            # Create concurrency semaphore based on policy
            semaphore = None
            if self.concurrency_policy == ConcurrencyPolicy.LIMITED:
                semaphore = asyncio.Semaphore(self.max_concurrent_operations)
            elif self.concurrency_policy == ConcurrencyPolicy.SEQUENTIAL:
                semaphore = asyncio.Semaphore(1)
            
            # Initialize monitoring
            resource_monitor = None
            performance_helper = None
            
            if enable_resource_monitoring:
                resource_monitor = AdvancedResourceMonitor(sampling_interval=0.5)
                
            if enable_performance_monitoring:
                performance_helper = PerformanceAssertionHelper(self.logger)
                performance_helper.establish_memory_baseline()
            
            # Create session
            session = AsyncTestSession(
                session_id=session_id,
                start_time=time.time(),
                concurrency_semaphore=semaphore,
                resource_monitor=resource_monitor,
                performance_helper=performance_helper
            )
            
            self.active_sessions[session_id] = session
            self.stats['total_sessions'] += 1
            
            # Start resource monitoring if enabled
            if resource_monitor:
                resource_monitor.start_monitoring()
            
            self.logger.info(f"Created async test session: {session_id}")
            return session_id
    
    async def add_operation(self,
                          session_id: str,
                          operation_spec: AsyncOperationSpec) -> None:
        """Add operation to session for execution."""
        session = self._get_session(session_id)
        
        if operation_spec.operation_id in session.operations:
            raise ValueError(f"Operation {operation_spec.operation_id} already exists in session {session_id}")
        
        session.operations[operation_spec.operation_id] = operation_spec
        session.dependency_graph[operation_spec.operation_id] = operation_spec.dependencies.copy()
        
        self.logger.debug(f"Added operation {operation_spec.operation_id} to session {session_id}")
    
    async def add_operations_batch(self,
                                 session_id: str,
                                 operation_specs: List[AsyncOperationSpec]) -> None:
        """Add multiple operations to session."""
        for spec in operation_specs:
            await self.add_operation(session_id, spec)
    
    async def execute_session(self, 
                            session_id: str,
                            fail_on_first_error: bool = False,
                            progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> Dict[str, AsyncOperationResult]:
        """
        Execute all operations in session with dependency resolution.
        
        Args:
            session_id: Session to execute
            fail_on_first_error: Whether to stop execution on first error
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping operation IDs to results
        """
        session = self._get_session(session_id)
        
        if not session.operations:
            self.logger.warning(f"No operations to execute in session {session_id}")
            return {}
        
        self.logger.info(f"Starting execution of {len(session.operations)} operations in session {session_id}")
        
        try:
            # Validate dependency graph
            self._validate_dependency_graph(session)
            
            # Execute operations with dependency resolution
            await self._execute_operations_with_dependencies(
                session, fail_on_first_error, progress_callback
            )
            
        except Exception as e:
            self.logger.error(f"Session execution failed: {e}")
            await self._cancel_pending_operations(session)
            raise
        
        # Update statistics
        for result in session.results.values():
            if result.succeeded:
                self.stats['successful_operations'] += 1
            elif result.failed:
                self.stats['failed_operations'] += 1
            elif result.state == AsyncTestState.CANCELLED:
                self.stats['cancelled_operations'] += 1
            
            self.stats['total_retries'] += result.retry_count
        
        self.stats['total_operations'] += len(session.results)
        
        self.logger.info(f"Session {session_id} execution completed with {len(session.results)} results")
        return session.results.copy()
    
    async def execute_single_operation(self,
                                     session_id: str,
                                     operation_id: str,
                                     timeout_override: Optional[float] = None) -> AsyncOperationResult:
        """Execute single operation by ID."""
        session = self._get_session(session_id)
        
        if operation_id not in session.operations:
            raise ValueError(f"Operation {operation_id} not found in session {session_id}")
        
        operation_spec = session.operations[operation_id]
        
        # Check dependencies
        if operation_spec.dependencies:
            unmet_deps = [dep for dep in operation_spec.dependencies if dep not in session.results or not session.results[dep].succeeded]
            if unmet_deps:
                raise ValueError(f"Operation {operation_id} has unmet dependencies: {unmet_deps}")
        
        # Execute operation
        timeout = timeout_override or operation_spec.timeout_seconds or self.default_timeout
        result = await self._execute_single_operation(session, operation_spec, timeout)
        
        session.results[operation_id] = result
        return result
    
    async def wait_for_completion(self,
                                session_id: str,
                                timeout: Optional[float] = None,
                                check_interval: float = 0.1) -> bool:
        """Wait for all operations in session to complete."""
        session = self._get_session(session_id)
        start_time = time.time()
        
        while True:
            # Check if all operations are complete
            if all(task.done() for task in session.active_tasks.values()):
                return True
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Wait timeout reached for session {session_id}")
                return False
            
            # Check global shutdown
            if self.global_shutdown_event.is_set():
                return False
            
            await asyncio.sleep(check_interval)
    
    async def cancel_session(self, session_id: str, reason: str = "User requested") -> None:
        """Cancel all operations in session."""
        session = self._get_session(session_id)
        
        self.logger.info(f"Cancelling session {session_id}: {reason}")
        await self._cancel_pending_operations(session)
        
        # Update results for cancelled operations
        for op_id in session.operations:
            if op_id not in session.results:
                session.results[op_id] = AsyncOperationResult(
                    operation_id=op_id,
                    state=AsyncTestState.CANCELLED,
                    start_time=time.time()
                )
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        session = self._get_session(session_id)
        
        try:
            # Cancel any remaining tasks
            await self._cancel_pending_operations(session)
            
            # Run cleanup callbacks
            for callback in session.cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    self.logger.warning(f"Cleanup callback failed: {e}")
            
            # Stop resource monitoring
            if session.resource_monitor:
                session.resource_monitor.stop_monitoring()
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            self.logger.info(f"Session {session_id} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Session cleanup failed for {session_id}: {e}")
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of session."""
        session = self._get_session(session_id)
        
        total_ops = len(session.operations)
        completed_ops = len([r for r in session.results.values() if r.state in [AsyncTestState.COMPLETED, AsyncTestState.FAILED]])
        active_ops = len(session.active_tasks)
        
        status = {
            'session_id': session_id,
            'start_time': session.start_time,
            'uptime_seconds': time.time() - session.start_time,
            'total_operations': total_ops,
            'completed_operations': completed_ops,
            'active_operations': active_ops,
            'pending_operations': total_ops - completed_ops - active_ops,
            'success_rate': (len([r for r in session.results.values() if r.succeeded]) / max(completed_ops, 1)) * 100,
            'operations_status': {op_id: result.state.value for op_id, result in session.results.items()}
        }
        
        # Add resource monitoring data if available
        if session.resource_monitor:
            status['resource_summary'] = session.resource_monitor.get_resource_summary()
            status['alert_summary'] = session.resource_monitor.get_alert_summary()
        
        # Add performance data if available
        if session.performance_helper:
            status['performance_summary'] = session.performance_helper.get_assertion_summary()
        
        return status
    
    # Private helper methods
    
    def _get_session(self, session_id: str) -> AsyncTestSession:
        """Get session by ID or raise error."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.active_sessions[session_id]
    
    def _validate_dependency_graph(self, session: AsyncTestSession) -> None:
        """Validate dependency graph for cycles and missing dependencies."""
        # Check for missing dependencies
        all_operation_ids = set(session.operations.keys())
        for op_id, deps in session.dependency_graph.items():
            missing_deps = set(deps) - all_operation_ids
            if missing_deps:
                raise ValueError(f"Operation {op_id} has missing dependencies: {missing_deps}")
        
        # Check for cycles using topological sort attempt
        try:
            self._topological_sort(session.dependency_graph)
        except ValueError as e:
            raise ValueError(f"Dependency cycle detected: {e}")
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort to detect cycles."""
        in_degree = {node: 0 for node in graph}
        
        # Calculate in-degrees
        for node in graph:
            for dep in graph[node]:
                in_degree[dep] += 1
        
        # Find nodes with no dependencies
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Remove edges from current node
            for dep in graph.get(current, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
        
        if len(result) != len(graph):
            raise ValueError("Cycle detected in dependency graph")
        
        return result
    
    async def _execute_operations_with_dependencies(self,
                                                   session: AsyncTestSession,
                                                   fail_on_first_error: bool,
                                                   progress_callback: Optional[Callable[[str, float], Awaitable[None]]]) -> None:
        """Execute operations respecting dependencies."""
        # Get execution order
        execution_order = self._topological_sort(session.dependency_graph)
        
        # Group operations by dependency level for parallel execution
        execution_groups = self._group_operations_by_dependency_level(session, execution_order)
        
        total_groups = len(execution_groups)
        
        for group_index, operation_ids in enumerate(execution_groups):
            self.logger.debug(f"Executing group {group_index + 1}/{total_groups}: {operation_ids}")
            
            # Execute operations in current group concurrently
            group_tasks = {}
            
            for op_id in operation_ids:
                operation_spec = session.operations[op_id]
                timeout = operation_spec.timeout_seconds or self.default_timeout
                
                task = asyncio.create_task(
                    self._execute_single_operation(session, operation_spec, timeout)
                )
                group_tasks[op_id] = task
                session.active_tasks[op_id] = task
            
            # Wait for group completion
            group_results = await asyncio.gather(*group_tasks.values(), return_exceptions=True)
            
            # Process results
            for op_id, result in zip(operation_ids, group_results):
                if isinstance(result, Exception):
                    session.results[op_id] = AsyncOperationResult(
                        operation_id=op_id,
                        state=AsyncTestState.FAILED,
                        start_time=time.time(),
                        exception=result
                    )
                else:
                    session.results[op_id] = result
                
                # Remove from active tasks
                session.active_tasks.pop(op_id, None)
            
            # Check for failures and handle fail_on_first_error
            if fail_on_first_error:
                failed_ops = [op_id for op_id in operation_ids if session.results[op_id].failed]
                if failed_ops:
                    self.logger.error(f"Stopping execution due to failed operations: {failed_ops}")
                    await self._cancel_pending_operations(session)
                    return
            
            # Report progress
            if progress_callback:
                progress = ((group_index + 1) / total_groups) * 100
                await progress_callback(session.session_id, progress)
    
    def _group_operations_by_dependency_level(self,
                                            session: AsyncTestSession,
                                            execution_order: List[str]) -> List[List[str]]:
        """Group operations by dependency level for parallel execution within groups."""
        dependency_levels = {}
        
        for op_id in execution_order:
            if not session.dependency_graph.get(op_id):
                dependency_levels[op_id] = 0
            else:
                max_dep_level = max(dependency_levels[dep] for dep in session.dependency_graph[op_id])
                dependency_levels[op_id] = max_dep_level + 1
        
        # Group by level
        level_groups = defaultdict(list)
        for op_id, level in dependency_levels.items():
            level_groups[level].append(op_id)
        
        return [level_groups[level] for level in sorted(level_groups.keys())]
    
    async def _execute_single_operation(self,
                                       session: AsyncTestSession,
                                       operation_spec: AsyncOperationSpec,
                                       timeout: float) -> AsyncOperationResult:
        """Execute single operation with retry logic."""
        operation_id = operation_spec.operation_id
        start_time = time.time()
        
        # Initialize result
        result = AsyncOperationResult(
            operation_id=operation_id,
            state=AsyncTestState.RUNNING,
            start_time=start_time
        )
        
        # Memory measurement
        memory_start = None
        if session.performance_helper:
            memory_start = session.performance_helper._get_memory_usage()
        
        retry_count = 0
        last_exception = None
        
        while retry_count <= operation_spec.retry_count:
            try:
                # Apply concurrency control
                if session.concurrency_semaphore:
                    async with session.concurrency_semaphore:
                        operation_result = await asyncio.wait_for(
                            self._call_operation_func(operation_spec),
                            timeout=timeout
                        )
                else:
                    operation_result = await asyncio.wait_for(
                        self._call_operation_func(operation_spec),
                        timeout=timeout
                    )
                
                # Success
                result.result = operation_result
                result.state = AsyncTestState.COMPLETED
                result.end_time = time.time()
                result.duration_ms = (result.end_time - result.start_time) * 1000
                result.retry_count = retry_count
                
                if memory_start is not None:
                    memory_end = session.performance_helper._get_memory_usage()
                    result.memory_usage_mb = memory_end - memory_start
                
                self.logger.debug(f"Operation {operation_id} completed successfully (attempt {retry_count + 1})")
                return result
                
            except asyncio.TimeoutError:
                last_exception = TimeoutError(f"Operation {operation_id} timed out after {timeout}s")
                self.logger.warning(f"Operation {operation_id} timed out (attempt {retry_count + 1})")
                
            except asyncio.CancelledError:
                result.state = AsyncTestState.CANCELLED
                result.end_time = time.time()
                result.exception = asyncio.CancelledError("Operation cancelled")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Operation {operation_id} failed (attempt {retry_count + 1}): {e}")
            
            retry_count += 1
            
            # Wait before retry if not the last attempt
            if retry_count <= operation_spec.retry_count:
                await asyncio.sleep(operation_spec.retry_delay_seconds)
        
        # All retries exhausted
        result.state = AsyncTestState.FAILED
        result.end_time = time.time()
        result.duration_ms = (result.end_time - result.start_time) * 1000
        result.exception = last_exception
        result.retry_count = retry_count - 1
        
        self.logger.error(f"Operation {operation_id} failed after {retry_count} attempts")
        return result
    
    async def _call_operation_func(self, operation_spec: AsyncOperationSpec) -> Any:
        """Call operation function with proper argument handling."""
        if asyncio.iscoroutinefunction(operation_spec.operation_func):
            return await operation_spec.operation_func(*operation_spec.args, **operation_spec.kwargs)
        else:
            # Run synchronous function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                functools.partial(operation_spec.operation_func, *operation_spec.args, **operation_spec.kwargs)
            )
    
    async def _cancel_pending_operations(self, session: AsyncTestSession) -> None:
        """Cancel all pending operations in session."""
        for task in session.active_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for cancellation to complete
        if session.active_tasks:
            try:
                await asyncio.wait(session.active_tasks.values(), timeout=2.0)
            except asyncio.TimeoutError:
                self.logger.warning("Some tasks did not cancel within timeout")
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global coordinator statistics."""
        return {
            'coordinator_stats': self.stats.copy(),
            'active_sessions': len(self.active_sessions),
            'session_ids': list(self.active_sessions.keys()),
            'total_active_operations': sum(len(s.active_tasks) for s in self.active_sessions.values()),
            'configuration': {
                'default_timeout': self.default_timeout,
                'max_concurrent_operations': self.max_concurrent_operations,
                'concurrency_policy': self.concurrency_policy.value
            }
        }


# =====================================================================
# ASYNC CONTEXT MANAGERS
# =====================================================================

@asynccontextmanager
async def async_test_environment(
    environment_spec: Optional[EnvironmentSpec] = None,
    session_timeout: float = 300.0,
    auto_cleanup: bool = True
):
    """
    Async context manager for complete test environment setup and teardown.
    
    Args:
        environment_spec: Optional environment specification
        session_timeout: Maximum session duration
        auto_cleanup: Whether to automatically cleanup on exit
    """
    coordinator = AsyncTestCoordinator(default_timeout=30.0)
    environment_manager = TestEnvironmentManager(environment_spec)
    
    try:
        # Setup environment
        environment_data = environment_manager.setup_environment()
        
        # Create async session
        session_id = await coordinator.create_session(
            enable_resource_monitoring=True,
            enable_performance_monitoring=True
        )
        
        session_start = time.time()
        
        context = {
            'coordinator': coordinator,
            'session_id': session_id,
            'environment_manager': environment_manager,
            'environment_data': environment_data,
            'session_start': session_start,
            'timeout': session_timeout
        }
        
        yield context
        
    finally:
        try:
            # Cleanup session if it exists
            if session_id in coordinator.active_sessions:
                await coordinator.cleanup_session(session_id)
        except Exception as e:
            logging.warning(f"Session cleanup error: {e}")
        
        # Cleanup environment
        if auto_cleanup:
            try:
                environment_manager.cleanup()
            except Exception as e:
                logging.warning(f"Environment cleanup error: {e}")


@asynccontextmanager
async def async_resource_manager(
    resources: Dict[str, Callable],
    cleanup_callbacks: Optional[Dict[str, Callable]] = None,
    resource_timeout: float = 30.0,
    monitor_resources: bool = True
):
    """
    Async context manager for resource management with automatic cleanup.
    
    Args:
        resources: Dictionary mapping resource names to async creation functions
        cleanup_callbacks: Optional cleanup functions for each resource
        resource_timeout: Timeout for resource operations
        monitor_resources: Whether to monitor resource usage
    """
    created_resources = {}
    resource_monitor = None
    
    try:
        # Start resource monitoring if requested
        if monitor_resources:
            resource_monitor = AdvancedResourceMonitor(sampling_interval=1.0)
            resource_monitor.start_monitoring()
        
        # Create resources
        for resource_name, create_func in resources.items():
            try:
                if asyncio.iscoroutinefunction(create_func):
                    resource = await asyncio.wait_for(create_func(), timeout=resource_timeout)
                else:
                    resource = await asyncio.get_event_loop().run_in_executor(None, create_func)
                
                created_resources[resource_name] = resource
                logging.info(f"Created resource: {resource_name}")
                
            except Exception as e:
                logging.error(f"Failed to create resource {resource_name}: {e}")
                raise
        
        # Create context
        context = {
            'resources': created_resources,
            'resource_monitor': resource_monitor,
            'created_count': len(created_resources)
        }
        
        yield context
        
    finally:
        # Cleanup resources
        if cleanup_callbacks:
            for resource_name, cleanup_func in cleanup_callbacks.items():
                if resource_name in created_resources:
                    try:
                        if asyncio.iscoroutinefunction(cleanup_func):
                            await asyncio.wait_for(
                                cleanup_func(created_resources[resource_name]),
                                timeout=resource_timeout
                            )
                        else:
                            await asyncio.get_event_loop().run_in_executor(
                                None, cleanup_func, created_resources[resource_name]
                            )
                        logging.info(f"Cleaned up resource: {resource_name}")
                    except Exception as e:
                        logging.warning(f"Resource cleanup failed for {resource_name}: {e}")
        
        # Stop resource monitoring
        if resource_monitor:
            try:
                resource_monitor.stop_monitoring()
            except Exception as e:
                logging.warning(f"Resource monitor cleanup failed: {e}")


@asynccontextmanager
async def async_error_injection(
    error_specs: List[Dict[str, Any]],
    injection_probability: float = 1.0,
    recovery_testing: bool = True
):
    """
    Async context manager for controlled error injection and recovery testing.
    
    Args:
        error_specs: List of error specifications with 'target', 'error_type', etc.
        injection_probability: Probability of injecting errors (0.0 to 1.0)
        recovery_testing: Whether to test error recovery mechanisms
    """
    injected_errors = []
    recovery_results = []
    
    class ErrorInjector:
        def __init__(self):
            self.active = True
            self.call_counts = defaultdict(int)
        
        async def should_inject_error(self, target: str) -> Optional[Exception]:
            """Check if error should be injected for target."""
            if not self.active or random.random() > injection_probability:
                return None
            
            self.call_counts[target] += 1
            
            for spec in error_specs:
                if spec.get('target') == target:
                    trigger_after = spec.get('trigger_after', 1)
                    if self.call_counts[target] >= trigger_after:
                        error_class = spec.get('error_type', Exception)
                        error_message = spec.get('message', f'Injected error for {target}')
                        error = error_class(error_message)
                        
                        injected_errors.append({
                            'target': target,
                            'error': error,
                            'timestamp': time.time(),
                            'call_count': self.call_counts[target]
                        })
                        
                        return error
            
            return None
        
        async def test_recovery(self, target: str, recovery_func: Callable) -> bool:
            """Test recovery mechanism for target."""
            if not recovery_testing:
                return True
            
            try:
                if asyncio.iscoroutinefunction(recovery_func):
                    await recovery_func()
                else:
                    recovery_func()
                
                recovery_results.append({
                    'target': target,
                    'recovered': True,
                    'timestamp': time.time()
                })
                return True
                
            except Exception as e:
                recovery_results.append({
                    'target': target,
                    'recovered': False,
                    'error': str(e),
                    'timestamp': time.time()
                })
                return False
    
    injector = ErrorInjector()
    
    try:
        context = {
            'injector': injector,
            'injected_errors': injected_errors,
            'recovery_results': recovery_results
        }
        
        yield context
        
    finally:
        # Deactivate injector
        injector.active = False
        
        # Log injection summary
        logging.info(f"Error injection completed: {len(injected_errors)} errors injected")
        if recovery_testing:
            recovered = len([r for r in recovery_results if r['recovered']])
            logging.info(f"Recovery testing: {recovered}/{len(recovery_results)} successful")


@asynccontextmanager  
async def async_performance_monitor(
    performance_thresholds: Optional[Dict[str, PerformanceThreshold]] = None,
    sampling_interval: float = 1.0,
    alert_on_threshold_breach: bool = True
):
    """
    Async context manager for performance monitoring during test execution.
    
    Args:
        performance_thresholds: Performance thresholds to monitor
        sampling_interval: How often to sample performance metrics
        alert_on_threshold_breach: Whether to alert on threshold breaches
    """
    # Initialize monitoring components
    resource_monitor = AdvancedResourceMonitor(sampling_interval=sampling_interval)
    performance_helper = PerformanceAssertionHelper()
    
    # Set up default thresholds if none provided
    if performance_thresholds is None:
        performance_thresholds = {
            'memory_usage_mb': PerformanceThreshold(
                'memory_usage_mb', 1000, 'lte', 'MB', 'warning',
                'Memory usage should stay under 1GB'
            ),
            'cpu_usage_percent': PerformanceThreshold(
                'cpu_usage_percent', 80, 'lte', '%', 'warning',
                'CPU usage should stay under 80%'
            )
        }
    
    # Performance data collection
    performance_samples = []
    threshold_breaches = []
    
    async def sample_performance():
        """Sample performance metrics and check thresholds."""
        while resource_monitor.monitoring:
            try:
                # Get current metrics
                if resource_monitor.snapshots:
                    latest_snapshot = resource_monitor.snapshots[-1]
                    
                    sample = {
                        'timestamp': time.time(),
                        'memory_mb': latest_snapshot.memory_mb,
                        'cpu_percent': latest_snapshot.cpu_percent,
                        'active_threads': latest_snapshot.active_threads
                    }
                    
                    performance_samples.append(sample)
                    
                    # Check thresholds
                    if alert_on_threshold_breach:
                        for threshold_name, threshold in performance_thresholds.items():
                            if threshold_name == 'memory_usage_mb':
                                value = sample['memory_mb']
                            elif threshold_name == 'cpu_usage_percent':
                                value = sample['cpu_percent']
                            else:
                                continue
                            
                            passed, message = threshold.check(value)
                            if not passed:
                                breach = {
                                    'timestamp': sample['timestamp'],
                                    'threshold_name': threshold_name,
                                    'threshold_value': threshold.threshold_value,
                                    'actual_value': value,
                                    'message': message
                                }
                                threshold_breaches.append(breach)
                                logging.warning(f"Performance threshold breach: {message}")
                
                await asyncio.sleep(sampling_interval)
                
            except Exception as e:
                logging.debug(f"Performance sampling error: {e}")
                await asyncio.sleep(sampling_interval)
    
    try:
        # Start monitoring
        performance_helper.establish_memory_baseline()
        resource_monitor.start_monitoring()
        
        # Start performance sampling task
        sampling_task = asyncio.create_task(sample_performance())
        
        context = {
            'resource_monitor': resource_monitor,
            'performance_helper': performance_helper,
            'performance_samples': performance_samples,
            'threshold_breaches': threshold_breaches,
            'thresholds': performance_thresholds
        }
        
        yield context
        
    finally:
        # Stop sampling
        if 'sampling_task' in locals():
            sampling_task.cancel()
            try:
                await sampling_task
            except asyncio.CancelledError:
                pass
        
        # Stop monitoring
        if resource_monitor.monitoring:
            resource_monitor.stop_monitoring()
        
        # Log performance summary
        if performance_samples:
            avg_memory = statistics.mean([s['memory_mb'] for s in performance_samples])
            avg_cpu = statistics.mean([s['cpu_percent'] for s in performance_samples])
            logging.info(f"Performance monitoring completed: avg memory {avg_memory:.1f}MB, avg CPU {avg_cpu:.1f}%")
            
        if threshold_breaches:
            logging.warning(f"Performance monitoring detected {len(threshold_breaches)} threshold breaches")


# =====================================================================
# ASYNC UTILITIES
# =====================================================================

class AsyncTestDataGenerator:
    """Utility for generating test data asynchronously."""
    
    def __init__(self, 
                 coordinator: Optional[AsyncTestCoordinator] = None,
                 generation_delay: float = 0.1):
        self.coordinator = coordinator
        self.generation_delay = generation_delay
        self.logger = logging.getLogger(f"async_data_gen_{id(self)}")
    
    async def generate_biomedical_queries(self, 
                                        count: int,
                                        disease_types: Optional[List[str]] = None) -> List[str]:
        """Generate biomedical query test data asynchronously."""
        if disease_types is None:
            disease_types = ['diabetes', 'cardiovascular', 'cancer', 'liver_disease']
        
        queries = []
        
        query_templates = {
            'diabetes': [
                "What metabolites are elevated in diabetes patients?",
                "How does glucose metabolism change in diabetes?", 
                "What are the key biomarkers for diabetes diagnosis?",
                "Which metabolic pathways are disrupted in diabetes?"
            ],
            'cardiovascular': [
                "What lipid biomarkers indicate cardiovascular disease risk?",
                "How does cholesterol metabolism affect heart disease?",
                "What are the inflammatory markers in cardiovascular disease?",
                "Which metabolites predict cardiac events?"
            ],
            'cancer': [
                "What metabolic changes occur in cancer cells?",
                "How does the Warburg effect manifest in tumor metabolism?",
                "What are the key oncometabolites in cancer?",
                "Which metabolic pathways are reprogrammed in cancer?"
            ],
            'liver_disease': [
                "What metabolites indicate liver dysfunction?",
                "How does hepatic metabolism change in liver disease?",
                "What are the biomarkers for liver fibrosis?",
                "Which bile acids are altered in liver disease?"
            ]
        }
        
        for i in range(count):
            disease = random.choice(disease_types)
            template = random.choice(query_templates[disease])
            queries.append(template)
            
            # Small delay to simulate realistic generation
            if self.generation_delay > 0:
                await asyncio.sleep(self.generation_delay / 1000)
            
            # Yield control periodically
            if i % 10 == 0:
                await asyncio.sleep(0)
        
        self.logger.info(f"Generated {len(queries)} biomedical queries")
        return queries
    
    async def generate_pdf_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate PDF test data asynchronously."""
        pdf_data = []
        
        for i in range(count):
            data = {
                'filename': f'test_paper_{i+1}.pdf',
                'title': f'Clinical Study {i+1}: Biomarker Discovery in Disease',
                'content': f'Abstract: This study investigates biomarkers in clinical population {i+1}. Methods: Advanced analytical techniques were used. Results: Significant findings were identified. Conclusions: Important clinical implications discovered.',
                'authors': [f'Dr. Author{j}' for j in range(random.randint(2, 5))],
                'year': random.randint(2020, 2024),
                'page_count': random.randint(8, 25)
            }
            pdf_data.append(data)
            
            if self.generation_delay > 0:
                await asyncio.sleep(self.generation_delay / 1000)
        
        self.logger.info(f"Generated {len(pdf_data)} PDF data entries")
        return pdf_data
    
    async def cleanup_generated_data(self) -> None:
        """Clean up any resources used for data generation."""
        # Force garbage collection of generated data
        gc.collect()
        self.logger.info("Cleaned up generated test data")


class AsyncOperationBatcher:
    """Utility for batching and coordinating async operations."""
    
    def __init__(self, 
                 batch_size: int = 10,
                 batch_delay: float = 0.1,
                 max_concurrent_batches: int = 3):
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.max_concurrent_batches = max_concurrent_batches
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        self.logger = logging.getLogger(f"async_batcher_{id(self)}")
    
    async def execute_batched_operations(self,
                                       operations: List[AsyncOperationSpec],
                                       progress_callback: Optional[Callable] = None) -> Dict[str, AsyncOperationResult]:
        """Execute operations in batches with concurrency control."""
        # Split operations into batches
        batches = [operations[i:i + self.batch_size] 
                  for i in range(0, len(operations), self.batch_size)]
        
        self.logger.info(f"Executing {len(operations)} operations in {len(batches)} batches")
        
        all_results = {}
        
        async def execute_batch(batch_ops: List[AsyncOperationSpec], batch_index: int) -> Dict[str, AsyncOperationResult]:
            """Execute single batch of operations."""
            async with self.semaphore:
                batch_results = {}
                batch_start = time.time()
                
                # Execute operations in batch concurrently  
                tasks = {}
                for op_spec in batch_ops:
                    task = asyncio.create_task(self._execute_operation(op_spec))
                    tasks[op_spec.operation_id] = task
                
                # Gather results
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                
                for op_id, result in zip(tasks.keys(), results):
                    if isinstance(result, Exception):
                        batch_results[op_id] = AsyncOperationResult(
                            operation_id=op_id,
                            state=AsyncTestState.FAILED,
                            start_time=batch_start,
                            end_time=time.time(),
                            exception=result
                        )
                    else:
                        batch_results[op_id] = result
                
                batch_duration = time.time() - batch_start
                self.logger.debug(f"Batch {batch_index + 1} completed in {batch_duration:.2f}s")
                
                return batch_results
        
        # Execute all batches
        batch_tasks = []
        for i, batch in enumerate(batches):
            task = asyncio.create_task(execute_batch(batch, i))
            batch_tasks.append(task)
            
            # Add small delay between batch starts
            if self.batch_delay > 0:
                await asyncio.sleep(self.batch_delay)
        
        # Collect all results
        batch_results_list = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for batch_results in batch_results_list:
            if isinstance(batch_results, dict):
                all_results.update(batch_results)
            elif isinstance(batch_results, Exception):
                self.logger.error(f"Batch execution failed: {batch_results}")
        
        # Report final progress
        if progress_callback:
            await progress_callback(100.0)
        
        self.logger.info(f"Batched execution completed: {len(all_results)} results")
        return all_results
    
    async def _execute_operation(self, op_spec: AsyncOperationSpec) -> AsyncOperationResult:
        """Execute single operation."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(op_spec.operation_func):
                result = await op_spec.operation_func(*op_spec.args, **op_spec.kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    functools.partial(op_spec.operation_func, *op_spec.args, **op_spec.kwargs)
                )
            
            return AsyncOperationResult(
                operation_id=op_spec.operation_id,
                state=AsyncTestState.COMPLETED,
                start_time=start_time,
                end_time=time.time(),
                result=result,
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return AsyncOperationResult(
                operation_id=op_spec.operation_id,
                state=AsyncTestState.FAILED,
                start_time=start_time,
                end_time=time.time(),
                exception=e,
                duration_ms=(time.time() - start_time) * 1000
            )


class AsyncResultAggregator:
    """Utility for aggregating and analyzing async test results."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"async_aggregator_{id(self)}")
    
    async def aggregate_results(self, 
                              results: Dict[str, AsyncOperationResult],
                              analysis_functions: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """Aggregate and analyze operation results."""
        if not results:
            return {'status': 'no_results', 'total_operations': 0}
        
        # Basic statistics
        total_operations = len(results)
        successful_operations = len([r for r in results.values() if r.succeeded])
        failed_operations = len([r for r in results.values() if r.failed])
        cancelled_operations = len([r for r in results.values() if r.state == AsyncTestState.CANCELLED])
        
        # Duration statistics
        durations = [r.duration_ms for r in results.values() if r.duration_ms is not None]
        duration_stats = {}
        if durations:
            duration_stats = {
                'mean_duration_ms': statistics.mean(durations),
                'median_duration_ms': statistics.median(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'std_duration_ms': statistics.stdev(durations) if len(durations) > 1 else 0
            }
        
        # Memory usage statistics
        memory_usages = [r.memory_usage_mb for r in results.values() if r.memory_usage_mb is not None]
        memory_stats = {}
        if memory_usages:
            memory_stats = {
                'mean_memory_mb': statistics.mean(memory_usages),
                'median_memory_mb': statistics.median(memory_usages),
                'max_memory_mb': max(memory_usages),
                'total_memory_mb': sum(memory_usages)
            }
        
        # Error analysis
        errors = [r.exception for r in results.values() if r.exception is not None]
        error_types = defaultdict(int)
        for error in errors:
            error_types[type(error).__name__] += 1
        
        # Retry analysis
        retry_counts = [r.retry_count for r in results.values()]
        retry_stats = {
            'total_retries': sum(retry_counts),
            'operations_with_retries': len([c for c in retry_counts if c > 0]),
            'max_retries': max(retry_counts) if retry_counts else 0
        }
        
        # Base aggregation
        aggregation = {
            'execution_summary': {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'failed_operations': failed_operations,
                'cancelled_operations': cancelled_operations,
                'success_rate_percent': (successful_operations / total_operations) * 100,
                'failure_rate_percent': (failed_operations / total_operations) * 100
            },
            'performance_statistics': {
                'duration_stats': duration_stats,
                'memory_stats': memory_stats
            },
            'error_analysis': {
                'error_types': dict(error_types),
                'total_errors': len(errors),
                'unique_error_types': len(error_types)
            },
            'retry_analysis': retry_stats,
            'aggregation_timestamp': time.time()
        }
        
        # Apply custom analysis functions
        if analysis_functions:
            custom_analysis = {}
            for analysis_func in analysis_functions:
                try:
                    if asyncio.iscoroutinefunction(analysis_func):
                        result = await analysis_func(results)
                    else:
                        result = analysis_func(results)
                    
                    func_name = analysis_func.__name__
                    custom_analysis[func_name] = result
                    
                except Exception as e:
                    self.logger.warning(f"Custom analysis function {analysis_func.__name__} failed: {e}")
            
            if custom_analysis:
                aggregation['custom_analysis'] = custom_analysis
        
        self.logger.info(f"Aggregated results for {total_operations} operations")
        return aggregation
    
    async def export_results(self, 
                           aggregation: Dict[str, Any],
                           filepath: Path,
                           include_raw_results: bool = False,
                           raw_results: Optional[Dict[str, AsyncOperationResult]] = None) -> None:
        """Export aggregated results to file."""
        export_data = aggregation.copy()
        
        if include_raw_results and raw_results:
            export_data['raw_results'] = {
                op_id: result.to_dict() for op_id, result in raw_results.items()
            }
        
        export_data['export_metadata'] = {
            'export_timestamp': datetime.now().isoformat(),
            'export_filepath': str(filepath),
            'include_raw_results': include_raw_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {filepath}")


class AsyncRetryManager:
    """Utility for implementing async retry mechanisms with various strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"async_retry_{id(self)}")
    
    async def retry_with_exponential_backoff(self,
                                           operation_func: Callable,
                                           max_retries: int = 3,
                                           initial_delay: float = 1.0,
                                           backoff_multiplier: float = 2.0,
                                           max_delay: float = 60.0,
                                           exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Any:
        """Retry operation with exponential backoff."""
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation_func):
                    return await operation_func()
                else:
                    return await asyncio.get_event_loop().run_in_executor(None, operation_func)
                    
            except exceptions as e:
                if attempt == max_retries:
                    self.logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                    raise
                
                self.logger.warning(f"Operation failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
                delay = min(delay * backoff_multiplier, max_delay)
        
        return None  # Should not reach here
    
    async def retry_with_jitter(self,
                              operation_func: Callable,
                              max_retries: int = 3,
                              base_delay: float = 1.0,
                              max_jitter: float = 1.0,
                              exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Any:
        """Retry operation with jitter to avoid thundering herd."""
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation_func):
                    return await operation_func()
                else:
                    return await asyncio.get_event_loop().run_in_executor(None, operation_func)
                    
            except exceptions as e:
                if attempt == max_retries:
                    raise
                
                # Calculate delay with jitter
                jitter = random.uniform(0, max_jitter)
                delay = base_delay + jitter
                
                self.logger.warning(f"Operation failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        return None


# =====================================================================
# PYTEST FIXTURES FOR ASYNC UTILITIES
# =====================================================================

@pytest_asyncio.fixture
async def async_test_coordinator():
    """Provide AsyncTestCoordinator for tests."""
    coordinator = AsyncTestCoordinator(default_timeout=30.0, max_concurrent_operations=5)
    yield coordinator
    
    # Cleanup all sessions
    for session_id in list(coordinator.active_sessions.keys()):
        try:
            await coordinator.cleanup_session(session_id)
        except Exception as e:
            logging.warning(f"Session cleanup failed for {session_id}: {e}")


@pytest_asyncio.fixture
async def async_test_session(async_test_coordinator):
    """Provide async test session with coordinator."""
    session_id = await async_test_coordinator.create_session(
        enable_resource_monitoring=True,
        enable_performance_monitoring=True
    )
    
    yield {
        'coordinator': async_test_coordinator,
        'session_id': session_id
    }
    
    # Session will be cleaned up by coordinator fixture


@pytest_asyncio.fixture
async def async_data_generator():
    """Provide AsyncTestDataGenerator for tests."""
    generator = AsyncTestDataGenerator(generation_delay=0.01)
    yield generator
    await generator.cleanup_generated_data()


@pytest_asyncio.fixture
async def async_operation_batcher():
    """Provide AsyncOperationBatcher for tests."""
    return AsyncOperationBatcher(batch_size=5, batch_delay=0.05, max_concurrent_batches=2)


@pytest_asyncio.fixture
async def async_result_aggregator():
    """Provide AsyncResultAggregator for tests."""
    return AsyncResultAggregator()


@pytest_asyncio.fixture
async def async_retry_manager():
    """Provide AsyncRetryManager for tests."""
    return AsyncRetryManager()


# =====================================================================
# CONVENIENCE FUNCTIONS FOR COMMON ASYNC TESTING PATTERNS
# =====================================================================

async def create_async_test_operations(operation_funcs: List[Callable],
                                     dependencies: Optional[Dict[str, List[str]]] = None,
                                     timeout: float = 30.0) -> List[AsyncOperationSpec]:
    """Create async operation specs from function list."""
    operations = []
    
    for i, func in enumerate(operation_funcs):
        op_id = f"operation_{i}_{func.__name__}"
        deps = dependencies.get(op_id, []) if dependencies else []
        
        spec = AsyncOperationSpec(
            operation_id=op_id,
            operation_func=func,
            dependencies=deps,
            timeout_seconds=timeout
        )
        operations.append(spec)
    
    return operations


async def run_coordinated_async_test(coordinator: AsyncTestCoordinator,
                                   operations: List[AsyncOperationSpec],
                                   session_id: Optional[str] = None,
                                   fail_on_first_error: bool = False) -> Dict[str, AsyncOperationResult]:
    """Run coordinated async test with operations."""
    if session_id is None:
        session_id = await coordinator.create_session()
    
    try:
        # Add operations to session
        await coordinator.add_operations_batch(session_id, operations)
        
        # Execute session
        results = await coordinator.execute_session(session_id, fail_on_first_error)
        
        return results
        
    finally:
        await coordinator.cleanup_session(session_id)


# Make utilities available at module level
__all__ = [
    'AsyncTestCoordinator',
    'AsyncOperationSpec',
    'AsyncOperationResult', 
    'AsyncTestSession',
    'AsyncTestState',
    'ConcurrencyPolicy',
    'AsyncTestDataGenerator',
    'AsyncOperationBatcher',
    'AsyncResultAggregator',
    'AsyncRetryManager',
    'async_test_environment',
    'async_resource_manager',
    'async_error_injection',
    'async_performance_monitor',
    'create_async_test_operations',
    'run_coordinated_async_test'
]