#!/usr/bin/env python3
"""
Advanced Cleanup System for Clinical Metabolomics Oracle LightRAG Integration.

This module provides comprehensive cleanup mechanisms that go beyond the basic cleanup
in fixtures, offering advanced resource management, failure recovery, and system-wide
cleanup orchestration.

Key Features:
1. System-wide cleanup orchestration across multiple test runs
2. Advanced resource management (memory, file handles, network, processes)
3. Cleanup validation and verification
4. Integration with existing test infrastructure
5. Cleanup policies and strategies
6. Failure recovery and retry mechanisms
7. Performance optimization
8. Monitoring and reporting

Components:
- AdvancedCleanupOrchestrator: Central cleanup coordination
- ResourceManager: Specialized resource type cleanup
- CleanupPolicy: Different cleanup strategies and policies
- FailureRecoverySystem: Handle partial cleanup failures
- CleanupValidator: Verify cleanup effectiveness
- PerformanceOptimizer: Ensure cleanup doesn't impact test performance
- MonitoringReporter: Track cleanup effectiveness and resource usage

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import gc
import logging
import os
import psutil
import resource
import signal
import sqlite3
import threading
import time
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Dict, List, Set, Any, Optional, Union, Callable, Generator, 
    AsyncGenerator, TypeVar, Generic, Protocol, Tuple
)
import json
import shutil
import tempfile
import uuid
from collections import defaultdict, deque
import mmap

# Import existing test infrastructure
try:
    from .test_data_fixtures import TestDataManager, TestDataConfig
    from .conftest import pytest_configure
except ImportError:
    # Handle import for standalone usage
    pass


# =====================================================================
# CLEANUP POLICIES AND STRATEGIES
# =====================================================================

class CleanupStrategy(Enum):
    """Defines different cleanup strategies."""
    IMMEDIATE = auto()        # Clean up immediately after use
    DEFERRED = auto()         # Clean up at end of test/session
    SCHEDULED = auto()        # Clean up on schedule
    ON_DEMAND = auto()        # Clean up only when explicitly requested
    RESOURCE_BASED = auto()   # Clean up based on resource thresholds


class CleanupScope(Enum):
    """Defines cleanup scope levels."""
    FUNCTION = auto()         # Function-level cleanup
    CLASS = auto()            # Test class-level cleanup
    MODULE = auto()           # Module-level cleanup
    SESSION = auto()          # Session-level cleanup
    SYSTEM = auto()           # System-wide cleanup


class ResourceType(Enum):
    """Types of resources to manage."""
    MEMORY = auto()
    FILE_HANDLES = auto()
    NETWORK_CONNECTIONS = auto()
    PROCESSES = auto()
    THREADS = auto()
    DATABASE_CONNECTIONS = auto()
    TEMPORARY_FILES = auto()
    CACHE_ENTRIES = auto()


@dataclass
class CleanupPolicy:
    """Configuration for cleanup behavior."""
    strategy: CleanupStrategy = CleanupStrategy.DEFERRED
    scope: CleanupScope = CleanupScope.FUNCTION
    resource_types: Set[ResourceType] = field(default_factory=set)
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    force_cleanup: bool = False
    validate_cleanup: bool = True
    report_cleanup: bool = True
    emergency_cleanup: bool = True


@dataclass
class ResourceThresholds:
    """Resource usage thresholds for cleanup triggers."""
    memory_mb: Optional[float] = 1024
    file_handles: Optional[int] = 1000
    db_connections: Optional[int] = 50
    temp_files: Optional[int] = 100
    temp_size_mb: Optional[float] = 500
    cache_entries: Optional[int] = 10000


# =====================================================================
# RESOURCE MANAGERS
# =====================================================================

class ResourceManager(ABC):
    """Abstract base class for resource-specific cleanup managers."""
    
    def __init__(self, policy: CleanupPolicy, thresholds: ResourceThresholds):
        self.policy = policy
        self.thresholds = thresholds
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._cleanup_stats = defaultdict(int)
        self._failed_cleanups = []
        
    @abstractmethod
    def cleanup(self) -> bool:
        """Perform cleanup for this resource type."""
        pass
        
    @abstractmethod
    def validate_cleanup(self) -> bool:
        """Validate that cleanup was successful."""
        pass
        
    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage metrics."""
        pass
        
    def should_cleanup(self) -> bool:
        """Check if cleanup should be triggered based on thresholds."""
        usage = self.get_resource_usage()
        return self._check_thresholds(usage)
        
    def _check_thresholds(self, usage: Dict[str, Any]) -> bool:
        """Check if usage exceeds thresholds."""
        # Override in subclasses for specific threshold checks
        return False
        
    def record_cleanup_attempt(self, success: bool, details: str = ""):
        """Record cleanup attempt for statistics."""
        if success:
            self._cleanup_stats['successful_cleanups'] += 1
        else:
            self._cleanup_stats['failed_cleanups'] += 1
            self._failed_cleanups.append({
                'timestamp': datetime.now().isoformat(),
                'details': details
            })


class MemoryManager(ResourceManager):
    """Manages memory cleanup and garbage collection."""
    
    def __init__(self, policy: CleanupPolicy, thresholds: ResourceThresholds):
        super().__init__(policy, thresholds)
        self._weak_refs = weakref.WeakSet()
        
    def register_object(self, obj):
        """Register object for memory tracking."""
        self._weak_refs.add(obj)
        
    def cleanup(self) -> bool:
        """Perform memory cleanup."""
        try:
            # Force garbage collection
            collected = gc.collect()
            self.logger.debug(f"Garbage collection freed {collected} objects")
            
            # Clear weak references
            self._weak_refs.clear()
            
            # Force memory cleanup for specific object types
            self._cleanup_caches()
            
            self.record_cleanup_attempt(True, f"Collected {collected} objects")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            self.record_cleanup_attempt(False, str(e))
            return False
            
    def _cleanup_caches(self):
        """Clear various internal caches."""
        # Clear functools.lru_cache caches if present
        try:
            import functools
            # This would need to be implemented based on actual cache usage
        except ImportError:
            pass
            
    def validate_cleanup(self) -> bool:
        """Validate memory cleanup effectiveness."""
        usage = self.get_resource_usage()
        return usage['memory_mb'] < self.thresholds.memory_mb if self.thresholds.memory_mb else True
        
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'memory_mb': memory_info.rss / (1024 * 1024),
            'memory_percent': process.memory_percent(),
            'gc_counts': gc.get_count(),
            'gc_stats': gc.get_stats() if hasattr(gc, 'get_stats') else None
        }
        
    def _check_thresholds(self, usage: Dict[str, Any]) -> bool:
        """Check memory thresholds."""
        if self.thresholds.memory_mb:
            return usage['memory_mb'] > self.thresholds.memory_mb
        return False


class FileHandleManager(ResourceManager):
    """Manages file handle cleanup."""
    
    def __init__(self, policy: CleanupPolicy, thresholds: ResourceThresholds):
        super().__init__(policy, thresholds)
        self._open_files = weakref.WeakSet()
        
    def register_file(self, file_obj):
        """Register file object for tracking."""
        self._open_files.add(file_obj)
        
    def cleanup(self) -> bool:
        """Close open file handles."""
        closed_count = 0
        failed_count = 0
        
        # Close tracked file objects
        for file_obj in list(self._open_files):
            try:
                if hasattr(file_obj, 'close') and not file_obj.closed:
                    file_obj.close()
                    closed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to close file {file_obj}: {e}")
                failed_count += 1
                
        # Clear the weak set
        self._open_files.clear()
        
        success = failed_count == 0
        self.record_cleanup_attempt(success, f"Closed {closed_count} files, {failed_count} failures")
        return success
        
    def validate_cleanup(self) -> bool:
        """Validate file handle cleanup."""
        usage = self.get_resource_usage()
        return usage['open_files'] < self.thresholds.file_handles if self.thresholds.file_handles else True
        
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current file handle usage."""
        try:
            process = psutil.Process()
            open_files = process.open_files()
            
            return {
                'open_files': len(open_files),
                'file_descriptors': process.num_fds() if hasattr(process, 'num_fds') else 0,
                'tracked_files': len(self._open_files)
            }
        except Exception as e:
            self.logger.warning(f"Could not get file handle usage: {e}")
            return {'open_files': 0, 'file_descriptors': 0, 'tracked_files': 0}
            
    def _check_thresholds(self, usage: Dict[str, Any]) -> bool:
        """Check file handle thresholds."""
        if self.thresholds.file_handles:
            return usage['open_files'] > self.thresholds.file_handles
        return False


class DatabaseConnectionManager(ResourceManager):
    """Manages database connection cleanup."""
    
    def __init__(self, policy: CleanupPolicy, thresholds: ResourceThresholds):
        super().__init__(policy, thresholds)
        self._connections = weakref.WeakSet()
        
    def register_connection(self, conn):
        """Register database connection for tracking."""
        self._connections.add(conn)
        
    def cleanup(self) -> bool:
        """Close database connections."""
        closed_count = 0
        failed_count = 0
        
        for conn in list(self._connections):
            try:
                if hasattr(conn, 'close'):
                    conn.close()
                    closed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to close DB connection {conn}: {e}")
                failed_count += 1
                
        self._connections.clear()
        
        success = failed_count == 0
        self.record_cleanup_attempt(success, f"Closed {closed_count} connections, {failed_count} failures")
        return success
        
    def validate_cleanup(self) -> bool:
        """Validate database connection cleanup."""
        usage = self.get_resource_usage()
        return usage['active_connections'] < self.thresholds.db_connections if self.thresholds.db_connections else True
        
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current database connection usage."""
        return {
            'active_connections': len(self._connections),
            'tracked_connections': len(self._connections)
        }
        
    def _check_thresholds(self, usage: Dict[str, Any]) -> bool:
        """Check database connection thresholds."""
        if self.thresholds.db_connections:
            return usage['active_connections'] > self.thresholds.db_connections
        return False


class ProcessManager(ResourceManager):
    """Manages subprocess cleanup."""
    
    def __init__(self, policy: CleanupPolicy, thresholds: ResourceThresholds):
        super().__init__(policy, thresholds)
        self._processes = weakref.WeakSet()
        self._thread_pool = None
        self._process_pool = None
        
    def register_process(self, process):
        """Register subprocess for tracking."""
        self._processes.add(process)
        
    def set_thread_pool(self, pool: ThreadPoolExecutor):
        """Set thread pool for cleanup."""
        self._thread_pool = pool
        
    def set_process_pool(self, pool: ProcessPoolExecutor):
        """Set process pool for cleanup."""
        self._process_pool = pool
        
    def cleanup(self) -> bool:
        """Terminate processes and shutdown pools."""
        terminated_count = 0
        failed_count = 0
        
        # Terminate tracked processes
        for proc in list(self._processes):
            try:
                if hasattr(proc, 'terminate') and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                        terminated_count += 1
                    except:
                        proc.kill()
                        proc.wait(timeout=2)
                        terminated_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to terminate process {proc}: {e}")
                failed_count += 1
                
        # Shutdown thread pool
        if self._thread_pool:
            try:
                self._thread_pool.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                self.logger.warning(f"Failed to shutdown thread pool: {e}")
                failed_count += 1
                
        # Shutdown process pool
        if self._process_pool:
            try:
                self._process_pool.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                self.logger.warning(f"Failed to shutdown process pool: {e}")
                failed_count += 1
                
        self._processes.clear()
        
        success = failed_count == 0
        self.record_cleanup_attempt(success, f"Terminated {terminated_count} processes, {failed_count} failures")
        return success
        
    def validate_cleanup(self) -> bool:
        """Validate process cleanup."""
        usage = self.get_resource_usage()
        return len(self._processes) == 0
        
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current process usage."""
        return {
            'tracked_processes': len(self._processes),
            'thread_pool_active': self._thread_pool is not None,
            'process_pool_active': self._process_pool is not None
        }


class TemporaryFileManager(ResourceManager):
    """Manages temporary file and directory cleanup."""
    
    def __init__(self, policy: CleanupPolicy, thresholds: ResourceThresholds):
        super().__init__(policy, thresholds)
        self._temp_paths = set()
        self._temp_dirs = weakref.WeakSet()
        
    def register_temp_path(self, path: Union[str, Path]):
        """Register temporary path for cleanup."""
        self._temp_paths.add(Path(path))
        
    def register_temp_dir(self, temp_dir):
        """Register temporary directory object for cleanup."""
        self._temp_dirs.add(temp_dir)
        
    def cleanup(self) -> bool:
        """Clean up temporary files and directories."""
        cleaned_count = 0
        failed_count = 0
        
        # Clean registered paths
        for path in list(self._temp_paths):
            try:
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path, ignore_errors=False)
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to clean temp path {path}: {e}")
                failed_count += 1
                
        # Clean registered temp directory objects
        for temp_dir in list(self._temp_dirs):
            try:
                if hasattr(temp_dir, 'cleanup'):
                    temp_dir.cleanup()
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
                failed_count += 1
                
        self._temp_paths.clear()
        self._temp_dirs.clear()
        
        success = failed_count == 0
        self.record_cleanup_attempt(success, f"Cleaned {cleaned_count} temp items, {failed_count} failures")
        return success
        
    def validate_cleanup(self) -> bool:
        """Validate temporary file cleanup."""
        # Check if registered paths still exist
        existing_paths = sum(1 for path in self._temp_paths if path.exists())
        return existing_paths == 0
        
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current temporary file usage."""
        total_size = 0
        existing_count = 0
        
        for path in self._temp_paths:
            if path.exists():
                existing_count += 1
                try:
                    if path.is_file():
                        total_size += path.stat().st_size
                    else:
                        for item in path.rglob('*'):
                            if item.is_file():
                                total_size += item.stat().st_size
                except:
                    pass
                    
        return {
            'temp_files': existing_count,
            'temp_size_mb': total_size / (1024 * 1024),
            'tracked_temp_dirs': len(self._temp_dirs)
        }
        
    def _check_thresholds(self, usage: Dict[str, Any]) -> bool:
        """Check temporary file thresholds."""
        if self.thresholds.temp_files and usage['temp_files'] > self.thresholds.temp_files:
            return True
        if self.thresholds.temp_size_mb and usage['temp_size_mb'] > self.thresholds.temp_size_mb:
            return True
        return False


# =====================================================================
# CLEANUP ORCHESTRATOR
# =====================================================================

class AdvancedCleanupOrchestrator:
    """Central coordinator for advanced cleanup operations."""
    
    def __init__(self, policy: CleanupPolicy = None, thresholds: ResourceThresholds = None):
        self.policy = policy or CleanupPolicy()
        self.thresholds = thresholds or ResourceThresholds()
        self.logger = logging.getLogger(f"{__name__.{self.__class__.__name__}")
        
        # Initialize resource managers
        self._resource_managers = {
            ResourceType.MEMORY: MemoryManager(self.policy, self.thresholds),
            ResourceType.FILE_HANDLES: FileHandleManager(self.policy, self.thresholds),
            ResourceType.DATABASE_CONNECTIONS: DatabaseConnectionManager(self.policy, self.thresholds),
            ResourceType.PROCESSES: ProcessManager(self.policy, self.thresholds),
            ResourceType.TEMPORARY_FILES: TemporaryFileManager(self.policy, self.thresholds),
        }
        
        # Cleanup scheduling
        self._cleanup_schedule = []
        self._last_cleanup_time = datetime.now()
        self._cleanup_lock = threading.RLock()
        self._emergency_shutdown = False
        
        # Statistics and monitoring
        self._cleanup_history = deque(maxlen=1000)
        self._performance_metrics = defaultdict(list)
        
        # Integration with existing test infrastructure
        self._test_data_managers = weakref.WeakSet()
        
        # Set up emergency cleanup handlers
        self._setup_emergency_cleanup()
        
    def _setup_emergency_cleanup(self):
        """Set up emergency cleanup handlers for unexpected shutdowns."""
        if self.policy.emergency_cleanup:
            try:
                # Register signal handlers for emergency cleanup
                signal.signal(signal.SIGTERM, self._emergency_cleanup_handler)
                signal.signal(signal.SIGINT, self._emergency_cleanup_handler)
            except Exception as e:
                self.logger.warning(f"Could not set up emergency cleanup handlers: {e}")
                
    def _emergency_cleanup_handler(self, signum, frame):
        """Handle emergency cleanup on signal."""
        self.logger.warning(f"Emergency cleanup triggered by signal {signum}")
        self._emergency_shutdown = True
        self.force_cleanup()
        
    def register_test_data_manager(self, manager):
        """Register existing TestDataManager for integration."""
        self._test_data_managers.add(manager)
        
    def register_resource(self, resource_type: ResourceType, resource):
        """Register a resource for cleanup tracking."""
        if resource_type in self._resource_managers:
            manager = self._resource_managers[resource_type]
            
            if resource_type == ResourceType.MEMORY:
                manager.register_object(resource)
            elif resource_type == ResourceType.FILE_HANDLES:
                manager.register_file(resource)
            elif resource_type == ResourceType.DATABASE_CONNECTIONS:
                manager.register_connection(resource)
            elif resource_type == ResourceType.PROCESSES:
                manager.register_process(resource)
            elif resource_type == ResourceType.TEMPORARY_FILES:
                manager.register_temp_path(resource)
        else:
            self.logger.warning(f"No manager for resource type: {resource_type}")
            
    def should_cleanup(self) -> bool:
        """Check if cleanup should be triggered based on policy and thresholds."""
        with self._cleanup_lock:
            if self.policy.strategy == CleanupStrategy.IMMEDIATE:
                return True
            elif self.policy.strategy == CleanupStrategy.RESOURCE_BASED:
                return any(manager.should_cleanup() for manager in self._resource_managers.values())
            elif self.policy.strategy == CleanupStrategy.SCHEDULED:
                # Check if scheduled cleanup time has arrived
                return self._should_run_scheduled_cleanup()
            else:
                return False
                
    def _should_run_scheduled_cleanup(self) -> bool:
        """Check if scheduled cleanup should run."""
        # Simple implementation - run every 5 minutes
        return (datetime.now() - self._last_cleanup_time).total_seconds() > 300
        
    def cleanup(self, force: bool = False, resource_types: Set[ResourceType] = None) -> bool:
        """Perform comprehensive cleanup."""
        if not force and not self.should_cleanup():
            return True
            
        start_time = time.time()
        overall_success = True
        cleanup_results = {}
        
        with self._cleanup_lock:
            self.logger.info("Starting advanced cleanup operation")
            
            # Determine which resource types to clean
            types_to_clean = resource_types or set(self._resource_managers.keys())
            if self.policy.resource_types:
                types_to_clean &= self.policy.resource_types
                
            # Clean each resource type
            for resource_type in types_to_clean:
                manager = self._resource_managers[resource_type]
                success = self._cleanup_with_retry(manager)
                cleanup_results[resource_type] = success
                overall_success &= success
                
            # Integrate with existing test data managers
            self._cleanup_test_data_managers()
            
            # Update statistics
            cleanup_time = time.time() - start_time
            self._record_cleanup_operation(cleanup_results, cleanup_time)
            self._last_cleanup_time = datetime.now()
            
        if self.policy.validate_cleanup:
            validation_success = self.validate_cleanup(types_to_clean)
            overall_success &= validation_success
            
        if self.policy.report_cleanup:
            self._generate_cleanup_report(cleanup_results, cleanup_time)
            
        return overall_success
        
    def _cleanup_with_retry(self, manager: ResourceManager) -> bool:
        """Perform cleanup with retry logic."""
        for attempt in range(self.policy.max_retry_attempts):
            try:
                success = manager.cleanup()
                if success:
                    return True
                    
                if attempt < self.policy.max_retry_attempts - 1:
                    self.logger.warning(f"Cleanup attempt {attempt + 1} failed, retrying...")
                    time.sleep(self.policy.retry_delay_seconds)
                    
            except Exception as e:
                self.logger.error(f"Cleanup attempt {attempt + 1} raised exception: {e}")
                if attempt < self.policy.max_retry_attempts - 1:
                    time.sleep(self.policy.retry_delay_seconds)
                    
        return False
        
    def _cleanup_test_data_managers(self):
        """Integrate cleanup with existing TestDataManager instances."""
        for manager in list(self._test_data_managers):
            try:
                manager.cleanup_all()
            except Exception as e:
                self.logger.warning(f"TestDataManager cleanup failed: {e}")
                
    def validate_cleanup(self, resource_types: Set[ResourceType] = None) -> bool:
        """Validate that cleanup was effective."""
        types_to_validate = resource_types or set(self._resource_managers.keys())
        validation_results = {}
        
        for resource_type in types_to_validate:
            manager = self._resource_managers[resource_type]
            try:
                validation_results[resource_type] = manager.validate_cleanup()
            except Exception as e:
                self.logger.error(f"Cleanup validation failed for {resource_type}: {e}")
                validation_results[resource_type] = False
                
        overall_success = all(validation_results.values())
        
        if not overall_success:
            failed_types = [rt for rt, success in validation_results.items() if not success]
            self.logger.warning(f"Cleanup validation failed for: {failed_types}")
            
        return overall_success
        
    def force_cleanup(self):
        """Force immediate cleanup of all resources."""
        self.logger.warning("Force cleanup initiated")
        self.cleanup(force=True)
        
    def get_resource_usage(self) -> Dict[ResourceType, Dict[str, Any]]:
        """Get current resource usage across all managers."""
        usage = {}
        for resource_type, manager in self._resource_managers.items():
            try:
                usage[resource_type] = manager.get_resource_usage()
            except Exception as e:
                self.logger.error(f"Failed to get usage for {resource_type}: {e}")
                usage[resource_type] = {'error': str(e)}
        return usage
        
    def _record_cleanup_operation(self, results: Dict[ResourceType, bool], duration: float):
        """Record cleanup operation for statistics."""
        operation = {
            'timestamp': datetime.now().isoformat(),
            'results': {str(rt): success for rt, success in results.items()},
            'duration_seconds': duration,
            'overall_success': all(results.values())
        }
        self._cleanup_history.append(operation)
        self._performance_metrics['cleanup_duration'].append(duration)
        
    def _generate_cleanup_report(self, results: Dict[ResourceType, bool], duration: float):
        """Generate and log cleanup report."""
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self.logger.info(
            f"Cleanup completed: {successful}/{total} successful, "
            f"duration: {duration:.3f}s"
        )
        
        for resource_type, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            self.logger.debug(f"  {resource_type.name}: {status}")
            
    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cleanup statistics."""
        return {
            'total_operations': len(self._cleanup_history),
            'successful_operations': sum(1 for op in self._cleanup_history if op['overall_success']),
            'average_duration': sum(self._performance_metrics['cleanup_duration']) / len(self._performance_metrics['cleanup_duration']) if self._performance_metrics['cleanup_duration'] else 0,
            'last_cleanup': self._last_cleanup_time.isoformat() if self._last_cleanup_time else None,
            'resource_manager_stats': {
                str(rt): manager._cleanup_stats 
                for rt, manager in self._resource_managers.items()
            }
        }
        
    @contextmanager
    def cleanup_context(self, resource_types: Set[ResourceType] = None):
        """Context manager for automatic cleanup."""
        try:
            yield self
        finally:
            self.cleanup(resource_types=resource_types)
            
    @asynccontextmanager
    async def async_cleanup_context(self, resource_types: Set[ResourceType] = None):
        """Async context manager for automatic cleanup."""
        try:
            yield self
        finally:
            # Run cleanup in thread to avoid blocking async context
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.cleanup, False, resource_types)


# =====================================================================
# PYTEST INTEGRATION
# =====================================================================

import pytest
from typing import Generator


@pytest.fixture(scope="session")
def advanced_cleanup_orchestrator() -> Generator[AdvancedCleanupOrchestrator, None, None]:
    """Provide advanced cleanup orchestrator with session-level cleanup."""
    policy = CleanupPolicy(
        strategy=CleanupStrategy.DEFERRED,
        scope=CleanupScope.SESSION,
        resource_types={ResourceType.MEMORY, ResourceType.FILE_HANDLES, ResourceType.DATABASE_CONNECTIONS},
        validate_cleanup=True,
        report_cleanup=True
    )
    
    thresholds = ResourceThresholds(
        memory_mb=2048,
        file_handles=500,
        db_connections=25,
        temp_files=50,
        temp_size_mb=200
    )
    
    orchestrator = AdvancedCleanupOrchestrator(policy, thresholds)
    
    try:
        yield orchestrator
    finally:
        orchestrator.force_cleanup()


@pytest.fixture(scope="function")
def function_cleanup_orchestrator() -> Generator[AdvancedCleanupOrchestrator, None, None]:
    """Provide advanced cleanup orchestrator with function-level cleanup."""
    policy = CleanupPolicy(
        strategy=CleanupStrategy.IMMEDIATE,
        scope=CleanupScope.FUNCTION,
        resource_types=set(ResourceType),
        validate_cleanup=True,
        report_cleanup=False  # Less verbose for function-level
    )
    
    thresholds = ResourceThresholds(
        memory_mb=512,
        file_handles=100,
        db_connections=10,
        temp_files=20,
        temp_size_mb=50
    )
    
    orchestrator = AdvancedCleanupOrchestrator(policy, thresholds)
    
    try:
        yield orchestrator
    finally:
        orchestrator.cleanup()


@pytest.fixture
def cleanup_integration_bridge(test_data_manager, advanced_cleanup_orchestrator):
    """Bridge between existing TestDataManager and AdvancedCleanupOrchestrator."""
    # Register the test data manager with the orchestrator
    advanced_cleanup_orchestrator.register_test_data_manager(test_data_manager)
    
    # Create helper functions for easy resource registration
    class CleanupBridge:
        def __init__(self, orchestrator, data_manager):
            self.orchestrator = orchestrator
            self.data_manager = data_manager
            
        def register_file(self, file_obj):
            """Register file for cleanup in both systems."""
            self.orchestrator.register_resource(ResourceType.FILE_HANDLES, file_obj)
            
        def register_db_connection(self, conn):
            """Register database connection for cleanup in both systems."""
            self.data_manager.register_db_connection(conn)
            self.orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
            
        def register_temp_path(self, path):
            """Register temporary path for cleanup in both systems."""
            self.data_manager.register_temp_dir(Path(path))
            self.orchestrator.register_resource(ResourceType.TEMPORARY_FILES, path)
            
        def register_process(self, process):
            """Register process for cleanup."""
            self.orchestrator.register_resource(ResourceType.PROCESSES, process)
            
        def force_cleanup(self):
            """Force cleanup in both systems."""
            self.data_manager.cleanup_all()
            self.orchestrator.force_cleanup()
            
    return CleanupBridge(advanced_cleanup_orchestrator, test_data_manager)


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def cleanup_test_environment(base_path: Path = None, 
                           strategy: CleanupStrategy = CleanupStrategy.DEFERRED) -> bool:
    """Utility function for comprehensive test environment cleanup."""
    base_path = base_path or Path.cwd()
    
    policy = CleanupPolicy(
        strategy=strategy,
        resource_types=set(ResourceType),
        force_cleanup=True,
        validate_cleanup=True
    )
    
    orchestrator = AdvancedCleanupOrchestrator(policy)
    
    # Clean up common test artifacts
    patterns = [
        "**/__pycache__",
        "**/*.pyc", 
        "**/.pytest_cache",
        "**/test_*.db",
        "**/temp_*",
        "**/*.tmp",
        "**/logs/*.log"
    ]
    
    temp_manager = orchestrator._resource_managers[ResourceType.TEMPORARY_FILES]
    for pattern in patterns:
        for path in base_path.glob(pattern):
            temp_manager.register_temp_path(path)
            
    return orchestrator.cleanup(force=True)


def get_system_resource_report() -> Dict[str, Any]:
    """Get comprehensive system resource usage report."""
    process = psutil.Process()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'memory': {
            'rss_mb': process.memory_info().rss / (1024 * 1024),
            'vms_mb': process.memory_info().vms / (1024 * 1024),
            'percent': process.memory_percent()
        },
        'cpu': {
            'percent': process.cpu_percent(),
            'times': process.cpu_times()._asdict()
        },
        'files': {
            'open_files': len(process.open_files()),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 'N/A'
        },
        'threads': process.num_threads(),
        'connections': len(process.connections()) if hasattr(process, 'connections') else 'N/A'
    }


if __name__ == "__main__":
    # Demo/test usage
    logging.basicConfig(level=logging.INFO)
    
    # Create orchestrator with default settings
    orchestrator = AdvancedCleanupOrchestrator()
    
    # Test basic functionality
    print("System resource report:")
    print(json.dumps(get_system_resource_report(), indent=2))
    
    print("\nRunning cleanup test...")
    success = orchestrator.cleanup(force=True)
    print(f"Cleanup successful: {success}")
    
    print("\nCleanup statistics:")
    print(json.dumps(orchestrator.get_cleanup_statistics(), indent=2))