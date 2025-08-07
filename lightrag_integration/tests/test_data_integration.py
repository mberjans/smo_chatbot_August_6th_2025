#!/usr/bin/env python3
"""
Test Data Integration Module for Clinical Metabolomics Oracle.

This module provides seamless integration between the new test data fixtures
and the existing pytest infrastructure. It ensures compatibility and provides
migration utilities for existing tests.

Key Features:
1. Backward compatibility with existing fixtures
2. Integration adapters for current test patterns
3. Migration utilities for upgrading existing tests
4. Performance optimizations for test data loading
5. Async integration with LightRAG components
6. Comprehensive error handling and recovery

Components:
- FixtureIntegrator: Bridges new and old fixture systems
- TestDataOrchestrator: Coordinates complex test data scenarios
- AsyncTestDataManager: Async support for LightRAG integration
- PerformanceOptimizer: Optimizes test data loading and cleanup
- MigrationHelper: Assists in upgrading existing tests

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import pytest_asyncio
import asyncio
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, AsyncGenerator, Generator, Type
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc
from datetime import datetime
import tempfile
import shutil
import sqlite3
import json

# Import our fixtures and utilities
from .test_data_fixtures import (
    TestDataManager, TestDataConfig, TestDataInfo,
    TEST_DATA_ROOT, PDF_DATA_DIR, DATABASE_DATA_DIR, MOCK_DATA_DIR
)
from .test_data_utilities import (
    TestDataFactory, DataValidationSuite, MockDataGenerator,
    BiochemicalCompound, ClinicalStudyData, TestScenario
)

# Import existing fixture systems for integration
try:
    from .comprehensive_test_fixtures import EnhancedPDFCreator
    COMPREHENSIVE_FIXTURES_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_FIXTURES_AVAILABLE = False
    logging.warning("Comprehensive test fixtures not available")

try:
    from .biomedical_test_fixtures import ClinicalMetabolomicsDataGenerator
    BIOMEDICAL_FIXTURES_AVAILABLE = True
except ImportError:
    BIOMEDICAL_FIXTURES_AVAILABLE = False
    logging.warning("Biomedical test fixtures not available")


# =====================================================================
# INTEGRATION CONFIGURATION
# =====================================================================

@dataclass
class IntegrationConfig:
    """Configuration for test data integration."""
    enable_performance_monitoring: bool = False
    async_pool_size: int = 4
    cache_test_data: bool = True
    validate_on_load: bool = True
    cleanup_frequency: str = "per_test"  # "per_test", "per_session", "manual"
    max_memory_usage_mb: int = 500
    enable_legacy_compatibility: bool = True
    migration_mode: bool = False


# =====================================================================
# FIXTURE INTEGRATOR
# =====================================================================

class FixtureIntegrator:
    """Bridges new test data fixtures with existing infrastructure."""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self._fixture_cache = {}
        self._active_managers = weakref.WeakSet()
        self._lock = threading.Lock()
        
    def create_integrated_manager(self, test_name: str) -> TestDataManager:
        """Create test data manager integrated with existing fixtures."""
        with self._lock:
            if test_name in self._fixture_cache and self.config.cache_test_data:
                return self._fixture_cache[test_name]
            
            # Create new manager with integration-specific config
            test_config = TestDataConfig(
                use_temp_dirs=True,
                auto_cleanup=(self.config.cleanup_frequency == "per_test"),
                validate_data=self.config.validate_on_load,
                async_support=True,
                performance_monitoring=self.config.enable_performance_monitoring
            )
            
            manager = TestDataManager(test_config)
            self._active_managers.add(manager)
            
            if self.config.cache_test_data:
                self._fixture_cache[test_name] = manager
                
            return manager
    
    def integrate_with_existing_fixtures(self, test_function: Callable) -> Dict[str, Any]:
        """Extract and integrate existing fixture dependencies."""
        integration_data = {
            "requires_pdf_creation": False,
            "requires_biomedical_data": False,
            "requires_async_support": False,
            "performance_critical": False,
            "fixtures_needed": []
        }
        
        # Analyze function for fixture dependencies
        if hasattr(test_function, 'pytestmark'):
            for mark in test_function.pytestmark:
                if mark.name == 'asyncio':
                    integration_data["requires_async_support"] = True
                elif mark.name == 'performance':
                    integration_data["performance_critical"] = True
        
        # Check function signature for known fixture patterns
        import inspect
        signature = inspect.signature(test_function)
        
        for param_name in signature.parameters:
            if 'pdf' in param_name.lower():
                integration_data["requires_pdf_creation"] = True
                integration_data["fixtures_needed"].append(param_name)
            elif any(term in param_name.lower() for term in ['biomedical', 'metabol', 'compound']):
                integration_data["requires_biomedical_data"] = True
                integration_data["fixtures_needed"].append(param_name)
            elif 'async' in param_name.lower():
                integration_data["requires_async_support"] = True
                integration_data["fixtures_needed"].append(param_name)
        
        return integration_data
    
    def cleanup_cached_fixtures(self) -> None:
        """Clean up cached fixtures and active managers."""
        with self._lock:
            for manager in self._active_managers:
                try:
                    manager.cleanup_all()
                except Exception as e:
                    logging.warning(f"Manager cleanup failed: {e}")
            
            self._fixture_cache.clear()
            # Force garbage collection
            gc.collect()


# =====================================================================
# ASYNC TEST DATA MANAGER  
# =====================================================================

class AsyncTestDataManager:
    """Async-compatible test data manager for LightRAG integration."""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.async_pool_size)
        self._data_cache = {}
        self._loading_locks = {}
        
    async def load_test_data_async(self, 
                                  data_type: str, 
                                  data_key: str,
                                  loader_func: Callable) -> Any:
        """Asynchronously load test data with caching."""
        cache_key = f"{data_type}:{data_key}"
        
        # Check cache first
        if cache_key in self._data_cache and self.config.cache_test_data:
            return self._data_cache[cache_key]
        
        # Prevent duplicate loading
        if cache_key not in self._loading_locks:
            self._loading_locks[cache_key] = asyncio.Lock()
        
        async with self._loading_locks[cache_key]:
            # Double-check cache after acquiring lock
            if cache_key in self._data_cache:
                return self._data_cache[cache_key]
            
            # Load data asynchronously
            loop = asyncio.get_event_loop()
            try:
                data = await loop.run_in_executor(self.executor, loader_func)
                
                if self.config.cache_test_data:
                    self._data_cache[cache_key] = data
                
                return data
            except Exception as e:
                logging.error(f"Async data loading failed for {cache_key}: {e}")
                raise
    
    async def create_async_test_database(self, schema_sql: str) -> sqlite3.Connection:
        """Create test database asynchronously."""
        loop = asyncio.get_event_loop()
        
        def _create_db():
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
                db_path = f.name
            
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.executescript(schema_sql)
            conn.commit()
            return conn, db_path
        
        conn, db_path = await loop.run_in_executor(self.executor, _create_db)
        
        # Store for cleanup
        self._temp_databases = getattr(self, '_temp_databases', [])
        self._temp_databases.append((conn, Path(db_path)))
        
        return conn
    
    async def generate_test_content_async(self,
                                        content_type: str,
                                        count: int = 1) -> List[Any]:
        """Generate test content asynchronously."""
        loop = asyncio.get_event_loop()
        factory = TestDataFactory()
        
        def _generate_content():
            if content_type == "compounds":
                return [factory.generate_compound() for _ in range(count)]
            elif content_type == "studies":
                return [factory.generate_clinical_study() for _ in range(count)]
            elif content_type == "mock_responses":
                generator = MockDataGenerator()
                return [generator.generate_api_response_mock("openai_chat") for _ in range(count)]
            else:
                return [f"Generated {content_type} content {i+1}" for i in range(count)]
        
        return await loop.run_in_executor(self.executor, _generate_content)
    
    async def cleanup_async(self) -> None:
        """Asynchronously clean up resources."""
        tasks = []
        
        # Cleanup temp databases
        if hasattr(self, '_temp_databases'):
            for conn, db_path in self._temp_databases:
                task = asyncio.create_task(self._cleanup_database(conn, db_path))
                tasks.append(task)
        
        # Wait for all cleanup tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear caches
        self._data_cache.clear()
        self._loading_locks.clear()
    
    async def _cleanup_database(self, conn: sqlite3.Connection, db_path: Path) -> None:
        """Clean up individual database asynchronously."""
        loop = asyncio.get_event_loop()
        
        def _cleanup():
            try:
                conn.close()
                db_path.unlink(missing_ok=True)
            except Exception as e:
                logging.warning(f"Database cleanup failed: {e}")
        
        await loop.run_in_executor(self.executor, _cleanup)


# =====================================================================
# PERFORMANCE OPTIMIZER
# =====================================================================

class PerformanceOptimizer:
    """Optimizes test data loading and management for performance."""
    
    def __init__(self):
        self.load_times = {}
        self.memory_usage = {}
        self.optimization_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_optimizations": 0,
            "load_time_improvements": 0
        }
        
    def profile_data_loading(self, data_key: str, loader_func: Callable) -> Callable:
        """Profile and optimize data loading function."""
        def profiled_loader(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = loader_func(*args, **kwargs)
                
                load_time = time.time() - start_time
                memory_used = self._get_memory_usage() - start_memory
                
                self.load_times[data_key] = load_time
                self.memory_usage[data_key] = memory_used
                
                # Log performance metrics
                logging.debug(f"Data loading profile for {data_key}: "
                            f"{load_time:.3f}s, {memory_used:.2f}MB")
                
                return result
                
            except Exception as e:
                logging.error(f"Profiled data loading failed for {data_key}: {e}")
                raise
        
        return profiled_loader
    
    def optimize_fixture_scope(self, fixture_name: str, usage_pattern: str) -> str:
        """Recommend optimal fixture scope based on usage pattern."""
        if usage_pattern == "single_test":
            return "function"
        elif usage_pattern == "test_class":
            return "class"
        elif usage_pattern == "test_module":
            return "module"
        elif usage_pattern == "all_tests":
            return "session"
        else:
            # Analyze based on fixture name patterns
            if "temp" in fixture_name.lower() or "unique" in fixture_name.lower():
                return "function"
            elif "sample" in fixture_name.lower() or "mock" in fixture_name.lower():
                return "session"
            else:
                return "module"  # Default
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance optimization report."""
        total_load_time = sum(self.load_times.values())
        total_memory = sum(self.memory_usage.values())
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_data_items": len(self.load_times),
            "total_load_time_seconds": round(total_load_time, 3),
            "total_memory_usage_mb": round(total_memory, 2),
            "average_load_time": round(total_load_time / len(self.load_times), 3) if self.load_times else 0,
            "optimization_stats": self.optimization_stats.copy(),
            "slowest_loads": sorted(
                [(k, v) for k, v in self.load_times.items()], 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            "memory_intensive_loads": sorted(
                [(k, v) for k, v in self.memory_usage.items()], 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }


# =====================================================================
# INTEGRATION FIXTURES
# =====================================================================

@pytest.fixture(scope="session")
def integration_config() -> IntegrationConfig:
    """Provide integration configuration."""
    return IntegrationConfig()


@pytest.fixture(scope="session") 
def fixture_integrator(integration_config: IntegrationConfig) -> Generator[FixtureIntegrator, None, None]:
    """Provide fixture integrator with cleanup."""
    integrator = FixtureIntegrator(integration_config)
    try:
        yield integrator
    finally:
        integrator.cleanup_cached_fixtures()


@pytest.fixture
def integrated_test_data_manager(
    fixture_integrator: FixtureIntegrator,
    request
) -> TestDataManager:
    """Provide integrated test data manager for current test."""
    test_name = request.node.name
    return fixture_integrator.create_integrated_manager(test_name)


@pytest_asyncio.fixture
async def async_test_data_manager(integration_config: IntegrationConfig) -> AsyncGenerator[AsyncTestDataManager, None]:
    """Provide async test data manager."""
    manager = AsyncTestDataManager(integration_config)
    try:
        yield manager
    finally:
        await manager.cleanup_async()


@pytest.fixture
def performance_optimizer() -> PerformanceOptimizer:
    """Provide performance optimizer for test data operations."""
    return PerformanceOptimizer()


# =====================================================================
# ENHANCED INTEGRATION FIXTURES
# =====================================================================

@pytest.fixture
def enhanced_pdf_data(integrated_test_data_manager: TestDataManager) -> Dict[str, Any]:
    """Enhanced PDF data with metadata and validation."""
    factory = TestDataFactory()
    
    # Generate multiple study types
    studies = {
        "diabetes_study": factory.generate_clinical_study(condition="Type 2 Diabetes"),
        "cancer_study": factory.generate_clinical_study(condition="Cancer"),
        "cardiovascular_study": factory.generate_clinical_study(condition="Cardiovascular Disease")
    }
    
    pdf_data = {}
    for study_name, study in studies.items():
        pdf_content = study.to_research_paper()
        pdf_data[study_name] = {
            "content": pdf_content,
            "metadata": {
                "study_id": study.study_id,
                "title": study.title,
                "sample_size": study.sample_size,
                "duration_months": study.duration_months,
                "compounds_studied": study.compounds_studied
            },
            "checksum": hashlib.sha256(pdf_content.encode()).hexdigest(),
            "size_bytes": len(pdf_content.encode())
        }
        
        # Register with manager
        integrated_test_data_manager.loaded_data[f"pdf_{study_name}"] = TestDataInfo(
            data_type="enhanced_pdf",
            source_path=Path(f"generated_{study_name}"),
            loaded_at=datetime.now(),
            size_bytes=len(pdf_content.encode()),
            metadata=pdf_data[study_name]["metadata"]
        )
    
    return pdf_data


@pytest.fixture
def comprehensive_mock_data(integrated_test_data_manager: TestDataManager) -> Dict[str, Any]:
    """Comprehensive mock data for all test scenarios."""
    generator = MockDataGenerator()
    
    mock_data = {
        "api_responses": {
            "success": generator.generate_api_response_mock("openai_chat", success=True),
            "failure": generator.generate_api_response_mock("openai_chat", success=False),
            "embedding_success": generator.generate_api_response_mock("embedding", success=True),
            "embedding_failure": generator.generate_api_response_mock("embedding", success=False)
        },
        "system_states": {
            "cost_monitor_healthy": generator.generate_system_state_mock("cost_monitor", healthy=True),
            "cost_monitor_exceeded": generator.generate_system_state_mock("cost_monitor", healthy=False),
            "lightrag_healthy": generator.generate_system_state_mock("lightrag_system", healthy=True),
            "lightrag_degraded": generator.generate_system_state_mock("lightrag_system", healthy=False)
        },
        "performance_data": {
            "normal_load": generator.generate_performance_test_data("normal_load"),
            "high_load": generator.generate_performance_test_data("high_load"),
            "stress_test": generator.generate_performance_test_data("stress_test")
        }
    }
    
    return mock_data


@pytest.fixture
async def async_biomedical_data(async_test_data_manager: AsyncTestDataManager) -> Dict[str, Any]:
    """Asynchronously loaded comprehensive biomedical data."""
    
    # Load different types of content asynchronously
    compounds_task = async_test_data_manager.generate_test_content_async("compounds", count=10)
    studies_task = async_test_data_manager.generate_test_content_async("studies", count=5)
    mock_responses_task = async_test_data_manager.generate_test_content_async("mock_responses", count=3)
    
    compounds, studies, mock_responses = await asyncio.gather(
        compounds_task, studies_task, mock_responses_task
    )
    
    return {
        "compounds": [comp.to_dict() for comp in compounds],
        "studies": [study.__dict__ for study in studies],
        "mock_responses": mock_responses,
        "metadata": {
            "loaded_at": datetime.now().isoformat(),
            "load_method": "async",
            "total_items": len(compounds) + len(studies) + len(mock_responses)
        }
    }


# =====================================================================
# MIGRATION HELPERS
# =====================================================================

def migrate_existing_fixture(old_fixture_name: str, 
                           test_data_manager: TestDataManager) -> Callable:
    """Helper to migrate existing fixtures to new system."""
    
    def migration_wrapper(original_fixture_func: Callable) -> Callable:
        def migrated_fixture(*args, **kwargs):
            # Log migration
            logging.info(f"Migrating fixture {old_fixture_name} to new test data system")
            
            # Call original fixture
            original_result = original_fixture_func(*args, **kwargs)
            
            # Enhance with new capabilities
            if isinstance(original_result, dict):
                original_result["_migrated"] = True
                original_result["_migration_timestamp"] = datetime.now().isoformat()
                original_result["_test_data_manager"] = test_data_manager
            
            return original_result
        
        return migrated_fixture
    return migration_wrapper


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def validate_integration_compatibility() -> Dict[str, bool]:
    """Check compatibility with existing fixture systems."""
    compatibility = {
        "comprehensive_fixtures": COMPREHENSIVE_FIXTURES_AVAILABLE,
        "biomedical_fixtures": BIOMEDICAL_FIXTURES_AVAILABLE,
        "pytest_asyncio": True,  # Should always be available
        "sqlite3": True,  # Built-in to Python
        "json_support": True,  # Built-in
        "pathlib_support": True,  # Built-in
        "threading_support": True,  # Built-in
    }
    
    return compatibility


def create_integration_test_scenario(name: str, **kwargs) -> Dict[str, Any]:
    """Create test scenario for integration testing."""
    return {
        "scenario_name": name,
        "timestamp": datetime.now().isoformat(),
        "configuration": kwargs,
        "compatibility_check": validate_integration_compatibility(),
        "requirements": {
            "pytest_version": pytest.__version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    }


# Import hashlib for checksum calculations
import hashlib
import sys