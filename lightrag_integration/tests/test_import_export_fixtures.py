"""
Test Fixtures and Utilities for Import/Export Testing

This module provides comprehensive test fixtures, utilities, and helper functions
for testing import and export functionality of the lightrag_integration module.

Components:
    - Mock factories for dependencies
    - Import isolation utilities
    - Performance measurement tools
    - Error simulation utilities
    - Module state management
    - Test data generators

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import contextlib
import importlib
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from unittest.mock import Mock, MagicMock, patch
import tempfile
import warnings

import pytest


class ImportTestFixtures:
    """Fixtures and utilities for import testing."""
    
    @staticmethod
    @contextlib.contextmanager
    def isolated_import_environment():
        """
        Create an isolated environment for import testing.
        
        This context manager ensures that imports don't interfere with
        each other by preserving and restoring the module cache.
        """
        # Save original module state
        original_modules = sys.modules.copy()
        original_path = sys.path.copy()
        
        try:
            yield
        finally:
            # Restore original state
            # Remove any modules that were imported during the test
            current_modules = set(sys.modules.keys())
            original_module_set = set(original_modules.keys())
            new_modules = current_modules - original_module_set
            
            for module_name in new_modules:
                if module_name.startswith('lightrag_integration'):
                    sys.modules.pop(module_name, None)
            
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)
            sys.path[:] = original_path

    @staticmethod
    @contextlib.contextmanager
    def mock_missing_dependency(dependency_name: str):
        """
        Mock a missing dependency for testing error handling.
        
        Args:
            dependency_name: Name of the dependency to mock as missing
        """
        original_module = sys.modules.get(dependency_name)
        
        # Create a mock that raises ImportError when accessed
        mock_module = Mock()
        mock_module.__name__ = dependency_name
        
        def import_error_side_effect(*args, **kwargs):
            raise ImportError(f"No module named '{dependency_name}'")
        
        with patch.dict('sys.modules', {dependency_name: None}):
            with patch('builtins.__import__', side_effect=import_error_side_effect) as mock_import:
                # Only interfere with the specific dependency
                def selective_import_error(name, *args, **kwargs):
                    if name == dependency_name or name.startswith(f'{dependency_name}.'):
                        raise ImportError(f"No module named '{name}'")
                    return importlib.__import__(name, *args, **kwargs)
                
                mock_import.side_effect = selective_import_error
                
                try:
                    yield
                finally:
                    # Restore original module if it existed
                    if original_module is not None:
                        sys.modules[dependency_name] = original_module

    @staticmethod
    def measure_import_time(module_name: str, iterations: int = 1) -> Dict[str, float]:
        """
        Measure import time for a module.
        
        Args:
            module_name: Name of the module to import
            iterations: Number of times to measure import
            
        Returns:
            Dictionary with timing statistics
        """
        times = []
        
        for i in range(iterations):
            # Clear module from cache for accurate measurement
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            start_time = time.perf_counter()
            try:
                importlib.import_module(module_name)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except ImportError as e:
                return {'error': str(e), 'times': times}
        
        if times:
            return {
                'min_time': min(times),
                'max_time': max(times),
                'avg_time': sum(times) / len(times),
                'total_time': sum(times),
                'times': times
            }
        else:
            return {'error': 'No successful imports', 'times': []}

    @staticmethod
    def create_mock_lightrag_config():
        """Create a mock LightRAGConfig for testing."""
        mock_config = Mock()
        mock_config.get_config.return_value = Mock()
        mock_config.working_dir = '/tmp/test_lightrag'
        mock_config.enable_cost_tracking = True
        mock_config.daily_budget_limit = 100.0
        mock_config.monthly_budget_limit = 1000.0
        return mock_config

    @staticmethod
    def create_mock_rag_system():
        """Create a mock ClinicalMetabolomicsRAG system for testing."""
        mock_rag = Mock()
        mock_rag.initialize_rag = AsyncMock()
        mock_rag.query = AsyncMock(return_value="Mock response")
        mock_rag.set_budget_limits = Mock()
        mock_rag.generate_cost_report = Mock(return_value="Mock cost report")
        return mock_rag

    @staticmethod
    def simulate_import_failure(module_name: str, error_type: Type[Exception] = ImportError):
        """
        Simulate an import failure for testing error handling.
        
        Args:
            module_name: Module to simulate failure for
            error_type: Type of exception to raise
        """
        def mock_import(name, *args, **kwargs):
            if name == module_name or name.startswith(f'{module_name}.'):
                raise error_type(f"Simulated import failure for {name}")
            return importlib.__import__(name, *args, **kwargs)
        
        return patch('builtins.__import__', side_effect=mock_import)


class PerformanceTestUtilities:
    """Utilities for performance testing of imports."""
    
    @staticmethod
    def profile_module_import(module_name: str) -> Dict[str, Any]:
        """
        Profile a module import to identify performance bottlenecks.
        
        Args:
            module_name: Module to profile
            
        Returns:
            Dictionary with profiling information
        """
        import cProfile
        import pstats
        import io
        
        # Clear module from cache
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # Profile the import
        pr = cProfile.Profile()
        pr.enable()
        
        try:
            importlib.import_module(module_name)
            pr.disable()
            
            # Analyze results
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            return {
                'success': True,
                'profile_output': s.getvalue(),
                'total_calls': ps.total_calls
            }
            
        except Exception as e:
            pr.disable()
            return {
                'success': False,
                'error': str(e),
                'profile_output': None
            }

    @staticmethod
    def measure_memory_usage_during_import(module_name: str) -> Dict[str, int]:
        """
        Measure memory usage during module import.
        
        Args:
            module_name: Module to measure
            
        Returns:
            Dictionary with memory usage information
        """
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Measure memory before import
            memory_before = process.memory_info().rss
            
            # Clear module from cache
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Import module
            importlib.import_module(module_name)
            
            # Measure memory after import
            memory_after = process.memory_info().rss
            
            return {
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_after - memory_before,
                'memory_delta_mb': (memory_after - memory_before) / (1024 * 1024)
            }
            
        except ImportError:
            return {'error': 'psutil not available for memory measurement'}
        except Exception as e:
            return {'error': str(e)}


class ModuleStateManager:
    """Utilities for managing module state during testing."""
    
    def __init__(self):
        self.original_modules = {}
        self.original_path = None
        
    def save_state(self):
        """Save current module state."""
        self.original_modules = sys.modules.copy()
        self.original_path = sys.path.copy()
    
    def restore_state(self):
        """Restore saved module state."""
        if self.original_modules:
            sys.modules.clear()
            sys.modules.update(self.original_modules)
        
        if self.original_path:
            sys.path[:] = self.original_path
    
    @contextlib.contextmanager
    def temporary_module_removal(self, module_names: List[str]):
        """Temporarily remove modules from sys.modules."""
        removed_modules = {}
        
        for module_name in module_names:
            if module_name in sys.modules:
                removed_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]
        
        try:
            yield
        finally:
            # Restore removed modules
            sys.modules.update(removed_modules)
    
    @contextlib.contextmanager
    def clean_import_state(self, target_package: str = 'lightrag_integration'):
        """Clean import state for a specific package."""
        self.save_state()
        
        # Remove all modules related to target package
        modules_to_remove = [
            name for name in sys.modules.keys() 
            if name.startswith(target_package)
        ]
        
        for module_name in modules_to_remove:
            del sys.modules[module_name]
        
        try:
            yield
        finally:
            self.restore_state()


class DependencyMockFactory:
    """Factory for creating mock dependencies."""
    
    @staticmethod
    def create_mock_openai():
        """Create mock OpenAI client."""
        mock_openai = Mock()
        mock_openai.Completion.create.return_value = Mock(
            choices=[Mock(text="Mock completion")]
        )
        return mock_openai
    
    @staticmethod
    def create_mock_lightrag():
        """Create mock LightRAG components."""
        mock_lightrag = Mock()
        mock_lightrag.LightRAG = Mock()
        mock_lightrag.QueryParam = Mock()
        return mock_lightrag
    
    @staticmethod
    def create_mock_pandas():
        """Create mock pandas."""
        mock_pd = Mock()
        mock_pd.DataFrame = Mock()
        mock_pd.read_csv = Mock()
        return mock_pd
    
    @staticmethod
    def create_mock_numpy():
        """Create mock numpy."""
        mock_np = Mock()
        mock_np.array = Mock()
        mock_np.mean = Mock()
        return mock_np
    
    @staticmethod
    def mock_all_dependencies():
        """Create mocks for all common dependencies."""
        return {
            'openai': DependencyMockFactory.create_mock_openai(),
            'lightrag': DependencyMockFactory.create_mock_lightrag(),
            'pandas': DependencyMockFactory.create_mock_pandas(),
            'numpy': DependencyMockFactory.create_mock_numpy()
        }


class CircularImportDetector:
    """Utilities for detecting circular imports."""
    
    def __init__(self):
        self.import_stack = []
        self.visited = set()
        self.circular_imports = []
    
    def detect_circular_imports(self, module_name: str, max_depth: int = 10) -> List[List[str]]:
        """
        Detect circular imports starting from a module.
        
        Args:
            module_name: Starting module name
            max_depth: Maximum depth to search
            
        Returns:
            List of circular import chains found
        """
        self.import_stack = []
        self.visited = set()
        self.circular_imports = []
        
        self._check_module_imports(module_name, max_depth)
        return self.circular_imports
    
    def _check_module_imports(self, module_name: str, max_depth: int):
        """Recursively check module imports."""
        if max_depth <= 0:
            return
        
        if module_name in self.import_stack:
            # Circular import detected
            cycle_start = self.import_stack.index(module_name)
            cycle = self.import_stack[cycle_start:] + [module_name]
            self.circular_imports.append(cycle)
            return
        
        if module_name in self.visited:
            return
        
        self.visited.add(module_name)
        self.import_stack.append(module_name)
        
        try:
            # Import the module to analyze its dependencies
            module = importlib.import_module(module_name)
            
            # Find imported modules
            imported_modules = self._find_imported_modules(module)
            
            for imported_module in imported_modules:
                if imported_module.startswith('lightrag_integration'):
                    self._check_module_imports(imported_module, max_depth - 1)
                    
        except ImportError:
            pass  # Skip modules that can't be imported
        finally:
            self.import_stack.pop()
    
    def _find_imported_modules(self, module) -> List[str]:
        """Find modules imported by a given module."""
        imported_modules = []
        
        # Check attributes that might be imported modules
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                if hasattr(attr, '__module__'):
                    module_name = attr.__module__
                    if module_name and module_name not in imported_modules:
                        imported_modules.append(module_name)
        
        return imported_modules


class AsyncImportTester:
    """Utilities for testing imports in async contexts."""
    
    @staticmethod
    async def async_import_test(module_name: str) -> Dict[str, Any]:
        """Test importing a module in an async context."""
        try:
            start_time = time.perf_counter()
            
            # Import in async context
            module = importlib.import_module(module_name)
            
            end_time = time.perf_counter()
            
            return {
                'success': True,
                'module': module,
                'import_time': end_time - start_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'module': None,
                'import_time': None,
                'error': str(e)
            }
    
    @staticmethod
    async def test_async_compatibility(module_name: str) -> bool:
        """Test if a module is compatible with async environments."""
        try:
            # Import module
            module = importlib.import_module(module_name)
            
            # Check if we can access attributes in async context
            if hasattr(module, '__all__'):
                all_list = module.__all__
                for attr_name in all_list[:5]:  # Check first 5 attributes
                    if hasattr(module, attr_name):
                        getattr(module, attr_name)
            
            # Small delay to test async behavior
            await asyncio.sleep(0.001)
            
            return True
            
        except Exception:
            return False


# Async mock helper
class AsyncMock(MagicMock):
    """Mock class that supports async methods."""
    
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


# Pytest fixtures for import/export testing
@pytest.fixture
def import_fixtures():
    """Provide import test fixtures."""
    return ImportTestFixtures()


@pytest.fixture
def performance_utils():
    """Provide performance testing utilities."""
    return PerformanceTestUtilities()


@pytest.fixture
def module_state_manager():
    """Provide module state management utilities."""
    return ModuleStateManager()


@pytest.fixture
def dependency_mocks():
    """Provide mock dependencies."""
    return DependencyMockFactory()


@pytest.fixture
def circular_import_detector():
    """Provide circular import detection utilities."""
    return CircularImportDetector()


@pytest.fixture
def async_import_tester():
    """Provide async import testing utilities."""
    return AsyncImportTester()


@pytest.fixture
def temp_module_dir():
    """Create temporary directory for module testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Add temp directory to Python path
        original_path = sys.path.copy()
        sys.path.insert(0, temp_dir)
        
        try:
            yield Path(temp_dir)
        finally:
            # Restore original path
            sys.path[:] = original_path


@pytest.fixture(scope="session")
def lightrag_integration_module():
    """Session-scoped fixture providing the lightrag_integration module."""
    return importlib.import_module('lightrag_integration')


if __name__ == "__main__":
    # Demo usage of the fixtures
    print("Import/Export Test Fixtures and Utilities")
    print("==========================================")
    
    # Test import time measurement
    fixtures = ImportTestFixtures()
    timing = fixtures.measure_import_time('lightrag_integration')
    print(f"Import timing: {timing}")
    
    # Test circular import detection
    detector = CircularImportDetector()
    circular = detector.detect_circular_imports('lightrag_integration')
    print(f"Circular imports: {circular}")
    
    print("Fixtures ready for use in tests!")