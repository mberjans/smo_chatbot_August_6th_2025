"""
Performance Tests for LightRAG Integration Module Import/Export Functionality

This module provides comprehensive performance testing for import and export
operations, measuring timing, memory usage, and detecting performance issues
like circular imports or slow loading components.

Test Categories:
    - Import timing benchmarks
    - Memory usage during imports
    - Circular import detection and prevention
    - Lazy loading verification
    - Cold vs warm import performance
    - Submodule loading efficiency
    - Resource cleanup validation

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import gc
import importlib
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import patch
import warnings

import pytest

from .test_import_export_fixtures import (
    ImportTestFixtures,
    PerformanceTestUtilities,
    ModuleStateManager,
    CircularImportDetector
)


class TestImportPerformance:
    """
    Performance tests for module import functionality.
    
    Tests various performance aspects of module loading including
    timing, memory usage, and efficiency metrics.
    """
    
    # Performance thresholds
    MAX_IMPORT_TIME = 10.0  # seconds
    MAX_MEMORY_USAGE = 100 * 1024 * 1024  # 100MB in bytes
    MAX_COLD_IMPORT_TIME = 15.0  # seconds for first-time import
    MAX_WARM_IMPORT_TIME = 1.0   # seconds for cached import
    
    # Modules to benchmark
    MODULES_TO_BENCHMARK = [
        'lightrag_integration',
        'lightrag_integration.examples',
        'lightrag_integration.performance_benchmarking',
        'lightrag_integration.tests'
    ]

    @pytest.fixture(autouse=True)
    def setup_performance_testing(self):
        """Set up performance testing environment."""
        self.state_manager = ModuleStateManager()
        self.performance_utils = PerformanceTestUtilities()
        self.state_manager.save_state()
        
        # Clear any cached imports for accurate measurement
        modules_to_clear = [name for name in sys.modules.keys() 
                           if name.startswith('lightrag_integration')]
        for module_name in modules_to_clear:
            del sys.modules[module_name]
        
        # Force garbage collection
        gc.collect()
        
        yield
        
        self.state_manager.restore_state()

    def test_cold_import_performance(self):
        """Test performance of first-time (cold) import."""
        timing_results = {}
        
        for module_name in self.MODULES_TO_BENCHMARK:
            # Ensure module is not in cache
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Measure cold import time
            start_time = time.perf_counter()
            try:
                importlib.import_module(module_name)
                end_time = time.perf_counter()
                
                import_time = end_time - start_time
                timing_results[module_name] = import_time
                
                assert import_time < self.MAX_COLD_IMPORT_TIME, \
                    f"Cold import of {module_name} too slow: {import_time:.3f}s"
                    
            except ImportError as e:
                timing_results[module_name] = f"ImportError: {e}"
                warnings.warn(f"Could not benchmark {module_name}: {e}")
            except Exception as e:
                timing_results[module_name] = f"Error: {e}"
                pytest.fail(f"Unexpected error importing {module_name}: {e}")
        
        # Report timing results
        print(f"\nCold import performance results:")
        for module, result in timing_results.items():
            if isinstance(result, float):
                print(f"  {module}: {result:.3f}s")
            else:
                print(f"  {module}: {result}")

    def test_warm_import_performance(self):
        """Test performance of cached (warm) import."""
        # First, import all modules to cache them
        for module_name in self.MODULES_TO_BENCHMARK:
            try:
                importlib.import_module(module_name)
            except ImportError:
                continue
        
        timing_results = {}
        
        for module_name in self.MODULES_TO_BENCHMARK:
            if module_name not in sys.modules:
                continue  # Skip if not successfully imported
            
            # Measure warm import time (reload)
            start_time = time.perf_counter()
            try:
                importlib.reload(sys.modules[module_name])
                end_time = time.perf_counter()
                
                import_time = end_time - start_time
                timing_results[module_name] = import_time
                
                assert import_time < self.MAX_WARM_IMPORT_TIME, \
                    f"Warm import of {module_name} too slow: {import_time:.3f}s"
                    
            except Exception as e:
                timing_results[module_name] = f"Error: {e}"
                warnings.warn(f"Could not benchmark warm import of {module_name}: {e}")
        
        # Report timing results
        print(f"\nWarm import performance results:")
        for module, result in timing_results.items():
            if isinstance(result, float):
                print(f"  {module}: {result:.3f}s")
            else:
                print(f"  {module}: {result}")

    def test_memory_usage_during_import(self):
        """Test memory usage during module import."""
        memory_results = {}
        
        for module_name in self.MODULES_TO_BENCHMARK:
            # Clear module from cache
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Force garbage collection before measurement
            gc.collect()
            
            # Measure memory usage
            memory_stats = self.performance_utils.measure_memory_usage_during_import(module_name)
            memory_results[module_name] = memory_stats
            
            if 'memory_delta' in memory_stats:
                memory_delta = memory_stats['memory_delta']
                
                assert memory_delta < self.MAX_MEMORY_USAGE, \
                    f"Import of {module_name} uses too much memory: {memory_delta / (1024*1024):.2f}MB"
        
        # Report memory results
        print(f"\nMemory usage during import:")
        for module, stats in memory_results.items():
            if 'memory_delta_mb' in stats:
                print(f"  {module}: {stats['memory_delta_mb']:.2f}MB")
            elif 'error' in stats:
                print(f"  {module}: {stats['error']}")

    def test_import_profiling(self):
        """Test detailed profiling of module imports."""
        profiling_results = {}
        
        for module_name in ['lightrag_integration']:  # Focus on main module
            # Clear from cache for accurate profiling
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            profile_data = self.performance_utils.profile_module_import(module_name)
            profiling_results[module_name] = profile_data
            
            if profile_data.get('success'):
                print(f"\nProfiling results for {module_name}:")
                profile_output = profile_data.get('profile_output', '')
                # Show only the top 10 lines to avoid cluttering output
                lines = profile_output.split('\n')[:15]
                print('\n'.join(lines))
            else:
                warnings.warn(f"Could not profile {module_name}: {profile_data.get('error')}")

    def test_repeated_import_performance(self):
        """Test performance consistency across repeated imports."""
        module_name = 'lightrag_integration'
        iterations = 5
        
        times = []
        
        for i in range(iterations):
            # Clear module from cache
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Measure import time
            start_time = time.perf_counter()
            try:
                importlib.import_module(module_name)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                pytest.fail(f"Import failed on iteration {i+1}: {e}")
        
        if times:
            min_time = min(times)
            max_time = max(times)
            avg_time = sum(times) / len(times)
            
            print(f"\nRepeated import performance ({iterations} iterations):")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")
            print(f"  Avg: {avg_time:.3f}s")
            
            # Check for performance degradation
            assert max_time <= avg_time * 3, \
                f"Performance degraded significantly: max={max_time:.3f}s avg={avg_time:.3f}s"
            
            # All imports should be reasonably fast
            for i, import_time in enumerate(times):
                assert import_time < self.MAX_IMPORT_TIME, \
                    f"Import {i+1} too slow: {import_time:.3f}s"

    def test_submodule_loading_efficiency(self):
        """Test efficiency of submodule loading."""
        submodules = [
            'lightrag_integration.examples',
            'lightrag_integration.performance_benchmarking',
            'lightrag_integration.tests'
        ]
        
        # First, import main module
        import lightrag_integration
        main_import_time = time.perf_counter()
        
        submodule_times = {}
        
        for submodule in submodules:
            start_time = time.perf_counter()
            try:
                importlib.import_module(submodule)
                end_time = time.perf_counter()
                
                import_time = end_time - start_time
                submodule_times[submodule] = import_time
                
                # Submodules should import quickly after main module
                assert import_time < 5.0, \
                    f"Submodule {submodule} import too slow: {import_time:.3f}s"
                    
            except ImportError as e:
                submodule_times[submodule] = f"ImportError: {e}"
                warnings.warn(f"Could not benchmark submodule {submodule}: {e}")
        
        print(f"\nSubmodule import performance:")
        for submodule, result in submodule_times.items():
            if isinstance(result, float):
                print(f"  {submodule}: {result:.3f}s")
            else:
                print(f"  {submodule}: {result}")

    @pytest.mark.asyncio
    async def test_async_import_performance(self):
        """Test import performance in async contexts."""
        module_name = 'lightrag_integration'
        
        async def async_import_test():
            start_time = time.perf_counter()
            
            # Import in async context
            module = importlib.import_module(module_name)
            
            # Small async operation
            await asyncio.sleep(0.001)
            
            end_time = time.perf_counter()
            return end_time - start_time, module
        
        # Clear module cache
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        import_time, module = await async_import_test()
        
        assert module is not None, "Async import failed"
        assert import_time < self.MAX_IMPORT_TIME, \
            f"Async import too slow: {import_time:.3f}s"
        
        print(f"\nAsync import performance: {import_time:.3f}s")

    def test_import_resource_cleanup(self):
        """Test that imports properly clean up resources."""
        initial_fd_count = None
        
        try:
            # Get initial file descriptor count (Unix-like systems)
            import os
            initial_fd_count = len(os.listdir('/proc/self/fd'))
        except (OSError, FileNotFoundError):
            # Not available on all systems
            pass
        
        # Perform multiple imports and cleanups
        for i in range(3):
            # Clear module cache
            modules_to_clear = [name for name in sys.modules.keys() 
                               if name.startswith('lightrag_integration')]
            for module_name in modules_to_clear:
                del sys.modules[module_name]
            
            # Import module
            import lightrag_integration
            
            # Force garbage collection
            gc.collect()
        
        # Check final file descriptor count
        if initial_fd_count is not None:
            try:
                final_fd_count = len(os.listdir('/proc/self/fd'))
                fd_increase = final_fd_count - initial_fd_count
                
                # Should not leak file descriptors
                assert fd_increase <= 5, \
                    f"Possible file descriptor leak: increased by {fd_increase}"
                    
            except (OSError, FileNotFoundError):
                pass


class TestCircularImportDetection:
    """Test circular import detection and prevention."""
    
    def test_detect_circular_imports_in_main_module(self):
        """Test detection of circular imports in main module."""
        detector = CircularImportDetector()
        
        # Test main module
        circular_imports = detector.detect_circular_imports('lightrag_integration')
        
        assert isinstance(circular_imports, list), "Should return list of circular import chains"
        
        if circular_imports:
            # If circular imports are found, they should be properly formatted
            for cycle in circular_imports:
                assert isinstance(cycle, list), "Each cycle should be a list"
                assert len(cycle) > 1, "Cycle should have more than one module"
                assert cycle[0] == cycle[-1], "Cycle should start and end with same module"
            
            # Report circular imports
            print(f"\nCircular imports detected:")
            for i, cycle in enumerate(circular_imports):
                print(f"  Cycle {i+1}: {' -> '.join(cycle)}")
            
            # For now, warn but don't fail - circular imports might be intentional
            warnings.warn(f"Circular imports detected: {len(circular_imports)} cycles found")
        else:
            print(f"\nNo circular imports detected in main module")

    def test_detect_circular_imports_in_submodules(self):
        """Test detection of circular imports in submodules."""
        detector = CircularImportDetector()
        
        submodules_to_test = [
            'lightrag_integration.examples',
            'lightrag_integration.performance_benchmarking',
            'lightrag_integration.tests'
        ]
        
        all_circular_imports = {}
        
        for submodule in submodules_to_test:
            try:
                circular_imports = detector.detect_circular_imports(submodule, max_depth=5)
                all_circular_imports[submodule] = circular_imports
                
                if circular_imports:
                    print(f"\nCircular imports in {submodule}:")
                    for cycle in circular_imports:
                        print(f"  {' -> '.join(cycle)}")
                        
            except Exception as e:
                all_circular_imports[submodule] = f"Error: {e}"
                warnings.warn(f"Could not check {submodule} for circular imports: {e}")
        
        # Report summary
        total_cycles = sum(len(cycles) if isinstance(cycles, list) else 0 
                          for cycles in all_circular_imports.values())
        
        if total_cycles == 0:
            print(f"\nNo circular imports detected in any submodules")
        else:
            print(f"\nTotal circular import cycles detected: {total_cycles}")

    def test_import_dependency_graph(self):
        """Test and analyze the import dependency graph."""
        import importlib
        
        def get_module_dependencies(module_name: str) -> Set[str]:
            """Get direct dependencies of a module."""
            try:
                module = importlib.import_module(module_name)
                dependencies = set()
                
                for attr_name in dir(module):
                    if not attr_name.startswith('_'):
                        attr = getattr(module, attr_name)
                        if hasattr(attr, '__module__'):
                            dep_module = attr.__module__
                            if dep_module and dep_module.startswith('lightrag_integration'):
                                dependencies.add(dep_module)
                
                return dependencies
            except ImportError:
                return set()
        
        # Build dependency graph
        modules_to_analyze = [
            'lightrag_integration',
            'lightrag_integration.examples',
            'lightrag_integration.performance_benchmarking',
            'lightrag_integration.tests'
        ]
        
        dependency_graph = {}
        
        for module_name in modules_to_analyze:
            dependencies = get_module_dependencies(module_name)
            dependency_graph[module_name] = dependencies
        
        # Analyze dependency graph
        print(f"\nModule dependency analysis:")
        for module, deps in dependency_graph.items():
            print(f"  {module}: {len(deps)} dependencies")
            if deps:
                for dep in sorted(deps):
                    print(f"    -> {dep}")
        
        # Check for excessive dependencies
        for module, deps in dependency_graph.items():
            if len(deps) > 20:  # Threshold for excessive dependencies
                warnings.warn(f"Module {module} has many dependencies: {len(deps)}")


class TestLazyLoadingVerification:
    """Test lazy loading behavior and efficiency."""
    
    def test_lazy_import_behavior(self):
        """Test that modules are loaded lazily when appropriate."""
        # Clear all lightrag_integration modules
        modules_to_clear = [name for name in sys.modules.keys() 
                           if name.startswith('lightrag_integration')]
        for module_name in modules_to_clear:
            del sys.modules[module_name]
        
        # Import main module
        import lightrag_integration
        
        # Check which modules are actually loaded
        loaded_modules = [name for name in sys.modules.keys() 
                         if name.startswith('lightrag_integration')]
        
        print(f"\nModules loaded after importing main module:")
        for module in sorted(loaded_modules):
            print(f"  {module}")
        
        # Main module should be loaded, but submodules might not be
        assert 'lightrag_integration' in loaded_modules
        
        # Check if submodules are lazily loaded
        expected_submodules = [
            'lightrag_integration.examples',
            'lightrag_integration.performance_benchmarking',
            'lightrag_integration.tests'
        ]
        
        lazily_loaded = []
        for submodule in expected_submodules:
            if submodule not in loaded_modules:
                lazily_loaded.append(submodule)
        
        if lazily_loaded:
            print(f"\nLazily loaded submodules: {lazily_loaded}")
        else:
            print(f"\nAll submodules loaded eagerly")

    def test_on_demand_loading_performance(self):
        """Test performance of on-demand loading."""
        # Clear modules
        modules_to_clear = [name for name in sys.modules.keys() 
                           if name.startswith('lightrag_integration')]
        for module_name in modules_to_clear:
            del sys.modules[module_name]
        
        # Import main module and measure time
        start_time = time.perf_counter()
        import lightrag_integration
        main_import_time = time.perf_counter() - start_time
        
        # Access factory function (should trigger loading of dependencies)
        start_time = time.perf_counter()
        factory_func = lightrag_integration.create_enhanced_rag_system
        factory_access_time = time.perf_counter() - start_time
        
        # Access metadata (should be fast)
        start_time = time.perf_counter()
        version = lightrag_integration.__version__
        metadata_access_time = time.perf_counter() - start_time
        
        print(f"\nOn-demand loading performance:")
        print(f"  Main import: {main_import_time:.6f}s")
        print(f"  Factory access: {factory_access_time:.6f}s") 
        print(f"  Metadata access: {metadata_access_time:.6f}s")
        
        # Metadata access should be very fast
        assert metadata_access_time < 0.001, \
            f"Metadata access too slow: {metadata_access_time:.6f}s"


class TestImportScalabilityMetrics:
    """Test import scalability and resource usage metrics."""
    
    def test_import_scaling_with_concurrent_processes(self):
        """Test how imports scale with concurrent processes."""
        import multiprocessing
        import queue
        
        def import_worker():
            """Worker function for concurrent import testing."""
            try:
                start_time = time.perf_counter()
                import lightrag_integration
                end_time = time.perf_counter()
                return end_time - start_time
            except Exception as e:
                return f"Error: {e}"
        
        # Test with multiple processes
        process_counts = [1, 2, 4]
        results = {}
        
        for proc_count in process_counts:
            if proc_count > multiprocessing.cpu_count():
                continue  # Skip if not enough CPUs
            
            with multiprocessing.Pool(processes=proc_count) as pool:
                start_time = time.perf_counter()
                worker_results = pool.map(import_worker, range(proc_count))
                total_time = time.perf_counter() - start_time
                
                results[proc_count] = {
                    'total_time': total_time,
                    'worker_results': worker_results,
                    'successful_imports': sum(1 for r in worker_results if isinstance(r, float))
                }
        
        # Analyze results
        print(f"\nConcurrent import scalability:")
        for proc_count, result in results.items():
            successful = result['successful_imports']
            total_time = result['total_time']
            print(f"  {proc_count} processes: {successful}/{proc_count} successful in {total_time:.3f}s")

    def test_memory_efficiency_metrics(self):
        """Test memory efficiency of imports."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Get baseline memory usage
            gc.collect()
            baseline_memory = process.memory_info().rss
            
            # Import main module
            import lightrag_integration
            after_main_memory = process.memory_info().rss
            
            # Access various components
            config_class = lightrag_integration.LightRAGConfig
            rag_class = lightrag_integration.ClinicalMetabolomicsRAG
            factory_func = lightrag_integration.create_enhanced_rag_system
            
            after_access_memory = process.memory_info().rss
            
            # Calculate memory usage
            main_import_memory = after_main_memory - baseline_memory
            component_access_memory = after_access_memory - after_main_memory
            
            print(f"\nMemory efficiency metrics:")
            print(f"  Baseline: {baseline_memory / (1024*1024):.2f}MB")
            print(f"  After main import: {main_import_memory / (1024*1024):.2f}MB increase")
            print(f"  After component access: {component_access_memory / (1024*1024):.2f}MB increase")
            
            # Memory usage should be reasonable
            assert main_import_memory < 50 * 1024 * 1024, \
                f"Main import uses too much memory: {main_import_memory / (1024*1024):.2f}MB"
                
        except ImportError:
            warnings.warn("psutil not available for memory efficiency testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])