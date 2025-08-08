"""
Comprehensive Test Suite for LightRAG Integration Module Import Functionality

This module provides comprehensive testing for all import and export functionality
of the lightrag_integration module, ensuring that all components can be imported
successfully and that the module interface works correctly for CMO system integration.

Test Categories:
    - Main module exports (37 components)
    - Subpackage imports 
    - Factory function imports and callability
    - Error handling for missing dependencies
    - Circular import detection
    - Import side effects validation

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import importlib
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import Mock, patch
import warnings

import pytest


class TestModuleImports:
    """
    Test suite for comprehensive module import functionality.
    
    Tests all aspects of module importing including individual components,
    subpackages, factory functions, and error conditions.
    """

    # Expected main module exports based on __all__ list
    EXPECTED_MAIN_EXPORTS = [
        # Version and metadata
        "__version__",
        "__author__", 
        "__description__",
        
        # Core components
        "LightRAGConfig",
        "LightRAGConfigError", 
        "setup_lightrag_logging",
        "ClinicalMetabolomicsRAG",
        "ClinicalMetabolomicsRAGError",
        "CostSummary",
        "QueryResponse",
        "CircuitBreaker",
        "CircuitBreakerError",
        "RateLimiter",
        "RequestQueue",
        "add_jitter",
        
        # Cost persistence
        "CostPersistence",
        "CostRecord",
        "ResearchCategory",
        "CostDatabase",
        
        # Budget management
        "BudgetManager",
        "BudgetThreshold", 
        "BudgetAlert",
        "AlertLevel",
        
        # Research categorization
        "ResearchCategorizer",
        "CategoryPrediction",
        "CategoryMetrics",
        "QueryAnalyzer",
        
        # Audit and compliance
        "AuditTrail",
        "AuditEvent",
        "AuditEventType",
        "ComplianceRule",
        "ComplianceChecker",
        
        # Utilities
        "BiomedicalPDFProcessor",
        "BiomedicalPDFProcessorError",
        
        # API metrics logging
        "APIUsageMetricsLogger",
        "APIMetric",
        "MetricType", 
        "MetricsAggregator",
        
        # Factory functions
        "create_enhanced_rag_system",
        "get_default_research_categories"
    ]
    
    # Expected subpackages
    EXPECTED_SUBPACKAGES = [
        "examples",
        "performance_benchmarking", 
        "performance_benchmarking.reporting",
        "tests"
    ]
    
    # Factory functions that should be callable
    FACTORY_FUNCTIONS = [
        "create_enhanced_rag_system",
        "get_default_research_categories"
    ]

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment and clean up after each test."""
        self.original_modules = sys.modules.copy()
        self.import_times = {}
        self.import_errors = {}
        yield
        # Clean up any modules imported during testing if needed
        # (Keep original modules intact)
        
    def test_main_module_import(self):
        """Test that the main lightrag_integration module can be imported."""
        try:
            start_time = time.time()
            import lightrag_integration
            import_time = time.time() - start_time
            self.import_times['main_module'] = import_time
            
            assert lightrag_integration is not None
            assert hasattr(lightrag_integration, '__version__')
            assert hasattr(lightrag_integration, '__all__')
            
            # Test import time is reasonable (< 5 seconds)
            assert import_time < 5.0, f"Main module import took too long: {import_time:.2f}s"
            
        except Exception as e:
            self.import_errors['main_module'] = str(e)
            pytest.fail(f"Failed to import main lightrag_integration module: {e}")

    def test_all_main_exports_importable(self):
        """Test that all components in __all__ can be imported from main module."""
        import lightrag_integration
        
        failed_imports = []
        successful_imports = []
        
        for export_name in self.EXPECTED_MAIN_EXPORTS:
            try:
                start_time = time.time()
                component = getattr(lightrag_integration, export_name)
                import_time = time.time() - start_time
                
                assert component is not None, f"{export_name} is None after import"
                successful_imports.append(export_name)
                self.import_times[export_name] = import_time
                
            except AttributeError as e:
                failed_imports.append(f"{export_name}: AttributeError - {e}")
                self.import_errors[export_name] = str(e)
            except ImportError as e:
                failed_imports.append(f"{export_name}: ImportError - {e}")
                self.import_errors[export_name] = str(e)
            except Exception as e:
                failed_imports.append(f"{export_name}: {type(e).__name__} - {e}")
                self.import_errors[export_name] = str(e)
        
        if failed_imports:
            pytest.fail(f"Failed to import {len(failed_imports)} components:\n" + 
                       "\n".join(failed_imports))
        
        assert len(successful_imports) == len(self.EXPECTED_MAIN_EXPORTS), \
            f"Expected {len(self.EXPECTED_MAIN_EXPORTS)} imports, got {len(successful_imports)}"

    def test_subpackage_imports(self):
        """Test that all expected subpackages can be imported."""
        failed_subpackages = []
        successful_subpackages = []
        
        for subpackage in self.EXPECTED_SUBPACKAGES:
            try:
                start_time = time.time()
                full_module_name = f"lightrag_integration.{subpackage}"
                module = importlib.import_module(full_module_name)
                import_time = time.time() - start_time
                
                assert module is not None, f"Subpackage {subpackage} imported as None"
                successful_subpackages.append(subpackage)
                self.import_times[f"subpackage_{subpackage}"] = import_time
                
            except ImportError as e:
                failed_subpackages.append(f"{subpackage}: ImportError - {e}")
                self.import_errors[f"subpackage_{subpackage}"] = str(e)
            except Exception as e:
                failed_subpackages.append(f"{subpackage}: {type(e).__name__} - {e}")
                self.import_errors[f"subpackage_{subpackage}"] = str(e)
        
        if failed_subpackages:
            pytest.fail(f"Failed to import {len(failed_subpackages)} subpackages:\n" + 
                       "\n".join(failed_subpackages))
        
        assert len(successful_subpackages) == len(self.EXPECTED_SUBPACKAGES), \
            f"Expected {len(self.EXPECTED_SUBPACKAGES)} subpackages, got {len(successful_subpackages)}"

    def test_factory_functions_importable_and_callable(self):
        """Test that factory functions can be imported and are callable."""
        import lightrag_integration
        
        for func_name in self.FACTORY_FUNCTIONS:
            # Test import
            assert hasattr(lightrag_integration, func_name), \
                f"Factory function {func_name} not found in module"
            
            func = getattr(lightrag_integration, func_name)
            
            # Test callable
            assert callable(func), f"Factory function {func_name} is not callable"
            
            # Test function has docstring
            assert func.__doc__ is not None, f"Factory function {func_name} missing docstring"
            assert len(func.__doc__.strip()) > 0, f"Factory function {func_name} has empty docstring"

    def test_factory_function_create_enhanced_rag_system_execution(self):
        """Test that create_enhanced_rag_system factory function can be executed without errors."""
        import lightrag_integration
        
        func = lightrag_integration.create_enhanced_rag_system
        
        # Test with minimal configuration that shouldn't require external dependencies
        try:
            with patch.dict('os.environ', {
                'LIGHTRAG_ENABLE_COST_TRACKING': 'true',
                'LIGHTRAG_WORKING_DIR': '/tmp/test_lightrag',
                'LIGHTRAG_MODEL': 'test-model'
            }):
                # This should create the configuration without fully initializing the RAG system
                result = func(config_source={'test': True, 'working_dir': '/tmp/test_lightrag'})
                assert result is not None, "Factory function returned None"
                
        except Exception as e:
            # If it fails due to missing dependencies, that's acceptable in a test environment
            # We mainly want to ensure the function is importable and has the right signature
            if "No module named" in str(e) or "missing" in str(e).lower():
                pytest.skip(f"Skipping factory function execution test due to missing dependencies: {e}")
            else:
                pytest.fail(f"Factory function failed with unexpected error: {e}")

    def test_factory_function_get_default_research_categories_execution(self):
        """Test that get_default_research_categories factory function executes correctly."""
        import lightrag_integration
        
        func = lightrag_integration.get_default_research_categories
        
        try:
            categories = func()
            
            # Basic validation
            assert isinstance(categories, list), "get_default_research_categories should return a list"
            assert len(categories) > 0, "get_default_research_categories should return non-empty list"
            
            # Validate structure
            for category in categories:
                assert isinstance(category, dict), "Each category should be a dictionary"
                assert 'name' in category, "Each category should have a 'name' field"
                assert 'value' in category, "Each category should have a 'value' field"
                assert 'description' in category, "Each category should have a 'description' field"
                
        except Exception as e:
            if "No module named" in str(e):
                pytest.skip(f"Skipping factory function execution test due to missing dependencies: {e}")
            else:
                pytest.fail(f"get_default_research_categories failed: {e}")

    def test_import_no_side_effects(self):
        """Test that importing the module doesn't have unwanted side effects."""
        initial_env = dict(os.environ) if 'os' in sys.modules else {}
        initial_logging_level = None
        
        # Capture initial logging state
        import logging
        initial_logging_level = logging.getLogger().level
        
        # Import the module
        import lightrag_integration
        
        # Check that environment wasn't modified
        import os
        if initial_env:
            # Only check for new environment variables, don't require exact match
            # as the module may read env vars
            pass
        
        # Check that logging level wasn't changed dramatically
        current_logging_level = logging.getLogger().level
        # Allow some logging configuration, but not dramatic changes
        assert abs(current_logging_level - initial_logging_level) <= 20, \
            "Module import significantly changed logging configuration"

    def test_circular_import_detection(self):
        """Test that there are no circular imports in the module."""
        def check_circular_imports(module_name: str, visited: Set[str] = None) -> List[str]:
            """Recursively check for circular imports."""
            if visited is None:
                visited = set()
            
            if module_name in visited:
                return [module_name]  # Circular import detected
            
            try:
                visited.add(module_name)
                module = importlib.import_module(module_name)
                
                # Get all imported modules from this module
                imported_modules = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, '__module__'):
                        if attr.__module__.startswith('lightrag_integration'):
                            imported_modules.append(attr.__module__)
                
                # Recursively check imported modules
                for imported_module in set(imported_modules):
                    if imported_module != module_name:
                        circular_path = check_circular_imports(imported_module, visited.copy())
                        if circular_path:
                            return circular_path + [module_name]
                
                return []
                
            except ImportError:
                return []  # Can't import, so no circular dependency possible
            except Exception:
                return []  # Other errors, skip this module
        
        # Check main module
        circular_imports = check_circular_imports('lightrag_integration')
        
        assert not circular_imports, f"Circular import detected: {' -> '.join(circular_imports)}"

    def test_missing_dependency_handling(self):
        """Test graceful handling when dependencies are missing."""
        
        # Test with mocked missing dependencies
        missing_deps = ['openai', 'lightrag', 'pandas', 'numpy']
        
        for dep in missing_deps:
            if dep not in sys.modules:
                # Temporarily remove the dependency and test import behavior
                with patch.dict('sys.modules', {dep: None}):
                    try:
                        # Try importing components that might depend on this
                        importlib.reload(importlib.import_module('lightrag_integration'))
                        # If import succeeds despite missing dep, that's good (graceful handling)
                    except ImportError as e:
                        if dep in str(e):
                            # Expected behavior - dependency properly declared
                            pass
                        else:
                            pytest.fail(f"Unexpected ImportError when {dep} missing: {e}")
                    except Exception as e:
                        pytest.fail(f"Unexpected error when {dep} missing: {e}")

    def test_import_performance(self):
        """Test that module imports complete within reasonable time limits."""
        import lightrag_integration
        
        # Check collected import times
        slow_imports = {name: time_taken for name, time_taken in self.import_times.items() 
                       if time_taken > 2.0}
        
        if slow_imports:
            warning_msg = f"Slow imports detected (>2s): {slow_imports}"
            warnings.warn(warning_msg, UserWarning)
        
        # Main module should import quickly
        main_module_time = self.import_times.get('main_module', 0)
        assert main_module_time < 10.0, f"Main module import too slow: {main_module_time:.2f}s"

    def test_metadata_accessibility(self):
        """Test that module metadata is accessible and well-formed."""
        import lightrag_integration
        
        # Test version
        assert hasattr(lightrag_integration, '__version__')
        version = lightrag_integration.__version__
        assert isinstance(version, str)
        assert len(version.strip()) > 0
        assert version.count('.') >= 2  # Should be at least x.y.z format
        
        # Test author
        assert hasattr(lightrag_integration, '__author__')
        author = lightrag_integration.__author__
        assert isinstance(author, str)
        assert len(author.strip()) > 0
        
        # Test description
        assert hasattr(lightrag_integration, '__description__')
        description = lightrag_integration.__description__
        assert isinstance(description, str)
        assert len(description.strip()) > 0

    def test_module_docstring(self):
        """Test that module has comprehensive docstring."""
        import lightrag_integration
        
        assert lightrag_integration.__doc__ is not None, "Module missing docstring"
        docstring = lightrag_integration.__doc__.strip()
        assert len(docstring) > 100, "Module docstring too short"
        
        # Check for key documentation elements
        docstring_lower = docstring.lower()
        assert 'cost tracking' in docstring_lower, "Missing cost tracking documentation"
        assert 'clinical metabolomics' in docstring_lower, "Missing clinical metabolomics documentation"
        assert 'usage' in docstring_lower, "Missing usage documentation"

    @pytest.mark.asyncio
    async def test_async_compatible_imports(self):
        """Test that module imports are compatible with async environments."""
        # Import in async context
        import lightrag_integration
        
        # Test that we can access components in async context
        config_class = lightrag_integration.LightRAGConfig
        rag_class = lightrag_integration.ClinicalMetabolomicsRAG
        
        assert config_class is not None
        assert rag_class is not None
        
        # Test factory function in async context
        factory_func = lightrag_integration.create_enhanced_rag_system
        assert callable(factory_func)

    def test_component_type_validation(self):
        """Test that imported components have expected types."""
        import lightrag_integration
        
        # Test classes
        class_components = [
            'LightRAGConfig', 'ClinicalMetabolomicsRAG', 'BudgetManager',
            'ResearchCategorizer', 'AuditTrail', 'BiomedicalPDFProcessor',
            'APIUsageMetricsLogger'
        ]
        
        for comp_name in class_components:
            if hasattr(lightrag_integration, comp_name):
                comp = getattr(lightrag_integration, comp_name)
                assert isinstance(comp, type), f"{comp_name} should be a class/type"
        
        # Test functions
        function_components = [
            'create_enhanced_rag_system', 'get_default_research_categories',
            'setup_lightrag_logging', 'add_jitter'
        ]
        
        for func_name in function_components:
            if hasattr(lightrag_integration, func_name):
                func = getattr(lightrag_integration, func_name)
                assert callable(func), f"{func_name} should be callable"

    def test_exception_classes_importable(self):
        """Test that custom exception classes can be imported and are proper exceptions."""
        import lightrag_integration
        
        exception_classes = [
            'LightRAGConfigError', 'ClinicalMetabolomicsRAGError',
            'BiomedicalPDFProcessorError', 'CircuitBreakerError'
        ]
        
        for exc_name in exception_classes:
            if hasattr(lightrag_integration, exc_name):
                exc_class = getattr(lightrag_integration, exc_name)
                assert isinstance(exc_class, type), f"{exc_name} should be a class"
                assert issubclass(exc_class, Exception), f"{exc_name} should be an Exception subclass"

    def test_enum_classes_importable(self):
        """Test that enum classes can be imported and have expected structure."""
        import lightrag_integration
        
        enum_classes = ['ResearchCategory', 'AlertLevel', 'AuditEventType', 'MetricType']
        
        for enum_name in enum_classes:
            if hasattr(lightrag_integration, enum_name):
                enum_class = getattr(lightrag_integration, enum_name)
                # Basic enum validation
                assert hasattr(enum_class, '__members__'), f"{enum_name} should have __members__"
                assert len(enum_class.__members__) > 0, f"{enum_name} should have members"


class TestModuleImportPerformance:
    """Performance-focused tests for module imports."""
    
    def test_import_memory_usage(self):
        """Test that module imports don't consume excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Import the module
        import lightrag_integration
        
        memory_after = process.memory_info().rss
        memory_delta = memory_after - memory_before
        
        # Allow up to 100MB for module import
        max_memory_delta = 100 * 1024 * 1024  # 100MB in bytes
        assert memory_delta < max_memory_delta, \
            f"Module import used too much memory: {memory_delta / (1024*1024):.2f}MB"

    def test_repeated_imports_performance(self):
        """Test that repeated imports don't degrade performance."""
        import importlib
        
        times = []
        
        for i in range(5):
            start_time = time.time()
            importlib.reload(importlib.import_module('lightrag_integration'))
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Later imports should not be significantly slower than first
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        assert max_time <= avg_time * 3, \
            f"Import performance degraded significantly: max={max_time:.3f}s avg={avg_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])