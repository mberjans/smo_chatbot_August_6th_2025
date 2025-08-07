#!/usr/bin/env python3
"""
Comprehensive Test Utilities for Clinical Metabolomics Oracle LightRAG Integration.

This module provides standardized test utilities and helper functions to eliminate
repetitive testing patterns and streamline test development. It implements:

1. TestEnvironmentManager: Centralized system environment setup and validation
2. MockSystemFactory: Standardized mock object creation with configurable behaviors
3. Integration with existing fixtures and async testing framework
4. Comprehensive error handling and logging utilities
5. Biomedical test data generators and helpers

Key Features:
- Reduces 40+ repetitive patterns identified in test analysis
- Seamless integration with existing conftest.py fixtures
- Async-first design compatible with current test infrastructure
- Comprehensive mock factories for all system components
- Centralized import management and error handling
- Performance monitoring and resource management utilities

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import logging
import tempfile
import shutil
import json
import time
import random
import threading
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Type
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict, deque
import sys
import os
import importlib
import traceback
from enum import Enum
import warnings


# =====================================================================
# CORE UTILITY CLASSES AND ENUMS
# =====================================================================

class SystemComponent(Enum):
    """Enumeration of system components for testing."""
    LIGHTRAG_SYSTEM = "lightrag_system"
    PDF_PROCESSOR = "pdf_processor"
    COST_MONITOR = "cost_monitor"
    PROGRESS_TRACKER = "progress_tracker"
    CONFIG = "config"
    LOGGER = "logger"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"


class TestComplexity(Enum):
    """Test complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    PRODUCTION = "production"


class MockBehavior(Enum):
    """Mock behavior patterns."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    PARTIAL_SUCCESS = "partial_success"
    RANDOM = "random"


@dataclass
class EnvironmentSpec:
    """Specification for test environment setup."""
    working_dir: Optional[Path] = None
    temp_dirs: List[str] = field(default_factory=lambda: ["logs", "pdfs", "output"])
    required_imports: List[str] = field(default_factory=list)
    mock_components: List[SystemComponent] = field(default_factory=list)
    async_context: bool = True
    performance_monitoring: bool = False
    memory_limits: Optional[Dict[str, int]] = None
    cleanup_on_exit: bool = True


@dataclass
class MockSpec:
    """Specification for mock object creation."""
    component: SystemComponent
    behavior: MockBehavior = MockBehavior.SUCCESS
    response_delay: float = 0.1
    failure_rate: float = 0.0
    custom_responses: Optional[Dict[str, Any]] = None
    side_effects: Optional[List[Exception]] = None
    call_tracking: bool = True


# =====================================================================
# TEST ENVIRONMENT MANAGER
# =====================================================================

class TestEnvironmentManager:
    """
    Centralized test environment management for Clinical Metabolomics Oracle testing.
    
    Handles system path management, import availability checking, environment
    validation, cleanup, and integration with existing test infrastructure.
    """
    
    def __init__(self, spec: Optional[EnvironmentSpec] = None):
        """Initialize test environment manager."""
        self.spec = spec or EnvironmentSpec()
        self.created_dirs = []
        self.imported_modules = {}
        self.mock_patches = []
        self.cleanup_callbacks = []
        self.environment_stats = {
            'setup_time': None,
            'cleanup_time': None,
            'total_dirs_created': 0,
            'successful_imports': 0,
            'failed_imports': 0,
            'active_mocks': 0
        }
        self.logger = self._setup_logger()
        self._setup_complete = False
        
    def _setup_logger(self) -> logging.Logger:
        """Set up environment-specific logger."""
        logger = logging.getLogger(f"test_env_{id(self)}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)  # Quiet by default for tests
        return logger
    
    def setup_environment(self) -> Dict[str, Any]:
        """
        Set up complete test environment according to specification.
        
        Returns:
            Dictionary containing environment setup results and metadata.
        """
        start_time = time.time()
        self.environment_stats['setup_time'] = start_time
        
        try:
            # Set up working directory
            if not self.spec.working_dir:
                temp_dir = tempfile.mkdtemp(prefix="cmo_test_")
                self.spec.working_dir = Path(temp_dir)
                self.created_dirs.append(self.spec.working_dir)
            
            # Create required subdirectories
            for subdir in self.spec.temp_dirs:
                dir_path = self.spec.working_dir / subdir
                dir_path.mkdir(parents=True, exist_ok=True)
                self.environment_stats['total_dirs_created'] += 1
            
            # Set up system path
            self._setup_system_path()
            
            # Validate imports
            import_results = self._validate_imports()
            
            # Set up memory monitoring if requested
            memory_monitor = None
            if self.spec.performance_monitoring:
                memory_monitor = self._setup_memory_monitor()
            
            setup_results = {
                'working_dir': self.spec.working_dir,
                'subdirectories': {name: self.spec.working_dir / name for name in self.spec.temp_dirs},
                'import_results': import_results,
                'memory_monitor': memory_monitor,
                'environment_id': id(self),
                'setup_duration': time.time() - start_time
            }
            
            self._setup_complete = True
            self.logger.info(f"Test environment setup complete in {setup_results['setup_duration']:.3f}s")
            
            return setup_results
            
        except Exception as e:
            self.logger.error(f"Environment setup failed: {e}")
            self.cleanup()
            raise
    
    def _setup_system_path(self):
        """Set up system path for imports."""
        current_file = Path(__file__).resolve()
        
        # Add common paths that tests need
        paths_to_add = [
            current_file.parent.parent,  # lightrag_integration/
            current_file.parent.parent.parent,  # project root
            current_file.parent,  # tests/
        ]
        
        for path in paths_to_add:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                self.logger.debug(f"Added to sys.path: {path_str}")
    
    def _validate_imports(self) -> Dict[str, Any]:
        """Validate required imports are available."""
        import_results = {
            'successful': [],
            'failed': [],
            'available_modules': {},
            'mock_modules': {}
        }
        
        # Core modules that tests commonly need
        core_modules = [
            'lightrag_integration.clinical_metabolomics_rag',
            'lightrag_integration.pdf_processor',
            'lightrag_integration.config',
            'lightrag_integration.progress_tracker',
            'lightrag_integration.cost_persistence',
            'lightrag_integration.budget_manager'
        ]
        
        # Add user-specified modules
        all_modules = core_modules + self.spec.required_imports
        
        for module_name in all_modules:
            try:
                module = importlib.import_module(module_name)
                self.imported_modules[module_name] = module
                import_results['successful'].append(module_name)
                import_results['available_modules'][module_name] = module
                self.environment_stats['successful_imports'] += 1
                
            except ImportError as e:
                import_results['failed'].append({
                    'module': module_name,
                    'error': str(e)
                })
                self.environment_stats['failed_imports'] += 1
                self.logger.warning(f"Import failed for {module_name}: {e}")
                
                # Create mock module for graceful degradation
                mock_module = self._create_fallback_mock(module_name)
                import_results['mock_modules'][module_name] = mock_module
        
        return import_results
    
    def _create_fallback_mock(self, module_name: str) -> Mock:
        """Create fallback mock for failed imports."""
        mock_module = Mock()
        mock_module.__name__ = module_name
        
        # Add common classes/functions that might be expected
        common_classes = [
            'ClinicalMetabolomicsRAG',
            'BiomedicalPDFProcessor', 
            'LightRAGConfig',
            'ProgressTracker',
            'CostPersistence',
            'BudgetManager'
        ]
        
        for class_name in common_classes:
            setattr(mock_module, class_name, Mock)
        
        return mock_module
    
    def _setup_memory_monitor(self) -> Optional['MemoryMonitor']:
        """Set up memory monitoring if performance monitoring is enabled."""
        if not self.spec.performance_monitoring:
            return None
        
        try:
            return MemoryMonitor(
                memory_limits=self.spec.memory_limits or {},
                monitoring_interval=0.5
            )
        except Exception as e:
            self.logger.warning(f"Memory monitor setup failed: {e}")
            return None
    
    def get_import(self, module_name: str, fallback_to_mock: bool = True) -> Any:
        """
        Get imported module with optional fallback to mock.
        
        Args:
            module_name: Name of module to get
            fallback_to_mock: Whether to return mock if import failed
            
        Returns:
            The imported module or mock
        """
        if module_name in self.imported_modules:
            return self.imported_modules[module_name]
        
        if fallback_to_mock:
            return self._create_fallback_mock(module_name)
        
        raise ImportError(f"Module {module_name} not available and fallback disabled")
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check current system health and resource usage."""
        process = psutil.Process()
        
        return {
            'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'working_dir_size_mb': self._get_dir_size_mb(self.spec.working_dir),
            'open_files': len(process.open_files()),
            'environment_age_seconds': time.time() - (self.environment_stats['setup_time'] or 0),
            'gc_objects': len(gc.get_objects()),
            'active_threads': threading.active_count()
        }
    
    def _get_dir_size_mb(self, directory: Path) -> float:
        """Get directory size in MB."""
        if not directory or not directory.exists():
            return 0.0
        
        try:
            total_size = sum(
                f.stat().st_size for f in directory.rglob('*') if f.is_file()
            )
            return total_size / 1024 / 1024
        except Exception:
            return 0.0
    
    def add_cleanup_callback(self, callback: Callable[[], None]):
        """Add cleanup callback to be executed during teardown."""
        self.cleanup_callbacks.append(callback)
    
    def cleanup(self):
        """Clean up test environment."""
        if not self._setup_complete and not self.created_dirs:
            return
        
        cleanup_start = time.time()
        
        try:
            # Run cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.warning(f"Cleanup callback failed: {e}")
            
            # Clean up mock patches
            for patch_obj in self.mock_patches:
                try:
                    patch_obj.stop()
                except Exception:
                    pass
            
            # Clean up created directories
            if self.spec.cleanup_on_exit:
                for directory in self.created_dirs:
                    try:
                        if directory.exists():
                            shutil.rmtree(directory)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {directory}: {e}")
            
            # Force garbage collection
            gc.collect()
            
            self.environment_stats['cleanup_time'] = time.time() - cleanup_start
            self.logger.info(f"Environment cleanup completed in {self.environment_stats['cleanup_time']:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


# =====================================================================
# MOCK SYSTEM FACTORY
# =====================================================================

class MockSystemFactory:
    """
    Factory for creating standardized mock objects with configurable behavior patterns.
    
    Provides consistent mock creation for all system components with support for
    different behavior patterns, response templates, and integration with cost tracking.
    """
    
    # Biomedical content templates for realistic responses
    BIOMEDICAL_RESPONSE_TEMPLATES = {
        'metabolomics_query': """
        Based on metabolomics research, several key findings emerge:
        1. Glucose and lactate levels show significant alterations in metabolic disorders
        2. Amino acid profiles provide insights into protein metabolism
        3. Lipid biomarkers are crucial for cardiovascular risk assessment
        4. Metabolic pathway analysis reveals dysregulation in glycolysis and TCA cycle
        """,
        'proteomics_query': """
        Proteomic analysis reveals important protein biomarkers:
        1. Inflammatory proteins like CRP and TNF-alpha indicate disease activity
        2. Enzymatic proteins provide functional insights
        3. Structural proteins reflect tissue damage or repair
        4. Regulatory proteins indicate pathway activation states
        """,
        'clinical_diagnosis': """
        Clinical metabolomics supports diagnostic applications through:
        1. Disease-specific metabolite signatures
        2. Biomarker panels with high sensitivity and specificity
        3. Pathway-based disease classification
        4. Personalized treatment response monitoring
        """,
        'pathway_analysis': """
        Metabolic pathway analysis indicates:
        1. Glycolysis pathway shows enhanced activity
        2. TCA cycle demonstrates altered flux patterns
        3. Amino acid metabolism pathways are dysregulated
        4. Fatty acid oxidation pathways show impairment
        """
    }
    
    def __init__(self, environment_manager: Optional[TestEnvironmentManager] = None):
        """Initialize mock factory."""
        self.environment_manager = environment_manager
        self.created_mocks = {}
        self.mock_call_logs = defaultdict(list)
        self.logger = logging.getLogger(f"mock_factory_{id(self)}")
    
    def create_lightrag_system(self, spec: MockSpec) -> AsyncMock:
        """
        Create mock LightRAG system with configurable behavior.
        
        Args:
            spec: Mock specification defining behavior patterns
            
        Returns:
            Configured AsyncMock for LightRAG system
        """
        mock_lightrag = AsyncMock()
        
        # Configure based on behavior pattern
        if spec.behavior == MockBehavior.SUCCESS:
            mock_lightrag.ainsert = AsyncMock(
                side_effect=self._create_insert_handler(spec)
            )
            mock_lightrag.aquery = AsyncMock(
                side_effect=self._create_query_handler(spec)
            )
            mock_lightrag.adelete = AsyncMock(
                return_value={'status': 'success', 'deleted_count': 1}
            )
            
        elif spec.behavior == MockBehavior.FAILURE:
            error = Exception("Mock LightRAG system failure")
            mock_lightrag.ainsert = AsyncMock(side_effect=error)
            mock_lightrag.aquery = AsyncMock(side_effect=error)
            mock_lightrag.adelete = AsyncMock(side_effect=error)
            
        elif spec.behavior == MockBehavior.TIMEOUT:
            async def timeout_handler(*args, **kwargs):
                await asyncio.sleep(10)  # Simulate timeout
                raise asyncio.TimeoutError("Mock timeout")
            
            mock_lightrag.ainsert = AsyncMock(side_effect=timeout_handler)
            mock_lightrag.aquery = AsyncMock(side_effect=timeout_handler)
            
        elif spec.behavior == MockBehavior.PARTIAL_SUCCESS:
            mock_lightrag.ainsert = AsyncMock(
                side_effect=self._create_partial_success_handler(spec)
            )
            mock_lightrag.aquery = AsyncMock(
                side_effect=self._create_query_handler(spec)
            )
        
        # Configure properties
        mock_lightrag.working_dir = "/tmp/test_lightrag"
        mock_lightrag.cost_accumulated = 0.0
        
        # Add call tracking if requested
        if spec.call_tracking:
            self._add_call_tracking(mock_lightrag, "lightrag_system")
        
        self.created_mocks['lightrag_system'] = mock_lightrag
        return mock_lightrag
    
    def create_pdf_processor(self, spec: MockSpec) -> Mock:
        """
        Create mock PDF processor with configurable behavior.
        
        Args:
            spec: Mock specification defining behavior patterns
            
        Returns:
            Configured Mock for PDF processor
        """
        mock_processor = Mock()
        
        # Configure async methods based on behavior
        if spec.behavior == MockBehavior.SUCCESS:
            mock_processor.process_pdf = AsyncMock(
                side_effect=self._create_pdf_process_handler(spec)
            )
            mock_processor.process_batch_pdfs = AsyncMock(
                side_effect=self._create_batch_process_handler(spec)
            )
            
        elif spec.behavior == MockBehavior.FAILURE:
            error = Exception("Mock PDF processing failure")
            mock_processor.process_pdf = AsyncMock(side_effect=error)
            mock_processor.process_batch_pdfs = AsyncMock(side_effect=error)
            
        elif spec.behavior == MockBehavior.TIMEOUT:
            async def timeout_handler(*args, **kwargs):
                await asyncio.sleep(spec.response_delay * 10)
                raise asyncio.TimeoutError("PDF processing timeout")
            
            mock_processor.process_pdf = AsyncMock(side_effect=timeout_handler)
            mock_processor.process_batch_pdfs = AsyncMock(side_effect=timeout_handler)
        
        # Configure metadata extraction
        mock_processor.extract_metadata = AsyncMock(
            return_value={
                "title": "Test Clinical Research Paper",
                "authors": ["Dr. Test", "Dr. Mock"],
                "journal": "Journal of Test Medicine",
                "year": 2024,
                "keywords": ["metabolomics", "biomarkers", "clinical"]
            }
        )
        
        # Add call tracking if requested
        if spec.call_tracking:
            self._add_call_tracking(mock_processor, "pdf_processor")
        
        self.created_mocks['pdf_processor'] = mock_processor
        return mock_processor
    
    def create_cost_monitor(self, spec: MockSpec) -> Mock:
        """
        Create mock cost monitoring system.
        
        Args:
            spec: Mock specification defining behavior patterns
            
        Returns:
            Configured Mock for cost monitor
        """
        mock_monitor = Mock()
        
        # Initialize tracking variables
        mock_monitor.total_cost = 0.0
        mock_monitor.operation_costs = []
        mock_monitor.budget_alerts = []
        
        def track_cost(operation_type: str, cost: float, **kwargs):
            """Mock cost tracking function."""
            if spec.behavior != MockBehavior.FAILURE:
                mock_monitor.total_cost += cost
                cost_record = {
                    'operation_type': operation_type,
                    'cost': cost,
                    'timestamp': time.time(),
                    **kwargs
                }
                mock_monitor.operation_costs.append(cost_record)
                
                # Generate budget alert if cost exceeds threshold
                if mock_monitor.total_cost > 10.0:
                    alert = {
                        'level': 'warning',
                        'message': f'Budget threshold exceeded: ${mock_monitor.total_cost:.2f}',
                        'timestamp': time.time()
                    }
                    mock_monitor.budget_alerts.append(alert)
                
                return cost_record
            else:
                raise Exception("Mock cost tracking failure")
        
        mock_monitor.track_cost = track_cost
        mock_monitor.get_total_cost = lambda: mock_monitor.total_cost
        mock_monitor.get_budget_alerts = lambda: mock_monitor.budget_alerts
        mock_monitor.get_cost_history = lambda: mock_monitor.operation_costs
        
        # Add call tracking if requested
        if spec.call_tracking:
            self._add_call_tracking(mock_monitor, "cost_monitor")
        
        self.created_mocks['cost_monitor'] = mock_monitor
        return mock_monitor
    
    def create_progress_tracker(self, spec: MockSpec) -> Mock:
        """
        Create mock progress tracking system.
        
        Args:
            spec: Mock specification defining behavior patterns
            
        Returns:
            Configured Mock for progress tracker
        """
        mock_tracker = Mock()
        
        # Initialize tracking variables
        mock_tracker.progress = 0.0
        mock_tracker.status = "initialized"
        mock_tracker.events = []
        mock_tracker.start_time = time.time()
        
        def update_progress(progress: float, status: str = None, **kwargs):
            """Mock progress update function."""
            if spec.behavior != MockBehavior.FAILURE:
                mock_tracker.progress = progress
                if status:
                    mock_tracker.status = status
                
                event = {
                    'timestamp': time.time(),
                    'progress': progress,
                    'status': status or mock_tracker.status,
                    **kwargs
                }
                mock_tracker.events.append(event)
                return event
            else:
                raise Exception("Mock progress tracking failure")
        
        def get_summary():
            """Get progress summary."""
            return {
                'current_progress': mock_tracker.progress,
                'current_status': mock_tracker.status,
                'elapsed_time': time.time() - mock_tracker.start_time,
                'total_events': len(mock_tracker.events)
            }
        
        mock_tracker.update_progress = update_progress
        mock_tracker.get_summary = get_summary
        mock_tracker.reset = lambda: setattr(mock_tracker, 'events', [])
        
        # Add call tracking if requested
        if spec.call_tracking:
            self._add_call_tracking(mock_tracker, "progress_tracker")
        
        self.created_mocks['progress_tracker'] = mock_tracker
        return mock_tracker
    
    def create_comprehensive_mock_set(self, 
                                    components: List[SystemComponent],
                                    behavior: MockBehavior = MockBehavior.SUCCESS) -> Dict[str, Any]:
        """
        Create comprehensive set of mock objects for multiple components.
        
        Args:
            components: List of components to mock
            behavior: Default behavior pattern for all mocks
            
        Returns:
            Dictionary mapping component names to mock objects
        """
        mock_set = {}
        
        for component in components:
            spec = MockSpec(
                component=component,
                behavior=behavior,
                call_tracking=True
            )
            
            if component == SystemComponent.LIGHTRAG_SYSTEM:
                mock_set['lightrag_system'] = self.create_lightrag_system(spec)
            elif component == SystemComponent.PDF_PROCESSOR:
                mock_set['pdf_processor'] = self.create_pdf_processor(spec)
            elif component == SystemComponent.COST_MONITOR:
                mock_set['cost_monitor'] = self.create_cost_monitor(spec)
            elif component == SystemComponent.PROGRESS_TRACKER:
                mock_set['progress_tracker'] = self.create_progress_tracker(spec)
            elif component == SystemComponent.CONFIG:
                mock_set['config'] = self._create_mock_config()
            elif component == SystemComponent.LOGGER:
                mock_set['logger'] = self._create_mock_logger()
        
        return mock_set
    
    def _create_insert_handler(self, spec: MockSpec):
        """Create handler for document insertion."""
        async def insert_handler(documents):
            await asyncio.sleep(spec.response_delay)
            
            if isinstance(documents, str):
                documents = [documents]
            
            cost = len(documents) * 0.01  # Mock cost calculation
            
            return {
                'status': 'success',
                'documents_processed': len(documents),
                'total_cost': cost,
                'entities_extracted': random.randint(10, 50),
                'relationships_found': random.randint(5, 25)
            }
        
        return insert_handler
    
    def _create_query_handler(self, spec: MockSpec):
        """Create handler for query processing."""
        async def query_handler(query, mode="hybrid"):
            await asyncio.sleep(spec.response_delay)
            
            # Select response template based on query content
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['metabolite', 'metabolomics']):
                response = self.BIOMEDICAL_RESPONSE_TEMPLATES['metabolomics_query']
            elif any(word in query_lower for word in ['protein', 'proteomics']):
                response = self.BIOMEDICAL_RESPONSE_TEMPLATES['proteomics_query']
            elif any(word in query_lower for word in ['diagnos', 'clinical']):
                response = self.BIOMEDICAL_RESPONSE_TEMPLATES['clinical_diagnosis']
            elif any(word in query_lower for word in ['pathway']):
                response = self.BIOMEDICAL_RESPONSE_TEMPLATES['pathway_analysis']
            else:
                response = "Mock response for general biomedical query."
            
            # Add custom response if specified
            if spec.custom_responses and 'query' in spec.custom_responses:
                response = spec.custom_responses['query']
            
            return response.strip()
        
        return query_handler
    
    def _create_partial_success_handler(self, spec: MockSpec):
        """Create handler for partial success scenarios."""
        async def partial_handler(documents):
            await asyncio.sleep(spec.response_delay)
            
            if isinstance(documents, str):
                documents = [documents]
            
            # Simulate partial failures
            successful = int(len(documents) * (1 - spec.failure_rate))
            failed = len(documents) - successful
            
            return {
                'status': 'partial_success',
                'documents_processed': successful,
                'documents_failed': failed,
                'total_cost': successful * 0.01,
                'errors': ['Mock processing error'] * failed if failed > 0 else []
            }
        
        return partial_handler
    
    def _create_pdf_process_handler(self, spec: MockSpec):
        """Create handler for PDF processing."""
        async def process_handler(pdf_path):
            await asyncio.sleep(spec.response_delay)
            
            # Generate realistic content based on filename
            filename = str(pdf_path).lower() if hasattr(pdf_path, 'lower') else str(pdf_path).lower()
            
            if 'diabetes' in filename or 'metabolomic' in filename:
                content = "This study investigates metabolomic profiles in diabetes patients using LC-MS analysis..."
                title = "Metabolomic Analysis of Diabetes Biomarkers"
            elif 'cardiovascular' in filename:
                content = "Cardiovascular research demonstrates altered lipid profiles in heart disease patients..."
                title = "Cardiovascular Biomarker Study"
            else:
                content = "Clinical research investigating biomedical mechanisms and therapeutic targets..."
                title = "Clinical Research Study"
            
            return {
                "text": content,
                "metadata": {
                    "title": title,
                    "page_count": random.randint(8, 20),
                    "file_size": random.randint(1024*100, 1024*1024)
                },
                "processing_time": spec.response_delay,
                "success": True
            }
        
        return process_handler
    
    def _create_batch_process_handler(self, spec: MockSpec):
        """Create handler for batch PDF processing."""
        async def batch_handler(pdf_paths):
            total_time = len(pdf_paths) * spec.response_delay
            await asyncio.sleep(min(total_time, 2.0))  # Cap simulation time
            
            results = []
            successful = 0
            failed = 0
            
            for pdf_path in pdf_paths:
                try:
                    if random.random() < spec.failure_rate:
                        failed += 1
                    else:
                        result = await self._create_pdf_process_handler(spec)(pdf_path)
                        results.append(result)
                        successful += 1
                except Exception:
                    failed += 1
            
            return {
                "results": results,
                "processed": successful,
                "failed": failed,
                "total_time": total_time
            }
        
        return batch_handler
    
    def _create_mock_config(self) -> Mock:
        """Create mock configuration object."""
        config = Mock()
        config.api_key = "test-api-key"
        config.model = "gpt-4o-mini"
        config.embedding_model = "text-embedding-3-small"
        config.max_tokens = 8192
        config.working_dir = Path("/tmp/test_working")
        config.enable_cost_tracking = True
        config.daily_budget_limit = 10.0
        config.log_level = "INFO"
        return config
    
    def _create_mock_logger(self) -> Mock:
        """Create mock logger object."""
        logger = Mock(spec=logging.Logger)
        logger.debug = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        logger.critical = Mock()
        return logger
    
    def _add_call_tracking(self, mock_obj: Mock, component_name: str):
        """Add call tracking to mock object."""
        original_call = mock_obj.__call__
        
        def tracked_call(*args, **kwargs):
            call_info = {
                'timestamp': time.time(),
                'args': args,
                'kwargs': kwargs,
                'component': component_name
            }
            self.mock_call_logs[component_name].append(call_info)
            return original_call(*args, **kwargs)
        
        mock_obj.__call__ = tracked_call
    
    def get_call_logs(self, component_name: Optional[str] = None) -> Dict[str, List]:
        """Get call logs for components."""
        if component_name:
            return {component_name: self.mock_call_logs.get(component_name, [])}
        return dict(self.mock_call_logs)
    
    def reset_call_logs(self):
        """Reset all call logs."""
        self.mock_call_logs.clear()
    
    def get_mock_statistics(self) -> Dict[str, Any]:
        """Get statistics about created mocks."""
        return {
            'total_mocks_created': len(self.created_mocks),
            'mock_types': list(self.created_mocks.keys()),
            'total_calls_tracked': sum(len(logs) for logs in self.mock_call_logs.values()),
            'calls_by_component': {k: len(v) for k, v in self.mock_call_logs.items()}
        }


# =====================================================================
# MEMORY MONITORING UTILITY
# =====================================================================

class MemoryMonitor:
    """Memory monitoring utility for performance testing."""
    
    def __init__(self, memory_limits: Dict[str, int], monitoring_interval: float = 1.0):
        """Initialize memory monitor."""
        self.memory_limits = memory_limits  # in MB
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.memory_samples = []
        self.alerts = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[Dict[str, Any]]:
        """Stop monitoring and return samples."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        return self.memory_samples.copy()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_info = process.memory_info()
                sample = {
                    'timestamp': time.time(),
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'cpu_percent': process.cpu_percent()
                }
                
                self.memory_samples.append(sample)
                
                # Check limits
                for limit_name, limit_mb in self.memory_limits.items():
                    if sample['rss_mb'] > limit_mb:
                        alert = {
                            'timestamp': time.time(),
                            'limit_type': limit_name,
                            'limit_mb': limit_mb,
                            'actual_mb': sample['rss_mb']
                        }
                        self.alerts.append(alert)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                # Continue monitoring even if sampling fails
                time.sleep(self.monitoring_interval)


# =====================================================================
# PYTEST INTEGRATION FIXTURES
# =====================================================================

@pytest.fixture
def test_environment_manager():
    """Provide TestEnvironmentManager for tests."""
    manager = TestEnvironmentManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def mock_system_factory(test_environment_manager):
    """Provide MockSystemFactory for tests."""
    return MockSystemFactory(test_environment_manager)


@pytest.fixture
def standard_test_environment(test_environment_manager):
    """Set up standard test environment with common requirements."""
    spec = EnvironmentSpec(
        temp_dirs=["logs", "pdfs", "output", "working"],
        required_imports=[
            "lightrag_integration.clinical_metabolomics_rag",
            "lightrag_integration.pdf_processor",
            "lightrag_integration.config"
        ],
        mock_components=[
            SystemComponent.LIGHTRAG_SYSTEM,
            SystemComponent.PDF_PROCESSOR,
            SystemComponent.COST_MONITOR
        ],
        async_context=True,
        performance_monitoring=False
    )
    
    test_environment_manager.spec = spec
    environment_data = test_environment_manager.setup_environment()
    yield environment_data
    test_environment_manager.cleanup()


@pytest.fixture
def comprehensive_mock_system(mock_system_factory):
    """Provide comprehensive mock system with all components."""
    components = [
        SystemComponent.LIGHTRAG_SYSTEM,
        SystemComponent.PDF_PROCESSOR,
        SystemComponent.COST_MONITOR,
        SystemComponent.PROGRESS_TRACKER,
        SystemComponent.CONFIG,
        SystemComponent.LOGGER
    ]
    
    return mock_system_factory.create_comprehensive_mock_set(components)


@pytest.fixture
def biomedical_test_data_generator():
    """Provide biomedical test data generator utility."""
    class BiomedicalDataGenerator:
        @staticmethod
        def generate_clinical_query(disease: str = "diabetes") -> str:
            queries = {
                'diabetes': "What metabolites are associated with diabetes progression?",
                'cardiovascular': "What are the key biomarkers for cardiovascular disease?",
                'cancer': "How does metabolomics help in cancer diagnosis?",
                'kidney': "What metabolic changes occur in kidney disease?"
            }
            return queries.get(disease, queries['diabetes'])
        
        @staticmethod
        def generate_test_pdf_content(topic: str = "metabolomics") -> str:
            templates = {
                'metabolomics': """
                Abstract: This study investigates metabolomic profiles in clinical populations.
                Methods: LC-MS/MS analysis was performed on plasma samples.
                Results: Significant alterations were found in glucose and amino acid metabolism.
                Conclusions: Metabolomics provides valuable biomarkers for clinical applications.
                """,
                'proteomics': """
                Abstract: Proteomic analysis of disease-specific protein alterations.
                Methods: Mass spectrometry-based protein identification and quantification.
                Results: Key proteins showed differential expression in disease vs. control.
                Conclusions: Protein biomarkers offer diagnostic and therapeutic insights.
                """
            }
            return templates.get(topic, templates['metabolomics']).strip()
    
    return BiomedicalDataGenerator()


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def create_quick_test_environment(async_support: bool = True) -> Tuple[TestEnvironmentManager, MockSystemFactory]:
    """
    Quick setup for test environment and mock factory.
    
    Args:
        async_support: Whether to enable async testing support
        
    Returns:
        Tuple of (TestEnvironmentManager, MockSystemFactory)
    """
    spec = EnvironmentSpec(
        temp_dirs=["logs", "output"],
        async_context=async_support,
        performance_monitoring=False,
        cleanup_on_exit=True
    )
    
    env_manager = TestEnvironmentManager(spec)
    env_manager.setup_environment()
    
    mock_factory = MockSystemFactory(env_manager)
    
    return env_manager, mock_factory


def create_performance_test_setup(memory_limit_mb: int = 512) -> Tuple[TestEnvironmentManager, MockSystemFactory]:
    """
    Create test setup optimized for performance testing.
    
    Args:
        memory_limit_mb: Memory limit in megabytes
        
    Returns:
        Tuple of (TestEnvironmentManager, MockSystemFactory)
    """
    spec = EnvironmentSpec(
        temp_dirs=["logs", "output", "perf_data"],
        performance_monitoring=True,
        memory_limits={'test_limit': memory_limit_mb},
        async_context=True
    )
    
    env_manager = TestEnvironmentManager(spec)
    env_manager.setup_environment()
    
    mock_factory = MockSystemFactory(env_manager)
    
    return env_manager, mock_factory


# =====================================================================
# ASYNC CONTEXT MANAGERS
# =====================================================================

@asynccontextmanager
async def async_test_context(timeout: float = 30.0):
    """
    Async context manager for test execution with timeout and cleanup.
    
    Args:
        timeout: Maximum execution time in seconds
    """
    tasks = []
    start_time = time.time()
    
    try:
        yield {
            'tasks': tasks,
            'start_time': start_time,
            'timeout': timeout
        }
    finally:
        # Cancel any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        
        # Force garbage collection
        gc.collect()


@asynccontextmanager
async def monitored_async_operation(operation_name: str, performance_tracking: bool = True):
    """
    Context manager for monitoring async operations.
    
    Args:
        operation_name: Name of the operation being monitored
        performance_tracking: Whether to track performance metrics
    """
    start_time = time.time()
    start_memory = None
    
    if performance_tracking:
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
    
    try:
        yield {
            'operation_name': operation_name,
            'start_time': start_time,
            'start_memory_mb': start_memory
        }
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        end_memory = None
        if performance_tracking:
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024 / 1024
        
        # Could log or store metrics here
        if performance_tracking and start_memory and end_memory:
            memory_delta = end_memory - start_memory
            if memory_delta > 50:  # Alert if memory usage increased by more than 50MB
                warnings.warn(f"Operation {operation_name} used {memory_delta:.1f}MB memory")


# Make key classes available at module level
__all__ = [
    'TestEnvironmentManager',
    'MockSystemFactory', 
    'SystemComponent',
    'TestComplexity',
    'MockBehavior',
    'EnvironmentSpec',
    'MockSpec',
    'MemoryMonitor',
    'create_quick_test_environment',
    'create_performance_test_setup',
    'async_test_context',
    'monitored_async_operation'
]