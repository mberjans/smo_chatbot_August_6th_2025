#!/usr/bin/env python3
"""
Integration Test for Test Data Fixtures System.

This module tests the integration between the new test data fixtures system
and the existing pytest infrastructure to ensure compatibility and proper
functionality.

Test Coverage:
1. Basic fixture loading and cleanup
2. Integration with existing fixtures
3. Async fixture functionality
4. Database fixtures and schema loading
5. Mock data fixtures
6. Performance and memory management
7. Error handling and recovery
8. Cross-fixture dependencies

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import sqlite3
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import threading

# Import our test data fixtures
from .test_data_fixtures import (
    TestDataManager, TestDataConfig, test_data_manager,
    sample_metabolomics_study, mock_metabolites_data, test_cost_db,
    test_temp_dir, async_test_data_manager, pdf_samples_dir,
    mock_biomedical_dir, mock_openai_responses, test_knowledge_db
)

from .test_data_utilities import (
    TestDataFactory, DataValidationSuite, MockDataGenerator,
    BiochemicalCompound, ClinicalStudyData
)

from .test_data_integration import (
    FixtureIntegrator, AsyncTestDataManager, PerformanceOptimizer, IntegrationConfig,
    integrated_test_data_manager, enhanced_pdf_data, comprehensive_mock_data,
    async_biomedical_data
)


# =====================================================================
# BASIC FIXTURE TESTS
# =====================================================================

def test_test_data_manager_creation(test_data_manager: TestDataManager):
    """Test basic test data manager creation and configuration."""
    assert test_data_manager is not None
    assert isinstance(test_data_manager, TestDataManager)
    assert test_data_manager.config is not None
    assert isinstance(test_data_manager.loaded_data, dict)
    assert isinstance(test_data_manager.temp_dirs, list)
    assert isinstance(test_data_manager.db_connections, list)


def test_sample_data_loading(sample_metabolomics_study: str):
    """Test loading of sample metabolomics study data."""
    assert sample_metabolomics_study is not None
    assert isinstance(sample_metabolomics_study, str)
    assert len(sample_metabolomics_study) > 0
    assert "metabolomics" in sample_metabolomics_study.lower()
    assert "clinical" in sample_metabolomics_study.lower()
    

def test_mock_data_loading(mock_metabolites_data: Dict[str, Any]):
    """Test loading of mock metabolites data."""
    assert mock_metabolites_data is not None
    assert isinstance(mock_metabolites_data, dict)
    assert "metabolite_database" in mock_metabolites_data
    
    db = mock_metabolites_data["metabolite_database"]
    assert "version" in db
    assert "metabolites" in db
    assert isinstance(db["metabolites"], list)
    assert len(db["metabolites"]) > 0
    
    # Validate first metabolite structure
    metabolite = db["metabolites"][0]
    required_fields = ["id", "name", "formula", "molecular_weight"]
    for field in required_fields:
        assert field in metabolite


def test_temp_directory_fixture(test_temp_dir: Path):
    """Test temporary directory fixture creation and cleanup."""
    assert test_temp_dir.exists()
    assert test_temp_dir.is_dir()
    
    # Test writing to temp directory
    test_file = test_temp_dir / "test_file.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    assert test_file.read_text() == "test content"


def test_database_fixture(test_cost_db: sqlite3.Connection):
    """Test database fixture creation and schema."""
    assert test_cost_db is not None
    
    # Test basic database operations
    cursor = test_cost_db.cursor()
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    assert len(tables) > 0  # Should have at least some tables from schema
    
    # Test basic operations
    try:
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        assert result[0] == 1
    except Exception as e:
        pytest.fail(f"Basic database operation failed: {e}")


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

def test_integrated_manager_functionality(integrated_test_data_manager: TestDataManager):
    """Test integrated test data manager functionality."""
    assert integrated_test_data_manager is not None
    
    # Test manager registration systems
    temp_dir = Path(tempfile.mkdtemp())
    integrated_test_data_manager.register_temp_dir(temp_dir)
    assert temp_dir in integrated_test_data_manager.temp_dirs
    
    # Test cleanup callback system
    cleanup_called = False
    def test_cleanup():
        nonlocal cleanup_called
        cleanup_called = True
    
    integrated_test_data_manager.add_cleanup_callback(test_cleanup)
    integrated_test_data_manager.cleanup_all()
    assert cleanup_called


def test_enhanced_pdf_data_fixture(enhanced_pdf_data: Dict[str, Any]):
    """Test enhanced PDF data fixture with metadata."""
    assert enhanced_pdf_data is not None
    assert isinstance(enhanced_pdf_data, dict)
    assert len(enhanced_pdf_data) > 0
    
    # Check structure of first PDF data entry
    first_key = next(iter(enhanced_pdf_data.keys()))
    pdf_entry = enhanced_pdf_data[first_key]
    
    required_fields = ["content", "metadata", "checksum", "size_bytes"]
    for field in required_fields:
        assert field in pdf_entry
    
    # Validate content
    assert isinstance(pdf_entry["content"], str)
    assert len(pdf_entry["content"]) > 0
    
    # Validate metadata
    metadata = pdf_entry["metadata"]
    assert "study_id" in metadata
    assert "title" in metadata
    assert "sample_size" in metadata


def test_comprehensive_mock_data_fixture(comprehensive_mock_data: Dict[str, Any]):
    """Test comprehensive mock data fixture."""
    assert comprehensive_mock_data is not None
    
    # Check main categories
    required_categories = ["api_responses", "system_states", "performance_data"]
    for category in required_categories:
        assert category in comprehensive_mock_data
    
    # Validate API responses
    api_responses = comprehensive_mock_data["api_responses"]
    assert "success" in api_responses
    assert "failure" in api_responses
    
    success_response = api_responses["success"]
    assert "status" in success_response
    assert success_response["status"] == "success"
    
    failure_response = api_responses["failure"]
    assert "status" in failure_response
    assert failure_response["status"] == "error"
    
    # Validate system states
    system_states = comprehensive_mock_data["system_states"]
    assert "cost_monitor_healthy" in system_states
    assert "lightrag_healthy" in system_states
    
    healthy_state = system_states["cost_monitor_healthy"]
    assert "status" in healthy_state
    assert healthy_state["status"] == "healthy"


# =====================================================================
# ASYNC FIXTURE TESTS
# =====================================================================

@pytest.mark.asyncio
async def test_async_test_data_manager(async_test_data_manager: AsyncTestDataManager):
    """Test async test data manager functionality."""
    assert async_test_data_manager is not None
    
    # Test async data loading
    async def sample_loader():
        await asyncio.sleep(0.1)  # Simulate async operation
        return {"test": "data", "loaded": True}
    
    result = await async_test_data_manager.load_test_data_async(
        "test_type", "test_key", sample_loader
    )
    
    assert result is not None
    assert result["test"] == "data"
    assert result["loaded"] is True


@pytest.mark.asyncio
async def test_async_biomedical_data_fixture(async_biomedical_data: Dict[str, Any]):
    """Test async biomedical data fixture."""
    assert async_biomedical_data is not None
    
    # Check main categories
    required_categories = ["compounds", "studies", "mock_responses", "metadata"]
    for category in required_categories:
        assert category in async_biomedical_data
    
    # Validate compounds
    compounds = async_biomedical_data["compounds"]
    assert isinstance(compounds, list)
    assert len(compounds) > 0
    
    first_compound = compounds[0]
    assert "id" in first_compound
    assert "name" in first_compound
    assert "formula" in first_compound
    
    # Validate studies
    studies = async_biomedical_data["studies"]
    assert isinstance(studies, list)
    assert len(studies) > 0
    
    first_study = studies[0]
    assert "study_id" in first_study
    assert "title" in first_study
    
    # Validate metadata
    metadata = async_biomedical_data["metadata"]
    assert "loaded_at" in metadata
    assert "load_method" in metadata
    assert metadata["load_method"] == "async"


@pytest.mark.asyncio
async def test_async_database_creation(async_test_data_manager: AsyncTestDataManager):
    """Test async database creation."""
    schema_sql = """
    CREATE TABLE test_table (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    conn = await async_test_data_manager.create_async_test_database(schema_sql)
    assert conn is not None
    
    # Test database operations
    cursor = conn.cursor()
    cursor.execute("INSERT INTO test_table (name) VALUES ('test_entry');")
    conn.commit()
    
    cursor.execute("SELECT * FROM test_table;")
    results = cursor.fetchall()
    assert len(results) == 1
    assert results[0][1] == "test_entry"


# =====================================================================
# UTILITY CLASS TESTS
# =====================================================================

def test_test_data_factory():
    """Test TestDataFactory functionality."""
    factory = TestDataFactory(seed=42)  # Use seed for reproducible tests
    
    # Test compound generation
    compound = factory.generate_compound()
    assert isinstance(compound, BiochemicalCompound)
    assert compound.id.startswith("met_")
    assert len(compound.name) > 0
    assert compound.molecular_weight > 0
    
    # Test database generation
    db = factory.generate_compound_database(count=5)
    assert "metabolite_database" in db
    assert len(db["metabolite_database"]["metabolites"]) == 5
    
    # Test clinical study generation
    study = factory.generate_clinical_study()
    assert isinstance(study, ClinicalStudyData)
    assert study.study_id.startswith("STUDY_")
    assert study.sample_size > 0
    assert len(study.compounds_studied) > 0


def test_data_validation_suite():
    """Test DataValidationSuite functionality."""
    validator = DataValidationSuite()
    
    # Test valid metabolite data
    valid_metabolite = {
        "id": "met_001",
        "name": "test_metabolite",
        "formula": "C6H12O6",
        "molecular_weight": 180.16
    }
    
    result = validator.validate_metabolite_data(valid_metabolite)
    assert result is True
    
    # Test invalid metabolite data (missing required field)
    invalid_metabolite = {
        "id": "met_002",
        "name": "test_metabolite"
        # Missing formula and molecular_weight
    }
    
    result = validator.validate_metabolite_data(invalid_metabolite)
    assert result is False
    
    # Get validation report
    report = validator.get_validation_report()
    assert "timestamp" in report
    assert "total_validations" in report
    assert report["total_validations"] == 2


def test_mock_data_generator():
    """Test MockDataGenerator functionality."""
    generator = MockDataGenerator()
    
    # Test API response generation
    success_response = generator.generate_api_response_mock("openai_chat", success=True)
    assert success_response["status"] == "success"
    assert "data" in success_response
    
    failure_response = generator.generate_api_response_mock("openai_chat", success=False)
    assert failure_response["status"] == "error"
    assert "error" in failure_response
    
    # Test system state generation
    healthy_state = generator.generate_system_state_mock("cost_monitor", healthy=True)
    assert healthy_state["status"] == "healthy"
    assert "current_cost" in healthy_state
    
    unhealthy_state = generator.generate_system_state_mock("cost_monitor", healthy=False)
    assert unhealthy_state["status"] == "budget_exceeded"
    
    # Test performance data generation
    perf_data = generator.generate_performance_test_data("normal_load", duration_seconds=10)
    assert perf_data["scenario"] == "normal_load"
    assert perf_data["duration_seconds"] == 10
    assert "metrics" in perf_data
    assert "summary" in perf_data


# =====================================================================
# ERROR HANDLING TESTS
# =====================================================================

def test_fixture_error_handling():
    """Test error handling in fixtures."""
    # Test with invalid configuration
    config = TestDataConfig()
    config.validate_data = True
    
    manager = TestDataManager(config)
    
    # Test cleanup with no resources
    manager.cleanup_all()  # Should not raise exception
    
    # Test registration of invalid paths
    invalid_path = Path("/nonexistent/path/test")
    manager.register_temp_dir(invalid_path)
    assert invalid_path in manager.temp_dirs


def test_missing_test_data_files():
    """Test behavior when test data files are missing."""
    # Test with non-existent file path
    non_existent_path = Path("/nonexistent/test_data.json")
    
    from .test_data_utilities import load_test_data_safe
    
    # Should return default value when file doesn't exist
    result = load_test_data_safe(non_existent_path, default="default_value")
    assert result == "default_value"
    
    # Should return None when no default provided
    result = load_test_data_safe(non_existent_path)
    assert result is None


# =====================================================================
# PERFORMANCE TESTS
# =====================================================================

def test_performance_optimizer():
    """Test PerformanceOptimizer functionality."""
    optimizer = PerformanceOptimizer()
    
    # Test load time profiling
    def sample_loader():
        time.sleep(0.01)  # Simulate loading time
        return "loaded_data"
    
    profiled_loader = optimizer.profile_data_loading("test_data", sample_loader)
    result = profiled_loader()
    
    assert result == "loaded_data"
    assert "test_data" in optimizer.load_times
    assert optimizer.load_times["test_data"] > 0
    
    # Test performance report
    report = optimizer.get_performance_report()
    assert "total_data_items" in report
    assert "total_load_time_seconds" in report
    assert report["total_data_items"] == 1


@pytest.mark.performance
def test_fixture_loading_performance(test_data_manager: TestDataManager):
    """Test performance of fixture loading under load."""
    start_time = time.time()
    
    # Create multiple temporary directories
    temp_dirs = []
    for i in range(10):
        temp_dir = Path(tempfile.mkdtemp(prefix=f"perf_test_{i}_"))
        test_data_manager.register_temp_dir(temp_dir)
        temp_dirs.append(temp_dir)
    
    # Measure registration time
    registration_time = time.time() - start_time
    
    # Cleanup and measure cleanup time
    cleanup_start = time.time()
    test_data_manager.cleanup_all()
    cleanup_time = time.time() - cleanup_start
    
    # Performance assertions
    assert registration_time < 1.0  # Should complete within 1 second
    assert cleanup_time < 1.0  # Cleanup should also be fast
    
    # Verify cleanup worked
    for temp_dir in temp_dirs:
        assert not temp_dir.exists()


# =====================================================================
# CONCURRENCY TESTS
# =====================================================================

def test_concurrent_fixture_access():
    """Test concurrent access to fixtures."""
    config = TestDataConfig()
    manager = TestDataManager(config)
    
    results = []
    errors = []
    
    def worker_function(worker_id: int):
        try:
            # Each worker creates temp directories
            temp_dir = Path(tempfile.mkdtemp(prefix=f"worker_{worker_id}_"))
            manager.register_temp_dir(temp_dir)
            
            # Simulate some work
            time.sleep(0.01)
            
            results.append(worker_id)
        except Exception as e:
            errors.append((worker_id, e))
    
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_function, args=(i,))
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Verify results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 5
    assert len(manager.temp_dirs) == 5
    
    # Cleanup
    manager.cleanup_all()


# =====================================================================
# INTEGRATION WITH EXISTING FIXTURES
# =====================================================================

@pytest.mark.skipif(
    not Path(__file__).parent.joinpath("conftest.py").exists(),
    reason="Main conftest.py not available"
)
def test_integration_with_existing_conftest(test_data_manager: TestDataManager):
    """Test integration with existing conftest.py fixtures."""
    # This test verifies that our fixtures work alongside existing ones
    assert test_data_manager is not None
    
    # Test that we can access our manager's functionality
    assert hasattr(test_data_manager, 'cleanup_all')
    assert hasattr(test_data_manager, 'register_temp_dir')
    assert hasattr(test_data_manager, 'loaded_data')
    
    # Verify integration doesn't break existing patterns
    temp_dir = Path(tempfile.mkdtemp())
    test_data_manager.register_temp_dir(temp_dir)
    
    assert temp_dir in test_data_manager.temp_dirs


# =====================================================================
# CLEANUP VERIFICATION
# =====================================================================

def test_fixture_cleanup_verification():
    """Verify that fixture cleanup actually works."""
    # Create a temporary manager for cleanup testing
    config = TestDataConfig(auto_cleanup=True)
    manager = TestDataManager(config)
    
    # Create test resources
    temp_dir = Path(tempfile.mkdtemp(prefix="cleanup_test_"))
    test_file = temp_dir / "test_file.txt"
    test_file.write_text("cleanup test")
    
    # Register for cleanup
    manager.register_temp_dir(temp_dir)
    
    # Verify resources exist before cleanup
    assert temp_dir.exists()
    assert test_file.exists()
    
    # Perform cleanup
    manager.cleanup_all()
    
    # Verify resources were cleaned up
    assert not temp_dir.exists()
    assert not test_file.exists()


@pytest.mark.asyncio
async def test_async_fixture_cleanup_verification():
    """Verify async fixture cleanup."""
    config = IntegrationConfig()
    async_manager = AsyncTestDataManager(config)
    
    # Create test database
    schema_sql = "CREATE TABLE test (id INTEGER);"
    conn = await async_manager.create_async_test_database(schema_sql)
    
    # Verify connection works
    cursor = conn.cursor()
    cursor.execute("SELECT 1;")
    result = cursor.fetchone()
    assert result[0] == 1
    
    # Perform async cleanup
    await async_manager.cleanup_async()
    
    # Connection should be closed (attempting to use it should fail)
    with pytest.raises(Exception):
        cursor.execute("SELECT 1;")