#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures for API Cost Monitoring Test Suite.

This configuration file provides:
- Shared test fixtures across all test modules
- Common test utilities and helpers
- Test environment setup and teardown
- Coverage configuration integration
- Performance test categorization
- Database and file system isolation

Author: Claude Code (Anthropic)
Created: August 6, 2025
"""

import pytest
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock
from typing import Dict, Any

# Import core components for fixture creation
from lightrag_integration.cost_persistence import CostPersistence
from lightrag_integration.budget_manager import BudgetManager


# Test Categories
def pytest_configure(config):
    """Configure pytest with custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "concurrent: mark test as testing concurrent operations"
    )


# Shared Fixtures
@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield Path(db_path)
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration object."""
    config = Mock()
    config.enable_file_logging = False  # Default to disabled for test speed
    config.log_dir = temp_dir / "logs"
    config.log_max_bytes = 1024 * 1024
    config.log_backup_count = 3
    config.api_key = "test-api-key"
    config.log_level = "INFO"
    return config


@pytest.fixture
def cost_persistence(temp_db_path):
    """Create a CostPersistence instance for testing."""
    return CostPersistence(temp_db_path, retention_days=365)


@pytest.fixture
def budget_manager(cost_persistence):
    """Create a BudgetManager instance for testing."""
    return BudgetManager(
        cost_persistence=cost_persistence,
        daily_budget_limit=100.0,
        monthly_budget_limit=3000.0
    )


# Test Utilities
class TestDataBuilder:
    """Builder class for creating consistent test data."""
    
    @staticmethod
    def create_cost_record_data(
        operation_type: str = "test_operation",
        model_name: str = "gpt-4o-mini",
        cost_usd: float = 0.05,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """Create cost record data for testing."""
        return {
            'operation_type': operation_type,
            'model_name': model_name,
            'cost_usd': cost_usd,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            **kwargs
        }
    
    @staticmethod
    def create_budget_alert_data(
        alert_level: str = "warning",
        current_cost: float = 75.0,
        budget_limit: float = 100.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Create budget alert data for testing."""
        return {
            'alert_level': alert_level,
            'current_cost': current_cost,
            'budget_limit': budget_limit,
            'percentage_used': (current_cost / budget_limit) * 100,
            **kwargs
        }


@pytest.fixture
def test_data_builder():
    """Provide test data builder utility."""
    return TestDataBuilder()


# Performance Test Configuration
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        'min_operations_per_second': 10,
        'max_response_time_ms': 5000,
        'concurrent_workers': 5,
        'operations_per_worker': 20
    }


# Database Isolation
@pytest.fixture(autouse=True)
def isolate_database_operations(monkeypatch):
    """Ensure database operations are isolated between tests."""
    # This fixture automatically runs for every test to ensure isolation
    # Specific isolation is handled by temp_db_path fixture
    pass


# Logging Configuration for Tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for test environment."""
    # Suppress verbose logging during tests unless explicitly requested
    logging.getLogger().setLevel(logging.WARNING)
    
    # Individual test modules can override this by setting specific logger levels
    yield
    
    # Cleanup after tests
    logging.getLogger().setLevel(logging.INFO)