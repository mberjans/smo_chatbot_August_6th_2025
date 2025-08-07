#!/usr/bin/env python3
"""
Test Data Fixtures for Clinical Metabolomics Oracle LightRAG Integration.

This module provides comprehensive pytest fixtures that integrate with the existing
test infrastructure while utilizing the new test_data/ directory structure. It bridges
the gap between the established fixture system and the structured test data management.

Key Features:
1. Integration with existing conftest.py and fixture infrastructure
2. Automatic loading and management of test data from test_data/ directories
3. Async support for LightRAG integration testing
4. Comprehensive cleanup mechanisms with proper teardown
5. Support for both unit and integration testing patterns
6. Error handling and data validation
7. Temporary directory management with lifecycle control

Components:
- TestDataManager: Central coordinator for test data operations
- PDF fixtures: Loading, validation, and management of PDF test data
- Database fixtures: Schema loading, sample data, and cleanup
- Mock data fixtures: API responses, system states, biomedical data
- Temporary directory fixtures: Staging, processing, cleanup areas
- Log file fixtures: Template loading and test log management
- Async fixtures: Support for async test operations
- Helper functions: Utilities for test data lifecycle management

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
import sqlite3
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, AsyncGenerator, Generator
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager, contextmanager
import os
import uuid
import threading
from datetime import datetime
import warnings

# Test data directory constants
TEST_DATA_ROOT = Path(__file__).parent / "test_data"
PDF_DATA_DIR = TEST_DATA_ROOT / "pdfs"
DATABASE_DATA_DIR = TEST_DATA_ROOT / "databases"
MOCK_DATA_DIR = TEST_DATA_ROOT / "mocks"
LOG_DATA_DIR = TEST_DATA_ROOT / "logs"
TEMP_DATA_DIR = TEST_DATA_ROOT / "temp"
UTILITIES_DIR = TEST_DATA_ROOT / "utilities"


# =====================================================================
# CORE TEST DATA MANAGEMENT
# =====================================================================

@dataclass
class TestDataConfig:
    """Configuration for test data management."""
    use_temp_dirs: bool = True
    auto_cleanup: bool = True
    validate_data: bool = True
    async_support: bool = True
    performance_monitoring: bool = False
    max_temp_size_mb: int = 100
    cleanup_on_failure: bool = True


@dataclass
class TestDataInfo:
    """Information about loaded test data."""
    data_type: str
    source_path: Path
    loaded_at: datetime
    size_bytes: int
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestDataManager:
    """Central manager for test data operations."""
    
    def __init__(self, config: TestDataConfig = None):
        self.config = config or TestDataConfig()
        self.loaded_data: Dict[str, TestDataInfo] = {}
        self.temp_dirs: List[Path] = []
        self.db_connections: List[sqlite3.Connection] = []
        self.cleanup_callbacks: List[Callable] = []
        self._lock = threading.Lock()
        
    def register_temp_dir(self, temp_dir: Path) -> None:
        """Register temporary directory for cleanup."""
        with self._lock:
            self.temp_dirs.append(temp_dir)
            
    def register_db_connection(self, conn: sqlite3.Connection) -> None:
        """Register database connection for cleanup."""
        with self._lock:
            self.db_connections.append(conn)
            
    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add cleanup callback."""
        with self._lock:
            self.cleanup_callbacks.append(callback)
            
    def cleanup_all(self) -> None:
        """Perform comprehensive cleanup."""
        with self._lock:
            # Execute cleanup callbacks
            for callback in reversed(self.cleanup_callbacks):
                try:
                    callback()
                except Exception as e:
                    logging.warning(f"Cleanup callback failed: {e}")
                    
            # Close database connections
            for conn in self.db_connections:
                try:
                    conn.close()
                except Exception as e:
                    logging.warning(f"DB connection cleanup failed: {e}")
                    
            # Remove temporary directories
            for temp_dir in self.temp_dirs:
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logging.warning(f"Temp directory cleanup failed: {e}")
                    
            # Clear tracking lists
            self.cleanup_callbacks.clear()
            self.db_connections.clear()
            self.temp_dirs.clear()
            self.loaded_data.clear()
            
    def create_async_test_database(self, schema_sql: str) -> sqlite3.Connection:
        """Create async test database - compatibility method for integration tests."""
        # Create unique test database
        db_path = Path(tempfile.mkdtemp()) / f"async_test_{uuid.uuid4().hex[:8]}.db"
        
        # Create connection and schema
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.executescript(schema_sql)
        conn.commit()
        
        # Register for cleanup
        self.register_db_connection(conn)
        self.add_cleanup_callback(lambda: db_path.unlink(missing_ok=True))
        
        return conn


# =====================================================================
# CORE FIXTURES
# =====================================================================

@pytest.fixture(scope="session")
def test_data_config() -> TestDataConfig:
    """Provide test data configuration."""
    return TestDataConfig()


@pytest.fixture(scope="session")
def session_test_data_manager(test_data_config: TestDataConfig) -> Generator[TestDataManager, None, None]:
    """Provide test data manager with session-level cleanup."""
    manager = TestDataManager(test_data_config)
    try:
        yield manager
    finally:
        manager.cleanup_all()


@pytest.fixture(scope="function")
def test_data_manager() -> Generator[TestDataManager, None, None]:
    """Provide test data manager with function-level cleanup."""
    config = TestDataConfig()
    manager = TestDataManager(config)
    try:
        yield manager
    finally:
        manager.cleanup_all()


# =====================================================================
# PDF DATA FIXTURES
# =====================================================================

@pytest.fixture(scope="session")
def pdf_samples_dir() -> Path:
    """Provide path to PDF samples directory."""
    return PDF_DATA_DIR / "samples"


@pytest.fixture(scope="session")
def pdf_templates_dir() -> Path:
    """Provide path to PDF templates directory."""
    return PDF_DATA_DIR / "templates"


@pytest.fixture(scope="session")
def pdf_corrupted_dir() -> Path:
    """Provide path to corrupted PDF directory."""
    return PDF_DATA_DIR / "corrupted"


@pytest.fixture
def sample_metabolomics_study(pdf_samples_dir: Path) -> str:
    """Load sample metabolomics study content."""
    study_file = pdf_samples_dir / "sample_metabolomics_study.txt"
    if not study_file.exists():
        pytest.skip(f"Sample metabolomics study not found: {study_file}")
    return study_file.read_text(encoding="utf-8")


@pytest.fixture
def sample_clinical_trial(pdf_samples_dir: Path) -> str:
    """Load sample clinical trial content."""
    trial_file = pdf_samples_dir / "sample_clinical_trial.txt"
    if not trial_file.exists():
        pytest.skip(f"Sample clinical trial not found: {trial_file}")
    return trial_file.read_text(encoding="utf-8")


@pytest.fixture
def pdf_template(pdf_templates_dir: Path) -> str:
    """Load biomedical PDF template."""
    template_file = pdf_templates_dir / "minimal_biomedical_template.txt"
    if not template_file.exists():
        pytest.skip(f"PDF template not found: {template_file}")
    return template_file.read_text(encoding="utf-8")


@pytest.fixture
def corrupted_pdf_content(pdf_corrupted_dir: Path) -> str:
    """Load corrupted PDF content for error testing."""
    corrupted_file = pdf_corrupted_dir / "corrupted_sample.txt"
    if not corrupted_file.exists():
        pytest.skip(f"Corrupted sample not found: {corrupted_file}")
    return corrupted_file.read_text(encoding="utf-8")


@pytest.fixture
def pdf_test_files(test_data_manager: TestDataManager, pdf_samples_dir: Path) -> Dict[str, str]:
    """Load all PDF test files."""
    files = {}
    for pdf_file in pdf_samples_dir.glob("*.txt"):
        try:
            content = pdf_file.read_text(encoding="utf-8")
            files[pdf_file.stem] = content
            
            # Register data info
            test_data_manager.loaded_data[f"pdf_{pdf_file.stem}"] = TestDataInfo(
                data_type="pdf_sample",
                source_path=pdf_file,
                loaded_at=datetime.now(),
                size_bytes=len(content.encode("utf-8")),
                metadata={"filename": pdf_file.name}
            )
        except Exception as e:
            logging.warning(f"Failed to load PDF test file {pdf_file}: {e}")
            
    return files


# =====================================================================
# DATABASE FIXTURES
# =====================================================================

@pytest.fixture(scope="session")
def database_schemas_dir() -> Path:
    """Provide path to database schemas directory."""
    return DATABASE_DATA_DIR / "schemas"


@pytest.fixture(scope="session")
def database_samples_dir() -> Path:
    """Provide path to database samples directory."""
    return DATABASE_DATA_DIR / "samples"


@pytest.fixture(scope="session")
def database_test_dir() -> Path:
    """Provide path to test databases directory."""
    return DATABASE_DATA_DIR / "test_dbs"


@pytest.fixture
def cost_tracking_schema(database_schemas_dir: Path) -> str:
    """Load cost tracking database schema."""
    schema_file = database_schemas_dir / "cost_tracking_schema.sql"
    if not schema_file.exists():
        pytest.skip(f"Cost tracking schema not found: {schema_file}")
    return schema_file.read_text(encoding="utf-8")


@pytest.fixture
def knowledge_base_schema(database_schemas_dir: Path) -> str:
    """Load knowledge base database schema."""
    schema_file = database_schemas_dir / "knowledge_base_schema.sql"
    if not schema_file.exists():
        pytest.skip(f"Knowledge base schema not found: {schema_file}")
    return schema_file.read_text(encoding="utf-8")


@pytest.fixture
def test_cost_db(
    test_data_manager,
    database_test_dir: Path,
    database_schemas_dir: Path
) -> sqlite3.Connection:
    """Create test cost tracking database."""
    # Ensure test directory exists
    database_test_dir.mkdir(parents=True, exist_ok=True)
    
    # Load schema
    schema_file = database_schemas_dir / "cost_tracking_schema.sql"
    if not schema_file.exists():
        # Create a basic schema for testing
        cost_tracking_schema = """
        CREATE TABLE IF NOT EXISTS cost_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            operation_type TEXT NOT NULL,
            cost_amount REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS processing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id TEXT,
            file_name TEXT,
            size_bytes INTEGER,
            processing_cost REAL
        );
        """
    else:
        cost_tracking_schema = schema_file.read_text(encoding="utf-8")
    
    # Create unique test database
    db_path = database_test_dir / f"test_cost_{uuid.uuid4().hex[:8]}.db"
    
    # Create connection and schema
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(cost_tracking_schema)
    conn.commit()
    
    # Register for cleanup
    test_data_manager.register_db_connection(conn)
    test_data_manager.add_cleanup_callback(lambda: db_path.unlink(missing_ok=True))
    
    return conn


@pytest.fixture
def test_knowledge_db(
    test_data_manager,
    database_test_dir: Path,
    database_schemas_dir: Path
) -> sqlite3.Connection:
    """Create test knowledge base database."""
    # Ensure test directory exists
    database_test_dir.mkdir(parents=True, exist_ok=True)
    
    # Load schema
    schema_file = database_schemas_dir / "knowledge_base_schema.sql"
    if not schema_file.exists():
        # Create a basic schema for testing
        knowledge_base_schema = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL,
            properties TEXT
        );
        """
    else:
        knowledge_base_schema = schema_file.read_text(encoding="utf-8")
    
    # Create unique test database
    db_path = database_test_dir / f"test_kb_{uuid.uuid4().hex[:8]}.db"
    
    # Create connection and schema
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(knowledge_base_schema)
    conn.commit()
    
    # Register for cleanup
    test_data_manager.register_db_connection(conn)
    test_data_manager.add_cleanup_callback(lambda: db_path.unlink(missing_ok=True))
    
    return conn


# =====================================================================
# MOCK DATA FIXTURES
# =====================================================================

@pytest.fixture(scope="session")
def mock_biomedical_dir() -> Path:
    """Provide path to mock biomedical data directory."""
    return MOCK_DATA_DIR / "biomedical_data"


@pytest.fixture(scope="session")
def mock_api_responses_dir() -> Path:
    """Provide path to mock API responses directory."""
    return MOCK_DATA_DIR / "api_responses"


@pytest.fixture(scope="session")
def mock_state_data_dir() -> Path:
    """Provide path to mock system state data directory."""
    return MOCK_DATA_DIR / "state_data"


@pytest.fixture
def mock_metabolites_data(mock_biomedical_dir: Path) -> Dict[str, Any]:
    """Load mock metabolites data."""
    metabolites_file = mock_biomedical_dir / "mock_metabolites.json"
    if not metabolites_file.exists():
        pytest.skip(f"Mock metabolites data not found: {metabolites_file}")
    
    with open(metabolites_file, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_openai_responses(mock_api_responses_dir: Path) -> Dict[str, Any]:
    """Load mock OpenAI API responses."""
    responses_file = mock_api_responses_dir / "openai_api_responses.json"
    if not responses_file.exists():
        pytest.skip(f"Mock OpenAI responses not found: {responses_file}")
    
    with open(responses_file, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_system_states(mock_state_data_dir: Path) -> Dict[str, Any]:
    """Load mock system states data."""
    states_file = mock_state_data_dir / "mock_system_states.json"
    if not states_file.exists():
        pytest.skip(f"Mock system states not found: {states_file}")
    
    with open(states_file, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# TEMPORARY DIRECTORY FIXTURES
# =====================================================================

@pytest.fixture
def test_temp_dir(test_data_manager: TestDataManager) -> Path:
    """Provide temporary directory for test use."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_data_"))
    test_data_manager.register_temp_dir(temp_dir)
    return temp_dir


@pytest.fixture
def test_staging_dir(test_data_manager: TestDataManager) -> Path:
    """Provide staging directory for test data preparation."""
    staging_dir = TEMP_DATA_DIR / "staging" / f"test_{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    test_data_manager.register_temp_dir(staging_dir)
    return staging_dir


@pytest.fixture
def test_processing_dir(test_data_manager: TestDataManager) -> Path:
    """Provide processing directory for test operations."""
    processing_dir = TEMP_DATA_DIR / "processing" / f"test_{uuid.uuid4().hex[:8]}"
    processing_dir.mkdir(parents=True, exist_ok=True)
    test_data_manager.register_temp_dir(processing_dir)
    return processing_dir


@pytest.fixture
def test_output_dir(test_data_manager: TestDataManager) -> Path:
    """Provide output directory for test results."""
    output_dir = Path(tempfile.mkdtemp(prefix="test_output_"))
    test_data_manager.register_temp_dir(output_dir)
    return output_dir


# =====================================================================
# LOG FILE FIXTURES
# =====================================================================

@pytest.fixture(scope="session")
def log_templates_dir() -> Path:
    """Provide path to log templates directory."""
    return LOG_DATA_DIR / "templates"


@pytest.fixture(scope="session")
def log_configs_dir() -> Path:
    """Provide path to log configurations directory."""
    return LOG_DATA_DIR / "configs"


@pytest.fixture(scope="session")
def log_samples_dir() -> Path:
    """Provide path to log samples directory."""
    return LOG_DATA_DIR / "samples"


@pytest.fixture
def logging_config(log_configs_dir: Path) -> Dict[str, Any]:
    """Load logging configuration template."""
    config_file = log_configs_dir / "logging_config_template.json"
    if not config_file.exists():
        pytest.skip(f"Logging config not found: {config_file}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def test_log_file(test_data_manager: TestDataManager, test_temp_dir: Path) -> Path:
    """Create temporary log file for testing."""
    log_file = test_temp_dir / "test.log"
    log_file.touch()
    return log_file


# =====================================================================
# ASYNC FIXTURES
# =====================================================================

@pytest_asyncio.fixture
async def async_test_data_manager() -> AsyncGenerator[TestDataManager, None]:
    """Provide async test data manager."""
    config = TestDataConfig()
    manager = TestDataManager(config)
    try:
        yield manager
    finally:
        # Cleanup can be sync even in async fixture
        manager.cleanup_all()


@pytest_asyncio.fixture
async def async_temp_dir(async_test_data_manager: TestDataManager) -> AsyncGenerator[Path, None]:
    """Provide async temporary directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="async_test_"))
    async_test_data_manager.register_temp_dir(temp_dir)
    yield temp_dir


# =====================================================================
# HELPER FUNCTIONS AND UTILITIES
# =====================================================================

def create_test_pdf_content(title: str, content: str, metadata: Dict[str, Any] = None) -> str:
    """Create test PDF content with standard structure."""
    metadata = metadata or {}
    
    pdf_content = f"""SAMPLE BIOMEDICAL RESEARCH DOCUMENT

Title: {title}
Authors: {metadata.get('authors', 'Test Author')}
Journal: {metadata.get('journal', 'Test Journal')}
Year: {metadata.get('year', datetime.now().year)}

ABSTRACT
{content[:500]}...

INTRODUCTION
This is a test document created for Clinical Metabolomics Oracle testing purposes.

METHODS
Test methods and procedures.

RESULTS
Test results and findings.

CONCLUSIONS
Test conclusions and implications.

KEYWORDS: {', '.join(metadata.get('keywords', ['test', 'metabolomics', 'clinical']))}
"""
    return pdf_content


def validate_test_data_integrity(data_path: Path) -> bool:
    """Validate test data integrity."""
    if not data_path.exists():
        return False
        
    if data_path.is_file():
        # Check if file is readable
        try:
            data_path.read_text(encoding="utf-8")
            return True
        except Exception:
            return False
    elif data_path.is_dir():
        # Check if directory has expected structure
        return len(list(data_path.iterdir())) > 0
    
    return False


@contextmanager
def temporary_test_file(content: str, suffix: str = ".txt", prefix: str = "test_"):
    """Context manager for temporary test files."""
    with tempfile.NamedTemporaryFile(
        mode="w", 
        suffix=suffix, 
        prefix=prefix, 
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


@asynccontextmanager
async def async_temporary_test_file(content: str, suffix: str = ".txt", prefix: str = "async_test_"):
    """Async context manager for temporary test files."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        prefix=prefix,
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def load_test_data_safe(file_path: Path, default: Any = None) -> Any:
    """Safely load test data with fallback."""
    try:
        if file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return file_path.read_text(encoding="utf-8")
    except Exception as e:
        logging.warning(f"Failed to load test data from {file_path}: {e}")
        return default


def cleanup_test_artifacts(base_path: Path, patterns: List[str] = None) -> int:
    """Clean up test artifacts matching patterns."""
    patterns = patterns or ["test_*", "temp_*", "*_test.*", "*.tmp"]
    cleaned_count = 0
    
    for pattern in patterns:
        for artifact in base_path.glob(pattern):
            try:
                if artifact.is_file():
                    artifact.unlink()
                elif artifact.is_dir():
                    shutil.rmtree(artifact, ignore_errors=True)
                cleaned_count += 1
            except Exception as e:
                logging.warning(f"Failed to clean up {artifact}: {e}")
                
    return cleaned_count


# =====================================================================
# INTEGRATION HELPERS
# =====================================================================

def get_test_data_root() -> Path:
    """Get the root test data directory."""
    return TEST_DATA_ROOT


def ensure_test_data_dirs() -> None:
    """Ensure all test data directories exist."""
    directories = [
        PDF_DATA_DIR / "samples",
        PDF_DATA_DIR / "templates", 
        PDF_DATA_DIR / "corrupted",
        DATABASE_DATA_DIR / "schemas",
        DATABASE_DATA_DIR / "samples",
        DATABASE_DATA_DIR / "test_dbs",
        MOCK_DATA_DIR / "biomedical_data",
        MOCK_DATA_DIR / "api_responses",
        MOCK_DATA_DIR / "state_data",
        LOG_DATA_DIR / "templates",
        LOG_DATA_DIR / "configs",
        LOG_DATA_DIR / "samples",
        TEMP_DATA_DIR / "staging",
        TEMP_DATA_DIR / "processing",
        TEMP_DATA_DIR / "cleanup",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Initialize test data directories on import
ensure_test_data_dirs()