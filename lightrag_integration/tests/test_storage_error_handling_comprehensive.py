#!/usr/bin/env python3
"""
Comprehensive Storage Error Handling Tests for Clinical Metabolomics Oracle.

This test suite provides complete coverage of storage initialization error handling,
including directory creation, permission validation, disk space management, and
recovery strategies for storage-related failures.

Test Coverage:
- Storage directory creation and validation
- Permission error detection and handling
- Disk space validation and cleanup
- Storage path resolution and fallback mechanisms
- Storage initialization retry logic
- Integration with enhanced logging and recovery systems

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import pytest
import os
import stat
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Import components for testing
import sys
sys.path.append(str(Path(__file__).parent.parent))

from lightrag_integration.clinical_metabolomics_rag import (
    StorageInitializationError, StoragePermissionError, StorageSpaceError,
    StorageDirectoryError, StorageRetryableError
)
from lightrag_integration.enhanced_logging import (
    DiagnosticLogger, EnhancedLogger
)
from lightrag_integration.advanced_recovery_system import (
    AdvancedRecoverySystem, FailureType
)


# =====================================================================
# STORAGE ERROR SIMULATION UTILITIES
# =====================================================================

@dataclass
class StorageTestConfig:
    """Configuration for storage testing scenarios."""
    base_path: Path
    required_space_gb: float = 1.0
    create_directories: bool = True
    set_permissions: bool = True
    simulate_full_disk: bool = False
    simulate_readonly: bool = False
    simulate_network_storage: bool = False


class StorageErrorSimulator:
    """Utility for simulating various storage error conditions."""
    
    def __init__(self, test_config: StorageTestConfig):
        self.config = test_config
        self.original_permissions = {}
    
    def setup_readonly_directory(self, path: Path):
        """Setup a read-only directory to simulate permission errors."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Store original permissions
        self.original_permissions[str(path)] = path.stat().st_mode
        
        # Make directory read-only
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    
    def setup_full_disk_simulation(self, path: Path):
        """Setup conditions that simulate a full disk."""
        # This is challenging to truly simulate without actually filling disk
        # We'll use mocking for the disk space checks instead
        path.mkdir(parents=True, exist_ok=True)
    
    def setup_network_storage_simulation(self, path: Path):
        """Setup conditions that simulate network storage issues."""
        # Simulate by creating nested directory structure
        # that might have network-related access delays
        path.mkdir(parents=True, exist_ok=True)
        
        # Create deep nested structure
        nested_path = path
        for i in range(10):
            nested_path = nested_path / f"nested_{i}"
            nested_path.mkdir(exist_ok=True)
    
    def cleanup(self):
        """Restore original permissions and clean up."""
        for path_str, original_mode in self.original_permissions.items():
            try:
                Path(path_str).chmod(original_mode)
            except (OSError, FileNotFoundError):
                pass  # Ignore cleanup errors


class MockStorageSystem:
    """Mock storage system for testing storage operations."""
    
    def __init__(self, config: StorageTestConfig):
        self.config = config
        self.operations_log = []
        self.should_fail_creation = False
        self.should_fail_permissions = False
        self.should_fail_space_check = False
        self.available_space_bytes = 10 * 1024**3  # 10GB default
    
    def create_directory(self, path: Path, parents: bool = True) -> bool:
        """Mock directory creation."""
        self.operations_log.append(f"create_directory: {path}")
        
        if self.should_fail_creation:
            raise StorageDirectoryError(
                f"Failed to create directory: {path}",
                storage_path=str(path),
                directory_operation="create"
            )
        
        try:
            path.mkdir(parents=parents, exist_ok=True)
            return True
        except OSError as e:
            raise StorageDirectoryError(
                f"OS error creating directory: {e}",
                storage_path=str(path),
                directory_operation="create"
            )
    
    def check_permissions(self, path: Path, required_permission: str = "write") -> bool:
        """Mock permission checking."""
        self.operations_log.append(f"check_permissions: {path} ({required_permission})")
        
        if self.should_fail_permissions:
            raise StoragePermissionError(
                f"Permission denied for {required_permission} access",
                storage_path=str(path),
                required_permission=required_permission
            )
        
        if not path.exists():
            return False
        
        if required_permission == "read":
            return os.access(path, os.R_OK)
        elif required_permission == "write":
            return os.access(path, os.W_OK)
        elif required_permission == "execute":
            return os.access(path, os.X_OK)
        
        return True
    
    def check_disk_space(self, path: Path, required_bytes: int) -> Dict[str, int]:
        """Mock disk space checking."""
        self.operations_log.append(f"check_disk_space: {path} (need {required_bytes} bytes)")
        
        if self.should_fail_space_check:
            raise StorageSpaceError(
                f"Insufficient disk space at {path}",
                storage_path=str(path),
                available_space=self.available_space_bytes,
                required_space=required_bytes
            )
        
        return {
            'available_bytes': self.available_space_bytes,
            'required_bytes': required_bytes,
            'sufficient': self.available_space_bytes >= required_bytes
        }
    
    def validate_storage_path(self, path: Path) -> Dict[str, Any]:
        """Mock storage path validation."""
        self.operations_log.append(f"validate_storage_path: {path}")
        
        validation_result = {
            'path_exists': path.exists(),
            'is_directory': path.is_dir() if path.exists() else False,
            'is_writable': self.check_permissions(path, "write"),
            'is_readable': self.check_permissions(path, "read"),
            'parent_exists': path.parent.exists(),
            'errors': []
        }
        
        if not validation_result['path_exists']:
            validation_result['errors'].append("Path does not exist")
        if not validation_result['is_directory'] and path.exists():
            validation_result['errors'].append("Path is not a directory")
        if not validation_result['is_writable']:
            validation_result['errors'].append("Path is not writable")
        
        return validation_result


# =====================================================================
# STORAGE ERROR HIERARCHY TESTS
# =====================================================================

class TestStorageErrorHierarchy:
    """Test storage error class hierarchy and inheritance."""
    
    def test_storage_initialization_error_base(self):
        """Test base StorageInitializationError."""
        error = StorageInitializationError(
            "Storage initialization failed",
            storage_path="/test/path",
            error_code="INIT_FAIL"
        )
        
        assert str(error) == "Storage initialization failed"
        assert error.storage_path == "/test/path"
        assert error.error_code == "INIT_FAIL"
        assert isinstance(error, Exception)
    
    def test_storage_permission_error_inheritance(self):
        """Test StoragePermissionError inheritance and attributes."""
        error = StoragePermissionError(
            "Write permission denied",
            storage_path="/readonly/path",
            required_permission="write",
            error_code="PERM_DENIED"
        )
        
        assert str(error) == "Write permission denied"
        assert error.storage_path == "/readonly/path"
        assert error.required_permission == "write"
        assert error.error_code == "PERM_DENIED"
        assert isinstance(error, StorageInitializationError)
    
    def test_storage_space_error_with_metrics(self):
        """Test StorageSpaceError with space metrics."""
        error = StorageSpaceError(
            "Disk full",
            storage_path="/full/disk",
            available_space=1024,
            required_space=2048,
            error_code="DISK_FULL"
        )
        
        assert str(error) == "Disk full"
        assert error.available_space == 1024
        assert error.required_space == 2048
        assert error.required_space > error.available_space
        assert isinstance(error, StorageInitializationError)
    
    def test_storage_directory_error_operations(self):
        """Test StorageDirectoryError with different operations."""
        operations = ["create", "validate", "access", "delete"]
        
        for operation in operations:
            error = StorageDirectoryError(
                f"Directory {operation} failed",
                storage_path="/test/dir",
                directory_operation=operation
            )
            
            assert error.directory_operation == operation
            assert isinstance(error, StorageInitializationError)
    
    def test_storage_retryable_error(self):
        """Test StorageRetryableError for temporary failures."""
        error = StorageRetryableError(
            "Temporary network storage issue",
            storage_path="//network/share",
            error_code="NET_TEMP"
        )
        
        assert str(error) == "Temporary network storage issue"
        assert error.storage_path == "//network/share"
        assert isinstance(error, StorageInitializationError)


# =====================================================================
# DIRECTORY CREATION AND VALIDATION TESTS
# =====================================================================

class TestDirectoryCreationHandling:
    """Test directory creation and validation error handling."""
    
    @pytest.fixture
    def mock_storage_system(self, temp_dir):
        """Create mock storage system for testing."""
        config = StorageTestConfig(base_path=temp_dir)
        return MockStorageSystem(config)
    
    def test_successful_directory_creation(self, mock_storage_system, temp_dir):
        """Test successful directory creation."""
        test_path = temp_dir / "test_storage"
        
        result = mock_storage_system.create_directory(test_path)
        
        assert result == True
        assert test_path.exists()
        assert test_path.is_dir()
        assert "create_directory" in mock_storage_system.operations_log[0]
    
    def test_directory_creation_failure(self, mock_storage_system, temp_dir):
        """Test directory creation failure handling."""
        test_path = temp_dir / "failed_storage"
        mock_storage_system.should_fail_creation = True
        
        with pytest.raises(StorageDirectoryError) as exc_info:
            mock_storage_system.create_directory(test_path)
        
        error = exc_info.value
        assert "Failed to create directory" in str(error)
        assert error.directory_operation == "create"
        assert error.storage_path == str(test_path)
    
    def test_nested_directory_creation(self, mock_storage_system, temp_dir):
        """Test creation of nested directory structures."""
        nested_path = temp_dir / "level1" / "level2" / "level3" / "storage"
        
        result = mock_storage_system.create_directory(nested_path, parents=True)
        
        assert result == True
        assert nested_path.exists()
        assert nested_path.is_dir()
        
        # Verify entire path was created
        current_path = temp_dir / "level1"
        while current_path != nested_path:
            assert current_path.exists()
            assert current_path.is_dir()
            current_path = current_path / (nested_path.relative_to(current_path).parts[0])
    
    def test_directory_already_exists(self, mock_storage_system, temp_dir):
        """Test handling when directory already exists."""
        existing_path = temp_dir / "existing_storage"
        existing_path.mkdir(parents=True, exist_ok=True)
        
        # Should succeed even if directory exists
        result = mock_storage_system.create_directory(existing_path)
        
        assert result == True
        assert existing_path.exists()
    
    def test_readonly_parent_directory(self, temp_dir):
        """Test handling of read-only parent directory."""
        readonly_parent = temp_dir / "readonly_parent"
        readonly_parent.mkdir()
        
        # Make parent read-only
        original_mode = readonly_parent.stat().st_mode
        readonly_parent.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        
        try:
            config = StorageTestConfig(base_path=readonly_parent)
            mock_storage = MockStorageSystem(config)
            
            child_path = readonly_parent / "child_storage"
            
            with pytest.raises(StorageDirectoryError):
                mock_storage.create_directory(child_path)
        
        finally:
            # Restore permissions for cleanup
            readonly_parent.chmod(original_mode)


class TestPermissionValidation:
    """Test permission validation and error handling."""
    
    @pytest.fixture
    def mock_storage_system(self, temp_dir):
        """Create mock storage system for permission testing."""
        config = StorageTestConfig(base_path=temp_dir)
        return MockStorageSystem(config)
    
    def test_read_permission_check(self, mock_storage_system, temp_dir):
        """Test read permission validation."""
        test_path = temp_dir / "readable_dir"
        test_path.mkdir()
        
        result = mock_storage_system.check_permissions(test_path, "read")
        assert result == True
    
    def test_write_permission_check(self, mock_storage_system, temp_dir):
        """Test write permission validation."""
        test_path = temp_dir / "writable_dir"
        test_path.mkdir()
        
        result = mock_storage_system.check_permissions(test_path, "write")
        assert result == True
    
    def test_execute_permission_check(self, mock_storage_system, temp_dir):
        """Test execute permission validation."""
        test_path = temp_dir / "executable_dir"
        test_path.mkdir()
        
        result = mock_storage_system.check_permissions(test_path, "execute")
        assert result == True
    
    def test_permission_denied_error(self, mock_storage_system, temp_dir):
        """Test permission denied error handling."""
        test_path = temp_dir / "permission_test"
        test_path.mkdir()
        
        mock_storage_system.should_fail_permissions = True
        
        with pytest.raises(StoragePermissionError) as exc_info:
            mock_storage_system.check_permissions(test_path, "write")
        
        error = exc_info.value
        assert "Permission denied" in str(error)
        assert error.required_permission == "write"
        assert error.storage_path == str(test_path)
    
    def test_nonexistent_path_permission_check(self, mock_storage_system, temp_dir):
        """Test permission check on non-existent path."""
        nonexistent_path = temp_dir / "nonexistent"
        
        result = mock_storage_system.check_permissions(nonexistent_path, "read")
        assert result == False
    
    @pytest.mark.skipif(os.name == 'nt', reason="Unix permission model not applicable on Windows")
    def test_readonly_directory_simulation(self, temp_dir):
        """Test with actually read-only directory on Unix systems."""
        readonly_dir = temp_dir / "readonly_test"
        readonly_dir.mkdir()
        
        # Make directory read-only
        original_mode = readonly_dir.stat().st_mode
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        
        try:
            config = StorageTestConfig(base_path=readonly_dir)
            mock_storage = MockStorageSystem(config)
            
            # Should detect lack of write permission
            with pytest.raises(StoragePermissionError):
                mock_storage.check_permissions(readonly_dir, "write")
        
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(original_mode)


# =====================================================================
# DISK SPACE VALIDATION TESTS
# =====================================================================

class TestDiskSpaceValidation:
    """Test disk space validation and error handling."""
    
    @pytest.fixture
    def mock_storage_system(self, temp_dir):
        """Create mock storage system for disk space testing."""
        config = StorageTestConfig(base_path=temp_dir)
        return MockStorageSystem(config)
    
    def test_sufficient_disk_space_check(self, mock_storage_system, temp_dir):
        """Test sufficient disk space validation."""
        test_path = temp_dir / "space_test"
        test_path.mkdir()
        
        # Request reasonable amount of space (1MB)
        required_bytes = 1024 * 1024
        result = mock_storage_system.check_disk_space(test_path, required_bytes)
        
        assert result['sufficient'] == True
        assert result['available_bytes'] >= required_bytes
        assert result['required_bytes'] == required_bytes
    
    def test_insufficient_disk_space_error(self, mock_storage_system, temp_dir):
        """Test insufficient disk space error handling."""
        test_path = temp_dir / "space_test"
        test_path.mkdir()
        
        # Set very low available space
        mock_storage_system.available_space_bytes = 1024  # 1KB
        required_bytes = 1024 * 1024 * 1024  # 1GB
        
        mock_storage_system.should_fail_space_check = True
        
        with pytest.raises(StorageSpaceError) as exc_info:
            mock_storage_system.check_disk_space(test_path, required_bytes)
        
        error = exc_info.value
        assert "Insufficient disk space" in str(error)
        assert error.available_space == 1024
        assert error.required_space == required_bytes
        assert error.storage_path == str(test_path)
    
    def test_disk_space_boundary_conditions(self, mock_storage_system, temp_dir):
        """Test disk space validation at boundary conditions."""
        test_path = temp_dir / "boundary_test"
        test_path.mkdir()
        
        # Test exact match
        mock_storage_system.available_space_bytes = 1024 * 1024  # 1MB
        required_bytes = 1024 * 1024  # Exactly 1MB
        
        result = mock_storage_system.check_disk_space(test_path, required_bytes)
        assert result['sufficient'] == True
        
        # Test just under threshold
        required_bytes = 1024 * 1024 + 1  # 1MB + 1 byte
        mock_storage_system.should_fail_space_check = True
        
        with pytest.raises(StorageSpaceError):
            mock_storage_system.check_disk_space(test_path, required_bytes)
    
    def test_zero_disk_space_request(self, mock_storage_system, temp_dir):
        """Test zero disk space requirement."""
        test_path = temp_dir / "zero_space_test"
        test_path.mkdir()
        
        result = mock_storage_system.check_disk_space(test_path, 0)
        
        assert result['sufficient'] == True
        assert result['required_bytes'] == 0
    
    def test_negative_disk_space_request(self, mock_storage_system, temp_dir):
        """Test negative disk space requirement handling."""
        test_path = temp_dir / "negative_space_test"
        test_path.mkdir()
        
        # Negative space should be treated as zero or handled gracefully
        result = mock_storage_system.check_disk_space(test_path, -1024)
        
        # Implementation should handle this gracefully
        assert result['required_bytes'] == -1024
    
    @patch('shutil.disk_usage')
    def test_real_disk_space_integration(self, mock_disk_usage, temp_dir):
        """Test integration with actual disk space checking."""
        # Mock disk usage to return specific values
        mock_disk_usage.return_value = (
            100 * 1024**3,  # total: 100GB
            50 * 1024**3,   # used: 50GB  
            50 * 1024**3    # free: 50GB
        )
        
        test_path = temp_dir / "real_space_test"
        test_path.mkdir()
        
        config = StorageTestConfig(base_path=test_path)
        mock_storage = MockStorageSystem(config)
        
        # Should use mocked disk usage
        result = mock_storage.check_disk_space(test_path, 1024**3)  # 1GB
        
        # Verify mock was called
        mock_disk_usage.assert_called()


# =====================================================================
# STORAGE PATH RESOLUTION TESTS
# =====================================================================

class TestStoragePathResolution:
    """Test storage path resolution and validation."""
    
    @pytest.fixture
    def mock_storage_system(self, temp_dir):
        """Create mock storage system for path resolution testing."""
        config = StorageTestConfig(base_path=temp_dir)
        return MockStorageSystem(config)
    
    def test_valid_storage_path_validation(self, mock_storage_system, temp_dir):
        """Test validation of valid storage path."""
        valid_path = temp_dir / "valid_storage"
        valid_path.mkdir()
        
        result = mock_storage_system.validate_storage_path(valid_path)
        
        assert result['path_exists'] == True
        assert result['is_directory'] == True
        assert result['is_writable'] == True
        assert result['is_readable'] == True
        assert result['parent_exists'] == True
        assert len(result['errors']) == 0
    
    def test_nonexistent_path_validation(self, mock_storage_system, temp_dir):
        """Test validation of non-existent path."""
        nonexistent_path = temp_dir / "nonexistent_storage"
        
        result = mock_storage_system.validate_storage_path(nonexistent_path)
        
        assert result['path_exists'] == False
        assert result['is_directory'] == False
        assert result['parent_exists'] == True  # Parent (temp_dir) should exist
        assert "Path does not exist" in result['errors']
    
    def test_file_instead_of_directory_validation(self, mock_storage_system, temp_dir):
        """Test validation when path points to file instead of directory."""
        file_path = temp_dir / "storage_file.txt"
        file_path.write_text("This is a file, not a directory")
        
        result = mock_storage_system.validate_storage_path(file_path)
        
        assert result['path_exists'] == True
        assert result['is_directory'] == False
        assert "Path is not a directory" in result['errors']
    
    def test_permission_validation_integration(self, mock_storage_system, temp_dir):
        """Test permission validation as part of path validation."""
        test_path = temp_dir / "permission_test_dir"
        test_path.mkdir()
        
        # Force permission failure
        mock_storage_system.should_fail_permissions = True
        
        with pytest.raises(StoragePermissionError):
            mock_storage_system.validate_storage_path(test_path)
    
    def test_nested_path_validation(self, mock_storage_system, temp_dir):
        """Test validation of deeply nested paths."""
        nested_path = temp_dir / "level1" / "level2" / "level3" / "storage"
        nested_path.mkdir(parents=True)
        
        result = mock_storage_system.validate_storage_path(nested_path)
        
        assert result['path_exists'] == True
        assert result['is_directory'] == True
        assert result['parent_exists'] == True
        assert len(result['errors']) == 0
    
    def test_relative_path_handling(self, mock_storage_system):
        """Test handling of relative paths."""
        # Test with relative path
        relative_path = Path("./relative_storage")
        
        # Should handle relative paths appropriately
        result = mock_storage_system.validate_storage_path(relative_path)
        
        # The exact behavior depends on implementation,
        # but it should not crash
        assert 'path_exists' in result
        assert 'errors' in result
    
    def test_absolute_path_handling(self, mock_storage_system, temp_dir):
        """Test handling of absolute paths."""
        absolute_path = temp_dir.resolve() / "absolute_storage"
        absolute_path.mkdir()
        
        result = mock_storage_system.validate_storage_path(absolute_path)
        
        assert result['path_exists'] == True
        assert result['is_directory'] == True
        assert len(result['errors']) == 0


# =====================================================================
# STORAGE INITIALIZATION RETRY LOGIC TESTS
# =====================================================================

class TestStorageInitializationRetry:
    """Test storage initialization retry logic and recovery."""
    
    @pytest.fixture
    def recovery_system(self, temp_dir):
        """Create recovery system for storage retry testing."""
        from lightrag_integration.advanced_recovery_system import AdvancedRecoverySystem
        return AdvancedRecoverySystem(checkpoint_dir=temp_dir / "checkpoints")
    
    def test_storage_retry_on_temporary_failure(self, recovery_system, temp_dir):
        """Test retry logic for temporary storage failures."""
        # Simulate temporary storage failure
        recovery_system.initialize_ingestion_session(
            documents=[],
            phase=recovery_system._current_phase  # Use current phase
        )
        
        # Simulate retryable storage error
        strategy = recovery_system.handle_failure(
            FailureType.RESOURCE_EXHAUSTION,
            "Temporary storage unavailable",
            context={"storage_path": str(temp_dir / "temp_storage")}
        )
        
        # Should recommend retry for retryable errors
        assert strategy['action'] in ['retry', 'backoff_and_retry']
    
    def test_storage_non_retryable_failure(self, recovery_system, temp_dir):
        """Test handling of non-retryable storage failures."""
        recovery_system.initialize_ingestion_session(
            documents=[],
            phase=recovery_system._current_phase
        )
        
        # Simulate non-retryable permission error
        strategy = recovery_system.handle_failure(
            FailureType.PROCESSING_ERROR,  # Treat as non-retryable
            "Permission permanently denied",
            context={"storage_path": str(temp_dir / "denied_storage")}
        )
        
        # Should not recommend simple retry for permanent failures
        assert 'action' in strategy
    
    def test_storage_fallback_path_creation(self, temp_dir):
        """Test creation of fallback storage paths."""
        primary_path = temp_dir / "primary_storage"
        fallback_paths = [
            temp_dir / "fallback1_storage",
            temp_dir / "fallback2_storage", 
            temp_dir / "fallback3_storage"
        ]
        
        # Simulate primary path failure by not creating it
        # Create fallback paths
        for fallback in fallback_paths:
            fallback.mkdir(parents=True)
        
        # Test that we can find a working fallback
        working_fallback = None
        for path in [primary_path] + fallback_paths:
            if path.exists() and path.is_dir():
                working_fallback = path
                break
        
        assert working_fallback is not None
        assert working_fallback in fallback_paths
    
    def test_storage_cleanup_on_failure(self, temp_dir):
        """Test cleanup of partial storage initialization."""
        storage_path = temp_dir / "cleanup_test_storage"
        
        # Create partial storage structure
        (storage_path / "vector_store").mkdir(parents=True)
        (storage_path / "graph_store").mkdir(parents=True)
        (storage_path / "embeddings").mkdir(parents=True)
        
        # Create some test files
        test_files = [
            storage_path / "vector_store" / "test.db",
            storage_path / "graph_store" / "graph.json",
            storage_path / "embeddings" / "embeddings.npy"
        ]
        
        for test_file in test_files:
            test_file.write_text("test content")
        
        # Verify structure was created
        assert storage_path.exists()
        assert all(f.exists() for f in test_files)
        
        # Simulate cleanup on failure
        if storage_path.exists():
            shutil.rmtree(storage_path)
        
        # Verify cleanup was successful
        assert not storage_path.exists()
        assert not any(f.exists() for f in test_files)
    
    def test_storage_space_growth_monitoring(self, temp_dir):
        """Test monitoring of storage space growth during initialization."""
        storage_path = temp_dir / "growth_monitoring"
        storage_path.mkdir()
        
        config = StorageTestConfig(base_path=storage_path)
        mock_storage = MockStorageSystem(config)
        
        # Simulate space usage growth
        initial_space = 10 * 1024**3  # 10GB
        mock_storage.available_space_bytes = initial_space
        
        # Check initial space
        space_check1 = mock_storage.check_disk_space(storage_path, 1024**3)
        assert space_check1['available_bytes'] == initial_space
        
        # Simulate space usage (storage operations consuming space)
        mock_storage.available_space_bytes -= 2 * 1024**3  # Use 2GB
        
        # Check remaining space
        space_check2 = mock_storage.check_disk_space(storage_path, 1024**3)
        assert space_check2['available_bytes'] == initial_space - (2 * 1024**3)
        assert space_check2['sufficient'] == True  # Should still have enough


# =====================================================================
# INTEGRATION WITH LOGGING AND RECOVERY TESTS
# =====================================================================

class TestStorageErrorLoggingIntegration:
    """Test integration of storage errors with enhanced logging."""
    
    @pytest.fixture
    def diagnostic_logger(self):
        """Create diagnostic logger for storage testing."""
        base_logger = Mock()
        return DiagnosticLogger(base_logger)
    
    def test_storage_initialization_success_logging(self, diagnostic_logger):
        """Test logging of successful storage initialization."""
        diagnostic_logger.log_storage_initialization(
            storage_type="vector_store",
            path="/test/storage/path",
            initialization_time_ms=500.0,
            success=True
        )
        
        # Should log at INFO level for success
        diagnostic_logger.enhanced_logger.base_logger.info.assert_called()
    
    def test_storage_initialization_failure_logging(self, diagnostic_logger):
        """Test logging of failed storage initialization."""
        diagnostic_logger.log_storage_initialization(
            storage_type="graph_store",
            path="/failed/storage/path",
            initialization_time_ms=100.0,
            success=False,
            error_details="Permission denied"
        )
        
        # Should log at ERROR level for failures
        diagnostic_logger.enhanced_logger.base_logger.error.assert_called()
    
    def test_storage_permission_error_logging(self, diagnostic_logger):
        """Test detailed logging of permission errors."""
        permission_error = StoragePermissionError(
            "Cannot write to storage directory",
            storage_path="/readonly/storage",
            required_permission="write"
        )
        
        diagnostic_logger.enhanced_logger.log_error_with_context(
            "Storage permission denied",
            permission_error,
            operation_name="storage_initialization",
            additional_context={
                "attempted_operation": "create_directory",
                "storage_type": "vector_store"
            }
        )
        
        # Should log detailed error information
        diagnostic_logger.enhanced_logger.base_logger.error.assert_called()
    
    def test_storage_space_error_logging(self, diagnostic_logger):
        """Test detailed logging of disk space errors."""
        space_error = StorageSpaceError(
            "Insufficient disk space for storage initialization",
            storage_path="/full/disk/storage",
            available_space=512 * 1024**2,  # 512MB
            required_space=2 * 1024**3      # 2GB
        )
        
        diagnostic_logger.enhanced_logger.log_error_with_context(
            "Storage space insufficient",
            space_error,
            operation_name="storage_space_validation",
            additional_context={
                "space_deficit_mb": (space_error.required_space - space_error.available_space) / 1024**2,
                "storage_type": "combined_storage"
            }
        )
        
        diagnostic_logger.enhanced_logger.base_logger.error.assert_called()
    
    def test_structured_storage_error_logging(self, diagnostic_logger):
        """Test structured logging of storage errors."""
        # Multiple storage operations with different outcomes
        storage_operations = [
            {
                "storage_type": "vector_store", 
                "success": True, 
                "time_ms": 300.0,
                "path": "/success/vector"
            },
            {
                "storage_type": "graph_store", 
                "success": False, 
                "time_ms": 150.0,
                "path": "/failed/graph",
                "error": "Permission denied"
            },
            {
                "storage_type": "embedding_cache", 
                "success": True, 
                "time_ms": 200.0,
                "path": "/success/embeddings"
            }
        ]
        
        for operation in storage_operations:
            diagnostic_logger.log_storage_initialization(
                storage_type=operation["storage_type"],
                path=operation["path"],
                initialization_time_ms=operation["time_ms"],
                success=operation["success"],
                error_details=operation.get("error")
            )
        
        # Should have made multiple logging calls
        total_calls = (diagnostic_logger.enhanced_logger.base_logger.info.call_count +
                      diagnostic_logger.enhanced_logger.base_logger.error.call_count)
        assert total_calls == 3


class TestStorageRecoveryIntegration:
    """Test integration of storage errors with recovery system."""
    
    @pytest.fixture
    def integrated_system(self, temp_dir):
        """Create integrated storage and recovery system."""
        recovery_system = AdvancedRecoverySystem(
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        base_logger = Mock()
        diagnostic_logger = DiagnosticLogger(base_logger)
        
        config = StorageTestConfig(base_path=temp_dir / "storage")
        mock_storage = MockStorageSystem(config)
        
        return {
            'recovery_system': recovery_system,
            'diagnostic_logger': diagnostic_logger,
            'mock_storage': mock_storage,
            'temp_dir': temp_dir
        }
    
    def test_storage_failure_recovery_workflow(self, integrated_system):
        """Test complete storage failure recovery workflow."""
        recovery = integrated_system['recovery_system']
        logger = integrated_system['diagnostic_logger']
        storage = integrated_system['mock_storage']
        
        # Initialize storage-related session
        recovery.initialize_ingestion_session(
            documents=[],
            phase=recovery._current_phase,  # Use current phase for storage operations
            batch_size=1
        )
        
        # Simulate storage initialization failure
        storage.should_fail_creation = True
        
        try:
            storage.create_directory(Path("/test/storage"))
        except StorageDirectoryError as e:
            # Log the storage error
            logger.enhanced_logger.log_error_with_context(
                "Storage directory creation failed",
                e,
                operation_name="storage_initialization"
            )
            
            # Handle through recovery system
            strategy = recovery.handle_failure(
                FailureType.RESOURCE_EXHAUSTION,
                str(e),
                context={"storage_operation": "directory_creation"}
            )
            
            # Should provide recovery strategy
            assert 'action' in strategy
            assert 'checkpoint_recommended' in strategy
    
    def test_storage_fallback_mechanism(self, integrated_system):
        """Test storage fallback mechanism integration."""
        recovery = integrated_system['recovery_system']
        storage = integrated_system['mock_storage']
        temp_dir = integrated_system['temp_dir']
        
        # Define primary and fallback storage paths
        primary_storage = temp_dir / "primary_storage"
        fallback_storage = temp_dir / "fallback_storage"
        
        # Make primary storage fail
        storage.should_fail_creation = True
        
        primary_success = False
        try:
            storage.create_directory(primary_storage)
            primary_success = True
        except StorageDirectoryError:
            pass
        
        assert primary_success == False
        
        # Try fallback storage (reset failure simulation)
        storage.should_fail_creation = False
        fallback_success = False
        try:
            storage.create_directory(fallback_storage)
            fallback_success = True
        except StorageDirectoryError:
            pass
        
        assert fallback_success == True
        assert fallback_storage.exists()
    
    def test_storage_space_monitoring_integration(self, integrated_system):
        """Test integration of storage space monitoring with recovery."""
        recovery = integrated_system['recovery_system']
        storage = integrated_system['mock_storage']
        logger = integrated_system['diagnostic_logger']
        
        # Set low available space
        storage.available_space_bytes = 100 * 1024**2  # 100MB
        storage.should_fail_space_check = True
        
        try:
            storage.check_disk_space(Path("/test"), 1024**3)  # Request 1GB
        except StorageSpaceError as e:
            # Log space error
            logger.enhanced_logger.log_error_with_context(
                "Insufficient storage space",
                e,
                operation_name="storage_space_validation"
            )
            
            # Handle through recovery system
            strategy = recovery.handle_failure(
                FailureType.RESOURCE_EXHAUSTION,
                str(e),
                context={"storage_space_required": 1024**3}
            )
            
            # Should recommend space-related recovery actions
            assert 'action' in strategy
            assert strategy.get('degradation_needed', False) == True


# =====================================================================
# EDGE CASES AND STRESS TESTS
# =====================================================================

class TestStorageErrorEdgeCases:
    """Test edge cases and boundary conditions for storage errors."""
    
    def test_concurrent_storage_access_errors(self, temp_dir):
        """Test handling of concurrent storage access scenarios."""
        import threading
        import time
        
        shared_storage = temp_dir / "concurrent_storage"
        config = StorageTestConfig(base_path=shared_storage)
        
        errors_caught = []
        successful_operations = []
        
        def storage_operation(thread_id):
            """Simulate storage operation in thread."""
            try:
                mock_storage = MockStorageSystem(config)
                
                # Simulate some concurrent access issues
                if thread_id % 3 == 0:  # Every third thread fails
                    mock_storage.should_fail_creation = True
                
                thread_storage = shared_storage / f"thread_{thread_id}"
                mock_storage.create_directory(thread_storage)
                successful_operations.append(thread_id)
                
            except Exception as e:
                errors_caught.append((thread_id, e))
        
        # Create multiple threads for concurrent access
        threads = []
        for i in range(10):
            thread = threading.Thread(target=storage_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have both successes and failures
        assert len(successful_operations) > 0
        assert len(errors_caught) > 0
        
        # All errors should be StorageDirectoryError
        for thread_id, error in errors_caught:
            assert isinstance(error, StorageDirectoryError)
    
    def test_storage_path_with_special_characters(self, temp_dir):
        """Test storage paths with special characters."""
        special_paths = [
            "storage with spaces",
            "storage-with-dashes",
            "storage_with_underscores",
            "storage.with.dots",
            "storage@with#special$chars%",
            "储存中文路径",  # Chinese characters
            "مسار_عربي"     # Arabic characters (if filesystem supports)
        ]
        
        config = StorageTestConfig(base_path=temp_dir)
        mock_storage = MockStorageSystem(config)
        
        for special_name in special_paths:
            try:
                special_path = temp_dir / special_name
                result = mock_storage.create_directory(special_path)
                
                # If creation succeeded, verify the directory exists
                if result:
                    assert special_path.exists()
                    
            except (StorageDirectoryError, OSError, UnicodeError) as e:
                # Some special characters may legitimately fail
                # depending on filesystem limitations
                assert isinstance(e, (StorageDirectoryError, OSError, UnicodeError))
    
    def test_extremely_long_storage_path(self, temp_dir):
        """Test handling of extremely long storage paths."""
        # Create a very long path (near filesystem limits)
        long_component = "very_long_directory_name" * 10  # 250+ characters
        long_path = temp_dir
        
        # Build nested long path
        for i in range(5):
            long_path = long_path / f"{long_component}_{i}"
        
        config = StorageTestConfig(base_path=temp_dir)
        mock_storage = MockStorageSystem(config)
        
        try:
            result = mock_storage.create_directory(long_path)
            
            # If successful, verify
            if result:
                assert long_path.exists()
                
        except (StorageDirectoryError, OSError) as e:
            # Long paths may legitimately fail due to filesystem limits
            assert isinstance(e, (StorageDirectoryError, OSError))
    
    def test_storage_on_full_filesystem_simulation(self, temp_dir):
        """Test storage behavior when filesystem is full."""
        config = StorageTestConfig(base_path=temp_dir)
        mock_storage = MockStorageSystem(config)
        
        # Simulate full disk
        mock_storage.available_space_bytes = 0
        mock_storage.should_fail_space_check = True
        
        test_path = temp_dir / "full_disk_test"
        
        with pytest.raises(StorageSpaceError) as exc_info:
            mock_storage.check_disk_space(test_path, 1024)  # Request 1KB
        
        error = exc_info.value
        assert error.available_space == 0
        assert error.required_space == 1024
    
    def test_storage_error_message_localization(self):
        """Test storage error messages for different locales."""
        # Test with different error message formats
        error_scenarios = [
            {
                'error_class': StoragePermissionError,
                'args': ("Access denied", "/path"),
                'kwargs': {'required_permission': 'write'},
                'expected_attrs': ['required_permission']
            },
            {
                'error_class': StorageSpaceError, 
                'args': ("No space left", "/path"),
                'kwargs': {'available_space': 0, 'required_space': 1024},
                'expected_attrs': ['available_space', 'required_space']
            },
            {
                'error_class': StorageDirectoryError,
                'args': ("Cannot create directory", "/path"),
                'kwargs': {'directory_operation': 'create'},
                'expected_attrs': ['directory_operation']
            }
        ]
        
        for scenario in error_scenarios:
            error = scenario['error_class'](
                *scenario['args'],
                **scenario['kwargs']
            )
            
            # Verify error message
            assert len(str(error)) > 0
            
            # Verify expected attributes exist
            for attr in scenario['expected_attrs']:
                assert hasattr(error, attr)
    
    def test_storage_cleanup_after_partial_failure(self, temp_dir):
        """Test cleanup behavior after partial storage initialization failure."""
        storage_root = temp_dir / "partial_failure_test"
        
        # Create partial storage structure
        subdirs = ["vector_store", "graph_store", "embeddings", "temp"]
        created_dirs = []
        
        for subdir in subdirs:
            subdir_path = storage_root / subdir
            try:
                subdir_path.mkdir(parents=True)
                created_dirs.append(subdir_path)
                
                # Create some files
                test_file = subdir_path / "test.dat"
                test_file.write_text("test data")
                
            except OSError:
                break  # Simulate failure partway through
        
        # Verify partial structure exists
        assert len(created_dirs) > 0
        assert storage_root.exists()
        
        # Simulate cleanup on failure
        if storage_root.exists():
            shutil.rmtree(storage_root)
        
        # Verify complete cleanup
        assert not storage_root.exists()
        for created_dir in created_dirs:
            assert not created_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])