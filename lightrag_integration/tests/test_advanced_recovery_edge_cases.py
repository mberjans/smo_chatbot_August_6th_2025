#!/usr/bin/env python3
"""
Advanced Recovery System Edge Cases and Performance Tests.

This test suite focuses on edge cases, boundary conditions, and performance
characteristics of the advanced recovery system components including resource
monitoring, adaptive backoff, checkpoint management, and degradation strategies.

Test Focus Areas:
- Resource monitoring edge cases and accuracy
- Adaptive backoff algorithm behavior under extreme conditions
- Checkpoint system stress testing and corruption handling
- Degradation mode transitions and stability
- Performance under high load and memory pressure
- Thread safety and concurrent operations

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import pytest
import asyncio
import threading
import time
import json
import tempfile
import random
import gc
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import components for testing
sys.path.append(str(Path(__file__).parent.parent))

from lightrag_integration.advanced_recovery_system import (
    AdvancedRecoverySystem, DegradationMode, FailureType, BackoffStrategy,
    ResourceThresholds, DegradationConfig, CheckpointData, CheckpointManager,
    SystemResourceMonitor, AdaptiveBackoffCalculator
)
from lightrag_integration.unified_progress_tracker import (
    KnowledgeBaseProgressTracker, KnowledgeBasePhase
)


# =====================================================================
# RESOURCE MONITORING EDGE CASES
# =====================================================================

class TestSystemResourceMonitorEdgeCases:
    """Test edge cases for system resource monitoring."""
    
    @pytest.fixture
    def resource_monitor(self):
        """Create resource monitor with extreme thresholds."""
        thresholds = ResourceThresholds(
            memory_warning_percent=10.0,   # Very low threshold
            memory_critical_percent=20.0,  # Still low
            disk_warning_percent=95.0,     # Very high threshold
            disk_critical_percent=99.0,    # Extremely high
            cpu_warning_percent=50.0,
            cpu_critical_percent=75.0
        )
        return SystemResourceMonitor(thresholds)
    
    @patch('psutil.virtual_memory')
    def test_memory_monitoring_with_invalid_data(self, mock_memory, resource_monitor):
        """Test memory monitoring handles invalid/missing data gracefully."""
        # Test with None return
        mock_memory.return_value = None
        
        resources = resource_monitor.get_current_resources()
        # Should handle gracefully and return empty dict or valid defaults
        assert isinstance(resources, dict)
        
        # Test with invalid memory object
        invalid_memory = Mock()
        invalid_memory.percent = -1  # Invalid percentage
        invalid_memory.available = -1024  # Invalid available memory
        mock_memory.return_value = invalid_memory
        
        resources = resource_monitor.get_current_resources()
        # Should still return a dict, possibly with no memory data
        assert isinstance(resources, dict)
    
    @patch('psutil.disk_usage')
    def test_disk_monitoring_with_inaccessible_paths(self, mock_disk_usage, resource_monitor):
        """Test disk monitoring with inaccessible paths."""
        # Simulate permission denied for disk usage
        mock_disk_usage.side_effect = PermissionError("Access denied")
        
        resources = resource_monitor.get_current_resources()
        
        # Should handle gracefully without disk information
        assert isinstance(resources, dict)
        # May or may not contain disk info, but should not crash
    
    @patch('psutil.cpu_percent')
    def test_cpu_monitoring_with_extreme_values(self, mock_cpu, resource_monitor):
        """Test CPU monitoring with extreme percentage values."""
        extreme_values = [-10.0, 0.0, 100.0, 150.0, 999.0]
        
        for cpu_value in extreme_values:
            mock_cpu.return_value = cpu_value
            
            resources = resource_monitor.get_current_resources()
            
            # Should handle all values gracefully
            assert isinstance(resources, dict)
            if 'cpu_percent' in resources:
                # Value should be stored as provided, validation happens elsewhere
                assert resources['cpu_percent'] == cpu_value
    
    def test_rapid_resource_checks(self, resource_monitor):
        """Test resource monitoring under rapid successive calls."""
        start_time = time.time()
        results = []
        
        # Make many rapid calls
        for _ in range(100):
            try:
                resources = resource_monitor.get_current_resources()
                results.append(resources)
            except Exception as e:
                # Should not fail under rapid calls
                pytest.fail(f"Resource monitoring failed under rapid calls: {e}")
        
        elapsed = time.time() - start_time
        
        # Should complete quickly (under 5 seconds even on slow systems)
        assert elapsed < 5.0
        assert len(results) == 100
        
        # All results should be valid dictionaries
        for result in results:
            assert isinstance(result, dict)
    
    def test_concurrent_resource_monitoring(self, resource_monitor):
        """Test concurrent resource monitoring from multiple threads."""
        results = {}
        errors = {}
        
        def monitor_resources(thread_id):
            """Monitor resources in a thread."""
            try:
                for i in range(10):
                    resources = resource_monitor.get_current_resources()
                    if thread_id not in results:
                        results[thread_id] = []
                    results[thread_id].append(resources)
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors[thread_id] = e
        
        # Start multiple monitoring threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=monitor_resources, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Should have results from all threads
        assert len(results) == 5
        
        for thread_id, thread_results in results.items():
            assert len(thread_results) == 10
            for result in thread_results:
                assert isinstance(result, dict)
    
    @patch('psutil.getloadavg')
    def test_load_average_on_unsupported_systems(self, mock_loadavg, resource_monitor):
        """Test load average monitoring on systems that don't support it."""
        # Simulate AttributeError (as would happen on Windows)
        mock_loadavg.side_effect = AttributeError("No getloadavg on this platform")
        
        resources = resource_monitor.get_current_resources()
        
        # Should handle gracefully
        assert isinstance(resources, dict)
        # load_average should be 0.0 or not present
        if 'load_average' in resources:
            assert resources['load_average'] == 0.0
    
    def test_resource_threshold_edge_values(self):
        """Test resource monitoring with edge threshold values."""
        # Test with extreme threshold values
        extreme_thresholds = ResourceThresholds(
            memory_warning_percent=0.0,
            memory_critical_percent=0.1,
            disk_warning_percent=99.9,
            disk_critical_percent=100.0,
            cpu_warning_percent=100.0,
            cpu_critical_percent=100.0
        )
        
        monitor = SystemResourceMonitor(extreme_thresholds)
        
        # Should create monitor without issues
        assert monitor.thresholds.memory_warning_percent == 0.0
        assert monitor.thresholds.disk_critical_percent == 100.0
        
        # Should be able to check pressure
        pressure = monitor.check_resource_pressure()
        assert isinstance(pressure, dict)


# =====================================================================
# ADAPTIVE BACKOFF EDGE CASES
# =====================================================================

class TestAdaptiveBackoffEdgeCases:
    """Test edge cases for adaptive backoff calculations."""
    
    @pytest.fixture
    def backoff_calculator(self):
        """Create backoff calculator for edge case testing."""
        return AdaptiveBackoffCalculator()
    
    def test_backoff_with_zero_attempt(self, backoff_calculator):
        """Test backoff calculation with zero or negative attempt numbers."""
        # Test with attempt = 0
        delay = backoff_calculator.calculate_backoff(
            FailureType.API_ERROR,
            0,  # Zero attempt
            BackoffStrategy.EXPONENTIAL
        )
        
        # Should handle gracefully and return reasonable delay
        assert delay > 0
        assert delay < 300.0  # Should be within max delay
        
        # Test with negative attempt
        delay = backoff_calculator.calculate_backoff(
            FailureType.API_ERROR,
            -1,  # Negative attempt
            BackoffStrategy.EXPONENTIAL
        )
        
        # Should handle gracefully
        assert delay > 0
    
    def test_extreme_attempt_numbers(self, backoff_calculator):
        """Test backoff with extremely high attempt numbers."""
        extreme_attempts = [100, 1000, 10000]
        
        for attempt in extreme_attempts:
            delay = backoff_calculator.calculate_backoff(
                FailureType.API_ERROR,
                attempt,
                BackoffStrategy.EXPONENTIAL,
                base_delay=1.0,
                max_delay=300.0
            )
            
            # Should respect max_delay even with extreme attempts
            assert delay <= 300.0
            assert delay > 0
    
    def test_fibonacci_backoff_large_numbers(self, backoff_calculator):
        """Test Fibonacci backoff with large attempt numbers."""
        large_attempts = [50, 75, 100]
        
        for attempt in large_attempts:
            delay = backoff_calculator.calculate_backoff(
                FailureType.API_ERROR,
                attempt,
                BackoffStrategy.FIBONACCI,
                base_delay=1.0,
                max_delay=600.0
            )
            
            # Fibonacci can grow very large, ensure max_delay is respected
            assert delay <= 600.0
            assert delay > 0
    
    def test_adaptive_backoff_with_extreme_failure_history(self, backoff_calculator):
        """Test adaptive backoff with extreme failure patterns."""
        # Create extreme failure history
        for _ in range(1000):  # Many failures
            backoff_calculator.calculate_backoff(
                FailureType.API_RATE_LIMIT,
                1,
                BackoffStrategy.ADAPTIVE
            )
        
        # Test backoff after extreme failure history
        delay = backoff_calculator.calculate_backoff(
            FailureType.API_RATE_LIMIT,
            5,
            BackoffStrategy.ADAPTIVE,
            max_delay=300.0
        )
        
        # Should still respect max delay despite extreme failure history
        assert delay <= 300.0
        assert delay > 0
        
        # Should be significantly affected by failure history
        # (compare with fresh calculator)
        fresh_calculator = AdaptiveBackoffCalculator()
        fresh_delay = fresh_calculator.calculate_backoff(
            FailureType.API_RATE_LIMIT,
            5,
            BackoffStrategy.ADAPTIVE,
            jitter=False
        )
        
        assert delay >= fresh_delay  # Should be equal or higher
    
    def test_success_history_overflow_handling(self, backoff_calculator):
        """Test handling of extreme success history."""
        # Record many successes
        for _ in range(10000):
            backoff_calculator.record_success()
        
        # Should handle large success history gracefully
        delay = backoff_calculator.calculate_backoff(
            FailureType.API_ERROR,
            3,
            BackoffStrategy.ADAPTIVE
        )
        
        assert delay > 0
        assert delay < 1000.0  # Should be reasonable
    
    def test_concurrent_backoff_calculations(self, backoff_calculator):
        """Test concurrent backoff calculations thread safety."""
        results = {}
        errors = {}
        
        def calculate_backoff_concurrent(thread_id):
            """Calculate backoffs in a thread."""
            try:
                thread_results = []
                for i in range(50):
                    delay = backoff_calculator.calculate_backoff(
                        FailureType.API_ERROR,
                        i % 10 + 1,
                        BackoffStrategy.ADAPTIVE
                    )
                    thread_results.append(delay)
                    
                    # Occasionally record success
                    if i % 5 == 0:
                        backoff_calculator.record_success()
                
                results[thread_id] = thread_results
            except Exception as e:
                errors[thread_id] = e
        
        # Run concurrent calculations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=calculate_backoff_concurrent, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Should have results from all threads
        assert len(results) == 10
        
        for thread_id, thread_results in results.items():
            assert len(thread_results) == 50
            for delay in thread_results:
                assert delay > 0
    
    def test_api_response_time_extreme_values(self, backoff_calculator):
        """Test adaptive backoff with extreme API response times."""
        extreme_response_times = [0.001, 0.0, -1.0, 100.0, 1000.0, float('inf')]
        
        for response_time in extreme_response_times:
            try:
                backoff_calculator.update_api_response_time(response_time)
                
                delay = backoff_calculator.calculate_backoff(
                    FailureType.API_ERROR,
                    1,
                    BackoffStrategy.ADAPTIVE
                )
                
                # Should handle extreme values gracefully
                assert delay > 0
                assert delay < 1000.0  # Should be reasonable
                
            except (ValueError, OverflowError):
                # Some extreme values may legitimately cause errors
                pass
    
    def test_backoff_jitter_consistency(self, backoff_calculator):
        """Test jitter behavior consistency."""
        # Test with jitter enabled
        delays_with_jitter = []
        for _ in range(100):
            delay = backoff_calculator.calculate_backoff(
                FailureType.API_ERROR,
                5,
                BackoffStrategy.EXPONENTIAL,
                base_delay=10.0,
                jitter=True
            )
            delays_with_jitter.append(delay)
        
        # Test without jitter
        delays_without_jitter = []
        for _ in range(100):
            delay = backoff_calculator.calculate_backoff(
                FailureType.API_ERROR,
                5,
                BackoffStrategy.EXPONENTIAL,
                base_delay=10.0,
                jitter=False
            )
            delays_without_jitter.append(delay)
        
        # With jitter should have variation
        jitter_variance = max(delays_with_jitter) - min(delays_with_jitter)
        assert jitter_variance > 0
        
        # Without jitter should be consistent
        no_jitter_variance = max(delays_without_jitter) - min(delays_without_jitter)
        assert no_jitter_variance == 0


# =====================================================================
# CHECKPOINT MANAGER STRESS TESTS
# =====================================================================

class TestCheckpointManagerStress:
    """Stress tests for checkpoint management system."""
    
    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager for stress testing."""
        return CheckpointManager(temp_dir / "stress_checkpoints")
    
    def test_rapid_checkpoint_creation(self, checkpoint_manager):
        """Test rapid creation of many checkpoints."""
        checkpoint_ids = []
        start_time = time.time()
        
        # Create many checkpoints rapidly
        for i in range(100):
            checkpoint_id = checkpoint_manager.create_checkpoint(
                phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
                processed_documents=[f"doc_{j}" for j in range(i)],
                failed_documents={f"failed_{j}": f"error_{j}" for j in range(i % 5)},
                pending_documents=[f"pending_{j}" for j in range(i, i + 10)],
                current_batch_size=5,
                degradation_mode=DegradationMode.OPTIMAL,
                system_resources={"memory_percent": random.uniform(30, 80)},
                error_counts={FailureType.API_ERROR: i % 10},
                metadata={"iteration": i}
            )
            checkpoint_ids.append(checkpoint_id)
        
        creation_time = time.time() - start_time
        
        # Should complete in reasonable time (under 10 seconds)
        assert creation_time < 10.0
        
        # All checkpoints should be unique
        assert len(set(checkpoint_ids)) == 100
        
        # All checkpoint files should exist
        for checkpoint_id in checkpoint_ids:
            checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.json"
            assert checkpoint_file.exists()
    
    def test_checkpoint_data_integrity_under_stress(self, checkpoint_manager):
        """Test checkpoint data integrity under stress conditions."""
        # Create checkpoints with large data sets
        large_processed = [f"large_doc_{i}" for i in range(1000)]
        large_failed = {f"failed_{i}": f"error_message_{i}" * 100 for i in range(100)}
        large_pending = [f"pending_{i}" for i in range(500)]
        complex_metadata = {
            "nested_data": {
                "level1": {
                    "level2": {
                        "data": [i for i in range(1000)]
                    }
                }
            },
            "large_string": "x" * 10000
        }
        
        checkpoint_id = checkpoint_manager.create_checkpoint(
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            processed_documents=large_processed,
            failed_documents=large_failed,
            pending_documents=large_pending,
            current_batch_size=10,
            degradation_mode=DegradationMode.MINIMAL,
            system_resources={"memory_percent": 75.0, "cpu_percent": 50.0},
            error_counts={
                FailureType.API_ERROR: 50,
                FailureType.NETWORK_ERROR: 25,
                FailureType.MEMORY_PRESSURE: 10
            },
            metadata=complex_metadata
        )
        
        # Load and verify data integrity
        loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_id)
        
        assert loaded_checkpoint is not None
        assert len(loaded_checkpoint.processed_documents) == 1000
        assert len(loaded_checkpoint.failed_documents) == 100
        assert len(loaded_checkpoint.pending_documents) == 500
        assert loaded_checkpoint.degradation_mode == DegradationMode.MINIMAL
        assert loaded_checkpoint.metadata["large_string"] == "x" * 10000
        
        # Verify nested data integrity
        assert loaded_checkpoint.metadata["nested_data"]["level1"]["level2"]["data"][-1] == 999
    
    def test_concurrent_checkpoint_operations(self, checkpoint_manager):
        """Test concurrent checkpoint creation and loading."""
        checkpoint_ids = []
        errors = []
        
        def checkpoint_worker(worker_id):
            """Worker function for concurrent checkpoint operations."""
            try:
                # Create checkpoint
                checkpoint_id = checkpoint_manager.create_checkpoint(
                    phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
                    processed_documents=[f"worker_{worker_id}_doc_{i}" for i in range(10)],
                    failed_documents={},
                    pending_documents=[f"worker_{worker_id}_pending_{i}" for i in range(5)],
                    current_batch_size=5,
                    degradation_mode=DegradationMode.OPTIMAL,
                    system_resources={"worker_id": worker_id},
                    error_counts={FailureType.API_ERROR: worker_id},
                    metadata={"worker": worker_id}
                )
                
                checkpoint_ids.append(checkpoint_id)
                
                # Immediately try to load it
                loaded = checkpoint_manager.load_checkpoint(checkpoint_id)
                if loaded is None:
                    errors.append(f"Worker {worker_id}: Failed to load checkpoint")
                elif loaded.metadata["worker"] != worker_id:
                    errors.append(f"Worker {worker_id}: Data corruption detected")
                    
            except Exception as e:
                errors.append(f"Worker {worker_id}: Exception - {e}")
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(checkpoint_worker, i) for i in range(20)]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future exception: {e}")
        
        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Should have created all checkpoints
        assert len(checkpoint_ids) == 20
        assert len(set(checkpoint_ids)) == 20  # All unique
    
    def test_checkpoint_corruption_recovery(self, checkpoint_manager):
        """Test recovery from checkpoint corruption scenarios."""
        # Create valid checkpoint
        valid_id = checkpoint_manager.create_checkpoint(
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            processed_documents=["doc1"],
            failed_documents={},
            pending_documents=["doc2"],
            current_batch_size=1,
            degradation_mode=DegradationMode.OPTIMAL,
            system_resources={},
            error_counts={}
        )
        
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{valid_id}.json"
        assert checkpoint_file.exists()
        
        # Create various corruption scenarios
        corruption_scenarios = [
            "invalid json content",
            '{"incomplete": json}',
            '{"valid_json": "but_missing_required_fields"}',
            "",  # Empty file
            "null",
            '{"checkpoint_id": null}'
        ]
        
        for i, corrupted_content in enumerate(corruption_scenarios):
            corrupted_id = f"corrupted_{i}"
            corrupted_file = checkpoint_manager.checkpoint_dir / f"{corrupted_id}.json"
            corrupted_file.write_text(corrupted_content)
            
            # Should handle corruption gracefully
            loaded = checkpoint_manager.load_checkpoint(corrupted_id)
            assert loaded is None  # Should return None for corrupted checkpoints
        
        # Original valid checkpoint should still work
        valid_loaded = checkpoint_manager.load_checkpoint(valid_id)
        assert valid_loaded is not None
        assert valid_loaded.checkpoint_id == valid_id
    
    def test_checkpoint_cleanup_performance(self, checkpoint_manager):
        """Test performance of checkpoint cleanup operations."""
        # Create many old checkpoints
        checkpoint_ids = []
        for i in range(200):
            checkpoint_id = checkpoint_manager.create_checkpoint(
                phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
                processed_documents=[f"old_doc_{i}"],
                failed_documents={},
                pending_documents=[],
                current_batch_size=1,
                degradation_mode=DegradationMode.OPTIMAL,
                system_resources={},
                error_counts={}
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Modify timestamps to make them old (simulate 2 days ago)
        old_time = time.time() - (48 * 3600)
        for checkpoint_id in checkpoint_ids[:150]:  # Make 150 of them old
            checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.json"
            checkpoint_file.touch()
            import os
            os.utime(checkpoint_file, (old_time, old_time))
        
        # Test cleanup performance
        start_time = time.time()
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(max_age_hours=24)
        cleanup_time = time.time() - start_time
        
        # Should complete quickly even with many files
        assert cleanup_time < 5.0
        
        # Should have deleted the old checkpoints
        assert deleted_count == 150
        
        # Recent checkpoints should remain
        remaining_checkpoints = checkpoint_manager.list_checkpoints()
        assert len(remaining_checkpoints) == 50


# =====================================================================
# DEGRADATION MODE STRESS TESTS
# =====================================================================

class TestDegradationModeStress:
    """Stress tests for degradation mode transitions and stability."""
    
    @pytest.fixture
    def recovery_system(self, temp_dir):
        """Create recovery system for degradation testing."""
        return AdvancedRecoverySystem(checkpoint_dir=temp_dir / "degradation_checkpoints")
    
    def test_rapid_degradation_mode_switching(self, recovery_system):
        """Test rapid switching between degradation modes."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1", "doc2", "doc3"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=10
        )
        
        initial_mode = recovery_system.current_degradation_mode
        mode_history = [initial_mode]
        
        # Generate rapid failures of different types
        failure_types = list(FailureType)
        
        for i in range(100):
            failure_type = random.choice(failure_types)
            
            strategy = recovery_system.handle_failure(
                failure_type,
                f"Stress test failure {i}",
                f"doc_{i % 3}"
            )
            
            mode_history.append(recovery_system.current_degradation_mode)
        
        # Should have gone through multiple modes
        unique_modes = set(mode_history)
        assert len(unique_modes) > 1
        
        # Final mode should be more restrictive than initial
        mode_hierarchy = {
            DegradationMode.OPTIMAL: 0,
            DegradationMode.ESSENTIAL: 1,
            DegradationMode.MINIMAL: 2,
            DegradationMode.OFFLINE: 3,
            DegradationMode.SAFE: 4
        }
        
        final_mode = recovery_system.current_degradation_mode
        assert mode_hierarchy[final_mode] >= mode_hierarchy[initial_mode]
    
    def test_degradation_mode_stability(self, recovery_system):
        """Test stability of degradation modes under consistent conditions."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1", "doc2", "doc3"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        # Force to specific degradation mode
        recovery_system.current_degradation_mode = DegradationMode.MINIMAL
        recovery_system._update_degradation_config()
        
        initial_batch_size = recovery_system._current_batch_size
        
        # Generate consistent successful operations
        for i in range(50):
            recovery_system.mark_document_processed(f"success_doc_{i}")
            recovery_system.backoff_calculator.record_success()
        
        # Mode should remain stable with successful operations
        assert recovery_system.current_degradation_mode == DegradationMode.MINIMAL
        
        # Batch size should be appropriate for minimal mode
        current_batch_size = recovery_system._current_batch_size
        assert current_batch_size <= 3  # Minimal mode constraint
    
    def test_degradation_config_consistency(self, recovery_system):
        """Test consistency of degradation configuration across mode changes."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        # Test each degradation mode
        modes_to_test = [
            DegradationMode.ESSENTIAL,
            DegradationMode.MINIMAL,
            DegradationMode.OFFLINE,
            DegradationMode.SAFE
        ]
        
        for mode in modes_to_test:
            recovery_system.current_degradation_mode = mode
            recovery_system._update_degradation_config()
            
            # Verify configuration is appropriate for mode
            config = recovery_system.degradation_config
            
            if mode == DegradationMode.ESSENTIAL:
                assert config.skip_optional_metadata == True
                assert config.reduce_batch_size == True
                assert config.max_retry_attempts == 2
            
            elif mode == DegradationMode.MINIMAL:
                assert config.skip_optional_metadata == True
                assert config.disable_advanced_chunking == True
                assert config.reduce_batch_size == True
                assert recovery_system._current_batch_size <= 3
            
            elif mode == DegradationMode.OFFLINE:
                assert config.max_retry_attempts == 1
                assert recovery_system._current_batch_size == 1
            
            elif mode == DegradationMode.SAFE:
                assert config.skip_optional_metadata == True
                assert config.disable_advanced_chunking == True
                assert config.max_retry_attempts == 5
                assert config.backoff_multiplier == 3.0
                assert recovery_system._current_batch_size == 1
    
    def test_batch_size_calculations_under_pressure(self, recovery_system):
        """Test batch size calculations under various resource pressures."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1", "doc2", "doc3", "doc4", "doc5"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=10
        )
        
        # Test different resource pressure scenarios
        pressure_scenarios = [
            {'memory': 'critical_reduce_batch_size'},
            {'cpu': 'critical_reduce_concurrency'},
            {'memory': 'warning_monitor_usage', 'cpu': 'warning_consider_throttling'},
            {'memory': 'critical_reduce_batch_size', 'cpu': 'critical_reduce_concurrency'},
            {}  # No pressure
        ]
        
        for pressure in pressure_scenarios:
            with patch.object(recovery_system.resource_monitor, 'check_resource_pressure') as mock_pressure:
                mock_pressure.return_value = pressure
                
                batch_size = recovery_system._calculate_optimal_batch_size()
                
                # Should return reasonable batch size
                assert batch_size >= 1
                assert batch_size <= 10
                
                # Critical memory pressure should reduce batch size significantly
                if 'memory' in pressure and 'critical' in pressure['memory']:
                    assert batch_size <= 3
    
    def test_document_prioritization_consistency(self, recovery_system):
        """Test document prioritization consistency across degradation modes."""
        documents = [
            "essential_doc_1", "critical_analysis_2", "important_study_3",
            "regular_doc_4", "standard_paper_5", "normal_study_6"
        ]
        
        recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        # Test prioritization in ESSENTIAL mode
        recovery_system.current_degradation_mode = DegradationMode.ESSENTIAL
        recovery_system.degradation_config.enable_document_priority = True
        
        prioritized_docs = recovery_system._prioritize_documents(documents.copy())
        
        # Should prioritize documents with essential/critical keywords
        essential_docs = [doc for doc in prioritized_docs if 
                         any(keyword in doc.lower() for keyword in ['essential', 'critical', 'important'])]
        
        other_docs = [doc for doc in prioritized_docs if 
                     not any(keyword in doc.lower() for keyword in ['essential', 'critical', 'important'])]
        
        # Essential docs should come first
        for i, doc in enumerate(prioritized_docs):
            if doc in essential_docs:
                essential_index = i
                break
        else:
            essential_index = len(prioritized_docs)
        
        for i, doc in enumerate(prioritized_docs):
            if doc in other_docs:
                other_index = i
                break
        else:
            other_index = len(prioritized_docs)
        
        # At least some essential docs should come before other docs
        if essential_docs and other_docs:
            assert essential_index <= other_index


# =====================================================================
# MEMORY AND PERFORMANCE STRESS TESTS
# =====================================================================

@pytest.mark.performance
class TestAdvancedRecoveryPerformance:
    """Performance and memory stress tests for advanced recovery system."""
    
    def test_memory_usage_with_large_document_sets(self, temp_dir):
        """Test memory usage with very large document sets."""
        recovery_system = AdvancedRecoverySystem(
            checkpoint_dir=temp_dir / "perf_checkpoints"
        )
        
        # Create very large document set
        large_doc_set = [f"performance_doc_{i}" for i in range(10000)]
        
        # Monitor memory before
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Initialize session with large document set
        recovery_system.initialize_ingestion_session(
            documents=large_doc_set,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=100
        )
        
        # Simulate processing with failures
        for i in range(1000):
            if i % 2 == 0:
                recovery_system.mark_document_processed(f"performance_doc_{i}")
            else:
                recovery_system.handle_failure(
                    FailureType.API_ERROR,
                    f"Performance test error {i}",
                    f"performance_doc_{i}"
                )
        
        # Monitor memory after
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 200MB)
        assert memory_increase < 200 * 1024 * 1024
        
        # System should still be responsive
        status = recovery_system.get_recovery_status()
        assert isinstance(status, dict)
        assert 'document_progress' in status
    
    def test_checkpoint_creation_performance_at_scale(self, temp_dir):
        """Test checkpoint creation performance with large datasets."""
        recovery_system = AdvancedRecoverySystem(
            checkpoint_dir=temp_dir / "scale_checkpoints"
        )
        
        # Create large datasets for checkpoint
        large_processed = [f"processed_{i}" for i in range(5000)]
        large_failed = {f"failed_{i}": f"error_message_{i}" for i in range(1000)}
        large_pending = [f"pending_{i}" for i in range(2000)]
        
        recovery_system.initialize_ingestion_session(
            documents=large_processed + list(large_failed.keys()) + large_pending,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        # Set internal state
        recovery_system._processed_documents = set(large_processed)
        recovery_system._failed_documents = large_failed
        recovery_system._pending_documents = large_pending
        
        # Time checkpoint creation
        start_time = time.time()
        
        checkpoint_id = recovery_system.create_checkpoint({
            "performance_test": True,
            "large_data": list(range(1000))
        })
        
        creation_time = time.time() - start_time
        
        # Should complete in reasonable time (under 5 seconds)
        assert creation_time < 5.0
        assert checkpoint_id is not None
        
        # Verify checkpoint can be loaded
        loaded = recovery_system.resume_from_checkpoint(checkpoint_id)
        assert loaded == True
        
        # Verify data integrity
        assert len(recovery_system._processed_documents) == 5000
        assert len(recovery_system._failed_documents) == 1000
        assert len(recovery_system._pending_documents) == 2000
    
    def test_concurrent_recovery_operations_performance(self, temp_dir):
        """Test performance of concurrent recovery operations."""
        recovery_systems = []
        
        # Create multiple recovery systems
        for i in range(5):
            recovery = AdvancedRecoverySystem(
                checkpoint_dir=temp_dir / f"concurrent_{i}"
            )
            recovery_systems.append(recovery)
        
        def recovery_worker(system_id):
            """Worker function for concurrent recovery operations."""
            recovery = recovery_systems[system_id]
            
            # Initialize with documents
            docs = [f"sys_{system_id}_doc_{i}" for i in range(100)]
            recovery.initialize_ingestion_session(
                documents=docs,
                phase=KnowledgeBasePhase.DOCUMENT_INGESTION
            )
            
            # Perform operations
            for i in range(50):
                if i % 3 == 0:
                    recovery.mark_document_processed(f"sys_{system_id}_doc_{i}")
                else:
                    recovery.handle_failure(
                        FailureType.API_ERROR,
                        f"Concurrent error {i}",
                        f"sys_{system_id}_doc_{i}"
                    )
                
                if i % 10 == 0:
                    recovery.create_checkpoint({"concurrent_test": system_id, "iteration": i})
        
        # Run concurrent workers
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(recovery_worker, i) for i in range(5)]
            
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        total_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert total_time < 30.0
        
        # All systems should be in valid state
        for recovery in recovery_systems:
            status = recovery.get_recovery_status()
            assert isinstance(status, dict)
            assert status['document_progress']['total'] > 0
    
    def test_resource_monitoring_performance_impact(self, temp_dir):
        """Test performance impact of resource monitoring."""
        # Create recovery system with resource monitoring
        recovery_with_monitoring = AdvancedRecoverySystem(
            checkpoint_dir=temp_dir / "with_monitoring"
        )
        
        recovery_with_monitoring.initialize_ingestion_session(
            documents=[f"doc_{i}" for i in range(1000)],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        # Test with resource monitoring
        start_time = time.time()
        
        for i in range(100):
            recovery_with_monitoring.handle_failure(
                FailureType.MEMORY_PRESSURE,
                f"Memory error {i}",
                f"doc_{i}"
            )
            
            # This will trigger resource monitoring
            recovery_with_monitoring.get_recovery_status()
        
        monitoring_time = time.time() - start_time
        
        # Resource monitoring shouldn't add significant overhead
        assert monitoring_time < 10.0
        
        # Should still provide accurate status
        final_status = recovery_with_monitoring.get_recovery_status()
        assert 'system_resources' in final_status
        assert 'resource_pressure' in final_status
    
    def test_garbage_collection_behavior(self, temp_dir):
        """Test garbage collection behavior with recovery system."""
        recovery_system = AdvancedRecoverySystem(
            checkpoint_dir=temp_dir / "gc_test"
        )
        
        # Monitor garbage collection
        import gc
        gc.collect()  # Clean start
        
        initial_objects = len(gc.get_objects())
        
        # Create and destroy many sessions
        for session in range(10):
            docs = [f"gc_session_{session}_doc_{i}" for i in range(1000)]
            recovery_system.initialize_ingestion_session(
                documents=docs,
                phase=KnowledgeBasePhase.DOCUMENT_INGESTION
            )
            
            # Process some documents
            for i in range(100):
                if i % 2 == 0:
                    recovery_system.mark_document_processed(f"gc_session_{session}_doc_{i}")
                else:
                    recovery_system.handle_failure(
                        FailureType.API_ERROR,
                        f"GC test error {i}",
                        f"gc_session_{session}_doc_{i}"
                    )
            
            # Create checkpoint
            checkpoint_id = recovery_system.create_checkpoint({"session": session})
            
            # Force cleanup
            if session % 3 == 0:
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count shouldn't grow excessively
        object_growth = final_objects - initial_objects
        
        # Allow some growth but not excessive (less than 50% increase)
        growth_ratio = object_growth / initial_objects
        assert growth_ratio < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])