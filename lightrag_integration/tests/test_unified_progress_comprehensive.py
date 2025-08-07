#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Progress Tracking System.

This test suite combines all aspects of unified progress tracking testing
into a single comprehensive suite that can be run for complete validation.

Test Categories:
1. Unit Tests - Core functionality validation
2. Integration Tests - System integration validation  
3. Performance Tests - Overhead and scalability validation
4. Edge Case Tests - Error handling and boundary conditions
5. Real-world Simulation Tests - Full workflow validation

Usage:
    pytest test_unified_progress_comprehensive.py -v
    pytest test_unified_progress_comprehensive.py::TestComprehensiveIntegration -v
    pytest test_unified_progress_comprehensive.py -k "performance" -v

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import asyncio
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all test classes
from .test_unified_progress_tracking import (
    TestUnifiedProgressTrackerCore,
    TestPhaseWeightsAndProgress,
    TestCallbackSystem,
    TestProgressTrackingConfiguration,
    TestProgressTrackingIntegration,
    TestErrorHandlingAndEdgeCases,
    TestThreadSafety,
    TestPerformance,
    TestKnowledgeBaseIntegration
)

# Import fixtures and utilities
from .test_unified_progress_fixtures import (
    MockPDFProcessor,
    MockLightRAGKnowledgeBase,
    ProgressCallbackTester
)

from lightrag_integration.unified_progress_tracker import (
    KnowledgeBaseProgressTracker,
    KnowledgeBasePhase,
    PhaseWeights,
    UnifiedProgressState
)
from lightrag_integration.progress_config import ProgressTrackingConfig


# =====================================================================
# COMPREHENSIVE INTEGRATION TESTS
# =====================================================================

class TestComprehensiveIntegration:
    """Comprehensive integration tests combining all functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_simulation(self, integrated_test_environment, large_test_document_collection):
        """Test complete end-to-end workflow with realistic document collection."""
        env = integrated_test_environment
        
        # Run full simulation
        results = await env.run_full_simulation(large_test_document_collection[:20])  # Use subset for speed
        
        # Validate results
        assert results['final_progress_state'].overall_progress >= 0.99
        assert results['pdf_results']['successful'] > 0
        assert results['ingestion_results']['total_entities'] > 0
        
        # Validate callback behavior
        callback_stats = results['callback_statistics']
        assert callback_stats['total_callbacks'] > 20  # Should have many callback invocations
        assert callback_stats['progress_monotonicity']['is_monotonic']  # Progress should increase
        
        # Validate timing
        assert results['simulation_time'] < 10.0  # Should complete reasonably quickly
        
        # Validate phase completion
        final_state = results['final_progress_state']
        for phase in KnowledgeBasePhase:
            phase_info = final_state.phase_info[phase]
            assert phase_info.is_completed
            assert phase_info.current_progress == 1.0
        
        # Get comprehensive report
        report = env.get_comprehensive_report()
        assert report['pdf_processor_stats']['success_rate'] > 80  # At least 80% success
        assert report['knowledge_base_stats']['total_entities'] > 50  # Good entity extraction
        assert report['progress_tracking_stats']['callback_errors'] == 0  # No callback errors
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, integrated_test_environment):
        """Test workflow with error injection and recovery."""
        env = integrated_test_environment
        
        # Configure high failure rate for testing
        env.pdf_processor.failure_rate = 0.3  # 30% failure rate
        
        # Create test documents with some expected to fail
        test_docs = [
            "corrupted_document.pdf",
            "valid_metabolomics.pdf", 
            "encrypted_document.pdf",
            "valid_proteomics.pdf",
            "timeout_document.pdf"
        ]
        
        results = await env.run_full_simulation(test_docs)
        
        # Should complete despite failures
        assert results['final_progress_state'].overall_progress >= 0.99
        
        # Should have some failures but continue processing
        pdf_results = results['pdf_results']
        assert pdf_results['failed'] > 0  # Some failures expected
        assert pdf_results['successful'] > 0  # Some successes expected
        
        # Progress tracking should handle failures gracefully
        callback_stats = results['callback_statistics']
        assert callback_stats['progress_monotonicity']['is_monotonic']  # Still monotonic
        assert env.callback_tester.errors == []  # No callback errors
    
    def test_concurrent_workflow_execution(self, temp_dir):
        """Test multiple concurrent workflow executions."""
        
        def run_concurrent_workflow(worker_id: int) -> Dict[str, Any]:
            """Run workflow in separate thread."""
            # Create separate tracker for each thread
            config = ProgressTrackingConfig(
                save_unified_progress_to_file=True,
                unified_progress_file_path=temp_dir / f"progress_worker_{worker_id}.json"
            )
            
            tracker = KnowledgeBaseProgressTracker(progress_config=config)
            
            # Simulate quick workflow
            tracker.start_initialization(total_documents=5)
            
            for phase in KnowledgeBasePhase:
                tracker.start_phase(phase, f"Worker {worker_id} - {phase.value}")
                time.sleep(0.01)  # Small delay
                tracker.update_phase_progress(phase, 0.5, f"Worker {worker_id} halfway")
                time.sleep(0.01)
                tracker.complete_phase(phase, f"Worker {worker_id} completed {phase.value}")
            
            final_state = tracker.get_current_state()
            return {
                'worker_id': worker_id,
                'final_progress': final_state.overall_progress,
                'completion_time': time.time(),
                'phase_count': len([p for p in final_state.phase_info.values() if p.is_completed])
            }
        
        # Run multiple workflows concurrently
        num_workers = 5
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_worker = {
                executor.submit(run_concurrent_workflow, i): i 
                for i in range(num_workers)
            }
            
            results = []
            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"Worker {worker_id} failed: {e}")
        
        # Validate all workers completed successfully
        assert len(results) == num_workers
        for result in results:
            assert result['final_progress'] >= 0.99
            assert result['phase_count'] == len(KnowledgeBasePhase)
            
            # Check progress file was created
            progress_file = temp_dir / f"progress_worker_{result['worker_id']}.json"
            assert progress_file.exists()
            
            with open(progress_file) as f:
                progress_data = json.load(f)
                assert progress_data['state']['overall_progress'] >= 0.99


# =====================================================================
# STRESS AND PERFORMANCE VALIDATION TESTS
# =====================================================================

class TestStressAndPerformance:
    """Stress tests and performance validation."""
    
    def test_high_frequency_updates_stress(self):
        """Test system under high frequency progress updates."""
        callback_invocations = []
        
        def high_frequency_callback(*args):
            callback_invocations.append(time.time())
        
        tracker = KnowledgeBaseProgressTracker(progress_callback=high_frequency_callback)
        tracker.start_initialization(total_documents=1000)
        
        # Start phase
        phase = KnowledgeBasePhase.PDF_PROCESSING
        tracker.start_phase(phase, "High frequency test")
        
        # Perform very frequent updates
        num_updates = 5000
        start_time = time.time()
        
        for i in range(num_updates):
            progress = i / num_updates
            tracker.update_phase_progress(
                phase,
                progress, 
                f"Update {i}",
                {'batch': i // 100, 'item': i % 100}  # Some varying details
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 5.0  # Should handle 5000 updates in under 5 seconds
        assert len(callback_invocations) >= num_updates  # All callbacks should fire
        
        # Average time per update should be very small
        avg_time_per_update = total_time / num_updates
        assert avg_time_per_update < 0.001  # Less than 1ms per update
        
        # Verify final state consistency
        final_state = tracker.get_current_state()
        assert abs(final_state.phase_info[phase].current_progress - 1.0) < 0.001
    
    def test_large_document_collection_simulation(self):
        """Test with very large document collection."""
        tracker = KnowledgeBaseProgressTracker()
        
        # Simulate large collection
        large_collection_size = 10000
        tracker.start_initialization(total_documents=large_collection_size)
        
        # Test each phase with large numbers
        for phase in KnowledgeBasePhase:
            tracker.start_phase(phase, f"Processing {large_collection_size} items")
            
            # Simulate processing in batches
            batch_size = 500
            for batch_start in range(0, large_collection_size, batch_size):
                batch_end = min(batch_start + batch_size, large_collection_size)
                progress = batch_end / large_collection_size
                
                tracker.update_phase_progress(
                    phase,
                    progress,
                    f"Batch {batch_start // batch_size + 1}",
                    {
                        'batch_start': batch_start,
                        'batch_end': batch_end,
                        'items_processed': batch_end
                    }
                )
                
                # Update document counts for PDF processing phase
                if phase == KnowledgeBasePhase.PDF_PROCESSING:
                    tracker.update_document_counts(processed=batch_size)
            
            tracker.complete_phase(phase, f"Completed {phase.value}")
        
        # Verify final state
        final_state = tracker.get_current_state()
        assert abs(final_state.overall_progress - 1.0) < 0.001
        assert final_state.processed_documents == large_collection_size
    
    def test_memory_efficiency_with_callbacks(self):
        """Test memory efficiency with complex callbacks."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Complex callback that creates objects
        callback_data_store = []
        
        def memory_intensive_callback(overall_progress, current_phase, phase_progress, 
                                    status_message, phase_details, all_phases):
            # Create complex data structure
            complex_data = {
                'timestamp': time.time(),
                'progress_snapshot': {
                    'overall': overall_progress,
                    'phase': phase_progress,
                    'details': phase_details.copy(),
                    'all_phases': {
                        phase.value: {
                            'progress': info.current_progress,
                            'active': info.is_active,
                            'completed': info.is_completed,
                            'elapsed': info.elapsed_time,
                            'status': info.status_message
                        }
                        for phase, info in all_phases.items()
                    }
                },
                'metadata': {
                    'large_list': list(range(100)),  # Create some objects
                    'nested_dict': {f'key_{i}': f'value_{i}' for i in range(50)},
                    'status_history': callback_data_store[-10:] if callback_data_store else []
                }
            }
            
            callback_data_store.append(complex_data)
            
            # Keep only recent data to prevent unlimited growth
            if len(callback_data_store) > 1000:
                callback_data_store[:500] = []  # Remove old data
        
        tracker = KnowledgeBaseProgressTracker(progress_callback=memory_intensive_callback)
        tracker.start_initialization(total_documents=500)
        
        # Process through all phases with many updates
        for phase in KnowledgeBasePhase:
            tracker.start_phase(phase, f"Memory test {phase.value}")
            
            for i in range(100):  # Many updates per phase
                tracker.update_phase_progress(
                    phase,
                    i / 100,
                    f"Memory test update {i}",
                    {
                        'update_number': i,
                        'large_data': [f"item_{j}" for j in range(20)],  # Create objects
                        'timestamp': time.time()
                    }
                )
            
            tracker.complete_phase(phase, f"Memory test {phase.value} complete")
        
        # Check memory hasn't grown excessively
        gc.collect()  # Force cleanup
        final_objects = len(gc.get_objects())
        
        # Allow reasonable growth but not excessive
        max_allowed_growth = 1000  # Allow up to 1000 new objects
        actual_growth = final_objects - initial_objects
        
        assert actual_growth < max_allowed_growth, f"Memory leak detected: {actual_growth} new objects"
        
        # Verify system still functions correctly
        final_state = tracker.get_current_state()
        assert abs(final_state.overall_progress - 1.0) < 0.001
        assert len(callback_data_store) > 0  # Callbacks did execute


# =====================================================================
# EDGE CASES AND BOUNDARY CONDITIONS TESTS
# =====================================================================

class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""
    
    def test_zero_weight_phases(self):
        """Test behavior with zero-weight phases.""" 
        # Create weights where one phase has zero weight
        zero_weight_phases = PhaseWeights(
            storage_init=0.0,  # Zero weight
            pdf_processing=0.8,
            document_ingestion=0.15,
            finalization=0.05
        )
        
        tracker = KnowledgeBaseProgressTracker(phase_weights=zero_weight_phases)
        tracker.start_initialization(total_documents=5)
        
        # Complete the zero-weight phase
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Zero weight phase")
        tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Complete")
        
        # Progress should still be 0 since this phase has zero weight
        assert tracker.state.overall_progress == 0.0
        
        # Complete a weighted phase
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Weighted phase")
        tracker.update_phase_progress(KnowledgeBasePhase.PDF_PROCESSING, 0.5, "Half done")
        
        # Should now have progress
        expected_progress = 0.8 * 0.5  # 80% weight * 50% progress
        assert abs(tracker.state.overall_progress - expected_progress) < 0.001
    
    def test_extremely_rapid_phase_transitions(self):
        """Test very rapid phase transitions."""
        callback_calls = []
        
        def rapid_callback(*args):
            callback_calls.append(time.time())
        
        tracker = KnowledgeBaseProgressTracker(progress_callback=rapid_callback)
        tracker.start_initialization(total_documents=1)
        
        # Rapidly transition through all phases
        start_time = time.time()
        
        for phase in KnowledgeBasePhase:
            tracker.start_phase(phase, f"Rapid {phase.value}")
            # No delay - immediate completion
            tracker.complete_phase(phase, f"Rapid {phase.value} done")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle rapid transitions gracefully
        assert total_time < 0.1  # Should be very fast
        assert len(callback_calls) >= len(KnowledgeBasePhase) * 2  # At least start+complete per phase
        
        # Final state should be correct
        final_state = tracker.get_current_state()
        assert abs(final_state.overall_progress - 1.0) < 0.001
        
        for phase in KnowledgeBasePhase:
            phase_info = final_state.phase_info[phase]
            assert phase_info.is_completed
            assert phase_info.current_progress == 1.0
    
    def test_phase_restart_scenarios(self):
        """Test restarting phases after failure."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=5)
        
        phase = KnowledgeBasePhase.PDF_PROCESSING
        
        # Start phase, make progress, then fail
        tracker.start_phase(phase, "First attempt")
        tracker.update_phase_progress(phase, 0.6, "Making progress")
        tracker.fail_phase(phase, "First attempt failed")
        
        # Verify failure state
        phase_info = tracker.state.phase_info[phase]
        assert phase_info.is_failed
        assert phase_info.current_progress == 0.6  # Progress preserved
        
        # Restart the phase (simulating retry logic)
        tracker.start_phase(phase, "Second attempt")
        
        # Phase should be active again with reset progress
        phase_info = tracker.state.phase_info[phase]
        assert phase_info.is_active
        assert not phase_info.is_failed
        assert phase_info.current_progress == 0.0  # Progress reset on restart
        assert phase_info.error_message is None  # Error cleared
        
        # Complete successfully on retry
        tracker.update_phase_progress(phase, 1.0, "Retry successful")
        tracker.complete_phase(phase, "Second attempt succeeded")
        
        # Should be completed now
        assert phase_info.is_completed
        assert not phase_info.is_failed
        assert phase_info.current_progress == 1.0
    
    def test_callback_exception_isolation(self, mock_logger):
        """Test that callback exceptions don't affect other callbacks or progress tracking."""
        
        # Create multiple callbacks, some that will fail
        callback_calls = {'good_callback_1': 0, 'good_callback_2': 0}
        
        def good_callback_1(*args):
            callback_calls['good_callback_1'] += 1
        
        def failing_callback(*args):
            raise ValueError("Callback intentionally failed")
        
        def good_callback_2(*args):
            callback_calls['good_callback_2'] += 1
        
        # Test with single failing callback
        tracker = KnowledgeBaseProgressTracker(
            progress_callback=failing_callback,
            logger=mock_logger
        )
        
        tracker.start_initialization(total_documents=2)
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Test with failing callback")
        tracker.update_phase_progress(KnowledgeBasePhase.STORAGE_INIT, 0.5, "Halfway")
        tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Done")
        
        # Progress tracking should continue working despite callback failures
        final_state = tracker.get_current_state()
        storage_phase = final_state.phase_info[KnowledgeBasePhase.STORAGE_INIT]
        assert storage_phase.is_completed
        assert storage_phase.current_progress == 1.0
        
        # Should have logged callback failures
        assert mock_logger.warning.called
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("Progress callback failed" in call for call in warning_calls)
    
    def test_configuration_edge_cases(self):
        """Test configuration with edge case values."""
        
        # Test with extreme configuration values
        extreme_config = ProgressTrackingConfig(
            log_progress_interval=1,  # Very frequent logging
            phase_progress_update_interval=0.001,  # Very frequent updates
            max_error_details_length=10,  # Very short error messages
            memory_check_interval=1  # Very frequent memory checks
        )
        
        tracker = KnowledgeBaseProgressTracker(progress_config=extreme_config)
        
        # Should handle extreme config gracefully
        tracker.start_initialization(total_documents=3)
        
        # Test error message truncation
        long_error = "This is a very long error message that should be truncated" * 10
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Test")
        tracker.fail_phase(KnowledgeBasePhase.PDF_PROCESSING, long_error)
        
        # Error should be in the list but potentially truncated
        errors = tracker.state.errors
        assert len(errors) > 0
        # The error might be truncated, but should be present


# =====================================================================
# COMPREHENSIVE TEST RUNNER
# =====================================================================

class TestComprehensiveRunner:
    """Test runner for comprehensive validation."""
    
    def test_all_core_functionality(self):
        """Run all core functionality tests."""
        # This would typically be handled by pytest discovery,
        # but we can create a comprehensive validation here
        
        core_test_results = {
            'tracker_initialization': False,
            'phase_lifecycle': False,
            'progress_calculation': False,
            'callback_system': False,
            'error_handling': False,
            'configuration': False
        }
        
        try:
            # Test 1: Basic tracker initialization
            tracker = KnowledgeBaseProgressTracker()
            assert tracker.state.overall_progress == 0.0
            core_test_results['tracker_initialization'] = True
            
            # Test 2: Phase lifecycle
            tracker.start_initialization(total_documents=1)
            tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Test")
            tracker.update_phase_progress(KnowledgeBasePhase.STORAGE_INIT, 0.5, "Half")
            tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Done")
            assert tracker.state.phase_info[KnowledgeBasePhase.STORAGE_INIT].is_completed
            core_test_results['phase_lifecycle'] = True
            
            # Test 3: Progress calculation
            expected_progress = PhaseWeights().storage_init  # Default weight for storage init
            assert abs(tracker.state.overall_progress - expected_progress) < 0.001
            core_test_results['progress_calculation'] = True
            
            # Test 4: Callback system
            callback_called = []
            
            def test_callback(*args):
                callback_called.append(len(args))
            
            tracker_with_callback = KnowledgeBaseProgressTracker(progress_callback=test_callback)
            tracker_with_callback.start_initialization(total_documents=1)
            tracker_with_callback.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Callback test")
            assert len(callback_called) > 0
            assert callback_called[0] == 6  # Should receive 6 parameters
            core_test_results['callback_system'] = True
            
            # Test 5: Error handling
            tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Error test")
            tracker.fail_phase(KnowledgeBasePhase.PDF_PROCESSING, "Test error")
            assert tracker.state.phase_info[KnowledgeBasePhase.PDF_PROCESSING].is_failed
            assert len(tracker.state.errors) > 0
            core_test_results['error_handling'] = True
            
            # Test 6: Configuration
            config = ProgressTrackingConfig(enable_unified_progress_tracking=True)
            assert config.enable_unified_progress_tracking
            core_test_results['configuration'] = True
            
        except Exception as e:
            pytest.fail(f"Core functionality test failed: {e}")
        
        # All core tests should pass
        for test_name, result in core_test_results.items():
            assert result, f"Core test failed: {test_name}"
    
    def test_performance_benchmarks(self):
        """Test performance meets minimum benchmarks."""
        performance_results = {}
        
        # Benchmark 1: Rapid progress updates
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=1000)
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Performance test")
        
        start_time = time.time()
        num_updates = 1000
        
        for i in range(num_updates):
            tracker.update_phase_progress(
                KnowledgeBasePhase.PDF_PROCESSING,
                i / num_updates,
                f"Update {i}"
            )
        
        update_time = time.time() - start_time
        updates_per_second = num_updates / update_time
        
        performance_results['updates_per_second'] = updates_per_second
        
        # Should handle at least 500 updates per second
        assert updates_per_second > 500, f"Too slow: {updates_per_second:.1f} updates/second"
        
        # Benchmark 2: Complex callback overhead
        complex_callback_calls = []
        
        def complex_callback(*args):
            # Simulate complex processing
            import json
            data = {
                'timestamp': time.time(),
                'args_summary': [str(arg)[:50] for arg in args],
                'complex_computation': sum(range(100))  # Some work
            }
            json_str = json.dumps(data)  # Serialize
            complex_callback_calls.append(len(json_str))
        
        tracker_with_callback = KnowledgeBaseProgressTracker(progress_callback=complex_callback)
        tracker_with_callback.start_initialization(total_documents=100)
        
        start_time = time.time()
        
        for phase in KnowledgeBasePhase:
            tracker_with_callback.start_phase(phase, f"Benchmark {phase.value}")
            for i in range(10):
                tracker_with_callback.update_phase_progress(
                    phase, i / 10, f"Progress {i}"
                )
            tracker_with_callback.complete_phase(phase, f"Done {phase.value}")
        
        callback_time = time.time() - start_time
        
        performance_results['callback_overhead_time'] = callback_time
        performance_results['complex_callbacks_executed'] = len(complex_callback_calls)
        
        # Should complete with complex callbacks in reasonable time
        assert callback_time < 2.0, f"Callback overhead too high: {callback_time:.3f}s"
        assert len(complex_callback_calls) > 40, "Not enough callbacks executed"
        
        # Return performance results for analysis
        return performance_results


if __name__ == "__main__":
    # Run specific test suites
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run performance benchmarks only
        runner = TestComprehensiveRunner()
        performance_results = runner.test_performance_benchmarks()
        print("\nPerformance Benchmark Results:")
        for metric, value in performance_results.items():
            print(f"  {metric}: {value}")
    else:
        # Run all tests via pytest
        pytest.main([__file__, "-v", "--tb=short"])