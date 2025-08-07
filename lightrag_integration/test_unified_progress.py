#!/usr/bin/env python3
"""
Test script for unified progress tracking system.

This script demonstrates and validates the unified progress tracking system
for knowledge base construction, including integration with existing components.

Run with: python -m lightrag_integration.test_unified_progress
"""

import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import our unified progress tracking components
try:
    from .unified_progress_tracker import (
        KnowledgeBaseProgressTracker,
        KnowledgeBasePhase,
        PhaseWeights,
        UnifiedProgressState
    )
    from .progress_integration import (
        create_unified_progress_tracker,
        ConsoleProgressCallback,
        ProgressCallbackBuilder
    )
    from .progress_config import ProgressTrackingConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)


class ProgressTestSuite:
    """Test suite for unified progress tracking system."""
    
    def __init__(self):
        self.logger = logging.getLogger("progress_test")
        self.test_results = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="progress_test_"))
        self.logger.info(f"Using temporary directory: {self.temp_dir}")
    
    async def run_all_tests(self):
        """Run all test cases."""
        print("🧪 Unified Progress Tracking System Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_basic_progress_tracking,
            self.test_phase_weights_calculation,
            self.test_callback_system,
            self.test_error_handling,
            self.test_progress_persistence,
            self.test_integration_simulation
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            print(f"\n📋 Running {test_name}...")
            
            try:
                start_time = time.time()
                await test_method()
                duration = time.time() - start_time
                
                self.test_results[test_name] = {
                    'status': 'PASSED',
                    'duration': duration,
                    'error': None
                }
                print(f"✅ {test_name} PASSED ({duration:.2f}s)")
                
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'duration': time.time() - start_time,
                    'error': str(e)
                }
                print(f"❌ {test_name} FAILED: {e}")
                self.logger.exception(f"Test {test_name} failed")
        
        self._print_test_summary()
    
    async def test_basic_progress_tracking(self):
        """Test basic progress tracking functionality."""
        # Create progress tracker
        progress_tracker = KnowledgeBaseProgressTracker()
        
        # Start initialization
        progress_tracker.start_initialization(total_documents=5)
        
        assert progress_tracker.state.total_documents == 5
        assert progress_tracker.state.overall_progress == 0.0
        
        # Test each phase
        for phase in KnowledgeBasePhase:
            progress_tracker.start_phase(phase, f"Testing {phase.value}")
            assert progress_tracker.state.current_phase == phase
            
            # Update progress within phase
            progress_tracker.update_phase_progress(
                phase, 0.5, f"50% through {phase.value}"
            )
            
            phase_info = progress_tracker.state.phase_info[phase]
            assert phase_info.current_progress == 0.5
            assert phase_info.is_active
            
            # Complete phase
            progress_tracker.complete_phase(phase, f"Completed {phase.value}")
            assert phase_info.is_completed
            assert not phase_info.is_active
        
        # Verify final progress
        final_state = progress_tracker.get_current_state()
        assert abs(final_state.overall_progress - 1.0) < 0.001
        
        print("  ✓ Basic progress tracking functionality verified")
    
    async def test_phase_weights_calculation(self):
        """Test phase weights and progress calculation."""
        # Test default weights
        default_weights = PhaseWeights()
        assert abs(sum([
            default_weights.storage_init,
            default_weights.pdf_processing,
            default_weights.document_ingestion,
            default_weights.finalization
        ]) - 1.0) < 0.001
        
        # Test custom weights
        custom_weights = PhaseWeights(
            storage_init=0.1,
            pdf_processing=0.7,
            document_ingestion=0.15,
            finalization=0.05
        )
        
        progress_tracker = KnowledgeBaseProgressTracker(phase_weights=custom_weights)
        progress_tracker.start_initialization()
        
        # Complete storage init (should be 10% of total)
        progress_tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage init")
        progress_tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Done")
        
        expected_progress = custom_weights.storage_init
        actual_progress = progress_tracker.state.overall_progress
        assert abs(actual_progress - expected_progress) < 0.001
        
        # Half complete PDF processing (should add 35% to total)
        progress_tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "PDF processing")
        progress_tracker.update_phase_progress(KnowledgeBasePhase.PDF_PROCESSING, 0.5, "Half done")
        
        expected_progress = custom_weights.storage_init + (custom_weights.pdf_processing * 0.5)
        actual_progress = progress_tracker.state.overall_progress
        assert abs(actual_progress - expected_progress) < 0.001
        
        print("  ✓ Phase weights calculation verified")
    
    async def test_callback_system(self):
        """Test progress callback system."""
        callback_calls = []
        
        def test_callback(overall_progress, current_phase, phase_progress, 
                         status_message, phase_details, all_phases):
            callback_calls.append({
                'overall_progress': overall_progress,
                'current_phase': current_phase,
                'phase_progress': phase_progress,
                'status_message': status_message,
                'timestamp': time.time()
            })
        
        progress_tracker = KnowledgeBaseProgressTracker(progress_callback=test_callback)
        progress_tracker.start_initialization(total_documents=3)
        
        # Trigger some progress updates
        progress_tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Starting storage")
        progress_tracker.update_phase_progress(KnowledgeBasePhase.STORAGE_INIT, 0.5, "Half done")
        progress_tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage complete")
        
        # Verify callbacks were called
        assert len(callback_calls) >= 3  # At least start, update, complete
        assert callback_calls[0]['current_phase'] == KnowledgeBasePhase.STORAGE_INIT
        assert callback_calls[-1]['overall_progress'] > callback_calls[0]['overall_progress']
        
        print(f"  ✓ Callback system verified ({len(callback_calls)} callbacks triggered)")
    
    async def test_error_handling(self):
        """Test error handling and recovery."""
        progress_tracker = KnowledgeBaseProgressTracker()
        progress_tracker.start_initialization(total_documents=2)
        
        # Start and fail a phase
        progress_tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Processing PDFs")
        progress_tracker.update_phase_progress(KnowledgeBasePhase.PDF_PROCESSING, 0.3, "Working...")
        
        # Simulate failure
        error_message = "Simulated PDF processing error"
        progress_tracker.fail_phase(KnowledgeBasePhase.PDF_PROCESSING, error_message)
        
        # Verify phase is marked as failed
        phase_info = progress_tracker.state.phase_info[KnowledgeBasePhase.PDF_PROCESSING]
        assert phase_info.is_failed
        assert phase_info.error_message == error_message
        # Check that error was added to the errors list (includes phase name)
        error_found = any(error_message in error for error in progress_tracker.state.errors)
        assert error_found
        
        # Test continuing after failure
        progress_tracker.start_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Attempting recovery")
        progress_tracker.complete_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Recovery successful")
        
        # Verify we can continue after failure
        ingestion_info = progress_tracker.state.phase_info[KnowledgeBasePhase.DOCUMENT_INGESTION]
        assert ingestion_info.is_completed
        
        print("  ✓ Error handling and recovery verified")
    
    async def test_progress_persistence(self):
        """Test progress persistence to file."""
        progress_file = self.temp_dir / "test_progress.json"
        
        config = ProgressTrackingConfig(
            save_unified_progress_to_file=True,
            unified_progress_file_path=progress_file
        )
        
        progress_tracker = KnowledgeBaseProgressTracker(progress_config=config)
        progress_tracker.start_initialization(total_documents=1)
        
        # Make some progress
        progress_tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Test persistence")
        progress_tracker.update_phase_progress(KnowledgeBasePhase.STORAGE_INIT, 0.7, "Almost done")
        # Force progress calculation
        progress_tracker.state.overall_progress = progress_tracker.state.calculate_overall_progress()
        
        # Give file write a moment
        await asyncio.sleep(0.1)
        
        # Verify file was created and contains progress data
        assert progress_file.exists()
        
        import json
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        assert 'state' in progress_data
        assert 'timestamp' in progress_data
        assert progress_data['state']['overall_progress'] > 0
        
        print("  ✓ Progress persistence verified")
    
    async def test_integration_simulation(self):
        """Test full integration simulation."""
        # Create comprehensive callback
        console_output = []
        
        def capture_console_callback(overall_progress, current_phase, phase_progress, 
                                   status_message, phase_details, all_phases):
            console_output.append(f"{overall_progress:.1%} | {current_phase.value}: {status_message}")
        
        # Create progress tracker with full configuration
        config = ProgressTrackingConfig(
            enable_unified_progress_tracking=True,
            enable_phase_based_progress=True,
            save_unified_progress_to_file=True,
            unified_progress_file_path=self.temp_dir / "integration_test.json",
            log_processing_stats=True
        )
        
        progress_tracker = create_unified_progress_tracker(
            progress_config=config,
            progress_callback=capture_console_callback,
            logger=self.logger
        )
        
        # Simulate full knowledge base initialization
        total_docs = 4
        progress_tracker.start_initialization(total_documents=total_docs)
        
        # Phase 1: Storage Initialization
        progress_tracker.start_phase(
            KnowledgeBasePhase.STORAGE_INIT,
            "Initializing storage directories"
        )
        await asyncio.sleep(0.05)  # Simulate work
        progress_tracker.update_phase_progress(
            KnowledgeBasePhase.STORAGE_INIT,
            0.5,
            "Creating vector databases"
        )
        await asyncio.sleep(0.05)
        progress_tracker.complete_phase(
            KnowledgeBasePhase.STORAGE_INIT,
            "Storage initialization completed"
        )
        
        # Phase 2: PDF Processing
        progress_tracker.start_phase(
            KnowledgeBasePhase.PDF_PROCESSING,
            "Processing PDF documents"
        )
        
        for i in range(total_docs):
            await asyncio.sleep(0.02)  # Simulate processing time
            progress_tracker.update_phase_progress(
                KnowledgeBasePhase.PDF_PROCESSING,
                (i + 1) / total_docs,
                f"Processed document {i + 1}/{total_docs}",
                {
                    'completed_files': i + 1,
                    'total_files': total_docs,
                    'success_rate': 100.0
                }
            )
            progress_tracker.update_document_counts(processed=1)
        
        progress_tracker.complete_phase(
            KnowledgeBasePhase.PDF_PROCESSING,
            f"Processed {total_docs} documents"
        )
        
        # Phase 3: Document Ingestion
        progress_tracker.start_phase(
            KnowledgeBasePhase.DOCUMENT_INGESTION,
            "Ingesting into knowledge graph"
        )
        
        # Simulate batch ingestion
        batch_size = 2
        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            await asyncio.sleep(0.03)
            
            progress_tracker.update_phase_progress(
                KnowledgeBasePhase.DOCUMENT_INGESTION,
                batch_end / total_docs,
                f"Ingested batch {batch_start // batch_size + 1}",
                {
                    'ingested_documents': batch_end,
                    'total_documents': total_docs
                }
            )
        
        progress_tracker.complete_phase(
            KnowledgeBasePhase.DOCUMENT_INGESTION,
            "Document ingestion completed"
        )
        
        # Phase 4: Finalization
        progress_tracker.start_phase(
            KnowledgeBasePhase.FINALIZATION,
            "Finalizing knowledge base"
        )
        await asyncio.sleep(0.02)
        progress_tracker.update_phase_progress(
            KnowledgeBasePhase.FINALIZATION,
            0.5,
            "Optimizing indices"
        )
        await asyncio.sleep(0.02)
        progress_tracker.complete_phase(
            KnowledgeBasePhase.FINALIZATION,
            "Knowledge base ready"
        )
        
        # Verify final state
        final_state = progress_tracker.get_current_state()
        assert abs(final_state.overall_progress - 1.0) < 0.001
        assert final_state.processed_documents == total_docs
        assert len(console_output) > 0
        
        # Verify all phases completed
        for phase in KnowledgeBasePhase:
            phase_info = final_state.phase_info[phase]
            assert phase_info.is_completed, f"Phase {phase.value} not completed"
        
        print(f"  ✓ Full integration simulation verified ({len(console_output)} progress updates)")
        
        # Print progress summary
        summary = progress_tracker.get_progress_summary()
        print(f"  📊 Final Summary: {summary}")
    
    def _print_test_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        failed = len(self.test_results) - passed
        total_time = sum(result['duration'] for result in self.test_results.values())
        
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Total Time: {total_time:.2f}s")
        
        if failed > 0:
            print("\n❌ Failed Tests:")
            for test_name, result in self.test_results.items():
                if result['status'] == 'FAILED':
                    print(f"  - {test_name}: {result['error']}")
        
        print("\n🧹 Cleanup:")
        print(f"Temporary directory: {self.temp_dir}")
        
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print("✅ Temporary files cleaned up")
        except Exception as e:
            print(f"⚠️  Could not clean up temp files: {e}")
        
        if failed == 0:
            print("\n🎉 All tests passed! The unified progress tracking system is working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
            return False
        
        return True


async def main():
    """Main test runner."""
    test_suite = ProgressTestSuite()
    success = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())