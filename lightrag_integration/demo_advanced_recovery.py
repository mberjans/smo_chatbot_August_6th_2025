#!/usr/bin/env python3
"""
Demonstration of Advanced Recovery and Graceful Degradation System.

This script shows how the advanced recovery mechanisms work in practice,
including progressive degradation, resource-aware recovery, intelligent
retry backoff, and checkpoint/resume capabilities.

Usage:
    python lightrag_integration/demo_advanced_recovery.py

Author: Claude Code (Anthropic)
Created: 2025-08-07
"""

import asyncio
import tempfile
import time
import random
from pathlib import Path

from lightrag_integration.advanced_recovery_system import (
    AdvancedRecoverySystem, DegradationMode, FailureType, 
    ResourceThresholds, DegradationConfig
)
from lightrag_integration.recovery_integration import RecoveryIntegratedProcessor
from lightrag_integration.unified_progress_tracker import (
    KnowledgeBaseProgressTracker, KnowledgeBasePhase
)
from lightrag_integration.progress_config import ProgressTrackingConfig


class MockRAGSystem:
    """Mock RAG system for demonstration purposes."""
    
    def __init__(self, failure_rate: float = 0.2):
        """Initialize mock system with configurable failure rate."""
        self.failure_rate = failure_rate
        self.api_call_count = 0
        
    async def process_document(self, doc_id: str) -> bool:
        """Mock document processing with simulated failures."""
        self.api_call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Simulate failures based on failure rate
        if random.random() < self.failure_rate:
            # Simulate different types of failures
            failure_types = [
                "Rate limit exceeded",
                "API timeout occurred",
                "Memory pressure detected",
                "Network connection error",
                "Processing error in document analysis"
            ]
            raise Exception(random.choice(failure_types))
        
        return True


async def demo_progressive_degradation():
    """Demonstrate progressive degradation strategies."""
    print("=== Progressive Degradation Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize recovery system
        recovery_system = AdvancedRecoverySystem(
            checkpoint_dir=Path(temp_dir) / "checkpoints"
        )
        
        # Initialize with sample documents
        documents = [f"document_{i:03d}" for i in range(20)]
        recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=5
        )
        
        print(f"Initialized with {len(documents)} documents")
        print(f"Initial degradation mode: {recovery_system.current_degradation_mode.value}")
        print(f"Initial batch size: {recovery_system._current_batch_size}")
        
        # Simulate various failure scenarios
        failure_scenarios = [
            (FailureType.API_RATE_LIMIT, "Rate limit exceeded - too many requests"),
            (FailureType.API_RATE_LIMIT, "Rate limit exceeded again"),
            (FailureType.MEMORY_PRESSURE, "High memory usage detected"),
            (FailureType.API_ERROR, "API service unavailable"),
            (FailureType.API_ERROR, "Another API error"),
            (FailureType.PROCESSING_ERROR, "Document processing failed")
        ]
        
        for i, (failure_type, error_msg) in enumerate(failure_scenarios):
            print(f"\n--- Failure {i+1}: {failure_type.value} ---")
            
            recovery_strategy = recovery_system.handle_failure(
                failure_type=failure_type,
                error_message=error_msg,
                document_id=f"document_{i:03d}"
            )
            
            print(f"Recovery strategy: {recovery_strategy['action']}")
            print(f"Degradation mode: {recovery_system.current_degradation_mode.value}")
            print(f"Batch size: {recovery_system._current_batch_size}")
            print(f"Backoff recommended: {recovery_strategy.get('backoff_seconds', 'N/A'):.2f}s")
            
            if recovery_strategy.get('checkpoint_recommended'):
                checkpoint_id = recovery_system.create_checkpoint({'failure_scenario': i+1})
                print(f"Checkpoint created: {checkpoint_id}")
        
        # Show final status
        status = recovery_system.get_recovery_status()
        print(f"\n--- Final Status ---")
        print(f"Degradation mode: {status['degradation_mode']}")
        print(f"Current batch size: {status['current_batch_size']}")
        print(f"Error counts: {status['error_counts']}")
        print(f"Success rate: {status['document_progress']['success_rate']:.2%}")


async def demo_resource_aware_recovery():
    """Demonstrate resource-aware recovery mechanisms."""
    print("\n=== Resource-Aware Recovery Demo ===")
    
    # Create custom resource thresholds for demonstration
    thresholds = ResourceThresholds(
        memory_warning_percent=50.0,   # Lower thresholds for demo
        memory_critical_percent=70.0,
        disk_warning_percent=60.0,
        disk_critical_percent=80.0
    )
    
    recovery_system = AdvancedRecoverySystem(
        resource_thresholds=thresholds
    )
    
    # Check current system resources
    resources = recovery_system.resource_monitor.get_current_resources()
    print("Current system resources:")
    for resource, value in resources.items():
        print(f"  {resource}: {value:.2f}")
    
    # Check for resource pressure
    pressure = recovery_system.resource_monitor.check_resource_pressure()
    print(f"\nResource pressure detected: {pressure}")
    
    if pressure:
        print("\nSimulating resource pressure response...")
        for resource, recommendation in pressure.items():
            print(f"  {resource}: {recommendation}")
            
            # Simulate handling resource pressure
            if 'memory' in resource:
                failure_type = FailureType.MEMORY_PRESSURE
            elif 'disk' in resource:
                failure_type = FailureType.DISK_SPACE
            else:
                failure_type = FailureType.RESOURCE_EXHAUSTION
            
            recovery_strategy = recovery_system.handle_failure(
                failure_type=failure_type,
                error_message=f"Resource pressure: {recommendation}",
                context={'resource': resource, 'pressure_level': recommendation}
            )
            
            print(f"    Recovery action: {recovery_strategy['action']}")
            print(f"    Batch size adjustment: {recovery_strategy.get('batch_size_adjustment', 1.0)}")
    else:
        print("\nNo resource pressure detected - system operating normally")


async def demo_intelligent_backoff():
    """Demonstrate intelligent retry backoff strategies."""
    print("\n=== Intelligent Backoff Demo ===")
    
    recovery_system = AdvancedRecoverySystem()
    backoff_calc = recovery_system.backoff_calculator
    
    failure_types = [
        FailureType.API_RATE_LIMIT,
        FailureType.API_TIMEOUT,
        FailureType.NETWORK_ERROR,
        FailureType.MEMORY_PRESSURE
    ]
    
    print("Backoff calculations for different failure types and attempts:")
    print("=" * 70)
    print(f"{'Failure Type':<20} {'Attempt':<8} {'Exponential':<12} {'Adaptive':<12}")
    print("=" * 70)
    
    for failure_type in failure_types:
        for attempt in range(1, 6):
            exp_backoff = backoff_calc.calculate_backoff(
                failure_type, attempt, strategy=recovery_system.BackoffStrategy.EXPONENTIAL
            )
            adaptive_backoff = backoff_calc.calculate_backoff(
                failure_type, attempt, strategy=recovery_system.BackoffStrategy.ADAPTIVE
            )
            
            print(f"{failure_type.value:<20} {attempt:<8} {exp_backoff:<12.2f} {adaptive_backoff:<12.2f}")
    
    print("\nSimulating failure pattern effects on adaptive backoff:")
    
    # Simulate some failures and successes
    for i in range(10):
        if i % 3 == 0:  # Every third operation succeeds
            backoff_calc.record_success()
            print(f"Operation {i+1}: SUCCESS")
        else:
            backoff_time = backoff_calc.calculate_backoff(FailureType.API_ERROR, 1)
            print(f"Operation {i+1}: FAILED (backoff: {backoff_time:.2f}s)")


async def demo_checkpoint_resume():
    """Demonstrate checkpoint and resume capabilities."""
    print("\n=== Checkpoint and Resume Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        recovery_system = AdvancedRecoverySystem(checkpoint_dir=checkpoint_dir)
        
        # Initialize processing session
        documents = [f"critical_doc_{i:03d}" for i in range(15)]
        recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        # Process some documents
        for i in range(0, 5):
            recovery_system.mark_document_processed(documents[i])
        
        # Simulate some failures
        for i in range(5, 8):
            recovery_system.handle_failure(
                FailureType.PROCESSING_ERROR,
                f"Processing failed for document {documents[i]}",
                document_id=documents[i]
            )
        
        print(f"Processed 5 documents, failed 3 documents")
        
        # Create checkpoint
        checkpoint_id = recovery_system.create_checkpoint({
            'demo_phase': 'mid_processing',
            'note': 'Checkpoint created during demonstration'
        })
        print(f"Created checkpoint: {checkpoint_id}")
        
        # Show current state
        status = recovery_system.get_recovery_status()
        print(f"Documents processed: {status['document_progress']['processed']}")
        print(f"Documents failed: {status['document_progress']['failed']}")
        print(f"Documents pending: {status['document_progress']['pending']}")
        
        # Simulate system restart by creating new recovery system
        print("\nSimulating system restart...")
        new_recovery_system = AdvancedRecoverySystem(checkpoint_dir=checkpoint_dir)
        
        # List available checkpoints
        checkpoints = new_recovery_system.checkpoint_manager.list_checkpoints()
        print(f"Available checkpoints: {checkpoints}")
        
        # Resume from checkpoint
        if new_recovery_system.resume_from_checkpoint(checkpoint_id):
            print(f"Successfully resumed from checkpoint {checkpoint_id}")
            
            # Show resumed state
            resumed_status = new_recovery_system.get_recovery_status()
            print(f"Resumed - Documents processed: {resumed_status['document_progress']['processed']}")
            print(f"Resumed - Documents failed: {resumed_status['document_progress']['failed']}")
            print(f"Resumed - Documents pending: {resumed_status['document_progress']['pending']}")
        else:
            print("Failed to resume from checkpoint")


async def demo_integrated_processing():
    """Demonstrate the integrated recovery processor."""
    print("\n=== Integrated Processing Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock RAG system with high failure rate for demonstration
        mock_rag = MockRAGSystem(failure_rate=0.3)
        
        # Create progress tracker
        progress_config = ProgressTrackingConfig(
            enable_progress_tracking=True,
            save_unified_progress_to_file=True,
            unified_progress_file_path=Path(temp_dir) / "progress.json"
        )
        progress_tracker = KnowledgeBaseProgressTracker(progress_config=progress_config)
        
        # Create recovery system
        recovery_system = AdvancedRecoverySystem(
            progress_tracker=progress_tracker,
            checkpoint_dir=Path(temp_dir) / "checkpoints"
        )
        
        # Create integrated processor
        processor = RecoveryIntegratedProcessor(
            rag_system=mock_rag,
            recovery_system=recovery_system,
            progress_tracker=progress_tracker,
            enable_checkpointing=True,
            checkpoint_interval=3  # Checkpoint every 3 documents
        )
        
        # Define test documents
        test_documents = [
            "essential_metabolomics_001",
            "critical_pathway_002", 
            "document_003",
            "important_biomarker_004",
            "document_005",
            "essential_protocol_006",
            "document_007",
            "key_analysis_008",
            "document_009",
            "document_010"
        ]
        
        print(f"Starting processing of {len(test_documents)} documents")
        
        # Define progress callback
        def progress_callback(doc_id: str, result: Dict[str, Any]):
            status = "SUCCESS" if result['success'] else f"FAILED ({result['error']})"
            print(f"  {doc_id}: {status}")
        
        # Process documents with recovery
        start_time = time.time()
        results = await processor.process_documents_with_recovery(
            documents=test_documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            initial_batch_size=4,
            progress_callback=progress_callback
        )
        processing_time = time.time() - start_time
        
        # Display results
        print(f"\n--- Processing Results ---")
        print(f"Total processing time: {processing_time:.2f}s")
        print(f"Documents processed: {len(results['processed_documents'])}")
        print(f"Documents failed: {len(results['failed_documents'])}")
        print(f"Documents skipped: {len(results['skipped_documents'])}")
        print(f"Recovery events: {len(results['recovery_events'])}")
        print(f"Checkpoints created: {len(results['checkpoints'])}")
        print(f"Final degradation mode: {results['final_degradation_mode'].value}")
        
        # Show statistics
        stats = results['statistics']
        print(f"\n--- Statistics ---")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Processing speed: {stats['documents_per_second']:.2f} docs/sec")
        print(f"Recovery actions taken: {stats['total_recovery_actions']}")
        
        # Show failed documents
        if results['failed_documents']:
            print(f"\n--- Failed Documents ---")
            for doc_id, error in results['failed_documents'].items():
                print(f"  {doc_id}: {error}")
        
        # Show recovery events summary
        if results['recovery_events']:
            print(f"\n--- Recovery Events Summary ---")
            event_types = {}
            for event in results['recovery_events']:
                action = event.get('action', 'unknown')
                event_types[action] = event_types.get(action, 0) + 1
            
            for action, count in event_types.items():
                print(f"  {action}: {count} times")


async def main():
    """Run all demonstrations."""
    print("Advanced Recovery and Graceful Degradation System Demonstration")
    print("=" * 70)
    
    await demo_progressive_degradation()
    await demo_resource_aware_recovery()
    await demo_intelligent_backoff()
    await demo_checkpoint_resume()
    await demo_integrated_processing()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("\nKey features demonstrated:")
    print("✓ Progressive degradation with multiple fallback strategies")
    print("✓ Resource-aware recovery with system monitoring")
    print("✓ Intelligent retry backoff with adaptive strategies")
    print("✓ Checkpoint and resume capability for long-running processes")
    print("✓ Multiple degradation modes (essential, minimal, offline, safe)")
    print("✓ Integration with existing progress tracking system")
    print("✓ Comprehensive recovery statistics and monitoring")


if __name__ == "__main__":
    import sys
    
    # Add the lightrag_integration directory to Python path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    asyncio.run(main())