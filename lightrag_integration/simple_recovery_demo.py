#!/usr/bin/env python3
"""
Simple demonstration of the Advanced Recovery System functionality.

This script demonstrates key features without complex imports.
"""

import asyncio
import tempfile
import time
import json
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List


class DegradationMode(Enum):
    """Degradation modes for recovery system."""
    OPTIMAL = "optimal"
    ESSENTIAL = "essential"
    MINIMAL = "minimal"
    OFFLINE = "offline"
    SAFE = "safe"


class FailureType(Enum):
    """Types of failures that can occur."""
    API_RATE_LIMIT = "api_rate_limit"
    API_TIMEOUT = "api_timeout"
    MEMORY_PRESSURE = "memory_pressure"
    PROCESSING_ERROR = "processing_error"


@dataclass
class RecoveryStrategy:
    """Recovery strategy result."""
    action: str
    degradation_needed: bool = False
    batch_size_adjustment: float = 1.0
    backoff_seconds: float = 0.0
    checkpoint_recommended: bool = False


class SimpleRecoveryDemo:
    """Simplified recovery system for demonstration."""
    
    def __init__(self):
        self.current_mode = DegradationMode.OPTIMAL
        self.batch_size = 10
        self.original_batch_size = 10
        self.error_counts = {}
        self.processed_docs = []
        self.failed_docs = {}
        self.pending_docs = []
    
    def initialize_session(self, documents: List[str]):
        """Initialize processing session."""
        self.pending_docs = documents.copy()
        self.processed_docs.clear()
        self.failed_docs.clear()
        self.error_counts.clear()
        print(f"Initialized session with {len(documents)} documents")
    
    def handle_failure(self, failure_type: FailureType, error_message: str, doc_id: str = None) -> RecoveryStrategy:
        """Handle a failure and return recovery strategy."""
        # Count the failure
        self.error_counts[failure_type] = self.error_counts.get(failure_type, 0) + 1
        
        if doc_id and doc_id in self.pending_docs:
            self.failed_docs[doc_id] = error_message
            self.pending_docs.remove(doc_id)
        
        # Determine recovery strategy
        strategy = RecoveryStrategy(action="retry")
        
        if failure_type == FailureType.API_RATE_LIMIT:
            strategy.action = "backoff_and_retry"
            strategy.backoff_seconds = min(60, 2 ** self.error_counts[failure_type])
            strategy.batch_size_adjustment = 0.5
            if self.error_counts[failure_type] > 3:
                strategy.degradation_needed = True
        
        elif failure_type == FailureType.MEMORY_PRESSURE:
            strategy.action = "reduce_resources"
            strategy.batch_size_adjustment = 0.3
            strategy.degradation_needed = True
            strategy.checkpoint_recommended = True
        
        elif failure_type == FailureType.PROCESSING_ERROR:
            if self.error_counts[failure_type] > 5:
                strategy.action = "degrade_to_safe_mode"
                strategy.degradation_needed = True
        
        # Apply strategy
        self._apply_strategy(strategy)
        
        return strategy
    
    def _apply_strategy(self, strategy: RecoveryStrategy):
        """Apply recovery strategy."""
        # Adjust batch size
        if strategy.batch_size_adjustment != 1.0:
            new_size = max(1, int(self.batch_size * strategy.batch_size_adjustment))
            print(f"Adjusting batch size: {self.batch_size} -> {new_size}")
            self.batch_size = new_size
        
        # Apply degradation
        if strategy.degradation_needed:
            if strategy.action == "reduce_resources":
                self.current_mode = DegradationMode.MINIMAL
            elif strategy.action == "degrade_to_safe_mode":
                self.current_mode = DegradationMode.SAFE
            else:
                self.current_mode = DegradationMode.ESSENTIAL
            
            print(f"Degraded to {self.current_mode.value} mode")
    
    def get_next_batch(self) -> List[str]:
        """Get next batch of documents to process."""
        if not self.pending_docs:
            return []
        
        batch_size = self._calculate_batch_size()
        batch = self.pending_docs[:batch_size]
        
        # Apply document prioritization in degraded modes
        if self.current_mode == DegradationMode.ESSENTIAL:
            # Prioritize essential documents
            essential = [d for d in batch if any(kw in d.lower() for kw in ['essential', 'critical', 'important'])]
            regular = [d for d in batch if d not in essential]
            batch = essential + regular
        
        return batch
    
    def _calculate_batch_size(self) -> int:
        """Calculate optimal batch size based on current mode."""
        base_size = self.batch_size
        
        if self.current_mode == DegradationMode.SAFE:
            return 1
        elif self.current_mode == DegradationMode.MINIMAL:
            return min(base_size, 3)
        elif self.current_mode == DegradationMode.ESSENTIAL:
            return min(base_size, 5)
        
        return base_size
    
    def mark_processed(self, doc_id: str):
        """Mark document as successfully processed."""
        if doc_id in self.pending_docs:
            self.pending_docs.remove(doc_id)
        self.processed_docs.append(doc_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        total = len(self.processed_docs) + len(self.failed_docs) + len(self.pending_docs)
        return {
            'degradation_mode': self.current_mode.value,
            'batch_size': self.batch_size,
            'original_batch_size': self.original_batch_size,
            'processed': len(self.processed_docs),
            'failed': len(self.failed_docs),
            'pending': len(self.pending_docs),
            'success_rate': len(self.processed_docs) / max(1, total - len(self.pending_docs)),
            'error_counts': {k.value: v for k, v in self.error_counts.items()}
        }


def demo_progressive_degradation():
    """Demonstrate progressive degradation."""
    print("=== Progressive Degradation Demo ===")
    
    recovery = SimpleRecoveryDemo()
    documents = [f"document_{i:03d}" for i in range(15)]
    recovery.initialize_session(documents)
    
    print(f"Initial state: {recovery.current_mode.value}, batch size: {recovery.batch_size}")
    
    # Simulate various failure scenarios
    failures = [
        (FailureType.API_RATE_LIMIT, "Rate limit exceeded - attempt 1"),
        (FailureType.API_RATE_LIMIT, "Rate limit exceeded - attempt 2"),
        (FailureType.API_RATE_LIMIT, "Rate limit exceeded - attempt 3"),
        (FailureType.API_RATE_LIMIT, "Rate limit exceeded - attempt 4"),  # Should trigger degradation
        (FailureType.MEMORY_PRESSURE, "High memory usage detected"),
    ]
    
    for i, (failure_type, error_msg) in enumerate(failures):
        print(f"\n--- Failure {i+1}: {failure_type.value} ---")
        doc_id = f"document_{i:03d}"
        
        strategy = recovery.handle_failure(failure_type, error_msg, doc_id)
        
        print(f"Recovery action: {strategy.action}")
        print(f"Backoff time: {strategy.backoff_seconds:.1f}s")
        print(f"Current mode: {recovery.current_mode.value}")
        print(f"Batch size: {recovery.batch_size}")
        
        if strategy.checkpoint_recommended:
            print("📋 Checkpoint recommended")
    
    # Show final status
    status = recovery.get_status()
    print(f"\n--- Final Status ---")
    print(f"Mode: {status['degradation_mode']}")
    print(f"Batch size: {status['batch_size']} (was {status['original_batch_size']})")
    print(f"Success rate: {status['success_rate']:.1%}")
    print(f"Error counts: {status['error_counts']}")


def demo_batch_processing():
    """Demonstrate batch processing with different modes."""
    print("\n=== Batch Processing Demo ===")
    
    recovery = SimpleRecoveryDemo()
    
    # Mix of regular and essential documents
    documents = [
        "essential_metabolomics_001",
        "regular_document_002",
        "critical_pathway_003",
        "standard_file_004", 
        "important_biomarker_005",
        "routine_analysis_006",
        "essential_protocol_007",
        "normal_report_008"
    ]
    
    recovery.initialize_session(documents)
    
    # Test different degradation modes
    modes = [DegradationMode.OPTIMAL, DegradationMode.ESSENTIAL, DegradationMode.MINIMAL, DegradationMode.SAFE]
    
    for mode in modes:
        recovery.current_mode = mode
        recovery.pending_docs = documents.copy()  # Reset for demo
        
        print(f"\n--- {mode.value.upper()} MODE ---")
        
        batch = recovery.get_next_batch()
        print(f"Batch size: {len(batch)} (max allowed: {recovery._calculate_batch_size()})")
        print(f"Documents in batch: {batch[:3]}...")  # Show first 3
        
        if mode == DegradationMode.ESSENTIAL:
            essential_docs = [d for d in batch if any(kw in d.lower() for kw in ['essential', 'critical', 'important'])]
            print(f"Essential documents prioritized: {len(essential_docs)}/{len(batch)}")


def demo_resource_monitoring():
    """Demonstrate resource monitoring simulation."""
    print("\n=== Resource Monitoring Demo ===")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print("Current system resources:")
        print(f"  Memory: {memory.percent:.1f}% used ({memory.available / (1024**3):.1f} GB available)")
        print(f"  Disk: {disk.percent:.1f}% used ({disk.free / (1024**3):.1f} GB free)")
        
        # Simulate resource pressure responses
        print("\nResource pressure simulation:")
        if memory.percent > 75:
            print("  🚨 High memory usage detected - would reduce batch size")
        else:
            print("  ✅ Memory usage normal")
            
        if disk.percent > 85:
            print("  🚨 Low disk space - would enable cleanup")
        else:
            print("  ✅ Disk space adequate")
            
    except ImportError:
        print("psutil not available - using simulated values")
        print("Simulated resources:")
        print("  Memory: 82.3% used (2.1 GB available)")
        print("  🚨 High memory usage detected - would reduce batch size")
        print("  Disk: 45.7% used (120.5 GB free)")
        print("  ✅ Disk space adequate")


def demo_checkpoint_simulation():
    """Demonstrate checkpoint functionality."""
    print("\n=== Checkpoint Simulation Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        recovery = SimpleRecoveryDemo()
        documents = [f"doc_{i:03d}" for i in range(10)]
        recovery.initialize_session(documents)
        
        # Process some documents
        for i in range(4):
            recovery.mark_processed(documents[i])
        
        # Simulate some failures
        recovery.handle_failure(FailureType.PROCESSING_ERROR, "Error 1", documents[4])
        recovery.handle_failure(FailureType.PROCESSING_ERROR, "Error 2", documents[5])
        
        # Create checkpoint
        checkpoint_data = {
            'timestamp': time.time(),
            'processed_docs': recovery.processed_docs.copy(),
            'failed_docs': recovery.failed_docs.copy(),
            'pending_docs': recovery.pending_docs.copy(),
            'current_mode': recovery.current_mode.value,
            'batch_size': recovery.batch_size,
            'error_counts': {k.value: v for k, v in recovery.error_counts.items()}
        }
        
        checkpoint_file = checkpoint_dir / "demo_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"Created checkpoint: {checkpoint_file}")
        print(f"Checkpoint size: {checkpoint_file.stat().st_size} bytes")
        print(f"Processed: {len(recovery.processed_docs)} documents")
        print(f"Failed: {len(recovery.failed_docs)} documents") 
        print(f"Pending: {len(recovery.pending_docs)} documents")
        
        # Simulate loading checkpoint
        print("\nSimulating system restart and checkpoint recovery...")
        
        new_recovery = SimpleRecoveryDemo()
        
        with open(checkpoint_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Restore state
        new_recovery.processed_docs = loaded_data['processed_docs']
        new_recovery.failed_docs = loaded_data['failed_docs']
        new_recovery.pending_docs = loaded_data['pending_docs']
        new_recovery.current_mode = DegradationMode(loaded_data['current_mode'])
        new_recovery.batch_size = loaded_data['batch_size']
        
        print("✅ Successfully restored from checkpoint")
        print(f"Restored state: {len(new_recovery.processed_docs)} processed, {len(new_recovery.pending_docs)} pending")


def main():
    """Run all demonstrations."""
    print("Advanced Recovery System - Simple Demonstration")
    print("=" * 60)
    
    demo_progressive_degradation()
    demo_batch_processing()
    demo_resource_monitoring() 
    demo_checkpoint_simulation()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully! 🎉")
    print("\nKey features demonstrated:")
    print("✅ Progressive degradation with multiple fallback modes")
    print("✅ Intelligent batch size adjustment based on failures")
    print("✅ Document prioritization in degraded modes")
    print("✅ Resource monitoring and pressure detection")
    print("✅ Checkpoint creation and recovery simulation")
    print("✅ Comprehensive error tracking and recovery strategies")


if __name__ == "__main__":
    main()