"""
Example Usage Patterns for Unified Progress Tracking System.

This module demonstrates various ways to use the unified progress tracking system
for knowledge base construction, including integration patterns, callback configurations,
and monitoring setups.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any

# Import the unified progress tracking components
from ..unified_progress_tracker import (
    KnowledgeBaseProgressTracker,
    KnowledgeBasePhase,
    PhaseWeights,
    UnifiedProgressCallback
)
from ..progress_integration import (
    create_unified_progress_tracker,
    setup_progress_integration,
    ProgressCallbackBuilder,
    ConsoleProgressCallback,
    estimate_phase_durations
)
from ..progress_config import ProgressTrackingConfig


# Example 1: Basic Console Progress Tracking
async def example_basic_console_tracking():
    """Demonstrate basic console-based progress tracking."""
    print("=== Example 1: Basic Console Progress Tracking ===")
    
    # Create progress tracker with console output
    progress_tracker = create_unified_progress_tracker(
        enable_console_output=True,
        console_update_interval=1.0  # Update every second for demo
    )
    
    # Simulate knowledge base initialization
    total_docs = 10
    progress_tracker.start_initialization(total_documents=total_docs)
    
    # Phase 1: Storage Initialization
    progress_tracker.start_phase(
        KnowledgeBasePhase.STORAGE_INIT,
        "Initializing storage directories"
    )
    
    # Simulate some work
    for i in range(3):
        await asyncio.sleep(0.5)
        progress_tracker.update_phase_progress(
            KnowledgeBasePhase.STORAGE_INIT,
            (i + 1) / 3,
            f"Created storage directory {i + 1}/3"
        )
    
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
        await asyncio.sleep(0.3)  # Simulate PDF processing time
        progress_tracker.update_phase_progress(
            KnowledgeBasePhase.PDF_PROCESSING,
            (i + 1) / total_docs,
            f"Processing document {i + 1}/{total_docs}",
            {
                'completed_files': i + 1,
                'total_files': total_docs,
                'current_file': f"document_{i + 1}.pdf"
            }
        )
        progress_tracker.update_document_counts(processed=1)
    
    progress_tracker.complete_phase(
        KnowledgeBasePhase.PDF_PROCESSING,
        f"Processed {total_docs} documents successfully"
    )
    
    # Phase 3: Document Ingestion
    progress_tracker.start_phase(
        KnowledgeBasePhase.DOCUMENT_INGESTION,
        "Ingesting documents into knowledge graph"
    )
    
    batch_size = 3
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_docs = batch_end - batch_start
        
        await asyncio.sleep(0.8)  # Simulate ingestion time
        progress_tracker.update_phase_progress(
            KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_end / total_docs,
            f"Ingested batch {batch_start // batch_size + 1}",
            {
                'ingested_documents': batch_end,
                'total_documents': total_docs,
                'batch_size': batch_docs
            }
        )
    
    progress_tracker.complete_phase(
        KnowledgeBasePhase.DOCUMENT_INGESTION,
        f"Ingested {total_docs} documents into knowledge graph"
    )
    
    # Phase 4: Finalization
    progress_tracker.start_phase(
        KnowledgeBasePhase.FINALIZATION,
        "Finalizing knowledge base"
    )
    
    await asyncio.sleep(0.5)
    progress_tracker.update_phase_progress(
        KnowledgeBasePhase.FINALIZATION,
        0.5,
        "Optimizing indices"
    )
    
    await asyncio.sleep(0.5)
    progress_tracker.complete_phase(
        KnowledgeBasePhase.FINALIZATION,
        "Knowledge base initialization completed"
    )
    
    # Display final summary
    print(f"\nFinal Summary: {progress_tracker.get_progress_summary()}")
    print("=" * 60)


# Example 2: Custom Phase Weights and File Logging
async def example_custom_weights_and_logging():
    """Demonstrate custom phase weights and file logging."""
    print("\n=== Example 2: Custom Phase Weights and File Logging ===")
    
    # Create custom phase weights (emphasize PDF processing even more)
    custom_weights = PhaseWeights(
        storage_init=0.05,
        pdf_processing=0.70,  # 70% of total progress
        document_ingestion=0.20,
        finalization=0.05
    )
    
    # Create progress configuration with file logging
    progress_config = ProgressTrackingConfig(
        enable_unified_progress_tracking=True,
        enable_phase_based_progress=True,
        save_unified_progress_to_file=True,
        unified_progress_file_path=Path("logs/demo_progress.json"),
        log_processing_stats=True,
        progress_log_level="INFO"
    )
    
    # Create progress tracker with custom settings
    progress_tracker = create_unified_progress_tracker(
        progress_config=progress_config,
        phase_weights=custom_weights,
        enable_console_output=True,
        console_update_interval=0.5
    )
    
    # Simulate faster initialization with custom weights
    progress_tracker.start_initialization(total_documents=5)
    
    # Quick storage init (5% weight)
    progress_tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Quick storage setup")
    await asyncio.sleep(0.2)
    progress_tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage ready")
    
    # Longer PDF processing (70% weight)
    progress_tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Heavy PDF processing")
    for i in range(5):
        await asyncio.sleep(0.6)  # Longer processing time
        progress_tracker.update_phase_progress(
            KnowledgeBasePhase.PDF_PROCESSING,
            (i + 1) / 5,
            f"Complex processing of document {i + 1}",
            {'success_rate': 95.0 + i}
        )
    progress_tracker.complete_phase(KnowledgeBasePhase.PDF_PROCESSING, "Heavy processing done")
    
    # Medium ingestion (20% weight)
    progress_tracker.start_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Knowledge graph ingestion")
    await asyncio.sleep(1.0)
    progress_tracker.complete_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Ingestion complete")
    
    # Quick finalization (5% weight)
    progress_tracker.start_phase(KnowledgeBasePhase.FINALIZATION, "Final optimizations")
    await asyncio.sleep(0.2)
    progress_tracker.complete_phase(KnowledgeBasePhase.FINALIZATION, "All done!")
    
    print(f"\nCustom Weights Summary: {progress_tracker.get_progress_summary()}")
    print("Progress saved to: logs/demo_progress.json")
    print("=" * 60)


# Example 3: Advanced Callback Builder Usage
async def example_advanced_callback_builder():
    """Demonstrate advanced callback builder features."""
    print("\n=== Example 3: Advanced Callback Builder Usage ===")
    
    # Setup logging
    logger = logging.getLogger("progress_demo")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Custom metrics collector
    metrics_data = []
    def collect_metrics(data: Dict[str, Any]):
        metrics_data.append({
            'timestamp': data['timestamp'],
            'phase': data['current_phase'],
            'progress': data['overall_progress'],
            'message': data['status_message']
        })
    
    # Build composite callback with multiple features
    callback = (ProgressCallbackBuilder()
                .with_console_output(update_interval=0.5, show_details=True)
                .with_logging(logger, log_level=logging.INFO, log_interval=1.0)
                .with_file_output("logs/demo_callback_output.json", update_interval=2.0)
                .with_metrics_collection(collect_metrics)
                .build())
    
    # Create progress tracker with composite callback
    progress_tracker = create_unified_progress_tracker(
        progress_callback=callback,
        logger=logger
    )
    
    # Run simulation
    progress_tracker.start_initialization(total_documents=3)
    
    for phase in [KnowledgeBasePhase.STORAGE_INIT, 
                  KnowledgeBasePhase.PDF_PROCESSING,
                  KnowledgeBasePhase.DOCUMENT_INGESTION,
                  KnowledgeBasePhase.FINALIZATION]:
        
        phase_name = phase.value.replace('_', ' ').title()
        progress_tracker.start_phase(phase, f"Starting {phase_name}")
        
        # Simulate variable work duration
        steps = 3 if phase == KnowledgeBasePhase.PDF_PROCESSING else 2
        for i in range(steps):
            await asyncio.sleep(0.4)
            progress_tracker.update_phase_progress(
                phase,
                (i + 1) / steps,
                f"{phase_name} step {i + 1}/{steps}",
                {'step': i + 1, 'total_steps': steps}
            )
        
        progress_tracker.complete_phase(phase, f"{phase_name} completed")
    
    print(f"\nAdvanced Callback Summary: {progress_tracker.get_progress_summary()}")
    print(f"Collected {len(metrics_data)} metrics data points")
    print("Multiple output formats created in logs/ directory")
    print("=" * 60)


# Example 4: Error Handling and Recovery
async def example_error_handling():
    """Demonstrate error handling and recovery in progress tracking."""
    print("\n=== Example 4: Error Handling and Recovery ===")
    
    progress_tracker = create_unified_progress_tracker(
        enable_console_output=True,
        console_update_interval=0.5
    )
    
    progress_tracker.start_initialization(total_documents=4)
    
    # Successful storage init
    progress_tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Initializing storage")
    await asyncio.sleep(0.3)
    progress_tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage initialized")
    
    # PDF processing with some failures
    progress_tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Processing documents")
    
    for i in range(4):
        await asyncio.sleep(0.4)
        
        # Simulate failure on document 2
        if i == 2:
            progress_tracker.update_phase_progress(
                KnowledgeBasePhase.PDF_PROCESSING,
                (i + 1) / 4,
                f"Failed to process document {i + 1}",
                {
                    'completed_files': i,
                    'failed_files': 1,
                    'current_error': 'Corrupted PDF format'
                }
            )
            progress_tracker.update_document_counts(failed=1)
        else:
            progress_tracker.update_phase_progress(
                KnowledgeBasePhase.PDF_PROCESSING,
                (i + 1) / 4,
                f"Processed document {i + 1}",
                {
                    'completed_files': i + 1 if i != 2 else i,
                    'failed_files': 1 if i >= 2 else 0,
                    'success_rate': ((i + 1 if i != 2 else i) / (i + 1)) * 100
                }
            )
            if i != 2:
                progress_tracker.update_document_counts(processed=1)
    
    progress_tracker.complete_phase(
        KnowledgeBasePhase.PDF_PROCESSING, 
        "PDF processing completed with 1 failure"
    )
    
    # Continue despite partial failure
    progress_tracker.start_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Ingesting successful documents")
    await asyncio.sleep(0.8)
    progress_tracker.complete_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Ingested 3/4 documents")
    
    # Demonstrate phase failure
    progress_tracker.start_phase(KnowledgeBasePhase.FINALIZATION, "Final optimizations")
    await asyncio.sleep(0.3)
    progress_tracker.fail_phase(
        KnowledgeBasePhase.FINALIZATION,
        "Index optimization failed due to insufficient memory"
    )
    
    print(f"\nError Handling Summary: {progress_tracker.get_progress_summary()}")
    
    # Show current state including errors
    current_state = progress_tracker.get_current_state()
    print(f"Errors encountered: {len(current_state.errors)}")
    for error in current_state.errors:
        print(f"  - {error}")
    
    print("=" * 60)


# Example 5: Integration with Mock ClinicalMetabolomicsRAG
class MockClinicalMetabolomicsRAG:
    """Mock ClinicalMetabolomicsRAG class for integration demonstration."""
    
    def __init__(self):
        self.logger = logging.getLogger("mock_rag")
        self.pdf_processor = None
        self._unified_progress_tracker = None
    
    async def initialize_knowledge_base_with_progress(self, 
                                                    papers_dir: str = "papers/",
                                                    total_documents: int = 5):
        """Mock implementation with integrated progress tracking."""
        # Setup progress tracking
        progress_config = ProgressTrackingConfig(
            enable_unified_progress_tracking=True,
            enable_phase_based_progress=True,
            save_unified_progress_to_file=True,
            enable_progress_callbacks=True
        )
        
        progress_tracker = setup_progress_integration(
            rag_instance=self,
            progress_config=progress_config,
            enable_callbacks=True,
            callback_interval=1.0
        )
        
        # Start initialization
        progress_tracker.start_initialization(total_documents=total_documents)
        
        try:
            # Phase 1: Storage Initialization
            progress_tracker.start_phase(
                KnowledgeBasePhase.STORAGE_INIT,
                "Initializing LightRAG storage systems"
            )
            await self._mock_storage_init(progress_tracker)
            progress_tracker.complete_phase(
                KnowledgeBasePhase.STORAGE_INIT,
                "Storage systems initialized"
            )
            
            # Phase 2: PDF Processing
            progress_tracker.start_phase(
                KnowledgeBasePhase.PDF_PROCESSING,
                "Processing PDF documents"
            )
            await self._mock_pdf_processing(progress_tracker, total_documents)
            progress_tracker.complete_phase(
                KnowledgeBasePhase.PDF_PROCESSING,
                f"Processed {total_documents} documents"
            )
            
            # Phase 3: Document Ingestion
            progress_tracker.start_phase(
                KnowledgeBasePhase.DOCUMENT_INGESTION,
                "Ingesting documents into knowledge graph"
            )
            await self._mock_document_ingestion(progress_tracker, total_documents)
            progress_tracker.complete_phase(
                KnowledgeBasePhase.DOCUMENT_INGESTION,
                "Document ingestion completed"
            )
            
            # Phase 4: Finalization
            progress_tracker.start_phase(
                KnowledgeBasePhase.FINALIZATION,
                "Finalizing knowledge base"
            )
            await self._mock_finalization(progress_tracker)
            progress_tracker.complete_phase(
                KnowledgeBasePhase.FINALIZATION,
                "Knowledge base initialization completed successfully"
            )
            
            return {
                'success': True,
                'documents_processed': total_documents,
                'progress_summary': progress_tracker.get_progress_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Knowledge base initialization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'progress_summary': progress_tracker.get_progress_summary()
            }
    
    async def _mock_storage_init(self, progress_tracker):
        """Mock storage initialization."""
        storage_dirs = ['vdb_chunks', 'vdb_entities', 'vdb_relationships']
        
        for i, dir_name in enumerate(storage_dirs):
            await asyncio.sleep(0.2)
            progress_tracker.update_phase_progress(
                KnowledgeBasePhase.STORAGE_INIT,
                (i + 1) / len(storage_dirs),
                f"Created {dir_name} directory",
                {'storage_dir': dir_name, 'dirs_created': i + 1}
            )
    
    async def _mock_pdf_processing(self, progress_tracker, total_docs):
        """Mock PDF processing."""
        for i in range(total_docs):
            await asyncio.sleep(0.5)  # Simulate processing time
            progress_tracker.update_phase_progress(
                KnowledgeBasePhase.PDF_PROCESSING,
                (i + 1) / total_docs,
                f"Extracted text from document {i + 1}",
                {
                    'completed_files': i + 1,
                    'total_files': total_docs,
                    'current_file': f"paper_{i + 1}.pdf",
                    'characters_extracted': (i + 1) * 5000,
                    'success_rate': ((i + 1) / (i + 1)) * 100
                }
            )
            progress_tracker.update_document_counts(processed=1)
    
    async def _mock_document_ingestion(self, progress_tracker, total_docs):
        """Mock document ingestion."""
        batch_size = 2
        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            
            await asyncio.sleep(0.7)  # Simulate ingestion time
            progress_tracker.update_phase_progress(
                KnowledgeBasePhase.DOCUMENT_INGESTION,
                batch_end / total_docs,
                f"Ingested batch {batch_start // batch_size + 1}",
                {
                    'ingested_documents': batch_end,
                    'total_documents': total_docs,
                    'batch_number': batch_start // batch_size + 1,
                    'entities_extracted': batch_end * 50,
                    'relationships_created': batch_end * 25
                }
            )
    
    async def _mock_finalization(self, progress_tracker):
        """Mock finalization."""
        tasks = ['Optimizing entity index', 'Building relationship graph', 'Validating integrity']
        
        for i, task in enumerate(tasks):
            await asyncio.sleep(0.3)
            progress_tracker.update_phase_progress(
                KnowledgeBasePhase.FINALIZATION,
                (i + 1) / len(tasks),
                task,
                {'finalization_step': task, 'steps_completed': i + 1}
            )


async def example_integrated_rag_system():
    """Demonstrate integration with mock RAG system."""
    print("\n=== Example 5: Integration with Mock RAG System ===")
    
    # Create mock RAG instance
    rag_system = MockClinicalMetabolomicsRAG()
    
    # Run initialization with integrated progress tracking
    result = await rag_system.initialize_knowledge_base_with_progress(
        papers_dir="demo_papers/",
        total_documents=6
    )
    
    print(f"\nIntegrated RAG System Result:")
    print(f"Success: {result['success']}")
    print(f"Documents Processed: {result.get('documents_processed', 'N/A')}")
    print(f"Progress Summary: {result['progress_summary']}")
    print("=" * 60)


# Main demo runner
async def run_all_examples():
    """Run all progress tracking examples."""
    print("🚀 Unified Progress Tracking System Examples")
    print("=" * 60)
    
    # Run all examples
    await example_basic_console_tracking()
    await asyncio.sleep(1)  # Brief pause between examples
    
    await example_custom_weights_and_logging()
    await asyncio.sleep(1)
    
    await example_advanced_callback_builder()
    await asyncio.sleep(1)
    
    await example_error_handling()
    await asyncio.sleep(1)
    
    await example_integrated_rag_system()
    
    print("\n✅ All examples completed successfully!")
    print("Check the logs/ directory for generated progress files.")


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run examples
    asyncio.run(run_all_examples())