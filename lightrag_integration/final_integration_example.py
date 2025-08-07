#!/usr/bin/env python3
"""
Final Integration Example: Unified Progress Tracking with ClinicalMetabolomicsRAG

This example demonstrates how to integrate the unified progress tracking system
with the actual ClinicalMetabolomicsRAG initialize_knowledge_base method.

Usage:
    python final_integration_example.py
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging for the example
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("integration_example")

# Import the unified progress tracking components
try:
    from .unified_progress_tracker import KnowledgeBasePhase
    from .progress_integration import (
        create_unified_progress_tracker,
        setup_progress_integration,
        ProgressCallbackBuilder
    )
    from .progress_config import ProgressTrackingConfig
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from lightrag_integration.unified_progress_tracker import KnowledgeBasePhase
    from lightrag_integration.progress_integration import (
        create_unified_progress_tracker,
        setup_progress_integration,
        ProgressCallbackBuilder
    )
    from lightrag_integration.progress_config import ProgressTrackingConfig


class EnhancedClinicalMetabolomicsRAG:
    """
    Enhanced version of ClinicalMetabolomicsRAG with unified progress tracking.
    
    This example shows how to integrate the unified progress tracking system
    into the existing initialize_knowledge_base method.
    """
    
    def __init__(self):
        """Initialize the enhanced RAG system."""
        self.logger = logger
        self.pdf_processor = None
        self.is_initialized = True
        self._knowledge_base_initialized = False
        self._unified_progress_tracker = None
    
    async def initialize_knowledge_base(self, 
                                      papers_dir: str = "papers/",
                                      progress_config: Optional[ProgressTrackingConfig] = None,
                                      batch_size: int = 10,
                                      max_memory_mb: int = 2048,
                                      enable_batch_processing: bool = True,
                                      force_reinitialize: bool = False,
                                      enable_progress_callbacks: bool = True) -> Dict[str, Any]:
        """
        Enhanced initialize_knowledge_base with unified progress tracking.
        
        This method demonstrates the integration points for unified progress tracking
        throughout the knowledge base initialization process.
        """
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized. Call constructor first.")
        
        # Convert papers_dir to Path object
        papers_path = Path(papers_dir)
        
        # Initialize result dictionary
        import time
        start_time = time.time()
        result = {
            'success': False,
            'documents_processed': 0,
            'documents_failed': 0,
            'total_documents': 0,
            'processing_time': 0.0,
            'storage_created': [],
            'errors': [],
            'metadata': {
                'papers_dir': str(papers_path),
                'batch_size': batch_size,
                'force_reinitialize': force_reinitialize
            }
        }
        
        # ==== INTEGRATION POINT 1: Setup Unified Progress Tracking ====
        unified_tracker = None
        if progress_config is None:
            # Create default configuration with unified progress tracking enabled
            progress_config = ProgressTrackingConfig(
                enable_unified_progress_tracking=True,
                enable_phase_based_progress=True,
                save_unified_progress_to_file=True,
                enable_progress_callbacks=enable_progress_callbacks,
                log_processing_stats=True
            )
        
        if progress_config.enable_unified_progress_tracking:
            # Setup comprehensive progress tracking
            if enable_progress_callbacks:
                # Create rich callback with console output, logging, and file output
                callback = (ProgressCallbackBuilder()
                           .with_console_output(update_interval=2.0, show_details=True)
                           .with_logging(self.logger, log_level=logging.INFO, log_interval=5.0)
                           .with_file_output(Path("logs/kb_init_progress.json"), update_interval=10.0)
                           .build())
            else:
                callback = None
            
            unified_tracker = setup_progress_integration(
                rag_instance=self,
                progress_config=progress_config,
                enable_callbacks=callback is not None,
                callback_interval=2.0
            )
            
            if callback:
                unified_tracker.progress_callback = callback
            
            # Estimate document count for progress tracking
            if papers_path.exists() and papers_path.is_dir():
                pdf_files = list(papers_path.glob("*.pdf"))
                estimated_documents = len(pdf_files)
                result['total_documents'] = estimated_documents
            else:
                estimated_documents = 0
            
            # Start unified progress tracking
            unified_tracker.start_initialization(total_documents=estimated_documents)
            self.logger.info(f"Started unified progress tracking for {estimated_documents} documents")
        
        try:
            # Check if already initialized and not forcing reinitialize
            if self._knowledge_base_initialized and not force_reinitialize:
                self.logger.info("Knowledge base already initialized, skipping")
                if unified_tracker:
                    # Complete all phases instantly for already initialized case
                    for phase in KnowledgeBasePhase:
                        unified_tracker.start_phase(phase, f"Already completed: {phase.value}")
                        unified_tracker.complete_phase(phase, "Previously initialized")
                
                result.update({
                    'success': True,
                    'already_initialized': True,
                    'processing_time': time.time() - start_time
                })
                return result
            
            # ==== INTEGRATION POINT 2: Phase 1 - Storage Initialization ====
            if unified_tracker:
                unified_tracker.start_phase(
                    KnowledgeBasePhase.STORAGE_INIT,
                    "Initializing LightRAG storage systems",
                    estimated_duration=5.0
                )
            
            self.logger.info("Initializing LightRAG storage systems")
            storage_paths = await self._initialize_lightrag_storage(unified_tracker)
            result['storage_created'] = [str(path) for path in storage_paths]
            
            if unified_tracker:
                unified_tracker.complete_phase(
                    KnowledgeBasePhase.STORAGE_INIT,
                    f"Storage systems initialized ({len(storage_paths)} paths created)"
                )
            
            # ==== INTEGRATION POINT 3: Phase 2 - PDF Processing ====
            if unified_tracker:
                unified_tracker.start_phase(
                    KnowledgeBasePhase.PDF_PROCESSING,
                    "Processing PDF documents",
                    estimated_duration=30.0 * estimated_documents
                )
            
            self.logger.info(f"Processing PDF documents from {papers_path}")
            
            # Simulate PDF processing with progress updates
            processed_documents = await self._process_pdf_documents(
                papers_path, unified_tracker, batch_size, max_memory_mb, enable_batch_processing
            )
            
            result['total_documents'] = len(processed_documents)
            
            if unified_tracker:
                unified_tracker.complete_phase(
                    KnowledgeBasePhase.PDF_PROCESSING,
                    f"Processed {len(processed_documents)} PDF documents"
                )
            
            # ==== INTEGRATION POINT 4: Phase 3 - Document Ingestion ====
            if processed_documents:
                if unified_tracker:
                    unified_tracker.start_phase(
                        KnowledgeBasePhase.DOCUMENT_INGESTION,
                        "Ingesting documents into knowledge graph",
                        estimated_duration=10.0 * len(processed_documents)
                    )
                
                self.logger.info("Ingesting documents into LightRAG knowledge graph")
                
                # Simulate document ingestion with batch processing
                successful_ingestions = await self._ingest_documents(
                    processed_documents, unified_tracker, batch_size
                )
                
                result['documents_processed'] = successful_ingestions
                result['documents_failed'] = len(processed_documents) - successful_ingestions
                
                if unified_tracker:
                    unified_tracker.update_document_counts(
                        processed=successful_ingestions,
                        failed=result['documents_failed']
                    )
                    unified_tracker.complete_phase(
                        KnowledgeBasePhase.DOCUMENT_INGESTION,
                        f"Ingested {successful_ingestions}/{len(processed_documents)} documents"
                    )
            else:
                self.logger.warning("No documents were successfully processed")
                result['errors'].append("No valid PDF documents found or processed")
                if unified_tracker:
                    unified_tracker.fail_phase(
                        KnowledgeBasePhase.PDF_PROCESSING,
                        "No documents could be processed"
                    )
            
            # ==== INTEGRATION POINT 5: Phase 4 - Finalization ====
            if unified_tracker:
                unified_tracker.start_phase(
                    KnowledgeBasePhase.FINALIZATION,
                    "Finalizing knowledge base",
                    estimated_duration=10.0
                )
            
            # Finalize knowledge base
            await self._finalize_knowledge_base(unified_tracker)
            
            # Mark knowledge base as initialized if we processed at least some documents
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            if result['documents_processed'] > 0:
                self._knowledge_base_initialized = True
                result['success'] = True
                
                if unified_tracker:
                    unified_tracker.complete_phase(
                        KnowledgeBasePhase.FINALIZATION,
                        "Knowledge base initialization completed successfully"
                    )
                
                self.logger.info(
                    f"Knowledge base initialization completed successfully: "
                    f"{result['documents_processed']}/{result['total_documents']} documents "
                    f"processed in {processing_time:.2f} seconds"
                )
            else:
                result['success'] = False
                error_msg = "Knowledge base initialization failed: no documents were successfully processed"
                self.logger.error(error_msg)
                result['errors'].append(error_msg)
                
                if unified_tracker:
                    unified_tracker.fail_phase(
                        KnowledgeBasePhase.FINALIZATION,
                        "No documents were successfully processed"
                    )
            
            # ==== INTEGRATION POINT 6: Final Progress Summary ====
            if unified_tracker:
                final_summary = unified_tracker.get_progress_summary()
                result['progress_summary'] = final_summary
                self.logger.info(f"Final Progress Summary: {final_summary}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['success'] = False
            
            error_msg = f"Knowledge base initialization failed: {e}"
            self.logger.error(error_msg)
            result['errors'].append(error_msg)
            
            # Handle progress tracking failure
            if unified_tracker and unified_tracker.state.current_phase:
                unified_tracker.fail_phase(unified_tracker.state.current_phase, str(e))
            
            # Re-raise for critical failures, but return result for partial failures
            if result['documents_processed'] == 0:
                raise RuntimeError(error_msg) from e
            
            return result
    
    async def _initialize_lightrag_storage(self, unified_tracker=None) -> list:
        """Mock storage initialization with progress updates."""
        storage_paths = []
        storage_dirs = ['vdb_chunks', 'vdb_entities', 'vdb_relationships']
        
        for i, dir_name in enumerate(storage_dirs):
            # Simulate storage creation time
            await asyncio.sleep(0.2)
            
            # Create mock directory path
            storage_path = Path(f"mock_storage/{dir_name}")
            storage_paths.append(storage_path)
            
            # Update progress if tracker available
            if unified_tracker:
                progress = (i + 1) / len(storage_dirs)
                unified_tracker.update_phase_progress(
                    KnowledgeBasePhase.STORAGE_INIT,
                    progress,
                    f"Created {dir_name} storage directory",
                    {
                        'storage_dir': dir_name,
                        'dirs_created': i + 1,
                        'total_dirs': len(storage_dirs)
                    }
                )
            
            self.logger.debug(f"Created storage directory: {storage_path}")
        
        return storage_paths
    
    async def _process_pdf_documents(self, papers_path, unified_tracker, batch_size, 
                                   max_memory_mb, enable_batch_processing):
        """Mock PDF processing with progress updates."""
        # Simulate finding PDF files
        if papers_path.exists():
            pdf_files = list(papers_path.glob("*.pdf"))
        else:
            # Create mock PDF files for demonstration
            pdf_files = [Path(f"mock_paper_{i:03d}.pdf") for i in range(8)]
        
        processed_documents = []
        
        for i, pdf_file in enumerate(pdf_files):
            # Simulate processing time
            await asyncio.sleep(0.3)
            
            # Simulate successful processing (90% success rate)
            if i % 10 != 9:  # 90% success
                processed_documents.append((pdf_file, {
                    'content': f"Mock content from {pdf_file.name}",
                    'metadata': {
                        'filename': pdf_file.name,
                        'pages': 10 + i,
                        'characters': 5000 + i * 100
                    }
                }))
                
                # Update progress
                if unified_tracker:
                    progress = (i + 1) / len(pdf_files)
                    unified_tracker.update_phase_progress(
                        KnowledgeBasePhase.PDF_PROCESSING,
                        progress,
                        f"Processed {pdf_file.name} ({i + 1}/{len(pdf_files)})",
                        {
                            'completed_files': len(processed_documents),
                            'failed_files': i + 1 - len(processed_documents),
                            'total_files': len(pdf_files),
                            'current_file': pdf_file.name,
                            'success_rate': (len(processed_documents) / (i + 1)) * 100
                        }
                    )
            else:
                # Simulate processing failure
                self.logger.warning(f"Failed to process {pdf_file.name}")
                
                if unified_tracker:
                    progress = (i + 1) / len(pdf_files)
                    unified_tracker.update_phase_progress(
                        KnowledgeBasePhase.PDF_PROCESSING,
                        progress,
                        f"Failed to process {pdf_file.name} ({i + 1}/{len(pdf_files)})",
                        {
                            'completed_files': len(processed_documents),
                            'failed_files': i + 1 - len(processed_documents),
                            'total_files': len(pdf_files),
                            'current_file': pdf_file.name,
                            'success_rate': (len(processed_documents) / (i + 1)) * 100,
                            'last_error': 'Simulated processing error'
                        }
                    )
        
        self.logger.info(f"PDF processing completed: {len(processed_documents)}/{len(pdf_files)} successful")
        return processed_documents
    
    async def _ingest_documents(self, processed_documents, unified_tracker, batch_size):
        """Mock document ingestion with progress updates."""
        successful_ingestions = 0
        
        # Process in batches
        for batch_start in range(0, len(processed_documents), batch_size):
            batch_end = min(batch_start + batch_size, len(processed_documents))
            batch = processed_documents[batch_start:batch_end]
            
            # Simulate ingestion time
            await asyncio.sleep(0.5)
            
            # Simulate batch ingestion (95% success rate per batch)
            batch_success = len(batch) if batch_start % 20 != 19 else len(batch) - 1
            successful_ingestions += batch_success
            
            # Update progress
            if unified_tracker:
                progress = batch_end / len(processed_documents)
                batch_num = batch_start // batch_size + 1
                total_batches = (len(processed_documents) + batch_size - 1) // batch_size
                
                unified_tracker.update_phase_progress(
                    KnowledgeBasePhase.DOCUMENT_INGESTION,
                    progress,
                    f"Ingested batch {batch_num}/{total_batches}",
                    {
                        'ingested_documents': successful_ingestions,
                        'total_documents': len(processed_documents),
                        'batch_number': batch_num,
                        'total_batches': total_batches,
                        'batch_size': len(batch),
                        'batch_success': batch_success,
                        'entities_extracted': successful_ingestions * 50,
                        'relationships_created': successful_ingestions * 25
                    }
                )
            
            self.logger.debug(f"Ingested batch {batch_num}: {batch_success}/{len(batch)} documents")
        
        self.logger.info(f"Document ingestion completed: {successful_ingestions}/{len(processed_documents)} successful")
        return successful_ingestions
    
    async def _finalize_knowledge_base(self, unified_tracker):
        """Mock finalization with progress updates."""
        finalization_tasks = [
            ('Optimizing entity indices', 0.3),
            ('Building relationship graph', 0.6),
            ('Validating knowledge base integrity', 0.9),
            ('Persisting final state', 1.0)
        ]
        
        for task_name, progress in finalization_tasks:
            await asyncio.sleep(0.2)
            
            if unified_tracker:
                unified_tracker.update_phase_progress(
                    KnowledgeBasePhase.FINALIZATION,
                    progress,
                    task_name,
                    {
                        'finalization_step': task_name,
                        'steps_completed': int(progress * len(finalization_tasks)),
                        'total_steps': len(finalization_tasks)
                    }
                )
            
            self.logger.debug(f"Finalization: {task_name}")
        
        self.logger.info("Knowledge base finalization completed")


async def run_integration_example():
    """Run the complete integration example."""
    print("🚀 Enhanced ClinicalMetabolomicsRAG with Unified Progress Tracking")
    print("=" * 70)
    
    # Create enhanced RAG system
    rag_system = EnhancedClinicalMetabolomicsRAG()
    
    # Create demo papers directory
    demo_papers_dir = Path("demo_papers")
    demo_papers_dir.mkdir(exist_ok=True)
    
    # Create some mock PDF files for demonstration
    for i in range(5):
        mock_pdf = demo_papers_dir / f"metabolomics_paper_{i:02d}.pdf"
        mock_pdf.touch()
    
    print(f"📁 Created demo papers directory with {len(list(demo_papers_dir.glob('*.pdf')))} PDF files")
    
    try:
        # Run knowledge base initialization with full progress tracking
        result = await rag_system.initialize_knowledge_base(
            papers_dir=str(demo_papers_dir),
            batch_size=3,
            enable_progress_callbacks=True,
            force_reinitialize=True
        )
        
        print("\n" + "=" * 70)
        print("📊 Knowledge Base Initialization Results")
        print("=" * 70)
        print(f"Success: {result['success']}")
        print(f"Documents Processed: {result['documents_processed']}/{result['total_documents']}")
        print(f"Documents Failed: {result['documents_failed']}")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print(f"Storage Paths Created: {len(result['storage_created'])}")
        
        if result.get('progress_summary'):
            print(f"Final Progress Summary: {result['progress_summary']}")
        
        if result.get('errors'):
            print(f"Errors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"  - {error}")
        
        print("\n✅ Integration example completed successfully!")
        print("Check the logs/ directory for generated progress files:")
        print("  - logs/knowledge_base_progress.json (unified progress state)")
        print("  - logs/kb_init_progress.json (detailed progress history)")
        
    except Exception as e:
        print(f"\n❌ Integration example failed: {e}")
        logger.exception("Integration example failed")
    
    finally:
        # Cleanup demo files
        import shutil
        try:
            shutil.rmtree(demo_papers_dir)
            print(f"\n🧹 Cleaned up demo directory: {demo_papers_dir}")
        except Exception as e:
            print(f"⚠️  Could not clean up demo directory: {e}")


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the integration example
    asyncio.run(run_integration_example())