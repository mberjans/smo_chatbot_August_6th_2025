#!/usr/bin/env python3
"""
Unified Progress Tracking Integration Example for Clinical Metabolomics Oracle

This example demonstrates the complete integration of unified progress tracking
with the ClinicalMetabolomicsRAG initialize_knowledge_base method, showcasing
real-time progress updates, phase-based tracking, and comprehensive reporting.

Usage:
    python unified_progress_example.py

Features Demonstrated:
    - Phase-weighted progress tracking (Storage: 10%, PDF: 60%, Ingestion: 25%, Finalization: 5%)
    - Real-time console progress updates with progress bars
    - File-based progress persistence
    - Custom progress callbacks
    - Integration with existing PDF processor progress tracking
    - Comprehensive error handling and reporting
    - Final progress summary and metrics
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/unified_progress_demo.log')
    ]
)
logger = logging.getLogger("unified_progress_demo")

async def demonstrate_unified_progress_tracking():
    """
    Comprehensive demonstration of unified progress tracking integration.
    
    This function shows how to:
    1. Enable unified progress tracking with custom callbacks
    2. Monitor progress across all phases of knowledge base initialization
    3. Handle progress persistence and reporting
    4. Demonstrate both console and file-based progress tracking
    """
    
    print("🚀 Clinical Metabolomics Oracle - Unified Progress Tracking Demo")
    print("=" * 70)
    
    try:
        # Import the RAG system and progress components
        from clinical_metabolomics_rag import ClinicalMetabolomicsRAG
        from progress_config import ProgressTrackingConfig
        from progress_integration import ProgressCallbackBuilder
        from unified_progress_tracker import KnowledgeBasePhase
        
        # Create enhanced progress configuration
        progress_config = ProgressTrackingConfig(
            enable_unified_progress_tracking=True,
            enable_phase_based_progress=True,
            enable_progress_callbacks=True,
            save_unified_progress_to_file=True,
            unified_progress_file_path=Path("logs/unified_progress_demo.json"),
            enable_progress_tracking=True,
            log_progress_interval=1,  # Log every file for demo
            enable_timing_details=True,
            enable_memory_monitoring=True
        )
        
        # Create custom progress callback with multiple outputs
        callback_builder = ProgressCallbackBuilder()
        progress_callback = (callback_builder
            .with_console_output(update_interval=2.0, show_details=True)
            .with_logging(logger, logging.INFO, log_interval=5.0)
            .with_file_output(Path("logs/progress_updates.json"), update_interval=10.0)
            .with_custom_callback(create_demo_metrics_callback())
            .build())
        
        print(f"📋 Configuration:")
        print(f"   - Progress tracking: {'✓ Enabled' if progress_config.enable_unified_progress_tracking else '✗ Disabled'}")
        print(f"   - Phase-based progress: {'✓ Enabled' if progress_config.enable_phase_based_progress else '✗ Disabled'}")
        print(f"   - Progress callbacks: {'✓ Enabled' if progress_config.enable_progress_callbacks else '✗ Disabled'}")
        print(f"   - Progress persistence: {'✓ Enabled' if progress_config.save_unified_progress_to_file else '✗ Disabled'}")
        print()
        
        # Setup demo papers directory
        demo_papers_dir = Path("demo_papers")
        demo_papers_dir.mkdir(exist_ok=True)
        
        if not list(demo_papers_dir.glob("*.pdf")):
            print("📁 Setting up demo papers directory...")
            create_demo_pdf_files(demo_papers_dir)
        
        pdf_count = len(list(demo_papers_dir.glob("*.pdf")))
        print(f"📁 Demo papers directory ready: {pdf_count} PDF files")
        print()
        
        # Initialize Clinical Metabolomics RAG system
        print("🔧 Initializing Clinical Metabolomics RAG system...")
        rag_system = ClinicalMetabolomicsRAG()
        print("✓ RAG system initialized")
        print()
        
        # Run knowledge base initialization with unified progress tracking
        print("🚀 Starting knowledge base initialization with unified progress tracking...")
        print("-" * 70)
        
        start_time = time.time()
        
        try:
            result = await rag_system.initialize_knowledge_base(
                papers_dir=demo_papers_dir,
                progress_config=progress_config,
                batch_size=2,  # Small batches for demo
                enable_unified_progress_tracking=True,
                progress_callback=progress_callback,
                force_reinitialize=True
            )
            
            elapsed_time = time.time() - start_time
            
            print()
            print("-" * 70)
            print("✅ Knowledge base initialization completed!")
            print()
            
            # Display comprehensive results
            display_initialization_results(result, elapsed_time)
            
            # Display unified progress tracking results
            if result.get('unified_progress', {}).get('enabled'):
                display_unified_progress_results(result['unified_progress'])
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print()
            print("-" * 70)
            print(f"❌ Knowledge base initialization failed after {elapsed_time:.2f}s")
            print(f"Error: {e}")
            logger.error(f"Initialization failed: {e}", exc_info=True)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required modules are available.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)

def create_demo_metrics_callback():
    """Create a custom metrics collection callback for demonstration."""
    
    metrics_data = {
        'phase_timings': {},
        'progress_updates': [],
        'total_updates': 0
    }
    
    def metrics_callback(overall_progress: float,
                        current_phase: KnowledgeBasePhase,
                        phase_progress: float,
                        status_message: str,
                        phase_details: Dict[str, Any],
                        all_phases: Dict[KnowledgeBasePhase, Any]) -> None:
        """Custom metrics collection callback."""
        metrics_data['total_updates'] += 1
        
        # Track phase timings
        phase_name = current_phase.value
        if phase_name not in metrics_data['phase_timings']:
            metrics_data['phase_timings'][phase_name] = {
                'start_time': time.time(),
                'updates': 0
            }
        
        metrics_data['phase_timings'][phase_name]['updates'] += 1
        
        # Store progress update
        update_entry = {
            'timestamp': time.time(),
            'overall_progress': overall_progress,
            'current_phase': phase_name,
            'phase_progress': phase_progress,
            'status_message': status_message
        }
        metrics_data['progress_updates'].append(update_entry)
        
        # Display periodic metrics (every 10 updates)
        if metrics_data['total_updates'] % 10 == 0:
            print(f"\n📊 Metrics Update #{metrics_data['total_updates']}:")
            print(f"   Overall Progress: {overall_progress:.1%}")
            print(f"   Current Phase: {phase_name} ({phase_progress:.1%})")
            if status_message:
                print(f"   Status: {status_message}")
    
    return metrics_callback

def create_demo_pdf_files(demo_dir: Path):
    """Create demo PDF files for testing (placeholder files)."""
    import hashlib
    
    demo_files = [
        "metabolomics_study_2024.pdf",
        "clinical_biomarkers_analysis.pdf", 
        "mass_spectrometry_methods.pdf",
        "omics_data_integration.pdf"
    ]
    
    for filename in demo_files:
        file_path = demo_dir / filename
        
        # Create a simple text file with PDF extension for demo
        # In real usage, these would be actual PDF files
        demo_content = f"""Demo PDF: {filename}
        
This is a placeholder PDF file created for demonstration purposes.
In a real scenario, this would be an actual PDF document containing
scientific literature about clinical metabolomics.

File: {filename}
Created: {time.strftime('%Y-%m-%d %H:%M:%S')}
Checksum: {hashlib.md5(filename.encode()).hexdigest()}

Content includes:
- Introduction to metabolomics
- Clinical applications
- Analytical methods
- Case studies
- Future perspectives
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(demo_content)

def display_initialization_results(result: Dict[str, Any], elapsed_time: float):
    """Display comprehensive initialization results."""
    
    print("📊 INITIALIZATION RESULTS")
    print("=" * 50)
    
    # Basic results
    print(f"Success: {'✅ Yes' if result.get('success') else '❌ No'}")
    print(f"Total Time: {elapsed_time:.2f} seconds")
    print(f"Documents Processed: {result.get('documents_processed', 0)}")
    print(f"Documents Failed: {result.get('documents_failed', 0)}")
    print(f"Total Documents: {result.get('total_documents', 0)}")
    
    # Processing rates
    if result.get('total_documents', 0) > 0:
        success_rate = (result.get('documents_processed', 0) / result.get('total_documents', 1)) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if elapsed_time > 0:
            throughput = result.get('total_documents', 0) / elapsed_time
            print(f"Throughput: {throughput:.2f} docs/second")
    
    # Cost information
    cost_summary = result.get('cost_summary', {})
    if cost_summary.get('total_cost', 0) > 0:
        print(f"Total Cost: ${cost_summary['total_cost']:.4f}")
    
    # Storage information
    storage_created = result.get('storage_created', [])
    if storage_created:
        print(f"Storage Paths Created: {len(storage_created)}")
    
    # Errors
    errors = result.get('errors', [])
    if errors:
        print(f"Errors Encountered: {len(errors)}")
        for i, error in enumerate(errors[:3], 1):  # Show first 3 errors
            print(f"  {i}. {error[:100]}{'...' if len(error) > 100 else ''}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more errors")
    
    print()

def display_unified_progress_results(unified_progress: Dict[str, Any]):
    """Display unified progress tracking results."""
    
    print("📈 UNIFIED PROGRESS TRACKING RESULTS")
    print("=" * 50)
    
    final_state = unified_progress.get('final_state', {})
    
    print(f"Final Progress: {final_state.get('overall_progress', 0):.1%}")
    print(f"Total Elapsed Time: {final_state.get('elapsed_time', 0):.2f} seconds")
    
    # Phase breakdown
    phase_info = final_state.get('phase_info', {})
    if phase_info:
        print("\n📋 Phase Breakdown:")
        
        for phase_name, phase_data in phase_info.items():
            status = "✅ Completed" if phase_data.get('is_completed') else \
                    "❌ Failed" if phase_data.get('is_failed') else \
                    "🔄 In Progress" if phase_data.get('is_active') else \
                    "⏳ Pending"
            
            elapsed = phase_data.get('elapsed_time', 0)
            progress = phase_data.get('current_progress', 0)
            
            print(f"  {phase_name}: {status} ({progress:.1%}, {elapsed:.1f}s)")
            
            if phase_data.get('status_message'):
                print(f"    └─ {phase_data['status_message']}")
    
    # Document processing summary
    total_docs = final_state.get('total_documents', 0)
    processed_docs = final_state.get('processed_documents', 0)
    failed_docs = final_state.get('failed_documents', 0)
    
    if total_docs > 0:
        print(f"\n📄 Document Processing:")
        print(f"  Total: {total_docs}")
        print(f"  Processed: {processed_docs} ({(processed_docs/total_docs)*100:.1f}%)")
        print(f"  Failed: {failed_docs} ({(failed_docs/total_docs)*100:.1f}%)")
    
    # Summary
    summary = unified_progress.get('summary', '')
    if summary:
        print(f"\n📝 Summary: {summary}")
    
    print()

async def main():
    """Main demo function."""
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    try:
        await demonstrate_unified_progress_tracking()
        
        print("🎉 Demo completed successfully!")
        print("\nGenerated files:")
        
        log_files = [
            "logs/unified_progress_demo.log",
            "logs/unified_progress_demo.json", 
            "logs/progress_updates.json"
        ]
        
        for log_file in log_files:
            if Path(log_file).exists():
                size = Path(log_file).stat().st_size
                print(f"  - {log_file} ({size} bytes)")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())