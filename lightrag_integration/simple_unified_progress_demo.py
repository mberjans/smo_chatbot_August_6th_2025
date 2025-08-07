#!/usr/bin/env python3
"""
Simple Unified Progress Tracking Demo for Clinical Metabolomics Oracle

A straightforward example demonstrating unified progress tracking integration
with the ClinicalMetabolomicsRAG system.

Usage:
    python simple_unified_progress_demo.py

This demo shows:
    - Basic unified progress tracking setup
    - Console progress updates
    - Phase-based progress monitoring
    - Integration with existing PDF processing
"""

import asyncio
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def run_simple_demo():
    """Run a simple demonstration of unified progress tracking."""
    
    print("🚀 Simple Unified Progress Tracking Demo")
    print("=" * 50)
    
    try:
        # Import the required modules
        from clinical_metabolomics_rag import ClinicalMetabolomicsRAG
        from progress_config import ProgressTrackingConfig
        
        # Create simple progress configuration
        progress_config = ProgressTrackingConfig(
            enable_unified_progress_tracking=True,
            enable_phase_based_progress=True,
            enable_progress_tracking=True,
            log_progress_interval=1,
            save_unified_progress_to_file=True
        )
        
        print("✓ Progress configuration created")
        
        # Check for papers directory
        papers_dir = Path("papers")
        if not papers_dir.exists():
            print(f"📁 Creating papers directory: {papers_dir}")
            papers_dir.mkdir(exist_ok=True)
            print("⚠️  No PDF files found. Please add PDF files to the papers/ directory.")
            return
        
        pdf_files = list(papers_dir.glob("*.pdf"))
        if not pdf_files:
            print("⚠️  No PDF files found in papers/ directory.")
            print("   Please add some PDF files to see the progress tracking in action.")
            return
        
        print(f"📁 Found {len(pdf_files)} PDF files in papers/ directory")
        
        # Initialize RAG system
        print("🔧 Initializing Clinical Metabolomics RAG system...")
        rag_system = ClinicalMetabolomicsRAG()
        print("✓ RAG system initialized")
        
        # Run knowledge base initialization with unified progress tracking
        print("\n🚀 Starting knowledge base initialization...")
        print("   (Watch for real-time progress updates below)")
        print("-" * 50)
        
        result = await rag_system.initialize_knowledge_base(
            papers_dir=papers_dir,
            progress_config=progress_config,
            batch_size=3,
            enable_unified_progress_tracking=True,
            progress_callback=None,  # Will use console output by default
            force_reinitialize=True
        )
        
        print("\n" + "-" * 50)
        print("✅ Knowledge base initialization completed!")
        
        # Display results
        print(f"\n📊 Results:")
        print(f"   Success: {'✅ Yes' if result.get('success') else '❌ No'}")
        print(f"   Documents Processed: {result.get('documents_processed', 0)}")
        print(f"   Documents Failed: {result.get('documents_failed', 0)}")
        print(f"   Total Documents: {result.get('total_documents', 0)}")
        print(f"   Processing Time: {result.get('processing_time', 0):.2f} seconds")
        
        # Display unified progress results if available
        unified_progress = result.get('unified_progress', {})
        if unified_progress.get('enabled'):
            print(f"\n📈 Unified Progress Tracking:")
            print(f"   Final Progress: {unified_progress.get('final_state', {}).get('overall_progress', 0):.1%}")
            print(f"   Summary: {unified_progress.get('summary', 'N/A')}")
            
            # Show phase completion status
            phase_info = unified_progress.get('final_state', {}).get('phase_info', {})
            completed_phases = [phase for phase, info in phase_info.items() if info.get('is_completed')]
            failed_phases = [phase for phase, info in phase_info.items() if info.get('is_failed')]
            
            if completed_phases:
                print(f"   Completed Phases: {', '.join(completed_phases)}")
            if failed_phases:
                print(f"   Failed Phases: {', '.join(failed_phases)}")
        
        # Check for generated progress files
        print(f"\n📄 Generated Files:")
        progress_files = [
            "logs/knowledge_base_progress.json",
            "logs/processing_progress.json",
            "logs/lightrag_integration.log"
        ]
        
        for file_path in progress_files:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                print(f"   ✓ {file_path} ({size} bytes)")
            else:
                print(f"   - {file_path} (not created)")
        
        print(f"\n🎉 Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure you're running from the lightrag_integration directory.")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logging.error(f"Demo failed: {e}", exc_info=True)

def main():
    """Main function to run the demo."""
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    try:
        asyncio.run(run_simple_demo())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Failed to run demo: {e}")

if __name__ == "__main__":
    main()