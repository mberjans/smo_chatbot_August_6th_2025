#!/usr/bin/env python3
"""
Quick launcher script for the BiomedicalPDFProcessor benchmark.

This script provides an easy way to run the comprehensive PDF processing benchmark
with sensible defaults and clear output formatting.

Usage:
    python run_pdf_benchmark.py
    python run_pdf_benchmark.py --help

Author: Clinical Metabolomics Oracle System
Date: August 6, 2025
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from lightrag_integration.benchmark_pdf_processing import PDFProcessingBenchmark


def print_banner():
    """Print the benchmark banner."""
    print("=" * 80)
    print("BIOMEDICAL PDF PROCESSOR - PERFORMANCE BENCHMARK")
    print("=" * 80)
    print("Clinical Metabolomics Oracle - LightRAG Integration")
    print("Task: CMO-LIGHTRAG-003-T10 - Performance benchmark with 5+ biomedical PDFs")
    print()


def print_usage_info():
    """Print usage information and recommendations."""
    print("USAGE INFORMATION:")
    print("-" * 40)
    print("• Place PDF files in the 'papers/' directory")
    print("• Results will be saved to 'benchmark_results/' directory")
    print("• For comprehensive testing, use 5+ diverse biomedical PDFs")
    print()
    
    print("RECOMMENDED PDF SOURCES:")
    print("-" * 40)
    print("• PubMed Central (PMC): https://www.ncbi.nlm.nih.gov/pmc/")
    print("• bioRxiv preprints: https://www.biorxiv.org/")
    print("• medRxiv preprints: https://www.medrxiv.org/")
    print("• PLOS journals: https://plos.org/")
    print("• Nature journals: https://www.nature.com/")
    print("• Clinical metabolomics journals from major publishers")
    print()


async def main():
    """Main function to run the benchmark."""
    print_banner()
    
    # Check if papers directory exists and has PDFs
    papers_dir = Path("papers")
    if not papers_dir.exists():
        print("❌ ERROR: 'papers/' directory not found!")
        print("   Please create the directory and add PDF files to benchmark.")
        return
    
    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ ERROR: No PDF files found in 'papers/' directory!")
        print()
        print_usage_info()
        return
    
    # Show current status
    print(f"📁 Found {len(pdf_files)} PDF file(s) in papers/ directory:")
    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / 1024 / 1024
        print(f"   • {pdf_file.name} ({size_mb:.2f} MB)")
    print()
    
    # Recommendation for more files
    if len(pdf_files) < 5:
        print("⚠️  RECOMMENDATION:")
        print(f"   Currently {len(pdf_files)} PDF(s) found. For comprehensive benchmarking,")
        print("   consider adding more diverse biomedical PDFs (target: 5+ files).")
        print()
    
    # Run the benchmark
    print("🚀 Starting comprehensive benchmark...")
    print("   This will test processing time, memory usage, quality, and error handling.")
    print()
    
    try:
        # Create benchmark instance
        benchmark = PDFProcessingBenchmark(
            papers_dir="papers/",
            output_dir="benchmark_results/",
            verbose=True
        )
        
        # Run comprehensive benchmark
        results = await benchmark.run_comprehensive_benchmark()
        
        # Show completion summary
        print("\n" + "=" * 80)
        print("✅ BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        summary = results.get('summary', {})
        print(f"📊 Results Summary:")
        print(f"   • Files processed: {summary.get('files_valid', 0)}/{summary.get('files_discovered', 0)}")
        
        perf_summary = summary.get('performance_summary', {})
        if perf_summary:
            print(f"   • Average processing time: {perf_summary.get('average_processing_time', 'N/A')} seconds")
            print(f"   • Fastest processing: {perf_summary.get('fastest_processing_time', 'N/A')} seconds")
            print(f"   • Slowest processing: {perf_summary.get('slowest_processing_time', 'N/A')} seconds")
        
        # Show recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n📋 Key Recommendations ({len(recommendations)} items):")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                priority = rec.get('priority', 'medium').upper()
                print(f"   {i}. [{priority}] {rec.get('recommendation', '')}")
        
        # Show where results are saved
        print(f"\n📁 Detailed results saved to:")
        results_dir = Path("benchmark_results")
        if results_dir.exists():
            json_files = list(results_dir.glob("benchmark_results_*.json"))
            report_files = list(results_dir.glob("benchmark_report_*.txt"))
            
            if json_files:
                latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
                print(f"   • JSON data: {latest_json}")
            
            if report_files:
                latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
                print(f"   • Human report: {latest_report}")
                
                # Show how to view the report
                print(f"\n📖 To view the detailed report:")
                print(f"   cat '{latest_report}'")
        
        print("\n" + "=" * 80)
        
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user (Ctrl+C)")
        print("   Partial results may be available in benchmark_results/")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        print("   Check the log files in benchmark_results/ for detailed error information")
        import traceback
        print(f"\nTraceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    # Check for help argument
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_banner()
        print_usage_info()
        print("COMMAND LINE OPTIONS:")
        print("-" * 40)
        print("  python run_pdf_benchmark.py          Run benchmark with default settings")
        print("  python run_pdf_benchmark.py --help   Show this help message")
        print()
        sys.exit(0)
    
    # Run the main benchmark
    asyncio.run(main())