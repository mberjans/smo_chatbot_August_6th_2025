#!/usr/bin/env python3
"""
Demonstration of Quality Validation Performance Benchmarking Suite.

This script demonstrates how to use the QualityValidationBenchmarkSuite to benchmark
quality validation components including factual accuracy validation, relevance scoring,
and integrated quality workflows.

Usage:
    python demo_quality_benchmarks.py [--benchmark-name BENCHMARK] [--output-dir DIR]

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from quality_performance_benchmarks import (
        QualityValidationBenchmarkSuite,
        QualityValidationMetrics,
        QualityBenchmarkConfiguration,
        create_standard_quality_benchmarks
    )
    from ..api_metrics_logger import APIUsageMetricsLogger
except ImportError as e:
    logger.error(f"Failed to import quality benchmarking components: {e}")
    logger.info("Make sure you're running this from the correct directory and all dependencies are installed")
    exit(1)


async def run_demo_benchmarks(benchmark_name: str = None, output_dir: str = None):
    """Run demonstration quality validation benchmarks."""
    
    logger.info("Starting Quality Validation Benchmark Suite Demonstration")
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path("demo_quality_benchmark_results")
    
    output_path.mkdir(exist_ok=True)
    
    # Initialize API metrics logger
    api_logger = APIUsageMetricsLogger(
        log_to_file=True,
        log_file_path=output_path / "api_metrics.log"
    )
    
    # Create quality benchmark suite
    logger.info("Initializing Quality Validation Benchmark Suite...")
    suite = QualityValidationBenchmarkSuite(
        output_dir=output_path,
        api_metrics_logger=api_logger
    )
    
    # Define custom test data for more realistic benchmarks
    custom_test_data = {
        'queries': [
            "What metabolites are associated with Type 2 diabetes?",
            "How does metabolomics contribute to cancer research?",
            "What are the key challenges in clinical metabolomics standardization?",
            "Explain the role of metabolomics in personalized medicine.",
            "What metabolic pathways are disrupted in cardiovascular disease?",
            "How can metabolomics improve drug discovery and development?",
            "What are the latest advances in metabolomics technology?",
            "How does metabolomics help in understanding aging processes?",
            "What is the significance of metabolomics in pediatric medicine?",
            "How are metabolomics biomarkers validated for clinical use?"
        ],
        'responses': [
            "Type 2 diabetes is associated with altered levels of several metabolites including elevated glucose, branched-chain amino acids (leucine, isoleucine, valine), and various lipid species. Studies have shown increased levels of palmitate and decreased levels of glycine in diabetic patients.",
            "Metabolomics contributes significantly to cancer research by identifying metabolic signatures that distinguish cancer cells from normal cells. Key metabolites like lactate, glutamine, and various fatty acids show altered levels in cancer, enabling biomarker discovery and therapeutic target identification.",
            "The key challenges in clinical metabolomics standardization include variability in sample collection and storage protocols, lack of standardized analytical methods, insufficient quality control measures, and the need for harmonized data processing and interpretation workflows.",
            "Metabolomics enables personalized medicine by providing individual metabolic profiles that reflect genetic variations, environmental exposures, and disease states. This allows for tailored treatment strategies, prediction of drug responses, and monitoring of therapeutic efficacy.",
            "Cardiovascular disease involves disruption of multiple metabolic pathways including fatty acid oxidation, glucose metabolism, amino acid catabolism, and energy production pathways. Key disrupted metabolites include increased TMAO, altered sphingolipids, and modified amino acid profiles.",
            "Metabolomics improves drug discovery by identifying novel therapeutic targets, predicting drug toxicity, monitoring drug efficacy, and understanding mechanism of action. It enables phenotypic drug screening and helps in patient stratification for clinical trials.",
            "Recent advances in metabolomics technology include improved mass spectrometry sensitivity, enhanced chromatographic separation techniques, better data processing algorithms, integration with other omics platforms, and development of targeted metabolomics panels.",
            "Metabolomics helps understand aging by identifying age-related metabolic changes, studying cellular senescence markers, investigating mitochondrial dysfunction, and exploring the metabolic basis of age-related diseases. Key aging metabolites include NAD+ precursors and oxidative stress markers.",
            "In pediatric medicine, metabolomics is significant for understanding developmental metabolism, identifying inborn errors of metabolism, monitoring growth and development, studying pediatric disease mechanisms, and developing age-appropriate therapeutic interventions.",
            "Metabolomics biomarkers are validated through analytical validation (precision, accuracy, stability), clinical validation (sensitivity, specificity), and regulatory validation. This includes multi-site studies, longitudinal validation, and assessment of clinical utility and cost-effectiveness."
        ]
    }
    
    # Determine which benchmarks to run
    if benchmark_name:
        benchmark_names = [benchmark_name]
        logger.info(f"Running specific benchmark: {benchmark_name}")
    else:
        benchmark_names = None  # Run all benchmarks
        logger.info("Running all available quality validation benchmarks")
    
    # Run the benchmark suite
    logger.info("Executing quality validation benchmarks...")
    
    try:
        results = await suite.run_quality_benchmark_suite(
            benchmark_names=benchmark_names,
            custom_test_data=custom_test_data
        )
        
        logger.info("Quality validation benchmarks completed successfully!")
        
        # Display summary results
        display_benchmark_summary(results)
        
        # Save demonstration results
        demo_results_path = output_path / f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(demo_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Demonstration results saved to: {demo_results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise


def display_benchmark_summary(results: dict):
    """Display a summary of benchmark results."""
    
    print("\n" + "="*80)
    print("QUALITY VALIDATION BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Overall summary
    summary = results.get('suite_execution_summary', {})
    print(f"\nExecution Summary:")
    print(f"  Total Benchmarks: {summary.get('total_quality_benchmarks', 0)}")
    print(f"  Passed: {summary.get('passed_benchmarks', 0)}")
    print(f"  Failed: {summary.get('failed_benchmarks', 0)}")
    print(f"  Success Rate: {summary.get('success_rate_percent', 0):.1f}%")
    
    # Quality statistics
    stats = results.get('overall_quality_statistics', {})
    if stats:
        print(f"\nQuality Performance Statistics:")
        print(f"  Total Operations: {stats.get('total_quality_operations', 0):,}")
        print(f"  Claims Extracted: {stats.get('total_claims_extracted', 0):,}")
        print(f"  Claims Validated: {stats.get('total_claims_validated', 0):,}")
        print(f"  Avg Efficiency Score: {stats.get('avg_quality_efficiency_score', 0):.1f}%")
        print(f"  Avg Extraction Time: {stats.get('avg_claim_extraction_time_ms', 0):.1f} ms")
        print(f"  Avg Validation Time: {stats.get('avg_validation_time_ms', 0):.1f} ms")
        print(f"  Validation Accuracy: {stats.get('avg_validation_accuracy_rate', 0):.1f}%")
    
    # Individual benchmark results
    benchmark_results = results.get('quality_benchmark_results', {})
    if benchmark_results:
        print(f"\nIndividual Benchmark Results:")
        for benchmark_name, result in benchmark_results.items():
            status = "✓ PASSED" if result.get('passed', False) else "✗ FAILED"
            print(f"  {benchmark_name}: {status}")
            
            # Show scenario details
            for scenario in result.get('scenario_results', []):
                scenario_status = "✓" if scenario.get('passed', False) else "✗"
                efficiency = scenario.get('quality_efficiency_score', 0)
                print(f"    {scenario_status} {scenario.get('scenario_name', 'Unknown')}: {efficiency:.1f}% efficiency")
    
    # Recommendations
    recommendations = results.get('quality_recommendations', [])
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "="*80)


def main():
    """Main demonstration function."""
    
    parser = argparse.ArgumentParser(
        description="Demonstrate Quality Validation Performance Benchmarking"
    )
    parser.add_argument(
        '--benchmark-name',
        type=str,
        help='Specific benchmark to run (run all if not specified)',
        choices=[
            'factual_accuracy_validation_benchmark',
            'relevance_scoring_benchmark', 
            'integrated_quality_workflow_benchmark',
            'quality_validation_load_test',
            'quality_validation_scalability_benchmark'
        ]
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for benchmark results',
        default='demo_quality_benchmark_results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run the demonstration
        results = asyncio.run(run_demo_benchmarks(
            benchmark_name=args.benchmark_name,
            output_dir=args.output_dir
        ))
        
        print(f"\nDemonstration completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())