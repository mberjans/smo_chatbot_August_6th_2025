#!/usr/bin/env python3
"""
Usage Example for Quality-Aware API Metrics Logger.

This example demonstrates how to integrate and use the QualityAwareAPIMetricsLogger
for tracking API costs and performance metrics specifically related to quality 
validation operations in the Clinical Metabolomics Oracle LightRAG system.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import time
import logging
from pathlib import Path
from quality_aware_metrics_logger import (
    QualityAwareAPIMetricsLogger,
    create_quality_aware_logger
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_quality_validation_workflow():
    """
    Example of using QualityAwareAPIMetricsLogger for a complete quality validation workflow.
    """
    
    print("Quality-Aware API Metrics Logger Usage Example")
    print("=" * 60)
    
    # Create quality-aware logger
    metrics_logger = create_quality_aware_logger()
    
    try:
        # Example 1: Track individual relevance scoring
        print("\n1. Tracking Relevance Scoring Operation...")
        with metrics_logger.track_quality_validation(
            operation_name="clinical_relevance_assessment",
            validation_type="relevance",
            quality_stage="scoring",
            assessment_method="semantic_analysis",
            model_name="gpt-4",
            research_category="metabolomics_clinical_analysis"
        ) as tracker:
            # Simulate API call for relevance scoring
            await asyncio.sleep(0.2)  # Simulate processing time
            
            # Record API usage
            tracker.set_tokens(prompt=180, completion=45)
            tracker.set_cost(total_cost_usd=0.008, quality_validation_cost_usd=0.008)
            
            # Record quality results
            tracker.set_quality_results(
                quality_score=87.5,
                relevance_score=87.5,
                confidence_score=92.0
            )
            
            # Record validation details
            tracker.set_validation_details(
                biomedical_terms_identified=15,
                validation_passed=True,
                validation_confidence=92.0
            )
            
            tracker.add_metadata("query_complexity", "high")
            tracker.add_metadata("domain_specificity", "metabolomics")
        
        print("   ✓ Relevance scoring tracked successfully")
        
        # Example 2: Track factual accuracy validation
        print("\n2. Tracking Factual Accuracy Validation...")
        with metrics_logger.track_quality_validation(
            operation_name="factual_accuracy_verification",
            validation_type="factual_accuracy",
            quality_stage="validation",
            assessment_method="document_verification",
            model_name="gpt-4",
            research_category="biomedical_fact_checking"
        ) as tracker:
            # Simulate claim extraction stage
            await asyncio.sleep(0.1)
            tracker.record_stage_completion("claim_extraction")
            
            # Simulate validation stage
            await asyncio.sleep(0.3)
            tracker.record_stage_completion("validation")
            
            # Record API usage
            tracker.set_tokens(prompt=250, completion=35)
            tracker.set_cost(total_cost_usd=0.012, quality_validation_cost_usd=0.010)
            
            # Record quality results
            tracker.set_quality_results(
                quality_score=82.3,
                factual_accuracy_score=82.3,
                confidence_score=88.5
            )
            
            # Record detailed validation metrics
            tracker.set_validation_details(
                claims_extracted=8,
                claims_validated=7,
                evidence_items_processed=25,
                validation_passed=True,
                validation_confidence=88.5
            )
            
            # Add quality flags and recommendations
            tracker.add_quality_flag("low_evidence_quality_claim_3")
            tracker.add_quality_recommendation("increase_evidence_threshold")
        
        print("   ✓ Factual accuracy validation tracked successfully")
        
        # Example 3: Track integrated quality workflow
        print("\n3. Tracking Integrated Quality Workflow...")
        with metrics_logger.track_integrated_quality_workflow(
            workflow_name="comprehensive_quality_assessment",
            components=["relevance", "factual_accuracy", "completeness", "clarity"],
            model_name="gpt-4",
            research_category="comprehensive_quality_validation"
        ) as tracker:
            # Simulate component processing
            await asyncio.sleep(0.5)
            
            # Set results for each component
            component_results = {
                'relevance': {'score': 87.5, 'cost': 0.008},
                'factual_accuracy': {'score': 82.3, 'cost': 0.010},
                'completeness': {'score': 79.1, 'cost': 0.006},
                'clarity': {'score': 85.7, 'cost': 0.004}
            }
            tracker.set_component_results(component_results)
            
            # Set overall workflow outcome
            tracker.set_workflow_outcome(
                overall_score=83.7,
                passed=True,
                confidence=89.2,
                integration_method='weighted_harmonic_mean'
            )
            
            # Record total API usage
            tracker.set_tokens(prompt=680, completion=120)
            tracker.set_cost(total_cost_usd=0.028)
            
            tracker.add_metadata("workflow_version", "v2.1")
            tracker.add_metadata("parallel_processing", True)
        
        print("   ✓ Integrated quality workflow tracked successfully")
        
        # Example 4: Log batch operation
        print("\n4. Logging Batch Quality Validation...")
        metrics_logger.log_quality_batch_operation(
            operation_name="batch_manuscript_validation",
            validation_type="comprehensive",
            batch_size=20,
            total_tokens=3500,
            total_cost=0.15,
            quality_validation_cost=0.12,
            processing_time_ms=4500,
            average_quality_score=81.2,
            success_count=20,
            validation_passed_count=18,
            error_count=0,
            research_category="manuscript_quality_assessment",
            metadata={
                "batch_type": "research_papers",
                "parallel_workers": 4,
                "retry_failed": True
            }
        )
        print("   ✓ Batch operation logged successfully")
        
        # Example 5: Get performance summary
        print("\n5. Retrieving Performance Summary...")
        performance_summary = metrics_logger.get_quality_performance_summary()
        
        print(f"\nSession Summary:")
        session_stats = performance_summary['quality_validation']['session_stats']
        print(f"   Total Quality Operations: {session_stats['total_quality_operations']}")
        print(f"   Total Quality Validation Cost: ${session_stats['quality_validation_cost']:.6f}")
        print(f"   Average Quality Score: {session_stats['average_quality_score']:.2f}")
        
        quality_summary = performance_summary['quality_validation'].get('quality_validation_summary', {})
        if quality_summary:
            print(f"\nValidation Type Performance:")
            for validation_type, stats in quality_summary.items():
                print(f"   {validation_type.title()}:")
                print(f"     Operations: {stats['total_operations']}")
                print(f"     Avg Quality: {stats['average_quality_score']:.1f}")
                print(f"     Success Rate: {stats['success_rate_percent']:.1f}%")
                print(f"     Total Cost: ${stats['total_cost_usd']:.6f}")
        
        # Example 6: Export comprehensive report
        print(f"\n6. Exporting Quality Metrics Report...")
        report_path = metrics_logger.export_quality_metrics_report(
            output_path=Path("detailed_quality_metrics_report.json"),
            format="json",
            include_raw_data=True
        )
        print(f"   ✓ Detailed report exported to: {report_path}")
        
        # Also export HTML report
        html_report_path = metrics_logger.export_quality_metrics_report(
            output_path=Path("quality_metrics_dashboard.html"),
            format="html",
            include_raw_data=False
        )
        print(f"   ✓ HTML dashboard exported to: {html_report_path}")
        
        print(f"\n{'='*60}")
        print("Quality-Aware API Metrics Logging Example Completed Successfully!")
        print("Key Benefits Demonstrated:")
        print("  • Detailed quality validation operation tracking")
        print("  • Cost attribution to specific quality stages") 
        print("  • Performance analysis across validation types")
        print("  • Comprehensive reporting and analysis capabilities")
        print("  • Integration with existing API usage tracking")
        
    except Exception as e:
        logger.error(f"Error in quality validation workflow: {e}")
        raise
    
    finally:
        # Clean shutdown
        metrics_logger.close()


def example_integration_with_existing_systems():
    """
    Example of integrating QualityAwareAPIMetricsLogger with existing systems.
    """
    
    print("\nIntegration with Existing Systems Example")
    print("-" * 50)
    
    # Example: Create logger with full system integration
    # (In real usage, these would be actual instances)
    try:
        from cost_persistence import CostPersistence
        from budget_manager import BudgetManager
        from research_categorizer import ResearchCategorizer
        from audit_trail import AuditTrail
        
        # Create integrated logger
        metrics_logger = QualityAwareAPIMetricsLogger(
            cost_persistence=CostPersistence(),
            budget_manager=BudgetManager(),
            research_categorizer=ResearchCategorizer(),
            audit_trail=AuditTrail()
        )
        
        print("✓ Successfully created fully integrated logger")
        
    except ImportError:
        # Fallback to basic logger
        metrics_logger = create_quality_aware_logger()
        print("✓ Created basic logger (some dependencies not available)")
    
    # Example usage with cost alerts
    with metrics_logger.track_quality_validation(
        operation_name="high_cost_validation",
        validation_type="comprehensive",
        quality_stage="multi_stage"
    ) as tracker:
        # Simulate expensive validation
        tracker.set_tokens(prompt=1000, completion=200)
        tracker.set_cost(total_cost_usd=0.050, quality_validation_cost_usd=0.045)
        tracker.set_quality_results(quality_score=95.2, confidence_score=97.8)
        tracker.set_validation_details(validation_passed=True)
    
    print("✓ High-cost validation tracked with budget integration")
    
    metrics_logger.close()


async def main():
    """Run all examples."""
    await example_quality_validation_workflow()
    example_integration_with_existing_systems()


if __name__ == "__main__":
    asyncio.run(main())