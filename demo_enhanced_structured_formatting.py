#!/usr/bin/env python3
"""
Enhanced Structured Response Formatting Demo

This script demonstrates the comprehensive structured response formatting capabilities
of the Clinical Metabolomics RAG system, including:

- Enhanced structured response templates
- Rich metadata generation
- Multi-format export options
- Biomedical context enhancement
- Statistical summary with visualization-ready data

Author: Clinical Metabolomics RAG System
Date: 2025-08-07
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add the project directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lightrag_integration.clinical_metabolomics_rag import BiomedicalResponseFormatter

def setup_logging():
    """Set up logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_structured_formatting.log')
        ]
    )
    return logging.getLogger(__name__)

def create_sample_metabolomics_response():
    """Create a sample metabolomics response for demonstration."""
    return """
    Clinical Metabolomics Analysis: Diabetes Biomarker Discovery

    Abstract:
    This study investigated metabolomic biomarkers for early diabetes detection using LC-MS/MS analysis
    of plasma samples from 250 patients and 200 healthy controls.

    Key Findings:
    Significant elevation in glucose (p < 0.001, 95% CI: 5.2-7.8 mmol/L) and insulin resistance markers
    were observed. The glycolysis pathway showed substantial dysregulation with multiple metabolite
    alterations including lactate (p = 0.003) and pyruvate (p = 0.012).

    Clinical Significance:
    These biomarkers demonstrate high diagnostic accuracy (AUC = 0.89) for early diabetes detection.
    The identified metabolic signature provides novel therapeutic targets and monitoring parameters
    for personalized diabetes management.

    Mechanisms:
    Disrupted glucose metabolism leads to enhanced glycolytic flux and altered TCA cycle activity.
    Key enzymes including hexokinase and pyruvate dehydrogenase show reduced activity, contributing
    to metabolic dysfunction.

    Statistical Results:
    - Sample size: n = 450 participants
    - Primary endpoint p-value: p < 0.001
    - Effect size: Cohen's d = 1.2 (large effect)
    - Sensitivity: 85% (95% CI: 78-92%)
    - Specificity: 87% (95% CI: 82-91%)
    - Correlation coefficient: r = 0.74 (p < 0.001)

    References:
    [1] Smith et al. Diabetes metabolomics study. Nature Medicine 2024. DOI: 10.1038/nm.2024.001
    [2] Jones et al. Glycolysis biomarkers. Cell Metabolism 2023. PMID: 12345678
    [3] Brown et al. LC-MS analysis methods. Analytical Chemistry 2024. PMC: PMC9876543
    """

def create_sample_metadata():
    """Create sample metadata for the demonstration."""
    return {
        'query': 'diabetes metabolomics biomarkers',
        'query_type': 'clinical_research',
        'timestamp': datetime.now().isoformat(),
        'source_count': 3,
        'processing_time': 2.5
    }

def demonstrate_comprehensive_format(formatter, response, metadata, logger):
    """Demonstrate comprehensive structured format."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING COMPREHENSIVE STRUCTURED FORMAT")
    logger.info("=" * 60)
    
    # Create comprehensive structured response
    structured_response = formatter.create_structured_response(
        raw_response=response,
        metadata=metadata,
        output_format='comprehensive'
    )
    
    # Display key sections
    logger.info(f"Response ID: {structured_response.get('response_id')}")
    logger.info(f"Format Type: {structured_response.get('format_type')}")
    logger.info(f"Version: {structured_response.get('version')}")
    
    # Executive Summary
    exec_summary = structured_response.get('executive_summary', {})
    logger.info(f"\nExecutive Summary:")
    logger.info(f"  Key Findings Count: {len(exec_summary.get('key_findings', []))}")
    logger.info(f"  Clinical Highlights: {len(exec_summary.get('clinical_significance', []))}")
    logger.info(f"  Confidence: {exec_summary.get('confidence_assessment', 'N/A')}")
    logger.info(f"  Complexity Score: {exec_summary.get('complexity_score', 'N/A')}")
    
    # Content Structure
    content_struct = structured_response.get('content_structure', {})
    logger.info(f"\nContent Structure Sections:")
    for section, data in content_struct.items():
        if isinstance(data, dict):
            logger.info(f"  {section}: {len(data)} subsections")
        else:
            logger.info(f"  {section}: Available")
    
    # Rich Metadata
    rich_metadata = structured_response.get('rich_metadata', {})
    logger.info(f"\nRich Metadata:")
    logger.info(f"  Processing Metadata: {bool(rich_metadata.get('processing_metadata'))}")
    logger.info(f"  Semantic Annotations: {bool(rich_metadata.get('semantic_annotations'))}")
    logger.info(f"  Provenance Tracking: {bool(rich_metadata.get('provenance_tracking'))}")
    
    # Export Formats
    export_formats = structured_response.get('export_formats', {})
    logger.info(f"\nAvailable Export Formats: {list(export_formats.keys())}")
    
    return structured_response

def demonstrate_clinical_report_format(formatter, response, metadata, logger):
    """Demonstrate clinical report format."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING CLINICAL REPORT FORMAT")
    logger.info("=" * 60)
    
    # Set format to clinical report
    formatter.set_output_format('clinical_report')
    
    clinical_report = formatter.create_structured_response(
        raw_response=response,
        metadata=metadata,
        output_format='clinical_report'
    )
    
    # Display clinical sections
    logger.info(f"Report ID: {clinical_report.get('report_id')}")
    logger.info(f"Report Type: {clinical_report.get('report_type')}")
    
    # Clinical Header
    clinical_header = clinical_report.get('clinical_header', {})
    logger.info(f"\nClinical Header:")
    logger.info(f"  Specialty: {clinical_header.get('specialty')}")
    logger.info(f"  Analysis Type: {clinical_header.get('analysis_type')}")
    logger.info(f"  Confidence Level: {clinical_header.get('confidence_level')}")
    logger.info(f"  Urgency Level: {clinical_header.get('urgency_level')}")
    
    # Clinical Sections
    logger.info(f"\nClinical Sections Available:")
    for section in ['clinical_findings', 'diagnostic_implications', 'therapeutic_considerations']:
        if section in clinical_report:
            logger.info(f"  ✓ {section.replace('_', ' ').title()}")
    
    # Evidence Base
    evidence_base = clinical_report.get('evidence_base', {})
    logger.info(f"\nEvidence Base:")
    logger.info(f"  Evidence Strength: {evidence_base.get('evidence_strength')}")
    logger.info(f"  Statistical Support: {evidence_base.get('statistical_support')}")
    logger.info(f"  Source Quality: {evidence_base.get('source_quality')}")
    
    return clinical_report

def demonstrate_research_summary_format(formatter, response, metadata, logger):
    """Demonstrate research summary format."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING RESEARCH SUMMARY FORMAT")
    logger.info("=" * 60)
    
    research_summary = formatter.create_structured_response(
        raw_response=response,
        metadata=metadata,
        output_format='research_summary'
    )
    
    # Display research sections
    logger.info(f"Summary ID: {research_summary.get('summary_id')}")
    logger.info(f"Summary Type: {research_summary.get('summary_type')}")
    
    # Research Header
    research_header = research_summary.get('research_header', {})
    logger.info(f"\nResearch Header:")
    logger.info(f"  Domain: {research_header.get('domain')}")
    logger.info(f"  Research Focus: {research_header.get('research_focus')}")
    logger.info(f"  Methodology: {research_header.get('methodology_type')}")
    logger.info(f"  Evidence Level: {research_header.get('evidence_level')}")
    
    # Visualization Data
    viz_data = research_summary.get('visualization_data', {})
    logger.info(f"\nVisualization Data Available:")
    for viz_type, data in viz_data.items():
        logger.info(f"  ✓ {viz_type.replace('_', ' ').title()}: {bool(data)}")
    
    # Future Directions
    future_directions = research_summary.get('future_directions', [])
    logger.info(f"\nFuture Research Directions: {len(future_directions)} identified")
    
    return research_summary

def demonstrate_api_friendly_format(formatter, response, metadata, logger):
    """Demonstrate API-friendly format."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING API-FRIENDLY FORMAT")
    logger.info("=" * 60)
    
    api_response = formatter.create_structured_response(
        raw_response=response,
        metadata=metadata,
        output_format='api_friendly'
    )
    
    # Display API structure
    logger.info(f"API Version: {api_response.get('api_version')}")
    logger.info(f"Response ID: {api_response.get('response_id')}")
    logger.info(f"Status: {api_response.get('status')}")
    
    # Data Section
    data_section = api_response.get('data', {})
    logger.info(f"\nStructured Data:")
    logger.info(f"  Summary: {bool(data_section.get('summary'))}")
    logger.info(f"  Entities: {bool(data_section.get('entities'))}")
    logger.info(f"  Metrics: {bool(data_section.get('metrics'))}")
    logger.info(f"  Relationships: {len(data_section.get('relationships', []))}")
    
    # Metadata for API consumers
    api_metadata = api_response.get('metadata', {})
    logger.info(f"\nAPI Metadata:")
    logger.info(f"  Confidence Scores: {bool(api_metadata.get('confidence_scores'))}")
    logger.info(f"  Data Quality: {bool(api_metadata.get('data_quality'))}")
    logger.info(f"  Semantic Annotations: {bool(api_metadata.get('semantic_annotations'))}")
    
    # Links and References
    links = api_response.get('links', {})
    logger.info(f"\nLinks Available: {list(links.keys())}")
    
    return api_response

def demonstrate_export_formats(structured_response, logger):
    """Demonstrate various export formats."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING EXPORT FORMATS")
    logger.info("=" * 60)
    
    export_formats = structured_response.get('export_formats', {})
    
    # JSON-LD Format
    json_ld = export_formats.get('json_ld', {})
    logger.info(f"JSON-LD Export:")
    logger.info(f"  Context: {bool(json_ld.get('@context'))}")
    logger.info(f"  Type: {json_ld.get('@type')}")
    logger.info(f"  Name: {json_ld.get('name')}")
    
    # Structured Markdown
    markdown = export_formats.get('structured_markdown', '')
    logger.info(f"\nStructured Markdown: {len(markdown)} characters")
    
    # CSV Data
    csv_data = export_formats.get('csv_data', {})
    logger.info(f"\nCSV Export:")
    if 'statistics_csv' in csv_data:
        logger.info(f"  Statistics CSV: {len(csv_data['statistics_csv'].get('rows', []))} rows")
    if 'entities_csv' in csv_data:
        logger.info(f"  Entities CSV: {len(csv_data['entities_csv'].get('rows', []))} rows")
    
    # BibTeX
    bibtex = export_formats.get('bibtex', '')
    logger.info(f"\nBibTeX Export: {len(bibtex)} characters")
    
    # XML Format
    xml_format = export_formats.get('xml_format', '')
    logger.info(f"XML Export: {len(xml_format)} characters")

def save_demo_results(results, logger):
    """Save demonstration results to files."""
    logger.info("=" * 60)
    logger.info("SAVING DEMONSTRATION RESULTS")
    logger.info("=" * 60)
    
    output_dir = Path('demo_results')
    output_dir.mkdir(exist_ok=True)
    
    for format_name, result in results.items():
        output_file = output_dir / f"{format_name}_demo_result.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"✓ Saved {format_name} result to: {output_file}")
        except Exception as e:
            logger.error(f"✗ Failed to save {format_name}: {e}")
    
    logger.info(f"\nAll demo results saved to: {output_dir}")

def main():
    """Main demonstration function."""
    logger = setup_logging()
    
    logger.info("Starting Enhanced Structured Response Formatting Demo")
    logger.info("=" * 80)
    
    try:
        # Initialize formatter with enhanced configuration
        formatter = BiomedicalResponseFormatter()
        
        # Update configuration for demonstration
        formatter.update_structured_formatting_config({
            'enable_structured_formatting': True,
            'generate_executive_summary': True,
            'enable_pathway_visualization': True,
            'include_semantic_annotations': True,
            'prepare_visualization_data': True
        })
        
        logger.info(f"Supported output formats: {formatter.get_supported_output_formats()}")
        
        # Create sample data
        sample_response = create_sample_metabolomics_response()
        sample_metadata = create_sample_metadata()
        
        # Store results for comparison
        demo_results = {}
        
        # Demonstrate each format type
        demo_results['comprehensive'] = demonstrate_comprehensive_format(
            formatter, sample_response, sample_metadata, logger
        )
        
        demo_results['clinical_report'] = demonstrate_clinical_report_format(
            formatter, sample_response, sample_metadata, logger
        )
        
        demo_results['research_summary'] = demonstrate_research_summary_format(
            formatter, sample_response, sample_metadata, logger
        )
        
        demo_results['api_friendly'] = demonstrate_api_friendly_format(
            formatter, sample_response, sample_metadata, logger
        )
        
        # Demonstrate export formats using comprehensive format
        demonstrate_export_formats(demo_results['comprehensive'], logger)
        
        # Save results
        save_demo_results(demo_results, logger)
        
        # Summary
        logger.info("=" * 80)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 80)
        logger.info("✓ Enhanced Structured Response Formatting Demo Completed Successfully!")
        logger.info(f"✓ Demonstrated {len(demo_results)} different output formats")
        logger.info("✓ Showcased comprehensive metadata generation")
        logger.info("✓ Demonstrated multi-format export capabilities")
        logger.info("✓ Illustrated biomedical context enhancement")
        logger.info("✓ Results saved for inspection")
        
        logger.info("\nKey Features Demonstrated:")
        logger.info("- Executive summaries with confidence assessment")
        logger.info("- Hierarchical content structure")
        logger.info("- Clinical decision support information")
        logger.info("- Research-focused analysis sections")
        logger.info("- API-friendly structured data")
        logger.info("- Rich semantic metadata")
        logger.info("- Multiple export format options")
        logger.info("- Biomedical entity relationship extraction")
        logger.info("- Statistical summary with visualization data")
        logger.info("- Comprehensive provenance tracking")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        logger.error("Check the logs for detailed error information")
        raise

if __name__ == "__main__":
    main()