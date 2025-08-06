# BiomedicalPDFProcessor Performance Benchmark

This directory contains comprehensive performance benchmark results for the BiomedicalPDFProcessor component of the Clinical Metabolomics Oracle system.

## Overview

The benchmark suite tests all key aspects of PDF processing:

- **Processing Performance**: Time per PDF, throughput rates, page processing speeds
- **Memory Usage**: Peak memory consumption, memory leak detection, stress testing
- **Text Quality**: Extraction completeness, preprocessing effectiveness, encoding validation
- **Error Handling**: Timeout handling, invalid inputs, edge cases, robustness
- **Metadata Extraction**: Completeness and accuracy of document metadata
- **Biomedical Content**: Recognition and preservation of scientific notation, terms, and formatting

## Quick Start

### Running the Benchmark

```bash
# Simple way - from project root:
python run_pdf_benchmark.py

# Advanced way - with custom options:
python -m lightrag_integration.benchmark_pdf_processing --papers-dir papers/ --output-dir benchmark_results/ --verbose
```

### Adding More PDF Samples

Currently, only 1 PDF file is available for testing. For comprehensive benchmarking, add 5+ diverse biomedical PDFs to the `papers/` directory.

**Recommended PDF Sources:**

1. **PubMed Central (PMC)**: https://www.ncbi.nlm.nih.gov/pmc/
   - Search for clinical metabolomics papers
   - Download full-text PDFs of open access articles

2. **bioRxiv Preprints**: https://www.biorxiv.org/
   - Filter by "Systems Biology" or "Biochemistry"
   - Look for metabolomics-related preprints

3. **medRxiv Preprints**: https://www.medrxiv.org/
   - Clinical research preprints
   - Search for "metabolomics" or "biomarker" studies

4. **PLOS Journals**: https://plos.org/
   - PLOS ONE, PLOS Computational Biology
   - Open access biomedical research

5. **Nature Journals**: https://www.nature.com/
   - Nature Metabolism, Nature Medicine
   - Scientific Reports (open access)

**Selection Criteria for Good Test PDFs:**

- **Variety in Size**: Include both short (5-10 pages) and long (20+ pages) papers
- **Content Diversity**: 
  - Research articles with figures and tables
  - Review papers with extensive references
  - Method papers with detailed protocols
  - Case studies with clinical data
- **Different Publishers**: Various PDF generation tools and formatting styles
- **Scientific Content**: Rich in biomedical terminology, chemical formulas, statistical data

## File Structure

```
benchmark_results/
├── README_BENCHMARK.md                    # This file
├── benchmark_YYYYMMDD_HHMMSS.log         # Detailed benchmark logs
├── benchmark_results_YYYYMMDD_HHMMSS.json # Machine-readable results
└── benchmark_report_YYYYMMDD_HHMMSS.txt   # Human-readable report
```

## Understanding Results

### Performance Metrics

- **Processing Time**: Total time to extract text from PDF
- **Throughput**: Characters processed per second
- **Memory Usage**: Peak memory consumption during processing
- **Pages per Second**: Page processing rate

### Quality Metrics

- **Text Completeness**: Percentage of pages with extractable content
- **Preprocessing Effectiveness**: Improvement in text quality after cleaning
- **Biomedical Content Recognition**: Detection of scientific terms, formulas, citations
- **Encoding Quality**: Unicode handling and character preservation

### Error Handling Tests

- **Timeout Handling**: Behavior with artificially short timeouts
- **Memory Limits**: Processing under constrained memory conditions
- **Invalid Inputs**: Response to corrupted files, wrong paths, etc.
- **Edge Cases**: Handling of unusual page ranges and parameters

## Current Status

### Task: CMO-LIGHTRAG-003-T10

**Objective**: Performance benchmark with 5+ different biomedical PDFs

**Current Status**: 
- ✅ Comprehensive benchmark framework implemented
- ✅ Successfully testing with 1 PDF (Clinical_Metabolomics_paper.pdf)
- ⚠️ Need 4+ additional diverse biomedical PDFs for complete testing

**Performance Results (Current PDF)**:
- File: Clinical_Metabolomics_paper.pdf (0.27 MB, 12 pages)
- Processing time: ~0.13s with preprocessing, ~0.05s without
- Throughput: ~400K chars/sec (with preprocessing)
- Memory usage: Stable, no leaks detected
- Text quality: Good extraction, some encoding issues detected
- Error handling: All timeout and edge case tests pass

## Interpretation Guidelines

### Good Performance Indicators

- **Processing Time**: < 1 second per MB for typical biomedical PDFs
- **Memory Usage**: < 100MB increase for documents under 10MB
- **Text Quality**: > 90% page extraction rate, < 1% encoding errors
- **Error Handling**: All timeout/memory limit tests trigger appropriate exceptions

### Warning Signs

- **Slow Processing**: > 5 seconds per MB may indicate optimization needs
- **Memory Leaks**: Increasing memory usage across repeated processing
- **Poor Text Quality**: High encoding error rates, low scientific content recognition
- **Inconsistent Error Handling**: Unexpected exceptions or missing error handling

### Recommendations Based on Results

The benchmark automatically generates recommendations including:

1. **Sample Size**: Need for additional diverse PDF samples
2. **Performance Optimization**: Areas where processing could be improved
3. **Quality Enhancement**: Text extraction and preprocessing improvements
4. **Error Handling**: Robustness improvements needed
5. **Memory Management**: Memory usage optimization opportunities

## Advanced Usage

### Custom Benchmark Configuration

Modify the benchmark behavior by editing the PDFProcessingBenchmark parameters:

```python
benchmark = PDFProcessingBenchmark(
    papers_dir="custom_papers/",        # Custom PDF directory
    output_dir="custom_results/",       # Custom output directory
    verbose=True                        # Enable detailed logging
)
```

### Processor Configuration Testing

Test different BiomedicalPDFProcessor configurations:

```python
processor = BiomedicalPDFProcessor(
    processing_timeout=600,     # 10 minutes
    memory_limit_mb=2048,       # 2GB
    max_page_text_size=2000000  # 2MB per page
)
```

### Continuous Benchmarking

For ongoing performance monitoring, consider:

1. **Automated Runs**: Schedule regular benchmarks with new PDF samples
2. **Performance Tracking**: Monitor trends in processing speed and quality
3. **Regression Testing**: Ensure changes don't degrade performance
4. **Comparative Analysis**: Compare results across different PDF types

## Troubleshooting

### Common Issues

1. **No PDFs Found**: Ensure PDF files are in the `papers/` directory
2. **Permission Errors**: Check file permissions on PDF files
3. **Memory Issues**: Reduce memory limits or use smaller test files
4. **Timeout Errors**: Increase timeout limits for very large PDFs

### Getting Help

For issues with the benchmark:

1. Check the detailed log files in `benchmark_results/`
2. Review the error handling test results for specific failure patterns
3. Ensure PDF files are valid and not corrupted
4. Verify system has sufficient memory and disk space

## Next Steps

To complete Task CMO-LIGHTRAG-003-T10:

1. **Add PDF Samples**: Download 4+ additional biomedical PDFs from recommended sources
2. **Run Full Benchmark**: Execute comprehensive testing with diverse sample set
3. **Analyze Results**: Review performance patterns across different PDF types
4. **Optimize as Needed**: Address any performance or quality issues identified
5. **Document Findings**: Update system documentation with benchmark results

---

*This benchmark suite is part of the Clinical Metabolomics Oracle (CMO) system's LightRAG integration project. For more information, see the main project documentation.*