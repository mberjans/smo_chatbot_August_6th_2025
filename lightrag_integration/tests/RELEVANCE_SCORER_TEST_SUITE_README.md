# Clinical Metabolomics Relevance Scorer Test Suite

## Overview

This comprehensive test suite provides thorough testing of the Clinical Metabolomics Relevance Scoring System implemented in `relevance_scorer.py`. The test suite is designed to validate all aspects of the relevance scoring functionality, including individual scoring dimensions, query classification, response quality validation, adaptive weighting schemes, edge cases, and performance characteristics.

## Test Coverage

### 1. Individual Scoring Dimension Tests (`TestIndividualScoringDimensions`)
- **Metabolomics Relevance**: Tests analytical method coverage, metabolite specificity, research context assessment
- **Clinical Applicability**: Validates disease relevance, diagnostic utility, therapeutic relevance scoring
- **Query Alignment**: Tests semantic similarity, keyword overlap, intent matching, context preservation
- **Scientific Rigor**: Validates evidence quality, statistical appropriateness, methodological soundness
- **Biomedical Context Depth**: Tests pathway integration, physiological relevance, multi-omics integration

### 2. Query Classification Tests (`TestQueryClassification`)
- **Basic Definition**: "What is metabolomics?", "Define biomarker", etc.
- **Clinical Application**: "How is metabolomics used in diagnosis?", clinical implementation queries
- **Analytical Method**: LC-MS protocols, GC-MS procedures, NMR spectroscopy methods
- **Research Design**: Study design, statistical analysis, validation strategies
- **Disease Specific**: Diabetes, cancer, cardiovascular disease metabolomics
- **Edge Cases**: Empty queries, nonsensical input, special characters

### 3. Response Quality Validation Tests (`TestResponseQualityValidation`)
- **Length Quality**: Appropriate response length for different query types
- **Structure Quality**: Formatting, organization, coherence, readability assessment
- **Formatting Assessment**: Markdown usage, bullet points, paragraph structure
- **Readability Evaluation**: Sentence length, technical terminology balance, clarity

### 4. Adaptive Weighting Scheme Tests (`TestAdaptiveWeightingSchemes`)
- **Completeness**: All query types have complete weighting schemes
- **Type-Specific Weighting**: Clinical queries weight clinical applicability higher
- **Consistency**: Same dimensions used across query types
- **Weight Validation**: Weights sum to ~1.0, values between 0-1

### 5. Edge Cases Tests (`TestEdgeCases`)
- **Empty Inputs**: Empty queries, empty responses, whitespace only
- **Very Long Inputs**: Performance with extremely long text
- **Nonsensical Inputs**: Random characters, emoji, repetitive text
- **Special Characters**: Unicode, JSON, XML, code snippets
- **Malformed Metadata**: NaN values, deeply nested objects, circular references

### 6. Performance Tests (`TestPerformance`)
- **Async Execution**: Concurrent vs sequential performance
- **Response Time**: Sub-second response times for typical inputs
- **Throughput**: Minimum operations per second requirements
- **Memory Efficiency**: No memory leaks during repeated operations
- **Concurrent Load**: Handling multiple simultaneous requests

### 7. Semantic Similarity Engine Tests (`TestSemanticSimilarityEngine`)
- **Basic Similarity**: Jaccard similarity with biomedical term weighting
- **Biomedical Boost**: Enhanced scoring for domain-specific terminology
- **Term Extraction**: Meaningful term extraction excluding stopwords
- **Symmetry**: Similarity calculation symmetry verification
- **Consistency**: Deterministic similarity calculations

### 8. Domain Expertise Validator Tests (`TestDomainExpertiseValidator`)
- **Expertise Validation**: Technical terminology usage assessment
- **Methodology Assessment**: Scientific methodology recognition
- **Error Penalties**: Detection of overstatements and unsupported claims
- **Evidence Quality**: Recognition of evidence-based statements

### 9. Integration and Pipeline Tests (`TestIntegrationAndPipeline`)
- **Complete Pipeline**: End-to-end relevance scoring workflow
- **Batch Processing**: Multiple query-response pairs processing
- **Confidence Scoring**: Score consistency-based confidence calculation
- **Explanation Generation**: Human-readable scoring explanations
- **Quality Validation**: Integration with response quality assessment

### 10. Stress and Robustness Tests (`TestStressAndRobustness`)
- **High Load**: 50+ concurrent requests handling
- **Exception Recovery**: Graceful handling of invalid inputs
- **Resource Cleanup**: Proper cleanup of async resources

### 11. Configuration Tests (`TestConfigurationAndCustomization`)
- **Default Configuration**: Reasonable default settings validation
- **Custom Configuration**: Custom parameter support
- **Processing Modes**: Parallel vs sequential processing comparison

### 12. Biomedical Domain Tests (`TestBiomedicalDomainSpecifics`)
- **Terminology Recognition**: Biomedical keyword detection
- **Clinical Context**: Clinical application recognition
- **Method Specificity**: Analytical method detail assessment
- **Keyword Coverage**: Comprehensive biomedical vocabulary

## Test Files Structure

```
tests/
├── test_relevance_scorer.py              # Main test suite (2,000+ lines)
├── relevance_scorer_test_fixtures.py     # Comprehensive test data (1,500+ lines)
├── run_relevance_scorer_tests.py         # Test runner with reporting (800+ lines)
└── RELEVANCE_SCORER_TEST_SUITE_README.md # This documentation
```

## Running the Tests

### Quick Test Run
```bash
python -m pytest test_relevance_scorer.py -v
```

### Comprehensive Test Run with Reporting
```bash
python run_relevance_scorer_tests.py --coverage --report-format html
```

### Category-Specific Testing
```bash
# Run only performance tests
python run_relevance_scorer_tests.py --category performance

# Run multiple categories
python run_relevance_scorer_tests.py --category dimensions classification quality

# Run with parallel execution
python run_relevance_scorer_tests.py --parallel --workers 4
```

### Advanced Options
```bash
# Full test suite with all reports
python run_relevance_scorer_tests.py \
    --category all \
    --parallel \
    --coverage \
    --performance \
    --report-format all \
    --output-dir ./test_reports \
    --verbose

# Stress testing
python run_relevance_scorer_tests.py --category stress --timeout 600
```

## Test Data and Fixtures

### Query Fixtures
- **Basic Definition**: 5 queries testing definition and explanation requests
- **Clinical Application**: 5 queries about clinical usage and implementation
- **Analytical Method**: 5 queries about LC-MS, GC-MS, NMR protocols
- **Research Design**: 5 queries about study design and methodology
- **Disease Specific**: 5 queries about disease-specific metabolomics
- **Edge Cases**: 5 problematic queries (empty, nonsensical, etc.)

### Response Fixtures
- **Excellent Responses**: 2 comprehensive, well-structured responses (500-2000 words)
- **Good Responses**: 2 solid responses with good coverage (200-500 words)
- **Fair Responses**: 1 adequate but limited response (100-200 words)
- **Poor Responses**: 2 inadequate responses (<50 words)
- **Edge Case Responses**: 4 problematic responses (empty, irrelevant, contradictory)

### Test Scenarios
- **Standard Scenarios**: 8 realistic query-response combinations
- **Performance Scenarios**: 100 automatically generated scenarios
- **Stress Scenarios**: 500 scenarios for load testing

## Expected Test Results

### Passing Criteria
- **Individual Dimensions**: All dimension scorers return values 0-100
- **Query Classification**: 90%+ accuracy on test queries
- **Quality Validation**: Appropriate quality scores for different response types
- **Performance**: Sub-second response times, 5+ ops/sec throughput
- **Edge Cases**: Graceful handling without crashes
- **Integration**: End-to-end pipeline produces valid RelevanceScore objects

### Performance Benchmarks
- **Response Time**: <1000ms for typical queries
- **Throughput**: >5 operations per second
- **Concurrent Load**: Handle 10+ simultaneous requests
- **Memory**: No significant memory leaks over 100+ operations

## Test Failure Analysis

### Common Failure Patterns
1. **Module Import Errors**: relevance_scorer.py not available
2. **Performance Timeouts**: Operations exceeding time limits  
3. **Score Range Violations**: Scores outside 0-100 range
4. **Type Errors**: Invalid return types from components
5. **Async Errors**: Improper async/await usage

### Debugging Tips
1. **Enable Verbose Logging**: Use `--verbose` flag
2. **Single Category Testing**: Test specific categories in isolation
3. **Mock Component Testing**: Test with mock components first
4. **Performance Profiling**: Use `--performance` flag for timing analysis

## Integration with CI/CD

### GitHub Actions Integration
```yaml
name: Relevance Scorer Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install Dependencies
        run: |
          pip install pytest pytest-asyncio pytest-cov
          pip install -r requirements.txt
      - name: Run Tests
        run: |
          python run_relevance_scorer_tests.py \
            --category all \
            --coverage \
            --report-format json \
            --output-dir ./test-results
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: ./test-results/
```

### Pre-commit Hooks
```yaml
repos:
  - repo: local
    hooks:
      - id: relevance-scorer-tests
        name: Relevance Scorer Tests
        entry: python run_relevance_scorer_tests.py --category dimensions classification
        language: system
        pass_filenames: false
```

## Test Maintenance

### Adding New Test Cases
1. **Query Tests**: Add to appropriate category in `QueryFixtures`
2. **Response Tests**: Add to quality level in `ResponseFixtures`
3. **Scenarios**: Add to `ScenarioFixtures.generate_standard_scenarios()`
4. **Edge Cases**: Add to `TestEdgeCases` class

### Updating Expected Ranges
1. Monitor test results over time
2. Adjust expected score ranges based on empirical data
3. Update performance benchmarks as system improves
4. Document changes in test commit messages

### Test Data Generation
```python
# Generate new test fixtures
python relevance_scorer_test_fixtures.py

# This creates relevance_scorer_fixtures.json with:
# - 30 test queries across 6 categories
# - 10 test responses across 5 quality levels
# - Metadata and statistics
```

## Reporting and Analytics

### Report Formats
- **JSON**: Machine-readable results for automation
- **HTML**: Rich visual reports with charts and tables  
- **Text**: Simple text summary for logs and CLI

### Key Metrics Tracked
- **Test Coverage**: Lines and branches covered
- **Pass/Fail Rates**: Success rates by category
- **Performance**: Response times and throughput
- **Score Distributions**: Statistical analysis of scores
- **Error Patterns**: Common failure modes

### Report Contents
- **Executive Summary**: High-level pass/fail status
- **Category Breakdown**: Detailed results by test category
- **Performance Analysis**: Timing and throughput metrics
- **Coverage Analysis**: Code coverage percentages
- **Recommendations**: Suggested improvements and fixes

## Best Practices

### Test Development
1. **Write Tests First**: TDD approach for new features
2. **Comprehensive Coverage**: Test both happy path and edge cases
3. **Clear Naming**: Descriptive test method names
4. **Independent Tests**: No dependencies between test methods
5. **Deterministic Results**: Consistent results across runs

### Maintenance
1. **Regular Execution**: Run tests on every commit
2. **Performance Monitoring**: Track performance trends
3. **Data Updates**: Keep test data current and relevant
4. **Documentation**: Update README with changes
5. **Cleanup**: Remove obsolete tests and data

### Debugging
1. **Isolation**: Test components independently
2. **Logging**: Use detailed logging for complex failures
3. **Assertions**: Clear assertion messages
4. **Parametrization**: Test multiple scenarios efficiently
5. **Mocking**: Mock external dependencies appropriately

## Future Enhancements

### Planned Improvements
1. **Machine Learning Integration**: ML-based scoring validation
2. **Comparative Analysis**: Compare with other scoring systems
3. **Real-time Monitoring**: Production scoring system monitoring
4. **Automated Tuning**: Parameter optimization based on test results
5. **Extended Coverage**: Additional biomedical domains

### Research Directions
1. **Semantic Embeddings**: Advanced similarity calculations
2. **Domain Adaptation**: Custom weighting for specialized domains
3. **User Feedback Integration**: Learning from user relevance ratings
4. **Multi-modal Scoring**: Integration with image and structured data
5. **Explainable AI**: Enhanced explanation generation

## Support and Contact

### Documentation
- **Code Documentation**: Inline docstrings and comments
- **API Reference**: Generated from docstrings
- **User Guide**: Step-by-step testing instructions
- **FAQ**: Common questions and solutions

### Getting Help
1. **Issue Tracker**: GitHub issues for bugs and features
2. **Documentation**: This README and inline comments
3. **Test Examples**: Examine existing test cases
4. **Code Review**: Request code review for complex changes

---

**Last Updated**: August 7, 2025  
**Version**: 1.0.0  
**Author**: Claude Code (Anthropic)  
**Related**: Clinical Metabolomics Oracle LightRAG Integration