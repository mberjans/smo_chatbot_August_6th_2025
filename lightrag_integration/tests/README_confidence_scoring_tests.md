# Intent Detection Confidence Scoring Tests - CMO-LIGHTRAG-012-T02

## Overview

This test suite provides comprehensive coverage for intent detection confidence scoring in the Clinical Metabolomics Oracle LightRAG Integration system. The tests validate that the system correctly calculates, categorizes, and applies confidence scores for various types of biomedical queries.

## Test Files

### 1. `test_intent_detection_confidence_scoring.py`
The main test suite containing comprehensive tests for:

- **Confidence Score Calculation** - Tests for different query types with expected confidence ranges
- **Confidence Threshold Handling** - Validation of high (0.8+), medium (0.6-0.79), low (0.4-0.59), and very low (<0.4) thresholds
- **Evidence-Based Scoring** - Tests for different evidence types (keywords, patterns, context, technical terms)
- **Confidence Normalization** - Boundary condition testing and score normalization
- **Confidence Consistency** - Reproducibility and consistency across similar queries
- **Confidence Degradation** - Testing with ambiguous, incomplete, or unclear queries
- **Query Type Integration** - Integration with QueryTypeClassifier system
- **Performance Testing** - Efficiency and scalability validation
- **Edge Cases** - Special characters, unicode, malformed input handling

### 2. `demo_confidence_scoring_tests.py`
Interactive demonstration script that shows:

- Basic confidence scoring examples
- Evidence-based scoring breakdown
- Performance characteristics
- Integration with query classification
- Real-time confidence validation

### 3. `README_confidence_scoring_tests.md` (this file)
Documentation and usage instructions

## Key Components Tested

### ResearchCategorizer Confidence System
- **Confidence Thresholds**: `{'high': 0.8, 'medium': 0.6, 'low': 0.4}`
- **Evidence Weights**: Multi-dimensional scoring with different evidence types
- **Normalization Factors**: Query complexity and length adjustments
- **Context Integration**: Bonus scoring for session and user context

### Evidence Types and Weights
```python
evidence_weights = {
    'exact_keyword_match': 1.0,      # Highest weight for exact matches
    'pattern_match': 0.8,            # Pattern/regex matches  
    'partial_keyword_match': 0.6,    # Partial term matches
    'context_bonus': 0.3,            # Session/user context
    'technical_terms_bonus': 0.2     # Technical vocabulary
}
```

### Query Types Supported
- `basic_definition` - "What is metabolomics?"
- `clinical_application` - Clinical usage queries
- `analytical_method` - LC-MS, GC-MS, NMR method queries
- `research_design` - Study methodology queries
- `disease_specific` - Disease-related metabolomics queries
- `general` - Fallback category

## Running the Tests

### 1. Run Demo Script (Recommended First Step)
```bash
cd lightrag_integration/tests
python demo_confidence_scoring_tests.py
```

This provides interactive examples and validates basic functionality.

### 2. Run Full Test Suite
```bash
# Run all confidence scoring tests
pytest test_intent_detection_confidence_scoring.py -v

# Run specific test classes
pytest test_intent_detection_confidence_scoring.py::TestConfidenceScoreCalculation -v
pytest test_intent_detection_confidence_scoring.py::TestConfidenceThresholdHandling -v
pytest test_intent_detection_confidence_scoring.py::TestEvidenceBasedConfidenceScoring -v

# Run with coverage
pytest test_intent_detection_confidence_scoring.py --cov=lightrag_integration.research_categorizer
```

### 3. Performance Testing
```bash
# Run performance-specific tests
pytest test_intent_detection_confidence_scoring.py::TestConfidencePerformance -v --tb=short

# Run with performance markers
pytest -m "performance" test_intent_detection_confidence_scoring.py
```

### 4. Integration Testing
```bash
# Test integration with query classification
pytest test_intent_detection_confidence_scoring.py::TestQueryTypeClassifierIntegration -v
```

## Test Structure

### Test Classes

1. **TestConfidenceScoreCalculation**
   - High confidence queries (>0.8)
   - Medium confidence queries (0.6-0.79)
   - Low confidence queries (<0.6)
   - Validates expected confidence ranges and evidence

2. **TestConfidenceThresholdHandling**
   - Threshold definitions and boundaries
   - Confidence level categorization
   - Boundary condition testing
   - Consistency across query types

3. **TestEvidenceBasedConfidenceScoring**
   - Evidence weight configuration
   - Keyword, pattern, and context scoring
   - Technical terms bonus validation
   - Multi-evidence combination testing

4. **TestConfidenceNormalizationBoundaries**
   - Score normalization to [0,1] range
   - Empty/minimal query handling
   - Very long query processing
   - Numerical stability testing

5. **TestConfidenceConsistency**
   - Similar query consistency
   - Reproducibility validation
   - Query order independence
   - Deterministic behavior

6. **TestConfidenceDegradation**
   - Ambiguous query handling
   - Conflicting terms impact
   - Incomplete query processing
   - Nonsensical query filtering

7. **TestQueryTypeClassifierIntegration**
   - Query type classification consistency
   - Confidence correlation with clarity
   - Cross-system validation

8. **TestConfidencePerformance**
   - Calculation speed (<10ms per query)
   - Scalability with query length
   - Memory efficiency
   - Throughput validation

9. **TestConfidenceEdgeCases**
   - Special characters and unicode
   - Malformed input handling
   - Extremely long queries
   - Numerical stability edge cases

10. **TestConfidenceIntegration**
    - End-to-end workflow validation
    - Statistics tracking
    - Complete system integration

### Test Fixtures

- **High Confidence Queries**: Complex, specific biomedical queries with clear intent
- **Medium Confidence Queries**: Moderately specific queries with some ambiguity  
- **Low Confidence Queries**: Vague or non-specific queries
- **Edge Case Queries**: Special characters, unicode, malformed inputs

## Expected Results

### High Confidence Examples (>0.8)
```python
"LC-MS/MS targeted metabolomics analysis of glucose and fructose in plasma samples using HILIC chromatography for diabetes biomarker identification"
# Expected: confidence ~0.85-0.95, category=METABOLITE_IDENTIFICATION, level='high'
```

### Medium Confidence Examples (0.6-0.79)
```python
"Analysis of metabolic changes in patient samples using mass spectrometry"
# Expected: confidence ~0.65-0.75, category varies, level='medium'
```

### Low Confidence Examples (<0.6)
```python
"How does this work?"
# Expected: confidence ~0.1-0.3, category=GENERAL_QUERY, level='very_low'
```

## Performance Benchmarks

- **Response Time**: < 10ms per query
- **Throughput**: > 100 queries per second
- **Memory**: < 1000 new objects per 50 queries
- **Consistency**: CV < 0.3 for similar queries
- **Reproducibility**: Identical scores for repeated queries

## Integration Points

### With Research Categorizer
- Category prediction with confidence scores
- Evidence collection and weighting
- Context-aware scoring adjustments

### With Query Type Classifier
- Query type determination for confidence correlation
- Type-specific confidence expectations
- Cross-system validation

### With Cost Persistence
- Research category enumeration
- Metadata integration
- Performance metrics tracking

## Error Handling

The tests validate proper error handling for:

- **Malformed Input**: Non-string inputs, None values
- **Edge Cases**: Empty queries, extremely long inputs
- **Unicode Support**: International characters and symbols
- **Special Characters**: Mathematical symbols, punctuation
- **Performance Limits**: Memory and time constraints

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Assertion Failures**: Check confidence threshold definitions match expected values
3. **Performance Issues**: Validate system resources and concurrent usage
4. **Unicode Errors**: Ensure proper text encoding handling

### Debug Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Inspect confidence calculation details
prediction = categorizer.categorize_query(query)
print(f"Category: {prediction.category}")
print(f"Confidence: {prediction.confidence}")
print(f"Evidence: {prediction.evidence}")
print(f"Metadata: {prediction.metadata}")
```

### Validation Steps

1. Run demo script first to validate basic functionality
2. Check individual test classes for specific failures
3. Validate confidence thresholds match system configuration
4. Ensure test data represents realistic biomedical queries
5. Verify performance benchmarks on target hardware

## Contributing

When adding new confidence scoring tests:

1. Follow existing test patterns and naming conventions
2. Include both positive and negative test cases
3. Add performance validation for new features
4. Update documentation with new test coverage
5. Ensure tests are deterministic and reproducible

## Related Documentation

- `research_categorizer.py` - Main categorization system
- `relevance_scorer.py` - Query type classification
- `cost_persistence.py` - Research category definitions
- `conftest.py` - Shared test fixtures and utilities

## Success Criteria

The test suite validates that:

✅ Confidence scores are calculated correctly for different query types
✅ Thresholds are properly applied and categorized
✅ Evidence-based scoring works with multiple evidence types
✅ Scores are normalized and bounded correctly
✅ Consistency is maintained across similar queries
✅ Degradation occurs appropriately for ambiguous queries
✅ Integration works with query classification system
✅ Performance meets benchmark requirements
✅ Edge cases are handled gracefully
✅ End-to-end workflows function correctly

This comprehensive test suite ensures the intent detection confidence scoring system provides reliable, accurate, and performant confidence assessment for biomedical metabolomics queries.