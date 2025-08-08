# Comprehensive Routing Decision Logic Validation Suite

**Task**: CMO-LIGHTRAG-013-T01 - Write tests for routing decision logic  
**Created**: 2025-08-08  
**Author**: Claude Code (Anthropic)

## Overview

This comprehensive test suite validates the routing decision logic for the Clinical Metabolomics Oracle system, ensuring >90% routing accuracy and meeting all performance and reliability requirements for production deployment.

## Quick Start

### Basic Validation
```bash
# Run comprehensive validation (recommended)
python run_comprehensive_routing_validation.py

# Run quick validation (faster, for development)
python run_comprehensive_routing_validation.py --quick

# Run with verbose output
python run_comprehensive_routing_validation.py --verbose
```

### Using pytest
```bash
# Run all critical tests
pytest lightrag_integration/tests/test_comprehensive_routing_validation_suite.py -m "critical" -v

# Run routing accuracy tests only
pytest lightrag_integration/tests/test_comprehensive_routing_validation_suite.py -m "routing" -v

# Run performance tests only  
pytest lightrag_integration/tests/test_comprehensive_routing_validation_suite.py -m "performance" -v
```

## Test Suite Components

### 1. Core Routing Decision Tests
- **LIGHTRAG Routing**: Knowledge graph queries (>90% accuracy)
- **PERPLEXITY Routing**: Real-time/temporal queries (>90% accuracy)  
- **EITHER Routing**: General/flexible queries (>85% accuracy)
- **HYBRID Routing**: Complex multi-part queries (>85% accuracy)

### 2. Uncertainty Detection and Handling
- Low confidence uncertainty detection
- High ambiguity pattern recognition
- Conflicting signals identification
- Fallback strategy activation (100% correctness)

### 3. Performance Requirements
- Routing time: <50ms average, <50ms 95th percentile
- Throughput: >100 QPS sustained
- Memory stability: <100MB growth per hour
- Concurrent load handling: stable under 100+ requests

### 4. System Integration
- End-to-end workflow validation
- Cross-component communication
- Health monitoring integration
- Circuit breaker functionality

### 5. Edge Cases and Error Handling
- Malformed input robustness
- Component failure resilience
- System health degradation adaptation
- Graceful error recovery

## Test Data

### Clinical Metabolomics Domain
The test suite uses comprehensive clinical metabolomics domain knowledge:

- **Biomedical Entities**: glucose, insulin, diabetes, metabolomics, LC-MS, biomarkers, pathways
- **Clinical Workflows**: biomarker discovery, diagnostic development, method validation
- **Analytical Methods**: LC-MS, GC-MS, NMR, mass spectrometry, sample preparation
- **Real Scenarios**: Based on actual clinical metabolomics research patterns

### Test Dataset Sizes
- **Comprehensive Mode**: 375+ test cases across all categories
- **Quick Mode**: 100+ test cases for rapid development testing
- **Domain Coverage**: All major clinical metabolomics query types

## Success Criteria

### Production Readiness Requirements
✅ **Overall Accuracy**: ≥90%  
✅ **Response Time**: ≤50ms average  
✅ **Throughput**: ≥100 QPS  
✅ **Uncertainty Detection**: ≥95% accuracy  
✅ **System Integration**: ≥95% success rate  
✅ **Reliability**: ≥95% across all metrics  

### Category-Specific Targets
| Category | Accuracy Target | Performance |
|----------|----------------|-------------|
| LIGHTRAG | ≥90% | Knowledge graph access |
| PERPLEXITY | ≥90% | Real-time information |
| EITHER | ≥85% | Flexible routing |
| HYBRID | ≥85% | Multi-service coordination |

## Output and Reporting

### Generated Reports
- **JSON Results**: Machine-readable validation metrics
- **Markdown Report**: Comprehensive human-readable analysis  
- **Summary Report**: Executive summary for stakeholders
- **Performance Charts**: Visual performance analysis (when applicable)

### Sample Output Structure
```
validation_results/
├── validation_results_20250808_143022.json
├── validation_report_20250808_143022.md
├── validation_summary_20250808_143022.txt
└── performance_charts/ (if generated)
```

### Key Metrics Tracked
- Overall and category-specific accuracy
- Response time statistics (avg, p95, p99, max)
- Throughput and concurrency performance
- Memory usage and stability
- Uncertainty detection effectiveness
- Integration success rates
- Error handling robustness

## Configuration

### Customization Options
Edit `validation_config.json` to customize:
- Test dataset sizes
- Success criteria thresholds
- Mock router behavior
- Domain-specific entities
- Performance requirements
- Reporting options

### Mock Router Configuration
The advanced mock router provides realistic behavior:
- Confidence threshold-based routing decisions
- System health impact simulation
- Circuit breaker activation
- Uncertainty pattern detection
- Performance characteristic modeling

## Advanced Usage

### Custom Test Development
```python
from lightrag_integration.tests.test_comprehensive_routing_validation_suite import *

# Create custom test data
generator = ComprehensiveTestDataGenerator()
custom_queries = generator.generate_lightrag_queries(50)

# Initialize mock router with custom config
router = AdvancedMockBiomedicalQueryRouter({
    'system_health': 0.85,
    'lightrag_confidence_min': 0.80
})

# Run custom validation
for test_case in custom_queries:
    result = router.route_query(test_case.query)
    # Custom validation logic
```

### Integration with CI/CD
```bash
# Add to CI pipeline
python run_comprehensive_routing_validation.py --quick
if [ $? -eq 0 ]; then
    echo "Validation passed - deploying"
else
    echo "Validation failed - blocking deployment"
    exit 1
fi
```

### Performance Monitoring
```python
# Monitor specific performance scenarios
def monitor_production_performance():
    router = get_production_router()
    test_queries = load_production_query_patterns()
    
    for query in test_queries:
        start_time = time.perf_counter()
        result = router.route_query(query)
        response_time = (time.perf_counter() - start_time) * 1000
        
        # Log metrics for monitoring
        log_performance_metric(query, result, response_time)
```

## Troubleshooting

### Common Issues

#### Low Accuracy
- Check query classification logic
- Validate biomedical entity recognition
- Review confidence threshold settings
- Examine uncertainty detection patterns

#### Performance Issues
- Profile query processing bottlenecks
- Check system health metrics
- Review concurrent processing efficiency
- Monitor memory usage patterns

#### Integration Failures
- Validate component communication
- Check health monitoring functionality
- Review circuit breaker configurations
- Test fallback mechanism activation

### Debug Mode
```bash
# Enable debug logging
python run_comprehensive_routing_validation.py --verbose

# Run specific test category
pytest -v -s lightrag_integration/tests/test_comprehensive_routing_validation_suite.py::TestCoreRoutingAccuracy::test_lightrag_routing_comprehensive_accuracy
```

## Contributing

### Adding New Test Cases
1. Extend `ComprehensiveTestDataGenerator` with new query patterns
2. Add validation logic to appropriate test classes
3. Update success criteria in `validation_config.json`
4. Document new test scenarios

### Extending Mock Router
1. Add new behavioral patterns to `AdvancedMockBiomedicalQueryRouter`
2. Implement realistic failure simulation
3. Add domain-specific routing logic
4. Update configuration options

## Dependencies

### Required Packages
```
pytest>=7.0.0
psutil>=5.8.0  # For memory monitoring
statistics  # Built-in Python module
concurrent.futures  # Built-in Python module
```

### Optional Dependencies
```
matplotlib>=3.5.0  # For performance charts
pandas>=1.4.0     # For advanced analytics
```

## License and Support

This validation suite was created specifically for the Clinical Metabolomics Oracle project (CMO-LIGHTRAG-013-T01) and is designed to ensure the highest quality routing decision logic for clinical applications.

For questions or issues with the validation suite, refer to the comprehensive documentation in the generated reports or review the detailed test implementation.

---

**Generated by Claude Code (Anthropic)**  
**CMO-LIGHTRAG-013-T01 Implementation**  
**Last Updated: 2025-08-08**