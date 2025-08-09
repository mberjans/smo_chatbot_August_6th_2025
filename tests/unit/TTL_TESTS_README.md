# TTL (Time-To-Live) Functionality Test Suite

## Overview

This comprehensive test suite validates TTL functionality for the Clinical Metabolomics Oracle caching system. The tests cover multi-tier TTL management, expiration scenarios, dynamic TTL adjustments, and performance impact across the entire caching architecture.

## Test Structure

### Core Test Files

1. **`test_ttl_functionality.py`** - Core TTL behavior testing
2. **`test_ttl_integration.py`** - TTL integration with cache systems  
3. **`test_ttl_performance.py`** - TTL performance impact testing

### Supporting Files

- **`cache_test_fixtures.py`** - Shared test fixtures and utilities
- **`TTL_TESTS_README.md`** - This documentation file

## TTL Configuration Used

Based on the existing system analysis, the tests use realistic TTL values:

- **L1 Cache**: 300s (5 minutes)
- **L2 Cache**: 3600s (1 hour)  
- **L3 Cache**: 86400s (24 hours)
- **Emergency Cache**: 86400s (24 hours)
- **Fallback System**: 1800s - 7200s (30 minutes - 2 hours)
- **High Confidence Responses**: 7200s (2 hours)
- **Low Confidence Responses**: 900s (15 minutes)
- **Temporal Queries**: 300s (5 minutes)
- **Clinical Critical**: 14400s (4 hours)

## Test Coverage

### 1. Core TTL Functionality (`test_ttl_functionality.py`)

#### `TestTTLBasicFunctionality`
- Basic TTL setting and retrieval
- Custom TTL value configuration  
- TTL expiration behavior
- TTL precision and timing accuracy

#### `TestTTLExpirationScenarios`  
- Multiple entry expiration with different TTLs
- Expiration cleanup timing and efficiency
- Boundary condition testing (just before/after expiration)
- Realistic biomedical query expiration patterns

#### `TestDynamicTTLManagement`
- TTL extension for active entries
- TTL refresh with new timestamp
- Adaptive TTL based on access patterns
- Confidence-based TTL adjustments
- Query-type specific TTL policies

#### `TestConfidenceBasedTTL`
- High confidence entries get longer TTL
- TTL scaling across confidence ranges
- Confidence threshold behavior validation

#### `TestMultiTierTTLCoordination`
- TTL cascading from higher to lower tiers
- TTL synchronization across cache tiers
- Cross-tier TTL consistency during promotion/demotion
- Tier-specific TTL policy enforcement

#### `TestTTLBoundaryConditions`
- Zero and negative TTL handling
- Very large TTL values
- Concurrent TTL operations
- Clock change simulation impact

#### `TestTTLExtensionRefresh`
- Multiple TTL extensions on same entry
- TTL refresh age reset verification
- Extension vs refresh behavior differences

#### `TestTTLPerformanceImpact`
- TTL cleanup performance measurement
- TTL extension operation performance
- Memory usage with TTL metadata
- Performance at different cache scales

### 2. TTL Integration Testing (`test_ttl_integration.py`)

#### `TestTTLEvictionIntegration`
- TTL expiration vs LRU eviction interaction
- Priority-based eviction considering TTL
- TTL-aware eviction strategies

#### `TestTTLCacheWarmingIntegration`  
- TTL assignment during cache warming
- Warming vs regular entry TTL interaction
- Warming performance impact validation
- Warmed entry expiration behavior

#### `TestTTLSystemRestartIntegration`
- TTL persistence across system restarts
- Expired entry exclusion from persistence
- TTL adjustment for elapsed restart time
- Restart behavior with/without persistence

#### `TestTTLHighLoadIntegration`
- TTL behavior under concurrent access
- High load mode TTL adjustments
- Load balancing with TTL considerations
- TTL cleanup performance under load

#### `TestTTLDistributedIntegration`
- TTL synchronization across distributed nodes
- Distributed TTL refresh coordination
- Network partition TTL behavior
- Multi-node TTL consistency

#### `TestTTLEmergencyIntegration`
- Emergency mode TTL adjustments
- Emergency cache TTL policies
- Emergency failover TTL consistency

#### `TestTTLMonitoringIntegration`
- TTL metrics collection and reporting
- TTL event logging for monitoring
- TTL performance monitoring impact
- TTL-based alerting threshold detection

### 3. TTL Performance Testing (`test_ttl_performance.py`)

#### `TestTTLOperationPerformance`
- TTL set operation performance benchmarks
- TTL get operation performance measurement
- TTL extension performance validation
- TTL cleanup operation performance

#### `TestTTLMemoryPerformance`
- TTL metadata memory overhead analysis
- Memory usage scaling with cache size
- Memory fragmentation impact assessment

#### `TestTTLScalabilityPerformance` 
- TTL performance scaling with cache size
- TTL cleanup scaling efficiency
- Extreme scale performance validation

#### `TestTTLConcurrencyPerformance`
- Concurrent TTL operation performance
- Mixed concurrent operation testing
- Concurrent cleanup performance impact

#### `TestTTLThroughputPerformance`
- TTL vs non-TTL throughput comparison  
- Sustained throughput over time
- Throughput stability measurement

#### `TestTTLBenchmarkSuite`
- Comprehensive TTL performance benchmarks
- Multi-scale performance comparison
- Concurrency scaling analysis
- Performance regression detection

## Performance Thresholds

The tests validate against specific performance targets:

- **TTL Set Operations**: < 1ms average
- **TTL Get Operations**: < 0.5ms average  
- **TTL Cleanup**: < 10ms per 1000 entries
- **TTL Extension**: < 0.1ms average
- **Memory Overhead**: < 20% additional memory
- **Throughput Degradation**: < 10% reduction

## Running the Tests

### Individual Test Files

```bash
# Core TTL functionality tests
python -m pytest test_ttl_functionality.py -v

# TTL integration tests  
python -m pytest test_ttl_integration.py -v

# TTL performance tests
python -m pytest test_ttl_performance.py -v
```

### Specific Test Classes

```bash  
# Basic TTL functionality
python -m pytest test_ttl_functionality.py::TestTTLBasicFunctionality -v

# TTL expiration scenarios
python -m pytest test_ttl_functionality.py::TestTTLExpirationScenarios -v

# TTL with cache eviction
python -m pytest test_ttl_integration.py::TestTTLEvictionIntegration -v

# TTL performance benchmarks
python -m pytest test_ttl_performance.py::TestTTLBenchmarkSuite -v
```

### All TTL Tests

```bash
# Run all TTL tests
python -m pytest test_ttl_*.py -v

# Run with coverage
python -m pytest test_ttl_*.py --cov=. --cov-report=html
```

## Test Scenarios

### Realistic Biomedical Examples

The tests use realistic biomedical queries and scenarios:

- **Metabolic Pathway Queries**: Long TTL for stable information
- **Clinical Biomarker Data**: Medium TTL with confidence-based adjustment
- **Current Research Queries**: Short TTL for time-sensitive information  
- **Emergency Medical Information**: Extended TTL for critical data
- **Drug Discovery Data**: Variable TTL based on data stability

### Edge Cases Covered

- Zero and negative TTL values
- Very large TTL values (2^31-1 seconds)
- Concurrent TTL modifications
- System clock changes
- Memory pressure scenarios
- Network partition conditions
- High load conditions
- Cache warming scenarios

### Multi-Tier Scenarios

- L1 → L2 → L3 cache promotion with TTL adjustment
- Cross-tier TTL synchronization
- Tier-specific TTL policies
- Emergency cache fallback TTL management
- Distributed cache TTL coordination

## Test Data and Fixtures

### Biomedical Test Data

The tests use comprehensive biomedical test data including:

- **Metabolism Queries**: Glucose metabolism, insulin regulation, diabetes metabolites
- **Clinical Applications**: Drug discovery, biomarker identification, toxicity screening
- **Disease Metabolomics**: Cancer metabolism, Alzheimer's disease, cardiovascular markers
- **Temporal Queries**: Latest research, current studies (short TTL)
- **Emergency Patterns**: Critical medical information, fallback responses

### Mock Cache Implementations

- **MockTTLCache**: Basic TTL functionality testing
- **IntegratedTTLCache**: Complex integration scenarios
- **PerformanceTTLCache**: High-performance testing with optimization

### Performance Measurement

- **PerformanceMetrics**: Comprehensive timing and throughput measurement
- **CachePerformanceMeasurer**: Multi-operation performance analysis
- **PerformanceTestRunner**: Automated benchmark execution

## Integration Points

### Cache System Integration

- **Multi-Tier Cache**: L1/L2/L3 cache coordination
- **Emergency Cache**: Fallback TTL management  
- **Query Router**: TTL-aware routing decisions
- **Storage Operations**: TTL-integrated storage backends

### System Integration

- **Monitoring**: TTL metrics and alerting
- **Load Balancing**: TTL-aware load distribution
- **Circuit Breakers**: TTL during system failures
- **Performance Monitoring**: TTL impact measurement

## Validation and Assertions

### Functional Validation

- TTL expiration accuracy (within 100ms tolerance)
- Cross-tier TTL consistency
- Confidence-based TTL adjustment correctness
- Emergency mode TTL override behavior

### Performance Validation  

- Operation timing within specified thresholds
- Memory overhead within acceptable limits
- Throughput degradation below 10%
- Scalability maintaining sub-linear growth

### Integration Validation

- TTL coordination across system restarts
- High load TTL adjustment accuracy  
- Distributed TTL synchronization
- Emergency failover TTL consistency

## Expected Test Results

### Functional Tests

- **Core TTL Functionality**: 100% pass rate expected
- **TTL Integration**: 95%+ pass rate (some timing-sensitive tests)
- **Edge Cases**: All boundary conditions handled correctly

### Performance Tests

- **Basic Operations**: Meet all performance thresholds
- **Scalability**: Maintain performance at 10K+ entries
- **Concurrency**: Handle 20+ concurrent threads
- **Memory**: < 20% overhead for TTL metadata

### Integration Tests

- **System Integration**: Seamless TTL coordination
- **Multi-Tier**: Proper TTL cascading and synchronization  
- **Emergency Scenarios**: Reliable TTL management under stress

## Troubleshooting

### Common Issues

1. **Timing-Sensitive Test Failures**
   - Increase tolerance for slow systems
   - Check system clock accuracy
   - Reduce concurrent test execution

2. **Memory-Related Failures**  
   - Increase available system memory
   - Reduce test scale for resource-constrained systems
   - Enable garbage collection between tests

3. **Performance Threshold Failures**
   - Verify system not under load during testing
   - Adjust thresholds for slower hardware
   - Check for background processes

### Debugging TTL Issues

1. **TTL Expiration Problems**
   - Verify system time accuracy
   - Check TTL calculation logic
   - Validate expiration cleanup timing

2. **Multi-Tier TTL Issues**
   - Verify tier-to-tier TTL propagation
   - Check TTL adjustment algorithms
   - Validate cross-tier synchronization

3. **Performance Issues**
   - Profile TTL operation timing
   - Check memory allocation patterns
   - Analyze cleanup algorithm efficiency

## Contributing

When adding new TTL tests:

1. Use realistic biomedical scenarios
2. Include comprehensive docstrings
3. Add appropriate performance assertions
4. Test both success and failure cases
5. Include boundary condition testing
6. Validate against existing TTL configurations

## Summary

This comprehensive TTL test suite provides thorough validation of TTL functionality across all aspects of the Clinical Metabolomics Oracle caching system. The tests ensure reliable TTL behavior, optimal performance, and robust integration with the broader system architecture while using realistic biomedical scenarios and data patterns.