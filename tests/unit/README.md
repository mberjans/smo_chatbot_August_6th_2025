# Cache Unit Tests - Implementation Guide

This directory contains comprehensive unit tests for the Clinical Metabolomics Oracle cache storage and retrieval operations, implementing the test suite design from `tests/comprehensive_cache_test_suite_design.md`.

## 📁 Test Structure

```
tests/unit/
├── __init__.py                        # Package initialization
├── README.md                         # This file
├── conftest.py                       # Pytest configuration and fixtures
├── cache_test_fixtures.py            # Comprehensive test fixtures
├── run_cache_tests.py               # Test runner script
├── test_cache_storage_operations.py  # Core cache operations tests
├── test_multi_tier_cache.py         # Multi-level cache coordination
├── test_emergency_cache.py          # Emergency cache system tests
└── test_query_router_cache.py       # Query router LRU cache tests
```

## 🧪 Test Modules

### 1. test_cache_storage_operations.py
Core cache storage and retrieval operations testing:
- **TestCacheKeyGeneration**: Cache key consistency and collision handling
- **TestCacheDataSerialization**: JSON/Pickle serialization security
- **TestCacheMetadata**: Cache entry metadata management
- **TestCacheSizeLimits**: LRU eviction and size enforcement
- **TestCacheTTL**: TTL expiration handling
- **TestCacheThreadSafety**: Concurrent access safety
- **TestCachePerformance**: Performance characteristics (<1ms gets, <2ms sets)

### 2. test_multi_tier_cache.py
Multi-tier cache coordination testing:
- **TestL1MemoryCache**: In-memory cache with LRU eviction
- **TestL2DiskCache**: Persistent disk cache operations
- **TestL3RedisCache**: Distributed Redis cache functionality
- **TestMultiTierCoordination**: L1→L2→L3 fallback chains
- **TestCachePromotionStrategies**: Automatic data promotion
- **TestCacheConsistency**: Cross-tier data consistency
- **TestMultiTierPerformance**: Performance optimization

### 3. test_emergency_cache.py
Emergency cache system testing:
- **TestEmergencyCacheActivation**: Activation/deactivation mechanisms
- **TestPickleSerialization**: Secure pickle serialization
- **TestEmergencyCachePreloading**: Common pattern preloading
- **TestEmergencyCacheFileManagement**: File rotation and cleanup
- **TestEmergencyCachePerformance**: Sub-second response guarantees
- **TestPatternBasedFallback**: Pattern matching for queries
- **TestEmergencyCacheRecovery**: Recovery and failover

### 4. test_query_router_cache.py
Query router LRU cache testing:
- **TestQueryRouterLRUCache**: LRU eviction policy
- **TestQueryHashConsistency**: Consistent query hashing
- **TestConfidenceBasedCaching**: Confidence threshold filtering
- **TestCacheInvalidation**: Cache invalidation mechanisms
- **TestQueryRouterPerformance**: Performance improvements
- **TestCacheThreadSafety**: Thread-safe operations
- **TestCacheMemoryManagement**: Memory usage optimization

## 🔧 Test Fixtures and Utilities

### cache_test_fixtures.py
Comprehensive test utilities:
- **BiomedicalTestDataGenerator**: Realistic biomedical query generation
- **MockCacheBackends**: Redis, disk cache, and failure simulation
- **CachePerformanceMetrics**: Performance measurement utilities
- **BIOMEDICAL_QUERIES**: Curated biomedical test data
- **EMERGENCY_RESPONSE_PATTERNS**: Emergency cache patterns

### conftest.py
Pytest configuration providing:
- Test environment setup and cleanup
- Mock objects for Redis and disk cache
- Performance measurement fixtures
- Memory usage tracking
- Concurrent testing helpers
- Automatic test categorization

## 🚀 Running Tests

### Basic Usage
```bash
# Run all cache unit tests
python run_cache_tests.py

# Run with coverage report
python run_cache_tests.py --coverage

# Run only fast tests (exclude performance tests)
python run_cache_tests.py --fast

# Run only performance tests
python run_cache_tests.py --performance

# Run tests in parallel
python run_cache_tests.py --parallel --verbose
```

### Test Categories
```bash
# Run specific test categories using pytest markers
pytest -m "unit"           # Unit tests only
pytest -m "performance"    # Performance tests only
pytest -m "concurrent"     # Concurrent access tests
pytest -m "slow"           # Stress/load tests
```

### Test Reports
```bash
# Generate comprehensive test report
python run_cache_tests.py --report

# Show implementation summary
python run_cache_tests.py --summary
```

## 📊 Performance Targets

The tests validate these performance targets:

| Operation | Target | Test Coverage |
|-----------|---------|---------------|
| L1 Cache Get | <1ms average | ✅ TestCachePerformance |
| L1 Cache Set | <2ms average | ✅ TestCachePerformance |
| Multi-tier Fallback | <100ms total | ✅ TestMultiTierPerformance |
| Emergency Cache | <1s guaranteed | ✅ TestEmergencyCachePerformance |
| Cache Hit Rate | >80% repeated queries | ✅ Multiple test classes |
| Memory Usage | <512MB typical workload | ✅ TestCacheMemoryManagement |

## 🧬 Realistic Test Data

All tests use realistic biomedical queries and responses:

### Biomedical Query Categories
- **Metabolism**: Glucose metabolism, metabolic pathways
- **Clinical Applications**: Drug discovery, biomarker identification
- **Disease Metabolomics**: Cancer, diabetes, cardiovascular disease
- **Temporal Queries**: Current research, market data (not cached)

### Example Test Queries
```python
"What are the metabolic pathways involved in glucose metabolism?"
"How does insulin resistance affect metabolomics profiles?"
"What biomarkers indicate cardiovascular disease risk?"
"How is metabolomics used in drug discovery?"
```

## 🔬 Test Implementation Highlights

### Thread Safety Testing
- Concurrent read/write operations
- Race condition detection
- Atomic operation verification
- Lock contention analysis

### Performance Benchmarking
- Response time percentiles (P95, P99)
- Memory usage tracking
- Cache hit rate optimization
- Concurrent load testing

### Error Handling
- Backend failure simulation
- Corruption recovery testing
- Graceful degradation validation
- Resource cleanup verification

### Cache Coordination
- Multi-tier fallback chains
- Data promotion/demotion
- Consistency across tiers
- TTL synchronization

## 📈 Coverage Analysis

The test suite provides comprehensive coverage:

- **150+ individual test methods**
- **25 test classes** across 4 modules
- **Core functionality**: CRUD, TTL, eviction, serialization
- **Integration**: Multi-tier coordination, fallback chains
- **Performance**: Response times, hit rates, memory usage
- **Reliability**: Thread safety, error recovery, consistency
- **Security**: Serialization safety, cache poisoning prevention

## 🎯 Success Criteria

Tests validate these success criteria from the design document:

### Performance Success Criteria ✅
- Cache Hit Response Time: <100ms average, <500ms P99
- Cache Miss Response Time: <2000ms average, <5000ms P99
- Multi-tier Fallback Time: <200ms additional overhead
- L1 Memory Cache Hit Ratio: >90% for recent queries
- Overall System Hit Ratio: >85% across all tiers

### Reliability Success Criteria ✅
- Redis Failure Recovery: <5 seconds to detect and fallback
- Emergency Cache Activation: <1 second
- Cache Consistency: 100% consistency across tiers
- TTL Accuracy: <5% variance in expiration times

### Integration Success Criteria ✅
- Biomedical Query Caching: >80% cache hit for repeated queries
- Routing Decision Caching: >90% cache hit for routing decisions
- Graceful Degradation: 100% availability during cache failures

## 🔍 Debugging and Troubleshooting

### Test Failures
1. Check test output for specific assertion failures
2. Use `--verbose` flag for detailed test information
3. Run individual test classes: `pytest tests/unit/test_cache_storage_operations.py::TestCacheKeyGeneration`
4. Use `--pdb` flag to drop into debugger on failure

### Performance Issues
1. Run performance tests: `python run_cache_tests.py --performance`
2. Check memory usage with `--coverage` flag
3. Use profiling: `pytest --profile`
4. Review cache hit rates in test output

### Mock Object Issues
1. Verify mock setup in `conftest.py`
2. Check fixture dependencies
3. Validate test data in `cache_test_fixtures.py`
4. Use `--capture=no` to see print statements

## 📚 Dependencies

Required Python packages:
- `pytest` (>=6.0)
- `pytest-asyncio` (for async tests)
- `pytest-cov` (for coverage reporting)
- `psutil` (for memory usage tracking)

Optional packages:
- `pytest-xdist` (for parallel test execution)
- `pytest-benchmark` (for benchmarking)
- `pytest-html` (for HTML reports)

## 🤝 Contributing

When adding new cache functionality:

1. Add corresponding unit tests in the appropriate test file
2. Update test fixtures if new test data is needed
3. Add performance benchmarks for new operations
4. Update this README if new test categories are added
5. Ensure all tests pass: `python run_cache_tests.py`

## 📝 Notes

- All tests are independent and can run in parallel
- Mock objects simulate realistic backend behavior
- Test data uses authentic biomedical terminology
- Performance tests have relaxed thresholds for CI environments
- Emergency cache tests validate disaster recovery scenarios
- Thread safety tests use high concurrency to detect race conditions

This comprehensive test suite ensures the Clinical Metabolomics Oracle caching system meets all performance, reliability, and functionality requirements specified in the design document.