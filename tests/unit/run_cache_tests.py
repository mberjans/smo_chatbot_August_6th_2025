#!/usr/bin/env python3
"""
Test runner for cache unit tests.

This script runs all cache unit tests with proper configuration,
generates coverage reports, and provides performance metrics.

Usage:
    python run_cache_tests.py [options]

Options:
    --fast          Run only fast tests (exclude performance/stress tests)
    --performance   Run only performance tests
    --coverage      Generate coverage report
    --verbose       Verbose output
    --parallel      Run tests in parallel
    --benchmark     Include benchmark tests

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_tests(args):
    """Run cache unit tests with specified options."""
    
    # Base pytest command
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        str(Path(__file__).parent),  # Test directory
        "-v" if args.verbose else "-q",
        "--tb=short",
        "--strict-markers"
    ]
    
    # Add test selection options
    if args.fast:
        pytest_cmd.extend(["-m", "not slow and not performance"])
    elif args.performance:
        pytest_cmd.extend(["-m", "performance"])
    elif args.benchmark:
        pytest_cmd.extend(["-m", "performance or slow"])
    
    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])
    
    # Add coverage reporting
    if args.coverage:
        pytest_cmd.extend([
            "--cov=.",
            "--cov-report=html:cache_test_coverage",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    # Add benchmark plugin if available
    if args.benchmark:
        try:
            import pytest_benchmark
            pytest_cmd.extend(["--benchmark-only"])
        except ImportError:
            print("Warning: pytest-benchmark not available, skipping benchmark tests")
    
    print(f"Running command: {' '.join(pytest_cmd)}")
    print("-" * 80)
    
    # Execute tests
    result = subprocess.run(pytest_cmd, cwd=PROJECT_ROOT)
    
    return result.returncode


def generate_test_report():
    """Generate comprehensive test report."""
    report = {
        "test_run_timestamp": datetime.now().isoformat(),
        "test_modules": {
            "test_cache_storage_operations.py": {
                "description": "Core cache storage and retrieval operations",
                "test_classes": [
                    "TestCacheKeyGeneration",
                    "TestCacheDataSerialization", 
                    "TestCacheMetadata",
                    "TestCacheSizeLimits",
                    "TestCacheTTL",
                    "TestCacheThreadSafety",
                    "TestCachePerformance"
                ],
                "coverage_areas": [
                    "Cache key generation and collision handling",
                    "Data serialization/deserialization",
                    "Cache entry metadata management",
                    "Size limit enforcement",
                    "TTL expiration handling",
                    "Thread safety",
                    "Performance characteristics"
                ]
            },
            "test_multi_tier_cache.py": {
                "description": "Multi-level cache coordination and operations",
                "test_classes": [
                    "TestL1MemoryCache",
                    "TestL2DiskCache",
                    "TestL3RedisCache", 
                    "TestMultiTierCoordination",
                    "TestCachePromotionStrategies",
                    "TestCacheConsistency",
                    "TestMultiTierPerformance"
                ],
                "coverage_areas": [
                    "L1 memory cache operations",
                    "L2 disk cache persistence",
                    "L3 Redis cache distributed operations",
                    "Cross-tier coordination",
                    "Cache promotion/demotion",
                    "Data consistency across tiers",
                    "Multi-tier performance optimization"
                ]
            },
            "test_emergency_cache.py": {
                "description": "Emergency cache system for failover scenarios",
                "test_classes": [
                    "TestEmergencyCacheActivation",
                    "TestPickleSerialization",
                    "TestEmergencyCachePreloading",
                    "TestEmergencyCacheFileManagement",
                    "TestEmergencyCachePerformance",
                    "TestPatternBasedFallback",
                    "TestEmergencyCacheRecovery"
                ],
                "coverage_areas": [
                    "Emergency cache activation/deactivation",
                    "Pickle serialization security",
                    "Common pattern preloading",
                    "File management and rotation",
                    "Sub-second response guarantees",
                    "Pattern-based fallback",
                    "Recovery mechanisms"
                ]
            },
            "test_query_router_cache.py": {
                "description": "Query router LRU cache functionality",
                "test_classes": [
                    "TestQueryRouterLRUCache",
                    "TestQueryHashConsistency",
                    "TestConfidenceBasedCaching",
                    "TestCacheInvalidation",
                    "TestQueryRouterPerformance",
                    "TestCacheThreadSafety",
                    "TestCacheMemoryManagement"
                ],
                "coverage_areas": [
                    "LRU eviction policy",
                    "Consistent query hashing",
                    "Confidence-based caching decisions",
                    "Cache invalidation mechanisms",
                    "Performance improvements",
                    "Thread safety",
                    "Memory usage optimization"
                ]
            }
        },
        "test_fixtures": {
            "cache_test_fixtures.py": {
                "description": "Comprehensive test fixtures and utilities",
                "components": [
                    "BiomedicalTestDataGenerator",
                    "MockCacheBackends", 
                    "CachePerformanceMetrics",
                    "CachePerformanceMeasurer"
                ]
            },
            "conftest.py": {
                "description": "Pytest configuration and shared fixtures",
                "fixtures": [
                    "temp_cache_dir",
                    "mock_redis_client",
                    "mock_disk_cache",
                    "failing_cache_backend",
                    "performance_measurer",
                    "memory_usage_tracker"
                ]
            }
        },
        "test_coverage_summary": {
            "total_test_files": 4,
            "total_test_classes": 25,
            "estimated_test_methods": 150,
            "coverage_areas": [
                "Core cache operations (CRUD, TTL, eviction)",
                "Multi-tier cache coordination",
                "Emergency failover systems",
                "Query router caching",
                "Performance optimization",
                "Thread safety and concurrency",
                "Memory management",
                "Error handling and recovery",
                "Realistic biomedical data scenarios"
            ]
        },
        "performance_targets": {
            "cache_get_operation": "<1ms average",
            "cache_set_operation": "<2ms average", 
            "multi_tier_fallback": "<100ms total",
            "emergency_cache_response": "<1s guaranteed",
            "overall_hit_rate": ">80% for repeated queries",
            "memory_usage": "<512MB for typical workload"
        }
    }
    
    # Save report
    report_file = PROJECT_ROOT / "cache_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTest report generated: {report_file}")
    return report


def print_test_summary():
    """Print test implementation summary."""
    print("\n" + "="*80)
    print("CACHE UNIT TESTS IMPLEMENTATION SUMMARY")
    print("="*80)
    
    print("""
✅ IMPLEMENTED TEST FILES:
   
   1. test_cache_storage_operations.py
      - Core cache CRUD operations
      - Cache key generation and collision handling
      - Data serialization/deserialization (JSON + Pickle fallback)
      - Cache metadata management
      - Size limit enforcement with LRU eviction
      - TTL expiration handling
      - Thread safety under concurrent access
      - Performance characteristics and benchmarks
      
   2. test_multi_tier_cache.py
      - L1 memory cache with LRU eviction
      - L2 disk cache with persistence
      - L3 Redis cache with distributed operations
      - Cross-tier coordination and fallback chains
      - Cache promotion/demotion strategies
      - Data consistency across tiers
      - Multi-tier performance optimization
      
   3. test_emergency_cache.py
      - Emergency cache activation/deactivation
      - Pickle-based serialization with security
      - Common pattern preloading
      - File management and rotation
      - Sub-second response guarantees
      - Pattern-based fallback matching
      - Recovery and failover mechanisms
      
   4. test_query_router_cache.py
      - LRU cache for routing decisions
      - Consistent query hashing
      - Confidence-based caching decisions
      - Cache invalidation on logic updates
      - Performance impact measurement
      - Thread safety and memory management

✅ COMPREHENSIVE TEST FIXTURES:
   
   - cache_test_fixtures.py: Realistic biomedical test data
   - conftest.py: Pytest configuration and shared fixtures
   - Mock backends for Redis, disk cache, and failure simulation
   - Performance measurement utilities
   - Concurrent testing helpers
   
✅ TEST COVERAGE HIGHLIGHTS:
   
   - 150+ individual test methods
   - 25 test classes across 4 modules
   - Realistic biomedical query scenarios
   - Performance benchmarks with targets
   - Thread safety and concurrent access
   - Error handling and recovery scenarios
   - Memory usage optimization
   - Multi-tier cache coordination
   
✅ KEY FEATURES TESTED:
   
   - Cache hit rates >80% for repeated queries
   - Response times <1ms for memory cache
   - Sub-second emergency cache responses
   - Thread-safe concurrent operations
   - Proper TTL expiration (5min - 24hr)
   - LRU eviction policies
   - Cross-tier data consistency
   - Graceful failure handling
    """)
    
    print("="*80)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run cache unit tests")
    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--benchmark", action="store_true", help="Include benchmark tests")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--summary", action="store_true", help="Show implementation summary")
    
    args = parser.parse_args()
    
    # Show summary if requested
    if args.summary:
        print_test_summary()
        return 0
    
    # Generate report if requested
    if args.report:
        report = generate_test_report()
        print(f"Generated comprehensive test report")
        return 0
    
    # Run tests
    exit_code = run_tests(args)
    
    if exit_code == 0:
        print("\n✅ All cache unit tests passed successfully!")
        if args.coverage:
            print("📊 Coverage report generated in cache_test_coverage/")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())