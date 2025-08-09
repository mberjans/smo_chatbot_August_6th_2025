#!/usr/bin/env python3
"""
Comprehensive test runner for cache invalidation strategies in the Clinical Metabolomics Oracle system.

This script runs all cache invalidation tests including unit tests, integration tests,
and performance tests with realistic biomedical query processing patterns. It provides
comprehensive validation of invalidation mechanisms and generates detailed reports.

Usage:
    python run_cache_invalidation_tests.py [--test-type] [--report-format] [--output-dir]

Test Coverage:
- Unit tests: Core invalidation mechanism testing
- Integration tests: Multi-tier invalidation coordination
- Performance tests: Performance impact and optimization
- End-to-end tests: Realistic biomedical query processing scenarios
- Stress tests: High-load invalidation scenarios
- Edge case tests: Boundary conditions and error handling

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import sys
import os
import argparse
import time
import json
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Add test directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'unit'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'integration'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'unit', 'performance'))

# Import test components
from cache_test_fixtures import (
    BIOMEDICAL_QUERIES,
    BiomedicalTestDataGenerator,
    PERFORMANCE_TEST_QUERIES,
    generate_cache_test_scenarios
)

from test_cache_invalidation import (
    MockInvalidatingCache,
    InvalidationEvent,
    InvalidationRule,
    INVALIDATION_STRATEGIES,
    INVALIDATION_TRIGGERS,
    INVALIDATION_POLICIES
)

from test_invalidation_coordination import (
    MultiTierCacheSystem,
    TierConfiguration,
    InvalidationCoordinationEvent
)

from test_invalidation_performance import (
    InvalidationPerformanceTester,
    PerformanceMetrics,
    PerformanceProfiler
)


@dataclass
class TestRunConfiguration:
    """Configuration for test run execution."""
    test_types: List[str] = field(default_factory=lambda: ['unit', 'integration', 'performance'])
    include_stress_tests: bool = False
    include_edge_case_tests: bool = True
    parallel_execution: bool = True
    report_format: str = 'json'  # json, html, markdown
    output_directory: str = 'test_results'
    log_level: str = 'INFO'
    test_timeout_seconds: int = 300
    performance_iterations: int = 100


@dataclass
class TestResult:
    """Results from a test execution."""
    test_name: str
    test_type: str
    status: str  # passed, failed, skipped
    execution_time_seconds: float
    assertions_passed: int
    assertions_failed: int
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestRunReport:
    """Comprehensive test run report."""
    run_id: str
    start_time: datetime
    end_time: datetime
    configuration: TestRunConfiguration
    test_results: List[TestResult]
    summary_statistics: Dict[str, Any]
    performance_summary: Dict[str, Any]
    recommendations: List[str]


class InvalidationTestRunner:
    """Comprehensive test runner for cache invalidation strategies."""
    
    def __init__(self, config: TestRunConfiguration):
        self.config = config
        self.data_generator = BiomedicalTestDataGenerator()
        self.performance_tester = InvalidationPerformanceTester()
        self.test_results: List[TestResult] = []
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(config.output_directory, exist_ok=True)
    
    def run_all_tests(self) -> TestRunReport:
        """Run comprehensive invalidation test suite."""
        run_id = f"cache_invalidation_test_run_{int(time.time())}"
        start_time = datetime.now()
        
        self.logger.info(f"Starting cache invalidation test run: {run_id}")
        
        try:
            # Run different test categories
            if 'unit' in self.config.test_types:
                self._run_unit_tests()
            
            if 'integration' in self.config.test_types:
                self._run_integration_tests()
            
            if 'performance' in self.config.test_types:
                self._run_performance_tests()
            
            if self.config.include_edge_case_tests:
                self._run_edge_case_tests()
            
            if self.config.include_stress_tests:
                self._run_stress_tests()
            
            # Run biomedical scenario tests
            self._run_biomedical_scenario_tests()
            
        except Exception as e:
            self.logger.error(f"Test run failed: {str(e)}")
            
        end_time = datetime.now()
        
        # Generate comprehensive report
        report = TestRunReport(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            configuration=self.config,
            test_results=self.test_results,
            summary_statistics=self._calculate_summary_statistics(),
            performance_summary=self._calculate_performance_summary(),
            recommendations=self._generate_recommendations()
        )
        
        self._save_report(report)
        self._log_summary(report)
        
        return report
    
    def _run_unit_tests(self):
        """Run unit tests for core invalidation mechanisms."""
        self.logger.info("Running unit tests for cache invalidation")
        
        # Test basic invalidation triggers
        self._test_invalidation_triggers()
        
        # Test manual invalidation
        self._test_manual_invalidation()
        
        # Test pattern-based invalidation
        self._test_pattern_based_invalidation()
        
        # Test access-based invalidation
        self._test_access_based_invalidation()
        
        # Test invalidation strategies
        self._test_invalidation_strategies()
    
    def _test_invalidation_triggers(self):
        """Test different invalidation trigger mechanisms."""
        cache = MockInvalidatingCache(max_size=100, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Time-based invalidation test
            cache.set("time_test", "value", ttl=1)
            time.sleep(1.1)
            result = cache.get("time_test")
            if result is None:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Size-based invalidation test
            for i in range(120):  # Overfill cache
                cache.set(f"size_test_{i}", f"value_{i}")
            
            if len(cache.storage) <= cache.max_size:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Manual invalidation test
            cache.set("manual_test", "value")
            success = cache.invalidate("manual_test", "Manual test")
            if success and cache.get("manual_test") is None:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="invalidation_triggers",
                test_type="unit",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="invalidation_triggers",
                test_type="unit",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_manual_invalidation(self):
        """Test manual invalidation operations."""
        cache = MockInvalidatingCache()
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Single entry invalidation
            cache.set("single_test", "value")
            success = cache.invalidate("single_test", "Single test")
            if success and cache.get("single_test") is None:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Bulk invalidation
            queries = ["bulk_1", "bulk_2", "bulk_3"]
            for query in queries:
                cache.set(query, f"value_{query}")
            
            invalidated = cache.bulk_invalidate(queries, "Bulk test")
            if invalidated == len(queries):
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="manual_invalidation",
                test_type="unit",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="manual_invalidation",
                test_type="unit",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_pattern_based_invalidation(self):
        """Test pattern-based invalidation mechanisms."""
        cache = MockInvalidatingCache()
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Pattern matching test
            test_queries = [
                "diabetes research query",
                "cancer treatment query", 
                "diabetes medication query",
                "heart disease query"
            ]
            
            for query in test_queries:
                cache.set(query, f"response for {query}")
            
            # Invalidate diabetes-related queries
            diabetes_invalidated = cache.invalidate_by_pattern("diabetes")
            if diabetes_invalidated == 2:  # Two diabetes queries
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Verify non-diabetes queries remain
            remaining = sum(1 for query in test_queries 
                          if "diabetes" not in query and cache.get(query) is not None)
            if remaining == 2:  # Cancer and heart disease queries
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="pattern_based_invalidation",
                test_type="unit",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="pattern_based_invalidation",
                test_type="unit",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_access_based_invalidation(self):
        """Test access pattern and confidence-based invalidation."""
        cache = MockInvalidatingCache()
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Confidence-based invalidation
            confidence_entries = [
                ("high_conf", "high_data", 0.95),
                ("med_conf", "med_data", 0.75), 
                ("low_conf", "low_data", 0.45)
            ]
            
            for query, value, confidence in confidence_entries:
                cache.set(query, value, confidence=confidence)
            
            # Invalidate low confidence entries
            low_conf_invalidated = cache.invalidate_by_confidence(0.7)
            if low_conf_invalidated == 1:  # One low confidence entry
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Access count-based invalidation
            access_queries = ["accessed", "not_accessed"]
            for query in access_queries:
                cache.set(query, f"value_{query}")
            
            # Access one query multiple times
            for _ in range(5):
                cache.get("accessed")
            
            # Invalidate low-access entries
            low_access_invalidated = cache.invalidate_by_access_count(3)
            if low_access_invalidated == 1:  # "not_accessed" should be invalidated
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="access_based_invalidation",
                test_type="unit",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="access_based_invalidation",
                test_type="unit",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_invalidation_strategies(self):
        """Test different invalidation strategies."""
        strategies = [
            INVALIDATION_STRATEGIES['IMMEDIATE'],
            INVALIDATION_STRATEGIES['DEFERRED'],
            INVALIDATION_STRATEGIES['BATCH']
        ]
        
        for strategy in strategies:
            cache = MockInvalidatingCache(invalidation_strategy=strategy)
            
            test_start = time.time()
            assertions_passed = 0
            assertions_failed = 0
            
            try:
                # Configure for deferred testing if needed
                if strategy == INVALIDATION_STRATEGIES['DEFERRED']:
                    rule = InvalidationRule(
                        rule_id="deferred_test",
                        trigger=INVALIDATION_TRIGGERS['MANUAL'],
                        condition="tag:defer",
                        action="defer",
                        priority=100
                    )
                    cache.add_invalidation_rule(rule)
                
                # Test strategy behavior
                cache.set("strategy_test", "value", tags=["defer"] if strategy == INVALIDATION_STRATEGIES['DEFERRED'] else [])
                
                if strategy == INVALIDATION_STRATEGIES['IMMEDIATE']:
                    success = cache.invalidate("strategy_test", "Strategy test")
                    if success and cache.get("strategy_test") is None:
                        assertions_passed += 1
                    else:
                        assertions_failed += 1
                
                elif strategy == INVALIDATION_STRATEGIES['DEFERRED']:
                    cache.invalidate("strategy_test", "Deferred test")
                    # Should still be accessible
                    if cache.get("strategy_test") is not None:
                        assertions_passed += 1
                    else:
                        assertions_failed += 1
                    
                    # Process deferred
                    processed = cache.process_deferred_invalidations()
                    if processed > 0 and cache.get("strategy_test") is None:
                        assertions_passed += 1
                    else:
                        assertions_failed += 1
                
                elif strategy == INVALIDATION_STRATEGIES['BATCH']:
                    queries = [f"batch_{i}" for i in range(10)]
                    for query in queries:
                        cache.set(query, f"value_{query}")
                    
                    invalidated = cache.bulk_invalidate(queries, "Batch strategy test")
                    if invalidated == len(queries):
                        assertions_passed += 1
                    else:
                        assertions_failed += 1
                
                test_result = TestResult(
                    test_name=f"invalidation_strategy_{strategy}",
                    test_type="unit",
                    status="passed" if assertions_failed == 0 else "failed",
                    execution_time_seconds=time.time() - test_start,
                    assertions_passed=assertions_passed,
                    assertions_failed=assertions_failed
                )
                
            except Exception as e:
                test_result = TestResult(
                    test_name=f"invalidation_strategy_{strategy}",
                    test_type="unit",
                    status="failed",
                    execution_time_seconds=time.time() - test_start,
                    assertions_passed=assertions_passed,
                    assertions_failed=assertions_failed + 1,
                    error_message=str(e)
                )
            
            self.test_results.append(test_result)
    
    def _run_integration_tests(self):
        """Run integration tests for multi-tier invalidation coordination."""
        self.logger.info("Running integration tests for multi-tier invalidation coordination")
        
        # Test cascading invalidation
        self._test_cascading_invalidation()
        
        # Test selective invalidation
        self._test_selective_invalidation()
        
        # Test distributed coordination
        self._test_distributed_coordination()
        
        # Test consistency maintenance
        self._test_consistency_maintenance()
    
    def _test_cascading_invalidation(self):
        """Test cascading invalidation across cache tiers."""
        tier_configs = {
            'L1': TierConfiguration('L1', 50, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
            'L2': TierConfiguration('L2', 200, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),
            'L3': TierConfiguration('L3', 500, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3)
        }
        
        cache_system = MultiTierCacheSystem(tier_configs)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Set entry across all tiers
            query = "cascading test query"
            value = "cascading test value"
            
            set_results = cache_system.set_across_tiers(query, value)
            if len(set_results) == 3:  # All tiers
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Perform cascading invalidation
            event = cache_system.invalidate_across_tiers(
                query, strategy="cascading", reason="Integration test"
            )
            
            # Verify all tiers invalidated
            if all(event.success_by_tier.values()):
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Verify entries are gone from all tiers
            invalidated_count = 0
            for tier_name in tier_configs.keys():
                if cache_system.tiers[tier_name].get(query) is None:
                    invalidated_count += 1
            
            if invalidated_count == len(tier_configs):
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="cascading_invalidation_integration",
                test_type="integration",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="cascading_invalidation_integration",
                test_type="integration",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_selective_invalidation(self):
        """Test selective tier invalidation."""
        tier_configs = {
            'L1': TierConfiguration('L1', 50, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
            'L2': TierConfiguration('L2', 200, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),
            'L3': TierConfiguration('L3', 500, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3)
        }
        
        cache_system = MultiTierCacheSystem(tier_configs)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            query = "selective test query"
            value = "selective test value"
            
            # Set across all tiers
            cache_system.set_across_tiers(query, value)
            
            # Selectively invalidate L1 only
            event = cache_system.invalidate_across_tiers(
                query, tiers=['L1'], strategy="selective", reason="Selective test"
            )
            
            # Verify only L1 invalidated
            if event.target_tiers == ['L1'] and event.success_by_tier['L1']:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Verify L1 empty, others still have data
            l1_empty = cache_system.tiers['L1'].get(query) is None
            l2_has_data = cache_system.tiers['L2'].get(query) is not None
            l3_has_data = cache_system.tiers['L3'].get(query) is not None
            
            if l1_empty and l2_has_data and l3_has_data:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="selective_invalidation_integration",
                test_type="integration",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="selective_invalidation_integration",
                test_type="integration",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_distributed_coordination(self):
        """Test distributed invalidation coordination."""
        # Create multiple cache system nodes
        node_configs = {
            'L1': TierConfiguration('L1', 50, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
            'L2': TierConfiguration('L2', 200, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2)
        }
        
        nodes = {}
        for i in range(3):
            nodes[f"node_{i}"] = MultiTierCacheSystem(node_configs.copy())
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            query = "distributed test query"
            value = "distributed test value"
            
            # Set in all nodes
            for node_system in nodes.values():
                node_system.set_across_tiers(query, value)
            
            # Parallel invalidation across all nodes
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def invalidate_node(node_system):
                return node_system.invalidate_across_tiers(
                    query, strategy="parallel", reason="Distributed test"
                )
            
            with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
                futures = [executor.submit(invalidate_node, node_system) 
                          for node_system in nodes.values()]
                results = [future.result() for future in as_completed(futures)]
            
            # Verify all nodes completed invalidation
            if len(results) == len(nodes):
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Verify all invalidations succeeded
            successful_invalidations = sum(1 for result in results 
                                         if all(result.success_by_tier.values()))
            
            if successful_invalidations == len(nodes):
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="distributed_coordination_integration",
                test_type="integration",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="distributed_coordination_integration",
                test_type="integration",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_consistency_maintenance(self):
        """Test consistency maintenance across tiers."""
        tier_configs = {
            'L1': TierConfiguration('L1', 50, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
            'L2': TierConfiguration('L2', 200, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),
            'L3': TierConfiguration('L3', 500, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3)
        }
        
        cache_system = MultiTierCacheSystem(tier_configs)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            query = "consistency test query"
            value = "consistency test value"
            
            # Set across all tiers
            cache_system.set_across_tiers(query, value)
            
            # Check initial consistency
            consistency_report = cache_system.check_consistency_across_tiers(query)
            if consistency_report['consistent']:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Invalidate across all tiers
            event = cache_system.invalidate_across_tiers(
                query, strategy="cascading", reason="Consistency test"
            )
            
            # Check post-invalidation consistency (all should be empty)
            post_consistency_report = cache_system.check_consistency_across_tiers(query)
            
            # All tiers should consistently show no entry
            all_empty = all(not state['has_entry'] 
                          for state in post_consistency_report['tier_states'].values())
            
            if all_empty:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="consistency_maintenance_integration",
                test_type="integration",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="consistency_maintenance_integration",
                test_type="integration",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _run_performance_tests(self):
        """Run performance tests for invalidation operations."""
        self.logger.info("Running performance tests for invalidation operations")
        
        # Test strategy performance
        self._test_strategy_performance()
        
        # Test throughput impact
        self._test_throughput_impact()
        
        # Test scalability
        self._test_scalability()
    
    def _test_strategy_performance(self):
        """Test performance of different invalidation strategies."""
        cache = MockInvalidatingCache(max_size=500, default_ttl=3600)
        
        strategies = [
            INVALIDATION_STRATEGIES['IMMEDIATE'],
            INVALIDATION_STRATEGIES['BATCH']
        ]
        
        for strategy in strategies:
            test_start = time.time()
            
            try:
                metrics = self.performance_tester.benchmark_invalidation_strategy(
                    cache, strategy, operations_count=self.config.performance_iterations
                )
                
                # Verify performance targets
                performance_ok = (
                    metrics.avg_time_ms < 100 and
                    metrics.operations_per_second > 10 and
                    metrics.success_rate > 0.9
                )
                
                test_result = TestResult(
                    test_name=f"strategy_performance_{strategy}",
                    test_type="performance",
                    status="passed" if performance_ok else "failed",
                    execution_time_seconds=time.time() - test_start,
                    assertions_passed=1 if performance_ok else 0,
                    assertions_failed=0 if performance_ok else 1,
                    performance_metrics=metrics.to_dict()
                )
                
            except Exception as e:
                test_result = TestResult(
                    test_name=f"strategy_performance_{strategy}",
                    test_type="performance",
                    status="failed",
                    execution_time_seconds=time.time() - test_start,
                    assertions_passed=0,
                    assertions_failed=1,
                    error_message=str(e)
                )
            
            self.test_results.append(test_result)
    
    def _test_throughput_impact(self):
        """Test throughput impact of invalidation operations."""
        cache = MockInvalidatingCache(max_size=200, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Measure baseline throughput
            baseline_queries = [f"baseline_{i}" for i in range(100)]
            for query in baseline_queries:
                cache.set(query, f"value_{query}")
            
            baseline_start = time.time()
            for query in baseline_queries:
                cache.get(query)
            baseline_time = time.time() - baseline_start
            baseline_ops_per_sec = len(baseline_queries) / baseline_time
            
            # Measure throughput with invalidations
            invalidation_queries = [f"invalidation_{i}" for i in range(100)]
            for query in invalidation_queries:
                cache.set(query, f"value_{query}")
            
            mixed_start = time.time()
            for i, query in enumerate(invalidation_queries):
                if i % 5 == 0:  # 20% invalidation rate
                    cache.invalidate(query, "Throughput test")
                else:
                    cache.get(query)
            mixed_time = time.time() - mixed_start
            mixed_ops_per_sec = len(invalidation_queries) / mixed_time
            
            # Throughput should remain reasonable with invalidations
            throughput_ratio = mixed_ops_per_sec / baseline_ops_per_sec
            if throughput_ratio > 0.7:  # At least 70% of baseline
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="throughput_impact_performance",
                test_type="performance",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                metadata={
                    'baseline_ops_per_sec': baseline_ops_per_sec,
                    'mixed_ops_per_sec': mixed_ops_per_sec,
                    'throughput_ratio': throughput_ratio
                }
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="throughput_impact_performance",
                test_type="performance",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_scalability(self):
        """Test invalidation scalability."""
        cache_sizes = [100, 500, 1000]
        
        for cache_size in cache_sizes:
            cache = MockInvalidatingCache(max_size=cache_size, default_ttl=3600)
            
            test_start = time.time()
            
            try:
                # Fill cache
                for i in range(cache_size):
                    cache.set(f"scale_test_{i}", f"value_{i}")
                
                # Measure bulk invalidation performance
                queries_to_invalidate = [f"scale_test_{i}" for i in range(cache_size // 2)]
                
                invalidation_start = time.time()
                invalidated = cache.bulk_invalidate(queries_to_invalidate, f"Scale test {cache_size}")
                invalidation_time = time.time() - invalidation_start
                
                entries_per_second = invalidated / invalidation_time
                
                # Performance should scale reasonably
                performance_ok = entries_per_second > 50  # At least 50 entries/sec
                
                test_result = TestResult(
                    test_name=f"scalability_performance_{cache_size}",
                    test_type="performance",
                    status="passed" if performance_ok else "failed",
                    execution_time_seconds=time.time() - test_start,
                    assertions_passed=1 if performance_ok else 0,
                    assertions_failed=0 if performance_ok else 1,
                    metadata={
                        'cache_size': cache_size,
                        'entries_per_second': entries_per_second,
                        'invalidation_time': invalidation_time
                    }
                )
                
            except Exception as e:
                test_result = TestResult(
                    test_name=f"scalability_performance_{cache_size}",
                    test_type="performance",
                    status="failed",
                    execution_time_seconds=time.time() - test_start,
                    assertions_passed=0,
                    assertions_failed=1,
                    error_message=str(e)
                )
            
            self.test_results.append(test_result)
    
    def _run_edge_case_tests(self):
        """Run edge case and boundary condition tests."""
        self.logger.info("Running edge case tests")
        
        # Test concurrent invalidation
        self._test_concurrent_invalidation()
        
        # Test malformed patterns
        self._test_malformed_patterns()
        
        # Test memory pressure scenarios
        self._test_memory_pressure()
    
    def _test_concurrent_invalidation(self):
        """Test concurrent invalidation operations."""
        cache = MockInvalidatingCache(max_size=100, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Pre-populate cache
            queries = [f"concurrent_test_{i}" for i in range(50)]
            for query in queries:
                cache.set(query, f"value_{query}")
            
            from concurrent.futures import ThreadPoolExecutor
            import threading
            
            def invalidate_worker(query_subset):
                results = []
                for query in query_subset:
                    success = cache.invalidate(query, "Concurrent test")
                    results.append(success)
                return results
            
            # Split queries for concurrent processing
            mid_point = len(queries) // 2
            query_set_1 = queries[:mid_point]
            query_set_2 = queries[mid_point:]
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(invalidate_worker, query_set_1)
                future2 = executor.submit(invalidate_worker, query_set_2)
                
                results1 = future1.result()
                results2 = future2.result()
            
            # Most invalidations should succeed despite concurrency
            total_successes = sum(results1) + sum(results2)
            success_rate = total_successes / len(queries)
            
            if success_rate > 0.8:  # At least 80% success
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="concurrent_invalidation_edge_case",
                test_type="edge_case",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                metadata={'success_rate': success_rate}
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="concurrent_invalidation_edge_case",
                test_type="edge_case",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_malformed_patterns(self):
        """Test invalidation with malformed regex patterns."""
        cache = MockInvalidatingCache()
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Set test entries
            cache.set("test_query_1", "value_1")
            cache.set("test_query_2", "value_2")
            
            # Test malformed patterns
            malformed_patterns = ["[unclosed", "(?P<incomplete", "*invalid"]
            
            for pattern in malformed_patterns:
                try:
                    invalidated = cache.invalidate_by_pattern(pattern)
                    # Should not crash and should return 0
                    if invalidated == 0:
                        assertions_passed += 1
                    else:
                        assertions_failed += 1
                except Exception:
                    # Exception is acceptable for malformed patterns
                    assertions_passed += 1
            
            # Original entries should still be present
            if cache.get("test_query_1") is not None and cache.get("test_query_2") is not None:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="malformed_patterns_edge_case",
                test_type="edge_case",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="malformed_patterns_edge_case",
                test_type="edge_case",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_memory_pressure(self):
        """Test invalidation under memory pressure."""
        cache = MockInvalidatingCache(max_size=1000, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Create memory pressure by filling cache with large objects
            large_data = "x" * 10000  # 10KB per entry
            
            for i in range(500):
                cache.set(f"memory_test_{i}", large_data)
            
            # Perform invalidation under memory pressure
            invalidation_start = time.time()
            queries_to_invalidate = [f"memory_test_{i}" for i in range(250)]  # Half
            invalidated = cache.bulk_invalidate(queries_to_invalidate, "Memory pressure test")
            invalidation_time = time.time() - invalidation_start
            
            # Should complete within reasonable time despite memory pressure
            if invalidation_time < 10.0:  # 10 seconds max
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Should invalidate requested entries
            if invalidated == len(queries_to_invalidate):
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="memory_pressure_edge_case",
                test_type="edge_case",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                metadata={'invalidation_time': invalidation_time}
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="memory_pressure_edge_case",
                test_type="edge_case",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _run_stress_tests(self):
        """Run stress tests under high load conditions."""
        self.logger.info("Running stress tests")
        
        # High-frequency invalidation stress test
        self._test_high_frequency_invalidation_stress()
        
        # Large-scale bulk invalidation stress test
        self._test_large_scale_bulk_invalidation_stress()
    
    def _test_high_frequency_invalidation_stress(self):
        """Test high-frequency invalidation operations."""
        cache = MockInvalidatingCache(max_size=500, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Pre-populate cache
            for i in range(400):
                cache.set(f"stress_test_{i}", f"value_{i}")
            
            # High-frequency invalidation
            operations = 0
            successful_operations = 0
            stress_duration = 10  # 10 seconds of stress
            
            stress_start = time.time()
            while time.time() - stress_start < stress_duration:
                query_id = operations % 400
                query = f"stress_test_{query_id}"
                
                if operations % 2 == 0:
                    # Invalidate
                    success = cache.invalidate(query, f"Stress test {operations}")
                    if success:
                        successful_operations += 1
                else:
                    # Re-add
                    cache.set(query, f"value_{query_id}_refresh")
                
                operations += 1
            
            ops_per_second = operations / stress_duration
            success_rate = successful_operations / (operations // 2)  # Only invalidation ops
            
            # Should maintain reasonable performance under stress
            if ops_per_second > 50:  # At least 50 ops/sec
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            if success_rate > 0.8:  # At least 80% success rate
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="high_frequency_invalidation_stress",
                test_type="stress",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                metadata={
                    'operations_per_second': ops_per_second,
                    'success_rate': success_rate,
                    'total_operations': operations
                }
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="high_frequency_invalidation_stress",
                test_type="stress",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_large_scale_bulk_invalidation_stress(self):
        """Test large-scale bulk invalidation operations."""
        cache = MockInvalidatingCache(max_size=5000, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Fill large cache
            large_dataset_size = 2000
            queries = [f"bulk_stress_{i}" for i in range(large_dataset_size)]
            
            for query in queries:
                cache.set(query, f"value_{query}")
            
            # Large bulk invalidation
            batch_sizes = [100, 500, 1000]
            
            for batch_size in batch_sizes:
                batch_queries = queries[:batch_size]
                
                batch_start = time.time()
                invalidated = cache.bulk_invalidate(batch_queries, f"Bulk stress {batch_size}")
                batch_time = time.time() - batch_start
                
                # Should complete efficiently
                entries_per_second = invalidated / batch_time
                
                if entries_per_second > 100:  # At least 100 entries/sec
                    assertions_passed += 1
                else:
                    assertions_failed += 1
                
                # Should invalidate all requested entries
                if invalidated == batch_size:
                    assertions_passed += 1
                else:
                    assertions_failed += 1
                
                # Re-populate for next test
                for query in batch_queries:
                    cache.set(query, f"value_{query}_refresh")
            
            test_result = TestResult(
                test_name="large_scale_bulk_invalidation_stress",
                test_type="stress",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="large_scale_bulk_invalidation_stress",
                test_type="stress",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _run_biomedical_scenario_tests(self):
        """Run comprehensive tests with realistic biomedical query processing patterns."""
        self.logger.info("Running biomedical scenario tests")
        
        # Test research data invalidation patterns
        self._test_research_data_invalidation()
        
        # Test clinical data update scenarios
        self._test_clinical_data_updates()
        
        # Test temporal query invalidation
        self._test_temporal_query_invalidation()
        
        # Test confidence-based biomedical invalidation
        self._test_confidence_based_biomedical_invalidation()
    
    def _test_research_data_invalidation(self):
        """Test invalidation patterns for research data updates."""
        cache = MockInvalidatingCache(max_size=200, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Load biomedical research data
            research_categories = ['metabolism', 'clinical_applications', 'disease_metabolomics']
            
            for category in research_categories:
                if category in BIOMEDICAL_QUERIES:
                    for query_data in BIOMEDICAL_QUERIES[category]:
                        query = query_data['query']
                        response = query_data['response']
                        confidence = response.get('confidence', 0.9)
                        
                        cache.set(query, response, confidence=confidence, tags=[category])
            
            # Simulate research area update (invalidate metabolism research)
            metabolism_invalidated = cache.invalidate_by_tags(['metabolism'])
            
            if metabolism_invalidated > 0:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Verify other categories remain
            clinical_remaining = sum(1 for query_data in BIOMEDICAL_QUERIES.get('clinical_applications', [])
                                   if cache.get(query_data['query']) is not None)
            
            if clinical_remaining > 0:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Simulate confidence-based invalidation (research validation)
            low_conf_invalidated = cache.invalidate_by_confidence(0.8)
            
            # Should maintain reasonable cache state
            final_stats = cache.get_invalidation_statistics()
            if final_stats['cache_size'] > 0:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="research_data_invalidation_biomedical",
                test_type="biomedical_scenario",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                metadata={
                    'metabolism_invalidated': metabolism_invalidated,
                    'low_confidence_invalidated': low_conf_invalidated,
                    'final_cache_size': final_stats['cache_size']
                }
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="research_data_invalidation_biomedical",
                test_type="biomedical_scenario",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_clinical_data_updates(self):
        """Test clinical data update invalidation scenarios."""
        cache = MockInvalidatingCache(max_size=150, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Simulate clinical metabolomics data
            clinical_queries = [
                ("What biomarkers indicate cardiovascular disease?", "CVD biomarkers", 0.95, ["clinical", "cardiology"]),
                ("How is diabetes diagnosed using metabolomics?", "Diabetes diagnosis", 0.90, ["clinical", "endocrinology"]),
                ("What are cancer metabolic signatures?", "Cancer signatures", 0.85, ["clinical", "oncology"]),
                ("How does drug metabolism affect treatment?", "Drug metabolism", 0.80, ["clinical", "pharmacology"])
            ]
            
            # Load clinical data
            for query, response, confidence, tags in clinical_queries:
                cache.set(query, response, confidence=confidence, tags=tags)
            
            # Simulate clinical guideline update (cardiology)
            cardiology_updated = cache.invalidate_by_tags(['cardiology'])
            
            if cardiology_updated == 1:  # One cardiology entry
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Simulate drug approval update (pharmacology)
            pharmacology_updated = cache.invalidate_by_pattern("drug")
            
            if pharmacology_updated == 1:  # One drug-related entry
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Verify other clinical areas unaffected
            remaining_clinical = sum(1 for query, _, _, _ in clinical_queries
                                   if cache.get(query) is not None)
            
            if remaining_clinical == 2:  # Two entries should remain
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="clinical_data_updates_biomedical",
                test_type="biomedical_scenario",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                metadata={
                    'cardiology_updated': cardiology_updated,
                    'pharmacology_updated': pharmacology_updated,
                    'remaining_clinical': remaining_clinical
                }
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="clinical_data_updates_biomedical",
                test_type="biomedical_scenario",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_temporal_query_invalidation(self):
        """Test invalidation of temporal/current queries."""
        cache = MockInvalidatingCache(max_size=100, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Mix of temporal and non-temporal queries
            temporal_queries = [
                ("Latest COVID-19 research 2024", "Current findings", ["temporal", "current"]),
                ("Recent drug approvals 2024", "New approvals", ["temporal", "current"]),
                ("Current clinical trials metabolomics", "Active trials", ["temporal", "current"])
            ]
            
            non_temporal_queries = [
                ("What is glucose metabolism?", "Stable knowledge", ["reference"]),
                ("How does ATP synthesis work?", "Biochemical process", ["reference"]),
                ("What are amino acid structures?", "Chemical structures", ["reference"])
            ]
            
            # Load both types
            for query, response, tags in temporal_queries + non_temporal_queries:
                cache.set(query, response, tags=tags)
            
            # Simulate temporal data invalidation (news update)
            temporal_invalidated = cache.invalidate_by_tags(['temporal'])
            
            if temporal_invalidated == len(temporal_queries):
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Verify reference data unaffected
            reference_remaining = sum(1 for query, _, _ in non_temporal_queries
                                    if cache.get(query) is not None)
            
            if reference_remaining == len(non_temporal_queries):
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Test pattern-based temporal invalidation
            cache.set("Latest diabetes research 2024", "Recent findings", tags=['temporal'])
            cache.set("Historical diabetes data", "Past research", tags=['historical'])
            
            recent_invalidated = cache.invalidate_by_pattern("Latest.*2024")
            
            if recent_invalidated == 1 and cache.get("Historical diabetes data") is not None:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="temporal_query_invalidation_biomedical",
                test_type="biomedical_scenario",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                metadata={
                    'temporal_invalidated': temporal_invalidated,
                    'reference_remaining': reference_remaining,
                    'recent_invalidated': recent_invalidated
                }
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="temporal_query_invalidation_biomedical",
                test_type="biomedical_scenario",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _test_confidence_based_biomedical_invalidation(self):
        """Test confidence-based invalidation for biomedical queries."""
        cache = MockInvalidatingCache(max_size=100, default_ttl=3600)
        
        test_start = time.time()
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Biomedical queries with varying confidence levels
            biomedical_confidence_data = [
                ("What is the structure of glucose?", "Well-established", 0.98, "established_fact"),
                ("How does metformin work?", "Well-understood mechanism", 0.92, "established_mechanism"),
                ("What causes Type 1 diabetes?", "Autoimmune destruction", 0.88, "well_supported"),
                ("How effective is new drug X?", "Limited clinical data", 0.65, "emerging_evidence"),
                ("What are biomarkers for rare disease Y?", "Preliminary research", 0.45, "speculative")
            ]
            
            # Load data with confidence levels
            for query, response, confidence, category in biomedical_confidence_data:
                cache.set(query, response, confidence=confidence, tags=[category])
            
            # Invalidate speculative research (low confidence)
            speculative_invalidated = cache.invalidate_by_confidence(0.7)
            
            if speculative_invalidated == 2:  # Two low-confidence entries
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Verify high-confidence entries remain
            high_confidence_remaining = sum(1 for query, _, confidence, _ in biomedical_confidence_data
                                          if confidence >= 0.7 and cache.get(query) is not None)
            
            if high_confidence_remaining == 3:  # Three high-confidence entries
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Test confidence threshold adjustment
            moderate_invalidated = cache.invalidate_by_confidence(0.9)
            
            if moderate_invalidated == 1:  # One moderate confidence entry (0.88)
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Verify very high confidence entries remain
            very_high_remaining = sum(1 for query, _, confidence, _ in biomedical_confidence_data
                                    if confidence >= 0.9 and cache.get(query) is not None)
            
            if very_high_remaining == 2:  # Two very high confidence entries
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            test_result = TestResult(
                test_name="confidence_based_biomedical_invalidation",
                test_type="biomedical_scenario",
                status="passed" if assertions_failed == 0 else "failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                metadata={
                    'speculative_invalidated': speculative_invalidated,
                    'moderate_invalidated': moderate_invalidated,
                    'high_confidence_remaining': high_confidence_remaining,
                    'very_high_remaining': very_high_remaining
                }
            )
            
        except Exception as e:
            test_result = TestResult(
                test_name="confidence_based_biomedical_invalidation",
                test_type="biomedical_scenario",
                status="failed",
                execution_time_seconds=time.time() - test_start,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed + 1,
                error_message=str(e)
            )
        
        self.test_results.append(test_result)
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for the test run."""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == 'passed'])
        failed_tests = len([r for r in self.test_results if r.status == 'failed'])
        skipped_tests = len([r for r in self.test_results if r.status == 'skipped'])
        
        total_execution_time = sum(r.execution_time_seconds for r in self.test_results)
        total_assertions_passed = sum(r.assertions_passed for r in self.test_results)
        total_assertions_failed = sum(r.assertions_failed for r in self.test_results)
        
        # Statistics by test type
        test_types = set(r.test_type for r in self.test_results)
        type_statistics = {}
        
        for test_type in test_types:
            type_results = [r for r in self.test_results if r.test_type == test_type]
            type_statistics[test_type] = {
                'total': len(type_results),
                'passed': len([r for r in type_results if r.status == 'passed']),
                'failed': len([r for r in type_results if r.status == 'failed']),
                'pass_rate': len([r for r in type_results if r.status == 'passed']) / len(type_results) if type_results else 0
            }
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'skipped_tests': skipped_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_execution_time': total_execution_time,
            'avg_execution_time': total_execution_time / total_tests if total_tests > 0 else 0,
            'total_assertions_passed': total_assertions_passed,
            'total_assertions_failed': total_assertions_failed,
            'assertion_success_rate': total_assertions_passed / (total_assertions_passed + total_assertions_failed) if (total_assertions_passed + total_assertions_failed) > 0 else 0,
            'statistics_by_type': type_statistics
        }
    
    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """Calculate performance summary from test results."""
        performance_results = [r for r in self.test_results 
                             if r.test_type == 'performance' and r.performance_metrics]
        
        if not performance_results:
            return {'no_performance_data': True}
        
        # Aggregate performance metrics
        avg_times = []
        throughput_values = []
        success_rates = []
        memory_usage = []
        
        for result in performance_results:
            metrics = result.performance_metrics
            if isinstance(metrics, dict):
                timing = metrics.get('timing', {})
                throughput = metrics.get('throughput', {})
                reliability = metrics.get('reliability', {})
                resources = metrics.get('resources', {})
                
                if timing.get('avg_time_ms'):
                    avg_times.append(timing['avg_time_ms'])
                
                if throughput.get('operations_per_second'):
                    throughput_values.append(throughput['operations_per_second'])
                
                if reliability.get('success_rate'):
                    success_rates.append(reliability['success_rate'])
                
                if resources.get('peak_memory_mb'):
                    memory_usage.append(resources['peak_memory_mb'])
        
        summary = {}
        
        if avg_times:
            summary['average_latency_ms'] = sum(avg_times) / len(avg_times)
            summary['min_latency_ms'] = min(avg_times)
            summary['max_latency_ms'] = max(avg_times)
        
        if throughput_values:
            summary['average_throughput_ops_sec'] = sum(throughput_values) / len(throughput_values)
            summary['min_throughput_ops_sec'] = min(throughput_values)
            summary['max_throughput_ops_sec'] = max(throughput_values)
        
        if success_rates:
            summary['average_success_rate'] = sum(success_rates) / len(success_rates)
            summary['min_success_rate'] = min(success_rates)
        
        if memory_usage:
            summary['average_memory_usage_mb'] = sum(memory_usage) / len(memory_usage)
            summary['peak_memory_usage_mb'] = max(memory_usage)
        
        summary['performance_tests_count'] = len(performance_results)
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze test results
        summary_stats = self._calculate_summary_statistics()
        performance_summary = self._calculate_performance_summary()
        
        # Pass rate recommendations
        if summary_stats['pass_rate'] < 0.95:
            recommendations.append(
                f"Test pass rate is {summary_stats['pass_rate']:.2%}, below 95% target. "
                "Review failed tests and address underlying issues."
            )
        
        # Performance recommendations
        if not performance_summary.get('no_performance_data'):
            if performance_summary.get('average_latency_ms', 0) > 100:
                recommendations.append(
                    f"Average invalidation latency is {performance_summary['average_latency_ms']:.1f}ms, "
                    "above 100ms target. Consider optimizing invalidation algorithms."
                )
            
            if performance_summary.get('average_throughput_ops_sec', 0) < 50:
                recommendations.append(
                    f"Average throughput is {performance_summary['average_throughput_ops_sec']:.1f} ops/sec, "
                    "below 50 ops/sec target. Consider implementing batch invalidation optimizations."
                )
            
            if performance_summary.get('average_success_rate', 1.0) < 0.95:
                recommendations.append(
                    f"Average success rate is {performance_summary['average_success_rate']:.2%}, "
                    "below 95% target. Investigate reliability issues in invalidation operations."
                )
        
        # Test type specific recommendations
        type_stats = summary_stats.get('statistics_by_type', {})
        
        for test_type, stats in type_stats.items():
            if stats['pass_rate'] < 0.9:
                recommendations.append(
                    f"{test_type.title()} tests have {stats['pass_rate']:.2%} pass rate. "
                    f"Focus on improving {test_type} test reliability."
                )
        
        # Edge case recommendations
        edge_case_stats = type_stats.get('edge_case', {})
        if edge_case_stats and edge_case_stats.get('failed', 0) > 0:
            recommendations.append(
                "Edge case tests are failing. Ensure proper handling of boundary conditions "
                "and error scenarios in invalidation logic."
            )
        
        # Stress test recommendations
        stress_stats = type_stats.get('stress', {})
        if stress_stats and stress_stats.get('failed', 0) > 0:
            recommendations.append(
                "Stress tests are failing. Consider implementing rate limiting, "
                "resource management, and performance optimizations for high-load scenarios."
            )
        
        if not recommendations:
            recommendations.append(
                "All tests are performing well! Consider adding more comprehensive "
                "test scenarios or increasing test coverage for edge cases."
            )
        
        return recommendations
    
    def _save_report(self, report: TestRunReport):
        """Save test report to file."""
        timestamp = report.start_time.strftime("%Y%m%d_%H%M%S")
        
        if self.config.report_format == 'json':
            filename = f"{self.config.output_directory}/cache_invalidation_test_report_{timestamp}.json"
            
            # Convert report to JSON-serializable format
            report_dict = {
                'run_id': report.run_id,
                'start_time': report.start_time.isoformat(),
                'end_time': report.end_time.isoformat(),
                'total_duration_seconds': (report.end_time - report.start_time).total_seconds(),
                'configuration': {
                    'test_types': report.configuration.test_types,
                    'include_stress_tests': report.configuration.include_stress_tests,
                    'include_edge_case_tests': report.configuration.include_edge_case_tests,
                    'performance_iterations': report.configuration.performance_iterations
                },
                'test_results': [
                    {
                        'test_name': r.test_name,
                        'test_type': r.test_type,
                        'status': r.status,
                        'execution_time_seconds': r.execution_time_seconds,
                        'assertions_passed': r.assertions_passed,
                        'assertions_failed': r.assertions_failed,
                        'error_message': r.error_message,
                        'performance_metrics': r.performance_metrics,
                        'metadata': r.metadata
                    } for r in report.test_results
                ],
                'summary_statistics': report.summary_statistics,
                'performance_summary': report.performance_summary,
                'recommendations': report.recommendations
            }
            
            with open(filename, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            self.logger.info(f"Test report saved to {filename}")
        
        elif self.config.report_format == 'markdown':
            filename = f"{self.config.output_directory}/cache_invalidation_test_report_{timestamp}.md"
            
            with open(filename, 'w') as f:
                f.write(f"# Cache Invalidation Test Report\n\n")
                f.write(f"**Run ID:** {report.run_id}\n")
                f.write(f"**Start Time:** {report.start_time}\n")
                f.write(f"**End Time:** {report.end_time}\n")
                f.write(f"**Duration:** {(report.end_time - report.start_time).total_seconds():.1f} seconds\n\n")
                
                f.write(f"## Summary Statistics\n\n")
                stats = report.summary_statistics
                f.write(f"- **Total Tests:** {stats['total_tests']}\n")
                f.write(f"- **Passed:** {stats['passed_tests']} ({stats['pass_rate']:.2%})\n")
                f.write(f"- **Failed:** {stats['failed_tests']}\n")
                f.write(f"- **Skipped:** {stats['skipped_tests']}\n\n")
                
                if not report.performance_summary.get('no_performance_data'):
                    f.write(f"## Performance Summary\n\n")
                    perf = report.performance_summary
                    if 'average_latency_ms' in perf:
                        f.write(f"- **Average Latency:** {perf['average_latency_ms']:.1f}ms\n")
                    if 'average_throughput_ops_sec' in perf:
                        f.write(f"- **Average Throughput:** {perf['average_throughput_ops_sec']:.1f} ops/sec\n")
                    if 'average_success_rate' in perf:
                        f.write(f"- **Average Success Rate:** {perf['average_success_rate']:.2%}\n")
                    f.write("\n")
                
                f.write(f"## Recommendations\n\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            
            self.logger.info(f"Markdown report saved to {filename}")
    
    def _log_summary(self, report: TestRunReport):
        """Log test run summary."""
        stats = report.summary_statistics
        duration = (report.end_time - report.start_time).total_seconds()
        
        self.logger.info(f"Test run completed in {duration:.1f} seconds")
        self.logger.info(f"Results: {stats['passed_tests']}/{stats['total_tests']} passed ({stats['pass_rate']:.2%})")
        
        if stats['failed_tests'] > 0:
            self.logger.warning(f"{stats['failed_tests']} tests failed")
            
            # Log failed test names
            failed_tests = [r.test_name for r in report.test_results if r.status == 'failed']
            self.logger.warning(f"Failed tests: {', '.join(failed_tests)}")
        
        if not report.performance_summary.get('no_performance_data'):
            perf = report.performance_summary
            self.logger.info(f"Performance: {perf.get('average_latency_ms', 0):.1f}ms avg latency, "
                           f"{perf.get('average_throughput_ops_sec', 0):.1f} ops/sec throughput")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive cache invalidation test runner"
    )
    
    parser.add_argument(
        '--test-types',
        nargs='+',
        choices=['unit', 'integration', 'performance'],
        default=['unit', 'integration', 'performance'],
        help='Types of tests to run'
    )
    
    parser.add_argument(
        '--include-stress',
        action='store_true',
        help='Include stress tests'
    )
    
    parser.add_argument(
        '--include-edge-cases',
        action='store_true',
        default=True,
        help='Include edge case tests'
    )
    
    parser.add_argument(
        '--report-format',
        choices=['json', 'markdown'],
        default='json',
        help='Report output format'
    )
    
    parser.add_argument(
        '--output-dir',
        default='test_results',
        help='Output directory for test reports'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--performance-iterations',
        type=int,
        default=100,
        help='Number of iterations for performance tests'
    )
    
    args = parser.parse_args()
    
    # Create test configuration
    config = TestRunConfiguration(
        test_types=args.test_types,
        include_stress_tests=args.include_stress,
        include_edge_case_tests=args.include_edge_cases,
        report_format=args.report_format,
        output_directory=args.output_dir,
        log_level=args.log_level,
        performance_iterations=args.performance_iterations
    )
    
    # Run tests
    runner = InvalidationTestRunner(config)
    report = runner.run_all_tests()
    
    # Exit with appropriate code
    if report.summary_statistics['failed_tests'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()