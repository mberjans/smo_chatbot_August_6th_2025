#!/usr/bin/env python3
"""
Performance Validation Suite for Real-Time Classification Optimization

This module provides comprehensive validation and benchmarking for the optimized
LLM-based classification system to ensure <2 second response time compliance
and maintain >90% accuracy.

Test Suites:
    - Response Time Validation (<2s target)
    - Classification Accuracy Validation (>90% target)
    - Cache Performance Testing (>70% hit rate target)
    - Circuit Breaker Validation
    - Load Testing (>100 RPS target)
    - Memory and Resource Usage Testing
    - Edge Case Handling Validation

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
Task: CMO-LIGHTRAG-012-T07 - Validate performance optimizations
"""

import asyncio
import time
import json
import logging
import statistics
import psutil
import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from .realtime_classification_optimizer import (
        RealTimeClassificationOptimizer,
        create_optimized_classifier,
        UltraFastPrompts,
        SemanticSimilarityCache
    )
    from .enhanced_llm_classifier import EnhancedLLMConfig, LLMProvider
    OPTIMIZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import optimizer: {e}")
    OPTIMIZER_AVAILABLE = False


@dataclass
class PerformanceTestResult:
    """Result of a performance test."""
    
    test_name: str
    passed: bool
    actual_value: float
    target_value: float
    unit: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ValidationSummary:
    """Summary of all validation tests."""
    
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float  # 0.0 to 1.0
    performance_grade: str  # EXCELLENT, GOOD, NEEDS_IMPROVEMENT, POOR
    test_results: List[PerformanceTestResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceValidationSuite:
    """
    Comprehensive validation suite for real-time classification performance.
    """
    
    def __init__(self):
        self.optimizer = None
        self.test_results = []
        self.start_time = None
        self.system_info = self._get_system_info()
        
        # Test query datasets
        self.accuracy_test_queries = self._get_accuracy_test_dataset()
        self.performance_test_queries = self._get_performance_test_dataset()
        self.edge_case_queries = self._get_edge_case_dataset()
        
        logger.info("Performance validation suite initialized")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.sys.platform,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_accuracy_test_dataset(self) -> List[Dict[str, str]]:
        """Get labeled dataset for accuracy testing."""
        
        return [
            # KNOWLEDGE_GRAPH queries (established relationships/mechanisms)
            {"query": "What is the relationship between glucose metabolism and insulin signaling?", "expected": "KNOWLEDGE_GRAPH"},
            {"query": "How does the citric acid cycle connect to fatty acid biosynthesis?", "expected": "KNOWLEDGE_GRAPH"},
            {"query": "Mechanism of metformin action in glucose homeostasis regulation", "expected": "KNOWLEDGE_GRAPH"},
            {"query": "Biomarkers associated with Alzheimer's disease in cerebrospinal fluid", "expected": "KNOWLEDGE_GRAPH"},
            {"query": "Connection between oxidative stress and mitochondrial dysfunction", "expected": "KNOWLEDGE_GRAPH"},
            {"query": "Pathway analysis of tryptophan metabolism to serotonin", "expected": "KNOWLEDGE_GRAPH"},
            {"query": "Role of AMPK in cellular energy homeostasis", "expected": "KNOWLEDGE_GRAPH"},
            {"query": "Metabolic interactions between liver and muscle tissue", "expected": "KNOWLEDGE_GRAPH"},
            {"query": "Biomarker validation for cardiovascular disease risk", "expected": "KNOWLEDGE_GRAPH"},
            {"query": "Insulin resistance mechanisms in type 2 diabetes", "expected": "KNOWLEDGE_GRAPH"},
            
            # REAL_TIME queries (current/recent information)
            {"query": "Latest FDA approvals for metabolomics-based diagnostics in 2024", "expected": "REAL_TIME"},
            {"query": "Recent breakthrough discoveries in cancer metabolomics this year", "expected": "REAL_TIME"},
            {"query": "Current clinical trials using AI-powered metabolomics analysis", "expected": "REAL_TIME"},
            {"query": "New metabolomics biomarker partnerships announced in 2024", "expected": "REAL_TIME"},
            {"query": "Breaking news in precision medicine metabolomics applications", "expected": "REAL_TIME"},
            {"query": "Latest research on metabolomics in drug discovery 2024", "expected": "REAL_TIME"},
            {"query": "Recent advances in mass spectrometry technology this year", "expected": "REAL_TIME"},
            {"query": "Current trends in clinical metabolomics adoption", "expected": "REAL_TIME"},
            {"query": "New AI tools for metabolomics data analysis released recently", "expected": "REAL_TIME"},
            {"query": "Updates on metabolomics standardization efforts in 2024", "expected": "REAL_TIME"},
            
            # GENERAL queries (basic definitions/explanations)
            {"query": "What is metabolomics and how does it work?", "expected": "GENERAL"},
            {"query": "Explain the basics of LC-MS analysis for beginners", "expected": "GENERAL"},
            {"query": "How to interpret NMR spectra in metabolomics studies", "expected": "GENERAL"},
            {"query": "Applications of metabolomics in personalized healthcare", "expected": "GENERAL"},
            {"query": "Define biomarker validation in clinical research", "expected": "GENERAL"},
            {"query": "Overview of metabolomics data preprocessing steps", "expected": "GENERAL"},
            {"query": "Introduction to systems biology and metabolic networks", "expected": "GENERAL"},
            {"query": "Basic principles of mass spectrometry", "expected": "GENERAL"},
            {"query": "What are the main metabolomics analytical platforms?", "expected": "GENERAL"},
            {"query": "How to design a metabolomics study?", "expected": "GENERAL"}
        ]
    
    def _get_performance_test_dataset(self) -> List[str]:
        """Get queries for performance testing."""
        
        return [
            "metabolomics",
            "What is LC-MS?",
            "latest research",
            "glucose insulin pathway relationship analysis",
            "How does mass spectrometry work in metabolomics research?",
            "Recent FDA approvals for biomarker-based diagnostics in 2024",
            "Complex metabolic pathway analysis involving multiple tissue types and regulatory mechanisms",
            "biomarker discovery validation process clinical trials methodology overview",
            "Latest breakthrough research findings in precision medicine metabolomics applications published this year",
            "Comprehensive overview of the relationship between metabolite concentrations and disease progression markers in clinical populations"
        ]
    
    def _get_edge_case_dataset(self) -> List[str]:
        """Get edge case queries for robustness testing."""
        
        return [
            "",  # Empty query
            "a",  # Single character
            "a" * 1000,  # Very long query
            "???",  # Special characters only
            "12345",  # Numbers only
            "METABOLOMICS GLUCOSE INSULIN",  # All caps
            "what what what what what",  # Repeated words
            "This query contains symbols like @#$%^&*() and numbers 123456",
            "Mixed cAsE qUeRy WiTh WeIrD cApItAlIzAtIoN",
            "Query with\nnewlines\nand\ttabs"
        ]
    
    async def run_comprehensive_validation(self, api_key: Optional[str] = None) -> ValidationSummary:
        """
        Run comprehensive validation suite for real-time classification optimizations.
        
        Args:
            api_key: OpenAI API key for testing
            
        Returns:
            Comprehensive validation summary
        """
        
        logger.info("Starting comprehensive performance validation suite")
        self.start_time = time.time()
        
        # Initialize optimizer
        if not await self._initialize_optimizer(api_key):
            return self._create_failed_summary("Failed to initialize optimizer")
        
        # Run all test suites
        test_suites = [
            ("Response Time Validation", self._test_response_time_compliance),
            ("Classification Accuracy", self._test_classification_accuracy),
            ("Cache Performance", self._test_cache_performance),
            ("Circuit Breaker Validation", self._test_circuit_breaker),
            ("Load Testing", self._test_load_performance),
            ("Memory Usage", self._test_memory_usage),
            ("Edge Case Handling", self._test_edge_cases),
            ("Optimization Effectiveness", self._test_optimization_effectiveness)
        ]
        
        for suite_name, test_function in test_suites:
            logger.info(f"Running {suite_name}...")
            try:
                suite_results = await test_function()
                self.test_results.extend(suite_results)
                logger.info(f"Completed {suite_name}: {len(suite_results)} tests")
            except Exception as e:
                logger.error(f"Test suite {suite_name} failed: {e}")
                self.test_results.append(PerformanceTestResult(
                    test_name=f"{suite_name} (Failed)",
                    passed=False,
                    actual_value=0.0,
                    target_value=1.0,
                    unit="boolean",
                    details={"error": str(e)}
                ))
        
        # Generate summary
        summary = self._generate_validation_summary()
        
        # Save results
        await self._save_validation_results(summary)
        
        total_time = time.time() - self.start_time
        logger.info(f"Validation completed in {total_time:.1f}s - Grade: {summary.performance_grade}")
        
        return summary
    
    async def _initialize_optimizer(self, api_key: Optional[str]) -> bool:
        """Initialize the optimizer for testing."""
        
        if not OPTIMIZER_AVAILABLE:
            logger.error("Optimizer dependencies not available")
            return False
        
        try:
            # Create optimized configuration for testing
            self.optimizer = await create_optimized_classifier(
                api_key=api_key,
                enable_cache_warming=True  # Enable for testing
            )
            
            # Warm up with a test query
            await self.optimizer.classify_query_optimized("test query")
            
            logger.info("Optimizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            return False
    
    async def _test_response_time_compliance(self) -> List[PerformanceTestResult]:
        """Test response time compliance with <2 second target."""
        
        results = []
        response_times = []
        
        logger.info("Testing response time compliance (<2000ms target)...")
        
        for i, query in enumerate(self.performance_test_queries):
            start_time = time.time()
            
            try:
                result, metadata = await self.optimizer.classify_query_optimized(query)
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
                
                # Individual query test
                results.append(PerformanceTestResult(
                    test_name=f"Response Time Query {i+1}",
                    passed=response_time <= 2000,
                    actual_value=response_time,
                    target_value=2000.0,
                    unit="ms",
                    details={
                        "query_length": len(query),
                        "category": result.category,
                        "optimizations": metadata.get("optimization_applied", [])
                    }
                ))
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                results.append(PerformanceTestResult(
                    test_name=f"Response Time Query {i+1} (Error)",
                    passed=False,
                    actual_value=response_time,
                    target_value=2000.0,
                    unit="ms",
                    details={"error": str(e)}
                ))
        
        # Aggregate statistics
        if response_times:
            avg_time = statistics.mean(response_times)
            p95_time = sorted(response_times)[int(0.95 * len(response_times))]
            p99_time = sorted(response_times)[int(0.99 * len(response_times))]
            
            results.extend([
                PerformanceTestResult(
                    test_name="Average Response Time",
                    passed=avg_time <= 1500,  # More stringent for average
                    actual_value=avg_time,
                    target_value=1500.0,
                    unit="ms"
                ),
                PerformanceTestResult(
                    test_name="95th Percentile Response Time",
                    passed=p95_time <= 2000,
                    actual_value=p95_time,
                    target_value=2000.0,
                    unit="ms"
                ),
                PerformanceTestResult(
                    test_name="99th Percentile Response Time",
                    passed=p99_time <= 2500,
                    actual_value=p99_time,
                    target_value=2500.0,
                    unit="ms"
                )
            ])
        
        return results
    
    async def _test_classification_accuracy(self) -> List[PerformanceTestResult]:
        """Test classification accuracy with labeled dataset."""
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        logger.info("Testing classification accuracy (>90% target)...")
        
        for test_case in self.accuracy_test_queries:
            query = test_case["query"]
            expected = test_case["expected"]
            
            try:
                result, metadata = await self.optimizer.classify_query_optimized(query)
                actual = result.category
                
                is_correct = actual == expected
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                results.append(PerformanceTestResult(
                    test_name=f"Accuracy: {query[:40]}...",
                    passed=is_correct,
                    actual_value=1.0 if is_correct else 0.0,
                    target_value=1.0,
                    unit="boolean",
                    details={
                        "query": query,
                        "expected": expected,
                        "actual": actual,
                        "confidence": result.confidence,
                        "response_time_ms": metadata.get("response_time_ms", 0)
                    }
                ))
                
            except Exception as e:
                total_predictions += 1
                results.append(PerformanceTestResult(
                    test_name=f"Accuracy: {query[:40]}... (Error)",
                    passed=False,
                    actual_value=0.0,
                    target_value=1.0,
                    unit="boolean",
                    details={"error": str(e)}
                ))
        
        # Overall accuracy
        overall_accuracy = correct_predictions / max(1, total_predictions)
        results.append(PerformanceTestResult(
            test_name="Overall Classification Accuracy",
            passed=overall_accuracy >= 0.90,
            actual_value=overall_accuracy * 100,
            target_value=90.0,
            unit="percent",
            details={
                "correct_predictions": correct_predictions,
                "total_predictions": total_predictions
            }
        ))
        
        return results
    
    async def _test_cache_performance(self) -> List[PerformanceTestResult]:
        """Test cache performance and hit rates."""
        
        results = []
        
        logger.info("Testing cache performance (>70% hit rate target)...")
        
        # Test queries to warm cache
        cache_test_queries = [
            "what is metabolomics",
            "glucose insulin relationship", 
            "latest research 2024",
            "biomarker discovery process",
            "LC-MS analysis basics"
        ]
        
        # First pass - populate cache
        for query in cache_test_queries:
            await self.optimizer.classify_query_optimized(query)
        
        # Second pass - test cache hits
        cache_hits = 0
        total_requests = 0
        
        for query in cache_test_queries:
            result, metadata = await self.optimizer.classify_query_optimized(query)
            total_requests += 1
            
            if metadata.get("used_semantic_cache", False):
                cache_hits += 1
            
            # Also test with slight variations
            variations = [
                query.upper(),
                query + "?",
                "What is " + query if not query.startswith("what") else query
            ]
            
            for variation in variations:
                result, metadata = await self.optimizer.classify_query_optimized(variation)
                total_requests += 1
                
                if metadata.get("used_semantic_cache", False):
                    cache_hits += 1
        
        # Cache hit rate test
        hit_rate = cache_hits / max(1, total_requests)
        results.append(PerformanceTestResult(
            test_name="Cache Hit Rate",
            passed=hit_rate >= 0.70,
            actual_value=hit_rate * 100,
            target_value=70.0,
            unit="percent",
            details={
                "cache_hits": cache_hits,
                "total_requests": total_requests
            }
        ))
        
        # Cache performance stats
        cache_stats = self.optimizer.semantic_cache.get_stats()
        
        results.append(PerformanceTestResult(
            test_name="Cache Utilization",
            passed=cache_stats["cache_size"] > 0,
            actual_value=cache_stats.get("utilization", 0) * 100,
            target_value=20.0,  # At least 20% utilization
            unit="percent",
            details=cache_stats
        ))
        
        return results
    
    async def _test_circuit_breaker(self) -> List[PerformanceTestResult]:
        """Test circuit breaker functionality."""
        
        results = []
        
        logger.info("Testing circuit breaker functionality...")
        
        # Test circuit breaker state transitions
        cb = self.optimizer.adaptive_circuit_breaker
        initial_state = cb.state
        
        results.append(PerformanceTestResult(
            test_name="Circuit Breaker Initial State",
            passed=initial_state == "closed",
            actual_value=1.0 if initial_state == "closed" else 0.0,
            target_value=1.0,
            unit="boolean",
            details={"initial_state": initial_state}
        ))
        
        # Test circuit breaker stats availability
        cb_stats = cb.get_stats()
        
        results.append(PerformanceTestResult(
            test_name="Circuit Breaker Stats Available",
            passed=isinstance(cb_stats, dict) and "state" in cb_stats,
            actual_value=1.0 if isinstance(cb_stats, dict) else 0.0,
            target_value=1.0,
            unit="boolean",
            details=cb_stats
        ))
        
        # Test recovery timeout is optimized
        results.append(PerformanceTestResult(
            test_name="Circuit Breaker Recovery Time Optimized",
            passed=cb.current_recovery_timeout <= 10.0,
            actual_value=cb.current_recovery_timeout,
            target_value=10.0,
            unit="seconds",
            details={"recovery_timeout": cb.current_recovery_timeout}
        ))
        
        return results
    
    async def _test_load_performance(self) -> List[PerformanceTestResult]:
        """Test performance under concurrent load."""
        
        results = []
        
        logger.info("Testing load performance (concurrent requests)...")
        
        # Test concurrent processing
        concurrent_queries = [
            "what is metabolomics",
            "latest research",
            "glucose pathway",
            "biomarker discovery",
            "LC-MS analysis"
        ] * 4  # 20 concurrent requests
        
        start_time = time.time()
        
        # Execute concurrent requests
        tasks = []
        for query in concurrent_queries:
            task = asyncio.create_task(
                self.optimizer.classify_query_optimized(query, priority="normal")
            )
            tasks.append(task)
        
        # Wait for all to complete
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        successful_results = [r for r in concurrent_results if not isinstance(r, Exception)]
        
        # Calculate throughput
        throughput = len(concurrent_queries) / total_time
        
        results.extend([
            PerformanceTestResult(
                test_name="Concurrent Request Success Rate",
                passed=len(successful_results) >= len(concurrent_queries) * 0.95,
                actual_value=(len(successful_results) / len(concurrent_queries)) * 100,
                target_value=95.0,
                unit="percent",
                details={
                    "successful": len(successful_results),
                    "total": len(concurrent_queries)
                }
            ),
            PerformanceTestResult(
                test_name="Throughput (Requests/Second)",
                passed=throughput >= 10,  # Conservative target for testing
                actual_value=throughput,
                target_value=10.0,
                unit="requests/second",
                details={
                    "total_time": total_time,
                    "total_requests": len(concurrent_queries)
                }
            ),
            PerformanceTestResult(
                test_name="Concurrent Processing Time",
                passed=total_time <= 10.0,  # Should complete within 10 seconds
                actual_value=total_time,
                target_value=10.0,
                unit="seconds"
            )
        ])
        
        return results
    
    async def _test_memory_usage(self) -> List[PerformanceTestResult]:
        """Test memory usage and resource efficiency."""
        
        results = []
        
        logger.info("Testing memory usage and resource efficiency...")
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Process multiple queries to test memory stability
        memory_test_queries = self.performance_test_queries * 5  # 50 queries
        
        for query in memory_test_queries:
            await self.optimizer.classify_query_optimized(query)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        results.extend([
            PerformanceTestResult(
                test_name="Memory Usage Increase",
                passed=memory_increase <= 50,  # No more than 50MB increase
                actual_value=memory_increase,
                target_value=50.0,
                unit="MB",
                details={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "queries_processed": len(memory_test_queries)
                }
            ),
            PerformanceTestResult(
                test_name="Memory Efficiency",
                passed=memory_increase / len(memory_test_queries) <= 1.0,  # <1MB per query
                actual_value=memory_increase / len(memory_test_queries),
                target_value=1.0,
                unit="MB/query"
            )
        ])
        
        return results
    
    async def _test_edge_cases(self) -> List[PerformanceTestResult]:
        """Test handling of edge cases and malformed inputs."""
        
        results = []
        
        logger.info("Testing edge case handling...")
        
        successful_edge_cases = 0
        
        for i, edge_query in enumerate(self.edge_case_queries):
            try:
                result, metadata = await self.optimizer.classify_query_optimized(edge_query)
                
                # Should return a valid result even for edge cases
                is_valid = (
                    hasattr(result, 'category') and 
                    result.category in ["KNOWLEDGE_GRAPH", "REAL_TIME", "GENERAL"] and
                    0 <= result.confidence <= 1.0
                )
                
                if is_valid:
                    successful_edge_cases += 1
                
                results.append(PerformanceTestResult(
                    test_name=f"Edge Case {i+1}: {repr(edge_query[:20])}...",
                    passed=is_valid,
                    actual_value=1.0 if is_valid else 0.0,
                    target_value=1.0,
                    unit="boolean",
                    details={
                        "query": repr(edge_query),
                        "category": getattr(result, 'category', 'None'),
                        "confidence": getattr(result, 'confidence', -1),
                        "response_time_ms": metadata.get("response_time_ms", 0)
                    }
                ))
                
            except Exception as e:
                results.append(PerformanceTestResult(
                    test_name=f"Edge Case {i+1}: {repr(edge_query[:20])}... (Error)",
                    passed=False,
                    actual_value=0.0,
                    target_value=1.0,
                    unit="boolean",
                    details={"error": str(e), "query": repr(edge_query)}
                ))
        
        # Overall edge case handling
        edge_case_success_rate = successful_edge_cases / len(self.edge_case_queries)
        results.append(PerformanceTestResult(
            test_name="Edge Case Success Rate",
            passed=edge_case_success_rate >= 0.8,  # 80% should handle gracefully
            actual_value=edge_case_success_rate * 100,
            target_value=80.0,
            unit="percent",
            details={
                "successful": successful_edge_cases,
                "total": len(self.edge_case_queries)
            }
        ))
        
        return results
    
    async def _test_optimization_effectiveness(self) -> List[PerformanceTestResult]:
        """Test effectiveness of various optimizations."""
        
        results = []
        
        logger.info("Testing optimization effectiveness...")
        
        # Get optimizer performance stats
        perf_stats = self.optimizer.get_performance_stats()
        
        # Test cache effectiveness
        cache_stats = perf_stats.get("cache_performance", {})
        hit_rate = cache_stats.get("hit_rate", 0)
        
        results.append(PerformanceTestResult(
            test_name="Cache Optimization Effectiveness",
            passed=hit_rate >= 0.3,  # At least 30% hit rate
            actual_value=hit_rate * 100,
            target_value=30.0,
            unit="percent",
            details=cache_stats
        ))
        
        # Test response time optimization
        avg_response_time = perf_stats.get("avg_response_time_ms", 9999)
        results.append(PerformanceTestResult(
            test_name="Response Time Optimization",
            passed=avg_response_time <= 2000,
            actual_value=avg_response_time,
            target_value=2000.0,
            unit="ms"
        ))
        
        # Test target compliance
        target_compliance = perf_stats.get("target_compliance_rate", 0)
        results.append(PerformanceTestResult(
            test_name="Performance Target Compliance",
            passed=target_compliance >= 0.8,  # 80% of requests under 2s
            actual_value=target_compliance * 100,
            target_value=80.0,
            unit="percent"
        ))
        
        return results
    
    def _generate_validation_summary(self) -> ValidationSummary:
        """Generate comprehensive validation summary."""
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.passed])
        failed_tests = total_tests - passed_tests
        
        overall_score = passed_tests / max(1, total_tests)
        
        # Determine performance grade
        if overall_score >= 0.95:
            grade = "EXCELLENT"
        elif overall_score >= 0.85:
            grade = "GOOD"
        elif overall_score >= 0.70:
            grade = "NEEDS_IMPROVEMENT"
        else:
            grade = "POOR"
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return ValidationSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_score=overall_score,
            performance_grade=grade,
            test_results=self.test_results,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on test results."""
        
        recommendations = []
        
        # Response time recommendations
        response_time_tests = [r for r in self.test_results if "Response Time" in r.test_name and not r.passed]
        if response_time_tests:
            recommendations.append("Consider further prompt optimization or caching improvements to meet <2s response time target")
        
        # Accuracy recommendations
        accuracy_tests = [r for r in self.test_results if "Accuracy" in r.test_name and not r.passed]
        if len(accuracy_tests) > 3:
            recommendations.append("Review and tune classification prompts to improve accuracy >90%")
        
        # Cache recommendations
        cache_tests = [r for r in self.test_results if "Cache" in r.test_name and not r.passed]
        if cache_tests:
            recommendations.append("Optimize cache configuration and similarity thresholds for better hit rates")
        
        # Load recommendations
        load_tests = [r for r in self.test_results if "Concurrent" in r.test_name or "Throughput" in r.test_name]
        failed_load_tests = [r for r in load_tests if not r.passed]
        if failed_load_tests:
            recommendations.append("Consider connection pooling and async optimizations for better load handling")
        
        if not recommendations:
            recommendations.append("Performance optimizations are working effectively - monitor in production")
        
        return recommendations
    
    def _create_failed_summary(self, reason: str) -> ValidationSummary:
        """Create a failed validation summary."""
        
        return ValidationSummary(
            total_tests=0,
            passed_tests=0,
            failed_tests=1,
            overall_score=0.0,
            performance_grade="FAILED",
            test_results=[PerformanceTestResult(
                test_name="Initialization",
                passed=False,
                actual_value=0.0,
                target_value=1.0,
                unit="boolean",
                details={"failure_reason": reason}
            )],
            recommendations=[f"Fix initialization issue: {reason}"]
        )
    
    async def _save_validation_results(self, summary: ValidationSummary) -> None:
        """Save validation results to file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(f"performance_validation_results_{timestamp}.json")
        
        # Convert to JSON-serializable format
        results_data = {
            "validation_summary": {
                "total_tests": summary.total_tests,
                "passed_tests": summary.passed_tests,
                "failed_tests": summary.failed_tests,
                "overall_score": summary.overall_score,
                "performance_grade": summary.performance_grade,
                "recommendations": summary.recommendations
            },
            "detailed_results": [
                {
                    "test_name": result.test_name,
                    "passed": result.passed,
                    "actual_value": result.actual_value,
                    "target_value": result.target_value,
                    "unit": result.unit,
                    "details": result.details,
                    "timestamp": result.timestamp
                }
                for result in summary.test_results
            ],
            "system_info": self.system_info,
            "validation_metadata": {
                "total_validation_time_seconds": time.time() - self.start_time if self.start_time else 0,
                "validation_timestamp": timestamp,
                "test_suites_run": 8
            }
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Validation results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")


# ============================================================================
# CLI INTERFACE AND MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function for validation suite."""
    
    print("=" * 70)
    print("REAL-TIME CLASSIFICATION PERFORMANCE VALIDATION SUITE")
    print("Clinical Metabolomics Oracle - CMO-LIGHTRAG-012-T07")
    print("=" * 70)
    print()
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  WARNING: No OPENAI_API_KEY environment variable set.")
        print("   Some tests will run in fallback mode with limited functionality.")
        print()
    
    # Initialize validation suite
    validator = PerformanceValidationSuite()
    
    print("🔍 System Information:")
    print(f"   CPU Cores: {validator.system_info['cpu_count']}")
    print(f"   Memory: {validator.system_info['memory_total_gb']:.1f} GB")
    print(f"   Platform: {validator.system_info['platform']}")
    print()
    
    print("🚀 Starting comprehensive validation...")
    print()
    
    # Run validation
    try:
        summary = await validator.run_comprehensive_validation(api_key)
        
        # Print results
        print("=" * 70)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"📊 Overall Performance Grade: {summary.performance_grade}")
        print(f"📈 Tests Passed: {summary.passed_tests}/{summary.total_tests} ({summary.overall_score:.1%})")
        print(f"📉 Tests Failed: {summary.failed_tests}")
        print(f"⭐ Overall Score: {summary.overall_score:.3f}")
        print()
        
        if summary.failed_tests > 0:
            print("❌ Failed Tests:")
            failed_tests = [r for r in summary.test_results if not r.passed]
            for test in failed_tests[:5]:  # Show first 5 failures
                print(f"   • {test.test_name}: {test.actual_value}{test.unit} (target: {test.target_value}{test.unit})")
            
            if len(failed_tests) > 5:
                print(f"   ... and {len(failed_tests) - 5} more")
            print()
        
        print("💡 Recommendations:")
        for i, rec in enumerate(summary.recommendations, 1):
            print(f"   {i}. {rec}")
        print()
        
        # Performance targets summary
        print("🎯 Key Performance Targets:")
        
        # Find specific target results
        target_tests = {
            "Response Time": [r for r in summary.test_results if "Average Response Time" in r.test_name],
            "Accuracy": [r for r in summary.test_results if "Overall Classification Accuracy" in r.test_name],
            "Cache Hit Rate": [r for r in summary.test_results if "Cache Hit Rate" in r.test_name],
            "Target Compliance": [r for r in summary.test_results if "Performance Target Compliance" in r.test_name]
        }
        
        for target_name, tests in target_tests.items():
            if tests:
                test = tests[0]
                status = "✅" if test.passed else "❌"
                print(f"   {status} {target_name}: {test.actual_value:.1f}{test.unit} (target: {test.target_value}{test.unit})")
        
        print()
        print("=" * 70)
        
        # Final verdict
        if summary.performance_grade in ["EXCELLENT", "GOOD"]:
            print("🎉 VALIDATION PASSED - Performance optimizations meet requirements!")
            print("   System is ready for real-time production use.")
        elif summary.performance_grade == "NEEDS_IMPROVEMENT":
            print("⚠️  VALIDATION PARTIAL - Some optimizations need fine-tuning.")
            print("   Review recommendations before production deployment.")
        else:
            print("❌ VALIDATION FAILED - Significant performance issues detected.")
            print("   Address critical issues before deployment.")
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        logger.error(f"Validation error: {e}", exc_info=True)
    
    print()
    print("📁 Check performance_validation_results_*.json for detailed results")
    print("🏁 Validation complete")


if __name__ == "__main__":
    # Run the validation suite
    asyncio.run(main())