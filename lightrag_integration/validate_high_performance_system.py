#!/usr/bin/env python3
"""
High-Performance Classification System Validation Script

This script provides comprehensive validation of the high-performance LLM-based
classification system to ensure consistent <2 second response times under
various conditions and load patterns.

Validation Components:
    - System initialization and health checks
    - Basic functionality verification
    - Performance benchmark suite execution
    - Load testing with various patterns
    - Resource utilization monitoring
    - Cache effectiveness validation
    - Stress testing and failure analysis
    - Optimization recommendations

Author: Claude Code (Anthropic)
Version: 1.0.0 - System Validation Script
Created: 2025-08-08
Target: Comprehensive validation for production readiness
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Import performance components
try:
    from .high_performance_classification_system import (
        HighPerformanceClassificationSystem,
        HighPerformanceConfig,
        high_performance_classification_context,
        create_high_performance_system
    )
    from .performance_benchmark_suite import (
        BenchmarkConfig,
        BenchmarkType,
        LoadPattern,
        PerformanceGrade,
        PerformanceBenchmarkRunner,
        BenchmarkReporter,
        run_comprehensive_benchmark,
        run_quick_performance_test,
        run_stress_test
    )
    from .test_high_performance_integration import TestComprehensiveIntegration
    from .llm_classification_prompts import ClassificationResult
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Required components not available: {e}")
    COMPONENTS_AVAILABLE = False


# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

class ValidationLevel:
    """Validation levels with increasing thoroughness."""
    BASIC = "basic"           # Quick smoke test
    STANDARD = "standard"     # Normal validation
    COMPREHENSIVE = "comprehensive"  # Full validation suite
    PRODUCTION = "production" # Production readiness validation


class ValidationConfig:
    """Configuration for system validation."""
    
    def __init__(self, 
                 validation_level: str = ValidationLevel.STANDARD,
                 target_response_time_ms: float = 1500.0,
                 max_response_time_ms: float = 2000.0,
                 enable_stress_testing: bool = True,
                 enable_load_testing: bool = True,
                 enable_cache_testing: bool = True,
                 export_results: bool = True,
                 output_directory: str = "validation_results"):
        
        self.validation_level = validation_level
        self.target_response_time_ms = target_response_time_ms
        self.max_response_time_ms = max_response_time_ms
        self.enable_stress_testing = enable_stress_testing
        self.enable_load_testing = enable_load_testing
        self.enable_cache_testing = enable_cache_testing
        self.export_results = export_results
        self.output_directory = output_directory
        
        # Configure test parameters based on validation level
        self.test_params = self._configure_test_parameters()
    
    def _configure_test_parameters(self) -> Dict[str, Any]:
        """Configure test parameters based on validation level."""
        
        if self.validation_level == ValidationLevel.BASIC:
            return {
                "quick_test_requests": 25,
                "load_test_requests": 100,
                "load_test_users": 10,
                "stress_test_requests": 200,
                "stress_test_users": 20,
                "cache_test_queries": 20,
                "enable_plots": False
            }
        elif self.validation_level == ValidationLevel.STANDARD:
            return {
                "quick_test_requests": 50,
                "load_test_requests": 200,
                "load_test_users": 15,
                "stress_test_requests": 500,
                "stress_test_users": 30,
                "cache_test_queries": 50,
                "enable_plots": True
            }
        elif self.validation_level == ValidationLevel.COMPREHENSIVE:
            return {
                "quick_test_requests": 100,
                "load_test_requests": 500,
                "load_test_users": 25,
                "stress_test_requests": 1000,
                "stress_test_users": 50,
                "cache_test_queries": 100,
                "enable_plots": True
            }
        else:  # PRODUCTION
            return {
                "quick_test_requests": 200,
                "load_test_requests": 1000,
                "load_test_users": 50,
                "stress_test_requests": 2000,
                "stress_test_users": 100,
                "cache_test_queries": 200,
                "enable_plots": True
            }


# ============================================================================
# VALIDATION RESULT CLASSES
# ============================================================================

class ValidationResult:
    """Individual validation test result."""
    
    def __init__(self, 
                 test_name: str,
                 passed: bool,
                 duration_seconds: float,
                 details: Dict[str, Any],
                 error_message: Optional[str] = None):
        
        self.test_name = test_name
        self.passed = passed
        self.duration_seconds = duration_seconds
        self.details = details
        self.error_message = error_message
        self.timestamp = datetime.now()


class ValidationSummary:
    """Complete validation summary."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        self.validation_config = None
        self.results: List[ValidationResult] = []
        self.system_info: Dict[str, Any] = {}
        self.overall_grade = None
        self.passed_count = 0
        self.failed_count = 0
        self.recommendations: List[str] = []
        self.export_paths: Dict[str, str] = {}
    
    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)
        if result.passed:
            self.passed_count += 1
        else:
            self.failed_count += 1
    
    def finalize(self):
        """Finalize the validation summary."""
        self.end_time = datetime.now()
        self.total_duration = (self.end_time - self.start_time).total_seconds()
        self.success_rate = self.passed_count / max(1, len(self.results))
        
        # Calculate overall grade
        if self.success_rate >= 1.0:
            self.overall_grade = "EXCELLENT"
        elif self.success_rate >= 0.9:
            self.overall_grade = "GOOD"
        elif self.success_rate >= 0.8:
            self.overall_grade = "ACCEPTABLE"
        else:
            self.overall_grade = "NEEDS_IMPROVEMENT"


# ============================================================================
# HIGH-PERFORMANCE SYSTEM VALIDATOR
# ============================================================================

class HighPerformanceSystemValidator:
    """Comprehensive validator for the high-performance classification system."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.summary = ValidationSummary()
        self.summary.validation_config = config
        
        # Output directory setup
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Validator initialized: {config.validation_level} level")
    
    async def run_validation(self) -> ValidationSummary:
        """Run the complete validation suite."""
        
        self.logger.info(f"Starting {self.config.validation_level} validation")
        
        try:
            # System info gathering
            await self._gather_system_info()
            
            # Basic validation tests
            await self._run_basic_validation()
            
            # Performance validation
            if self.config.validation_level != ValidationLevel.BASIC:
                await self._run_performance_validation()
            
            # Load testing
            if self.config.enable_load_testing and self.config.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
                await self._run_load_testing()
            
            # Stress testing
            if self.config.enable_stress_testing and self.config.validation_level == ValidationLevel.PRODUCTION:
                await self._run_stress_testing()
            
            # Cache validation
            if self.config.enable_cache_testing:
                await self._run_cache_validation()
            
            # Generate recommendations
            self._generate_recommendations()
            
            # Export results
            if self.config.export_results:
                await self._export_results()
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.summary.add_result(ValidationResult(
                test_name="validation_execution",
                passed=False,
                duration_seconds=0,
                details={"error": str(e)},
                error_message=str(e)
            ))
        
        finally:
            self.summary.finalize()
        
        return self.summary
    
    async def _gather_system_info(self):
        """Gather system information for validation context."""
        
        test_start = time.time()
        
        try:
            import psutil
            import platform
            
            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "validation_level": self.config.validation_level,
                "target_response_time_ms": self.config.target_response_time_ms,
                "max_response_time_ms": self.config.max_response_time_ms,
                "components_available": COMPONENTS_AVAILABLE
            }
            
            self.summary.system_info = system_info
            
            self.summary.add_result(ValidationResult(
                test_name="system_info_gathering",
                passed=True,
                duration_seconds=time.time() - test_start,
                details=system_info
            ))
            
            self.logger.info(f"System info: {platform.platform()}, {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="system_info_gathering",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
    
    async def _run_basic_validation(self):
        """Run basic system validation tests."""
        
        self.logger.info("Running basic validation tests...")
        
        # Test 1: System initialization
        await self._test_system_initialization()
        
        # Test 2: Single query classification
        await self._test_single_query_classification()
        
        # Test 3: Multiple query types
        await self._test_multiple_query_types()
        
        # Test 4: Component health checks
        await self._test_component_health_checks()
    
    async def _test_system_initialization(self):
        """Test system initialization and basic functionality."""
        
        test_start = time.time()
        
        try:
            if not COMPONENTS_AVAILABLE:
                raise ImportError("Required components not available")
            
            # Test system creation
            hp_config = HighPerformanceConfig(
                target_response_time_ms=self.config.target_response_time_ms,
                max_response_time_ms=self.config.max_response_time_ms
            )
            
            async with high_performance_classification_context(hp_config) as hp_system:
                # Verify system components
                assert hp_system is not None, "System not initialized"
                assert hp_system.cache is not None, "Cache not initialized"
                assert hp_system.request_optimizer is not None, "Request optimizer not initialized"
                assert hp_system.llm_optimizer is not None, "LLM optimizer not initialized"
                assert hp_system.resource_manager is not None, "Resource manager not initialized"
                
                # Test basic configuration
                assert hp_system.config.target_response_time_ms == self.config.target_response_time_ms
                
            self.summary.add_result(ValidationResult(
                test_name="system_initialization",
                passed=True,
                duration_seconds=time.time() - test_start,
                details={
                    "target_response_time_ms": self.config.target_response_time_ms,
                    "components_initialized": ["cache", "request_optimizer", "llm_optimizer", "resource_manager"]
                }
            ))
            
            self.logger.info("✓ System initialization test passed")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="system_initialization",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ System initialization test failed: {e}")
    
    async def _test_single_query_classification(self):
        """Test single query classification performance."""
        
        test_start = time.time()
        
        try:
            hp_config = HighPerformanceConfig(
                target_response_time_ms=self.config.target_response_time_ms
            )
            
            async with high_performance_classification_context(hp_config) as hp_system:
                # Test query
                query = "What is metabolomics analysis in clinical research?"
                
                # Execute classification
                classify_start = time.time()
                result, metadata = await hp_system.classify_query_optimized(query)
                response_time = (time.time() - classify_start) * 1000
                
                # Validate result
                assert isinstance(result, ClassificationResult), f"Invalid result type: {type(result)}"
                assert result.category in ["GENERAL", "KNOWLEDGE_GRAPH", "REAL_TIME"], f"Invalid category: {result.category}"
                assert 0.0 <= result.confidence <= 1.0, f"Invalid confidence: {result.confidence}"
                assert len(result.reasoning) > 0, "Empty reasoning"
                
                # Validate metadata
                assert isinstance(metadata, dict), "Invalid metadata type"
                assert "response_time_ms" in metadata, "Missing response time in metadata"
                assert "optimizations_applied" in metadata, "Missing optimizations in metadata"
                
                # Validate performance
                assert response_time <= self.config.max_response_time_ms, f"Response time {response_time:.1f}ms exceeds {self.config.max_response_time_ms}ms"
                
                target_met = response_time <= self.config.target_response_time_ms
                
            self.summary.add_result(ValidationResult(
                test_name="single_query_classification",
                passed=True,
                duration_seconds=time.time() - test_start,
                details={
                    "query": query,
                    "classification_category": result.category,
                    "confidence": result.confidence,
                    "response_time_ms": response_time,
                    "target_met": target_met,
                    "optimizations_applied": metadata.get("optimizations_applied", [])
                }
            ))
            
            self.logger.info(f"✓ Single query test passed: {response_time:.1f}ms, {result.category}")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="single_query_classification",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ Single query test failed: {e}")
    
    async def _test_multiple_query_types(self):
        """Test multiple query types for classification accuracy and performance."""
        
        test_start = time.time()
        
        try:
            test_queries = [
                ("What is metabolomics?", "GENERAL"),
                ("LC-MS analysis for biomarker identification", "KNOWLEDGE_GRAPH"),
                ("Latest research in clinical metabolomics 2025", "REAL_TIME"),
                ("Relationship between glucose metabolism and diabetes", "KNOWLEDGE_GRAPH"),
                ("Statistical analysis of metabolomics data", "GENERAL"),
                ("Recent advances in biomarker discovery", "REAL_TIME")
            ]
            
            hp_config = HighPerformanceConfig(
                target_response_time_ms=self.config.target_response_time_ms
            )
            
            results_details = []
            all_passed = True
            total_response_time = 0
            
            async with high_performance_classification_context(hp_config) as hp_system:
                for query_text, expected_category in test_queries:
                    try:
                        classify_start = time.time()
                        result, metadata = await hp_system.classify_query_optimized(query_text)
                        response_time = (time.time() - classify_start) * 1000
                        
                        total_response_time += response_time
                        
                        # Validate basic result structure
                        query_passed = (
                            isinstance(result, ClassificationResult) and
                            result.category in ["GENERAL", "KNOWLEDGE_GRAPH", "REAL_TIME"] and
                            0.0 <= result.confidence <= 1.0 and
                            len(result.reasoning) > 0 and
                            response_time <= self.config.max_response_time_ms
                        )
                        
                        if not query_passed:
                            all_passed = False
                        
                        results_details.append({
                            "query": query_text[:50] + "..." if len(query_text) > 50 else query_text,
                            "expected_category": expected_category,
                            "actual_category": result.category,
                            "confidence": result.confidence,
                            "response_time_ms": response_time,
                            "passed": query_passed
                        })
                        
                    except Exception as e:
                        all_passed = False
                        results_details.append({
                            "query": query_text[:50] + "...",
                            "expected_category": expected_category,
                            "error": str(e),
                            "passed": False
                        })
            
            avg_response_time = total_response_time / len(test_queries) if test_queries else 0
            target_compliance = sum(1 for r in results_details if r.get("response_time_ms", 0) <= self.config.target_response_time_ms) / len(test_queries)
            
            self.summary.add_result(ValidationResult(
                test_name="multiple_query_types",
                passed=all_passed,
                duration_seconds=time.time() - test_start,
                details={
                    "total_queries": len(test_queries),
                    "avg_response_time_ms": avg_response_time,
                    "target_compliance_rate": target_compliance,
                    "query_results": results_details
                }
            ))
            
            if all_passed:
                self.logger.info(f"✓ Multiple query types test passed: {avg_response_time:.1f}ms average")
            else:
                self.logger.warning(f"⚠ Multiple query types test had issues: {avg_response_time:.1f}ms average")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="multiple_query_types",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ Multiple query types test failed: {e}")
    
    async def _test_component_health_checks(self):
        """Test individual component health and functionality."""
        
        test_start = time.time()
        
        try:
            hp_config = HighPerformanceConfig(
                target_response_time_ms=self.config.target_response_time_ms
            )
            
            component_health = {}
            
            async with high_performance_classification_context(hp_config) as hp_system:
                # Cache health check
                try:
                    cache_stats = hp_system.cache.get_cache_stats()
                    component_health["cache"] = {
                        "status": "healthy",
                        "l1_cache_size": cache_stats["l1_cache"]["size"],
                        "total_requests": cache_stats["overall"]["total_requests"]
                    }
                except Exception as e:
                    component_health["cache"] = {"status": "error", "error": str(e)}
                
                # Request optimizer health check
                try:
                    optimizer_stats = hp_system.request_optimizer.get_optimization_stats()
                    component_health["request_optimizer"] = {
                        "status": "healthy",
                        "total_requests": optimizer_stats["total_requests"],
                        "batching_enabled": optimizer_stats["batching_enabled"],
                        "deduplication_enabled": optimizer_stats["deduplication_enabled"]
                    }
                except Exception as e:
                    component_health["request_optimizer"] = {"status": "error", "error": str(e)}
                
                # Resource manager health check
                try:
                    resource_stats = hp_system.resource_manager.get_resource_stats()
                    component_health["resource_manager"] = {
                        "status": "healthy",
                        "cpu_usage_percent": resource_stats["cpu"]["current_usage_percent"],
                        "memory_usage_percent": resource_stats["memory"]["current_usage_percent"],
                        "max_worker_threads": resource_stats["threading"]["max_worker_threads"]
                    }
                except Exception as e:
                    component_health["resource_manager"] = {"status": "error", "error": str(e)}
                
                # LLM optimizer health check
                try:
                    llm_stats = hp_system.llm_optimizer.get_llm_optimization_stats()
                    component_health["llm_optimizer"] = {
                        "status": "healthy",
                        "total_requests": llm_stats["total_requests"],
                        "cache_hit_rate": llm_stats["cache_hit_rate"],
                        "token_optimization_enabled": llm_stats["token_optimization_enabled"]
                    }
                except Exception as e:
                    component_health["llm_optimizer"] = {"status": "error", "error": str(e)}
            
            # Determine overall health
            healthy_components = sum(1 for comp in component_health.values() if comp["status"] == "healthy")
            total_components = len(component_health)
            all_healthy = healthy_components == total_components
            
            self.summary.add_result(ValidationResult(
                test_name="component_health_checks",
                passed=all_healthy,
                duration_seconds=time.time() - test_start,
                details={
                    "total_components": total_components,
                    "healthy_components": healthy_components,
                    "component_status": component_health
                }
            ))
            
            if all_healthy:
                self.logger.info(f"✓ Component health checks passed: {healthy_components}/{total_components} healthy")
            else:
                self.logger.warning(f"⚠ Component health checks issues: {healthy_components}/{total_components} healthy")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="component_health_checks",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ Component health checks failed: {e}")
    
    async def _run_performance_validation(self):
        """Run performance validation tests."""
        
        self.logger.info("Running performance validation tests...")
        
        # Quick performance test
        await self._run_quick_performance_test()
        
        # Response time consistency test
        await self._run_response_time_consistency_test()
        
        # Concurrent request test
        if self.config.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
            await self._run_concurrent_request_test()
    
    async def _run_quick_performance_test(self):
        """Run quick performance test using the benchmark suite."""
        
        test_start = time.time()
        
        try:
            test_requests = self.config.test_params["quick_test_requests"]
            
            # Run benchmark
            results = await run_quick_performance_test(
                target_response_time_ms=self.config.target_response_time_ms,
                total_requests=test_requests
            )
            
            # Evaluate results
            performance_passed = (
                results.metrics.avg_response_time_ms <= self.config.max_response_time_ms and
                results.metrics.success_rate >= 0.95 and
                results.metrics.target_compliance_rate >= 0.90
            )
            
            self.summary.add_result(ValidationResult(
                test_name="quick_performance_test",
                passed=performance_passed,
                duration_seconds=time.time() - test_start,
                details={
                    "total_requests": test_requests,
                    "avg_response_time_ms": results.metrics.avg_response_time_ms,
                    "p95_response_time_ms": results.metrics.p95_response_time_ms,
                    "success_rate": results.metrics.success_rate,
                    "target_compliance_rate": results.metrics.target_compliance_rate,
                    "performance_grade": results.metrics.performance_grade.value,
                    "cache_hit_rate": results.metrics.cache_hit_rate
                }
            ))
            
            if performance_passed:
                self.logger.info(f"✓ Quick performance test passed: {results.metrics.avg_response_time_ms:.1f}ms average, {results.metrics.performance_grade.value}")
            else:
                self.logger.warning(f"⚠ Quick performance test issues: {results.metrics.avg_response_time_ms:.1f}ms average, {results.metrics.performance_grade.value}")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="quick_performance_test",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ Quick performance test failed: {e}")
    
    async def _run_response_time_consistency_test(self):
        """Test response time consistency across multiple requests."""
        
        test_start = time.time()
        
        try:
            hp_config = HighPerformanceConfig(
                target_response_time_ms=self.config.target_response_time_ms
            )
            
            test_query = "Analyze the metabolomics pathway for biomarker discovery"
            num_requests = 30
            response_times = []
            
            async with high_performance_classification_context(hp_config) as hp_system:
                for i in range(num_requests):
                    classify_start = time.time()
                    result, metadata = await hp_system.classify_query_optimized(test_query)
                    response_time = (time.time() - classify_start) * 1000
                    response_times.append(response_time)
            
            # Calculate consistency metrics
            import statistics
            avg_time = statistics.mean(response_times)
            std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
            min_time = min(response_times)
            max_time = max(response_times)
            
            # Consistency criteria
            consistency_passed = (
                avg_time <= self.config.target_response_time_ms and
                max_time <= self.config.max_response_time_ms and
                std_dev <= 200  # Less than 200ms standard deviation
            )
            
            self.summary.add_result(ValidationResult(
                test_name="response_time_consistency",
                passed=consistency_passed,
                duration_seconds=time.time() - test_start,
                details={
                    "num_requests": num_requests,
                    "avg_response_time_ms": avg_time,
                    "std_dev_ms": std_dev,
                    "min_response_time_ms": min_time,
                    "max_response_time_ms": max_time,
                    "consistency_coefficient": std_dev / avg_time if avg_time > 0 else 0
                }
            ))
            
            if consistency_passed:
                self.logger.info(f"✓ Response time consistency test passed: {avg_time:.1f}ms ± {std_dev:.1f}ms")
            else:
                self.logger.warning(f"⚠ Response time consistency issues: {avg_time:.1f}ms ± {std_dev:.1f}ms")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="response_time_consistency",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ Response time consistency test failed: {e}")
    
    async def _run_concurrent_request_test(self):
        """Test performance under concurrent load."""
        
        test_start = time.time()
        
        try:
            hp_config = HighPerformanceConfig(
                target_response_time_ms=self.config.target_response_time_ms
            )
            
            test_query = "Clinical metabolomics analysis workflow optimization"
            concurrent_users = 15
            requests_per_user = 5
            
            async def user_requests(user_id):
                """Simulate requests from a single user."""
                user_times = []
                async with high_performance_classification_context(hp_config) as hp_system:
                    for _ in range(requests_per_user):
                        classify_start = time.time()
                        result, metadata = await hp_system.classify_query_optimized(test_query)
                        response_time = (time.time() - classify_start) * 1000
                        user_times.append(response_time)
                        await asyncio.sleep(0.1)  # Brief pause
                return user_times
            
            # Execute concurrent users
            concurrent_start = time.time()
            user_tasks = [asyncio.create_task(user_requests(i)) for i in range(concurrent_users)]
            all_user_times = await asyncio.gather(*user_tasks)
            concurrent_duration = time.time() - concurrent_start
            
            # Flatten results
            all_response_times = [time for user_times in all_user_times for time in user_times]
            total_requests = len(all_response_times)
            
            # Calculate metrics
            import statistics
            avg_time = statistics.mean(all_response_times)
            max_time = max(all_response_times)
            throughput = total_requests / concurrent_duration
            failures = sum(1 for t in all_response_times if t > self.config.max_response_time_ms)
            
            # Performance criteria
            concurrent_passed = (
                avg_time <= self.config.target_response_time_ms * 1.2 and  # Allow 20% degradation under load
                failures == 0 and
                throughput >= 10  # Minimum 10 RPS
            )
            
            self.summary.add_result(ValidationResult(
                test_name="concurrent_request_test",
                passed=concurrent_passed,
                duration_seconds=time.time() - test_start,
                details={
                    "concurrent_users": concurrent_users,
                    "requests_per_user": requests_per_user,
                    "total_requests": total_requests,
                    "avg_response_time_ms": avg_time,
                    "max_response_time_ms": max_time,
                    "throughput_rps": throughput,
                    "failures": failures,
                    "test_duration_seconds": concurrent_duration
                }
            ))
            
            if concurrent_passed:
                self.logger.info(f"✓ Concurrent request test passed: {avg_time:.1f}ms average, {throughput:.1f} RPS")
            else:
                self.logger.warning(f"⚠ Concurrent request test issues: {avg_time:.1f}ms average, {failures} failures")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="concurrent_request_test",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ Concurrent request test failed: {e}")
    
    async def _run_load_testing(self):
        """Run comprehensive load testing."""
        
        self.logger.info("Running load testing...")
        
        test_start = time.time()
        
        try:
            # Configure load test
            load_config = BenchmarkConfig(
                benchmark_name="validation_load_test",
                benchmark_type=BenchmarkType.LOAD,
                load_pattern=LoadPattern.WAVE,  # Wave pattern for realistic load
                concurrent_users=self.config.test_params["load_test_users"],
                total_requests=self.config.test_params["load_test_requests"],
                target_response_time_ms=self.config.target_response_time_ms,
                max_response_time_ms=self.config.max_response_time_ms,
                enable_real_time_monitoring=True,
                export_results=self.config.export_results,
                generate_plots=self.config.test_params["enable_plots"],
                output_directory=str(self.output_dir / "load_test")
            )
            
            # Run load test
            results = await run_comprehensive_benchmark(load_config)
            
            # Evaluate results
            load_passed = (
                results.metrics.avg_response_time_ms <= self.config.max_response_time_ms and
                results.metrics.success_rate >= 0.99 and
                results.metrics.target_compliance_rate >= 0.85 and
                results.metrics.performance_grade not in [PerformanceGrade.FAILING]
            )
            
            self.summary.add_result(ValidationResult(
                test_name="load_testing",
                passed=load_passed,
                duration_seconds=time.time() - test_start,
                details={
                    "load_pattern": load_config.load_pattern.value,
                    "concurrent_users": load_config.concurrent_users,
                    "total_requests": load_config.total_requests,
                    "avg_response_time_ms": results.metrics.avg_response_time_ms,
                    "p95_response_time_ms": results.metrics.p95_response_time_ms,
                    "success_rate": results.metrics.success_rate,
                    "target_compliance_rate": results.metrics.target_compliance_rate,
                    "performance_grade": results.metrics.performance_grade.value,
                    "throughput_rps": results.metrics.actual_throughput_rps,
                    "cache_hit_rate": results.metrics.cache_hit_rate
                }
            ))
            
            if load_passed:
                self.logger.info(f"✓ Load testing passed: {results.metrics.performance_grade.value}, {results.metrics.avg_response_time_ms:.1f}ms average")
            else:
                self.logger.warning(f"⚠ Load testing issues: {results.metrics.performance_grade.value}, {results.metrics.avg_response_time_ms:.1f}ms average")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="load_testing",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ Load testing failed: {e}")
    
    async def _run_stress_testing(self):
        """Run stress testing to determine system limits."""
        
        self.logger.info("Running stress testing...")
        
        test_start = time.time()
        
        try:
            # Configure stress test
            stress_requests = self.config.test_params["stress_test_requests"]
            stress_users = self.config.test_params["stress_test_users"]
            
            # Run stress test
            results = await run_stress_test(
                max_concurrent_users=stress_users,
                duration_seconds=300  # 5 minutes
            )
            
            # Evaluate stress test results (more lenient criteria)
            stress_passed = (
                results.metrics.success_rate >= 0.95 and  # Allow some failures under stress
                results.metrics.max_compliance_rate >= 0.80 and  # 80% within max limit
                results.metrics.performance_grade not in [PerformanceGrade.FAILING]
            )
            
            self.summary.add_result(ValidationResult(
                test_name="stress_testing",
                passed=stress_passed,
                duration_seconds=time.time() - test_start,
                details={
                    "max_concurrent_users": stress_users,
                    "total_requests": results.metrics.total_requests,
                    "avg_response_time_ms": results.metrics.avg_response_time_ms,
                    "p95_response_time_ms": results.metrics.p95_response_time_ms,
                    "p99_response_time_ms": results.metrics.p99_response_time_ms,
                    "success_rate": results.metrics.success_rate,
                    "max_compliance_rate": results.metrics.max_compliance_rate,
                    "performance_grade": results.metrics.performance_grade.value,
                    "throughput_rps": results.metrics.actual_throughput_rps,
                    "regression_detected": results.metrics.regression_detected
                }
            ))
            
            if stress_passed:
                self.logger.info(f"✓ Stress testing passed: {results.metrics.performance_grade.value}, {results.metrics.success_rate:.1%} success rate")
            else:
                self.logger.warning(f"⚠ Stress testing issues: {results.metrics.performance_grade.value}, {results.metrics.success_rate:.1%} success rate")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="stress_testing",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ Stress testing failed: {e}")
    
    async def _run_cache_validation(self):
        """Run cache effectiveness validation."""
        
        self.logger.info("Running cache validation...")
        
        test_start = time.time()
        
        try:
            hp_config = HighPerformanceConfig(
                target_response_time_ms=self.config.target_response_time_ms,
                l1_cache_size=1000,
                enable_cache_warming=True
            )
            
            test_queries = [
                "What is metabolomics analysis?",
                "LC-MS biomarker identification methods",
                "Pathway enrichment statistical analysis",
                "Clinical metabolomics diagnostic applications"
            ] * (self.config.test_params["cache_test_queries"] // 4)
            
            cache_miss_times = []
            cache_hit_times = []
            
            async with high_performance_classification_context(hp_config) as hp_system:
                # First round - cache misses expected
                for query in test_queries[:10]:
                    classify_start = time.time()
                    result, metadata = await hp_system.classify_query_optimized(query)
                    response_time = (time.time() - classify_start) * 1000
                    cache_miss_times.append(response_time)
                
                # Brief pause for cache population
                await asyncio.sleep(1.0)
                
                # Second round - cache hits expected
                cache_hits = 0
                for query in test_queries[:10]:
                    classify_start = time.time()
                    result, metadata = await hp_system.classify_query_optimized(query)
                    response_time = (time.time() - classify_start) * 1000
                    cache_hit_times.append(response_time)
                    
                    if metadata.get("cache_hit", False):
                        cache_hits += 1
                
                # Get cache statistics
                cache_stats = hp_system.cache.get_cache_stats()
            
            # Calculate cache effectiveness
            import statistics
            avg_miss_time = statistics.mean(cache_miss_times) if cache_miss_times else 0
            avg_hit_time = statistics.mean(cache_hit_times) if cache_hit_times else 0
            cache_hit_rate = cache_hits / len(test_queries[:10]) if test_queries else 0
            improvement_factor = avg_miss_time / avg_hit_time if avg_hit_time > 0 else 1
            
            # Cache validation criteria
            cache_passed = (
                cache_hit_rate >= 0.7 and  # At least 70% cache hit rate
                avg_hit_time <= avg_miss_time * 0.8 and  # Cache should be at least 20% faster
                cache_stats["overall"]["hit_rate"] > 0
            )
            
            self.summary.add_result(ValidationResult(
                test_name="cache_validation",
                passed=cache_passed,
                duration_seconds=time.time() - test_start,
                details={
                    "cache_hit_rate": cache_hit_rate,
                    "avg_cache_miss_time_ms": avg_miss_time,
                    "avg_cache_hit_time_ms": avg_hit_time,
                    "improvement_factor": improvement_factor,
                    "l1_cache_hit_rate": cache_stats["l1_cache"]["hit_rate"],
                    "overall_cache_hit_rate": cache_stats["overall"]["hit_rate"],
                    "total_cache_requests": cache_stats["overall"]["total_requests"]
                }
            ))
            
            if cache_passed:
                self.logger.info(f"✓ Cache validation passed: {cache_hit_rate:.1%} hit rate, {improvement_factor:.1f}x improvement")
            else:
                self.logger.warning(f"⚠ Cache validation issues: {cache_hit_rate:.1%} hit rate, {improvement_factor:.1f}x improvement")
            
        except Exception as e:
            self.summary.add_result(ValidationResult(
                test_name="cache_validation",
                passed=False,
                duration_seconds=time.time() - test_start,
                details={},
                error_message=str(e)
            ))
            
            self.logger.error(f"✗ Cache validation failed: {e}")
    
    def _generate_recommendations(self):
        """Generate optimization recommendations based on validation results."""
        
        recommendations = []
        
        # Analyze failed tests
        failed_tests = [r for r in self.summary.results if not r.passed]
        
        if failed_tests:
            recommendations.append("CRITICAL: Address failed validation tests before production deployment")
            
            for test in failed_tests:
                if test.test_name == "system_initialization":
                    recommendations.append("- Fix system initialization issues - check component dependencies")
                elif test.test_name == "single_query_classification":
                    recommendations.append("- Resolve basic classification functionality problems")
                elif test.test_name == "response_time_consistency":
                    recommendations.append("- Improve response time consistency - check for resource contention")
                elif test.test_name == "concurrent_request_test":
                    recommendations.append("- Optimize concurrent processing - consider increasing parallelism limits")
                elif test.test_name == "load_testing":
                    recommendations.append("- Address load testing issues - review cache configuration and resource limits")
                elif test.test_name == "stress_testing":
                    recommendations.append("- Improve system resilience under stress - implement better error handling")
                elif test.test_name == "cache_validation":
                    recommendations.append("- Optimize caching strategy - review cache warming and TTL settings")
        
        # Performance recommendations based on successful tests
        performance_tests = [r for r in self.summary.results if r.passed and "performance" in r.test_name]
        
        for test in performance_tests:
            if test.details.get("avg_response_time_ms", 0) > self.config.target_response_time_ms:
                recommendations.append("- Consider increasing cache sizes to improve average response times")
            
            if test.details.get("target_compliance_rate", 1) < 0.95:
                recommendations.append("- Implement more aggressive optimization to improve target compliance")
        
        # Cache recommendations
        cache_tests = [r for r in self.summary.results if r.test_name == "cache_validation" and r.passed]
        
        for test in cache_tests:
            if test.details.get("cache_hit_rate", 0) < 0.8:
                recommendations.append("- Improve cache hit rate through better cache warming strategies")
            
            if test.details.get("improvement_factor", 1) < 2.0:
                recommendations.append("- Optimize cache performance to achieve better speed improvements")
        
        # System resource recommendations
        component_tests = [r for r in self.summary.results if r.test_name == "component_health_checks" and r.passed]
        
        for test in component_tests:
            resource_details = test.details.get("component_status", {}).get("resource_manager", {})
            
            if resource_details.get("status") == "healthy":
                cpu_usage = resource_details.get("cpu_usage_percent", 0)
                memory_usage = resource_details.get("memory_usage_percent", 0)
                
                if cpu_usage > 80:
                    recommendations.append("- Monitor CPU usage - consider horizontal scaling for high load scenarios")
                
                if memory_usage > 80:
                    recommendations.append("- Monitor memory usage - consider memory optimization strategies")
        
        # General recommendations based on validation level
        if self.config.validation_level == ValidationLevel.PRODUCTION:
            recommendations.extend([
                "- Set up production monitoring and alerting",
                "- Implement automated performance regression testing",
                "- Plan for horizontal scaling based on load requirements",
                "- Establish performance baselines and SLA monitoring"
            ])
        
        # If no specific recommendations and all tests passed
        if not recommendations and self.summary.success_rate == 1.0:
            recommendations.append("✓ System performs excellently - ready for production deployment")
            recommendations.append("- Consider implementing continuous performance monitoring")
            recommendations.append("- Plan for capacity scaling based on expected load growth")
        
        self.summary.recommendations = recommendations
    
    async def _export_results(self):
        """Export validation results to files."""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export JSON summary
            summary_data = {
                "validation_config": {
                    "validation_level": self.config.validation_level,
                    "target_response_time_ms": self.config.target_response_time_ms,
                    "max_response_time_ms": self.config.max_response_time_ms
                },
                "summary": {
                    "start_time": self.summary.start_time.isoformat(),
                    "end_time": self.summary.end_time.isoformat() if self.summary.end_time else None,
                    "total_duration_seconds": getattr(self.summary, 'total_duration', 0),
                    "passed_count": self.summary.passed_count,
                    "failed_count": self.summary.failed_count,
                    "success_rate": getattr(self.summary, 'success_rate', 0),
                    "overall_grade": self.summary.overall_grade
                },
                "system_info": self.summary.system_info,
                "test_results": [
                    {
                        "test_name": r.test_name,
                        "passed": r.passed,
                        "duration_seconds": r.duration_seconds,
                        "details": r.details,
                        "error_message": r.error_message,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in self.summary.results
                ],
                "recommendations": self.summary.recommendations
            }
            
            json_path = self.output_dir / f"validation_results_{timestamp}.json"
            with open(json_path, 'w') as f:
                import json
                json.dump(summary_data, f, indent=2, default=str)
            
            self.summary.export_paths["json_summary"] = str(json_path)
            
            # Export text report
            report_content = self._generate_text_report()
            report_path = self.output_dir / f"validation_report_{timestamp}.txt"
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            self.summary.export_paths["text_report"] = str(report_path)
            
            self.logger.info(f"Validation results exported to: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
    
    def _generate_text_report(self) -> str:
        """Generate a comprehensive text report."""
        
        report = []
        report.append("=" * 100)
        report.append("HIGH-PERFORMANCE CLASSIFICATION SYSTEM VALIDATION REPORT")
        report.append("=" * 100)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append(f"Validation Level: {self.config.validation_level}")
        report.append(f"Overall Grade: {self.summary.overall_grade}")
        report.append(f"Success Rate: {getattr(self.summary, 'success_rate', 0):.1%}")
        report.append(f"Tests Passed: {self.summary.passed_count}")
        report.append(f"Tests Failed: {self.summary.failed_count}")
        report.append(f"Total Duration: {getattr(self.summary, 'total_duration', 0):.1f} seconds")
        report.append("")
        
        # System Information
        if self.summary.system_info:
            report.append("SYSTEM INFORMATION")
            report.append("-" * 20)
            for key, value in self.summary.system_info.items():
                report.append(f"{key}: {value}")
            report.append("")
        
        # Test Results
        report.append("DETAILED TEST RESULTS")
        report.append("-" * 25)
        
        for result in self.summary.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report.append(f"{status} - {result.test_name}")
            report.append(f"  Duration: {result.duration_seconds:.2f} seconds")
            
            if result.error_message:
                report.append(f"  Error: {result.error_message}")
            
            # Key details
            if result.details:
                key_details = {}
                if "avg_response_time_ms" in result.details:
                    key_details["Avg Response Time"] = f"{result.details['avg_response_time_ms']:.1f}ms"
                if "success_rate" in result.details:
                    key_details["Success Rate"] = f"{result.details['success_rate']:.1%}"
                if "target_compliance_rate" in result.details:
                    key_details["Target Compliance"] = f"{result.details['target_compliance_rate']:.1%}"
                if "cache_hit_rate" in result.details:
                    key_details["Cache Hit Rate"] = f"{result.details['cache_hit_rate']:.1%}"
                if "performance_grade" in result.details:
                    key_details["Performance Grade"] = result.details['performance_grade']
                
                for key, value in key_details.items():
                    report.append(f"  {key}: {value}")
            
            report.append("")
        
        # Recommendations
        if self.summary.recommendations:
            report.append("RECOMMENDATIONS")
            report.append("-" * 15)
            for i, rec in enumerate(self.summary.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Conclusion
        report.append("CONCLUSION")
        report.append("-" * 15)
        
        if self.summary.overall_grade == "EXCELLENT":
            report.append("🎉 VALIDATION SUCCESSFUL - SYSTEM READY FOR PRODUCTION")
            report.append("✓ All performance targets met consistently")
            report.append("✓ System demonstrates excellent reliability and speed")
            report.append("✓ Recommended for immediate production deployment")
        elif self.summary.overall_grade == "GOOD":
            report.append("✅ VALIDATION SUCCESSFUL - SYSTEM READY WITH MINOR OPTIMIZATIONS")
            report.append("✓ Core performance targets met")
            report.append("⚠️  Minor optimizations recommended for optimal performance")
            report.append("✓ Suitable for production deployment with monitoring")
        elif self.summary.overall_grade == "ACCEPTABLE":
            report.append("⚠️  VALIDATION ACCEPTABLE - OPTIMIZATION REQUIRED")
            report.append("⚠️  Basic performance requirements met")
            report.append("⚠️  Significant optimizations needed for production readiness")
            report.append("❌ Review recommendations before deployment")
        else:
            report.append("❌ VALIDATION FAILED - SYSTEM NOT READY FOR PRODUCTION")
            report.append("❌ Critical issues identified requiring immediate attention")
            report.append("❌ System does not meet minimum performance requirements")
            report.append("❌ Address all failed tests before proceeding")
        
        report.append("")
        
        # Export information
        if self.summary.export_paths:
            report.append("EXPORTED ARTIFACTS")
            report.append("-" * 20)
            for artifact_type, path in self.summary.export_paths.items():
                report.append(f"{artifact_type.replace('_', ' ').title()}: {path}")
            report.append("")
        
        report.append("=" * 100)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 100)
        
        return "\n".join(report)


# ============================================================================
# MAIN VALIDATION FUNCTIONS
# ============================================================================

async def validate_high_performance_system(validation_level: str = ValidationLevel.STANDARD,
                                          target_response_time_ms: float = 1500.0,
                                          export_results: bool = True) -> ValidationSummary:
    """
    Run comprehensive validation of the high-performance classification system.
    
    Args:
        validation_level: Level of validation thoroughness
        target_response_time_ms: Target response time in milliseconds
        export_results: Whether to export results to files
        
    Returns:
        Complete validation summary
    """
    
    config = ValidationConfig(
        validation_level=validation_level,
        target_response_time_ms=target_response_time_ms,
        export_results=export_results
    )
    
    validator = HighPerformanceSystemValidator(config)
    return await validator.run_validation()


async def run_quick_validation() -> ValidationSummary:
    """Run quick validation for development testing."""
    
    return await validate_high_performance_system(
        validation_level=ValidationLevel.BASIC,
        target_response_time_ms=2000.0,
        export_results=False
    )


async def run_production_validation() -> ValidationSummary:
    """Run comprehensive production readiness validation."""
    
    return await validate_high_performance_system(
        validation_level=ValidationLevel.PRODUCTION,
        target_response_time_ms=1500.0,
        export_results=True
    )


def main():
    """Main validation script entry point."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('validation.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Check if components are available
    if not COMPONENTS_AVAILABLE:
        logger.error("Required performance components are not available")
        logger.error("Please ensure all dependencies are installed and components are properly imported")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="High-Performance Classification System Validation")
    parser.add_argument("--level", 
                       choices=[ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION],
                       default=ValidationLevel.STANDARD,
                       help="Validation level (default: standard)")
    parser.add_argument("--target", 
                       type=float, 
                       default=1500.0,
                       help="Target response time in ms (default: 1500)")
    parser.add_argument("--no-export", 
                       action="store_true",
                       help="Disable result export")
    
    args = parser.parse_args()
    
    async def run_validation():
        """Run the validation asynchronously."""
        
        logger.info("=" * 80)
        logger.info("HIGH-PERFORMANCE CLASSIFICATION SYSTEM VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Validation Level: {args.level}")
        logger.info(f"Target Response Time: {args.target}ms")
        logger.info(f"Export Results: {not args.no_export}")
        logger.info("")
        
        try:
            # Run validation
            summary = await validate_high_performance_system(
                validation_level=args.level,
                target_response_time_ms=args.target,
                export_results=not args.no_export
            )
            
            # Print summary
            logger.info("=" * 80)
            logger.info("VALIDATION COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Overall Grade: {summary.overall_grade}")
            logger.info(f"Success Rate: {getattr(summary, 'success_rate', 0):.1%}")
            logger.info(f"Tests Passed: {summary.passed_count}/{len(summary.results)}")
            
            if summary.failed_count > 0:
                logger.warning(f"Failed Tests: {summary.failed_count}")
                for result in summary.results:
                    if not result.passed:
                        logger.warning(f"  - {result.test_name}: {result.error_message or 'Performance criteria not met'}")
            
            if summary.recommendations:
                logger.info("\nKey Recommendations:")
                for rec in summary.recommendations[:5]:  # Top 5 recommendations
                    logger.info(f"  • {rec}")
            
            # Export paths
            if summary.export_paths:
                logger.info("\nExported Files:")
                for artifact_type, path in summary.export_paths.items():
                    logger.info(f"  • {artifact_type}: {path}")
            
            logger.info("=" * 80)
            
            # Exit with appropriate code
            if summary.overall_grade in ["EXCELLENT", "GOOD"]:
                logger.info("🎉 VALIDATION SUCCESSFUL - SYSTEM READY FOR PRODUCTION")
                sys.exit(0)
            elif summary.overall_grade == "ACCEPTABLE":
                logger.warning("⚠️  VALIDATION ACCEPTABLE - REVIEW RECOMMENDATIONS")
                sys.exit(1)
            else:
                logger.error("❌ VALIDATION FAILED - SYSTEM NOT READY")
                sys.exit(2)
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(3)
    
    # Run the validation
    asyncio.run(run_validation())


if __name__ == "__main__":
    main()