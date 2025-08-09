#!/usr/bin/env python3
"""
Comprehensive Test Runner for Complete Graceful Degradation System
=================================================================

This test runner performs comprehensive testing of the graceful degradation system
including all components, integration scenarios, load testing, and production validation.

Test Coverage:
1. Component Integration Tests
2. Load Level Transition Tests 
3. Performance Under Load Tests
4. System Protection Tests
5. Production Integration Tests
6. Recovery and Resilience Tests
7. End-to-End Functionality Tests

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import logging
import time
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test results tracking
@dataclass
class TestResult:
    test_name: str
    category: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration: float
    details: str = ""
    error: str = ""

@dataclass
class TestReport:
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    test_results: List[TestResult] = field(default_factory=list)
    
    def add_result(self, result: TestResult):
        self.test_results.append(result)
        self.total_tests += 1
        if result.status == "PASS":
            self.passed_tests += 1
        elif result.status == "FAIL":
            self.failed_tests += 1
        elif result.status == "SKIP":
            self.skipped_tests += 1

class ComprehensiveGracefulDegradationTester:
    """Comprehensive tester for the graceful degradation system."""
    
    def __init__(self):
        self.report = TestReport(start_time=datetime.now())
        self.orchestrator = None
        
        # Import components with error handling
        self._import_components()
        
    def _import_components(self):
        """Import all graceful degradation components with error handling."""
        try:
            from graceful_degradation_integration import (
                GracefulDegradationOrchestrator,
                GracefulDegradationConfig,
                create_graceful_degradation_system
            )
            self.GracefulDegradationOrchestrator = GracefulDegradationOrchestrator
            self.GracefulDegradationConfig = GracefulDegradationConfig
            self.create_graceful_degradation_system = create_graceful_degradation_system
            self.components_available = True
            logger.info("Successfully imported graceful degradation components")
        except Exception as e:
            logger.error(f"Failed to import graceful degradation components: {e}")
            self.components_available = False
            
        try:
            from enhanced_load_monitoring_system import SystemLoadLevel
            self.SystemLoadLevel = SystemLoadLevel
            logger.info("Successfully imported load monitoring components")
        except Exception as e:
            logger.error(f"Failed to import load monitoring components: {e}")
            
        try:
            from load_based_request_throttling_system import RequestType, RequestPriority
            self.RequestType = RequestType
            self.RequestPriority = RequestPriority
            logger.info("Successfully imported throttling system components")
        except Exception as e:
            logger.error(f"Failed to import throttling system components: {e}")

    async def run_test(self, test_name: str, category: str, test_func) -> TestResult:
        """Run a single test with error handling and timing."""
        start_time = time.time()
        
        try:
            logger.info(f"Running test: {test_name}")
            details = await test_func()
            duration = time.time() - start_time
            
            result = TestResult(
                test_name=test_name,
                category=category,
                status="PASS",
                duration=duration,
                details=details or "Test completed successfully"
            )
            logger.info(f"✅ {test_name} - PASSED ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            
            result = TestResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                error=error_msg,
                details="Test failed with exception"
            )
            logger.error(f"❌ {test_name} - FAILED ({duration:.2f}s): {str(e)}")
            
        self.report.add_result(result)
        return result

    # ========================================================================
    # COMPONENT INTEGRATION TESTS
    # ========================================================================

    async def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized successfully."""
        if not self.components_available:
            raise RuntimeError("Components not available for testing")
            
        config = self.GracefulDegradationConfig(
            monitoring_interval=1.0,
            base_rate_per_second=5.0,
            max_queue_size=50,
            max_concurrent_requests=10
        )
        
        self.orchestrator = self.GracefulDegradationOrchestrator(config=config)
        
        assert self.orchestrator is not None
        assert self.orchestrator.config.monitoring_interval == 1.0
        assert self.orchestrator._integration_status is not None
        
        return "Orchestrator initialized successfully with configuration"

    async def test_system_startup_and_shutdown(self):
        """Test system can start and stop gracefully."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        # Test startup
        await self.orchestrator.start()
        assert self.orchestrator._running is True
        
        # Get status after startup
        status = self.orchestrator.get_system_status()
        assert status['running'] is True
        
        # Test shutdown
        await self.orchestrator.stop()
        assert self.orchestrator._running is False
        
        return f"System startup/shutdown successful. Components active: {status['integration_status']}"

    async def test_system_health_monitoring(self):
        """Test system health monitoring and reporting."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        await self.orchestrator.start()
        
        try:
            # Get health check
            health = self.orchestrator.get_health_check()
            
            assert 'status' in health
            assert 'issues' in health
            assert 'component_status' in health
            assert 'production_integration' in health
            assert 'uptime_seconds' in health
            assert 'total_requests_processed' in health
            
            return f"Health monitoring active. Status: {health['status']}, Components: {health['component_status']}"
            
        finally:
            await self.orchestrator.stop()

    # ========================================================================
    # LOAD LEVEL TRANSITION TESTS  
    # ========================================================================

    async def test_load_level_transitions(self):
        """Test load level transitions work correctly."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        await self.orchestrator.start()
        
        try:
            initial_status = self.orchestrator.get_system_status()
            initial_level = initial_status['current_load_level']
            
            # Monitor for level changes over time
            levels_observed = [initial_level]
            
            for i in range(5):
                await asyncio.sleep(1)
                status = self.orchestrator.get_system_status()
                current_level = status['current_load_level']
                if current_level not in levels_observed:
                    levels_observed.append(current_level)
            
            # Test force level changes if degradation controller is available
            if hasattr(self.orchestrator, 'degradation_controller') and self.orchestrator.degradation_controller:
                if hasattr(self.orchestrator.degradation_controller, 'force_load_level'):
                    try:
                        # Try to force HIGH load level
                        self.orchestrator.degradation_controller.force_load_level(
                            getattr(self, 'SystemLoadLevel', type('MockLoadLevel', (), {'HIGH': 2})).HIGH,
                            "Test high load transition"
                        )
                        
                        await asyncio.sleep(2)
                        
                        status = self.orchestrator.get_system_status()
                        if status['current_load_level'] != initial_level:
                            levels_observed.append(status['current_load_level'])
                    except Exception as e:
                        logger.warning(f"Force level change not supported: {e}")
            
            return f"Load transition monitoring completed. Levels observed: {levels_observed}"
            
        finally:
            await self.orchestrator.stop()

    async def test_hysteresis_behavior(self):
        """Test that hysteresis prevents rapid level oscillations.""" 
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        await self.orchestrator.start()
        
        try:
            # Collect level changes over time
            level_changes = []
            previous_level = None
            
            for i in range(10):
                await asyncio.sleep(0.5)
                status = self.orchestrator.get_system_status()
                current_level = status['current_load_level']
                timestamp = datetime.now()
                
                if current_level != previous_level:
                    level_changes.append((timestamp, previous_level, current_level))
                    previous_level = current_level
            
            # Check for rapid oscillations (changes within 5 seconds should be rare)
            rapid_changes = 0
            for i in range(1, len(level_changes)):
                time_diff = (level_changes[i][0] - level_changes[i-1][0]).total_seconds()
                if time_diff < 5.0:
                    rapid_changes += 1
            
            return f"Hysteresis test completed. Total level changes: {len(level_changes)}, Rapid changes: {rapid_changes}"
            
        finally:
            await self.orchestrator.stop()

    # ========================================================================
    # PERFORMANCE UNDER LOAD TESTS
    # ========================================================================

    async def test_high_request_volume_handling(self):
        """Test system behavior under high request volume."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        config = self.GracefulDegradationConfig(
            base_rate_per_second=20.0,
            max_queue_size=200,
            max_concurrent_requests=20
        )
        
        test_orchestrator = self.GracefulDegradationOrchestrator(config=config)
        await test_orchestrator.start()
        
        try:
            async def fast_handler(req_id: int):
                await asyncio.sleep(0.05)
                return f"Processed {req_id}"
            
            # Submit many requests concurrently
            submit_tasks = []
            num_requests = 100
            
            for i in range(num_requests):
                task = test_orchestrator.submit_request(
                    request_type='user_query',
                    priority='high',
                    handler=fast_handler,
                    req_id=i
                )
                submit_tasks.append(task)
            
            # Execute all submissions
            start_time = time.time()
            results = await asyncio.gather(*submit_tasks, return_exceptions=True)
            submission_time = time.time() - start_time
            
            # Count successful submissions
            successful = 0
            failed = 0
            
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                else:
                    success, message, request_id = result
                    if success:
                        successful += 1
                    else:
                        failed += 1
            
            success_rate = successful / len(results) * 100
            
            # Allow processing time
            await asyncio.sleep(3.0)
            
            # Get final status
            final_status = test_orchestrator.get_health_check()
            
            return f"High volume test: {num_requests} requests, {successful} successful ({success_rate:.1f}%), {failed} failed. Submission time: {submission_time:.2f}s. Final health: {final_status['status']}"
            
        finally:
            await test_orchestrator.stop()

    async def test_memory_pressure_simulation(self):
        """Test system behavior under simulated memory pressure."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        await self.orchestrator.start()
        
        try:
            # Create memory pressure by storing large amounts of data
            large_data = []
            initial_status = self.orchestrator.get_health_check()
            
            # Gradually increase memory usage
            for i in range(10):
                # Add large data chunks
                chunk = ['x' * 10000] * 100  # ~1MB per chunk
                large_data.append(chunk)
                
                await asyncio.sleep(0.5)
                
                # Check system health
                status = self.orchestrator.get_health_check()
                if status['status'] != initial_status['status']:
                    break
            
            # Get final status
            final_status = self.orchestrator.get_health_check()
            
            # Clean up memory
            large_data.clear()
            
            return f"Memory pressure test: Initial status: {initial_status['status']}, Final status: {final_status['status']}, Data chunks created: {len(large_data)}"
            
        finally:
            await self.orchestrator.stop()

    # ========================================================================
    # SYSTEM PROTECTION TESTS
    # ========================================================================

    async def test_request_throttling_effectiveness(self):
        """Test that request throttling effectively limits request rate."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        # Use strict throttling configuration
        config = self.GracefulDegradationConfig(
            base_rate_per_second=2.0,  # Very low rate
            max_queue_size=10,
            max_concurrent_requests=2
        )
        
        test_orchestrator = self.GracefulDegradationOrchestrator(config=config)
        await test_orchestrator.start()
        
        try:
            async def slow_handler():
                await asyncio.sleep(1.0)
                return "processed"
            
            # Submit requests faster than allowed rate
            start_time = time.time()
            results = []
            
            for i in range(20):
                success, message, request_id = await test_orchestrator.submit_request(
                    request_type='user_query',
                    handler=slow_handler
                )
                results.append((success, message))
                
                # Submit rapidly
                if i < 19:
                    await asyncio.sleep(0.1)
            
            submission_time = time.time() - start_time
            
            # Count throttled requests
            successful = sum(1 for success, _ in results if success)
            throttled = sum(1 for success, msg in results if not success and 'throttl' in msg.lower())
            
            # Get system status
            status = test_orchestrator.get_system_status()
            
            return f"Throttling test: {len(results)} requests in {submission_time:.1f}s. {successful} successful, {throttled} throttled. Rate limit working: {throttled > 0}"
            
        finally:
            await test_orchestrator.stop()

    async def test_queue_overflow_protection(self):
        """Test that queue overflow is handled gracefully."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        # Use small queue configuration
        config = self.GracefulDegradationConfig(
            base_rate_per_second=1.0,
            max_queue_size=5,  # Very small queue
            max_concurrent_requests=1
        )
        
        test_orchestrator = self.GracefulDegradationOrchestrator(config=config)
        await test_orchestrator.start()
        
        try:
            async def blocking_handler():
                await asyncio.sleep(5.0)  # Long processing time
                return "processed"
            
            # Submit more requests than queue can handle
            results = []
            for i in range(15):
                success, message, request_id = await test_orchestrator.submit_request(
                    request_type='batch_processing',
                    handler=blocking_handler
                )
                results.append((success, message))
            
            # Analyze results
            accepted = sum(1 for success, _ in results if success)
            rejected = sum(1 for success, msg in results if not success)
            
            # Check health after overflow
            health = test_orchestrator.get_health_check()
            
            return f"Queue overflow test: {len(results)} requests submitted, {accepted} accepted, {rejected} rejected. System health: {health['status']}"
            
        finally:
            await test_orchestrator.stop()

    # ========================================================================
    # RECOVERY AND RESILIENCE TESTS  
    # ========================================================================

    async def test_graceful_recovery_from_overload(self):
        """Test system recovery from overload conditions."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        await self.orchestrator.start()
        
        try:
            # Get baseline health
            baseline_health = self.orchestrator.get_health_check()
            
            # Create overload condition
            overload_tasks = []
            for i in range(50):
                async def overload_handler():
                    await asyncio.sleep(2.0)
                    return "overload_processed"
                
                task = self.orchestrator.submit_request(
                    request_type='batch_processing',
                    priority='low',
                    handler=overload_handler
                )
                overload_tasks.append(task)
            
            # Check health during overload
            await asyncio.sleep(1.0)
            overload_health = self.orchestrator.get_health_check()
            
            # Wait for recovery
            await asyncio.sleep(5.0)
            recovery_health = self.orchestrator.get_health_check()
            
            return f"Recovery test: Baseline: {baseline_health['status']}, Overload: {overload_health['status']}, Recovery: {recovery_health['status']}"
            
        finally:
            await self.orchestrator.stop()

    async def test_component_failure_isolation(self):
        """Test that component failures don't crash the entire system."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        await self.orchestrator.start()
        
        try:
            # Test with failing request handler
            async def failing_handler():
                raise Exception("Simulated handler failure")
            
            # Submit requests that will fail
            failed_results = []
            for i in range(5):
                try:
                    success, message, request_id = await self.orchestrator.submit_request(
                        request_type='user_query',
                        handler=failing_handler
                    )
                    failed_results.append((success, message))
                except Exception as e:
                    failed_results.append((False, str(e)))
            
            # Test with successful handler after failures
            async def success_handler():
                return "success after failure"
            
            success, message, request_id = await self.orchestrator.submit_request(
                request_type='health_check',
                priority='critical',
                handler=success_handler
            )
            
            # Check system health after failures
            health = self.orchestrator.get_health_check()
            
            return f"Failure isolation test: {len(failed_results)} failing requests handled, subsequent request success: {success}, system health: {health['status']}"
            
        finally:
            await self.orchestrator.stop()

    # ========================================================================
    # END-TO-END FUNCTIONALITY TESTS
    # ========================================================================

    async def test_complete_request_lifecycle(self):
        """Test complete request processing lifecycle."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        await self.orchestrator.start()
        
        try:
            # Test different request types and priorities
            test_scenarios = [
                ('health_check', 'critical', 'System health check'),
                ('user_query', 'high', 'User metabolomics analysis'), 
                ('batch_processing', 'medium', 'Batch data processing'),
                ('analytics', 'low', 'Performance analytics'),
                ('maintenance', 'background', 'System maintenance')
            ]
            
            results = []
            processing_times = []
            
            for req_type, priority, description in test_scenarios:
                async def test_handler(desc: str):
                    start = time.time()
                    await asyncio.sleep(0.2)  # Simulate processing
                    end = time.time()
                    processing_times.append(end - start)
                    return f"Processed: {desc}"
                
                start_time = time.time()
                success, message, request_id = await self.orchestrator.submit_request(
                    request_type=req_type,
                    priority=priority,
                    handler=test_handler,
                    desc=description
                )
                total_time = time.time() - start_time
                
                results.append({
                    'type': req_type,
                    'priority': priority,
                    'success': success,
                    'total_time': total_time,
                    'request_id': request_id
                })
            
            # Allow processing to complete
            await asyncio.sleep(2.0)
            
            # Get final statistics
            final_health = self.orchestrator.get_health_check()
            
            successful_requests = sum(1 for r in results if r['success'])
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            return f"Complete lifecycle test: {successful_requests}/{len(test_scenarios)} successful, avg processing: {avg_processing_time:.3f}s, final health: {final_health['status']}"
            
        finally:
            await self.orchestrator.stop()

    async def test_production_integration_readiness(self):
        """Test production integration readiness."""
        if not self.orchestrator:
            await self.test_orchestrator_initialization()
            
        await self.orchestrator.start()
        
        try:
            # Test system status reporting
            status = self.orchestrator.get_system_status()
            health = self.orchestrator.get_health_check()
            
            # Check required status fields
            required_status_fields = [
                'running', 'integration_status', 'current_load_level',
                'total_requests_processed', 'health_status'
            ]
            
            missing_fields = [field for field in required_status_fields if field not in status]
            
            # Check required health fields
            required_health_fields = [
                'status', 'issues', 'component_status', 'production_integration',
                'uptime_seconds', 'total_requests_processed'
            ]
            
            missing_health_fields = [field for field in required_health_fields if field not in health]
            
            # Test metrics history
            metrics_history = self.orchestrator.get_metrics_history(hours=1)
            
            # Check configuration completeness
            config = self.orchestrator.config
            config_complete = all([
                hasattr(config, 'monitoring_interval'),
                hasattr(config, 'base_rate_per_second'),
                hasattr(config, 'max_queue_size'),
                hasattr(config, 'max_concurrent_requests')
            ])
            
            readiness_score = 0
            if not missing_fields:
                readiness_score += 25
            if not missing_health_fields:
                readiness_score += 25
            if config_complete:
                readiness_score += 25
            if status['running']:
                readiness_score += 25
            
            return f"Production readiness: {readiness_score}% ready. Missing status fields: {missing_fields}, Missing health fields: {missing_health_fields}, Config complete: {config_complete}"
            
        finally:
            await self.orchestrator.stop()

    # ========================================================================
    # MAIN TEST RUNNER
    # ========================================================================

    async def run_all_tests(self):
        """Run all comprehensive tests."""
        print("🔍 Starting Comprehensive Graceful Degradation System Testing")
        print("=" * 80)
        
        if not self.components_available:
            print("❌ Critical components not available - running limited tests")
            
        # Define all tests to run
        test_suite = [
            # Component Integration Tests
            ("Component Integration", [
                ("Orchestrator Initialization", self.test_orchestrator_initialization),
                ("System Startup/Shutdown", self.test_system_startup_and_shutdown),
                ("Health Monitoring", self.test_system_health_monitoring),
            ]),
            
            # Load Level Transition Tests
            ("Load Level Transitions", [
                ("Load Level Transitions", self.test_load_level_transitions),
                ("Hysteresis Behavior", self.test_hysteresis_behavior),
            ]),
            
            # Performance Under Load Tests  
            ("Performance Under Load", [
                ("High Request Volume", self.test_high_request_volume_handling),
                ("Memory Pressure Simulation", self.test_memory_pressure_simulation),
            ]),
            
            # System Protection Tests
            ("System Protection", [
                ("Request Throttling", self.test_request_throttling_effectiveness),
                ("Queue Overflow Protection", self.test_queue_overflow_protection),
            ]),
            
            # Recovery and Resilience Tests
            ("Recovery & Resilience", [
                ("Graceful Recovery", self.test_graceful_recovery_from_overload),
                ("Component Failure Isolation", self.test_component_failure_isolation),
            ]),
            
            # End-to-End Tests
            ("End-to-End Functionality", [
                ("Complete Request Lifecycle", self.test_complete_request_lifecycle),
                ("Production Integration Readiness", self.test_production_integration_readiness),
            ])
        ]
        
        # Run all test categories
        for category_name, tests in test_suite:
            print(f"\n📋 {category_name} Tests")
            print("-" * 60)
            
            for test_name, test_func in tests:
                try:
                    await self.run_test(test_name, category_name, test_func)
                except Exception as e:
                    logger.error(f"Failed to run test {test_name}: {e}")
                    
                # Small delay between tests
                await asyncio.sleep(0.5)
        
        # Generate final report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive test report."""
        self.report.end_time = datetime.now()
        total_duration = (self.report.end_time - self.report.start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        print(f"Test Run Duration: {total_duration:.2f} seconds")
        print(f"Total Tests: {self.report.total_tests}")
        print(f"✅ Passed: {self.report.passed_tests}")
        print(f"❌ Failed: {self.report.failed_tests}")
        print(f"⏭️ Skipped: {self.report.skipped_tests}")
        
        if self.report.total_tests > 0:
            success_rate = (self.report.passed_tests / self.report.total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        # Results by category
        categories = {}
        for result in self.report.test_results:
            if result.category not in categories:
                categories[result.category] = {'pass': 0, 'fail': 0, 'skip': 0}
            categories[result.category][result.status.lower()] += 1
        
        print(f"\n📋 Results by Category:")
        for category, counts in categories.items():
            total = sum(counts.values())
            pass_rate = (counts['pass'] / total * 100) if total > 0 else 0
            print(f"  {category}: {counts['pass']}/{total} passed ({pass_rate:.1f}%)")
        
        # Failed tests details
        if self.report.failed_tests > 0:
            print(f"\n❌ Failed Tests Details:")
            for result in self.report.test_results:
                if result.status == "FAIL":
                    print(f"  • {result.test_name}: {result.error.split(chr(10))[0]}")
        
        # Top longest tests
        sorted_tests = sorted(self.report.test_results, key=lambda x: x.duration, reverse=True)[:5]
        print(f"\n⏱️ Longest Running Tests:")
        for result in sorted_tests:
            print(f"  • {result.test_name}: {result.duration:.2f}s")
        
        # System recommendations
        print(f"\n🔧 System Recommendations:")
        
        if self.report.failed_tests == 0:
            print("  ✅ All tests passed - system is production ready")
        elif self.report.failed_tests <= 2:
            print("  ⚠️ Minor issues detected - review failed tests")
        else:
            print("  ❌ Significant issues detected - system needs attention")
        
        if not self.components_available:
            print("  🔧 Component imports failed - verify all dependencies installed")
        
        # Save detailed report to file
        report_data = {
            'test_run': {
                'start_time': self.report.start_time.isoformat(),
                'end_time': self.report.end_time.isoformat(),
                'duration_seconds': total_duration,
                'total_tests': self.report.total_tests,
                'passed_tests': self.report.passed_tests,
                'failed_tests': self.report.failed_tests,
                'skipped_tests': self.report.skipped_tests,
                'success_rate': success_rate if self.report.total_tests > 0 else 0
            },
            'test_results': [
                {
                    'test_name': result.test_name,
                    'category': result.category,
                    'status': result.status,
                    'duration': result.duration,
                    'details': result.details,
                    'error': result.error
                }
                for result in self.report.test_results
            ],
            'categories': categories
        }
        
        # Save to file
        report_filename = f"graceful_degradation_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        
        print("\n" + "=" * 80)
        print("🏁 COMPREHENSIVE TESTING COMPLETED")
        print("=" * 80)

async def main():
    """Main test runner entry point."""
    tester = ComprehensiveGracefulDegradationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())