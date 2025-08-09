"""
Comprehensive Test Suite for Graceful Degradation System

This script validates the graceful degradation system implementation,
including load detection, timeout management, query simplification,
and integration with existing production components.

Test Coverage:
1. Load Detection System - Metric collection and threshold evaluation
2. Timeout Management - Dynamic timeout scaling across services  
3. Query Simplification - Parameter reduction under load
4. Feature Toggle Control - Progressive feature disabling
5. Integration Adapters - Production component coordination
6. End-to-End Scenarios - Complete degradation workflows

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import logging
import time
import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import datetime

# Import graceful degradation components
try:
    from lightrag_integration.graceful_degradation_system import (
        GracefulDegradationManager,
        LoadDetectionSystem,
        TimeoutManager,
        QuerySimplificationEngine,
        FeatureToggleController,
        SystemLoadLevel,
        SystemLoadMetrics,
        LoadThresholds,
        create_production_degradation_system,
        create_development_degradation_system
    )
    
    from lightrag_integration.production_degradation_integration import (
        ProductionDegradationIntegration,
        LoadBalancerDegradationAdapter,
        RAGDegradationAdapter,
        FallbackDegradationAdapter,
        create_integrated_production_system
    )
    DEGRADATION_AVAILABLE = True
except ImportError as e:
    print(f"Degradation system not available: {e}")
    DEGRADATION_AVAILABLE = False


class TestLoadDetectionSystem(unittest.TestCase):
    """Test the load detection and threshold evaluation system."""
    
    def setUp(self):
        if not DEGRADATION_AVAILABLE:
            self.skipTest("Degradation system not available")
        
        # Use relaxed thresholds for testing
        self.test_thresholds = LoadThresholds(
            cpu_normal=30.0,
            cpu_elevated=50.0,
            cpu_high=70.0,
            cpu_critical=85.0,
            cpu_emergency=95.0
        )
        
        self.load_detector = LoadDetectionSystem(
            thresholds=self.test_thresholds,
            monitoring_interval=1.0
        )
    
    def test_load_level_calculation(self):
        """Test load level calculation based on metrics."""
        # Test normal load
        normal_metrics = SystemLoadMetrics(
            timestamp=datetime.now(),
            cpu_utilization=25.0,
            memory_pressure=40.0,
            request_queue_depth=5,
            response_time_p95=800.0,
            response_time_p99=1500.0,
            error_rate=0.1,
            active_connections=10,
            disk_io_wait=0.0
        )
        
        level = self.load_detector._calculate_load_level(normal_metrics)
        self.assertEqual(level, SystemLoadLevel.NORMAL)
        
        # Test high load
        high_metrics = SystemLoadMetrics(
            timestamp=datetime.now(),
            cpu_utilization=75.0,
            memory_pressure=70.0,
            request_queue_depth=40,
            response_time_p95=3500.0,
            response_time_p99=5500.0,
            error_rate=1.5,
            active_connections=80,
            disk_io_wait=0.0
        )
        
        level = self.load_detector._calculate_load_level(high_metrics)
        self.assertGreaterEqual(level, SystemLoadLevel.HIGH)
    
    def test_load_score_calculation(self):
        """Test load score calculation."""
        test_metrics = SystemLoadMetrics(
            timestamp=datetime.now(),
            cpu_utilization=50.0,
            memory_pressure=60.0,
            request_queue_depth=25,
            response_time_p95=2000.0,
            response_time_p99=3000.0,
            error_rate=1.0,
            active_connections=50,
            disk_io_wait=0.0
        )
        
        score = self.load_detector._calculate_load_score(test_metrics)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_metrics_collection(self):
        """Test system metrics collection."""
        metrics = self.load_detector.get_system_metrics()
        
        self.assertIsInstance(metrics, SystemLoadMetrics)
        self.assertIsInstance(metrics.cpu_utilization, float)
        self.assertIsInstance(metrics.memory_pressure, float)
        self.assertIsInstance(metrics.load_level, SystemLoadLevel)
        self.assertIsInstance(metrics.load_score, float)
    
    def test_request_tracking(self):
        """Test request queue tracking."""
        initial_depth = self.load_detector._request_queue_depth
        
        # Record request starts
        self.load_detector.record_request_start()
        self.load_detector.record_request_start()
        self.assertEqual(self.load_detector._request_queue_depth, initial_depth + 2)
        
        # Record request completions
        self.load_detector.record_request_complete(1500.0, error=False)
        self.assertEqual(self.load_detector._request_queue_depth, initial_depth + 1)
        
        self.load_detector.record_request_complete(2500.0, error=True)
        self.assertEqual(self.load_detector._request_queue_depth, initial_depth)
        
        # Check error tracking
        self.assertEqual(self.load_detector._error_count, 1)


class TestTimeoutManager(unittest.TestCase):
    """Test dynamic timeout management."""
    
    def setUp(self):
        if not DEGRADATION_AVAILABLE:
            self.skipTest("Degradation system not available")
        
        self.timeout_manager = TimeoutManager()
    
    def test_base_timeouts(self):
        """Test base timeout values."""
        base_timeouts = self.timeout_manager.base_timeouts
        
        self.assertEqual(base_timeouts['lightrag_query'], 60.0)
        self.assertEqual(base_timeouts['literature_search'], 90.0)
        self.assertEqual(base_timeouts['openai_api'], 45.0)
        self.assertEqual(base_timeouts['perplexity_api'], 35.0)
    
    def test_timeout_scaling(self):
        """Test timeout scaling for different load levels."""
        # Normal load - no scaling
        self.timeout_manager.update_timeouts_for_load_level(SystemLoadLevel.NORMAL)
        normal_timeout = self.timeout_manager.get_timeout('lightrag_query')
        self.assertEqual(normal_timeout, 60.0)
        
        # High load - reduced timeouts
        self.timeout_manager.update_timeouts_for_load_level(SystemLoadLevel.HIGH)
        high_timeout = self.timeout_manager.get_timeout('lightrag_query')
        self.assertLess(high_timeout, normal_timeout)
        
        # Critical load - significantly reduced timeouts
        self.timeout_manager.update_timeouts_for_load_level(SystemLoadLevel.CRITICAL)
        critical_timeout = self.timeout_manager.get_timeout('lightrag_query')
        self.assertLess(critical_timeout, high_timeout)
        
        # Emergency load - minimal timeouts
        self.timeout_manager.update_timeouts_for_load_level(SystemLoadLevel.EMERGENCY)
        emergency_timeout = self.timeout_manager.get_timeout('lightrag_query')
        self.assertLess(emergency_timeout, critical_timeout)
    
    def test_get_all_timeouts(self):
        """Test getting all timeouts."""
        all_timeouts = self.timeout_manager.get_all_timeouts()
        
        self.assertIn('lightrag_query', all_timeouts)
        self.assertIn('literature_search', all_timeouts)
        self.assertIn('openai_api', all_timeouts)
        self.assertIn('perplexity_api', all_timeouts)


class TestQuerySimplificationEngine(unittest.TestCase):
    """Test query parameter simplification."""
    
    def setUp(self):
        if not DEGRADATION_AVAILABLE:
            self.skipTest("Degradation system not available")
        
        self.simplifier = QuerySimplificationEngine()
    
    def test_normal_load_no_simplification(self):
        """Test that normal load doesn't simplify queries."""
        original_params = {
            'max_total_tokens': 8000,
            'top_k': 10,
            'response_type': 'Multiple Paragraphs',
            'mode': 'hybrid'
        }
        
        self.simplifier.update_load_level(SystemLoadLevel.NORMAL)
        simplified = self.simplifier.simplify_query_params(original_params)
        
        self.assertEqual(simplified, original_params)
    
    def test_high_load_simplification(self):
        """Test query simplification under high load."""
        original_params = {
            'max_total_tokens': 8000,
            'top_k': 10,
            'response_type': 'Multiple Paragraphs',
            'mode': 'hybrid'
        }
        
        self.simplifier.update_load_level(SystemLoadLevel.HIGH)
        simplified = self.simplifier.simplify_query_params(original_params)
        
        # Should reduce tokens and top_k
        self.assertLess(simplified['max_total_tokens'], original_params['max_total_tokens'])
        self.assertLess(simplified['top_k'], original_params['top_k'])
        self.assertEqual(simplified['response_type'], 'Short Answer')
    
    def test_critical_load_aggressive_simplification(self):
        """Test aggressive simplification under critical load."""
        original_params = {
            'max_total_tokens': 8000,
            'top_k': 10,
            'response_type': 'Multiple Paragraphs',
            'mode': 'hybrid'
        }
        
        self.simplifier.update_load_level(SystemLoadLevel.CRITICAL)
        simplified = self.simplifier.simplify_query_params(original_params)
        
        # Should significantly reduce parameters
        self.assertLessEqual(simplified['max_total_tokens'], 2000)
        self.assertLessEqual(simplified['top_k'], 3)
        self.assertEqual(simplified['response_type'], 'Short Answer')
        self.assertEqual(simplified['mode'], 'local')  # Simplified mode
    
    def test_emergency_load_minimal_processing(self):
        """Test minimal processing under emergency load."""
        original_params = {
            'max_total_tokens': 8000,
            'top_k': 10,
            'response_type': 'Multiple Paragraphs',
            'mode': 'hybrid'
        }
        
        self.simplifier.update_load_level(SystemLoadLevel.EMERGENCY)
        simplified = self.simplifier.simplify_query_params(original_params)
        
        # Should use minimal parameters
        self.assertEqual(simplified['max_total_tokens'], 1000)
        self.assertEqual(simplified['top_k'], 1)
        self.assertEqual(simplified['response_type'], 'Short Answer')
        self.assertEqual(simplified['mode'], 'local')
    
    def test_feature_skipping(self):
        """Test feature skipping logic."""
        # Normal load - no features skipped
        self.simplifier.update_load_level(SystemLoadLevel.NORMAL)
        self.assertFalse(self.simplifier.should_skip_feature('confidence_analysis'))
        self.assertFalse(self.simplifier.should_skip_feature('detailed_logging'))
        
        # High load - some features skipped
        self.simplifier.update_load_level(SystemLoadLevel.HIGH)
        self.assertTrue(self.simplifier.should_skip_feature('detailed_logging'))
        self.assertTrue(self.simplifier.should_skip_feature('complex_analytics'))
        
        # Critical load - more features skipped
        self.simplifier.update_load_level(SystemLoadLevel.CRITICAL)
        self.assertTrue(self.simplifier.should_skip_feature('confidence_analysis'))
        self.assertTrue(self.simplifier.should_skip_feature('confidence_scoring'))


class TestFeatureToggleController(unittest.TestCase):
    """Test feature toggle control system."""
    
    def setUp(self):
        if not DEGRADATION_AVAILABLE:
            self.skipTest("Degradation system not available")
        
        self.controller = FeatureToggleController()
    
    def test_feature_state_tracking(self):
        """Test feature state tracking across load levels."""
        # Start with normal load
        self.controller.update_load_level(SystemLoadLevel.NORMAL)
        self.assertTrue(self.controller.is_feature_enabled('confidence_analysis'))
        self.assertTrue(self.controller.is_feature_enabled('detailed_logging'))
        
        # Move to high load
        self.controller.update_load_level(SystemLoadLevel.HIGH)
        self.assertTrue(self.controller.is_feature_enabled('confidence_analysis'))
        self.assertFalse(self.controller.is_feature_enabled('detailed_logging'))
        
        # Move to critical load
        self.controller.update_load_level(SystemLoadLevel.CRITICAL)
        self.assertFalse(self.controller.is_feature_enabled('confidence_analysis'))
        self.assertFalse(self.controller.is_feature_enabled('detailed_logging'))
    
    def test_resource_limits(self):
        """Test resource limit adjustments."""
        # Normal load limits
        self.controller.update_load_level(SystemLoadLevel.NORMAL)
        normal_limits = self.controller.get_resource_limits()
        self.assertEqual(normal_limits['max_concurrent_requests'], 100)
        
        # High load limits
        self.controller.update_load_level(SystemLoadLevel.HIGH)
        high_limits = self.controller.get_resource_limits()
        self.assertLess(high_limits['max_concurrent_requests'], normal_limits['max_concurrent_requests'])
        
        # Emergency load limits
        self.controller.update_load_level(SystemLoadLevel.EMERGENCY)
        emergency_limits = self.controller.get_resource_limits()
        self.assertEqual(emergency_limits['max_concurrent_requests'], 10)
    
    def test_feature_callbacks(self):
        """Test feature change callbacks."""
        callback_called = False
        callback_state = None
        
        def test_callback(enabled: bool):
            nonlocal callback_called, callback_state
            callback_called = True
            callback_state = enabled
        
        self.controller.add_feature_callback('confidence_analysis', test_callback)
        
        # Change load level to trigger callback
        self.controller.update_load_level(SystemLoadLevel.CRITICAL)
        
        self.assertTrue(callback_called)
        self.assertFalse(callback_state)  # Should be disabled at critical load


class TestGracefulDegradationManager(unittest.TestCase):
    """Test the main degradation manager."""
    
    def setUp(self):
        if not DEGRADATION_AVAILABLE:
            self.skipTest("Degradation system not available")
        
        # Use fast monitoring for testing
        test_thresholds = LoadThresholds(
            cpu_high=60.0,
            memory_high=60.0,
            queue_high=20
        )
        
        self.manager = GracefulDegradationManager(
            load_thresholds=test_thresholds,
            monitoring_interval=0.5
        )
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        self.assertIsNotNone(self.manager.load_detector)
        self.assertIsNotNone(self.manager.timeout_manager)
        self.assertIsNotNone(self.manager.query_simplifier)
        self.assertIsNotNone(self.manager.feature_controller)
    
    def test_status_reporting(self):
        """Test status reporting."""
        status = self.manager.get_current_status()
        
        self.assertIn('load_level', status)
        self.assertIn('degradation_active', status)
        self.assertIn('current_timeouts', status)
        self.assertIn('resource_limits', status)
        self.assertIn('metrics', status)
    
    def test_feature_integration(self):
        """Test feature integration methods."""
        # Test timeout retrieval
        timeout = self.manager.get_timeout_for_service('lightrag_query')
        self.assertIsInstance(timeout, float)
        self.assertGreater(timeout, 0)
        
        # Test query simplification
        params = {'max_total_tokens': 8000, 'top_k': 10}
        simplified = self.manager.simplify_query_params(params)
        self.assertIn('max_total_tokens', simplified)
        self.assertIn('top_k', simplified)
        
        # Test feature checking
        enabled = self.manager.is_feature_enabled('confidence_analysis')
        self.assertIsInstance(enabled, bool)
    
    def test_request_acceptance(self):
        """Test request acceptance logic."""
        # Normal load should accept requests
        self.manager.current_load_level = SystemLoadLevel.NORMAL
        self.assertTrue(self.manager.should_accept_request())
        
        # Emergency load should reject requests
        self.manager.current_load_level = SystemLoadLevel.EMERGENCY
        self.assertFalse(self.manager.should_accept_request())


class TestProductionIntegration(unittest.TestCase):
    """Test production system integration."""
    
    def setUp(self):
        if not DEGRADATION_AVAILABLE:
            self.skipTest("Degradation system not available")
        
        # Create mock production components
        self.mock_load_balancer = Mock()
        self.mock_load_balancer.backend_instances = {}
        
        self.mock_clinical_rag = Mock()
        self.mock_clinical_rag.config = {'max_tokens': 8000}
        
        self.mock_fallback = Mock()
        self.mock_fallback.config = {}
        
        self.integration = ProductionDegradationIntegration(
            production_load_balancer=self.mock_load_balancer,
            clinical_rag=self.mock_clinical_rag,
            fallback_orchestrator=self.mock_fallback,
            monitoring_interval=0.5
        )
    
    def test_integration_initialization(self):
        """Test integration system initialization."""
        self.assertIsNotNone(self.integration.degradation_manager)
        self.assertIn('load_balancer', self.integration.adapters)
        self.assertIn('rag', self.integration.adapters)
        self.assertIn('fallback', self.integration.adapters)
    
    def test_adapter_creation(self):
        """Test adapter creation for different components."""
        lb_adapter = self.integration.adapters.get('load_balancer')
        self.assertIsInstance(lb_adapter, LoadBalancerDegradationAdapter)
        
        rag_adapter = self.integration.adapters.get('rag')
        self.assertIsInstance(rag_adapter, RAGDegradationAdapter)
        
        fallback_adapter = self.integration.adapters.get('fallback')
        self.assertIsInstance(fallback_adapter, FallbackDegradationAdapter)
    
    def test_status_reporting(self):
        """Test integration status reporting."""
        status = self.integration.get_integration_status()
        
        self.assertIn('integration_active', status)
        self.assertIn('current_load_level', status)
        self.assertIn('adapters_active', status)
        self.assertIn('components_integrated', status)
    
    def test_forced_load_level(self):
        """Test forcing load levels for testing."""
        original_level = self.integration.current_load_level
        
        # Force high load level
        self.integration.force_load_level(SystemLoadLevel.HIGH)
        self.assertEqual(self.integration.current_load_level, SystemLoadLevel.HIGH)
        
        # Force back to normal
        self.integration.force_load_level(SystemLoadLevel.NORMAL)
        self.assertEqual(self.integration.current_load_level, SystemLoadLevel.NORMAL)


class TestEndToEndScenarios(unittest.TestCase):
    """Test complete end-to-end degradation scenarios."""
    
    def setUp(self):
        if not DEGRADATION_AVAILABLE:
            self.skipTest("Degradation system not available")
        
        # Create production degradation system
        self.degradation_system = create_production_degradation_system(
            monitoring_interval=0.5
        )
    
    def test_load_progression_scenario(self):
        """Test complete load progression from normal to emergency."""
        # Start with normal load
        self.assertEqual(self.degradation_system.current_load_level, SystemLoadLevel.NORMAL)
        
        # Simulate load progression
        load_levels = [
            SystemLoadLevel.ELEVATED,
            SystemLoadLevel.HIGH,
            SystemLoadLevel.CRITICAL,
            SystemLoadLevel.EMERGENCY
        ]
        
        for level in load_levels:
            # Simulate load change
            mock_metrics = SystemLoadMetrics(
                timestamp=datetime.now(),
                cpu_utilization=50.0 + (level * 15.0),
                memory_pressure=40.0 + (level * 15.0),
                request_queue_depth=5 + (level * 20),
                response_time_p95=1000.0 + (level * 1000.0),
                response_time_p99=2000.0 + (level * 1000.0),
                error_rate=0.1 + (level * 0.5),
                active_connections=10 + (level * 20),
                disk_io_wait=0.0,
                load_level=level,
                load_score=level / 4.0,
                degradation_recommended=level > SystemLoadLevel.NORMAL
            )
            
            # Trigger load change
            self.degradation_system._handle_load_change(mock_metrics)
            
            # Verify degradation applied
            status = self.degradation_system.get_current_status()
            self.assertEqual(status['load_level'], level.name)
            self.assertEqual(status['degradation_active'], level > SystemLoadLevel.NORMAL)
            
            # Check timeout reductions
            lightrag_timeout = self.degradation_system.get_timeout_for_service('lightrag_query')
            if level >= SystemLoadLevel.HIGH:
                self.assertLess(lightrag_timeout, 60.0)
            
            # Check feature disabling
            if level >= SystemLoadLevel.HIGH:
                self.assertFalse(self.degradation_system.is_feature_enabled('detailed_logging'))
            
            if level >= SystemLoadLevel.CRITICAL:
                self.assertFalse(self.degradation_system.is_feature_enabled('confidence_analysis'))


# ============================================================================
# PERFORMANCE AND STRESS TESTS
# ============================================================================

class TestPerformanceAndStress(unittest.TestCase):
    """Test performance and stress scenarios."""
    
    def setUp(self):
        if not DEGRADATION_AVAILABLE:
            self.skipTest("Degradation system not available")
        
        self.manager = create_development_degradation_system(monitoring_interval=0.1)
    
    def test_rapid_load_changes(self):
        """Test system behavior under rapid load changes."""
        start_time = time.time()
        
        # Simulate rapid load changes
        for i in range(50):
            level = SystemLoadLevel(i % 4)  # Cycle through levels
            mock_metrics = SystemLoadMetrics(
                timestamp=datetime.now(),
                cpu_utilization=20.0 + (level * 20.0),
                memory_pressure=30.0 + (level * 20.0),
                request_queue_depth=level * 10,
                response_time_p95=1000.0 + (level * 500.0),
                response_time_p99=2000.0 + (level * 500.0),
                error_rate=level * 0.5,
                active_connections=10 + (level * 10),
                disk_io_wait=0.0,
                load_level=level,
                load_score=level / 4.0,
                degradation_recommended=level > SystemLoadLevel.NORMAL
            )
            
            self.manager._handle_load_change(mock_metrics)
        
        elapsed = time.time() - start_time
        # Should handle 50 load changes quickly
        self.assertLess(elapsed, 1.0)
    
    def test_concurrent_access(self):
        """Test concurrent access to degradation system."""
        import threading
        
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    # Get status
                    status = self.manager.get_current_status()
                    results.append(status['load_level'])
                    
                    # Get timeouts
                    timeout = self.manager.get_timeout_for_service('lightrag_query')
                    results.append(timeout)
                    
                    # Check features
                    enabled = self.manager.is_feature_enabled('confidence_analysis')
                    results.append(enabled)
                    
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Start multiple worker threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have no errors and many results
        self.assertEqual(len(errors), 0)
        self.assertGreater(len(results), 100)


# ============================================================================
# TEST EXECUTION AND REPORTING
# ============================================================================

def run_degradation_tests():
    """Run all degradation system tests."""
    if not DEGRADATION_AVAILABLE:
        print("❌ Graceful degradation system not available for testing")
        return False
    
    print("🧪 Running Graceful Degradation System Tests")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestLoadDetectionSystem,
        TestTimeoutManager,
        TestQuerySimplificationEngine,
        TestFeatureToggleController,
        TestGracefulDegradationManager,
        TestProductionIntegration,
        TestEndToEndScenarios,
        TestPerformanceAndStress
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n⚠️  {len(result.failures + result.errors)} tests failed")
    
    return success


async def run_integration_demo():
    """Run integration demonstration."""
    if not DEGRADATION_AVAILABLE:
        print("❌ Graceful degradation system not available for demo")
        return
    
    print("\n🚀 Graceful Degradation Integration Demo")
    print("=" * 50)
    
    # Create integrated system
    integration = create_integrated_production_system(monitoring_interval=1.0)
    
    try:
        await integration.start()
        print("✅ Integration system started")
        
        # Simulate load scenarios
        scenarios = [
            (SystemLoadLevel.NORMAL, "Normal operations"),
            (SystemLoadLevel.HIGH, "High load detected"),
            (SystemLoadLevel.CRITICAL, "Critical load - aggressive degradation"),
            (SystemLoadLevel.EMERGENCY, "Emergency mode - minimal functionality"),
            (SystemLoadLevel.NORMAL, "Load normalized - restoration")
        ]
        
        for level, description in scenarios:
            print(f"\n📊 {description}")
            integration.force_load_level(level)
            
            status = integration.get_integration_status()
            degradation = status['degradation_status']
            
            print(f"   Load Level: {status['current_load_level']}")
            print(f"   LightRAG Timeout: {degradation['current_timeouts']['lightrag_query']:.1f}s")
            print(f"   Max Concurrent: {degradation['resource_limits']['max_concurrent_requests']}")
            print(f"   Degradation Active: {degradation['degradation_active']}")
            
            await asyncio.sleep(1)
        
        print("\n✅ Demo completed successfully")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
    finally:
        await integration.stop()
        print("🛑 Integration system stopped")


def main():
    """Main test execution function."""
    print("Clinical Metabolomics Oracle - Graceful Degradation System Test Suite")
    print("=" * 80)
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(levelname)s - %(name)s - %(message)s'
    )
    
    try:
        # Run unit tests
        test_success = run_degradation_tests()
        
        # Run integration demo
        print("\n" + "=" * 80)
        asyncio.run(run_integration_demo())
        
        # Final summary
        print("\n" + "=" * 80)
        if test_success:
            print("🎉 All tests completed successfully!")
            print("✅ Graceful degradation system is ready for production deployment")
        else:
            print("⚠️  Some tests failed - review results before deployment")
        
        return test_success
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)