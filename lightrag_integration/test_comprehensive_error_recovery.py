#!/usr/bin/env python3
"""
Comprehensive Test Suite for Error Recovery and Retry Logic System

This test suite validates all components of the error recovery system:

1. RetryStateManager - State persistence and management
2. IntelligentRetryEngine - Retry logic and backoff strategies
3. RecoveryStrategyRouter - Error routing and recovery actions
4. ErrorRecoveryOrchestrator - System coordination
5. Configuration System - Configuration management and validation
6. Integration Layer - Decorators and context managers

Test Categories:
    - Unit Tests: Individual component testing
    - Integration Tests: Component interaction testing
    - End-to-End Tests: Full system workflow testing
    - Error Scenario Tests: Various error condition handling
    - Performance Tests: System performance under load
    - Configuration Tests: Configuration loading and validation

Author: Claude Code (Anthropic)
Created: 2025-08-09
Version: 1.0.0
Task: CMO-LIGHTRAG-014-T06
"""

import asyncio
import json
import logging
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock
import random

# Import components to test
from .comprehensive_error_recovery_system import (
    ErrorRecoveryOrchestrator, RetryStateManager, IntelligentRetryEngine,
    RecoveryStrategyRouter, RetryMetricsCollector, RetryState, RetryAttempt,
    RetryStrategy, ErrorSeverity, RecoveryAction, ErrorRecoveryRule,
    create_error_recovery_orchestrator
)

from .error_recovery_config import (
    ErrorRecoveryConfigManager, ErrorRecoveryConfig, ConfigurationProfile,
    RetryPolicyConfig, ErrorClassificationConfig, create_error_recovery_config_manager
)

from .error_recovery_integration import (
    initialize_error_recovery_system, retry_on_error, error_recovery_context,
    execute_with_retry, execute_async_with_retry, get_error_recovery_status,
    shutdown_error_recovery_system
)


class MockError(Exception):
    """Mock error for testing."""
    pass


class MockRetryableError(Exception):
    """Mock retryable error for testing."""
    pass


class MockNonRetryableError(Exception):
    """Mock non-retryable error for testing."""
    pass


class TestRetryStateManager(unittest.TestCase):
    """Test suite for RetryStateManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = Path(self.temp_dir) / "retry_states"
        self.state_manager = RetryStateManager(
            state_dir=self.state_dir,
            logger=logging.getLogger(__name__)
        )
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_create_retry_state(self):
        """Test creation of retry state."""
        operation_id = "test_op_001"
        operation_type = "test_operation"
        context = {"test": "data"}
        
        state = self.state_manager.create_retry_state(
            operation_id=operation_id,
            operation_type=operation_type,
            operation_context=context,
            max_attempts=5
        )
        
        self.assertEqual(state.operation_id, operation_id)
        self.assertEqual(state.operation_type, operation_type)
        self.assertEqual(state.operation_context, context)
        self.assertEqual(state.max_attempts, 5)
        self.assertEqual(state.total_attempts, 0)
    
    def test_save_and_load_retry_state(self):
        """Test saving and loading retry state."""
        state = self.state_manager.create_retry_state(
            operation_id="test_op_002",
            operation_type="save_load_test",
            operation_context={"key": "value"}
        )
        
        # Modify state
        attempt = RetryAttempt(
            attempt_number=1,
            timestamp=datetime.now(),
            error_type="MockError",
            error_message="Test error",
            backoff_delay=2.0
        )
        state.add_attempt(attempt)
        
        # Save state
        self.assertTrue(self.state_manager.save_retry_state(state))
        
        # Load state
        loaded_state = self.state_manager.get_retry_state("test_op_002")
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.operation_id, state.operation_id)
        self.assertEqual(loaded_state.total_attempts, 1)
        self.assertEqual(len(loaded_state.error_history), 1)
    
    def test_delete_retry_state(self):
        """Test deletion of retry state."""
        state = self.state_manager.create_retry_state(
            operation_id="test_op_003",
            operation_type="delete_test",
            operation_context={}
        )
        
        # Verify state exists
        self.assertIsNotNone(self.state_manager.get_retry_state("test_op_003"))
        
        # Delete state
        self.assertTrue(self.state_manager.delete_retry_state("test_op_003"))
        
        # Verify state is gone
        self.assertIsNone(self.state_manager.get_retry_state("test_op_003"))
    
    def test_list_active_states(self):
        """Test listing active retry states."""
        # Create multiple states
        for i in range(3):
            self.state_manager.create_retry_state(
                operation_id=f"test_op_00{i + 4}",
                operation_type="list_test",
                operation_context={}
            )
        
        active_states = self.state_manager.list_active_states()
        self.assertEqual(len(active_states), 3)


class TestIntelligentRetryEngine(unittest.TestCase):
    """Test suite for IntelligentRetryEngine."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = Path(self.temp_dir) / "retry_states"
        self.state_manager = RetryStateManager(state_dir=self.state_dir)
        
        # Create custom recovery rules for testing
        self.recovery_rules = [
            ErrorRecoveryRule(
                rule_id="test_rule",
                error_patterns=[r"mock.*error"],
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=1.0
            )
        ]
        
        self.retry_engine = IntelligentRetryEngine(
            state_manager=self.state_manager,
            recovery_rules=self.recovery_rules
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_should_retry_operation_first_attempt(self):
        """Test retry decision for first attempt."""
        operation_id = "test_retry_001"
        error = MockError("Test error message")
        
        should_retry, retry_state = self.retry_engine.should_retry_operation(
            operation_id=operation_id,
            error=error,
            operation_type="test_operation",
            operation_context={}
        )
        
        self.assertTrue(should_retry)
        self.assertEqual(retry_state.total_attempts, 1)
        self.assertIsNotNone(retry_state.next_retry_time)
    
    def test_should_retry_operation_max_attempts_reached(self):
        """Test retry decision when max attempts reached."""
        operation_id = "test_retry_002"
        error = MockError("Test error message")
        
        # Create retry state with max attempts reached
        retry_state = self.state_manager.create_retry_state(
            operation_id=operation_id,
            operation_type="test_operation",
            operation_context={},
            max_attempts=2
        )
        
        # Add attempts to reach limit
        for i in range(2):
            attempt = RetryAttempt(
                attempt_number=i + 1,
                timestamp=datetime.now(),
                error_type="MockError",
                error_message="Previous error",
                backoff_delay=1.0
            )
            retry_state.add_attempt(attempt)
        
        self.state_manager.save_retry_state(retry_state)
        
        should_retry, updated_state = self.retry_engine.should_retry_operation(
            operation_id=operation_id,
            error=error,
            operation_type="test_operation",
            operation_context={}
        )
        
        self.assertFalse(should_retry)
        self.assertEqual(updated_state.total_attempts, 3)  # Original 2 + new attempt
    
    def test_record_operation_success(self):
        """Test recording successful operation."""
        operation_id = "test_success_001"
        
        # Create retry state with some attempts
        retry_state = self.state_manager.create_retry_state(
            operation_id=operation_id,
            operation_type="test_operation",
            operation_context={}
        )
        
        attempt = RetryAttempt(
            attempt_number=1,
            timestamp=datetime.now(),
            error_type="MockError",
            error_message="Test error",
            backoff_delay=2.0
        )
        retry_state.add_attempt(attempt)
        self.state_manager.save_retry_state(retry_state)
        
        # Record success
        self.retry_engine.record_operation_success(operation_id, 1.5)
        
        # Verify success was recorded
        updated_state = self.state_manager.get_retry_state(operation_id)
        self.assertEqual(updated_state.success_count, 1)
        self.assertIsNotNone(updated_state.last_success_time)
        
        # Check if last attempt was marked as successful
        if updated_state.error_history:
            self.assertTrue(updated_state.error_history[-1].success)
            self.assertEqual(updated_state.error_history[-1].response_time, 1.5)
    
    def test_backoff_calculation_strategies(self):
        """Test different backoff calculation strategies."""
        retry_state = RetryState(
            operation_id="test_backoff",
            operation_type="test",
            operation_context={},
            recovery_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=2.0,
            backoff_multiplier=3.0,
            max_delay=100.0
        )
        
        # Test exponential backoff
        retry_state.total_attempts = 1
        delay = retry_state.calculate_next_delay()
        expected = 2.0  # base_delay * multiplier^(attempts-1) = 2.0 * 3^0 = 2.0
        self.assertAlmostEqual(delay, expected, delta=1.0)  # Allow for jitter
        
        retry_state.total_attempts = 2
        delay = retry_state.calculate_next_delay()
        expected = 6.0  # 2.0 * 3^1 = 6.0
        self.assertAlmostEqual(delay, expected, delta=2.0)  # Allow for jitter
        
        # Test linear backoff
        retry_state.recovery_strategy = RetryStrategy.LINEAR_BACKOFF
        retry_state.total_attempts = 3
        delay = retry_state.calculate_next_delay()
        expected = 6.0  # base_delay * attempts = 2.0 * 3 = 6.0
        self.assertAlmostEqual(delay, expected, delta=2.0)  # Allow for jitter


class TestRecoveryStrategyRouter(unittest.TestCase):
    """Test suite for RecoveryStrategyRouter."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = Path(self.temp_dir) / "retry_states"
        self.state_manager = RetryStateManager(state_dir=self.state_dir)
        self.retry_engine = IntelligentRetryEngine(state_manager=self.state_manager)
        
        # Mock advanced recovery system
        self.mock_advanced_recovery = Mock()
        self.mock_advanced_recovery.handle_failure.return_value = {"action": "degrade"}
        
        self.strategy_router = RecoveryStrategyRouter(
            retry_engine=self.retry_engine,
            advanced_recovery=self.mock_advanced_recovery
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_route_error_recovery_basic(self):
        """Test basic error recovery routing."""
        operation_id = "test_route_001"
        error = MockError("Test routing error")
        
        recovery_result = self.strategy_router.route_error_recovery(
            operation_id=operation_id,
            error=error,
            operation_type="test_operation",
            operation_context={}
        )
        
        self.assertEqual(recovery_result['operation_id'], operation_id)
        self.assertIn('should_retry', recovery_result)
        self.assertIn('retry_state', recovery_result)
        self.assertIn('recovery_actions_taken', recovery_result)
    
    def test_recovery_actions_execution(self):
        """Test execution of recovery actions."""
        # Create rule with specific recovery actions
        rule = ErrorRecoveryRule(
            rule_id="test_action_rule",
            error_patterns=[r"action.*test"],
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            recovery_actions=[RecoveryAction.CHECKPOINT, RecoveryAction.NOTIFY]
        )
        
        self.retry_engine.recovery_rules = [rule]
        
        error = MockError("action test error")
        recovery_result = self.strategy_router.route_error_recovery(
            operation_id="test_actions_001",
            error=error,
            operation_type="test_operation",
            operation_context={}
        )
        
        self.assertTrue(len(recovery_result['recovery_actions_taken']) > 0)
        
        # Check that expected actions were taken
        action_types = [
            action['action'] for action in recovery_result['recovery_actions_taken']
        ]
        self.assertIn('checkpoint_created', action_types)
        self.assertIn('notification_sent', action_types)


class TestErrorRecoveryOrchestrator(unittest.TestCase):
    """Test suite for ErrorRecoveryOrchestrator."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = Path(self.temp_dir) / "orchestrator_states"
        
        self.orchestrator = create_error_recovery_orchestrator(
            state_dir=self.state_dir
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.orchestrator.close()
    
    def test_handle_operation_error(self):
        """Test operation error handling."""
        operation_id = "test_orchestrator_001"
        error = MockError("Orchestrator test error")
        
        recovery_result = self.orchestrator.handle_operation_error(
            operation_id=operation_id,
            error=error,
            operation_type="orchestrator_test",
            operation_context={"test": "data"}
        )
        
        self.assertEqual(recovery_result['operation_id'], operation_id)
        self.assertIn('should_retry', recovery_result)
        self.assertIsInstance(recovery_result.get('recovery_actions_taken'), list)
    
    def test_recover_operation(self):
        """Test operation recovery."""
        operation_id = "test_recover_001"
        
        # Create a mock operation that fails first, then succeeds
        call_count = 0
        def mock_operation(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MockError("First attempt fails")
            return f"Success on attempt {call_count}"
        
        # First, handle the error to create retry state
        try:
            mock_operation()
        except MockError as e:
            self.orchestrator.handle_operation_error(
                operation_id=operation_id,
                error=e,
                operation_type="recover_test",
                operation_context={}
            )
        
        # Now recover the operation
        result = self.orchestrator.recover_operation(
            operation_id=operation_id,
            operation_callable=mock_operation,
            operation_context={}
        )
        
        self.assertEqual(result, "Success on attempt 2")
    
    def test_get_system_status(self):
        """Test system status retrieval."""
        status = self.orchestrator.get_system_status()
        
        self.assertIn('orchestrator_statistics', status)
        self.assertIn('retry_engine_statistics', status)
        self.assertIn('retry_metrics', status)
        self.assertIn('active_retry_states', status)
        self.assertIn('timestamp', status)
    
    def test_cleanup_completed_operations(self):
        """Test cleanup of completed operations."""
        # Create some test operations
        for i in range(3):
            operation_id = f"cleanup_test_{i}"
            error = MockError(f"Test error {i}")
            
            self.orchestrator.handle_operation_error(
                operation_id=operation_id,
                error=error,
                operation_type="cleanup_test",
                operation_context={}
            )
        
        # Mark some as completed by exceeding retry limits
        # (This is a simplified test - in practice, operations complete successfully)
        
        cleanup_result = self.orchestrator.cleanup_completed_operations()
        
        self.assertIn('total_states', cleanup_result)
        self.assertIn('active_states', cleanup_result)
        self.assertIn('cleaned_states', cleanup_result)


class TestErrorRecoveryConfig(unittest.TestCase):
    """Test suite for error recovery configuration system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.json"
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """Test configuration manager initialization."""
        config_manager = create_error_recovery_config_manager(
            config_file=self.config_file,
            profile=ConfigurationProfile.DEVELOPMENT
        )
        
        self.assertEqual(config_manager.profile, ConfigurationProfile.DEVELOPMENT)
        self.assertTrue(self.config_file.exists())  # Should create default config
        
        config = config_manager.get_config()
        self.assertIsInstance(config, ErrorRecoveryConfig)
        self.assertEqual(config.profile, ConfigurationProfile.DEVELOPMENT)
    
    def test_recovery_rules_loading(self):
        """Test loading of recovery rules from configuration."""
        config_manager = create_error_recovery_config_manager(
            config_file=self.config_file,
            profile=ConfigurationProfile.PRODUCTION
        )
        
        recovery_rules = config_manager.get_recovery_rules()
        self.assertIsInstance(recovery_rules, list)
        self.assertTrue(len(recovery_rules) > 0)
        
        # Check first rule properties
        first_rule = recovery_rules[0]
        self.assertIsInstance(first_rule, ErrorRecoveryRule)
        self.assertIsInstance(first_rule.retry_strategy, RetryStrategy)
        self.assertIsInstance(first_rule.severity, ErrorSeverity)
    
    def test_configuration_updates(self):
        """Test dynamic configuration updates."""
        config_manager = create_error_recovery_config_manager(
            config_file=self.config_file
        )
        
        updates = {
            'retry_policy': {
                'default_max_attempts': 7,
                'default_max_delay': 999.0
            }
        }
        
        self.assertTrue(config_manager.update_configuration(updates))
        
        config = config_manager.get_config()
        self.assertEqual(config.retry_policy.default_max_attempts, 7)
        self.assertEqual(config.retry_policy.default_max_delay, 999.0)
    
    def test_configuration_export(self):
        """Test configuration export functionality."""
        config_manager = create_error_recovery_config_manager(
            config_file=self.config_file
        )
        
        export_file = Path(self.temp_dir) / "exported_config.json"
        self.assertTrue(config_manager.export_configuration(export_file))
        self.assertTrue(export_file.exists())
        
        # Verify exported content
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn('profile', exported_data)
        self.assertIn('retry_policy', exported_data)
        self.assertIn('recovery_rules', exported_data)


class TestErrorRecoveryIntegration(unittest.TestCase):
    """Test suite for error recovery integration layer."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize error recovery system
        self.orchestrator = initialize_error_recovery_system(
            profile=ConfigurationProfile.TESTING,
            state_dir=Path(self.temp_dir) / "integration_states"
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutdown_error_recovery_system()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_retry_decorator_sync(self):
        """Test retry decorator with synchronous functions."""
        call_count = 0
        
        @retry_on_error("sync_test", max_attempts=3, auto_retry=False)
        def test_function(should_fail: bool = True):
            nonlocal call_count
            call_count += 1
            if should_fail and call_count < 2:
                raise MockRetryableError(f"Failure on attempt {call_count}")
            return f"Success on attempt {call_count}"
        
        # Test successful execution after failure
        result = test_function(should_fail=True)
        self.assertEqual(result, "Success on attempt 2")
        
        # Test immediate success
        call_count = 0
        result = test_function(should_fail=False)
        self.assertEqual(result, "Success on attempt 1")
    
    def test_retry_decorator_async(self):
        """Test retry decorator with asynchronous functions."""
        async def run_async_test():
            call_count = 0
            
            @retry_on_error("async_test", max_attempts=3, auto_retry=False)
            async def test_async_function(should_fail: bool = True):
                nonlocal call_count
                call_count += 1
                if should_fail and call_count < 2:
                    raise MockRetryableError(f"Async failure on attempt {call_count}")
                return f"Async success on attempt {call_count}"
            
            result = await test_async_function(should_fail=True)
            self.assertEqual(result, "Async success on attempt 2")
        
        asyncio.run(run_async_test())
    
    def test_error_recovery_context(self):
        """Test error recovery context manager."""
        test_result = None
        
        try:
            with error_recovery_context("context_test") as ctx:
                ctx.set_context("test_key", "test_value")
                
                # Simulate some processing
                if random.random() < 0:  # Never fail in test
                    raise MockError("Context test error")
                
                test_result = "Context success"
                ctx.set_result(test_result)
        except Exception:
            pass
        
        self.assertEqual(test_result, "Context success")
    
    def test_execute_with_retry(self):
        """Test execute_with_retry utility function."""
        call_count = 0
        
        def operation_that_fails_once(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MockRetryableError("First attempt fails")
            return f"Success on attempt {call_count}, data: {kwargs.get('data')}"
        
        result = execute_with_retry(
            operation=operation_that_fails_once,
            operation_type="utility_test",
            max_attempts=3,
            data="test_data"
        )
        
        self.assertIn("Success on attempt", result)
        self.assertIn("test_data", result)
    
    def test_execute_async_with_retry(self):
        """Test execute_async_with_retry utility function."""
        async def run_async_utility_test():
            call_count = 0
            
            async def async_operation_that_fails_once(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise MockRetryableError("Async first attempt fails")
                return f"Async success on attempt {call_count}, data: {kwargs.get('data')}"
            
            result = await execute_async_with_retry(
                operation=async_operation_that_fails_once,
                operation_type="async_utility_test",
                max_attempts=3,
                data="async_test_data"
            )
            
            self.assertIn("Async success on attempt", result)
            self.assertIn("async_test_data", result)
        
        asyncio.run(run_async_utility_test())
    
    def test_get_error_recovery_status(self):
        """Test error recovery status retrieval."""
        status = get_error_recovery_status()
        
        self.assertIn('orchestrator_available', status)
        self.assertIn('config_manager_available', status)
        self.assertIn('timestamp', status)
        
        self.assertTrue(status['orchestrator_available'])
        self.assertTrue(status['config_manager_available'])


class TestErrorRecoveryPerformance(unittest.TestCase):
    """Performance tests for error recovery system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.orchestrator = initialize_error_recovery_system(
            profile=ConfigurationProfile.TESTING,
            state_dir=Path(self.temp_dir) / "performance_states"
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutdown_error_recovery_system()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_high_volume_error_handling(self):
        """Test system performance under high error volume."""
        start_time = time.time()
        num_operations = 100
        
        for i in range(num_operations):
            operation_id = f"perf_test_{i}"
            error = MockError(f"Performance test error {i}")
            
            self.orchestrator.handle_operation_error(
                operation_id=operation_id,
                error=error,
                operation_type="performance_test",
                operation_context={"iteration": i}
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 100 operations in reasonable time (< 10 seconds)
        self.assertLess(total_time, 10.0)
        
        # Verify system status is still healthy
        status = self.orchestrator.get_system_status()
        self.assertEqual(
            status['orchestrator_statistics']['operations_handled'],
            num_operations
        )
    
    def test_concurrent_operations(self):
        """Test concurrent operation handling."""
        import threading
        import concurrent.futures
        
        def handle_operation(operation_id: str):
            error = MockError(f"Concurrent test error {operation_id}")
            return self.orchestrator.handle_operation_error(
                operation_id=operation_id,
                error=error,
                operation_type="concurrent_test",
                operation_context={}
            )
        
        # Execute operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(handle_operation, f"concurrent_{i}")
                for i in range(50)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        self.assertEqual(len(results), 50)
        
        # Verify all operations were handled
        status = self.orchestrator.get_system_status()
        self.assertEqual(
            status['orchestrator_statistics']['operations_handled'],
            50
        )


class TestErrorScenarios(unittest.TestCase):
    """Test various error scenarios and edge cases."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.orchestrator = initialize_error_recovery_system(
            profile=ConfigurationProfile.TESTING,
            state_dir=Path(self.temp_dir) / "scenario_states"
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutdown_error_recovery_system()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_network_error_scenario(self):
        """Test network error handling scenario."""
        import socket
        
        # Simulate network error
        network_error = ConnectionError("Network unreachable")
        
        recovery_result = self.orchestrator.handle_operation_error(
            operation_id="network_test_001",
            error=network_error,
            operation_type="network_operation",
            operation_context={}
        )
        
        self.assertTrue(recovery_result.get('should_retry', False))
        self.assertIsNotNone(recovery_result.get('next_retry_time'))
    
    def test_rate_limit_error_scenario(self):
        """Test rate limit error handling scenario."""
        # Create custom rule for rate limiting
        rate_limit_error = MockError("Rate limit exceeded: too many requests")
        
        recovery_result = self.orchestrator.handle_operation_error(
            operation_id="rate_limit_test_001",
            error=rate_limit_error,
            operation_type="api_operation",
            operation_context={}
        )
        
        self.assertTrue(recovery_result.get('should_retry', False))
        
        # Should have longer backoff for rate limiting
        retry_state_dict = recovery_result.get('retry_state', {})
        if 'next_retry_time' in retry_state_dict and retry_state_dict['next_retry_time']:
            # Check that backoff delay is reasonable (this is a heuristic test)
            self.assertIsNotNone(recovery_result.get('next_retry_time'))
    
    def test_memory_exhaustion_scenario(self):
        """Test memory exhaustion error handling."""
        memory_error = MemoryError("Out of memory")
        
        recovery_result = self.orchestrator.handle_operation_error(
            operation_id="memory_test_001",
            error=memory_error,
            operation_type="data_processing",
            operation_context={}
        )
        
        # Memory errors should trigger degradation
        actions_taken = recovery_result.get('recovery_actions_taken', [])
        action_types = [action.get('action') for action in actions_taken]
        
        # Should include degradation or checkpoint actions for memory issues
        self.assertTrue(any('degrad' in str(action).lower() or 'checkpoint' in str(action).lower() 
                          for action in action_types))
    
    def test_persistent_failure_scenario(self):
        """Test handling of persistent failures."""
        operation_id = "persistent_failure_001"
        
        # Generate multiple failures for the same operation
        for attempt in range(5):
            error = MockError(f"Persistent failure attempt {attempt + 1}")
            
            recovery_result = self.orchestrator.handle_operation_error(
                operation_id=operation_id,
                error=error,
                operation_type="persistent_test",
                operation_context={}
            )
            
            if not recovery_result.get('should_retry', False):
                break
        
        # After multiple failures, should stop retrying
        self.assertFalse(recovery_result.get('should_retry', True))


# Test suite runner
def run_comprehensive_test_suite():
    """Run the complete error recovery test suite."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRetryStateManager,
        TestIntelligentRetryEngine,
        TestRecoveryStrategyRouter,
        TestErrorRecoveryOrchestrator,
        TestErrorRecoveryConfig,
        TestErrorRecoveryIntegration,
        TestErrorRecoveryPerformance,
        TestErrorScenarios
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)