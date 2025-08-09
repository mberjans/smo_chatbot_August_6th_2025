"""
Comprehensive test fixtures for circuit breaker functionality testing.

This module provides shared fixtures and configuration for testing the circuit breaker
system, including basic CircuitBreaker and CostBasedCircuitBreaker functionality.
"""

import time
import pytest
import asyncio
import threading
import logging
import statistics
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime, timezone

# Import circuit breaker classes
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag_integration.clinical_metabolomics_rag import CircuitBreaker, CircuitBreakerError
from lightrag_integration.cost_based_circuit_breaker import (
    CostBasedCircuitBreaker,
    CostThresholdRule,
    CostThresholdType,
    CircuitBreakerState,
    OperationCostEstimator,
    CostCircuitBreakerManager
)


# =============================================================================
# TIME CONTROL FIXTURES
# =============================================================================

@pytest.fixture
def mock_time():
    """Provide controlled time for testing time-dependent behavior."""
    with patch('time.time') as mock_time_func:
        current_time = [0.0]  # Use list to allow mutation in nested functions
        
        def time_side_effect():
            return current_time[0]
            
        def advance_time(seconds):
            current_time[0] += seconds
            
        mock_time_func.side_effect = time_side_effect
        mock_time_func.advance = advance_time
        mock_time_func.current = lambda: current_time[0]
        
        yield mock_time_func


@pytest.fixture
def time_controller():
    """Provide advanced time controller for complex scenarios."""
    class TimeController:
        def __init__(self):
            self.current_time = 0.0
            self.time_multiplier = 1.0
            
        def advance(self, seconds):
            self.current_time += seconds * self.time_multiplier
            
        def set_multiplier(self, multiplier):
            self.time_multiplier = multiplier
            
        def reset(self):
            self.current_time = 0.0
            self.time_multiplier = 1.0
            
        def time(self):
            return self.current_time
    
    controller = TimeController()
    with patch('time.time', side_effect=controller.time):
        yield controller


# =============================================================================
# BASIC CIRCUIT BREAKER FIXTURES
# =============================================================================

@pytest.fixture
def basic_circuit_breaker():
    """Provide a basic CircuitBreaker instance for testing."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=1.0,
        expected_exception=Exception
    )


@pytest.fixture
def custom_circuit_breaker():
    """Provide a CircuitBreaker with custom parameters."""
    def create_breaker(failure_threshold=5, recovery_timeout=60.0, expected_exception=Exception):
        return CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception
        )
    return create_breaker


# =============================================================================
# FUNCTION SIMULATION FIXTURES
# =============================================================================

@pytest.fixture
def failing_function_factory():
    """Factory for creating functions with controlled failure patterns."""
    def create_failing_function(fail_count=0, exception_type=Exception, 
                               success_value="success", failure_message="test failure"):
        call_count = 0
        
        def failing_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= fail_count:
                raise exception_type(failure_message)
            return success_value
        
        failing_func.call_count = lambda: call_count
        failing_func.reset = lambda: None  # Reset function for call count
        
        return failing_func
    
    return create_failing_function


@pytest.fixture
def async_failing_function_factory():
    """Factory for creating async functions with controlled failure patterns."""
    def create_async_failing_function(fail_count=0, exception_type=Exception,
                                     success_value="async_success", failure_message="async test failure"):
        call_count = 0
        
        async def async_failing_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= fail_count:
                raise exception_type(failure_message)
            await asyncio.sleep(0.01)  # Simulate async work
            return success_value
        
        async_failing_func.call_count = lambda: call_count
        return async_failing_func
    
    return create_async_failing_function


@pytest.fixture
def intermittent_failure_function():
    """Create function that fails intermittently based on pattern."""
    def create_intermittent_function(failure_pattern):
        """
        failure_pattern: list of boolean values indicating when to fail
        Example: [True, False, True, False] fails on calls 1 and 3
        """
        call_count = 0
        
        def intermittent_func(*args, **kwargs):
            nonlocal call_count
            should_fail = failure_pattern[call_count % len(failure_pattern)]
            call_count += 1
            
            if should_fail:
                raise Exception(f"Intermittent failure on call {call_count}")
            return f"success_call_{call_count}"
        
        intermittent_func.call_count = lambda: call_count
        return intermittent_func
    
    return create_intermittent_function


# =============================================================================
# COST-BASED CIRCUIT BREAKER FIXTURES
# =============================================================================

@pytest.fixture
def mock_budget_manager():
    """Provide a mock BudgetManager with configurable behavior."""
    manager = Mock()
    
    # Default budget status
    manager.get_budget_summary.return_value = {
        'daily_budget': {
            'budget': 10.0,
            'total_cost': 5.0,
            'percentage_used': 50.0,
            'over_budget': False
        },
        'monthly_budget': {
            'budget': 100.0,
            'total_cost': 30.0,
            'percentage_used': 30.0,
            'over_budget': False
        }
    }
    
    return manager


@pytest.fixture
def budget_scenario_factory(mock_budget_manager):
    """Factory for creating specific budget scenarios."""
    def create_budget_scenario(daily_used_pct=50.0, monthly_used_pct=30.0, 
                              daily_over=False, monthly_over=False):
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 10.0,
                'total_cost': daily_used_pct * 0.1,
                'percentage_used': daily_used_pct,
                'over_budget': daily_over
            },
            'monthly_budget': {
                'budget': 100.0,
                'total_cost': monthly_used_pct * 1.0,
                'percentage_used': monthly_used_pct,
                'over_budget': monthly_over
            }
        }
        return mock_budget_manager
    
    return create_budget_scenario


@pytest.fixture
def mock_cost_persistence():
    """Provide a mock CostPersistence for testing."""
    persistence = Mock()
    
    # Mock storage for cost records
    persistence._records = []
    
    def store_cost_side_effect(record):
        persistence._records.append(record)
        return True
    
    def get_costs_side_effect(start_time=None, end_time=None, category=None):
        filtered_records = persistence._records
        # Simple filtering logic for testing
        if category:
            filtered_records = [r for r in filtered_records if r.category == category]
        return filtered_records
    
    persistence.store_cost_record.side_effect = store_cost_side_effect
    persistence.get_cost_records.side_effect = get_costs_side_effect
    
    return persistence


@pytest.fixture
def mock_cost_estimator():
    """Provide a mock OperationCostEstimator with realistic behavior."""
    estimator = Mock()
    
    # Default estimation behavior
    def estimate_side_effect(operation_type, **kwargs):
        cost_defaults = {
            'llm_call': 0.02,
            'embedding_call': 0.001,
            'batch_operation': 0.05,
            'document_processing': 0.01
        }
        
        base_cost = cost_defaults.get(operation_type, 0.005)
        
        return {
            'estimated_cost': base_cost,
            'confidence': 0.8,
            'method': 'default_mock',
            'operation_type': operation_type
        }
    
    estimator.estimate_operation_cost.side_effect = estimate_side_effect
    estimator.update_historical_costs.return_value = None
    
    return estimator


@pytest.fixture
def cost_estimation_scenario_factory(mock_cost_estimator):
    """Factory for creating cost estimation scenarios."""
    def create_cost_scenario(scenarios):
        """
        scenarios: dict mapping operation_type to cost estimation dict
        Example: {'llm_call': {'estimated_cost': 0.05, 'confidence': 0.9}}
        """
        def estimate_side_effect(operation_type, **kwargs):
            if operation_type in scenarios:
                return scenarios[operation_type]
            return {
                'estimated_cost': 0.005,
                'confidence': 0.5,
                'method': 'fallback_mock'
            }
        
        mock_cost_estimator.estimate_operation_cost.side_effect = estimate_side_effect
        return mock_cost_estimator
    
    return create_cost_scenario


@pytest.fixture
def cost_threshold_rules():
    """Provide standard cost threshold rules for testing."""
    return [
        CostThresholdRule(
            rule_id="daily_budget_80",
            threshold_type=CostThresholdType.PERCENTAGE_DAILY,
            threshold_value=80.0,
            action="throttle",
            priority=10,
            throttle_factor=0.5,
            cooldown_minutes=5.0
        ),
        CostThresholdRule(
            rule_id="operation_cost_limit",
            threshold_type=CostThresholdType.OPERATION_COST,
            threshold_value=0.50,
            action="block",
            priority=20,
            cooldown_minutes=2.0
        ),
        CostThresholdRule(
            rule_id="monthly_budget_90",
            threshold_type=CostThresholdType.PERCENTAGE_MONTHLY,
            threshold_value=90.0,
            action="alert_only",
            priority=5,
            cooldown_minutes=10.0
        )
    ]


@pytest.fixture
def cost_based_circuit_breaker(mock_budget_manager, mock_cost_estimator, cost_threshold_rules):
    """Provide a CostBasedCircuitBreaker instance for testing."""
    return CostBasedCircuitBreaker(
        name="test_breaker",
        budget_manager=mock_budget_manager,
        cost_estimator=mock_cost_estimator,
        threshold_rules=cost_threshold_rules,
        failure_threshold=3,
        recovery_timeout=1.0
    )


@pytest.fixture
def circuit_breaker_manager(mock_budget_manager, mock_cost_persistence):
    """Provide a CostCircuitBreakerManager instance for testing."""
    return CostCircuitBreakerManager(
        budget_manager=mock_budget_manager,
        cost_persistence=mock_cost_persistence
    )


# =============================================================================
# LOAD AND PERFORMANCE TEST FIXTURES
# =============================================================================

@pytest.fixture
def load_generator():
    """Provide load generation capabilities for performance testing."""
    class LoadGenerator:
        def __init__(self):
            self.threads = []
            self.results = []
            self.stop_event = threading.Event()
        
        def generate_load(self, target_function, duration_seconds=10, 
                         threads=5, requests_per_second=10):
            """Generate load against target function."""
            def worker():
                start_time = time.time()
                request_interval = 1.0 / requests_per_second
                
                while (time.time() - start_time < duration_seconds and 
                       not self.stop_event.is_set()):
                    try:
                        start = time.time()
                        result = target_function()
                        latency = time.time() - start
                        self.results.append({
                            'success': True,
                            'latency': latency,
                            'timestamp': time.time(),
                            'result': result
                        })
                    except Exception as e:
                        self.results.append({
                            'success': False,
                            'error': str(e),
                            'timestamp': time.time()
                        })
                    
                    # Rate limiting
                    time.sleep(request_interval)
            
            # Start worker threads
            for _ in range(threads):
                thread = threading.Thread(target=worker)
                thread.start()
                self.threads.append(thread)
        
        def stop_load(self):
            """Stop load generation."""
            self.stop_event.set()
            for thread in self.threads:
                thread.join(timeout=5.0)
        
        def get_metrics(self):
            """Get performance metrics from load test."""
            if not self.results:
                return {}
            
            successful_requests = [r for r in self.results if r['success']]
            failed_requests = [r for r in self.results if not r['success']]
            
            if successful_requests:
                latencies = [r['latency'] for r in successful_requests]
                avg_latency = statistics.mean(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)
                p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
            else:
                avg_latency = max_latency = min_latency = p95_latency = 0
            
            return {
                'total_requests': len(self.results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / len(self.results) * 100 if self.results else 0,
                'average_latency': avg_latency,
                'max_latency': max_latency,
                'min_latency': min_latency,
                'p95_latency': p95_latency,
                'errors': [r['error'] for r in failed_requests]
            }
        
        def reset(self):
            """Reset load generator for reuse."""
            self.stop_event.clear()
            self.threads = []
            self.results = []
    
    return LoadGenerator()


# =============================================================================
# STATE VERIFICATION FIXTURES
# =============================================================================

@pytest.fixture
def circuit_breaker_state_verifier():
    """Provide state verification utilities for circuit breakers."""
    class StateVerifier:
        @staticmethod
        def assert_basic_circuit_state(cb, expected_state, expected_failure_count=None):
            """Verify basic circuit breaker state."""
            assert cb.state == expected_state, f"Expected state {expected_state}, got {cb.state}"
            if expected_failure_count is not None:
                assert cb.failure_count == expected_failure_count, \
                    f"Expected failure count {expected_failure_count}, got {cb.failure_count}"
        
        @staticmethod
        def assert_cost_circuit_state(cb, expected_state, expected_failure_count=None, expected_throttle_rate=None):
            """Verify cost-based circuit breaker state."""
            assert cb.state == expected_state, f"Expected state {expected_state}, got {cb.state}"
            if expected_failure_count is not None:
                assert cb.failure_count == expected_failure_count, \
                    f"Expected failure count {expected_failure_count}, got {cb.failure_count}"
            if expected_throttle_rate is not None:
                assert abs(cb._throttle_rate - expected_throttle_rate) < 0.01, \
                    f"Expected throttle rate {expected_throttle_rate}, got {cb._throttle_rate}"
        
        @staticmethod
        def wait_for_state_transition(cb, expected_state, timeout=5.0, check_interval=0.1):
            """Wait for circuit breaker to transition to expected state."""
            start_time = time.time()
            while time.time() - start_time < timeout:
                if cb.state == expected_state:
                    return True
                time.sleep(check_interval)
            return False
    
    return StateVerifier()


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

@pytest.fixture
def test_data_generator():
    """Provide test data generation utilities."""
    class TestDataGenerator:
        @staticmethod
        def generate_failure_sequence(length, failure_rate=0.3, seed=42):
            """Generate realistic failure sequences for testing."""
            import random
            random.seed(seed)
            
            sequence = []
            consecutive_failures = 0
            
            for i in range(length):
                # Increase failure probability after consecutive successes
                adjusted_rate = failure_rate * (1 + consecutive_failures * 0.1)
                
                if random.random() < adjusted_rate:
                    sequence.append('failure')
                    consecutive_failures += 1
                else:
                    sequence.append('success')
                    consecutive_failures = 0
            
            return sequence
        
        @staticmethod
        def generate_cost_time_series(duration_hours, base_cost_per_hour, volatility=0.2):
            """Generate realistic cost time series with trends and spikes."""
            import random
            import math
            
            timestamps = []
            costs = []
            
            # 5-minute intervals
            interval_seconds = 300
            total_intervals = int(duration_hours * 3600 / interval_seconds)
            
            for i in range(total_intervals):
                timestamp = i * interval_seconds
                
                # Base trend (daily pattern)
                trend = math.sin(2 * math.pi * timestamp / (24 * 3600)) * base_cost_per_hour * 0.3
                
                # Add volatility
                noise = random.uniform(-volatility, volatility) * base_cost_per_hour
                
                # Occasional spikes
                spike = 0
                if random.random() < 0.05:  # 5% chance of spike
                    spike = random.exponential(base_cost_per_hour)
                
                cost = max(0, base_cost_per_hour + trend + noise + spike)
                
                timestamps.append(timestamp)
                costs.append(cost)
            
            return list(zip(timestamps, costs))
        
        @staticmethod
        def generate_budget_crisis_scenario(crisis_type='gradual', duration_hours=24):
            """Generate budget crisis scenarios for testing."""
            scenarios = {
                'gradual': {
                    'budget_depletion_rate': 'linear',
                    'crisis_point_hour': duration_hours * 0.8,
                    'recovery_possible': True,
                    'depletion_pattern': lambda t: t / duration_hours
                },
                'sudden_spike': {
                    'budget_depletion_rate': 'exponential',
                    'crisis_point_hour': duration_hours * 0.3,
                    'spike_magnitude': 5.0,
                    'recovery_possible': False,
                    'depletion_pattern': lambda t: min(1.0, (t / (duration_hours * 0.3)) ** 2)
                },
                'oscillating': {
                    'budget_depletion_rate': 'sine_wave',
                    'crisis_cycles': 3,
                    'amplitude': 0.4,
                    'recovery_possible': True,
                    'depletion_pattern': lambda t: 0.5 + 0.4 * math.sin(2 * math.pi * t / (duration_hours / 3))
                }
            }
            
            return scenarios.get(crisis_type, scenarios['gradual'])
    
    return TestDataGenerator()


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically cleanup after each test."""
    yield  # Test runs here
    
    # Cleanup code - reset any global state if needed
    # This ensures test isolation
    pass


@pytest.fixture(scope="session")
def test_logger():
    """Provide a logger for testing."""
    logger = logging.getLogger("circuit_breaker_tests")
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# =============================================================================
# PARAMETRIZED FIXTURES
# =============================================================================

@pytest.fixture(params=[1, 3, 5, 10])
def failure_threshold(request):
    """Parametrized fixture for testing different failure thresholds."""
    return request.param


@pytest.fixture(params=[0.1, 1.0, 5.0, 30.0])
def recovery_timeout(request):
    """Parametrized fixture for testing different recovery timeouts."""
    return request.param


@pytest.fixture(params=[
    (CostThresholdType.PERCENTAGE_DAILY, 80.0, "throttle"),
    (CostThresholdType.PERCENTAGE_MONTHLY, 90.0, "block"),
    (CostThresholdType.OPERATION_COST, 1.0, "alert_only"),
    (CostThresholdType.RATE_BASED, 10.0, "throttle")
])
def cost_rule_params(request):
    """Parametrized fixture for cost rule testing."""
    threshold_type, threshold_value, action = request.param
    return CostThresholdRule(
        rule_id=f"test_rule_{threshold_type.value}_{threshold_value}",
        threshold_type=threshold_type,
        threshold_value=threshold_value,
        action=action,
        priority=10
    )


# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def integration_test_environment():
    """Provide a complete integration test environment."""
    class IntegrationEnvironment:
        def __init__(self):
            self.circuit_breakers = {}
            self.mock_apis = {}
            self.budget_manager = Mock()
            self.cost_persistence = Mock()
            self.monitoring_system = Mock()
        
        def add_circuit_breaker(self, name, breaker):
            self.circuit_breakers[name] = breaker
        
        def add_mock_api(self, name, api_mock):
            self.mock_apis[name] = api_mock
        
        def simulate_failure(self, api_name, error_type):
            if api_name in self.mock_apis:
                api_mock = self.mock_apis[api_name]
                if hasattr(api_mock, 'side_effect'):
                    api_mock.side_effect = error_type
        
        def reset_all(self):
            for breaker in self.circuit_breakers.values():
                if hasattr(breaker, 'force_close'):
                    breaker.force_close("Test reset")
    
    return IntegrationEnvironment()


# =============================================================================
# TEST UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def assert_helpers():
    """Provide assertion helper functions."""
    class AssertHelpers:
        @staticmethod
        def assert_within_tolerance(actual, expected, tolerance_pct=5.0):
            """Assert that actual value is within tolerance of expected."""
            tolerance = expected * (tolerance_pct / 100.0)
            assert abs(actual - expected) <= tolerance, \
                f"Expected {actual} to be within {tolerance_pct}% of {expected}"
        
        @staticmethod
        def assert_timing_precision(actual_time, expected_time, precision_ms=100):
            """Assert that timing is precise within specified milliseconds."""
            precision_seconds = precision_ms / 1000.0
            assert abs(actual_time - expected_time) <= precision_seconds, \
                f"Timing precision failed: {actual_time} vs {expected_time} (±{precision_ms}ms)"
        
        @staticmethod
        def assert_state_sequence(actual_states, expected_states):
            """Assert that state sequence matches expected pattern."""
            assert len(actual_states) == len(expected_states), \
                f"State sequence length mismatch: {len(actual_states)} vs {len(expected_states)}"
            for i, (actual, expected) in enumerate(zip(actual_states, expected_states)):
                assert actual == expected, \
                    f"State mismatch at position {i}: {actual} vs {expected}"
    
    return AssertHelpers()