"""
Unit tests for CostBasedCircuitBreaker functionality.

This module provides comprehensive unit tests for the cost-aware circuit breaker
system, including cost estimation, budget management, threshold rules, and 
advanced state management.
"""

import pytest
import time
import threading
import statistics
from unittest.mock import Mock, patch, MagicMock
from dataclasses import replace

from lightrag_integration.cost_based_circuit_breaker import (
    CostBasedCircuitBreaker,
    CostThresholdRule,
    CostThresholdType,
    CircuitBreakerState,
    OperationCostEstimator,
    CostCircuitBreakerManager
)
from lightrag_integration.clinical_metabolomics_rag import CircuitBreakerError


class TestCostThresholdRule:
    """Test CostThresholdRule validation and behavior."""
    
    def test_valid_rule_creation(self):
        """Test creation of valid cost threshold rules."""
        rule = CostThresholdRule(
            rule_id="test_rule",
            threshold_type=CostThresholdType.PERCENTAGE_DAILY,
            threshold_value=80.0,
            action="throttle",
            priority=10,
            throttle_factor=0.5
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.threshold_type == CostThresholdType.PERCENTAGE_DAILY
        assert rule.threshold_value == 80.0
        assert rule.action == "throttle"
        assert rule.priority == 10
        assert rule.throttle_factor == 0.5
    
    def test_invalid_threshold_value(self):
        """Test that invalid threshold values raise errors."""
        with pytest.raises(ValueError, match="Threshold value must be positive"):
            CostThresholdRule(
                rule_id="invalid_rule",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=-10.0,
                action="block"
            )
        
        with pytest.raises(ValueError, match="Threshold value must be positive"):
            CostThresholdRule(
                rule_id="invalid_rule",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=0.0,
                action="block"
            )
    
    def test_invalid_action(self):
        """Test that invalid actions raise errors."""
        with pytest.raises(ValueError, match="Action must be 'block', 'throttle', or 'alert_only'"):
            CostThresholdRule(
                rule_id="invalid_rule",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=80.0,
                action="invalid_action"
            )
    
    def test_invalid_throttle_factor(self):
        """Test that invalid throttle factors raise errors."""
        with pytest.raises(ValueError, match="Throttle factor must be between 0 and 1"):
            CostThresholdRule(
                rule_id="invalid_rule",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=80.0,
                action="throttle",
                throttle_factor=1.5
            )
        
        with pytest.raises(ValueError, match="Throttle factor must be between 0 and 1"):
            CostThresholdRule(
                rule_id="invalid_rule",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=80.0,
                action="throttle",
                throttle_factor=0.0
            )


class TestOperationCostEstimator:
    """Test OperationCostEstimator functionality."""
    
    def test_initialization(self, mock_cost_persistence, test_logger):
        """Test cost estimator initialization."""
        estimator = OperationCostEstimator(mock_cost_persistence, test_logger)
        
        assert estimator.cost_persistence == mock_cost_persistence
        assert estimator.logger == test_logger
        assert len(estimator._token_cost_rates) > 0
    
    def test_token_based_cost_estimation(self, mock_cost_persistence):
        """Test cost estimation using token counts and model rates."""
        estimator = OperationCostEstimator(mock_cost_persistence)
        
        # Test GPT-4o-mini estimation
        result = estimator.estimate_operation_cost(
            operation_type="llm_call",
            model_name="gpt-4o-mini",
            estimated_tokens={"input": 1000, "output": 500}
        )
        
        # Expected: (1000 * 0.000150/1000) + (500 * 0.000600/1000) = 0.00045
        assert result['method'] == 'token_based'
        assert result['confidence'] == 0.9
        assert abs(result['estimated_cost'] - 0.00045) < 0.0001
        assert result['model_used'] == "gpt-4o-mini"
    
    def test_token_based_cost_estimation_gpt4o(self, mock_cost_persistence):
        """Test cost estimation for GPT-4o model."""
        estimator = OperationCostEstimator(mock_cost_persistence)
        
        result = estimator.estimate_operation_cost(
            operation_type="llm_call",
            model_name="gpt-4o",
            estimated_tokens={"prompt": 500, "completion": 200}
        )
        
        # Expected: (500 * 0.005/1000) + (200 * 0.015/1000) = 0.0055
        assert result['method'] == 'token_based'
        assert result['confidence'] == 0.9
        assert abs(result['estimated_cost'] - 0.0055) < 0.0001
    
    def test_historical_average_estimation(self, mock_cost_persistence):
        """Test cost estimation using historical operation data."""
        estimator = OperationCostEstimator(mock_cost_persistence)
        
        # Add historical data
        historical_costs = [0.01, 0.015, 0.008, 0.012, 0.02]
        for cost in historical_costs:
            estimator.update_historical_costs("custom_operation", cost)
        
        result = estimator.estimate_operation_cost(operation_type="custom_operation")
        
        expected_avg = statistics.mean(historical_costs)
        expected_std = statistics.stdev(historical_costs)
        expected_cost = expected_avg + expected_std
        
        assert result['method'] == 'historical_average'
        assert abs(result['estimated_cost'] - expected_cost) < 0.001
        assert result['samples'] == len(historical_costs)
        assert result['confidence'] == len(historical_costs) / 100  # 0.05
    
    def test_default_cost_estimation_fallback(self, mock_cost_persistence):
        """Test default cost estimates when no data available."""
        estimator = OperationCostEstimator(mock_cost_persistence)
        
        result = estimator.estimate_operation_cost(operation_type="unknown_operation")
        
        assert result['method'] == 'default_estimate'
        assert result['confidence'] == 0.3
        assert result['estimated_cost'] == 0.005  # Default fallback
    
    def test_default_operation_types(self, mock_cost_persistence):
        """Test default estimates for known operation types."""
        estimator = OperationCostEstimator(mock_cost_persistence)
        
        test_cases = [
            ("llm_call", 0.01),
            ("embedding_call", 0.001),
            ("batch_operation", 0.05),
            ("document_processing", 0.02)
        ]
        
        for operation_type, expected_cost in test_cases:
            result = estimator.estimate_operation_cost(operation_type=operation_type)
            assert result['estimated_cost'] == expected_cost
            assert result['method'] == 'default_estimate'
    
    def test_update_historical_costs(self, mock_cost_persistence):
        """Test updating historical cost data for learning."""
        estimator = OperationCostEstimator(mock_cost_persistence)
        
        # Add costs and verify they're stored
        costs = [0.01, 0.02, 0.015]
        for cost in costs:
            estimator.update_historical_costs("test_op", cost)
        
        assert len(estimator._operation_costs["test_op"]) == 3
        assert estimator._operation_costs["test_op"] == costs
    
    def test_historical_costs_limit(self, mock_cost_persistence):
        """Test that historical costs are limited to prevent memory issues."""
        estimator = OperationCostEstimator(mock_cost_persistence)
        
        # Add more than 1000 costs
        for i in range(1100):
            estimator.update_historical_costs("test_op", 0.01)
        
        # Should be limited to 1000
        assert len(estimator._operation_costs["test_op"]) == 1000
    
    def test_estimation_error_handling(self, mock_cost_persistence):
        """Test error handling in cost estimation."""
        estimator = OperationCostEstimator(mock_cost_persistence)
        
        # Mock an error in the estimation process
        with patch.object(estimator, '_token_cost_rates', side_effect=Exception("Test error")):
            result = estimator.estimate_operation_cost(
                operation_type="llm_call",
                model_name="gpt-4o-mini",
                estimated_tokens={"input": 1000, "output": 500}
            )
        
        # Should return fallback estimation
        assert result['method'] == 'fallback'
        assert result['estimated_cost'] == 0.01
        assert result['confidence'] == 0.1
        assert 'error' in result


class TestCostBasedCircuitBreakerInitialization:
    """Test CostBasedCircuitBreaker initialization and configuration."""
    
    def test_basic_initialization(self, mock_budget_manager, mock_cost_estimator, cost_threshold_rules):
        """Test basic initialization of cost-based circuit breaker."""
        cb = CostBasedCircuitBreaker(
            name="test_cb",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=cost_threshold_rules,
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        assert cb.name == "test_cb"
        assert cb.budget_manager == mock_budget_manager
        assert cb.cost_estimator == mock_cost_estimator
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60.0
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert len(cb.threshold_rules) == len(cost_threshold_rules)
    
    def test_rule_priority_sorting(self, mock_budget_manager, mock_cost_estimator):
        """Test that rules are sorted by priority."""
        rules = [
            CostThresholdRule("rule_low", CostThresholdType.PERCENTAGE_DAILY, 80.0, priority=5),
            CostThresholdRule("rule_high", CostThresholdType.PERCENTAGE_DAILY, 90.0, priority=20),
            CostThresholdRule("rule_med", CostThresholdType.PERCENTAGE_DAILY, 85.0, priority=10),
        ]
        
        cb = CostBasedCircuitBreaker(
            name="test_cb",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=rules
        )
        
        # Should be sorted by priority (highest first)
        assert cb.threshold_rules[0].rule_id == "rule_high"  # priority 20
        assert cb.threshold_rules[1].rule_id == "rule_med"   # priority 10
        assert cb.threshold_rules[2].rule_id == "rule_low"   # priority 5
    
    def test_statistics_initialization(self, cost_based_circuit_breaker):
        """Test that operation statistics are properly initialized."""
        stats = cost_based_circuit_breaker._operation_stats
        
        expected_keys = [
            'total_calls', 'allowed_calls', 'blocked_calls', 'throttled_calls',
            'cost_blocked_calls', 'total_estimated_cost', 'total_actual_cost', 'cost_savings'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert stats[key] == 0 or stats[key] == 0.0


class TestCostBasedCircuitBreakerStateManagement:
    """Test advanced state management for cost-based circuit breaker."""
    
    def test_closed_state_behavior(self, cost_based_circuit_breaker):
        """Test behavior in closed state."""
        assert cost_based_circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Update state should maintain closed if no conditions met
        state = cost_based_circuit_breaker._update_state()
        assert state == CircuitBreakerState.CLOSED
    
    def test_budget_exceeded_opens_circuit(self, cost_based_circuit_breaker, budget_scenario_factory):
        """Test that budget exceeded condition opens circuit."""
        # Set budget to over budget
        budget_scenario_factory(daily_over=True)
        
        state = cost_based_circuit_breaker._update_state()
        assert state == CircuitBreakerState.OPEN
        assert cost_based_circuit_breaker.state == CircuitBreakerState.OPEN
    
    def test_budget_limited_state_transition(self, cost_based_circuit_breaker, budget_scenario_factory):
        """Test transition to budget limited state."""
        # Set budget to 96% used (above 95% threshold)
        budget_scenario_factory(daily_used_pct=96.0)
        
        state = cost_based_circuit_breaker._update_state()
        assert state == CircuitBreakerState.BUDGET_LIMITED
        assert cost_based_circuit_breaker.state == CircuitBreakerState.BUDGET_LIMITED
    
    def test_monthly_budget_priority(self, cost_based_circuit_breaker, budget_scenario_factory):
        """Test that monthly budget takes priority when higher."""
        # Daily at 80%, monthly at 97%
        budget_scenario_factory(daily_used_pct=80.0, monthly_used_pct=97.0)
        
        state = cost_based_circuit_breaker._update_state()
        assert state == CircuitBreakerState.BUDGET_LIMITED
    
    def test_traditional_failure_threshold_opens_circuit(self, cost_based_circuit_breaker, failing_function_factory):
        """Test that traditional failure threshold still works."""
        failing_func = failing_function_factory(fail_count=5, exception_type=Exception)
        
        # Cause failures to reach threshold
        for i in range(3):  # failure_threshold = 3
            try:
                cost_based_circuit_breaker.call(failing_func, operation_type="test")
            except:
                pass
        
        assert cost_based_circuit_breaker.state == CircuitBreakerState.OPEN
        assert cost_based_circuit_breaker.failure_count == 3
    
    def test_state_transition_from_open_to_half_open(self, cost_based_circuit_breaker, mock_time, failing_function_factory):
        """Test transition from open to half-open after recovery timeout."""
        failing_func = failing_function_factory(fail_count=5)
        
        # Open the circuit
        for i in range(3):
            try:
                cost_based_circuit_breaker.call(failing_func, operation_type="test")
            except:
                pass
        
        assert cost_based_circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Advance time past recovery timeout
        mock_time.advance(1.5)
        
        state = cost_based_circuit_breaker._update_state()
        assert state == CircuitBreakerState.HALF_OPEN
    
    def test_state_transition_from_half_open_to_closed(self, cost_based_circuit_breaker, mock_time, failing_function_factory):
        """Test transition from half-open to closed on success."""
        failing_func = failing_function_factory(fail_count=3)
        
        # Open circuit, then recover
        for i in range(3):
            try:
                cost_based_circuit_breaker.call(failing_func, operation_type="test")
            except:
                pass
        
        mock_time.advance(1.5)
        
        # Successful call should close circuit
        result = cost_based_circuit_breaker.call(failing_func, operation_type="test")
        assert result == "success"
        assert cost_based_circuit_breaker.state == CircuitBreakerState.CLOSED
        assert cost_based_circuit_breaker.failure_count == 0


class TestCostRuleEvaluation:
    """Test cost-based rule evaluation logic."""
    
    def test_percentage_daily_threshold_rule(self, cost_based_circuit_breaker, budget_scenario_factory):
        """Test percentage-based daily threshold rule evaluation."""
        # Set daily usage to 85% (above 80% threshold from fixture)
        budget_scenario_factory(daily_used_pct=85.0)
        
        # Mock the cost estimate
        cost_estimate = {'estimated_cost': 0.01, 'confidence': 0.8}
        
        result = cost_based_circuit_breaker._check_cost_rules(
            estimated_cost=0.01,
            operation_type="test_op",
            cost_estimate=cost_estimate
        )
        
        # Should trigger the daily_budget_80 rule (throttle action)
        assert not result['allowed']  # throttle still blocks in this implementation
        assert result['rule_triggered'] == 'daily_budget_80'
        assert result['action'] == 'throttle'
    
    def test_operation_cost_threshold_rule(self, cost_based_circuit_breaker):
        """Test per-operation cost threshold rule."""
        # High cost operation (above 0.50 threshold)
        cost_estimate = {'estimated_cost': 0.75, 'confidence': 0.9}
        
        result = cost_based_circuit_breaker._check_cost_rules(
            estimated_cost=0.75,
            operation_type="test_op",
            cost_estimate=cost_estimate
        )
        
        # Should trigger the operation_cost_limit rule (block action)
        assert not result['allowed']
        assert result['rule_triggered'] == 'operation_cost_limit'
        assert result['action'] == 'block'
    
    def test_monthly_percentage_alert_only_rule(self, cost_based_circuit_breaker, budget_scenario_factory):
        """Test monthly percentage rule with alert_only action."""
        # Set monthly usage to 92% (above 90% threshold)
        budget_scenario_factory(monthly_used_pct=92.0)
        
        cost_estimate = {'estimated_cost': 0.01, 'confidence': 0.8}
        
        result = cost_based_circuit_breaker._check_cost_rules(
            estimated_cost=0.01,
            operation_type="test_op",
            cost_estimate=cost_estimate
        )
        
        # Should trigger monthly_budget_90 rule (alert_only action)
        assert result['allowed']  # alert_only allows operation
        assert result['rule_triggered'] == 'monthly_budget_90'
        assert result['action'] == 'alert_only'
    
    def test_rule_priority_ordering(self, mock_budget_manager, mock_cost_estimator):
        """Test that rules are evaluated in priority order."""
        # Create rules with different priorities and thresholds
        rules = [
            CostThresholdRule("low_priority", CostThresholdType.OPERATION_COST, 0.30, 
                            action="block", priority=5),
            CostThresholdRule("high_priority", CostThresholdType.OPERATION_COST, 0.60, 
                            action="throttle", priority=20),
        ]
        
        cb = CostBasedCircuitBreaker(
            name="test_cb",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=rules
        )
        
        # Cost of 0.40 should trigger low_priority rule but not high_priority
        result = cb._check_cost_rules(
            estimated_cost=0.40,
            operation_type="test_op",
            cost_estimate={'estimated_cost': 0.40, 'confidence': 0.8}
        )
        
        # Higher priority rule should be checked first, but not triggered
        # Lower priority rule should trigger
        assert not result['allowed']
        assert result['rule_triggered'] == 'low_priority'
        assert result['action'] == 'block'
    
    def test_rule_cooldown_mechanism(self, cost_based_circuit_breaker, mock_time):
        """Test rule cooldown periods prevent repeated triggers."""
        cost_estimate = {'estimated_cost': 0.75, 'confidence': 0.9}
        
        # First trigger should work
        result1 = cost_based_circuit_breaker._check_cost_rules(
            estimated_cost=0.75,
            operation_type="test_op",
            cost_estimate=cost_estimate
        )
        assert not result1['allowed']
        assert result1['rule_triggered'] == 'operation_cost_limit'
        
        # Immediate second call should not trigger due to cooldown
        result2 = cost_based_circuit_breaker._check_cost_rules(
            estimated_cost=0.75,
            operation_type="test_op",
            cost_estimate=cost_estimate
        )
        assert result2['allowed']  # No rule triggered due to cooldown
        
        # After cooldown period (2.0 minutes), should trigger again
        mock_time.advance(121)  # 2.01 minutes
        result3 = cost_based_circuit_breaker._check_cost_rules(
            estimated_cost=0.75,
            operation_type="test_op",
            cost_estimate=cost_estimate
        )
        assert not result3['allowed']
        assert result3['rule_triggered'] == 'operation_cost_limit'
    
    def test_operation_type_filtering(self, mock_budget_manager, mock_cost_estimator):
        """Test that rules can be filtered by operation type."""
        rules = [
            CostThresholdRule(
                "specific_operation_rule", 
                CostThresholdType.OPERATION_COST, 
                0.01,  # Very low threshold
                action="block",
                applies_to_operations=["expensive_op"]
            )
        ]
        
        cb = CostBasedCircuitBreaker(
            name="test_cb",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=rules
        )
        
        # Rule should not apply to different operation type
        result1 = cb._check_cost_rules(
            estimated_cost=0.50,
            operation_type="regular_op",
            cost_estimate={'estimated_cost': 0.50, 'confidence': 0.8}
        )
        assert result1['allowed']
        
        # Rule should apply to specified operation type
        result2 = cb._check_cost_rules(
            estimated_cost=0.50,
            operation_type="expensive_op",
            cost_estimate={'estimated_cost': 0.50, 'confidence': 0.8}
        )
        assert not result2['allowed']
        assert result2['rule_triggered'] == 'specific_operation_rule'
    
    def test_absolute_cost_thresholds(self, mock_budget_manager, mock_cost_estimator, budget_scenario_factory):
        """Test absolute cost threshold evaluation."""
        # Set up budget scenario with known costs
        budget_scenario_factory(daily_used_pct=50.0)  # $5 used of $10 budget
        
        rules = [
            CostThresholdRule(
                "absolute_daily",
                CostThresholdType.ABSOLUTE_DAILY,
                7.0,  # $7 threshold
                action="block"
            )
        ]
        
        cb = CostBasedCircuitBreaker(
            name="test_cb",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=rules
        )
        
        # $3 operation would put us at $8, above $7 threshold
        result = cb._check_cost_rules(
            estimated_cost=3.0,
            operation_type="test_op",
            cost_estimate={'estimated_cost': 3.0, 'confidence': 0.8}
        )
        
        assert not result['allowed']
        assert result['rule_triggered'] == 'absolute_daily'
        assert result['current_value'] == 8.0  # $5 current + $3 estimated
    
    def test_rate_based_threshold(self, mock_budget_manager, mock_cost_estimator):
        """Test rate-based cost threshold evaluation."""
        rules = [
            CostThresholdRule(
                "rate_limit",
                CostThresholdType.RATE_BASED,
                5.0,  # $5 per hour
                action="throttle",
                time_window_minutes=60
            )
        ]
        
        cb = CostBasedCircuitBreaker(
            name="test_cb",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=rules
        )
        
        # Mock recent cost to be high
        with patch.object(cb, '_get_recent_cost', return_value=6.0):  # $6 in last hour
            result = cb._check_cost_rules(
                estimated_cost=0.01,
                operation_type="test_op",
                cost_estimate={'estimated_cost': 0.01, 'confidence': 0.8}
            )
        
        assert not result['allowed']
        assert result['rule_triggered'] == 'rate_limit'
        assert result['action'] == 'throttle'


class TestCostBasedCircuitBreakerOperations:
    """Test cost-based circuit breaker operation execution."""
    
    def test_successful_operation_execution(self, cost_based_circuit_breaker, failing_function_factory):
        """Test successful operation execution through cost-based circuit breaker."""
        success_func = failing_function_factory(fail_count=0)
        
        result = cost_based_circuit_breaker.call(
            success_func,
            operation_type="test_op",
            model_name="gpt-4o-mini",
            estimated_tokens={"input": 100, "output": 50}
        )
        
        assert result == "success"
        assert cost_based_circuit_breaker._operation_stats['total_calls'] == 1
        assert cost_based_circuit_breaker._operation_stats['allowed_calls'] == 1
        assert cost_based_circuit_breaker._operation_stats['blocked_calls'] == 0
        assert cost_based_circuit_breaker._operation_stats['total_estimated_cost'] > 0
    
    def test_operation_blocked_by_cost_rule(self, cost_based_circuit_breaker, failing_function_factory):
        """Test operation blocked by cost-based rule."""
        success_func = failing_function_factory(fail_count=0)
        
        # High cost operation that should trigger the operation_cost_limit rule
        with pytest.raises(CircuitBreakerError, match="blocked by cost-based circuit breaker"):
            cost_based_circuit_breaker.call(
                success_func,
                operation_type="expensive_op",
                estimated_tokens={"input": 10000, "output": 5000}  # Very high token count
            )
        
        assert cost_based_circuit_breaker._operation_stats['cost_blocked_calls'] == 1
        assert cost_based_circuit_breaker._operation_stats['cost_savings'] > 0
    
    def test_operation_blocked_by_circuit_open(self, cost_based_circuit_breaker, failing_function_factory, mock_time):
        """Test operation blocked when circuit is open."""
        failing_func = failing_function_factory(fail_count=5)
        
        # Open the circuit by causing failures
        for i in range(3):
            try:
                cost_based_circuit_breaker.call(failing_func, operation_type="test_op")
            except:
                pass
        
        assert cost_based_circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Next call should be blocked by open circuit
        success_func = failing_function_factory(fail_count=0)
        with pytest.raises(CircuitBreakerError, match="Circuit breaker.*is open"):
            cost_based_circuit_breaker.call(success_func, operation_type="test_op")
        
        assert cost_based_circuit_breaker._operation_stats['blocked_calls'] == 1
    
    def test_throttling_behavior(self, mock_budget_manager, mock_cost_estimator, budget_scenario_factory):
        """Test throttling behavior when rules trigger throttle action."""
        # Set up scenario that triggers throttle rule
        budget_scenario_factory(daily_used_pct=85.0)  # Above 80% threshold
        
        # Create circuit breaker with throttle rule
        rules = [
            CostThresholdRule(
                "throttle_rule",
                CostThresholdType.PERCENTAGE_DAILY,
                80.0,
                action="throttle",
                throttle_factor=0.5,
                priority=10
            )
        ]
        
        cb = CostBasedCircuitBreaker(
            name="test_cb",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=rules,
            failure_threshold=5
        )
        
        success_func = lambda: "success"
        
        # First call should trigger throttle rule but still fail
        with pytest.raises(CircuitBreakerError):
            cb.call(success_func, operation_type="test_op")
        
        # Throttle rate should be set
        assert cb._throttle_rate == 0.5
    
    def test_throttle_delay_calculation(self, cost_based_circuit_breaker):
        """Test throttle delay calculation."""
        # Set throttle rate to 0.7 (30% reduction)
        cost_based_circuit_breaker._throttle_rate = 0.7
        
        delay = cost_based_circuit_breaker._calculate_throttle_delay()
        
        # Base delay should be 1.0 - 0.7 = 0.3, with jitter between 0.8-1.2
        assert 0.24 <= delay <= 0.36  # 0.3 * 0.8 to 0.3 * 1.2
    
    def test_no_throttle_delay_when_rate_is_one(self, cost_based_circuit_breaker):
        """Test no delay when throttle rate is 1.0."""
        cost_based_circuit_breaker._throttle_rate = 1.0
        
        delay = cost_based_circuit_breaker._calculate_throttle_delay()
        assert delay == 0.0
    
    def test_operation_statistics_tracking(self, cost_based_circuit_breaker, failing_function_factory):
        """Test that operation statistics are properly tracked."""
        success_func = failing_function_factory(fail_count=0)
        failing_func = failing_function_factory(fail_count=5)
        
        # Execute successful operation
        result = cost_based_circuit_breaker.call(success_func, operation_type="test_op")
        assert result == "success"
        
        # Try high-cost operation (should be blocked)
        try:
            cost_based_circuit_breaker.call(
                success_func,
                operation_type="expensive_op",
                estimated_tokens={"input": 100000, "output": 50000}
            )
        except CircuitBreakerError:
            pass
        
        # Check statistics
        stats = cost_based_circuit_breaker._operation_stats
        assert stats['total_calls'] == 2
        assert stats['allowed_calls'] == 1
        assert stats['cost_blocked_calls'] == 1
        assert stats['total_estimated_cost'] > 0
        assert stats['cost_savings'] > 0
    
    def test_update_actual_cost(self, cost_based_circuit_breaker):
        """Test updating actual operation cost for learning."""
        operation_id = "test_op_123"
        actual_cost = 0.025
        operation_type = "llm_call"
        
        cost_based_circuit_breaker.update_actual_cost(operation_id, actual_cost, operation_type)
        
        # Should update total actual cost and call cost estimator
        assert cost_based_circuit_breaker._operation_stats['total_actual_cost'] == actual_cost
        cost_based_circuit_breaker.cost_estimator.update_historical_costs.assert_called_with(
            operation_type, actual_cost
        )


class TestCostBasedCircuitBreakerForceOperations:
    """Test forced state changes and manual control."""
    
    def test_force_open(self, cost_based_circuit_breaker):
        """Test manually forcing circuit breaker to open state."""
        reason = "Manual test intervention"
        cost_based_circuit_breaker.force_open(reason)
        
        assert cost_based_circuit_breaker.state == CircuitBreakerState.OPEN
        assert cost_based_circuit_breaker.last_failure_time is not None
    
    def test_force_close(self, cost_based_circuit_breaker, failing_function_factory):
        """Test manually forcing circuit breaker to closed state."""
        # First open the circuit
        failing_func = failing_function_factory(fail_count=5)
        for i in range(3):
            try:
                cost_based_circuit_breaker.call(failing_func, operation_type="test_op")
            except:
                pass
        
        assert cost_based_circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Force close
        reason = "Manual reset"
        cost_based_circuit_breaker.force_close(reason)
        
        assert cost_based_circuit_breaker.state == CircuitBreakerState.CLOSED
        assert cost_based_circuit_breaker.failure_count == 0
        assert cost_based_circuit_breaker._throttle_rate == 1.0
        assert len(cost_based_circuit_breaker._rule_cooldowns) == 0
    
    def test_get_status_comprehensive(self, cost_based_circuit_breaker, failing_function_factory):
        """Test comprehensive status reporting."""
        # Execute some operations to generate statistics
        success_func = failing_function_factory(fail_count=0)
        cost_based_circuit_breaker.call(success_func, operation_type="test_op")
        
        status = cost_based_circuit_breaker.get_status()
        
        # Check all expected fields are present
        expected_fields = [
            'name', 'state', 'failure_count', 'throttle_rate', 'active_rules',
            'statistics', 'last_failure_time', 'last_success_time', 
            'recovery_timeout', 'rules_count', 'cost_efficiency', 'timestamp'
        ]
        
        for field in expected_fields:
            assert field in status
        
        # Check specific values
        assert status['name'] == "test_breaker"
        assert status['state'] == CircuitBreakerState.CLOSED.value
        assert status['rules_count'] == 3  # From fixture
        assert 'estimated_vs_actual_ratio' in status['cost_efficiency']
        assert 'cost_savings' in status['cost_efficiency']
        assert 'block_rate' in status['cost_efficiency']


class TestCostCircuitBreakerManager:
    """Test CostCircuitBreakerManager functionality."""
    
    def test_manager_initialization(self, circuit_breaker_manager):
        """Test circuit breaker manager initialization."""
        assert circuit_breaker_manager.budget_manager is not None
        assert circuit_breaker_manager.cost_persistence is not None
        assert circuit_breaker_manager.cost_estimator is not None
        assert len(circuit_breaker_manager._circuit_breakers) == 0
        assert len(circuit_breaker_manager._default_rules) > 0
    
    def test_create_circuit_breaker(self, circuit_breaker_manager):
        """Test creating circuit breakers through manager."""
        breaker_name = "test_breaker"
        
        cb = circuit_breaker_manager.create_circuit_breaker(breaker_name)
        
        assert cb.name == breaker_name
        assert breaker_name in circuit_breaker_manager._circuit_breakers
        assert circuit_breaker_manager._manager_stats['breakers_created'] == 1
    
    def test_create_duplicate_circuit_breaker_raises_error(self, circuit_breaker_manager):
        """Test that creating duplicate circuit breaker raises error."""
        breaker_name = "duplicate_test"
        
        circuit_breaker_manager.create_circuit_breaker(breaker_name)
        
        with pytest.raises(ValueError, match="Circuit breaker.*already exists"):
            circuit_breaker_manager.create_circuit_breaker(breaker_name)
    
    def test_get_circuit_breaker(self, circuit_breaker_manager):
        """Test retrieving circuit breaker by name."""
        breaker_name = "test_breaker"
        created_cb = circuit_breaker_manager.create_circuit_breaker(breaker_name)
        
        retrieved_cb = circuit_breaker_manager.get_circuit_breaker(breaker_name)
        
        assert retrieved_cb == created_cb
        assert retrieved_cb.name == breaker_name
    
    def test_get_nonexistent_circuit_breaker(self, circuit_breaker_manager):
        """Test retrieving non-existent circuit breaker returns None."""
        result = circuit_breaker_manager.get_circuit_breaker("nonexistent")
        assert result is None
    
    def test_execute_with_protection(self, circuit_breaker_manager):
        """Test executing operation with circuit breaker protection."""
        def test_operation(**kwargs):
            return "operation_result"
        
        result = circuit_breaker_manager.execute_with_protection(
            "auto_created_breaker",
            test_operation,
            "test_operation"
        )
        
        assert result == "operation_result"
        assert circuit_breaker_manager._manager_stats['total_operations'] == 1
        assert "auto_created_breaker" in circuit_breaker_manager._circuit_breakers
    
    def test_execute_with_protection_blocked_operation(self, circuit_breaker_manager, budget_scenario_factory):
        """Test operation blocked by circuit breaker protection."""
        # Set budget to over limit to trigger blocking
        budget_scenario_factory(daily_over=True)
        
        def test_operation(**kwargs):
            return "should_not_execute"
        
        with pytest.raises(CircuitBreakerError):
            circuit_breaker_manager.execute_with_protection(
                "blocking_test_breaker",
                test_operation,
                "expensive_operation"
            )
        
        assert circuit_breaker_manager._manager_stats['total_blocks'] == 1
    
    def test_update_operation_cost_across_system(self, circuit_breaker_manager):
        """Test updating operation cost across the system."""
        breaker_name = "cost_update_test"
        circuit_breaker_manager.create_circuit_breaker(breaker_name)
        
        operation_id = "test_op_456"
        actual_cost = 0.035
        operation_type = "embedding_call"
        
        circuit_breaker_manager.update_operation_cost(
            breaker_name, operation_id, actual_cost, operation_type
        )
        
        # Should update both the specific breaker and global estimator
        cb = circuit_breaker_manager.get_circuit_breaker(breaker_name)
        assert cb._operation_stats['total_actual_cost'] == actual_cost
    
    def test_get_system_status(self, circuit_breaker_manager):
        """Test comprehensive system status reporting."""
        # Create a few circuit breakers
        circuit_breaker_manager.create_circuit_breaker("breaker1")
        circuit_breaker_manager.create_circuit_breaker("breaker2")
        
        status = circuit_breaker_manager.get_system_status()
        
        # Check top-level structure
        assert 'circuit_breakers' in status
        assert 'manager_statistics' in status
        assert 'cost_estimator_stats' in status
        assert 'system_health' in status
        assert 'timestamp' in status
        
        # Check circuit breaker statuses
        assert len(status['circuit_breakers']) == 2
        assert 'breaker1' in status['circuit_breakers']
        assert 'breaker2' in status['circuit_breakers']
        
        # Check manager stats
        assert status['manager_statistics']['breakers_created'] == 2
    
    def test_system_health_assessment_healthy(self, circuit_breaker_manager):
        """Test system health assessment when all breakers are healthy."""
        circuit_breaker_manager.create_circuit_breaker("healthy_breaker")
        
        health = circuit_breaker_manager._assess_system_health()
        
        assert health['status'] == 'healthy'
        assert health['total_breakers'] == 1
        assert 'All circuit breakers operational' in health['message']
    
    def test_system_health_assessment_open_breakers(self, circuit_breaker_manager):
        """Test system health assessment with open breakers."""
        cb = circuit_breaker_manager.create_circuit_breaker("failing_breaker")
        cb.force_open("Test failure")
        
        health = circuit_breaker_manager._assess_system_health()
        
        assert health['status'] == 'degraded'
        assert 'failing_breaker' in health['open_breakers']
        assert 'Circuit breakers open' in health['message']
    
    def test_system_health_assessment_budget_limited(self, circuit_breaker_manager, budget_scenario_factory):
        """Test system health assessment with budget limited breakers."""
        # Create breaker and trigger budget limited state
        cb = circuit_breaker_manager.create_circuit_breaker("budget_limited_breaker")
        budget_scenario_factory(daily_used_pct=96.0)
        cb._update_state()  # Force state update
        
        health = circuit_breaker_manager._assess_system_health()
        
        assert health['status'] == 'budget_limited'
        assert 'budget_limited_breaker' in health['limited_breakers']
    
    def test_emergency_shutdown(self, circuit_breaker_manager):
        """Test emergency shutdown of all circuit breakers."""
        cb1 = circuit_breaker_manager.create_circuit_breaker("breaker1")
        cb2 = circuit_breaker_manager.create_circuit_breaker("breaker2")
        
        reason = "Emergency test shutdown"
        circuit_breaker_manager.emergency_shutdown(reason)
        
        assert cb1.state == CircuitBreakerState.OPEN
        assert cb2.state == CircuitBreakerState.OPEN
    
    def test_reset_all_breakers(self, circuit_breaker_manager):
        """Test resetting all circuit breakers."""
        cb1 = circuit_breaker_manager.create_circuit_breaker("breaker1")
        cb2 = circuit_breaker_manager.create_circuit_breaker("breaker2")
        
        # Force open both breakers
        cb1.force_open("Test")
        cb2.force_open("Test")
        
        # Reset all
        reason = "Test reset all"
        circuit_breaker_manager.reset_all_breakers(reason)
        
        assert cb1.state == CircuitBreakerState.CLOSED
        assert cb2.state == CircuitBreakerState.CLOSED
        assert cb1.failure_count == 0
        assert cb2.failure_count == 0


class TestThreadSafetyAndConcurrency:
    """Test thread safety and concurrent access for cost-based circuit breaker."""
    
    def test_concurrent_operations(self, cost_based_circuit_breaker, load_generator):
        """Test concurrent operations through cost-based circuit breaker."""
        def test_operation():
            return cost_based_circuit_breaker.call(
                lambda: "concurrent_success",
                operation_type="concurrent_test"
            )
        
        # Generate concurrent load
        load_generator.generate_load(test_operation, duration_seconds=2, threads=5, requests_per_second=10)
        load_generator.stop_load()
        
        metrics = load_generator.get_metrics()
        
        # Should handle concurrent access gracefully
        assert metrics['total_requests'] > 0
        assert metrics['success_rate'] > 80  # Some may be blocked by cost rules
    
    def test_concurrent_state_updates(self, cost_based_circuit_breaker):
        """Test concurrent state updates are thread safe."""
        def update_state():
            cost_based_circuit_breaker._update_state()
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=update_state)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should maintain consistent state
        assert cost_based_circuit_breaker.state in [
            CircuitBreakerState.CLOSED,
            CircuitBreakerState.OPEN,
            CircuitBreakerState.HALF_OPEN,
            CircuitBreakerState.BUDGET_LIMITED
        ]
    
    def test_concurrent_cost_rule_evaluation(self, cost_based_circuit_breaker):
        """Test concurrent cost rule evaluation."""
        def evaluate_rules():
            return cost_based_circuit_breaker._check_cost_rules(
                estimated_cost=0.01,
                operation_type="concurrent_test",
                cost_estimate={'estimated_cost': 0.01, 'confidence': 0.8}
            )
        
        threads = []
        results = []
        
        def thread_worker():
            result = evaluate_rules()
            results.append(result)
        
        # Start multiple threads
        for _ in range(10):
            thread = threading.Thread(target=thread_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All results should be consistent
        assert len(results) == 10
        for result in results:
            assert 'allowed' in result


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for cost-based circuit breaker."""
    
    def test_budget_manager_error_handling(self, mock_budget_manager, mock_cost_estimator, cost_threshold_rules):
        """Test handling of budget manager errors."""
        # Make budget manager raise exception
        mock_budget_manager.get_budget_summary.side_effect = Exception("Budget service unavailable")
        
        cb = CostBasedCircuitBreaker(
            name="error_test",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=cost_threshold_rules
        )
        
        # Should handle error gracefully and default to closed state
        state = cb._check_cost_based_state()
        assert state == CircuitBreakerState.CLOSED
        
        # Should still allow operations despite budget manager error
        result = cb._check_cost_rules(
            estimated_cost=0.01,
            operation_type="test_op",
            cost_estimate={'estimated_cost': 0.01}
        )
        assert result['allowed']  # Should fail open for safety
    
    def test_cost_estimator_error_handling(self, cost_based_circuit_breaker):
        """Test handling of cost estimator errors."""
        # Make cost estimator raise exception
        cost_based_circuit_breaker.cost_estimator.estimate_operation_cost.side_effect = Exception("Estimation failed")
        
        # Should handle gracefully and allow operation
        def test_op():
            return "success"
        
        # Should not raise exception despite cost estimator failure
        result = cost_based_circuit_breaker.call(test_op, operation_type="test_op")
        assert result == "success"
    
    def test_zero_threshold_values(self, mock_budget_manager, mock_cost_estimator):
        """Test behavior with edge case threshold values."""
        # Note: CostThresholdRule should prevent zero values, but test robustness
        rules = []  # No rules to avoid validation
        
        cb = CostBasedCircuitBreaker(
            name="zero_test",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=rules
        )
        
        # Should work normally without rules
        result = cb.call(lambda: "success", operation_type="test_op")
        assert result == "success"
    
    def test_extremely_high_cost_operation(self, cost_based_circuit_breaker):
        """Test handling of extremely high cost operations."""
        def expensive_operation():
            return "expensive_result"
        
        # Should be blocked by operation cost rule
        with pytest.raises(CircuitBreakerError):
            cost_based_circuit_breaker.call(
                expensive_operation,
                operation_type="super_expensive",
                estimated_tokens={"input": 1000000, "output": 500000}  # Extremely high
            )
        
        assert cost_based_circuit_breaker._operation_stats['cost_blocked_calls'] == 1
    
    def test_negative_cost_estimation(self, cost_based_circuit_breaker, cost_estimation_scenario_factory):
        """Test handling of negative cost estimations."""
        # Create scenario with negative cost (edge case)
        scenario = {'test_op': {'estimated_cost': -0.01, 'confidence': 0.5}}
        cost_estimation_scenario_factory(scenario)
        
        # Should handle gracefully (negative costs don't make sense but shouldn't break)
        result = cost_based_circuit_breaker.call(
            lambda: "success",
            operation_type="test_op"
        )
        assert result == "success"
    
    def test_missing_operation_metadata(self, cost_based_circuit_breaker):
        """Test handling operations without proper metadata."""
        def simple_operation():
            return "simple_result"
        
        # Call without operation_type, model_name, etc.
        result = cost_based_circuit_breaker.call(simple_operation)
        
        # Should work with defaults
        assert result == "simple_result"
        assert cost_based_circuit_breaker._operation_stats['total_calls'] == 1
    
    def test_recovery_with_cost_constraints(self, cost_based_circuit_breaker, failing_function_factory, mock_time, budget_scenario_factory):
        """Test recovery behavior when cost constraints are active."""
        failing_func = failing_function_factory(fail_count=3)
        
        # Open circuit with failures
        for i in range(3):
            try:
                cost_based_circuit_breaker.call(failing_func, operation_type="test_op")
            except:
                pass
        
        assert cost_based_circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Set budget to over limit
        budget_scenario_factory(daily_over=True)
        
        # Advance time past recovery timeout
        mock_time.advance(1.5)
        
        # Even with recovery timeout elapsed, budget constraints should keep it limited
        state = cost_based_circuit_breaker._update_state()
        # Should be OPEN due to budget exceeded, not HALF_OPEN due to recovery
        assert state == CircuitBreakerState.OPEN


class TestCostBasedCircuitBreakerIntegration:
    """Integration tests for cost-based circuit breaker with realistic scenarios."""
    
    def test_realistic_budget_exhaustion_scenario(self, circuit_breaker_manager, budget_scenario_factory):
        """Test realistic budget exhaustion and recovery scenario."""
        # Start with healthy budget
        budget_scenario_factory(daily_used_pct=60.0)
        
        def api_call(**kwargs):
            return {"result": "api_response"}
        
        # Normal operations should work
        result = circuit_breaker_manager.execute_with_protection(
            "api_breaker",
            api_call,
            "api_call"
        )
        assert result["result"] == "api_response"
        
        # Budget approaching limit should trigger throttling
        budget_scenario_factory(daily_used_pct=92.0)
        
        # Operations should still work but may be throttled
        result = circuit_breaker_manager.execute_with_protection(
            "api_breaker",
            api_call,
            "api_call"
        )
        # Should still succeed but circuit breaker may be in budget_limited state
        
        # Budget exceeded should block operations
        budget_scenario_factory(daily_over=True)
        
        with pytest.raises(CircuitBreakerError):
            circuit_breaker_manager.execute_with_protection(
                "api_breaker",
                api_call,
                "expensive_api_call"
            )
    
    def test_multi_operation_type_cost_management(self, circuit_breaker_manager):
        """Test cost management across different operation types."""
        def llm_call(**kwargs):
            return "llm_response"
        
        def embedding_call(**kwargs):
            return "embedding_response"
        
        def batch_operation(**kwargs):
            return "batch_response"
        
        # Execute different types of operations
        operations = [
            ("llm_call", llm_call),
            ("embedding_call", embedding_call),
            ("batch_operation", batch_operation)
        ]
        
        for operation_type, operation_func in operations:
            result = circuit_breaker_manager.execute_with_protection(
                f"{operation_type}_breaker",
                operation_func,
                operation_type
            )
            assert result == f"{operation_type.split('_')[0]}_response"
        
        # Verify different breakers were created and managed
        assert len(circuit_breaker_manager._circuit_breakers) == 3
        
        # Get system status
        status = circuit_breaker_manager.get_system_status()
        assert len(status['circuit_breakers']) == 3
    
    def test_cost_learning_and_adaptation(self, circuit_breaker_manager):
        """Test that cost estimator learns and adapts over time."""
        def learning_operation(**kwargs):
            return "learning_result"
        
        # Execute operation multiple times
        for i in range(5):
            circuit_breaker_manager.execute_with_protection(
                "learning_breaker",
                learning_operation,
                "learning_operation"
            )
        
        # Simulate updating with actual costs
        actual_costs = [0.008, 0.012, 0.015, 0.009, 0.011]
        for i, cost in enumerate(actual_costs):
            circuit_breaker_manager.update_operation_cost(
                "learning_breaker",
                f"operation_{i}",
                cost,
                "learning_operation"
            )
        
        # Cost estimator should now have historical data
        estimator = circuit_breaker_manager.cost_estimator
        assert len(estimator._operation_costs["learning_operation"]) == 5
        
        # Next estimation should use historical data
        estimate = estimator.estimate_operation_cost("learning_operation")
        assert estimate['method'] == 'historical_average'
        assert estimate['confidence'] > 0.3  # Should be more confident with data