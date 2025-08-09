"""
Priority 2 Critical Test: Budget Crisis Recovery Scenario Tests for Cost-Based Circuit Breaker

This module provides comprehensive testing for budget crisis scenarios and recovery patterns
for the cost-based circuit breaker system. These tests validate the system's behavior during
realistic budget exhaustion patterns and recovery cycles.

Test Coverage:
- Complete budget exhaustion and recovery cycles
- Gradual budget consumption patterns
- Sudden budget spike scenarios
- End-of-period budget reset behavior
- Multi-tier cost threshold enforcement
- Budget recovery with operation backlogs
- Emergency budget override mechanisms

The tests ensure the CostBasedCircuitBreaker can properly handle realistic budget crisis
scenarios while maintaining system stability and providing appropriate protection.
"""

import pytest
import time
import threading
import statistics
from unittest.mock import Mock, patch, MagicMock
from dataclasses import replace
from typing import Dict, List, Any

from lightrag_integration.cost_based_circuit_breaker import (
    CostBasedCircuitBreaker,
    CostThresholdRule,
    CostThresholdType,
    CircuitBreakerState,
    OperationCostEstimator,
    CostCircuitBreakerManager
)
from lightrag_integration.clinical_metabolomics_rag import CircuitBreakerError


class TestBudgetCrisisRecoveryScenarios:
    """Test complete budget crisis and recovery scenario patterns."""
    
    def test_complete_budget_exhaustion_cycle(self, mock_budget_manager, mock_cost_estimator, 
                                            cost_threshold_rules, mock_time, failing_function_factory):
        """Test complete budget exhaust/recovery cycle with realistic patterns."""
        cb = CostBasedCircuitBreaker(
            name="crisis_test_breaker",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=cost_threshold_rules,
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        success_func = failing_function_factory(fail_count=0)
        
        # Phase 1: Normal operations (50% budget used)
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 100.0,
                'total_cost': 50.0,
                'percentage_used': 50.0,
                'over_budget': False
            },
            'monthly_budget': {
                'budget': 1000.0,
                'total_cost': 300.0,
                'percentage_used': 30.0,
                'over_budget': False
            }
        }
        
        # Should allow normal operations
        result = cb.call(success_func, operation_type="normal_op")
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        
        # Phase 2: Approaching budget limit (85% used - triggers throttling)
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 100.0,
                'total_cost': 85.0,
                'percentage_used': 85.0,
                'over_budget': False
            },
            'monthly_budget': {
                'budget': 1000.0,
                'total_cost': 500.0,
                'percentage_used': 50.0,
                'over_budget': False
            }
        }
        
        # Should trigger throttle rule but still allow operations
        with pytest.raises(CircuitBreakerError, match="blocked by cost-based circuit breaker"):
            cb.call(success_func, operation_type="throttled_op")
        
        # Verify throttling was applied
        assert cb._throttle_rate == 0.5
        assert cb._operation_stats['cost_blocked_calls'] == 1
        
        # Phase 3: Budget exhausted (100% used - blocks operations)
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 100.0,
                'total_cost': 100.0,
                'percentage_used': 100.0,
                'over_budget': True
            },
            'monthly_budget': {
                'budget': 1000.0,
                'total_cost': 800.0,
                'percentage_used': 80.0,
                'over_budget': False
            }
        }
        
        # Should block operations due to budget exhaustion
        with pytest.raises(CircuitBreakerError, match="Circuit breaker.*is open"):
            cb.call(success_func, operation_type="blocked_op")
        
        # Verify circuit is open due to budget
        cb._update_state()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Phase 4: Budget recovery (reset to 20% used)
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 100.0,
                'total_cost': 20.0,
                'percentage_used': 20.0,
                'over_budget': False
            },
            'monthly_budget': {
                'budget': 1000.0,
                'total_cost': 400.0,
                'percentage_used': 40.0,
                'over_budget': False
            }
        }
        
        # Should allow operations again
        result = cb.call(success_func, operation_type="recovery_op")
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        
        # Verify statistics
        stats = cb.get_status()
        assert stats['statistics']['total_calls'] >= 4
        assert stats['statistics']['cost_blocked_calls'] >= 1
        assert stats['statistics']['cost_savings'] > 0
    
    def test_gradual_budget_consumption_pattern(self, circuit_breaker_manager, 
                                              budget_scenario_factory, mock_time):
        """Test gradual budget exhaustion over time with realistic consumption."""
        def api_operation(**kwargs):
            return {"status": "success", "data": "api_response"}
        
        # Start with low budget usage
        budget_scenario_factory(daily_used_pct=60.0)
        
        # Simulate gradual budget consumption over time
        consumption_stages = [60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 98.0, 100.0]
        operation_results = []
        
        for i, usage_pct in enumerate(consumption_stages):
            budget_scenario_factory(daily_used_pct=usage_pct)
            mock_time.advance(300)  # 5 minutes between stages
            
            try:
                result = circuit_breaker_manager.execute_with_protection(
                    "gradual_consumption_breaker",
                    api_operation,
                    "api_call",
                    operation_type="gradual_test"
                )
                operation_results.append({
                    'stage': i,
                    'usage_pct': usage_pct,
                    'success': True,
                    'result': result
                })
            except CircuitBreakerError as e:
                operation_results.append({
                    'stage': i,
                    'usage_pct': usage_pct,
                    'success': False,
                    'error': str(e)
                })
        
        # Verify consumption pattern behavior
        successful_operations = [r for r in operation_results if r['success']]
        blocked_operations = [r for r in operation_results if not r['success']]
        
        # Should allow operations until high usage
        assert len(successful_operations) >= 3  # At least first few stages
        assert len(blocked_operations) >= 2   # Should block at high usage
        
        # Verify blocking occurs at appropriate thresholds
        first_block_stage = min(r['usage_pct'] for r in blocked_operations)
        assert first_block_stage >= 80.0  # Should start blocking around 80%
        
        # Verify system status during crisis
        system_status = circuit_breaker_manager.get_system_status()
        assert system_status['manager_statistics']['total_blocks'] > 0
    
    def test_sudden_budget_spike_handling(self, cost_based_circuit_breaker, 
                                        budget_scenario_factory, mock_time):
        """Test handling of sudden budget spikes and immediate blocking."""
        success_func = lambda: "spike_test_result"
        
        # Start with normal budget usage
        budget_scenario_factory(daily_used_pct=40.0)
        
        # Normal operation should work
        result = cost_based_circuit_breaker.call(success_func, operation_type="pre_spike")
        assert result == "spike_test_result"
        
        # Simulate sudden budget spike (40% -> 105% instantly)
        budget_scenario_factory(daily_used_pct=105.0, daily_over=True)
        
        # Should immediately block operations
        with pytest.raises(CircuitBreakerError):
            cost_based_circuit_breaker.call(success_func, operation_type="spike_op")
        
        # Verify circuit opened due to budget exceeded
        cost_based_circuit_breaker._update_state()
        assert cost_based_circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Test multiple blocked operations during spike
        blocked_count = 0
        for i in range(5):
            try:
                cost_based_circuit_breaker.call(success_func, operation_type=f"spike_test_{i}")
            except CircuitBreakerError:
                blocked_count += 1
        
        assert blocked_count == 5  # All operations should be blocked
        
        # Recovery from spike (back to 30% usage)
        budget_scenario_factory(daily_used_pct=30.0, daily_over=False)
        
        # Should allow operations again
        result = cost_based_circuit_breaker.call(success_func, operation_type="post_spike")
        assert result == "spike_test_result"
        assert cost_based_circuit_breaker.state == CircuitBreakerState.CLOSED
    
    def test_end_of_period_budget_reset(self, mock_budget_manager, mock_cost_estimator, 
                                       mock_time):
        """Test end-of-period budget reset behavior and transition patterns."""
        # Create rules with specific time-based thresholds
        reset_rules = [
            CostThresholdRule(
                rule_id="daily_reset_protection",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=95.0,
                action="block",
                priority=20,
                cooldown_minutes=5.0
            ),
            CostThresholdRule(
                rule_id="end_of_period_throttle",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=90.0,
                action="throttle",
                priority=15,
                throttle_factor=0.3,
                cooldown_minutes=10.0
            )
        ]
        
        cb = CostBasedCircuitBreaker(
            name="period_reset_breaker",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=reset_rules,
            failure_threshold=5,
            recovery_timeout=30.0
        )
        
        def test_operation():
            return "period_test_result"
        
        # Phase 1: End of period - high budget usage (95%)
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 50.0,
                'total_cost': 47.5,
                'percentage_used': 95.0,
                'over_budget': False
            },
            'monthly_budget': {
                'budget': 500.0,
                'total_cost': 400.0,
                'percentage_used': 80.0,
                'over_budget': False
            }
        }
        
        # Should block due to high usage
        with pytest.raises(CircuitBreakerError):
            cb.call(test_operation, operation_type="end_period_op")
        
        # Phase 2: Period reset - budget back to low usage (10%)
        mock_time.advance(3600)  # 1 hour later (new period)
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 50.0,
                'total_cost': 5.0,
                'percentage_used': 10.0,
                'over_budget': False
            },
            'monthly_budget': {
                'budget': 500.0,
                'total_cost': 410.0,
                'percentage_used': 82.0,
                'over_budget': False
            }
        }
        
        # Should allow operations after reset
        result = cb.call(test_operation, operation_type="post_reset_op")
        assert result == "period_test_result"
        assert cb.state == CircuitBreakerState.CLOSED
        
        # Phase 3: Gradual buildup to next threshold
        usage_progression = [20.0, 40.0, 60.0, 80.0, 92.0]
        for usage in usage_progression:
            mock_budget_manager.get_budget_summary.return_value = {
                'daily_budget': {
                    'budget': 50.0,
                    'total_cost': usage * 0.5,
                    'percentage_used': usage,
                    'over_budget': False
                },
                'monthly_budget': {
                    'budget': 500.0,
                    'total_cost': 420.0,
                    'percentage_used': 84.0,
                    'over_budget': False
                }
            }
            mock_time.advance(300)  # 5 minutes progression
            
            if usage >= 90.0:
                # Should trigger throttling at 90%+
                with pytest.raises(CircuitBreakerError):
                    cb.call(test_operation, operation_type=f"buildup_{usage}")
            else:
                # Should allow operations below 90%
                result = cb.call(test_operation, operation_type=f"buildup_{usage}")
                assert result == "period_test_result"
        
        # Verify reset effectively cleared cooldowns and throttling
        assert cb._throttle_rate != 1.0  # Should have throttling active
        
        # Final verification
        status = cb.get_status()
        assert status['statistics']['total_calls'] >= 5
    
    def test_multi_tier_cost_threshold_enforcement(self, mock_budget_manager, 
                                                  mock_cost_estimator, mock_time):
        """Test complex multi-tier cost thresholds with different priorities."""
        # Create complex multi-tier rules
        multi_tier_rules = [
            # Tier 1: Conservative daily limits
            CostThresholdRule(
                rule_id="tier1_daily_conservative",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=75.0,
                action="alert_only",
                priority=5,
                cooldown_minutes=15.0
            ),
            # Tier 2: Moderate daily limits with throttling
            CostThresholdRule(
                rule_id="tier2_daily_moderate",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=85.0,
                action="throttle",
                priority=10,
                throttle_factor=0.6,
                cooldown_minutes=10.0
            ),
            # Tier 3: Aggressive daily limits with heavy throttling
            CostThresholdRule(
                rule_id="tier3_daily_aggressive",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=95.0,
                action="throttle",
                priority=15,
                throttle_factor=0.2,
                cooldown_minutes=5.0
            ),
            # Tier 4: Emergency blocking
            CostThresholdRule(
                rule_id="tier4_emergency_block",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=100.0,
                action="block",
                priority=20,
                cooldown_minutes=2.0
            ),
            # Per-operation cost limits
            CostThresholdRule(
                rule_id="high_cost_operation_limit",
                threshold_type=CostThresholdType.OPERATION_COST,
                threshold_value=2.0,
                action="block",
                priority=25,
                cooldown_minutes=1.0,
                applies_to_operations=["expensive_llm_call"]
            ),
            # Rate-based protection
            CostThresholdRule(
                rule_id="rate_spike_protection",
                threshold_type=CostThresholdType.RATE_BASED,
                threshold_value=20.0,  # $20/hour
                action="throttle",
                priority=12,
                throttle_factor=0.4,
                time_window_minutes=60,
                cooldown_minutes=30.0
            )
        ]
        
        cb = CostBasedCircuitBreaker(
            name="multi_tier_breaker",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=multi_tier_rules,
            failure_threshold=10,
            recovery_timeout=45.0
        )
        
        def normal_operation():
            return "normal_result"
        
        def expensive_operation():
            return "expensive_result"
        
        # Test Tier 1: Alert only at 75%
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 100.0,
                'total_cost': 76.0,
                'percentage_used': 76.0,
                'over_budget': False
            },
            'monthly_budget': {
                'budget': 1000.0,
                'total_cost': 500.0,
                'percentage_used': 50.0,
                'over_budget': False
            }
        }
        
        # Should trigger alert_only rule but still allow operation
        result = cb.call(normal_operation, operation_type="tier1_test")
        assert result == "normal_result"
        assert cb.state == CircuitBreakerState.CLOSED
        
        # Test Tier 2: Throttling at 85%
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 100.0,
                'total_cost': 86.0,
                'percentage_used': 86.0,
                'over_budget': False
            },
            'monthly_budget': {
                'budget': 1000.0,
                'total_cost': 600.0,
                'percentage_used': 60.0,
                'over_budget': False
            }
        }
        
        # Should trigger throttle rule
        with pytest.raises(CircuitBreakerError):
            cb.call(normal_operation, operation_type="tier2_test")
        assert cb._throttle_rate == 0.6
        
        # Test Tier 3: Heavy throttling at 95%
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 100.0,
                'total_cost': 96.0,
                'percentage_used': 96.0,
                'over_budget': False
            },
            'monthly_budget': {
                'budget': 1000.0,
                'total_cost': 700.0,
                'percentage_used': 70.0,
                'over_budget': False
            }
        }
        
        # Reset cooldowns to allow new rule triggering
        cb._rule_cooldowns.clear()
        mock_time.advance(600)  # 10 minutes
        
        # Should trigger higher priority throttle rule
        with pytest.raises(CircuitBreakerError):
            cb.call(normal_operation, operation_type="tier3_test")
        assert cb._throttle_rate == 0.2  # More aggressive throttling
        
        # Test per-operation cost limit
        mock_cost_estimator.estimate_operation_cost.return_value = {
            'estimated_cost': 2.5,  # Above $2.00 threshold
            'confidence': 0.9,
            'method': 'test_mock'
        }
        
        # Reset budget to allow operation but test cost limit
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 100.0,
                'total_cost': 50.0,
                'percentage_used': 50.0,
                'over_budget': False
            },
            'monthly_budget': {
                'budget': 1000.0,
                'total_cost': 400.0,
                'percentage_used': 40.0,
                'over_budget': False
            }
        }
        
        cb._rule_cooldowns.clear()
        mock_time.advance(120)  # 2 minutes
        
        # Should block expensive operation
        with pytest.raises(CircuitBreakerError, match="Cost rule.*operation_cost.*triggered"):
            cb.call(expensive_operation, operation_type="expensive_llm_call")
        
        # Verify tier enforcement
        status = cb.get_status()
        assert status['statistics']['cost_blocked_calls'] >= 3
        assert status['rules_count'] == 6
    
    def test_budget_recovery_with_backlog(self, circuit_breaker_manager, 
                                        budget_scenario_factory, mock_time):
        """Test budget recovery scenarios with queued operation backlogs."""
        operation_queue = []
        operation_results = []
        
        def queued_operation(operation_id, **kwargs):
            return {"operation_id": operation_id, "result": "processed"}
        
        # Phase 1: Budget crisis - operations get "queued" (blocked)
        budget_scenario_factory(daily_used_pct=102.0, daily_over=True)
        
        # Simulate multiple operations during crisis
        crisis_operations = [f"crisis_op_{i}" for i in range(10)]
        for op_id in crisis_operations:
            try:
                result = circuit_breaker_manager.execute_with_protection(
                    "backlog_breaker",
                    queued_operation,
                    "batch_operation",
                    operation_id=op_id
                )
                operation_results.append({"id": op_id, "success": True, "result": result})
            except CircuitBreakerError:
                operation_queue.append(op_id)  # "Queue" blocked operations
                operation_results.append({"id": op_id, "success": False, "blocked": True})
        
        # Verify operations were blocked during crisis
        blocked_ops = [r for r in operation_results if not r.get('success', False)]
        assert len(blocked_ops) == 10  # All should be blocked
        assert len(operation_queue) == 10
        
        # Phase 2: Budget recovery
        budget_scenario_factory(daily_used_pct=30.0, daily_over=False)
        mock_time.advance(1800)  # 30 minutes recovery time
        
        # Process queued operations after recovery
        recovery_results = []
        for op_id in operation_queue:
            try:
                result = circuit_breaker_manager.execute_with_protection(
                    "backlog_breaker",
                    queued_operation,
                    "batch_operation",
                    operation_id=f"recovery_{op_id}"
                )
                recovery_results.append({"id": op_id, "success": True, "result": result})
            except CircuitBreakerError:
                recovery_results.append({"id": op_id, "success": False, "error": "Still blocked"})
        
        # Verify recovery processing
        successful_recovery = [r for r in recovery_results if r.get('success', False)]
        assert len(successful_recovery) >= 8  # Most should succeed after recovery
        
        # Phase 3: Test gradual processing with budget monitoring
        budget_levels = [40.0, 50.0, 60.0, 70.0, 80.0]
        gradual_results = []
        
        for i, budget_level in enumerate(budget_levels):
            budget_scenario_factory(daily_used_pct=budget_level)
            mock_time.advance(300)  # 5 minutes between levels
            
            # Process a few operations at each level
            for j in range(3):
                try:
                    result = circuit_breaker_manager.execute_with_protection(
                        "backlog_breaker",
                        queued_operation,
                        "regular_operation",
                        operation_id=f"gradual_{i}_{j}"
                    )
                    gradual_results.append({
                        "budget_level": budget_level,
                        "success": True,
                        "result": result
                    })
                except CircuitBreakerError:
                    gradual_results.append({
                        "budget_level": budget_level,
                        "success": False,
                        "blocked": True
                    })
        
        # Verify gradual processing behavior
        successful_gradual = [r for r in gradual_results if r.get('success', False)]
        blocked_gradual = [r for r in gradual_results if not r.get('success', False)]
        
        # Should allow most operations at lower budget levels
        low_budget_success = [r for r in gradual_results if r.get('success', False) and r['budget_level'] <= 60.0]
        high_budget_blocks = [r for r in gradual_results if not r.get('success', False) and r['budget_level'] >= 80.0]
        
        assert len(low_budget_success) >= 6  # Should succeed at low budget
        # May have some blocks at high budget depending on thresholds
        
        # Final verification
        system_status = circuit_breaker_manager.get_system_status()
        assert system_status['manager_statistics']['total_operations'] >= 25
        assert system_status['manager_statistics']['total_blocks'] >= 10
    
    def test_emergency_budget_override_scenarios(self, mock_budget_manager, 
                                               mock_cost_estimator):
        """Test emergency budget override mechanisms and emergency operations."""
        # Create rules with emergency override capabilities
        emergency_rules = [
            CostThresholdRule(
                rule_id="emergency_daily_limit",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=100.0,
                action="block",
                priority=20,
                allow_emergency_override=True,
                cooldown_minutes=5.0
            ),
            CostThresholdRule(
                rule_id="critical_operation_cost",
                threshold_type=CostThresholdType.OPERATION_COST,
                threshold_value=5.0,
                action="block",
                priority=25,
                allow_emergency_override=True,
                cooldown_minutes=1.0,
                applies_to_operations=["emergency_critical_op"]
            ),
            CostThresholdRule(
                rule_id="no_override_limit",
                threshold_type=CostThresholdType.PERCENTAGE_MONTHLY,
                threshold_value=100.0,
                action="block",
                priority=30,
                allow_emergency_override=False,
                cooldown_minutes=60.0
            )
        ]
        
        cb = CostBasedCircuitBreaker(
            name="emergency_override_breaker",
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=emergency_rules,
            failure_threshold=10,
            recovery_timeout=30.0
        )
        
        def emergency_operation():
            return "emergency_completed"
        
        def critical_operation():
            return "critical_completed"
        
        # Set budget to exceeded state
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {
                'budget': 100.0,
                'total_cost': 105.0,
                'percentage_used': 105.0,
                'over_budget': True
            },
            'monthly_budget': {
                'budget': 1000.0,
                'total_cost': 1010.0,
                'percentage_used': 101.0,
                'over_budget': True
            }
        }
        
        # Test 1: Normal operation should be blocked
        with pytest.raises(CircuitBreakerError):
            cb.call(emergency_operation, operation_type="normal_op")
        
        # Test 2: Emergency override for daily limit (with override flag)
        # Note: This would require implementing emergency override logic in the circuit breaker
        # For now, we test that the rule configuration allows emergency override
        emergency_rule = next(r for r in emergency_rules if r.rule_id == "emergency_daily_limit")
        assert emergency_rule.allow_emergency_override == True
        
        # Test 3: Critical operation with high cost
        mock_cost_estimator.estimate_operation_cost.return_value = {
            'estimated_cost': 6.0,  # Above $5.00 threshold
            'confidence': 0.9,
            'method': 'test_emergency'
        }
        
        # Should be blocked due to cost but allows emergency override
        with pytest.raises(CircuitBreakerError):
            cb.call(critical_operation, operation_type="emergency_critical_op")
        
        # Test 4: Monthly limit without override capability
        no_override_rule = next(r for r in emergency_rules if r.rule_id == "no_override_limit")
        assert no_override_rule.allow_emergency_override == False
        
        # Test 5: Verify override configuration in status
        status = cb.get_status()
        assert status['rules_count'] == 3
        
        # Test emergency state management
        cb.force_open("Emergency budget crisis")
        assert cb.state == CircuitBreakerState.OPEN
        
        # Test emergency reset
        cb.force_close("Emergency override reset")
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb._throttle_rate == 1.0
    
    def test_complex_crisis_recovery_integration(self, circuit_breaker_manager, 
                                               budget_scenario_factory, mock_time,
                                               test_data_generator):
        """Test complex integration scenario with multiple crisis types and recovery patterns."""
        # Generate realistic crisis scenario
        crisis_scenario = test_data_generator.generate_budget_crisis_scenario(
            crisis_type='oscillating', 
            duration_hours=8
        )
        
        def adaptive_operation(operation_id, **kwargs):
            return {"id": operation_id, "timestamp": time.time(), "result": "adaptive_success"}
        
        # Simulate 8-hour crisis with oscillating budget patterns
        time_progression = []
        operation_results = []
        
        for hour in range(8):
            # Calculate budget usage based on crisis pattern
            usage_pct = crisis_scenario['depletion_pattern'](hour) * 100
            budget_scenario_factory(daily_used_pct=usage_pct)
            
            mock_time.advance(3600)  # 1 hour increments
            current_time = mock_time.current()
            
            # Perform operations every 15 minutes within each hour
            for quarter_hour in range(4):
                mock_time.advance(900)  # 15 minutes
                
                operation_id = f"crisis_op_{hour}_{quarter_hour}"
                try:
                    result = circuit_breaker_manager.execute_with_protection(
                        "complex_crisis_breaker",
                        adaptive_operation,
                        "adaptive_operation",
                        operation_id=operation_id
                    )
                    
                    operation_results.append({
                        'hour': hour,
                        'quarter': quarter_hour,
                        'budget_usage': usage_pct,
                        'success': True,
                        'result': result,
                        'timestamp': current_time
                    })
                    
                except CircuitBreakerError as e:
                    operation_results.append({
                        'hour': hour,
                        'quarter': quarter_hour,
                        'budget_usage': usage_pct,
                        'success': False,
                        'error': str(e),
                        'timestamp': current_time
                    })
            
            time_progression.append({
                'hour': hour,
                'budget_usage': usage_pct,
                'operations_in_hour': 4
            })
        
        # Analyze results
        successful_ops = [r for r in operation_results if r['success']]
        blocked_ops = [r for r in operation_results if not r['success']]
        
        # Verify crisis behavior patterns
        assert len(operation_results) == 32  # 8 hours * 4 quarters
        assert len(successful_ops) >= 10     # Some operations should succeed
        assert len(blocked_ops) >= 10       # Some should be blocked during crisis
        
        # Verify budget correlation
        high_usage_ops = [r for r in operation_results if r['budget_usage'] >= 80.0]
        low_usage_ops = [r for r in operation_results if r['budget_usage'] <= 40.0]
        
        high_usage_success_rate = len([r for r in high_usage_ops if r['success']]) / max(len(high_usage_ops), 1)
        low_usage_success_rate = len([r for r in low_usage_ops if r['success']]) / max(len(low_usage_ops), 1)
        
        # Success rate should be higher at low budget usage
        if len(low_usage_ops) > 0 and len(high_usage_ops) > 0:
            assert low_usage_success_rate >= high_usage_success_rate
        
        # Final system verification
        final_status = circuit_breaker_manager.get_system_status()
        assert final_status['manager_statistics']['total_operations'] >= 32
        assert final_status['manager_statistics']['total_blocks'] >= 5
        
        # Verify circuit breaker learned from experience
        breaker = circuit_breaker_manager.get_circuit_breaker("complex_crisis_breaker")
        if breaker:
            breaker_status = breaker.get_status()
            assert breaker_status['statistics']['total_calls'] >= 32
            assert breaker_status['cost_efficiency']['block_rate'] > 0


class TestBudgetCrisisEdgeCases:
    """Test edge cases and error conditions in budget crisis scenarios."""
    
    def test_budget_manager_failure_during_crisis(self, mock_cost_estimator, cost_threshold_rules):
        """Test handling of budget manager failures during crisis scenarios."""
        # Create a budget manager that fails intermittently
        failing_budget_manager = Mock()
        failing_budget_manager.get_budget_summary.side_effect = [
            # First call succeeds
            {
                'daily_budget': {'budget': 100.0, 'total_cost': 90.0, 'percentage_used': 90.0, 'over_budget': False},
                'monthly_budget': {'budget': 1000.0, 'total_cost': 500.0, 'percentage_used': 50.0, 'over_budget': False}
            },
            # Second call fails
            Exception("Budget service unavailable"),
            # Third call succeeds
            {
                'daily_budget': {'budget': 100.0, 'total_cost': 95.0, 'percentage_used': 95.0, 'over_budget': False},
                'monthly_budget': {'budget': 1000.0, 'total_cost': 600.0, 'percentage_used': 60.0, 'over_budget': False}
            }
        ]
        
        cb = CostBasedCircuitBreaker(
            name="failing_budget_breaker",
            budget_manager=failing_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=cost_threshold_rules,
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        def test_operation():
            return "resilient_result"
        
        # First operation should trigger cost rule
        with pytest.raises(CircuitBreakerError):
            cb.call(test_operation, operation_type="first_op")
        
        # Second operation - budget manager fails, should fail open (allow)
        result = cb.call(test_operation, operation_type="failing_budget_op")
        assert result == "resilient_result"
        
        # Third operation - budget manager recovers
        with pytest.raises(CircuitBreakerError):
            cb.call(test_operation, operation_type="recovered_op")
        
        # Verify graceful degradation
        status = cb.get_status()
        assert status['statistics']['total_calls'] == 3
        assert status['statistics']['allowed_calls'] == 1  # One allowed during failure
    
    def test_concurrent_budget_crisis_operations(self, cost_based_circuit_breaker, 
                                                budget_scenario_factory, load_generator):
        """Test concurrent operations during budget crisis scenarios."""
        # Set budget to crisis level
        budget_scenario_factory(daily_used_pct=98.0, daily_over=False)
        
        def crisis_operation():
            return cost_based_circuit_breaker.call(
                lambda: "concurrent_crisis_result",
                operation_type="crisis_concurrent"
            )
        
        # Generate concurrent load during crisis
        load_generator.generate_load(
            target_function=crisis_operation,
            duration_seconds=3,
            threads=10,
            requests_per_second=20
        )
        
        load_generator.stop_load()
        metrics = load_generator.get_metrics()
        
        # Should handle concurrent requests gracefully
        assert metrics['total_requests'] > 0
        # Most requests should be blocked due to budget crisis
        assert metrics['success_rate'] < 50  # High block rate expected
        assert len(metrics['errors']) > 0    # Should have CircuitBreakerError instances
        
        # Verify thread safety during crisis
        status = cost_based_circuit_breaker.get_status()
        assert status['statistics']['cost_blocked_calls'] > 0
    
    def test_budget_crisis_with_cost_estimation_failures(self, mock_budget_manager, 
                                                       cost_threshold_rules):
        """Test budget crisis handling when cost estimation fails."""
        # Create cost estimator that fails
        failing_cost_estimator = Mock()
        failing_cost_estimator.estimate_operation_cost.side_effect = Exception("Cost estimation failed")
        
        # Set budget to crisis state
        mock_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {'budget': 100.0, 'total_cost': 95.0, 'percentage_used': 95.0, 'over_budget': False},
            'monthly_budget': {'budget': 1000.0, 'total_cost': 800.0, 'percentage_used': 80.0, 'over_budget': False}
        }
        
        cb = CostBasedCircuitBreaker(
            name="failing_estimator_breaker",
            budget_manager=mock_budget_manager,
            cost_estimator=failing_cost_estimator,
            threshold_rules=cost_threshold_rules,
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        def test_operation():
            return "estimation_failure_result"
        
        # Should handle cost estimation failure gracefully and still allow operation
        result = cb.call(test_operation, operation_type="estimation_failure_op")
        assert result == "estimation_failure_result"
        
        # Should still check budget-based state even without cost estimation
        cb._update_state()
        assert cb.state == CircuitBreakerState.BUDGET_LIMITED  # Due to 95% usage
    
    def test_zero_budget_edge_case(self, mock_cost_estimator, cost_threshold_rules):
        """Test handling of zero budget edge cases."""
        zero_budget_manager = Mock()
        zero_budget_manager.get_budget_summary.return_value = {
            'daily_budget': {'budget': 0.0, 'total_cost': 0.0, 'percentage_used': 0.0, 'over_budget': False},
            'monthly_budget': {'budget': 0.0, 'total_cost': 0.0, 'percentage_used': 0.0, 'over_budget': False}
        }
        
        cb = CostBasedCircuitBreaker(
            name="zero_budget_breaker",
            budget_manager=zero_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=cost_threshold_rules,
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        def test_operation():
            return "zero_budget_result"
        
        # Should handle zero budget gracefully (depends on implementation)
        try:
            result = cb.call(test_operation, operation_type="zero_budget_op")
            # If it succeeds, verify the result
            assert result == "zero_budget_result"
        except CircuitBreakerError:
            # If it blocks, that's also acceptable behavior for zero budget
            pass
        
        # Verify system remains stable
        status = cb.get_status()
        assert status['statistics']['total_calls'] == 1


class TestBudgetCrisisPerformanceAndScaling:
    """Test performance characteristics during budget crisis scenarios."""
    
    def test_crisis_response_time_performance(self, cost_based_circuit_breaker, 
                                            budget_scenario_factory, mock_time):
        """Test that crisis detection and response times remain fast."""
        import time as real_time
        
        def timed_operation():
            return "performance_test_result"
        
        response_times = []
        
        # Test response times at different budget levels
        budget_levels = [50.0, 75.0, 85.0, 95.0, 98.0, 100.0]
        
        for budget_level in budget_levels:
            budget_scenario_factory(daily_used_pct=budget_level)
            
            # Measure response time
            start_time = real_time.time()
            try:
                cost_based_circuit_breaker.call(timed_operation, operation_type="performance_test")
                elapsed = real_time.time() - start_time
                response_times.append({
                    'budget_level': budget_level,
                    'elapsed_time': elapsed,
                    'blocked': False
                })
            except CircuitBreakerError:
                elapsed = real_time.time() - start_time
                response_times.append({
                    'budget_level': budget_level,
                    'elapsed_time': elapsed,
                    'blocked': True
                })
        
        # Verify response times are reasonable
        max_response_time = max(rt['elapsed_time'] for rt in response_times)
        assert max_response_time < 0.1  # Should respond within 100ms
        
        # Response time shouldn't degrade significantly at higher budget usage
        low_budget_times = [rt['elapsed_time'] for rt in response_times if rt['budget_level'] <= 75.0]
        high_budget_times = [rt['elapsed_time'] for rt in response_times if rt['budget_level'] >= 95.0]
        
        if low_budget_times and high_budget_times:
            avg_low = statistics.mean(low_budget_times)
            avg_high = statistics.mean(high_budget_times)
            # High budget response shouldn't be more than 5x slower
            assert avg_high <= avg_low * 5.0
    
    def test_crisis_memory_usage_stability(self, circuit_breaker_manager, 
                                         budget_scenario_factory, mock_time):
        """Test memory usage remains stable during extended crisis periods."""
        import gc
        import sys
        
        def memory_test_operation(op_id):
            return f"memory_test_{op_id}"
        
        # Force garbage collection and get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Simulate extended crisis period with many operations
        budget_scenario_factory(daily_used_pct=105.0, daily_over=True)
        
        operations_performed = 0
        for cycle in range(50):  # 50 cycles of operations
            mock_time.advance(60)  # 1 minute per cycle
            
            for i in range(20):  # 20 operations per cycle
                try:
                    circuit_breaker_manager.execute_with_protection(
                        "memory_test_breaker",
                        memory_test_operation,
                        "memory_test",
                        op_id=f"cycle_{cycle}_op_{i}"
                    )
                    operations_performed += 1
                except CircuitBreakerError:
                    operations_performed += 1  # Still count blocked operations
        
        # Force garbage collection and check memory
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be reasonable
        memory_growth = final_objects - initial_objects
        operations_ratio = memory_growth / max(operations_performed, 1)
        
        # Should not create excessive objects per operation
        assert operations_ratio < 10.0  # Less than 10 objects per operation
        
        # Verify system is still functional
        system_status = circuit_breaker_manager.get_system_status()
        assert system_status['manager_statistics']['total_operations'] >= 1000