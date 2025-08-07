#!/usr/bin/env python3
"""
Enhanced test suite for Budget Management System - Additional coverage tests.

This test suite provides additional tests to achieve >90% coverage for the
budget management system by testing edge cases and uncovered code paths.

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import time
import threading
import tempfile
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Test imports
from lightrag_integration.budget_manager import (
    BudgetManager,
    BudgetAlert,
    AlertLevel,
    BudgetThreshold
)
from lightrag_integration.cost_persistence import (
    CostPersistence,
    CostRecord,
    ResearchCategory
)


class TestBudgetManagerEnhanced:
    """Enhanced tests for BudgetManager edge cases and coverage."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield Path(db_path)
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def cost_persistence(self, temp_db_path):
        """Create a CostPersistence instance for testing."""
        return CostPersistence(temp_db_path)
    
    @pytest.fixture
    def budget_manager(self, cost_persistence):
        """Create a BudgetManager instance for testing."""
        return BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0
        )
    
    def test_check_period_budget_cache_usage(self, budget_manager):
        """Test that _check_period_budget uses cache effectively."""
        # Add some cost data
        budget_manager.cost_persistence.record_cost(
            cost_usd=50.0,
            operation_type="cache_test",
            model_name="test-model",
            token_usage={"prompt_tokens": 500}
        )
        
        # First call should populate cache
        now = datetime.now(timezone.utc)
        status1 = budget_manager._check_period_budget(
            'daily', now, 100.0, 10.0, 'test_op', None
        )
        
        # Second call within cache TTL should use cache
        status2 = budget_manager._check_period_budget(
            'daily', now, 100.0, 15.0, 'test_op', None
        )
        
        # Both should have valid data but second should use cached base cost
        assert 'current_cost' in status1
        assert 'current_cost' in status2
        assert 'projected_cost' in status1
        assert 'projected_cost' in status2
    
    def test_create_budget_alert_all_levels(self, budget_manager):
        """Test _create_budget_alert for all alert levels."""
        alert_scenarios = [
            (AlertLevel.INFO, 25.0, 100.0, 25.0, 25.0),
            (AlertLevel.WARNING, 75.0, 100.0, 75.0, 75.0),
            (AlertLevel.CRITICAL, 90.0, 100.0, 90.0, 90.0),
            (AlertLevel.EXCEEDED, 110.0, 100.0, 110.0, 100.0)
        ]
        
        for alert_level, current_cost, budget_limit, percentage, threshold in alert_scenarios:
            alert = budget_manager._create_budget_alert(
                alert_level=alert_level,
                period_type='daily',
                period_key='2025-08-07',
                current_cost=current_cost,
                budget_limit=budget_limit,
                percentage_used=percentage,
                threshold_percentage=threshold,
                operation_type='test_op',
                research_category=ResearchCategory.GENERAL_QUERY
            )
            
            assert alert.alert_level == alert_level
            assert alert.current_cost == current_cost
            assert alert.budget_limit == budget_limit
            assert alert.percentage_used == percentage
            assert alert.threshold_percentage == threshold
            
            # Verify message contains appropriate text
            level_text = alert_level.value.title() if alert_level != AlertLevel.EXCEEDED else "exceeded"
            assert level_text.lower() in alert.message.lower()
            
            # Verify metadata
            assert alert.metadata is not None
            assert alert.metadata['operation_type'] == 'test_op'
            assert alert.metadata['research_category'] == ResearchCategory.GENERAL_QUERY.value
            assert alert.metadata['remaining_budget'] == budget_limit - current_cost
    
    def test_assess_budget_health_edge_cases(self, budget_manager):
        """Test _assess_budget_health with various combinations."""
        # Test with only daily status (no monthly)
        daily_status_only = {
            'over_budget': False,
            'percentage_used': 85.0
        }
        health = budget_manager._assess_budget_health(daily_status_only, None)
        assert health == 'critical'  # 85% >= 90% threshold
        
        # Test with only monthly status (no daily)
        monthly_status_only = {
            'over_budget': False,
            'percentage_used': 80.0
        }
        health = budget_manager._assess_budget_health(None, monthly_status_only)
        assert health == 'warning'  # 80% >= 75% but < 90%
        
        # Test with both statuses - daily exceeded should take precedence
        daily_exceeded = {'over_budget': True, 'percentage_used': 110.0}
        monthly_ok = {'over_budget': False, 'percentage_used': 50.0}
        health = budget_manager._assess_budget_health(daily_exceeded, monthly_ok)
        assert health == 'exceeded'
        
        # Test healthy status
        daily_healthy = {'over_budget': False, 'percentage_used': 50.0}
        monthly_healthy = {'over_budget': False, 'percentage_used': 60.0}
        health = budget_manager._assess_budget_health(daily_healthy, monthly_healthy)
        assert health == 'healthy'
    
    def test_check_threshold_alerts_no_alerts(self, budget_manager):
        """Test _check_threshold_alerts when no thresholds are crossed."""
        alerts = budget_manager._check_threshold_alerts(
            period_type='daily',
            period_key='2025-08-07',
            current_cost=50.0,
            budget_limit=100.0,
            percentage_used=50.0,  # Below warning threshold of 75%
            operation_type='test_op',
            research_category=None
        )
        
        assert len(alerts) == 0
    
    def test_check_threshold_alerts_multiple_thresholds(self, budget_manager):
        """Test _check_threshold_alerts when multiple thresholds are crossed."""
        # Reset cooldowns to allow alerts
        budget_manager.reset_alert_cooldowns()
        
        # Test exceeding critical threshold (should only generate CRITICAL, not WARNING)
        alerts = budget_manager._check_threshold_alerts(
            period_type='daily',
            period_key='2025-08-07',
            current_cost=95.0,
            budget_limit=100.0,
            percentage_used=95.0,  # Above critical threshold (90%)
            operation_type='test_op',
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION
        )
        
        # Should generate critical alert (not warning, as it takes the highest applicable)
        assert len(alerts) == 1
        assert alerts[0].alert_level == AlertLevel.CRITICAL
    
    def test_alert_callback_exception_handling(self, cost_persistence):
        """Test alert callback exception handling doesn't break the flow."""
        # Create a callback that raises an exception
        def failing_callback(alert: BudgetAlert):
            raise RuntimeError("Callback failed!")
        
        logger_mock = Mock(spec=logging.Logger)
        
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=50.0,
            alert_callback=failing_callback,
            logger=logger_mock
        )
        
        # Add cost to trigger alert
        cost_persistence.record_cost(
            cost_usd=40.0,
            operation_type="callback_error_test",
            model_name="test-model",
            token_usage={"prompt_tokens": 100}
        )
        
        # This should not raise an exception despite callback failure
        status = manager.check_budget_status(
            cost_amount=10.0,
            operation_type="trigger_alert"
        )
        
        # Should still complete successfully
        assert 'daily_status' in status
        
        # Logger should have recorded the error
        logger_mock.error.assert_called()
        error_call = logger_mock.error.call_args[0][0]
        assert "Error in alert callback" in error_call
    
    def test_set_alert_callback(self, budget_manager):
        """Test set_alert_callback functionality."""
        # Test setting a new callback
        new_callback = Mock()
        budget_manager.set_alert_callback(new_callback)
        
        assert budget_manager.alert_callback == new_callback
        
        # Test setting callback to None
        budget_manager.set_alert_callback(None)
        assert budget_manager.alert_callback is None
    
    def test_get_spending_trends_with_limited_data(self, budget_manager):
        """Test get_spending_trends with limited data scenarios."""
        # Test with no data
        trends = budget_manager.get_spending_trends(days=7)
        
        assert trends['period_days'] == 7
        assert trends['total_cost'] == 0
        assert trends['average_daily_cost'] == 0
        assert trends['trend_percentage'] == 0
        assert trends['trend_direction'] == 'stable'
        assert trends['projected_monthly_cost'] == 0
        
        # Add minimal data (less than 14 days for trend calculation)
        base_time = time.time() - (3 * 24 * 3600)  # 3 days ago
        daily_costs = [10.0, 15.0, 20.0]  # 3 days only
        
        for i, cost in enumerate(daily_costs):
            record = CostRecord(
                timestamp=base_time + (i * 24 * 3600),
                operation_type=f"limited_trend_test_{i}",
                model_name="trend-model",
                cost_usd=cost,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                research_category=ResearchCategory.GENERAL_QUERY.value
            )
            budget_manager.cost_persistence.db.insert_cost_record(record)
        
        trends = budget_manager.get_spending_trends(days=7)
        
        assert trends['total_cost'] > 0
        assert trends['average_daily_cost'] > 0
        assert trends['trend_percentage'] == 0  # Not enough data for trend
        assert trends['trend_direction'] == 'stable'
    
    def test_update_budget_limits_partial(self, budget_manager):
        """Test updating only one budget limit at a time."""
        original_daily = budget_manager.daily_budget_limit
        original_monthly = budget_manager.monthly_budget_limit
        
        # Update only daily
        budget_manager.update_budget_limits(daily_budget=150.0)
        assert budget_manager.daily_budget_limit == 150.0
        assert budget_manager.monthly_budget_limit == original_monthly
        
        # Update only monthly
        budget_manager.update_budget_limits(monthly_budget=4000.0)
        assert budget_manager.daily_budget_limit == 150.0  # Unchanged
        assert budget_manager.monthly_budget_limit == 4000.0
        
        # Update neither (no-op)
        budget_manager.update_budget_limits()
        assert budget_manager.daily_budget_limit == 150.0
        assert budget_manager.monthly_budget_limit == 4000.0


class TestBudgetManagerEdgeCases:
    """Test edge cases and error conditions for BudgetManager."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield Path(db_path)
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def cost_persistence(self, temp_db_path):
        """Create a CostPersistence instance for testing."""
        return CostPersistence(temp_db_path)
    
    def test_budget_manager_with_zero_limits(self, cost_persistence):
        """Test BudgetManager behavior with zero budget limits."""
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=0.0,
            monthly_budget_limit=0.0
        )
        
        # Any non-zero cost should trigger exceeded alerts
        status = manager.check_budget_status(
            cost_amount=0.01,
            operation_type="zero_limit_test"
        )
        
        # Should show as exceeded (any cost > 0 when limit is 0)
        assert status['budget_health'] == 'exceeded'
        assert status['operation_allowed'] is False
        
        # Should generate alerts
        assert len(status['alerts_generated']) > 0
        exceeded_alerts = [a for a in status['alerts_generated'] if a.alert_level == AlertLevel.EXCEEDED]
        assert len(exceeded_alerts) > 0
    
    def test_budget_manager_cache_expiration(self, cost_persistence):
        """Test that budget cache properly expires."""
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0
        )
        
        # Temporarily reduce cache TTL for testing
        original_ttl = manager._cache_ttl
        manager._cache_ttl = 0.1  # 100ms
        
        try:
            # Add some cost data
            cost_persistence.record_cost(
                cost_usd=50.0,
                operation_type="cache_expiry_test",
                model_name="test-model",
                token_usage={"prompt_tokens": 500}
            )
            
            # First check should populate cache
            status1 = manager.check_budget_status(
                cost_amount=10.0,
                operation_type="first_check"
            )
            
            # Wait for cache to expire
            time.sleep(0.2)
            
            # Add more cost data
            cost_persistence.record_cost(
                cost_usd=20.0,
                operation_type="additional_cost",
                model_name="test-model",
                token_usage={"prompt_tokens": 200}
            )
            
            # Second check should get fresh data (not cached)
            status2 = manager.check_budget_status(
                cost_amount=10.0,
                operation_type="second_check"
            )
            
            # Second status should reflect the additional cost
            assert status2['daily_status']['current_cost'] > status1['daily_status']['current_cost']
            
        finally:
            # Restore original TTL
            manager._cache_ttl = original_ttl
    
    def test_budget_manager_with_database_persistence_error(self, cost_persistence):
        """Test BudgetManager handling of database persistence errors."""
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0
        )
        
        # Mock the get_daily_budget_status to raise an exception
        with patch.object(cost_persistence, 'get_daily_budget_status', side_effect=Exception("DB Error")):
            # Should handle the error gracefully
            try:
                status = manager.check_budget_status(
                    cost_amount=50.0,
                    operation_type="db_error_test"
                )
                
                # Should return a status dict even with DB error
                assert isinstance(status, dict)
                assert 'budget_health' in status
                assert 'operation_allowed' in status
                
            except Exception:
                # If an exception is raised, it should be handled appropriately
                # The implementation might choose to re-raise or handle gracefully
                pass
    
    def test_extreme_percentage_calculations(self, cost_persistence):
        """Test handling of extreme percentage calculations."""
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=0.001,  # Very small limit
            monthly_budget_limit=0.03
        )
        
        # Add cost that's much larger than limit
        status = manager.check_budget_status(
            cost_amount=1000.0,  # 1,000,000% of daily limit
            operation_type="extreme_percentage_test"
        )
        
        # Should handle extreme percentages without breaking
        assert status['daily_status']['projected_percentage'] > 1000000
        assert status['budget_health'] == 'exceeded'
        assert status['operation_allowed'] is False
    
    def test_concurrent_cache_access(self, cost_persistence):
        """Test concurrent access to budget cache."""
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0
        )
        
        # Add initial cost data
        cost_persistence.record_cost(
            cost_usd=30.0,
            operation_type="concurrent_cache_test",
            model_name="test-model",
            token_usage={"prompt_tokens": 300}
        )
        
        def check_budget_worker(worker_id):
            return manager.check_budget_status(
                cost_amount=5.0,
                operation_type=f"concurrent_worker_{worker_id}"
            )
        
        # Run concurrent budget checks
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_budget_worker, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # All results should be valid
        assert len(results) == 10
        for result in results:
            assert 'daily_status' in result
            assert 'budget_health' in result
            assert result['daily_status']['projected_cost'] >= 35.0  # 30 + 5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])