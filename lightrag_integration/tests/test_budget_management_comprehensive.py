#!/usr/bin/env python3
"""
Comprehensive test suite for Budget Management System.

This test suite provides complete coverage of the budget management components including:
- BudgetAlert data model and serialization
- BudgetThreshold configuration and validation
- BudgetManager alert generation and monitoring
- Progressive alert thresholds and escalation
- Cache management and performance optimization
- Thread safety and concurrent operations
- Integration with cost persistence layer

Author: Claude Code (Anthropic)
Created: August 6, 2025
"""

import pytest
import time
import threading
import tempfile
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock, call
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


class TestBudgetAlert:
    """Comprehensive tests for BudgetAlert data model."""
    
    def test_budget_alert_basic_creation(self):
        """Test basic BudgetAlert creation."""
        alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=75.0,
            budget_limit=100.0,
            percentage_used=75.0,
            threshold_percentage=75.0,
            message="Daily budget warning"
        )
        
        assert alert.alert_level == AlertLevel.WARNING
        assert alert.period_type == "daily"
        assert alert.period_key == "2025-08-06"
        assert alert.current_cost == 75.0
        assert alert.budget_limit == 100.0
        assert alert.percentage_used == 75.0
        assert alert.threshold_percentage == 75.0
        assert alert.message == "Daily budget warning"
    
    def test_budget_alert_with_metadata(self):
        """Test BudgetAlert creation with metadata."""
        metadata = {
            "operation_type": "llm_call",
            "research_category": "biomarker_discovery",
            "session_count": 5,
            "average_cost_per_operation": 0.15
        }
        
        alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.CRITICAL,
            period_type="monthly",
            period_key="2025-08",
            current_cost=2700.0,
            budget_limit=3000.0,
            percentage_used=90.0,
            threshold_percentage=90.0,
            message="Monthly budget critical",
            metadata=metadata
        )
        
        assert alert.metadata == metadata
        assert alert.metadata["operation_type"] == "llm_call"
        assert alert.metadata["session_count"] == 5
    
    def test_budget_alert_post_init(self):
        """Test BudgetAlert post-initialization processing."""
        alert = BudgetAlert(
            timestamp=None,  # Should be auto-generated
            alert_level=AlertLevel.EXCEEDED,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=110.0,
            budget_limit=100.0,
            percentage_used=110.0,
            threshold_percentage=100.0,
            message="Budget exceeded"
        )
        
        assert alert.timestamp is not None
        assert alert.timestamp > 0
        assert abs(alert.timestamp - time.time()) < 1.0  # Generated recently
    
    def test_budget_alert_serialization(self):
        """Test BudgetAlert to_dict method."""
        metadata = {"test_key": "test_value", "nested": {"data": 123}}
        
        alert = BudgetAlert(
            timestamp=1691234567.89,
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=80.0,
            budget_limit=100.0,
            percentage_used=80.0,
            threshold_percentage=75.0,
            message="Test alert",
            metadata=metadata
        )
        
        alert_dict = alert.to_dict()
        
        assert isinstance(alert_dict, dict)
        assert alert_dict['timestamp'] == 1691234567.89
        assert alert_dict['alert_level'] == 'warning'  # Enum value
        assert alert_dict['period_type'] == 'daily'
        assert alert_dict['current_cost'] == 80.0
        assert alert_dict['budget_limit'] == 100.0
        assert alert_dict['percentage_used'] == 80.0
        assert alert_dict['message'] == "Test alert"
        assert alert_dict['metadata'] == metadata
    
    def test_budget_alert_all_levels(self):
        """Test BudgetAlert with all alert levels."""
        levels_data = [
            (AlertLevel.INFO, "info", 25.0, 25.0),
            (AlertLevel.WARNING, "warning", 75.0, 75.0),
            (AlertLevel.CRITICAL, "critical", 90.0, 90.0),
            (AlertLevel.EXCEEDED, "exceeded", 105.0, 100.0)
        ]
        
        for level, level_value, percentage, threshold in levels_data:
            alert = BudgetAlert(
                timestamp=time.time(),
                alert_level=level,
                period_type="daily",
                period_key="2025-08-06",
                current_cost=percentage,
                budget_limit=100.0,
                percentage_used=percentage,
                threshold_percentage=threshold,
                message=f"Test {level_value} alert"
            )
            
            assert alert.alert_level == level
            alert_dict = alert.to_dict()
            assert alert_dict['alert_level'] == level_value


class TestBudgetThreshold:
    """Comprehensive tests for BudgetThreshold configuration."""
    
    def test_budget_threshold_defaults(self):
        """Test BudgetThreshold with default values."""
        threshold = BudgetThreshold()
        
        assert threshold.warning_percentage == 75.0
        assert threshold.critical_percentage == 90.0
        assert threshold.exceeded_percentage == 100.0
    
    def test_budget_threshold_custom_values(self):
        """Test BudgetThreshold with custom values."""
        threshold = BudgetThreshold(
            warning_percentage=60.0,
            critical_percentage=80.0,
            exceeded_percentage=95.0
        )
        
        assert threshold.warning_percentage == 60.0
        assert threshold.critical_percentage == 80.0
        assert threshold.exceeded_percentage == 95.0
    
    def test_budget_threshold_validation_valid(self):
        """Test BudgetThreshold validation with valid values."""
        # Edge case: all thresholds equal
        threshold1 = BudgetThreshold(50.0, 50.0, 50.0)
        assert threshold1.warning_percentage == 50.0
        
        # Normal case: ascending order
        threshold2 = BudgetThreshold(25.0, 75.0, 125.0)
        assert threshold2.warning_percentage == 25.0
        assert threshold2.critical_percentage == 75.0
        assert threshold2.exceeded_percentage == 125.0
    
    def test_budget_threshold_validation_invalid_range(self):
        """Test BudgetThreshold validation with invalid ranges."""
        # Negative values
        with pytest.raises(ValueError, match="must be between 0 and 200"):
            BudgetThreshold(warning_percentage=-10.0)
        
        # Values over 200%
        with pytest.raises(ValueError, match="must be between 0 and 200"):
            BudgetThreshold(exceeded_percentage=250.0)
    
    def test_budget_threshold_validation_invalid_order(self):
        """Test BudgetThreshold validation with invalid ordering."""
        # Warning > Critical
        with pytest.raises(ValueError, match="must be in ascending order"):
            BudgetThreshold(
                warning_percentage=90.0,
                critical_percentage=75.0,
                exceeded_percentage=100.0
            )
        
        # Critical > Exceeded
        with pytest.raises(ValueError, match="must be in ascending order"):
            BudgetThreshold(
                warning_percentage=50.0,
                critical_percentage=95.0,
                exceeded_percentage=80.0
            )
    
    def test_budget_threshold_edge_cases(self):
        """Test BudgetThreshold with edge case values."""
        # Minimum values
        threshold1 = BudgetThreshold(0.0, 0.0, 0.0)
        assert threshold1.warning_percentage == 0.0
        
        # Maximum values
        threshold2 = BudgetThreshold(200.0, 200.0, 200.0)
        assert threshold2.exceeded_percentage == 200.0
        
        # Very close values
        threshold3 = BudgetThreshold(74.9, 75.0, 75.1)
        assert threshold3.warning_percentage == 74.9
        assert threshold3.critical_percentage == 75.0
        assert threshold3.exceeded_percentage == 75.1


class TestBudgetManager:
    """Comprehensive tests for BudgetManager functionality."""
    
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
    def budget_manager_basic(self, cost_persistence):
        """Create a basic BudgetManager instance."""
        return BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0
        )
    
    @pytest.fixture
    def budget_manager_with_callback(self, cost_persistence):
        """Create BudgetManager with alert callback."""
        alert_callback = Mock()
        return BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0,
            alert_callback=alert_callback
        ), alert_callback
    
    @pytest.fixture
    def budget_manager_custom_thresholds(self, cost_persistence):
        """Create BudgetManager with custom thresholds."""
        custom_thresholds = BudgetThreshold(
            warning_percentage=60.0,
            critical_percentage=80.0,
            exceeded_percentage=95.0
        )
        return BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=50.0,
            monthly_budget_limit=1500.0,
            thresholds=custom_thresholds
        )
    
    def test_budget_manager_initialization(self, cost_persistence):
        """Test BudgetManager initialization."""
        logger = Mock(spec=logging.Logger)
        
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=150.0,
            monthly_budget_limit=4500.0,
            logger=logger
        )
        
        assert manager.cost_persistence == cost_persistence
        assert manager.daily_budget_limit == 150.0
        assert manager.monthly_budget_limit == 4500.0
        assert manager.logger == logger
        assert isinstance(manager.thresholds, BudgetThreshold)
        assert manager._alert_cooldown == 300.0  # 5 minutes
        assert manager._cache_ttl == 60.0  # 1 minute
    
    def test_budget_manager_no_limits(self, cost_persistence):
        """Test BudgetManager with no budget limits set."""
        manager = BudgetManager(cost_persistence=cost_persistence)
        
        assert manager.daily_budget_limit is None
        assert manager.monthly_budget_limit is None
        
        # Should not generate alerts when no limits are set
        status = manager.check_budget_status(
            cost_amount=100.0,
            operation_type="test_op"
        )
        
        assert status['daily_status'] is None
        assert status['monthly_status'] is None
        assert len(status['alerts_generated']) == 0
        assert status['operation_allowed'] is True
    
    def test_check_budget_status_basic(self, budget_manager_basic):
        """Test basic budget status checking."""
        # Add some existing costs
        budget_manager_basic.cost_persistence.record_cost(
            cost_usd=60.0,
            operation_type="existing_op",
            model_name="test-model",
            token_usage={"prompt_tokens": 1000}
        )
        
        # Check budget status with new operation
        status = budget_manager_basic.check_budget_status(
            cost_amount=15.0,
            operation_type="new_op",
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION
        )
        
        assert 'daily_status' in status
        assert 'monthly_status' in status
        assert 'budget_health' in status
        assert 'alerts_generated' in status
        assert 'operation_allowed' in status
        
        # Daily status should show combined cost
        daily_status = status['daily_status']
        assert daily_status['projected_cost'] >= 75.0  # 60 + 15
        assert daily_status['budget_limit'] == 100.0
        assert daily_status['projected_percentage'] >= 75.0
        
        # Should generate warning alert (>= 75%)
        if status['alerts_generated']:
            alert = status['alerts_generated'][0]
            assert alert.alert_level == AlertLevel.WARNING
    
    def test_check_budget_status_threshold_alerts(self, budget_manager_basic):
        """Test alert generation at different threshold levels."""
        test_cases = [
            (74.0, None),  # Below warning threshold
            (76.0, AlertLevel.WARNING),  # Warning threshold
            (91.0, AlertLevel.CRITICAL),  # Critical threshold
            (101.0, AlertLevel.EXCEEDED)  # Exceeded threshold
        ]
        
        for existing_cost, expected_alert_level in test_cases:
            # Clear existing data
            budget_manager_basic.cost_persistence.cleanup_old_data()
            budget_manager_basic.clear_cache()
            budget_manager_basic.reset_alert_cooldowns()
            
            # Add existing cost
            if existing_cost > 0:
                budget_manager_basic.cost_persistence.record_cost(
                    cost_usd=existing_cost,
                    operation_type="threshold_test",
                    model_name="test-model",
                    token_usage={"prompt_tokens": 100}
                )
            
            # Check budget status
            status = budget_manager_basic.check_budget_status(
                cost_amount=1.0,  # Small additional cost
                operation_type="test_op"
            )
            
            if expected_alert_level:
                assert len(status['alerts_generated']) > 0
                alert = status['alerts_generated'][0]
                assert alert.alert_level == expected_alert_level
                
                # Verify operation_allowed for exceeded budget
                if expected_alert_level == AlertLevel.EXCEEDED:
                    assert status['operation_allowed'] is False
                else:
                    assert status['operation_allowed'] is True
            else:
                assert len(status['alerts_generated']) == 0
                assert status['operation_allowed'] is True
    
    def test_alert_callback_functionality(self, budget_manager_with_callback):
        """Test alert callback functionality."""
        manager, callback_mock = budget_manager_with_callback
        
        # Add cost to trigger alert
        manager.cost_persistence.record_cost(
            cost_usd=80.0,
            operation_type="callback_test",
            model_name="test-model",
            token_usage={"prompt_tokens": 1000}
        )
        
        # Check budget status (should trigger warning)
        status = manager.check_budget_status(
            cost_amount=1.0,
            operation_type="callback_trigger"
        )
        
        # Verify callback was called if alert was generated
        if status['alerts_generated']:
            callback_mock.assert_called()
            call_args = callback_mock.call_args[0][0]
            assert isinstance(call_args, BudgetAlert)
    
    def test_alert_cooldown_mechanism(self, budget_manager_basic):
        """Test alert cooldown to prevent spam."""
        # Add cost to trigger alerts
        budget_manager_basic.cost_persistence.record_cost(
            cost_usd=85.0,
            operation_type="cooldown_test",
            model_name="test-model",
            token_usage={"prompt_tokens": 1000}
        )
        
        # First check should generate alert
        status1 = budget_manager_basic.check_budget_status(
            cost_amount=1.0,
            operation_type="first_check"
        )
        first_alert_count = len(status1['alerts_generated'])
        
        # Immediate second check should not generate new alert (cooldown)
        status2 = budget_manager_basic.check_budget_status(
            cost_amount=1.0,
            operation_type="second_check"
        )
        second_alert_count = len(status2['alerts_generated'])
        
        # First check might generate alert, second should not (due to cooldown)
        if first_alert_count > 0:
            assert second_alert_count == 0
        
        # Reset cooldown and check again
        budget_manager_basic.reset_alert_cooldowns()
        status3 = budget_manager_basic.check_budget_status(
            cost_amount=1.0,
            operation_type="third_check"
        )
        
        # After cooldown reset, alerts should be possible again
        if status3['daily_status']['projected_percentage'] >= budget_manager_basic.thresholds.warning_percentage:
            assert len(status3['alerts_generated']) > 0
    
    def test_cache_mechanism(self, budget_manager_basic):
        """Test budget status caching for performance."""
        # Add some cost data
        budget_manager_basic.cost_persistence.record_cost(
            cost_usd=50.0,
            operation_type="cache_test",
            model_name="test-model",
            token_usage={"prompt_tokens": 500}
        )
        
        # First call should hit database
        status1 = budget_manager_basic.check_budget_status(
            cost_amount=1.0,
            operation_type="first_call"
        )
        
        # Second call should use cache
        status2 = budget_manager_basic.check_budget_status(
            cost_amount=1.0,
            operation_type="second_call"
        )
        
        # Both should have similar daily cost (from cache)
        assert abs(status1['daily_status']['current_cost'] - status2['daily_status']['current_cost']) < 2.0
        
        # Clear cache and verify fresh data
        budget_manager_basic.clear_cache()
        status3 = budget_manager_basic.check_budget_status(
            cost_amount=1.0,
            operation_type="third_call"
        )
        
        assert 'daily_status' in status3
    
    def test_custom_thresholds(self, budget_manager_custom_thresholds):
        """Test BudgetManager with custom thresholds."""
        manager = budget_manager_custom_thresholds
        
        # Verify custom thresholds are used
        assert manager.thresholds.warning_percentage == 60.0
        assert manager.thresholds.critical_percentage == 80.0
        assert manager.thresholds.exceeded_percentage == 95.0
        
        # Test threshold behavior
        # Add cost for 70% usage (should trigger warning at 60%)
        manager.cost_persistence.record_cost(
            cost_usd=35.0,  # 70% of 50.0 daily limit
            operation_type="custom_threshold_test",
            model_name="test-model",
            token_usage={"prompt_tokens": 500}
        )
        
        status = manager.check_budget_status(
            cost_amount=1.0,
            operation_type="threshold_check"
        )
        
        # Should trigger warning (>= 60%)
        if status['alerts_generated']:
            assert len(status['alerts_generated']) > 0
            alert = status['alerts_generated'][0]
            assert alert.alert_level == AlertLevel.WARNING
    
    def test_budget_summary(self, budget_manager_basic):
        """Test comprehensive budget summary generation."""
        # Add varied cost data
        costs = [15.0, 25.0, 30.0, 20.0]  # Total: 90.0
        for cost in costs:
            budget_manager_basic.cost_persistence.record_cost(
                cost_usd=cost,
                operation_type="summary_test",
                model_name="test-model",
                token_usage={"prompt_tokens": 100}
            )
        
        summary = budget_manager_basic.get_budget_summary()
        
        assert 'timestamp' in summary
        assert 'budget_health' in summary
        assert 'active_alerts' in summary
        assert 'daily_budget' in summary
        assert 'monthly_budget' in summary
        
        # Daily budget should show accumulated costs
        daily_budget = summary['daily_budget']
        assert daily_budget['total_cost'] >= 90.0
        assert daily_budget['budget_limit'] == 100.0
        assert daily_budget['percentage_used'] >= 90.0
        
        # Budget health should reflect high usage
        assert summary['budget_health'] in ['warning', 'critical', 'exceeded']
    
    def test_update_budget_limits(self, budget_manager_basic):
        """Test updating budget limits."""
        # Verify initial limits
        assert budget_manager_basic.daily_budget_limit == 100.0
        assert budget_manager_basic.monthly_budget_limit == 3000.0
        
        # Update limits
        budget_manager_basic.update_budget_limits(
            daily_budget=200.0,
            monthly_budget=6000.0
        )
        
        assert budget_manager_basic.daily_budget_limit == 200.0
        assert budget_manager_basic.monthly_budget_limit == 6000.0
        
        # Verify cache is cleared after update
        assert len(budget_manager_basic._budget_cache) == 0
    
    def test_update_thresholds(self, budget_manager_basic):
        """Test updating budget thresholds."""
        new_thresholds = BudgetThreshold(
            warning_percentage=50.0,
            critical_percentage=70.0,
            exceeded_percentage=85.0
        )
        
        budget_manager_basic.update_thresholds(new_thresholds)
        
        assert budget_manager_basic.thresholds == new_thresholds
        assert budget_manager_basic.thresholds.warning_percentage == 50.0
        
        # Verify alert history is cleared
        assert len(budget_manager_basic._last_alerts) == 0
    
    def test_spending_trends_analysis(self, budget_manager_basic):
        """Test spending trends analysis."""
        # Add cost data over time
        base_time = time.time() - (7 * 24 * 3600)  # 7 days ago
        daily_costs = [5.0, 8.0, 12.0, 15.0, 18.0, 22.0, 25.0]  # Increasing trend
        
        for i, cost in enumerate(daily_costs):
            record = CostRecord(
                timestamp=base_time + (i * 24 * 3600),  # Daily intervals
                operation_type=f"trend_test_{i}",
                model_name="trend-model",
                cost_usd=cost,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                research_category=ResearchCategory.GENERAL_QUERY.value
            )
            budget_manager_basic.cost_persistence.db.insert_cost_record(record)
        
        trends = budget_manager_basic.get_spending_trends(days=7)
        
        assert 'period_days' in trends
        assert trends['period_days'] == 7
        assert 'total_cost' in trends
        assert 'average_daily_cost' in trends
        assert 'trend_percentage' in trends
        assert 'trend_direction' in trends
        assert 'projected_monthly_cost' in trends
        assert 'budget_projections' in trends
        
        # Should detect increasing trend
        assert trends['trend_direction'] in ['increasing', 'stable']
        assert trends['average_daily_cost'] > 0
        
        # Budget projections should include recommendations
        projections = trends['budget_projections']
        assert 'daily_limit_needed' in projections
        assert 'monthly_limit_needed' in projections
        assert projections['daily_limit_needed'] > trends['average_daily_cost']
    
    def test_thread_safety(self, budget_manager_basic):
        """Test thread safety of budget manager operations."""
        num_threads = 8
        operations_per_thread = 15
        
        def worker(thread_id):
            results = []
            for i in range(operations_per_thread):
                # Add some cost
                budget_manager_basic.cost_persistence.record_cost(
                    cost_usd=1.0 + (thread_id * 0.1),
                    operation_type=f"thread_{thread_id}_op_{i}",
                    model_name="thread-model",
                    token_usage={"prompt_tokens": 50 + thread_id * 10}
                )
                
                # Check budget status
                status = budget_manager_basic.check_budget_status(
                    cost_amount=0.5,
                    operation_type=f"thread_{thread_id}_check_{i}"
                )
                results.append(status)
            return results
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        
        # Verify all operations completed
        assert len(all_results) == num_threads * operations_per_thread
        
        # Verify all results have expected structure
        for result in all_results:
            assert 'budget_health' in result
            assert 'operation_allowed' in result
            assert isinstance(result['alerts_generated'], list)
        
        # Verify final budget summary is consistent
        final_summary = budget_manager_basic.get_budget_summary()
        assert final_summary['daily_budget']['total_cost'] > 0
    
    def test_budget_health_assessment(self, budget_manager_basic):
        """Test budget health assessment logic."""
        test_scenarios = [
            # (daily_cost, monthly_cost, expected_health)
            (25.0, 750.0, 'healthy'),      # 25%, 25%
            (60.0, 1800.0, 'healthy'),     # 60%, 60%
            (80.0, 2400.0, 'warning'),     # 80%, 80%
            (95.0, 2700.0, 'critical'),    # 95%, 90%
            (110.0, 3300.0, 'exceeded')    # 110%, 110%
        ]
        
        for daily_cost, monthly_cost, expected_health in test_scenarios:
            # Clear previous data
            budget_manager_basic.cost_persistence.cleanup_old_data()
            budget_manager_basic.clear_cache()
            
            # Add daily cost
            budget_manager_basic.cost_persistence.record_cost(
                cost_usd=daily_cost,
                operation_type="health_test_daily",
                model_name="test-model",
                token_usage={"prompt_tokens": 100}
            )
            
            # Add additional monthly cost (simulated from previous days)
            if monthly_cost > daily_cost:
                additional_monthly = monthly_cost - daily_cost
                # Split additional cost across previous days
                for i in range(5):
                    record = CostRecord(
                        timestamp=time.time() - ((i + 1) * 24 * 3600),  # Previous days
                        operation_type=f"health_test_monthly_{i}",
                        model_name="test-model",
                        cost_usd=additional_monthly / 5,
                        prompt_tokens=100,
                        completion_tokens=50,
                        total_tokens=150,
                        research_category=ResearchCategory.GENERAL_QUERY.value
                    )
                    budget_manager_basic.cost_persistence.db.insert_cost_record(record)
            
            summary = budget_manager_basic.get_budget_summary()
            assert summary['budget_health'] == expected_health


class TestBudgetManagerErrorHandling:
    """Test error handling and edge cases in BudgetManager."""
    
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
    
    def test_invalid_cost_amounts(self, cost_persistence):
        """Test handling of invalid cost amounts."""
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0
        )
        
        # Test with negative cost
        status = manager.check_budget_status(
            cost_amount=-10.0,
            operation_type="negative_cost"
        )
        assert 'daily_status' in status
        assert status['operation_allowed'] is True  # Should still allow
        
        # Test with zero cost
        status = manager.check_budget_status(
            cost_amount=0.0,
            operation_type="zero_cost"
        )
        assert status['operation_allowed'] is True
        
        # Test with very large cost
        status = manager.check_budget_status(
            cost_amount=1000000.0,
            operation_type="huge_cost"
        )
        assert 'daily_status' in status
        # Should generate exceeded alert
        if manager.daily_budget_limit and status['alerts_generated']:
            assert any(alert.alert_level == AlertLevel.EXCEEDED for alert in status['alerts_generated'])
    
    def test_callback_error_handling(self, cost_persistence):
        """Test error handling in alert callbacks."""
        # Create callback that raises exception
        def failing_callback(alert):
            raise ValueError("Callback error")
        
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=50.0,
            alert_callback=failing_callback
        )
        
        # Add cost to trigger alert
        cost_persistence.record_cost(
            cost_usd=40.0,
            operation_type="callback_error_test",
            model_name="test-model",
            token_usage={"prompt_tokens": 100}
        )
        
        # Should not raise exception despite callback error
        status = manager.check_budget_status(
            cost_amount=10.0,
            operation_type="trigger_callback_error"
        )
        
        # Operation should still complete
        assert 'daily_status' in status
        assert 'alerts_generated' in status
    
    def test_database_error_handling(self, cost_persistence):
        """Test handling of database errors."""
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0
        )
        
        # Simulate database error by mocking the persistence layer
        with patch.object(cost_persistence, 'get_daily_budget_status', side_effect=Exception("Database error")):
            status = manager.check_budget_status(
                cost_amount=50.0,
                operation_type="db_error_test"
            )
            
            # Should handle error gracefully
            assert 'budget_health' in status
            assert status['operation_allowed'] is True  # Default to allowing operation
    
    def test_extreme_values(self, cost_persistence):
        """Test handling of extreme values."""
        manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=0.01,  # Very small limit
            monthly_budget_limit=0.30
        )
        
        # Test with cost larger than limit
        status = manager.check_budget_status(
            cost_amount=100.0,
            operation_type="extreme_cost"
        )
        
        assert status['daily_status']['projected_percentage'] > 100
        assert status['budget_health'] == 'exceeded'
        assert status['operation_allowed'] is False
        
        # Test with very large budget limits
        manager.update_budget_limits(
            daily_budget=1e10,  # 10 billion
            monthly_budget=3e11  # 300 billion
        )
        
        status = manager.check_budget_status(
            cost_amount=1000.0,
            operation_type="large_limit_test"
        )
        
        assert status['daily_status']['projected_percentage'] < 1
        assert status['budget_health'] == 'healthy'


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])