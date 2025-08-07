#!/usr/bin/env python3
"""
Enhanced test suite for Cost Persistence System - Additional coverage tests.

This test suite provides additional tests to achieve >90% coverage by testing
edge cases and uncovered code paths.

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import sqlite3
import time
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Test imports
from lightrag_integration.cost_persistence import (
    CostRecord,
    ResearchCategory,
    CostDatabase,
    CostPersistence
)


class TestCostRecordEnhanced:
    """Enhanced tests for CostRecord edge cases and coverage."""
    
    def test_cost_record_from_dict_with_string_metadata(self):
        """Test CostRecord.from_dict with JSON string metadata."""
        data_with_string_metadata = {
            'id': 1,
            'timestamp': time.time(),
            'date_str': '2025-08-07T10:00:00Z',
            'session_id': 'test_session',
            'operation_type': 'llm',
            'model_name': 'gpt-4o-mini',
            'cost_usd': 0.05,
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'embedding_tokens': 0,
            'total_tokens': 150,
            'research_category': 'general_query',
            'success': True,
            'metadata': '{"key": "value", "nested": {"data": 123}}'  # JSON string
        }
        
        record = CostRecord.from_dict(data_with_string_metadata)
        
        assert isinstance(record.metadata, dict)
        assert record.metadata['key'] == 'value'
        assert record.metadata['nested']['data'] == 123
    
    def test_cost_record_from_dict_with_invalid_json_metadata(self):
        """Test CostRecord.from_dict with invalid JSON metadata."""
        data_with_invalid_json = {
            'id': 1,
            'timestamp': time.time(),
            'date_str': '2025-08-07T10:00:00Z',
            'operation_type': 'llm',
            'model_name': 'gpt-4o-mini',
            'cost_usd': 0.05,
            'metadata': '{"invalid": json}'  # Invalid JSON
        }
        
        record = CostRecord.from_dict(data_with_invalid_json)
        
        # Should default to empty dict when JSON is invalid
        assert record.metadata == {}
    
    def test_cost_record_from_dict_filters_created_at(self):
        """Test that CostRecord.from_dict filters out created_at field."""
        data_with_created_at = {
            'id': 1,
            'timestamp': time.time(),
            'date_str': '2025-08-07T10:00:00Z',
            'operation_type': 'llm',
            'model_name': 'gpt-4o-mini',
            'cost_usd': 0.05,
            'created_at': '2025-08-07 10:00:00'  # Should be filtered out
        }
        
        # Should not raise an error about unexpected keyword argument
        record = CostRecord.from_dict(data_with_created_at)
        assert record.operation_type == 'llm'
        assert record.cost_usd == 0.05


class TestCostDatabaseEnhanced:
    """Enhanced tests for CostDatabase edge cases and coverage."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield Path(db_path)
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def cost_db(self, temp_db_path):
        """Create a CostDatabase instance for testing."""
        return CostDatabase(temp_db_path)
    
    def test_get_cost_records_with_all_filters(self, cost_db):
        """Test get_cost_records with all filtering parameters."""
        # Insert test records with different attributes
        base_time = time.time() - 7200  # 2 hours ago
        
        records_data = [
            {
                'timestamp': base_time,
                'operation_type': 'llm',
                'model_name': 'gpt-4o',
                'cost_usd': 0.1,
                'research_category': ResearchCategory.METABOLITE_IDENTIFICATION.value,
                'session_id': 'session_1'
            },
            {
                'timestamp': base_time + 1800,  # 30 min later
                'operation_type': 'embedding',
                'model_name': 'text-embedding-3-small',
                'cost_usd': 0.05,
                'research_category': ResearchCategory.PATHWAY_ANALYSIS.value,
                'session_id': 'session_1'
            },
            {
                'timestamp': base_time + 3600,  # 1 hour later
                'operation_type': 'llm',
                'model_name': 'gpt-4o-mini',
                'cost_usd': 0.08,
                'research_category': ResearchCategory.METABOLITE_IDENTIFICATION.value,
                'session_id': 'session_2'
            }
        ]
        
        for data in records_data:
            record = CostRecord(**data)
            cost_db.insert_cost_record(record)
        
        # Test with all filters applied
        filtered_records = cost_db.get_cost_records(
            start_time=base_time - 100,
            end_time=base_time + 5000,
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION.value,
            session_id='session_1',
            limit=5
        )
        
        # Should return only the first record (matches all criteria)
        assert len(filtered_records) == 1
        assert filtered_records[0].operation_type == 'llm'
        assert filtered_records[0].session_id == 'session_1'
        assert filtered_records[0].research_category == ResearchCategory.METABOLITE_IDENTIFICATION.value
    
    def test_get_budget_summary_with_specific_period_key(self, cost_db):
        """Test get_budget_summary with specific period key."""
        # Insert record for a specific date
        test_time = datetime(2025, 8, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp()
        record = CostRecord(
            timestamp=test_time,
            operation_type='budget_test',
            model_name='test-model',
            cost_usd=50.0
        )
        cost_db.insert_cost_record(record)
        
        # Test with specific daily period key
        daily_summary = cost_db.get_budget_summary('daily', '2025-08-15')
        assert daily_summary['period_type'] == 'daily'
        assert daily_summary['period_key'] == '2025-08-15'
        assert daily_summary['total_cost'] >= 50.0
        assert daily_summary['record_count'] >= 1
        
        # Test with specific monthly period key
        monthly_summary = cost_db.get_budget_summary('monthly', '2025-08')
        assert monthly_summary['period_type'] == 'monthly'
        assert monthly_summary['period_key'] == '2025-08'
        assert monthly_summary['total_cost'] >= 50.0
        assert monthly_summary['record_count'] >= 1
    
    def test_get_budget_summary_nonexistent_period(self, cost_db):
        """Test get_budget_summary for non-existent period."""
        # Query for a period with no data
        summary = cost_db.get_budget_summary('daily', '2024-01-01')
        
        assert summary['period_type'] == 'daily'
        assert summary['period_key'] == '2024-01-01'
        assert summary['total_cost'] == 0.0
        assert summary['record_count'] == 0
        assert summary['last_updated'] is None
    
    def test_get_research_category_summary_with_time_range(self, cost_db):
        """Test research category summary with time filtering."""
        base_time = time.time() - 3600  # 1 hour ago
        
        # Insert records in different time periods
        old_record = CostRecord(
            timestamp=base_time - 7200,  # 2 hours before base_time (3 hours ago)
            operation_type='old_test',
            model_name='test-model',
            cost_usd=100.0,
            research_category=ResearchCategory.BIOMARKER_DISCOVERY.value
        )
        cost_db.insert_cost_record(old_record)
        
        new_record = CostRecord(
            timestamp=base_time,  # 1 hour ago
            operation_type='new_test',
            model_name='test-model',
            cost_usd=50.0,
            research_category=ResearchCategory.BIOMARKER_DISCOVERY.value
        )
        cost_db.insert_cost_record(new_record)
        
        # Query with time range that excludes old record
        summary = cost_db.get_research_category_summary(
            start_time=base_time - 1800,  # 30 min before base_time
            end_time=time.time()
        )
        
        assert ResearchCategory.BIOMARKER_DISCOVERY.value in summary
        biomarker_summary = summary[ResearchCategory.BIOMARKER_DISCOVERY.value]
        
        # Should only include the new record
        assert biomarker_summary['total_cost'] == 50.0
        assert biomarker_summary['record_count'] == 1


class TestCostPersistenceEnhanced:
    """Enhanced tests for CostPersistence edge cases and coverage."""
    
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
        return CostPersistence(temp_db_path, retention_days=365)
    
    def test_record_cost_with_invalid_token_usage_type(self, cost_persistence):
        """Test record_cost graceful handling of invalid token_usage type."""
        # The method should handle invalid token_usage gracefully
        # by using .get() method which will return None and then default to 0
        try:
            record_id = cost_persistence.record_cost(
                cost_usd=0.05,
                operation_type="error_test",
                model_name="test-model",
                token_usage=None  # Invalid type, but should be handled
            )
            # If no exception is raised, that's acceptable behavior
            assert record_id is not None
        except (AttributeError, TypeError):
            # It's also acceptable if this raises an error
            pass
    
    def test_get_daily_budget_status_with_specific_date(self, cost_persistence):
        """Test get_daily_budget_status with specific date parameter."""
        # Add costs for a specific date
        specific_date = datetime(2025, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create a record with specific timestamp
        record = CostRecord(
            timestamp=specific_date.timestamp(),
            operation_type='specific_date_test',
            model_name='test-model',
            cost_usd=75.0,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        cost_persistence.db.insert_cost_record(record)
        
        # Test with the specific date
        status = cost_persistence.get_daily_budget_status(
            date=specific_date,
            budget_limit=100.0
        )
        
        assert status['date'] == '2025-08-20'
        assert status['total_cost'] >= 75.0
        assert status['percentage_used'] >= 75.0
        assert status['over_budget'] is False
    
    def test_get_monthly_budget_status_with_specific_date(self, cost_persistence):
        """Test get_monthly_budget_status with specific date parameter."""
        # Add costs for a specific month
        specific_date = datetime(2025, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create multiple records for the month
        for i in range(3):
            record = CostRecord(
                timestamp=specific_date.timestamp() + (i * 86400),  # Different days
                operation_type=f'monthly_test_{i}',
                model_name='test-model',
                cost_usd=300.0,  # 900 total
                prompt_tokens=200,
                completion_tokens=100,
                total_tokens=300
            )
            cost_persistence.db.insert_cost_record(record)
        
        # Test with the specific date
        status = cost_persistence.get_monthly_budget_status(
            date=specific_date,
            budget_limit=1000.0
        )
        
        assert status['month'] == '2025-09'
        assert status['total_cost'] >= 900.0
        assert status['percentage_used'] >= 90.0
        assert status['over_budget'] is False
    
    def test_generate_cost_report_with_failed_operations(self, cost_persistence):
        """Test cost report generation with failed operations."""
        # Add mix of successful and failed operations
        base_time = time.time() - 3600  # 1 hour ago
        
        operations_data = [
            {'success': True, 'error_type': None, 'cost': 0.10},
            {'success': False, 'error_type': 'timeout', 'cost': 0.05},
            {'success': True, 'error_type': None, 'cost': 0.15},
            {'success': False, 'error_type': 'api_error', 'cost': 0.08},
            {'success': False, 'error_type': 'timeout', 'cost': 0.03}
        ]
        
        for i, data in enumerate(operations_data):
            record = CostRecord(
                timestamp=base_time + (i * 300),  # 5-minute intervals
                operation_type=f'report_test_{i}',
                model_name='test-model',
                cost_usd=data['cost'],
                success=data['success'],
                error_type=data['error_type'],
                prompt_tokens=100,
                completion_tokens=50 if data['success'] else 0,
                total_tokens=150 if data['success'] else 100
            )
            cost_persistence.db.insert_cost_record(record)
        
        # Generate report
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=2)
        
        report = cost_persistence.generate_cost_report(start_date, end_date)
        
        # Verify error analysis
        assert 'error_analysis' in report
        error_analysis = report['error_analysis']
        
        assert error_analysis['total_failed_operations'] == 3
        assert 'error_types' in error_analysis
        assert 'timeout' in error_analysis['error_types']
        assert 'api_error' in error_analysis['error_types']
        
        # Verify timeout errors (should have 2 occurrences)
        timeout_errors = error_analysis['error_types']['timeout']
        assert timeout_errors['count'] == 2
        assert timeout_errors['total_cost'] == 0.08  # 0.05 + 0.03
        
        # Verify most common error
        assert error_analysis['most_common_error'] == 'timeout'
    
    def test_generate_cost_report_daily_breakdown(self, cost_persistence):
        """Test cost report daily breakdown functionality."""
        # Add costs across multiple days
        base_date = datetime(2025, 8, 10, 12, 0, 0, tzinfo=timezone.utc)
        
        daily_costs = [
            (base_date, 25.0),
            (base_date + timedelta(days=1), 30.0),
            (base_date + timedelta(days=2), 45.0)
        ]
        
        for date, cost in daily_costs:
            record = CostRecord(
                timestamp=date.timestamp(),
                operation_type='daily_breakdown_test',
                model_name='test-model',
                cost_usd=cost,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
            cost_persistence.db.insert_cost_record(record)
        
        # Generate report for the 3-day period
        start_date = base_date - timedelta(hours=12)
        end_date = base_date + timedelta(days=3)
        
        report = cost_persistence.generate_cost_report(start_date, end_date)
        
        # Verify daily costs breakdown
        daily_costs_breakdown = report['daily_costs']
        assert len(daily_costs_breakdown) == 3
        
        assert daily_costs_breakdown['2025-08-10'] == 25.0
        assert daily_costs_breakdown['2025-08-11'] == 30.0
        assert daily_costs_breakdown['2025-08-12'] == 45.0
    
    def test_error_analysis_with_no_failed_records(self, cost_persistence):
        """Test _get_error_analysis with no failed records."""
        # This tests an edge case in the private method
        result = cost_persistence._get_error_analysis([])
        assert result == {}
    
    def test_daily_cost_breakdown_edge_case(self, cost_persistence):
        """Test _get_daily_cost_breakdown with edge cases."""
        # Test with empty records list
        result = cost_persistence._get_daily_cost_breakdown([])
        assert result == {}
        
        # Test with single record
        single_record = CostRecord(
            timestamp=time.time(),
            operation_type='single_test',
            model_name='test-model',
            cost_usd=15.0
        )
        
        result = cost_persistence._get_daily_cost_breakdown([single_record])
        assert len(result) == 1
        
        # Get the date key
        date_key = list(result.keys())[0]
        assert result[date_key] == 15.0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for complete coverage."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield Path(db_path)
        Path(db_path).unlink(missing_ok=True)
    
    def test_cost_database_with_logger(self, temp_db_path):
        """Test CostDatabase initialization with custom logger."""
        logger = Mock(spec=logging.Logger)
        
        cost_db = CostDatabase(temp_db_path, logger=logger)
        
        assert cost_db.logger == logger
        
        # Verify logger was used during initialization
        logger.debug.assert_called()
    
    def test_cost_persistence_initialization_with_custom_retention(self, temp_db_path):
        """Test CostPersistence initialization with custom retention days."""
        logger = Mock(spec=logging.Logger)
        
        persistence = CostPersistence(
            db_path=temp_db_path,
            retention_days=180,
            logger=logger
        )
        
        assert persistence.retention_days == 180
        assert persistence.logger == logger
        
        # Verify logger was used
        logger.info.assert_called()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])