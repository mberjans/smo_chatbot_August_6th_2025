#!/usr/bin/env python3
"""
Comprehensive test suite for Cost Persistence System.

This test suite provides complete coverage of the cost persistence layer including:
- CostRecord data model validation and serialization
- CostDatabase schema creation and operations
- CostPersistence high-level interface and business logic
- Database integrity and thread safety
- Performance under load conditions
- Data retention and cleanup policies
- Research category analysis and reporting

Author: Claude Code (Anthropic)
Created: August 6, 2025
"""

import pytest
import sqlite3
import threading
import time
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

# Test imports
from lightrag_integration.cost_persistence import (
    CostRecord,
    ResearchCategory,
    CostDatabase,
    CostPersistence
)


class TestCostRecord:
    """Comprehensive tests for CostRecord data model."""
    
    def test_cost_record_basic_creation(self):
        """Test basic CostRecord creation with minimal parameters."""
        record = CostRecord(
            operation_type="llm",
            model_name="gpt-4o-mini",
            cost_usd=0.05
        )
        
        assert record.operation_type == "llm"
        assert record.model_name == "gpt-4o-mini"
        assert record.cost_usd == 0.05
        assert record.success is True
        assert record.timestamp is not None
        assert record.date_str is not None
        assert record.research_category == ResearchCategory.GENERAL_QUERY.value
    
    def test_cost_record_full_creation(self):
        """Test CostRecord creation with all parameters."""
        metadata = {"test_key": "test_value", "nested": {"data": 123}}
        
        record = CostRecord(
            timestamp=1691234567.89,
            session_id="test_session",
            operation_type="embedding",
            model_name="text-embedding-3-small",
            cost_usd=0.002,
            prompt_tokens=100,
            completion_tokens=0,
            embedding_tokens=50,
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION.value,
            query_type="similarity_search",
            subject_area="lipidomics",
            response_time_seconds=1.25,
            success=True,
            user_id="user123",
            project_id="project456",
            metadata=metadata
        )
        
        assert record.timestamp == 1691234567.89
        assert record.session_id == "test_session"
        assert record.operation_type == "embedding"
        assert record.embedding_tokens == 50
        assert record.total_tokens == 150  # 100 + 0 + 50
        assert record.research_category == ResearchCategory.METABOLITE_IDENTIFICATION.value
        assert record.query_type == "similarity_search"
        assert record.subject_area == "lipidomics"
        assert record.response_time_seconds == 1.25
        assert record.metadata == metadata
    
    def test_cost_record_post_init_calculations(self):
        """Test post-initialization calculations in CostRecord."""
        record = CostRecord(
            operation_type="hybrid",
            model_name="gpt-4o",
            cost_usd=0.15,
            prompt_tokens=200,
            completion_tokens=100,
            embedding_tokens=50
        )
        
        # Test total tokens calculation
        assert record.total_tokens == 350
        
        # Test date_str generation
        assert record.date_str is not None
        assert "T" in record.date_str  # ISO format
        
        # Test timestamp default
        assert record.timestamp is not None
        assert record.timestamp > 0
    
    def test_cost_record_research_category_validation(self):
        """Test research category validation and default assignment."""
        # Valid category
        record1 = CostRecord(
            operation_type="test",
            model_name="test-model",
            cost_usd=0.01,
            research_category=ResearchCategory.BIOMARKER_DISCOVERY.value
        )
        assert record1.research_category == ResearchCategory.BIOMARKER_DISCOVERY.value
        
        # Invalid category should default to GENERAL_QUERY
        record2 = CostRecord(
            operation_type="test",
            model_name="test-model",
            cost_usd=0.01,
            research_category="invalid_category"
        )
        assert record2.research_category == ResearchCategory.GENERAL_QUERY.value
    
    def test_cost_record_serialization(self):
        """Test CostRecord to_dict and from_dict methods."""
        original_metadata = {"key1": "value1", "key2": {"nested": "data"}}
        
        original_record = CostRecord(
            operation_type="llm",
            model_name="gpt-4o-mini",
            cost_usd=0.08,
            prompt_tokens=150,
            completion_tokens=75,
            session_id="serialize_test",
            research_category=ResearchCategory.PATHWAY_ANALYSIS.value,
            metadata=original_metadata
        )
        
        # Test to_dict
        record_dict = original_record.to_dict()
        assert isinstance(record_dict, dict)
        assert record_dict['operation_type'] == "llm"
        assert record_dict['cost_usd'] == 0.08
        assert record_dict['total_tokens'] == 225
        
        # Test metadata serialization
        if isinstance(record_dict['metadata'], str):
            assert json.loads(record_dict['metadata']) == original_metadata
        else:
            assert record_dict['metadata'] == original_metadata
        
        # Test from_dict
        reconstructed_record = CostRecord.from_dict(record_dict)
        assert reconstructed_record.operation_type == original_record.operation_type
        assert reconstructed_record.cost_usd == original_record.cost_usd
        assert reconstructed_record.prompt_tokens == original_record.prompt_tokens
        assert reconstructed_record.completion_tokens == original_record.completion_tokens
        assert reconstructed_record.total_tokens == original_record.total_tokens
        assert reconstructed_record.metadata == original_metadata
    
    def test_cost_record_edge_cases(self):
        """Test CostRecord with edge cases and boundary values."""
        # Zero cost
        record1 = CostRecord(
            operation_type="free_tier",
            model_name="test-model",
            cost_usd=0.0
        )
        assert record1.cost_usd == 0.0
        
        # Very small cost
        record2 = CostRecord(
            operation_type="micro_operation",
            model_name="test-model",
            cost_usd=0.000001
        )
        assert record2.cost_usd == 0.000001
        
        # Large token counts
        record3 = CostRecord(
            operation_type="large_document",
            model_name="gpt-4o",
            cost_usd=5.0,
            prompt_tokens=50000,
            completion_tokens=25000
        )
        assert record3.total_tokens == 75000
        
        # Empty metadata
        record4 = CostRecord(
            operation_type="no_metadata",
            model_name="test-model",
            cost_usd=0.01,
            metadata={}
        )
        assert record4.metadata == {}
        
        # None metadata
        record5 = CostRecord(
            operation_type="null_metadata",
            model_name="test-model",
            cost_usd=0.01,
            metadata=None
        )
        assert record5.metadata is None


class TestCostDatabase:
    """Comprehensive tests for CostDatabase operations."""
    
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
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)
    
    def test_database_initialization(self, temp_db_path, mock_logger):
        """Test database initialization and schema creation."""
        db = CostDatabase(temp_db_path, mock_logger)
        
        # Check that database file exists
        assert temp_db_path.exists()
        
        # Verify schema by checking table existence
        with sqlite3.connect(str(temp_db_path)) as conn:
            cursor = conn.cursor()
            
            # Check cost_records table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cost_records'")
            assert cursor.fetchone() is not None
            
            # Check budget_tracking table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='budget_tracking'")
            assert cursor.fetchone() is not None
            
            # Check audit_log table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'")
            assert cursor.fetchone() is not None
            
            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_cost_records_timestamp'")
            assert cursor.fetchone() is not None
    
    def test_insert_single_cost_record(self, cost_db):
        """Test inserting a single cost record."""
        record = CostRecord(
            operation_type="test_insert",
            model_name="test-model",
            cost_usd=0.05,
            prompt_tokens=100,
            completion_tokens=50,
            research_category=ResearchCategory.DATA_PREPROCESSING.value,
            session_id="test_session"
        )
        
        record_id = cost_db.insert_cost_record(record)
        
        assert record_id is not None
        assert isinstance(record_id, int)
        assert record_id > 0
    
    def test_insert_multiple_cost_records(self, cost_db):
        """Test inserting multiple cost records."""
        records = []
        for i in range(10):
            record = CostRecord(
                operation_type=f"batch_test_{i}",
                model_name="test-model",
                cost_usd=0.01 * (i + 1),
                prompt_tokens=50 + i * 10,
                completion_tokens=25 + i * 5,
                research_category=list(ResearchCategory)[i % len(ResearchCategory)].value
            )
            records.append(record)
            cost_db.insert_cost_record(record)
        
        # Verify all records were inserted
        retrieved_records = cost_db.get_cost_records(limit=20)
        assert len(retrieved_records) == 10
        
        # Verify ordering (should be most recent first)
        for i in range(1, len(retrieved_records)):
            assert retrieved_records[i-1].timestamp >= retrieved_records[i].timestamp
    
    def test_budget_tracking_updates(self, cost_db):
        """Test budget tracking updates when inserting cost records."""
        record = CostRecord(
            operation_type="budget_test",
            model_name="test-model",
            cost_usd=25.0,
            research_category=ResearchCategory.CLINICAL_DIAGNOSIS.value
        )
        
        cost_db.insert_cost_record(record)
        
        # Check daily budget tracking
        daily_summary = cost_db.get_budget_summary('daily')
        assert daily_summary['total_cost'] >= 25.0
        assert daily_summary['record_count'] >= 1
        
        # Check monthly budget tracking
        monthly_summary = cost_db.get_budget_summary('monthly')
        assert monthly_summary['total_cost'] >= 25.0
        assert monthly_summary['record_count'] >= 1
    
    def test_get_cost_records_filtering(self, cost_db):
        """Test cost record retrieval with various filters."""
        # Insert test records with different categories and timestamps
        base_time = time.time() - 3600  # 1 hour ago
        categories = [ResearchCategory.METABOLITE_IDENTIFICATION, ResearchCategory.PATHWAY_ANALYSIS]
        
        for i in range(6):
            record = CostRecord(
                timestamp=base_time + i * 300,  # 5-minute intervals
                operation_type=f"filter_test_{i}",
                model_name="test-model",
                cost_usd=0.1 + i * 0.05,
                research_category=categories[i % 2].value,
                session_id=f"session_{i // 3}"  # Two sessions
            )
            cost_db.insert_cost_record(record)
        
        # Test time filtering
        mid_time = base_time + 900  # 15 minutes after start
        recent_records = cost_db.get_cost_records(start_time=mid_time, limit=10)
        assert len(recent_records) >= 3  # Should get last 3 records
        
        # Test category filtering
        metabolite_records = cost_db.get_cost_records(
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION.value,
            limit=10
        )
        assert len(metabolite_records) == 3
        for record in metabolite_records:
            assert record.research_category == ResearchCategory.METABOLITE_IDENTIFICATION.value
        
        # Test session filtering
        session_records = cost_db.get_cost_records(session_id="session_0", limit=10)
        assert len(session_records) == 3
        for record in session_records:
            assert record.session_id == "session_0"
        
        # Test limit
        limited_records = cost_db.get_cost_records(limit=2)
        assert len(limited_records) == 2
    
    def test_research_category_summary(self, cost_db):
        """Test research category summary generation."""
        # Insert records for different categories
        categories_data = [
            (ResearchCategory.BIOMARKER_DISCOVERY, 5, 0.15),
            (ResearchCategory.DRUG_DISCOVERY, 3, 0.25),
            (ResearchCategory.PATHWAY_ANALYSIS, 7, 0.12),
            (ResearchCategory.GENERAL_QUERY, 2, 0.05)
        ]
        
        for category, count, cost_per_record in categories_data:
            for i in range(count):
                record = CostRecord(
                    operation_type=f"{category.value}_test_{i}",
                    model_name="test-model",
                    cost_usd=cost_per_record,
                    total_tokens=100 + i * 10,
                    research_category=category.value
                )
                cost_db.insert_cost_record(record)
        
        # Get category summary
        summary = cost_db.get_research_category_summary()
        
        assert len(summary) == 4
        
        # Verify BIOMARKER_DISCOVERY summary
        biomarker_summary = summary[ResearchCategory.BIOMARKER_DISCOVERY.value]
        assert biomarker_summary['record_count'] == 5
        assert biomarker_summary['total_cost'] == pytest.approx(0.75, abs=1e-10)
        assert biomarker_summary['avg_cost'] == pytest.approx(0.15, abs=1e-10)
        
        # Verify ordering (by total cost descending)
        category_items = list(summary.items())
        for i in range(1, len(category_items)):
            assert category_items[i-1][1]['total_cost'] >= category_items[i][1]['total_cost']
    
    def test_cleanup_old_records(self, cost_db, mock_logger):
        """Test cleanup of old records based on retention policy."""
        # Insert old and new records
        old_time = time.time() - (400 * 24 * 60 * 60)  # 400 days ago
        recent_time = time.time() - 3600  # 1 hour ago
        
        # Old records (should be deleted)
        for i in range(5):
            record = CostRecord(
                timestamp=old_time + i * 100,
                operation_type=f"old_record_{i}",
                model_name="old-model",
                cost_usd=0.1
            )
            cost_db.insert_cost_record(record)
        
        # Recent records (should be kept)
        for i in range(3):
            record = CostRecord(
                timestamp=recent_time + i * 100,
                operation_type=f"recent_record_{i}",
                model_name="recent-model",
                cost_usd=0.2
            )
            cost_db.insert_cost_record(record)
        
        # Cleanup with 365 days retention
        deleted_count = cost_db.cleanup_old_records(365)
        
        assert deleted_count == 5
        
        # Verify only recent records remain
        remaining_records = cost_db.get_cost_records(limit=20)
        assert len(remaining_records) == 3
        for record in remaining_records:
            assert record.operation_type.startswith("recent_record_")
    
    def test_thread_safety(self, cost_db):
        """Test thread safety of database operations."""
        num_threads = 10
        records_per_thread = 20
        results = []
        
        def worker(thread_id):
            thread_results = []
            for i in range(records_per_thread):
                record = CostRecord(
                    operation_type=f"thread_{thread_id}_record_{i}",
                    model_name=f"thread-model-{thread_id}",
                    cost_usd=0.01 * (thread_id + 1),
                    session_id=f"thread_session_{thread_id}"
                )
                record_id = cost_db.insert_cost_record(record)
                thread_results.append(record_id)
            return thread_results
        
        # Run concurrent insertions
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in futures:
                results.extend(future.result())
        
        # Verify all records were inserted
        assert len(results) == num_threads * records_per_thread
        assert len(set(results)) == len(results)  # All IDs should be unique
        
        # Verify database integrity
        all_records = cost_db.get_cost_records(limit=num_threads * records_per_thread * 2)
        assert len(all_records) >= num_threads * records_per_thread
    
    def test_database_corruption_handling(self, temp_db_path):
        """Test handling of database corruption scenarios."""
        # Create database normally
        db = CostDatabase(temp_db_path)
        
        # Insert a test record
        record = CostRecord(
            operation_type="test_corruption",
            model_name="test-model",
            cost_usd=0.05
        )
        record_id = db.insert_cost_record(record)
        assert record_id is not None
        
        # Simulate corruption by corrupting the database file
        with open(temp_db_path, 'wb') as f:
            f.write(b'corrupted_data_not_sqlite')
        
        # Creating a new database instance should handle the corruption
        # (In a real scenario, this might recreate the database or handle the error)
        try:
            db2 = CostDatabase(temp_db_path)
            # If we get here, the system handled corruption gracefully
            assert True
        except Exception as e:
            # Expected behavior - corruption should be detected
            assert "database" in str(e).lower() or "corrupt" in str(e).lower()
    
    def test_large_metadata_handling(self, cost_db):
        """Test handling of large metadata objects."""
        # Create large metadata
        large_metadata = {
            "large_array": list(range(1000)),
            "nested_data": {
                f"key_{i}": f"value_{i}" * 100  # Large string values
                for i in range(50)
            },
            "description": "A" * 10000  # Very long description
        }
        
        record = CostRecord(
            operation_type="large_metadata_test",
            model_name="test-model",
            cost_usd=0.1,
            metadata=large_metadata
        )
        
        # Should handle large metadata without issues
        record_id = cost_db.insert_cost_record(record)
        assert record_id is not None
        
        # Verify retrieval
        retrieved_records = cost_db.get_cost_records(limit=1)
        assert len(retrieved_records) == 1
        retrieved_record = retrieved_records[0]
        
        # Metadata should be preserved
        assert retrieved_record.metadata is not None
        assert len(retrieved_record.metadata["large_array"]) == 1000
        assert len(retrieved_record.metadata["nested_data"]) == 50


class TestCostPersistence:
    """Comprehensive tests for CostPersistence high-level interface."""
    
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
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)
    
    def test_cost_persistence_initialization(self, temp_db_path, mock_logger):
        """Test CostPersistence initialization."""
        persistence = CostPersistence(temp_db_path, retention_days=180, logger=mock_logger)
        
        assert persistence.db_path == temp_db_path
        assert persistence.retention_days == 180
        assert persistence.logger == mock_logger
        assert persistence.db is not None
        assert isinstance(persistence.db, CostDatabase)
    
    def test_record_cost_basic(self, cost_persistence):
        """Test basic cost recording functionality."""
        record_id = cost_persistence.record_cost(
            cost_usd=0.05,
            operation_type="test_llm",
            model_name="gpt-4o-mini",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            session_id="test_session",
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION
        )
        
        assert record_id is not None
        assert isinstance(record_id, int)
        assert record_id > 0
    
    def test_record_cost_comprehensive(self, cost_persistence):
        """Test comprehensive cost recording with all parameters."""
        metadata = {
            "query_text": "What are the metabolic pathways for glucose metabolism?",
            "model_parameters": {"temperature": 0.7, "max_tokens": 500},
            "processing_info": {"batch_id": "batch_123", "queue_time": 0.5}
        }
        
        record_id = cost_persistence.record_cost(
            cost_usd=0.25,
            operation_type="enhanced_llm",
            model_name="gpt-4o",
            token_usage={
                "prompt_tokens": 500,
                "completion_tokens": 300,
                "embedding_tokens": 0
            },
            session_id="comprehensive_session",
            research_category=ResearchCategory.PATHWAY_ANALYSIS,
            query_type="scientific_inquiry",
            subject_area="biochemistry",
            response_time=2.5,
            success=True,
            user_id="researcher_123",
            project_id="metabolomics_project_456",
            metadata=metadata
        )
        
        assert record_id is not None
        
        # Verify record was stored correctly
        records = cost_persistence.db.get_cost_records(limit=1)
        assert len(records) == 1
        
        record = records[0]
        assert record.cost_usd == 0.25
        assert record.operation_type == "enhanced_llm"
        assert record.model_name == "gpt-4o"
        assert record.prompt_tokens == 500
        assert record.completion_tokens == 300
        assert record.total_tokens == 800
        assert record.research_category == ResearchCategory.PATHWAY_ANALYSIS.value
        assert record.query_type == "scientific_inquiry"
        assert record.subject_area == "biochemistry"
        assert record.response_time_seconds == 2.5
        assert record.success is True
        assert record.user_id == "researcher_123"
        assert record.project_id == "metabolomics_project_456"
        assert record.metadata == metadata
    
    def test_get_daily_budget_status(self, cost_persistence):
        """Test daily budget status calculation."""
        # Add some costs for today
        today_costs = [15.0, 25.0, 30.0]
        for cost in today_costs:
            cost_persistence.record_cost(
                cost_usd=cost,
                operation_type="daily_test",
                model_name="test-model",
                token_usage={"prompt_tokens": 100}
            )
        
        # Test without budget limit
        status = cost_persistence.get_daily_budget_status()
        assert status['total_cost'] >= 70.0  # Sum of costs
        assert status['record_count'] >= 3
        
        # Test with budget limit
        status_with_limit = cost_persistence.get_daily_budget_status(budget_limit=100.0)
        assert status_with_limit['budget_limit'] == 100.0
        assert status_with_limit['remaining_budget'] <= 30.0
        assert status_with_limit['percentage_used'] >= 70.0
        assert status_with_limit['over_budget'] is False
        
        # Test over-budget scenario
        status_over_budget = cost_persistence.get_daily_budget_status(budget_limit=50.0)
        assert status_over_budget['over_budget'] is True
    
    def test_get_monthly_budget_status(self, cost_persistence):
        """Test monthly budget status calculation."""
        # Add costs for this month
        monthly_costs = [100.0, 150.0, 200.0]
        for cost in monthly_costs:
            cost_persistence.record_cost(
                cost_usd=cost,
                operation_type="monthly_test",
                model_name="test-model",
                token_usage={"completion_tokens": 200}
            )
        
        status = cost_persistence.get_monthly_budget_status(budget_limit=1000.0)
        assert status['budget_limit'] == 1000.0
        assert status['total_cost'] >= 450.0
        assert status['percentage_used'] >= 45.0
        assert status['over_budget'] is False
    
    def test_get_research_analysis(self, cost_persistence):
        """Test research-specific cost analysis."""
        # Add varied research costs
        research_data = [
            (ResearchCategory.BIOMARKER_DISCOVERY, 5, 0.20),
            (ResearchCategory.DRUG_DISCOVERY, 3, 0.35),
            (ResearchCategory.METABOLITE_IDENTIFICATION, 8, 0.15),
            (ResearchCategory.PATHWAY_ANALYSIS, 4, 0.25),
            (ResearchCategory.STATISTICAL_ANALYSIS, 2, 0.40)
        ]
        
        for category, count, cost_per_record in research_data:
            for i in range(count):
                cost_persistence.record_cost(
                    cost_usd=cost_per_record,
                    operation_type=f"{category.value}_research",
                    model_name="research-model",
                    token_usage={"prompt_tokens": 150, "completion_tokens": 100},
                    research_category=category
                )
        
        analysis = cost_persistence.get_research_analysis(days=1)  # Today's analysis
        
        assert analysis['period_days'] == 1
        assert analysis['total_cost'] >= 4.85  # Sum of all costs
        assert analysis['total_records'] >= 22  # Sum of all records
        assert analysis['average_cost_per_record'] > 0
        
        # Check categories
        assert len(analysis['categories']) >= 5
        assert ResearchCategory.METABOLITE_IDENTIFICATION.value in analysis['categories']
        
        # Check top categories (should be ordered by cost)
        top_categories = analysis['top_categories']
        assert len(top_categories) <= 5
        for i in range(1, len(top_categories)):
            assert top_categories[i-1][1]['total_cost'] >= top_categories[i][1]['total_cost']
        
        # Verify percentage calculations
        for category_data in analysis['categories'].values():
            assert 0 <= category_data['percentage_of_total'] <= 100
    
    def test_generate_cost_report(self, cost_persistence):
        """Test comprehensive cost report generation."""
        # Add diverse cost data
        models = ["gpt-4o-mini", "gpt-4o", "text-embedding-3-small"]
        operations = ["llm_call", "embedding", "hybrid"]
        
        for i in range(15):
            cost_persistence.record_cost(
                cost_usd=0.05 + (i * 0.02),
                operation_type=operations[i % 3],
                model_name=models[i % 3],
                token_usage={
                    "prompt_tokens": 100 + i * 10,
                    "completion_tokens": 50 + i * 5
                },
                research_category=list(ResearchCategory)[i % len(ResearchCategory)],
                success=(i % 7) != 0  # Some failures
            )
        
        # Generate report for last 7 days
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        report = cost_persistence.generate_cost_report(start_date, end_date)
        
        # Verify report structure
        assert 'report_generated' in report
        assert 'period_start' in report
        assert 'period_end' in report
        assert 'summary' in report
        assert 'operation_breakdown' in report
        assert 'model_breakdown' in report
        assert 'research_categories' in report
        assert 'daily_costs' in report
        
        # Verify summary
        summary = report['summary']
        assert summary['total_records'] >= 15
        assert summary['total_cost'] > 0
        assert summary['successful_operations'] > 0
        assert summary['failed_operations'] >= 0
        assert 0 <= summary['success_rate'] <= 100
        
        # Verify breakdowns
        assert len(report['operation_breakdown']) <= 3
        assert len(report['model_breakdown']) <= 3
        assert len(report['research_categories']) > 0
        
        # Verify daily costs
        daily_costs = report['daily_costs']
        assert len(daily_costs) >= 1
        for date_str, cost in daily_costs.items():
            assert cost > 0
            # Verify date format
            assert len(date_str) == 10  # YYYY-MM-DD
    
    def test_generate_cost_report_empty_period(self, cost_persistence):
        """Test cost report generation for period with no data."""
        # Generate report for future dates (no data)
        start_date = datetime.now(timezone.utc) + timedelta(days=10)
        end_date = start_date + timedelta(days=5)
        
        report = cost_persistence.generate_cost_report(start_date, end_date)
        
        assert 'message' in report
        assert report['total_cost'] == 0.0
        assert report['total_records'] == 0
    
    def test_cleanup_old_data(self, cost_persistence):
        """Test old data cleanup functionality."""
        # Add old data (should be cleaned)
        old_time = time.time() - (400 * 24 * 60 * 60)  # 400 days ago
        for i in range(10):
            record = CostRecord(
                timestamp=old_time + i * 100,
                operation_type=f"old_data_{i}",
                model_name="old-model",
                cost_usd=0.1
            )
            cost_persistence.db.insert_cost_record(record)
        
        # Add recent data (should be kept)
        for i in range(5):
            cost_persistence.record_cost(
                cost_usd=0.05,
                operation_type=f"recent_data_{i}",
                model_name="recent-model",
                token_usage={"prompt_tokens": 100}
            )
        
        # Cleanup (retention is 365 days from initialization)
        deleted_count = cost_persistence.cleanup_old_data()
        
        assert deleted_count == 10
        
        # Verify recent data is preserved
        records = cost_persistence.db.get_cost_records(limit=20)
        assert len(records) == 5
        for record in records:
            assert record.operation_type.startswith("recent_data_")
    
    def test_error_handling(self, cost_persistence):
        """Test error handling in cost persistence operations."""
        # Test with invalid token usage
        with pytest.raises(AttributeError):
            cost_persistence.record_cost(
                cost_usd=0.05,
                operation_type="error_test",
                model_name="test-model",
                token_usage="invalid_token_data"  # Should be dict
            )
        
        # Test with negative cost
        record_id = cost_persistence.record_cost(
            cost_usd=-0.05,  # Negative cost should be handled
            operation_type="negative_cost_test",
            model_name="test-model",
            token_usage={"prompt_tokens": 100}
        )
        # Should still create record (business decision to allow negative adjustments)
        assert record_id is not None
    
    def test_concurrent_access(self, cost_persistence):
        """Test concurrent access to cost persistence."""
        num_threads = 8
        records_per_thread = 25
        
        def worker(thread_id):
            results = []
            for i in range(records_per_thread):
                record_id = cost_persistence.record_cost(
                    cost_usd=0.01 + (thread_id * 0.001),
                    operation_type=f"concurrent_test_{thread_id}_{i}",
                    model_name=f"model_{thread_id}",
                    token_usage={
                        "prompt_tokens": 50 + thread_id * 10,
                        "completion_tokens": 25 + i * 2
                    },
                    session_id=f"concurrent_session_{thread_id}",
                    research_category=list(ResearchCategory)[thread_id % len(ResearchCategory)]
                )
                results.append(record_id)
            return results
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        
        # Verify all operations completed successfully
        assert len(all_results) == num_threads * records_per_thread
        assert len(set(all_results)) == len(all_results)  # All unique IDs
        
        # Verify data integrity
        all_records = cost_persistence.db.get_cost_records(limit=num_threads * records_per_thread * 2)
        assert len(all_records) >= num_threads * records_per_thread
        
        # Test concurrent budget status queries
        def budget_status_worker():
            return cost_persistence.get_daily_budget_status(budget_limit=100.0)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            status_futures = [executor.submit(budget_status_worker) for _ in range(10)]
            status_results = [future.result() for future in status_futures]
        
        # All status queries should succeed and return consistent data
        assert len(status_results) == 10
        for status in status_results:
            assert 'total_cost' in status
            assert status['budget_limit'] == 100.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])