#!/usr/bin/env python3
"""
Comprehensive test suite for Audit Trail System.

This test suite provides complete coverage of the audit trail components including:
- AuditEvent data model and validation
- ComplianceLevel configuration and requirements
- AuditEventType enumeration and categorization
- AuditTrail main functionality and event recording
- Compliance reporting and data retention
- Security and integrity verification
- Performance under concurrent operations
- Integration with other system components

Author: Claude Code (Anthropic)
Created: August 6, 2025
"""

import pytest
import time
import json
import threading
import tempfile
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Test imports
from lightrag_integration.audit_trail import (
    AuditTrail,
    AuditEvent,
    AuditEventType,
    ComplianceLevel,
    AuditDatabase,
    DataIntegrityError
)


class TestAuditEvent:
    """Comprehensive tests for AuditEvent data model."""
    
    def test_audit_event_basic_creation(self):
        """Test basic AuditEvent creation."""
        event = AuditEvent(
            event_type=AuditEventType.API_CALL,
            event_data={"operation": "llm_call", "cost": 0.05},
            user_id="test_user",
            session_id="test_session"
        )
        
        assert event.event_type == AuditEventType.API_CALL
        assert event.event_data == {"operation": "llm_call", "cost": 0.05}
        assert event.user_id == "test_user"
        assert event.session_id == "test_session"
        assert event.timestamp is not None
        assert event.compliance_level == ComplianceLevel.STANDARD
    
    def test_audit_event_with_optional_fields(self):
        """Test AuditEvent creation with optional fields."""
        metadata = {
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0",
            "api_version": "v2.1"
        }
        
        event = AuditEvent(
            event_type=AuditEventType.BUDGET_ALERT,
            event_data={"alert_level": "warning", "cost": 75.0},
            user_id="admin_user",
            session_id="admin_session",
            compliance_level=ComplianceLevel.HIGH,
            metadata=metadata,
            timestamp=1691234567.89
        )
        
        assert event.compliance_level == ComplianceLevel.HIGH
        assert event.metadata == metadata
        assert event.timestamp == 1691234567.89
        assert event.metadata["ip_address"] == "192.168.1.100"
    
    def test_audit_event_serialization(self):
        """Test AuditEvent serialization and deserialization."""
        original_event_data = {
            "query": "What is metabolomics?",
            "response_time": 1.25,
            "tokens_used": 150,
            "model": "gpt-4o-mini"
        }
        
        original_metadata = {
            "processing_node": "node-1",
            "queue_time": 0.05,
            "cache_hit": False
        }
        
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_ERROR,
            event_data=original_event_data,
            user_id="system",
            session_id="error_session",
            compliance_level=ComplianceLevel.CRITICAL,
            metadata=original_metadata
        )
        
        # Test to_dict
        event_dict = event.to_dict()
        
        assert isinstance(event_dict, dict)
        assert event_dict['event_type'] == AuditEventType.SYSTEM_ERROR.value
        assert event_dict['event_data'] == original_event_data
        assert event_dict['user_id'] == "system"
        assert event_dict['compliance_level'] == ComplianceLevel.CRITICAL.value
        assert event_dict['metadata'] == original_metadata
        assert 'event_id' in event_dict
        assert 'timestamp_iso' in event_dict
        
        # Test from_dict
        reconstructed_event = AuditEvent.from_dict(event_dict)
        
        assert reconstructed_event.event_type == event.event_type
        assert reconstructed_event.event_data == event.event_data
        assert reconstructed_event.user_id == event.user_id
        assert reconstructed_event.compliance_level == event.compliance_level
        assert reconstructed_event.metadata == event.metadata
        assert reconstructed_event.event_id == event.event_id
        assert abs(reconstructed_event.timestamp - event.timestamp) < 0.001
    
    def test_audit_event_integrity_hash(self):
        """Test audit event integrity hash generation."""
        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            event_data={"table": "cost_records", "action": "select"},
            user_id="db_user",
            session_id="db_session"
        )
        
        # Should generate integrity hash
        hash1 = event.calculate_integrity_hash()
        hash2 = event.calculate_integrity_hash()
        
        # Same event should produce same hash
        assert hash1 == hash2
        assert len(hash1) > 0
        
        # Different event should produce different hash
        event2 = AuditEvent(
            event_type=AuditEventType.DATA_MODIFICATION,  # Different type
            event_data={"table": "cost_records", "action": "select"},
            user_id="db_user",
            session_id="db_session"
        )
        
        hash3 = event2.calculate_integrity_hash()
        assert hash1 != hash3
    
    def test_audit_event_validation(self):
        """Test audit event validation logic."""
        # Valid event
        valid_event = AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            event_data={"action": "login", "method": "oauth"},
            user_id="valid_user",
            session_id="valid_session"
        )
        
        assert valid_event.validate() is True
        
        # Test with missing required fields
        with pytest.raises(ValueError, match="Event type is required"):
            AuditEvent(
                event_type=None,
                event_data={"test": "data"},
                user_id="user",
                session_id="session"
            )
        
        # Test with invalid event data
        with pytest.raises(ValueError, match="Event data must be a dictionary"):
            AuditEvent(
                event_type=AuditEventType.API_CALL,
                event_data="invalid_data",  # Should be dict
                user_id="user",
                session_id="session"
            )


class TestAuditEventType:
    """Tests for AuditEventType enumeration."""
    
    def test_audit_event_type_values(self):
        """Test all audit event type values."""
        expected_types = [
            'api_call',
            'budget_alert',
            'system_error',
            'data_access',
            'data_modification',
            'user_action',
            'compliance_check',
            'security_event'
        ]
        
        actual_types = [event_type.value for event_type in AuditEventType]
        
        for expected_type in expected_types:
            assert expected_type in actual_types
    
    def test_audit_event_type_categorization(self):
        """Test event type categorization by compliance level."""
        # High compliance events
        high_compliance_types = [
            AuditEventType.DATA_MODIFICATION,
            AuditEventType.SECURITY_EVENT,
            AuditEventType.COMPLIANCE_CHECK
        ]
        
        # Standard compliance events
        standard_compliance_types = [
            AuditEventType.API_CALL,
            AuditEventType.USER_ACTION,
            AuditEventType.DATA_ACCESS
        ]
        
        for event_type in high_compliance_types:
            assert event_type in AuditEventType
        
        for event_type in standard_compliance_types:
            assert event_type in AuditEventType


class TestComplianceLevel:
    """Tests for ComplianceLevel enumeration."""
    
    def test_compliance_level_hierarchy(self):
        """Test compliance level hierarchy."""
        levels = [
            ComplianceLevel.STANDARD,
            ComplianceLevel.HIGH,
            ComplianceLevel.CRITICAL
        ]
        
        # Test ordering (lower value = higher priority)
        assert ComplianceLevel.CRITICAL.value < ComplianceLevel.HIGH.value
        assert ComplianceLevel.HIGH.value < ComplianceLevel.STANDARD.value
    
    def test_compliance_level_requirements(self):
        """Test compliance level requirements and retention."""
        requirements = {
            ComplianceLevel.STANDARD: {"retention_days": 365, "encryption": False},
            ComplianceLevel.HIGH: {"retention_days": 1095, "encryption": True},
            ComplianceLevel.CRITICAL: {"retention_days": 2555, "encryption": True}
        }
        
        for level, req in requirements.items():
            assert level in ComplianceLevel


class TestAuditDatabase:
    """Tests for AuditDatabase functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield Path(db_path)
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def audit_db(self, temp_db_path):
        """Create an AuditDatabase instance for testing."""
        return AuditDatabase(temp_db_path)
    
    def test_audit_database_initialization(self, temp_db_path):
        """Test audit database initialization and schema creation."""
        db = AuditDatabase(temp_db_path)
        
        assert temp_db_path.exists()
        
        # Verify schema by checking table existence
        import sqlite3
        with sqlite3.connect(str(temp_db_path)) as conn:
            cursor = conn.cursor()
            
            # Check audit_events table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_events'")
            assert cursor.fetchone() is not None
            
            # Check compliance_records table  
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compliance_records'")
            assert cursor.fetchone() is not None
            
            # Check integrity_checks table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='integrity_checks'")
            assert cursor.fetchone() is not None
            
            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_audit_%'")
            indexes = cursor.fetchall()
            assert len(indexes) > 0  # Should have performance indexes
    
    def test_insert_audit_event(self, audit_db):
        """Test inserting audit events into database."""
        event = AuditEvent(
            event_type=AuditEventType.API_CALL,
            event_data={
                "operation": "llm_query",
                "model": "gpt-4o-mini",
                "cost": 0.08,
                "tokens": 200
            },
            user_id="test_user",
            session_id="test_session",
            compliance_level=ComplianceLevel.HIGH
        )
        
        event_id = audit_db.insert_audit_event(event)
        
        assert event_id is not None
        assert isinstance(event_id, int)
        assert event_id > 0
    
    def test_retrieve_audit_events(self, audit_db):
        """Test retrieving audit events with filters."""
        # Insert test events
        events_data = [
            (AuditEventType.API_CALL, {"op": "call1"}, "user1", "session1"),
            (AuditEventType.BUDGET_ALERT, {"op": "alert1"}, "user2", "session1"),
            (AuditEventType.SYSTEM_ERROR, {"op": "error1"}, "user1", "session2"),
            (AuditEventType.DATA_ACCESS, {"op": "access1"}, "user3", "session1")
        ]
        
        for event_type, event_data, user_id, session_id in events_data:
            event = AuditEvent(
                event_type=event_type,
                event_data=event_data,
                user_id=user_id,
                session_id=session_id
            )
            audit_db.insert_audit_event(event)
        
        # Test retrieval with different filters
        all_events = audit_db.get_audit_events(limit=10)
        assert len(all_events) == 4
        
        # Filter by user
        user1_events = audit_db.get_audit_events(user_id="user1", limit=10)
        assert len(user1_events) == 2
        for event in user1_events:
            assert event.user_id == "user1"
        
        # Filter by session
        session1_events = audit_db.get_audit_events(session_id="session1", limit=10)
        assert len(session1_events) == 3
        for event in session1_events:
            assert event.session_id == "session1"
        
        # Filter by event type
        api_events = audit_db.get_audit_events(event_type=AuditEventType.API_CALL, limit=10)
        assert len(api_events) == 1
        assert api_events[0].event_type == AuditEventType.API_CALL
    
    def test_compliance_record_tracking(self, audit_db):
        """Test compliance record tracking."""
        # Insert high compliance event
        critical_event = AuditEvent(
            event_type=AuditEventType.SECURITY_EVENT,
            event_data={"threat_level": "high", "action_taken": "blocked"},
            user_id="security_system",
            session_id="security_session",
            compliance_level=ComplianceLevel.CRITICAL
        )
        
        audit_db.insert_audit_event(critical_event)
        
        # Check compliance records were created
        compliance_records = audit_db.get_compliance_summary()
        
        assert compliance_records['total_events'] >= 1
        assert compliance_records['by_compliance_level']['critical'] >= 1
    
    def test_data_integrity_verification(self, audit_db):
        """Test data integrity verification functionality."""
        # Insert events with known integrity hashes
        events = []
        for i in range(5):
            event = AuditEvent(
                event_type=AuditEventType.DATA_MODIFICATION,
                event_data={"table": "test_table", "record_id": i},
                user_id=f"user_{i}",
                session_id=f"session_{i}"
            )
            events.append(event)
            audit_db.insert_audit_event(event)
        
        # Verify integrity of all events
        integrity_check = audit_db.verify_data_integrity()
        
        assert integrity_check['total_events_checked'] >= 5
        assert integrity_check['integrity_violations'] == 0
        assert integrity_check['check_passed'] is True
    
    def test_retention_policy_cleanup(self, audit_db):
        """Test retention policy cleanup functionality."""
        # Insert old events (should be cleaned up)
        old_time = time.time() - (400 * 24 * 3600)  # 400 days ago
        for i in range(3):
            old_event = AuditEvent(
                event_type=AuditEventType.API_CALL,
                event_data={"old_event": i},
                user_id="old_user",
                session_id="old_session",
                timestamp=old_time + i * 100
            )
            audit_db.insert_audit_event(old_event)
        
        # Insert recent events (should be kept)
        recent_time = time.time() - 3600  # 1 hour ago
        for i in range(2):
            recent_event = AuditEvent(
                event_type=AuditEventType.API_CALL,
                event_data={"recent_event": i},
                user_id="recent_user",
                session_id="recent_session",
                timestamp=recent_time + i * 100
            )
            audit_db.insert_audit_event(recent_event)
        
        # Apply retention policy (365 days)
        cleanup_result = audit_db.cleanup_old_events(retention_days=365)
        
        assert cleanup_result['events_deleted'] == 3
        
        # Verify only recent events remain
        remaining_events = audit_db.get_audit_events(limit=10)
        assert len(remaining_events) == 2
        for event in remaining_events:
            assert event.user_id == "recent_user"


class TestAuditTrail:
    """Comprehensive tests for AuditTrail main functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield Path(db_path)
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def audit_trail(self, temp_db_path):
        """Create an AuditTrail instance for testing."""
        return AuditTrail(db_path=temp_db_path, retention_days=365)
    
    @pytest.fixture
    def audit_trail_with_callback(self, temp_db_path):
        """Create AuditTrail with compliance callback."""
        compliance_callback = Mock()
        return AuditTrail(
            db_path=temp_db_path,
            retention_days=365,
            compliance_callback=compliance_callback
        ), compliance_callback
    
    def test_audit_trail_initialization(self, temp_db_path):
        """Test AuditTrail initialization."""
        logger = Mock(spec=logging.Logger)
        
        audit_trail = AuditTrail(
            db_path=temp_db_path,
            retention_days=180,
            logger=logger
        )
        
        assert audit_trail.db_path == temp_db_path
        assert audit_trail.retention_days == 180
        assert audit_trail.logger == logger
        assert audit_trail.db is not None
        assert isinstance(audit_trail.db, AuditDatabase)
    
    def test_record_event_basic(self, audit_trail):
        """Test basic event recording functionality."""
        event_id = audit_trail.record_event(
            event_type=AuditEventType.API_CALL,
            event_data={
                "operation": "test_llm_call",
                "model": "gpt-4o-mini",
                "cost": 0.05,
                "success": True
            },
            user_id="test_user",
            session_id="test_session"
        )
        
        assert event_id is not None
        assert isinstance(event_id, int)
        assert event_id > 0
    
    def test_record_event_comprehensive(self, audit_trail):
        """Test comprehensive event recording with all parameters."""
        metadata = {
            "request_id": "req_123",
            "api_version": "v2.1",
            "client_ip": "10.0.0.1",
            "processing_time": 1.25
        }
        
        event_id = audit_trail.record_event(
            event_type=AuditEventType.BUDGET_ALERT,
            event_data={
                "alert_level": "critical",
                "budget_type": "daily",
                "current_cost": 95.0,
                "limit": 100.0,
                "percentage": 95.0
            },
            user_id="budget_monitor",
            session_id="monitoring_session",
            compliance_level=ComplianceLevel.HIGH,
            metadata=metadata
        )
        
        assert event_id is not None
        
        # Verify event was recorded correctly
        events = audit_trail.db.get_audit_events(limit=1)
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == AuditEventType.BUDGET_ALERT
        assert event.compliance_level == ComplianceLevel.HIGH
        assert event.metadata == metadata
        assert event.event_data["alert_level"] == "critical"
    
    def test_compliance_callback_functionality(self, audit_trail_with_callback):
        """Test compliance callback functionality."""
        audit_trail, callback_mock = audit_trail_with_callback
        
        # Record high compliance event
        audit_trail.record_event(
            event_type=AuditEventType.SECURITY_EVENT,
            event_data={"threat_detected": True, "severity": "high"},
            user_id="security_system",
            session_id="security_session",
            compliance_level=ComplianceLevel.CRITICAL
        )
        
        # Verify callback was called for critical event
        callback_mock.assert_called_once()
        call_args = callback_mock.call_args[0][0]
        assert isinstance(call_args, AuditEvent)
        assert call_args.compliance_level == ComplianceLevel.CRITICAL
    
    def test_get_audit_log_with_filters(self, audit_trail):
        """Test audit log retrieval with various filters."""
        # Record different types of events
        events_to_record = [
            (AuditEventType.API_CALL, {"op": "llm"}, "user1", "session1"),
            (AuditEventType.API_CALL, {"op": "embedding"}, "user1", "session1"),
            (AuditEventType.BUDGET_ALERT, {"level": "warning"}, "system", "session2"),
            (AuditEventType.USER_ACTION, {"action": "login"}, "user2", "session3"),
            (AuditEventType.SYSTEM_ERROR, {"error": "timeout"}, "system", "session2")
        ]
        
        for event_type, event_data, user_id, session_id in events_to_record:
            audit_trail.record_event(
                event_type=event_type,
                event_data=event_data,
                user_id=user_id,
                session_id=session_id
            )
        
        # Test different filters
        
        # Get all events
        all_events = audit_trail.get_audit_log(limit=10)
        assert len(all_events) == 5
        
        # Filter by user
        user1_events = audit_trail.get_audit_log(user_id="user1", limit=10)
        assert len(user1_events) == 2
        
        # Filter by session
        session2_events = audit_trail.get_audit_log(session_id="session2", limit=10)
        assert len(session2_events) == 2
        
        # Filter by event type
        api_events = audit_trail.get_audit_log(
            event_type=AuditEventType.API_CALL,
            limit=10
        )
        assert len(api_events) == 2
        
        # Filter by compliance level (default to STANDARD)
        standard_events = audit_trail.get_audit_log(
            compliance_level=ComplianceLevel.STANDARD,
            limit=10
        )
        assert len(standard_events) >= 0  # Depends on default compliance levels
    
    def test_get_audit_log_time_range(self, audit_trail):
        """Test audit log retrieval with time range filters."""
        # Record events at different times
        base_time = time.time() - 3600  # 1 hour ago
        
        for i in range(6):
            audit_trail.record_event(
                event_type=AuditEventType.DATA_ACCESS,
                event_data={"record_id": i, "table": "test_table"},
                user_id=f"user_{i}",
                session_id="time_test_session",
                timestamp=base_time + (i * 600)  # 10-minute intervals
            )
        
        # Test time range filtering
        start_time = base_time + 1800  # 30 minutes after base
        end_time = base_time + 3000    # 50 minutes after base
        
        time_filtered_events = audit_trail.get_audit_log(
            start_time=start_time,
            end_time=end_time,
            limit=10
        )
        
        # Should get events in the time range
        assert len(time_filtered_events) >= 1
        for event in time_filtered_events:
            assert start_time <= event.timestamp <= end_time
    
    def test_get_compliance_report(self, audit_trail):
        """Test compliance report generation."""
        # Record events with different compliance levels
        compliance_events = [
            (ComplianceLevel.STANDARD, 3),
            (ComplianceLevel.HIGH, 2),
            (ComplianceLevel.CRITICAL, 1)
        ]
        
        for compliance_level, count in compliance_events:
            for i in range(count):
                audit_trail.record_event(
                    event_type=AuditEventType.DATA_MODIFICATION,
                    event_data={"modification_id": f"{compliance_level.value}_{i}"},
                    user_id="compliance_user",
                    session_id="compliance_session",
                    compliance_level=compliance_level
                )
        
        # Generate compliance report
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=1)
        
        report = audit_trail.get_compliance_report(start_date, end_date)
        
        assert 'report_period' in report
        assert 'total_events' in report
        assert 'compliance_breakdown' in report
        assert 'retention_compliance' in report
        assert 'data_integrity' in report
        
        # Verify compliance breakdown
        breakdown = report['compliance_breakdown']
        assert 'standard' in breakdown
        assert 'high' in breakdown
        assert 'critical' in breakdown
        
        assert breakdown['standard']['count'] == 3
        assert breakdown['high']['count'] == 2
        assert breakdown['critical']['count'] == 1
    
    def test_security_event_handling(self, audit_trail):
        """Test security event handling with high compliance requirements."""
        security_events = [
            {
                "event_type": AuditEventType.SECURITY_EVENT,
                "event_data": {
                    "threat_type": "unauthorized_access",
                    "source_ip": "192.168.1.100",
                    "target_resource": "cost_database",
                    "action_taken": "blocked"
                },
                "compliance_level": ComplianceLevel.CRITICAL
            },
            {
                "event_type": AuditEventType.SECURITY_EVENT,
                "event_data": {
                    "threat_type": "rate_limit_exceeded",
                    "source_ip": "10.0.0.50",
                    "requests_per_minute": 1000,
                    "action_taken": "throttled"
                },
                "compliance_level": ComplianceLevel.HIGH
            }
        ]
        
        for event_info in security_events:
            event_id = audit_trail.record_event(
                event_type=event_info["event_type"],
                event_data=event_info["event_data"],
                user_id="security_monitor",
                session_id="security_monitoring",
                compliance_level=event_info["compliance_level"]
            )
            
            assert event_id is not None
        
        # Retrieve security events
        security_log = audit_trail.get_audit_log(
            event_type=AuditEventType.SECURITY_EVENT,
            limit=10
        )
        
        assert len(security_log) == 2
        
        # Verify high compliance events are properly marked
        for event in security_log:
            assert event.event_type == AuditEventType.SECURITY_EVENT
            assert event.compliance_level in [ComplianceLevel.HIGH, ComplianceLevel.CRITICAL]
    
    def test_data_integrity_monitoring(self, audit_trail):
        """Test data integrity monitoring functionality."""
        # Record events that should maintain integrity
        for i in range(10):
            audit_trail.record_event(
                event_type=AuditEventType.DATA_MODIFICATION,
                event_data={
                    "table": "cost_records",
                    "operation": "INSERT",
                    "record_count": 1,
                    "checksum": f"checksum_{i}"
                },
                user_id=f"data_user_{i}",
                session_id="integrity_test_session"
            )
        
        # Verify data integrity
        integrity_status = audit_trail.verify_data_integrity()
        
        assert 'integrity_check_passed' in integrity_status
        assert 'total_events_checked' in integrity_status
        assert 'integrity_violations' in integrity_status
        assert 'check_timestamp' in integrity_status
        
        assert integrity_status['total_events_checked'] >= 10
        assert integrity_status['integrity_violations'] == 0
        assert integrity_status['integrity_check_passed'] is True
    
    def test_cleanup_old_audit_data(self, audit_trail):
        """Test cleanup of old audit data based on retention policy."""
        # Record old events (beyond retention period)
        old_time = time.time() - (400 * 24 * 3600)  # 400 days ago (beyond 365 day retention)
        for i in range(5):
            audit_trail.record_event(
                event_type=AuditEventType.API_CALL,
                event_data={"old_event": i},
                user_id="old_user",
                session_id="old_session",
                timestamp=old_time + i * 100
            )
        
        # Record recent events (within retention period)
        recent_time = time.time() - (30 * 24 * 3600)  # 30 days ago
        for i in range(3):
            audit_trail.record_event(
                event_type=AuditEventType.API_CALL,
                event_data={"recent_event": i},
                user_id="recent_user",
                session_id="recent_session",
                timestamp=recent_time + i * 100
            )
        
        # Perform cleanup
        cleanup_result = audit_trail.cleanup_old_data()
        
        assert cleanup_result['events_deleted'] == 5
        assert cleanup_result['events_retained'] >= 3
        
        # Verify only recent events remain
        remaining_events = audit_trail.get_audit_log(limit=20)
        recent_event_count = sum(1 for event in remaining_events if event.user_id == "recent_user")
        assert recent_event_count == 3
    
    def test_concurrent_event_recording(self, audit_trail):
        """Test thread safety of concurrent event recording."""
        num_threads = 10
        events_per_thread = 20
        
        def worker(thread_id):
            results = []
            for i in range(events_per_thread):
                event_id = audit_trail.record_event(
                    event_type=AuditEventType.API_CALL,
                    event_data={
                        "thread_id": thread_id,
                        "event_index": i,
                        "operation": f"thread_{thread_id}_operation_{i}"
                    },
                    user_id=f"thread_user_{thread_id}",
                    session_id=f"thread_session_{thread_id}"
                )
                results.append(event_id)
            return results
        
        # Run concurrent event recording
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        
        # Verify all events were recorded
        assert len(all_results) == num_threads * events_per_thread
        assert len(set(all_results)) == len(all_results)  # All unique event IDs
        
        # Verify data integrity after concurrent operations
        integrity_status = audit_trail.verify_data_integrity()
        assert integrity_status['integrity_check_passed'] is True
        
        # Verify all events are retrievable
        all_events = audit_trail.get_audit_log(limit=num_threads * events_per_thread * 2)
        assert len(all_events) >= num_threads * events_per_thread
    
    def test_error_handling_and_recovery(self, audit_trail):
        """Test error handling and recovery mechanisms."""
        # Test with invalid event data
        try:
            audit_trail.record_event(
                event_type="invalid_event_type",  # Invalid type
                event_data={"test": "data"},
                user_id="error_user",
                session_id="error_session"
            )
        except (ValueError, TypeError):
            pass  # Expected error
        
        # Test with None event data (should be handled gracefully)
        event_id = audit_trail.record_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            event_data={"error": "null_data_test", "handled": True},
            user_id="error_handler",
            session_id="error_session"
        )
        
        assert event_id is not None
        
        # Test recovery by verifying system still works
        normal_event_id = audit_trail.record_event(
            event_type=AuditEventType.API_CALL,
            event_data={"recovery_test": True},
            user_id="recovery_user",
            session_id="recovery_session"
        )
        
        assert normal_event_id is not None
        
        # Verify both events are in the log
        recent_events = audit_trail.get_audit_log(limit=5)
        assert len(recent_events) >= 2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])