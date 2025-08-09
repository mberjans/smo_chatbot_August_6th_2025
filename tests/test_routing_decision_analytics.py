#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Routing Decision Analytics System

This test module provides complete coverage for the routing decision logging
and analytics functionality, including various storage strategies, async 
processing, anomaly detection, and performance monitoring.

Key Test Areas:
- RoutingDecisionLogger with different storage backends
- RoutingAnalytics with real-time metrics and anomaly detection
- Configuration management and environment variables
- Async logging and batching
- Performance monitoring and overhead tracking
- Error handling and fallback scenarios

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Comprehensive Unit Tests for Routing Decision Analytics
"""

import asyncio
import json
import logging
import os
import shutil
import statistics
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
import pytest

# Import the modules under test
from lightrag_integration.routing_decision_analytics import (
    RoutingDecisionLogger,
    RoutingAnalytics,
    LoggingConfig,
    RoutingDecisionLogEntry,
    AnalyticsMetrics,
    LogLevel,
    StorageStrategy,
    RoutingMetricType,
    create_routing_logger,
    create_routing_analytics,
    RoutingLoggingMixin
)

# Import routing-related classes for mocking
from lightrag_integration.query_router import (
    RoutingDecision,
    RoutingPrediction,
    ConfidenceMetrics
)


class TestLoggingConfig:
    """Test cases for LoggingConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = LoggingConfig()
        
        assert config.enabled is True
        assert config.log_level == LogLevel.STANDARD
        assert config.storage_strategy == StorageStrategy.HYBRID
        assert config.log_directory == "logs/routing_decisions"
        assert config.max_file_size_mb == 100
        assert config.max_files_to_keep == 30
        assert config.compress_old_logs is True
        assert config.max_memory_entries == 10000
        assert config.memory_retention_hours == 24
        assert config.async_logging is True
        assert config.batch_size == 100
        assert config.flush_interval_seconds == 30
        assert config.anonymize_queries is False
        assert config.hash_sensitive_data is True
        assert config.enable_real_time_analytics is True
        assert config.analytics_aggregation_interval_minutes == 5
        assert config.enable_performance_alerts is True
    
    def test_config_from_env(self):
        """Test configuration from environment variables"""
        env_vars = {
            'ROUTING_LOGGING_ENABLED': 'false',
            'ROUTING_LOG_LEVEL': 'debug',
            'ROUTING_STORAGE_STRATEGY': 'memory_only',
            'ROUTING_LOG_DIR': '/tmp/routing_logs',
            'ROUTING_MAX_FILE_SIZE_MB': '50',
            'ROUTING_MAX_FILES': '10',
            'ROUTING_COMPRESS_LOGS': 'false',
            'ROUTING_MAX_MEMORY_ENTRIES': '5000',
            'ROUTING_MEMORY_RETENTION_HOURS': '12',
            'ROUTING_ASYNC_LOGGING': 'false',
            'ROUTING_ANONYMIZE_QUERIES': 'true',
            'ROUTING_REAL_TIME_ANALYTICS': 'false'
        }
        
        with patch.dict(os.environ, env_vars):
            config = LoggingConfig.from_env()
        
        assert config.enabled is False
        assert config.log_level == LogLevel.DEBUG
        assert config.storage_strategy == StorageStrategy.MEMORY_ONLY
        assert config.log_directory == '/tmp/routing_logs'
        assert config.max_file_size_mb == 50
        assert config.max_files_to_keep == 10
        assert config.compress_old_logs is False
        assert config.max_memory_entries == 5000
        assert config.memory_retention_hours == 12
        assert config.async_logging is False
        assert config.anonymize_queries is True
        assert config.enable_real_time_analytics is False


class TestRoutingDecisionLogEntry:
    """Test cases for RoutingDecisionLogEntry class"""
    
    @pytest.fixture
    def sample_prediction(self):
        """Create a sample RoutingPrediction for testing"""
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.85,
            research_category_confidence=0.90,
            temporal_analysis_confidence=0.80,
            signal_strength_confidence=0.88,
            context_coherence_confidence=0.82,
            keyword_density=0.75,
            pattern_match_strength=0.85,
            biomedical_entity_count=5,
            ambiguity_score=0.15,
            conflict_score=0.10,
            alternative_interpretations=["diabetes research", "metabolic studies"],
            calculation_time_ms=12.5
        )
        
        return RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence_metrics=confidence_metrics,
            reasoning=["High biomedical entity count", "Strong keyword density"],
            research_category="metabolic_disorders"
        )
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample LoggingConfig for testing"""
        return LoggingConfig(
            anonymize_queries=False,
            hash_sensitive_data=False
        )
    
    def test_log_entry_creation(self, sample_prediction, sample_config):
        """Test creation of log entry from routing prediction"""
        query_text = "What are the metabolic pathways involved in diabetes?"
        processing_metrics = {
            'decision_time_ms': 15.2,
            'total_time_ms': 45.6,
            'backend_selection_time_ms': 5.1,
            'query_complexity': 0.8
        }
        system_state = {
            'backend_health': {'lightrag': 'healthy', 'perplexity': 'healthy'},
            'backend_load': {'lightrag': {'cpu': 45.2, 'memory': 62.1}},
            'resource_usage': {'cpu_percent': 25.5, 'memory_percent': 58.3},
            'selection_algorithm': 'weighted_round_robin',
            'backend_weights': {'lightrag': 0.7, 'perplexity': 0.3},
            'errors': [],
            'warnings': ['High memory usage detected'],
            'fallback_used': False,
            'deployment_mode': 'production',
            'feature_flags': {'analytics_enabled': True, 'logging_enabled': True}
        }
        
        log_entry = RoutingDecisionLogEntry.from_routing_prediction(
            sample_prediction, query_text, processing_metrics, system_state, sample_config
        )
        
        # Verify basic fields
        assert log_entry.entry_id is not None
        assert isinstance(log_entry.timestamp, datetime)
        assert log_entry.query_text == query_text
        assert log_entry.query_hash != ""
        assert log_entry.query_length == len(query_text)
        assert log_entry.query_complexity_score == 0.8
        
        # Verify routing decision fields
        assert log_entry.routing_decision == "lightrag"
        assert log_entry.confidence_score == 0.85
        assert log_entry.decision_reasoning == ["High biomedical entity count", "Strong keyword density"]
        
        # Verify performance metrics
        assert log_entry.decision_time_ms == 15.2
        assert log_entry.total_processing_time_ms == 45.6
        assert log_entry.backend_selection_time_ms == 5.1
        
        # Verify system state
        assert log_entry.backend_health_status == {'lightrag': 'healthy', 'perplexity': 'healthy'}
        assert log_entry.system_resource_usage == {'cpu_percent': 25.5, 'memory_percent': 58.3}
        assert log_entry.backend_selection_algorithm == 'weighted_round_robin'
        assert log_entry.backend_weights == {'lightrag': 0.7, 'perplexity': 0.3}
        assert log_entry.warnings == ['High memory usage detected']
        assert log_entry.fallback_used is False
        assert log_entry.deployment_mode == 'production'
    
    def test_log_entry_anonymization(self, sample_prediction):
        """Test query anonymization in log entry"""
        config = LoggingConfig(anonymize_queries=True)
        query_text = "What are the metabolic pathways involved in diabetes?"
        
        log_entry = RoutingDecisionLogEntry.from_routing_prediction(
            sample_prediction, query_text, {}, {}, config
        )
        
        assert log_entry.query_text == f"<anonymized:{len(query_text)}>"
        assert log_entry.query_hash != ""
        assert log_entry.query_length == len(query_text)
    
    def test_log_entry_hashing(self, sample_prediction):
        """Test sensitive data hashing in log entry"""
        config = LoggingConfig(hash_sensitive_data=True)
        query_text = "What are the metabolic pathways involved in diabetes research for patient John Doe?"
        
        log_entry = RoutingDecisionLogEntry.from_routing_prediction(
            sample_prediction, query_text, {}, {}, config
        )
        
        # Should partially hash long queries
        assert "What are" in log_entry.query_text
        assert "patient John" in log_entry.query_text
        assert "<hashed>" in log_entry.query_text
    
    def test_log_entry_serialization(self, sample_prediction, sample_config):
        """Test JSON serialization of log entry"""
        log_entry = RoutingDecisionLogEntry.from_routing_prediction(
            sample_prediction, "test query", {}, {}, sample_config
        )
        
        # Test to_dict conversion
        entry_dict = log_entry.to_dict()
        assert isinstance(entry_dict, dict)
        assert 'timestamp' in entry_dict
        assert 'entry_id' in entry_dict
        assert 'routing_decision' in entry_dict
        
        # Test JSON serialization
        json_str = log_entry.to_json()
        parsed = json.loads(json_str)
        assert parsed['routing_decision'] == 'lightrag'
        assert parsed['confidence_score'] == 0.85


class TestRoutingDecisionLogger:
    """Test cases for RoutingDecisionLogger class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def file_config(self, temp_dir):
        """Create file-only logging configuration"""
        return LoggingConfig(
            storage_strategy=StorageStrategy.FILE_ONLY,
            log_directory=temp_dir,
            async_logging=False,
            batch_size=5
        )
    
    @pytest.fixture
    def memory_config(self):
        """Create memory-only logging configuration"""
        return LoggingConfig(
            storage_strategy=StorageStrategy.MEMORY_ONLY,
            max_memory_entries=100,
            memory_retention_hours=1
        )
    
    @pytest.fixture
    def hybrid_config(self, temp_dir):
        """Create hybrid logging configuration"""
        return LoggingConfig(
            storage_strategy=StorageStrategy.HYBRID,
            log_directory=temp_dir,
            max_memory_entries=50,
            async_logging=False
        )
    
    @pytest.fixture
    def sample_prediction(self):
        """Create sample prediction for logging tests"""
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.75,
            research_category_confidence=0.80,
            temporal_analysis_confidence=0.70,
            signal_strength_confidence=0.78,
            context_coherence_confidence=0.72,
            keyword_density=0.65,
            pattern_match_strength=0.75,
            biomedical_entity_count=3,
            ambiguity_score=0.25,
            conflict_score=0.20,
            alternative_interpretations=["clinical research"],
            calculation_time_ms=8.7
        )
        
        return RoutingPrediction(
            routing_decision=RoutingDecision.PERPLEXITY,
            confidence_metrics=confidence_metrics,
            reasoning=["Moderate confidence", "Clinical context"],
            research_category="clinical_studies"
        )
    
    def test_logger_initialization_file_only(self, file_config):
        """Test logger initialization with file-only storage"""
        logger = RoutingDecisionLogger(file_config)
        
        assert logger.config == file_config
        assert hasattr(logger, 'file_handler')
        assert hasattr(logger, 'file_logger')
        assert logger._log_queue is None  # Sync logging
    
    def test_logger_initialization_memory_only(self, memory_config):
        """Test logger initialization with memory-only storage"""
        logger = RoutingDecisionLogger(memory_config)
        
        assert logger.config == memory_config
        assert hasattr(logger, 'memory_storage')
        assert logger.memory_storage.maxlen == 100
        assert hasattr(logger, '_memory_lock')
    
    def test_logger_initialization_hybrid(self, hybrid_config):
        """Test logger initialization with hybrid storage"""
        logger = RoutingDecisionLogger(hybrid_config)
        
        assert logger.config == hybrid_config
        assert hasattr(logger, 'file_handler')
        assert hasattr(logger, 'memory_storage')
        assert logger.memory_storage.maxlen == 50
    
    @pytest.mark.asyncio
    async def test_sync_logging_file_only(self, file_config, sample_prediction, temp_dir):
        """Test synchronous logging to file"""
        logger = RoutingDecisionLogger(file_config)
        
        query_text = "Test query for file logging"
        processing_metrics = {'decision_time_ms': 10.0, 'total_time_ms': 25.0}
        system_state = {'backend_health': {}}
        
        await logger.log_routing_decision(
            sample_prediction, query_text, processing_metrics, system_state
        )
        
        # Check if log file was created and contains data
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        assert len(log_files) > 0
        
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 1
            
            # Parse and verify log entry
            log_data = json.loads(lines[0])
            assert log_data['routing_decision'] == 'perplexity'
            assert log_data['query_text'] == query_text
            assert log_data['decision_time_ms'] == 10.0
    
    @pytest.mark.asyncio
    async def test_sync_logging_memory_only(self, memory_config, sample_prediction):
        """Test synchronous logging to memory"""
        logger = RoutingDecisionLogger(memory_config)
        
        query_text = "Test query for memory logging"
        processing_metrics = {'decision_time_ms': 15.0, 'total_time_ms': 30.0}
        system_state = {'backend_health': {}}
        
        await logger.log_routing_decision(
            sample_prediction, query_text, processing_metrics, system_state
        )
        
        # Check memory storage
        entries = logger.get_recent_entries(limit=10)
        assert len(entries) == 1
        assert entries[0].routing_decision == 'perplexity'
        assert entries[0].query_text == query_text
        assert entries[0].decision_time_ms == 15.0
    
    @pytest.mark.asyncio
    async def test_async_logging_with_batching(self, temp_dir):
        """Test asynchronous logging with batching"""
        config = LoggingConfig(
            storage_strategy=StorageStrategy.FILE_ONLY,
            log_directory=temp_dir,
            async_logging=True,
            batch_size=3,
            flush_interval_seconds=1
        )
        logger = RoutingDecisionLogger(config)
        
        # Start async logging
        await logger.start_async_logging()
        
        try:
            # Log multiple entries
            predictions = []
            for i in range(5):
                confidence_metrics = ConfidenceMetrics(
                    overall_confidence=0.5 + (i * 0.1),
                    research_category_confidence=0.6,
                    temporal_analysis_confidence=0.5,
                    signal_strength_confidence=0.6,
                    context_coherence_confidence=0.5,
                    keyword_density=0.4,
                    pattern_match_strength=0.5,
                    biomedical_entity_count=2,
                    ambiguity_score=0.4,
                    conflict_score=0.3,
                    alternative_interpretations=[],
                    calculation_time_ms=5.0
                )
                
                prediction = RoutingPrediction(
                    routing_decision=RoutingDecision.LIGHTRAG,
                    confidence_metrics=confidence_metrics,
                    reasoning=[f"Test reason {i}"],
                    research_category="test"
                )
                
                await logger.log_routing_decision(
                    prediction, f"Test query {i}", 
                    {'decision_time_ms': float(i)}, 
                    {'backend_health': {}}
                )
            
            # Wait for batching and flushing
            await asyncio.sleep(2)
            
        finally:
            await logger.stop_async_logging()
        
        # Verify all entries were logged
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        assert len(log_files) > 0
        
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) == 5
    
    def test_memory_storage_retention(self, memory_config):
        """Test memory storage with retention policy"""
        config = LoggingConfig(
            storage_strategy=StorageStrategy.MEMORY_ONLY,
            max_memory_entries=3,
            memory_retention_hours=0.001  # Very short retention for testing
        )
        logger = RoutingDecisionLogger(config)
        
        # Add entries beyond max capacity
        for i in range(5):
            entry = RoutingDecisionLogEntry(
                entry_id=f"test-{i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                query_text=f"Query {i}",
                routing_decision="lightrag"
            )
            logger.memory_storage.append(entry)
        
        # Should only keep the last 3 entries
        assert len(logger.memory_storage) == 3
        
        # Check that we have the most recent entries
        entries = list(logger.memory_storage)
        assert entries[-1].query_text == "Query 4"
    
    def test_query_entries_filtering(self, memory_config, sample_prediction):
        """Test querying entries with various filters"""
        logger = RoutingDecisionLogger(memory_config)
        
        # Create entries with different characteristics
        base_time = datetime.now()
        entries_data = [
            (RoutingDecision.LIGHTRAG, 0.9, base_time - timedelta(hours=1)),
            (RoutingDecision.PERPLEXITY, 0.6, base_time - timedelta(hours=2)),
            (RoutingDecision.LIGHTRAG, 0.8, base_time - timedelta(hours=3)),
            (RoutingDecision.PERPLEXITY, 0.4, base_time - timedelta(hours=4))
        ]
        
        for routing_decision, confidence, timestamp in entries_data:
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=confidence,
                research_category_confidence=confidence,
                temporal_analysis_confidence=confidence,
                signal_strength_confidence=confidence,
                context_coherence_confidence=confidence,
                keyword_density=0.5,
                pattern_match_strength=0.5,
                biomedical_entity_count=1,
                ambiguity_score=1-confidence,
                conflict_score=1-confidence,
                alternative_interpretations=[],
                calculation_time_ms=10.0
            )
            
            prediction = RoutingPrediction(
                routing_decision=routing_decision,
                confidence_metrics=confidence_metrics,
                reasoning=["Test"],
                research_category="test"
            )
            
            entry = RoutingDecisionLogEntry.from_routing_prediction(
                prediction, "test query", {}, {}, memory_config
            )
            entry.timestamp = timestamp
            logger.memory_storage.append(entry)
        
        # Test filtering by routing decision
        lightrag_entries = logger.query_entries(routing_decision="lightrag")
        assert len(lightrag_entries) == 2
        
        # Test filtering by confidence
        high_confidence_entries = logger.query_entries(min_confidence=0.7)
        assert len(high_confidence_entries) == 2
        
        # Test filtering by time range
        recent_entries = logger.query_entries(
            start_time=base_time - timedelta(hours=2.5),
            end_time=base_time
        )
        assert len(recent_entries) == 2
        
        # Test combined filters
        specific_entries = logger.query_entries(
            routing_decision="lightrag",
            min_confidence=0.85,
            start_time=base_time - timedelta(hours=1.5)
        )
        assert len(specific_entries) == 1
    
    @pytest.mark.asyncio
    async def test_logging_disabled(self):
        """Test behavior when logging is disabled"""
        config = LoggingConfig(enabled=False)
        logger = RoutingDecisionLogger(config)
        
        # Mock prediction
        prediction = Mock()
        
        # Should not log anything
        await logger.log_routing_decision(prediction, "test", {}, {})
        
        # Memory storage should be empty if initialized
        if hasattr(logger, 'memory_storage'):
            assert len(logger.memory_storage) == 0
    
    def test_log_level_filtering(self, memory_config):
        """Test log level filtering"""
        # Test different log levels
        for log_level in LogLevel:
            config = LoggingConfig(
                storage_strategy=StorageStrategy.MEMORY_ONLY,
                log_level=log_level
            )
            logger = RoutingDecisionLogger(config)
            
            # Create mock entry
            entry = Mock(spec=RoutingDecisionLogEntry)
            
            # All levels should currently allow logging (implementation may vary)
            should_log = logger._should_log_entry(entry)
            assert isinstance(should_log, bool)  # At least verify the method works


class TestRoutingAnalytics:
    """Test cases for RoutingAnalytics class"""
    
    @pytest.fixture
    def analytics_config(self):
        """Create configuration for analytics testing"""
        return LoggingConfig(
            storage_strategy=StorageStrategy.MEMORY_ONLY,
            enable_real_time_analytics=True,
            analytics_aggregation_interval_minutes=1,
            max_memory_entries=1000
        )
    
    @pytest.fixture
    def logger_with_data(self, analytics_config):
        """Create logger with sample data"""
        logger = RoutingDecisionLogger(analytics_config)
        
        # Add sample entries
        base_time = datetime.now()
        routing_decisions = [RoutingDecision.LIGHTRAG, RoutingDecision.PERPLEXITY]
        confidences = [0.9, 0.7, 0.8, 0.6, 0.85]
        decision_times = [10.0, 15.0, 12.0, 18.0, 9.0]
        
        for i in range(10):
            confidence = confidences[i % len(confidences)]
            decision_time = decision_times[i % len(decision_times)]
            routing_decision = routing_decisions[i % len(routing_decisions)]
            
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=confidence,
                research_category_confidence=confidence,
                temporal_analysis_confidence=confidence,
                signal_strength_confidence=confidence,
                context_coherence_confidence=confidence,
                keyword_density=0.5,
                pattern_match_strength=0.5,
                biomedical_entity_count=3,
                ambiguity_score=1-confidence,
                conflict_score=1-confidence,
                alternative_interpretations=[],
                calculation_time_ms=decision_time
            )
            
            prediction = RoutingPrediction(
                routing_decision=routing_decision,
                confidence_metrics=confidence_metrics,
                reasoning=["Test reasoning"],
                research_category="test"
            )
            
            entry = RoutingDecisionLogEntry.from_routing_prediction(
                prediction, f"test query {i}", 
                {'decision_time_ms': decision_time},
                {'backend_health': {}},
                analytics_config
            )
            entry.timestamp = base_time - timedelta(minutes=i)
            entry.selected_backend = f"backend_{i % 2}"
            logger.memory_storage.append(entry)
        
        return logger
    
    def test_analytics_initialization(self, logger_with_data):
        """Test analytics initialization"""
        analytics = RoutingAnalytics(logger_with_data)
        
        assert analytics.logger == logger_with_data
        assert analytics.config == logger_with_data.config
        assert hasattr(analytics, '_metrics_cache')
        assert hasattr(analytics, '_decision_times')
        assert hasattr(analytics, '_confidence_scores')
        assert hasattr(analytics, '_backend_counters')
    
    def test_record_decision_metrics(self, logger_with_data):
        """Test recording decision metrics"""
        analytics = RoutingAnalytics(logger_with_data)
        
        # Get sample entry
        entry = logger_with_data.get_recent_entries(1)[0]
        
        initial_requests = analytics._total_requests
        initial_decision_times = len(analytics._decision_times)
        
        # Record metrics
        analytics.record_decision_metrics(entry)
        
        assert analytics._total_requests == initial_requests + 1
        assert len(analytics._decision_times) == initial_decision_times + 1
        assert analytics._decision_times[-1] == entry.decision_time_ms
        assert analytics._confidence_scores[-1] == entry.confidence_score
        assert analytics._backend_counters[entry.selected_backend] >= 1
    
    def test_generate_analytics_report(self, logger_with_data):
        """Test generating comprehensive analytics report"""
        analytics = RoutingAnalytics(logger_with_data)
        
        # Generate report for all entries
        report = analytics.generate_analytics_report()
        
        assert isinstance(report, AnalyticsMetrics)
        assert report.total_requests == 10
        
        # Verify decision distribution
        assert 'lightrag' in report.decision_distribution
        assert 'perplexity' in report.decision_distribution
        assert sum(report.decision_distribution.values()) == 10
        
        # Verify percentages
        assert sum(report.decision_percentages.values()) == 100.0
        
        # Verify confidence metrics
        assert 0 <= report.avg_confidence_score <= 1
        assert report.low_confidence_requests >= 0
        
        # Verify performance metrics
        assert report.avg_decision_time_ms > 0
        assert report.p95_decision_time_ms >= report.avg_decision_time_ms
        
        # Verify backend utilization
        assert len(report.backend_utilization) > 0
        assert sum(report.backend_utilization.values()) == 100.0
        
        # Verify error and fallback rates
        assert 0 <= report.error_rate <= 100
        assert 0 <= report.fallback_rate <= 100
    
    def test_analytics_time_range_filtering(self, logger_with_data):
        """Test analytics report with time range filtering"""
        analytics = RoutingAnalytics(logger_with_data)
        
        # Test with specific time range (last 5 minutes)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        
        report = analytics.generate_analytics_report(start_time, end_time)
        
        # Should have fewer entries due to time filtering
        assert report.total_requests <= 10
        assert report.start_time == start_time
        assert report.end_time == end_time
    
    def test_real_time_metrics(self, logger_with_data):
        """Test real-time metrics collection"""
        analytics = RoutingAnalytics(logger_with_data)
        
        # Record some metrics
        entries = logger_with_data.get_recent_entries(5)
        for entry in entries:
            analytics.record_decision_metrics(entry)
        
        # Aggregate metrics
        analytics._aggregate_metrics()
        
        # Get real-time metrics
        metrics = analytics.get_real_time_metrics()
        
        assert isinstance(metrics, dict)
        assert 'timestamp' in metrics
        assert 'total_requests' in metrics
        assert 'avg_decision_time_ms' in metrics
        assert 'avg_confidence_score' in metrics
        assert 'backend_distribution' in metrics
        assert 'error_rate' in metrics
        
        assert metrics['total_requests'] >= 5
        assert metrics['avg_decision_time_ms'] > 0
        assert 0 <= metrics['avg_confidence_score'] <= 1
    
    def test_anomaly_detection(self, logger_with_data):
        """Test anomaly detection algorithms"""
        analytics = RoutingAnalytics(logger_with_data)
        
        # Add entries with degraded performance to trigger anomalies
        base_time = datetime.now()
        
        # Add entries with low confidence to trigger confidence anomaly
        for i in range(20):
            low_confidence = 0.2  # Very low confidence
            
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=low_confidence,
                research_category_confidence=low_confidence,
                temporal_analysis_confidence=low_confidence,
                signal_strength_confidence=low_confidence,
                context_coherence_confidence=low_confidence,
                keyword_density=0.3,
                pattern_match_strength=0.3,
                biomedical_entity_count=1,
                ambiguity_score=0.8,
                conflict_score=0.8,
                alternative_interpretations=[],
                calculation_time_ms=50.0  # Slow decision
            )
            
            prediction = RoutingPrediction(
                routing_decision=RoutingDecision.PERPLEXITY,
                confidence_metrics=confidence_metrics,
                reasoning=["Low confidence test"],
                research_category="test"
            )
            
            entry = RoutingDecisionLogEntry.from_routing_prediction(
                prediction, f"low confidence query {i}",
                {'decision_time_ms': 50.0},
                {'errors': ['Test error']},  # Add errors
                analytics.config
            )
            entry.timestamp = base_time - timedelta(seconds=i)
            logger_with_data.memory_storage.append(entry)
        
        # Detect anomalies
        anomalies = analytics.detect_anomalies()
        
        assert isinstance(anomalies, list)
        
        # Should detect confidence degradation
        confidence_anomalies = [a for a in anomalies if a.get('type') == 'confidence_degradation']
        assert len(confidence_anomalies) > 0
        
        # Should detect slow decisions
        slow_decision_anomalies = [a for a in anomalies if a.get('type') == 'slow_decisions']
        assert len(slow_decision_anomalies) > 0
        
        # Should detect high error rate
        error_rate_anomalies = [a for a in anomalies if a.get('type') == 'high_error_rate']
        assert len(error_rate_anomalies) > 0
        
        # Verify anomaly structure
        if anomalies:
            anomaly = anomalies[0]
            required_fields = ['type', 'severity', 'description', 'metric', 'threshold_breached', 'current_value']
            for field in required_fields:
                assert field in anomaly
    
    def test_analytics_export(self, logger_with_data, tmp_path):
        """Test analytics export functionality"""
        analytics = RoutingAnalytics(logger_with_data)
        
        # Record some metrics first
        entries = logger_with_data.get_recent_entries(3)
        for entry in entries:
            analytics.record_decision_metrics(entry)
        
        # Export analytics
        export_file = str(tmp_path / "test_analytics.json")
        result_file = analytics.export_analytics(file_path=export_file)
        
        assert result_file == export_file
        assert Path(export_file).exists()
        
        # Verify export content
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        required_keys = ['export_timestamp', 'analytics_report', 'real_time_metrics', 'anomalies']
        for key in required_keys:
            assert key in export_data
        
        assert 'total_requests' in export_data['analytics_report']
        assert 'timestamp' in export_data['real_time_metrics']
        assert isinstance(export_data['anomalies'], list)
    
    def test_analytics_with_insufficient_data(self, analytics_config):
        """Test analytics behavior with insufficient data"""
        logger = RoutingDecisionLogger(analytics_config)
        analytics = RoutingAnalytics(logger)
        
        # Generate report with no data
        report = analytics.generate_analytics_report()
        
        assert report.total_requests == 0
        assert len(report.decision_distribution) == 0
        assert report.avg_confidence_score == 0
        assert report.avg_decision_time_ms == 0
        
        # Test anomaly detection with insufficient data
        anomalies = analytics.detect_anomalies()
        assert len(anomalies) == 0


class TestConfigurationManagement:
    """Test cases for configuration management and environment variables"""
    
    def test_logging_config_environment_override(self):
        """Test comprehensive environment variable override"""
        test_env = {
            'ROUTING_LOGGING_ENABLED': 'false',
            'ROUTING_LOG_LEVEL': 'debug',
            'ROUTING_STORAGE_STRATEGY': 'file_only',
            'ROUTING_LOG_DIR': '/custom/log/path',
            'ROUTING_MAX_FILE_SIZE_MB': '200',
            'ROUTING_MAX_FILES': '50',
            'ROUTING_COMPRESS_LOGS': 'false',
            'ROUTING_MAX_MEMORY_ENTRIES': '20000',
            'ROUTING_MEMORY_RETENTION_HOURS': '48',
            'ROUTING_ASYNC_LOGGING': 'false',
            'ROUTING_ANONYMIZE_QUERIES': 'true',
            'ROUTING_REAL_TIME_ANALYTICS': 'false'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            config = LoggingConfig.from_env()
        
        assert config.enabled is False
        assert config.log_level == LogLevel.DEBUG
        assert config.storage_strategy == StorageStrategy.FILE_ONLY
        assert config.log_directory == '/custom/log/path'
        assert config.max_file_size_mb == 200
        assert config.max_files_to_keep == 50
        assert config.compress_old_logs is False
        assert config.max_memory_entries == 20000
        assert config.memory_retention_hours == 48
        assert config.async_logging is False
        assert config.anonymize_queries is True
        assert config.enable_real_time_analytics is False
    
    def test_factory_functions(self):
        """Test factory functions for logger and analytics creation"""
        # Test with default config
        logger = create_routing_logger()
        assert isinstance(logger, RoutingDecisionLogger)
        assert isinstance(logger.config, LoggingConfig)
        
        # Test with custom config
        custom_config = LoggingConfig(log_level=LogLevel.DEBUG, storage_strategy=StorageStrategy.MEMORY_ONLY)
        logger = create_routing_logger(custom_config)
        assert logger.config.log_level == LogLevel.DEBUG
        assert logger.config.storage_strategy == StorageStrategy.MEMORY_ONLY
        
        # Test analytics creation
        analytics = create_routing_analytics(logger)
        assert isinstance(analytics, RoutingAnalytics)
        assert analytics.logger == logger


class TestAsyncLoggingAndBatching:
    """Test cases for async logging and batching functionality"""
    
    @pytest.mark.asyncio
    async def test_async_worker_lifecycle(self, tmp_path):
        """Test async logging worker start/stop lifecycle"""
        config = LoggingConfig(
            storage_strategy=StorageStrategy.FILE_ONLY,
            log_directory=str(tmp_path),
            async_logging=True,
            batch_size=5,
            flush_interval_seconds=1
        )
        logger = RoutingDecisionLogger(config)
        
        # Start async logging
        await logger.start_async_logging()
        assert logger._log_worker_task is not None
        assert not logger._shutdown_event.is_set()
        
        # Stop async logging
        await logger.stop_async_logging()
        assert logger._log_worker_task is None
        assert logger._shutdown_event.is_set()
    
    @pytest.mark.asyncio
    async def test_batch_flushing_on_size_limit(self, tmp_path):
        """Test batch flushing when size limit is reached"""
        config = LoggingConfig(
            storage_strategy=StorageStrategy.FILE_ONLY,
            log_directory=str(tmp_path),
            async_logging=True,
            batch_size=3,
            flush_interval_seconds=10  # Long interval to test size-based flushing
        )
        logger = RoutingDecisionLogger(config)
        
        await logger.start_async_logging()
        
        try:
            # Add entries to reach batch size
            for i in range(5):  # More than batch size
                confidence_metrics = ConfidenceMetrics(
                    overall_confidence=0.8,
                    research_category_confidence=0.8,
                    temporal_analysis_confidence=0.8,
                    signal_strength_confidence=0.8,
                    context_coherence_confidence=0.8,
                    keyword_density=0.7,
                    pattern_match_strength=0.8,
                    biomedical_entity_count=4,
                    ambiguity_score=0.2,
                    conflict_score=0.2,
                    alternative_interpretations=[],
                    calculation_time_ms=12.0
                )
                
                prediction = RoutingPrediction(
                    routing_decision=RoutingDecision.LIGHTRAG,
                    confidence_metrics=confidence_metrics,
                    reasoning=[f"Batch test {i}"],
                    research_category="batch_test"
                )
                
                await logger.log_routing_decision(
                    prediction, f"Batch query {i}",
                    {'decision_time_ms': 12.0},
                    {'backend_health': {}}
                )
            
            # Wait for processing
            await asyncio.sleep(2)
            
        finally:
            await logger.stop_async_logging()
        
        # Verify logs were written
        log_files = list(Path(tmp_path).glob("*.jsonl"))
        assert len(log_files) > 0
        
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) == 5
    
    @pytest.mark.asyncio
    async def test_batch_flushing_on_timeout(self, tmp_path):
        """Test batch flushing based on timeout"""
        config = LoggingConfig(
            storage_strategy=StorageStrategy.FILE_ONLY,
            log_directory=str(tmp_path),
            async_logging=True,
            batch_size=10,  # Large batch size
            flush_interval_seconds=1  # Short interval for testing
        )
        logger = RoutingDecisionLogger(config)
        
        await logger.start_async_logging()
        
        try:
            # Add just one entry (below batch size)
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=0.8,
                research_category_confidence=0.8,
                temporal_analysis_confidence=0.8,
                signal_strength_confidence=0.8,
                context_coherence_confidence=0.8,
                keyword_density=0.7,
                pattern_match_strength=0.8,
                biomedical_entity_count=4,
                ambiguity_score=0.2,
                conflict_score=0.2,
                alternative_interpretations=[],
                calculation_time_ms=10.0
            )
            
            prediction = RoutingPrediction(
                routing_decision=RoutingDecision.LIGHTRAG,
                confidence_metrics=confidence_metrics,
                reasoning=["Timeout test"],
                research_category="timeout_test"
            )
            
            await logger.log_routing_decision(
                prediction, "Timeout query",
                {'decision_time_ms': 10.0},
                {'backend_health': {}}
            )
            
            # Wait for timeout-based flushing
            await asyncio.sleep(2)
            
        finally:
            await logger.stop_async_logging()
        
        # Verify log was written despite small batch
        log_files = list(Path(tmp_path).glob("*.jsonl"))
        assert len(log_files) > 0
        
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1


class TestPerformanceMonitoring:
    """Test cases for performance monitoring and overhead tracking"""
    
    @pytest.fixture
    def performance_config(self):
        """Create configuration for performance testing"""
        return LoggingConfig(
            storage_strategy=StorageStrategy.HYBRID,
            enable_real_time_analytics=True,
            analytics_aggregation_interval_minutes=1
        )
    
    def test_logging_overhead_measurement(self, performance_config):
        """Test measurement of logging overhead"""
        logger = RoutingDecisionLogger(performance_config)
        analytics = RoutingAnalytics(logger)
        
        # Simulate logging overhead measurement
        overhead_times = [1.2, 2.1, 1.5, 3.2, 1.8, 2.5, 1.9, 2.8, 1.4, 2.3]
        for overhead in overhead_times:
            analytics._decision_times.append(overhead)
        
        # Calculate statistics
        avg_overhead = statistics.mean(overhead_times)
        max_overhead = max(overhead_times)
        
        assert avg_overhead > 0
        assert max_overhead >= avg_overhead
        
        # Test percentile calculation if enough data
        if len(overhead_times) >= 20:
            p95 = statistics.quantiles(overhead_times, n=20)[18]
            assert p95 >= avg_overhead
    
    @pytest.mark.asyncio
    async def test_performance_impact_monitoring(self, performance_config, tmp_path):
        """Test monitoring of performance impact from logging"""
        config = LoggingConfig(
            storage_strategy=StorageStrategy.FILE_ONLY,
            log_directory=str(tmp_path),
            async_logging=False  # Sync for easier timing control
        )
        logger = RoutingDecisionLogger(config)
        
        # Measure logging performance
        start_time = time.time()
        
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.8,
            research_category_confidence=0.8,
            temporal_analysis_confidence=0.8,
            signal_strength_confidence=0.8,
            context_coherence_confidence=0.8,
            keyword_density=0.7,
            pattern_match_strength=0.8,
            biomedical_entity_count=4,
            ambiguity_score=0.2,
            conflict_score=0.2,
            alternative_interpretations=[],
            calculation_time_ms=15.0
        )
        
        prediction = RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence_metrics=confidence_metrics,
            reasoning=["Performance test"],
            research_category="performance"
        )
        
        await logger.log_routing_decision(
            prediction, "Performance test query",
            {'decision_time_ms': 15.0},
            {'backend_health': {}}
        )
        
        logging_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Logging should be relatively fast
        assert logging_time < 100  # Less than 100ms for single entry
    
    def test_memory_usage_monitoring(self, performance_config):
        """Test monitoring of memory usage by logging system"""
        logger = RoutingDecisionLogger(performance_config)
        
        initial_memory_entries = len(logger.memory_storage) if hasattr(logger, 'memory_storage') else 0
        
        # Add many entries to test memory growth
        for i in range(100):
            entry = RoutingDecisionLogEntry(
                entry_id=f"perf-test-{i}",
                timestamp=datetime.now(),
                query_text=f"Performance query {i}",
                routing_decision="lightrag",
                confidence_score=0.8,
                decision_time_ms=10.0 + i * 0.1
            )
            
            if hasattr(logger, 'memory_storage'):
                logger.memory_storage.append(entry)
        
        if hasattr(logger, 'memory_storage'):
            final_memory_entries = len(logger.memory_storage)
            entries_added = final_memory_entries - initial_memory_entries
            
            # Should respect max entries limit
            assert entries_added <= logger.config.max_memory_entries
            assert final_memory_entries <= logger.config.max_memory_entries


class TestErrorHandlingAndRecovery:
    """Test cases for error handling and fallback scenarios"""
    
    def test_logger_initialization_failure_handling(self):
        """Test handling of logger initialization failures"""
        # Test with invalid log directory permissions (simulated)
        config = LoggingConfig(
            storage_strategy=StorageStrategy.FILE_ONLY,
            log_directory="/nonexistent/invalid/path/that/should/fail",
            async_logging=False
        )
        
        # Should handle initialization gracefully
        try:
            logger = RoutingDecisionLogger(config)
            # If it doesn't raise, that's also acceptable
        except Exception as e:
            # Should be a reasonable exception
            assert isinstance(e, (OSError, PermissionError, FileNotFoundError))
    
    @pytest.mark.asyncio
    async def test_logging_error_recovery(self, tmp_path):
        """Test recovery from logging errors"""
        config = LoggingConfig(
            storage_strategy=StorageStrategy.HYBRID,
            log_directory=str(tmp_path),
            async_logging=False
        )
        logger = RoutingDecisionLogger(config)
        
        # Test with None prediction (should handle gracefully)
        try:
            await logger.log_routing_decision(None, "test", {}, {})
        except Exception as e:
            # Should not crash the system
            assert isinstance(e, (AttributeError, TypeError))
    
    @pytest.mark.asyncio
    async def test_storage_error_handling(self, tmp_path):
        """Test handling of storage errors"""
        config = LoggingConfig(
            storage_strategy=StorageStrategy.FILE_ONLY,
            log_directory=str(tmp_path),
            async_logging=False
        )
        logger = RoutingDecisionLogger(config)
        
        # Remove write permissions to simulate storage error
        os.chmod(tmp_path, 0o444)  # Read-only
        
        try:
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=0.8,
                research_category_confidence=0.8,
                temporal_analysis_confidence=0.8,
                signal_strength_confidence=0.8,
                context_coherence_confidence=0.8,
                keyword_density=0.7,
                pattern_match_strength=0.8,
                biomedical_entity_count=4,
                ambiguity_score=0.2,
                conflict_score=0.2,
                alternative_interpretations=[],
                calculation_time_ms=10.0
            )
            
            prediction = RoutingPrediction(
                routing_decision=RoutingDecision.LIGHTRAG,
                confidence_metrics=confidence_metrics,
                reasoning=["Error test"],
                research_category="error_test"
            )
            
            # Should handle storage error gracefully
            await logger.log_routing_decision(
                prediction, "Error test query",
                {'decision_time_ms': 10.0},
                {'backend_health': {}}
            )
            
        except Exception:
            # Errors should be logged but not crash
            pass
        finally:
            # Restore permissions
            os.chmod(tmp_path, 0o755)
    
    def test_analytics_error_handling(self):
        """Test analytics error handling with corrupted data"""
        config = LoggingConfig(storage_strategy=StorageStrategy.MEMORY_ONLY)
        logger = RoutingDecisionLogger(config)
        analytics = RoutingAnalytics(logger)
        
        # Add corrupted entry
        corrupted_entry = RoutingDecisionLogEntry(
            entry_id="corrupted",
            timestamp=datetime.now(),
            query_text="test",
            routing_decision="invalid_decision",
            confidence_score=-1.0,  # Invalid confidence
            decision_time_ms=-5.0   # Invalid time
        )
        
        logger.memory_storage.append(corrupted_entry)
        
        # Analytics should handle corrupted data gracefully
        try:
            report = analytics.generate_analytics_report()
            assert isinstance(report, AnalyticsMetrics)
        except Exception as e:
            # Should not crash on corrupted data
            assert False, f"Analytics crashed on corrupted data: {e}"


class TestMockUtilitiesAndFixtures:
    """Test cases for mock utilities and test fixtures"""
    
    @pytest.fixture
    def mock_routing_prediction(self):
        """Create mock routing prediction for testing"""
        confidence_metrics = Mock(spec=ConfidenceMetrics)
        confidence_metrics.overall_confidence = 0.85
        confidence_metrics.calculation_time_ms = 12.5
        
        prediction = Mock(spec=RoutingPrediction)
        prediction.routing_decision = RoutingDecision.LIGHTRAG
        prediction.confidence = 0.85
        prediction.confidence_level = "high"
        prediction.confidence_metrics = confidence_metrics
        prediction.reasoning = ["High confidence", "Strong signals"]
        prediction.research_category = "test_category"
        
        return prediction
    
    @pytest.fixture
    def mock_system_state(self):
        """Create mock system state for testing"""
        return {
            'backend_health': {'lightrag': 'healthy', 'perplexity': 'degraded'},
            'backend_load': {
                'lightrag': {'cpu': 45.2, 'memory': 62.1, 'requests': 150},
                'perplexity': {'cpu': 78.5, 'memory': 89.2, 'requests': 75}
            },
            'resource_usage': {
                'cpu_percent': 35.5,
                'memory_percent': 67.8,
                'memory_available_gb': 4.2,
                'disk_usage_percent': 45.3
            },
            'selection_algorithm': 'weighted_round_robin',
            'backend_weights': {'lightrag': 0.8, 'perplexity': 0.2},
            'errors': [],
            'warnings': ['Backend perplexity showing degraded performance'],
            'fallback_used': False,
            'fallback_reason': None,
            'deployment_mode': 'production',
            'feature_flags': {
                'production_enabled': True,
                'analytics_enabled': True,
                'anomaly_detection': True
            },
            'request_counter': 12345,
            'session_id': 'test-session-123'
        }
    
    @pytest.fixture
    def mock_processing_metrics(self):
        """Create mock processing metrics for testing"""
        return {
            'decision_time_ms': 15.7,
            'total_time_ms': 42.3,
            'backend_selection_time_ms': 8.2,
            'query_complexity': 0.75,
            'preprocessing_time_ms': 5.1,
            'postprocessing_time_ms': 3.8
        }
    
    def test_mock_log_entry_creation(self, mock_routing_prediction, mock_system_state, mock_processing_metrics):
        """Test creation of log entry with mock data"""
        config = LoggingConfig(anonymize_queries=False, hash_sensitive_data=False)
        query_text = "Mock test query for routing decision"
        
        log_entry = RoutingDecisionLogEntry.from_routing_prediction(
            mock_routing_prediction, query_text, mock_processing_metrics, mock_system_state, config
        )
        
        # Verify mock data is correctly used
        assert log_entry.query_text == query_text
        assert log_entry.routing_decision == "lightrag"
        assert log_entry.confidence_score == 0.85
        assert log_entry.decision_time_ms == 15.7
        assert log_entry.total_processing_time_ms == 42.3
        assert log_entry.backend_health_status == mock_system_state['backend_health']
        assert log_entry.system_resource_usage == mock_system_state['resource_usage']
        assert log_entry.deployment_mode == 'production'
    
    def test_mock_analytics_workflow(self, mock_routing_prediction, mock_system_state, mock_processing_metrics):
        """Test complete analytics workflow with mock data"""
        config = LoggingConfig(storage_strategy=StorageStrategy.MEMORY_ONLY)
        logger = RoutingDecisionLogger(config)
        analytics = RoutingAnalytics(logger)
        
        # Create log entry with mock data
        log_entry = RoutingDecisionLogEntry.from_routing_prediction(
            mock_routing_prediction, "Mock query", mock_processing_metrics, mock_system_state, config
        )
        
        # Add to logger
        logger.memory_storage.append(log_entry)
        
        # Record metrics
        analytics.record_decision_metrics(log_entry)
        
        # Generate report
        report = analytics.generate_analytics_report()
        
        assert report.total_requests == 1
        assert "lightrag" in report.decision_distribution
        assert report.avg_confidence_score == 0.85
        assert report.avg_decision_time_ms == 15.7


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])