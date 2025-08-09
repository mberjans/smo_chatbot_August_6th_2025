#!/usr/bin/env python3
"""
Integration Tests for Enhanced Production Router with Routing Decision Analytics

This test module provides comprehensive integration testing for the
EnhancedProductionIntelligentQueryRouter with its logging and analytics
capabilities in realistic scenarios.

Key Test Areas:
- End-to-end routing with logging integration
- Performance impact of logging on routing decisions
- Production environment simulation
- Load balancer integration with analytics
- Anomaly detection in production scenarios
- Concurrent access and thread safety
- Configuration migration and environment setup

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Integration Tests for Enhanced Production Router Logging
"""

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
import pytest

# Import the enhanced router and dependencies
from lightrag_integration.enhanced_production_router import (
    EnhancedProductionIntelligentQueryRouter,
    EnhancedFeatureFlags,
    create_enhanced_production_router
)
from lightrag_integration.routing_decision_analytics import (
    LoggingConfig,
    LogLevel,
    StorageStrategy,
    RoutingDecisionLogger,
    RoutingAnalytics
)
from lightrag_integration.production_intelligent_query_router import (
    ProductionIntelligentQueryRouter,
    DeploymentMode,
    ProductionFeatureFlags
)
from lightrag_integration.intelligent_query_router import (
    IntelligentQueryRouter,
    LoadBalancingConfig,
    HealthCheckConfig,
    BackendType
)
from lightrag_integration.query_router import (
    BiomedicalQueryRouter,
    RoutingDecision,
    RoutingPrediction,
    ConfidenceMetrics
)


class TestEnhancedRouterIntegration:
    """Integration tests for enhanced production router with logging"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for test logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_base_router(self):
        """Create mock base router for testing"""
        base_router = Mock(spec=BiomedicalQueryRouter)
        base_router.route_query = AsyncMock(return_value=Mock(spec=RoutingPrediction))
        return base_router
    
    @pytest.fixture
    def enhanced_feature_flags(self):
        """Create enhanced feature flags for testing"""
        return EnhancedFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION,
            production_traffic_percentage=100.0,
            enable_routing_logging=True,
            routing_log_level=LogLevel.DETAILED,
            routing_storage_strategy=StorageStrategy.HYBRID,
            enable_real_time_analytics=True,
            analytics_aggregation_interval_minutes=1,
            enable_anomaly_detection=True,
            enable_performance_impact_monitoring=True,
            max_logging_overhead_ms=10.0,
            anonymize_query_content=False,
            hash_sensitive_data=True
        )
    
    @pytest.fixture
    def logging_config(self, temp_log_dir):
        """Create logging configuration for testing"""
        return LoggingConfig(
            enabled=True,
            log_level=LogLevel.DETAILED,
            storage_strategy=StorageStrategy.HYBRID,
            log_directory=temp_log_dir,
            max_file_size_mb=10,
            max_memory_entries=100,
            async_logging=True,
            batch_size=5,
            flush_interval_seconds=2
        )
    
    @pytest.fixture
    async def enhanced_router(self, mock_base_router, enhanced_feature_flags, logging_config):
        """Create enhanced router for testing"""
        router = EnhancedProductionIntelligentQueryRouter(
            base_router=mock_base_router,
            feature_flags=enhanced_feature_flags,
            logging_config=logging_config
        )
        
        await router.start_monitoring()
        yield router
        await router.stop_monitoring()
    
    def test_enhanced_router_initialization(self, mock_base_router, enhanced_feature_flags, logging_config):
        """Test enhanced router initialization with all components"""
        router = EnhancedProductionIntelligentQueryRouter(
            base_router=mock_base_router,
            feature_flags=enhanced_feature_flags,
            logging_config=logging_config
        )
        
        # Verify enhanced feature flags
        assert router.enhanced_feature_flags == enhanced_feature_flags
        assert router.enhanced_feature_flags.enable_routing_logging
        assert router.enhanced_feature_flags.routing_log_level == LogLevel.DETAILED
        
        # Verify logging system initialization
        if enhanced_feature_flags.enable_routing_logging:
            assert router.routing_logger is not None
            assert router.routing_analytics is not None
            assert isinstance(router.routing_logger, RoutingDecisionLogger)
            assert isinstance(router.routing_analytics, RoutingAnalytics)
        
        # Verify performance monitoring
        assert hasattr(router, 'logging_overhead_metrics')
        assert hasattr(router, 'total_logged_decisions')
        assert hasattr(router, 'detected_anomalies')
    
    @pytest.mark.asyncio
    async def test_end_to_end_routing_with_logging(self, enhanced_router):
        """Test complete routing workflow with logging integration"""
        # Setup mock prediction
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
        
        mock_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence_metrics=confidence_metrics,
            reasoning=["High biomedical entity count", "Strong keyword density"],
            research_category="metabolic_disorders"
        )
        
        # Configure base router mock
        enhanced_router.legacy_router.route_query.return_value = mock_prediction
        
        # Test query
        test_query = "What are the latest developments in diabetes treatment and metabolic pathway research?"
        
        initial_logged_decisions = enhanced_router.total_logged_decisions
        
        # Route query
        result = await enhanced_router.route_query(test_query)
        
        # Verify routing result
        assert result == mock_prediction
        assert result.routing_decision == RoutingDecision.LIGHTRAG
        assert result.confidence == 0.85
        
        # Verify logging occurred
        assert enhanced_router.total_logged_decisions == initial_logged_decisions + 1
        
        # Wait for async processing
        await asyncio.sleep(1)
        
        # Verify log entries
        if enhanced_router.routing_logger:
            recent_entries = enhanced_router.routing_logger.get_recent_entries(5)
            assert len(recent_entries) >= 1
            
            logged_entry = recent_entries[-1]
            assert logged_entry.query_text == test_query
            assert logged_entry.routing_decision == "lightrag"
            assert logged_entry.confidence_score == 0.85
    
    @pytest.mark.asyncio
    async def test_logging_performance_impact(self, enhanced_router):
        """Test performance impact of logging on routing decisions"""
        # Setup mock prediction
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.75,
            research_category_confidence=0.80,
            temporal_analysis_confidence=0.70,
            signal_strength_confidence=0.75,
            context_coherence_confidence=0.72,
            keyword_density=0.65,
            pattern_match_strength=0.75,
            biomedical_entity_count=3,
            ambiguity_score=0.25,
            conflict_score=0.20,
            alternative_interpretations=[],
            calculation_time_ms=8.0
        )
        
        mock_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.PERPLEXITY,
            confidence_metrics=confidence_metrics,
            reasoning=["Moderate confidence"],
            research_category="general_research"
        )
        
        enhanced_router.legacy_router.route_query.return_value = mock_prediction
        
        # Measure routing performance with logging
        test_queries = [
            "How do metabolic pathways affect cellular energy production?",
            "What are the biomarkers for early diabetes detection?",
            "Explain the role of insulin in glucose metabolism",
            "Recent advances in personalized medicine for metabolic disorders",
            "Clinical trials for new diabetes treatments"
        ]
        
        routing_times = []
        
        for query in test_queries:
            start_time = time.time()
            await enhanced_router.route_query(query)
            end_time = time.time()
            routing_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Verify reasonable performance
        avg_routing_time = sum(routing_times) / len(routing_times)
        max_routing_time = max(routing_times)
        
        # Routing with logging should still be fast
        assert avg_routing_time < 100  # Less than 100ms average
        assert max_routing_time < 200   # Less than 200ms max
        
        # Verify logging overhead tracking
        if enhanced_router.logging_overhead_metrics:
            avg_overhead = sum(enhanced_router.logging_overhead_metrics) / len(enhanced_router.logging_overhead_metrics)
            assert avg_overhead < enhanced_router.enhanced_feature_flags.max_logging_overhead_ms
    
    @pytest.mark.asyncio
    async def test_concurrent_routing_with_logging(self, enhanced_router):
        """Test concurrent routing requests with logging"""
        # Setup mock prediction
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.80,
            research_category_confidence=0.85,
            temporal_analysis_confidence=0.75,
            signal_strength_confidence=0.80,
            context_coherence_confidence=0.78,
            keyword_density=0.70,
            pattern_match_strength=0.80,
            biomedical_entity_count=4,
            ambiguity_score=0.20,
            conflict_score=0.15,
            alternative_interpretations=[],
            calculation_time_ms=10.0
        )
        
        mock_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence_metrics=confidence_metrics,
            reasoning=["Concurrent test"],
            research_category="concurrent_test"
        )
        
        enhanced_router.legacy_router.route_query.return_value = mock_prediction
        
        # Create concurrent queries
        concurrent_queries = [
            f"Concurrent query {i}: metabolic research topic {i}"
            for i in range(10)
        ]
        
        initial_logged_decisions = enhanced_router.total_logged_decisions
        
        # Execute concurrent queries
        tasks = [
            enhanced_router.route_query(query)
            for query in concurrent_queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all queries were processed
        assert len(results) == len(concurrent_queries)
        for result in results:
            assert result.routing_decision == RoutingDecision.LIGHTRAG
        
        # Wait for logging to complete
        await asyncio.sleep(2)
        
        # Verify all queries were logged
        final_logged_decisions = enhanced_router.total_logged_decisions
        assert final_logged_decisions == initial_logged_decisions + len(concurrent_queries)
        
        # Verify thread safety of logging system
        if enhanced_router.routing_logger:
            recent_entries = enhanced_router.routing_logger.get_recent_entries(15)
            logged_query_count = sum(1 for entry in recent_entries if entry.query_text.startswith("Concurrent query"))
            assert logged_query_count == len(concurrent_queries)
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_integration(self, enhanced_router):
        """Test anomaly detection in production routing scenarios"""
        # Simulate degraded performance scenario
        degraded_confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.25,  # Very low confidence
            research_category_confidence=0.30,
            temporal_analysis_confidence=0.20,
            signal_strength_confidence=0.25,
            context_coherence_confidence=0.22,
            keyword_density=0.15,
            pattern_match_strength=0.25,
            biomedical_entity_count=1,
            ambiguity_score=0.75,
            conflict_score=0.80,
            alternative_interpretations=[],
            calculation_time_ms=75.0  # Slow decision
        )
        
        degraded_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.PERPLEXITY,
            confidence_metrics=degraded_confidence_metrics,
            reasoning=["Low confidence", "Degraded performance"],
            research_category="degraded_test"
        )
        
        enhanced_router.legacy_router.route_query.return_value = degraded_prediction
        
        # Generate multiple degraded queries to trigger anomaly detection
        degraded_queries = [
            f"Degraded query {i}: unclear medical question"
            for i in range(25)  # Enough to trigger anomaly detection
        ]
        
        for query in degraded_queries:
            await enhanced_router.route_query(query)
        
        # Wait for analytics processing
        await asyncio.sleep(3)
        
        # Check for detected anomalies
        if enhanced_router.enhanced_feature_flags.enable_anomaly_detection and enhanced_router.routing_analytics:
            anomalies = enhanced_router.routing_analytics.detect_anomalies()
            
            # Should detect confidence degradation
            confidence_anomalies = [a for a in anomalies if a.get('type') == 'confidence_degradation']
            assert len(confidence_anomalies) > 0, "Should detect confidence degradation anomaly"
            
            # Should detect slow decisions
            slow_decision_anomalies = [a for a in anomalies if a.get('type') == 'slow_decisions']
            assert len(slow_decision_anomalies) > 0, "Should detect slow decision anomaly"
            
            # Verify anomalies are stored in router
            assert len(enhanced_router.detected_anomalies) >= len(anomalies)
    
    @pytest.mark.asyncio
    async def test_error_handling_with_logging(self, enhanced_router):
        """Test error handling integration with logging system"""
        # Configure base router to raise an exception
        enhanced_router.legacy_router.route_query.side_effect = Exception("Simulated routing error")
        
        initial_logged_decisions = enhanced_router.total_logged_decisions
        
        # Attempt to route query with error
        with pytest.raises(Exception, match="Simulated routing error"):
            await enhanced_router.route_query("Error test query")
        
        # Wait for error logging
        await asyncio.sleep(1)
        
        # Verify error was logged
        assert enhanced_router.total_logged_decisions == initial_logged_decisions + 1
        
        if enhanced_router.routing_logger:
            recent_entries = enhanced_router.routing_logger.get_recent_entries(5)
            if recent_entries:
                error_entry = recent_entries[-1]
                assert "Error test query" in error_entry.query_text
                assert len(error_entry.errors_encountered) > 0
    
    def test_analytics_summary_integration(self, enhanced_router):
        """Test integration of analytics summary with production router"""
        summary = enhanced_router.get_routing_analytics_summary()
        
        # Verify structure
        assert isinstance(summary, dict)
        assert 'routing_analytics' in summary
        
        routing_analytics = summary['routing_analytics']
        
        if enhanced_router.enhanced_feature_flags.enable_routing_logging:
            assert routing_analytics['status'] == 'enabled'
            assert 'total_logged_decisions' in routing_analytics
            assert 'logging_performance' in routing_analytics or routing_analytics['total_logged_decisions'] == 0
            assert 'analytics_report' in routing_analytics
            assert 'real_time_metrics' in routing_analytics
            assert 'detected_anomalies' in routing_analytics
        else:
            assert routing_analytics['status'] == 'disabled'
    
    def test_health_status_integration(self, enhanced_router):
        """Test health status integration with routing analytics"""
        health_status = enhanced_router.get_health_status()
        
        # Verify enhanced health status structure
        assert isinstance(health_status, dict)
        assert 'routing_analytics' in health_status
        
        routing_health = health_status['routing_analytics']
        
        # Verify routing analytics health fields
        assert 'logging_enabled' in routing_health
        assert 'analytics_enabled' in routing_health
        assert 'total_logged_decisions' in routing_health
        assert 'logging_errors' in routing_health
        assert 'anomaly_detection_status' in routing_health
        assert 'detected_anomalies_count' in routing_health
        
        assert routing_health['logging_enabled'] == enhanced_router.enhanced_feature_flags.enable_routing_logging
        assert routing_health['analytics_enabled'] == enhanced_router.enhanced_feature_flags.enable_real_time_analytics
    
    @pytest.mark.asyncio
    async def test_comprehensive_analytics_export(self, enhanced_router, temp_log_dir):
        """Test comprehensive analytics export functionality"""
        # Generate some routing activity
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.78,
            research_category_confidence=0.82,
            temporal_analysis_confidence=0.74,
            signal_strength_confidence=0.78,
            context_coherence_confidence=0.76,
            keyword_density=0.68,
            pattern_match_strength=0.78,
            biomedical_entity_count=3,
            ambiguity_score=0.22,
            conflict_score=0.18,
            alternative_interpretations=[],
            calculation_time_ms=12.0
        )
        
        mock_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence_metrics=confidence_metrics,
            reasoning=["Export test"],
            research_category="export_test"
        )
        
        enhanced_router.legacy_router.route_query.return_value = mock_prediction
        
        # Generate routing activity
        for i in range(5):
            await enhanced_router.route_query(f"Export test query {i}")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Export comprehensive analytics
        export_file = enhanced_router.export_comprehensive_analytics()
        
        # Verify export file was created
        assert os.path.exists(export_file)
        
        # Verify export content
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        # Verify comprehensive export structure
        assert 'export_timestamp' in export_data
        assert 'export_type' in export_data
        assert export_data['export_type'] == 'comprehensive_analytics'
        assert 'deployment_config' in export_data
        assert 'performance_report' in export_data
        assert 'routing_analytics_summary' in export_data
        assert 'detected_anomalies' in export_data
        
        # Verify deployment config
        deployment_config = export_data['deployment_config']
        assert deployment_config['mode'] == enhanced_router.enhanced_feature_flags.deployment_mode.value
        assert deployment_config['logging_enabled'] == enhanced_router.enhanced_feature_flags.enable_routing_logging
        
        # Cleanup
        os.unlink(export_file)
    
    def test_factory_function_integration(self, temp_log_dir):
        """Test factory function for creating enhanced router"""
        # Test with default parameters
        router = create_enhanced_production_router(
            enable_logging=True,
            log_level="detailed"
        )
        
        assert isinstance(router, EnhancedProductionIntelligentQueryRouter)
        assert router.enhanced_feature_flags.enable_routing_logging
        assert router.enhanced_feature_flags.routing_log_level == LogLevel.DETAILED
        
        # Test with custom parameters
        router2 = create_enhanced_production_router(
            enable_production=True,
            deployment_mode="a_b_testing",
            traffic_percentage=50.0,
            enable_logging=True,
            log_level="debug"
        )
        
        assert router2.enhanced_feature_flags.enable_production_load_balancer
        assert router2.enhanced_feature_flags.deployment_mode == DeploymentMode.A_B_TESTING
        assert router2.enhanced_feature_flags.production_traffic_percentage == 50.0
        assert router2.enhanced_feature_flags.routing_log_level == LogLevel.DEBUG


class TestProductionEnvironmentSimulation:
    """Test production environment simulation scenarios"""
    
    @pytest.fixture
    def production_router_config(self, temp_log_dir):
        """Create production-like router configuration"""
        feature_flags = EnhancedFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION,
            production_traffic_percentage=100.0,
            enable_routing_logging=True,
            routing_log_level=LogLevel.STANDARD,  # Production would use standard, not debug
            routing_storage_strategy=StorageStrategy.FILE_ONLY,  # Production might prefer file storage
            enable_real_time_analytics=True,
            analytics_aggregation_interval_minutes=5,
            enable_anomaly_detection=True,
            enable_performance_impact_monitoring=True,
            max_logging_overhead_ms=5.0,  # Stricter in production
            anonymize_query_content=True,  # Production would anonymize
            hash_sensitive_data=True
        )
        
        logging_config = LoggingConfig(
            enabled=True,
            log_level=LogLevel.STANDARD,
            storage_strategy=StorageStrategy.FILE_ONLY,
            log_directory=temp_log_dir,
            max_file_size_mb=500,  # Larger files in production
            max_files_to_keep=100,  # Keep more files
            compress_old_logs=True,
            async_logging=True,
            batch_size=50,  # Larger batches
            flush_interval_seconds=10,  # Less frequent flushes
            anonymize_queries=True,
            hash_sensitive_data=True
        )
        
        return feature_flags, logging_config
    
    @pytest.fixture
    def mock_production_router(self, production_router_config, temp_log_dir):
        """Create production-configured enhanced router"""
        feature_flags, logging_config = production_router_config
        
        # Mock base router
        base_router = Mock(spec=BiomedicalQueryRouter)
        base_router.route_query = AsyncMock()
        
        router = EnhancedProductionIntelligentQueryRouter(
            base_router=base_router,
            feature_flags=feature_flags,
            logging_config=logging_config
        )
        
        return router
    
    @pytest.mark.asyncio
    async def test_production_load_simulation(self, mock_production_router):
        """Test production load simulation with logging"""
        # Setup varying predictions to simulate production diversity
        predictions = []
        
        # High confidence LIGHTRAG queries (common in production)
        for i in range(30):
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=0.85 + (i % 10) * 0.01,
                research_category_confidence=0.90,
                temporal_analysis_confidence=0.80,
                signal_strength_confidence=0.85,
                context_coherence_confidence=0.82,
                keyword_density=0.75,
                pattern_match_strength=0.85,
                biomedical_entity_count=5,
                ambiguity_score=0.15,
                conflict_score=0.10,
                alternative_interpretations=[],
                calculation_time_ms=10.0 + (i % 5)
            )
            predictions.append(RoutingPrediction(
                routing_decision=RoutingDecision.LIGHTRAG,
                confidence_metrics=confidence_metrics,
                reasoning=["High confidence biomedical query"],
                research_category="biomedical_research"
            ))
        
        # Moderate confidence PERPLEXITY queries
        for i in range(20):
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=0.55 + (i % 10) * 0.02,
                research_category_confidence=0.60,
                temporal_analysis_confidence=0.50,
                signal_strength_confidence=0.55,
                context_coherence_confidence=0.52,
                keyword_density=0.45,
                pattern_match_strength=0.55,
                biomedical_entity_count=2,
                ambiguity_score=0.45,
                conflict_score=0.40,
                alternative_interpretations=[],
                calculation_time_ms=15.0 + (i % 3)
            )
            predictions.append(RoutingPrediction(
                routing_decision=RoutingDecision.PERPLEXITY,
                confidence_metrics=confidence_metrics,
                reasoning=["Moderate confidence, general query"],
                research_category="general_inquiry"
            ))
        
        # Configure mock to return different predictions
        prediction_cycle = iter(predictions)
        mock_production_router.legacy_router.route_query.side_effect = lambda q: next(prediction_cycle, predictions[0])
        
        await mock_production_router.start_monitoring()
        
        try:
            # Simulate production query patterns
            production_queries = [
                "What are the latest treatments for Type 2 diabetes?",
                "How do statins affect cholesterol metabolism?",
                "Clinical trials for Alzheimer's disease treatments",
                "Biomarkers for early cancer detection",
                "Metabolic syndrome risk factors",
                "Gene therapy approaches for rare diseases",
                "Immunotherapy effectiveness in cancer treatment",
                "Drug interactions with common medications",
                "Personalized medicine in cardiology",
                "Stem cell research applications"
            ] * 5  # 50 total queries
            
            start_time = time.time()
            
            # Execute queries with realistic timing
            for i, query in enumerate(production_queries):
                await mock_production_router.route_query(query)
                
                # Simulate realistic query intervals (1-5 seconds between queries)
                if i % 10 == 0:  # Every 10th query, simulate brief pause
                    await asyncio.sleep(0.1)
            
            end_time = time.time()
            
            # Wait for all logging to complete
            await asyncio.sleep(3)
            
            # Verify production performance
            total_time = end_time - start_time
            avg_query_time = total_time / len(production_queries)
            
            assert avg_query_time < 0.1, f"Average query time too high: {avg_query_time:.3f}s"
            assert mock_production_router.total_logged_decisions == len(production_queries)
            
            # Verify analytics collected data
            if mock_production_router.routing_analytics:
                real_time_metrics = mock_production_router.routing_analytics.get_real_time_metrics()
                assert real_time_metrics.get('total_requests', 0) >= len(production_queries)
                
                # Generate analytics report
                analytics_report = mock_production_router.routing_analytics.generate_analytics_report()
                assert analytics_report.total_requests >= len(production_queries)
                
                # Verify decision distribution reflects our test data
                assert 'lightrag' in analytics_report.decision_distribution
                assert 'perplexity' in analytics_report.decision_distribution
                
                # Should have more LIGHTRAG decisions (30 vs 20)
                assert analytics_report.decision_distribution['lightrag'] > analytics_report.decision_distribution['perplexity']
        
        finally:
            await mock_production_router.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_production_anomaly_scenarios(self, mock_production_router):
        """Test anomaly detection in production-like scenarios"""
        await mock_production_router.start_monitoring()
        
        try:
            # Scenario 1: Gradual performance degradation
            degradation_queries = []
            for i in range(25):
                # Gradually decrease confidence and increase response time
                confidence = max(0.2, 0.9 - (i * 0.03))  # Decreasing confidence
                response_time = 8.0 + (i * 2.0)  # Increasing response time
                
                confidence_metrics = ConfidenceMetrics(
                    overall_confidence=confidence,
                    research_category_confidence=confidence,
                    temporal_analysis_confidence=confidence,
                    signal_strength_confidence=confidence,
                    context_coherence_confidence=confidence,
                    keyword_density=confidence * 0.8,
                    pattern_match_strength=confidence,
                    biomedical_entity_count=max(1, int(confidence * 5)),
                    ambiguity_score=1 - confidence,
                    conflict_score=1 - confidence,
                    alternative_interpretations=[],
                    calculation_time_ms=response_time
                )
                
                prediction = RoutingPrediction(
                    routing_decision=RoutingDecision.PERPLEXITY if confidence < 0.6 else RoutingDecision.LIGHTRAG,
                    confidence_metrics=confidence_metrics,
                    reasoning=[f"Degradation test {i}"],
                    research_category="degradation_test"
                )
                
                mock_production_router.legacy_router.route_query.return_value = prediction
                await mock_production_router.route_query(f"Degradation query {i}")
                
                # Small delay to simulate realistic timing
                await asyncio.sleep(0.01)
            
            # Wait for anomaly detection
            await asyncio.sleep(3)
            
            # Verify anomalies were detected
            if mock_production_router.routing_analytics:
                anomalies = mock_production_router.routing_analytics.detect_anomalies()
                
                # Should detect multiple anomaly types
                anomaly_types = {anomaly.get('type') for anomaly in anomalies}
                assert 'confidence_degradation' in anomaly_types
                assert 'slow_decisions' in anomaly_types
                
                # Verify anomalies are tracked by router
                assert len(mock_production_router.detected_anomalies) >= len(anomalies)
        
        finally:
            await mock_production_router.stop_monitoring()
    
    def test_production_configuration_validation(self, mock_production_router):
        """Test production configuration meets requirements"""
        config = mock_production_router.enhanced_feature_flags
        
        # Verify production settings
        assert config.deployment_mode == DeploymentMode.PRODUCTION
        assert config.production_traffic_percentage == 100.0
        assert config.anonymize_query_content is True  # Privacy requirement
        assert config.hash_sensitive_data is True
        assert config.max_logging_overhead_ms <= 5.0  # Performance requirement
        
        # Verify logging configuration
        logging_config = mock_production_router.routing_logger.config
        assert logging_config.anonymize_queries is True
        assert logging_config.compress_old_logs is True
        assert logging_config.async_logging is True
        assert logging_config.batch_size >= 50  # Efficient batching
    
    @pytest.mark.asyncio
    async def test_production_error_recovery(self, mock_production_router):
        """Test error recovery in production environment"""
        await mock_production_router.start_monitoring()
        
        try:
            # Test various error scenarios
            error_scenarios = [
                Exception("Database connection timeout"),
                ValueError("Invalid query format"),
                RuntimeError("Service temporarily unavailable"),
                TimeoutError("Request timeout"),
                ConnectionError("Network unavailable")
            ]
            
            for i, error in enumerate(error_scenarios):
                mock_production_router.legacy_router.route_query.side_effect = error
                
                # Should handle error gracefully
                with pytest.raises(type(error)):
                    await mock_production_router.route_query(f"Error scenario {i}")
                
                # Wait for error logging
                await asyncio.sleep(0.1)
            
            # Reset mock for normal operation
            normal_prediction = RoutingPrediction(
                routing_decision=RoutingDecision.LIGHTRAG,
                confidence_metrics=ConfidenceMetrics(
                    overall_confidence=0.85,
                    research_category_confidence=0.85,
                    temporal_analysis_confidence=0.85,
                    signal_strength_confidence=0.85,
                    context_coherence_confidence=0.85,
                    keyword_density=0.8,
                    pattern_match_strength=0.85,
                    biomedical_entity_count=4,
                    ambiguity_score=0.15,
                    conflict_score=0.15,
                    alternative_interpretations=[],
                    calculation_time_ms=12.0
                ),
                reasoning=["Recovery test"],
                research_category="recovery"
            )
            
            mock_production_router.legacy_router.route_query.side_effect = None
            mock_production_router.legacy_router.route_query.return_value = normal_prediction
            
            # Should recover and work normally
            result = await mock_production_router.route_query("Recovery test query")
            assert result.routing_decision == RoutingDecision.LIGHTRAG
            
            # Wait for processing
            await asyncio.sleep(1)
            
            # Verify all requests were logged (including errors)
            assert mock_production_router.total_logged_decisions == len(error_scenarios) + 1
        
        finally:
            await mock_production_router.stop_monitoring()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-x"])