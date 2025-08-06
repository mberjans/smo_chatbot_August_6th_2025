#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Budget Management System.

This test suite provides complete end-to-end integration testing of the budget management
ecosystem, validating the interaction between all components:

- Cost Persistence ↔ Budget Manager ↔ Alert System
- API Metrics Logger ↔ Cost Tracking ↔ Research Categorizer  
- Real-time Monitoring ↔ Circuit Breaker ↔ Dashboard API
- Audit Trail ↔ Compliance ↔ Alert Escalation
- Concurrent Operations ↔ Performance ↔ System Health

Integration Scenarios:
- Complete cost-to-alert workflow
- Multi-component budget monitoring
- Real-time threshold management
- Cross-system data consistency
- Error propagation and recovery
- Performance under load

Author: Claude Code (Anthropic)
Created: August 6, 2025
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

# Test imports - all integrated components
from lightrag_integration.cost_persistence import (
    CostPersistence,
    CostRecord,
    ResearchCategory
)
from lightrag_integration.budget_manager import (
    BudgetManager,
    BudgetAlert,
    AlertLevel,
    BudgetThreshold
)
from lightrag_integration.api_metrics_logger import (
    APIUsageMetricsLogger,
    APIMetric,
    MetricType
)
from lightrag_integration.research_categorizer import ResearchCategorizer
from lightrag_integration.audit_trail import AuditTrail, AuditEventType
from lightrag_integration.alert_system import (
    AlertNotificationSystem,
    AlertEscalationManager,
    AlertConfig,
    AlertChannel
)


class TestIntegratedBudgetManagementWorkflow:
    """Test complete integrated workflow from cost recording to alert delivery."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for integration testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield Path(db_path)
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def integrated_system(self, temp_db_path):
        """Create fully integrated budget management system."""
        # Initialize core persistence layer
        cost_persistence = CostPersistence(temp_db_path, retention_days=365)
        
        # Initialize budget manager with thresholds
        custom_thresholds = BudgetThreshold(
            warning_percentage=70.0,
            critical_percentage=85.0,
            exceeded_percentage=100.0
        )
        
        budget_manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0,
            thresholds=custom_thresholds
        )
        
        # Initialize supporting systems
        research_categorizer = ResearchCategorizer()
        audit_trail = AuditTrail(db_path=temp_db_path, retention_days=365)
        
        # Initialize alert system
        alert_config = AlertConfig(
            enabled_channels={AlertChannel.LOGGING, AlertChannel.CONSOLE},
            rate_limit_window=60.0,
            max_alerts_per_window=10,
            dedupe_window=30.0
        )
        alert_system = AlertNotificationSystem(alert_config)
        escalation_manager = AlertEscalationManager(alert_system)
        
        # Set up budget manager alert callback
        def alert_callback(alert):
            escalation_manager.process_alert(alert)
        
        budget_manager.alert_callback = alert_callback
        
        # Initialize API metrics logger with full integration
        mock_config = Mock()
        mock_config.enable_file_logging = False  # Disable for test speed
        
        api_metrics_logger = APIUsageMetricsLogger(
            config=mock_config,
            cost_persistence=cost_persistence,
            budget_manager=budget_manager,
            research_categorizer=research_categorizer,
            audit_trail=audit_trail
        )
        
        return {
            'cost_persistence': cost_persistence,
            'budget_manager': budget_manager,
            'research_categorizer': research_categorizer,
            'audit_trail': audit_trail,
            'alert_system': alert_system,
            'escalation_manager': escalation_manager,
            'api_metrics_logger': api_metrics_logger
        }
    
    def test_complete_cost_to_alert_workflow(self, integrated_system):
        """Test complete workflow from API call to budget alert delivery."""
        components = integrated_system
        
        # Simulate API usage that should trigger alerts
        with components['api_metrics_logger'].track_api_call(
            "metabolite_identification_query",
            "gpt-4o",
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION.value
        ) as tracker:
            
            tracker.set_tokens(prompt=2000, completion=1500)
            tracker.set_cost(0.75)  # 75% of daily budget
            tracker.set_response_details(response_time_ms=2500)
            tracker.add_metadata("complexity", "high")
            tracker.metric.user_id = "researcher_001"
            tracker.metric.project_id = "metabolomics_project"
        
        # Verify cost was recorded
        cost_records = components['cost_persistence'].db.get_cost_records(limit=1)
        assert len(cost_records) == 1
        cost_record = cost_records[0]
        assert cost_record.cost_usd == 0.75
        assert cost_record.research_category == ResearchCategory.METABOLITE_IDENTIFICATION.value
        
        # Verify API metrics were logged
        metrics_buffer = components['api_metrics_logger'].metrics_aggregator._metrics_buffer
        assert len(metrics_buffer) == 1
        metric = metrics_buffer[0]
        assert metric.cost_usd == 0.75
        assert metric.total_tokens == 3500
        
        # Verify budget status shows warning level (75% >= 70% warning threshold)
        budget_summary = components['budget_manager'].get_budget_summary()
        assert budget_summary['daily_budget']['percentage_used'] >= 75.0
        assert budget_summary['budget_health'] in ['warning', 'critical']
        
        # Verify audit events were recorded
        audit_events = components['audit_trail'].get_audit_log(limit=5)
        assert len(audit_events) >= 1
        
        # Look for API usage audit event
        api_usage_events = [
            event for event in audit_events 
            if hasattr(event, 'event_type') and event.event_type == AuditEventType.API_CALL
        ]
        assert len(api_usage_events) >= 1
    
    def test_budget_threshold_escalation_integration(self, integrated_system):
        """Test budget threshold escalation across multiple components."""
        components = integrated_system
        
        # Add costs to approach different threshold levels
        cost_scenarios = [
            (0.50, "approach_warning"),      # 50% - below warning
            (0.25, "reach_warning"),         # 75% total - at warning  
            (0.12, "reach_critical"),        # 87% total - at critical
            (0.15, "exceed_budget")          # 102% total - exceeded
        ]
        
        for cost_amount, scenario_name in cost_scenarios:
            with components['api_metrics_logger'].track_api_call(
                f"budget_escalation_{scenario_name}",
                "gpt-4o-mini"
            ) as tracker:
                
                tracker.set_tokens(prompt=int(cost_amount * 2000), completion=int(cost_amount * 1000))
                tracker.set_cost(cost_amount)
            
            # Check budget status after each addition
            budget_status = components['budget_manager'].check_budget_status(
                cost_amount=0.01,  # Small additional cost to trigger check
                operation_type=f"check_after_{scenario_name}"
            )
            
            # Verify appropriate response based on cumulative cost
            total_cost = sum(cost for cost, _ in cost_scenarios[:cost_scenarios.index((cost_amount, scenario_name)) + 1])
            expected_percentage = total_cost * 100
            
            if expected_percentage >= 100:
                assert budget_status['budget_health'] == 'exceeded'
                assert budget_status['operation_allowed'] is False
            elif expected_percentage >= 85:
                assert budget_status['budget_health'] == 'critical'
                assert budget_status['operation_allowed'] is True
            elif expected_percentage >= 70:
                assert budget_status['budget_health'] == 'warning'
                assert budget_status['operation_allowed'] is True
    
    def test_research_categorization_cost_analysis_integration(self, integrated_system):
        """Test integration between research categorization and cost analysis."""
        components = integrated_system
        
        # Test different research categories with varying costs
        research_scenarios = [
            (ResearchCategory.METABOLITE_IDENTIFICATION, "identify unknown metabolite", 0.15, 300),
            (ResearchCategory.PATHWAY_ANALYSIS, "analyze KEGG pathway", 0.12, 250),
            (ResearchCategory.BIOMARKER_DISCOVERY, "find disease biomarkers", 0.18, 350),
            (ResearchCategory.DRUG_DISCOVERY, "screen compound library", 0.22, 400),
            (ResearchCategory.CLINICAL_DIAGNOSIS, "interpret clinical metabolomics", 0.14, 280)
        ]
        
        for category, query_text, cost, tokens in research_scenarios:
            with components['api_metrics_logger'].track_api_call(
                f"research_integration_{category.value}",
                "gpt-4o-mini"
            ) as tracker:
                
                tracker.set_tokens(prompt=tokens//2, completion=tokens//2)
                tracker.set_cost(cost)
                tracker.metric.query_type = query_text
                tracker.metric.research_category = category.value
        
        # Analyze research category costs
        research_analysis = components['cost_persistence'].get_research_analysis(days=1)
        
        assert research_analysis['total_records'] >= 5
        assert len(research_analysis['categories']) >= 5
        
        # Verify each category appears in analysis
        for category, _, cost, _ in research_scenarios:
            assert category.value in research_analysis['categories']
            category_data = research_analysis['categories'][category.value]
            assert category_data['total_cost'] >= cost * 0.9  # Allow for small precision differences
    
    def test_concurrent_operations_with_budget_monitoring(self, integrated_system):
        """Test concurrent operations with integrated budget monitoring."""
        components = integrated_system
        num_workers = 8
        operations_per_worker = 10
        
        def concurrent_worker(worker_id):
            worker_results = []
            for i in range(operations_per_worker):
                try:
                    with components['api_metrics_logger'].track_api_call(
                        f"concurrent_worker_{worker_id}_op_{i}",
                        "gpt-4o-mini"
                    ) as tracker:
                        
                        # Varying cost and complexity
                        base_cost = 0.03 + (worker_id * 0.005)
                        tracker.set_tokens(
                            prompt=100 + worker_id * 20,
                            completion=50 + i * 5
                        )
                        tracker.set_cost(base_cost)
                        tracker.metric.research_category = list(ResearchCategory)[
                            (worker_id + i) % len(ResearchCategory)
                        ].value
                    
                    # Check budget status periodically
                    if i % 3 == 0:  # Every 3rd operation
                        budget_status = components['budget_manager'].check_budget_status(
                            cost_amount=0.01,
                            operation_type=f"concurrent_check_{worker_id}_{i}"
                        )
                        worker_results.append(budget_status)
                        
                except Exception as e:
                    print(f"Worker {worker_id} error: {e}")
                    worker_results.append({'error': str(e)})
            
            return worker_results
        
        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(concurrent_worker, i) for i in range(num_workers)]
            results = [future.result() for future in futures]
        
        # Verify concurrent operations completed successfully
        total_results = sum(len(worker_results) for worker_results in results)
        assert total_results > 0
        
        # Verify system integrity after concurrent operations
        final_summary = components['budget_manager'].get_budget_summary()
        assert final_summary['daily_budget']['total_cost'] > 0
        
        # Verify all operations were tracked
        total_expected_operations = num_workers * operations_per_worker
        metrics_logged = len(components['api_metrics_logger'].metrics_aggregator._metrics_buffer)
        assert metrics_logged >= total_expected_operations * 0.95  # Allow for some failures
    
    def test_alert_system_integration_with_escalation(self, integrated_system):
        """Test alert system integration with escalation management."""
        components = integrated_system
        
        # Generate increasing costs to trigger multiple alert levels
        cost_progression = [0.30, 0.25, 0.20, 0.18, 0.15]  # Cumulative: 30%, 55%, 75%, 93%, 108%
        
        for i, cost in enumerate(cost_progression):
            with components['api_metrics_logger'].track_api_call(
                f"escalation_test_step_{i}",
                "gpt-4o"
            ) as tracker:
                
                tracker.set_tokens(prompt=int(cost * 1500), completion=int(cost * 1000))
                tracker.set_cost(cost)
                tracker.metric.user_id = "escalation_tester"
            
            # Small delay to ensure timestamp ordering
            time.sleep(0.1)
        
        # Verify escalation manager has history
        escalation_status = components['escalation_manager'].get_escalation_status()
        assert escalation_status['active_escalation_keys'] > 0
        
        # Verify alert delivery statistics
        alert_stats = components['alert_system'].get_delivery_stats()
        assert alert_stats['channels']['logging']['total_attempts'] > 0
    
    def test_audit_trail_integration_across_components(self, integrated_system):
        """Test audit trail integration across all components."""
        components = integrated_system
        
        # Perform operations that should generate audit events
        with components['api_metrics_logger'].track_api_call(
            "audit_integration_test",
            "gpt-4o"
        ) as tracker:
            
            tracker.set_tokens(prompt=500, completion=300)
            tracker.set_cost(0.20)
            tracker.metric.user_id = "audit_user"
            tracker.metric.project_id = "audit_project"
            tracker.metric.compliance_level = "high"
        
        # Generate a manual system event
        components['audit_trail'].record_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            event_data={
                "error_type": "integration_test",
                "component": "budget_management", 
                "severity": "low"
            },
            user_id="system",
            session_id="integration_test_session"
        )
        
        # Verify comprehensive audit log
        audit_events = components['audit_trail'].get_audit_log(limit=10)
        assert len(audit_events) >= 2
        
        # Verify different event types are recorded
        event_types = set()
        for event in audit_events:
            if hasattr(event, 'event_type'):
                event_types.add(event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type))
        
        # Should have at least API usage and system events
        expected_types = {'api_call', 'system_error', 'api_usage'}
        overlap = event_types & expected_types
        assert len(overlap) >= 1
    
    def test_data_consistency_across_components(self, integrated_system):
        """Test data consistency across all integrated components."""
        components = integrated_system
        
        # Record a significant API operation
        operation_cost = 0.45
        operation_tokens = 2000
        
        with components['api_metrics_logger'].track_api_call(
            "consistency_test",
            "gpt-4o",
            research_category=ResearchCategory.PATHWAY_ANALYSIS.value
        ) as tracker:
            
            tracker.set_tokens(prompt=1200, completion=800)
            tracker.set_cost(operation_cost)
            tracker.set_response_details(response_time_ms=3000)
            tracker.metric.user_id = "consistency_user"
            tracker.metric.session_id = "consistency_session"
        
        # Verify data consistency across all components
        
        # 1. Cost Persistence
        cost_records = components['cost_persistence'].db.get_cost_records(limit=1)
        assert len(cost_records) == 1
        cost_record = cost_records[0]
        assert cost_record.cost_usd == operation_cost
        assert cost_record.total_tokens == operation_tokens
        assert cost_record.research_category == ResearchCategory.PATHWAY_ANALYSIS.value
        
        # 2. API Metrics Logger
        metrics = components['api_metrics_logger'].metrics_aggregator._metrics_buffer
        assert len(metrics) >= 1
        metric = metrics[-1]  # Get last metric
        assert metric.cost_usd == operation_cost
        assert metric.total_tokens == operation_tokens
        
        # 3. Budget Manager
        budget_summary = components['budget_manager'].get_budget_summary()
        assert budget_summary['daily_budget']['total_cost'] >= operation_cost
        
        # 4. Research Analysis
        research_analysis = components['cost_persistence'].get_research_analysis(days=1)
        pathway_data = research_analysis['categories'].get(ResearchCategory.PATHWAY_ANALYSIS.value)
        if pathway_data:  # Might be aggregated with other operations
            assert pathway_data['total_cost'] >= operation_cost
        
        # 5. Audit Trail
        audit_events = components['audit_trail'].get_audit_log(limit=3)
        api_events = [e for e in audit_events if hasattr(e, 'user_id') and e.user_id == "consistency_user"]
        assert len(api_events) >= 1
    
    def test_error_handling_and_system_resilience(self, integrated_system):
        """Test error handling and system resilience across components."""
        components = integrated_system
        
        # Test with component failures
        original_audit_trail = components['api_metrics_logger'].audit_trail
        
        # Temporarily disable audit trail to test resilience
        components['api_metrics_logger'].audit_trail = None
        
        # System should continue to function
        with components['api_metrics_logger'].track_api_call(
            "resilience_test",
            "gpt-4o-mini"
        ) as tracker:
            
            tracker.set_tokens(prompt=200, completion=100)
            tracker.set_cost(0.08)
        
        # Verify operation still completed
        assert len(components['api_metrics_logger'].metrics_aggregator._metrics_buffer) >= 1
        
        # Restore audit trail
        components['api_metrics_logger'].audit_trail = original_audit_trail
        
        # Test with invalid data
        try:
            with components['api_metrics_logger'].track_api_call(
                "error_test", 
                "invalid-model"
            ) as tracker:
                
                tracker.set_tokens(prompt=-100, completion=-50)  # Invalid negative tokens
                tracker.set_cost(-0.05)  # Invalid negative cost
        except:
            pass  # Errors are acceptable here
        
        # System should recover
        with components['api_metrics_logger'].track_api_call(
            "recovery_test",
            "gpt-4o-mini"
        ) as tracker:
            
            tracker.set_tokens(prompt=150, completion=75)
            tracker.set_cost(0.06)
        
        # Verify system recovery
        recent_metrics = components['api_metrics_logger'].metrics_aggregator._metrics_buffer[-1]
        assert recent_metrics.operation_name == "recovery_test"
    
    def test_performance_under_integrated_load(self, integrated_system):
        """Test system performance under integrated load."""
        components = integrated_system
        
        start_time = time.time()
        num_operations = 100
        
        # Perform many integrated operations
        for i in range(num_operations):
            with components['api_metrics_logger'].track_api_call(
                f"performance_test_{i}",
                "gpt-4o-mini"
            ) as tracker:
                
                tracker.set_tokens(prompt=100 + i, completion=50 + i//2)
                tracker.set_cost(0.05 + i * 0.001)
                tracker.metric.research_category = list(ResearchCategory)[i % len(ResearchCategory)].value
                
                # Periodically check budget (adds system load)
                if i % 10 == 0:
                    components['budget_manager'].check_budget_status(
                        cost_amount=0.01,
                        operation_type=f"perf_check_{i}"
                    )
        
        end_time = time.time()
        total_time = end_time - start_time
        operations_per_second = num_operations / total_time
        
        # System should maintain reasonable performance
        assert operations_per_second > 10  # At least 10 ops/sec with full integration
        
        # Verify all components recorded the operations
        assert len(components['api_metrics_logger'].metrics_aggregator._metrics_buffer) >= num_operations
        
        final_budget = components['budget_manager'].get_budget_summary()
        assert final_budget['daily_budget']['record_count'] >= num_operations
        
        performance_summary = components['api_metrics_logger'].get_performance_summary()
        assert performance_summary['current_day']['total_calls'] >= num_operations
        
        print(f"Performance test: {num_operations} operations in {total_time:.2f}s = {operations_per_second:.1f} ops/sec")


class TestCrossComponentDataFlow:
    """Test data flow and dependencies between components."""
    
    @pytest.fixture
    def isolated_components(self):
        """Create components in isolation for dependency testing."""
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = Path(temp_db.name)
        temp_db.close()
        
        try:
            cost_persistence = CostPersistence(db_path)
            budget_manager = BudgetManager(cost_persistence=cost_persistence)
            research_categorizer = ResearchCategorizer()
            audit_trail = AuditTrail(db_path=db_path)
            
            yield {
                'cost_persistence': cost_persistence,
                'budget_manager': budget_manager,
                'research_categorizer': research_categorizer,
                'audit_trail': audit_trail,
                'db_path': db_path
            }
        finally:
            db_path.unlink(missing_ok=True)
    
    def test_cost_persistence_to_budget_manager_flow(self, isolated_components):
        """Test data flow from cost persistence to budget manager."""
        cost_persistence = isolated_components['cost_persistence']
        budget_manager = isolated_components['budget_manager']
        
        # Record costs directly in persistence layer
        costs = [25.0, 30.0, 20.0]  # Total: 75.0
        for i, cost in enumerate(costs):
            cost_persistence.record_cost(
                cost_usd=cost,
                operation_type=f"direct_record_{i}",
                model_name="test-model",
                token_usage={"prompt_tokens": 100}
            )
        
        # Budget manager should see accumulated costs
        daily_status = cost_persistence.get_daily_budget_status(budget_limit=100.0)
        assert daily_status['total_cost'] >= 75.0
        assert daily_status['percentage_used'] >= 75.0
        
        # Budget manager should use this data
        budget_summary = budget_manager.get_budget_summary()
        assert budget_summary['daily_budget']['total_cost'] >= 75.0
    
    def test_research_categorizer_integration_flow(self, isolated_components):
        """Test research categorizer integration with other components."""
        research_categorizer = isolated_components['research_categorizer']
        
        # Test categorization of different query types
        test_queries = [
            ("What is the structure of this metabolite?", ResearchCategory.METABOLITE_IDENTIFICATION),
            ("Analyze KEGG pathway for glucose metabolism", ResearchCategory.PATHWAY_ANALYSIS), 
            ("Find biomarkers for diabetes", ResearchCategory.BIOMARKER_DISCOVERY),
            ("Screen compounds for drug activity", ResearchCategory.DRUG_DISCOVERY)
        ]
        
        for query, expected_category in test_queries:
            prediction = research_categorizer.categorize_query(query)
            
            # Verification depends on categorizer implementation
            # At minimum, should return a CategoryPrediction object
            assert hasattr(prediction, 'category')
            assert hasattr(prediction, 'confidence')
            assert hasattr(prediction, 'evidence')
            
            # For this test, we verify the system can categorize
            assert prediction.confidence >= 0.0
    
    def test_audit_trail_cross_component_events(self, isolated_components):
        """Test audit trail recording events from multiple components."""
        audit_trail = isolated_components['audit_trail']
        
        # Record events from different "components"
        component_events = [
            (AuditEventType.API_CALL, {"component": "api_metrics", "operation": "llm_call"}),
            (AuditEventType.BUDGET_ALERT, {"component": "budget_manager", "alert_level": "warning"}),
            (AuditEventType.DATA_MODIFICATION, {"component": "cost_persistence", "table": "cost_records"}),
            (AuditEventType.SYSTEM_ERROR, {"component": "circuit_breaker", "error": "rate_limit"})
        ]
        
        for event_type, event_data in component_events:
            audit_trail.record_event(
                event_type=event_type,
                event_data=event_data,
                user_id="integration_test",
                session_id="cross_component_session"
            )
        
        # Verify all events recorded
        audit_events = audit_trail.get_audit_log(limit=10)
        assert len(audit_events) >= 4
        
        # Verify different components represented
        components_seen = set()
        for event in audit_events:
            if hasattr(event, 'event_data') and 'component' in event.event_data:
                components_seen.add(event.event_data['component'])
        
        expected_components = {'api_metrics', 'budget_manager', 'cost_persistence', 'circuit_breaker'}
        assert components_seen == expected_components


class TestSystemHealthAndMonitoring:
    """Test system health monitoring across integrated components."""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create system for health monitoring testing."""
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = Path(temp_db.name)
        temp_db.close()
        
        try:
            # Create components with health monitoring focus
            cost_persistence = CostPersistence(db_path, retention_days=30)
            
            budget_manager = BudgetManager(
                cost_persistence=cost_persistence,
                daily_budget_limit=50.0,  # Lower limit for testing
                monthly_budget_limit=1500.0
            )
            
            mock_config = Mock()
            mock_config.enable_file_logging = False
            
            api_metrics_logger = APIUsageMetricsLogger(
                config=mock_config,
                cost_persistence=cost_persistence,
                budget_manager=budget_manager
            )
            
            yield {
                'cost_persistence': cost_persistence,
                'budget_manager': budget_manager,
                'api_metrics_logger': api_metrics_logger,
                'db_path': db_path
            }
        finally:
            db_path.unlink(missing_ok=True)
    
    def test_system_health_indicators(self, monitoring_system):
        """Test system health indicators across components."""
        components = monitoring_system
        
        # Generate some system activity
        for i in range(5):
            with components['api_metrics_logger'].track_api_call(
                f"health_test_{i}",
                "gpt-4o-mini"
            ) as tracker:
                
                tracker.set_tokens(prompt=100 + i * 20, completion=50 + i * 10)
                tracker.set_cost(0.05 + i * 0.01)
        
        # Check various health indicators
        
        # 1. API Metrics Logger Health
        performance_summary = components['api_metrics_logger'].get_performance_summary()
        assert 'system' in performance_summary
        assert performance_summary['system']['active_operations'] == 0  # All completed
        assert performance_summary['system']['memory_usage_mb'] > 0
        
        # 2. Budget Manager Health  
        budget_summary = components['budget_manager'].get_budget_summary()
        assert budget_summary['budget_health'] in ['healthy', 'warning', 'critical', 'exceeded']
        assert budget_summary['daily_budget']['record_count'] >= 5
        
        # 3. Cost Persistence Health
        daily_status = components['cost_persistence'].get_daily_budget_status()
        assert daily_status['total_cost'] > 0
        assert daily_status['record_count'] >= 5
    
    def test_system_performance_monitoring(self, monitoring_system):
        """Test system performance monitoring capabilities."""
        components = monitoring_system
        
        # Generate load to test performance monitoring
        start_time = time.time()
        num_operations = 50
        
        for i in range(num_operations):
            with components['api_metrics_logger'].track_api_call(
                f"perf_monitor_{i}",
                "gpt-4o-mini"
            ) as tracker:
                
                tracker.set_tokens(prompt=80 + i, completion=40 + i//2)
                tracker.set_cost(0.04 + i * 0.001)
                tracker.set_response_details(response_time_ms=800 + i * 10)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check performance metrics
        performance_summary = components['api_metrics_logger'].get_performance_summary()
        
        current_hour = performance_summary['current_hour']
        assert current_hour['total_calls'] >= num_operations
        assert current_hour['avg_response_time_ms'] > 0
        
        # Performance should be reasonable
        ops_per_second = num_operations / total_time
        assert ops_per_second > 5  # Should handle at least 5 ops/sec
        
        print(f"Performance monitoring: {ops_per_second:.1f} ops/sec")
    
    def test_error_rate_monitoring(self, monitoring_system):
        """Test error rate monitoring across system."""
        components = monitoring_system
        
        # Generate mix of successful and failed operations
        total_operations = 20
        failed_operations = 5
        
        # Successful operations
        for i in range(total_operations - failed_operations):
            with components['api_metrics_logger'].track_api_call(
                f"success_{i}",
                "gpt-4o-mini"
            ) as tracker:
                
                tracker.set_tokens(prompt=100, completion=50)
                tracker.set_cost(0.05)
        
        # Failed operations
        for i in range(failed_operations):
            with components['api_metrics_logger'].track_api_call(
                f"error_{i}",
                "gpt-4o-mini"
            ) as tracker:
                
                tracker.set_tokens(prompt=100, completion=0)
                tracker.set_cost(0.0)
                tracker.set_error("TestError", f"Simulated error {i}")
        
        # Check error rate monitoring
        performance_summary = components['api_metrics_logger'].get_performance_summary()
        
        current_hour = performance_summary['current_hour']
        assert current_hour['error_count'] >= failed_operations
        
        expected_error_rate = (failed_operations / total_operations) * 100
        actual_error_rate = current_hour['error_rate_percent']
        
        # Allow some tolerance for timing differences
        assert abs(actual_error_rate - expected_error_rate) < 10


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])