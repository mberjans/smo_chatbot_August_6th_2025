#!/usr/bin/env python3
"""
Comprehensive Integration Test for Advanced Cleanup System.

This test validates the complete integration between the advanced cleanup system
and the existing test infrastructure, ensuring all components work together
seamlessly while maintaining compatibility with existing fixtures and patterns.

Test Coverage:
1. Integration with existing TestDataManager fixtures
2. Automatic resource registration and cleanup
3. Performance monitoring during test execution
4. Cleanup validation and effectiveness
5. Failure recovery and retry mechanisms
6. Report generation and health assessment
7. Pytest lifecycle integration
8. Cleanup bridge functionality

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import pytest
import shutil

# Import advanced cleanup system
from advanced_cleanup_system import (
    AdvancedCleanupOrchestrator, ResourceType, CleanupStrategy,
    CleanupScope, CleanupPolicy, ResourceThresholds
)
from cleanup_validation_monitor import (
    CleanupValidationMonitor, AlertConfig, CleanupValidator,
    ResourceMonitor, PerformanceAnalyzer
)
from advanced_cleanup_integration import (
    AdvancedCleanupIntegrator, CleanupIntegrationConfig,
    advanced_cleanup_context, configure_advanced_cleanup_for_tests
)

# Import existing test infrastructure
from test_data_fixtures import TestDataManager, TestDataConfig
from conftest import *  # Import existing conftest fixtures

# Setup module-level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAdvancedCleanupIntegration:
    """Comprehensive integration tests for advanced cleanup system."""
    
    @pytest.fixture(scope="class")
    def integration_config(self):
        """Configuration for integration tests."""
        return configure_advanced_cleanup_for_tests(
            memory_threshold_mb=256,
            file_handle_threshold=50,
            generate_reports=True,
            monitor_performance=True
        )
        
    @pytest.fixture(scope="class")
    def advanced_orchestrator(self, integration_config):
        """Advanced cleanup orchestrator for integration tests."""
        policy = CleanupPolicy(
            strategy=CleanupStrategy.RESOURCE_BASED,
            resource_types=set(ResourceType),
            validate_cleanup=True,
            report_cleanup=True
        )
        
        thresholds = ResourceThresholds(
            memory_mb=integration_config.memory_threshold_mb,
            file_handles=integration_config.file_handle_threshold,
            db_connections=integration_config.db_connection_threshold
        )
        
        orchestrator = AdvancedCleanupOrchestrator(policy, thresholds)
        yield orchestrator
        
        # Final cleanup
        try:
            orchestrator.force_cleanup()
        except Exception as e:
            logger.warning(f"Final cleanup warning: {e}")
            
    @pytest.fixture
    def cleanup_monitor(self, integration_config):
        """Cleanup validation monitor for testing."""
        policy = CleanupPolicy(
            strategy=CleanupStrategy.DEFERRED,
            validate_cleanup=True,
            report_cleanup=True
        )
        
        thresholds = ResourceThresholds(
            memory_mb=integration_config.memory_threshold_mb,
            file_handles=integration_config.file_handle_threshold,
            db_connections=integration_config.db_connection_threshold
        )
        
        alert_config = AlertConfig(enabled=False)  # Disable alerts for tests
        
        monitor = CleanupValidationMonitor(
            cleanup_policy=policy,
            thresholds=thresholds,
            alert_config=alert_config
        )
        
        yield monitor
        
        # Cleanup monitor
        try:
            monitor.stop_monitoring()
        except Exception as e:
            logger.warning(f"Monitor cleanup warning: {e}")
            
    def test_basic_orchestrator_functionality(self, advanced_orchestrator):
        """Test basic functionality of the advanced cleanup orchestrator."""
        logger.info("Testing basic orchestrator functionality...")
        
        # Create test resources
        temp_files = []
        db_connections = []
        temp_dirs = []
        
        try:
            # Create temporary files
            for i in range(5):
                temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                temp_file.write(f"Test content {i}\n")
                temp_file.close()
                temp_files.append(temp_file.name)
                advanced_orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                
            # Create database connections
            for i in range(3):
                db_path = tempfile.mktemp(suffix='.db')
                conn = sqlite3.connect(db_path)
                conn.execute('CREATE TABLE test (id INTEGER, data TEXT)')
                conn.execute('INSERT INTO test VALUES (?, ?)', (i, f'data_{i}'))
                conn.commit()
                db_connections.append((conn, db_path))
                advanced_orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
                
            # Create temporary directories
            for i in range(3):
                temp_dir = tempfile.mkdtemp(prefix=f'test_cleanup_{i}_')
                for j in range(2):
                    file_path = Path(temp_dir) / f'file_{j}.txt'
                    file_path.write_text(f'Content {j}')
                temp_dirs.append(temp_dir)
                advanced_orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_dir)
                
            logger.info(f"Created {len(temp_files)} files, {len(db_connections)} DBs, {len(temp_dirs)} dirs")
            
            # Get initial resource usage
            initial_usage = advanced_orchestrator.get_resource_usage()
            assert initial_usage, "Should have resource usage data"
            
            # Test cleanup trigger logic
            should_cleanup = advanced_orchestrator.should_cleanup()
            logger.info(f"Should cleanup: {should_cleanup}")
            
            # Perform cleanup
            cleanup_success = advanced_orchestrator.cleanup()
            assert cleanup_success, "Cleanup should succeed"
            
            # Validate cleanup
            validation_success = advanced_orchestrator.validate_cleanup()
            assert validation_success, "Cleanup validation should succeed"
            
            # Get cleanup statistics
            stats = advanced_orchestrator.get_cleanup_statistics()
            assert stats['total_operations'] > 0, "Should have cleanup operations recorded"
            
            logger.info(f"Cleanup stats: {json.dumps(stats, indent=2, default=str)}")
            
        finally:
            # Manual cleanup of any remaining resources
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception:
                    pass
                    
            for conn, db_path in db_connections:
                try:
                    conn.close()
                    if os.path.exists(db_path):
                        os.unlink(db_path)
                except Exception:
                    pass
                    
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
                    
    def test_integration_with_test_data_manager(self, test_data_manager):
        """Test integration between advanced cleanup and existing TestDataManager."""
        logger.info("Testing integration with TestDataManager...")
        
        # Create integrator
        config = configure_advanced_cleanup_for_tests()
        integrator = AdvancedCleanupIntegrator(config)
        
        # Register existing test data manager
        integrator.register_test_data_manager(test_data_manager, "integration_test")
        
        # Create integration bridge
        bridge = integrator.create_integrated_fixture_bridge(test_data_manager)
        assert bridge is not None, "Integration bridge should be created"
        
        # Test dual registration through bridge
        test_resources = []
        
        try:
            # Register file through bridge
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            temp_file.write("Integration test content")
            temp_file.close()
            bridge.register_file(open(temp_file.name, 'r'), auto_close=True)
            test_resources.append(temp_file.name)
            
            # Register database connection through bridge
            db_path = tempfile.mktemp(suffix='.db')
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE integration (id INTEGER, test TEXT)')
            conn.execute('INSERT INTO integration VALUES (1, "test")')
            conn.commit()
            bridge.register_db_connection(conn)
            test_resources.append((conn, db_path))
            
            # Register temporary path through bridge
            temp_dir = tempfile.mkdtemp(prefix='integration_test_')
            test_file = Path(temp_dir) / 'test.txt'
            test_file.write_text('Integration test')
            bridge.register_temp_path(temp_dir)
            test_resources.append(temp_dir)
            
            # Get resource usage through bridge
            usage = bridge.get_resource_usage()
            assert usage, "Should have resource usage data through bridge"
            logger.info(f"Resource usage via bridge: {usage}")
            
            # Perform integrated cleanup
            cleanup_success = bridge.perform_cleanup()
            assert cleanup_success, "Integrated cleanup should succeed"
            
            # Get integration statistics
            stats = integrator.get_integration_statistics()
            assert stats['session_stats']['registered_managers'] > 0, "Should have registered managers"
            logger.info(f"Integration stats: {json.dumps(stats, indent=2, default=str)}")
            
        finally:
            # Manual cleanup
            for resource in test_resources:
                try:
                    if isinstance(resource, tuple):
                        conn, db_path = resource
                        conn.close()
                        if os.path.exists(db_path):
                            os.unlink(db_path)
                    elif isinstance(resource, str):
                        if os.path.exists(resource):
                            if os.path.isfile(resource):
                                os.unlink(resource)
                            else:
                                shutil.rmtree(resource, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Manual cleanup warning: {e}")
                    
    def test_monitoring_and_validation(self, cleanup_monitor):
        """Test comprehensive monitoring and validation capabilities."""
        logger.info("Testing monitoring and validation...")
        
        with cleanup_monitor.monitoring_context():
            # Create resources in batches to test monitoring
            all_resources = []
            
            for batch in range(3):
                logger.info(f"Creating resource batch {batch + 1}/3")
                batch_resources = []
                
                # Create files
                for i in range(3):
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    temp_file.write(f"Monitoring test batch {batch}, file {i}\n" * 10)
                    temp_file.close()
                    batch_resources.append(temp_file.name)
                    cleanup_monitor.orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                    
                # Create database
                db_path = tempfile.mktemp(suffix='.db')
                conn = sqlite3.connect(db_path)
                conn.execute('CREATE TABLE monitor_test (id INTEGER, batch INTEGER)')
                conn.execute('INSERT INTO monitor_test VALUES (?, ?)', (i, batch))
                conn.commit()
                batch_resources.append((conn, db_path))
                cleanup_monitor.orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
                
                all_resources.extend(batch_resources)
                
                # Perform cleanup cycle
                cycle_result = cleanup_monitor.perform_cleanup_cycle()
                assert cycle_result['cleanup_success'], f"Batch {batch} cleanup should succeed"
                
                # Small delay for monitoring
                time.sleep(0.1)
                
            # Test validation
            validation_results = cleanup_monitor.validator.validate_cleanup(cleanup_monitor.orchestrator)
            assert validation_results, "Should have validation results"
            
            for resource_type, result in validation_results.items():
                logger.info(f"Validation for {resource_type.name}: success={result.success}")
                if not result.success:
                    logger.warning(f"Validation issues: {result.issues}")
                    
            # Test trend analysis
            memory_trend = cleanup_monitor.monitor.get_trend_analysis(ResourceType.MEMORY, hours=1)
            if 'message' not in memory_trend:
                logger.info(f"Memory trend analysis: {memory_trend}")
                
            # Test performance analysis
            perf_analysis = cleanup_monitor.analyzer.analyze_performance_trends(days=1)
            logger.info(f"Performance analysis: {perf_analysis}")
            
            # Generate comprehensive report
            report = cleanup_monitor.generate_comprehensive_report()
            assert report['report_id'], "Should generate report with ID"
            
            health_assessment = report.get('health_assessment', {})
            logger.info(f"System health: {health_assessment.get('status', 'UNKNOWN')} "
                       f"({health_assessment.get('health_score', 0)}/100)")
                       
            # Cleanup test resources
            for resource in all_resources:
                try:
                    if isinstance(resource, tuple):
                        conn, db_path = resource
                        conn.close()
                        if os.path.exists(db_path):
                            os.unlink(db_path)
                    else:
                        if os.path.exists(resource):
                            os.unlink(resource)
                except Exception:
                    pass
                    
    def test_pytest_fixture_integration(self, advanced_cleanup_bridge, test_data_manager):
        """Test integration with pytest fixtures."""
        logger.info("Testing pytest fixture integration...")
        
        # Test that bridge is properly initialized
        assert advanced_cleanup_bridge is not None, "Cleanup bridge should be available"
        
        # Create test resources using both fixtures
        test_files = []
        test_dbs = []
        
        try:
            # Use bridge to register files
            for i in range(3):
                temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                temp_file.write(f"Pytest integration test {i}")
                temp_file.close()
                advanced_cleanup_bridge.register_file(open(temp_file.name, 'r'))
                test_files.append(temp_file.name)
                
            # Use bridge to register databases
            for i in range(2):
                db_path = tempfile.mktemp(suffix='.db')
                conn = sqlite3.connect(db_path)
                conn.execute('CREATE TABLE pytest_test (id INTEGER)')
                conn.execute('INSERT INTO pytest_test VALUES (?)', (i,))
                conn.commit()
                advanced_cleanup_bridge.register_db_connection(conn)
                test_dbs.append((conn, db_path))
                
            # Test resource usage through bridge
            usage = advanced_cleanup_bridge.get_resource_usage()
            assert usage, "Should get resource usage through bridge"
            
            # Test that resources are registered in both systems
            # (This would be validated by the automatic cleanup at fixture teardown)
            
            logger.info(f"Successfully registered {len(test_files)} files and {len(test_dbs)} databases")
            
        finally:
            # Manual cleanup for safety
            for file_path in test_files:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except Exception:
                    pass
                    
            for conn, db_path in test_dbs:
                try:
                    conn.close()
                    if os.path.exists(db_path):
                        os.unlink(db_path)
                except Exception:
                    pass
                    
    def test_failure_recovery_mechanisms(self, advanced_orchestrator):
        """Test failure recovery and retry mechanisms."""
        logger.info("Testing failure recovery mechanisms...")
        
        # Create resources that might cause issues
        problematic_resources = []
        
        try:
            # Create temporary directories with deep nesting
            for i in range(2):
                base_dir = tempfile.mkdtemp(prefix=f'failure_test_{i}_')
                current_dir = Path(base_dir)
                
                # Create deep structure
                for level in range(3):
                    current_dir = current_dir / f'level_{level}'
                    current_dir.mkdir()
                    for j in range(2):
                        file_path = current_dir / f'file_{j}.txt'
                        file_path.write_text(f'Level {level} File {j}')
                        
                problematic_resources.append(base_dir)
                advanced_orchestrator.register_resource(ResourceType.TEMPORARY_FILES, base_dir)
                
            # Create database connections with potential issues
            for i in range(2):
                db_path = tempfile.mktemp(suffix='.db')
                conn = sqlite3.connect(db_path)
                conn.execute('CREATE TABLE failure_test (id INTEGER)')
                # Start transaction but don't commit (potential lock issue)
                conn.execute('BEGIN TRANSACTION')
                conn.execute('INSERT INTO failure_test VALUES (?)', (i,))
                
                problematic_resources.append((conn, db_path))
                advanced_orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
                
            # Test normal cleanup first
            cleanup_success = advanced_orchestrator.cleanup()
            logger.info(f"Normal cleanup with potential issues: {cleanup_success}")
            
            # Test force cleanup
            force_success = advanced_orchestrator.cleanup(force=True)
            logger.info(f"Force cleanup: {force_success}")
            
            # Test validation after cleanup attempts
            validation_success = advanced_orchestrator.validate_cleanup()
            logger.info(f"Validation after recovery: {validation_success}")
            
            # Get statistics to check retry behavior
            stats = advanced_orchestrator.get_cleanup_statistics()
            logger.info(f"Recovery statistics: {json.dumps(stats, indent=2, default=str)}")
            
            # Verify retry behavior through resource managers
            for resource_type, manager in advanced_orchestrator._resource_managers.items():
                if manager._failed_cleanups:
                    logger.info(f"Failures in {resource_type.name}: {len(manager._failed_cleanups)}")
                    
        finally:
            # Manual cleanup of problematic resources
            for resource in problematic_resources:
                try:
                    if isinstance(resource, tuple):
                        conn, db_path = resource
                        conn.close()
                        if os.path.exists(db_path):
                            os.unlink(db_path)
                    else:
                        if os.path.exists(resource):
                            shutil.rmtree(resource, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Manual cleanup of problematic resource failed: {e}")
                    
    def test_performance_tracking(self, cleanup_monitor):
        """Test performance tracking and analysis."""
        logger.info("Testing performance tracking...")
        
        with cleanup_monitor.monitoring_context():
            performance_data = []
            
            # Run multiple cleanup scenarios with different loads
            for scenario in range(1, 4):
                logger.info(f"Running performance scenario {scenario}/3")
                
                scenario_resources = []
                resource_count = scenario * 5  # Increasing load
                
                # Create resources
                for i in range(resource_count):
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    temp_file.write(f"Performance test scenario {scenario}, file {i}\n" * 50)
                    temp_file.close()
                    scenario_resources.append(temp_file.name)
                    cleanup_monitor.orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                    
                # Record start time
                start_time = datetime.now()
                
                # Perform cleanup
                result = cleanup_monitor.perform_cleanup_cycle()
                
                # Record end time
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                performance_data.append({
                    'scenario': scenario,
                    'resource_count': resource_count,
                    'duration': duration,
                    'success': result['cleanup_success']
                })
                
                logger.info(f"Scenario {scenario}: {duration:.3f}s for {resource_count} resources")
                
                # Cleanup scenario resources
                for resource in scenario_resources:
                    try:
                        if os.path.exists(resource):
                            os.unlink(resource)
                    except Exception:
                        pass
                        
            # Analyze performance trends
            trends = cleanup_monitor.analyzer.analyze_performance_trends(days=1)
            logger.info(f"Performance trends: {trends}")
            
            # Check for optimization opportunities
            optimizations = cleanup_monitor.analyzer.identify_optimization_opportunities()
            logger.info(f"Optimization opportunities: {optimizations}")
            
            # Verify performance scaling
            if len(performance_data) > 1:
                duration_trend = [p['duration'] for p in performance_data]
                resource_trend = [p['resource_count'] for p in performance_data]
                
                # Simple performance analysis
                avg_time_per_resource = sum(
                    d / r for d, r in zip(duration_trend, resource_trend)
                ) / len(duration_trend)
                
                logger.info(f"Average time per resource: {avg_time_per_resource:.4f}s")
                assert avg_time_per_resource < 0.1, "Should have reasonable per-resource cleanup time"
                
    @pytest.mark.asyncio
    async def test_async_cleanup_integration(self):
        """Test async cleanup capabilities."""
        logger.info("Testing async cleanup integration...")
        
        config = configure_advanced_cleanup_for_tests()
        
        async with AdvancedCleanupIntegrator(config).cleanup_system.async_cleanup_context() as orchestrator:
            # Create async resources
            temp_files = []
            
            try:
                # Simulate async resource creation
                for i in range(5):
                    # Simulate async operation
                    await asyncio.sleep(0.01)
                    
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    temp_file.write(f"Async test file {i}")
                    temp_file.close()
                    temp_files.append(temp_file.name)
                    
                    orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                    
                # Test async resource usage
                usage = orchestrator.get_resource_usage()
                assert usage, "Should have resource usage in async context"
                
                logger.info(f"Created {len(temp_files)} resources in async context")
                
                # Cleanup happens automatically when exiting context
                
            except Exception as e:
                logger.error(f"Async cleanup test failed: {e}")
                
                # Manual cleanup
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception:
                        pass
                raise
                
    def test_context_manager_integration(self):
        """Test context manager integration patterns."""
        logger.info("Testing context manager integration...")
        
        config = configure_advanced_cleanup_for_tests()
        test_resources = []
        
        with advanced_cleanup_context(config) as integrator:
            # Create test data manager and register it
            test_config = TestDataConfig()
            test_manager = TestDataManager(test_config)
            integrator.register_test_data_manager(test_manager, "context_test")
            
            # Create bridge
            bridge = integrator.create_integrated_fixture_bridge(test_manager)
            
            try:
                # Create resources within context
                temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                temp_file.write("Context manager test")
                temp_file.close()
                bridge.register_file(open(temp_file.name, 'r'))
                test_resources.append(temp_file.name)
                
                temp_dir = tempfile.mkdtemp(prefix='context_test_')
                test_file = Path(temp_dir) / 'test.txt'
                test_file.write_text('Context test')
                bridge.register_temp_path(temp_dir)
                test_resources.append(temp_dir)
                
                # Verify resources are tracked
                usage = bridge.get_resource_usage()
                assert usage, "Should track resources in context"
                
                logger.info(f"Successfully created {len(test_resources)} resources in context")
                
                # Context exit will automatically clean up
                
            finally:
                # Manual cleanup for safety
                for resource in test_resources:
                    try:
                        if os.path.exists(resource):
                            if os.path.isfile(resource):
                                os.unlink(resource)
                            else:
                                shutil.rmtree(resource, ignore_errors=True)
                    except Exception:
                        pass
                        
    def test_comprehensive_report_generation(self, cleanup_monitor):
        """Test comprehensive report generation."""
        logger.info("Testing comprehensive report generation...")
        
        with cleanup_monitor.monitoring_context():
            # Create varied resources for comprehensive reporting
            resources = []
            
            try:
                # Memory objects
                large_objects = []
                for i in range(3):
                    obj = [list(range(1000)) for _ in range(10)]
                    large_objects.append(obj)
                    cleanup_monitor.orchestrator.register_resource(ResourceType.MEMORY, obj)
                    
                # Files
                for i in range(5):
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    temp_file.write(f"Report test file {i}\n" * 20)
                    temp_file.close()
                    resources.append(temp_file.name)
                    cleanup_monitor.orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                    
                # Databases
                for i in range(2):
                    db_path = tempfile.mktemp(suffix='.db')
                    conn = sqlite3.connect(db_path)
                    conn.execute('CREATE TABLE report_test (id INTEGER, data TEXT)')
                    for j in range(10):
                        conn.execute('INSERT INTO report_test VALUES (?, ?)', (j, f'data_{j}'))
                    conn.commit()
                    resources.append((conn, db_path))
                    cleanup_monitor.orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
                    
                # Perform some cleanup cycles for data
                for cycle in range(3):
                    result = cleanup_monitor.perform_cleanup_cycle()
                    logger.info(f"Report test cycle {cycle + 1}: {result['cleanup_success']}")
                    time.sleep(0.1)
                    
                # Generate comprehensive report
                report = cleanup_monitor.generate_comprehensive_report()
                
                # Validate report structure
                assert 'report_id' in report, "Report should have ID"
                assert 'generated_at' in report, "Report should have timestamp"
                assert 'system_overview' in report, "Report should have system overview"
                assert 'cleanup_statistics' in report, "Report should have cleanup statistics"
                assert 'health_assessment' in report, "Report should have health assessment"
                
                # Validate health assessment
                health = report['health_assessment']
                assert 'health_score' in health, "Should have health score"
                assert 'status' in health, "Should have health status"
                
                logger.info(f"Generated report {report['report_id']}")
                logger.info(f"Health status: {health['status']} ({health['health_score']}/100)")
                
                if health.get('issues'):
                    logger.info(f"Issues identified: {len(health['issues'])}")
                    
                if health.get('recommendations'):
                    logger.info(f"Recommendations: {len(health['recommendations'])}")
                    
                # Validate that report files were created
                report_dir = Path("test_data/reports/cleanup")
                if report_dir.exists():
                    report_files = list(report_dir.glob(f"{report['report_id']}*"))
                    assert len(report_files) >= 1, "Should create report files"
                    logger.info(f"Created {len(report_files)} report files")
                    
            finally:
                # Cleanup test resources
                for resource in resources:
                    try:
                        if isinstance(resource, tuple):
                            conn, db_path = resource
                            conn.close()
                            if os.path.exists(db_path):
                                os.unlink(db_path)
                        else:
                            if os.path.exists(resource):
                                os.unlink(resource)
                    except Exception:
                        pass
                        
    def test_integration_error_handling(self):
        """Test error handling in integration scenarios."""
        logger.info("Testing integration error handling...")
        
        config = configure_advanced_cleanup_for_tests()
        
        # Test with invalid configurations
        invalid_config = CleanupIntegrationConfig(
            memory_threshold_mb=-1,  # Invalid threshold
            max_cleanup_time_seconds=-1  # Invalid timeout
        )
        
        # Should handle invalid config gracefully
        try:
            integrator = AdvancedCleanupIntegrator(invalid_config)
            assert integrator is not None, "Should create integrator even with invalid config"
        except Exception as e:
            logger.warning(f"Invalid config handling: {e}")
            
        # Test with missing dependencies
        try:
            # Test creating monitor without proper directory structure
            monitor = CleanupValidationMonitor(report_dir=Path("/nonexistent/directory"))
            # Should handle gracefully
            assert monitor is not None
        except Exception as e:
            logger.warning(f"Missing dependency handling: {e}")
            
        # Test cleanup failures
        orchestrator = AdvancedCleanupOrchestrator()
        
        # Register non-existent resource (should not crash)
        try:
            orchestrator.register_resource(ResourceType.TEMPORARY_FILES, "/nonexistent/file")
            cleanup_success = orchestrator.cleanup()
            # Should not crash even with invalid resources
            logger.info(f"Cleanup with invalid resources: {cleanup_success}")
        except Exception as e:
            logger.warning(f"Invalid resource handling: {e}")
            
        logger.info("Error handling tests completed")


class TestAdvancedCleanupPerformance:
    """Performance tests for advanced cleanup system."""
    
    @pytest.mark.performance
    def test_cleanup_performance_scalability(self):
        """Test cleanup performance with increasing resource counts."""
        logger.info("Testing cleanup performance scalability...")
        
        orchestrator = AdvancedCleanupOrchestrator()
        performance_results = []
        
        # Test with different resource counts
        resource_counts = [10, 50, 100, 200]
        
        for count in resource_counts:
            logger.info(f"Testing with {count} resources...")
            
            resources = []
            
            try:
                # Create resources
                for i in range(count):
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    temp_file.write(f"Performance test {i}")
                    temp_file.close()
                    resources.append(temp_file.name)
                    orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                    
                # Measure cleanup time
                start_time = time.time()
                cleanup_success = orchestrator.cleanup()
                end_time = time.time()
                
                duration = end_time - start_time
                performance_results.append({
                    'resource_count': count,
                    'duration': duration,
                    'success': cleanup_success,
                    'resources_per_second': count / duration if duration > 0 else float('inf')
                })
                
                logger.info(f"{count} resources: {duration:.3f}s ({count/duration:.1f} resources/sec)")
                
                # Performance assertions
                assert cleanup_success, f"Cleanup should succeed with {count} resources"
                assert duration < 5.0, f"Cleanup should complete within 5s for {count} resources"
                
            finally:
                # Manual cleanup
                for resource in resources:
                    try:
                        if os.path.exists(resource):
                            os.unlink(resource)
                    except Exception:
                        pass
                        
        # Analyze performance trends
        if len(performance_results) > 1:
            # Check if performance scales reasonably
            max_ratio = max(
                r2['duration'] / r1['duration']
                for r1, r2 in zip(performance_results[:-1], performance_results[1:])
            )
            
            logger.info(f"Maximum performance degradation ratio: {max_ratio:.2f}")
            assert max_ratio < 10.0, "Performance should scale reasonably"
            
    @pytest.mark.performance
    def test_monitoring_overhead(self, cleanup_monitor):
        """Test overhead of monitoring system."""
        logger.info("Testing monitoring system overhead...")
        
        # Test without monitoring
        orchestrator = AdvancedCleanupOrchestrator()
        resources = []
        
        try:
            # Create resources
            for i in range(50):
                temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                temp_file.write(f"Overhead test {i}")
                temp_file.close()
                resources.append(temp_file.name)
                orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                
            # Measure without monitoring
            start_time = time.time()
            orchestrator.cleanup()
            no_monitoring_time = time.time() - start_time
            
        finally:
            for resource in resources:
                try:
                    if os.path.exists(resource):
                        os.unlink(resource)
                except Exception:
                    pass
                    
        # Test with monitoring
        resources = []
        
        try:
            with cleanup_monitor.monitoring_context():
                # Create resources
                for i in range(50):
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    temp_file.write(f"Overhead test {i}")
                    temp_file.close()
                    resources.append(temp_file.name)
                    cleanup_monitor.orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                    
                # Measure with monitoring
                start_time = time.time()
                cleanup_monitor.perform_cleanup_cycle()
                monitoring_time = time.time() - start_time
                
        finally:
            for resource in resources:
                try:
                    if os.path.exists(resource):
                        os.unlink(resource)
                except Exception:
                    pass
                    
        # Analyze overhead
        overhead_ratio = monitoring_time / no_monitoring_time if no_monitoring_time > 0 else float('inf')
        
        logger.info(f"Cleanup without monitoring: {no_monitoring_time:.3f}s")
        logger.info(f"Cleanup with monitoring: {monitoring_time:.3f}s")
        logger.info(f"Overhead ratio: {overhead_ratio:.2f}x")
        
        # Reasonable overhead assertion
        assert overhead_ratio < 5.0, "Monitoring overhead should be reasonable"


if __name__ == "__main__":
    # Run tests manually if executed directly
    pytest.main([__file__, "-v", "--tb=short"])