#!/usr/bin/env python3
"""
Comprehensive Demo of Advanced Cleanup System for Clinical Metabolomics Oracle LightRAG Integration.

This script demonstrates the complete advanced cleanup system, including:
1. Resource management across different types
2. Performance monitoring and validation
3. Integration with existing test infrastructure
4. Cleanup failure handling and recovery
5. Comprehensive reporting and monitoring

The demo simulates realistic test scenarios and shows how the cleanup system
handles various resource management challenges.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import logging
import sqlite3
import tempfile
import threading
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Import our advanced cleanup system
try:
    from advanced_cleanup_system import (
        AdvancedCleanupOrchestrator, ResourceType, CleanupStrategy,
        CleanupScope, CleanupPolicy, ResourceThresholds
    )
    from cleanup_validation_monitor import (
        CleanupValidationMonitor, AlertConfig
    )
    from advanced_cleanup_integration import (
        AdvancedCleanupIntegrator, CleanupIntegrationConfig,
        advanced_cleanup_context
    )
    from test_data_fixtures import TestDataManager, TestDataConfig
except ImportError as e:
    logger.error(f"Could not import cleanup system modules: {e}")
    logger.info("Make sure you're running this from the tests directory")
    sys.exit(1)


class CleanupSystemDemo:
    """Comprehensive demo of the advanced cleanup system."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.demo_results = {}
        self.temp_resources = []
        
    def run_complete_demo(self):
        """Run the complete demo showcasing all cleanup system features."""
        self.logger.info("="*60)
        self.logger.info("ADVANCED CLEANUP SYSTEM COMPREHENSIVE DEMO")
        self.logger.info("="*60)
        
        try:
            # Demo 1: Basic resource management
            self.demo_basic_resource_management()
            
            # Demo 2: Advanced monitoring and validation
            self.demo_monitoring_and_validation()
            
            # Demo 3: Integration with existing test infrastructure
            self.demo_test_infrastructure_integration()
            
            # Demo 4: Performance analysis and optimization
            self.demo_performance_analysis()
            
            # Demo 5: Failure handling and recovery
            self.demo_failure_handling()
            
            # Demo 6: Comprehensive reporting
            self.demo_comprehensive_reporting()
            
            # Final summary
            self.print_demo_summary()
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}", exc_info=True)
        finally:
            self.cleanup_demo_resources()
            
    def demo_basic_resource_management(self):
        """Demonstrate basic resource management capabilities."""
        self.logger.info("\n" + "="*50)
        self.logger.info("DEMO 1: BASIC RESOURCE MANAGEMENT")
        self.logger.info("="*50)
        
        # Create orchestrator with custom policy
        policy = CleanupPolicy(
            strategy=CleanupStrategy.RESOURCE_BASED,
            resource_types={ResourceType.MEMORY, ResourceType.FILE_HANDLES, 
                          ResourceType.DATABASE_CONNECTIONS, ResourceType.TEMPORARY_FILES},
            validate_cleanup=True,
            report_cleanup=True
        )
        
        thresholds = ResourceThresholds(
            memory_mb=256,
            file_handles=50,
            db_connections=5,
            temp_files=10,
            temp_size_mb=20
        )
        
        orchestrator = AdvancedCleanupOrchestrator(policy, thresholds)
        
        self.logger.info("Created AdvancedCleanupOrchestrator with custom policy")
        
        # Create various resources to manage
        self.logger.info("Creating test resources...")
        
        # 1. File handles
        temp_files = []
        for i in range(5):
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            temp_file.write(f"Test content {i}\n" * 100)  # Add some content
            temp_file.flush()
            temp_files.append(temp_file)
            orchestrator.register_resource(ResourceType.FILE_HANDLES, temp_file)
            self.temp_resources.append(temp_file.name)
            
        self.logger.info(f"Created {len(temp_files)} temporary files")
        
        # 2. Database connections
        db_connections = []
        for i in range(3):
            db_path = tempfile.mktemp(suffix='.db')
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE test (id INTEGER, data TEXT)')
            conn.execute(f'INSERT INTO test VALUES (?, ?)', (i, f'test_data_{i}'))
            conn.commit()
            db_connections.append(conn)
            orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
            self.temp_resources.append(db_path)
            
        self.logger.info(f"Created {len(db_connections)} database connections")
        
        # 3. Temporary files and directories
        temp_paths = []
        for i in range(7):
            temp_path = tempfile.mkdtemp(prefix=f'cleanup_demo_{i}_')
            # Create some files in the temp directory
            for j in range(3):
                file_path = Path(temp_path) / f'file_{j}.txt'
                file_path.write_text(f'Content for file {j} in directory {i}')
            temp_paths.append(temp_path)
            orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_path)
            self.temp_resources.append(temp_path)
            
        self.logger.info(f"Created {len(temp_paths)} temporary directories")
        
        # 4. Memory objects (large lists to consume memory)
        large_objects = []
        for i in range(3):
            large_obj = list(range(10000))  # Create sizeable objects
            large_objects.append(large_obj)
            orchestrator.register_resource(ResourceType.MEMORY, large_obj)
            
        self.logger.info(f"Created {len(large_objects)} large memory objects")
        
        # Get initial resource usage
        initial_usage = orchestrator.get_resource_usage()
        self.logger.info("Initial resource usage:")
        for resource_type, usage in initial_usage.items():
            self.logger.info(f"  {resource_type.name}: {usage}")
            
        # Check if cleanup should be triggered
        should_cleanup = orchestrator.should_cleanup()
        self.logger.info(f"Should cleanup be triggered: {should_cleanup}")
        
        # Perform cleanup
        self.logger.info("Performing cleanup...")
        start_time = time.time()
        cleanup_success = orchestrator.cleanup()
        cleanup_duration = time.time() - start_time
        
        self.logger.info(f"Cleanup completed in {cleanup_duration:.3f}s, success: {cleanup_success}")
        
        # Get post-cleanup resource usage
        final_usage = orchestrator.get_resource_usage()
        self.logger.info("Post-cleanup resource usage:")
        for resource_type, usage in final_usage.items():
            self.logger.info(f"  {resource_type.name}: {usage}")
            
        # Validate cleanup
        validation_success = orchestrator.validate_cleanup()
        self.logger.info(f"Cleanup validation: {validation_success}")
        
        # Get statistics
        stats = orchestrator.get_cleanup_statistics()
        self.logger.info(f"Cleanup statistics: {json.dumps(stats, indent=2, default=str)}")
        
        self.demo_results['basic_resource_management'] = {
            'cleanup_success': cleanup_success,
            'cleanup_duration': cleanup_duration,
            'validation_success': validation_success,
            'initial_usage': {str(k): v for k, v in initial_usage.items()},
            'final_usage': {str(k): v for k, v in final_usage.items()},
            'statistics': stats
        }
        
    def demo_monitoring_and_validation(self):
        """Demonstrate monitoring and validation capabilities."""
        self.logger.info("\n" + "="*50)
        self.logger.info("DEMO 2: MONITORING AND VALIDATION")
        self.logger.info("="*50)
        
        # Create comprehensive monitoring system
        policy = CleanupPolicy(
            strategy=CleanupStrategy.DEFERRED,
            validate_cleanup=True,
            report_cleanup=True
        )
        
        thresholds = ResourceThresholds(
            memory_mb=128,
            file_handles=20,
            db_connections=3
        )
        
        alert_config = AlertConfig(
            enabled=True,
            memory_threshold_mb=256,
            file_handle_threshold=30,
            cleanup_failure_threshold=2
        )
        
        # Create report directory
        report_dir = Path("test_data/reports/cleanup/demo")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        monitor = CleanupValidationMonitor(
            cleanup_policy=policy,
            thresholds=thresholds,
            alert_config=alert_config,
            report_dir=report_dir
        )
        
        self.logger.info("Created CleanupValidationMonitor with comprehensive configuration")
        
        # Start monitoring
        with monitor.monitoring_context():
            self.logger.info("Started resource monitoring")
            
            # Create resources over time to show monitoring in action
            resources = []
            for i in range(5):
                self.logger.info(f"Creating resource batch {i+1}/5...")
                
                # Create some temporary files
                for j in range(4):
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    temp_file.write(f"Batch {i}, file {j}\n" * 50)
                    temp_file.close()
                    resources.append(temp_file.name)
                    monitor.orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                    
                # Create a database connection
                db_path = tempfile.mktemp(suffix='.db')
                conn = sqlite3.connect(db_path)
                conn.execute('CREATE TABLE batch_test (id INTEGER, batch INTEGER)')
                conn.execute('INSERT INTO batch_test VALUES (?, ?)', (j, i))
                conn.commit()
                resources.append((conn, db_path))
                monitor.orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
                
                # Wait a bit to let monitoring collect data
                time.sleep(0.5)
                
                # Perform cleanup cycle
                self.logger.info(f"Performing cleanup cycle {i+1}...")
                cycle_result = monitor.perform_cleanup_cycle()
                self.logger.info(f"Cleanup cycle result: {cycle_result['cleanup_success']}")
                
                # Check for alerts
                alerts = cycle_result.get('alerts', [])
                if alerts:
                    self.logger.warning(f"Generated {len(alerts)} alerts")
                    for alert in alerts:
                        self.logger.warning(f"  Alert: {alert['message']}")
                        
            # Get trend analysis
            self.logger.info("Analyzing resource usage trends...")
            for resource_type in [ResourceType.MEMORY, ResourceType.FILE_HANDLES, 
                                ResourceType.DATABASE_CONNECTIONS, ResourceType.TEMPORARY_FILES]:
                trend = monitor.monitor.get_trend_analysis(resource_type, hours=1)
                if 'message' not in trend:
                    self.logger.info(f"Trend for {resource_type.name}: {json.dumps(trend, indent=2, default=str)}")
                    
            # Generate comprehensive report
            self.logger.info("Generating comprehensive report...")
            report = monitor.generate_comprehensive_report()
            report_id = report.get('report_id', 'unknown')
            self.logger.info(f"Generated report: {report_id}")
            
            # Get alert summary
            alert_summary = monitor.alert_system.get_alert_summary(hours=1)
            self.logger.info(f"Alert summary: {json.dumps(alert_summary, indent=2, default=str)}")
            
            self.demo_results['monitoring_and_validation'] = {
                'report_id': report_id,
                'alert_summary': alert_summary,
                'final_health_score': report.get('health_assessment', {}).get('health_score', 0)
            }
            
        self.logger.info("Monitoring demo completed")
        
        # Clean up demo resources
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
            except Exception as e:
                self.logger.warning(f"Error cleaning up resource {resource}: {e}")
                
    def demo_test_infrastructure_integration(self):
        """Demonstrate integration with existing test infrastructure."""
        self.logger.info("\n" + "="*50)
        self.logger.info("DEMO 3: TEST INFRASTRUCTURE INTEGRATION")
        self.logger.info("="*50)
        
        # Create test-friendly integration configuration
        config = CleanupIntegrationConfig(
            enabled=True,
            auto_register_resources=True,
            monitor_performance=True,
            generate_reports=True,
            validate_cleanup=True,
            memory_threshold_mb=128,
            file_handle_threshold=20,
            db_connection_threshold=5
        )
        
        with advanced_cleanup_context(config) as integrator:
            self.logger.info("Created advanced cleanup integration context")
            
            # Create existing TestDataManager (simulating pytest fixture)
            test_config = TestDataConfig(
                use_temp_dirs=True,
                auto_cleanup=True,
                validate_data=True
            )
            test_manager = TestDataManager(test_config)
            
            # Register with integrator
            integrator.register_test_data_manager(test_manager, "demo_integration_test")
            self.logger.info("Registered TestDataManager with integrator")
            
            # Create integration bridge
            bridge = integrator.create_integrated_fixture_bridge(test_manager)
            self.logger.info("Created integration bridge")
            
            # Simulate test operations using both systems
            self.logger.info("Simulating test operations...")
            
            # 1. Register files with bridge
            test_files = []
            for i in range(3):
                temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                temp_file.write(f"Integration test file {i}\n")
                bridge.register_file(temp_file, auto_close=True)
                test_files.append(temp_file)
                
            self.logger.info(f"Registered {len(test_files)} files with bridge")
            
            # 2. Register database connections
            test_dbs = []
            for i in range(2):
                db_path = tempfile.mktemp(suffix='.db')
                conn = sqlite3.connect(db_path)
                conn.execute('CREATE TABLE integration_test (id INTEGER, test_name TEXT)')
                conn.execute('INSERT INTO integration_test VALUES (?, ?)', 
                           (i, f'integration_test_{i}'))
                conn.commit()
                bridge.register_db_connection(conn)
                test_dbs.append((conn, db_path))
                
            self.logger.info(f"Registered {len(test_dbs)} database connections with bridge")
            
            # 3. Register temporary paths
            temp_dirs = []
            for i in range(3):
                temp_dir = tempfile.mkdtemp(prefix=f'integration_test_{i}_')
                # Create some content
                for j in range(2):
                    file_path = Path(temp_dir) / f'test_file_{j}.txt'
                    file_path.write_text(f'Integration test content {j}')
                bridge.register_temp_path(temp_dir)
                temp_dirs.append(temp_dir)
                
            self.logger.info(f"Registered {len(temp_dirs)} temporary directories with bridge")
            
            # Get resource usage through bridge
            resource_usage = bridge.get_resource_usage()
            self.logger.info("Resource usage via bridge:")
            for resource_type, usage in resource_usage.items():
                self.logger.info(f"  {resource_type}: {usage}")
                
            # Perform cleanup through bridge
            self.logger.info("Performing cleanup via integration bridge...")
            cleanup_success = bridge.perform_cleanup()
            self.logger.info(f"Integrated cleanup success: {cleanup_success}")
            
            # Get integration statistics
            integration_stats = integrator.get_integration_statistics()
            self.logger.info(f"Integration statistics: {json.dumps(integration_stats, indent=2, default=str)}")
            
            self.demo_results['test_infrastructure_integration'] = {
                'integration_success': cleanup_success,
                'statistics': integration_stats
            }
            
        self.logger.info("Test infrastructure integration demo completed")
        
    def demo_performance_analysis(self):
        """Demonstrate performance analysis capabilities."""
        self.logger.info("\n" + "="*50)
        self.logger.info("DEMO 4: PERFORMANCE ANALYSIS")
        self.logger.info("="*50)
        
        # Create monitor with performance tracking
        monitor = CleanupValidationMonitor()
        
        with monitor.monitoring_context():
            self.logger.info("Started performance monitoring")
            
            # Perform multiple cleanup operations with different characteristics
            performance_results = []
            
            for scenario in range(1, 6):  # 5 different scenarios
                self.logger.info(f"Running performance scenario {scenario}/5")
                
                # Create different resource loads for each scenario
                if scenario == 1:
                    # Light load
                    resource_count = 5
                    resource_types = {ResourceType.MEMORY, ResourceType.TEMPORARY_FILES}
                elif scenario == 2:
                    # Medium load
                    resource_count = 15
                    resource_types = {ResourceType.FILE_HANDLES, ResourceType.DATABASE_CONNECTIONS}
                elif scenario == 3:
                    # Heavy load
                    resource_count = 25
                    resource_types = set(ResourceType)
                elif scenario == 4:
                    # Database-heavy load
                    resource_count = 10
                    resource_types = {ResourceType.DATABASE_CONNECTIONS}
                elif scenario == 5:
                    # File-heavy load
                    resource_count = 30
                    resource_types = {ResourceType.FILE_HANDLES, ResourceType.TEMPORARY_FILES}
                    
                # Create resources for scenario
                scenario_resources = []
                for i in range(resource_count):
                    if ResourceType.TEMPORARY_FILES in resource_types:
                        temp_path = tempfile.mkdtemp(prefix=f'perf_scenario_{scenario}_{i}_')
                        monitor.orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_path)
                        scenario_resources.append(temp_path)
                        
                    if ResourceType.FILE_HANDLES in resource_types:
                        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                        temp_file.write(f"Performance test data for scenario {scenario}")
                        temp_file.close()
                        monitor.orchestrator.register_resource(ResourceType.FILE_HANDLES, temp_file)
                        scenario_resources.append(temp_file.name)
                        
                    if ResourceType.DATABASE_CONNECTIONS in resource_types:
                        db_path = tempfile.mktemp(suffix='.db')
                        conn = sqlite3.connect(db_path)
                        conn.execute('CREATE TABLE perf_test (id INTEGER, scenario INTEGER, data TEXT)')
                        for j in range(10):  # Add some data
                            conn.execute('INSERT INTO perf_test VALUES (?, ?, ?)', 
                                       (j, scenario, f'test_data_{j}'))
                        conn.commit()
                        monitor.orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
                        scenario_resources.append((conn, db_path))
                        
                    if ResourceType.MEMORY in resource_types:
                        large_obj = [list(range(1000)) for _ in range(10)]  # Memory-intensive object
                        monitor.orchestrator.register_resource(ResourceType.MEMORY, large_obj)
                        scenario_resources.append(large_obj)
                        
                # Record start time and perform cleanup
                start_time = datetime.now()
                cleanup_result = monitor.perform_cleanup_cycle(resource_types=resource_types)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                success = cleanup_result.get('cleanup_success', False)
                
                self.logger.info(f"Scenario {scenario}: {duration:.3f}s, success: {success}")
                
                performance_results.append({
                    'scenario': scenario,
                    'resource_count': resource_count,
                    'resource_types': [rt.name for rt in resource_types],
                    'duration': duration,
                    'success': success,
                    'performance_metrics': cleanup_result.get('performance_metrics', {})
                })
                
                # Clean up scenario resources
                for resource in scenario_resources:
                    try:
                        if isinstance(resource, tuple):
                            conn, db_path = resource
                            conn.close()
                            if os.path.exists(db_path):
                                os.unlink(db_path)
                        elif isinstance(resource, str) and os.path.exists(resource):
                            if os.path.isfile(resource):
                                os.unlink(resource)
                            else:
                                import shutil
                                shutil.rmtree(resource, ignore_errors=True)
                    except Exception as e:
                        self.logger.warning(f"Error cleaning scenario resource: {e}")
                        
                # Small delay between scenarios
                time.sleep(0.2)
                
            # Analyze performance trends
            self.logger.info("Analyzing performance trends...")
            perf_analysis = monitor.analyzer.analyze_performance_trends(days=1)
            self.logger.info(f"Performance analysis: {json.dumps(perf_analysis, indent=2, default=str)}")
            
            # Identify optimization opportunities
            self.logger.info("Identifying optimization opportunities...")
            optimizations = monitor.analyzer.identify_optimization_opportunities()
            self.logger.info(f"Optimization opportunities: {json.dumps(optimizations, indent=2, default=str)}")
            
            self.demo_results['performance_analysis'] = {
                'scenario_results': performance_results,
                'performance_analysis': perf_analysis,
                'optimization_opportunities': optimizations
            }
            
        self.logger.info("Performance analysis demo completed")
        
    def demo_failure_handling(self):
        """Demonstrate failure handling and recovery capabilities."""
        self.logger.info("\n" + "="*50)
        self.logger.info("DEMO 5: FAILURE HANDLING AND RECOVERY")
        self.logger.info("="*50)
        
        # Create orchestrator with retry policies
        policy = CleanupPolicy(
            strategy=CleanupStrategy.IMMEDIATE,
            max_retry_attempts=3,
            retry_delay_seconds=0.1,  # Quick retries for demo
            force_cleanup=True,
            emergency_cleanup=True
        )
        
        orchestrator = AdvancedCleanupOrchestrator(policy)
        self.logger.info("Created orchestrator with failure recovery configuration")
        
        # Create resources that might cause cleanup issues
        problematic_resources = []
        
        # 1. Files with permissions issues (simulate)
        temp_files = []
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            temp_file.write(f"File that might cause issues {i}")
            temp_file.close()
            temp_files.append(temp_file.name)
            orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
            
        self.logger.info(f"Created {len(temp_files)} potentially problematic files")
        
        # 2. Database connections that might fail to close
        db_connections = []
        for i in range(2):
            db_path = tempfile.mktemp(suffix='.db')
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE failure_test (id INTEGER)')
            conn.commit()
            # Simulate some operations that might leave connections in problematic state
            try:
                conn.execute('BEGIN TRANSACTION')
                conn.execute('INSERT INTO failure_test VALUES (?)', (i,))
                # Don't commit to leave transaction open
            except Exception:
                pass
            db_connections.append((conn, db_path))
            orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
            
        self.logger.info(f"Created {len(db_connections)} potentially problematic database connections")
        
        # First, try normal cleanup to establish baseline
        self.logger.info("Attempting normal cleanup...")
        normal_success = orchestrator.cleanup()
        self.logger.info(f"Normal cleanup success: {normal_success}")
        
        # Now simulate failure scenario by creating more complex resources
        self.logger.info("Creating failure-prone scenarios...")
        
        # Create temporary directories with deep nesting
        deep_temp_dirs = []
        for i in range(2):
            base_dir = tempfile.mkdtemp(prefix=f'deep_failure_test_{i}_')
            # Create deep directory structure
            current_dir = Path(base_dir)
            for level in range(5):
                current_dir = current_dir / f'level_{level}'
                current_dir.mkdir()
                # Create files at each level
                for j in range(3):
                    file_path = current_dir / f'file_{j}.txt'
                    file_path.write_text(f'Content at level {level}, file {j}')
            deep_temp_dirs.append(base_dir)
            orchestrator.register_resource(ResourceType.TEMPORARY_FILES, base_dir)
            
        self.logger.info(f"Created {len(deep_temp_dirs)} deep directory structures")
        
        # Try cleanup with failure scenarios
        self.logger.info("Attempting cleanup with potential failure scenarios...")
        start_time = time.time()
        failure_scenario_success = orchestrator.cleanup()
        failure_cleanup_duration = time.time() - start_time
        
        self.logger.info(f"Failure scenario cleanup: success={failure_scenario_success}, duration={failure_cleanup_duration:.3f}s")
        
        # Check validation after failure scenarios
        validation_success = orchestrator.validate_cleanup()
        self.logger.info(f"Post-failure cleanup validation: {validation_success}")
        
        # Test force cleanup
        self.logger.info("Testing force cleanup...")
        force_start_time = time.time()
        orchestrator.force_cleanup()
        force_cleanup_duration = time.time() - force_start_time
        
        self.logger.info(f"Force cleanup completed in {force_cleanup_duration:.3f}s")
        
        # Get final statistics
        final_stats = orchestrator.get_cleanup_statistics()
        self.logger.info(f"Final cleanup statistics: {json.dumps(final_stats, indent=2, default=str)}")
        
        self.demo_results['failure_handling'] = {
            'normal_cleanup_success': normal_success,
            'failure_scenario_success': failure_scenario_success,
            'failure_cleanup_duration': failure_cleanup_duration,
            'validation_success': validation_success,
            'force_cleanup_duration': force_cleanup_duration,
            'final_statistics': final_stats
        }
        
        # Clean up remaining demo resources
        for resource in temp_files + [path for _, path in db_connections] + deep_temp_dirs:
            try:
                if os.path.exists(resource):
                    if os.path.isfile(resource):
                        os.unlink(resource)
                    else:
                        import shutil
                        shutil.rmtree(resource, ignore_errors=True)
            except Exception as e:
                self.logger.warning(f"Error cleaning up failure demo resource {resource}: {e}")
                
        self.logger.info("Failure handling demo completed")
        
    def demo_comprehensive_reporting(self):
        """Demonstrate comprehensive reporting capabilities."""
        self.logger.info("\n" + "="*50)
        self.logger.info("DEMO 6: COMPREHENSIVE REPORTING")
        self.logger.info("="*50)
        
        # Create monitoring system for comprehensive reporting
        report_dir = Path("test_data/reports/cleanup/comprehensive_demo")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        monitor = CleanupValidationMonitor(report_dir=report_dir)
        
        self.logger.info(f"Created monitoring system with report directory: {report_dir}")
        
        with monitor.monitoring_context():
            # Simulate a comprehensive test session
            self.logger.info("Simulating comprehensive test session...")
            
            session_resources = []
            
            # Create resources in batches to show progression
            for batch in range(3):
                self.logger.info(f"Creating resource batch {batch+1}/3...")
                
                batch_resources = []
                
                # Files
                for i in range(5):
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                    temp_file.write(f"Comprehensive demo batch {batch}, file {i}\n" * 20)
                    temp_file.close()
                    batch_resources.append(temp_file.name)
                    monitor.orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_file.name)
                    
                # Databases
                for i in range(2):
                    db_path = tempfile.mktemp(suffix='.db')
                    conn = sqlite3.connect(db_path)
                    conn.execute('CREATE TABLE comprehensive_demo (id INTEGER, batch INTEGER, data TEXT)')
                    for j in range(10):
                        conn.execute('INSERT INTO comprehensive_demo VALUES (?, ?, ?)',
                                   (j, batch, f'demo_data_{j}'))
                    conn.commit()
                    batch_resources.append((conn, db_path))
                    monitor.orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
                    
                session_resources.extend(batch_resources)
                
                # Perform cleanup cycle for this batch
                cycle_result = monitor.perform_cleanup_cycle()
                self.logger.info(f"Batch {batch+1} cleanup: {cycle_result['cleanup_success']}")
                
                # Add some delay to show monitoring over time
                time.sleep(0.3)
                
            # Generate intermediate reports
            self.logger.info("Generating intermediate reports...")
            
            # Performance analysis
            perf_analysis = monitor.analyzer.analyze_performance_trends(days=1)
            
            # Validation summary
            validation_summary = monitor.validator.get_validation_summary(days=1)
            
            # Alert summary
            alert_summary = monitor.alert_system.get_alert_summary(hours=1)
            
            # Resource monitoring summary
            monitoring_summary = monitor.monitor.get_monitoring_summary()
            
            self.logger.info("Generated intermediate analysis reports")
            
            # Final comprehensive report
            self.logger.info("Generating final comprehensive report...")
            comprehensive_report = monitor.generate_comprehensive_report()
            
            report_id = comprehensive_report.get('report_id', 'unknown')
            self.logger.info(f"Generated comprehensive report: {report_id}")
            
            # Display key metrics from the report
            health_assessment = comprehensive_report.get('health_assessment', {})
            self.logger.info(f"System Health: {health_assessment.get('status', 'UNKNOWN')} "
                           f"({health_assessment.get('health_score', 0)}/100)")
            
            if health_assessment.get('issues'):
                self.logger.info("Issues identified:")
                for issue in health_assessment['issues'][:3]:  # Show top 3
                    self.logger.info(f"  - {issue}")
                    
            if health_assessment.get('recommendations'):
                self.logger.info("Recommendations:")
                for rec in health_assessment['recommendations'][:3]:  # Show top 3
                    self.logger.info(f"  - {rec}")
                    
            # Save demo results
            self.demo_results['comprehensive_reporting'] = {
                'report_id': report_id,
                'health_score': health_assessment.get('health_score', 0),
                'health_status': health_assessment.get('status', 'UNKNOWN'),
                'issues_count': len(health_assessment.get('issues', [])),
                'recommendations_count': len(health_assessment.get('recommendations', [])),
                'performance_analysis': perf_analysis,
                'validation_summary': validation_summary,
                'alert_summary': alert_summary,
                'monitoring_summary': monitoring_summary
            }
            
            # Demonstrate report file locations
            self.logger.info("Report files generated:")
            if report_dir.exists():
                for report_file in sorted(report_dir.glob("*.json")):
                    self.logger.info(f"  - {report_file.name}")
                for summary_file in sorted(report_dir.glob("*_summary.txt")):
                    self.logger.info(f"  - {summary_file.name}")
                    
            self.logger.info("Comprehensive reporting demo completed")
            
            # Clean up session resources
            for resource in session_resources:
                try:
                    if isinstance(resource, tuple):
                        conn, db_path = resource
                        conn.close()
                        if os.path.exists(db_path):
                            os.unlink(db_path)
                    elif isinstance(resource, str) and os.path.exists(resource):
                        os.unlink(resource)
                except Exception as e:
                    self.logger.warning(f"Error cleaning up reporting demo resource: {e}")
                    
    def print_demo_summary(self):
        """Print comprehensive summary of all demo results."""
        self.logger.info("\n" + "="*60)
        self.logger.info("COMPREHENSIVE DEMO SUMMARY")
        self.logger.info("="*60)
        
        if not self.demo_results:
            self.logger.warning("No demo results available")
            return
            
        for demo_name, results in self.demo_results.items():
            self.logger.info(f"\n{demo_name.upper().replace('_', ' ')}:")
            self.logger.info("-" * 40)
            
            if demo_name == 'basic_resource_management':
                self.logger.info(f"  Cleanup Success: {results.get('cleanup_success', False)}")
                self.logger.info(f"  Cleanup Duration: {results.get('cleanup_duration', 0):.3f}s")
                self.logger.info(f"  Validation Success: {results.get('validation_success', False)}")
                
            elif demo_name == 'monitoring_and_validation':
                self.logger.info(f"  Report ID: {results.get('report_id', 'N/A')}")
                self.logger.info(f"  Health Score: {results.get('final_health_score', 0)}/100")
                alert_summary = results.get('alert_summary', {})
                self.logger.info(f"  Total Alerts: {alert_summary.get('total_alerts', 0)}")
                
            elif demo_name == 'test_infrastructure_integration':
                self.logger.info(f"  Integration Success: {results.get('integration_success', False)}")
                stats = results.get('statistics', {})
                session_stats = stats.get('session_stats', {})
                self.logger.info(f"  Registered Managers: {session_stats.get('registered_managers', 0)}")
                
            elif demo_name == 'performance_analysis':
                scenario_results = results.get('scenario_results', [])
                if scenario_results:
                    avg_duration = sum(r['duration'] for r in scenario_results) / len(scenario_results)
                    success_rate = sum(1 for r in scenario_results if r['success']) / len(scenario_results) * 100
                    self.logger.info(f"  Scenarios Tested: {len(scenario_results)}")
                    self.logger.info(f"  Average Duration: {avg_duration:.3f}s")
                    self.logger.info(f"  Success Rate: {success_rate:.1f}%")
                    
            elif demo_name == 'failure_handling':
                self.logger.info(f"  Normal Cleanup: {results.get('normal_cleanup_success', False)}")
                self.logger.info(f"  Failure Scenario: {results.get('failure_scenario_success', False)}")
                self.logger.info(f"  Validation Success: {results.get('validation_success', False)}")
                self.logger.info(f"  Force Cleanup Duration: {results.get('force_cleanup_duration', 0):.3f}s")
                
            elif demo_name == 'comprehensive_reporting':
                self.logger.info(f"  Report ID: {results.get('report_id', 'N/A')}")
                self.logger.info(f"  Health Status: {results.get('health_status', 'UNKNOWN')}")
                self.logger.info(f"  Health Score: {results.get('health_score', 0)}/100")
                self.logger.info(f"  Issues Identified: {results.get('issues_count', 0)}")
                self.logger.info(f"  Recommendations: {results.get('recommendations_count', 0)}")
                
        # Overall summary
        self.logger.info("\n" + "="*60)
        self.logger.info("OVERALL DEMO STATUS")
        self.logger.info("="*60)
        
        total_demos = len(self.demo_results)
        successful_demos = 0
        
        for demo_name, results in self.demo_results.items():
            # Determine if demo was successful based on key metrics
            if demo_name == 'basic_resource_management':
                if results.get('cleanup_success') and results.get('validation_success'):
                    successful_demos += 1
            elif demo_name == 'monitoring_and_validation':
                if results.get('final_health_score', 0) >= 70:
                    successful_demos += 1
            elif demo_name == 'test_infrastructure_integration':
                if results.get('integration_success'):
                    successful_demos += 1
            elif demo_name == 'performance_analysis':
                scenario_results = results.get('scenario_results', [])
                if scenario_results and all(r['success'] for r in scenario_results):
                    successful_demos += 1
            elif demo_name == 'failure_handling':
                if results.get('validation_success'):
                    successful_demos += 1
            elif demo_name == 'comprehensive_reporting':
                if results.get('health_score', 0) >= 60:
                    successful_demos += 1
                    
        success_rate = (successful_demos / total_demos * 100) if total_demos > 0 else 0
        
        self.logger.info(f"Successful Demos: {successful_demos}/{total_demos}")
        self.logger.info(f"Overall Success Rate: {success_rate:.1f}%")
        self.logger.info(f"Demo Status: {'PASS' if success_rate >= 80 else 'PARTIAL' if success_rate >= 60 else 'FAIL'}")
        
        self.logger.info("\n" + "="*60)
        
    def cleanup_demo_resources(self):
        """Clean up any remaining demo resources."""
        self.logger.info("Cleaning up demo resources...")
        
        cleanup_count = 0
        for resource_path in self.temp_resources:
            try:
                if os.path.exists(resource_path):
                    if os.path.isfile(resource_path):
                        os.unlink(resource_path)
                    else:
                        import shutil
                        shutil.rmtree(resource_path, ignore_errors=True)
                    cleanup_count += 1
            except Exception as e:
                self.logger.warning(f"Could not clean up demo resource {resource_path}: {e}")
                
        self.logger.info(f"Cleaned up {cleanup_count} demo resources")


def main():
    """Main entry point for the demo."""
    print("Advanced Cleanup System Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases all features of the advanced cleanup system")
    print("including resource management, monitoring, validation, integration,")
    print("performance analysis, failure handling, and comprehensive reporting.")
    print("=" * 60)
    print()
    
    demo = CleanupSystemDemo()
    
    try:
        demo.run_complete_demo()
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("Check the logs above for detailed results and")
        print("the test_data/reports/cleanup/ directory for generated reports.")
        print("=" * 60)
        return 0
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        demo.cleanup_demo_resources()
        return 1
    except Exception as e:
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        demo.cleanup_demo_resources()
        return 1


if __name__ == "__main__":
    sys.exit(main())