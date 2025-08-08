"""
Migration Testing Framework for Safe Deployment Validation
==========================================================

This module provides a comprehensive testing framework for validating safe
migration and deployment of the LLM-enhanced Clinical Metabolomics Oracle
system, ensuring zero-downtime deployment and graceful rollback capabilities.

Key Features:
- Zero-downtime deployment testing
- Configuration migration validation
- Rollback procedure verification
- Database state preservation testing
- Feature flag-based gradual rollout validation
- Performance monitoring during deployment
- Data integrity verification
- Service health monitoring

Test Categories:
1. Pre-deployment validation
2. Configuration migration testing
3. Zero-downtime deployment simulation
4. Post-deployment verification
5. Rollback testing
6. Data integrity validation
7. Performance monitoring during migration

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import pytest
import asyncio
import time
import json
import logging
import tempfile
import shutil
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import subprocess
import os

# Import system components
from lightrag_integration.query_router import BiomedicalQueryRouter
from lightrag_integration.llm_query_classifier import LLMQueryClassifier, LLMClassificationConfig
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.cost_persistence import CostTracker

# Test utilities
from .biomedical_test_fixtures import BiomedicalTestFixtures
from .performance_test_utilities import PerformanceTestUtilities


@dataclass
class MigrationTestConfig:
    """Configuration for migration testing."""
    
    # Test environment settings
    test_data_dir: str
    backup_dir: str
    temp_dir: str
    
    # Database settings
    test_db_path: str
    backup_db_path: str
    
    # Service settings
    test_port: int = 8080
    health_check_interval: int = 5
    max_deployment_time: int = 300  # 5 minutes
    
    # Migration settings
    enable_gradual_rollout: bool = True
    rollout_percentage: float = 10.0  # Start with 10% traffic
    rollback_threshold: float = 0.05  # 5% error rate triggers rollback
    
    # Performance thresholds
    max_acceptable_downtime: int = 0  # Zero downtime requirement
    max_response_time_degradation: float = 1.5  # 50% increase max
    min_success_rate: float = 0.95  # 95% success rate minimum


@dataclass
class DeploymentState:
    """State tracking for deployment process."""
    
    phase: str = "not_started"  # not_started, pre_check, deploying, post_check, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Service health
    old_service_healthy: bool = True
    new_service_healthy: bool = False
    traffic_split_percentage: float = 0.0
    
    # Performance metrics
    response_times: List[float] = None
    error_rates: List[float] = None
    success_count: int = 0
    failure_count: int = 0
    
    # Configuration state
    config_migrated: bool = False
    database_migrated: bool = False
    features_enabled: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.error_rates is None:
            self.error_rates = []
        if self.features_enabled is None:
            self.features_enabled = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result


class MockServiceManager:
    """Mock service manager for testing deployment scenarios."""
    
    def __init__(self, config: MigrationTestConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.services = {}  # service_name -> service_status
        self.health_checks = {}
        
    def start_service(self, service_name: str, version: str = "old") -> bool:
        """Start a service instance."""
        try:
            self.services[service_name] = {
                'status': 'running',
                'version': version,
                'start_time': datetime.now(),
                'health': True,
                'port': self.config.test_port + len(self.services)
            }
            self.logger.info(f"Started {service_name} ({version}) on port {self.services[service_name]['port']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a service instance.""" 
        try:
            if service_name in self.services:
                self.services[service_name]['status'] = 'stopped'
                self.services[service_name]['health'] = False
                self.logger.info(f"Stopped {service_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to stop {service_name}: {e}")
            return False
    
    def health_check(self, service_name: str) -> bool:
        """Check service health."""
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        if service['status'] != 'running':
            return False
        
        # Simulate occasional health check failures
        import random
        if random.random() < 0.05:  # 5% chance of temporary failure
            self.logger.warning(f"Temporary health check failure for {service_name}")
            return False
        
        return service['health']
    
    def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get service performance metrics."""
        if service_name not in self.services:
            return {}
        
        # Simulate realistic metrics
        import random
        base_response_time = 50 if self.services[service_name]['version'] == 'old' else 70
        
        return {
            'avg_response_time_ms': base_response_time + random.uniform(-10, 20),
            'success_rate': random.uniform(0.95, 1.0),
            'requests_per_second': random.uniform(10, 50),
            'cpu_usage': random.uniform(20, 80),
            'memory_usage': random.uniform(30, 70)
        }
    
    def split_traffic(self, old_service: str, new_service: str, new_percentage: float) -> bool:
        """Split traffic between old and new services."""
        try:
            if old_service not in self.services or new_service not in self.services:
                return False
            
            self.logger.info(f"Traffic split: {new_percentage}% to {new_service}, {100-new_percentage}% to {old_service}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to split traffic: {e}")
            return False


class ConfigurationMigrator:
    """Handles configuration migration between versions."""
    
    def __init__(self, config: MigrationTestConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def backup_configuration(self) -> bool:
        """Backup current configuration."""
        try:
            # Simulate configuration backup
            backup_path = Path(self.config.backup_dir) / "config_backup.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Mock configuration data
            config_data = {
                'routing_thresholds': {
                    'high_confidence': 0.8,
                    'medium_confidence': 0.6,
                    'low_confidence': 0.4
                },
                'llm_settings': {
                    'enabled': False,  # Initially disabled
                    'model': 'gpt-4o-mini',
                    'timeout': 3.0
                },
                'feature_flags': {
                    'enable_llm_classification': False,
                    'enable_hybrid_confidence': False,
                    'enable_enhanced_logging': True
                },
                'backup_timestamp': datetime.now().isoformat()
            }
            
            with open(backup_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Configuration backed up to {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration backup failed: {e}")
            return False
    
    def migrate_configuration(self) -> bool:
        """Migrate configuration to new version."""
        try:
            # Load current config
            backup_path = Path(self.config.backup_dir) / "config_backup.json"
            if not backup_path.exists():
                self.logger.error("No configuration backup found")
                return False
            
            with open(backup_path, 'r') as f:
                old_config = json.load(f)
            
            # Create migrated configuration
            new_config = old_config.copy()
            
            # Add new LLM settings while preserving existing values
            new_config['llm_settings'].update({
                'daily_budget': 5.0,
                'cache_enabled': True,
                'fallback_enabled': True
            })
            
            # Add new feature flags
            new_config['feature_flags'].update({
                'enable_comprehensive_confidence': False,  # Start disabled for safety
                'enable_performance_monitoring': True,
                'enable_gradual_rollout': True
            })
            
            # Add enhanced routing settings
            new_config['enhanced_routing'] = {
                'confidence_calibration': True,
                'multi_dimensional_scoring': True,
                'uncertainty_quantification': True
            }
            
            # Save migrated configuration
            new_config_path = Path(self.config.test_data_dir) / "migrated_config.json"
            new_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(new_config_path, 'w') as f:
                json.dump(new_config, f, indent=2)
            
            self.logger.info(f"Configuration migrated to {new_config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration migration failed: {e}")
            return False
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate migrated configuration."""
        issues = []
        
        try:
            config_path = Path(self.config.test_data_dir) / "migrated_config.json"
            if not config_path.exists():
                issues.append("Migrated configuration file not found")
                return False, issues
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required sections
            required_sections = ['routing_thresholds', 'llm_settings', 'feature_flags']
            for section in required_sections:
                if section not in config:
                    issues.append(f"Missing configuration section: {section}")
            
            # Validate routing thresholds
            if 'routing_thresholds' in config:
                thresholds = config['routing_thresholds']
                required_thresholds = ['high_confidence', 'medium_confidence', 'low_confidence']
                for threshold in required_thresholds:
                    if threshold not in thresholds:
                        issues.append(f"Missing routing threshold: {threshold}")
                    elif not isinstance(thresholds[threshold], (int, float)):
                        issues.append(f"Invalid routing threshold type: {threshold}")
                    elif not (0 <= thresholds[threshold] <= 1):
                        issues.append(f"Routing threshold out of range: {threshold}")
            
            # Validate LLM settings
            if 'llm_settings' in config:
                llm_settings = config['llm_settings']
                if 'timeout' in llm_settings and llm_settings['timeout'] <= 0:
                    issues.append("LLM timeout must be positive")
                if 'daily_budget' in llm_settings and llm_settings['daily_budget'] < 0:
                    issues.append("LLM daily budget must be non-negative")
            
            # Validate feature flags
            if 'feature_flags' in config:
                flags = config['feature_flags']
                for flag_name, flag_value in flags.items():
                    if not isinstance(flag_value, bool):
                        issues.append(f"Feature flag must be boolean: {flag_name}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Configuration validation error: {str(e)}")
            return False, issues
    
    def rollback_configuration(self) -> bool:
        """Rollback to previous configuration."""
        try:
            backup_path = Path(self.config.backup_dir) / "config_backup.json"
            if not backup_path.exists():
                self.logger.error("No configuration backup found for rollback")
                return False
            
            # Restore from backup
            current_config_path = Path(self.config.test_data_dir) / "migrated_config.json"
            shutil.copy2(backup_path, current_config_path)
            
            self.logger.info("Configuration rolled back successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration rollback failed: {e}")
            return False


class DatabaseMigrator:
    """Handles database migration and validation."""
    
    def __init__(self, config: MigrationTestConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def setup_test_database(self) -> bool:
        """Setup test database with initial data."""
        try:
            # Create test database
            Path(self.config.test_db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.config.test_db_path)
            cursor = conn.cursor()
            
            # Create tables (simplified cost tracking schema)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    cost_cents INTEGER NOT NULL,
                    provider TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS confidence_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    predicted_confidence REAL NOT NULL,
                    actual_accuracy REAL NOT NULL,
                    query_hash TEXT NOT NULL
                )
            ''')
            
            # Insert test data
            test_data = [
                ('2025-08-01 10:00:00', 'What are metabolomics biomarkers?', 'biomarker_discovery', 15, 'openai'),
                ('2025-08-01 11:00:00', 'LC-MS analysis methods', 'methodology', 12, 'openai'),
                ('2025-08-01 12:00:00', 'Latest diabetes research 2024', 'literature_search', 20, 'openai')
            ]
            
            cursor.executemany('''
                INSERT INTO query_costs (timestamp, query_text, category, cost_cents, provider)
                VALUES (?, ?, ?, ?, ?)
            ''', test_data)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Test database setup completed: {self.config.test_db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Test database setup failed: {e}")
            return False
    
    def backup_database(self) -> bool:
        """Backup database before migration."""
        try:
            if not Path(self.config.test_db_path).exists():
                self.logger.error("Test database does not exist")
                return False
            
            Path(self.config.backup_db_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.config.test_db_path, self.config.backup_db_path)
            
            self.logger.info(f"Database backed up to {self.config.backup_db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return False
    
    def migrate_database_schema(self) -> bool:
        """Migrate database schema for new features."""
        try:
            conn = sqlite3.connect(self.config.test_db_path)
            cursor = conn.cursor()
            
            # Add new columns for enhanced features
            migrations = [
                '''
                ALTER TABLE query_costs 
                ADD COLUMN llm_enhanced BOOLEAN DEFAULT 0
                ''',
                '''
                ALTER TABLE query_costs 
                ADD COLUMN confidence_score REAL DEFAULT 0.5
                ''',
                '''
                CREATE TABLE IF NOT EXISTS llm_classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    classification_result TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    used_fallback BOOLEAN DEFAULT 0
                )
                '''
            ]
            
            for migration in migrations:
                try:
                    cursor.execute(migration)
                    self.logger.debug(f"Executed migration: {migration[:50]}...")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e) or "already exists" in str(e):
                        # Column or table already exists, skip
                        continue
                    else:
                        raise e
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database schema migration completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Database schema migration failed: {e}")
            return False
    
    def validate_data_integrity(self) -> Tuple[bool, List[str]]:
        """Validate data integrity after migration."""
        issues = []
        
        try:
            conn = sqlite3.connect(self.config.test_db_path)
            cursor = conn.cursor()
            
            # Check that original data is preserved
            cursor.execute("SELECT COUNT(*) FROM query_costs WHERE timestamp < '2025-08-01 13:00:00'")
            original_count = cursor.fetchone()[0]
            
            if original_count < 3:
                issues.append(f"Original data missing: expected 3, found {original_count}")
            
            # Check new columns exist
            cursor.execute("PRAGMA table_info(query_costs)")
            columns = {row[1] for row in cursor.fetchall()}
            
            expected_new_columns = {'llm_enhanced', 'confidence_score'}
            missing_columns = expected_new_columns - columns
            if missing_columns:
                issues.append(f"Missing new columns: {missing_columns}")
            
            # Check new table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_classifications'")
            if not cursor.fetchone():
                issues.append("New table 'llm_classifications' not found")
            
            # Validate data types and constraints
            cursor.execute("SELECT confidence_score FROM query_costs WHERE confidence_score IS NOT NULL")
            confidence_scores = cursor.fetchall()
            for (score,) in confidence_scores:
                if not (0 <= score <= 1):
                    issues.append(f"Invalid confidence score: {score}")
            
            conn.close()
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Data integrity validation error: {str(e)}")
            return False, issues
    
    def rollback_database(self) -> bool:
        """Rollback database to backup."""
        try:
            if not Path(self.config.backup_db_path).exists():
                self.logger.error("No database backup found for rollback")
                return False
            
            shutil.copy2(self.config.backup_db_path, self.config.test_db_path)
            
            self.logger.info("Database rolled back successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Database rollback failed: {e}")
            return False


class DeploymentOrchestrator:
    """Orchestrates the entire deployment process."""
    
    def __init__(self, config: MigrationTestConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.service_manager = MockServiceManager(config, logger)
        self.config_migrator = ConfigurationMigrator(config, logger)
        self.db_migrator = DatabaseMigrator(config, logger)
        
        # Deployment state
        self.state = DeploymentState()
        self.deployment_log = []
        
        # Health monitoring
        self.health_monitor_active = False
        self.health_monitor_thread = None
    
    def _log_deployment_event(self, event: str, details: str = ""):
        """Log deployment event."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'details': details,
            'state_phase': self.state.phase
        }
        self.deployment_log.append(log_entry)
        self.logger.info(f"Deployment: {event} - {details}")
    
    def _start_health_monitoring(self):
        """Start health monitoring in background thread."""
        def monitor_health():
            while self.health_monitor_active:
                try:
                    # Check old service health
                    old_healthy = self.service_manager.health_check('old_service')
                    new_healthy = self.service_manager.health_check('new_service')
                    
                    self.state.old_service_healthy = old_healthy
                    self.state.new_service_healthy = new_healthy
                    
                    # Get performance metrics
                    old_metrics = self.service_manager.get_service_metrics('old_service')
                    new_metrics = self.service_manager.get_service_metrics('new_service')
                    
                    if old_metrics:
                        self.state.response_times.append(old_metrics['avg_response_time_ms'])
                        self.state.error_rates.append(1.0 - old_metrics['success_rate'])
                    
                    # Check rollback conditions
                    if len(self.state.error_rates) >= 5:  # Check last 5 measurements
                        recent_error_rate = sum(self.state.error_rates[-5:]) / 5
                        if recent_error_rate > self.config.rollback_threshold:
                            self._log_deployment_event("ROLLBACK_TRIGGERED", 
                                                     f"High error rate: {recent_error_rate:.3f}")
                            break
                    
                    time.sleep(self.config.health_check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(self.config.health_check_interval)
        
        self.health_monitor_active = True
        self.health_monitor_thread = threading.Thread(target=monitor_health)
        self.health_monitor_thread.daemon = True
        self.health_monitor_thread.start()
    
    def _stop_health_monitoring(self):
        """Stop health monitoring."""
        self.health_monitor_active = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=10)
    
    async def run_pre_deployment_checks(self) -> Tuple[bool, List[str]]:
        """Run comprehensive pre-deployment validation."""
        self.state.phase = "pre_check"
        self._log_deployment_event("PRE_DEPLOYMENT_CHECKS_START")
        
        issues = []
        
        try:
            # 1. Setup test environment
            if not self.db_migrator.setup_test_database():
                issues.append("Failed to setup test database")
            
            # 2. Backup current state
            if not self.config_migrator.backup_configuration():
                issues.append("Failed to backup configuration")
            
            if not self.db_migrator.backup_database():
                issues.append("Failed to backup database")
            
            # 3. Validate current system health
            if not self.service_manager.start_service('old_service', 'baseline'):
                issues.append("Failed to start baseline service")
            
            # Wait for service startup
            await asyncio.sleep(2)
            
            if not self.service_manager.health_check('old_service'):
                issues.append("Baseline service health check failed")
            
            # 4. Test current system functionality
            baseline_router = BiomedicalQueryRouter()
            test_query = "What are metabolomics biomarkers for diabetes?"
            
            try:
                prediction = baseline_router.route_query(test_query)
                if prediction is None:
                    issues.append("Baseline system query processing failed")
                elif prediction.confidence < 0.3:
                    issues.append("Baseline system confidence too low")
            except Exception as e:
                issues.append(f"Baseline system error: {str(e)}")
            
            # 5. Validate migration scripts
            config_valid, config_issues = self.config_migrator.validate_configuration()
            if not config_valid:
                issues.extend([f"Config validation: {issue}" for issue in config_issues])
            
            success = len(issues) == 0
            
            if success:
                self._log_deployment_event("PRE_DEPLOYMENT_CHECKS_PASSED")
            else:
                self._log_deployment_event("PRE_DEPLOYMENT_CHECKS_FAILED", 
                                         f"{len(issues)} issues found")
            
            return success, issues
            
        except Exception as e:
            issues.append(f"Pre-deployment check error: {str(e)}")
            return False, issues
    
    async def execute_zero_downtime_deployment(self) -> Tuple[bool, List[str]]:
        """Execute zero-downtime deployment with gradual traffic shift."""
        self.state.phase = "deploying"
        self.state.start_time = datetime.now()
        self._log_deployment_event("ZERO_DOWNTIME_DEPLOYMENT_START")
        
        issues = []
        
        try:
            # 1. Migrate configuration
            if not self.config_migrator.migrate_configuration():
                issues.append("Configuration migration failed")
                return False, issues
            
            # 2. Migrate database schema
            if not self.db_migrator.migrate_database_schema():
                issues.append("Database schema migration failed")
                return False, issues
            
            # 3. Start new service with migrated configuration
            self._log_deployment_event("STARTING_NEW_SERVICE")
            if not self.service_manager.start_service('new_service', 'enhanced'):
                issues.append("Failed to start new service")
                return False, issues
            
            # Wait for new service to be ready
            await asyncio.sleep(3)
            
            # 4. Health check new service
            if not self.service_manager.health_check('new_service'):
                issues.append("New service health check failed")
                return False, issues
            
            # 5. Start health monitoring
            self._start_health_monitoring()
            
            # 6. Gradual traffic shifting
            if self.config.enable_gradual_rollout:
                await self._execute_gradual_rollout()
            else:
                # Immediate switch
                self._log_deployment_event("IMMEDIATE_TRAFFIC_SWITCH")
                if not self.service_manager.split_traffic('old_service', 'new_service', 100.0):
                    issues.append("Failed to switch traffic")
                    return False, issues
            
            # 7. Monitor deployment for stability period
            self._log_deployment_event("MONITORING_STABILITY")
            await asyncio.sleep(10)  # Monitor for 10 seconds in test
            
            # 8. Validate data integrity after migration
            data_valid, data_issues = self.db_migrator.validate_data_integrity()
            if not data_valid:
                issues.extend([f"Data integrity: {issue}" for issue in data_issues])
            
            success = len(issues) == 0 and self.state.new_service_healthy
            
            if success:
                self._log_deployment_event("ZERO_DOWNTIME_DEPLOYMENT_SUCCESS")
                self.state.phase = "completed"
                self.state.end_time = datetime.now()
            else:
                self._log_deployment_event("ZERO_DOWNTIME_DEPLOYMENT_FAILED",
                                         f"{len(issues)} issues")
                self.state.phase = "failed"
            
            return success, issues
            
        except Exception as e:
            issues.append(f"Deployment execution error: {str(e)}")
            self.state.phase = "failed"
            return False, issues
        
        finally:
            self._stop_health_monitoring()
    
    async def _execute_gradual_rollout(self):
        """Execute gradual traffic rollout."""
        rollout_steps = [10, 25, 50, 75, 100]  # Percentage steps
        
        for percentage in rollout_steps:
            self._log_deployment_event("TRAFFIC_SPLIT", f"{percentage}% to new service")
            
            # Update traffic split
            self.service_manager.split_traffic('old_service', 'new_service', percentage)
            self.state.traffic_split_percentage = percentage
            
            # Monitor for stability
            monitoring_duration = 5 if percentage < 100 else 10  # seconds
            await asyncio.sleep(monitoring_duration)
            
            # Check for issues
            if not self.state.new_service_healthy:
                self._log_deployment_event("ROLLOUT_PAUSED", "New service unhealthy")
                break
            
            # Check error rates
            if len(self.state.error_rates) >= 3:
                recent_error_rate = sum(self.state.error_rates[-3:]) / 3
                if recent_error_rate > self.config.rollback_threshold:
                    self._log_deployment_event("ROLLOUT_STOPPED", "High error rate detected")
                    break
    
    async def test_rollback_procedure(self) -> Tuple[bool, List[str]]:
        """Test rollback procedure."""
        self._log_deployment_event("ROLLBACK_TEST_START")
        
        issues = []
        
        try:
            # 1. Stop new service
            if not self.service_manager.stop_service('new_service'):
                issues.append("Failed to stop new service")
            
            # 2. Restore traffic to old service
            if not self.service_manager.split_traffic('old_service', 'new_service', 0.0):
                issues.append("Failed to restore traffic to old service")
            
            # 3. Rollback database
            if not self.db_migrator.rollback_database():
                issues.append("Database rollback failed")
            
            # 4. Rollback configuration
            if not self.config_migrator.rollback_configuration():
                issues.append("Configuration rollback failed")
            
            # 5. Validate old service still works
            await asyncio.sleep(2)
            if not self.service_manager.health_check('old_service'):
                issues.append("Old service health check failed after rollback")
            
            # 6. Test functionality after rollback
            baseline_router = BiomedicalQueryRouter()
            test_query = "What are metabolomics biomarkers for diabetes?"
            
            try:
                prediction = baseline_router.route_query(test_query)
                if prediction is None:
                    issues.append("System functionality failed after rollback")
                elif prediction.confidence < 0.3:
                    issues.append("System performance degraded after rollback")
            except Exception as e:
                issues.append(f"System error after rollback: {str(e)}")
            
            success = len(issues) == 0
            
            if success:
                self._log_deployment_event("ROLLBACK_TEST_SUCCESS")
            else:
                self._log_deployment_event("ROLLBACK_TEST_FAILED", 
                                         f"{len(issues)} issues")
            
            return success, issues
            
        except Exception as e:
            issues.append(f"Rollback test error: {str(e)}")
            return False, issues
    
    def get_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        
        # Calculate deployment metrics
        total_time = None
        if self.state.start_time and self.state.end_time:
            total_time = (self.state.end_time - self.state.start_time).total_seconds()
        
        # Calculate performance metrics
        avg_response_time = sum(self.state.response_times) / len(self.state.response_times) if self.state.response_times else 0
        avg_error_rate = sum(self.state.error_rates) / len(self.state.error_rates) if self.state.error_rates else 0
        
        success_rate = self.state.success_count / max(1, self.state.success_count + self.state.failure_count)
        
        return {
            'deployment_summary': {
                'phase': self.state.phase,
                'total_time_seconds': total_time,
                'overall_success': self.state.phase == 'completed',
                'zero_downtime_achieved': total_time is not None and total_time < self.config.max_acceptable_downtime + 1
            },
            'service_health': {
                'old_service_healthy': self.state.old_service_healthy,
                'new_service_healthy': self.state.new_service_healthy,
                'final_traffic_split': self.state.traffic_split_percentage
            },
            'performance_metrics': {
                'average_response_time_ms': avg_response_time,
                'average_error_rate': avg_error_rate,
                'success_rate': success_rate,
                'meets_sla': avg_error_rate < self.config.rollback_threshold and success_rate > self.config.min_success_rate
            },
            'migration_status': {
                'config_migrated': self.state.config_migrated,
                'database_migrated': self.state.database_migrated,
                'features_enabled': self.state.features_enabled
            },
            'deployment_log': self.deployment_log[-20:],  # Last 20 events
            'recommendations': self._generate_deployment_recommendations()
        }
    
    def _generate_deployment_recommendations(self) -> List[Dict[str, str]]:
        """Generate deployment recommendations based on results."""
        recommendations = []
        
        # Analyze deployment performance
        if self.state.phase == 'completed':
            recommendations.append({
                'type': 'success',
                'priority': 'info',
                'recommendation': 'Deployment completed successfully. Monitor system for 24 hours.',
                'action': 'Continue monitoring'
            })
        elif self.state.phase == 'failed':
            recommendations.append({
                'type': 'failure',
                'priority': 'critical',
                'recommendation': 'Deployment failed. Execute rollback procedure immediately.',
                'action': 'Execute rollback'
            })
        
        # Analyze performance impact
        if len(self.state.response_times) > 0:
            avg_response_time = sum(self.state.response_times) / len(self.state.response_times)
            if avg_response_time > 100:  # 100ms threshold
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'recommendation': f'Response time elevated ({avg_response_time:.1f}ms). Investigate performance optimization.',
                    'action': 'Optimize performance'
                })
        
        # Analyze error rates
        if len(self.state.error_rates) > 0:
            avg_error_rate = sum(self.state.error_rates) / len(self.state.error_rates)
            if avg_error_rate > 0.02:  # 2% threshold
                recommendations.append({
                    'type': 'reliability',
                    'priority': 'high',
                    'recommendation': f'Error rate elevated ({avg_error_rate:.1%}). Investigate error sources.',
                    'action': 'Debug errors'
                })
        
        return recommendations


class MigrationTestSuite:
    """Comprehensive migration test suite."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup test configuration
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config = MigrationTestConfig(
                test_data_dir=os.path.join(temp_dir, "test_data"),
                backup_dir=os.path.join(temp_dir, "backups"),
                temp_dir=temp_dir,
                test_db_path=os.path.join(temp_dir, "test_data", "test.db"),
                backup_db_path=os.path.join(temp_dir, "backups", "test_backup.db")
            )
    
    async def test_complete_migration_workflow(self) -> Dict[str, Any]:
        """Test complete migration workflow from start to finish."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fresh config for this test
            config = MigrationTestConfig(
                test_data_dir=os.path.join(temp_dir, "test_data"),
                backup_dir=os.path.join(temp_dir, "backups"),
                temp_dir=temp_dir,
                test_db_path=os.path.join(temp_dir, "test_data", "test.db"),
                backup_db_path=os.path.join(temp_dir, "backups", "test_backup.db")
            )
            
            orchestrator = DeploymentOrchestrator(config, self.logger)
            
            workflow_results = {
                'test_name': 'Complete Migration Workflow',
                'passed': True,
                'failures': [],
                'workflow_stages': {}
            }
            
            try:
                # Stage 1: Pre-deployment checks
                self.logger.info("Stage 1: Pre-deployment checks")
                pre_check_success, pre_check_issues = await orchestrator.run_pre_deployment_checks()
                
                workflow_results['workflow_stages']['pre_deployment'] = {
                    'passed': pre_check_success,
                    'issues': pre_check_issues
                }
                
                if not pre_check_success:
                    workflow_results['passed'] = False
                    workflow_results['failures'].extend(pre_check_issues)
                
                # Stage 2: Zero-downtime deployment
                self.logger.info("Stage 2: Zero-downtime deployment")
                deployment_success, deployment_issues = await orchestrator.execute_zero_downtime_deployment()
                
                workflow_results['workflow_stages']['deployment'] = {
                    'passed': deployment_success,
                    'issues': deployment_issues
                }
                
                if not deployment_success:
                    workflow_results['passed'] = False
                    workflow_results['failures'].extend(deployment_issues)
                
                # Stage 3: Post-deployment validation
                self.logger.info("Stage 3: Post-deployment validation")
                post_validation_success = await self._validate_post_deployment(orchestrator)
                
                workflow_results['workflow_stages']['post_validation'] = {
                    'passed': post_validation_success,
                    'enhanced_features_working': post_validation_success
                }
                
                if not post_validation_success:
                    workflow_results['passed'] = False
                    workflow_results['failures'].append("Post-deployment validation failed")
                
                # Stage 4: Rollback testing
                self.logger.info("Stage 4: Rollback testing")
                rollback_success, rollback_issues = await orchestrator.test_rollback_procedure()
                
                workflow_results['workflow_stages']['rollback_test'] = {
                    'passed': rollback_success,
                    'issues': rollback_issues
                }
                
                if not rollback_success:
                    workflow_results['passed'] = False
                    workflow_results['failures'].extend(rollback_issues)
                
                # Generate deployment report
                workflow_results['deployment_report'] = orchestrator.get_deployment_report()
                
            except Exception as e:
                workflow_results['passed'] = False
                workflow_results['failures'].append(f"Workflow test error: {str(e)}")
            
            return workflow_results
    
    async def _validate_post_deployment(self, orchestrator: DeploymentOrchestrator) -> bool:
        """Validate system functionality after deployment."""
        try:
            # Test that enhanced features work
            # (In real implementation, this would test actual enhanced functionality)
            
            # Simulate enhanced system testing
            await asyncio.sleep(1)
            
            # Check service health
            if not orchestrator.state.new_service_healthy:
                return False
            
            # Test configuration is working
            config_path = Path(orchestrator.config.test_data_dir) / "migrated_config.json"
            if not config_path.exists():
                return False
            
            # Test database migration worked
            data_valid, _ = orchestrator.db_migrator.validate_data_integrity()
            if not data_valid:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Post-deployment validation error: {e}")
            return False


# Pytest test class
@pytest.mark.asyncio
class TestMigrationFramework:
    """Main test class for migration framework testing."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_suite = MigrationTestSuite(self.logger)
    
    async def test_complete_migration_workflow(self):
        """Test complete migration workflow."""
        results = await self.test_suite.test_complete_migration_workflow()
        
        # Assert overall success
        assert results['passed'], f"Migration workflow failed: {results['failures']}"
        
        # Assert all stages passed
        stages = results['workflow_stages']
        for stage_name, stage_result in stages.items():
            assert stage_result['passed'], f"Migration stage {stage_name} failed: {stage_result.get('issues', [])}"
        
        # Assert deployment report shows success
        deployment_report = results.get('deployment_report', {})
        deployment_summary = deployment_report.get('deployment_summary', {})
        assert deployment_summary.get('overall_success', False), "Deployment report shows failure"
        assert deployment_summary.get('zero_downtime_achieved', False), "Zero downtime not achieved"
        
        self.logger.info("Complete migration workflow test passed successfully")
    
    async def test_rollback_procedure_isolation(self):
        """Test rollback procedure in isolation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MigrationTestConfig(
                test_data_dir=os.path.join(temp_dir, "test_data"),
                backup_dir=os.path.join(temp_dir, "backups"),
                temp_dir=temp_dir,
                test_db_path=os.path.join(temp_dir, "test_data", "test.db"),
                backup_db_path=os.path.join(temp_dir, "backups", "test_backup.db")
            )
            
            orchestrator = DeploymentOrchestrator(config, self.logger)
            
            # Setup initial state
            await orchestrator.run_pre_deployment_checks()
            
            # Simulate failed deployment that needs rollback
            await orchestrator.execute_zero_downtime_deployment()
            
            # Test rollback
            rollback_success, rollback_issues = await orchestrator.test_rollback_procedure()
            
            assert rollback_success, f"Rollback failed: {rollback_issues}"
            self.logger.info("Rollback procedure test passed")
    
    async def test_configuration_migration_safety(self):
        """Test that configuration migration is safe and reversible."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MigrationTestConfig(
                test_data_dir=os.path.join(temp_dir, "test_data"),
                backup_dir=os.path.join(temp_dir, "backups"),
                temp_dir=temp_dir,
                test_db_path=os.path.join(temp_dir, "test_data", "test.db"),
                backup_db_path=os.path.join(temp_dir, "backups", "test_backup.db")
            )
            
            config_migrator = ConfigurationMigrator(config, self.logger)
            
            # Test backup
            backup_success = config_migrator.backup_configuration()
            assert backup_success, "Configuration backup failed"
            
            # Test migration
            migrate_success = config_migrator.migrate_configuration()
            assert migrate_success, "Configuration migration failed"
            
            # Test validation
            valid, issues = config_migrator.validate_configuration()
            assert valid, f"Configuration validation failed: {issues}"
            
            # Test rollback
            rollback_success = config_migrator.rollback_configuration()
            assert rollback_success, "Configuration rollback failed"
            
            self.logger.info("Configuration migration safety test passed")
    
    async def test_database_migration_integrity(self):
        """Test that database migration preserves data integrity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MigrationTestConfig(
                test_data_dir=os.path.join(temp_dir, "test_data"),
                backup_dir=os.path.join(temp_dir, "backups"),
                temp_dir=temp_dir,
                test_db_path=os.path.join(temp_dir, "test_data", "test.db"),
                backup_db_path=os.path.join(temp_dir, "backups", "test_backup.db")
            )
            
            db_migrator = DatabaseMigrator(config, self.logger)
            
            # Setup test database
            setup_success = db_migrator.setup_test_database()
            assert setup_success, "Test database setup failed"
            
            # Test backup
            backup_success = db_migrator.backup_database()
            assert backup_success, "Database backup failed"
            
            # Test schema migration
            migrate_success = db_migrator.migrate_database_schema()
            assert migrate_success, "Database schema migration failed"
            
            # Test data integrity
            integrity_valid, integrity_issues = db_migrator.validate_data_integrity()
            assert integrity_valid, f"Data integrity validation failed: {integrity_issues}"
            
            # Test rollback
            rollback_success = db_migrator.rollback_database()
            assert rollback_success, "Database rollback failed"
            
            self.logger.info("Database migration integrity test passed")


# Export main classes
__all__ = [
    'TestMigrationFramework',
    'MigrationTestSuite',
    'DeploymentOrchestrator',
    'ConfigurationMigrator',
    'DatabaseMigrator',
    'MigrationTestConfig',
    'DeploymentState'
]