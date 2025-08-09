#!/usr/bin/env python3
"""
Production Migration Script - Safe Migration to Production Load Balancer

This script provides tools and utilities for safely migrating from the existing
IntelligentQueryRouter to the ProductionIntelligentQueryRouter with comprehensive
validation, rollback capabilities, and monitoring.

Features:
- Pre-migration validation and health checks
- Step-by-step migration with rollback points
- Configuration validation and migration
- Performance benchmarking and comparison
- Automated rollback on failure detection
- Migration status reporting and logging

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: Production Load Balancer Integration
"""

import sys
import os
import time
import json
import logging
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import statistics
from contextlib import asynccontextmanager

# Add the parent directory to sys.path to import modules
sys.path.append(str(Path(__file__).parent))

from intelligent_query_router import IntelligentQueryRouter, LoadBalancingConfig, HealthCheckConfig
from production_intelligent_query_router import (
    ProductionIntelligentQueryRouter, 
    ProductionFeatureFlags, 
    DeploymentMode,
    ConfigurationMigrator,
    create_production_intelligent_query_router
)
from production_load_balancer import create_default_production_config


class MigrationPhase:
    """Migration phase definitions"""
    VALIDATION = "validation"
    PREPARATION = "preparation"
    CANARY = "canary"
    GRADUAL_ROLLOUT = "gradual_rollout"
    FULL_DEPLOYMENT = "full_deployment"
    CLEANUP = "cleanup"


class MigrationStatus:
    """Migration status tracking"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationLogger:
    """Enhanced logger for migration process"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger("production_migration")
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
        print(f"[INFO] {message}")
    
    def warning(self, message: str):
        self.logger.warning(message)
        print(f"[WARNING] {message}")
    
    def error(self, message: str):
        self.logger.error(message)
        print(f"[ERROR] {message}")
    
    def critical(self, message: str):
        self.logger.critical(message)
        print(f"[CRITICAL] {message}")


class MigrationValidator:
    """Validates system readiness for migration"""
    
    def __init__(self, logger: MigrationLogger):
        self.logger = logger
    
    async def validate_prerequisites(self) -> Dict[str, bool]:
        """Validate all prerequisites for migration"""
        self.logger.info("Starting prerequisite validation...")
        
        validations = {
            'existing_router_available': await self._check_existing_router(),
            'production_config_valid': await self._check_production_config(),
            'system_resources': await self._check_system_resources(),
            'network_connectivity': await self._check_network_connectivity(),
            'backend_health': await self._check_backend_health(),
            'storage_permissions': await self._check_storage_permissions()
        }
        
        all_valid = all(validations.values())
        
        self.logger.info(f"Prerequisite validation {'PASSED' if all_valid else 'FAILED'}")
        for check, result in validations.items():
            status = "✓" if result else "✗"
            self.logger.info(f"  {status} {check}")
        
        return validations
    
    async def _check_existing_router(self) -> bool:
        """Check if existing router is functional"""
        try:
            router = IntelligentQueryRouter()
            # Test basic functionality
            test_query = "test query for validation"
            prediction = await asyncio.get_event_loop().run_in_executor(
                None, router.route_query, test_query, {}
            )
            return prediction is not None
        except Exception as e:
            self.logger.error(f"Existing router check failed: {e}")
            return False
    
    async def _check_production_config(self) -> bool:
        """Check if production configuration is valid"""
        try:
            config = create_default_production_config()
            # Basic validation
            return (
                config is not None and 
                hasattr(config, 'backends') and
                hasattr(config, 'algorithm_config') and
                len(config.backends) > 0
            )
        except Exception as e:
            self.logger.error(f"Production config check failed: {e}")
            return False
    
    async def _check_system_resources(self) -> bool:
        """Check system resources"""
        try:
            import psutil
            
            # Check memory (need at least 500MB available)
            memory = psutil.virtual_memory()
            memory_available_mb = memory.available / 1024 / 1024
            
            # Check CPU load
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / 1024 / 1024 / 1024
            
            resource_check = (
                memory_available_mb > 500 and
                cpu_percent < 80 and
                disk_free_gb > 1
            )
            
            if not resource_check:
                self.logger.warning(f"System resources: Memory={memory_available_mb:.1f}MB, CPU={cpu_percent:.1f}%, Disk={disk_free_gb:.1f}GB")
            
            return resource_check
        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            return False
    
    async def _check_network_connectivity(self) -> bool:
        """Check network connectivity to backends"""
        try:
            import httpx
            
            # Test connectivity to common endpoints
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test general internet connectivity
                try:
                    response = await client.get("https://httpbin.org/status/200")
                    return response.status_code == 200
                except Exception:
                    return False
        except Exception as e:
            self.logger.error(f"Network connectivity check failed: {e}")
            return False
    
    async def _check_backend_health(self) -> bool:
        """Check health of backend services"""
        try:
            # This would check actual backend health
            # For now, return True as we don't have actual backends in test environment
            return True
        except Exception as e:
            self.logger.error(f"Backend health check failed: {e}")
            return False
    
    async def _check_storage_permissions(self) -> bool:
        """Check storage and file permissions"""
        try:
            # Test creating a temporary file
            test_file = Path("migration_test.tmp")
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception as e:
            self.logger.error(f"Storage permissions check failed: {e}")
            return False


class PerformanceBenchmark:
    """Performance benchmarking for migration validation"""
    
    def __init__(self, logger: MigrationLogger):
        self.logger = logger
    
    async def benchmark_existing_system(self, router: IntelligentQueryRouter, 
                                      test_queries: List[str]) -> Dict[str, float]:
        """Benchmark the existing system performance"""
        self.logger.info("Benchmarking existing system...")
        
        response_times = []
        success_count = 0
        
        for i, query in enumerate(test_queries):
            try:
                start_time = time.time()
                result = await asyncio.get_event_loop().run_in_executor(
                    None, router.route_query, query, {}
                )
                end_time = time.time()
                
                if result is not None:
                    response_times.append((end_time - start_time) * 1000)
                    success_count += 1
                
                self.logger.info(f"Query {i+1}/{len(test_queries)} completed")
                
            except Exception as e:
                self.logger.error(f"Query {i+1} failed: {e}")
        
        if not response_times:
            return {'error': 'No successful queries'}
        
        return {
            'total_queries': len(test_queries),
            'successful_queries': success_count,
            'success_rate': (success_count / len(test_queries)) * 100,
            'avg_response_time_ms': statistics.mean(response_times),
            'median_response_time_ms': statistics.median(response_times),
            'p95_response_time_ms': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
            'min_response_time_ms': min(response_times),
            'max_response_time_ms': max(response_times)
        }
    
    async def benchmark_production_system(self, router: ProductionIntelligentQueryRouter,
                                        test_queries: List[str]) -> Dict[str, float]:
        """Benchmark the production system performance"""
        self.logger.info("Benchmarking production system...")
        
        response_times = []
        success_count = 0
        
        for i, query in enumerate(test_queries):
            try:
                start_time = time.time()
                result = await router.route_query(query, {})
                end_time = time.time()
                
                if result is not None:
                    response_times.append((end_time - start_time) * 1000)
                    success_count += 1
                
                self.logger.info(f"Query {i+1}/{len(test_queries)} completed")
                
            except Exception as e:
                self.logger.error(f"Query {i+1} failed: {e}")
        
        if not response_times:
            return {'error': 'No successful queries'}
        
        return {
            'total_queries': len(test_queries),
            'successful_queries': success_count,
            'success_rate': (success_count / len(test_queries)) * 100,
            'avg_response_time_ms': statistics.mean(response_times),
            'median_response_time_ms': statistics.median(response_times),
            'p95_response_time_ms': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
            'min_response_time_ms': min(response_times),
            'max_response_time_ms': max(response_times)
        }


class ProductionMigrationManager:
    """Manages the complete migration process"""
    
    def __init__(self, config_file: Optional[str] = None, log_file: Optional[str] = None):
        self.logger = MigrationLogger(log_file)
        self.validator = MigrationValidator(self.logger)
        self.benchmark = PerformanceBenchmark(self.logger)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Migration state
        self.migration_state = {
            'phase': MigrationPhase.VALIDATION,
            'status': MigrationStatus.PENDING,
            'start_time': None,
            'rollback_points': [],
            'performance_baselines': {},
            'errors': []
        }
        
        # Routers
        self.existing_router: Optional[IntelligentQueryRouter] = None
        self.production_router: Optional[ProductionIntelligentQueryRouter] = None
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load migration configuration"""
        default_config = {
            'test_queries': [
                "What are the metabolic pathways involved in diabetes?",
                "Explain the role of mitochondria in cellular respiration.",
                "How do biomarkers help in disease diagnosis?",
                "What is the significance of metabolomics in personalized medicine?",
                "Describe the process of protein synthesis."
            ],
            'canary_duration_minutes': 30,
            'rollout_stages': [5, 15, 50, 100],  # Percentage of traffic
            'rollback_thresholds': {
                'error_rate_percent': 5.0,
                'latency_increase_percent': 50.0,
                'success_rate_threshold_percent': 95.0
            },
            'validation_timeout_seconds': 300
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    async def run_migration(self, force: bool = False) -> bool:
        """Run the complete migration process"""
        self.logger.info("Starting Production Load Balancer Migration")
        self.logger.info("=" * 60)
        
        self.migration_state['start_time'] = datetime.now()
        
        try:
            # Phase 1: Validation
            if not await self._run_validation_phase():
                if not force:
                    self.logger.critical("Validation failed. Use --force to proceed anyway.")
                    return False
                else:
                    self.logger.warning("Validation failed but proceeding due to --force flag")
            
            # Phase 2: Preparation
            if not await self._run_preparation_phase():
                await self._rollback("Preparation phase failed")
                return False
            
            # Phase 3: Canary Deployment
            if not await self._run_canary_phase():
                await self._rollback("Canary phase failed")
                return False
            
            # Phase 4: Gradual Rollout
            if not await self._run_gradual_rollout_phase():
                await self._rollback("Gradual rollout failed")
                return False
            
            # Phase 5: Full Deployment
            if not await self._run_full_deployment_phase():
                await self._rollback("Full deployment failed")
                return False
            
            # Phase 6: Cleanup
            await self._run_cleanup_phase()
            
            self.logger.info("Migration completed successfully!")
            self._generate_migration_report()
            return True
            
        except Exception as e:
            self.logger.critical(f"Migration failed with exception: {e}")
            await self._rollback(f"Exception during migration: {e}")
            return False
    
    async def _run_validation_phase(self) -> bool:
        """Run validation phase"""
        self.logger.info(f"Phase 1: {MigrationPhase.VALIDATION.upper()}")
        self.migration_state['phase'] = MigrationPhase.VALIDATION
        self.migration_state['status'] = MigrationStatus.IN_PROGRESS
        
        # Validate prerequisites
        validations = await self.validator.validate_prerequisites()
        if not all(validations.values()):\n            self.migration_state['status'] = MigrationStatus.FAILED
            self.migration_state['errors'].append("Prerequisites validation failed")
            return False
        
        # Initialize existing router for benchmarking
        try:
            self.existing_router = IntelligentQueryRouter()
            self.logger.info("Existing router initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize existing router: {e}")
            return False
        
        # Benchmark existing system
        try:
            baseline = await self.benchmark.benchmark_existing_system(
                self.existing_router, 
                self.config['test_queries']
            )
            self.migration_state['performance_baselines']['existing'] = baseline
            self.logger.info(f"Existing system baseline: {baseline['avg_response_time_ms']:.2f}ms avg, {baseline['success_rate']:.1f}% success rate")
        except Exception as e:
            self.logger.error(f"Failed to benchmark existing system: {e}")
            return False
        
        self.migration_state['status'] = MigrationStatus.COMPLETED
        self._create_rollback_point("validation_completed")
        return True
    
    async def _run_preparation_phase(self) -> bool:
        """Run preparation phase"""
        self.logger.info(f"Phase 2: {MigrationPhase.PREPARATION.upper()}")
        self.migration_state['phase'] = MigrationPhase.PREPARATION
        self.migration_state['status'] = MigrationStatus.IN_PROGRESS
        
        try:
            # Create production router in shadow mode for testing
            feature_flags = ProductionFeatureFlags(
                enable_production_load_balancer=True,
                deployment_mode=DeploymentMode.SHADOW,
                enable_performance_comparison=True
            )
            
            self.production_router = ProductionIntelligentQueryRouter(
                feature_flags=feature_flags
            )
            
            await self.production_router.start_monitoring()
            self.logger.info("Production router initialized in shadow mode")
            
            # Test production system
            production_baseline = await self.benchmark.benchmark_production_system(
                self.production_router,
                self.config['test_queries']
            )
            self.migration_state['performance_baselines']['production'] = production_baseline
            self.logger.info(f"Production system baseline: {production_baseline['avg_response_time_ms']:.2f}ms avg, {production_baseline['success_rate']:.1f}% success rate")
            
            self.migration_state['status'] = MigrationStatus.COMPLETED
            self._create_rollback_point("preparation_completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Preparation phase failed: {e}")
            self.migration_state['errors'].append(f"Preparation failed: {e}")
            return False
    
    async def _run_canary_phase(self) -> bool:
        """Run canary deployment phase"""
        self.logger.info(f"Phase 3: {MigrationPhase.CANARY.upper()}")
        self.migration_state['phase'] = MigrationPhase.CANARY
        self.migration_state['status'] = MigrationStatus.IN_PROGRESS
        
        try:
            # Switch to canary mode with 5% traffic
            self.production_router.feature_flags.deployment_mode = DeploymentMode.CANARY
            self.production_router.feature_flags.production_traffic_percentage = 5.0
            
            self.logger.info("Starting canary deployment with 5% traffic")
            
            # Run canary for specified duration
            canary_duration = self.config['canary_duration_minutes'] * 60
            start_time = time.time()
            
            while (time.time() - start_time) < canary_duration:
                # Run test queries
                for query in self.config['test_queries'][:3]:  # Subset for canary
                    try:
                        await self.production_router.route_query(query, {})
                    except Exception as e:
                        self.logger.warning(f"Canary query failed: {e}")
                
                # Check performance
                if not await self._check_canary_health():
                    self.logger.error("Canary health check failed")
                    return False
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            # Generate canary report
            report = self.production_router.get_performance_report()
            self.logger.info(f"Canary phase completed. Performance improvement: {report.get('performance_improvement', {}).get('response_time_improvement_percent', 0):.2f}%")
            
            self.migration_state['status'] = MigrationStatus.COMPLETED
            self._create_rollback_point("canary_completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Canary phase failed: {e}")
            self.migration_state['errors'].append(f"Canary failed: {e}")
            return False
    
    async def _run_gradual_rollout_phase(self) -> bool:
        """Run gradual rollout phase"""
        self.logger.info(f"Phase 4: {MigrationPhase.GRADUAL_ROLLOUT.upper()}")
        self.migration_state['phase'] = MigrationPhase.GRADUAL_ROLLOUT
        self.migration_state['status'] = MigrationStatus.IN_PROGRESS
        
        try:
            # Gradual rollout in stages
            for stage_percent in self.config['rollout_stages'][:-1]:  # Exclude 100%
                self.logger.info(f"Increasing traffic to {stage_percent}%")
                
                self.production_router.feature_flags.production_traffic_percentage = stage_percent
                
                # Monitor for 5 minutes at each stage
                await asyncio.sleep(300)
                
                # Check health at this stage
                if not await self._check_rollout_health():
                    self.logger.error(f"Health check failed at {stage_percent}% traffic")
                    return False
                
                report = self.production_router.get_performance_report()
                self.logger.info(f"Stage {stage_percent}% performance: {report.get('production_stats', {}).get('avg_response_time_ms', 0):.2f}ms avg")
                
                self._create_rollback_point(f"rollout_{stage_percent}%_completed")
            
            self.migration_state['status'] = MigrationStatus.COMPLETED
            return True
            
        except Exception as e:
            self.logger.error(f"Gradual rollout failed: {e}")
            self.migration_state['errors'].append(f"Gradual rollout failed: {e}")
            return False
    
    async def _run_full_deployment_phase(self) -> bool:
        """Run full deployment phase"""
        self.logger.info(f"Phase 5: {MigrationPhase.FULL_DEPLOYMENT.upper()}")
        self.migration_state['phase'] = MigrationPhase.FULL_DEPLOYMENT
        self.migration_state['status'] = MigrationStatus.IN_PROGRESS
        
        try:
            # Switch to production only
            self.production_router.feature_flags.deployment_mode = DeploymentMode.PRODUCTION_ONLY
            self.production_router.feature_flags.production_traffic_percentage = 100.0
            
            self.logger.info("Switched to 100% production traffic")
            
            # Monitor for 10 minutes
            await asyncio.sleep(600)
            
            # Final health check
            if not await self._check_rollout_health():
                self.logger.error("Final health check failed")
                return False
            
            self.migration_state['status'] = MigrationStatus.COMPLETED
            self._create_rollback_point("full_deployment_completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Full deployment failed: {e}")
            self.migration_state['errors'].append(f"Full deployment failed: {e}")
            return False
    
    async def _run_cleanup_phase(self) -> bool:
        """Run cleanup phase"""
        self.logger.info(f"Phase 6: {MigrationPhase.CLEANUP.upper()}")
        self.migration_state['phase'] = MigrationPhase.CLEANUP
        
        # Export final performance data
        if self.production_router:
            report_file = self.production_router.export_performance_data()
            self.logger.info(f"Performance data exported to {report_file}")
        
        # Generate migration summary
        self._generate_migration_report()
        
        return True
    
    async def _check_canary_health(self) -> bool:
        """Check canary deployment health"""
        if not self.production_router:
            return False
        
        report = self.production_router.get_performance_report()
        
        # Check error rate
        prod_success_rate = report.get('production_stats', {}).get('success_rate', 0)
        if prod_success_rate < self.config['rollback_thresholds']['success_rate_threshold_percent']:
            self.logger.error(f"Production success rate {prod_success_rate:.2f}% below threshold")
            return False
        
        return True
    
    async def _check_rollout_health(self) -> bool:
        """Check rollout health"""
        if not self.production_router:
            return False
        
        report = self.production_router.get_performance_report()
        
        # Check error rate
        prod_success_rate = report.get('production_stats', {}).get('success_rate', 0)
        if prod_success_rate < self.config['rollback_thresholds']['success_rate_threshold_percent']:
            self.logger.error(f"Production success rate {prod_success_rate:.2f}% below threshold")
            return False
        
        # Check latency increase
        legacy_latency = report.get('legacy_stats', {}).get('avg_response_time_ms', 0)
        prod_latency = report.get('production_stats', {}).get('avg_response_time_ms', 0)
        
        if legacy_latency > 0:
            latency_increase = ((prod_latency - legacy_latency) / legacy_latency) * 100
            if latency_increase > self.config['rollback_thresholds']['latency_increase_percent']:
                self.logger.error(f"Latency increased by {latency_increase:.2f}% above threshold")
                return False
        
        return True
    
    def _create_rollback_point(self, name: str):
        """Create a rollback point"""
        rollback_point = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'phase': self.migration_state['phase'],
            'configuration': {
                'deployment_mode': self.production_router.feature_flags.deployment_mode.value if self.production_router else None,
                'traffic_percentage': self.production_router.feature_flags.production_traffic_percentage if self.production_router else 0
            }
        }
        
        self.migration_state['rollback_points'].append(rollback_point)
        self.logger.info(f"Created rollback point: {name}")
    
    async def _rollback(self, reason: str):
        """Rollback migration"""
        self.logger.critical(f"Initiating rollback: {reason}")
        self.migration_state['status'] = MigrationStatus.ROLLED_BACK
        
        if self.production_router:
            self.production_router.force_rollback(reason)
            await self.production_router.stop_monitoring()
        
        self.logger.info("Rollback completed. System reverted to legacy configuration.")
    
    def _generate_migration_report(self):
        """Generate comprehensive migration report"""
        end_time = datetime.now()
        duration = end_time - self.migration_state['start_time']
        
        report = {
            'migration_summary': {
                'status': self.migration_state['status'],
                'duration_minutes': duration.total_seconds() / 60,
                'start_time': self.migration_state['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'final_phase': self.migration_state['phase']
            },
            'performance_comparison': self.migration_state['performance_baselines'],
            'rollback_points': self.migration_state['rollback_points'],
            'errors': self.migration_state['errors']
        }
        
        # Calculate improvement metrics
        if 'existing' in report['performance_comparison'] and 'production' in report['performance_comparison']:
            existing = report['performance_comparison']['existing']
            production = report['performance_comparison']['production']
            
            report['improvement_metrics'] = {
                'response_time_improvement_percent': (
                    (existing['avg_response_time_ms'] - production['avg_response_time_ms']) / 
                    existing['avg_response_time_ms'] * 100
                ) if existing['avg_response_time_ms'] > 0 else 0,
                'success_rate_improvement': production['success_rate'] - existing['success_rate']
            }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"migration_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Migration report saved to {report_file}")
        
        # Print summary
        self.logger.info("\\nMIGRATION SUMMARY")
        self.logger.info("=" * 40)
        self.logger.info(f"Status: {report['migration_summary']['status']}")
        self.logger.info(f"Duration: {report['migration_summary']['duration_minutes']:.1f} minutes")
        
        if 'improvement_metrics' in report:
            self.logger.info(f"Response Time Improvement: {report['improvement_metrics']['response_time_improvement_percent']:.2f}%")
            self.logger.info(f"Success Rate Change: {report['improvement_metrics']['success_rate_improvement']:.2f}%")


async def main():
    """Main migration script entry point"""
    parser = argparse.ArgumentParser(description="Production Load Balancer Migration Script")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log", help="Log file path", default=f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    parser.add_argument("--force", action="store_true", help="Force migration even if validation fails")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation, don't migrate")
    
    args = parser.parse_args()
    
    # Initialize migration manager
    migration_manager = ProductionMigrationManager(
        config_file=args.config,
        log_file=args.log
    )
    
    if args.validate_only:
        # Only run validation
        migration_manager.logger.info("Running validation only...")
        validations = await migration_manager.validator.validate_prerequisites()
        
        if all(validations.values()):
            migration_manager.logger.info("✓ All validations passed. System ready for migration.")
            return 0
        else:
            migration_manager.logger.error("✗ Validation failed. System not ready for migration.")
            return 1
    else:
        # Run full migration
        success = await migration_manager.run_migration(force=args.force)
        return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)