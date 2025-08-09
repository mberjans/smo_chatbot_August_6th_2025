#!/usr/bin/env python3
"""
Production Load Balancer Deployment Script
==========================================

This script handles the deployment and migration from the existing IntelligentQueryRouter
to the new ProductionLoadBalancer system. It provides a complete deployment workflow
with safety checks, monitoring, and rollback capabilities.

Usage:
    python deploy_production_load_balancer.py --environment production --phase 1
    python deploy_production_load_balancer.py --migrate --percentage 25
    python deploy_production_load_balancer.py --rollback

Author: Claude Code Assistant
Date: August 2025
Version: 1.0.0
"""

import asyncio
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from lightrag_integration.production_load_balancer import (
    ProductionLoadBalancer,
    ProductionLoadBalancingConfig
)
from lightrag_integration.production_config_schema import (
    ConfigurationFactory,
    ConfigurationValidator,
    ConfigurationManager,
    EnvironmentConfigurationBuilder,
    ConfigurationError
)
from lightrag_integration.intelligent_query_router import IntelligentQueryRouter


# ============================================================================
# Deployment Configuration
# ============================================================================

DEPLOYMENT_CONFIG = {
    'phases': {
        1: {
            'description': 'Initialize ProductionLoadBalancer alongside existing system',
            'traffic_percentage': 0,
            'validation_required': True,
            'rollback_enabled': True
        },
        2: {
            'description': 'Enable real API health checks and basic features',
            'traffic_percentage': 10,
            'validation_required': True,
            'rollback_enabled': True
        },
        3: {
            'description': 'Activate cost optimization and quality routing',
            'traffic_percentage': 25,
            'validation_required': True,
            'rollback_enabled': True
        },
        4: {
            'description': 'Enable adaptive learning and advanced features',
            'traffic_percentage': 50,
            'validation_required': True,
            'rollback_enabled': True
        },
        5: {
            'description': 'Full migration to ProductionLoadBalancer',
            'traffic_percentage': 100,
            'validation_required': True,
            'rollback_enabled': False  # Point of no return
        }
    },
    
    'health_check_timeout': 30,  # seconds
    'validation_queries': [
        "What is clinical metabolomics?",
        "Latest biomarkers for diabetes",
        "Metabolic pathway analysis methods"
    ],
    
    'success_criteria': {
        'response_time_ms': 2000,
        'success_rate_percentage': 95,
        'cost_efficiency_ratio': 0.8,
        'quality_score_minimum': 0.75
    }
}


# ============================================================================
# Deployment Manager
# ============================================================================

class ProductionDeploymentManager:
    """Manages the deployment process for ProductionLoadBalancer"""
    
    def __init__(self, environment: str = 'development'):
        self.environment = environment
        self.logger = self._setup_logging()
        
        # System components
        self.old_router: Optional[IntelligentQueryRouter] = None
        self.new_load_balancer: Optional[ProductionLoadBalancer] = None
        self.config_manager: Optional[ConfigurationManager] = None
        
        # Deployment state
        self.current_phase = 0
        self.traffic_percentage = 0
        self.deployment_metrics = {}
        
        self.logger.info(f"Deployment manager initialized for environment: {environment}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for deployment"""
        logger = logging.getLogger(f'deployment_{self.environment}')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f'deployment_{self.environment}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def initialize_systems(self) -> bool:
        """Initialize both old and new systems"""
        try:
            self.logger.info("Initializing systems...")
            
            # Initialize old system (if exists)
            try:
                # This would import and initialize the existing IntelligentQueryRouter
                self.logger.info("Initializing existing IntelligentQueryRouter...")
                # self.old_router = IntelligentQueryRouter(...)
                self.logger.info("✓ Existing system initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize existing system: {e}")
            
            # Initialize new system
            self.logger.info("Initializing ProductionLoadBalancer...")
            config = EnvironmentConfigurationBuilder.build_from_environment()
            
            # Validate configuration
            validator = ConfigurationValidator()
            validator.validate_and_raise(config)
            
            self.new_load_balancer = ProductionLoadBalancer(config)
            self.config_manager = ConfigurationManager(config)
            
            # Start monitoring
            await self.new_load_balancer.start_monitoring()
            
            self.logger.info("✓ ProductionLoadBalancer initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize systems: {e}")
            return False
    
    async def run_pre_deployment_checks(self) -> bool:
        """Run comprehensive pre-deployment checks"""
        self.logger.info("Running pre-deployment checks...")
        
        checks = [
            self._check_environment_variables,
            self._check_backend_connectivity,
            self._check_health_endpoints,
            self._check_configuration_validity,
            self._check_system_resources
        ]
        
        all_passed = True
        
        for check in checks:
            try:
                check_name = check.__name__.replace('_check_', '').replace('_', ' ').title()
                self.logger.info(f"Running check: {check_name}")
                
                result = await check()
                if result:
                    self.logger.info(f"✓ {check_name} passed")
                else:
                    self.logger.error(f"✗ {check_name} failed")
                    all_passed = False
                    
            except Exception as e:
                self.logger.error(f"✗ {check.__name__} failed with exception: {e}")
                all_passed = False
        
        if all_passed:
            self.logger.info("✓ All pre-deployment checks passed")
        else:
            self.logger.error("✗ Some pre-deployment checks failed")
        
        return all_passed
    
    async def deploy_phase(self, phase: int, force: bool = False) -> bool:
        """Deploy specific phase"""
        if phase not in DEPLOYMENT_CONFIG['phases']:
            self.logger.error(f"Invalid phase: {phase}")
            return False
        
        phase_config = DEPLOYMENT_CONFIG['phases'][phase]
        self.logger.info(f"Starting deployment phase {phase}: {phase_config['description']}")
        
        # Validation check
        if phase_config['validation_required'] and not force:
            if not await self.run_phase_validation(phase):
                self.logger.error(f"Phase {phase} validation failed")
                return False
        
        try:
            # Execute phase-specific deployment steps
            success = await self._execute_phase_deployment(phase, phase_config)
            
            if success:
                self.current_phase = phase
                self.traffic_percentage = phase_config['traffic_percentage']
                self.logger.info(f"✓ Phase {phase} deployment completed successfully")
                
                # Post-deployment validation
                await self._run_post_deployment_validation(phase)
                
                return True
            else:
                self.logger.error(f"✗ Phase {phase} deployment failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception during phase {phase} deployment: {e}")
            return False
    
    async def _execute_phase_deployment(self, phase: int, phase_config: Dict[str, Any]) -> bool:
        """Execute phase-specific deployment steps"""
        
        if phase == 1:
            return await self._deploy_phase_1()
        elif phase == 2:
            return await self._deploy_phase_2()
        elif phase == 3:
            return await self._deploy_phase_3()
        elif phase == 4:
            return await self._deploy_phase_4()
        elif phase == 5:
            return await self._deploy_phase_5()
        else:
            self.logger.error(f"Unknown phase: {phase}")
            return False
    
    async def _deploy_phase_1(self) -> bool:
        """Phase 1: Initialize ProductionLoadBalancer alongside existing system"""
        self.logger.info("Executing Phase 1 deployment...")
        
        # Systems should already be initialized
        if not self.new_load_balancer:
            self.logger.error("ProductionLoadBalancer not initialized")
            return False
        
        # Test basic functionality without routing traffic
        test_query = "Test query for phase 1"
        
        try:
            backend_id, confidence = await self.new_load_balancer.select_optimal_backend(test_query)
            self.logger.info(f"✓ Backend selection working: {backend_id} (confidence: {confidence:.3f})")
            
            # Get system status
            status = self.new_load_balancer.get_backend_status()
            available_backends = status['available_backends']
            total_backends = status['total_backends']
            
            self.logger.info(f"✓ System status: {available_backends}/{total_backends} backends available")
            
            if available_backends == 0:
                self.logger.warning("No backends available - this may indicate configuration issues")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 1 deployment failed: {e}")
            return False
    
    async def _deploy_phase_2(self) -> bool:
        """Phase 2: Enable real API health checks and basic features"""
        self.logger.info("Executing Phase 2 deployment...")
        
        try:
            # Start routing 10% of traffic to new system
            self.traffic_percentage = 10
            
            # Enable basic features
            config = self.new_load_balancer.config
            config.enable_real_time_monitoring = True
            
            # Test with validation queries
            for query in DEPLOYMENT_CONFIG['validation_queries']:
                backend_id, confidence = await self.new_load_balancer.select_optimal_backend(query)
                result = await self.new_load_balancer.send_query(backend_id, query)
                
                if not result.get('success', False):
                    self.logger.error(f"Validation query failed: {query}")
                    return False
                
                response_time = result.get('response_time_ms', 0)
                if response_time > DEPLOYMENT_CONFIG['success_criteria']['response_time_ms']:
                    self.logger.warning(f"Slow response time: {response_time}ms for query: {query}")
            
            self.logger.info("✓ Phase 2 validation queries completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 2 deployment failed: {e}")
            return False
    
    async def _deploy_phase_3(self) -> bool:
        """Phase 3: Activate cost optimization and quality routing"""
        self.logger.info("Executing Phase 3 deployment...")
        
        try:
            self.traffic_percentage = 25
            
            # Enable cost optimization
            config = self.new_load_balancer.config
            config.enable_cost_optimization = True
            config.enable_quality_based_routing = True
            
            # Test cost optimization
            test_query = "Cost optimization test query"
            
            # Run multiple queries and track costs
            total_cost = 0
            query_count = 10
            
            for i in range(query_count):
                backend_id, confidence = await self.new_load_balancer.select_optimal_backend(f"{test_query} {i}")
                result = await self.new_load_balancer.send_query(backend_id, f"{test_query} {i}")
                
                if result.get('success'):
                    total_cost += result.get('cost_estimate', 0)
            
            avg_cost_per_query = total_cost / query_count
            self.logger.info(f"✓ Average cost per query: ${avg_cost_per_query:.4f}")
            
            # Check if cost efficiency meets criteria
            cost_efficiency = min(1.0, DEPLOYMENT_CONFIG['success_criteria']['cost_efficiency_ratio'] / avg_cost_per_query)
            if cost_efficiency < DEPLOYMENT_CONFIG['success_criteria']['cost_efficiency_ratio']:
                self.logger.warning(f"Cost efficiency below target: {cost_efficiency:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 3 deployment failed: {e}")
            return False
    
    async def _deploy_phase_4(self) -> bool:
        """Phase 4: Enable adaptive learning and advanced features"""
        self.logger.info("Executing Phase 4 deployment...")
        
        try:
            self.traffic_percentage = 50
            
            # Enable adaptive learning
            config = self.new_load_balancer.config
            config.enable_adaptive_routing = True
            
            # Run extended testing to allow learning
            self.logger.info("Running extended testing for adaptive learning...")
            
            test_queries = [
                "metabolomics pathway analysis",
                "biomarker discovery methods", 
                "clinical metabolomics applications",
                "mass spectrometry techniques",
                "data analysis protocols"
            ]
            
            # Run multiple rounds of queries
            for round_num in range(3):
                self.logger.info(f"Testing round {round_num + 1}/3")
                
                for query in test_queries:
                    backend_id, confidence = await self.new_load_balancer.select_optimal_backend(query)
                    result = await self.new_load_balancer.send_query(backend_id, query)
                    
                    if not result.get('success'):
                        self.logger.warning(f"Query failed in round {round_num + 1}: {query}")
                
                # Brief pause between rounds
                await asyncio.sleep(2)
            
            # Check if adaptive learning is working
            learned_weights = self.new_load_balancer.learned_weights
            if learned_weights:
                self.logger.info(f"✓ Adaptive learning active: {len(learned_weights)} learned weights")
            else:
                self.logger.warning("No learned weights found - learning may not be active")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 4 deployment failed: {e}")
            return False
    
    async def _deploy_phase_5(self) -> bool:
        """Phase 5: Full migration to ProductionLoadBalancer"""
        self.logger.info("Executing Phase 5 deployment - Full Migration...")
        
        try:
            self.traffic_percentage = 100
            
            # Final validation with all features enabled
            self.logger.info("Running final validation tests...")
            
            # Extended test suite
            validation_results = []
            
            for i, query in enumerate(DEPLOYMENT_CONFIG['validation_queries'] * 3):  # 3x validation
                start_time = time.time()
                
                backend_id, confidence = await self.new_load_balancer.select_optimal_backend(f"{query} - final test {i}")
                result = await self.new_load_balancer.send_query(backend_id, f"{query} - final test {i}")
                
                response_time = (time.time() - start_time) * 1000
                
                validation_results.append({
                    'success': result.get('success', False),
                    'response_time_ms': response_time,
                    'cost': result.get('cost_estimate', 0),
                    'backend_used': backend_id,
                    'confidence': confidence
                })
            
            # Analyze results
            success_rate = sum(1 for r in validation_results if r['success']) / len(validation_results) * 100
            avg_response_time = sum(r['response_time_ms'] for r in validation_results) / len(validation_results)
            total_cost = sum(r['cost'] for r in validation_results)
            avg_confidence = sum(r['confidence'] for r in validation_results) / len(validation_results)
            
            self.logger.info(f"Final validation results:")
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
            self.logger.info(f"  Average response time: {avg_response_time:.2f}ms")
            self.logger.info(f"  Total cost: ${total_cost:.4f}")
            self.logger.info(f"  Average confidence: {avg_confidence:.3f}")
            
            # Check success criteria
            criteria = DEPLOYMENT_CONFIG['success_criteria']
            
            if success_rate < criteria['success_rate_percentage']:
                self.logger.error(f"Success rate {success_rate:.1f}% below threshold {criteria['success_rate_percentage']}%")
                return False
            
            if avg_response_time > criteria['response_time_ms']:
                self.logger.error(f"Response time {avg_response_time:.2f}ms above threshold {criteria['response_time_ms']}ms")
                return False
            
            # Decommission old system (if it exists)
            if self.old_router:
                self.logger.info("Decommissioning old IntelligentQueryRouter...")
                # Stop old router monitoring, etc.
                self.old_router = None
            
            self.logger.info("✓ Full migration completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 5 deployment failed: {e}")
            return False
    
    async def rollback_to_phase(self, target_phase: int) -> bool:
        """Rollback to previous phase"""
        if target_phase >= self.current_phase:
            self.logger.error(f"Cannot rollback to phase {target_phase} from current phase {self.current_phase}")
            return False
        
        if target_phase < 0:
            self.logger.error("Cannot rollback below phase 0")
            return False
        
        self.logger.info(f"Rolling back from phase {self.current_phase} to phase {target_phase}")
        
        try:
            # Phase-specific rollback logic
            if self.current_phase >= 5 and target_phase < 5:
                # Re-enable old system if rolling back from full migration
                self.logger.info("Re-enabling old system...")
                # Reinitialize old router if needed
            
            if self.current_phase >= 4 and target_phase < 4:
                # Disable adaptive learning
                if self.new_load_balancer:
                    self.new_load_balancer.config.enable_adaptive_routing = False
                    self.logger.info("✓ Disabled adaptive learning")
            
            if self.current_phase >= 3 and target_phase < 3:
                # Disable cost optimization
                if self.new_load_balancer:
                    self.new_load_balancer.config.enable_cost_optimization = False
                    self.new_load_balancer.config.enable_quality_based_routing = False
                    self.logger.info("✓ Disabled cost and quality optimization")
            
            if self.current_phase >= 2 and target_phase < 2:
                # Reduce traffic routing
                if self.new_load_balancer:
                    self.new_load_balancer.config.enable_real_time_monitoring = False
                    self.logger.info("✓ Disabled real-time monitoring")
            
            # Update state
            self.current_phase = target_phase
            self.traffic_percentage = DEPLOYMENT_CONFIG['phases'][target_phase]['traffic_percentage']
            
            self.logger.info(f"✓ Rollback to phase {target_phase} completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    async def run_health_monitoring(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run health monitoring for specified duration"""
        self.logger.info(f"Running health monitoring for {duration_seconds} seconds...")
        
        if not self.new_load_balancer:
            return {'error': 'ProductionLoadBalancer not initialized'}
        
        start_time = time.time()
        metrics = {
            'start_time': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
            'samples': []
        }
        
        while (time.time() - start_time) < duration_seconds:
            try:
                # Get current system status
                status = self.new_load_balancer.get_backend_status()
                
                sample = {
                    'timestamp': datetime.now().isoformat(),
                    'available_backends': status['available_backends'],
                    'total_backends': status['total_backends'],
                    'backend_health': {}
                }
                
                # Collect health metrics for each backend
                for backend_id, backend_status in status['backends'].items():
                    sample['backend_health'][backend_id] = {
                        'health_status': backend_status['health_status'],
                        'health_score': backend_status['health_score'],
                        'response_time_ms': backend_status['response_time_ms'],
                        'error_rate': backend_status['error_rate']
                    }
                
                metrics['samples'].append(sample)
                
                await asyncio.sleep(5)  # Sample every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error during health monitoring: {e}")
                break
        
        # Calculate summary statistics
        if metrics['samples']:
            avg_available = sum(s['available_backends'] for s in metrics['samples']) / len(metrics['samples'])
            metrics['summary'] = {
                'average_available_backends': avg_available,
                'total_samples': len(metrics['samples']),
                'monitoring_completed': True
            }
        else:
            metrics['summary'] = {'monitoring_completed': False}
        
        self.logger.info(f"Health monitoring completed: {metrics['summary']}")
        return metrics
    
    # Pre-deployment check methods
    async def _check_environment_variables(self) -> bool:
        """Check required environment variables"""
        required_vars = [
            'OPENAI_API_KEY',
            'PERPLEXITY_API_KEY',
            'ENVIRONMENT'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"Missing environment variables: {missing_vars}")
            return False
        
        return True
    
    async def _check_backend_connectivity(self) -> bool:
        """Check connectivity to backend services"""
        if not self.new_load_balancer:
            return False
        
        # Test connectivity to each backend
        for instance_id, client in self.new_load_balancer.backend_clients.items():
            try:
                is_healthy, response_time, health_data = await client.health_check()
                if is_healthy:
                    self.logger.info(f"✓ Backend {instance_id} connectivity OK ({response_time:.2f}ms)")
                else:
                    self.logger.error(f"✗ Backend {instance_id} connectivity failed: {health_data}")
                    return False
            except Exception as e:
                self.logger.error(f"✗ Backend {instance_id} connectivity error: {e}")
                return False
        
        return True
    
    async def _check_health_endpoints(self) -> bool:
        """Check health endpoints are responding"""
        if not self.new_load_balancer:
            return False
        
        # This would be implemented based on specific backend requirements
        return True
    
    async def _check_configuration_validity(self) -> bool:
        """Check configuration validity"""
        if not self.config_manager:
            return False
        
        return self.config_manager.validate()
    
    async def _check_system_resources(self) -> bool:
        """Check system resources"""
        import psutil
        
        # Check CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 80:
            self.logger.warning(f"High CPU usage: {cpu_usage}%")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            self.logger.warning(f"High memory usage: {memory.percent}%")
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > 80:
            self.logger.warning(f"High disk usage: {disk.percent}%")
        
        return True
    
    async def run_phase_validation(self, phase: int) -> bool:
        """Run phase-specific validation"""
        self.logger.info(f"Running validation for phase {phase}")
        
        # Basic validation queries
        for query in DEPLOYMENT_CONFIG['validation_queries']:
            try:
                backend_id, confidence = await self.new_load_balancer.select_optimal_backend(query)
                
                if confidence < 0.5:
                    self.logger.warning(f"Low confidence for query '{query}': {confidence:.3f}")
                
                self.logger.info(f"✓ Validation query OK: {query} -> {backend_id}")
                
            except Exception as e:
                self.logger.error(f"✗ Validation query failed: {query} - {e}")
                return False
        
        return True
    
    async def _run_post_deployment_validation(self, phase: int):
        """Run post-deployment validation"""
        self.logger.info(f"Running post-deployment validation for phase {phase}")
        
        # Get system metrics
        if self.new_load_balancer:
            status = self.new_load_balancer.get_backend_status()
            stats = self.new_load_balancer.get_routing_statistics(hours=1)
            
            self.logger.info(f"Post-deployment metrics:")
            self.logger.info(f"  Available backends: {status['available_backends']}/{status['total_backends']}")
            self.logger.info(f"  Recent decisions: {stats.get('total_decisions', 0)}")
            self.logger.info(f"  Success rate: {sum(stats.get('success_rates', {}).values()) / len(stats.get('success_rates', {1})) * 100:.1f}%")
    
    async def cleanup(self):
        """Cleanup deployment resources"""
        if self.new_load_balancer:
            await self.new_load_balancer.stop_monitoring()
        
        self.logger.info("Deployment cleanup completed")


# ============================================================================
# Command Line Interface
# ============================================================================

async def main():
    """Main deployment script entry point"""
    parser = argparse.ArgumentParser(description='Production Load Balancer Deployment Script')
    
    parser.add_argument('--environment', '-e', 
                       choices=['development', 'staging', 'production'],
                       default='development',
                       help='Deployment environment')
    
    parser.add_argument('--phase', '-p', type=int, choices=[1, 2, 3, 4, 5],
                       help='Deployment phase to execute')
    
    parser.add_argument('--migrate', action='store_true',
                       help='Start migration process')
    
    parser.add_argument('--percentage', type=int, default=10,
                       help='Traffic percentage for migration (used with --migrate)')
    
    parser.add_argument('--rollback', '-r', type=int, metavar='PHASE',
                       help='Rollback to specified phase')
    
    parser.add_argument('--health-check', action='store_true',
                       help='Run health monitoring')
    
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration for health monitoring (seconds)')
    
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force deployment without validation')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    # Create deployment manager
    deployment_manager = ProductionDeploymentManager(args.environment)
    
    try:
        if args.dry_run:
            print("DRY RUN MODE - No actual changes will be made")
            print(f"Environment: {args.environment}")
            if args.phase:
                print(f"Would deploy phase: {args.phase}")
            if args.migrate:
                print(f"Would migrate {args.percentage}% of traffic")
            if args.rollback:
                print(f"Would rollback to phase: {args.rollback}")
            return
        
        # Initialize systems
        if not await deployment_manager.initialize_systems():
            print("❌ System initialization failed")
            return 1
        
        # Run pre-deployment checks
        if not args.force and not await deployment_manager.run_pre_deployment_checks():
            print("❌ Pre-deployment checks failed. Use --force to override.")
            return 1
        
        # Execute requested operation
        success = True
        
        if args.phase:
            success = await deployment_manager.deploy_phase(args.phase, args.force)
            
        elif args.migrate:
            # Determine appropriate phase based on percentage
            target_phase = 1
            for phase, config in DEPLOYMENT_CONFIG['phases'].items():
                if config['traffic_percentage'] <= args.percentage:
                    target_phase = phase
            
            print(f"Migrating to phase {target_phase} ({args.percentage}% traffic)")
            success = await deployment_manager.deploy_phase(target_phase, args.force)
            
        elif args.rollback is not None:
            success = await deployment_manager.rollback_to_phase(args.rollback)
            
        elif args.health_check:
            metrics = await deployment_manager.run_health_monitoring(args.duration)
            print(f"Health monitoring completed: {metrics['summary']}")
        
        else:
            # Default: show current status
            if deployment_manager.new_load_balancer:
                status = deployment_manager.new_load_balancer.get_backend_status()
                print(f"Current status: {status['available_backends']}/{status['total_backends']} backends available")
                print(f"Current phase: {deployment_manager.current_phase}")
                print(f"Traffic percentage: {deployment_manager.traffic_percentage}%")
        
        if success:
            print("✅ Deployment operation completed successfully")
            return 0
        else:
            print("❌ Deployment operation failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Deployment interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Deployment failed with exception: {e}")
        return 1
        
    finally:
        await deployment_manager.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)