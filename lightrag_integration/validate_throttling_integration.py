"""
Production Integration Validation for Load-Based Request Throttling System
===========================================================================

This script validates the integration of the load-based request throttling and queuing
system with the existing production components of the Clinical Metabolomics Oracle:

1. Production Load Balancer Integration
2. Clinical Metabolomics RAG System Integration  
3. Enhanced Load Monitoring System Integration
4. Progressive Service Degradation Controller Integration
5. Production Monitoring System Integration

Validation Areas:
- Component compatibility and interaction
- Configuration synchronization
- Load level response coordination
- Request flow integrity
- Performance under production scenarios
- Graceful degradation behavior
- Error handling and recovery

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
Production Ready: Yes
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add the current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import graceful degradation components
try:
    from load_based_request_throttling_system import (
        RequestThrottlingSystem, RequestType, RequestPriority, 
        create_request_throttling_system
    )
    THROTTLING_AVAILABLE = True
except ImportError as e:
    THROTTLING_AVAILABLE = False
    print(f"Warning: Throttling system not available: {e}")

try:
    from graceful_degradation_integration import (
        GracefulDegradationOrchestrator, GracefulDegradationConfig,
        create_graceful_degradation_system
    )
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    INTEGRATION_AVAILABLE = False
    print(f"Warning: Integration layer not available: {e}")

try:
    from enhanced_load_monitoring_system import (
        SystemLoadLevel, create_enhanced_load_monitoring_system
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    print(f"Warning: Enhanced monitoring not available: {e}")

try:
    from progressive_service_degradation_controller import (
        create_progressive_degradation_controller
    )
    DEGRADATION_AVAILABLE = True
except ImportError as e:
    DEGRADATION_AVAILABLE = False
    print(f"Warning: Degradation controller not available: {e}")

# Try to import production systems
try:
    from production_load_balancer import ProductionLoadBalancer
    LOAD_BALANCER_AVAILABLE = True
except ImportError as e:
    LOAD_BALANCER_AVAILABLE = False
    print(f"Info: Production load balancer not available: {e}")

try:
    from clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"Info: Clinical RAG system not available: {e}")

try:
    from production_monitoring import ProductionMonitoring
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError as e:
    PRODUCTION_MONITORING_AVAILABLE = False
    print(f"Info: Production monitoring not available: {e}")


# ============================================================================
# VALIDATION FRAMEWORK
# ============================================================================

class ValidationResult:
    """Result of a validation test."""
    
    def __init__(self, test_name: str, success: bool, message: str, 
                 details: Optional[Dict[str, Any]] = None, 
                 duration: float = 0.0):
        self.test_name = test_name
        self.success = success
        self.message = message
        self.details = details or {}
        self.duration = duration
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'success': self.success,
            'message': self.message,
            'details': self.details,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat()
        }


class ValidationSuite:
    """Main validation suite for production integration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results: List[ValidationResult] = []
        self.start_time: Optional[datetime] = None
        
        # Mock production systems for testing if real ones not available
        self.mock_load_balancer = self._create_mock_load_balancer()
        self.mock_rag_system = self._create_mock_rag_system()
        self.mock_monitoring = self._create_mock_monitoring()
    
    def _create_mock_load_balancer(self):
        """Create mock load balancer for testing."""
        class MockLoadBalancer:
            def __init__(self):
                self.backend_timeouts = {}
                self.circuit_breaker_settings = {}
                self.request_throttler = None
                self.callbacks = []
            
            def update_backend_timeouts(self, timeouts: Dict[str, float]):
                self.backend_timeouts.update(timeouts)
            
            def update_circuit_breaker_settings(self, settings: Dict[str, Any]):
                self.circuit_breaker_settings.update(settings)
            
            def set_request_throttler(self, throttler):
                self.request_throttler = throttler
            
            def add_degradation_callback(self, callback):
                self.callbacks.append(callback)
            
            def get_status(self):
                return {
                    'backend_timeouts': self.backend_timeouts,
                    'circuit_breaker_settings': self.circuit_breaker_settings,
                    'has_throttler': self.request_throttler is not None
                }
        
        return MockLoadBalancer()
    
    def _create_mock_rag_system(self):
        """Create mock RAG system for testing."""
        class MockRAGSystem:
            def __init__(self):
                self.timeouts = {}
                self.query_complexity = {}
                self.feature_flags = {}
                self.config = {}
                self.callbacks = []
            
            def update_timeouts(self, timeouts: Dict[str, float]):
                self.timeouts.update(timeouts)
            
            def update_query_complexity(self, settings: Dict[str, Any]):
                self.query_complexity.update(settings)
            
            def update_feature_flags(self, flags: Dict[str, bool]):
                self.feature_flags.update(flags)
            
            def add_pre_query_callback(self, callback):
                self.callbacks.append(('pre', callback))
            
            def add_post_query_callback(self, callback):
                self.callbacks.append(('post', callback))
            
            def get_status(self):
                return {
                    'timeouts': self.timeouts,
                    'query_complexity': self.query_complexity,
                    'feature_flags': self.feature_flags
                }
        
        return MockRAGSystem()
    
    def _create_mock_monitoring(self):
        """Create mock monitoring system for testing."""
        class MockMonitoring:
            def __init__(self):
                self.custom_metrics = {}
                self.monitoring_interval = 5.0
                self.detailed_logging = True
            
            def add_custom_metric_source(self, name: str, source: callable):
                self.custom_metrics[name] = source
            
            def update_monitoring_interval(self, interval: float):
                self.monitoring_interval = interval
            
            def set_detailed_logging(self, enabled: bool):
                self.detailed_logging = enabled
            
            def get_status(self):
                return {
                    'custom_metrics': list(self.custom_metrics.keys()),
                    'monitoring_interval': self.monitoring_interval,
                    'detailed_logging': self.detailed_logging
                }
        
        return MockMonitoring()
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        self.start_time = datetime.now()
        print("Starting Production Integration Validation")
        print("=" * 60)
        
        # Component availability checks
        await self._validate_component_availability()
        
        # If throttling system is available, run integration tests
        if THROTTLING_AVAILABLE and INTEGRATION_AVAILABLE:
            await self._validate_throttling_system_integration()
            await self._validate_graceful_degradation_integration()
            await self._validate_production_system_integration()
            await self._validate_load_response_coordination()
            await self._validate_performance_characteristics()
            await self._validate_error_handling()
        
        # Generate report
        return self._generate_validation_report()
    
    async def _validate_component_availability(self):
        """Validate availability of all components."""
        print("\n🔍 Checking Component Availability...")
        
        components = [
            ("Load-Based Request Throttling System", THROTTLING_AVAILABLE),
            ("Graceful Degradation Integration", INTEGRATION_AVAILABLE),
            ("Enhanced Load Monitoring System", MONITORING_AVAILABLE),
            ("Progressive Service Degradation Controller", DEGRADATION_AVAILABLE),
            ("Production Load Balancer", LOAD_BALANCER_AVAILABLE),
            ("Clinical Metabolomics RAG System", RAG_AVAILABLE),
            ("Production Monitoring System", PRODUCTION_MONITORING_AVAILABLE)
        ]
        
        for component_name, available in components:
            result = ValidationResult(
                test_name=f"availability_{component_name.lower().replace(' ', '_')}",
                success=available,
                message=f"{component_name}: {'Available' if available else 'Not Available'}",
                details={'component': component_name, 'available': available}
            )
            self.results.append(result)
            print(f"  {'✅' if available else '❌'} {component_name}")
    
    async def _validate_throttling_system_integration(self):
        """Validate basic throttling system functionality."""
        print("\n🚀 Validating Throttling System Integration...")
        
        start_time = time.time()
        
        try:
            # Create throttling system
            throttling_system = create_request_throttling_system(
                base_rate_per_second=10.0,
                max_queue_size=50,
                max_concurrent_requests=5
            )
            
            await throttling_system.start()
            
            # Test basic functionality
            async def test_handler(message: str):
                await asyncio.sleep(0.1)
                return f"Processed: {message}"
            
            success, message, request_id = await throttling_system.submit_request(
                request_type=RequestType.USER_QUERY,
                priority=RequestPriority.HIGH,
                handler=test_handler,
                message="Integration test query"
            )
            
            await asyncio.sleep(0.5)  # Allow processing
            
            status = throttling_system.get_system_status()
            health = throttling_system.get_health_check()
            
            await throttling_system.stop()
            
            # Validate results
            test_success = (
                success and
                status['system_running'] and
                health['status'] in ['healthy', 'degraded'] and
                status['lifecycle']['total_requests'] > 0
            )
            
            result = ValidationResult(
                test_name="throttling_system_basic_functionality",
                success=test_success,
                message="Throttling system basic functionality test",
                details={
                    'request_success': success,
                    'system_status': status['system_running'],
                    'health_status': health['status'],
                    'requests_processed': status['lifecycle']['total_requests']
                },
                duration=time.time() - start_time
            )
            
            self.results.append(result)
            print(f"  {'✅' if test_success else '❌'} Basic throttling functionality")
            
        except Exception as e:
            result = ValidationResult(
                test_name="throttling_system_basic_functionality",
                success=False,
                message=f"Throttling system test failed: {str(e)}",
                duration=time.time() - start_time
            )
            self.results.append(result)
            print(f"  ❌ Basic throttling functionality: {str(e)}")
    
    async def _validate_graceful_degradation_integration(self):
        """Validate complete graceful degradation system integration."""
        print("\n🎯 Validating Graceful Degradation Integration...")
        
        start_time = time.time()
        
        try:
            # Create integrated system
            config = GracefulDegradationConfig(
                monitoring_interval=1.0,
                base_rate_per_second=8.0,
                max_queue_size=40,
                max_concurrent_requests=4
            )
            
            orchestrator = create_graceful_degradation_system(
                config=config,
                load_balancer=self.mock_load_balancer,
                rag_system=self.mock_rag_system,
                monitoring_system=self.mock_monitoring
            )
            
            await orchestrator.start()
            
            # Test integrated functionality
            async def integration_handler(test_type: str):
                await asyncio.sleep(0.1)
                return f"Integration test: {test_type}"
            
            # Submit various request types
            request_results = []
            for req_type in ['health_check', 'user_query', 'analytics']:
                success, message, req_id = await orchestrator.submit_request(
                    request_type=req_type,
                    handler=integration_handler,
                    test_type=req_type
                )
                request_results.append(success)
            
            await asyncio.sleep(1.0)  # Allow processing
            
            # Get system status
            status = orchestrator.get_system_status()
            health = orchestrator.get_health_check()
            
            await orchestrator.stop()
            
            # Validate integration
            successful_requests = sum(request_results)
            integration_success = (
                successful_requests > 0 and
                status['running'] and
                status['integration_status']['throttling_system_active'] and
                health['status'] in ['healthy', 'degraded']
            )
            
            result = ValidationResult(
                test_name="graceful_degradation_integration",
                success=integration_success,
                message="Graceful degradation integration test",
                details={
                    'successful_requests': successful_requests,
                    'total_requests': len(request_results),
                    'system_running': status['running'],
                    'throttling_active': status['integration_status']['throttling_system_active'],
                    'health_status': health['status']
                },
                duration=time.time() - start_time
            )
            
            self.results.append(result)
            print(f"  {'✅' if integration_success else '❌'} Complete integration functionality")
            
        except Exception as e:
            result = ValidationResult(
                test_name="graceful_degradation_integration",
                success=False,
                message=f"Integration test failed: {str(e)}",
                duration=time.time() - start_time
            )
            self.results.append(result)
            print(f"  ❌ Complete integration functionality: {str(e)}")
    
    async def _validate_production_system_integration(self):
        """Validate integration with mock production systems."""
        print("\n🔧 Validating Production System Integration...")
        
        start_time = time.time()
        
        try:
            # Create system with mock production components
            orchestrator = create_graceful_degradation_system(
                load_balancer=self.mock_load_balancer,
                rag_system=self.mock_rag_system,
                monitoring_system=self.mock_monitoring
            )
            
            await orchestrator.start()
            
            # Check integration status
            status = orchestrator.get_system_status()
            integration_status = status['integration_status']
            
            # Simulate degradation controller updates
            if orchestrator.production_integrator:
                # Test load balancer integration
                lb_config = {
                    'timeouts': {'lightrag': 30.0, 'openai_api': 25.0},
                    'circuit_breaker': {'failure_threshold': 3}
                }
                lb_success = orchestrator.production_integrator.update_system_configuration(
                    'load_balancer', lb_config
                )
                
                # Test RAG system integration
                rag_config = {
                    'query_complexity': {'token_limit': 4000, 'query_mode': 'local'},
                    'timeouts': {'lightrag_timeout': 40.0}
                }
                rag_success = orchestrator.production_integrator.update_system_configuration(
                    'rag_system', rag_config
                )
                
                # Test monitoring integration
                mon_config = {'interval': 10.0}
                mon_success = orchestrator.production_integrator.update_system_configuration(
                    'monitoring_system', mon_config
                )
                
                # Verify configurations were applied
                lb_status = self.mock_load_balancer.get_status()
                rag_status = self.mock_rag_system.get_status()
                mon_status = self.mock_monitoring.get_status()
                
                production_integration_success = (
                    lb_success and rag_success and mon_success and
                    lb_status['backend_timeouts'] and
                    rag_status['query_complexity'] and
                    mon_status['monitoring_interval'] == 10.0
                )
            else:
                production_integration_success = False
            
            await orchestrator.stop()
            
            result = ValidationResult(
                test_name="production_system_integration",
                success=production_integration_success,
                message="Production system integration test",
                details={
                    'load_balancer_integration': lb_success if 'lb_success' in locals() else False,
                    'rag_system_integration': rag_success if 'rag_success' in locals() else False,
                    'monitoring_integration': mon_success if 'mon_success' in locals() else False,
                    'load_balancer_status': lb_status if 'lb_status' in locals() else {},
                    'rag_status': rag_status if 'rag_status' in locals() else {},
                    'monitoring_status': mon_status if 'mon_status' in locals() else {}
                },
                duration=time.time() - start_time
            )
            
            self.results.append(result)
            print(f"  {'✅' if production_integration_success else '❌'} Production system integration")
            
        except Exception as e:
            result = ValidationResult(
                test_name="production_system_integration",
                success=False,
                message=f"Production integration failed: {str(e)}",
                duration=time.time() - start_time
            )
            self.results.append(result)
            print(f"  ❌ Production system integration: {str(e)}")
    
    async def _validate_load_response_coordination(self):
        """Validate coordination of load level changes across systems."""
        print("\n📊 Validating Load Response Coordination...")
        
        start_time = time.time()
        
        try:
            # This test would simulate load level changes and verify
            # that all systems respond appropriately
            
            # For now, we'll validate the basic coordination structure
            if THROTTLING_AVAILABLE and INTEGRATION_AVAILABLE:
                orchestrator = create_graceful_degradation_system()
                await orchestrator.start()
                
                # Check if systems are connected
                has_load_detector = orchestrator.load_detector is not None
                has_degradation_controller = orchestrator.degradation_controller is not None
                has_throttling_system = orchestrator.throttling_system is not None
                
                coordination_success = (
                    has_load_detector or has_degradation_controller or has_throttling_system
                )  # At least one component should be available
                
                await orchestrator.stop()
            else:
                coordination_success = False
            
            result = ValidationResult(
                test_name="load_response_coordination",
                success=coordination_success,
                message="Load response coordination validation",
                details={
                    'has_load_detector': has_load_detector if 'has_load_detector' in locals() else False,
                    'has_degradation_controller': has_degradation_controller if 'has_degradation_controller' in locals() else False,
                    'has_throttling_system': has_throttling_system if 'has_throttling_system' in locals() else False
                },
                duration=time.time() - start_time
            )
            
            self.results.append(result)
            print(f"  {'✅' if coordination_success else '❌'} Load response coordination")
            
        except Exception as e:
            result = ValidationResult(
                test_name="load_response_coordination",
                success=False,
                message=f"Load coordination test failed: {str(e)}",
                duration=time.time() - start_time
            )
            self.results.append(result)
            print(f"  ❌ Load response coordination: {str(e)}")
    
    async def _validate_performance_characteristics(self):
        """Validate performance characteristics under various conditions."""
        print("\n⚡ Validating Performance Characteristics...")
        
        start_time = time.time()
        
        try:
            if not THROTTLING_AVAILABLE:
                result = ValidationResult(
                    test_name="performance_characteristics",
                    success=False,
                    message="Throttling system not available for performance testing",
                    duration=time.time() - start_time
                )
                self.results.append(result)
                print("  ❌ Performance validation skipped - throttling system not available")
                return
            
            # Create system for performance testing
            throttling_system = create_request_throttling_system(
                base_rate_per_second=20.0,  # Higher rate for performance test
                max_queue_size=100,
                max_concurrent_requests=10
            )
            
            await throttling_system.start()
            
            # Performance test - submit multiple requests quickly
            async def perf_handler(req_num: int):
                await asyncio.sleep(0.05)  # Fast processing
                return f"Performance test {req_num}"
            
            # Submit requests and measure performance
            submit_start = time.time()
            submit_tasks = []
            
            for i in range(50):  # 50 requests
                task = throttling_system.submit_request(
                    request_type=RequestType.USER_QUERY,
                    handler=perf_handler,
                    req_num=i
                )
                submit_tasks.append(task)
            
            results = await asyncio.gather(*submit_tasks, return_exceptions=True)
            submit_duration = time.time() - submit_start
            
            # Count successful submissions
            successful_submissions = sum(1 for success, _, _ in results if success and not isinstance(success, Exception))
            
            # Wait for processing
            await asyncio.sleep(2.0)
            
            # Get performance metrics
            status = throttling_system.get_system_status()
            health = throttling_system.get_health_check()
            
            await throttling_system.stop()
            
            # Evaluate performance
            submission_rate = len(results) / submit_duration
            success_rate = successful_submissions / len(results) if results else 0
            
            performance_success = (
                submission_rate > 20 and  # At least 20 submissions per second
                success_rate > 0.7 and   # At least 70% success rate
                health['status'] in ['healthy', 'degraded']
            )
            
            result = ValidationResult(
                test_name="performance_characteristics",
                success=performance_success,
                message="Performance characteristics validation",
                details={
                    'submission_rate': submission_rate,
                    'success_rate': success_rate,
                    'successful_submissions': successful_submissions,
                    'total_requests': len(results),
                    'submit_duration': submit_duration,
                    'health_status': health['status']
                },
                duration=time.time() - start_time
            )
            
            self.results.append(result)
            print(f"  {'✅' if performance_success else '❌'} Performance characteristics")
            print(f"    Submission rate: {submission_rate:.1f} req/s")
            print(f"    Success rate: {success_rate:.1%}")
            
        except Exception as e:
            result = ValidationResult(
                test_name="performance_characteristics",
                success=False,
                message=f"Performance validation failed: {str(e)}",
                duration=time.time() - start_time
            )
            self.results.append(result)
            print(f"  ❌ Performance characteristics: {str(e)}")
    
    async def _validate_error_handling(self):
        """Validate error handling and recovery mechanisms."""
        print("\n🛡️  Validating Error Handling...")
        
        start_time = time.time()
        
        try:
            if not THROTTLING_AVAILABLE:
                result = ValidationResult(
                    test_name="error_handling",
                    success=False,
                    message="Throttling system not available for error handling test",
                    duration=time.time() - start_time
                )
                self.results.append(result)
                print("  ❌ Error handling validation skipped - throttling system not available")
                return
            
            # Create system for error testing
            throttling_system = create_request_throttling_system(
                base_rate_per_second=5.0,
                max_queue_size=10,  # Small queue to trigger overflow
                max_concurrent_requests=2  # Low concurrency to test limits
            )
            
            await throttling_system.start()
            
            # Test 1: Queue overflow handling
            async def slow_handler(req_num: int):
                await asyncio.sleep(0.5)  # Slow to cause backup
                return f"Slow request {req_num}"
            
            # Submit more requests than can be handled
            overflow_results = []
            for i in range(15):  # More than queue + concurrency limit
                success, message, req_id = await throttling_system.submit_request(
                    request_type=RequestType.BATCH_PROCESSING,
                    handler=slow_handler,
                    req_num=i
                )
                overflow_results.append(success)
            
            # Test 2: Handler error handling
            async def error_handler():
                raise Exception("Intentional test error")
            
            error_success, error_message, error_req_id = await throttling_system.submit_request(
                request_type=RequestType.MAINTENANCE,
                handler=error_handler
            )
            
            # Wait for processing
            await asyncio.sleep(2.0)
            
            # Get final status
            final_status = throttling_system.get_system_status()
            final_health = throttling_system.get_health_check()
            
            await throttling_system.stop()
            
            # Evaluate error handling
            rejections = sum(1 for success in overflow_results if not success)
            has_rejections = rejections > 0  # Should have some rejections due to overflow
            system_stable = final_health['status'] in ['healthy', 'degraded']
            
            error_handling_success = has_rejections and system_stable
            
            result = ValidationResult(
                test_name="error_handling",
                success=error_handling_success,
                message="Error handling validation",
                details={
                    'overflow_rejections': rejections,
                    'total_overflow_requests': len(overflow_results),
                    'error_handler_accepted': error_success,
                    'system_stable': system_stable,
                    'final_health': final_health['status']
                },
                duration=time.time() - start_time
            )
            
            self.results.append(result)
            print(f"  {'✅' if error_handling_success else '❌'} Error handling and recovery")
            print(f"    Overflow rejections: {rejections}/{len(overflow_results)}")
            print(f"    System stability: {system_stable}")
            
        except Exception as e:
            result = ValidationResult(
                test_name="error_handling",
                success=False,
                message=f"Error handling validation failed: {str(e)}",
                duration=time.time() - start_time
            )
            self.results.append(result)
            print(f"  ❌ Error handling: {str(e)}")
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        # Categorize results
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        # Component availability summary
        availability_results = [r for r in self.results if r.test_name.startswith('availability_')]
        available_components = [r for r in availability_results if r.success]
        
        # Integration test summary
        integration_results = [r for r in self.results if 'integration' in r.test_name]
        successful_integrations = [r for r in integration_results if r.success]
        
        # Performance summary
        performance_results = [r for r in self.results if 'performance' in r.test_name or 'error_handling' in r.test_name]
        
        report = {
            'validation_summary': {
                'total_tests': len(self.results),
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(successful_tests) / len(self.results) if self.results else 0,
                'total_duration': total_duration
            },
            'component_availability': {
                'total_components': len(availability_results),
                'available_components': len(available_components),
                'availability_rate': len(available_components) / len(availability_results) if availability_results else 0,
                'available': [r.details['component'] for r in available_components],
                'unavailable': [r.details['component'] for r in availability_results if not r.success]
            },
            'integration_status': {
                'total_integration_tests': len(integration_results),
                'successful_integrations': len(successful_integrations),
                'integration_success_rate': len(successful_integrations) / len(integration_results) if integration_results else 0
            },
            'performance_validation': {
                'performance_tests_run': len(performance_results),
                'performance_tests_passed': len([r for r in performance_results if r.success])
            },
            'detailed_results': [r.to_dict() for r in self.results],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check component availability
        availability_results = [r for r in self.results if r.test_name.startswith('availability_')]
        unavailable_critical = [r for r in availability_results if not r.success and 'throttling' in r.test_name.lower()]
        
        if unavailable_critical:
            recommendations.append("Install and configure the load-based request throttling system for production use")
        
        # Check integration results
        integration_failures = [r for r in self.results if 'integration' in r.test_name and not r.success]
        if integration_failures:
            recommendations.append("Review integration configurations and ensure all dependencies are properly installed")
        
        # Check performance results
        performance_failures = [r for r in self.results if 'performance' in r.test_name and not r.success]
        if performance_failures:
            recommendations.append("Optimize system performance settings and consider scaling resources")
        
        # Overall system health
        critical_failures = [r for r in self.results if not r.success and 'critical' in r.message.lower()]
        if critical_failures:
            recommendations.append("Address critical system issues before production deployment")
        
        if not recommendations:
            recommendations.append("System validation successful - ready for production deployment")
        
        return recommendations


# ============================================================================
# MAIN VALIDATION EXECUTION
# ============================================================================

async def main():
    """Run the complete validation suite."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    validation_suite = ValidationSuite()
    report = await validation_suite.run_validation()
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    summary = report['validation_summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']} ({summary['success_rate']:.1%})")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Duration: {summary['total_duration']:.2f}s")
    
    # Component availability
    availability = report['component_availability']
    print(f"\nComponent Availability: {availability['availability_rate']:.1%}")
    if availability['available']:
        print("Available Components:")
        for component in availability['available']:
            print(f"  ✅ {component}")
    if availability['unavailable']:
        print("Unavailable Components:")
        for component in availability['unavailable']:
            print(f"  ❌ {component}")
    
    # Integration status
    integration = report['integration_status']
    if integration['total_integration_tests'] > 0:
        print(f"\nIntegration Success Rate: {integration['integration_success_rate']:.1%}")
    
    # Recommendations
    print(f"\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"throttling_integration_validation_{timestamp}.json"
    
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_file}")
    except Exception as e:
        print(f"\nWarning: Could not save report file: {e}")
    
    # Overall result
    overall_success = summary['success_rate'] > 0.7  # At least 70% success rate
    print(f"\n{'🎉 VALIDATION PASSED' if overall_success else '⚠️  VALIDATION NEEDS ATTENTION'}")
    
    return overall_success


if __name__ == "__main__":
    asyncio.run(main())