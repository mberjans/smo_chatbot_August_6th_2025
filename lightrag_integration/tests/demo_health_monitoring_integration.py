#!/usr/bin/env python3
"""
System Health Monitoring Integration Demo

This demonstration script shows how the health monitoring integration works
in practice, illustrating key concepts and behaviors of the health-aware routing system.

Features:
- Interactive health monitoring demonstration
- Real-time health status visualization  
- Circuit breaker behavior demonstration
- Service recovery simulation
- Performance impact illustration

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: Health monitoring integration demonstration
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import test components (these would be the real components in production)
try:
    from test_system_health_monitoring_integration import (
        MockSystemHealthManager,
        HealthAwareRouter,
        ServiceStatus
    )
except ImportError as e:
    print(f"Could not import test components: {e}")
    print("This demo requires the test infrastructure to be available.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class HealthMonitoringDemo:
    """Interactive demonstration of health monitoring integration."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.health_manager = MockSystemHealthManager()
        self.router = HealthAwareRouter(self.health_manager)
        
        # Demo queries
        self.sample_queries = [
            "What is the relationship between glucose and insulin?",
            "Latest metabolomics research 2025", 
            "How does mass spectrometry work?",
            "Current advances in biomarker discovery",
            "Mechanism of protein folding in diabetes"
        ]
    
    def run_interactive_demo(self):
        """Run interactive health monitoring demonstration."""
        print("=" * 80)
        print("SYSTEM HEALTH MONITORING INTEGRATION DEMONSTRATION")
        print("=" * 80)
        print()
        
        # Phase 1: Normal Operation
        print("📊 PHASE 1: Normal System Operation")
        print("-" * 50)
        self._demonstrate_normal_operation()
        self._wait_for_input("\nPress Enter to continue to Phase 2...")
        
        # Phase 2: Service Degradation
        print("\n⚠️  PHASE 2: Service Performance Degradation")
        print("-" * 50)
        self._demonstrate_service_degradation()
        self._wait_for_input("\nPress Enter to continue to Phase 3...")
        
        # Phase 3: Service Failure and Circuit Breaker
        print("\n🚨 PHASE 3: Service Failure and Circuit Breaker")
        print("-" * 50)
        self._demonstrate_service_failure()
        self._wait_for_input("\nPress Enter to continue to Phase 4...")
        
        # Phase 4: Service Recovery
        print("\n✅ PHASE 4: Service Recovery")
        print("-" * 50)
        self._demonstrate_service_recovery()
        self._wait_for_input("\nPress Enter to continue to summary...")
        
        # Summary
        print("\n📈 DEMONSTRATION SUMMARY")
        print("-" * 50)
        self._show_final_summary()
    
    def _demonstrate_normal_operation(self):
        """Demonstrate normal system operation with healthy services."""
        print("All services are healthy. Observing normal routing behavior...")
        print()
        
        # Show initial health status
        self._display_service_health()
        print()
        
        # Test routing with healthy services
        print("Testing routing with healthy services:")
        for i, query in enumerate(self.sample_queries[:3], 1):
            print(f"\n{i}. Query: '{query[:50]}...'")
            result = self.router.route_query_with_health_awareness(query)
            
            print(f"   → Routed to: {result.routing_decision}")
            print(f"   → Confidence: {result.confidence:.3f}")
            print(f"   → Health Score: {result.metadata.get('global_health_score', 0):.3f}")
            
            # Brief delay for realism
            time.sleep(0.2)
        
        # Show routing statistics
        print(f"\n📊 Routing Statistics:")
        stats = self.router.get_routing_statistics()
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Current Global Health: {stats['current_global_health']:.3f}")
        print(f"   Healthy Services: {stats['healthy_services']}")
    
    def _demonstrate_service_degradation(self):
        """Demonstrate system behavior when services degrade."""
        print("Injecting performance degradation into LightRAG service...")
        
        # Inject performance degradation
        self.health_manager.inject_service_degradation('lightrag', enabled=True)
        
        # Let degradation be detected
        print("Allowing system to detect performance degradation...")
        for _ in range(15):
            self.health_manager.services['lightrag'].simulate_request()
        
        # Show degraded health status
        print("\n🔍 Service Health After Degradation:")
        self._display_service_health()
        print()
        
        # Test routing with degraded service
        print("Testing routing with degraded LightRAG service:")
        query = "How does the glycolysis pathway relate to diabetes?"
        print(f"Query: '{query}'")
        
        result = self.router.route_query_with_health_awareness(query)
        print(f"   → Routing Decision: {result.routing_decision}")
        print(f"   → Confidence: {result.confidence:.3f}")
        print(f"   → Reasoning: {result.reasoning}")
        
        # Show adaptation in routing
        print("\n📈 System Adaptation:")
        stats = self.router.get_routing_statistics()
        print(f"   Health-Based Routing Decisions: {stats['health_based_decisions']}")
        print(f"   Current Global Health: {stats['current_global_health']:.3f}")
    
    def _demonstrate_service_failure(self):
        """Demonstrate circuit breaker activation on service failure."""
        print("Injecting complete failure into Perplexity service...")
        
        # Inject service failure
        self.health_manager.inject_service_failure('perplexity', enabled=True)
        
        # Trigger circuit breaker
        print("Simulating requests to trigger circuit breaker...")
        perplexity_monitor = self.health_manager.services['perplexity']
        for _ in range(12):
            success, response_time = perplexity_monitor.simulate_request()
            if not success:
                print(f"   Request failed (failures: {perplexity_monitor.consecutive_failures})")
        
        # Show circuit breaker activation
        print(f"\n🔴 Circuit Breaker Status: {perplexity_monitor.circuit_breaker_state.value.upper()}")
        
        # Show health status with failure
        print("\n🔍 Service Health After Failure:")
        self._display_service_health()
        print()
        
        # Test routing with failed service
        print("Testing routing with failed Perplexity service:")
        query = "Latest advances in clinical metabolomics 2025"
        print(f"Query: '{query}'")
        
        result = self.router.route_query_with_health_awareness(query)
        print(f"   → Routing Decision: {result.routing_decision}")
        print(f"   → Confidence: {result.confidence:.3f}")
        print(f"   → Reasoning: {result.reasoning}")
        
        # Show circuit breaker statistics
        print("\n⚡ Circuit Breaker Impact:")
        stats = self.router.get_routing_statistics()
        print(f"   Circuit Breaker Blocks: {stats['circuit_breaker_blocks']}")
        print(f"   Unhealthy Services: {stats['unhealthy_services']}")
    
    def _demonstrate_service_recovery(self):
        """Demonstrate service recovery and routing restoration."""
        print("Initiating service recovery...")
        
        # Disable failure injection (service recovery)
        self.health_manager.inject_service_failure('perplexity', enabled=False)
        self.health_manager.inject_service_degradation('lightrag', enabled=False)
        
        # Simulate recovery
        print("Simulating successful requests to demonstrate recovery...")
        for service_name in ['lightrag', 'perplexity']:
            monitor = self.health_manager.services[service_name]
            for _ in range(20):
                monitor.simulate_request()
        
        # Show recovered health status
        print("\n🔍 Service Health After Recovery:")
        self._display_service_health()
        print()
        
        # Test routing after recovery
        print("Testing routing after service recovery:")
        for i, query in enumerate(self.sample_queries[-2:], 1):
            print(f"\n{i}. Query: '{query[:50]}...'")
            result = self.router.route_query_with_health_awareness(query)
            
            print(f"   → Routed to: {result.routing_decision}")
            print(f"   → Confidence: {result.confidence:.3f}")
            print(f"   → Health considerations in reasoning: {'yes' if any('health' in r.lower() for r in result.reasoning) else 'no'}")
    
    def _display_service_health(self):
        """Display current health status of all services."""
        for service_name in ['lightrag', 'perplexity', 'llm_classifier']:
            health = self.health_manager.get_service_health(service_name)
            if health:
                status_icon = {
                    ServiceStatus.HEALTHY: "🟢",
                    ServiceStatus.DEGRADED: "🟡", 
                    ServiceStatus.UNHEALTHY: "🔴",
                    ServiceStatus.UNKNOWN: "⚪"
                }.get(health.status, "❓")
                
                print(f"   {status_icon} {service_name:15} | Status: {health.status.value:10} | "
                      f"Response: {health.response_time_ms:6.1f}ms | "
                      f"Error Rate: {health.error_rate:5.1%} | "
                      f"Score: {health.performance_score:.3f}")
    
    def _show_final_summary(self):
        """Show final demonstration summary."""
        print("Health monitoring integration demonstration completed successfully!")
        print()
        
        # Final statistics
        stats = self.router.get_routing_statistics()
        print("📊 Final System Statistics:")
        print(f"   Total Requests Processed: {stats['total_requests']}")
        print(f"   Health-Based Routing Decisions: {stats['health_based_decisions']}")
        print(f"   Fallback Decisions: {stats['fallback_decisions']}")
        print(f"   Circuit Breaker Blocks: {stats['circuit_breaker_blocks']}")
        print(f"   Final Global Health Score: {stats['current_global_health']:.3f}")
        
        # Calculate percentages
        total = max(stats['total_requests'], 1)
        print(f"\n📈 Performance Metrics:")
        print(f"   Health-Based Routing Rate: {(stats['health_based_decisions'] / total) * 100:.1f}%")
        print(f"   Fallback Rate: {(stats['fallback_decisions'] / total) * 100:.1f}%")
        print(f"   Circuit Breaker Block Rate: {(stats['circuit_breaker_blocks'] / total) * 100:.1f}%")
        
        print("\n✅ Key Behaviors Demonstrated:")
        print("   • Health monitoring integration with routing decisions")
        print("   • Automatic service degradation detection and response") 
        print("   • Circuit breaker activation on service failures")
        print("   • Service recovery detection and routing restoration")
        print("   • Performance impact on routing confidence")
        print("   • Load balancing based on service health")
        
        print("\n🎯 Production Benefits:")
        print("   • Improved system resilience and fault tolerance")
        print("   • Automatic adaptation to service health changes")
        print("   • Prevention of cascading failures")
        print("   • Optimized resource utilization")
        print("   • Enhanced user experience through intelligent routing")
    
    def _wait_for_input(self, message: str):
        """Wait for user input with message."""
        try:
            input(message)
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            sys.exit(0)
    
    def run_automated_demo(self):
        """Run automated demonstration without user interaction."""
        print("=" * 80)
        print("AUTOMATED HEALTH MONITORING INTEGRATION DEMO")
        print("=" * 80)
        
        phases = [
            ("Normal Operation", self._demonstrate_normal_operation),
            ("Service Degradation", self._demonstrate_service_degradation),
            ("Service Failure", self._demonstrate_service_failure),
            ("Service Recovery", self._demonstrate_service_recovery)
        ]
        
        for phase_name, phase_func in phases:
            print(f"\n{'='*20} {phase_name.upper()} {'='*20}")
            phase_func()
            time.sleep(2)  # Brief pause between phases
        
        print(f"\n{'='*20} SUMMARY {'='*20}")
        self._show_final_summary()


def main():
    """Main demonstration function."""
    try:
        demo = HealthMonitoringDemo()
        
        # Check if running interactively
        if len(sys.argv) > 1 and sys.argv[1] == "--automated":
            demo.run_automated_demo()
        else:
            demo.run_interactive_demo()
        
        print("\n" + "=" * 80)
        print("Thank you for exploring the health monitoring integration!")
        print("For more details, see: SYSTEM_HEALTH_MONITORING_INTEGRATION_README.md")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()