"""
Progressive Service Degradation System - Complete Integration Demo
================================================================

This demonstration shows the complete progressive service degradation system in action,
integrating with the Clinical Metabolomics Oracle's production systems.

The demo illustrates:
1. Enhanced load monitoring detecting system stress
2. Progressive degradation controller responding to load changes
3. Dynamic timeout adjustments across all services
4. Query complexity reduction under load
5. Feature disabling to preserve system stability
6. Integration with production load balancer and RAG systems

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Configure logging for demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Import our progressive degradation system
try:
    try:
        from .progressive_service_degradation_controller import (
            ProgressiveServiceDegradationController,
            DegradationConfiguration,
            TimeoutConfiguration,
            QueryComplexityConfiguration,
            FeatureControlConfiguration,
            SystemLoadLevel
        )
    except ImportError:
        from progressive_service_degradation_controller import (
            ProgressiveServiceDegradationController,
            DegradationConfiguration,
            TimeoutConfiguration,
            QueryComplexityConfiguration,
            FeatureControlConfiguration,
            SystemLoadLevel
        )
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False
    # Define mock classes
    class SystemLoadLevel:
        NORMAL = 0
        ELEVATED = 1 
        HIGH = 2
        CRITICAL = 3
        EMERGENCY = 4
    
    class DegradationConfiguration:
        pass
    
    class TimeoutConfiguration:
        pass
    
    class QueryComplexityConfiguration:
        pass
    
    class FeatureControlConfiguration:
        pass

try:
    from .progressive_degradation_integrations import (
        create_fully_integrated_degradation_system,
        create_degradation_system_from_existing_components,
        ProgressiveDegradationIntegrationManager
    )
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False

try:
    from .enhanced_load_monitoring_system import (
        create_enhanced_load_monitoring_system,
        EnhancedLoadDetectionSystem,
        SystemLoadLevel
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Try to import production systems for real integration
try:
    from .production_load_balancer import ProductionLoadBalancer
    LOAD_BALANCER_AVAILABLE = True
except ImportError:
    LOAD_BALANCER_AVAILABLE = False

try:
    from .clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# ============================================================================
# MOCK SYSTEMS FOR DEMONSTRATION
# ============================================================================

class MockProductionLoadBalancer:
    """Mock production load balancer for demonstration."""
    
    def __init__(self):
        self.backend_instances = {
            'lightrag_1': MockBackendConfig('lightrag_1', 'lightrag', 60.0),
            'lightrag_2': MockBackendConfig('lightrag_2', 'lightrag', 60.0),
            'perplexity_1': MockBackendConfig('perplexity_1', 'perplexity', 35.0),
            'openai_1': MockBackendConfig('openai_1', 'openai', 45.0)
        }
        self.circuit_breaker_settings = {'failure_threshold': 5, 'recovery_timeout': 60}
        self.health_check_settings = {'interval_seconds': 30, 'timeout_seconds': 10}
        
    def update_backend_timeouts(self, timeout_mapping: Dict[str, float]):
        for backend_id, config in self.backend_instances.items():
            for service, timeout in timeout_mapping.items():
                if service.lower() in config.backend_type.lower():
                    config.timeout_seconds = timeout
                    print(f"🔧 Updated {backend_id} timeout to {timeout:.1f}s")
                    break
    
    def update_circuit_breaker_settings(self, settings: Dict[str, Any]):
        self.circuit_breaker_settings.update(settings)
        print(f"🔧 Updated circuit breaker: {settings}")
    
    def update_health_check_settings(self, settings: Dict[str, Any]):
        self.health_check_settings.update(settings)
        print(f"🔧 Updated health checks: {settings}")


class MockBackendConfig:
    """Mock backend configuration."""
    
    def __init__(self, backend_id: str, backend_type: str, timeout: float):
        self.backend_id = backend_id
        self.backend_type = backend_type
        self.timeout_seconds = timeout
        self.health_check_timeout_seconds = 10.0


class MockClinicalRAG:
    """Mock clinical RAG system for demonstration."""
    
    def __init__(self):
        self.config = MockRAGConfig()
        self.query_count = 0
    
    def query(self, query_text: str, **kwargs):
        """Mock synchronous query method."""
        self.query_count += 1
        
        # Show how degradation affects the query
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        mode = kwargs.get('mode', 'hybrid')
        timeout = kwargs.get('timeout', self.config.timeout)
        
        print(f"📝 RAG Query #{self.query_count}:")
        print(f"   Query: '{query_text[:50]}{'...' if len(query_text) > 50 else ''}'")
        print(f"   Params: tokens={max_tokens}, mode={mode}, timeout={timeout:.1f}s")
        
        # Simulate some processing time
        time.sleep(0.1)
        
        return {
            'answer': f"Mock response to query (tokens limited to {max_tokens})",
            'processing_time': 0.1,
            'mode_used': mode,
            'query_simplified': len(query_text) != len(kwargs.get('original_query', query_text))
        }
    
    async def aquery(self, query_text: str, **kwargs):
        """Mock asynchronous query method."""
        return self.query(query_text, **kwargs)


class MockRAGConfig:
    """Mock RAG configuration."""
    
    def __init__(self):
        self.max_tokens = 8000
        self.timeout = 60.0
        self.enable_complex_analytics = True
        self.enable_detailed_logging = True
        self.mode = 'hybrid'


class MockProductionMonitoring:
    """Mock production monitoring system."""
    
    def __init__(self):
        self.monitoring_interval = 5.0
        self.detailed_logging_enabled = True
        
    def update_monitoring_interval(self, interval: float):
        self.monitoring_interval = interval
        print(f"📊 Updated monitoring interval to {interval:.1f}s")
        
    def set_detailed_logging(self, enabled: bool):
        self.detailed_logging_enabled = enabled
        print(f"📊 Detailed logging: {'enabled' if enabled else 'disabled'}")


# ============================================================================
# COMPREHENSIVE DEMONSTRATION
# ============================================================================

class ProgressiveDegradationDemo:
    """Complete demonstration of the progressive degradation system."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enhanced_detector = None
        self.controller = None
        self.integration_manager = None
        
        # Mock systems for demonstration
        self.mock_load_balancer = MockProductionLoadBalancer()
        self.mock_clinical_rag = MockClinicalRAG()
        self.mock_monitoring = MockProductionMonitoring()
        
        # Demo state
        self.current_scenario = None
        self.scenario_start_time = None
        self.load_changes = []
    
    async def initialize_system(self):
        """Initialize the complete degradation system."""
        print("🚀 Initializing Progressive Service Degradation System")
        print("=" * 60)
        
        # Show system availability
        availability = {
            'Controller': CONTROLLER_AVAILABLE,
            'Integrations': INTEGRATIONS_AVAILABLE,  
            'Enhanced Monitoring': MONITORING_AVAILABLE,
            'Load Balancer': LOAD_BALANCER_AVAILABLE,
            'Clinical RAG': RAG_AVAILABLE
        }
        
        print("Component Availability:")
        for component, available in availability.items():
            status = "✓" if available else "✗ (using mock)"
            print(f"  {status} {component}")
        print()
        
        # Create the degradation system with custom configuration
        custom_config = self._create_custom_configuration()
        
        if CONTROLLER_AVAILABLE and INTEGRATIONS_AVAILABLE:
            # Use real components where available
            existing_systems = {
                'load_balancer': self.mock_load_balancer,
                'clinical_rag': self.mock_clinical_rag,
                'monitoring': self.mock_monitoring
            }
            
            self.enhanced_detector, self.controller, self.integration_manager = \
                create_degradation_system_from_existing_components(
                    existing_systems=existing_systems,
                    monitoring_interval=2.0,  # Faster for demo
                    custom_config=custom_config
                )
            
            print("✅ Real progressive degradation system initialized")
            
        else:
            print("❌ Real system components not available - using simplified demo")
            return False
        
        # Add callback to track load changes for demo
        if self.controller:
            self.controller.add_load_change_callback(self._on_load_change_demo)
        
        return True
    
    def _create_custom_configuration(self) -> DegradationConfiguration:
        """Create custom degradation configuration optimized for demo."""
        
        # More aggressive timeout scaling for demo
        timeout_config = TimeoutConfiguration(
            lightrag_base_timeout=30.0,  # Reduced for demo
            literature_search_base_timeout=45.0,
            openai_api_base_timeout=25.0,
            perplexity_api_base_timeout=20.0,
            # More aggressive scaling factors
            lightrag_factors=[1.0, 0.7, 0.4, 0.25, 0.15],
            openai_api_factors=[1.0, 0.75, 0.5, 0.3, 0.2]
        )
        
        # More dramatic complexity reduction
        complexity_config = QueryComplexityConfiguration(
            token_limits=[6000, 4000, 2500, 1500, 800],
            result_depths=[8, 6, 4, 2, 1],
            query_modes=['hybrid', 'local', 'local', 'simple', 'simple']
        )
        
        # Progressive feature disabling
        feature_config = FeatureControlConfiguration(
            # Disable features more aggressively for demo
            detailed_logging=[True, True, False, False, False],
            complex_analytics=[True, False, False, False, False],
            confidence_analysis=[True, True, False, False, False]
        )
        
        return DegradationConfiguration(
            timeout_config=timeout_config,
            complexity_config=complexity_config,
            feature_config=feature_config,
            min_degradation_duration=10.0,  # Shorter for demo
            recovery_delay=15.0
        )
    
    def _on_load_change_demo(self, previous_level: SystemLoadLevel, new_level: SystemLoadLevel):
        """Demo callback for load level changes."""
        change_info = {
            'timestamp': datetime.now().isoformat(),
            'previous_level': previous_level.name,
            'new_level': new_level.name,
            'scenario': self.current_scenario
        }
        self.load_changes.append(change_info)
        
        print(f"\n🔄 LOAD LEVEL CHANGE: {previous_level.name} → {new_level.name}")
        print(f"   Scenario: {self.current_scenario}")
        print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Show immediate effects
        status = self.controller.get_current_status()
        self._show_degradation_effects(status)
    
    def _show_degradation_effects(self, status: Dict[str, Any]):
        """Show the effects of degradation."""
        
        # Timeout effects
        timeouts = status['timeouts']
        print(f"   ⏱️  Timeouts: LightRAG={timeouts.get('lightrag', 0):.1f}s, "
              f"OpenAI={timeouts.get('openai_api', 0):.1f}s")
        
        # Complexity effects
        complexity = status['query_complexity']
        print(f"   🧠 Complexity: {complexity.get('token_limit', 0)} tokens, "
              f"mode={complexity.get('query_mode', 'unknown')}")
        
        # Feature effects
        features = status['feature_settings']
        disabled_count = len([k for k, v in features.items() if not v])
        print(f"   🔧 Features: {disabled_count} disabled")
        
        if status['emergency_mode']:
            print(f"   🚨 EMERGENCY MODE ACTIVE")
        
        print("   " + "-" * 40)
    
    async def run_load_scenarios(self):
        """Run through various load scenarios."""
        print("\n🎭 Running Load Scenarios")
        print("=" * 40)
        
        scenarios = [
            {
                'name': 'Normal Operations',
                'level': SystemLoadLevel.NORMAL,
                'description': 'System operating normally with full features',
                'duration': 3
            },
            {
                'name': 'Traffic Spike',
                'level': SystemLoadLevel.ELEVATED,
                'description': 'Moderate traffic increase - minor optimizations',
                'duration': 4
            },
            {
                'name': 'High Load Event',
                'level': SystemLoadLevel.HIGH,
                'description': 'High system load - timeout reductions and feature limits',
                'duration': 5
            },
            {
                'name': 'Critical Overload',
                'level': SystemLoadLevel.CRITICAL,
                'description': 'Critical system overload - aggressive degradation',
                'duration': 4
            },
            {
                'name': 'Emergency Situation',
                'level': SystemLoadLevel.EMERGENCY,
                'description': '🚨 EMERGENCY - Maximum degradation to preserve system',
                'duration': 6
            },
            {
                'name': 'Gradual Recovery',
                'level': SystemLoadLevel.HIGH,
                'description': 'Load decreasing - partial recovery',
                'duration': 3
            },
            {
                'name': 'Full Recovery',
                'level': SystemLoadLevel.NORMAL,
                'description': 'System fully recovered - all features restored',
                'duration': 3
            }
        ]
        
        for scenario in scenarios:
            await self._run_scenario(scenario)
    
    async def _run_scenario(self, scenario: Dict[str, Any]):
        """Run a single load scenario."""
        self.current_scenario = scenario['name']
        self.scenario_start_time = datetime.now()
        
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Target Level: {scenario['level'].name}")
        
        # Force the load level
        self.controller.force_load_level(scenario['level'], f"Demo scenario: {scenario['name']}")
        
        # Demonstrate query processing under this load level
        await self._demonstrate_query_processing(scenario['level'])
        
        # Wait for scenario duration
        print(f"   ⏳ Running scenario for {scenario['duration']} seconds...")
        await asyncio.sleep(scenario['duration'])
    
    async def _demonstrate_query_processing(self, load_level: SystemLoadLevel):
        """Demonstrate query processing at the current load level."""
        
        # Sample queries of varying complexity
        sample_queries = [
            "What is metabolomics?",
            "Explain the role of metabolomics in cardiovascular disease diagnosis and how metabolite profiles can be used as biomarkers for early detection of heart conditions, including the specific metabolic pathways involved.",
            "How do metabolic biomarkers help in cancer research?"
        ]
        
        for i, query in enumerate(sample_queries):
            if i >= len(sample_queries) - 1 and load_level >= SystemLoadLevel.CRITICAL:
                # Skip complex queries under critical load
                print(f"   🚫 Skipping complex query due to {load_level.name} load")
                continue
            
            # Process query through the degradation-aware system
            simplified_query = self.controller.simplify_query(query)
            
            if simplified_query != query:
                print(f"   ✂️  Query simplified: {len(query)} → {len(simplified_query)} chars")
            
            # Show timeout that would be applied
            timeout = self.controller.get_timeout_for_service('lightrag')
            print(f"   ⏱️  Query timeout: {timeout:.1f}s")
            
            # Simulate processing through RAG system
            if hasattr(self.mock_clinical_rag, 'query'):
                result = self.mock_clinical_rag.query(
                    simplified_query,
                    max_tokens=self.controller.complexity_manager.get_query_params().get('token_limit', 8000),
                    timeout=timeout,
                    mode=self.controller.complexity_manager.get_query_params().get('query_mode', 'hybrid')
                )
    
    async def show_system_metrics(self):
        """Show comprehensive system metrics during demo."""
        print(f"\n📊 System Metrics Summary")
        print("=" * 40)
        
        if self.controller:
            status = self.controller.get_current_status()
            
            print(f"Current Status:")
            print(f"  Load Level: {status['load_level']}")
            print(f"  Degradation Active: {status['degradation_active']}")
            print(f"  Emergency Mode: {status['emergency_mode']}")
            
            print(f"\nPerformance Metrics:")
            metrics = status['metrics']
            print(f"  Level Changes: {metrics['level_changes']}")
            print(f"  Total Degradations: {metrics['total_degradations']}")
            print(f"  Emergency Activations: {metrics['emergency_activations']}")
            
            print(f"\nIntegrated Systems:")
            for system in status['integrated_systems']:
                print(f"  ✓ {system}")
        
        if self.integration_manager:
            integration_status = self.integration_manager.get_integration_status()
            print(f"\nIntegration Results:")
            for system, success in integration_status['integration_results'].items():
                status_icon = "✓" if success else "✗"
                print(f"  {status_icon} {system}")
        
        print(f"\nLoad Change History:")
        for i, change in enumerate(self.load_changes[-5:]):  # Show last 5 changes
            print(f"  {i+1}. {change['previous_level']} → {change['new_level']} "
                  f"({change['scenario']})")
    
    async def cleanup(self):
        """Clean up the demonstration."""
        print(f"\n🧹 Cleaning Up")
        
        if self.integration_manager:
            self.integration_manager.rollback_all_integrations()
        
        if self.enhanced_detector:
            await self.enhanced_detector.stop_monitoring()
        
        print("✅ Cleanup completed")


# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

async def run_complete_progressive_degradation_demo():
    """Run the complete progressive degradation demonstration."""
    print("🎪 Clinical Metabolomics Oracle - Progressive Service Degradation Demo")
    print("=" * 80)
    print()
    
    demo = ProgressiveDegradationDemo()
    
    try:
        # Initialize the system
        if not await demo.initialize_system():
            print("❌ Could not initialize demo system - exiting")
            return
        
        # Start monitoring if available
        if demo.enhanced_detector:
            await demo.enhanced_detector.start_monitoring()
            print("✅ Enhanced load monitoring started")
        
        # Run through load scenarios
        await demo.run_load_scenarios()
        
        # Show final metrics
        await demo.show_system_metrics()
        
        print(f"\n🎊 Progressive Service Degradation Demo Completed Successfully!")
        print(f"The system demonstrated:")
        print(f"  • Dynamic timeout scaling across {len(demo.load_changes)} load changes")
        print(f"  • Progressive query complexity reduction")
        print(f"  • Selective feature disabling under load")
        print(f"  • Seamless integration with production systems")
        print(f"  • Automatic recovery as load decreases")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        logging.exception("Demo error")
    finally:
        await demo.cleanup()


def create_production_degradation_config() -> Dict[str, Any]:
    """Create a production-ready degradation configuration."""
    return {
        'timeout_scaling': {
            'lightrag': {
                'base_timeout': 60.0,
                'scaling_factors': [1.0, 0.75, 0.5, 0.33, 0.17],
                'description': 'LightRAG query timeouts by load level'
            },
            'openai_api': {
                'base_timeout': 45.0,
                'scaling_factors': [1.0, 0.8, 0.6, 0.49, 0.31],
                'description': 'OpenAI API timeouts by load level'
            },
            'perplexity_api': {
                'base_timeout': 35.0,
                'scaling_factors': [1.0, 0.8, 0.6, 0.49, 0.29],
                'description': 'Perplexity API timeouts by load level'
            }
        },
        'query_complexity': {
            'token_limits': [8000, 6000, 4000, 2000, 1000],
            'result_depths': [10, 8, 5, 2, 1],
            'query_modes': ['hybrid', 'hybrid', 'local', 'simple', 'simple'],
            'description': 'Progressive query complexity reduction'
        },
        'feature_control': {
            'detailed_logging': [True, True, False, False, False],
            'complex_analytics': [True, True, True, False, False],
            'confidence_analysis': [True, True, False, False, False],
            'background_tasks': [True, True, True, False, False],
            'description': 'Features disabled at each load level'
        },
        'load_levels': [
            {'level': 'NORMAL', 'description': 'Full functionality, optimal performance'},
            {'level': 'ELEVATED', 'description': 'Minor optimizations, reduced logging detail'},
            {'level': 'HIGH', 'description': 'Timeout reductions, query complexity limits'},
            {'level': 'CRITICAL', 'description': 'Aggressive timeout cuts, feature disabling'},
            {'level': 'EMERGENCY', 'description': 'Minimal functionality, system preservation'}
        ]
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Show production configuration
    print("Production Configuration:")
    config = create_production_degradation_config()
    print(json.dumps(config, indent=2))
    print()
    
    # Run the complete demonstration
    asyncio.run(run_complete_progressive_degradation_demo())