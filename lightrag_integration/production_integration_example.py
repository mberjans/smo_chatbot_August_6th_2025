#!/usr/bin/env python3
"""
Production Integration Example for Enhanced LLM Query Classifier

This script demonstrates how to integrate the Enhanced LLM Query Classifier
into a production Clinical Metabolomics Oracle system with full compatibility
with existing infrastructure, monitoring, and operational requirements.

Key Features Demonstrated:
    - Production-ready configuration management
    - Integration with existing BiomedicalQueryRouter
    - Health monitoring and alerting
    - Cost management and budget controls
    - Performance optimization
    - Graceful degradation and fallback mechanisms
    - Comprehensive logging and metrics

Usage:
    python production_integration_example.py

Requirements:
    - OpenAI API key (set OPENAI_API_KEY environment variable)
    - All dependencies from requirements.txt
    - Existing Clinical Metabolomics Oracle infrastructure

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import asyncio
import logging
import json
import time
import os
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('production_llm_classifier.log'),
        logging.FileHandler('system_health.log')
    ]
)

logger = logging.getLogger(__name__)

# Import enhanced LLM classifier components
try:
    from enhanced_llm_classifier import (
        EnhancedLLMQueryClassifier,
        EnhancedLLMConfig,
        LLMProvider,
        create_enhanced_llm_classifier,
        llm_classifier_context,
        convert_enhanced_result_to_routing_prediction
    )
    
    from llm_config_manager import (
        ConfigManager,
        ConfigValidator,
        DeploymentEnvironment,
        UseCaseType,
        validate_environment_setup,
        create_optimized_config
    )
    
    # Import existing infrastructure for integration
    try:
        from query_router import BiomedicalQueryRouter, RoutingPrediction
        from query_classification_system import QueryClassificationEngine, ClassificationResult
        EXISTING_INFRASTRUCTURE_AVAILABLE = True
        logger.info("Successfully imported existing infrastructure")
    except ImportError as e:
        EXISTING_INFRASTRUCTURE_AVAILABLE = False
        logger.warning(f"Existing infrastructure not available: {e}")
        logger.warning("Running in standalone mode")
    
except ImportError as e:
    logger.error(f"Failed to import enhanced LLM classifier: {e}")
    sys.exit(1)


@dataclass
class SystemHealthMetrics:
    """System health metrics for monitoring."""
    
    timestamp: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    cache_hit_rate: float
    daily_cost: float
    daily_budget_utilization: float
    circuit_breaker_state: str
    system_uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProductionLLMClassificationService:
    """
    Production-ready LLM classification service that integrates with existing
    Clinical Metabolomics Oracle infrastructure.
    
    Features:
    - Comprehensive health monitoring
    - Cost management and alerting
    - Performance optimization
    - Graceful degradation
    - Integration with existing systems
    """
    
    def __init__(self):
        """Initialize the production service."""
        self.start_time = time.time()
        self.config_manager = ConfigManager(logger)
        self.validator = ConfigValidator(logger)
        
        # Service state
        self.is_running = False
        self.classifier = None
        self.config = None
        self.biomedical_router = None
        
        # Metrics tracking
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.health_check_interval = 60  # seconds
        self.last_health_check = 0
        
        # Alerts and notifications
        self.alert_handlers = []
        self.performance_degradation_threshold = 5000  # ms
        self.cost_alert_threshold = 0.9  # 90% of budget
        
        logger.info("Production LLM Classification Service initialized")
    
    async def initialize(self, 
                        environment: str = None,
                        use_case: str = None,
                        config_file: str = None) -> None:
        """
        Initialize the service with configuration and dependencies.
        
        Args:
            environment: Deployment environment ("production", "staging", etc.)
            use_case: Use case optimization ("high_volume", "cost_sensitive", etc.)
            config_file: Optional configuration file path
        """
        
        logger.info("Initializing production LLM classification service...")
        
        # Validate environment setup
        env_status = validate_environment_setup()
        if not env_status["ready"]:
            error_msg = f"Environment not ready: {env_status['recommendations']}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("✅ Environment validation passed")
        
        # Load and validate configuration
        try:
            env_enum = DeploymentEnvironment(environment.lower()) if environment else None
            use_case_enum = UseCaseType(use_case.lower()) if use_case else None
            
            self.config = self.config_manager.load_config(
                environment=env_enum,
                use_case=use_case_enum,
                config_file=config_file,
                validate=True
            )
            
            logger.info(f"✅ Configuration loaded for {environment or 'auto-detected'} environment")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
        
        # Initialize existing infrastructure if available
        if EXISTING_INFRASTRUCTURE_AVAILABLE:
            try:
                self.biomedical_router = BiomedicalQueryRouter(logger)
                logger.info("✅ Biomedical router initialized for fallback")
            except Exception as e:
                logger.warning(f"Could not initialize biomedical router: {e}")
        
        # Create enhanced LLM classifier
        try:
            self.classifier = await create_enhanced_llm_classifier(
                config=self.config,
                biomedical_router=self.biomedical_router,
                logger=logger
            )
            
            logger.info("✅ Enhanced LLM classifier created")
            
        except Exception as e:
            logger.error(f"Failed to create LLM classifier: {e}")
            raise
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.is_running = True
        logger.info("🚀 Production LLM Classification Service ready")
    
    async def classify_query_with_monitoring(self,
                                           query_text: str,
                                           context: Optional[Dict[str, Any]] = None,
                                           priority: str = "normal") -> Tuple[Dict[str, Any], SystemHealthMetrics]:
        """
        Classify query with comprehensive monitoring and health tracking.
        
        Args:
            query_text: Query text to classify
            context: Optional context information
            priority: Request priority ("low", "normal", "high")
            
        Returns:
            Tuple of (classification result, health metrics)
        """
        
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Perform classification
            result, metadata = await self.classifier.classify_query(
                query_text=query_text,
                context=context,
                priority=priority
            )
            
            # Convert to compatible format
            response = {
                "classification_result": {
                    "category": result.category,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "biomedical_signals": result.biomedical_signals,
                    "temporal_signals": result.temporal_signals,
                    "uncertainty_indicators": result.uncertainty_indicators
                },
                "metadata": metadata,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat(),
                "service_version": "2.0.0"
            }
            
            # Convert to legacy format if requested
            if context and context.get("return_legacy_format"):
                try:
                    routing_prediction = convert_enhanced_result_to_routing_prediction(
                        result, metadata, query_text
                    )
                    response["legacy_routing_prediction"] = {
                        "routing_decision": routing_prediction.routing_decision.value,
                        "confidence": routing_prediction.confidence,
                        "research_category": routing_prediction.research_category.value,
                        "reasoning": routing_prediction.reasoning
                    }
                except Exception as e:
                    logger.warning(f"Failed to convert to legacy format: {e}")
            
            self.successful_requests += 1
            
            # Check for performance degradation
            response_time = response["processing_time_ms"]
            if response_time > self.performance_degradation_threshold:
                await self._handle_performance_alert(response_time, query_text)
            
            # Periodic health check
            await self._periodic_health_check()
            
            # Generate health metrics
            health_metrics = await self._generate_health_metrics()
            
            return response, health_metrics
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Classification failed for query '{query_text[:50]}...': {e}")
            
            # Return error response
            error_response = {
                "error": str(e),
                "classification_result": None,
                "metadata": {"error": True, "used_fallback": True},
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat(),
                "service_version": "2.0.0"
            }
            
            health_metrics = await self._generate_health_metrics()
            return error_response, health_metrics
    
    async def _periodic_health_check(self) -> None:
        """Perform periodic health checks and optimizations."""
        
        current_time = time.time()
        
        if current_time - self.last_health_check > self.health_check_interval:
            self.last_health_check = current_time
            
            # Get comprehensive system statistics
            stats = self.classifier.get_comprehensive_stats()
            
            # Check cost alerts
            cost_utilization = stats["cost_stats"]["daily_utilization"]
            if cost_utilization > self.cost_alert_threshold:
                await self._handle_cost_alert(cost_utilization, stats["cost_stats"])
            
            # Check circuit breaker status
            cb_state = stats["circuit_breaker_stats"]["state"]
            if cb_state == "open":
                await self._handle_circuit_breaker_alert(stats["circuit_breaker_stats"])
            
            # Performance optimization
            performance_stats = stats["performance_stats"]
            if performance_stats["avg_response_time"] > self.config.performance.target_response_time_ms:
                await self._handle_performance_optimization(stats)
            
            # Log health summary
            logger.info(f"Health Check - Requests: {self.request_count}, "
                       f"Success Rate: {self.successful_requests/max(1,self.request_count)*100:.1f}%, "
                       f"Avg Response: {performance_stats['avg_response_time']:.1f}ms, "
                       f"Cost: ${stats['cost_stats']['daily_cost']:.4f}")
    
    async def _generate_health_metrics(self) -> SystemHealthMetrics:
        """Generate current system health metrics."""
        
        if not self.classifier:
            return SystemHealthMetrics(
                timestamp=datetime.now().isoformat(),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time_ms=0,
                cache_hit_rate=0,
                daily_cost=0,
                daily_budget_utilization=0,
                circuit_breaker_state="unknown",
                system_uptime_seconds=0
            )
        
        stats = self.classifier.get_comprehensive_stats()
        
        return SystemHealthMetrics(
            timestamp=datetime.now().isoformat(),
            total_requests=self.request_count,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            avg_response_time_ms=stats["performance_stats"]["avg_response_time"],
            cache_hit_rate=stats["cache_stats"]["hit_rate"],
            daily_cost=stats["cost_stats"]["daily_cost"],
            daily_budget_utilization=stats["cost_stats"]["daily_utilization"],
            circuit_breaker_state=stats["circuit_breaker_stats"]["state"],
            system_uptime_seconds=time.time() - self.start_time
        )
    
    async def _handle_performance_alert(self, response_time_ms: float, query: str) -> None:
        """Handle performance degradation alerts."""
        
        alert_data = {
            "type": "performance_degradation",
            "response_time_ms": response_time_ms,
            "threshold_ms": self.performance_degradation_threshold,
            "query_snippet": query[:100],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(f"Performance Alert: {response_time_ms:.1f}ms response time "
                      f"exceeds {self.performance_degradation_threshold}ms threshold")
        
        await self._send_alert(alert_data)
    
    async def _handle_cost_alert(self, utilization: float, cost_stats: Dict[str, Any]) -> None:
        """Handle budget utilization alerts."""
        
        alert_data = {
            "type": "budget_utilization",
            "utilization": utilization,
            "threshold": self.cost_alert_threshold,
            "daily_cost": cost_stats["daily_cost"],
            "daily_budget": cost_stats["daily_budget"],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(f"Cost Alert: {utilization:.1%} budget utilization "
                      f"exceeds {self.cost_alert_threshold:.1%} threshold")
        
        await self._send_alert(alert_data)
    
    async def _handle_circuit_breaker_alert(self, cb_stats: Dict[str, Any]) -> None:
        """Handle circuit breaker alerts."""
        
        alert_data = {
            "type": "circuit_breaker_open",
            "state": cb_stats["state"],
            "failure_count": cb_stats["failure_count"],
            "success_rate": cb_stats["success_rate"],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.error(f"Circuit Breaker Alert: Circuit is {cb_stats['state']} "
                    f"with {cb_stats['success_rate']:.1%} success rate")
        
        await self._send_alert(alert_data)
    
    async def _handle_performance_optimization(self, stats: Dict[str, Any]) -> None:
        """Handle automatic performance optimization."""
        
        logger.info("Performing automatic performance optimization...")
        
        try:
            # Get optimization recommendations
            optimization = self.classifier.get_optimization_recommendations()
            
            if optimization["overall_health"] == "needs_attention":
                high_priority_recs = [r for r in optimization["recommendations"] if r["priority"] == "high"]
                
                if high_priority_recs:
                    logger.info(f"Applying {len(high_priority_recs)} high-priority optimizations")
                    
                    # Apply automatic optimizations
                    optimization_result = await self.classifier.optimize_system(auto_apply=True)
                    
                    if optimization_result["actions_taken"]:
                        logger.info(f"Applied optimizations: {optimization_result['actions_taken']}")
        
        except Exception as e:
            logger.error(f"Failed to apply performance optimizations: {e}")
    
    async def _send_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send alert to configured handlers."""
        
        # Log alert
        logger.info(f"Sending alert: {alert_data['type']}")
        
        # Send to external alert handlers (webhook, email, etc.)
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Failed to send alert via handler: {e}")
    
    def add_alert_handler(self, handler) -> None:
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring dashboards."""
        
        if not self.classifier:
            return {"status": "not_initialized", "error": "Service not initialized"}
        
        try:
            stats = self.classifier.get_comprehensive_stats()
            optimization = self.classifier.get_optimization_recommendations()
            health_metrics = await self._generate_health_metrics()
            
            return {
                "status": "running" if self.is_running else "stopped",
                "service_info": {
                    "version": "2.0.0",
                    "uptime_seconds": time.time() - self.start_time,
                    "total_requests": self.request_count,
                    "success_rate": self.successful_requests / max(1, self.request_count)
                },
                "performance": {
                    "avg_response_time_ms": stats["performance_stats"]["avg_response_time"],
                    "target_response_time_ms": self.config.performance.target_response_time_ms,
                    "target_compliance_rate": stats["performance_stats"]["target_compliance_rate"],
                    "p95_response_time_ms": stats["performance_stats"]["p95_response_time"]
                },
                "reliability": {
                    "circuit_breaker_state": stats["circuit_breaker_stats"]["state"],
                    "circuit_breaker_success_rate": stats["circuit_breaker_stats"]["success_rate"],
                    "fallback_rate": stats["performance_stats"]["fallback_rate"]
                },
                "cost_management": {
                    "daily_cost": stats["cost_stats"]["daily_cost"],
                    "daily_budget": stats["cost_stats"]["daily_budget"],
                    "budget_utilization": stats["cost_stats"]["daily_utilization"],
                    "cost_per_request": stats["cost_stats"]["avg_cost_per_classification"]
                },
                "cache_performance": {
                    "hit_rate": stats["cache_stats"]["hit_rate"],
                    "cache_size": stats["cache_stats"]["cache_size"],
                    "utilization": stats["cache_stats"]["utilization"]
                },
                "health": {
                    "overall_health": optimization["overall_health"],
                    "health_score": optimization["health_score"],
                    "recommendations_count": len(optimization["recommendations"])
                },
                "configuration": {
                    "model": self.config.model_name,
                    "timeout_seconds": self.config.timeout_seconds,
                    "cache_enabled": self.config.cache.enable_caching
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"status": "error", "error": str(e)}
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the service."""
        
        logger.info("Shutting down production LLM classification service...")
        self.is_running = False
        
        if self.classifier:
            # Log final statistics
            final_stats = self.classifier.get_comprehensive_stats()
            logger.info(f"Final Statistics - Total Requests: {self.request_count}, "
                       f"Success Rate: {self.successful_requests/max(1,self.request_count)*100:.1f}%, "
                       f"Total Cost: ${final_stats['cost_stats']['daily_cost']:.4f}")
        
        logger.info("✅ Service shutdown complete")


# ============================================================================
# PRODUCTION DEMONSTRATION AND TESTING
# ============================================================================

class ProductionDemo:
    """Demonstration of production integration features."""
    
    def __init__(self):
        self.service = ProductionLLMClassificationService()
        self.test_queries = [
            {
                "query": "What is the relationship between glucose metabolism and insulin resistance in metabolic syndrome?",
                "expected_category": "KNOWLEDGE_GRAPH",
                "context": {"user_id": "demo_user", "session_id": "demo_session_1"}
            },
            {
                "query": "Latest FDA approvals for metabolomics-based diagnostic tests in 2024",
                "expected_category": "REAL_TIME", 
                "context": {"return_legacy_format": True}
            },
            {
                "query": "How to perform NMR-based metabolomics data analysis?",
                "expected_category": "GENERAL",
                "context": {"source": "tutorial_request"}
            }
        ]
    
    async def run_production_demo(self) -> None:
        """Run complete production demonstration."""
        
        print("=" * 100)
        print("🏭 PRODUCTION LLM CLASSIFICATION SERVICE DEMONSTRATION")
        print("Clinical Metabolomics Oracle - Production Integration")
        print("=" * 100)
        print()
        
        # Initialize service
        try:
            await self.service.initialize(
                environment="production",
                use_case="high_volume"
            )
            print("✅ Service initialized successfully")
        except Exception as e:
            print(f"❌ Service initialization failed: {e}")
            return
        
        # Add sample alert handler
        async def sample_alert_handler(alert_data):
            print(f"🚨 ALERT: {alert_data['type']} - {alert_data}")
        
        self.service.add_alert_handler(sample_alert_handler)
        
        print()
        print("🧪 Running Production Classification Tests...")
        print("-" * 70)
        
        # Process test queries
        results = []
        for i, test_case in enumerate(self.test_queries, 1):
            query = test_case["query"]
            context = test_case["context"]
            
            print(f"\nTest {i}: {test_case['expected_category']} Query")
            print(f"Query: \"{query[:80]}{'...' if len(query) > 80 else ''}\"")
            
            try:
                # Classify with monitoring
                result, health_metrics = await self.service.classify_query_with_monitoring(
                    query_text=query,
                    context=context,
                    priority="normal"
                )
                
                # Display results
                if result.get("error"):
                    print(f"❌ Error: {result['error']}")
                else:
                    classification = result["classification_result"]
                    metadata = result["metadata"]
                    
                    print(f"✅ Classification: {classification['category']} (conf: {classification['confidence']:.3f})")
                    print(f"   Source: {'🤖 LLM' if metadata.get('used_llm') else '⚡ Cache' if metadata.get('used_cache') else '🔄 Fallback'}")
                    print(f"   Response Time: {result['processing_time_ms']:.1f}ms")
                    print(f"   Cost Estimate: ${metadata.get('cost_estimate', 0):.6f}")
                    
                    # Show legacy compatibility if requested
                    if "legacy_routing_prediction" in result:
                        legacy = result["legacy_routing_prediction"]
                        print(f"   Legacy Format: {legacy['routing_decision']} -> {legacy['research_category']}")
                    
                    # Health metrics
                    print(f"   System Health: {health_metrics.circuit_breaker_state.upper()}, "
                          f"Cache Hit Rate: {health_metrics.cache_hit_rate:.1%}")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Classification failed: {e}")
        
        print()
        print("📊 SYSTEM STATUS AND HEALTH MONITORING")
        print("-" * 70)
        
        # Get comprehensive system status
        system_status = await self.service.get_system_status()
        
        print(f"Service Status: {system_status['status'].upper()}")
        print(f"Version: {system_status['service_info']['version']}")
        print(f"Uptime: {system_status['service_info']['uptime_seconds']:.1f}s")
        print(f"Total Requests: {system_status['service_info']['total_requests']}")
        print(f"Success Rate: {system_status['service_info']['success_rate']:.1%}")
        print()
        
        print("Performance Metrics:")
        perf = system_status['performance']
        print(f"  Average Response Time: {perf['avg_response_time_ms']:.1f}ms")
        print(f"  Target Response Time: {perf['target_response_time_ms']:.1f}ms")
        print(f"  Target Compliance: {perf['target_compliance_rate']:.1%}")
        print(f"  95th Percentile: {perf['p95_response_time_ms']:.1f}ms")
        print()
        
        print("Cost Management:")
        cost = system_status['cost_management']
        print(f"  Daily Cost: ${cost['daily_cost']:.4f}")
        print(f"  Daily Budget: ${cost['daily_budget']:.2f}")
        print(f"  Budget Utilization: {cost['budget_utilization']:.1%}")
        print(f"  Cost per Request: ${cost['cost_per_request']:.6f}")
        print()
        
        print("Reliability:")
        reliability = system_status['reliability']
        print(f"  Circuit Breaker: {reliability['circuit_breaker_state'].upper()}")
        print(f"  CB Success Rate: {reliability['circuit_breaker_success_rate']:.1%}")
        print(f"  Fallback Rate: {reliability['fallback_rate']:.1%}")
        print()
        
        print("Cache Performance:")
        cache = system_status['cache_performance']
        print(f"  Hit Rate: {cache['hit_rate']:.1%}")
        print(f"  Cache Size: {cache['cache_size']} entries")
        print(f"  Utilization: {cache['utilization']:.1%}")
        print()
        
        print("Overall Health:")
        health = system_status['health']
        print(f"  Health Status: {health['overall_health'].upper()}")
        print(f"  Health Score: {health['health_score']:.1f}/100")
        print(f"  Pending Recommendations: {health['recommendations_count']}")
        print()
        
        # Performance summary
        print("🎯 PRODUCTION READINESS ASSESSMENT")
        print("-" * 70)
        
        readiness_score = 0
        max_score = 7
        
        # Response time check
        if perf['avg_response_time_ms'] < 2000:
            print("✅ Response Time: Target <2s achieved")
            readiness_score += 1
        else:
            print("⚠️  Response Time: Exceeds <2s target")
        
        # Success rate check
        if system_status['service_info']['success_rate'] > 0.95:
            print("✅ Reliability: >95% success rate")
            readiness_score += 1
        else:
            print("⚠️  Reliability: <95% success rate")
        
        # Cost management check
        if cost['budget_utilization'] < 0.9:
            print("✅ Cost Management: Within budget limits")
            readiness_score += 1
        else:
            print("⚠️  Cost Management: High budget utilization")
        
        # Cache performance check
        if cache['hit_rate'] > 0.3:
            print("✅ Cache Performance: Good hit rate")
            readiness_score += 1
        else:
            print("⚠️  Cache Performance: Low hit rate")
        
        # Circuit breaker check
        if reliability['circuit_breaker_state'] == 'closed':
            print("✅ Circuit Breaker: Healthy state")
            readiness_score += 1
        else:
            print("⚠️  Circuit Breaker: Not in closed state")
        
        # Health score check
        if health['health_score'] > 70:
            print("✅ Overall Health: Good system health")
            readiness_score += 1
        else:
            print("⚠️  Overall Health: Needs attention")
        
        # Integration check
        if EXISTING_INFRASTRUCTURE_AVAILABLE:
            print("✅ Integration: Compatible with existing infrastructure")
            readiness_score += 1
        else:
            print("⚠️  Integration: Limited infrastructure available")
        
        print()
        print(f"Production Readiness Score: {readiness_score}/{max_score} ({readiness_score/max_score*100:.1f}%)")
        
        if readiness_score >= 6:
            print("🚀 System is READY for production deployment")
        elif readiness_score >= 4:
            print("⚠️  System needs minor adjustments before production")
        else:
            print("❌ System needs significant improvements before production")
        
        print()
        
        # Shutdown service
        await self.service.shutdown()


async def main():
    """Main production demonstration function."""
    
    print("Initializing Production LLM Classification Service Demo...")
    print()
    
    # Check environment
    env_status = validate_environment_setup()
    if not env_status["ready"]:
        print("❌ Environment not ready:")
        for rec in env_status["recommendations"]:
            print(f"   - {rec}")
        print("\nPlease fix environment issues and try again.")
        return
    
    try:
        # Run production demo
        demo = ProductionDemo()
        await demo.run_production_demo()
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        
    except Exception as e:
        logger.error(f"Production demo failed: {e}")
        print(f"\n❌ Demo failed: {str(e)}")
        print("Check production_llm_classifier.log for detailed error information")
    
    print("\n🏁 Production demonstration completed")


if __name__ == "__main__":
    # Set up environment for demo
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Set OPENAI_API_KEY environment variable to run with actual API calls")
        print("   The demo will show limited functionality without an API key")
        print()
    
    # Run the production demonstration
    asyncio.run(main())