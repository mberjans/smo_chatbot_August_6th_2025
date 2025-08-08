#!/usr/bin/env python3
"""
Comprehensive Feature Flag System Usage Example

This script demonstrates the complete LightRAG feature flag system functionality
including configuration, routing decisions, A/B testing, circuit breaker protection,
and performance monitoring.

The example shows how to:
1. Configure the system with different settings
2. Make routing decisions based on user context
3. Handle A/B testing scenarios
4. Monitor performance and circuit breaker status
5. Implement gradual rollout strategies
6. Handle error scenarios and recovery

Author: Claude Code (Anthropic)
Created: 2025-08-08
Version: 1.0.0
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the lightrag_integration directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lightrag_integration'))

from lightrag_integration.config import LightRAGConfig
from lightrag_integration.feature_flag_manager import (
    FeatureFlagManager, 
    RoutingContext, 
    RoutingDecision, 
    RoutingReason, 
    UserCohort
)


def setup_logger() -> logging.Logger:
    """Set up a logger for the examples."""
    logger = logging.getLogger('feature_flag_example')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def example_1_basic_configuration():
    """Example 1: Basic feature flag configuration and usage."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Feature Flag Configuration")
    print("="*70)
    
    # Create configuration for development environment
    config = LightRAGConfig(
        api_key="sk-example-development-key",
        lightrag_integration_enabled=True,
        lightrag_rollout_percentage=25.0,  # 25% rollout
        lightrag_user_hash_salt="dev_salt_2025",
        lightrag_fallback_to_perplexity=True,
        lightrag_enable_circuit_breaker=True,
        lightrag_circuit_breaker_failure_threshold=3,
        auto_create_dirs=False
    )
    
    logger = setup_logger()
    manager = FeatureFlagManager(config, logger)
    
    print(f"✓ Configuration created with {config.lightrag_rollout_percentage}% rollout")
    print(f"✓ Integration enabled: {config.lightrag_integration_enabled}")
    print(f"✓ Circuit breaker enabled: {config.lightrag_enable_circuit_breaker}")
    
    # Test routing for different users
    users = ["alice_123", "bob_456", "charlie_789", "diana_012"]
    print(f"\nTesting routing decisions for {len(users)} users:")
    
    routing_results = {}
    for user_id in users:
        context = RoutingContext(
            user_id=user_id,
            query_text="What are the latest developments in metabolomics?",
            query_type="biomedical_research",
            timestamp=datetime.now()
        )
        
        result = manager.should_use_lightrag(context)
        routing_results[user_id] = result
        
        print(f"  {user_id}: {result.decision.value} (reason: {result.reason.value})")
    
    # Show distribution
    lightrag_count = sum(1 for r in routing_results.values() if r.decision == RoutingDecision.LIGHTRAG)
    print(f"\nResult: {lightrag_count}/{len(users)} users ({lightrag_count/len(users)*100:.1f}%) routed to LightRAG")
    
    return manager


def example_2_ab_testing():
    """Example 2: A/B testing with cohort assignment."""
    print("\n" + "="*70)
    print("EXAMPLE 2: A/B Testing with Cohort Assignment")
    print("="*70)
    
    # Configuration with A/B testing enabled
    config = LightRAGConfig(
        api_key="sk-example-ab-testing-key",
        lightrag_integration_enabled=True,
        lightrag_rollout_percentage=50.0,  # 50% rollout for A/B test
        lightrag_enable_ab_testing=True,   # Enable A/B testing
        lightrag_user_hash_salt="ab_test_salt_2025",
        lightrag_enable_performance_comparison=True,
        lightrag_enable_quality_metrics=True,
        auto_create_dirs=False
    )
    
    logger = setup_logger()
    manager = FeatureFlagManager(config, logger)
    
    print(f"✓ A/B testing configuration with {config.lightrag_rollout_percentage}% rollout")
    print(f"✓ Performance comparison enabled: {config.lightrag_enable_performance_comparison}")
    
    # Simulate A/B test with multiple users
    test_users = [f"user_{i:04d}" for i in range(20)]
    cohort_assignments = {"lightrag": 0, "perplexity": 0, "control": 0}
    
    print(f"\nRunning A/B test with {len(test_users)} users:")
    
    for user_id in test_users:
        context = RoutingContext(
            user_id=user_id,
            query_text="How do metabolic pathways interact in cellular respiration?",
            query_type="biochemistry",
            timestamp=datetime.now()
        )
        
        result = manager.should_use_lightrag(context)
        
        if result.user_cohort:
            cohort_assignments[result.user_cohort.value] += 1
    
    print(f"Cohort Distribution:")
    for cohort, count in cohort_assignments.items():
        percentage = (count / len(test_users)) * 100
        print(f"  {cohort.capitalize()}: {count}/{len(test_users)} users ({percentage:.1f}%)")
    
    # Simulate performance data collection
    print(f"\nSimulating performance metrics collection...")
    
    # Record some performance metrics for both services
    manager.record_success('lightrag', 1.8, 0.87)  # 1.8s response, 87% quality
    manager.record_success('lightrag', 2.1, 0.91)
    manager.record_success('lightrag', 1.9, 0.85)
    
    manager.record_success('perplexity', 1.2, 0.78)  # Faster but lower quality
    manager.record_success('perplexity', 1.4, 0.82)
    manager.record_success('perplexity', 1.1, 0.75)
    
    # Show performance summary
    summary = manager.get_performance_summary()
    lr_perf = summary['performance']['lightrag']
    pp_perf = summary['performance']['perplexity']
    
    print(f"\nPerformance Comparison:")
    print(f"  LightRAG:  {lr_perf['avg_response_time']:.2f}s avg, {lr_perf['avg_quality_score']:.2f} quality")
    print(f"  Perplexity: {pp_perf['avg_response_time']:.2f}s avg, {pp_perf['avg_quality_score']:.2f} quality")
    
    return manager


def example_3_circuit_breaker():
    """Example 3: Circuit breaker protection and recovery."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Circuit Breaker Protection and Recovery")
    print("="*70)
    
    # Configuration with sensitive circuit breaker settings
    config = LightRAGConfig(
        api_key="sk-example-circuit-breaker-key",
        lightrag_integration_enabled=True,
        lightrag_rollout_percentage=100.0,  # Full rollout normally
        lightrag_enable_circuit_breaker=True,
        lightrag_circuit_breaker_failure_threshold=2,  # Low threshold for demo
        lightrag_circuit_breaker_recovery_timeout=5.0,  # Short timeout for demo
        auto_create_dirs=False
    )
    
    logger = setup_logger()
    manager = FeatureFlagManager(config, logger)
    
    print(f"✓ Circuit breaker configuration:")
    print(f"  - Failure threshold: {config.lightrag_circuit_breaker_failure_threshold}")
    print(f"  - Recovery timeout: {config.lightrag_circuit_breaker_recovery_timeout}s")
    
    test_user = "circuit_test_user"
    context = RoutingContext(
        user_id=test_user,
        query_text="Test query for circuit breaker demonstration",
        query_type="test",
        timestamp=datetime.now()
    )
    
    # Show normal operation
    print(f"\n1. Normal operation:")
    result1 = manager.should_use_lightrag(context)
    print(f"   Decision: {result1.decision.value} (reason: {result1.reason.value})")
    
    # Record some failures to trigger circuit breaker
    print(f"\n2. Recording failures to trigger circuit breaker...")
    manager.record_failure('lightrag', 'Simulated API timeout')
    manager.record_failure('lightrag', 'Simulated service unavailable')
    
    # Clear cache and check again
    manager.clear_caches()
    result2 = manager.should_use_lightrag(context)
    print(f"   Decision after failures: {result2.decision.value} (reason: {result2.reason.value})")
    
    # Show circuit breaker status
    cb_summary = manager.get_performance_summary()['circuit_breaker']
    print(f"   Circuit breaker status: {'OPEN' if cb_summary['is_open'] else 'CLOSED'}")
    print(f"   Failure count: {cb_summary['failure_count']}")
    
    # Wait for recovery and test again
    print(f"\n3. Waiting {config.lightrag_circuit_breaker_recovery_timeout}s for recovery...")
    time.sleep(config.lightrag_circuit_breaker_recovery_timeout + 1)
    
    manager.clear_caches()
    result3 = manager.should_use_lightrag(context)
    print(f"   Decision after recovery: {result3.decision.value} (reason: {result3.reason.value})")
    
    # Record a success to help recovery
    manager.record_success('lightrag', 1.5, 0.85)
    print(f"   Recorded successful request to aid recovery")
    
    return manager


def example_4_conditional_routing():
    """Example 4: Conditional routing based on query characteristics."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Conditional Routing Based on Query Characteristics")
    print("="*70)
    
    # Configuration with conditional routing
    routing_rules = {
        "complex_queries": {
            "type": "query_length",
            "min_length": 100,
            "max_length": 1000
        },
        "biomedical_queries": {
            "type": "query_type",
            "allowed_types": ["biomedical", "biochemistry", "metabolomics"]
        }
    }
    
    config = LightRAGConfig(
        api_key="sk-example-conditional-key",
        lightrag_integration_enabled=True,
        lightrag_rollout_percentage=100.0,  # Would normally use LightRAG
        lightrag_enable_conditional_routing=True,
        lightrag_routing_rules=routing_rules,
        auto_create_dirs=False
    )
    
    logger = setup_logger()
    manager = FeatureFlagManager(config, logger)
    
    print(f"✓ Conditional routing configuration:")
    print(f"  - Rules configured: {len(routing_rules)}")
    print(f"  - Complex query length: {routing_rules['complex_queries']['min_length']}-{routing_rules['complex_queries']['max_length']} chars")
    
    # Test different query scenarios
    test_scenarios = [
        {
            "name": "Short query",
            "context": RoutingContext(
                user_id="user_short",
                query_text="What is ATP?",  # Too short
                query_type="biochemistry",
                timestamp=datetime.now()
            ),
            "expected_routing": "Should route to Perplexity (too short)"
        },
        {
            "name": "Complex biomedical query",
            "context": RoutingContext(
                user_id="user_complex",
                query_text="How do metabolic pathways regulate cellular energy production through ATP synthesis and what are the key enzymatic control points in glycolysis and the citric acid cycle?",  # Complex enough
                query_type="biochemistry",  # Allowed type
                timestamp=datetime.now()
            ),
            "expected_routing": "Should route to LightRAG (meets both conditions)"
        },
        {
            "name": "Non-biomedical query",
            "context": RoutingContext(
                user_id="user_general",
                query_text="What are the latest developments in artificial intelligence and machine learning algorithms for natural language processing?",  # Complex but wrong type
                query_type="technology",
                timestamp=datetime.now()
            ),
            "expected_routing": "Should route to Perplexity (wrong query type)"
        }
    ]
    
    print(f"\nTesting conditional routing scenarios:")
    
    for scenario in test_scenarios:
        result = manager.should_use_lightrag(scenario["context"])
        print(f"\n  {scenario['name']}:")
        print(f"    Query length: {len(scenario['context'].query_text)} chars")
        print(f"    Query type: {scenario['context'].query_type}")
        print(f"    Decision: {result.decision.value} (reason: {result.reason.value})")
        print(f"    Expected: {scenario['expected_routing']}")
    
    return manager


def example_5_production_rollout():
    """Example 5: Production rollout simulation with monitoring."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Production Rollout Simulation with Monitoring")
    print("="*70)
    
    logger = setup_logger()
    
    # Simulate a gradual rollout over time
    rollout_stages = [
        {"name": "Canary", "percentage": 1.0, "users": 1000},
        {"name": "Small rollout", "percentage": 5.0, "users": 1000},
        {"name": "Medium rollout", "percentage": 25.0, "users": 1000},
        {"name": "Large rollout", "percentage": 75.0, "users": 1000},
        {"name": "Full rollout", "percentage": 100.0, "users": 1000},
    ]
    
    print("Simulating production rollout stages:")
    
    for stage in rollout_stages:
        print(f"\n{stage['name']} Stage: {stage['percentage']}% rollout")
        print("-" * 50)
        
        # Configure for this stage
        config = LightRAGConfig(
            api_key="sk-production-rollout-key",
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=stage["percentage"],
            lightrag_user_hash_salt="production_salt_v1",
            lightrag_enable_ab_testing=True,
            lightrag_enable_performance_comparison=True,
            lightrag_enable_quality_metrics=True,
            lightrag_enable_circuit_breaker=True,
            auto_create_dirs=False
        )
        
        manager = FeatureFlagManager(config, logger)
        
        # Simulate user requests
        lightrag_users = 0
        perplexity_users = 0
        
        for i in range(stage["users"]):
            user_id = f"prod_user_{i:05d}"
            context = RoutingContext(
                user_id=user_id,
                query_text="Production query about metabolomic analysis techniques",
                query_type="metabolomics",
                timestamp=datetime.now()
            )
            
            result = manager.should_use_lightrag(context)
            if result.decision == RoutingDecision.LIGHTRAG:
                lightrag_users += 1
            else:
                perplexity_users += 1
        
        # Calculate actual rollout percentage
        actual_percentage = (lightrag_users / stage["users"]) * 100
        
        print(f"  Target rollout: {stage['percentage']:.1f}%")
        print(f"  Actual rollout: {actual_percentage:.1f}%")
        print(f"  LightRAG users: {lightrag_users:,}")
        print(f"  Perplexity users: {perplexity_users:,}")
        print(f"  Deviation: {abs(actual_percentage - stage['percentage']):.1f}%")
        
        # Simulate some performance metrics
        if lightrag_users > 0:
            # Add some realistic performance data
            for _ in range(min(10, lightrag_users)):  # Sample of requests
                response_time = 1.5 + (0.5 * (stage["percentage"] / 100))  # Slight increase with load
                quality_score = 0.85 + (0.1 * (stage["percentage"] / 100))  # Improve with more data
                manager.record_success('lightrag', response_time, quality_score)
        
        if perplexity_users > 0:
            for _ in range(min(10, perplexity_users)):
                manager.record_success('perplexity', 1.2, 0.78)  # Baseline performance
        
        # Show performance summary
        summary = manager.get_performance_summary()
        lr_perf = summary['performance']['lightrag']
        pp_perf = summary['performance']['perplexity']
        
        if lr_perf['total_responses'] > 0:
            print(f"  LightRAG performance: {lr_perf['avg_response_time']:.2f}s, quality {lr_perf['avg_quality_score']:.2f}")
        if pp_perf['total_responses'] > 0:
            print(f"  Perplexity performance: {pp_perf['avg_response_time']:.2f}s, quality {pp_perf['avg_quality_score']:.2f}")
        
        # Short pause between stages
        time.sleep(0.1)
    
    print(f"\n✓ Production rollout simulation completed successfully!")
    return manager


def example_6_error_handling():
    """Example 6: Error handling and recovery scenarios."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Error Handling and Recovery Scenarios")
    print("="*70)
    
    config = LightRAGConfig(
        api_key="sk-example-error-handling-key",
        lightrag_integration_enabled=True,
        lightrag_rollout_percentage=100.0,
        lightrag_enable_circuit_breaker=True,
        lightrag_circuit_breaker_failure_threshold=3,
        lightrag_fallback_to_perplexity=True,
        auto_create_dirs=False
    )
    
    logger = setup_logger()
    manager = FeatureFlagManager(config, logger)
    
    print("✓ Error handling configuration enabled")
    print(f"  - Circuit breaker threshold: {config.lightrag_circuit_breaker_failure_threshold}")
    print(f"  - Fallback to Perplexity: {config.lightrag_fallback_to_perplexity}")
    
    test_user = "error_test_user"
    context = RoutingContext(
        user_id=test_user,
        query_text="Test query for error handling demonstration",
        timestamp=datetime.now()
    )
    
    # Scenario 1: Configuration errors
    print(f"\n1. Testing invalid configuration handling:")
    try:
        invalid_config = LightRAGConfig(
            api_key="",  # Invalid empty API key
            lightrag_rollout_percentage=150.0,  # Invalid percentage
            auto_create_dirs=False
        )
        # Should be corrected during __post_init__
        print(f"   ✓ Invalid percentage corrected: {invalid_config.lightrag_rollout_percentage}")
    except Exception as e:
        print(f"   Configuration error handled: {e}")
    
    # Scenario 2: Service failures and circuit breaker
    print(f"\n2. Testing service failure cascade:")
    
    # Initial state
    result1 = manager.should_use_lightrag(context)
    print(f"   Initial decision: {result1.decision.value}")
    
    # Simulate cascading failures
    failures = ["API timeout", "Service unavailable", "Rate limit exceeded"]
    for i, failure in enumerate(failures, 1):
        manager.record_failure('lightrag', failure)
        print(f"   Failure {i}: {failure}")
    
    # Check circuit breaker activation
    manager.clear_caches()
    result2 = manager.should_use_lightrag(context)
    print(f"   Decision after failures: {result2.decision.value} (reason: {result2.reason.value})")
    
    # Scenario 3: Manual recovery
    print(f"\n3. Testing manual recovery:")
    
    # Reset circuit breaker manually
    manager.reset_circuit_breaker()
    print(f"   Circuit breaker manually reset")
    
    manager.clear_caches()
    result3 = manager.should_use_lightrag(context)
    print(f"   Decision after reset: {result3.decision.value}")
    
    # Scenario 4: Performance degradation
    print(f"\n4. Testing performance degradation detection:")
    
    # Record poor performance
    manager.record_success('lightrag', 5.0, 0.40)  # Slow and low quality
    manager.record_success('lightrag', 6.0, 0.35)
    manager.record_success('lightrag', 4.8, 0.38)
    
    summary = manager.get_performance_summary()
    lr_perf = summary['performance']['lightrag']
    
    print(f"   Recent performance: {lr_perf['avg_response_time']:.2f}s, quality {lr_perf['avg_quality_score']:.2f}")
    
    if lr_perf['avg_quality_score'] < config.lightrag_min_quality_threshold:
        print(f"   ⚠️  Quality below threshold ({config.lightrag_min_quality_threshold})")
        
        # Test quality-based routing
        context_quality = RoutingContext(
            user_id="quality_test_user",
            query_text="Quality test query",
            timestamp=datetime.now()
        )
        
        # Enable quality metrics for this test
        config.lightrag_enable_quality_metrics = True
        
        result4 = manager.should_use_lightrag(context_quality)
        if result4.reason == RoutingReason.QUALITY_THRESHOLD:
            print(f"   ✓ Quality-based routing activated: {result4.decision.value}")
    
    print(f"\n✓ Error handling scenarios completed!")
    return manager


def main():
    """Run all feature flag system examples."""
    print("LightRAG Feature Flag System - Comprehensive Usage Examples")
    print("=" * 80)
    print("This script demonstrates the complete feature flag system functionality.")
    print("Each example shows different aspects of the system in action.")
    
    start_time = time.time()
    
    try:
        # Run all examples
        examples = [
            ("Basic Configuration", example_1_basic_configuration),
            ("A/B Testing", example_2_ab_testing),
            ("Circuit Breaker", example_3_circuit_breaker),
            ("Conditional Routing", example_4_conditional_routing),
            ("Production Rollout", example_5_production_rollout),
            ("Error Handling", example_6_error_handling),
        ]
        
        managers = {}
        
        for name, example_func in examples:
            print(f"\n\nRunning {name} example...")
            try:
                manager = example_func()
                managers[name] = manager
                print(f"✅ {name} example completed successfully!")
            except Exception as e:
                print(f"❌ {name} example failed: {e}")
        
        # Final summary
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("EXAMPLES SUMMARY")
        print("="*80)
        print(f"Total examples run: {len(examples)}")
        print(f"Successful examples: {len(managers)}")
        print(f"Failed examples: {len(examples) - len(managers)}")
        print(f"Total execution time: {total_time:.2f} seconds")
        
        if len(managers) == len(examples):
            print("\n🎉 All examples completed successfully!")
            print("\nKey takeaways:")
            print("• Feature flag system provides flexible, safe rollout capabilities")
            print("• Hash-based routing ensures consistent user experience")
            print("• Circuit breaker protects against service failures")
            print("• A/B testing enables performance comparison")
            print("• Conditional routing allows query-specific logic")
            print("• Comprehensive monitoring supports production operations")
        
        # Final performance summary from last manager
        if managers:
            last_manager = list(managers.values())[-1]
            summary = last_manager.get_performance_summary()
            
            print(f"\nFinal System State:")
            print(f"• Integration enabled: {summary['configuration']['integration_enabled']}")
            print(f"• Rollout percentage: {summary['configuration']['rollout_percentage']}%")
            print(f"• Circuit breaker status: {'OPEN' if summary['circuit_breaker']['is_open'] else 'CLOSED'}")
            print(f"• Total requests processed: {summary['circuit_breaker']['total_requests']}")
            
        print(f"\n📚 For more information, see:")
        print(f"• Configuration: lightrag_integration/config.py")
        print(f"• Feature flags: lightrag_integration/feature_flag_manager.py")
        print(f"• Environment variables: lightrag_integration/FEATURE_FLAG_ENVIRONMENT_VARIABLES.md")
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Examples failed with error: {e}")
    
    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()