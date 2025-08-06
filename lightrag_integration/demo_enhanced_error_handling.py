#!/usr/bin/env python3
"""
Demonstration of enhanced error handling features in ClinicalMetabolomicsRAG.

This script shows how the enhanced error handling works in practice,
including rate limiting, circuit breakers, retry logic with jitter,
and comprehensive monitoring.

Usage:
    python lightrag_integration/demo_enhanced_error_handling.py

Author: Claude Code (Anthropic)
Created: 2025-08-06
"""

import asyncio
import tempfile
import time
from pathlib import Path

from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import (
    ClinicalMetabolomicsRAG,
    CircuitBreaker,
    RateLimiter,
    RequestQueue,
    add_jitter
)


async def demo_circuit_breaker():
    """Demonstrate circuit breaker functionality."""
    print("=== Circuit Breaker Demo ===")
    
    # Create a circuit breaker with low threshold for demo
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0, expected_exception=ValueError)
    
    async def sometimes_failing_function(should_fail=True):
        """Function that can be configured to fail or succeed."""
        if should_fail:
            raise ValueError("Simulated API failure")
        return "Success!"
    
    print("Testing circuit breaker with failing function...")
    
    # Fail a few times to trip the circuit breaker
    for i in range(5):
        try:
            result = await cb.call(sometimes_failing_function, should_fail=True)
            print(f"Call {i+1}: {result}")
        except ValueError as e:
            print(f"Call {i+1}: Failed with {e}")
        except Exception as e:
            print(f"Call {i+1}: Circuit breaker error: {e}")
        
        print(f"  Circuit state: {cb.state}, Failures: {cb.failure_count}")
    
    print("\nWaiting for circuit breaker recovery...")
    await asyncio.sleep(3)  # Wait for recovery timeout
    
    print("Testing with successful function after recovery...")
    try:
        result = await cb.call(sometimes_failing_function, should_fail=False)
        print(f"Recovery call: {result}")
        print(f"  Circuit state: {cb.state}, Failures: {cb.failure_count}")
    except Exception as e:
        print(f"Recovery call failed: {e}")


async def demo_rate_limiter():
    """Demonstrate rate limiter functionality."""
    print("\n=== Rate Limiter Demo ===")
    
    # Create rate limiter with very low limits for demo
    limiter = RateLimiter(max_requests=3, time_window=5.0)  # 3 requests per 5 seconds
    
    print("Testing rate limiter (3 requests per 5 seconds)...")
    
    for i in range(6):
        start_time = time.time()
        
        if await limiter.acquire():
            elapsed = time.time() - start_time
            print(f"Request {i+1}: Allowed (waited {elapsed:.2f}s), tokens remaining: {limiter.tokens:.1f}")
        else:
            print(f"Request {i+1}: Rate limited, tokens: {limiter.tokens:.1f}")
            print("  Waiting for token...")
            await limiter.wait_for_token()
            elapsed = time.time() - start_time
            print(f"  Got token after {elapsed:.2f}s")


async def demo_request_queue():
    """Demonstrate request queue functionality."""
    print("\n=== Request Queue Demo ===")
    
    # Create request queue with low concurrency limit for demo
    queue = RequestQueue(max_concurrent=2)
    
    async def slow_task(task_id, duration=1.0):
        """Simulate a slow API call."""
        print(f"  Task {task_id}: Starting (active: {queue.active_requests})")
        await asyncio.sleep(duration)
        print(f"  Task {task_id}: Completed")
        return f"Result {task_id}"
    
    print("Starting 5 concurrent tasks (max 2 concurrent allowed)...")
    
    # Start 5 tasks concurrently
    tasks = []
    for i in range(5):
        task = asyncio.create_task(queue.execute(slow_task, i+1, 0.5))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    print(f"All tasks completed: {results}")
    print(f"Final active requests: {queue.active_requests}")


def demo_jitter():
    """Demonstrate jitter functionality."""
    print("\n=== Jitter Demo ===")
    
    base_wait_time = 2.0
    print(f"Base wait time: {base_wait_time}s")
    
    print("Wait times with jitter (10 samples):")
    for i in range(10):
        jittered_time = add_jitter(base_wait_time, jitter_factor=0.2)
        print(f"  Sample {i+1}: {jittered_time:.3f}s")


async def demo_rag_error_handling():
    """Demonstrate RAG system error handling."""
    print("\n=== RAG Error Handling Demo ===")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LightRAGConfig(
            working_dir=Path(temp_dir),
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            max_tokens=4096,
            api_key="demo-api-key-not-real"
        )
        
        # Initialize RAG with custom error handling configuration
        rag = ClinicalMetabolomicsRAG(
            config=config,
            circuit_breaker={
                'failure_threshold': 3,
                'recovery_timeout': 10.0
            },
            rate_limiter={
                'requests_per_minute': 30,
                'max_concurrent_requests': 3
            },
            retry_config={
                'max_attempts': 3,
                'backoff_factor': 2,
                'max_wait_time': 30
            }
        )
        
        print("RAG system initialized with enhanced error handling")
        print(f"  Circuit breaker threshold: {rag.llm_circuit_breaker.failure_threshold}")
        print(f"  Rate limit: {rag.rate_limiter.max_requests} requests/min")
        print(f"  Max concurrent: {rag.request_queue.max_concurrent}")
        print(f"  Retry attempts: {rag.retry_config['max_attempts']}")
        
        # Display initial metrics
        metrics = rag.get_error_metrics()
        print("\nInitial error metrics:")
        print(f"  System health: {metrics['health_indicators']['is_healthy']}")
        print(f"  Circuit breakers: {metrics['circuit_breaker_status']['llm_circuit_state']}")
        print(f"  Rate limit tokens: {metrics['rate_limiting']['current_tokens']:.1f}")
        print(f"  API calls: {metrics['api_performance']['total_calls']}")
        
        # Simulate some metrics (in real use, these would come from actual API calls)
        print("\nSimulating error scenarios...")
        rag.error_metrics['rate_limit_events'] = 2
        rag.error_metrics['retry_attempts'] = 5
        rag.error_metrics['api_call_stats']['total_calls'] = 25
        rag.error_metrics['api_call_stats']['successful_calls'] = 22
        rag.error_metrics['api_call_stats']['failed_calls'] = 3
        rag._update_average_response_time(0.75)
        
        # Display updated metrics
        metrics = rag.get_error_metrics()
        print("\nSimulated error metrics:")
        print(f"  System health: {metrics['health_indicators']['is_healthy']}")
        print(f"  Success rate: {metrics['api_performance']['success_rate']:.2%}")
        print(f"  Rate limit events: {metrics['error_counts']['rate_limit_events']}")
        print(f"  Retry attempts: {metrics['error_counts']['retry_attempts']}")
        print(f"  Avg response time: {metrics['api_performance']['average_response_time']:.2f}s")
        
        # Test metrics reset
        print("\nResetting error metrics...")
        rag.reset_error_metrics()
        
        metrics = rag.get_error_metrics()
        print(f"  After reset - Total calls: {metrics['api_performance']['total_calls']}")
        print(f"  After reset - Rate limit events: {metrics['error_counts']['rate_limit_events']}")


async def main():
    """Run all demonstrations."""
    print("Enhanced Error Handling Demonstration")
    print("====================================")
    
    await demo_circuit_breaker()
    await demo_rate_limiter()
    await demo_request_queue()
    demo_jitter()
    await demo_rag_error_handling()
    
    print("\n=== Demo Complete ===")
    print("All error handling components demonstrated successfully!")
    print("\nKey features shown:")
    print("✓ Circuit breaker pattern with automatic recovery")
    print("✓ Token bucket rate limiting with queue waiting")
    print("✓ Request queue with concurrency control")
    print("✓ Retry jitter to prevent thundering herd problems")
    print("✓ Comprehensive error metrics and health monitoring")
    print("✓ RAG system integration with all error handling patterns")


if __name__ == "__main__":
    asyncio.run(main())