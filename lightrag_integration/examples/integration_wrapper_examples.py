#!/usr/bin/env python3
"""
Integration Wrapper Examples: Comprehensive usage examples for the enhanced integration wrapper.

This module demonstrates various usage patterns for the IntegratedQueryService
and related components, including factory functions, circuit breaker patterns,
A/B testing, health monitoring, and backward compatibility.

Examples included:
1. Basic Integration Setup
2. Production Service with Quality Assessment
3. A/B Testing and Performance Comparison
4. Circuit Breaker and Health Monitoring
5. Backward Compatibility Patterns
6. Advanced Configuration and Monitoring

Author: Claude Code (Anthropic)
Created: 2025-08-08
Version: 1.0.0
"""

import asyncio
import logging
import os
from typing import Dict, Any

# Set up the Python path to find the lightrag_integration module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LightRAGConfig
from integration_wrapper import (
    IntegratedQueryService, QueryRequest, ServiceResponse, QualityMetric,
    create_integrated_service, create_production_service, create_service_with_fallback,
    managed_query_service
)


def setup_example_logging() -> logging.Logger:
    """Set up logging for examples."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("integration_examples")


def create_example_config() -> LightRAGConfig:
    """Create example LightRAG configuration."""
    
    # Set required environment variables for the example
    os.environ.setdefault('OPENAI_API_KEY', 'your-openai-api-key-here')
    os.environ.setdefault('PERPLEXITY_API_KEY', 'your-perplexity-api-key-here')
    
    # Configure feature flags for testing
    config = LightRAGConfig.get_config()
    config.lightrag_integration_enabled = True
    config.lightrag_rollout_percentage = 50.0  # 50% rollout
    config.lightrag_enable_ab_testing = True
    config.lightrag_enable_circuit_breaker = True
    config.lightrag_enable_quality_metrics = True
    config.lightrag_fallback_to_perplexity = True
    
    return config


def example_quality_assessor(response: ServiceResponse) -> Dict[QualityMetric, float]:
    """Example quality assessment function."""
    
    # This is a simplified quality assessor for demonstration
    # In production, you'd implement more sophisticated quality metrics
    
    quality_scores = {}
    
    # Assess response length (longer responses might be more comprehensive)
    content_length = len(response.content)
    if content_length > 500:
        quality_scores[QualityMetric.COMPLETENESS] = 0.9
    elif content_length > 200:
        quality_scores[QualityMetric.COMPLETENESS] = 0.7
    else:
        quality_scores[QualityMetric.COMPLETENESS] = 0.5
    
    # Assess response time (faster is better, up to a point)
    if response.processing_time < 2.0:
        quality_scores[QualityMetric.RESPONSE_TIME] = 0.9
    elif response.processing_time < 5.0:
        quality_scores[QualityMetric.RESPONSE_TIME] = 0.7
    else:
        quality_scores[QualityMetric.RESPONSE_TIME] = 0.5
    
    # Assess citation quality (if citations are present)
    if response.citations and len(response.citations) > 0:
        quality_scores[QualityMetric.CITATION_QUALITY] = 0.8
    else:
        quality_scores[QualityMetric.CITATION_QUALITY] = 0.6
    
    # Basic relevance assessment (simplified - check for scientific terms)
    scientific_terms = ['metabolomics', 'biomarker', 'pathway', 'metabolism', 'clinical']
    relevance_score = sum(1 for term in scientific_terms if term.lower() in response.content.lower()) / len(scientific_terms)
    quality_scores[QualityMetric.RELEVANCE] = min(0.9, 0.5 + relevance_score)
    
    return quality_scores


async def example_1_basic_integration(logger: logging.Logger):
    """Example 1: Basic integration setup with default settings."""
    
    logger.info("=== Example 1: Basic Integration Setup ===")
    
    config = create_example_config()
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY', 'demo-key')
    
    # Create integrated service using factory function
    service = create_integrated_service(
        config=config,
        perplexity_api_key=perplexity_api_key,
        logger=logger
    )
    
    # Create a test query
    query = QueryRequest(
        query_text="What are the main biomarkers for metabolic syndrome?",
        user_id="example_user_1",
        session_id="session_123",
        query_type="biomarker_query"
    )
    
    try:
        # Execute query
        response = await service.query_async(query)
        
        logger.info(f"Query successful: {response.is_success}")
        logger.info(f"Response type: {response.response_type.value}")
        logger.info(f"Processing time: {response.processing_time:.2f}s")
        logger.info(f"Content length: {len(response.content)} characters")
        
        if response.metadata:
            logger.info(f"Routing decision: {response.metadata.get('routing_decision')}")
            logger.info(f"Routing reason: {response.metadata.get('routing_reason')}")
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
    
    finally:
        await service.shutdown()


async def example_2_production_service(logger: logging.Logger):
    """Example 2: Production service with quality assessment."""
    
    logger.info("=== Example 2: Production Service with Quality Assessment ===")
    
    config = create_example_config()
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY', 'demo-key')
    
    # Create production service with quality assessor
    service = create_production_service(
        config=config,
        perplexity_api_key=perplexity_api_key,
        quality_assessor=example_quality_assessor,
        logger=logger
    )
    
    # Test multiple queries to see quality assessment in action
    queries = [
        QueryRequest(
            query_text="Explain the role of glucose metabolism in diabetes.",
            user_id="prod_user_1",
            query_type="metabolism_query"
        ),
        QueryRequest(
            query_text="What is metabolomics?",
            user_id="prod_user_2", 
            query_type="definition_query"
        ),
        QueryRequest(
            query_text="Compare lipidomics and metabolomics approaches in clinical research.",
            user_id="prod_user_3",
            query_type="comparison_query"
        )
    ]
    
    try:
        for i, query in enumerate(queries, 1):
            logger.info(f"\nExecuting query {i}: {query.query_text[:50]}...")
            
            response = await service.query_async(query)
            
            logger.info(f"Success: {response.is_success}")
            logger.info(f"Average quality score: {response.average_quality_score:.2f}")
            
            if response.quality_scores:
                for metric, score in response.quality_scores.items():
                    logger.info(f"  {metric.value}: {score:.2f}")
        
        # Get performance summary
        summary = service.get_performance_summary()
        logger.info("\nPerformance Summary:")
        logger.info(f"A/B Testing Metrics: {summary.get('ab_testing', {})}")
        
    except Exception as e:
        logger.error(f"Production service error: {e}")
    
    finally:
        await service.shutdown()


async def example_3_ab_testing_demo(logger: logging.Logger):
    """Example 3: A/B testing and performance comparison."""
    
    logger.info("=== Example 3: A/B Testing and Performance Comparison ===")
    
    config = create_example_config()
    config.lightrag_enable_ab_testing = True
    config.lightrag_rollout_percentage = 100.0  # Enable for all users for demo
    
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY', 'demo-key')
    
    service = create_service_with_fallback(
        lightrag_config=config,
        perplexity_api_key=perplexity_api_key,
        enable_ab_testing=True,
        logger=logger
    )
    
    # Simulate multiple users to see A/B distribution
    users = [f"ab_test_user_{i}" for i in range(10)]
    
    try:
        for user_id in users:
            query = QueryRequest(
                query_text="What are the latest advances in clinical metabolomics?",
                user_id=user_id,
                session_id=f"session_{user_id}",
                query_type="research_query"
            )
            
            response = await service.query_async(query)
            routing_decision = response.metadata.get('routing_decision', 'unknown')
            user_cohort = response.metadata.get('user_cohort', 'unknown')
            
            logger.info(f"User {user_id}: {routing_decision} (cohort: {user_cohort})")
        
        # Get A/B testing metrics
        ab_metrics = service.get_ab_test_metrics()
        logger.info("\nA/B Testing Results:")
        
        for service_name, metrics in ab_metrics.items():
            logger.info(f"\n{service_name.upper()}:")
            logger.info(f"  Sample size: {metrics.get('sample_size', 0)}")
            logger.info(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
            logger.info(f"  Avg response time: {metrics.get('avg_response_time', 0):.2f}s")
            logger.info(f"  Avg quality score: {metrics.get('avg_quality_score', 0):.2f}")
        
    except Exception as e:
        logger.error(f"A/B testing demo error: {e}")
    
    finally:
        await service.shutdown()


async def example_4_circuit_breaker_demo(logger: logging.Logger):
    """Example 4: Circuit breaker and health monitoring demonstration."""
    
    logger.info("=== Example 4: Circuit Breaker and Health Monitoring ===")
    
    config = create_example_config()
    config.lightrag_enable_circuit_breaker = True
    config.lightrag_circuit_breaker_failure_threshold = 2  # Low threshold for demo
    config.lightrag_circuit_breaker_recovery_timeout = 5.0  # Short timeout for demo
    
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY', 'demo-key')
    
    service = create_integrated_service(
        config=config,
        perplexity_api_key=perplexity_api_key,
        logger=logger
    )
    
    # Simulate a scenario that might trigger circuit breaker
    queries = [
        QueryRequest(query_text="Valid query about metabolomics", user_id="cb_user_1"),
        QueryRequest(query_text="", user_id="cb_user_2"),  # Empty query might fail
        QueryRequest(query_text="Another valid query", user_id="cb_user_3"),
    ]
    
    try:
        for i, query in enumerate(queries, 1):
            logger.info(f"\nExecuting query {i}...")
            
            response = await service.query_async(query)
            
            logger.info(f"Success: {response.is_success}")
            logger.info(f"Response type: {response.response_type.value}")
            
            if response.error_details:
                logger.info(f"Error: {response.error_details}")
        
        # Check circuit breaker status
        summary = service.get_performance_summary()
        for service_name, service_info in summary.get('services', {}).items():
            cb_state = service_info.get('circuit_breaker')
            if cb_state:
                logger.info(f"\n{service_name} Circuit Breaker Status:")
                logger.info(f"  Is open: {cb_state.get('is_open')}")
                logger.info(f"  Failure count: {cb_state.get('failure_count')}")
                logger.info(f"  Recovery attempts: {cb_state.get('recovery_attempts')}")
        
        # Health monitoring status
        health_status = summary.get('health_monitoring', {})
        logger.info("\nHealth Monitoring Status:")
        for service_name, health_info in health_status.items():
            logger.info(f"  {service_name}: {'Healthy' if health_info.get('is_healthy') else 'Unhealthy'}")
            logger.info(f"    Consecutive failures: {health_info.get('consecutive_failures', 0)}")
            logger.info(f"    Total checks: {health_info.get('total_checks', 0)}")
        
    except Exception as e:
        logger.error(f"Circuit breaker demo error: {e}")
    
    finally:
        await service.shutdown()


async def example_5_backward_compatibility(logger: logging.Logger):
    """Example 5: Backward compatibility patterns."""
    
    logger.info("=== Example 5: Backward Compatibility Patterns ===")
    
    config = create_example_config()
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY', 'demo-key')
    
    # Use backward compatibility factory
    service = create_service_with_fallback(
        lightrag_config=config,
        perplexity_api_key=perplexity_api_key,
        enable_ab_testing=False,  # Disable advanced features for compatibility
        logger=logger
    )
    
    # Simple query that would work with legacy systems
    query = QueryRequest(
        query_text="What is the importance of metabolomics in precision medicine?",
        user_id="legacy_user",
        session_id="legacy_session"
    )
    
    try:
        response = await service.query_async(query)
        
        logger.info("Backward compatibility test:")
        logger.info(f"Success: {response.is_success}")
        logger.info(f"Content available: {bool(response.content)}")
        logger.info(f"Response type: {response.response_type.value}")
        
        # Test serialization (important for backward compatibility)
        response_dict = response.to_dict()
        logger.info(f"Response serializable: {isinstance(response_dict, dict)}")
        logger.info(f"Dictionary keys: {list(response_dict.keys())}")
        
    except Exception as e:
        logger.error(f"Backward compatibility test error: {e}")
    
    finally:
        await service.shutdown()


async def example_6_managed_service_context(logger: logging.Logger):
    """Example 6: Using managed service context manager."""
    
    logger.info("=== Example 6: Managed Service Context Manager ===")
    
    config = create_example_config()
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY', 'demo-key')
    
    # Use context manager for automatic lifecycle management
    async with managed_query_service(config, perplexity_api_key, logger) as service:
        
        # Service is automatically initialized and will be shutdown when exiting context
        
        query = QueryRequest(
            query_text="How does metabolomics contribute to drug discovery?",
            user_id="context_user",
            session_id="context_session"
        )
        
        response = await service.query_async(query)
        
        logger.info("Context manager test:")
        logger.info(f"Service available: {service is not None}")
        logger.info(f"Query success: {response.is_success}")
        logger.info(f"Processing time: {response.processing_time:.2f}s")
        
        # Get comprehensive performance summary
        summary = service.get_performance_summary()
        logger.info(f"Total requests tracked: {summary.get('recent_performance', {}).get('total_requests', 0)}")
    
    # Service is automatically shutdown here
    logger.info("Service automatically shutdown by context manager")


async def run_all_examples():
    """Run all integration wrapper examples."""
    
    logger = setup_example_logging()
    
    logger.info("Starting Integration Wrapper Examples")
    logger.info("=" * 50)
    
    examples = [
        example_1_basic_integration,
        example_2_production_service,
        example_3_ab_testing_demo,
        example_4_circuit_breaker_demo,
        example_5_backward_compatibility,
        example_6_managed_service_context
    ]
    
    for example_func in examples:
        try:
            await example_func(logger)
            logger.info("\n" + "=" * 50 + "\n")
            
            # Brief pause between examples
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Example {example_func.__name__} failed: {e}")
            logger.info("\n" + "=" * 50 + "\n")
    
    logger.info("All examples completed!")


if __name__ == "__main__":
    # Run examples
    asyncio.run(run_all_examples())