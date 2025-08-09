#!/usr/bin/env python3
"""
Comprehensive Error Recovery System Demo for Clinical Metabolomics Oracle

This demonstration script showcases the complete error recovery and retry logic system,
including all components working together in realistic scenarios.

Demo Scenarios:
1. Basic retry operations with different backoff strategies
2. Integration with existing Clinical Metabolomics Oracle components
3. Configuration management and dynamic updates
4. Error recovery under various failure conditions
5. Performance monitoring and metrics collection
6. Advanced recovery system integration
7. Circuit breaker coordination
8. State persistence and recovery across restarts

Features Demonstrated:
    - Automatic retry with intelligent backoff
    - Error classification and recovery strategy selection
    - State persistence for resumable operations
    - Monitoring and metrics collection
    - Configuration-driven behavior
    - Integration decorators and context managers
    - Async operation support
    - Circuit breaker coordination
    - Graceful degradation integration

Author: Claude Code (Anthropic)
Created: 2025-08-09
Version: 1.0.0
Task: CMO-LIGHTRAG-014-T06
"""

import asyncio
import json
import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Import error recovery system components
from .comprehensive_error_recovery_system import (
    create_error_recovery_orchestrator, ErrorRecoveryRule, RetryStrategy,
    ErrorSeverity, RecoveryAction
)
from .error_recovery_config import (
    create_error_recovery_config_manager, ConfigurationProfile
)
from .error_recovery_integration import (
    initialize_error_recovery_system, retry_on_error, error_recovery_context,
    execute_with_retry, execute_async_with_retry, get_error_recovery_status,
    shutdown_error_recovery_system, ClinicalMetabolomicsErrorRecoveryMixin
)

# Mock existing Clinical Metabolomics Oracle components
class MockClinicalMetabolomicsRAG(ClinicalMetabolomicsErrorRecoveryMixin):
    """Mock Clinical Metabolomics RAG system with error recovery."""
    
    def __init__(self, failure_rate: float = 0.3):
        """Initialize mock RAG system."""
        super().__init__()
        self.failure_rate = failure_rate
        self.query_count = 0
        self.logger = logging.getLogger(f"{__name__}.MockRAG")
    
    @retry_on_error("rag_query", max_attempts=3, auto_retry=True)
    def query(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        """Execute query with error recovery."""
        self.query_count += 1
        
        # Simulate various types of failures
        if random.random() < self.failure_rate:
            failure_type = random.choice([
                "network_error",
                "api_rate_limit", 
                "api_error",
                "processing_error"
            ])
            
            if failure_type == "network_error":
                raise ConnectionError("Network connection failed")
            elif failure_type == "api_rate_limit":
                raise Exception("API rate limit exceeded")
            elif failure_type == "api_error":
                raise Exception("500 Internal Server Error")
            else:
                raise Exception(f"Query processing error: {failure_type}")
        
        # Simulate successful response
        return {
            "query": query,
            "mode": mode,
            "results": [
                {"title": f"Clinical result {i}", "score": random.uniform(0.7, 1.0)}
                for i in range(random.randint(1, 5))
            ],
            "metadata": {
                "query_count": self.query_count,
                "processing_time": random.uniform(0.1, 2.0)
            }
        }
    
    @retry_on_error("rag_ingestion", max_attempts=5, auto_retry=True)
    def ingest_document(self, document_path: str) -> Dict[str, Any]:
        """Ingest document with error recovery."""
        # Simulate ingestion failures
        if random.random() < self.failure_rate * 0.8:  # Lower failure rate for ingestion
            failure_type = random.choice([
                "memory_error",
                "file_error",
                "processing_error",
                "api_error"
            ])
            
            if failure_type == "memory_error":
                raise MemoryError("Insufficient memory for document processing")
            elif failure_type == "file_error":
                raise FileNotFoundError(f"Document not found: {document_path}")
            elif failure_type == "api_error":
                raise Exception("Embedding API unavailable")
            else:
                raise Exception(f"Document processing failed: {failure_type}")
        
        return {
            "document_path": document_path,
            "document_id": str(uuid.uuid4()),
            "chunks_processed": random.randint(5, 50),
            "embeddings_generated": random.randint(10, 100),
            "status": "success"
        }


class MockAdvancedRecoverySystem:
    """Mock advanced recovery system for integration demo."""
    
    def __init__(self):
        """Initialize mock advanced recovery system."""
        self.degradation_mode = "optimal"
        self.logger = logging.getLogger(f"{__name__}.MockAdvancedRecovery")
    
    def handle_failure(self, failure_type: str, error_message: str, 
                      document_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle failure with degradation."""
        self.logger.info(f"Advanced recovery handling {failure_type}: {error_message}")
        
        if "memory" in error_message.lower():
            self.degradation_mode = "minimal"
            return {
                "action": "degrade_to_minimal",
                "degradation_mode": self.degradation_mode,
                "batch_size_reduction": 0.5
            }
        elif "rate limit" in error_message.lower():
            self.degradation_mode = "throttled"
            return {
                "action": "apply_throttling", 
                "degradation_mode": self.degradation_mode,
                "backoff_multiplier": 3.0
            }
        else:
            return {
                "action": "continue",
                "degradation_mode": self.degradation_mode
            }


class MockCircuitBreakerManager:
    """Mock circuit breaker manager for integration demo."""
    
    def __init__(self):
        """Initialize mock circuit breaker manager."""
        self.breakers = {}
        self.logger = logging.getLogger(f"{__name__}.MockCircuitBreaker")
    
    def get_circuit_breaker(self, name: str):
        """Get mock circuit breaker."""
        if name not in self.breakers:
            self.breakers[name] = MockCircuitBreaker(name)
        return self.breakers[name]


class MockCircuitBreaker:
    """Mock circuit breaker for demo."""
    
    def __init__(self, name: str):
        """Initialize mock circuit breaker."""
        self.name = name
        self.state = "closed"
        self.failure_count = 0
    
    def force_open(self, reason: str):
        """Force circuit breaker open."""
        self.state = "open"
        logging.getLogger(__name__).info(f"Circuit breaker {self.name} forced open: {reason}")
    
    def force_close(self, reason: str):
        """Force circuit breaker closed."""
        self.state = "closed"
        self.failure_count = 0
        logging.getLogger(__name__).info(f"Circuit breaker {self.name} forced closed: {reason}")


def setup_demo_environment() -> tuple:
    """Set up demonstration environment."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/error_recovery_demo.log')
        ]
    )
    
    # Create demo directory
    demo_dir = Path("logs/error_recovery_demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock systems
    mock_advanced_recovery = MockAdvancedRecoverySystem()
    mock_circuit_breaker = MockCircuitBreakerManager()
    
    # Initialize error recovery system
    orchestrator = initialize_error_recovery_system(
        config_file=demo_dir / "demo_config.yaml",
        profile=ConfigurationProfile.DEVELOPMENT,
        state_dir=demo_dir / "retry_states",
        advanced_recovery=mock_advanced_recovery,
        circuit_breaker_manager=mock_circuit_breaker
    )
    
    return orchestrator, mock_advanced_recovery, mock_circuit_breaker, demo_dir


def demo_basic_retry_scenarios():
    """Demonstrate basic retry scenarios."""
    print("\n" + "="*60)
    print("DEMO: Basic Retry Scenarios")
    print("="*60)
    
    # Create mock RAG system with high failure rate for demo
    rag_system = MockClinicalMetabolomicsRAG(failure_rate=0.7)
    
    # Test queries with different scenarios
    test_queries = [
        "What are the biomarkers for diabetes?",
        "Metabolic pathways in cancer research",
        "Clinical applications of metabolomics",
        "Biomarker discovery methods"
    ]
    
    successful_queries = 0
    failed_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        try:
            result = rag_system.query(query, mode="hybrid")
            print(f"✓ Success: Found {len(result['results'])} results")
            successful_queries += 1
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed_queries += 1
    
    print(f"\nBasic Retry Results:")
    print(f"Successful queries: {successful_queries}/{len(test_queries)}")
    print(f"Failed queries: {failed_queries}/{len(test_queries)}")
    print(f"Success rate: {successful_queries/len(test_queries)*100:.1f}%")


def demo_document_ingestion_with_recovery():
    """Demonstrate document ingestion with error recovery."""
    print("\n" + "="*60)
    print("DEMO: Document Ingestion with Error Recovery")
    print("="*60)
    
    rag_system = MockClinicalMetabolomicsRAG(failure_rate=0.5)
    
    # Test document ingestion
    test_documents = [
        "research_papers/diabetes_biomarkers.pdf",
        "clinical_studies/metabolomics_cancer.pdf", 
        "reviews/biomarker_discovery.pdf",
        "protocols/metabolic_analysis.pdf",
        "data/patient_metabolome_profiles.csv"
    ]
    
    successful_ingestions = 0
    failed_ingestions = 0
    
    for doc_path in test_documents:
        print(f"\nIngesting: {doc_path}")
        try:
            result = rag_system.ingest_document(doc_path)
            print(f"✓ Success: Processed {result['chunks_processed']} chunks, "
                  f"Generated {result['embeddings_generated']} embeddings")
            successful_ingestions += 1
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed_ingestions += 1
    
    print(f"\nDocument Ingestion Results:")
    print(f"Successful ingestions: {successful_ingestions}/{len(test_documents)}")
    print(f"Failed ingestions: {failed_ingestions}/{len(test_documents)}")
    print(f"Success rate: {successful_ingestions/len(test_documents)*100:.1f}%")


async def demo_async_operations():
    """Demonstrate asynchronous operations with error recovery."""
    print("\n" + "="*60)
    print("DEMO: Asynchronous Operations with Error Recovery")
    print("="*60)
    
    @retry_on_error("async_processing", max_attempts=3, auto_retry=True)
    async def async_process_batch(batch_id: str, items: List[str]) -> Dict[str, Any]:
        """Process batch asynchronously with potential failures."""
        print(f"Processing batch {batch_id} with {len(items)} items...")
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Simulate failures
        if random.random() < 0.4:
            failure_types = ["network_timeout", "processing_error", "resource_exhaustion"]
            failure = random.choice(failure_types)
            raise Exception(f"Async batch processing failed: {failure}")
        
        return {
            "batch_id": batch_id,
            "items_processed": len(items),
            "processing_time": random.uniform(1.0, 3.0),
            "status": "completed"
        }
    
    # Process multiple batches concurrently
    batches = [
        ("batch_001", ["item1", "item2", "item3"]),
        ("batch_002", ["item4", "item5"]),
        ("batch_003", ["item6", "item7", "item8", "item9"]),
        ("batch_004", ["item10"]),
        ("batch_005", ["item11", "item12", "item13"])
    ]
    
    tasks = [async_process_batch(batch_id, items) for batch_id, items in batches]
    
    successful_batches = 0
    failed_batches = 0
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        batch_id = batches[i][0]
        if isinstance(result, Exception):
            print(f"✗ {batch_id} failed: {result}")
            failed_batches += 1
        else:
            print(f"✓ {batch_id} completed: {result['items_processed']} items processed")
            successful_batches += 1
    
    print(f"\nAsync Processing Results:")
    print(f"Successful batches: {successful_batches}/{len(batches)}")
    print(f"Failed batches: {failed_batches}/{len(batches)}")
    print(f"Success rate: {successful_batches/len(batches)*100:.1f}%")


def demo_error_recovery_context():
    """Demonstrate error recovery context manager."""
    print("\n" + "="*60)
    print("DEMO: Error Recovery Context Manager")
    print("="*60)
    
    def process_clinical_data(data_type: str, sample_count: int):
        """Process clinical data with error recovery context."""
        with error_recovery_context("clinical_data_processing") as ctx:
            ctx.set_context("data_type", data_type)
            ctx.set_context("sample_count", sample_count)
            
            print(f"Processing {sample_count} {data_type} samples...")
            
            # Simulate processing steps with potential failures
            if random.random() < 0.3:
                raise Exception(f"Processing failed for {data_type} data")
            
            # Simulate processing time
            time.sleep(random.uniform(0.1, 0.3))
            
            result = {
                "data_type": data_type,
                "samples_processed": sample_count,
                "biomarkers_identified": random.randint(5, 20),
                "processing_time": random.uniform(1.0, 5.0)
            }
            
            ctx.set_result(result)
            return result
    
    # Test different data types
    data_types = [
        ("metabolomic", 150),
        ("proteomic", 75),
        ("genomic", 200),
        ("transcriptomic", 100)
    ]
    
    successful_processing = 0
    failed_processing = 0
    
    for data_type, sample_count in data_types:
        try:
            result = process_clinical_data(data_type, sample_count)
            print(f"✓ {data_type}: {result['biomarkers_identified']} biomarkers identified")
            successful_processing += 1
        except Exception as e:
            print(f"✗ {data_type}: {e}")
            if hasattr(e, 'recovery_info'):
                print(f"  Recovery info available: {e.recovery_info.get('should_retry', False)}")
            failed_processing += 1
    
    print(f"\nContext Manager Results:")
    print(f"Successful processing: {successful_processing}/{len(data_types)}")
    print(f"Failed processing: {failed_processing}/{len(data_types)}")


def demo_configuration_management():
    """Demonstrate configuration management."""
    print("\n" + "="*60)
    print("DEMO: Configuration Management")
    print("="*60)
    
    config_manager = get_error_recovery_config_manager()
    if not config_manager:
        print("Configuration manager not available")
        return
    
    # Show current configuration
    print("Current Configuration:")
    summary = config_manager.get_configuration_summary()
    print(json.dumps(summary, indent=2, default=str))
    
    # Test dynamic configuration update
    print("\nTesting dynamic configuration update...")
    updates = {
        'retry_policy': {
            'default_max_attempts': 5,
            'default_max_delay': 120.0
        },
        'monitoring': {
            'high_failure_rate_threshold': 0.9
        }
    }
    
    if config_manager.update_configuration(updates):
        print("✓ Configuration updated successfully")
        
        # Verify updates
        new_config = config_manager.get_config()
        print(f"New max attempts: {new_config.retry_policy.default_max_attempts}")
        print(f"New max delay: {new_config.retry_policy.default_max_delay}")
        print(f"New failure threshold: {new_config.monitoring.high_failure_rate_threshold}")
    else:
        print("✗ Configuration update failed")


def demo_system_monitoring():
    """Demonstrate system monitoring and metrics."""
    print("\n" + "="*60)
    print("DEMO: System Monitoring and Metrics")
    print("="*60)
    
    # Get comprehensive system status
    status = get_error_recovery_status()
    
    print("System Status:")
    print(f"Orchestrator available: {status['orchestrator_available']}")
    print(f"Config manager available: {status['config_manager_available']}")
    
    if status['orchestrator_available']:
        orchestrator_stats = status['orchestrator_status']
        print(f"\nOrchestrator Statistics:")
        print(f"Operations handled: {orchestrator_stats['orchestrator_statistics']['operations_handled']}")
        print(f"Successful recoveries: {orchestrator_stats['orchestrator_statistics']['successful_recoveries']}")
        print(f"Failed recoveries: {orchestrator_stats['orchestrator_statistics']['failed_recoveries']}")
        print(f"Active retry states: {len(orchestrator_stats['active_retry_states'])}")
        print(f"Recovery rules: {orchestrator_stats['recovery_rules_count']}")
        
        # Show retry metrics if available
        if 'retry_metrics' in orchestrator_stats:
            metrics = orchestrator_stats['retry_metrics']
            print(f"\nRetry Metrics:")
            recent = metrics.get('recent_metrics', {})
            print(f"Recent attempts: {recent.get('total_attempts', 0)}")
            print(f"Recent successes: {recent.get('successful_attempts', 0)}")
            print(f"Success rate: {recent.get('success_rate', 0)*100:.1f}%")


def demo_integration_scenarios():
    """Demonstrate various integration scenarios."""
    print("\n" + "="*60) 
    print("DEMO: Integration Scenarios")
    print("="*60)
    
    # Test utility functions
    print("Testing utility functions:")
    
    def failing_operation(**kwargs):
        """Operation that fails randomly."""
        if random.random() < 0.6:
            raise Exception(f"Utility test failure with {kwargs}")
        return f"Utility success with {kwargs}"
    
    try:
        result = execute_with_retry(
            operation=failing_operation,
            operation_type="utility_test",
            max_attempts=4,
            test_param="demo_value"
        )
        print(f"✓ Utility function succeeded: {result}")
    except Exception as e:
        print(f"✗ Utility function failed: {e}")
    
    # Test async utility
    async def test_async_utility():
        async def async_failing_operation(**kwargs):
            if random.random() < 0.6:
                raise Exception(f"Async utility test failure with {kwargs}")
            return f"Async utility success with {kwargs}"
        
        try:
            result = await execute_async_with_retry(
                operation=async_failing_operation,
                operation_type="async_utility_test",
                max_attempts=4,
                test_param="async_demo_value"
            )
            print(f"✓ Async utility succeeded: {result}")
        except Exception as e:
            print(f"✗ Async utility failed: {e}")
    
    asyncio.run(test_async_utility())


def demo_error_pattern_analysis():
    """Demonstrate error pattern analysis and adaptive behavior."""
    print("\n" + "="*60)
    print("DEMO: Error Pattern Analysis and Adaptive Behavior")
    print("="*60)
    
    # Simulate operation with different error patterns
    operation_id = "pattern_analysis_demo"
    orchestrator = get_error_recovery_orchestrator()
    
    if not orchestrator:
        print("Orchestrator not available for pattern analysis demo")
        return
    
    # Pattern 1: Alternating errors
    print("Simulating alternating error pattern...")
    error_types = ["NetworkError", "APIError"]
    
    for i in range(6):
        error_type = error_types[i % 2]
        error = Exception(f"{error_type}: Pattern test {i+1}")
        
        recovery_result = orchestrator.handle_operation_error(
            operation_id=f"{operation_id}_alternating_{i}",
            error=error,
            operation_type="pattern_analysis",
            operation_context={"pattern": "alternating", "iteration": i}
        )
        
        should_retry = recovery_result.get('should_retry', False)
        print(f"  Iteration {i+1} ({error_type}): {'Retry' if should_retry else 'Stop'}")
    
    # Pattern 2: Escalating errors  
    print("\nSimulating escalating error severity...")
    for i in range(4):
        if i < 2:
            error = Exception("Minor processing error")
        else:
            error = MemoryError("Critical memory exhaustion")
            
        recovery_result = orchestrator.handle_operation_error(
            operation_id=f"{operation_id}_escalating_{i}",
            error=error,
            operation_type="pattern_analysis",
            operation_context={"pattern": "escalating", "iteration": i}
        )
        
        actions = recovery_result.get('recovery_actions_taken', [])
        action_summary = ", ".join([action.get('action', 'unknown') for action in actions])
        print(f"  Iteration {i+1}: Error severity {'minor' if i < 2 else 'critical'}, Actions: {action_summary}")


def demo_performance_under_load():
    """Demonstrate system performance under load."""
    print("\n" + "="*60)
    print("DEMO: Performance Under Load")
    print("="*60)
    
    orchestrator = get_error_recovery_orchestrator()
    if not orchestrator:
        print("Orchestrator not available for performance demo")
        return
    
    # Simulate high load scenario
    num_operations = 50
    start_time = time.time()
    
    print(f"Processing {num_operations} operations concurrently...")
    
    import concurrent.futures
    
    def simulate_operation(op_id: int):
        """Simulate operation with random failures."""
        error_types = [
            "NetworkError: Connection timeout",
            "APIError: Service unavailable", 
            "ProcessingError: Invalid data format",
            "RateLimitError: Too many requests"
        ]
        
        error = Exception(random.choice(error_types))
        
        return orchestrator.handle_operation_error(
            operation_id=f"load_test_{op_id}",
            error=error,
            operation_type="load_test",
            operation_context={"load_test": True, "operation_id": op_id}
        )
    
    # Process operations concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(simulate_operation, i) for i in range(num_operations)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Analyze results
    retry_count = sum(1 for result in results if result.get('should_retry', False))
    
    print(f"\nPerformance Results:")
    print(f"Total operations: {num_operations}")
    print(f"Processing time: {total_time:.2f} seconds")
    print(f"Operations per second: {num_operations/total_time:.1f}")
    print(f"Operations eligible for retry: {retry_count}")
    print(f"Retry rate: {retry_count/num_operations*100:.1f}%")
    
    # Show final system status
    status = orchestrator.get_system_status()
    print(f"Total operations handled by orchestrator: {status['orchestrator_statistics']['operations_handled']}")


async def run_comprehensive_demo():
    """Run comprehensive demonstration of error recovery system."""
    print("🚀 Starting Comprehensive Error Recovery System Demo")
    print("=" * 80)
    
    try:
        # Setup
        orchestrator, mock_advanced_recovery, mock_circuit_breaker, demo_dir = setup_demo_environment()
        
        print(f"Demo environment initialized in: {demo_dir}")
        print(f"Orchestrator initialized: {orchestrator is not None}")
        
        # Run demonstration scenarios
        demo_basic_retry_scenarios()
        demo_document_ingestion_with_recovery()
        await demo_async_operations()
        demo_error_recovery_context()
        demo_configuration_management()
        demo_system_monitoring()
        demo_integration_scenarios()
        demo_error_pattern_analysis()
        demo_performance_under_load()
        
        # Final system cleanup and summary
        print("\n" + "="*60)
        print("DEMO COMPLETION SUMMARY")
        print("="*60)
        
        # Get final status
        final_status = get_error_recovery_status()
        if final_status['orchestrator_available']:
            orchestrator_stats = final_status['orchestrator_status']['orchestrator_statistics']
            print(f"Total operations processed: {orchestrator_stats['operations_handled']}")
            print(f"Successful recoveries: {orchestrator_stats['successful_recoveries']}")
            print(f"Failed recoveries: {orchestrator_stats['failed_recoveries']}")
            
            if orchestrator_stats['operations_handled'] > 0:
                recovery_rate = (orchestrator_stats['successful_recoveries'] / 
                               orchestrator_stats['operations_handled'] * 100)
                print(f"Recovery success rate: {recovery_rate:.1f}%")
        
        print(f"\nDemo completed successfully! 🎉")
        print(f"Check logs in: {demo_dir}")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nShutting down error recovery system...")
        shutdown_error_recovery_system()
        print("Demo cleanup completed.")


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run comprehensive demo
    asyncio.run(run_comprehensive_demo())