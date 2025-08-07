#!/usr/bin/env python3
"""
Demonstration of Test Utilities Integration.

This script demonstrates how the new TestEnvironmentManager and MockSystemFactory
utilities integrate with existing test infrastructure to eliminate repetitive
patterns and streamline test development.

Usage:
    python demo_test_utilities.py

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import asyncio
import time
import pytest
from pathlib import Path

# Import the new utilities
from test_utilities import (
    TestEnvironmentManager, MockSystemFactory,
    SystemComponent, MockBehavior, EnvironmentSpec, MockSpec,
    create_quick_test_environment, async_test_context
)


def demo_basic_environment_setup():
    """Demonstrate basic environment setup with TestEnvironmentManager."""
    print("\n=== Basic Environment Setup Demo ===")
    
    # Create environment specification
    spec = EnvironmentSpec(
        temp_dirs=["logs", "pdfs", "output", "working"],
        required_imports=[
            "lightrag_integration.clinical_metabolomics_rag",
            "lightrag_integration.pdf_processor"
        ],
        async_context=True,
        performance_monitoring=True
    )
    
    # Set up environment
    env_manager = TestEnvironmentManager(spec)
    environment_data = env_manager.setup_environment()
    
    print(f"✓ Working directory: {environment_data['working_dir']}")
    print(f"✓ Subdirectories: {list(environment_data['subdirectories'].keys())}")
    print(f"✓ Successful imports: {len(environment_data['import_results']['successful'])}")
    print(f"✓ Failed imports: {len(environment_data['import_results']['failed'])}")
    print(f"✓ Setup duration: {environment_data['setup_duration']:.3f}s")
    
    # Show system health check
    health = env_manager.check_system_health()
    print(f"✓ Memory usage: {health['memory_usage_mb']:.1f} MB")
    print(f"✓ Working dir size: {health['working_dir_size_mb']:.1f} MB")
    
    # Cleanup
    env_manager.cleanup()
    print("✓ Environment cleaned up successfully")


def demo_mock_system_factory():
    """Demonstrate MockSystemFactory capabilities."""
    print("\n=== Mock System Factory Demo ===")
    
    # Set up factory
    env_manager = TestEnvironmentManager()
    env_manager.setup_environment()
    factory = MockSystemFactory(env_manager)
    
    # Create individual mocks with different behaviors
    print("\n--- Creating Individual Mocks ---")
    
    # Success behavior mock
    success_spec = MockSpec(
        component=SystemComponent.LIGHTRAG_SYSTEM,
        behavior=MockBehavior.SUCCESS,
        response_delay=0.1,
        call_tracking=True
    )
    lightrag_mock = factory.create_lightrag_system(success_spec)
    print("✓ Created LightRAG mock with success behavior")
    
    # Failure behavior mock  
    failure_spec = MockSpec(
        component=SystemComponent.PDF_PROCESSOR,
        behavior=MockBehavior.FAILURE,
        call_tracking=True
    )
    pdf_mock = factory.create_pdf_processor(failure_spec)
    print("✓ Created PDF processor mock with failure behavior")
    
    # Cost monitor mock
    cost_spec = MockSpec(
        component=SystemComponent.COST_MONITOR,
        behavior=MockBehavior.SUCCESS,
        call_tracking=True
    )
    cost_mock = factory.create_cost_monitor(cost_spec)
    print("✓ Created cost monitor mock")
    
    # Progress tracker mock
    progress_spec = MockSpec(
        component=SystemComponent.PROGRESS_TRACKER,
        behavior=MockBehavior.SUCCESS,
        call_tracking=True
    )
    progress_mock = factory.create_progress_tracker(progress_spec)
    print("✓ Created progress tracker mock")
    
    # Comprehensive mock set
    print("\n--- Creating Comprehensive Mock Set ---")
    components = [
        SystemComponent.LIGHTRAG_SYSTEM,
        SystemComponent.PDF_PROCESSOR, 
        SystemComponent.COST_MONITOR,
        SystemComponent.PROGRESS_TRACKER,
        SystemComponent.CONFIG,
        SystemComponent.LOGGER
    ]
    
    comprehensive_mocks = factory.create_comprehensive_mock_set(components)
    print(f"✓ Created comprehensive mock set with {len(comprehensive_mocks)} components")
    
    # Show factory statistics
    stats = factory.get_mock_statistics()
    print(f"✓ Total mocks created: {stats['total_mocks_created']}")
    print(f"✓ Mock types: {', '.join(stats['mock_types'])}")
    
    # Cleanup
    env_manager.cleanup()


async def demo_async_operations():
    """Demonstrate async operations with mock systems."""
    print("\n=== Async Operations Demo ===")
    
    # Quick test environment setup
    env_manager, factory = create_quick_test_environment(async_support=True)
    
    # Create mock system
    mock_spec = MockSpec(
        component=SystemComponent.LIGHTRAG_SYSTEM,
        behavior=MockBehavior.SUCCESS,
        response_delay=0.05,  # Fast for demo
        call_tracking=True
    )
    lightrag_mock = factory.create_lightrag_system(mock_spec)
    
    # Demonstrate async context manager
    async with async_test_context(timeout=10.0) as context:
        print("✓ Async test context established")
        
        # Test document insertion
        documents = [
            "Clinical metabolomics research focuses on biomarker discovery.",
            "Proteomics analysis reveals disease-specific protein alterations.",
            "Genomics studies identify susceptibility variants."
        ]
        
        insert_result = await lightrag_mock.ainsert(documents)
        print(f"✓ Inserted {insert_result['documents_processed']} documents")
        print(f"✓ Cost: ${insert_result['total_cost']:.4f}")
        print(f"✓ Entities extracted: {insert_result['entities_extracted']}")
        
        # Test queries with different content types
        queries = [
            "What metabolites are associated with diabetes?",
            "What proteins are involved in cardiovascular disease?", 
            "How does clinical metabolomics support diagnosis?",
            "What pathways are dysregulated in cancer?"
        ]
        
        for query in queries:
            response = await lightrag_mock.aquery(query)
            print(f"✓ Query: {query[:50]}...")
            print(f"  Response length: {len(response)} characters")
    
    # Show call tracking
    call_logs = factory.get_call_logs("lightrag_system")
    print(f"✓ Tracked {len(call_logs.get('lightrag_system', []))} calls")
    
    # Cleanup
    env_manager.cleanup()
    print("✓ Async demo completed successfully")


def demo_integration_with_existing_fixtures():
    """Demonstrate integration with existing pytest fixtures."""
    print("\n=== Integration with Existing Fixtures Demo ===")
    
    # Show how utilities work with existing conftest.py patterns
    print("The utilities integrate with existing fixtures in several ways:")
    print("1. TestEnvironmentManager can be used as a pytest fixture")
    print("2. MockSystemFactory works with existing async fixtures") 
    print("3. Biomedical content generation complements existing generators")
    print("4. Performance monitoring extends existing test categories")
    
    # Create a mock test scenario that would use both old and new patterns
    print("\n--- Mock Test Scenario ---")
    
    # Old pattern (repetitive setup)
    print("OLD PATTERN (repetitive):")
    print("  - Manual temp directory creation")
    print("  - Repetitive sys.path management") 
    print("  - Individual mock creation for each test")
    print("  - Manual cleanup handling")
    print("  - Duplicated import error handling")
    
    # New pattern (streamlined)  
    print("\nNEW PATTERN (streamlined):")
    print("  - Standardized environment setup")
    print("  - Automatic sys.path management")
    print("  - Factory-based mock creation")
    print("  - Automatic cleanup with context managers")
    print("  - Centralized import validation")
    
    print("✓ Integration patterns demonstrated")


def demo_error_handling_and_recovery():
    """Demonstrate error handling and recovery capabilities."""
    print("\n=== Error Handling and Recovery Demo ===")
    
    env_manager, factory = create_quick_test_environment()
    
    # Test failure scenarios
    print("--- Testing Failure Scenarios ---")
    
    # Create mock that fails
    failure_spec = MockSpec(
        component=SystemComponent.PDF_PROCESSOR,
        behavior=MockBehavior.FAILURE,
        call_tracking=True
    )
    pdf_mock = factory.create_pdf_processor(failure_spec)
    
    try:
        # This should fail
        result = pdf_mock.process_pdf.side_effect
        print(f"✓ Failure behavior configured: {type(result).__name__}")
    except Exception as e:
        print(f"✓ Expected failure handled: {e}")
    
    # Test timeout scenarios
    timeout_spec = MockSpec(
        component=SystemComponent.LIGHTRAG_SYSTEM,
        behavior=MockBehavior.TIMEOUT,
        response_delay=0.1,
        call_tracking=True
    )
    timeout_mock = factory.create_lightrag_system(timeout_spec)
    print("✓ Timeout behavior mock created")
    
    # Test partial success scenarios
    partial_spec = MockSpec(
        component=SystemComponent.LIGHTRAG_SYSTEM, 
        behavior=MockBehavior.PARTIAL_SUCCESS,
        failure_rate=0.3,  # 30% failure rate
        call_tracking=True
    )
    partial_mock = factory.create_lightrag_system(partial_spec)
    print("✓ Partial success behavior mock created")
    
    # Show graceful import fallback
    print("\n--- Testing Import Fallback ---")
    try:
        # Try to get non-existent module
        fallback_module = env_manager.get_import('non_existent_module', fallback_to_mock=True)
        print(f"✓ Fallback mock created for missing module: {type(fallback_module).__name__}")
    except Exception as e:
        print(f"✗ Fallback failed: {e}")
    
    env_manager.cleanup()
    print("✓ Error handling demo completed")


def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""  
    print("\n=== Performance Monitoring Demo ===")
    
    # Create performance test setup
    env_manager, factory = create_quick_test_environment()
    
    # Enable performance monitoring
    spec = EnvironmentSpec(
        performance_monitoring=True,
        memory_limits={'test_limit': 256}  # 256MB limit
    )
    
    perf_env = TestEnvironmentManager(spec)
    perf_data = perf_env.setup_environment()
    
    if perf_data['memory_monitor']:
        print("✓ Memory monitoring enabled")
        
        # Simulate some work
        perf_data['memory_monitor'].start_monitoring()
        time.sleep(1.0)  # Simulate work
        samples = perf_data['memory_monitor'].stop_monitoring()
        
        if samples:
            print(f"✓ Collected {len(samples)} memory samples")
            print(f"✓ Peak memory: {max(s['rss_mb'] for s in samples):.1f} MB")
        else:
            print("✓ Memory monitoring configured (no samples in short demo)")
    else:
        print("✓ Performance monitoring setup attempted")
    
    # Show system health monitoring
    health = perf_env.check_system_health()
    print(f"✓ Current memory usage: {health['memory_usage_mb']:.1f} MB")
    print(f"✓ Active threads: {health['active_threads']}")
    
    perf_env.cleanup()
    print("✓ Performance monitoring demo completed")


async def run_all_demos():
    """Run all demonstration scenarios."""
    print("Clinical Metabolomics Oracle - Test Utilities Demonstration")
    print("=" * 60)
    
    # Run sync demos
    demo_basic_environment_setup()
    demo_mock_system_factory()
    demo_integration_with_existing_fixtures()
    demo_error_handling_and_recovery()
    demo_performance_monitoring()
    
    # Run async demo
    await demo_async_operations()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("\nSUMMARY:")
    print("- TestEnvironmentManager: Eliminates repetitive environment setup")
    print("- MockSystemFactory: Standardizes mock creation with behavior patterns")
    print("- Integration: Works seamlessly with existing fixtures and patterns")
    print("- Error Handling: Provides robust fallback mechanisms")
    print("- Performance: Includes monitoring and resource management")
    print("- Async Support: Full compatibility with async test infrastructure")
    print("\nThese utilities reduce 40+ repetitive patterns identified in test analysis.")


if __name__ == "__main__":
    asyncio.run(run_all_demos())