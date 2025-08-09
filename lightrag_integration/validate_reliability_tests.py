#!/usr/bin/env python3
"""
Reliability Tests Validation Script
===================================

Quick validation script to ensure all reliability test modules can be imported
and basic functionality is working correctly.

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import sys
import traceback
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_imports():
    """Validate that all test modules can be imported correctly."""
    logger.info("🔍 Validating test module imports...")
    
    import_tests = [
        ("Reliability Test Framework", "tests.reliability_test_framework"),
        ("Stress Testing Scenarios", "tests.test_stress_testing_scenarios"),
        ("Network Reliability Scenarios", "tests.test_network_reliability_scenarios"),
        ("Data Integrity Scenarios", "tests.test_data_integrity_scenarios"),
        ("Production Scenarios", "tests.test_production_scenarios"),
        ("Integration Reliability Scenarios", "tests.test_integration_reliability_scenarios"),
        ("Test Runner", "run_reliability_validation_tests")
    ]
    
    results = {}
    
    for test_name, module_name in import_tests:
        try:
            __import__(module_name)
            results[test_name] = {"status": "✅ SUCCESS", "error": None}
            logger.info(f"  ✅ {test_name}: Import successful")
        except Exception as e:
            results[test_name] = {"status": "❌ FAILED", "error": str(e)}
            logger.error(f"  ❌ {test_name}: Import failed - {str(e)}")
    
    return results

async def validate_framework_creation():
    """Validate that the reliability test framework can be created."""
    logger.info("🏗️  Validating framework creation...")
    
    try:
        from tests.reliability_test_framework import (
            ReliabilityValidationFramework,
            ReliabilityTestConfig,
            create_test_orchestrator
        )
        
        # Test configuration creation
        config = ReliabilityTestConfig()
        logger.info(f"  ✅ Config created: base_rps={config.base_rps}, max_rps={config.max_rps}")
        
        # Test framework creation
        framework = ReliabilityValidationFramework(config)
        logger.info("  ✅ Framework created successfully")
        
        # Test orchestrator creation
        orchestrator = await create_test_orchestrator(config)
        logger.info("  ✅ Test orchestrator created successfully")
        
        # Test basic orchestrator functionality
        if hasattr(orchestrator, 'get_health_check'):
            health = orchestrator.get_health_check()
            logger.info(f"  ✅ Health check: {health}")
        else:
            logger.info("  ℹ️  Health check method not available (using mock)")
        
        return {"status": "✅ SUCCESS", "error": None}
        
    except Exception as e:
        logger.error(f"  ❌ Framework validation failed: {str(e)}")
        return {"status": "❌ FAILED", "error": str(e)}

async def validate_test_functions():
    """Validate that test functions can be imported and have correct signatures."""
    logger.info("⚙️  Validating test function signatures...")
    
    test_functions = [
        ("ST-001", "tests.test_stress_testing_scenarios", "test_progressive_load_escalation"),
        ("ST-002", "tests.test_stress_testing_scenarios", "test_burst_load_handling"),
        ("NR-001", "tests.test_network_reliability_scenarios", "test_lightrag_service_degradation"),
        ("DI-001", "tests.test_data_integrity_scenarios", "test_cross_source_response_consistency"),
        ("PS-001", "tests.test_production_scenarios", "test_peak_hour_load_simulation"),
        ("IR-001", "tests.test_integration_reliability_scenarios", "test_circuit_breaker_threshold_validation")
    ]
    
    results = {}
    
    for test_id, module_name, function_name in test_functions:
        try:
            module = __import__(module_name, fromlist=[function_name])
            test_func = getattr(module, function_name)
            
            # Check if it's callable
            if callable(test_func):
                results[test_id] = {"status": "✅ SUCCESS", "error": None}
                logger.info(f"  ✅ {test_id} ({function_name}): Function available")
            else:
                results[test_id] = {"status": "❌ FAILED", "error": "Not callable"}
                logger.error(f"  ❌ {test_id} ({function_name}): Not callable")
                
        except Exception as e:
            results[test_id] = {"status": "❌ FAILED", "error": str(e)}
            logger.error(f"  ❌ {test_id} ({function_name}): {str(e)}")
    
    return results

async def validate_runner_functions():
    """Validate that the test runner functions are available."""
    logger.info("🏃 Validating test runner functions...")
    
    runner_functions = [
        ("Stress Tests Runner", "tests.test_stress_testing_scenarios", "run_all_stress_tests"),
        ("Network Tests Runner", "tests.test_network_reliability_scenarios", "run_all_network_reliability_tests"),
        ("Data Integrity Tests Runner", "tests.test_data_integrity_scenarios", "run_all_data_integrity_tests"),
        ("Production Tests Runner", "tests.test_production_scenarios", "run_all_production_scenario_tests"),
        ("Integration Tests Runner", "tests.test_integration_reliability_scenarios", "run_all_integration_reliability_tests")
    ]
    
    results = {}
    
    for runner_name, module_name, function_name in runner_functions:
        try:
            module = __import__(module_name, fromlist=[function_name])
            runner_func = getattr(module, function_name)
            
            if callable(runner_func):
                results[runner_name] = {"status": "✅ SUCCESS", "error": None}
                logger.info(f"  ✅ {runner_name}: Available")
            else:
                results[runner_name] = {"status": "❌ FAILED", "error": "Not callable"}
                logger.error(f"  ❌ {runner_name}: Not callable")
                
        except Exception as e:
            results[runner_name] = {"status": "❌ FAILED", "error": str(e)}
            logger.error(f"  ❌ {runner_name}: {str(e)}")
    
    return results

def validate_file_structure():
    """Validate that all expected test files exist."""
    logger.info("📁 Validating file structure...")
    
    expected_files = [
        "tests/reliability_test_framework.py",
        "tests/test_stress_testing_scenarios.py",
        "tests/test_network_reliability_scenarios.py",
        "tests/test_data_integrity_scenarios.py",
        "tests/test_production_scenarios.py",
        "tests/test_integration_reliability_scenarios.py",
        "run_reliability_validation_tests.py"
    ]
    
    results = {}
    
    for file_path in expected_files:
        path = Path(file_path)
        if path.exists():
            results[file_path] = {"status": "✅ EXISTS", "size": path.stat().st_size}
            logger.info(f"  ✅ {file_path}: {path.stat().st_size} bytes")
        else:
            results[file_path] = {"status": "❌ MISSING", "size": 0}
            logger.error(f"  ❌ {file_path}: File not found")
    
    return results

async def run_basic_integration_test():
    """Run a very basic integration test to ensure the system works end-to-end."""
    logger.info("🧪 Running basic integration test...")
    
    try:
        from tests.reliability_test_framework import (
            ReliabilityValidationFramework,
            ReliabilityTestConfig
        )
        
        # Create minimal test configuration
        config = ReliabilityTestConfig(
            max_test_duration_minutes=1,  # Very short for validation
            base_rps=1.0,
            max_rps=5.0
        )
        
        framework = ReliabilityValidationFramework(config)
        
        # Setup test environment
        await framework.setup_test_environment()
        
        # Define a simple test function
        async def simple_test(orchestrator, config):
            """Simple test that just submits one request."""
            async def test_handler():
                await asyncio.sleep(0.1)
                return "Test successful"
            
            result = await orchestrator.submit_request(
                request_type='user_query',
                priority='medium',
                handler=test_handler,
                timeout=5.0
            )
            
            assert result[0], f"Test request failed: {result[1]}"
            return {"message": "Simple integration test passed"}
        
        # Execute the test
        result = await framework.execute_monitored_test(
            test_name="basic_integration_test",
            test_func=simple_test,
            category="validation"
        )
        
        # Cleanup
        await framework.cleanup_test_environment()
        
        if result.status == 'passed':
            logger.info("  ✅ Basic integration test passed")
            return {"status": "✅ SUCCESS", "details": result.details}
        else:
            logger.error(f"  ❌ Basic integration test failed: {result.error}")
            return {"status": "❌ FAILED", "error": result.error}
            
    except Exception as e:
        logger.error(f"  ❌ Basic integration test failed: {str(e)}")
        logger.error(f"  📍 Traceback: {traceback.format_exc()}")
        return {"status": "❌ FAILED", "error": str(e)}

async def main():
    """Main validation function."""
    logger.info("🚀 Starting Reliability Tests Validation")
    logger.info("=" * 60)
    
    validation_results = {}
    
    # Run all validation tests
    validation_results["file_structure"] = validate_file_structure()
    validation_results["imports"] = validate_imports()
    validation_results["framework_creation"] = await validate_framework_creation()
    validation_results["test_functions"] = await validate_test_functions()
    validation_results["runner_functions"] = await validate_runner_functions()
    validation_results["basic_integration"] = await run_basic_integration_test()
    
    logger.info("=" * 60)
    logger.info("🏁 Validation Summary")
    
    # Calculate overall results
    total_validations = 0
    successful_validations = 0
    
    for validation_name, validation_data in validation_results.items():
        logger.info(f"\n📋 {validation_name.replace('_', ' ').title()}:")
        
        if isinstance(validation_data, dict):
            if "status" in validation_data:
                # Single validation result
                logger.info(f"  {validation_data['status']}")
                total_validations += 1
                if "SUCCESS" in validation_data['status']:
                    successful_validations += 1
            else:
                # Multiple validation results
                for item_name, item_result in validation_data.items():
                    if isinstance(item_result, dict) and "status" in item_result:
                        logger.info(f"  {item_result['status']} {item_name}")
                        total_validations += 1
                        if "SUCCESS" in item_result['status'] or "EXISTS" in item_result['status']:
                            successful_validations += 1
    
    success_rate = (successful_validations / total_validations * 100) if total_validations > 0 else 0
    
    logger.info(f"\n🎯 Overall Validation Results:")
    logger.info(f"   Total Validations: {total_validations}")
    logger.info(f"   Successful: {successful_validations}")
    logger.info(f"   Failed: {total_validations - successful_validations}")
    logger.info(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        logger.info("🎉 VALIDATION PASSED - Reliability tests are ready for execution!")
        return True
    elif success_rate >= 70:
        logger.warning("⚠️  VALIDATION PARTIALLY PASSED - Some issues detected but tests should be mostly functional")
        return True
    else:
        logger.error("❌ VALIDATION FAILED - Significant issues detected, please fix before running tests")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Validation failed with unexpected error: {str(e)}")
        logger.error(f"📍 Traceback: {traceback.format_exc()}")
        sys.exit(1)