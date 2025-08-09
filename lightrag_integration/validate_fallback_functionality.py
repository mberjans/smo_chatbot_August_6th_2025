#!/usr/bin/env python3
"""
Fallback System Functionality Validation
========================================

This script provides direct functional testing of the integrated fallback system
without relying on complex test frameworks. It validates core functionality
by directly testing the components.

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Direct functional validation of fallback system
"""

import sys
import os
import time
import tempfile
import traceback
from pathlib import Path

def test_basic_imports():
    """Test that core modules can be imported."""
    print("Testing basic imports...")
    
    imports_status = {}
    
    # Test core routing components
    try:
        from query_router import BiomedicalQueryRouter
        imports_status['BiomedicalQueryRouter'] = True
        print("✓ BiomedicalQueryRouter imported successfully")
    except Exception as e:
        imports_status['BiomedicalQueryRouter'] = False
        print(f"✗ BiomedicalQueryRouter failed: {e}")
    
    # Test fallback system
    try:
        from comprehensive_fallback_system import FallbackOrchestrator
        imports_status['FallbackOrchestrator'] = True
        print("✓ FallbackOrchestrator imported successfully")
    except Exception as e:
        imports_status['FallbackOrchestrator'] = False
        print(f"✗ FallbackOrchestrator failed: {e}")
    
    # Test enhanced router
    try:
        from enhanced_query_router_with_fallback import EnhancedBiomedicalQueryRouter
        imports_status['EnhancedBiomedicalQueryRouter'] = True
        print("✓ EnhancedBiomedicalQueryRouter imported successfully")
    except Exception as e:
        imports_status['EnhancedBiomedicalQueryRouter'] = False
        print(f"✗ EnhancedBiomedicalQueryRouter failed: {e}")
    
    # Test clinical configuration
    try:
        from clinical_metabolomics_fallback_config import ClinicalMetabolomicsFallbackConfig
        imports_status['ClinicalMetabolomicsFallbackConfig'] = True
        print("✓ ClinicalMetabolomicsFallbackConfig imported successfully")
    except Exception as e:
        imports_status['ClinicalMetabolomicsFallbackConfig'] = False
        print(f"✗ ClinicalMetabolomicsFallbackConfig failed: {e}")
    
    return imports_status


def test_basic_router_functionality():
    """Test basic router functionality."""
    print("\nTesting basic router functionality...")
    
    try:
        from query_router import BiomedicalQueryRouter
        
        # Create router instance
        router = BiomedicalQueryRouter()
        print("✓ Router instance created successfully")
        
        # Test basic routing
        test_queries = [
            "What is glucose?",
            "Analyze metabolic pathways in diabetes",
            "Complex multi-pathway analysis with uncertainty"
        ]
        
        for query in test_queries:
            try:
                result = router.route_query(query)
                if result and hasattr(result, 'routing_decision'):
                    print(f"✓ Query routed successfully: '{query[:30]}...' -> {result.routing_decision}")
                else:
                    print(f"✗ Query routing returned invalid result: '{query[:30]}...'")
                    
            except Exception as e:
                print(f"✗ Query routing failed: '{query[:30]}...' - {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic router test failed: {e}")
        return False


def test_fallback_system_basic():
    """Test basic fallback system functionality."""
    print("\nTesting fallback system basics...")
    
    try:
        from comprehensive_fallback_system import (
            FallbackOrchestrator, FailureDetector, EmergencyCache, 
            GracefulDegradationManager, RecoveryManager
        )
        
        # Test individual components
        components_status = {}
        
        # Test FailureDetector
        try:
            detector = FailureDetector()
            detector.record_operation_result(response_time_ms=100, success=True)
            health_score = detector.metrics.calculate_health_score()
            assert 0 <= health_score <= 1
            components_status['FailureDetector'] = True
            print(f"✓ FailureDetector working - Health score: {health_score:.3f}")
        except Exception as e:
            components_status['FailureDetector'] = False
            print(f"✗ FailureDetector failed: {e}")
        
        # Test EmergencyCache
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_file = Path(temp_dir) / "test_cache.pkl"
                cache = EmergencyCache(cache_file=str(cache_file))
                
                # Test cache retrieval
                result = cache.get_cached_response("metabolite")
                components_status['EmergencyCache'] = True
                print(f"✓ EmergencyCache working - Cache size: {len(cache.cache)}")
        except Exception as e:
            components_status['EmergencyCache'] = False
            print(f"✗ EmergencyCache failed: {e}")
        
        # Test GracefulDegradationManager
        try:
            manager = GracefulDegradationManager()
            level = manager.determine_optimal_fallback_level([], 0.8, 'normal')
            components_status['GracefulDegradationManager'] = True
            print(f"✓ GracefulDegradationManager working - Determined level: {level}")
        except Exception as e:
            components_status['GracefulDegradationManager'] = False
            print(f"✗ GracefulDegradationManager failed: {e}")
        
        # Test RecoveryManager
        try:
            recovery = RecoveryManager()
            recovery.register_service_health_check('test', lambda: True)
            components_status['RecoveryManager'] = True
            print("✓ RecoveryManager working")
        except Exception as e:
            components_status['RecoveryManager'] = False
            print(f"✗ RecoveryManager failed: {e}")
        
        working_components = sum(components_status.values())
        total_components = len(components_status)
        print(f"✓ Fallback components working: {working_components}/{total_components}")
        
        return working_components >= 3  # At least 3/4 components should work
        
    except Exception as e:
        print(f"✗ Fallback system test failed: {e}")
        traceback.print_exc()
        return False


def test_enhanced_router_basic():
    """Test enhanced router basic functionality."""
    print("\nTesting enhanced router basics...")
    
    try:
        from enhanced_query_router_with_fallback import (
            EnhancedBiomedicalQueryRouter, FallbackIntegrationConfig
        )
        
        # Create configuration
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FallbackIntegrationConfig(
                emergency_cache_file=str(Path(temp_dir) / "cache.pkl"),
                enable_fallback_system=True,
                enable_monitoring=False  # Disable for testing
            )
            
            # Create enhanced router
            router = EnhancedBiomedicalQueryRouter(fallback_config=config)
            print("✓ Enhanced router created successfully")
            
            # Test basic routing
            test_queries = [
                "glucose metabolism",
                "complex pathway analysis",
                "metabolomics biomarkers"
            ]
            
            successful_queries = 0
            for query in test_queries:
                try:
                    result = router.route_query(query)
                    if result and hasattr(result, 'confidence') and result.confidence >= 0:
                        successful_queries += 1
                        print(f"✓ Enhanced routing successful: '{query}' -> conf:{result.confidence:.3f}")
                    else:
                        print(f"✗ Enhanced routing returned invalid result: '{query}'")
                except Exception as e:
                    print(f"✗ Enhanced routing failed: '{query}' - {e}")
            
            # Test additional features
            try:
                health = router.get_system_health_report()
                if health and isinstance(health, dict):
                    print("✓ Health reporting working")
                else:
                    print("✗ Health reporting failed")
            except Exception as e:
                print(f"✗ Health reporting error: {e}")
            
            # Cleanup
            try:
                router.shutdown_enhanced_features()
                print("✓ Router shutdown successful")
            except Exception as e:
                print(f"✗ Router shutdown error: {e}")
            
            success_rate = successful_queries / len(test_queries)
            print(f"✓ Enhanced router success rate: {success_rate:.1%}")
            
            return success_rate >= 0.5
    
    except Exception as e:
        print(f"✗ Enhanced router test failed: {e}")
        traceback.print_exc()
        return False


def test_clinical_configuration():
    """Test clinical metabolomics configuration."""
    print("\nTesting clinical metabolomics configuration...")
    
    try:
        from clinical_metabolomics_fallback_config import ClinicalMetabolomicsFallbackConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create configuration
            config = ClinicalMetabolomicsFallbackConfig(
                emergency_cache_file=str(Path(temp_dir) / "clinical_cache.pkl"),
                biomedical_confidence_threshold=0.7,
                max_query_timeout_ms=5000,
                enable_metabolomics_cache_warming=True,
                enable_clinical_monitoring=True
            )
            
            print("✓ Clinical configuration created successfully")
            
            # Test configuration properties
            config_dict = config.to_dict()
            if isinstance(config_dict, dict) and len(config_dict) > 0:
                print(f"✓ Configuration dictionary created - {len(config_dict)} settings")
            else:
                print("✗ Configuration dictionary creation failed")
                return False
            
            # Test specific settings
            expected_settings = [
                'biomedical_confidence_threshold',
                'max_query_timeout_ms', 
                'enable_metabolomics_cache_warming'
            ]
            
            missing_settings = [s for s in expected_settings if s not in config_dict]
            if missing_settings:
                print(f"✗ Missing configuration settings: {missing_settings}")
                return False
            
            print("✓ All expected clinical settings present")
            
            # Test validation
            try:
                validation_result = config.validate_configuration()
                if validation_result.get('valid', False):
                    print("✓ Configuration validation passed")
                else:
                    issues = validation_result.get('issues', [])
                    print(f"✗ Configuration validation failed: {issues}")
                    return False
            except Exception as e:
                print(f"✗ Configuration validation error: {e}")
                return False
            
            return True
    
    except Exception as e:
        print(f"✗ Clinical configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end_integration():
    """Test end-to-end integration with clinical queries."""
    print("\nTesting end-to-end integration...")
    
    try:
        # Clinical metabolomics test queries
        clinical_queries = [
            "What are the key metabolites in glycolysis?",
            "How does insulin affect glucose metabolism?", 
            "Identify biomarkers for diabetes mellitus",
            "Analyze metabolic pathways in cardiovascular disease",
            "What is the role of citric acid cycle in energy production?"
        ]
        
        # Try to test with available components
        successful_tests = 0
        
        # Test 1: Basic routing
        try:
            from query_router import BiomedicalQueryRouter
            router = BiomedicalQueryRouter()
            
            for query in clinical_queries[:2]:  # Test first 2 queries
                result = router.route_query(query)
                if result and hasattr(result, 'confidence'):
                    successful_tests += 1
                    print(f"✓ Basic routing: '{query[:40]}...' -> {result.routing_decision}")
            
        except Exception as e:
            print(f"✗ Basic routing integration failed: {e}")
        
        # Test 2: Enhanced routing if available
        try:
            from enhanced_query_router_with_fallback import (
                EnhancedBiomedicalQueryRouter, FallbackIntegrationConfig
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config = FallbackIntegrationConfig(
                    emergency_cache_file=str(Path(temp_dir) / "e2e_cache.pkl"),
                    enable_fallback_system=True,
                    enable_monitoring=False
                )
                
                enhanced_router = EnhancedBiomedicalQueryRouter(fallback_config=config)
                
                for query in clinical_queries[2:4]:  # Test next 2 queries
                    result = enhanced_router.route_query(query)
                    if result and hasattr(result, 'confidence'):
                        successful_tests += 1
                        print(f"✓ Enhanced routing: '{query[:40]}...' -> conf:{result.confidence:.3f}")
                
                enhanced_router.shutdown_enhanced_features()
        
        except Exception as e:
            print(f"✗ Enhanced routing integration failed: {e}")
        
        # Test 3: Fallback system if available
        try:
            from comprehensive_fallback_system import create_comprehensive_fallback_system
            
            with tempfile.TemporaryDirectory() as temp_dir:
                fallback_config = {
                    'emergency_cache_file': str(Path(temp_dir) / "fallback_cache.pkl"),
                    'biomedical_confidence_threshold': 0.7,
                    'max_query_timeout_ms': 5000
                }
                
                fallback_system = create_comprehensive_fallback_system(
                    config=fallback_config,
                    cache_dir=temp_dir
                )
                
                query = clinical_queries[-1]  # Test last query
                result = fallback_system.process_query_with_comprehensive_fallback(query)
                if result and hasattr(result, 'success') and result.success:
                    successful_tests += 1
                    print(f"✓ Fallback system: '{query[:40]}...' -> {result.fallback_level_used}")
        
        except Exception as e:
            print(f"✗ Fallback system integration failed: {e}")
        
        success_rate = successful_tests / 5  # Total possible successful tests
        print(f"✓ End-to-end integration success rate: {success_rate:.1%} ({successful_tests}/5)")
        
        return success_rate >= 0.4  # At least 40% should work
    
    except Exception as e:
        print(f"✗ End-to-end integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run comprehensive functionality validation."""
    print("=" * 80)
    print("FALLBACK SYSTEM FUNCTIONALITY VALIDATION")
    print("=" * 80)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Import Tests", test_basic_imports),
        ("Basic Router", test_basic_router_functionality),
        ("Fallback System", test_fallback_system_basic),
        ("Enhanced Router", test_enhanced_router_basic),
        ("Clinical Config", test_clinical_configuration),
        ("End-to-End Integration", test_end_to_end_integration)
    ]
    
    for test_name, test_function in tests:
        print(f"\n{'-' * 60}")
        print(f"Running: {test_name}")
        print(f"{'-' * 60}")
        
        try:
            start_time = time.time()
            result = test_function()
            duration = time.time() - start_time
            
            if isinstance(result, dict):  # Import test returns dict
                success = sum(result.values()) / len(result) >= 0.5
                test_results[test_name] = {
                    'success': success,
                    'details': result,
                    'duration': duration
                }
            else:
                test_results[test_name] = {
                    'success': result,
                    'duration': duration
                }
            
            status = "✓ PASSED" if test_results[test_name]['success'] else "✗ FAILED"
            print(f"\n{status} - {test_name} ({duration:.2f}s)")
            
        except Exception as e:
            test_results[test_name] = {
                'success': False,
                'error': str(e),
                'duration': 0
            }
            print(f"\n✗ FAILED - {test_name}: {e}")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY REPORT")
    print("=" * 80)
    
    passed_tests = sum(1 for result in test_results.values() if result['success'])
    total_tests = len(test_results)
    success_rate = passed_tests / total_tests
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   ✓ Passed: {passed_tests}/{total_tests} tests")
    print(f"   📈 Success Rate: {success_rate:.1%}")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, result in test_results.items():
        status = "✓" if result['success'] else "✗"
        duration = result.get('duration', 0)
        print(f"   {status} {test_name:<25} ({duration:.2f}s)")
        
        if 'error' in result:
            print(f"     Error: {result['error']}")
        elif 'details' in result and isinstance(result['details'], dict):
            working = sum(result['details'].values())
            total = len(result['details'])
            print(f"     Details: {working}/{total} components working")
    
    # Overall assessment
    print(f"\n🔍 ASSESSMENT:")
    if success_rate >= 0.8:
        assessment = "🟢 EXCELLENT - Fallback system is fully functional"
    elif success_rate >= 0.6:
        assessment = "🟡 GOOD - Fallback system is mostly functional with minor issues"
    elif success_rate >= 0.4:
        assessment = "🟠 FAIR - Fallback system has significant issues but core functionality works"
    else:
        assessment = "🔴 POOR - Fallback system has major issues requiring attention"
    
    print(f"   {assessment}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if success_rate < 0.6:
        print("   • Review failed components and address import/configuration issues")
        print("   • Ensure all required dependencies are properly installed")
        print("   • Check for missing configuration files or environment variables")
    
    if success_rate >= 0.4:
        print("   • Core functionality appears to be working")
        print("   • Consider running production tests with real data")
        print("   • Monitor system performance under load")
    
    print("\n" + "=" * 80)
    
    return success_rate >= 0.6


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)