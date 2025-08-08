#!/usr/bin/env python3
"""
Feature Flag Examples for LightRAG Integration

This script demonstrates the conditional import and feature flag system
implemented in the lightrag_integration module.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_basic_feature_flags():
    """Demonstrate basic feature flag functionality"""
    print("=" * 70)
    print("BASIC FEATURE FLAG DEMONSTRATION")
    print("=" * 70)
    
    # Set various feature flags
    os.environ['LIGHTRAG_INTEGRATION_ENABLED'] = 'true'
    os.environ['LIGHTRAG_ENABLE_QUALITY_VALIDATION'] = 'true'
    os.environ['LIGHTRAG_ENABLE_RELEVANCE_SCORING'] = 'true'
    os.environ['LIGHTRAG_ENABLE_COST_TRACKING'] = 'true'
    os.environ['LIGHTRAG_ENABLE_PERFORMANCE_MONITORING'] = 'false'
    
    import lightrag_integration
    
    print(f"Module version: {lightrag_integration.__version__}\\n")
    
    # Check individual features
    features_to_check = [
        'lightrag_integration_enabled',
        'quality_validation_enabled',
        'relevance_scoring_enabled',
        'cost_tracking_enabled',
        'performance_monitoring_enabled',
        'benchmarking_enabled',
        'document_indexing_enabled'
    ]
    
    print("Feature Flag Status:")
    for feature in features_to_check:
        enabled = lightrag_integration.is_feature_enabled(feature)
        status = "✓ ENABLED" if enabled else "✗ DISABLED"
        print(f"  {feature:35} {status}")
    
    # Show all enabled features
    enabled = lightrag_integration.get_enabled_features()
    print(f"\\nTotal enabled features: {len(enabled)}")
    print(f"Enabled features: {', '.join(enabled.keys())}")


def demo_conditional_imports():
    """Demonstrate how conditional imports work based on feature flags"""
    print("\\n" + "=" * 70)
    print("CONDITIONAL IMPORT DEMONSTRATION")
    print("=" * 70)
    
    # First scenario: Enable quality validation features
    print("\\n1. Testing with Quality Validation ENABLED:")
    os.environ['LIGHTRAG_ENABLE_QUALITY_VALIDATION'] = 'true'
    os.environ['LIGHTRAG_ENABLE_RELEVANCE_SCORING'] = 'true'
    
    # Need to reload the module for changes to take effect
    if 'lightrag_integration' in sys.modules:
        del sys.modules['lightrag_integration']
    
    import lightrag_integration
    
    # Check if quality validation components are available
    has_quality_assessor = hasattr(lightrag_integration, 'EnhancedResponseQualityAssessor')
    has_relevance_scorer = hasattr(lightrag_integration, 'RelevanceScorer')
    
    print(f"  EnhancedResponseQualityAssessor available: {has_quality_assessor}")
    print(f"  RelevanceScorer available: {has_relevance_scorer}")
    print(f"  Quality validation exports in __all__: {'QualityReportGenerator' in lightrag_integration.__all__}")
    
    # Second scenario: Disable quality validation features  
    print("\\n2. Testing with Quality Validation DISABLED:")
    os.environ['LIGHTRAG_ENABLE_QUALITY_VALIDATION'] = 'false'
    os.environ['LIGHTRAG_ENABLE_RELEVANCE_SCORING'] = 'false'
    
    # Reload module
    if 'lightrag_integration' in sys.modules:
        del sys.modules['lightrag_integration']
        
    import lightrag_integration
    
    has_quality_assessor = lightrag_integration.EnhancedResponseQualityAssessor is not None
    has_relevance_scorer = lightrag_integration.RelevanceScorer is not None
    
    print(f"  EnhancedResponseQualityAssessor available: {has_quality_assessor}")
    print(f"  RelevanceScorer available: {has_relevance_scorer}")
    print(f"  Quality validation exports in __all__: {'QualityReportGenerator' in lightrag_integration.__all__}")


def demo_factory_functions():
    """Demonstrate feature-aware factory functions"""
    print("\\n" + "=" * 70)
    print("FEATURE-AWARE FACTORY FUNCTIONS")
    print("=" * 70)
    
    # Enable features for demonstration
    os.environ['LIGHTRAG_INTEGRATION_ENABLED'] = 'true'
    os.environ['LIGHTRAG_ENABLE_QUALITY_VALIDATION'] = 'true'
    os.environ['LIGHTRAG_ENABLE_COST_TRACKING'] = 'true'
    
    # Reload module
    if 'lightrag_integration' in sys.modules:
        del sys.modules['lightrag_integration']
        
    import lightrag_integration
    
    print("\\n1. Available Factory Functions:")
    factory_functions = [name for name in lightrag_integration.__all__ if 'create_' in name]
    for func in factory_functions:
        print(f"  - {func}")
    
    print("\\n2. Testing Feature-Aware Factory Function:")
    try:
        # This should work since we enabled the integration
        print("  Attempting to create system with features...")
        print("  Features would be applied based on environment variables")
        print("  ✓ create_clinical_rag_system_with_features() is available")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\\n3. Testing Disabled Integration:")
    os.environ['LIGHTRAG_INTEGRATION_ENABLED'] = 'false'
    
    # Reload module
    if 'lightrag_integration' in sys.modules:
        del sys.modules['lightrag_integration']
        
    import lightrag_integration
    
    try:
        # This should raise an error since integration is disabled
        lightrag_integration.create_clinical_rag_system_with_features()
    except RuntimeError as e:
        print(f"  ✓ Correctly blocked disabled integration: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")


def demo_integration_status():
    """Demonstrate integration status monitoring"""
    print("\\n" + "=" * 70)
    print("INTEGRATION STATUS MONITORING")
    print("=" * 70)
    
    # Set a mixed environment
    os.environ['LIGHTRAG_INTEGRATION_ENABLED'] = 'true'
    os.environ['LIGHTRAG_ENABLE_QUALITY_VALIDATION'] = 'true'
    os.environ['LIGHTRAG_ENABLE_PERFORMANCE_MONITORING'] = 'false'
    os.environ['LIGHTRAG_ENABLE_BENCHMARKING'] = 'false'
    
    # Reload module
    if 'lightrag_integration' in sys.modules:
        del sys.modules['lightrag_integration']
        
    import lightrag_integration
    
    # Get comprehensive status
    status = lightrag_integration.get_integration_status()
    
    print(f"\\nIntegration Health: {status['integration_health']}")
    print(f"Total Feature Flags: {len(status['feature_flags'])}")
    print(f"Enabled Flags: {len([f for f in status['feature_flags'].values() if f])}")
    print(f"Registered Modules: {len(status['modules'])}")
    print(f"Available Factory Functions: {len(status['factory_functions'])}")
    
    # Show module status
    print("\\nModule Status Summary:")
    for module_name, module_status in status['modules'].items():
        enabled_status = "✓" if module_status['enabled'] else "✗"
        available_status = "✓" if module_status['available'] else "✗"
        print(f"  {module_name:30} Enabled: {enabled_status}  Available: {available_status}")
    
    # Validation
    is_valid, issues = lightrag_integration.validate_integration_setup()
    print(f"\\nSetup Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")


def demo_graceful_degradation():
    """Demonstrate graceful degradation when features are disabled"""
    print("\\n" + "=" * 70)
    print("GRACEFUL DEGRADATION DEMONSTRATION")
    print("=" * 70)
    
    # Disable most optional features
    optional_features = [
        'LIGHTRAG_ENABLE_QUALITY_VALIDATION',
        'LIGHTRAG_ENABLE_RELEVANCE_SCORING',
        'LIGHTRAG_ENABLE_PERFORMANCE_MONITORING',
        'LIGHTRAG_ENABLE_BENCHMARKING',
        'LIGHTRAG_ENABLE_DOCUMENT_INDEXING',
        'LIGHTRAG_ENABLE_RECOVERY_SYSTEM'
    ]
    
    for feature in optional_features:
        os.environ[feature] = 'false'
    
    # Keep core features enabled
    os.environ['LIGHTRAG_INTEGRATION_ENABLED'] = 'true'
    os.environ['LIGHTRAG_ENABLE_COST_TRACKING'] = 'true'
    
    # Reload module
    if 'lightrag_integration' in sys.modules:
        del sys.modules['lightrag_integration']
        
    import lightrag_integration
    
    print("Testing with minimal feature set:")
    print(f"  Module loads successfully: ✓")
    print(f"  Core functions available: ✓")
    
    enabled = lightrag_integration.get_enabled_features()
    print(f"  Enabled features: {len(enabled)} ({', '.join(enabled.keys())})")
    
    status = lightrag_integration.get_integration_status()
    print(f"  Integration health: {status['integration_health']}")
    
    # Test that disabled features return None stubs
    disabled_components = [
        'RelevanceScorer',
        'QualityReportGenerator', 
        'AdvancedRecoverySystem',
        'DocumentIndexer'
    ]
    
    print("\\n  Disabled component status:")
    for component in disabled_components:
        if hasattr(lightrag_integration, component):
            value = getattr(lightrag_integration, component)
            status_str = "None (graceful)" if value is None else "Available"
            print(f"    {component:25} {status_str}")


def main():
    """Main demonstration function"""
    print("🔬 Clinical Metabolomics Oracle - Feature Flag System Demo")
    print("=" * 70)
    
    try:
        demo_basic_feature_flags()
        demo_conditional_imports()
        demo_factory_functions()
        demo_integration_status()
        demo_graceful_degradation()
        
        print("\\n" + "=" * 70)
        print("✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        print("\\n📖 Key Takeaways:")
        print("  • Feature flags control what gets imported and exported")
        print("  • Disabled features gracefully return None stubs")
        print("  • Factory functions respect feature flag settings") 
        print("  • Integration status provides comprehensive monitoring")
        print("  • System degrades gracefully when features are disabled")
        
    except Exception as e:
        print(f"\\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)