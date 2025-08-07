#!/usr/bin/env python3
"""
Validation Script for Comprehensive Error Handling Tests.

This script validates that all test files can be imported and basic
test discovery works correctly. It's useful for CI/CD validation
before running the full test suite.

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import sys
import importlib
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_imports():
    """Validate that all required modules can be imported."""
    print("Validating error handling component imports...")
    
    try:
        # Core error classes
        from lightrag_integration.clinical_metabolomics_rag import (
            ClinicalMetabolomicsRAGError, IngestionError, IngestionRetryableError,
            IngestionNonRetryableError, IngestionResourceError, IngestionNetworkError,
            IngestionAPIError, StorageInitializationError, StoragePermissionError,
            StorageSpaceError, StorageDirectoryError, StorageRetryableError,
            CircuitBreakerError, CircuitBreaker, RateLimiter
        )
        print("  ✓ Core error classes imported successfully")
        
        # Advanced recovery system
        from lightrag_integration.advanced_recovery_system import (
            AdvancedRecoverySystem, DegradationMode, FailureType, BackoffStrategy,
            ResourceThresholds, DegradationConfig, CheckpointData, CheckpointManager,
            SystemResourceMonitor, AdaptiveBackoffCalculator
        )
        print("  ✓ Advanced recovery system imported successfully")
        
        # Enhanced logging
        from lightrag_integration.enhanced_logging import (
            EnhancedLogger, IngestionLogger, DiagnosticLogger, CorrelationIDManager,
            CorrelationContext, StructuredLogRecord, PerformanceMetrics, PerformanceTracker,
            correlation_manager, performance_logged
        )
        print("  ✓ Enhanced logging components imported successfully")
        
        # Progress tracking
        from lightrag_integration.unified_progress_tracker import (
            KnowledgeBaseProgressTracker, KnowledgeBasePhase
        )
        print("  ✓ Progress tracking components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False

def validate_test_files():
    """Validate that test files exist and can be imported."""
    print("\nValidating test file structure...")
    
    test_files = [
        "test_comprehensive_error_handling.py",
        "test_storage_error_handling_comprehensive.py",
        "test_advanced_recovery_edge_cases.py"
    ]
    
    base_dir = Path(__file__).parent
    
    for test_file in test_files:
        test_path = base_dir / test_file
        
        if not test_path.exists():
            print(f"  ✗ Test file not found: {test_file}")
            return False
        
        # Try to parse the file
        try:
            import ast
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            print(f"  ✓ {test_file} - syntax valid")
        except SyntaxError as e:
            print(f"  ✗ {test_file} - syntax error: {e}")
            return False
        except Exception as e:
            print(f"  ✗ {test_file} - error: {e}")
            return False
    
    return True

def validate_pytest_discovery():
    """Validate that pytest can discover tests."""
    print("\nValidating pytest test discovery...")
    
    try:
        import subprocess
        import sys
        
        # Run pytest --collect-only to validate test discovery
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--collect-only",
            "-q",
            str(Path(__file__).parent)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Count discovered tests
            lines = result.stdout.split('\n')
            test_count = sum(1 for line in lines if '::test_' in line)
            print(f"  ✓ Pytest discovered {test_count} tests successfully")
            return True
        else:
            print(f"  ✗ Pytest collection failed:")
            print(f"    stdout: {result.stdout}")
            print(f"    stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ✗ Pytest collection timed out")
        return False
    except FileNotFoundError:
        print("  ✗ Pytest not found - ensure pytest is installed")
        return False
    except Exception as e:
        print(f"  ✗ Pytest validation error: {e}")
        return False

def validate_basic_functionality():
    """Validate basic functionality of error handling components."""
    print("\nValidating basic error handling functionality...")
    
    try:
        # Test error creation
        from lightrag_integration.clinical_metabolomics_rag import IngestionAPIError
        error = IngestionAPIError("Test error", status_code=500)
        assert str(error) == "Test error"
        assert error.status_code == 500
        print("  ✓ Error creation works correctly")
        
        # Test backoff calculation
        from lightrag_integration.advanced_recovery_system import (
            AdaptiveBackoffCalculator, FailureType, BackoffStrategy
        )
        calculator = AdaptiveBackoffCalculator()
        delay = calculator.calculate_backoff(
            FailureType.API_ERROR, 1, BackoffStrategy.EXPONENTIAL
        )
        assert delay > 0
        print("  ✓ Backoff calculation works correctly")
        
        # Test resource monitoring
        from lightrag_integration.advanced_recovery_system import SystemResourceMonitor
        monitor = SystemResourceMonitor()
        resources = monitor.get_current_resources()
        assert isinstance(resources, dict)
        print("  ✓ Resource monitoring works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic functionality error: {e}")
        return False

def main():
    """Main validation function."""
    print("=" * 60)
    print("ERROR HANDLING TEST SUITE VALIDATION")
    print("=" * 60)
    
    all_passed = True
    
    # Run all validations
    validations = [
        ("Import validation", validate_imports),
        ("Test file validation", validate_test_files),
        ("Pytest discovery validation", validate_pytest_discovery),
        ("Basic functionality validation", validate_basic_functionality)
    ]
    
    for name, validation_func in validations:
        try:
            passed = validation_func()
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  ✗ {name} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("\nThe error handling test suite is ready to run.")
        print("Execute: python run_comprehensive_error_handling_tests.py")
        sys.exit(0)
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("\nPlease fix the issues above before running the test suite.")
        sys.exit(1)

if __name__ == "__main__":
    main()