#!/usr/bin/env python3
"""
Batch Processing Test Infrastructure Validation Script.

This script validates that all components required for comprehensive batch
PDF processing tests are properly installed and configured.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import sys
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Any

def check_python_version() -> Dict[str, Any]:
    """Check Python version compatibility."""
    version_info = sys.version_info
    major, minor = version_info.major, version_info.minor
    
    is_compatible = major == 3 and minor >= 8
    
    return {
        'test': 'Python Version',
        'status': 'PASS' if is_compatible else 'FAIL',
        'details': f"Python {major}.{minor}.{version_info.micro}",
        'requirement': 'Python 3.8+',
        'compatible': is_compatible
    }

def check_required_packages() -> List[Dict[str, Any]]:
    """Check required Python packages."""
    required_packages = [
        ('pytest', 'pytest'),
        ('pytest_asyncio', 'pytest-asyncio'),
        ('asyncio', 'asyncio'),
        ('pathlib', 'pathlib'),
        ('psutil', 'psutil'),
        ('numpy', 'numpy'),
        ('fitz', 'PyMuPDF'),
        ('statistics', 'statistics'),
        ('tempfile', 'tempfile'),
        ('json', 'json'),
        ('logging', 'logging'),
        ('random', 'random'),
        ('time', 'time'),
        ('threading', 'threading'),
        ('concurrent.futures', 'concurrent.futures'),
        ('dataclasses', 'dataclasses'),
        ('typing', 'typing'),
        ('unittest.mock', 'unittest.mock')
    ]
    
    results = []
    
    for module_name, package_name in required_packages:
        try:
            importlib.import_module(module_name)
            status = 'PASS'
            details = f"{package_name} imported successfully"
        except ImportError as e:
            status = 'FAIL'
            details = f"ImportError: {e}"
        
        results.append({
            'test': f'Package: {package_name}',
            'status': status,
            'details': details,
            'module': module_name
        })
    
    return results

def check_test_infrastructure() -> List[Dict[str, Any]]:
    """Check test infrastructure components."""
    test_dir = Path(__file__).parent
    
    infrastructure_files = [
        ('comprehensive_test_fixtures.py', 'Comprehensive test fixtures'),
        ('performance_test_fixtures.py', 'Performance test fixtures'),
        ('conftest.py', 'Pytest configuration'),
        ('test_comprehensive_batch_pdf_processing.py', 'Main batch processing tests'),
        ('run_comprehensive_batch_processing_tests.py', 'Test runner script'),
        ('COMPREHENSIVE_BATCH_PROCESSING_TEST_GUIDE.md', 'Test documentation')
    ]
    
    results = []
    
    for filename, description in infrastructure_files:
        file_path = test_dir / filename
        
        if file_path.exists():
            status = 'PASS'
            details = f"{description} found at {file_path}"
        else:
            status = 'FAIL'
            details = f"{description} missing at {file_path}"
        
        results.append({
            'test': f'File: {filename}',
            'status': status,
            'details': details,
            'path': str(file_path)
        })
    
    return results

def check_lightrag_integration() -> List[Dict[str, Any]]:
    """Check LightRAG integration components."""
    lightrag_dir = Path(__file__).parent.parent
    
    lightrag_components = [
        ('pdf_processor.py', 'PDF processor module'),
        ('clinical_metabolomics_rag.py', 'Clinical metabolomics RAG'),
        ('config.py', 'Configuration module'),
        ('progress_config.py', 'Progress configuration'),
        ('progress_tracker.py', 'Progress tracker')
    ]
    
    results = []
    
    for filename, description in lightrag_components:
        file_path = lightrag_dir / filename
        
        if file_path.exists():
            status = 'PASS'
            details = f"{description} found"
        else:
            status = 'FAIL'
            details = f"{description} missing at {file_path}"
        
        results.append({
            'test': f'LightRAG: {filename}',
            'status': status,
            'details': details,
            'path': str(file_path)
        })
    
    return results

def check_import_capability() -> List[Dict[str, Any]]:
    """Check if test modules can be imported."""
    test_modules = [
        'test_comprehensive_batch_pdf_processing',
        'comprehensive_test_fixtures',
        'performance_test_fixtures'
    ]
    
    results = []
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    for module_name in test_modules:
        try:
            importlib.import_module(module_name)
            status = 'PASS'
            details = f"Successfully imported {module_name}"
        except ImportError as e:
            status = 'FAIL'
            details = f"Import failed: {e}"
        except Exception as e:
            status = 'WARN'
            details = f"Import warning: {e}"
        
        results.append({
            'test': f'Import: {module_name}',
            'status': status,
            'details': details,
            'module': module_name
        })
    
    return results

def check_system_resources() -> Dict[str, Any]:
    """Check system resources for testing."""
    try:
        import psutil
        
        # Get system info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        cpu_count = psutil.cpu_count()
        disk_free_gb = psutil.disk_usage('/').free / (1024**3)
        
        # Check if resources are adequate for testing
        adequate_memory = memory_gb >= 4.0  # 4GB minimum
        adequate_disk = disk_free_gb >= 2.0  # 2GB free space
        adequate_cpu = cpu_count >= 2  # 2+ cores
        
        overall_adequate = adequate_memory and adequate_disk and adequate_cpu
        
        return {
            'test': 'System Resources',
            'status': 'PASS' if overall_adequate else 'WARN',
            'details': f"Memory: {memory_gb:.1f}GB total, {available_gb:.1f}GB available, "
                      f"CPUs: {cpu_count}, Disk: {disk_free_gb:.1f}GB free",
            'adequate': overall_adequate,
            'memory_gb': memory_gb,
            'available_gb': available_gb,
            'cpu_count': cpu_count,
            'disk_free_gb': disk_free_gb
        }
    
    except ImportError:
        return {
            'test': 'System Resources',
            'status': 'FAIL',
            'details': 'psutil not available - cannot check system resources',
            'adequate': False
        }

def run_validation() -> Dict[str, Any]:
    """Run complete infrastructure validation."""
    print("🔍 Validating Batch Processing Test Infrastructure...")
    print("=" * 60)
    
    validation_results = {
        'python_version': check_python_version(),
        'required_packages': check_required_packages(),
        'test_infrastructure': check_test_infrastructure(),
        'lightrag_integration': check_lightrag_integration(),
        'import_capability': check_import_capability(),
        'system_resources': check_system_resources()
    }
    
    # Print results
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    warned_tests = 0
    
    for category, results in validation_results.items():
        if isinstance(results, list):
            # List of test results
            print(f"\n📋 {category.replace('_', ' ').title()}:")
            for result in results:
                status_emoji = {
                    'PASS': '✅',
                    'FAIL': '❌', 
                    'WARN': '⚠️'
                }.get(result['status'], '❓')
                
                print(f"  {status_emoji} {result['test']}: {result['details']}")
                
                total_tests += 1
                if result['status'] == 'PASS':
                    passed_tests += 1
                elif result['status'] == 'FAIL':
                    failed_tests += 1
                elif result['status'] == 'WARN':
                    warned_tests += 1
        
        else:
            # Single test result
            print(f"\n📋 {category.replace('_', ' ').title()}:")
            status_emoji = {
                'PASS': '✅',
                'FAIL': '❌',
                'WARN': '⚠️'
            }.get(results['status'], '❓')
            
            print(f"  {status_emoji} {results['test']}: {results['details']}")
            
            total_tests += 1
            if results['status'] == 'PASS':
                passed_tests += 1
            elif results['status'] == 'FAIL':
                failed_tests += 1
            elif results['status'] == 'WARN':
                warned_tests += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Summary:")
    print(f"  Total Tests: {total_tests}")
    print(f"  ✅ Passed: {passed_tests}")
    print(f"  ❌ Failed: {failed_tests}")
    print(f"  ⚠️  Warnings: {warned_tests}")
    
    # Overall status
    if failed_tests == 0:
        if warned_tests == 0:
            print("\n🎉 Infrastructure validation completed successfully!")
            print("   All components are ready for batch processing tests.")
            overall_status = 'READY'
        else:
            print("\n✅ Infrastructure validation completed with warnings.")
            print("   Most components are ready, but some optimizations may be needed.")
            overall_status = 'READY_WITH_WARNINGS'
    else:
        print("\n❌ Infrastructure validation failed!")
        print("   Some critical components are missing or misconfigured.")
        overall_status = 'NOT_READY'
    
    # Add summary to results
    validation_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'warned_tests': warned_tests,
        'overall_status': overall_status
    }
    
    return validation_results

def print_recommendations(results: Dict[str, Any]):
    """Print recommendations based on validation results."""
    print("\n💡 Recommendations:")
    
    failed_packages = [
        r for r in results['required_packages'] 
        if r['status'] == 'FAIL'
    ]
    
    if failed_packages:
        print("  📦 Install missing packages:")
        for pkg in failed_packages:
            print(f"     pip install {pkg['module']}")
    
    failed_imports = [
        r for r in results['import_capability']
        if r['status'] == 'FAIL'
    ]
    
    if failed_imports:
        print("  🔧 Fix import issues:")
        for imp in failed_imports:
            print(f"     Check module: {imp['module']}")
    
    system_resources = results['system_resources']
    if system_resources['status'] != 'PASS':
        print("  💻 System resources:")
        if not system_resources.get('adequate', False):
            print("     Consider upgrading system resources for optimal performance")
    
    if results['summary']['overall_status'] == 'READY':
        print("  🚀 Ready to run batch processing tests:")
        print("     python run_comprehensive_batch_processing_tests.py --test-level basic")

def main():
    """Main entry point for validation script."""
    try:
        results = run_validation()
        print_recommendations(results)
        
        # Exit with appropriate code
        if results['summary']['overall_status'] == 'NOT_READY':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()