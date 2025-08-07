#!/usr/bin/env python3
"""
Infrastructure Validation Script for PDF Error Handling Test Suite.

This script validates that all required components and dependencies are available
for running the comprehensive PDF error handling test suite.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import importlib.util

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class TestInfrastructureValidator:
    """Validates test infrastructure components and dependencies."""
    
    def __init__(self):
        self.validation_results = {}
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging for validation."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def validate_python_dependencies(self) -> Dict[str, Any]:
        """Validate required Python packages are available."""
        required_packages = [
            'pytest',
            'pytest_asyncio', 
            'fitz',  # PyMuPDF
            'psutil',
            'asyncio',
            'pathlib',
            'unittest.mock',
            'concurrent.futures',
            'threading',
            'tempfile',
            'json',
            'time',
            'logging',
            'gc'
        ]
        
        results = {
            'available': [],
            'missing': [],
            'versions': {}
        }
        
        for package in required_packages:
            try:
                if package == 'fitz':
                    import fitz
                    results['available'].append(package)
                    try:
                        results['versions'][package] = fitz.version[0]
                    except:
                        results['versions'][package] = 'unknown'
                elif package == 'pytest_asyncio':
                    import pytest_asyncio
                    results['available'].append(package)
                    try:
                        results['versions'][package] = pytest_asyncio.__version__
                    except:
                        results['versions'][package] = 'unknown'
                else:
                    module = importlib.import_module(package)
                    results['available'].append(package)
                    try:
                        results['versions'][package] = getattr(module, '__version__', 'unknown')
                    except:
                        results['versions'][package] = 'unknown'
                        
            except ImportError:
                results['missing'].append(package)
        
        return results
    
    def validate_core_components(self) -> Dict[str, Any]:
        """Validate core system components are available."""
        components = [
            ('pdf_processor', 'lightrag_integration.pdf_processor'),
            ('clinical_rag', 'lightrag_integration.clinical_metabolomics_rag'),
            ('recovery_system', 'lightrag_integration.advanced_recovery_system'),
            ('enhanced_logging', 'lightrag_integration.enhanced_logging'),
            ('progress_tracker', 'lightrag_integration.unified_progress_tracker')
        ]
        
        results = {
            'available': [],
            'missing': [],
            'import_errors': {}
        }
        
        for component_name, module_path in components:
            try:
                module = importlib.import_module(module_path)
                results['available'].append(component_name)
            except ImportError as e:
                results['missing'].append(component_name)
                results['import_errors'][component_name] = str(e)
        
        return results
    
    def validate_test_fixtures(self) -> Dict[str, Any]:
        """Validate test fixture components are available."""
        test_modules = [
            'comprehensive_test_fixtures',
            'biomedical_test_fixtures',
            'query_test_fixtures',
            'performance_test_fixtures',
            'validation_fixtures'
        ]
        
        results = {
            'available': [],
            'missing': [],
            'import_errors': {}
        }
        
        test_dir = Path(__file__).parent
        
        for module_name in test_modules:
            module_path = test_dir / f"{module_name}.py"
            
            if module_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    results['available'].append(module_name)
                except Exception as e:
                    results['missing'].append(module_name)
                    results['import_errors'][module_name] = str(e)
            else:
                results['missing'].append(module_name)
                results['import_errors'][module_name] = "Module file not found"
        
        return results
    
    def validate_test_files(self) -> Dict[str, Any]:
        """Validate test file structure and accessibility."""
        test_dir = Path(__file__).parent
        
        required_files = [
            'test_pdf_processing_error_handling_comprehensive.py',
            'run_pdf_error_handling_tests.py',
            'conftest.py',
            'pytest.ini'
        ]
        
        results = {
            'available': [],
            'missing': [],
            'file_info': {}
        }
        
        for filename in required_files:
            file_path = test_dir / filename
            
            if file_path.exists():
                results['available'].append(filename)
                results['file_info'][filename] = {
                    'size_bytes': file_path.stat().st_size,
                    'readable': file_path.is_file() and file_path.stat().st_mode & 0o444,
                    'executable': bool(file_path.stat().st_mode & 0o111) if filename.endswith('.py') else False
                }
            else:
                results['missing'].append(filename)
        
        return results
    
    def validate_system_resources(self) -> Dict[str, Any]:
        """Validate system resources for test execution."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            results = {
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'percent_used': memory.percent,
                    'sufficient': memory.available > 2 * (1024**3)  # At least 2GB available
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'percent_used': round((disk.used / disk.total) * 100, 2),
                    'sufficient': disk.free > 1 * (1024**3)  # At least 1GB free
                },
                'cpu_count': psutil.cpu_count(),
                'system_suitable': True
            }
            
            # Overall system suitability
            results['system_suitable'] = (
                results['memory']['sufficient'] and 
                results['disk']['sufficient'] and
                results['cpu_count'] >= 2
            )
            
            return results
            
        except ImportError:
            return {
                'error': 'psutil not available - cannot validate system resources',
                'system_suitable': False
            }
    
    def validate_pytest_configuration(self) -> Dict[str, Any]:
        """Validate pytest configuration and async support."""
        test_dir = Path(__file__).parent
        pytest_ini = test_dir / 'pytest.ini'
        
        results = {
            'config_file_exists': pytest_ini.exists(),
            'async_support': False,
            'required_plugins': [],
            'missing_plugins': [],
            'configuration_valid': False
        }
        
        if pytest_ini.exists():
            try:
                with open(pytest_ini, 'r') as f:
                    config_content = f.read()
                
                # Check for async support
                if 'asyncio' in config_content:
                    results['async_support'] = True
                
                results['configuration_valid'] = True
                
            except Exception as e:
                results['config_error'] = str(e)
        
        # Check for required pytest plugins
        required_plugins = ['pytest-asyncio']
        
        for plugin in required_plugins:
            try:
                importlib.import_module(plugin.replace('-', '_'))
                results['required_plugins'].append(plugin)
            except ImportError:
                results['missing_plugins'].append(plugin)
        
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all components."""
        self.logger.info("Starting comprehensive test infrastructure validation...")
        
        validation_results = {
            'timestamp': Path(__file__).parent.parent.parent.name,  # Use directory name as timestamp
            'overall_status': 'UNKNOWN',
            'component_results': {}
        }
        
        # Run all validations
        validations = [
            ('python_dependencies', self.validate_python_dependencies),
            ('core_components', self.validate_core_components),
            ('test_fixtures', self.validate_test_fixtures),
            ('test_files', self.validate_test_files),
            ('system_resources', self.validate_system_resources),
            ('pytest_configuration', self.validate_pytest_configuration)
        ]
        
        all_passed = True
        
        for validation_name, validation_func in validations:
            self.logger.info(f"Validating {validation_name}...")
            
            try:
                result = validation_func()
                validation_results['component_results'][validation_name] = result
                
                # Check if this validation passed
                if validation_name == 'python_dependencies':
                    passed = len(result['missing']) == 0
                elif validation_name == 'core_components':
                    passed = len(result['missing']) == 0
                elif validation_name == 'test_fixtures':
                    passed = len(result['missing']) <= 1  # Allow some optional fixtures to be missing
                elif validation_name == 'test_files':
                    passed = len(result['missing']) == 0
                elif validation_name == 'system_resources':
                    passed = result.get('system_suitable', False)
                elif validation_name == 'pytest_configuration':
                    passed = result.get('configuration_valid', False)
                else:
                    passed = True
                
                if not passed:
                    all_passed = False
                    self.logger.warning(f"{validation_name} validation failed")
                else:
                    self.logger.info(f"{validation_name} validation passed")
                    
            except Exception as e:
                self.logger.error(f"Error during {validation_name} validation: {e}")
                validation_results['component_results'][validation_name] = {
                    'error': str(e),
                    'status': 'FAILED'
                }
                all_passed = False
        
        # Set overall status
        validation_results['overall_status'] = 'PASSED' if all_passed else 'FAILED'
        
        return validation_results
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        report_lines = [
            "="*80,
            "PDF ERROR HANDLING TEST INFRASTRUCTURE VALIDATION REPORT",
            "="*80,
            f"Overall Status: {results['overall_status']}",
            f"Validation Time: {results['timestamp']}",
            ""
        ]
        
        for component_name, component_result in results['component_results'].items():
            report_lines.extend([
                f"Component: {component_name.upper()}",
                "-" * 40
            ])
            
            if 'error' in component_result:
                report_lines.append(f"  STATUS: FAILED - {component_result['error']}")
            elif component_name == 'python_dependencies':
                report_lines.extend([
                    f"  Available packages: {len(component_result['available'])}",
                    f"  Missing packages: {len(component_result['missing'])}",
                ])
                if component_result['missing']:
                    report_lines.append(f"  Missing: {', '.join(component_result['missing'])}")
                    
            elif component_name == 'core_components':
                report_lines.extend([
                    f"  Available components: {len(component_result['available'])}",
                    f"  Missing components: {len(component_result['missing'])}"
                ])
                if component_result['missing']:
                    report_lines.append(f"  Missing: {', '.join(component_result['missing'])}")
                    
            elif component_name == 'test_fixtures':
                report_lines.extend([
                    f"  Available fixtures: {len(component_result['available'])}",
                    f"  Missing fixtures: {len(component_result['missing'])}"
                ])
                
            elif component_name == 'test_files':
                report_lines.extend([
                    f"  Available files: {len(component_result['available'])}",
                    f"  Missing files: {len(component_result['missing'])}"
                ])
                if component_result['missing']:
                    report_lines.append(f"  Missing: {', '.join(component_result['missing'])}")
                    
            elif component_name == 'system_resources':
                if 'error' not in component_result:
                    report_lines.extend([
                        f"  Memory: {component_result['memory']['available_gb']}GB available",
                        f"  Disk: {component_result['disk']['free_gb']}GB free",
                        f"  CPU cores: {component_result['cpu_count']}",
                        f"  System suitable: {component_result['system_suitable']}"
                    ])
                    
            elif component_name == 'pytest_configuration':
                report_lines.extend([
                    f"  Config file exists: {component_result['config_file_exists']}",
                    f"  Async support: {component_result['async_support']}",
                    f"  Configuration valid: {component_result['configuration_valid']}"
                ])
            
            report_lines.append("")
        
        # Add recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 40
        ])
        
        if results['overall_status'] == 'PASSED':
            report_lines.append("  All validations passed. Test infrastructure is ready.")
        else:
            report_lines.append("  Some validations failed. Address the following issues:")
            
            for component_name, component_result in results['component_results'].items():
                if 'missing' in component_result and component_result['missing']:
                    report_lines.append(f"  - Install missing {component_name}: {', '.join(component_result['missing'])}")
                if 'error' in component_result:
                    report_lines.append(f"  - Fix {component_name} error: {component_result['error']}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)


def main():
    """Main validation execution."""
    validator = TestInfrastructureValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Generate and display report
    report = validator.generate_validation_report(results)
    print(report)
    
    # Save report to file
    report_file = Path(__file__).parent / 'infrastructure_validation_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed validation report saved to: {report_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_status'] == 'PASSED' else 1
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)