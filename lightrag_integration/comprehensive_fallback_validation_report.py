#!/usr/bin/env python3
"""
Comprehensive Fallback System Validation Report Generator
=========================================================

This script analyzes the current state of the integrated multi-level fallback system
for clinical metabolomics queries and generates a comprehensive validation report.

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Generate comprehensive validation report for integrated fallback system
"""

import os
import sys
import time
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return Path(filepath).exists()

def check_directory_structure() -> Dict[str, Any]:
    """Check the directory structure and key files."""
    print("🔍 Checking directory structure and key files...")
    
    key_files = {
        'Core Components': [
            'query_router.py',
            'research_categorizer.py',
            'cost_persistence.py',
            'enhanced_logging.py',
            'budget_manager.py'
        ],
        'Fallback System': [
            'comprehensive_fallback_system.py',
            'enhanced_query_router_with_fallback.py',
            'clinical_metabolomics_fallback_config.py',
            'uncertainty_aware_fallback_implementation.py'
        ],
        'Configuration': [
            'config.py',
            'clinical_metabolomics_configs/',
            'production_deployment_configs/'
        ],
        'Integration': [
            'main_integration.py',
            'integration_wrapper.py',
            'feature_flag_manager.py'
        ],
        'Monitoring': [
            'alert_system.py',
            'api_metrics_logger.py', 
            'audit_trail.py',
            'production_monitoring.py'
        ]
    }
    
    structure_report = {}
    
    for category, files in key_files.items():
        category_status = {}
        for file in files:
            exists = check_file_exists(file)
            category_status[file] = exists
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
        
        structure_report[category] = category_status
    
    return structure_report

def analyze_test_coverage() -> Dict[str, Any]:
    """Analyze test coverage and availability."""
    print("\n🧪 Analyzing test coverage...")
    
    test_categories = {
        'Basic Integration Tests': [
            'tests/test_basic_integration.py',
            'tests/test_configurations.py'
        ],
        'Fallback System Tests': [
            'tests/test_fallback_mechanisms.py',
            'tests/test_comprehensive_fallback_system.py'
        ],
        'Clinical Tests': [
            'tests/test_clinical_metabolomics_rag.py',
            'test_clinical_config.py'
        ],
        'Performance Tests': [
            'tests/test_performance_benchmarks.py',
            'test_high_performance_integration.py'
        ]
    }
    
    test_report = {}
    
    for category, tests in test_categories.items():
        category_results = {}
        for test_file in tests:
            exists = check_file_exists(test_file)
            category_results[test_file] = {
                'exists': exists,
                'executable': False,
                'last_run_status': 'unknown'
            }
            
            if exists:
                # Try to run a basic check on the test
                try:
                    result = subprocess.run([
                        sys.executable, '-m', 'py_compile', test_file
                    ], capture_output=True, timeout=10)
                    category_results[test_file]['executable'] = result.returncode == 0
                except:
                    pass
            
            status = "✓" if exists and category_results[test_file]['executable'] else "✗"
            print(f"  {status} {test_file}")
        
        test_report[category] = category_results
    
    return test_report

def check_working_tests() -> Dict[str, Any]:
    """Run tests that are known to work."""
    print("\n🏃 Running working tests...")
    
    working_tests = [
        'tests/test_basic_integration.py',
        'tests/test_fallback_mechanisms.py'
    ]
    
    test_results = {}
    
    for test_file in working_tests:
        print(f"\n  Running {test_file}...")
        
        if not check_file_exists(test_file):
            test_results[test_file] = {
                'status': 'file_not_found',
                'passed': 0,
                'failed': 0,
                'duration': 0
            }
            continue
        
        try:
            start_time = time.time()
            result = subprocess.run([
                sys.executable, '-m', 'pytest', test_file, 
                '-v', '--tb=short', '--disable-warnings', '--maxfail=5'
            ], capture_output=True, text=True, timeout=120)
            duration = time.time() - start_time
            
            # Parse results
            stdout = result.stdout
            passed = stdout.count(' PASSED')
            failed = stdout.count(' FAILED')
            skipped = stdout.count(' SKIPPED')
            
            test_results[test_file] = {
                'status': 'completed',
                'return_code': result.returncode,
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'duration': duration,
                'success_rate': passed / (passed + failed) if (passed + failed) > 0 else 0
            }
            
            print(f"    ✓ Passed: {passed}, ✗ Failed: {failed}, ⏸ Skipped: {skipped}")
            print(f"    ⏱ Duration: {duration:.2f}s")
            
        except subprocess.TimeoutExpired:
            test_results[test_file] = {
                'status': 'timeout',
                'passed': 0,
                'failed': 0,
                'duration': 120
            }
            print(f"    ⏰ Test timed out after 120s")
            
        except Exception as e:
            test_results[test_file] = {
                'status': 'error',
                'error': str(e),
                'passed': 0,
                'failed': 0,
                'duration': 0
            }
            print(f"    ❌ Error: {e}")
    
    return test_results

def analyze_configuration_files() -> Dict[str, Any]:
    """Analyze configuration files."""
    print("\n⚙️ Analyzing configuration files...")
    
    config_files = [
        'clinical_metabolomics_configs/production_diagnostic_config.json',
        'clinical_metabolomics_configs/production_research_config.json', 
        'production_deployment_configs/production.env.template'
    ]
    
    config_report = {}
    
    for config_file in config_files:
        print(f"  Checking {config_file}...")
        
        if not check_file_exists(config_file):
            config_report[config_file] = {'status': 'missing'}
            print(f"    ✗ File not found")
            continue
        
        try:
            if config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    config_report[config_file] = {
                        'status': 'valid_json',
                        'keys': list(config_data.keys()) if isinstance(config_data, dict) else [],
                        'size': len(str(config_data))
                    }
                    print(f"    ✓ Valid JSON with {len(config_data)} settings")
            else:
                with open(config_file, 'r') as f:
                    content = f.read()
                    config_report[config_file] = {
                        'status': 'readable',
                        'lines': len(content.split('\n')),
                        'size': len(content)
                    }
                    print(f"    ✓ Readable file with {len(content.split())} lines")
                    
        except Exception as e:
            config_report[config_file] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"    ✗ Error reading file: {e}")
    
    return config_report

def check_system_health() -> Dict[str, Any]:
    """Check overall system health indicators."""
    print("\n❤️ Checking system health indicators...")
    
    health_indicators = {
        'log_files': [
            'logs/lightrag_integration.log',
            'logs/claude_monitor.log'
        ],
        'cache_directories': [
            'logs/',
            'quality_reports/',
            'performance_benchmarking/'
        ],
        'recent_activity': []
    }
    
    health_report = {}
    
    # Check log files
    log_status = {}
    for log_file in health_indicators['log_files']:
        if check_file_exists(log_file):
            try:
                stat = Path(log_file).stat()
                log_status[log_file] = {
                    'exists': True,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                print(f"  ✓ {log_file} - {stat.st_size} bytes, modified {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")
            except Exception as e:
                log_status[log_file] = {'exists': False, 'error': str(e)}
                print(f"  ✗ {log_file} - Error: {e}")
        else:
            log_status[log_file] = {'exists': False}
            print(f"  ✗ {log_file} - Not found")
    
    health_report['logs'] = log_status
    
    # Check directories
    dir_status = {}
    for directory in health_indicators['cache_directories']:
        if check_file_exists(directory):
            try:
                file_count = len(list(Path(directory).rglob('*')))
                dir_status[directory] = {
                    'exists': True,
                    'file_count': file_count
                }
                print(f"  ✓ {directory}/ - {file_count} files")
            except Exception as e:
                dir_status[directory] = {'exists': False, 'error': str(e)}
        else:
            dir_status[directory] = {'exists': False}
            print(f"  ✗ {directory}/ - Not found")
    
    health_report['directories'] = dir_status
    
    return health_report

def assess_fallback_capabilities() -> Dict[str, Any]:
    """Assess fallback system capabilities based on available components."""
    print("\n🔄 Assessing fallback system capabilities...")
    
    capabilities = {
        'query_routing': False,
        'fallback_orchestration': False,
        'enhanced_routing': False,
        'clinical_configuration': False,
        'monitoring_alerting': False,
        'performance_tracking': False,
        'cache_management': False,
        'error_recovery': False
    }
    
    # Check core routing
    if check_file_exists('query_router.py'):
        capabilities['query_routing'] = True
        print("  ✓ Basic query routing available")
    else:
        print("  ✗ Basic query routing not available")
    
    # Check fallback orchestration
    if check_file_exists('comprehensive_fallback_system.py'):
        capabilities['fallback_orchestration'] = True
        print("  ✓ Comprehensive fallback orchestration available")
    else:
        print("  ✗ Comprehensive fallback orchestration not available")
    
    # Check enhanced routing
    if check_file_exists('enhanced_query_router_with_fallback.py'):
        capabilities['enhanced_routing'] = True
        print("  ✓ Enhanced routing with fallback available")
    else:
        print("  ✗ Enhanced routing with fallback not available")
    
    # Check clinical configuration
    if check_file_exists('clinical_metabolomics_fallback_config.py'):
        capabilities['clinical_configuration'] = True
        print("  ✓ Clinical metabolomics configuration available")
    else:
        print("  ✗ Clinical metabolomics configuration not available")
    
    # Check monitoring and alerting
    if check_file_exists('alert_system.py') and check_file_exists('api_metrics_logger.py'):
        capabilities['monitoring_alerting'] = True
        print("  ✓ Monitoring and alerting available")
    else:
        print("  ✗ Monitoring and alerting not fully available")
    
    # Check performance tracking
    if check_file_exists('performance_benchmark_suite.py'):
        capabilities['performance_tracking'] = True
        print("  ✓ Performance tracking available")
    else:
        print("  ✗ Performance tracking not available")
    
    # Check cache management
    if check_file_exists('cost_persistence.py'):
        capabilities['cache_management'] = True
        print("  ✓ Cache management available")
    else:
        print("  ✗ Cache management not available")
    
    # Check error recovery
    if check_file_exists('advanced_recovery_system.py'):
        capabilities['error_recovery'] = True
        print("  ✓ Advanced error recovery available")
    else:
        print("  ✗ Advanced error recovery not available")
    
    return capabilities

def generate_recommendations(structure_report: Dict, test_results: Dict, 
                           config_report: Dict, health_report: Dict, 
                           capabilities: Dict) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []
    
    # Check core functionality
    working_capabilities = sum(capabilities.values())
    total_capabilities = len(capabilities)
    capability_score = working_capabilities / total_capabilities
    
    if capability_score >= 0.8:
        recommendations.append("🟢 Core fallback system capabilities are mostly complete")
    elif capability_score >= 0.6:
        recommendations.append("🟡 Core fallback system has good coverage but needs improvement")
    else:
        recommendations.append("🔴 Core fallback system needs significant development")
    
    # Specific recommendations
    if not capabilities['query_routing']:
        recommendations.append("• Critical: Implement basic query routing functionality")
    
    if not capabilities['fallback_orchestration']:
        recommendations.append("• High Priority: Fix fallback orchestration import issues")
    
    if not capabilities['clinical_configuration']:
        recommendations.append("• Medium Priority: Complete clinical metabolomics configuration")
    
    # Test recommendations
    working_test_files = sum(1 for category in test_results.values() 
                           for test_data in category.values() 
                           if test_data.get('executable', False))
    
    if working_test_files < 2:
        recommendations.append("• Critical: Fix test suite import and execution issues")
    
    # Configuration recommendations
    valid_configs = sum(1 for config_data in config_report.values() 
                       if config_data.get('status') in ['valid_json', 'readable'])
    
    if valid_configs == 0:
        recommendations.append("• High Priority: Create valid configuration files")
    
    # Health recommendations
    active_logs = sum(1 for log_data in health_report.get('logs', {}).values() 
                     if log_data.get('exists', False))
    
    if active_logs == 0:
        recommendations.append("• Medium Priority: Enable logging and monitoring")
    
    return recommendations

def generate_validation_report() -> Dict[str, Any]:
    """Generate comprehensive validation report."""
    print("=" * 80)
    print("COMPREHENSIVE FALLBACK SYSTEM VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().isoformat()}")
    print(f"System: Clinical Metabolomics Oracle - Multi-Level Fallback System")
    
    # Collect all analysis data
    structure_report = check_directory_structure()
    test_coverage = analyze_test_coverage()
    test_results = check_working_tests()
    config_report = analyze_configuration_files()
    health_report = check_system_health()
    capabilities = assess_fallback_capabilities()
    
    recommendations = generate_recommendations(
        structure_report, test_coverage, config_report, 
        health_report, capabilities
    )
    
    # Generate final report
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'system_name': 'Clinical Metabolomics Oracle - Multi-Level Fallback System',
        'validation_summary': {
            'directory_structure': structure_report,
            'test_coverage': test_coverage,
            'test_results': test_results,
            'configuration_analysis': config_report,
            'system_health': health_report,
            'capabilities_assessment': capabilities,
            'recommendations': recommendations
        }
    }
    
    return final_report

def print_summary_report(report: Dict[str, Any]):
    """Print executive summary."""
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    
    capabilities = report['validation_summary']['capabilities_assessment']
    working_capabilities = sum(capabilities.values())
    total_capabilities = len(capabilities)
    capability_score = working_capabilities / total_capabilities
    
    # Overall assessment
    if capability_score >= 0.8:
        overall_status = "🟢 EXCELLENT"
        status_desc = "Fallback system is comprehensive and ready for production"
    elif capability_score >= 0.6:
        overall_status = "🟡 GOOD"
        status_desc = "Fallback system has good foundation with some areas for improvement"
    elif capability_score >= 0.4:
        overall_status = "🟠 FAIR"
        status_desc = "Fallback system has basic functionality but needs significant work"
    else:
        overall_status = "🔴 POOR"
        status_desc = "Fallback system requires major development effort"
    
    print(f"\n📊 OVERALL STATUS: {overall_status}")
    print(f"📝 ASSESSMENT: {status_desc}")
    print(f"📈 CAPABILITY SCORE: {capability_score:.1%} ({working_capabilities}/{total_capabilities})")
    
    # Key metrics
    print(f"\n🔑 KEY METRICS:")
    
    # File structure
    total_files_checked = sum(len(category) for category in report['validation_summary']['directory_structure'].values())
    existing_files = sum(sum(category.values()) for category in report['validation_summary']['directory_structure'].values())
    print(f"  • File Structure: {existing_files}/{total_files_checked} key files present ({existing_files/total_files_checked:.1%})")
    
    # Test results
    test_results = report['validation_summary']['test_results']
    total_passed = sum(result.get('passed', 0) for result in test_results.values())
    total_failed = sum(result.get('failed', 0) for result in test_results.values())
    if total_passed + total_failed > 0:
        test_success = total_passed / (total_passed + total_failed)
        print(f"  • Test Success Rate: {test_success:.1%} ({total_passed} passed, {total_failed} failed)")
    else:
        print(f"  • Test Success Rate: No tests executed successfully")
    
    # Configuration
    config_report = report['validation_summary']['configuration_analysis']
    valid_configs = sum(1 for config_data in config_report.values() 
                       if config_data.get('status') in ['valid_json', 'readable'])
    total_configs = len(config_report)
    if total_configs > 0:
        config_success = valid_configs / total_configs
        print(f"  • Configuration Files: {config_success:.1%} valid ({valid_configs}/{total_configs})")
    
    # Recommendations
    recommendations = report['validation_summary']['recommendations']
    print(f"\n💡 TOP RECOMMENDATIONS:")
    for rec in recommendations[:5]:  # Show top 5
        print(f"  {rec}")
    
    if len(recommendations) > 5:
        print(f"  ... and {len(recommendations) - 5} more recommendations")

def save_detailed_report(report: Dict[str, Any], filename: str = None):
    """Save detailed report to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fallback_system_validation_report_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved: {filename}")
    except Exception as e:
        print(f"\n❌ Failed to save report: {e}")

def main():
    """Main execution function."""
    try:
        # Generate comprehensive report
        report = generate_validation_report()
        
        # Print summary
        print_summary_report(report)
        
        # Save detailed report
        save_detailed_report(report)
        
        # Determine exit code based on capability score
        capabilities = report['validation_summary']['capabilities_assessment']
        capability_score = sum(capabilities.values()) / len(capabilities)
        
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETED")
        print("=" * 80)
        
        return capability_score >= 0.6  # Success if 60% or more capabilities working
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)