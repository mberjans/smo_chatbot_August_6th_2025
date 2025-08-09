#!/usr/bin/env python3
"""
Multi-Level Fallback Test Runner for Clinical Metabolomics Oracle
================================================================

This script runs comprehensive tests for the multi-level fallback system
(LightRAG → Perplexity → Cache) with detailed reporting and validation.

Features:
- Automated test execution with proper environment setup
- Performance benchmarking and analysis
- Detailed HTML and XML reporting
- Test categorization and selective execution
- Integration with existing CI/CD pipelines
- Comprehensive logging and analytics

Author: Claude Code (Anthropic)
Task: CMO-LIGHTRAG-014-T01-TEST - Test Runner
Created: August 9, 2025
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'fallback_test_runner.log')
    ]
)
logger = logging.getLogger(__name__)


class FallbackTestRunner:
    """Comprehensive test runner for multi-level fallback scenarios."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Test categories with their markers
        self.test_categories = {
            'core_fallback': ['fallback', 'multi_level'],
            'performance': ['performance', 'stress'],
            'integration': ['integration', 'production'],
            'edge_cases': ['edge_case', 'simulation'],
            'monitoring': ['monitoring', 'recovery'],
            'all': []  # Empty means all tests
        }
        
        # Ensure required directories exist
        self._setup_directories()
    
    def _setup_directories(self):
        """Set up required directories for test execution."""
        required_dirs = [
            self.project_root / 'logs',
            self.project_root / 'reports',
            self.project_root / 'tests' / 'temp',
            self.project_root / 'logs' / 'alerts',
            self.project_root / 'logs' / 'routing_decisions'
        ]
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def check_environment(self) -> Dict[str, Any]:
        """Check test environment and dependencies."""
        logger.info("Checking test environment...")
        
        environment_status = {
            'python_version': sys.version,
            'working_directory': str(self.project_root),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error(f"Python 3.8+ required, found {sys.version_info}")
            environment_status['python_version_ok'] = False
        else:
            environment_status['python_version_ok'] = True
        
        # Check required modules
        required_modules = [
            'pytest',
            'pytest-asyncio', 
            'pytest-cov',
            'pytest-html',
            'pytest-timeout'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module.replace('-', '_'))
                logger.debug(f"Module {module} is available")
            except ImportError:
                missing_modules.append(module)
                logger.warning(f"Module {module} not found")
        
        environment_status['missing_modules'] = missing_modules
        environment_status['all_modules_available'] = len(missing_modules) == 0
        
        # Check project structure
        required_paths = [
            self.project_root / 'tests' / 'test_multi_level_fallback_scenarios.py',
            self.project_root / 'tests' / 'test_fallback_test_config.py',
            self.project_root / 'tests' / 'pytest_fallback_scenarios.ini',
            self.project_root / 'lightrag_integration'
        ]
        
        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))
                logger.warning(f"Required path not found: {path}")
            else:
                logger.debug(f"Required path exists: {path}")
        
        environment_status['missing_paths'] = missing_paths
        environment_status['project_structure_ok'] = len(missing_paths) == 0
        
        # Overall environment status
        environment_status['ready_for_testing'] = (
            environment_status['python_version_ok'] and
            environment_status['all_modules_available'] and
            environment_status['project_structure_ok']
        )
        
        return environment_status
    
    def run_test_category(self, category: str, additional_args: List[str] = None) -> Dict[str, Any]:
        """Run tests for a specific category."""
        logger.info(f"Running test category: {category}")
        
        if category not in self.test_categories:
            raise ValueError(f"Unknown test category: {category}. Available: {list(self.test_categories.keys())}")
        
        # Build pytest command
        pytest_cmd = [
            sys.executable, '-m', 'pytest',
            '-c', str(self.project_root / 'tests' / 'pytest_fallback_scenarios.ini')
        ]
        
        # Add markers for category
        markers = self.test_categories[category]
        if markers:
            marker_expression = ' or '.join(markers)
            pytest_cmd.extend(['-m', marker_expression])
        
        # Add test file path
        test_file = self.project_root / 'tests' / 'test_multi_level_fallback_scenarios.py'
        pytest_cmd.append(str(test_file))
        
        # Add additional arguments
        if additional_args:
            pytest_cmd.extend(additional_args)
        
        # Set up environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root)
        env['FALLBACK_TEST_MODE'] = 'true'
        
        logger.info(f"Executing command: {' '.join(pytest_cmd)}")
        
        # Run tests
        start_time = time.time()
        try:
            result = subprocess.run(
                pytest_cmd,
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            test_result = {
                'category': category,
                'command': ' '.join(pytest_cmd),
                'return_code': result.returncode,
                'execution_time_seconds': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Parse test results if successful
            if result.returncode == 0:
                test_result['status'] = 'PASSED'
                logger.info(f"✅ {category} tests PASSED in {execution_time:.2f}s")
            else:
                test_result['status'] = 'FAILED'
                logger.error(f"❌ {category} tests FAILED in {execution_time:.2f}s")
                logger.error(f"Error output: {result.stderr}")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"⏰ {category} tests TIMED OUT after {execution_time:.2f}s")
            
            return {
                'category': category,
                'status': 'TIMEOUT',
                'execution_time_seconds': execution_time,
                'error': 'Test execution timed out',
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"💥 {category} tests CRASHED: {e}")
            
            return {
                'category': category,
                'status': 'CRASHED',
                'execution_time_seconds': execution_time,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_tests(self, categories: List[str] = None, additional_args: List[str] = None) -> Dict[str, Any]:
        """Run all specified test categories."""
        self.start_time = time.time()
        logger.info("🚀 Starting comprehensive fallback test execution")
        
        if categories is None:
            categories = ['core_fallback', 'performance', 'integration', 'edge_cases']
        
        # Check environment first
        env_status = self.check_environment()
        if not env_status['ready_for_testing']:
            logger.error("❌ Environment not ready for testing")
            return {
                'status': 'ENVIRONMENT_ERROR',
                'environment_check': env_status,
                'total_time_seconds': time.time() - self.start_time
            }
        
        logger.info("✅ Environment check passed")
        
        # Run test categories
        category_results = {}
        total_categories = len(categories)
        
        for i, category in enumerate(categories, 1):
            logger.info(f"📋 Running category {i}/{total_categories}: {category}")
            
            try:
                result = self.run_test_category(category, additional_args)
                category_results[category] = result
                
                # Short break between categories to prevent resource exhaustion
                if i < total_categories:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Failed to run category {category}: {e}")
                category_results[category] = {
                    'category': category,
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        # Calculate summary statistics
        passed_categories = [cat for cat, result in category_results.items() 
                           if result.get('status') == 'PASSED']
        failed_categories = [cat for cat, result in category_results.items() 
                           if result.get('status') != 'PASSED']
        
        success_rate = len(passed_categories) / len(categories) if categories else 0
        
        # Compile final results
        final_results = {
            'test_execution_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat(),
                'total_execution_time_seconds': total_time,
                'categories_requested': categories,
                'total_categories': total_categories,
                'passed_categories': passed_categories,
                'failed_categories': failed_categories,
                'success_rate': success_rate,
                'overall_status': 'PASSED' if success_rate == 1.0 else 'FAILED'
            },
            'environment_check': env_status,
            'category_results': category_results,
            'recommendations': self._generate_recommendations(category_results)
        }
        
        # Log summary
        if success_rate == 1.0:
            logger.info(f"🎉 ALL TESTS PASSED! ({len(passed_categories)}/{total_categories}) in {total_time:.2f}s")
        else:
            logger.warning(f"⚠️  PARTIAL SUCCESS: {len(passed_categories)}/{total_categories} categories passed in {total_time:.2f}s")
            for failed_cat in failed_categories:
                logger.error(f"   ❌ Failed: {failed_cat}")
        
        return final_results
    
    def _generate_recommendations(self, category_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for category, result in category_results.items():
            if result.get('status') != 'PASSED':
                if result.get('status') == 'TIMEOUT':
                    recommendations.append(
                        f"Consider optimizing {category} tests for better performance "
                        f"or increasing timeout limits"
                    )
                elif result.get('status') == 'FAILED':
                    recommendations.append(
                        f"Review {category} test failures and fix underlying issues"
                    )
                elif result.get('status') == 'CRASHED':
                    recommendations.append(
                        f"Investigate {category} test crashes - may indicate "
                        f"environment or dependency issues"
                    )
                
                # Check execution time
                exec_time = result.get('execution_time_seconds', 0)
                if exec_time > 300:  # 5 minutes
                    recommendations.append(
                        f"{category} tests are taking too long ({exec_time:.1f}s) - "
                        f"consider parallelization or optimization"
                    )
        
        # General recommendations
        passed_count = sum(1 for r in category_results.values() if r.get('status') == 'PASSED')
        if passed_count == len(category_results):
            recommendations.append(
                "All fallback tests passed! The multi-level fallback system "
                "is functioning correctly and ready for production deployment."
            )
        elif passed_count > 0:
            recommendations.append(
                "Some fallback tests passed but issues remain. "
                "Prioritize fixing failed test categories before deployment."
            )
        else:
            recommendations.append(
                "All fallback tests failed. This indicates serious issues "
                "with the fallback system that must be resolved immediately."
            )
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], output_file: Optional[Path] = None):
        """Save test results to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.project_root / 'reports' / f'fallback_tests_results_{timestamp}.json'
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📄 Test results saved to: {output_file}")
            
            # Also save a summary file
            summary_file = output_file.parent / f'fallback_tests_summary_{timestamp}.txt'
            with open(summary_file, 'w') as f:
                f.write("Multi-Level Fallback Test Results Summary\n")
                f.write("=" * 50 + "\n\n")
                
                summary = results['test_execution_summary']
                f.write(f"Overall Status: {summary['overall_status']}\n")
                f.write(f"Success Rate: {summary['success_rate']:.1%}\n")
                f.write(f"Total Time: {summary['total_execution_time_seconds']:.2f}s\n")
                f.write(f"Categories Passed: {len(summary['passed_categories'])}/{summary['total_categories']}\n\n")
                
                if summary['failed_categories']:
                    f.write("Failed Categories:\n")
                    for cat in summary['failed_categories']:
                        f.write(f"  - {cat}\n")
                    f.write("\n")
                
                f.write("Recommendations:\n")
                for rec in results['recommendations']:
                    f.write(f"  • {rec}\n")
            
            logger.info(f"📋 Test summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Multi-Level Fallback Test Runner for Clinical Metabolomics Oracle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_fallback_tests.py                           # Run core fallback tests
  python run_fallback_tests.py --category all           # Run all test categories  
  python run_fallback_tests.py --category performance   # Run only performance tests
  python run_fallback_tests.py --verbose --coverage     # Run with verbose output and coverage
  python run_fallback_tests.py --quick                  # Run only essential tests
        """
    )
    
    parser.add_argument(
        '--category',
        choices=['core_fallback', 'performance', 'integration', 'edge_cases', 'monitoring', 'all'],
        default='core_fallback',
        help='Test category to run (default: core_fallback)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Enable code coverage reporting'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run only essential tests (faster execution)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for test results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        help='Number of parallel test workers'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = FallbackTestRunner(PROJECT_ROOT)
    
    # Build additional pytest arguments
    additional_args = []
    
    if args.verbose:
        additional_args.extend(['--verbose', '-s'])
    
    if not args.coverage:
        # Disable coverage for faster execution
        additional_args.extend(['--no-cov'])
    
    if args.quick:
        additional_args.extend(['-k', 'not slow', '--maxfail=3'])
    
    if args.parallel:
        additional_args.extend(['-n', str(args.parallel)])
    
    # Determine categories to run
    if args.category == 'all':
        categories = ['core_fallback', 'performance', 'integration', 'edge_cases']
    else:
        categories = [args.category]
    
    # Run tests
    try:
        logger.info(f"Starting fallback test execution for categories: {categories}")
        results = runner.run_all_tests(categories, additional_args)
        
        # Save results
        runner.save_results(results, args.output)
        
        # Exit with appropriate code
        overall_status = results['test_execution_summary']['overall_status']
        if overall_status == 'PASSED':
            logger.info("🎉 Test execution completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Test execution completed with failures!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("⚠️  Test execution interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"💥 Test execution crashed: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()