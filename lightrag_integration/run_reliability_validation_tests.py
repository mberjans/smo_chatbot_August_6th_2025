#!/usr/bin/env python3
"""
Comprehensive Reliability Validation Test Runner
===============================================

Main execution runner for CMO-LIGHTRAG-014-T08 reliability validation tests.
This script orchestrates the execution of all reliability test scenarios and
generates comprehensive reports.

Test Categories Covered:
1. Stress Testing & Load Limits (ST-001 to ST-004)
2. Network Reliability (NR-001 to NR-004)  
3. Data Integrity & Consistency (DI-001 to DI-003)
4. Production Scenario Testing (PS-001 to PS-003)
5. Integration Reliability (IR-001 to IR-003)

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import argparse
import logging
import json
import time
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

# Import test modules
from tests.reliability_test_framework import (
    ReliabilityValidationFramework,
    ReliabilityTestConfig,
    ReliabilityMetrics,
    TestResult
)

from tests.test_stress_testing_scenarios import run_all_stress_tests
from tests.test_network_reliability_scenarios import run_all_network_reliability_tests
from tests.test_data_integrity_scenarios import run_all_data_integrity_tests
from tests.test_production_scenarios import run_all_production_scenario_tests
from tests.test_integration_reliability_scenarios import run_all_integration_reliability_tests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reliability_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# RELIABILITY TEST SUITE ORCHESTRATOR
# ============================================================================

class ReliabilityTestSuiteOrchestrator:
    """Main orchestrator for reliability validation test suite."""
    
    def __init__(self, config: Optional[ReliabilityTestConfig] = None):
        self.config = config or ReliabilityTestConfig()
        self.framework = ReliabilityValidationFramework(self.config)
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    async def run_complete_test_suite(
        self, 
        categories: Optional[List[str]] = None,
        include_long_running: bool = True,
        parallel_execution: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete reliability validation test suite.
        
        Args:
            categories: List of test categories to run (None = all)
            include_long_running: Whether to include long-running tests
            parallel_execution: Whether to run test categories in parallel
            
        Returns:
            Dict containing comprehensive test results
        """
        logger.info("🚀 Starting Comprehensive Reliability Validation Test Suite")
        logger.info("=" * 80)
        
        self.start_time = datetime.now()
        
        # Define available test categories
        available_categories = {
            'stress_testing': {
                'name': 'Stress Testing & Load Limits',
                'runner': self._run_stress_testing_category,
                'estimated_duration': 45,  # minutes
                'long_running': True
            },
            'network_reliability': {
                'name': 'Network Reliability',
                'runner': self._run_network_reliability_category,
                'estimated_duration': 35,  # minutes
                'long_running': True
            },
            'data_integrity': {
                'name': 'Data Integrity & Consistency',
                'runner': self._run_data_integrity_category,
                'estimated_duration': 25,  # minutes
                'long_running': False
            },
            'production_scenarios': {
                'name': 'Production Scenario Testing',
                'runner': self._run_production_scenarios_category,
                'estimated_duration': 40,  # minutes
                'long_running': True
            },
            'integration_reliability': {
                'name': 'Integration Reliability',
                'runner': self._run_integration_reliability_category,
                'estimated_duration': 30,  # minutes
                'long_running': False
            }
        }
        
        # Filter categories based on parameters
        if categories:
            test_categories = {k: v for k, v in available_categories.items() if k in categories}
        else:
            test_categories = available_categories
            
        if not include_long_running:
            test_categories = {k: v for k, v in test_categories.items() if not v['long_running']}
        
        # Calculate total estimated duration
        total_estimated_duration = sum(cat['estimated_duration'] for cat in test_categories.values())
        logger.info(f"📊 Running {len(test_categories)} test categories")
        logger.info(f"⏱️  Estimated total duration: {total_estimated_duration} minutes")
        logger.info("")
        
        try:
            # Setup test environment
            await self.framework.setup_test_environment()
            
            # Execute test categories
            if parallel_execution and len(test_categories) > 1:
                await self._run_categories_parallel(test_categories)
            else:
                await self._run_categories_sequential(test_categories)
            
        finally:
            # Cleanup test environment
            await self.framework.cleanup_test_environment()
            
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        final_report = await self._generate_final_report()
        
        logger.info("🏁 Reliability Validation Test Suite Completed")
        logger.info("=" * 80)
        
        return final_report
    
    async def _run_categories_sequential(self, test_categories: Dict[str, Any]):
        """Run test categories sequentially."""
        for category_id, category_info in test_categories.items():
            logger.info(f"📋 Starting {category_info['name']} Tests")
            logger.info("-" * 60)
            
            category_start_time = time.time()
            
            try:
                category_results = await category_info['runner']()
                category_duration = time.time() - category_start_time
                
                self.test_results[category_id] = {
                    'category_name': category_info['name'],
                    'status': 'completed',
                    'duration': category_duration,
                    'results': category_results
                }
                
                logger.info(f"✅ {category_info['name']} completed in {category_duration/60:.1f} minutes")
                
            except Exception as e:
                category_duration = time.time() - category_start_time
                
                self.test_results[category_id] = {
                    'category_name': category_info['name'],
                    'status': 'failed',
                    'duration': category_duration,
                    'error': str(e)
                }
                
                logger.error(f"❌ {category_info['name']} failed: {str(e)}")
            
            # Brief recovery between categories
            await asyncio.sleep(10)
    
    async def _run_categories_parallel(self, test_categories: Dict[str, Any]):
        """Run test categories in parallel (experimental)."""
        logger.info("🔄 Running test categories in parallel")
        
        # Create tasks for each category
        category_tasks = {}
        for category_id, category_info in test_categories.items():
            task = asyncio.create_task(
                self._run_category_with_error_handling(category_id, category_info),
                name=f"category_{category_id}"
            )
            category_tasks[category_id] = task
        
        # Wait for all categories to complete
        await asyncio.gather(*category_tasks.values(), return_exceptions=True)
        
        # Collect results from tasks
        for category_id, task in category_tasks.items():
            if task.done() and not task.exception():
                self.test_results[category_id] = await task
            else:
                self.test_results[category_id] = {
                    'category_name': test_categories[category_id]['name'],
                    'status': 'failed',
                    'error': str(task.exception()) if task.exception() else 'Unknown error'
                }
    
    async def _run_category_with_error_handling(
        self, 
        category_id: str, 
        category_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single category with error handling."""
        category_start_time = time.time()
        
        try:
            category_results = await category_info['runner']()
            category_duration = time.time() - category_start_time
            
            return {
                'category_name': category_info['name'],
                'status': 'completed',
                'duration': category_duration,
                'results': category_results
            }
            
        except Exception as e:
            category_duration = time.time() - category_start_time
            
            return {
                'category_name': category_info['name'],
                'status': 'failed',
                'duration': category_duration,
                'error': str(e)
            }
    
    async def _run_stress_testing_category(self) -> Dict[str, Any]:
        """Run stress testing scenarios."""
        logger.info("Running Stress Testing scenarios (ST-001 to ST-004)")
        return await run_all_stress_tests()
    
    async def _run_network_reliability_category(self) -> Dict[str, Any]:
        """Run network reliability scenarios."""
        logger.info("Running Network Reliability scenarios (NR-001 to NR-004)")  
        return await run_all_network_reliability_tests()
    
    async def _run_data_integrity_category(self) -> Dict[str, Any]:
        """Run data integrity scenarios."""
        logger.info("Running Data Integrity scenarios (DI-001 to DI-003)")
        return await run_all_data_integrity_tests()
    
    async def _run_production_scenarios_category(self) -> Dict[str, Any]:
        """Run production scenario tests."""
        logger.info("Running Production Scenario tests (PS-001 to PS-003)")
        return await run_all_production_scenario_tests()
    
    async def _run_integration_reliability_category(self) -> Dict[str, Any]:
        """Run integration reliability tests."""
        logger.info("Running Integration Reliability tests (IR-001 to IR-003)")
        return await run_all_integration_reliability_tests()
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        total_duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        # Calculate overall statistics
        total_categories = len(self.test_results)
        completed_categories = len([r for r in self.test_results.values() if r['status'] == 'completed'])
        failed_categories = total_categories - completed_categories
        
        # Calculate test-level statistics
        all_test_results = []
        for category_results in self.test_results.values():
            if 'results' in category_results:
                all_test_results.extend(category_results['results'].values())
        
        passed_tests = len([t for t in all_test_results if t.get('status') == 'passed'])
        total_tests = len(all_test_results)
        
        # Calculate reliability score
        reliability_score = self._calculate_overall_reliability_score()
        
        final_report = {
            'execution_summary': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'total_duration_seconds': total_duration,
                'total_duration_minutes': total_duration / 60,
                'categories_executed': total_categories,
                'categories_completed': completed_categories,
                'categories_failed': failed_categories,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'overall_success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'overall_reliability_score': reliability_score
            },
            'category_results': self.test_results,
            'recommendations': self._generate_recommendations(),
            'risk_assessment': self._assess_reliability_risks(),
            'compliance_status': self._check_reliability_compliance()
        }
        
        # Save report to file
        report_filename = f"reliability_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"📄 Comprehensive report saved to: {report_filename}")
        
        # Print executive summary
        self._print_executive_summary(final_report)
        
        return final_report
    
    def _calculate_overall_reliability_score(self) -> float:
        """Calculate overall reliability score (0-100)."""
        category_weights = {
            'stress_testing': 0.25,
            'network_reliability': 0.25,
            'data_integrity': 0.20,
            'production_scenarios': 0.20,
            'integration_reliability': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for category_id, results in self.test_results.items():
            weight = category_weights.get(category_id, 0.1)
            
            if results['status'] == 'completed' and 'results' in results:
                category_tests = results['results'].values()
                category_success_rate = len([t for t in category_tests if t.get('status') == 'passed']) / len(category_tests) if category_tests else 0
                weighted_score += category_success_rate * weight
            else:
                # Failed categories contribute 0
                pass
            
            total_weight += weight
        
        return (weighted_score / total_weight * 100) if total_weight > 0 else 0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate reliability improvement recommendations."""
        recommendations = []
        
        # Analyze failed categories
        failed_categories = [cat_id for cat_id, results in self.test_results.items() if results['status'] == 'failed']
        
        if 'stress_testing' in failed_categories:
            recommendations.append("Review and optimize stress testing resilience - system may not handle peak loads effectively")
        
        if 'network_reliability' in failed_categories:
            recommendations.append("Strengthen network reliability mechanisms - external service dependencies need better handling")
        
        if 'data_integrity' in failed_categories:
            recommendations.append("Improve data consistency and integrity checks - data quality issues detected")
        
        if not recommendations:
            recommendations.append("All test categories passed successfully - system demonstrates high reliability")
        
        # General recommendations
        reliability_score = self._calculate_overall_reliability_score()
        if reliability_score < 85:
            recommendations.append(f"Overall reliability score ({reliability_score:.1f}%) below production threshold - review all failed tests")
        elif reliability_score < 95:
            recommendations.append(f"Good reliability score ({reliability_score:.1f}%) with room for improvement - focus on edge cases")
        else:
            recommendations.append(f"Excellent reliability score ({reliability_score:.1f}%) - system is production-ready")
        
        return recommendations
    
    def _assess_reliability_risks(self) -> Dict[str, str]:
        """Assess reliability risks based on test results."""
        risks = {}
        
        reliability_score = self._calculate_overall_reliability_score()
        
        if reliability_score < 75:
            risks['overall'] = 'HIGH - System may not be suitable for production use'
        elif reliability_score < 85:
            risks['overall'] = 'MEDIUM - System requires improvements before production'
        elif reliability_score < 95:
            risks['overall'] = 'LOW - System is suitable for production with monitoring'
        else:
            risks['overall'] = 'MINIMAL - System demonstrates excellent reliability'
        
        # Category-specific risks
        failed_categories = [cat_id for cat_id, results in self.test_results.items() if results['status'] == 'failed']
        
        for category in failed_categories:
            category_name = self.test_results[category].get('category_name', category)
            risks[category] = f'HIGH - {category_name} failures indicate serious reliability concerns'
        
        return risks
    
    def _check_reliability_compliance(self) -> Dict[str, bool]:
        """Check compliance with reliability standards."""
        reliability_score = self._calculate_overall_reliability_score()
        
        compliance = {
            'minimum_reliability_threshold': reliability_score >= 85,  # 85% minimum
            'production_readiness': reliability_score >= 90,  # 90% for production
            'high_availability_standard': reliability_score >= 95,  # 95% for HA
            'all_critical_tests_passed': all(
                results['status'] == 'completed' 
                for cat_id, results in self.test_results.items() 
                if cat_id in ['stress_testing', 'network_reliability']
            ),
            'no_critical_failures': len([
                cat_id for cat_id, results in self.test_results.items() 
                if results['status'] == 'failed'
            ]) == 0
        }
        
        return compliance
    
    def _print_executive_summary(self, report: Dict[str, Any]):
        """Print executive summary to console."""
        summary = report['execution_summary']
        recommendations = report['recommendations']
        risks = report['risk_assessment']
        compliance = report['compliance_status']
        
        print("\n" + "=" * 80)
        print("📊 RELIABILITY VALIDATION EXECUTIVE SUMMARY")
        print("=" * 80)
        print(f"Execution Duration: {summary['total_duration_minutes']:.1f} minutes")
        print(f"Categories Tested: {summary['categories_executed']}")
        print(f"Tests Executed: {summary['total_tests']}")
        print(f"Tests Passed: {summary['passed_tests']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"Reliability Score: {summary['overall_reliability_score']:.1f}%")
        
        print(f"\n🎯 COMPLIANCE STATUS:")
        for standard, compliant in compliance.items():
            status = "✅ PASS" if compliant else "❌ FAIL"
            print(f"  {standard.replace('_', ' ').title()}: {status}")
        
        print(f"\n⚠️  RISK ASSESSMENT:")
        for category, risk_level in risks.items():
            print(f"  {category.replace('_', ' ').title()}: {risk_level}")
        
        print(f"\n🔧 KEY RECOMMENDATIONS:")
        for i, recommendation in enumerate(recommendations, 1):
            print(f"  {i}. {recommendation}")
        
        print("=" * 80)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Reliability Validation Test Runner for CMO-LIGHTRAG-014-T08",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all test categories
  python run_reliability_validation_tests.py

  # Run only stress and network tests
  python run_reliability_validation_tests.py --categories stress_testing network_reliability

  # Run quick tests only (exclude long-running tests)
  python run_reliability_validation_tests.py --quick

  # Run with custom configuration
  python run_reliability_validation_tests.py --config custom_config.json

  # Run with detailed logging
  python run_reliability_validation_tests.py --verbose
        """
    )
    
    parser.add_argument(
        '--categories', 
        nargs='*',
        choices=['stress_testing', 'network_reliability', 'data_integrity', 'production_scenarios', 'integration_reliability'],
        help='Test categories to run (default: all categories)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (exclude long-running tests)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run test categories in parallel (experimental)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom test configuration JSON file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save test reports (default: current directory)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without actually executing tests'
    )
    
    return parser


def load_config_from_file(config_path: str) -> ReliabilityTestConfig:
    """Load test configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create config with loaded parameters
        config = ReliabilityTestConfig()
        
        # Update config attributes from JSON
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return ReliabilityTestConfig()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Change to output directory
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
        logger.info(f"Changed to output directory: {args.output_dir}")
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = ReliabilityTestConfig()
    
    # Create orchestrator
    orchestrator = ReliabilityTestSuiteOrchestrator(config)
    
    # Handle dry run
    if args.dry_run:
        print("🧪 DRY RUN MODE - Showing what would be executed:")
        print(f"Categories: {args.categories or 'all'}")
        print(f"Quick mode: {args.quick}")
        print(f"Parallel execution: {args.parallel}")
        print(f"Output directory: {args.output_dir}")
        return
    
    try:
        # Execute test suite
        final_report = await orchestrator.run_complete_test_suite(
            categories=args.categories,
            include_long_running=not args.quick,
            parallel_execution=args.parallel
        )
        
        # Exit with appropriate code
        reliability_score = final_report['execution_summary']['overall_reliability_score']
        
        if reliability_score >= 90:
            logger.info("🎉 Reliability validation completed successfully - System is production-ready")
            sys.exit(0)
        elif reliability_score >= 75:
            logger.warning("⚠️  Reliability validation completed with concerns - Review recommendations")
            sys.exit(1)
        else:
            logger.error("❌ Reliability validation failed - System requires significant improvements")
            sys.exit(2)
            
    except KeyboardInterrupt:
        logger.info("🛑 Test execution interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"💥 Test execution failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())