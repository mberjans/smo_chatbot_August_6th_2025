#!/usr/bin/env python3
"""
Comprehensive Test Runner for Feature Flag System Tests.

This script provides a convenient interface for running the comprehensive
feature flag test suite with various options and configurations.

Usage:
    python run_feature_flag_tests.py [OPTIONS]

Options:
    --suite {all,unit,integration,performance,edge_cases} : Test suite to run
    --coverage : Run with coverage reporting
    --parallel : Run tests in parallel
    --verbose : Increase verbosity
    --fast : Skip slow tests
    --html-report : Generate HTML test report
    --performance-baseline : Run performance baseline tests
    --stress : Include stress tests
    --profile : Profile memory usage during tests

Author: Claude Code (Anthropic)
Created: 2025-08-08
"""

import os
import sys
import subprocess
import argparse
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional


class FeatureFlagTestRunner:
    """Comprehensive test runner for feature flag system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "lightrag_integration" / "tests"
        self.logs_dir = self.project_root / "logs"
        self.reports_dir = self.project_root / "test_reports"
        
        # Ensure directories exist
        self.logs_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run_tests(self, 
                  suite: str = "all",
                  coverage: bool = False,
                  parallel: bool = False,
                  verbose: bool = False,
                  fast: bool = False,
                  html_report: bool = False,
                  performance_baseline: bool = False,
                  stress: bool = False,
                  profile: bool = False) -> int:
        """Run test suite with specified options."""
        
        self.logger.info(f"Starting feature flag test suite: {suite}")
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Test selection based on suite
        test_files = self._get_test_files(suite)
        cmd.extend(test_files)
        
        # Add markers based on options
        markers = self._build_markers(suite, fast, performance_baseline, stress)
        if markers:
            cmd.extend(["-m", markers])
        
        # Verbosity options
        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")
        
        # Coverage options
        if coverage:
            coverage_args = [
                "--cov=lightrag_integration.feature_flag_manager",
                "--cov=lightrag_integration.integration_wrapper", 
                "--cov=lightrag_integration.config",
                "--cov-report=html:test_reports/htmlcov",
                "--cov-report=term-missing",
                "--cov-report=xml:test_reports/coverage.xml",
                "--cov-fail-under=85"
            ]
            cmd.extend(coverage_args)
        
        # Parallel execution
        if parallel:
            try:
                import pytest_xdist
                cmd.extend(["-n", "auto"])
            except ImportError:
                self.logger.warning("pytest-xdist not installed, running sequentially")
        
        # HTML reporting
        if html_report:
            try:
                import pytest_html
                cmd.extend([
                    "--html=test_reports/feature_flag_tests.html",
                    "--self-contained-html"
                ])
            except ImportError:
                self.logger.warning("pytest-html not installed, skipping HTML report")
        
        # Memory profiling
        if profile:
            try:
                import pytest_memray
                cmd.extend([
                    "--memray",
                    "--memray-bin-path=test_reports/memray"
                ])
            except ImportError:
                self.logger.warning("pytest-memray not installed, skipping memory profiling")
        
        # Additional pytest options
        cmd.extend([
            "--tb=short",
            "--durations=10",
            "--strict-markers",
            f"--junitxml=test_reports/junit_{suite}.xml"
        ])
        
        # Set environment variables for testing
        env = os.environ.copy()
        env.update(self._get_test_environment())
        
        # Run tests
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                timeout=1800  # 30 minute timeout
            )
            
            # Generate summary report
            self._generate_summary_report(suite, result.returncode)
            
            return result.returncode
            
        except subprocess.TimeoutExpired:
            self.logger.error("Test execution timed out after 30 minutes")
            return 2
        except KeyboardInterrupt:
            self.logger.info("Test execution interrupted by user")
            return 130
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return 1
    
    def _get_test_files(self, suite: str) -> List[str]:
        """Get test files based on selected suite."""
        if suite == "all":
            return [
                "lightrag_integration/tests/test_feature_flag_manager.py",
                "lightrag_integration/tests/test_integration_wrapper.py", 
                "lightrag_integration/tests/test_feature_flag_configuration.py",
                "lightrag_integration/tests/test_conditional_imports.py",
                "lightrag_integration/tests/test_feature_flag_integration.py",
                "lightrag_integration/tests/test_feature_flag_edge_cases.py",
                "lightrag_integration/tests/test_feature_flag_performance.py"
            ]
        elif suite == "unit":
            return [
                "lightrag_integration/tests/test_feature_flag_manager.py",
                "lightrag_integration/tests/test_integration_wrapper.py",
                "lightrag_integration/tests/test_feature_flag_configuration.py"
            ]
        elif suite == "integration":
            return [
                "lightrag_integration/tests/test_conditional_imports.py",
                "lightrag_integration/tests/test_feature_flag_integration.py"
            ]
        elif suite == "performance":
            return [
                "lightrag_integration/tests/test_feature_flag_performance.py"
            ]
        elif suite == "edge_cases":
            return [
                "lightrag_integration/tests/test_feature_flag_edge_cases.py"
            ]
        else:
            return [f"lightrag_integration/tests/test_{suite}.py"]
    
    def _build_markers(self, suite: str, fast: bool, performance_baseline: bool, stress: bool) -> Optional[str]:
        """Build pytest markers based on options."""
        markers = []
        
        # Suite-based markers
        if suite == "unit":
            markers.append("unit")
        elif suite == "integration":
            markers.append("integration")
        elif suite == "performance":
            markers.append("performance")
        elif suite == "edge_cases":
            markers.append("edge_cases")
        
        # Speed markers
        if fast:
            markers.append("not slow")
        
        # Performance markers
        if performance_baseline:
            markers.append("performance")
        
        # Stress markers
        if stress:
            markers.append("stress")
        
        return " and ".join(markers) if markers else None
    
    def _get_test_environment(self) -> Dict[str, str]:
        """Get environment variables for testing."""
        return {
            # Feature flag test settings
            "LIGHTRAG_INTEGRATION_ENABLED": "true",
            "LIGHTRAG_ENABLE_QUALITY_VALIDATION": "true",
            "LIGHTRAG_ENABLE_RELEVANCE_SCORING": "true",
            "LIGHTRAG_ENABLE_CIRCUIT_BREAKER": "true",
            "LIGHTRAG_ENABLE_AB_TESTING": "true",
            
            # Test-specific settings
            "LIGHTRAG_ROLLOUT_PERCENTAGE": "50.0",
            "LIGHTRAG_USER_HASH_SALT": "test_salt_2025",
            "LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "3",
            "LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "10.0",
            "LIGHTRAG_MIN_QUALITY_THRESHOLD": "0.7",
            
            # API keys for testing (mock values)
            "OPENAI_API_KEY": "test-openai-key-for-feature-flag-tests",
            "LIGHTRAG_MODEL": "gpt-4o-mini",
            "LIGHTRAG_EMBEDDING_MODEL": "text-embedding-3-small",
            
            # Logging configuration
            "LIGHTRAG_LOG_LEVEL": "WARNING",  # Reduce noise during tests
            "LIGHTRAG_ENABLE_FILE_LOGGING": "false",
            
            # Testing framework
            "PYTEST_CURRENT_TEST": "feature_flag_tests"
        }
    
    def _generate_summary_report(self, suite: str, return_code: int) -> None:
        """Generate summary report after test execution."""
        timestamp = Path(f"test_reports/summary_{suite}_{return_code}.txt")
        
        status = "PASSED" if return_code == 0 else "FAILED"
        
        summary = f"""
Feature Flag Test Suite Summary
{'=' * 50}

Suite: {suite}
Status: {status} (exit code: {return_code})
Timestamp: {timestamp}

Test Files Executed:
{chr(10).join('  - ' + f for f in self._get_test_files(suite))}

Reports Generated:
  - JUnit XML: test_reports/junit_{suite}.xml
  - HTML Report: test_reports/feature_flag_tests.html
  - Coverage Report: test_reports/htmlcov/index.html
  - Memory Profile: test_reports/memray/

Environment Configuration:
{chr(10).join(f'  {k}={v}' for k, v in self._get_test_environment().items())}

Next Steps:
"""
        
        if return_code == 0:
            summary += """
  ✅ All tests passed successfully!
  - Review coverage report for any gaps
  - Check performance metrics in detailed reports
  - Consider running stress tests if not already included
"""
        else:
            summary += """
  ❌ Some tests failed. Please review:
  - Check detailed test output for failure reasons
  - Review JUnit XML report for structured failure data
  - Check HTML report for visual failure analysis
  - Ensure all dependencies are installed and configured
"""
        
        summary_path = self.reports_dir / f"summary_{suite}_{timestamp.stem}.txt"
        summary_path.write_text(summary)
        
        self.logger.info(f"Summary report written to: {summary_path}")
    
    def run_quick_health_check(self) -> int:
        """Run quick health check to verify test setup."""
        self.logger.info("Running quick health check...")
        
        cmd = [
            "python", "-m", "pytest", 
            "lightrag_integration/tests/test_conditional_imports.py::TestFeatureFlagLoading::test_feature_flags_loaded_on_import",
            "-v"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, timeout=30)
            
            if result.returncode == 0:
                self.logger.info("✅ Health check passed - test environment is ready")
            else:
                self.logger.error("❌ Health check failed - please check test environment")
            
            return result.returncode
            
        except subprocess.TimeoutExpired:
            self.logger.error("Health check timed out")
            return 2
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return 1
    
    def list_available_tests(self) -> None:
        """List all available tests."""
        print("\nAvailable Feature Flag Tests:")
        print("=" * 50)
        
        test_files = {
            "test_feature_flag_manager.py": "FeatureFlagManager core functionality",
            "test_integration_wrapper.py": "IntegratedQueryService and service routing",
            "test_feature_flag_configuration.py": "Configuration parsing and validation",
            "test_conditional_imports.py": "Conditional import system and graceful degradation",
            "test_feature_flag_integration.py": "End-to-end integration workflows", 
            "test_feature_flag_edge_cases.py": "Edge cases and error conditions",
            "test_feature_flag_performance.py": "Performance and stress testing"
        }
        
        for file, description in test_files.items():
            print(f"  📝 {file}")
            print(f"     {description}")
            print()
        
        print("Test Suites:")
        print("  🎯 all - Run all tests")
        print("  🔧 unit - Unit tests only")
        print("  🔗 integration - Integration tests only")
        print("  ⚡ performance - Performance tests only")
        print("  🚨 edge_cases - Edge case tests only")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Feature Flag Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_feature_flag_tests.py --suite all --coverage
  python run_feature_flag_tests.py --suite unit --fast --parallel
  python run_feature_flag_tests.py --suite performance --performance-baseline
  python run_feature_flag_tests.py --health-check
  python run_feature_flag_tests.py --list-tests
        """
    )
    
    parser.add_argument(
        "--suite",
        choices=["all", "unit", "integration", "performance", "edge_cases"],
        default="all",
        help="Test suite to run (default: all)"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase test output verbosity"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow-running tests"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report (requires pytest-html)"
    )
    
    parser.add_argument(
        "--performance-baseline",
        action="store_true",
        help="Run performance baseline tests"
    )
    
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Include stress tests"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile memory usage during tests (requires pytest-memray)"
    )
    
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run quick health check only"
    )
    
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests and exit"
    )
    
    args = parser.parse_args()
    
    runner = FeatureFlagTestRunner()
    
    if args.list_tests:
        runner.list_available_tests()
        return 0
    
    if args.health_check:
        return runner.run_quick_health_check()
    
    return runner.run_tests(
        suite=args.suite,
        coverage=args.coverage,
        parallel=args.parallel,
        verbose=args.verbose,
        fast=args.fast,
        html_report=args.html_report,
        performance_baseline=args.performance_baseline,
        stress=args.stress,
        profile=args.profile
    )


if __name__ == "__main__":
    sys.exit(main())