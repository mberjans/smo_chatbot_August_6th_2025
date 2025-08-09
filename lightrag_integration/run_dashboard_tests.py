#!/usr/bin/env python3
"""
Test Runner for Unified System Health Dashboard
==============================================

This script provides convenient ways to run the comprehensive test suite
for the unified system health monitoring dashboard with various configurations.

Usage:
    python run_dashboard_tests.py [OPTIONS]

Examples:
    # Run all tests
    python run_dashboard_tests.py

    # Run only unit tests
    python run_dashboard_tests.py --unit

    # Run tests with coverage report
    python run_dashboard_tests.py --coverage

    # Run performance tests
    python run_dashboard_tests.py --performance

    # Run tests in parallel
    python run_dashboard_tests.py --parallel

    # Generate detailed HTML report
    python run_dashboard_tests.py --html-report

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
Production Ready: Yes
Task: CMO-LIGHTRAG-014-T07 - Dashboard Test Runner
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


class DashboardTestRunner:
    """Test runner for the unified dashboard test suite."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_file = self.project_root / "test_unified_system_health_dashboard_comprehensive.py"
        self.pytest_config = self.project_root / "pytest.ini"
        
    def run_command(self, command: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        print(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root
            )
            return result
        except FileNotFoundError as e:
            print(f"Error: Command not found: {e}")
            print("Please ensure pytest is installed: pip install -r test_requirements.txt")
            sys.exit(1)
    
    def install_test_dependencies(self) -> bool:
        """Install test dependencies if needed."""
        requirements_file = self.project_root / "test_requirements.txt"
        
        if not requirements_file.exists():
            print("Warning: test_requirements.txt not found, skipping dependency installation")
            return True
        
        print("Installing test dependencies...")
        result = self.run_command([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        
        if result.returncode != 0:
            print("Failed to install test dependencies")
            return False
        
        print("Test dependencies installed successfully")
        return True
    
    def run_unit_tests(self, extra_args: List[str] = None) -> int:
        """Run unit tests only."""
        print("Running unit tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "-m", "unit or not (integration or api or websocket or performance or security)",
            "-v"
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        return result.returncode
    
    def run_integration_tests(self, extra_args: List[str] = None) -> int:
        """Run integration tests only."""
        print("Running integration tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "-m", "integration",
            "-v"
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        return result.returncode
    
    def run_api_tests(self, extra_args: List[str] = None) -> int:
        """Run API tests only."""
        print("Running API tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "-m", "api",
            "-v"
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        return result.returncode
    
    def run_websocket_tests(self, extra_args: List[str] = None) -> int:
        """Run WebSocket tests only."""
        print("Running WebSocket tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "-m", "websocket",
            "-v"
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        return result.returncode
    
    def run_performance_tests(self, extra_args: List[str] = None) -> int:
        """Run performance tests only."""
        print("Running performance tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "-m", "performance",
            "-v",
            "-s"  # Show output for performance metrics
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        return result.returncode
    
    def run_security_tests(self, extra_args: List[str] = None) -> int:
        """Run security tests only."""
        print("Running security tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "-m", "security",
            "-v"
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        return result.returncode
    
    def run_all_tests(self, extra_args: List[str] = None) -> int:
        """Run all tests."""
        print("Running all tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "-v"
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        return result.returncode
    
    def run_with_coverage(self, extra_args: List[str] = None) -> int:
        """Run tests with coverage reporting."""
        print("Running tests with coverage...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "--cov=unified_system_health_dashboard",
            "--cov=dashboard_integration_helper",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "-v"
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("\nCoverage reports generated:")
            print("  - Terminal: displayed above")
            print("  - HTML: htmlcov/index.html")
            print("  - XML: coverage.xml")
        
        return result.returncode
    
    def run_parallel_tests(self, num_workers: int = None, extra_args: List[str] = None) -> int:
        """Run tests in parallel using pytest-xdist."""
        if num_workers is None:
            import multiprocessing
            num_workers = multiprocessing.cpu_count()
        
        print(f"Running tests in parallel with {num_workers} workers...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            f"-n{num_workers}",
            "-v"
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        return result.returncode
    
    def generate_html_report(self, extra_args: List[str] = None) -> int:
        """Generate detailed HTML test report."""
        print("Generating HTML test report...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "--html=dashboard_test_report.html",
            "--self-contained-html",
            "-v"
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("\nHTML test report generated: dashboard_test_report.html")
        
        return result.returncode
    
    def run_quick_tests(self, extra_args: List[str] = None) -> int:
        """Run a quick subset of tests for development."""
        print("Running quick test subset...")
        
        command = [
            sys.executable, "-m", "pytest",
            str(self.test_file),
            "-m", "not (slow or performance)",
            "--tb=line",
            "-x"  # Stop on first failure
        ]
        
        if extra_args:
            command.extend(extra_args)
        
        result = self.run_command(command)
        return result.returncode
    
    def run_continuous_testing(self, watch_files: List[str] = None) -> int:
        """Run tests continuously when files change."""
        print("Starting continuous testing mode...")
        print("Tests will re-run when files change. Press Ctrl+C to stop.")
        
        if watch_files is None:
            watch_files = [
                "unified_system_health_dashboard.py",
                "dashboard_integration_helper.py",
                "test_unified_system_health_dashboard_comprehensive.py"
            ]
        
        try:
            # Simple file watching implementation
            last_modified = {}
            
            for file_path in watch_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    last_modified[full_path] = full_path.stat().st_mtime
            
            print(f"Watching files: {', '.join(watch_files)}")
            
            # Run tests initially
            self.run_quick_tests()
            
            while True:
                time.sleep(2)  # Check every 2 seconds
                
                file_changed = False
                for file_path, last_time in last_modified.items():
                    if file_path.exists():
                        current_time = file_path.stat().st_mtime
                        if current_time > last_time:
                            last_modified[file_path] = current_time
                            file_changed = True
                            print(f"\nFile changed: {file_path.name}")
                            break
                
                if file_changed:
                    print("Re-running tests...")
                    self.run_quick_tests()
                    print("\nWatching for changes...")
        
        except KeyboardInterrupt:
            print("\nContinuous testing stopped.")
            return 0
    
    def validate_test_environment(self) -> bool:
        """Validate that the test environment is properly set up."""
        print("Validating test environment...")
        
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append(f"Python 3.8+ required, found {sys.version_info}")
        
        # Check test file exists
        if not self.test_file.exists():
            issues.append(f"Test file not found: {self.test_file}")
        
        # Check pytest is available
        try:
            result = self.run_command([sys.executable, "-m", "pytest", "--version"], capture_output=True)
            if result.returncode != 0:
                issues.append("pytest not available or not working")
        except:
            issues.append("pytest not installed")
        
        # Check essential modules can be imported
        essential_modules = [
            "fastapi", "uvicorn", "websockets", "sqlite3", "asyncio", "json"
        ]
        
        for module in essential_modules:
            try:
                __import__(module)
            except ImportError:
                issues.append(f"Required module not available: {module}")
        
        if issues:
            print("❌ Environment validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Test environment validation passed")
            return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for Unified System Health Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Test type selection (mutually exclusive)
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--unit", action="store_true",
        help="Run unit tests only"
    )
    test_group.add_argument(
        "--integration", action="store_true",
        help="Run integration tests only"
    )
    test_group.add_argument(
        "--api", action="store_true",
        help="Run API tests only"
    )
    test_group.add_argument(
        "--websocket", action="store_true",
        help="Run WebSocket tests only"
    )
    test_group.add_argument(
        "--performance", action="store_true",
        help="Run performance tests only"
    )
    test_group.add_argument(
        "--security", action="store_true",
        help="Run security tests only"
    )
    test_group.add_argument(
        "--quick", action="store_true",
        help="Run quick test subset for development"
    )
    
    # Test execution options
    parser.add_argument(
        "--coverage", action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--html-report", action="store_true",
        help="Generate HTML test report"
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run tests continuously when files change"
    )
    
    # Configuration options
    parser.add_argument(
        "--install-deps", action="store_true",
        help="Install test dependencies before running"
    )
    parser.add_argument(
        "--validate-env", action="store_true",
        help="Validate test environment setup"
    )
    parser.add_argument(
        "--workers", type=int,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Test timeout in seconds (default: 300)"
    )
    
    # Additional pytest arguments
    parser.add_argument(
        "pytest_args", nargs="*",
        help="Additional arguments to pass to pytest"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = DashboardTestRunner()
    
    # Validate environment if requested
    if args.validate_env:
        if not runner.validate_test_environment():
            sys.exit(1)
        return
    
    # Install dependencies if requested
    if args.install_deps:
        if not runner.install_test_dependencies():
            sys.exit(1)
    
    # Prepare extra arguments
    extra_args = []
    if args.timeout:
        extra_args.extend(["--timeout", str(args.timeout)])
    if args.pytest_args:
        extra_args.extend(args.pytest_args)
    
    # Run appropriate test suite
    start_time = time.time()
    
    try:
        if args.continuous:
            return_code = runner.run_continuous_testing()
        elif args.unit:
            return_code = runner.run_unit_tests(extra_args)
        elif args.integration:
            return_code = runner.run_integration_tests(extra_args)
        elif args.api:
            return_code = runner.run_api_tests(extra_args)
        elif args.websocket:
            return_code = runner.run_websocket_tests(extra_args)
        elif args.performance:
            return_code = runner.run_performance_tests(extra_args)
        elif args.security:
            return_code = runner.run_security_tests(extra_args)
        elif args.quick:
            return_code = runner.run_quick_tests(extra_args)
        elif args.coverage:
            return_code = runner.run_with_coverage(extra_args)
        elif args.parallel:
            return_code = runner.run_parallel_tests(args.workers, extra_args)
        elif args.html_report:
            return_code = runner.generate_html_report(extra_args)
        else:
            # Default: run all tests
            return_code = runner.run_all_tests(extra_args)
        
        # Print summary
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nTest execution completed in {duration:.2f} seconds")
        
        if return_code == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        
        return return_code
    
    except KeyboardInterrupt:
        print("\n⏹ Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())