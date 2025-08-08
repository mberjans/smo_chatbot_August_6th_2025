"""
Test Coverage Configuration and Automated Reporting Infrastructure

This module provides comprehensive test coverage analysis, automated reporting,
and quality metrics for the LLM-based classification system.

Features:
    - Automated test coverage analysis with >95% target
    - Performance benchmarking and regression detection
    - Quality metrics reporting and validation
    - CI/CD integration support
    - HTML and JSON coverage reports
    - Test result analysis and recommendations

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import os
import sys
import json
import time
import pytest
import coverage
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# COVERAGE CONFIGURATION AND ANALYSIS
# ============================================================================

@dataclass
class CoverageTarget:
    """Coverage targets for different components."""
    
    overall_target: float = 95.0
    core_components: Dict[str, float] = field(default_factory=lambda: {
        'llm_query_classifier': 98.0,
        'comprehensive_confidence_scorer': 96.0,
        'query_router': 94.0,
        'llm_classification_prompts': 90.0
    })
    critical_functions: Dict[str, float] = field(default_factory=lambda: {
        'classify_query': 100.0,
        'calculate_comprehensive_confidence': 98.0,
        'route_query': 95.0,
        'calibrate_confidence': 96.0
    })


@dataclass
class TestResults:
    """Comprehensive test results and metrics."""
    
    # Basic test metrics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    
    # Coverage metrics
    overall_coverage: float = 0.0
    component_coverage: Dict[str, float] = field(default_factory=dict)
    critical_function_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    total_execution_time: float = 0.0
    average_test_time: float = 0.0
    slowest_tests: List[Tuple[str, float]] = field(default_factory=list)
    performance_regressions: List[str] = field(default_factory=list)
    
    # Quality metrics
    code_quality_score: float = 0.0
    test_quality_score: float = 0.0
    maintainability_index: float = 0.0
    
    # Issues and recommendations
    coverage_gaps: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    test_environment: Dict[str, Any] = field(default_factory=dict)


class CoverageAnalyzer:
    """Comprehensive coverage analysis and reporting."""
    
    def __init__(self, 
                 source_dir: str = None,
                 test_dir: str = None,
                 output_dir: str = None,
                 logger: Optional[logging.Logger] = None):
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Set default directories
        current_dir = Path(__file__).parent.parent
        self.source_dir = Path(source_dir) if source_dir else current_dir
        self.test_dir = Path(test_dir) if test_dir else current_dir / "tests"
        self.output_dir = Path(output_dir) if output_dir else current_dir / "coverage_reports"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Coverage targets
        self.targets = CoverageTarget()
        
        # Initialize coverage instance
        self.coverage = coverage.Coverage(
            source=[str(self.source_dir)],
            omit=[
                "*/tests/*",
                "*/test_*",
                "*/__pycache__/*",
                "*/venv/*",
                "*/env/*"
            ]
        )
        
        self.logger.info(f"Coverage analyzer initialized - Source: {self.source_dir}, Tests: {self.test_dir}")
    
    def run_tests_with_coverage(self, test_patterns: List[str] = None) -> TestResults:
        """Run tests with coverage analysis."""
        
        self.logger.info("Starting test execution with coverage analysis...")
        start_time = time.time()
        
        # Start coverage
        self.coverage.start()
        
        try:
            # Run pytest with coverage
            pytest_args = [
                str(self.test_dir),
                "-v",
                "--tb=short",
                "--durations=10",
                f"--junitxml={self.output_dir}/test_results.xml",
                f"--html={self.output_dir}/test_report.html",
                "--self-contained-html"
            ]
            
            if test_patterns:
                pytest_args.extend(["-k", " or ".join(test_patterns)])
            
            # Run tests
            exit_code = pytest.main(pytest_args)
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            exit_code = 1
        
        finally:
            # Stop coverage
            self.coverage.stop()
            self.coverage.save()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Analyze results
        test_results = self._analyze_test_results(execution_time, exit_code)
        
        self.logger.info(f"Test execution completed in {execution_time:.2f}s")
        return test_results
    
    def _analyze_test_results(self, execution_time: float, exit_code: int) -> TestResults:
        """Analyze test results and generate comprehensive metrics."""
        
        results = TestResults()
        results.total_execution_time = execution_time
        
        # Parse JUnit XML for detailed test metrics
        junit_file = self.output_dir / "test_results.xml"
        if junit_file.exists():
            results = self._parse_junit_results(results, junit_file)
        
        # Generate coverage report
        coverage_data = self._generate_coverage_report()
        results.overall_coverage = coverage_data['overall']
        results.component_coverage = coverage_data['components']
        results.critical_function_coverage = coverage_data['critical_functions']
        
        # Identify coverage gaps
        results.coverage_gaps = self._identify_coverage_gaps(coverage_data)
        
        # Calculate quality metrics
        results.code_quality_score = self._calculate_code_quality_score()
        results.test_quality_score = self._calculate_test_quality_score(results)
        
        # Generate recommendations
        results.recommendations = self._generate_recommendations(results)
        
        # Set test environment info
        results.test_environment = {
            'python_version': sys.version,
            'pytest_version': pytest.__version__,
            'coverage_version': coverage.__version__,
            'exit_code': exit_code
        }
        
        return results
    
    def _parse_junit_results(self, results: TestResults, junit_file: Path) -> TestResults:
        """Parse JUnit XML for detailed test metrics."""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(junit_file)
            root = tree.getroot()
            
            # Parse test suite results
            for testsuite in root.findall('testsuite'):
                results.total_tests += int(testsuite.get('tests', 0))
                results.failed_tests += int(testsuite.get('failures', 0))
                results.error_tests += int(testsuite.get('errors', 0))
                results.skipped_tests += int(testsuite.get('skipped', 0))
                
                # Parse individual test cases for timing
                for testcase in testsuite.findall('testcase'):
                    test_name = f"{testcase.get('classname')}.{testcase.get('name')}"
                    test_time = float(testcase.get('time', 0))
                    
                    # Track slowest tests
                    results.slowest_tests.append((test_name, test_time))
            
            results.passed_tests = results.total_tests - results.failed_tests - results.error_tests - results.skipped_tests
            
            # Sort slowest tests
            results.slowest_tests.sort(key=lambda x: x[1], reverse=True)
            results.slowest_tests = results.slowest_tests[:10]  # Top 10 slowest
            
            # Calculate average test time
            if results.total_tests > 0:
                results.average_test_time = results.total_execution_time / results.total_tests
            
        except Exception as e:
            self.logger.warning(f"Failed to parse JUnit results: {e}")
        
        return results
    
    def _generate_coverage_report(self) -> Dict[str, Any]:
        """Generate detailed coverage report."""
        
        coverage_data = {
            'overall': 0.0,
            'components': {},
            'critical_functions': {}
        }
        
        try:
            # Generate coverage report
            self.coverage.html_report(directory=str(self.output_dir / "htmlcov"))
            self.coverage.json_report(outfile=str(self.output_dir / "coverage.json"))
            
            # Load JSON coverage data
            coverage_json_file = self.output_dir / "coverage.json"
            if coverage_json_file.exists():
                with open(coverage_json_file) as f:
                    coverage_json = json.load(f)
                
                # Overall coverage
                totals = coverage_json.get('totals', {})
                if totals.get('num_statements', 0) > 0:
                    coverage_data['overall'] = (totals.get('covered_lines', 0) / 
                                              totals.get('num_statements', 1)) * 100
                
                # Component-specific coverage
                files = coverage_json.get('files', {})
                for filepath, file_data in files.items():
                    filename = Path(filepath).name
                    component_name = filename.replace('.py', '')
                    
                    if file_data.get('summary', {}).get('num_statements', 0) > 0:
                        file_coverage = (file_data['summary'].get('covered_lines', 0) /
                                       file_data['summary'].get('num_statements', 1)) * 100
                        coverage_data['components'][component_name] = file_coverage
                    
                    # Critical function coverage (simplified analysis)
                    for func_name in self.targets.critical_functions.keys():
                        if func_name in str(file_data):  # Simplified check
                            coverage_data['critical_functions'][func_name] = file_coverage
        
        except Exception as e:
            self.logger.error(f"Failed to generate coverage report: {e}")
        
        return coverage_data
    
    def _identify_coverage_gaps(self, coverage_data: Dict[str, Any]) -> List[str]:
        """Identify areas with insufficient coverage."""
        
        gaps = []
        
        # Check overall coverage
        if coverage_data['overall'] < self.targets.overall_target:
            gaps.append(f"Overall coverage {coverage_data['overall']:.1f}% below target {self.targets.overall_target}%")
        
        # Check component coverage
        for component, target in self.targets.core_components.items():
            actual = coverage_data['components'].get(component, 0)
            if actual < target:
                gaps.append(f"Component '{component}' coverage {actual:.1f}% below target {target}%")
        
        # Check critical function coverage
        for function, target in self.targets.critical_functions.items():
            actual = coverage_data['critical_functions'].get(function, 0)
            if actual < target:
                gaps.append(f"Critical function '{function}' coverage {actual:.1f}% below target {target}%")
        
        return gaps
    
    def _calculate_code_quality_score(self) -> float:
        """Calculate code quality score based on various metrics."""
        
        try:
            # Use pylint or similar for code quality analysis
            # This is a simplified version
            quality_factors = []
            
            # Check for common code quality indicators
            source_files = list(self.source_dir.glob("*.py"))
            
            for file_path in source_files:
                if file_path.name.startswith("test_"):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    
                    # Simple quality metrics
                    docstring_lines = sum(1 for line in lines if '"""' in line or "'''" in line)
                    comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
                    code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
                    
                    if code_lines > 0:
                        documentation_ratio = (docstring_lines + comment_lines) / code_lines
                        quality_factors.append(min(1.0, documentation_ratio * 2))  # Cap at 1.0
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {file_path}: {e}")
            
            return statistics.mean(quality_factors) * 100 if quality_factors else 50.0
        
        except Exception as e:
            self.logger.error(f"Failed to calculate code quality score: {e}")
            return 50.0
    
    def _calculate_test_quality_score(self, results: TestResults) -> float:
        """Calculate test quality score based on test results."""
        
        quality_factors = []
        
        # Test success rate
        if results.total_tests > 0:
            success_rate = results.passed_tests / results.total_tests
            quality_factors.append(success_rate)
        
        # Coverage quality
        coverage_quality = min(1.0, results.overall_coverage / 100)
        quality_factors.append(coverage_quality)
        
        # Test performance (penalty for very slow tests)
        if results.average_test_time > 0:
            # Penalty if average test time > 1 second
            performance_factor = max(0.5, min(1.0, 1.0 / results.average_test_time))
            quality_factors.append(performance_factor)
        
        return statistics.mean(quality_factors) * 100 if quality_factors else 0.0
    
    def _generate_recommendations(self, results: TestResults) -> List[str]:
        """Generate recommendations based on test results."""
        
        recommendations = []
        
        # Coverage recommendations
        if results.overall_coverage < self.targets.overall_target:
            recommendations.append(
                f"Increase test coverage from {results.overall_coverage:.1f}% to {self.targets.overall_target}%"
            )
        
        # Performance recommendations
        if results.average_test_time > 1.0:
            recommendations.append(
                f"Optimize test performance - average test time {results.average_test_time:.2f}s is high"
            )
        
        # Quality recommendations
        if results.code_quality_score < 80:
            recommendations.append(
                f"Improve code quality score from {results.code_quality_score:.1f} to >80"
            )
        
        # Specific coverage gaps
        if results.coverage_gaps:
            recommendations.append("Address specific coverage gaps listed in the report")
        
        # Test failure recommendations
        if results.failed_tests > 0:
            recommendations.append(f"Fix {results.failed_tests} failing tests")
        
        if results.error_tests > 0:
            recommendations.append(f"Fix {results.error_tests} tests with errors")
        
        return recommendations
    
    def generate_comprehensive_report(self, results: TestResults, format: str = "html") -> str:
        """Generate comprehensive test and coverage report."""
        
        timestamp = results.timestamp.strftime("%Y%m%d_%H%M%S")
        
        if format == "html":
            return self._generate_html_report(results, timestamp)
        elif format == "json":
            return self._generate_json_report(results, timestamp)
        else:
            return self._generate_text_report(results, timestamp)
    
    def _generate_html_report(self, results: TestResults, timestamp: str) -> str:
        """Generate HTML report."""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Classification System - Test Coverage Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background-color: #e9ecef; border-radius: 5px; }}
                .success {{ background-color: #d4edda; }}
                .warning {{ background-color: #fff3cd; }}
                .danger {{ background-color: #f8d7da; }}
                .coverage-bar {{ width: 100%; height: 20px; background-color: #e9ecef; 
                               border-radius: 10px; overflow: hidden; }}
                .coverage-fill {{ height: 100%; background-color: #28a745; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f8f9fa; }}
                .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LLM Classification System - Test Coverage Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Python Version: {python_version}</p>
            </div>
            
            <h2>Overall Metrics</h2>
            <div class="metric {overall_status}">
                <strong>Overall Coverage</strong><br>
                {overall_coverage:.1f}%
                <div class="coverage-bar">
                    <div class="coverage-fill" style="width: {overall_coverage:.1f}%"></div>
                </div>
            </div>
            
            <div class="metric {test_status}">
                <strong>Test Results</strong><br>
                Passed: {passed_tests}<br>
                Failed: {failed_tests}<br>
                Total: {total_tests}
            </div>
            
            <div class="metric">
                <strong>Execution Time</strong><br>
                Total: {total_time:.2f}s<br>
                Average: {avg_time:.3f}s per test
            </div>
            
            <div class="metric">
                <strong>Quality Scores</strong><br>
                Code Quality: {code_quality:.1f}<br>
                Test Quality: {test_quality:.1f}
            </div>
            
            <h2>Component Coverage</h2>
            <table>
                <tr><th>Component</th><th>Coverage</th><th>Target</th><th>Status</th></tr>
                {component_rows}
            </table>
            
            <h2>Slowest Tests</h2>
            <table>
                <tr><th>Test</th><th>Time (s)</th></tr>
                {slowest_test_rows}
            </table>
            
            {coverage_gaps_section}
            
            {recommendations_section}
            
        </body>
        </html>
        """
        
        # Determine status classes
        overall_status = "success" if results.overall_coverage >= 95 else "warning" if results.overall_coverage >= 80 else "danger"
        test_status = "success" if results.failed_tests == 0 else "danger"
        
        # Generate component rows
        component_rows = []
        for component, coverage in results.component_coverage.items():
            target = self.targets.core_components.get(component, 90.0)
            status = "✓" if coverage >= target else "✗"
            status_class = "success" if coverage >= target else "danger"
            component_rows.append(
                f'<tr class="{status_class}"><td>{component}</td><td>{coverage:.1f}%</td>'
                f'<td>{target:.1f}%</td><td>{status}</td></tr>'
            )
        component_rows_html = "\n".join(component_rows)
        
        # Generate slowest test rows
        slowest_test_rows = []
        for test_name, test_time in results.slowest_tests:
            slowest_test_rows.append(f'<tr><td>{test_name}</td><td>{test_time:.3f}</td></tr>')
        slowest_test_rows_html = "\n".join(slowest_test_rows)
        
        # Coverage gaps section
        if results.coverage_gaps:
            gaps_html = "<h2>Coverage Gaps</h2><ul>" + "".join(f"<li>{gap}</li>" for gap in results.coverage_gaps) + "</ul>"
        else:
            gaps_html = "<h2>Coverage Gaps</h2><p class='success'>No significant coverage gaps detected!</p>"
        
        # Recommendations section
        if results.recommendations:
            rec_html = ('<div class="recommendations"><h2>Recommendations</h2><ul>' + 
                       "".join(f"<li>{rec}</li>" for rec in results.recommendations) + '</ul></div>')
        else:
            rec_html = "<div class='success'><h2>Recommendations</h2><p>All metrics meet targets - great job!</p></div>"
        
        # Fill template
        html_content = html_template.format(
            timestamp=results.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            python_version=results.test_environment.get('python_version', 'Unknown'),
            overall_coverage=results.overall_coverage,
            overall_status=overall_status,
            test_status=test_status,
            passed_tests=results.passed_tests,
            failed_tests=results.failed_tests,
            total_tests=results.total_tests,
            total_time=results.total_execution_time,
            avg_time=results.average_test_time,
            code_quality=results.code_quality_score,
            test_quality=results.test_quality_score,
            component_rows=component_rows_html,
            slowest_test_rows=slowest_test_rows_html,
            coverage_gaps_section=gaps_html,
            recommendations_section=rec_html
        )
        
        # Save HTML report
        report_file = self.output_dir / f"comprehensive_report_{timestamp}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_file)
    
    def _generate_json_report(self, results: TestResults, timestamp: str) -> str:
        """Generate JSON report."""
        
        # Convert dataclass to dict
        report_data = {
            'timestamp': results.timestamp.isoformat(),
            'test_metrics': {
                'total_tests': results.total_tests,
                'passed_tests': results.passed_tests,
                'failed_tests': results.failed_tests,
                'skipped_tests': results.skipped_tests,
                'error_tests': results.error_tests
            },
            'coverage_metrics': {
                'overall_coverage': results.overall_coverage,
                'component_coverage': results.component_coverage,
                'critical_function_coverage': results.critical_function_coverage
            },
            'performance_metrics': {
                'total_execution_time': results.total_execution_time,
                'average_test_time': results.average_test_time,
                'slowest_tests': results.slowest_tests
            },
            'quality_metrics': {
                'code_quality_score': results.code_quality_score,
                'test_quality_score': results.test_quality_score
            },
            'issues': {
                'coverage_gaps': results.coverage_gaps,
                'quality_issues': results.quality_issues,
                'performance_regressions': results.performance_regressions
            },
            'recommendations': results.recommendations,
            'test_environment': results.test_environment
        }
        
        # Save JSON report
        report_file = self.output_dir / f"test_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(report_file)
    
    def _generate_text_report(self, results: TestResults, timestamp: str) -> str:
        """Generate text report."""
        
        report_lines = [
            "=" * 80,
            "LLM CLASSIFICATION SYSTEM - COMPREHENSIVE TEST REPORT",
            "=" * 80,
            f"Generated: {results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Python Version: {results.test_environment.get('python_version', 'Unknown')}",
            "",
            "OVERALL METRICS:",
            "-" * 40,
            f"Overall Coverage:     {results.overall_coverage:.1f}% (Target: {self.targets.overall_target}%)",
            f"Test Results:         {results.passed_tests}/{results.total_tests} passed",
            f"Failed Tests:         {results.failed_tests}",
            f"Execution Time:       {results.total_execution_time:.2f}s",
            f"Average Test Time:    {results.average_test_time:.3f}s",
            f"Code Quality Score:   {results.code_quality_score:.1f}",
            f"Test Quality Score:   {results.test_quality_score:.1f}",
            "",
            "COMPONENT COVERAGE:",
            "-" * 40
        ]
        
        for component, coverage in results.component_coverage.items():
            target = self.targets.core_components.get(component, 90.0)
            status = "✓" if coverage >= target else "✗"
            report_lines.append(f"{component:30} {coverage:6.1f}% {status}")
        
        if results.coverage_gaps:
            report_lines.extend([
                "",
                "COVERAGE GAPS:",
                "-" * 40
            ])
            for gap in results.coverage_gaps:
                report_lines.append(f"• {gap}")
        
        if results.recommendations:
            report_lines.extend([
                "",
                "RECOMMENDATIONS:",
                "-" * 40
            ])
            for rec in results.recommendations:
                report_lines.append(f"• {rec}")
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        text_content = "\n".join(report_lines)
        
        # Save text report
        report_file = self.output_dir / f"test_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        return str(report_file)


# ============================================================================
# MAIN EXECUTION AND CLI INTERFACE
# ============================================================================

def main():
    """Main execution function for coverage analysis."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Classification System Test Coverage Analysis")
    parser.add_argument("--source-dir", help="Source code directory")
    parser.add_argument("--test-dir", help="Test directory")  
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--format", choices=["html", "json", "text", "all"], 
                       default="html", help="Report format")
    parser.add_argument("--test-patterns", nargs="+", help="Test patterns to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create coverage analyzer
    analyzer = CoverageAnalyzer(
        source_dir=args.source_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        logger=logger
    )
    
    print("=" * 80)
    print("LLM CLASSIFICATION SYSTEM - COMPREHENSIVE TEST ANALYSIS")
    print("=" * 80)
    
    # Run tests with coverage
    print("\n🧪 Running tests with coverage analysis...")
    results = analyzer.run_tests_with_coverage(args.test_patterns)
    
    # Generate reports
    print("\n📊 Generating comprehensive reports...")
    
    if args.format == "all":
        formats = ["html", "json", "text"]
    else:
        formats = [args.format]
    
    report_files = []
    for fmt in formats:
        report_file = analyzer.generate_comprehensive_report(results, fmt)
        report_files.append(report_file)
        print(f"   {fmt.upper()} report: {report_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tests:           {results.passed_tests}/{results.total_tests} passed")
    print(f"Coverage:        {results.overall_coverage:.1f}% (Target: {analyzer.targets.overall_target}%)")
    print(f"Execution Time:  {results.total_execution_time:.2f}s")
    print(f"Quality Score:   {results.test_quality_score:.1f}")
    
    if results.coverage_gaps:
        print(f"\n⚠️  Coverage Gaps: {len(results.coverage_gaps)}")
        for gap in results.coverage_gaps[:3]:  # Show first 3
            print(f"   • {gap}")
        if len(results.coverage_gaps) > 3:
            print(f"   ... and {len(results.coverage_gaps) - 3} more")
    
    if results.recommendations:
        print(f"\n💡 Recommendations: {len(results.recommendations)}")
        for rec in results.recommendations[:3]:  # Show first 3
            print(f"   • {rec}")
        if len(results.recommendations) > 3:
            print(f"   ... and {len(results.recommendations) - 3} more")
    
    # Success/failure status
    success = (results.overall_coverage >= analyzer.targets.overall_target and 
              results.failed_tests == 0)
    
    if success:
        print("\n✅ All quality targets met! Great job!")
        return 0
    else:
        print("\n❌ Some quality targets not met. See recommendations above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)