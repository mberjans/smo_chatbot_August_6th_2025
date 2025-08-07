#!/usr/bin/env python3
"""
Test Runner for Clinical Metabolomics Relevance Scoring System Tests.

This script provides a comprehensive test runner for the relevance scoring system
with various test execution modes, reporting, and performance analysis.

Features:
- Run all relevance scorer tests or specific test categories
- Generate detailed test reports with performance metrics
- Support for parallel test execution
- Coverage analysis and reporting
- Integration with existing test infrastructure

Usage:
    python run_relevance_scorer_tests.py [OPTIONS]

Options:
    --category CATEGORY    Run specific test category (e.g., dimensions, classification, performance)
    --parallel            Run tests in parallel mode
    --coverage            Generate coverage report
    --performance         Include performance benchmarking
    --verbose             Verbose output
    --report-format       Report format (json, html, text)
    --output-dir          Output directory for reports

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import sys
import os
import argparse
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import asyncio

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class TestRunConfig:
    """Configuration for test execution."""
    test_categories: List[str] = field(default_factory=list)
    parallel_execution: bool = False
    generate_coverage: bool = False
    include_performance: bool = False
    verbose: bool = False
    report_format: str = "text"
    output_dir: Path = field(default_factory=lambda: Path("test_results"))
    timeout_seconds: int = 300
    max_workers: int = 4

@dataclass
class TestResult:
    """Test execution result."""
    category: str
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    coverage_percent: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class RelevanceScorerTestRunner:
    """Test runner for relevance scorer tests."""
    
    def __init__(self, config: TestRunConfig):
        self.config = config
        self.results: List[TestResult] = []
        
        # Test categories and their corresponding test patterns
        self.test_categories = {
            'dimensions': 'TestIndividualScoringDimensions',
            'classification': 'TestQueryClassification',
            'quality': 'TestResponseQualityValidation',
            'weighting': 'TestAdaptiveWeightingSchemes',
            'edge_cases': 'TestEdgeCases',
            'performance': 'TestPerformance',
            'semantic': 'TestSemanticSimilarityEngine',
            'domain': 'TestDomainExpertiseValidator',
            'integration': 'TestIntegrationAndPipeline',
            'stress': 'TestStressAndRobustness',
            'config': 'TestConfigurationAndCustomization',
            'biomedical': 'TestBiomedicalDomainSpecifics',
            'all': '*'
        }
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_tests(self) -> Dict[str, Any]:
        """Run the specified tests and return results."""
        print("🔬 Clinical Metabolomics Relevance Scorer Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Determine which test categories to run
            categories_to_run = self._determine_test_categories()
            print(f"📋 Running test categories: {', '.join(categories_to_run)}")
            
            # Run tests for each category
            for category in categories_to_run:
                print(f"\n🧪 Running {category} tests...")
                result = self._run_test_category(category)
                self.results.append(result)
                
                # Print immediate feedback
                self._print_category_summary(result)
            
            # Generate overall summary
            total_duration = time.time() - start_time
            summary = self._generate_summary(total_duration)
            
            # Generate reports
            if self.config.report_format != "none":
                self._generate_reports(summary)
            
            print(f"\n✅ Test execution completed in {total_duration:.2f}s")
            return summary
            
        except Exception as e:
            print(f"\n❌ Test execution failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _determine_test_categories(self) -> List[str]:
        """Determine which test categories to run."""
        if not self.config.test_categories or 'all' in self.config.test_categories:
            return [cat for cat in self.test_categories.keys() if cat != 'all']
        
        # Validate requested categories
        valid_categories = []
        for category in self.config.test_categories:
            if category in self.test_categories:
                valid_categories.append(category)
            else:
                print(f"⚠️  Warning: Unknown test category '{category}', skipping")
        
        return valid_categories or ['all']
    
    def _run_test_category(self, category: str) -> TestResult:
        """Run tests for a specific category."""
        start_time = time.time()
        
        # Build pytest command
        cmd = self._build_pytest_command(category)
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Parse pytest output
            test_result = self._parse_pytest_output(result, category, duration)
            
            # Run coverage analysis if requested
            if self.config.generate_coverage and category != 'performance':
                test_result.coverage_percent = self._run_coverage_analysis(category)
            
            return test_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                category=category,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=duration,
                errors=[f"Test category '{category}' timed out after {self.config.timeout_seconds}s"]
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                category=category,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=duration,
                errors=[f"Error running category '{category}': {str(e)}"]
            )
    
    def _build_pytest_command(self, category: str) -> List[str]:
        """Build pytest command for category."""
        cmd = ["python", "-m", "pytest"]
        
        # Add test file
        test_file = Path(__file__).parent / "test_relevance_scorer.py"
        cmd.append(str(test_file))
        
        # Add category-specific selection
        if category != 'all':
            test_pattern = self.test_categories[category]
            if test_pattern != '*':
                cmd.extend(["-k", test_pattern])
        
        # Add common options
        cmd.extend([
            "-v",  # Verbose
            "--tb=short",  # Short traceback
        ])
        
        # Add parallel execution if requested
        if self.config.parallel_execution:
            cmd.extend(["-n", str(self.config.max_workers)])
        
        # Add performance options
        if category == 'performance' or self.config.include_performance:
            cmd.append("--durations=10")
        
        # Add markers to skip unavailable components
        cmd.extend(["-m", "not slow"])
        
        return cmd
    
    def _parse_pytest_output(self, result: subprocess.CompletedProcess, category: str, duration: float) -> TestResult:
        """Parse pytest output to extract test results."""
        output = result.stdout + result.stderr
        
        # Initialize counters
        passed = failed = skipped = 0
        errors = []
        warnings = []
        
        # Parse output lines
        for line in output.split('\n'):
            if '::' in line and ('PASSED' in line or 'FAILED' in line or 'SKIPPED' in line):
                if 'PASSED' in line:
                    passed += 1
                elif 'FAILED' in line:
                    failed += 1
                elif 'SKIPPED' in line:
                    skipped += 1
            elif 'ERROR' in line or 'Exception' in line:
                errors.append(line.strip())
            elif 'WARNING' in line or 'UserWarning' in line:
                warnings.append(line.strip())
        
        # Extract summary line if available
        for line in reversed(output.split('\n')):
            if 'passed' in line or 'failed' in line:
                # Try to extract numbers from summary line
                import re
                numbers = re.findall(r'(\d+) (passed|failed|skipped)', line)
                if numbers:
                    for count, status in numbers:
                        if status == 'passed':
                            passed = max(passed, int(count))
                        elif status == 'failed':
                            failed = max(failed, int(count))
                        elif status == 'skipped':
                            skipped = max(skipped, int(count))
                break
        
        return TestResult(
            category=category,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration_seconds=duration,
            errors=errors[:10],  # Limit to first 10 errors
            warnings=warnings[:5]  # Limit to first 5 warnings
        )
    
    def _run_coverage_analysis(self, category: str) -> Optional[float]:
        """Run coverage analysis for category."""
        try:
            cmd = [
                "python", "-m", "coverage", "run", "--source=relevance_scorer",
                "-m", "pytest", "test_relevance_scorer.py",
                "-k", self.test_categories[category],
                "-q"
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=60)
            
            # Get coverage report
            result = subprocess.run(
                ["python", "-m", "coverage", "report", "--show-missing"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse coverage percentage
            for line in result.stdout.split('\n'):
                if 'relevance_scorer' in line:
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            return float(part[:-1])
            
        except Exception as e:
            print(f"Coverage analysis failed for {category}: {e}")
        
        return None
    
    def _print_category_summary(self, result: TestResult):
        """Print summary for a test category."""
        total_tests = result.passed + result.failed + result.skipped
        
        if total_tests == 0:
            print(f"   📝 No tests found for category '{result.category}'")
            return
        
        # Status icon
        if result.failed > 0:
            status_icon = "❌"
        elif result.skipped > 0 and result.passed == 0:
            status_icon = "⏭️ "
        else:
            status_icon = "✅"
        
        print(f"   {status_icon} {result.category}: {result.passed} passed, {result.failed} failed, {result.skipped} skipped ({result.duration_seconds:.2f}s)")
        
        # Print coverage if available
        if result.coverage_percent is not None:
            print(f"      📊 Coverage: {result.coverage_percent:.1f}%")
        
        # Print first few errors
        if result.errors:
            print(f"      🚨 Errors: {len(result.errors)} total")
            for error in result.errors[:2]:  # Show first 2 errors
                print(f"         - {error[:100]}...")
    
    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate overall test summary."""
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_tests = total_passed + total_failed + total_skipped
        
        # Calculate average coverage
        coverages = [r.coverage_percent for r in self.results if r.coverage_percent is not None]
        avg_coverage = sum(coverages) / len(coverages) if coverages else None
        
        summary = {
            "status": "passed" if total_failed == 0 else "failed",
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "total_duration": total_duration,
            "average_coverage": avg_coverage,
            "categories": [
                {
                    "name": r.category,
                    "passed": r.passed,
                    "failed": r.failed,
                    "skipped": r.skipped,
                    "duration": r.duration_seconds,
                    "coverage": r.coverage_percent
                }
                for r in self.results
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "parallel": self.config.parallel_execution,
                "coverage": self.config.generate_coverage,
                "performance": self.config.include_performance
            }
        }
        
        # Print overall summary
        self._print_overall_summary(summary)
        
        return summary
    
    def _print_overall_summary(self, summary: Dict[str, Any]):
        """Print overall test summary."""
        print(f"\n📊 Overall Test Results:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['total_passed']} ✅")
        print(f"   Failed: {summary['total_failed']} ❌")
        print(f"   Skipped: {summary['total_skipped']} ⏭️")
        print(f"   Duration: {summary['total_duration']:.2f}s")
        
        if summary['average_coverage']:
            print(f"   Average Coverage: {summary['average_coverage']:.1f}%")
        
        # Status summary
        if summary['total_failed'] == 0:
            print(f"\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {summary['total_failed']} test(s) failed")
    
    def _generate_reports(self, summary: Dict[str, Any]):
        """Generate test reports in requested formats."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if self.config.report_format in ["json", "all"]:
            json_file = self.config.output_dir / f"relevance_scorer_test_report_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📄 JSON report saved to: {json_file}")
        
        if self.config.report_format in ["text", "all"]:
            text_file = self.config.output_dir / f"relevance_scorer_test_report_{timestamp}.txt"
            with open(text_file, 'w') as f:
                self._write_text_report(f, summary)
            print(f"📄 Text report saved to: {text_file}")
        
        if self.config.report_format in ["html", "all"]:
            html_file = self.config.output_dir / f"relevance_scorer_test_report_{timestamp}.html"
            with open(html_file, 'w') as f:
                self._write_html_report(f, summary)
            print(f"📄 HTML report saved to: {html_file}")
    
    def _write_text_report(self, file, summary: Dict[str, Any]):
        """Write text format report."""
        file.write("Clinical Metabolomics Relevance Scorer Test Report\n")
        file.write("=" * 60 + "\n\n")
        
        file.write(f"Test Execution Summary:\n")
        file.write(f"  Timestamp: {summary['timestamp']}\n")
        file.write(f"  Status: {summary['status'].upper()}\n")
        file.write(f"  Total Tests: {summary['total_tests']}\n")
        file.write(f"  Passed: {summary['total_passed']}\n")
        file.write(f"  Failed: {summary['total_failed']}\n")
        file.write(f"  Skipped: {summary['total_skipped']}\n")
        file.write(f"  Duration: {summary['total_duration']:.2f}s\n")
        
        if summary['average_coverage']:
            file.write(f"  Average Coverage: {summary['average_coverage']:.1f}%\n")
        
        file.write(f"\nTest Category Details:\n")
        file.write("-" * 40 + "\n")
        
        for cat in summary['categories']:
            file.write(f"\n{cat['name'].title()}:\n")
            file.write(f"  Passed: {cat['passed']}\n")
            file.write(f"  Failed: {cat['failed']}\n")
            file.write(f"  Skipped: {cat['skipped']}\n")
            file.write(f"  Duration: {cat['duration']:.2f}s\n")
            if cat['coverage']:
                file.write(f"  Coverage: {cat['coverage']:.1f}%\n")
        
        file.write(f"\nConfiguration:\n")
        file.write(f"  Parallel Execution: {summary['config']['parallel']}\n")
        file.write(f"  Coverage Analysis: {summary['config']['coverage']}\n")
        file.write(f"  Performance Testing: {summary['config']['performance']}\n")
    
    def _write_html_report(self, file, summary: Dict[str, Any]):
        """Write HTML format report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Relevance Scorer Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 8px; }}
        .summary {{ margin: 20px 0; }}
        .category {{ margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .skipped {{ color: #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .status-{'passed' if summary['status'] == 'passed' else 'failed'} {{ 
            color: {'#28a745' if summary['status'] == 'passed' else '#dc3545'}; 
            font-weight: bold; 
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Clinical Metabolomics Relevance Scorer Test Report</h1>
        <p>Generated: {summary['timestamp']}</p>
        <p class="status-{summary['status']}">Status: {summary['status'].upper()}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Test Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{summary['total_tests']}</td></tr>
            <tr><td class="passed">Passed</td><td>{summary['total_passed']}</td></tr>
            <tr><td class="failed">Failed</td><td>{summary['total_failed']}</td></tr>
            <tr><td class="skipped">Skipped</td><td>{summary['total_skipped']}</td></tr>
            <tr><td>Duration</td><td>{summary['total_duration']:.2f}s</td></tr>"""
        
        if summary['average_coverage']:
            html_content += f"<tr><td>Average Coverage</td><td>{summary['average_coverage']:.1f}%</td></tr>"
        
        html_content += """
        </table>
    </div>
    
    <div class="categories">
        <h2>📋 Test Categories</h2>"""
        
        for cat in summary['categories']:
            status_class = 'failed' if cat['failed'] > 0 else 'passed'
            html_content += f"""
        <div class="category">
            <h3>{cat['name'].title()}</h3>
            <p><span class="passed">✅ {cat['passed']} passed</span> | 
               <span class="failed">❌ {cat['failed']} failed</span> | 
               <span class="skipped">⏭️ {cat['skipped']} skipped</span></p>
            <p>⏱️ Duration: {cat['duration']:.2f}s</p>"""
            
            if cat['coverage']:
                html_content += f"<p>📊 Coverage: {cat['coverage']:.1f}%</p>"
            
            html_content += "</div>"
        
        html_content += """
    </div>
    
    <div class="config">
        <h2>⚙️ Configuration</h2>
        <ul>"""
        
        config = summary['config']
        html_content += f"""
            <li>Parallel Execution: {'✅' if config['parallel'] else '❌'}</li>
            <li>Coverage Analysis: {'✅' if config['coverage'] else '❌'}</li>
            <li>Performance Testing: {'✅' if config['performance'] else '❌'}</li>
        </ul>
    </div>
</body>
</html>"""
        
        file.write(html_content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Clinical Metabolomics Relevance Scorer Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available test categories:
  dimensions    - Individual scoring dimension tests
  classification - Query classification tests  
  quality       - Response quality validation tests
  weighting     - Adaptive weighting scheme tests
  edge_cases    - Edge cases and error handling
  performance   - Performance and timing tests
  semantic      - Semantic similarity engine tests
  domain        - Domain expertise validator tests
  integration   - Integration and pipeline tests
  stress        - Stress and robustness tests
  config        - Configuration and customization tests
  biomedical    - Biomedical domain-specific tests
  all           - All test categories

Examples:
  python run_relevance_scorer_tests.py
  python run_relevance_scorer_tests.py --category dimensions performance
  python run_relevance_scorer_tests.py --parallel --coverage --report-format json
        """
    )
    
    parser.add_argument(
        "--category", "-c", 
        nargs="+", 
        choices=["dimensions", "classification", "quality", "weighting", "edge_cases", 
                "performance", "semantic", "domain", "integration", "stress", "config", 
                "biomedical", "all"],
        default=["all"],
        help="Test categories to run (default: all)"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true", 
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Include performance benchmarking"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--report-format",
        choices=["json", "html", "text", "all", "none"],
        default="text",
        help="Report format (default: text)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Output directory for reports"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Test timeout in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = TestRunConfig(
        test_categories=args.category,
        parallel_execution=args.parallel,
        generate_coverage=args.coverage,
        include_performance=args.performance,
        verbose=args.verbose,
        report_format=args.report_format,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout,
        max_workers=args.workers
    )
    
    # Run tests
    runner = RelevanceScorerTestRunner(config)
    summary = runner.run_tests()
    
    # Exit with appropriate code
    exit_code = 0 if summary.get('status') == 'passed' else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()