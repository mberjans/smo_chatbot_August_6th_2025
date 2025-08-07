#!/usr/bin/env python3
"""
Performance Benchmark Test Runner for CMO-LIGHTRAG-008-T05

This script provides a comprehensive test runner for performance benchmarks
that integrates with the existing pytest infrastructure and generates
detailed performance reports for the Clinical Metabolomics Oracle.

Features:
- Command-line interface for different benchmark modes
- Integration with existing performance test infrastructure
- Comprehensive reporting with actionable insights
- Performance regression detection
- CI/CD integration support
- Export capabilities for different formats

Usage:
    python run_performance_benchmarks.py [options]

Options:
    --mode: benchmark mode (quick, full, regression, custom)
    --output-dir: output directory for reports
    --export-format: export format (json, html, csv)
    --thresholds: custom performance thresholds file
    --verbose: enable verbose output
    --ci-mode: run in CI/CD mode with exit codes

Examples:
    python run_performance_benchmarks.py --mode quick
    python run_performance_benchmarks.py --mode full --output-dir ./reports
    python run_performance_benchmarks.py --mode regression --thresholds custom_thresholds.json

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import argparse
import sys
import json
import logging
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import subprocess

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import test components
try:
    from test_performance_benchmarks import (
        PerformanceBenchmarkSuite,
        BenchmarkReportGenerator,
        BenchmarkTarget
    )
    BENCHMARK_SUITE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import benchmark suite: {e}")
    BENCHMARK_SUITE_AVAILABLE = False

try:
    from performance_test_fixtures import (
        PerformanceTestExecutor,
        LoadTestScenarioGenerator
    )
    PERFORMANCE_FIXTURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Performance fixtures not available: {e}")
    PERFORMANCE_FIXTURES_AVAILABLE = False


class BenchmarkRunner:
    """
    Main benchmark runner that orchestrates performance testing.
    """
    
    def __init__(self, output_dir: Path = None, verbose: bool = False):
        self.output_dir = output_dir or Path("lightrag_integration/tests/performance_test_results")
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize components
        self.benchmark_suite = None
        if BENCHMARK_SUITE_AVAILABLE:
            self.benchmark_suite = PerformanceBenchmarkSuite()
        
        self.results: Dict[str, Any] = {}
        self.start_time = None
        self.end_time = None
    
    async def run_quick_benchmarks(self) -> Dict[str, Any]:
        """Run quick performance benchmarks (subset of full suite)."""
        logger.info("Starting quick performance benchmarks")
        
        if not self.benchmark_suite:
            return self._create_mock_results("quick", "benchmark_suite_unavailable")
        
        self.start_time = time.time()
        
        # Run essential benchmarks only
        quick_benchmarks = [
            ('simple_query', self.benchmark_suite._run_simple_query_benchmark),
            ('medium_query', self.benchmark_suite._run_medium_query_benchmark),
            ('concurrent_users', self.benchmark_suite._run_concurrent_users_benchmark)
        ]
        
        results = {
            'benchmark_mode': 'quick',
            'start_time': self.start_time,
            'benchmarks': [],
            'summary': {}
        }
        
        for benchmark_name, benchmark_method in quick_benchmarks:
            try:
                logger.info(f"Running {benchmark_name} benchmark...")
                benchmark_result = await benchmark_method()
                benchmark_result['benchmark_id'] = benchmark_name
                results['benchmarks'].append(benchmark_result)
                logger.info(f"✓ {benchmark_name} completed: {benchmark_result.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"✗ {benchmark_name} failed: {e}")
                results['benchmarks'].append({
                    'benchmark_id': benchmark_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        self.end_time = time.time()
        results['end_time'] = self.end_time
        results['duration_seconds'] = self.end_time - self.start_time
        results['summary'] = self._calculate_summary(results['benchmarks'])
        
        self.results = results
        return results
    
    async def run_full_benchmarks(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite."""
        logger.info("Starting full performance benchmark suite")
        
        if not self.benchmark_suite:
            return self._create_mock_results("full", "benchmark_suite_unavailable")
        
        self.start_time = time.time()
        
        # Run complete benchmark suite
        results = await self.benchmark_suite.run_benchmark_suite()
        
        self.end_time = time.time()
        self.results = results
        return results
    
    async def run_regression_benchmarks(self, baseline_file: Optional[Path] = None) -> Dict[str, Any]:
        """Run regression testing against baseline results."""
        logger.info("Starting performance regression benchmarks")
        
        # Load baseline results if provided
        baseline_results = None
        if baseline_file and baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    baseline_results = json.load(f)
                logger.info(f"Loaded baseline results from {baseline_file}")
            except Exception as e:
                logger.warning(f"Failed to load baseline results: {e}")
        
        # Run current benchmarks
        current_results = await self.run_full_benchmarks()
        
        # Perform regression analysis
        regression_analysis = self._analyze_regression(baseline_results, current_results)
        
        # Enhance results with regression analysis
        current_results['regression_analysis'] = regression_analysis
        current_results['baseline_file'] = str(baseline_file) if baseline_file else None
        
        return current_results
    
    def run_pytest_benchmarks(self, test_markers: List[str] = None) -> Dict[str, Any]:
        """Run benchmarks using pytest infrastructure."""
        logger.info("Running benchmarks via pytest")
        
        # Construct pytest command
        pytest_cmd = [
            "python", "-m", "pytest",
            "test_performance_benchmarks.py",
            "-v",
            "--tb=short",
            f"--junitxml={self.output_dir}/pytest_performance_results.xml"
        ]
        
        # Add test markers
        if test_markers:
            for marker in test_markers:
                pytest_cmd.extend(["-m", marker])
        else:
            pytest_cmd.extend(["-m", "performance"])
        
        # Add output directory
        pytest_cmd.extend(["--output-dir", str(self.output_dir)])
        
        try:
            logger.info(f"Executing: {' '.join(pytest_cmd)}")
            result = subprocess.run(
                pytest_cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            return {
                'benchmark_mode': 'pytest',
                'execution_time': time.time(),
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(pytest_cmd),
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Pytest benchmarks timed out")
            return {
                'benchmark_mode': 'pytest',
                'execution_time': time.time(),
                'return_code': 124,  # Timeout return code
                'error': 'Execution timed out after 30 minutes',
                'success': False
            }
        except Exception as e:
            logger.error(f"Failed to execute pytest benchmarks: {e}")
            return {
                'benchmark_mode': 'pytest',
                'execution_time': time.time(),
                'return_code': 1,
                'error': str(e),
                'success': False
            }
    
    def _create_mock_results(self, mode: str, reason: str) -> Dict[str, Any]:
        """Create mock results when benchmark suite is unavailable."""
        return {
            'benchmark_mode': mode,
            'start_time': time.time(),
            'end_time': time.time() + 1,
            'duration_seconds': 1.0,
            'status': 'mock',
            'reason': reason,
            'benchmarks': [],
            'summary': {
                'total_benchmarks': 0,
                'passed_benchmarks': 0,
                'failed_benchmarks': 0,
                'success_rate_percent': 0.0,
                'overall_grade': 'Unavailable'
            }
        }
    
    def _calculate_summary(self, benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate benchmark summary statistics."""
        total_benchmarks = len(benchmarks)
        passed_benchmarks = sum(1 for b in benchmarks if b.get('status') == 'passed')
        failed_benchmarks = total_benchmarks - passed_benchmarks
        
        success_rate = (passed_benchmarks / total_benchmarks * 100) if total_benchmarks > 0 else 0
        
        # Determine overall grade
        if success_rate >= 90:
            overall_grade = "Excellent"
        elif success_rate >= 80:
            overall_grade = "Good"
        elif success_rate >= 70:
            overall_grade = "Satisfactory"
        elif success_rate >= 50:
            overall_grade = "Needs Improvement"
        else:
            overall_grade = "Poor"
        
        return {
            'total_benchmarks': total_benchmarks,
            'passed_benchmarks': passed_benchmarks,
            'failed_benchmarks': failed_benchmarks,
            'success_rate_percent': success_rate,
            'overall_grade': overall_grade
        }
    
    def _analyze_regression(self, baseline: Optional[Dict], current: Dict) -> Dict[str, Any]:
        """Analyze performance regression between baseline and current results."""
        if not baseline:
            return {
                'status': 'no_baseline',
                'message': 'No baseline results available for comparison'
            }
        
        regression_analysis = {
            'status': 'analyzed',
            'baseline_summary': baseline.get('summary', {}),
            'current_summary': current.get('summary', {}),
            'regressions': [],
            'improvements': [],
            'summary': {}
        }
        
        # Compare overall success rates
        baseline_success = baseline.get('summary', {}).get('success_rate_percent', 0)
        current_success = current.get('summary', {}).get('success_rate_percent', 0)
        
        success_rate_change = current_success - baseline_success
        
        if success_rate_change < -10:  # More than 10% drop in success rate
            regression_analysis['regressions'].append({
                'metric': 'success_rate_percent',
                'baseline': baseline_success,
                'current': current_success,
                'change': success_rate_change,
                'severity': 'critical'
            })
        elif success_rate_change > 10:  # More than 10% improvement
            regression_analysis['improvements'].append({
                'metric': 'success_rate_percent',
                'baseline': baseline_success,
                'current': current_success,
                'change': success_rate_change
            })
        
        # Analyze individual benchmark performance
        baseline_benchmarks = {b.get('benchmark_id', b.get('benchmark_type', 'unknown')): b 
                             for b in baseline.get('benchmarks', [])}
        current_benchmarks = {b.get('benchmark_id', b.get('benchmark_type', 'unknown')): b 
                            for b in current.get('benchmarks', [])}
        
        for benchmark_id in baseline_benchmarks:
            if benchmark_id in current_benchmarks:
                baseline_benchmark = baseline_benchmarks[benchmark_id]
                current_benchmark = current_benchmarks[benchmark_id]
                
                # Compare response times if available
                baseline_latency = baseline_benchmark.get('metrics', {}).get('average_latency_ms', 0)
                current_latency = current_benchmark.get('metrics', {}).get('average_latency_ms', 0)
                
                if baseline_latency > 0 and current_latency > 0:
                    latency_change_percent = ((current_latency - baseline_latency) / baseline_latency) * 100
                    
                    if latency_change_percent > 20:  # More than 20% increase in latency
                        regression_analysis['regressions'].append({
                            'benchmark': benchmark_id,
                            'metric': 'average_latency_ms',
                            'baseline': baseline_latency,
                            'current': current_latency,
                            'change_percent': latency_change_percent,
                            'severity': 'high' if latency_change_percent > 50 else 'medium'
                        })
        
        # Summary
        regression_analysis['summary'] = {
            'total_regressions': len(regression_analysis['regressions']),
            'total_improvements': len(regression_analysis['improvements']),
            'critical_regressions': len([r for r in regression_analysis['regressions'] 
                                       if r.get('severity') == 'critical']),
            'recommendation': self._get_regression_recommendation(regression_analysis)
        }
        
        return regression_analysis
    
    def _get_regression_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Get recommendation based on regression analysis."""
        regressions = analysis.get('regressions', [])
        improvements = analysis.get('improvements', [])
        
        if not regressions and not improvements:
            return "Performance remains stable compared to baseline"
        elif not regressions and improvements:
            return "Performance has improved compared to baseline"
        elif len([r for r in regressions if r.get('severity') == 'critical']) > 0:
            return "Critical performance regressions detected - immediate investigation required"
        elif len(regressions) > len(improvements):
            return "Performance regressions detected - optimization recommended"
        else:
            return "Mixed performance changes - review individual benchmarks"
    
    def generate_reports(self, results: Dict[str, Any], export_formats: List[str] = None) -> List[Path]:
        """Generate benchmark reports in specified formats."""
        if export_formats is None:
            export_formats = ['json']
        
        generated_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        if 'json' in export_formats:
            json_file = self.output_dir / f"Performance_Benchmark_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            generated_files.append(json_file)
            logger.info(f"JSON report generated: {json_file}")
        
        # Generate summary text report
        if 'txt' in export_formats:
            txt_file = self.output_dir / f"Performance_Benchmark_{timestamp}_summary.txt"
            self._generate_text_summary(results, txt_file)
            generated_files.append(txt_file)
            logger.info(f"Text summary generated: {txt_file}")
        
        # Generate CSV report for metrics
        if 'csv' in export_formats:
            csv_file = self.output_dir / f"Performance_Benchmark_{timestamp}_metrics.csv"
            self._generate_csv_report(results, csv_file)
            generated_files.append(csv_file)
            logger.info(f"CSV report generated: {csv_file}")
        
        # Generate HTML report
        if 'html' in export_formats:
            html_file = self.output_dir / f"Performance_Benchmark_{timestamp}_report.html"
            self._generate_html_report(results, html_file)
            generated_files.append(html_file)
            logger.info(f"HTML report generated: {html_file}")
        
        return generated_files
    
    def _generate_text_summary(self, results: Dict[str, Any], output_file: Path):
        """Generate text summary report."""
        summary = results.get('summary', {})
        
        content = f"""
Clinical Metabolomics Oracle Performance Benchmark Report
=========================================================

Generated: {datetime.now().isoformat()}
Task: CMO-LIGHTRAG-008-T05
Mode: {results.get('benchmark_mode', 'unknown')}
Duration: {results.get('duration_seconds', 0):.1f} seconds

OVERALL RESULTS
---------------
Total Benchmarks: {summary.get('total_benchmarks', 0)}
Passed: {summary.get('passed_benchmarks', 0)}
Failed: {summary.get('failed_benchmarks', 0)}
Success Rate: {summary.get('success_rate_percent', 0):.1f}%
Overall Grade: {summary.get('overall_grade', 'Unknown')}

"""
        
        # Add benchmark details
        benchmarks = results.get('benchmarks', [])
        if benchmarks:
            content += "BENCHMARK DETAILS\n"
            content += "-----------------\n"
            for benchmark in benchmarks:
                benchmark_id = benchmark.get('benchmark_id', 'unknown')
                status = benchmark.get('status', 'unknown')
                content += f"{benchmark_id}: {status.upper()}\n"
                
                if 'metrics' in benchmark:
                    metrics = benchmark['metrics']
                    content += f"  Response Time: {metrics.get('average_latency_ms', 0):.1f}ms\n"
                    content += f"  Throughput: {metrics.get('throughput_ops_per_sec', 0):.2f} ops/sec\n"
                    content += f"  Error Rate: {metrics.get('error_rate_percent', 0):.1f}%\n"
                content += "\n"
        
        # Add regression analysis if available
        if 'regression_analysis' in results:
            regression = results['regression_analysis']
            content += "REGRESSION ANALYSIS\n"
            content += "-------------------\n"
            content += f"Status: {regression.get('status', 'unknown')}\n"
            content += f"Total Regressions: {regression.get('summary', {}).get('total_regressions', 0)}\n"
            content += f"Critical Issues: {regression.get('summary', {}).get('critical_regressions', 0)}\n"
            content += f"Recommendation: {regression.get('summary', {}).get('recommendation', 'None')}\n"
        
        with open(output_file, 'w') as f:
            f.write(content)
    
    def _generate_csv_report(self, results: Dict[str, Any], output_file: Path):
        """Generate CSV report with benchmark metrics."""
        import csv
        
        benchmarks = results.get('benchmarks', [])
        if not benchmarks:
            return
        
        with open(output_file, 'w', newline='') as f:
            fieldnames = [
                'benchmark_id', 'status', 'response_time_ms', 'throughput_ops_per_sec',
                'memory_usage_mb', 'error_rate_percent', 'operations_count'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for benchmark in benchmarks:
                metrics = benchmark.get('metrics', {})
                row = {
                    'benchmark_id': benchmark.get('benchmark_id', 'unknown'),
                    'status': benchmark.get('status', 'unknown'),
                    'response_time_ms': metrics.get('average_latency_ms', 0),
                    'throughput_ops_per_sec': metrics.get('throughput_ops_per_sec', 0),
                    'memory_usage_mb': metrics.get('memory_usage_mb', 0),
                    'error_rate_percent': metrics.get('error_rate_percent', 0),
                    'operations_count': metrics.get('operations_count', 0)
                }
                writer.writerow(row)
    
    def _generate_html_report(self, results: Dict[str, Any], output_file: Path):
        """Generate HTML report with charts and visualizations."""
        summary = results.get('summary', {})
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; }}
        .summary {{ background-color: #f9f9f9; padding: 10px; margin: 15px 0; border-radius: 5px; }}
        .benchmark {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .passed {{ border-left: 5px solid #28a745; }}
        .failed {{ border-left: 5px solid #dc3545; }}
        .metric {{ margin: 5px 0; }}
        .grade-excellent {{ color: #28a745; }}
        .grade-good {{ color: #17a2b8; }}
        .grade-satisfactory {{ color: #ffc107; }}
        .grade-poor {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Clinical Metabolomics Oracle Performance Benchmark Report</h1>
        <p><strong>Task:</strong> CMO-LIGHTRAG-008-T05</p>
        <p><strong>Generated:</strong> {datetime.now().isoformat()}</p>
        <p><strong>Mode:</strong> {results.get('benchmark_mode', 'unknown')}</p>
        <p><strong>Duration:</strong> {results.get('duration_seconds', 0):.1f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Overall Results</h2>
        <p><strong>Total Benchmarks:</strong> {summary.get('total_benchmarks', 0)}</p>
        <p><strong>Passed:</strong> {summary.get('passed_benchmarks', 0)}</p>
        <p><strong>Failed:</strong> {summary.get('failed_benchmarks', 0)}</p>
        <p><strong>Success Rate:</strong> {summary.get('success_rate_percent', 0):.1f}%</p>
        <p><strong>Overall Grade:</strong> <span class="grade-{summary.get('overall_grade', '').lower()}">{summary.get('overall_grade', 'Unknown')}</span></p>
    </div>
    
    <h2>Benchmark Details</h2>
"""
        
        # Add benchmark details
        benchmarks = results.get('benchmarks', [])
        for benchmark in benchmarks:
            benchmark_id = benchmark.get('benchmark_id', 'unknown')
            status = benchmark.get('status', 'unknown')
            status_class = 'passed' if status == 'passed' else 'failed'
            
            html_content += f"""
    <div class="benchmark {status_class}">
        <h3>{benchmark_id.title().replace('_', ' ')} - {status.upper()}</h3>
"""
            
            if 'metrics' in benchmark:
                metrics = benchmark['metrics']
                html_content += f"""
        <div class="metric"><strong>Response Time:</strong> {metrics.get('average_latency_ms', 0):.1f}ms</div>
        <div class="metric"><strong>Throughput:</strong> {metrics.get('throughput_ops_per_sec', 0):.2f} ops/sec</div>
        <div class="metric"><strong>Memory Usage:</strong> {metrics.get('memory_usage_mb', 0):.1f}MB</div>
        <div class="metric"><strong>Error Rate:</strong> {metrics.get('error_rate_percent', 0):.1f}%</div>
        <div class="metric"><strong>Operations:</strong> {metrics.get('operations_count', 0)}</div>
"""
            
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Performance Benchmark Test Runner for CMO-LIGHTRAG-008-T05",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_performance_benchmarks.py --mode quick
  python run_performance_benchmarks.py --mode full --export-format json,html
  python run_performance_benchmarks.py --mode regression --baseline results.json
  python run_performance_benchmarks.py --mode pytest --markers performance,slow
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'full', 'regression', 'pytest'],
        default='quick',
        help='Benchmark mode to run (default: quick)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('lightrag_integration/tests/performance_test_results'),
        help='Output directory for reports'
    )
    
    parser.add_argument(
        '--export-format',
        default='json,txt',
        help='Export formats (comma-separated): json,txt,csv,html'
    )
    
    parser.add_argument(
        '--baseline',
        type=Path,
        help='Baseline results file for regression testing'
    )
    
    parser.add_argument(
        '--markers',
        default='performance',
        help='Pytest markers for filtering (comma-separated)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--ci-mode',
        action='store_true',
        help='Run in CI/CD mode with appropriate exit codes'
    )
    
    args = parser.parse_args()
    
    # Parse export formats
    export_formats = [fmt.strip() for fmt in args.export_format.split(',')]
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    results = None
    success = False
    
    try:
        if args.mode == 'quick':
            results = asyncio.run(runner.run_quick_benchmarks())
            success = results.get('summary', {}).get('success_rate_percent', 0) >= 70
            
        elif args.mode == 'full':
            results = asyncio.run(runner.run_full_benchmarks())
            success = results.get('summary', {}).get('success_rate_percent', 0) >= 70
            
        elif args.mode == 'regression':
            results = asyncio.run(runner.run_regression_benchmarks(args.baseline))
            # Success criteria for regression: no critical regressions
            regression_analysis = results.get('regression_analysis', {})
            critical_regressions = regression_analysis.get('summary', {}).get('critical_regressions', 0)
            success = critical_regressions == 0
            
        elif args.mode == 'pytest':
            markers = [m.strip() for m in args.markers.split(',')]
            results = runner.run_pytest_benchmarks(markers)
            success = results.get('success', False)
        
        # Generate reports
        if results:
            generated_files = runner.generate_reports(results, export_formats)
            
            print("\nBenchmark Results Summary:")
            print("=" * 50)
            
            if args.mode == 'pytest':
                print(f"Mode: {args.mode}")
                print(f"Success: {success}")
                print(f"Return Code: {results.get('return_code', 'unknown')}")
                if results.get('stdout'):
                    print("\nPytest Output:")
                    print(results['stdout'])
            else:
                summary = results.get('summary', {})
                print(f"Mode: {args.mode}")
                print(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
                print(f"Passed: {summary.get('passed_benchmarks', 0)}")
                print(f"Failed: {summary.get('failed_benchmarks', 0)}")
                print(f"Success Rate: {summary.get('success_rate_percent', 0):.1f}%")
                print(f"Overall Grade: {summary.get('overall_grade', 'Unknown')}")
                
                if 'regression_analysis' in results:
                    regression = results['regression_analysis']
                    print(f"\nRegression Analysis:")
                    print(f"Total Regressions: {regression.get('summary', {}).get('total_regressions', 0)}")
                    print(f"Critical Issues: {regression.get('summary', {}).get('critical_regressions', 0)}")
            
            print(f"\nGenerated Reports:")
            for report_file in generated_files:
                print(f"  - {report_file}")
    
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        success = False
        
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    # Exit with appropriate code for CI/CD
    if args.ci_mode:
        sys.exit(0 if success else 1)
    
    return success


if __name__ == "__main__":
    main()