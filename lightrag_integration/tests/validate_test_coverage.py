#!/usr/bin/env python3
"""
Test Coverage Analysis and Reporting for Factual Accuracy Validation System.

This script provides comprehensive test coverage analysis, validation, and reporting
for the factual accuracy validation system to ensure robust testing coverage and
identify areas needing additional tests.

Features:
- Code coverage analysis and reporting
- Test completeness validation
- Missing test identification
- Coverage quality assessment
- Integration with CI/CD pipelines
- Detailed HTML and JSON reports

Usage:
    python validate_test_coverage.py --help
    python validate_test_coverage.py --analyze
    python validate_test_coverage.py --validate --min-coverage 90
    python validate_test_coverage.py --report --format html

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import argparse
import ast
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from collections import defaultdict
import re
import inspect


@dataclass
class CoverageMetrics:
    """Coverage metrics for a module or function."""
    name: str
    lines_total: int
    lines_covered: int
    lines_missed: int
    coverage_percentage: float
    branch_total: int = 0
    branch_covered: int = 0
    branch_percentage: float = 0.0
    complexity: int = 0
    missing_lines: List[int] = None
    
    def __post_init__(self):
        if self.missing_lines is None:
            self.missing_lines = []


@dataclass
class TestCoverageReport:
    """Complete test coverage report."""
    timestamp: str
    overall_metrics: CoverageMetrics
    module_metrics: Dict[str, CoverageMetrics]
    function_metrics: Dict[str, Dict[str, CoverageMetrics]]
    test_completeness: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    recommendations: List[str]
    configuration: Dict[str, Any]


class ValidationTestCoverageAnalyzer:
    """Comprehensive test coverage analyzer for validation system."""
    
    def __init__(self):
        """Initialize coverage analyzer."""
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        self.source_dir = self.project_root
        self.results_dir = self.test_dir / "coverage_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Modules to analyze
        self.validation_modules = [
            'accuracy_scorer.py',
            'factual_accuracy_validator.py', 
            'claim_extractor.py',
            'document_indexer.py'
        ]
        
        # Test files mapping
        self.test_mappings = {
            'accuracy_scorer.py': [
                'test_accuracy_scorer_comprehensive.py'
            ],
            'factual_accuracy_validator.py': [
                'test_integrated_factual_validation.py',
                'test_validation_error_handling.py'
            ],
            'claim_extractor.py': [
                'test_validation_mocks.py',
                'test_integrated_factual_validation.py'
            ],
            'document_indexer.py': [
                'test_validation_mocks.py',
                'test_validation_error_handling.py'
            ]
        }
        
        # Configure logging
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_file = self.results_dir / f"coverage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Coverage analyzer initialized - logging to {log_file}")
    
    async def run_coverage_analysis(self, min_coverage: float = 90.0) -> TestCoverageReport:
        """
        Run comprehensive coverage analysis.
        
        Args:
            min_coverage: Minimum required coverage percentage
            
        Returns:
            Complete coverage report
        """
        
        self.logger.info("🔍 Starting comprehensive coverage analysis")
        start_time = time.time()
        
        try:
            # Run tests with coverage
            coverage_data = await self._run_tests_with_coverage()
            
            # Analyze coverage data
            overall_metrics = self._analyze_overall_coverage(coverage_data)
            module_metrics = self._analyze_module_coverage(coverage_data)
            function_metrics = self._analyze_function_coverage(coverage_data)
            
            # Assess test completeness
            test_completeness = self._assess_test_completeness()
            
            # Perform quality assessment
            quality_assessment = self._assess_coverage_quality(
                overall_metrics, module_metrics, function_metrics
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_metrics, module_metrics, test_completeness, min_coverage
            )
            
            # Create comprehensive report
            report = TestCoverageReport(
                timestamp=datetime.now().isoformat(),
                overall_metrics=overall_metrics,
                module_metrics=module_metrics,
                function_metrics=function_metrics,
                test_completeness=test_completeness,
                quality_assessment=quality_assessment,
                recommendations=recommendations,
                configuration={
                    'min_coverage_required': min_coverage,
                    'modules_analyzed': len(self.validation_modules),
                    'test_files_found': len(self._get_all_test_files()),
                    'analysis_duration_seconds': time.time() - start_time
                }
            )
            
            self.logger.info(f"✅ Coverage analysis completed in {time.time() - start_time:.2f}s")
            self.logger.info(f"📊 Overall coverage: {overall_metrics.coverage_percentage:.1f}%")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Coverage analysis failed: {e}")
            raise
    
    async def _run_tests_with_coverage(self) -> Dict[str, Any]:
        """
        Run tests with coverage collection.
        
        Returns:
            Coverage data dictionary
        """
        
        self.logger.info("🧪 Running tests with coverage collection")
        
        # Prepare coverage command
        coverage_file = self.results_dir / f"coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        html_dir = self.results_dir / "coverage_html"
        
        cmd = [
            'python', '-m', 'pytest',
            str(self.test_dir),
            '--cov=lightrag_integration',
            '--cov-report=xml:' + str(coverage_file),
            '--cov-report=html:' + str(html_dir),
            '--cov-report=term-missing',
            '--cov-branch',
            '-v'
        ]
        
        # Run coverage
        try:
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Tests completed with issues: {result.stderr}")
            
            # Parse coverage XML
            coverage_data = self._parse_coverage_xml(coverage_file)
            coverage_data['html_report_dir'] = html_dir
            coverage_data['xml_file'] = coverage_file
            
            return coverage_data
            
        except subprocess.TimeoutExpired:
            self.logger.error("Coverage analysis timed out")
            raise
        except Exception as e:
            self.logger.error(f"Coverage execution failed: {e}")
            raise
        finally:
            os.chdir(original_cwd)
    
    def _parse_coverage_xml(self, xml_file: Path) -> Dict[str, Any]:
        """
        Parse coverage XML file.
        
        Args:
            xml_file: Path to coverage XML file
            
        Returns:
            Parsed coverage data
        """
        
        if not xml_file.exists():
            raise FileNotFoundError(f"Coverage XML file not found: {xml_file}")
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        coverage_data = {
            'timestamp': root.get('timestamp'),
            'version': root.get('version'),
            'packages': {},
            'overall': {
                'lines_covered': 0,
                'lines_valid': 0,
                'line_rate': 0.0,
                'branches_covered': 0,
                'branches_valid': 0,
                'branch_rate': 0.0
            }
        }
        
        # Parse overall coverage
        coverage_elem = root.find('.')
        if coverage_elem is not None:
            coverage_data['overall'] = {
                'lines_covered': int(float(coverage_elem.get('lines-covered', 0))),
                'lines_valid': int(float(coverage_elem.get('lines-valid', 0))),
                'line_rate': float(coverage_elem.get('line-rate', 0.0)),
                'branches_covered': int(float(coverage_elem.get('branches-covered', 0))),
                'branches_valid': int(float(coverage_elem.get('branches-valid', 0))),
                'branch_rate': float(coverage_elem.get('branch-rate', 0.0))
            }
        
        # Parse packages and classes
        packages = root.find('packages')
        if packages is not None:
            for package in packages.findall('package'):
                package_name = package.get('name')
                package_data = {
                    'line_rate': float(package.get('line-rate', 0.0)),
                    'branch_rate': float(package.get('branch-rate', 0.0)),
                    'classes': {}
                }
                
                classes = package.find('classes')
                if classes is not None:
                    for cls in classes.findall('class'):
                        class_name = cls.get('name')
                        class_data = {
                            'filename': cls.get('filename'),
                            'line_rate': float(cls.get('line-rate', 0.0)),
                            'branch_rate': float(cls.get('branch-rate', 0.0)),
                            'lines': {},
                            'methods': {}
                        }
                        
                        # Parse lines
                        lines = cls.find('lines')
                        if lines is not None:
                            for line in lines.findall('line'):
                                line_num = int(line.get('number'))
                                hits = int(line.get('hits', 0))
                                branch = line.get('branch') == 'true'
                                
                                class_data['lines'][line_num] = {
                                    'hits': hits,
                                    'branch': branch
                                }
                        
                        # Parse methods
                        methods = cls.find('methods')
                        if methods is not None:
                            for method in methods.findall('method'):
                                method_name = method.get('name')
                                signature = method.get('signature', '')
                                
                                method_lines = method.find('lines')
                                method_line_data = {}
                                if method_lines is not None:
                                    for line in method_lines.findall('line'):
                                        line_num = int(line.get('number'))
                                        hits = int(line.get('hits', 0))
                                        method_line_data[line_num] = hits
                                
                                class_data['methods'][method_name] = {
                                    'signature': signature,
                                    'lines': method_line_data
                                }
                        
                        package_data['classes'][class_name] = class_data
                
                coverage_data['packages'][package_name] = package_data
        
        return coverage_data
    
    def _analyze_overall_coverage(self, coverage_data: Dict[str, Any]) -> CoverageMetrics:
        """
        Analyze overall coverage metrics.
        
        Args:
            coverage_data: Parsed coverage data
            
        Returns:
            Overall coverage metrics
        """
        
        overall = coverage_data['overall']
        
        return CoverageMetrics(
            name='Overall',
            lines_total=overall['lines_valid'],
            lines_covered=overall['lines_covered'],
            lines_missed=overall['lines_valid'] - overall['lines_covered'],
            coverage_percentage=overall['line_rate'] * 100,
            branch_total=overall['branches_valid'],
            branch_covered=overall['branches_covered'],
            branch_percentage=overall['branch_rate'] * 100
        )
    
    def _analyze_module_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, CoverageMetrics]:
        """
        Analyze module-level coverage metrics.
        
        Args:
            coverage_data: Parsed coverage data
            
        Returns:
            Module coverage metrics
        """
        
        module_metrics = {}
        
        for package_name, package_data in coverage_data['packages'].items():
            for class_name, class_data in package_data['classes'].items():
                filename = class_data['filename']
                
                # Extract module name
                module_name = Path(filename).stem
                
                if any(module in filename for module in self.validation_modules):
                    # Count lines
                    total_lines = len(class_data['lines'])
                    covered_lines = sum(1 for line_data in class_data['lines'].values() if line_data['hits'] > 0)
                    missed_lines = total_lines - covered_lines
                    
                    coverage_pct = (covered_lines / total_lines * 100) if total_lines > 0 else 0
                    
                    # Find missing line numbers
                    missing_line_nums = [
                        line_num for line_num, line_data in class_data['lines'].items()
                        if line_data['hits'] == 0
                    ]
                    
                    module_metrics[module_name] = CoverageMetrics(
                        name=module_name,
                        lines_total=total_lines,
                        lines_covered=covered_lines,
                        lines_missed=missed_lines,
                        coverage_percentage=coverage_pct,
                        branch_percentage=class_data['branch_rate'] * 100,
                        missing_lines=missing_line_nums
                    )
        
        return module_metrics
    
    def _analyze_function_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, Dict[str, CoverageMetrics]]:
        """
        Analyze function-level coverage metrics.
        
        Args:
            coverage_data: Parsed coverage data
            
        Returns:
            Function coverage metrics by module
        """
        
        function_metrics = defaultdict(dict)
        
        for package_name, package_data in coverage_data['packages'].items():
            for class_name, class_data in package_data['classes'].items():
                filename = class_data['filename']
                module_name = Path(filename).stem
                
                if any(module in filename for module in self.validation_modules):
                    for method_name, method_data in class_data['methods'].items():
                        method_lines = method_data['lines']
                        
                        if method_lines:
                            total_lines = len(method_lines)
                            covered_lines = sum(1 for hits in method_lines.values() if hits > 0)
                            coverage_pct = (covered_lines / total_lines * 100) if total_lines > 0 else 0
                            
                            function_metrics[module_name][method_name] = CoverageMetrics(
                                name=method_name,
                                lines_total=total_lines,
                                lines_covered=covered_lines,
                                lines_missed=total_lines - covered_lines,
                                coverage_percentage=coverage_pct
                            )
        
        return dict(function_metrics)
    
    def _assess_test_completeness(self) -> Dict[str, Any]:
        """
        Assess test completeness for validation modules.
        
        Returns:
            Test completeness assessment
        """
        
        self.logger.info("📋 Assessing test completeness")
        
        completeness = {
            'modules_analyzed': len(self.validation_modules),
            'test_files_found': 0,
            'coverage_by_module': {},
            'missing_tests': [],
            'test_quality_score': 0.0
        }
        
        test_files = self._get_all_test_files()
        completeness['test_files_found'] = len(test_files)
        
        # Analyze each module
        for module in self.validation_modules:
            module_name = Path(module).stem
            mapped_tests = self.test_mappings.get(module, [])
            
            existing_tests = [test for test in mapped_tests if (self.test_dir / test).exists()]
            
            module_assessment = {
                'module_file': module,
                'mapped_test_files': mapped_tests,
                'existing_test_files': existing_tests,
                'test_coverage_ratio': len(existing_tests) / len(mapped_tests) if mapped_tests else 0,
                'has_comprehensive_tests': len(existing_tests) >= 1,
                'test_types_covered': self._analyze_test_types_for_module(existing_tests)
            }
            
            completeness['coverage_by_module'][module_name] = module_assessment
            
            if not module_assessment['has_comprehensive_tests']:
                completeness['missing_tests'].append(module_name)
        
        # Calculate overall quality score
        module_scores = [
            assessment['test_coverage_ratio'] 
            for assessment in completeness['coverage_by_module'].values()
        ]
        
        completeness['test_quality_score'] = sum(module_scores) / len(module_scores) if module_scores else 0
        
        return completeness
    
    def _analyze_test_types_for_module(self, test_files: List[str]) -> Dict[str, bool]:
        """
        Analyze what types of tests exist for a module.
        
        Args:
            test_files: List of test file names
            
        Returns:
            Dictionary of test types found
        """
        
        test_types = {
            'unit_tests': False,
            'integration_tests': False,
            'performance_tests': False,
            'error_handling_tests': False,
            'mock_tests': False
        }
        
        for test_file in test_files:
            file_path = self.test_dir / test_file
            if not file_path.exists():
                continue
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                
                # Check for test types based on filename and content
                if 'comprehensive' in test_file or 'unit' in test_file:
                    test_types['unit_tests'] = True
                
                if 'integration' in test_file:
                    test_types['integration_tests'] = True
                
                if 'performance' in test_file:
                    test_types['performance_tests'] = True
                
                if 'error_handling' in test_file or 'error' in content:
                    test_types['error_handling_tests'] = True
                
                if 'mock' in test_file or 'mock' in content:
                    test_types['mock_tests'] = True
                    
            except Exception:
                continue
        
        return test_types
    
    def _assess_coverage_quality(self,
                               overall_metrics: CoverageMetrics,
                               module_metrics: Dict[str, CoverageMetrics],
                               function_metrics: Dict[str, Dict[str, CoverageMetrics]]) -> Dict[str, Any]:
        """
        Assess the quality of test coverage.
        
        Args:
            overall_metrics: Overall coverage metrics
            module_metrics: Module-level metrics
            function_metrics: Function-level metrics
            
        Returns:
            Coverage quality assessment
        """
        
        quality_assessment = {
            'overall_quality_grade': 'Unknown',
            'quality_indicators': {},
            'strengths': [],
            'weaknesses': [],
            'critical_gaps': [],
            'quality_score': 0.0
        }
        
        # Quality indicators
        quality_assessment['quality_indicators'] = {
            'high_coverage_modules': len([m for m in module_metrics.values() if m.coverage_percentage >= 90]),
            'medium_coverage_modules': len([m for m in module_metrics.values() if 70 <= m.coverage_percentage < 90]),
            'low_coverage_modules': len([m for m in module_metrics.values() if m.coverage_percentage < 70]),
            'uncovered_functions': sum(
                len([f for f in funcs.values() if f.coverage_percentage == 0])
                for funcs in function_metrics.values()
            ),
            'branch_coverage_available': overall_metrics.branch_total > 0,
            'average_module_coverage': sum(m.coverage_percentage for m in module_metrics.values()) / len(module_metrics) if module_metrics else 0
        }
        
        # Identify strengths
        if overall_metrics.coverage_percentage >= 90:
            quality_assessment['strengths'].append("Excellent overall line coverage")
        
        if overall_metrics.branch_percentage >= 80:
            quality_assessment['strengths'].append("Strong branch coverage")
        
        high_cov_modules = [name for name, m in module_metrics.items() if m.coverage_percentage >= 90]
        if len(high_cov_modules) > len(module_metrics) / 2:
            quality_assessment['strengths'].append("Majority of modules have high coverage")
        
        # Identify weaknesses
        if overall_metrics.coverage_percentage < 80:
            quality_assessment['weaknesses'].append("Overall line coverage below recommended 80%")
        
        if overall_metrics.branch_percentage < 70:
            quality_assessment['weaknesses'].append("Branch coverage below recommended 70%")
        
        low_cov_modules = [name for name, m in module_metrics.items() if m.coverage_percentage < 70]
        if low_cov_modules:
            quality_assessment['weaknesses'].append(f"Low coverage modules: {', '.join(low_cov_modules)}")
        
        # Identify critical gaps
        zero_cov_modules = [name for name, m in module_metrics.items() if m.coverage_percentage == 0]
        if zero_cov_modules:
            quality_assessment['critical_gaps'].append(f"Zero coverage modules: {', '.join(zero_cov_modules)}")
        
        # Calculate quality score (0-100)
        coverage_score = overall_metrics.coverage_percentage * 0.4
        branch_score = overall_metrics.branch_percentage * 0.2
        module_consistency_score = (1 - (max(m.coverage_percentage for m in module_metrics.values()) - 
                                        min(m.coverage_percentage for m in module_metrics.values())) / 100) * 100 * 0.2 if module_metrics else 0
        completeness_score = quality_assessment['quality_indicators']['high_coverage_modules'] / len(module_metrics) * 100 * 0.2 if module_metrics else 0
        
        quality_assessment['quality_score'] = coverage_score + branch_score + module_consistency_score + completeness_score
        
        # Determine quality grade
        if quality_assessment['quality_score'] >= 90:
            quality_assessment['overall_quality_grade'] = 'Excellent'
        elif quality_assessment['quality_score'] >= 80:
            quality_assessment['overall_quality_grade'] = 'Good'
        elif quality_assessment['quality_score'] >= 70:
            quality_assessment['overall_quality_grade'] = 'Acceptable'
        elif quality_assessment['quality_score'] >= 60:
            quality_assessment['overall_quality_grade'] = 'Marginal'
        else:
            quality_assessment['overall_quality_grade'] = 'Poor'
        
        return quality_assessment
    
    def _generate_recommendations(self,
                                overall_metrics: CoverageMetrics,
                                module_metrics: Dict[str, CoverageMetrics],
                                test_completeness: Dict[str, Any],
                                min_coverage: float) -> List[str]:
        """
        Generate coverage improvement recommendations.
        
        Args:
            overall_metrics: Overall coverage metrics
            module_metrics: Module-level metrics
            test_completeness: Test completeness assessment
            min_coverage: Minimum required coverage
            
        Returns:
            List of recommendations
        """
        
        recommendations = []
        
        # Overall coverage recommendations
        if overall_metrics.coverage_percentage < min_coverage:
            gap = min_coverage - overall_metrics.coverage_percentage
            recommendations.append(f"Overall coverage is {gap:.1f}% below target - prioritize adding tests for uncovered code")
        
        # Module-specific recommendations
        low_coverage_modules = [name for name, m in module_metrics.items() if m.coverage_percentage < min_coverage]
        for module_name in low_coverage_modules:
            module = module_metrics[module_name]
            recommendations.append(f"Module '{module_name}' has {module.coverage_percentage:.1f}% coverage - add {module.lines_missed} more covered lines")
        
        # Branch coverage recommendations
        if overall_metrics.branch_percentage < 80:
            recommendations.append("Branch coverage is low - add tests for conditional logic and error handling paths")
        
        # Test completeness recommendations
        if test_completeness['missing_tests']:
            recommendations.append(f"Add comprehensive tests for modules: {', '.join(test_completeness['missing_tests'])}")
        
        # Test type recommendations
        for module_name, module_assessment in test_completeness['coverage_by_module'].items():
            test_types = module_assessment['test_types_covered']
            
            missing_types = [test_type for test_type, covered in test_types.items() if not covered]
            if missing_types:
                recommendations.append(f"Add {', '.join(missing_types).replace('_', ' ')} for module '{module_name}'")
        
        # Quality improvement recommendations
        if not recommendations:
            recommendations.append("Coverage quality is good - maintain current testing standards")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _get_all_test_files(self) -> List[str]:
        """Get list of all test files."""
        test_files = []
        for file_path in self.test_dir.glob("test_*.py"):
            test_files.append(file_path.name)
        return test_files
    
    def generate_coverage_report(self, report: TestCoverageReport, format_type: str = 'json') -> Path:
        """
        Generate coverage report in specified format.
        
        Args:
            report: Coverage report data
            format_type: Report format ('json', 'html', 'text')
            
        Returns:
            Path to generated report file
        """
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == 'json':
            report_file = self.results_dir / f"coverage_report_{timestamp}.json"
            
            # Convert dataclasses to dicts for JSON serialization
            report_dict = asdict(report)
            
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
        
        elif format_type == 'html':
            report_file = self.results_dir / f"coverage_report_{timestamp}.html"
            
            html_content = self._generate_html_report(report)
            
            with open(report_file, 'w') as f:
                f.write(html_content)
        
        elif format_type == 'text':
            report_file = self.results_dir / f"coverage_report_{timestamp}.txt"
            
            text_content = self._generate_text_report(report)
            
            with open(report_file, 'w') as f:
                f.write(text_content)
        
        else:
            raise ValueError(f"Unsupported report format: {format_type}")
        
        self.logger.info(f"📄 Coverage report generated: {report_file}")
        return report_file
    
    def _generate_html_report(self, report: TestCoverageReport) -> str:
        """Generate HTML coverage report."""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Coverage Report - Factual Accuracy Validation System</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; border-radius: 5px; }}
                .excellent {{ background: #d4edda; border: 1px solid #c3e6cb; }}
                .good {{ background: #fff3cd; border: 1px solid #ffeaa7; }}
                .poor {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
                .module-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .module-table th, .module-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .module-table th {{ background-color: #f2f2f2; }}
                .recommendations {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Test Coverage Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Overall Coverage:</strong> {overall_coverage:.1f}%</p>
                <p><strong>Quality Grade:</strong> {quality_grade}</p>
            </div>
            
            <h2>Coverage Metrics</h2>
            <div class="metric {overall_class}">
                <h3>Overall Coverage</h3>
                <p>{overall_coverage:.1f}% ({lines_covered}/{lines_total} lines)</p>
                <p>Branch: {branch_coverage:.1f}%</p>
            </div>
            
            <h2>Module Coverage</h2>
            <table class="module-table">
                <tr>
                    <th>Module</th>
                    <th>Coverage %</th>
                    <th>Lines Covered</th>
                    <th>Lines Total</th>
                    <th>Missing Lines</th>
                </tr>
                {module_rows}
            </table>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
                    {recommendation_items}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Determine overall class
        overall_class = 'excellent' if report.overall_metrics.coverage_percentage >= 90 else 'good' if report.overall_metrics.coverage_percentage >= 70 else 'poor'
        
        # Generate module rows
        module_rows = ''
        for name, metrics in report.module_metrics.items():
            row_class = 'excellent' if metrics.coverage_percentage >= 90 else 'good' if metrics.coverage_percentage >= 70 else 'poor'
            missing_lines_str = ', '.join(map(str, metrics.missing_lines[:10]))  # Show first 10
            if len(metrics.missing_lines) > 10:
                missing_lines_str += f', ... ({len(metrics.missing_lines) - 10} more)'
            
            module_rows += f"""
                <tr class="{row_class}">
                    <td>{name}</td>
                    <td>{metrics.coverage_percentage:.1f}%</td>
                    <td>{metrics.lines_covered}</td>
                    <td>{metrics.lines_total}</td>
                    <td>{missing_lines_str}</td>
                </tr>
            """
        
        # Generate recommendation items
        recommendation_items = '\n'.join(f'<li>{rec}</li>' for rec in report.recommendations)
        
        return html_template.format(
            timestamp=report.timestamp,
            overall_coverage=report.overall_metrics.coverage_percentage,
            quality_grade=report.quality_assessment['overall_quality_grade'],
            overall_class=overall_class,
            lines_covered=report.overall_metrics.lines_covered,
            lines_total=report.overall_metrics.lines_total,
            branch_coverage=report.overall_metrics.branch_percentage,
            module_rows=module_rows,
            recommendation_items=recommendation_items
        )
    
    def _generate_text_report(self, report: TestCoverageReport) -> str:
        """Generate text coverage report."""
        
        lines = [
            "=" * 80,
            "TEST COVERAGE REPORT - FACTUAL ACCURACY VALIDATION SYSTEM",
            "=" * 80,
            f"Generated: {report.timestamp}",
            f"Overall Coverage: {report.overall_metrics.coverage_percentage:.1f}%",
            f"Quality Grade: {report.quality_assessment['overall_quality_grade']}",
            "",
            "OVERALL METRICS:",
            f"  Lines Covered: {report.overall_metrics.lines_covered}/{report.overall_metrics.lines_total}",
            f"  Branch Coverage: {report.overall_metrics.branch_percentage:.1f}%",
            "",
            "MODULE COVERAGE:",
        ]
        
        for name, metrics in report.module_metrics.items():
            lines.append(f"  {name}: {metrics.coverage_percentage:.1f}% ({metrics.lines_covered}/{metrics.lines_total} lines)")
        
        lines.extend([
            "",
            "RECOMMENDATIONS:",
        ])
        
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(lines)


async def main():
    """Main entry point for coverage analysis."""
    
    parser = argparse.ArgumentParser(
        description="Test Coverage Analysis for Factual Accuracy Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run comprehensive coverage analysis'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate coverage against requirements'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate coverage report only'
    )
    
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=90.0,
        help='Minimum required coverage percentage (default: 90.0)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'html', 'text'],
        default='json',
        help='Report format (default: json)'
    )
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.validate, args.report]):
        parser.print_help()
        return
    
    # Initialize analyzer
    analyzer = ValidationTestCoverageAnalyzer()
    
    try:
        if args.analyze or args.validate:
            # Run full analysis
            report = await analyzer.run_coverage_analysis(args.min_coverage)
            
            # Generate report
            report_file = analyzer.generate_coverage_report(report, args.format)
            
            print(f"📊 Coverage Analysis Results:")
            print(f"   Overall Coverage: {report.overall_metrics.coverage_percentage:.1f}%")
            print(f"   Quality Grade: {report.quality_assessment['overall_quality_grade']}")
            print(f"   Report saved to: {report_file}")
            
            # Validation check
            if args.validate:
                meets_requirements = report.overall_metrics.coverage_percentage >= args.min_coverage
                
                if meets_requirements:
                    print("✅ Coverage validation PASSED")
                    sys.exit(0)
                else:
                    print(f"❌ Coverage validation FAILED (required: {args.min_coverage}%, actual: {report.overall_metrics.coverage_percentage:.1f}%)")
                    sys.exit(1)
        
        elif args.report:
            print("Report-only mode not yet implemented - use --analyze to generate reports")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("🛑 Coverage analysis interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"💥 Coverage analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())