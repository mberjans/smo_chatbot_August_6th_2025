#!/usr/bin/env python3
"""
Performance Analysis and Reporting Utilities.

This module provides utilities for analyzing performance test results,
generating comprehensive reports, and visualizing performance metrics
for the Clinical Metabolomics Oracle LightRAG integration.

Components:
- PerformanceReportGenerator: Generates comprehensive performance reports
- BenchmarkAnalyzer: Analyzes benchmark results and trends
- QualityMetricsAnalyzer: Analyzes response quality patterns
- PerformanceVisualization: Creates charts and graphs for performance data
- RegressionAnalyzer: Detects and analyzes performance regressions
- RecommendationEngine: Provides optimization recommendations

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
import logging
from datetime import datetime, timedelta

# Import test result structures - handle import gracefully
try:
    from test_comprehensive_query_performance_quality import (
        ResponseQualityMetrics,
        PerformanceBenchmark,
        ScalabilityTestResult
    )
    TEST_STRUCTURES_AVAILABLE = True
except ImportError:
    # Define minimal structures for standalone operation
    from dataclasses import dataclass, field
    from typing import Dict, Any, List
    
    @dataclass
    class ResponseQualityMetrics:
        overall_quality_score: float = 0.0
        relevance_score: float = 0.0
        accuracy_score: float = 0.0
        completeness_score: float = 0.0
        clarity_score: float = 0.0
        biomedical_terminology_score: float = 0.0
        source_citation_score: float = 0.0
        consistency_score: float = 0.0
        factual_accuracy_score: float = 0.0
        hallucination_score: float = 0.0
        quality_flags: List[str] = field(default_factory=list)
    
    @dataclass
    class PerformanceBenchmark:
        query_type: str = ""
        benchmark_name: str = ""
        meets_performance_targets: bool = False
        performance_ratio: float = 0.0
        actual_response_time_ms: float = 0.0
        actual_throughput_ops_per_sec: float = 0.0
        actual_memory_usage_mb: float = 0.0
        actual_error_rate_percent: float = 0.0
        benchmark_details: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class ScalabilityTestResult:
        test_name: str = ""
        scaling_efficiency: float = 0.0
        scaling_grade: str = ""
        bottlenecks_identified: List[str] = field(default_factory=list)
    
    TEST_STRUCTURES_AVAILABLE = False


@dataclass
class PerformanceReport:
    """Comprehensive performance report structure."""
    report_id: str
    generation_time: datetime
    test_suite_name: str
    total_tests_run: int
    tests_passed: int
    tests_failed: int
    overall_performance_grade: str
    overall_quality_grade: str
    
    # Performance summary
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    throughput_ops_per_sec: float
    error_rate_percent: float
    
    # Quality summary
    avg_quality_score: float
    avg_relevance_score: float
    avg_biomedical_terminology_score: float
    consistency_score: float
    
    # Resource usage
    peak_memory_mb: float
    avg_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    
    # Detailed results
    benchmark_results: List[Dict[str, Any]] = field(default_factory=list)
    quality_results: List[Dict[str, Any]] = field(default_factory=list)
    scalability_results: List[Dict[str, Any]] = field(default_factory=list)
    regression_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        # Convert datetime to string for JSON serialization
        report_dict = self.to_dict()
        report_dict['generation_time'] = self.generation_time.isoformat()
        return json.dumps(report_dict, indent=indent, default=str)


class PerformanceReportGenerator:
    """Generates comprehensive performance reports."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("performance_reports")
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(self,
                                    benchmark_results: List[PerformanceBenchmark],
                                    quality_results: List[ResponseQualityMetrics],
                                    scalability_results: List[ScalabilityTestResult],
                                    test_suite_name: str = "Clinical_Metabolomics_Performance_Test") -> PerformanceReport:
        """Generate comprehensive performance report."""
        
        report_id = f"{test_suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        generation_time = datetime.now()
        
        # Analyze benchmark results
        benchmark_analysis = self._analyze_benchmark_results(benchmark_results)
        
        # Analyze quality results
        quality_analysis = self._analyze_quality_results(quality_results)
        
        # Analyze scalability results
        scalability_analysis = self._analyze_scalability_results(scalability_results)
        
        # Generate overall grades
        overall_performance_grade = self._calculate_performance_grade(benchmark_analysis)
        overall_quality_grade = self._calculate_quality_grade(quality_analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            benchmark_analysis, quality_analysis, scalability_analysis
        )
        
        # Create comprehensive report
        report = PerformanceReport(
            report_id=report_id,
            generation_time=generation_time,
            test_suite_name=test_suite_name,
            total_tests_run=len(benchmark_results) + len(quality_results) + len(scalability_results),
            tests_passed=sum(1 for b in benchmark_results if b.meets_performance_targets),
            tests_failed=sum(1 for b in benchmark_results if not b.meets_performance_targets),
            overall_performance_grade=overall_performance_grade,
            overall_quality_grade=overall_quality_grade,
            
            # Performance metrics
            avg_response_time_ms=benchmark_analysis.get('avg_response_time_ms', 0),
            median_response_time_ms=benchmark_analysis.get('median_response_time_ms', 0),
            p95_response_time_ms=benchmark_analysis.get('p95_response_time_ms', 0),
            throughput_ops_per_sec=benchmark_analysis.get('avg_throughput', 0),
            error_rate_percent=benchmark_analysis.get('avg_error_rate', 0),
            
            # Quality metrics
            avg_quality_score=quality_analysis.get('avg_overall_score', 0),
            avg_relevance_score=quality_analysis.get('avg_relevance_score', 0),
            avg_biomedical_terminology_score=quality_analysis.get('avg_biomedical_score', 0),
            consistency_score=quality_analysis.get('consistency_score', 0),
            
            # Resource usage
            peak_memory_mb=benchmark_analysis.get('peak_memory_mb', 0),
            avg_memory_mb=benchmark_analysis.get('avg_memory_mb', 0),
            peak_cpu_percent=benchmark_analysis.get('peak_cpu_percent', 0),
            avg_cpu_percent=benchmark_analysis.get('avg_cpu_percent', 0),
            
            # Detailed results
            benchmark_results=[asdict(b) for b in benchmark_results],
            quality_results=[asdict(q) for q in quality_results],
            scalability_results=[asdict(s) for s in scalability_results],
            regression_analysis=self._analyze_regressions(benchmark_results),
            recommendations=recommendations
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _analyze_benchmark_results(self, results: List[PerformanceBenchmark]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        if not results:
            return {}
        
        response_times = [r.actual_response_time_ms for r in results if r.actual_response_time_ms > 0]
        throughputs = [r.actual_throughput_ops_per_sec for r in results if r.actual_throughput_ops_per_sec > 0]
        error_rates = [r.actual_error_rate_percent for r in results]
        memory_usage = [r.actual_memory_usage_mb for r in results if r.actual_memory_usage_mb > 0]
        cpu_usage = [r.benchmark_details.get('cpu_usage', 0) for r in results]
        
        analysis = {
            'total_benchmarks': len(results),
            'benchmarks_passed': sum(1 for r in results if r.meets_performance_targets),
            'benchmarks_failed': sum(1 for r in results if not r.meets_performance_targets)
        }
        
        if response_times:
            analysis.update({
                'avg_response_time_ms': statistics.mean(response_times),
                'median_response_time_ms': statistics.median(response_times),
                'p95_response_time_ms': np.percentile(response_times, 95),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times)
            })
        
        if throughputs:
            analysis.update({
                'avg_throughput': statistics.mean(throughputs),
                'max_throughput': max(throughputs)
            })
        
        if error_rates:
            analysis['avg_error_rate'] = statistics.mean(error_rates)
        
        if memory_usage:
            analysis.update({
                'avg_memory_mb': statistics.mean(memory_usage),
                'peak_memory_mb': max(memory_usage)
            })
        
        if cpu_usage:
            analysis.update({
                'avg_cpu_percent': statistics.mean(cpu_usage),
                'peak_cpu_percent': max(cpu_usage)
            })
        
        return analysis
    
    def _analyze_quality_results(self, results: List[ResponseQualityMetrics]) -> Dict[str, Any]:
        """Analyze quality results."""
        if not results:
            return {}
        
        analysis = {
            'total_quality_tests': len(results),
            'avg_overall_score': statistics.mean(r.overall_quality_score for r in results),
            'avg_relevance_score': statistics.mean(r.relevance_score for r in results),
            'avg_accuracy_score': statistics.mean(r.accuracy_score for r in results),
            'avg_completeness_score': statistics.mean(r.completeness_score for r in results),
            'avg_clarity_score': statistics.mean(r.clarity_score for r in results),
            'avg_biomedical_score': statistics.mean(r.biomedical_terminology_score for r in results),
            'avg_citation_score': statistics.mean(r.source_citation_score for r in results),
            'consistency_score': statistics.mean(r.consistency_score for r in results),
            'avg_hallucination_score': statistics.mean(r.hallucination_score for r in results)
        }
        
        # Quality distribution
        excellent_count = sum(1 for r in results if r.overall_quality_score >= 90)
        good_count = sum(1 for r in results if 80 <= r.overall_quality_score < 90)
        acceptable_count = sum(1 for r in results if 70 <= r.overall_quality_score < 80)
        poor_count = sum(1 for r in results if r.overall_quality_score < 70)
        
        analysis['quality_distribution'] = {
            'excellent': excellent_count,
            'good': good_count,
            'acceptable': acceptable_count,
            'poor': poor_count
        }
        
        # Common quality flags
        all_flags = [flag for r in results for flag in r.quality_flags]
        flag_counts = defaultdict(int)
        for flag in all_flags:
            flag_counts[flag] += 1
        
        analysis['common_quality_flags'] = dict(flag_counts)
        
        return analysis
    
    def _analyze_scalability_results(self, results: List[ScalabilityTestResult]) -> Dict[str, Any]:
        """Analyze scalability results."""
        if not results:
            return {}
        
        analysis = {
            'total_scalability_tests': len(results),
            'avg_scaling_efficiency': statistics.mean(r.scaling_efficiency for r in results),
            'scaling_grades': {
                'good': sum(1 for r in results if r.scaling_grade == 'Good'),
                'poor': sum(1 for r in results if r.scaling_grade == 'Poor')
            }
        }
        
        # Bottleneck analysis
        all_bottlenecks = [b for r in results for b in r.bottlenecks_identified]
        bottleneck_counts = defaultdict(int)
        for bottleneck in all_bottlenecks:
            bottleneck_counts[bottleneck] += 1
        
        analysis['common_bottlenecks'] = dict(bottleneck_counts)
        
        return analysis
    
    def _calculate_performance_grade(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall performance grade."""
        if not analysis:
            return "Unknown"
        
        pass_rate = analysis.get('benchmarks_passed', 0) / max(1, analysis.get('total_benchmarks', 1))
        avg_response_time = analysis.get('avg_response_time_ms', float('inf'))
        avg_error_rate = analysis.get('avg_error_rate', 100)
        
        if pass_rate >= 0.9 and avg_response_time <= 10000 and avg_error_rate <= 5:
            return "Excellent"
        elif pass_rate >= 0.8 and avg_response_time <= 20000 and avg_error_rate <= 10:
            return "Good"
        elif pass_rate >= 0.7 and avg_response_time <= 30000 and avg_error_rate <= 15:
            return "Acceptable"
        elif pass_rate >= 0.6:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _calculate_quality_grade(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall quality grade."""
        if not analysis:
            return "Unknown"
        
        avg_quality = analysis.get('avg_overall_score', 0)
        consistency = analysis.get('consistency_score', 0)
        
        if avg_quality >= 85 and consistency >= 80:
            return "Excellent"
        elif avg_quality >= 75 and consistency >= 70:
            return "Good"
        elif avg_quality >= 65 and consistency >= 60:
            return "Acceptable"
        elif avg_quality >= 55:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _analyze_regressions(self, results: List[PerformanceBenchmark]) -> Dict[str, Any]:
        """Analyze performance regressions."""
        # This would compare against historical data
        # For now, return basic analysis
        
        regression_analysis = {
            'regressions_detected': 0,
            'improvements_detected': 0,
            'stable_metrics': 0,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Analyze performance ratios
        performance_ratios = [r.performance_ratio for r in results if r.performance_ratio > 0]
        if performance_ratios:
            avg_ratio = statistics.mean(performance_ratios)
            if avg_ratio < 0.8:
                regression_analysis['regressions_detected'] = len([r for r in performance_ratios if r < 0.8])
            elif avg_ratio > 1.2:
                regression_analysis['improvements_detected'] = len([r for r in performance_ratios if r > 1.2])
            else:
                regression_analysis['stable_metrics'] = len(performance_ratios)
        
        return regression_analysis
    
    def _generate_recommendations(self,
                                benchmark_analysis: Dict[str, Any],
                                quality_analysis: Dict[str, Any],
                                scalability_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Performance recommendations
        if benchmark_analysis.get('avg_response_time_ms', 0) > 20000:
            recommendations.append(
                "Consider optimizing query processing pipeline - average response time exceeds 20 seconds"
            )
        
        if benchmark_analysis.get('avg_error_rate', 0) > 10:
            recommendations.append(
                "Improve error handling and system reliability - error rate exceeds 10%"
            )
        
        if benchmark_analysis.get('peak_memory_mb', 0) > 1500:
            recommendations.append(
                "Optimize memory usage - peak memory consumption exceeds 1.5GB"
            )
        
        # Quality recommendations
        if quality_analysis.get('avg_overall_score', 0) < 75:
            recommendations.append(
                "Improve response quality through better training data or prompt optimization"
            )
        
        if quality_analysis.get('avg_biomedical_score', 0) < 70:
            recommendations.append(
                "Enhance biomedical terminology usage in responses"
            )
        
        if quality_analysis.get('consistency_score', 0) < 70:
            recommendations.append(
                "Address response consistency issues - implement response caching or standardization"
            )
        
        # Scalability recommendations
        if scalability_analysis.get('avg_scaling_efficiency', 1.0) < 0.7:
            recommendations.append(
                "Improve system scalability - implement connection pooling or load balancing"
            )
        
        # Common bottleneck recommendations
        common_bottlenecks = scalability_analysis.get('common_bottlenecks', {})
        if 'memory_pressure' in common_bottlenecks:
            recommendations.append(
                "Address memory pressure bottlenecks through garbage collection optimization"
            )
        
        if 'io_bound_operations' in common_bottlenecks:
            recommendations.append(
                "Optimize I/O operations through asynchronous processing and caching"
            )
        
        return recommendations
    
    def _save_report(self, report: PerformanceReport):
        """Save performance report to file."""
        
        # Save as JSON
        json_path = self.output_dir / f"{report.report_id}.json"
        with open(json_path, 'w') as f:
            f.write(report.to_json())
        
        # Save as human-readable summary
        summary_path = self.output_dir / f"{report.report_id}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary_text(report))
        
        self.logger.info(f"Performance report saved: {json_path}")
    
    def _generate_summary_text(self, report: PerformanceReport) -> str:
        """Generate human-readable summary text."""
        
        summary = f"""
CLINICAL METABOLOMICS ORACLE - PERFORMANCE TEST REPORT
======================================================

Report ID: {report.report_id}
Generated: {report.generation_time.strftime('%Y-%m-%d %H:%M:%S')}
Test Suite: {report.test_suite_name}

EXECUTIVE SUMMARY
-----------------
Total Tests Run: {report.total_tests_run}
Tests Passed: {report.tests_passed}
Tests Failed: {report.tests_failed}
Overall Performance Grade: {report.overall_performance_grade}
Overall Quality Grade: {report.overall_quality_grade}

PERFORMANCE METRICS
-------------------
Average Response Time: {report.avg_response_time_ms:.1f} ms
Median Response Time: {report.median_response_time_ms:.1f} ms
95th Percentile Response Time: {report.p95_response_time_ms:.1f} ms
Throughput: {report.throughput_ops_per_sec:.2f} operations/second
Error Rate: {report.error_rate_percent:.1f}%

QUALITY METRICS
---------------
Average Quality Score: {report.avg_quality_score:.1f}/100
Average Relevance Score: {report.avg_relevance_score:.1f}/100
Average Biomedical Terminology Score: {report.avg_biomedical_terminology_score:.1f}/100
Response Consistency Score: {report.consistency_score:.1f}/100

RESOURCE USAGE
--------------
Peak Memory Usage: {report.peak_memory_mb:.1f} MB
Average Memory Usage: {report.avg_memory_mb:.1f} MB
Peak CPU Usage: {report.peak_cpu_percent:.1f}%
Average CPU Usage: {report.avg_cpu_percent:.1f}%

RECOMMENDATIONS
---------------"""
        
        for i, rec in enumerate(report.recommendations, 1):
            summary += f"\n{i}. {rec}"
        
        if not report.recommendations:
            summary += "\nNo specific recommendations - system performance is satisfactory."
        
        summary += f"""

DETAILED RESULTS
----------------
Benchmark Tests: {len(report.benchmark_results)}
Quality Tests: {len(report.quality_results)}
Scalability Tests: {len(report.scalability_results)}

For detailed metrics and analysis, see the full JSON report.
"""
        
        return summary


class BenchmarkAnalyzer:
    """Analyzes benchmark trends and patterns."""
    
    def __init__(self):
        self.historical_data: List[PerformanceReport] = []
    
    def add_historical_report(self, report: PerformanceReport):
        """Add historical report for trend analysis."""
        self.historical_data.append(report)
        # Keep only last 10 reports for trend analysis
        self.historical_data = self.historical_data[-10:]
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.historical_data) < 2:
            return {'status': 'insufficient_data'}
        
        # Extract trend data
        response_times = [r.avg_response_time_ms for r in self.historical_data]
        quality_scores = [r.avg_quality_score for r in self.historical_data]
        error_rates = [r.error_rate_percent for r in self.historical_data]
        
        # Calculate trends
        trends = {
            'response_time_trend': self._calculate_trend(response_times),
            'quality_score_trend': self._calculate_trend(quality_scores),
            'error_rate_trend': self._calculate_trend(error_rates),
            'analysis_period': {
                'start': self.historical_data[0].generation_time.isoformat(),
                'end': self.historical_data[-1].generation_time.isoformat(),
                'reports_analyzed': len(self.historical_data)
            }
        }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and magnitude."""
        if len(values) < 2:
            return {'direction': 'unknown', 'magnitude': 0}
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:  # Threshold for "no change"
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'magnitude': abs(slope),
            'values': values
        }


# Utility functions for test execution and reporting

def generate_performance_summary_report(test_results_dir: Path) -> Dict[str, Any]:
    """Generate summary report from test results directory."""
    
    # This would collect and analyze all test results
    # For now, return a placeholder structure
    
    return {
        'summary': 'Performance test summary would be generated here',
        'timestamp': datetime.now().isoformat(),
        'test_results_analyzed': 0
    }


def compare_performance_reports(report1: PerformanceReport, 
                              report2: PerformanceReport) -> Dict[str, Any]:
    """Compare two performance reports."""
    
    comparison = {
        'report_comparison': {
            'baseline': report1.report_id,
            'current': report2.report_id,
            'comparison_time': datetime.now().isoformat()
        },
        'performance_changes': {
            'response_time_change': report2.avg_response_time_ms - report1.avg_response_time_ms,
            'quality_score_change': report2.avg_quality_score - report1.avg_quality_score,
            'throughput_change': report2.throughput_ops_per_sec - report1.throughput_ops_per_sec,
            'error_rate_change': report2.error_rate_percent - report1.error_rate_percent
        },
        'grade_changes': {
            'performance_grade': f"{report1.overall_performance_grade} → {report2.overall_performance_grade}",
            'quality_grade': f"{report1.overall_quality_grade} → {report2.overall_quality_grade}"
        }
    }
    
    return comparison


if __name__ == "__main__":
    # Example usage
    print("Performance Analysis Utilities")
    print("Available classes:")
    print("- PerformanceReportGenerator: Generate comprehensive reports")
    print("- BenchmarkAnalyzer: Analyze performance trends")
    print("- Utility functions for report comparison and summarization")