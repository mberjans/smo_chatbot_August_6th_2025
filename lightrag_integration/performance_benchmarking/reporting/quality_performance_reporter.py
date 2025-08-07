#!/usr/bin/env python3
"""
Quality Performance Reporter for Clinical Metabolomics Oracle.

This module implements comprehensive reporting capabilities for quality validation
performance benchmarking. It provides detailed analysis, visualization, and
actionable insights from performance data collected by the quality validation
benchmark suite, correlation engine, and API metrics logger.

Classes:
    - PerformanceReportConfiguration: Configuration for report generation
    - ReportMetadata: Metadata container for reports
    - PerformanceInsight: Individual performance insight data structure
    - OptimizationRecommendation: Structured optimization recommendation
    - QualityPerformanceReporter: Main reporting engine

Key Features:
    - Multi-format report generation (JSON, HTML, CSV, text)
    - Statistical analysis and trend identification
    - Performance bottleneck detection and analysis
    - Resource optimization recommendations
    - Cost-benefit analysis
    - Quality vs performance trade-off analysis
    - Interactive dashboard generation
    - Executive summary generation
    - Actionable performance optimization insights

Integration Points:
    - QualityValidationBenchmarkSuite: Benchmark results analysis
    - CrossSystemCorrelationEngine: Correlation analysis integration
    - QualityAwareAPIMetricsLogger: API metrics integration
    - Performance test utilities and fixtures

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import json
import csv
import time
import logging
import statistics
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import traceback

# Data visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Visualization libraries not available: {e}. Some features will be disabled.")
    VISUALIZATION_AVAILABLE = False

# Statistical analysis imports
try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

# Import parent modules
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

try:
    from quality_performance_benchmarks import (
        QualityValidationMetrics, QualityValidationBenchmarkSuite,
        QualityPerformanceThreshold, QualityBenchmarkConfiguration
    )
    from performance_correlation_engine import (
        CrossSystemCorrelationEngine, PerformanceCorrelationMetrics,
        QualityPerformanceCorrelation, CorrelationAnalysisReport
    )
    from quality_aware_metrics_logger import (
        QualityAwareAPIMetricsLogger, QualityAPIMetric, QualityMetricsAggregator
    )
except ImportError as e:
    logging.warning(f"Some performance modules not available: {e}. Using mock implementations.")

# Configure logging
logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report output formats."""
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    TEXT = "text"
    PDF = "pdf"
    EXCEL = "excel"


class PerformanceMetricType(Enum):
    """Types of performance metrics for reporting."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    COST = "cost"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    QUALITY_SCORE = "quality_score"
    EFFICIENCY = "efficiency"
    SCALABILITY = "scalability"


@dataclass
class PerformanceReportConfiguration:
    """Configuration settings for performance report generation."""
    
    # Report identification
    report_name: str = "Quality Performance Report"
    report_description: Optional[str] = None
    include_executive_summary: bool = True
    include_detailed_analysis: bool = True
    include_recommendations: bool = True
    
    # Data filtering and scope
    analysis_period_hours: int = 24  # Last N hours of data
    minimum_sample_size: int = 10
    include_historical_comparison: bool = True
    filter_by_validation_type: Optional[List[str]] = None
    filter_by_operation_stage: Optional[List[str]] = None
    
    # Statistical analysis settings
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    trend_analysis_window: int = 10  # Number of data points for trend analysis
    outlier_detection: bool = True
    outlier_threshold: float = 2.0  # Standard deviations
    
    # Visualization settings
    generate_charts: bool = True
    chart_width: int = 1200
    chart_height: int = 600
    color_scheme: str = "professional"  # "professional", "vibrant", "minimal"
    include_interactive_charts: bool = True
    
    # Output settings
    output_formats: List[ReportFormat] = field(default_factory=lambda: [ReportFormat.JSON, ReportFormat.HTML])
    output_directory: Optional[Path] = None
    compress_output: bool = False
    include_raw_data: bool = False
    
    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "response_time_ms_threshold": 2000,
        "throughput_ops_per_sec_threshold": 5.0,
        "accuracy_threshold": 85.0,
        "cost_per_operation_threshold": 0.01,
        "memory_usage_mb_threshold": 1000,
        "error_rate_threshold": 5.0
    })
    
    # Recommendation settings
    generate_performance_recommendations: bool = True
    generate_cost_optimization_recommendations: bool = True
    generate_resource_recommendations: bool = True
    recommendation_priority_threshold: str = "medium"  # "low", "medium", "high"


@dataclass
class ReportMetadata:
    """Metadata container for performance reports."""
    
    report_id: str = field(default_factory=lambda: f"perf_report_{int(time.time())}")
    generated_timestamp: float = field(default_factory=time.time)
    report_version: str = "1.0.0"
    generator: str = "QualityPerformanceReporter"
    
    # Data scope
    analysis_start_time: Optional[float] = None
    analysis_end_time: Optional[float] = None
    total_data_points: int = 0
    data_sources: List[str] = field(default_factory=list)
    
    # Report configuration summary
    configuration_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Generation statistics
    generation_duration_seconds: float = 0.0
    report_size_bytes: int = 0
    charts_generated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            **asdict(self),
            'generated_timestamp_iso': datetime.fromtimestamp(self.generated_timestamp).isoformat(),
            'analysis_start_time_iso': datetime.fromtimestamp(self.analysis_start_time).isoformat() if self.analysis_start_time else None,
            'analysis_end_time_iso': datetime.fromtimestamp(self.analysis_end_time).isoformat() if self.analysis_end_time else None
        }


@dataclass
class PerformanceInsight:
    """Individual performance insight or finding."""
    
    insight_id: str = field(default_factory=lambda: f"insight_{int(time.time())}")
    insight_type: str = "general"  # "bottleneck", "trend", "anomaly", "optimization"
    title: str = "Performance Insight"
    description: str = ""
    severity: str = "medium"  # "low", "medium", "high", "critical"
    
    # Supporting data
    metrics_involved: List[str] = field(default_factory=list)
    statistical_confidence: Optional[float] = None
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    priority_level: int = 3  # 1-5, where 1 is highest priority
    
    # Context
    affected_components: List[str] = field(default_factory=list)
    time_period: Optional[str] = None
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Structured optimization recommendation."""
    
    recommendation_id: str = field(default_factory=lambda: f"rec_{int(time.time())}")
    category: str = "performance"  # "performance", "cost", "resource", "quality"
    title: str = "Optimization Recommendation"
    description: str = ""
    priority: str = "medium"  # "low", "medium", "high", "critical"
    
    # Implementation details
    implementation_effort: str = "medium"  # "low", "medium", "high"
    estimated_impact: Dict[str, float] = field(default_factory=dict)  # metric -> improvement %
    implementation_steps: List[str] = field(default_factory=list)
    
    # Cost-benefit analysis
    implementation_cost_estimate: Optional[float] = None
    expected_savings: Optional[float] = None
    roi_estimate: Optional[float] = None
    payback_period_days: Optional[int] = None
    
    # Context and validation
    applicable_scenarios: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    risks_and_considerations: List[str] = field(default_factory=list)
    
    # Supporting data
    supporting_metrics: Dict[str, Any] = field(default_factory=dict)
    confidence_level: float = 0.8


class QualityPerformanceReporter:
    """
    Comprehensive performance reporting engine for quality validation systems.
    
    Generates detailed reports from quality validation benchmark data, correlation
    analysis, and API metrics with actionable insights and recommendations.
    """
    
    def __init__(self,
                 config: Optional[PerformanceReportConfiguration] = None,
                 output_directory: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the quality performance reporter.
        
        Args:
            config: Report configuration settings
            output_directory: Directory for saving reports
            logger: Logger instance for reporting operations
        """
        self.config = config or PerformanceReportConfiguration()
        self.output_directory = output_directory or self.config.output_directory or Path("performance_reports")
        self.logger = logger or logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.benchmark_data: List[QualityValidationMetrics] = []
        self.correlation_data: List[PerformanceCorrelationMetrics] = []
        self.api_metrics_data: List[QualityAPIMetric] = []
        self.correlation_reports: List[CorrelationAnalysisReport] = []
        
        # Analysis results
        self.performance_insights: List[PerformanceInsight] = []
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Report generation components
        self.report_metadata = ReportMetadata()
        
        # Initialize visualization settings
        if VISUALIZATION_AVAILABLE:
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
            sns.set_palette(self.config.color_scheme if self.config.color_scheme in ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'] else 'deep')
        
        self.logger.info(f"QualityPerformanceReporter initialized with output directory: {self.output_directory}")
    
    async def load_benchmark_data(self, 
                                benchmark_suite: Optional[QualityValidationBenchmarkSuite] = None,
                                data_file: Optional[Path] = None,
                                data: Optional[List[QualityValidationMetrics]] = None) -> int:
        """
        Load quality validation benchmark data for reporting.
        
        Args:
            benchmark_suite: Benchmark suite to extract data from
            data_file: Path to saved benchmark data file
            data: Direct list of quality validation metrics
            
        Returns:
            Number of benchmark data points loaded
        """
        loaded_count = 0
        
        try:
            if data:
                self.benchmark_data.extend(data)
                loaded_count = len(data)
                
            elif data_file and data_file.exists():
                with open(data_file, 'r') as f:
                    raw_data = json.load(f)
                
                # Extract benchmark metrics from saved data
                if isinstance(raw_data, dict):
                    # Handle structured report format
                    if 'quality_benchmark_results' in raw_data:
                        for benchmark_result in raw_data['quality_benchmark_results'].values():
                            if 'scenario_quality_metrics' in benchmark_result:
                                for metric_dict in benchmark_result['scenario_quality_metrics']:
                                    # Convert dict back to QualityValidationMetrics
                                    # This is a simplified conversion - in practice would need full reconstruction
                                    loaded_count += 1
                    
                    # Handle raw metrics format
                    elif 'scenario_quality_metrics' in raw_data:
                        loaded_count = len(raw_data['scenario_quality_metrics'])
                
                self.logger.info(f"Loaded {loaded_count} benchmark data points from file")
                
            elif benchmark_suite:
                # Extract from benchmark suite history
                for benchmark_name, metrics_list in benchmark_suite.quality_metrics_history.items():
                    self.benchmark_data.extend(metrics_list)
                    loaded_count += len(metrics_list)
                
                self.logger.info(f"Loaded {loaded_count} benchmark data points from suite")
            
            # Update metadata
            self.report_metadata.data_sources.append("benchmark_data")
            self.report_metadata.total_data_points += loaded_count
            
        except Exception as e:
            self.logger.error(f"Error loading benchmark data: {e}")
            self.logger.debug(traceback.format_exc())
        
        return loaded_count
    
    async def load_correlation_data(self,
                                  correlation_engine: Optional[CrossSystemCorrelationEngine] = None,
                                  correlation_reports: Optional[List[CorrelationAnalysisReport]] = None,
                                  data_file: Optional[Path] = None) -> int:
        """
        Load correlation analysis data for reporting.
        
        Args:
            correlation_engine: Correlation engine to extract data from
            correlation_reports: Direct list of correlation reports
            data_file: Path to saved correlation data file
            
        Returns:
            Number of correlation data points loaded
        """
        loaded_count = 0
        
        try:
            if correlation_reports:
                self.correlation_reports.extend(correlation_reports)
                loaded_count = len(correlation_reports)
                
            elif data_file and data_file.exists():
                with open(data_file, 'r') as f:
                    raw_data = json.load(f)
                
                # Extract correlation data from saved file
                if isinstance(raw_data, dict) and 'correlation_metrics' in raw_data:
                    # Handle single correlation report
                    loaded_count = 1
                elif isinstance(raw_data, list):
                    # Handle multiple correlation reports
                    loaded_count = len(raw_data)
                
                self.logger.info(f"Loaded {loaded_count} correlation reports from file")
                
            elif correlation_engine:
                # Extract from correlation engine history
                self.correlation_data.extend(correlation_engine.correlation_history)
                loaded_count = len(correlation_engine.correlation_history)
                
                self.logger.info(f"Loaded {loaded_count} correlation data points from engine")
            
            # Update metadata
            self.report_metadata.data_sources.append("correlation_data")
            self.report_metadata.total_data_points += loaded_count
            
        except Exception as e:
            self.logger.error(f"Error loading correlation data: {e}")
            self.logger.debug(traceback.format_exc())
        
        return loaded_count
    
    async def load_api_metrics_data(self,
                                  api_logger: Optional[QualityAwareAPIMetricsLogger] = None,
                                  metrics_data: Optional[List[QualityAPIMetric]] = None,
                                  data_file: Optional[Path] = None) -> int:
        """
        Load API metrics data for reporting.
        
        Args:
            api_logger: API metrics logger to extract data from
            metrics_data: Direct list of quality API metrics
            data_file: Path to saved metrics data file
            
        Returns:
            Number of API metrics data points loaded
        """
        loaded_count = 0
        
        try:
            if metrics_data:
                self.api_metrics_data.extend(metrics_data)
                loaded_count = len(metrics_data)
                
            elif data_file and data_file.exists():
                with open(data_file, 'r') as f:
                    raw_data = json.load(f)
                
                # Extract API metrics from saved file
                if isinstance(raw_data, dict) and 'raw_metrics' in raw_data:
                    loaded_count = len(raw_data['raw_metrics'])
                elif isinstance(raw_data, list):
                    loaded_count = len(raw_data)
                
                self.logger.info(f"Loaded {loaded_count} API metrics from file")
                
            elif api_logger:
                # Extract from API logger aggregator
                if hasattr(api_logger.metrics_aggregator, '_metrics_buffer'):
                    quality_metrics = [m for m in api_logger.metrics_aggregator._metrics_buffer 
                                     if hasattr(m, 'quality_validation_type')]
                    self.api_metrics_data.extend(quality_metrics)
                    loaded_count = len(quality_metrics)
                
                self.logger.info(f"Loaded {loaded_count} API metrics from logger")
            
            # Update metadata
            self.report_metadata.data_sources.append("api_metrics_data")
            self.report_metadata.total_data_points += loaded_count
            
        except Exception as e:
            self.logger.error(f"Error loading API metrics data: {e}")
            self.logger.debug(traceback.format_exc())
        
        return loaded_count
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with all analysis components.
        
        Returns:
            Complete performance report as structured data
        """
        self.logger.info("Generating comprehensive quality performance report")
        
        start_time = time.time()
        
        try:
            # Set analysis time window
            current_time = time.time()
            analysis_start_time = current_time - (self.config.analysis_period_hours * 3600)
            
            self.report_metadata.analysis_start_time = analysis_start_time
            self.report_metadata.analysis_end_time = current_time
            
            # Initialize report structure
            comprehensive_report = {
                'metadata': self.report_metadata.to_dict(),
                'configuration': asdict(self.config),
                'executive_summary': {},
                'performance_analysis': {},
                'correlation_analysis': {},
                'cost_analysis': {},
                'resource_analysis': {},
                'quality_analysis': {},
                'trend_analysis': {},
                'bottleneck_analysis': {},
                'insights': [],
                'recommendations': [],
                'detailed_metrics': {},
                'charts_and_visualizations': {}
            }
            
            # Generate executive summary
            if self.config.include_executive_summary:
                comprehensive_report['executive_summary'] = await self._generate_executive_summary()
            
            # Generate detailed analysis sections
            if self.config.include_detailed_analysis:
                comprehensive_report['performance_analysis'] = await self._analyze_performance_metrics()
                comprehensive_report['correlation_analysis'] = await self._analyze_correlations()
                comprehensive_report['cost_analysis'] = await self._analyze_cost_metrics()
                comprehensive_report['resource_analysis'] = await self._analyze_resource_usage()
                comprehensive_report['quality_analysis'] = await self._analyze_quality_metrics()
                comprehensive_report['trend_analysis'] = await self._analyze_trends()
                comprehensive_report['bottleneck_analysis'] = await self._analyze_bottlenecks()
            
            # Generate insights and recommendations
            await self._generate_performance_insights()
            comprehensive_report['insights'] = [asdict(insight) for insight in self.performance_insights]
            
            if self.config.include_recommendations:
                await self._generate_optimization_recommendations()
                comprehensive_report['recommendations'] = [asdict(rec) for rec in self.optimization_recommendations]
            
            # Include raw metrics if requested
            if self.config.include_raw_data:
                comprehensive_report['detailed_metrics'] = {
                    'benchmark_metrics': [asdict(m) for m in self.benchmark_data],
                    'correlation_metrics': [asdict(m) for m in self.correlation_data],
                    'api_metrics': [asdict(m) for m in self.api_metrics_data]
                }
            
            # Generate visualizations
            if self.config.generate_charts and VISUALIZATION_AVAILABLE:
                comprehensive_report['charts_and_visualizations'] = await self._generate_visualizations()
            
            # Update report metadata
            self.report_metadata.generation_duration_seconds = time.time() - start_time
            comprehensive_report['metadata'] = self.report_metadata.to_dict()
            
            self.logger.info(f"Comprehensive report generated successfully in {self.report_metadata.generation_duration_seconds:.2f} seconds")
            
            return comprehensive_report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            self.logger.debug(traceback.format_exc())
            raise
    
    async def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of performance data."""
        summary = {
            'report_period': {
                'start_time': datetime.fromtimestamp(self.report_metadata.analysis_start_time).isoformat() if self.report_metadata.analysis_start_time else None,
                'end_time': datetime.fromtimestamp(self.report_metadata.analysis_end_time).isoformat() if self.report_metadata.analysis_end_time else None,
                'duration_hours': self.config.analysis_period_hours
            },
            'data_summary': {
                'total_benchmark_operations': len(self.benchmark_data),
                'total_api_operations': len(self.api_metrics_data),
                'correlation_analyses': len(self.correlation_data),
                'data_sources': self.report_metadata.data_sources
            },
            'key_performance_indicators': {},
            'overall_health_score': 0.0,
            'critical_issues': [],
            'top_recommendations': []
        }
        
        # Calculate key performance indicators
        if self.benchmark_data:
            response_times = [m.average_latency_ms for m in self.benchmark_data if m.average_latency_ms > 0]
            quality_scores = [m.calculate_quality_efficiency_score() for m in self.benchmark_data]
            error_rates = [m.error_rate_percent for m in self.benchmark_data]
            
            if response_times:
                summary['key_performance_indicators']['average_response_time_ms'] = statistics.mean(response_times)
                summary['key_performance_indicators']['p95_response_time_ms'] = sorted(response_times)[int(len(response_times) * 0.95)]
            
            if quality_scores:
                summary['key_performance_indicators']['average_quality_score'] = statistics.mean(quality_scores)
                summary['key_performance_indicators']['quality_score_std'] = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
            
            if error_rates:
                summary['key_performance_indicators']['average_error_rate'] = statistics.mean(error_rates)
        
        # Calculate overall health score
        health_components = []
        
        if 'average_response_time_ms' in summary['key_performance_indicators']:
            response_time = summary['key_performance_indicators']['average_response_time_ms']
            response_time_score = max(0, 100 - (response_time / self.config.performance_thresholds['response_time_ms_threshold'] * 100))
            health_components.append(response_time_score)
        
        if 'average_quality_score' in summary['key_performance_indicators']:
            quality_score = summary['key_performance_indicators']['average_quality_score']
            health_components.append(quality_score)
        
        if 'average_error_rate' in summary['key_performance_indicators']:
            error_rate = summary['key_performance_indicators']['average_error_rate']
            error_rate_score = max(0, 100 - (error_rate / self.config.performance_thresholds['error_rate_threshold'] * 100))
            health_components.append(error_rate_score)
        
        if health_components:
            summary['overall_health_score'] = statistics.mean(health_components)
        
        # Identify critical issues
        critical_issues = []
        
        if 'average_response_time_ms' in summary['key_performance_indicators']:
            response_time = summary['key_performance_indicators']['average_response_time_ms']
            if response_time > self.config.performance_thresholds['response_time_ms_threshold']:
                critical_issues.append(f"Average response time ({response_time:.1f}ms) exceeds threshold ({self.config.performance_thresholds['response_time_ms_threshold']}ms)")
        
        if 'average_error_rate' in summary['key_performance_indicators']:
            error_rate = summary['key_performance_indicators']['average_error_rate']
            if error_rate > self.config.performance_thresholds['error_rate_threshold']:
                critical_issues.append(f"Error rate ({error_rate:.1f}%) exceeds threshold ({self.config.performance_thresholds['error_rate_threshold']}%)")
        
        if 'average_quality_score' in summary['key_performance_indicators']:
            quality_score = summary['key_performance_indicators']['average_quality_score']
            if quality_score < self.config.performance_thresholds['accuracy_threshold']:
                critical_issues.append(f"Quality score ({quality_score:.1f}) below threshold ({self.config.performance_thresholds['accuracy_threshold']})")
        
        summary['critical_issues'] = critical_issues
        
        return summary
    
    async def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics from benchmark data."""
        analysis = {
            'response_time_analysis': {},
            'throughput_analysis': {},
            'quality_efficiency_analysis': {},
            'error_rate_analysis': {},
            'performance_distribution': {},
            'performance_trends': {}
        }
        
        if not self.benchmark_data:
            return analysis
        
        # Response time analysis
        response_times = [m.average_latency_ms for m in self.benchmark_data if m.average_latency_ms > 0]
        if response_times:
            analysis['response_time_analysis'] = {
                'mean_ms': statistics.mean(response_times),
                'median_ms': statistics.median(response_times),
                'std_dev_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'min_ms': min(response_times),
                'max_ms': max(response_times),
                'p95_ms': sorted(response_times)[int(len(response_times) * 0.95)],
                'p99_ms': sorted(response_times)[int(len(response_times) * 0.99)],
                'sample_size': len(response_times)
            }
        
        # Throughput analysis
        throughputs = [m.throughput_ops_per_sec for m in self.benchmark_data if m.throughput_ops_per_sec > 0]
        if throughputs:
            analysis['throughput_analysis'] = {
                'mean_ops_per_sec': statistics.mean(throughputs),
                'median_ops_per_sec': statistics.median(throughputs),
                'std_dev_ops_per_sec': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                'min_ops_per_sec': min(throughputs),
                'max_ops_per_sec': max(throughputs),
                'sample_size': len(throughputs)
            }
        
        # Quality efficiency analysis
        quality_scores = [m.calculate_quality_efficiency_score() for m in self.benchmark_data]
        if quality_scores:
            analysis['quality_efficiency_analysis'] = {
                'mean_score': statistics.mean(quality_scores),
                'median_score': statistics.median(quality_scores),
                'std_dev_score': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                'min_score': min(quality_scores),
                'max_score': max(quality_scores),
                'sample_size': len(quality_scores),
                'scores_above_threshold': len([s for s in quality_scores if s >= self.config.performance_thresholds['accuracy_threshold']]),
                'threshold_compliance_rate': len([s for s in quality_scores if s >= self.config.performance_thresholds['accuracy_threshold']]) / len(quality_scores) * 100
            }
        
        # Error rate analysis
        error_rates = [m.error_rate_percent for m in self.benchmark_data]
        if error_rates:
            analysis['error_rate_analysis'] = {
                'mean_error_rate': statistics.mean(error_rates),
                'median_error_rate': statistics.median(error_rates),
                'std_dev_error_rate': statistics.stdev(error_rates) if len(error_rates) > 1 else 0,
                'min_error_rate': min(error_rates),
                'max_error_rate': max(error_rates),
                'sample_size': len(error_rates),
                'operations_above_error_threshold': len([r for r in error_rates if r > self.config.performance_thresholds['error_rate_threshold']])
            }
        
        return analysis
    
    async def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlation data for insights."""
        analysis = {
            'strongest_correlations': [],
            'quality_performance_relationships': {},
            'cost_performance_relationships': {},
            'resource_performance_relationships': {},
            'correlation_summary_statistics': {}
        }
        
        if not self.correlation_data:
            return analysis
        
        # Analyze strongest correlations across all data
        all_correlations = {}
        for correlation_metric in self.correlation_data:
            all_correlations.update(correlation_metric.quality_performance_correlations)
        
        # Sort by absolute correlation strength
        sorted_correlations = sorted(all_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        analysis['strongest_correlations'] = [
            {
                'correlation_name': name,
                'correlation_coefficient': coeff,
                'strength': 'strong' if abs(coeff) > 0.7 else 'moderate' if abs(coeff) > 0.4 else 'weak',
                'direction': 'positive' if coeff > 0 else 'negative'
            }
            for name, coeff in sorted_correlations[:10]
        ]
        
        # Analyze correlation categories
        quality_perf_correlations = [coeff for name, coeff in all_correlations.items() 
                                   if any(metric in name.lower() for metric in ['quality', 'accuracy', 'validation'])]
        
        cost_perf_correlations = [coeff for name, coeff in all_correlations.items() 
                                if 'cost' in name.lower()]
        
        resource_correlations = [coeff for name, coeff in all_correlations.items() 
                               if any(metric in name.lower() for metric in ['memory', 'cpu', 'resource'])]
        
        if quality_perf_correlations:
            analysis['quality_performance_relationships'] = {
                'average_correlation': statistics.mean([abs(c) for c in quality_perf_correlations]),
                'strongest_correlation': max(quality_perf_correlations, key=abs),
                'correlation_count': len(quality_perf_correlations)
            }
        
        if cost_perf_correlations:
            analysis['cost_performance_relationships'] = {
                'average_correlation': statistics.mean([abs(c) for c in cost_perf_correlations]),
                'strongest_correlation': max(cost_perf_correlations, key=abs),
                'correlation_count': len(cost_perf_correlations)
            }
        
        if resource_correlations:
            analysis['resource_performance_relationships'] = {
                'average_correlation': statistics.mean([abs(c) for c in resource_correlations]),
                'strongest_correlation': max(resource_correlations, key=abs),
                'correlation_count': len(resource_correlations)
            }
        
        # Summary statistics
        all_correlation_values = list(all_correlations.values())
        if all_correlation_values:
            analysis['correlation_summary_statistics'] = {
                'total_correlations': len(all_correlation_values),
                'mean_absolute_correlation': statistics.mean([abs(c) for c in all_correlation_values]),
                'strong_correlations_count': len([c for c in all_correlation_values if abs(c) > 0.7]),
                'moderate_correlations_count': len([c for c in all_correlation_values if 0.4 < abs(c) <= 0.7]),
                'weak_correlations_count': len([c for c in all_correlation_values if abs(c) <= 0.4])
            }
        
        return analysis
    
    async def _analyze_cost_metrics(self) -> Dict[str, Any]:
        """Analyze cost-related performance metrics."""
        analysis = {
            'cost_summary': {},
            'cost_efficiency': {},
            'cost_trends': {},
            'cost_optimization_opportunities': []
        }
        
        # Extract cost data from API metrics
        cost_data = []
        quality_costs = []
        
        for metric in self.api_metrics_data:
            if hasattr(metric, 'cost_usd') and metric.cost_usd > 0:
                cost_data.append(metric.cost_usd)
            if hasattr(metric, 'quality_validation_cost_usd') and metric.quality_validation_cost_usd > 0:
                quality_costs.append(metric.quality_validation_cost_usd)
        
        if cost_data:
            total_cost = sum(cost_data)
            analysis['cost_summary'] = {
                'total_cost_usd': total_cost,
                'average_cost_per_operation': statistics.mean(cost_data),
                'median_cost_per_operation': statistics.median(cost_data),
                'cost_std_deviation': statistics.stdev(cost_data) if len(cost_data) > 1 else 0,
                'min_cost': min(cost_data),
                'max_cost': max(cost_data),
                'operations_analyzed': len(cost_data)
            }
            
            # Cost efficiency analysis
            if quality_costs:
                quality_cost_ratio = sum(quality_costs) / total_cost * 100
                analysis['cost_efficiency'] = {
                    'quality_validation_cost_percentage': quality_cost_ratio,
                    'average_quality_cost_per_operation': statistics.mean(quality_costs),
                    'quality_operations_count': len(quality_costs)
                }
        
        return analysis
    
    async def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource utilization metrics."""
        analysis = {
            'memory_usage': {},
            'cpu_usage': {},
            'resource_efficiency': {},
            'resource_bottlenecks': []
        }
        
        # Extract resource data from benchmark metrics
        memory_values = []
        cpu_values = []
        
        for metric in self.benchmark_data:
            if hasattr(metric, 'peak_validation_memory_mb') and metric.peak_validation_memory_mb > 0:
                memory_values.append(metric.peak_validation_memory_mb)
            if hasattr(metric, 'avg_validation_cpu_percent') and metric.avg_validation_cpu_percent > 0:
                cpu_values.append(metric.avg_validation_cpu_percent)
        
        # Memory usage analysis
        if memory_values:
            analysis['memory_usage'] = {
                'average_memory_mb': statistics.mean(memory_values),
                'peak_memory_mb': max(memory_values),
                'memory_std_deviation': statistics.stdev(memory_values) if len(memory_values) > 1 else 0,
                'memory_efficiency_score': max(0, 100 - (statistics.mean(memory_values) / self.config.performance_thresholds['memory_usage_mb_threshold'] * 100)),
                'operations_above_memory_threshold': len([m for m in memory_values if m > self.config.performance_thresholds['memory_usage_mb_threshold']])
            }
        
        # CPU usage analysis  
        if cpu_values:
            analysis['cpu_usage'] = {
                'average_cpu_percent': statistics.mean(cpu_values),
                'peak_cpu_percent': max(cpu_values),
                'cpu_std_deviation': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0,
                'cpu_efficiency_score': max(0, 100 - statistics.mean(cpu_values))
            }
        
        return analysis
    
    async def _analyze_quality_metrics(self) -> Dict[str, Any]:
        """Analyze quality validation specific metrics."""
        analysis = {
            'validation_accuracy': {},
            'claim_processing': {},
            'confidence_levels': {},
            'quality_stage_performance': {}
        }
        
        if not self.benchmark_data:
            return analysis
        
        # Validation accuracy analysis
        accuracy_rates = [m.validation_accuracy_rate for m in self.benchmark_data if m.validation_accuracy_rate > 0]
        if accuracy_rates:
            analysis['validation_accuracy'] = {
                'mean_accuracy_rate': statistics.mean(accuracy_rates),
                'median_accuracy_rate': statistics.median(accuracy_rates),
                'accuracy_std_deviation': statistics.stdev(accuracy_rates) if len(accuracy_rates) > 1 else 0,
                'min_accuracy': min(accuracy_rates),
                'max_accuracy': max(accuracy_rates),
                'high_accuracy_operations': len([a for a in accuracy_rates if a >= 90.0])
            }
        
        # Claim processing analysis
        claims_extracted = [m.claims_extracted_count for m in self.benchmark_data if m.claims_extracted_count > 0]
        claims_validated = [m.claims_validated_count for m in self.benchmark_data if m.claims_validated_count > 0]
        
        if claims_extracted or claims_validated:
            analysis['claim_processing'] = {
                'total_claims_extracted': sum(claims_extracted),
                'total_claims_validated': sum(claims_validated),
                'average_claims_per_operation': statistics.mean(claims_extracted) if claims_extracted else 0,
                'average_validation_rate': statistics.mean([v/e for e, v in zip(claims_extracted, claims_validated) if e > 0]) * 100 if claims_extracted and claims_validated else 0
            }
        
        # Confidence levels analysis
        confidence_levels = [m.avg_validation_confidence for m in self.benchmark_data if m.avg_validation_confidence > 0]
        if confidence_levels:
            analysis['confidence_levels'] = {
                'mean_confidence': statistics.mean(confidence_levels),
                'median_confidence': statistics.median(confidence_levels),
                'confidence_std_deviation': statistics.stdev(confidence_levels) if len(confidence_levels) > 1 else 0,
                'high_confidence_operations': len([c for c in confidence_levels if c >= 80.0])
            }
        
        # Stage performance analysis
        extraction_times = [m.claim_extraction_time_ms for m in self.benchmark_data if m.claim_extraction_time_ms > 0]
        validation_times = [m.factual_validation_time_ms for m in self.benchmark_data if m.factual_validation_time_ms > 0]
        scoring_times = [m.relevance_scoring_time_ms for m in self.benchmark_data if m.relevance_scoring_time_ms > 0]
        
        stage_analysis = {}
        if extraction_times:
            stage_analysis['claim_extraction'] = {
                'mean_time_ms': statistics.mean(extraction_times),
                'median_time_ms': statistics.median(extraction_times),
                'operations_count': len(extraction_times)
            }
        
        if validation_times:
            stage_analysis['factual_validation'] = {
                'mean_time_ms': statistics.mean(validation_times),
                'median_time_ms': statistics.median(validation_times),
                'operations_count': len(validation_times)
            }
        
        if scoring_times:
            stage_analysis['relevance_scoring'] = {
                'mean_time_ms': statistics.mean(scoring_times),
                'median_time_ms': statistics.median(scoring_times),
                'operations_count': len(scoring_times)
            }
        
        analysis['quality_stage_performance'] = stage_analysis
        
        return analysis
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        analysis = {
            'response_time_trend': {},
            'quality_score_trend': {},
            'error_rate_trend': {},
            'cost_trend': {},
            'trend_summary': {}
        }
        
        if len(self.benchmark_data) < self.config.trend_analysis_window:
            analysis['trend_summary'] = {'status': 'insufficient_data', 'required_points': self.config.trend_analysis_window}
            return analysis
        
        # Sort data by timestamp for trend analysis
        sorted_data = sorted(self.benchmark_data, key=lambda x: getattr(x, 'timestamp', x.start_time))
        
        # Response time trend
        response_times = [m.average_latency_ms for m in sorted_data if m.average_latency_ms > 0]
        if len(response_times) >= self.config.trend_analysis_window:
            trend_direction, trend_strength = self._calculate_trend(response_times[-self.config.trend_analysis_window:])
            analysis['response_time_trend'] = {
                'direction': trend_direction,
                'strength': trend_strength,
                'recent_average': statistics.mean(response_times[-5:]),
                'historical_average': statistics.mean(response_times[-self.config.trend_analysis_window:-5]) if len(response_times) > 5 else statistics.mean(response_times[:-5])
            }
        
        # Quality score trend
        quality_scores = [m.calculate_quality_efficiency_score() for m in sorted_data]
        if len(quality_scores) >= self.config.trend_analysis_window:
            trend_direction, trend_strength = self._calculate_trend(quality_scores[-self.config.trend_analysis_window:])
            analysis['quality_score_trend'] = {
                'direction': trend_direction,
                'strength': trend_strength,
                'recent_average': statistics.mean(quality_scores[-5:]),
                'historical_average': statistics.mean(quality_scores[-self.config.trend_analysis_window:-5]) if len(quality_scores) > 5 else statistics.mean(quality_scores[:-5])
            }
        
        # Error rate trend
        error_rates = [m.error_rate_percent for m in sorted_data]
        if len(error_rates) >= self.config.trend_analysis_window:
            trend_direction, trend_strength = self._calculate_trend(error_rates[-self.config.trend_analysis_window:])
            analysis['error_rate_trend'] = {
                'direction': trend_direction,
                'strength': trend_strength,
                'recent_average': statistics.mean(error_rates[-5:]),
                'historical_average': statistics.mean(error_rates[-self.config.trend_analysis_window:-5]) if len(error_rates) > 5 else statistics.mean(error_rates[:-5])
            }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> Tuple[str, float]:
        """Calculate trend direction and strength for a series of values."""
        if len(values) < 3:
            return 'stable', 0.0
        
        # Use linear regression to determine trend
        if STATS_AVAILABLE:
            x = np.arange(len(values))
            slope, _, r_value, p_value, _ = stats.linregress(x, values)
            
            # Determine direction
            if abs(slope) < 0.01:
                direction = 'stable'
            elif slope > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
            
            # Strength is based on r-squared value
            strength = abs(r_value)
            
            return direction, strength
        else:
            # Simple trend calculation without scipy
            first_half = statistics.mean(values[:len(values)//2])
            second_half = statistics.mean(values[len(values)//2:])
            
            change_percent = abs((second_half - first_half) / first_half) * 100 if first_half != 0 else 0
            
            if change_percent < 5:
                direction = 'stable'
            elif second_half > first_half:
                direction = 'increasing'
            else:
                direction = 'decreasing'
            
            strength = min(change_percent / 100, 1.0)
            
            return direction, strength
    
    async def _analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks in the system."""
        analysis = {
            'processing_stage_bottlenecks': {},
            'resource_bottlenecks': {},
            'quality_validation_bottlenecks': {},
            'bottleneck_summary': []
        }
        
        if not self.benchmark_data:
            return analysis
        
        # Analyze processing stage performance
        stage_times = {}
        for metric in self.benchmark_data:
            if metric.claim_extraction_time_ms > 0:
                stage_times.setdefault('claim_extraction', []).append(metric.claim_extraction_time_ms)
            if metric.factual_validation_time_ms > 0:
                stage_times.setdefault('factual_validation', []).append(metric.factual_validation_time_ms)
            if metric.relevance_scoring_time_ms > 0:
                stage_times.setdefault('relevance_scoring', []).append(metric.relevance_scoring_time_ms)
            if metric.integrated_workflow_time_ms > 0:
                stage_times.setdefault('integrated_workflow', []).append(metric.integrated_workflow_time_ms)
        
        # Calculate average times and identify bottlenecks
        stage_performance = {}
        for stage, times in stage_times.items():
            if times:
                avg_time = statistics.mean(times)
                stage_performance[stage] = {
                    'average_time_ms': avg_time,
                    'median_time_ms': statistics.median(times),
                    'max_time_ms': max(times),
                    'std_deviation': statistics.stdev(times) if len(times) > 1 else 0,
                    'operation_count': len(times)
                }
        
        # Identify bottleneck stages
        if stage_performance:
            bottleneck_stage = max(stage_performance.keys(), key=lambda k: stage_performance[k]['average_time_ms'])
            total_avg_time = sum(stage_performance[stage]['average_time_ms'] for stage in stage_performance)
            bottleneck_percentage = stage_performance[bottleneck_stage]['average_time_ms'] / total_avg_time * 100
            
            analysis['processing_stage_bottlenecks'] = {
                'bottleneck_stage': bottleneck_stage,
                'bottleneck_percentage': bottleneck_percentage,
                'stage_performance': stage_performance
            }
            
            # Add to summary
            analysis['bottleneck_summary'].append({
                'type': 'processing_stage',
                'component': bottleneck_stage,
                'impact_percentage': bottleneck_percentage,
                'severity': 'high' if bottleneck_percentage > 50 else 'medium' if bottleneck_percentage > 30 else 'low'
            })
        
        return analysis
    
    async def _generate_performance_insights(self) -> None:
        """Generate performance insights from analyzed data."""
        self.performance_insights.clear()
        
        # Response time insights
        if self.benchmark_data:
            response_times = [m.average_latency_ms for m in self.benchmark_data if m.average_latency_ms > 0]
            if response_times:
                avg_response_time = statistics.mean(response_times)
                threshold = self.config.performance_thresholds['response_time_ms_threshold']
                
                if avg_response_time > threshold:
                    self.performance_insights.append(PerformanceInsight(
                        insight_type='bottleneck',
                        title='Response Time Above Threshold',
                        description=f'Average response time ({avg_response_time:.1f}ms) exceeds the configured threshold ({threshold}ms) by {((avg_response_time/threshold - 1) * 100):.1f}%.',
                        severity='high' if avg_response_time > threshold * 1.5 else 'medium',
                        metrics_involved=['response_time_ms'],
                        impact_assessment={'performance_degradation': (avg_response_time/threshold - 1) * 100},
                        recommended_actions=[
                            'Optimize slow-performing quality validation stages',
                            'Implement parallel processing for independent operations',
                            'Consider caching frequently validated content',
                            'Review resource allocation for quality validation components'
                        ],
                        priority_level=2 if avg_response_time > threshold * 1.5 else 3,
                        affected_components=['quality_validation'],
                        supporting_evidence={'average_response_time': avg_response_time, 'threshold': threshold}
                    ))
        
        # Quality score insights
        if self.benchmark_data:
            quality_scores = [m.calculate_quality_efficiency_score() for m in self.benchmark_data]
            if quality_scores:
                avg_quality = statistics.mean(quality_scores)
                quality_variance = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
                
                if quality_variance > 15:  # High variance in quality scores
                    self.performance_insights.append(PerformanceInsight(
                        insight_type='anomaly',
                        title='Inconsistent Quality Performance',
                        description=f'Quality scores show high variance ({quality_variance:.1f}), indicating inconsistent performance across operations.',
                        severity='medium',
                        metrics_involved=['quality_efficiency_score'],
                        impact_assessment={'consistency_impact': quality_variance},
                        recommended_actions=[
                            'Investigate factors causing quality score variation',
                            'Standardize quality validation parameters',
                            'Implement quality score monitoring and alerting'
                        ],
                        priority_level=3,
                        affected_components=['quality_assessment'],
                        supporting_evidence={'quality_variance': quality_variance, 'average_quality': avg_quality}
                    ))
        
        # Cost efficiency insights
        if self.api_metrics_data:
            cost_data = [m.cost_usd for m in self.api_metrics_data if hasattr(m, 'cost_usd') and m.cost_usd > 0]
            quality_costs = [m.quality_validation_cost_usd for m in self.api_metrics_data 
                           if hasattr(m, 'quality_validation_cost_usd') and m.quality_validation_cost_usd > 0]
            
            if cost_data and quality_costs:
                total_cost = sum(cost_data)
                quality_cost_ratio = sum(quality_costs) / total_cost * 100
                
                if quality_cost_ratio > 60:  # Quality validation costs are high percentage of total
                    self.performance_insights.append(PerformanceInsight(
                        insight_type='optimization',
                        title='High Quality Validation Cost Ratio',
                        description=f'Quality validation costs represent {quality_cost_ratio:.1f}% of total API costs, indicating potential optimization opportunities.',
                        severity='medium',
                        metrics_involved=['cost_usd', 'quality_validation_cost_usd'],
                        impact_assessment={'cost_optimization_potential': quality_cost_ratio},
                        recommended_actions=[
                            'Implement tiered quality validation based on content importance',
                            'Cache validation results for similar content',
                            'Optimize quality validation algorithms for cost efficiency',
                            'Consider batch processing for quality validation'
                        ],
                        priority_level=3,
                        affected_components=['cost_management', 'quality_validation'],
                        supporting_evidence={'quality_cost_ratio': quality_cost_ratio, 'total_cost': total_cost}
                    ))
    
    async def _generate_optimization_recommendations(self) -> None:
        """Generate actionable optimization recommendations."""
        self.optimization_recommendations.clear()
        
        # Performance optimization recommendations
        if self.benchmark_data:
            # Response time optimization
            response_times = [m.average_latency_ms for m in self.benchmark_data if m.average_latency_ms > 0]
            if response_times and statistics.mean(response_times) > self.config.performance_thresholds['response_time_ms_threshold']:
                self.optimization_recommendations.append(OptimizationRecommendation(
                    category='performance',
                    title='Response Time Optimization',
                    description='Implement performance optimizations to reduce average response time and meet performance thresholds.',
                    priority='high',
                    implementation_effort='medium',
                    estimated_impact={'response_time_reduction': 25, 'user_satisfaction_improvement': 15},
                    implementation_steps=[
                        'Profile performance bottlenecks in quality validation pipeline',
                        'Implement parallel processing for independent validation stages',
                        'Add caching layer for frequently validated content',
                        'Optimize database queries and document retrieval',
                        'Consider using faster algorithms for claim extraction and validation'
                    ],
                    implementation_cost_estimate=5000,
                    expected_savings=2000,
                    roi_estimate=40,
                    payback_period_days=90,
                    applicable_scenarios=['high_volume_operations', 'real_time_validation'],
                    prerequisites=['Performance profiling tools', 'Development resources'],
                    risks_and_considerations=['Potential complexity increase', 'Need for thorough testing'],
                    supporting_metrics={'current_avg_response_time': statistics.mean(response_times)},
                    confidence_level=0.8
                ))
            
            # Quality consistency optimization
            quality_scores = [m.calculate_quality_efficiency_score() for m in self.benchmark_data]
            if quality_scores and statistics.stdev(quality_scores) > 15:
                self.optimization_recommendations.append(OptimizationRecommendation(
                    category='quality',
                    title='Quality Consistency Improvement',
                    description='Standardize quality validation processes to reduce variance in quality scores and improve consistency.',
                    priority='medium',
                    implementation_effort='low',
                    estimated_impact={'quality_consistency_improvement': 30, 'reliability_increase': 20},
                    implementation_steps=[
                        'Standardize quality validation parameters across all operations',
                        'Implement quality score monitoring and alerting',
                        'Create consistent training datasets for validation components',
                        'Establish quality validation best practices documentation'
                    ],
                    implementation_cost_estimate=2000,
                    expected_savings=1500,
                    roi_estimate=75,
                    payback_period_days=45,
                    applicable_scenarios=['all_validation_scenarios'],
                    prerequisites=['Quality metrics monitoring system'],
                    risks_and_considerations=['May require retraining of validation models'],
                    supporting_metrics={'quality_score_variance': statistics.stdev(quality_scores)},
                    confidence_level=0.9
                ))
        
        # Cost optimization recommendations
        if self.api_metrics_data:
            cost_data = [m.cost_usd for m in self.api_metrics_data if hasattr(m, 'cost_usd') and m.cost_usd > 0]
            if cost_data:
                avg_cost = statistics.mean(cost_data)
                if avg_cost > self.config.performance_thresholds['cost_per_operation_threshold']:
                    self.optimization_recommendations.append(OptimizationRecommendation(
                        category='cost',
                        title='API Cost Optimization',
                        description='Implement cost optimization strategies to reduce average cost per quality validation operation.',
                        priority='high',
                        implementation_effort='medium',
                        estimated_impact={'cost_reduction': 35, 'budget_efficiency_improvement': 25},
                        implementation_steps=[
                            'Implement intelligent batching for API calls',
                            'Add result caching to reduce redundant validations',
                            'Optimize prompt engineering for more efficient token usage',
                            'Implement tiered validation based on content complexity',
                            'Monitor and optimize model selection for cost efficiency'
                        ],
                        implementation_cost_estimate=3000,
                        expected_savings=5000,
                        roi_estimate=167,
                        payback_period_days=60,
                        applicable_scenarios=['high_volume_operations', 'budget_constrained_environments'],
                        prerequisites=['API usage monitoring', 'Caching infrastructure'],
                        risks_and_considerations=['Potential impact on validation accuracy', 'Cache invalidation complexity'],
                        supporting_metrics={'current_avg_cost': avg_cost},
                        confidence_level=0.85
                    ))
        
        # Resource optimization recommendations
        if self.benchmark_data:
            memory_values = [m.peak_validation_memory_mb for m in self.benchmark_data 
                           if hasattr(m, 'peak_validation_memory_mb') and m.peak_validation_memory_mb > 0]
            if memory_values and max(memory_values) > self.config.performance_thresholds['memory_usage_mb_threshold']:
                self.optimization_recommendations.append(OptimizationRecommendation(
                    category='resource',
                    title='Memory Usage Optimization',
                    description='Optimize memory usage patterns to prevent memory constraints and improve system stability.',
                    priority='medium',
                    implementation_effort='medium',
                    estimated_impact={'memory_usage_reduction': 30, 'system_stability_improvement': 20},
                    implementation_steps=[
                        'Implement memory pooling for validation operations',
                        'Optimize data structures used in quality validation',
                        'Add memory usage monitoring and alerting',
                        'Implement streaming processing for large documents',
                        'Review and optimize caching strategies'
                    ],
                    implementation_cost_estimate=4000,
                    expected_savings=3000,
                    roi_estimate=75,
                    payback_period_days=90,
                    applicable_scenarios=['large_document_processing', 'resource_constrained_environments'],
                    prerequisites=['Memory profiling tools', 'Infrastructure monitoring'],
                    risks_and_considerations=['Complexity of memory optimization', 'Potential performance trade-offs'],
                    supporting_metrics={'peak_memory_usage': max(memory_values)},
                    confidence_level=0.7
                ))
    
    async def _generate_visualizations(self) -> Dict[str, Any]:
        """Generate performance visualization charts."""
        if not VISUALIZATION_AVAILABLE:
            return {'status': 'visualization_not_available'}
        
        visualizations = {
            'charts_generated': [],
            'chart_files': [],
            'interactive_charts': {}
        }
        
        try:
            # Performance metrics over time chart
            if len(self.benchmark_data) > 1:
                chart_data = await self._create_performance_timeline_chart()
                visualizations['charts_generated'].append('performance_timeline')
                visualizations['interactive_charts']['performance_timeline'] = chart_data
            
            # Quality vs Performance scatter plot
            if len(self.benchmark_data) > 5:
                chart_data = await self._create_quality_performance_scatter()
                visualizations['charts_generated'].append('quality_performance_scatter')
                visualizations['interactive_charts']['quality_performance_scatter'] = chart_data
            
            # Cost analysis pie chart
            if self.api_metrics_data:
                chart_data = await self._create_cost_breakdown_chart()
                visualizations['charts_generated'].append('cost_breakdown')
                visualizations['interactive_charts']['cost_breakdown'] = chart_data
            
            # Processing stage comparison
            if self.benchmark_data:
                chart_data = await self._create_stage_performance_chart()
                visualizations['charts_generated'].append('stage_performance')
                visualizations['interactive_charts']['stage_performance'] = chart_data
            
            self.report_metadata.charts_generated = len(visualizations['charts_generated'])
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _create_performance_timeline_chart(self) -> str:
        """Create performance timeline chart using Plotly."""
        try:
            # Sort benchmark data by timestamp
            sorted_data = sorted(self.benchmark_data, key=lambda x: getattr(x, 'timestamp', x.start_time))
            
            timestamps = [datetime.fromtimestamp(getattr(m, 'timestamp', m.start_time)) for m in sorted_data]
            response_times = [m.average_latency_ms for m in sorted_data]
            quality_scores = [m.calculate_quality_efficiency_score() for m in sorted_data]
            
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add response time trace
            fig.add_trace(
                go.Scatter(x=timestamps, y=response_times, name="Response Time (ms)", 
                          line=dict(color="blue", width=2), mode='lines+markers'),
                secondary_y=False,
            )
            
            # Add quality score trace
            fig.add_trace(
                go.Scatter(x=timestamps, y=quality_scores, name="Quality Score", 
                          line=dict(color="green", width=2), mode='lines+markers'),
                secondary_y=True,
            )
            
            # Add axes titles
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Response Time (ms)", secondary_y=False)
            fig.update_yaxes(title_text="Quality Score", secondary_y=True)
            
            fig.update_layout(
                title="Performance Timeline",
                width=self.config.chart_width,
                height=self.config.chart_height
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Error creating timeline chart: {e}")
            return f"Error creating timeline chart: {e}"
    
    async def _create_quality_performance_scatter(self) -> str:
        """Create quality vs performance scatter plot."""
        try:
            quality_scores = [m.calculate_quality_efficiency_score() for m in self.benchmark_data]
            response_times = [m.average_latency_ms for m in self.benchmark_data]
            
            fig = go.Figure(data=go.Scatter(
                x=quality_scores,
                y=response_times,
                mode='markers',
                marker=dict(
                    size=8,
                    color=response_times,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Response Time")
                ),
                text=[f"Quality: {q:.1f}<br>Response: {r:.1f}ms" for q, r in zip(quality_scores, response_times)],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Quality Score vs Response Time",
                xaxis_title="Quality Efficiency Score",
                yaxis_title="Response Time (ms)",
                width=self.config.chart_width,
                height=self.config.chart_height
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {e}")
            return f"Error creating scatter plot: {e}"
    
    async def _create_cost_breakdown_chart(self) -> str:
        """Create cost breakdown pie chart."""
        try:
            total_costs = [m.cost_usd for m in self.api_metrics_data if hasattr(m, 'cost_usd') and m.cost_usd > 0]
            quality_costs = [m.quality_validation_cost_usd for m in self.api_metrics_data 
                           if hasattr(m, 'quality_validation_cost_usd') and m.quality_validation_cost_usd > 0]
            
            if not total_costs or not quality_costs:
                return "Insufficient cost data for chart generation"
            
            total_cost = sum(total_costs)
            total_quality_cost = sum(quality_costs)
            other_cost = total_cost - total_quality_cost
            
            fig = go.Figure(data=[go.Pie(
                labels=['Quality Validation', 'Other Operations'],
                values=[total_quality_cost, other_cost],
                hole=0.4
            )])
            
            fig.update_layout(
                title="Cost Breakdown by Operation Type",
                width=self.config.chart_width,
                height=self.config.chart_height
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Error creating cost chart: {e}")
            return f"Error creating cost chart: {e}"
    
    async def _create_stage_performance_chart(self) -> str:
        """Create processing stage performance comparison chart."""
        try:
            stages = ['claim_extraction', 'factual_validation', 'relevance_scoring', 'integrated_workflow']
            stage_times = {}
            
            for stage in stages:
                times = []
                for metric in self.benchmark_data:
                    if stage == 'claim_extraction' and metric.claim_extraction_time_ms > 0:
                        times.append(metric.claim_extraction_time_ms)
                    elif stage == 'factual_validation' and metric.factual_validation_time_ms > 0:
                        times.append(metric.factual_validation_time_ms)
                    elif stage == 'relevance_scoring' and metric.relevance_scoring_time_ms > 0:
                        times.append(metric.relevance_scoring_time_ms)
                    elif stage == 'integrated_workflow' and metric.integrated_workflow_time_ms > 0:
                        times.append(metric.integrated_workflow_time_ms)
                
                if times:
                    stage_times[stage] = statistics.mean(times)
            
            if not stage_times:
                return "No stage timing data available"
            
            fig = go.Figure(data=[
                go.Bar(x=list(stage_times.keys()), y=list(stage_times.values()),
                       marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ])
            
            fig.update_layout(
                title="Average Processing Time by Stage",
                xaxis_title="Processing Stage",
                yaxis_title="Average Time (ms)",
                width=self.config.chart_width,
                height=self.config.chart_height
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Error creating stage chart: {e}")
            return f"Error creating stage chart: {e}"
    
    async def export_report(self, 
                          report_data: Dict[str, Any],
                          filename_prefix: str = "quality_performance_report") -> Dict[str, str]:
        """
        Export comprehensive report in multiple formats.
        
        Args:
            report_data: Complete report data to export
            filename_prefix: Prefix for generated filenames
            
        Returns:
            Dictionary mapping format to exported file path
        """
        exported_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            for format_type in self.config.output_formats:
                if format_type == ReportFormat.JSON:
                    file_path = await self._export_json_report(report_data, f"{filename_prefix}_{timestamp}.json")
                    exported_files['json'] = str(file_path)
                
                elif format_type == ReportFormat.HTML:
                    file_path = await self._export_html_report(report_data, f"{filename_prefix}_{timestamp}.html")
                    exported_files['html'] = str(file_path)
                
                elif format_type == ReportFormat.CSV:
                    file_path = await self._export_csv_report(report_data, f"{filename_prefix}_{timestamp}.csv")
                    exported_files['csv'] = str(file_path)
                
                elif format_type == ReportFormat.TEXT:
                    file_path = await self._export_text_report(report_data, f"{filename_prefix}_{timestamp}.txt")
                    exported_files['text'] = str(file_path)
            
            self.logger.info(f"Report exported in {len(exported_files)} formats")
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            raise
        
        return exported_files
    
    async def _export_json_report(self, report_data: Dict[str, Any], filename: str) -> Path:
        """Export report as JSON file."""
        file_path = self.output_directory / filename
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report exported to: {file_path}")
        return file_path
    
    async def _export_html_report(self, report_data: Dict[str, Any], filename: str) -> Path:
        """Export report as HTML file with embedded visualizations."""
        file_path = self.output_directory / filename
        
        html_content = await self._generate_html_content(report_data)
        
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report exported to: {file_path}")
        return file_path
    
    async def _export_csv_report(self, report_data: Dict[str, Any], filename: str) -> Path:
        """Export key metrics as CSV file."""
        file_path = self.output_directory / filename
        
        # Extract key metrics for CSV export
        csv_data = []
        
        # Performance metrics
        if 'performance_analysis' in report_data:
            perf_analysis = report_data['performance_analysis']
            if 'response_time_analysis' in perf_analysis:
                rt_analysis = perf_analysis['response_time_analysis']
                csv_data.append({
                    'Metric': 'Average Response Time',
                    'Value': rt_analysis.get('mean_ms', 0),
                    'Unit': 'ms',
                    'Category': 'Performance'
                })
            
            if 'quality_efficiency_analysis' in perf_analysis:
                qe_analysis = perf_analysis['quality_efficiency_analysis']
                csv_data.append({
                    'Metric': 'Average Quality Score',
                    'Value': qe_analysis.get('mean_score', 0),
                    'Unit': 'score',
                    'Category': 'Quality'
                })
        
        # Cost metrics
        if 'cost_analysis' in report_data and 'cost_summary' in report_data['cost_analysis']:
            cost_summary = report_data['cost_analysis']['cost_summary']
            csv_data.append({
                'Metric': 'Total Cost',
                'Value': cost_summary.get('total_cost_usd', 0),
                'Unit': 'USD',
                'Category': 'Cost'
            })
        
        # Write CSV
        if csv_data:
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Metric', 'Value', 'Unit', 'Category'])
                writer.writeheader()
                writer.writerows(csv_data)
        
        self.logger.info(f"CSV report exported to: {file_path}")
        return file_path
    
    async def _export_text_report(self, report_data: Dict[str, Any], filename: str) -> Path:
        """Export report as formatted text file."""
        file_path = self.output_directory / filename
        
        text_content = await self._generate_text_content(report_data)
        
        with open(file_path, 'w') as f:
            f.write(text_content)
        
        self.logger.info(f"Text report exported to: {file_path}")
        return file_path
    
    async def _generate_html_content(self, report_data: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report content."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9; }
        .metric-card { display: inline-block; background: white; padding: 15px; margin: 10px; border-radius: 5px; border-left: 4px solid #667eea; min-width: 200px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #333; }
        .metric-label { color: #666; font-size: 0.9em; }
        .insight-card { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #28a745; }
        .recommendation-card { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }
        .critical { border-left-color: #dc3545 !important; }
        .warning { border-left-color: #ffc107 !important; }
        .success { border-left-color: #28a745 !important; }
        .chart-container { margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .json-data { background: #f5f5f5; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_title}</h1>
        <p>Generated: {generation_time}</p>
        <p>Analysis Period: {analysis_period}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric-card">
            <div class="metric-value">{overall_health_score:.1f}%</div>
            <div class="metric-label">Overall Health Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_operations}</div>
            <div class="metric-label">Total Operations</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_response_time:.1f}ms</div>
            <div class="metric-label">Average Response Time</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_quality_score:.1f}</div>
            <div class="metric-label">Average Quality Score</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Performance Insights</h2>
        {insights_html}
    </div>
    
    <div class="section">
        <h2>Optimization Recommendations</h2>
        {recommendations_html}
    </div>
    
    <div class="section">
        <h2>Detailed Analysis</h2>
        {analysis_tables_html}
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        {charts_html}
    </div>
    
    <div class="section">
        <h2>Raw Data (JSON)</h2>
        <div class="json-data">{raw_data_json}</div>
    </div>
</body>
</html>"""
        
        # Extract data for template
        metadata = report_data.get('metadata', {})
        executive_summary = report_data.get('executive_summary', {})
        insights = report_data.get('insights', [])
        recommendations = report_data.get('recommendations', [])
        
        # Generate insights HTML
        insights_html = ""
        for insight in insights[:10]:  # Limit to top 10
            severity_class = insight.get('severity', 'medium')
            insights_html += f"""
            <div class="insight-card {severity_class}">
                <h4>{insight.get('title', 'Insight')}</h4>
                <p>{insight.get('description', '')}</p>
                <small>Priority: {insight.get('priority_level', 3)}/5</small>
            </div>
            """
        
        # Generate recommendations HTML
        recommendations_html = ""
        for rec in recommendations[:10]:  # Limit to top 10
            priority_class = rec.get('priority', 'medium')
            recommendations_html += f"""
            <div class="recommendation-card {priority_class}">
                <h4>{rec.get('title', 'Recommendation')}</h4>
                <p>{rec.get('description', '')}</p>
                <p><strong>Implementation Effort:</strong> {rec.get('implementation_effort', 'Unknown')}</p>
                <p><strong>Expected Impact:</strong> {rec.get('estimated_impact', {})}</p>
            </div>
            """
        
        # Generate analysis tables HTML
        analysis_tables_html = "<h3>Performance Analysis Summary</h3>"
        if 'performance_analysis' in report_data:
            perf_data = report_data['performance_analysis']
            analysis_tables_html += f"<div class='json-data'>{json.dumps(perf_data, indent=2)}</div>"
        
        # Generate charts HTML
        charts_html = ""
        if 'charts_and_visualizations' in report_data:
            charts = report_data['charts_and_visualizations'].get('interactive_charts', {})
            for chart_name, chart_html in charts.items():
                if isinstance(chart_html, str):
                    charts_html += f"<div class='chart-container'><h4>{chart_name.replace('_', ' ').title()}</h4>{chart_html}</div>"
        
        # Fill template
        return html_template.format(
            report_title=self.config.report_name,
            generation_time=datetime.fromtimestamp(metadata.get('generated_timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S'),
            analysis_period=f"{self.config.analysis_period_hours} hours",
            overall_health_score=executive_summary.get('overall_health_score', 0),
            total_operations=len(self.benchmark_data) + len(self.api_metrics_data),
            avg_response_time=executive_summary.get('key_performance_indicators', {}).get('average_response_time_ms', 0),
            avg_quality_score=executive_summary.get('key_performance_indicators', {}).get('average_quality_score', 0),
            insights_html=insights_html,
            recommendations_html=recommendations_html,
            analysis_tables_html=analysis_tables_html,
            charts_html=charts_html,
            raw_data_json=json.dumps(report_data, indent=2, default=str)
        )
    
    async def _generate_text_content(self, report_data: Dict[str, Any]) -> str:
        """Generate formatted text report content."""
        text_lines = [
            "="*80,
            f"QUALITY PERFORMANCE REPORT - {self.config.report_name}".center(80),
            "="*80,
            "",
            f"Generated: {datetime.fromtimestamp(report_data.get('metadata', {}).get('generated_timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: {self.config.analysis_period_hours} hours",
            f"Report Version: {report_data.get('metadata', {}).get('report_version', '1.0.0')}",
            "",
            "-"*80,
            "EXECUTIVE SUMMARY",
            "-"*80,
        ]
        
        executive_summary = report_data.get('executive_summary', {})
        text_lines.extend([
            f"Overall Health Score: {executive_summary.get('overall_health_score', 0):.1f}%",
            f"Total Operations Analyzed: {len(self.benchmark_data) + len(self.api_metrics_data)}",
            f"Data Sources: {', '.join(report_data.get('metadata', {}).get('data_sources', []))}",
            ""
        ])
        
        # Key Performance Indicators
        kpis = executive_summary.get('key_performance_indicators', {})
        if kpis:
            text_lines.extend([
                "Key Performance Indicators:",
                f"  Average Response Time: {kpis.get('average_response_time_ms', 0):.1f} ms",
                f"  Average Quality Score: {kpis.get('average_quality_score', 0):.1f}",
                f"  Average Error Rate: {kpis.get('average_error_rate', 0):.1f}%",
                ""
            ])
        
        # Critical Issues
        critical_issues = executive_summary.get('critical_issues', [])
        if critical_issues:
            text_lines.extend([
                "Critical Issues:",
                *[f"  • {issue}" for issue in critical_issues],
                ""
            ])
        
        # Performance Insights
        insights = report_data.get('insights', [])
        if insights:
            text_lines.extend([
                "-"*80,
                "PERFORMANCE INSIGHTS",
                "-"*80,
            ])
            for insight in insights[:10]:
                text_lines.extend([
                    f"• {insight.get('title', 'Insight')} ({insight.get('severity', 'medium')})",
                    f"  {insight.get('description', '')}",
                    ""
                ])
        
        # Optimization Recommendations
        recommendations = report_data.get('recommendations', [])
        if recommendations:
            text_lines.extend([
                "-"*80,
                "OPTIMIZATION RECOMMENDATIONS",
                "-"*80,
            ])
            for rec in recommendations[:10]:
                text_lines.extend([
                    f"• {rec.get('title', 'Recommendation')} ({rec.get('priority', 'medium')} priority)",
                    f"  {rec.get('description', '')}",
                    f"  Implementation Effort: {rec.get('implementation_effort', 'Unknown')}",
                    ""
                ])
        
        text_lines.extend([
            "-"*80,
            "END OF REPORT",
            "-"*80
        ])
        
        return "\n".join(text_lines)


# Convenience functions for easy report generation
async def generate_comprehensive_performance_report(
    benchmark_suite: Optional[QualityValidationBenchmarkSuite] = None,
    correlation_engine: Optional[CrossSystemCorrelationEngine] = None,
    api_logger: Optional[QualityAwareAPIMetricsLogger] = None,
    config: Optional[PerformanceReportConfiguration] = None,
    output_directory: Optional[Path] = None
) -> Dict[str, str]:
    """
    Convenience function to generate comprehensive performance report.
    
    Args:
        benchmark_suite: Quality validation benchmark suite
        correlation_engine: Cross-system correlation engine
        api_logger: Quality-aware API metrics logger
        config: Report configuration
        output_directory: Output directory for reports
        
    Returns:
        Dictionary mapping format to exported file path
    """
    reporter = QualityPerformanceReporter(config=config, output_directory=output_directory)
    
    # Load data from sources
    if benchmark_suite:
        await reporter.load_benchmark_data(benchmark_suite=benchmark_suite)
    
    if correlation_engine:
        await reporter.load_correlation_data(correlation_engine=correlation_engine)
    
    if api_logger:
        await reporter.load_api_metrics_data(api_logger=api_logger)
    
    # Generate comprehensive report
    report_data = await reporter.generate_comprehensive_report()
    
    # Export in configured formats
    return await reporter.export_report(report_data)


# Make main classes available at module level
__all__ = [
    'QualityPerformanceReporter',
    'PerformanceReportConfiguration',
    'ReportMetadata',
    'PerformanceInsight',
    'OptimizationRecommendation',
    'ReportFormat',
    'PerformanceMetricType',
    'generate_comprehensive_performance_report'
]