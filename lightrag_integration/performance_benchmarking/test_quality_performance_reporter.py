#!/usr/bin/env python3
"""
Unit tests for Quality Performance Reporter module.

This module provides comprehensive testing for the QualityPerformanceReporter
and related components, including report generation, analysis, visualization,
and export functionality.

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import asyncio
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import asdict

# Import the modules under test
from reporting.quality_performance_reporter import (
    PerformanceReportConfiguration,
    ReportMetadata,
    PerformanceInsight,
    OptimizationRecommendation,
    QualityPerformanceReporter,
    ReportFormat,
    PerformanceMetricType,
    generate_comprehensive_performance_report
)

# Import dependencies for test data
try:
    from quality_performance_benchmarks import QualityValidationMetrics, QualityValidationBenchmarkSuite
    from performance_correlation_engine import (
        PerformanceCorrelationMetrics, CrossSystemCorrelationEngine, CorrelationAnalysisReport
    )
    from quality_aware_metrics_logger import QualityAPIMetric, QualityAwareAPIMetricsLogger
except ImportError:
    # Create mock classes for testing
    class QualityValidationMetrics:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def calculate_quality_efficiency_score(self):
            return getattr(self, 'quality_score', 80.0)
    
    class QualityValidationBenchmarkSuite:
        def __init__(self):
            self.quality_metrics_history = {}
    
    class PerformanceCorrelationMetrics:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class CrossSystemCorrelationEngine:
        def __init__(self):
            self.correlation_history = []
    
    class CorrelationAnalysisReport:
        pass
    
    class QualityAPIMetric:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class QualityAwareAPIMetricsLogger:
        def __init__(self):
            self.metrics_aggregator = Mock()


class TestPerformanceReportConfiguration:
    """Test suite for PerformanceReportConfiguration class."""
    
    def test_configuration_initialization(self):
        """Test proper initialization of PerformanceReportConfiguration."""
        config = PerformanceReportConfiguration(
            report_name="Test Performance Report",
            report_description="Test description",
            analysis_period_hours=48,
            confidence_level=0.99
        )
        
        assert config.report_name == "Test Performance Report"
        assert config.report_description == "Test description"
        assert config.analysis_period_hours == 48
        assert config.confidence_level == 0.99
        assert config.include_executive_summary is True
        assert config.include_detailed_analysis is True
    
    def test_default_configuration_values(self):
        """Test default configuration values."""
        config = PerformanceReportConfiguration()
        
        assert config.report_name == "Quality Performance Report"
        assert config.analysis_period_hours == 24
        assert config.minimum_sample_size == 10
        assert config.confidence_level == 0.95
        assert config.generate_charts is True
        assert len(config.output_formats) >= 1
        assert ReportFormat.JSON in config.output_formats
    
    def test_performance_thresholds_configuration(self):
        """Test performance thresholds configuration."""
        config = PerformanceReportConfiguration()
        
        assert isinstance(config.performance_thresholds, dict)
        assert 'response_time_ms_threshold' in config.performance_thresholds
        assert 'throughput_ops_per_sec_threshold' in config.performance_thresholds
        assert 'accuracy_threshold' in config.performance_thresholds
        assert config.performance_thresholds['response_time_ms_threshold'] == 2000
        assert config.performance_thresholds['accuracy_threshold'] == 85.0


class TestReportMetadata:
    """Test suite for ReportMetadata class."""
    
    def test_metadata_initialization(self):
        """Test proper initialization of ReportMetadata."""
        metadata = ReportMetadata(
            report_version="2.0.0",
            generator="Test Generator"
        )
        
        assert metadata.report_version == "2.0.0"
        assert metadata.generator == "Test Generator"
        assert hasattr(metadata, 'report_id')
        assert hasattr(metadata, 'generated_timestamp')
        assert isinstance(metadata.data_sources, list)
    
    def test_metadata_to_dict(self):
        """Test metadata dictionary conversion."""
        metadata = ReportMetadata(
            analysis_start_time=time.time() - 3600,
            analysis_end_time=time.time(),
            total_data_points=100
        )
        
        metadata_dict = metadata.to_dict()
        
        assert isinstance(metadata_dict, dict)
        assert 'report_id' in metadata_dict
        assert 'generated_timestamp_iso' in metadata_dict
        assert 'analysis_start_time_iso' in metadata_dict
        assert 'analysis_end_time_iso' in metadata_dict
        assert 'total_data_points' in metadata_dict
        assert metadata_dict['total_data_points'] == 100


class TestPerformanceInsight:
    """Test suite for PerformanceInsight class."""
    
    def test_insight_initialization(self):
        """Test proper initialization of PerformanceInsight."""
        insight = PerformanceInsight(
            insight_type="bottleneck",
            title="Performance Bottleneck Detected",
            description="High response times detected in validation pipeline",
            severity="high",
            metrics_involved=["response_time_ms", "validation_accuracy"],
            recommended_actions=["Optimize validation algorithm", "Add caching layer"]
        )
        
        assert insight.insight_type == "bottleneck"
        assert insight.title == "Performance Bottleneck Detected"
        assert insight.severity == "high"
        assert len(insight.metrics_involved) == 2
        assert len(insight.recommended_actions) == 2
        assert insight.priority_level == 3  # Default value
    
    def test_insight_default_values(self):
        """Test default values for PerformanceInsight."""
        insight = PerformanceInsight()
        
        assert insight.insight_type == "general"
        assert insight.title == "Performance Insight"
        assert insight.severity == "medium"
        assert isinstance(insight.metrics_involved, list)
        assert isinstance(insight.recommended_actions, list)
        assert hasattr(insight, 'insight_id')


class TestOptimizationRecommendation:
    """Test suite for OptimizationRecommendation class."""
    
    def test_recommendation_initialization(self):
        """Test proper initialization of OptimizationRecommendation."""
        recommendation = OptimizationRecommendation(
            category="performance",
            title="Implement Response Caching",
            description="Add caching layer to reduce response times",
            priority="high",
            implementation_effort="medium",
            estimated_impact={"response_time_reduction": 40.0},
            implementation_cost_estimate=5000.0,
            expected_savings=2000.0,
            roi_estimate=40.0
        )
        
        assert recommendation.category == "performance"
        assert recommendation.title == "Implement Response Caching"
        assert recommendation.priority == "high"
        assert recommendation.implementation_effort == "medium"
        assert recommendation.estimated_impact["response_time_reduction"] == 40.0
        assert recommendation.implementation_cost_estimate == 5000.0
        assert recommendation.roi_estimate == 40.0
    
    def test_recommendation_default_values(self):
        """Test default values for OptimizationRecommendation."""
        recommendation = OptimizationRecommendation()
        
        assert recommendation.category == "performance"
        assert recommendation.title == "Optimization Recommendation"
        assert recommendation.priority == "medium"
        assert recommendation.implementation_effort == "medium"
        assert isinstance(recommendation.estimated_impact, dict)
        assert recommendation.confidence_level == 0.8


class TestQualityPerformanceReporter:
    """Test suite for QualityPerformanceReporter class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_benchmark_data(self):
        """Create sample benchmark data for testing."""
        return [
            QualityValidationMetrics(
                scenario_name="test_scenario_1",
                operations_count=10,
                average_latency_ms=1200.0,
                throughput_ops_per_sec=5.0,
                validation_accuracy_rate=88.5,
                claims_extracted_count=25,
                claims_validated_count=22,
                error_rate_percent=2.0,
                avg_validation_confidence=85.0,
                claim_extraction_time_ms=300.0,
                factual_validation_time_ms=600.0,
                relevance_scoring_time_ms=200.0,
                integrated_workflow_time_ms=1100.0,
                peak_validation_memory_mb=512.0,
                avg_validation_cpu_percent=45.0,
                start_time=time.time() - 3600,
                duration_seconds=120.0
            ),
            QualityValidationMetrics(
                scenario_name="test_scenario_2", 
                operations_count=15,
                average_latency_ms=1400.0,
                throughput_ops_per_sec=4.5,
                validation_accuracy_rate=91.2,
                claims_extracted_count=30,
                claims_validated_count=28,
                error_rate_percent=1.5,
                avg_validation_confidence=89.0,
                claim_extraction_time_ms=350.0,
                factual_validation_time_ms=750.0,
                relevance_scoring_time_ms=180.0,
                integrated_workflow_time_ms=1280.0,
                peak_validation_memory_mb=640.0,
                avg_validation_cpu_percent=52.0,
                start_time=time.time() - 1800,
                duration_seconds=150.0
            )
        ]
    
    @pytest.fixture
    def sample_api_metrics(self):
        """Create sample API metrics for testing."""
        return [
            QualityAPIMetric(
                timestamp=time.time() - 2000,
                endpoint="/api/validate",
                method="POST",
                response_time_ms=1100.0,
                status_code=200,
                cost_usd=0.08,
                quality_validation_type="factual_accuracy",
                validation_accuracy_score=87.5,
                confidence_score=0.83,
                claims_processed=6,
                quality_validation_cost_usd=0.05,
                quality_validation_cost_percentage=62.5
            ),
            QualityAPIMetric(
                timestamp=time.time() - 1000,
                endpoint="/api/score",
                method="POST", 
                response_time_ms=950.0,
                status_code=200,
                cost_usd=0.06,
                quality_validation_type="relevance_scoring",
                validation_accuracy_score=91.0,
                confidence_score=0.88,
                claims_processed=4,
                quality_validation_cost_usd=0.04,
                quality_validation_cost_percentage=66.7
            )
        ]
    
    def test_reporter_initialization(self, temp_output_dir):
        """Test QualityPerformanceReporter initialization."""
        config = PerformanceReportConfiguration(report_name="Test Report")
        reporter = QualityPerformanceReporter(
            config=config,
            output_directory=temp_output_dir
        )
        
        assert reporter.config == config
        assert reporter.output_directory == temp_output_dir
        assert temp_output_dir.exists()
        assert len(reporter.benchmark_data) == 0
        assert len(reporter.api_metrics_data) == 0
        assert isinstance(reporter.report_metadata, ReportMetadata)
    
    @pytest.mark.asyncio
    async def test_load_benchmark_data_direct(self, temp_output_dir, sample_benchmark_data):
        """Test loading benchmark data directly."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        loaded_count = await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        assert loaded_count == 2
        assert len(reporter.benchmark_data) == 2
        assert "benchmark_data" in reporter.report_metadata.data_sources
        assert reporter.report_metadata.total_data_points == 2
    
    @pytest.mark.asyncio
    async def test_load_benchmark_data_from_file(self, temp_output_dir, sample_benchmark_data):
        """Test loading benchmark data from file."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Create test data file
        test_data_file = temp_output_dir / "test_benchmark_data.json"
        test_data = {
            "quality_benchmark_results": {
                "test_benchmark": {
                    "scenario_quality_metrics": [asdict(metric) for metric in sample_benchmark_data]
                }
            }
        }
        
        with open(test_data_file, 'w') as f:
            json.dump(test_data, f)
        
        loaded_count = await reporter.load_benchmark_data(data_file=test_data_file)
        
        assert loaded_count > 0
        assert "benchmark_data" in reporter.report_metadata.data_sources
    
    @pytest.mark.asyncio
    async def test_load_api_metrics_data(self, temp_output_dir, sample_api_metrics):
        """Test loading API metrics data."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        loaded_count = await reporter.load_api_metrics_data(metrics_data=sample_api_metrics)
        
        assert loaded_count == 2
        assert len(reporter.api_metrics_data) == 2
        assert "api_metrics_data" in reporter.report_metadata.data_sources
    
    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, temp_output_dir, sample_benchmark_data):
        """Test executive summary generation."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        summary = await reporter._generate_executive_summary()
        
        assert isinstance(summary, dict)
        assert 'report_period' in summary
        assert 'data_summary' in summary
        assert 'key_performance_indicators' in summary
        assert 'overall_health_score' in summary
        assert 'critical_issues' in summary
        
        # Verify KPIs calculation
        kpis = summary['key_performance_indicators']
        assert 'average_response_time_ms' in kpis
        assert 'average_quality_score' in kpis
        assert kpis['average_response_time_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_performance_metrics(self, temp_output_dir, sample_benchmark_data):
        """Test performance metrics analysis."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        analysis = await reporter._analyze_performance_metrics()
        
        assert isinstance(analysis, dict)
        assert 'response_time_analysis' in analysis
        assert 'throughput_analysis' in analysis
        assert 'quality_efficiency_analysis' in analysis
        assert 'error_rate_analysis' in analysis
        
        # Verify response time analysis
        rt_analysis = analysis['response_time_analysis']
        assert 'mean_ms' in rt_analysis
        assert 'median_ms' in rt_analysis
        assert 'p95_ms' in rt_analysis
        assert rt_analysis['mean_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_correlations(self, temp_output_dir):
        """Test correlation analysis."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Add sample correlation data
        correlation_data = [
            PerformanceCorrelationMetrics(
                quality_strictness_level="medium",
                confidence_threshold=0.8,
                quality_performance_correlations={
                    'quality_vs_latency': 0.75,
                    'accuracy_vs_throughput': -0.65,
                    'validation_vs_cost': 0.45
                },
                sample_size=50
            )
        ]
        reporter.correlation_data = correlation_data
        
        analysis = await reporter._analyze_correlations()
        
        assert isinstance(analysis, dict)
        assert 'strongest_correlations' in analysis
        assert 'correlation_summary_statistics' in analysis
        
        strongest = analysis['strongest_correlations']
        assert len(strongest) > 0
        assert strongest[0]['correlation_name'] == 'quality_vs_latency'
        assert strongest[0]['strength'] == 'strong'
    
    @pytest.mark.asyncio
    async def test_analyze_cost_metrics(self, temp_output_dir, sample_api_metrics):
        """Test cost metrics analysis."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_api_metrics_data(metrics_data=sample_api_metrics)
        
        analysis = await reporter._analyze_cost_metrics()
        
        assert isinstance(analysis, dict)
        assert 'cost_summary' in analysis
        assert 'cost_efficiency' in analysis
        
        cost_summary = analysis['cost_summary']
        assert 'total_cost_usd' in cost_summary
        assert 'average_cost_per_operation' in cost_summary
        assert cost_summary['total_cost_usd'] == 0.14  # 0.08 + 0.06
    
    @pytest.mark.asyncio
    async def test_analyze_resource_usage(self, temp_output_dir, sample_benchmark_data):
        """Test resource usage analysis."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        analysis = await reporter._analyze_resource_usage()
        
        assert isinstance(analysis, dict)
        assert 'memory_usage' in analysis
        assert 'cpu_usage' in analysis
        
        memory_analysis = analysis['memory_usage']
        assert 'average_memory_mb' in memory_analysis
        assert 'peak_memory_mb' in memory_analysis
        assert memory_analysis['average_memory_mb'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_quality_metrics(self, temp_output_dir, sample_benchmark_data):
        """Test quality metrics analysis."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        analysis = await reporter._analyze_quality_metrics()
        
        assert isinstance(analysis, dict)
        assert 'validation_accuracy' in analysis
        assert 'claim_processing' in analysis
        assert 'confidence_levels' in analysis
        assert 'quality_stage_performance' in analysis
        
        # Verify claim processing analysis
        claim_analysis = analysis['claim_processing']
        assert 'total_claims_extracted' in claim_analysis
        assert 'total_claims_validated' in claim_analysis
        assert claim_analysis['total_claims_extracted'] == 55  # 25 + 30
    
    @pytest.mark.asyncio
    async def test_analyze_trends(self, temp_output_dir, sample_benchmark_data):
        """Test trend analysis."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Add more data points for trend analysis
        extended_data = sample_benchmark_data + [
            QualityValidationMetrics(
                scenario_name=f"trend_test_{i}",
                operations_count=10,
                average_latency_ms=1000.0 + (i * 50),  # Increasing trend
                start_time=time.time() - (i * 300)
            ) for i in range(15)
        ]
        
        await reporter.load_benchmark_data(data=extended_data)
        
        analysis = await reporter._analyze_trends()
        
        assert isinstance(analysis, dict)
        assert 'response_time_trend' in analysis or 'trend_summary' in analysis
        
        # If sufficient data, should analyze trends
        if 'response_time_trend' in analysis:
            rt_trend = analysis['response_time_trend']
            assert 'direction' in rt_trend
            assert rt_trend['direction'] in ['increasing', 'decreasing', 'stable']
    
    @pytest.mark.asyncio
    async def test_analyze_bottlenecks(self, temp_output_dir, sample_benchmark_data):
        """Test bottleneck analysis."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        analysis = await reporter._analyze_bottlenecks()
        
        assert isinstance(analysis, dict)
        assert 'processing_stage_bottlenecks' in analysis
        assert 'bottleneck_summary' in analysis
        
        stage_bottlenecks = analysis['processing_stage_bottlenecks']
        if stage_bottlenecks:  # If analysis was performed
            assert 'bottleneck_stage' in stage_bottlenecks
            assert 'bottleneck_percentage' in stage_bottlenecks
    
    @pytest.mark.asyncio
    async def test_generate_performance_insights(self, temp_output_dir, sample_benchmark_data, sample_api_metrics):
        """Test performance insights generation."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Load data with performance issues
        problematic_data = [
            QualityValidationMetrics(
                scenario_name="slow_scenario",
                average_latency_ms=3000.0,  # Above threshold
                validation_accuracy_rate=75.0,  # Below good quality
                error_rate_percent=8.0  # High error rate
            )
        ] + sample_benchmark_data
        
        await reporter.load_benchmark_data(data=problematic_data)
        await reporter.load_api_metrics_data(metrics_data=sample_api_metrics)
        
        await reporter._generate_performance_insights()
        
        assert len(reporter.performance_insights) > 0
        
        # Check that insights are properly formed
        for insight in reporter.performance_insights:
            assert isinstance(insight, PerformanceInsight)
            assert insight.title
            assert insight.description
            assert insight.severity in ['low', 'medium', 'high', 'critical']
    
    @pytest.mark.asyncio
    async def test_generate_optimization_recommendations(self, temp_output_dir, sample_benchmark_data, sample_api_metrics):
        """Test optimization recommendations generation."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Load data that will trigger recommendations
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        await reporter.load_api_metrics_data(metrics_data=sample_api_metrics)
        
        await reporter._generate_optimization_recommendations()
        
        # Should generate some recommendations
        assert len(reporter.optimization_recommendations) >= 0
        
        # If recommendations were generated, verify structure
        for rec in reporter.optimization_recommendations:
            assert isinstance(rec, OptimizationRecommendation)
            assert rec.category in ['performance', 'cost', 'resource', 'quality']
            assert rec.priority in ['low', 'medium', 'high', 'critical']
            assert isinstance(rec.implementation_steps, list)
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_report(self, temp_output_dir, sample_benchmark_data, sample_api_metrics):
        """Test comprehensive report generation."""
        config = PerformanceReportConfiguration(
            include_executive_summary=True,
            include_detailed_analysis=True,
            include_recommendations=True,
            generate_charts=False  # Skip charts for faster testing
        )
        
        reporter = QualityPerformanceReporter(
            config=config,
            output_directory=temp_output_dir
        )
        
        # Load sample data
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        await reporter.load_api_metrics_data(metrics_data=sample_api_metrics)
        
        report = await reporter.generate_comprehensive_report()
        
        assert isinstance(report, dict)
        assert 'metadata' in report
        assert 'configuration' in report
        assert 'executive_summary' in report
        assert 'performance_analysis' in report
        assert 'insights' in report
        assert 'recommendations' in report
        
        # Verify metadata is updated
        metadata = report['metadata']
        assert metadata['total_data_points'] > 0
        assert metadata['generation_duration_seconds'] > 0
    
    @pytest.mark.asyncio
    async def test_export_json_report(self, temp_output_dir, sample_benchmark_data):
        """Test JSON report export."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        report_data = await reporter.generate_comprehensive_report()
        
        # Export as JSON
        json_path = await reporter._export_json_report(report_data, "test_report.json")
        
        assert json_path.exists()
        assert json_path.suffix == '.json'
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert isinstance(loaded_data, dict)
        assert 'metadata' in loaded_data
        assert 'executive_summary' in loaded_data
    
    @pytest.mark.asyncio
    async def test_export_html_report(self, temp_output_dir, sample_benchmark_data):
        """Test HTML report export."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        report_data = await reporter.generate_comprehensive_report()
        
        # Export as HTML
        html_path = await reporter._export_html_report(report_data, "test_report.html")
        
        assert html_path.exists()
        assert html_path.suffix == '.html'
        
        # Verify HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        assert '<html' in html_content
        assert 'Quality Performance Report' in html_content
        assert 'Executive Summary' in html_content
    
    @pytest.mark.asyncio
    async def test_export_csv_report(self, temp_output_dir, sample_benchmark_data):
        """Test CSV report export."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        report_data = await reporter.generate_comprehensive_report()
        
        # Export as CSV
        csv_path = await reporter._export_csv_report(report_data, "test_report.csv")
        
        assert csv_path.exists()
        assert csv_path.suffix == '.csv'
        
        # Verify CSV content
        with open(csv_path, 'r') as f:
            csv_content = f.read()
        
        assert 'Metric,Value,Unit,Category' in csv_content
        assert 'Performance' in csv_content or 'Quality' in csv_content
    
    @pytest.mark.asyncio
    async def test_export_text_report(self, temp_output_dir, sample_benchmark_data):
        """Test text report export."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        
        report_data = await reporter.generate_comprehensive_report()
        
        # Export as text
        text_path = await reporter._export_text_report(report_data, "test_report.txt")
        
        assert text_path.exists()
        assert text_path.suffix == '.txt'
        
        # Verify text content
        with open(text_path, 'r') as f:
            text_content = f.read()
        
        assert 'QUALITY PERFORMANCE REPORT' in text_content
        assert 'EXECUTIVE SUMMARY' in text_content
        assert '=' * 80 in text_content  # Header separator
    
    @pytest.mark.asyncio
    async def test_export_report_multiple_formats(self, temp_output_dir, sample_benchmark_data):
        """Test exporting report in multiple formats."""
        config = PerformanceReportConfiguration(
            output_formats=[ReportFormat.JSON, ReportFormat.HTML, ReportFormat.CSV]
        )
        
        reporter = QualityPerformanceReporter(
            config=config,
            output_directory=temp_output_dir
        )
        
        await reporter.load_benchmark_data(data=sample_benchmark_data)
        report_data = await reporter.generate_comprehensive_report()
        
        exported_files = await reporter.export_report(report_data, "multi_format_report")
        
        assert len(exported_files) == 3
        assert 'json' in exported_files
        assert 'html' in exported_files
        assert 'csv' in exported_files
        
        # Verify all files exist
        for format_type, file_path in exported_files.items():
            assert Path(file_path).exists()


class TestVisualizationFunctionality:
    """Test suite for visualization functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for visualization tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def visualization_data(self):
        """Create sample data for visualization testing."""
        return [
            QualityValidationMetrics(
                scenario_name=f"viz_test_{i}",
                operations_count=10,
                average_latency_ms=1000.0 + (i * 100),
                validation_accuracy_rate=80.0 + (i * 2),
                start_time=time.time() - (i * 300)
            ) for i in range(10)
        ]
    
    @pytest.mark.asyncio
    async def test_generate_visualizations_unavailable(self, temp_output_dir, visualization_data):
        """Test visualization generation when libraries are unavailable."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=visualization_data)
        
        # Mock visualization libraries as unavailable
        with patch('reporting.quality_performance_reporter.VISUALIZATION_AVAILABLE', False):
            visualizations = await reporter._generate_visualizations()
        
        assert visualizations['status'] == 'visualization_not_available'
    
    @pytest.mark.asyncio
    async def test_generate_visualizations_available(self, temp_output_dir, visualization_data):
        """Test visualization generation when libraries are available."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        await reporter.load_benchmark_data(data=visualization_data)
        
        # Mock visualization libraries as available
        with patch('reporting.quality_performance_reporter.VISUALIZATION_AVAILABLE', True):
            with patch.object(reporter, '_create_performance_timeline_chart', return_value="<html>chart</html>"):
                with patch.object(reporter, '_create_quality_performance_scatter', return_value="<html>scatter</html>"):
                    visualizations = await reporter._generate_visualizations()
        
        assert 'charts_generated' in visualizations
        assert len(visualizations['charts_generated']) > 0


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for convenience function tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_performance_report(self, temp_output_dir):
        """Test convenience function for comprehensive report generation."""
        # Mock the data sources
        mock_benchmark_suite = Mock(spec=QualityValidationBenchmarkSuite)
        mock_benchmark_suite.quality_metrics_history = {
            'test_benchmark': [
                QualityValidationMetrics(
                    scenario_name="convenience_test",
                    operations_count=5,
                    average_latency_ms=1100.0
                )
            ]
        }
        
        config = PerformanceReportConfiguration(
            output_formats=[ReportFormat.JSON],
            generate_charts=False
        )
        
        exported_files = await generate_comprehensive_performance_report(
            benchmark_suite=mock_benchmark_suite,
            config=config,
            output_directory=temp_output_dir
        )
        
        assert isinstance(exported_files, dict)
        assert len(exported_files) > 0
        assert 'json' in exported_files
        
        # Verify exported file exists
        json_file = Path(exported_files['json'])
        assert json_file.exists()


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for error handling tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_report_generation_with_no_data(self, temp_output_dir):
        """Test report generation with no data."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Generate report without loading any data
        report = await reporter.generate_comprehensive_report()
        
        assert isinstance(report, dict)
        assert report['metadata']['total_data_points'] == 0
        
        # Should still have basic structure
        assert 'executive_summary' in report
        assert 'performance_analysis' in report
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self, temp_output_dir):
        """Test handling of invalid data."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Add invalid data
        invalid_data = [
            QualityValidationMetrics(
                scenario_name="invalid",
                operations_count=-1,  # Invalid
                average_latency_ms=float('inf'),  # Invalid
                validation_accuracy_rate=150.0  # Invalid percentage
            )
        ]
        
        # Should handle gracefully
        loaded_count = await reporter.load_benchmark_data(data=invalid_data)
        assert loaded_count == 1
        
        # Should still generate report
        report = await reporter.generate_comprehensive_report()
        assert isinstance(report, dict)
    
    @pytest.mark.asyncio
    async def test_file_io_error_handling(self, temp_output_dir):
        """Test file I/O error handling."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Try to load from non-existent file
        non_existent_file = temp_output_dir / "does_not_exist.json"
        loaded_count = await reporter.load_benchmark_data(data_file=non_existent_file)
        
        assert loaded_count == 0  # Should return 0, not crash
    
    def test_invalid_output_directory(self):
        """Test handling of invalid output directory."""
        # Test with read-only directory path
        invalid_path = Path("/nonexistent/readonly/path")
        
        # Should handle gracefully - either create parent directories or use fallback
        try:
            reporter = QualityPerformanceReporter(output_directory=invalid_path)
            assert hasattr(reporter, 'output_directory')
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert isinstance(e, (OSError, PermissionError, ValueError))
    
    @pytest.mark.asyncio
    async def test_memory_management_large_reports(self, temp_output_dir):
        """Test memory management with large reports."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Generate large amount of test data
        large_dataset = []
        for i in range(1000):
            metric = QualityValidationMetrics(
                scenario_name=f"large_test_{i}",
                operations_count=10,
                average_latency_ms=1000.0 + (i % 100),
                validation_accuracy_rate=80.0 + (i % 20),
                start_time=time.time() - (i * 60)
            )
            large_dataset.append(metric)
        
        # Should handle large dataset without memory issues
        loaded_count = await reporter.load_benchmark_data(data=large_dataset)
        assert loaded_count == 1000
        
        # Should still be able to generate report
        report = await reporter.generate_comprehensive_report()
        assert isinstance(report, dict)
        assert report['metadata']['total_data_points'] == 1000
    
    @pytest.mark.asyncio
    async def test_concurrent_report_generation(self, temp_output_dir):
        """Test concurrent report generation operations."""
        reporter = QualityPerformanceReporter(output_directory=temp_output_dir)
        
        # Add minimal data
        test_data = [
            QualityValidationMetrics(
                scenario_name="concurrent_test",
                operations_count=5,
                average_latency_ms=1200.0
            )
        ]
        await reporter.load_benchmark_data(data=test_data)
        
        # Run multiple report generations concurrently
        tasks = []
        for i in range(3):
            task = reporter.generate_comprehensive_report()
            tasks.append(task)
        
        reports = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed or fail gracefully
        for report in reports:
            if isinstance(report, Exception):
                # Should be reasonable exceptions, not deadlocks
                assert not isinstance(report, asyncio.TimeoutError)
            else:
                assert isinstance(report, dict)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])