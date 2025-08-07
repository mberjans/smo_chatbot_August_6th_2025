#!/usr/bin/env python3
"""
Performance Benchmarking Reporting Module for Clinical Metabolomics Oracle.

This module provides comprehensive reporting capabilities for quality validation
performance benchmarking data, including dashboard generation, statistical analysis,
and actionable recommendations for optimization.

Key Components:
    - QualityPerformanceReporter: Main reporting engine
    - PerformanceDashboard: Interactive dashboard generation
    - RecommendationEngine: Performance optimization insights
    - StatisticalAnalyzer: Statistical analysis and trend visualization
    - MultiFormatReporter: Reports in multiple formats (JSON, HTML, CSV, text)

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

from .quality_performance_reporter import (
    QualityPerformanceReporter,
    PerformanceReportConfiguration,
    ReportMetadata,
    PerformanceInsight,
    OptimizationRecommendation
)
from .performance_dashboard import PerformanceDashboard, DashboardConfiguration
from .recommendation_engine import RecommendationEngine, RecommendationType
from .statistical_analyzer import StatisticalAnalyzer, TrendAnalysis

__version__ = "1.0.0"

__all__ = [
    'QualityPerformanceReporter',
    'PerformanceReportConfiguration', 
    'ReportMetadata',
    'PerformanceInsight',
    'OptimizationRecommendation',
    'PerformanceDashboard',
    'DashboardConfiguration',
    'RecommendationEngine',
    'RecommendationType',
    'StatisticalAnalyzer',
    'TrendAnalysis'
]