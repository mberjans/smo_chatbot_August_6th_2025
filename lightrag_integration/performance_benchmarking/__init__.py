"""
Performance Benchmarking Module for Clinical Metabolomics Oracle LightRAG Integration.

This module provides specialized performance benchmarking utilities for quality validation
components in the Clinical Metabolomics Oracle system, extending the existing performance
monitoring infrastructure with quality-specific metrics and benchmarks.

Key Components:
    - QualityValidationBenchmarkSuite: Specialized benchmarks for quality validation
    - Quality-specific performance thresholds and metrics
    - Integration with existing PerformanceBenchmarkSuite infrastructure

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

from .quality_performance_benchmarks import (
    QualityValidationBenchmarkSuite,
    QualityValidationMetrics,
    QualityBenchmarkConfiguration,
    QualityPerformanceThreshold
)

__all__ = [
    'QualityValidationBenchmarkSuite',
    'QualityValidationMetrics', 
    'QualityBenchmarkConfiguration',
    'QualityPerformanceThreshold'
]

__version__ = '1.0.0'
__author__ = 'Claude Code (Anthropic)'