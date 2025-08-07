#!/usr/bin/env python3
"""
Unit tests for Performance Correlation Engine module.

This module provides comprehensive testing for the CrossSystemCorrelationEngine
and related components, including correlation analysis, prediction models,
and optimization recommendations.

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import asyncio
import time
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import asdict

# Import the modules under test
from performance_correlation_engine import (
    PerformanceCorrelationMetrics,
    QualityPerformanceCorrelation,
    PerformancePredictionModel,
    CrossSystemCorrelationEngine,
    CorrelationAnalysisReport,
    create_correlation_engine,
    analyze_system_correlations
)

# Import dependencies for test data
try:
    from quality_performance_benchmarks import QualityValidationMetrics
    from quality_aware_metrics_logger import QualityAPIMetric
    from ..api_metrics_logger import APIMetric
    from ..tests.performance_test_fixtures import PerformanceMetrics, ResourceUsageSnapshot
except ImportError:
    # Create mock classes for testing
    class QualityValidationMetrics:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class QualityAPIMetric:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class APIMetric:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class PerformanceMetrics:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class ResourceUsageSnapshot:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


class TestPerformanceCorrelationMetrics:
    """Test suite for PerformanceCorrelationMetrics class."""
    
    def test_metrics_initialization(self):
        """Test proper initialization of PerformanceCorrelationMetrics."""
        metrics = PerformanceCorrelationMetrics(
            quality_strictness_level="strict",
            confidence_threshold=0.8
        )
        
        assert metrics.quality_strictness_level == "strict"
        assert metrics.confidence_threshold == 0.8
        assert isinstance(metrics.quality_performance_correlations, dict)
        assert isinstance(metrics.resource_quality_correlations, dict)
        assert metrics.sample_size == 0
    
    def test_calculate_overall_correlation_strength(self):
        """Test overall correlation strength calculation."""
        metrics = PerformanceCorrelationMetrics()
        metrics.quality_performance_correlations = {
            'quality_vs_latency': 0.8,
            'accuracy_vs_throughput': -0.6,
            'validation_vs_cost': 0.4,
            'confidence_vs_memory': 0.2
        }
        
        strength = metrics.calculate_overall_correlation_strength()
        expected = (0.8 + 0.6 + 0.4 + 0.2) / 4  # Mean of absolute values
        
        assert abs(strength - expected) < 0.001
    
    def test_get_strongest_correlations(self):
        """Test retrieval of strongest correlations."""
        metrics = PerformanceCorrelationMetrics()
        metrics.quality_performance_correlations = {
            'weak_correlation': 0.1,
            'strong_positive': 0.9,
            'strong_negative': -0.8,
            'moderate_correlation': 0.5,
            'very_weak': -0.05
        }
        
        strongest = metrics.get_strongest_correlations(top_n=3)
        
        assert len(strongest) == 3
        assert strongest[0][0] == 'strong_positive'
        assert strongest[0][1] == 0.9
        assert strongest[1][0] == 'strong_negative'
        assert abs(strongest[1][1] + 0.8) < 0.001


class TestQualityPerformanceCorrelation:
    """Test suite for QualityPerformanceCorrelation class."""
    
    def test_correlation_initialization(self):
        """Test proper initialization of QualityPerformanceCorrelation."""
        correlation = QualityPerformanceCorrelation(
            quality_metric_name="validation_accuracy",
            performance_metric_name="response_time",
            correlation_coefficient=0.75,
            confidence_interval=(0.6, 0.9),
            p_value=0.01,
            sample_size=100,
            correlation_type="positive",
            strength="strong"
        )
        
        assert correlation.quality_metric_name == "validation_accuracy"
        assert correlation.performance_metric_name == "response_time"
        assert correlation.correlation_coefficient == 0.75
        assert correlation.correlation_type == "positive"
        assert correlation.strength == "strong"
    
    def test_is_statistically_significant(self):
        """Test statistical significance detection."""
        # Significant correlation
        significant_corr = QualityPerformanceCorrelation(
            quality_metric_name="test",
            performance_metric_name="test",
            correlation_coefficient=0.8,
            confidence_interval=(0.7, 0.9),
            p_value=0.001,  # Very significant
            sample_size=100,
            correlation_type="positive",
            strength="strong"
        )
        
        assert significant_corr.is_statistically_significant is True
        
        # Non-significant correlation
        non_significant_corr = QualityPerformanceCorrelation(
            quality_metric_name="test",
            performance_metric_name="test",
            correlation_coefficient=0.2,
            confidence_interval=(0.1, 0.3),
            p_value=0.15,  # Not significant
            sample_size=50,
            correlation_type="weak",
            strength="weak"
        )
        
        assert non_significant_corr.is_statistically_significant is False
    
    def test_correlation_description(self):
        """Test human-readable correlation description."""
        correlation = QualityPerformanceCorrelation(
            quality_metric_name="validation_accuracy",
            performance_metric_name="response_time",
            correlation_coefficient=0.75,
            confidence_interval=(0.6, 0.9),
            p_value=0.01,
            sample_size=100,
            correlation_type="positive",
            strength="strong"
        )
        
        description = correlation.correlation_description
        
        assert "validation_accuracy" in description
        assert "response_time" in description
        assert "increases" in description  # Positive correlation
        assert "strong" in description


class TestPerformancePredictionModel:
    """Test suite for PerformancePredictionModel class."""
    
    def test_model_initialization(self):
        """Test proper initialization of prediction models."""
        # Test different model types
        linear_model = PerformancePredictionModel(model_type="linear")
        assert linear_model.model_type == "linear"
        assert linear_model.is_trained is False
        
        rf_model = PerformancePredictionModel(model_type="random_forest")
        assert rf_model.model_type == "random_forest"
        
        # Test invalid model type
        with pytest.raises(ValueError):
            PerformancePredictionModel(model_type="invalid_model")
    
    def test_model_training_success(self):
        """Test successful model training."""
        model = PerformancePredictionModel(model_type="linear")
        
        # Create sample training data
        quality_features = [
            [80.0, 5.0, 85.0, 1000.0, 3000.0, 500.0, 82.0],
            [85.0, 4.5, 90.0, 1200.0, 3200.0, 550.0, 85.0],
            [78.0, 6.0, 80.0, 900.0, 2800.0, 480.0, 78.5],
            [90.0, 4.0, 95.0, 1100.0, 3100.0, 520.0, 88.0],
            [82.0, 5.5, 87.0, 1050.0, 2900.0, 490.0, 83.5]
        ]
        performance_targets = [2.5, 2.2, 2.8, 2.0, 2.6]  # Response times
        feature_names = [
            "validation_accuracy", "claims_per_second", "validation_confidence",
            "extraction_time", "validation_time", "scoring_time", "efficiency_score"
        ]
        
        # Train model
        results = model.train(quality_features, performance_targets, feature_names)
        
        assert results["training_successful"] is True
        assert model.is_trained is True
        assert "r2_score" in results
        assert "mean_absolute_error" in results
        assert model.feature_importance is not None
    
    def test_model_training_failure(self):
        """Test model training with invalid data."""
        model = PerformancePredictionModel(model_type="linear")
        
        # Test with mismatched feature and target lengths
        quality_features = [[1, 2, 3], [4, 5, 6]]
        performance_targets = [1]  # Wrong length
        
        with pytest.raises(ValueError):
            model.train(quality_features, performance_targets)
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        model = PerformancePredictionModel(model_type="linear")
        
        # Train model first
        quality_features = [
            [80.0, 5.0, 85.0, 1000.0],
            [85.0, 4.5, 90.0, 1200.0],
            [78.0, 6.0, 80.0, 900.0],
            [90.0, 4.0, 95.0, 1100.0]
        ]
        performance_targets = [2.5, 2.2, 2.8, 2.0]
        
        model.train(quality_features, performance_targets)
        
        # Test prediction
        test_features = [82.0, 5.2, 86.0, 1050.0]
        prediction, confidence = model.predict(test_features)
        
        assert isinstance(prediction, float)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_model_prediction_without_training(self):
        """Test prediction failure when model is not trained."""
        model = PerformancePredictionModel(model_type="linear")
        
        with pytest.raises(ValueError):
            model.predict([1, 2, 3, 4])
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        model = PerformancePredictionModel(model_type="linear")
        
        # Train model
        quality_features = [[1, 2], [3, 4], [5, 6], [7, 8]]
        performance_targets = [1.0, 2.0, 3.0, 4.0]
        model.train(quality_features, performance_targets)
        
        # Test batch prediction
        batch_features = [[2, 3], [4, 5], [6, 7]]
        predictions = model.predict_batch(batch_features)
        
        assert len(predictions) == 3
        assert all(isinstance(pred, tuple) for pred in predictions)
        assert all(len(pred) == 2 for pred in predictions)  # (prediction, confidence)
    
    def test_model_save_and_load(self):
        """Test model saving and loading functionality."""
        model = PerformancePredictionModel(model_type="linear")
        
        # Train model
        quality_features = [[1, 2], [3, 4], [5, 6]]
        performance_targets = [1.0, 2.0, 3.0]
        model.train(quality_features, performance_targets)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            success = model.save_model(tmp_path)
            assert success is True
            
            # Load model into new instance
            new_model = PerformancePredictionModel(model_type="linear")
            load_success = new_model.load_model(tmp_path)
            
            assert load_success is True
            assert new_model.is_trained is True
            assert new_model.model_type == "linear"
            
            # Test that loaded model can make predictions
            prediction, confidence = new_model.predict([4, 5])
            assert isinstance(prediction, float)
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestCrossSystemCorrelationEngine:
    """Test suite for CrossSystemCorrelationEngine class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data for correlation analysis."""
        performance_data = [
            PerformanceMetrics(
                test_name="test1",
                operations_count=100,
                average_latency_ms=1200,
                throughput_ops_per_sec=5.0,
                memory_usage_mb=800,
                cpu_usage_percent=60
            ),
            PerformanceMetrics(
                test_name="test2",
                operations_count=150,
                average_latency_ms=1500,
                throughput_ops_per_sec=4.5,
                memory_usage_mb=900,
                cpu_usage_percent=70
            )
        ]
        
        quality_data = [
            QualityValidationMetrics(
                scenario_name="scenario1",
                validation_accuracy_rate=85.0,
                claims_per_second=6.0,
                avg_validation_confidence=80.0
            ),
            QualityValidationMetrics(
                scenario_name="scenario2", 
                validation_accuracy_rate=78.0,
                claims_per_second=7.5,
                avg_validation_confidence=75.0
            )
        ]
        
        api_metrics = [
            APIMetric(cost_usd=0.05, response_time_ms=1100),
            APIMetric(cost_usd=0.08, response_time_ms=1400)
        ]
        
        resource_snapshots = [
            ResourceUsageSnapshot(memory_mb=850, cpu_percent=65),
            ResourceUsageSnapshot(memory_mb=920, cpu_percent=72)
        ]
        
        return {
            'performance_data': performance_data,
            'quality_data': quality_data,
            'api_metrics': api_metrics,
            'resource_snapshots': resource_snapshots
        }
    
    def test_engine_initialization(self, temp_output_dir):
        """Test correlation engine initialization."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        assert engine.output_dir == temp_output_dir
        assert engine.output_dir.exists()
        assert hasattr(engine, 'performance_data')
        assert hasattr(engine, 'quality_data')
        assert hasattr(engine, 'api_metrics_data')
        assert hasattr(engine, 'correlation_history')
    
    def test_add_data_methods(self, temp_output_dir, sample_data):
        """Test data addition methods."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Test adding performance data
        for perf_data in sample_data['performance_data']:
            engine.add_performance_data(perf_data)
        assert len(engine.performance_data) == 2
        
        # Test adding quality data
        for quality_data in sample_data['quality_data']:
            engine.add_quality_data(quality_data)
        assert len(engine.quality_data) == 2
        
        # Test adding API metrics
        engine.add_api_metrics(sample_data['api_metrics'])
        assert len(engine.api_metrics_data) == 2
        
        # Test adding resource snapshots
        engine.add_resource_snapshots(sample_data['resource_snapshots'])
        assert len(engine.resource_snapshots) == 2
    
    @pytest.mark.asyncio
    async def test_analyze_quality_performance_correlation(self, temp_output_dir, sample_data):
        """Test quality-performance correlation analysis."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add sample data
        for perf_data in sample_data['performance_data']:
            engine.add_performance_data(perf_data)
        for quality_data in sample_data['quality_data']:
            engine.add_quality_data(quality_data)
        
        # Perform correlation analysis
        correlation_metrics = await engine.analyze_quality_performance_correlation()
        
        assert isinstance(correlation_metrics, PerformanceCorrelationMetrics)
        assert correlation_metrics.sample_size > 0
        assert isinstance(correlation_metrics.quality_performance_correlations, dict)
    
    @pytest.mark.asyncio
    async def test_analyze_correlation_insufficient_data(self, temp_output_dir):
        """Test correlation analysis with insufficient data."""
        engine = CrossSystemCorrelationEngine(
            output_dir=temp_output_dir,
            config={'min_sample_size': 10}
        )
        
        # Add minimal data (below threshold)
        engine.add_performance_data(PerformanceMetrics(test_name="test1"))
        
        correlation_metrics = await engine.analyze_quality_performance_correlation()
        
        assert isinstance(correlation_metrics, PerformanceCorrelationMetrics)
        assert correlation_metrics.sample_size < engine.min_sample_size
    
    @pytest.mark.asyncio
    async def test_predict_performance_impact(self, temp_output_dir, sample_data):
        """Test performance impact prediction."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add training data
        for perf_data in sample_data['performance_data']:
            engine.add_performance_data(perf_data)
        for quality_data in sample_data['quality_data']:
            engine.add_quality_data(quality_data)
        
        # Test prediction
        quality_requirements = {
            'confidence_threshold': 0.8,
            'accuracy_requirement': 85.0,
            'validation_strictness_level': 1.0
        }
        
        predictions = await engine.predict_performance_impact(
            quality_requirements, 
            target_metrics=['response_time_ms', 'throughput_ops_per_sec']
        )
        
        assert isinstance(predictions, dict)
        assert 'response_time_ms' in predictions
        assert 'throughput_ops_per_sec' in predictions
        
        for metric, (value, confidence) in predictions.items():
            assert isinstance(value, float)
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_optimize_resource_allocation(self, temp_output_dir, sample_data):
        """Test resource allocation optimization."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add sample data
        for perf_data in sample_data['performance_data']:
            engine.add_performance_data(perf_data)
        for quality_data in sample_data['quality_data']:
            engine.add_quality_data(quality_data)
        
        quality_targets = {'confidence_threshold': 0.85, 'accuracy_requirement': 90.0}
        performance_constraints = {'response_time_ms': 2000, 'memory_usage_mb': 1000}
        cost_budget = 100.0
        
        optimization_results = await engine.optimize_resource_allocation(
            quality_targets, performance_constraints, cost_budget
        )
        
        assert isinstance(optimization_results, dict)
        assert 'recommendations' in optimization_results
        assert 'trade_offs' in optimization_results
        assert 'pareto_solutions' in optimization_results
        assert 'feasibility_analysis' in optimization_results
    
    @pytest.mark.asyncio
    async def test_generate_correlation_report(self, temp_output_dir, sample_data):
        """Test comprehensive correlation report generation."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add sample data
        for perf_data in sample_data['performance_data']:
            engine.add_performance_data(perf_data)
        for quality_data in sample_data['quality_data']:
            engine.add_quality_data(quality_data)
        
        report = await engine.generate_correlation_report(
            include_predictions=True,
            include_recommendations=True
        )
        
        assert isinstance(report, CorrelationAnalysisReport)
        assert report.correlation_metrics is not None
        assert len(report.resource_allocation_recommendations) >= 0
        assert len(report.performance_optimization_suggestions) >= 0
    
    @pytest.mark.asyncio
    async def test_save_correlation_report(self, temp_output_dir, sample_data):
        """Test correlation report saving functionality."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add sample data and generate report
        for perf_data in sample_data['performance_data']:
            engine.add_performance_data(perf_data)
        
        report = await engine.generate_correlation_report()
        
        # Save report
        report_path = await engine.save_correlation_report(report, include_raw_data=True)
        
        assert isinstance(report_path, Path)
        assert report_path.exists()
        assert report_path.suffix == '.json'
        
        # Verify report content
        with open(report_path, 'r') as f:
            saved_data = json.load(f)
        
        assert 'report_id' in saved_data
        assert 'generated_timestamp' in saved_data
        assert 'correlation_metrics' in saved_data
    
    def test_analyze_quality_validation_performance(self, temp_output_dir, sample_data):
        """Test quality validation performance analysis."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add quality data
        for quality_data in sample_data['quality_data']:
            engine.add_quality_data(quality_data)
        
        analysis = engine._analyze_quality_validation_performance()
        
        assert isinstance(analysis, dict)
        assert 'avg_validation_accuracy_rate' in analysis
        assert 'avg_claims_per_second' in analysis
        assert 'avg_efficiency_score' in analysis
    
    def test_analyze_api_usage_correlations(self, temp_output_dir, sample_data):
        """Test API usage correlation analysis."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add API metrics
        engine.add_api_metrics(sample_data['api_metrics'])
        
        analysis = engine._analyze_api_usage_correlations()
        
        assert isinstance(analysis, dict)
        assert 'avg_cost_usd' in analysis
        assert 'success_rate' in analysis
    
    def test_analyze_resource_utilization_correlations(self, temp_output_dir, sample_data):
        """Test resource utilization correlation analysis."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add resource snapshots
        engine.add_resource_snapshots(sample_data['resource_snapshots'])
        
        analysis = engine._analyze_resource_utilization_correlations()
        
        assert isinstance(analysis, dict)
        assert 'avg_cpu_percent' in analysis
        assert 'avg_memory_mb' in analysis
        assert 'max_memory_mb' in analysis


class TestCorrelationAnalysisReport:
    """Test suite for CorrelationAnalysisReport class."""
    
    def test_report_initialization(self):
        """Test report initialization."""
        report = CorrelationAnalysisReport()
        
        assert hasattr(report, 'report_id')
        assert hasattr(report, 'generated_timestamp')
        assert isinstance(report.quality_performance_correlations, list)
        assert isinstance(report.resource_allocation_recommendations, list)
    
    def test_get_executive_summary(self):
        """Test executive summary generation."""
        # Create report with sample data
        correlation_metrics = PerformanceCorrelationMetrics()
        correlation_metrics.quality_performance_correlations = {
            'quality_vs_latency': 0.8,
            'accuracy_vs_cost': -0.6
        }
        
        report = CorrelationAnalysisReport(
            correlation_metrics=correlation_metrics,
            prediction_accuracy=0.85
        )
        
        summary = report.get_executive_summary()
        
        assert isinstance(summary, dict)
        assert 'analysis_timestamp' in summary
        assert 'overall_correlation_strength' in summary
        assert 'prediction_accuracy' in summary
        assert summary['prediction_accuracy'] == 0.85
    
    def test_get_executive_summary_no_data(self):
        """Test executive summary with no data."""
        report = CorrelationAnalysisReport()
        summary = report.get_executive_summary()
        
        assert summary['status'] == 'no_data'


class TestIntegrationAndConvenienceFunctions:
    """Test suite for integration and convenience functions."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for integration tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_create_correlation_engine(self, temp_output_dir):
        """Test correlation engine creation convenience function."""
        engine = create_correlation_engine(output_dir=temp_output_dir)
        
        assert isinstance(engine, CrossSystemCorrelationEngine)
        assert engine.output_dir == temp_output_dir
    
    @pytest.mark.asyncio
    async def test_analyze_system_correlations(self, temp_output_dir):
        """Test system correlation analysis convenience function."""
        # Create sample data
        performance_data = [
            PerformanceMetrics(
                test_name="test1",
                operations_count=100,
                average_latency_ms=1200
            )
        ]
        
        quality_data = [
            QualityValidationMetrics(
                scenario_name="scenario1",
                validation_accuracy_rate=85.0
            )
        ]
        
        report = await analyze_system_correlations(
            performance_data=performance_data,
            quality_data=quality_data
        )
        
        assert isinstance(report, CorrelationAnalysisReport)
        assert report.correlation_metrics is not None


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for correlation engine."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for error handling tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_correlation_analysis_with_invalid_data(self, temp_output_dir):
        """Test correlation analysis with invalid or inconsistent data."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add data with inconsistent attributes
        invalid_performance = PerformanceMetrics(test_name="invalid")
        # Missing required attributes
        engine.add_performance_data(invalid_performance)
        
        # Should handle gracefully
        correlation_metrics = await engine.analyze_quality_performance_correlation()
        assert isinstance(correlation_metrics, PerformanceCorrelationMetrics)
    
    @pytest.mark.asyncio
    async def test_prediction_with_no_training_data(self, temp_output_dir):
        """Test prediction functionality with no training data."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        quality_requirements = {'confidence_threshold': 0.8}
        
        predictions = await engine.predict_performance_impact(quality_requirements)
        
        # Should return predictions even without training data (using fallback methods)
        assert isinstance(predictions, dict)
        for metric, (value, confidence) in predictions.items():
            assert isinstance(value, float)
            assert isinstance(confidence, float)
    
    def test_feature_extraction_edge_cases(self, temp_output_dir):
        """Test feature extraction with edge case inputs."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Test with missing keys
        incomplete_requirements = {'confidence_threshold': 0.7}
        features = engine._extract_quality_features(incomplete_requirements)
        
        assert isinstance(features, list)
        assert len(features) > 0  # Should use default values
        
        # Test with invalid values
        invalid_requirements = {
            'confidence_threshold': -1.0,  # Invalid negative
            'validation_strictness_level': 10.0,  # Out of range
            'max_claims_per_response': -5  # Invalid negative
        }
        features = engine._extract_quality_features(invalid_requirements)
        
        assert isinstance(features, list)
        # Should handle invalid values gracefully
    
    @pytest.mark.asyncio
    async def test_optimization_with_conflicting_constraints(self, temp_output_dir):
        """Test resource optimization with conflicting constraints."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add minimal data
        engine.add_performance_data(PerformanceMetrics(test_name="test"))
        
        # Set conflicting constraints (very high quality, very low resources)
        quality_targets = {'confidence_threshold': 0.99, 'accuracy_requirement': 99.0}
        performance_constraints = {'response_time_ms': 100, 'memory_usage_mb': 50}  # Very tight
        cost_budget = 0.01  # Very low budget
        
        optimization_results = await engine.optimize_resource_allocation(
            quality_targets, performance_constraints, cost_budget
        )
        
        assert isinstance(optimization_results, dict)
        assert 'feasibility_analysis' in optimization_results
        
        # Should identify infeasibility
        feasibility = optimization_results['feasibility_analysis']
        assert 'overall_feasible' in feasibility
    
    def test_memory_management_large_datasets(self, temp_output_dir):
        """Test memory management with large datasets."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add large amount of data
        for i in range(1000):
            performance_data = PerformanceMetrics(
                test_name=f"test_{i}",
                operations_count=100,
                average_latency_ms=1000 + i
            )
            engine.add_performance_data(performance_data)
        
        # Should handle large datasets without memory issues
        assert len(engine.performance_data) == 1000
        
        # Clear cache periodically
        engine._clear_analysis_cache()
        assert len(engine._analysis_cache) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_operations(self, temp_output_dir):
        """Test concurrent correlation analysis operations."""
        engine = CrossSystemCorrelationEngine(output_dir=temp_output_dir)
        
        # Add sample data
        engine.add_performance_data(PerformanceMetrics(test_name="test"))
        engine.add_quality_data(QualityValidationMetrics(scenario_name="scenario"))
        
        # Run multiple analyses concurrently
        tasks = [
            engine.analyze_quality_performance_correlation(),
            engine.predict_performance_impact({'confidence_threshold': 0.8}),
            engine.generate_correlation_report()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle concurrent operations
        for result in results:
            if isinstance(result, Exception):
                # Should be reasonable exceptions, not deadlocks
                assert not isinstance(result, asyncio.TimeoutError)
    
    def test_cache_functionality(self, temp_output_dir):
        """Test caching functionality and TTL."""
        engine = CrossSystemCorrelationEngine(
            output_dir=temp_output_dir,
            config={'cache_ttl': 1}  # 1 second TTL for testing
        )
        
        # Test cache miss
        cache_key = "test_cache_key"
        result = engine._get_cached_result(cache_key)
        assert result is None
        
        # Test cache hit
        test_data = {"test": "data"}
        engine._cache_result(cache_key, test_data)
        result = engine._get_cached_result(cache_key)
        assert result == test_data
        
        # Test cache expiry
        time.sleep(1.1)  # Wait for TTL to expire
        result = engine._get_cached_result(cache_key)
        assert result is None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])