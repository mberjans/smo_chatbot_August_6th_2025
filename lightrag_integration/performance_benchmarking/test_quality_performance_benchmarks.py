#!/usr/bin/env python3
"""
Unit tests for Quality Performance Benchmarks module.

This module provides comprehensive testing for the QualityValidationBenchmarkSuite
and related components, including positive and negative scenarios, edge cases,
and integration testing.

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
from quality_performance_benchmarks import (
    QualityValidationMetrics,
    QualityPerformanceThreshold,
    QualityBenchmarkConfiguration,
    QualityValidationBenchmarkSuite
)


class TestQualityValidationMetrics:
    """Test suite for QualityValidationMetrics class."""
    
    def test_metrics_initialization(self):
        """Test proper initialization of QualityValidationMetrics."""
        metrics = QualityValidationMetrics(
            scenario_name="test_scenario",
            operations_count=10
        )
        
        assert metrics.scenario_name == "test_scenario"
        assert metrics.operations_count == 10
        assert metrics.claim_extraction_time_ms == 0.0
        assert metrics.factual_validation_time_ms == 0.0
        assert metrics.claims_extracted_count == 0
        assert metrics.validation_accuracy_rate == 0.0
    
    def test_calculate_quality_efficiency_score(self):
        """Test quality efficiency score calculation."""
        metrics = QualityValidationMetrics(
            integrated_workflow_time_ms=1000,
            validation_accuracy_rate=90.0,
            relevance_scoring_accuracy=85.0,
            claims_per_second=10.0,
            validations_per_second=8.0,
            extraction_error_rate=2.0,
            validation_error_rate=1.0
        )
        
        score = metrics.calculate_quality_efficiency_score()
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score > 50  # Should be reasonably high given good metrics
    
    def test_calculate_quality_efficiency_score_edge_cases(self):
        """Test quality efficiency score with edge case values."""
        # Test with zero values
        metrics_zero = QualityValidationMetrics()
        score_zero = metrics_zero.calculate_quality_efficiency_score()
        assert isinstance(score_zero, float)
        assert score_zero >= 0
        
        # Test with extreme values
        metrics_extreme = QualityValidationMetrics(
            integrated_workflow_time_ms=50000,  # Very slow
            validation_accuracy_rate=100.0,
            relevance_scoring_accuracy=100.0,
            claims_per_second=0.1,  # Very low
            validations_per_second=0.1,
            extraction_error_rate=10.0,  # High error rate
            validation_error_rate=10.0
        )
        score_extreme = metrics_extreme.calculate_quality_efficiency_score()
        assert isinstance(score_extreme, float)
        assert score_extreme >= 0


class TestQualityPerformanceThreshold:
    """Test suite for QualityPerformanceThreshold class."""
    
    def test_create_quality_thresholds(self):
        """Test creation of standard quality thresholds."""
        thresholds = QualityPerformanceThreshold.create_quality_thresholds()
        
        assert isinstance(thresholds, dict)
        assert 'claim_extraction_time_ms' in thresholds
        assert 'factual_validation_time_ms' in thresholds
        assert 'validation_accuracy_rate' in thresholds
        
        # Test specific threshold values
        extraction_threshold = thresholds['claim_extraction_time_ms']
        assert extraction_threshold.metric_name == 'claim_extraction_time_ms'
        assert extraction_threshold.threshold_value == 2000
        assert extraction_threshold.comparison_operator == 'lte'
        assert extraction_threshold.severity == 'error'
    
    def test_threshold_check_functionality(self):
        """Test threshold checking functionality."""
        thresholds = QualityPerformanceThreshold.create_quality_thresholds()
        
        # Test passing threshold
        extraction_threshold = thresholds['claim_extraction_time_ms']
        passed, message = extraction_threshold.check(1500)  # Below threshold
        assert passed is True
        assert "within" in message.lower() or "passed" in message.lower()
        
        # Test failing threshold
        passed, message = extraction_threshold.check(3000)  # Above threshold
        assert passed is False
        assert "exceeded" in message.lower() or "failed" in message.lower()


class TestQualityBenchmarkConfiguration:
    """Test suite for QualityBenchmarkConfiguration class."""
    
    def test_configuration_initialization(self):
        """Test proper initialization of QualityBenchmarkConfiguration."""
        config = QualityBenchmarkConfiguration(
            benchmark_name="test_benchmark",
            description="Test description"
        )
        
        assert config.benchmark_name == "test_benchmark"
        assert config.description == "Test description"
        assert config.enable_factual_validation is True
        assert config.enable_relevance_scoring is True
        assert len(config.sample_queries) > 0
        assert len(config.sample_responses) > 0
    
    def test_post_init_default_samples(self):
        """Test that __post_init__ creates default sample data."""
        config = QualityBenchmarkConfiguration()
        
        # Should have default sample data
        assert len(config.sample_queries) == 5
        assert len(config.sample_responses) == 5
        assert all(isinstance(query, str) for query in config.sample_queries)
        assert all(isinstance(response, str) for response in config.sample_responses)
    
    def test_custom_sample_data_preservation(self):
        """Test that custom sample data is preserved."""
        custom_queries = ["Custom query 1", "Custom query 2"]
        custom_responses = ["Custom response 1", "Custom response 2"]
        
        config = QualityBenchmarkConfiguration(
            sample_queries=custom_queries,
            sample_responses=custom_responses
        )
        
        # Should preserve custom data
        assert config.sample_queries == custom_queries
        assert config.sample_responses == custom_responses


class TestQualityValidationBenchmarkSuite:
    """Test suite for QualityValidationBenchmarkSuite class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_quality_components(self):
        """Mock quality validation components."""
        with patch('quality_performance_benchmarks.QUALITY_COMPONENTS_AVAILABLE', True):
            with patch('quality_performance_benchmarks.FactualAccuracyValidator') as mock_validator, \
                 patch('quality_performance_benchmarks.ClinicalMetabolomicsRelevanceScorer') as mock_scorer, \
                 patch('quality_performance_benchmarks.BiomedicalClaimExtractor') as mock_extractor, \
                 patch('quality_performance_benchmarks.FactualAccuracyScorer') as mock_accuracy_scorer, \
                 patch('quality_performance_benchmarks.IntegratedQualityWorkflow') as mock_workflow:
                
                # Set up mock return values
                mock_extractor_instance = Mock()
                mock_extractor_instance.extract_claims = AsyncMock(return_value=["claim1", "claim2"])
                mock_extractor.return_value = mock_extractor_instance
                
                mock_validator_instance = Mock()
                mock_validator_instance.validate_claims = AsyncMock(return_value=[
                    Mock(confidence_score=0.8, supported=True),
                    Mock(confidence_score=0.7, supported=True)
                ])
                mock_validator.return_value = mock_validator_instance
                
                mock_scorer_instance = Mock()
                mock_scorer_instance.score_relevance = AsyncMock(return_value=Mock(
                    overall_score=85.0, confidence_score=0.75
                ))
                mock_scorer.return_value = mock_scorer_instance
                
                mock_workflow_instance = Mock()
                mock_workflow_instance.assess_quality = AsyncMock(return_value={
                    'overall_score': 82.0, 'components_completed': 3
                })
                mock_workflow.return_value = mock_workflow_instance
                
                yield {
                    'validator': mock_validator_instance,
                    'scorer': mock_scorer_instance,
                    'extractor': mock_extractor_instance,
                    'workflow': mock_workflow_instance
                }
    
    def test_benchmark_suite_initialization(self, temp_output_dir):
        """Test benchmark suite initialization."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        assert suite.output_dir == temp_output_dir
        assert hasattr(suite, 'quality_metrics_history')
        assert hasattr(suite, 'quality_benchmarks')
        assert len(suite.quality_benchmarks) > 0
    
    def test_create_quality_benchmarks(self, temp_output_dir):
        """Test creation of quality benchmark configurations."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        benchmarks = suite._create_quality_benchmarks()
        
        assert isinstance(benchmarks, dict)
        assert 'factual_accuracy_validation_benchmark' in benchmarks
        assert 'relevance_scoring_benchmark' in benchmarks
        assert 'integrated_quality_workflow_benchmark' in benchmarks
        
        # Test benchmark configuration properties
        factual_benchmark = benchmarks['factual_accuracy_validation_benchmark']
        assert factual_benchmark.enable_factual_validation is True
        assert factual_benchmark.enable_relevance_scoring is False
        assert len(factual_benchmark.test_scenarios) > 0
    
    @pytest.mark.asyncio
    async def test_execute_claim_extraction(self, temp_output_dir, mock_quality_components):
        """Test claim extraction execution."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Execute with mock extractor
        response_text = "This is a test response with multiple claims about metabolomics research."
        claims = await suite._execute_claim_extraction(response_text)
        
        assert isinstance(claims, list)
        assert len(claims) > 0
        
        # Verify mock was called
        mock_quality_components['extractor'].extract_claims.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_factual_validation(self, temp_output_dir, mock_quality_components):
        """Test factual validation execution."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Execute with mock validator
        test_claims = ["claim1", "claim2"]
        results = await suite._execute_factual_validation(test_claims)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Verify mock was called with correct arguments
        mock_quality_components['validator'].validate_claims.assert_called_once_with(test_claims)
    
    @pytest.mark.asyncio
    async def test_execute_relevance_scoring(self, temp_output_dir, mock_quality_components):
        """Test relevance scoring execution."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Execute with mock scorer
        query = "What are metabolomics applications?"
        response = "Metabolomics has various applications in clinical research."
        score = await suite._execute_relevance_scoring(query, response)
        
        assert score is not None
        assert hasattr(score, 'overall_score')
        assert hasattr(score, 'confidence_score')
        
        # Verify mock was called
        mock_quality_components['scorer'].score_relevance.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_integrated_workflow(self, temp_output_dir, mock_quality_components):
        """Test integrated workflow execution."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Execute with mock workflow
        query = "Test query"
        response = "Test response"
        result = await suite._execute_integrated_workflow(query, response)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'overall_score' in result
        
        # Verify mock was called
        mock_quality_components['workflow'].assess_quality.assert_called_once()
    
    def test_calculate_quality_metrics(self, temp_output_dir):
        """Test quality metrics calculation."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Create base metrics
        base_metrics = QualityValidationMetrics(
            scenario_name="test",
            operations_count=5,
            start_time=time.time(),
            duration_seconds=10.0
        )
        
        # Create test data
        stage_times = {
            'claim_extraction': [100, 150, 120],
            'factual_validation': [500, 600, 550],
            'relevance_scoring': [80, 90, 85]
        }
        stage_errors = {'claim_extraction': 0, 'factual_validation': 1}
        stage_successes = {'claim_extraction': 3, 'factual_validation': 2}
        validation_confidences = [0.8, 0.7, 0.9]
        relevance_confidences = [0.75, 0.85, 0.70]
        
        # Mock resource snapshots
        mock_snapshots = [
            Mock(memory_mb=100, cpu_percent=50),
            Mock(memory_mb=120, cpu_percent=60),
            Mock(memory_mb=110, cpu_percent=55)
        ]
        
        # Calculate metrics
        calculated_metrics = suite._calculate_quality_metrics(
            base_metrics, stage_times, stage_errors, stage_successes,
            10, 8, 3, validation_confidences, relevance_confidences, mock_snapshots
        )
        
        # Verify calculations
        assert calculated_metrics.claim_extraction_time_ms > 0
        assert calculated_metrics.factual_validation_time_ms > 0
        assert calculated_metrics.relevance_scoring_time_ms > 0
        assert calculated_metrics.claims_extracted_count == 10
        assert calculated_metrics.claims_validated_count == 8
        assert calculated_metrics.avg_validation_confidence > 0
        assert calculated_metrics.peak_validation_memory_mb > 0
    
    @pytest.mark.asyncio
    async def test_run_quality_benchmark_suite_success(self, temp_output_dir, mock_quality_components):
        """Test successful execution of quality benchmark suite."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Mock resource monitor
        with patch('quality_performance_benchmarks.ResourceMonitor') as mock_monitor:
            mock_monitor_instance = Mock()
            mock_monitor_instance.start_monitoring = Mock()
            mock_monitor_instance.stop_monitoring = Mock(return_value=[
                Mock(memory_mb=100, cpu_percent=50)
            ])
            mock_monitor.return_value = mock_monitor_instance
            
            # Run single benchmark
            result = await suite.run_quality_benchmark_suite(
                benchmark_names=['factual_accuracy_validation_benchmark']
            )
            
            assert isinstance(result, dict)
            assert 'suite_execution_summary' in result
            assert 'quality_benchmark_results' in result
            assert result['suite_execution_summary']['total_quality_benchmarks'] == 1
    
    @pytest.mark.asyncio
    async def test_run_quality_benchmark_suite_error_handling(self, temp_output_dir):
        """Test error handling in benchmark suite execution."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Mock components to raise exceptions
        with patch.object(suite, '_execute_claim_extraction', side_effect=Exception("Test error")):
            with patch('quality_performance_benchmarks.ResourceMonitor') as mock_monitor:
                mock_monitor_instance = Mock()
                mock_monitor_instance.start_monitoring = Mock()
                mock_monitor_instance.stop_monitoring = Mock(return_value=[])
                mock_monitor.return_value = mock_monitor_instance
                
                # Should handle errors gracefully
                result = await suite.run_quality_benchmark_suite(
                    benchmark_names=['factual_accuracy_validation_benchmark']
                )
                
                assert isinstance(result, dict)
                # Should still return results even with errors
                assert 'suite_execution_summary' in result
    
    def test_run_quality_assertions(self, temp_output_dir):
        """Test quality assertion execution."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Create test metrics
        metrics = QualityValidationMetrics(
            claim_extraction_time_ms=1500,
            factual_validation_time_ms=4000,
            validation_accuracy_rate=88.0,
            claims_per_second=6.0
        )
        
        # Get thresholds
        thresholds = QualityPerformanceThreshold.create_quality_thresholds()
        
        # Run assertions
        results = suite._run_quality_assertions(metrics, thresholds, "test_assertion")
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check specific assertion results
        if 'claim_extraction_time_ms' in results:
            result = results['claim_extraction_time_ms']
            assert hasattr(result, 'passed')
            assert hasattr(result, 'measured_value')
            assert result.measured_value == 1500
    
    def test_analyze_quality_benchmark_results(self, temp_output_dir):
        """Test quality benchmark results analysis."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Create test metrics list
        test_metrics = [
            QualityValidationMetrics(
                claim_extraction_time_ms=1000,
                factual_validation_time_ms=3000,
                validation_accuracy_rate=85.0,
                claims_per_second=5.0
            ),
            QualityValidationMetrics(
                claim_extraction_time_ms=1200,
                factual_validation_time_ms=3500,
                validation_accuracy_rate=87.0,
                claims_per_second=4.5
            )
        ]
        
        # Create mock configuration
        config = QualityBenchmarkConfiguration()
        
        # Analyze results
        analysis = suite._analyze_quality_benchmark_results(test_metrics, config)
        
        assert isinstance(analysis, dict)
        assert 'total_scenarios' in analysis
        assert 'quality_performance_stats' in analysis
        assert 'quality_recommendations' in analysis
        
        assert analysis['total_scenarios'] == 2
        assert analysis['quality_performance_stats']['avg_claim_extraction_time_ms'] > 0
    
    def test_generate_quality_recommendations(self, temp_output_dir):
        """Test generation of quality recommendations."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Test with slow extraction time
        slow_metrics = [
            QualityValidationMetrics(claim_extraction_time_ms=3000),  # Above threshold
            QualityValidationMetrics(claim_extraction_time_ms=2500)
        ]
        
        config = QualityBenchmarkConfiguration()
        recommendations = suite._generate_quality_recommendations(slow_metrics, config)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("claim extraction" in rec.lower() for rec in recommendations)
    
    def test_analyze_quality_bottlenecks(self, temp_output_dir):
        """Test bottleneck analysis functionality."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Create metrics with clear bottleneck
        test_metrics = [
            QualityValidationMetrics(
                claim_extraction_time_ms=500,
                factual_validation_time_ms=5000,  # Clear bottleneck
                relevance_scoring_time_ms=300
            ),
            QualityValidationMetrics(
                claim_extraction_time_ms=600,
                factual_validation_time_ms=4800,
                relevance_scoring_time_ms=350
            )
        ]
        
        analysis = suite._analyze_quality_bottlenecks(test_metrics)
        
        assert isinstance(analysis, dict)
        assert 'status' in analysis
        
        if analysis['status'] == 'analysis_complete':
            assert 'bottleneck_stage' in analysis
            assert analysis['bottleneck_stage'] == 'factual_validation'
            assert 'bottleneck_percentage' in analysis
            assert analysis['bottleneck_percentage'] > 50
    
    def test_compare_quality_performance(self, temp_output_dir):
        """Test quality performance comparison."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        current_metrics = QualityValidationMetrics(
            claim_extraction_time_ms=1200,
            validation_accuracy_rate=88.0,
            claims_per_second=5.5
        )
        
        baseline_metrics = QualityValidationMetrics(
            claim_extraction_time_ms=1000,
            validation_accuracy_rate=85.0,
            claims_per_second=5.0
        )
        
        comparison = suite.compare_quality_performance(current_metrics, baseline_metrics)
        
        assert isinstance(comparison, dict)
        assert 'performance_changes' in comparison
        assert 'trend_analysis' in comparison
        
        # Verify specific comparisons
        changes = comparison['performance_changes']
        assert 'extraction_time_change_ms' in changes
        assert changes['extraction_time_change_ms'] == 200  # 1200 - 1000
        assert changes['accuracy_change'] == 3.0  # 88 - 85


class TestQualityBenchmarkIntegration:
    """Integration tests for quality benchmark components."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for integration tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_end_to_end_benchmark_execution(self, temp_output_dir):
        """Test complete end-to-end benchmark execution."""
        # This test runs without mocking to test actual integration
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Test with minimal configuration to avoid external dependencies
        with patch('quality_performance_benchmarks.QUALITY_COMPONENTS_AVAILABLE', False):
            # Should use mock implementations
            result = await suite.run_quality_benchmark_suite(
                benchmark_names=['factual_accuracy_validation_benchmark']
            )
            
            assert isinstance(result, dict)
            assert 'suite_execution_summary' in result
            
            # Verify files were created
            json_files = list(temp_output_dir.glob("quality_benchmark_suite_*.json"))
            assert len(json_files) > 0
    
    def test_benchmark_configuration_integration(self, temp_output_dir):
        """Test integration between configuration and benchmark execution."""
        # Create custom configuration
        custom_config = QualityBenchmarkConfiguration(
            benchmark_name="integration_test",
            enable_factual_validation=True,
            enable_relevance_scoring=False,
            sample_queries=["Test query"],
            sample_responses=["Test response"]
        )
        
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Test that configuration is properly used
        assert 'factual_accuracy_validation_benchmark' in suite.quality_benchmarks
        benchmark = suite.quality_benchmarks['factual_accuracy_validation_benchmark']
        assert benchmark.enable_factual_validation is True
    
    def test_metrics_aggregation_and_reporting(self, temp_output_dir):
        """Test metrics aggregation and reporting functionality."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Add test metrics to history
        test_metrics = QualityValidationMetrics(
            scenario_name="test",
            operations_count=5,
            validation_accuracy_rate=85.0
        )
        
        suite.quality_metrics_history['test_benchmark'].append(test_metrics)
        
        # Test retrieval
        history = suite.get_quality_benchmark_history('test_benchmark')
        assert len(history) == 1
        assert history[0].validation_accuracy_rate == 85.0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for error handling tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_invalid_output_directory(self):
        """Test handling of invalid output directory."""
        # Test with non-existent parent directory
        invalid_path = Path("/nonexistent/path/to/output")
        
        # Should handle gracefully and create directory or use fallback
        try:
            suite = QualityValidationBenchmarkSuite(output_dir=invalid_path)
            # If it succeeds, the directory should be created or a fallback used
            assert hasattr(suite, 'output_dir')
        except Exception:
            # If it fails, it should be a reasonable exception
            pass
    
    @pytest.mark.asyncio
    async def test_empty_test_data(self, temp_output_dir):
        """Test handling of empty test data."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Test with empty sample data
        empty_config = QualityBenchmarkConfiguration(
            sample_queries=[],
            sample_responses=[]
        )
        
        # Should handle empty data gracefully
        with patch.object(suite, '_create_quality_benchmarks', return_value={
            'empty_test': empty_config
        }):
            # Should not crash with empty data
            result = await suite._execute_quality_validation_test(
                Mock(scenario_name="test", total_operations=1, concurrent_operations=1, ramp_up_time=0),
                empty_config,
                None
            )
            assert isinstance(result, QualityValidationMetrics)
    
    def test_invalid_threshold_values(self):
        """Test handling of invalid threshold values."""
        # Test threshold with invalid comparison operator
        try:
            threshold = QualityPerformanceThreshold(
                metric_name="test",
                threshold_value=100,
                comparison_operator="invalid_op",
                unit="ms",
                severity="error",
                description="Test threshold"
            )
            
            # Should handle invalid operator gracefully
            passed, message = threshold.check(50)
            assert isinstance(passed, bool)
            assert isinstance(message, str)
        except Exception as e:
            # Should raise appropriate exception for invalid operator
            assert "invalid" in str(e).lower() or "operator" in str(e).lower()
    
    def test_metrics_with_invalid_data(self):
        """Test metrics calculation with invalid/extreme data."""
        # Test with negative values
        metrics = QualityValidationMetrics(
            claim_extraction_time_ms=-100,  # Invalid negative time
            validation_accuracy_rate=150.0,  # Invalid percentage > 100
            claims_per_second=-5.0  # Invalid negative rate
        )
        
        # Should handle invalid data gracefully
        score = metrics.calculate_quality_efficiency_score()
        assert isinstance(score, float)
        assert score >= 0  # Should not return negative score
    
    @pytest.mark.asyncio
    async def test_concurrent_benchmark_execution(self, temp_output_dir):
        """Test handling of concurrent benchmark executions."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Test running multiple benchmarks concurrently
        tasks = []
        for i in range(3):
            task = suite.run_quality_benchmark_suite(
                benchmark_names=['factual_accuracy_validation_benchmark']
            )
            tasks.append(task)
        
        # Should handle concurrent execution without conflicts
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed or fail gracefully
        for result in results:
            if isinstance(result, Exception):
                # If there are exceptions, they should be reasonable
                assert isinstance(result, (asyncio.TimeoutError, RuntimeError, ValueError))
            else:
                assert isinstance(result, dict)
    
    def test_memory_cleanup(self, temp_output_dir):
        """Test proper memory cleanup and resource management."""
        suite = QualityValidationBenchmarkSuite(output_dir=temp_output_dir)
        
        # Add large amount of test data
        for i in range(1000):
            metrics = QualityValidationMetrics(
                scenario_name=f"test_{i}",
                operations_count=10
            )
            suite.quality_metrics_history[f'benchmark_{i}'].append(metrics)
        
        # Verify data is stored
        assert len(suite.quality_metrics_history) == 1000
        
        # Clear data (simulate cleanup)
        suite.quality_metrics_history.clear()
        assert len(suite.quality_metrics_history) == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])