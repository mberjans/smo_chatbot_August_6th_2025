#!/usr/bin/env python3
"""
Cross-System Performance Correlation Engine for Clinical Metabolomics Oracle.

This module implements the CrossSystemCorrelationEngine class that analyzes performance
metrics across different system components, correlating quality and performance metrics
to provide deep insights into how quality requirements affect system performance and
help optimize resource allocation for quality validation workflows.

Classes:
    - PerformanceCorrelationMetrics: Correlation-specific metrics container
    - QualityPerformanceCorrelation: Quality vs performance correlation data
    - PerformancePredictionModel: Performance prediction based on quality requirements
    - CrossSystemCorrelationEngine: Main correlation analysis engine
    - CorrelationAnalysisReport: Comprehensive correlation analysis report

Key Features:
    - Quality vs performance correlation analysis
    - Performance impact prediction based on quality requirements
    - Resource allocation optimization recommendations
    - Cross-component bottleneck identification
    - Statistical correlation analysis with confidence intervals
    - Performance prediction modeling
    - Quality-performance trade-off analysis
    - System optimization recommendations

Integration Points:
    - APIUsageMetricsLogger: API usage and cost metrics
    - AdvancedResourceMonitor: System resource monitoring
    - QualityValidationBenchmarkSuite: Quality validation metrics
    - Performance benchmarking infrastructure
    - Cost tracking and budget management systems

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import time
import logging
import statistics
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import threading
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle

# Import existing performance and quality components
from ..api_metrics_logger import APIUsageMetricsLogger, MetricType, APIMetric, MetricsAggregator
from ..tests.performance_test_fixtures import (
    PerformanceMetrics, LoadTestScenario, ResourceUsageSnapshot, ResourceMonitor
)
from .quality_performance_benchmarks import (
    QualityValidationMetrics, QualityValidationBenchmarkSuite,
    QualityPerformanceThreshold, QualityBenchmarkConfiguration
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceCorrelationMetrics:
    """
    Metrics container for performance correlation analysis.
    
    Captures relationships between quality requirements and system performance
    across different components and operations.
    """
    
    # Correlation identification
    correlation_id: str = field(default_factory=lambda: f"corr_{int(time.time())}")
    timestamp: float = field(default_factory=time.time)
    analysis_duration_seconds: float = 0.0
    
    # Component performance metrics
    quality_validation_performance: Dict[str, float] = field(default_factory=dict)
    api_usage_metrics: Dict[str, float] = field(default_factory=dict)
    resource_utilization_metrics: Dict[str, float] = field(default_factory=dict)
    system_throughput_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Quality requirement parameters
    quality_strictness_level: str = "standard"  # lenient, standard, strict
    confidence_threshold: float = 0.7
    validation_depth: str = "standard"  # shallow, standard, deep
    accuracy_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Correlation coefficients
    quality_performance_correlations: Dict[str, float] = field(default_factory=dict)
    resource_quality_correlations: Dict[str, float] = field(default_factory=dict)
    cost_quality_correlations: Dict[str, float] = field(default_factory=dict)
    
    # Performance impact scores
    quality_impact_on_latency: float = 0.0
    quality_impact_on_throughput: float = 0.0
    quality_impact_on_resource_usage: float = 0.0
    quality_impact_on_cost: float = 0.0
    
    # Statistical measures
    correlation_confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    
    # Prediction accuracy
    prediction_accuracy_score: float = 0.0
    prediction_error_margin: float = 0.0
    
    def calculate_overall_correlation_strength(self) -> float:
        """Calculate overall correlation strength across all metrics."""
        if not self.quality_performance_correlations:
            return 0.0
        
        correlations = list(self.quality_performance_correlations.values())
        return statistics.mean([abs(corr) for corr in correlations])
    
    def get_strongest_correlations(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get the strongest correlations by absolute value."""
        correlations = [(k, v) for k, v in self.quality_performance_correlations.items()]
        return sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:top_n]


@dataclass
class QualityPerformanceCorrelation:
    """
    Represents correlation between specific quality metrics and performance outcomes.
    """
    
    quality_metric_name: str
    performance_metric_name: str
    correlation_coefficient: float
    confidence_interval: Tuple[float, float]
    p_value: float
    sample_size: int
    correlation_type: str  # "positive", "negative", "none"
    strength: str  # "weak", "moderate", "strong"
    
    # Performance impact quantification
    performance_impact_per_quality_unit: float = 0.0
    optimal_quality_threshold: Optional[float] = None
    diminishing_returns_point: Optional[float] = None
    
    # Context information
    system_components_involved: List[str] = field(default_factory=list)
    operational_conditions: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_statistically_significant(self) -> bool:
        """Check if correlation is statistically significant."""
        return self.p_value < 0.05
    
    @property
    def correlation_description(self) -> str:
        """Generate human-readable correlation description."""
        direction = "increases" if self.correlation_coefficient > 0 else "decreases"
        return (f"As {self.quality_metric_name} increases, "
                f"{self.performance_metric_name} {direction} "
                f"({self.strength} correlation: {self.correlation_coefficient:.3f})")


class PerformancePredictionModel:
    """
    Machine learning model for predicting performance based on quality requirements.
    
    Uses multiple regression techniques to predict system performance metrics
    based on quality validation requirements and historical data.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize performance prediction model.
        
        Args:
            model_type: Type of ML model ("linear", "ridge", "random_forest")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        self.prediction_accuracy = 0.0
        
        # Initialize model based on type
        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "ridge":
            self.model = Ridge(alpha=1.0)
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Initialized {model_type} performance prediction model")
    
    def train(self, 
              quality_features: List[List[float]], 
              performance_targets: List[float],
              feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Train the prediction model with historical data.
        
        Args:
            quality_features: Quality requirement feature vectors
            performance_targets: Corresponding performance outcomes
            feature_names: Names of features for interpretability
            
        Returns:
            Training results and model evaluation metrics
        """
        if len(quality_features) != len(performance_targets):
            raise ValueError("Features and targets must have same length")
        
        if len(quality_features) < 5:
            logger.warning("Training with very few samples - predictions may be unreliable")
        
        try:
            # Convert to numpy arrays
            X = np.array(quality_features)
            y = np.array(performance_targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Calculate training accuracy
            y_pred = self.model.predict(X_scaled)
            self.prediction_accuracy = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # Extract feature importance
            if hasattr(self.model, 'feature_importances_'):
                if feature_names and len(feature_names) == X.shape[1]:
                    self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
                else:
                    self.feature_importance = {f"feature_{i}": imp 
                                             for i, imp in enumerate(self.model.feature_importances_)}
            elif hasattr(self.model, 'coef_'):
                if feature_names and len(feature_names) == X.shape[1]:
                    self.feature_importance = dict(zip(feature_names, abs(self.model.coef_)))
                else:
                    self.feature_importance = {f"feature_{i}": abs(coef) 
                                             for i, coef in enumerate(self.model.coef_)}
            
            training_results = {
                "model_type": self.model_type,
                "training_samples": len(quality_features),
                "r2_score": self.prediction_accuracy,
                "mean_absolute_error": mae,
                "feature_importance": self.feature_importance,
                "training_successful": True
            }
            
            logger.info(f"Model training completed - R2: {self.prediction_accuracy:.3f}, MAE: {mae:.3f}")
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {
                "training_successful": False,
                "error": str(e)
            }
    
    def predict(self, quality_features: List[float]) -> Tuple[float, float]:
        """
        Predict performance based on quality requirements.
        
        Args:
            quality_features: Quality requirement feature vector
            
        Returns:
            Tuple of (predicted_performance, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Scale features
            X = np.array([quality_features])
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Calculate confidence based on training accuracy
            confidence = max(0.1, self.prediction_accuracy)
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0, 0.0
    
    def predict_batch(self, quality_features_batch: List[List[float]]) -> List[Tuple[float, float]]:
        """
        Predict performance for batch of quality requirements.
        
        Args:
            quality_features_batch: Batch of quality requirement feature vectors
            
        Returns:
            List of (predicted_performance, confidence_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Scale features
            X = np.array(quality_features_batch)
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            confidence = max(0.1, self.prediction_accuracy)
            
            return [(pred, confidence) for pred in predictions]
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [(0.0, 0.0)] * len(quality_features_batch)
    
    def save_model(self, filepath: Path) -> bool:
        """Save trained model to file."""
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_importance': self.feature_importance,
                'prediction_accuracy': self.prediction_accuracy
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: Path) -> bool:
        """Load trained model from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.feature_importance = model_data['feature_importance']
            self.prediction_accuracy = model_data['prediction_accuracy']
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


@dataclass
class CorrelationAnalysisReport:
    """
    Comprehensive correlation analysis report containing insights and recommendations.
    """
    
    # Report metadata
    report_id: str = field(default_factory=lambda: f"report_{int(time.time())}")
    generated_timestamp: float = field(default_factory=time.time)
    analysis_period: Tuple[float, float] = (0.0, 0.0)
    report_version: str = "1.0.0"
    
    # Correlation analysis results
    correlation_metrics: PerformanceCorrelationMetrics = None
    quality_performance_correlations: List[QualityPerformanceCorrelation] = field(default_factory=list)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Performance predictions
    prediction_accuracy: float = 0.0
    prediction_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    performance_forecasts: Dict[str, float] = field(default_factory=dict)
    
    # Optimization recommendations
    resource_allocation_recommendations: List[str] = field(default_factory=list)
    quality_threshold_recommendations: Dict[str, float] = field(default_factory=dict)
    performance_optimization_suggestions: List[str] = field(default_factory=list)
    cost_optimization_opportunities: List[str] = field(default_factory=list)
    
    # Bottleneck analysis
    performance_bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    resource_constraint_analysis: Dict[str, Any] = field(default_factory=dict)
    scalability_insights: List[str] = field(default_factory=list)
    
    # Quality-performance trade-offs
    trade_off_analysis: Dict[str, Any] = field(default_factory=dict)
    pareto_optimal_points: List[Dict[str, float]] = field(default_factory=list)
    quality_cost_curves: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    
    def get_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of correlation analysis."""
        if not self.correlation_metrics:
            return {"status": "no_data"}
        
        return {
            "analysis_timestamp": datetime.fromtimestamp(self.generated_timestamp).isoformat(),
            "overall_correlation_strength": self.correlation_metrics.calculate_overall_correlation_strength(),
            "strongest_correlations": self.correlation_metrics.get_strongest_correlations(3),
            "prediction_accuracy": self.prediction_accuracy,
            "key_bottlenecks": [b.get("component", "unknown") for b in self.performance_bottlenecks[:3]],
            "top_recommendations": self.resource_allocation_recommendations[:3],
            "cost_impact_summary": {
                "quality_cost_correlation": self.correlation_metrics.cost_quality_correlations.get("overall", 0.0),
                "optimization_potential": len(self.cost_optimization_opportunities)
            }
        }


class CrossSystemCorrelationEngine:
    """
    Cross-system performance correlation engine that analyzes performance metrics
    across different system components, correlating quality and performance metrics
    to provide insights and optimization recommendations.
    
    This engine integrates with APIUsageMetricsLogger, AdvancedResourceMonitor,
    and quality validation systems to provide comprehensive correlation analysis.
    """
    
    def __init__(self,
                 api_metrics_logger: Optional[APIUsageMetricsLogger] = None,
                 output_dir: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cross-system correlation engine.
        
        Args:
            api_metrics_logger: API usage metrics logger for integration
            output_dir: Directory for saving analysis results
            config: Configuration parameters for correlation analysis
        """
        self.api_metrics_logger = api_metrics_logger
        self.output_dir = output_dir or Path("correlation_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or {}
        self.min_sample_size = self.config.get('min_sample_size', 10)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.3)
        
        # Data storage
        self.performance_data: List[PerformanceMetrics] = []
        self.quality_data: List[QualityValidationMetrics] = []
        self.api_metrics_data: List[APIMetric] = []
        self.resource_snapshots: List[ResourceUsageSnapshot] = []
        
        # Analysis components
        self.prediction_models: Dict[str, PerformancePredictionModel] = {}
        self.correlation_history: List[PerformanceCorrelationMetrics] = []
        
        # Threading and monitoring
        self._lock = threading.Lock()
        self._analysis_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info("CrossSystemCorrelationEngine initialized")
    
    def add_performance_data(self, performance_metrics: PerformanceMetrics) -> None:
        """Add performance metrics data for correlation analysis."""
        with self._lock:
            self.performance_data.append(performance_metrics)
            self._clear_analysis_cache()
        logger.debug(f"Added performance data: {performance_metrics.test_name}")
    
    def add_quality_data(self, quality_metrics: QualityValidationMetrics) -> None:
        """Add quality validation metrics for correlation analysis."""
        with self._lock:
            self.quality_data.append(quality_metrics)
            self._clear_analysis_cache()
        logger.debug(f"Added quality data for scenario: {quality_metrics.scenario_name}")
    
    def add_api_metrics(self, api_metrics: List[APIMetric]) -> None:
        """Add API usage metrics for correlation analysis."""
        with self._lock:
            self.api_metrics_data.extend(api_metrics)
            self._clear_analysis_cache()
        logger.debug(f"Added {len(api_metrics)} API metrics records")
    
    def add_resource_snapshots(self, snapshots: List[ResourceUsageSnapshot]) -> None:
        """Add resource usage snapshots for correlation analysis."""
        with self._lock:
            self.resource_snapshots.extend(snapshots)
            self._clear_analysis_cache()
        logger.debug(f"Added {len(snapshots)} resource usage snapshots")
    
    async def analyze_quality_performance_correlation(self,
                                                    quality_metrics: List[str] = None,
                                                    performance_metrics: List[str] = None) -> PerformanceCorrelationMetrics:
        """
        Analyze correlation between quality metrics and performance outcomes.
        
        Args:
            quality_metrics: Specific quality metrics to analyze
            performance_metrics: Specific performance metrics to analyze
            
        Returns:
            Performance correlation metrics with analysis results
        """
        logger.info("Starting quality-performance correlation analysis")
        
        # Check cache
        cache_key = f"quality_perf_correlation_{hash(str(quality_metrics))}_{hash(str(performance_metrics))}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        # Validate data availability
        if len(self.quality_data) < self.min_sample_size or len(self.performance_data) < self.min_sample_size:
            logger.warning(f"Insufficient data for correlation analysis: quality={len(self.quality_data)}, performance={len(self.performance_data)}")
            return PerformanceCorrelationMetrics(
                sample_size=len(self.quality_data) + len(self.performance_data),
                analysis_duration_seconds=time.time() - start_time
            )
        
        # Prepare correlation metrics
        correlation_metrics = PerformanceCorrelationMetrics(
            analysis_duration_seconds=0.0,
            sample_size=min(len(self.quality_data), len(self.performance_data))
        )
        
        # Analyze quality validation performance
        correlation_metrics.quality_validation_performance = self._analyze_quality_validation_performance()
        
        # Analyze API usage correlations
        correlation_metrics.api_usage_metrics = self._analyze_api_usage_correlations()
        
        # Analyze resource utilization correlations
        correlation_metrics.resource_utilization_metrics = self._analyze_resource_utilization_correlations()
        
        # Calculate quality-performance correlations
        correlation_metrics.quality_performance_correlations = await self._calculate_quality_performance_correlations(
            quality_metrics, performance_metrics
        )
        
        # Calculate performance impact scores
        correlation_metrics = self._calculate_performance_impact_scores(correlation_metrics)
        
        # Calculate statistical measures
        correlation_metrics = self._calculate_statistical_measures(correlation_metrics)
        
        # Complete analysis
        correlation_metrics.analysis_duration_seconds = time.time() - start_time
        
        # Cache result
        self._cache_result(cache_key, correlation_metrics)
        
        # Store in history
        with self._lock:
            self.correlation_history.append(correlation_metrics)
        
        logger.info(f"Quality-performance correlation analysis completed in {correlation_metrics.analysis_duration_seconds:.2f} seconds")
        
        return correlation_metrics
    
    async def predict_performance_impact(self,
                                       quality_requirements: Dict[str, Any],
                                       target_metrics: List[str] = None) -> Dict[str, Tuple[float, float]]:
        """
        Predict performance impact based on quality requirements.
        
        Args:
            quality_requirements: Dictionary of quality requirement parameters
            target_metrics: List of performance metrics to predict
            
        Returns:
            Dictionary mapping metric names to (predicted_value, confidence) tuples
        """
        logger.info(f"Predicting performance impact for quality requirements: {quality_requirements}")
        
        # Default target metrics
        if target_metrics is None:
            target_metrics = [
                "response_time_ms", "throughput_ops_per_sec", 
                "memory_usage_mb", "cpu_usage_percent", "cost_usd"
            ]
        
        # Prepare quality features
        quality_features = self._extract_quality_features(quality_requirements)
        
        predictions = {}
        
        for metric_name in target_metrics:
            try:
                # Get or create prediction model for this metric
                model = await self._get_prediction_model(metric_name)
                
                if model and model.is_trained:
                    # Make prediction
                    predicted_value, confidence = model.predict(quality_features)
                    predictions[metric_name] = (predicted_value, confidence)
                    
                    logger.debug(f"Predicted {metric_name}: {predicted_value:.3f} (confidence: {confidence:.3f})")
                else:
                    # Use fallback statistical prediction
                    predicted_value, confidence = self._statistical_prediction(metric_name, quality_requirements)
                    predictions[metric_name] = (predicted_value, confidence)
                    
                    logger.debug(f"Statistical prediction for {metric_name}: {predicted_value:.3f}")
                    
            except Exception as e:
                logger.error(f"Failed to predict {metric_name}: {e}")
                predictions[metric_name] = (0.0, 0.0)
        
        return predictions
    
    async def optimize_resource_allocation(self,
                                         quality_targets: Dict[str, float],
                                         performance_constraints: Dict[str, float],
                                         cost_budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize resource allocation for given quality targets and performance constraints.
        
        Args:
            quality_targets: Target quality metrics to achieve
            performance_constraints: Performance constraints to respect
            cost_budget: Optional cost budget constraint
            
        Returns:
            Resource allocation optimization recommendations
        """
        logger.info(f"Optimizing resource allocation for quality targets: {quality_targets}")
        
        optimization_results = {
            "optimization_timestamp": time.time(),
            "quality_targets": quality_targets,
            "performance_constraints": performance_constraints,
            "cost_budget": cost_budget,
            "recommendations": [],
            "trade_offs": [],
            "pareto_solutions": [],
            "feasibility_analysis": {}
        }
        
        try:
            # Analyze current performance vs quality correlations
            correlation_metrics = await self.analyze_quality_performance_correlation()
            
            # Generate resource allocation recommendations
            recommendations = await self._generate_resource_allocation_recommendations(
                quality_targets, performance_constraints, correlation_metrics, cost_budget
            )
            optimization_results["recommendations"] = recommendations
            
            # Analyze quality-performance trade-offs
            trade_offs = await self._analyze_quality_performance_tradeoffs(
                quality_targets, performance_constraints
            )
            optimization_results["trade_offs"] = trade_offs
            
            # Find Pareto-optimal solutions
            pareto_solutions = await self._find_pareto_optimal_solutions(
                quality_targets, performance_constraints, cost_budget
            )
            optimization_results["pareto_solutions"] = pareto_solutions
            
            # Feasibility analysis
            feasibility = await self._analyze_feasibility(
                quality_targets, performance_constraints, cost_budget
            )
            optimization_results["feasibility_analysis"] = feasibility
            
            logger.info(f"Resource allocation optimization completed with {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Resource allocation optimization failed: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def generate_correlation_report(self,
                                        analysis_period: Optional[Tuple[float, float]] = None,
                                        include_predictions: bool = True,
                                        include_recommendations: bool = True) -> CorrelationAnalysisReport:
        """
        Generate comprehensive correlation analysis report.
        
        Args:
            analysis_period: Time period for analysis (start_time, end_time)
            include_predictions: Whether to include performance predictions
            include_recommendations: Whether to include optimization recommendations
            
        Returns:
            Comprehensive correlation analysis report
        """
        logger.info("Generating comprehensive correlation analysis report")
        
        start_time = time.time()
        
        # Initialize report
        report = CorrelationAnalysisReport(
            analysis_period=analysis_period or (0.0, time.time())
        )
        
        try:
            # Analyze correlations
            correlation_metrics = await self.analyze_quality_performance_correlation()
            report.correlation_metrics = correlation_metrics
            
            # Generate quality-performance correlations
            report.quality_performance_correlations = await self._generate_quality_performance_correlations()
            
            # Statistical summary
            report.statistical_summary = self._generate_statistical_summary()
            
            # Performance predictions
            if include_predictions:
                prediction_results = await self._generate_prediction_analysis()
                report.prediction_accuracy = prediction_results.get("accuracy", 0.0)
                report.prediction_models = prediction_results.get("models", {})
                report.performance_forecasts = prediction_results.get("forecasts", {})
            
            # Optimization recommendations
            if include_recommendations:
                recommendations = await self._generate_optimization_recommendations()
                report.resource_allocation_recommendations = recommendations.get("resource_allocation", [])
                report.quality_threshold_recommendations = recommendations.get("quality_thresholds", {})
                report.performance_optimization_suggestions = recommendations.get("performance_optimization", [])
                report.cost_optimization_opportunities = recommendations.get("cost_optimization", [])
            
            # Bottleneck analysis
            report.performance_bottlenecks = await self._analyze_performance_bottlenecks()
            report.resource_constraint_analysis = await self._analyze_resource_constraints()
            report.scalability_insights = await self._generate_scalability_insights()
            
            # Quality-performance trade-offs
            trade_off_analysis = await self._analyze_comprehensive_tradeoffs()
            report.trade_off_analysis = trade_off_analysis.get("analysis", {})
            report.pareto_optimal_points = trade_off_analysis.get("pareto_points", [])
            report.quality_cost_curves = trade_off_analysis.get("cost_curves", {})
            
            logger.info(f"Correlation analysis report generated in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to generate correlation report: {e}")
            report.statistical_summary = {"error": str(e)}
        
        return report
    
    async def save_correlation_report(self, 
                                    report: CorrelationAnalysisReport,
                                    include_raw_data: bool = False) -> Path:
        """
        Save correlation analysis report to file.
        
        Args:
            report: Correlation analysis report to save
            include_raw_data: Whether to include raw data in the report
            
        Returns:
            Path to saved report file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"correlation_analysis_report_{timestamp}.json"
        
        try:
            # Prepare report data
            report_data = asdict(report)
            
            if include_raw_data:
                report_data["raw_data"] = {
                    "performance_metrics": [asdict(m) for m in self.performance_data[-100:]],  # Last 100
                    "quality_metrics": [asdict(m) for m in self.quality_data[-100:]],
                    "api_metrics": [asdict(m) for m in self.api_metrics_data[-100:]],
                    "resource_snapshots": [m.to_dict() for m in self.resource_snapshots[-100:]]
                }
            
            # Save JSON report
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Generate summary text file
            summary_path = self.output_dir / f"correlation_summary_{timestamp}.txt"
            with open(summary_path, 'w') as f:
                f.write(self._generate_report_summary_text(report))
            
            logger.info(f"Correlation analysis report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save correlation report: {e}")
            raise
        
        return report_path
    
    def _analyze_quality_validation_performance(self) -> Dict[str, float]:
        """Analyze quality validation performance metrics."""
        if not self.quality_data:
            return {}
        
        quality_performance = {}
        
        # Average performance metrics
        quality_performance["avg_claim_extraction_time_ms"] = statistics.mean([
            m.claim_extraction_time_ms for m in self.quality_data if m.claim_extraction_time_ms > 0
        ]) if any(m.claim_extraction_time_ms > 0 for m in self.quality_data) else 0.0
        
        quality_performance["avg_factual_validation_time_ms"] = statistics.mean([
            m.factual_validation_time_ms for m in self.quality_data if m.factual_validation_time_ms > 0
        ]) if any(m.factual_validation_time_ms > 0 for m in self.quality_data) else 0.0
        
        quality_performance["avg_relevance_scoring_time_ms"] = statistics.mean([
            m.relevance_scoring_time_ms for m in self.quality_data if m.relevance_scoring_time_ms > 0
        ]) if any(m.relevance_scoring_time_ms > 0 for m in self.quality_data) else 0.0
        
        quality_performance["avg_validation_accuracy_rate"] = statistics.mean([
            m.validation_accuracy_rate for m in self.quality_data
        ])
        
        quality_performance["avg_claims_per_second"] = statistics.mean([
            m.claims_per_second for m in self.quality_data
        ])
        
        quality_performance["avg_efficiency_score"] = statistics.mean([
            m.calculate_quality_efficiency_score() for m in self.quality_data
        ])
        
        return quality_performance
    
    def _analyze_api_usage_correlations(self) -> Dict[str, float]:
        """Analyze API usage metrics correlations."""
        if not self.api_metrics_data:
            return {}
        
        api_metrics = {}
        
        # Calculate API usage statistics
        api_metrics["avg_response_time_ms"] = statistics.mean([
            m.response_time_ms for m in self.api_metrics_data if m.response_time_ms
        ]) if any(m.response_time_ms for m in self.api_metrics_data) else 0.0
        
        api_metrics["avg_total_tokens"] = statistics.mean([
            m.total_tokens for m in self.api_metrics_data
        ])
        
        api_metrics["avg_cost_usd"] = statistics.mean([
            m.cost_usd for m in self.api_metrics_data
        ])
        
        api_metrics["success_rate"] = (
            len([m for m in self.api_metrics_data if m.success]) / 
            len(self.api_metrics_data) * 100
        ) if self.api_metrics_data else 0.0
        
        api_metrics["avg_throughput"] = statistics.mean([
            m.throughput_tokens_per_sec for m in self.api_metrics_data 
            if m.throughput_tokens_per_sec
        ]) if any(m.throughput_tokens_per_sec for m in self.api_metrics_data) else 0.0
        
        return api_metrics
    
    def _analyze_resource_utilization_correlations(self) -> Dict[str, float]:
        """Analyze resource utilization correlations."""
        if not self.resource_snapshots:
            return {}
        
        resource_metrics = {}
        
        # Calculate resource usage statistics
        resource_metrics["avg_cpu_percent"] = statistics.mean([
            s.cpu_percent for s in self.resource_snapshots
        ])
        
        resource_metrics["avg_memory_mb"] = statistics.mean([
            s.memory_mb for s in self.resource_snapshots
        ])
        
        resource_metrics["max_memory_mb"] = max([
            s.memory_mb for s in self.resource_snapshots
        ])
        
        resource_metrics["avg_disk_io_read_mb"] = statistics.mean([
            s.disk_io_read_mb for s in self.resource_snapshots
        ])
        
        resource_metrics["avg_disk_io_write_mb"] = statistics.mean([
            s.disk_io_write_mb for s in self.resource_snapshots
        ])
        
        resource_metrics["avg_active_threads"] = statistics.mean([
            s.active_threads for s in self.resource_snapshots
        ])
        
        return resource_metrics
    
    async def _calculate_quality_performance_correlations(self,
                                                        quality_metrics: List[str] = None,
                                                        performance_metrics: List[str] = None) -> Dict[str, float]:
        """Calculate correlation coefficients between quality and performance metrics."""
        correlations = {}
        
        # Default metrics if not specified
        if quality_metrics is None:
            quality_metrics = [
                "validation_accuracy_rate", "claims_per_second", "avg_validation_confidence"
            ]
        
        if performance_metrics is None:
            performance_metrics = [
                "response_time_ms", "throughput_ops_per_sec", "memory_usage_mb", "cost_usd"
            ]
        
        # Calculate correlations between all quality-performance metric pairs
        for quality_metric in quality_metrics:
            for performance_metric in performance_metrics:
                correlation_key = f"{quality_metric}_vs_{performance_metric}"
                
                try:
                    # Extract data for correlation calculation
                    quality_values = self._extract_quality_metric_values(quality_metric)
                    performance_values = self._extract_performance_metric_values(performance_metric)
                    
                    if len(quality_values) >= 3 and len(performance_values) >= 3:
                        # Align data points by timestamp or index
                        aligned_quality, aligned_performance = self._align_data_points(
                            quality_values, performance_values
                        )
                        
                        if len(aligned_quality) >= 3 and len(aligned_performance) >= 3:
                            # Calculate Pearson correlation
                            correlation_coeff, p_value = stats.pearsonr(aligned_quality, aligned_performance)
                            
                            if not np.isnan(correlation_coeff):
                                correlations[correlation_key] = correlation_coeff
                                logger.debug(f"Correlation {correlation_key}: {correlation_coeff:.3f} (p={p_value:.3f})")
                            
                except Exception as e:
                    logger.debug(f"Failed to calculate correlation for {correlation_key}: {e}")
        
        return correlations
    
    def _calculate_performance_impact_scores(self, 
                                           correlation_metrics: PerformanceCorrelationMetrics) -> PerformanceCorrelationMetrics:
        """Calculate performance impact scores based on correlations."""
        
        # Impact on latency (higher quality generally increases latency)
        latency_correlations = [
            v for k, v in correlation_metrics.quality_performance_correlations.items()
            if "response_time" in k.lower() or "latency" in k.lower()
        ]
        correlation_metrics.quality_impact_on_latency = statistics.mean(latency_correlations) if latency_correlations else 0.0
        
        # Impact on throughput (higher quality generally decreases throughput)
        throughput_correlations = [
            v for k, v in correlation_metrics.quality_performance_correlations.items()
            if "throughput" in k.lower() or "ops_per_sec" in k.lower()
        ]
        correlation_metrics.quality_impact_on_throughput = statistics.mean(throughput_correlations) if throughput_correlations else 0.0
        
        # Impact on resource usage (higher quality generally increases resource usage)
        resource_correlations = [
            v for k, v in correlation_metrics.quality_performance_correlations.items()
            if "memory" in k.lower() or "cpu" in k.lower()
        ]
        correlation_metrics.quality_impact_on_resource_usage = statistics.mean(resource_correlations) if resource_correlations else 0.0
        
        # Impact on cost (higher quality generally increases cost)
        cost_correlations = [
            v for k, v in correlation_metrics.quality_performance_correlations.items()
            if "cost" in k.lower()
        ]
        correlation_metrics.quality_impact_on_cost = statistics.mean(cost_correlations) if cost_correlations else 0.0
        
        return correlation_metrics
    
    def _calculate_statistical_measures(self, 
                                      correlation_metrics: PerformanceCorrelationMetrics) -> PerformanceCorrelationMetrics:
        """Calculate statistical measures for correlations."""
        
        # Calculate confidence intervals for correlations
        for correlation_key, correlation_value in correlation_metrics.quality_performance_correlations.items():
            if abs(correlation_value) > 0.01:  # Only for non-zero correlations
                # Calculate confidence interval using Fisher transformation
                n = correlation_metrics.sample_size
                if n > 3:
                    z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
                    fisher_z = np.arctanh(correlation_value)
                    se_z = 1 / np.sqrt(n - 3)
                    
                    lower_z = fisher_z - z_score * se_z
                    upper_z = fisher_z + z_score * se_z
                    
                    lower_r = np.tanh(lower_z)
                    upper_r = np.tanh(upper_z)
                    
                    correlation_metrics.correlation_confidence_intervals[correlation_key] = (lower_r, upper_r)
        
        # Calculate statistical significance (p-values)
        for correlation_key, correlation_value in correlation_metrics.quality_performance_correlations.items():
            # Approximate p-value calculation
            n = correlation_metrics.sample_size
            if n > 2:
                t_stat = correlation_value * np.sqrt((n - 2) / (1 - correlation_value ** 2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                correlation_metrics.statistical_significance[correlation_key] = p_value
        
        return correlation_metrics
    
    def _extract_quality_features(self, quality_requirements: Dict[str, Any]) -> List[float]:
        """Extract numerical features from quality requirements."""
        features = []
        
        # Standard quality requirement features
        features.append(quality_requirements.get('confidence_threshold', 0.7))
        features.append(quality_requirements.get('validation_strictness_level', 1.0))  # 0=lenient, 1=standard, 2=strict
        features.append(quality_requirements.get('max_claims_per_response', 50))
        features.append(quality_requirements.get('accuracy_requirement', 85.0))
        features.append(quality_requirements.get('enable_factual_validation', 1.0))
        features.append(quality_requirements.get('enable_relevance_scoring', 1.0))
        features.append(quality_requirements.get('enable_claim_extraction', 1.0))
        
        return features
    
    async def _get_prediction_model(self, metric_name: str) -> Optional[PerformancePredictionModel]:
        """Get or create prediction model for specific metric."""
        if metric_name not in self.prediction_models:
            # Create new model
            model = PerformancePredictionModel(model_type="random_forest")
            
            # Train model if we have sufficient data
            if len(self.quality_data) >= self.min_sample_size and len(self.performance_data) >= self.min_sample_size:
                await self._train_prediction_model(model, metric_name)
            
            self.prediction_models[metric_name] = model
        
        return self.prediction_models[metric_name]
    
    async def _train_prediction_model(self, model: PerformancePredictionModel, metric_name: str) -> None:
        """Train prediction model with available data."""
        try:
            # Prepare training data
            quality_features = []
            performance_targets = []
            
            # Align quality and performance data
            for i in range(min(len(self.quality_data), len(self.performance_data))):
                quality_metric = self.quality_data[i]
                performance_metric = self.performance_data[i]
                
                # Extract features
                features = [
                    quality_metric.validation_accuracy_rate,
                    quality_metric.claims_per_second,
                    quality_metric.avg_validation_confidence,
                    quality_metric.claim_extraction_time_ms,
                    quality_metric.factual_validation_time_ms,
                    quality_metric.relevance_scoring_time_ms,
                    quality_metric.calculate_quality_efficiency_score()
                ]
                
                # Extract target based on metric name
                target = self._extract_performance_target(performance_metric, metric_name)
                
                if target is not None:
                    quality_features.append(features)
                    performance_targets.append(target)
            
            # Train model
            if len(quality_features) >= self.min_sample_size:
                feature_names = [
                    "validation_accuracy", "claims_per_second", "validation_confidence",
                    "extraction_time", "validation_time", "scoring_time", "efficiency_score"
                ]
                
                training_results = model.train(quality_features, performance_targets, feature_names)
                
                if training_results.get("training_successful", False):
                    logger.info(f"Model trained for {metric_name} - R2: {training_results.get('r2_score', 0):.3f}")
                else:
                    logger.warning(f"Failed to train model for {metric_name}: {training_results.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Failed to train prediction model for {metric_name}: {e}")
    
    def _extract_performance_target(self, performance_metric: PerformanceMetrics, metric_name: str) -> Optional[float]:
        """Extract performance target value based on metric name."""
        if metric_name == "response_time_ms":
            return performance_metric.average_latency_ms
        elif metric_name == "throughput_ops_per_sec":
            return performance_metric.throughput_ops_per_sec
        elif metric_name == "memory_usage_mb":
            return performance_metric.memory_usage_mb
        elif metric_name == "cpu_usage_percent":
            return performance_metric.cpu_usage_percent
        elif metric_name == "cost_usd":
            # Estimate cost based on API metrics
            return self._estimate_cost_from_performance(performance_metric)
        else:
            return None
    
    def _estimate_cost_from_performance(self, performance_metric: PerformanceMetrics) -> float:
        """Estimate cost based on performance metrics."""
        # Simple cost estimation based on operations and average API costs
        if self.api_metrics_data:
            avg_cost_per_operation = statistics.mean([m.cost_usd for m in self.api_metrics_data if m.cost_usd > 0])
            return performance_metric.operations_count * avg_cost_per_operation
        return 0.0
    
    def _statistical_prediction(self, metric_name: str, quality_requirements: Dict[str, Any]) -> Tuple[float, float]:
        """Fallback statistical prediction when ML model is not available."""
        # Simple statistical prediction based on historical averages
        if metric_name == "response_time_ms" and self.performance_data:
            base_time = statistics.mean([m.average_latency_ms for m in self.performance_data])
            # Adjust based on quality strictness
            strictness_multiplier = quality_requirements.get('validation_strictness_level', 1.0)
            predicted_time = base_time * (1.0 + strictness_multiplier * 0.3)
            return predicted_time, 0.6
        
        elif metric_name == "throughput_ops_per_sec" and self.performance_data:
            base_throughput = statistics.mean([m.throughput_ops_per_sec for m in self.performance_data])
            # Adjust based on quality requirements
            quality_impact = quality_requirements.get('confidence_threshold', 0.7)
            predicted_throughput = base_throughput * (2.0 - quality_impact)
            return predicted_throughput, 0.5
        
        elif metric_name == "memory_usage_mb" and self.performance_data:
            base_memory = statistics.mean([m.memory_usage_mb for m in self.performance_data])
            # Higher quality requirements typically use more memory
            quality_multiplier = 1.0 + quality_requirements.get('accuracy_requirement', 85.0) / 100.0 * 0.2
            predicted_memory = base_memory * quality_multiplier
            return predicted_memory, 0.4
        
        return 0.0, 0.0
    
    def _extract_quality_metric_values(self, metric_name: str) -> List[float]:
        """Extract values for a specific quality metric."""
        values = []
        
        for quality_metric in self.quality_data:
            if metric_name == "validation_accuracy_rate":
                values.append(quality_metric.validation_accuracy_rate)
            elif metric_name == "claims_per_second":
                values.append(quality_metric.claims_per_second)
            elif metric_name == "avg_validation_confidence":
                values.append(quality_metric.avg_validation_confidence)
            elif metric_name == "efficiency_score":
                values.append(quality_metric.calculate_quality_efficiency_score())
        
        return values
    
    def _extract_performance_metric_values(self, metric_name: str) -> List[float]:
        """Extract values for a specific performance metric."""
        values = []
        
        for performance_metric in self.performance_data:
            if metric_name == "response_time_ms":
                values.append(performance_metric.average_latency_ms)
            elif metric_name == "throughput_ops_per_sec":
                values.append(performance_metric.throughput_ops_per_sec)
            elif metric_name == "memory_usage_mb":
                values.append(performance_metric.memory_usage_mb)
            elif metric_name == "cpu_usage_percent":
                values.append(performance_metric.cpu_usage_percent)
        
        # Also extract from API metrics if available
        if metric_name == "cost_usd" and self.api_metrics_data:
            values.extend([m.cost_usd for m in self.api_metrics_data])
        
        return values
    
    def _align_data_points(self, quality_values: List[float], performance_values: List[float]) -> Tuple[List[float], List[float]]:
        """Align quality and performance data points for correlation analysis."""
        # Simple alignment - take the minimum length
        min_length = min(len(quality_values), len(performance_values))
        return quality_values[:min_length], performance_values[:min_length]
    
    async def _generate_resource_allocation_recommendations(self,
                                                          quality_targets: Dict[str, float],
                                                          performance_constraints: Dict[str, float],
                                                          correlation_metrics: PerformanceCorrelationMetrics,
                                                          cost_budget: Optional[float]) -> List[str]:
        """Generate resource allocation recommendations."""
        recommendations = []
        
        # Analyze current correlations
        strong_correlations = correlation_metrics.get_strongest_correlations(5)
        
        for correlation_name, correlation_value in strong_correlations:
            if abs(correlation_value) > self.correlation_threshold:
                if "memory" in correlation_name.lower() and correlation_value > 0:
                    recommendations.append(
                        f"Quality improvements show strong correlation with memory usage (+{correlation_value:.2f}). "
                        f"Consider allocating additional memory resources for quality validation workflows."
                    )
                elif "response_time" in correlation_name.lower() and correlation_value > 0:
                    recommendations.append(
                        f"Higher quality requirements significantly impact response time (+{correlation_value:.2f}). "
                        f"Consider implementing parallel processing or caching to mitigate latency increases."
                    )
                elif "throughput" in correlation_name.lower() and correlation_value < 0:
                    recommendations.append(
                        f"Quality validation reduces throughput ({correlation_value:.2f}). "
                        f"Consider batch processing or async validation to maintain system throughput."
                    )
        
        # Cost-based recommendations
        if cost_budget and correlation_metrics.quality_impact_on_cost > 0.3:
            recommendations.append(
                f"Quality validation has significant cost impact ({correlation_metrics.quality_impact_on_cost:.2f}). "
                f"Consider implementing tiered quality validation based on content importance."
            )
        
        # Resource optimization based on bottlenecks
        if correlation_metrics.quality_impact_on_resource_usage > 0.5:
            recommendations.append(
                "High resource utilization correlation detected. Recommend implementing resource pooling "
                "and load balancing for quality validation components."
            )
        
        if not recommendations:
            recommendations.append("Current resource allocation appears optimal for given quality targets.")
        
        return recommendations
    
    async def _analyze_quality_performance_tradeoffs(self,
                                                   quality_targets: Dict[str, float],
                                                   performance_constraints: Dict[str, float]) -> List[Dict[str, Any]]:
        """Analyze quality-performance trade-offs."""
        trade_offs = []
        
        # Simulate different quality levels and their performance impact
        quality_levels = [0.6, 0.7, 0.8, 0.9, 0.95]
        
        for quality_level in quality_levels:
            quality_requirements = {
                "confidence_threshold": quality_level,
                "accuracy_requirement": quality_level * 100,
                "validation_strictness_level": 1.0 if quality_level < 0.8 else 2.0
            }
            
            # Predict performance impact
            predictions = await self.predict_performance_impact(quality_requirements)
            
            trade_off = {
                "quality_level": quality_level,
                "predicted_response_time_ms": predictions.get("response_time_ms", (0, 0))[0],
                "predicted_throughput": predictions.get("throughput_ops_per_sec", (0, 0))[0],
                "predicted_memory_mb": predictions.get("memory_usage_mb", (0, 0))[0],
                "predicted_cost": predictions.get("cost_usd", (0, 0))[0],
                "meets_constraints": True
            }
            
            # Check if predictions meet performance constraints
            for constraint, max_value in performance_constraints.items():
                predicted_value = trade_off.get(f"predicted_{constraint}", 0)
                if predicted_value > max_value:
                    trade_off["meets_constraints"] = False
                    break
            
            trade_offs.append(trade_off)
        
        return trade_offs
    
    async def _find_pareto_optimal_solutions(self,
                                           quality_targets: Dict[str, float],
                                           performance_constraints: Dict[str, float],
                                           cost_budget: Optional[float]) -> List[Dict[str, float]]:
        """Find Pareto-optimal solutions balancing quality and performance."""
        solutions = []
        
        # Generate candidate solutions with different quality-performance trade-offs
        for quality_weight in [0.2, 0.4, 0.6, 0.8]:
            performance_weight = 1.0 - quality_weight
            
            # Calculate optimal parameters for this weight combination
            solution = {
                "quality_weight": quality_weight,
                "performance_weight": performance_weight,
                "confidence_threshold": 0.5 + quality_weight * 0.4,
                "validation_strictness": "lenient" if quality_weight < 0.4 else "standard" if quality_weight < 0.7 else "strict",
                "enable_parallel_processing": performance_weight > 0.6,
                "use_caching": performance_weight > 0.5
            }
            
            # Estimate solution performance
            quality_requirements = {
                "confidence_threshold": solution["confidence_threshold"],
                "validation_strictness_level": {"lenient": 0, "standard": 1, "strict": 2}[solution["validation_strictness"]]
            }
            
            predictions = await self.predict_performance_impact(quality_requirements)
            
            solution.update({
                "estimated_quality_score": quality_weight * 100,
                "estimated_response_time": predictions.get("response_time_ms", (0, 0))[0],
                "estimated_throughput": predictions.get("throughput_ops_per_sec", (0, 0))[0],
                "estimated_cost": predictions.get("cost_usd", (0, 0))[0]
            })
            
            solutions.append(solution)
        
        return solutions
    
    async def _analyze_feasibility(self,
                                 quality_targets: Dict[str, float],
                                 performance_constraints: Dict[str, float],
                                 cost_budget: Optional[float]) -> Dict[str, Any]:
        """Analyze feasibility of achieving quality targets within performance constraints."""
        feasibility = {
            "overall_feasible": True,
            "feasibility_score": 0.0,
            "constraint_violations": [],
            "recommendations": []
        }
        
        # Predict performance for target quality levels
        quality_requirements = {
            "confidence_threshold": quality_targets.get("confidence_threshold", 0.8),
            "accuracy_requirement": quality_targets.get("accuracy_requirement", 85.0),
            "validation_strictness_level": 1.0
        }
        
        predictions = await self.predict_performance_impact(quality_requirements)
        
        # Check constraints
        violations = 0
        total_constraints = len(performance_constraints)
        
        for constraint_name, max_value in performance_constraints.items():
            predicted_value = predictions.get(constraint_name, (0, 0))[0]
            
            if predicted_value > max_value:
                violations += 1
                feasibility["constraint_violations"].append({
                    "constraint": constraint_name,
                    "max_allowed": max_value,
                    "predicted": predicted_value,
                    "violation_percentage": ((predicted_value - max_value) / max_value) * 100
                })
                feasibility["overall_feasible"] = False
        
        # Calculate feasibility score
        if total_constraints > 0:
            feasibility["feasibility_score"] = max(0.0, (total_constraints - violations) / total_constraints * 100)
        
        # Generate recommendations for infeasible scenarios
        if not feasibility["overall_feasible"]:
            feasibility["recommendations"].extend([
                "Consider relaxing quality requirements to meet performance constraints",
                "Implement performance optimization techniques (caching, parallel processing)",
                "Increase resource allocation for quality validation components",
                "Use tiered quality validation based on content importance"
            ])
        
        return feasibility
    
    async def _generate_quality_performance_correlations(self) -> List[QualityPerformanceCorrelation]:
        """Generate detailed quality-performance correlation objects."""
        correlations = []
        
        if not self.correlation_history:
            return correlations
        
        latest_correlation = self.correlation_history[-1]
        
        for correlation_name, correlation_value in latest_correlation.quality_performance_correlations.items():
            # Parse correlation name
            parts = correlation_name.split("_vs_")
            if len(parts) != 2:
                continue
            
            quality_metric = parts[0]
            performance_metric = parts[1]
            
            # Determine correlation characteristics
            correlation_type = "positive" if correlation_value > 0.1 else "negative" if correlation_value < -0.1 else "none"
            
            if abs(correlation_value) > 0.7:
                strength = "strong"
            elif abs(correlation_value) > 0.4:
                strength = "moderate"
            else:
                strength = "weak"
            
            # Get confidence interval and p-value
            confidence_interval = latest_correlation.correlation_confidence_intervals.get(correlation_name, (0.0, 0.0))
            p_value = latest_correlation.statistical_significance.get(correlation_name, 1.0)
            
            correlation = QualityPerformanceCorrelation(
                quality_metric_name=quality_metric,
                performance_metric_name=performance_metric,
                correlation_coefficient=correlation_value,
                confidence_interval=confidence_interval,
                p_value=p_value,
                sample_size=latest_correlation.sample_size,
                correlation_type=correlation_type,
                strength=strength
            )
            
            correlations.append(correlation)
        
        return correlations
    
    def _generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate statistical summary of correlation analysis."""
        if not self.correlation_history:
            return {"status": "no_data"}
        
        latest_metrics = self.correlation_history[-1]
        
        return {
            "sample_size": latest_metrics.sample_size,
            "analysis_timestamp": datetime.fromtimestamp(latest_metrics.timestamp).isoformat(),
            "overall_correlation_strength": latest_metrics.calculate_overall_correlation_strength(),
            "number_of_correlations": len(latest_metrics.quality_performance_correlations),
            "significant_correlations": len([
                p for p in latest_metrics.statistical_significance.values() if p < 0.05
            ]),
            "performance_impact_summary": {
                "latency_impact": latest_metrics.quality_impact_on_latency,
                "throughput_impact": latest_metrics.quality_impact_on_throughput,
                "resource_impact": latest_metrics.quality_impact_on_resource_usage,
                "cost_impact": latest_metrics.quality_impact_on_cost
            }
        }
    
    async def _generate_prediction_analysis(self) -> Dict[str, Any]:
        """Generate prediction analysis results."""
        prediction_analysis = {
            "accuracy": 0.0,
            "models": {},
            "forecasts": {}
        }
        
        total_accuracy = 0.0
        model_count = 0
        
        for metric_name, model in self.prediction_models.items():
            if model.is_trained:
                prediction_analysis["models"][metric_name] = {
                    "model_type": model.model_type,
                    "prediction_accuracy": model.prediction_accuracy,
                    "feature_importance": model.feature_importance
                }
                total_accuracy += model.prediction_accuracy
                model_count += 1
                
                # Generate forecasts for different quality scenarios
                scenarios = [
                    {"confidence_threshold": 0.7, "accuracy_requirement": 80.0},
                    {"confidence_threshold": 0.8, "accuracy_requirement": 85.0},
                    {"confidence_threshold": 0.9, "accuracy_requirement": 90.0}
                ]
                
                forecasts = []
                for scenario in scenarios:
                    features = self._extract_quality_features(scenario)
                    predicted_value, confidence = model.predict(features)
                    forecasts.append({
                        "scenario": scenario,
                        "predicted_value": predicted_value,
                        "confidence": confidence
                    })
                
                prediction_analysis["forecasts"][metric_name] = forecasts
        
        if model_count > 0:
            prediction_analysis["accuracy"] = total_accuracy / model_count
        
        return prediction_analysis
    
    async def _generate_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations."""
        recommendations = {
            "resource_allocation": [],
            "quality_thresholds": {},
            "performance_optimization": [],
            "cost_optimization": []
        }
        
        if not self.correlation_history:
            return recommendations
        
        latest_metrics = self.correlation_history[-1]
        
        # Resource allocation recommendations
        if latest_metrics.quality_impact_on_resource_usage > 0.5:
            recommendations["resource_allocation"].extend([
                "Increase memory allocation for quality validation components",
                "Implement resource pooling to handle peak quality validation loads",
                "Consider horizontal scaling for quality validation services"
            ])
        
        # Quality threshold recommendations
        optimal_thresholds = self._calculate_optimal_quality_thresholds()
        recommendations["quality_thresholds"] = optimal_thresholds
        
        # Performance optimization
        if latest_metrics.quality_impact_on_latency > 0.4:
            recommendations["performance_optimization"].extend([
                "Implement caching for frequently validated claims",
                "Use parallel processing for batch quality validation",
                "Optimize claim extraction algorithms for better performance"
            ])
        
        # Cost optimization
        if latest_metrics.quality_impact_on_cost > 0.3:
            recommendations["cost_optimization"].extend([
                "Implement tiered quality validation based on content importance",
                "Use result caching to reduce redundant API calls",
                "Consider batch processing to optimize API usage costs"
            ])
        
        return recommendations
    
    def _calculate_optimal_quality_thresholds(self) -> Dict[str, float]:
        """Calculate optimal quality thresholds based on performance trade-offs."""
        # Simple heuristic-based optimization
        # In practice, this could use more sophisticated optimization algorithms
        
        optimal_thresholds = {
            "confidence_threshold": 0.75,  # Balance between accuracy and performance
            "accuracy_requirement": 82.0,  # Slightly below strict requirements
            "max_claims_per_response": 30   # Limit to control processing time
        }
        
        # Adjust based on historical performance
        if self.correlation_history:
            latest_metrics = self.correlation_history[-1]
            
            # If high performance impact, reduce thresholds
            if latest_metrics.quality_impact_on_latency > 0.6:
                optimal_thresholds["confidence_threshold"] = 0.7
                optimal_thresholds["accuracy_requirement"] = 80.0
            
            # If low performance impact, can increase thresholds
            if latest_metrics.quality_impact_on_latency < 0.3:
                optimal_thresholds["confidence_threshold"] = 0.85
                optimal_thresholds["accuracy_requirement"] = 87.0
        
        return optimal_thresholds
    
    async def _analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks across system components."""
        bottlenecks = []
        
        # Analyze quality validation bottlenecks
        if self.quality_data:
            avg_times = {
                "claim_extraction": statistics.mean([m.claim_extraction_time_ms for m in self.quality_data if m.claim_extraction_time_ms > 0]),
                "factual_validation": statistics.mean([m.factual_validation_time_ms for m in self.quality_data if m.factual_validation_time_ms > 0]),
                "relevance_scoring": statistics.mean([m.relevance_scoring_time_ms for m in self.quality_data if m.relevance_scoring_time_ms > 0])
            }
            
            # Identify biggest time consumers
            for component, avg_time in avg_times.items():
                if avg_time > 0:
                    bottlenecks.append({
                        "component": f"quality_validation_{component}",
                        "average_time_ms": avg_time,
                        "impact": "high" if avg_time > 2000 else "medium" if avg_time > 1000 else "low",
                        "recommendations": self._get_component_optimization_recommendations(component)
                    })
        
        # Analyze resource bottlenecks
        if self.resource_snapshots:
            avg_cpu = statistics.mean([s.cpu_percent for s in self.resource_snapshots])
            avg_memory = statistics.mean([s.memory_mb for s in self.resource_snapshots])
            
            if avg_cpu > 80:
                bottlenecks.append({
                    "component": "cpu_utilization",
                    "average_usage": avg_cpu,
                    "impact": "high",
                    "recommendations": ["Scale horizontally", "Optimize CPU-intensive algorithms"]
                })
            
            if avg_memory > 2000:  # 2GB threshold
                bottlenecks.append({
                    "component": "memory_utilization",
                    "average_usage_mb": avg_memory,
                    "impact": "medium",
                    "recommendations": ["Implement memory pooling", "Optimize data structures"]
                })
        
        return bottlenecks
    
    def _get_component_optimization_recommendations(self, component: str) -> List[str]:
        """Get optimization recommendations for specific components."""
        recommendations_map = {
            "claim_extraction": [
                "Optimize text processing algorithms",
                "Implement parallel claim extraction",
                "Use pre-compiled regex patterns"
            ],
            "factual_validation": [
                "Improve document indexing for faster lookups",
                "Implement validation result caching",
                "Use parallel validation for multiple claims"
            ],
            "relevance_scoring": [
                "Optimize scoring model parameters",
                "Reduce feature dimensionality",
                "Implement score caching for similar queries"
            ]
        }
        
        return recommendations_map.get(component, ["Optimize component implementation"])
    
    async def _analyze_resource_constraints(self) -> Dict[str, Any]:
        """Analyze resource constraints affecting performance."""
        constraints = {
            "memory_constraints": {},
            "cpu_constraints": {},
            "io_constraints": {},
            "network_constraints": {}
        }
        
        if self.resource_snapshots:
            # Memory analysis
            memory_values = [s.memory_mb for s in self.resource_snapshots]
            constraints["memory_constraints"] = {
                "peak_usage_mb": max(memory_values),
                "average_usage_mb": statistics.mean(memory_values),
                "usage_variance": statistics.stdev(memory_values) if len(memory_values) > 1 else 0,
                "constraint_level": "high" if max(memory_values) > 3000 else "medium" if max(memory_values) > 1500 else "low"
            }
            
            # CPU analysis
            cpu_values = [s.cpu_percent for s in self.resource_snapshots]
            constraints["cpu_constraints"] = {
                "peak_usage_percent": max(cpu_values),
                "average_usage_percent": statistics.mean(cpu_values),
                "usage_variance": statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0,
                "constraint_level": "high" if max(cpu_values) > 90 else "medium" if max(cpu_values) > 70 else "low"
            }
            
            # I/O analysis
            io_read_values = [s.disk_io_read_mb for s in self.resource_snapshots]
            io_write_values = [s.disk_io_write_mb for s in self.resource_snapshots]
            constraints["io_constraints"] = {
                "avg_read_mb": statistics.mean(io_read_values) if io_read_values else 0,
                "avg_write_mb": statistics.mean(io_write_values) if io_write_values else 0,
                "constraint_level": "low"  # Simple classification for now
            }
        
        return constraints
    
    async def _generate_scalability_insights(self) -> List[str]:
        """Generate insights about system scalability characteristics."""
        insights = []
        
        if not self.performance_data or not self.quality_data:
            return ["Insufficient data for scalability analysis"]
        
        # Analyze throughput scaling
        if len(self.performance_data) > 3:
            throughputs = [m.throughput_ops_per_sec for m in self.performance_data]
            throughput_trend = np.polyfit(range(len(throughputs)), throughputs, 1)[0]
            
            if throughput_trend > 0:
                insights.append("System shows positive throughput scaling characteristics")
            elif throughput_trend < -0.1:
                insights.append("System shows degrading throughput under load - investigate bottlenecks")
            else:
                insights.append("System throughput remains stable under varying loads")
        
        # Analyze quality-performance scaling
        if self.correlation_history:
            latest_metrics = self.correlation_history[-1]
            
            if latest_metrics.quality_impact_on_throughput < -0.5:
                insights.append("Quality validation significantly impacts system scalability - consider async processing")
            
            if latest_metrics.quality_impact_on_resource_usage > 0.6:
                insights.append("Quality requirements have high resource impact - plan for vertical scaling")
        
        # Memory scaling analysis
        if self.resource_snapshots and len(self.resource_snapshots) > 10:
            memory_values = [s.memory_mb for s in self.resource_snapshots]
            memory_growth = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
            
            if memory_growth > 1.0:  # More than 1MB per sample
                insights.append("Memory usage shows concerning growth pattern - investigate memory leaks")
            else:
                insights.append("Memory usage remains stable during operations")
        
        if not insights:
            insights.append("System scalability appears adequate for current workloads")
        
        return insights
    
    async def _analyze_comprehensive_tradeoffs(self) -> Dict[str, Any]:
        """Analyze comprehensive quality-performance trade-offs."""
        tradeoff_analysis = {
            "analysis": {},
            "pareto_points": [],
            "cost_curves": {}
        }
        
        # Quality vs Performance analysis
        quality_performance_points = []
        
        # Simulate different quality configurations
        for confidence in [0.6, 0.7, 0.8, 0.9]:
            for accuracy in [75, 80, 85, 90, 95]:
                quality_requirements = {
                    "confidence_threshold": confidence,
                    "accuracy_requirement": accuracy
                }
                
                predictions = await self.predict_performance_impact(quality_requirements)
                
                point = {
                    "quality_score": (confidence * 100 + accuracy) / 2,
                    "performance_score": 100 - predictions.get("response_time_ms", (1000, 0))[0] / 10,  # Normalized
                    "cost": predictions.get("cost_usd", (0, 0))[0],
                    "configuration": quality_requirements
                }
                quality_performance_points.append(point)
        
        # Find Pareto frontier
        pareto_points = self._find_pareto_frontier(quality_performance_points)
        tradeoff_analysis["pareto_points"] = pareto_points
        
        # Generate cost curves
        quality_levels = [60, 70, 80, 90, 95]
        cost_curve = []
        
        for quality_level in quality_levels:
            quality_req = {
                "confidence_threshold": quality_level / 100,
                "accuracy_requirement": quality_level
            }
            predictions = await self.predict_performance_impact(quality_req)
            cost_curve.append((quality_level, predictions.get("cost_usd", (0, 0))[0]))
        
        tradeoff_analysis["cost_curves"]["quality_vs_cost"] = cost_curve
        
        # Analysis summary
        tradeoff_analysis["analysis"] = {
            "optimal_quality_range": self._determine_optimal_quality_range(pareto_points),
            "cost_efficiency_sweet_spot": self._find_cost_efficiency_sweet_spot(cost_curve),
            "diminishing_returns_threshold": self._find_diminishing_returns_threshold(cost_curve)
        }
        
        return tradeoff_analysis
    
    def _find_pareto_frontier(self, points: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Find Pareto-optimal points on quality-performance frontier."""
        pareto_points = []
        
        for point in points:
            is_pareto = True
            for other_point in points:
                if (other_point["quality_score"] >= point["quality_score"] and 
                    other_point["performance_score"] >= point["performance_score"] and
                    (other_point["quality_score"] > point["quality_score"] or 
                     other_point["performance_score"] > point["performance_score"])):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_points.append(point)
        
        return sorted(pareto_points, key=lambda x: x["quality_score"])
    
    def _determine_optimal_quality_range(self, pareto_points: List[Dict[str, float]]) -> Dict[str, float]:
        """Determine optimal quality range from Pareto frontier."""
        if not pareto_points:
            return {"min_quality": 80.0, "max_quality": 85.0}
        
        # Find the range where quality/performance ratio is most balanced
        ratios = [(p["quality_score"] / max(p["performance_score"], 1)) for p in pareto_points]
        optimal_idx = ratios.index(min(ratios))  # Most balanced point
        
        optimal_point = pareto_points[optimal_idx]
        
        return {
            "optimal_quality": optimal_point["quality_score"],
            "min_quality": max(70.0, optimal_point["quality_score"] - 5),
            "max_quality": min(95.0, optimal_point["quality_score"] + 5)
        }
    
    def _find_cost_efficiency_sweet_spot(self, cost_curve: List[Tuple[float, float]]) -> Dict[str, float]:
        """Find cost efficiency sweet spot on quality-cost curve."""
        if len(cost_curve) < 2:
            return {"quality_level": 80.0, "cost_per_quality_unit": 1.0}
        
        # Calculate cost per quality unit
        efficiency_points = []
        for i in range(1, len(cost_curve)):
            quality_delta = cost_curve[i][0] - cost_curve[i-1][0]
            cost_delta = cost_curve[i][1] - cost_curve[i-1][1]
            
            if quality_delta > 0:
                efficiency = cost_delta / quality_delta
                efficiency_points.append((cost_curve[i][0], efficiency))
        
        # Find point with minimum cost per quality unit
        if efficiency_points:
            sweet_spot = min(efficiency_points, key=lambda x: x[1])
            return {
                "quality_level": sweet_spot[0],
                "cost_per_quality_unit": sweet_spot[1]
            }
        
        return {"quality_level": 80.0, "cost_per_quality_unit": 1.0}
    
    def _find_diminishing_returns_threshold(self, cost_curve: List[Tuple[float, float]]) -> float:
        """Find threshold where diminishing returns start."""
        if len(cost_curve) < 3:
            return 85.0
        
        # Calculate second derivative to find inflection point
        second_derivatives = []
        for i in range(1, len(cost_curve) - 1):
            d1 = (cost_curve[i][1] - cost_curve[i-1][1]) / (cost_curve[i][0] - cost_curve[i-1][0])
            d2 = (cost_curve[i+1][1] - cost_curve[i][1]) / (cost_curve[i+1][0] - cost_curve[i][0])
            second_derivative = d2 - d1
            second_derivatives.append((cost_curve[i][0], second_derivative))
        
        # Find point where acceleration becomes significant (positive)
        for quality, accel in second_derivatives:
            if accel > 0.1:  # Threshold for significant acceleration
                return quality
        
        return 85.0  # Default threshold
    
    def _generate_report_summary_text(self, report: CorrelationAnalysisReport) -> str:
        """Generate human-readable report summary."""
        executive_summary = report.get_executive_summary()
        
        summary_text = f"""
CLINICAL METABOLOMICS ORACLE - CROSS-SYSTEM CORRELATION ANALYSIS REPORT
======================================================================

Executive Summary:
- Analysis Timestamp: {executive_summary.get('analysis_timestamp', 'Unknown')}
- Overall Correlation Strength: {executive_summary.get('overall_correlation_strength', 0.0):.3f}
- Prediction Accuracy: {executive_summary.get('prediction_accuracy', 0.0):.1%}
- Key Performance Bottlenecks: {', '.join(executive_summary.get('key_bottlenecks', []))}

Top Correlations:
"""
        
        strongest_correlations = executive_summary.get('strongest_correlations', [])
        for i, (correlation_name, strength) in enumerate(strongest_correlations, 1):
            summary_text += f"{i}. {correlation_name}: {strength:.3f}\n"
        
        summary_text += "\nResource Allocation Recommendations:\n"
        for i, rec in enumerate(report.resource_allocation_recommendations[:5], 1):
            summary_text += f"{i}. {rec}\n"
        
        summary_text += "\nPerformance Optimization Suggestions:\n"
        for i, suggestion in enumerate(report.performance_optimization_suggestions[:5], 1):
            summary_text += f"{i}. {suggestion}\n"
        
        if report.trade_off_analysis:
            summary_text += f"\nQuality-Performance Trade-off Analysis:\n"
            optimal_range = report.trade_off_analysis.get('optimal_quality_range', {})
            if optimal_range:
                summary_text += f"- Optimal Quality Range: {optimal_range.get('min_quality', 0):.1f}% - {optimal_range.get('max_quality', 0):.1f}%\n"
            
            cost_sweet_spot = report.trade_off_analysis.get('cost_efficiency_sweet_spot', {})
            if cost_sweet_spot:
                summary_text += f"- Cost Efficiency Sweet Spot: {cost_sweet_spot.get('quality_level', 0):.1f}% quality\n"
        
        summary_text += "\nFor detailed analysis and statistical measures, see the complete JSON report.\n"
        
        return summary_text
    
    def _clear_analysis_cache(self) -> None:
        """Clear analysis cache when data is updated."""
        self._analysis_cache.clear()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached analysis result if still valid."""
        if cache_key in self._analysis_cache:
            result, timestamp = self._analysis_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return result
        return None
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache analysis result with timestamp."""
        self._analysis_cache[cache_key] = (result, time.time())


# Convenience functions for easy usage
def create_correlation_engine(api_metrics_logger: Optional[APIUsageMetricsLogger] = None,
                            output_dir: Optional[Path] = None) -> CrossSystemCorrelationEngine:
    """Create correlation engine with standard configuration."""
    return CrossSystemCorrelationEngine(
        api_metrics_logger=api_metrics_logger,
        output_dir=output_dir
    )


async def analyze_system_correlations(performance_data: List[PerformanceMetrics],
                                    quality_data: List[QualityValidationMetrics],
                                    api_metrics: List[APIMetric] = None,
                                    resource_snapshots: List[ResourceUsageSnapshot] = None) -> CorrelationAnalysisReport:
    """
    Convenience function to analyze system correlations with provided data.
    
    Args:
        performance_data: Performance metrics from benchmarking
        quality_data: Quality validation metrics
        api_metrics: Optional API usage metrics
        resource_snapshots: Optional resource usage snapshots
        
    Returns:
        Comprehensive correlation analysis report
    """
    engine = create_correlation_engine()
    
    # Add data to engine
    for metric in performance_data:
        engine.add_performance_data(metric)
    
    for metric in quality_data:
        engine.add_quality_data(metric)
    
    if api_metrics:
        engine.add_api_metrics(api_metrics)
    
    if resource_snapshots:
        engine.add_resource_snapshots(resource_snapshots)
    
    # Generate comprehensive report
    return await engine.generate_correlation_report()


# Make main classes available at module level
__all__ = [
    'CrossSystemCorrelationEngine',
    'PerformanceCorrelationMetrics',
    'QualityPerformanceCorrelation',
    'PerformancePredictionModel',
    'CorrelationAnalysisReport',
    'create_correlation_engine',
    'analyze_system_correlations'
]