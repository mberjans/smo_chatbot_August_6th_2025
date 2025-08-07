#!/usr/bin/env python3
"""
Statistical Analyzer for Clinical Metabolomics Oracle Performance Data.

This module provides comprehensive statistical analysis capabilities for
performance benchmarking data, including trend analysis, anomaly detection,
correlation analysis, and predictive modeling for quality validation systems.

Classes:
    - TrendAnalysis: Statistical trend analysis results
    - AnomalyDetection: Anomaly detection results and methods
    - CorrelationMatrix: Correlation analysis between metrics
    - PredictiveModel: Statistical models for performance prediction
    - StatisticalAnalyzer: Main statistical analysis engine

Key Features:
    - Time series analysis and forecasting
    - Anomaly detection using multiple algorithms
    - Performance correlation analysis
    - Statistical significance testing
    - Distribution analysis and modeling
    - Regression analysis and prediction
    - Confidence interval calculations
    - Pattern recognition in performance data

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import math

# Statistical analysis imports
try:
    import numpy as np
    from scipy import stats
    from scipy.signal import find_peaks
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    STATS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced statistical libraries not available: {e}. Some features will be limited.")
    STATS_AVAILABLE = False
    
    # Create mock numpy for basic operations
    class MockNumPy:
        @staticmethod
        def array(data): return list(data)
        @staticmethod
        def mean(data): return statistics.mean(data) if data else 0
        @staticmethod
        def std(data): return statistics.stdev(data) if len(data) > 1 else 0
    
    np = MockNumPy()

# Configure logging
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction classifications."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    SEASONAL = "seasonal"
    UNKNOWN = "unknown"


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    OUTLIER = "outlier"
    TREND_CHANGE = "trend_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    LEVEL_SHIFT = "level_shift"
    SPIKE = "spike"
    DIP = "dip"
    PATTERN_BREAK = "pattern_break"


class StatisticalSignificance(Enum):
    """Statistical significance levels."""
    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01
    SIGNIFICANT = "significant"                # p < 0.05
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # p < 0.10
    NOT_SIGNIFICANT = "not_significant"        # p >= 0.10


@dataclass
class TrendAnalysis:
    """Results of statistical trend analysis."""
    
    # Trend identification
    trend_direction: TrendDirection = TrendDirection.UNKNOWN
    trend_strength: float = 0.0  # 0.0 to 1.0
    trend_start_timestamp: Optional[float] = None
    trend_confidence: float = 0.0  # 0.0 to 1.0
    
    # Statistical measures
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    p_value: float = 1.0
    statistical_significance: StatisticalSignificance = StatisticalSignificance.NOT_SIGNIFICANT
    
    # Trend characteristics
    linear_trend_coefficient: float = 0.0
    polynomial_coefficients: List[float] = field(default_factory=list)
    seasonal_components: Dict[str, float] = field(default_factory=dict)
    volatility_measure: float = 0.0
    
    # Predictions and forecasts
    short_term_forecast: Optional[List[float]] = None
    forecast_confidence_intervals: Optional[List[Tuple[float, float]]] = None
    trend_reversal_probability: float = 0.0
    
    # Context and metadata
    analysis_window_size: int = 0
    data_points_analyzed: int = 0
    analysis_timestamp: float = field(default_factory=time.time)
    metric_name: str = ""
    
    def get_trend_description(self) -> str:
        """Get human-readable trend description."""
        strength_desc = "weak" if self.trend_strength < 0.3 else "moderate" if self.trend_strength < 0.7 else "strong"
        direction_desc = self.trend_direction.value.replace('_', ' ')
        significance_desc = self.statistical_significance.value.replace('_', ' ')
        
        return f"{strength_desc} {direction_desc} trend ({significance_desc}, R²={self.r_squared:.3f})"


@dataclass
class AnomalyDetectionResult:
    """Results of anomaly detection analysis."""
    
    # Anomaly identification
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)
    anomaly_count: int = 0
    anomaly_rate: float = 0.0  # Percentage of data points that are anomalous
    
    # Anomaly characteristics
    anomaly_types: Dict[AnomalyType, int] = field(default_factory=dict)
    severity_distribution: Dict[str, int] = field(default_factory=dict)  # low, medium, high, critical
    
    # Detection parameters
    detection_method: str = "isolation_forest"
    sensitivity_threshold: float = 0.05
    confidence_level: float = 0.95
    
    # Statistical measures
    anomaly_scores: List[float] = field(default_factory=list)
    threshold_value: float = 0.0
    false_positive_rate: Optional[float] = None
    
    # Context
    analysis_timestamp: float = field(default_factory=time.time)
    data_points_analyzed: int = 0
    metric_name: str = ""
    
    def get_anomaly_summary(self) -> str:
        """Get summary of anomaly detection results."""
        if self.anomaly_count == 0:
            return "No anomalies detected"
        
        return f"Detected {self.anomaly_count} anomalies ({self.anomaly_rate:.1f}% of data points)"


@dataclass
class CorrelationMatrix:
    """Results of correlation analysis between metrics."""
    
    # Correlation matrix data
    correlation_coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)
    p_values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    
    # Correlation analysis
    strong_correlations: List[Tuple[str, str, float]] = field(default_factory=list)  # metric1, metric2, coefficient
    weak_correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    significant_correlations: List[Tuple[str, str, float, float]] = field(default_factory=list)  # includes p-value
    
    # Meta-analysis
    metrics_analyzed: List[str] = field(default_factory=list)
    sample_size: int = 0
    correlation_method: str = "pearson"
    analysis_timestamp: float = field(default_factory=time.time)
    
    def get_correlation_strength(self, metric1: str, metric2: str) -> str:
        """Get correlation strength description."""
        coeff = abs(self.correlation_coefficients.get(metric1, {}).get(metric2, 0))
        
        if coeff >= 0.8:
            return "very strong"
        elif coeff >= 0.6:
            return "strong"
        elif coeff >= 0.4:
            return "moderate"
        elif coeff >= 0.2:
            return "weak"
        else:
            return "very weak"


@dataclass
class PredictiveModel:
    """Statistical model for performance prediction."""
    
    # Model identification
    model_name: str = "performance_predictor"
    model_type: str = "linear_regression"
    target_metric: str = ""
    feature_metrics: List[str] = field(default_factory=list)
    
    # Model performance
    training_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    mean_squared_error: float = 0.0
    mean_absolute_error: float = 0.0
    r_squared_score: float = 0.0
    
    # Model parameters
    model_coefficients: Dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Prediction capabilities
    prediction_horizon_steps: int = 10
    confidence_intervals: bool = True
    uncertainty_quantification: bool = True
    
    # Training metadata
    training_samples: int = 0
    training_timestamp: float = field(default_factory=time.time)
    model_version: str = "1.0.0"
    
    def predict_performance(self, feature_values: Dict[str, float]) -> Tuple[float, float]:
        """Make performance prediction with confidence."""
        # Simple linear prediction (would use trained model in practice)
        prediction = self.intercept
        for feature, value in feature_values.items():
            if feature in self.model_coefficients:
                prediction += self.model_coefficients[feature] * value
        
        # Confidence based on training accuracy
        confidence = self.training_accuracy
        
        return prediction, confidence


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis engine for performance data.
    
    Provides advanced statistical analysis capabilities including trend analysis,
    anomaly detection, correlation analysis, and predictive modeling for
    quality validation performance data.
    """
    
    def __init__(self,
                 confidence_level: float = 0.95,
                 anomaly_sensitivity: float = 0.05,
                 trend_window_size: int = 20,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the statistical analyzer.
        
        Args:
            confidence_level: Statistical confidence level (0.0-1.0)
            anomaly_sensitivity: Sensitivity threshold for anomaly detection
            trend_window_size: Window size for trend analysis
            logger: Logger instance for analysis operations
        """
        self.confidence_level = confidence_level
        self.anomaly_sensitivity = anomaly_sensitivity
        self.trend_window_size = trend_window_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Data storage
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        self.timestamps: List[float] = []
        self.metadata: Dict[str, Any] = {}
        
        # Analysis results cache
        self.trend_analyses: Dict[str, TrendAnalysis] = {}
        self.anomaly_results: Dict[str, AnomalyDetectionResult] = {}
        self.correlation_matrix: Optional[CorrelationMatrix] = None
        self.predictive_models: Dict[str, PredictiveModel] = {}
        
        # Statistical tools
        if STATS_AVAILABLE:
            self.scaler = StandardScaler()
            self.anomaly_detector = IsolationForest(
                contamination=anomaly_sensitivity,
                random_state=42
            )
        
        self.logger.info("StatisticalAnalyzer initialized")
    
    async def load_performance_data(self,
                                  benchmark_data: Optional[List[Any]] = None,
                                  api_metrics_data: Optional[List[Any]] = None,
                                  time_series_data: Optional[Dict[str, List[Tuple[float, float]]]] = None) -> int:
        """
        Load performance data for statistical analysis.
        
        Args:
            benchmark_data: Quality validation benchmark data
            api_metrics_data: API usage and cost metrics
            time_series_data: Pre-formatted time series data {metric: [(timestamp, value), ...]}
            
        Returns:
            Total number of data points loaded
        """
        total_loaded = 0
        
        try:
            # Clear existing data
            self.performance_data.clear()
            self.timestamps.clear()
            
            # Load from time series data (most direct)
            if time_series_data:
                for metric, time_series in time_series_data.items():
                    for timestamp, value in time_series:
                        if value is not None and not math.isnan(value):
                            self.performance_data[metric].append(value)
                            if timestamp not in self.timestamps:
                                self.timestamps.append(timestamp)
                            total_loaded += 1
            
            # Load from benchmark data
            if benchmark_data:
                for data_point in benchmark_data:
                    timestamp = getattr(data_point, 'timestamp', None) or getattr(data_point, 'start_time', time.time())
                    self.timestamps.append(timestamp)
                    
                    # Extract numeric metrics
                    for attr_name in dir(data_point):
                        if not attr_name.startswith('_'):
                            try:
                                value = getattr(data_point, attr_name)
                                if isinstance(value, (int, float)) and not math.isnan(value):
                                    self.performance_data[attr_name].append(value)
                                    total_loaded += 1
                            except (AttributeError, TypeError):
                                continue
            
            # Load from API metrics data
            if api_metrics_data:
                for data_point in api_metrics_data:
                    timestamp = getattr(data_point, 'timestamp', time.time())
                    if timestamp not in self.timestamps:
                        self.timestamps.append(timestamp)
                    
                    # Extract numeric metrics
                    for attr_name in ['cost_usd', 'response_time_ms', 'total_tokens', 'quality_score']:
                        if hasattr(data_point, attr_name):
                            try:
                                value = getattr(data_point, attr_name)
                                if isinstance(value, (int, float)) and not math.isnan(value) and value > 0:
                                    self.performance_data[attr_name].append(value)
                                    total_loaded += 1
                            except (AttributeError, TypeError):
                                continue
            
            # Sort timestamps and align data
            if self.timestamps:
                self.timestamps = sorted(list(set(self.timestamps)))
                self._align_data_to_timestamps()
            
            self.logger.info(f"Loaded {total_loaded} data points across {len(self.performance_data)} metrics")
            
        except Exception as e:
            self.logger.error(f"Error loading performance data for analysis: {e}")
            raise
        
        return total_loaded
    
    def _align_data_to_timestamps(self) -> None:
        """Align performance data to common timestamps."""
        # This is a simplified alignment - in practice would interpolate or handle missing data more sophisticatedly
        for metric in list(self.performance_data.keys()):
            if len(self.performance_data[metric]) != len(self.timestamps):
                # Truncate or pad data to match timestamps
                data_len = len(self.performance_data[metric])
                time_len = len(self.timestamps)
                
                if data_len > time_len:
                    self.performance_data[metric] = self.performance_data[metric][:time_len]
                elif data_len < time_len:
                    # Remove excess timestamps or pad data - for now, just remove timestamps
                    self.timestamps = self.timestamps[:data_len]
    
    async def analyze_trends(self, 
                           metrics: Optional[List[str]] = None,
                           window_size: Optional[int] = None) -> Dict[str, TrendAnalysis]:
        """
        Perform trend analysis on performance metrics.
        
        Args:
            metrics: Specific metrics to analyze (all if None)
            window_size: Analysis window size (uses default if None)
            
        Returns:
            Dictionary mapping metric names to trend analysis results
        """
        self.logger.info(f"Performing trend analysis on {len(metrics) if metrics else 'all'} metrics")
        
        window_size = window_size or self.trend_window_size
        metrics_to_analyze = metrics or list(self.performance_data.keys())
        
        trend_results = {}
        
        for metric in metrics_to_analyze:
            if metric not in self.performance_data or len(self.performance_data[metric]) < 3:
                continue
            
            try:
                trend_analysis = await self._analyze_single_metric_trend(metric, window_size)
                trend_results[metric] = trend_analysis
                self.trend_analyses[metric] = trend_analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing trend for metric {metric}: {e}")
        
        self.logger.info(f"Completed trend analysis for {len(trend_results)} metrics")
        return trend_results
    
    async def _analyze_single_metric_trend(self, metric: str, window_size: int) -> TrendAnalysis:
        """Analyze trend for a single metric."""
        values = self.performance_data[metric]
        
        # Use most recent data points for trend analysis
        recent_values = values[-window_size:] if len(values) > window_size else values
        
        trend_analysis = TrendAnalysis(
            metric_name=metric,
            data_points_analyzed=len(recent_values),
            analysis_window_size=len(recent_values)
        )
        
        if len(recent_values) < 3:
            return trend_analysis
        
        # Calculate basic trend using linear regression
        x_values = list(range(len(recent_values)))
        
        if STATS_AVAILABLE:
            try:
                # Scipy-based analysis
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, recent_values)
                
                trend_analysis.slope = slope
                trend_analysis.intercept = intercept
                trend_analysis.r_squared = r_value ** 2
                trend_analysis.p_value = p_value
                trend_analysis.linear_trend_coefficient = slope
                
                # Determine statistical significance
                if p_value < 0.01:
                    trend_analysis.statistical_significance = StatisticalSignificance.HIGHLY_SIGNIFICANT
                elif p_value < 0.05:
                    trend_analysis.statistical_significance = StatisticalSignificance.SIGNIFICANT
                elif p_value < 0.10:
                    trend_analysis.statistical_significance = StatisticalSignificance.MARGINALLY_SIGNIFICANT
                else:
                    trend_analysis.statistical_significance = StatisticalSignificance.NOT_SIGNIFICANT
                
                # Determine trend direction and strength
                if abs(slope) < 0.01:
                    trend_analysis.trend_direction = TrendDirection.STABLE
                    trend_analysis.trend_strength = 0.1
                elif slope > 0:
                    trend_analysis.trend_direction = TrendDirection.INCREASING
                    trend_analysis.trend_strength = min(1.0, abs(r_value))
                else:
                    trend_analysis.trend_direction = TrendDirection.DECREASING
                    trend_analysis.trend_strength = min(1.0, abs(r_value))
                
                # Calculate volatility
                detrended = [val - (slope * i + intercept) for i, val in enumerate(recent_values)]
                trend_analysis.volatility_measure = statistics.stdev(detrended) if len(detrended) > 1 else 0
                
                # Generate short-term forecast
                forecast_steps = min(5, len(recent_values) // 4)
                if forecast_steps > 0:
                    forecast_x = list(range(len(recent_values), len(recent_values) + forecast_steps))
                    forecast_values = [slope * x + intercept for x in forecast_x]
                    trend_analysis.short_term_forecast = forecast_values
                
                # Confidence in trend analysis
                trend_analysis.trend_confidence = min(1.0, abs(r_value) * (1 - p_value))
                
            except Exception as e:
                self.logger.debug(f"Advanced trend analysis failed for {metric}: {e}")
                # Fall back to simple analysis
        
        # Simple trend analysis fallback
        if trend_analysis.slope == 0:
            first_half = recent_values[:len(recent_values)//2]
            second_half = recent_values[len(recent_values)//2:]
            
            if first_half and second_half:
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                
                change_percent = ((second_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
                
                if abs(change_percent) < 5:
                    trend_analysis.trend_direction = TrendDirection.STABLE
                    trend_analysis.trend_strength = 0.2
                elif change_percent > 0:
                    trend_analysis.trend_direction = TrendDirection.INCREASING
                    trend_analysis.trend_strength = min(1.0, abs(change_percent) / 50)
                else:
                    trend_analysis.trend_direction = TrendDirection.DECREASING
                    trend_analysis.trend_strength = min(1.0, abs(change_percent) / 50)
        
        return trend_analysis
    
    async def detect_anomalies(self,
                             metrics: Optional[List[str]] = None,
                             method: str = "isolation_forest") -> Dict[str, AnomalyDetectionResult]:
        """
        Detect anomalies in performance metrics.
        
        Args:
            metrics: Specific metrics to analyze (all if None)
            method: Anomaly detection method ("isolation_forest", "statistical", "zscore")
            
        Returns:
            Dictionary mapping metric names to anomaly detection results
        """
        self.logger.info(f"Performing anomaly detection using {method} method")
        
        metrics_to_analyze = metrics or list(self.performance_data.keys())
        anomaly_results = {}
        
        for metric in metrics_to_analyze:
            if metric not in self.performance_data or len(self.performance_data[metric]) < 5:
                continue
            
            try:
                anomaly_result = await self._detect_anomalies_single_metric(metric, method)
                anomaly_results[metric] = anomaly_result
                self.anomaly_results[metric] = anomaly_result
                
            except Exception as e:
                self.logger.error(f"Error detecting anomalies for metric {metric}: {e}")
        
        self.logger.info(f"Completed anomaly detection for {len(anomaly_results)} metrics")
        return anomaly_results
    
    async def _detect_anomalies_single_metric(self, metric: str, method: str) -> AnomalyDetectionResult:
        """Detect anomalies for a single metric."""
        values = self.performance_data[metric]
        
        result = AnomalyDetectionResult(
            metric_name=metric,
            data_points_analyzed=len(values),
            detection_method=method,
            sensitivity_threshold=self.anomaly_sensitivity
        )
        
        if len(values) < 5:
            return result
        
        anomaly_indices = []
        anomaly_scores = []
        
        if method == "isolation_forest" and STATS_AVAILABLE:
            try:
                # Reshape data for sklearn
                X = np.array(values).reshape(-1, 1)
                
                # Fit isolation forest
                anomaly_labels = self.anomaly_detector.fit_predict(X)
                anomaly_scores = self.anomaly_detector.decision_function(X)
                
                # Find anomalies (labeled as -1)
                anomaly_indices = [i for i, label in enumerate(anomaly_labels) if label == -1]
                result.anomaly_scores = anomaly_scores.tolist()
                
            except Exception as e:
                self.logger.debug(f"Isolation forest failed for {metric}: {e}")
                method = "statistical"  # Fall back
        
        if method == "statistical" or method == "zscore":
            # Statistical anomaly detection using z-score
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if std_val > 0:
                z_scores = [(val - mean_val) / std_val for val in values]
                threshold = stats.norm.ppf(1 - self.anomaly_sensitivity / 2) if STATS_AVAILABLE else 2.0
                
                anomaly_indices = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
                anomaly_scores = [abs(z) for z in z_scores]
                result.threshold_value = threshold
        
        # Process detected anomalies
        result.anomaly_count = len(anomaly_indices)
        result.anomaly_rate = (len(anomaly_indices) / len(values)) * 100
        
        # Classify anomalies
        for idx in anomaly_indices:
            anomaly_value = values[idx]
            anomaly_score = anomaly_scores[idx] if idx < len(anomaly_scores) else 0
            
            # Determine anomaly type and severity
            anomaly_type = self._classify_anomaly_type(values, idx)
            severity = self._determine_anomaly_severity(anomaly_score)
            
            result.anomalies_detected.append({
                'index': idx,
                'timestamp': self.timestamps[idx] if idx < len(self.timestamps) else None,
                'value': anomaly_value,
                'anomaly_score': anomaly_score,
                'anomaly_type': anomaly_type.value,
                'severity': severity
            })
            
            # Update counters
            result.anomaly_types[anomaly_type] = result.anomaly_types.get(anomaly_type, 0) + 1
            result.severity_distribution[severity] = result.severity_distribution.get(severity, 0) + 1
        
        return result
    
    def _classify_anomaly_type(self, values: List[float], anomaly_index: int) -> AnomalyType:
        """Classify the type of anomaly detected."""
        if anomaly_index == 0 or anomaly_index == len(values) - 1:
            return AnomalyType.OUTLIER
        
        anomaly_value = values[anomaly_index]
        prev_value = values[anomaly_index - 1]
        next_value = values[anomaly_index + 1]
        
        # Check for spike or dip
        if anomaly_value > prev_value * 1.5 and anomaly_value > next_value * 1.5:
            return AnomalyType.SPIKE
        elif anomaly_value < prev_value * 0.5 and anomaly_value < next_value * 0.5:
            return AnomalyType.DIP
        
        # Check for trend change
        window_size = min(5, anomaly_index, len(values) - anomaly_index - 1)
        if window_size >= 2:
            before_trend = statistics.mean(values[anomaly_index-window_size:anomaly_index])
            after_trend = statistics.mean(values[anomaly_index+1:anomaly_index+window_size+1])
            
            if abs(after_trend - before_trend) > statistics.stdev(values) * 2:
                return AnomalyType.TREND_CHANGE
        
        return AnomalyType.OUTLIER
    
    def _determine_anomaly_severity(self, anomaly_score: float) -> str:
        """Determine severity level of anomaly."""
        if anomaly_score > 4:
            return "critical"
        elif anomaly_score > 3:
            return "high"
        elif anomaly_score > 2:
            return "medium"
        else:
            return "low"
    
    async def calculate_correlations(self,
                                   metrics: Optional[List[str]] = None,
                                   method: str = "pearson") -> CorrelationMatrix:
        """
        Calculate correlation matrix between metrics.
        
        Args:
            metrics: Specific metrics to analyze (all if None)
            method: Correlation method ("pearson", "spearman", "kendall")
            
        Returns:
            Correlation matrix with statistical significance
        """
        self.logger.info(f"Calculating correlation matrix using {method} method")
        
        metrics_to_analyze = metrics or list(self.performance_data.keys())
        
        # Filter metrics with sufficient data
        valid_metrics = []
        for metric in metrics_to_analyze:
            if metric in self.performance_data and len(self.performance_data[metric]) >= 3:
                valid_metrics.append(metric)
        
        if len(valid_metrics) < 2:
            self.logger.warning("Insufficient metrics for correlation analysis")
            return CorrelationMatrix()
        
        correlation_matrix = CorrelationMatrix(
            metrics_analyzed=valid_metrics,
            correlation_method=method,
            sample_size=min(len(self.performance_data[m]) for m in valid_metrics)
        )
        
        # Calculate pairwise correlations
        for i, metric1 in enumerate(valid_metrics):
            correlation_matrix.correlation_coefficients[metric1] = {}
            correlation_matrix.p_values[metric1] = {}
            correlation_matrix.confidence_intervals[metric1] = {}
            
            for metric2 in valid_metrics:
                if metric1 == metric2:
                    correlation_matrix.correlation_coefficients[metric1][metric2] = 1.0
                    correlation_matrix.p_values[metric1][metric2] = 0.0
                    continue
                
                try:
                    # Get aligned data
                    data1 = self.performance_data[metric1]
                    data2 = self.performance_data[metric2]
                    
                    # Align data lengths
                    min_len = min(len(data1), len(data2))
                    aligned_data1 = data1[:min_len]
                    aligned_data2 = data2[:min_len]
                    
                    if min_len < 3:
                        continue
                    
                    # Calculate correlation
                    if STATS_AVAILABLE:
                        if method == "pearson":
                            corr_coeff, p_value = stats.pearsonr(aligned_data1, aligned_data2)
                        elif method == "spearman":
                            corr_coeff, p_value = stats.spearmanr(aligned_data1, aligned_data2)
                        elif method == "kendall":
                            corr_coeff, p_value = stats.kendalltau(aligned_data1, aligned_data2)
                        else:
                            corr_coeff, p_value = stats.pearsonr(aligned_data1, aligned_data2)
                        
                        # Calculate confidence interval
                        if method == "pearson" and min_len > 3:
                            # Fisher z-transformation for confidence interval
                            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
                            fisher_z = np.arctanh(corr_coeff)
                            se_z = 1 / np.sqrt(min_len - 3)
                            
                            lower_z = fisher_z - z_score * se_z
                            upper_z = fisher_z + z_score * se_z
                            
                            ci_lower = np.tanh(lower_z)
                            ci_upper = np.tanh(upper_z)
                            
                            correlation_matrix.confidence_intervals[metric1][metric2] = (ci_lower, ci_upper)
                        
                    else:
                        # Simple correlation calculation
                        mean1 = statistics.mean(aligned_data1)
                        mean2 = statistics.mean(aligned_data2)
                        
                        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(aligned_data1, aligned_data2))
                        
                        sum_sq1 = sum((x - mean1) ** 2 for x in aligned_data1)
                        sum_sq2 = sum((y - mean2) ** 2 for y in aligned_data2)
                        
                        denominator = math.sqrt(sum_sq1 * sum_sq2)
                        
                        if denominator > 0:
                            corr_coeff = numerator / denominator
                        else:
                            corr_coeff = 0.0
                        
                        p_value = 1.0  # Can't calculate without stats library
                    
                    correlation_matrix.correlation_coefficients[metric1][metric2] = corr_coeff
                    correlation_matrix.p_values[metric1][metric2] = p_value
                    
                    # Classify correlation strength
                    if abs(corr_coeff) >= 0.7:
                        correlation_matrix.strong_correlations.append((metric1, metric2, corr_coeff))
                    elif abs(corr_coeff) <= 0.3:
                        correlation_matrix.weak_correlations.append((metric1, metric2, corr_coeff))
                    
                    # Check statistical significance
                    if p_value < 0.05:
                        correlation_matrix.significant_correlations.append((metric1, metric2, corr_coeff, p_value))
                
                except Exception as e:
                    self.logger.debug(f"Error calculating correlation between {metric1} and {metric2}: {e}")
        
        self.correlation_matrix = correlation_matrix
        self.logger.info(f"Calculated correlations for {len(valid_metrics)} metrics")
        
        return correlation_matrix
    
    async def build_predictive_model(self,
                                   target_metric: str,
                                   feature_metrics: Optional[List[str]] = None,
                                   model_type: str = "linear_regression") -> PredictiveModel:
        """
        Build predictive model for performance forecasting.
        
        Args:
            target_metric: Metric to predict
            feature_metrics: Features to use for prediction
            model_type: Type of model ("linear_regression", "ridge", "polynomial")
            
        Returns:
            Trained predictive model
        """
        self.logger.info(f"Building {model_type} model to predict {target_metric}")
        
        if target_metric not in self.performance_data:
            raise ValueError(f"Target metric {target_metric} not found in data")
        
        # Select feature metrics
        if feature_metrics is None:
            feature_metrics = [m for m in self.performance_data.keys() 
                             if m != target_metric and len(self.performance_data[m]) >= 5]
        
        if not feature_metrics:
            raise ValueError("No suitable feature metrics found")
        
        model = PredictiveModel(
            model_name=f"{target_metric}_predictor",
            model_type=model_type,
            target_metric=target_metric,
            feature_metrics=feature_metrics
        )
        
        try:
            # Prepare training data
            target_values = self.performance_data[target_metric]
            
            # Align feature data
            min_samples = min(len(target_values), 
                            min(len(self.performance_data[m]) for m in feature_metrics))
            
            if min_samples < 5:
                raise ValueError("Insufficient data for model training")
            
            X = []
            for i in range(min_samples):
                feature_vector = []
                for feature in feature_metrics:
                    feature_vector.append(self.performance_data[feature][i])
                X.append(feature_vector)
            
            y = target_values[:min_samples]
            model.training_samples = min_samples
            
            if STATS_AVAILABLE:
                # Split data for validation
                if min_samples > 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                else:
                    X_train, X_test, y_train, y_test = X, X, y, y
                
                # Train model
                if model_type == "linear_regression":
                    sklearn_model = LinearRegression()
                elif model_type == "ridge":
                    sklearn_model = Ridge(alpha=1.0)
                else:
                    sklearn_model = LinearRegression()
                
                sklearn_model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred_train = sklearn_model.predict(X_train)
                y_pred_test = sklearn_model.predict(X_test)
                
                model.training_accuracy = r2_score(y_train, y_pred_train)
                model.validation_accuracy = r2_score(y_test, y_pred_test)
                model.mean_squared_error = mean_squared_error(y_test, y_pred_test)
                model.r_squared_score = model.validation_accuracy
                
                # Extract model parameters
                model.intercept = sklearn_model.intercept_
                for i, feature in enumerate(feature_metrics):
                    model.model_coefficients[feature] = sklearn_model.coef_[i]
                
                # Calculate feature importance (absolute coefficient values normalized)
                coeff_sum = sum(abs(coef) for coef in sklearn_model.coef_)
                if coeff_sum > 0:
                    for i, feature in enumerate(feature_metrics):
                        model.feature_importance[feature] = abs(sklearn_model.coef_[i]) / coeff_sum
            
            else:
                # Simple linear regression fallback
                if len(feature_metrics) == 1:
                    feature_values = [row[0] for row in X]
                    
                    # Calculate simple linear regression
                    n = len(feature_values)
                    sum_x = sum(feature_values)
                    sum_y = sum(y)
                    sum_xy = sum(x * y for x, y in zip(feature_values, y))
                    sum_x2 = sum(x * x for x in feature_values)
                    
                    if n * sum_x2 - sum_x * sum_x != 0:
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                        intercept = (sum_y - slope * sum_x) / n
                        
                        model.model_coefficients[feature_metrics[0]] = slope
                        model.intercept = intercept
                        
                        # Calculate R-squared
                        y_mean = statistics.mean(y)
                        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
                        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(feature_values, y))
                        
                        if ss_tot > 0:
                            model.r_squared_score = 1 - (ss_res / ss_tot)
                            model.training_accuracy = model.r_squared_score
                            model.validation_accuracy = model.r_squared_score
                        
                        model.feature_importance[feature_metrics[0]] = 1.0
            
            self.predictive_models[target_metric] = model
            self.logger.info(f"Built predictive model for {target_metric} (R²={model.r_squared_score:.3f})")
            
        except Exception as e:
            self.logger.error(f"Error building predictive model: {e}")
            raise
        
        return model
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all statistical analyses."""
        summary = {
            'data_summary': {
                'metrics_analyzed': list(self.performance_data.keys()),
                'total_data_points': sum(len(values) for values in self.performance_data.values()),
                'time_range_hours': (max(self.timestamps) - min(self.timestamps)) / 3600 if self.timestamps else 0,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'trend_analysis': {
                'metrics_with_trends': len(self.trend_analyses),
                'trend_summary': {}
            },
            'anomaly_detection': {
                'metrics_analyzed': len(self.anomaly_results),
                'total_anomalies': sum(result.anomaly_count for result in self.anomaly_results.values()),
                'anomaly_summary': {}
            },
            'correlation_analysis': {
                'correlation_matrix_available': self.correlation_matrix is not None,
                'strong_correlations_found': len(self.correlation_matrix.strong_correlations) if self.correlation_matrix else 0,
                'significant_correlations_found': len(self.correlation_matrix.significant_correlations) if self.correlation_matrix else 0
            },
            'predictive_models': {
                'models_built': len(self.predictive_models),
                'model_performance': {}
            }
        }
        
        # Trend analysis summary
        for metric, trend in self.trend_analyses.items():
            summary['trend_analysis']['trend_summary'][metric] = {
                'direction': trend.trend_direction.value,
                'strength': trend.trend_strength,
                'significance': trend.statistical_significance.value,
                'confidence': trend.trend_confidence
            }
        
        # Anomaly detection summary
        for metric, anomalies in self.anomaly_results.items():
            summary['anomaly_detection']['anomaly_summary'][metric] = {
                'anomaly_count': anomalies.anomaly_count,
                'anomaly_rate': anomalies.anomaly_rate,
                'detection_method': anomalies.detection_method
            }
        
        # Predictive model summary
        for metric, model in self.predictive_models.items():
            summary['predictive_models']['model_performance'][metric] = {
                'r_squared': model.r_squared_score,
                'training_accuracy': model.training_accuracy,
                'validation_accuracy': model.validation_accuracy,
                'model_type': model.model_type
            }
        
        return summary
    
    async def export_analysis_results(self, output_file: str) -> str:
        """Export all analysis results to JSON file."""
        try:
            export_data = {
                'export_metadata': {
                    'generated_timestamp': datetime.now().isoformat(),
                    'analyzer_version': '1.0.0',
                    'confidence_level': self.confidence_level,
                    'anomaly_sensitivity': self.anomaly_sensitivity
                },
                'analysis_summary': self.get_analysis_summary(),
                'trend_analyses': {metric: asdict(trend) for metric, trend in self.trend_analyses.items()},
                'anomaly_results': {metric: asdict(result) for metric, result in self.anomaly_results.items()},
                'correlation_matrix': asdict(self.correlation_matrix) if self.correlation_matrix else None,
                'predictive_models': {metric: asdict(model) for metric, model in self.predictive_models.items()}
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Analysis results exported to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis results: {e}")
            raise


# Convenience functions
async def analyze_performance_statistics(
    benchmark_data: Optional[List[Any]] = None,
    api_metrics_data: Optional[List[Any]] = None,
    metrics_to_analyze: Optional[List[str]] = None,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive statistical analysis.
    
    Args:
        benchmark_data: Quality validation benchmark data
        api_metrics_data: API usage and cost metrics
        metrics_to_analyze: Specific metrics to focus on
        confidence_level: Statistical confidence level
        
    Returns:
        Complete statistical analysis results
    """
    analyzer = StatisticalAnalyzer(confidence_level=confidence_level)
    
    await analyzer.load_performance_data(
        benchmark_data=benchmark_data,
        api_metrics_data=api_metrics_data
    )
    
    # Perform all analyses
    await analyzer.analyze_trends(metrics=metrics_to_analyze)
    await analyzer.detect_anomalies(metrics=metrics_to_analyze)
    await analyzer.calculate_correlations(metrics=metrics_to_analyze)
    
    # Build predictive models for key metrics
    if metrics_to_analyze:
        for metric in metrics_to_analyze[:3]:  # Limit to top 3 metrics
            try:
                await analyzer.build_predictive_model(target_metric=metric)
            except Exception:
                continue  # Skip if model building fails
    
    return analyzer.get_analysis_summary()


# Make main classes available at module level
__all__ = [
    'StatisticalAnalyzer',
    'TrendAnalysis',
    'AnomalyDetectionResult',
    'CorrelationMatrix',
    'PredictiveModel',
    'TrendDirection',
    'AnomalyType',
    'StatisticalSignificance',
    'analyze_performance_statistics'
]