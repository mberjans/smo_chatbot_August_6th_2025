#!/usr/bin/env python3
"""
Recommendation Engine for Clinical Metabolomics Oracle Performance Optimization.

This module analyzes performance data and generates actionable recommendations
for optimizing quality validation systems. It uses pattern recognition,
statistical analysis, and domain expertise to provide targeted suggestions
for performance improvements, cost optimization, and resource allocation.

Classes:
    - RecommendationType: Types of recommendations available
    - RecommendationPriority: Priority levels for recommendations
    - RecommendationCategory: Categories for organizing recommendations
    - PerformanceRecommendation: Individual recommendation data structure
    - RecommendationEngine: Main engine for generating recommendations
    - RecommendationValidator: Validates recommendation applicability

Key Features:
    - Performance bottleneck identification and resolution
    - Cost optimization recommendations
    - Resource allocation optimization
    - Quality improvement suggestions
    - Scalability recommendations
    - Preventive maintenance suggestions
    - ROI-based recommendation prioritization
    - Context-aware recommendation filtering

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
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import math

# Import statistical analysis tools
try:
    import numpy as np
    from scipy import stats
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of recommendations that can be generated."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    COST_REDUCTION = "cost_reduction"
    RESOURCE_ALLOCATION = "resource_allocation"
    QUALITY_IMPROVEMENT = "quality_improvement"
    SCALABILITY_ENHANCEMENT = "scalability_enhancement"
    ERROR_MITIGATION = "error_mitigation"
    CONFIGURATION_TUNING = "configuration_tuning"
    INFRASTRUCTURE_UPGRADE = "infrastructure_upgrade"
    PROCESS_AUTOMATION = "process_automation"
    MONITORING_ENHANCEMENT = "monitoring_enhancement"
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    CAPACITY_PLANNING = "capacity_planning"


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"             # Action needed within 1 week
    MEDIUM = "medium"         # Action needed within 1 month
    LOW = "low"               # Nice to have improvement
    DEFERRED = "deferred"     # Future consideration


class RecommendationCategory(Enum):
    """Categories for organizing recommendations."""
    IMMEDIATE_ACTION = "immediate_action"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    BUSINESS = "business"


class ImpactLevel(Enum):
    """Expected impact levels for recommendations."""
    TRANSFORMATIONAL = "transformational"  # > 50% improvement
    SIGNIFICANT = "significant"            # 20-50% improvement
    MODERATE = "moderate"                  # 5-20% improvement
    MINOR = "minor"                       # < 5% improvement
    UNKNOWN = "unknown"                   # Impact cannot be quantified


@dataclass
class PerformanceRecommendation:
    """Individual performance optimization recommendation."""
    
    # Identification
    recommendation_id: str = field(default_factory=lambda: f"rec_{int(time.time())}")
    title: str = "Performance Recommendation"
    description: str = ""
    
    # Classification
    recommendation_type: RecommendationType = RecommendationType.PERFORMANCE_OPTIMIZATION
    category: RecommendationCategory = RecommendationCategory.OPERATIONAL
    priority: RecommendationPriority = RecommendationPriority.MEDIUM
    
    # Impact assessment
    expected_impact_level: ImpactLevel = ImpactLevel.MODERATE
    estimated_improvement_percentage: Optional[float] = None
    affected_metrics: List[str] = field(default_factory=list)
    expected_outcomes: Dict[str, Union[float, str]] = field(default_factory=dict)
    
    # Implementation details
    implementation_complexity: str = "medium"  # "low", "medium", "high", "very_high"
    estimated_implementation_time_hours: Optional[int] = None
    required_resources: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    
    # Cost-benefit analysis
    implementation_cost_estimate_usd: Optional[float] = None
    expected_savings_annual_usd: Optional[float] = None
    roi_estimate_percentage: Optional[float] = None
    payback_period_months: Optional[int] = None
    
    # Applicability and constraints
    applicable_scenarios: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    constraints_and_limitations: List[str] = field(default_factory=list)
    risks_and_considerations: List[str] = field(default_factory=list)
    
    # Supporting evidence
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    data_confidence_level: float = 0.8  # 0.0 - 1.0
    recommendation_source: str = "automated_analysis"
    
    # Context and metadata
    generated_timestamp: float = field(default_factory=time.time)
    applicable_until_timestamp: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    related_recommendations: List[str] = field(default_factory=list)
    
    # Success metrics
    success_criteria: List[str] = field(default_factory=list)
    measurement_methods: List[str] = field(default_factory=list)
    review_timeline_days: int = 30
    
    def calculate_priority_score(self) -> float:
        """Calculate numerical priority score for ranking."""
        priority_weights = {
            RecommendationPriority.CRITICAL: 10,
            RecommendationPriority.HIGH: 8,
            RecommendationPriority.MEDIUM: 6,
            RecommendationPriority.LOW: 4,
            RecommendationPriority.DEFERRED: 2
        }
        
        impact_weights = {
            ImpactLevel.TRANSFORMATIONAL: 5,
            ImpactLevel.SIGNIFICANT: 4,
            ImpactLevel.MODERATE: 3,
            ImpactLevel.MINOR: 2,
            ImpactLevel.UNKNOWN: 1
        }
        
        complexity_weights = {
            "low": 1.5,
            "medium": 1.0,
            "high": 0.7,
            "very_high": 0.4
        }
        
        base_score = priority_weights.get(self.priority, 6)
        impact_score = impact_weights.get(self.expected_impact_level, 2)
        complexity_multiplier = complexity_weights.get(self.implementation_complexity, 1.0)
        confidence_multiplier = self.data_confidence_level
        
        # ROI boost
        roi_boost = 1.0
        if self.roi_estimate_percentage:
            roi_boost = min(2.0, 1.0 + self.roi_estimate_percentage / 100)
        
        return base_score * impact_score * complexity_multiplier * confidence_multiplier * roi_boost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        result = asdict(self)
        result['recommendation_type'] = self.recommendation_type.value
        result['category'] = self.category.value
        result['priority'] = self.priority.value
        result['expected_impact_level'] = self.expected_impact_level.value
        result['priority_score'] = self.calculate_priority_score()
        result['generated_timestamp_iso'] = datetime.fromtimestamp(self.generated_timestamp).isoformat()
        return result


class RecommendationEngine:
    """
    Intelligent recommendation engine for performance optimization.
    
    Analyzes performance data patterns and generates actionable recommendations
    for improving quality validation system performance, reducing costs,
    and optimizing resource utilization.
    """
    
    def __init__(self,
                 performance_thresholds: Optional[Dict[str, float]] = None,
                 cost_targets: Optional[Dict[str, float]] = None,
                 quality_targets: Optional[Dict[str, float]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            performance_thresholds: Performance metric thresholds
            cost_targets: Cost optimization targets
            quality_targets: Quality score targets
            logger: Logger instance for recommendation operations
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration thresholds
        self.performance_thresholds = performance_thresholds or {
            'response_time_ms': 2000,
            'throughput_ops_per_sec': 5.0,
            'error_rate_percent': 5.0,
            'memory_usage_mb': 1500,
            'cpu_usage_percent': 80.0,
            'quality_score_threshold': 85.0
        }
        
        self.cost_targets = cost_targets or {
            'cost_per_operation_usd': 0.01,
            'monthly_budget_usd': 1000.0,
            'cost_efficiency_target': 0.8
        }
        
        self.quality_targets = quality_targets or {
            'min_quality_score': 85.0,
            'target_accuracy_rate': 90.0,
            'min_confidence_level': 80.0
        }
        
        # Data storage
        self.benchmark_data: List[Any] = []
        self.api_metrics_data: List[Any] = []
        self.correlation_data: List[Any] = []
        self.historical_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Generated recommendations
        self.recommendations: List[PerformanceRecommendation] = []
        self.recommendation_templates: Dict[str, Dict[str, Any]] = self._load_recommendation_templates()
        
        # Analysis results cache
        self.analysis_cache: Dict[str, Any] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        self.logger.info("RecommendationEngine initialized")
    
    def _load_recommendation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined recommendation templates."""
        return {
            "high_response_time": {
                "title": "Optimize Response Time Performance",
                "type": RecommendationType.PERFORMANCE_OPTIMIZATION,
                "category": RecommendationCategory.IMMEDIATE_ACTION,
                "implementation_steps": [
                    "Identify performance bottlenecks through profiling",
                    "Implement caching for frequently accessed data",
                    "Optimize database queries and indexing",
                    "Add parallel processing for independent operations",
                    "Consider load balancing for high-traffic scenarios"
                ],
                "success_criteria": [
                    "Response time reduced by at least 25%",
                    "95th percentile latency within threshold",
                    "User satisfaction improvement measured"
                ]
            },
            
            "high_error_rate": {
                "title": "Reduce System Error Rate",
                "type": RecommendationType.ERROR_MITIGATION,
                "category": RecommendationCategory.IMMEDIATE_ACTION,
                "implementation_steps": [
                    "Analyze error logs to identify root causes",
                    "Implement robust error handling and retry logic",
                    "Add input validation and sanitization",
                    "Improve system monitoring and alerting",
                    "Create error recovery procedures"
                ],
                "success_criteria": [
                    "Error rate reduced below 2%",
                    "Mean time to recovery (MTTR) decreased",
                    "System reliability improved"
                ]
            },
            
            "high_cost_per_operation": {
                "title": "Optimize API Cost Efficiency",
                "type": RecommendationType.COST_REDUCTION,
                "category": RecommendationCategory.SHORT_TERM,
                "implementation_steps": [
                    "Analyze cost patterns and identify expensive operations",
                    "Implement intelligent request batching",
                    "Add result caching to reduce redundant API calls",
                    "Optimize prompt engineering for token efficiency",
                    "Consider alternative models or providers for cost savings"
                ],
                "success_criteria": [
                    "Cost per operation reduced by at least 30%",
                    "Monthly API costs within budget",
                    "Maintained or improved quality metrics"
                ]
            },
            
            "low_quality_scores": {
                "title": "Improve Quality Validation Accuracy",
                "type": RecommendationType.QUALITY_IMPROVEMENT,
                "category": RecommendationCategory.OPERATIONAL,
                "implementation_steps": [
                    "Review and enhance training datasets",
                    "Fine-tune validation model parameters",
                    "Implement ensemble validation methods",
                    "Add human-in-the-loop validation for edge cases",
                    "Continuously monitor and adjust quality thresholds"
                ],
                "success_criteria": [
                    "Quality scores consistently above 85%",
                    "Validation accuracy improved by 15%",
                    "Reduced false positive/negative rates"
                ]
            },
            
            "memory_pressure": {
                "title": "Optimize Memory Usage",
                "type": RecommendationType.RESOURCE_ALLOCATION,
                "category": RecommendationCategory.TECHNICAL,
                "implementation_steps": [
                    "Profile memory usage patterns and identify leaks",
                    "Implement memory pooling for frequent operations",
                    "Optimize data structures and algorithms",
                    "Add garbage collection tuning",
                    "Consider streaming processing for large datasets"
                ],
                "success_criteria": [
                    "Peak memory usage reduced by 20%",
                    "Memory growth rate stabilized",
                    "System stability improved under load"
                ]
            },
            
            "scaling_bottleneck": {
                "title": "Enhance System Scalability",
                "type": RecommendationType.SCALABILITY_ENHANCEMENT,
                "category": RecommendationCategory.STRATEGIC,
                "implementation_steps": [
                    "Identify scalability bottlenecks through load testing",
                    "Implement horizontal scaling capabilities",
                    "Add load balancing and service discovery",
                    "Optimize data partitioning strategies",
                    "Design for cloud-native scalability"
                ],
                "success_criteria": [
                    "System handles 3x current load without degradation",
                    "Auto-scaling policies implemented and tested",
                    "Performance maintained during scaling events"
                ]
            }
        }
    
    async def load_performance_data(self,
                                  benchmark_data: Optional[List[Any]] = None,
                                  api_metrics_data: Optional[List[Any]] = None,
                                  correlation_data: Optional[List[Any]] = None) -> int:
        """
        Load performance data for recommendation analysis.
        
        Args:
            benchmark_data: Quality validation benchmark data
            api_metrics_data: API usage and cost metrics
            correlation_data: System correlation analysis data
            
        Returns:
            Total number of data points loaded
        """
        total_loaded = 0
        
        try:
            if benchmark_data:
                self.benchmark_data = benchmark_data
                total_loaded += len(benchmark_data)
                self.logger.info(f"Loaded {len(benchmark_data)} benchmark data points")
            
            if api_metrics_data:
                self.api_metrics_data = api_metrics_data
                total_loaded += len(api_metrics_data)
                self.logger.info(f"Loaded {len(api_metrics_data)} API metrics data points")
            
            if correlation_data:
                self.correlation_data = correlation_data
                total_loaded += len(correlation_data)
                self.logger.info(f"Loaded {len(correlation_data)} correlation data points")
            
            # Update historical trends
            await self._update_historical_trends()
            
            # Clear cache to force fresh analysis
            self.analysis_cache.clear()
            
            self.logger.info(f"Total data points loaded: {total_loaded}")
            
        except Exception as e:
            self.logger.error(f"Error loading performance data: {e}")
            raise
        
        return total_loaded
    
    async def generate_recommendations(self,
                                     focus_areas: Optional[List[RecommendationType]] = None,
                                     max_recommendations: int = 20,
                                     min_confidence_level: float = 0.6) -> List[PerformanceRecommendation]:
        """
        Generate performance optimization recommendations.
        
        Args:
            focus_areas: Specific recommendation types to focus on
            max_recommendations: Maximum number of recommendations to generate
            min_confidence_level: Minimum confidence level for recommendations
            
        Returns:
            List of prioritized performance recommendations
        """
        self.logger.info(f"Generating recommendations with focus areas: {focus_areas}")
        
        start_time = time.time()
        self.recommendations.clear()
        
        try:
            # Analyze current performance state
            performance_analysis = await self._analyze_performance_state()
            
            # Generate specific recommendation types
            if not focus_areas:
                focus_areas = list(RecommendationType)
            
            for recommendation_type in focus_areas:
                type_recommendations = await self._generate_recommendations_by_type(
                    recommendation_type, performance_analysis
                )
                self.recommendations.extend(type_recommendations)
            
            # Filter by confidence level
            high_confidence_recommendations = [
                rec for rec in self.recommendations 
                if rec.data_confidence_level >= min_confidence_level
            ]
            
            # Calculate cross-recommendation relationships
            await self._identify_recommendation_relationships(high_confidence_recommendations)
            
            # Prioritize and rank recommendations
            prioritized_recommendations = self._prioritize_recommendations(high_confidence_recommendations)
            
            # Limit to max recommendations
            final_recommendations = prioritized_recommendations[:max_recommendations]
            
            generation_time = time.time() - start_time
            self.logger.info(f"Generated {len(final_recommendations)} recommendations in {generation_time:.2f} seconds")
            
            return final_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            raise
    
    async def _analyze_performance_state(self) -> Dict[str, Any]:
        """Analyze current system performance state."""
        cache_key = "performance_state_analysis"
        
        # Check cache
        if cache_key in self.analysis_cache:
            cache_entry = self.analysis_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl_seconds:
                return cache_entry['data']
        
        analysis = {
            'response_time_analysis': {},
            'throughput_analysis': {},
            'quality_analysis': {},
            'cost_analysis': {},
            'resource_analysis': {},
            'error_analysis': {},
            'trend_analysis': {},
            'bottleneck_analysis': {}
        }
        
        # Response time analysis
        if self.benchmark_data:
            response_times = [getattr(m, 'average_latency_ms', 0) for m in self.benchmark_data if hasattr(m, 'average_latency_ms') and m.average_latency_ms > 0]
            if response_times:
                analysis['response_time_analysis'] = {
                    'current_average': statistics.mean(response_times),
                    'threshold': self.performance_thresholds['response_time_ms'],
                    'threshold_exceeded': statistics.mean(response_times) > self.performance_thresholds['response_time_ms'],
                    'p95_latency': sorted(response_times)[int(len(response_times) * 0.95)],
                    'variance': statistics.stdev(response_times) ** 2 if len(response_times) > 1 else 0,
                    'trend': self._calculate_metric_trend(response_times)
                }
        
        # Quality analysis
        if self.benchmark_data:
            quality_scores = []
            for m in self.benchmark_data:
                if hasattr(m, 'calculate_quality_efficiency_score'):
                    quality_scores.append(m.calculate_quality_efficiency_score())
                elif hasattr(m, 'quality_efficiency_score'):
                    quality_scores.append(m.quality_efficiency_score)
            
            if quality_scores:
                analysis['quality_analysis'] = {
                    'current_average': statistics.mean(quality_scores),
                    'threshold': self.quality_targets['min_quality_score'],
                    'below_threshold': statistics.mean(quality_scores) < self.quality_targets['min_quality_score'],
                    'consistency': 100 - (statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0),
                    'trend': self._calculate_metric_trend(quality_scores)
                }
        
        # Cost analysis
        if self.api_metrics_data:
            costs = [getattr(m, 'cost_usd', 0) for m in self.api_metrics_data if hasattr(m, 'cost_usd') and m.cost_usd > 0]
            if costs:
                analysis['cost_analysis'] = {
                    'current_average_cost': statistics.mean(costs),
                    'target_cost': self.cost_targets['cost_per_operation_usd'],
                    'above_target': statistics.mean(costs) > self.cost_targets['cost_per_operation_usd'],
                    'total_cost': sum(costs),
                    'cost_variance': statistics.stdev(costs) ** 2 if len(costs) > 1 else 0,
                    'trend': self._calculate_metric_trend(costs)
                }
        
        # Error rate analysis
        if self.benchmark_data:
            error_rates = [getattr(m, 'error_rate_percent', 0) for m in self.benchmark_data if hasattr(m, 'error_rate_percent')]
            if error_rates:
                analysis['error_analysis'] = {
                    'current_error_rate': statistics.mean(error_rates),
                    'threshold': self.performance_thresholds['error_rate_percent'],
                    'above_threshold': statistics.mean(error_rates) > self.performance_thresholds['error_rate_percent'],
                    'max_error_rate': max(error_rates),
                    'trend': self._calculate_metric_trend(error_rates)
                }
        
        # Resource analysis
        if self.benchmark_data:
            memory_usage = [getattr(m, 'peak_validation_memory_mb', 0) for m in self.benchmark_data if hasattr(m, 'peak_validation_memory_mb') and m.peak_validation_memory_mb > 0]
            cpu_usage = [getattr(m, 'avg_validation_cpu_percent', 0) for m in self.benchmark_data if hasattr(m, 'avg_validation_cpu_percent') and m.avg_validation_cpu_percent > 0]
            
            analysis['resource_analysis'] = {
                'memory_analysis': {
                    'current_average': statistics.mean(memory_usage) if memory_usage else 0,
                    'threshold': self.performance_thresholds['memory_usage_mb'],
                    'above_threshold': statistics.mean(memory_usage) > self.performance_thresholds['memory_usage_mb'] if memory_usage else False,
                    'peak_usage': max(memory_usage) if memory_usage else 0
                },
                'cpu_analysis': {
                    'current_average': statistics.mean(cpu_usage) if cpu_usage else 0,
                    'threshold': self.performance_thresholds['cpu_usage_percent'],
                    'above_threshold': statistics.mean(cpu_usage) > self.performance_thresholds['cpu_usage_percent'] if cpu_usage else False
                }
            }
        
        # Cache the analysis
        self.analysis_cache[cache_key] = {
            'data': analysis,
            'timestamp': time.time()
        }
        
        return analysis
    
    def _calculate_metric_trend(self, values: List[float], window_size: int = 10) -> str:
        """Calculate trend direction for a metric."""
        if len(values) < window_size:
            return "insufficient_data"
        
        recent_values = values[-window_size:]
        older_values = values[-2*window_size:-window_size] if len(values) >= 2*window_size else values[:-window_size]
        
        if not older_values:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent_values)
        older_avg = statistics.mean(older_values)
        
        change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
        
        if abs(change_percent) < 5:
            return "stable"
        elif change_percent > 0:
            return "increasing"
        else:
            return "decreasing"
    
    async def _generate_recommendations_by_type(self,
                                              recommendation_type: RecommendationType,
                                              performance_analysis: Dict[str, Any]) -> List[PerformanceRecommendation]:
        """Generate recommendations for a specific type."""
        recommendations = []
        
        try:
            if recommendation_type == RecommendationType.PERFORMANCE_OPTIMIZATION:
                recommendations.extend(await self._generate_performance_recommendations(performance_analysis))
            
            elif recommendation_type == RecommendationType.COST_REDUCTION:
                recommendations.extend(await self._generate_cost_recommendations(performance_analysis))
            
            elif recommendation_type == RecommendationType.QUALITY_IMPROVEMENT:
                recommendations.extend(await self._generate_quality_recommendations(performance_analysis))
            
            elif recommendation_type == RecommendationType.RESOURCE_ALLOCATION:
                recommendations.extend(await self._generate_resource_recommendations(performance_analysis))
            
            elif recommendation_type == RecommendationType.ERROR_MITIGATION:
                recommendations.extend(await self._generate_error_mitigation_recommendations(performance_analysis))
            
            elif recommendation_type == RecommendationType.SCALABILITY_ENHANCEMENT:
                recommendations.extend(await self._generate_scalability_recommendations(performance_analysis))
            
            # Add more recommendation types as needed
            
        except Exception as e:
            self.logger.error(f"Error generating {recommendation_type.value} recommendations: {e}")
        
        return recommendations
    
    async def _generate_performance_recommendations(self, analysis: Dict[str, Any]) -> List[PerformanceRecommendation]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # High response time recommendation
        rt_analysis = analysis.get('response_time_analysis', {})
        if rt_analysis.get('threshold_exceeded', False):
            current_avg = rt_analysis['current_average']
            threshold = rt_analysis['threshold']
            excess_percent = ((current_avg - threshold) / threshold) * 100
            
            template = self.recommendation_templates["high_response_time"]
            
            recommendation = PerformanceRecommendation(
                title=template["title"],
                description=f"Average response time ({current_avg:.1f}ms) exceeds threshold ({threshold}ms) by {excess_percent:.1f}%. This impacts user experience and system throughput.",
                recommendation_type=template["type"],
                category=template["category"],
                priority=RecommendationPriority.HIGH if excess_percent > 50 else RecommendationPriority.MEDIUM,
                expected_impact_level=ImpactLevel.SIGNIFICANT if excess_percent > 50 else ImpactLevel.MODERATE,
                estimated_improvement_percentage=min(50, excess_percent),
                affected_metrics=['response_time_ms', 'throughput_ops_per_sec', 'user_satisfaction'],
                implementation_steps=template["implementation_steps"],
                success_criteria=template["success_criteria"],
                implementation_complexity="medium",
                estimated_implementation_time_hours=40,
                implementation_cost_estimate_usd=5000,
                expected_savings_annual_usd=8000,
                roi_estimate_percentage=60,
                supporting_data={
                    'current_response_time': current_avg,
                    'threshold': threshold,
                    'excess_percentage': excess_percent,
                    'p95_latency': rt_analysis.get('p95_latency', 0),
                    'trend': rt_analysis.get('trend', 'unknown')
                },
                data_confidence_level=0.9,
                tags=['response_time', 'performance', 'user_experience']
            )
            
            recommendations.append(recommendation)
        
        # Add more performance-specific recommendations based on other metrics
        
        return recommendations
    
    async def _generate_cost_recommendations(self, analysis: Dict[str, Any]) -> List[PerformanceRecommendation]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        cost_analysis = analysis.get('cost_analysis', {})
        if cost_analysis.get('above_target', False):
            current_cost = cost_analysis['current_average_cost']
            target_cost = cost_analysis['target_cost']
            excess_percent = ((current_cost - target_cost) / target_cost) * 100
            
            template = self.recommendation_templates["high_cost_per_operation"]
            
            recommendation = PerformanceRecommendation(
                title=template["title"],
                description=f"Average cost per operation (${current_cost:.4f}) exceeds target (${target_cost:.4f}) by {excess_percent:.1f}%. Implementing cost optimization strategies could significantly reduce operational expenses.",
                recommendation_type=template["type"],
                category=template["category"],
                priority=RecommendationPriority.HIGH if excess_percent > 100 else RecommendationPriority.MEDIUM,
                expected_impact_level=ImpactLevel.SIGNIFICANT,
                estimated_improvement_percentage=min(40, excess_percent),
                affected_metrics=['cost_usd', 'cost_per_operation', 'budget_utilization'],
                implementation_steps=template["implementation_steps"],
                success_criteria=template["success_criteria"],
                implementation_complexity="medium",
                estimated_implementation_time_hours=32,
                implementation_cost_estimate_usd=3000,
                expected_savings_annual_usd=cost_analysis.get('total_cost', 0) * 0.3,
                roi_estimate_percentage=200,
                supporting_data={
                    'current_cost': current_cost,
                    'target_cost': target_cost,
                    'excess_percentage': excess_percent,
                    'total_monthly_cost': cost_analysis.get('total_cost', 0),
                    'trend': cost_analysis.get('trend', 'unknown')
                },
                data_confidence_level=0.85,
                tags=['cost_optimization', 'budget', 'efficiency']
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_quality_recommendations(self, analysis: Dict[str, Any]) -> List[PerformanceRecommendation]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        quality_analysis = analysis.get('quality_analysis', {})
        if quality_analysis.get('below_threshold', False):
            current_quality = quality_analysis['current_average']
            threshold = quality_analysis['threshold']
            deficit_percent = ((threshold - current_quality) / threshold) * 100
            
            template = self.recommendation_templates["low_quality_scores"]
            
            recommendation = PerformanceRecommendation(
                title=template["title"],
                description=f"Average quality score ({current_quality:.1f}) is below target ({threshold}). Improving validation accuracy is critical for maintaining system reliability.",
                recommendation_type=template["type"],
                category=template["category"],
                priority=RecommendationPriority.HIGH,
                expected_impact_level=ImpactLevel.SIGNIFICANT,
                estimated_improvement_percentage=deficit_percent,
                affected_metrics=['quality_score', 'accuracy_rate', 'user_trust'],
                implementation_steps=template["implementation_steps"],
                success_criteria=template["success_criteria"],
                implementation_complexity="high",
                estimated_implementation_time_hours=60,
                implementation_cost_estimate_usd=8000,
                expected_savings_annual_usd=15000,  # From reduced errors and improved reliability
                roi_estimate_percentage=87,
                supporting_data={
                    'current_quality': current_quality,
                    'target_quality': threshold,
                    'deficit_percentage': deficit_percent,
                    'consistency_score': quality_analysis.get('consistency', 0),
                    'trend': quality_analysis.get('trend', 'unknown')
                },
                data_confidence_level=0.8,
                tags=['quality', 'accuracy', 'reliability']
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_resource_recommendations(self, analysis: Dict[str, Any]) -> List[PerformanceRecommendation]:
        """Generate resource allocation recommendations."""
        recommendations = []
        
        resource_analysis = analysis.get('resource_analysis', {})
        
        # Memory optimization
        memory_analysis = resource_analysis.get('memory_analysis', {})
        if memory_analysis.get('above_threshold', False):
            current_usage = memory_analysis['current_average']
            threshold = memory_analysis['threshold']
            
            template = self.recommendation_templates["memory_pressure"]
            
            recommendation = PerformanceRecommendation(
                title=template["title"],
                description=f"Memory usage ({current_usage:.1f}MB) exceeds threshold ({threshold}MB). Optimizing memory usage will improve system stability and performance.",
                recommendation_type=RecommendationType.RESOURCE_ALLOCATION,
                category=template["category"],
                priority=RecommendationPriority.MEDIUM,
                expected_impact_level=ImpactLevel.MODERATE,
                estimated_improvement_percentage=25,
                affected_metrics=['memory_usage_mb', 'system_stability', 'performance'],
                implementation_steps=template["implementation_steps"],
                success_criteria=template["success_criteria"],
                implementation_complexity="medium",
                estimated_implementation_time_hours=24,
                supporting_data={
                    'current_memory_usage': current_usage,
                    'threshold': threshold,
                    'peak_usage': memory_analysis.get('peak_usage', 0)
                },
                data_confidence_level=0.8,
                tags=['memory', 'resources', 'optimization']
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_error_mitigation_recommendations(self, analysis: Dict[str, Any]) -> List[PerformanceRecommendation]:
        """Generate error mitigation recommendations."""
        recommendations = []
        
        error_analysis = analysis.get('error_analysis', {})
        if error_analysis.get('above_threshold', False):
            current_error_rate = error_analysis['current_error_rate']
            threshold = error_analysis['threshold']
            
            template = self.recommendation_templates["high_error_rate"]
            
            recommendation = PerformanceRecommendation(
                title=template["title"],
                description=f"Error rate ({current_error_rate:.2f}%) exceeds acceptable threshold ({threshold}%). Reducing errors is critical for system reliability.",
                recommendation_type=template["type"],
                category=template["category"],
                priority=RecommendationPriority.CRITICAL if current_error_rate > threshold * 2 else RecommendationPriority.HIGH,
                expected_impact_level=ImpactLevel.SIGNIFICANT,
                estimated_improvement_percentage=60,
                affected_metrics=['error_rate_percent', 'reliability', 'user_satisfaction'],
                implementation_steps=template["implementation_steps"],
                success_criteria=template["success_criteria"],
                implementation_complexity="medium",
                estimated_implementation_time_hours=36,
                supporting_data={
                    'current_error_rate': current_error_rate,
                    'threshold': threshold,
                    'max_error_rate': error_analysis.get('max_error_rate', 0),
                    'trend': error_analysis.get('trend', 'unknown')
                },
                data_confidence_level=0.9,
                tags=['errors', 'reliability', 'stability']
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_scalability_recommendations(self, analysis: Dict[str, Any]) -> List[PerformanceRecommendation]:
        """Generate scalability enhancement recommendations."""
        recommendations = []
        
        # Check for scalability indicators
        rt_analysis = analysis.get('response_time_analysis', {})
        resource_analysis = analysis.get('resource_analysis', {})
        
        # High variance in response times may indicate scaling issues
        if rt_analysis.get('variance', 0) > 1000000:  # High variance threshold
            template = self.recommendation_templates["scaling_bottleneck"]
            
            recommendation = PerformanceRecommendation(
                title=template["title"],
                description="High variance in response times suggests potential scalability bottlenecks. Implementing scalability enhancements will ensure consistent performance under varying loads.",
                recommendation_type=template["type"],
                category=template["category"],
                priority=RecommendationPriority.MEDIUM,
                expected_impact_level=ImpactLevel.SIGNIFICANT,
                estimated_improvement_percentage=40,
                affected_metrics=['scalability', 'response_time_consistency', 'throughput'],
                implementation_steps=template["implementation_steps"],
                success_criteria=template["success_criteria"],
                implementation_complexity="high",
                estimated_implementation_time_hours=80,
                implementation_cost_estimate_usd=15000,
                expected_savings_annual_usd=25000,
                roi_estimate_percentage=67,
                supporting_data={
                    'response_time_variance': rt_analysis.get('variance', 0),
                    'resource_pressure': resource_analysis.get('memory_analysis', {}).get('above_threshold', False)
                },
                data_confidence_level=0.7,
                tags=['scalability', 'architecture', 'performance']
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _identify_recommendation_relationships(self, recommendations: List[PerformanceRecommendation]) -> None:
        """Identify relationships between recommendations."""
        for i, rec1 in enumerate(recommendations):
            for j, rec2 in enumerate(recommendations[i+1:], i+1):
                # Check for overlapping affected metrics
                common_metrics = set(rec1.affected_metrics) & set(rec2.affected_metrics)
                if common_metrics:
                    rec1.related_recommendations.append(rec2.recommendation_id)
                    rec2.related_recommendations.append(rec1.recommendation_id)
                
                # Check for conflicting recommendations
                if self._are_recommendations_conflicting(rec1, rec2):
                    rec1.constraints_and_limitations.append(f"May conflict with recommendation: {rec2.title}")
                    rec2.constraints_and_limitations.append(f"May conflict with recommendation: {rec1.title}")
    
    def _are_recommendations_conflicting(self, rec1: PerformanceRecommendation, rec2: PerformanceRecommendation) -> bool:
        """Check if two recommendations might conflict."""
        # Simple heuristic: recommendations that affect same metrics with different approaches may conflict
        common_metrics = set(rec1.affected_metrics) & set(rec2.affected_metrics)
        if common_metrics and rec1.recommendation_type != rec2.recommendation_type:
            return True
        return False
    
    def _prioritize_recommendations(self, recommendations: List[PerformanceRecommendation]) -> List[PerformanceRecommendation]:
        """Prioritize recommendations based on impact, urgency, and feasibility."""
        # Calculate priority scores
        for rec in recommendations:
            rec.priority_score = rec.calculate_priority_score()
        
        # Sort by priority score (descending)
        prioritized = sorted(recommendations, key=lambda r: r.priority_score, reverse=True)
        
        # Group by priority level for final ordering
        priority_groups = defaultdict(list)
        for rec in prioritized:
            priority_groups[rec.priority].append(rec)
        
        # Final ordering: Critical -> High -> Medium -> Low -> Deferred
        final_order = []
        for priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH, 
                        RecommendationPriority.MEDIUM, RecommendationPriority.LOW, 
                        RecommendationPriority.DEFERRED]:
            group = priority_groups[priority]
            # Within each priority group, sort by priority score
            group.sort(key=lambda r: r.priority_score, reverse=True)
            final_order.extend(group)
        
        return final_order
    
    async def _update_historical_trends(self) -> None:
        """Update historical trends for pattern analysis."""
        # Extract time-series data for key metrics
        if self.benchmark_data:
            for metric_name in ['average_latency_ms', 'error_rate_percent', 'quality_efficiency_score']:
                values = []
                for data_point in self.benchmark_data:
                    if hasattr(data_point, metric_name):
                        value = getattr(data_point, metric_name)
                        if value is not None:
                            values.append(value)
                
                if values:
                    self.historical_trends[metric_name] = values[-50:]  # Keep last 50 values
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of generated recommendations."""
        if not self.recommendations:
            return {'status': 'no_recommendations'}
        
        summary = {
            'total_recommendations': len(self.recommendations),
            'by_priority': {},
            'by_type': {},
            'by_category': {},
            'by_impact_level': {},
            'total_estimated_savings': 0.0,
            'total_implementation_cost': 0.0,
            'average_roi': 0.0,
            'high_confidence_count': 0
        }
        
        # Count by various dimensions
        for rec in self.recommendations:
            # By priority
            priority = rec.priority.value
            summary['by_priority'][priority] = summary['by_priority'].get(priority, 0) + 1
            
            # By type
            rec_type = rec.recommendation_type.value
            summary['by_type'][rec_type] = summary['by_type'].get(rec_type, 0) + 1
            
            # By category
            category = rec.category.value
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # By impact level
            impact = rec.expected_impact_level.value
            summary['by_impact_level'][impact] = summary['by_impact_level'].get(impact, 0) + 1
            
            # Financial metrics
            if rec.expected_savings_annual_usd:
                summary['total_estimated_savings'] += rec.expected_savings_annual_usd
            
            if rec.implementation_cost_estimate_usd:
                summary['total_implementation_cost'] += rec.implementation_cost_estimate_usd
            
            # High confidence recommendations
            if rec.data_confidence_level >= 0.8:
                summary['high_confidence_count'] += 1
        
        # Calculate average ROI
        roi_values = [rec.roi_estimate_percentage for rec in self.recommendations if rec.roi_estimate_percentage]
        if roi_values:
            summary['average_roi'] = statistics.mean(roi_values)
        
        return summary
    
    async def export_recommendations(self, output_file: str) -> str:
        """Export recommendations to JSON file."""
        try:
            export_data = {
                'export_metadata': {
                    'generated_timestamp': datetime.now().isoformat(),
                    'total_recommendations': len(self.recommendations),
                    'engine_version': '1.0.0'
                },
                'summary': self.get_recommendation_summary(),
                'recommendations': [rec.to_dict() for rec in self.recommendations]
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Recommendations exported to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error exporting recommendations: {e}")
            raise


# Convenience functions
async def generate_performance_recommendations(
    benchmark_data: Optional[List[Any]] = None,
    api_metrics_data: Optional[List[Any]] = None,
    correlation_data: Optional[List[Any]] = None,
    focus_areas: Optional[List[RecommendationType]] = None,
    performance_thresholds: Optional[Dict[str, float]] = None
) -> List[PerformanceRecommendation]:
    """
    Convenience function to generate performance recommendations.
    
    Args:
        benchmark_data: Quality validation benchmark data
        api_metrics_data: API usage and cost metrics
        correlation_data: System correlation analysis data
        focus_areas: Specific recommendation types to focus on
        performance_thresholds: Performance metric thresholds
        
    Returns:
        List of prioritized performance recommendations
    """
    engine = RecommendationEngine(performance_thresholds=performance_thresholds)
    
    await engine.load_performance_data(
        benchmark_data=benchmark_data,
        api_metrics_data=api_metrics_data,
        correlation_data=correlation_data
    )
    
    return await engine.generate_recommendations(focus_areas=focus_areas)


# Make main classes available at module level
__all__ = [
    'RecommendationEngine',
    'PerformanceRecommendation',
    'RecommendationType',
    'RecommendationPriority',
    'RecommendationCategory',
    'ImpactLevel',
    'generate_performance_recommendations'
]