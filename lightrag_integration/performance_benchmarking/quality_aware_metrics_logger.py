#!/usr/bin/env python3
"""
Quality-Aware API Metrics Logger for Clinical Metabolomics Oracle LightRAG Integration.

This module extends the existing APIUsageMetricsLogger with quality validation specific
tracking capabilities. It provides comprehensive API cost tracking and monitoring utilities
for quality validation operations, including detailed context managers and aggregators
for quality-specific metrics.

Classes:
    - QualityMetricType: Extended metric types for quality operations
    - QualityAPIMetric: Extended API metric with quality validation context
    - QualityMetricsAggregator: Quality-specific metrics aggregation
    - QualityAwareAPIMetricsLogger: Main logger extending APIUsageMetricsLogger

Key Features:
    - Quality validation operation tracking with context managers
    - Enhanced tracking for different quality validation stages
    - API cost tracking specific to quality validation operations
    - Integration with existing comprehensive API tracking system
    - Backward compatibility with existing MetricType and APIMetric models
    - Quality validation performance analysis and reporting

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: Quality Validation Performance Benchmarking
"""

import json
import time
import threading
import logging
import psutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter, deque
import uuid
import statistics

# Import parent API metrics logger
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from api_metrics_logger import (
        MetricType, APIMetric, APIUsageMetricsLogger, MetricsAggregator
    )
    from cost_persistence import CostRecord, ResearchCategory, CostPersistence
    from budget_manager import BudgetManager, BudgetAlert, AlertLevel
    from research_categorizer import ResearchCategorizer
    from audit_trail import AuditTrail
except ImportError as e:
    # Create mock classes for standalone operation
    print(f"Warning: Some dependencies not available ({e}). Using mock implementations.")
    
    from enum import Enum
    
    class MetricType(Enum):
        LLM_CALL = "llm_call"
        EMBEDDING_CALL = "embedding_call"
        HYBRID_OPERATION = "hybrid_operation"
        RESPONSE_TIME = "response_time"
        THROUGHPUT = "throughput"
        ERROR_RATE = "error_rate"
        TOKEN_USAGE = "token_usage"
        COST_TRACKING = "cost_tracking"
        BUDGET_UTILIZATION = "budget_utilization"
        RESEARCH_CATEGORY = "research_category"
        KNOWLEDGE_EXTRACTION = "knowledge_extraction"
        DOCUMENT_PROCESSING = "document_processing"
        MEMORY_USAGE = "memory_usage"
        CONCURRENT_OPERATIONS = "concurrent_operations"
        RETRY_PATTERNS = "retry_patterns"
    
    @dataclass
    class APIMetric:
        id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
        timestamp: float = field(default_factory=time.time)
        session_id: Optional[str] = None
        metric_type: MetricType = MetricType.LLM_CALL
        operation_name: str = "unknown"
        model_name: Optional[str] = None
        api_provider: str = "openai"
        endpoint_used: Optional[str] = None
        prompt_tokens: int = 0
        completion_tokens: int = 0
        embedding_tokens: int = 0
        total_tokens: int = 0
        cost_usd: float = 0.0
        cost_per_token: Optional[float] = None
        response_time_ms: Optional[float] = None
        queue_time_ms: Optional[float] = None
        processing_time_ms: Optional[float] = None
        throughput_tokens_per_sec: Optional[float] = None
        success: bool = True
        error_type: Optional[str] = None
        error_message: Optional[str] = None
        retry_count: int = 0
        final_attempt: bool = True
        research_category: str = "general_query"
        query_type: Optional[str] = None
        subject_area: Optional[str] = None
        document_type: Optional[str] = None
        memory_usage_mb: Optional[float] = None
        cpu_usage_percent: Optional[float] = None
        concurrent_operations: int = 1
        request_size_bytes: Optional[int] = None
        response_size_bytes: Optional[int] = None
        context_length: Optional[int] = None
        temperature_used: Optional[float] = None
        daily_budget_used_percent: Optional[float] = None
        monthly_budget_used_percent: Optional[float] = None
        compliance_level: str = "standard"
        user_id: Optional[str] = None
        project_id: Optional[str] = None
        experiment_id: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
        tags: List[str] = field(default_factory=list)
        
        def __post_init__(self):
            if self.total_tokens == 0:
                self.total_tokens = self.prompt_tokens + self.completion_tokens + self.embedding_tokens
        
        def to_dict(self) -> Dict[str, Any]:
            result = asdict(self)
            result['metric_type'] = self.metric_type.value
            result['timestamp_iso'] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
            return result
    
    class MetricsAggregator:
        def __init__(self, logger: Optional[logging.Logger] = None):
            self.logger = logger or logging.getLogger(__name__)
            self._lock = threading.Lock()
            self._metrics_buffer = []
        
        def add_metric(self, metric):
            with self._lock:
                self._metrics_buffer.append(metric)
        
        def get_current_stats(self) -> Dict[str, Any]:
            return {'buffer_size': len(self._metrics_buffer)}
    
    class APIUsageMetricsLogger:
        def __init__(self, config=None, cost_persistence=None, budget_manager=None, 
                     research_categorizer=None, audit_trail=None, logger=None):
            self.config = config
            self.cost_persistence = cost_persistence
            self.budget_manager = budget_manager
            self.research_categorizer = research_categorizer
            self.audit_trail = audit_trail
            self.logger = logger or logging.getLogger(__name__)
            self.metrics_aggregator = MetricsAggregator(self.logger)
            self._lock = threading.Lock()
            self.session_id = str(uuid.uuid4())
            self.start_time = time.time()
            self._active_operations = {}
            self._operation_counter = 0
            
            # Mock audit logger
            self.audit_logger = self.logger
        
        def _get_memory_usage(self) -> float:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except:
                return 0.0
        
        def _log_metric(self, metric):
            self.logger.info(f"Metric logged: {metric.operation_name}")
        
        def _integrate_with_cost_systems(self, metric):
            pass
        
        def _log_audit_trail(self, metric):
            pass
        
        def get_performance_summary(self) -> Dict[str, Any]:
            return {
                'session_id': self.session_id,
                'uptime_seconds': time.time() - self.start_time,
                'total_operations': self._operation_counter
            }
        
        def close(self):
            self.logger.info("API Usage Metrics Logger closed")
    
    # Mock other classes
    class CostRecord:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ResearchCategory(Enum):
        GENERAL_QUERY = "general_query"
    
    class CostPersistence:
        def record_cost(self, record): pass
    
    class BudgetManager:
        def check_budget_status(self): return {}
    
    class BudgetAlert: pass
    class AlertLevel: pass
    class ResearchCategorizer: pass
    class AuditTrail: pass


class QualityMetricType(Enum):
    """Extended metric types for quality validation operations."""
    
    # Inherit core types from base MetricType
    LLM_CALL = "llm_call"
    EMBEDDING_CALL = "embedding_call"
    HYBRID_OPERATION = "hybrid_operation"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    COST_TRACKING = "cost_tracking"
    BUDGET_UTILIZATION = "budget_utilization"
    RESEARCH_CATEGORY = "research_category"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    DOCUMENT_PROCESSING = "document_processing"
    MEMORY_USAGE = "memory_usage"
    CONCURRENT_OPERATIONS = "concurrent_operations"
    RETRY_PATTERNS = "retry_patterns"
    
    # Quality validation specific metrics
    QUALITY_VALIDATION = "quality_validation"
    RELEVANCE_SCORING = "relevance_scoring"
    FACTUAL_ACCURACY_VALIDATION = "factual_accuracy_validation"
    CLAIM_EXTRACTION = "claim_extraction"
    EVIDENCE_VALIDATION = "evidence_validation"
    BIOMEDICAL_TERMINOLOGY_ANALYSIS = "biomedical_terminology_analysis"
    CITATION_ANALYSIS = "citation_analysis"
    CONSISTENCY_ANALYSIS = "consistency_analysis"
    HALLUCINATION_DETECTION = "hallucination_detection"
    COMPLETENESS_ASSESSMENT = "completeness_assessment"
    CLARITY_ASSESSMENT = "clarity_assessment"
    INTEGRATED_QUALITY_WORKFLOW = "integrated_quality_workflow"
    QUALITY_BENCHMARKING = "quality_benchmarking"
    
    # Quality-specific performance metrics
    QUALITY_PROCESSING_TIME = "quality_processing_time"
    QUALITY_THROUGHPUT = "quality_throughput"
    QUALITY_ERROR_RATE = "quality_error_rate"
    QUALITY_ACCURACY_RATE = "quality_accuracy_rate"
    QUALITY_CONFIDENCE_LEVEL = "quality_confidence_level"


@dataclass
class QualityAPIMetric(APIMetric):
    """
    Extended API metric with quality validation specific context.
    
    Inherits all functionality from APIMetric and adds quality validation
    specific fields for detailed tracking and analysis.
    """
    
    # Quality validation context
    quality_operation_stage: Optional[str] = None  # e.g., "claim_extraction", "validation", "scoring"
    quality_validation_type: Optional[str] = None  # e.g., "relevance", "factual_accuracy", "integrated"
    quality_assessment_method: Optional[str] = None  # e.g., "semantic_analysis", "document_verification"
    
    # Quality metrics tracking
    quality_score: Optional[float] = None  # Overall quality score (0-100)
    relevance_score: Optional[float] = None  # Relevance score (0-100)
    factual_accuracy_score: Optional[float] = None  # Factual accuracy score (0-100)
    completeness_score: Optional[float] = None  # Completeness score (0-100)
    clarity_score: Optional[float] = None  # Clarity score (0-100)
    confidence_score: Optional[float] = None  # Confidence in assessment (0-100)
    
    # Quality validation performance
    claims_extracted: int = 0  # Number of claims extracted
    claims_validated: int = 0  # Number of claims validated
    evidence_items_processed: int = 0  # Number of evidence items processed
    biomedical_terms_identified: int = 0  # Number of biomedical terms identified
    citations_found: int = 0  # Number of citations found
    
    # Quality validation results
    validation_passed: bool = True  # Whether validation criteria were met
    validation_confidence: Optional[float] = None  # Confidence in validation result
    quality_flags: List[str] = field(default_factory=list)  # Quality warning/error flags
    quality_recommendations: List[str] = field(default_factory=list)  # Quality improvement recommendations
    
    # Quality processing stages timing
    claim_extraction_time_ms: Optional[float] = None
    validation_time_ms: Optional[float] = None  
    scoring_time_ms: Optional[float] = None
    integration_time_ms: Optional[float] = None
    
    # Quality validation costs
    quality_validation_cost_usd: float = 0.0  # Cost specifically for quality validation
    cost_per_quality_point: Optional[float] = None  # Cost per quality score point
    cost_effectiveness_ratio: Optional[float] = None  # Cost vs quality improvement ratio
    
    def __post_init__(self):
        """Extended post-initialization for quality metrics."""
        # Call parent post-init
        super().__post_init__()
        
        # Calculate quality-specific derived metrics
        if self.quality_score is not None and self.quality_validation_cost_usd > 0:
            self.cost_per_quality_point = self.quality_validation_cost_usd / max(self.quality_score, 1)
        
        # Calculate cost effectiveness (quality improvement per dollar)
        if self.quality_validation_cost_usd > 0:
            quality_improvement = self.quality_score or 0
            self.cost_effectiveness_ratio = quality_improvement / self.quality_validation_cost_usd
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert quality metric to dictionary with quality context."""
        result = super().to_dict()
        
        # Add quality-specific fields
        quality_fields = {
            'quality_operation_stage': self.quality_operation_stage,
            'quality_validation_type': self.quality_validation_type,
            'quality_assessment_method': self.quality_assessment_method,
            'quality_score': self.quality_score,
            'relevance_score': self.relevance_score,
            'factual_accuracy_score': self.factual_accuracy_score,
            'completeness_score': self.completeness_score,
            'clarity_score': self.clarity_score,
            'confidence_score': self.confidence_score,
            'claims_extracted': self.claims_extracted,
            'claims_validated': self.claims_validated,
            'evidence_items_processed': self.evidence_items_processed,
            'biomedical_terms_identified': self.biomedical_terms_identified,
            'citations_found': self.citations_found,
            'validation_passed': self.validation_passed,
            'validation_confidence': self.validation_confidence,
            'quality_flags': self.quality_flags,
            'quality_recommendations': self.quality_recommendations,
            'claim_extraction_time_ms': self.claim_extraction_time_ms,
            'validation_time_ms': self.validation_time_ms,
            'scoring_time_ms': self.scoring_time_ms,
            'integration_time_ms': self.integration_time_ms,
            'quality_validation_cost_usd': self.quality_validation_cost_usd,
            'cost_per_quality_point': self.cost_per_quality_point,
            'cost_effectiveness_ratio': self.cost_effectiveness_ratio
        }
        
        result.update(quality_fields)
        return result


class QualityMetricsAggregator(MetricsAggregator):
    """
    Enhanced metrics aggregator with quality-specific aggregation capabilities.
    
    Extends the base MetricsAggregator to provide detailed aggregation and 
    analysis of quality validation metrics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize quality metrics aggregator."""
        super().__init__(logger)
        
        # Quality-specific aggregation data
        self._quality_stats = defaultdict(lambda: defaultdict(float))
        self._quality_operation_stats = defaultdict(lambda: defaultdict(float))
        self._quality_validation_costs = defaultdict(float)
        self._quality_trends = defaultdict(lambda: deque(maxlen=100))
        self._validation_success_rates = defaultdict(list)
        
        # Quality benchmarking
        self._quality_benchmarks = {
            'excellent_threshold': 90.0,
            'good_threshold': 80.0,
            'acceptable_threshold': 70.0,
            'poor_threshold': 60.0
        }
    
    def add_metric(self, metric: Union[APIMetric, QualityAPIMetric]) -> None:
        """Add a metric with quality-aware aggregation."""
        # Call parent aggregation
        super().add_metric(metric)
        
        # Add quality-specific aggregation if it's a QualityAPIMetric
        if isinstance(metric, QualityAPIMetric):
            self._update_quality_aggregations(metric)
    
    def _update_quality_aggregations(self, metric: QualityAPIMetric) -> None:
        """Update quality-specific aggregations."""
        timestamp = datetime.fromtimestamp(metric.timestamp, tz=timezone.utc)
        hour_key = timestamp.strftime('%Y-%m-%d-%H')
        day_key = timestamp.strftime('%Y-%m-%d')
        
        # Update quality statistics
        if metric.quality_validation_type:
            quality_type_stats = self._quality_stats[metric.quality_validation_type]
            quality_type_stats['total_operations'] += 1
            quality_type_stats['total_cost'] += metric.quality_validation_cost_usd
            
            if metric.quality_score is not None:
                quality_type_stats['total_quality_score'] += metric.quality_score
                quality_type_stats['quality_score_count'] += 1
            
            if metric.validation_passed:
                quality_type_stats['successful_validations'] += 1
            else:
                quality_type_stats['failed_validations'] += 1
        
        # Update operation stage statistics
        if metric.quality_operation_stage:
            stage_stats = self._quality_operation_stats[metric.quality_operation_stage]
            stage_stats['total_operations'] += 1
            stage_stats['total_time_ms'] += metric.response_time_ms or 0
            stage_stats['total_cost'] += metric.quality_validation_cost_usd
            
            # Track stage-specific metrics
            if metric.quality_operation_stage == 'claim_extraction':
                stage_stats['total_claims_extracted'] += metric.claims_extracted
            elif metric.quality_operation_stage == 'validation':
                stage_stats['total_claims_validated'] += metric.claims_validated
                stage_stats['total_evidence_processed'] += metric.evidence_items_processed
        
        # Update cost tracking
        self._quality_validation_costs[day_key] += metric.quality_validation_cost_usd
        
        # Update trends
        if metric.quality_score is not None:
            self._quality_trends['quality_score'].append(metric.quality_score)
        if metric.confidence_score is not None:
            self._quality_trends['confidence_score'].append(metric.confidence_score)
        if metric.cost_effectiveness_ratio is not None:
            self._quality_trends['cost_effectiveness'].append(metric.cost_effectiveness_ratio)
        
        # Update success rates
        if metric.quality_validation_type:
            self._validation_success_rates[metric.quality_validation_type].append(
                metric.validation_passed
            )
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get comprehensive quality validation statistics."""
        with self._lock:
            stats = {
                'quality_validation_summary': self._get_quality_validation_summary(),
                'operation_stage_performance': self._get_operation_stage_performance(),
                'quality_trends': self._get_quality_trends(),
                'cost_analysis': self._get_quality_cost_analysis(),
                'validation_success_rates': self._get_validation_success_rates(),
                'quality_benchmarks': self._get_quality_benchmark_analysis(),
                'recommendations': self._generate_quality_recommendations()
            }
            
            return stats
    
    def _get_quality_validation_summary(self) -> Dict[str, Any]:
        """Get summary of quality validation operations."""
        summary = {}
        
        for validation_type, stats in self._quality_stats.items():
            total_ops = stats.get('total_operations', 0)
            if total_ops > 0:
                avg_quality = (
                    stats.get('total_quality_score', 0) / 
                    max(stats.get('quality_score_count', 1), 1)
                )
                success_rate = (
                    stats.get('successful_validations', 0) / total_ops * 100
                )
                avg_cost = stats.get('total_cost', 0) / total_ops
                
                summary[validation_type] = {
                    'total_operations': total_ops,
                    'average_quality_score': round(avg_quality, 2),
                    'success_rate_percent': round(success_rate, 2),
                    'average_cost_usd': round(avg_cost, 6),
                    'total_cost_usd': round(stats.get('total_cost', 0), 6)
                }
        
        return summary
    
    def _get_operation_stage_performance(self) -> Dict[str, Any]:
        """Get performance analysis by operation stage."""
        stage_performance = {}
        
        for stage, stats in self._quality_operation_stats.items():
            total_ops = stats.get('total_operations', 0)
            if total_ops > 0:
                avg_time = stats.get('total_time_ms', 0) / total_ops
                avg_cost = stats.get('total_cost', 0) / total_ops
                
                stage_performance[stage] = {
                    'total_operations': total_ops,
                    'average_time_ms': round(avg_time, 2),
                    'average_cost_usd': round(avg_cost, 6),
                    'total_cost_usd': round(stats.get('total_cost', 0), 6)
                }
                
                # Add stage-specific metrics
                if stage == 'claim_extraction':
                    avg_claims = stats.get('total_claims_extracted', 0) / total_ops
                    stage_performance[stage]['average_claims_extracted'] = round(avg_claims, 2)
                elif stage == 'validation':
                    avg_validated = stats.get('total_claims_validated', 0) / total_ops
                    avg_evidence = stats.get('total_evidence_processed', 0) / total_ops
                    stage_performance[stage]['average_claims_validated'] = round(avg_validated, 2)
                    stage_performance[stage]['average_evidence_processed'] = round(avg_evidence, 2)
        
        return stage_performance
    
    def _get_quality_trends(self) -> Dict[str, Any]:
        """Get quality trend analysis."""
        trends = {}
        
        for metric_name, values in self._quality_trends.items():
            if values:
                values_list = list(values)
                trends[metric_name] = {
                    'current_value': values_list[-1],
                    'average': round(statistics.mean(values_list), 2),
                    'median': round(statistics.median(values_list), 2),
                    'std_deviation': round(statistics.stdev(values_list) if len(values_list) > 1 else 0, 2),
                    'min_value': min(values_list),
                    'max_value': max(values_list),
                    'sample_size': len(values_list)
                }
                
                # Calculate trend direction
                if len(values_list) >= 10:
                    recent_avg = statistics.mean(values_list[-10:])
                    historical_avg = statistics.mean(values_list[:-10])
                    trend_direction = 'improving' if recent_avg > historical_avg else 'declining'
                    trends[metric_name]['trend_direction'] = trend_direction
        
        return trends
    
    def _get_quality_cost_analysis(self) -> Dict[str, Any]:
        """Get quality validation cost analysis."""
        now = datetime.now(tz=timezone.utc)
        today_key = now.strftime('%Y-%m-%d')
        
        # Calculate daily costs
        daily_costs = dict(self._quality_validation_costs)
        total_cost = sum(daily_costs.values())
        
        # Calculate cost trends
        sorted_days = sorted(daily_costs.keys())
        recent_cost_trend = 'stable'
        
        if len(sorted_days) >= 7:
            recent_week = sum(daily_costs.get(day, 0) for day in sorted_days[-7:])
            previous_week = sum(daily_costs.get(day, 0) for day in sorted_days[-14:-7])
            if recent_week > previous_week * 1.1:
                recent_cost_trend = 'increasing'
            elif recent_week < previous_week * 0.9:
                recent_cost_trend = 'decreasing'
        
        return {
            'total_quality_validation_cost_usd': round(total_cost, 6),
            'daily_costs': {day: round(cost, 6) for day, cost in daily_costs.items()},
            'today_cost_usd': round(daily_costs.get(today_key, 0), 6),
            'average_daily_cost_usd': round(
                total_cost / max(len(daily_costs), 1), 6
            ),
            'cost_trend': recent_cost_trend,
            'cost_tracking_days': len(daily_costs)
        }
    
    def _get_validation_success_rates(self) -> Dict[str, Any]:
        """Get validation success rate analysis."""
        success_rates = {}
        
        for validation_type, results in self._validation_success_rates.items():
            if results:
                success_count = sum(results)
                total_count = len(results)
                success_rate = success_count / total_count * 100
                
                success_rates[validation_type] = {
                    'success_rate_percent': round(success_rate, 2),
                    'successful_validations': success_count,
                    'total_validations': total_count,
                    'recent_success_rate': round(
                        sum(results[-10:]) / min(len(results[-10:]), 10) * 100, 2
                    ) if results else 0
                }
        
        return success_rates
    
    def _get_quality_benchmark_analysis(self) -> Dict[str, Any]:
        """Get quality benchmark analysis."""
        if not self._quality_trends.get('quality_score'):
            return {'status': 'insufficient_data'}
        
        quality_scores = list(self._quality_trends['quality_score'])
        avg_quality = statistics.mean(quality_scores)
        
        # Determine quality grade
        if avg_quality >= self._quality_benchmarks['excellent_threshold']:
            grade = 'excellent'
        elif avg_quality >= self._quality_benchmarks['good_threshold']:
            grade = 'good'
        elif avg_quality >= self._quality_benchmarks['acceptable_threshold']:
            grade = 'acceptable'
        elif avg_quality >= self._quality_benchmarks['poor_threshold']:
            grade = 'needs_improvement'
        else:
            grade = 'poor'
        
        # Calculate distribution
        excellent_count = sum(1 for score in quality_scores 
                            if score >= self._quality_benchmarks['excellent_threshold'])
        good_count = sum(1 for score in quality_scores 
                        if self._quality_benchmarks['good_threshold'] <= score < self._quality_benchmarks['excellent_threshold'])
        acceptable_count = sum(1 for score in quality_scores 
                             if self._quality_benchmarks['acceptable_threshold'] <= score < self._quality_benchmarks['good_threshold'])
        poor_count = len(quality_scores) - excellent_count - good_count - acceptable_count
        
        return {
            'overall_grade': grade,
            'average_quality_score': round(avg_quality, 2),
            'quality_distribution': {
                'excellent': excellent_count,
                'good': good_count, 
                'acceptable': acceptable_count,
                'poor': poor_count
            },
            'quality_distribution_percent': {
                'excellent': round(excellent_count / len(quality_scores) * 100, 2),
                'good': round(good_count / len(quality_scores) * 100, 2),
                'acceptable': round(acceptable_count / len(quality_scores) * 100, 2),
                'poor': round(poor_count / len(quality_scores) * 100, 2)
            },
            'benchmarks': self._quality_benchmarks
        }
    
    def _generate_quality_recommendations(self) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Analyze quality trends
        if self._quality_trends.get('quality_score'):
            quality_scores = list(self._quality_trends['quality_score'])
            avg_quality = statistics.mean(quality_scores)
            
            if avg_quality < self._quality_benchmarks['acceptable_threshold']:
                recommendations.append(
                    f"Quality scores are below acceptable threshold ({self._quality_benchmarks['acceptable_threshold']}). "
                    "Consider reviewing validation criteria and improving assessment methods."
                )
            
            # Check for declining trends
            if len(quality_scores) >= 10:
                recent_avg = statistics.mean(quality_scores[-5:])
                historical_avg = statistics.mean(quality_scores[-10:-5])
                if recent_avg < historical_avg * 0.95:
                    recommendations.append(
                        "Quality scores show declining trend. Review recent changes and consider "
                        "adjusting validation parameters or improving training data."
                    )
        
        # Analyze cost effectiveness
        if self._quality_trends.get('cost_effectiveness'):
            cost_effectiveness = list(self._quality_trends['cost_effectiveness'])
            if cost_effectiveness:
                avg_effectiveness = statistics.mean(cost_effectiveness)
                if avg_effectiveness < 1.0:
                    recommendations.append(
                        "Cost effectiveness is low. Consider optimizing validation processes "
                        "or adjusting quality thresholds to improve cost efficiency."
                    )
        
        # Analyze success rates
        for validation_type, results in self._validation_success_rates.items():
            if results:
                success_rate = sum(results) / len(results) * 100
                if success_rate < 80:
                    recommendations.append(
                        f"{validation_type} validation has low success rate ({success_rate:.1f}%). "
                        "Consider reviewing validation criteria or improving input quality."
                    )
        
        # Default recommendation if no issues found
        if not recommendations:
            recommendations.append(
                "Quality validation performance is within acceptable parameters. "
                "Continue monitoring for consistency and consider gradual improvements."
            )
        
        return recommendations


class QualityAwareAPIMetricsLogger(APIUsageMetricsLogger):
    """
    Enhanced API metrics logger with quality validation specific tracking.
    
    Extends APIUsageMetricsLogger to provide comprehensive tracking of quality
    validation operations with detailed context and specialized analysis.
    """
    
    def __init__(self, 
                 config: Any = None,
                 cost_persistence: Optional[CostPersistence] = None,
                 budget_manager: Optional[BudgetManager] = None,
                 research_categorizer: Optional[ResearchCategorizer] = None,
                 audit_trail: Optional[AuditTrail] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize quality-aware API metrics logger.
        
        Args:
            config: Configuration object with logging settings
            cost_persistence: Cost persistence layer for integration
            budget_manager: Budget manager for cost tracking
            research_categorizer: Research categorizer for metrics
            audit_trail: Audit trail for compliance logging
            logger: Logger instance for metrics logging
        """
        # Initialize parent logger
        super().__init__(config, cost_persistence, budget_manager, 
                        research_categorizer, audit_trail, logger)
        
        # Replace aggregator with quality-aware version
        self.metrics_aggregator = QualityMetricsAggregator(self.logger)
        
        # Quality-specific tracking
        self._quality_session_stats = {
            'total_quality_operations': 0,
            'quality_validation_cost': 0.0,
            'average_quality_score': 0.0,
            'validation_success_rate': 0.0
        }
        
        # Quality validation context tracking
        self._active_quality_operations = {}
        
        self.logger.info(f"Quality-Aware API Usage Metrics Logger initialized with session ID: {self.session_id}")
    
    @contextmanager
    def track_quality_validation(self, 
                               operation_name: str,
                               validation_type: str,  # e.g., 'relevance', 'factual_accuracy', 'integrated'
                               quality_stage: str = 'validation',  # e.g., 'claim_extraction', 'validation', 'scoring'
                               assessment_method: Optional[str] = None,
                               model_name: Optional[str] = None,
                               research_category: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking quality validation operations.
        
        Usage:
            with logger.track_quality_validation("factual_accuracy", "claim_validation") as tracker:
                # Perform quality validation
                result = validate_claims(response)
                tracker.set_quality_results(quality_score=85, confidence=90)
                tracker.set_validation_details(claims_extracted=5, claims_validated=4)
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Create quality metric template
        metric = QualityAPIMetric(
            session_id=self.session_id,
            metric_type=QualityMetricType.QUALITY_VALIDATION,
            operation_name=operation_name,
            model_name=model_name,
            research_category=research_category or ResearchCategory.GENERAL_QUERY.value,
            quality_validation_type=validation_type,
            quality_operation_stage=quality_stage,
            quality_assessment_method=assessment_method,
            metadata=metadata or {},
            memory_usage_mb=start_memory
        )
        
        # Track active quality operation
        with self._lock:
            self._operation_counter += 1
            self._active_quality_operations[operation_id] = {
                'metric': metric,
                'start_time': start_time,
                'start_memory': start_memory,
                'stage_timings': {}
            }
            metric.concurrent_operations = len(self._active_operations) + len(self._active_quality_operations)
        
        class QualityValidationTracker:
            def __init__(self, metric: QualityAPIMetric, logger: 'QualityAwareAPIMetricsLogger'):
                self.metric = metric
                self.logger = logger
                self.completed = False
                self.stage_start_time = time.time()
            
            def set_quality_results(self, 
                                  quality_score: Optional[float] = None,
                                  relevance_score: Optional[float] = None,
                                  factual_accuracy_score: Optional[float] = None,
                                  completeness_score: Optional[float] = None,
                                  clarity_score: Optional[float] = None,
                                  confidence_score: Optional[float] = None):
                """Set quality assessment results."""
                if quality_score is not None:
                    self.metric.quality_score = quality_score
                if relevance_score is not None:
                    self.metric.relevance_score = relevance_score
                if factual_accuracy_score is not None:
                    self.metric.factual_accuracy_score = factual_accuracy_score
                if completeness_score is not None:
                    self.metric.completeness_score = completeness_score
                if clarity_score is not None:
                    self.metric.clarity_score = clarity_score
                if confidence_score is not None:
                    self.metric.confidence_score = confidence_score
            
            def set_validation_details(self,
                                     claims_extracted: int = 0,
                                     claims_validated: int = 0,
                                     evidence_items_processed: int = 0,
                                     biomedical_terms_identified: int = 0,
                                     citations_found: int = 0,
                                     validation_passed: bool = True,
                                     validation_confidence: Optional[float] = None):
                """Set validation process details."""
                self.metric.claims_extracted = claims_extracted
                self.metric.claims_validated = claims_validated
                self.metric.evidence_items_processed = evidence_items_processed
                self.metric.biomedical_terms_identified = biomedical_terms_identified
                self.metric.citations_found = citations_found
                self.metric.validation_passed = validation_passed
                self.metric.validation_confidence = validation_confidence
            
            def add_quality_flag(self, flag: str):
                """Add quality warning/error flag."""
                self.metric.quality_flags.append(flag)
            
            def add_quality_recommendation(self, recommendation: str):
                """Add quality improvement recommendation."""
                self.metric.quality_recommendations.append(recommendation)
            
            def set_stage_timing(self, stage: str, time_ms: float):
                """Set timing for specific quality validation stage."""
                if stage == 'claim_extraction':
                    self.metric.claim_extraction_time_ms = time_ms
                elif stage == 'validation':
                    self.metric.validation_time_ms = time_ms
                elif stage == 'scoring':
                    self.metric.scoring_time_ms = time_ms
                elif stage == 'integration':
                    self.metric.integration_time_ms = time_ms
            
            def record_stage_completion(self, stage: str):
                """Record completion of a quality validation stage."""
                stage_time = (time.time() - self.stage_start_time) * 1000
                self.set_stage_timing(stage, stage_time)
                self.stage_start_time = time.time()  # Reset for next stage
            
            def set_tokens(self, prompt: int = 0, completion: int = 0, embedding: int = 0):
                """Set token usage for quality validation."""
                self.metric.prompt_tokens = prompt
                self.metric.completion_tokens = completion
                self.metric.embedding_tokens = embedding
                self.metric.total_tokens = prompt + completion + embedding
            
            def set_cost(self, total_cost_usd: float, quality_validation_cost_usd: Optional[float] = None):
                """Set cost information for quality validation."""
                self.metric.cost_usd = total_cost_usd
                self.metric.quality_validation_cost_usd = quality_validation_cost_usd or total_cost_usd
            
            def set_error(self, error_type: str, error_message: str):
                """Set error information."""
                self.metric.success = False
                self.metric.error_type = error_type
                self.metric.error_message = error_message
                self.metric.validation_passed = False
                self.add_quality_flag(f"Error: {error_type}")
            
            def add_metadata(self, key: str, value: Any):
                """Add metadata to the quality metric."""
                self.metric.metadata[key] = value
            
            def complete(self):
                """Complete the quality validation tracking."""
                if not self.completed:
                    self.logger._complete_quality_operation(operation_id, self.metric)
                    self.completed = True
        
        tracker = QualityValidationTracker(metric, self)
        
        try:
            yield tracker
        except Exception as e:
            tracker.set_error(type(e).__name__, str(e))
            raise
        finally:
            if not tracker.completed:
                tracker.complete()
    
    def _complete_quality_operation(self, operation_id: str, metric: QualityAPIMetric) -> None:
        """Complete a quality validation operation and log metrics."""
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Update metric with completion details
        if not metric.response_time_ms:
            metric.response_time_ms = (end_time - metric.timestamp) * 1000
        
        # Memory usage delta
        with self._lock:
            if operation_id in self._active_quality_operations:
                start_memory = self._active_quality_operations[operation_id]['start_memory']
                del self._active_quality_operations[operation_id]
                metric.memory_usage_mb = max(end_memory - start_memory, 0)
        
        # Update session statistics
        self._update_quality_session_stats(metric)
        
        # Add to aggregator (will handle both base and quality aggregation)
        self.metrics_aggregator.add_metric(metric)
        
        # Log structured metrics
        self._log_quality_metric(metric)
        
        # Integrate with cost tracking systems
        self._integrate_with_cost_systems(metric)
        
        # Log quality-specific audit trail
        self._log_quality_audit_trail(metric)
    
    def _update_quality_session_stats(self, metric: QualityAPIMetric) -> None:
        """Update session-level quality statistics."""
        with self._lock:
            self._quality_session_stats['total_quality_operations'] += 1
            self._quality_session_stats['quality_validation_cost'] += metric.quality_validation_cost_usd
            
            # Update rolling averages
            total_ops = self._quality_session_stats['total_quality_operations']
            if metric.quality_score is not None:
                current_avg = self._quality_session_stats['average_quality_score']
                new_avg = ((current_avg * (total_ops - 1)) + metric.quality_score) / total_ops
                self._quality_session_stats['average_quality_score'] = new_avg
            
            if metric.validation_passed is not None:
                success_count = sum(1 for op_data in self._active_quality_operations.values() 
                                  if op_data['metric'].validation_passed)
                self._quality_session_stats['validation_success_rate'] = (success_count / total_ops) * 100
    
    def _log_quality_metric(self, metric: QualityAPIMetric) -> None:
        """Log quality-specific structured metric data."""
        try:
            # Call parent logging first
            self._log_metric(metric)
            
            # Create quality-specific log entry
            quality_summary = (
                f"Quality Validation: {metric.operation_name} | "
                f"Type: {metric.quality_validation_type} | "
                f"Stage: {metric.quality_operation_stage} | "
                f"Quality Score: {metric.quality_score or 'N/A'} | "
                f"Validation: {'Passed' if metric.validation_passed else 'Failed'} | "
                f"Cost: ${metric.quality_validation_cost_usd:.6f} | "
                f"Time: {metric.response_time_ms:.1f}ms"
            )
            
            if metric.claims_extracted > 0:
                quality_summary += f" | Claims: {metric.claims_extracted}"
            if metric.confidence_score is not None:
                quality_summary += f" | Confidence: {metric.confidence_score:.1f}"
            
            log_level = logging.INFO if metric.success and metric.validation_passed else logging.WARNING
            self.logger.log(log_level, quality_summary)
            
            # Log quality flags and recommendations if present
            if metric.quality_flags:
                self.logger.warning(f"Quality Flags: {', '.join(metric.quality_flags)}")
            if metric.quality_recommendations:
                self.logger.info(f"Quality Recommendations: {', '.join(metric.quality_recommendations)}")
            
        except Exception as e:
            self.logger.error(f"Error logging quality metric: {e}")
    
    def _log_quality_audit_trail(self, metric: QualityAPIMetric) -> None:
        """Log quality-specific audit trail."""
        try:
            # Call parent audit logging
            self._log_audit_trail(metric)
            
            # Create quality-specific audit data
            quality_audit_data = {
                'event_type': 'quality_validation',
                'timestamp': metric.timestamp,
                'session_id': metric.session_id,
                'operation': metric.operation_name,
                'validation_type': metric.quality_validation_type,
                'operation_stage': metric.quality_operation_stage,
                'quality_score': metric.quality_score,
                'validation_passed': metric.validation_passed,
                'validation_confidence': metric.validation_confidence,
                'claims_extracted': metric.claims_extracted,
                'claims_validated': metric.claims_validated,
                'quality_validation_cost_usd': metric.quality_validation_cost_usd,
                'cost_effectiveness_ratio': metric.cost_effectiveness_ratio,
                'quality_flags': metric.quality_flags,
                'user_id': metric.user_id,
                'project_id': metric.project_id
            }
            
            if not metric.success:
                quality_audit_data['error_type'] = metric.error_type
                quality_audit_data['error_message'] = metric.error_message
            
            self.audit_logger.info(f"QUALITY_VALIDATION: {json.dumps(quality_audit_data)}")
            
            # Record in audit trail system if available
            if self.audit_trail:
                self.audit_trail.record_event(
                    event_type='quality_validation',
                    event_data=quality_audit_data,
                    user_id=metric.user_id,
                    session_id=metric.session_id
                )
                
        except Exception as e:
            self.logger.error(f"Error logging quality audit trail: {e}")
    
    def get_quality_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality validation performance summary."""
        try:
            # Get base performance summary
            base_summary = super().get_performance_summary()
            
            # Add quality-specific statistics
            quality_stats = self.metrics_aggregator.get_quality_stats()
            
            # Combine summaries
            quality_summary = {
                **base_summary,
                'quality_validation': {
                    'session_stats': self._quality_session_stats.copy(),
                    'active_quality_operations': len(self._active_quality_operations),
                    **quality_stats
                }
            }
            
            return quality_summary
            
        except Exception as e:
            self.logger.error(f"Error getting quality performance summary: {e}")
            return {'error': str(e)}
    
    def log_quality_batch_operation(self, 
                                  operation_name: str,
                                  validation_type: str,
                                  batch_size: int,
                                  total_tokens: int,
                                  total_cost: float,
                                  quality_validation_cost: float,
                                  processing_time_ms: float,
                                  average_quality_score: float,
                                  success_count: int,
                                  validation_passed_count: int,
                                  error_count: int,
                                  research_category: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log metrics for quality validation batch operations.
        
        Args:
            operation_name: Name of the batch operation
            validation_type: Type of quality validation performed
            batch_size: Number of items in the batch
            total_tokens: Total tokens consumed
            total_cost: Total cost in USD
            quality_validation_cost: Cost specifically for quality validation
            processing_time_ms: Total processing time in milliseconds
            average_quality_score: Average quality score across batch
            success_count: Number of successful operations
            validation_passed_count: Number of validations that passed
            error_count: Number of failed operations
            research_category: Research category for the batch
            metadata: Additional metadata
        """
        metric = QualityAPIMetric(
            session_id=self.session_id,
            metric_type=QualityMetricType.QUALITY_VALIDATION,
            operation_name=f"batch_{operation_name}",
            total_tokens=total_tokens,
            cost_usd=total_cost,
            response_time_ms=processing_time_ms,
            success=error_count == 0,
            quality_validation_type=validation_type,
            quality_operation_stage='batch_processing',
            quality_score=average_quality_score,
            validation_passed=validation_passed_count == batch_size,
            quality_validation_cost_usd=quality_validation_cost,
            research_category=research_category or ResearchCategory.GENERAL_QUERY.value,
            metadata={
                **(metadata or {}),
                'batch_size': batch_size,
                'success_count': success_count,
                'validation_passed_count': validation_passed_count,
                'error_count': error_count,
                'success_rate': success_count / batch_size if batch_size > 0 else 0,
                'validation_success_rate': validation_passed_count / batch_size if batch_size > 0 else 0
            }
        )
        
        # Add to aggregator and log
        self.metrics_aggregator.add_metric(metric)
        self._log_quality_metric(metric)
        self._integrate_with_cost_systems(metric)
        self._log_quality_audit_trail(metric)
    
    @contextmanager
    def track_integrated_quality_workflow(self,
                                        workflow_name: str,
                                        components: List[str],  # e.g., ['relevance', 'factual_accuracy', 'quality_assessment']
                                        model_name: Optional[str] = None,
                                        research_category: Optional[str] = None,
                                        metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking integrated quality validation workflows.
        
        Usage:
            with logger.track_integrated_quality_workflow(
                "comprehensive_validation", 
                ["relevance", "factual_accuracy", "quality_assessment"]
            ) as tracker:
                # Perform integrated quality workflow
                results = integrated_workflow.run()
                tracker.set_component_results(results)
                tracker.set_workflow_outcome(overall_score=88, passed=True)
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create integrated workflow metric
        metric = QualityAPIMetric(
            session_id=self.session_id,
            metric_type=QualityMetricType.INTEGRATED_QUALITY_WORKFLOW,
            operation_name=workflow_name,
            model_name=model_name,
            research_category=research_category or ResearchCategory.GENERAL_QUERY.value,
            quality_validation_type='integrated',
            quality_operation_stage='workflow',
            quality_assessment_method='multi_component',
            metadata={
                **(metadata or {}),
                'components': components,
                'component_count': len(components)
            }
        )
        
        class IntegratedQualityTracker:
            def __init__(self, metric: QualityAPIMetric, logger: 'QualityAwareAPIMetricsLogger'):
                self.metric = metric
                self.logger = logger
                self.completed = False
                self.component_results = {}
                self.component_costs = {}
            
            def set_component_result(self, component: str, score: float, cost: float = 0.0):
                """Set result for individual component."""
                self.component_results[component] = score
                self.component_costs[component] = cost
                
                # Update specific score fields
                if component == 'relevance':
                    self.metric.relevance_score = score
                elif component == 'factual_accuracy':
                    self.metric.factual_accuracy_score = score
                elif component in ['completeness', 'quality_assessment']:
                    self.metric.completeness_score = score
                elif component == 'clarity':
                    self.metric.clarity_score = score
            
            def set_component_results(self, results: Dict[str, Dict[str, Any]]):
                """Set results for multiple components."""
                for component, result in results.items():
                    score = result.get('score', 0.0)
                    cost = result.get('cost', 0.0)
                    self.set_component_result(component, score, cost)
            
            def set_workflow_outcome(self, 
                                   overall_score: float,
                                   passed: bool,
                                   confidence: Optional[float] = None,
                                   integration_method: str = 'weighted_average'):
                """Set overall workflow outcome."""
                self.metric.quality_score = overall_score
                self.metric.validation_passed = passed
                self.metric.confidence_score = confidence
                self.metric.quality_assessment_method = integration_method
                
                # Calculate total quality validation cost
                self.metric.quality_validation_cost_usd = sum(self.component_costs.values())
            
            def add_workflow_timing(self, component: str, time_ms: float):
                """Add timing for workflow component."""
                timing_key = f"{component}_time_ms"
                self.metric.metadata[timing_key] = time_ms
            
            def set_tokens(self, prompt: int = 0, completion: int = 0, embedding: int = 0):
                self.metric.prompt_tokens = prompt
                self.metric.completion_tokens = completion
                self.metric.embedding_tokens = embedding
                self.metric.total_tokens = prompt + completion + embedding
            
            def set_cost(self, total_cost_usd: float):
                self.metric.cost_usd = total_cost_usd
                if not self.metric.quality_validation_cost_usd:
                    self.metric.quality_validation_cost_usd = total_cost_usd
            
            def add_metadata(self, key: str, value: Any):
                self.metric.metadata[key] = value
            
            def complete(self):
                if not self.completed:
                    # Store component results in metadata
                    self.metric.metadata['component_results'] = self.component_results
                    self.metric.metadata['component_costs'] = self.component_costs
                    
                    self.logger._complete_quality_operation(operation_id, self.metric)
                    self.completed = True
        
        tracker = IntegratedQualityTracker(metric, self)
        
        try:
            yield tracker
        except Exception as e:
            tracker.metric.success = False
            tracker.metric.error_type = type(e).__name__
            tracker.metric.error_message = str(e)
            tracker.metric.validation_passed = False
            raise
        finally:
            if not tracker.completed:
                tracker.complete()
    
    def export_quality_metrics_report(self, 
                                    output_path: Union[str, Path],
                                    format: str = 'json',
                                    include_raw_data: bool = False) -> str:
        """
        Export comprehensive quality validation metrics report.
        
        Args:
            output_path: Path to export the report
            format: Export format ('json' or 'html')
            include_raw_data: Whether to include raw metric data
            
        Returns:
            Path to the exported report file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate comprehensive report
            report_data = {
                'report_metadata': {
                    'generated_timestamp': datetime.now(timezone.utc).isoformat(),
                    'session_id': self.session_id,
                    'report_version': '1.0.0',
                    'format': format,
                    'include_raw_data': include_raw_data
                },
                'session_summary': self._quality_session_stats.copy(),
                'quality_performance': self.get_quality_performance_summary(),
                'detailed_analysis': self.metrics_aggregator.get_quality_stats()
            }
            
            if include_raw_data:
                # Add raw metrics data (limited to recent entries)
                report_data['raw_metrics'] = []
                for metric in list(self.metrics_aggregator._metrics_buffer)[-100:]:
                    if isinstance(metric, QualityAPIMetric):
                        report_data['raw_metrics'].append(metric.to_dict())
            
            # Export based on format
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            elif format.lower() == 'html':
                html_content = self._generate_html_report(report_data)
                with open(output_path, 'w') as f:
                    f.write(html_content)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Quality metrics report exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting quality metrics report: {e}")
            raise
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report from report data."""
        # Basic HTML template for quality metrics report
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Quality Validation Metrics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; padding: 8px; background-color: #f9f9f9; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .json-data {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; font-family: monospace; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Validation Metrics Report</h1>
        <p>Generated: {generated_time}</p>
        <p>Session ID: {session_id}</p>
    </div>
    
    <div class="section">
        <h2>Session Summary</h2>
        <div class="json-data">{session_summary}</div>
    </div>
    
    <div class="section">
        <h2>Quality Performance Analysis</h2>
        <div class="json-data">{quality_performance}</div>
    </div>
    
    <div class="section">
        <h2>Detailed Analysis</h2>
        <div class="json-data">{detailed_analysis}</div>
    </div>
</body>
</html>""".format(
            generated_time=report_data['report_metadata']['generated_timestamp'],
            session_id=report_data['report_metadata']['session_id'],
            session_summary=json.dumps(report_data['session_summary'], indent=2),
            quality_performance=json.dumps(report_data['quality_performance'], indent=2, default=str),
            detailed_analysis=json.dumps(report_data['detailed_analysis'], indent=2, default=str)
        )
        
        return html_template
    
    def close(self) -> None:
        """Clean shutdown of quality-aware metrics logging."""
        try:
            # Log final quality performance summary
            final_quality_summary = self.get_quality_performance_summary()
            self.logger.info(f"Quality-Aware API Metrics Logger shutdown - Final Summary: {json.dumps(final_quality_summary, default=str)}")
            
            # Call parent close method
            super().close()
            
        except Exception as e:
            self.logger.error(f"Error during quality-aware metrics logger shutdown: {e}")


# Convenience functions for quality-aware metrics logging
def create_quality_aware_logger(config: Any = None,
                              cost_persistence: Optional[CostPersistence] = None,
                              budget_manager: Optional[BudgetManager] = None,
                              research_categorizer: Optional[ResearchCategorizer] = None,
                              audit_trail: Optional[AuditTrail] = None,
                              logger: Optional[logging.Logger] = None) -> QualityAwareAPIMetricsLogger:
    """Create a quality-aware API metrics logger with all integrations."""
    return QualityAwareAPIMetricsLogger(
        config=config,
        cost_persistence=cost_persistence,
        budget_manager=budget_manager,
        research_categorizer=research_categorizer,
        audit_trail=audit_trail,
        logger=logger
    )


if __name__ == "__main__":
    # Quality-aware metrics logging demonstration
    def test_quality_aware_metrics_logging():
        """Test quality-aware metrics logging functionality."""
        
        print("Quality-Aware API Metrics Logging Test")
        print("=" * 50)
        
        # Create logger
        logger = QualityAwareAPIMetricsLogger()
        
        # Simulate quality validation operations
        print("Simulating quality validation operations...")
        
        # Test relevance scoring
        with logger.track_quality_validation("relevance_assessment", "relevance", "scoring") as tracker:
            time.sleep(0.1)  # Simulate processing
            tracker.set_tokens(prompt=150, completion=50)
            tracker.set_cost(0.005)
            tracker.set_quality_results(
                quality_score=85.0,
                relevance_score=85.0,
                confidence_score=90.0
            )
            tracker.set_validation_details(validation_passed=True, validation_confidence=90.0)
        
        # Test factual accuracy validation
        with logger.track_quality_validation("factual_accuracy_check", "factual_accuracy", "validation") as tracker:
            time.sleep(0.15)  # Simulate processing
            tracker.set_tokens(prompt=200, completion=30)
            tracker.set_cost(0.008, quality_validation_cost_usd=0.006)
            tracker.set_quality_results(
                quality_score=78.0,
                factual_accuracy_score=78.0,
                confidence_score=85.0
            )
            tracker.set_validation_details(
                claims_extracted=5,
                claims_validated=4,
                evidence_items_processed=12,
                validation_passed=True,
                validation_confidence=85.0
            )
            tracker.record_stage_completion("claim_extraction")
            tracker.record_stage_completion("validation")
        
        # Test integrated workflow
        with logger.track_integrated_quality_workflow(
            "comprehensive_quality_check", 
            ["relevance", "factual_accuracy", "quality_assessment"]
        ) as tracker:
            time.sleep(0.2)  # Simulate processing
            tracker.set_tokens(prompt=400, completion=100)
            tracker.set_cost(0.015)
            tracker.set_component_results({
                'relevance': {'score': 85.0, 'cost': 0.005},
                'factual_accuracy': {'score': 78.0, 'cost': 0.006},
                'quality_assessment': {'score': 82.0, 'cost': 0.004}
            })
            tracker.set_workflow_outcome(
                overall_score=81.7,
                passed=True,
                confidence=87.0,
                integration_method='weighted_average'
            )
        
        # Test batch operation
        logger.log_quality_batch_operation(
            operation_name="batch_quality_validation",
            validation_type="integrated",
            batch_size=10,
            total_tokens=2000,
            total_cost=0.08,
            quality_validation_cost=0.06,
            processing_time_ms=1500,
            average_quality_score=83.5,
            success_count=10,
            validation_passed_count=9,
            error_count=0
        )
        
        print("Quality validation operations completed.")
        
        # Get performance summary
        print("\nQuality Performance Summary:")
        summary = logger.get_quality_performance_summary()
        
        print(f"Total Quality Operations: {summary['quality_validation']['session_stats']['total_quality_operations']}")
        print(f"Quality Validation Cost: ${summary['quality_validation']['session_stats']['quality_validation_cost']:.6f}")
        print(f"Average Quality Score: {summary['quality_validation']['session_stats']['average_quality_score']:.2f}")
        print(f"Validation Success Rate: {summary['quality_validation']['session_stats']['validation_success_rate']:.1f}%")
        
        # Display quality statistics
        quality_stats = summary['quality_validation'].get('quality_validation_summary', {})
        if quality_stats:
            print(f"\nQuality Validation Summary:")
            for validation_type, stats in quality_stats.items():
                print(f"  {validation_type}:")
                print(f"    Operations: {stats['total_operations']}")
                print(f"    Avg Quality: {stats['average_quality_score']}")
                print(f"    Success Rate: {stats['success_rate_percent']}%")
                print(f"    Total Cost: ${stats['total_cost_usd']:.6f}")
        
        # Test report export
        print(f"\nExporting quality metrics report...")
        report_path = logger.export_quality_metrics_report(
            output_path=Path("quality_metrics_report.json"),
            format="json",
            include_raw_data=True
        )
        print(f"Report exported to: {report_path}")
        
        logger.close()
        print(f"\nQuality-aware metrics logging test completed successfully")
    
    # Run test
    test_quality_aware_metrics_logging()