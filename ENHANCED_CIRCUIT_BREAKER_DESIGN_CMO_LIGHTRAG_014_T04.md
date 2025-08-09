# Enhanced Circuit Breaker Design - CMO-LIGHTRAG-014-T04

**Task**: Design enhanced circuit breaker patterns for Clinical Metabolomics Oracle  
**Author**: Claude Code (Anthropic)  
**Created**: August 9, 2025  
**Version**: 1.0  

## Executive Summary

This document presents an enhanced circuit breaker architecture that builds upon the existing excellent cost-based circuit breaker foundation in the Clinical Metabolomics Oracle system. The design introduces service-specific circuit breakers, advanced coordination patterns, adaptive threshold management, and enhanced observability while maintaining full backward compatibility.

## Current Architecture Analysis

### Existing Strengths

The current system demonstrates strong foundations:

1. **Cost-Based Circuit Breakers** (`cost_based_circuit_breaker.py`)
   - Budget-aware operation limiting
   - Multiple threshold rule types (daily, monthly, rate-based)
   - Integration with BudgetManager and cost tracking
   - Comprehensive statistics and monitoring

2. **Production Circuit Breakers** (`production_load_balancer.py`)
   - Adaptive failure detection with proactive opening
   - Enhanced recovery time calculation
   - Performance-based degradation detection
   - Integration with load balancing decisions

3. **Fallback Systems** (`fallback_decision_logging_metrics.py`)
   - Uncertainty-aware cascade patterns
   - Comprehensive decision logging
   - Performance metrics collection
   - Real-time dashboard capabilities

### Identified Enhancement Opportunities

1. **Service-Specific Isolation**: Need API-specific circuit breakers for OpenAI, Perplexity, etc.
2. **Cross-Service Coordination**: Better coordination between different circuit breaker types
3. **Adaptive Thresholds**: Real-time threshold adjustment based on system conditions
4. **Enhanced Observability**: Better debugging and monitoring capabilities

## Enhanced Circuit Breaker Architecture

### Core Design Principles

1. **Build Upon Existing Foundation**: Extend rather than replace current implementations
2. **Service-Specific Protection**: Isolate failures per API service
3. **Hierarchical Coordination**: Coordinate between service, cost, and system-level breakers
4. **Adaptive Intelligence**: Learn and adjust thresholds dynamically
5. **Enhanced Observability**: Provide comprehensive monitoring and debugging

## Class Architecture Overview

```
EnhancedCircuitBreakerSystem
├── ServiceSpecificCircuitBreakerManager
│   ├── OpenAICircuitBreaker
│   ├── PerplexityCircuitBreaker
│   ├── LightRAGCircuitBreaker
│   └── CacheCircuitBreaker
├── CrossServiceCoordinator
│   ├── CircuitBreakerOrchestrator
│   ├── FailureCorrelationAnalyzer
│   └── ProgressiveDegradationManager
├── AdaptiveThresholdManager
│   ├── ThresholdLearningEngine
│   ├── SystemConditionMonitor
│   └── DynamicThresholdAdjuster
└── EnhancedObservabilityProvider
    ├── CircuitBreakerMetricsCollector
    ├── FailurePatternAnalyzer
    └── DebugInsightsGenerator
```

## Technical Design Specifications

### 1. Service-Specific Circuit Breaker Manager

#### Core Interface

```python
@dataclass
class ServiceCircuitBreakerConfig:
    """Configuration for service-specific circuit breakers"""
    
    service_name: str
    api_endpoint_patterns: List[str]
    
    # Failure detection
    failure_threshold: int = 5
    timeout_threshold_ms: float = 30000
    error_rate_threshold: float = 0.5  # 50%
    
    # Recovery settings
    recovery_timeout_seconds: float = 60
    half_open_max_requests: int = 3
    
    # Service-specific thresholds
    rate_limit_backoff_multiplier: float = 2.0
    quota_exceeded_cooldown_minutes: float = 15
    authentication_failure_escalation: bool = True
    
    # Integration settings
    coordinate_with_cost_breaker: bool = True
    cascade_to_fallback: bool = True
    enable_adaptive_thresholds: bool = True

class ServiceSpecificCircuitBreaker(ABC):
    """Base class for service-specific circuit breakers"""
    
    def __init__(self, 
                 config: ServiceCircuitBreakerConfig,
                 cost_breaker: Optional[CostBasedCircuitBreaker] = None,
                 coordinator: Optional['CrossServiceCoordinator'] = None):
        self.config = config
        self.cost_breaker = cost_breaker
        self.coordinator = coordinator
        
        # Service-specific state
        self.state = CircuitBreakerState.CLOSED
        self.service_health_score = 1.0
        self.api_quota_status = {}
        
        # Enhanced metrics
        self.service_metrics = ServiceMetrics()
        self.failure_patterns = FailurePatternTracker()
        
    @abstractmethod
    def should_allow_request(self, request_context: Dict[str, Any]) -> Tuple[bool, str]:
        """Enhanced request filtering with service-specific logic"""
        pass
        
    @abstractmethod
    def record_service_specific_failure(self, 
                                      error: Exception, 
                                      response_metadata: Dict[str, Any]) -> None:
        """Record service-specific failure patterns"""
        pass
        
    def execute_with_protection(self, 
                              operation: Callable,
                              request_context: Dict[str, Any],
                              **kwargs) -> Any:
        """Execute operation with enhanced protection"""
        
        # Pre-execution checks
        allowed, reason = self.should_allow_request(request_context)
        if not allowed:
            raise ServiceSpecificCircuitBreakerError(f"{self.config.service_name}: {reason}")
        
        # Cost-aware coordination
        if self.cost_breaker and self.config.coordinate_with_cost_breaker:
            try:
                return self.cost_breaker.call(operation, **kwargs)
            except CircuitBreakerError as e:
                self._handle_cost_breaker_block(e, request_context)
                raise
        
        # Direct execution with service monitoring
        return self._execute_with_monitoring(operation, request_context, **kwargs)
```

#### OpenAI-Specific Circuit Breaker

```python
class OpenAICircuitBreaker(ServiceSpecificCircuitBreaker):
    """OpenAI-specific circuit breaker with API-aware failure detection"""
    
    def __init__(self, config: ServiceCircuitBreakerConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # OpenAI-specific tracking
        self.rate_limit_windows = {}
        self.model_specific_health = defaultdict(float)
        self.token_usage_patterns = TokenUsageTracker()
        
    def should_allow_request(self, request_context: Dict[str, Any]) -> Tuple[bool, str]:
        """OpenAI-specific request filtering"""
        
        # Check model-specific health
        model = request_context.get('model', 'unknown')
        if self.model_specific_health[model] < 0.3:
            return False, f"Model {model} health too low: {self.model_specific_health[model]}"
        
        # Check rate limiting patterns
        if self._is_rate_limit_approaching(request_context):
            return False, "Approaching rate limit threshold"
        
        # Check token quota patterns
        estimated_tokens = request_context.get('estimated_tokens', 0)
        if self._would_exceed_token_budget(estimated_tokens):
            return False, "Would exceed token budget threshold"
        
        return super().should_allow_request(request_context)
    
    def record_service_specific_failure(self, error: Exception, response_metadata: Dict[str, Any]):
        """Record OpenAI-specific failure patterns"""
        
        if hasattr(error, 'status_code'):
            if error.status_code == 429:  # Rate limit
                self._update_rate_limit_tracking(response_metadata)
            elif error.status_code == 401:  # Auth failure
                self._handle_auth_failure(response_metadata)
            elif error.status_code >= 500:  # Server error
                model = response_metadata.get('model')
                if model:
                    self.model_specific_health[model] *= 0.7  # Degrade health
        
        # Update token usage patterns
        self.token_usage_patterns.record_failure(response_metadata)
        
        # Record in failure patterns
        self.failure_patterns.add_failure(
            error_type=type(error).__name__,
            error_details=str(error),
            context=response_metadata
        )
```

### 2. Cross-Service Coordination

#### Circuit Breaker Orchestrator

```python
class CircuitBreakerOrchestrator:
    """Coordinates multiple circuit breakers for system-wide protection"""
    
    def __init__(self, 
                 cost_breaker_manager: CostCircuitBreakerManager,
                 service_breakers: Dict[str, ServiceSpecificCircuitBreaker],
                 production_breakers: Dict[str, ProductionCircuitBreaker]):
        
        self.cost_breaker_manager = cost_breaker_manager
        self.service_breakers = service_breakers
        self.production_breakers = production_breakers
        
        # Coordination state
        self.system_health_score = 1.0
        self.cascade_rules = self._initialize_cascade_rules()
        self.coordination_history = deque(maxlen=1000)
        
        # Failure correlation
        self.failure_correlator = FailureCorrelationAnalyzer()
        
    def execute_coordinated_request(self,
                                  service_name: str,
                                  operation: Callable,
                                  request_context: Dict[str, Any],
                                  **kwargs) -> Any:
        """Execute request with coordinated circuit breaker protection"""
        
        coordination_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # System-level health check
            if not self._is_system_healthy():
                raise SystemCircuitBreakerError("System health too degraded for new requests")
            
            # Progressive degradation check
            degradation_level = self._assess_degradation_level()
            if degradation_level > 0.7:
                request_context['degraded_mode'] = True
                request_context['max_timeout'] = request_context.get('max_timeout', 30) * 0.5
            
            # Service-specific execution
            if service_name in self.service_breakers:
                result = self.service_breakers[service_name].execute_with_protection(
                    operation, request_context, **kwargs
                )
            else:
                # Fallback to cost-based protection
                result = self.cost_breaker_manager.execute_with_protection(
                    f"general_{service_name}", operation, "api_call", **kwargs
                )
            
            # Record success
            self._record_coordination_success(coordination_id, service_name, time.time() - start_time)
            return result
            
        except Exception as e:
            # Coordinated failure handling
            self._handle_coordinated_failure(coordination_id, service_name, e, request_context)
            raise
    
    def _assess_degradation_level(self) -> float:
        """Assess system degradation level (0.0 = healthy, 1.0 = critical)"""
        
        factors = []
        
        # Cost breaker status
        cost_status = self.cost_breaker_manager.get_system_status()
        if cost_status['system_health']['status'] == 'degraded':
            factors.append(0.5)
        elif cost_status['system_health']['status'] == 'budget_limited':
            factors.append(0.7)
        
        # Service breaker status
        service_failures = 0
        for service, breaker in self.service_breakers.items():
            if breaker.state == CircuitBreakerState.OPEN:
                factors.append(0.8)
                service_failures += 1
            elif breaker.state == CircuitBreakerState.HALF_OPEN:
                factors.append(0.4)
        
        # System-wide failure correlation
        correlation_score = self.failure_correlator.get_correlation_severity()
        factors.append(correlation_score)
        
        return min(1.0, statistics.mean(factors) if factors else 0.0)
```

#### Failure Correlation Analyzer

```python
class FailureCorrelationAnalyzer:
    """Analyzes failure patterns across services to detect system-wide issues"""
    
    def __init__(self):
        self.failure_timeline = deque(maxlen=10000)
        self.correlation_patterns = {}
        self.system_event_tracker = SystemEventTracker()
        
    def record_failure(self, 
                      service: str, 
                      failure_type: str, 
                      timestamp: float,
                      context: Dict[str, Any]):
        """Record failure for correlation analysis"""
        
        failure_event = FailureEvent(
            service=service,
            failure_type=failure_type,
            timestamp=timestamp,
            context=context
        )
        
        self.failure_timeline.append(failure_event)
        self._analyze_correlations()
    
    def _analyze_correlations(self):
        """Analyze recent failures for correlation patterns"""
        
        if len(self.failure_timeline) < 3:
            return
        
        recent_window = 300  # 5 minutes
        now = time.time()
        recent_failures = [
            f for f in self.failure_timeline 
            if now - f.timestamp < recent_window
        ]
        
        if len(recent_failures) < 2:
            return
        
        # Time-based correlation
        self._detect_time_correlations(recent_failures)
        
        # Service pattern correlation  
        self._detect_service_patterns(recent_failures)
        
        # External factor correlation
        self._detect_external_correlations(recent_failures)
    
    def get_correlation_severity(self) -> float:
        """Get current correlation severity (0.0 = no correlation, 1.0 = high correlation)"""
        
        now = time.time()
        recent_patterns = [
            pattern for pattern in self.correlation_patterns.values()
            if now - pattern.last_seen < 600  # 10 minutes
        ]
        
        if not recent_patterns:
            return 0.0
        
        # Calculate severity based on pattern strength and recency
        severities = [p.severity * (1.0 - (now - p.last_seen) / 600) for p in recent_patterns]
        return min(1.0, max(severities) if severities else 0.0)
```

### 3. Adaptive Threshold Management

#### Threshold Learning Engine

```python
class ThresholdLearningEngine:
    """Learns optimal circuit breaker thresholds from historical data"""
    
    def __init__(self):
        self.historical_data = HistoricalDataStore()
        self.learning_models = {
            'failure_prediction': FailurePredictionModel(),
            'performance_prediction': PerformancePredictionModel(),
            'recovery_estimation': RecoveryEstimationModel()
        }
        self.threshold_recommendations = {}
        
    def analyze_and_recommend(self, 
                            breaker_id: str, 
                            current_config: ServiceCircuitBreakerConfig) -> Dict[str, Any]:
        """Analyze historical performance and recommend threshold adjustments"""
        
        # Gather historical data
        historical_performance = self.historical_data.get_breaker_history(
            breaker_id, days=30
        )
        
        if not historical_performance:
            return {'recommendations': [], 'confidence': 0.0}
        
        recommendations = []
        
        # Analyze failure threshold effectiveness
        failure_analysis = self._analyze_failure_threshold_effectiveness(
            historical_performance, current_config.failure_threshold
        )
        
        if failure_analysis['adjustment_needed']:
            recommendations.append({
                'parameter': 'failure_threshold',
                'current_value': current_config.failure_threshold,
                'recommended_value': failure_analysis['recommended_value'],
                'confidence': failure_analysis['confidence'],
                'reasoning': failure_analysis['reasoning']
            })
        
        # Analyze recovery timeout optimization
        recovery_analysis = self._analyze_recovery_timeout_effectiveness(
            historical_performance, current_config.recovery_timeout_seconds
        )
        
        if recovery_analysis['adjustment_needed']:
            recommendations.append({
                'parameter': 'recovery_timeout_seconds',
                'current_value': current_config.recovery_timeout_seconds,
                'recommended_value': recovery_analysis['recommended_value'],
                'confidence': recovery_analysis['confidence'],
                'reasoning': recovery_analysis['reasoning']
            })
        
        # Analyze error rate threshold
        error_rate_analysis = self._analyze_error_rate_threshold(
            historical_performance, current_config.error_rate_threshold
        )
        
        if error_rate_analysis['adjustment_needed']:
            recommendations.append({
                'parameter': 'error_rate_threshold',
                'current_value': current_config.error_rate_threshold,
                'recommended_value': error_rate_analysis['recommended_value'],
                'confidence': error_rate_analysis['confidence'],
                'reasoning': error_rate_analysis['reasoning']
            })
        
        overall_confidence = (
            statistics.mean([r['confidence'] for r in recommendations]) 
            if recommendations else 1.0
        )
        
        return {
            'recommendations': recommendations,
            'confidence': overall_confidence,
            'analysis_timestamp': time.time(),
            'data_quality_score': self._assess_data_quality(historical_performance)
        }
```

#### Dynamic Threshold Adjuster

```python
class DynamicThresholdAdjuster:
    """Dynamically adjusts circuit breaker thresholds based on real-time conditions"""
    
    def __init__(self, learning_engine: ThresholdLearningEngine):
        self.learning_engine = learning_engine
        self.adjustment_history = deque(maxlen=1000)
        self.system_condition_monitor = SystemConditionMonitor()
        
    def should_adjust_thresholds(self, 
                               breaker: ServiceSpecificCircuitBreaker,
                               current_conditions: Dict[str, Any]) -> bool:
        """Determine if thresholds should be adjusted based on current conditions"""
        
        # Check if enough data is available
        if not self._has_sufficient_data(breaker):
            return False
        
        # Check time since last adjustment
        last_adjustment = self._get_last_adjustment_time(breaker)
        if last_adjustment and time.time() - last_adjustment < 3600:  # 1 hour minimum
            return False
        
        # Check if conditions warrant adjustment
        condition_score = self._assess_adjustment_need(breaker, current_conditions)
        return condition_score > 0.7  # High confidence threshold
    
    def apply_dynamic_adjustments(self,
                                breaker: ServiceSpecificCircuitBreaker,
                                current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dynamic threshold adjustments"""
        
        adjustments = {}
        
        # System load adjustment
        system_load = current_conditions.get('system_load', 0.5)
        if system_load > 0.8:
            # Under high load, be more aggressive
            adjustments['failure_threshold'] = max(2, int(breaker.config.failure_threshold * 0.7))
            adjustments['timeout_threshold_ms'] = breaker.config.timeout_threshold_ms * 0.8
        elif system_load < 0.3:
            # Under low load, be more lenient
            adjustments['failure_threshold'] = min(10, int(breaker.config.failure_threshold * 1.3))
            adjustments['timeout_threshold_ms'] = breaker.config.timeout_threshold_ms * 1.2
        
        # Error rate trend adjustment
        error_rate_trend = current_conditions.get('error_rate_trend', 'stable')
        if error_rate_trend == 'increasing':
            adjustments['error_rate_threshold'] = max(0.1, breaker.config.error_rate_threshold * 0.8)
        elif error_rate_trend == 'decreasing':
            adjustments['error_rate_threshold'] = min(0.8, breaker.config.error_rate_threshold * 1.1)
        
        # Budget pressure adjustment
        budget_pressure = current_conditions.get('budget_pressure', 0.0)
        if budget_pressure > 0.8:
            # Under budget pressure, be more conservative
            adjustments['recovery_timeout_seconds'] = breaker.config.recovery_timeout_seconds * 1.5
        
        # Apply adjustments
        if adjustments:
            self._apply_adjustments(breaker, adjustments)
            self._record_adjustment(breaker, adjustments, current_conditions)
        
        return adjustments
```

### 4. Enhanced Observability and Monitoring

#### Circuit Breaker Metrics Collector

```python
@dataclass
class CircuitBreakerMetrics:
    """Comprehensive circuit breaker metrics"""
    
    # Basic metrics
    breaker_id: str
    service_name: str
    timestamp: float
    
    # State information
    current_state: CircuitBreakerState
    state_duration_ms: float
    state_transitions_count: int
    
    # Performance metrics
    request_count_success: int = 0
    request_count_failure: int = 0
    request_count_blocked: int = 0
    request_count_throttled: int = 0
    
    # Timing metrics
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Error analysis
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    server_error_rate: float = 0.0
    client_error_rate: float = 0.0
    
    # Recovery metrics
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    recovery_success_rate: float = 0.0
    avg_recovery_time_ms: float = 0.0
    
    # Cost metrics (when integrated with cost breakers)
    estimated_cost_saved: float = 0.0
    actual_cost_impact: float = 0.0
    
    # Correlation metrics
    correlated_failures: int = 0
    cascade_triggers: int = 0
    
    def to_prometheus_metrics(self) -> List[str]:
        """Convert to Prometheus metric format"""
        metrics = []
        
        labels = f'breaker_id="{self.breaker_id}",service="{self.service_name}"'
        
        # State metrics
        state_value = {'closed': 0, 'half_open': 1, 'open': 2}.get(self.current_state.value, -1)
        metrics.append(f'circuit_breaker_state{{{labels}}} {state_value}')
        metrics.append(f'circuit_breaker_state_duration_ms{{{labels}}} {self.state_duration_ms}')
        
        # Request metrics
        metrics.append(f'circuit_breaker_requests_total{{{labels},result="success"}} {self.request_count_success}')
        metrics.append(f'circuit_breaker_requests_total{{{labels},result="failure"}} {self.request_count_failure}')
        metrics.append(f'circuit_breaker_requests_total{{{labels},result="blocked"}} {self.request_count_blocked}')
        
        # Performance metrics
        metrics.append(f'circuit_breaker_response_time_ms{{{labels},quantile="0.50"}} {self.avg_response_time_ms}')
        metrics.append(f'circuit_breaker_response_time_ms{{{labels},quantile="0.95"}} {self.p95_response_time_ms}')
        metrics.append(f'circuit_breaker_response_time_ms{{{labels},quantile="0.99"}} {self.p99_response_time_ms}')
        
        # Error rates
        metrics.append(f'circuit_breaker_error_rate{{{labels}}} {self.error_rate}')
        metrics.append(f'circuit_breaker_timeout_rate{{{labels}}} {self.timeout_rate}')
        
        return metrics

class CircuitBreakerMetricsCollector:
    """Collects and aggregates circuit breaker metrics"""
    
    def __init__(self, 
                 export_interval_seconds: int = 60,
                 retention_hours: int = 24):
        
        self.export_interval = export_interval_seconds
        self.retention_hours = retention_hours
        
        # Metrics storage
        self.current_metrics: Dict[str, CircuitBreakerMetrics] = {}
        self.historical_metrics = deque(maxlen=retention_hours * 3600 // export_interval_seconds)
        
        # Aggregation state
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_counts: Dict[str, Counter] = defaultdict(Counter)
        
        # Export configuration
        self.prometheus_enabled = False
        self.dashboard_callback: Optional[Callable] = None
        
    def record_request(self,
                      breaker_id: str,
                      service_name: str,
                      success: bool,
                      response_time_ms: float,
                      error_type: Optional[str] = None,
                      blocked: bool = False,
                      throttled: bool = False):
        """Record individual request metrics"""
        
        # Ensure metrics object exists
        if breaker_id not in self.current_metrics:
            self.current_metrics[breaker_id] = CircuitBreakerMetrics(
                breaker_id=breaker_id,
                service_name=service_name,
                timestamp=time.time(),
                current_state=CircuitBreakerState.CLOSED
            )
        
        metrics = self.current_metrics[breaker_id]
        
        # Update counters
        if blocked:
            metrics.request_count_blocked += 1
        elif throttled:
            metrics.request_count_throttled += 1
        elif success:
            metrics.request_count_success += 1
        else:
            metrics.request_count_failure += 1
            if error_type:
                self.error_counts[breaker_id][error_type] += 1
        
        # Update timing metrics
        if not blocked:
            self.response_times[breaker_id].append(response_time_ms)
            self._update_timing_metrics(breaker_id)
    
    def generate_insights_report(self, 
                               time_window_hours: int = 1) -> Dict[str, Any]:
        """Generate actionable insights from metrics"""
        
        insights = {
            'timestamp': time.time(),
            'time_window_hours': time_window_hours,
            'breaker_insights': {},
            'system_insights': {},
            'recommendations': []
        }
        
        # Per-breaker insights
        for breaker_id, metrics in self.current_metrics.items():
            breaker_insights = self._analyze_breaker_performance(breaker_id, metrics)
            insights['breaker_insights'][breaker_id] = breaker_insights
            
            # Add recommendations
            if breaker_insights['recommendations']:
                insights['recommendations'].extend(breaker_insights['recommendations'])
        
        # System-wide insights
        insights['system_insights'] = self._analyze_system_wide_patterns()
        
        return insights
```

## Configuration Schema Design

### Enhanced Configuration Structure

```python
@dataclass
class EnhancedCircuitBreakerConfiguration:
    """Complete configuration for enhanced circuit breaker system"""
    
    # System-level settings
    system_config: SystemCircuitBreakerConfig
    
    # Service-specific configurations
    service_configs: Dict[str, ServiceCircuitBreakerConfig]
    
    # Coordination settings
    coordination_config: CoordinationConfig
    
    # Adaptive threshold settings
    adaptive_config: AdaptiveThresholdConfig
    
    # Monitoring and observability
    observability_config: ObservabilityConfig
    
    # Integration settings
    integration_config: IntegrationConfig

@dataclass  
class SystemCircuitBreakerConfig:
    """System-wide circuit breaker configuration"""
    
    # Global health thresholds
    system_health_threshold: float = 0.3  # Minimum system health
    cascade_failure_threshold: int = 3     # Max failed services before cascade
    
    # Emergency settings
    emergency_mode_enabled: bool = True
    emergency_recovery_timeout_minutes: float = 30
    
    # Resource protection
    max_concurrent_recovery_attempts: int = 5
    resource_exhaustion_threshold: float = 0.9

@dataclass
class CoordinationConfig:
    """Configuration for cross-service coordination"""
    
    # Cascade behavior
    enable_failure_cascading: bool = True
    cascade_delay_ms: float = 100
    max_cascade_depth: int = 3
    
    # Progressive degradation
    enable_progressive_degradation: bool = True
    degradation_steps: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    
    # Coordination timeouts
    coordination_timeout_ms: float = 5000
    cross_service_sync_interval_ms: float = 1000

@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive threshold management"""
    
    # Learning settings
    enable_threshold_learning: bool = True
    learning_window_days: int = 30
    min_data_points_for_learning: int = 100
    
    # Adjustment constraints
    max_adjustment_percentage: float = 0.3  # Max 30% adjustment
    adjustment_cooldown_hours: int = 1
    confidence_threshold_for_adjustment: float = 0.8
    
    # Real-time adaptation
    enable_real_time_adjustment: bool = False
    real_time_adjustment_sensitivity: float = 0.1

@dataclass 
class ObservabilityConfig:
    """Configuration for enhanced observability"""
    
    # Metrics collection
    enable_detailed_metrics: bool = True
    metrics_export_interval_seconds: int = 60
    metrics_retention_hours: int = 24
    
    # Prometheus integration
    enable_prometheus_export: bool = False
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    
    # Dashboard integration
    enable_dashboard_integration: bool = True
    dashboard_update_interval_seconds: int = 30
    
    # Alerting
    enable_alerting: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "webhook"])
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.1,
        "circuit_open_duration_minutes": 10,
        "cascade_failure_count": 2
    })
```

## Integration Specifications

### Integration with Existing Systems

```python
class EnhancedCircuitBreakerIntegration:
    """Integration layer for enhanced circuit breakers with existing systems"""
    
    def __init__(self, 
                 existing_cost_manager: CostCircuitBreakerManager,
                 existing_production_breakers: Dict[str, ProductionCircuitBreaker],
                 fallback_orchestrator: FallbackOrchestrator):
        
        self.cost_manager = existing_cost_manager
        self.production_breakers = existing_production_breakers
        self.fallback_orchestrator = fallback_orchestrator
        
        # Enhanced components
        self.service_manager = ServiceSpecificCircuitBreakerManager()
        self.coordinator = CircuitBreakerOrchestrator(
            cost_breaker_manager=existing_cost_manager,
            service_breakers=self.service_manager.get_all_breakers(),
            production_breakers=existing_production_breakers
        )
        
        # Migration and compatibility
        self.compatibility_layer = BackwardCompatibilityLayer()
        self.migration_manager = SystemMigrationManager()
    
    def migrate_existing_configuration(self) -> MigrationReport:
        """Migrate existing circuit breaker configuration to enhanced system"""
        
        migration_report = MigrationReport()
        
        # Migrate cost-based configurations
        cost_configs = self._extract_cost_configurations()
        for config in cost_configs:
            enhanced_config = self._convert_cost_config_to_enhanced(config)
            migration_report.add_conversion('cost_breaker', config.name, enhanced_config)
        
        # Migrate production configurations  
        prod_configs = self._extract_production_configurations()
        for config in prod_configs:
            enhanced_config = self._convert_production_config_to_enhanced(config)
            migration_report.add_conversion('production_breaker', config.id, enhanced_config)
        
        # Validate migrations
        validation_results = self._validate_migrations(migration_report)
        migration_report.validation_results = validation_results
        
        return migration_report
    
    def create_backward_compatible_interface(self) -> BackwardCompatibleInterface:
        """Create interface that maintains backward compatibility"""
        
        return BackwardCompatibleInterface(
            enhanced_system=self,
            cost_manager_proxy=CostManagerProxy(self.cost_manager, self.coordinator),
            production_breaker_proxy=ProductionBreakerProxy(self.production_breakers, self.coordinator)
        )
```

## Implementation Roadmap

### Phase 1: Foundation Enhancement (Weeks 1-2)

1. **Service-Specific Circuit Breaker Framework**
   - Implement `ServiceSpecificCircuitBreaker` base class
   - Create `OpenAICircuitBreaker` and `PerplexityCircuitBreaker`
   - Integrate with existing cost-based breakers
   - Add comprehensive unit tests

2. **Basic Coordination Layer**
   - Implement `CircuitBreakerOrchestrator`
   - Add failure correlation tracking
   - Create coordination integration points

### Phase 2: Intelligence and Adaptation (Weeks 3-4)

1. **Adaptive Threshold Management**
   - Implement `ThresholdLearningEngine`
   - Create `DynamicThresholdAdjuster`
   - Add machine learning models for threshold optimization

2. **Enhanced Observability**
   - Implement `CircuitBreakerMetricsCollector`
   - Add Prometheus metrics export
   - Create dashboard integration

### Phase 3: Production Integration (Weeks 5-6)

1. **Migration and Compatibility**
   - Implement migration tools
   - Create backward compatibility layer
   - Add comprehensive integration tests

2. **Monitoring and Alerting**
   - Implement advanced alerting rules
   - Add failure pattern analysis
   - Create operational dashboards

## Performance Specifications

### Performance Targets

- **Request Processing Overhead**: < 2ms per request
- **Memory Usage**: < 50MB additional for full system
- **Metrics Collection**: < 1ms per metric point
- **Coordination Latency**: < 10ms for cross-service coordination
- **Threshold Adjustment**: < 100ms for real-time adjustments

### Scalability Requirements

- **Concurrent Requests**: Support 10,000+ concurrent requests per service
- **Metrics Retention**: 24 hours of detailed metrics per breaker
- **Historical Analysis**: 30 days of learning data
- **Service Scaling**: Support 20+ simultaneous service-specific breakers

## Security Considerations

### Data Protection

- **Metrics Data**: Encrypt sensitive metrics at rest and in transit
- **Configuration**: Secure configuration storage with access controls
- **API Keys**: Secure integration with existing credential management

### Access Control

- **Administrative Functions**: Role-based access for configuration changes
- **Monitoring Access**: Separate read-only access for monitoring
- **Emergency Controls**: Secure emergency override mechanisms

## Testing Strategy

### Test Coverage Requirements

1. **Unit Tests**: 95%+ coverage for all circuit breaker components
2. **Integration Tests**: Full integration with existing systems
3. **Performance Tests**: Load testing under realistic conditions
4. **Chaos Testing**: Fault injection and recovery validation
5. **Migration Tests**: Comprehensive migration scenario testing

### Test Scenarios

- **Service-Specific Failure Isolation**
- **Cross-Service Failure Correlation**
- **Adaptive Threshold Learning**
- **Emergency Recovery Procedures**
- **Backward Compatibility Validation**

## Conclusion

This enhanced circuit breaker design builds upon the existing excellent foundation in the Clinical Metabolomics Oracle system while adding sophisticated service-specific protection, intelligent coordination, and adaptive capabilities. The design prioritizes:

1. **Seamless Integration**: Full compatibility with existing systems
2. **Enhanced Protection**: Service-specific isolation and coordination
3. **Intelligent Adaptation**: Learning and self-optimization
4. **Comprehensive Observability**: Deep insights and monitoring
5. **Production Readiness**: Enterprise-grade reliability and performance

The implementation follows a phased approach that allows for gradual rollout and validation, ensuring system stability throughout the enhancement process.

**Files Referenced:**
- `/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/lightrag_integration/cost_based_circuit_breaker.py`
- `/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/lightrag_integration/production_load_balancer.py`
- `/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/lightrag_integration/fallback_decision_logging_metrics.py`