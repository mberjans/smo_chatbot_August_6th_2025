"""
CMO Load Metrics Enhanced Methods - For Integration into Main File
==================================================================

This file contains the enhanced methods for the CMOLoadMetrics class
that provide real-time monitoring, advanced analytics, and comprehensive
analysis capabilities.

These methods should be integrated into the CMOLoadMetrics class in
cmo_metrics_and_integration.py
"""

# Methods to add to CMOLoadMetrics class:

async def start_real_time_monitoring(self):
    """Start real-time performance monitoring at 100ms intervals."""
    if self._monitoring_active:
        return
    
    self._monitoring_active = True
    self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    logging.info("CMO real-time performance monitoring started (100ms intervals)")

async def stop_real_time_monitoring(self):
    """Stop real-time performance monitoring."""
    if not self._monitoring_active:
        return
    
    self._monitoring_active = False
    if self._monitoring_task:
        self._monitoring_task.cancel()
        try:
            await self._monitoring_task
        except asyncio.CancelledError:
            pass
    
    logging.info("CMO real-time performance monitoring stopped")

async def _monitoring_loop(self):
    """Real-time monitoring loop - samples every 100ms."""
    while self._monitoring_active:
        try:
            await self._collect_performance_sample()
            await asyncio.sleep(self._monitoring_interval)
        except Exception as e:
            logging.error(f"Error in CMO monitoring loop: {e}")
            await asyncio.sleep(self._monitoring_interval)

async def _collect_performance_sample(self):
    """Collect a single performance sample."""
    current_time = time.time()
    
    # Sample system resources
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Calculate recent performance metrics
        recent_response_times = list(self.response_times)[-50:] if len(self.response_times) > 0 else []
        recent_throughput = self._calculate_recent_throughput()
        
        sample = {
            'timestamp': current_time,
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'recent_avg_response_time': statistics.mean(recent_response_times) if recent_response_times else 0.0,
            'recent_throughput': recent_throughput,
            'success_rate': self.get_success_rate(),
            'cache_hit_rate': self.cache_metrics.get_overall_hit_rate(),
            'lightrag_success_rate': self.lightrag_metrics.get_success_rate(),
            'fallback_success_rate': self.fallback_metrics.get_overall_success_rate()
        }
        
        async with self._metrics_lock:
            self.performance_samples.append(sample)
            self._update_trend_analysis(sample)
            self._update_resource_efficiency(sample)
            self._update_performance_grade(sample)
        
    except Exception as e:
        logging.warning(f"Error collecting performance sample: {e}")

def _calculate_recent_throughput(self) -> float:
    """Calculate throughput over recent time window."""
    current_time = time.time()
    window_seconds = 10.0
    cutoff_time = current_time - window_seconds
    
    # Count operations in recent window
    recent_ops = 0
    for sample in reversed(self.performance_samples):
        if sample['timestamp'] >= cutoff_time:
            recent_ops += 1
        else:
            break
    
    return recent_ops / window_seconds

def _update_trend_analysis(self, sample: Dict[str, Any]):
    """Update trend analysis with new sample."""
    metrics_to_track = [
        'recent_avg_response_time', 'recent_throughput', 'success_rate',
        'cache_hit_rate', 'memory_mb', 'cpu_percent'
    ]
    
    for metric in metrics_to_track:
        if metric in sample:
            self.trend_analysis[metric].append(sample[metric])
            # Keep only last 100 samples for trend analysis
            if len(self.trend_analysis[metric]) > 100:
                self.trend_analysis[metric] = self.trend_analysis[metric][-100:]

def _update_resource_efficiency(self, sample: Dict[str, Any]):
    """Update resource efficiency calculations."""
    # Resource efficiency = throughput per unit of resource usage
    throughput = sample.get('recent_throughput', 0.0)
    memory_mb = sample.get('memory_mb', 1.0)
    cpu_percent = sample.get('cpu_percent', 1.0)
    
    # Efficiency score: throughput per MB of memory and per CPU%
    memory_efficiency = throughput / max(memory_mb, 1.0)
    cpu_efficiency_score = throughput / max(cpu_percent, 1.0)
    
    self.resource_efficiency_samples.append((memory_efficiency + cpu_efficiency_score) / 2)
    
    # Keep only recent samples
    if len(self.resource_efficiency_samples) > 500:
        self.resource_efficiency_samples = self.resource_efficiency_samples[-500:]

def _update_performance_grade(self, sample: Dict[str, Any]):
    """Update current performance grade based on sample."""
    success_rate = sample.get('success_rate', 0.0)
    response_time = sample.get('recent_avg_response_time', float('inf'))
    
    # Grade based on success rate and response time
    if success_rate >= 0.99 and response_time <= 500:
        grade = PerformanceGrade.A_PLUS
    elif success_rate >= 0.95 and response_time <= 1000:
        grade = PerformanceGrade.A
    elif success_rate >= 0.90 and response_time <= 1500:
        grade = PerformanceGrade.B
    elif success_rate >= 0.85 and response_time <= 2500:
        grade = PerformanceGrade.C
    elif success_rate >= 0.75 and response_time <= 4000:
        grade = PerformanceGrade.D
    else:
        grade = PerformanceGrade.F
    
    if grade != self.current_grade:
        self.grade_history.append((datetime.now(), grade))
        self.current_grade = grade

def get_trend_direction(self, metric: str) -> MetricTrendDirection:
    """Analyze trend direction for a specific metric."""
    if metric not in self.trend_analysis or len(self.trend_analysis[metric]) < 10:
        return MetricTrendDirection.STABLE
    
    values = self.trend_analysis[metric][-20:]  # Last 20 samples
    
    # Calculate linear trend
    x = np.arange(len(values))
    z = np.polyfit(x, values, 1)
    slope = z[0]
    
    # Calculate coefficient of variation
    cv = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) > 0 else 0
    
    # Determine trend direction
    if cv > 0.3:  # High variability
        return MetricTrendDirection.VOLATILE
    elif abs(slope) < 0.01:  # Very small slope
        return MetricTrendDirection.STABLE
    elif slope > 0:
        return MetricTrendDirection.IMPROVING if metric in ['success_rate', 'cache_hit_rate', 'recent_throughput'] else MetricTrendDirection.DEGRADING
    else:
        return MetricTrendDirection.DEGRADING if metric in ['success_rate', 'cache_hit_rate', 'recent_throughput'] else MetricTrendDirection.IMPROVING

def get_component_health_assessment(self) -> Dict[str, Dict[str, Any]]:
    """Assess health of all CMO components."""
    health_assessment = {}
    
    # LightRAG Health
    lightrag_success_rate = self.lightrag_metrics.get_success_rate()
    lightrag_avg_response = statistics.mean(self.lightrag_metrics.response_times) if self.lightrag_metrics.response_times else 0
    
    if lightrag_success_rate >= 0.95 and lightrag_avg_response <= 1000:
        lightrag_health = ComponentHealthStatus.HEALTHY
    elif lightrag_success_rate >= 0.90 and lightrag_avg_response <= 2000:
        lightrag_health = ComponentHealthStatus.WARNING
    elif lightrag_success_rate >= 0.80:
        lightrag_health = ComponentHealthStatus.CRITICAL
    else:
        lightrag_health = ComponentHealthStatus.FAILING
    
    health_assessment['lightrag'] = {
        'status': lightrag_health,
        'success_rate': lightrag_success_rate,
        'avg_response_time': lightrag_avg_response,
        'mode_distribution': self.lightrag_metrics.get_mode_distribution()
    }
    
    # Cache Health
    cache_hit_rate = self.cache_metrics.get_overall_hit_rate()
    cache_effectiveness = self.cache_metrics.get_cache_effectiveness_score()
    
    if cache_hit_rate >= 0.80 and cache_effectiveness >= 0.8:
        cache_health = ComponentHealthStatus.HEALTHY
    elif cache_hit_rate >= 0.70 and cache_effectiveness >= 0.6:
        cache_health = ComponentHealthStatus.WARNING
    elif cache_hit_rate >= 0.50:
        cache_health = ComponentHealthStatus.CRITICAL
    else:
        cache_health = ComponentHealthStatus.FAILING
    
    health_assessment['cache'] = {
        'status': cache_health,
        'overall_hit_rate': cache_hit_rate,
        'effectiveness_score': cache_effectiveness,
        'tier_hit_rates': self.cache_metrics.get_tier_hit_rates()
    }
    
    # Circuit Breaker Health
    cb_availability = self.circuit_breaker_metrics.get_availability_percentage()
    cb_recovery_rate = self.circuit_breaker_metrics.get_recovery_success_rate()
    
    if cb_availability >= 95 and cb_recovery_rate >= 0.9:
        cb_health = ComponentHealthStatus.HEALTHY
    elif cb_availability >= 90 and cb_recovery_rate >= 0.8:
        cb_health = ComponentHealthStatus.WARNING
    elif cb_availability >= 80:
        cb_health = ComponentHealthStatus.CRITICAL
    else:
        cb_health = ComponentHealthStatus.FAILING
    
    health_assessment['circuit_breaker'] = {
        'status': cb_health,
        'availability_percent': cb_availability,
        'recovery_success_rate': cb_recovery_rate,
        'state_distribution': self.circuit_breaker_metrics.get_state_distribution()
    }
    
    # Fallback System Health
    fallback_success_rate = self.fallback_metrics.get_overall_success_rate()
    cost_efficiency = self.fallback_metrics.get_cost_efficiency_score()
    
    if fallback_success_rate >= 0.95 and cost_efficiency >= 0.8:
        fallback_health = ComponentHealthStatus.HEALTHY
    elif fallback_success_rate >= 0.90 and cost_efficiency >= 0.6:
        fallback_health = ComponentHealthStatus.WARNING
    elif fallback_success_rate >= 0.80:
        fallback_health = ComponentHealthStatus.CRITICAL
    else:
        fallback_health = ComponentHealthStatus.FAILING
    
    health_assessment['fallback_system'] = {
        'status': fallback_health,
        'overall_success_rate': fallback_success_rate,
        'cost_efficiency_score': cost_efficiency,
        'success_rates_by_level': self.fallback_metrics.get_fallback_success_rates()
    }
    
    return health_assessment

def get_advanced_percentiles(self) -> Dict[str, float]:
    """Calculate advanced response time percentiles with trend analysis."""
    if not self.response_times:
        return {}
    
    times = sorted(self.response_times)
    percentiles = super().get_percentiles()
    
    # Add additional percentiles for detailed analysis
    percentiles.update({
        'p25': np.percentile(times, 25),
        'p85': np.percentile(times, 85),
        'p99.5': np.percentile(times, 99.5) if len(times) >= 200 else times[-1],
        'p99.9': np.percentile(times, 99.9) if len(times) >= 1000 else times[-1]
    })
    
    # Add trend information
    percentiles['trend_direction'] = self.get_trend_direction('recent_avg_response_time').value
    
    return percentiles

def get_resource_efficiency_score(self) -> Dict[str, float]:
    """Calculate comprehensive resource efficiency scores."""
    if not self.resource_efficiency_samples:
        return {'overall': 0.0, 'memory': 0.0, 'cpu': 0.0, 'trend': 0.0}
    
    # Overall efficiency score
    overall_efficiency = statistics.mean(self.resource_efficiency_samples)
    
    # Memory efficiency (throughput per MB growth)
    memory_growth = max(self.memory_samples) - min(self.memory_samples) if self.memory_samples else 0
    memory_efficiency = self.get_average_throughput() / max(memory_growth, 1.0)
    
    # CPU efficiency (throughput per CPU usage)
    avg_cpu = statistics.mean(self.cpu_samples) if self.cpu_samples else 1.0
    cpu_efficiency = self.get_average_throughput() / max(avg_cpu, 1.0)
    
    # Trend efficiency (improving vs degrading)
    trend_direction = self.get_trend_direction('recent_throughput')
    trend_score = 1.0 if trend_direction == MetricTrendDirection.IMPROVING else 0.5 if trend_direction == MetricTrendDirection.STABLE else 0.0
    
    return {
        'overall': overall_efficiency,
        'memory': memory_efficiency,
        'cpu': cpu_efficiency,
        'trend': trend_score,
        'grade': self.current_grade.value
    }

def detect_performance_regressions(self, baseline_metrics: Optional['CMOLoadMetrics'] = None) -> Dict[str, Any]:
    """Detect performance regressions compared to baseline."""
    if not baseline_metrics:
        return {'regressions_detected': False, 'message': 'No baseline provided'}
    
    regressions = []
    improvements = []
    
    # Response time regression
    current_p95 = self.get_advanced_percentiles().get('p95', 0)
    baseline_p95 = baseline_metrics.get_advanced_percentiles().get('p95', 0)
    
    if baseline_p95 > 0:
        p95_change = (current_p95 - baseline_p95) / baseline_p95
        if p95_change > 0.15:  # >15% regression
            regressions.append({
                'metric': 'p95_response_time',
                'change_percent': p95_change * 100,
                'current': current_p95,
                'baseline': baseline_p95,
                'severity': 'high' if p95_change > 0.3 else 'medium'
            })
    
    # Success rate regression
    current_success = self.get_success_rate()
    baseline_success = baseline_metrics.get_success_rate()
    success_change = current_success - baseline_success
    
    if success_change < -0.05:  # >5% regression
        regressions.append({
            'metric': 'success_rate',
            'change_percent': success_change * 100,
            'current': current_success,
            'baseline': baseline_success,
            'severity': 'critical' if success_change < -0.1 else 'high'
        })
    
    # LightRAG-specific regressions
    current_lightrag_success = self.lightrag_metrics.get_success_rate()
    baseline_lightrag_success = baseline_metrics.lightrag_metrics.get_success_rate()
    
    if baseline_lightrag_success > 0:
        lightrag_change = current_lightrag_success - baseline_lightrag_success
        if lightrag_change < -0.03:  # >3% regression for critical component
            regressions.append({
                'metric': 'lightrag_success_rate',
                'change_percent': lightrag_change * 100,
                'current': current_lightrag_success,
                'baseline': baseline_lightrag_success,
                'severity': 'critical'
            })
    
    # Cache effectiveness regression
    current_cache_effectiveness = self.cache_metrics.get_cache_effectiveness_score()
    baseline_cache_effectiveness = baseline_metrics.cache_metrics.get_cache_effectiveness_score()
    
    if baseline_cache_effectiveness > 0:
        cache_change = (current_cache_effectiveness - baseline_cache_effectiveness) / baseline_cache_effectiveness
        if cache_change < -0.1:  # >10% regression
            regressions.append({
                'metric': 'cache_effectiveness',
                'change_percent': cache_change * 100,
                'current': current_cache_effectiveness,
                'baseline': baseline_cache_effectiveness,
                'severity': 'medium'
            })
    
    return {
        'regressions_detected': len(regressions) > 0,
        'regression_count': len(regressions),
        'regressions': regressions,
        'improvements': improvements,
        'overall_assessment': self._assess_regression_severity(regressions)
    }

def _assess_regression_severity(self, regressions: List[Dict[str, Any]]) -> str:
    """Assess overall severity of detected regressions."""
    if not regressions:
        return "No regressions detected"
    
    critical_count = sum(1 for r in regressions if r.get('severity') == 'critical')
    high_count = sum(1 for r in regressions if r.get('severity') == 'high')
    
    if critical_count > 0:
        return f"CRITICAL: {critical_count} critical regressions detected"
    elif high_count >= 2:
        return f"HIGH: {high_count} high-severity regressions detected"
    elif len(regressions) >= 3:
        return f"MODERATE: {len(regressions)} regressions detected"
    else:
        return "LOW: Minor regressions detected"

def generate_comprehensive_analysis(self) -> Dict[str, Any]:
    """Generate comprehensive performance analysis report."""
    return {
        'executive_summary': {
            'current_grade': self.current_grade.value,
            'overall_success_rate': self.get_success_rate(),
            'performance_trend': self.get_trend_direction('success_rate').value,
            'system_health': 'healthy' if self.current_grade in [PerformanceGrade.A_PLUS, PerformanceGrade.A, PerformanceGrade.B] else 'needs_attention'
        },
        
        'detailed_metrics': {
            'response_times': self.get_advanced_percentiles(),
            'throughput_analysis': {
                'current': self.get_average_throughput(),
                'trend': self.get_trend_direction('recent_throughput').value,
                'efficiency_scores': self.get_resource_efficiency_score()
            },
            'resource_utilization': {
                'memory_growth_mb': max(self.memory_samples) - min(self.memory_samples) if self.memory_samples else 0,
                'memory_efficiency': self.get_resource_efficiency_score()['memory'],
                'cpu_efficiency': self.get_resource_efficiency_score()['cpu']
            }
        },
        
        'cmo_specific_analysis': {
            'lightrag_performance': {
                'success_rate': self.lightrag_metrics.get_success_rate(),
                'mode_effectiveness': self.lightrag_metrics.mode_success_rates,
                'cost_efficiency': self.lightrag_metrics.get_average_cost(),
                'token_usage': self.lightrag_metrics.get_token_efficiency()
            },
            'cache_analysis': {
                'tier_performance': self.cache_metrics.get_tier_hit_rates(),
                'effectiveness_score': self.cache_metrics.get_cache_effectiveness_score(),
                'response_times': self.cache_metrics.get_average_response_times()
            },
            'circuit_breaker_analysis': {
                'availability': self.circuit_breaker_metrics.get_availability_percentage(),
                'recovery_effectiveness': self.circuit_breaker_metrics.get_recovery_success_rate(),
                'state_health': self.circuit_breaker_metrics.is_functioning_properly()
            },
            'fallback_system_analysis': {
                'chain_success_rates': self.fallback_metrics.get_fallback_success_rates(),
                'cost_efficiency': self.fallback_metrics.get_cost_efficiency_score(),
                'overall_effectiveness': self.fallback_metrics.get_overall_success_rate()
            }
        },
        
        'component_health': self.get_component_health_assessment(),
        
        'recommendations': self._generate_automated_recommendations(),
        
        'trend_analysis': {
            metric: {
                'direction': self.get_trend_direction(metric).value,
                'recent_values': self.trend_analysis[metric][-10:] if metric in self.trend_analysis else []
            }
            for metric in ['success_rate', 'recent_avg_response_time', 'cache_hit_rate', 'recent_throughput']
        }
    }

def _generate_automated_recommendations(self) -> List[str]:
    """Generate automated performance optimization recommendations."""
    recommendations = []
    
    # LightRAG recommendations
    if self.lightrag_metrics.get_success_rate() < 0.95:
        recommendations.append(
            f"LightRAG success rate ({self.lightrag_metrics.get_success_rate():.1%}) below target. "
            "Consider optimizing query processing or implementing additional error handling."
        )
    
    mode_dist = self.lightrag_metrics.get_mode_distribution()
    if mode_dist.get('hybrid', 0) < 0.7:
        recommendations.append(
            f"LightRAG hybrid mode usage ({mode_dist.get('hybrid', 0):.1%}) is low. "
            "Hybrid mode typically provides best performance - investigate routing logic."
        )
    
    # Cache recommendations
    cache_hit_rates = self.cache_metrics.get_tier_hit_rates()
    if cache_hit_rates.get('l1', 0) < 0.8:
        recommendations.append(
            f"L1 cache hit rate ({cache_hit_rates.get('l1', 0):.1%}) below 80% target. "
            "Consider increasing L1 cache size or optimizing cache key strategies."
        )
    
    if cache_hit_rates.get('l2', 0) < 0.7:
        recommendations.append(
            f"L2 cache hit rate ({cache_hit_rates.get('l2', 0):.1%}) below 70% target. "
            "Review L2 cache TTL settings and eviction policies."
        )
    
    # Circuit breaker recommendations
    if self.circuit_breaker_metrics.get_availability_percentage() < 95:
        recommendations.append(
            f"Circuit breaker availability ({self.circuit_breaker_metrics.get_availability_percentage():.1f}%) below 95% target. "
            "Review failure thresholds and recovery timing parameters."
        )
    
    # Resource efficiency recommendations
    efficiency = self.get_resource_efficiency_score()
    if efficiency['overall'] < 1.0:
        recommendations.append(
            f"Resource efficiency score ({efficiency['overall']:.2f}) indicates suboptimal performance. "
            "Consider optimizing memory usage and CPU utilization patterns."
        )
    
    # Performance grade recommendations
    if self.current_grade in [PerformanceGrade.D, PerformanceGrade.F]:
        recommendations.append(
            f"Current performance grade ({self.current_grade.value}) requires immediate attention. "
            "Implement comprehensive performance optimization across all components."
        )
    
    # Trend-based recommendations
    success_trend = self.get_trend_direction('success_rate')
    if success_trend == MetricTrendDirection.DEGRADING:
        recommendations.append(
            "Success rate is trending downward. Monitor for cascading failures and "
            "consider implementing proactive scaling or circuit breaker tuning."
        )
    
    response_trend = self.get_trend_direction('recent_avg_response_time')
    if response_trend == MetricTrendDirection.DEGRADING:
        recommendations.append(
            "Response times are increasing. Investigate query complexity, cache effectiveness, "
            "and consider horizontal scaling or performance optimization."
        )
    
    return recommendations[:10]  # Limit to top 10 recommendations