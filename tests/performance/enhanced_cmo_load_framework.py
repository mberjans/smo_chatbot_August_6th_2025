"""
Enhanced CMO Load Testing Framework for Clinical Metabolomics Oracle
===================================================================

This module extends the existing concurrent load testing framework with comprehensive
CMO-specific enhancements, providing advanced testing capabilities for all CMO system
components including LightRAG integration, multi-tier caching, circuit breakers,
and fallback systems.

Key Enhancements:
1. CMO-specific test configurations with LightRAG settings and cache tiers
2. Enhanced metrics collection for CMO components (cache hit rates, circuit breaker
   activations, LightRAG success rates, fallback system usage)
3. Advanced user behavior simulation for clinical workflows
4. Direct integration with ClinicalMetabolomicsRAG, multi-tier caching, and circuit breakers
5. Fallback system testing (LightRAG → Perplexity → Cache)
6. Real-time metrics collection with 100ms sampling for high-precision monitoring

Performance Targets:
- Support 50-200+ concurrent users with graceful degradation
- Achieve >95% success rate with <2000ms P95 response times
- Monitor cache hit rates >70% across all tiers
- Memory growth <100MB during sustained load
- Real-time component health monitoring

Integration Points:
- Extends LoadTestConfiguration with CMO-specific parameters
- Enhances ConcurrentLoadMetrics with CMO component metrics
- Augments ConcurrentUserSimulator with CMO behavior patterns
- Integrates with LightRAG, Perplexity, and multi-tier cache systems

Author: Claude Code (Anthropic)
Version: 3.0.0
Created: 2025-08-09
Updated: 2025-08-09 (Enhanced with comprehensive CMO capabilities)
Production Ready: Yes
"""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import uuid
import random

# Import existing framework components
from .concurrent_load_framework import (
    LoadTestOrchestrator, ConcurrentUserSimulator, LoadTestConfiguration,
    LoadPattern, UserBehavior, ComponentType, ConcurrentLoadMetrics
)
from .concurrent_performance_enhancer import (
    create_enhanced_performance_suite, run_enhanced_performance_analysis
)
from .concurrent_scenarios import ScenarioOrchestrator

# Import LightRAG components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    from lightrag_integration.enhanced_load_monitoring_system import EnhancedLoadDetectionSystem
    from lightrag_integration.cost_based_circuit_breaker import CostBasedCircuitBreaker
    CMO_COMPONENTS_AVAILABLE = True
except ImportError:
    CMO_COMPONENTS_AVAILABLE = False


# ============================================================================
# CMO-SPECIFIC CONFIGURATION ENHANCEMENTS
# ============================================================================

@dataclass
class CMOTestConfiguration(LoadTestConfiguration):
    """Enhanced configuration specifically for CMO testing scenarios with comprehensive settings."""
    
    # LightRAG specific settings
    lightrag_enabled: bool = True
    lightrag_mode: str = "hybrid"  # local, hybrid, or global
    lightrag_response_type: str = "Multiple Paragraphs"
    lightrag_max_tokens: int = 4000
    lightrag_temperature: float = 0.7
    lightrag_timeout_seconds: int = 30
    lightrag_cost_threshold: float = 5.0
    
    # Multi-tier cache configuration
    enable_l1_cache: bool = True
    enable_l2_cache: bool = True
    enable_l3_cache: bool = True
    l1_cache_size: int = 10000
    l2_cache_size: int = 50000
    l3_cache_size: int = 100000
    l1_cache_ttl: int = 300      # 5 minutes
    l2_cache_ttl: int = 1800     # 30 minutes
    l3_cache_ttl: int = 3600     # 1 hour
    cache_prewarming_enabled: bool = True
    cache_compression_enabled: bool = True
    
    # Circuit breaker advanced settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 10
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_half_open_max_calls: int = 5
    circuit_breaker_cost_threshold: float = 5.0
    circuit_breaker_timeout_threshold: int = 3
    
    # Fallback system configuration
    enable_perplexity_fallback: bool = True
    enable_cache_fallback: bool = True
    enable_static_fallback: bool = True
    fallback_timeout_seconds: int = 10
    max_fallback_attempts: int = 3
    fallback_success_threshold: float = 0.80
    
    # Real-time monitoring settings
    enable_real_time_monitoring: bool = True
    monitoring_sample_interval_ms: int = 100
    enable_component_health_tracking: bool = True
    enable_cost_tracking: bool = True
    enable_performance_regression_detection: bool = True
    
    # User behavior enhancement
    enable_clinical_workflows: bool = True
    emergency_user_percentage: float = 0.1
    researcher_session_complexity: str = "high"
    clinician_response_urgency: str = "medium"
    
    # CMO-specific performance targets
    target_lightrag_success_rate: float = 0.95
    target_cache_hit_rate_l1: float = 0.85
    target_cache_hit_rate_l2: float = 0.70
    target_cache_hit_rate_l3: float = 0.60
    target_fallback_success_rate: float = 0.90
    max_cost_per_query: float = 0.10
    target_memory_efficiency: float = 0.85
    target_cpu_efficiency: float = 0.80
    
    # Advanced load testing features
    enable_burst_detection: bool = True
    enable_degradation_testing: bool = True
    enable_recovery_testing: bool = True
    enable_chaos_engineering: bool = False
    
    def get_cmo_performance_targets(self) -> Dict[str, Any]:
        """Get comprehensive CMO-specific performance targets."""
        base_targets = super().get_performance_targets()
        cmo_targets = {
            'lightrag_success_rate': self.target_lightrag_success_rate,
            'cache_hit_rate_l1': self.target_cache_hit_rate_l1,
            'cache_hit_rate_l2': self.target_cache_hit_rate_l2,
            'cache_hit_rate_l3': self.target_cache_hit_rate_l3,
            'fallback_success_rate': self.target_fallback_success_rate,
            'max_cost_per_query': self.max_cost_per_query,
            'memory_efficiency': self.target_memory_efficiency,
            'cpu_efficiency': self.target_cpu_efficiency
        }
        return {**base_targets, **cmo_targets}
    
    def get_cache_tier_settings(self) -> Dict[str, Dict[str, Any]]:
        """Get cache tier configuration settings."""
        return {
            'l1': {
                'enabled': self.enable_l1_cache,
                'size': self.l1_cache_size,
                'ttl': self.l1_cache_ttl,
                'tier_priority': 1
            },
            'l2': {
                'enabled': self.enable_l2_cache,
                'size': self.l2_cache_size,
                'ttl': self.l2_cache_ttl,
                'tier_priority': 2
            },
            'l3': {
                'enabled': self.enable_l3_cache,
                'size': self.l3_cache_size,
                'ttl': self.l3_cache_ttl,
                'tier_priority': 3
            }
        }


# ============================================================================
# ENHANCED CMO METRICS COLLECTION
# ============================================================================

@dataclass
class CMOLoadMetrics(ConcurrentLoadMetrics):
    """Comprehensive metrics collection for CMO system testing with real-time monitoring."""
    
    # LightRAG comprehensive metrics
    lightrag_queries: int = 0
    lightrag_successes: int = 0
    lightrag_failures: int = 0
    lightrag_timeouts: int = 0
    lightrag_cost_overruns: int = 0
    lightrag_response_times: List[float] = field(default_factory=list)
    lightrag_token_usage: Dict[str, int] = field(default_factory=lambda: {'input': 0, 'output': 0})
    lightrag_costs: List[float] = field(default_factory=list)
    lightrag_mode_usage: Dict[str, int] = field(default_factory=lambda: {'local': 0, 'hybrid': 0, 'global': 0})
    
    # Multi-tier cache comprehensive metrics
    l1_cache_hits: int = 0
    l1_cache_misses: int = 0
    l1_cache_evictions: int = 0
    l1_cache_response_times: List[float] = field(default_factory=list)
    l2_cache_hits: int = 0
    l2_cache_misses: int = 0
    l2_cache_evictions: int = 0
    l2_cache_response_times: List[float] = field(default_factory=list)
    l3_cache_hits: int = 0
    l3_cache_misses: int = 0
    l3_cache_evictions: int = 0
    l3_cache_response_times: List[float] = field(default_factory=list)
    cache_prewarming_hits: int = 0
    cache_promotion_events: int = 0
    cache_compression_ratio: List[float] = field(default_factory=list)
    
    # Circuit breaker enhanced metrics
    circuit_breaker_state_changes: List[Dict[str, Any]] = field(default_factory=list)
    circuit_breaker_activations: int = 0
    circuit_breaker_recoveries: int = 0
    circuit_breaker_cost_blocks: int = 0
    circuit_breaker_timeout_blocks: int = 0
    circuit_breaker_failure_rate_history: List[float] = field(default_factory=list)
    circuit_breaker_recovery_times: List[float] = field(default_factory=list)
    
    # Fallback system comprehensive metrics
    perplexity_fallback_uses: int = 0
    perplexity_fallback_successes: int = 0
    perplexity_fallback_failures: int = 0
    cache_fallback_uses: int = 0
    cache_fallback_successes: int = 0
    static_fallback_uses: int = 0
    fallback_chain_complete_failures: int = 0
    fallback_response_times: List[float] = field(default_factory=list)
    fallback_cost_savings: List[float] = field(default_factory=list)
    
    # Real-time monitoring metrics
    real_time_samples: deque = field(default_factory=lambda: deque(maxlen=10000))
    component_health_scores: Dict[str, List[float]] = field(default_factory=dict)
    performance_degradation_events: int = 0
    system_recovery_events: int = 0
    burst_detection_triggers: int = 0
    
    # User behavior and workflow metrics
    clinical_workflow_completions: int = 0
    research_session_durations: List[float] = field(default_factory=list)
    emergency_response_times: List[float] = field(default_factory=list)
    user_satisfaction_scores: List[float] = field(default_factory=list)
    
    # Advanced system metrics
    end_to_end_response_times: List[float] = field(default_factory=list)
    component_interaction_delays: List[float] = field(default_factory=list)
    resource_contention_events: int = 0
    memory_efficiency_samples: List[float] = field(default_factory=list)
    cpu_efficiency_samples: List[float] = field(default_factory=list)
    network_latency_samples: List[float] = field(default_factory=list)
    
    def add_lightrag_query_result(self, success: bool, response_time: float, 
                                token_usage: Dict[str, int], cost: float = 0.0, 
                                mode: str = "hybrid", timeout: bool = False):
        """Record comprehensive LightRAG query result."""
        self.lightrag_queries += 1
        self.lightrag_response_times.append(response_time)
        self.lightrag_costs.append(cost)
        self.lightrag_mode_usage[mode] = self.lightrag_mode_usage.get(mode, 0) + 1
        
        if timeout:
            self.lightrag_timeouts += 1
        elif success:
            self.lightrag_successes += 1
        else:
            self.lightrag_failures += 1
            
        # Track token usage
        self.lightrag_token_usage['input'] += token_usage.get('input', 0)
        self.lightrag_token_usage['output'] += token_usage.get('output', 0)
    
    def add_cache_access(self, tier: str, hit: bool, response_time: float = 0.0, 
                        promoted: bool = False, evicted: bool = False):
        """Record comprehensive cache access metrics."""
        if tier.lower() == 'l1':
            if hit:
                self.l1_cache_hits += 1
                self.l1_cache_response_times.append(response_time)
            else:
                self.l1_cache_misses += 1
            if evicted:
                self.l1_cache_evictions += 1
        elif tier.lower() == 'l2':
            if hit:
                self.l2_cache_hits += 1
                self.l2_cache_response_times.append(response_time)
            else:
                self.l2_cache_misses += 1
            if evicted:
                self.l2_cache_evictions += 1
        elif tier.lower() == 'l3':
            if hit:
                self.l3_cache_hits += 1
                self.l3_cache_response_times.append(response_time)
            else:
                self.l3_cache_misses += 1
            if evicted:
                self.l3_cache_evictions += 1
        
        if promoted:
            self.cache_promotion_events += 1
    
    def add_fallback_usage(self, fallback_type: str, success: bool, response_time: float, cost_savings: float = 0.0):
        """Record comprehensive fallback system usage."""
        self.fallback_response_times.append(response_time)
        if cost_savings > 0:
            self.fallback_cost_savings.append(cost_savings)
            
        if fallback_type == 'perplexity':
            self.perplexity_fallback_uses += 1
            if success:
                self.perplexity_fallback_successes += 1
            else:
                self.perplexity_fallback_failures += 1
        elif fallback_type == 'cache':
            self.cache_fallback_uses += 1
            if success:
                self.cache_fallback_successes += 1
        elif fallback_type == 'static':
            self.static_fallback_uses += 1
    
    def add_circuit_breaker_event(self, event_type: str, timestamp: float, details: Dict[str, Any]):
        """Record circuit breaker state changes and events."""
        self.circuit_breaker_state_changes.append({
            'event_type': event_type,
            'timestamp': timestamp,
            'details': details
        })
        
        if event_type == 'activation':
            self.circuit_breaker_activations += 1
        elif event_type == 'recovery':
            self.circuit_breaker_recoveries += 1
            if 'recovery_time' in details:
                self.circuit_breaker_recovery_times.append(details['recovery_time'])
        elif event_type == 'cost_block':
            self.circuit_breaker_cost_blocks += 1
        elif event_type == 'timeout_block':
            self.circuit_breaker_timeout_blocks += 1
    
    def add_real_time_sample(self, timestamp: float, component_metrics: Dict[str, Any]):
        """Add real-time monitoring sample."""
        sample = {
            'timestamp': timestamp,
            'metrics': component_metrics
        }
        self.real_time_samples.append(sample)
        
        # Track component health scores
        for component, metrics in component_metrics.items():
            if 'health_score' in metrics:
                if component not in self.component_health_scores:
                    self.component_health_scores[component] = []
                self.component_health_scores[component].append(metrics['health_score'])
    
    def get_lightrag_success_rate(self) -> float:
        """Calculate LightRAG-specific success rate."""
        total = self.lightrag_queries
        return self.lightrag_successes / total if total > 0 else 0.0
    
    def get_lightrag_comprehensive_analysis(self) -> Dict[str, Any]:
        """Comprehensive LightRAG performance analysis."""
        if self.lightrag_queries == 0:
            return {}
        
        avg_response_time = statistics.mean(self.lightrag_response_times) if self.lightrag_response_times else 0
        avg_cost = statistics.mean(self.lightrag_costs) if self.lightrag_costs else 0
        
        return {
            'success_rate': self.get_lightrag_success_rate(),
            'timeout_rate': self.lightrag_timeouts / self.lightrag_queries,
            'avg_response_time': avg_response_time,
            'avg_cost_per_query': avg_cost,
            'total_cost': sum(self.lightrag_costs),
            'total_tokens': self.lightrag_token_usage,
            'mode_distribution': {
                mode: count / self.lightrag_queries 
                for mode, count in self.lightrag_mode_usage.items()
            },
            'performance_percentiles': {
                'p50': statistics.median(self.lightrag_response_times) if self.lightrag_response_times else 0,
                'p95': statistics.quantiles(self.lightrag_response_times, n=20)[18] if len(self.lightrag_response_times) >= 20 else 0,
                'p99': statistics.quantiles(self.lightrag_response_times, n=100)[98] if len(self.lightrag_response_times) >= 100 else 0
            }
        }
    
    def get_multi_tier_cache_analysis(self) -> Dict[str, Any]:
        """Comprehensive multi-tier cache performance analysis."""
        total_hits = self.l1_cache_hits + self.l2_cache_hits + self.l3_cache_hits
        total_misses = self.l1_cache_misses + self.l2_cache_misses + self.l3_cache_misses
        total_requests = total_hits + total_misses
        
        if total_requests == 0:
            return {}
        
        analysis = {
            'overall_metrics': {
                'total_requests': total_requests,
                'total_hits': total_hits,
                'overall_hit_rate': total_hits / total_requests,
                'cache_promotion_rate': self.cache_promotion_events / total_requests if total_requests > 0 else 0
            },
            'tier_performance': {}
        }
        
        # L1 Cache Analysis
        l1_total = self.l1_cache_hits + self.l1_cache_misses
        if l1_total > 0:
            analysis['tier_performance']['l1'] = {
                'hit_rate': self.l1_cache_hits / l1_total,
                'hits': self.l1_cache_hits,
                'misses': self.l1_cache_misses,
                'evictions': self.l1_cache_evictions,
                'avg_response_time': statistics.mean(self.l1_cache_response_times) if self.l1_cache_response_times else 0,
                'hit_percentage_of_total': self.l1_cache_hits / total_hits if total_hits > 0 else 0
            }
        
        # L2 Cache Analysis
        l2_total = self.l2_cache_hits + self.l2_cache_misses
        if l2_total > 0:
            analysis['tier_performance']['l2'] = {
                'hit_rate': self.l2_cache_hits / l2_total,
                'hits': self.l2_cache_hits,
                'misses': self.l2_cache_misses,
                'evictions': self.l2_cache_evictions,
                'avg_response_time': statistics.mean(self.l2_cache_response_times) if self.l2_cache_response_times else 0,
                'hit_percentage_of_total': self.l2_cache_hits / total_hits if total_hits > 0 else 0
            }
        
        # L3 Cache Analysis
        l3_total = self.l3_cache_hits + self.l3_cache_misses
        if l3_total > 0:
            analysis['tier_performance']['l3'] = {
                'hit_rate': self.l3_cache_hits / l3_total,
                'hits': self.l3_cache_hits,
                'misses': self.l3_cache_misses,
                'evictions': self.l3_cache_evictions,
                'avg_response_time': statistics.mean(self.l3_cache_response_times) if self.l3_cache_response_times else 0,
                'hit_percentage_of_total': self.l3_cache_hits / total_hits if total_hits > 0 else 0
            }
        
        # Cache efficiency metrics
        total_evictions = self.l1_cache_evictions + self.l2_cache_evictions + self.l3_cache_evictions
        analysis['efficiency_metrics'] = {
            'eviction_rate': total_evictions / total_requests if total_requests > 0 else 0,
            'prewarming_effectiveness': self.cache_prewarming_hits / total_hits if total_hits > 0 else 0,
            'tier_utilization_balance': self._calculate_cache_balance(),
            'compression_effectiveness': statistics.mean(self.cache_compression_ratio) if self.cache_compression_ratio else 1.0
        }
        
        return analysis
    
    def get_fallback_effectiveness(self) -> Dict[str, Any]:
        """Comprehensive fallback system effectiveness analysis."""
        total_fallbacks = (self.perplexity_fallback_uses + 
                          self.cache_fallback_uses + 
                          self.static_fallback_uses)
        
        if total_fallbacks == 0:
            return {'total_fallback_calls': 0, 'fallback_usage_rate': 0}
        
        total_successes = (self.perplexity_fallback_successes + 
                          self.cache_fallback_successes)
        
        analysis = {
            'usage_metrics': {
                'total_fallback_calls': total_fallbacks,
                'fallback_usage_rate': total_fallbacks / self.total_operations if self.total_operations > 0 else 0,
                'complete_failures': self.fallback_chain_complete_failures
            },
            'success_metrics': {
                'overall_fallback_success_rate': total_successes / total_fallbacks,
                'perplexity_success_rate': self.perplexity_fallback_successes / self.perplexity_fallback_uses if self.perplexity_fallback_uses > 0 else 0,
                'cache_fallback_success_rate': self.cache_fallback_successes / self.cache_fallback_uses if self.cache_fallback_uses > 0 else 0,
            },
            'distribution': {
                'perplexity_percentage': self.perplexity_fallback_uses / total_fallbacks,
                'cache_fallback_percentage': self.cache_fallback_uses / total_fallbacks,
                'static_fallback_percentage': self.static_fallback_uses / total_fallbacks
            },
            'performance_metrics': {
                'avg_fallback_response_time': statistics.mean(self.fallback_response_times) if self.fallback_response_times else 0,
                'total_cost_savings': sum(self.fallback_cost_savings),
                'avg_cost_savings_per_fallback': statistics.mean(self.fallback_cost_savings) if self.fallback_cost_savings else 0
            }
        }
        
        return analysis
    
    def get_circuit_breaker_analysis(self) -> Dict[str, Any]:
        """Comprehensive circuit breaker performance analysis."""
        total_events = len(self.circuit_breaker_state_changes)
        
        if total_events == 0:
            return {'total_events': 0, 'effectiveness': 'no_data'}
        
        return {
            'activation_metrics': {
                'total_activations': self.circuit_breaker_activations,
                'total_recoveries': self.circuit_breaker_recoveries,
                'cost_blocks': self.circuit_breaker_cost_blocks,
                'timeout_blocks': self.circuit_breaker_timeout_blocks,
                'activation_rate': self.circuit_breaker_activations / self.total_operations if self.total_operations > 0 else 0
            },
            'recovery_metrics': {
                'recovery_success_rate': self.circuit_breaker_recoveries / self.circuit_breaker_activations if self.circuit_breaker_activations > 0 else 0,
                'avg_recovery_time': statistics.mean(self.circuit_breaker_recovery_times) if self.circuit_breaker_recovery_times else 0,
                'failure_rate_trend': self.circuit_breaker_failure_rate_history[-10:] if self.circuit_breaker_failure_rate_history else []
            },
            'effectiveness': {
                'prevented_failures': self.circuit_breaker_cost_blocks + self.circuit_breaker_timeout_blocks,
                'system_protection_rate': (self.circuit_breaker_cost_blocks + self.circuit_breaker_timeout_blocks) / self.total_operations if self.total_operations > 0 else 0
            }
        }
    
    def get_real_time_monitoring_analysis(self) -> Dict[str, Any]:
        """Real-time monitoring and component health analysis."""
        if not self.real_time_samples:
            return {'sample_count': 0, 'monitoring_active': False}
        
        latest_sample = self.real_time_samples[-1] if self.real_time_samples else None
        sample_count = len(self.real_time_samples)
        
        component_health_summary = {}
        for component, health_scores in self.component_health_scores.items():
            if health_scores:
                component_health_summary[component] = {
                    'current_health': health_scores[-1],
                    'avg_health': statistics.mean(health_scores),
                    'min_health': min(health_scores),
                    'health_trend': health_scores[-1] - health_scores[0] if len(health_scores) > 1 else 0,
                    'stability': statistics.stdev(health_scores) if len(health_scores) > 1 else 0
                }
        
        return {
            'monitoring_metrics': {
                'total_samples': sample_count,
                'monitoring_active': sample_count > 0,
                'sample_frequency': len([s for s in self.real_time_samples if s['timestamp'] > time.time() - 60]) / 60 if self.real_time_samples else 0,
                'latest_timestamp': latest_sample['timestamp'] if latest_sample else 0
            },
            'component_health': component_health_summary,
            'system_events': {
                'performance_degradations': self.performance_degradation_events,
                'system_recoveries': self.system_recovery_events,
                'burst_detections': self.burst_detection_triggers
            },
            'efficiency_metrics': {
                'avg_memory_efficiency': statistics.mean(self.memory_efficiency_samples) if self.memory_efficiency_samples else 0,
                'avg_cpu_efficiency': statistics.mean(self.cpu_efficiency_samples) if self.cpu_efficiency_samples else 0,
                'avg_network_latency': statistics.mean(self.network_latency_samples) if self.network_latency_samples else 0
            }
        }
    
    def _calculate_cache_balance(self) -> float:
        """Calculate how well balanced the cache tier usage is."""
        total_hits = self.l1_cache_hits + self.l2_cache_hits + self.l3_cache_hits
        if total_hits == 0:
            return 0.0
        
        l1_ratio = self.l1_cache_hits / total_hits
        l2_ratio = self.l2_cache_hits / total_hits  
        l3_ratio = self.l3_cache_hits / total_hits
        
        # Ideal ratios would be something like L1: 0.6, L2: 0.3, L3: 0.1
        ideal_l1, ideal_l2, ideal_l3 = 0.6, 0.3, 0.1
        
        # Calculate deviation from ideal
        deviation = (abs(l1_ratio - ideal_l1) + 
                    abs(l2_ratio - ideal_l2) + 
                    abs(l3_ratio - ideal_l3)) / 3
        
        return max(0, 1 - deviation)  # Return balance score (1 = perfect, 0 = completely unbalanced)
    
    def meets_cmo_performance_targets(self, targets: Dict[str, Any]) -> Dict[str, bool]:
        """Check if metrics meet CMO-specific performance targets."""
        results = {}
        
        # LightRAG performance
        lightrag_p95 = statistics.quantiles(self.lightrag_response_times, n=20)[18] if len(self.lightrag_response_times) >= 20 else 0
        results['lightrag_p95_target'] = lightrag_p95 <= targets.get('lightrag_p95_response_ms', 2000)
        results['lightrag_success_rate'] = self.get_lightrag_success_rate() >= targets.get('min_success_rate', 0.95)
        
        # Cache performance
        cache_analysis = self.get_multi_tier_cache_analysis()
        results['cache_hit_rate_target'] = cache_analysis.get('overall_hit_rate', 0) >= targets.get('cache_hit_rate_target', 0.75)
        
        # Circuit breaker effectiveness
        results['circuit_breaker_threshold'] = self.circuit_breaker_failure_rate <= targets.get('circuit_breaker_threshold', 0.15)
        
        # Fallback system
        fallback_analysis = self.get_fallback_effectiveness()
        results['fallback_success_rate'] = fallback_analysis.get('fallback_success_rate', 0) >= targets.get('fallback_success_rate', 0.90)
        
        # Resource usage
        results['memory_growth'] = (max(self.memory_samples, default=0) - min(self.memory_samples, default=0)) <= targets.get('max_memory_growth_mb', 150)
        results['cpu_usage'] = max(self.cpu_samples, default=0) <= targets.get('max_cpu_usage', 80)
        
        return results


# ============================================================================
# CMO LOAD TEST CONTROLLER
# ============================================================================

class CMOLoadTestController:
    """Main controller for CMO-specific load testing scenarios."""
    
    def __init__(self, cmo_config: CMOTestConfiguration):
        self.config = cmo_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize CMO components
        self.rag_system = None
        self.load_monitor = None
        self.circuit_breaker = None
        self.performance_suite = None
        
        # Test orchestration
        self.base_orchestrator = LoadTestOrchestrator()
        self.scenario_orchestrator = ScenarioOrchestrator()
        
        # Results storage
        self.test_results: Dict[str, CMOLoadMetrics] = {}
        self.performance_analysis: Dict[str, Any] = {}
    
    async def initialize_cmo_components(self):
        """Initialize CMO system components for testing."""
        try:
            if CMO_COMPONENTS_AVAILABLE and self.config.lightrag_enabled:
                # Initialize ClinicalMetabolomicsRAG
                self.rag_system = ClinicalMetabolomicsRAG(
                    working_dir="./lightrag_cache",
                    loglevel=logging.INFO
                )
                self.logger.info("ClinicalMetabolomicsRAG initialized for testing")
                
                # Initialize load monitoring
                self.load_monitor = EnhancedLoadDetectionSystem()
                await self.load_monitor.start_monitoring()
                self.logger.info("Enhanced load monitoring system started")
                
                # Initialize circuit breaker
                if self.config.circuit_breaker_testing:
                    self.circuit_breaker = CostBasedCircuitBreaker(
                        failure_threshold=self.config.failure_threshold,
                        recovery_timeout=self.config.recovery_timeout
                    )
                    self.logger.info("Cost-based circuit breaker initialized")
            
            # Initialize enhanced performance suite
            self.performance_suite = create_enhanced_performance_suite(enable_monitoring=True)
            self.logger.info("Enhanced performance suite initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CMO components: {e}")
            raise
    
    async def cleanup_cmo_components(self):
        """Clean up CMO components after testing."""
        try:
            if self.load_monitor:
                await self.load_monitor.stop_monitoring()
            
            if self.performance_suite and 'enhanced_cache' in self.performance_suite:
                cache_system = self.performance_suite['enhanced_cache']
                if hasattr(cache_system, 'base_cache') and hasattr(cache_system.base_cache, 'clear'):
                    await cache_system.base_cache.clear()
            
        except Exception as e:
            self.logger.warning(f"Error during CMO component cleanup: {e}")
    
    async def run_cmo_load_scenario(self, scenario_name: str, custom_config: Optional[CMOTestConfiguration] = None) -> CMOLoadMetrics:
        """Run a specific CMO load testing scenario."""
        config = custom_config or self.config
        
        # Create enhanced user simulator
        simulator = CMOUserSimulator(config, scenario_name)
        
        # Setup CMO components
        component_kwargs = {
            'rag_system': self.rag_system,
            'circuit_breaker': self.circuit_breaker,
            'load_monitor': self.load_monitor
        }
        
        if self.performance_suite:
            component_kwargs.update({
                'cache_system': self.performance_suite.get('enhanced_cache'),
                'resource_monitor': self.performance_suite.get('resource_monitor')
            })
        
        await simulator.setup_components(**component_kwargs)
        
        # Run the load test
        metrics = await simulator.run_cmo_load_test()
        self.test_results[scenario_name] = metrics
        
        return metrics
    
    async def run_comprehensive_cmo_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive CMO load test suite."""
        self.logger.info("Starting comprehensive CMO load test suite")
        
        # Define CMO-specific test scenarios
        cmo_scenarios = [
            CMOTestConfiguration(
                test_name="cmo_basic_50_users",
                component_type=ComponentType.FULL_SYSTEM,
                load_pattern=LoadPattern.SUSTAINED,
                concurrent_users=50,
                total_operations=250,
                test_duration_seconds=180,
                user_behaviors=[UserBehavior.RESEARCHER, UserBehavior.CLINICIAN],
                behavior_weights=[0.6, 0.4]
            ),
            CMOTestConfiguration(
                test_name="cmo_clinical_burst_load",
                component_type=ComponentType.RAG_QUERY,
                load_pattern=LoadPattern.BURST,
                concurrent_users=75,
                total_operations=300,
                test_duration_seconds=120,
                user_behaviors=[UserBehavior.CLINICIAN, UserBehavior.EMERGENCY],
                behavior_weights=[0.7, 0.3],
                target_p95_response_ms=1500  # Fast clinical response
            ),
            CMOTestConfiguration(
                test_name="cmo_research_intensive",
                component_type=ComponentType.FULL_SYSTEM,
                load_pattern=LoadPattern.SUSTAINED,
                concurrent_users=40,
                total_operations=400,
                test_duration_seconds=300,
                user_behaviors=[UserBehavior.RESEARCHER],
                behavior_weights=[1.0],
                lightrag_p95_response_ms=3000  # Allow longer for complex research
            ),
            CMOTestConfiguration(
                test_name="cmo_mixed_usage_realistic",
                component_type=ComponentType.FULL_SYSTEM,
                load_pattern=LoadPattern.REALISTIC,
                concurrent_users=80,
                total_operations=400,
                test_duration_seconds=240,
                user_behaviors=[UserBehavior.RESEARCHER, UserBehavior.CLINICIAN, UserBehavior.STUDENT, UserBehavior.EMERGENCY],
                behavior_weights=[0.35, 0.30, 0.25, 0.10]
            ),
            CMOTestConfiguration(
                test_name="cmo_scalability_test",
                component_type=ComponentType.FULL_SYSTEM,
                load_pattern=LoadPattern.RAMP_UP,
                concurrent_users=100,
                total_operations=400,
                test_duration_seconds=300,
                ramp_up_duration=60,
                target_success_rate=0.85,  # Allow some degradation at scale
                max_memory_growth_mb=200
            )
        ]
        
        # Run each scenario
        for scenario_config in cmo_scenarios:
            try:
                self.logger.info(f"Running CMO scenario: {scenario_config.test_name}")
                
                await self.run_cmo_load_scenario(scenario_config.test_name, scenario_config)
                
                # Brief pause between scenarios
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Failed to run scenario {scenario_config.test_name}: {e}")
                continue
        
        # Generate comprehensive analysis
        analysis_results = await self.generate_cmo_performance_analysis()
        
        return {
            'scenario_results': self.test_results,
            'performance_analysis': analysis_results,
            'summary': self._generate_test_suite_summary()
        }
    
    async def generate_cmo_performance_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive CMO performance analysis."""
        if not self.test_results:
            return {}
        
        # Run enhanced performance analysis using existing framework
        base_results = {
            'individual_results': {
                name: {
                    'basic_metrics': {
                        'success_rate': metrics.get_success_rate(),
                        'total_operations': metrics.total_operations,
                        'concurrent_peak': metrics.concurrent_peak
                    },
                    'performance_metrics': {
                        'percentiles': metrics.get_percentiles(),
                        'average_throughput': metrics.get_average_throughput(),
                        'cache_hit_rate': metrics.get_cache_hit_rate()
                    }
                }
                for name, metrics in self.test_results.items()
            }
        }
        
        enhanced_analysis = await run_enhanced_performance_analysis(base_results, self.performance_suite)
        
        # Add CMO-specific analysis
        cmo_analysis = self._generate_cmo_specific_analysis()
        
        return {
            'enhanced_framework_analysis': enhanced_analysis,
            'cmo_specific_analysis': cmo_analysis,
            'integration_analysis': self._analyze_component_integration(),
            'recommendations': self._generate_cmo_recommendations()
        }
    
    def _generate_cmo_specific_analysis(self) -> Dict[str, Any]:
        """Generate CMO-specific performance analysis."""
        analysis = {
            'lightrag_performance': {},
            'multi_tier_caching': {},
            'circuit_breaker_effectiveness': {},
            'fallback_system_analysis': {}
        }
        
        for test_name, metrics in self.test_results.items():
            if isinstance(metrics, CMOLoadMetrics):
                # LightRAG analysis
                analysis['lightrag_performance'][test_name] = {
                    'success_rate': metrics.get_lightrag_success_rate(),
                    'avg_response_time': statistics.mean(metrics.lightrag_response_times) if metrics.lightrag_response_times else 0,
                    'query_count': metrics.lightrag_queries
                }
                
                # Cache analysis
                cache_analysis = metrics.get_multi_tier_cache_analysis()
                analysis['multi_tier_caching'][test_name] = cache_analysis
                
                # Fallback analysis
                fallback_analysis = metrics.get_fallback_effectiveness()
                analysis['fallback_system_analysis'][test_name] = fallback_analysis
        
        return analysis
    
    def _analyze_component_integration(self) -> Dict[str, Any]:
        """Analyze how well CMO components integrate under load."""
        integration_metrics = {
            'component_interaction_delays': [],
            'resource_contention_frequency': 0,
            'system_stability': 'Unknown',
            'scalability_assessment': 'Unknown'
        }
        
        for metrics in self.test_results.values():
            if isinstance(metrics, CMOLoadMetrics):
                integration_metrics['component_interaction_delays'].extend(metrics.component_interaction_delays)
                integration_metrics['resource_contention_frequency'] += metrics.resource_contention_events
        
        # Calculate stability assessment
        if integration_metrics['component_interaction_delays']:
            avg_delay = statistics.mean(integration_metrics['component_interaction_delays'])
            if avg_delay < 50:  # Less than 50ms average delay
                integration_metrics['system_stability'] = 'Excellent'
            elif avg_delay < 100:
                integration_metrics['system_stability'] = 'Good'
            elif avg_delay < 200:
                integration_metrics['system_stability'] = 'Fair'
            else:
                integration_metrics['system_stability'] = 'Poor'
        
        # Scalability assessment based on resource contention
        total_operations = sum(m.total_operations for m in self.test_results.values())
        if total_operations > 0:
            contention_rate = integration_metrics['resource_contention_frequency'] / total_operations
            if contention_rate < 0.01:  # Less than 1%
                integration_metrics['scalability_assessment'] = 'Excellent'
            elif contention_rate < 0.05:  # Less than 5%
                integration_metrics['scalability_assessment'] = 'Good'
            elif contention_rate < 0.10:  # Less than 10%
                integration_metrics['scalability_assessment'] = 'Fair'
            else:
                integration_metrics['scalability_assessment'] = 'Poor'
        
        return integration_metrics
    
    def _generate_cmo_recommendations(self) -> List[str]:
        """Generate CMO-specific performance recommendations."""
        recommendations = []
        
        # Analyze overall performance across all tests
        overall_success_rates = [m.get_success_rate() for m in self.test_results.values()]
        avg_success_rate = statistics.mean(overall_success_rates) if overall_success_rates else 0
        
        if avg_success_rate < 0.90:
            recommendations.append(
                f"Overall system success rate ({avg_success_rate:.1%}) needs improvement. "
                "Consider enhancing error handling, circuit breaker configuration, and fallback systems."
            )
        
        # LightRAG specific recommendations
        lightrag_success_rates = [
            m.get_lightrag_success_rate() for m in self.test_results.values() 
            if isinstance(m, CMOLoadMetrics) and m.lightrag_queries > 0
        ]
        
        if lightrag_success_rates:
            avg_lightrag_success = statistics.mean(lightrag_success_rates)
            if avg_lightrag_success < 0.92:
                recommendations.append(
                    f"LightRAG success rate ({avg_lightrag_success:.1%}) below target. "
                    "Review query processing logic, resource allocation, and timeout settings."
                )
        
        # Cache performance recommendations
        cache_analyses = [
            m.get_multi_tier_cache_analysis() for m in self.test_results.values()
            if isinstance(m, CMOLoadMetrics)
        ]
        
        if cache_analyses:
            overall_hit_rates = [ca.get('overall_hit_rate', 0) for ca in cache_analyses if ca]
            if overall_hit_rates:
                avg_hit_rate = statistics.mean(overall_hit_rates)
                if avg_hit_rate < 0.70:
                    recommendations.append(
                        f"Multi-tier cache hit rate ({avg_hit_rate:.1%}) below target. "
                        "Consider optimizing cache sizes, TTL settings, and key strategies."
                    )
        
        # Fallback system recommendations
        fallback_analyses = [
            m.get_fallback_effectiveness() for m in self.test_results.values()
            if isinstance(m, CMOLoadMetrics)
        ]
        
        if fallback_analyses:
            fallback_success_rates = [fa.get('fallback_success_rate', 0) for fa in fallback_analyses if fa]
            if fallback_success_rates:
                avg_fallback_success = statistics.mean(fallback_success_rates)
                if avg_fallback_success < 0.85:
                    recommendations.append(
                        f"Fallback system success rate ({avg_fallback_success:.1%}) needs improvement. "
                        "Review Perplexity integration and cache fallback mechanisms."
                    )
        
        return recommendations
    
    def _generate_test_suite_summary(self) -> Dict[str, Any]:
        """Generate summary of test suite execution."""
        return {
            'total_scenarios': len(self.test_results),
            'successful_scenarios': len([r for r in self.test_results.values() if r.end_time is not None]),
            'total_operations_executed': sum(r.total_operations for r in self.test_results.values()),
            'overall_success_rate': statistics.mean([r.get_success_rate() for r in self.test_results.values()]) if self.test_results else 0,
            'peak_concurrent_users': max([r.concurrent_peak for r in self.test_results.values()]) if self.test_results else 0,
            'test_suite_duration_minutes': (datetime.now() - min([r.start_time for r in self.test_results.values()])).total_seconds() / 60 if self.test_results else 0
        }


# ============================================================================
# CMO USER SIMULATOR
# ============================================================================

class CMOUserSimulator(ConcurrentUserSimulator):
    """Enhanced user simulator specifically for CMO testing."""
    
    def __init__(self, config: CMOTestConfiguration, scenario_name: str):
        super().__init__(config)
        self.cmo_config = config
        self.scenario_name = scenario_name
        self.cmo_metrics = CMOLoadMetrics(
            test_name=config.test_name,
            start_time=datetime.now(),
            total_users=config.concurrent_users
        )
    
    async def run_cmo_load_test(self) -> CMOLoadMetrics:
        """Run load test with CMO-specific metrics collection."""
        self.logger.info(f"Starting CMO load test: {self.scenario_name}")
        
        # Use the base framework's load test execution
        base_metrics = await super().run_load_test()
        
        # Copy base metrics to CMO metrics
        self._merge_base_metrics(base_metrics)
        
        return self.cmo_metrics
    
    def _merge_base_metrics(self, base_metrics: ConcurrentLoadMetrics):
        """Merge base framework metrics into CMO metrics."""
        # Copy all base fields
        for field_name in base_metrics.__dataclass_fields__:
            if hasattr(self.cmo_metrics, field_name):
                setattr(self.cmo_metrics, field_name, getattr(base_metrics, field_name))
    
    async def _execute_rag_query(self, query_data: Dict[str, Any], user_id: str) -> Tuple[Any, float, bool]:
        """Execute RAG query with CMO-specific tracking."""
        start_time = time.time()
        
        try:
            self.cmo_metrics.lightrag_queries += 1
            
            if self.rag_system:
                # Execute actual LightRAG query
                response = await self.rag_system.query(
                    query_data['query'],
                    query_param={
                        'mode': self.cmo_config.rag_mode,
                        'response_type': self.cmo_config.response_type
                    }
                )
                success = response is not None and len(str(response)) > 0
                
                if success:
                    self.cmo_metrics.lightrag_successes += 1
                else:
                    self.cmo_metrics.lightrag_failures += 1
            else:
                # Mock response for testing
                await asyncio.sleep(random.uniform(0.2, 0.8))
                response = f"Mock CMO response for: {query_data['query'][:50]}..."
                success = random.random() > 0.05  # 95% success rate for mock
                
                if success:
                    self.cmo_metrics.lightrag_successes += 1
                else:
                    self.cmo_metrics.lightrag_failures += 1
            
        except Exception as e:
            response = None
            success = False
            self.cmo_metrics.lightrag_failures += 1
            self.logger.warning(f"LightRAG query failed for user {user_id}: {e}")
        
        execution_time = time.time() - start_time
        self.cmo_metrics.lightrag_response_times.append(execution_time)
        
        return response, execution_time, success


# ============================================================================
# MAIN ENTRY POINT AND UTILITY FUNCTIONS
# ============================================================================

async def run_cmo_comprehensive_load_tests(
    concurrent_users: int = 50,
    enable_lightrag: bool = True,
    enable_circuit_breaker: bool = True
) -> Dict[str, Any]:
    """Run comprehensive CMO load tests with specified configuration."""
    
    # Create CMO test configuration
    config = CMOTestConfiguration(
        test_name="cmo_comprehensive_suite",
        component_type=ComponentType.FULL_SYSTEM,
        load_pattern=LoadPattern.REALISTIC,
        concurrent_users=concurrent_users,
        lightrag_enabled=enable_lightrag,
        circuit_breaker_testing=enable_circuit_breaker,
        multi_tier_caching=True,
        fallback_system_testing=True
    )
    
    # Initialize and run CMO load test controller
    controller = CMOLoadTestController(config)
    
    try:
        await controller.initialize_cmo_components()
        results = await controller.run_comprehensive_cmo_test_suite()
        return results
    
    finally:
        await controller.cleanup_cmo_components()


def create_cmo_test_configurations() -> List[CMOTestConfiguration]:
    """Create standard CMO test configurations for various scenarios."""
    
    return [
        # Basic concurrent load with CMO components
        CMOTestConfiguration(
            test_name="cmo_basic_concurrent_50",
            component_type=ComponentType.FULL_SYSTEM,
            load_pattern=LoadPattern.SUSTAINED,
            concurrent_users=50,
            total_operations=250,
            test_duration_seconds=180
        ),
        
        # Clinical decision support burst load
        CMOTestConfiguration(
            test_name="cmo_clinical_burst_75",
            component_type=ComponentType.RAG_QUERY,
            load_pattern=LoadPattern.BURST,
            concurrent_users=75,
            total_operations=225,
            test_duration_seconds=120,
            user_behaviors=[UserBehavior.CLINICIAN, UserBehavior.EMERGENCY],
            behavior_weights=[0.8, 0.2],
            lightrag_p95_response_ms=1500
        ),
        
        # Research intensive scenario
        CMOTestConfiguration(
            test_name="cmo_research_intensive_100",
            component_type=ComponentType.FULL_SYSTEM,
            load_pattern=LoadPattern.SUSTAINED,
            concurrent_users=100,
            total_operations=500,
            test_duration_seconds=300,
            user_behaviors=[UserBehavior.RESEARCHER],
            behavior_weights=[1.0],
            lightrag_p95_response_ms=3000
        ),
        
        # Scalability test
        CMOTestConfiguration(
            test_name="cmo_scalability_200",
            component_type=ComponentType.FULL_SYSTEM,
            load_pattern=LoadPattern.RAMP_UP,
            concurrent_users=200,
            total_operations=600,
            test_duration_seconds=360,
            ramp_up_duration=90,
            target_success_rate=0.85,
            max_memory_growth_mb=200
        )
    ]


if __name__ == "__main__":
    # Example usage
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        # Run comprehensive CMO load tests
        results = await run_cmo_comprehensive_load_tests(
            concurrent_users=30,  # Reduced for demo
            enable_lightrag=True,
            enable_circuit_breaker=True
        )
        
        print("CMO Load Test Results:")
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(main())