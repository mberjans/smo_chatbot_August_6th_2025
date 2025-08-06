"""
Cost-Based Circuit Breaker System for Clinical Metabolomics Oracle LightRAG Integration

This module provides cost-aware circuit breaker functionality that integrates with the existing
circuit breaker infrastructure to implement budget-based operation limiting and protection.

Classes:
    - CostThresholdRule: Rule definition for cost-based circuit breaking
    - CostBasedCircuitBreaker: Enhanced circuit breaker with cost awareness
    - BudgetProtectionPolicy: Policy configuration for budget protection
    - CostCircuitBreakerManager: Manager for multiple cost-aware circuit breakers
    - OperationCostEstimator: Cost estimation for operations before execution

The cost-based circuit breaker system supports:
    - Integration with existing CircuitBreaker infrastructure
    - Budget-aware operation limiting and throttling
    - Predictive cost analysis before operation execution
    - Dynamic threshold adjustment based on budget utilization
    - Multi-tier protection with graceful degradation
    - Integration with budget monitoring and alert systems
    - Comprehensive audit trail for budget protection decisions
"""

import time
import threading
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import uuid

from .budget_manager import BudgetManager, BudgetAlert, AlertLevel
from .cost_persistence import CostPersistence, CostRecord, ResearchCategory
from .realtime_budget_monitor import RealTimeBudgetMonitor, BudgetMonitoringEvent


class CircuitBreakerState(Enum):
    """States for cost-based circuit breaker."""
    
    CLOSED = "closed"           # Normal operation
    OPEN = "open"              # Blocking operations
    HALF_OPEN = "half_open"    # Testing recovery
    BUDGET_LIMITED = "budget_limited"  # Budget-based throttling


class CostThresholdType(Enum):
    """Types of cost thresholds for circuit breaking."""
    
    ABSOLUTE_DAILY = "absolute_daily"     # Absolute daily cost limit
    ABSOLUTE_MONTHLY = "absolute_monthly" # Absolute monthly cost limit
    PERCENTAGE_DAILY = "percentage_daily" # Percentage of daily budget
    PERCENTAGE_MONTHLY = "percentage_monthly" # Percentage of monthly budget
    RATE_BASED = "rate_based"            # Cost rate per hour/minute
    OPERATION_COST = "operation_cost"    # Per-operation cost limit


@dataclass
class CostThresholdRule:
    """Rule definition for cost-based circuit breaking."""
    
    rule_id: str
    threshold_type: CostThresholdType
    threshold_value: float
    action: str = "block"  # block, throttle, alert_only
    priority: int = 1      # Higher number = higher priority
    
    # Conditions
    applies_to_operations: Optional[List[str]] = None
    applies_to_categories: Optional[List[str]] = None
    time_window_minutes: Optional[float] = None
    
    # Action parameters
    throttle_factor: float = 0.5  # Reduce operation rate by this factor
    allow_emergency_override: bool = False
    cooldown_minutes: float = 5.0
    
    # Recovery conditions
    recovery_threshold: Optional[float] = None
    recovery_window_minutes: float = 10.0
    
    def __post_init__(self):
        """Validate rule configuration."""
        if self.threshold_value <= 0:
            raise ValueError("Threshold value must be positive")
        
        if self.action not in ["block", "throttle", "alert_only"]:
            raise ValueError("Action must be 'block', 'throttle', or 'alert_only'")
        
        if self.throttle_factor <= 0 or self.throttle_factor > 1:
            raise ValueError("Throttle factor must be between 0 and 1")


class OperationCostEstimator:
    """Cost estimation system for operations before execution."""
    
    def __init__(self, cost_persistence: CostPersistence, logger: Optional[logging.Logger] = None):
        """Initialize operation cost estimator."""
        self.cost_persistence = cost_persistence
        self.logger = logger or logging.getLogger(__name__)
        
        # Cost models and historical data
        self._operation_costs: Dict[str, List[float]] = defaultdict(list)
        self._model_costs: Dict[str, Dict[str, float]] = {}  # model -> {metric -> cost}
        self._token_cost_rates: Dict[str, Dict[str, float]] = {}  # model -> {token_type -> rate}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        self._initialize_cost_models()
    
    def _initialize_cost_models(self) -> None:
        """Initialize cost models with default rates."""
        # OpenAI pricing (as of 2025) - these should be configurable
        self._token_cost_rates = {
            "gpt-4o-mini": {
                "input": 0.000150 / 1000,   # $0.150 per 1M tokens
                "output": 0.000600 / 1000   # $0.600 per 1M tokens
            },
            "gpt-4o": {
                "input": 0.005000 / 1000,   # $5.00 per 1M tokens  
                "output": 0.015000 / 1000   # $15.00 per 1M tokens
            },
            "text-embedding-3-small": {
                "input": 0.000020 / 1000    # $0.02 per 1M tokens
            },
            "text-embedding-3-large": {
                "input": 0.000130 / 1000    # $0.13 per 1M tokens
            }
        }
        
        self.logger.info("Cost estimation models initialized")
    
    def update_historical_costs(self, operation_type: str, actual_cost: float) -> None:
        """Update historical cost data for improved estimation."""
        with self._lock:
            self._operation_costs[operation_type].append(actual_cost)
            
            # Keep only recent costs for estimation
            if len(self._operation_costs[operation_type]) > 1000:
                self._operation_costs[operation_type] = self._operation_costs[operation_type][-1000:]
    
    def estimate_operation_cost(self,
                              operation_type: str,
                              model_name: Optional[str] = None,
                              estimated_tokens: Optional[Dict[str, int]] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Estimate cost for an operation before execution.
        
        Args:
            operation_type: Type of operation (llm_call, embedding_call, etc.)
            model_name: Model to be used
            estimated_tokens: Token estimates {input: count, output: count}
            **kwargs: Additional parameters for cost estimation
            
        Returns:
            Dict containing cost estimate and confidence level
        """
        with self._lock:
            try:
                # Model-based estimation if tokens provided
                if model_name and estimated_tokens and model_name in self._token_cost_rates:
                    model_rates = self._token_cost_rates[model_name]
                    estimated_cost = 0.0
                    
                    for token_type, count in estimated_tokens.items():
                        rate_key = "input" if token_type in ["input", "prompt"] else "output"
                        if rate_key in model_rates:
                            estimated_cost += count * model_rates[rate_key]
                    
                    return {
                        'estimated_cost': estimated_cost,
                        'confidence': 0.9,  # High confidence for model-based estimates
                        'method': 'token_based',
                        'model_used': model_name,
                        'tokens_estimated': estimated_tokens
                    }
                
                # Historical average estimation
                if operation_type in self._operation_costs and self._operation_costs[operation_type]:
                    costs = self._operation_costs[operation_type]
                    avg_cost = statistics.mean(costs[-100:])  # Recent average
                    std_cost = statistics.stdev(costs[-100:]) if len(costs) > 1 else avg_cost * 0.1
                    
                    # Use conservative estimate (mean + 1 std)
                    estimated_cost = avg_cost + std_cost
                    confidence = min(0.8, len(costs) / 100)
                    
                    return {
                        'estimated_cost': estimated_cost,
                        'confidence': confidence,
                        'method': 'historical_average',
                        'samples': len(costs),
                        'average': avg_cost,
                        'std_deviation': std_cost
                    }
                
                # Default estimation based on operation type
                default_estimates = {
                    'llm_call': 0.01,
                    'embedding_call': 0.001,
                    'batch_operation': 0.05,
                    'document_processing': 0.02
                }
                
                estimated_cost = default_estimates.get(operation_type, 0.005)
                
                return {
                    'estimated_cost': estimated_cost,
                    'confidence': 0.3,  # Low confidence for defaults
                    'method': 'default_estimate',
                    'note': 'Using default estimate - consider providing token estimates for better accuracy'
                }
                
            except Exception as e:
                self.logger.error(f"Error estimating operation cost: {e}")
                return {
                    'estimated_cost': 0.01,  # Conservative fallback
                    'confidence': 0.1,
                    'method': 'fallback',
                    'error': str(e)
                }


class CostBasedCircuitBreaker:
    """Enhanced circuit breaker with cost awareness and budget protection."""
    
    def __init__(self,
                 name: str,
                 budget_manager: BudgetManager,
                 cost_estimator: OperationCostEstimator,
                 threshold_rules: List[CostThresholdRule],
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize cost-based circuit breaker.
        
        Args:
            name: Name identifier for the circuit breaker
            budget_manager: Budget management system
            cost_estimator: Operation cost estimator
            threshold_rules: Cost-based threshold rules
            failure_threshold: Traditional failure threshold
            recovery_timeout: Recovery timeout in seconds
            logger: Logger instance
        """
        self.name = name
        self.budget_manager = budget_manager
        self.cost_estimator = cost_estimator
        self.threshold_rules = sorted(threshold_rules, key=lambda r: r.priority, reverse=True)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.logger = logger or logging.getLogger(__name__)
        
        # Circuit breaker state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.half_open_start_time: Optional[float] = None
        
        # Cost-based state
        self._active_rules: List[CostThresholdRule] = []
        self._rule_cooldowns: Dict[str, float] = {}
        self._throttle_rate: float = 1.0  # 1.0 = no throttling
        self._last_throttle_check: float = 0
        
        # Statistics and monitoring
        self._operation_stats = {
            'total_calls': 0,
            'allowed_calls': 0,
            'blocked_calls': 0,
            'throttled_calls': 0,
            'cost_blocked_calls': 0,
            'total_estimated_cost': 0.0,
            'total_actual_cost': 0.0,
            'cost_savings': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info(f"Cost-based circuit breaker '{name}' initialized with {len(threshold_rules)} rules")
    
    def call(self, operation_callable: Callable, *args, **kwargs) -> Any:
        """
        Execute operation through cost-aware circuit breaker.
        
        Args:
            operation_callable: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation if allowed
            
        Raises:
            CircuitBreakerError: If operation is blocked by circuit breaker
        """
        with self._lock:
            self._operation_stats['total_calls'] += 1
            
            # Check circuit breaker state
            current_state = self._update_state()
            
            # Extract operation metadata
            operation_type = kwargs.get('operation_type', 'unknown')
            model_name = kwargs.get('model_name')
            estimated_tokens = kwargs.get('estimated_tokens')
            
            # Estimate operation cost
            cost_estimate = self.cost_estimator.estimate_operation_cost(
                operation_type=operation_type,
                model_name=model_name,
                estimated_tokens=estimated_tokens
            )
            
            estimated_cost = cost_estimate['estimated_cost']
            self._operation_stats['total_estimated_cost'] += estimated_cost
            
            # Check cost-based rules
            cost_check_result = self._check_cost_rules(
                estimated_cost=estimated_cost,
                operation_type=operation_type,
                cost_estimate=cost_estimate
            )
            
            if not cost_check_result['allowed']:
                self._handle_cost_block(cost_check_result, estimated_cost)
                from .clinical_metabolomics_rag import CircuitBreakerError
                raise CircuitBreakerError(
                    f"Operation blocked by cost-based circuit breaker: {cost_check_result['reason']}"
                )
            
            # Check traditional circuit breaker state
            if current_state == CircuitBreakerState.OPEN:
                self._operation_stats['blocked_calls'] += 1
                from .clinical_metabolomics_rag import CircuitBreakerError
                raise CircuitBreakerError(f"Circuit breaker '{self.name}' is open")
            
            # Apply throttling if needed
            if self._throttle_rate < 1.0:
                throttle_delay = self._calculate_throttle_delay()
                if throttle_delay > 0:
                    time.sleep(throttle_delay)
                    self._operation_stats['throttled_calls'] += 1
            
            # Execute operation
            try:
                start_time = time.time()
                result = operation_callable(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record success
                self._record_success(estimated_cost, execution_time)
                return result
                
            except Exception as e:
                # Record failure
                self._record_failure(estimated_cost)
                raise
    
    def _update_state(self) -> CircuitBreakerState:
        """Update circuit breaker state based on current conditions."""
        now = time.time()
        
        # Check if we should transition from OPEN to HALF_OPEN
        if (self.state == CircuitBreakerState.OPEN and 
            self.last_failure_time and 
            now - self.last_failure_time >= self.recovery_timeout):
            
            self.state = CircuitBreakerState.HALF_OPEN
            self.half_open_start_time = now
            self.logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
        
        # Check if we should transition from HALF_OPEN to CLOSED
        elif (self.state == CircuitBreakerState.HALF_OPEN and
              self.last_success_time and 
              self.last_success_time > (self.half_open_start_time or 0)):
            
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
        
        # Check cost-based conditions
        cost_state = self._check_cost_based_state()
        if cost_state != CircuitBreakerState.CLOSED:
            self.state = cost_state
        
        return self.state
    
    def _check_cost_based_state(self) -> CircuitBreakerState:
        """Check if cost conditions require state change."""
        try:
            budget_status = self.budget_manager.get_budget_summary()
            
            # Check if budget is exceeded
            daily_budget = budget_status.get('daily_budget', {})
            monthly_budget = budget_status.get('monthly_budget', {})
            
            if (daily_budget.get('over_budget') or monthly_budget.get('over_budget')):
                return CircuitBreakerState.OPEN
            
            # Check for budget limiting conditions
            daily_pct = daily_budget.get('percentage_used', 0)
            monthly_pct = monthly_budget.get('percentage_used', 0)
            max_pct = max(daily_pct, monthly_pct)
            
            if max_pct >= 95:  # 95% threshold for budget limiting
                return CircuitBreakerState.BUDGET_LIMITED
            
            return CircuitBreakerState.CLOSED
            
        except Exception as e:
            self.logger.error(f"Error checking cost-based state: {e}")
            return CircuitBreakerState.CLOSED
    
    def _check_cost_rules(self,
                         estimated_cost: float,
                         operation_type: str,
                         cost_estimate: Dict[str, Any]) -> Dict[str, Any]:
        """Check cost-based rules for operation approval."""
        now = time.time()
        
        try:
            budget_status = self.budget_manager.get_budget_summary()
            
            for rule in self.threshold_rules:
                # Check cooldown
                if rule.rule_id in self._rule_cooldowns:
                    if now - self._rule_cooldowns[rule.rule_id] < (rule.cooldown_minutes * 60):
                        continue
                
                # Check if rule applies to this operation
                if rule.applies_to_operations and operation_type not in rule.applies_to_operations:
                    continue
                
                # Evaluate rule threshold
                threshold_exceeded, current_value = self._evaluate_rule_threshold(rule, budget_status, estimated_cost)
                
                if threshold_exceeded:
                    return {
                        'allowed': rule.action == 'alert_only',
                        'rule_triggered': rule.rule_id,
                        'action': rule.action,
                        'reason': f"Cost rule '{rule.rule_id}' triggered: {current_value} >= {rule.threshold_value}",
                        'current_value': current_value,
                        'threshold': rule.threshold_value,
                        'estimated_cost': estimated_cost
                    }
            
            return {'allowed': True}
            
        except Exception as e:
            self.logger.error(f"Error checking cost rules: {e}")
            return {'allowed': True}  # Fail open for safety
    
    def _evaluate_rule_threshold(self,
                                rule: CostThresholdRule,
                                budget_status: Dict[str, Any],
                                estimated_cost: float) -> Tuple[bool, float]:
        """Evaluate if a specific rule threshold is exceeded."""
        
        if rule.threshold_type == CostThresholdType.ABSOLUTE_DAILY:
            daily_cost = budget_status.get('daily_budget', {}).get('total_cost', 0)
            projected_cost = daily_cost + estimated_cost
            return projected_cost >= rule.threshold_value, projected_cost
        
        elif rule.threshold_type == CostThresholdType.ABSOLUTE_MONTHLY:
            monthly_cost = budget_status.get('monthly_budget', {}).get('total_cost', 0)
            projected_cost = monthly_cost + estimated_cost
            return projected_cost >= rule.threshold_value, projected_cost
        
        elif rule.threshold_type == CostThresholdType.PERCENTAGE_DAILY:
            daily_pct = budget_status.get('daily_budget', {}).get('percentage_used', 0)
            return daily_pct >= rule.threshold_value, daily_pct
        
        elif rule.threshold_type == CostThresholdType.PERCENTAGE_MONTHLY:
            monthly_pct = budget_status.get('monthly_budget', {}).get('percentage_used', 0)
            return monthly_pct >= rule.threshold_value, monthly_pct
        
        elif rule.threshold_type == CostThresholdType.OPERATION_COST:
            return estimated_cost >= rule.threshold_value, estimated_cost
        
        elif rule.threshold_type == CostThresholdType.RATE_BASED:
            # Check cost rate per hour (simplified)
            window_minutes = rule.time_window_minutes or 60
            window_seconds = window_minutes * 60
            recent_cost = self._get_recent_cost(window_seconds)
            cost_rate_per_hour = (recent_cost / window_seconds) * 3600
            return cost_rate_per_hour >= rule.threshold_value, cost_rate_per_hour
        
        return False, 0.0
    
    def _get_recent_cost(self, window_seconds: float) -> float:
        """Get total cost in recent time window."""
        # Simplified implementation - would normally query cost persistence
        return self._operation_stats.get('total_actual_cost', 0) * (window_seconds / 3600)
    
    def _handle_cost_block(self, cost_check_result: Dict[str, Any], estimated_cost: float) -> None:
        """Handle cost-based operation blocking."""
        rule_id = cost_check_result.get('rule_triggered', 'unknown')
        action = cost_check_result.get('action', 'block')
        
        # Update statistics
        self._operation_stats['cost_blocked_calls'] += 1
        self._operation_stats['cost_savings'] += estimated_cost
        
        # Set cooldown for triggered rule
        self._rule_cooldowns[rule_id] = time.time()
        
        # Update throttling if action is throttle
        if action == 'throttle':
            for rule in self.threshold_rules:
                if rule.rule_id == rule_id:
                    self._throttle_rate = rule.throttle_factor
                    break
        
        # Log the block
        self.logger.warning(
            f"Cost-based circuit breaker '{self.name}' blocked operation: {cost_check_result['reason']}"
        )
    
    def _calculate_throttle_delay(self) -> float:
        """Calculate throttle delay based on current throttle rate."""
        if self._throttle_rate >= 1.0:
            return 0.0
        
        # Simple throttling: delay = (1 - throttle_rate) seconds
        base_delay = 1.0 - self._throttle_rate
        
        # Add some randomness to avoid thundering herd
        import random
        jitter = random.uniform(0.8, 1.2)
        
        return base_delay * jitter
    
    def _record_success(self, estimated_cost: float, execution_time: float) -> None:
        """Record successful operation execution."""
        self.last_success_time = time.time()
        self._operation_stats['allowed_calls'] += 1
        
        # Reset failure count if we were in failure state
        if self.state in [CircuitBreakerState.HALF_OPEN]:
            self.failure_count = 0
        
        # Update cost estimator with actual cost (if available)
        # This would typically be updated after getting actual cost from API response
    
    def _record_failure(self, estimated_cost: float) -> None:
        """Record failed operation execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Transition to OPEN if failure threshold exceeded
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.error(f"Circuit breaker '{self.name}' opened due to failures")
    
    def update_actual_cost(self, operation_id: str, actual_cost: float, operation_type: str) -> None:
        """Update with actual operation cost for learning."""
        with self._lock:
            self._operation_stats['total_actual_cost'] += actual_cost
            self.cost_estimator.update_historical_costs(operation_type, actual_cost)
    
    def force_open(self, reason: str = "Manual intervention") -> None:
        """Force circuit breaker to open state."""
        with self._lock:
            self.state = CircuitBreakerState.OPEN
            self.last_failure_time = time.time()
            self.logger.warning(f"Circuit breaker '{self.name}' forced open: {reason}")
    
    def force_close(self, reason: str = "Manual intervention") -> None:
        """Force circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self._throttle_rate = 1.0
            self._rule_cooldowns.clear()
            self.logger.info(f"Circuit breaker '{self.name}' forced closed: {reason}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status and statistics."""
        with self._lock:
            now = time.time()
            
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'throttle_rate': self._throttle_rate,
                'active_rules': len(self._active_rules),
                'statistics': self._operation_stats.copy(),
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'time_since_last_failure': now - self.last_failure_time if self.last_failure_time else None,
                'recovery_timeout': self.recovery_timeout,
                'rules_count': len(self.threshold_rules),
                'cost_efficiency': {
                    'estimated_vs_actual_ratio': (
                        self._operation_stats['total_estimated_cost'] / max(self._operation_stats['total_actual_cost'], 0.001)
                    ),
                    'cost_savings': self._operation_stats['cost_savings'],
                    'block_rate': (
                        self._operation_stats['cost_blocked_calls'] / max(self._operation_stats['total_calls'], 1)
                    )
                },
                'timestamp': now
            }


class CostCircuitBreakerManager:
    """Manager for multiple cost-aware circuit breakers."""
    
    def __init__(self,
                 budget_manager: BudgetManager,
                 cost_persistence: CostPersistence,
                 logger: Optional[logging.Logger] = None):
        """Initialize cost circuit breaker manager."""
        self.budget_manager = budget_manager
        self.cost_persistence = cost_persistence
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize cost estimator
        self.cost_estimator = OperationCostEstimator(cost_persistence, logger)
        
        # Circuit breakers registry
        self._circuit_breakers: Dict[str, CostBasedCircuitBreaker] = {}
        self._default_rules: List[CostThresholdRule] = []
        
        # Manager statistics
        self._manager_stats = {
            'breakers_created': 0,
            'total_operations': 0,
            'total_blocks': 0,
            'total_cost_saved': 0.0,
            'start_time': time.time()
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        self._initialize_default_rules()
        self.logger.info("Cost circuit breaker manager initialized")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default cost-based rules."""
        self._default_rules = [
            # Daily budget protection
            CostThresholdRule(
                rule_id="daily_budget_90_pct",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=90.0,
                action="throttle",
                priority=10,
                throttle_factor=0.5,
                cooldown_minutes=10.0
            ),
            CostThresholdRule(
                rule_id="daily_budget_100_pct",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=100.0,
                action="block",
                priority=20,
                cooldown_minutes=5.0
            ),
            
            # Monthly budget protection
            CostThresholdRule(
                rule_id="monthly_budget_95_pct",
                threshold_type=CostThresholdType.PERCENTAGE_MONTHLY,
                threshold_value=95.0,
                action="throttle",
                priority=15,
                throttle_factor=0.3,
                cooldown_minutes=15.0
            ),
            CostThresholdRule(
                rule_id="monthly_budget_100_pct",
                threshold_type=CostThresholdType.PERCENTAGE_MONTHLY,
                threshold_value=100.0,
                action="block",
                priority=25,
                cooldown_minutes=5.0
            ),
            
            # High-cost operation protection
            CostThresholdRule(
                rule_id="high_cost_operation",
                threshold_type=CostThresholdType.OPERATION_COST,
                threshold_value=1.0,  # $1.00 per operation
                action="alert_only",
                priority=5,
                cooldown_minutes=1.0
            ),
            
            # Rate-based protection
            CostThresholdRule(
                rule_id="cost_rate_spike",
                threshold_type=CostThresholdType.RATE_BASED,
                threshold_value=10.0,  # $10 per hour
                action="throttle",
                priority=8,
                throttle_factor=0.7,
                time_window_minutes=60,
                cooldown_minutes=30.0
            )
        ]
    
    def create_circuit_breaker(self,
                             name: str,
                             threshold_rules: Optional[List[CostThresholdRule]] = None,
                             failure_threshold: int = 5,
                             recovery_timeout: float = 60.0) -> CostBasedCircuitBreaker:
        """Create a new cost-based circuit breaker."""
        with self._lock:
            if name in self._circuit_breakers:
                raise ValueError(f"Circuit breaker '{name}' already exists")
            
            # Use provided rules or defaults
            rules = threshold_rules or self._default_rules.copy()
            
            # Create circuit breaker
            circuit_breaker = CostBasedCircuitBreaker(
                name=name,
                budget_manager=self.budget_manager,
                cost_estimator=self.cost_estimator,
                threshold_rules=rules,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                logger=self.logger
            )
            
            self._circuit_breakers[name] = circuit_breaker
            self._manager_stats['breakers_created'] += 1
            
            self.logger.info(f"Created cost-based circuit breaker '{name}'")
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CostBasedCircuitBreaker]:
        """Get circuit breaker by name."""
        return self._circuit_breakers.get(name)
    
    def execute_with_protection(self,
                              breaker_name: str,
                              operation_callable: Callable,
                              operation_type: str,
                              *args,
                              **kwargs) -> Any:
        """Execute operation with cost-based circuit breaker protection."""
        with self._lock:
            self._manager_stats['total_operations'] += 1
            
            # Get or create circuit breaker
            if breaker_name not in self._circuit_breakers:
                circuit_breaker = self.create_circuit_breaker(breaker_name)
            else:
                circuit_breaker = self._circuit_breakers[breaker_name]
            
            # Add operation metadata
            kwargs['operation_type'] = operation_type
            
            try:
                return circuit_breaker.call(operation_callable, *args, **kwargs)
            except Exception as e:
                if "blocked by cost-based circuit breaker" in str(e):
                    self._manager_stats['total_blocks'] += 1
                raise
    
    def update_operation_cost(self,
                            breaker_name: str,
                            operation_id: str,
                            actual_cost: float,
                            operation_type: str) -> None:
        """Update actual operation cost across relevant circuit breakers."""
        if breaker_name in self._circuit_breakers:
            self._circuit_breakers[breaker_name].update_actual_cost(
                operation_id, actual_cost, operation_type
            )
        
        # Update global cost estimator
        self.cost_estimator.update_historical_costs(operation_type, actual_cost)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            breaker_statuses = {}
            total_cost_saved = 0.0
            
            for name, breaker in self._circuit_breakers.items():
                status = breaker.get_status()
                breaker_statuses[name] = status
                total_cost_saved += status['statistics']['cost_savings']
            
            self._manager_stats['total_cost_saved'] = total_cost_saved
            
            return {
                'circuit_breakers': breaker_statuses,
                'manager_statistics': self._manager_stats.copy(),
                'cost_estimator_stats': {
                    'operation_types_tracked': len(self.cost_estimator._operation_costs),
                    'total_historical_datapoints': sum(
                        len(costs) for costs in self.cost_estimator._operation_costs.values()
                    )
                },
                'system_health': self._assess_system_health(),
                'timestamp': time.time()
            }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        if not self._circuit_breakers:
            return {'status': 'no_breakers', 'message': 'No circuit breakers configured'}
        
        open_breakers = [
            name for name, breaker in self._circuit_breakers.items()
            if breaker.state == CircuitBreakerState.OPEN
        ]
        
        budget_limited_breakers = [
            name for name, breaker in self._circuit_breakers.items()
            if breaker.state == CircuitBreakerState.BUDGET_LIMITED
        ]
        
        if open_breakers:
            return {
                'status': 'degraded',
                'message': f'Circuit breakers open: {", ".join(open_breakers)}',
                'open_breakers': open_breakers
            }
        
        if budget_limited_breakers:
            return {
                'status': 'budget_limited',
                'message': f'Budget-limited breakers: {", ".join(budget_limited_breakers)}',
                'limited_breakers': budget_limited_breakers
            }
        
        return {
            'status': 'healthy',
            'message': 'All circuit breakers operational',
            'total_breakers': len(self._circuit_breakers)
        }
    
    def emergency_shutdown(self, reason: str = "Emergency shutdown") -> None:
        """Emergency shutdown of all circuit breakers."""
        with self._lock:
            for name, breaker in self._circuit_breakers.items():
                breaker.force_open(reason)
            
            self.logger.critical(f"Emergency shutdown of all circuit breakers: {reason}")
    
    def reset_all_breakers(self, reason: str = "Manual reset") -> None:
        """Reset all circuit breakers to closed state."""
        with self._lock:
            for name, breaker in self._circuit_breakers.items():
                breaker.force_close(reason)
            
            self.logger.info(f"All circuit breakers reset: {reason}")
    
    def close(self) -> None:
        """Clean shutdown of circuit breaker manager."""
        try:
            self.logger.info("Shutting down cost circuit breaker manager")
            # Circuit breakers don't need explicit cleanup
        except Exception as e:
            self.logger.error(f"Error during circuit breaker manager shutdown: {e}")