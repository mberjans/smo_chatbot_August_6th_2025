# LightRAG Integration Patterns and Feature Flag Guide

## Table of Contents

1. [Feature Flag System Overview](#feature-flag-system-overview)
2. [Optional Integration Patterns](#optional-integration-patterns)
3. [Rollout Strategies](#rollout-strategies)
4. [Circuit Breaker Patterns](#circuit-breaker-patterns)
5. [A/B Testing Framework](#ab-testing-framework)
6. [Configuration Management](#configuration-management)
7. [Monitoring and Analytics](#monitoring-and-analytics)
8. [Best Practices](#best-practices)

---

## Feature Flag System Overview

The LightRAG integration includes a sophisticated feature flag system that enables safe, gradual deployment and A/B testing. The system is designed to minimize risk while maximizing flexibility in rolling out the new RAG capabilities.

### Core Components

```python
from lightrag_integration import (
    FeatureFlagManager,
    RolloutManager,
    CircuitBreaker,
    ABTestManager,
    UserCohortManager
)

# Initialize feature flag system
feature_flags = FeatureFlagManager(
    config_source="environment",  # or "database", "config_file"
    fallback_behavior="perplexity",  # Fallback to existing system
    circuit_breaker_enabled=True
)
```

### Environment Configuration

```bash
# Basic Feature Flags
LIGHTRAG_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=25  # 25% of users get LightRAG

# Quality Gates
LIGHTRAG_QUALITY_THRESHOLD=0.8
LIGHTRAG_MIN_SUCCESS_RATE=0.95

# Circuit Breaker
LIGHTRAG_CIRCUIT_BREAKER_ENABLED=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
LIGHTRAG_CIRCUIT_BREAKER_TIMEOUT_SECONDS=300

# A/B Testing
LIGHTRAG_AB_TEST_ENABLED=true
LIGHTRAG_AB_TEST_NAME="lightrag_vs_perplexity_q1_2025"
LIGHTRAG_AB_TEST_SAMPLE_SIZE=1000

# Budget Control
LIGHTRAG_DAILY_BUDGET_USD=100.0
LIGHTRAG_BUDGET_ALERT_THRESHOLD=0.8
LIGHTRAG_EMERGENCY_STOP_ENABLED=true
```

---

## Optional Integration Patterns

### 1. Shadow Mode Integration

Deploy LightRAG in shadow mode where it processes queries in parallel but doesn't affect user responses:

```python
class ShadowModeIntegration:
    """Run LightRAG in shadow mode for validation"""
    
    def __init__(self):
        self.lightrag_system = None
        self.perplexity_system = None
        self.comparison_logger = ComparisonLogger()
        
    async def initialize(self):
        self.lightrag_system = await create_clinical_rag_system()
        # Existing Perplexity system initialization
        
    async def process_query_shadow_mode(self, query: str, user_id: str):
        """Process with both systems, return only Perplexity result"""
        
        # Primary response (what user sees)
        perplexity_response = await self.perplexity_system.query(query)
        
        # Shadow response (for comparison only)
        asyncio.create_task(
            self._process_shadow_query(query, user_id, perplexity_response)
        )
        
        return perplexity_response  # User only sees this
    
    async def _process_shadow_query(self, query: str, user_id: str, perplexity_response: str):
        """Background shadow processing"""
        try:
            lightrag_response = await self.lightrag_system.query(query)
            
            # Compare responses
            comparison = await self._compare_responses(
                query=query,
                perplexity_response=perplexity_response,
                lightrag_response=lightrag_response
            )
            
            # Log for analysis
            await self.comparison_logger.log_comparison(
                user_id=user_id,
                query=query,
                comparison=comparison
            )
            
        except Exception as e:
            logger.warning(f"Shadow mode error: {e}")
```

### 2. Canary Deployment Integration

Deploy to a small percentage of users with automatic promotion/rollback:

```python
from lightrag_integration import CanaryDeployment

class CanaryIntegration:
    """Canary deployment with automatic promotion"""
    
    def __init__(self):
        self.canary = CanaryDeployment(
            canary_percentage=5,  # Start with 5%
            promotion_criteria={
                'success_rate': 0.95,      # 95% success rate
                'quality_score': 0.8,      # 80% quality score
                'response_time_p95': 5000   # 95th percentile under 5s
            },
            observation_period_hours=24,
            max_canary_percentage=50,
            auto_promote=True,
            auto_rollback=True
        )
        
    async def process_query_canary(self, query: str, user_id: str):
        """Process query with canary deployment logic"""
        
        if self.canary.should_use_canary(user_id):
            try:
                # Use LightRAG (canary)
                response = await self.lightrag_system.query(query)
                
                # Track canary metrics
                await self.canary.track_success(
                    user_id=user_id,
                    response_time=response.processing_time,
                    quality_score=response.quality_score
                )
                
                return response, "lightrag_canary"
                
            except Exception as e:
                # Track failure
                await self.canary.track_failure(user_id, str(e))
                
                # Fallback to stable system
                return await self.perplexity_system.query(query), "perplexity_fallback"
        else:
            # Use stable system (Perplexity)
            return await self.perplexity_system.query(query), "perplexity_stable"
```

### 3. Blue-Green Deployment Integration

Maintain two complete environments for zero-downtime deployments:

```python
class BlueGreenIntegration:
    """Blue-green deployment for zero-downtime updates"""
    
    def __init__(self):
        self.blue_environment = None   # Current production
        self.green_environment = None  # New version being deployed
        self.traffic_manager = TrafficManager()
        self.health_checker = HealthChecker()
        
    async def initialize_environments(self):
        """Initialize both blue and green environments"""
        
        # Blue (current production)
        self.blue_environment = await create_production_system("blue")
        
        # Green (new version)
        self.green_environment = await create_production_system("green")
        
    async def process_query_blue_green(self, query: str, user_id: str):
        """Process query with blue-green deployment"""
        
        # Determine which environment to use
        active_environment = await self.traffic_manager.get_active_environment()
        
        if active_environment == "blue":
            system = self.blue_environment
        else:
            system = self.green_environment
        
        try:
            response = await system.query(query)
            
            # Track health metrics for active environment
            await self.health_checker.track_query_success(
                environment=active_environment,
                response_time=response.processing_time,
                quality_score=response.quality_score
            )
            
            return response
            
        except Exception as e:
            # If active environment fails, try the other
            fallback_env = "green" if active_environment == "blue" else "blue"
            fallback_system = self.green_environment if active_environment == "blue" else self.blue_environment
            
            logger.error(f"Environment {active_environment} failed, trying {fallback_env}")
            
            return await fallback_system.query(query)
    
    async def switch_traffic(self, target_environment: str):
        """Switch traffic from blue to green or vice versa"""
        
        # Validate target environment health
        health_status = await self.health_checker.check_environment_health(target_environment)
        
        if not health_status.is_healthy:
            raise Exception(f"Cannot switch to unhealthy environment: {target_environment}")
        
        # Gradually switch traffic
        await self.traffic_manager.gradual_traffic_switch(
            target_environment=target_environment,
            switch_duration_minutes=10
        )
        
        logger.info(f"Traffic switched to {target_environment}")
```

### 4. Feature Toggle Integration

Fine-grained control over specific LightRAG features:

```python
class FeatureToggleIntegration:
    """Fine-grained feature toggle control"""
    
    def __init__(self):
        self.feature_toggles = {
            'lightrag_enabled': FeatureToggle('LIGHTRAG_ENABLED', default=False),
            'quality_assessment': FeatureToggle('LIGHTRAG_QUALITY_ASSESSMENT', default=True),
            'response_caching': FeatureToggle('LIGHTRAG_RESPONSE_CACHING', default=True),
            'cost_monitoring': FeatureToggle('LIGHTRAG_COST_MONITORING', default=True),
            'advanced_routing': FeatureToggle('LIGHTRAG_ADVANCED_ROUTING', default=False),
            'multi_language': FeatureToggle('LIGHTRAG_MULTI_LANGUAGE', default=True),
            'citation_processing': FeatureToggle('LIGHTRAG_CITATIONS', default=True),
            'analytics_collection': FeatureToggle('LIGHTRAG_ANALYTICS', default=True)
        }
        
    async def process_query_with_toggles(self, query: str, user_id: str):
        """Process query with feature toggles"""
        
        if not self.feature_toggles['lightrag_enabled'].is_enabled(user_id):
            return await self.perplexity_system.query(query)
        
        # Use LightRAG with selective features
        response_data = await self.lightrag_system.query(query)
        
        # Apply optional enhancements based on toggles
        if self.feature_toggles['quality_assessment'].is_enabled(user_id):
            response_data.quality_score = await self.assess_quality(query, response_data.response)
        
        if self.feature_toggles['response_caching'].is_enabled(user_id):
            await self.cache_response(query, response_data)
        
        if self.feature_toggles['cost_monitoring'].is_enabled(user_id):
            await self.track_costs(query, response_data)
        
        if self.feature_toggles['citation_processing'].is_enabled(user_id):
            response_data.citations = await self.extract_citations(response_data.response)
        
        if self.feature_toggles['analytics_collection'].is_enabled(user_id):
            await self.collect_analytics(user_id, query, response_data)
        
        return response_data

class FeatureToggle:
    """Individual feature toggle implementation"""
    
    def __init__(self, env_var: str, default: bool = False, user_percentage: float = 100.0):
        self.env_var = env_var
        self.default = default
        self.user_percentage = user_percentage
        
    def is_enabled(self, user_id: str = None) -> bool:
        """Check if feature is enabled for user"""
        
        # Check environment variable
        env_value = os.getenv(self.env_var)
        if env_value is not None:
            base_enabled = env_value.lower() in ('true', '1', 'yes', 'on')
        else:
            base_enabled = self.default
        
        if not base_enabled:
            return False
        
        # Check user percentage if user_id provided
        if user_id and self.user_percentage < 100.0:
            user_hash = hash(f"{self.env_var}_{user_id}") % 100
            return user_hash < self.user_percentage
        
        return True
```

---

## Rollout Strategies

### 1. Percentage-Based Gradual Rollout

```python
from lightrag_integration import GradualRolloutManager

class GradualRolloutStrategy:
    """Implement gradual percentage-based rollout"""
    
    def __init__(self):
        self.rollout_manager = GradualRolloutManager(
            initial_percentage=5,
            stages=[5, 10, 25, 50, 75, 100],
            stage_duration_hours=24,
            success_criteria={
                'error_rate': {'max': 0.05},
                'quality_score': {'min': 0.8},
                'user_satisfaction': {'min': 0.85}
            }
        )
        
    async def should_use_lightrag(self, user_id: str) -> bool:
        """Determine if user should get LightRAG based on rollout percentage"""
        
        current_percentage = self.rollout_manager.get_current_percentage()
        
        # Consistent user assignment based on hash
        user_hash = hash(user_id) % 100
        
        return user_hash < current_percentage
    
    async def advance_rollout_if_ready(self):
        """Advance to next rollout stage if criteria met"""
        
        current_metrics = await self.get_current_metrics()
        
        if self.rollout_manager.should_advance_stage(current_metrics):
            await self.rollout_manager.advance_to_next_stage()
            
            # Notify team of advancement
            await self.notify_rollout_advancement()
        
        elif self.rollout_manager.should_rollback(current_metrics):
            await self.rollout_manager.rollback_to_previous_stage()
            
            # Alert team of rollback
            await self.alert_rollout_rollback(current_metrics)
```

### 2. Cohort-Based Rollout

```python
from lightrag_integration import UserCohortManager

class CohortBasedRollout:
    """Rollout based on user cohorts and characteristics"""
    
    def __init__(self):
        self.cohort_manager = UserCohortManager()
        self.rollout_config = {
            'power_users': {'enabled': True, 'percentage': 100},
            'beta_testers': {'enabled': True, 'percentage': 100},
            'research_institutions': {'enabled': True, 'percentage': 75},
            'educational_users': {'enabled': True, 'percentage': 50},
            'general_users': {'enabled': True, 'percentage': 25},
            'new_users': {'enabled': False, 'percentage': 0}
        }
        
    async def should_use_lightrag_for_cohort(self, user_id: str, user_metadata: dict) -> bool:
        """Determine LightRAG usage based on user cohort"""
        
        # Identify user cohort
        cohort = await self.cohort_manager.identify_user_cohort(user_id, user_metadata)
        
        # Get rollout config for cohort
        cohort_config = self.rollout_config.get(cohort, {'enabled': False, 'percentage': 0})
        
        if not cohort_config['enabled']:
            return False
        
        # Apply percentage within cohort
        if cohort_config['percentage'] < 100:
            user_hash = hash(f"{cohort}_{user_id}") % 100
            return user_hash < cohort_config['percentage']
        
        return True
    
    async def update_cohort_rollout(self, cohort: str, enabled: bool, percentage: float):
        """Update rollout configuration for specific cohort"""
        
        self.rollout_config[cohort] = {
            'enabled': enabled,
            'percentage': percentage
        }
        
        # Persist configuration
        await self.persist_rollout_config()
        
        logger.info(f"Updated {cohort} rollout: enabled={enabled}, percentage={percentage}%")
```

### 3. Geographic Rollout

```python
class GeographicRollout:
    """Rollout based on geographic regions"""
    
    def __init__(self):
        self.geographic_config = {
            'us_west': {'enabled': True, 'percentage': 100},
            'us_east': {'enabled': True, 'percentage': 75},
            'europe': {'enabled': True, 'percentage': 50},
            'asia_pacific': {'enabled': False, 'percentage': 0},
            'other': {'enabled': False, 'percentage': 0}
        }
        
    async def should_use_lightrag_for_region(self, user_ip: str, user_id: str) -> bool:
        """Determine LightRAG usage based on user's geographic region"""
        
        # Determine user's region from IP
        region = await self.get_user_region(user_ip)
        
        # Get regional configuration
        region_config = self.geographic_config.get(region, {'enabled': False, 'percentage': 0})
        
        if not region_config['enabled']:
            return False
        
        if region_config['percentage'] < 100:
            user_hash = hash(f"{region}_{user_id}") % 100
            return user_hash < region_config['percentage']
        
        return True
```

---

## Circuit Breaker Patterns

### 1. Basic Circuit Breaker

```python
from lightrag_integration import CircuitBreaker

class LightRAGCircuitBreaker:
    """Circuit breaker for LightRAG system protection"""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,      # Open after 5 failures
            recovery_timeout=300,     # Try recovery after 5 minutes
            expected_exception=Exception,
            name="lightrag_circuit_breaker"
        )
        
    async def protected_query(self, query: str) -> str:
        """Query with circuit breaker protection"""
        
        try:
            # Circuit breaker protection
            response = await self.circuit_breaker.call(
                self.lightrag_system.query, query, mode="hybrid"
            )
            return response, "lightrag"
            
        except CircuitBreakerOpenException:
            logger.warning("LightRAG circuit breaker open, using fallback")
            return await self.perplexity_fallback(query), "perplexity_fallback"
            
        except Exception as e:
            logger.error(f"LightRAG query failed: {e}")
            return await self.perplexity_fallback(query), "perplexity_error_fallback"
```

### 2. Quality-Based Circuit Breaker

```python
class QualityBasedCircuitBreaker:
    """Circuit breaker that opens based on quality metrics"""
    
    def __init__(self):
        self.quality_threshold = 0.7
        self.quality_window = 10  # Last 10 responses
        self.quality_scores = deque(maxlen=self.quality_window)
        self.circuit_open = False
        self.circuit_open_time = None
        self.recovery_timeout = 600  # 10 minutes
        
    async def protected_query_with_quality(self, query: str) -> tuple:
        """Query with quality-based circuit breaker"""
        
        # Check if circuit should be closed (recovery attempt)
        if self.circuit_open and self._should_attempt_recovery():
            self.circuit_open = False
            logger.info("Attempting circuit breaker recovery")
        
        if self.circuit_open:
            logger.warning("Quality circuit breaker open, using fallback")
            return await self.perplexity_fallback(query), "quality_circuit_open"
        
        try:
            # Process with LightRAG
            response = await self.lightrag_system.query(query)
            
            # Assess quality
            quality_score = await self.quality_suite.assess_response_quality(query, response)
            
            # Track quality
            self.quality_scores.append(quality_score)
            
            # Check if circuit should open
            if self._should_open_circuit():
                self.circuit_open = True
                self.circuit_open_time = time.time()
                logger.error("Quality circuit breaker opened due to low quality scores")
                
                # Send alert
                await self.send_quality_alert()
            
            return response, "lightrag"
            
        except Exception as e:
            logger.error(f"LightRAG error: {e}")
            return await self.perplexity_fallback(query), "lightrag_error"
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should open based on quality scores"""
        
        if len(self.quality_scores) < self.quality_window:
            return False
        
        avg_quality = sum(self.quality_scores) / len(self.quality_scores)
        return avg_quality < self.quality_threshold
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        
        if not self.circuit_open_time:
            return False
        
        return time.time() - self.circuit_open_time > self.recovery_timeout
```

### 3. Budget-Based Circuit Breaker

```python
class BudgetBasedCircuitBreaker:
    """Circuit breaker that opens when budget limits are exceeded"""
    
    def __init__(self, daily_budget: float, alert_threshold: float = 0.8):
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.current_spend = 0.0
        self.circuit_open = False
        self.last_reset = datetime.now().date()
        
    async def protected_query_with_budget(self, query: str) -> tuple:
        """Query with budget-based circuit breaker"""
        
        # Reset daily budget if needed
        self._reset_daily_budget_if_needed()
        
        # Check if budget exceeded
        if self.circuit_open or self.current_spend >= self.daily_budget:
            if not self.circuit_open:
                self.circuit_open = True
                await self.send_budget_exceeded_alert()
            
            logger.warning("Budget circuit breaker open, using free fallback")
            return await self.perplexity_fallback(query), "budget_exceeded"
        
        # Estimate query cost
        estimated_cost = await self.estimate_query_cost(query)
        
        # Check if this query would exceed budget
        if self.current_spend + estimated_cost > self.daily_budget:
            self.circuit_open = True
            await self.send_budget_exceeded_alert()
            return await self.perplexity_fallback(query), "budget_would_exceed"
        
        try:
            # Process with LightRAG
            response = await self.lightrag_system.query(query)
            
            # Track actual cost
            actual_cost = await self.calculate_actual_cost(query, response)
            self.current_spend += actual_cost
            
            # Send alert if approaching budget limit
            if self.current_spend / self.daily_budget > self.alert_threshold:
                await self.send_budget_alert()
            
            return response, "lightrag"
            
        except Exception as e:
            logger.error(f"LightRAG error: {e}")
            return await self.perplexity_fallback(query), "lightrag_error"
    
    def _reset_daily_budget_if_needed(self):
        """Reset daily budget if it's a new day"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.current_spend = 0.0
            self.circuit_open = False
            self.last_reset = today
            logger.info("Daily budget reset")
```

---

## A/B Testing Framework

### 1. Statistical A/B Testing

```python
from lightrag_integration import StatisticalABTest

class StatisticalABTesting:
    """Statistically rigorous A/B testing framework"""
    
    def __init__(self):
        self.ab_test = StatisticalABTest(
            name="lightrag_vs_perplexity_quality",
            control_group_ratio=0.5,
            treatment_group_ratio=0.5,
            minimum_sample_size=1000,
            statistical_power=0.8,
            significance_level=0.05
        )
        
        self.metrics_to_track = [
            'response_time',
            'quality_score',
            'user_satisfaction',
            'error_rate'
        ]
        
    async def process_ab_test_query(self, query: str, user_id: str):
        """Process query as part of A/B test"""
        
        # Get user assignment (consistent for same user)
        assignment = self.ab_test.get_assignment(user_id)
        
        start_time = time.time()
        
        if assignment.group == "treatment":
            # Use LightRAG
            try:
                response = await self.lightrag_system.query(query)
                source = "lightrag"
                error_occurred = False
                quality_score = await self.assess_quality(query, response)
            except Exception as e:
                response = await self.perplexity_fallback(query)
                source = "lightrag_fallback"
                error_occurred = True
                quality_score = None
                
        else:  # control group
            # Use Perplexity
            response = await self.perplexity_system.query(query)
            source = "perplexity"
            error_occurred = False
            quality_score = None  # We don't assess Perplexity quality the same way
        
        processing_time = time.time() - start_time
        
        # Track metrics for statistical analysis
        await self.ab_test.track_metrics(
            user_id=user_id,
            assignment=assignment,
            metrics={
                'response_time': processing_time,
                'quality_score': quality_score,
                'error_occurred': error_occurred,
                'response_length': len(response)
            }
        )
        
        return {
            'response': response,
            'source': source,
            'group': assignment.group,
            'processing_time': processing_time,
            'quality_score': quality_score
        }
    
    async def get_ab_test_analysis(self):
        """Get statistical analysis of A/B test results"""
        
        data = await self.ab_test.get_data()
        
        if len(data) < self.ab_test.minimum_sample_size:
            return {
                'status': 'insufficient_data',
                'current_sample_size': len(data),
                'required_sample_size': self.ab_test.minimum_sample_size
            }
        
        # Statistical analysis
        analysis = {}
        
        for metric in self.metrics_to_track:
            if metric in data.columns:
                metric_analysis = await self.ab_test.analyze_metric(metric, data)
                analysis[metric] = metric_analysis
        
        # Overall recommendation
        recommendation = self._generate_recommendation(analysis)
        
        return {
            'status': 'analysis_complete',
            'sample_size': len(data),
            'analysis': analysis,
            'recommendation': recommendation
        }
    
    def _generate_recommendation(self, analysis):
        """Generate recommendation based on statistical analysis"""
        
        significant_improvements = 0
        significant_degradations = 0
        
        for metric, metric_analysis in analysis.items():
            if metric_analysis['p_value'] < 0.05:  # Statistically significant
                if metric_analysis['treatment_better']:
                    significant_improvements += 1
                else:
                    significant_degradations += 1
        
        if significant_improvements > significant_degradations:
            return {
                'action': 'deploy_lightrag',
                'confidence': 'high' if significant_improvements >= 3 else 'medium',
                'reason': f'{significant_improvements} significant improvements vs {significant_degradations} degradations'
            }
        elif significant_degradations > significant_improvements:
            return {
                'action': 'keep_perplexity',
                'confidence': 'high',
                'reason': f'{significant_degradations} significant degradations vs {significant_improvements} improvements'
            }
        else:
            return {
                'action': 'continue_testing',
                'confidence': 'low',
                'reason': 'No clear statistical winner, continue testing'
            }
```

### 2. Multi-Variate Testing

```python
class MultiVariateABTesting:
    """Multi-variate A/B testing for different LightRAG configurations"""
    
    def __init__(self):
        self.variants = {
            'control': {
                'system': 'perplexity',
                'config': None
            },
            'lightrag_conservative': {
                'system': 'lightrag',
                'config': {
                    'mode': 'local',
                    'max_tokens': 4000,
                    'temperature': 0.3
                }
            },
            'lightrag_balanced': {
                'system': 'lightrag',
                'config': {
                    'mode': 'hybrid',
                    'max_tokens': 6000,
                    'temperature': 0.5
                }
            },
            'lightrag_aggressive': {
                'system': 'lightrag',
                'config': {
                    'mode': 'global',
                    'max_tokens': 8000,
                    'temperature': 0.7
                }
            }
        }
        
        self.traffic_allocation = {
            'control': 0.4,              # 40% control
            'lightrag_conservative': 0.2, # 20% conservative
            'lightrag_balanced': 0.3,     # 30% balanced
            'lightrag_aggressive': 0.1    # 10% aggressive
        }
        
    def get_variant_assignment(self, user_id: str) -> dict:
        """Get variant assignment for user"""
        
        user_hash = hash(user_id) % 100
        cumulative_percentage = 0
        
        for variant, percentage in self.traffic_allocation.items():
            cumulative_percentage += percentage * 100
            if user_hash < cumulative_percentage:
                return {
                    'variant': variant,
                    'config': self.variants[variant]
                }
        
        # Default fallback
        return {
            'variant': 'control',
            'config': self.variants['control']
        }
    
    async def process_multivariate_query(self, query: str, user_id: str):
        """Process query with multivariate testing"""
        
        assignment = self.get_variant_assignment(user_id)
        variant = assignment['variant']
        config = assignment['config']
        
        if config['system'] == 'perplexity':
            response = await self.perplexity_system.query(query)
            processing_details = {'source': 'perplexity'}
        else:
            # Use LightRAG with specific configuration
            response = await self.lightrag_system.query(
                query,
                mode=config['config']['mode'],
                max_tokens=config['config']['max_tokens'],
                temperature=config['config']['temperature']
            )
            processing_details = {
                'source': 'lightrag',
                'mode': config['config']['mode']
            }
        
        # Track variant performance
        await self.track_variant_metrics(
            user_id=user_id,
            variant=variant,
            query=query,
            response=response,
            processing_details=processing_details
        )
        
        return {
            'response': response,
            'variant': variant,
            **processing_details
        }
```

This comprehensive guide covers all the major integration patterns and feature flag strategies available in the LightRAG system. Each pattern is designed to minimize risk while maximizing the ability to safely deploy and test the new RAG capabilities.