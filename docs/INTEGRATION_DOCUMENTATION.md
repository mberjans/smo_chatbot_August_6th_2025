# Clinical Metabolomics Oracle - LightRAG Integration Documentation

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Integration Patterns and Examples](#integration-patterns-and-examples)
3. [Feature Flag Integration Guide](#feature-flag-integration-guide)
4. [Configuration Management](#configuration-management)
5. [Quality Assurance Integration](#quality-assurance-integration)
6. [API Integration Examples](#api-integration-examples)
7. [Production Deployment Guide](#production-deployment-guide)
8. [Migration Strategies](#migration-strategies)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## System Architecture Overview

### High-Level Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CMO Chatbot Frontend                     │
│                     (Chainlit UI)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Feature Flag Manager                          │
│          (FeatureFlagManager + RolloutManager)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Integration Router                           │
│         (Routes between LightRAG and Perplexity)          │
└─────────┬───────────────────────────────────────┬─────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│   LightRAG System   │                 │  Perplexity API     │
│                     │                 │   (Existing)        │
│ • Knowledge Graph   │                 │                     │
│ • PDF Processing    │                 │ • Real-time Data    │
│ • Local RAG         │                 │ • Web Search        │
└─────────────────────┘                 └─────────────────────┘
          │                                       │
          └─────────┬───────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Response Processing Layer                      │
│    • Citation Processing  • Quality Scoring               │
│    • Translation Support  • Cost Tracking                 │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **User Query Reception**: Chainlit interface receives user query
2. **Feature Flag Evaluation**: FeatureFlagManager determines routing strategy
3. **Intelligent Routing**: Based on query type and user cohort
4. **Response Generation**: Either LightRAG or Perplexity API
5. **Quality Assessment**: Real-time quality scoring and validation
6. **Response Enhancement**: Citation processing, translation, formatting
7. **Cost Tracking**: API usage monitoring and budget management
8. **User Response**: Final formatted response delivered to user

### Data Flow Diagram

```
Query → Feature Flags → Routing Decision → Response Generation → Quality Check → Cost Tracking → Final Response
  │                                                                     ↓                    ↑
  └── User Context ──── System Health ──── Budget Status ──── Quality Metrics ──── Performance Data
```

---

## Integration Patterns and Examples

### 1. Basic Integration Pattern

The simplest integration replaces Perplexity API calls with LightRAG:

```python
from lightrag_integration import ClinicalMetabolomicsRAG, LightRAGConfig

# Basic integration in existing main.py
async def initialize_lightrag():
    """Initialize LightRAG system"""
    config = LightRAGConfig()
    rag_system = ClinicalMetabolomicsRAG(config)
    await rag_system.initialize_knowledge_base()
    return rag_system

# Global LightRAG instance
lightrag_system = None

@cl.on_chat_start
async def start():
    global lightrag_system
    if not lightrag_system:
        lightrag_system = await initialize_lightrag()
    
    cl.user_session.set("lightrag_system", lightrag_system)

@cl.on_message
async def on_message(message: cl.Message):
    """Enhanced message handler with LightRAG"""
    rag_system = cl.user_session.get("lightrag_system")
    
    # Use LightRAG instead of Perplexity
    response = await rag_system.query(
        message.content,
        mode="hybrid"  # Uses both local and global knowledge
    )
    
    # Continue with existing citation and translation processing
    await cl.Message(content=response).send()
```

### 2. Advanced Hybrid Integration with Feature Flags

Production-ready integration with intelligent routing:

```python
from lightrag_integration import (
    FeatureFlagManager, 
    ClinicalMetabolomicsRAG,
    create_clinical_rag_system
)

class HybridCMOSystem:
    """Advanced hybrid system with intelligent routing"""
    
    def __init__(self):
        self.feature_flags = FeatureFlagManager()
        self.lightrag_system = None
        self.perplexity_system = None  # Existing system
        
    async def initialize(self):
        """Initialize both systems"""
        self.lightrag_system = await create_clinical_rag_system()
        # Existing Perplexity initialization...
        
    async def process_query(self, query: str, user_id: str) -> str:
        """Intelligent query routing"""
        
        # Get routing decision from feature flags
        routing_decision = await self.feature_flags.get_routing_decision(
            user_id=user_id,
            query=query,
            context={
                'system_health': await self.check_system_health(),
                'budget_status': await self.check_budget_status()
            }
        )
        
        try:
            if routing_decision.use_lightrag:
                response = await self.lightrag_system.query(query, mode="hybrid")
                source = "lightrag"
            else:
                response = await self.perplexity_query(query)  # Existing
                source = "perplexity"
                
            # Apply quality assessment and enhancement
            enhanced_response = await self.enhance_response(response, source)
            return enhanced_response
            
        except Exception as e:
            # Fallback to alternative system
            return await self.fallback_query(query, routing_decision)
```

### 3. Production Deployment Pattern

Complete production-ready integration:

```python
from lightrag_integration import (
    create_production_rag_system,
    ProductionConfig,
    QualityAssessmentSuite,
    BudgetManager,
    AlertSystem
)

class ProductionCMOSystem:
    """Production-ready CMO system with full monitoring"""
    
    def __init__(self):
        self.config = ProductionConfig()
        self.rag_system = None
        self.quality_suite = QualityAssessmentSuite()
        self.budget_manager = BudgetManager()
        self.alert_system = AlertSystem()
        
    async def initialize(self):
        """Production initialization with full monitoring"""
        
        # Initialize with production configuration
        self.rag_system = await create_production_rag_system(
            config=self.config,
            monitoring_enabled=True,
            cost_tracking_enabled=True,
            quality_validation_enabled=True
        )
        
        # Setup monitoring and alerting
        await self.alert_system.initialize()
        
    async def process_query_with_monitoring(self, query: str, user_id: str):
        """Complete query processing with monitoring"""
        
        start_time = time.time()
        
        try:
            # Pre-query budget check
            if not await self.budget_manager.can_process_query():
                return await self.handle_budget_exceeded()
            
            # Process query
            response = await self.rag_system.query(query, mode="hybrid")
            
            # Quality assessment
            quality_score = await self.quality_suite.assess_response(
                query=query,
                response=response,
                source_documents=self.rag_system.get_last_sources()
            )
            
            # Cost tracking
            await self.budget_manager.track_usage(
                query=query,
                response=response,
                processing_time=time.time() - start_time
            )
            
            # Performance monitoring
            await self.monitor_performance(
                query_time=time.time() - start_time,
                quality_score=quality_score,
                user_id=user_id
            )
            
            return {
                'response': response,
                'quality_score': quality_score,
                'processing_time': time.time() - start_time,
                'source': 'lightrag'
            }
            
        except Exception as e:
            await self.alert_system.send_error_alert(e, query, user_id)
            return await self.fallback_response(query)
```

---

## Feature Flag Integration Guide

### FeatureFlagManager Configuration

```python
from lightrag_integration import FeatureFlagManager, RolloutConfig

# Initialize feature flag manager
feature_flags = FeatureFlagManager(
    rollout_config=RolloutConfig(
        initial_percentage=10,      # Start with 10% of users
        quality_gate_threshold=0.8, # Require 80% quality score
        circuit_breaker_enabled=True,
        max_daily_budget=100.0
    )
)

# Environment-based configuration
feature_flags.configure_from_env({
    'LIGHTRAG_ROLLOUT_PERCENTAGE': '25',  # Current rollout percentage
    'LIGHTRAG_QUALITY_THRESHOLD': '0.85',
    'LIGHTRAG_CIRCUIT_BREAKER': 'true',
    'LIGHTRAG_MAX_BUDGET': '150.0'
})
```

### Gradual Rollout Strategy

```python
class GradualRolloutManager:
    """Manages gradual rollout from 10% to 100%"""
    
    def __init__(self):
        self.rollout_stages = [10, 25, 50, 75, 100]
        self.current_stage = 0
        self.validation_period_hours = 24
        
    async def advance_rollout(self):
        """Advance to next rollout stage if quality gates pass"""
        
        # Check quality metrics from last 24 hours
        quality_metrics = await self.get_quality_metrics(
            hours_back=self.validation_period_hours
        )
        
        if self.should_advance_rollout(quality_metrics):
            self.current_stage += 1
            new_percentage = self.rollout_stages[self.current_stage]
            
            await self.feature_flags.update_rollout_percentage(new_percentage)
            await self.notify_team(f"Rollout advanced to {new_percentage}%")
            
        else:
            await self.handle_rollout_pause(quality_metrics)
    
    def should_advance_rollout(self, metrics):
        """Quality gate validation"""
        return (
            metrics['average_quality'] > 0.8 and
            metrics['error_rate'] < 0.05 and
            metrics['user_satisfaction'] > 0.85
        )
```

### A/B Testing Configuration

```python
from lightrag_integration import ABTestManager

ab_test = ABTestManager(
    test_name="lightrag_vs_perplexity",
    control_group_size=0.5,  # 50% control (Perplexity)
    treatment_group_size=0.5,  # 50% treatment (LightRAG)
    minimum_sample_size=1000,
    statistical_power=0.8
)

async def run_ab_test(query: str, user_id: str):
    """Run A/B test between systems"""
    
    assignment = ab_test.get_user_assignment(user_id)
    
    if assignment.group == "treatment":
        response = await lightrag_system.query(query)
        source = "lightrag"
    else:
        response = await perplexity_query(query)
        source = "perplexity"
    
    # Track metrics for statistical analysis
    await ab_test.track_metrics(
        user_id=user_id,
        query=query,
        response=response,
        source=source,
        metrics={
            'response_time': response.processing_time,
            'quality_score': response.quality_score,
            'user_rating': None  # To be filled by user feedback
        }
    )
    
    return response
```

---

## Configuration Management

### Environment Variable Setup

Create `.env` file with required configurations:

```bash
# Core LightRAG Configuration
OPENAI_API_KEY=your_openai_api_key_here
LIGHTRAG_WORKING_DIR=./lightrag_storage
LIGHTRAG_PAPERS_DIR=./papers

# Feature Flag Configuration
LIGHTRAG_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=25
LIGHTRAG_QUALITY_THRESHOLD=0.8
LIGHTRAG_CIRCUIT_BREAKER=true

# Budget Management
LIGHTRAG_DAILY_BUDGET=100.0
LIGHTRAG_MONTHLY_BUDGET=2500.0
LIGHTRAG_COST_ALERT_THRESHOLD=0.8

# Quality Validation
ENABLE_QUALITY_VALIDATION=true
QUALITY_RELEVANCE_THRESHOLD=0.75
QUALITY_ACCURACY_THRESHOLD=0.8

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING=true
PERFORMANCE_LOG_LEVEL=INFO
METRICS_COLLECTION_ENABLED=true

# Database Configuration (for cost tracking)
DATABASE_URL=sqlite:///cost_tracking.db

# Alert Configuration
SLACK_WEBHOOK_URL=your_slack_webhook
EMAIL_ALERTS_ENABLED=true
ALERT_EMAIL=admin@yourorg.com
```

### Multi-Environment Configuration

```python
from lightrag_integration import EnvironmentConfig

class ConfigManager:
    """Manages configuration across environments"""
    
    @staticmethod
    def get_config(environment: str = None):
        """Get environment-specific configuration"""
        
        env = environment or os.getenv('ENVIRONMENT', 'development')
        
        if env == 'production':
            return ProductionConfig(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                working_dir='/opt/lightrag/storage',
                papers_dir='/opt/lightrag/papers',
                max_daily_budget=500.0,
                quality_threshold=0.85,
                circuit_breaker_enabled=True,
                monitoring_enabled=True,
                alert_enabled=True
            )
            
        elif env == 'staging':
            return StagingConfig(
                openai_api_key=os.getenv('STAGING_OPENAI_API_KEY'),
                working_dir='./staging_storage',
                papers_dir='./staging_papers',
                max_daily_budget=100.0,
                quality_threshold=0.8,
                circuit_breaker_enabled=True,
                monitoring_enabled=True
            )
            
        else:  # development
            return DevelopmentConfig(
                openai_api_key=os.getenv('DEV_OPENAI_API_KEY'),
                working_dir='./dev_storage',
                papers_dir='./papers',
                max_daily_budget=50.0,
                quality_threshold=0.7,
                circuit_breaker_enabled=False,
                monitoring_enabled=False
            )
```

### Configuration Validation

```python
from lightrag_integration import ConfigValidator

def validate_configuration():
    """Validate all configuration settings"""
    
    validator = ConfigValidator()
    
    # Required environment variables
    required_vars = [
        'OPENAI_API_KEY',
        'LIGHTRAG_WORKING_DIR',
        'LIGHTRAG_PAPERS_DIR'
    ]
    
    validation_result = validator.validate_environment(required_vars)
    
    if not validation_result.is_valid:
        raise ConfigurationError(
            f"Missing required configuration: {validation_result.missing_vars}"
        )
    
    # Validate API connectivity
    api_validation = await validator.validate_openai_connectivity()
    if not api_validation.is_valid:
        raise ConfigurationError(
            f"OpenAI API validation failed: {api_validation.error}"
        )
    
    # Validate directory permissions
    dir_validation = validator.validate_directory_permissions()
    if not dir_validation.is_valid:
        raise ConfigurationError(
            f"Directory permission validation failed: {dir_validation.errors}"
        )
    
    return True
```

---

## Quality Assurance Integration

### Quality Validation Setup

```python
from lightrag_integration import (
    QualityAssessmentSuite,
    RelevanceScorer,
    AccuracyScorer,
    FactualAccuracyValidator
)

class QualityManager:
    """Manages quality assessment and validation"""
    
    def __init__(self):
        self.relevance_scorer = RelevanceScorer()
        self.accuracy_scorer = AccuracyScorer()
        self.factual_validator = FactualAccuracyValidator()
        
    async def assess_response_quality(self, query: str, response: str, sources: list):
        """Comprehensive quality assessment"""
        
        # Relevance scoring (0.0 to 1.0)
        relevance_score = await self.relevance_scorer.score_response(
            query=query,
            response=response
        )
        
        # Accuracy scoring against source documents
        accuracy_score = await self.accuracy_scorer.validate_against_sources(
            response=response,
            sources=sources
        )
        
        # Factual accuracy validation
        factual_validation = await self.factual_validator.validate_claims(
            response=response,
            sources=sources
        )
        
        # Composite quality score
        composite_score = self.calculate_composite_score(
            relevance=relevance_score,
            accuracy=accuracy_score,
            factual=factual_validation.confidence
        )
        
        return QualityAssessment(
            relevance_score=relevance_score,
            accuracy_score=accuracy_score,
            factual_accuracy=factual_validation.confidence,
            composite_score=composite_score,
            meets_threshold=composite_score >= 0.8,
            issues=factual_validation.issues
        )
        
    def calculate_composite_score(self, relevance: float, accuracy: float, factual: float):
        """Calculate weighted composite quality score"""
        return (relevance * 0.4) + (accuracy * 0.3) + (factual * 0.3)
```

### Response Scoring Integration

```python
async def enhanced_query_processing(query: str, user_id: str):
    """Query processing with quality scoring"""
    
    # Generate response
    response = await lightrag_system.query(query, mode="hybrid")
    
    # Get source documents used
    sources = lightrag_system.get_last_sources()
    
    # Quality assessment
    quality_assessment = await quality_manager.assess_response_quality(
        query=query,
        response=response,
        sources=sources
    )
    
    # Handle low-quality responses
    if not quality_assessment.meets_threshold:
        
        # Log quality issue
        logger.warning(f"Low quality response: {quality_assessment.composite_score}")
        
        # Fallback to Perplexity if quality too low
        if quality_assessment.composite_score < 0.6:
            response = await perplexity_fallback_query(query)
            source = "perplexity_fallback"
        else:
            source = "lightrag_low_quality"
    else:
        source = "lightrag"
    
    # Track quality metrics
    await metrics_tracker.track_quality(
        query=query,
        source=source,
        quality_score=quality_assessment.composite_score,
        user_id=user_id
    )
    
    return EnhancedResponse(
        content=response,
        quality_score=quality_assessment.composite_score,
        source=source,
        quality_details=quality_assessment
    )
```

---

## API Integration Examples

### Complete Chainlit Integration

```python
import chainlit as cl
from lightrag_integration import (
    create_clinical_rag_system,
    FeatureFlagManager,
    QualityAssessmentSuite,
    BudgetManager,
    EnhancedResponseProcessor
)

# Global system components
rag_system = None
feature_flags = None
quality_suite = None
budget_manager = None
response_processor = None

@cl.on_chat_start
async def start():
    """Initialize LightRAG-enhanced CMO system"""
    global rag_system, feature_flags, quality_suite, budget_manager, response_processor
    
    # Initialize components
    rag_system = await create_clinical_rag_system()
    feature_flags = FeatureFlagManager()
    quality_suite = QualityAssessmentSuite()
    budget_manager = BudgetManager()
    response_processor = EnhancedResponseProcessor()
    
    # Store in session
    cl.user_session.set("rag_system", rag_system)
    cl.user_session.set("feature_flags", feature_flags)
    cl.user_session.set("user_id", generate_user_id())
    
    await cl.Message(
        content="🧬 Clinical Metabolomics Oracle with LightRAG is ready! "
               "I can help you with biomedical research questions using our enhanced knowledge base."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Enhanced message processing with LightRAG"""
    
    # Get session components
    rag_system = cl.user_session.get("rag_system")
    feature_flags = cl.user_session.get("feature_flags")
    user_id = cl.user_session.get("user_id")
    
    query = message.content.strip()
    
    try:
        # Show processing indicator
        processing_msg = await cl.Message(content="🔬 Processing your query...").send()
        
        # Check feature flags and routing
        routing_decision = await feature_flags.get_routing_decision(
            user_id=user_id,
            query=query,
            context={'budget_status': await budget_manager.get_status()}
        )
        
        if routing_decision.use_lightrag:
            # Use LightRAG system
            response = await rag_system.query(query, mode="hybrid")
            
            # Quality assessment
            quality_score = await quality_suite.assess_response(
                query=query,
                response=response,
                sources=rag_system.get_last_sources()
            )
            
            source_indicator = "🧠 LightRAG Knowledge Graph"
            
        else:
            # Fallback to existing Perplexity system
            response = await perplexity_query(query)  # Existing function
            quality_score = None
            source_indicator = "🌐 Perplexity Search"
        
        # Process and enhance response
        enhanced_response = await response_processor.process(
            response=response,
            query=query,
            quality_score=quality_score,
            include_citations=True,
            include_translation=detect_language(query) != 'en'
        )
        
        # Update processing message with final response
        await processing_msg.update(content=enhanced_response.formatted_content)
        
        # Add quality indicator if available
        if quality_score:
            quality_indicator = get_quality_indicator(quality_score.composite_score)
            await cl.Message(
                content=f"{source_indicator} | Quality: {quality_indicator}",
                author="System"
            ).send()
        
        # Track usage for budget and analytics
        await budget_manager.track_query(
            query=query,
            response=response,
            user_id=user_id,
            source="lightrag" if routing_decision.use_lightrag else "perplexity"
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        await cl.Message(
            content="I apologize, but I encountered an error processing your query. "
                   "Please try again or rephrase your question."
        ).send()

def get_quality_indicator(score: float) -> str:
    """Get quality indicator emoji"""
    if score >= 0.9:
        return "🌟 Excellent"
    elif score >= 0.8:
        return "✅ Good"
    elif score >= 0.7:
        return "⚡ Acceptable"
    else:
        return "⚠️ Low Quality"
```

### Async Integration Pattern

```python
import asyncio
from lightrag_integration import AsyncClinicalRAG

class AsyncCMOIntegration:
    """Async-first integration pattern"""
    
    def __init__(self):
        self.rag_system = None
        self.processing_queue = asyncio.Queue()
        self.response_cache = {}
        
    async def initialize(self):
        """Initialize async components"""
        self.rag_system = AsyncClinicalRAG()
        await self.rag_system.initialize()
        
        # Start background processing
        asyncio.create_task(self.process_queue())
        
    async def process_query_async(self, query: str, user_id: str) -> str:
        """Non-blocking query processing"""
        
        # Check cache first
        cache_key = hash(query)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Add to processing queue
        future = asyncio.Future()
        await self.processing_queue.put({
            'query': query,
            'user_id': user_id,
            'future': future
        })
        
        # Wait for result
        result = await future
        
        # Cache result
        self.response_cache[cache_key] = result
        return result
        
    async def process_queue(self):
        """Background queue processor"""
        while True:
            try:
                item = await self.processing_queue.get()
                
                result = await self.rag_system.query(
                    item['query'],
                    mode="hybrid"
                )
                
                item['future'].set_result(result)
                
            except Exception as e:
                item['future'].set_exception(e)
```

---

## Production Deployment Guide

### Step-by-Step Deployment Process

#### 1. Pre-Deployment Validation

```bash
# 1. Validate environment configuration
python -c "from lightrag_integration import validate_configuration; validate_configuration()"

# 2. Run comprehensive tests
pytest lightrag_integration/tests/ -v --cov=lightrag_integration

# 3. Validate knowledge base
python -c "
from lightrag_integration import test_knowledge_base_integrity
test_knowledge_base_integrity()
"

# 4. Check API connectivity
python -c "
from lightrag_integration import test_openai_connectivity
test_openai_connectivity()
"
```

#### 2. Staged Deployment

```yaml
# deployment.yml
version: '3.8'
services:
  cmo-lightrag:
    image: cmo-lightrag:latest
    environment:
      - ENVIRONMENT=production
      - LIGHTRAG_ROLLOUT_PERCENTAGE=10  # Start with 10%
      - LIGHTRAG_QUALITY_THRESHOLD=0.85
      - LIGHTRAG_CIRCUIT_BREAKER=true
      - LIGHTRAG_MAX_DAILY_BUDGET=500
    volumes:
      - ./papers:/opt/lightrag/papers
      - ./storage:/opt/lightrag/storage
      - ./config:/opt/lightrag/config
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python", "-c", "from lightrag_integration import health_check; health_check()"]
      interval: 30s
      timeout: 10s
      retries: 3
```

#### 3. Monitoring Setup

```python
from lightrag_integration import ProductionMonitor

monitor = ProductionMonitor(
    alert_webhooks=[
        "https://hooks.slack.com/your-webhook",
        "https://your-monitoring-system.com/webhook"
    ],
    metrics_endpoints=[
        "https://your-metrics-system.com/api/metrics"
    ]
)

# Monitor key metrics
@monitor.track_metric("response_time")
@monitor.track_metric("quality_score")
@monitor.track_metric("error_rate")
async def monitored_query_processing(query: str):
    """Query processing with monitoring"""
    return await rag_system.query(query)

# Set up alerts
monitor.add_alert(
    metric="error_rate",
    threshold=0.05,  # 5% error rate
    duration_minutes=10,
    severity="critical"
)

monitor.add_alert(
    metric="quality_score",
    threshold=0.8,
    operator="less_than",
    duration_minutes=15,
    severity="warning"
)
```

### Performance Optimization

#### 1. Caching Configuration

```python
from lightrag_integration import ResponseCache, CacheConfig

cache = ResponseCache(
    config=CacheConfig(
        backend="redis",  # or "memory", "database"
        ttl_seconds=3600,  # 1 hour cache
        max_entries=10000,
        compression=True
    )
)

async def cached_query_processing(query: str):
    """Query processing with caching"""
    
    # Check cache first
    cached_response = await cache.get(query)
    if cached_response:
        return cached_response
    
    # Generate new response
    response = await rag_system.query(query)
    
    # Cache the response
    await cache.set(query, response)
    
    return response
```

#### 2. Connection Pooling

```python
from lightrag_integration import ConnectionPool

# Configure connection pools
openai_pool = ConnectionPool(
    service="openai",
    max_connections=50,
    timeout_seconds=30,
    retry_attempts=3
)

async def optimized_query(query: str):
    """Query with connection pooling"""
    async with openai_pool.get_connection() as conn:
        return await rag_system.query(query, connection=conn)
```

### Scaling Considerations

#### 1. Horizontal Scaling

```python
from lightrag_integration import DistributedRAGSystem

distributed_rag = DistributedRAGSystem(
    nodes=[
        "http://rag-node-1:8000",
        "http://rag-node-2:8000", 
        "http://rag-node-3:8000"
    ],
    load_balancer="round_robin",  # or "least_connections", "weighted"
    health_check_interval=30
)

async def distributed_query(query: str):
    """Query processing across multiple nodes"""
    return await distributed_rag.query(query)
```

#### 2. Auto-scaling Configuration

```python
from lightrag_integration import AutoScaler

auto_scaler = AutoScaler(
    min_instances=2,
    max_instances=10,
    target_cpu_utilization=70,
    target_response_time_ms=5000,
    scale_up_cooldown_minutes=5,
    scale_down_cooldown_minutes=15
)

# Auto-scaling triggers
auto_scaler.add_scaling_rule(
    metric="queue_length",
    scale_up_threshold=100,
    scale_down_threshold=10
)
```

---

## Migration Strategies

### Safe Migration from Perplexity to LightRAG

#### 1. Phase 1: Shadow Mode (0% Traffic)

```python
class ShadowModeIntegration:
    """Run LightRAG in shadow mode for validation"""
    
    async def shadow_mode_query(self, query: str, user_id: str):
        """Process query with both systems, return only Perplexity"""
        
        # Primary: Perplexity (user sees this)
        perplexity_response = await self.perplexity_query(query)
        
        # Shadow: LightRAG (for comparison only)
        try:
            lightrag_response = await self.lightrag_query(query)
            
            # Compare responses (async, don't block user)
            asyncio.create_task(
                self.compare_responses(
                    query, perplexity_response, lightrag_response
                )
            )
        except Exception as e:
            logger.warning(f"Shadow mode LightRAG error: {e}")
        
        return perplexity_response  # User only sees Perplexity
```

#### 2. Phase 2: Canary Deployment (5-10% Traffic)

```python
from lightrag_integration import CanaryDeployment

canary = CanaryDeployment(
    canary_percentage=5,  # 5% of users get LightRAG
    success_criteria={
        'error_rate': {'max': 0.02},  # Max 2% errors
        'response_time': {'p95': 5000},  # 95% under 5 seconds
        'quality_score': {'min': 0.8}   # Min 80% quality
    },
    duration_hours=24,  # Run canary for 24 hours
    auto_promote=True   # Auto-promote if criteria met
)

async def canary_query_processing(query: str, user_id: str):
    """Query processing with canary deployment"""
    
    if canary.should_use_canary(user_id):
        try:
            response = await lightrag_system.query(query)
            
            # Track canary metrics
            await canary.track_success(
                user_id=user_id,
                response_time=response.processing_time,
                quality_score=response.quality_score
            )
            
            return response
            
        except Exception as e:
            await canary.track_failure(user_id, str(e))
            # Fallback to Perplexity
            return await perplexity_query(query)
    
    else:
        return await perplexity_query(query)
```

#### 3. Phase 3: Gradual Rollout (10% → 100%)

```python
from lightrag_integration import GradualRolloutManager

rollout = GradualRolloutManager(
    stages=[10, 25, 50, 75, 100],
    stage_duration_hours=48,  # 48 hours per stage
    quality_gate_threshold=0.85,
    auto_rollback_enabled=True
)

async def gradual_rollout_processing(query: str, user_id: str):
    """Processing with gradual rollout"""
    
    current_percentage = rollout.get_current_percentage()
    
    if rollout.should_use_lightrag(user_id):
        try:
            response = await lightrag_system.query(query)
            
            # Track rollout success
            await rollout.track_metrics(
                user_id=user_id,
                quality_score=response.quality_score,
                error_occurred=False
            )
            
            return response
            
        except Exception as e:
            # Track rollout failure
            await rollout.track_metrics(
                user_id=user_id,
                error_occurred=True,
                error_message=str(e)
            )
            
            # Check if should rollback
            if await rollout.should_rollback():
                await rollout.execute_rollback()
            
            # Fallback to Perplexity
            return await perplexity_query(query)
    
    else:
        return await perplexity_query(query)
```

### Validation and Testing Procedures

#### 1. Response Quality Validation

```python
from lightrag_integration import ResponseValidator, ValidationSuite

validator = ResponseValidator(
    test_queries=[
        "What is clinical metabolomics?",
        "How are biomarkers used in metabolic disease diagnosis?",
        "What are the latest advances in metabolomics research?",
        "Explain the role of mass spectrometry in metabolomics"
    ]
)

async def run_migration_validation():
    """Validate migration readiness"""
    
    validation_results = []
    
    for query in validator.test_queries:
        # Get responses from both systems
        perplexity_response = await perplexity_query(query)
        lightrag_response = await lightrag_system.query(query)
        
        # Compare quality
        comparison = await validator.compare_responses(
            query=query,
            response_a=perplexity_response,
            response_b=lightrag_response
        )
        
        validation_results.append(comparison)
    
    # Generate validation report
    report = validator.generate_migration_report(validation_results)
    
    if report.migration_recommended:
        logger.info("Migration validation passed")
        return True
    else:
        logger.warning(f"Migration validation failed: {report.issues}")
        return False
```

### Rollback Procedures

#### 1. Automatic Rollback

```python
from lightrag_integration import AutoRollback

auto_rollback = AutoRollback(
    triggers=[
        {'metric': 'error_rate', 'threshold': 0.1, 'duration_minutes': 5},
        {'metric': 'quality_score', 'threshold': 0.7, 'duration_minutes': 10},
        {'metric': 'response_time', 'threshold': 10000, 'duration_minutes': 5}
    ],
    rollback_target_percentage=0,  # Rollback to 0% LightRAG
    notification_channels=['slack', 'email']
)

async def monitor_and_rollback():
    """Monitor system and auto-rollback if needed"""
    
    while True:
        current_metrics = await get_current_metrics()
        
        if auto_rollback.should_trigger_rollback(current_metrics):
            logger.critical("Auto-rollback triggered!")
            
            # Execute rollback
            await auto_rollback.execute_rollback()
            
            # Notify team
            await auto_rollback.send_notifications(
                "LightRAG system rolled back due to quality issues",
                current_metrics
            )
            
            break
            
        await asyncio.sleep(60)  # Check every minute
```

#### 2. Manual Rollback

```python
from lightrag_integration import ManualRollback

manual_rollback = ManualRollback()

async def emergency_rollback(reason: str):
    """Emergency manual rollback"""
    
    logger.critical(f"Emergency rollback initiated: {reason}")
    
    # Set rollout to 0%
    await feature_flags.set_rollout_percentage(0)
    
    # Clear caches
    await response_cache.clear_all()
    
    # Notify team
    await send_emergency_notification(
        f"EMERGENCY ROLLBACK: {reason}"
    )
    
    # Generate incident report
    await manual_rollback.generate_incident_report(reason)
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. LightRAG Initialization Failures

**Issue**: System fails to initialize LightRAG components

```python
# Troubleshooting script
from lightrag_integration import DiagnosticTools

async def diagnose_initialization_failure():
    """Diagnose LightRAG initialization issues"""
    
    diagnostics = DiagnosticTools()
    
    # Check API connectivity
    api_status = await diagnostics.check_openai_connectivity()
    if not api_status.success:
        return f"OpenAI API issue: {api_status.error}"
    
    # Check directory permissions
    dir_status = diagnostics.check_directory_permissions()
    if not dir_status.success:
        return f"Directory permission issue: {dir_status.error}"
    
    # Check knowledge base integrity
    kb_status = await diagnostics.check_knowledge_base_integrity()
    if not kb_status.success:
        return f"Knowledge base issue: {kb_status.error}"
    
    # Check memory availability
    memory_status = diagnostics.check_memory_availability()
    if not memory_status.success:
        return f"Memory issue: {memory_status.error}"
    
    return "All diagnostic checks passed"
```

**Solutions**:
- Verify OpenAI API key is valid
- Check directory permissions for storage and papers directories
- Ensure sufficient memory is available (minimum 4GB recommended)
- Verify network connectivity for API calls

#### 2. Poor Response Quality

**Issue**: LightRAG responses have low quality scores

```python
async def diagnose_quality_issues():
    """Diagnose quality issues"""
    
    # Check knowledge base coverage
    coverage_report = await quality_diagnostics.analyze_knowledge_coverage()
    
    if coverage_report.coverage_percentage < 0.7:
        return "Insufficient knowledge base coverage. Add more relevant PDFs."
    
    # Check query complexity
    query_complexity = await quality_diagnostics.analyze_query_patterns()
    
    if query_complexity.complexity_score > 0.8:
        return "Queries too complex for current knowledge base"
    
    # Check embedding quality
    embedding_quality = await quality_diagnostics.check_embedding_quality()
    
    if embedding_quality.score < 0.75:
        return "Poor embedding quality. Consider re-indexing knowledge base."
    
    return "Quality diagnostics completed"
```

**Solutions**:
- Add more relevant PDF documents to knowledge base
- Adjust QueryParam settings for better retrieval
- Re-initialize knowledge base if embeddings are stale
- Fine-tune quality scoring thresholds

#### 3. High API Costs

**Issue**: OpenAI API costs exceed budget

```python
from lightrag_integration import CostAnalyzer

async def analyze_cost_issues():
    """Analyze and optimize API costs"""
    
    cost_analyzer = CostAnalyzer()
    
    # Get cost breakdown
    cost_report = await cost_analyzer.generate_cost_report()
    
    recommendations = []
    
    if cost_report.embedding_costs > cost_report.llm_costs:
        recommendations.append(
            "High embedding costs: Consider caching embeddings or reducing chunk overlap"
        )
    
    if cost_report.average_query_cost > 0.10:
        recommendations.append(
            "High query costs: Optimize QueryParam settings to reduce token usage"
        )
    
    if cost_report.failed_query_ratio > 0.05:
        recommendations.append(
            "High failed query rate: Implement better error handling to avoid wasted API calls"
        )
    
    return recommendations
```

**Solutions**:
- Implement response caching to reduce duplicate API calls
- Optimize QueryParam settings to use fewer tokens
- Set up budget alerts and circuit breakers
- Consider using smaller models for simpler queries

#### 4. Performance Issues

**Issue**: Slow response times

```python
from lightrag_integration import PerformanceProfiler

async def profile_performance():
    """Profile system performance"""
    
    profiler = PerformanceProfiler()
    
    # Profile query processing pipeline
    profile_result = await profiler.profile_query_pipeline()
    
    bottlenecks = []
    
    if profile_result.embedding_time > 2.0:
        bottlenecks.append("Slow embedding generation")
    
    if profile_result.retrieval_time > 3.0:
        bottlenecks.append("Slow document retrieval")
    
    if profile_result.llm_time > 10.0:
        bottlenecks.append("Slow LLM response generation")
    
    return {
        'bottlenecks': bottlenecks,
        'optimization_suggestions': profiler.get_optimization_suggestions()
    }
```

**Solutions**:
- Implement connection pooling for API calls
- Add response caching for frequently asked questions
- Optimize chunk size and overlap parameters
- Consider using faster embedding models
- Implement async processing for non-blocking operations

#### 5. Feature Flag Issues

**Issue**: Feature flags not working correctly

```python
from lightrag_integration import FeatureFlagDiagnostics

def diagnose_feature_flags():
    """Diagnose feature flag issues"""
    
    diagnostics = FeatureFlagDiagnostics()
    
    # Check configuration
    config_issues = diagnostics.validate_feature_flag_config()
    
    # Check user assignment consistency
    assignment_issues = diagnostics.check_user_assignment_consistency()
    
    # Check rollout percentage application
    rollout_issues = diagnostics.validate_rollout_percentage()
    
    return {
        'config_issues': config_issues,
        'assignment_issues': assignment_issues,
        'rollout_issues': rollout_issues
    }
```

**Solutions**:
- Verify environment variables are set correctly
- Check that user ID hashing is consistent
- Ensure rollout percentages are within valid ranges (0-100)
- Verify circuit breaker settings are appropriate

### Emergency Procedures

#### 1. Complete System Shutdown

```python
async def emergency_shutdown():
    """Emergency system shutdown procedure"""
    
    logger.critical("EMERGENCY SHUTDOWN INITIATED")
    
    # Set feature flags to disable LightRAG
    await feature_flags.emergency_disable()
    
    # Stop all processing
    await rag_system.stop_all_processing()
    
    # Clear all caches
    await response_cache.emergency_clear()
    
    # Send notifications
    await emergency_notifications.send_all(
        "LightRAG system emergency shutdown completed"
    )
```

#### 2. Health Check Endpoints

```python
from fastapi import FastAPI
from lightrag_integration import HealthChecker

app = FastAPI()
health_checker = HealthChecker()

@app.get("/health")
async def health_check():
    """System health check endpoint"""
    
    health_status = await health_checker.comprehensive_health_check()
    
    return {
        'status': 'healthy' if health_status.is_healthy else 'unhealthy',
        'components': health_status.component_status,
        'timestamp': health_status.timestamp,
        'details': health_status.details
    }

@app.get("/health/lightrag")
async def lightrag_health_check():
    """LightRAG-specific health check"""
    
    return await health_checker.check_lightrag_health()
```

This comprehensive integration documentation provides everything needed to successfully integrate and deploy the LightRAG system with the Clinical Metabolomics Oracle chatbot. The documentation covers all aspects from basic integration to production deployment, with detailed examples and troubleshooting guidance.