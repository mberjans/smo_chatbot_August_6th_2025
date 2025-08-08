# LightRAG Integration Procedures
## Comprehensive Integration Documentation for Clinical Metabolomics Oracle

**Document Version:** 1.0  
**Last Updated:** 2025-08-08  
**Author:** Claude Code (Anthropic)

---

## Table of Contents

1. [Integration Architecture Overview](#1-integration-architecture-overview)
2. [Technical Integration Steps](#2-technical-integration-steps)
3. [Integration Points Documentation](#3-integration-points-documentation)
4. [Production Deployment Strategy](#4-production-deployment-strategy)
5. [Data Flow Documentation](#5-data-flow-documentation)
6. [Configuration Management](#6-configuration-management)
7. [Monitoring and Validation](#7-monitoring-and-validation)
8. [Troubleshooting Guide](#8-troubleshooting-guide)

---

## 1. Integration Architecture Overview

### 1.1 System Architecture

The LightRAG integration uses a sophisticated routing architecture that maintains full backward compatibility while adding advanced capabilities:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     User        │───▶│   main.py       │───▶│ IntegratedQuery │
│   (Chainlit)    │    │  (Lines 177-218)│    │     Service     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │ FeatureFlagMgr  │
                                            │ (Routing Logic) │
                                            └─────────────────┘
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ▼                 ▼                 ▼
                              ┌───────────┐    ┌───────────┐    ┌───────────┐
                              │ LightRAG  │    │Perplexity │    │  Cache/   │
                              │  Service  │    │  Service  │    │ Fallback  │
                              └───────────┘    └───────────┘    └───────────┘
```

### 1.2 How LightRAG Fits Into Existing System

The integration operates through **conditional routing** at the existing Perplexity API call point:

- **Current System**: Direct Perplexity API calls in `main.py` lines 177-218
- **Integrated System**: `IntegratedQueryService` replaces direct API calls
- **Transparent Operation**: Existing Chainlit UI and user experience unchanged
- **Backward Compatibility**: Perplexity remains as fallback service

### 1.3 Routing Mechanism Between LightRAG and Perplexity

The routing system uses multiple decision factors:

1. **Feature Flags**: Enable/disable LightRAG globally or per user cohort
2. **Rollout Percentage**: Gradual deployment to percentage of users
3. **User Cohort Assignment**: Consistent A/B testing based on user hash
4. **Circuit Breaker**: Automatic fallback when service fails
5. **Quality Thresholds**: Route based on confidence scores
6. **Performance Metrics**: Route to faster/higher quality service

### 1.4 Feature Flag-Based Rollout Strategy

```python
# Environment Variables Control Rollout
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=25.0  # Start with 25%
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true
```

## 2. Technical Integration Steps

### 2.1 Environment Variable Setup

Create or update `.env` file with LightRAG configuration:

```bash
# Core LightRAG Configuration
OPENAI_API_KEY=your_openai_api_key_here
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
LIGHTRAG_WORKING_DIR=./lightrag_data
LIGHTRAG_MAX_ASYNC=16
LIGHTRAG_MAX_TOKENS=32768

# Integration Feature Flags
LIGHTRAG_INTEGRATION_ENABLED=false  # Start disabled
LIGHTRAG_ROLLOUT_PERCENTAGE=0.0     # Start at 0%
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true

# Circuit Breaker Configuration
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300

# Logging and Monitoring
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_DIR=./logs

# Budget and Cost Tracking
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_DAILY_BUDGET_LIMIT=10.0
LIGHTRAG_COST_ALERT_THRESHOLD=80.0
```

### 2.2 Database Initialization Steps

Initialize LightRAG knowledge base with biomedical content:

```python
# Initialize knowledge base (run once)
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

async def initialize_knowledge_base():
    rag = ClinicalMetabolomicsRAG()
    
    # Add your biomedical documents
    pdf_directory = "./biomedical_pdfs"
    await rag.process_pdf_directory(pdf_directory)
    
    print("Knowledge base initialization complete")

# Run initialization
import asyncio
asyncio.run(initialize_knowledge_base())
```

### 2.3 Specific Code Changes Required in main.py

Replace the existing Perplexity API call (lines 177-218) with integrated service:

#### Step 1: Import Required Modules

Add to imports at the top of `main.py`:

```python
# Add these imports after existing imports
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.integration_wrapper import (
    IntegratedQueryService, QueryRequest, ServiceResponse
)
```

#### Step 2: Initialize Integration Service

Add to `on_chat_start()` function after line 107:

```python
# Initialize LightRAG integration service
try:
    config = LightRAGConfig.from_environment()
    perplexity_api_key = os.environ["PERPLEXITY_API"]
    
    integrated_service = IntegratedQueryService(
        config=config,
        perplexity_api_key=perplexity_api_key,
        logger=logging.getLogger(__name__)
    )
    cl.user_session.set("integrated_service", integrated_service)
    
    logging.info("LightRAG integration service initialized successfully")
except Exception as e:
    logging.warning(f"LightRAG integration initialization failed: {e}")
    logging.info("Falling back to direct Perplexity API calls")
    cl.user_session.set("integrated_service", None)
```

#### Step 3: Replace Query Logic

Replace the existing query logic (lines 176-218) with:

```python
# Replace existing Perplexity API call with integrated service
integrated_service = cl.user_session.get("integrated_service")

if integrated_service:
    # Use integrated routing service
    try:
        query_request = QueryRequest(
            query_text=content,
            user_id=cl.user_session.get("user_id", "anonymous"),
            session_id=cl.user_session.get("session_id"),
            query_type="clinical_metabolomics",
            timeout_seconds=30.0
        )
        
        service_response = await integrated_service.query_async(query_request)
        
        if service_response.is_success:
            content = service_response.content
            citations = service_response.citations or []
            
            # Log routing information
            routing_info = service_response.metadata.get('routing_decision', 'unknown')
            logging.info(f"Query routed to: {routing_info}")
        else:
            # Fallback to error message
            content = "I apologize, but I'm experiencing technical difficulties. Please try again."
            citations = []
            logging.error(f"Integrated service failed: {service_response.error_details}")
    
    except Exception as e:
        logging.error(f"Integrated service error: {e}")
        content = "I apologize, but I'm experiencing technical difficulties. Please try again."
        citations = []
else:
    # Fallback to original Perplexity API logic
    url = "https://api.perplexity.ai/chat/completions"
    
    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in clinical metabolomics. You respond to"
                    "user queries in a helpful manner, with a focus on correct"
                    "scientific detail. Include peer-reviewed sources for all claims."
                    "For each source/claim, provide a confidence score from 0.0-1.0, formatted as (confidence score: X.X)"
                    "Respond in a single paragraph, never use lists unless explicitly asked."
                ),
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        "temperature": 0.1,
        "search_domain_filter": ["-wikipedia.org"],
    }
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        citations = response_data['citations']
    else:
        logging.error(f"Perplexity API error: {response.status_code}, {response.text}")
        content = ""
        citations = []
```

### 2.4 Citation Processing Integration

The integrated service provides unified citation handling. Update citation processing:

```python
# The integrated service already handles citation processing
# Keep existing bibliography formatting logic but modify source:

bibliography_dict = {}
if citations:
    counter = 1
    for citation in citations:
        bibliography_dict[str(counter)] = [citation]
        counter += 1

# Keep existing confidence score extraction and bibliography formatting
# (lines 230-258 in original main.py)
```

## 3. Integration Points Documentation

### 3.1 Exact Integration Location

**File:** `src/main.py`  
**Function:** `on_message()`  
**Lines:** 177-218 (Perplexity API call)

**Integration Method:** Replace direct API call with `IntegratedQueryService`

### 3.2 How integration_wrapper.py Works

The `integration_wrapper.py` provides the core routing logic:

```python
class IntegratedQueryService:
    """
    Main integration service that:
    1. Evaluates feature flags for routing decisions
    2. Handles LightRAG and Perplexity service calls
    3. Implements fallback logic on failures
    4. Provides circuit breaker protection
    5. Collects performance metrics
    6. Manages response caching
    """
    
    async def query_async(self, request: QueryRequest) -> ServiceResponse:
        # 1. Check cache for existing response
        # 2. Evaluate routing decision via feature flags
        # 3. Call appropriate service (LightRAG or Perplexity)
        # 4. Handle failures with fallback logic
        # 5. Record metrics and cache successful responses
        # 6. Return unified ServiceResponse
```

### 3.3 Configuration Management Through Feature Flags

Feature flags are managed through environment variables and the `FeatureFlagManager`:

```python
class FeatureFlagManager:
    def should_use_lightrag(self, context: RoutingContext) -> RoutingResult:
        # 1. Check if integration is globally enabled
        # 2. Evaluate rollout percentage
        # 3. Assign user to cohort for consistent routing
        # 4. Check circuit breaker status
        # 5. Apply conditional routing rules
        # 6. Return routing decision with reasoning
```

### 3.4 Error Handling and Fallback Mechanisms

The system implements multiple layers of error handling:

1. **Service-Level**: Individual service error handling
2. **Circuit Breaker**: Automatic service protection
3. **Fallback Logic**: LightRAG → Perplexity fallback
4. **Graceful Degradation**: User-friendly error messages

```python
# Error handling flow:
try:
    # Attempt LightRAG service
    response = await lightrag_service.query_async(request)
    if not response.is_success:
        # Fall back to Perplexity
        response = await perplexity_service.query_async(request)
except CircuitBreakerOpen:
    # Circuit breaker activated
    response = await perplexity_service.query_async(request)
except Exception as e:
    # Final fallback with user-friendly message
    response = create_error_response(e)
```

## 4. Production Deployment Strategy

### 4.1 A/B Testing Setup Using Feature Flags

#### Phase 1: Infrastructure Testing (Week 1-2)
```bash
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=1.0    # 1% of users
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true
```

#### Phase 2: Limited Rollout (Week 3-4)
```bash
LIGHTRAG_ROLLOUT_PERCENTAGE=5.0    # 5% of users
```

#### Phase 3: Expanded Testing (Week 5-8)
```bash
LIGHTRAG_ROLLOUT_PERCENTAGE=25.0   # 25% of users
```

#### Phase 4: Full Rollout (Week 9+)
```bash
LIGHTRAG_ROLLOUT_PERCENTAGE=100.0  # All users
```

### 4.2 Gradual Rollout Percentage Configuration

Use environment variables to control rollout:

```python
# Gradual rollout schedule
rollout_schedule = {
    "week_1": {"percentage": 1.0, "monitoring": "intensive"},
    "week_2": {"percentage": 1.0, "monitoring": "intensive"},
    "week_3": {"percentage": 5.0, "monitoring": "enhanced"},
    "week_4": {"percentage": 5.0, "monitoring": "enhanced"},
    "week_5": {"percentage": 25.0, "monitoring": "standard"},
    "week_8": {"percentage": 50.0, "monitoring": "standard"},
    "week_10": {"percentage": 100.0, "monitoring": "standard"}
}
```

### 4.3 Monitoring and Performance Validation

Monitor key metrics during rollout:

1. **Response Times**: Compare LightRAG vs Perplexity
2. **Success Rates**: Track query success/failure rates
3. **User Satisfaction**: Monitor user feedback
4. **Cost Analysis**: Track API usage costs
5. **Quality Metrics**: Assess response accuracy

```python
# Access performance metrics
performance_summary = integrated_service.get_performance_summary()
ab_test_metrics = integrated_service.get_ab_test_metrics()
```

### 4.4 Rollback Procedures

#### Immediate Rollback (Emergency)
```bash
# Disable LightRAG immediately
LIGHTRAG_INTEGRATION_ENABLED=false
```

#### Gradual Rollback
```bash
# Reduce percentage gradually
LIGHTRAG_ROLLOUT_PERCENTAGE=25.0  # From 50%
LIGHTRAG_ROLLOUT_PERCENTAGE=10.0  # From 25%
LIGHTRAG_ROLLOUT_PERCENTAGE=1.0   # From 10%
LIGHTRAG_ROLLOUT_PERCENTAGE=0.0   # Disabled
```

#### Circuit Breaker Adjustment
```bash
# Make circuit breaker more sensitive
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=1  # From 3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=600  # From 300
```

## 5. Data Flow Documentation

### 5.1 Complete Request/Response Flow with LightRAG Enabled

```
1. User Input → Chainlit → main.py:on_message()
2. Language Detection & Translation (if needed)
3. QueryRequest Creation
4. IntegratedQueryService.query_async()
   ├── Cache Check (return if hit)
   ├── Feature Flag Evaluation
   │   ├── Global Enable Check
   │   ├── Rollout Percentage Check
   │   ├── User Cohort Assignment
   │   └── Circuit Breaker Status
   ├── Service Selection (LightRAG or Perplexity)
   ├── Query Execution with Timeout
   ├── Fallback on Failure (if configured)
   ├── Quality Assessment (if configured)
   ├── Performance Metrics Collection
   └── Response Caching (if successful)
5. Citation Processing & Formatting
6. Translation (if needed)
7. Response to User via Chainlit
```

### 5.2 How Queries Are Routed Between Systems

```python
def routing_decision_tree():
    if not config.lightrag_integration_enabled:
        return "perplexity"
    
    if circuit_breaker.is_open:
        return "perplexity"
    
    user_hash = hash_user_identifier(user_id, session_id)
    rollout_bucket = user_hash % 100
    
    if rollout_bucket < config.lightrag_rollout_percentage:
        if user_cohort == UserCohort.LIGHTRAG:
            return "lightrag"
        elif user_cohort == UserCohort.PERPLEXITY:
            return "perplexity"
        else:
            # A/B test assignment based on hash
            return "lightrag" if user_hash % 2 == 0 else "perplexity"
    
    return "perplexity"  # Default fallback
```

### 5.3 Citation and Confidence Score Integration

Both services provide citation data in unified format:

```python
# LightRAG Response
ServiceResponse(
    content="Clinical response text...",
    citations=None,  # LightRAG doesn't provide direct citations
    confidence_scores={},  # Confidence assessment via quality scorer
    response_type=ResponseType.LIGHTRAG
)

# Perplexity Response  
ServiceResponse(
    content="Clinical response text...",
    citations=[{"url": "...", "title": "..."}],  # Direct citations
    confidence_scores={"citation_url": 0.85},  # Extracted from content
    response_type=ResponseType.PERPLEXITY
)
```

### 5.4 Multi-language Support Preservation

Language detection and translation occur before and after service calls:

```
User Input (Any Language) 
  ↓ [Language Detection]
Input Translation to English (if needed)
  ↓ [Service Call]
English Response
  ↓ [Translation]
Response in User's Language
  ↓ [Citation Formatting]
Final Response to User
```

## 6. Configuration Management

### 6.1 Environment Variables Reference

#### Core Integration Settings
```bash
# Primary integration control
LIGHTRAG_INTEGRATION_ENABLED=true|false

# Rollout management
LIGHTRAG_ROLLOUT_PERCENTAGE=0.0-100.0
LIGHTRAG_ENABLE_AB_TESTING=true|false

# Fallback behavior
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true|false
LIGHTRAG_FALLBACK_TIMEOUT=30.0

# Circuit breaker protection
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true|false
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300
```

#### LightRAG Service Settings
```bash
# API configuration
OPENAI_API_KEY=sk-your-api-key
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small

# Performance settings
LIGHTRAG_MAX_ASYNC=16
LIGHTRAG_MAX_TOKENS=32768
LIGHTRAG_WORKING_DIR=./lightrag_data

# Knowledge base settings
LIGHTRAG_ENABLE_AUTO_INDEX=true
LIGHTRAG_INDEX_UPDATE_INTERVAL=3600
```

#### Monitoring and Logging
```bash
# Logging configuration
LIGHTRAG_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
LIGHTRAG_ENABLE_FILE_LOGGING=true|false
LIGHTRAG_LOG_DIR=./logs

# Performance monitoring
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true|false
LIGHTRAG_PERFORMANCE_SAMPLE_SIZE=100
LIGHTRAG_ENABLE_QUALITY_ASSESSMENT=true|false
```

### 6.2 Configuration Validation

The system validates configuration at startup:

```python
def validate_integration_config():
    required_vars = [
        "OPENAI_API_KEY",
        "PERPLEXITY_API",
        "LIGHTRAG_WORKING_DIR"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise LightRAGConfigError(f"Missing required environment variables: {missing}")
    
    # Validate percentage ranges
    rollout_pct = float(os.getenv("LIGHTRAG_ROLLOUT_PERCENTAGE", "0.0"))
    if not 0.0 <= rollout_pct <= 100.0:
        raise LightRAGConfigError("LIGHTRAG_ROLLOUT_PERCENTAGE must be 0.0-100.0")
```

## 7. Monitoring and Validation

### 7.1 Key Performance Indicators (KPIs)

Monitor these metrics during integration:

1. **Response Time Metrics**
   - Average response time: LightRAG vs Perplexity
   - 95th percentile response times
   - Timeout rates

2. **Success Rate Metrics**
   - Query success rates per service
   - Fallback activation rates
   - Circuit breaker activation frequency

3. **Quality Metrics**
   - Response relevance scores
   - Citation quality assessments
   - User satisfaction ratings

4. **Cost Metrics**
   - API usage costs per service
   - Cost per successful query
   - Budget utilization rates

### 7.2 Health Monitoring

```python
# Access health status
health_status = integrated_service.health_monitor.get_all_health_status()

# Example output:
{
    "lightrag": {
        "is_healthy": true,
        "last_check": "2025-08-08T10:30:00",
        "consecutive_failures": 0,
        "success_rate": 0.95
    },
    "perplexity": {
        "is_healthy": true,
        "last_check": "2025-08-08T10:30:00",
        "consecutive_failures": 0,
        "success_rate": 0.98
    }
}
```

### 7.3 Performance Comparison Reports

```python
# Generate performance comparison
ab_test_results = integrated_service.get_ab_test_metrics()

# Example output:
{
    "lightrag": {
        "sample_size": 150,
        "success_rate": 0.94,
        "avg_response_time": 2.3,
        "avg_quality_score": 0.87
    },
    "perplexity": {
        "sample_size": 148,
        "success_rate": 0.97,
        "avg_response_time": 1.8,
        "avg_quality_score": 0.82
    }
}
```

## 8. Troubleshooting Guide

### 8.1 Common Integration Issues

#### Issue: LightRAG Service Not Initializing
**Symptoms:**
- Logs show "LightRAG initialization failed"
- All queries route to Perplexity

**Resolution:**
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $LIGHTRAG_WORKING_DIR

# Verify directory permissions
ls -la $LIGHTRAG_WORKING_DIR

# Check OpenAI API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"text-embedding-3-small","input":"test"}' \
     https://api.openai.com/v1/embeddings
```

#### Issue: Circuit Breaker Frequently Activating
**Symptoms:**
- Logs show "Circuit breaker opened"
- Performance degradation

**Resolution:**
```bash
# Adjust circuit breaker sensitivity
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5  # Increase from 3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=600  # Increase timeout
```

#### Issue: High Response Times
**Symptoms:**
- Slow query responses
- Timeout errors

**Resolution:**
```bash
# Reduce async concurrency
LIGHTRAG_MAX_ASYNC=8  # Reduce from 16

# Adjust timeout settings
LIGHTRAG_FALLBACK_TIMEOUT=60.0  # Increase timeout

# Enable performance mode
LIGHTRAG_ENABLE_PERFORMANCE_OPTIMIZATION=true
```

### 8.2 Debugging Steps

1. **Check Environment Variables**
   ```bash
   env | grep LIGHTRAG
   env | grep OPENAI
   ```

2. **Validate Configuration**
   ```python
   from lightrag_integration.config import LightRAGConfig
   config = LightRAGConfig.from_environment()
   print(config.validate())
   ```

3. **Test Service Health**
   ```python
   health_status = integrated_service.health_monitor.get_all_health_status()
   print(json.dumps(health_status, indent=2))
   ```

4. **Review Performance Metrics**
   ```python
   performance = integrated_service.get_performance_summary()
   print(json.dumps(performance, indent=2))
   ```

### 8.3 Log Analysis

Key log patterns to monitor:

```bash
# Successful integration
grep "LightRAG integration service initialized successfully" logs/

# Routing decisions
grep "Routing decision:" logs/

# Circuit breaker events
grep "Circuit breaker" logs/

# Performance metrics
grep "Query routed to:" logs/
```

---

## Summary

This integration procedure provides a comprehensive, production-ready approach to integrating LightRAG with the existing Clinical Metabolomics Oracle system. The integration:

- ✅ Maintains full backward compatibility
- ✅ Provides gradual rollout capabilities
- ✅ Includes comprehensive error handling and fallback mechanisms
- ✅ Supports A/B testing for performance comparison
- ✅ Implements circuit breaker protection
- ✅ Preserves existing multi-language support
- ✅ Includes detailed monitoring and validation procedures

The modular design allows for safe deployment with minimal risk to the existing production system while providing advanced capabilities for performance optimization and quality assessment.

**Next Steps:**
1. Set up development environment with provided configuration
2. Initialize LightRAG knowledge base with biomedical content
3. Deploy with 1% rollout for initial testing
4. Monitor performance metrics and gradually increase rollout percentage
5. Compare A/B test results and optimize configuration

For additional support, refer to the existing documentation in the `lightrag_integration/` directory or contact the development team.