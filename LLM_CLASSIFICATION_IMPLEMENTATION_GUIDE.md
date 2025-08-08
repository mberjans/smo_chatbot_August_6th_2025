# LLM-Based Query Classification Implementation Guide
## Clinical Metabolomics Oracle System

### Overview

This guide documents the comprehensive LLM-based query classification system designed for the Clinical Metabolomics Oracle. The system enhances the existing keyword-based classification with semantic understanding while maintaining performance and cost efficiency.

---

## 🎯 System Design

### Architecture Overview

```
User Query
    ↓
LLM Query Classifier
    ├── Cache Check (optional)
    ├── LLM Classification (primary)
    ├── Keyword Fallback (backup)
    └── Validation (low confidence)
    ↓
Classification Result
    ↓
Routing Decision
    ├── KNOWLEDGE_GRAPH → LightRAG
    ├── REAL_TIME → Perplexity API
    └── GENERAL → Flexible routing
```

### Key Components

1. **LLM Classification Prompts** (`llm_classification_prompts.py`)
   - Token-optimized prompts for clinical metabolomics
   - Few-shot examples for each category
   - Structured JSON output schema

2. **LLM Query Classifier** (`llm_query_classifier.py`)
   - Main classification engine with LLM integration
   - Intelligent caching and fallback mechanisms
   - Performance monitoring and cost tracking

3. **Demo Script** (`demo_llm_classification.py`)
   - Comprehensive demonstration of system capabilities
   - Performance comparison with keyword-based system
   - Cost analysis and optimization recommendations

---

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from lightrag_integration.llm_query_classifier import (
    LLMQueryClassifier, 
    LLMClassificationConfig,
    LLMProvider
)

async def classify_query():
    # Configure LLM classifier
    config = LLMClassificationConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        api_key="your-openai-api-key",
        timeout_seconds=3.0
    )
    
    # Initialize classifier
    classifier = LLMQueryClassifier(config)
    
    # Classify a query
    query = "What is the relationship between glucose metabolism and insulin signaling?"
    result, used_llm = await classifier.classify_query(query)
    
    print(f"Category: {result.category}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")

# Run the example
asyncio.run(classify_query())
```

### Integration with Existing Router

```python
from lightrag_integration.query_router import BiomedicalQueryRouter
from lightrag_integration.llm_query_classifier import create_llm_enhanced_router

# Create enhanced router with LLM + keyword fallback
config = LLMClassificationConfig(api_key="your-key")
router = await create_llm_enhanced_router(config)

# Use same interface as existing router
result, used_llm = await router.classify_query(
    "Latest clinical trials for metabolomics biomarkers"
)
```

---

## 📊 Classification Categories

### KNOWLEDGE_GRAPH
**Route to LightRAG knowledge graph system**

- Established relationships between metabolites, pathways, diseases
- Mechanistic questions about biological processes
- Structural queries about molecular connections
- Drug mechanisms and biomarker associations

**Examples:**
- "What is the relationship between glucose metabolism and insulin signaling?"
- "How does the citric acid cycle connect to fatty acid biosynthesis?"
- "Find biomarkers associated with Alzheimer's disease"

### REAL_TIME
**Route to Perplexity API for current information**

- Latest research findings and publications (2024+)
- Recent clinical trials and FDA approvals
- Breaking news in metabolomics and drug discovery
- Current market developments and technologies

**Examples:**
- "Latest 2024 FDA approvals for metabolomics diagnostics"
- "Recent breakthrough discoveries in cancer metabolomics"
- "Current clinical trials using mass spectrometry"

### GENERAL
**Route flexibly (either system can handle)**

- Basic definitional or educational questions
- Simple explanations of metabolomics concepts
- Broad introductory topics without temporal requirements
- General methodology inquiries

**Examples:**
- "What is metabolomics and how does it work?"
- "Explain the basics of LC-MS analysis"
- "Define biomarker and its role in medicine"

---

## 🎛️ Configuration Options

### LLMClassificationConfig Parameters

```python
@dataclass
class LLMClassificationConfig:
    # LLM Provider Settings
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o-mini"  # Fast, cost-effective
    api_key: Optional[str] = None
    max_tokens: int = 200
    temperature: float = 0.1  # Low for consistency
    
    # Performance Settings
    timeout_seconds: float = 3.0  # Max wait time
    max_retries: int = 2
    fallback_to_keywords: bool = True
    
    # Prompt Strategy
    use_examples_for_uncertain: bool = True
    primary_confidence_threshold: float = 0.7
    validation_threshold: float = 0.5
    
    # Caching Settings
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    max_cache_size: int = 1000
    
    # Cost Optimization
    daily_api_budget: float = 5.0  # USD
    cost_per_1k_tokens: float = 0.0005
```

### Performance Tuning

**For Speed Optimization:**
```python
config = LLMClassificationConfig(
    model_name="gpt-4o-mini",  # Fastest model
    timeout_seconds=2.0,       # Aggressive timeout
    use_examples_for_uncertain=False,  # Skip examples
    enable_caching=True,       # Cache aggressively
    cache_ttl_hours=48         # Longer cache lifetime
)
```

**For Accuracy Optimization:**
```python
config = LLMClassificationConfig(
    model_name="gpt-4o",       # More accurate model
    use_examples_for_uncertain=True,   # Use examples
    primary_confidence_threshold=0.5,  # Lower threshold
    validation_threshold=0.3,  # More validation
    max_retries=3              # More retries
)
```

**For Cost Optimization:**
```python
config = LLMClassificationConfig(
    model_name="gpt-4o-mini",  # Cheapest model
    daily_api_budget=1.0,      # Strict budget
    enable_caching=True,       # Maximize cache hits
    fallback_to_keywords=True, # Use fallback often
    timeout_seconds=2.0        # Fail fast
)
```

---

## 🎯 Prompt Engineering

### Primary Classification Prompt

The system uses a carefully engineered prompt optimized for:
- **Clinical metabolomics domain expertise**
- **Token efficiency** (<600 tokens typical)
- **Structured JSON output**
- **Biomedical entity recognition**
- **Temporal signal detection**

### Prompt Strategies

1. **Primary Prompt**: Full semantic analysis with domain context
2. **Fallback Prompt**: Ultra-fast classification (~100 tokens)
3. **Validation Prompt**: Confidence verification for uncertain cases

### Few-Shot Examples

Each category includes 5 carefully crafted examples:
- **Domain-specific queries** (metabolomics, clinical research)
- **Varied complexity levels** (simple to complex)
- **Clear decision boundaries** (temporal vs. established knowledge)

---

## 📈 Performance Monitoring

### Key Metrics Tracked

```python
# Get comprehensive statistics
stats = classifier.get_classification_statistics()

# Classification performance
print(f"LLM Success Rate: {stats['classification_metrics']['success_rate']:.1f}%")
print(f"Avg Response Time: {stats['performance_metrics']['avg_response_time_ms']:.1f}ms")
print(f"Cache Hit Rate: {stats['classification_metrics']['cache_hits']/stats['classification_metrics']['total_classifications']*100:.1f}%")

# Cost tracking
print(f"Daily Cost: ${stats['cost_metrics']['daily_api_cost']:.4f}")
print(f"Budget Utilization: {stats['cost_metrics']['budget_utilization']:.1f}%")
```

### Performance Targets

| Metric | Target | Optimized For |
|--------|--------|---------------|
| Response Time | <2 seconds | User experience |
| Classification Accuracy | >85% | Routing quality |
| LLM Success Rate | >90% | System reliability |
| Daily API Cost | <$5.00 | Cost efficiency |
| Cache Hit Rate | >30% | Performance & cost |

---

## 🔧 Integration Examples

### With Existing BiomedicalQueryRouter

```python
from lightrag_integration.query_router import BiomedicalQueryRouter
from lightrag_integration.llm_query_classifier import convert_llm_result_to_routing_prediction

# Existing router for fallback
biomedical_router = BiomedicalQueryRouter(logger)

# LLM classifier with fallback
llm_classifier = LLMQueryClassifier(config, biomedical_router)

# Classify and convert to routing prediction
query = "Recent advances in metabolomics"
llm_result, used_llm = await llm_classifier.classify_query(query)
routing_prediction = convert_llm_result_to_routing_prediction(llm_result, query, used_llm)

# Use with existing infrastructure
if routing_prediction.routing_decision == RoutingDecision.LIGHTRAG:
    # Route to LightRAG
    response = await lightrag_system.query(query)
elif routing_prediction.routing_decision == RoutingDecision.PERPLEXITY:
    # Route to Perplexity API
    response = await perplexity_system.query(query)
```

### Async Batch Processing

```python
async def classify_batch(queries: List[str]) -> List[ClassificationResult]:
    """Classify multiple queries efficiently."""
    
    # Create tasks for concurrent processing
    tasks = [classifier.classify_query(query) for query in queries]
    
    # Execute with rate limiting
    results = []
    for task in asyncio.as_completed(tasks):
        result, used_llm = await task
        results.append(result)
        
        # Optional: add delay for rate limiting
        if used_llm:
            await asyncio.sleep(0.1)
    
    return results

# Usage
queries = ["query1", "query2", "query3"]
results = await classify_batch(queries)
```

---

## 🚨 Error Handling & Fallbacks

### Fallback Strategy

```
LLM Classification
    ↓ (if fails)
Keyword-based Classification (BiomedicalQueryRouter)
    ↓ (if fails)  
Simple Pattern Matching
    ↓ (if fails)
Default GENERAL Category
```

### Error Handling Best Practices

```python
async def robust_classification(query: str) -> ClassificationResult:
    """Robust classification with comprehensive error handling."""
    
    try:
        # Primary: LLM classification
        result, used_llm = await classifier.classify_query(query)
        
        # Validate result quality
        if result.confidence < 0.3:
            logger.warning(f"Low confidence classification: {result.confidence}")
        
        return result
        
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        
        # Fallback: simple rule-based classification
        if any(word in query.lower() for word in ['latest', 'recent', '2024', '2025']):
            category = "REAL_TIME"
        elif any(word in query.lower() for word in ['relationship', 'pathway', 'mechanism']):
            category = "KNOWLEDGE_GRAPH"
        else:
            category = "GENERAL"
        
        return ClassificationResult(
            category=category,
            confidence=0.3,
            reasoning="Emergency fallback classification",
            alternative_categories=[],
            uncertainty_indicators=["fallback_used", "low_confidence"],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
```

---

## 💰 Cost Management

### Cost Optimization Strategies

1. **Intelligent Caching**
   - Cache successful classifications for 24+ hours
   - Use query similarity matching for cache hits
   - Implement LRU eviction for memory efficiency

2. **Adaptive Prompt Selection**
   - Use fallback prompt for simple/repeated queries
   - Reserve examples for uncertain classifications
   - Implement query complexity scoring

3. **Budget Controls**
   - Set daily spending limits with alerts
   - Track cost per classification
   - Automatic fallback when budget exceeded

### Cost Analysis

```python
# Daily cost breakdown
stats = classifier.get_classification_statistics()
cost_metrics = stats['cost_metrics']

print(f"Classifications today: {stats['classification_metrics']['total_classifications']}")
print(f"API cost today: ${cost_metrics['daily_api_cost']:.4f}")
print(f"Average cost per query: ${cost_metrics['estimated_cost_per_classification']:.6f}")
print(f"Tokens used: {cost_metrics['daily_token_usage']}")
print(f"Budget remaining: ${cost_metrics['daily_budget'] - cost_metrics['daily_api_cost']:.4f}")
```

---

## 🔍 Testing & Validation

### Running the Demo

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run comprehensive demo
cd lightrag_integration
python demo_llm_classification.py
```

### Test Coverage

The demo script includes:
- **18 test queries** across all categories
- **Performance comparison** with keyword system
- **Edge case handling** (ambiguous, short queries)
- **Cost tracking** and budget monitoring
- **Optimization recommendations**

### Validation Metrics

```python
# Compare LLM vs Keyword accuracy
demo = LLMClassificationDemo()
await demo.run_demo(api_key)

# Results show:
# - LLM accuracy on test set
# - Speed comparison (LLM vs keywords)
# - Confidence score analysis
# - Cost per classification
```

---

## 🎛️ Advanced Configuration

### Custom Model Integration

```python
# Example: Adding support for local models
class LocalLLMProvider(LLMProvider):
    LOCAL_OPENAI_COMPATIBLE = "local_openai"

# Configure for local endpoint
config = LLMClassificationConfig(
    provider=LocalLLMProvider.LOCAL_OPENAI_COMPATIBLE,
    model_name="llama-3-8b-instruct",
    api_key="not-needed",
    base_url="http://localhost:8000/v1"  # Local server
)
```

### Custom Prompt Templates

```python
# Override default prompts
custom_prompts = LLMClassificationPrompts()
custom_prompts.PRIMARY_CLASSIFICATION_PROMPT = """
Your custom prompt template here...
Categories: {categories}
Query: {query_text}
Response: JSON only
"""

# Use custom prompts
classifier = LLMQueryClassifier(config, custom_prompts=custom_prompts)
```

### Performance Monitoring Integration

```python
# Integration with monitoring systems
import prometheus_client as prom

# Create metrics
classification_counter = prom.Counter('classifications_total', 'Total classifications', ['category', 'method'])
response_time_histogram = prom.Histogram('classification_response_time_seconds', 'Response time')
confidence_gauge = prom.Gauge('classification_confidence', 'Classification confidence')

# Update metrics after classification
result, used_llm = await classifier.classify_query(query)
classification_counter.labels(category=result.category, method='llm' if used_llm else 'keyword').inc()
confidence_gauge.set(result.confidence)
```

---

## 🚀 Deployment Considerations

### Production Deployment

1. **Environment Configuration**
   ```bash
   # Required environment variables
   export OPENAI_API_KEY="your-production-key"
   export LLM_CLASSIFICATION_BUDGET="10.0"  # Daily budget
   export LLM_CLASSIFICATION_TIMEOUT="3.0"  # Timeout seconds
   export LLM_CLASSIFICATION_CACHE_TTL="24"  # Cache TTL hours
   ```

2. **Monitoring Setup**
   - Track classification accuracy over time
   - Monitor API costs and usage patterns
   - Set up alerts for high error rates or budget overruns
   - Log classification decisions for audit and improvement

3. **Scaling Considerations**
   - Use connection pooling for high-throughput scenarios
   - Implement distributed caching (Redis) for multi-instance deployments
   - Consider async batch processing for bulk operations
   - Monitor rate limits and implement backoff strategies

### Health Checks

```python
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for production monitoring."""
    
    try:
        # Test classification with simple query
        test_query = "What is metabolomics?"
        result, used_llm = await classifier.classify_query(test_query)
        
        # Get current statistics
        stats = classifier.get_classification_statistics()
        
        return {
            "status": "healthy",
            "llm_success_rate": stats["classification_metrics"]["success_rate"],
            "avg_response_time_ms": stats["performance_metrics"]["avg_response_time_ms"],
            "daily_budget_used": stats["cost_metrics"]["budget_utilization"],
            "cache_hit_rate": stats["classification_metrics"]["cache_hits"] / max(1, stats["classification_metrics"]["total_classifications"]) * 100,
            "last_successful_classification": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

---

## 📚 API Reference

### LLMQueryClassifier

```python
class LLMQueryClassifier:
    def __init__(self, config: LLMClassificationConfig, 
                 biomedical_router: Optional[BiomedicalQueryRouter] = None,
                 logger: Optional[logging.Logger] = None)
    
    async def classify_query(self, query_text: str, 
                           context: Optional[Dict[str, Any]] = None,
                           force_llm: bool = False) -> Tuple[ClassificationResult, bool]
    
    def get_classification_statistics(self) -> Dict[str, Any]
    
    def optimize_configuration(self) -> Dict[str, Any]
```

### ClassificationResult

```python
@dataclass
class ClassificationResult:
    category: str  # "KNOWLEDGE_GRAPH", "REAL_TIME", "GENERAL"
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Brief explanation
    alternative_categories: List[Dict[str, float]]  # Alternative options
    uncertainty_indicators: List[str]  # Uncertainty factors
    biomedical_signals: Dict[str, Any]  # Detected biomedical entities
    temporal_signals: Dict[str, Any]  # Detected temporal patterns
```

---

## 🎓 Best Practices

### Query Preprocessing

```python
def preprocess_query(query: str) -> str:
    """Preprocess query for better classification."""
    
    # Clean and normalize
    query = query.strip().lower()
    
    # Expand common abbreviations
    abbreviations = {
        'ms': 'mass spectrometry',
        'nmr': 'nuclear magnetic resonance',
        'lc-ms': 'liquid chromatography mass spectrometry',
        'gc-ms': 'gas chromatography mass spectrometry'
    }
    
    for abbrev, expansion in abbreviations.items():
        query = query.replace(abbrev, expansion)
    
    return query
```

### Confidence Calibration

```python
def calibrate_confidence(result: ClassificationResult, 
                        validation_data: List[Dict]) -> float:
    """Calibrate confidence based on historical validation data."""
    
    # Find similar historical classifications
    similar = [v for v in validation_data 
              if v['predicted_category'] == result.category 
              and abs(v['predicted_confidence'] - result.confidence) < 0.1]
    
    if similar:
        # Average actual accuracy for similar predictions
        actual_accuracy = sum(v['was_correct'] for v in similar) / len(similar)
        # Blend with original confidence
        calibrated = 0.7 * result.confidence + 0.3 * actual_accuracy
        return max(0.1, min(0.9, calibrated))
    
    return result.confidence
```

### Batch Optimization

```python
async def optimize_batch_processing(queries: List[str]) -> List[ClassificationResult]:
    """Optimize batch processing with intelligent grouping."""
    
    # Group by expected complexity
    simple_queries = [q for q in queries if len(q.split()) <= 5]
    complex_queries = [q for q in queries if len(q.split()) > 5]
    
    results = []
    
    # Process simple queries with fallback prompt
    for query in simple_queries:
        result, _ = await classifier.classify_query(query, force_fallback=True)
        results.append(result)
        
        # Rate limiting
        await asyncio.sleep(0.05)
    
    # Process complex queries with full prompt
    for query in complex_queries:
        result, _ = await classifier.classify_query(query)
        results.append(result)
        await asyncio.sleep(0.1)
    
    return results
```

---

## 🔄 Future Enhancements

### Planned Features

1. **Fine-tuned Models**
   - Train domain-specific models on metabolomics data
   - Reduce inference costs with smaller, specialized models
   - Improve accuracy for domain-specific terminology

2. **Active Learning**
   - Collect user feedback on classification accuracy
   - Retrain models based on real-world usage patterns
   - Implement uncertainty-based query selection for labeling

3. **Multi-modal Classification**
   - Incorporate query context (user history, session data)
   - Use embeddings for semantic similarity clustering
   - Implement query intent understanding beyond categorization

4. **Advanced Caching**
   - Semantic similarity-based cache matching
   - Hierarchical caching with category-specific TTLs
   - Predictive caching based on usage patterns

### Extension Points

```python
# Custom classification logic
class CustomClassificationLogic:
    def pre_process(self, query: str) -> str:
        """Custom query preprocessing."""
        pass
    
    def post_process(self, result: ClassificationResult) -> ClassificationResult:
        """Custom result post-processing."""
        pass
    
    def should_use_llm(self, query: str) -> bool:
        """Custom logic for LLM usage decision."""
        pass

# Integration
classifier = LLMQueryClassifier(config, custom_logic=CustomClassificationLogic())
```

---

## 📞 Support & Resources

### Troubleshooting

**Common Issues:**

1. **High API Costs**
   - Reduce `daily_api_budget`
   - Increase `cache_ttl_hours`
   - Set `fallback_to_keywords=True`

2. **Slow Response Times**
   - Reduce `timeout_seconds`
   - Use `gpt-4o-mini` model
   - Set `use_examples_for_uncertain=False`

3. **Low Accuracy**
   - Increase `primary_confidence_threshold`
   - Enable `use_examples_for_uncertain=True`
   - Review and improve few-shot examples

### Getting Help

- **Documentation**: This guide and inline code documentation
- **Demo Script**: `demo_llm_classification.py` for examples and testing
- **Logging**: Enable DEBUG level logging for detailed operation traces
- **Monitoring**: Use `get_classification_statistics()` for performance insights

### Contributing

To extend or improve the system:

1. **Add new categories**: Extend `ClassificationCategory` enum and add examples
2. **Improve prompts**: Modify templates in `llm_classification_prompts.py`
3. **Add providers**: Implement new LLM providers in `llm_query_classifier.py`
4. **Enhance caching**: Improve cache logic in `ClassificationCache`

---

This completes the comprehensive LLM-based query classification system for the Clinical Metabolomics Oracle. The system provides semantic understanding while maintaining the performance, reliability, and cost-effectiveness required for production use.