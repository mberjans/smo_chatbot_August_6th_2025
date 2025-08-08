# LightRAG Integration Troubleshooting Guide

## Table of Contents

1. [Common Integration Issues](#common-integration-issues)
2. [Diagnostic Tools and Scripts](#diagnostic-tools-and-scripts)
3. [Performance Issues](#performance-issues)
4. [Quality and Response Issues](#quality-and-response-issues)
5. [Feature Flag Issues](#feature-flag-issues)
6. [Budget and Cost Issues](#budget-and-cost-issues)
7. [API and Connectivity Issues](#api-and-connectivity-issues)
8. [Knowledge Base Issues](#knowledge-base-issues)
9. [Monitoring and Alerting Issues](#monitoring-and-alerting-issues)
10. [Emergency Procedures](#emergency-procedures)

---

## Common Integration Issues

### Issue 1: LightRAG System Fails to Initialize

**Symptoms:**
- System crashes during startup
- "Failed to initialize LightRAG" errors
- Chainlit interface doesn't load properly

**Diagnostic Steps:**

```python
# Diagnostic script: diagnose_initialization.py
import asyncio
from lightrag_integration import (
    validate_configuration,
    DiagnosticTools,
    SystemHealthChecker
)

async def diagnose_initialization_failure():
    """Comprehensive initialization diagnostics"""
    
    print("🔍 Running LightRAG initialization diagnostics...")
    
    # 1. Check environment configuration
    print("\\n1. Checking environment configuration...")
    try:
        await validate_configuration()
        print("✅ Configuration valid")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # 2. Check OpenAI API connectivity
    print("\\n2. Checking OpenAI API connectivity...")
    diagnostics = DiagnosticTools()
    api_status = await diagnostics.check_openai_connectivity()
    
    if api_status.success:
        print("✅ OpenAI API connectivity working")
    else:
        print(f"❌ OpenAI API error: {api_status.error}")
        print("   Check your OPENAI_API_KEY environment variable")
        return
    
    # 3. Check directory permissions
    print("\\n3. Checking directory permissions...")
    dir_status = diagnostics.check_directory_permissions()
    
    if dir_status.success:
        print("✅ Directory permissions OK")
    else:
        print(f"❌ Directory permission error: {dir_status.error}")
        print("   Fix directory permissions and try again")
        return
    
    # 4. Check system resources
    print("\\n4. Checking system resources...")
    resource_status = diagnostics.check_system_resources()
    
    if resource_status.memory_available_gb < 4:
        print(f"⚠️  Low memory: {resource_status.memory_available_gb:.1f}GB available")
        print("   LightRAG requires at least 4GB of available memory")
    else:
        print("✅ Sufficient memory available")
    
    # 5. Check knowledge base integrity
    print("\\n5. Checking knowledge base...")
    try:
        kb_status = await diagnostics.check_knowledge_base_integrity()
        if kb_status.success:
            print(f"✅ Knowledge base OK ({kb_status.document_count} documents)")
        else:
            print(f"❌ Knowledge base error: {kb_status.error}")
    except Exception as e:
        print(f"❌ Knowledge base check failed: {e}")
    
    print("\\n🔍 Initialization diagnostics complete")

if __name__ == "__main__":
    asyncio.run(diagnose_initialization_failure())
```

**Solutions:**

1. **Environment Variable Issues:**
```bash
# Check required environment variables
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:+SET}"
echo "LIGHTRAG_WORKING_DIR: ${LIGHTRAG_WORKING_DIR:-./lightrag_storage}"
echo "LIGHTRAG_PAPERS_DIR: ${LIGHTRAG_PAPERS_DIR:-./papers}"

# Set missing variables
export OPENAI_API_KEY="your_api_key_here"
export LIGHTRAG_WORKING_DIR="./lightrag_storage"
export LIGHTRAG_PAPERS_DIR="./papers"
```

2. **Directory Permission Issues:**
```bash
# Fix directory permissions
chmod -R 755 ./lightrag_storage
chmod -R 755 ./papers
mkdir -p ./lightrag_storage ./papers
```

3. **Memory Issues:**
```bash
# Check available memory
free -h

# If insufficient memory, try reducing chunk size
export LIGHTRAG_CHUNK_SIZE=800  # Default is 1200
export LIGHTRAG_MAX_TOKENS=4000  # Default is 8000
```

### Issue 2: Feature Flags Not Working

**Symptoms:**
- All users getting same treatment
- Rollout percentage not being applied
- A/B test assignments inconsistent

**Diagnostic Script:**

```python
# diagnose_feature_flags.py
from lightrag_integration import FeatureFlagManager, FeatureFlagDiagnostics

def diagnose_feature_flags():
    """Diagnose feature flag issues"""
    
    print("🚩 Diagnosing feature flag system...")
    
    diagnostics = FeatureFlagDiagnostics()
    
    # 1. Check configuration
    print("\\n1. Checking feature flag configuration...")
    config_issues = diagnostics.validate_feature_flag_config()
    
    if not config_issues:
        print("✅ Feature flag configuration OK")
    else:
        print("❌ Configuration issues found:")
        for issue in config_issues:
            print(f"   • {issue}")
    
    # 2. Check user assignment consistency
    print("\\n2. Testing user assignment consistency...")
    test_users = ["user1", "user2", "user3", "user4", "user5"]
    
    feature_flags = FeatureFlagManager()
    
    for user_id in test_users:
        assignment1 = feature_flags.get_user_assignment(user_id)
        assignment2 = feature_flags.get_user_assignment(user_id)
        
        if assignment1.group == assignment2.group:
            print(f"✅ User {user_id}: Consistent assignment ({assignment1.group})")
        else:
            print(f"❌ User {user_id}: Inconsistent assignment ({assignment1.group} vs {assignment2.group})")
    
    # 3. Test rollout percentage
    print("\\n3. Testing rollout percentage application...")
    rollout_percentage = int(os.getenv('LIGHTRAG_ROLLOUT_PERCENTAGE', '0'))
    
    if rollout_percentage == 0:
        print("⚠️  Rollout percentage is 0% - no users will get LightRAG")
    elif rollout_percentage == 100:
        print("ℹ️  Rollout percentage is 100% - all users will get LightRAG")
    else:
        # Test with 100 users
        lightrag_users = 0
        for i in range(100):
            user_id = f"test_user_{i}"
            assignment = feature_flags.get_user_assignment(user_id)
            if assignment.use_lightrag:
                lightrag_users += 1
        
        print(f"📊 Actual rollout: {lightrag_users}% (target: {rollout_percentage}%)")
        
        if abs(lightrag_users - rollout_percentage) <= 5:  # Allow 5% tolerance
            print("✅ Rollout percentage working correctly")
        else:
            print("❌ Rollout percentage not working as expected")

if __name__ == "__main__":
    diagnose_feature_flags()
```

**Solutions:**

1. **Environment Variable Fix:**
```bash
# Check and set feature flag variables
export LIGHTRAG_ROLLOUT_PERCENTAGE=25
export LIGHTRAG_ENABLED=true
export LIGHTRAG_CIRCUIT_BREAKER=true
```

2. **Hash Consistency Fix:**
```python
# Ensure consistent user ID hashing
def get_consistent_user_id(session_id: str) -> str:
    """Generate consistent user ID from session"""
    return f"user_{hash(session_id) % 1000000}"
```

### Issue 3: Poor Response Quality

**Symptoms:**
- Low quality scores consistently
- Responses don't match query intent
- Hallucinated information in responses

**Diagnostic Script:**

```python
# diagnose_quality.py
import asyncio
from lightrag_integration import (
    QualityAssessmentSuite,
    KnowledgeBaseAnalyzer,
    QueryAnalyzer
)

async def diagnose_quality_issues():
    """Diagnose response quality issues"""
    
    print("📊 Diagnosing response quality issues...")
    
    quality_suite = QualityAssessmentSuite()
    kb_analyzer = KnowledgeBaseAnalyzer()
    query_analyzer = QueryAnalyzer()
    
    # 1. Analyze knowledge base coverage
    print("\\n1. Analyzing knowledge base coverage...")
    kb_analysis = await kb_analyzer.analyze_coverage()
    
    print(f"   Documents in KB: {kb_analysis.document_count}")
    print(f"   Total chunks: {kb_analysis.chunk_count}")
    print(f"   Average chunk size: {kb_analysis.avg_chunk_size}")
    print(f"   Coverage score: {kb_analysis.coverage_score:.2f}")
    
    if kb_analysis.coverage_score < 0.7:
        print("❌ Low knowledge base coverage detected")
        print("   Recommendation: Add more relevant PDF documents")
    else:
        print("✅ Knowledge base coverage adequate")
    
    # 2. Test with sample queries
    print("\\n2. Testing sample queries...")
    test_queries = [
        "What is clinical metabolomics?",
        "How are biomarkers used in metabolic diseases?",
        "What are the latest advances in metabolomics research?",
        "Explain mass spectrometry in metabolomics"
    ]
    
    from lightrag_integration import create_clinical_rag_system
    rag_system = await create_clinical_rag_system()
    
    quality_scores = []
    
    for query in test_queries:
        try:
            response = await rag_system.query(query)
            
            quality_assessment = await quality_suite.assess_response(
                query=query,
                response=response,
                sources=rag_system.get_last_sources()
            )
            
            quality_scores.append(quality_assessment.composite_score)
            
            print(f"   Query: '{query[:50]}...'")
            print(f"   Quality: {quality_assessment.composite_score:.2f}")
            
            if quality_assessment.composite_score < 0.7:
                print(f"   ❌ Low quality - Issues: {quality_assessment.issues}")
            else:
                print(f"   ✅ Good quality")
                
        except Exception as e:
            print(f"   ❌ Error processing query: {e}")
    
    # 3. Overall quality assessment
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"\\n📊 Average quality score: {avg_quality:.2f}")
        
        if avg_quality < 0.7:
            print("❌ Overall quality below threshold")
            print("   Recommendations:")
            print("   • Add more relevant documents to knowledge base")
            print("   • Adjust QueryParam settings")
            print("   • Consider re-indexing knowledge base")
        else:
            print("✅ Overall quality acceptable")

if __name__ == "__main__":
    asyncio.run(diagnose_quality_issues())
```

**Solutions:**

1. **Knowledge Base Enhancement:**
```python
# Add more relevant documents
# Place additional PDF files in ./papers directory
# Re-run knowledge base initialization

from lightrag_integration import create_clinical_rag_system

async def rebuild_knowledge_base():
    rag_system = await create_clinical_rag_system()
    await rag_system.initialize_knowledge_base(force_rebuild=True)
```

2. **QueryParam Optimization:**
```python
# Adjust query parameters for better retrieval
from lightrag import QueryParam

optimized_params = QueryParam(
    mode="hybrid",  # Use both local and global retrieval
    top_k=15,       # Increase from default 10
    max_total_tokens=6000,  # Increase context window
    response_type="Multiple Paragraphs"
)

response = await rag_system.query(query, param=optimized_params)
```

---

## Performance Issues

### Issue 4: Slow Response Times

**Symptoms:**
- Responses taking >30 seconds
- Timeout errors
- Users experiencing delays

**Performance Diagnostic Script:**

```python
# diagnose_performance.py
import asyncio
import time
from lightrag_integration import (
    PerformanceProfiler,
    create_clinical_rag_system
)

async def diagnose_performance_issues():
    """Diagnose performance bottlenecks"""
    
    print("⚡ Diagnosing performance issues...")
    
    profiler = PerformanceProfiler()
    rag_system = await create_clinical_rag_system()
    
    # Test queries with timing
    test_queries = [
        "What is clinical metabolomics?",  # Simple query
        "Explain the relationship between metabolomics and personalized medicine in the context of cardiovascular disease",  # Complex query
        "Define biomarker"  # Very simple query
    ]
    
    print("\\n🔍 Performance profiling:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n{i}. Testing query: '{query[:50]}...'")
        
        # Profile the complete pipeline
        start_time = time.time()
        
        try:
            with profiler.profile_query_processing() as profile_context:
                response = await rag_system.query(query)
            
            total_time = time.time() - start_time
            profile_results = profile_context.get_results()
            
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Embedding time: {profile_results.embedding_time:.2f}s")
            print(f"   Retrieval time: {profile_results.retrieval_time:.2f}s")
            print(f"   LLM time: {profile_results.llm_time:.2f}s")
            print(f"   Response length: {len(response)} chars")
            
            # Identify bottlenecks
            if profile_results.embedding_time > 3.0:
                print("   ❌ Slow embedding generation - consider caching")
            
            if profile_results.retrieval_time > 5.0:
                print("   ❌ Slow retrieval - consider optimizing chunk size")
            
            if profile_results.llm_time > 15.0:
                print("   ❌ Slow LLM response - consider reducing max_tokens")
            
            if total_time < 10.0:
                print("   ✅ Good performance")
            elif total_time < 20.0:
                print("   ⚠️  Acceptable performance")
            else:
                print("   ❌ Poor performance - optimization needed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # System resource check
    print("\\n💻 System resources:")
    resource_usage = profiler.get_system_resource_usage()
    
    print(f"   CPU usage: {resource_usage.cpu_percent}%")
    print(f"   Memory usage: {resource_usage.memory_used_gb:.1f}GB / {resource_usage.memory_total_gb:.1f}GB")
    print(f"   Memory percentage: {resource_usage.memory_percent}%")
    
    if resource_usage.cpu_percent > 80:
        print("   ❌ High CPU usage - consider reducing concurrent processing")
    
    if resource_usage.memory_percent > 85:
        print("   ❌ High memory usage - consider reducing chunk size or cache size")

if __name__ == "__main__":
    asyncio.run(diagnose_performance_issues())
```

**Performance Optimization Solutions:**

1. **Enable Response Caching:**
```python
from lightrag_integration import ResponseCache

# Add caching to reduce repeated processing
cache = ResponseCache(
    backend="memory",  # or "redis" for distributed systems
    ttl_seconds=3600,  # Cache for 1 hour
    max_cache_size_mb=256
)

async def cached_query_processing(query: str):
    # Check cache first
    cached_response = await cache.get(query)
    if cached_response:
        return cached_response
    
    # Process and cache
    response = await rag_system.query(query)
    await cache.set(query, response)
    return response
```

2. **Optimize QueryParam Settings:**
```python
# Optimize for faster responses
fast_query_params = QueryParam(
    mode="local",  # Use only local retrieval (faster)
    top_k=8,       # Reduce from default 10
    max_total_tokens=4000,  # Reduce from default 8000
    chunk_token_size=800,   # Reduce from default 1200
)

response = await rag_system.query(query, param=fast_query_params)
```

3. **Implement Connection Pooling:**
```python
from lightrag_integration import ConnectionPool

# Set up connection pooling for OpenAI API
openai_pool = ConnectionPool(
    service="openai",
    max_connections=20,
    timeout_seconds=30
)

# Use pooled connections
async def optimized_query(query: str):
    async with openai_pool.get_connection() as conn:
        return await rag_system.query(query, connection=conn)
```

### Issue 5: High Memory Usage

**Symptoms:**
- System running out of memory
- Gradual memory increase over time
- Application crashes with memory errors

**Memory Diagnostic Script:**

```python
# diagnose_memory.py
import psutil
import gc
from lightrag_integration import MemoryProfiler

def diagnose_memory_issues():
    """Diagnose memory usage issues"""
    
    print("🧠 Diagnosing memory usage...")
    
    profiler = MemoryProfiler()
    
    # Current memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"\\nCurrent memory usage:")
    print(f"   RSS (Resident Set Size): {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"   VMS (Virtual Memory Size): {memory_info.vms / 1024 / 1024:.1f} MB")
    
    # System memory
    system_memory = psutil.virtual_memory()
    print(f"\\nSystem memory:")
    print(f"   Total: {system_memory.total / 1024 / 1024 / 1024:.1f} GB")
    print(f"   Available: {system_memory.available / 1024 / 1024 / 1024:.1f} GB")
    print(f"   Used: {system_memory.percent}%")
    
    # Memory recommendations
    if memory_info.rss > 2 * 1024 * 1024 * 1024:  # 2GB
        print("❌ High memory usage (>2GB)")
        print("   Recommendations:")
        print("   • Reduce LIGHTRAG_CHUNK_SIZE")
        print("   • Reduce cache size")
        print("   • Implement memory cleanup")
    
    # Check for memory leaks
    print("\\n🔍 Checking for potential memory leaks...")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"   Garbage collected: {collected} objects")
    
    # Check unreachable objects
    if hasattr(gc, 'get_stats'):
        stats = gc.get_stats()
        print(f"   GC statistics: {stats}")

if __name__ == "__main__":
    diagnose_memory_issues()
```

**Memory Optimization Solutions:**

1. **Reduce Memory Footprint:**
```bash
# Reduce memory usage through environment variables
export LIGHTRAG_CHUNK_SIZE=600          # Reduce from 1200
export LIGHTRAG_CHUNK_OVERLAP=50        # Reduce from 100
export LIGHTRAG_MAX_TOKENS=4000         # Reduce from 8000
export LIGHTRAG_CACHE_SIZE_MB=128       # Reduce cache size
```

2. **Implement Memory Cleanup:**
```python
import gc
from lightrag_integration import MemoryManager

class MemoryOptimizedRAG:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.query_count = 0
        
    async def query_with_cleanup(self, query: str):
        """Query with automatic memory cleanup"""
        
        try:
            response = await self.rag_system.query(query)
            
            self.query_count += 1
            
            # Cleanup every 10 queries
            if self.query_count % 10 == 0:
                await self.memory_manager.cleanup()
                gc.collect()
            
            return response
            
        finally:
            # Clean up large objects
            if hasattr(self, '_last_response_data'):
                del self._last_response_data
```

---

## API and Connectivity Issues

### Issue 6: OpenAI API Errors

**Symptoms:**
- "API key invalid" errors
- Rate limit exceeded errors
- Connection timeout errors

**API Diagnostic Script:**

```python
# diagnose_api.py
import asyncio
import openai
from lightrag_integration import APIHealthChecker

async def diagnose_api_issues():
    """Diagnose OpenAI API connectivity issues"""
    
    print("🔗 Diagnosing OpenAI API issues...")
    
    health_checker = APIHealthChecker()
    
    # 1. Check API key validity
    print("\\n1. Checking API key validity...")
    try:
        api_key_status = await health_checker.validate_api_key()
        if api_key_status.valid:
            print("✅ API key is valid")
        else:
            print("❌ API key is invalid")
            print("   Check your OPENAI_API_KEY environment variable")
    except Exception as e:
        print(f"❌ API key validation error: {e}")
    
    # 2. Check API connectivity
    print("\\n2. Testing API connectivity...")
    try:
        connectivity_test = await health_checker.test_connectivity()
        print(f"   Response time: {connectivity_test.response_time:.2f}s")
        
        if connectivity_test.success:
            print("✅ API connectivity working")
        else:
            print(f"❌ API connectivity failed: {connectivity_test.error}")
    except Exception as e:
        print(f"❌ Connectivity test failed: {e}")
    
    # 3. Check rate limits
    print("\\n3. Checking rate limits...")
    try:
        rate_limit_info = await health_checker.check_rate_limits()
        
        print(f"   Requests per minute: {rate_limit_info.requests_remaining}/{rate_limit_info.requests_limit}")
        print(f"   Tokens per minute: {rate_limit_info.tokens_remaining}/{rate_limit_info.tokens_limit}")
        
        if rate_limit_info.requests_remaining < 10:
            print("⚠️  Low request rate limit remaining")
        
        if rate_limit_info.tokens_remaining < 1000:
            print("⚠️  Low token rate limit remaining")
        
    except Exception as e:
        print(f"❌ Rate limit check failed: {e}")
    
    # 4. Test API endpoints
    print("\\n4. Testing API endpoints...")
    test_endpoints = [
        ("chat/completions", "GPT model test"),
        ("embeddings", "Embedding model test")
    ]
    
    for endpoint, description in test_endpoints:
        try:
            endpoint_status = await health_checker.test_endpoint(endpoint)
            if endpoint_status.working:
                print(f"   ✅ {description}: Working")
            else:
                print(f"   ❌ {description}: Failed - {endpoint_status.error}")
        except Exception as e:
            print(f"   ❌ {description}: Error - {e}")

if __name__ == "__main__":
    asyncio.run(diagnose_api_issues())
```

**API Issue Solutions:**

1. **API Key Issues:**
```bash
# Verify API key format
echo $OPENAI_API_KEY | grep -E '^sk-[A-Za-z0-9]{48}$'

# Test API key directly
curl -H "Authorization: Bearer $OPENAI_API_KEY" \\
     -H "Content-Type: application/json" \\
     https://api.openai.com/v1/models
```

2. **Rate Limit Handling:**
```python
from lightrag_integration import RateLimitHandler

# Implement rate limit handling
rate_limiter = RateLimitHandler(
    requests_per_minute=60,
    tokens_per_minute=40000,
    retry_on_limit=True,
    max_retries=3
)

async def api_call_with_rate_limiting(prompt: str):
    async with rate_limiter.throttle():
        return await openai_api_call(prompt)
```

3. **Connection Timeout Handling:**
```python
from lightrag_integration import TimeoutHandler

timeout_handler = TimeoutHandler(
    connect_timeout=30,
    read_timeout=60,
    total_timeout=120
)

async def robust_api_call(prompt: str):
    try:
        async with timeout_handler.timeout_context():
            return await openai_api_call(prompt)
    except TimeoutError:
        # Implement fallback or retry logic
        return await fallback_response_generator(prompt)
```

---

## Emergency Procedures

### Emergency Procedure 1: Complete System Rollback

```python
# emergency_rollback.py
import asyncio
from lightrag_integration import EmergencyRollback

async def emergency_system_rollback(reason: str):
    """Execute emergency rollback to Perplexity-only system"""
    
    print(f"🚨 EXECUTING EMERGENCY ROLLBACK: {reason}")
    
    rollback = EmergencyRollback()
    
    try:
        # 1. Disable LightRAG feature flags
        print("1. Disabling LightRAG feature flags...")
        await rollback.disable_all_lightrag_features()
        
        # 2. Clear response cache
        print("2. Clearing response cache...")
        await rollback.clear_all_caches()
        
        # 3. Stop LightRAG processing
        print("3. Stopping LightRAG processing...")
        await rollback.stop_lightrag_processing()
        
        # 4. Switch to Perplexity-only mode
        print("4. Switching to Perplexity-only mode...")
        await rollback.activate_perplexity_only_mode()
        
        # 5. Send notifications
        print("5. Sending emergency notifications...")
        await rollback.send_emergency_notifications(reason)
        
        # 6. Create incident report
        print("6. Creating incident report...")
        incident_id = await rollback.create_incident_report(reason)
        
        print(f"✅ Emergency rollback completed. Incident ID: {incident_id}")
        
    except Exception as e:
        print(f"❌ Emergency rollback failed: {e}")
        # Fallback to manual intervention required
        print("🚨 MANUAL INTERVENTION REQUIRED")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        reason = " ".join(sys.argv[1:])
    else:
        reason = "Manual emergency rollback"
    
    asyncio.run(emergency_system_rollback(reason))
```

### Emergency Procedure 2: Circuit Breaker Override

```python
# emergency_circuit_override.py
from lightrag_integration import CircuitBreakerManager

def emergency_circuit_breaker_override(action: str):
    """Emergency circuit breaker override"""
    
    print(f"🚨 EMERGENCY CIRCUIT BREAKER OVERRIDE: {action}")
    
    circuit_manager = CircuitBreakerManager()
    
    if action.lower() == "open":
        # Force open all circuit breakers (disable LightRAG)
        circuit_manager.force_open_all_circuits()
        print("✅ All circuit breakers forced open - LightRAG disabled")
        
    elif action.lower() == "close":
        # Force close all circuit breakers (enable LightRAG)
        circuit_manager.force_close_all_circuits()
        print("✅ All circuit breakers forced closed - LightRAG enabled")
        
    elif action.lower() == "reset":
        # Reset all circuit breakers to normal operation
        circuit_manager.reset_all_circuits()
        print("✅ All circuit breakers reset to normal operation")
        
    else:
        print("❌ Invalid action. Use: open, close, or reset")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = input("Enter action (open/close/reset): ")
    
    emergency_circuit_breaker_override(action)
```

### Health Check Script for Monitoring

```python
# health_check.py
import asyncio
import json
from lightrag_integration import ComprehensiveHealthChecker

async def comprehensive_health_check():
    """Comprehensive system health check for monitoring"""
    
    health_checker = ComprehensiveHealthChecker()
    
    health_status = {
        "timestamp": time.time(),
        "overall_status": "unknown",
        "components": {}
    }
    
    # Check all components
    components = [
        "lightrag_system",
        "openai_api",
        "feature_flags",
        "quality_assessment",
        "budget_manager",
        "response_cache",
        "knowledge_base"
    ]
    
    healthy_components = 0
    total_components = len(components)
    
    for component in components:
        try:
            component_health = await health_checker.check_component_health(component)
            health_status["components"][component] = {
                "status": "healthy" if component_health.is_healthy else "unhealthy",
                "details": component_health.details,
                "last_check": component_health.timestamp
            }
            
            if component_health.is_healthy:
                healthy_components += 1
                
        except Exception as e:
            health_status["components"][component] = {
                "status": "error",
                "error": str(e),
                "last_check": time.time()
            }
    
    # Overall status
    if healthy_components == total_components:
        health_status["overall_status"] = "healthy"
    elif healthy_components >= total_components * 0.8:
        health_status["overall_status"] = "degraded"
    else:
        health_status["overall_status"] = "unhealthy"
    
    # Output for monitoring systems
    print(json.dumps(health_status, indent=2))
    
    # Exit code for monitoring
    if health_status["overall_status"] == "healthy":
        return 0
    elif health_status["overall_status"] == "degraded":
        return 1
    else:
        return 2

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(comprehensive_health_check())
    sys.exit(exit_code)
```

### Monitoring Integration

```bash
#!/bin/bash
# monitor_integration.sh

# Run health check and handle results
python health_check.py > /tmp/health_status.json
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ System healthy"
elif [ $exit_code -eq 1 ]; then
    echo "⚠️  System degraded - sending warning"
    # Send warning notification
    curl -X POST $SLACK_WEBHOOK -d '{"text":"CMO LightRAG system degraded"}'
elif [ $exit_code -eq 2 ]; then
    echo "❌ System unhealthy - executing emergency procedures"
    # Send critical alert
    curl -X POST $SLACK_WEBHOOK -d '{"text":"🚨 CMO LightRAG system unhealthy - manual intervention required"}'
    
    # Optional: Automatic rollback
    # python emergency_rollback.py "Automatic rollback due to health check failure"
fi
```

This troubleshooting guide provides comprehensive diagnostic tools and solutions for common integration issues. Each diagnostic script can be run independently to identify and resolve specific problems with the LightRAG integration.