# LightRAG Integration Troubleshooting Guide

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Common Integration Issues](#common-integration-issues)
3. [Runtime Issues](#runtime-issues)
4. [System Monitoring and Health Checks](#system-monitoring-and-health-checks)
5. [Debugging Procedures](#debugging-procedures)
6. [Emergency Procedures](#emergency-procedures)
7. [Performance Troubleshooting](#performance-troubleshooting)
8. [Diagnostic Scripts and Tools](#diagnostic-scripts-and-tools)

---

## Quick Reference

### Essential Environment Variables Check
```bash
# Quick environment verification
echo "OPENAI_API_KEY: $(echo $OPENAI_API_KEY | head -c 10)..."
echo "LIGHTRAG_INTEGRATION_ENABLED: $LIGHTRAG_INTEGRATION_ENABLED"
echo "LIGHTRAG_ROLLOUT_PERCENTAGE: $LIGHTRAG_ROLLOUT_PERCENTAGE"
echo "LIGHTRAG_WORKING_DIR: $LIGHTRAG_WORKING_DIR"
```

### Emergency Disable Commands
```bash
# Disable LightRAG integration immediately
export LIGHTRAG_INTEGRATION_ENABLED=false
export LIGHTRAG_ROLLOUT_PERCENTAGE=0.0

# Force fallback to Perplexity
export LIGHTRAG_FORCE_USER_COHORT=perplexity
```

### Health Check Endpoint
```python
# Quick health check
python -c "
from lightrag_integration import ClinicalMetabolomicsRAG
from lightrag_integration.config import LightRAGConfig

try:
    config = LightRAGConfig.get_config()
    print(f'✓ Configuration loaded successfully')
    print(f'✓ Working directory: {config.working_dir}')
    print(f'✓ API key present: {bool(config.api_key)}')
except Exception as e:
    print(f'✗ Health check failed: {e}')
"
```

---

## Common Integration Issues

### 1. Installation and Dependency Problems

#### Issue: LightRAG module not found
```
ModuleNotFoundError: No module named 'lightrag'
```

**Diagnosis:**
```bash
# Check if LightRAG is installed
pip list | grep -i lightrag
python -c "import lightrag; print(lightrag.__version__)"
```

**Resolution:**
```bash
# Install LightRAG
pip install lightrag

# Or install from requirements
pip install -r requirements.txt

# Verify installation
python -c "from lightrag import LightRAG; print('✓ LightRAG installed')"
```

#### Issue: Version compatibility problems
```
AttributeError: 'LightRAG' object has no attribute 'query'
```

**Diagnosis:**
```bash
pip show lightrag
python -c "import lightrag; print(dir(lightrag.LightRAG))"
```

**Resolution:**
```bash
# Update to compatible version
pip install lightrag>=0.9.0
# Or pin specific version
pip install lightrag==0.9.0
```

### 2. Configuration Errors and Validation Failures

#### Issue: API key validation failure
```
LightRAGConfigError: API key is required and cannot be empty
```

**Diagnosis:**
```bash
# Check API key environment variable
echo "API Key length: $(echo -n $OPENAI_API_KEY | wc -c)"
python -c "
import os
key = os.getenv('OPENAI_API_KEY')
if key:
    print(f'API key starts with: {key[:10]}...')
    print(f'API key length: {len(key)}')
else:
    print('API key not set')
"
```

**Resolution:**
```bash
# Set API key
export OPENAI_API_KEY="your-api-key-here"

# Persist in shell profile
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# Verify
python -c "
from lightrag_integration.config import LightRAGConfig
config = LightRAGConfig.get_config()
print('✓ API key validated')
"
```

#### Issue: Directory creation failures
```
OSError: [Errno 13] Permission denied: '/some/path/lightrag'
```

**Diagnosis:**
```bash
# Check directory permissions
ls -la $(dirname "$LIGHTRAG_WORKING_DIR")
id
umask
```

**Resolution:**
```bash
# Set writable working directory
export LIGHTRAG_WORKING_DIR="$HOME/.lightrag"
mkdir -p "$LIGHTRAG_WORKING_DIR"
chmod 755 "$LIGHTRAG_WORKING_DIR"

# Or use temporary directory
export LIGHTRAG_WORKING_DIR="/tmp/lightrag_$(whoami)"
```

#### Issue: Configuration parameter validation
```
LightRAGConfigError: max_async must be positive
```

**Diagnosis:**
```python
from lightrag_integration.config import LightRAGConfig
import os

# Check all config values
for key, value in os.environ.items():
    if key.startswith('LIGHTRAG_'):
        print(f"{key}: {value}")
```

**Resolution:**
```bash
# Fix numeric parameters
export LIGHTRAG_MAX_ASYNC=16
export LIGHTRAG_MAX_TOKENS=32768
export LIGHTRAG_ROLLOUT_PERCENTAGE=10.0

# Validate configuration
python -c "
from lightrag_integration.config import LightRAGConfig
config = LightRAGConfig.get_config(validate_config=True)
print('✓ Configuration valid')
"
```

### 3. API Connectivity Issues

#### Issue: OpenAI API connection failures
```
openai.APIConnectionError: Connection error
```

**Diagnosis:**
```python
import openai
import requests

# Test OpenAI API connectivity
try:
    client = openai.OpenAI()
    response = client.models.list()
    print("✓ OpenAI API accessible")
except Exception as e:
    print(f"✗ OpenAI API error: {e}")

# Test network connectivity
try:
    response = requests.get("https://api.openai.com", timeout=10)
    print(f"✓ Network connectivity: {response.status_code}")
except Exception as e:
    print(f"✗ Network error: {e}")
```

**Resolution:**
```bash
# Check network connectivity
curl -I https://api.openai.com

# Test with different DNS
export PYTHONHTTPSVERIFY=0  # Only for testing
nslookup api.openai.com

# Set proxy if needed
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="https://proxy.example.com:8080"
```

#### Issue: Rate limiting and quota exceeded
```
openai.RateLimitError: Rate limit exceeded
```

**Diagnosis:**
```python
from lightrag_integration.budget_manager import BudgetManager

# Check current usage
budget_manager = BudgetManager()
status = budget_manager.get_budget_status()
print(f"Daily usage: ${status['daily_cost']:.2f}")
print(f"Monthly usage: ${status['monthly_cost']:.2f}")
```

**Resolution:**
```bash
# Enable rate limiting
export LIGHTRAG_ENABLE_COST_TRACKING=true
export LIGHTRAG_DAILY_BUDGET_LIMIT=50.0
export LIGHTRAG_MONTHLY_BUDGET_LIMIT=500.0

# Reduce concurrent requests
export LIGHTRAG_MAX_ASYNC=4
```

### 4. Knowledge Base Initialization Problems

#### Issue: Knowledge base directory not found
```
FileNotFoundError: Papers directory not found: /path/to/papers
```

**Diagnosis:**
```bash
# Check directory structure
ls -la "$LIGHTRAG_WORKING_DIR"
find "$LIGHTRAG_WORKING_DIR" -name "*.pdf" -o -name "*.txt"

# Check initialization logs
tail -f logs/lightrag_integration.log | grep "knowledge_base"
```

**Resolution:**
```python
from lightrag_integration import ClinicalMetabolomicsRAG
import tempfile

# Initialize with empty knowledge base
with tempfile.TemporaryDirectory() as temp_dir:
    papers_dir = Path(temp_dir) / "papers"
    papers_dir.mkdir(exist_ok=True)
    
    rag = ClinicalMetabolomicsRAG()
    rag.initialize_knowledge_base(papers_dir)
```

#### Issue: PDF processing failures
```
PdfProcessingError: Failed to extract text from PDF
```

**Diagnosis:**
```python
from lightrag_integration.pdf_processor import PDFProcessor
import logging

logging.basicConfig(level=logging.DEBUG)
processor = PDFProcessor()

# Test specific PDF
try:
    result = processor.process_pdf("path/to/problem.pdf")
    print(f"✓ PDF processed: {len(result['text'])} chars")
except Exception as e:
    print(f"✗ PDF processing error: {e}")
```

**Resolution:**
```bash
# Install additional PDF dependencies
pip install pymupdf
pip install pdfplumber

# Check PDF file integrity
file path/to/problem.pdf
pdfinfo path/to/problem.pdf
```

### 5. Routing and Fallback Issues

#### Issue: Feature flags not working
```
User cohort assignment failed: hash computation error
```

**Diagnosis:**
```python
from lightrag_integration.feature_flag_manager import FeatureFlagManager
from lightrag_integration.config import LightRAGConfig

config = LightRAGConfig.get_config()
flag_manager = FeatureFlagManager(config)

# Test routing decision
context = {"user_id": "test_user", "session_id": "test_session"}
result = flag_manager.should_use_lightrag(context)
print(f"Routing decision: {result}")
```

**Resolution:**
```bash
# Check feature flag configuration
export LIGHTRAG_INTEGRATION_ENABLED=true
export LIGHTRAG_ROLLOUT_PERCENTAGE=50.0
export LIGHTRAG_USER_HASH_SALT="cmo_lightrag_2025"

# Force specific cohort for testing
export LIGHTRAG_FORCE_USER_COHORT=lightrag  # or perplexity
```

#### Issue: Circuit breaker blocking requests
```
CircuitBreakerOpen: LightRAG integration circuit breaker is open
```

**Diagnosis:**
```python
from lightrag_integration.feature_flag_manager import FeatureFlagManager

flag_manager = FeatureFlagManager.get_instance()
breaker_status = flag_manager.get_circuit_breaker_status()
print(f"Circuit breaker state: {breaker_status}")
```

**Resolution:**
```python
# Reset circuit breaker
flag_manager.reset_circuit_breaker()

# Or adjust thresholds
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
export LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=600
```

---

## Runtime Issues

### 1. Query Processing Errors

#### Issue: Query timeout errors
```
asyncio.TimeoutError: Query processing timeout after 30s
```

**Diagnosis:**
```python
import asyncio
import time
from lightrag_integration import ClinicalMetabolomicsRAG

async def test_query_performance():
    rag = ClinicalMetabolomicsRAG()
    start_time = time.time()
    
    try:
        result = await rag.aquery("test query", timeout=10)
        elapsed = time.time() - start_time
        print(f"✓ Query completed in {elapsed:.2f}s")
    except asyncio.TimeoutError:
        print(f"✗ Query timeout after {time.time() - start_time:.2f}s")

asyncio.run(test_query_performance())
```

**Resolution:**
```bash
# Increase timeout
export LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS=60.0

# Optimize query parameters
export LIGHTRAG_MAX_ASYNC=8
export LIGHTRAG_MAX_TOKENS=16384
```

#### Issue: Memory exhaustion during processing
```
MemoryError: Unable to allocate memory
```

**Diagnosis:**
```python
import psutil
import os

def check_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"RSS Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS Memory: {memory_info.vms / 1024 / 1024:.2f} MB")
    print(f"System Memory: {psutil.virtual_memory().percent}%")

check_memory_usage()
```

**Resolution:**
```bash
# Reduce memory usage
export LIGHTRAG_MAX_ASYNC=4
export LIGHTRAG_MAX_TOKENS=8192

# Enable batch processing
export LIGHTRAG_BATCH_SIZE=5
export LIGHTRAG_MAX_MEMORY_MB=2048
```

### 2. Performance Problems

#### Issue: Slow response times
```
WARNING: Query processing taking longer than expected: 45.2s
```

**Diagnosis:**
```python
from lightrag_integration.performance_benchmarking import QualityPerformanceBenchmarks

benchmarks = QualityPerformanceBenchmarks()
results = benchmarks.run_performance_benchmark()
print(f"Average response time: {results['avg_response_time']:.2f}s")
print(f"P95 response time: {results['p95_response_time']:.2f}s")
```

**Resolution:**
```bash
# Enable performance monitoring
export LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true
export LIGHTRAG_ENABLE_QUALITY_METRICS=true

# Optimize model settings
export LIGHTRAG_MODEL="gpt-4o-mini"  # Faster model
export LIGHTRAG_EMBEDDING_MODEL="text-embedding-3-small"
```

### 3. Database Connectivity Problems

#### Issue: Cost tracking database errors
```
sqlite3.OperationalError: database is locked
```

**Diagnosis:**
```python
from lightrag_integration.cost_persistence import CostPersistence
import sqlite3

# Check database status
db_path = "cost_tracking.db"
try:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("SELECT COUNT(*) FROM cost_entries")
    print("✓ Database accessible")
    conn.close()
except Exception as e:
    print(f"✗ Database error: {e}")
```

**Resolution:**
```bash
# Move database to writable location
export LIGHTRAG_COST_DB_PATH="$HOME/.lightrag/cost_tracking.db"

# Or disable cost tracking temporarily
export LIGHTRAG_ENABLE_COST_TRACKING=false
```

### 4. Feature Flag and A/B Testing Issues

#### Issue: Inconsistent user cohort assignment
```
WARNING: User cohort changed between sessions
```

**Diagnosis:**
```python
from lightrag_integration.feature_flag_manager import FeatureFlagManager

flag_manager = FeatureFlagManager.get_instance()

# Test consistency
user_id = "test_user_123"
cohorts = []
for i in range(10):
    context = {"user_id": user_id, "session_id": f"session_{i}"}
    result = flag_manager.should_use_lightrag(context)
    cohorts.append(result.user_cohort)

print(f"Cohort consistency: {len(set(cohorts)) == 1}")
```

**Resolution:**
```bash
# Ensure consistent salt
export LIGHTRAG_USER_HASH_SALT="cmo_lightrag_2025"

# Check rollout percentage
export LIGHTRAG_ROLLOUT_PERCENTAGE=50.0
```

---

## System Monitoring and Health Checks

### 1. Health Check Implementation

```python
#!/usr/bin/env python3
"""
LightRAG Integration Health Check Script
Usage: python health_check.py
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from lightrag_integration import ClinicalMetabolomicsRAG
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.feature_flag_manager import FeatureFlagManager


async def comprehensive_health_check():
    """Comprehensive health check for LightRAG integration."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "checks": {}
    }
    
    # 1. Configuration Check
    try:
        config = LightRAGConfig.get_config(validate_config=True)
        results["checks"]["configuration"] = {
            "status": "pass",
            "working_dir": str(config.working_dir),
            "api_key_present": bool(config.api_key),
            "integration_enabled": config.lightrag_integration_enabled
        }
    except Exception as e:
        results["checks"]["configuration"] = {
            "status": "fail",
            "error": str(e)
        }
    
    # 2. Directory Check
    try:
        config.ensure_directories()
        results["checks"]["directories"] = {
            "status": "pass",
            "working_dir_exists": config.working_dir.exists(),
            "graph_storage_exists": config.graph_storage_dir.exists()
        }
    except Exception as e:
        results["checks"]["directories"] = {
            "status": "fail",
            "error": str(e)
        }
    
    # 3. Feature Flag Check
    try:
        flag_manager = FeatureFlagManager(config)
        test_result = flag_manager.should_use_lightrag({
            "user_id": "health_check",
            "session_id": "health_check_session"
        })
        results["checks"]["feature_flags"] = {
            "status": "pass",
            "routing_decision": test_result.decision.value,
            "circuit_breaker_state": test_result.circuit_breaker_state
        }
    except Exception as e:
        results["checks"]["feature_flags"] = {
            "status": "fail",
            "error": str(e)
        }
    
    # 4. RAG Initialization Check
    try:
        rag = ClinicalMetabolomicsRAG(config=config)
        results["checks"]["rag_initialization"] = {
            "status": "pass",
            "instance_created": True
        }
    except Exception as e:
        results["checks"]["rag_initialization"] = {
            "status": "fail",
            "error": str(e)
        }
    
    # 5. Simple Query Test
    try:
        test_query = "What is metabolomics?"
        start_time = time.time()
        response = await rag.aquery(test_query, mode="naive")
        query_time = time.time() - start_time
        
        results["checks"]["query_processing"] = {
            "status": "pass",
            "query_time": query_time,
            "response_length": len(response) if response else 0
        }
    except Exception as e:
        results["checks"]["query_processing"] = {
            "status": "fail",
            "error": str(e)
        }
    
    # Determine overall status
    failed_checks = [name for name, check in results["checks"].items() 
                    if check["status"] == "fail"]
    
    if not failed_checks:
        results["status"] = "healthy"
    elif len(failed_checks) == len(results["checks"]):
        results["status"] = "critical"
    else:
        results["status"] = "degraded"
    
    results["failed_checks"] = failed_checks
    
    return results


if __name__ == "__main__":
    async def main():
        print("🔍 Running LightRAG Integration Health Check...")
        results = await comprehensive_health_check()
        
        print(f"\n📊 Health Check Results ({results['timestamp']})")
        print(f"Overall Status: {results['status'].upper()}")
        
        for check_name, check_result in results["checks"].items():
            status_icon = "✅" if check_result["status"] == "pass" else "❌"
            print(f"{status_icon} {check_name.replace('_', ' ').title()}: {check_result['status']}")
            
            if check_result["status"] == "fail":
                print(f"   Error: {check_result.get('error', 'Unknown error')}")
        
        # Save results to file
        output_file = f"health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Detailed results saved to: {output_file}")
        
        return 0 if results["status"] == "healthy" else 1
    
    exit(asyncio.run(main()))
```

### 2. Key Metrics to Monitor

#### System Metrics
```bash
# Monitor these metrics continuously:

# 1. Response Time Distribution
grep "Query processed successfully" logs/lightrag_integration.log | \
    grep -o "[0-9.]*s" | sed 's/s//' | sort -n

# 2. Error Rate
grep "ERROR" logs/lightrag_integration.log | wc -l

# 3. Memory Usage
ps aux | grep python | grep lightrag | awk '{print $6}'

# 4. Disk Usage
du -sh "$LIGHTRAG_WORKING_DIR"

# 5. API Cost Tracking
python -c "
from lightrag_integration.budget_manager import BudgetManager
bm = BudgetManager()
status = bm.get_budget_status()
print(f'Daily: \${status[\"daily_cost\"]:.2f}')
print(f'Monthly: \${status[\"monthly_cost\"]:.2f}')
"
```

#### Performance Thresholds
```python
# Configure alerting thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time_p95": 30.0,  # seconds
    "error_rate": 0.05,         # 5%
    "memory_usage_mb": 2048,    # MB
    "daily_cost": 50.0,         # USD
    "circuit_breaker_failures": 3
}
```

### 3. Automated Monitoring Script

```python
#!/usr/bin/env python3
"""
Continuous monitoring script for LightRAG integration.
"""

import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path

from lightrag_integration.budget_manager import BudgetManager
from lightrag_integration.feature_flag_manager import FeatureFlagManager


class LightRAGMonitor:
    def __init__(self, check_interval=300):  # 5 minutes
        self.check_interval = check_interval
        self.metrics = []
        self.alerts = []
        
    def collect_metrics(self):
        """Collect current system metrics."""
        timestamp = datetime.now()
        
        # System metrics
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Budget metrics
        try:
            budget_manager = BudgetManager()
            budget_status = budget_manager.get_budget_status()
            daily_cost = budget_status.get("daily_cost", 0)
            monthly_cost = budget_status.get("monthly_cost", 0)
        except Exception:
            daily_cost = monthly_cost = 0
        
        # Feature flag metrics
        try:
            flag_manager = FeatureFlagManager.get_instance()
            performance_metrics = flag_manager.get_performance_metrics()
            error_rate = performance_metrics.get("error_rate", 0)
            avg_response_time = performance_metrics.get("avg_response_time", 0)
        except Exception:
            error_rate = avg_response_time = 0
        
        metric = {
            "timestamp": timestamp.isoformat(),
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time
        }
        
        self.metrics.append(metric)
        
        # Keep only last 24 hours of metrics
        cutoff = timestamp - timedelta(days=1)
        self.metrics = [m for m in self.metrics 
                       if datetime.fromisoformat(m["timestamp"]) > cutoff]
        
        return metric
    
    def check_thresholds(self, metric):
        """Check if any thresholds are exceeded."""
        alerts = []
        
        if metric["memory_mb"] > 2048:
            alerts.append({
                "type": "memory",
                "severity": "warning",
                "message": f"High memory usage: {metric['memory_mb']:.1f} MB"
            })
        
        if metric["error_rate"] > 0.1:
            alerts.append({
                "type": "error_rate",
                "severity": "critical",
                "message": f"High error rate: {metric['error_rate']:.2%}"
            })
        
        if metric["avg_response_time"] > 30:
            alerts.append({
                "type": "response_time",
                "severity": "warning",
                "message": f"Slow response time: {metric['avg_response_time']:.1f}s"
            })
        
        if metric["daily_cost"] > 50:
            alerts.append({
                "type": "cost",
                "severity": "warning",
                "message": f"High daily cost: ${metric['daily_cost']:.2f}"
            })
        
        return alerts
    
    async def monitor_loop(self):
        """Main monitoring loop."""
        logging.info("Starting LightRAG monitoring...")
        
        while True:
            try:
                # Collect metrics
                metric = self.collect_metrics()
                
                # Check thresholds
                alerts = self.check_thresholds(metric)
                
                # Log metrics
                logging.info(f"Metrics: Memory={metric['memory_mb']:.1f}MB, "
                           f"Cost=${metric['daily_cost']:.2f}, "
                           f"ErrorRate={metric['error_rate']:.2%}, "
                           f"ResponseTime={metric['avg_response_time']:.1f}s")
                
                # Handle alerts
                for alert in alerts:
                    logging.warning(f"ALERT [{alert['severity']}] {alert['message']}")
                    self.alerts.append({
                        **alert,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Save metrics to file
                with open("lightrag_metrics.json", "w") as f:
                    json.dump({
                        "metrics": self.metrics[-100:],  # Last 100 metrics
                        "alerts": self.alerts[-50:]      # Last 50 alerts
                    }, f, indent=2)
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
            
            await asyncio.sleep(self.check_interval)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    monitor = LightRAGMonitor(check_interval=300)  # 5 minutes
    asyncio.run(monitor.monitor_loop())
```

---

## Debugging Procedures

### 1. Step-by-step Debugging Workflow

#### Phase 1: Initial Assessment
```bash
# 1. Check system status
systemctl status lightrag  # If running as service
ps aux | grep lightrag

# 2. Check recent logs
tail -100 logs/lightrag_integration.log
tail -100 logs/structured_logs.jsonl

# 3. Verify environment
env | grep LIGHTRAG | sort
env | grep OPENAI
```

#### Phase 2: Configuration Debugging
```python
# debug_config.py
from lightrag_integration.config import LightRAGConfig
import json

try:
    config = LightRAGConfig.get_config(validate_config=False)
    print("✓ Configuration loaded")
    
    # Export configuration for inspection
    config_dict = config.to_dict()
    with open("debug_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print("Configuration exported to debug_config.json")
    
    # Validate configuration
    config.validate()
    print("✓ Configuration valid")
    
except Exception as e:
    print(f"✗ Configuration error: {e}")
    import traceback
    traceback.print_exc()
```

#### Phase 3: Component-by-Component Testing
```python
# debug_components.py
import asyncio
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.feature_flag_manager import FeatureFlagManager
from lightrag_integration import ClinicalMetabolomicsRAG

async def debug_components():
    print("🔍 Testing LightRAG Integration Components\n")
    
    # Test 1: Configuration
    print("1. Configuration Test")
    try:
        config = LightRAGConfig.get_config()
        print("   ✓ Configuration loaded")
        print(f"   - Working directory: {config.working_dir}")
        print(f"   - Integration enabled: {config.lightrag_integration_enabled}")
    except Exception as e:
        print(f"   ✗ Configuration failed: {e}")
        return
    
    # Test 2: Feature Flags
    print("\n2. Feature Flag Test")
    try:
        flag_manager = FeatureFlagManager(config)
        result = flag_manager.should_use_lightrag({
            "user_id": "debug_user",
            "session_id": "debug_session"
        })
        print(f"   ✓ Feature flags working")
        print(f"   - Decision: {result.decision}")
        print(f"   - Reason: {result.reason}")
    except Exception as e:
        print(f"   ✗ Feature flags failed: {e}")
    
    # Test 3: RAG Initialization
    print("\n3. RAG Initialization Test")
    try:
        rag = ClinicalMetabolomicsRAG(config=config)
        print("   ✓ RAG initialized")
    except Exception as e:
        print(f"   ✗ RAG initialization failed: {e}")
        return
    
    # Test 4: Simple Query
    print("\n4. Query Processing Test")
    try:
        response = await rag.aquery("What is metabolomics?", mode="naive")
        print(f"   ✓ Query processed")
        print(f"   - Response length: {len(response) if response else 0}")
    except Exception as e:
        print(f"   ✗ Query processing failed: {e}")
    
    print("\n🎉 Component debugging complete")

if __name__ == "__main__":
    asyncio.run(debug_components())
```

### 2. Log Analysis Techniques

#### Parse and Analyze Logs
```bash
#!/bin/bash
# log_analysis.sh

LOG_FILE="logs/lightrag_integration.log"
STRUCTURED_LOG="logs/structured_logs.jsonl"

echo "📊 LightRAG Log Analysis Report"
echo "================================"

# Basic statistics
echo "📈 Log Statistics:"
echo "- Total log entries: $(wc -l < "$LOG_FILE")"
echo "- Error entries: $(grep -c "ERROR" "$LOG_FILE")"
echo "- Warning entries: $(grep -c "WARNING" "$LOG_FILE")"
echo "- Recent activity (last hour): $(grep "$(date -d '1 hour ago' '+%Y-%m-%d %H')" "$LOG_FILE" | wc -l)"

echo -e "\n🔥 Recent Errors:"
grep "ERROR" "$LOG_FILE" | tail -5

echo -e "\n⚠️  Recent Warnings:"
grep "WARNING" "$LOG_FILE" | tail -5

echo -e "\n⏱️  Query Performance:"
grep "Query processed successfully" "$LOG_FILE" | \
    grep -o "[0-9.]*s" | sed 's/s//' | \
    awk '{sum+=$1; count++; if($1>max) max=$1} END {
        if(count>0) {
            print "- Average response time: " sum/count "s"
            print "- Maximum response time: " max "s"
            print "- Total queries processed: " count
        }
    }'

# Analyze structured logs if available
if [ -f "$STRUCTURED_LOG" ]; then
    echo -e "\n📋 Structured Log Analysis:"
    python3 -c "
import json
import sys
from collections import Counter

errors = Counter()
operations = Counter()

try:
    with open('$STRUCTURED_LOG') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('level') == 'ERROR':
                    errors[data.get('message', 'Unknown')] += 1
                operations[data.get('operation_name', 'Unknown')] += 1
            except json.JSONDecodeError:
                continue
    
    print('Top error messages:')
    for msg, count in errors.most_common(5):
        print(f'  - {msg}: {count}')
    
    print('Top operations:')
    for op, count in operations.most_common(5):
        print(f'  - {op}: {count}')
        
except FileNotFoundError:
    print('Structured log file not found')
"
fi
```

#### Extract Error Patterns
```python
# error_pattern_analyzer.py
import re
import json
from collections import Counter, defaultdict
from datetime import datetime

def analyze_error_patterns(log_file):
    """Analyze error patterns in log files."""
    
    error_patterns = Counter()
    error_timeline = defaultdict(list)
    
    # Common error patterns
    patterns = {
        "api_timeout": r"timeout|TimeoutError",
        "api_rate_limit": r"rate.?limit|RateLimitError",
        "memory_error": r"MemoryError|memory",
        "connection_error": r"Connection|ConnectionError",
        "authentication": r"auth|API key|unauthorized",
        "file_not_found": r"FileNotFoundError|not found",
        "permission_denied": r"PermissionError|permission denied",
        "circuit_breaker": r"circuit.?breaker|CircuitBreakerOpen"
    }
    
    with open(log_file) as f:
        for line in f:
            if "ERROR" in line or "CRITICAL" in line:
                # Extract timestamp
                timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
                if timestamp_match:
                    timestamp = timestamp_match.group()
                else:
                    timestamp = "unknown"
                
                # Check patterns
                for pattern_name, pattern in patterns.items():
                    if re.search(pattern, line, re.IGNORECASE):
                        error_patterns[pattern_name] += 1
                        error_timeline[pattern_name].append({
                            "timestamp": timestamp,
                            "message": line.strip()
                        })
                        break
                else:
                    # Unknown error pattern
                    error_patterns["other"] += 1
                    error_timeline["other"].append({
                        "timestamp": timestamp,
                        "message": line.strip()
                    })
    
    return dict(error_patterns), dict(error_timeline)

if __name__ == "__main__":
    patterns, timeline = analyze_error_patterns("logs/lightrag_integration.log")
    
    print("🚨 Error Pattern Analysis")
    print("=" * 50)
    
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"{pattern.replace('_', ' ').title()}: {count}")
            
            # Show recent examples
            recent = timeline[pattern][-3:]  # Last 3 occurrences
            for example in recent:
                print(f"  {example['timestamp']}: {example['message'][:100]}...")
            print()
```

### 3. Performance Profiling

#### CPU and Memory Profiling
```python
# performance_profiler.py
import asyncio
import cProfile
import pstats
import tracemalloc
from functools import wraps

def profile_performance(func):
    """Decorator to profile function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Start memory tracing
        tracemalloc.start()
        
        # Profile CPU
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = await func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Save CPU profile
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.dump_stats(f"{func.__name__}_cpu_profile.prof")
            
            # Get memory snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            print(f"\n📊 Performance Profile for {func.__name__}:")
            print("Top CPU consumers:")
            stats.print_stats(10)
            
            print("\nTop memory consumers:")
            for stat in top_stats:
                print(stat)
        
        return result
    
    return wrapper

# Usage example
from lightrag_integration import ClinicalMetabolomicsRAG

@profile_performance
async def test_query_performance():
    rag = ClinicalMetabolomicsRAG()
    return await rag.aquery("What are the main applications of metabolomics?")

if __name__ == "__main__":
    asyncio.run(test_query_performance())
```

---

## Emergency Procedures

### 1. Quick Disable LightRAG Integration

#### Immediate Disable Script
```bash
#!/bin/bash
# emergency_disable.sh

echo "🚨 Emergency LightRAG Integration Disable"
echo "========================================"

# Set emergency environment variables
export LIGHTRAG_INTEGRATION_ENABLED=false
export LIGHTRAG_ROLLOUT_PERCENTAGE=0.0
export LIGHTRAG_FORCE_USER_COHORT=perplexity

# Create emergency config file
cat > emergency_config.env << EOF
LIGHTRAG_INTEGRATION_ENABLED=false
LIGHTRAG_ROLLOUT_PERCENTAGE=0.0
LIGHTRAG_FORCE_USER_COHORT=perplexity
EOF

echo "✓ Environment variables set"
echo "✓ Emergency config created: emergency_config.env"

# Test the disable
python3 -c "
from lightrag_integration.feature_flag_manager import FeatureFlagManager
from lightrag_integration.config import LightRAGConfig

config = LightRAGConfig.get_config()
flag_manager = FeatureFlagManager(config)

result = flag_manager.should_use_lightrag({
    'user_id': 'emergency_test',
    'session_id': 'emergency_test'
})

print(f'Routing decision: {result.decision}')
print(f'Integration disabled: {result.decision.value != \"lightrag\"}')
"

echo "🎯 LightRAG integration disabled successfully"
echo "   Users will now be routed to Perplexity"
```

### 2. Rollback Procedures

#### Automated Rollback Script
```python
#!/usr/bin/env python3
# emergency_rollback.py

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

class EmergencyRollback:
    def __init__(self):
        self.backup_dir = Path("backups")
        self.config_backup = self.backup_dir / "config_backup.json"
        self.env_backup = self.backup_dir / "env_backup.json"
        
    def create_backup(self):
        """Create backup of current configuration."""
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup environment variables
        env_vars = {k: v for k, v in os.environ.items() 
                   if k.startswith(('LIGHTRAG_', 'OPENAI_'))}
        
        with open(self.env_backup, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "environment": env_vars
            }, f, indent=2)
        
        print(f"✓ Backup created: {self.env_backup}")
        
    def rollback_to_safe_config(self):
        """Rollback to known safe configuration."""
        safe_config = {
            "LIGHTRAG_INTEGRATION_ENABLED": "false",
            "LIGHTRAG_ROLLOUT_PERCENTAGE": "0.0",
            "LIGHTRAG_FORCE_USER_COHORT": "perplexity",
            "LIGHTRAG_ENABLE_CIRCUIT_BREAKER": "true",
            "LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "1"
        }
        
        # Apply safe configuration
        for key, value in safe_config.items():
            os.environ[key] = value
            print(f"✓ Set {key}={value}")
        
        # Create persistent config file
        with open("rollback_config.env", "w") as f:
            for key, value in safe_config.items():
                f.write(f"{key}={value}\n")
        
        print("✓ Safe configuration applied")
        print("✓ Rollback config saved: rollback_config.env")
        
    def verify_rollback(self):
        """Verify rollback was successful."""
        try:
            from lightrag_integration.feature_flag_manager import FeatureFlagManager
            from lightrag_integration.config import LightRAGConfig
            
            config = LightRAGConfig.get_config()
            flag_manager = FeatureFlagManager(config)
            
            # Test routing
            result = flag_manager.should_use_lightrag({
                "user_id": "rollback_test",
                "session_id": "rollback_test"
            })
            
            if result.decision.value != "lightrag":
                print("✅ Rollback verified: LightRAG disabled")
                return True
            else:
                print("❌ Rollback failed: LightRAG still active")
                return False
                
        except Exception as e:
            print(f"❌ Rollback verification error: {e}")
            return False
    
    def cleanup_problematic_state(self):
        """Clean up potentially problematic state."""
        
        # Remove circuit breaker state
        state_files = [
            "circuit_breaker_state.json",
            "feature_flag_state.json",
            "performance_metrics.json"
        ]
        
        for state_file in state_files:
            if Path(state_file).exists():
                Path(state_file).unlink()
                print(f"✓ Removed {state_file}")
        
        # Clear temporary caches
        cache_dirs = [
            ".lightrag_cache",
            "__pycache__",
            "temp_storage"
        ]
        
        for cache_dir in cache_dirs:
            if Path(cache_dir).exists():
                shutil.rmtree(cache_dir)
                print(f"✓ Cleared {cache_dir}")

if __name__ == "__main__":
    print("🚨 Emergency Rollback Procedure")
    print("===============================")
    
    rollback = EmergencyRollback()
    
    # Step 1: Create backup
    print("\n1. Creating backup...")
    rollback.create_backup()
    
    # Step 2: Apply safe configuration
    print("\n2. Applying safe configuration...")
    rollback.rollback_to_safe_config()
    
    # Step 3: Clean up problematic state
    print("\n3. Cleaning up state...")
    rollback.cleanup_problematic_state()
    
    # Step 4: Verify rollback
    print("\n4. Verifying rollback...")
    success = rollback.verify_rollback()
    
    if success:
        print("\n🎉 Emergency rollback completed successfully")
        print("   System is now in safe mode")
        print("   All users will be routed to Perplexity")
    else:
        print("\n💥 Emergency rollback failed")
        print("   Manual intervention required")
        print("   Contact system administrator")
```

### 3. System Recovery Steps

#### Recovery Checklist
```bash
#!/bin/bash
# recovery_checklist.sh

echo "🔧 LightRAG Integration Recovery Checklist"
echo "=========================================="

RECOVERY_LOG="recovery_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RECOVERY_LOG") 2>&1

echo "Recovery started at: $(date)"

# Step 1: System Health Check
echo -e "\n1. System Health Check"
echo "----------------------"

# Check disk space
df -h | grep -E "(/$|/tmp)" | awk '{print "Disk space " $1 ": " $4 " available"}'

# Check memory
free -h | grep "Mem:" | awk '{print "Memory: " $3 " used of " $2}'

# Check load
uptime | awk '{print "System load: " $(NF-2) " " $(NF-1) " " $NF}'

# Step 2: Service Status Check
echo -e "\n2. Service Status Check"
echo "-----------------------"

# Check if Python is working
python3 --version && echo "✓ Python available" || echo "✗ Python not available"

# Check if required packages are installed
python3 -c "import lightrag; print('✓ LightRAG available')" 2>/dev/null || echo "✗ LightRAG not available"
python3 -c "import openai; print('✓ OpenAI available')" 2>/dev/null || echo "✗ OpenAI not available"

# Step 3: Configuration Recovery
echo -e "\n3. Configuration Recovery"
echo "-------------------------"

# Check if backup exists
if [ -f "backups/env_backup.json" ]; then
    echo "✓ Configuration backup found"
    # Option to restore from backup
    read -p "Restore from backup? (y/N): " restore_backup
    if [ "$restore_backup" = "y" ]; then
        python3 -c "
import json
import os
with open('backups/env_backup.json') as f:
    backup = json.load(f)
    for key, value in backup['environment'].items():
        os.environ[key] = value
        print(f'Restored {key}')
"
        echo "✓ Configuration restored from backup"
    fi
else
    echo "⚠ No configuration backup found"
    echo "  Using default configuration"
    
    # Set minimum required configuration
    export LIGHTRAG_INTEGRATION_ENABLED=false
    export LIGHTRAG_ROLLOUT_PERCENTAGE=0.0
    export LIGHTRAG_FALLBACK_TO_PERPLEXITY=true
fi

# Step 4: Directory Structure Recovery
echo -e "\n4. Directory Structure Recovery"
echo "-------------------------------"

REQUIRED_DIRS=(
    "logs"
    "backups"
    "$LIGHTRAG_WORKING_DIR"
    "$(dirname $LIGHTRAG_COST_DB_PATH)"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -n "$dir" ] && [ ! -d "$dir" ]; then
        mkdir -p "$dir" && echo "✓ Created directory: $dir" || echo "✗ Failed to create: $dir"
    else
        echo "✓ Directory exists: $dir"
    fi
done

# Step 5: Permission Recovery
echo -e "\n5. Permission Recovery"
echo "----------------------"

# Fix common permission issues
find logs -type f -exec chmod 644 {} \; 2>/dev/null && echo "✓ Fixed log file permissions"
find logs -type d -exec chmod 755 {} \; 2>/dev/null && echo "✓ Fixed log directory permissions"

if [ -n "$LIGHTRAG_WORKING_DIR" ] && [ -d "$LIGHTRAG_WORKING_DIR" ]; then
    chmod -R 755 "$LIGHTRAG_WORKING_DIR" 2>/dev/null && echo "✓ Fixed LightRAG directory permissions"
fi

# Step 6: Component Testing
echo -e "\n6. Component Testing"
echo "--------------------"

# Test configuration loading
python3 -c "
from lightrag_integration.config import LightRAGConfig
try:
    config = LightRAGConfig.get_config(validate_config=False)
    print('✓ Configuration loads successfully')
except Exception as e:
    print(f'✗ Configuration error: {e}')
" 2>/dev/null

# Test basic import
python3 -c "
try:
    from lightrag_integration import ClinicalMetabolomicsRAG
    print('✓ Main module imports successfully')
except Exception as e:
    print(f'✗ Import error: {e}')
" 2>/dev/null

echo -e "\n7. Recovery Summary"
echo "-------------------"
echo "Recovery completed at: $(date)"
echo "Recovery log saved to: $RECOVERY_LOG"

# Final health check
python3 -c "
from lightrag_integration.feature_flag_manager import FeatureFlagManager
from lightrag_integration.config import LightRAGConfig

try:
    config = LightRAGConfig.get_config()
    flag_manager = FeatureFlagManager(config)
    result = flag_manager.should_use_lightrag({'user_id': 'recovery_test'})
    
    if result.decision.value != 'lightrag':
        print('✅ System recovered successfully')
        print('   Integration is safely disabled')
        print('   Users will be routed to Perplexity')
    else:
        print('⚠ System recovered with LightRAG enabled')
        print('   Monitor system carefully')
except Exception as e:
    print(f'❌ Recovery incomplete: {e}')
    print('   Manual intervention required')
"

echo -e "\nRecommendations:"
echo "- Monitor system closely for next 30 minutes"
echo "- Check error logs regularly: tail -f logs/lightrag_integration.log"
echo "- Run health check: python health_check.py"
echo "- Gradually re-enable features if needed"
```

---

## Performance Troubleshooting

### 1. Slow Query Diagnosis

#### Query Performance Analyzer
```python
# query_performance_analyzer.py
import asyncio
import time
import json
import statistics
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

from lightrag_integration import ClinicalMetabolomicsRAG
from lightrag_integration.config import LightRAGConfig

@dataclass
class QueryPerformanceResult:
    query: str
    mode: str
    response_time: float
    response_length: int
    success: bool
    error: str = None
    memory_usage: int = 0

class QueryPerformanceDiagnostic:
    def __init__(self):
        self.config = LightRAGConfig.get_config()
        self.rag = ClinicalMetabolomicsRAG(config=self.config)
        
    async def benchmark_query_modes(self, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark different query modes."""
        modes = ["naive", "local", "global", "hybrid"]
        results = []
        
        for query in test_queries:
            print(f"Testing query: {query[:50]}...")
            
            for mode in modes:
                print(f"  Mode: {mode}")
                
                try:
                    start_time = time.time()
                    response = await self.rag.aquery(query, mode=mode)
                    end_time = time.time()
                    
                    result = QueryPerformanceResult(
                        query=query,
                        mode=mode,
                        response_time=end_time - start_time,
                        response_length=len(response) if response else 0,
                        success=True
                    )
                    
                except Exception as e:
                    result = QueryPerformanceResult(
                        query=query,
                        mode=mode,
                        response_time=0,
                        response_length=0,
                        success=False,
                        error=str(e)
                    )
                
                results.append(result)
                print(f"    Time: {result.response_time:.2f}s, "
                      f"Length: {result.response_length}, "
                      f"Success: {result.success}")
        
        return self.analyze_results(results)
    
    def analyze_results(self, results: List[QueryPerformanceResult]) -> Dict[str, Any]:
        """Analyze performance results."""
        # Group by mode
        by_mode = {}
        for result in results:
            if result.mode not in by_mode:
                by_mode[result.mode] = []
            by_mode[result.mode].append(result)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "mode_analysis": {}
        }
        
        for mode, mode_results in by_mode.items():
            successful = [r for r in mode_results if r.success]
            failed = [r for r in mode_results if not r.success]
            
            if successful:
                response_times = [r.response_time for r in successful]
                response_lengths = [r.response_length for r in successful]
                
                analysis["mode_analysis"][mode] = {
                    "total_queries": len(mode_results),
                    "successful": len(successful),
                    "failed": len(failed),
                    "success_rate": len(successful) / len(mode_results),
                    "avg_response_time": statistics.mean(response_times),
                    "median_response_time": statistics.median(response_times),
                    "p95_response_time": sorted(response_times)[int(0.95 * len(response_times))],
                    "avg_response_length": statistics.mean(response_lengths),
                    "errors": [r.error for r in failed if r.error]
                }
            else:
                analysis["mode_analysis"][mode] = {
                    "total_queries": len(mode_results),
                    "successful": 0,
                    "failed": len(failed),
                    "success_rate": 0,
                    "errors": [r.error for r in failed if r.error]
                }
        
        return analysis

async def main():
    """Run query performance diagnostic."""
    print("🏃 Query Performance Diagnostic")
    print("=" * 50)
    
    diagnostic = QueryPerformanceDiagnostic()
    
    # Test queries of varying complexity
    test_queries = [
        "What is metabolomics?",  # Simple
        "How does mass spectrometry work in metabolomics studies?",  # Medium
        "Compare the advantages and limitations of targeted versus untargeted metabolomics approaches for clinical biomarker discovery.",  # Complex
        "What are the key metabolic pathways involved in diabetes progression?",  # Biomedical
        "Explain the role of machine learning in metabolomics data analysis."  # Technical
    ]
    
    results = await diagnostic.benchmark_query_modes(test_queries)
    
    # Print summary
    print("\n📊 Performance Summary:")
    print("-" * 30)
    
    for mode, stats in results["mode_analysis"].items():
        print(f"\n{mode.upper()} Mode:")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        if stats['successful'] > 0:
            print(f"  Avg response time: {stats['avg_response_time']:.2f}s")
            print(f"  P95 response time: {stats['p95_response_time']:.2f}s")
            print(f"  Avg response length: {stats['avg_response_length']:.0f} chars")
        if stats['failed'] > 0:
            print(f"  Common errors: {set(stats['errors'])}")
    
    # Save detailed results
    output_file = f"query_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Memory Usage Optimization

#### Memory Usage Monitor
```python
# memory_optimizer.py
import gc
import psutil
import tracemalloc
import asyncio
from contextlib import asynccontextmanager

class MemoryOptimizer:
    def __init__(self):
        self.process = psutil.Process()
        
    @asynccontextmanager
    async def memory_monitoring(self, operation_name: str):
        """Context manager for monitoring memory usage."""
        # Start memory tracing
        tracemalloc.start()
        
        # Get initial memory state
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"🧠 Starting {operation_name}")
        print(f"   Initial memory: {initial_memory:.1f} MB")
        
        try:
            yield self
        finally:
            # Get final memory state
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            # Get top memory consumers
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:5]
            
            print(f"   Final memory: {final_memory:.1f} MB")
            print(f"   Memory change: {memory_delta:+.1f} MB")
            
            if memory_delta > 100:  # Alert if memory increased by >100MB
                print("   ⚠️ High memory usage detected!")
                print("   Top memory consumers:")
                for i, stat in enumerate(top_stats):
                    print(f"     {i+1}. {stat}")
            
            # Force garbage collection
            collected = gc.collect()
            if collected > 0:
                print(f"   🗑️ Garbage collected: {collected} objects")
            
            tracemalloc.stop()
    
    def optimize_for_memory(self):
        """Apply memory optimizations."""
        optimizations = []
        
        # Check if we can reduce batch sizes
        import os
        current_batch = int(os.getenv("LIGHTRAG_BATCH_SIZE", "10"))
        if current_batch > 5:
            os.environ["LIGHTRAG_BATCH_SIZE"] = "5"
            optimizations.append("Reduced batch size to 5")
        
        # Reduce max async operations
        current_async = int(os.getenv("LIGHTRAG_MAX_ASYNC", "16"))
        if current_async > 8:
            os.environ["LIGHTRAG_MAX_ASYNC"] = "8"
            optimizations.append("Reduced max async to 8")
        
        # Set memory limit
        os.environ["LIGHTRAG_MAX_MEMORY_MB"] = "1024"
        optimizations.append("Set memory limit to 1024MB")
        
        return optimizations

# Usage example
async def test_memory_optimization():
    optimizer = MemoryOptimizer()
    
    # Apply optimizations
    optimizations = optimizer.optimize_for_memory()
    print("Applied optimizations:")
    for opt in optimizations:
        print(f"  - {opt}")
    
    # Test with monitoring
    from lightrag_integration import ClinicalMetabolomicsRAG
    
    async with optimizer.memory_monitoring("Query processing"):
        rag = ClinicalMetabolomicsRAG()
        
        # Process multiple queries
        queries = [
            "What is metabolomics?",
            "How does mass spectrometry work?",
            "What are metabolic pathways?"
        ]
        
        for query in queries:
            response = await rag.aquery(query)
            print(f"  Processed query: {len(response)} chars")

if __name__ == "__main__":
    asyncio.run(test_memory_optimization())
```

### 3. API Rate Limiting Issues

#### Rate Limit Handler
```python
# rate_limit_handler.py
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any
import openai

class RateLimitHandler:
    def __init__(self):
        self.request_history = []
        self.backoff_multiplier = 1.0
        self.max_retries = 3
        
    async def handle_api_request(self, request_func, *args, **kwargs):
        """Handle API request with rate limiting and backoff."""
        
        for attempt in range(self.max_retries):
            try:
                # Check if we should wait
                wait_time = self.calculate_wait_time()
                if wait_time > 0:
                    print(f"⏳ Rate limit protection: waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                
                # Record request
                self.request_history.append(datetime.now())
                
                # Make request
                result = await request_func(*args, **kwargs)
                
                # Success - reset backoff
                self.backoff_multiplier = 1.0
                return result
                
            except openai.RateLimitError as e:
                print(f"💥 Rate limit hit (attempt {attempt + 1})")
                
                # Extract retry-after from headers if available
                retry_after = getattr(e, 'retry_after', None) or 60
                
                # Apply exponential backoff
                backoff_time = retry_after * self.backoff_multiplier
                self.backoff_multiplier *= 2
                
                print(f"   Backing off for {backoff_time:.1f}s")
                await asyncio.sleep(backoff_time)
                
            except Exception as e:
                print(f"❌ API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"API request failed after {self.max_retries} attempts")
    
    def calculate_wait_time(self) -> float:
        """Calculate wait time based on request history."""
        now = datetime.now()
        
        # Remove old requests (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.request_history = [req for req in self.request_history if req > cutoff]
        
        # OpenAI rate limits (approximate):
        # - GPT-4: 500 RPM (requests per minute)
        # - GPT-3.5: 3500 RPM
        # - Embeddings: 3000 RPM
        
        # Conservative limit: 100 requests per minute
        max_requests_per_minute = 100
        
        if len(self.request_history) >= max_requests_per_minute:
            # Calculate time to wait until oldest request is >1 minute old
            oldest_request = min(self.request_history)
            wait_until = oldest_request + timedelta(minutes=1)
            wait_seconds = (wait_until - now).total_seconds()
            return max(0, wait_seconds)
        
        # Smooth out requests - minimum 0.6s between requests (100 RPM)
        if self.request_history:
            last_request = max(self.request_history)
            time_since_last = (now - last_request).total_seconds()
            min_interval = 60.0 / max_requests_per_minute  # 0.6 seconds
            
            if time_since_last < min_interval:
                return min_interval - time_since_last
        
        return 0

# Integration with LightRAG
class RateLimitedClinicalRAG:
    def __init__(self):
        from lightrag_integration import ClinicalMetabolomicsRAG
        from lightrag_integration.config import LightRAGConfig
        
        self.config = LightRAGConfig.get_config()
        self.rag = ClinicalMetabolomicsRAG(config=self.config)
        self.rate_limiter = RateLimitHandler()
    
    async def aquery(self, query: str, **kwargs):
        """Rate-limited query method."""
        return await self.rate_limiter.handle_api_request(
            self.rag.aquery, query, **kwargs
        )

# Usage example
async def test_rate_limiting():
    rag = RateLimitedClinicalRAG()
    
    queries = [
        "What is metabolomics?",
        "How does NMR spectroscopy work?",
        "What are the applications of mass spectrometry?",
        "Explain metabolic pathway analysis.",
        "What is systems biology?"
    ]
    
    print("🚦 Testing rate-limited queries...")
    start_time = time.time()
    
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")
        try:
            response = await rag.aquery(query)
            print(f"✓ Response: {len(response)} characters")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total time: {total_time:.1f}s")
    print(f"   Average per query: {total_time/len(queries):.1f}s")

if __name__ == "__main__":
    asyncio.run(test_rate_limiting())
```

### 4. Concurrent User Handling

#### Load Testing Script
```python
# concurrent_load_test.py
import asyncio
import aiohttp
import time
import statistics
from dataclasses import dataclass
from typing import List
from concurrent.futures import ThreadPoolExecutor

@dataclass
class LoadTestResult:
    user_id: str
    query: str
    response_time: float
    success: bool
    error_message: str = None
    response_size: int = 0

class ConcurrentLoadTester:
    def __init__(self, max_concurrent_users: int = 10):
        self.max_concurrent_users = max_concurrent_users
        self.semaphore = asyncio.Semaphore(max_concurrent_users)
    
    async def simulate_user_query(self, user_id: str, query: str) -> LoadTestResult:
        """Simulate a single user query."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                from lightrag_integration import ClinicalMetabolomicsRAG
                
                rag = ClinicalMetabolomicsRAG()
                
                start_time = time.time()
                response = await rag.aquery(query)
                end_time = time.time()
                
                return LoadTestResult(
                    user_id=user_id,
                    query=query,
                    response_time=end_time - start_time,
                    success=True,
                    response_size=len(response) if response else 0
                )
                
            except Exception as e:
                return LoadTestResult(
                    user_id=user_id,
                    query=query,
                    response_time=0,
                    success=False,
                    error_message=str(e)
                )
    
    async def run_load_test(self, num_users: int, queries_per_user: int) -> List[LoadTestResult]:
        """Run concurrent load test."""
        print(f"🚀 Starting load test: {num_users} users, {queries_per_user} queries each")
        
        # Generate test queries
        base_queries = [
            "What is metabolomics?",
            "How does mass spectrometry work?",
            "What are metabolic pathways?",
            "Explain biomarker discovery.",
            "What is systems biology?"
        ]
        
        # Create tasks for all user queries
        tasks = []
        for user_i in range(num_users):
            user_id = f"user_{user_i:03d}"
            for query_i in range(queries_per_user):
                query = base_queries[query_i % len(base_queries)]
                task = self.simulate_user_query(user_id, query)
                tasks.append(task)
        
        # Run all queries concurrently
        print(f"⚡ Executing {len(tasks)} concurrent queries...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Filter out exceptions and process results
        valid_results = []
        exceptions = []
        
        for result in results:
            if isinstance(result, Exception):
                exceptions.append(result)
            else:
                valid_results.append(result)
        
        print(f"✅ Load test completed in {end_time - start_time:.1f}s")
        print(f"   Valid results: {len(valid_results)}")
        print(f"   Exceptions: {len(exceptions)}")
        
        return valid_results
    
    def analyze_results(self, results: List[LoadTestResult]) -> dict:
        """Analyze load test results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if successful:
            response_times = [r.response_time for r in successful]
            response_sizes = [r.response_size for r in successful]
            
            analysis = {
                "total_queries": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results),
                "response_time_stats": {
                    "min": min(response_times),
                    "max": max(response_times),
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "p95": sorted(response_times)[int(0.95 * len(response_times))]
                },
                "throughput": len(successful) / sum(response_times) if response_times else 0,
                "avg_response_size": statistics.mean(response_sizes) if response_sizes else 0,
                "error_messages": [r.error_message for r in failed if r.error_message]
            }
        else:
            analysis = {
                "total_queries": len(results),
                "successful": 0,
                "failed": len(failed),
                "success_rate": 0,
                "error_messages": [r.error_message for r in failed if r.error_message]
            }
        
        return analysis

async def main():
    """Run concurrent load test."""
    tester = ConcurrentLoadTester(max_concurrent_users=5)
    
    # Test scenarios
    scenarios = [
        (5, 2),   # 5 users, 2 queries each
        (10, 1),  # 10 users, 1 query each
        (3, 5),   # 3 users, 5 queries each
    ]
    
    for num_users, queries_per_user in scenarios:
        print(f"\n🧪 Scenario: {num_users} users, {queries_per_user} queries per user")
        print("=" * 60)
        
        results = await tester.run_load_test(num_users, queries_per_user)
        analysis = tester.analyze_results(results)
        
        print(f"\n📊 Results Analysis:")
        print(f"   Success rate: {analysis['success_rate']:.1%}")
        
        if analysis['successful'] > 0:
            stats = analysis['response_time_stats']
            print(f"   Response times:")
            print(f"     Min: {stats['min']:.2f}s")
            print(f"     Max: {stats['max']:.2f}s")
            print(f"     Mean: {stats['mean']:.2f}s")
            print(f"     Median: {stats['median']:.2f}s")
            print(f"     P95: {stats['p95']:.2f}s")
            print(f"   Throughput: {analysis['throughput']:.2f} queries/second")
            print(f"   Avg response size: {analysis['avg_response_size']:.0f} chars")
        
        if analysis['failed'] > 0:
            print(f"   Common errors:")
            error_counts = {}
            for error in analysis['error_messages']:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     {error}: {count} times")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Diagnostic Scripts and Tools

### 1. All-in-One Diagnostic Script

```python
#!/usr/bin/env python3
# comprehensive_diagnostic.py

"""
Comprehensive LightRAG Integration Diagnostic Tool
==================================================

This script runs a complete diagnostic of the LightRAG integration,
checking all components and generating a detailed report.

Usage: python comprehensive_diagnostic.py [--output-dir diagnostics]
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

class ComprehensiveDiagnostic:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "checks": {},
            "summary": {},
            "recommendations": []
        }
    
    async def run_all_diagnostics(self):
        """Run all diagnostic checks."""
        print("🔬 Comprehensive LightRAG Diagnostic")
        print("=" * 50)
        
        # List of all diagnostic checks
        checks = [
            ("System Environment", self.check_system_environment),
            ("Python Dependencies", self.check_python_dependencies),
            ("Configuration", self.check_configuration),
            ("Directory Structure", self.check_directory_structure),
            ("API Connectivity", self.check_api_connectivity),
            ("Feature Flags", self.check_feature_flags),
            ("RAG Initialization", self.check_rag_initialization),
            ("Query Processing", self.check_query_processing),
            ("Performance Metrics", self.check_performance_metrics),
            ("Log Analysis", self.check_log_analysis),
            ("Resource Usage", self.check_resource_usage)
        ]
        
        for check_name, check_func in checks:
            print(f"\n🔍 {check_name}")
            print("-" * 30)
            
            try:
                result = await check_func()
                self.results["checks"][check_name.lower().replace(" ", "_")] = result
                
                # Print summary
                status = result.get("status", "unknown")
                if status == "pass":
                    print(f"✅ {check_name}: PASS")
                elif status == "warning":
                    print(f"⚠️  {check_name}: WARNING")
                elif status == "fail":
                    print(f"❌ {check_name}: FAIL")
                else:
                    print(f"❓ {check_name}: {status.upper()}")
                
                # Print key details
                if "details" in result:
                    for key, value in result["details"].items():
                        if isinstance(value, bool):
                            icon = "✓" if value else "✗"
                            print(f"   {icon} {key}")
                        elif isinstance(value, (int, float)):
                            print(f"   {key}: {value}")
                        elif isinstance(value, str) and len(value) < 100:
                            print(f"   {key}: {value}")
                
            except Exception as e:
                print(f"❌ {check_name}: ERROR - {e}")
                self.results["checks"][check_name.lower().replace(" ", "_")] = {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        # Generate summary and recommendations
        self.generate_summary()
        self.generate_recommendations()
        
        # Save results
        await self.save_results()
        
        print(f"\n📊 Diagnostic Complete")
        print(f"Results saved to: {self.output_dir}")
    
    async def check_system_environment(self) -> Dict[str, Any]:
        """Check system environment."""
        import platform
        import psutil
        
        result = {
            "status": "pass",
            "details": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "disk_space_gb": round(shutil.disk_usage("/").free / (1024**3), 2)
            }
        }
        
        # Check for issues
        if psutil.virtual_memory().available < 1024**3:  # Less than 1GB
            result["status"] = "warning"
            result["warning"] = "Low available memory"
        
        if shutil.disk_usage("/").free < 5 * 1024**3:  # Less than 5GB
            result["status"] = "warning" if result["status"] == "pass" else "fail"
            result["warning"] = "Low disk space"
        
        return result
    
    async def check_python_dependencies(self) -> Dict[str, Any]:
        """Check Python package dependencies."""
        import pkg_resources
        
        required_packages = [
            "lightrag",
            "openai", 
            "numpy",
            "asyncio",
            "pathlib",
            "json",
            "logging"
        ]
        
        installed = {}
        missing = []
        
        for package in required_packages:
            try:
                if package in ["asyncio", "pathlib", "json", "logging"]:
                    # Built-in modules
                    __import__(package)
                    installed[package] = "built-in"
                else:
                    dist = pkg_resources.get_distribution(package)
                    installed[package] = dist.version
            except (pkg_resources.DistributionNotFound, ImportError):
                missing.append(package)
        
        result = {
            "status": "pass" if not missing else "fail",
            "details": {
                "installed_packages": installed,
                "missing_packages": missing
            }
        }
        
        if missing:
            result["error"] = f"Missing packages: {', '.join(missing)}"
        
        return result
    
    async def check_configuration(self) -> Dict[str, Any]:
        """Check LightRAG configuration."""
        try:
            from lightrag_integration.config import LightRAGConfig
            
            # Try to load configuration
            config = LightRAGConfig.get_config(validate_config=False)
            
            # Check key configuration values
            result = {
                "status": "pass",
                "details": {
                    "api_key_present": bool(config.api_key),
                    "working_dir": str(config.working_dir),
                    "integration_enabled": config.lightrag_integration_enabled,
                    "rollout_percentage": config.lightrag_rollout_percentage,
                    "model": config.model,
                    "embedding_model": config.embedding_model,
                    "max_async": config.max_async,
                    "max_tokens": config.max_tokens
                }
            }
            
            # Validate configuration
            try:
                config.validate()
                result["details"]["configuration_valid"] = True
            except Exception as e:
                result["status"] = "warning"
                result["details"]["configuration_valid"] = False
                result["warning"] = f"Configuration validation failed: {e}"
            
            # Check environment variables
            env_vars = {k: v for k, v in os.environ.items() 
                       if k.startswith(('LIGHTRAG_', 'OPENAI_'))}
            result["details"]["environment_variables_count"] = len(env_vars)
            
        except Exception as e:
            result = {
                "status": "fail",
                "error": str(e),
                "details": {}
            }
        
        return result
    
    async def check_directory_structure(self) -> Dict[str, Any]:
        """Check directory structure."""
        try:
            from lightrag_integration.config import LightRAGConfig
            config = LightRAGConfig.get_config()
            
            directories_to_check = [
                ("working_dir", config.working_dir),
                ("graph_storage_dir", config.graph_storage_dir),
                ("log_dir", config.log_dir)
            ]
            
            result = {
                "status": "pass",
                "details": {}
            }
            
            for dir_name, dir_path in directories_to_check:
                exists = dir_path.exists()
                is_writable = False
                
                if exists:
                    try:
                        # Test write permissions
                        test_file = dir_path / ".write_test"
                        test_file.touch()
                        test_file.unlink()
                        is_writable = True
                    except Exception:
                        pass
                
                result["details"][f"{dir_name}_exists"] = exists
                result["details"][f"{dir_name}_writable"] = is_writable
                result["details"][f"{dir_name}_path"] = str(dir_path)
                
                if not exists or not is_writable:
                    result["status"] = "warning"
            
        except Exception as e:
            result = {
                "status": "fail",
                "error": str(e),
                "details": {}
            }
        
        return result
    
    async def check_api_connectivity(self) -> Dict[str, Any]:
        """Check API connectivity."""
        try:
            import openai
            from lightrag_integration.config import LightRAGConfig
            
            config = LightRAGConfig.get_config()
            
            if not config.api_key:
                return {
                    "status": "fail",
                    "error": "No API key configured",
                    "details": {}
                }
            
            # Test OpenAI API connectivity
            client = openai.OpenAI(api_key=config.api_key)
            
            try:
                # Simple API call
                models = client.models.list()
                api_accessible = True
                model_count = len(list(models.data))
            except Exception as api_error:
                api_accessible = False
                model_count = 0
                api_error_msg = str(api_error)
            
            result = {
                "status": "pass" if api_accessible else "fail",
                "details": {
                    "api_accessible": api_accessible,
                    "available_models": model_count,
                    "configured_model": config.model,
                    "configured_embedding_model": config.embedding_model
                }
            }
            
            if not api_accessible:
                result["error"] = f"API not accessible: {api_error_msg}"
            
        except Exception as e:
            result = {
                "status": "fail",
                "error": str(e),
                "details": {}
            }
        
        return result
    
    async def check_feature_flags(self) -> Dict[str, Any]:
        """Check feature flag system."""
        try:
            from lightrag_integration.feature_flag_manager import FeatureFlagManager
            from lightrag_integration.config import LightRAGConfig
            
            config = LightRAGConfig.get_config()
            flag_manager = FeatureFlagManager(config)
            
            # Test routing decision
            test_context = {
                "user_id": "diagnostic_test",
                "session_id": "diagnostic_session"
            }
            
            routing_result = flag_manager.should_use_lightrag(test_context)
            
            result = {
                "status": "pass",
                "details": {
                    "integration_enabled": config.lightrag_integration_enabled,
                    "rollout_percentage": config.lightrag_rollout_percentage,
                    "routing_decision": routing_result.decision.value,
                    "routing_reason": routing_result.reason.value,
                    "circuit_breaker_enabled": config.lightrag_enable_circuit_breaker,
                    "fallback_enabled": config.lightrag_fallback_to_perplexity
                }
            }
            
            # Check circuit breaker status
            try:
                breaker_status = flag_manager.get_circuit_breaker_status()
                result["details"]["circuit_breaker_status"] = breaker_status
            except AttributeError:
                # Method might not exist in all versions
                pass
            
        except Exception as e:
            result = {
                "status": "fail",
                "error": str(e),
                "details": {}
            }
        
        return result
    
    async def check_rag_initialization(self) -> Dict[str, Any]:
        """Check RAG system initialization."""
        try:
            from lightrag_integration import ClinicalMetabolomicsRAG
            from lightrag_integration.config import LightRAGConfig
            
            config = LightRAGConfig.get_config()
            
            # Test RAG initialization
            rag = ClinicalMetabolomicsRAG(config=config)
            
            result = {
                "status": "pass",
                "details": {
                    "rag_initialized": True,
                    "config_loaded": True
                }
            }
            
        except Exception as e:
            result = {
                "status": "fail",
                "error": str(e),
                "details": {
                    "rag_initialized": False
                }
            }
        
        return result
    
    async def check_query_processing(self) -> Dict[str, Any]:
        """Check query processing capabilities."""
        try:
            from lightrag_integration import ClinicalMetabolomicsRAG
            
            rag = ClinicalMetabolomicsRAG()
            
            # Test simple query
            test_query = "What is metabolomics?"
            
            import time
            start_time = time.time()
            
            try:
                response = await rag.aquery(test_query, mode="naive")
                processing_time = time.time() - start_time
                query_success = True
                response_length = len(response) if response else 0
            except Exception as query_error:
                processing_time = time.time() - start_time
                query_success = False
                response_length = 0
                query_error_msg = str(query_error)
            
            result = {
                "status": "pass" if query_success else "fail",
                "details": {
                    "query_processing_works": query_success,
                    "processing_time": round(processing_time, 2),
                    "response_length": response_length,
                    "test_query": test_query
                }
            }
            
            if not query_success:
                result["error"] = f"Query processing failed: {query_error_msg}"
            
        except Exception as e:
            result = {
                "status": "fail",
                "error": str(e),
                "details": {}
            }
        
        return result
    
    async def check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance metrics and monitoring."""
        result = {
            "status": "pass",
            "details": {}
        }
        
        try:
            # Check if performance monitoring is enabled
            from lightrag_integration.config import LightRAGConfig
            config = LightRAGConfig.get_config()
            
            result["details"]["cost_tracking_enabled"] = config.enable_cost_tracking
            result["details"]["performance_comparison_enabled"] = config.lightrag_enable_performance_comparison
            result["details"]["quality_metrics_enabled"] = config.lightrag_enable_quality_metrics
            
            # Check budget manager
            try:
                from lightrag_integration.budget_manager import BudgetManager
                budget_manager = BudgetManager()
                budget_status = budget_manager.get_budget_status()
                
                result["details"]["budget_tracking_works"] = True
                result["details"]["daily_cost"] = budget_status.get("daily_cost", 0)
                result["details"]["monthly_cost"] = budget_status.get("monthly_cost", 0)
            except Exception:
                result["details"]["budget_tracking_works"] = False
            
        except Exception as e:
            result["status"] = "warning"
            result["warning"] = str(e)
        
        return result
    
    async def check_log_analysis(self) -> Dict[str, Any]:
        """Analyze logs for issues."""
        result = {
            "status": "pass",
            "details": {}
        }
        
        log_files = [
            "logs/lightrag_integration.log",
            "logs/structured_logs.jsonl"
        ]
        
        for log_file in log_files:
            if Path(log_file).exists():
                try:
                    with open(log_file) as f:
                        lines = f.readlines()
                    
                    total_lines = len(lines)
                    error_lines = len([line for line in lines if "ERROR" in line])
                    warning_lines = len([line for line in lines if "WARNING" in line])
                    
                    result["details"][f"{Path(log_file).name}_total_lines"] = total_lines
                    result["details"][f"{Path(log_file).name}_errors"] = error_lines
                    result["details"][f"{Path(log_file).name}_warnings"] = warning_lines
                    
                    if error_lines > total_lines * 0.1:  # More than 10% errors
                        result["status"] = "warning"
                    
                except Exception:
                    result["details"][f"{Path(log_file).name}_readable"] = False
            else:
                result["details"][f"{Path(log_file).name}_exists"] = False
        
        return result
    
    async def check_resource_usage(self) -> Dict[str, Any]:
        """Check current resource usage."""
        try:
            import psutil
            
            process = psutil.Process()
            
            result = {
                "status": "pass",
                "details": {
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": round(process.memory_info().rss / (1024**2), 2),
                    "open_files": len(process.open_files()),
                    "num_threads": process.num_threads()
                }
            }
            
            # Check for resource issues
            if result["details"]["memory_mb"] > 2048:  # More than 2GB
                result["status"] = "warning"
                result["warning"] = "High memory usage"
            
            if result["details"]["open_files"] > 1000:
                result["status"] = "warning"
                result["warning"] = "Many open files"
            
        except Exception as e:
            result = {
                "status": "fail",
                "error": str(e),
                "details": {}
            }
        
        return result
    
    def generate_summary(self):
        """Generate diagnostic summary."""
        total_checks = len(self.results["checks"])
        passed = sum(1 for check in self.results["checks"].values() 
                    if check.get("status") == "pass")
        warnings = sum(1 for check in self.results["checks"].values() 
                      if check.get("status") == "warning")
        failed = sum(1 for check in self.results["checks"].values() 
                    if check.get("status") == "fail")
        errors = sum(1 for check in self.results["checks"].values() 
                    if check.get("status") == "error")
        
        self.results["summary"] = {
            "total_checks": total_checks,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "errors": errors,
            "overall_health": "healthy" if failed == 0 and errors == 0 else (
                "degraded" if warnings > 0 else "critical"
            )
        }
    
    def generate_recommendations(self):
        """Generate recommendations based on diagnostic results."""
        recommendations = []
        
        # Check for common issues and generate recommendations
        for check_name, check_result in self.results["checks"].items():
            if check_result.get("status") in ["fail", "error"]:
                error = check_result.get("error", "Unknown error")
                
                if "API key" in error:
                    recommendations.append({
                        "priority": "high",
                        "category": "configuration",
                        "issue": "Missing or invalid API key",
                        "recommendation": "Set OPENAI_API_KEY environment variable with valid OpenAI API key"
                    })
                
                elif "permission" in error.lower():
                    recommendations.append({
                        "priority": "medium",
                        "category": "system",
                        "issue": "Permission denied accessing files/directories",
                        "recommendation": "Check file and directory permissions, ensure write access to working directory"
                    })
                
                elif "memory" in error.lower():
                    recommendations.append({
                        "priority": "medium",
                        "category": "performance",
                        "issue": "Memory-related error",
                        "recommendation": "Reduce LIGHTRAG_MAX_ASYNC and LIGHTRAG_BATCH_SIZE, increase available system memory"
                    })
            
            elif check_result.get("status") == "warning":
                warning = check_result.get("warning", "")
                
                if "memory" in warning.lower():
                    recommendations.append({
                        "priority": "low",
                        "category": "performance",
                        "issue": "High memory usage detected",
                        "recommendation": "Monitor memory usage and consider optimizing batch sizes"
                    })
        
        # General recommendations based on configuration
        config_check = self.results["checks"].get("configuration", {})
        if config_check.get("details", {}).get("integration_enabled") is False:
            recommendations.append({
                "priority": "info",
                "category": "configuration",
                "issue": "LightRAG integration is disabled",
                "recommendation": "Set LIGHTRAG_INTEGRATION_ENABLED=true to enable integration"
            })
        
        self.results["recommendations"] = recommendations
    
    async def save_results(self):
        """Save diagnostic results to files."""
        # Save main results
        results_file = self.output_dir / f"diagnostic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / "diagnostic_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("LightRAG Integration Diagnostic Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Overall Health: {self.results['summary']['overall_health'].upper()}\n\n")
            
            f.write("Check Results:\n")
            f.write("-" * 20 + "\n")
            for check_name, check_result in self.results["checks"].items():
                status = check_result.get("status", "unknown").upper()
                f.write(f"{check_name.replace('_', ' ').title()}: {status}\n")
            
            f.write(f"\nSummary: {self.results['summary']['passed']} passed, ")
            f.write(f"{self.results['summary']['warnings']} warnings, ")
            f.write(f"{self.results['summary']['failed']} failed, ")
            f.write(f"{self.results['summary']['errors']} errors\n\n")
            
            if self.results["recommendations"]:
                f.write("Recommendations:\n")
                f.write("-" * 20 + "\n")
                for rec in self.results["recommendations"]:
                    f.write(f"[{rec['priority'].upper()}] {rec['issue']}\n")
                    f.write(f"  Solution: {rec['recommendation']}\n\n")
        
        print(f"📄 Results saved:")
        print(f"   Detailed: {results_file}")
        print(f"   Summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="LightRAG Integration Comprehensive Diagnostic")
    parser.add_argument("--output-dir", default="diagnostics", 
                       help="Output directory for diagnostic results")
    
    args = parser.parse_args()
    
    diagnostic = ComprehensiveDiagnostic(args.output_dir)
    
    try:
        asyncio.run(diagnostic.run_all_diagnostics())
        
        # Print final summary
        summary = diagnostic.results["summary"]
        health = summary["overall_health"]
        
        print(f"\n🎯 Final Assessment: {health.upper()}")
        
        if health == "healthy":
            print("✅ All systems operational")
            sys.exit(0)
        elif health == "degraded":
            print("⚠️  Some issues detected but system is functional")
            sys.exit(1)
        else:  # critical
            print("❌ Critical issues require immediate attention")
            sys.exit(2)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Diagnostic interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Diagnostic failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Save this as `comprehensive_diagnostic.py` and run it to get a complete health check of your LightRAG integration.

### 2. Quick Health Check Script

```bash
#!/bin/bash
# quick_health_check.sh

echo "⚡ Quick LightRAG Health Check"
echo "=============================="

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1: OK"
    else
        echo "❌ $1: FAILED"
    fi
}

# 1. Check Python
python3 --version > /dev/null 2>&1
check_status "Python 3"

# 2. Check imports
python3 -c "import lightrag" > /dev/null 2>&1
check_status "LightRAG import"

python3 -c "from lightrag_integration import ClinicalMetabolomicsRAG" > /dev/null 2>&1
check_status "Integration import"

# 3. Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ API Key: NOT SET"
else
    echo "✅ API Key: SET (${#OPENAI_API_KEY} chars)"
fi

# 4. Check directories
if [ -d "$LIGHTRAG_WORKING_DIR" ]; then
    echo "✅ Working Dir: EXISTS"
else
    echo "⚠️  Working Dir: MISSING"
fi

if [ -d "logs" ]; then
    echo "✅ Logs Dir: EXISTS"
else
    echo "⚠️  Logs Dir: MISSING"
fi

# 5. Check configuration
python3 -c "
from lightrag_integration.config import LightRAGConfig
try:
    config = LightRAGConfig.get_config(validate_config=True)
    print('✅ Configuration: VALID')
except Exception as e:
    print('❌ Configuration: INVALID -', str(e))
" 2>/dev/null

# 6. Quick memory check
python3 -c "
import psutil
mem = psutil.virtual_memory()
if mem.available > 1024**3:
    print('✅ Memory: SUFFICIENT ({:.1f} GB available)'.format(mem.available / 1024**3))
else:
    print('⚠️  Memory: LOW ({:.1f} GB available)'.format(mem.available / 1024**3))
"

echo ""
echo "🏃 Run 'python comprehensive_diagnostic.py' for detailed analysis"
```

This comprehensive troubleshooting guide provides:

1. **Quick Reference** - Essential commands and checks for immediate issues
2. **Common Integration Issues** - Step-by-step resolution for typical problems
3. **Runtime Issues** - Solutions for problems that occur during operation
4. **System Monitoring** - Tools and scripts for ongoing health monitoring
5. **Debugging Procedures** - Systematic approach to diagnosing complex issues
6. **Emergency Procedures** - Quick disable, rollback, and recovery procedures
7. **Performance Troubleshooting** - Tools for diagnosing and fixing performance issues
8. **Diagnostic Scripts** - Automated tools for comprehensive system analysis

The guide is designed to be practical and actionable, with specific error messages, diagnostic commands, and resolution steps that developers and operators can use to quickly resolve problems with the LightRAG integration.