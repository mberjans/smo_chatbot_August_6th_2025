# LightRAG Developer Integration Guide
## Practical Implementation Steps for Clinical Metabolomics Oracle

**Version:** 1.0  
**Last Updated:** 2025-08-08  
**Author:** Claude Code (Anthropic)  
**Target Audience:** Developers implementing LightRAG integration

---

## Table of Contents

1. [Prerequisites and Setup](#1-prerequisites-and-setup)
2. [Step-by-Step Integration Process](#2-step-by-step-integration-process)
3. [Testing and Validation Steps](#3-testing-and-validation-steps)
4. [Development Best Practices](#4-development-best-practices)
5. [Common Implementation Patterns](#5-common-implementation-patterns)
6. [Quick Reference Commands](#6-quick-reference-commands)
7. [Troubleshooting Common Issues](#7-troubleshooting-common-issues)

---

## 1. Prerequisites and Setup

### 1.1 System Requirements Check

Before starting, verify your development environment meets these requirements:

```bash
# Check Python version (3.8+ required)
python3 --version

# Check if required system packages are available
pip3 --version
git --version

# Verify disk space (minimum 2GB for LightRAG data)
df -h .

# Check memory (minimum 4GB recommended)
free -h
```

**Required System Specifications:**
- Python 3.8 or higher
- At least 4GB RAM
- 2GB free disk space
- Git for version control
- Active internet connection for API access

### 1.2 Dependencies Installation

Install all required dependencies in order:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Upgrade pip to latest version
pip install --upgrade pip

# 3. Install base requirements
pip install -r requirements.txt

# 4. Install LightRAG requirements
pip install -r requirements_lightrag.txt

# 5. Install additional development dependencies
pip install pytest pytest-cov pytest-asyncio coverage

# 6. Verify LightRAG installation
python -c "from lightrag import LightRAG, QueryParam; print('LightRAG installed successfully')"

# 7. Verify OpenAI client
python -c "import openai; print('OpenAI client installed successfully')"
```

**Dependency Versions (Critical):**
```bash
# Verify these exact versions are installed
pip list | grep -E "(lightrag|openai|chainlit|requests)"

# Expected output:
# lightrag-hku>=1.4.6
# openai>=1.0.0
# chainlit==1.0.401
# requests>=2.28.0
```

### 1.3 Environment Preparation

#### Step 1: Create Environment File

Create `.env` file in project root:

```bash
# Copy example environment file
cp .env.example .env  # If exists, otherwise create manually

# Open .env file and configure the following:
nano .env
```

#### Step 2: Configure Required Environment Variables

Add these REQUIRED variables to your `.env` file:

```bash
# === REQUIRED API KEYS ===
OPENAI_API_KEY=sk-your-openai-api-key-here
PERPLEXITY_API=your-perplexity-api-key-here

# === LIGHTRAG CORE CONFIGURATION ===
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
LIGHTRAG_WORKING_DIR=./lightrag_data
LIGHTRAG_MAX_ASYNC=16
LIGHTRAG_MAX_TOKENS=32768

# === INTEGRATION FEATURE FLAGS ===
LIGHTRAG_INTEGRATION_ENABLED=false  # Start disabled for safety
LIGHTRAG_ROLLOUT_PERCENTAGE=0.0     # Start at 0%
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true

# === LOGGING CONFIGURATION ===
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_DIR=./logs

# === COST TRACKING ===
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_DAILY_BUDGET_LIMIT=10.0
LIGHTRAG_COST_ALERT_THRESHOLD=80.0
```

#### Step 3: Validate API Keys

Test your API keys before proceeding:

```bash
# Test OpenAI API key
python3 -c "
import openai
import os
from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
try:
    response = client.embeddings.create(
        input='test',
        model='text-embedding-3-small'
    )
    print('✅ OpenAI API key is valid')
except Exception as e:
    print(f'❌ OpenAI API key failed: {e}')
"

# Test Perplexity API key
python3 -c "
import requests
import os
from dotenv import load_dotenv
load_dotenv()

headers = {
    'Authorization': f\"Bearer {os.getenv('PERPLEXITY_API')}\",
    'Content-Type': 'application/json'
}
payload = {
    'model': 'sonar',
    'messages': [{'role': 'user', 'content': 'test'}],
    'temperature': 0.1
}
try:
    response = requests.post(
        'https://api.perplexity.ai/chat/completions',
        json=payload,
        headers=headers,
        timeout=10
    )
    if response.status_code == 200:
        print('✅ Perplexity API key is valid')
    else:
        print(f'❌ Perplexity API key failed: {response.status_code}')
except Exception as e:
    print(f'❌ Perplexity API test failed: {e}')
"
```

#### Step 4: Create Required Directories

```bash
# Create all necessary directories
mkdir -p lightrag_data
mkdir -p logs
mkdir -p papers
mkdir -p quality_reports

# Set appropriate permissions
chmod 755 lightrag_data logs papers quality_reports

# Verify directories were created
ls -la | grep -E "(lightrag_data|logs|papers|quality_reports)"
```

---

## 2. Step-by-Step Integration Process

### 2.1 Phase 1: Initialize LightRAG Knowledge Base

#### Step 1: Prepare Biomedical Documents

```bash
# 1. Place your PDF documents in the papers directory
cp /path/to/your/clinical_papers/*.pdf papers/

# 2. Verify PDF files are readable
python3 -c "
import os
from pathlib import Path
pdf_files = list(Path('papers').glob('*.pdf'))
print(f'Found {len(pdf_files)} PDF files:')
for pdf in pdf_files[:5]:  # Show first 5
    size = os.path.getsize(pdf) / 1024 / 1024  # MB
    print(f'  - {pdf.name} ({size:.1f} MB)')
"
```

#### Step 2: Initialize Knowledge Base

Create initialization script `initialize_kb.py`:

```python
#!/usr/bin/env python3
"""
Knowledge Base Initialization Script
Run this once to set up the LightRAG knowledge base with your documents.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add lightrag_integration to Python path
sys.path.append('lightrag_integration')

from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from lightrag_integration.config import LightRAGConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def initialize_knowledge_base():
    """Initialize the LightRAG knowledge base with PDF documents."""
    try:
        # Create configuration
        config = LightRAGConfig.from_environment()
        logger.info(f"Using working directory: {config.working_dir}")
        
        # Initialize RAG system
        logger.info("Initializing LightRAG system...")
        rag = ClinicalMetabolomicsRAG(config)
        
        # Process PDF directory if it exists
        pdf_directory = Path("papers")
        if pdf_directory.exists() and list(pdf_directory.glob("*.pdf")):
            logger.info(f"Processing PDFs from {pdf_directory}")
            await rag.process_pdf_directory(str(pdf_directory))
            logger.info("✅ Knowledge base initialized successfully!")
        else:
            logger.warning("No PDF files found in papers/ directory")
            logger.info("Knowledge base initialized without documents")
        
        # Test a simple query
        logger.info("Testing knowledge base with sample query...")
        test_query = "What is clinical metabolomics?"
        response = await rag.query_async(
            query_text=test_query,
            mode="hybrid"
        )
        
        logger.info(f"Test query successful. Response length: {len(response.content)} characters")
        return True
        
    except Exception as e:
        logger.error(f"Knowledge base initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(initialize_knowledge_base())
    if success:
        print("\n🎉 Knowledge base initialization completed!")
        print("You can now proceed to integrate LightRAG with main.py")
    else:
        print("\n❌ Knowledge base initialization failed!")
        print("Check the logs above for error details")
        sys.exit(1)
```

Run the initialization:

```bash
# Make the script executable
chmod +x initialize_kb.py

# Run initialization (this may take 10-30 minutes depending on document count)
python3 initialize_kb.py

# Monitor progress in another terminal
tail -f logs/lightrag_integration.log
```

### 2.2 Phase 2: Integrate with main.py

#### Step 1: Backup Existing main.py

```bash
# Always backup before making changes
cp src/main.py src/main.py.backup.$(date +%Y%m%d_%H%M%S)
ls -la src/main.py*
```

#### Step 2: Add Required Imports

Edit `src/main.py` and add these imports at the top (after existing imports):

```python
# Add after line 24 (after existing imports)

# === LightRAG Integration Imports ===
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LightRAG integration components
sys.path.append('lightrag_integration')
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.integration_wrapper import (
    IntegratedQueryService, QueryRequest, ServiceResponse
)
from lightrag_integration.feature_flag_manager import FeatureFlagManager

# Global variables for LightRAG integration
integrated_service = None
lightrag_config = None
```

#### Step 3: Initialize Integration Service

Add initialization function after the existing `on_chat_start()` function:

```python
async def initialize_lightrag_integration():
    """Initialize LightRAG integration service with error handling."""
    global integrated_service, lightrag_config
    
    try:
        # Load configuration from environment
        lightrag_config = LightRAGConfig.from_environment()
        
        # Check if integration is enabled
        if not lightrag_config.lightrag_integration_enabled:
            logging.info("LightRAG integration is disabled via configuration")
            return False
        
        # Initialize integrated service
        integrated_service = IntegratedQueryService(
            config=lightrag_config,
            perplexity_api_key=PERPLEXITY_API,
            logger=logging.getLogger(__name__)
        )
        
        # Test the service
        await integrated_service.health_check()
        logging.info("✅ LightRAG integration service initialized successfully")
        return True
        
    except Exception as e:
        logging.warning(f"LightRAG integration initialization failed: {e}")
        logging.info("Will fall back to Perplexity-only mode")
        integrated_service = None
        return False
```

#### Step 4: Modify on_chat_start Function

Update the existing `on_chat_start()` function to include LightRAG initialization:

```python
@cl.on_chat_start
async def on_chat_start(accepted: bool = False):
    # === Existing code remains the same until line 107 ===
    
    # Add after line 107 (after existing initialization)
    
    # Initialize LightRAG integration
    lightrag_success = await initialize_lightrag_integration()
    cl.user_session.set("lightrag_enabled", lightrag_success)
    
    if lightrag_success:
        cl.user_session.set("integrated_service", integrated_service)
        logging.info("Session configured with LightRAG integration")
    else:
        cl.user_session.set("integrated_service", None)
        logging.info("Session configured with Perplexity-only mode")
    
    # === Rest of existing code remains the same ===
```

#### Step 5: Replace Query Processing Logic

Find the Perplexity API call section (around lines 177-218) and replace with:

```python
# Replace the existing Perplexity API call section with this integrated approach

@cl.on_message
async def on_message(message: cl.Message):
    # === Keep all existing preprocessing code ===
    # (language detection, translation, etc.)
    
    # Get integrated service from session
    integrated_service = cl.user_session.get("integrated_service")
    lightrag_enabled = cl.user_session.get("lightrag_enabled", False)
    
    # Initialize response variables
    content = ""
    citations = []
    service_used = "unknown"
    
    if lightrag_enabled and integrated_service:
        # Use integrated routing service
        try:
            # Create query request
            query_request = QueryRequest(
                query_text=content,  # The processed query text
                user_id=cl.user_session.get("user").identifier if cl.user_session.get("user") else "anonymous",
                session_id=cl.user_session.get("id"),
                query_type="clinical_metabolomics",
                timeout_seconds=30.0,
                metadata={
                    "original_language": detected_language,
                    "chainlit_session": True
                }
            )
            
            # Execute query with integrated service
            service_response = await integrated_service.query_async(query_request)
            
            if service_response.is_success:
                content = service_response.content
                citations = service_response.citations or []
                service_used = service_response.response_type.value
                
                # Log routing information for monitoring
                routing_info = service_response.metadata.get('routing_decision', 'unknown')
                logging.info(f"Query successfully routed to: {routing_info}")
                
                # Add service information to response metadata
                if service_response.metadata.get('performance_metrics'):
                    perf_metrics = service_response.metadata['performance_metrics']
                    logging.info(f"Response time: {perf_metrics.get('response_time', 'unknown')}s")
            else:
                # Integrated service failed, log error and fall back
                logging.error(f"Integrated service failed: {service_response.error_details}")
                raise Exception(f"Service error: {service_response.error_details}")
                
        except Exception as e:
            logging.error(f"Integrated service error, falling back to Perplexity: {e}")
            # Fall through to Perplexity fallback below
            lightrag_enabled = False
    
    # Perplexity fallback (original logic)
    if not lightrag_enabled or not content:
        service_used = "perplexity_fallback"
        logging.info("Using Perplexity API (fallback mode)")
        
        # === Keep original Perplexity API code ===
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
            citations = response_data.get('citations', [])
        else:
            logging.error(f"Perplexity API error: {response.status_code}, {response.text}")
            content = "I apologize, but I'm experiencing technical difficulties. Please try again."
            citations = []
    
    # === Keep all existing post-processing code ===
    # (citation processing, confidence scores, translation, etc.)
    
    # Add service information to the response for debugging
    if service_used and logging.getLogger().isEnabledFor(logging.INFO):
        logging.info(f"Response generated using: {service_used}")
```

#### Step 6: Add Development Monitoring

Add this helper function for development monitoring:

```python
@cl.action_callback("check_integration_status")
async def check_integration_status(action):
    """Development helper to check integration status."""
    lightrag_enabled = cl.user_session.get("lightrag_enabled", False)
    integrated_service = cl.user_session.get("integrated_service")
    
    status_info = {
        "lightrag_enabled": lightrag_enabled,
        "service_available": integrated_service is not None,
        "config_loaded": lightrag_config is not None
    }
    
    if integrated_service:
        try:
            health_status = await integrated_service.health_check()
            status_info["health_check"] = "passed"
        except Exception as e:
            status_info["health_check"] = f"failed: {e}"
    
    await cl.Message(
        content=f"Integration Status:\n```json\n{json.dumps(status_info, indent=2)}\n```"
    ).send()
```

---

## 3. Testing and Validation Steps

### 3.1 Pre-Integration Testing

Before enabling the integration, test each component:

#### Test 1: Configuration Validation

```bash
# Create test script: test_config.py
cat > test_config.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.append('lightrag_integration')

from lightrag_integration.config import LightRAGConfig

try:
    config = LightRAGConfig.from_environment()
    print("✅ Configuration loaded successfully")
    print(f"  - Working directory: {config.working_dir}")
    print(f"  - Model: {config.model}")
    print(f"  - Embedding model: {config.embedding_model}")
    print(f"  - Integration enabled: {config.lightrag_integration_enabled}")
    print(f"  - Rollout percentage: {config.lightrag_rollout_percentage}%")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)
EOF

python3 test_config.py
```

#### Test 2: LightRAG Service Health Check

```bash
# Create test script: test_lightrag.py
cat > test_lightrag.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import sys
sys.path.append('lightrag_integration')

from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from lightrag_integration.config import LightRAGConfig

async def test_lightrag():
    try:
        config = LightRAGConfig.from_environment()
        rag = ClinicalMetabolomicsRAG(config)
        
        # Test simple query
        response = await rag.query_async(
            query_text="What is metabolomics?",
            mode="hybrid"
        )
        
        print("✅ LightRAG service test passed")
        print(f"  - Response length: {len(response.content)} characters")
        print(f"  - Response preview: {response.content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ LightRAG service test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_lightrag())
    sys.exit(0 if success else 1)
EOF

python3 test_lightrag.py
```

#### Test 3: Integration Wrapper Test

```bash
# Create test script: test_integration.py
cat > test_integration.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append('lightrag_integration')

from lightrag_integration.config import LightRAGConfig
from lightrag_integration.integration_wrapper import IntegratedQueryService, QueryRequest

async def test_integration():
    try:
        config = LightRAGConfig.from_environment()
        service = IntegratedQueryService(
            config=config,
            perplexity_api_key=os.getenv('PERPLEXITY_API'),
            logger=None
        )
        
        # Test query
        request = QueryRequest(
            query_text="What are biomarkers in metabolomics?",
            user_id="test_user",
            session_id="test_session",
            query_type="clinical_metabolomics"
        )
        
        response = await service.query_async(request)
        
        print("✅ Integration wrapper test passed")
        print(f"  - Success: {response.is_success}")
        print(f"  - Service type: {response.response_type.value}")
        print(f"  - Response length: {len(response.content)} characters")
        return True
    except Exception as e:
        print(f"❌ Integration wrapper test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)
EOF

python3 test_integration.py
```

### 3.2 Integration Testing

#### Test 4: Chainlit Integration Test

```bash
# Start Chainlit with integration disabled for baseline
LIGHTRAG_INTEGRATION_ENABLED=false chainlit run src/main.py -w

# In another terminal, test Perplexity-only mode
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is clinical metabolomics?"}'
```

#### Test 5: Gradual Rollout Test

```bash
# Test with 1% rollout
LIGHTRAG_INTEGRATION_ENABLED=true LIGHTRAG_ROLLOUT_PERCENTAGE=1.0 chainlit run src/main.py -w

# Monitor logs in separate terminal
tail -f logs/lightrag_integration.log | grep -E "(routing|fallback|error)"
```

### 3.3 Validation Queries

Use these test queries to validate functionality:

```python
# Create validation script: validate_integration.py
cat > validate_integration.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import sys
import time
import json

sys.path.append('lightrag_integration')

from lightrag_integration.config import LightRAGConfig
from lightrag_integration.integration_wrapper import IntegratedQueryService, QueryRequest

VALIDATION_QUERIES = [
    # Basic functionality
    "What is clinical metabolomics?",
    "Explain metabolic biomarkers",
    
    # Domain-specific queries
    "How are mass spectrometry used in metabolomics?",
    "What are the applications of NMR in metabolomics?",
    
    # Complex queries
    "Compare LC-MS and GC-MS techniques in metabolomics research",
    "Describe the metabolomics workflow from sample to data analysis",
]

async def validate_queries():
    """Run validation queries and collect metrics."""
    config = LightRAGConfig.from_environment()
    service = IntegratedQueryService(
        config=config,
        perplexity_api_key=os.getenv('PERPLEXITY_API'),
        logger=None
    )
    
    results = []
    
    for i, query in enumerate(VALIDATION_QUERIES, 1):
        print(f"Testing query {i}/{len(VALIDATION_QUERIES)}: {query[:50]}...")
        
        start_time = time.time()
        request = QueryRequest(
            query_text=query,
            user_id="validation_test",
            session_id="validation_session",
            query_type="clinical_metabolomics"
        )
        
        try:
            response = await service.query_async(request)
            response_time = time.time() - start_time
            
            result = {
                "query": query,
                "success": response.is_success,
                "service": response.response_type.value,
                "response_time": response_time,
                "response_length": len(response.content) if response.content else 0,
                "error": response.error_details
            }
            results.append(result)
            
            print(f"  ✅ Success: {response.is_success}, Service: {response.response_type.value}, Time: {response_time:.2f}s")
            
        except Exception as e:
            result = {
                "query": query,
                "success": False,
                "service": "error",
                "response_time": time.time() - start_time,
                "response_length": 0,
                "error": str(e)
            }
            results.append(result)
            print(f"  ❌ Failed: {e}")
        
        # Brief pause between queries
        await asyncio.sleep(1)
    
    # Print summary
    success_count = sum(1 for r in results if r["success"])
    print(f"\nValidation Summary:")
    print(f"  Total queries: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(results) - success_count}")
    print(f"  Success rate: {success_count/len(results)*100:.1f}%")
    
    # Save detailed results
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to validation_results.json")
    
    return success_count == len(results)

if __name__ == "__main__":
    success = asyncio.run(validate_queries())
    sys.exit(0 if success else 1)
EOF

python3 validate_integration.py
```

### 3.4 Performance Testing

```bash
# Create performance test script
cat > test_performance.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import time
import statistics
import sys

sys.path.append('lightrag_integration')

from lightrag_integration.config import LightRAGConfig
from lightrag_integration.integration_wrapper import IntegratedQueryService, QueryRequest

async def performance_test():
    """Test response times for both services."""
    config = LightRAGConfig.from_environment()
    service = IntegratedQueryService(
        config=config,
        perplexity_api_key=os.getenv('PERPLEXITY_API'),
        logger=None
    )
    
    test_query = "What is clinical metabolomics?"
    iterations = 5
    
    lightrag_times = []
    perplexity_times = []
    
    print(f"Running performance test with {iterations} iterations...")
    
    for i in range(iterations):
        # Test with LightRAG enabled
        request = QueryRequest(
            query_text=test_query,
            user_id="perf_test",
            session_id=f"session_{i}",
            query_type="clinical_metabolomics"
        )
        
        start_time = time.time()
        response = await service.query_async(request)
        response_time = time.time() - start_time
        
        if response.response_type.value == "lightrag":
            lightrag_times.append(response_time)
        elif response.response_type.value == "perplexity":
            perplexity_times.append(response_time)
        
        print(f"  Iteration {i+1}: {response.response_type.value} - {response_time:.2f}s")
        await asyncio.sleep(2)  # Rate limiting
    
    # Print statistics
    if lightrag_times:
        print(f"\nLightRAG Performance:")
        print(f"  Samples: {len(lightrag_times)}")
        print(f"  Average: {statistics.mean(lightrag_times):.2f}s")
        print(f"  Median: {statistics.median(lightrag_times):.2f}s")
        print(f"  Min/Max: {min(lightrag_times):.2f}s / {max(lightrag_times):.2f}s")
    
    if perplexity_times:
        print(f"\nPerplexity Performance:")
        print(f"  Samples: {len(perplexity_times)}")
        print(f"  Average: {statistics.mean(perplexity_times):.2f}s")
        print(f"  Median: {statistics.median(perplexity_times):.2f}s")
        print(f"  Min/Max: {min(perplexity_times):.2f}s / {max(perplexity_times):.2f}s")

if __name__ == "__main__":
    asyncio.run(performance_test())
EOF

python3 test_performance.py
```

---

## 4. Development Best Practices

### 4.1 Coding Standards

#### Error Handling Pattern

Always use this error handling pattern in your integration code:

```python
# ✅ Good: Comprehensive error handling
async def query_with_error_handling(service, query_text):
    """Example of proper error handling in LightRAG integration."""
    try:
        request = QueryRequest(
            query_text=query_text,
            user_id="user",
            session_id="session",
            query_type="clinical_metabolomics",
            timeout_seconds=30.0
        )
        
        response = await service.query_async(request)
        
        if not response.is_success:
            logger.warning(f"Query failed: {response.error_details}")
            return None
            
        return response.content
        
    except asyncio.TimeoutError:
        logger.error(f"Query timeout for: {query_text[:50]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in query processing: {e}")
        return None

# ❌ Bad: Insufficient error handling
async def query_bad_example(service, query_text):
    response = await service.query_async(query_text)  # No try/catch
    return response.content  # No validation
```

#### Configuration Validation

Always validate configuration before using:

```python
# ✅ Good: Configuration validation
def validate_integration_config():
    """Validate all required configuration before starting."""
    required_env_vars = [
        "OPENAI_API_KEY",
        "PERPLEXITY_API", 
        "LIGHTRAG_WORKING_DIR"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Validate file paths
    working_dir = Path(os.getenv("LIGHTRAG_WORKING_DIR"))
    if not working_dir.exists():
        try:
            working_dir.mkdir(parents=True)
        except OSError as e:
            raise ValueError(f"Cannot create working directory {working_dir}: {e}")
    
    logger.info("✅ Configuration validation passed")

# ❌ Bad: No validation
def bad_config_example():
    config = LightRAGConfig()  # Assumes everything works
    return config
```

### 4.2 Logging Recommendations

#### Structured Logging Setup

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Example structured logger for LightRAG integration."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.setup_handler()
    
    def setup_handler(self):
        """Setup structured logging handler."""
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_query_event(self, event_type: str, **kwargs):
        """Log query-related events with structured data."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            **kwargs
        }
        self.logger.info(f"QUERY_EVENT: {json.dumps(log_data)}")
    
    def log_performance_metric(self, metric_name: str, value: float, **context):
        """Log performance metrics."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_name": metric_name,
            "value": value,
            **context
        }
        self.logger.info(f"PERFORMANCE_METRIC: {json.dumps(log_data)}")

# Usage example
logger = StructuredLogger("lightrag_integration")

# Log query events
logger.log_query_event(
    "query_started",
    query_id="abc123",
    user_id="user456", 
    service="lightrag"
)

# Log performance metrics
logger.log_performance_metric(
    "response_time",
    2.34,
    service="lightrag",
    query_type="clinical_metabolomics"
)
```

#### Log Analysis Helpers

```bash
# Create log analysis script
cat > analyze_logs.sh << 'EOF'
#!/bin/bash

LOG_FILE="logs/lightrag_integration.log"

echo "=== LightRAG Integration Log Analysis ==="
echo

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "📊 Query Statistics:"
echo "  Total queries: $(grep -c 'QUERY_EVENT.*query_started' "$LOG_FILE")"
echo "  LightRAG queries: $(grep -c 'service.*lightrag' "$LOG_FILE")"
echo "  Perplexity queries: $(grep -c 'service.*perplexity' "$LOG_FILE")"
echo

echo "⚠️  Error Summary:"
echo "  Total errors: $(grep -c 'ERROR' "$LOG_FILE")"
echo "  Timeout errors: $(grep -c 'timeout' "$LOG_FILE")"
echo "  API errors: $(grep -c 'API.*error' "$LOG_FILE")"
echo

echo "⚡ Performance Metrics (last 10):"
grep 'PERFORMANCE_METRIC.*response_time' "$LOG_FILE" | tail -10 | \
    while read line; do
        echo "  $line" | sed 's/.*"value": \([0-9.]*\).*/Response time: \1s/'
    done

echo
echo "🔧 Recent Routing Decisions:"
grep 'routing.*decision' "$LOG_FILE" | tail -5
EOF

chmod +x analyze_logs.sh
./analyze_logs.sh
```

### 4.3 Security Considerations

#### API Key Protection

```python
import os
import re
from typing import Optional

def secure_api_key_validation(api_key: Optional[str], service_name: str) -> bool:
    """Validate API key format without logging the actual key."""
    if not api_key:
        logger.error(f"{service_name} API key not provided")
        return False
    
    # Define expected patterns (don't log actual keys)
    patterns = {
        "openai": r"^sk-[a-zA-Z0-9]{48}$",
        "perplexity": r"^pplx-[a-zA-Z0-9]{40}$"
    }
    
    pattern = patterns.get(service_name.lower())
    if pattern and not re.match(pattern, api_key):
        logger.error(f"{service_name} API key format invalid")
        return False
    
    logger.info(f"✅ {service_name} API key format validated")
    return True

def mask_sensitive_data(data_dict: dict) -> dict:
    """Mask sensitive information in logging data."""
    masked = data_dict.copy()
    
    sensitive_keys = ['api_key', 'token', 'password', 'secret']
    for key in sensitive_keys:
        if key in masked:
            masked[key] = "***MASKED***"
    
    return masked

# Usage
config_data = {
    "api_key": "sk-real-api-key-here",
    "model": "gpt-4o-mini",
    "timeout": 30
}

# Safe to log
logger.info(f"Config loaded: {mask_sensitive_data(config_data)}")
```

#### Input Validation

```python
import re
from typing import Optional

class QueryValidator:
    """Validate and sanitize user queries."""
    
    MAX_QUERY_LENGTH = 2000
    MIN_QUERY_LENGTH = 3
    
    @staticmethod
    def validate_query_text(query: str) -> tuple[bool, Optional[str]]:
        """Validate query text and return (is_valid, error_message)."""
        if not query or not query.strip():
            return False, "Query text cannot be empty"
        
        query = query.strip()
        
        if len(query) < QueryValidator.MIN_QUERY_LENGTH:
            return False, f"Query too short (minimum {QueryValidator.MIN_QUERY_LENGTH} characters)"
        
        if len(query) > QueryValidator.MAX_QUERY_LENGTH:
            return False, f"Query too long (maximum {QueryValidator.MAX_QUERY_LENGTH} characters)"
        
        # Check for potential injection attempts
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'data:.*base64',
            r'exec\s*\(',
            r'eval\s*\(',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains potentially unsafe content"
        
        return True, None
    
    @staticmethod
    def sanitize_query(query: str) -> str:
        """Sanitize query text for safe processing."""
        # Remove control characters
        query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query

# Usage example
validator = QueryValidator()

def process_user_query(raw_query: str) -> Optional[str]:
    """Process and validate user query."""
    is_valid, error_message = validator.validate_query_text(raw_query)
    
    if not is_valid:
        logger.warning(f"Invalid query rejected: {error_message}")
        return None
    
    sanitized_query = validator.sanitize_query(raw_query)
    logger.info(f"Query validated and sanitized: {len(sanitized_query)} chars")
    
    return sanitized_query
```

---

## 5. Common Implementation Patterns

### 5.1 Integration Wrapper Usage

#### Basic Usage Pattern

```python
from lightrag_integration.integration_wrapper import IntegratedQueryService, QueryRequest

async def basic_query_example():
    """Basic query using the integration wrapper."""
    # Initialize service
    service = IntegratedQueryService(
        config=LightRAGConfig.from_environment(),
        perplexity_api_key=os.getenv('PERPLEXITY_API'),
        logger=logging.getLogger(__name__)
    )
    
    # Create request
    request = QueryRequest(
        query_text="What are metabolic biomarkers?",
        user_id="example_user",
        session_id="example_session",
        query_type="clinical_metabolomics"
    )
    
    # Execute query
    response = await service.query_async(request)
    
    # Handle response
    if response.is_success:
        print(f"Response: {response.content}")
        print(f"Service used: {response.response_type.value}")
        if response.citations:
            print(f"Citations: {len(response.citations)}")
    else:
        print(f"Query failed: {response.error_details}")
```

#### Advanced Usage with Callbacks

```python
async def advanced_query_example():
    """Advanced query with custom callbacks and monitoring."""
    
    # Define callbacks for monitoring
    def on_routing_decision(decision_info):
        logger.info(f"Routing decision: {decision_info}")
    
    def on_performance_metric(metric_name, value):
        logger.info(f"Performance: {metric_name} = {value}")
    
    # Initialize service with callbacks
    service = IntegratedQueryService(
        config=LightRAGConfig.from_environment(),
        perplexity_api_key=os.getenv('PERPLEXITY_API'),
        logger=logging.getLogger(__name__),
        callbacks={
            'routing_decision': on_routing_decision,
            'performance_metric': on_performance_metric
        }
    )
    
    # Create request with metadata
    request = QueryRequest(
        query_text="Explain LC-MS/MS in metabolomics",
        user_id="advanced_user",
        session_id="advanced_session",
        query_type="clinical_metabolomics",
        timeout_seconds=45.0,
        metadata={
            "priority": "high",
            "client_version": "1.0.0",
            "experiment_group": "A"
        }
    )
    
    # Execute with context manager for cleanup
    async with service:
        response = await service.query_async(request)
        
        # Process response with quality assessment
        if response.is_success:
            quality_score = response.average_quality_score
            logger.info(f"Response quality score: {quality_score}")
            
            if quality_score < 0.7:
                logger.warning("Low quality response detected")
            
            return response
        else:
            logger.error(f"Query failed: {response.error_details}")
            return None
```

### 5.2 Feature Flag Management

#### Environment-Based Configuration

```python
class FeatureFlagConfig:
    """Centralized feature flag configuration."""
    
    @staticmethod
    def get_rollout_config() -> dict:
        """Get current rollout configuration from environment."""
        return {
            "integration_enabled": os.getenv('LIGHTRAG_INTEGRATION_ENABLED', 'false').lower() == 'true',
            "rollout_percentage": float(os.getenv('LIGHTRAG_ROLLOUT_PERCENTAGE', '0.0')),
            "ab_testing_enabled": os.getenv('LIGHTRAG_ENABLE_AB_TESTING', 'true').lower() == 'true',
            "circuit_breaker_enabled": os.getenv('LIGHTRAG_ENABLE_CIRCUIT_BREAKER', 'true').lower() == 'true',
            "fallback_enabled": os.getenv('LIGHTRAG_FALLBACK_TO_PERPLEXITY', 'true').lower() == 'true'
        }
    
    @staticmethod
    def get_user_cohort_assignment(user_id: str) -> str:
        """Assign user to cohort for A/B testing."""
        import hashlib
        
        salt = os.getenv('LIGHTRAG_USER_HASH_SALT', 'default_salt')
        hash_input = f"{user_id}_{salt}".encode()
        user_hash = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
        
        # Assign to cohort based on hash
        if user_hash % 2 == 0:
            return "lightrag_group"
        else:
            return "perplexity_group"
    
    @staticmethod
    def should_use_lightrag(user_id: str, session_id: str = None) -> dict:
        """Determine if user should use LightRAG."""
        config = FeatureFlagConfig.get_rollout_config()
        
        result = {
            "use_lightrag": False,
            "reason": "integration_disabled",
            "cohort": "none",
            "rollout_eligible": False
        }
        
        # Check if integration is enabled
        if not config["integration_enabled"]:
            return result
        
        # Check rollout percentage
        import hashlib
        hash_input = f"{user_id}_{session_id or ''}".encode()
        user_hash = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
        rollout_bucket = user_hash % 100
        
        if rollout_bucket >= config["rollout_percentage"]:
            result["reason"] = "outside_rollout_percentage"
            return result
        
        result["rollout_eligible"] = True
        
        # A/B testing assignment
        if config["ab_testing_enabled"]:
            cohort = FeatureFlagConfig.get_user_cohort_assignment(user_id)
            result["cohort"] = cohort
            
            if cohort == "lightrag_group":
                result["use_lightrag"] = True
                result["reason"] = "ab_test_assignment"
            else:
                result["reason"] = "ab_test_control_group"
        else:
            # No A/B testing, all eligible users get LightRAG
            result["use_lightrag"] = True
            result["reason"] = "full_rollout"
            result["cohort"] = "lightrag_group"
        
        return result

# Usage example
def route_query_example(user_id: str, session_id: str, query: str):
    """Example of feature flag-based query routing."""
    routing_decision = FeatureFlagConfig.should_use_lightrag(user_id, session_id)
    
    logger.info(f"Routing decision for user {user_id}: {routing_decision}")
    
    if routing_decision["use_lightrag"]:
        logger.info("Routing to LightRAG service")
        # Route to LightRAG
    else:
        logger.info(f"Routing to Perplexity: {routing_decision['reason']}")
        # Route to Perplexity
```

#### Dynamic Feature Flag Updates

```python
import json
from pathlib import Path

class DynamicFeatureFlags:
    """Dynamic feature flags that can be updated without restart."""
    
    def __init__(self, config_file: str = "feature_flags.json"):
        self.config_file = Path(config_file)
        self._flags = {}
        self._last_modified = 0
        self.load_flags()
    
    def load_flags(self):
        """Load feature flags from file."""
        if not self.config_file.exists():
            # Create default config
            default_flags = {
                "lightrag_integration_enabled": False,
                "lightrag_rollout_percentage": 0.0,
                "lightrag_enable_ab_testing": True,
                "lightrag_enable_circuit_breaker": True,
                "lightrag_min_quality_threshold": 0.7,
                "last_updated": int(time.time())
            }
            self.save_flags(default_flags)
            return
        
        try:
            stat = self.config_file.stat()
            if stat.st_mtime > self._last_modified:
                with open(self.config_file) as f:
                    self._flags = json.load(f)
                self._last_modified = stat.st_mtime
                logger.info("Feature flags reloaded from file")
        except Exception as e:
            logger.error(f"Failed to load feature flags: {e}")
    
    def save_flags(self, flags: dict):
        """Save feature flags to file."""
        try:
            flags["last_updated"] = int(time.time())
            with open(self.config_file, 'w') as f:
                json.dump(flags, f, indent=2)
            logger.info("Feature flags saved to file")
        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")
    
    def get_flag(self, flag_name: str, default=None):
        """Get feature flag value with automatic refresh."""
        self.load_flags()  # Auto-refresh
        return self._flags.get(flag_name, default)
    
    def set_flag(self, flag_name: str, value):
        """Set feature flag value and persist."""
        self._flags[flag_name] = value
        self.save_flags(self._flags)
    
    def update_rollout(self, percentage: float):
        """Update rollout percentage."""
        if 0.0 <= percentage <= 100.0:
            self.set_flag("lightrag_rollout_percentage", percentage)
            logger.info(f"Rollout percentage updated to {percentage}%")
        else:
            raise ValueError("Rollout percentage must be between 0.0 and 100.0")

# Usage example
flags = DynamicFeatureFlags()

# Check current rollout
current_rollout = flags.get_flag("lightrag_rollout_percentage", 0.0)
logger.info(f"Current rollout percentage: {current_rollout}%")

# Gradually increase rollout
if current_rollout < 10.0:
    flags.update_rollout(min(current_rollout + 1.0, 10.0))
```

### 5.3 A/B Testing Setup

#### Comprehensive A/B Test Framework

```python
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum

class ABTestStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused" 
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class ABTestMetrics:
    """Metrics for A/B test tracking."""
    group_name: str
    sample_size: int = 0
    success_count: int = 0
    total_response_time: float = 0.0
    total_quality_score: float = 0.0
    error_count: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.sample_size if self.sample_size > 0 else 0.0
    
    @property
    def avg_response_time(self) -> float:
        return self.total_response_time / self.sample_size if self.sample_size > 0 else 0.0
    
    @property
    def avg_quality_score(self) -> float:
        return self.total_quality_score / self.sample_size if self.sample_size > 0 else 0.0

class ABTestManager:
    """Manage A/B testing for LightRAG integration."""
    
    def __init__(self, test_id: str, results_file: str = "ab_test_results.json"):
        self.test_id = test_id
        self.results_file = Path(results_file)
        self.metrics = {
            "lightrag": ABTestMetrics("lightrag"),
            "perplexity": ABTestMetrics("perplexity")
        }
        self.start_time = time.time()
        self.status = ABTestStatus.ACTIVE
        
        self.load_existing_results()
    
    def load_existing_results(self):
        """Load existing A/B test results if available."""
        if self.results_file.exists():
            try:
                with open(self.results_file) as f:
                    data = json.load(f)
                    if self.test_id in data:
                        test_data = data[self.test_id]
                        for group_name, metrics_data in test_data.get("metrics", {}).items():
                            if group_name in self.metrics:
                                self.metrics[group_name] = ABTestMetrics(**metrics_data)
                        self.start_time = test_data.get("start_time", self.start_time)
                        self.status = ABTestStatus(test_data.get("status", "active"))
            except Exception as e:
                logger.error(f"Failed to load A/B test results: {e}")
    
    def record_query_result(self, group: str, success: bool, response_time: float, 
                           quality_score: float = 0.0):
        """Record the result of a query for A/B testing."""
        if group not in self.metrics:
            return
        
        metrics = self.metrics[group]
        metrics.sample_size += 1
        
        if success:
            metrics.success_count += 1
            metrics.total_response_time += response_time
            metrics.total_quality_score += quality_score
        else:
            metrics.error_count += 1
        
        # Save results periodically
        if metrics.sample_size % 10 == 0:
            self.save_results()
    
    def save_results(self):
        """Save A/B test results to file."""
        try:
            existing_data = {}
            if self.results_file.exists():
                with open(self.results_file) as f:
                    existing_data = json.load(f)
            
            test_results = {
                "test_id": self.test_id,
                "start_time": self.start_time,
                "status": self.status.value,
                "metrics": {name: asdict(metrics) for name, metrics in self.metrics.items()},
                "last_updated": time.time()
            }
            
            existing_data[self.test_id] = test_results
            
            with open(self.results_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save A/B test results: {e}")
    
    def get_statistical_significance(self) -> dict:
        """Calculate statistical significance between groups."""
        lightrag_metrics = self.metrics["lightrag"]
        perplexity_metrics = self.metrics["perplexity"]
        
        # Simple statistical analysis (you may want to use proper statistical libraries)
        total_samples = lightrag_metrics.sample_size + perplexity_metrics.sample_size
        
        if total_samples < 30:  # Minimum sample size
            return {
                "significant": False,
                "reason": "insufficient_sample_size",
                "required_samples": 30,
                "current_samples": total_samples
            }
        
        # Calculate confidence intervals and significance
        # This is a simplified version - consider using scipy.stats for proper analysis
        lightrag_rate = lightrag_metrics.success_rate
        perplexity_rate = perplexity_metrics.success_rate
        
        rate_difference = abs(lightrag_rate - perplexity_rate)
        
        return {
            "significant": rate_difference > 0.05,  # 5% difference threshold
            "lightrag_success_rate": lightrag_rate,
            "perplexity_success_rate": perplexity_rate,
            "rate_difference": rate_difference,
            "sample_sizes": {
                "lightrag": lightrag_metrics.sample_size,
                "perplexity": perplexity_metrics.sample_size
            }
        }
    
    def generate_report(self) -> str:
        """Generate A/B test report."""
        lightrag = self.metrics["lightrag"]
        perplexity = self.metrics["perplexity"]
        significance = self.get_statistical_significance()
        
        report = f"""
A/B Test Report: {self.test_id}
==============================

Test Duration: {(time.time() - self.start_time) / 3600:.1f} hours
Status: {self.status.value}

LightRAG Group:
  - Sample size: {lightrag.sample_size}
  - Success rate: {lightrag.success_rate:.2%}
  - Avg response time: {lightrag.avg_response_time:.2f}s
  - Avg quality score: {lightrag.avg_quality_score:.2f}
  - Error count: {lightrag.error_count}

Perplexity Group:
  - Sample size: {perplexity.sample_size}
  - Success rate: {perplexity.success_rate:.2%}
  - Avg response time: {perplexity.avg_response_time:.2f}s
  - Avg quality score: {perplexity.avg_quality_score:.2f}
  - Error count: {perplexity.error_count}

Statistical Significance:
  - Significant difference: {significance['significant']}
  - Rate difference: {significance.get('rate_difference', 0):.2%}
  
Recommendation:
"""
        
        if significance['significant']:
            if lightrag.success_rate > perplexity.success_rate:
                report += "  → Consider increasing LightRAG rollout percentage"
            else:
                report += "  → Consider reducing LightRAG rollout or investigating issues"
        else:
            report += "  → Continue testing - no significant difference detected"
        
        return report

# Usage in integration
ab_test_manager = ABTestManager("lightrag_rollout_2025_08")

async def query_with_ab_testing(service, request: QueryRequest):
    """Execute query with A/B testing tracking."""
    start_time = time.time()
    
    try:
        response = await service.query_async(request)
        response_time = time.time() - start_time
        
        # Record result for A/B testing
        ab_test_manager.record_query_result(
            group=response.response_type.value,
            success=response.is_success,
            response_time=response_time,
            quality_score=response.average_quality_score
        )
        
        return response
        
    except Exception as e:
        response_time = time.time() - start_time
        # Record failure
        ab_test_manager.record_query_result(
            group="error",
            success=False,
            response_time=response_time
        )
        raise

# Generate periodic reports
def generate_daily_ab_report():
    """Generate and log daily A/B test report."""
    report = ab_test_manager.generate_report()
    logger.info(f"Daily A/B Test Report:\n{report}")
    
    # Save report to file
    with open(f"reports/ab_test_report_{int(time.time())}.txt", "w") as f:
        f.write(report)
```

---

## 6. Quick Reference Commands

### 6.1 Development Commands

```bash
# === Environment Setup ===
# Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt -r requirements_lightrag.txt

# Initialize knowledge base
python3 initialize_kb.py

# === Testing Commands ===
# Test configuration
python3 -c "from lightrag_integration.config import LightRAGConfig; print('✅ Config OK' if LightRAGConfig.from_environment() else '❌ Config failed')"

# Test API keys
python3 test_api_keys.py

# Run validation suite
python3 validate_integration.py

# Performance test
python3 test_performance.py

# === Development Server ===
# Start with integration disabled (safe default)
LIGHTRAG_INTEGRATION_ENABLED=false chainlit run src/main.py -w

# Start with 1% rollout
LIGHTRAG_INTEGRATION_ENABLED=true LIGHTRAG_ROLLOUT_PERCENTAGE=1.0 chainlit run src/main.py -w

# Start with full LightRAG (development only)
LIGHTRAG_INTEGRATION_ENABLED=true LIGHTRAG_ROLLOUT_PERCENTAGE=100.0 chainlit run src/main.py -w

# === Monitoring Commands ===
# Monitor logs in real-time
tail -f logs/lightrag_integration.log | grep -E "(ERROR|routing|fallback)"

# Analyze recent performance
./analyze_logs.sh

# Check system health
python3 -c "
import asyncio
from lightrag_integration.integration_wrapper import IntegratedQueryService
from lightrag_integration.config import LightRAGConfig

async def health_check():
    service = IntegratedQueryService(LightRAGConfig.from_environment(), None, None)
    status = await service.health_check()
    print('Health Status:', status)

asyncio.run(health_check())
"

# === Deployment Commands ===
# Gradual rollout increase
export LIGHTRAG_ROLLOUT_PERCENTAGE=5.0  # Increase by 5%

# Emergency disable
export LIGHTRAG_INTEGRATION_ENABLED=false

# Check current feature flags
env | grep LIGHTRAG

# === Debugging Commands ===
# Check knowledge base status
python3 -c "
from pathlib import Path
kb_dir = Path('lightrag_data')
if kb_dir.exists():
    files = list(kb_dir.rglob('*'))
    print(f'Knowledge base files: {len(files)}')
    for f in files[:5]:
        print(f'  {f}')
else:
    print('Knowledge base not found')
"

# Test single query
python3 -c "
import asyncio
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from lightrag_integration.config import LightRAGConfig

async def test_query():
    rag = ClinicalMetabolomicsRAG(LightRAGConfig.from_environment())
    response = await rag.query_async('What is metabolomics?', mode='hybrid')
    print(f'Response: {response.content[:200]}...')

asyncio.run(test_query())
"
```

### 6.2 Environment Variables Quick Reference

```bash
# === Required Variables ===
export OPENAI_API_KEY="sk-your-key-here"
export PERPLEXITY_API="your-perplexity-key"
export LIGHTRAG_WORKING_DIR="./lightrag_data"

# === Integration Control ===
export LIGHTRAG_INTEGRATION_ENABLED="true"       # Enable/disable integration
export LIGHTRAG_ROLLOUT_PERCENTAGE="10.0"        # Rollout percentage (0.0-100.0)
export LIGHTRAG_ENABLE_AB_TESTING="true"         # Enable A/B testing
export LIGHTRAG_FALLBACK_TO_PERPLEXITY="true"    # Enable fallback

# === Performance Tuning ===
export LIGHTRAG_MAX_ASYNC="16"                   # Max concurrent operations
export LIGHTRAG_MAX_TOKENS="32768"               # Token limit
export LIGHTRAG_MODEL="gpt-4o-mini"             # LLM model
export LIGHTRAG_EMBEDDING_MODEL="text-embedding-3-small"  # Embedding model

# === Circuit Breaker ===
export LIGHTRAG_ENABLE_CIRCUIT_BREAKER="true"
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD="3"
export LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT="300"

# === Logging ===
export LIGHTRAG_LOG_LEVEL="INFO"                 # DEBUG, INFO, WARNING, ERROR
export LIGHTRAG_ENABLE_FILE_LOGGING="true"
export LIGHTRAG_LOG_DIR="./logs"

# === Cost Tracking ===
export LIGHTRAG_ENABLE_COST_TRACKING="true"
export LIGHTRAG_DAILY_BUDGET_LIMIT="10.0"        # Daily budget in USD
export LIGHTRAG_COST_ALERT_THRESHOLD="80.0"      # Alert at 80% of budget
```

### 6.3 Common Integration Patterns

```python
# === Pattern 1: Basic Integration ===
from lightrag_integration import IntegratedQueryService, QueryRequest

service = IntegratedQueryService(config, perplexity_key, logger)
request = QueryRequest(query_text="test", user_id="user", session_id="session")
response = await service.query_async(request)

# === Pattern 2: Error Handling ===
try:
    response = await service.query_async(request)
    if response.is_success:
        return response.content
    else:
        logger.error(f"Query failed: {response.error_details}")
        return fallback_response
except Exception as e:
    logger.error(f"Service error: {e}")
    return error_response

# === Pattern 3: Performance Monitoring ===
start_time = time.time()
response = await service.query_async(request)
response_time = time.time() - start_time

logger.info(f"Query completed in {response_time:.2f}s using {response.response_type.value}")

# === Pattern 4: Feature Flag Checking ===
from lightrag_integration.feature_flag_manager import FeatureFlagManager

flags = FeatureFlagManager(config, logger)
routing_decision = flags.should_use_lightrag(user_id, session_id)

if routing_decision.use_lightrag:
    # Use LightRAG
    pass
else:
    # Use Perplexity
    pass

# === Pattern 5: A/B Testing ===
ab_test = ABTestManager("test_id")

response = await service.query_async(request)
ab_test.record_query_result(
    group=response.response_type.value,
    success=response.is_success,
    response_time=response.processing_time
)
```

---

## 7. Troubleshooting Common Issues

### 7.1 Integration Issues

#### Issue: "LightRAG integration service not found"

**Symptoms:**
```
ModuleNotFoundError: No module named 'lightrag_integration'
```

**Solution:**
```bash
# Check if the module path is correct
ls -la lightrag_integration/
python3 -c "import sys; print('\n'.join(sys.path))"

# Add to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to your script
import sys
sys.path.append('lightrag_integration')
```

#### Issue: "Configuration validation failed"

**Symptoms:**
```
LightRAGConfigError: Missing required environment variables: ['OPENAI_API_KEY']
```

**Solution:**
```bash
# Check environment variables
env | grep -E "(OPENAI_API_KEY|PERPLEXITY_API|LIGHTRAG_)"

# Verify .env file is loaded
python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; print('API key loaded:', bool(os.getenv('OPENAI_API_KEY')))"

# Create .env file if missing
cat > .env << 'EOF'
OPENAI_API_KEY=your_key_here
PERPLEXITY_API=your_key_here
LIGHTRAG_WORKING_DIR=./lightrag_data
EOF
```

#### Issue: "Knowledge base initialization failed"

**Symptoms:**
```
lightrag.base.logger - ERROR - Failed to initialize knowledge base
```

**Solution:**
```bash
# Check directory permissions
ls -la lightrag_data/
chmod 755 lightrag_data/

# Check disk space
df -h .

# Verify OpenAI API key
python3 -c "
import openai
client = openai.OpenAI(api_key='your_key')
response = client.embeddings.create(input='test', model='text-embedding-3-small')
print('API test successful')
"

# Reinitialize with debug logging
LIGHTRAG_LOG_LEVEL=DEBUG python3 initialize_kb.py
```

### 7.2 Runtime Issues

#### Issue: "All queries routing to Perplexity"

**Symptoms:**
- Logs show "Query routed to: perplexity" for all queries
- No LightRAG queries being processed

**Debugging Steps:**
```python
# Check feature flag status
python3 -c "
from lightrag_integration.config import LightRAGConfig
config = LightRAGConfig.from_environment()
print(f'Integration enabled: {config.lightrag_integration_enabled}')
print(f'Rollout percentage: {config.lightrag_rollout_percentage}%')
"

# Test routing decision
python3 -c "
from lightrag_integration.feature_flag_manager import FeatureFlagManager, RoutingContext
from lightrag_integration.config import LightRAGConfig
config = LightRAGConfig.from_environment()
manager = FeatureFlagManager(config, None)
context = RoutingContext(user_id='test_user', session_id='test_session')
decision = manager.should_use_lightrag(context)
print(f'Routing decision: {decision}')
"
```

**Solutions:**
```bash
# Enable integration
export LIGHTRAG_INTEGRATION_ENABLED=true

# Increase rollout percentage
export LIGHTRAG_ROLLOUT_PERCENTAGE=100.0  # For testing

# Disable circuit breaker temporarily
export LIGHTRAG_ENABLE_CIRCUIT_BREAKER=false

# Check circuit breaker status
python3 -c "
from lightrag_integration.integration_wrapper import CircuitBreaker
# Check if circuit breaker is open and reset if needed
"
```

#### Issue: "Slow response times"

**Symptoms:**
- Queries taking >10 seconds to respond
- Timeout errors in logs

**Debugging:**
```bash
# Monitor response times
tail -f logs/lightrag_integration.log | grep -E "(response_time|timeout)"

# Check system resources
htop  # Check CPU/memory usage
iostat 1  # Check disk I/O

# Test with smaller queries
python3 -c "
import asyncio
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from lightrag_integration.config import LightRAGConfig
import time

async def test_performance():
    rag = ClinicalMetabolomicsRAG(LightRAGConfig.from_environment())
    start = time.time()
    response = await rag.query_async('Short test query', mode='naive')  # Use fastest mode
    print(f'Response time: {time.time() - start:.2f}s')

asyncio.run(test_performance())
"
```

**Solutions:**
```bash
# Reduce concurrency
export LIGHTRAG_MAX_ASYNC=8  # Reduce from 16

# Use faster query mode
# In your code, use mode='naive' or 'local' instead of 'hybrid'

# Increase timeout
export LIGHTRAG_QUERY_TIMEOUT=60  # Increase timeout

# Enable performance optimizations
export LIGHTRAG_ENABLE_PERFORMANCE_OPTIMIZATION=true
```

#### Issue: "High API costs"

**Symptoms:**
- Budget alerts being triggered
- High OpenAI API usage

**Monitoring:**
```bash
# Check cost tracking
python3 -c "
from lightrag_integration.budget_manager import BudgetManager
from lightrag_integration.config import LightRAGConfig
manager = BudgetManager(LightRAGConfig.from_environment())
status = manager.get_budget_status()
print(f'Current usage: ${status.current_usage:.2f}')
print(f'Budget limit: ${status.budget_limit:.2f}')
print(f'Remaining: ${status.remaining_budget:.2f}')
"

# Analyze cost by model
grep -E "(gpt-4|text-embedding)" logs/lightrag_integration.log | tail -20
```

**Solutions:**
```bash
# Reduce budget limit
export LIGHTRAG_DAILY_BUDGET_LIMIT=5.0  # Reduce from 10.0

# Enable cost-based circuit breaker
export LIGHTRAG_ENABLE_COST_BASED_CIRCUIT_BREAKER=true
export LIGHTRAG_COST_CIRCUIT_BREAKER_THRESHOLD=80.0  # Stop at 80% budget

# Use cheaper models
export LIGHTRAG_MODEL=gpt-3.5-turbo  # Instead of gpt-4o-mini
export LIGHTRAG_EMBEDDING_MODEL=text-embedding-ada-002  # Cheaper embedding model

# Reduce rollout percentage
export LIGHTRAG_ROLLOUT_PERCENTAGE=10.0  # Reduce usage
```

### 7.3 Development Issues

#### Issue: "Tests failing after integration"

**Solution:**
```bash
# Run tests with integration disabled
LIGHTRAG_INTEGRATION_ENABLED=false python3 -m pytest

# Mock LightRAG for testing
# Add to your test file:
# @pytest.fixture(autouse=True)
# def mock_lightrag():
#     with patch('lightrag_integration.IntegratedQueryService') as mock:
#         yield mock

# Test only the integration components
python3 -m pytest tests/test_lightrag_integration.py -v
```

#### Issue: "Chainlit startup errors"

**Solution:**
```bash
# Start with minimal configuration
LIGHTRAG_INTEGRATION_ENABLED=false chainlit run src/main.py

# Check for import errors
python3 -c "
import src.main
print('Main module imports successfully')
"

# Debug chainlit issues
chainlit run src/main.py --debug
```

#### Issue: "PDF processing errors"

**Symptoms:**
```
Error processing PDF: /path/to/file.pdf
```

**Solution:**
```bash
# Check PDF file integrity
python3 -c "
import PyMuPDF
doc = PyMuPDF.open('papers/your_file.pdf')
print(f'PDF has {len(doc)} pages')
doc.close()
"

# Process PDFs individually
python3 -c "
import asyncio
from lightrag_integration.pdf_processor import PDFProcessor
from pathlib import Path

async def test_pdf():
    processor = PDFProcessor()
    for pdf_file in Path('papers').glob('*.pdf'):
        try:
            content = await processor.process_pdf(str(pdf_file))
            print(f'✅ {pdf_file.name}: {len(content)} characters')
        except Exception as e:
            print(f'❌ {pdf_file.name}: {e}')

asyncio.run(test_pdf())
"

# Skip problematic PDFs
mkdir -p papers/problematic
mv papers/problematic_file.pdf papers/problematic/
```

### 7.4 Emergency Procedures

#### Complete Rollback

```bash
# 1. Disable integration immediately
export LIGHTRAG_INTEGRATION_ENABLED=false

# 2. Restart service
pkill -f "chainlit run"
chainlit run src/main.py &

# 3. Verify fallback is working
curl -X POST http://localhost:8000/chat -d '{"message":"test"}' -H "Content-Type: application/json"
```

#### Reset Knowledge Base

```bash
# 1. Backup existing knowledge base
cp -r lightrag_data lightrag_data.backup.$(date +%Y%m%d_%H%M%S)

# 2. Clear knowledge base
rm -rf lightrag_data/*

# 3. Reinitialize
python3 initialize_kb.py
```

#### Debug Mode Activation

```bash
# Enable comprehensive debug logging
export LIGHTRAG_LOG_LEVEL=DEBUG
export LIGHTRAG_ENABLE_FILE_LOGGING=true
export LIGHTRAG_ENABLE_PERFORMANCE_MONITORING=true

# Start with debug mode
chainlit run src/main.py --debug

# Monitor all logs
tail -f logs/*.log
```

---

## Summary

This developer integration guide provides practical, step-by-step instructions for implementing the LightRAG integration with the Clinical Metabolomics Oracle. The guide covers:

✅ **Complete setup process** with validation steps  
✅ **Practical code examples** with error handling  
✅ **Comprehensive testing procedures** for validation  
✅ **Production-ready patterns** for feature flags and A/B testing  
✅ **Detailed troubleshooting** for common issues  
✅ **Quick reference commands** for daily development  

The integration maintains full backward compatibility while adding advanced capabilities through feature flags and gradual rollout mechanisms.

**Next Steps:**
1. Follow the setup procedures in order
2. Start with integration disabled for testing
3. Gradually enable with low rollout percentage
4. Monitor performance and adjust configuration
5. Scale rollout based on A/B test results

For additional support, refer to the comprehensive integration procedures document or the troubleshooting section above.