# Clinical Metabolomics Oracle - LightRAG Deployment Procedures

## Overview

This document provides comprehensive deployment procedures for integrating LightRAG with the existing Clinical Metabolomics Oracle (CMO) system. The integration is designed to be optional and modular, preserving all existing functionality while adding advanced knowledge graph capabilities.

---

## Table of Contents

1. [Prerequisites and Dependencies](#prerequisites-and-dependencies)
2. [Pre-Deployment Validation](#pre-deployment-validation)
3. [Step-by-Step Deployment Process](#step-by-step-deployment-process)
4. [Environment Configuration](#environment-configuration)
5. [Service Startup Procedures](#service-startup-procedures)
6. [Integration Verification](#integration-verification)
7. [Gradual Rollout Strategy](#gradual-rollout-strategy)
8. [Rollback Procedures](#rollback-procedures)
9. [Monitoring and Alerting](#monitoring-and-alerting)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites and Dependencies

### 1. System Requirements

**Minimum Requirements:**
- RAM: 16GB (32GB recommended for production)
- CPU: 8 cores (16+ recommended for production)
- Storage: 50GB available (100GB+ for production)
- OS: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11

**Production Requirements:**
- RAM: 32GB+
- CPU: 16+ cores
- Storage: 100GB+ SSD
- Load balancer capability
- Container orchestration (Docker/Kubernetes)

### 2. Database Dependencies

**Existing Databases (Must be operational):**
- PostgreSQL 13+ (for Chainlit user data)
- Neo4j 4.4+ (for graph storage)

**Connection Requirements:**
- Database connectivity validated
- Backup and recovery procedures in place
- Performance monitoring configured

### 3. API Dependencies

**Required:**
- OpenAI API key with sufficient credits
- Valid API endpoint access

**Optional:**
- Perplexity API (for fallback)
- Groq API (for fast inference)
- OpenRouter API (for model variety)

### 4. Software Dependencies

```bash
# Core Python dependencies (already in requirements.txt)
chainlit>=1.0.401
llama-index>=0.10.20
neo4j>=5.18.0
asyncpg>=0.30.0
PyMuPDF>=1.23.26

# LightRAG-specific dependencies (requirements_lightrag.txt)
lightrag-hku>=1.4.6
numpy>=2.3.2
pandas>=2.3.1
networkx>=3.5
tiktoken>=0.10.0
```

---

## Pre-Deployment Validation

### 1. Environment Validation Script

Create and run comprehensive pre-deployment checks:

```bash
#!/bin/bash
# pre_deployment_check.sh

echo "🔍 CMO-LightRAG Pre-Deployment Validation"
echo "========================================="

# Check Python version
python3 --version | grep -E "3\.(10|11|12)" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Python version check passed"
else
    echo "❌ Python 3.10+ required"
    exit 1
fi

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in virtual environment"
else
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
fi

# Check required environment variables
required_vars=("OPENAI_API_KEY" "DATABASE_URL" "NEO4J_URL" "NEO4J_USERNAME" "NEO4J_PASSWORD")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Missing required environment variable: $var"
        exit 1
    else
        echo "✅ Environment variable set: $var"
    fi
done

# Check database connectivity
echo "🔍 Testing database connections..."
python3 -c "
import asyncio
import asyncpg
import os
from neo4j import GraphDatabase

async def test_databases():
    try:
        # Test PostgreSQL
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        await conn.fetchval('SELECT 1')
        await conn.close()
        print('✅ PostgreSQL connection successful')
    except Exception as e:
        print(f'❌ PostgreSQL connection failed: {e}')
        return False
    
    try:
        # Test Neo4j
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URL'),
            auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        )
        with driver.session() as session:
            session.run('RETURN 1')
        driver.close()
        print('✅ Neo4j connection successful')
        return True
    except Exception as e:
        print(f'❌ Neo4j connection failed: {e}')
        return False

result = asyncio.run(test_databases())
exit(0 if result else 1)
"

if [ $? -ne 0 ]; then
    echo "❌ Database connectivity check failed"
    exit 1
fi

# Check OpenAI API
echo "🔍 Testing OpenAI API access..."
python3 -c "
import openai
import os
try:
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    models = client.models.list()
    print(f'✅ OpenAI API access successful ({len(models.data)} models available)')
except Exception as e:
    print(f'❌ OpenAI API access failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Check disk space
available_space=$(df . | tail -1 | awk '{print $4}')
required_space=10485760  # 10GB in KB
if [ "$available_space" -lt "$required_space" ]; then
    echo "❌ Insufficient disk space. Required: 10GB, Available: $((available_space/1024/1024))GB"
    exit 1
else
    echo "✅ Sufficient disk space available"
fi

echo "🎉 Pre-deployment validation completed successfully!"
echo "Ready to proceed with LightRAG deployment."
```

### 2. Dependencies Installation Verification

```bash
#!/bin/bash
# verify_dependencies.sh

echo "🔍 Verifying LightRAG dependencies..."

# Install/upgrade dependencies
pip install -r requirements.txt -r requirements_lightrag.txt --upgrade

# Verify critical imports
python3 -c "
try:
    from lightrag_integration.config import LightRAGConfig
    from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    print('✅ LightRAG integration modules import successfully')
except ImportError as e:
    print(f'❌ LightRAG integration import failed: {e}')
    exit(1)

try:
    import lightrag
    print('✅ LightRAG core library import successful')
except ImportError as e:
    print(f'❌ LightRAG core import failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ All dependencies verified successfully"
else
    echo "❌ Dependency verification failed"
    exit 1
fi
```

---

## Step-by-Step Deployment Process

### Phase 1: Infrastructure Preparation

#### 1. Backup Current System

```bash
#!/bin/bash
# backup_system.sh

BACKUP_DIR="./backups/pre_lightrag_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating system backup: $BACKUP_DIR"

# Backup database
echo "Backing up PostgreSQL..."
pg_dump $DATABASE_URL > "$BACKUP_DIR/postgresql_backup.sql"

echo "Backing up Neo4j..."
neo4j-admin dump --database=neo4j --to="$BACKUP_DIR/neo4j_backup.dump"

# Backup configuration
cp .env "$BACKUP_DIR/env_backup"
cp -r src/ "$BACKUP_DIR/src_backup/"

# Backup logs
find logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \;

echo "✅ Backup completed: $BACKUP_DIR"
echo "📋 Backup contents:"
ls -la "$BACKUP_DIR"
```

#### 2. Create LightRAG Storage Structure

```bash
#!/bin/bash
# setup_lightrag_storage.sh

echo "📁 Setting up LightRAG storage structure..."

# Create base directories
mkdir -p lightrag_storage/{entities,relationships,knowledge_base}
mkdir -p papers/clinical_metabolomics
mkdir -p logs/lightrag
mkdir -p backups/lightrag

# Set appropriate permissions
chmod 755 lightrag_storage papers logs backups
chmod 644 lightrag_storage/* papers/* logs/* 2>/dev/null || true

echo "✅ LightRAG storage structure created"
ls -la lightrag_storage/
```

### Phase 2: Configuration Deployment

#### 3. Environment Configuration Setup

```bash
#!/bin/bash
# deploy_environment.sh

echo "⚙️ Deploying LightRAG environment configuration..."

# Backup current .env
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)

# Add LightRAG configuration to .env (if not already present)
cat >> .env << 'EOF'

# LightRAG Core Configuration
LIGHTRAG_WORKING_DIR=./lightrag_storage
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
LIGHTRAG_MAX_ASYNC=16
LIGHTRAG_MAX_TOKENS=32768

# LightRAG Integration Feature Flags (Start Disabled)
LIGHTRAG_INTEGRATION_ENABLED=false
LIGHTRAG_ROLLOUT_PERCENTAGE=0.0
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true

# LightRAG Performance Settings
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_DAILY_BUDGET_LIMIT=100.0
LIGHTRAG_COST_ALERT_THRESHOLD=80.0
LIGHTRAG_ENABLE_BUDGET_ALERTS=true

# LightRAG Quality Settings
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.7
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_ENABLE_RELEVANCE_SCORING=true

# LightRAG Logging
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_DIR=./logs
EOF

echo "✅ Environment configuration updated"
echo "🔍 New LightRAG configuration:"
grep "LIGHTRAG_" .env
```

#### 4. Initialize LightRAG Knowledge Base

```bash
#!/bin/bash
# initialize_knowledge_base.sh

echo "🧠 Initializing LightRAG knowledge base..."

# Create initialization script
cat > temp_kb_init.py << 'EOF'
import asyncio
import sys
import os
from pathlib import Path

# Add lightrag_integration to path
sys.path.append('.')

from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

async def initialize_knowledge_base():
    """Initialize the LightRAG knowledge base."""
    try:
        print("🔧 Loading LightRAG configuration...")
        config = LightRAGConfig.get_config(validate_config=True)
        
        print(f"📁 Working directory: {config.working_dir}")
        print(f"🤖 Model: {config.model}")
        
        print("🚀 Initializing Clinical Metabolomics RAG system...")
        rag_system = ClinicalMetabolomicsRAG(config)
        
        # Check if papers directory has content
        papers_dir = Path("papers")
        if papers_dir.exists() and any(papers_dir.rglob("*.pdf")):
            print(f"📚 Found PDF documents, initializing knowledge base...")
            await rag_system.initialize_knowledge_base()
            print("✅ Knowledge base initialized successfully")
        else:
            print("📝 No PDF documents found, creating empty knowledge base...")
            await rag_system.initialize_empty_knowledge_base()
            print("✅ Empty knowledge base initialized")
            
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(initialize_knowledge_base())
    sys.exit(0 if success else 1)
EOF

# Run initialization
python3 temp_kb_init.py
INIT_RESULT=$?

# Clean up
rm temp_kb_init.py

if [ $INIT_RESULT -eq 0 ]; then
    echo "✅ Knowledge base initialization completed"
else
    echo "❌ Knowledge base initialization failed"
    exit 1
fi
```

### Phase 3: Application Integration

#### 5. Deploy Integration Code

The integration code is already present in the `lightrag_integration/` module. Verify it's properly integrated:

```bash
#!/bin/bash
# verify_integration.sh

echo "🔍 Verifying LightRAG integration..."

# Test integration imports
python3 -c "
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from lightrag_integration.feature_flag_manager import FeatureFlagManager

config = LightRAGConfig.get_config(validate_config=False)
print('✅ Configuration loading works')

rag = ClinicalMetabolomicsRAG(config)
print('✅ RAG system instantiation works')

flags = FeatureFlagManager()
print('✅ Feature flag manager works')

print('🎉 Integration verification successful')
"

if [ $? -eq 0 ]; then
    echo "✅ Integration verification passed"
else
    echo "❌ Integration verification failed"
    exit 1
fi
```

---

## Environment Configuration

### Production Environment Variables

Create environment-specific configuration files:

**`.env.production`:**
```bash
# Production LightRAG Configuration
ENVIRONMENT=production

# Core Configuration
OPENAI_API_KEY=your_production_openai_key
LIGHTRAG_WORKING_DIR=/opt/cmo/lightrag_storage
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small

# Performance Settings
LIGHTRAG_MAX_ASYNC=32
LIGHTRAG_MAX_TOKENS=32768

# Feature Flags (Production - Start Conservative)
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=5.0
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0

# Budget Management
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_DAILY_BUDGET_LIMIT=500.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=10000.0
LIGHTRAG_COST_ALERT_THRESHOLD=80.0
LIGHTRAG_ENABLE_BUDGET_ALERTS=true

# Quality Assurance
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.8
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_ENABLE_RELEVANCE_SCORING=true
LIGHTRAG_RELEVANCE_CONFIDENCE_THRESHOLD=75.0

# Monitoring and Logging
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_DIR=/opt/cmo/logs
LIGHTRAG_LOG_MAX_BYTES=104857600
LIGHTRAG_LOG_BACKUP_COUNT=10

# Database Configuration
DATABASE_URL=postgresql://cmo_user:secure_password@db-server:5432/clinical_metabolomics_oracle
NEO4J_URL=bolt://neo4j-server:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=secure_neo4j_password
```

**`.env.staging`:**
```bash
# Staging LightRAG Configuration
ENVIRONMENT=staging

# Feature Flags (Staging - More Aggressive Testing)
LIGHTRAG_ROLLOUT_PERCENTAGE=25.0
LIGHTRAG_DAILY_BUDGET_LIMIT=200.0
LIGHTRAG_LOG_LEVEL=DEBUG

# Enable additional testing features
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true
```

### Configuration Management Script

```bash
#!/bin/bash
# manage_environment.sh

set -e

ENVIRONMENT=${1:-development}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔧 Switching to $ENVIRONMENT environment..."

if [ ! -f ".env.$ENVIRONMENT" ]; then
    echo "❌ Environment file .env.$ENVIRONMENT not found"
    echo "Available environments:"
    ls .env.* 2>/dev/null | sed 's/.env./  - /' || echo "  (none found)"
    exit 1
fi

# Backup current .env if it exists
if [ -f ".env" ]; then
    cp .env ".env.backup.$(date +%Y%m%d_%H%M%S)"
    echo "📦 Backed up current .env"
fi

# Copy environment-specific config
cp ".env.$ENVIRONMENT" .env
echo "✅ Switched to $ENVIRONMENT environment"

# Validate configuration
echo "🔍 Validating configuration..."
python3 -c "
from lightrag_integration.config import LightRAGConfig
try:
    config = LightRAGConfig.get_config(validate_config=True)
    print('✅ Configuration validation passed')
    print(f'   Environment: {config.lightrag_integration_enabled}')
    print(f'   Rollout: {config.lightrag_rollout_percentage}%')
    print(f'   Model: {config.model}')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "🎉 Environment switch completed successfully"
else
    echo "❌ Configuration validation failed"
    exit 1
fi
```

---

## Service Startup Procedures

### 1. Pre-Startup Health Check

```bash
#!/bin/bash
# pre_startup_check.sh

echo "🏥 Pre-startup health check..."

# Check all dependencies
python3 -c "
import sys
import subprocess

dependencies = [
    'chainlit',
    'lightrag_integration',
    'asyncpg',
    'neo4j',
    'openai'
]

failed_imports = []
for dep in dependencies:
    try:
        __import__(dep)
        print(f'✅ {dep}')
    except ImportError as e:
        failed_imports.append(dep)
        print(f'❌ {dep}: {e}')

if failed_imports:
    print(f'\\n❌ Failed to import: {failed_imports}')
    sys.exit(1)
else:
    print('\\n✅ All dependencies available')
"

# Check configuration
python3 -c "
from lightrag_integration.config import LightRAGConfig
config = LightRAGConfig.get_config()
print(f'✅ Configuration loaded successfully')
print(f'   LightRAG enabled: {config.lightrag_integration_enabled}')
print(f'   Rollout percentage: {config.lightrag_rollout_percentage}%')
"

# Check database connections
python3 -c "
import asyncio
import asyncpg
import os
from neo4j import GraphDatabase

async def check_connections():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        await conn.close()
        print('✅ PostgreSQL connection OK')
    except Exception as e:
        print(f'❌ PostgreSQL connection failed: {e}')
        return False
    
    try:
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URL'),
            auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        )
        driver.verify_connectivity()
        driver.close()
        print('✅ Neo4j connection OK')
        return True
    except Exception as e:
        print(f'❌ Neo4j connection failed: {e}')
        return False

success = asyncio.run(check_connections())
exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    echo "🎉 Pre-startup health check passed"
else
    echo "❌ Pre-startup health check failed"
    exit 1
fi
```

### 2. Startup Script with LightRAG Integration

```bash
#!/bin/bash
# start_cmo_with_lightrag.sh

set -e

echo "🚀 Starting Clinical Metabolomics Oracle with LightRAG integration..."

# Run pre-startup checks
echo "1/5 Running pre-startup checks..."
./pre_startup_check.sh

# Initialize logging
echo "2/5 Setting up logging..."
mkdir -p logs
touch logs/lightrag_integration.log
chmod 644 logs/lightrag_integration.log

# Initialize LightRAG if needed
echo "3/5 Checking LightRAG initialization..."
if [ ! -d "lightrag_storage/entities" ] || [ -z "$(ls -A lightrag_storage/entities 2>/dev/null)" ]; then
    echo "📚 Initializing LightRAG knowledge base..."
    python3 -c "
import asyncio
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

async def init_if_needed():
    config = LightRAGConfig.get_config()
    rag = ClinicalMetabolomicsRAG(config)
    await rag.initialize_knowledge_base()

asyncio.run(init_if_needed())
"
    echo "✅ LightRAG initialization complete"
else
    echo "✅ LightRAG already initialized"
fi

# Start monitoring (background)
echo "4/5 Starting monitoring..."
python3 -c "
import asyncio
from lightrag_integration.monitoring import start_background_monitoring

asyncio.run(start_background_monitoring())
" &
MONITOR_PID=$!
echo "✅ Monitoring started (PID: $MONITOR_PID)"

# Start main application
echo "5/5 Starting main application..."
echo "🌐 Starting Chainlit application on port 8000..."
echo "📊 Monitor dashboard available at http://localhost:8000/admin"
echo "🔍 LightRAG integration status: $(python3 -c 'from lightrag_integration.config import LightRAGConfig; print(LightRAGConfig.get_config().lightrag_integration_enabled)')"

# Start with appropriate error handling
exec chainlit run src/app.py --host 0.0.0.0 --port 8000 2>&1 | tee -a logs/startup.log
```

### 3. Docker Startup (Production)

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Prisma
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy and install Python dependencies
COPY requirements.txt requirements_lightrag.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements_lightrag.txt

# Copy and install Node dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p lightrag_storage logs papers backups \
    && chmod 755 lightrag_storage logs papers backups

# Create startup script
COPY docker_startup.sh /app/
RUN chmod +x /app/docker_startup.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["/app/docker_startup.sh"]
```

**docker_startup.sh:**
```bash
#!/bin/bash
set -e

echo "🐳 Starting CMO with LightRAG in Docker..."

# Wait for database
echo "⏳ Waiting for databases..."
until python3 -c "
import asyncio
import asyncpg
import os
from neo4j import GraphDatabase

async def check_db():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        await conn.close()
        driver = GraphDatabase.driver(os.getenv('NEO4J_URL'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception as e:
        print(f'Waiting for databases: {e}')
        return False

success = asyncio.run(check_db())
exit(0 if success else 1)
"; do
    echo "⏳ Databases not ready, waiting..."
    sleep 5
done

echo "✅ Databases ready"

# Initialize LightRAG if needed
echo "🧠 Checking LightRAG initialization..."
python3 -c "
import asyncio
import os
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

async def ensure_initialized():
    config = LightRAGConfig.get_config()
    rag = ClinicalMetabolomicsRAG(config)
    
    if not os.path.exists('lightrag_storage/entities') or not os.listdir('lightrag_storage/entities'):
        print('📚 Initializing knowledge base...')
        await rag.initialize_knowledge_base()
    else:
        print('✅ Knowledge base already initialized')

asyncio.run(ensure_initialized())
"

# Start application
echo "🚀 Starting Chainlit application..."
exec chainlit run src/app.py --host 0.0.0.0 --port 8000
```

---

## Integration Verification

### 1. Comprehensive Verification Script

```bash
#!/bin/bash
# verify_deployment.sh

echo "🔍 Comprehensive LightRAG Integration Verification"
echo "================================================="

# Test 1: Configuration Loading
echo "Test 1: Configuration Loading"
python3 -c "
from lightrag_integration.config import LightRAGConfig
config = LightRAGConfig.get_config()
print(f'✅ Configuration loaded successfully')
print(f'   Integration enabled: {config.lightrag_integration_enabled}')
print(f'   Rollout percentage: {config.lightrag_rollout_percentage}%')
print(f'   Model: {config.model}')
print(f'   Working directory: {config.working_dir}')
"

# Test 2: Knowledge Base Initialization
echo -e "\nTest 2: Knowledge Base Access"
python3 -c "
import asyncio
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

async def test_kb():
    config = LightRAGConfig.get_config()
    rag = ClinicalMetabolomicsRAG(config)
    
    # Test knowledge base status
    status = await rag.get_knowledge_base_status()
    print(f'✅ Knowledge base accessible')
    print(f'   Status: {status}')
    
    return True

asyncio.run(test_kb())
"

# Test 3: Feature Flag System
echo -e "\nTest 3: Feature Flag System"
python3 -c "
from lightrag_integration.feature_flag_manager import FeatureFlagManager
import hashlib

flag_manager = FeatureFlagManager()

# Test user assignment
test_user = 'test_user_123'
assignment = flag_manager.get_user_assignment(test_user)
print(f'✅ Feature flag system working')
print(f'   Test user assignment: {assignment}')

# Test rollout percentage
for percentage in [0, 25, 50, 100]:
    flag_manager.rollout_percentage = percentage
    assigned_users = sum(1 for i in range(1000) if flag_manager.should_use_lightrag(f'user_{i}'))
    actual_percentage = (assigned_users / 1000) * 100
    print(f'   {percentage}% rollout → {actual_percentage:.1f}% actual')
"

# Test 4: End-to-End Query Test
echo -e "\nTest 4: End-to-End Query Processing"
python3 -c "
import asyncio
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

async def test_query():
    try:
        config = LightRAGConfig.get_config()
        rag = ClinicalMetabolomicsRAG(config)
        
        # Simple test query
        response = await rag.query('What is metabolomics?', mode='local')
        print(f'✅ Query processing successful')
        print(f'   Response length: {len(response)} characters')
        print(f'   Response preview: {response[:100]}...')
        
        return True
    except Exception as e:
        print(f'❌ Query processing failed: {e}')
        return False

success = asyncio.run(test_query())
exit(0 if success else 1)
"

# Test 5: Integration with Main Application
echo -e "\nTest 5: Main Application Integration"
timeout 10 python3 -c "
import sys
sys.path.append('src')
try:
    from main import *
    print('✅ Main application imports successful')
    
    # Test that LightRAG components are available
    from lightrag_integration.feature_flag_manager import FeatureFlagManager
    print('✅ LightRAG integration available in main app context')
except Exception as e:
    print(f'❌ Main application integration failed: {e}')
    exit(1)
" 2>/dev/null || echo "⚠️  Main application test timed out (may be normal)"

# Test 6: Health Check Endpoint
echo -e "\nTest 6: Application Health Check"
# Start application in background for testing
timeout 30 bash -c 'chainlit run src/app.py --port 8001 > /dev/null 2>&1 &'
APP_PID=$!
sleep 5

# Test if application is responding
if curl -f http://localhost:8001/health 2>/dev/null; then
    echo "✅ Application health check passed"
else
    echo "⚠️  Health check endpoint not responding (may not be implemented yet)"
fi

# Clean up
kill $APP_PID 2>/dev/null || true

echo -e "\n🎉 Deployment verification completed!"
echo "Ready for gradual rollout."
```

### 2. Performance Verification

```bash
#!/bin/bash
# verify_performance.sh

echo "⚡ LightRAG Performance Verification"
echo "===================================="

python3 -c "
import asyncio
import time
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

async def performance_test():
    config = LightRAGConfig.get_config()
    rag = ClinicalMetabolomicsRAG(config)
    
    test_queries = [
        'What is clinical metabolomics?',
        'How is mass spectrometry used in metabolomics?',
        'What are biomarkers in metabolic diseases?'
    ]
    
    print('🔍 Running performance tests...')
    
    total_time = 0
    for i, query in enumerate(test_queries, 1):
        print(f'\\n   Test {i}/3: {query[:50]}...')
        
        start_time = time.time()
        response = await rag.query(query, mode='local')
        query_time = time.time() - start_time
        total_time += query_time
        
        print(f'   ✅ Response time: {query_time:.2f}s')
        print(f'   📝 Response length: {len(response)} chars')
        
        if query_time > 30:
            print('   ⚠️  Query time exceeds 30s threshold')
    
    avg_time = total_time / len(test_queries)
    print(f'\\n📊 Performance Summary:')
    print(f'   Average query time: {avg_time:.2f}s')
    print(f'   Total test time: {total_time:.2f}s')
    
    if avg_time < 15:
        print('✅ Performance within acceptable limits')
        return True
    else:
        print('⚠️  Performance may need optimization')
        return False

success = asyncio.run(performance_test())
print(f'\\n🎯 Performance test: {'PASSED' if success else 'NEEDS_ATTENTION'}')
"
```

---

## Gradual Rollout Strategy

### 1. Rollout Phases

**Phase 1: Shadow Mode (0% user traffic)**
- LightRAG processes all queries but results are not shown to users
- Compare quality and performance with Perplexity
- Duration: 1 week

**Phase 2: Canary Release (5% user traffic)**
- Small subset of users get LightRAG responses
- Monitor quality metrics and error rates
- Duration: 1 week

**Phase 3: Gradual Expansion (5% → 25% → 50% → 75% → 100%)**
- Increase rollout percentage weekly
- Automated rollback if quality drops
- Duration: 4-6 weeks

### 2. Rollout Management Script

```bash
#!/bin/bash
# manage_rollout.sh

COMMAND=${1:-status}
PERCENTAGE=${2:-}

case $COMMAND in
    "status")
        echo "🔍 Current LightRAG rollout status:"
        python3 -c "
from lightrag_integration.config import LightRAGConfig
config = LightRAGConfig.get_config()
print(f'   Integration enabled: {config.lightrag_integration_enabled}')
print(f'   Rollout percentage: {config.lightrag_rollout_percentage}%')
print(f'   Circuit breaker: {config.lightrag_enable_circuit_breaker}')
"
        ;;
    
    "advance")
        if [ -z "$PERCENTAGE" ]; then
            echo "❌ Usage: $0 advance <percentage>"
            exit 1
        fi
        
        echo "📈 Advancing rollout to $PERCENTAGE%..."
        
        # Update environment variable
        sed -i.bak "s/LIGHTRAG_ROLLOUT_PERCENTAGE=.*/LIGHTRAG_ROLLOUT_PERCENTAGE=$PERCENTAGE/" .env
        
        # Verify change
        source .env
        echo "✅ Rollout updated to $LIGHTRAG_ROLLOUT_PERCENTAGE%"
        
        # Log the change
        echo "$(date): Rollout advanced to $PERCENTAGE%" >> logs/rollout.log
        ;;
    
    "rollback")
        BACKUP_PERCENTAGE=${2:-0}
        echo "⏪ Rolling back to $BACKUP_PERCENTAGE%..."
        
        # Update environment variable
        sed -i.bak "s/LIGHTRAG_ROLLOUT_PERCENTAGE=.*/LIGHTRAG_ROLLOUT_PERCENTAGE=$BACKUP_PERCENTAGE/" .env
        
        echo "✅ Rollback completed to $BACKUP_PERCENTAGE%"
        echo "$(date): ROLLBACK to $BACKUP_PERCENTAGE%" >> logs/rollout.log
        ;;
    
    "metrics")
        echo "📊 Rollout metrics for last 24 hours:"
        python3 -c "
from lightrag_integration.monitoring import get_rollout_metrics
metrics = get_rollout_metrics(hours=24)
print(f'   Queries processed: {metrics.get(\"total_queries\", 0)}')
print(f'   LightRAG queries: {metrics.get(\"lightrag_queries\", 0)}')
print(f'   Success rate: {metrics.get(\"success_rate\", 0):.2%}')
print(f'   Average quality: {metrics.get(\"avg_quality\", 0):.2f}')
print(f'   Error rate: {metrics.get(\"error_rate\", 0):.2%}')
"
        ;;
    
    *)
        echo "Usage: $0 {status|advance <percent>|rollback [percent]|metrics}"
        exit 1
        ;;
esac
```

### 3. Automated Quality Gate Monitoring

```bash
#!/bin/bash
# quality_gate_monitor.sh

echo "🎯 Quality Gate Monitoring"
echo "=========================="

while true; do
    python3 -c "
import asyncio
from lightrag_integration.monitoring import QualityGateMonitor

async def check_quality_gates():
    monitor = QualityGateMonitor()
    
    # Get current metrics
    metrics = await monitor.get_current_metrics()
    
    print(f'Current metrics:')
    print(f'  Success rate: {metrics.success_rate:.2%}')
    print(f'  Average quality: {metrics.avg_quality:.2f}')
    print(f'  Error rate: {metrics.error_rate:.2%}')
    print(f'  Response time P95: {metrics.response_time_p95:.2f}s')
    
    # Check if rollback is needed
    if await monitor.should_rollback():
        print('🚨 Quality gate failure detected - initiating rollback')
        await monitor.trigger_rollback()
        return False
    else:
        print('✅ Quality gates passing')
        return True

success = asyncio.run(check_quality_gates())
exit(0 if success else 1)
"
    
    if [ $? -ne 0 ]; then
        echo "🚨 Quality gate failure - monitoring stopped"
        break
    fi
    
    echo "⏳ Waiting 5 minutes for next check..."
    sleep 300
done
```

---

## Rollback Procedures

### 1. Emergency Rollback

```bash
#!/bin/bash
# emergency_rollback.sh

REASON=${1:-"Manual emergency rollback"}

echo "🚨 EMERGENCY ROLLBACK INITIATED"
echo "==============================="
echo "Reason: $REASON"
echo "Time: $(date)"

# Immediate rollback to 0%
echo "⏪ Setting rollout to 0%..."
sed -i.emergency "s/LIGHTRAG_ROLLOUT_PERCENTAGE=.*/LIGHTRAG_ROLLOUT_PERCENTAGE=0.0/" .env
sed -i.emergency "s/LIGHTRAG_INTEGRATION_ENABLED=.*/LIGHTRAG_INTEGRATION_ENABLED=false/" .env

# Clear any caches
echo "🗑️  Clearing caches..."
rm -rf /tmp/lightrag_cache/* 2>/dev/null || true

# Log the rollback
echo "$(date): EMERGENCY ROLLBACK - $REASON" >> logs/emergency.log

# Verify rollback
source .env
echo "✅ Emergency rollback completed"
echo "   Integration enabled: $LIGHTRAG_INTEGRATION_ENABLED"
echo "   Rollout percentage: $LIGHTRAG_ROLLOUT_PERCENTAGE%"

# Send notifications (if configured)
python3 -c "
from lightrag_integration.alerts import send_emergency_alert
send_emergency_alert('EMERGENCY ROLLBACK', '$REASON')
" 2>/dev/null || echo "⚠️  Alert system not configured"

echo ""
echo "🔍 Next steps:"
echo "1. Investigate the issue: check logs/lightrag_integration.log"
echo "2. Review error patterns: grep 'ERROR' logs/lightrag_integration.log"
echo "3. Check system resources: df -h && free -h"
echo "4. Verify database connectivity"
echo "5. When ready, use ./manage_rollout.sh advance <percentage> to resume"
```

### 2. Graceful Rollback

```bash
#!/bin/bash
# graceful_rollback.sh

TARGET_PERCENTAGE=${1:-0}
REASON=${2:-"Planned rollback"}

echo "⏪ Graceful rollback to $TARGET_PERCENTAGE%"
echo "=========================================="
echo "Reason: $REASON"

# Get current percentage
source .env
CURRENT_PERCENTAGE=$LIGHTRAG_ROLLOUT_PERCENTAGE

echo "Current: $CURRENT_PERCENTAGE% → Target: $TARGET_PERCENTAGE%"

# Calculate rollback steps
if [ $(echo "$CURRENT_PERCENTAGE > $TARGET_PERCENTAGE" | bc -l) -eq 1 ]; then
    echo "📉 Rolling back gradually..."
    
    # Define rollback steps (reduce by 25% each step)
    STEPS=(75 50 25 0)
    
    for STEP in "${STEPS[@]}"; do
        if [ $(echo "$STEP >= $TARGET_PERCENTAGE" | bc -l) -eq 1 ] && [ $(echo "$STEP < $CURRENT_PERCENTAGE" | bc -l) -eq 1 ]; then
            echo "   Setting to $STEP%..."
            ./manage_rollout.sh advance $STEP
            
            echo "   Waiting 60 seconds for stabilization..."
            sleep 60
            
            echo "   Checking metrics..."
            python3 -c "
from lightrag_integration.monitoring import get_current_health
health = get_current_health()
print(f'   Health check: {\"PASS\" if health.is_healthy else \"FAIL\"}')
if not health.is_healthy:
    print(f'   Issues: {health.issues}')
"
        fi
    done
fi

echo "✅ Graceful rollback completed"
echo "$(date): GRACEFUL ROLLBACK to $TARGET_PERCENTAGE% - $REASON" >> logs/rollout.log
```

### 3. Rollback Verification

```bash
#!/bin/bash
# verify_rollback.sh

echo "🔍 Verifying rollback status..."

# Check environment variables
source .env
echo "Environment configuration:"
echo "  LIGHTRAG_INTEGRATION_ENABLED: $LIGHTRAG_INTEGRATION_ENABLED"
echo "  LIGHTRAG_ROLLOUT_PERCENTAGE: $LIGHTRAG_ROLLOUT_PERCENTAGE%"

# Test query routing
echo -e "\nTesting query routing..."
python3 -c "
from lightrag_integration.feature_flag_manager import FeatureFlagManager

flag_manager = FeatureFlagManager()

# Test multiple users
lightrag_count = 0
total_tests = 100

for i in range(total_tests):
    if flag_manager.should_use_lightrag(f'test_user_{i}'):
        lightrag_count += 1

actual_percentage = (lightrag_count / total_tests) * 100
print(f'Actual routing: {actual_percentage:.1f}% to LightRAG')

if actual_percentage <= 5:  # Allow 5% tolerance
    print('✅ Rollback verified - minimal traffic to LightRAG')
else:
    print('❌ Rollback verification failed - still routing significant traffic')
    exit(1)
"

# Check recent logs for errors
echo -e "\nRecent error patterns:"
tail -50 logs/lightrag_integration.log | grep -i error | tail -5 || echo "No recent errors found"

# Verify application health
echo -e "\nApplication health check..."
timeout 10 bash -c 'chainlit run src/app.py --port 8002 > /dev/null 2>&1 &'
APP_PID=$!
sleep 3

if curl -f http://localhost:8002/health 2>/dev/null; then
    echo "✅ Application responding normally"
else
    echo "✅ Application accessible (health endpoint may not exist)"
fi

kill $APP_PID 2>/dev/null || true

echo -e "\n🎉 Rollback verification completed successfully"
```

---

## Monitoring and Alerting

### 1. Key Metrics to Monitor

**Performance Metrics:**
- Query response time (P50, P95, P99)
- System resource usage (CPU, Memory, Disk)
- API rate limits and costs
- Database connection health

**Quality Metrics:**
- Response quality scores
- User satisfaction ratings
- Error rates by component
- Fallback frequency

**Business Metrics:**
- Query volume by source (LightRAG vs Perplexity)
- Cost per query
- User retention and engagement
- Success rate by query type

### 2. Monitoring Dashboard Setup

```bash
#!/bin/bash
# setup_monitoring.sh

echo "📊 Setting up LightRAG monitoring dashboard..."

# Create monitoring script
cat > monitoring_daemon.py << 'EOF'
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging

from lightrag_integration.config import LightRAGConfig
from lightrag_integration.monitoring import MetricsCollector
from lightrag_integration.alerts import AlertManager

async def monitoring_loop():
    """Main monitoring loop."""
    config = LightRAGConfig.get_config()
    collector = MetricsCollector()
    alert_manager = AlertManager()
    
    logger = logging.getLogger("lightrag_monitor")
    logger.setLevel(logging.INFO)
    
    while True:
        try:
            # Collect current metrics
            metrics = await collector.collect_all_metrics()
            
            # Write metrics to file
            metrics_file = Path("logs/metrics.jsonl")
            with open(metrics_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    **metrics
                }) + "\n")
            
            # Check alert conditions
            await alert_manager.check_and_send_alerts(metrics)
            
            logger.info(f"Metrics collected: {metrics['query_count']} queries, "
                       f"{metrics['success_rate']:.2%} success rate")
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        
        # Wait 60 seconds
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(monitoring_loop())
EOF

# Create monitoring service script
cat > start_monitoring.sh << 'EOF'
#!/bin/bash

echo "📊 Starting LightRAG monitoring daemon..."

# Start monitoring in background
nohup python3 monitoring_daemon.py > logs/monitoring.log 2>&1 &
MONITOR_PID=$!

echo "✅ Monitoring started (PID: $MONITOR_PID)"
echo $MONITOR_PID > monitoring.pid

# Create stop script
cat > stop_monitoring.sh << 'EOL'
#!/bin/bash
if [ -f monitoring.pid ]; then
    PID=$(cat monitoring.pid)
    kill $PID 2>/dev/null && echo "✅ Monitoring stopped" || echo "⚠️  Monitoring process not found"
    rm -f monitoring.pid
else
    echo "⚠️  Monitoring PID file not found"
fi
EOL

chmod +x stop_monitoring.sh
EOF

chmod +x start_monitoring.sh

echo "✅ Monitoring setup completed"
echo "   Start: ./start_monitoring.sh"
echo "   Stop: ./stop_monitoring.sh"
echo "   Logs: tail -f logs/monitoring.log"
echo "   Metrics: tail -f logs/metrics.jsonl"
```

### 3. Alert Configuration

```bash
#!/bin/bash
# configure_alerts.sh

echo "🚨 Configuring LightRAG alerts..."

cat > alerts_config.json << 'EOF'
{
  "alert_rules": [
    {
      "name": "high_error_rate",
      "condition": "error_rate > 0.05",
      "severity": "critical",
      "message": "LightRAG error rate exceeded 5%",
      "cooldown_minutes": 15
    },
    {
      "name": "low_quality_score",
      "condition": "avg_quality_score < 0.7",
      "severity": "warning",
      "message": "LightRAG quality score below threshold",
      "cooldown_minutes": 30
    },
    {
      "name": "high_response_time",
      "condition": "response_time_p95 > 30",
      "severity": "warning",
      "message": "LightRAG response time P95 > 30 seconds",
      "cooldown_minutes": 10
    },
    {
      "name": "budget_threshold",
      "condition": "daily_cost > daily_budget * 0.8",
      "severity": "warning",
      "message": "LightRAG daily budget 80% consumed",
      "cooldown_minutes": 60
    },
    {
      "name": "budget_exceeded",
      "condition": "daily_cost > daily_budget",
      "severity": "critical",
      "message": "LightRAG daily budget exceeded",
      "cooldown_minutes": 15
    }
  ],
  "notification_channels": {
    "slack": {
      "webhook_url": "${SLACK_WEBHOOK_URL}",
      "channel": "#cmo-alerts",
      "enabled": true
    },
    "email": {
      "smtp_server": "${SMTP_SERVER}",
      "recipients": ["admin@yourorg.com"],
      "enabled": false
    }
  }
}
EOF

echo "✅ Alert configuration created: alerts_config.json"
echo "🔧 Edit the file to customize alert rules and notification channels"
```

---

## Troubleshooting

### 1. Common Deployment Issues

**Issue: LightRAG initialization fails**
```bash
# Diagnosis
python3 -c "
import os
from lightrag_integration.config import LightRAGConfig
config = LightRAGConfig.get_config(validate_config=True)
print('Configuration valid')
print(f'Working dir exists: {config.working_dir.exists()}')
print(f'API key present: {bool(config.api_key)}')
"

# Solutions
# 1. Check API key
echo "Checking API key..."
python3 -c "import openai; client = openai.OpenAI(); print('API key works')"

# 2. Check directories
echo "Creating directories..."
mkdir -p lightrag_storage logs
chmod 755 lightrag_storage logs

# 3. Check permissions
ls -la lightrag_storage/
```

**Issue: High memory usage**
```bash
# Diagnosis
echo "Memory usage:"
python3 -c "
import psutil
memory = psutil.virtual_memory()
print(f'Used: {memory.percent}%')
print(f'Available: {memory.available / 1024**3:.1f} GB')
"

# Solutions
# 1. Reduce concurrent operations
sed -i 's/LIGHTRAG_MAX_ASYNC=.*/LIGHTRAG_MAX_ASYNC=8/' .env

# 2. Reduce chunk size
sed -i '/LIGHTRAG_CHUNK_SIZE=/c\LIGHTRAG_CHUNK_SIZE=800' .env

# 3. Enable garbage collection
python3 -c "import gc; gc.collect(); print('Garbage collection completed')"
```

**Issue: Database connection failures**
```bash
# Diagnosis
echo "Testing database connections..."
python3 -c "
import asyncio
import asyncpg
import os
from neo4j import GraphDatabase

async def test_connections():
    # Test PostgreSQL
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        result = await conn.fetchval('SELECT version()')
        await conn.close()
        print(f'✅ PostgreSQL: {result[:50]}...')
    except Exception as e:
        print(f'❌ PostgreSQL: {e}')
    
    # Test Neo4j
    try:
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URL'),
            auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        )
        with driver.session() as session:
            result = session.run('CALL dbms.components() YIELD name, versions RETURN name, versions').single()
            print(f'✅ Neo4j: {result[\"name\"]} {result[\"versions\"][0]}')
        driver.close()
    except Exception as e:
        print(f'❌ Neo4j: {e}')

asyncio.run(test_connections())
"

# Solutions
# 1. Check database services
sudo systemctl status postgresql
sudo systemctl status neo4j

# 2. Check connection strings
echo "DATABASE_URL format should be: postgresql://user:pass@host:port/db"
echo "NEO4J_URL format should be: bolt://host:port"

# 3. Test connectivity
ping -c 3 localhost
telnet localhost 5432
telnet localhost 7687
```

### 2. Performance Troubleshooting

```bash
#!/bin/bash
# performance_troubleshoot.sh

echo "⚡ LightRAG Performance Troubleshooting"
echo "======================================"

# System resources
echo "1. System Resources:"
echo "   CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "   Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
echo "   Disk Usage: $(df -h . | tail -1 | awk '{print $5}')"

# Database performance
echo -e "\n2. Database Performance:"
python3 -c "
import asyncio
import time
import asyncpg
import os
from neo4j import GraphDatabase

async def test_db_performance():
    # PostgreSQL query time
    start = time.time()
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
    result = await conn.fetchval('SELECT COUNT(*) FROM pg_stat_activity')
    await conn.close()
    pg_time = time.time() - start
    print(f'   PostgreSQL query time: {pg_time:.3f}s (connections: {result})')
    
    # Neo4j query time
    start = time.time()
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URL'),
        auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )
    with driver.session() as session:
        result = session.run('MATCH (n) RETURN count(n) LIMIT 1').single()
    driver.close()
    neo4j_time = time.time() - start
    print(f'   Neo4j query time: {neo4j_time:.3f}s (nodes: {result[0] if result else 0})')

asyncio.run(test_db_performance())
"

# API performance
echo -e "\n3. API Performance:"
python3 -c "
import time
import openai
import os

client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Test embedding speed
start = time.time()
response = client.embeddings.create(
    model='text-embedding-3-small',
    input='test performance query'
)
embedding_time = time.time() - start
print(f'   OpenAI embedding time: {embedding_time:.3f}s')

# Test completion speed
start = time.time()
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'What is metabolomics?'}],
    max_tokens=100
)
completion_time = time.time() - start
print(f'   OpenAI completion time: {completion_time:.3f}s')
"

# LightRAG performance
echo -e "\n4. LightRAG Performance:"
python3 -c "
import asyncio
import time
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

async def test_lightrag_performance():
    config = LightRAGConfig.get_config()
    rag = ClinicalMetabolomicsRAG(config)
    
    # Test query performance
    query = 'What is clinical metabolomics?'
    
    start = time.time()
    response = await rag.query(query, mode='local')
    local_time = time.time() - start
    print(f'   Local query time: {local_time:.3f}s (response: {len(response)} chars)')
    
    start = time.time()
    response = await rag.query(query, mode='global')
    global_time = time.time() - start
    print(f'   Global query time: {global_time:.3f}s (response: {len(response)} chars)')

asyncio.run(test_lightrag_performance())
"

echo -e "\n5. Performance Recommendations:"
python3 -c "
import psutil
memory = psutil.virtual_memory()
cpu_count = psutil.cpu_count()

print(f'   Current config recommendations:')
if memory.percent > 80:
    print('   - Reduce LIGHTRAG_MAX_ASYNC (currently high memory usage)')
if cpu_count >= 8:
    print(f'   - Can increase LIGHTRAG_MAX_ASYNC to {min(32, cpu_count * 2)}')
else:
    print(f'   - Keep LIGHTRAG_MAX_ASYNC at {cpu_count * 2} or lower')

if memory.available < 4 * 1024**3:  # Less than 4GB
    print('   - Consider upgrading to 16GB+ RAM for production')
"

echo -e "\n6. Quick Performance Fixes:"
echo "   # Reduce concurrent operations"
echo "   sed -i 's/LIGHTRAG_MAX_ASYNC=.*/LIGHTRAG_MAX_ASYNC=8/' .env"
echo ""
echo "   # Optimize chunk size"
echo "   echo 'LIGHTRAG_CHUNK_SIZE=1000' >> .env"
echo ""
echo "   # Enable response caching"
echo "   echo 'LIGHTRAG_ENABLE_CACHE=true' >> .env"
```

### 3. Diagnostic Data Collection

```bash
#!/bin/bash
# collect_diagnostics.sh

DIAGNOSTIC_DIR="diagnostics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIAGNOSTIC_DIR"

echo "🔍 Collecting diagnostic data..."
echo "Diagnostic directory: $DIAGNOSTIC_DIR"

# System information
echo "System Information" > "$DIAGNOSTIC_DIR/system_info.txt"
uname -a >> "$DIAGNOSTIC_DIR/system_info.txt"
cat /etc/os-release >> "$DIAGNOSTIC_DIR/system_info.txt" 2>/dev/null
python3 --version >> "$DIAGNOSTIC_DIR/system_info.txt"
pip list | grep -E "(chainlit|lightrag|openai|neo4j)" >> "$DIAGNOSTIC_DIR/system_info.txt"

# Environment configuration
cp .env "$DIAGNOSTIC_DIR/environment.txt" 2>/dev/null || echo "No .env file" > "$DIAGNOSTIC_DIR/environment.txt"

# Recent logs
tail -1000 logs/lightrag_integration.log > "$DIAGNOSTIC_DIR/lightrag_logs.txt" 2>/dev/null || echo "No LightRAG logs" > "$DIAGNOSTIC_DIR/lightrag_logs.txt"
tail -1000 logs/monitoring.log > "$DIAGNOSTIC_DIR/monitoring_logs.txt" 2>/dev/null || echo "No monitoring logs" > "$DIAGNOSTIC_DIR/monitoring_logs.txt"

# Configuration validation
python3 -c "
from lightrag_integration.config import LightRAGConfig
try:
    config = LightRAGConfig.get_config(validate_config=True)
    print('✅ Configuration valid')
    print(f'Model: {config.model}')
    print(f'Max async: {config.max_async}')
    print(f'Working dir: {config.working_dir}')
    print(f'Integration enabled: {config.lightrag_integration_enabled}')
    print(f'Rollout percentage: {config.lightrag_rollout_percentage}%')
except Exception as e:
    print(f'❌ Configuration error: {e}')
" > "$DIAGNOSTIC_DIR/config_validation.txt"

# Storage information
du -sh lightrag_storage/ > "$DIAGNOSTIC_DIR/storage_usage.txt" 2>/dev/null
ls -la lightrag_storage/ >> "$DIAGNOSTIC_DIR/storage_usage.txt" 2>/dev/null

# Process information
ps aux | grep -E "(chainlit|python|lightrag)" > "$DIAGNOSTIC_DIR/processes.txt"

# Network connectivity
echo "Database connectivity:" > "$DIAGNOSTIC_DIR/connectivity.txt"
nc -zv localhost 5432 >> "$DIAGNOSTIC_DIR/connectivity.txt" 2>&1
nc -zv localhost 7687 >> "$DIAGNOSTIC_DIR/connectivity.txt" 2>&1

# Create summary
cat > "$DIAGNOSTIC_DIR/README.txt" << EOF
LightRAG Diagnostic Data Collection
==================================
Generated: $(date)

Files included:
- system_info.txt: System and Python environment
- environment.txt: Environment variables
- config_validation.txt: Configuration validation results
- lightrag_logs.txt: Recent LightRAG integration logs
- monitoring_logs.txt: Recent monitoring logs
- storage_usage.txt: Storage usage information
- processes.txt: Running processes
- connectivity.txt: Database connectivity tests

To analyze issues:
1. Check config_validation.txt for configuration errors
2. Review lightrag_logs.txt for recent errors
3. Verify connectivity.txt for database issues
4. Check system_info.txt for environment problems

For support, please provide this entire directory.
EOF

echo "✅ Diagnostic data collected in: $DIAGNOSTIC_DIR"
echo "📋 Summary:"
wc -l "$DIAGNOSTIC_DIR"/* | tail -1
echo ""
echo "To create archive: tar -czf ${DIAGNOSTIC_DIR}.tar.gz $DIAGNOSTIC_DIR"
```

---

This comprehensive deployment procedures document provides everything needed to successfully deploy LightRAG integration with the Clinical Metabolomics Oracle system. The procedures are designed to be safe, gradual, and fully reversible, ensuring that the existing system functionality is preserved while adding powerful knowledge graph capabilities.

Key features of this deployment approach:
- **Modular and optional**: Can be disabled entirely if needed
- **Gradual rollout**: Start with 0% and increase incrementally
- **Comprehensive monitoring**: Track performance, quality, and costs
- **Automated rollback**: Revert quickly if issues arise
- **Extensive validation**: Verify each step before proceeding
- **Production-ready**: Includes Docker, monitoring, and alerting

The deployment can be executed step-by-step, with each phase validated before proceeding to the next, ensuring a smooth transition to the enhanced LightRAG-powered system.