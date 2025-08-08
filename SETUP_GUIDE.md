# Clinical Metabolomics Oracle - LightRAG Integration
## Complete Setup and Installation Guide

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [System Requirements](#system-requirements)
4. [Installation Steps](#installation-steps)
5. [Database Setup](#database-setup)
6. [Environment Configuration](#environment-configuration)
7. [API Keys Setup](#api-keys-setup)
8. [Verification and Testing](#verification-and-testing)
9. [Troubleshooting](#troubleshooting)
10. [Production Deployment](#production-deployment)
11. [Advanced Configuration](#advanced-configuration)
12. [Maintenance](#maintenance)

---

## Quick Start

For experienced users who want to get started immediately:

```bash
# 1. Clone and navigate
cd Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements_lightrag.txt

# 4. Setup environment
cp .env.example .env
# Edit .env with your API keys and configuration

# 5. Start PostgreSQL and Neo4j databases

# 6. Run database migration
npx prisma migrate deploy

# 7. Test installation
python -c "from lightrag_integration.config import LightRAGConfig; print('✅ Setup complete!')"

# 8. Start the application
chainlit run src/app.py
```

---

## Prerequisites

### Required Software

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.10+ | Core runtime environment |
| **pip** | 21.0+ | Package management |
| **PostgreSQL** | 13+ | Primary database for Chainlit |
| **Neo4j** | 4.4+ | Graph database for knowledge graphs |
| **Node.js** | 16+ | For Prisma database management |
| **npm/yarn** | Latest | Package management for JS dependencies |

### API Access

| Service | Required | Purpose |
|---------|----------|---------|
| **OpenAI** | ✅ Required | LLM operations and embeddings |
| **Perplexity** | Optional | Real-time web search |
| **Groq** | Optional | Fast inference alternative |
| **OpenRouter** | Optional | Multiple LLM provider access |

### System Resources

**Minimum Requirements:**
- RAM: 8GB (16GB recommended)
- Storage: 10GB free space (50GB+ for large knowledge bases)
- CPU: 4 cores (8+ cores for production)
- Network: Stable internet connection for API calls

**Recommended for Production:**
- RAM: 32GB+
- Storage: 100GB+ SSD
- CPU: 16+ cores
- Network: High-bandwidth connection

---

## System Requirements

### Operating System Support

- **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+
- **macOS**: macOS 10.15+
- **Windows**: Windows 10/11 with WSL2 recommended

### Python Environment

```bash
# Check Python version
python3 --version
# Should be 3.10.0 or higher

# Check pip version
pip --version
# Should be 21.0.0 or higher
```

### Database Requirements

#### PostgreSQL Setup

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS (Homebrew):**
```bash
brew install postgresql
brew services start postgresql
```

**Windows:**
Download from [PostgreSQL Official Site](https://www.postgresql.org/download/windows/)

#### Neo4j Setup

**Ubuntu/Debian:**
```bash
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j
sudo systemctl start neo4j
```

**macOS (Homebrew):**
```bash
brew install neo4j
brew services start neo4j
```

**Docker (All Platforms):**
```bash
# PostgreSQL
docker run --name postgres-cmo -e POSTGRES_PASSWORD=yourpassword -d -p 5432:5432 postgres:13

# Neo4j
docker run --name neo4j-cmo -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/yourpassword neo4j:4.4
```

---

## Installation Steps

### 1. Environment Setup

#### Clone Repository
```bash
git clone <repository-url>
cd Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025
```

#### Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

#### Verify Environment
```bash
which python  # Should point to venv/bin/python
python --version  # Should be 3.10+
```

### 2. Dependencies Installation

#### Core Dependencies
```bash
# Install main application dependencies
pip install -r requirements.txt
```

**Main Dependencies Include:**
- `chainlit>=1.0.401` - Web UI framework
- `llama-index>=0.10.20` - Document indexing and retrieval
- `neo4j>=5.18.0` - Neo4j graph database driver
- `asyncpg>=0.30.0` - PostgreSQL async driver
- `PyMuPDF>=1.23.26` - PDF processing

#### LightRAG Dependencies
```bash
# Install LightRAG-specific dependencies
pip install -r requirements_lightrag.txt
```

**LightRAG Dependencies Include:**
- `lightrag-hku>=1.4.6` - Core LightRAG functionality
- `numpy>=2.3.2` - Numerical computing
- `pandas>=2.3.1` - Data manipulation
- `networkx>=3.5` - Graph algorithms
- `tiktoken>=0.10.0` - Token counting

#### Development Dependencies (Optional)
```bash
# For development and testing
pip install pytest pytest-asyncio coverage black flake8
```

#### Verify Installation
```bash
pip list | grep -E "(chainlit|lightrag|neo4j|asyncpg)"
```

### 3. Node.js Dependencies

#### Install Node.js Dependencies
```bash
npm install
# OR
yarn install
```

This installs Prisma for database management.

#### Verify Prisma Installation
```bash
npx prisma --version
```

---

## Database Setup

### PostgreSQL Configuration

#### 1. Create Database User
```bash
sudo -u postgres psql

# In PostgreSQL shell:
CREATE USER cmo_user WITH PASSWORD 'secure_password';
CREATE DATABASE clinical_metabolomics_oracle;
GRANT ALL PRIVILEGES ON DATABASE clinical_metabolomics_oracle TO cmo_user;
\q
```

#### 2. Test Connection
```bash
psql -h localhost -U cmo_user -d clinical_metabolomics_oracle -c "SELECT version();"
```

#### 3. Configure Connection String
```bash
# In .env file:
DATABASE_URL=postgresql://cmo_user:secure_password@localhost:5432/clinical_metabolomics_oracle
```

### Neo4j Configuration

#### 1. Set Initial Password
```bash
# Access Neo4j browser at http://localhost:7474
# Default credentials: neo4j/neo4j
# You'll be prompted to change password on first login
```

#### 2. Create Database for CMO
```cypher
// In Neo4j browser:
CREATE DATABASE clinical_metabolomics;
:use clinical_metabolomics;
```

#### 3. Configure Connection
```bash
# In .env file:
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_new_password
```

### Database Migration

#### Run Prisma Migration
```bash
npx prisma migrate deploy
```

#### Verify Database Schema
```bash
npx prisma db pull
npx prisma generate
```

---

## Environment Configuration

### 1. Create Environment File
```bash
cp .env.example .env
```

### 2. Required Environment Variables

#### Core Configuration
```bash
# Database connections
DATABASE_URL=postgresql://cmo_user:password@localhost:5432/clinical_metabolomics_oracle
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Required API key
OPENAI_API_KEY=sk-your_openai_api_key_here

# LightRAG core settings
LIGHTRAG_WORKING_DIR=./lightrag_storage
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
```

#### Application Settings
```bash
# Chainlit configuration
CHAINLIT_AUTH_SECRET=your_secure_auth_secret_here

# Application environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Enable LightRAG integration
ENABLE_LIGHTRAG=true
```

### 3. Directory Structure Creation
```bash
# Create required directories
mkdir -p lightrag_storage
mkdir -p papers
mkdir -p logs
mkdir -p backups
mkdir -p test_data
```

### 4. Security Configuration
```bash
# Set proper permissions
chmod 600 .env
chmod 755 lightrag_storage
chmod 755 logs
```

---

## API Keys Setup

### OpenAI API Key (Required)

#### 1. Get API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)

#### 2. Configure in Environment
```bash
# Add to .env
OPENAI_API_KEY=sk-your_actual_openai_key_here
```

#### 3. Test API Access
```bash
python -c "
import openai
import os
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('✅ OpenAI API access confirmed')
"
```

### Perplexity API Key (Optional)

#### 1. Get API Key
1. Visit [Perplexity API Settings](https://www.perplexity.ai/settings/api)
2. Generate API key
3. Copy key (starts with `pplx-`)

#### 2. Configure
```bash
# Add to .env
PERPLEXITY_API=pplx-your_perplexity_key_here
```

### Additional API Keys (Optional)

#### Groq API (Fast Inference)
```bash
# Get key from: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_key_here
```

#### OpenRouter API (Multiple Providers)
```bash
# Get key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-your_openrouter_key_here
```

### Generate Chainlit Auth Secret
```bash
# Generate secure secret
python -c "import secrets; print(f'CHAINLIT_AUTH_SECRET={secrets.token_urlsafe(32)}')" >> .env
```

---

## Verification and Testing

### 1. Configuration Validation

#### Test Environment Loading
```bash
python -c "
from lightrag_integration.config import LightRAGConfig
try:
    config = LightRAGConfig.get_config(validate_config=True)
    print('✅ Configuration loaded successfully')
    print(f'Model: {config.model}')
    print(f'Working Dir: {config.working_dir}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

#### Test Database Connections
```bash
# Test PostgreSQL
python -c "
import asyncpg
import os
import asyncio

async def test_pg():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        result = await conn.fetchval('SELECT version()')
        await conn.close()
        print('✅ PostgreSQL connection successful')
    except Exception as e:
        print(f'❌ PostgreSQL error: {e}')

asyncio.run(test_pg())
"

# Test Neo4j
python -c "
from neo4j import GraphDatabase
import os

try:
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URL'),
        auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )
    with driver.session() as session:
        result = session.run('RETURN 1')
        print('✅ Neo4j connection successful')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j error: {e}')
"
```

### 2. Core Functionality Tests

#### Test LightRAG Integration
```bash
python -c "
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

try:
    config = LightRAGConfig.get_config()
    cmo_rag = ClinicalMetabolomicsRAG(config)
    print('✅ LightRAG integration initialized successfully')
except Exception as e:
    print(f'❌ LightRAG integration error: {e}')
"
```

#### Run Basic Tests
```bash
# Run unit tests
pytest lightrag_integration/tests/ -v -m unit

# Run integration tests
pytest lightrag_integration/tests/ -v -m integration

# Check test coverage
pytest --cov=lightrag_integration --cov-report=html
```

### 3. Application Startup Test

#### Test Chainlit Application
```bash
# Start in test mode (background)
chainlit run src/app.py --headless &
CHAINLIT_PID=$!

# Test if server is responding
sleep 5
curl -f http://localhost:8000/healthz || echo "Health check failed"

# Stop test server
kill $CHAINLIT_PID
```

### 4. Complete System Health Check

Create and run the health check script:

```bash
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
"""Comprehensive system health check for CMO-LightRAG installation."""

import os
import sys
import asyncio
import asyncpg
from neo4j import GraphDatabase
from lightrag_integration.config import LightRAGConfig
import openai

async def check_postgresql():
    """Test PostgreSQL connection."""
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        await conn.fetchval('SELECT 1')
        await conn.close()
        return True, "PostgreSQL connection successful"
    except Exception as e:
        return False, f"PostgreSQL error: {e}"

def check_neo4j():
    """Test Neo4j connection."""
    try:
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URL'),
            auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        )
        with driver.session() as session:
            session.run('RETURN 1')
        driver.close()
        return True, "Neo4j connection successful"
    except Exception as e:
        return False, f"Neo4j error: {e}"

def check_openai():
    """Test OpenAI API access."""
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        models = client.models.list()
        return True, f"OpenAI API access successful ({len(models.data)} models available)"
    except Exception as e:
        return False, f"OpenAI API error: {e}"

def check_lightrag_config():
    """Test LightRAG configuration."""
    try:
        config = LightRAGConfig.get_config(validate_config=True)
        return True, f"LightRAG config valid (model: {config.model})"
    except Exception as e:
        return False, f"LightRAG config error: {e}"

def check_directories():
    """Check required directories exist."""
    required_dirs = ['lightrag_storage', 'papers', 'logs']
    missing = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing.append(dir_name)
    
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    return True, "All required directories exist"

async def main():
    """Run all health checks."""
    print("🔍 Clinical Metabolomics Oracle - LightRAG Health Check")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_directories),
        ("LightRAG Configuration", check_lightrag_config),
        ("OpenAI API", check_openai),
        ("PostgreSQL Database", check_postgresql),
        ("Neo4j Database", check_neo4j),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        if asyncio.iscoroutinefunction(check_func):
            passed, message = await check_func()
        else:
            passed, message = check_func()
        
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {message}")
        
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All systems operational! Your installation is ready.")
        sys.exit(0)
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

python health_check.py
```

---

## Troubleshooting

### Common Installation Issues

#### Python Version Issues
```bash
# Error: Python version too old
# Solution: Install Python 3.10+
sudo apt update
sudo apt install python3.10 python3.10-venv
python3.10 -m venv venv
```

#### Package Installation Failures
```bash
# Error: Failed building wheel for package
# Solution: Install build dependencies
sudo apt install python3-dev build-essential
# OR on macOS:
xcode-select --install
```

#### Database Connection Issues

**PostgreSQL Connection Failed:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check connection details
psql -h localhost -U postgres -c "\l"

# Reset password if needed
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'newpassword';"
```

**Neo4j Connection Failed:**
```bash
# Check Neo4j status
sudo systemctl status neo4j

# Check logs
sudo tail -f /var/log/neo4j/neo4j.log

# Reset password
sudo neo4j-admin set-initial-password newpassword
```

#### Environment Variable Issues
```bash
# Check if .env file is loaded
python -c "import os; print('API Key:', 'Found' if os.getenv('OPENAI_API_KEY') else 'Missing')"

# Check file permissions
ls -la .env
# Should show -rw------- (600)

# Fix permissions if needed
chmod 600 .env
```

#### Permission Issues
```bash
# Fix directory permissions
sudo chown -R $USER:$USER lightrag_storage logs
chmod 755 lightrag_storage logs
```

### Performance Issues

#### Slow PDF Processing
```bash
# Increase async processing limits
# In .env:
LIGHTRAG_MAX_ASYNC=32
LIGHTRAG_MAX_TOKENS=32768
```

#### High Memory Usage
```bash
# Monitor memory usage
pip install psutil
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / (1024**3):.1f} GB')
"

# Reduce memory usage in .env:
LIGHTRAG_MAX_ASYNC=8
LIGHTRAG_CHUNK_SIZE=800
```

#### API Rate Limiting
```bash
# In .env, reduce concurrency:
LIGHTRAG_MAX_ASYNC=4
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT_SECONDS=60
```

### Debug Mode

Enable comprehensive debugging:

```bash
# In .env:
LOG_LEVEL=DEBUG
LIGHTRAG_LOG_LEVEL=DEBUG
LIGHTRAG_VERBOSE_LOGGING=true
LIGHTRAG_ENABLE_FILE_LOGGING=true
ENABLE_TEST_MODE=true
```

View logs:
```bash
tail -f logs/lightrag_integration.log
tail -f logs/structured_logs.jsonl
```

---

## Production Deployment

### Security Configuration

#### 1. API Key Management
```bash
# Use environment-specific secrets management:
# AWS Secrets Manager, Azure Key Vault, etc.

# For Docker deployments:
docker run -d \
  --name cmo-lightrag \
  --env-file .env.production \
  cmo-lightrag:latest
```

#### 2. Database Security
```bash
# PostgreSQL security
# Create restricted user
CREATE USER cmo_app WITH PASSWORD 'complex_password';
GRANT CONNECT ON DATABASE clinical_metabolomics_oracle TO cmo_app;
GRANT USAGE ON SCHEMA public TO cmo_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cmo_app;

# Neo4j security
# Enable auth and use strong passwords
# Configure SSL/TLS for connections
```

#### 3. Network Security
```bash
# In .env.production:
ENABLE_CORS=true
CORS_ALLOWED_ORIGINS=https://your-domain.com
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_MINUTE=30

# Enable SSL if available:
ENABLE_SSL=true
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

### Performance Optimization

#### 1. Resource Allocation
```bash
# Production .env settings:
LIGHTRAG_MAX_ASYNC=32
LIGHTRAG_MAX_TOKENS=32768
MAX_CONCURRENT_REQUESTS=20
CACHE_EXPIRATION_SECONDS=3600
ENABLE_RESPONSE_CACHE=true
```

#### 2. Database Optimization
```bash
# PostgreSQL optimization
# In postgresql.conf:
shared_buffers = 256MB
work_mem = 4MB
maintenance_work_mem = 64MB

# Neo4j optimization  
# In neo4j.conf:
dbms.memory.heap.initial_size=512m
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G
```

#### 3. Monitoring Setup
```bash
# Enable comprehensive monitoring:
ENABLE_MONITORING=true
ENABLE_ANALYTICS=true

# Set up log rotation:
LIGHTRAG_LOG_MAX_BYTES=104857600  # 100MB
LIGHTRAG_LOG_BACKUP_COUNT=10

# Enable backups:
ENABLE_AUTO_BACKUP=true
BACKUP_FREQUENCY_HOURS=6
MAX_BACKUPS=30
```

### Docker Deployment

#### 1. Create Dockerfile
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

# Copy requirements and install Python dependencies
COPY requirements.txt requirements_lightrag.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements_lightrag.txt

# Copy package.json and install Node.js dependencies
COPY package.json package-lock.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p lightrag_storage logs papers backups

# Set permissions
RUN chmod 755 lightrag_storage logs papers backups

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run application
CMD ["chainlit", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Docker Compose Setup
```yaml
version: '3.8'

services:
  cmo-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://cmo_user:${POSTGRES_PASSWORD}@postgres:5432/clinical_metabolomics_oracle
      - NEO4J_URL=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    env_file:
      - .env.production
    volumes:
      - ./lightrag_storage:/app/lightrag_storage
      - ./logs:/app/logs
      - ./papers:/app/papers
    depends_on:
      - postgres
      - neo4j

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=clinical_metabolomics_oracle
      - POSTGRES_USER=cmo_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  neo4j:
    image: neo4j:4.4
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"

volumes:
  postgres_data:
  neo4j_data:
```

#### 3. Deploy with Docker Compose
```bash
# Create production environment file
cp .env.example .env.production
# Edit .env.production with production values

# Deploy
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f cmo-app
```

---

## Advanced Configuration

### Multi-Environment Setup

#### 1. Environment-Specific Configuration
```bash
# Development
cp .env.example .env.development
# Staging  
cp .env.example .env.staging
# Production
cp .env.example .env.production
```

#### 2. Configuration Management Script
```bash
cat > manage_config.sh << 'EOF'
#!/bin/bash

ENV=${1:-development}

if [ ! -f ".env.$ENV" ]; then
    echo "Error: .env.$ENV not found"
    exit 1
fi

# Backup current .env
if [ -f ".env" ]; then
    cp .env .env.backup
fi

# Copy environment-specific config
cp ".env.$ENV" .env
echo "Switched to $ENV environment"
EOF

chmod +x manage_config.sh

# Usage:
./manage_config.sh development
./manage_config.sh staging
./manage_config.sh production
```

### Custom Configuration Profiles

#### 1. Research-Focused Profile
```bash
# .env.research
LIGHTRAG_MODEL=gpt-4o
LIGHTRAG_MAX_TOKENS=32768
LIGHTRAG_CHUNK_SIZE=1500
LIGHTRAG_ENTITY_GLEANING=3
LIGHTRAG_MAX_ASYNC=8
```

#### 2. Clinical-Optimized Profile
```bash
# .env.clinical
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_MAX_TOKENS=16384
LIGHTRAG_MAX_ASYNC=32
ENABLE_RESPONSE_CACHE=true
CACHE_EXPIRATION_SECONDS=1800
```

#### 3. Development Profile
```bash
# .env.development
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_MAX_ASYNC=4
LOG_LEVEL=DEBUG
LIGHTRAG_LOG_LEVEL=DEBUG
ENABLE_TEST_MODE=true
```

### Load Testing Setup

#### 1. Install Load Testing Tools
```bash
pip install locust pytest-benchmark
```

#### 2. Create Load Test Script
```bash
cat > load_test.py << 'EOF'
from locust import HttpUser, task, between

class CMOUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def test_health_check(self):
        self.client.get("/healthz")
    
    @task(3)
    def test_query(self):
        self.client.post("/query", json={
            "message": "What is clinical metabolomics?",
            "mode": "hybrid"
        })
EOF
```

#### 3. Run Load Tests
```bash
locust -f load_test.py --host=http://localhost:8000
```

---

## Maintenance

### Backup Procedures

#### 1. Automated Backup Script
```bash
cat > backup.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup LightRAG storage
cp -r lightrag_storage "$BACKUP_DIR/"

# Backup PostgreSQL
pg_dump $DATABASE_URL > "$BACKUP_DIR/postgres_backup.sql"

# Backup Neo4j
neo4j-admin dump --database=clinical_metabolomics --to="$BACKUP_DIR/neo4j_backup.dump"

# Backup configuration
cp .env "$BACKUP_DIR/"

# Backup logs (last 7 days)
find logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \;

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh
```

#### 2. Schedule Regular Backups
```bash
# Add to crontab
crontab -e

# Backup daily at 2 AM
0 2 * * * /path/to/your/project/backup.sh
```

### Update Procedures

#### 1. Update Dependencies
```bash
# Update Python packages
pip list --outdated
pip install --upgrade -r requirements.txt -r requirements_lightrag.txt

# Update Node.js packages
npm audit fix
npm update
```

#### 2. Update Database Schema
```bash
# Pull latest schema changes
npx prisma db pull

# Generate new client
npx prisma generate

# Apply migrations
npx prisma migrate deploy
```

#### 3. Update LightRAG Knowledge Base
```bash
# Add new papers to papers/ directory
# Then run re-indexing
python -c "
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from lightrag_integration.config import LightRAGConfig

config = LightRAGConfig.get_config()
rag = ClinicalMetabolomicsRAG(config)
rag.rebuild_knowledge_base()
"
```

### Monitoring and Logging

#### 1. Log Analysis
```bash
# View recent errors
grep -i error logs/lightrag_integration.log | tail -20

# Monitor real-time logs
tail -f logs/lightrag_integration.log logs/structured_logs.jsonl

# Analyze performance
grep "response_time" logs/structured_logs.jsonl | jq '.response_time' | sort -n
```

#### 2. System Monitoring
```bash
# Check system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\".\").percent}%')
"

# Check database connections
netstat -an | grep -E "(5432|7687)"
```

#### 3. Health Monitoring
```bash
# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash

# Check application health
curl -s http://localhost:8000/healthz || echo "❌ Application health check failed"

# Check database connectivity
python -c "
import asyncio
import asyncpg
import os
from neo4j import GraphDatabase

async def check_databases():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        await conn.close()
        print('✅ PostgreSQL OK')
    except Exception as e:
        print(f'❌ PostgreSQL failed: {e}')
    
    try:
        driver = GraphDatabase.driver(os.getenv('NEO4J_URL'), 
                                    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
        driver.verify_connectivity()
        driver.close()
        print('✅ Neo4j OK')
    except Exception as e:
        print(f'❌ Neo4j failed: {e}')

asyncio.run(check_databases())
"
EOF

chmod +x monitor.sh
```

---

## Support and Documentation

### Additional Resources

- **[ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md)** - Complete environment variable reference
- **[LIGHTRAG_CONFIGURATION_GUIDE.md](./LIGHTRAG_CONFIGURATION_GUIDE.md)** - Detailed configuration guide
- **[docs/INTEGRATION_DOCUMENTATION.md](./docs/INTEGRATION_DOCUMENTATION.md)** - Integration patterns and examples
- **[docs/INTEGRATION_TROUBLESHOOTING_GUIDE.md](./docs/INTEGRATION_TROUBLESHOOTING_GUIDE.md)** - Common issues and solutions

### Getting Help

1. **Configuration Issues**: Check environment variables and API keys
2. **Database Problems**: Verify database connections and permissions
3. **Performance Issues**: Review resource allocation and limits
4. **API Errors**: Check API key validity and rate limits

### Contributing

When making changes to the system:

1. Test changes in development environment first
2. Update relevant documentation
3. Run full test suite before deployment
4. Create backups before major changes

---

## Version Information

- **Setup Guide Version**: 1.0
- **CMO-LightRAG Version**: Phase 1 MVP Complete
- **LightRAG Compatibility**: 1.4.6+
- **Python Compatibility**: 3.10+
- **Last Updated**: August 8, 2025
- **Task Reference**: CMO-LIGHTRAG-011-T02

---

*This guide provides comprehensive coverage for setting up the Clinical Metabolomics Oracle with LightRAG integration. For specific technical issues, refer to the troubleshooting section or consult the additional documentation files.*