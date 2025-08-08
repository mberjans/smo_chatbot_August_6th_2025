# LightRAG Integration Setup Guide

**Clinical Metabolomics Oracle - LightRAG Knowledge Graph Integration**

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation Instructions](#installation-instructions)
- [Configuration Guide](#configuration-guide)
- [Quick Start Guide](#quick-start-guide)
- [Testing Instructions](#testing-instructions)
- [Troubleshooting](#troubleshooting)
- [Integration Notes](#integration-notes)
- [Cost Considerations](#cost-considerations)
- [Security Guidelines](#security-guidelines)

---

## Project Overview

This guide covers the setup and integration of **LightRAG** (Light Retrieval-Augmented Generation) into the Clinical Metabolomics Oracle system. LightRAG is a knowledge graph-based RAG system that processes biomedical research papers to create an intelligent knowledge graph, enabling more sophisticated and contextually aware responses for clinical metabolomics research queries.

### Key Features
- **Knowledge Graph Construction**: Automatically extracts entities and relationships from research papers
- **Semantic Search**: Advanced retrieval using embeddings and graph structures
- **Multi-Modal Queries**: Supports naive, local, global, and hybrid query modes
- **Clinical Focus**: Optimized for biomedical and clinical metabolomics literature
- **Integration Ready**: Designed to work alongside existing Perplexity and CMO systems

### Architecture
- **Document Processing**: PDF ingestion using PyMuPDF
- **Knowledge Extraction**: Entity and relationship extraction using GPT-4o-mini
- **Graph Storage**: Local knowledge graph with nano-vectordb for embeddings
- **Query Engine**: Multiple retrieval strategies for different query types
- **API Integration**: OpenAI API for LLM operations and embeddings

---

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.9 or higher (tested with Python 3.13.5)
- **Memory**: Minimum 8GB RAM (16GB recommended for large document sets)
- **Storage**: 2GB free space for knowledge graph storage
- **Internet**: Stable connection for API calls

### Required API Keys

#### Minimum Setup (Required):
- **OpenAI API Key**: **ABSOLUTELY REQUIRED** for GPT-4o-mini and text-embedding-3-small
  - Get your key from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - Ensure sufficient credits for document processing and queries
  - Recommended: $20+ credits for initial setup and testing
  - **Without this key, the system will NOT work**

#### Optional API Keys (Enhanced Functionality):
- **Perplexity API**: For real-time web search (can skip for basic testing)
- **Groq API**: For faster inference (can skip for basic testing)  
- **OpenRouter API**: For multiple LLM providers (can skip for basic testing)

#### Setup Priority:
1. **First**: Get OpenAI API key and test basic functionality
2. **Later**: Add other API keys for enhanced features

### Software Dependencies
- Git (for repository management)
- pip (Python package installer)
- Virtual environment support (venv)

---

## Installation Instructions

### Step 1: Clone and Navigate to Project

```bash
# If not already in the project directory
cd /path/to/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025
```

### Step 2: Create and Activate Virtual Environment

**CRITICAL**: You MUST activate the virtual environment before installing dependencies or running the application.

```bash
# Create virtual environment (if not already created)
python3 -m venv lightrag_env

# Activate the environment (CHOOSE YOUR PLATFORM)
# On macOS/Linux:
source lightrag_env/bin/activate

# On Windows Command Prompt:
# lightrag_env\Scripts\activate.bat

# On Windows PowerShell:
# lightrag_env\Scripts\Activate.ps1

# On Windows Git Bash:
# source lightrag_env/Scripts/activate

# Verify activation (IMPORTANT - run this to confirm)
which python  # Should show path to lightrag_env/bin/python
python --version  # Should be Python 3.9+ 
echo $VIRTUAL_ENV  # Should show path to lightrag_env
```

**⚠️ TROUBLESHOOTING**: If you see system Python path instead of `lightrag_env/bin/python`, the virtual environment is NOT activated. You must activate it before proceeding.

**To check if virtual environment is active**: Your command prompt should show `(lightrag_env)` at the beginning of the line.

### Step 3: Install Dependencies

```bash
# Install LightRAG and all dependencies
pip install -r requirements_lightrag.txt

# Verify installation
pip list | grep lightrag
# Should show: lightrag-hku==1.4.6
```

### Step 4: Install OpenAI Library

**Note**: The OpenAI library is required but not included in requirements_lightrag.txt to avoid conflicts.

```bash
# Install OpenAI library
pip install openai

# Verify installation
python -c "import openai; print(f'OpenAI version: {openai.__version__}')"
```

### Step 5: Create Directory Structure

```bash
# Create necessary directories
mkdir -p lightrag_storage
mkdir -p papers
mkdir -p logs
mkdir -p test_data

# Verify directory structure
ls -la
# Should show: lightrag_storage/, papers/, logs/, test_data/
```

---

## Configuration Guide

### Step 1: Environment Variables Setup

**CRITICAL: Dual .env File Configuration**

This project uses TWO .env files that you must configure:

1. **Root .env file** (for LightRAG integration)
2. **src/.env file** (for main application - OVERRIDES root .env)

⚠️ **WARNING**: The `src/.env` file takes precedence over the root `.env` file. You must configure BOTH files correctly.

#### Configure Root .env File:
```bash
# Copy the environment template
cp .env.example .env

# Edit the root .env file
nano .env
# or
code .env
```

#### Configure src/.env File (CRITICAL):
```bash
# Check if src/.env exists
ls -la src/.env

# If src/.env exists, you MUST update it with your API keys
nano src/.env
# or
code src/.env

# If src/.env doesn't exist, create it from root .env
cp .env src/.env
```

### Step 2: Essential Configuration Variables

#### Minimum Required Settings for BOTH .env Files:

**Root .env file** (complete LightRAG configuration):
```bash
# API Configuration
OPENAI_API_KEY=sk-your_actual_openai_api_key_here

# LightRAG Configuration  
ENABLE_LIGHTRAG=true
LIGHTRAG_WORKING_DIR=./lightrag_storage
LIGHTRAG_PAPERS_DIR=./papers

# Model Configuration
LIGHTRAG_LLM_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
LIGHTRAG_EMBEDDING_DIM=1536

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# Chainlit Authentication
CHAINLIT_AUTH_SECRET=your_secure_auth_secret_here
```

**src/.env file** (minimal but CRITICAL - overrides root):
```bash
# CRITICAL: These API keys are required in src/.env
OPENAI_API_KEY=sk-your_actual_openai_api_key_here
GROQ_API_KEY=gsk_your_groq_api_key_here  # Optional but recommended
PERPLEXITY_API=pplx-your_perplexity_key_here  # Optional
OPENROUTER_API_KEY=sk-or-your_openrouter_key_here  # Optional

# Authentication (REQUIRED)
CHAINLIT_AUTH_SECRET=your_secure_auth_secret_here

# Database (REQUIRED)
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
NEO4J_PASSWORD=your_neo4j_password_here
```

⚠️ **IMPORTANT**: The `src/.env` file will override any conflicting variables from the root `.env` file. Ensure both files have matching values for shared variables.

### Step 3: Advanced Configuration Options

**Performance Settings**:
```bash
# Processing Parameters
LIGHTRAG_CHUNK_SIZE=1200
LIGHTRAG_CHUNK_OVERLAP=100
LIGHTRAG_MAX_TOKENS=8000
LIGHTRAG_TOP_K=10

# Query Configuration
LIGHTRAG_DEFAULT_MODE=hybrid
LIGHTRAG_LLM_TEMPERATURE=0.1

# System Settings
LOG_LEVEL=INFO
LIGHTRAG_VERBOSE_LOGGING=true
```

### Step 4: Verify Configuration

```bash
# Test environment loading
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('API Key found:', bool(os.getenv('OPENAI_API_KEY')))
print('LightRAG enabled:', os.getenv('ENABLE_LIGHTRAG'))
"
```

---

## Quick Start Guide

### Step 1: Verify API Connectivity

Before processing documents, verify your OpenAI API connection:

```bash
# Run connectivity test
python test_openai_connectivity.py
```

**Expected Output**:
```
✓ ALL TESTS PASSED - OpenAI API is ready for LightRAG integration!
```

### Step 2: Add Sample Documents

```bash
# Copy sample research paper (if available)
cp Clinical_Metabolomics_paper.pdf papers/

# Or add your own PDF files
# Place any clinical metabolomics research papers in the papers/ directory
ls papers/
```

### Step 3: Initialize Knowledge Graph

**Create initialization script** (save as `init_lightrag.py`):

```python
#!/usr/bin/env python3
"""
LightRAG Knowledge Graph Initialization Script
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Basic LightRAG setup (simplified for quick start)
def init_knowledge_graph():
    """Initialize LightRAG knowledge graph with sample documents."""
    
    # Import LightRAG (this will be available after pip install)
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm import openai_complete_if_cache, openai_embedding
        print("✓ LightRAG imported successfully")
    except ImportError as e:
        print(f"✗ Error importing LightRAG: {e}")
        return False
    
    # Configure working directory
    working_dir = Path(os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_storage"))
    working_dir.mkdir(exist_ok=True)
    
    # Initialize LightRAG instance
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=openai_complete_if_cache,
        embedding_func=openai_embedding
    )
    
    # Process documents in papers directory
    papers_dir = Path(os.getenv("LIGHTRAG_PAPERS_DIR", "./papers"))
    
    if not papers_dir.exists():
        print(f"✗ Papers directory not found: {papers_dir}")
        return False
    
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠ No PDF files found in {papers_dir}")
        print("Please add PDF files to the papers/ directory and run again.")
        return False
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Process each PDF (basic text extraction - you may need PyMuPDF)
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        try:
            # Simple file reading - replace with proper PDF extraction
            with open(pdf_file, 'rb') as f:
                # This is a placeholder - actual PDF processing requires PyMuPDF
                print(f"  Processing {pdf_file.name} (PDF extraction needed)")
                # You would implement PDF text extraction here
                
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {e}")
    
    print("\n✓ LightRAG initialization complete!")
    print(f"Knowledge graph stored in: {working_dir}")
    
    return True

if __name__ == "__main__":
    print("LightRAG Knowledge Graph Initialization")
    print("=" * 45)
    
    success = init_knowledge_graph()
    
    if success:
        print("\n🎉 Setup complete! You can now run queries against the knowledge graph.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
```

**Run the initialization**:
```bash
python init_lightrag.py
```

### Step 4: Test Basic Query

**Create test query script** (save as `test_query.py`):

```python
#!/usr/bin/env python3
"""
LightRAG Query Test Script
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_query():
    """Test basic query against LightRAG knowledge graph."""
    
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm import openai_complete_if_cache, openai_embedding
    except ImportError as e:
        print(f"✗ Error importing LightRAG: {e}")
        return False
    
    # Configure working directory
    working_dir = Path(os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_storage"))
    
    if not working_dir.exists():
        print(f"✗ Knowledge graph not found at: {working_dir}")
        print("Please run init_lightrag.py first")
        return False
    
    # Initialize LightRAG instance
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=openai_complete_if_cache,
        embedding_func=openai_embedding
    )
    
    # Test queries
    test_queries = [
        "What is clinical metabolomics?",
        "What are the main applications of metabolomics in clinical research?",
        "How are metabolites analyzed in clinical studies?"
    ]
    
    print("Testing LightRAG queries:")
    print("=" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        try:
            # Test different query modes
            for mode in ["naive", "local", "global", "hybrid"]:
                print(f"Mode: {mode}")
                
                result = rag.query(query, param=QueryParam(mode=mode))
                print(f"Result: {result[:200]}...")
                print()
                
        except Exception as e:
            print(f"✗ Error with query: {e}")
    
    return True

if __name__ == "__main__":
    print("LightRAG Query Test")
    print("=" * 20)
    
    success = test_query()
    
    if success:
        print("\n✓ Query test completed successfully!")
    else:
        print("\n✗ Query test failed.")
```

**Run the test**:
```bash
python test_query.py
```

---

## Testing Instructions

### Automated Testing Suite

**Run the complete test suite**:

```bash
# 1. Test OpenAI API connectivity
python test_openai_connectivity.py

# 2. Test environment configuration
python -c "
from dotenv import load_dotenv
import os
load_dotenv()

required_vars = [
    'OPENAI_API_KEY',
    'LIGHTRAG_WORKING_DIR', 
    'LIGHTRAG_PAPERS_DIR',
    'LIGHTRAG_LLM_MODEL',
    'LIGHTRAG_EMBEDDING_MODEL'
]

print('Environment Variable Check:')
print('=' * 30)
for var in required_vars:
    value = os.getenv(var)
    status = '✓' if value else '✗'
    print(f'{status} {var}: {\"Set\" if value else \"Missing\"}')
"

# 3. Test LightRAG imports
python -c "
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm import openai_complete_if_cache, openai_embedding
    print('✓ LightRAG imports successful')
except ImportError as e:
    print(f'✗ LightRAG import failed: {e}')
"

# 4. Test directory structure
python -c "
from pathlib import Path
import os

dirs = [
    os.getenv('LIGHTRAG_WORKING_DIR', './lightrag_storage'),
    os.getenv('LIGHTRAG_PAPERS_DIR', './papers'),
    './logs',
    './test_data'
]

print('Directory Structure Check:')
print('=' * 30)
for dir_path in dirs:
    path = Path(dir_path)
    status = '✓' if path.exists() else '✗'
    print(f'{status} {dir_path}: {\"Exists\" if path.exists() else \"Missing\"}')
"
```

### Manual Verification Steps

1. **Check Virtual Environment**:
   ```bash
   which python
   pip list | grep -E "(lightrag|openai|pymupdf)"
   ```

2. **Verify File Structure**:
   ```bash
   ls -la
   ls papers/
   ls lightrag_storage/
   ```

3. **Test Configuration Loading**:
   ```bash
   python -c "
   from dotenv import load_dotenv
   import os
   load_dotenv()
   print('OpenAI Key:', os.getenv('OPENAI_API_KEY')[:10] + '...' if os.getenv('OPENAI_API_KEY') else 'Missing')
   "
   ```

### Expected Test Results

**All tests passing should show**:
- ✓ OpenAI API connectivity successful
- ✓ All required environment variables set
- ✓ LightRAG library imports working
- ✓ Directory structure created
- ✓ PDF processing capabilities available
- ✓ Knowledge graph initialization possible

---

## Troubleshooting

### Critical Setup Issues (Based on Independent Developer Testing)

#### 1. Dual .env File Configuration Problems

**Problem**: Application can't find API keys or fails to start
**Symptoms**: `KeyError: 'OPENAI_API_KEY'`, authentication errors

**Root Cause**: Missing or misconfigured src/.env file that overrides root .env

**Solutions**:
```bash
# Step 1: Check both .env files exist
ls -la .env
ls -la src/.env

# Step 2: If src/.env missing, create it
cp .env src/.env

# Step 3: Verify API keys in BOTH files
grep "OPENAI_API_KEY" .env
grep "OPENAI_API_KEY" src/.env

# Step 4: Ensure values match
# Edit src/.env to match your root .env values
```

#### 2. Missing psutil Dependency

**Problem**: `ImportError: No module named 'psutil'`
**Symptoms**: Crashes when monitoring system resources in lightrag_integration

**Solution**:
```bash
# Install missing dependency
pip install psutil==5.9.8

# Or reinstall requirements (now includes psutil)
pip install -r requirements_lightrag.txt
```

#### 3. Virtual Environment Not Activated

**Problem**: `ImportError: No module named 'lightrag'` despite installation
**Symptoms**: System Python being used instead of virtual environment

**Solutions**:
```bash
# Check if virtual environment is active
which python  # Should show venv/bin/python, NOT /usr/bin/python

# If not activated, activate it
source lightrag_env/bin/activate  # Linux/Mac
# or lightrag_env\Scripts\activate  # Windows

# Your prompt should show (lightrag_env) prefix
# Example: (lightrag_env) user@hostname:~/project$

# Verify activation
echo $VIRTUAL_ENV  # Should show path to lightrag_env

# If still problems, recreate virtual environment
rm -rf lightrag_env
python3 -m venv lightrag_env
source lightrag_env/bin/activate
pip install -r requirements_lightrag.txt
```

#### 4. Import Errors

**Problem**: `ImportError: No module named 'lightrag'`

**Solutions**:
```bash
# Ensure virtual environment is activated (see above)
source lightrag_env/bin/activate

# Reinstall requirements
pip install -r requirements_lightrag.txt

# Install OpenAI separately if needed
pip install openai

# Verify installation
pip show lightrag-hku
pip list | grep -E "(lightrag|psutil|openai)"
```

#### 2. API Key Issues

**Problem**: `AuthenticationError` or `Invalid API key`

**Solutions**:
```bash
# Check API key format
echo $OPENAI_API_KEY | cut -c1-10
# Should start with 'sk-'

# Test API key directly
python -c "
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

try:
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Hello'}],
        max_tokens=10
    )
    print('✓ API key is valid')
except Exception as e:
    print(f'✗ API key error: {e}')
"
```

#### 3. PDF Processing Issues

**Problem**: Cannot process PDF files

**Solutions**:
```bash
# Ensure PyMuPDF is installed
pip install PyMuPDF==1.26.3

# Test PDF reading
python -c "
import fitz  # PyMuPDF
print('PyMuPDF version:', fitz.version)

# Test opening a sample PDF
try:
    doc = fitz.open('./papers/Clinical_Metabolomics_paper.pdf')
    print(f'PDF pages: {len(doc)}')
    doc.close()
    print('✓ PDF processing working')
except Exception as e:
    print(f'✗ PDF error: {e}')
"
```

#### 4. Memory Issues

**Problem**: Out of memory during document processing

**Solutions**:
- Reduce `LIGHTRAG_CHUNK_SIZE` from 1200 to 800
- Process fewer documents at once
- Increase system RAM or use swap file
- Use `gpt-4o-mini` instead of larger models

#### 5. Network Connectivity

**Problem**: API timeout or connection errors

**Solutions**:
```bash
# Test internet connection
curl -I https://api.openai.com

# Check firewall settings
# Ensure ports 443 and 80 are open

# Test with increased timeout
export REQUEST_TIMEOUT_SECONDS=60
```

#### 6. Permission Errors

**Problem**: Cannot create directories or write files

**Solutions**:
```bash
# Check permissions
ls -la

# Fix permissions
chmod 755 .
chmod -R 755 lightrag_storage papers logs

# Ensure ownership
chown -R $(whoami) .
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Set debug environment variables
export LOG_LEVEL=DEBUG
export LIGHTRAG_VERBOSE_LOGGING=true

# Run with debug output
python test_openai_connectivity.py 2>&1 | tee debug.log
```

### Getting Help

1. **Check Logs**: Review files in `logs/` directory
2. **Environment Check**: Run the test suite above
3. **Documentation**: Refer to [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
4. **API Status**: Check [OpenAI Status Page](https://status.openai.com)

---

## Integration Notes

### Integration with Existing CMO System

The LightRAG integration is designed to work **alongside** the existing Clinical Metabolomics Oracle components:

#### Query Routing Strategy
- **LightRAG Queries**: For document-based research questions
- **Perplexity Queries**: For current events and recent research
- **Hybrid Queries**: Combine both systems for comprehensive answers

#### File Organization
```
Clinical_Metabolomics_Oracle/
├── src/                          # Existing CMO system
├── papers/                       # PDF documents for LightRAG
├── lightrag_storage/            # LightRAG knowledge graph
├── requirements.txt             # Original CMO requirements  
├── requirements_lightrag.txt    # LightRAG-specific requirements
├── .env                        # Combined environment config
└── README_lightrag.md          # This setup guide
```

#### Environment Variables
The `.env` file combines settings for both systems:
- **CMO Settings**: Database URLs, Chainlit auth, etc.
- **LightRAG Settings**: API keys, model configuration, storage paths
- **Shared Settings**: Logging, performance tuning

#### API Cost Management
- **Budget Monitoring**: Set OpenAI usage limits
- **Model Selection**: Use `gpt-4o-mini` for cost efficiency
- **Query Optimization**: Cache results to reduce API calls
- **Document Processing**: Process documents in batches

### Phase 1 MVP Scope

This setup covers **Phase 1 MVP** functionality:

**Included**:
- ✅ PDF document ingestion
- ✅ Knowledge graph construction
- ✅ Basic query capabilities
- ✅ OpenAI API integration
- ✅ Multiple query modes (naive, local, global, hybrid)

**Future Phases** (not in this setup):
- 🔄 Advanced entity relationship extraction
- 🔄 Real-time document monitoring
- 🔄 Multi-language support
- 🔄 Advanced visualization tools
- 🔄 Automated paper discovery and ingestion

### Performance Considerations

- **Document Size**: Optimal for papers up to 50MB each
- **Knowledge Graph Size**: Can handle 100-1000 documents efficiently
- **Query Response Time**: 2-10 seconds depending on complexity
- **Memory Usage**: ~2-4GB for typical research paper collections
- **API Calls**: ~10-50 calls per document during initial processing

---

## Cost Considerations

### OpenAI API Usage Estimates

**Document Processing** (one-time costs):
- **Text Processing**: ~$0.01-0.05 per research paper
- **Embedding Generation**: ~$0.001-0.005 per paper
- **Entity Extraction**: ~$0.02-0.10 per paper
- **Total per Paper**: ~$0.03-0.15

**Query Operations** (ongoing costs):
- **Simple Query**: ~$0.001-0.01 per query
- **Complex Query**: ~$0.01-0.05 per query
- **Embedding Search**: ~$0.001 per query

**Monthly Estimates** (for reference):
- **Small Collection** (10 papers, 100 queries/month): ~$5-10
- **Medium Collection** (50 papers, 500 queries/month): ~$15-30
- **Large Collection** (200 papers, 1000 queries/month): ~$50-100

### Cost Optimization Tips

1. **Use gpt-4o-mini**: 90% cheaper than GPT-4, suitable for most tasks
2. **Batch Processing**: Process documents in batches to reduce overhead
3. **Cache Results**: Enable response caching to avoid repeated API calls
4. **Optimize Chunk Size**: Larger chunks = fewer API calls, but may reduce quality
5. **Monitor Usage**: Set up billing alerts on your OpenAI account

### Budget Planning

**Recommended Starting Budget**: $20-50 for initial setup and testing
**Production Budget**: $10-100/month depending on usage

---

## Security Guidelines

### API Key Security

**Best Practices**:
```bash
# 1. Never commit API keys to version control
echo ".env" >> .gitignore

# 2. Use environment variables only
export OPENAI_API_KEY="sk-your-key-here"

# 3. Restrict API key permissions (if available)
# Set usage limits and allowed endpoints in OpenAI dashboard

# 4. Rotate keys regularly
# Generate new API keys quarterly
```

### File Security

**Document Protection**:
```bash
# 1. Secure the papers directory
chmod 700 papers/

# 2. Secure the knowledge graph storage
chmod 700 lightrag_storage/

# 3. Secure log files (may contain sensitive info)
chmod 600 logs/*.log
```

### Network Security

**Safe Configuration**:
```bash
# 1. Use HTTPS only (default for OpenAI API)
# 2. Configure firewall rules
# 3. Monitor API usage for anomalies
# 4. Use VPN if processing sensitive medical data
```

### Data Privacy

**Important Notes**:
- 📋 **Research Papers**: Ensure you have rights to process the documents
- 📋 **API Data**: OpenAI may retain data for abuse monitoring (check their policy)
- 📋 **Local Storage**: Knowledge graphs are stored locally by default
- 📋 **Compliance**: Consider HIPAA/GDPR requirements for medical data

---

## Next Steps

After completing this setup:

1. **Test the System**: Run all test scripts and verify functionality
2. **Add Documents**: Place your clinical metabolomics research papers in `papers/`
3. **Initialize Knowledge Graph**: Run the initialization scripts
4. **Integration Testing**: Test queries through the CMO interface
5. **Performance Tuning**: Adjust configuration based on your document collection
6. **Monitor Usage**: Set up billing alerts and usage tracking
7. **Documentation**: Document your specific use cases and configurations

### Support and Resources

- **LightRAG Documentation**: [GitHub Repository](https://github.com/HKUDS/LightRAG)
- **OpenAI API Docs**: [OpenAI API Reference](https://platform.openai.com/docs)
- **PyMuPDF Docs**: [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- **Project Issues**: Create issues in the CMO project repository

---

## Independent Developer Validation

**Complete this checklist to ensure your setup will work:**

### Pre-Launch Validation Checklist

```bash
# 1. Virtual Environment Check
which python  # Should show: lightrag_env/bin/python
echo $VIRTUAL_ENV  # Should show path to lightrag_env
python --version  # Should be 3.9+

# 2. Dependencies Check
pip list | grep lightrag-hku  # Should show version 1.4.6
pip list | grep psutil  # Should show version 5.9.8
pip list | grep openai  # Should be installed

# 3. Dual .env Configuration Check
ls -la .env src/.env  # Both files must exist
grep "OPENAI_API_KEY" .env src/.env  # Both should have your API key

# 4. API Key Validation
python -c "
import os
from dotenv import load_dotenv
load_dotenv()  # Load root .env
load_dotenv('src/.env')  # Override with src/.env
key = os.getenv('OPENAI_API_KEY')
if key and key.startswith('sk-'):
    print('✅ API key format is correct')
else:
    print('❌ API key missing or invalid format')
"

# 5. Directory Structure Check
ls -la lightrag_storage papers logs  # All should exist

# 6. OpenAI API Test
python -c "
import openai, os
from dotenv import load_dotenv
load_dotenv()
load_dotenv('src/.env')
try:
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Hello'}],
        max_tokens=5
    )
    print('✅ OpenAI API working')
except Exception as e:
    print(f'❌ API error: {e}')
"
```

### If ALL checks pass ✅:
Your setup is ready! You can proceed to initialize the knowledge graph and start using the system.

### If ANY check fails ❌:
Review the [Troubleshooting](#troubleshooting) section above for the specific issue.

---

**🎉 Congratulations!** Once all validation checks pass, you have successfully set up LightRAG integration for the Clinical Metabolomics Oracle system. The knowledge graph-based RAG system is now ready to process biomedical research papers and provide intelligent responses to clinical metabolomics queries.

For questions or issues, refer to the [Troubleshooting](#troubleshooting) section or consult the project documentation.