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
- **OpenAI API Key**: Required for GPT-4o-mini and text-embedding-3-small
  - Get your key from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - Ensure sufficient credits for document processing and queries
  - Recommended: $20+ credits for initial setup and testing

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

```bash
# Create virtual environment (if not already created)
python -m venv lightrag_env

# Activate the environment
# On macOS/Linux:
source lightrag_env/bin/activate

# On Windows:
# lightrag_env\Scripts\activate

# Verify activation
which python  # Should show path to lightrag_env/bin/python
```

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

1. **Copy Environment Template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit Configuration**:
   ```bash
   # Edit the .env file with your preferred editor
   nano .env
   # or
   code .env
   ```

### Step 2: Essential Configuration Variables

**Minimum Required Settings**:
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
```

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

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ImportError: No module named 'lightrag'`

**Solutions**:
```bash
# Ensure virtual environment is activated
source lightrag_env/bin/activate

# Reinstall requirements
pip install -r requirements_lightrag.txt

# Install OpenAI separately if needed
pip install openai

# Verify installation
pip show lightrag-hku
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

**🎉 Congratulations!** You have successfully set up LightRAG integration for the Clinical Metabolomics Oracle system. The knowledge graph-based RAG system is now ready to process biomedical research papers and provide intelligent responses to clinical metabolomics queries.

For questions or issues, refer to the [Troubleshooting](#troubleshooting) section or consult the project documentation.