# Clinical Metabolomics Oracle - LightRAG Integration Plan

## Executive Summary

This document outlines a comprehensive two-phase implementation plan for integrating LightRAG into the Clinical Metabolomics Oracle (CMO) system. The plan prioritizes a modular approach that preserves existing functionality while adding advanced knowledge graph capabilities for biomedical research.

## Current System Overview

**Existing Architecture:**
- **Frontend**: Chainlit-based chat interface with FastAPI backend
- **Knowledge Base**: Neo4j graph database with specialized biomedical queries
- **Response Generation**: Perplexity API for real-time responses
- **Features**: Multi-language support, citation processing, confidence scoring
- **Data Sources**: PubMed, PubChem, HMDB, KEGG, and other biomedical databases

**Key Challenge**: Current system bypasses traditional RAG pipeline and relies heavily on Perplexity API, requiring careful integration to preserve specialized biomedical features.

---

## Phase 1: MVP (Minimum Viable Product)

**Timeline**: 6-8 weeks  
**Goal**: Create a standalone LightRAG component that can be tested and validated independently

### 1.1 Environment Setup and Dependencies

**Week 1: Infrastructure Setup**

```bash
# Create LightRAG environment
python -m venv lightrag_env
source lightrag_env/bin/activate  # On Windows: lightrag_env\Scripts\activate

# Install dependencies
pip install lightrag-hku
pip install PyMuPDF  # For PDF processing
pip install python-dotenv
pip install asyncio
pip install pytest  # For testing
```

**Directory Structure:**
```
smo_chatbot_August_6th_2025/
├── lightrag_integration/
│   ├── __init__.py
│   ├── lightrag_component.py
│   ├── pdf_processor.py
│   ├── config.py
│   └── tests/
├── papers/                    # PDF knowledge base
├── docs/
│   └── plan.md               # This file
└── requirements_lightrag.txt
```

### 1.2 PDF Processing Module

**Week 1-2: PDF Ingestion System**

Create `lightrag_integration/pdf_processor.py`:

```python
import PyMuPDF
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple
import logging

class BiomedicalPDFProcessor:
    """Specialized PDF processor for biomedical papers"""
    
    def __init__(self, papers_dir: str = "papers/"):
        self.papers_dir = Path(papers_dir)
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract text and metadata from biomedical PDF"""
        doc = PyMuPDF.open(pdf_path)
        text = ""
        metadata = {
            "filename": pdf_path.name,
            "pages": len(doc),
            "source": "local_pdf"
        }
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            # Clean and preprocess text for biomedical content
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        doc.close()
        return text, metadata
    
    async def process_all_pdfs(self) -> List[Tuple[str, Dict]]:
        """Process all PDFs in the papers directory"""
        documents = []
        
        if not self.papers_dir.exists():
            self.logger.warning(f"Papers directory {self.papers_dir} does not exist")
            return documents
        
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                text, metadata = self.extract_text_from_pdf(pdf_file)
                documents.append((text, metadata))
                self.logger.info(f"Processed: {pdf_file.name}")
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file.name}: {e}")
        
        return documents
```

### 1.3 LightRAG Component Module

**Week 2-3: Core LightRAG Integration**

Create `lightrag_integration/lightrag_component.py`:

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embedding
from lightrag.utils import EmbeddingFunc
from .pdf_processor import BiomedicalPDFProcessor
from .config import LightRAGConfig
import logging

class ClinicalMetabolomicsRAG:
    """LightRAG component specialized for clinical metabolomics"""
    
    def __init__(self, config: LightRAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pdf_processor = BiomedicalPDFProcessor(config.papers_dir)
        self.rag = None
        self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize LightRAG with biomedical-specific configuration"""
        self.rag = LightRAG(
            working_dir=self.config.working_dir,
            llm_model_func=self._get_llm_function(),
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                func=self._get_embedding_function()
            ),
            chunk_token_size=1200,  # Optimized for biomedical papers
            chunk_overlap_token_size=100,
            entity_extract_max_gleaning=2,  # More thorough for scientific content
        )
    
    def _get_llm_function(self):
        """Get LLM function based on configuration"""
        async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await gpt_4o_mini_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.config.openai_api_key,
                **kwargs
            )
        return llm_func
    
    def _get_embedding_function(self):
        """Get embedding function for biomedical content"""
        async def embed_func(texts):
            return await openai_embedding(
                texts,
                model="text-embedding-3-small",
                api_key=self.config.openai_api_key
            )
        return embed_func
    
    async def initialize_knowledge_base(self):
        """Initialize the knowledge base from PDF files"""
        self.logger.info("Initializing LightRAG storages...")
        await self.rag.initialize_storages()
        
        self.logger.info("Processing PDF files...")
        documents = await self.pdf_processor.process_all_pdfs()
        
        if not documents:
            self.logger.warning("No documents found to process")
            return
        
        # Extract text content for LightRAG
        text_documents = [doc[0] for doc in documents]
        
        self.logger.info(f"Inserting {len(text_documents)} documents into LightRAG...")
        await self.rag.ainsert(text_documents)
        
        self.logger.info("Knowledge base initialization complete")
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        """Query the LightRAG system"""
        if not self.rag:
            raise RuntimeError("LightRAG not initialized")
        
        try:
            response = await self.rag.aquery(
                question,
                param=QueryParam(
                    mode=mode,
                    response_type="Multiple Paragraphs",
                    top_k=10,
                    max_total_tokens=8000
                )
            )
            return response
        except Exception as e:
            self.logger.error(f"Query error: {e}")
            raise
    
    async def get_context_only(self, question: str) -> str:
        """Get only the context without generating a response"""
        response = await self.rag.aquery(
            question,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=10
            )
        )
        return response
```

### 1.4 Configuration Module

**Week 2: Configuration Management**

Create `lightrag_integration/config.py`:

```python
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LightRAGConfig:
    """Configuration for LightRAG integration"""
    
    # Directories
    working_dir: str = "./lightrag_storage"
    papers_dir: str = "papers/"
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # LightRAG Settings
    chunk_size: int = 1200
    chunk_overlap: int = 100
    max_tokens: int = 8000
    
    # Testing
    test_question: str = "What is clinical metabolomics?"
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Create directories if they don't exist
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        Path(self.papers_dir).mkdir(parents=True, exist_ok=True)

def get_config() -> LightRAGConfig:
    """Get configuration instance"""
    return LightRAGConfig()
```

### 1.5 Testing Framework

**Week 3-4: Testing and Validation**

Create `lightrag_integration/tests/test_mvp.py`:

```python
import pytest
import asyncio
from pathlib import Path
from ..lightrag_component import ClinicalMetabolomicsRAG
from ..config import get_config

class TestLightRAGMVP:
    """Test suite for LightRAG MVP"""
    
    @pytest.fixture
    async def rag_system(self):
        """Initialize RAG system for testing"""
        config = get_config()
        rag = ClinicalMetabolomicsRAG(config)
        await rag.initialize_knowledge_base()
        return rag
    
    @pytest.mark.asyncio
    async def test_initialization(self, rag_system):
        """Test that the system initializes correctly"""
        assert rag_system.rag is not None
        assert Path(rag_system.config.working_dir).exists()
    
    @pytest.mark.asyncio
    async def test_clinical_metabolomics_query(self, rag_system):
        """Test the primary success criterion"""
        question = "What is clinical metabolomics?"
        response = await rag_system.query(question)
        
        # Validation criteria
        assert len(response) > 100  # Substantial response
        assert "metabolomics" in response.lower()
        assert any(term in response.lower() for term in [
            "clinical", "biomarker", "metabolism", "disease", "diagnostic"
        ])
    
    @pytest.mark.asyncio
    async def test_context_retrieval(self, rag_system):
        """Test context-only retrieval"""
        question = "What is clinical metabolomics?"
        context = await rag_system.get_context_only(question)
        
        assert len(context) > 50
        assert "metabolomics" in context.lower()
    
    def test_pdf_processing(self):
        """Test PDF processing functionality"""
        from ..pdf_processor import BiomedicalPDFProcessor
        
        processor = BiomedicalPDFProcessor("papers/")
        # Test will pass if papers directory exists and contains PDFs
        assert processor.papers_dir.exists()
```

### 1.6 MVP Success Metrics

**Week 4: Validation Criteria**

**Primary Success Criterion:**
- System must accurately answer "What is clinical metabolomics?" using only information from ingested PDFs

**Technical Validation:**
- [ ] PDF files successfully processed and ingested
- [ ] Knowledge graph constructed with biomedical entities
- [ ] Query response contains relevant metabolomics information
- [ ] Response time under 30 seconds for standard queries
- [ ] System handles at least 10 PDF files without errors

**Quality Metrics:**
- Response relevance score > 80% (manual evaluation)
- Factual accuracy verified against source papers
- No hallucinated information not present in source documents

### 1.7 Integration Preparation

**Week 5-6: Modular Integration Setup**

Create `lightrag_integration/__init__.py`:

```python
"""
LightRAG Integration Module for Clinical Metabolomics Oracle

This module provides a standalone LightRAG component that can be
integrated into the existing CMO system.
"""

from .lightrag_component import ClinicalMetabolomicsRAG
from .config import LightRAGConfig, get_config
from .pdf_processor import BiomedicalPDFProcessor

__all__ = [
    'ClinicalMetabolomicsRAG',
    'LightRAGConfig', 
    'get_config',
    'BiomedicalPDFProcessor'
]

# Version info
__version__ = "1.0.0-mvp"
```

**Integration Example for Existing System:**

```python
# In existing main.py, add LightRAG as optional component
from lightrag_integration import ClinicalMetabolomicsRAG, get_config

# Global variable for LightRAG (optional)
lightrag_system = None

async def initialize_lightrag():
    """Initialize LightRAG system if enabled"""
    global lightrag_system
    if os.getenv("ENABLE_LIGHTRAG", "false").lower() == "true":
        config = get_config()
        lightrag_system = ClinicalMetabolomicsRAG(config)
        await lightrag_system.initialize_knowledge_base()

# In message handler, add LightRAG option
@cl.on_message
async def on_message(message: cl.Message):
    # ... existing code ...
    
    # Optional: Use LightRAG for specific queries
    if lightrag_system and should_use_lightrag(content):
        lightrag_response = await lightrag_system.query(content)
        # Combine with existing citation processing
        # ... rest of existing logic ...
```

---

## Phase 1 Deliverables

**Week 6-8: Documentation and Handoff**

1. **Functional MVP System**
   - Standalone LightRAG component
   - PDF processing pipeline
   - Test suite with passing tests
   - Configuration management

2. **Documentation**
   - API documentation for all modules
   - Setup and installation guide
   - Testing procedures
   - Integration examples

3. **Validation Report**
   - Performance benchmarks
   - Quality assessment results
   - Comparison with existing system responses
   - Recommendations for Phase 2

**Phase 1 Resource Requirements:**
- **Development Time**: 6-8 weeks (1 developer)
- **Infrastructure**: OpenAI API access, local development environment
- **Testing Data**: 10-20 clinical metabolomics PDF papers
- **Budget**: ~$200-500 for API costs during development and testing

---

## Phase 2: Long-term Solution

**Timeline**: 12-16 weeks  
**Goal**: Full integration with intelligent routing and production deployment

### 2.1 Intelligent Query Routing System

**Week 1-3: LLM-Based Router Implementation**

The routing system will analyze incoming queries and determine the optimal response strategy:

```python
class IntelligentQueryRouter:
    """Routes queries between LightRAG and Perplexity based on context"""
    
    ROUTING_CATEGORIES = {
        "knowledge_graph": [
            "relationships", "connections", "pathways", "mechanisms",
            "biomarkers", "metabolites", "diseases", "clinical studies"
        ],
        "real_time": [
            "latest", "recent", "current", "new", "breaking",
            "today", "this year", "2024", "2025"
        ],
        "general": [
            "what is", "define", "explain", "overview", "introduction"
        ]
    }
    
    async def route_query(self, query: str, conversation_history: list) -> str:
        """Determine optimal routing strategy"""
        # Use LLM to classify query intent
        classification_prompt = f"""
        Analyze this query and determine the best response strategy:
        Query: "{query}"
        
        Categories:
        1. KNOWLEDGE_GRAPH: Complex relationships, biomedical connections, established knowledge
        2. REAL_TIME: Current events, latest research, breaking news
        3. HYBRID: Combination of established knowledge and current information
        
        Respond with: KNOWLEDGE_GRAPH, REAL_TIME, or HYBRID
        """
        
        # Implementation details...
        return routing_decision
```

### 2.2 Enhanced Architecture Integration

**Week 4-8: Robust System Integration**

**Error Handling and Fallback Mechanisms:**
- Primary: LightRAG knowledge graph query
- Fallback 1: Perplexity API with LightRAG context
- Fallback 2: Pure Perplexity API query
- Emergency: Cached response or error message

**Performance Optimization:**
- Async query processing
- Response caching
- Connection pooling
- Load balancing between services

### 2.3 Multi-Language and Citation Integration

**Week 9-12: Feature Integration**

**Translation System Integration:**
```python
async def process_multilingual_query(query: str, language: str) -> str:
    """Process query with full translation support"""
    
    # 1. Translate query to English if needed
    english_query = await translate_if_needed(query, language)
    
    # 2. Route and process query
    routing_decision = await router.route_query(english_query)
    
    if routing_decision == "KNOWLEDGE_GRAPH":
        response = await lightrag_system.query(english_query)
    else:
        response = await perplexity_query(english_query)
    
    # 3. Process citations and confidence scores
    response_with_citations = await process_citations(response)
    
    # 4. Translate response back if needed
    final_response = await translate_if_needed(response_with_citations, "en", language)
    
    return final_response
```

### 2.4 Scalability and Maintenance

**Week 13-16: Production Readiness**

**Scalability Considerations:**
- Horizontal scaling with multiple LightRAG instances
- Database sharding for large document collections
- CDN integration for static assets
- Monitoring and alerting systems

**Maintenance Procedures:**
- Automated PDF ingestion pipeline
- Incremental knowledge base updates
- Performance monitoring and optimization
- Regular system health checks

---

## Technical Requirements

### Hardware Requirements
- **Development**: 16GB RAM, 4-core CPU, 100GB storage
- **Production**: 32GB RAM, 8-core CPU, 500GB SSD, GPU optional

### Software Dependencies
- Python 3.9+
- OpenAI API access
- Neo4j (existing)
- PostgreSQL (existing)
- Docker (for deployment)

### API Rate Limits and Costs
- OpenAI API: ~$50-200/month for moderate usage
- Perplexity API: Existing costs
- Infrastructure: ~$100-300/month for cloud deployment

## Risk Assessment and Mitigation

### High-Risk Items
1. **Integration Complexity**: Mitigate with modular design and extensive testing
2. **Performance Impact**: Mitigate with caching and async processing
3. **Data Quality**: Mitigate with validation pipelines and manual review

### Medium-Risk Items
1. **API Cost Overruns**: Mitigate with usage monitoring and rate limiting
2. **User Adoption**: Mitigate with gradual rollout and user training

## Success Metrics

### Phase 1 Success Criteria
- [ ] MVP system answers "What is clinical metabolomics?" accurately
- [ ] PDF processing pipeline handles 10+ documents
- [ ] Response time < 30 seconds
- [ ] Integration module ready for Phase 2

### Phase 2 Success Criteria
- [ ] Intelligent routing achieves 90%+ accuracy
- [ ] System handles 100+ concurrent users
- [ ] Response quality maintained or improved
- [ ] Full feature parity with existing system
- [ ] Production deployment successful

This comprehensive plan provides a structured approach to integrating LightRAG while preserving the specialized biomedical capabilities of the Clinical Metabolomics Oracle system.
