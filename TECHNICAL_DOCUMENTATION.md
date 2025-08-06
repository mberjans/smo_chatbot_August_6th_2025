# Clinical Metabolomics Oracle (CMO) - Technical Documentation

## Project Overview

The Clinical Metabolomics Oracle (CMO) is an AI-powered chatbot designed to provide cited, explainable responses to scientific queries in the field of clinical metabolomics. Built on the Chainlit framework, it combines knowledge graph retrieval with large language models to deliver accurate, source-backed answers to researchers and healthcare professionals.

### Key Features
- **Multi-language Support**: Automatic language detection and translation using Google Translate and OPUS-MT
- **Knowledge Graph Integration**: Neo4j-based graph database for complex biomedical relationships
- **Citation Management**: Automatic citation generation and bibliography formatting
- **Real-time Search**: Integration with Perplexity AI for up-to-date information
- **User Authentication**: Simple password-based authentication system
- **Conversation Logging**: PostgreSQL database for storing chat history and user feedback

### Target Users
- Clinical researchers in metabolomics
- Healthcare professionals seeking metabolomics information
- Scientists working with biomedical data
- Students and educators in related fields

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chainlit UI   │────│   FastAPI App    │────│   Main Logic    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 │                                 │
                ┌──────▼──────┐                 ┌────────▼────────┐                ┌──────▼──────┐
                │ Translation │                 │ Knowledge Graph │                │ Perplexity  │
                │   System    │                 │   (Neo4j)       │                │     AI      │
                └─────────────┘                 └─────────────────┘                └─────────────┘
                       │                                 │                                 │
                ┌──────▼──────┐                 ┌────────▼────────┐                ┌──────▼──────┐
                │   Google    │                 │   Custom RAG    │                │  Citation   │
                │ Translate/  │                 │   Retriever     │                │ Processing  │
                │  OPUS-MT    │                 └─────────────────┘                └─────────────┘
                └─────────────┘
```

### Core Components

1. **Web Interface Layer** (`app.py`, `main.py`)
   - FastAPI application serving the Chainlit interface
   - User authentication and session management
   - Message handling and response formatting

2. **Knowledge Retrieval System** (`pipelines.py`, `retrievers.py`, `query_engine.py`)
   - Custom Knowledge Graph RAG retriever
   - Integration with multiple LLM providers (OpenAI, Groq, OpenRouter)
   - Citation-aware query processing

3. **Graph Database Layer** (`graph_stores.py`, `textualize.py`)
   - Custom Neo4j graph store implementation
   - Specialized queries for biomedical relationships
   - Data textualization for different entity types

4. **Translation System** (`translation.py`, `translators/`)
   - Multi-language support with automatic detection
   - Google Translate and OPUS-MT integration
   - Language confidence scoring

5. **Data Processing** (`embeddings.py`, `citation.py`, `callbacks.py`)
   - Custom sentence transformer embeddings
   - Citation formatting and bibliography generation
   - Callback handling for LlamaIndex operations

## File Structure

```
smo_chatbot_August_6th_2025/
├── src/                          # Main source code directory
│   ├── app.py                    # FastAPI application entry point
│   ├── main.py                   # Chainlit main application logic
│   ├── pipelines.py              # LLM and retrieval pipeline configuration
│   ├── query_engine.py           # Custom citation query engine
│   ├── retrievers.py             # Knowledge graph RAG retriever
│   ├── graph_stores.py           # Custom Neo4j graph store
│   ├── embeddings.py             # Sentence transformer embeddings
│   ├── translation.py            # Translation system
│   ├── citation.py               # Citation processing and formatting
│   ├── callbacks.py              # Custom LlamaIndex callbacks
│   ├── textualize.py             # Graph data textualization
│   ├── lingua_iso_codes.py       # Language code mappings
│   ├── reader.py                 # Document readers
│   ├── chat_engine/              # Custom chat engine implementations
│   │   ├── citation_types.py     # Chat mode enumerations
│   │   └── citation_condense_plus_context.py  # Citation chat engine
│   ├── translators/              # Translation implementations
│   │   ├── opusmt.py             # OPUS-MT translator
│   │   └── llm.py                # LLM-based translator
│   └── public/                   # Static assets
│       ├── favicon.png
│       ├── logo_dark.png
│       ├── logo_light.png
│       └── custom.js
├── prisma/                       # Database schema and migrations
│   ├── schema.prisma             # Prisma database schema
│   └── migrations/               # Database migration files
├── requirements.txt              # Python dependencies
├── package.json                  # Node.js dependencies (Prisma)
├── chainlit.md                   # Chainlit welcome page content
└── README.md                     # Basic project information
```

## Dependencies

### Python Dependencies (requirements.txt)
- **Core Framework**: `chainlit==1.0.401` - Main chat interface framework
- **LLM Integration**: 
  - `llama-index==0.10.20` - Core LLM framework
  - `llama_index_llms_groq==0.1.3` - Groq LLM integration
  - `llama-index-llms-openai==0.1.16` - OpenAI integration
  - `llama-index-llms-openrouter==0.1.4` - OpenRouter integration
- **Embeddings**: 
  - `sentence_transformers==2.5.1` - Sentence embeddings
  - `llama-index-embeddings-huggingface==0.1.4` - HuggingFace embeddings
- **Graph Database**: 
  - `neo4j==5.18.0` - Neo4j Python driver
  - `llama-index-graph-stores-neo4j==0.1.3` - Neo4j integration
- **Translation**: 
  - `deep-translator==1.11.4` - Google Translate integration
  - `lingua-language-detector==2.0.2` - Language detection
- **Citation Processing**: 
  - `metapub==0.5.5` - PubMed metadata retrieval
  - `pybtex==0.24.0` - Bibliography formatting
  - `pybtex-apa-style==1.3` - APA citation style
- **Database**: `asyncpg==0.30.0` - PostgreSQL async driver

### Node.js Dependencies (package.json)
- **Database ORM**: 
  - `@prisma/client==^6.1.0` - Prisma client
  - `prisma==^6.1.0` - Prisma CLI (dev dependency)

## Configuration

### Environment Variables
The application requires several environment variables:

```bash
# Database
DATABASE_URL=postgresql://username:password@host:port/database

# Neo4j Graph Database
NEO4J_PASSWORD=your_neo4j_password

# LLM API Keys
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
PERPLEXITY_API=your_perplexity_key
```

### Database Schema (Prisma)
The PostgreSQL database uses the following main entities:
- **User**: User accounts and metadata
- **Thread**: Conversation threads
- **Step**: Individual conversation steps
- **Element**: UI elements and attachments
- **Feedback**: User feedback on responses

## Data Flow

### Query Processing Pipeline

1. **User Input Processing**
   ```
   User Message → Language Detection → Translation (if needed) → Entity Extraction
   ```

2. **Knowledge Retrieval**
   ```
   Entities → Neo4j Graph Query → Knowledge Sequence → Node Building → Context Assembly
   ```

3. **Response Generation**
   ```
   Context + Query → Perplexity AI → Citation Processing → Translation (if needed) → User Response
   ```

### Knowledge Graph Structure
The Neo4j database contains biomedical entities with relationships:
- **Entities**: Diseases, genes, phenotypes, organizations
- **Relationships**: Has phenotype, has prevalence, associations, interactions
- **Citations**: PubMed IDs, references, confidence scores

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- PostgreSQL database
- Neo4j database (cloud or local)

### Step-by-Step Installation

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd smo_chatbot_August_6th_2025
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   npm install
   ```

3. **Database Setup**
   ```bash
   # Generate Prisma client
   npx prisma generate
   
   # Run database migrations
   npx prisma migrate deploy
   ```

4. **Environment Configuration**
   ```bash
   # Create .env file with required variables
   cp .env.example .env
   # Edit .env with your API keys and database URLs
   ```

5. **Run Application**
   ```bash
   # Development mode
   chainlit run src/main.py -w
   
   # Production mode with FastAPI
   uvicorn src.app:app --host 0.0.0.0 --port 8000
   ```

## Usage Examples

### Basic Query
```
User: "What are the biomarkers for diabetes?"
CMO: [Provides detailed response with citations and confidence scores]
```

### Multi-language Support
```
User: "¿Cuáles son los biomarcadores para la diabetes?" (Spanish)
CMO: [Detects Spanish, translates to English, processes, responds in Spanish]
```

### Citation Format
Responses include:
- Inline citations with confidence scores
- Reference list with PubMed links
- Further reading suggestions

## Testing

### Current Test Structure
The project currently lacks a comprehensive test suite. Recommended testing approach:

1. **Unit Tests**: Test individual components (embeddings, translation, citation processing)
2. **Integration Tests**: Test pipeline components together
3. **End-to-End Tests**: Test complete user workflows
4. **Performance Tests**: Test response times and resource usage

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests (when implemented)
pytest tests/
```

## Development Notes

### Key Design Decisions
1. **Perplexity Integration**: Currently uses Perplexity AI instead of the full LlamaIndex pipeline for faster responses
2. **Custom Graph Store**: Extended Neo4j integration for biomedical-specific queries
3. **Citation Processing**: Custom citation extraction and formatting for scientific accuracy
4. **Multi-language**: Comprehensive translation support for global accessibility

### Known Limitations
1. Limited to clinical metabolomics domain
2. Requires active internet connection for Perplexity API
3. Neo4j database needs to be populated with relevant biomedical data
4. Simple authentication system (not production-ready)

### Future Enhancements
1. Implement comprehensive test suite
2. Add more sophisticated authentication
3. Expand to other biomedical domains
4. Implement caching for improved performance
5. Add admin interface for content management

## API Documentation

### Core Classes and Methods

#### CustomNeo4jGraphStore (`graph_stores.py`)
Custom Neo4j graph store implementation for biomedical data.

**Key Methods:**
- `get_rel_map(subjs, depth, limit)`: Retrieve relationship mappings for entities
- `get_rel_map_phenotype(subjs, limit)`: Get phenotype relationships
- `get_rel_map_prevalence(subjs, limit)`: Get prevalence data
- `get_rel_map_pubtator3(subjs, limit)`: Get PubTator3 relationships

**Usage Example:**
```python
graph_store = CustomNeo4jGraphStore(
    username="neo4j",
    password=os.environ["NEO4J_PASSWORD"],
    url="neo4j+s://your-instance.databases.neo4j.io",
    database="neo4j",
    node_label="S_PHENOTYPE"
)
rel_map = graph_store.get_rel_map(["diabetes"], depth=1, limit=30)
```

#### KG_RAG_KnowledgeGraphRAGRetriever (`retrievers.py`)
Custom retriever for knowledge graph-based RAG.

**Key Parameters:**
- `graph_traversal_depth`: Depth of graph traversal (default: 1)
- `max_entities`: Maximum entities to extract (default: 5)
- `similarity_top_k`: Top-k similar nodes (default: 30)
- `max_knowledge_sequence`: Maximum knowledge sequence length (default: 1000)

**Usage Example:**
```python
retriever = KG_RAG_KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    graph_traversal_depth=1,
    max_entities=5,
    similarity_top_k=30
)
nodes = retriever.retrieve("diabetes biomarkers")
```

#### SentenceTransformerEmbeddings (`embeddings.py`)
Custom embedding implementation using SentenceTransformers.

**Key Methods:**
- `_get_query_embedding(query)`: Get embedding for query text
- `_get_text_embedding(text)`: Get embedding for document text
- `_get_text_embeddings(texts)`: Batch embedding generation

**Usage Example:**
```python
embeddings = SentenceTransformerEmbeddings(
    model_name_or_path="intfloat/e5-base-v2",
    embed_batch_size=8
)
query_embedding = embeddings._get_query_embedding("diabetes")
```

#### Translation System (`translation.py`)
Multi-language support with automatic detection and translation.

**Key Functions:**
- `get_translator(translator="google")`: Get translator instance
- `detect_language(detector, content)`: Detect content language
- `translate(translator, content, source, target)`: Translate text

**Usage Example:**
```python
translator = get_translator("google")
detector = get_language_detector(IsoCode639_1.ENGLISH, IsoCode639_1.SPANISH)
detection = await detect_language(detector, "Hola mundo")
translation = await translate(translator, "Hola mundo", source="es", target="en")
```

### Pipeline Configuration (`pipelines.py`)

#### LLM Providers
The system supports multiple LLM providers:

**OpenAI:**
```python
llm = get_llm("openai:gpt-4")
```

**Groq:**
```python
llm = get_llm("groq:Llama-3.3-70b-Versatile")
```

**OpenRouter:**
```python
llm = get_llm("openrouter:anthropic/claude-3-sonnet")
```

#### Embedding Models
Multiple embedding options available:

**HuggingFace:**
```python
embed_model, dim = get_huggingface_embed_model("mixedbread-ai/mxbai-embed-large-v1")
```

**SentenceTransformers:**
```python
embed_model, dim = get_sentence_transformer_embed_model("intfloat/e5-base-v2")
```

**Ollama:**
```python
embed_model, dim = get_ollama_embed_model("mxbai-embed-large:335m-v1-fp16")
```

## Data Sources Integration

The Clinical Metabolomics Oracle integrates data from multiple biomedical databases:

### Primary Sources
- **PubMed Articles**: Scientific literature and research papers
- **PubChem**: Chemical compound database
- **HMDB (Human Metabolome Database)**: Metabolite information
- **KEGG**: Pathway and enzyme data

### Secondary Sources
- **Massbank**: Mass spectrometry database
- **MoNa (MassBank of North America)**: Additional MS data
- **LipidBlast**: Lipid mass spectra
- **Metlin**: Metabolite database
- **mzCloud**: High-resolution MS/MS spectra

### Graph Database Schema
The Neo4j database contains structured relationships between:

**Entity Types:**
- `S_PHENOTYPE`: Phenotypic information
- `PubTator3`: Literature-derived entities
- `Disease`: Disease entities
- `Gene`: Genetic information

**Relationship Types:**
- `R_hasPhenotype`: Entity-phenotype relationships
- `R_hasPrevalence`: Prevalence information
- `R_rel`: General relationships
- `associate_PubTator3`: Literature associations
- `treat_PubTator3`: Treatment relationships
- `cause_PubTator3`: Causal relationships

## Performance Considerations

### Optimization Strategies
1. **Caching**: Schema caching for Neo4j queries
2. **Batch Processing**: Embedding batch processing for efficiency
3. **Connection Pooling**: Database connection management
4. **Async Operations**: Asynchronous processing where possible

### Resource Requirements
- **Memory**: Minimum 8GB RAM for embedding models
- **Storage**: Variable based on graph database size
- **Network**: Stable internet for API calls
- **CPU**: Multi-core recommended for parallel processing

### Monitoring and Logging
- **Chainlit Callbacks**: Custom callback handlers for operation tracking
- **Database Logging**: Conversation and feedback storage
- **Error Handling**: Comprehensive error logging and recovery

## Security Considerations

### Current Security Measures
1. **Authentication**: Basic password authentication
2. **Environment Variables**: Secure API key storage
3. **Database Security**: Prisma ORM with parameterized queries
4. **HTTPS**: Recommended for production deployment

### Security Recommendations
1. **Enhanced Authentication**: Implement OAuth2 or JWT tokens
2. **Rate Limiting**: Prevent API abuse
3. **Input Validation**: Sanitize user inputs
4. **Audit Logging**: Track user actions and system events
5. **Network Security**: Use VPNs and firewalls for database access

## Deployment Guide

### Production Deployment Options

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY prisma/ ./prisma/
COPY package.json .

RUN npm install
RUN npx prisma generate

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Cloud Deployment
**Recommended Platforms:**
- **AWS**: ECS with RDS (PostgreSQL) and Neo4j Aura
- **Google Cloud**: Cloud Run with Cloud SQL and Neo4j Aura
- **Azure**: Container Instances with Azure Database

### Environment-Specific Configurations

#### Development
```bash
# .env.development
DATABASE_URL=postgresql://localhost:5432/cmo_dev
NEO4J_PASSWORD=dev_password
DEBUG=true
```

#### Production
```bash
# .env.production
DATABASE_URL=postgresql://prod-host:5432/cmo_prod
NEO4J_PASSWORD=secure_password
DEBUG=false
RATE_LIMIT=100
```

## Troubleshooting Guide

### Common Issues

#### Database Connection Issues
```python
# Check Neo4j connectivity
try:
    driver.verify_connectivity()
except neo4j.exceptions.ServiceUnavailable:
    print("Neo4j service unavailable")
except neo4j.exceptions.AuthError:
    print("Authentication failed")
```

#### Translation Errors
```python
# Handle translation failures
try:
    translation = await translate(translator, content, source, target)
except Exception as e:
    print(f"Translation failed: {e}")
    # Fallback to original content
```

#### Memory Issues
- Monitor embedding model memory usage
- Implement batch size optimization
- Use model quantization if available

### Performance Debugging
1. **Profile Code**: Use cProfile for performance analysis
2. **Monitor Queries**: Log slow Neo4j queries
3. **Track API Calls**: Monitor external API response times
4. **Memory Profiling**: Use memory_profiler for memory usage analysis

## Contributing Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints where possible
- Document all public methods and classes
- Write descriptive commit messages

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request with description
5. Code review and merge

### Testing Requirements
- Unit tests for all new functions
- Integration tests for API endpoints
- Documentation updates for new features
- Performance impact assessment

This comprehensive technical documentation provides a complete overview of the Clinical Metabolomics Oracle system, its architecture, components, and usage patterns. The modular design allows for easy extension and maintenance while providing robust functionality for biomedical question answering.
