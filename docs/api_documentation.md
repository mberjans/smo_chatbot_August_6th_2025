# Clinical Metabolomics Oracle - LightRAG Integration API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Core Classes](#core-classes)
4. [Configuration System](#configuration-system)
5. [Quality Validation APIs](#quality-validation-apis)
6. [Cost Management APIs](#cost-management-apis)
7. [Document Processing APIs](#document-processing-apis)
8. [Data Models and Types](#data-models-and-types)
9. [Exception Classes](#exception-classes)
10. [Utility Functions](#utility-functions)
11. [Feature Flag System](#feature-flag-system)
12. [Integration Examples](#integration-examples)
13. [Environment Variables](#environment-variables)
14. [Advanced Usage](#advanced-usage)

---

## Overview

The Clinical Metabolomics Oracle - LightRAG Integration provides a comprehensive API for incorporating advanced Retrieval-Augmented Generation (RAG) capabilities into biomedical applications. This documentation covers all public APIs, configuration options, and integration patterns.

### Key Features
- **Advanced RAG System**: Optimized for clinical metabolomics queries
- **Cost Tracking**: Comprehensive budget management and API usage monitoring
- **Quality Validation**: Relevance scoring, accuracy assessment, and factual validation
- **Document Processing**: Specialized PDF handling for biomedical literature
- **Feature Flags**: Flexible feature toggling and A/B testing
- **Error Recovery**: Robust error handling and circuit breaker patterns
- **Performance Monitoring**: Benchmarking and performance optimization tools

---

## Getting Started

### Quick Start

```python
from lightrag_integration import create_clinical_rag_system

# Create a fully configured system
rag = await create_clinical_rag_system(
    daily_budget_limit=50.0,
    enable_quality_validation=True
)

# Initialize the system
await rag.initialize_rag()

# Process a metabolomics query
result = await rag.query(
    "What are the key metabolites in glucose metabolism?",
    mode="hybrid"
)

print(f"Response: {result.content}")
print(f"Cost: ${result.cost:.4f}")
```

### Environment Setup

```bash
# Core Configuration
export OPENAI_API_KEY="your-api-key-here"
export LIGHTRAG_MODEL="gpt-4o-mini"
export LIGHTRAG_EMBEDDING_MODEL="text-embedding-3-small"
export LIGHTRAG_WORKING_DIR="./lightrag_data"

# Cost Management
export LIGHTRAG_ENABLE_COST_TRACKING="true"
export LIGHTRAG_DAILY_BUDGET_LIMIT="50.0"
export LIGHTRAG_MONTHLY_BUDGET_LIMIT="1000.0"

# Quality Validation
export LIGHTRAG_ENABLE_QUALITY_VALIDATION="true"
export LIGHTRAG_RELEVANCE_THRESHOLD="0.75"
```

---

## Core Classes

### ClinicalMetabolomicsRAG

The main class for RAG operations with clinical metabolomics optimization.

#### Constructor

```python
class ClinicalMetabolomicsRAG:
    def __init__(self, config: LightRAGConfig, **kwargs):
        """
        Initialize the Clinical Metabolomics RAG system.
        
        Args:
            config: LightRAGConfig instance with validated configuration
            **kwargs: Optional parameters:
                - custom_model: Override the LLM model from config
                - custom_max_tokens: Override max tokens from config
                - enable_cost_tracking: Enable/disable cost tracking
                - pdf_processor: Optional BiomedicalPDFProcessor instance
                - rate_limiter: Custom rate limiter configuration
                - retry_config: Custom retry configuration
        
        Raises:
            ValueError: If config is None or invalid
            LightRAGConfigError: If configuration validation fails
            ClinicalMetabolomicsRAGError: If initialization fails
        """
```

#### Core Methods

##### `async initialize_rag()`

```python
async def initialize_rag(
    self, 
    papers_dir: Optional[Path] = None,
    auto_ingest: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Initialize the RAG system with document ingestion.
    
    Args:
        papers_dir: Directory containing PDF papers to ingest
        auto_ingest: Whether to automatically ingest PDFs
        **kwargs: Additional initialization parameters
        
    Returns:
        Dict containing initialization status and statistics
        
    Raises:
        ClinicalMetabolomicsRAGError: If initialization fails
        LightRAGConfigError: If configuration is invalid
    """
```

##### `async query()`

```python
async def query(
    self,
    query_text: str,
    mode: str = "hybrid",
    **query_params
) -> QueryResponse:
    """
    Execute a RAG query with cost tracking and quality assessment.
    
    Args:
        query_text: The query string
        mode: Query mode ("naive", "local", "global", "hybrid")
        **query_params: Additional query parameters:
            - top_k: Number of top results to return
            - max_tokens: Maximum response tokens
            - response_type: Response format
            - include_sources: Include source information
            - enable_quality_scoring: Enable quality assessment
            
    Returns:
        QueryResponse with content, metadata, cost, and performance data
        
    Raises:
        QueryError: If query processing fails
        CircuitBreakerError: If circuit breaker is open
        BudgetExceededError: If budget limits are exceeded
    """
```

##### `async ingest_documents()`

```python
async def ingest_documents(
    self,
    documents: Union[str, Path, List[Union[str, Path]]],
    batch_size: int = 10,
    show_progress: bool = True,
    **processing_kwargs
) -> Dict[str, Any]:
    """
    Ingest documents into the RAG system.
    
    Args:
        documents: Document path(s) or content to ingest
        batch_size: Number of documents to process in parallel
        show_progress: Whether to show progress indicators
        **processing_kwargs: Additional processing parameters
        
    Returns:
        Dict containing ingestion statistics and results
        
    Raises:
        BiomedicalPDFProcessorError: If document processing fails
        IngestionError: If ingestion fails
    """
```

##### Cost and Budget Methods

```python
def get_cost_summary(self) -> CostSummary:
    """Get comprehensive cost and usage statistics."""

async def check_budget_status(self) -> Dict[str, Any]:
    """Check current budget status and remaining limits."""

async def generate_cost_report(
    self, 
    report_type: str = "summary",
    export_format: str = "json"
) -> Union[Dict, str]:
    """Generate detailed cost reports."""
```

### BiomedicalPDFProcessor

Specialized processor for biomedical PDF documents.

#### Constructor

```python
class BiomedicalPDFProcessor:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        progress_tracker: Optional['PDFProcessingProgressTracker'] = None
    ):
        """
        Initialize the biomedical PDF processor.
        
        Args:
            config: Processing configuration options
            progress_tracker: Optional progress tracking instance
        """
```

#### Methods

```python
async def extract_text_from_pdf(
    self,
    pdf_path: Union[str, Path],
    **kwargs
) -> Dict[str, Any]:
    """
    Extract text and metadata from a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        **kwargs: Processing options:
            - page_range: Tuple of (start, end) pages
            - include_metadata: Include document metadata
            - preprocess_text: Apply biomedical preprocessing
            
    Returns:
        Dict containing extracted text, metadata, and statistics
        
    Raises:
        BiomedicalPDFProcessorError: If processing fails
        PDFValidationError: If PDF validation fails
    """

async def batch_process_pdfs(
    self,
    pdf_paths: List[Union[str, Path]],
    batch_size: int = 5,
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """Process multiple PDFs with memory management and error recovery."""
```

---

## Configuration System

### LightRAGConfig

Central configuration management for the RAG system.

#### Constructor

```python
@dataclass
class LightRAGConfig:
    """Comprehensive configuration class for LightRAG integration."""
    
    # Core Configuration
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("LIGHTRAG_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("LIGHTRAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    working_dir: Path = field(default_factory=lambda: Path(os.getenv("LIGHTRAG_WORKING_DIR", Path.cwd())))
    max_async: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_ASYNC", "16")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_TOKENS", "32768")))
    
    # Cost Tracking
    enable_cost_tracking: bool = field(default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_COST_TRACKING", "true").lower() in ("true", "1"))
    daily_budget_limit: Optional[float] = field(default_factory=lambda: float(os.getenv("LIGHTRAG_DAILY_BUDGET_LIMIT")) if os.getenv("LIGHTRAG_DAILY_BUDGET_LIMIT") else None)
    monthly_budget_limit: Optional[float] = field(default_factory=lambda: float(os.getenv("LIGHTRAG_MONTHLY_BUDGET_LIMIT")) if os.getenv("LIGHTRAG_MONTHLY_BUDGET_LIMIT") else None)
    
    # Quality Validation
    enable_relevance_scoring: bool = field(default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_RELEVANCE_SCORING", "true").lower() in ("true", "1"))
    relevance_confidence_threshold: float = field(default_factory=lambda: float(os.getenv("LIGHTRAG_RELEVANCE_CONFIDENCE_THRESHOLD", "70.0")))
    relevance_minimum_threshold: float = field(default_factory=lambda: float(os.getenv("LIGHTRAG_RELEVANCE_MINIMUM_THRESHOLD", "50.0")))
```

#### Factory Methods

```python
@classmethod
def from_env(cls, **overrides) -> 'LightRAGConfig':
    """Create configuration from environment variables with optional overrides."""

@classmethod
def from_file(cls, config_path: Union[str, Path], **overrides) -> 'LightRAGConfig':
    """Load configuration from JSON file with optional overrides."""

@classmethod
def from_dict(cls, config_dict: Dict[str, Any], **overrides) -> 'LightRAGConfig':
    """Create configuration from dictionary with optional overrides."""

@classmethod
def get_config(
    cls,
    source: Optional[Union[str, Path, Dict[str, Any]]] = None,
    validate_config: bool = True,
    ensure_dirs: bool = True,
    **overrides
) -> 'LightRAGConfig':
    """Universal configuration factory method."""
```

---

## Quality Validation APIs

### RelevanceScorer

Clinical metabolomics-specific relevance scoring.

```python
class ClinicalMetabolomicsRelevanceScorer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize relevance scorer with biomedical optimization."""
    
    async def score_relevance(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RelevanceScore:
        """Score the relevance of a response to a metabolomics query."""

@dataclass
class RelevanceScore:
    """Relevance scoring result with detailed metrics."""
    overall_score: float
    confidence: float
    biomedical_relevance: float
    metabolomics_specificity: float
    evidence_quality: float
    reasoning: str
    metadata: Dict[str, Any]
```

### QualityReportGenerator

```python
class QualityReportGenerator:
    def __init__(self, rag_system: ClinicalMetabolomicsRAG):
        """Initialize quality report generator."""
    
    async def generate_comprehensive_report(
        self,
        output_format: str = "html",
        include_charts: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """Generate comprehensive quality assessment report."""
```

---

## Cost Management APIs

### BudgetManager

```python
class BudgetManager:
    def __init__(self, config: Dict[str, Any]):
        """Initialize budget management system."""
    
    async def check_budget_availability(
        self,
        estimated_cost: float,
        operation_type: str = "query"
    ) -> BudgetStatus:
        """Check if operation is within budget limits."""
    
    async def record_cost(
        self,
        cost: float,
        operation_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record API cost for budget tracking."""

@dataclass
class BudgetStatus:
    """Budget status information."""
    within_budget: bool
    remaining_daily: Optional[float]
    remaining_monthly: Optional[float]
    usage_percentage: float
    alert_level: str
```

### CostPersistence

```python
class CostPersistence:
    def __init__(self, db_path: Path):
        """Initialize cost persistence layer."""
    
    async def save_cost_record(self, record: CostRecord) -> None:
        """Save cost record to database."""
    
    async def get_cost_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        category: Optional[ResearchCategory] = None
    ) -> Dict[str, Any]:
        """Retrieve cost summary for specified period and category."""

@dataclass
class CostRecord:
    """Individual cost record for database storage."""
    timestamp: datetime
    operation_type: str
    cost: float
    tokens_used: int
    query_text: str
    research_category: ResearchCategory
    session_id: Optional[str]
    metadata: Dict[str, Any]
```

---

## Document Processing APIs

### Document Indexing

```python
class DocumentIndexer:
    def __init__(self, rag_system: ClinicalMetabolomicsRAG):
        """Initialize document indexer."""
    
    async def index_document(
        self,
        document_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IndexingResult:
        """Index a single document into the RAG system."""

@dataclass
class IndexingResult:
    """Result of document indexing operation."""
    success: bool
    document_id: str
    tokens_processed: int
    processing_time: float
    metadata: Dict[str, Any]
    errors: List[str]
```

---

## Data Models and Types

### Query Response Types

```python
@dataclass
class QueryResponse:
    """Complete response from RAG query operation."""
    content: str
    metadata: Dict[str, Any]
    cost: float
    token_usage: Dict[str, int]
    query_mode: str
    processing_time: float

@dataclass
class CostSummary:
    """API cost and usage statistics summary."""
    total_cost: float
    total_queries: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    embedding_tokens: int
    average_cost_per_query: float
    query_history_count: int
```

### Research Categories

```python
class ResearchCategory(Enum):
    """Research category classification for cost tracking."""
    METABOLITE_IDENTIFICATION = "metabolite_identification"
    PATHWAY_ANALYSIS = "pathway_analysis"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    DRUG_DISCOVERY = "drug_discovery"
    CLINICAL_DIAGNOSIS = "clinical_diagnosis"
    DATA_PREPROCESSING = "data_preprocessing"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    LITERATURE_SEARCH = "literature_search"
    DATABASE_INTEGRATION = "database_integration"
    EXPERIMENTAL_VALIDATION = "experimental_validation"
    GENERAL_QUERY = "general_query"
```

---

## Exception Classes

### Core Exceptions

```python
class ClinicalMetabolomicsRAGError(Exception):
    """Base exception for RAG system errors."""

class LightRAGConfigError(Exception):
    """Configuration validation and setup errors."""

class CircuitBreakerError(Exception):
    """Circuit breaker protection errors."""

class QueryError(ClinicalMetabolomicsRAGError):
    """Base class for query processing errors."""
    def __init__(self, message: str, query: str, original_exception: Exception):
        self.query = query
        self.original_exception = original_exception
        super().__init__(message)

class BudgetExceededError(QueryError):
    """Exception raised when budget limits are exceeded."""

class BiomedicalPDFProcessorError(Exception):
    """Base exception for PDF processing errors."""

class PDFValidationError(BiomedicalPDFProcessorError):
    """PDF file validation failures."""
```

---

## Utility Functions

### Factory Functions

```python
def create_clinical_rag_system(
    config_source=None,
    **config_overrides
) -> ClinicalMetabolomicsRAG:
    """
    Primary factory function to create Clinical Metabolomics RAG system.
    
    Args:
        config_source: Configuration source (None for env, path for file, dict for direct)
        **config_overrides: Configuration parameter overrides
        
    Returns:
        Fully configured RAG system with enhanced features
        
    Example:
        ```python
        # Basic usage
        rag = create_clinical_rag_system()
        
        # Custom configuration
        rag = create_clinical_rag_system(
            daily_budget_limit=50.0,
            enable_quality_validation=True,
            model="gpt-4o"
        )
        
        # From config file
        rag = create_clinical_rag_system("config.json")
        ```
    """

def get_quality_validation_config(**overrides) -> Dict[str, Any]:
    """Get configuration optimized for quality validation workflows."""

def get_default_research_categories() -> List[Dict[str, str]]:
    """Get default research categories for metabolomics research tracking."""
```

### Validation Functions

```python
def validate_integration_setup() -> Tuple[bool, List[str]]:
    """Validate integration setup and return status with any issues."""

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific feature is enabled via feature flags."""

def get_enabled_features() -> Dict[str, bool]:
    """Get all currently enabled features and their status."""
```

---

## Feature Flag System

### Feature Flag Management

```python
def is_feature_enabled(feature_name: str) -> bool:
    """
    Check if a specific feature is enabled via environment variables.
    
    Args:
        feature_name: Name of the feature to check
        
    Returns:
        True if the feature is enabled, False otherwise
        
    Common Features:
        - quality_validation_enabled: Enable quality validation
        - performance_monitoring_enabled: Enable performance monitoring
        - cost_tracking_enabled: Enable cost tracking
        - relevance_scoring_enabled: Enable relevance scoring
        - pdf_processing_enabled: Enable PDF document processing
        - benchmarking_enabled: Enable performance benchmarking
        - recovery_system_enabled: Enable advanced error recovery
    """

def get_enabled_features() -> Dict[str, bool]:
    """Get all currently enabled features and their status."""

def get_integration_status() -> Dict[str, Any]:
    """Get comprehensive integration status including feature flags and module availability."""
```

---

## Integration Examples

### Basic Integration

```python
import asyncio
from lightrag_integration import create_clinical_rag_system

async def basic_example():
    # Create RAG system with default settings
    rag = create_clinical_rag_system()
    
    # Initialize with PDF documents
    await rag.initialize_rag(
        papers_dir="./papers",
        auto_ingest=True
    )
    
    # Process a query
    response = await rag.query(
        "What are the key metabolites in diabetes?",
        mode="hybrid"
    )
    
    print(f"Response: {response.content}")
    print(f"Cost: ${response.cost:.4f}")
    
    # Check budget status
    budget_status = await rag.check_budget_status()
    print(f"Remaining daily budget: ${budget_status.get('remaining_daily', 0):.2f}")

asyncio.run(basic_example())
```

### Advanced Integration with Quality Assessment

```python
from lightrag_integration import (
    create_clinical_rag_system,
    QualityReportGenerator,
    get_quality_validation_config
)

async def advanced_example():
    # Create system with quality validation
    config = get_quality_validation_config(
        relevance_threshold=0.85,
        accuracy_threshold=0.90,
        enable_claim_extraction=True
    )
    
    rag = create_clinical_rag_system(**config)
    await rag.initialize_rag()
    
    # Process query with quality assessment
    response = await rag.query(
        "Explain the metabolic pathways in liver disease",
        mode="hybrid",
        enable_quality_scoring=True
    )
    
    # Generate quality report
    if hasattr(rag, 'relevance_scorer') and rag.relevance_scorer:
        reporter = QualityReportGenerator(rag)
        quality_report = await reporter.generate_comprehensive_report(
            output_format="html",
            include_charts=True
        )
        
        print("Quality report generated successfully")

asyncio.run(advanced_example())
```

### Chainlit Integration

```python
import chainlit as cl
from lightrag_integration import create_clinical_rag_system

# Global RAG system
rag_system = None

@cl.on_chat_start
async def start():
    global rag_system
    
    # Initialize RAG system
    rag_system = create_clinical_rag_system(
        daily_budget_limit=50.0,
        enable_quality_validation=True
    )
    
    await rag_system.initialize_rag(
        papers_dir="./papers",
        auto_ingest=True
    )
    
    cl.user_session.set("rag_system", rag_system)
    
    await cl.Message(
        content="🧬 Clinical Metabolomics Oracle ready! Ask me about metabolomics research."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    rag_system = cl.user_session.get("rag_system")
    
    # Process query
    response = await rag_system.query(
        message.content,
        mode="hybrid"
    )
    
    # Send response
    await cl.Message(
        content=response.content
    ).send()
    
    # Show cost information
    await cl.Message(
        content=f"💰 Query cost: ${response.cost:.4f} | Processing time: {response.processing_time:.2f}s",
        author="System"
    ).send()
```

---

## Environment Variables

### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key for LLM and embedding operations |
| `LIGHTRAG_MODEL` | `gpt-4o-mini` | LLM model for query processing |
| `LIGHTRAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for document processing |
| `LIGHTRAG_WORKING_DIR` | Current directory | Working directory for LightRAG data storage |
| `LIGHTRAG_MAX_ASYNC` | `16` | Maximum concurrent async operations |
| `LIGHTRAG_MAX_TOKENS` | `32768` | Maximum token limit for responses |

### Cost Management

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_ENABLE_COST_TRACKING` | `true` | Enable cost tracking and budget management |
| `LIGHTRAG_DAILY_BUDGET_LIMIT` | None | Daily budget limit in USD |
| `LIGHTRAG_MONTHLY_BUDGET_LIMIT` | None | Monthly budget limit in USD |
| `LIGHTRAG_COST_ALERT_THRESHOLD` | `80.0` | Budget alert threshold percentage |
| `LIGHTRAG_ENABLE_BUDGET_ALERTS` | `true` | Enable budget alert notifications |

### Quality Validation

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_ENABLE_QUALITY_VALIDATION` | `true` | Enable quality validation features |
| `LIGHTRAG_ENABLE_RELEVANCE_SCORING` | `true` | Enable relevance scoring |
| `LIGHTRAG_RELEVANCE_CONFIDENCE_THRESHOLD` | `70.0` | Relevance confidence threshold |
| `LIGHTRAG_RELEVANCE_MINIMUM_THRESHOLD` | `50.0` | Minimum relevance threshold |
| `LIGHTRAG_ENABLE_ACCURACY_VALIDATION` | `false` | Enable accuracy validation |
| `LIGHTRAG_ENABLE_FACTUAL_VALIDATION` | `false` | Enable factual validation |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_INTEGRATION_ENABLED` | `false` | Master integration toggle |
| `LIGHTRAG_ENABLE_PERFORMANCE_MONITORING` | `false` | Enable performance monitoring |
| `LIGHTRAG_ENABLE_BENCHMARKING` | `false` | Enable performance benchmarking |
| `LIGHTRAG_ENABLE_RECOVERY_SYSTEM` | `false` | Enable advanced error recovery |
| `LIGHTRAG_ENABLE_DOCUMENT_INDEXING` | `false` | Enable document indexing features |

---

## Advanced Usage

### Custom Error Recovery

```python
from lightrag_integration import (
    ClinicalMetabolomicsRAG,
    CircuitBreaker,
    RateLimiter
)

# Create system with custom error recovery
rag = ClinicalMetabolomicsRAG(
    config=config,
    circuit_breaker={
        'failure_threshold': 3,
        'recovery_timeout': 30.0
    },
    rate_limiter={
        'max_concurrent_requests': 10,
        'request_timeout': 60.0
    }
)
```

### Batch Processing

```python
async def batch_process_queries():
    rag = create_clinical_rag_system()
    await rag.initialize_rag()
    
    queries = [
        "What are metabolites in diabetes?",
        "Explain glycolysis pathway",
        "Biomarkers for liver disease"
    ]
    
    # Process queries concurrently
    tasks = [
        rag.query(query, mode="hybrid")
        for query in queries
    ]
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"Query {i} failed: {response}")
        else:
            print(f"Query {i} cost: ${response.cost:.4f}")
```

### Performance Monitoring

```python
from lightrag_integration.performance_benchmarking import (
    QualityValidationBenchmarkSuite,
    QualityBenchmarkConfiguration
)

# Create benchmark suite
config = QualityBenchmarkConfiguration(
    test_query_count=100,
    include_quality_metrics=True,
    benchmark_timeout=300
)

benchmarks = QualityValidationBenchmarkSuite(
    rag_system=rag,
    config=config
)

# Run comprehensive benchmarks
results = await benchmarks.run_comprehensive_benchmarks()
print(f"Average response time: {results['performance']['avg_response_time']:.2f}s")
print(f"Average relevance score: {results['quality']['avg_relevance_score']:.2f}")
```

---

This API documentation provides comprehensive coverage of the Clinical Metabolomics Oracle - LightRAG integration system. For additional examples and troubleshooting, refer to the `examples/` directory and `docs/INTEGRATION_TROUBLESHOOTING_GUIDE.md`.