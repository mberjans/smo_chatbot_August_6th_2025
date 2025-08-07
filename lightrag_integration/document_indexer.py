"""
SourceDocumentIndex for Clinical Metabolomics Oracle - Document content extraction and indexing system.

This module provides the SourceDocumentIndex class for extracting and indexing key factual
information from PDF documents to support factual accuracy validation against source documents
in the Clinical Metabolomics Oracle LightRAG integration project.

Classes:
    - DocumentIndexError: Base custom exception for document indexing errors
    - ContentExtractionError: Exception for content extraction failures
    - IndexingError: Exception for indexing operation failures
    - ClaimVerificationError: Exception for claim verification process failures
    - IndexedContent: Data class for structured indexed content
    - NumericFact: Data class for numeric facts and measurements
    - ScientificStatement: Data class for scientific relationships and statements
    - MethodologicalInfo: Data class for methodological information
    - SourceDocumentIndex: Main class for document content indexing and retrieval

The indexer handles:
    - Extracting structured content from processed PDFs using BiomedicalPDFProcessor
    - Indexing content by different categories (numeric data, relationships, methodologies)
    - Providing fast lookup capabilities for claim verification
    - Integration with existing LightRAG storage systems
    - Async support for performance optimization
    - Comprehensive error handling and recovery mechanisms

Key Features:
    - Multi-level content extraction (numeric facts, scientific statements, methodologies)
    - Efficient indexing with multiple search strategies
    - Fast retrieval methods for claim matching and verification
    - Integration with existing document processing pipeline
    - Async processing capabilities for large document collections
    - Structured storage format for indexed content
    - Advanced text analysis for factual content identification
    - Support for different content types and scientific domains
"""

import asyncio
import json
import logging
import re
import sqlite3
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict
import pickle

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .pdf_processor import BiomedicalPDFProcessor

# Enhanced logging imports
try:
    from .enhanced_logging import (
        EnhancedLogger, correlation_manager, performance_logged, PerformanceTracker
    )
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    # Fallback for when enhanced logging is not available
    ENHANCED_LOGGING_AVAILABLE = False
    
    def performance_logged(description="", track_memory=True):
        """Fallback performance logging decorator."""
        def decorator(func):
            return func
        return decorator


class DocumentIndexError(Exception):
    """Base custom exception for document indexing errors."""
    pass


class ContentExtractionError(DocumentIndexError):
    """Exception raised when content extraction fails."""
    pass


class IndexingError(DocumentIndexError):
    """Exception raised when indexing operations fail."""
    pass


class ClaimVerificationError(DocumentIndexError):
    """Exception raised when claim verification fails."""
    pass


@dataclass
class NumericFact:
    """
    Data class for numeric facts and measurements extracted from documents.
    
    Attributes:
        value: The numeric value
        unit: Unit of measurement (e.g., 'mg/L', 'μM', 'years')
        context: Surrounding context describing what the value represents
        confidence: Confidence score for extraction accuracy (0.0-1.0)
        source_location: Location in source document (page, section)
        variable_name: Name of the measured variable
        method: Measurement method if specified
        error_margin: Error margin or standard deviation if provided
    """
    value: float
    unit: Optional[str]
    context: str
    confidence: float
    source_location: Dict[str, Any]
    variable_name: Optional[str] = None
    method: Optional[str] = None
    error_margin: Optional[float] = None
    
    def __post_init__(self):
        """Validate numeric fact data after initialization."""
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Value must be numeric, got {type(self.value)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class ScientificStatement:
    """
    Data class for scientific relationships and statements extracted from documents.
    
    Attributes:
        subject: Subject of the statement (e.g., 'glucose', 'insulin resistance')
        predicate: Relationship or action (e.g., 'correlates with', 'increases')
        object: Object of the relationship (e.g., 'diabetes risk', 'metabolite concentration')
        context: Full context of the statement
        confidence: Confidence score for extraction accuracy (0.0-1.0)
        source_location: Location in source document
        study_type: Type of study supporting the statement
        evidence_strength: Strength of evidence (weak, moderate, strong)
        statistical_significance: p-value or significance indicator if available
    """
    subject: str
    predicate: str
    object: str
    context: str
    confidence: float
    source_location: Dict[str, Any]
    study_type: Optional[str] = None
    evidence_strength: Optional[str] = None
    statistical_significance: Optional[str] = None
    
    def __post_init__(self):
        """Validate scientific statement data after initialization."""
        if not all([self.subject, self.predicate, self.object]):
            raise ValueError("Subject, predicate, and object are required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class MethodologicalInfo:
    """
    Data class for methodological information extracted from documents.
    
    Attributes:
        method_type: Type of method (e.g., 'LC-MS', 'statistical analysis', 'sample preparation')
        description: Detailed description of the method
        parameters: Key parameters and settings
        equipment: Equipment or software used
        validation: Validation information if provided
        limitations: Known limitations of the method
        context: Context where method is described
        confidence: Confidence score for extraction accuracy (0.0-1.0)
        source_location: Location in source document
    """
    method_type: str
    description: str
    parameters: Dict[str, Any]
    equipment: List[str]
    validation: Optional[str]
    limitations: Optional[str]
    context: str
    confidence: float
    source_location: Dict[str, Any]
    
    def __post_init__(self):
        """Validate methodological info data after initialization."""
        if not self.method_type or not self.description:
            raise ValueError("Method type and description are required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class IndexedContent:
    """
    Data class for complete indexed content from a document.
    
    Attributes:
        document_id: Unique identifier for the document
        document_path: Path to the original document
        content_hash: Hash of the processed content for change detection
        numeric_facts: List of extracted numeric facts
        scientific_statements: List of extracted scientific statements
        methodological_info: List of extracted methodological information
        extraction_metadata: Metadata about the extraction process
        created_at: Timestamp of content creation
        updated_at: Timestamp of last update
    """
    document_id: str
    document_path: str
    content_hash: str
    numeric_facts: List[NumericFact] = field(default_factory=list)
    scientific_statements: List[ScientificStatement] = field(default_factory=list)
    methodological_info: List[MethodologicalInfo] = field(default_factory=list)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate indexed content data after initialization."""
        if not self.document_id or not self.document_path:
            raise ValueError("Document ID and path are required")
        if not self.content_hash:
            raise ValueError("Content hash is required")


class SourceDocumentIndex:
    """
    Document content extraction and indexing system for factual accuracy validation.
    
    This class provides comprehensive document content indexing capabilities specifically
    designed for the Clinical Metabolomics Oracle's factual accuracy validation system.
    It extracts and indexes key factual information from PDF documents processed by
    BiomedicalPDFProcessor and provides fast lookup capabilities for claim verification.
    
    Key Features:
        - Multi-level content extraction (numeric facts, scientific statements, methodologies)
        - Efficient SQLite-based indexing with full-text search capabilities
        - Integration with existing BiomedicalPDFProcessor
        - Async support for high-performance processing
        - Advanced text analysis for factual content identification
        - Claim verification and matching algorithms
        - Comprehensive error handling and recovery
        
    Attributes:
        index_dir: Directory for storing index files
        logger: Logger instance for tracking operations
        db_path: Path to the SQLite database file
        pdf_processor: Instance of BiomedicalPDFProcessor for document processing
        
    Example:
        indexer = SourceDocumentIndex(index_dir="./document_index")
        await indexer.initialize()
        
        # Index a document
        indexed_content = await indexer.index_document("path/to/document.pdf")
        
        # Verify a claim
        verification_result = await indexer.verify_claim(
            "Glucose levels were 150 mg/dL in diabetic patients"
        )
    """
    
    def __init__(self,
                 index_dir: Union[str, Path] = "./document_index",
                 logger: Optional[logging.Logger] = None,
                 pdf_processor: Optional['BiomedicalPDFProcessor'] = None,
                 enable_full_text_search: bool = True,
                 content_extraction_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SourceDocumentIndex.
        
        Args:
            index_dir: Directory for storing index files and database
            logger: Optional logger instance. If None, creates a default logger
            pdf_processor: Optional BiomedicalPDFProcessor instance. If None, creates one
            enable_full_text_search: Whether to enable SQLite FTS for advanced search
            content_extraction_config: Configuration for content extraction parameters
        """
        self.index_dir = Path(index_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.pdf_processor = pdf_processor
        self.enable_full_text_search = enable_full_text_search
        
        # Initialize content extraction configuration
        self.content_extraction_config = content_extraction_config or {
            'numeric_confidence_threshold': 0.7,
            'statement_confidence_threshold': 0.6,
            'method_confidence_threshold': 0.8,
            'max_context_length': 500,
            'enable_statistical_detection': True,
            'enable_unit_normalization': True
        }
        
        # Database and file paths
        self.db_path = self.index_dir / "document_index.db"
        self.content_cache_path = self.index_dir / "content_cache"
        
        # Internal state
        self._db_initialized = False
        self._content_extractors = {}
        self._performance_tracker = PerformanceTracker() if ENHANCED_LOGGING_AVAILABLE else None
        
        # Content extraction patterns (compiled for performance)
        self._compile_extraction_patterns()
        
        self.logger.info(f"SourceDocumentIndex initialized with index directory: {self.index_dir}")
    
    def _compile_extraction_patterns(self):
        """Compile regular expressions for content extraction."""
        # Numeric patterns for various scientific measurements
        self._numeric_patterns = {
            'basic_measurement': re.compile(
                r'(\w+(?:\s+\w+)*)\s*(?:was|were|is|are|measured|found|observed|detected|showed|showed|equals?)\s*'
                r'(?:at\s+|to\s+be\s+|as\s+|approximately\s+|about\s+)?'
                r'([\d,]+\.?\d*)\s*(?:±\s*([\d,]+\.?\d*))?\s*'
                r'([μmMnpkgLlmolsHzhA-Z/°%]+)',
                re.IGNORECASE
            ),
            'range_measurement': re.compile(
                r'(\w+(?:\s+\w+)*)\s*(?:ranged?\s+from\s+|between\s+)'
                r'([\d,]+\.?\d*)\s*(?:and\s+|to\s+|-\s+)([\d,]+\.?\d*)\s*'
                r'([μmMnpkgLlmolsHzhA-Z/°%]+)',
                re.IGNORECASE
            ),
            'statistical_value': re.compile(
                r'(p[-\s]*value|p|r²|R²|correlation|confidence\s+interval|CI|mean|median|std|standard\s+deviation)\s*'
                r'[=<>≤≥]\s*([\d,]+\.?\d*(?:[eE][-+]?\d+)?)',
                re.IGNORECASE
            )
        }
        
        # Scientific statement patterns
        self._statement_patterns = {
            'correlation': re.compile(
                r'(\w+(?:\s+\w+)*)\s+(correlat[es|ed|ing]*|associat[es|ed|ing]*|link[s|ed|ing]*)\s+(?:with|to)\s+(\w+(?:\s+\w+)*)',
                re.IGNORECASE
            ),
            'causation': re.compile(
                r'(\w+(?:\s+\w+)*)\s+(caus[es|ed|ing]*|lead[s|ing]*\s+to|result[s|ed|ing]*\s+in|induc[es|ed|ing]*)\s+(\w+(?:\s+\w+)*)',
                re.IGNORECASE
            ),
            'comparison': re.compile(
                r'(\w+(?:\s+\w+)*)\s+(?:was|were|is|are)\s+(higher|lower|greater|smaller|increased|decreased|elevated|reduced)\s+(?:than|compared\s+to)\s+(\w+(?:\s+\w+)*)',
                re.IGNORECASE
            ),
            'effect': re.compile(
                r'(\w+(?:\s+\w+)*)\s+(increas[es|ed|ing]*|decreas[es|ed|ing]*|reduc[es|ed|ing]*|elevat[es|ed|ing]*|lower[s|ed|ing]*)\s+(\w+(?:\s+\w+)*)',
                re.IGNORECASE
            )
        }
        
        # Methodological patterns
        self._method_patterns = {
            'analytical_method': re.compile(
                r'(LC-MS|GC-MS|HPLC|NMR|ELISA|Western\s+blot|qPCR|PCR|sequencing|chromatography|spectrometry|spectroscopy)',
                re.IGNORECASE
            ),
            'statistical_method': re.compile(
                r'(t-test|ANOVA|regression|correlation\s+analysis|Mann-Whitney|Wilcoxon|chi-square|Fisher|Pearson|Spearman)',
                re.IGNORECASE
            ),
            'sample_preparation': re.compile(
                r'(extraction|purification|centrifug|filtrat|dilut|incubat|homogeniz|lyophiliz|precipitat)',
                re.IGNORECASE
            )
        }
    
    async def initialize(self) -> None:
        """
        Initialize the document indexing system.
        
        This method sets up the database, creates necessary directories,
        and initializes the PDF processor if not provided.
        
        Raises:
            IndexingError: If initialization fails
        """
        try:
            # Create directories
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.content_cache_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize PDF processor if not provided
            if self.pdf_processor is None:
                from .pdf_processor import BiomedicalPDFProcessor
                self.pdf_processor = BiomedicalPDFProcessor(logger=self.logger)
                self.logger.info("Created BiomedicalPDFProcessor instance")
            
            # Initialize database
            await self._initialize_database()
            
            # Load any existing content extractors
            self._load_content_extractors()
            
            self.logger.info("SourceDocumentIndex initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SourceDocumentIndex: {e}")
            raise IndexingError(f"Initialization failed: {e}")
    
    async def _initialize_database(self) -> None:
        """Initialize the SQLite database with required tables and indexes."""
        try:
            # Create database connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enable foreign keys
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Create main documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL,
                    extraction_metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create numeric facts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS numeric_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    context TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_location TEXT,
                    variable_name TEXT,
                    method TEXT,
                    error_margin REAL,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            ''')
            
            # Create scientific statements table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scientific_statements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    context TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_location TEXT,
                    study_type TEXT,
                    evidence_strength TEXT,
                    statistical_significance TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            ''')
            
            # Create methodological info table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS methodological_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    method_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    parameters TEXT,
                    equipment TEXT,
                    validation TEXT,
                    limitations TEXT,
                    context TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_location TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for better query performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_documents_path ON documents (path)",
                "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents (content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_numeric_facts_document ON numeric_facts (document_id)",
                "CREATE INDEX IF NOT EXISTS idx_numeric_facts_value ON numeric_facts (value)",
                "CREATE INDEX IF NOT EXISTS idx_numeric_facts_unit ON numeric_facts (unit)",
                "CREATE INDEX IF NOT EXISTS idx_numeric_facts_variable ON numeric_facts (variable_name)",
                "CREATE INDEX IF NOT EXISTS idx_statements_document ON scientific_statements (document_id)",
                "CREATE INDEX IF NOT EXISTS idx_statements_subject ON scientific_statements (subject)",
                "CREATE INDEX IF NOT EXISTS idx_statements_predicate ON scientific_statements (predicate)",
                "CREATE INDEX IF NOT EXISTS idx_statements_object ON scientific_statements (object)",
                "CREATE INDEX IF NOT EXISTS idx_methods_document ON methodological_info (document_id)",
                "CREATE INDEX IF NOT EXISTS idx_methods_type ON methodological_info (method_type)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            # Create full-text search tables if enabled
            if self.enable_full_text_search:
                # FTS table for numeric facts
                cursor.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS numeric_facts_fts USING fts5(
                        context, variable_name, method,
                        content='numeric_facts',
                        content_rowid='id'
                    )
                ''')
                
                # FTS table for scientific statements
                cursor.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS statements_fts USING fts5(
                        subject, predicate, object, context,
                        content='scientific_statements',
                        content_rowid='id'
                    )
                ''')
                
                # FTS table for methodological info
                cursor.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS methods_fts USING fts5(
                        method_type, description, context,
                        content='methodological_info',
                        content_rowid='id'
                    )
                ''')
                
                # Create triggers to maintain FTS indexes
                fts_triggers = [
                    '''
                    CREATE TRIGGER IF NOT EXISTS numeric_facts_fts_insert AFTER INSERT ON numeric_facts BEGIN
                        INSERT INTO numeric_facts_fts(rowid, context, variable_name, method)
                        VALUES (new.id, new.context, new.variable_name, new.method);
                    END
                    ''',
                    '''
                    CREATE TRIGGER IF NOT EXISTS numeric_facts_fts_delete AFTER DELETE ON numeric_facts BEGIN
                        INSERT INTO numeric_facts_fts(numeric_facts_fts, rowid, context, variable_name, method)
                        VALUES ('delete', old.id, old.context, old.variable_name, old.method);
                    END
                    ''',
                    '''
                    CREATE TRIGGER IF NOT EXISTS statements_fts_insert AFTER INSERT ON scientific_statements BEGIN
                        INSERT INTO statements_fts(rowid, subject, predicate, object, context)
                        VALUES (new.id, new.subject, new.predicate, new.object, new.context);
                    END
                    ''',
                    '''
                    CREATE TRIGGER IF NOT EXISTS statements_fts_delete AFTER DELETE ON scientific_statements BEGIN
                        INSERT INTO statements_fts(statements_fts, rowid, subject, predicate, object, context)
                        VALUES ('delete', old.id, old.subject, old.predicate, old.object, old.context);
                    END
                    ''',
                    '''
                    CREATE TRIGGER IF NOT EXISTS methods_fts_insert AFTER INSERT ON methodological_info BEGIN
                        INSERT INTO methods_fts(rowid, method_type, description, context)
                        VALUES (new.id, new.method_type, new.description, new.context);
                    END
                    ''',
                    '''
                    CREATE TRIGGER IF NOT EXISTS methods_fts_delete AFTER DELETE ON methodological_info BEGIN
                        INSERT INTO methods_fts(methods_fts, rowid, method_type, description, context)
                        VALUES ('delete', old.id, old.method_type, old.description, old.context);
                    END
                    '''
                ]
                
                for trigger_sql in fts_triggers:
                    cursor.execute(trigger_sql)
            
            conn.commit()
            conn.close()
            
            self._db_initialized = True
            self.logger.info("Database initialization completed successfully")
            
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise IndexingError(f"Database setup failed: {e}")
    
    def _load_content_extractors(self) -> None:
        """Load any custom content extractors from the cache directory."""
        try:
            extractor_cache = self.content_cache_path / "extractors.pkl"
            if extractor_cache.exists():
                with open(extractor_cache, 'rb') as f:
                    self._content_extractors = pickle.load(f)
                self.logger.info(f"Loaded {len(self._content_extractors)} custom content extractors")
        except Exception as e:
            self.logger.warning(f"Could not load content extractors: {e}")
            self._content_extractors = {}
    
    def _save_content_extractors(self) -> None:
        """Save custom content extractors to cache."""
        try:
            extractor_cache = self.content_cache_path / "extractors.pkl"
            with open(extractor_cache, 'wb') as f:
                pickle.dump(self._content_extractors, f)
            self.logger.debug("Saved custom content extractors to cache")
        except Exception as e:
            self.logger.warning(f"Could not save content extractors: {e}")
    
    def _generate_document_id(self, document_path: Union[str, Path]) -> str:
        """
        Generate a unique document ID based on the document path.
        
        Args:
            document_path: Path to the document
            
        Returns:
            str: Unique document ID
        """
        path_str = str(Path(document_path).resolve())
        return hashlib.md5(path_str.encode()).hexdigest()
    
    def _generate_content_hash(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a hash of the document content for change detection.
        
        Args:
            text: Extracted text content
            metadata: Document metadata
            
        Returns:
            str: Content hash
        """
        content_data = {
            'text': text,
            'metadata': {k: v for k, v in metadata.items() if k not in ['processing_timestamp']}
        }
        content_str = json.dumps(content_data, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    @performance_logged("document indexing")
    async def index_document(self, 
                           document_path: Union[str, Path],
                           force_reindex: bool = False,
                           extract_config: Optional[Dict[str, Any]] = None) -> IndexedContent:
        """
        Index a document by extracting and storing its factual content.
        
        This method processes a PDF document using BiomedicalPDFProcessor,
        extracts structured factual content, and stores it in the index
        for fast retrieval during claim verification.
        
        Args:
            document_path: Path to the PDF document to index
            force_reindex: Whether to force reindexing even if document exists
            extract_config: Optional extraction configuration overrides
            
        Returns:
            IndexedContent: The indexed content from the document
            
        Raises:
            ContentExtractionError: If content extraction fails
            IndexingError: If indexing operations fail
        """
        document_path = Path(document_path)
        document_id = self._generate_document_id(document_path)
        
        self.logger.info(f"Starting document indexing for: {document_path}")
        
        try:
            # Check if document is already indexed
            if not force_reindex:
                existing_content = await self._get_existing_indexed_content(document_id)
                if existing_content:
                    # Verify content hasn't changed
                    current_content = await self._extract_document_content(document_path)
                    current_hash = self._generate_content_hash(
                        current_content['text'], 
                        current_content['metadata']
                    )
                    
                    if current_hash == existing_content.content_hash:
                        self.logger.info(f"Document {document_path.name} already indexed and unchanged")
                        return existing_content
                    else:
                        self.logger.info(f"Document {document_path.name} content changed, reindexing")
            
            # Extract document content using PDF processor
            extraction_start = time.time()
            document_content = await self._extract_document_content(document_path)
            extraction_time = time.time() - extraction_start
            
            # Generate content hash
            content_hash = self._generate_content_hash(
                document_content['text'], 
                document_content['metadata']
            )
            
            # Extract structured content
            indexing_start = time.time()
            structured_content = await self._extract_structured_content(
                document_content['text'],
                document_content['page_texts'],
                document_content['metadata'],
                extract_config or {}
            )
            indexing_time = time.time() - indexing_start
            
            # Create indexed content object
            indexed_content = IndexedContent(
                document_id=document_id,
                document_path=str(document_path),
                content_hash=content_hash,
                numeric_facts=structured_content['numeric_facts'],
                scientific_statements=structured_content['scientific_statements'],
                methodological_info=structured_content['methodological_info'],
                extraction_metadata={
                    **document_content['metadata'],
                    'extraction_time': extraction_time,
                    'indexing_time': indexing_time,
                    'total_facts': len(structured_content['numeric_facts']),
                    'total_statements': len(structured_content['scientific_statements']),
                    'total_methods': len(structured_content['methodological_info']),
                    'extraction_config': extract_config or {}
                }
            )
            
            # Store indexed content in database
            await self._store_indexed_content(indexed_content)
            
            self.logger.info(
                f"Successfully indexed document {document_path.name}: "
                f"{len(indexed_content.numeric_facts)} numeric facts, "
                f"{len(indexed_content.scientific_statements)} statements, "
                f"{len(indexed_content.methodological_info)} methods"
            )
            
            return indexed_content
            
        except Exception as e:
            self.logger.error(f"Failed to index document {document_path}: {e}")
            raise IndexingError(f"Document indexing failed: {e}")
    
    async def _extract_document_content(self, document_path: Path) -> Dict[str, Any]:
        """
        Extract content from a PDF document using BiomedicalPDFProcessor.
        
        Args:
            document_path: Path to the PDF document
            
        Returns:
            Dict[str, Any]: Extracted content including text, metadata, and page texts
            
        Raises:
            ContentExtractionError: If content extraction fails
        """
        try:
            # Use the PDF processor to extract content
            result = self.pdf_processor.extract_text_from_pdf(
                document_path, 
                preprocess_text=True
            )
            
            self.logger.debug(
                f"Extracted {len(result['text'])} characters from "
                f"{result['metadata']['pages']} pages in {document_path.name}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content extraction failed for {document_path}: {e}")
            raise ContentExtractionError(f"PDF content extraction failed: {e}")
    
    async def _extract_structured_content(self,
                                        text: str,
                                        page_texts: List[str],
                                        metadata: Dict[str, Any],
                                        extract_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured content (facts, statements, methods) from document text.
        
        Args:
            text: Full document text
            page_texts: List of text from individual pages
            metadata: Document metadata
            extract_config: Configuration for extraction process
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted structured content
        """
        # Merge extraction configuration with defaults
        config = {**self.content_extraction_config, **extract_config}
        
        # Initialize results
        numeric_facts = []
        scientific_statements = []
        methodological_info = []
        
        self.logger.debug("Starting structured content extraction")
        
        try:
            # Extract numeric facts
            numeric_facts = await self._extract_numeric_facts(
                text, page_texts, metadata, config
            )
            
            # Extract scientific statements
            scientific_statements = await self._extract_scientific_statements(
                text, page_texts, metadata, config
            )
            
            # Extract methodological information
            methodological_info = await self._extract_methodological_info(
                text, page_texts, metadata, config
            )
            
            self.logger.debug(
                f"Extracted {len(numeric_facts)} numeric facts, "
                f"{len(scientific_statements)} statements, "
                f"{len(methodological_info)} methods"
            )
            
            return {
                'numeric_facts': numeric_facts,
                'scientific_statements': scientific_statements,
                'methodological_info': methodological_info
            }
            
        except Exception as e:
            self.logger.error(f"Structured content extraction failed: {e}")
            raise ContentExtractionError(f"Structured extraction failed: {e}")
    
    async def _extract_numeric_facts(self,
                                   text: str,
                                   page_texts: List[str],
                                   metadata: Dict[str, Any],
                                   config: Dict[str, Any]) -> List[NumericFact]:
        """
        Extract numeric facts and measurements from document text.
        
        Args:
            text: Full document text
            page_texts: List of page texts for location tracking
            metadata: Document metadata
            config: Extraction configuration
            
        Returns:
            List[NumericFact]: List of extracted numeric facts
        """
        numeric_facts = []
        
        # Process text in chunks to maintain page location context
        for page_num, page_text in enumerate(page_texts):
            if not page_text.strip():
                continue
            
            # Extract basic measurements
            facts = self._extract_basic_measurements(
                page_text, page_num, metadata, config
            )
            numeric_facts.extend(facts)
            
            # Extract range measurements
            facts = self._extract_range_measurements(
                page_text, page_num, metadata, config
            )
            numeric_facts.extend(facts)
            
            # Extract statistical values
            if config.get('enable_statistical_detection', True):
                facts = self._extract_statistical_values(
                    page_text, page_num, metadata, config
                )
                numeric_facts.extend(facts)
        
        # Filter by confidence threshold
        threshold = config.get('numeric_confidence_threshold', 0.7)
        filtered_facts = [f for f in numeric_facts if f.confidence >= threshold]
        
        self.logger.debug(
            f"Extracted {len(numeric_facts)} numeric facts, "
            f"{len(filtered_facts)} above confidence threshold {threshold}"
        )
        
        return filtered_facts
    
    def _extract_basic_measurements(self,
                                  text: str,
                                  page_num: int,
                                  metadata: Dict[str, Any],
                                  config: Dict[str, Any]) -> List[NumericFact]:
        """Extract basic numeric measurements from text."""
        facts = []
        pattern = self._numeric_patterns['basic_measurement']
        
        for match in pattern.finditer(text):
            try:
                variable_name = match.group(1).strip()
                value_str = match.group(2).replace(',', '')
                error_str = match.group(3)
                unit = match.group(4)
                
                # Parse numeric value
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                
                # Parse error margin if present
                error_margin = None
                if error_str:
                    try:
                        error_margin = float(error_str.replace(',', ''))
                    except ValueError:
                        pass
                
                # Extract context around the match
                context = self._extract_context(text, match.start(), match.end(), config)
                
                # Calculate confidence based on various factors
                confidence = self._calculate_numeric_confidence(
                    variable_name, value, unit, context, config
                )
                
                # Normalize units if enabled
                if config.get('enable_unit_normalization', True):
                    unit = self._normalize_unit(unit)
                
                fact = NumericFact(
                    value=value,
                    unit=unit,
                    context=context,
                    confidence=confidence,
                    source_location={
                        'page': page_num,
                        'position': match.start(),
                        'document_metadata': metadata
                    },
                    variable_name=variable_name,
                    error_margin=error_margin
                )
                
                facts.append(fact)
                
            except Exception as e:
                self.logger.debug(f"Failed to parse numeric measurement: {e}")
                continue
        
        return facts
    
    def _extract_range_measurements(self,
                                  text: str,
                                  page_num: int,
                                  metadata: Dict[str, Any],
                                  config: Dict[str, Any]) -> List[NumericFact]:
        """Extract range measurements from text."""
        facts = []
        pattern = self._numeric_patterns['range_measurement']
        
        for match in pattern.finditer(text):
            try:
                variable_name = match.group(1).strip()
                min_value_str = match.group(2).replace(',', '')
                max_value_str = match.group(3).replace(',', '')
                unit = match.group(4)
                
                # Parse numeric values
                try:
                    min_value = float(min_value_str)
                    max_value = float(max_value_str)
                except ValueError:
                    continue
                
                # Use midpoint as the primary value
                value = (min_value + max_value) / 2
                error_margin = (max_value - min_value) / 2
                
                # Extract context
                context = self._extract_context(text, match.start(), match.end(), config)
                
                # Calculate confidence
                confidence = self._calculate_numeric_confidence(
                    variable_name, value, unit, context, config
                )
                
                # Normalize units
                if config.get('enable_unit_normalization', True):
                    unit = self._normalize_unit(unit)
                
                fact = NumericFact(
                    value=value,
                    unit=unit,
                    context=context,
                    confidence=confidence,
                    source_location={
                        'page': page_num,
                        'position': match.start(),
                        'document_metadata': metadata,
                        'range': {'min': min_value, 'max': max_value}
                    },
                    variable_name=variable_name,
                    error_margin=error_margin
                )
                
                facts.append(fact)
                
            except Exception as e:
                self.logger.debug(f"Failed to parse range measurement: {e}")
                continue
        
        return facts
    
    def _extract_statistical_values(self,
                                  text: str,
                                  page_num: int,
                                  metadata: Dict[str, Any],
                                  config: Dict[str, Any]) -> List[NumericFact]:
        """Extract statistical values like p-values, correlations, etc."""
        facts = []
        pattern = self._numeric_patterns['statistical_value']
        
        for match in pattern.finditer(text):
            try:
                stat_type = match.group(1).strip().lower()
                value_str = match.group(2)
                
                # Parse numeric value
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                
                # Extract context
                context = self._extract_context(text, match.start(), match.end(), config)
                
                # Higher confidence for well-formatted statistical values
                confidence = 0.85 if 'p' in stat_type else 0.75
                
                # Adjust confidence based on value reasonableness
                if stat_type.startswith('p') and (value > 1.0 or value < 0.0):
                    confidence *= 0.5  # Invalid p-value range
                elif 'correlation' in stat_type and abs(value) > 1.0:
                    confidence *= 0.5  # Invalid correlation range
                
                fact = NumericFact(
                    value=value,
                    unit=None,  # Statistical values typically don't have units
                    context=context,
                    confidence=confidence,
                    source_location={
                        'page': page_num,
                        'position': match.start(),
                        'document_metadata': metadata
                    },
                    variable_name=stat_type,
                    method='statistical_analysis'
                )
                
                facts.append(fact)
                
            except Exception as e:
                self.logger.debug(f"Failed to parse statistical value: {e}")
                continue
        
        return facts
    
    async def _extract_scientific_statements(self,
                                           text: str,
                                           page_texts: List[str],
                                           metadata: Dict[str, Any],
                                           config: Dict[str, Any]) -> List[ScientificStatement]:
        """
        Extract scientific statements and relationships from document text.
        
        Args:
            text: Full document text
            page_texts: List of page texts
            metadata: Document metadata
            config: Extraction configuration
            
        Returns:
            List[ScientificStatement]: List of extracted scientific statements
        """
        statements = []
        
        # Process each page for location tracking
        for page_num, page_text in enumerate(page_texts):
            if not page_text.strip():
                continue
            
            # Extract different types of statements
            for pattern_name, pattern in self._statement_patterns.items():
                page_statements = self._extract_statements_by_pattern(
                    page_text, pattern, pattern_name, page_num, metadata, config
                )
                statements.extend(page_statements)
        
        # Filter by confidence threshold
        threshold = config.get('statement_confidence_threshold', 0.6)
        filtered_statements = [s for s in statements if s.confidence >= threshold]
        
        self.logger.debug(
            f"Extracted {len(statements)} statements, "
            f"{len(filtered_statements)} above confidence threshold {threshold}"
        )
        
        return filtered_statements
    
    def _extract_statements_by_pattern(self,
                                     text: str,
                                     pattern: re.Pattern,
                                     pattern_type: str,
                                     page_num: int,
                                     metadata: Dict[str, Any],
                                     config: Dict[str, Any]) -> List[ScientificStatement]:
        """Extract statements matching a specific pattern."""
        statements = []
        
        for match in pattern.finditer(text):
            try:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                obj = match.group(3).strip() if match.lastindex >= 3 else ""
                
                # Extract context
                context = self._extract_context(text, match.start(), match.end(), config)
                
                # Calculate confidence based on pattern type and context
                confidence = self._calculate_statement_confidence(
                    subject, predicate, obj, context, pattern_type, config
                )
                
                # Extract evidence strength indicators
                evidence_strength = self._detect_evidence_strength(context)
                study_type = self._detect_study_type(context)
                statistical_significance = self._extract_statistical_significance(context)
                
                statement = ScientificStatement(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    context=context,
                    confidence=confidence,
                    source_location={
                        'page': page_num,
                        'position': match.start(),
                        'document_metadata': metadata,
                        'pattern_type': pattern_type
                    },
                    study_type=study_type,
                    evidence_strength=evidence_strength,
                    statistical_significance=statistical_significance
                )
                
                statements.append(statement)
                
            except Exception as e:
                self.logger.debug(f"Failed to parse scientific statement: {e}")
                continue
        
        return statements
    
    async def _extract_methodological_info(self,
                                         text: str,
                                         page_texts: List[str],
                                         metadata: Dict[str, Any],
                                         config: Dict[str, Any]) -> List[MethodologicalInfo]:
        """
        Extract methodological information from document text.
        
        Args:
            text: Full document text
            page_texts: List of page texts
            metadata: Document metadata
            config: Extraction configuration
            
        Returns:
            List[MethodologicalInfo]: List of extracted methodological information
        """
        methods = []
        
        # Process each page
        for page_num, page_text in enumerate(page_texts):
            if not page_text.strip():
                continue
            
            # Extract different types of methods
            for pattern_name, pattern in self._method_patterns.items():
                page_methods = self._extract_methods_by_pattern(
                    page_text, pattern, pattern_name, page_num, metadata, config
                )
                methods.extend(page_methods)
        
        # Filter by confidence threshold
        threshold = config.get('method_confidence_threshold', 0.8)
        filtered_methods = [m for m in methods if m.confidence >= threshold]
        
        self.logger.debug(
            f"Extracted {len(methods)} methods, "
            f"{len(filtered_methods)} above confidence threshold {threshold}"
        )
        
        return filtered_methods
    
    def _extract_methods_by_pattern(self,
                                  text: str,
                                  pattern: re.Pattern,
                                  pattern_type: str,
                                  page_num: int,
                                  metadata: Dict[str, Any],
                                  config: Dict[str, Any]) -> List[MethodologicalInfo]:
        """Extract methods matching a specific pattern."""
        methods = []
        
        for match in pattern.finditer(text):
            try:
                method_name = match.group(0).strip()
                
                # Extract extended context for methods
                extended_context = self._extract_extended_method_context(
                    text, match.start(), match.end(), config
                )
                
                # Parse method details from context
                method_details = self._parse_method_details(
                    method_name, extended_context, pattern_type
                )
                
                # Calculate confidence
                confidence = self._calculate_method_confidence(
                    method_name, extended_context, pattern_type, config
                )
                
                method_info = MethodologicalInfo(
                    method_type=pattern_type,
                    description=method_details['description'],
                    parameters=method_details['parameters'],
                    equipment=method_details['equipment'],
                    validation=method_details.get('validation'),
                    limitations=method_details.get('limitations'),
                    context=extended_context,
                    confidence=confidence,
                    source_location={
                        'page': page_num,
                        'position': match.start(),
                        'document_metadata': metadata,
                        'pattern_type': pattern_type
                    }
                )
                
                methods.append(method_info)
                
            except Exception as e:
                self.logger.debug(f"Failed to parse methodological info: {e}")
                continue
        
        return methods
    
    # Helper methods for content extraction
    
    def _extract_context(self, text: str, start: int, end: int, config: Dict[str, Any]) -> str:
        """Extract context around a match position."""
        max_length = config.get('max_context_length', 500)
        
        # Expand context window
        context_start = max(0, start - max_length // 2)
        context_end = min(len(text), end + max_length // 2)
        
        context = text[context_start:context_end].strip()
        
        # Clean up context
        context = re.sub(r'\s+', ' ', context)
        return context
    
    def _extract_extended_method_context(self, text: str, start: int, end: int, config: Dict[str, Any]) -> str:
        """Extract extended context for method information."""
        max_length = config.get('max_context_length', 500) * 2  # Longer for methods
        
        # Look for sentence boundaries
        context_start = max(0, start - max_length)
        context_end = min(len(text), end + max_length)
        
        # Try to find sentence boundaries
        before_text = text[context_start:start]
        after_text = text[end:context_end]
        
        # Find last sentence start before match
        sentence_start = before_text.rfind('. ')
        if sentence_start != -1:
            context_start = context_start + sentence_start + 2
        
        # Find first sentence end after match
        sentence_end = after_text.find('. ')
        if sentence_end != -1:
            context_end = end + sentence_end + 1
        
        context = text[context_start:context_end].strip()
        context = re.sub(r'\s+', ' ', context)
        return context
    
    def _calculate_numeric_confidence(self, variable_name: str, value: float, unit: str, 
                                    context: str, config: Dict[str, Any]) -> float:
        """Calculate confidence score for numeric facts."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for well-known variables
        known_variables = [
            'glucose', 'insulin', 'cholesterol', 'triglyceride', 'hemoglobin', 
            'creatinine', 'urea', 'albumin', 'bilirubin', 'age', 'weight', 'bmi'
        ]
        
        if any(var in variable_name.lower() for var in known_variables):
            confidence += 0.2
        
        # Boost confidence for recognized units
        if unit and unit.lower() in ['mg/dl', 'mg/l', 'mmol/l', 'μmol/l', 'g/l', 'years', 'kg', 'cm']:
            confidence += 0.15
        
        # Boost confidence for reasonable value ranges
        if self._is_reasonable_value(variable_name, value, unit):
            confidence += 0.1
        
        # Boost confidence for clear context indicators
        context_indicators = ['measured', 'found', 'detected', 'observed', 'showed']
        if any(indicator in context.lower() for indicator in context_indicators):
            confidence += 0.1
        
        # Penalize for suspicious patterns
        if value == 0.0 or value > 1e6:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_statement_confidence(self, subject: str, predicate: str, obj: str, 
                                      context: str, pattern_type: str, config: Dict[str, Any]) -> float:
        """Calculate confidence score for scientific statements."""
        confidence = 0.4  # Base confidence
        
        # Boost confidence based on pattern type
        pattern_confidence = {
            'correlation': 0.7,
            'causation': 0.8,
            'comparison': 0.6,
            'effect': 0.75
        }
        confidence = pattern_confidence.get(pattern_type, confidence)
        
        # Boost for statistical context
        statistical_terms = ['significant', 'p <', 'p=', 'correlation', 'regression', 'analysis']
        if any(term in context.lower() for term in statistical_terms):
            confidence += 0.1
        
        # Boost for study context
        study_terms = ['study', 'trial', 'experiment', 'cohort', 'participants', 'subjects']
        if any(term in context.lower() for term in study_terms):
            confidence += 0.1
        
        # Penalize for vague terms
        vague_terms = ['may', 'might', 'could', 'possibly', 'perhaps']
        if any(term in context.lower() for term in vague_terms):
            confidence -= 0.15
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_method_confidence(self, method_name: str, context: str, 
                                   pattern_type: str, config: Dict[str, Any]) -> float:
        """Calculate confidence score for methodological information."""
        confidence = 0.6  # Base confidence
        
        # Boost for well-established methods
        if pattern_type == 'analytical_method':
            confidence += 0.2
        elif pattern_type == 'statistical_method':
            confidence += 0.15
        
        # Boost for detailed context
        detail_indicators = ['protocol', 'procedure', 'method', 'analysis', 'performed', 'conducted']
        detail_count = sum(1 for indicator in detail_indicators if indicator in context.lower())
        confidence += min(0.2, detail_count * 0.05)
        
        # Boost for parameters or specifications
        if any(char in context for char in ['(', ')', '°C', 'min', 'sec', 'mL', 'μL']):
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _normalize_unit(self, unit: str) -> str:
        """Normalize units to standard forms."""
        unit_mappings = {
            'mg/dl': 'mg/dL',
            'mg/l': 'mg/L',
            'mmol/l': 'mmol/L',
            'umol/l': 'μmol/L',
            'μmol/l': 'μmol/L',
            'g/l': 'g/L',
            'ng/ml': 'ng/mL',
            'μg/ml': 'μg/mL',
            'pg/ml': 'pg/mL'
        }
        
        return unit_mappings.get(unit.lower(), unit)
    
    def _is_reasonable_value(self, variable_name: str, value: float, unit: str) -> bool:
        """Check if a numeric value is reasonable for the given variable."""
        variable_lower = variable_name.lower()
        
        # Define reasonable ranges for common biomedical variables
        reasonable_ranges = {
            'glucose': (50, 500),  # mg/dL
            'cholesterol': (100, 400),  # mg/dL
            'age': (0, 120),  # years
            'weight': (1, 300),  # kg
            'height': (30, 250),  # cm
            'bmi': (10, 60),  # kg/m²
            'hemoglobin': (5, 20),  # g/dL
            'creatinine': (0.3, 10)  # mg/dL
        }
        
        for var_key, (min_val, max_val) in reasonable_ranges.items():
            if var_key in variable_lower:
                return min_val <= value <= max_val
        
        # If no specific range, use general reasonableness checks
        return 0 < value < 1e6
    
    def _detect_evidence_strength(self, context: str) -> Optional[str]:
        """Detect evidence strength indicators in context."""
        strong_indicators = ['highly significant', 'strong evidence', 'clearly demonstrated', 'definitively']
        moderate_indicators = ['significant', 'evidence suggests', 'indicates', 'associated with']
        weak_indicators = ['may suggest', 'possible', 'potentially', 'preliminary']
        
        context_lower = context.lower()
        
        if any(indicator in context_lower for indicator in strong_indicators):
            return 'strong'
        elif any(indicator in context_lower for indicator in moderate_indicators):
            return 'moderate'
        elif any(indicator in context_lower for indicator in weak_indicators):
            return 'weak'
        
        return None
    
    def _detect_study_type(self, context: str) -> Optional[str]:
        """Detect study type from context."""
        study_types = {
            'randomized controlled trial': 'RCT',
            'clinical trial': 'clinical_trial',
            'cohort study': 'cohort',
            'case-control': 'case_control',
            'cross-sectional': 'cross_sectional',
            'meta-analysis': 'meta_analysis',
            'systematic review': 'systematic_review',
            'observational': 'observational'
        }
        
        context_lower = context.lower()
        for study_phrase, study_type in study_types.items():
            if study_phrase in context_lower:
                return study_type
        
        return None
    
    def _extract_statistical_significance(self, context: str) -> Optional[str]:
        """Extract statistical significance information from context."""
        # Look for p-values
        p_pattern = re.compile(r'p\s*[<>=]\s*(0\.\d+)', re.IGNORECASE)
        match = p_pattern.search(context)
        if match:
            return f"p {match.group(0).split()[1]} {match.group(1)}"
        
        # Look for significance statements
        if 'significant' in context.lower():
            if 'highly significant' in context.lower():
                return 'highly significant'
            elif 'not significant' in context.lower():
                return 'not significant'
            else:
                return 'significant'
        
        return None
    
    def _parse_method_details(self, method_name: str, context: str, pattern_type: str) -> Dict[str, Any]:
        """Parse detailed method information from context."""
        details = {
            'description': method_name,
            'parameters': {},
            'equipment': []
        }
        
        # Extract parameters (numbers with units)
        param_pattern = re.compile(r'(\d+\.?\d*)\s*([A-Za-z°μ/]+)')
        for match in param_pattern.finditer(context):
            value, unit = match.groups()
            details['parameters'][f'parameter_{len(details["parameters"])}'] = {
                'value': value,
                'unit': unit
            }
        
        # Extract equipment/software mentions
        equipment_patterns = [
            r'\b([A-Z][a-z]+ \d+[a-zA-Z]*)\b',  # Model numbers
            r'\b([A-Z][A-Za-z]+ [A-Z][A-Za-z]+)\b',  # Brand names
            r'\(([^)]+)\)',  # Parenthetical information
        ]
        
        for pattern_str in equipment_patterns:
            pattern = re.compile(pattern_str)
            for match in pattern.finditer(context):
                equipment = match.group(1).strip()
                if len(equipment) > 2 and equipment not in details['equipment']:
                    details['equipment'].append(equipment)
        
        # Look for validation information
        validation_indicators = ['validated', 'calibrated', 'quality control', 'qc', 'standard']
        for indicator in validation_indicators:
            if indicator in context.lower():
                details['validation'] = f"Method includes {indicator}"
                break
        
        # Look for limitations
        limitation_indicators = ['limitation', 'limited', 'however', 'but', 'except']
        for indicator in limitation_indicators:
            if indicator in context.lower():
                # Extract sentence containing limitation
                sentences = context.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        details['limitations'] = sentence.strip()
                        break
                break
        
        return details
    
    # Storage and retrieval methods
    
    async def _store_indexed_content(self, indexed_content: IndexedContent) -> None:
        """
        Store indexed content in the database.
        
        Args:
            indexed_content: The indexed content to store
            
        Raises:
            IndexingError: If storage fails
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                # Insert or update document record
                cursor.execute('''
                    INSERT OR REPLACE INTO documents (
                        id, path, content_hash, extraction_metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    indexed_content.document_id,
                    indexed_content.document_path,
                    indexed_content.content_hash,
                    json.dumps(indexed_content.extraction_metadata, default=str),
                    indexed_content.created_at.isoformat(),
                    indexed_content.updated_at.isoformat()
                ))
                
                # Clear existing facts/statements/methods for this document
                cursor.execute("DELETE FROM numeric_facts WHERE document_id = ?", (indexed_content.document_id,))
                cursor.execute("DELETE FROM scientific_statements WHERE document_id = ?", (indexed_content.document_id,))
                cursor.execute("DELETE FROM methodological_info WHERE document_id = ?", (indexed_content.document_id,))
                
                # Insert numeric facts
                for fact in indexed_content.numeric_facts:
                    cursor.execute('''
                        INSERT INTO numeric_facts (
                            document_id, value, unit, context, confidence, source_location,
                            variable_name, method, error_margin
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        indexed_content.document_id,
                        fact.value,
                        fact.unit,
                        fact.context,
                        fact.confidence,
                        json.dumps(fact.source_location, default=str),
                        fact.variable_name,
                        fact.method,
                        fact.error_margin
                    ))
                
                # Insert scientific statements
                for statement in indexed_content.scientific_statements:
                    cursor.execute('''
                        INSERT INTO scientific_statements (
                            document_id, subject, predicate, object, context, confidence,
                            source_location, study_type, evidence_strength, statistical_significance
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        indexed_content.document_id,
                        statement.subject,
                        statement.predicate,
                        statement.object,
                        statement.context,
                        statement.confidence,
                        json.dumps(statement.source_location, default=str),
                        statement.study_type,
                        statement.evidence_strength,
                        statement.statistical_significance
                    ))
                
                # Insert methodological info
                for method in indexed_content.methodological_info:
                    cursor.execute('''
                        INSERT INTO methodological_info (
                            document_id, method_type, description, parameters, equipment,
                            validation, limitations, context, confidence, source_location
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        indexed_content.document_id,
                        method.method_type,
                        method.description,
                        json.dumps(method.parameters, default=str),
                        json.dumps(method.equipment, default=str),
                        method.validation,
                        method.limitations,
                        method.context,
                        method.confidence,
                        json.dumps(method.source_location, default=str)
                    ))
                
                # Commit transaction
                cursor.execute("COMMIT")
                
                self.logger.debug(f"Stored indexed content for document {indexed_content.document_id}")
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
            
            finally:
                conn.close()
                
        except sqlite3.Error as e:
            self.logger.error(f"Database error storing indexed content: {e}")
            raise IndexingError(f"Failed to store indexed content: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error storing indexed content: {e}")
            raise IndexingError(f"Storage operation failed: {e}")
    
    async def _get_existing_indexed_content(self, document_id: str) -> Optional[IndexedContent]:
        """
        Retrieve existing indexed content for a document.
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            Optional[IndexedContent]: Existing indexed content or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get document record
            cursor.execute('''
                SELECT id, path, content_hash, extraction_metadata, created_at, updated_at
                FROM documents WHERE id = ?
            ''', (document_id,))
            
            doc_row = cursor.fetchone()
            if not doc_row:
                return None
            
            doc_id, path, content_hash, metadata_json, created_at, updated_at = doc_row
            
            # Parse metadata
            try:
                extraction_metadata = json.loads(metadata_json) if metadata_json else {}
            except json.JSONDecodeError:
                extraction_metadata = {}
            
            # Get numeric facts
            cursor.execute('''
                SELECT value, unit, context, confidence, source_location, variable_name, method, error_margin
                FROM numeric_facts WHERE document_id = ?
            ''', (document_id,))
            
            numeric_facts = []
            for row in cursor.fetchall():
                value, unit, context, confidence, location_json, variable_name, method, error_margin = row
                try:
                    source_location = json.loads(location_json) if location_json else {}
                except json.JSONDecodeError:
                    source_location = {}
                
                fact = NumericFact(
                    value=value,
                    unit=unit,
                    context=context,
                    confidence=confidence,
                    source_location=source_location,
                    variable_name=variable_name,
                    method=method,
                    error_margin=error_margin
                )
                numeric_facts.append(fact)
            
            # Get scientific statements
            cursor.execute('''
                SELECT subject, predicate, object, context, confidence, source_location,
                       study_type, evidence_strength, statistical_significance
                FROM scientific_statements WHERE document_id = ?
            ''', (document_id,))
            
            scientific_statements = []
            for row in cursor.fetchall():
                subject, predicate, obj, context, confidence, location_json, study_type, evidence_strength, stat_sig = row
                try:
                    source_location = json.loads(location_json) if location_json else {}
                except json.JSONDecodeError:
                    source_location = {}
                
                statement = ScientificStatement(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    context=context,
                    confidence=confidence,
                    source_location=source_location,
                    study_type=study_type,
                    evidence_strength=evidence_strength,
                    statistical_significance=stat_sig
                )
                scientific_statements.append(statement)
            
            # Get methodological info
            cursor.execute('''
                SELECT method_type, description, parameters, equipment, validation,
                       limitations, context, confidence, source_location
                FROM methodological_info WHERE document_id = ?
            ''', (document_id,))
            
            methodological_info = []
            for row in cursor.fetchall():
                method_type, description, params_json, equip_json, validation, limitations, context, confidence, location_json = row
                
                try:
                    parameters = json.loads(params_json) if params_json else {}
                except json.JSONDecodeError:
                    parameters = {}
                
                try:
                    equipment = json.loads(equip_json) if equip_json else []
                except json.JSONDecodeError:
                    equipment = []
                
                try:
                    source_location = json.loads(location_json) if location_json else {}
                except json.JSONDecodeError:
                    source_location = {}
                
                method_info = MethodologicalInfo(
                    method_type=method_type,
                    description=description,
                    parameters=parameters,
                    equipment=equipment,
                    validation=validation,
                    limitations=limitations,
                    context=context,
                    confidence=confidence,
                    source_location=source_location
                )
                methodological_info.append(method_info)
            
            conn.close()
            
            # Create IndexedContent object
            indexed_content = IndexedContent(
                document_id=doc_id,
                document_path=path,
                content_hash=content_hash,
                numeric_facts=numeric_facts,
                scientific_statements=scientific_statements,
                methodological_info=methodological_info,
                extraction_metadata=extraction_metadata,
                created_at=datetime.fromisoformat(created_at),
                updated_at=datetime.fromisoformat(updated_at)
            )
            
            return indexed_content
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error retrieving indexed content: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving indexed content: {e}")
            return None
    
    @performance_logged("claim verification")
    async def verify_claim(self, 
                          claim: str,
                          verification_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify a claim against indexed document content.
        
        This method searches through indexed content to find supporting or contradicting
        evidence for a given claim. It uses multiple search strategies and provides
        a comprehensive verification result with confidence scores.
        
        Args:
            claim: The claim to verify
            verification_config: Optional configuration for verification process
            
        Returns:
            Dict[str, Any]: Verification result containing:
                - 'verification_status': 'supported', 'contradicted', 'insufficient_evidence'
                - 'confidence': Overall confidence score (0.0-1.0)
                - 'supporting_evidence': List of supporting evidence items
                - 'contradicting_evidence': List of contradicting evidence items
                - 'related_facts': Relevant numeric facts
                - 'related_statements': Relevant scientific statements  
                - 'related_methods': Relevant methodological information
                - 'verification_metadata': Metadata about the verification process
                
        Raises:
            ClaimVerificationError: If verification process fails
        """
        try:
            # Merge with default config
            config = {
                'min_confidence_threshold': 0.5,
                'max_results_per_type': 10,
                'enable_fuzzy_matching': True,
                'similarity_threshold': 0.7,
                'enable_semantic_search': True
            }
            if verification_config:
                config.update(verification_config)
            
            self.logger.info(f"Starting claim verification: '{claim[:100]}...'")
            verification_start = time.time()
            
            # Parse the claim to extract key components
            claim_components = await self._parse_claim_components(claim)
            
            # Search for relevant evidence
            search_results = await self._search_relevant_evidence(claim_components, config)
            
            # Analyze evidence for support/contradiction
            evidence_analysis = await self._analyze_evidence(claim, claim_components, search_results, config)
            
            # Calculate overall verification status and confidence
            verification_result = await self._calculate_verification_result(
                claim, evidence_analysis, config
            )
            
            verification_time = time.time() - verification_start
            
            # Add metadata
            verification_result['verification_metadata'] = {
                'claim_length': len(claim),
                'verification_time': verification_time,
                'search_results_count': len(search_results),
                'claim_components': claim_components,
                'config_used': config,
                'verification_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(
                f"Claim verification completed: {verification_result['verification_status']} "
                f"(confidence: {verification_result['confidence']:.2f}) in {verification_time:.2f}s"
            )
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Claim verification failed: {e}")
            raise ClaimVerificationError(f"Verification process failed: {e}")
    
    async def _parse_claim_components(self, claim: str) -> Dict[str, Any]:
        """Parse a claim to extract key components for verification."""
        components = {
            'numeric_values': [],
            'units': [],
            'variables': [],
            'relationships': [],
            'statistical_terms': [],
            'methods': [],
            'keywords': []
        }
        
        # Extract numeric values and units
        numeric_pattern = re.compile(r'(\d+\.?\d*)\s*([a-zA-Zμ%/]+)?')
        for match in numeric_pattern.finditer(claim):
            value_str, unit = match.groups()
            try:
                value = float(value_str)
                components['numeric_values'].append(value)
                if unit:
                    components['units'].append(self._normalize_unit(unit))
            except ValueError:
                pass
        
        # Extract potential variable names (medical/scientific terms)
        medical_terms = [
            'glucose', 'insulin', 'cholesterol', 'triglyceride', 'hemoglobin', 'creatinine',
            'diabetes', 'hypertension', 'metabolite', 'biomarker', 'concentration', 'level'
        ]
        
        claim_lower = claim.lower()
        for term in medical_terms:
            if term in claim_lower:
                components['variables'].append(term)
        
        # Extract relationship indicators
        relationship_terms = [
            'increase', 'decrease', 'correlate', 'associate', 'cause', 'effect',
            'higher', 'lower', 'greater', 'smaller', 'significant'
        ]
        
        for term in relationship_terms:
            if term in claim_lower:
                components['relationships'].append(term)
        
        # Extract statistical terms
        statistical_terms = ['p-value', 'correlation', 'significant', 'confidence interval', 'mean', 'median']
        for term in statistical_terms:
            if term in claim_lower:
                components['statistical_terms'].append(term)
        
        # Extract method mentions
        method_terms = ['hplc', 'lc-ms', 'elisa', 'western blot', 'pcr', 'analysis', 'assay']
        for term in method_terms:
            if term in claim_lower:
                components['methods'].append(term)
        
        # Extract general keywords (words longer than 3 characters)
        words = re.findall(r'\b\w{4,}\b', claim_lower)
        components['keywords'] = list(set(words))
        
        return components
    
    async def _search_relevant_evidence(self, 
                                      claim_components: Dict[str, Any],
                                      config: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Search for relevant evidence based on claim components."""
        search_results = {
            'numeric_facts': [],
            'scientific_statements': [],
            'methodological_info': []
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            max_results = config.get('max_results_per_type', 10)
            
            # Search numeric facts
            if claim_components['numeric_values'] or claim_components['variables']:
                facts = await self._search_numeric_facts(cursor, claim_components, max_results)
                search_results['numeric_facts'] = facts
            
            # Search scientific statements
            if claim_components['relationships'] or claim_components['variables']:
                statements = await self._search_scientific_statements(cursor, claim_components, max_results)
                search_results['scientific_statements'] = statements
            
            # Search methodological info
            if claim_components['methods']:
                methods = await self._search_methodological_info(cursor, claim_components, max_results)
                search_results['methodological_info'] = methods
            
            conn.close()
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error during evidence search: {e}")
        
        return search_results
    
    async def _search_numeric_facts(self, cursor, claim_components: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Search for relevant numeric facts."""
        facts = []
        
        # Search by variable name
        if claim_components['variables']:
            for variable in claim_components['variables']:
                cursor.execute('''
                    SELECT * FROM numeric_facts 
                    WHERE variable_name LIKE ? OR context LIKE ?
                    ORDER BY confidence DESC
                    LIMIT ?
                ''', (f'%{variable}%', f'%{variable}%', max_results))
                
                for row in cursor.fetchall():
                    fact_dict = {
                        'type': 'numeric_fact',
                        'data': self._row_to_numeric_fact(row),
                        'relevance_score': 0.8  # High relevance for variable match
                    }
                    facts.append(fact_dict)
        
        # Search by numeric value proximity
        if claim_components['numeric_values']:
            for value in claim_components['numeric_values']:
                # Search for values within 20% of the claim value
                tolerance = value * 0.2
                cursor.execute('''
                    SELECT * FROM numeric_facts 
                    WHERE value BETWEEN ? AND ?
                    ORDER BY confidence DESC
                    LIMIT ?
                ''', (value - tolerance, value + tolerance, max_results))
                
                for row in cursor.fetchall():
                    fact_dict = {
                        'type': 'numeric_fact',
                        'data': self._row_to_numeric_fact(row),
                        'relevance_score': 0.7  # Moderate relevance for value proximity
                    }
                    facts.append(fact_dict)
        
        # Search by unit
        if claim_components['units']:
            for unit in claim_components['units']:
                cursor.execute('''
                    SELECT * FROM numeric_facts 
                    WHERE unit = ?
                    ORDER BY confidence DESC
                    LIMIT ?
                ''', (unit, max_results))
                
                for row in cursor.fetchall():
                    fact_dict = {
                        'type': 'numeric_fact',
                        'data': self._row_to_numeric_fact(row),
                        'relevance_score': 0.6  # Lower relevance for unit match only
                    }
                    facts.append(fact_dict)
        
        return facts[:max_results]
    
    async def _search_scientific_statements(self, cursor, claim_components: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Search for relevant scientific statements."""
        statements = []
        
        # Search by subject/object
        if claim_components['variables']:
            for variable in claim_components['variables']:
                cursor.execute('''
                    SELECT * FROM scientific_statements 
                    WHERE subject LIKE ? OR object LIKE ? OR context LIKE ?
                    ORDER BY confidence DESC
                    LIMIT ?
                ''', (f'%{variable}%', f'%{variable}%', f'%{variable}%', max_results))
                
                for row in cursor.fetchall():
                    statement_dict = {
                        'type': 'scientific_statement',
                        'data': self._row_to_scientific_statement(row),
                        'relevance_score': 0.8
                    }
                    statements.append(statement_dict)
        
        # Search by relationship type
        if claim_components['relationships']:
            for relationship in claim_components['relationships']:
                cursor.execute('''
                    SELECT * FROM scientific_statements 
                    WHERE predicate LIKE ? OR context LIKE ?
                    ORDER BY confidence DESC
                    LIMIT ?
                ''', (f'%{relationship}%', f'%{relationship}%', max_results))
                
                for row in cursor.fetchall():
                    statement_dict = {
                        'type': 'scientific_statement',
                        'data': self._row_to_scientific_statement(row),
                        'relevance_score': 0.7
                    }
                    statements.append(statement_dict)
        
        return statements[:max_results]
    
    async def _search_methodological_info(self, cursor, claim_components: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Search for relevant methodological information."""
        methods = []
        
        if claim_components['methods']:
            for method in claim_components['methods']:
                cursor.execute('''
                    SELECT * FROM methodological_info 
                    WHERE method_type LIKE ? OR description LIKE ? OR context LIKE ?
                    ORDER BY confidence DESC
                    LIMIT ?
                ''', (f'%{method}%', f'%{method}%', f'%{method}%', max_results))
                
                for row in cursor.fetchall():
                    method_dict = {
                        'type': 'methodological_info',
                        'data': self._row_to_methodological_info(row),
                        'relevance_score': 0.7
                    }
                    methods.append(method_dict)
        
        return methods[:max_results]
    
    def _row_to_numeric_fact(self, row) -> NumericFact:
        """Convert database row to NumericFact object."""
        (_, document_id, value, unit, context, confidence, location_json, 
         variable_name, method, error_margin) = row
        
        try:
            source_location = json.loads(location_json) if location_json else {}
        except json.JSONDecodeError:
            source_location = {}
        
        return NumericFact(
            value=value,
            unit=unit,
            context=context,
            confidence=confidence,
            source_location=source_location,
            variable_name=variable_name,
            method=method,
            error_margin=error_margin
        )
    
    def _row_to_scientific_statement(self, row) -> ScientificStatement:
        """Convert database row to ScientificStatement object."""
        (_, document_id, subject, predicate, obj, context, confidence, location_json,
         study_type, evidence_strength, statistical_significance) = row
        
        try:
            source_location = json.loads(location_json) if location_json else {}
        except json.JSONDecodeError:
            source_location = {}
        
        return ScientificStatement(
            subject=subject,
            predicate=predicate,
            object=obj,
            context=context,
            confidence=confidence,
            source_location=source_location,
            study_type=study_type,
            evidence_strength=evidence_strength,
            statistical_significance=statistical_significance
        )
    
    def _row_to_methodological_info(self, row) -> MethodologicalInfo:
        """Convert database row to MethodologicalInfo object."""
        (_, document_id, method_type, description, params_json, equip_json,
         validation, limitations, context, confidence, location_json) = row
        
        try:
            parameters = json.loads(params_json) if params_json else {}
        except json.JSONDecodeError:
            parameters = {}
        
        try:
            equipment = json.loads(equip_json) if equip_json else []
        except json.JSONDecodeError:
            equipment = []
        
        try:
            source_location = json.loads(location_json) if location_json else {}
        except json.JSONDecodeError:
            source_location = {}
        
        return MethodologicalInfo(
            method_type=method_type,
            description=description,
            parameters=parameters,
            equipment=equipment,
            validation=validation,
            limitations=limitations,
            context=context,
            confidence=confidence,
            source_location=source_location
        )
    
    async def _analyze_evidence(self, claim: str, claim_components: Dict[str, Any], 
                              search_results: Dict[str, List[Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze search results to determine supporting/contradicting evidence."""
        analysis = {
            'supporting_evidence': [],
            'contradicting_evidence': [],
            'neutral_evidence': [],
            'evidence_summary': {}
        }
        
        # Analyze numeric facts
        for fact_item in search_results['numeric_facts']:
            fact = fact_item['data']
            relevance_score = fact_item['relevance_score']
            
            # Check if numeric fact supports or contradicts the claim
            support_status = self._evaluate_numeric_support(claim, claim_components, fact, config)
            
            evidence_item = {
                'type': 'numeric_fact',
                'content': fact,
                'relevance_score': relevance_score,
                'confidence': fact.confidence,
                'support_strength': support_status['strength']
            }
            
            if support_status['type'] == 'supporting':
                analysis['supporting_evidence'].append(evidence_item)
            elif support_status['type'] == 'contradicting':
                analysis['contradicting_evidence'].append(evidence_item)
            else:
                analysis['neutral_evidence'].append(evidence_item)
        
        # Analyze scientific statements
        for statement_item in search_results['scientific_statements']:
            statement = statement_item['data']
            relevance_score = statement_item['relevance_score']
            
            support_status = self._evaluate_statement_support(claim, claim_components, statement, config)
            
            evidence_item = {
                'type': 'scientific_statement',
                'content': statement,
                'relevance_score': relevance_score,
                'confidence': statement.confidence,
                'support_strength': support_status['strength']
            }
            
            if support_status['type'] == 'supporting':
                analysis['supporting_evidence'].append(evidence_item)
            elif support_status['type'] == 'contradicting':
                analysis['contradicting_evidence'].append(evidence_item)
            else:
                analysis['neutral_evidence'].append(evidence_item)
        
        # Analyze methodological info
        for method_item in search_results['methodological_info']:
            method = method_item['data']
            relevance_score = method_item['relevance_score']
            
            evidence_item = {
                'type': 'methodological_info',
                'content': method,
                'relevance_score': relevance_score,
                'confidence': method.confidence,
                'support_strength': 0.5  # Methods are generally neutral
            }
            
            analysis['neutral_evidence'].append(evidence_item)
        
        # Generate evidence summary
        analysis['evidence_summary'] = {
            'total_evidence_items': len(analysis['supporting_evidence']) + len(analysis['contradicting_evidence']) + len(analysis['neutral_evidence']),
            'supporting_count': len(analysis['supporting_evidence']),
            'contradicting_count': len(analysis['contradicting_evidence']),
            'neutral_count': len(analysis['neutral_evidence']),
            'avg_supporting_confidence': self._calculate_average_confidence(analysis['supporting_evidence']),
            'avg_contradicting_confidence': self._calculate_average_confidence(analysis['contradicting_evidence']),
            'strongest_supporting_confidence': self._get_strongest_confidence(analysis['supporting_evidence']),
            'strongest_contradicting_confidence': self._get_strongest_confidence(analysis['contradicting_evidence'])
        }
        
        return analysis
    
    def _evaluate_numeric_support(self, claim: str, claim_components: Dict[str, Any], 
                                fact: NumericFact, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether a numeric fact supports or contradicts the claim."""
        # Check for value alignment
        if claim_components['numeric_values'] and fact.value is not None:
            for claim_value in claim_components['numeric_values']:
                # Calculate percentage difference
                if claim_value != 0:
                    percent_diff = abs(fact.value - claim_value) / claim_value
                    
                    if percent_diff <= 0.1:  # Within 10%
                        return {'type': 'supporting', 'strength': 0.9}
                    elif percent_diff <= 0.3:  # Within 30%
                        return {'type': 'supporting', 'strength': 0.7}
                    elif percent_diff <= 0.5:  # Within 50%
                        return {'type': 'supporting', 'strength': 0.5}
                    else:
                        return {'type': 'contradicting', 'strength': 0.6}
        
        # Check for variable name alignment
        if claim_components['variables'] and fact.variable_name:
            for variable in claim_components['variables']:
                if variable.lower() in fact.variable_name.lower():
                    return {'type': 'supporting', 'strength': 0.6}
        
        # Check context for contradictory statements
        claim_lower = claim.lower()
        context_lower = fact.context.lower()
        
        contradiction_indicators = ['not', 'no', 'absence', 'lack', 'without', 'decreased', 'reduced']
        support_indicators = ['increased', 'elevated', 'higher', 'present', 'detected']
        
        if any(indicator in context_lower for indicator in contradiction_indicators):
            if any(indicator in claim_lower for indicator in support_indicators):
                return {'type': 'contradicting', 'strength': 0.7}
        
        return {'type': 'neutral', 'strength': 0.5}
    
    def _evaluate_statement_support(self, claim: str, claim_components: Dict[str, Any], 
                                  statement: ScientificStatement, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether a scientific statement supports or contradicts the claim."""
        claim_lower = claim.lower()
        
        # Check for subject/object alignment with claim variables
        alignment_score = 0.0
        if claim_components['variables']:
            for variable in claim_components['variables']:
                if variable in statement.subject.lower() or variable in statement.object.lower():
                    alignment_score += 0.3
        
        # Check for relationship alignment
        if claim_components['relationships']:
            for relationship in claim_components['relationships']:
                if relationship in statement.predicate.lower():
                    alignment_score += 0.4
        
        # Analyze statement strength
        strength_multiplier = 1.0
        if statement.evidence_strength == 'strong':
            strength_multiplier = 1.2
        elif statement.evidence_strength == 'weak':
            strength_multiplier = 0.8
        
        # Check for contradiction indicators
        contradiction_terms = ['however', 'but', 'although', 'despite', 'contrary', 'opposite']
        if any(term in statement.context.lower() for term in contradiction_terms):
            if any(term in claim_lower for term in ['increase', 'higher', 'greater']):
                return {'type': 'contradicting', 'strength': min(0.8, alignment_score * strength_multiplier)}
        
        # Determine support type based on alignment
        if alignment_score >= 0.6:
            return {'type': 'supporting', 'strength': min(1.0, alignment_score * strength_multiplier)}
        elif alignment_score >= 0.3:
            return {'type': 'supporting', 'strength': min(0.7, alignment_score * strength_multiplier)}
        else:
            return {'type': 'neutral', 'strength': alignment_score * strength_multiplier}
    
    def _calculate_average_confidence(self, evidence_list: List[Dict[str, Any]]) -> float:
        """Calculate average confidence of evidence items."""
        if not evidence_list:
            return 0.0
        
        total_confidence = sum(item['confidence'] for item in evidence_list)
        return total_confidence / len(evidence_list)
    
    def _get_strongest_confidence(self, evidence_list: List[Dict[str, Any]]) -> float:
        """Get the highest confidence score from evidence items."""
        if not evidence_list:
            return 0.0
        
        return max(item['confidence'] for item in evidence_list)
    
    async def _calculate_verification_result(self, claim: str, evidence_analysis: Dict[str, Any], 
                                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the final verification result based on evidence analysis."""
        supporting_evidence = evidence_analysis['supporting_evidence']
        contradicting_evidence = evidence_analysis['contradicting_evidence']
        evidence_summary = evidence_analysis['evidence_summary']
        
        # Calculate support and contradiction scores
        support_score = 0.0
        contradiction_score = 0.0
        
        for evidence in supporting_evidence:
            weight = evidence['relevance_score'] * evidence['confidence'] * evidence['support_strength']
            support_score += weight
        
        for evidence in contradicting_evidence:
            weight = evidence['relevance_score'] * evidence['confidence'] * evidence['support_strength']
            contradiction_score += weight
        
        # Normalize scores
        total_evidence = len(supporting_evidence) + len(contradicting_evidence)
        if total_evidence > 0:
            support_score /= total_evidence
            contradiction_score /= total_evidence
        
        # Determine verification status
        min_threshold = config.get('min_confidence_threshold', 0.5)
        
        if support_score > contradiction_score and support_score >= min_threshold:
            if support_score >= 0.8:
                verification_status = 'strongly_supported'
            elif support_score >= 0.6:
                verification_status = 'supported'
            else:
                verification_status = 'weakly_supported'
            overall_confidence = support_score
        elif contradiction_score > support_score and contradiction_score >= min_threshold:
            if contradiction_score >= 0.8:
                verification_status = 'strongly_contradicted'
            elif contradiction_score >= 0.6:
                verification_status = 'contradicted'
            else:
                verification_status = 'weakly_contradicted'
            overall_confidence = contradiction_score
        else:
            verification_status = 'insufficient_evidence'
            overall_confidence = max(support_score, contradiction_score)
        
        return {
            'verification_status': verification_status,
            'confidence': overall_confidence,
            'support_score': support_score,
            'contradiction_score': contradiction_score,
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'related_facts': [item for item in evidence_analysis['neutral_evidence'] if item['type'] == 'numeric_fact'],
            'related_statements': [item for item in evidence_analysis['neutral_evidence'] if item['type'] == 'scientific_statement'],
            'related_methods': [item for item in evidence_analysis['neutral_evidence'] if item['type'] == 'methodological_info'],
            'evidence_summary': evidence_summary
        }
    
    # Additional utility methods
    
    async def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Get a list of all indexed documents.
        
        Returns:
            List[Dict[str, Any]]: List of document metadata
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, path, content_hash, extraction_metadata, created_at, updated_at
                FROM documents ORDER BY updated_at DESC
            ''')
            
            documents = []
            for row in cursor.fetchall():
                doc_id, path, content_hash, metadata_json, created_at, updated_at = row
                
                try:
                    extraction_metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError:
                    extraction_metadata = {}
                
                documents.append({
                    'document_id': doc_id,
                    'path': path,
                    'content_hash': content_hash,
                    'extraction_metadata': extraction_metadata,
                    'created_at': created_at,
                    'updated_at': updated_at
                })
            
            conn.close()
            return documents
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error getting indexed documents: {e}")
            return []
    
    async def delete_indexed_document(self, document_id: str) -> bool:
        """
        Delete an indexed document and all its associated content.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                # Delete from all tables (cascading should handle this, but being explicit)
                cursor.execute("DELETE FROM numeric_facts WHERE document_id = ?", (document_id,))
                cursor.execute("DELETE FROM scientific_statements WHERE document_id = ?", (document_id,))
                cursor.execute("DELETE FROM methodological_info WHERE document_id = ?", (document_id,))
                cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                
                # Check if any rows were affected
                if cursor.rowcount > 0:
                    cursor.execute("COMMIT")
                    self.logger.info(f"Successfully deleted document {document_id}")
                    return True
                else:
                    cursor.execute("ROLLBACK")
                    self.logger.warning(f"No document found with ID {document_id}")
                    return False
                    
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
                
            finally:
                conn.close()
                
        except sqlite3.Error as e:
            self.logger.error(f"Database error deleting document: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return False
    
    async def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the document index.
        
        Returns:
            Dict[str, Any]: Index statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Document counts
            cursor.execute("SELECT COUNT(*) FROM documents")
            stats['total_documents'] = cursor.fetchone()[0]
            
            # Content counts
            cursor.execute("SELECT COUNT(*) FROM numeric_facts")
            stats['total_numeric_facts'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM scientific_statements")
            stats['total_scientific_statements'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM methodological_info")
            stats['total_methodological_info'] = cursor.fetchone()[0]
            
            # Average confidence scores
            cursor.execute("SELECT AVG(confidence) FROM numeric_facts")
            result = cursor.fetchone()[0]
            stats['avg_numeric_fact_confidence'] = round(result, 3) if result else 0.0
            
            cursor.execute("SELECT AVG(confidence) FROM scientific_statements")
            result = cursor.fetchone()[0]
            stats['avg_statement_confidence'] = round(result, 3) if result else 0.0
            
            cursor.execute("SELECT AVG(confidence) FROM methodological_info")
            result = cursor.fetchone()[0]
            stats['avg_method_confidence'] = round(result, 3) if result else 0.0
            
            # Database size
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            stats['database_size_bytes'] = page_size * page_count
            stats['database_size_mb'] = round(stats['database_size_bytes'] / 1024 / 1024, 2)
            
            # Most recent update
            cursor.execute("SELECT MAX(updated_at) FROM documents")
            result = cursor.fetchone()[0]
            stats['last_update'] = result if result else None
            
            conn.close()
            
            # Add calculated fields
            stats['total_content_items'] = (
                stats['total_numeric_facts'] + 
                stats['total_scientific_statements'] + 
                stats['total_methodological_info']
            )
            
            if stats['total_documents'] > 0:
                stats['avg_content_per_document'] = round(
                    stats['total_content_items'] / stats['total_documents'], 1
                )
            else:
                stats['avg_content_per_document'] = 0.0
            
            return stats
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error getting statistics: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def close(self) -> None:
        """Close the document indexer and clean up resources."""
        try:
            # Save any unsaved content extractors
            self._save_content_extractors()
            
            self.logger.info("SourceDocumentIndex closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing SourceDocumentIndex: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Note: In async context, this would be __aexit__, but for compatibility
        # we provide both sync and async context managers
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Integration example and utility functions

async def create_integrated_document_indexer(
    index_dir: Union[str, Path] = "./document_index",
    pdf_processor: Optional['BiomedicalPDFProcessor'] = None,
    logger: Optional[logging.Logger] = None
) -> SourceDocumentIndex:
    """
    Factory function to create a SourceDocumentIndex with proper integration.
    
    This function demonstrates the proper way to integrate the document indexer
    with existing systems in the Clinical Metabolomics Oracle project.
    
    Args:
        index_dir: Directory for storing index files
        pdf_processor: Optional existing BiomedicalPDFProcessor instance
        logger: Optional logger instance
        
    Returns:
        SourceDocumentIndex: Initialized and ready-to-use document indexer
        
    Example:
        # Basic usage
        indexer = await create_integrated_document_indexer()
        
        # With custom configuration
        custom_processor = BiomedicalPDFProcessor(
            processing_timeout=600,
            memory_limit_mb=2048
        )
        indexer = await create_integrated_document_indexer(
            index_dir="./my_custom_index",
            pdf_processor=custom_processor
        )
        
        # Index documents and verify claims
        indexed_content = await indexer.index_document("path/to/document.pdf")
        verification = await indexer.verify_claim("Glucose levels were 150 mg/dL")
    """
    # Create logger if not provided
    if logger is None:
        logger = logging.getLogger("document_indexer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    # Create PDF processor if not provided
    if pdf_processor is None:
        # Import here to avoid circular imports
        from .pdf_processor import BiomedicalPDFProcessor
        pdf_processor = BiomedicalPDFProcessor(
            logger=logger,
            processing_timeout=300,  # 5 minutes
            memory_limit_mb=1024,    # 1GB
            max_page_text_size=1000000  # 1MB per page
        )
        logger.info("Created integrated BiomedicalPDFProcessor")
    
    # Configure content extraction for biomedical documents
    content_extraction_config = {
        'numeric_confidence_threshold': 0.7,
        'statement_confidence_threshold': 0.6,
        'method_confidence_threshold': 0.8,
        'max_context_length': 500,
        'enable_statistical_detection': True,
        'enable_unit_normalization': True
    }
    
    # Create and initialize the indexer
    indexer = SourceDocumentIndex(
        index_dir=index_dir,
        logger=logger,
        pdf_processor=pdf_processor,
        enable_full_text_search=True,
        content_extraction_config=content_extraction_config
    )
    
    # Initialize the indexer
    await indexer.initialize()
    
    logger.info("SourceDocumentIndex created and initialized successfully")
    return indexer


async def demonstrate_indexer_workflow():
    """
    Demonstration of the complete document indexing and claim verification workflow.
    
    This function shows how to use the SourceDocumentIndex in practice,
    including document indexing, claim verification, and result interpretation.
    """
    print("Clinical Metabolomics Oracle - Document Indexing Workflow Demonstration")
    print("=" * 70)
    
    try:
        # Create the indexer
        print("1. Creating integrated document indexer...")
        async with create_integrated_document_indexer() as indexer:
            
            # Get initial statistics
            print("2. Getting initial index statistics...")
            stats = await indexer.get_index_statistics()
            print(f"   - Total documents: {stats['total_documents']}")
            print(f"   - Total content items: {stats['total_content_items']}")
            print(f"   - Database size: {stats['database_size_mb']} MB")
            
            # Example document indexing (if documents exist)
            print("\n3. Document indexing example...")
            documents = await indexer.get_indexed_documents()
            
            if documents:
                print(f"   Found {len(documents)} existing indexed documents:")
                for doc in documents[:3]:  # Show first 3
                    metadata = doc.get('extraction_metadata', {})
                    print(f"   - {doc['path']}: {metadata.get('total_facts', 0)} facts, "
                          f"{metadata.get('total_statements', 0)} statements")
            else:
                print("   No documents currently indexed.")
                print("   To index documents, place PDF files in the 'papers/' directory")
                print("   and call: await indexer.index_document('path/to/document.pdf')")
            
            # Example claim verification
            print("\n4. Claim verification examples...")
            
            example_claims = [
                "Glucose levels were 150 mg/dL in diabetic patients",
                "Cholesterol is significantly correlated with cardiovascular disease risk",
                "LC-MS analysis was performed using a Waters instrument",
                "The study included 100 participants with diabetes"
            ]
            
            for i, claim in enumerate(example_claims, 1):
                print(f"\n   Example {i}: '{claim}'")
                
                verification_result = await indexer.verify_claim(
                    claim,
                    verification_config={
                        'min_confidence_threshold': 0.5,
                        'max_results_per_type': 5
                    }
                )
                
                print(f"   Status: {verification_result['verification_status']}")
                print(f"   Confidence: {verification_result['confidence']:.2f}")
                print(f"   Supporting evidence: {len(verification_result['supporting_evidence'])} items")
                print(f"   Contradicting evidence: {len(verification_result['contradicting_evidence'])} items")
                
                if verification_result['supporting_evidence']:
                    evidence = verification_result['supporting_evidence'][0]
                    print(f"   Top supporting evidence: {evidence['type']} "
                          f"(confidence: {evidence['confidence']:.2f})")
            
            # Final statistics
            print("\n5. Final index statistics...")
            final_stats = await indexer.get_index_statistics()
            print(f"   - Total documents: {final_stats['total_documents']}")
            print(f"   - Total numeric facts: {final_stats['total_numeric_facts']}")
            print(f"   - Total statements: {final_stats['total_scientific_statements']}")
            print(f"   - Total methods: {final_stats['total_methodological_info']}")
            print(f"   - Average content per document: {final_stats['avg_content_per_document']}")
            
        print("\nWorkflow demonstration completed successfully!")
        
    except Exception as e:
        print(f"Error during workflow demonstration: {e}")
        import traceback
        traceback.print_exc()


# Integration helper functions for LightRAG compatibility

def format_indexed_content_for_lightrag(indexed_content: IndexedContent) -> Dict[str, Any]:
    """
    Format indexed content for integration with LightRAG storage systems.
    
    This function converts the structured indexed content into a format
    that can be easily integrated with LightRAG's document storage and
    retrieval mechanisms.
    
    Args:
        indexed_content: The indexed content to format
        
    Returns:
        Dict[str, Any]: Formatted content for LightRAG integration
    """
    formatted_content = {
        'document_metadata': {
            'document_id': indexed_content.document_id,
            'document_path': indexed_content.document_path,
            'content_hash': indexed_content.content_hash,
            'created_at': indexed_content.created_at.isoformat(),
            'updated_at': indexed_content.updated_at.isoformat(),
            'extraction_metadata': indexed_content.extraction_metadata
        },
        'structured_content': {
            'numeric_facts': [asdict(fact) for fact in indexed_content.numeric_facts],
            'scientific_statements': [asdict(stmt) for stmt in indexed_content.scientific_statements],
            'methodological_info': [asdict(method) for method in indexed_content.methodological_info]
        },
        'content_summary': {
            'total_facts': len(indexed_content.numeric_facts),
            'total_statements': len(indexed_content.scientific_statements),
            'total_methods': len(indexed_content.methodological_info),
            'high_confidence_facts': len([f for f in indexed_content.numeric_facts if f.confidence >= 0.8]),
            'high_confidence_statements': len([s for s in indexed_content.scientific_statements if s.confidence >= 0.8])
        }
    }
    
    return formatted_content


def extract_entities_for_lightrag(indexed_content: IndexedContent) -> List[Dict[str, Any]]:
    """
    Extract entities from indexed content for LightRAG entity recognition.
    
    This function creates a list of entities that can be used to enhance
    LightRAG's entity recognition and relationship extraction capabilities.
    
    Args:
        indexed_content: The indexed content to extract entities from
        
    Returns:
        List[Dict[str, Any]]: List of entities for LightRAG
    """
    entities = []
    
    # Extract entities from numeric facts
    for fact in indexed_content.numeric_facts:
        if fact.variable_name:
            entities.append({
                'name': fact.variable_name,
                'type': 'biomedical_variable',
                'value': fact.value,
                'unit': fact.unit,
                'confidence': fact.confidence,
                'context': fact.context[:200],  # Truncate for brevity
                'source_document': indexed_content.document_id
            })
    
    # Extract entities from scientific statements
    for statement in indexed_content.scientific_statements:
        # Add subject entity
        entities.append({
            'name': statement.subject,
            'type': 'biomedical_entity',
            'confidence': statement.confidence,
            'context': statement.context[:200],
            'source_document': indexed_content.document_id,
            'relationship_role': 'subject'
        })
        
        # Add object entity
        if statement.object:
            entities.append({
                'name': statement.object,
                'type': 'biomedical_entity',
                'confidence': statement.confidence,
                'context': statement.context[:200],
                'source_document': indexed_content.document_id,
                'relationship_role': 'object'
            })
    
    # Extract method entities
    for method in indexed_content.methodological_info:
        entities.append({
            'name': method.description[:50],  # Use first part of description as name
            'type': 'analytical_method',
            'method_type': method.method_type,
            'confidence': method.confidence,
            'equipment': method.equipment,
            'parameters': method.parameters,
            'source_document': indexed_content.document_id
        })
    
    return entities


def create_relationships_for_lightrag(indexed_content: IndexedContent) -> List[Dict[str, Any]]:
    """
    Create relationship data from indexed content for LightRAG relationship extraction.
    
    This function extracts relationships that can enhance LightRAG's
    understanding of connections between biomedical entities.
    
    Args:
        indexed_content: The indexed content to extract relationships from
        
    Returns:
        List[Dict[str, Any]]: List of relationships for LightRAG
    """
    relationships = []
    
    # Create relationships from scientific statements
    for statement in indexed_content.scientific_statements:
        relationships.append({
            'source': statement.subject,
            'relation': statement.predicate,
            'target': statement.object,
            'confidence': statement.confidence,
            'evidence_strength': statement.evidence_strength,
            'study_type': statement.study_type,
            'statistical_significance': statement.statistical_significance,
            'context': statement.context,
            'source_document': indexed_content.document_id,
            'source_location': statement.source_location
        })
    
    # Create implicit relationships from numeric facts
    for fact in indexed_content.numeric_facts:
        if fact.variable_name and fact.method:
            relationships.append({
                'source': fact.variable_name,
                'relation': 'measured_by',
                'target': fact.method,
                'confidence': fact.confidence,
                'measurement_value': fact.value,
                'measurement_unit': fact.unit,
                'context': fact.context,
                'source_document': indexed_content.document_id,
                'source_location': fact.source_location
            })
    
    return relationships


if __name__ == "__main__":
    """
    Example usage of the SourceDocumentIndex system.
    """
    import asyncio
    
    # Run the demonstration workflow
    asyncio.run(demonstrate_indexer_workflow())