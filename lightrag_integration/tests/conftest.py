#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures for API Cost Monitoring Test Suite.

This configuration file provides:
- Shared test fixtures across all test modules
- Common test utilities and helpers
- Test environment setup and teardown
- Coverage configuration integration
- Performance test categorization
- Database and file system isolation

Author: Claude Code (Anthropic)
Created: August 6, 2025
"""

import pytest
import pytest_asyncio
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock
from typing import Dict, Any

# Import core components for fixture creation
from lightrag_integration.cost_persistence import CostPersistence
from lightrag_integration.budget_manager import BudgetManager


# Test Categories
def pytest_configure(config):
    """Configure pytest with custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "concurrent: mark test as testing concurrent operations"
    )
    config.addinivalue_line(
        "markers", "async: mark test as requiring async functionality"
    )
    config.addinivalue_line(
        "markers", "lightrag: mark test as LightRAG integration test"
    )
    config.addinivalue_line(
        "markers", "biomedical: mark test as biomedical-specific functionality"
    )


# Shared Fixtures
@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield Path(db_path)
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration object."""
    config = Mock()
    config.enable_file_logging = False  # Default to disabled for test speed
    config.log_dir = temp_dir / "logs"
    config.log_max_bytes = 1024 * 1024
    config.log_backup_count = 3
    config.api_key = "test-api-key"
    config.log_level = "INFO"
    return config


@pytest.fixture
def cost_persistence(temp_db_path):
    """Create a CostPersistence instance for testing."""
    return CostPersistence(temp_db_path, retention_days=365)


@pytest.fixture
def budget_manager(cost_persistence):
    """Create a BudgetManager instance for testing."""
    return BudgetManager(
        cost_persistence=cost_persistence,
        daily_budget_limit=100.0,
        monthly_budget_limit=3000.0
    )


# Test Utilities
class TestDataBuilder:
    """Builder class for creating consistent test data."""
    
    @staticmethod
    def create_cost_record_data(
        operation_type: str = "test_operation",
        model_name: str = "gpt-4o-mini",
        cost_usd: float = 0.05,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """Create cost record data for testing."""
        return {
            'operation_type': operation_type,
            'model_name': model_name,
            'cost_usd': cost_usd,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            **kwargs
        }
    
    @staticmethod
    def create_budget_alert_data(
        alert_level: str = "warning",
        current_cost: float = 75.0,
        budget_limit: float = 100.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Create budget alert data for testing."""
        return {
            'alert_level': alert_level,
            'current_cost': current_cost,
            'budget_limit': budget_limit,
            'percentage_used': (current_cost / budget_limit) * 100,
            **kwargs
        }


@pytest.fixture
def test_data_builder():
    """Provide test data builder utility."""
    return TestDataBuilder()


# =====================================================================
# ASYNC TESTING FIXTURES AND EVENT LOOP CONFIGURATION
# =====================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Configure event loop policy for async testing."""
    import asyncio
    
    # Use the default event loop policy
    policy = asyncio.get_event_loop_policy()
    return policy


@pytest_asyncio.fixture(scope="function")
async def async_test_context():
    """Provide async test context with proper setup and cleanup."""
    import asyncio
    
    # Create a context for async operations
    context = {
        'start_time': asyncio.get_event_loop().time(),
        'tasks': [],
        'cleanup_callbacks': []
    }
    
    yield context
    
    # Cleanup: cancel any remaining tasks
    for task in context.get('tasks', []):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    # Run cleanup callbacks
    for callback in context.get('cleanup_callbacks', []):
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()
        except Exception:
            pass


@pytest_asyncio.fixture
async def async_mock_lightrag():
    """Provide async mock LightRAG system for testing."""
    from unittest.mock import AsyncMock
    
    mock_lightrag = AsyncMock()
    
    # Configure async methods
    mock_lightrag.ainsert = AsyncMock(return_value={'status': 'success', 'cost': 0.01})
    mock_lightrag.aquery = AsyncMock(return_value="Mock response from LightRAG system")
    mock_lightrag.adelete = AsyncMock(return_value={'status': 'deleted'})
    
    # Configure properties
    mock_lightrag.working_dir = "/tmp/test_lightrag"
    mock_lightrag.cost_accumulated = 0.0
    
    yield mock_lightrag
    
    # Cleanup
    mock_lightrag.reset_mock()


@pytest_asyncio.fixture
async def async_cost_tracker():
    """Provide async cost tracking for testing."""
    import asyncio
    
    class AsyncCostTracker:
        def __init__(self):
            self.costs = []
            self.total = 0.0
            self._lock = asyncio.Lock()
        
        async def track_cost(self, operation: str, cost: float, **kwargs):
            """Track cost asynchronously."""
            async with self._lock:
                record = {
                    'operation': operation,
                    'cost': cost,
                    'timestamp': asyncio.get_event_loop().time(),
                    **kwargs
                }
                self.costs.append(record)
                self.total += cost
                return record
        
        async def get_total(self) -> float:
            """Get total cost."""
            async with self._lock:
                return self.total
        
        async def get_costs(self):
            """Get all cost records."""
            async with self._lock:
                return self.costs.copy()
        
        async def reset(self):
            """Reset cost tracking."""
            async with self._lock:
                self.costs.clear()
                self.total = 0.0
    
    tracker = AsyncCostTracker()
    yield tracker
    await tracker.reset()


@pytest_asyncio.fixture
async def async_progress_monitor():
    """Provide async progress monitoring for testing."""
    import asyncio
    
    class AsyncProgressMonitor:
        def __init__(self):
            self.progress = 0.0
            self.status = "initialized"
            self.events = []
            self.start_time = asyncio.get_event_loop().time()
            self._lock = asyncio.Lock()
        
        async def update(self, progress: float, status: str = None, **kwargs):
            """Update progress asynchronously."""
            async with self._lock:
                self.progress = progress
                if status:
                    self.status = status
                
                event = {
                    'timestamp': asyncio.get_event_loop().time(),
                    'progress': progress,
                    'status': status or self.status,
                    **kwargs
                }
                self.events.append(event)
                return event
        
        async def get_summary(self):
            """Get progress summary."""
            async with self._lock:
                return {
                    'current_progress': self.progress,
                    'current_status': self.status,
                    'elapsed_time': asyncio.get_event_loop().time() - self.start_time,
                    'total_events': len(self.events)
                }
        
        async def wait_for_completion(self, timeout: float = 10.0):
            """Wait for progress to reach 100%."""
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < timeout:
                async with self._lock:
                    if self.progress >= 100.0:
                        return True
                await asyncio.sleep(0.1)
            return False
    
    monitor = AsyncProgressMonitor()
    yield monitor


@pytest.fixture
def async_timeout():
    """Configure timeout for async tests."""
    return 30.0  # 30 second timeout for async tests


# Performance Test Configuration
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        'min_operations_per_second': 10,
        'max_response_time_ms': 5000,
        'concurrent_workers': 5,
        'operations_per_worker': 20
    }


# Database Isolation
@pytest.fixture(autouse=True)
def isolate_database_operations(monkeypatch):
    """Ensure database operations are isolated between tests."""
    # This fixture automatically runs for every test to ensure isolation
    # Specific isolation is handled by temp_db_path fixture
    pass


# Logging Configuration for Tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for test environment."""
    # Suppress verbose logging during tests unless explicitly requested
    logging.getLogger().setLevel(logging.WARNING)
    
    # Individual test modules can override this by setting specific logger levels
    yield
    
    # Cleanup after tests
    logging.getLogger().setLevel(logging.INFO)


# =====================================================================
# INTEGRATION TEST FIXTURES FOR PDF PROCESSING AND LIGHTRAG
# =====================================================================

import json
import asyncio
import time
import random
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Union, Tuple
from unittest.mock import MagicMock, AsyncMock
import fitz  # PyMuPDF for PDF creation


@dataclass
class PDFTestDocument:
    """Represents a test PDF document with metadata and content."""
    filename: str
    title: str
    authors: List[str]
    journal: str
    year: int
    doi: str
    keywords: List[str]
    content: str
    page_count: int = 1
    file_size_bytes: int = 1024
    processing_time: float = 0.1
    should_fail: bool = False
    failure_type: str = None
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get document metadata as dictionary."""
        return {
            'title': self.title,
            'authors': self.authors,
            'journal': self.journal,
            'year': self.year,
            'doi': self.doi,
            'keywords': self.keywords,
            'page_count': self.page_count,
            'file_size': self.file_size_bytes
        }


@dataclass
class MockLightRAGResponse:
    """Mock response from LightRAG system."""
    content: str
    cost_usd: float = 0.01
    model_used: str = "gpt-4o-mini"
    tokens_used: int = 100
    processing_time: float = 0.5
    entities_extracted: List[str] = field(default_factory=list)
    relationships_found: List[str] = field(default_factory=list)


class BiomedicalPDFGenerator:
    """Generates realistic biomedical PDF test documents."""
    
    # Biomedical content templates
    CONTENT_TEMPLATES = {
        'metabolomics': {
            'title_patterns': [
                "Metabolomic Analysis of {} in {} Patients",
                "{} Metabolomics: Biomarker Discovery in {}",
                "Clinical Metabolomics Study of {} Using {}"
            ],
            'abstract_template': """
            Abstract: This study presents a comprehensive metabolomic analysis of {condition} 
            in a cohort of {patient_count} patients. We employed {technique} to identify and 
            quantify metabolites associated with {outcome}. Key findings include {findings}.
            Statistical analysis was performed using {analysis_method} with p-values < 0.05 
            considered significant. These results suggest {conclusion}.
            """,
            'methods_template': """
            Methods: {sample_type} samples were collected from {patient_count} patients and 
            {control_count} controls. Sample preparation involved {preparation}. Analysis was 
            performed using {instrument} with {separation_method}. Data processing utilized 
            {software} with {statistical_method} for statistical analysis.
            """,
            'results_template': """
            Results: We identified {metabolite_count} significantly altered metabolites 
            (p < {p_value}). Key findings include elevated levels of {elevated_metabolites} 
            and decreased concentrations of {decreased_metabolites}. Pathway analysis revealed 
            enrichment in {pathways}.
            """,
            'variables': {
                'conditions': ['diabetes', 'cardiovascular disease', 'liver disease', 'cancer', 'kidney disease'],
                'techniques': ['LC-MS/MS', 'GC-MS', 'NMR spectroscopy', 'CE-MS', 'HILIC-MS'],
                'outcomes': ['disease progression', 'treatment response', 'biomarker identification'],
                'sample_types': ['plasma', 'serum', 'urine', 'tissue', 'CSF'],
                'instruments': ['Agilent 6550 Q-TOF', 'Thermo Q Exactive', 'Waters Xevo TQ-S', 'Bruker Avance'],
                'pathways': ['glycolysis', 'TCA cycle', 'amino acid metabolism', 'fatty acid oxidation']
            }
        },
        'proteomics': {
            'title_patterns': [
                "Proteomic Profiling of {} in {} Disease",
                "{} Proteomics: Novel Therapeutic Targets in {}",
                "Mass Spectrometry-Based Proteomics of {}"
            ],
            'variables': {
                'conditions': ['Alzheimer\'s disease', 'Parkinson\'s disease', 'multiple sclerosis'],
                'techniques': ['iTRAQ', 'TMT', 'SILAC', 'label-free quantification'],
                'sample_types': ['brain tissue', 'CSF', 'blood', 'cell culture']
            }
        },
        'genomics': {
            'title_patterns': [
                "Genomic Analysis of {} Susceptibility Variants",
                "GWAS Study of {} in {} Population",
                "Whole Exome Sequencing in {} Patients"
            ],
            'variables': {
                'conditions': ['type 2 diabetes', 'hypertension', 'coronary artery disease'],
                'techniques': ['RNA-seq', 'ChIP-seq', 'ATAC-seq', 'single-cell RNA-seq'],
                'populations': ['European', 'Asian', 'African', 'Hispanic']
            }
        }
    }
    
    @classmethod
    def generate_biomedical_content(cls, content_type: str = 'metabolomics', size: str = 'medium') -> str:
        """Generate realistic biomedical content."""
        template = cls.CONTENT_TEMPLATES.get(content_type, cls.CONTENT_TEMPLATES['metabolomics'])
        variables = template['variables']
        
        # Select random variables
        condition = random.choice(variables.get('conditions', ['disease']))
        technique = random.choice(variables.get('techniques', ['LC-MS']))
        sample_type = random.choice(variables.get('sample_types', ['plasma']))
        
        # Generate content sections
        abstract = template['abstract_template'].format(
            condition=condition,
            patient_count=random.randint(50, 500),
            technique=technique,
            outcome="biomarker identification",
            findings="altered metabolite profiles",
            analysis_method="R software",
            conclusion="metabolomic profiling provides valuable insights"
        )
        
        methods = template.get('methods_template', '').format(
            sample_type=sample_type,
            patient_count=random.randint(100, 300),
            control_count=random.randint(30, 100),
            preparation="protein precipitation",
            instrument=random.choice(variables.get('instruments', ['LC-MS'])),
            separation_method="reverse-phase chromatography",
            software="MassHunter",
            statistical_method="t-tests"
        )
        
        results = template.get('results_template', '').format(
            metabolite_count=random.randint(20, 100),
            p_value=0.05,
            elevated_metabolites="glucose, lactate",
            decreased_metabolites="amino acids, fatty acids",
            pathways=", ".join(random.sample(variables.get('pathways', ['metabolism']), 2))
        )
        
        base_content = f"{abstract}\n\n{methods}\n\n{results}"
        
        # Adjust content size
        if size == 'small':
            return base_content[:1000]
        elif size == 'large':
            # Repeat and expand content
            expanded = base_content
            for i in range(3):
                expanded += f"\n\nSection {i+2}: {base_content}"
            return expanded
        else:  # medium
            return base_content
    
    @classmethod
    def create_test_documents(cls, count: int = 5) -> List[PDFTestDocument]:
        """Create a collection of test PDF documents."""
        documents = []
        content_types = ['metabolomics', 'proteomics', 'genomics']
        sizes = ['small', 'medium', 'large']
        
        for i in range(count):
            content_type = random.choice(content_types)
            size = random.choice(sizes)
            
            # Generate realistic metadata
            condition = random.choice(cls.CONTENT_TEMPLATES[content_type]['variables']['conditions'])
            technique = random.choice(cls.CONTENT_TEMPLATES[content_type]['variables']['techniques'])
            
            title = random.choice(cls.CONTENT_TEMPLATES[content_type]['title_patterns']).format(
                technique, condition
            )
            
            doc = PDFTestDocument(
                filename=f"test_paper_{i+1}_{content_type}.pdf",
                title=title,
                authors=[f"Dr. Author{j}" for j in range(1, random.randint(2, 5))],
                journal=f"Journal of {content_type.title()} Research",
                year=random.randint(2020, 2024),
                doi=f"10.1000/test.{2020+i}.{random.randint(100, 999):03d}",
                keywords=[content_type, condition, technique, "biomarkers", "clinical"],
                content=cls.generate_biomedical_content(content_type, size),
                page_count=random.randint(8, 25),
                file_size_bytes=random.randint(1024*100, 1024*1024*5),  # 100KB to 5MB
                processing_time=random.uniform(0.5, 3.0)
            )
            
            documents.append(doc)
        
        return documents


class MockLightRAGSystem:
    """Mock LightRAG system with realistic behavior."""
    
    def __init__(self, working_dir: Path, response_delay: float = 0.1):
        self.working_dir = working_dir
        self.response_delay = response_delay
        self.documents_indexed = []
        self.query_count = 0
        self.cost_accumulated = 0.0
        self.entities_db = {}
        self.relationships_db = {}
        
        # Realistic biomedical entities and relationships
        self.entity_patterns = {
            'METABOLITE': ['glucose', 'lactate', 'pyruvate', 'alanine', 'glutamine', 'TMAO', 'carnitine'],
            'PROTEIN': ['insulin', 'albumin', 'hemoglobin', 'transferrin', 'CRP', 'TNF-alpha'],
            'GENE': ['APOE', 'PPAR', 'CYP2D6', 'MTHFR', 'ACE', 'LDLR'],
            'DISEASE': ['diabetes', 'cardiovascular disease', 'cancer', 'liver disease', 'kidney disease'],
            'PATHWAY': ['glycolysis', 'TCA cycle', 'fatty acid oxidation', 'amino acid metabolism']
        }
        
        self.relationship_patterns = [
            "{entity1} regulates {entity2}",
            "{entity1} is associated with {entity2}",
            "{entity1} increases in {entity2}",
            "{entity1} is a biomarker for {entity2}",
            "{entity1} pathway involves {entity2}"
        ]
    
    async def ainsert(self, documents: Union[str, List[str]]) -> Dict[str, Any]:
        """Mock document insertion."""
        await asyncio.sleep(self.response_delay)
        
        if isinstance(documents, str):
            documents = [documents]
        
        inserted_count = 0
        cost = 0.0
        
        for doc in documents:
            # Simulate processing cost and time
            doc_cost = len(doc) / 1000 * 0.001  # $0.001 per 1K characters
            cost += doc_cost
            
            # Extract mock entities
            entities = self._extract_mock_entities(doc)
            relationships = self._extract_mock_relationships(doc, entities)
            
            self.documents_indexed.append({
                'content': doc[:100] + "..." if len(doc) > 100 else doc,
                'entities': entities,
                'relationships': relationships,
                'cost': doc_cost,
                'timestamp': time.time()
            })
            
            inserted_count += 1
        
        self.cost_accumulated += cost
        
        return {
            'status': 'success',
            'documents_processed': inserted_count,
            'total_cost': cost,
            'entities_extracted': sum(len(doc['entities']) for doc in self.documents_indexed[-inserted_count:]),
            'relationships_found': sum(len(doc['relationships']) for doc in self.documents_indexed[-inserted_count:])
        }
    
    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """Mock query execution."""
        await asyncio.sleep(self.response_delay)
        
        self.query_count += 1
        query_cost = 0.01  # Fixed cost per query
        self.cost_accumulated += query_cost
        
        # Generate realistic response based on query content
        response = self._generate_mock_response(query)
        
        return response
    
    def _extract_mock_entities(self, text: str) -> List[str]:
        """Extract mock entities from text."""
        entities = []
        text_lower = text.lower()
        
        for entity_type, entity_list in self.entity_patterns.items():
            for entity in entity_list:
                if entity.lower() in text_lower:
                    entities.append(f"{entity_type}:{entity}")
        
        return entities[:10]  # Limit to 10 entities
    
    def _extract_mock_relationships(self, text: str, entities: List[str]) -> List[str]:
        """Extract mock relationships from text and entities."""
        relationships = []
        
        if len(entities) >= 2:
            # Create relationships between entities
            for i in range(min(3, len(entities) - 1)):
                entity1 = entities[i].split(':')[1]
                entity2 = entities[i + 1].split(':')[1]
                relationship = random.choice(self.relationship_patterns).format(
                    entity1=entity1, entity2=entity2
                )
                relationships.append(relationship)
        
        return relationships
    
    def _generate_mock_response(self, query: str) -> str:
        """Generate mock response based on query content."""
        query_lower = query.lower()
        
        # Response patterns based on query type
        if any(word in query_lower for word in ['metabolite', 'metabolomics', 'biomarker']):
            return """Based on the metabolomics literature, several key metabolites are associated with this condition. 
            Studies have identified elevated levels of glucose, lactate, and TMAO, while amino acids and fatty acid 
            derivatives show decreased concentrations. These metabolic changes are linked to altered glycolysis and 
            TCA cycle activity. The findings suggest potential therapeutic targets and diagnostic biomarkers."""
        
        elif any(word in query_lower for word in ['protein', 'proteomics']):
            return """Proteomic analysis reveals significant changes in protein expression profiles. 
            Key proteins including insulin, albumin, and inflammatory markers like CRP and TNF-alpha 
            show altered levels. These protein changes are associated with disease progression and 
            provide insights into underlying pathophysiological mechanisms."""
        
        elif any(word in query_lower for word in ['gene', 'genetic', 'genomics']):
            return """Genomic studies have identified several susceptibility variants and gene expression changes. 
            Important genes include APOE, PPAR, and CYP2D6, which are involved in metabolic pathways and 
            drug metabolism. These genetic factors contribute to disease risk and treatment response variability."""
        
        else:
            return """The clinical literature provides extensive evidence supporting the role of multi-omics 
            approaches in understanding complex diseases. Integration of metabolomics, proteomics, and genomics 
            data offers comprehensive insights into disease mechanisms, biomarker discovery, and personalized 
            treatment strategies."""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'documents_indexed': len(self.documents_indexed),
            'queries_processed': self.query_count,
            'total_cost': self.cost_accumulated,
            'entities_extracted': sum(len(doc['entities']) for doc in self.documents_indexed),
            'relationships_found': sum(len(doc['relationships']) for doc in self.documents_indexed),
            'working_dir': str(self.working_dir)
        }


class ErrorInjector:
    """Utility for injecting controlled errors during testing."""
    
    def __init__(self):
        self.injection_rules = {}
        self.call_count = {}
    
    def add_rule(self, target: str, error_type: Exception, 
                 trigger_after: int = 1, probability: float = 1.0):
        """Add error injection rule."""
        self.injection_rules[target] = {
            'error_type': error_type,
            'trigger_after': trigger_after,
            'probability': probability
        }
        self.call_count[target] = 0
    
    def should_inject_error(self, target: str) -> Optional[Exception]:
        """Check if error should be injected."""
        if target not in self.injection_rules:
            return None
        
        self.call_count[target] += 1
        rule = self.injection_rules[target]
        
        if (self.call_count[target] >= rule['trigger_after'] and 
            random.random() < rule['probability']):
            return rule['error_type']
        
        return None


# =====================================================================
# INTEGRATION TEST FIXTURES
# =====================================================================

@pytest.fixture
def pdf_test_documents():
    """Provide realistic PDF test documents."""
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class PDFTestDocument:
        filename: str
        title: str
        authors: List[str]
        journal: str
        year: int
        doi: str
        keywords: List[str]
        content: str
        page_count: int = 1
        file_size_bytes: int = 1024
        should_fail: bool = False
    
    # Create simple test documents
    return [
        PDFTestDocument(
            filename="test_metabolomics_1.pdf",
            title="Clinical Metabolomics Analysis of Diabetes",
            authors=["Dr. Smith", "Dr. Johnson"],
            journal="Journal of Metabolomics",
            year=2023,
            doi="10.1000/test.001",
            keywords=["metabolomics", "diabetes", "biomarkers"],
            content="This study investigates metabolomic profiles in diabetes patients. We analyzed plasma samples from 100 patients and 50 controls using LC-MS techniques. Significant alterations were found in glucose metabolism pathways. Statistical analysis revealed 25 differentially abundant metabolites with p<0.05. These findings suggest potential biomarkers for diabetes progression.",
            page_count=8,
            file_size_bytes=2048
        ),
        PDFTestDocument(
            filename="test_cardiovascular_2.pdf", 
            title="Biomarker Discovery in Heart Disease",
            authors=["Dr. Brown", "Dr. Wilson"],
            journal="Cardiovascular Research",
            year=2023,
            doi="10.1000/test.002",
            keywords=["cardiovascular", "biomarkers", "proteomics"],
            content="Cardiovascular disease remains a leading cause of mortality. This research explores novel protein biomarkers in heart failure patients. Mass spectrometry analysis identified 45 proteins with altered expression. Pathway analysis revealed involvement in cardiac remodeling processes. These results provide insights into disease mechanisms and potential therapeutic targets.",
            page_count=12,
            file_size_bytes=3072
        ),
        PDFTestDocument(
            filename="test_cancer_3.pdf",
            title="Metabolic Reprogramming in Cancer",
            authors=["Dr. Davis", "Dr. Miller"],  
            journal="Cancer Metabolism",
            year=2024,
            doi="10.1000/test.003",
            keywords=["cancer", "metabolism", "oncology"],
            content="Cancer cells exhibit distinct metabolic signatures. We profiled metabolites from tumor and normal tissue samples using GC-MS and LC-MS platforms. Glycolysis and glutamine metabolism showed significant upregulation in cancer samples. These metabolic alterations may serve as diagnostic markers and therapeutic targets for precision oncology approaches.",
            page_count=15,
            file_size_bytes=4096
        )
    ]


@pytest.fixture
def small_pdf_collection(pdf_test_documents):
    """Provide small collection of PDF documents for quick tests."""
    return pdf_test_documents[:2]


@pytest.fixture
def large_pdf_collection(pdf_test_documents):
    """Provide large collection of PDF documents for performance tests."""
    # Replicate test documents to simulate a larger collection
    return pdf_test_documents * 5


@pytest.fixture
def temp_pdf_files(temp_dir, pdf_test_documents):
    """Create actual PDF files for testing."""
    pdf_files = []
    
    for doc in pdf_test_documents:
        # Create simple PDF file using PyMuPDF
        pdf_path = temp_dir / doc.filename
        
        try:
            pdf_doc = fitz.open()  # Create new PDF
            page = pdf_doc.new_page()  # Add page
            
            # Add content to PDF
            text = f"Title: {doc.title}\n\n{doc.content}"
            page.insert_text((50, 50), text, fontsize=11)
            
            # Save PDF
            pdf_doc.save(str(pdf_path))
            pdf_doc.close()
            
            pdf_files.append(pdf_path)
            
        except Exception:
            # Fallback: create text file if PDF creation fails
            pdf_path.write_text(f"Title: {doc.title}\n\n{doc.content}")
            pdf_files.append(pdf_path)
    
    yield pdf_files
    
    # Cleanup
    for pdf_file in pdf_files:
        try:
            pdf_file.unlink()
        except:
            pass


@pytest.fixture
def mock_lightrag_system(temp_dir):
    """Provide mock LightRAG system for integration testing."""
    return MockLightRAGSystem(temp_dir)


@pytest.fixture
def integration_config(temp_dir):
    """Provide configuration for integration testing."""
    from lightrag_integration.config import LightRAGConfig
    
    return LightRAGConfig(
        api_key="test-integration-key",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        working_dir=temp_dir / "lightrag_working",
        max_async=4,
        max_tokens=8192,
        auto_create_dirs=True,
        enable_cost_tracking=True,
        daily_budget_limit=10.0
    )


@pytest.fixture
def mock_pdf_processor():
    """Provide comprehensive mock PDF processor for integration testing."""
    from lightrag_integration.pdf_processor import BiomedicalPDFProcessor
    
    processor = MagicMock(spec=BiomedicalPDFProcessor)
    
    async def mock_process_pdf(pdf_path) -> Dict[str, Any]:
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # Handle both Path and string inputs
        if hasattr(pdf_path, 'name'):
            filename = pdf_path.name.lower()
        else:
            filename = str(pdf_path).lower()
        
        # Generate response based on filename patterns
        if "diabetes" in filename or "metabolomic" in filename:
            content = "This study investigates metabolomic profiles in diabetes patients. We analyzed plasma samples from 100 patients and 50 controls using LC-MS techniques. Significant alterations were found in glucose metabolism pathways."
            title = "Metabolomic Analysis of Diabetes Biomarkers"
        elif "protein" in filename or "proteomics" in filename:
            content = "Proteomic analysis revealed significant differences between disease and control samples. Mass spectrometry identified key protein biomarkers with potential clinical applications."
            title = "Proteomic Profiling in Disease"
        else:
            content = "This biomedical research paper investigates molecular mechanisms underlying disease progression through comprehensive omics approaches."
            title = "Clinical Research Study"
        
        return {
            "text": content,
            "metadata": {
                "title": title,
                "page_count": random.randint(5, 15),
                "file_size": 1024*100  # Just use a fixed size for mocking
            },
            "processing_time": random.uniform(0.5, 2.0),
            "success": True
        }
    
    async def mock_process_batch(pdf_paths: List[Path]) -> Dict[str, Any]:
        results = []
        successful = 0
        failed = 0
        
        for pdf_path in pdf_paths:
            try:
                result = await mock_process_pdf(pdf_path)
                results.append(result)
                successful += 1
            except Exception:
                failed += 1
        
        return {
            "results": results,
            "processed": successful,
            "failed": failed,
            "total_time": len(pdf_paths) * 0.5
        }
    
    processor.process_pdf = AsyncMock(side_effect=mock_process_pdf)
    processor.process_batch_pdfs = AsyncMock(side_effect=mock_process_batch)
    processor.extract_metadata = AsyncMock(return_value={
        "title": "Test Document",
        "authors": ["Dr. Test"],
        "journal": "Test Journal",
        "year": 2024,
        "keywords": ["test", "research"]
    })
    
    return processor


@pytest.fixture
def mock_cost_monitor():
    """Provide mock cost monitoring system."""
    monitor = MagicMock()
    
    monitor.total_cost = 0.0
    monitor.operation_costs = []
    monitor.budget_alerts = []
    
    def track_cost(operation_type: str, cost: float, **kwargs):
        monitor.total_cost += cost
        monitor.operation_costs.append({
            'operation_type': operation_type,
            'cost': cost,
            'timestamp': time.time(),
            **kwargs
        })
        
        # Generate budget alert if cost exceeds threshold
        if monitor.total_cost > 10.0:  # $10 threshold
            monitor.budget_alerts.append({
                'level': 'warning',
                'message': f'Budget threshold exceeded: ${monitor.total_cost:.2f}',
                'timestamp': time.time()
            })
    
    monitor.track_cost = track_cost
    monitor.get_total_cost = lambda: monitor.total_cost
    monitor.get_budget_alerts = lambda: monitor.budget_alerts
    
    return monitor


@pytest.fixture
def mock_progress_tracker():
    """Provide mock progress tracking system."""
    tracker = MagicMock()
    
    tracker.progress = 0.0
    tracker.status = "initialized"
    tracker.events = []
    tracker.start_time = time.time()
    
    def update_progress(progress: float, status: str = None, **kwargs):
        tracker.progress = progress
        if status:
            tracker.status = status
        
        tracker.events.append({
            'timestamp': time.time(),
            'progress': progress,
            'status': status,
            **kwargs
        })
    
    def get_summary():
        return {
            'current_progress': tracker.progress,
            'current_status': tracker.status,
            'elapsed_time': time.time() - tracker.start_time,
            'total_events': len(tracker.events)
        }
    
    tracker.update_progress = update_progress
    tracker.get_summary = get_summary
    tracker.reset = lambda: setattr(tracker, 'events', [])
    
    return tracker


@pytest.fixture
def error_injector():
    """Provide error injection utility for testing failure scenarios."""
    return ErrorInjector()


@pytest.fixture
def integration_test_environment(temp_dir, integration_config, mock_lightrag_system, 
                                mock_pdf_processor, mock_cost_monitor, mock_progress_tracker):
    """Provide complete integration test environment."""
    
    class IntegrationTestEnv:
        def __init__(self):
            self.temp_dir = temp_dir
            self.config = integration_config
            self.lightrag_system = mock_lightrag_system
            self.pdf_processor = mock_pdf_processor
            self.cost_monitor = mock_cost_monitor
            self.progress_tracker = mock_progress_tracker
            
            # Create working directory structure
            self.working_dir = temp_dir / "integration_test"
            self.working_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (self.working_dir / "pdfs").mkdir(exist_ok=True)
            (self.working_dir / "logs").mkdir(exist_ok=True)
            (self.working_dir / "output").mkdir(exist_ok=True)
            
            self.stats = {
                'tests_run': 0,
                'assertions_passed': 0,
                'setup_time': time.time()
            }
        
        def cleanup(self):
            """Clean up test environment."""
            try:
                if self.working_dir.exists():
                    shutil.rmtree(self.working_dir)
            except:
                pass
        
        def create_test_pdf_collection(self, count: int = 5) -> List[Path]:
            """Create test PDF files in the environment."""
            test_docs = BiomedicalPDFGenerator.create_test_documents(count)
            pdf_paths = []
            
            for doc in test_docs:
                pdf_path = self.working_dir / "pdfs" / doc.filename
                
                # Create simple PDF content
                content = f"Title: {doc.title}\nAuthors: {', '.join(doc.authors)}\n\n{doc.content}"
                pdf_path.write_text(content)  # Simple text file for testing
                pdf_paths.append(pdf_path)
            
            return pdf_paths
        
        def get_statistics(self):
            """Get environment statistics."""
            return {
                **self.stats,
                'uptime': time.time() - self.stats['setup_time'],
                'lightrag_stats': self.lightrag_system.get_statistics(),
                'cost_stats': {
                    'total_cost': self.cost_monitor.get_total_cost(),
                    'operations': len(self.cost_monitor.operation_costs)
                },
                'progress_stats': self.progress_tracker.get_summary()
            }
    
    env = IntegrationTestEnv()
    yield env
    env.cleanup()


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring for tests."""
    import time
    from contextlib import asynccontextmanager
    
    class PerformanceMonitor:
        def __init__(self):
            self.operations = []
            
        @asynccontextmanager
        async def monitor_operation(self, operation_name, **kwargs):
            start_time = time.time()
            try:
                yield self
            finally:
                end_time = time.time()
                duration = end_time - start_time
                self.operations.append({
                    'operation': operation_name,
                    'duration': duration,
                    'start_time': start_time,
                    'end_time': end_time,
                    **kwargs
                })
        
        def get_stats(self):
            return {
                'total_operations': len(self.operations),
                'operations': self.operations
            }
    
    return PerformanceMonitor()


@pytest.fixture
def disease_specific_content():
    """Generate disease-specific content for testing."""
    
    def generate_content(disease_type, complexity='medium'):
        """Generate biomedical content for specific diseases."""
        templates = {
            'diabetes': {
                'simple': "Diabetes is a metabolic disorder affecting glucose regulation. Key metabolites include glucose, insulin, and glucagon.",
                'complex': """Type 2 diabetes mellitus represents a complex metabolic disorder characterized by insulin resistance and progressive β-cell dysfunction. Recent metabolomic studies have identified several key biomarkers including elevated branched-chain amino acids (leucine, isoleucine, valine), altered glucose metabolism intermediates, and disrupted lipid profiles. Pathway analysis reveals significant alterations in glycolysis, gluconeogenesis, and fatty acid oxidation. These metabolic signatures provide insights into disease progression and potential therapeutic targets for precision medicine approaches."""
            },
            'cardiovascular': {
                'simple': "Cardiovascular disease affects heart and blood vessels. Key biomarkers include cholesterol, triglycerides, and inflammatory markers.",
                'complex': """Cardiovascular disease encompasses a spectrum of conditions affecting the heart and vascular system, with metabolomic profiling revealing distinct signatures. Lipidomic analysis shows elevated ceramides, altered phospholipid species, and disrupted bile acid metabolism. Protein biomarkers include troponin, BNP, and inflammatory cytokines. Pathway analysis indicates dysfunction in fatty acid oxidation, mitochondrial metabolism, and oxidative stress pathways. These findings support the development of metabolic-based diagnostic panels and targeted therapeutic interventions."""
            },
            'cancer': {
                'simple': "Cancer involves uncontrolled cell growth. Metabolic changes include altered glucose and amino acid metabolism.",
                'complex': """Oncometabolism represents a hallmark of cancer, characterized by fundamental reprogramming of cellular metabolism to support rapid proliferation. Key alterations include enhanced glycolysis (Warburg effect), glutamine addiction, and altered one-carbon metabolism. Metabolomic profiling reveals elevated lactate, altered amino acid profiles, and disrupted TCA cycle intermediates. Pathway analysis indicates activation of mTOR signaling, altered p53-mediated metabolic control, and dysregulated hypoxia-inducible factor (HIF) responses. These metabolic vulnerabilities represent promising targets for cancer therapeutics."""
            }
        }
        
        content = templates.get(disease_type, templates['diabetes']).get(complexity, templates[disease_type]['simple'])
        return content
    
    return generate_content


# =====================================================================
# IMPORT COMPREHENSIVE TEST FIXTURES
# =====================================================================

# Import comprehensive fixtures to make them available to all tests
try:
    from .comprehensive_test_fixtures import *
    from .biomedical_test_fixtures import *
except ImportError as e:
    logging.warning(f"Could not import comprehensive test fixtures: {e}")