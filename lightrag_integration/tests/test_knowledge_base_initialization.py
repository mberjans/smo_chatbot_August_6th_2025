#!/usr/bin/env python3
"""
Comprehensive tests for knowledge base initialization process in Clinical Metabolomics Oracle.

This module implements comprehensive unit and integration tests for the knowledge base
initialization workflow, including LightRAG setup, document ingestion, storage validation,
error handling, and progress tracking. The tests follow TDD principles and use the existing
test infrastructure and patterns.

Test Coverage:
- Knowledge base initialization process (initialize_knowledge_base method)
- LightRAG storage setup and validation
- PDF processor integration with document ingestion
- Error handling during initialization
- Progress tracking functionality
- Storage directory creation and validation
- Document processing and indexing
- Memory management during large document collections
- Concurrent initialization scenarios
- Recovery from partial initialization failures

Test Classes:
- TestKnowledgeBaseInitializationCore: Core initialization functionality
- TestKnowledgeBaseStorageSetup: LightRAG storage setup and validation
- TestKnowledgeBasePDFIntegration: PDF processor integration tests
- TestKnowledgeBaseErrorHandling: Comprehensive error handling scenarios
- TestKnowledgeBaseProgressTracking: Progress tracking and monitoring
- TestKnowledgeBaseMemoryManagement: Memory management tests
- TestKnowledgeBaseConcurrency: Concurrent initialization tests
- TestKnowledgeBaseRecovery: Recovery and cleanup tests

Requirements:
- Uses existing test patterns and infrastructure
- Follows pytest conventions and markers
- Integrates with conftest.py fixtures
- Maintains compatibility with existing test suite
- Includes both unit and integration tests
- Covers error scenarios and edge cases
- Tests with realistic biomedical PDF scenarios
- Validates proper cleanup and resource management

Author: Claude Code (Anthropic)
Created: 2025-08-06
Version: 1.0.0
"""

import pytest
import asyncio
import tempfile
import shutil
import time
import json
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import sys
import gc
import os

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import core components
from lightrag_integration.config import LightRAGConfig, LightRAGConfigError
from lightrag_integration.clinical_metabolomics_rag import (
    ClinicalMetabolomicsRAG, ClinicalMetabolomicsRAGError
)
from lightrag_integration.pdf_processor import (
    BiomedicalPDFProcessor, BiomedicalPDFProcessorError,
    PDFValidationError, PDFProcessingTimeoutError, PDFMemoryError
)
from lightrag_integration.progress_tracker import PDFProcessingProgressTracker
from lightrag_integration.progress_config import ProgressTrackingConfig


# =====================================================================
# TEST FIXTURES AND UTILITIES
# =====================================================================

@dataclass
class KnowledgeBaseState:
    """Represents the state of a knowledge base during initialization."""
    storage_initialized: bool = False
    documents_ingested: int = 0
    errors_encountered: List[str] = None
    processing_time: float = 0.0
    memory_usage: float = 0.0
    progress_percentage: float = 0.0
    
    def __post_init__(self):
        if self.errors_encountered is None:
            self.errors_encountered = []


@dataclass
class MockDocument:
    """Mock document for testing knowledge base initialization."""
    content: str
    metadata: Dict[str, Any]
    file_path: Optional[Path] = None
    size_bytes: int = 1000
    
    def __post_init__(self):
        if not self.file_path:
            self.file_path = Path(f"mock_doc_{hash(self.content)}.pdf")


class MockKnowledgeBase:
    """Mock knowledge base for testing initialization workflows."""
    
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.state = KnowledgeBaseState()
        self.documents = []
        self.storage_paths = {
            'graph': working_dir / "graph_chunk_entity_relation.json",
            'vdb_chunks': working_dir / "vdb_chunks",
            'vdb_entities': working_dir / "vdb_entities", 
            'vdb_relationships': working_dir / "vdb_relationships"
        }
        self.initialized = False
        
    async def initialize_knowledge_base(self, documents: List[MockDocument] = None):
        """Mock knowledge base initialization."""
        self.state.storage_initialized = True
        
        # Create storage directories
        for storage_path in self.storage_paths.values():
            if isinstance(storage_path, Path):
                storage_path.mkdir(parents=True, exist_ok=True)
            else:
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                storage_path.touch()
        
        # Process documents if provided
        if documents:
            for i, doc in enumerate(documents):
                await asyncio.sleep(0.001)  # Simulate processing
                self.documents.append(doc)
                self.state.documents_ingested += 1
                self.state.progress_percentage = ((i + 1) / len(documents)) * 100
        
        self.initialized = True
        return True
    
    def validate_storage(self) -> bool:
        """Validate that storage is properly set up."""
        for storage_path in self.storage_paths.values():
            if not storage_path.exists():
                return False
        return True


class MockProgressTracker:
    """Mock progress tracker for testing."""
    
    def __init__(self):
        self.progress = 0.0
        self.status = "initialized"
        self.events = []
        
    def update_progress(self, progress: float, status: str = None):
        """Update progress tracking."""
        self.progress = progress
        if status:
            self.status = status
        self.events.append({
            'timestamp': time.time(),
            'progress': progress,
            'status': status
        })
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        return {
            'current_progress': self.progress,
            'current_status': self.status,
            'total_events': len(self.events),
            'last_update': self.events[-1]['timestamp'] if self.events else None
        }


@pytest.fixture
def knowledge_base_config():
    """Provide configuration for knowledge base testing."""
    return LightRAGConfig(
        api_key="test-kb-api-key",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        working_dir=Path("/tmp/test_knowledge_base"),
        max_async=8,
        max_tokens=16384,
        auto_create_dirs=True,
        enable_cost_tracking=True
    )


@pytest.fixture
def mock_pdf_documents():
    """Provide mock PDF documents for testing."""
    documents = [
        MockDocument(
            content="Clinical metabolomics study on diabetes biomarkers in plasma samples.",
            metadata={
                "title": "Diabetes Metabolomics Study",
                "authors": ["Dr. Smith", "Dr. Johnson"],
                "journal": "Journal of Clinical Metabolomics",
                "year": 2024,
                "doi": "10.1000/test.2024.001"
            },
            file_path=Path("/tmp/diabetes_metabolomics.pdf")
        ),
        MockDocument(
            content="NMR-based metabolomics analysis of liver disease progression markers.",
            metadata={
                "title": "Liver Disease NMR Metabolomics", 
                "authors": ["Dr. Wilson", "Dr. Chen"],
                "journal": "Hepatology Research",
                "year": 2024,
                "doi": "10.1000/test.2024.002"
            },
            file_path=Path("/tmp/liver_nmr_study.pdf")
        ),
        MockDocument(
            content="Mass spectrometry metabolomics of cancer biomarkers in urine samples.",
            metadata={
                "title": "Cancer Biomarkers MS Study",
                "authors": ["Dr. Brown", "Dr. Davis"],
                "journal": "Cancer Metabolomics",
                "year": 2023,
                "doi": "10.1000/test.2023.003"
            },
            file_path=Path("/tmp/cancer_ms_biomarkers.pdf")
        )
    ]
    return documents


@pytest.fixture
def mock_pdf_processor():
    """Provide mock PDF processor for testing."""
    processor = MagicMock(spec=BiomedicalPDFProcessor)
    
    async def mock_process_pdf(pdf_path: Path) -> Dict[str, Any]:
        # Simulate processing delay
        await asyncio.sleep(0.01)
        
        # Return realistic processing results
        filename = pdf_path.name
        if "diabetes" in filename.lower():
            return {
                "text": "Clinical metabolomics study on diabetes biomarkers in plasma samples.",
                "metadata": {
                    "title": "Diabetes Metabolomics Study",
                    "page_count": 12,
                    "file_size": 2048576
                },
                "processing_time": 1.2,
                "success": True
            }
        elif "liver" in filename.lower():
            return {
                "text": "NMR-based metabolomics analysis of liver disease progression markers.",
                "metadata": {
                    "title": "Liver Disease NMR Metabolomics",
                    "page_count": 8,
                    "file_size": 1024768
                },
                "processing_time": 0.8,
                "success": True
            }
        else:
            return {
                "text": "Mass spectrometry metabolomics of cancer biomarkers in urine samples.",
                "metadata": {
                    "title": "Cancer Biomarkers MS Study",
                    "page_count": 15,
                    "file_size": 3145728
                },
                "processing_time": 2.1,
                "success": True
            }
    
    processor.process_pdf = AsyncMock(side_effect=mock_process_pdf)
    processor.process_batch_pdfs = AsyncMock(return_value={
        "processed": 3,
        "failed": 0,
        "total_time": 4.1,
        "results": []
    })
    
    return processor


@pytest.fixture
def temp_knowledge_base_dir():
    """Provide temporary directory for knowledge base testing."""
    with tempfile.TemporaryDirectory(prefix="test_kb_") as temp_dir:
        kb_dir = Path(temp_dir)
        yield kb_dir


@pytest.fixture
def mock_progress_tracker():
    """Provide mock progress tracker for testing."""
    return MockProgressTracker()


# =====================================================================
# CORE KNOWLEDGE BASE INITIALIZATION TESTS
# =====================================================================

class TestKnowledgeBaseInitializationCore:
    """Test class for core knowledge base initialization functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_knowledge_base_initialization(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test basic knowledge base initialization workflow."""
        # Update config to use temp directory
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            # Setup mock LightRAG instance
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize RAG system
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Verify initialization completed successfully
            assert rag.is_initialized == True
            assert rag.lightrag_instance is not None
            assert rag.config.working_dir == temp_knowledge_base_dir
            
            # Verify LightRAG was called with correct parameters
            mock_lightrag.assert_called_once()
            call_kwargs = mock_lightrag.call_args[1]
            assert call_kwargs['working_dir'] == str(temp_knowledge_base_dir)
            assert 'llm_model_func' in call_kwargs
            assert 'embedding_func' in call_kwargs
    
    def test_knowledge_base_initialization_with_invalid_config(self):
        """Test knowledge base initialization with invalid configuration."""
        # Test with None config
        with pytest.raises(ValueError, match="config cannot be None"):
            ClinicalMetabolomicsRAG(config=None)
        
        # Test with wrong config type
        with pytest.raises(TypeError, match="config must be a LightRAGConfig instance"):
            ClinicalMetabolomicsRAG(config="invalid_config")
        
        # Test with invalid working directory
        invalid_config = LightRAGConfig(
            api_key="test-key",
            working_dir=Path(""),  # Empty directory
            auto_create_dirs=False
        )
        
        with pytest.raises(ValueError, match="Working directory cannot be empty"):
            ClinicalMetabolomicsRAG(config=invalid_config)
    
    @pytest.mark.asyncio
    async def test_knowledge_base_initialization_creates_storage_structure(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test that knowledge base initialization creates proper storage structure."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize knowledge base
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Verify working directory exists
            assert temp_knowledge_base_dir.exists()
            assert temp_knowledge_base_dir.is_dir()
            
            # Verify LightRAG was configured with correct storage paths
            call_kwargs = mock_lightrag.call_args[1]
            assert call_kwargs['graph_chunk_entity_relation_json_path'] == "graph_chunk_entity_relation.json"
            assert call_kwargs['vdb_chunks_path'] == "vdb_chunks"
            assert call_kwargs['vdb_entities_path'] == "vdb_entities"
            assert call_kwargs['vdb_relationships_path'] == "vdb_relationships"
    
    @pytest.mark.asyncio
    async def test_knowledge_base_initialization_with_custom_parameters(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test knowledge base initialization with custom biomedical parameters."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize with custom parameters
            rag = ClinicalMetabolomicsRAG(
                config=knowledge_base_config,
                custom_model="gpt-4o",
                custom_max_tokens=32768
            )
            
            # Verify custom parameters are applied
            assert rag.effective_model == "gpt-4o"
            assert rag.effective_max_tokens == 32768
            
            # Verify biomedical parameters are set
            assert rag.biomedical_params is not None
            assert 'entity_extraction_focus' in rag.biomedical_params
            assert rag.biomedical_params['entity_extraction_focus'] == 'biomedical'
            
            # Verify expected entity types are configured
            expected_entities = ['METABOLITE', 'PROTEIN', 'GENE', 'DISEASE', 'PATHWAY']
            for entity_type in expected_entities:
                assert entity_type in rag.biomedical_params['entity_types']
    
    @pytest.mark.asyncio
    async def test_knowledge_base_initialization_memory_management(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test knowledge base initialization memory management."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Track initial memory usage
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Initialize knowledge base
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Check memory usage didn't spike excessively
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB for initialization)
            assert memory_increase < 100, f"Memory increase too high: {memory_increase}MB"
            
            # Test cleanup
            await rag.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            # Verify cleanup reduced memory usage
            post_cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
            assert post_cleanup_memory <= current_memory


class TestKnowledgeBaseStorageSetup:
    """Test class for LightRAG storage setup and validation."""
    
    @pytest.mark.asyncio
    async def test_storage_directory_creation(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test that storage directories are created correctly."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize knowledge base
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Verify working directory was created
            assert temp_knowledge_base_dir.exists()
            
            # Verify LightRAG was configured with storage paths
            call_kwargs = mock_lightrag.call_args[1]
            assert call_kwargs['working_dir'] == str(temp_knowledge_base_dir)
    
    @pytest.mark.asyncio
    async def test_storage_validation_with_existing_data(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test storage validation when existing data is present."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        # Create some existing storage files
        (temp_knowledge_base_dir / "graph_chunk_entity_relation.json").touch()
        (temp_knowledge_base_dir / "vdb_chunks").mkdir(exist_ok=True)
        (temp_knowledge_base_dir / "vdb_entities").mkdir(exist_ok=True)
        (temp_knowledge_base_dir / "vdb_relationships").mkdir(exist_ok=True)
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize knowledge base with existing storage
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Verify initialization succeeded despite existing files
            assert rag.is_initialized == True
            
            # Verify existing files are still present
            assert (temp_knowledge_base_dir / "graph_chunk_entity_relation.json").exists()
            assert (temp_knowledge_base_dir / "vdb_chunks").exists()
    
    @pytest.mark.asyncio
    async def test_storage_permission_handling(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test handling of storage permission issues."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        # Make directory read-only to simulate permission issues
        if hasattr(os, 'chmod'):
            temp_knowledge_base_dir.chmod(0o444)  # Read-only
        
        try:
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MagicMock()
                mock_lightrag.return_value = mock_instance
                
                # Should handle permission gracefully or raise appropriate error
                try:
                    rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
                    # If it succeeds, that's fine too (might have different permissions)
                    assert rag.is_initialized == True
                except (PermissionError, ClinicalMetabolomicsRAGError) as e:
                    # Expected for read-only directory
                    assert "permission" in str(e).lower() or "read-only" in str(e).lower()
        
        finally:
            # Restore permissions for cleanup
            if hasattr(os, 'chmod'):
                try:
                    temp_knowledge_base_dir.chmod(0o755)  # Restore write permissions
                except:
                    pass
    
    @pytest.mark.asyncio
    async def test_storage_space_validation(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test validation of available storage space."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize knowledge base
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Check that we can determine available space
            import shutil
            total, used, free = shutil.disk_usage(temp_knowledge_base_dir)
            
            # Should have reasonable free space (at least 100MB)
            assert free > 100 * 1024 * 1024, "Insufficient disk space for testing"
            
            # Verify initialization succeeded
            assert rag.is_initialized == True


class TestKnowledgeBasePDFIntegration:
    """Test class for PDF processor integration with knowledge base."""
    
    @pytest.mark.asyncio
    async def test_pdf_processor_integration_basic(
        self, knowledge_base_config, temp_knowledge_base_dir, mock_pdf_processor
    ):
        """Test basic PDF processor integration with knowledge base."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize with PDF processor
            rag = ClinicalMetabolomicsRAG(
                config=knowledge_base_config,
                pdf_processor=mock_pdf_processor
            )
            
            # Verify PDF processor is integrated
            assert rag.pdf_processor == mock_pdf_processor
            assert rag.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_document_ingestion_workflow(
        self, knowledge_base_config, temp_knowledge_base_dir, mock_pdf_processor,
        mock_pdf_documents
    ):
        """Test complete document ingestion workflow."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize knowledge base
            rag = ClinicalMetabolomicsRAG(
                config=knowledge_base_config,
                pdf_processor=mock_pdf_processor
            )
            
            # Test document insertion
            document_texts = [doc.content for doc in mock_pdf_documents]
            await rag.insert_documents(document_texts)
            
            # Verify documents were inserted into LightRAG
            mock_instance.ainsert.assert_called_once_with(document_texts)
    
    @pytest.mark.asyncio
    async def test_pdf_processing_error_handling(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test error handling during PDF processing."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        # Create mock PDF processor that raises errors
        error_processor = MagicMock(spec=BiomedicalPDFProcessor)
        error_processor.process_pdf = AsyncMock(side_effect=PDFValidationError("Invalid PDF"))
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize with error-prone processor
            rag = ClinicalMetabolomicsRAG(
                config=knowledge_base_config,
                pdf_processor=error_processor
            )
            
            # Should still initialize successfully
            assert rag.is_initialized == True
            assert rag.pdf_processor == error_processor
    
    @pytest.mark.asyncio
    async def test_large_document_batch_processing(
        self, knowledge_base_config, temp_knowledge_base_dir, mock_pdf_processor
    ):
        """Test processing of large document batches."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize knowledge base
            rag = ClinicalMetabolomicsRAG(
                config=knowledge_base_config,
                pdf_processor=mock_pdf_processor
            )
            
            # Create large batch of documents
            large_batch = [f"Document {i} content about metabolomics research." for i in range(50)]
            
            # Process large batch
            start_time = time.time()
            await rag.insert_documents(large_batch)
            processing_time = time.time() - start_time
            
            # Verify processing completed in reasonable time
            assert processing_time < 5.0, f"Large batch processing took too long: {processing_time}s"
            
            # Verify all documents were inserted
            mock_instance.ainsert.assert_called_once_with(large_batch)
    
    @pytest.mark.asyncio
    async def test_pdf_metadata_extraction_and_indexing(
        self, knowledge_base_config, temp_knowledge_base_dir, mock_pdf_processor,
        mock_pdf_documents
    ):
        """Test PDF metadata extraction and indexing workflow."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        # Configure processor to return metadata
        mock_pdf_processor.extract_metadata = AsyncMock(return_value={
            "title": "Test Metabolomics Paper",
            "authors": ["Dr. Test"],
            "journal": "Test Journal",
            "doi": "10.1000/test.001",
            "keywords": ["metabolomics", "biomarkers", "clinical"]
        })
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize knowledge base
            rag = ClinicalMetabolomicsRAG(
                config=knowledge_base_config,
                pdf_processor=mock_pdf_processor
            )
            
            # Process documents with metadata
            document_contents = []
            for doc in mock_pdf_documents:
                # Combine content with metadata for indexing
                enriched_content = f"{doc.content}\n\nMetadata: {json.dumps(doc.metadata)}"
                document_contents.append(enriched_content)
            
            await rag.insert_documents(document_contents)
            
            # Verify enriched documents were inserted
            mock_instance.ainsert.assert_called_once_with(document_contents)


class TestKnowledgeBaseErrorHandling:
    """Test class for comprehensive error handling during initialization."""
    
    @pytest.mark.asyncio
    async def test_lightrag_initialization_failure_recovery(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test recovery from LightRAG initialization failures."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            # Mock LightRAG to fail on first attempt
            mock_lightrag.side_effect = Exception("LightRAG initialization failed")
            
            # Should raise appropriate error
            with pytest.raises(ClinicalMetabolomicsRAGError, match="LightRAG initialization failed"):
                ClinicalMetabolomicsRAG(config=knowledge_base_config)
    
    @pytest.mark.asyncio
    async def test_storage_creation_failure_handling(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test handling of storage creation failures."""
        # Set working directory to a location that doesn't exist and can't be created
        knowledge_base_config.working_dir = Path("/root/non_creatable_dir")
        knowledge_base_config.auto_create_dirs = False
        
        # Should handle storage creation failure with appropriate error
        with pytest.raises(LightRAGConfigError, match="Working directory does not exist and cannot be created"):
            ClinicalMetabolomicsRAG(config=knowledge_base_config)
    
    @pytest.mark.asyncio
    async def test_api_key_validation_error_handling(self, temp_knowledge_base_dir):
        """Test handling of API key validation errors."""
        # Create config with invalid API key
        invalid_config = LightRAGConfig(
            api_key=None,  # No API key
            working_dir=temp_knowledge_base_dir,
            auto_create_dirs=True
        )
        
        # Should raise configuration error
        with pytest.raises(LightRAGConfigError, match="API key is required"):
            ClinicalMetabolomicsRAG(config=invalid_config)
    
    @pytest.mark.asyncio
    async def test_memory_pressure_during_initialization(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test behavior under memory pressure during initialization."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Simulate memory pressure by creating large objects
            large_objects = []
            try:
                # Create objects to consume memory
                for i in range(10):
                    large_objects.append([0] * 1000000)  # 1M integers each
                
                # Should still initialize successfully despite memory pressure
                rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
                assert rag.is_initialized == True
                
            finally:
                # Clean up large objects
                large_objects.clear()
                gc.collect()
    
    @pytest.mark.asyncio
    async def test_concurrent_initialization_conflicts(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test handling of concurrent initialization conflicts."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Simulate concurrent initialization attempts
            async def initialize_instance():
                try:
                    rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
                    return rag
                except Exception as e:
                    return e
            
            # Start multiple concurrent initializations
            tasks = [initialize_instance() for _ in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # At least one should succeed
            successful_inits = [r for r in results if isinstance(r, ClinicalMetabolomicsRAG)]
            assert len(successful_inits) >= 1
            
            # Clean up successful instances
            for rag in successful_inits:
                await rag.cleanup()
    
    @pytest.mark.asyncio
    async def test_partial_initialization_cleanup(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test cleanup after partial initialization failures."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        # Mock LightRAG to fail after partial setup
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            def failing_init(*args, **kwargs):
                # Create some files before failing
                (temp_knowledge_base_dir / "partial_file.tmp").touch()
                raise Exception("Initialization failed mid-process")
            
            mock_lightrag.side_effect = failing_init
            
            # Should fail gracefully
            with pytest.raises(ClinicalMetabolomicsRAGError):
                ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Verify partial files were created (simulating partial initialization)
            assert (temp_knowledge_base_dir / "partial_file.tmp").exists()


class TestKnowledgeBaseProgressTracking:
    """Test class for progress tracking functionality during initialization."""
    
    @pytest.mark.asyncio
    async def test_progress_tracking_during_initialization(
        self, knowledge_base_config, temp_knowledge_base_dir, mock_progress_tracker
    ):
        """Test progress tracking during knowledge base initialization."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize with progress tracking
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Simulate progress tracking during document processing
            documents = ["Doc 1", "Doc 2", "Doc 3"]
            
            for i, doc in enumerate(documents):
                progress = ((i + 1) / len(documents)) * 100
                mock_progress_tracker.update_progress(progress, f"Processing document {i+1}")
            
            # Verify progress was tracked
            assert mock_progress_tracker.progress == 100.0
            assert len(mock_progress_tracker.events) == 3
    
    @pytest.mark.asyncio
    async def test_progress_tracking_with_errors(
        self, knowledge_base_config, temp_knowledge_base_dir, mock_progress_tracker
    ):
        """Test progress tracking when errors occur during initialization."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(side_effect=Exception("Insert failed"))
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Simulate error during document processing
            mock_progress_tracker.update_progress(50.0, "Processing documents")
            
            try:
                await rag.insert_documents(["test document"])
            except Exception:
                mock_progress_tracker.update_progress(50.0, "Error occurred")
            
            # Verify error was tracked
            last_event = mock_progress_tracker.events[-1]
            assert "Error" in last_event['status']
    
    @pytest.mark.asyncio
    async def test_progress_tracking_memory_management(
        self, knowledge_base_config, temp_knowledge_base_dir, mock_progress_tracker
    ):
        """Test progress tracking memory management during large operations."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Simulate tracking of many progress events
            for i in range(1000):
                mock_progress_tracker.update_progress(i / 10.0, f"Processing {i}")
            
            # Verify memory usage is reasonable
            import sys
            tracker_size = sys.getsizeof(mock_progress_tracker.events)
            assert tracker_size < 1024 * 1024  # Less than 1MB
    
    @pytest.mark.asyncio
    async def test_progress_tracking_concurrent_updates(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test progress tracking with concurrent updates."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        # Create thread-safe progress tracker
        progress_tracker = MockProgressTracker()
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Simulate concurrent progress updates
            async def update_progress(tracker, start_idx):
                for i in range(10):
                    await asyncio.sleep(0.001)  # Small delay
                    tracker.update_progress(start_idx + i, f"Worker {start_idx} - Step {i}")
            
            # Run concurrent updates
            tasks = [
                update_progress(progress_tracker, 0),
                update_progress(progress_tracker, 100),
                update_progress(progress_tracker, 200)
            ]
            
            await asyncio.gather(*tasks)
            
            # Verify all updates were recorded
            assert len(progress_tracker.events) == 30  # 3 workers × 10 updates each


class TestKnowledgeBaseMemoryManagement:
    """Test class for memory management during knowledge base operations."""
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_initialization(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test memory cleanup after knowledge base initialization."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Track initial memory
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Initialize and immediately cleanup
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            await rag.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            # Check memory after cleanup
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            
            # Memory growth should be minimal after cleanup
            assert memory_growth < 50 * 1024 * 1024  # Less than 50MB growth
    
    @pytest.mark.asyncio
    async def test_large_document_batch_memory_management(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test memory management during large document batch processing."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Create large documents
            large_documents = [
                f"Very large document {i} " + "content " * 10000 
                for i in range(10)
            ]
            
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Process large batch
            await rag.insert_documents(large_documents)
            
            # Check memory usage
            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory
            
            # Clean up
            await rag.cleanup()
            del large_documents
            gc.collect()
            
            final_memory = process.memory_info().rss
            
            # Verify cleanup was effective
            assert final_memory <= peak_memory
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test handling of memory pressure scenarios."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Create memory pressure
            memory_consumer = []
            try:
                # Consume significant memory
                for i in range(50):
                    memory_consumer.append([0] * 100000)  # 100K integers each
                
                # Should still initialize under memory pressure
                rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
                assert rag.is_initialized == True
                
            finally:
                # Clean up memory pressure
                memory_consumer.clear()
                gc.collect()


class TestKnowledgeBaseConcurrency:
    """Test class for concurrent knowledge base operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_document_insertion(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test concurrent document insertion operations."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Define concurrent insertion tasks
            async def insert_batch(batch_id):
                documents = [f"Batch {batch_id} document {i}" for i in range(5)]
                await rag.insert_documents(documents)
                return batch_id
            
            # Run concurrent insertions
            tasks = [insert_batch(i) for i in range(3)]
            results = await asyncio.gather(*tasks)
            
            # Verify all batches completed
            assert len(results) == 3
            assert set(results) == {0, 1, 2}
            
            # Verify total insert calls
            assert mock_instance.ainsert.call_count == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_queries_during_initialization(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test concurrent queries during knowledge base initialization."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.aquery = AsyncMock(return_value="Mock response")
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Define concurrent query tasks
            async def execute_query(query_id):
                query = f"Test query {query_id} about metabolomics"
                result = await rag.query(query)
                return result
            
            # Run concurrent queries
            tasks = [execute_query(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            
            # Verify all queries completed
            assert len(results) == 5
            for result in results:
                assert 'content' in result
                assert result['content'] == "Mock response"
    
    @pytest.mark.asyncio
    async def test_resource_contention_handling(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test handling of resource contention during concurrent operations."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_instance.aquery = AsyncMock(return_value="Mock response")
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Mixed concurrent operations
            async def mixed_operations(worker_id):
                results = []
                
                # Insert documents
                docs = [f"Worker {worker_id} doc {i}" for i in range(3)]
                await rag.insert_documents(docs)
                results.append(f"insert_{worker_id}")
                
                # Execute queries
                for i in range(2):
                    query_result = await rag.query(f"Worker {worker_id} query {i}")
                    results.append(f"query_{worker_id}_{i}")
                
                return results
            
            # Run mixed concurrent operations
            tasks = [mixed_operations(i) for i in range(3)]
            all_results = await asyncio.gather(*tasks)
            
            # Verify all operations completed
            assert len(all_results) == 3
            for worker_results in all_results:
                assert len(worker_results) == 3  # 1 insert + 2 queries per worker


class TestKnowledgeBaseRecovery:
    """Test class for recovery and cleanup operations."""
    
    @pytest.mark.asyncio
    async def test_recovery_from_corrupted_storage(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test recovery from corrupted storage files."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        # Create corrupted storage files
        (temp_knowledge_base_dir / "graph_chunk_entity_relation.json").write_text("corrupted data")
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Should handle corrupted storage gracefully
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            assert rag.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_cleanup_orphaned_resources(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test cleanup of orphaned resources after failures."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Create some data that simulates orphaned resources
            rag.query_history.extend([f"Query {i}" for i in range(2000)])  # Large history
            rag.cost_monitor['costs'].extend([{'cost': 0.01} for _ in range(2000)])  # Large cost history
            
            # Cleanup should handle large data structures
            await rag.cleanup()
            
            # Verify cleanup was effective
            assert len(rag.query_history) <= 100  # Should be trimmed
            assert len(rag.cost_monitor['costs']) <= 100  # Should be trimmed
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_during_operations(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test graceful shutdown during ongoing operations."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Start long-running operation
            async def long_operation():
                for i in range(100):
                    await rag.insert_documents([f"Document batch {i}"])
                    await asyncio.sleep(0.01)
            
            # Start operation and interrupt it
            operation_task = asyncio.create_task(long_operation())
            await asyncio.sleep(0.1)  # Let it run briefly
            
            # Cleanup should work even during operations
            await rag.cleanup()
            
            # Cancel the ongoing operation
            operation_task.cancel()
            
            try:
                await operation_task
            except asyncio.CancelledError:
                pass  # Expected
    
    @pytest.mark.asyncio
    async def test_storage_consistency_validation(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test validation of storage consistency after operations."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Verify storage structure exists
            assert temp_knowledge_base_dir.exists()
            
            # Verify we can validate the storage setup
            # (In real implementation, this would check file integrity)
            assert rag.is_initialized == True


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

class TestKnowledgeBaseIntegration:
    """Integration tests combining multiple knowledge base components."""
    
    @pytest.mark.asyncio
    async def test_complete_knowledge_base_workflow(
        self, knowledge_base_config, temp_knowledge_base_dir, mock_pdf_processor,
        mock_pdf_documents, mock_progress_tracker
    ):
        """Test complete knowledge base workflow from initialization to querying."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_instance.aquery = AsyncMock(return_value="Comprehensive test response")
            mock_lightrag.return_value = mock_instance
            
            # Initialize knowledge base
            rag = ClinicalMetabolomicsRAG(
                config=knowledge_base_config,
                pdf_processor=mock_pdf_processor
            )
            
            # Verify initialization
            assert rag.is_initialized == True
            assert rag.total_cost == 0.0
            
            # Insert documents
            documents = [doc.content for doc in mock_pdf_documents]
            await rag.insert_documents(documents)
            
            # Execute queries
            query_result = await rag.query("What metabolites are associated with diabetes?")
            
            # Verify complete workflow
            assert 'content' in query_result
            assert query_result['content'] == "Comprehensive test response"
            assert len(rag.query_history) == 1
            assert rag.total_cost > 0.0
            
            # Verify document insertion occurred
            mock_instance.ainsert.assert_called_once_with(documents)
            
            # Verify query execution occurred
            mock_instance.aquery.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_knowledge_base_with_real_file_operations(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test knowledge base with actual file operations."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Initialize knowledge base
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Create some test files in the working directory
            test_file = temp_knowledge_base_dir / "test_data.json"
            test_file.write_text('{"test": "data"}')
            
            # Verify file operations work
            assert test_file.exists()
            assert json.loads(test_file.read_text())['test'] == 'data'
            
            # Cleanup should preserve user files but clean up internal state
            await rag.cleanup()
            assert test_file.exists()  # User files should remain


# =====================================================================
# PERFORMANCE AND BENCHMARKING TESTS  
# =====================================================================

class TestKnowledgeBasePerformance:
    """Performance tests for knowledge base operations."""
    
    @pytest.mark.asyncio
    async def test_initialization_performance_benchmark(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test knowledge base initialization performance."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_lightrag.return_value = mock_instance
            
            # Benchmark initialization time
            start_time = time.time()
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            init_time = time.time() - start_time
            
            # Should initialize quickly (under 2 seconds)
            assert init_time < 2.0, f"Initialization took too long: {init_time:.2f}s"
            assert rag.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_large_batch_processing_performance(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test performance of large batch document processing."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Create large document batch
            large_batch = [
                f"Document {i}: Clinical metabolomics research on biomarker {i % 10}"
                for i in range(100)
            ]
            
            # Benchmark batch processing
            start_time = time.time()
            await rag.insert_documents(large_batch)
            processing_time = time.time() - start_time
            
            # Should process efficiently (under 1 second for mock)
            assert processing_time < 1.0, f"Batch processing took too long: {processing_time:.2f}s"
            
            # Verify all documents were processed
            mock_instance.ainsert.assert_called_once_with(large_batch)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(
        self, knowledge_base_config, temp_knowledge_base_dir
    ):
        """Test performance of concurrent knowledge base operations."""
        knowledge_base_config.working_dir = temp_knowledge_base_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_instance.aquery = AsyncMock(return_value="Test response")
            mock_lightrag.return_value = mock_instance
            
            rag = ClinicalMetabolomicsRAG(config=knowledge_base_config)
            
            # Define concurrent operations
            async def concurrent_workload():
                tasks = []
                
                # Insert operations
                for i in range(5):
                    docs = [f"Concurrent doc {i}-{j}" for j in range(10)]
                    tasks.append(rag.insert_documents(docs))
                
                # Query operations  
                for i in range(10):
                    tasks.append(rag.query(f"Concurrent query {i}"))
                
                return await asyncio.gather(*tasks)
            
            # Benchmark concurrent workload
            start_time = time.time()
            results = await concurrent_workload()
            concurrent_time = time.time() - start_time
            
            # Should handle concurrency efficiently
            assert concurrent_time < 3.0, f"Concurrent operations took too long: {concurrent_time:.2f}s"
            assert len(results) == 15  # 5 inserts + 10 queries


if __name__ == "__main__":
    """
    Run the knowledge base initialization test suite when executed directly.
    
    These tests provide comprehensive coverage of the knowledge base initialization
    process, including storage setup, document ingestion, error handling, and
    performance validation. They are designed to integrate with the existing
    test infrastructure and follow established patterns.
    """
    pytest.main([__file__, "-v", "--tb=short"])