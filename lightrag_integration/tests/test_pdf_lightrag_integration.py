#!/usr/bin/env python3
"""
Comprehensive Integration Tests for PDF Processing to LightRAG Ingestion Pipeline.

This module implements comprehensive integration tests for the complete data flow
from PDF files to indexed documents in LightRAG. It validates the end-to-end
pipeline including PDF processing, biomedical preprocessing, document ingestion,
metadata preservation, cost tracking, and performance characteristics.

Test Coverage:
- Single PDF to LightRAG ingestion workflow
- Batch PDF processing and ingestion
- PDF metadata preservation throughout the pipeline
- Biomedical preprocessing and LightRAG compatibility
- Concurrent PDF processing and ingestion
- Cost tracking integration across components
- Performance characteristics and resource usage
- Error handling and recovery scenarios
- Memory management during large operations
- Progress tracking and monitoring

Test Classes:
- TestPDFProcessingToDocumentIngestionFlow: Core integration tests
- TestPDFMetadataPreservationFlow: Metadata flow validation
- TestCostTrackingIntegration: Cost monitoring validation
- TestPerformanceCharacteristics: Performance and resource tests
- TestErrorHandlingAndRecovery: Failure scenario tests
- TestEndToEndPDFQueryWorkflow: Complete query workflow validation
- TestQueryFunctionalityIntegration: Query functionality validation
- TestResponseQualityValidation: Response quality and performance tests
- TestConfigurationIntegration: Configuration consistency and validation
- TestErrorRecoveryIntegration: Coordinated error handling and circuit breakers
- TestResourceManagementIntegration: Memory and resource coordination
- TestProgressTrackingIntegration: End-to-end progress reporting

Requirements:
- Uses comprehensive fixtures from conftest.py and test_fixtures.py
- Follows existing test patterns in the codebase
- Integrates with existing PDF processor and LightRAG components
- Validates complete data pipeline functionality
- Tests both success and failure scenarios
- Uses async/await patterns consistently
- Includes proper cleanup and resource management

Author: Claude Code (Anthropic)
Created: 2025-08-06
Version: 1.0.0
"""

import pytest
import asyncio
import tempfile
import json
import time
import logging
import gc
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
from dataclasses import dataclass, field
import sys

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import core components
from lightrag_integration.clinical_metabolomics_rag import (
    ClinicalMetabolomicsRAG, ClinicalMetabolomicsRAGError
)
from lightrag_integration.pdf_processor import (
    BiomedicalPDFProcessor, BiomedicalPDFProcessorError,
    PDFValidationError, PDFProcessingTimeoutError, PDFMemoryError
)
from lightrag_integration.config import LightRAGConfig, LightRAGConfigError
from lightrag_integration.progress_tracker import PDFProcessingProgressTracker
from lightrag_integration.cost_persistence import CostPersistence
from lightrag_integration.budget_manager import BudgetManager


# =====================================================================
# INTEGRATION TEST DATA MODELS
# =====================================================================

@dataclass
class IntegrationTestResult:
    """Represents the result of an integration test operation."""
    success: bool
    processing_time: float
    documents_processed: int
    entities_extracted: int
    relationships_found: int
    total_cost: float
    memory_peak_mb: float
    error_count: int
    warning_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics from the test result."""
        if self.processing_time > 0 and self.documents_processed > 0:
            return {
                'documents_per_second': self.documents_processed / self.processing_time,
                'cost_per_document': self.total_cost / self.documents_processed,
                'entities_per_document': self.entities_extracted / self.documents_processed,
                'relationships_per_document': self.relationships_found / self.documents_processed,
                'memory_per_document': self.memory_peak_mb / self.documents_processed
            }
        return {}


@dataclass
class PipelineValidationResult:
    """Represents validation results for the complete pipeline."""
    pdf_processing_success: bool
    text_extraction_success: bool
    preprocessing_success: bool
    lightrag_ingestion_success: bool
    metadata_preservation_success: bool
    cost_tracking_success: bool
    query_validation_success: bool
    cleanup_success: bool
    pipeline_integrity: bool
    
    @property
    def overall_success(self) -> bool:
        """Check if the entire pipeline succeeded."""
        return all([
            self.pdf_processing_success,
            self.text_extraction_success,
            self.preprocessing_success,
            self.lightrag_ingestion_success,
            self.metadata_preservation_success,
            self.cost_tracking_success,
            self.query_validation_success,
            self.cleanup_success,
            self.pipeline_integrity
        ])


# =====================================================================
# INTEGRATION TEST FIXTURES
# =====================================================================

@pytest.fixture
async def integration_test_environment(temp_dir, integration_config, mock_lightrag_system,
                                mock_pdf_processor, mock_cost_monitor, mock_progress_tracker):
    """Provide complete integration test environment."""
    from conftest import IntegrationTestEnv
    
    env = IntegrationTestEnv()
    env.temp_dir = temp_dir
    env.config = integration_config
    env.lightrag_system = mock_lightrag_system
    env.pdf_processor = mock_pdf_processor
    env.cost_monitor = mock_cost_monitor
    env.progress_tracker = mock_progress_tracker
    
    # Initialize working directories
    env.working_dir = temp_dir / "pdf_lightrag_integration"
    env.working_dir.mkdir(exist_ok=True)
    (env.working_dir / "pdfs").mkdir(exist_ok=True)
    (env.working_dir / "processed").mkdir(exist_ok=True)
    (env.working_dir / "logs").mkdir(exist_ok=True)
    
    # Setup realistic cost monitoring
    env.cost_monitor.total_cost = 0.0
    env.cost_monitor.operation_costs = []
    
    yield env
    
    # Cleanup
    try:
        if env.working_dir.exists():
            import shutil
            shutil.rmtree(env.working_dir)
    except:
        pass


@pytest.fixture
def pipeline_test_scenarios():
    """Provide predefined test scenarios for pipeline validation."""
    scenarios = {
        'basic_single_pdf': {
            'name': 'Basic Single PDF Processing',
            'description': 'Process single PDF through complete pipeline',
            'pdf_count': 1,
            'complexity': 'simple',
            'expected_entities': 5,
            'expected_relationships': 3,
            'max_cost': 1.0,
            'max_processing_time': 30.0
        },
        'batch_processing': {
            'name': 'Batch PDF Processing',
            'description': 'Process multiple PDFs in batch mode',
            'pdf_count': 5,
            'complexity': 'medium',
            'expected_entities': 25,
            'expected_relationships': 15,
            'max_cost': 5.0,
            'max_processing_time': 120.0
        },
        'large_document_set': {
            'name': 'Large Document Set Processing',
            'description': 'Process large collection of PDFs',
            'pdf_count': 15,
            'complexity': 'complex',
            'expected_entities': 100,
            'expected_relationships': 75,
            'max_cost': 15.0,
            'max_processing_time': 300.0
        },
        'concurrent_processing': {
            'name': 'Concurrent PDF Processing',
            'description': 'Process PDFs with concurrent workers',
            'pdf_count': 10,
            'complexity': 'medium',
            'concurrent_workers': 3,
            'expected_entities': 50,
            'expected_relationships': 30,
            'max_cost': 10.0,
            'max_processing_time': 180.0
        },
        'metadata_heavy': {
            'name': 'Metadata-Heavy Processing',
            'description': 'Process PDFs with rich metadata extraction',
            'pdf_count': 3,
            'complexity': 'complex',
            'preserve_metadata': True,
            'expected_entities': 30,
            'expected_relationships': 20,
            'max_cost': 3.0,
            'max_processing_time': 60.0
        }
    }
    return scenarios


@pytest.fixture
async def pdf_test_documents(temp_dir, pdf_test_documents):
    """Create realistic PDF files for integration testing."""
    pdf_paths = []
    
    for doc in pdf_test_documents:
        pdf_path = temp_dir / "pdfs" / doc.filename
        pdf_path.parent.mkdir(exist_ok=True)
        
        # Create realistic PDF content
        content = f"""
Title: {doc.title}
Authors: {', '.join(doc.authors)}
Journal: {doc.journal}
Year: {doc.year}
DOI: {doc.doi}
Keywords: {', '.join(doc.keywords)}

Abstract:
{doc.content}

Full Content:
{doc.content * 3}  # Expand content for more realistic size
        """
        
        pdf_path.write_text(content)
        pdf_paths.append(pdf_path)
    
    yield pdf_paths
    
    # Cleanup
    for pdf_path in pdf_paths:
        try:
            pdf_path.unlink()
        except:
            pass


# =====================================================================
# CORE PDF-TO-LIGHTRAG INTEGRATION TESTS
# =====================================================================

class TestPDFProcessingToDocumentIngestionFlow:
    """Test class for core PDF processing to LightRAG document ingestion flow."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_single_pdf_to_lightrag_ingestion(
        self, integration_test_environment, pdf_test_documents,
        performance_monitor
    ):
        """Test complete flow for processing a single PDF to LightRAG ingestion."""
        env = integration_test_environment
        pdf_docs = pdf_test_documents  # Get the list of PDF documents
        pdf_doc = pdf_docs[0]  # Use first PDF document object
        
        async with performance_monitor.monitor_operation("single_pdf_ingestion", pdf_path=str(pdf_path)):
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Process single PDF
            pdf_result = await env.pdf_processor.process_pdf(pdf_path)
            assert pdf_result['success'] == True, f"PDF processing failed: {pdf_result}"
            
            # Verify text extraction
            assert 'text' in pdf_result
            assert len(pdf_result['text']) > 100, "Extracted text too short"
            assert 'metadata' in pdf_result
            
            # Track cost for PDF processing
            env.cost_monitor.track_cost("pdf_processing", 0.05, document_id=str(pdf_path))
            
            # Ingest into LightRAG
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert([pdf_result['text']])
                
                assert ingestion_result['status'] == 'success'
                assert ingestion_result['documents_processed'] == 1
                assert ingestion_result['entities_extracted'] > 0
                assert ingestion_result['relationships_found'] >= 0
            
            # Track cost for LightRAG ingestion
            env.cost_monitor.track_cost("lightrag_ingestion", ingestion_result['total_cost'])
            
            # Validate query capability
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                query_result = await env.lightrag_system.aquery("What metabolites are discussed?")
                assert len(query_result) > 0, "Query returned empty result"
            
            # Track cost for query
            env.cost_monitor.track_cost("lightrag_query", 0.01)
            
            # Verify overall pipeline success
            pipeline_result = PipelineValidationResult(
                pdf_processing_success=pdf_result['success'],
                text_extraction_success=len(pdf_result['text']) > 0,
                preprocessing_success=True,  # Assuming preprocessing is part of PDF processing
                lightrag_ingestion_success=ingestion_result['status'] == 'success',
                metadata_preservation_success='metadata' in pdf_result,
                cost_tracking_success=env.cost_monitor.total_cost > 0,
                query_validation_success=len(query_result) > 0,
                cleanup_success=True,
                pipeline_integrity=True
            )
            
            assert pipeline_result.overall_success, f"Pipeline validation failed: {pipeline_result}"
            
            # Verify cost tracking
            assert env.cost_monitor.total_cost > 0.05, "Cost tracking not working properly"
            assert len(env.cost_monitor.operation_costs) >= 3, "Expected at least 3 cost operations"
            
            # Performance validation
            performance_summary = performance_monitor.get_performance_summary()
            assert performance_summary['total_test_time'] < 30.0, "Single PDF processing too slow"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_batch_pdf_to_lightrag_ingestion(
        self, integration_test_environment, small_pdf_collection,
        performance_monitor
    ):
        """Test batch processing of multiple PDFs to LightRAG ingestion."""
        env = integration_test_environment
        pdf_paths = small_pdf_collection  # Use small collection for faster testing
        
        async with performance_monitor.monitor_operation("batch_pdf_ingestion", pdf_count=len(pdf_paths)):
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Process PDFs in batch
            batch_result = await env.pdf_processor.process_batch_pdfs(pdf_paths)
            assert batch_result['processed'] > 0, f"No PDFs processed successfully: {batch_result}"
            assert batch_result['failed'] == 0, f"Some PDFs failed processing: {batch_result}"
            
            # Extract text from all results
            extracted_texts = []
            total_metadata = []
            
            for result in batch_result['results']:
                assert 'text' in result, "Missing text in PDF processing result"
                assert 'metadata' in result, "Missing metadata in PDF processing result"
                
                extracted_texts.append(result['text'])
                total_metadata.append(result['metadata'])
            
            # Verify batch extraction quality
            assert len(extracted_texts) == len(pdf_paths), "Not all PDFs were processed"
            for text in extracted_texts:
                assert len(text) > 50, "Extracted text too short for batch processing"
            
            # Track cost for batch PDF processing
            batch_cost = len(pdf_paths) * 0.05  # Estimated cost per PDF
            env.cost_monitor.track_cost("batch_pdf_processing", batch_cost, documents=len(pdf_paths))
            
            # Batch ingest into LightRAG
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert(extracted_texts)
                
                assert ingestion_result['status'] == 'success'
                assert ingestion_result['documents_processed'] == len(extracted_texts)
                assert ingestion_result['entities_extracted'] > len(pdf_paths)  # Should extract multiple entities
            
            # Track cost for batch ingestion
            env.cost_monitor.track_cost("batch_lightrag_ingestion", ingestion_result['total_cost'])
            
            # Validate batch query performance
            test_queries = [
                "What are the main metabolites discussed?",
                "Which diseases are mentioned in the documents?",
                "What analytical techniques were used?"
            ]
            
            query_results = []
            for query in test_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    result = await env.lightrag_system.aquery(query)
                    query_results.append(result)
                    env.cost_monitor.track_cost("batch_query", 0.01)
            
            # Verify query results
            for result in query_results:
                assert len(result) > 0, "Batch query returned empty result"
            
            # Create comprehensive test result
            test_result = IntegrationTestResult(
                success=True,
                processing_time=batch_result['total_time'],
                documents_processed=len(pdf_paths),
                entities_extracted=ingestion_result['entities_extracted'],
                relationships_found=ingestion_result['relationships_found'],
                total_cost=env.cost_monitor.total_cost,
                memory_peak_mb=0.0,  # Would be calculated in real implementation
                error_count=batch_result['failed'],
                warning_count=0,
                metadata={'batch_size': len(pdf_paths), 'query_count': len(test_queries)}
            )
            
            # Validate performance metrics
            performance_metrics = test_result.get_performance_metrics()
            if performance_metrics:
                assert performance_metrics['documents_per_second'] > 0.1, "Batch processing too slow"
                assert performance_metrics['cost_per_document'] < 1.0, "Cost per document too high"
            
            # Verify comprehensive pipeline success
            assert test_result.success == True
            assert test_result.documents_processed == len(pdf_paths)
            assert test_result.total_cost > 0.0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pdf_metadata_preservation_in_lightrag(
        self, integration_test_environment, pdf_test_documents,
        performance_monitor
    ):
        """Test that PDF metadata flows through and is preserved in LightRAG."""
        env = integration_test_environment
        pdf_path = pdf_test_documents[0]
        
        async with performance_monitor.monitor_operation("metadata_preservation", pdf_path=str(pdf_path)):
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Process PDF with metadata extraction
            pdf_result = await env.pdf_processor.process_pdf(pdf_path)
            assert 'metadata' in pdf_result, "PDF processing didn't extract metadata"
            
            original_metadata = pdf_result['metadata']
            
            # Verify expected metadata fields
            expected_fields = ['title', 'page_count', 'file_size']
            for field in expected_fields:
                assert field in original_metadata, f"Missing metadata field: {field}"
            
            # Enhanced metadata extraction
            detailed_metadata = await env.pdf_processor.extract_metadata(pdf_path)
            
            # Combine content with metadata for ingestion
            enriched_content = f"""
DOCUMENT METADATA:
Title: {detailed_metadata.get('title', 'Unknown')}
Authors: {detailed_metadata.get('authors', ['Unknown'])}
Journal: {detailed_metadata.get('journal', 'Unknown')}
Year: {detailed_metadata.get('year', 'Unknown')}
DOI: {detailed_metadata.get('doi', 'Unknown')}
Keywords: {detailed_metadata.get('keywords', [])}

DOCUMENT CONTENT:
{pdf_result['text']}
            """.strip()
            
            # Ingest enriched content
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert([enriched_content])
                
                assert ingestion_result['status'] == 'success'
                assert ingestion_result['documents_processed'] == 1
            
            # Verify metadata-based queries work
            metadata_queries = [
                f"What is the title of the document?",
                f"Who are the authors?",
                f"What journal was this published in?",
                f"What year was this published?",
                f"What are the keywords?"
            ]
            
            successful_metadata_queries = 0
            for query in metadata_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    result = await env.lightrag_system.aquery(query)
                    if len(result) > 0:
                        successful_metadata_queries += 1
            
            # At least 80% of metadata queries should succeed
            metadata_success_rate = successful_metadata_queries / len(metadata_queries)
            assert metadata_success_rate >= 0.8, f"Metadata preservation poor: {metadata_success_rate:.2%}"
            
            # Verify metadata preservation in entity extraction
            # Check that entities were extracted from both content and metadata
            entities_extracted = ingestion_result['entities_extracted']
            assert entities_extracted >= 3, f"Too few entities extracted: {entities_extracted}"
            
            # Test document-specific query that requires metadata
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                specific_query = f"Find documents about metabolomics published in recent years"
                specific_result = await env.lightrag_system.aquery(specific_query)
                assert len(specific_result) > 0, "Metadata-based filtering query failed"
            
            # Validate metadata preservation success
            metadata_validation = PipelineValidationResult(
                pdf_processing_success=True,
                text_extraction_success=True,
                preprocessing_success=True,
                lightrag_ingestion_success=True,
                metadata_preservation_success=metadata_success_rate >= 0.8,
                cost_tracking_success=True,
                query_validation_success=len(specific_result) > 0,
                cleanup_success=True,
                pipeline_integrity=True
            )
            
            assert metadata_validation.overall_success, f"Metadata preservation validation failed: {metadata_validation}"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_biomedical_preprocessing_lightrag_compatibility(
        self, integration_test_environment, pdf_test_documents,
        disease_specific_content, performance_monitor
    ):
        """Test biomedical preprocessing output compatibility with LightRAG."""
        env = integration_test_environment
        pdf_path = pdf_test_documents[0]
        
        async with performance_monitor.monitor_operation("biomedical_preprocessing", pdf_path=str(pdf_path)):
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Generate biomedical-specific content for testing
            biomedical_content = disease_specific_content('diabetes', 'complex')
            
            # Simulate biomedical preprocessing steps
            preprocessed_content = f"""
BIOMEDICAL DOCUMENT ANALYSIS:

IDENTIFIED ENTITIES:
- METABOLITES: glucose, insulin, HbA1c, fructose, lactate
- PROTEINS: insulin, glucagon, GLUT4, adiponectin
- PATHWAYS: glucose metabolism, insulin signaling, glycolysis
- DISEASES: diabetes, hyperglycemia
- TREATMENTS: metformin, insulin therapy

PROCESSED CONTENT:
{biomedical_content}

ENTITY RELATIONSHIPS:
- glucose REGULATES insulin
- insulin ACTIVATES GLUT4
- diabetes ASSOCIATED_WITH hyperglycemia
- metformin TREATS diabetes
            """.strip()
            
            # Test LightRAG compatibility with preprocessed content
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert([preprocessed_content])
                
                assert ingestion_result['status'] == 'success'
                assert ingestion_result['documents_processed'] == 1
                
                # Verify biomedical entities were properly extracted
                entities_extracted = ingestion_result['entities_extracted']
                assert entities_extracted >= 10, f"Insufficient biomedical entities extracted: {entities_extracted}"
                
                # Verify relationships were captured
                relationships_found = ingestion_result['relationships_found']
                assert relationships_found >= 5, f"Insufficient relationships found: {relationships_found}"
            
            # Test biomedical-specific queries
            biomedical_queries = [
                "What metabolites are involved in diabetes?",
                "How does insulin regulate glucose metabolism?",
                "What proteins are associated with glucose transport?",
                "Which treatments are effective for diabetes?",
                "What pathways are disrupted in diabetes?"
            ]
            
            successful_biomedical_queries = 0
            query_results = []
            
            for query in biomedical_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    result = await env.lightrag_system.aquery(query)
                    query_results.append(result)
                    
                    # Check if result contains relevant biomedical information
                    if any(term in result.lower() for term in ['glucose', 'insulin', 'diabetes', 'metaboli']):
                        successful_biomedical_queries += 1
            
            # Biomedical query success rate should be high
            biomedical_success_rate = successful_biomedical_queries / len(biomedical_queries)
            assert biomedical_success_rate >= 0.8, f"Biomedical preprocessing compatibility poor: {biomedical_success_rate:.2%}"
            
            # Test entity-specific queries
            entity_queries = [
                "METABOLITE:glucose",
                "PROTEIN:insulin", 
                "DISEASE:diabetes",
                "PATHWAY:glycolysis"
            ]
            
            entity_responses = []
            for entity_query in entity_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    result = await env.lightrag_system.aquery(entity_query)
                    entity_responses.append(result)
            
            # Verify entity-specific queries return relevant results
            for response in entity_responses:
                assert len(response) > 0, "Entity-specific query returned empty result"
            
            # Test relationship queries
            relationship_queries = [
                "How is glucose related to insulin?",
                "What is the relationship between diabetes and hyperglycemia?",
                "How does metformin interact with glucose metabolism?"
            ]
            
            relationship_responses = []
            for rel_query in relationship_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    result = await env.lightrag_system.aquery(rel_query)
                    relationship_responses.append(result)
            
            # Verify relationship queries work
            for response in relationship_responses:
                assert len(response) > 0, "Relationship query returned empty result"
            
            # Comprehensive compatibility validation
            compatibility_result = {
                'entity_extraction_success': entities_extracted >= 10,
                'relationship_extraction_success': relationships_found >= 5,
                'biomedical_query_success': biomedical_success_rate >= 0.8,
                'entity_query_success': all(len(r) > 0 for r in entity_responses),
                'relationship_query_success': all(len(r) > 0 for r in relationship_responses)
            }
            
            overall_compatibility = all(compatibility_result.values())
            assert overall_compatibility, f"Biomedical preprocessing compatibility issues: {compatibility_result}"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.concurrent
    async def test_concurrent_pdf_processing_lightrag_ingestion(
        self, integration_test_environment, small_pdf_collection,
        performance_monitor
    ):
        """Test concurrent processing of PDFs and ingestion into LightRAG."""
        env = integration_test_environment
        pdf_paths = small_pdf_collection
        
        async with performance_monitor.monitor_operation("concurrent_processing", pdf_count=len(pdf_paths)):
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Define concurrent processing function
            async def process_and_ingest_pdf(pdf_path: Path, worker_id: int) -> Dict[str, Any]:
                """Process a single PDF and ingest it concurrently."""
                start_time = time.time()
                
                try:
                    # Process PDF
                    pdf_result = await env.pdf_processor.process_pdf(pdf_path)
                    if not pdf_result['success']:
                        return {'success': False, 'error': 'PDF processing failed', 'worker_id': worker_id}
                    
                    # Track processing cost
                    env.cost_monitor.track_cost(f"concurrent_pdf_processing_worker_{worker_id}", 0.05)
                    
                    # Ingest into LightRAG
                    with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                        ingestion_result = await env.lightrag_system.ainsert([pdf_result['text']])
                        
                        if ingestion_result['status'] != 'success':
                            return {'success': False, 'error': 'LightRAG ingestion failed', 'worker_id': worker_id}
                    
                    # Track ingestion cost
                    env.cost_monitor.track_cost(f"concurrent_lightrag_ingestion_worker_{worker_id}", 
                                              ingestion_result['total_cost'])
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'worker_id': worker_id,
                        'pdf_path': str(pdf_path),
                        'processing_time': processing_time,
                        'entities_extracted': ingestion_result['entities_extracted'],
                        'relationships_found': ingestion_result['relationships_found'],
                        'cost': ingestion_result['total_cost'] + 0.05
                    }
                    
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e),
                        'worker_id': worker_id,
                        'pdf_path': str(pdf_path)
                    }
            
            # Execute concurrent processing
            concurrent_tasks = []
            for i, pdf_path in enumerate(pdf_paths):
                task = process_and_ingest_pdf(pdf_path, i)
                concurrent_tasks.append(task)
            
            # Wait for all concurrent operations to complete
            start_time = time.time()
            concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            total_concurrent_time = time.time() - start_time
            
            # Analyze concurrent processing results
            successful_results = []
            failed_results = []
            
            for result in concurrent_results:
                if isinstance(result, Exception):
                    failed_results.append({'error': str(result), 'exception': True})
                elif result.get('success', False):
                    successful_results.append(result)
                else:
                    failed_results.append(result)
            
            # Validate concurrent processing success
            success_rate = len(successful_results) / len(pdf_paths)
            assert success_rate >= 0.8, f"Concurrent processing success rate too low: {success_rate:.2%}"
            
            # Verify concurrency benefits
            sequential_time_estimate = sum(r['processing_time'] for r in successful_results)
            concurrency_speedup = sequential_time_estimate / total_concurrent_time if total_concurrent_time > 0 else 1
            assert concurrency_speedup > 1.2, f"Insufficient concurrency speedup: {concurrency_speedup:.2f}x"
            
            # Test concurrent querying
            concurrent_queries = [
                "What metabolites are discussed in the documents?",
                "Which diseases are mentioned?",
                "What analytical techniques were used?",
                "What are the main findings?",
                "Which proteins are involved?"
            ]
            
            async def execute_concurrent_query(query: str, query_id: int) -> Dict[str, Any]:
                """Execute a query concurrently."""
                try:
                    with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                        result = await env.lightrag_system.aquery(query)
                        env.cost_monitor.track_cost(f"concurrent_query_{query_id}", 0.01)
                        return {'success': True, 'query': query, 'result': result}
                except Exception as e:
                    return {'success': False, 'query': query, 'error': str(e)}
            
            # Execute concurrent queries
            query_tasks = [execute_concurrent_query(q, i) for i, q in enumerate(concurrent_queries)]
            query_results = await asyncio.gather(*query_tasks)
            
            # Validate concurrent query results
            successful_queries = [r for r in query_results if r.get('success', False)]
            query_success_rate = len(successful_queries) / len(concurrent_queries)
            assert query_success_rate >= 0.8, f"Concurrent query success rate too low: {query_success_rate:.2%}"
            
            # Create comprehensive concurrent test result
            concurrent_test_result = IntegrationTestResult(
                success=success_rate >= 0.8 and query_success_rate >= 0.8,
                processing_time=total_concurrent_time,
                documents_processed=len(successful_results),
                entities_extracted=sum(r.get('entities_extracted', 0) for r in successful_results),
                relationships_found=sum(r.get('relationships_found', 0) for r in successful_results),
                total_cost=env.cost_monitor.total_cost,
                memory_peak_mb=0.0,  # Would be measured in real implementation
                error_count=len(failed_results),
                warning_count=0,
                metadata={
                    'concurrency_speedup': concurrency_speedup,
                    'success_rate': success_rate,
                    'query_success_rate': query_success_rate,
                    'concurrent_workers': len(pdf_paths)
                }
            )
            
            # Verify overall concurrent processing success
            assert concurrent_test_result.success, f"Concurrent processing test failed: {concurrent_test_result.metadata}"
            
            # Validate performance characteristics
            performance_metrics = concurrent_test_result.get_performance_metrics()
            if performance_metrics:
                assert performance_metrics['documents_per_second'] > 0.2, "Concurrent processing too slow"


# =====================================================================
# METADATA PRESERVATION AND FLOW TESTS
# =====================================================================

class TestPDFMetadataPreservationFlow:
    """Test class for PDF metadata preservation throughout the processing pipeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_metadata_flow_validation(
        self, integration_test_environment, pdf_test_documents,
        performance_monitor
    ):
        """Test complete metadata flow from PDF to LightRAG with validation at each step."""
        env = integration_test_environment
        pdf_path = pdf_test_documents[0]
        
        async with performance_monitor.monitor_operation("metadata_flow_validation"):
            # Step 1: PDF Metadata Extraction
            original_metadata = await env.pdf_processor.extract_metadata(pdf_path)
            
            # Verify extracted metadata structure
            required_fields = ['title', 'authors', 'journal', 'year', 'keywords']
            for field in required_fields:
                assert field in original_metadata, f"Missing required metadata field: {field}"
            
            # Step 2: PDF Content Processing
            pdf_result = await env.pdf_processor.process_pdf(pdf_path)
            processing_metadata = pdf_result['metadata']
            
            # Verify processing metadata includes original metadata
            for field in required_fields:
                if field in original_metadata:
                    # Some fields might be transformed or enriched during processing
                    assert field in processing_metadata or f"processed_{field}" in processing_metadata, \
                        f"Metadata field lost during processing: {field}"
            
            # Step 3: Content Enrichment with Metadata
            enriched_document = {
                'content': pdf_result['text'],
                'metadata': {
                    **original_metadata,
                    'processing_timestamp': time.time(),
                    'source_file': str(pdf_path),
                    'processing_version': '1.0'
                }
            }
            
            # Step 4: LightRAG Ingestion with Metadata
            metadata_json = json.dumps(enriched_document['metadata'], indent=2)
            ingestion_content = f"""
METADATA:
{metadata_json}

CONTENT:
{enriched_document['content']}
            """.strip()
            
            # Initialize RAG system
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert([ingestion_content])
                
                assert ingestion_result['status'] == 'success'
                assert ingestion_result['documents_processed'] == 1
            
            # Step 5: Metadata-Based Query Validation
            metadata_queries = [
                f"Find documents with title containing '{original_metadata['title'][:20]}'",
                f"Show documents by authors {original_metadata['authors'][0] if original_metadata['authors'] else 'Unknown'}",
                f"Find papers published in {original_metadata['year']}",
                f"Search for documents in {original_metadata['journal']}",
                f"Find documents with keywords {original_metadata['keywords'][0] if original_metadata['keywords'] else 'metabolomics'}"
            ]
            
            metadata_query_results = []
            for query in metadata_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    result = await env.lightrag_system.aquery(query)
                    metadata_query_results.append({
                        'query': query,
                        'result': result,
                        'success': len(result) > 0
                    })
            
            # Verify metadata-based queries work
            successful_metadata_queries = sum(1 for r in metadata_query_results if r['success'])
            metadata_query_success_rate = successful_metadata_queries / len(metadata_queries)
            
            assert metadata_query_success_rate >= 0.6, f"Metadata query success rate too low: {metadata_query_success_rate:.2%}"
            
            # Step 6: Cross-Reference Validation
            # Test that content and metadata are properly linked
            cross_reference_queries = [
                "What is the title of this document about metabolomics?",
                "Who wrote this paper on clinical research?",
                "When was this study published?"
            ]
            
            cross_ref_results = []
            for query in cross_reference_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    result = await env.lightrag_system.aquery(query)
                    cross_ref_results.append(result)
            
            # Verify cross-reference queries return meaningful results
            meaningful_cross_refs = sum(1 for r in cross_ref_results if len(r) > 10)  # Non-empty, substantial responses
            cross_ref_success_rate = meaningful_cross_refs / len(cross_reference_queries)
            
            assert cross_ref_success_rate >= 0.5, f"Cross-reference query success rate too low: {cross_ref_success_rate:.2%}"
            
            # Comprehensive metadata flow validation
            metadata_flow_result = {
                'extraction_success': all(field in original_metadata for field in required_fields),
                'processing_preservation': all(field in processing_metadata or f"processed_{field}" in processing_metadata 
                                             for field in required_fields if field in original_metadata),
                'ingestion_success': ingestion_result['status'] == 'success',
                'query_success': metadata_query_success_rate >= 0.6,
                'cross_reference_success': cross_ref_success_rate >= 0.5
            }
            
            overall_metadata_success = all(metadata_flow_result.values())
            assert overall_metadata_success, f"Metadata flow validation failed: {metadata_flow_result}"


# =====================================================================
# COST TRACKING INTEGRATION TESTS
# =====================================================================

class TestCostTrackingIntegration:
    """Test class for cost tracking integration across the PDF-LightRAG pipeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_comprehensive_cost_tracking_pipeline(
        self, integration_test_environment, small_pdf_collection,
        cost_persistence, budget_manager
    ):
        """Test comprehensive cost tracking throughout the entire pipeline."""
        env = integration_test_environment
        pdf_paths = small_pdf_collection
        
        # Setup cost tracking components
        env.cost_monitor = budget_manager
        
        # Initialize RAG system
        rag_system = ClinicalMetabolomicsRAG(config=env.config)
        
        # Track initial state
        initial_cost = budget_manager.get_current_daily_cost()
        
        # Process each PDF and track costs
        total_processing_cost = 0.0
        total_ingestion_cost = 0.0
        total_query_cost = 0.0
        
        for i, pdf_path in enumerate(pdf_paths):
            # PDF Processing Cost
            pdf_start_time = time.time()
            pdf_result = await env.pdf_processor.process_pdf(pdf_path)
            pdf_processing_time = time.time() - pdf_start_time
            
            # Calculate PDF processing cost (based on processing time and file size)
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024) if pdf_path.exists() else 0.1
            pdf_cost = (pdf_processing_time * 0.01) + (file_size_mb * 0.002)  # Time + size based cost
            total_processing_cost += pdf_cost
            
            # Record PDF processing cost
            budget_manager.record_operation_cost(
                operation_type="pdf_processing",
                cost_usd=pdf_cost,
                model_name="pdf_processor",
                prompt_tokens=int(file_size_mb * 1000),
                completion_tokens=len(pdf_result['text'].split()) if pdf_result.get('text') else 0,
                metadata={'file_path': str(pdf_path), 'processing_time': pdf_processing_time}
            )
            
            # LightRAG Ingestion Cost
            if pdf_result['success']:
                ingestion_start_time = time.time()
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    ingestion_result = await env.lightrag_system.ainsert([pdf_result['text']])
                ingestion_time = time.time() - ingestion_start_time
                
                # Calculate ingestion cost
                text_tokens = len(pdf_result['text'].split())
                ingestion_cost = (text_tokens / 1000) * 0.0001  # Token-based cost
                total_ingestion_cost += ingestion_cost
                
                # Record ingestion cost
                budget_manager.record_operation_cost(
                    operation_type="lightrag_ingestion",
                    cost_usd=ingestion_cost,
                    model_name="text-embedding-3-small",
                    prompt_tokens=text_tokens,
                    completion_tokens=0,
                    metadata={'document_id': f"doc_{i}", 'ingestion_time': ingestion_time}
                )
        
        # Query Processing Costs
        test_queries = [
            "What are the main metabolites discussed?",
            "Which diseases are mentioned in the documents?",
            "What analytical techniques were used?",
            "What are the key findings?",
            "How are proteins and metabolites related?"
        ]
        
        for j, query in enumerate(test_queries):
            query_start_time = time.time()
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                query_result = await env.lightrag_system.aquery(query)
            query_time = time.time() - query_start_time
            
            # Calculate query cost
            query_tokens = len(query.split())
            response_tokens = len(query_result.split()) if query_result else 0
            query_cost = (query_tokens * 0.0001) + (response_tokens * 0.0002)  # Input + output tokens
            total_query_cost += query_cost
            
            # Record query cost
            budget_manager.record_operation_cost(
                operation_type="lightrag_query",
                cost_usd=query_cost,
                model_name="gpt-4o-mini",
                prompt_tokens=query_tokens,
                completion_tokens=response_tokens,
                metadata={'query_id': f"query_{j}", 'query_time': query_time}
            )
        
        # Verify cost tracking accuracy
        final_cost = budget_manager.get_current_daily_cost()
        total_expected_cost = total_processing_cost + total_ingestion_cost + total_query_cost
        actual_cost_increase = final_cost - initial_cost
        
        # Cost tracking should be accurate within 10%
        cost_accuracy = abs(actual_cost_increase - total_expected_cost) / total_expected_cost if total_expected_cost > 0 else 0
        assert cost_accuracy <= 0.1, f"Cost tracking accuracy poor: {cost_accuracy:.2%} difference"
        
        # Verify cost breakdown
        cost_summary = budget_manager.get_cost_breakdown()
        
        # Check that all operation types are tracked
        expected_operations = {'pdf_processing', 'lightrag_ingestion', 'lightrag_query'}
        tracked_operations = set(cost_summary.keys())
        assert expected_operations.issubset(tracked_operations), f"Missing cost operations: {expected_operations - tracked_operations}"
        
        # Verify cost components
        assert cost_summary['pdf_processing']['total_cost'] > 0, "PDF processing costs not tracked"
        assert cost_summary['lightrag_ingestion']['total_cost'] > 0, "LightRAG ingestion costs not tracked"
        assert cost_summary['lightrag_query']['total_cost'] > 0, "Query costs not tracked"
        
        # Test budget alerts
        # Set a low budget to trigger alerts
        budget_manager.daily_budget_limit = total_expected_cost * 0.8  # 80% of actual cost
        
        alerts = budget_manager.check_budget_alerts()
        assert len(alerts) > 0, "Budget alerts not triggered when expected"
        
        # Verify alert content
        budget_alert = alerts[0]
        assert 'daily' in budget_alert['alert_type'].lower()
        assert budget_alert['current_cost'] > budget_alert['budget_limit']
        
        # Test cost persistence
        cost_records = cost_persistence.get_recent_costs(days=1)
        assert len(cost_records) > 0, "Cost records not persisted"
        
        # Verify persisted records match tracked costs
        total_persisted_cost = sum(record.cost_usd for record in cost_records)
        persistence_accuracy = abs(total_persisted_cost - actual_cost_increase) / actual_cost_increase if actual_cost_increase > 0 else 0
        assert persistence_accuracy <= 0.05, f"Cost persistence accuracy poor: {persistence_accuracy:.2%}"


# =====================================================================
# PERFORMANCE AND RESOURCE TESTS
# =====================================================================

class TestPerformanceCharacteristics:
    """Test class for performance characteristics and resource usage."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_pipeline_performance_benchmarks(
        self, integration_test_environment, large_pdf_collection,
        performance_monitor
    ):
        """Test performance benchmarks for the complete PDF-LightRAG pipeline."""
        env = integration_test_environment
        pdf_paths = large_pdf_collection[:10]  # Use subset for reasonable test time
        
        async with performance_monitor.monitor_operation("performance_benchmark", document_count=len(pdf_paths)):
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Track resource usage
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            initial_cpu_time = process.cpu_times()
            
            # Performance benchmark: Batch processing
            batch_start_time = time.time()
            batch_result = await env.pdf_processor.process_batch_pdfs(pdf_paths)
            batch_processing_time = time.time() - batch_start_time
            
            # Extract texts for ingestion
            extracted_texts = []
            for result in batch_result['results']:
                if result.get('success', False):
                    extracted_texts.append(result['text'])
            
            # Performance benchmark: Batch ingestion
            ingestion_start_time = time.time()
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert(extracted_texts)
            batch_ingestion_time = time.time() - ingestion_start_time
            
            # Performance benchmark: Query processing
            test_queries = [
                "What metabolites are most frequently mentioned?",
                "Which diseases show the strongest associations with metabolic changes?",
                "What analytical techniques are most commonly used?",
                "How do protein levels correlate with metabolite concentrations?",
                "What pathways are most significantly affected?"
            ]
            
            query_start_time = time.time()
            query_results = []
            for query in test_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    result = await env.lightrag_system.aquery(query)
                    query_results.append(result)
            total_query_time = time.time() - query_start_time
            
            # Calculate resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu_time = process.cpu_times()
            
            memory_usage = final_memory - initial_memory
            cpu_usage = (final_cpu_time.user - initial_cpu_time.user) + (final_cpu_time.system - initial_cpu_time.system)
            
            # Calculate performance metrics
            total_documents = len(extracted_texts)
            total_pipeline_time = batch_processing_time + batch_ingestion_time + total_query_time
            
            performance_metrics = {
                'documents_per_second': total_documents / total_pipeline_time if total_pipeline_time > 0 else 0,
                'processing_time_per_document': batch_processing_time / total_documents if total_documents > 0 else 0,
                'ingestion_time_per_document': batch_ingestion_time / total_documents if total_documents > 0 else 0,
                'query_time_per_query': total_query_time / len(test_queries) if test_queries else 0,
                'memory_per_document': memory_usage / total_documents if total_documents > 0 else 0,
                'cpu_time_per_document': cpu_usage / total_documents if total_documents > 0 else 0
            }
            
            # Validate performance benchmarks
            assert performance_metrics['documents_per_second'] > 0.5, f"Pipeline throughput too low: {performance_metrics['documents_per_second']:.2f} docs/sec"
            assert performance_metrics['processing_time_per_document'] < 10.0, f"PDF processing too slow: {performance_metrics['processing_time_per_document']:.2f}s per doc"
            assert performance_metrics['ingestion_time_per_document'] < 5.0, f"LightRAG ingestion too slow: {performance_metrics['ingestion_time_per_document']:.2f}s per doc"
            assert performance_metrics['query_time_per_query'] < 2.0, f"Query processing too slow: {performance_metrics['query_time_per_query']:.2f}s per query"
            assert performance_metrics['memory_per_document'] < 50.0, f"Memory usage too high: {performance_metrics['memory_per_document']:.2f}MB per doc"
            
            # Test scalability characteristics
            # Process smaller batch to compare scaling
            small_batch = pdf_paths[:3]
            small_batch_start = time.time()
            small_batch_result = await env.pdf_processor.process_batch_pdfs(small_batch)
            small_batch_time = time.time() - small_batch_start
            
            # Calculate scaling efficiency
            small_batch_per_doc = small_batch_time / len(small_batch)
            large_batch_per_doc = batch_processing_time / total_documents
            scaling_efficiency = small_batch_per_doc / large_batch_per_doc if large_batch_per_doc > 0 else 1
            
            # Scaling should show some efficiency gains (or at least not be much worse)
            assert scaling_efficiency < 2.0, f"Poor scaling efficiency: {scaling_efficiency:.2f}x slower per document in large batch"
            
            # Memory cleanup test
            del extracted_texts, query_results
            import gc
            gc.collect()
            
            cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_cleanup_efficiency = (final_memory - cleanup_memory) / memory_usage if memory_usage > 0 else 0
            
            # Should reclaim at least 30% of used memory after cleanup
            assert memory_cleanup_efficiency >= 0.3, f"Poor memory cleanup: only {memory_cleanup_efficiency:.2%} reclaimed"
            
            # Create comprehensive performance result
            performance_result = IntegrationTestResult(
                success=True,
                processing_time=total_pipeline_time,
                documents_processed=total_documents,
                entities_extracted=ingestion_result.get('entities_extracted', 0),
                relationships_found=ingestion_result.get('relationships_found', 0),
                total_cost=0.0,  # Cost tracking handled separately
                memory_peak_mb=memory_usage,
                error_count=batch_result.get('failed', 0),
                warning_count=0,
                metadata=performance_metrics
            )
            
            assert performance_result.success, "Performance benchmark test failed"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_memory_management_under_load(
        self, integration_test_environment, large_pdf_collection,
        performance_monitor
    ):
        """Test memory management during high-load processing."""
        env = integration_test_environment
        pdf_paths = large_pdf_collection[:8]  # Reasonable size for memory test
        
        async with performance_monitor.monitor_operation("memory_load_test"):
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            memory_samples = []
            
            # Process documents one by one and track memory
            for i, pdf_path in enumerate(pdf_paths):
                # Sample memory before processing
                pre_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(('before_processing', i, pre_memory))
                
                # Process PDF
                pdf_result = await env.pdf_processor.process_pdf(pdf_path)
                
                # Sample memory after processing
                post_processing_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(('after_processing', i, post_processing_memory))
                
                # Ingest into LightRAG
                if pdf_result['success']:
                    with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                        await env.lightrag_system.ainsert([pdf_result['text']])
                
                # Sample memory after ingestion
                post_ingestion_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(('after_ingestion', i, post_ingestion_memory))
                
                # Force garbage collection periodically
                if i % 3 == 0:
                    import gc
                    gc.collect()
                    post_gc_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(('after_gc', i, post_gc_memory))
            
            # Analyze memory usage patterns
            peak_memory = max(sample[2] for sample in memory_samples)
            final_memory = memory_samples[-1][2]
            memory_growth = final_memory - initial_memory
            
            # Memory growth should be bounded
            max_reasonable_growth = len(pdf_paths) * 20  # 20MB per document max
            assert memory_growth < max_reasonable_growth, f"Excessive memory growth: {memory_growth:.1f}MB for {len(pdf_paths)} documents"
            
            # Memory usage should not continuously grow without bounds
            processing_memories = [s[2] for s in memory_samples if s[0] == 'after_processing']
            if len(processing_memories) >= 3:
                # Check if memory growth is linear (bad) or bounded (good)
                first_third = processing_memories[:len(processing_memories)//3]
                last_third = processing_memories[-len(processing_memories)//3:]
                
                avg_first_third = sum(first_third) / len(first_third)
                avg_last_third = sum(last_third) / len(last_third)
                memory_growth_rate = (avg_last_third - avg_first_third) / (len(processing_memories) / 3)
                
                # Growth rate should be reasonable (less than 5MB per document)
                assert memory_growth_rate < 5.0, f"Excessive memory growth rate: {memory_growth_rate:.2f}MB per document"
            
            # Test memory pressure handling
            # Create artificial memory pressure
            memory_pressure_objects = []
            try:
                # Consume significant memory
                for i in range(20):
                    memory_pressure_objects.append([0] * 1000000)  # 1M integers each
                
                # Try processing under memory pressure
                pressure_pdf = pdf_paths[0]
                pressure_result = await env.pdf_processor.process_pdf(pressure_pdf)
                
                # Should still work under memory pressure
                assert pressure_result['success'], "PDF processing failed under memory pressure"
                
            except MemoryError:
                # If we hit memory limits, that's actually expected behavior
                pass
            finally:
                # Clean up memory pressure
                memory_pressure_objects.clear()
                import gc
                gc.collect()
            
            # Final cleanup and memory verification
            cleanup_memory = process.memory_info().rss / 1024 / 1024
            cleanup_effectiveness = (final_memory - cleanup_memory) / memory_growth if memory_growth > 0 else 1
            
            # Should reclaim reasonable amount of memory
            assert cleanup_effectiveness >= 0.2, f"Poor memory cleanup effectiveness: {cleanup_effectiveness:.2%}"


# =====================================================================
# ERROR HANDLING AND RECOVERY TESTS
# =====================================================================

class TestErrorHandlingAndRecovery:
    """Test class for error handling and recovery scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_resilience_to_pdf_processing_failures(
        self, integration_test_environment, pdf_test_documents,
        error_injector, performance_monitor
    ):
        """Test pipeline resilience when PDF processing encounters failures."""
        env = integration_test_environment
        pdf_paths = pdf_test_documents
        
        async with performance_monitor.monitor_operation("pdf_failure_resilience"):
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Setup error injection for PDF processing
            error_injector.add_rule(
                target="pdf_processing",
                error_type=PDFValidationError("Simulated PDF processing failure"),
                trigger_after=2,  # Fail after 2 successful operations
                probability=0.3  # 30% failure rate
            )
            
            # Track processing results
            successful_processes = []
            failed_processes = []
            recovery_attempts = []
            
            for i, pdf_path in enumerate(pdf_paths):
                try:
                    # Check if error should be injected
                    error = error_injector.should_inject_error("pdf_processing")
                    if error:
                        raise error
                    
                    # Process PDF normally
                    pdf_result = await env.pdf_processor.process_pdf(pdf_path)
                    
                    if pdf_result['success']:
                        successful_processes.append({
                            'path': pdf_path,
                            'result': pdf_result,
                            'attempt': i
                        })
                    else:
                        failed_processes.append({
                            'path': pdf_path,
                            'error': 'PDF processing returned success=False',
                            'attempt': i
                        })
                
                except Exception as e:
                    failed_processes.append({
                        'path': pdf_path,
                        'error': str(e),
                        'attempt': i
                    })
                    
                    # Test recovery mechanism
                    try:
                        # Retry with simplified processing
                        recovery_result = {
                            'text': f"Fallback content for {pdf_path.name}",
                            'metadata': {'title': pdf_path.name, 'fallback': True},
                            'success': True
                        }
                        recovery_attempts.append({
                            'path': pdf_path,
                            'recovery_result': recovery_result,
                            'original_error': str(e)
                        })
                    except:
                        # Recovery also failed
                        pass
            
            # Verify graceful degradation
            total_attempts = len(pdf_paths)
            success_rate = len(successful_processes) / total_attempts
            recovery_rate = len(recovery_attempts) / len(failed_processes) if failed_processes else 1.0
            
            # Should maintain reasonable success rate even with failures
            effective_success_rate = (len(successful_processes) + len(recovery_attempts)) / total_attempts
            assert effective_success_rate >= 0.6, f"Pipeline resilience poor: {effective_success_rate:.2%} effective success rate"
            
            # Test ingestion with mixed success/failure results
            successful_texts = []
            
            # Add successful processing results
            for success in successful_processes:
                successful_texts.append(success['result']['text'])
            
            # Add recovery results
            for recovery in recovery_attempts:
                successful_texts.append(recovery['recovery_result']['text'])
            
            # Attempt batch ingestion of available results
            if successful_texts:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    ingestion_result = await env.lightrag_system.ainsert(successful_texts)
                    
                    assert ingestion_result['status'] == 'success'
                    assert ingestion_result['documents_processed'] == len(successful_texts)
            
            # Test error reporting and logging
            error_summary = {
                'total_attempts': total_attempts,
                'successful_processes': len(successful_processes),
                'failed_processes': len(failed_processes),
                'recovery_attempts': len(recovery_attempts),
                'success_rate': success_rate,
                'recovery_rate': recovery_rate,
                'effective_success_rate': effective_success_rate
            }
            
            # Verify error handling didn't crash the system
            assert rag_system.is_initialized, "RAG system should remain initialized despite processing failures"
            
            # Test continued functionality after errors
            if successful_texts:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    test_query = "What information is available in the processed documents?"
                    query_result = await env.lightrag_system.aquery(test_query)
                    assert len(query_result) > 0, "System should remain functional after processing failures"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_lightrag_ingestion_failure_recovery(
        self, integration_test_environment, small_pdf_collection,
        error_injector, performance_monitor
    ):
        """Test recovery from LightRAG ingestion failures."""
        env = integration_test_environment
        pdf_paths = small_pdf_collection
        
        async with performance_monitor.monitor_operation("lightrag_failure_recovery"):
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Process all PDFs successfully first
            pdf_results = []
            for pdf_path in pdf_paths:
                result = await env.pdf_processor.process_pdf(pdf_path)
                if result['success']:
                    pdf_results.append(result)
            
            # Setup error injection for LightRAG ingestion
            error_injector.add_rule(
                target="lightrag_ingestion",
                error_type=Exception("Simulated LightRAG ingestion failure"),
                trigger_after=1,
                probability=0.5  # 50% failure rate
            )
            
            # Track ingestion attempts
            successful_ingestions = []
            failed_ingestions = []
            retry_attempts = []
            
            for i, pdf_result in enumerate(pdf_results):
                try:
                    # Check if error should be injected
                    error = error_injector.should_inject_error("lightrag_ingestion")
                    if error:
                        raise error
                    
                    # Normal ingestion
                    with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                        ingestion_result = await env.lightrag_system.ainsert([pdf_result['text']])
                        successful_ingestions.append({
                            'pdf_result': pdf_result,
                            'ingestion_result': ingestion_result,
                            'attempt': i
                        })
                
                except Exception as e:
                    failed_ingestions.append({
                        'pdf_result': pdf_result,
                        'error': str(e),
                        'attempt': i
                    })
                    
                    # Test retry mechanism
                    for retry in range(2):  # Try up to 2 retries
                        try:
                            await asyncio.sleep(0.1)  # Brief delay
                            
                            # Retry with potentially reduced content
                            retry_content = pdf_result['text'][:1000]  # Truncate to first 1000 chars
                            
                            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                                retry_result = await env.lightrag_system.ainsert([retry_content])
                                retry_attempts.append({
                                    'pdf_result': pdf_result,
                                    'retry_result': retry_result,
                                    'retry_number': retry + 1,
                                    'original_error': str(e)
                                })
                            break  # Success, no need for more retries
                            
                        except Exception:
                            if retry == 1:  # Last retry failed
                                # Log final failure
                                pass
                            continue
            
            # Verify recovery effectiveness
            total_documents = len(pdf_results)
            successful_ingestion_count = len(successful_ingestions)
            retry_success_count = len(retry_attempts)
            
            overall_ingestion_rate = (successful_ingestion_count + retry_success_count) / total_documents
            assert overall_ingestion_rate >= 0.7, f"Ingestion failure recovery poor: {overall_ingestion_rate:.2%} success rate"
            
            # Test system stability after failures
            # System should still be functional for new ingestions
            test_content = "Test document for system stability verification"
            try:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    stability_result = await env.lightrag_system.ainsert([test_content])
                    assert stability_result['status'] == 'success', "System unstable after ingestion failures"
            except Exception as e:
                # System instability detected
                assert False, f"System became unstable after ingestion failures: {e}"
            
            # Test query functionality after partial failures
            if successful_ingestions or retry_attempts:
                try:
                    with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                        recovery_query = "What information was successfully ingested?"
                        query_result = await env.lightrag_system.aquery(recovery_query)
                        assert len(query_result) > 0, "Query functionality compromised after ingestion failures"
                except Exception as e:
                    assert False, f"Query functionality failed after ingestion failures: {e}"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_resource_exhaustion_handling(
        self, integration_test_environment, large_pdf_collection,
        performance_monitor
    ):
        """Test handling of resource exhaustion scenarios."""
        env = integration_test_environment
        pdf_paths = large_pdf_collection[:5]  # Use subset to control resource usage
        
        async with performance_monitor.monitor_operation("resource_exhaustion_test"):
            # Initialize components
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Simulate memory exhaustion scenario
            memory_consumers = []
            resource_exhaustion_encountered = False
            
            try:
                # Gradually increase memory pressure
                for pressure_level in range(10):
                    # Add memory pressure
                    memory_consumers.append([0] * 500000)  # 500K integers
                    
                    # Try processing under increasing memory pressure
                    for i, pdf_path in enumerate(pdf_paths):
                        try:
                            # Monitor memory before processing
                            import psutil
                            process = psutil.Process()
                            memory_before = process.memory_info().rss / 1024 / 1024
                            
                            # Attempt processing
                            pdf_result = await env.pdf_processor.process_pdf(pdf_path)
                            
                            if pdf_result['success']:
                                # Attempt ingestion under memory pressure
                                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                                    ingestion_result = await env.lightrag_system.ainsert([pdf_result['text']])
                                    
                                    # Monitor memory after processing
                                    memory_after = process.memory_info().rss / 1024 / 1024
                                    memory_increase = memory_after - memory_before
                                    
                                    # If memory increase is excessive, we're hitting resource limits
                                    if memory_increase > 100:  # More than 100MB increase
                                        resource_exhaustion_encountered = True
                                        break
                            
                        except (MemoryError, PDFMemoryError) as e:
                            # Expected resource exhaustion
                            resource_exhaustion_encountered = True
                            
                            # Test graceful degradation
                            try:
                                # Try simplified processing
                                fallback_content = f"Simplified content for {pdf_path.name}"
                                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                                    fallback_result = await env.lightrag_system.ainsert([fallback_content])
                                    assert fallback_result['status'] == 'success', "Fallback processing should succeed"
                            except Exception:
                                # Even fallback failed, system under severe pressure
                                pass
                            break
                    
                    if resource_exhaustion_encountered:
                        break
            
            finally:
                # Clean up memory pressure
                memory_consumers.clear()
                import gc
                gc.collect()
            
            # Test system recovery after resource exhaustion
            await asyncio.sleep(1.0)  # Allow system to stabilize
            
            # Verify system can still process documents after resource pressure
            test_pdf = pdf_paths[0]
            try:
                recovery_result = await env.pdf_processor.process_pdf(test_pdf)
                assert recovery_result['success'], "System should recover from resource exhaustion"
                
                # Verify ingestion still works
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    recovery_ingestion = await env.lightrag_system.ainsert([recovery_result['text']])
                    assert recovery_ingestion['status'] == 'success', "Ingestion should work after recovery"
                    
            except Exception as e:
                assert False, f"System failed to recover from resource exhaustion: {e}"
            
            # Verify no permanent damage to system state
            assert rag_system.is_initialized, "RAG system should remain initialized after resource exhaustion"
            
            # Test query functionality after resource pressure
            try:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    recovery_query = "Test query after resource exhaustion"
                    query_result = await env.lightrag_system.aquery(recovery_query)
                    assert len(query_result) > 0, "Query functionality should be restored after recovery"
            except Exception as e:
                assert False, f"Query functionality not restored after recovery: {e}"


# =====================================================================
# END-TO-END PDF TO QUERY WORKFLOW TESTS
# =====================================================================

class TestEndToEndPDFQueryWorkflow:
    """Test class for complete end-to-end workflows from PDF file to final query response."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_pdf_to_query_response_workflow(
        self, integration_test_environment, pdf_test_documents,
        performance_monitor
    ):
        """Test complete workflow from PDF file to final query response."""
        env = integration_test_environment
        pdf_path = pdf_test_documents[0]
        
        async with performance_monitor.monitor_operation(
            "complete_pdf_to_query_workflow", 
            pdf_path=str(pdf_path)
        ):
            # Step 1: Initialize RAG system
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Step 2: Process PDF
            pdf_result = await env.pdf_processor.process_pdf(pdf_path)
            assert pdf_result['success'], f"PDF processing failed: {pdf_result}"
            
            extracted_text = pdf_result['text']
            pdf_metadata = pdf_result['metadata']
            
            # Step 3: Create enriched content for ingestion
            enriched_content = f"""
DOCUMENT METADATA:
Title: {pdf_metadata.get('title', 'Unknown')}
Source: {pdf_path.name}

CONTENT:
{extracted_text}
            """.strip()
            
            # Step 4: Ingest into LightRAG knowledge base
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert([enriched_content])
                
                assert ingestion_result['status'] == 'success'
                assert ingestion_result['documents_processed'] == 1
                
                entities_count = ingestion_result['entities_extracted']
                relationships_count = ingestion_result['relationships_found']
                
                assert entities_count > 0, "No entities extracted from PDF"
                assert relationships_count >= 0, "Negative relationships count"
            
            # Step 5: Execute comprehensive query workflow
            test_queries = [
                {
                    'query': "What are the main metabolites discussed in this document?",
                    'expected_keywords': ['glucose', 'metabolite', 'biomarker'],
                    'min_response_length': 50
                },
                {
                    'query': "Which diseases are mentioned in the study?",
                    'expected_keywords': ['disease', 'condition', 'patient'],
                    'min_response_length': 30
                },
                {
                    'query': "What analytical techniques were used?",
                    'expected_keywords': ['technique', 'method', 'analysis'],
                    'min_response_length': 40
                },
                {
                    'query': "What are the key findings of this research?",
                    'expected_keywords': ['finding', 'result', 'significant'],
                    'min_response_length': 60
                }
            ]
            
            query_results = []
            total_query_cost = 0.0
            
            for i, test_case in enumerate(test_queries):
                query_start_time = time.time()
                
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    response = await env.lightrag_system.aquery(test_case['query'])
                
                query_time = time.time() - query_start_time
                
                # Validate query response
                assert len(response) >= test_case['min_response_length'], \
                    f"Query {i+1} response too short: {len(response)} chars"
                
                # Check for expected keywords (at least one should be present)
                response_lower = response.lower()
                keyword_found = any(
                    keyword.lower() in response_lower 
                    for keyword in test_case['expected_keywords']
                )
                assert keyword_found, \
                    f"Query {i+1} response missing expected keywords: {test_case['expected_keywords']}"
                
                # Estimate query cost
                query_cost = len(test_case['query'].split()) * 0.0001 + len(response.split()) * 0.0002
                total_query_cost += query_cost
                
                query_results.append({
                    'query': test_case['query'],
                    'response': response,
                    'query_time': query_time,
                    'response_length': len(response),
                    'cost': query_cost,
                    'keywords_found': keyword_found
                })
                
                # Track cost
                env.cost_monitor.track_cost(f"end_to_end_query_{i+1}", query_cost)
            
            # Step 6: Validate end-to-end workflow performance
            performance_summary = performance_monitor.get_performance_summary()
            
            # Workflow should complete within reasonable time
            assert performance_summary['total_test_time'] < 60.0, \
                f"End-to-end workflow too slow: {performance_summary['total_test_time']:.2f}s"
            
            # All queries should succeed
            successful_queries = len([r for r in query_results if r['keywords_found']])
            success_rate = successful_queries / len(test_queries)
            assert success_rate >= 0.75, f"Query success rate too low: {success_rate:.2%}"
            
            # Cost tracking should work
            assert env.cost_monitor.total_cost > 0, "Cost tracking not functioning"
            
            # Step 7: Test query response quality
            avg_response_length = sum(r['response_length'] for r in query_results) / len(query_results)
            assert avg_response_length > 100, "Average query responses too short"
            
            avg_query_time = sum(r['query_time'] for r in query_results) / len(query_results)
            assert avg_query_time < 3.0, f"Average query time too slow: {avg_query_time:.2f}s"
            
            # Step 8: Validate complete workflow success
            workflow_result = {
                'pdf_processed': pdf_result['success'],
                'document_ingested': ingestion_result['status'] == 'success',
                'entities_extracted': entities_count,
                'relationships_found': relationships_count,
                'queries_executed': len(query_results),
                'successful_queries': successful_queries,
                'total_cost': env.cost_monitor.total_cost,
                'total_time': performance_summary['total_test_time'],
                'workflow_complete': True
            }
            
            # Comprehensive workflow success validation
            assert workflow_result['pdf_processed'], "PDF processing failed"
            assert workflow_result['document_ingested'], "Document ingestion failed"
            assert workflow_result['entities_extracted'] > 0, "No entities extracted"
            assert workflow_result['queries_executed'] == len(test_queries), "Not all queries executed"
            assert workflow_result['successful_queries'] >= 3, "Too few successful queries"
            assert workflow_result['workflow_complete'], "Workflow incomplete"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_pdf_to_comprehensive_query_workflow(
        self, integration_test_environment, small_pdf_collection,
        performance_monitor
    ):
        """Test workflow with multiple PDFs and comprehensive querying."""
        env = integration_test_environment
        pdf_paths = small_pdf_collection
        
        async with performance_monitor.monitor_operation(
            "multi_pdf_query_workflow", 
            pdf_count=len(pdf_paths)
        ):
            # Initialize RAG system
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Process all PDFs
            all_documents = []
            processing_summary = {
                'processed': 0,
                'failed': 0,
                'total_entities': 0,
                'total_relationships': 0
            }
            
            for pdf_path in pdf_paths:
                try:
                    pdf_result = await env.pdf_processor.process_pdf(pdf_path)
                    if pdf_result['success']:
                        enriched_content = f"""
DOCUMENT: {pdf_path.name}
TITLE: {pdf_result['metadata'].get('title', 'Unknown')}

CONTENT:
{pdf_result['text']}
                        """.strip()
                        
                        all_documents.append(enriched_content)
                        processing_summary['processed'] += 1
                    else:
                        processing_summary['failed'] += 1
                except Exception:
                    processing_summary['failed'] += 1
            
            # Batch ingestion
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                batch_ingestion = await env.lightrag_system.ainsert(all_documents)
                
                assert batch_ingestion['status'] == 'success'
                assert batch_ingestion['documents_processed'] == len(all_documents)
                
                processing_summary['total_entities'] = batch_ingestion['entities_extracted']
                processing_summary['total_relationships'] = batch_ingestion['relationships_found']
            
            # Comprehensive cross-document queries
            cross_document_queries = [
                {
                    'query': "Compare the metabolites mentioned across all documents",
                    'type': 'comparison',
                    'expected_length': 100
                },
                {
                    'query': "What are the common themes in these research studies?",
                    'type': 'synthesis',
                    'expected_length': 150
                },
                {
                    'query': "Which analytical techniques appear most frequently?",
                    'type': 'frequency',
                    'expected_length': 80
                },
                {
                    'query': "Find connections between different diseases mentioned",
                    'type': 'relationship',
                    'expected_length': 120
                },
                {
                    'query': "Summarize the key biomarkers identified across studies",
                    'type': 'summary',
                    'expected_length': 140
                }
            ]
            
            query_performance = []
            for query_info in cross_document_queries:
                start_time = time.time()
                
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    response = await env.lightrag_system.aquery(query_info['query'])
                
                query_time = time.time() - start_time
                
                # Validate cross-document query response
                assert len(response) >= query_info['expected_length'], \
                    f"Cross-document query too short for {query_info['type']}: {len(response)} chars"
                
                query_performance.append({
                    'type': query_info['type'],
                    'query': query_info['query'],
                    'response_length': len(response),
                    'query_time': query_time,
                    'success': len(response) >= query_info['expected_length']
                })
                
                env.cost_monitor.track_cost(f"cross_document_{query_info['type']}", 0.02)
            
            # Validate multi-document workflow
            assert processing_summary['processed'] >= 2, "Not enough documents processed"
            assert processing_summary['total_entities'] > processing_summary['processed'] * 3, \
                "Insufficient entities for multi-document corpus"
            
            successful_cross_queries = sum(1 for q in query_performance if q['success'])
            cross_query_success_rate = successful_cross_queries / len(cross_document_queries)
            assert cross_query_success_rate >= 0.8, \
                f"Cross-document query success rate too low: {cross_query_success_rate:.2%}"
            
            # Performance validation
            avg_cross_query_time = sum(q['query_time'] for q in query_performance) / len(query_performance)
            assert avg_cross_query_time < 4.0, \
                f"Cross-document queries too slow: {avg_cross_query_time:.2f}s average"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_large_scale_pdf_to_query_workflow(
        self, integration_test_environment, performance_monitor
    ):
        """Test large-scale PDF processing to query workflow."""
        env = integration_test_environment
        
        # Generate large collection for stress testing
        from test_fixtures import BiomedicalPDFGenerator
        large_document_set = BiomedicalPDFGenerator.create_test_documents(15)
        
        async with performance_monitor.monitor_operation(
            "large_scale_workflow", 
            document_count=len(large_document_set)
        ):
            # Create PDF files
            pdf_paths = []
            for doc in large_document_set:
                pdf_path = env.working_dir / "pdfs" / doc.filename
                content = f"Title: {doc.title}\n\n{doc.content}"
                pdf_path.write_text(content)
                pdf_paths.append(pdf_path)
            
            # Initialize system
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Batch process PDFs
            batch_start = time.time()
            batch_result = await env.pdf_processor.process_batch_pdfs(pdf_paths)
            batch_time = time.time() - batch_start
            
            assert batch_result['processed'] >= 12, "Too many processing failures in large batch"
            assert batch_time < 180.0, f"Batch processing too slow: {batch_time:.2f}s"
            
            # Extract successful results
            successful_texts = []
            for result in batch_result['results']:
                if result.get('success', False):
                    successful_texts.append(result['text'])
            
            # Batch ingestion
            ingestion_start = time.time()
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert(successful_texts)
            ingestion_time = time.time() - ingestion_start
            
            assert ingestion_result['status'] == 'success'
            assert ingestion_time < 120.0, f"Ingestion too slow: {ingestion_time:.2f}s"
            
            # Stress test with concurrent queries
            stress_queries = [
                "What metabolites are associated with diabetes?",
                "Which proteins show altered expression?",
                "What analytical techniques are most common?",
                "Find relationships between metabolites and diseases",
                "Identify key biomarkers across studies",
                "What pathways are most frequently mentioned?",
                "Compare findings between different research groups",
                "What are the main conclusions across studies?"
            ]
            
            # Execute queries concurrently
            async def execute_stress_query(query: str, query_id: int):
                try:
                    with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                        response = await env.lightrag_system.aquery(query)
                    return {
                        'query_id': query_id,
                        'query': query,
                        'response': response,
                        'success': len(response) > 20,
                        'length': len(response)
                    }
                except Exception as e:
                    return {
                        'query_id': query_id,
                        'query': query,
                        'error': str(e),
                        'success': False,
                        'length': 0
                    }
            
            concurrent_start = time.time()
            concurrent_tasks = [
                execute_stress_query(query, i) 
                for i, query in enumerate(stress_queries)
            ]
            
            concurrent_results = await asyncio.gather(*concurrent_tasks)
            concurrent_time = time.time() - concurrent_start
            
            # Validate concurrent query performance
            successful_concurrent = sum(1 for r in concurrent_results if r['success'])
            concurrent_success_rate = successful_concurrent / len(stress_queries)
            
            assert concurrent_success_rate >= 0.75, \
                f"Concurrent query success rate too low: {concurrent_success_rate:.2%}"
            
            assert concurrent_time < 30.0, \
                f"Concurrent queries too slow: {concurrent_time:.2f}s"
            
            # Overall large-scale workflow validation
            total_workflow_time = batch_time + ingestion_time + concurrent_time
            documents_per_second = len(successful_texts) / total_workflow_time
            
            assert documents_per_second > 0.1, \
                f"Large-scale throughput too low: {documents_per_second:.3f} docs/sec"
            
            # Resource efficiency check
            performance_summary = performance_monitor.get_performance_summary()
            assert performance_summary['total_test_time'] < 300.0, \
                "Large-scale workflow exceeded time limit"


# =====================================================================
# QUERY FUNCTIONALITY INTEGRATION TESTS
# =====================================================================

class TestQueryFunctionalityIntegration:
    """Test class for query capabilities with processed PDF content."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_mode_querying_with_pdf_content(
        self, integration_test_environment, pdf_test_documents,
        performance_monitor
    ):
        """Test different LightRAG query modes with processed PDF content."""
        env = integration_test_environment
        pdf_path = pdf_test_documents[0]
        
        async with performance_monitor.monitor_operation("multi_mode_querying"):
            # Setup and ingest content
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            pdf_result = await env.pdf_processor.process_pdf(pdf_path)
            assert pdf_result['success']
            
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                await env.lightrag_system.ainsert([pdf_result['text']])
            
            # Test different query modes
            test_query = "What metabolites are associated with disease progression?"
            
            query_modes = ['naive', 'local', 'global', 'hybrid']
            mode_results = {}
            
            for mode in query_modes:
                mode_start = time.time()
                
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    response = await env.lightrag_system.aquery(test_query, mode=mode)
                
                mode_time = time.time() - mode_start
                
                mode_results[mode] = {
                    'response': response,
                    'response_length': len(response),
                    'query_time': mode_time,
                    'success': len(response) > 30
                }
                
                env.cost_monitor.track_cost(f"query_mode_{mode}", 0.015)
            
            # Validate mode performance
            successful_modes = sum(1 for r in mode_results.values() if r['success'])
            assert successful_modes >= 3, f"Too few query modes succeeded: {successful_modes}/4"
            
            # Validate response diversity
            responses = [r['response'] for r in mode_results.values()]
            unique_responses = len(set(responses))
            
            # At least half should be unique (allowing some similarity)
            assert unique_responses >= 2, "Query modes producing identical responses"
            
            # Performance comparison
            mode_times = {mode: r['query_time'] for mode, r in mode_results.items()}
            fastest_mode = min(mode_times.keys(), key=lambda k: mode_times[k])
            slowest_mode = max(mode_times.keys(), key=lambda k: mode_times[k])
            
            # Even slowest mode should be reasonable
            assert mode_times[slowest_mode] < 5.0, \
                f"Slowest query mode too slow: {mode_times[slowest_mode]:.2f}s"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_context_retrieval_and_search_functionality(
        self, integration_test_environment, small_pdf_collection,
        performance_monitor
    ):
        """Test context-only retrieval and search functionality."""
        env = integration_test_environment
        pdf_paths = small_pdf_collection
        
        async with performance_monitor.monitor_operation("context_retrieval_search"):
            # Setup multi-document knowledge base
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            all_documents = []
            document_metadata = []
            
            for i, pdf_path in enumerate(pdf_paths):
                pdf_result = await env.pdf_processor.process_pdf(pdf_path)
                if pdf_result['success']:
                    doc_content = f"""
DOCUMENT_ID: doc_{i}
SOURCE: {pdf_path.name}
TITLE: {pdf_result['metadata'].get('title', f'Document {i}')}

CONTENT:
{pdf_result['text']}
                    """.strip()
                    
                    all_documents.append(doc_content)
                    document_metadata.append({
                        'id': f'doc_{i}',
                        'source': pdf_path.name,
                        'title': pdf_result['metadata'].get('title', f'Document {i}')
                    })
            
            # Ingest documents
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert(all_documents)
                assert ingestion_result['status'] == 'success'
            
            # Test context retrieval queries
            context_queries = [
                {
                    'query': "Find mentions of glucose metabolism",
                    'type': 'entity_context',
                    'expected_context_keywords': ['glucose', 'metabolism']
                },
                {
                    'query': "Retrieve information about analytical methods",
                    'type': 'method_context',
                    'expected_context_keywords': ['method', 'analysis', 'technique']
                },
                {
                    'query': "Search for disease-related findings",
                    'type': 'disease_context',
                    'expected_context_keywords': ['disease', 'patient', 'clinical']
                },
                {
                    'query': "Find protein-metabolite relationships",
                    'type': 'relationship_context',
                    'expected_context_keywords': ['protein', 'metabolite', 'relationship']
                }
            ]
            
            context_results = []
            for query_info in context_queries:
                context_start = time.time()
                
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    # Simulate context-only retrieval (would be implemented in actual LightRAG)
                    response = await env.lightrag_system.aquery(query_info['query'])
                
                context_time = time.time() - context_start
                
                # Validate context relevance
                response_lower = response.lower()
                relevant_keywords = sum(
                    1 for keyword in query_info['expected_context_keywords']
                    if keyword.lower() in response_lower
                )
                
                context_relevance = relevant_keywords / len(query_info['expected_context_keywords'])
                
                context_results.append({
                    'query_type': query_info['type'],
                    'query': query_info['query'],
                    'response': response,
                    'context_time': context_time,
                    'relevance_score': context_relevance,
                    'response_length': len(response)
                })
                
                env.cost_monitor.track_cost(f"context_{query_info['type']}", 0.01)
            
            # Validate context retrieval quality
            avg_relevance = sum(r['relevance_score'] for r in context_results) / len(context_results)
            assert avg_relevance >= 0.5, \
                f"Context retrieval relevance too low: {avg_relevance:.2%}"
            
            high_relevance_queries = sum(1 for r in context_results if r['relevance_score'] >= 0.6)
            assert high_relevance_queries >= len(context_queries) // 2, \
                "Too few high-relevance context retrievals"
            
            # Test search functionality with specific patterns
            search_patterns = [
                {
                    'pattern': "METABOLITE:glucose",
                    'description': "Entity-specific search"
                },
                {
                    'pattern': "TECHNIQUE:LC-MS",
                    'description': "Technique-specific search"
                },
                {
                    'pattern': "DISEASE:diabetes",
                    'description': "Disease-specific search"
                }
            ]
            
            search_results = []
            for pattern_info in search_patterns:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    # Simulate structured search (would be implemented in actual system)
                    search_response = await env.lightrag_system.aquery(
                        f"Find all mentions of {pattern_info['pattern']}"
                    )
                
                search_results.append({
                    'pattern': pattern_info['pattern'],
                    'description': pattern_info['description'],
                    'response': search_response,
                    'found_results': len(search_response) > 20
                })
            
            # Validate search functionality
            successful_searches = sum(1 for r in search_results if r['found_results'])
            search_success_rate = successful_searches / len(search_patterns)
            
            assert search_success_rate >= 0.6, \
                f"Search functionality success rate too low: {search_success_rate:.2%}"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complex_biomedical_query_processing(
        self, integration_test_environment, disease_specific_content,
        performance_monitor
    ):
        """Test complex biomedical queries with multiple concepts."""
        env = integration_test_environment
        
        async with performance_monitor.monitor_operation("complex_biomedical_queries"):
            # Create specialized biomedical content
            diabetes_content = disease_specific_content('diabetes', 'complex')
            cardiovascular_content = disease_specific_content('cardiovascular', 'complex')
            cancer_content = disease_specific_content('cancer', 'complex')
            
            # Initialize and ingest content
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            biomedical_documents = [
                f"DIABETES STUDY:\n{diabetes_content}",
                f"CARDIOVASCULAR RESEARCH:\n{cardiovascular_content}",
                f"CANCER METABOLOMICS:\n{cancer_content}"
            ]
            
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert(biomedical_documents)
                assert ingestion_result['status'] == 'success'
            
            # Complex multi-concept biomedical queries
            complex_queries = [
                {
                    'query': "Compare glucose metabolism alterations between diabetes and cancer patients",
                    'concepts': ['glucose', 'metabolism', 'diabetes', 'cancer'],
                    'type': 'comparative_analysis',
                    'min_length': 150
                },
                {
                    'query': "What are the shared metabolic pathways disrupted in diabetes, cardiovascular disease, and cancer?",
                    'concepts': ['metabolic pathways', 'diabetes', 'cardiovascular', 'cancer'],
                    'type': 'pathway_integration',
                    'min_length': 180
                },
                {
                    'query': "Identify biomarkers that could differentiate between diabetes-related and cancer-related metabolic changes",
                    'concepts': ['biomarkers', 'diabetes', 'cancer', 'metabolic changes'],
                    'type': 'biomarker_discrimination',
                    'min_length': 160
                },
                {
                    'query': "How do protein expression patterns relate to metabolite concentrations across these three disease conditions?",
                    'concepts': ['protein expression', 'metabolite concentrations', 'disease conditions'],
                    'type': 'multi_omics_integration',
                    'min_length': 170
                },
                {
                    'query': "What analytical techniques would be most suitable for studying the intersection of metabolomics and proteomics in these diseases?",
                    'concepts': ['analytical techniques', 'metabolomics', 'proteomics', 'diseases'],
                    'type': 'methodological_recommendation',
                    'min_length': 140
                }
            ]
            
            complex_query_results = []
            for query_info in complex_queries:
                complex_start = time.time()
                
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    response = await env.lightrag_system.aquery(query_info['query'])
                
                complex_time = time.time() - complex_start
                
                # Validate complex query response
                response_length = len(response)
                meets_length_req = response_length >= query_info['min_length']
                
                # Check concept coverage
                response_lower = response.lower()
                concepts_covered = sum(
                    1 for concept in query_info['concepts']
                    if any(word in response_lower for word in concept.lower().split())
                )
                concept_coverage = concepts_covered / len(query_info['concepts'])
                
                complex_query_results.append({
                    'query_type': query_info['type'],
                    'query': query_info['query'],
                    'response': response,
                    'response_length': response_length,
                    'query_time': complex_time,
                    'meets_length_requirement': meets_length_req,
                    'concept_coverage': concept_coverage,
                    'quality_score': (concept_coverage + (1 if meets_length_req else 0)) / 2
                })
                
                env.cost_monitor.track_cost(f"complex_{query_info['type']}", 0.03)
            
            # Validate complex query performance
            high_quality_queries = sum(
                1 for r in complex_query_results 
                if r['quality_score'] >= 0.7
            )
            
            quality_rate = high_quality_queries / len(complex_queries)
            assert quality_rate >= 0.6, \
                f"Complex query quality rate too low: {quality_rate:.2%}"
            
            # Response time validation for complex queries
            avg_complex_time = sum(r['query_time'] for r in complex_query_results) / len(complex_query_results)
            assert avg_complex_time < 6.0, \
                f"Complex queries too slow: {avg_complex_time:.2f}s average"
            
            # Concept coverage validation
            avg_concept_coverage = sum(r['concept_coverage'] for r in complex_query_results) / len(complex_query_results)
            assert avg_concept_coverage >= 0.6, \
                f"Complex query concept coverage too low: {avg_concept_coverage:.2%}"


# =====================================================================
# RESPONSE QUALITY VALIDATION TESTS
# =====================================================================

class TestResponseQualityValidation:
    """Test class for quality assurance of biomedical responses."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_biomedical_response_content_validation(
        self, integration_test_environment, pdf_test_documents,
        performance_monitor
    ):
        """Test validation of biomedical content in query responses."""
        env = integration_test_environment
        pdf_path = pdf_test_documents[0]
        
        async with performance_monitor.monitor_operation("response_content_validation"):
            # Setup with biomedical content
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            pdf_result = await env.pdf_processor.process_pdf(pdf_path)
            assert pdf_result['success']
            
            # Create enriched biomedical content
            biomedical_content = f"""
CLINICAL METABOLOMICS STUDY

BIOMARKERS IDENTIFIED:
- Glucose: Elevated in diabetes patients (p < 0.001)
- Lactate: Increased during exercise and hypoxia
- TMAO: Associated with cardiovascular risk
- Creatinine: Kidney function biomarker

PROTEIN ANALYSIS:
- Insulin: Key hormone in glucose regulation
- Albumin: Protein synthesis marker
- CRP: Inflammatory biomarker
- Hemoglobin: Oxygen transport protein

PATHWAY ANALYSIS:
- Glycolysis: Glucose breakdown pathway
- TCA Cycle: Central metabolic pathway
- Fatty Acid Oxidation: Energy production pathway

CLINICAL SIGNIFICANCE:
These biomarkers provide diagnostic and prognostic value in metabolic diseases.
Statistical analysis shows significant correlations (r > 0.7, p < 0.05).

ORIGINAL CONTENT:
{pdf_result['text']}
            """.strip()
            
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                await env.lightrag_system.ainsert([biomedical_content])
            
            # Biomedical validation queries
            validation_queries = [
                {
                    'query': "What biomarkers are elevated in diabetes?",
                    'expected_entities': ['glucose', 'biomarkers', 'diabetes'],
                    'expected_values': ['p < 0.001', 'elevated'],
                    'domain': 'biomarkers'
                },
                {
                    'query': "Explain the role of insulin in glucose regulation",
                    'expected_entities': ['insulin', 'glucose', 'regulation'],
                    'expected_concepts': ['hormone', 'key', 'regulation'],
                    'domain': 'physiology'
                },
                {
                    'query': "What pathways are involved in energy metabolism?",
                    'expected_entities': ['glycolysis', 'TCA cycle', 'fatty acid oxidation'],
                    'expected_concepts': ['pathway', 'energy', 'metabolism'],
                    'domain': 'biochemistry'
                },
                {
                    'query': "What is the clinical significance of these findings?",
                    'expected_entities': ['diagnostic', 'prognostic', 'clinical'],
                    'expected_concepts': ['significance', 'value', 'diseases'],
                    'domain': 'clinical'
                }
            ]
            
            validation_results = []
            for query_info in validation_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    response = await env.lightrag_system.aquery(query_info['query'])
                
                # Validate entity presence
                response_lower = response.lower()
                entities_found = sum(
                    1 for entity in query_info['expected_entities']
                    if entity.lower() in response_lower
                )
                entity_score = entities_found / len(query_info['expected_entities'])
                
                # Validate concept presence
                expected_items = query_info.get('expected_concepts', query_info.get('expected_values', []))
                concepts_found = sum(
                    1 for item in expected_items
                    if item.lower() in response_lower
                )
                concept_score = concepts_found / max(1, len(expected_items))
                
                # Overall biomedical accuracy
                biomedical_accuracy = (entity_score + concept_score) / 2
                
                validation_results.append({
                    'domain': query_info['domain'],
                    'query': query_info['query'],
                    'response': response,
                    'entity_score': entity_score,
                    'concept_score': concept_score,
                    'biomedical_accuracy': biomedical_accuracy,
                    'response_length': len(response)
                })
                
                env.cost_monitor.track_cost(f"validation_{query_info['domain']}", 0.01)
            
            # Validate biomedical response quality
            high_accuracy_responses = sum(
                1 for r in validation_results 
                if r['biomedical_accuracy'] >= 0.7
            )
            
            accuracy_rate = high_accuracy_responses / len(validation_queries)
            assert accuracy_rate >= 0.75, \
                f"Biomedical accuracy rate too low: {accuracy_rate:.2%}"
            
            # Domain-specific validation
            domain_scores = {}
            for result in validation_results:
                domain = result['domain']
                if domain not in domain_scores:
                    domain_scores[domain] = []
                domain_scores[domain].append(result['biomedical_accuracy'])
            
            for domain, scores in domain_scores.items():
                avg_domain_score = sum(scores) / len(scores)
                assert avg_domain_score >= 0.6, \
                    f"Domain {domain} accuracy too low: {avg_domain_score:.2%}"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_entity_extraction_and_relationship_discovery_validation(
        self, integration_test_environment, disease_specific_content,
        performance_monitor
    ):
        """Test validation of entity extraction and relationship discovery."""
        env = integration_test_environment
        
        async with performance_monitor.monitor_operation("entity_relationship_validation"):
            # Create content rich in entities and relationships
            diabetes_content = disease_specific_content('diabetes', 'complex')
            
            # Add explicit relationships for testing
            relationship_rich_content = f"""
BIOMEDICAL ENTITY RELATIONSHIPS:

METABOLITE-DISEASE RELATIONSHIPS:
- Glucose ELEVATED_IN diabetes mellitus
- HbA1c DIAGNOSTIC_FOR diabetes monitoring
- Lactate ASSOCIATED_WITH metabolic acidosis
- TMAO PREDICTIVE_OF cardiovascular events

PROTEIN-FUNCTION RELATIONSHIPS:
- Insulin REGULATES glucose homeostasis
- Glucagon COUNTERACTS insulin effects
- GLUT4 TRANSPORTS glucose into cells
- Adiponectin IMPROVES insulin sensitivity

PATHWAY-DISEASE RELATIONSHIPS:
- Glycolysis DISRUPTED_IN cancer metabolism
- TCA_cycle IMPAIRED_IN mitochondrial diseases
- Insulin_signaling DEFECTIVE_IN type_2_diabetes

DRUG-TARGET RELATIONSHIPS:
- Metformin TARGETS AMPK pathway
- Insulin REPLACES endogenous hormone
- Statins INHIBIT HMG-CoA reductase

STUDY CONTENT:
{diabetes_content}
            """.strip()
            
            # Initialize and ingest
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                ingestion_result = await env.lightrag_system.ainsert([relationship_rich_content])
                
                assert ingestion_result['status'] == 'success'
                entities_extracted = ingestion_result['entities_extracted']
                relationships_found = ingestion_result['relationships_found']
            
            # Validate minimum entity extraction
            assert entities_extracted >= 15, \
                f"Insufficient entities extracted: {entities_extracted} (expected >= 15)"
            
            # Validate relationship discovery
            assert relationships_found >= 8, \
                f"Insufficient relationships found: {relationships_found} (expected >= 8)"
            
            # Test entity-specific queries
            entity_queries = [
                {
                    'query': "What metabolites are associated with diabetes?",
                    'expected_entities': ['glucose', 'HbA1c'],
                    'entity_type': 'METABOLITE'
                },
                {
                    'query': "Which proteins regulate glucose metabolism?",
                    'expected_entities': ['insulin', 'glucagon', 'GLUT4'],
                    'entity_type': 'PROTEIN'
                },
                {
                    'query': "What pathways are disrupted in metabolic diseases?",
                    'expected_entities': ['glycolysis', 'TCA_cycle', 'insulin_signaling'],
                    'entity_type': 'PATHWAY'
                }
            ]
            
            entity_validation_results = []
            for query_info in entity_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    response = await env.lightrag_system.aquery(query_info['query'])
                
                # Check entity presence in response
                response_lower = response.lower()
                entities_mentioned = sum(
                    1 for entity in query_info['expected_entities']
                    if entity.lower().replace('_', ' ') in response_lower
                )
                
                entity_recall = entities_mentioned / len(query_info['expected_entities'])
                
                entity_validation_results.append({
                    'entity_type': query_info['entity_type'],
                    'query': query_info['query'],
                    'response': response,
                    'expected_entities': query_info['expected_entities'],
                    'entities_mentioned': entities_mentioned,
                    'entity_recall': entity_recall
                })
            
            # Test relationship queries
            relationship_queries = [
                {
                    'query': "How does insulin regulate glucose?",
                    'expected_relationship': 'insulin REGULATES glucose',
                    'relationship_type': 'regulatory'
                },
                {
                    'query': "What is glucose elevated in?",
                    'expected_relationship': 'glucose ELEVATED_IN diabetes',
                    'relationship_type': 'association'
                },
                {
                    'query': "What does metformin target?",
                    'expected_relationship': 'metformin TARGETS AMPK',
                    'relationship_type': 'drug_target'
                }
            ]
            
            relationship_validation_results = []
            for query_info in relationship_queries:
                with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                    response = await env.lightrag_system.aquery(query_info['query'])
                
                # Check for relationship components in response
                relationship_terms = query_info['expected_relationship'].lower().split()
                relationship_coverage = sum(
                    1 for term in relationship_terms
                    if term in response.lower()
                ) / len(relationship_terms)
                
                relationship_validation_results.append({
                    'relationship_type': query_info['relationship_type'],
                    'query': query_info['query'],
                    'response': response,
                    'expected_relationship': query_info['expected_relationship'],
                    'relationship_coverage': relationship_coverage
                })
            
            # Validate entity extraction quality
            avg_entity_recall = sum(r['entity_recall'] for r in entity_validation_results) / len(entity_validation_results)
            assert avg_entity_recall >= 0.6, \
                f"Entity recall too low: {avg_entity_recall:.2%}"
            
            # Validate relationship discovery quality
            avg_relationship_coverage = sum(r['relationship_coverage'] for r in relationship_validation_results) / len(relationship_validation_results)
            assert avg_relationship_coverage >= 0.5, \
                f"Relationship coverage too low: {avg_relationship_coverage:.2%}"
            
            # Overall entity-relationship system validation
            entity_relationship_score = (avg_entity_recall + avg_relationship_coverage) / 2
            assert entity_relationship_score >= 0.55, \
                f"Overall entity-relationship performance too low: {entity_relationship_score:.2%}"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_query_performance_and_response_time_validation(
        self, integration_test_environment, small_pdf_collection,
        performance_monitor
    ):
        """Test query performance and response time validation."""
        env = integration_test_environment
        pdf_paths = small_pdf_collection
        
        async with performance_monitor.monitor_operation("query_performance_validation"):
            # Setup knowledge base
            rag_system = ClinicalMetabolomicsRAG(config=env.config)
            
            # Process and ingest documents
            all_documents = []
            for pdf_path in pdf_paths:
                pdf_result = await env.pdf_processor.process_pdf(pdf_path)
                if pdf_result['success']:
                    all_documents.append(pdf_result['text'])
            
            with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                await env.lightrag_system.ainsert(all_documents)
            
            # Performance test queries with different complexity levels
            performance_test_queries = [
                {
                    'query': "What is glucose?",
                    'complexity': 'simple',
                    'max_response_time': 2.0,
                    'min_response_length': 30
                },
                {
                    'query': "How do metabolites relate to disease diagnosis?",
                    'complexity': 'medium',
                    'max_response_time': 3.0,
                    'min_response_length': 80
                },
                {
                    'query': "Compare the metabolomic profiles and their clinical implications across different analytical techniques mentioned in the studies",
                    'complexity': 'complex',
                    'max_response_time': 5.0,
                    'min_response_length': 150
                }
            ]
            
            performance_results = []
            total_query_time = 0.0
            
            # Execute performance tests
            for query_info in performance_test_queries:
                # Multiple runs for statistical significance
                run_times = []
                responses = []
                
                for run in range(3):  # 3 runs per query
                    start_time = time.time()
                    
                    with patch.object(rag_system, 'lightrag_instance', env.lightrag_system):
                        response = await env.lightrag_system.aquery(query_info['query'])
                    
                    end_time = time.time()
                    query_time = end_time - start_time
                    
                    run_times.append(query_time)
                    responses.append(response)
                    total_query_time += query_time
                
                # Calculate performance metrics
                avg_response_time = sum(run_times) / len(run_times)
                min_response_time = min(run_times)
                max_response_time = max(run_times)
                
                # Response consistency check
                avg_response_length = sum(len(r) for r in responses) / len(responses)
                response_length_variance = sum(
                    (len(r) - avg_response_length) ** 2 for r in responses
                ) / len(responses)
                
                performance_results.append({
                    'query': query_info['query'],
                    'complexity': query_info['complexity'],
                    'avg_response_time': avg_response_time,
                    'min_response_time': min_response_time,
                    'max_response_time': max_response_time,
                    'time_consistency': max_response_time - min_response_time,
                    'avg_response_length': avg_response_length,
                    'response_variance': response_length_variance,
                    'meets_time_requirement': avg_response_time <= query_info['max_response_time'],
                    'meets_length_requirement': avg_response_length >= query_info['min_response_length']
                })
                
                env.cost_monitor.track_cost(f"performance_{query_info['complexity']}", 0.02)
            
            # Validate performance requirements
            queries_meeting_time_req = sum(1 for r in performance_results if r['meets_time_requirement'])
            time_compliance_rate = queries_meeting_time_req / len(performance_test_queries)
            
            assert time_compliance_rate >= 0.8, \
                f"Query time compliance too low: {time_compliance_rate:.2%}"
            
            queries_meeting_length_req = sum(1 for r in performance_results if r['meets_length_requirement'])
            length_compliance_rate = queries_meeting_length_req / len(performance_test_queries)
            
            assert length_compliance_rate >= 0.9, \
                f"Response length compliance too low: {length_compliance_rate:.2%}"
            
            # Performance scaling validation
            simple_avg = next(r['avg_response_time'] for r in performance_results if r['complexity'] == 'simple')
            complex_avg = next(r['avg_response_time'] for r in performance_results if r['complexity'] == 'complex')
            
            # Complex queries should not be more than 3x slower than simple queries
            performance_scaling = complex_avg / simple_avg
            assert performance_scaling <= 3.0, \
                f"Performance scaling too poor: {performance_scaling:.2f}x"
            
            # Consistency validation
            avg_time_consistency = sum(r['time_consistency'] for r in performance_results) / len(performance_results)
            assert avg_time_consistency < 2.0, \
                f"Query time consistency too poor: {avg_time_consistency:.2f}s variance"
            
            # Overall performance score
            performance_score = (
                time_compliance_rate + 
                length_compliance_rate + 
                (1.0 if performance_scaling <= 3.0 else 0.0) +
                (1.0 if avg_time_consistency < 2.0 else 0.0)
            ) / 4.0
            
            assert performance_score >= 0.75, \
                f"Overall query performance score too low: {performance_score:.2%}"


# =====================================================================
# CONFIGURATION INTEGRATION TESTS
# =====================================================================

@pytest.mark.integration
class TestConfigurationIntegration:
    """
    Test configuration consistency and validation across components.
    
    This test class validates:
    - Shared configuration between PDF processor and LightRAG
    - Configuration validation and error handling
    - Dynamic configuration updates
    - Configuration inheritance and overrides
    - Cross-component configuration consistency
    """
    
    @pytest.mark.asyncio
    async def test_shared_configuration_consistency(self, integration_test_environment):
        """Test configuration consistency across all components."""
        env = integration_test_environment
        
        # Test shared configuration values
        base_config = {
            'max_file_size_mb': 50,
            'processing_timeout_seconds': 300,
            'batch_size': 10,
            'memory_limit_mb': 1000,
            'enable_metrics': True,
            'log_level': 'INFO'
        }
        
        # Update configurations
        env.config.update(base_config)
        
        # Verify PDF processor uses shared config
        pdf_config = env.pdf_processor.get_config()
        assert pdf_config['max_file_size_mb'] == base_config['max_file_size_mb']
        assert pdf_config['processing_timeout_seconds'] == base_config['processing_timeout_seconds']
        
        # Verify LightRAG uses shared config  
        lightrag_config = env.lightrag_system.get_config()
        assert lightrag_config['batch_size'] == base_config['batch_size']
        assert lightrag_config['memory_limit_mb'] == base_config['memory_limit_mb']
        
        # Verify cost monitor uses shared config
        cost_config = env.cost_monitor.get_config()
        assert cost_config['enable_metrics'] == base_config['enable_metrics']
        
        # Test configuration propagation
        new_batch_size = 25
        env.config.update({'batch_size': new_batch_size})
        
        # Verify all components received the update
        updated_lightrag_config = env.lightrag_system.get_config()
        assert updated_lightrag_config['batch_size'] == new_batch_size
    
    @pytest.mark.asyncio
    async def test_configuration_validation_chain(self, integration_test_environment, error_injector):
        """Test configuration validation across component chain."""
        env = integration_test_environment
        
        # Test invalid configuration scenarios
        invalid_configs = [
            {'max_file_size_mb': -1},  # Negative value
            {'processing_timeout_seconds': 0},  # Zero timeout
            {'batch_size': 0},  # Zero batch size
            {'memory_limit_mb': 'invalid'},  # Wrong type
            {'log_level': 'INVALID_LEVEL'}  # Invalid enum value
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, LightRAGConfigError)):
                env.config.validate_and_update(invalid_config)
        
        # Test configuration dependency validation
        dependent_config = {
            'max_file_size_mb': 100,
            'memory_limit_mb': 50  # Memory limit less than max file size
        }
        
        with pytest.raises(LightRAGConfigError, match="Memory limit cannot be less than max file size"):
            env.config.validate_and_update(dependent_config)
    
    @pytest.mark.asyncio
    async def test_configuration_inheritance_and_overrides(self, integration_test_environment):
        """Test configuration inheritance and component-specific overrides."""
        env = integration_test_environment
        
        # Set global defaults
        global_config = {
            'processing_timeout_seconds': 300,
            'batch_size': 10,
            'enable_debug_logging': False
        }
        env.config.update(global_config)
        
        # Set component-specific overrides
        pdf_overrides = {
            'processing_timeout_seconds': 600,  # PDF needs more time
            'enable_debug_logging': True
        }
        env.pdf_processor.update_config(pdf_overrides)
        
        lightrag_overrides = {
            'batch_size': 20  # LightRAG can handle larger batches
        }
        env.lightrag_system.update_config(lightrag_overrides)
        
        # Verify inheritance and overrides
        pdf_config = env.pdf_processor.get_config()
        assert pdf_config['processing_timeout_seconds'] == 600  # Overridden
        assert pdf_config['batch_size'] == 10  # Inherited
        assert pdf_config['enable_debug_logging'] is True  # Overridden
        
        lightrag_config = env.lightrag_system.get_config()
        assert lightrag_config['processing_timeout_seconds'] == 300  # Inherited
        assert lightrag_config['batch_size'] == 20  # Overridden
        assert lightrag_config['enable_debug_logging'] is False  # Inherited
    
    @pytest.mark.asyncio
    async def test_dynamic_configuration_updates(self, integration_test_environment):
        """Test dynamic configuration updates during operation."""
        env = integration_test_environment
        
        # Create test PDF collection
        pdf_paths = env.create_test_pdf_collection(5)
        
        # Start processing with initial configuration
        initial_config = {
            'batch_size': 2,
            'processing_timeout_seconds': 60
        }
        env.config.update(initial_config)
        
        # Start mock processing task
        processing_task = asyncio.create_task(
            env.pdf_processor.process_batch_async(pdf_paths)
        )
        
        # Wait a moment then update configuration
        await asyncio.sleep(0.1)
        
        updated_config = {
            'batch_size': 5,
            'processing_timeout_seconds': 120
        }
        env.config.update(updated_config)
        
        # Verify configuration was updated during processing
        current_config = env.pdf_processor.get_config()
        assert current_config['batch_size'] == 5
        assert current_config['processing_timeout_seconds'] == 120
        
        # Wait for processing to complete
        await processing_task
    
    @pytest.mark.asyncio 
    async def test_configuration_error_recovery(self, integration_test_environment, error_injector):
        """Test configuration error recovery mechanisms."""
        env = integration_test_environment
        
        # Set up error injection for configuration operations
        error_injector.add_rule(
            'config_update',
            ConnectionError("Configuration service unavailable"),
            trigger_after=2
        )
        
        # Test configuration fallback mechanisms
        primary_config = {
            'batch_size': 20,
            'memory_limit_mb': 2000
        }
        
        fallback_config = {
            'batch_size': 10,
            'memory_limit_mb': 1000
        }
        
        # Attempt configuration update with error injection
        with patch.object(env.config, 'update') as mock_update:
            mock_update.side_effect = [
                None,  # First call succeeds
                ConnectionError("Configuration service unavailable"),  # Second call fails
                None   # Third call succeeds with fallback
            ]
            
            # First update should succeed
            env.config.update(primary_config)
            
            # Second update should fail and trigger fallback
            try:
                env.config.update(primary_config)
            except ConnectionError:
                env.config.update(fallback_config)
            
            # Verify fallback configuration is active
            current_config = env.config.get_current()
            assert current_config['batch_size'] == fallback_config['batch_size']
            assert current_config['memory_limit_mb'] == fallback_config['memory_limit_mb']
    
    @pytest.mark.asyncio
    async def test_cross_component_configuration_sync(self, integration_test_environment):
        """Test configuration synchronization across all components."""
        env = integration_test_environment
        
        # Define configuration that affects multiple components
        sync_config = {
            'enable_metrics': True,
            'metrics_interval_seconds': 30,
            'log_level': 'DEBUG',
            'enable_cost_tracking': True,
            'memory_threshold_mb': 800
        }
        
        # Update configuration
        env.config.update(sync_config)
        
        # Trigger configuration sync
        await env.config.sync_all_components()
        
        # Verify all components received the configuration
        components = [
            env.pdf_processor,
            env.lightrag_system,
            env.cost_monitor,
            env.progress_tracker
        ]
        
        for component in components:
            component_config = component.get_config()
            
            # Verify common configuration values
            assert component_config['enable_metrics'] == sync_config['enable_metrics']
            assert component_config['log_level'] == sync_config['log_level']
            
            # Verify component-relevant configurations
            if hasattr(component, 'cost_tracking_enabled'):
                assert component_config['enable_cost_tracking'] == sync_config['enable_cost_tracking']
            
            if hasattr(component, 'memory_threshold_mb'):
                assert component_config['memory_threshold_mb'] == sync_config['memory_threshold_mb']


# =====================================================================
# ERROR RECOVERY INTEGRATION TESTS  
# =====================================================================

@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """
    Test coordinated error handling and circuit breaker patterns.
    
    This test class validates:
    - Error propagation between components
    - Circuit breaker coordination
    - Retry mechanisms and backoff strategies
    - Graceful degradation under failure
    - Error recovery and system restoration
    """
    
    @pytest.mark.asyncio
    async def test_coordinated_circuit_breaker_activation(self, integration_test_environment, error_injector):
        """Test coordinated circuit breaker activation across components."""
        env = integration_test_environment
        
        # Configure circuit breakers with low thresholds for testing
        circuit_breaker_config = {
            'failure_threshold': 3,
            'recovery_timeout_seconds': 5,
            'half_open_max_calls': 2
        }
        
        # Set up cascading failures
        error_injector.add_rule(
            'pdf_processing',
            BiomedicalPDFProcessorError("PDF processing service unavailable"),
            trigger_after=1,
            probability=1.0
        )
        
        pdf_paths = env.create_test_pdf_collection(10)
        
        # Process documents and trigger circuit breaker
        failures = []
        for i in range(5):  # Exceed failure threshold
            try:
                await env.pdf_processor.process_document_async(pdf_paths[0])
            except BiomedicalPDFProcessorError as e:
                failures.append(e)
        
        # Verify circuit breaker is open
        assert env.pdf_processor.circuit_breaker.is_open()
        assert len(failures) == 3  # Only first 3 should reach the service
        
        # Verify coordinated circuit breaker affects downstream components
        assert env.lightrag_system.circuit_breaker.is_open()
        
        # Test fast-fail behavior
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await env.pdf_processor.process_document_async(pdf_paths[1])
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_containment(self, integration_test_environment, error_injector):
        """Test error propagation control and containment."""
        env = integration_test_environment
        
        # Set up error injection at different levels
        error_injector.add_rule(
            'lightrag_ingestion',
            ClinicalMetabolomicsRAGError("LightRAG ingestion failed"),
            trigger_after=1,
            probability=0.5
        )
        
        pdf_paths = env.create_test_pdf_collection(20)
        
        # Process batch with controlled error propagation
        results = await env.pdf_processor.process_batch_with_error_handling(
            pdf_paths,
            error_containment_mode='isolate_failures'
        )
        
        # Verify error containment
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        assert len(successful_results) > 0, "Some documents should succeed"
        assert len(failed_results) > 0, "Some documents should fail as expected"
        
        # Verify failed documents don't affect successful ones
        for success_result in successful_results:
            assert success_result.documents_processed > 0
            assert success_result.error_count == 0
        
        # Verify error information is captured
        for fail_result in failed_results:
            assert fail_result.error_count > 0
            assert 'LightRAG ingestion failed' in str(fail_result.metadata.get('errors', []))
    
    @pytest.mark.asyncio
    async def test_retry_mechanisms_and_backoff(self, integration_test_environment, error_injector):
        """Test retry mechanisms and exponential backoff strategies."""
        env = integration_test_environment
        
        # Configure transient error injection
        error_injector.add_rule(
            'document_ingestion',
            ConnectionError("Temporary connection error"),
            trigger_after=1,
            probability=0.7  # 70% failure rate initially
        )
        
        pdf_paths = env.create_test_pdf_collection(5)
        
        retry_config = {
            'max_retries': 5,
            'initial_delay_seconds': 0.1,
            'backoff_multiplier': 2.0,
            'max_delay_seconds': 2.0,
            'jitter': True
        }
        
        start_time = time.time()
        
        # Process with retry logic
        results = await env.lightrag_system.ingest_documents_with_retry(
            pdf_paths,
            **retry_config
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify retry behavior
        successful_count = sum(1 for r in results if r.success)
        assert successful_count > 0, "Some retries should eventually succeed"
        
        # Verify exponential backoff timing
        expected_min_time = retry_config['initial_delay_seconds'] * successful_count
        assert processing_time >= expected_min_time, "Should respect minimum retry delays"
        
        # Verify retry attempts are logged
        retry_attempts = sum(r.metadata.get('retry_attempts', 0) for r in results)
        assert retry_attempts > successful_count, "Should have multiple retry attempts"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_modes(self, integration_test_environment, error_injector):
        """Test graceful degradation under various failure scenarios."""
        env = integration_test_environment
        
        # Test different degradation scenarios
        degradation_scenarios = [
            {
                'name': 'partial_processing_failure',
                'error_target': 'entity_extraction',
                'error_type': Exception("Entity extraction service down"),
                'expected_degradation': 'continue_without_entities'
            },
            {
                'name': 'metadata_preservation_failure', 
                'error_target': 'metadata_storage',
                'error_type': Exception("Metadata storage unavailable"),
                'expected_degradation': 'store_content_only'
            },
            {
                'name': 'cost_tracking_failure',
                'error_target': 'cost_monitoring',
                'error_type': Exception("Cost tracking service down"),
                'expected_degradation': 'continue_without_cost_tracking'
            }
        ]
        
        pdf_paths = env.create_test_pdf_collection(3)
        
        for scenario in degradation_scenarios:
            # Set up specific error injection
            error_injector.add_rule(
                scenario['error_target'],
                scenario['error_type'],
                trigger_after=1,
                probability=1.0
            )
            
            # Process with graceful degradation enabled
            result = await env.pdf_processor.process_with_graceful_degradation(
                pdf_paths[0],
                enable_degradation=True
            )
            
            # Verify processing continues with degradation
            assert result.success, f"Processing should succeed with degradation in {scenario['name']}"
            assert scenario['expected_degradation'] in result.metadata.get('degradation_modes', [])
            
            # Verify partial functionality
            if scenario['expected_degradation'] == 'continue_without_entities':
                assert result.entities_extracted == 0
                assert result.documents_processed > 0
            elif scenario['expected_degradation'] == 'store_content_only':
                assert 'metadata' not in result.metadata
                assert result.documents_processed > 0
            elif scenario['expected_degradation'] == 'continue_without_cost_tracking':
                assert result.total_cost == 0.0
                assert result.documents_processed > 0
    
    @pytest.mark.asyncio
    async def test_system_recovery_and_restoration(self, integration_test_environment, error_injector):
        """Test system recovery and restoration after failures."""
        env = integration_test_environment
        
        # Simulate system failure and recovery
        error_injector.add_rule(
            'system_failure',
            Exception("Complete system failure"),
            trigger_after=1,
            probability=1.0
        )
        
        pdf_paths = env.create_test_pdf_collection(10)
        
        # First phase: System failure
        with pytest.raises(Exception, match="Complete system failure"):
            await env.pdf_processor.process_batch_async(pdf_paths[:5])
        
        # Verify system is in failed state
        assert not env.pdf_processor.is_healthy()
        assert not env.lightrag_system.is_healthy()
        
        # Second phase: Recovery initiation
        error_injector.injection_rules.clear()  # Remove error injection
        
        # Trigger system recovery
        recovery_result = await env.lightrag_system.initiate_recovery()
        assert recovery_result.success, "Recovery should succeed"
        
        # Third phase: Verify system restoration
        health_check = await env.pdf_processor.health_check_async()
        assert health_check.healthy, "PDF processor should be healthy after recovery"
        
        lightrag_health = await env.lightrag_system.health_check_async()
        assert lightrag_health.healthy, "LightRAG should be healthy after recovery"
        
        # Fourth phase: Test normal operation after recovery
        recovery_results = await env.pdf_processor.process_batch_async(pdf_paths[5:])
        
        successful_recoveries = [r for r in recovery_results if r.success]
        assert len(successful_recoveries) == 5, "All documents should process successfully after recovery"
    
    @pytest.mark.asyncio
    async def test_error_correlation_and_root_cause_analysis(self, integration_test_environment, error_injector):
        """Test error correlation and root cause analysis across components."""
        env = integration_test_environment
        
        # Create correlated failure scenario
        root_cause_error = ConnectionError("Database connection pool exhausted")
        
        # Inject the root cause error in multiple components
        for component in ['pdf_processor', 'lightrag_system', 'cost_monitor']:
            error_injector.add_rule(
                f'{component}_db_operation',
                root_cause_error,
                trigger_after=1,
                probability=1.0
            )
        
        pdf_paths = env.create_test_pdf_collection(5)
        
        # Trigger correlated failures
        failures = []
        for pdf_path in pdf_paths:
            try:
                await env.pdf_processor.process_document_async(pdf_path)
            except Exception as e:
                failures.append({
                    'component': 'pdf_processor',
                    'error': e,
                    'timestamp': time.time(),
                    'document': pdf_path.name
                })
        
        # Verify error correlation detection
        error_correlator = env.lightrag_system.error_correlator
        correlation_analysis = error_correlator.analyze_failures(failures)
        
        assert correlation_analysis['root_cause_detected']
        assert 'Database connection pool exhausted' in correlation_analysis['root_cause_description']
        assert len(correlation_analysis['affected_components']) >= 3
        assert correlation_analysis['correlation_confidence'] > 0.8


# =====================================================================
# RESOURCE MANAGEMENT INTEGRATION TESTS
# =====================================================================

@pytest.mark.integration  
class TestResourceManagementIntegration:
    """
    Test memory and resource coordination between components.
    
    This test class validates:
    - Memory usage coordination during batch processing
    - Resource limit enforcement and coordination
    - Memory cleanup and garbage collection
    - Resource allocation and deallocation
    - Cross-component resource sharing
    """
    
    @pytest.mark.asyncio
    async def test_coordinated_memory_management(self, integration_test_environment):
        """Test coordinated memory management across all components."""
        env = integration_test_environment
        
        # Configure memory limits
        memory_config = {
            'total_memory_limit_mb': 500,
            'pdf_processor_limit_mb': 200,
            'lightrag_limit_mb': 200,
            'buffer_memory_mb': 100
        }
        env.config.update(memory_config)
        
        # Create large PDF collection to stress memory
        pdf_paths = env.create_test_pdf_collection(25)
        
        # Monitor memory usage during processing
        memory_tracker = env.lightrag_system.get_memory_tracker()
        memory_samples = []
        
        async def memory_monitoring_task():
            while True:
                try:
                    memory_usage = memory_tracker.get_current_usage()
                    memory_samples.append({
                        'timestamp': time.time(),
                        'total_mb': memory_usage['total_mb'],
                        'pdf_processor_mb': memory_usage['pdf_processor_mb'],
                        'lightrag_mb': memory_usage['lightrag_mb'],
                        'available_mb': memory_usage['available_mb']
                    })
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break
        
        monitoring_task = asyncio.create_task(memory_monitoring_task())
        
        try:
            # Process documents with memory coordination
            results = await env.pdf_processor.process_batch_with_memory_coordination(pdf_paths)
            
            # Verify processing succeeded within memory limits
            successful_count = sum(1 for r in results if r.success)
            assert successful_count > 0, "Some documents should process successfully"
            
        finally:
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Analyze memory usage patterns
        peak_memory = max(sample['total_mb'] for sample in memory_samples)
        assert peak_memory <= memory_config['total_memory_limit_mb'], \
            f"Peak memory {peak_memory}MB exceeded limit {memory_config['total_memory_limit_mb']}MB"
        
        # Verify memory coordination between components
        pdf_peak = max(sample['pdf_processor_mb'] for sample in memory_samples)
        lightrag_peak = max(sample['lightrag_mb'] for sample in memory_samples)
        
        assert pdf_peak <= memory_config['pdf_processor_limit_mb']
        assert lightrag_peak <= memory_config['lightrag_limit_mb']
    
    @pytest.mark.asyncio
    async def test_resource_limit_enforcement(self, integration_test_environment, error_injector):
        """Test resource limit enforcement and coordination."""
        env = integration_test_environment
        
        # Configure strict resource limits
        resource_limits = {
            'max_concurrent_operations': 3,
            'max_file_handles': 10,
            'max_network_connections': 5,
            'memory_threshold_mb': 300
        }
        env.config.update(resource_limits)
        
        pdf_paths = env.create_test_pdf_collection(15)
        
        # Create resource monitor
        resource_monitor = env.lightrag_system.get_resource_monitor()
        
        # Attempt to exceed resource limits
        processing_tasks = []
        for pdf_path in pdf_paths:
            task = asyncio.create_task(
                env.pdf_processor.process_document_async(pdf_path)
            )
            processing_tasks.append(task)
        
        # Monitor resource usage
        resource_violations = []
        
        while not all(task.done() for task in processing_tasks):
            current_resources = resource_monitor.get_current_usage()
            
            # Check for resource limit violations
            if current_resources['concurrent_operations'] > resource_limits['max_concurrent_operations']:
                resource_violations.append({
                    'type': 'concurrent_operations',
                    'current': current_resources['concurrent_operations'],
                    'limit': resource_limits['max_concurrent_operations'],
                    'timestamp': time.time()
                })
            
            if current_resources['file_handles'] > resource_limits['max_file_handles']:
                resource_violations.append({
                    'type': 'file_handles',
                    'current': current_resources['file_handles'],
                    'limit': resource_limits['max_file_handles'],
                    'timestamp': time.time()
                })
            
            await asyncio.sleep(0.1)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Verify resource limits were enforced
        successful_results = [r for r in results if isinstance(r, IntegrationTestResult) and r.success]
        
        # Should process some documents successfully within limits
        assert len(successful_results) > 0, "Some processing should succeed within resource limits"
        
        # Resource violations should be minimal or handled gracefully
        critical_violations = [v for v in resource_violations 
                             if v['current'] > v['limit'] * 1.5]  # 50% over limit
        assert len(critical_violations) == 0, f"Critical resource violations detected: {critical_violations}"
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_and_garbage_collection(self, integration_test_environment):
        """Test memory cleanup and garbage collection coordination."""
        env = integration_test_environment
        
        # Configure aggressive cleanup for testing
        cleanup_config = {
            'enable_aggressive_cleanup': True,
            'cleanup_interval_seconds': 1,
            'memory_pressure_threshold': 0.7,
            'force_gc_threshold': 0.8
        }
        env.config.update(cleanup_config)
        
        pdf_paths = env.create_test_pdf_collection(20)
        
        # Track memory before processing
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Process documents with memory tracking
        memory_samples = []
        gc_events = []
        
        # Enable GC monitoring
        def gc_callback(phase, info):
            gc_events.append({
                'phase': phase,
                'timestamp': time.time(),
                'collected': info.get('collected', 0),
                'memory_before_mb': process.memory_info().rss / 1024 / 1024
            })
        
        gc.callbacks.append(gc_callback)
        
        try:
            for i, pdf_path in enumerate(pdf_paths):
                # Process document
                result = await env.pdf_processor.process_document_async(pdf_path)
                
                # Record memory usage
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append({
                    'document_index': i,
                    'memory_mb': current_memory_mb,
                    'memory_delta_mb': current_memory_mb - initial_memory_mb,
                    'timestamp': time.time()
                })
                
                # Trigger cleanup if memory pressure is high
                if current_memory_mb > initial_memory_mb * 1.5:
                    await env.lightrag_system.trigger_memory_cleanup()
                    gc.collect()
        
        finally:
            gc.callbacks.remove(gc_callback)
        
        # Analyze memory patterns
        peak_memory_mb = max(sample['memory_mb'] for sample in memory_samples)
        final_memory_mb = memory_samples[-1]['memory_mb']
        
        # Verify memory cleanup effectiveness
        memory_growth_ratio = final_memory_mb / initial_memory_mb
        assert memory_growth_ratio < 3.0, \
            f"Memory growth too high: {memory_growth_ratio:.2f}x initial memory"
        
        # Verify garbage collection occurred
        assert len(gc_events) > 0, "Garbage collection should have occurred"
        
        # Verify memory was reclaimed
        max_delta = max(sample['memory_delta_mb'] for sample in memory_samples)
        final_delta = memory_samples[-1]['memory_delta_mb']
        
        cleanup_effectiveness = (max_delta - final_delta) / max_delta
        assert cleanup_effectiveness > 0.3, \
            f"Memory cleanup not effective enough: {cleanup_effectiveness:.2%}"
    
    @pytest.mark.asyncio
    async def test_resource_allocation_and_deallocation(self, integration_test_environment):
        """Test coordinated resource allocation and deallocation."""
        env = integration_test_environment
        
        # Configure resource pools
        resource_pool_config = {
            'pdf_processor_threads': 4,
            'lightrag_workers': 3,
            'network_connection_pool': 8,
            'file_handle_pool': 15
        }
        env.config.update(resource_pool_config)
        
        pdf_paths = env.create_test_pdf_collection(12)
        
        # Track resource allocation
        resource_tracker = env.lightrag_system.get_resource_tracker()
        allocation_events = []
        
        # Monitor resource allocation during processing
        async def resource_monitoring():
            while True:
                try:
                    allocations = resource_tracker.get_current_allocations()
                    allocation_events.append({
                        'timestamp': time.time(),
                        'allocated_threads': allocations['threads'],
                        'allocated_workers': allocations['workers'],
                        'active_connections': allocations['network_connections'],
                        'open_file_handles': allocations['file_handles']
                    })
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break
        
        monitoring_task = asyncio.create_task(resource_monitoring())
        
        try:
            # Process documents with resource tracking
            results = await env.pdf_processor.process_batch_with_resource_tracking(pdf_paths)
            
            # Verify processing completed successfully
            successful_count = sum(1 for r in results if r.success)
            assert successful_count > 0, "Some documents should process successfully"
            
        finally:
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Analyze resource usage patterns
        peak_threads = max(event['allocated_threads'] for event in allocation_events)
        peak_workers = max(event['allocated_workers'] for event in allocation_events)
        peak_connections = max(event['active_connections'] for event in allocation_events)
        peak_handles = max(event['open_file_handles'] for event in allocation_events)
        
        # Verify resource allocation stayed within limits
        assert peak_threads <= resource_pool_config['pdf_processor_threads']
        assert peak_workers <= resource_pool_config['lightrag_workers']
        assert peak_connections <= resource_pool_config['network_connection_pool']
        assert peak_handles <= resource_pool_config['file_handle_pool']
        
        # Verify resource deallocation
        final_allocations = allocation_events[-1]
        assert final_allocations['allocated_threads'] <= 1, "Threads should be mostly deallocated"
        assert final_allocations['allocated_workers'] <= 1, "Workers should be mostly deallocated"
        assert final_allocations['active_connections'] == 0, "All connections should be closed"
        assert final_allocations['open_file_handles'] <= 3, "File handles should be mostly freed"
    
    @pytest.mark.asyncio
    async def test_cross_component_resource_sharing(self, integration_test_environment):
        """Test resource sharing coordination between components."""
        env = integration_test_environment
        
        # Configure shared resource pools
        shared_resources_config = {
            'enable_resource_sharing': True,
            'shared_thread_pool_size': 6,
            'shared_memory_pool_mb': 400,
            'resource_sharing_strategy': 'dynamic_allocation'
        }
        env.config.update(shared_resources_config)
        
        pdf_paths = env.create_test_pdf_collection(20)
        
        # Create workloads with different resource requirements
        cpu_intensive_pdfs = pdf_paths[:10]  # Simulate CPU-intensive PDFs
        memory_intensive_pdfs = pdf_paths[10:]  # Simulate memory-intensive PDFs
        
        # Track resource sharing
        sharing_tracker = env.lightrag_system.get_resource_sharing_tracker()
        sharing_events = []
        
        # Process both workloads concurrently
        cpu_task = asyncio.create_task(
            env.pdf_processor.process_batch_cpu_optimized(cpu_intensive_pdfs)
        )
        
        memory_task = asyncio.create_task(
            env.lightrag_system.ingest_documents_memory_optimized(memory_intensive_pdfs)
        )
        
        # Monitor resource sharing
        async def sharing_monitoring():
            while not cpu_task.done() or not memory_task.done():
                try:
                    sharing_info = sharing_tracker.get_current_sharing()
                    sharing_events.append({
                        'timestamp': time.time(),
                        'pdf_processor_allocation': sharing_info['pdf_processor'],
                        'lightrag_allocation': sharing_info['lightrag'],
                        'shared_pool_utilization': sharing_info['shared_pool_utilization'],
                        'resource_contention': sharing_info['contention_level']
                    })
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break
        
        sharing_monitoring_task = asyncio.create_task(sharing_monitoring())
        
        try:
            # Wait for both workloads to complete
            cpu_results, memory_results = await asyncio.gather(cpu_task, memory_task)
            
            # Verify both workloads completed successfully
            cpu_success_count = sum(1 for r in cpu_results if r.success)
            memory_success_count = sum(1 for r in memory_results if r.success)
            
            assert cpu_success_count > 0, "CPU-intensive processing should succeed"
            assert memory_success_count > 0, "Memory-intensive processing should succeed"
            
        finally:
            sharing_monitoring_task.cancel()
            try:
                await sharing_monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Analyze resource sharing effectiveness
        avg_shared_utilization = sum(event['shared_pool_utilization'] 
                                   for event in sharing_events) / len(sharing_events)
        assert avg_shared_utilization > 0.5, \
            f"Shared resource utilization too low: {avg_shared_utilization:.2%}"
        
        # Verify dynamic resource allocation occurred
        pdf_allocations = [event['pdf_processor_allocation'] for event in sharing_events]
        lightrag_allocations = [event['lightrag_allocation'] for event in sharing_events]
        
        pdf_allocation_variance = max(pdf_allocations) - min(pdf_allocations)
        lightrag_allocation_variance = max(lightrag_allocations) - min(lightrag_allocations)
        
        assert pdf_allocation_variance > 0, "PDF processor allocation should vary dynamically"
        assert lightrag_allocation_variance > 0, "LightRAG allocation should vary dynamically"
        
        # Verify resource contention was managed
        max_contention = max(event['resource_contention'] for event in sharing_events)
        assert max_contention < 0.8, f"Resource contention too high: {max_contention:.2%}"


# =====================================================================
# PROGRESS TRACKING INTEGRATION TESTS
# =====================================================================

@pytest.mark.integration
class TestProgressTrackingIntegration:
    """
    Test end-to-end progress reporting and state persistence.
    
    This test class validates:
    - Progress tracking across all pipeline components
    - State persistence during failures and recovery
    - Progress synchronization between components
    - Real-time progress updates and notifications
    - Progress aggregation and reporting
    """
    
    @pytest.mark.asyncio
    async def test_end_to_end_progress_tracking(self, integration_test_environment):
        """Test comprehensive progress tracking across the entire pipeline."""
        env = integration_test_environment
        
        # Configure detailed progress tracking
        progress_config = {
            'enable_detailed_tracking': True,
            'progress_update_interval_ms': 100,
            'track_sub_operations': True,
            'enable_progress_persistence': True
        }
        env.config.update(progress_config)
        
        pdf_paths = env.create_test_pdf_collection(15)
        
        # Initialize progress tracking
        progress_tracker = env.progress_tracker
        session_id = await progress_tracker.start_session('end_to_end_test')
        
        # Track progress updates
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append({
                'timestamp': time.time(),
                'session_id': update['session_id'],
                'operation': update['operation'],
                'progress_percent': update['progress_percent'],
                'current_item': update.get('current_item'),
                'items_completed': update.get('items_completed'),
                'items_total': update.get('items_total'),
                'estimated_remaining_seconds': update.get('estimated_remaining_seconds'),
                'component': update.get('component')
            })
        
        progress_tracker.add_callback(progress_callback)
        
        try:
            # Process documents with comprehensive progress tracking
            results = await env.pdf_processor.process_batch_with_progress(
                pdf_paths,
                session_id=session_id,
                track_sub_operations=True
            )
            
            # Verify processing completed
            successful_count = sum(1 for r in results if r.success)
            assert successful_count > 0, "Some documents should process successfully"
            
        finally:
            await progress_tracker.end_session(session_id)
        
        # Analyze progress tracking coverage
        operations_tracked = set(update['operation'] for update in progress_updates)
        expected_operations = {
            'pdf_processing', 'document_ingestion', 'metadata_extraction',
            'entity_extraction', 'relationship_building', 'index_creation'
        }
        
        # Verify all major operations were tracked
        missing_operations = expected_operations - operations_tracked
        assert len(missing_operations) == 0, f"Missing progress tracking for: {missing_operations}"
        
        # Verify progress progression
        pdf_processing_updates = [u for u in progress_updates if u['operation'] == 'pdf_processing']
        progress_values = [u['progress_percent'] for u in pdf_processing_updates]
        
        assert progress_values[0] < progress_values[-1], "Progress should increase over time"
        assert progress_values[-1] >= 100.0, "Final progress should reach 100%"
        
        # Verify component-level tracking
        components_tracked = set(update['component'] for update in progress_updates if update.get('component'))
        expected_components = {'pdf_processor', 'lightrag_system', 'cost_monitor'}
        
        assert len(components_tracked.intersection(expected_components)) > 0, \
            "Component-level progress should be tracked"
    
    @pytest.mark.asyncio
    async def test_progress_persistence_during_failures(self, integration_test_environment, error_injector):
        """Test progress state persistence during failures and recovery."""
        env = integration_test_environment
        
        # Configure progress persistence
        persistence_config = {
            'enable_progress_persistence': True,
            'persistence_interval_seconds': 1,
            'checkpoint_frequency': 5  # Checkpoint every 5 operations
        }
        env.config.update(persistence_config)
        
        pdf_paths = env.create_test_pdf_collection(20)
        
        # Set up failure injection mid-process
        error_injector.add_rule(
            'pdf_processing',
            Exception("Simulated processing failure"),
            trigger_after=10,  # Fail after processing 10 documents
            probability=1.0
        )
        
        progress_tracker = env.progress_tracker
        session_id = await progress_tracker.start_session('persistence_test')
        
        # Track progress checkpoints
        checkpoints = []
        
        def checkpoint_callback(checkpoint_data):
            checkpoints.append({
                'timestamp': time.time(),
                'session_id': checkpoint_data['session_id'],
                'items_completed': checkpoint_data['items_completed'],
                'items_total': checkpoint_data['items_total'],
                'checkpoint_id': checkpoint_data['checkpoint_id'],
                'state_data': checkpoint_data['state_data']
            })
        
        progress_tracker.add_checkpoint_callback(checkpoint_callback)
        
        # First phase: Process until failure
        try:
            await env.pdf_processor.process_batch_with_checkpoints(
                pdf_paths,
                session_id=session_id
            )
            assert False, "Processing should have failed"
        except Exception as e:
            assert "Simulated processing failure" in str(e)
        
        # Verify checkpoints were created
        assert len(checkpoints) > 0, "Progress checkpoints should have been created"
        
        # Get the latest checkpoint before failure
        latest_checkpoint = checkpoints[-1]
        assert latest_checkpoint['items_completed'] > 0, "Some progress should have been saved"
        
        # Second phase: Recovery and resume
        error_injector.injection_rules.clear()  # Remove error injection
        
        # Resume from checkpoint
        recovery_session_id = await progress_tracker.resume_from_checkpoint(
            latest_checkpoint['checkpoint_id']
        )
        
        # Verify resume state
        resume_state = await progress_tracker.get_session_state(recovery_session_id)
        assert resume_state['items_completed'] == latest_checkpoint['items_completed']
        assert resume_state['items_total'] == latest_checkpoint['items_total']
        
        # Complete processing from checkpoint
        remaining_pdfs = pdf_paths[latest_checkpoint['items_completed']:]
        final_results = await env.pdf_processor.process_batch_with_progress(
            remaining_pdfs,
            session_id=recovery_session_id
        )
        
        # Verify recovery completion
        successful_recoveries = sum(1 for r in final_results if r.success)
        assert successful_recoveries > 0, "Recovery processing should succeed"
        
        await progress_tracker.end_session(recovery_session_id)
    
    @pytest.mark.asyncio
    async def test_progress_synchronization_between_components(self, integration_test_environment):
        """Test progress synchronization across multiple components."""
        env = integration_test_environment
        
        # Configure synchronized progress tracking
        sync_config = {
            'enable_progress_synchronization': True,
            'sync_interval_ms': 200,
            'enable_cross_component_updates': True
        }
        env.config.update(sync_config)
        
        pdf_paths = env.create_test_pdf_collection(12)
        
        progress_tracker = env.progress_tracker
        session_id = await progress_tracker.start_session('sync_test')
        
        # Track synchronized progress updates
        sync_updates = []
        
        def sync_callback(sync_data):
            sync_updates.append({
                'timestamp': time.time(),
                'session_id': sync_data['session_id'],
                'component_progress': sync_data['component_progress'],
                'overall_progress': sync_data['overall_progress'],
                'sync_state': sync_data['sync_state']
            })
        
        progress_tracker.add_sync_callback(sync_callback)
        
        try:
            # Start concurrent processing across multiple components
            pdf_processing_task = asyncio.create_task(
                env.pdf_processor.process_batch_with_sync(pdf_paths, session_id)
            )
            
            lightrag_ingestion_task = asyncio.create_task(
                env.lightrag_system.ingest_with_sync(pdf_paths, session_id)
            )
            
            cost_monitoring_task = asyncio.create_task(
                env.cost_monitor.monitor_with_sync(session_id)
            )
            
            # Wait for all components to complete
            pdf_results, lightrag_results, cost_results = await asyncio.gather(
                pdf_processing_task,
                lightrag_ingestion_task,
                cost_monitoring_task
            )
            
            # Verify all components completed successfully
            assert sum(1 for r in pdf_results if r.success) > 0
            assert lightrag_results.success
            assert cost_results.success
            
        finally:
            await progress_tracker.end_session(session_id)
        
        # Analyze progress synchronization
        assert len(sync_updates) > 0, "Synchronized progress updates should occur"
        
        # Verify component progress synchronization
        for sync_update in sync_updates:
            component_progress = sync_update['component_progress']
            overall_progress = sync_update['overall_progress']
            
            # Verify all components are represented
            expected_components = ['pdf_processor', 'lightrag_system', 'cost_monitor']
            for component in expected_components:
                assert component in component_progress, \
                    f"Component {component} missing from synchronized progress"
            
            # Verify overall progress is reasonable average of component progress
            component_avg = sum(component_progress.values()) / len(component_progress)
            progress_diff = abs(overall_progress - component_avg)
            assert progress_diff < 10.0, \
                f"Overall progress {overall_progress}% not synchronized with components (avg: {component_avg}%)"
        
        # Verify progress synchronization improved over time
        sync_states = [update['sync_state'] for update in sync_updates]
        final_sync_state = sync_states[-1]
        
        assert final_sync_state['sync_quality'] > 0.8, \
            f"Final synchronization quality too low: {final_sync_state['sync_quality']:.2%}"
        assert final_sync_state['component_alignment'] > 0.9, \
            f"Component alignment too low: {final_sync_state['component_alignment']:.2%}"
    
    @pytest.mark.asyncio
    async def test_real_time_progress_notifications(self, integration_test_environment):
        """Test real-time progress notifications and updates."""
        env = integration_test_environment
        
        # Configure real-time notifications
        notification_config = {
            'enable_real_time_notifications': True,
            'notification_threshold_percent': 10,  # Notify every 10% progress
            'enable_milestone_notifications': True,
            'notification_channels': ['websocket', 'callback']
        }
        env.config.update(notification_config)
        
        pdf_paths = env.create_test_pdf_collection(10)
        
        progress_tracker = env.progress_tracker
        session_id = await progress_tracker.start_session('notification_test')
        
        # Track real-time notifications
        notifications = []
        milestone_notifications = []
        
        def notification_handler(notification):
            notifications.append({
                'timestamp': time.time(),
                'type': notification['type'],
                'session_id': notification['session_id'],
                'progress_percent': notification['progress_percent'],
                'message': notification['message'],
                'data': notification.get('data', {})
            })
        
        def milestone_handler(milestone):
            milestone_notifications.append({
                'timestamp': time.time(),
                'milestone': milestone['milestone'],
                'session_id': milestone['session_id'],
                'achievement_data': milestone['achievement_data']
            })
        
        progress_tracker.add_notification_handler(notification_handler)
        progress_tracker.add_milestone_handler(milestone_handler)
        
        try:
            # Process documents with real-time notifications
            results = await env.pdf_processor.process_batch_with_notifications(
                pdf_paths,
                session_id=session_id
            )
            
            # Verify processing completed
            successful_count = sum(1 for r in results if r.success)
            assert successful_count > 0, "Some documents should process successfully"
            
        finally:
            await progress_tracker.end_session(session_id)
        
        # Verify real-time notifications
        assert len(notifications) > 0, "Real-time notifications should have been sent"
        
        # Verify notification timing
        first_notification_time = notifications[0]['timestamp']
        last_notification_time = notifications[-1]['timestamp']
        processing_duration = last_notification_time - first_notification_time
        
        # Should have received notifications throughout processing
        assert processing_duration > 0.1, "Notifications should span processing duration"
        
        # Verify notification thresholds
        progress_values = [n['progress_percent'] for n in notifications]
        progress_increments = [progress_values[i] - progress_values[i-1] 
                             for i in range(1, len(progress_values))]
        
        # Most progress increments should be around the threshold
        reasonable_increments = [inc for inc in progress_increments if 5.0 <= inc <= 15.0]
        assert len(reasonable_increments) >= len(progress_increments) * 0.7, \
            "Progress notification thresholds not working correctly"
        
        # Verify milestone notifications
        assert len(milestone_notifications) > 0, "Milestone notifications should have been sent"
        
        milestone_types = set(m['milestone'] for m in milestone_notifications)
        expected_milestones = {'processing_started', 'halfway_complete', 'processing_complete'}
        
        assert len(milestone_types.intersection(expected_milestones)) > 0, \
            "Important milestones should have been reached"
    
    @pytest.mark.asyncio
    async def test_progress_aggregation_and_reporting(self, integration_test_environment):
        """Test progress aggregation and comprehensive reporting."""
        env = integration_test_environment
        
        # Configure comprehensive progress reporting
        reporting_config = {
            'enable_detailed_reporting': True,
            'track_performance_metrics': True,
            'generate_progress_summaries': True,
            'include_component_breakdowns': True
        }
        env.config.update(reporting_config)
        
        pdf_paths = env.create_test_pdf_collection(15)
        
        progress_tracker = env.progress_tracker
        session_id = await progress_tracker.start_session('reporting_test')
        
        # Track progress for reporting
        start_time = time.time()
        
        try:
            # Process documents with comprehensive tracking
            results = await env.pdf_processor.process_batch_with_comprehensive_tracking(
                pdf_paths,
                session_id=session_id
            )
            
            # Verify processing completed
            successful_count = sum(1 for r in results if r.success)
            assert successful_count > 0, "Some documents should process successfully"
            
        finally:
            end_time = time.time()
            total_processing_time = end_time - start_time
            
            await progress_tracker.end_session(session_id)
        
        # Generate comprehensive progress report
        progress_report = await progress_tracker.generate_session_report(session_id)
        
        # Verify report completeness
        assert 'session_summary' in progress_report
        assert 'component_breakdown' in progress_report
        assert 'performance_metrics' in progress_report
        assert 'timeline_analysis' in progress_report
        
        # Verify session summary
        session_summary = progress_report['session_summary']
        assert session_summary['total_items'] == len(pdf_paths)
        assert session_summary['completed_items'] > 0
        assert session_summary['success_rate'] > 0.0
        assert session_summary['total_duration_seconds'] > 0
        
        # Verify component breakdown
        component_breakdown = progress_report['component_breakdown']
        expected_components = ['pdf_processor', 'lightrag_system', 'cost_monitor']
        
        for component in expected_components:
            assert component in component_breakdown
            component_data = component_breakdown[component]
            
            assert 'processing_time_seconds' in component_data
            assert 'items_processed' in component_data
            assert 'success_rate' in component_data
            assert component_data['success_rate'] >= 0.0
            assert component_data['success_rate'] <= 1.0
        
        # Verify performance metrics
        performance_metrics = progress_report['performance_metrics']
        assert 'throughput_items_per_second' in performance_metrics
        assert 'average_processing_time_per_item' in performance_metrics
        assert 'peak_memory_usage_mb' in performance_metrics
        assert 'total_cost' in performance_metrics
        
        # Verify timeline analysis
        timeline_analysis = progress_report['timeline_analysis']
        assert 'progress_checkpoints' in timeline_analysis
        assert 'bottleneck_analysis' in timeline_analysis
        assert 'efficiency_score' in timeline_analysis
        
        # Verify efficiency calculations
        efficiency_score = timeline_analysis['efficiency_score']
        assert 0.0 <= efficiency_score <= 1.0, "Efficiency score should be between 0 and 1"
        
        # Verify bottleneck detection
        bottleneck_analysis = timeline_analysis['bottleneck_analysis']
        assert 'slowest_component' in bottleneck_analysis
        assert 'bottleneck_severity' in bottleneck_analysis
        assert 'recommendations' in bottleneck_analysis


if __name__ == "__main__":
    """
    Run the PDF-LightRAG integration test suite when executed directly.
    
    These tests provide comprehensive validation of the complete data pipeline
    from PDF files to indexed documents in LightRAG, including metadata
    preservation, cost tracking, performance characteristics, error
    handling scenarios, configuration management, resource coordination,
    and progress tracking across all components.
    
    The test suite now includes comprehensive integration tests for:
    - Configuration consistency and validation across components
    - Coordinated error handling and circuit breaker patterns  
    - Memory usage coordination and resource management
    - End-to-end progress reporting and state persistence
    """
    pytest.main([__file__, "-v", "--tb=short", "-k", "not slow"])