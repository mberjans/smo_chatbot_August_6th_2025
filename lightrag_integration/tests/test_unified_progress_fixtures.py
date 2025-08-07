#!/usr/bin/env python3
"""
Test Fixtures and Utilities for Unified Progress Tracking Tests.

This module provides specialized fixtures, mock implementations, and utility
classes for testing the unified progress tracking system in isolation and
integration scenarios.

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import asyncio
import json
import time
import threading
import random
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from lightrag_integration.unified_progress_tracker import (
    KnowledgeBaseProgressTracker,
    KnowledgeBasePhase,
    PhaseWeights,
    PhaseProgressInfo,
    UnifiedProgressState
)
from lightrag_integration.progress_config import ProgressTrackingConfig, ProcessingMetrics
from lightrag_integration.progress_tracker import PDFProcessingProgressTracker


# =====================================================================
# SPECIALIZED TEST FIXTURES
# =====================================================================

@pytest.fixture
def realistic_phase_weights():
    """Realistic phase weights based on actual workload distribution."""
    return PhaseWeights(
        storage_init=0.05,      # Quick initialization
        pdf_processing=0.75,    # Most time-consuming 
        document_ingestion=0.15, # Knowledge graph building
        finalization=0.05       # Final optimization
    )

@pytest.fixture
def progress_tracking_config_full():
    """Full featured progress tracking configuration."""
    return ProgressTrackingConfig(
        enable_progress_tracking=True,
        enable_unified_progress_tracking=True,
        enable_phase_based_progress=True,
        enable_progress_callbacks=True,
        save_unified_progress_to_file=True,
        log_progress_interval=2,
        log_detailed_errors=True,
        log_processing_stats=True,
        progress_log_level="INFO",
        error_log_level="ERROR",
        phase_progress_update_interval=1.0,
        enable_memory_monitoring=True,
        memory_check_interval=5
    )

@pytest.fixture
def progress_tracking_config_minimal():
    """Minimal progress tracking configuration for performance tests."""
    return ProgressTrackingConfig(
        enable_progress_tracking=True,
        enable_unified_progress_tracking=True,
        enable_phase_based_progress=False,
        enable_progress_callbacks=False,
        save_unified_progress_to_file=False,
        log_progress_interval=100,  # Less frequent logging
        log_detailed_errors=False,
        log_processing_stats=False
    )


# =====================================================================
# MOCK IMPLEMENTATIONS
# =====================================================================

@dataclass
class MockDocumentMetadata:
    """Mock metadata for test documents."""
    filename: str
    title: str
    authors: List[str]
    journal: str
    year: int
    doi: str
    keywords: List[str]
    page_count: int = 1
    file_size_bytes: int = 1024
    processing_time: float = 0.5
    extraction_success: bool = True
    error_message: Optional[str] = None

class MockPDFProcessor:
    """
    Enhanced mock PDF processor with realistic behavior patterns.
    
    This mock simulates realistic PDF processing scenarios including:
    - Variable processing times
    - Batch processing capabilities
    - Realistic failure patterns
    - Progress reporting integration
    """
    
    def __init__(self, 
                 base_processing_time: float = 0.5,
                 failure_rate: float = 0.05,
                 enable_delays: bool = True):
        """
        Initialize mock PDF processor.
        
        Args:
            base_processing_time: Base time per document processing
            failure_rate: Probability of processing failure (0.0 to 1.0)
            enable_delays: Whether to simulate realistic processing delays
        """
        self.base_processing_time = base_processing_time
        self.failure_rate = failure_rate
        self.enable_delays = enable_delays
        self.processed_files = []
        self.failed_files = []
        self.total_processing_time = 0.0
        self.metrics = ProcessingMetrics()
        
        # Realistic error types
        self.error_types = [
            "PDF encryption detected",
            "Corrupted PDF structure", 
            "Memory allocation failed",
            "Timeout during extraction",
            "Unsupported PDF version",
            "Text extraction failed"
        ]
    
    async def process_single_document(self, document_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single document with realistic behavior."""
        start_time = time.time()
        
        # Simulate processing delay
        if self.enable_delays:
            processing_time = random.uniform(
                self.base_processing_time * 0.5,
                self.base_processing_time * 2.0
            )
            await asyncio.sleep(processing_time)
        
        # Determine if processing should fail
        should_fail = random.random() < self.failure_rate
        
        if should_fail:
            error_msg = random.choice(self.error_types)
            self.failed_files.append({
                'path': str(document_path),
                'error': error_msg,
                'processing_time': time.time() - start_time
            })
            self.metrics.failed_files += 1
            self.metrics.add_error(error_msg.split()[0])  # First word as error type
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - start_time,
                'metadata': None,
                'text': None
            }
        
        # Generate realistic extracted content
        filename = Path(document_path).name
        
        # Simulate realistic text extraction based on filename
        if 'metabolomics' in filename.lower():
            text_content = self._generate_metabolomics_content()
        elif 'proteomics' in filename.lower():
            text_content = self._generate_proteomics_content()
        else:
            text_content = self._generate_generic_biomedical_content()
        
        metadata = MockDocumentMetadata(
            filename=filename,
            title=f"Clinical Study: {filename.replace('.pdf', '').replace('_', ' ').title()}",
            authors=[f"Dr. Author{i}" for i in range(1, random.randint(2, 6))],
            journal=f"Journal of {random.choice(['Clinical', 'Biomedical', 'Medical'])} Research",
            year=random.randint(2020, 2024),
            doi=f"10.1000/test.{random.randint(1000, 9999)}",
            keywords=['biomarkers', 'clinical', 'research', 'analysis'],
            page_count=random.randint(8, 25),
            file_size_bytes=random.randint(1024*100, 1024*1024*3),  # 100KB to 3MB
            processing_time=time.time() - start_time
        )
        
        result = {
            'success': True,
            'text': text_content,
            'metadata': metadata.__dict__,
            'processing_time': time.time() - start_time,
            'error': None
        }
        
        self.processed_files.append(result)
        self.metrics.completed_files += 1
        self.metrics.total_pages += metadata.page_count
        self.metrics.total_characters += len(text_content)
        
        self.total_processing_time += result['processing_time']
        self.metrics.update_processing_time()
        
        return result
    
    async def process_batch(self, document_paths: List[Union[str, Path]], 
                          batch_size: int = 5,
                          progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process documents in batches with progress reporting."""
        self.metrics.total_files = len(document_paths)
        self.metrics.start_time = datetime.now()
        
        batch_results = []
        
        # Process in batches
        for batch_start in range(0, len(document_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(document_paths))
            batch_paths = document_paths[batch_start:batch_end]
            
            # Process batch concurrently (simulate)
            batch_tasks = [
                self.process_single_document(path) for path in batch_paths
            ]
            batch_outcomes = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for outcome in batch_outcomes:
                if isinstance(outcome, Exception):
                    # Handle unexpected errors
                    self.metrics.failed_files += 1
                    self.metrics.add_error("UnexpectedError")
                    batch_results.append({
                        'success': False,
                        'error': str(outcome),
                        'processing_time': 0.1
                    })
                else:
                    batch_results.append(outcome)
            
            # Report progress if callback provided
            if progress_callback:
                progress = batch_end / len(document_paths)
                progress_callback(
                    progress,
                    f"Processed batch {(batch_start // batch_size) + 1}",
                    {
                        'batch_number': (batch_start // batch_size) + 1,
                        'completed_files': self.metrics.completed_files,
                        'failed_files': self.metrics.failed_files,
                        'total_files': self.metrics.total_files
                    }
                )
        
        self.metrics.end_time = datetime.now()
        
        return {
            'results': batch_results,
            'total_processed': len(document_paths),
            'successful': self.metrics.completed_files,
            'failed': self.metrics.failed_files,
            'total_time': self.metrics.processing_time,
            'metrics': self.metrics.to_dict()
        }
    
    def get_current_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics."""
        return self.metrics
    
    def reset_metrics(self):
        """Reset processing metrics."""
        self.metrics = ProcessingMetrics()
        self.processed_files.clear()
        self.failed_files.clear()
        self.total_processing_time = 0.0
    
    def _generate_metabolomics_content(self) -> str:
        """Generate realistic metabolomics content."""
        return """
        Abstract: This study presents a comprehensive metabolomic analysis of diabetes 
        progression in a cohort of 150 patients. Using LC-MS/MS techniques, we identified 
        45 significantly altered metabolites (p < 0.05). Key findings include elevated 
        branched-chain amino acids and altered glucose metabolism intermediates.
        
        Methods: Plasma samples were collected and analyzed using Agilent 6550 Q-TOF 
        mass spectrometer. Statistical analysis employed R software with MetaboAnalyst.
        
        Results: Principal component analysis revealed clear separation between patient 
        groups. Pathway analysis indicated disrupted amino acid metabolism and altered 
        glycolysis. These findings suggest potential biomarkers for disease monitoring.
        
        Conclusion: Metabolomic profiling provides valuable insights into diabetes 
        pathophysiology and may inform personalized treatment strategies.
        """
    
    def _generate_proteomics_content(self) -> str:
        """Generate realistic proteomics content."""
        return """
        Abstract: Proteomic analysis of cardiovascular disease biomarkers using mass 
        spectrometry revealed 38 differentially expressed proteins in patient sera 
        compared to healthy controls. Key proteins include troponin, CRP, and BNP.
        
        Methods: Protein extraction and digestion followed by LC-MS/MS analysis on 
        Thermo Q Exactive platform. Database searching performed using MaxQuant.
        
        Results: Volcano plot analysis identified significant protein changes. Gene 
        ontology enrichment revealed involvement in inflammatory pathways and cardiac 
        remodeling processes. Receiver operating characteristic analysis demonstrated 
        diagnostic potential.
        
        Conclusion: These protein biomarkers offer potential for improved cardiovascular 
        disease diagnosis and monitoring of treatment response.
        """
    
    def _generate_generic_biomedical_content(self) -> str:
        """Generate generic biomedical research content.""" 
        return """
        Abstract: This clinical research study investigates molecular mechanisms 
        underlying disease progression through integrated omics approaches. Analysis 
        of patient samples revealed significant molecular signatures associated with 
        disease severity and treatment outcomes.
        
        Methods: Multi-omics analysis combining genomics, proteomics, and metabolomics 
        data from patient cohorts. Statistical methods included multivariate analysis 
        and machine learning approaches for biomarker discovery.
        
        Results: Integration of omics data identified key molecular pathways involved 
        in disease pathogenesis. Biomarker panels showed high diagnostic accuracy and 
        prognostic value for patient stratification.
        
        Conclusion: This study demonstrates the power of integrated omics approaches 
        for understanding complex diseases and developing precision medicine strategies.
        """


class MockLightRAGKnowledgeBase:
    """
    Mock LightRAG knowledge base with realistic ingestion behavior.
    
    This mock simulates the document ingestion process into a knowledge graph,
    including entity extraction, relationship discovery, and indexing operations.
    """
    
    def __init__(self,
                 ingestion_delay: float = 0.1,
                 entity_extraction_rate: int = 10,
                 relationship_discovery_rate: int = 5):
        """
        Initialize mock knowledge base.
        
        Args:
            ingestion_delay: Base delay per document ingestion
            entity_extraction_rate: Average entities extracted per document
            relationship_discovery_rate: Average relationships found per document
        """
        self.ingestion_delay = ingestion_delay
        self.entity_extraction_rate = entity_extraction_rate
        self.relationship_discovery_rate = relationship_discovery_rate
        
        self.ingested_documents = []
        self.entities_database = {}
        self.relationships_database = []
        self.ingestion_metrics = {
            'total_documents': 0,
            'total_entities': 0,
            'total_relationships': 0,
            'ingestion_time': 0.0
        }
    
    async def ingest_document(self, document_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a single document into the knowledge base."""
        start_time = time.time()
        
        # Simulate ingestion processing time
        await asyncio.sleep(random.uniform(
            self.ingestion_delay * 0.5, 
            self.ingestion_delay * 1.5
        ))
        
        # Extract mock entities
        entities = self._extract_entities(document_text)
        
        # Discover mock relationships
        relationships = self._discover_relationships(document_text, entities)
        
        # Store in mock database
        doc_id = f"doc_{len(self.ingested_documents) + 1}"
        ingested_doc = {
            'id': doc_id,
            'metadata': metadata.copy(),
            'entities': entities,
            'relationships': relationships,
            'text_length': len(document_text),
            'ingestion_time': time.time() - start_time
        }
        
        self.ingested_documents.append(ingested_doc)
        
        # Update metrics
        self.ingestion_metrics['total_documents'] += 1
        self.ingestion_metrics['total_entities'] += len(entities)
        self.ingestion_metrics['total_relationships'] += len(relationships)
        self.ingestion_metrics['ingestion_time'] += ingested_doc['ingestion_time']
        
        return {
            'success': True,
            'document_id': doc_id,
            'entities_found': len(entities),
            'relationships_found': len(relationships),
            'processing_time': ingested_doc['ingestion_time']
        }
    
    async def ingest_batch(self, documents: List[Dict[str, Any]], 
                          progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Ingest multiple documents with progress reporting."""
        batch_start_time = time.time()
        batch_results = []
        
        for i, doc in enumerate(documents):
            result = await self.ingest_document(doc['text'], doc['metadata'])
            batch_results.append(result)
            
            # Report progress
            if progress_callback:
                progress = (i + 1) / len(documents)
                progress_callback(
                    progress,
                    f"Ingested document {i + 1}/{len(documents)}",
                    {
                        'current_document': i + 1,
                        'total_documents': len(documents),
                        'entities_extracted': sum(r['entities_found'] for r in batch_results),
                        'relationships_found': sum(r['relationships_found'] for r in batch_results)
                    }
                )
        
        total_time = time.time() - batch_start_time
        
        return {
            'results': batch_results,
            'total_documents': len(documents),
            'successful_ingestions': sum(1 for r in batch_results if r['success']),
            'total_entities': sum(r['entities_found'] for r in batch_results),
            'total_relationships': sum(r['relationships_found'] for r in batch_results),
            'batch_processing_time': total_time
        }
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Get current ingestion statistics."""
        return self.ingestion_metrics.copy()
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract mock entities from text."""
        # Biomedical entity patterns
        entity_patterns = {
            'METABOLITE': ['glucose', 'lactate', 'pyruvate', 'alanine', 'glutamine'],
            'PROTEIN': ['insulin', 'albumin', 'hemoglobin', 'transferrin', 'CRP'],
            'GENE': ['APOE', 'PPAR', 'CYP2D6', 'MTHFR', 'ACE'],
            'DISEASE': ['diabetes', 'cardiovascular', 'cancer', 'hypertension'],
            'PATHWAY': ['glycolysis', 'TCA cycle', 'oxidative phosphorylation']
        }
        
        entities = []
        text_lower = text.lower()
        
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    entities.append({
                        'type': entity_type,
                        'name': pattern,
                        'confidence': random.uniform(0.7, 0.95)
                    })
        
        # Add some random entities to reach target count
        while len(entities) < self.entity_extraction_rate:
            entity_type = random.choice(list(entity_patterns.keys()))
            entity_name = f"entity_{len(entities) + 1}"
            entities.append({
                'type': entity_type,
                'name': entity_name,
                'confidence': random.uniform(0.6, 0.8)
            })
        
        return entities[:self.entity_extraction_rate]
    
    def _discover_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover mock relationships between entities."""
        relationships = []
        
        # Create relationships between entities
        relationship_types = ['regulates', 'associated_with', 'biomarker_for', 'pathway_involves']
        
        for i in range(min(self.relationship_discovery_rate, len(entities) - 1)):
            if i + 1 < len(entities):
                relationship = {
                    'source': entities[i]['name'],
                    'target': entities[i + 1]['name'],
                    'type': random.choice(relationship_types),
                    'confidence': random.uniform(0.5, 0.9)
                }
                relationships.append(relationship)
        
        return relationships


class ProgressCallbackTester:
    """
    Utility class for testing progress callbacks with detailed analysis.
    
    This class captures callback invocations and provides methods to analyze
    callback behavior, timing, and data consistency.
    """
    
    def __init__(self):
        self.callback_history = []
        self.timing_data = []
        self.callback_count = 0
        self.errors = []
        self._start_time = None
    
    def __call__(self, overall_progress: float, current_phase: KnowledgeBasePhase, 
                 phase_progress: float, status_message: str, 
                 phase_details: Dict[str, Any], all_phases: Dict) -> None:
        """Capture callback invocation with detailed analysis."""
        try:
            current_time = time.time()
            if self._start_time is None:
                self._start_time = current_time
            
            self.callback_count += 1
            
            # Capture full callback data
            callback_data = {
                'call_number': self.callback_count,
                'timestamp': current_time,
                'relative_time': current_time - self._start_time,
                'overall_progress': overall_progress,
                'current_phase': current_phase,
                'phase_progress': phase_progress,
                'status_message': status_message,
                'phase_details': phase_details.copy() if phase_details else {},
                'all_phases_summary': {
                    phase: {
                        'progress': info.current_progress,
                        'is_active': info.is_active,
                        'is_completed': info.is_completed,
                        'is_failed': info.is_failed
                    }
                    for phase, info in all_phases.items()
                }
            }
            
            self.callback_history.append(callback_data)
            
            # Timing analysis
            if len(self.callback_history) > 1:
                time_since_last = current_time - self.callback_history[-2]['timestamp']
                self.timing_data.append(time_since_last)
            
        except Exception as e:
            self.errors.append({
                'error': str(e),
                'timestamp': time.time(),
                'call_number': self.callback_count
            })
    
    def get_progress_sequence(self) -> List[float]:
        """Get sequence of overall progress values."""
        return [call['overall_progress'] for call in self.callback_history]
    
    def get_phase_transitions(self) -> List[Dict[str, Any]]:
        """Get information about phase transitions."""
        transitions = []
        current_phase = None
        
        for call in self.callback_history:
            if call['current_phase'] != current_phase:
                transitions.append({
                    'from_phase': current_phase,
                    'to_phase': call['current_phase'],
                    'timestamp': call['timestamp'],
                    'overall_progress': call['overall_progress']
                })
                current_phase = call['current_phase']
        
        return transitions
    
    def analyze_progress_monotonicity(self) -> Dict[str, Any]:
        """Analyze whether progress increases monotonically."""
        progress_values = self.get_progress_sequence()
        
        violations = []
        for i in range(1, len(progress_values)):
            if progress_values[i] < progress_values[i-1]:
                violations.append({
                    'position': i,
                    'previous_progress': progress_values[i-1],
                    'current_progress': progress_values[i],
                    'decrease': progress_values[i-1] - progress_values[i]
                })
        
        return {
            'is_monotonic': len(violations) == 0,
            'violations': violations,
            'total_calls': len(progress_values),
            'final_progress': progress_values[-1] if progress_values else 0.0
        }
    
    def get_timing_statistics(self) -> Dict[str, float]:
        """Get timing statistics for callback intervals."""
        if not self.timing_data:
            return {'count': 0}
        
        return {
            'count': len(self.timing_data),
            'mean_interval': sum(self.timing_data) / len(self.timing_data),
            'min_interval': min(self.timing_data),
            'max_interval': max(self.timing_data),
            'total_time': sum(self.timing_data)
        }
    
    def get_phase_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about progress within each phase."""
        phase_stats = {}
        
        for call in self.callback_history:
            phase = call['current_phase']
            if phase not in phase_stats:
                phase_stats[phase] = {
                    'call_count': 0,
                    'progress_updates': [],
                    'status_messages': [],
                    'first_seen': call['timestamp'],
                    'last_seen': call['timestamp']
                }
            
            stats = phase_stats[phase]
            stats['call_count'] += 1
            stats['progress_updates'].append(call['phase_progress'])
            stats['status_messages'].append(call['status_message'])
            stats['last_seen'] = call['timestamp']
        
        # Calculate derived statistics
        for phase, stats in phase_stats.items():
            stats['duration'] = stats['last_seen'] - stats['first_seen']
            if stats['progress_updates']:
                stats['max_progress'] = max(stats['progress_updates'])
                stats['min_progress'] = min(stats['progress_updates'])
                stats['final_progress'] = stats['progress_updates'][-1]
        
        return phase_stats
    
    def reset(self):
        """Reset all captured data."""
        self.callback_history.clear()
        self.timing_data.clear()
        self.callback_count = 0
        self.errors.clear()
        self._start_time = None


# =====================================================================
# PYTEST FIXTURES USING MOCK IMPLEMENTATIONS
# =====================================================================

@pytest.fixture
def mock_pdf_processor_realistic():
    """Provide realistic mock PDF processor."""
    return MockPDFProcessor(
        base_processing_time=0.05,  # Fast for tests
        failure_rate=0.1,  # 10% failure rate
        enable_delays=True
    )

@pytest.fixture
def mock_pdf_processor_fast():
    """Provide fast mock PDF processor for performance tests."""
    return MockPDFProcessor(
        base_processing_time=0.001,  # Very fast
        failure_rate=0.0,  # No failures
        enable_delays=False
    )

@pytest.fixture
def mock_lightrag_kb():
    """Provide mock LightRAG knowledge base."""
    return MockLightRAGKnowledgeBase(
        ingestion_delay=0.02,  # Fast for tests
        entity_extraction_rate=8,
        relationship_discovery_rate=4
    )

@pytest.fixture
def callback_tester():
    """Provide progress callback testing utility."""
    return ProgressCallbackTester()

@pytest.fixture
def test_document_collection():
    """Provide collection of test document paths."""
    return [
        "test_metabolomics_diabetes.pdf",
        "test_proteomics_cardiovascular.pdf", 
        "test_genomics_cancer.pdf",
        "test_lipidomics_neurological.pdf",
        "test_transcriptomics_immunology.pdf"
    ]

@pytest.fixture
def large_test_document_collection():
    """Provide large collection for performance testing."""
    docs = []
    categories = ['metabolomics', 'proteomics', 'genomics', 'lipidomics']
    diseases = ['diabetes', 'cardiovascular', 'cancer', 'neurological']
    
    for i in range(50):  # 50 documents
        category = random.choice(categories)
        disease = random.choice(diseases)
        docs.append(f"test_{category}_{disease}_{i:03d}.pdf")
    
    return docs

@pytest.fixture
async def integrated_test_environment(
    mock_pdf_processor_realistic, 
    mock_lightrag_kb,
    callback_tester,
    progress_tracking_config_full,
    temp_dir
):
    """Provide complete integrated test environment."""
    
    class IntegratedTestEnvironment:
        def __init__(self):
            self.pdf_processor = mock_pdf_processor_realistic
            self.knowledge_base = mock_lightrag_kb
            self.callback_tester = callback_tester
            self.progress_config = progress_tracking_config_full
            self.temp_dir = temp_dir
            
            # Setup progress file path
            self.progress_config.unified_progress_file_path = temp_dir / "progress.json"
            
            # Create unified progress tracker
            self.progress_tracker = KnowledgeBaseProgressTracker(
                progress_config=self.progress_config,
                progress_callback=self.callback_tester
            )
            
            self.test_results = {}
            self.performance_metrics = {}
        
        async def run_full_simulation(self, document_paths: List[str]) -> Dict[str, Any]:
            """Run complete knowledge base initialization simulation."""
            simulation_start = time.time()
            
            # Initialize
            self.progress_tracker.start_initialization(total_documents=len(document_paths))
            
            # Phase 1: Storage
            self.progress_tracker.start_phase(
                KnowledgeBasePhase.STORAGE_INIT, 
                "Setting up storage directories"
            )
            await asyncio.sleep(0.01)
            self.progress_tracker.complete_phase(
                KnowledgeBasePhase.STORAGE_INIT,
                "Storage initialization complete"
            )
            
            # Phase 2: PDF Processing
            self.progress_tracker.start_phase(
                KnowledgeBasePhase.PDF_PROCESSING,
                f"Processing {len(document_paths)} documents"
            )
            
            def pdf_progress_callback(progress, status, details):
                self.progress_tracker.update_phase_progress(
                    KnowledgeBasePhase.PDF_PROCESSING,
                    progress,
                    status,
                    details
                )
            
            pdf_results = await self.pdf_processor.process_batch(
                document_paths,
                progress_callback=pdf_progress_callback
            )
            
            self.progress_tracker.complete_phase(
                KnowledgeBasePhase.PDF_PROCESSING,
                f"PDF processing complete: {pdf_results['successful']} successful"
            )
            
            # Phase 3: Document Ingestion
            self.progress_tracker.start_phase(
                KnowledgeBasePhase.DOCUMENT_INGESTION,
                "Ingesting documents into knowledge graph"
            )
            
            # Prepare documents for ingestion
            successful_docs = [
                {'text': result['text'], 'metadata': result['metadata']}
                for result in pdf_results['results']
                if result['success']
            ]
            
            def ingestion_progress_callback(progress, status, details):
                self.progress_tracker.update_phase_progress(
                    KnowledgeBasePhase.DOCUMENT_INGESTION,
                    progress,
                    status,
                    details
                )
            
            ingestion_results = await self.knowledge_base.ingest_batch(
                successful_docs,
                progress_callback=ingestion_progress_callback
            )
            
            self.progress_tracker.complete_phase(
                KnowledgeBasePhase.DOCUMENT_INGESTION,
                f"Ingestion complete: {ingestion_results['total_entities']} entities extracted"
            )
            
            # Phase 4: Finalization
            self.progress_tracker.start_phase(
                KnowledgeBasePhase.FINALIZATION,
                "Finalizing knowledge base"
            )
            await asyncio.sleep(0.01)
            self.progress_tracker.complete_phase(
                KnowledgeBasePhase.FINALIZATION,
                "Knowledge base ready for queries"
            )
            
            simulation_time = time.time() - simulation_start
            
            return {
                'simulation_time': simulation_time,
                'pdf_results': pdf_results,
                'ingestion_results': ingestion_results,
                'final_progress_state': self.progress_tracker.get_current_state(),
                'callback_statistics': {
                    'total_callbacks': self.callback_tester.callback_count,
                    'progress_monotonicity': self.callback_tester.analyze_progress_monotonicity(),
                    'timing_stats': self.callback_tester.get_timing_statistics(),
                    'phase_stats': self.callback_tester.get_phase_statistics()
                }
            }
        
        def get_comprehensive_report(self) -> Dict[str, Any]:
            """Get comprehensive test report."""
            return {
                'pdf_processor_stats': self.pdf_processor.get_current_metrics().to_dict(),
                'knowledge_base_stats': self.knowledge_base.get_ingestion_statistics(),
                'progress_tracking_stats': {
                    'total_callbacks': self.callback_tester.callback_count,
                    'callback_errors': len(self.callback_tester.errors),
                    'progress_sequence_length': len(self.callback_tester.get_progress_sequence())
                }
            }
    
    return IntegratedTestEnvironment()


if __name__ == "__main__":
    # This module is for fixtures only, no direct execution
    pass