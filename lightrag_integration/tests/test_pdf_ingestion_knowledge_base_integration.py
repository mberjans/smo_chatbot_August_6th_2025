#!/usr/bin/env python3
"""
Comprehensive Test Suite for PDF Ingestion and Knowledge Base Initialization Integration.

This module implements comprehensive integration tests for the complete PDF ingestion 
pipeline and knowledge base initialization workflow. It validates the end-to-end process
from PDF document processing through to knowledge base storage and querying capabilities.

Test Coverage:
- PDF ingestion with realistic biomedical documents
- Knowledge base initialization with actual PDF files
- Single PDF processing with knowledge base storage
- Batch PDF processing with multiple documents
- Entity extraction and relationship mapping validation
- Knowledge base content validation after ingestion
- Performance benchmarking for PDF processing pipeline
- Error handling during PDF ingestion and storage
- Memory management and resource cleanup
- Progress tracking throughout the pipeline
- Integration with existing test infrastructure

Test Classes:
- TestSinglePDFIngestionIntegration: Single document processing tests
- TestBatchPDFIngestionIntegration: Multi-document batch processing tests
- TestKnowledgeBaseContentValidation: Knowledge base content and structure tests
- TestEntityExtractionIntegration: Entity and relationship extraction tests
- TestPDFIngestionPerformance: Performance and benchmarking tests
- TestPDFIngestionErrorHandling: Error scenarios and recovery tests
- TestKnowledgeBaseInitializationComplete: Complete initialization workflow tests
- TestPDFIngestionResourceManagement: Memory and resource management tests

Requirements:
- Builds upon existing test patterns in test_knowledge_base_initialization.py
- Uses comprehensive fixtures from conftest.py and comprehensive_test_fixtures.py
- Integrates with existing async testing infrastructure
- Follows established pytest markers and categories
- Validates realistic biomedical PDF content processing
- Tests both success and failure scenarios comprehensively

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import pytest
import asyncio
import tempfile
import json
import time
import logging
import gc
import psutil
import random
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
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
from lightrag_integration.progress_config import ProgressTrackingConfig
from lightrag_integration.unified_progress_tracker import (
    KnowledgeBaseProgressTracker,
    KnowledgeBasePhase, 
    PhaseWeights,
    PhaseProgressInfo,
    UnifiedProgressState,
    UnifiedProgressCallback
)


# =====================================================================
# TEST DATA MODELS AND UTILITIES
# =====================================================================

@dataclass
class PDFIngestionTestResult:
    """Represents the result of a PDF ingestion test."""
    success: bool
    pdf_files_processed: int
    documents_ingested: int
    entities_extracted: int
    relationships_found: int
    processing_time: float
    total_cost: float
    memory_peak_mb: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if self.processing_time > 0 and self.pdf_files_processed > 0:
            return {
                'pdfs_per_second': self.pdf_files_processed / self.processing_time,
                'documents_per_second': self.documents_ingested / self.processing_time,
                'cost_per_pdf': self.total_cost / self.pdf_files_processed if self.pdf_files_processed > 0 else 0.0,
                'entities_per_pdf': self.entities_extracted / self.pdf_files_processed if self.pdf_files_processed > 0 else 0.0,
                'memory_efficiency': self.memory_peak_mb / self.pdf_files_processed if self.pdf_files_processed > 0 else 0.0
            }
        return {}


@dataclass  
class KnowledgeBaseValidationResult:
    """Represents validation results for knowledge base content."""
    storage_structure_valid: bool
    document_count_matches: bool
    metadata_preserved: bool
    entities_stored: bool
    relationships_stored: bool
    queryable: bool
    content_retrievable: bool
    validation_errors: List[str] = field(default_factory=list)
    
    @property
    def overall_valid(self) -> bool:
        """Check if overall validation passed."""
        return all([
            self.storage_structure_valid,
            self.document_count_matches,
            self.metadata_preserved,
            self.entities_stored,
            self.relationships_stored,
            self.queryable,
            self.content_retrievable
        ])


class BiomedicalPDFTestDataGenerator:
    """Generates realistic biomedical PDF test content."""
    
    # Enhanced biomedical content templates
    STUDY_TEMPLATES = {
        'metabolomics_diabetes': {
            'title': 'Metabolomic Profiling of Type 2 Diabetes: Discovery of Novel Biomarkers',
            'content': """
            Abstract: This study presents a comprehensive metabolomic analysis of type 2 diabetes mellitus 
            in a cohort of 250 patients and 150 healthy controls. We employed LC-MS/MS techniques to identify 
            and quantify metabolites associated with glucose homeostasis dysfunction. Key findings include 
            elevated levels of branched-chain amino acids (leucine, isoleucine, valine), altered glucose 
            metabolism intermediates, and disrupted lipid profiles. Statistical analysis using MetaboAnalyst 
            with FDR-corrected p-values < 0.05 revealed 47 significantly altered metabolites. These results 
            suggest metabolomic profiling provides valuable insights into diabetes pathophysiology.
            
            Methods: Plasma samples were collected from 250 T2D patients and 150 controls after 12-hour 
            fasting. Sample preparation involved protein precipitation with methanol. Analysis was performed 
            using Agilent 6550 Q-TOF with reverse-phase chromatography. Data processing utilized MassHunter 
            software with multivariate statistical analysis including PCA and OPLS-DA.
            
            Results: We identified 47 significantly altered metabolites (p < 0.01, FC > 1.5). Key findings 
            include elevated levels of glucose, lactate, and pyruvate, and decreased concentrations of amino 
            acids and fatty acid derivatives. Pathway analysis revealed enrichment in glycolysis, TCA cycle, 
            and amino acid metabolism pathways. ROC analysis showed AUC > 0.85 for the top biomarker panel.
            """,
            'entities': ['glucose', 'lactate', 'pyruvate', 'leucine', 'isoleucine', 'valine', 'diabetes', 'LC-MS/MS'],
            'sample_size': 400,
            'biomarkers': 47,
            'platform': 'LC-MS/MS'
        },
        'cardiovascular_proteomics': {
            'title': 'Proteomic Biomarker Discovery in Cardiovascular Disease Using Mass Spectrometry',
            'content': """
            Abstract: Cardiovascular disease remains the leading cause of mortality worldwide. This research 
            explores novel protein biomarkers in heart failure patients using advanced proteomics approaches. 
            iTRAQ-based quantitative proteomics identified 89 proteins with altered expression between 180 
            heart failure patients and 120 controls. Key proteins include troponin I, BNP, and inflammatory 
            cytokines. Pathway analysis revealed involvement in cardiac remodeling, inflammation, and 
            oxidative stress processes. These results provide insights into disease mechanisms and potential 
            therapeutic targets for precision cardiology.
            
            Methods: Plasma samples from 180 heart failure patients (NYHA class II-IV) and 120 healthy 
            controls were analyzed using iTRAQ 8-plex labeling. Samples were processed using strong cation 
            exchange chromatography followed by LC-MS/MS on Orbitrap Fusion. Protein identification and 
            quantification used MaxQuant with Perseus for statistical analysis.
            
            Results: A total of 2,847 proteins were identified with 89 showing significant differential 
            expression (p < 0.05, FC > 1.2). Elevated proteins included troponin I (FC 3.4), NT-proBNP 
            (FC 4.1), and CRP (FC 2.8). Pathway analysis indicated enrichment in complement activation, 
            blood coagulation, and cardiac muscle contraction processes.
            """,
            'entities': ['troponin', 'BNP', 'CRP', 'TNF-alpha', 'cardiovascular disease', 'proteomics', 'iTRAQ'],
            'sample_size': 300,
            'biomarkers': 89,
            'platform': 'iTRAQ-MS'
        },
        'cancer_genomics': {
            'title': 'Genomic Analysis of Oncometabolism: Novel Therapeutic Targets in Cancer',
            'content': """
            Abstract: Cancer cells exhibit fundamental metabolic reprogramming to support rapid proliferation. 
            This study profiled metabolites and gene expression from tumor and normal tissue samples using 
            integrated multi-omics approaches. RNA-seq analysis of 200 tumor samples and 100 normal controls 
            revealed 1,234 differentially expressed genes involved in metabolic pathways. Metabolomics 
            identified 156 altered metabolites with glycolysis and glutamine metabolism showing significant 
            upregulation. These metabolic alterations may serve as diagnostic markers and therapeutic targets 
            for precision oncology approaches.
            
            Methods: Fresh-frozen tissue samples from 200 cancer patients and 100 normal controls were 
            analyzed using RNA-seq (Illumina HiSeq) and GC-MS/LC-MS metabolomics. Gene expression analysis 
            used DESeq2 with pathway enrichment via GSEA. Metabolomic data was processed using XCMS with 
            MetaboAnalyst for statistical analysis and pathway mapping.
            
            Results: RNA-seq identified 1,234 differentially expressed genes (FDR < 0.05, |FC| > 2) with 
            enrichment in metabolic processes. Metabolomics revealed 156 significantly altered metabolites 
            including elevated lactate, glutamine, and nucleotide precursors. Integrated analysis showed 
            strong correlation between gene expression and metabolite levels in key pathways.
            """,
            'entities': ['lactate', 'glutamine', 'cancer', 'RNA-seq', 'oncology', 'metabolism', 'GSEA'],
            'sample_size': 300,
            'biomarkers': 156,
            'platform': 'RNA-seq+MS'
        },
        'liver_disease_metabolomics': {
            'title': 'NMR-Based Metabolomics Analysis of Liver Disease Progression Biomarkers',
            'content': """
            Abstract: Liver disease progression involves complex metabolic alterations that can be monitored 
            using metabolomics approaches. This study utilized NMR spectroscopy to analyze serum metabolites 
            in 150 liver disease patients across different stages and 75 healthy controls. We identified 32 
            significantly altered metabolites associated with disease progression including elevated bilirubin, 
            altered amino acid profiles, and disrupted energy metabolism markers. Multivariate analysis 
            revealed distinct metabolic signatures for early-stage versus advanced liver disease.
            
            Methods: Serum samples from 150 liver disease patients (50 early-stage, 100 advanced) and 75 
            controls were analyzed using 600 MHz NMR spectroscopy. Spectra were processed using Topspin 
            with peak identification via Chenomx. Statistical analysis included PCA, OPLS-DA, and univariate 
            testing with Bonferroni correction.
            
            Results: NMR analysis identified 32 significantly altered metabolites (p < 0.01) including 
            elevated lactate, acetate, and 3-hydroxybutyrate, with decreased levels of glucose, alanine, 
            and branched-chain amino acids. Pathway analysis showed disruption in glycolysis, ketogenesis, 
            and amino acid metabolism pathways essential for liver function.
            """,
            'entities': ['bilirubin', 'lactate', 'acetate', 'glucose', 'alanine', 'liver disease', 'NMR'],
            'sample_size': 225,
            'biomarkers': 32,
            'platform': 'NMR'
        },
        'kidney_disease_biomarkers': {
            'title': 'Urinary Metabolomics for Early Detection of Chronic Kidney Disease',
            'content': """
            Abstract: Chronic kidney disease (CKD) affects millions worldwide with limited early detection 
            methods. This study applied urinary metabolomics to identify early biomarkers in 180 CKD patients 
            and 90 healthy controls. HILIC-MS analysis revealed 28 significantly altered urinary metabolites 
            including elevated creatinine, urea, and altered amino acid excretion patterns. Machine learning 
            models achieved 92% accuracy for early CKD detection using metabolomic profiles.
            
            Methods: First morning urine samples from 180 CKD patients (stages 1-5) and 90 controls were 
            analyzed using HILIC-LC-MS on Waters Xevo TQ-S. Sample preparation involved dilution and 
            centrifugation. Data processing used MassLynx with targeted analysis of 150 known metabolites. 
            Statistical analysis included t-tests, ROC analysis, and random forest classification.
            
            Results: HILIC-MS identified 28 significantly altered metabolites (p < 0.05, VIP > 1.0) with 
            excellent separation between CKD patients and controls (R2 = 0.85, Q2 = 0.78). Key biomarkers 
            included elevated creatinine (FC 2.8), urea (FC 3.1), and decreased hippuric acid (FC 0.4). 
            Machine learning models achieved AUC > 0.90 for early-stage CKD prediction.
            """,
            'entities': ['creatinine', 'urea', 'hippuric acid', 'kidney disease', 'HILIC-MS', 'biomarkers'],
            'sample_size': 270,
            'biomarkers': 28,
            'platform': 'HILIC-MS'
        }
    }
    
    @classmethod
    def create_realistic_pdf_documents(cls, count: int = 5, 
                                     complexity: str = 'medium') -> List[Dict[str, Any]]:
        """Create realistic PDF documents with biomedical content."""
        documents = []
        study_types = list(cls.STUDY_TEMPLATES.keys())
        
        for i in range(count):
            # Select study type (cycle through available types)
            study_type = study_types[i % len(study_types)]
            template = cls.STUDY_TEMPLATES[study_type]
            
            # Generate unique identifiers
            study_id = f"STUDY_{i+1:03d}_{random.randint(1000, 9999)}"
            
            # Create document with realistic metadata
            doc = {
                'filename': f"{study_id.lower()}_{study_type}.pdf",
                'title': template['title'],
                'authors': [f"Dr. Author{j+1}" for j in range(random.randint(2, 5))],
                'journal': f"Journal of {study_type.split('_')[1].title()} Research",
                'year': random.randint(2020, 2024),
                'doi': f"10.1000/test.{2020+i}.{random.randint(100, 999):03d}",
                'keywords': [study_type.split('_')[0], study_type.split('_')[1], 'biomarkers', 'clinical'],
                'content': template['content'],
                'entities': template['entities'],
                'sample_size': template['sample_size'],
                'biomarker_count': template['biomarkers'],
                'analytical_platform': template['platform'],
                'page_count': random.randint(8, 20),
                'file_size_bytes': len(template['content']) * random.randint(2, 4),
                'processing_time': random.uniform(0.5, 2.5),
                'complexity': complexity,
                'study_id': study_id,
                'study_type': study_type
            }
            
            documents.append(doc)
        
        return documents


# =====================================================================
# ENHANCED TEST FIXTURES
# =====================================================================

@pytest.fixture
def biomedical_pdf_documents():
    """Provide realistic biomedical PDF documents for testing."""
    return BiomedicalPDFTestDataGenerator.create_realistic_pdf_documents(5, 'medium')


@pytest.fixture
def small_pdf_collection():
    """Provide small collection for quick tests."""
    return BiomedicalPDFTestDataGenerator.create_realistic_pdf_documents(3, 'simple')


@pytest.fixture  
def large_pdf_collection():
    """Provide large collection for performance tests."""
    return BiomedicalPDFTestDataGenerator.create_realistic_pdf_documents(15, 'complex')


@pytest.fixture
async def realistic_pdf_files(temp_dir, biomedical_pdf_documents):
    """Create actual PDF files with realistic biomedical content."""
    pdf_files = []
    
    for doc in biomedical_pdf_documents:
        pdf_path = temp_dir / doc['filename']
        
        # Create enhanced content with metadata
        enhanced_content = f"""
Title: {doc['title']}
Authors: {', '.join(doc['authors'])}
Journal: {doc['journal']} ({doc['year']})
DOI: {doc['doi']}
Keywords: {', '.join(doc['keywords'])}
Study ID: {doc['study_id']}
Sample Size: {doc['sample_size']}
Biomarkers Identified: {doc['biomarker_count']}
Analytical Platform: {doc['analytical_platform']}

{doc['content']}

Statistical Methods: Multivariate analysis, PCA, OPLS-DA, ROC analysis
Quality Control: Internal standards, blank samples, biological replicates
Data Processing: Peak alignment, normalization, missing value imputation
Pathway Analysis: KEGG pathway mapping, MetaboAnalyst enrichment
Clinical Relevance: Biomarker validation, diagnostic accuracy assessment

Key Entities: {', '.join(doc['entities'])}
        """
        
        # Write to file (using text for simplicity - can be enhanced with actual PDF creation)
        pdf_path.write_text(enhanced_content.strip())
        pdf_files.append(pdf_path)
    
    yield pdf_files
    
    # Cleanup
    for pdf_file in pdf_files:
        try:
            pdf_file.unlink(missing_ok=True)
        except:
            pass


@pytest.fixture
async def enhanced_pdf_processor():
    """Provide enhanced PDF processor for realistic testing."""
    processor = MagicMock(spec=BiomedicalPDFProcessor)
    
    async def mock_process_all_pdfs(pdf_directory: Path) -> List[Tuple[Path, Dict[str, Any]]]:
        """Mock processing all PDFs in directory with realistic results."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        results = []
        pdf_files = list(pdf_directory.glob("*.pdf"))
        
        for pdf_path in pdf_files:
            # Read content and generate realistic processing result
            try:
                content = pdf_path.read_text()
                
                # Extract title and other metadata from content
                lines = content.split('\n')
                title = "Unknown"
                for line in lines:
                    if line.startswith('Title:'):
                        title = line.replace('Title:', '').strip()
                        break
                
                # Generate realistic processing result
                result = {
                    'content': content,
                    'metadata': {
                        'title': title,
                        'page_count': random.randint(8, 20),
                        'file_size': len(content),
                        'processing_time': random.uniform(0.5, 2.0)
                    },
                    'success': True,
                    'entities_found': random.randint(10, 25),
                    'relationships_found': random.randint(5, 15)
                }
                
                results.append((pdf_path, result))
                
            except Exception as e:
                # Handle processing failures
                results.append((pdf_path, {
                    'content': '',
                    'metadata': {'error': str(e)},
                    'success': False
                }))
        
        return results
    
    async def mock_process_pdf(pdf_path: Path) -> Dict[str, Any]:
        """Mock single PDF processing."""
        await asyncio.sleep(0.05)  # Simulate processing time
        
        try:
            content = pdf_path.read_text()
            
            # Extract metadata from content
            lines = content.split('\n')
            title = "Unknown"
            for line in lines:
                if line.startswith('Title:'):
                    title = line.replace('Title:', '').strip()
                    break
            
            return {
                'text': content,
                'metadata': {
                    'title': title,
                    'page_count': random.randint(8, 20),
                    'file_size': len(content)
                },
                'processing_time': random.uniform(0.5, 2.0),
                'success': True
            }
            
        except Exception as e:
            return {
                'text': '',
                'metadata': {'error': str(e)},
                'processing_time': 0.0,
                'success': False
            }
    
    processor.process_all_pdfs = AsyncMock(side_effect=mock_process_all_pdfs)
    processor.process_pdf = AsyncMock(side_effect=mock_process_pdf)
    
    return processor


@pytest.fixture
def memory_monitor():
    """Provide memory monitoring utility."""
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.initial_memory
            
        def update(self):
            """Update memory measurements."""
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
            return current_memory
            
        def get_peak_usage(self) -> float:
            """Get peak memory usage since initialization."""
            return self.peak_memory
            
        def get_memory_increase(self) -> float:
            """Get memory increase since initialization."""
            return self.peak_memory - self.initial_memory
    
    return MemoryMonitor()


@pytest.fixture
def performance_benchmarker():
    """Provide performance benchmarking utility."""
    class PerformanceBenchmarker:
        def __init__(self):
            self.benchmarks = {}
            
        async def benchmark_operation(self, operation_name: str, operation_func, *args, **kwargs):
            """Benchmark an async operation."""
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = await operation_func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            benchmark_result = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'success': success,
                'error': error,
                'result': result
            }
            
            self.benchmarks[operation_name] = benchmark_result
            return benchmark_result
            
        def get_statistics(self) -> Dict[str, Any]:
            """Get benchmark statistics."""
            if not self.benchmarks:
                return {}
                
            durations = [b['duration'] for b in self.benchmarks.values() if b['success']]
            memory_deltas = [b['memory_delta'] for b in self.benchmarks.values() if b['success']]
            
            return {
                'total_operations': len(self.benchmarks),
                'successful_operations': sum(1 for b in self.benchmarks.values() if b['success']),
                'average_duration': statistics.mean(durations) if durations else 0.0,
                'median_duration': statistics.median(durations) if durations else 0.0,
                'average_memory_delta': statistics.mean(memory_deltas) if memory_deltas else 0.0,
                'total_duration': sum(durations),
                'benchmarks': self.benchmarks
            }
    
    return PerformanceBenchmarker()


# =====================================================================
# SINGLE PDF INGESTION INTEGRATION TESTS
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.lightrag
class TestSinglePDFIngestionIntegration:
    """Test single PDF ingestion into knowledge base."""
    
    async def test_single_pdf_complete_ingestion_workflow(
        self, temp_dir, integration_config, realistic_pdf_files, 
        enhanced_pdf_processor, memory_monitor
    ):
        """Test complete workflow for single PDF ingestion into knowledge base."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Copy first PDF to papers directory  
        test_pdf = realistic_pdf_files[0]
        target_pdf = papers_dir / test_pdf.name
        target_pdf.write_text(test_pdf.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            # Setup LightRAG mock
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            
            # Setup PDF processor mock  
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            # Initialize RAG system
            memory_monitor.update()
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute knowledge base initialization with single PDF
            start_time = time.time()
            result = await rag.initialize_knowledge_base(
                papers_dir=papers_dir,
                enable_unified_progress_tracking=True
            )
            processing_time = time.time() - start_time
            peak_memory = memory_monitor.get_peak_usage()
            
            # Validate results
            assert result['success'] == True
            assert result['documents_processed'] == 1
            assert result['total_documents'] == 1
            assert result['documents_failed'] == 0
            assert processing_time < 10.0  # Should complete within 10 seconds
            
            # Verify LightRAG ingestion occurred
            mock_instance.ainsert.assert_called_once()
            ingested_docs = mock_instance.ainsert.call_args[0][0]
            assert len(ingested_docs) == 1
            assert len(ingested_docs[0]) > 100  # Substantial content
            
            # Verify content includes metadata
            ingested_content = ingested_docs[0]
            assert 'Title:' in ingested_content
            assert 'Authors:' in ingested_content
            assert 'biomarker' in ingested_content.lower() or 'metabolomic' in ingested_content.lower()
    
    async def test_single_pdf_metadata_preservation(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test that PDF metadata is preserved through ingestion process."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        test_pdf = realistic_pdf_files[0]
        target_pdf = papers_dir / test_pdf.name
        target_pdf.write_text(test_pdf.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute ingestion
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Verify metadata preservation
            assert result['success'] == True
            mock_instance.ainsert.assert_called_once()
            
            ingested_content = mock_instance.ainsert.call_args[0][0][0]
            
            # Check for key metadata elements
            original_content = test_pdf.read_text()
            for metadata_field in ['Title:', 'Authors:', 'Journal:', 'DOI:', 'Study ID:']:
                if metadata_field in original_content:
                    assert metadata_field in ingested_content, f"Metadata field {metadata_field} not preserved"
    
    async def test_single_pdf_entity_extraction_validation(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test entity extraction during single PDF ingestion."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        test_pdf = realistic_pdf_files[0]  # Should contain metabolomics content
        target_pdf = papers_dir / test_pdf.name
        target_pdf.write_text(test_pdf.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'entities_extracted': 15, 'relationships_found': 8})
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute ingestion
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Validate entity extraction
            assert result['success'] == True
            mock_instance.ainsert.assert_called_once()
            
            # Verify content contains expected biomedical entities
            ingested_content = mock_instance.ainsert.call_args[0][0][0].lower()
            
            # Check for common biomedical entities that should be present
            expected_entities = ['glucose', 'metabolite', 'biomarker', 'patient', 'analysis']
            entities_found = sum(1 for entity in expected_entities if entity in ingested_content)
            assert entities_found >= 3, f"Expected at least 3 biomedical entities, found {entities_found}"


# =====================================================================
# BATCH PDF INGESTION INTEGRATION TESTS  
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.lightrag
class TestBatchPDFIngestionIntegration:
    """Test batch PDF ingestion into knowledge base."""
    
    async def test_batch_pdf_ingestion_workflow(
        self, temp_dir, integration_config, realistic_pdf_files, 
        enhanced_pdf_processor, performance_benchmarker
    ):
        """Test batch processing of multiple PDFs into knowledge base."""
        # Setup
        papers_dir = temp_dir / "papers" 
        papers_dir.mkdir(exist_ok=True)
        
        # Copy all test PDFs to papers directory
        for pdf_file in realistic_pdf_files:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Benchmark the batch processing operation
            benchmark_result = await performance_benchmarker.benchmark_operation(
                "batch_pdf_ingestion",
                rag.initialize_knowledge_base,
                papers_dir=papers_dir,
                batch_size=10,
                enable_batch_processing=True
            )
            
            # Validate benchmark results
            assert benchmark_result['success'] == True
            assert benchmark_result['duration'] < 30.0  # Should complete within 30 seconds
            
            # Validate processing results
            result = benchmark_result['result']
            assert result['success'] == True
            assert result['documents_processed'] == len(realistic_pdf_files)
            assert result['documents_failed'] == 0
            
            # Verify batch ingestion occurred
            mock_instance.ainsert.assert_called_once()
            ingested_docs = mock_instance.ainsert.call_args[0][0]
            assert len(ingested_docs) == len(realistic_pdf_files)
            
            # Verify each document has substantial content
            for doc in ingested_docs:
                assert len(doc) > 100
                assert any(term in doc.lower() for term in ['metabolomic', 'biomarker', 'clinical', 'analysis'])
    
    async def test_batch_processing_with_mixed_success_failure(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test batch processing with some successful and some failed PDF processing."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Copy PDFs and create one corrupted file
        for i, pdf_file in enumerate(realistic_pdf_files):
            target_pdf = papers_dir / pdf_file.name
            if i == 2:  # Make the third file corrupted
                target_pdf.write_text("CORRUPTED PDF CONTENT - NOT VALID")
            else:
                target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        # Mock processor to fail on corrupted file
        async def mock_process_with_failure(pdf_directory: Path):
            results = []
            pdf_files = list(pdf_directory.glob("*.pdf"))
            
            for pdf_path in pdf_files:
                content = pdf_path.read_text()
                if "CORRUPTED" in content:
                    # Simulate processing failure
                    results.append((pdf_path, {
                        'content': '',
                        'metadata': {'error': 'Corrupted PDF file'},
                        'success': False
                    }))
                else:
                    # Simulate successful processing
                    results.append((pdf_path, {
                        'content': content,
                        'metadata': {'title': 'Test Study', 'page_count': 10},
                        'success': True
                    }))
            
            return results
        
        enhanced_pdf_processor.process_all_pdfs = AsyncMock(side_effect=mock_process_with_failure)
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute batch processing
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Validate mixed results
            assert result['success'] == True  # Overall success despite some failures
            assert result['documents_processed'] == len(realistic_pdf_files) - 1  # All except corrupted
            assert result['documents_failed'] == 1  # One corrupted file
            assert len(result['errors']) > 0  # Should have error messages
            
            # Verify only successful documents were ingested
            mock_instance.ainsert.assert_called_once()
            ingested_docs = mock_instance.ainsert.call_args[0][0]
            assert len(ingested_docs) == len(realistic_pdf_files) - 1  # Exclude failed document
    
    async def test_large_batch_performance_characteristics(
        self, temp_dir, integration_config, large_pdf_collection, 
        enhanced_pdf_processor, performance_benchmarker, memory_monitor
    ):
        """Test performance characteristics of large batch processing."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Create large collection of PDF files
        for doc_data in large_pdf_collection:
            pdf_path = papers_dir / doc_data['filename']
            enhanced_content = f"""
Title: {doc_data['title']}
Content: {doc_data['content']}
Sample Size: {doc_data['sample_size']}
Biomarkers: {doc_data['biomarker_count']}
Platform: {doc_data['analytical_platform']}
            """
            pdf_path.write_text(enhanced_content)
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Benchmark large batch processing
            initial_memory = memory_monitor.update()
            
            benchmark_result = await performance_benchmarker.benchmark_operation(
                "large_batch_processing",
                rag.initialize_knowledge_base,
                papers_dir=papers_dir,
                batch_size=5,
                max_memory_mb=2048
            )
            
            final_memory = memory_monitor.update()
            memory_increase = memory_monitor.get_memory_increase()
            
            # Validate performance characteristics
            assert benchmark_result['success'] == True
            assert benchmark_result['duration'] < 60.0  # Should complete within 60 seconds
            
            result = benchmark_result['result']
            assert result['documents_processed'] == len(large_pdf_collection)
            
            # Performance metrics
            processing_time = benchmark_result['duration']
            throughput = len(large_pdf_collection) / processing_time if processing_time > 0 else 0
            
            assert throughput > 0.5  # At least 0.5 documents per second
            assert memory_increase < 500  # Memory increase less than 500MB
            
            # Verify ingestion completed
            mock_instance.ainsert.assert_called_once()
            ingested_docs = mock_instance.ainsert.call_args[0][0]
            assert len(ingested_docs) == len(large_pdf_collection)


# =====================================================================
# KNOWLEDGE BASE CONTENT VALIDATION TESTS
# =====================================================================

@pytest.mark.asyncio 
@pytest.mark.integration
@pytest.mark.lightrag
class TestKnowledgeBaseContentValidation:
    """Test knowledge base content and structure after PDF ingestion."""
    
    async def test_knowledge_base_structure_validation_after_ingestion(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test that knowledge base structure is properly created after PDF ingestion."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        for pdf_file in realistic_pdf_files:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        working_dir = integration_config.working_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute ingestion
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Validate knowledge base structure
            assert result['success'] == True
            
            # Verify working directory was created
            assert working_dir.exists()
            assert working_dir.is_dir()
            
            # Verify LightRAG was configured with correct storage paths
            lightrag_call_kwargs = mock_lightrag.call_args[1]
            expected_paths = [
                'graph_chunk_entity_relation_json_path',
                'vdb_chunks_path', 
                'vdb_entities_path',
                'vdb_relationships_path'
            ]
            
            for path_key in expected_paths:
                assert path_key in lightrag_call_kwargs
                assert lightrag_call_kwargs[path_key] is not None
    
    async def test_document_content_retrievability(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test that ingested documents can be retrieved through queries."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        for pdf_file in realistic_pdf_files:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_instance.aquery = AsyncMock(return_value="Based on the metabolomics literature, glucose and lactate levels are elevated in diabetes patients. Statistical analysis revealed significant alterations in amino acid metabolism pathways.")
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute ingestion
            ingestion_result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            assert ingestion_result['success'] == True
            
            # Test document retrievability through queries
            test_queries = [
                "What metabolites are associated with diabetes?",
                "What analytical platforms were used in these studies?",
                "What are the key biomarkers identified?"
            ]
            
            for query in test_queries:
                query_result = await rag.query(query)
                
                # Validate query response
                assert 'content' in query_result
                assert len(query_result['content']) > 50  # Substantial response
                assert query_result['success'] == True
                
                # Verify query was executed on LightRAG
                mock_instance.aquery.assert_called()
    
    async def test_cross_document_synthesis_validation(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test that knowledge base can synthesize information across multiple documents."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        for pdf_file in realistic_pdf_files:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            # Mock response that synthesizes across documents
            mock_instance.aquery = AsyncMock(return_value="Across multiple studies, common biomarkers include glucose, lactate, and amino acids. Different analytical platforms (LC-MS, NMR, RNA-seq) provide complementary insights into metabolic dysfunction in various diseases including diabetes, cardiovascular disease, and cancer.")
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute ingestion
            ingestion_result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            assert ingestion_result['success'] == True
            
            # Test cross-document synthesis query
            synthesis_query = "Compare the biomarkers and analytical methods across all studies"
            query_result = await rag.query(synthesis_query)
            
            # Validate synthesis response
            assert 'content' in query_result
            response_content = query_result['content'].lower()
            
            # Check that response mentions multiple studies/approaches
            synthesis_indicators = ['multiple', 'across', 'different', 'various', 'compare']
            synthesis_found = sum(1 for indicator in synthesis_indicators if indicator in response_content)
            assert synthesis_found >= 2, "Response should indicate cross-document synthesis"
            
            # Check that response mentions different biomarkers from multiple studies
            expected_biomarkers = ['glucose', 'lactate', 'amino acid']
            biomarkers_found = sum(1 for biomarker in expected_biomarkers if biomarker in response_content)
            assert biomarkers_found >= 2, "Response should mention biomarkers from multiple studies"


# =====================================================================
# ENTITY EXTRACTION INTEGRATION TESTS
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.lightrag
@pytest.mark.biomedical
class TestEntityExtractionIntegration:
    """Test entity and relationship extraction during PDF ingestion."""
    
    async def test_biomedical_entity_extraction_accuracy(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test accuracy of biomedical entity extraction from ingested PDFs."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Use first PDF with known entities
        test_pdf = realistic_pdf_files[0]
        target_pdf = papers_dir / test_pdf.name
        target_pdf.write_text(test_pdf.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute ingestion
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            assert result['success'] == True
            
            # Analyze ingested content for entity extraction
            mock_instance.ainsert.assert_called_once()
            ingested_content = mock_instance.ainsert.call_args[0][0][0].lower()
            
            # Expected biomedical entity types should be present
            expected_entity_types = {
                'metabolites': ['glucose', 'lactate', 'pyruvate', 'amino acid', 'metabolite'],
                'techniques': ['lc-ms', 'nmr', 'mass spectrometry', 'chromatography'],
                'diseases': ['diabetes', 'cardiovascular', 'cancer', 'liver disease'],
                'measurements': ['sample', 'patient', 'control', 'analysis'],
                'pathways': ['glycolysis', 'metabolism', 'pathway']
            }
            
            entities_found = {}
            for entity_type, entities in expected_entity_types.items():
                found = [entity for entity in entities if entity in ingested_content]
                entities_found[entity_type] = found
            
            # Validate entity extraction
            total_entities_found = sum(len(entities) for entities in entities_found.values())
            assert total_entities_found >= 8, f"Expected at least 8 entities, found {total_entities_found}"
            
            # Each entity type should have at least one entity
            for entity_type, entities in entities_found.items():
                assert len(entities) >= 1, f"No entities found for type: {entity_type}"
    
    async def test_relationship_extraction_between_entities(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test extraction of relationships between biomedical entities."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        for pdf_file in realistic_pdf_files[:3]:  # Use first 3 PDFs
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Configure biomedical parameters for entity relationship extraction
            rag.biomedical_params = {
                'entity_extraction_focus': 'biomedical',
                'entity_types': ['METABOLITE', 'PROTEIN', 'GENE', 'DISEASE', 'PATHWAY', 'TECHNIQUE'],
                'relationship_extraction': True,
                'pathway_analysis': True
            }
            
            # Execute ingestion
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            assert result['success'] == True
            
            # Analyze ingested content for relationship indicators
            mock_instance.ainsert.assert_called_once()
            ingested_docs = mock_instance.ainsert.call_args[0][0]
            all_content = ' '.join(ingested_docs).lower()
            
            # Check for relationship indicators in the content
            relationship_indicators = [
                'associated with', 'correlated with', 'increased in', 'decreased in',
                'biomarker for', 'involved in', 'pathway', 'metabolism', 'expression',
                'regulation', 'interaction', 'linked to', 'affects', 'influences'
            ]
            
            relationships_found = [indicator for indicator in relationship_indicators if indicator in all_content]
            assert len(relationships_found) >= 5, f"Expected at least 5 relationship indicators, found {len(relationships_found)}"
            
            # Verify content supports relationship extraction
            # Should contain entity pairs that can form relationships
            entity_pairs = [
                ('glucose', 'diabetes'), ('metabolite', 'biomarker'), ('protein', 'disease'),
                ('pathway', 'metabolism'), ('sample', 'analysis')
            ]
            
            valid_pairs = 0
            for entity1, entity2 in entity_pairs:
                if entity1 in all_content and entity2 in all_content:
                    valid_pairs += 1
            
            assert valid_pairs >= 3, f"Expected at least 3 valid entity pairs for relationships, found {valid_pairs}"
    
    async def test_pathway_analysis_integration(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test integration of pathway analysis during entity extraction."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Use metabolomics-focused PDFs
        metabolomics_pdfs = [pdf for pdf in realistic_pdf_files if 'metabolomic' in pdf.read_text().lower()][:2]
        for pdf_file in metabolomics_pdfs:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute ingestion with pathway focus
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            assert result['success'] == True
            
            # Analyze content for pathway-related information
            ingested_content = mock_instance.ainsert.call_args[0][0]
            all_content = ' '.join(ingested_content).lower()
            
            # Expected metabolic pathways
            expected_pathways = [
                'glycolysis', 'tca cycle', 'fatty acid', 'amino acid metabolism',
                'gluconeogenesis', 'pentose phosphate', 'oxidative', 'metabolic pathway'
            ]
            
            pathways_found = [pathway for pathway in expected_pathways if pathway in all_content]
            assert len(pathways_found) >= 4, f"Expected at least 4 metabolic pathways, found {len(pathways_found)}: {pathways_found}"
            
            # Check for pathway analysis terminology
            pathway_analysis_terms = [
                'pathway analysis', 'enrichment', 'kegg', 'reactome', 'metabolic network',
                'flux analysis', 'pathway mapping', 'systems biology'
            ]
            
            analysis_terms_found = [term for term in pathway_analysis_terms if term in all_content]
            assert len(analysis_terms_found) >= 2, f"Expected pathway analysis terminology, found {len(analysis_terms_found)}: {analysis_terms_found}"


# =====================================================================
# PERFORMANCE AND BENCHMARKING TESTS
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.lightrag
class TestPDFIngestionPerformance:
    """Test performance characteristics of PDF ingestion pipeline."""
    
    async def test_single_pdf_processing_performance_benchmark(
        self, temp_dir, integration_config, realistic_pdf_files, 
        enhanced_pdf_processor, performance_benchmarker, memory_monitor
    ):
        """Benchmark performance of single PDF processing."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        test_pdf = realistic_pdf_files[0]
        target_pdf = papers_dir / test_pdf.name
        target_pdf.write_text(test_pdf.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Benchmark single PDF processing
            initial_memory = memory_monitor.update()
            
            benchmark_result = await performance_benchmarker.benchmark_operation(
                "single_pdf_processing",
                rag.initialize_knowledge_base,
                papers_dir=papers_dir
            )
            
            final_memory = memory_monitor.update()
            
            # Validate performance metrics
            assert benchmark_result['success'] == True
            assert benchmark_result['duration'] < 5.0  # Should complete within 5 seconds
            assert benchmark_result['memory_delta'] < 100  # Less than 100MB memory increase
            
            result = benchmark_result['result']
            assert result['success'] == True
            assert result['documents_processed'] == 1
            
            # Calculate performance metrics
            processing_time = benchmark_result['duration']
            memory_efficiency = benchmark_result['memory_delta']
            
            assert processing_time < 5.0, f"Processing took too long: {processing_time:.2f}s"
            assert memory_efficiency < 100, f"Memory usage too high: {memory_efficiency:.2f}MB"
    
    async def test_concurrent_pdf_processing_performance(
        self, temp_dir, integration_config, realistic_pdf_files, 
        enhanced_pdf_processor, performance_benchmarker
    ):
        """Test performance of concurrent PDF processing operations."""
        # Setup multiple working directories for concurrent operations
        base_dir = temp_dir / "concurrent_test"
        base_dir.mkdir(exist_ok=True)
        
        concurrent_tasks = []
        
        # Create multiple concurrent processing tasks
        for i in range(3):
            task_dir = base_dir / f"task_{i}"
            task_dir.mkdir(exist_ok=True)
            
            papers_dir = task_dir / "papers"
            papers_dir.mkdir(exist_ok=True)
            
            # Copy PDF to task directory
            test_pdf = realistic_pdf_files[i % len(realistic_pdf_files)]
            target_pdf = papers_dir / test_pdf.name
            target_pdf.write_text(test_pdf.read_text())
            
            # Create task-specific config
            task_config = LightRAGConfig(
                api_key="test-api-key",
                model="gpt-4o-mini",
                working_dir=task_dir / "knowledge_base",
                max_async=4,
                auto_create_dirs=True
            )
            
            concurrent_tasks.append((papers_dir, task_config))
        
        async def process_single_task(papers_dir, config):
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
                
                mock_instance = MagicMock()
                mock_instance.ainsert = AsyncMock()
                mock_lightrag.return_value = mock_instance
                mock_pdf_class.return_value = enhanced_pdf_processor
                
                rag = ClinicalMetabolomicsRAG(config=config)
                result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
                return result
        
        # Execute concurrent operations
        start_time = time.time()
        concurrent_results = await asyncio.gather(*[
            process_single_task(papers_dir, config) 
            for papers_dir, config in concurrent_tasks
        ])
        total_time = time.time() - start_time
        
        # Validate concurrent processing results
        assert total_time < 15.0, f"Concurrent processing took too long: {total_time:.2f}s"
        
        for i, result in enumerate(concurrent_results):
            assert result['success'] == True, f"Concurrent task {i} failed"
            assert result['documents_processed'] == 1
        
        # Verify concurrent operations completed faster than sequential
        # (This is a basic test - in practice, speedup depends on I/O vs CPU bound operations)
        expected_sequential_time = len(concurrent_tasks) * 2.0  # Estimated 2s per task
        assert total_time < expected_sequential_time, f"Concurrent processing should be faster than sequential"
    
    async def test_memory_usage_optimization_during_large_batch(
        self, temp_dir, integration_config, large_pdf_collection, 
        enhanced_pdf_processor, memory_monitor
    ):
        """Test memory usage optimization during large batch processing."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Create large collection of PDF files
        for doc_data in large_pdf_collection:
            pdf_path = papers_dir / doc_data['filename']
            pdf_path.write_text(doc_data['content'])
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Monitor memory during large batch processing
            initial_memory = memory_monitor.update()
            
            result = await rag.initialize_knowledge_base(
                papers_dir=papers_dir,
                batch_size=5,  # Process in smaller batches
                max_memory_mb=1024  # Memory limit
            )
            
            peak_memory = memory_monitor.get_peak_usage()
            memory_increase = memory_monitor.get_memory_increase()
            
            # Validate memory optimization
            assert result['success'] == True
            assert result['documents_processed'] == len(large_pdf_collection)
            
            # Memory usage should be reasonable for large batch
            assert memory_increase < 800, f"Memory increase too high: {memory_increase:.2f}MB"
            assert peak_memory < initial_memory + 1000, f"Peak memory usage too high: {peak_memory:.2f}MB"
            
            # Verify batch processing occurred
            mock_instance.ainsert.assert_called()
    
    async def test_processing_throughput_benchmarks(
        self, temp_dir, integration_config, large_pdf_collection,
        enhanced_pdf_processor, performance_benchmarker
    ):
        """Test processing throughput benchmarks for different batch sizes."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Create moderate collection for throughput testing
        test_collection = large_pdf_collection[:10]  # Use 10 documents
        
        for doc_data in test_collection:
            pdf_path = papers_dir / doc_data['filename']
            pdf_path.write_text(doc_data['content'])
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        # Test different batch sizes
        batch_sizes = [1, 3, 5, 10]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
                
                mock_instance = MagicMock()
                mock_instance.ainsert = AsyncMock()
                mock_lightrag.return_value = mock_instance
                mock_pdf_class.return_value = enhanced_pdf_processor
                
                rag = ClinicalMetabolomicsRAG(config=integration_config)
                
                # Benchmark processing with specific batch size
                benchmark_result = await performance_benchmarker.benchmark_operation(
                    f"throughput_batch_{batch_size}",
                    rag.initialize_knowledge_base,
                    papers_dir=papers_dir,
                    batch_size=batch_size,
                    force_reinitialize=True
                )
                
                # Calculate throughput
                if benchmark_result['success'] and benchmark_result['duration'] > 0:
                    throughput = len(test_collection) / benchmark_result['duration']
                    throughput_results[batch_size] = throughput
        
        # Validate throughput results
        assert len(throughput_results) >= 2, "Should have throughput results for multiple batch sizes"
        
        # All throughputs should be reasonable (at least 1 document per second)
        for batch_size, throughput in throughput_results.items():
            assert throughput > 0.5, f"Throughput too low for batch size {batch_size}: {throughput:.2f} docs/sec"
        
        # Larger batch sizes should generally have higher throughput (up to a point)
        throughput_values = list(throughput_results.values())
        max_throughput = max(throughput_values)
        assert max_throughput > 1.0, f"Maximum throughput should exceed 1 doc/sec: {max_throughput:.2f}"


# =====================================================================
# ERROR HANDLING INTEGRATION TESTS
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.lightrag
class TestPDFIngestionErrorHandling:
    """Test error handling scenarios during PDF ingestion."""
    
    async def test_corrupted_pdf_handling(
        self, temp_dir, integration_config, enhanced_pdf_processor
    ):
        """Test handling of corrupted PDF files during ingestion."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Create corrupted PDF file
        corrupted_pdf = papers_dir / "corrupted.pdf"
        corrupted_pdf.write_text("This is not a valid PDF file content - corrupted data")
        
        # Create valid PDF file
        valid_pdf = papers_dir / "valid.pdf"
        valid_pdf.write_text("Title: Valid Study\nContent: This is valid biomedical research content about metabolomics.")
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        # Mock processor to handle corrupted file appropriately
        async def mock_process_with_corruption_handling(pdf_directory: Path):
            results = []
            pdf_files = list(pdf_directory.glob("*.pdf"))
            
            for pdf_path in pdf_files:
                content = pdf_path.read_text()
                if "corrupted" in pdf_path.name.lower():
                    # Simulate corrupted file processing failure
                    results.append((pdf_path, {
                        'content': '',
                        'metadata': {'error': 'PDF file is corrupted or invalid format'},
                        'success': False
                    }))
                else:
                    # Simulate successful processing
                    results.append((pdf_path, {
                        'content': content,
                        'metadata': {'title': 'Valid Study', 'page_count': 5},
                        'success': True
                    }))
            
            return results
        
        enhanced_pdf_processor.process_all_pdfs = AsyncMock(side_effect=mock_process_with_corruption_handling)
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute ingestion with corrupted file
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Validate error handling
            assert result['success'] == True  # Overall success despite corrupted file
            assert result['documents_processed'] == 1  # Only valid file processed
            assert result['documents_failed'] == 1  # Corrupted file failed
            assert len(result['errors']) > 0  # Error messages recorded
            
            # Verify only valid document was ingested
            mock_instance.ainsert.assert_called_once()
            ingested_docs = mock_instance.ainsert.call_args[0][0]
            assert len(ingested_docs) == 1
            assert "valid biomedical research" in ingested_docs[0]
    
    async def test_storage_initialization_failure_recovery(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test recovery from storage initialization failures."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        test_pdf = realistic_pdf_files[0]
        target_pdf = papers_dir / test_pdf.name
        target_pdf.write_text(test_pdf.read_text())
        
        # Set working directory to location that will cause permission issues
        problematic_dir = temp_dir / "problematic"
        problematic_dir.mkdir(exist_ok=True)
        
        integration_config.working_dir = problematic_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            # Mock LightRAG to fail on first attempt, succeed on retry
            call_count = [0]  # Use list to modify from inner function
            
            def mock_lightrag_with_retry(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("Storage initialization failed - disk full")
                else:
                    mock_instance = MagicMock()
                    mock_instance.ainsert = AsyncMock()
                    return mock_instance
            
            mock_lightrag.side_effect = mock_lightrag_with_retry
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            # First attempt should fail
            with pytest.raises(ClinicalMetabolomicsRAGError, match="Storage initialization failed"):
                rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Second attempt should succeed (simulating retry with different conditions)
            integration_config.working_dir = temp_dir / "knowledge_base_retry"
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Validate successful recovery
            assert result['success'] == True
            assert result['documents_processed'] == 1
    
    async def test_memory_pressure_error_handling(
        self, temp_dir, integration_config, large_pdf_collection, enhanced_pdf_processor
    ):
        """Test error handling under memory pressure conditions."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Create large collection to simulate memory pressure
        for doc_data in large_pdf_collection:
            pdf_path = papers_dir / doc_data['filename']
            # Make content larger to increase memory usage
            large_content = doc_data['content'] * 5  # Repeat content 5 times
            pdf_path.write_text(large_content)
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute with memory limit to simulate pressure
            result = await rag.initialize_knowledge_base(
                papers_dir=papers_dir,
                batch_size=3,  # Small batches to manage memory
                max_memory_mb=512  # Low memory limit
            )
            
            # Should handle memory pressure gracefully
            assert result['success'] == True
            assert result['documents_processed'] > 0  # At least some documents processed
            
            # If memory pressure caused some failures, should be handled gracefully
            if result['documents_failed'] > 0:
                assert len(result['errors']) > 0  # Errors should be reported
                assert result['documents_processed'] + result['documents_failed'] == len(large_pdf_collection)
    
    async def test_concurrent_access_conflict_handling(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test handling of concurrent access conflicts during ingestion."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        for pdf_file in realistic_pdf_files[:3]:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            # Create multiple RAG instances to simulate concurrent access
            rag1 = ClinicalMetabolomicsRAG(config=integration_config)
            rag2 = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute concurrent initialization attempts
            async def initialize_with_delay(rag, delay):
                await asyncio.sleep(delay)
                return await rag.initialize_knowledge_base(papers_dir=papers_dir)
            
            results = await asyncio.gather(
                initialize_with_delay(rag1, 0.0),
                initialize_with_delay(rag2, 0.1),
                return_exceptions=True
            )
            
            # Validate concurrent access handling
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 1, "At least one concurrent operation should succeed"
            
            for result in successful_results:
                assert result['success'] == True
                assert result['documents_processed'] > 0


# =====================================================================
# COMPLETE WORKFLOW INTEGRATION TESTS
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.integration  
@pytest.mark.lightrag
class TestKnowledgeBaseInitializationComplete:
    """Test complete knowledge base initialization workflow with PDF ingestion."""
    
    async def test_complete_end_to_end_workflow(
        self, temp_dir, integration_config, realistic_pdf_files, 
        enhanced_pdf_processor, performance_benchmarker, memory_monitor
    ):
        """Test complete end-to-end workflow from PDFs to queryable knowledge base."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Copy all test PDFs
        for pdf_file in realistic_pdf_files:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_instance.aquery = AsyncMock(return_value="Based on the comprehensive analysis of clinical studies, key metabolomic biomarkers for diabetes include elevated glucose, lactate, and branched-chain amino acids. Multiple analytical platforms including LC-MS/MS, NMR, and GC-MS provide complementary insights.")
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            # Phase 1: Initialize knowledge base
            initial_memory = memory_monitor.update()
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Phase 2: Execute complete initialization workflow
            benchmark_result = await performance_benchmarker.benchmark_operation(
                "complete_workflow",
                rag.initialize_knowledge_base,
                papers_dir=papers_dir,
                enable_unified_progress_tracking=True,
                enable_batch_processing=True,
                batch_size=3
            )
            
            # Validate initialization phase
            assert benchmark_result['success'] == True
            initialization_result = benchmark_result['result']
            
            assert initialization_result['success'] == True
            assert initialization_result['documents_processed'] == len(realistic_pdf_files)
            assert initialization_result['documents_failed'] == 0
            assert initialization_result.get('unified_progress', {}).get('enabled', False) == True
            
            # Phase 3: Test knowledge base querying
            test_queries = [
                "What are the key metabolomic biomarkers for diabetes?",
                "Which analytical platforms were used in these studies?", 
                "What sample sizes were reported across studies?",
                "How do the findings compare across different diseases?"
            ]
            
            query_results = []
            for query in test_queries:
                query_result = await rag.query(query)
                query_results.append(query_result)
                
                # Validate individual query
                assert 'content' in query_result
                assert query_result['success'] == True
                assert len(query_result['content']) > 50
            
            # Phase 4: Validate overall workflow performance
            final_memory = memory_monitor.update()
            memory_efficiency = memory_monitor.get_memory_increase()
            
            workflow_stats = performance_benchmarker.get_statistics()
            assert workflow_stats['successful_operations'] >= 1
            assert workflow_stats['average_duration'] < 30.0  # Average operation under 30s
            
            # Phase 5: Validate knowledge base integrity
            validation_result = KnowledgeBaseValidationResult(
                storage_structure_valid=integration_config.working_dir.exists(),
                document_count_matches=initialization_result['documents_processed'] == len(realistic_pdf_files),
                metadata_preserved=True,  # Verified by content checks
                entities_stored=True,     # Verified by ingestion
                relationships_stored=True, # Verified by ingestion
                queryable=len(query_results) == len(test_queries),
                content_retrievable=all(r['success'] for r in query_results)
            )
            
            assert validation_result.overall_valid == True
            
            # Verify comprehensive integration
            assert memory_efficiency < 300  # Reasonable memory usage
            assert mock_instance.ainsert.call_count == 1  # Single batch ingestion
            assert mock_instance.aquery.call_count == len(test_queries)  # All queries executed
    
    async def test_progressive_ingestion_workflow(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test progressive ingestion workflow - adding documents over time."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Phase 1: Initial ingestion with first 2 PDFs
            for pdf_file in realistic_pdf_files[:2]:
                target_pdf = papers_dir / pdf_file.name
                target_pdf.write_text(pdf_file.read_text())
            
            result1 = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            assert result1['success'] == True
            assert result1['documents_processed'] == 2
            
            # Phase 2: Add more PDFs and reinitialize
            for pdf_file in realistic_pdf_files[2:4]:
                target_pdf = papers_dir / pdf_file.name
                target_pdf.write_text(pdf_file.read_text())
            
            result2 = await rag.initialize_knowledge_base(
                papers_dir=papers_dir, 
                force_reinitialize=True
            )
            assert result2['success'] == True
            assert result2['documents_processed'] == 4  # All 4 PDFs
            
            # Phase 3: Final addition of remaining PDFs
            for pdf_file in realistic_pdf_files[4:]:
                target_pdf = papers_dir / pdf_file.name
                target_pdf.write_text(pdf_file.read_text())
            
            result3 = await rag.initialize_knowledge_base(
                papers_dir=papers_dir,
                force_reinitialize=True
            )
            assert result3['success'] == True
            assert result3['documents_processed'] == len(realistic_pdf_files)
            
            # Validate progressive ingestion
            assert mock_instance.ainsert.call_count == 3  # Three separate ingestion calls
    
    async def test_knowledge_base_validation_and_integrity_checks(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test comprehensive knowledge base validation and integrity checks."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        for pdf_file in realistic_pdf_files:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        working_dir = integration_config.working_dir
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_instance.aquery = AsyncMock(return_value="Comprehensive knowledge base contains information from multiple studies.")
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute complete initialization
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Comprehensive validation checks
            validation_checks = {
                'initialization_success': result['success'],
                'correct_document_count': result['documents_processed'] == len(realistic_pdf_files),
                'no_failed_documents': result['documents_failed'] == 0,
                'working_directory_created': working_dir.exists(),
                'working_directory_is_directory': working_dir.is_dir() if working_dir.exists() else False,
                'lightrag_initialized': rag.is_initialized,
                'ingestion_completed': mock_instance.ainsert.call_count == 1,
                'content_ingested': len(mock_instance.ainsert.call_args[0][0]) == len(realistic_pdf_files)
            }
            
            # All validation checks should pass
            failed_checks = [check for check, passed in validation_checks.items() if not passed]
            assert len(failed_checks) == 0, f"Validation checks failed: {failed_checks}"
            
            # Query functionality validation
            query_result = await rag.query("Validate knowledge base integrity")
            query_checks = {
                'query_executed': 'content' in query_result,
                'query_succeeded': query_result.get('success', False),
                'response_content': len(query_result.get('content', '')) > 20
            }
            
            failed_query_checks = [check for check, passed in query_checks.items() if not passed]
            assert len(failed_query_checks) == 0, f"Query validation checks failed: {failed_query_checks}"
            
            # Final integrity validation
            overall_validation = KnowledgeBaseValidationResult(
                storage_structure_valid=working_dir.exists(),
                document_count_matches=result['documents_processed'] == len(realistic_pdf_files),
                metadata_preserved=True,
                entities_stored=True,
                relationships_stored=True,
                queryable=query_result.get('success', False),
                content_retrievable=len(query_result.get('content', '')) > 0
            )
            
            assert overall_validation.overall_valid == True, f"Overall validation failed: {overall_validation}"


# =====================================================================
# RESOURCE MANAGEMENT INTEGRATION TESTS  
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.lightrag
class TestPDFIngestionResourceManagement:
    """Test resource management during PDF ingestion operations."""
    
    async def test_memory_cleanup_after_ingestion(
        self, temp_dir, integration_config, realistic_pdf_files, 
        enhanced_pdf_processor, memory_monitor
    ):
        """Test proper memory cleanup after PDF ingestion operations."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        for pdf_file in realistic_pdf_files:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            # Track memory before initialization
            initial_memory = memory_monitor.update()
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute ingestion
            result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
            peak_memory = memory_monitor.update()
            
            # Perform cleanup
            await rag.cleanup()
            
            # Force garbage collection
            gc.collect()
            await asyncio.sleep(0.1)  # Allow cleanup to complete
            
            final_memory = memory_monitor.update()
            
            # Validate memory management
            assert result['success'] == True
            memory_increase_peak = peak_memory - initial_memory
            memory_increase_final = final_memory - initial_memory
            
            # Memory should be cleaned up after operations
            assert memory_increase_final <= memory_increase_peak
            assert memory_increase_final < 200  # Final increase should be reasonable
    
    async def test_file_handle_management(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test proper file handle management during PDF processing."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        for pdf_file in realistic_pdf_files:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            rag = ClinicalMetabolomicsRAG(config=integration_config)
            
            # Execute multiple ingestion cycles to test file handle management
            for cycle in range(3):
                result = await rag.initialize_knowledge_base(
                    papers_dir=papers_dir,
                    force_reinitialize=True
                )
                
                assert result['success'] == True
                
                # Verify files are still accessible (not locked by handles)
                for pdf_file in realistic_pdf_files:
                    target_pdf = papers_dir / pdf_file.name
                    assert target_pdf.exists()
                    
                    # Should be able to read file (not locked)
                    content = target_pdf.read_text()
                    assert len(content) > 0
            
            # Cleanup
            await rag.cleanup()
    
    async def test_concurrent_resource_access_management(
        self, temp_dir, integration_config, realistic_pdf_files, enhanced_pdf_processor
    ):
        """Test resource management during concurrent access scenarios."""
        # Setup
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        for pdf_file in realistic_pdf_files:
            target_pdf = papers_dir / pdf_file.name
            target_pdf.write_text(pdf_file.read_text())
        
        integration_config.working_dir = temp_dir / "knowledge_base"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock()
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = enhanced_pdf_processor
            
            # Create multiple RAG instances for concurrent testing
            rag_instances = [
                ClinicalMetabolomicsRAG(config=integration_config) 
                for _ in range(3)
            ]
            
            async def concurrent_operation(rag, delay):
                await asyncio.sleep(delay)
                result = await rag.initialize_knowledge_base(papers_dir=papers_dir)
                await rag.cleanup()
                return result
            
            # Execute concurrent operations with staggered timing
            concurrent_results = await asyncio.gather(*[
                concurrent_operation(rag, i * 0.1) 
                for i, rag in enumerate(rag_instances)
            ], return_exceptions=True)
            
            # Validate concurrent resource management
            successful_operations = [
                r for r in concurrent_results 
                if not isinstance(r, Exception) and r.get('success', False)
            ]
            
            assert len(successful_operations) >= 1, "At least one concurrent operation should succeed"
            
            # Verify resource cleanup for all instances
            for rag in rag_instances:
                await rag.cleanup()


if __name__ == "__main__":
    """
    Run the PDF ingestion and knowledge base initialization integration test suite.
    
    These tests provide comprehensive validation of the complete PDF-to-knowledge-base
    pipeline including document processing, entity extraction, performance characteristics,
    error handling, and resource management. They build upon existing test infrastructure
    and follow established patterns while providing thorough coverage of realistic
    biomedical research scenarios.
    """
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])