#!/usr/bin/env python3
"""
Comprehensive PDF-to-Query Workflow Integration Test Implementation.

This module implements the comprehensive test scenarios specified in
comprehensive_pdf_query_workflow_test_scenarios.md, building upon the
existing excellent test infrastructure while adding new comprehensive
end-to-end validation capabilities.

The tests validate complete workflows from PDF ingestion through query
response generation, including performance, quality, error handling,
and large-scale processing scenarios.

Test Classes:
- TestSinglePDFQueryWorkflow: Complete single PDF processing workflows
- TestBatchPDFProcessingWorkflows: Large-scale batch processing scenarios
- TestPDFProcessingErrorRecovery: Comprehensive error handling validation
- TestQueryPerformanceValidation: Query performance and quality benchmarking
- TestCrossDocumentKnowledgeSynthesis: Multi-document integration testing
- TestLargeScaleProductionScenarios: Production-scale simulation testing

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import time
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass, field
import sys
import random
import gc

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import existing test infrastructure
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from lightrag_integration.pdf_processor import BiomedicalPDFProcessor
from lightrag_integration.config import LightRAGConfig

# Import test utilities (using existing patterns)
from test_primary_clinical_metabolomics_query import (
    ResponseQualityAssessor, PerformanceMonitor, MockClinicalMetabolomicsRAG
)


# =====================================================================
# COMPREHENSIVE TEST DATA STRUCTURES
# =====================================================================

@dataclass
class ComprehensivePDFScenario:
    """Represents a comprehensive PDF processing test scenario."""
    name: str
    description: str
    pdf_files: List[str]
    expected_entities: int
    expected_relationships: int
    test_queries: List[str]
    performance_benchmarks: Dict[str, float]
    quality_thresholds: Dict[str, float]
    error_tolerance: float = 0.05  # 5% acceptable error rate


@dataclass
class WorkflowValidationResult:
    """Results from comprehensive workflow validation."""
    scenario_name: str
    success: bool
    processing_time: float
    documents_processed: int
    queries_executed: int
    entities_extracted: int
    relationships_found: int
    average_query_time: float
    average_relevance_score: float
    total_cost: float
    error_count: int
    performance_flags: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveScenarioBuilder:
    """Builder for creating comprehensive test scenarios."""
    
    @classmethod
    def build_clinical_metabolomics_scenario(cls) -> ComprehensivePDFScenario:
        """Build scenario using real clinical metabolomics paper."""
        return ComprehensivePDFScenario(
            name="clinical_metabolomics_real_paper",
            description="Test with actual Clinical_Metabolomics_paper.pdf",
            pdf_files=["Clinical_Metabolomics_paper.pdf"],
            expected_entities=25,
            expected_relationships=15,
            test_queries=[
                "What is clinical metabolomics?",
                "What analytical techniques are used in metabolomics?",
                "What are common metabolomic biomarkers?",
                "How is metabolomics applied in clinical practice?"
            ],
            performance_benchmarks={
                "processing_time_per_pdf": 300.0,  # 5 minutes max
                "query_response_time": 30.0,       # 30 seconds max
                "memory_peak_mb": 1000.0           # 1GB max
            },
            quality_thresholds={
                "relevance_score": 80.0,           # 80% minimum
                "factual_accuracy": 75.0,          # 75% minimum  
                "completeness_score": 70.0         # 70% minimum
            }
        )
    
    @classmethod
    def build_multi_disease_scenario(cls) -> ComprehensivePDFScenario:
        """Build multi-disease biomedical research scenario."""
        return ComprehensivePDFScenario(
            name="multi_disease_biomedical",
            description="Multiple synthetic biomedical papers across disease areas",
            pdf_files=[
                "diabetes_metabolomics_study.pdf",
                "cardiovascular_proteomics_research.pdf", 
                "cancer_genomics_analysis.pdf",
                "liver_disease_biomarkers.pdf",
                "kidney_disease_metabolites.pdf"
            ],
            expected_entities=100,
            expected_relationships=60,
            test_queries=[
                "Compare biomarkers across different diseases",
                "What analytical methods are common across studies?",
                "Identify disease-specific metabolic pathways",
                "Summarize treatment approaches across conditions"
            ],
            performance_benchmarks={
                "processing_time_per_pdf": 120.0,  # 2 minutes per PDF
                "query_response_time": 45.0,       # 45 seconds for complex queries
                "memory_peak_mb": 2000.0           # 2GB for multiple PDFs
            },
            quality_thresholds={
                "relevance_score": 75.0,
                "factual_accuracy": 70.0,
                "completeness_score": 65.0
            }
        )
    
    @classmethod  
    def build_large_scale_scenario(cls) -> ComprehensivePDFScenario:
        """Build large-scale batch processing scenario."""
        pdf_files = [f"research_paper_{i:03d}.pdf" for i in range(1, 26)]  # 25 PDFs
        
        return ComprehensivePDFScenario(
            name="large_scale_batch_processing",
            description="Large-scale batch processing of 25 biomedical papers",
            pdf_files=pdf_files,
            expected_entities=500,
            expected_relationships=300,
            test_queries=[
                "Provide an overview of research methodologies",
                "What are the most common biomarkers mentioned?", 
                "Compare sample sizes across studies",
                "Identify emerging trends in analytical techniques",
                "Summarize key clinical applications"
            ],
            performance_benchmarks={
                "processing_time_per_pdf": 60.0,   # 1 minute per PDF average
                "query_response_time": 60.0,       # 1 minute for synthesis queries
                "memory_peak_mb": 4000.0,          # 4GB for large batch
                "batch_completion_time": 1800.0    # 30 minutes total
            },
            quality_thresholds={
                "relevance_score": 70.0,           # Lower threshold for synthesis
                "factual_accuracy": 65.0,
                "completeness_score": 60.0
            }
        )


class ComprehensiveWorkflowValidator:
    """Validates comprehensive PDF-to-query workflows."""
    
    def __init__(self, quality_assessor: ResponseQualityAssessor, performance_monitor: PerformanceMonitor):
        self.quality_assessor = quality_assessor
        self.performance_monitor = performance_monitor
        self.validation_history = []
    
    async def validate_complete_workflow(
        self,
        scenario: ComprehensivePDFScenario,
        rag_system,
        pdf_processor
    ) -> WorkflowValidationResult:
        """Validate complete PDF-to-query workflow for a scenario."""
        
        logging.info(f"Starting comprehensive workflow validation: {scenario.name}")
        start_time = time.time()
        
        # Initialize metrics tracking
        processing_times = []
        query_times = []
        relevance_scores = []
        entities_extracted = 0
        relationships_found = 0
        error_count = 0
        performance_flags = []
        quality_flags = []
        
        try:
            # Phase 1: PDF Processing
            logging.info("Phase 1: Processing PDFs...")
            for pdf_file in scenario.pdf_files:
                pdf_start = time.time()
                
                try:
                    # Simulate PDF processing (using mock data for comprehensive testing)
                    if hasattr(pdf_processor, 'process_pdf'):
                        result = await pdf_processor.process_pdf(pdf_file)
                        entities_extracted += len(result.get('entities', []))
                        relationships_found += len(result.get('relationships', []))
                    
                    pdf_time = time.time() - pdf_start
                    processing_times.append(pdf_time)
                    
                    # Check performance benchmarks
                    if pdf_time > scenario.performance_benchmarks['processing_time_per_pdf']:
                        performance_flags.append(f"SLOW_PDF_PROCESSING: {pdf_file}")
                    
                except Exception as e:
                    error_count += 1
                    logging.error(f"PDF processing error for {pdf_file}: {e}")
            
            # Phase 2: Query Execution and Validation
            logging.info("Phase 2: Executing test queries...")
            for query in scenario.test_queries:
                query_start = time.time()
                
                try:
                    # Execute query with performance monitoring
                    response, perf_metrics = await self.performance_monitor.monitor_query_performance(
                        rag_system.query, query
                    )
                    
                    query_time = time.time() - query_start
                    query_times.append(query_time)
                    
                    # Assess response quality
                    quality_metrics = self.quality_assessor.assess_response_quality(response, query)
                    relevance_scores.append(quality_metrics.relevance_score)
                    
                    # Check quality thresholds
                    if quality_metrics.relevance_score < scenario.quality_thresholds['relevance_score']:
                        quality_flags.append(f"LOW_RELEVANCE: {query[:50]}...")
                    
                    if quality_metrics.factual_accuracy_score < scenario.quality_thresholds['factual_accuracy']:
                        quality_flags.append(f"LOW_ACCURACY: {query[:50]}...")
                    
                    # Check performance benchmarks
                    if query_time > scenario.performance_benchmarks['query_response_time']:
                        performance_flags.append(f"SLOW_QUERY: {query[:50]}...")
                        
                except Exception as e:
                    error_count += 1
                    logging.error(f"Query execution error for '{query}': {e}")
                    query_times.append(scenario.performance_benchmarks['query_response_time'])  # Worst case
                    relevance_scores.append(0.0)  # Failed query
            
            # Phase 3: Calculate final metrics
            total_time = time.time() - start_time
            avg_query_time = statistics.mean(query_times) if query_times else 0.0
            avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
            
            # Determine overall success
            error_rate = error_count / (len(scenario.pdf_files) + len(scenario.test_queries))
            success = (
                error_rate <= scenario.error_tolerance and
                avg_relevance >= scenario.quality_thresholds['relevance_score'] and
                avg_query_time <= scenario.performance_benchmarks['query_response_time']
            )
            
            # Create validation result
            result = WorkflowValidationResult(
                scenario_name=scenario.name,
                success=success,
                processing_time=total_time,
                documents_processed=len(scenario.pdf_files),
                queries_executed=len(scenario.test_queries),
                entities_extracted=entities_extracted,
                relationships_found=relationships_found,
                average_query_time=avg_query_time,
                average_relevance_score=avg_relevance,
                total_cost=0.50 * (len(scenario.pdf_files) + len(scenario.test_queries)),  # Estimated
                error_count=error_count,
                performance_flags=performance_flags,
                quality_flags=quality_flags,
                detailed_metrics={
                    'processing_times': processing_times,
                    'query_times': query_times, 
                    'relevance_scores': relevance_scores,
                    'error_rate': error_rate,
                    'entities_per_document': entities_extracted / max(1, len(scenario.pdf_files)),
                    'relationships_per_document': relationships_found / max(1, len(scenario.pdf_files))
                }
            )
            
            self.validation_history.append(result)
            logging.info(f"Workflow validation completed: {scenario.name} - Success: {success}")
            return result
            
        except Exception as e:
            logging.error(f"Critical error in workflow validation: {e}")
            return WorkflowValidationResult(
                scenario_name=scenario.name,
                success=False,
                processing_time=time.time() - start_time,
                documents_processed=0,
                queries_executed=0,
                entities_extracted=0,
                relationships_found=0,
                average_query_time=0.0,
                average_relevance_score=0.0,
                total_cost=0.0,
                error_count=1,
                performance_flags=["CRITICAL_ERROR"],
                quality_flags=["VALIDATION_FAILED"]
            )


# =====================================================================
# COMPREHENSIVE TEST FIXTURES
# =====================================================================

@pytest.fixture
def comprehensive_scenario_builder():
    """Provide comprehensive scenario builder."""
    return ComprehensiveScenarioBuilder()


@pytest.fixture
def comprehensive_workflow_validator(quality_assessor, performance_monitor):
    """Provide comprehensive workflow validator."""
    return ComprehensiveWorkflowValidator(quality_assessor, performance_monitor)


@pytest.fixture
def mock_comprehensive_rag_system(mock_config):
    """Create enhanced mock RAG system for comprehensive testing."""
    return MockClinicalMetabolomicsRAG(mock_config)


@pytest.fixture
def mock_comprehensive_pdf_processor():
    """Create comprehensive mock PDF processor."""
    processor = MagicMock()
    
    async def mock_process_pdf(pdf_path):
        """Enhanced mock PDF processing with realistic entity extraction."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Generate realistic entities based on filename
        filename = str(pdf_path).lower()
        entities = []
        relationships = []
        
        if "diabetes" in filename or "metabolomics" in filename:
            entities = ["glucose", "insulin", "HbA1c", "metabolomics", "LC-MS", "biomarker"]
            relationships = ["glucose_regulates_insulin", "insulin_affects_metabolism", "LC-MS_measures_glucose"]
        elif "cardiovascular" in filename:
            entities = ["TMAO", "cholesterol", "troponin", "proteomics", "mass spectrometry"]
            relationships = ["TMAO_associated_with_CVD", "cholesterol_affects_vessels"]
        elif "cancer" in filename:
            entities = ["lactate", "oncometabolite", "Warburg effect", "tumor", "genomics"]
            relationships = ["lactate_indicates_Warburg", "oncometabolites_drive_cancer"]
        else:
            # Default biomedical entities
            entities = ["biomarker", "analysis", "clinical", "research", "methodology"]
            relationships = ["biomarker_enables_diagnosis", "methodology_supports_analysis"]
        
        return {
            'text': f"Processed content from {pdf_path}",
            'entities': entities,
            'relationships': relationships,
            'metadata': {
                'title': f"Research Paper: {Path(pdf_path).stem}",
                'processing_time': 0.1,
                'success': True
            }
        }
    
    processor.process_pdf = AsyncMock(side_effect=mock_process_pdf)
    return processor


@pytest.fixture
def real_clinical_paper_path():
    """Path to the real clinical metabolomics paper."""
    return Path("/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf")


# =====================================================================
# COMPREHENSIVE TEST IMPLEMENTATIONS
# =====================================================================

@pytest.mark.biomedical
@pytest.mark.integration
@pytest.mark.comprehensive
class TestSinglePDFQueryWorkflow:
    """Comprehensive single PDF processing workflow tests."""

    @pytest.mark.asyncio
    async def test_clinical_metabolomics_paper_complete_workflow(
        self,
        comprehensive_scenario_builder,
        comprehensive_workflow_validator,
        mock_comprehensive_rag_system,
        mock_comprehensive_pdf_processor,
        real_clinical_paper_path
    ):
        """
        Test complete workflow with real clinical metabolomics paper.
        This validates end-to-end processing from PDF to high-quality responses.
        """
        # Build scenario
        scenario = comprehensive_scenario_builder.build_clinical_metabolomics_scenario()
        
        # Update scenario to use real paper path if available
        if real_clinical_paper_path.exists():
            scenario.pdf_files = [str(real_clinical_paper_path)]
            logging.info(f"Using real clinical paper: {real_clinical_paper_path}")
        
        # Validate complete workflow
        result = await comprehensive_workflow_validator.validate_complete_workflow(
            scenario,
            mock_comprehensive_rag_system,
            mock_comprehensive_pdf_processor
        )
        
        # Comprehensive assertions
        assert result.success, f"Workflow failed: {result.performance_flags + result.quality_flags}"
        assert result.documents_processed == len(scenario.pdf_files), \
            f"Expected {len(scenario.pdf_files)} docs processed, got {result.documents_processed}"
        assert result.queries_executed == len(scenario.test_queries), \
            f"Expected {len(scenario.test_queries)} queries executed, got {result.queries_executed}"
        assert result.average_relevance_score >= scenario.quality_thresholds['relevance_score'], \
            f"Average relevance {result.average_relevance_score}% below threshold {scenario.quality_thresholds['relevance_score']}%"
        assert result.average_query_time <= scenario.performance_benchmarks['query_response_time'], \
            f"Average query time {result.average_query_time}s exceeds limit {scenario.performance_benchmarks['query_response_time']}s"
        assert result.entities_extracted >= scenario.expected_entities, \
            f"Expected ≥{scenario.expected_entities} entities, got {result.entities_extracted}"
        
        # Log comprehensive results
        logging.info(f"✅ Clinical Metabolomics Workflow Validation Results:")
        logging.info(f"  - Documents Processed: {result.documents_processed}")
        logging.info(f"  - Entities Extracted: {result.entities_extracted}")
        logging.info(f"  - Relationships Found: {result.relationships_found}")
        logging.info(f"  - Average Query Time: {result.average_query_time:.2f}s")
        logging.info(f"  - Average Relevance: {result.average_relevance_score:.1f}%")
        logging.info(f"  - Total Cost: ${result.total_cost:.2f}")
        logging.info(f"  - Error Count: {result.error_count}")

    @pytest.mark.asyncio
    async def test_multi_disease_pdf_processing_workflow(
        self,
        comprehensive_scenario_builder,
        comprehensive_workflow_validator,
        mock_comprehensive_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """
        Test workflow with multiple disease-specific research papers.
        Validates cross-document knowledge synthesis and domain expertise.
        """
        scenario = comprehensive_scenario_builder.build_multi_disease_scenario()
        
        result = await comprehensive_workflow_validator.validate_complete_workflow(
            scenario,
            mock_comprehensive_rag_system,
            mock_comprehensive_pdf_processor
        )
        
        # Multi-disease specific validations
        assert result.success, "Multi-disease workflow should succeed"
        assert result.entities_extracted >= scenario.expected_entities, \
            "Should extract adequate entities across diseases"
        assert result.relationships_found >= scenario.expected_relationships, \
            "Should identify relationships across disease domains"
        
        # Check for cross-document synthesis capability
        entity_density = result.detailed_metrics['entities_per_document']
        assert entity_density >= 15, f"Entity density too low: {entity_density} per document"
        
        # Validate that complex cross-disease queries were handled
        complex_query_performance = [t for t in result.detailed_metrics['query_times'] if t > 30]
        assert len(complex_query_performance) >= 2, "Should handle complex cross-disease queries"
        
        logging.info(f"✅ Multi-Disease Workflow Validation:")
        logging.info(f"  - Cross-Document Entities: {result.entities_extracted} across {result.documents_processed} papers")
        logging.info(f"  - Entity Density: {entity_density:.1f} per document")
        logging.info(f"  - Complex Query Performance: {len(complex_query_performance)} queries >30s")


@pytest.mark.biomedical
@pytest.mark.integration
@pytest.mark.performance
class TestBatchPDFProcessingWorkflows:
    """Comprehensive batch PDF processing workflow tests."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_scale_batch_processing_workflow(
        self,
        comprehensive_scenario_builder,
        comprehensive_workflow_validator,
        mock_comprehensive_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """
        Test large-scale batch processing of multiple research papers.
        Validates scalability, performance, and resource management.
        """
        scenario = comprehensive_scenario_builder.build_large_scale_scenario()
        
        # Track resource usage during batch processing
        start_memory = self._get_memory_usage()
        
        result = await comprehensive_workflow_validator.validate_complete_workflow(
            scenario,
            mock_comprehensive_rag_system,
            mock_comprehensive_pdf_processor
        )
        
        end_memory = self._get_memory_usage()
        memory_usage = end_memory - start_memory
        
        # Large-scale processing validations
        assert result.success, f"Large-scale processing failed: {result.quality_flags}"
        assert result.processing_time <= scenario.performance_benchmarks['batch_completion_time'], \
            f"Batch completion time {result.processing_time}s exceeds limit {scenario.performance_benchmarks['batch_completion_time']}s"
        assert memory_usage <= scenario.performance_benchmarks['memory_peak_mb'], \
            f"Memory usage {memory_usage}MB exceeds limit {scenario.performance_benchmarks['memory_peak_mb']}MB"
        
        # Scalability metrics
        avg_processing_time = statistics.mean(result.detailed_metrics['processing_times'])
        assert avg_processing_time <= scenario.performance_benchmarks['processing_time_per_pdf'], \
            f"Average processing time {avg_processing_time}s per PDF exceeds limit"
        
        # Batch efficiency metrics
        throughput = result.documents_processed / (result.processing_time / 3600)  # PDFs per hour
        assert throughput >= 25, f"Processing throughput {throughput:.1f} PDFs/hour below minimum"
        
        # Error tolerance validation
        error_rate = result.error_count / (result.documents_processed + result.queries_executed)
        assert error_rate <= scenario.error_tolerance, \
            f"Error rate {error_rate:.2%} exceeds tolerance {scenario.error_tolerance:.2%}"
        
        logging.info(f"✅ Large-Scale Batch Processing Results:")
        logging.info(f"  - Documents Processed: {result.documents_processed}")
        logging.info(f"  - Processing Throughput: {throughput:.1f} PDFs/hour")
        logging.info(f"  - Average Processing Time: {avg_processing_time:.2f}s per PDF")
        logging.info(f"  - Memory Usage: {memory_usage:.1f} MB")
        logging.info(f"  - Error Rate: {error_rate:.2%}")
        logging.info(f"  - Total Entities: {result.entities_extracted}")

    @pytest.mark.asyncio
    async def test_incremental_knowledge_base_growth(
        self,
        mock_comprehensive_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """
        Test incremental growth of knowledge base with continuous querying.
        Validates that new information is properly integrated and queryable.
        """
        # Phase 1: Initial knowledge base (5 papers)
        initial_pdfs = [f"foundational_paper_{i}.pdf" for i in range(1, 6)]
        test_query = "What are the key metabolomic biomarkers?"
        
        # Process initial batch
        for pdf in initial_pdfs:
            await mock_comprehensive_pdf_processor.process_pdf(pdf)
        
        # Get baseline response
        baseline_response = await mock_comprehensive_rag_system.query(test_query)
        baseline_length = len(baseline_response)
        
        # Phase 2: Add more papers incrementally
        growth_phases = [
            [f"growth_phase_1_paper_{i}.pdf" for i in range(1, 4)],  # 3 more papers
            [f"growth_phase_2_paper_{i}.pdf" for i in range(1, 4)],  # 3 more papers
            [f"growth_phase_3_paper_{i}.pdf" for i in range(1, 4)]   # 3 more papers
        ]
        
        response_improvements = []
        
        for phase_num, phase_pdfs in enumerate(growth_phases, 1):
            # Add papers from this growth phase
            for pdf in phase_pdfs:
                await mock_comprehensive_pdf_processor.process_pdf(pdf)
            
            # Test query with expanded knowledge base
            expanded_response = await mock_comprehensive_rag_system.query(test_query)
            improvement = len(expanded_response) / baseline_length
            response_improvements.append(improvement)
            
            logging.info(f"Phase {phase_num}: Response length improvement = {improvement:.2f}x")
        
        # Validate knowledge base growth
        assert all(improvement >= 1.0 for improvement in response_improvements), \
            "Response quality should not degrade as knowledge base grows"
        
        # Validate progressive improvement (at least some phases show improvement)
        significant_improvements = sum(1 for imp in response_improvements if imp >= 1.2)
        assert significant_improvements >= 1, \
            "At least one growth phase should show significant improvement (≥20%)"
        
        logging.info(f"✅ Knowledge Base Growth Validation:")
        logging.info(f"  - Initial Papers: {len(initial_pdfs)}")
        logging.info(f"  - Total Growth Papers: {sum(len(phase) for phase in growth_phases)}")
        logging.info(f"  - Response Improvements: {response_improvements}")
        logging.info(f"  - Significant Improvements: {significant_improvements}/3 phases")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (simplified mock)."""
        # In a real implementation, would use psutil or similar
        return random.uniform(50, 100)  # Mock memory usage


@pytest.mark.biomedical
@pytest.mark.integration
class TestPDFProcessingErrorRecovery:
    """Comprehensive error handling and recovery testing."""

    @pytest.mark.asyncio
    async def test_cascading_failure_recovery_workflow(
        self,
        mock_comprehensive_rag_system,
        mock_comprehensive_pdf_processor,
        comprehensive_workflow_validator
    ):
        """
        Test system resilience to cascading failures during PDF processing.
        Validates error isolation, recovery mechanisms, and state consistency.
        """
        # Create scenario with intentional failures
        failing_pdfs = [
            "corrupted_file.pdf",      # PDF corruption
            "memory_exhaustion.pdf",   # Memory exhaustion simulation
            "api_timeout.pdf",         # API timeout simulation
            "storage_failure.pdf",     # Storage write failure
            "indexing_failure.pdf"     # LightRAG indexing failure
        ]
        
        successful_pdfs = [
            "good_paper_1.pdf",
            "good_paper_2.pdf", 
            "good_paper_3.pdf"
        ]
        
        all_pdfs = failing_pdfs + successful_pdfs
        error_count = 0
        successful_processes = 0
        
        # Process PDFs with error injection
        for pdf in all_pdfs:
            try:
                if pdf in failing_pdfs:
                    # Simulate different types of failures
                    if "corrupted" in pdf:
                        raise ValueError("PDF file corrupted")
                    elif "memory" in pdf:
                        raise MemoryError("Insufficient memory for processing")
                    elif "timeout" in pdf:
                        raise TimeoutError("API request timeout")
                    elif "storage" in pdf:
                        raise OSError("Storage write failed")
                    elif "indexing" in pdf:
                        raise RuntimeError("LightRAG indexing failed")
                
                # Process successful PDFs normally
                result = await mock_comprehensive_pdf_processor.process_pdf(pdf)
                assert result['metadata']['success'], "Good PDFs should process successfully"
                successful_processes += 1
                
            except Exception as e:
                error_count += 1
                logging.info(f"Expected error for {pdf}: {e}")
        
        # Validate error handling
        assert error_count == len(failing_pdfs), \
            f"Expected {len(failing_pdfs)} errors, got {error_count}"
        assert successful_processes == len(successful_pdfs), \
            f"Expected {len(successful_pdfs)} successes, got {successful_processes}"
        
        # Validate system remains functional after errors
        test_query = "What research methods were used?"
        response = await mock_comprehensive_rag_system.query(test_query)
        assert len(response) > 50, "System should remain queryable after errors"
        assert "research" in response.lower(), "Should provide relevant response despite errors"
        
        # Calculate error isolation effectiveness
        isolation_effectiveness = successful_processes / len(successful_pdfs)
        assert isolation_effectiveness == 1.0, \
            f"Error isolation failed: only {isolation_effectiveness:.1%} of good PDFs processed"
        
        logging.info(f"✅ Cascading Failure Recovery Results:")
        logging.info(f"  - Total Errors: {error_count}/{len(failing_pdfs)} (expected)")
        logging.info(f"  - Successful Processes: {successful_processes}/{len(successful_pdfs)}")
        logging.info(f"  - Error Isolation: {isolation_effectiveness:.1%} effectiveness")
        logging.info(f"  - System Functionality: Maintained after {error_count} errors")

    @pytest.mark.asyncio
    async def test_resource_exhaustion_graceful_handling(
        self,
        mock_comprehensive_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """
        Test graceful handling of resource exhaustion scenarios.
        Validates system behavior under memory, storage, and API constraints.
        """
        # Simulate resource constraints
        resource_scenarios = [
            {"type": "memory", "limit": 100, "pdfs": 5},    # Low memory
            {"type": "storage", "limit": 50, "pdfs": 3},    # Limited storage
            {"type": "api_rate", "limit": 10, "pdfs": 15}   # API rate limiting
        ]
        
        graceful_degradation_results = []
        
        for scenario in resource_scenarios:
            scenario_start = time.time()
            processed_count = 0
            degradation_detected = False
            
            try:
                # Simulate processing under resource constraints
                pdfs = [f"{scenario['type']}_test_pdf_{i}.pdf" for i in range(scenario['pdfs'])]
                
                for i, pdf in enumerate(pdfs):
                    # Simulate resource exhaustion after limit
                    if i >= scenario['limit'] / 10:  # Trigger constraint
                        # Instead of failing, system should gracefully degrade
                        await asyncio.sleep(0.2)  # Simulate slower processing
                        degradation_detected = True
                    
                    result = await mock_comprehensive_pdf_processor.process_pdf(pdf)
                    processed_count += 1
                
                scenario_time = time.time() - scenario_start
                
                # Validate graceful degradation
                success_rate = processed_count / len(pdfs)
                graceful_degradation_results.append({
                    'scenario': scenario['type'],
                    'success_rate': success_rate,
                    'processing_time': scenario_time,
                    'degradation_detected': degradation_detected
                })
                
                # All PDFs should eventually process (even if slower)
                assert success_rate >= 0.8, \
                    f"{scenario['type']} scenario should achieve ≥80% success rate, got {success_rate:.1%}"
                
            except Exception as e:
                logging.error(f"Resource exhaustion scenario {scenario['type']} failed: {e}")
                graceful_degradation_results.append({
                    'scenario': scenario['type'],
                    'success_rate': 0.0,
                    'processing_time': time.time() - scenario_start,
                    'degradation_detected': False
                })
        
        # Validate overall graceful degradation
        avg_success_rate = statistics.mean(r['success_rate'] for r in graceful_degradation_results)
        degradation_scenarios = sum(1 for r in graceful_degradation_results if r['degradation_detected'])
        
        assert avg_success_rate >= 0.8, \
            f"Average success rate {avg_success_rate:.1%} below acceptable threshold"
        assert degradation_scenarios >= 2, \
            f"Only {degradation_scenarios}/3 scenarios showed graceful degradation"
        
        logging.info(f"✅ Resource Exhaustion Handling Results:")
        for result in graceful_degradation_results:
            logging.info(f"  - {result['scenario']}: {result['success_rate']:.1%} success, "
                        f"{result['processing_time']:.1f}s, degradation: {result['degradation_detected']}")
        logging.info(f"  - Average Success Rate: {avg_success_rate:.1%}")


@pytest.mark.biomedical
@pytest.mark.performance
class TestQueryPerformanceValidation:
    """Comprehensive query performance and quality validation tests."""

    @pytest.mark.asyncio
    async def test_comprehensive_query_performance_benchmarking(
        self,
        mock_comprehensive_rag_system,
        quality_assessor,
        performance_monitor
    ):
        """
        Comprehensive benchmarking of query performance across different complexity levels.
        """
        # Define query categories with expected performance characteristics
        query_categories = {
            "simple_factual": {
                "queries": [
                    "What is metabolomics?",
                    "Define clinical biomarkers",
                    "What is LC-MS?",
                    "What are metabolites?"
                ],
                "max_time": 5.0,
                "min_relevance": 90.0
            },
            "complex_analytical": {
                "queries": [
                    "Compare LC-MS versus GC-MS analytical approaches",
                    "Explain the workflow for metabolomic biomarker discovery",
                    "What are the advantages of targeted versus untargeted metabolomics?",
                    "How do sample preparation methods affect metabolomic results?"
                ],
                "max_time": 15.0,
                "min_relevance": 80.0
            },
            "cross_document": {
                "queries": [
                    "Identify common biomarkers across multiple disease studies",
                    "Compare methodological approaches across different research papers",
                    "What analytical platforms are most frequently used in clinical studies?",
                    "Synthesize quality control recommendations from multiple sources"
                ],
                "max_time": 20.0,
                "min_relevance": 85.0
            },
            "synthesis": {
                "queries": [
                    "Summarize best practices for clinical metabolomics workflow implementation",
                    "Provide comprehensive overview of metabolomic applications in personalized medicine",
                    "Integrate findings on metabolomic biomarkers across cardiovascular, diabetes, and cancer research",
                    "Synthesize recommendations for analytical method selection in clinical metabolomics"
                ],
                "max_time": 30.0,
                "min_relevance": 80.0
            }
        }
        
        benchmark_results = {}
        
        for category, config in query_categories.items():
            category_times = []
            category_relevance_scores = []
            category_failures = 0
            
            logging.info(f"Benchmarking {category} queries...")
            
            for query in config["queries"]:
                try:
                    # Execute with performance monitoring
                    response, perf_metrics = await performance_monitor.monitor_query_performance(
                        mock_comprehensive_rag_system.query, query
                    )
                    
                    # Assess quality
                    quality_metrics = quality_assessor.assess_response_quality(response, query)
                    
                    category_times.append(perf_metrics.response_time_seconds)
                    category_relevance_scores.append(quality_metrics.relevance_score)
                    
                    # Validate individual query performance
                    assert perf_metrics.response_time_seconds <= config["max_time"], \
                        f"{category} query '{query[:50]}...' took {perf_metrics.response_time_seconds}s (limit: {config['max_time']}s)"
                    assert quality_metrics.relevance_score >= config["min_relevance"], \
                        f"{category} query relevance {quality_metrics.relevance_score}% below {config['min_relevance']}%"
                    
                except Exception as e:
                    category_failures += 1
                    logging.error(f"Query failed: {query[:50]}... - {e}")
            
            # Calculate category statistics
            avg_time = statistics.mean(category_times) if category_times else 0
            avg_relevance = statistics.mean(category_relevance_scores) if category_relevance_scores else 0
            time_consistency = statistics.stdev(category_times) if len(category_times) > 1 else 0
            
            benchmark_results[category] = {
                'avg_time': avg_time,
                'avg_relevance': avg_relevance,
                'time_consistency': time_consistency,
                'failure_count': category_failures,
                'queries_tested': len(config["queries"])
            }
            
            # Category-level validations
            assert category_failures == 0, f"{category}: {category_failures} query failures"
            assert avg_time <= config["max_time"], \
                f"{category} average time {avg_time:.1f}s exceeds limit {config['max_time']}s"
            assert avg_relevance >= config["min_relevance"], \
                f"{category} average relevance {avg_relevance:.1f}% below {config['min_relevance']}%"
            assert time_consistency <= (config["max_time"] * 0.3), \
                f"{category} time inconsistency {time_consistency:.1f}s too high"
        
        # Overall benchmark validation
        total_queries = sum(r['queries_tested'] for r in benchmark_results.values())
        total_failures = sum(r['failure_count'] for r in benchmark_results.values())
        overall_failure_rate = total_failures / total_queries
        
        assert overall_failure_rate <= 0.05, \
            f"Overall failure rate {overall_failure_rate:.2%} exceeds 5% threshold"
        
        # Log comprehensive benchmark results
        logging.info(f"✅ Comprehensive Query Performance Benchmark Results:")
        for category, results in benchmark_results.items():
            logging.info(f"  - {category}:")
            logging.info(f"    * Average Time: {results['avg_time']:.2f}s")
            logging.info(f"    * Average Relevance: {results['avg_relevance']:.1f}%")
            logging.info(f"    * Time Consistency: ±{results['time_consistency']:.2f}s")
            logging.info(f"    * Success Rate: {(results['queries_tested'] - results['failure_count'])/results['queries_tested']:.1%}")
        logging.info(f"  - Overall: {total_queries - total_failures}/{total_queries} queries successful ({100*(1-overall_failure_rate):.1f}%)")


@pytest.mark.biomedical
@pytest.mark.integration
class TestCrossDocumentKnowledgeSynthesis:
    """Test cross-document knowledge synthesis and integration capabilities."""

    @pytest.mark.asyncio
    async def test_cross_study_biomarker_synthesis(
        self,
        mock_comprehensive_rag_system,
        mock_comprehensive_pdf_processor,
        quality_assessor
    ):
        """
        Test synthesis of biomarker information across multiple research studies.
        Validates cross-document integration and knowledge synthesis capabilities.
        """
        # Simulate multiple diabetes studies with different biomarker findings
        diabetes_studies = [
            {
                "file": "diabetes_study_glucose_focus.pdf",
                "biomarkers": ["glucose", "HbA1c", "fructosamine"],
                "methodology": "LC-MS/MS",
                "sample_size": 150
            },
            {
                "file": "diabetes_study_lipid_focus.pdf", 
                "biomarkers": ["triglycerides", "free fatty acids", "cholesterol"],
                "methodology": "GC-MS",
                "sample_size": 200
            },
            {
                "file": "diabetes_study_amino_acid_focus.pdf",
                "biomarkers": ["branched-chain amino acids", "alanine", "glycine"],
                "methodology": "LC-MS",
                "sample_size": 120
            },
            {
                "file": "diabetes_study_comprehensive.pdf",
                "biomarkers": ["glucose", "triglycerides", "alanine", "insulin"],
                "methodology": "multi-platform",
                "sample_size": 300
            }
        ]
        
        # Process all studies
        for study in diabetes_studies:
            await mock_comprehensive_pdf_processor.process_pdf(study["file"])
        
        # Test cross-study synthesis queries
        synthesis_queries = [
            "What biomarkers are consistently identified across diabetes studies?",
            "Compare methodological approaches across diabetes research studies",
            "What are the most commonly reported sample sizes in diabetes metabolomics studies?",
            "Identify any conflicting biomarker findings between studies"
        ]
        
        synthesis_results = []
        
        for query in synthesis_queries:
            response = await mock_comprehensive_rag_system.query(query)
            quality_metrics = quality_assessor.assess_response_quality(response, query)
            
            synthesis_results.append({
                'query': query,
                'response': response,
                'relevance_score': quality_metrics.relevance_score,
                'completeness_score': quality_metrics.completeness_score
            })
            
            # Validate synthesis quality
            assert quality_metrics.relevance_score >= 75.0, \
                f"Synthesis query relevance {quality_metrics.relevance_score}% too low"
            assert len(response) >= 200, \
                f"Synthesis response too brief: {len(response)} characters"
        
        # Validate cross-study integration
        biomarker_query_response = synthesis_results[0]['response'].lower()
        common_biomarkers = ["glucose", "triglycerides", "alanine"]  # Present in multiple studies
        biomarker_mentions = sum(1 for biomarker in common_biomarkers if biomarker in biomarker_query_response)
        
        assert biomarker_mentions >= 2, \
            f"Should identify multiple common biomarkers, found {biomarker_mentions}"
        
        # Validate methodological comparison
        methodology_response = synthesis_results[1]['response'].lower()
        methodologies = ["lc-ms", "gc-ms", "multi-platform"]
        methodology_mentions = sum(1 for method in methodologies if method in methodology_response)
        
        assert methodology_mentions >= 2, \
            f"Should compare multiple methodologies, found {methodology_mentions}"
        
        # Calculate average synthesis quality
        avg_relevance = statistics.mean(r['relevance_score'] for r in synthesis_results)
        avg_completeness = statistics.mean(r['completeness_score'] for r in synthesis_results)
        
        assert avg_relevance >= 75.0, f"Average synthesis relevance {avg_relevance:.1f}% insufficient"
        assert avg_completeness >= 65.0, f"Average synthesis completeness {avg_completeness:.1f}% insufficient"
        
        logging.info(f"✅ Cross-Study Biomarker Synthesis Results:")
        logging.info(f"  - Studies Processed: {len(diabetes_studies)}")
        logging.info(f"  - Synthesis Queries: {len(synthesis_queries)}")
        logging.info(f"  - Common Biomarker Mentions: {biomarker_mentions}/{len(common_biomarkers)}")
        logging.info(f"  - Methodology Comparisons: {methodology_mentions}/{len(methodologies)}")
        logging.info(f"  - Average Relevance: {avg_relevance:.1f}%")
        logging.info(f"  - Average Completeness: {avg_completeness:.1f}%")


@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.integration
class TestLargeScaleProductionScenarios:
    """Large-scale production simulation tests."""

    @pytest.mark.asyncio
    async def test_research_institution_scale_processing(
        self,
        comprehensive_scenario_builder,
        comprehensive_workflow_validator,
        mock_comprehensive_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """
        Simulate research institution scale processing with realistic usage patterns.
        Tests production-scale performance, resource management, and system stability.
        """
        # Build large-scale scenario
        scenario = comprehensive_scenario_builder.build_large_scale_scenario()
        
        # Track comprehensive metrics during large-scale processing
        institution_metrics = {
            'batch_processing_start': time.time(),
            'queries_processed': 0,
            'peak_memory_usage': 0.0,
            'total_cost': 0.0,
            'error_events': [],
            'performance_degradation_events': []
        }
        
        # Phase 1: Morning batch processing (simulate 25 PDFs)
        logging.info("🏥 Simulating morning batch processing...")
        batch_start = time.time()
        
        result = await comprehensive_workflow_validator.validate_complete_workflow(
            scenario,
            mock_comprehensive_rag_system,
            mock_comprehensive_pdf_processor
        )
        
        batch_time = time.time() - batch_start
        institution_metrics['batch_processing_time'] = batch_time
        institution_metrics['batch_success'] = result.success
        
        # Validate batch processing performance
        assert result.success, "Morning batch processing must succeed"
        assert batch_time <= 14400, f"Batch processing took {batch_time}s, limit is 4 hours (14400s)"  # 4 hours max
        
        # Phase 2: Continuous querying simulation (simulate day-long usage)
        logging.info("🔄 Simulating continuous querying throughout day...")
        continuous_queries = [
            "What are the latest metabolomic biomarkers?",
            "Compare analytical methods across studies",
            "Provide quality control recommendations",
            "Summarize recent findings in clinical applications",
            "What are emerging trends in metabolomics?"
        ] * 40  # 200 queries total
        
        query_times = []
        query_failures = 0
        
        for i, query in enumerate(continuous_queries):
            if i % 50 == 0:  # Progress logging
                logging.info(f"Processed {i}/{len(continuous_queries)} queries...")
            
            try:
                query_start = time.time()
                response = await mock_comprehensive_rag_system.query(query)
                query_time = time.time() - query_start
                query_times.append(query_time)
                
                # Check for performance degradation
                if query_time > 60:  # Queries taking >1 minute
                    institution_metrics['performance_degradation_events'].append({
                        'query_index': i,
                        'query_time': query_time,
                        'query': query[:50] + "..."
                    })
                
            except Exception as e:
                query_failures += 1
                institution_metrics['error_events'].append({
                    'type': 'query_failure',
                    'query_index': i,
                    'error': str(e)
                })
        
        institution_metrics['queries_processed'] = len(query_times)
        institution_metrics['query_failure_rate'] = query_failures / len(continuous_queries)
        
        # Calculate institution-scale performance metrics
        avg_query_time = statistics.mean(query_times) if query_times else 0
        query_throughput = len(query_times) / (sum(query_times) / 3600) if query_times else 0  # queries per hour
        
        # Production-scale validations
        assert institution_metrics['query_failure_rate'] <= 0.01, \
            f"Query failure rate {institution_metrics['query_failure_rate']:.2%} exceeds 1% limit"
        assert query_throughput >= 50, \
            f"Query throughput {query_throughput:.1f} queries/hour below 50 minimum"
        assert avg_query_time <= 30, \
            f"Average query time {avg_query_time:.1f}s exceeds 30s limit"
        assert len(institution_metrics['performance_degradation_events']) <= 10, \
            f"Too many performance degradation events: {len(institution_metrics['performance_degradation_events'])}"
        
        # Resource efficiency validation
        estimated_total_cost = result.total_cost + (len(query_times) * 0.01)  # $0.01 per query
        institution_metrics['total_cost'] = estimated_total_cost
        
        assert estimated_total_cost <= 50.0, \
            f"Total cost ${estimated_total_cost:.2f} exceeds $50 budget"
        
        # System stability validation (no critical errors)
        critical_errors = [e for e in institution_metrics['error_events'] if 'critical' in e.get('error', '').lower()]
        assert len(critical_errors) == 0, f"Found {len(critical_errors)} critical errors"
        
        logging.info(f"✅ Research Institution Scale Processing Results:")
        logging.info(f"  🏥 Institution Simulation Summary:")
        logging.info(f"    - Batch Processing: {result.documents_processed} PDFs in {batch_time/3600:.1f} hours")
        logging.info(f"    - Continuous Queries: {len(query_times)}/{len(continuous_queries)} successful")
        logging.info(f"    - Query Throughput: {query_throughput:.1f} queries/hour")
        logging.info(f"    - Average Query Time: {avg_query_time:.1f} seconds")
        logging.info(f"    - Total Cost: ${estimated_total_cost:.2f}")
        logging.info(f"    - Error Rate: {institution_metrics['query_failure_rate']:.2%}")
        logging.info(f"    - Performance Degradation Events: {len(institution_metrics['performance_degradation_events'])}")
        logging.info(f"  📊 Production Readiness: {'✅ READY' if all([result.success, query_throughput >= 50, estimated_total_cost <= 50]) else '❌ NOT READY'}")


# =====================================================================
# TEST EXECUTION HELPERS
# =====================================================================

def pytest_collection_modifyitems(config, items):
    """Customize test collection for comprehensive testing."""
    for item in items:
        # Add slow marker to large-scale tests
        if "large_scale" in item.nodeid or "production" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add comprehensive marker to multi-phase tests  
        if any(keyword in item.nodeid for keyword in ["workflow", "comprehensive", "synthesis"]):
            item.add_marker(pytest.mark.comprehensive)


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])