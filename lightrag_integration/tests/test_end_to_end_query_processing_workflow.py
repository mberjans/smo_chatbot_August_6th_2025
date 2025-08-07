#!/usr/bin/env python3
"""
Comprehensive End-to-End Query Processing Workflow Test Suite.

This module implements comprehensive tests for the complete end-to-end query processing
workflow, building upon the existing test infrastructure and PDF ingestion tests.
It validates the full pipeline from PDF ingestion through knowledge base construction
to query processing and response generation for biomedical research scenarios.

Test Coverage:
- Complete PDF-to-query response pipeline validation
- Multiple query types: simple factual, complex analytical, cross-document synthesis
- Query response validation for biomedical accuracy and relevance
- Different query modes: hybrid, local, global, naive
- Performance benchmarking for query response times
- Cross-document knowledge synthesis capabilities
- Context retrieval functionality validation
- Edge cases and error scenarios
- Resource management and cleanup

Test Classes:
- TestEndToEndQueryWorkflow: Core end-to-end workflow tests
- TestQueryTypeValidation: Validation of different query types and complexity levels
- TestQueryModeComparison: Testing and comparison of different query modes
- TestCrossDocumentSynthesis: Cross-document knowledge synthesis validation
- TestQueryPerformanceValidation: Performance benchmarking and optimization
- TestContextRetrievalValidation: Context retrieval accuracy and relevance
- TestErrorScenarioHandling: Edge cases and error handling during querying
- TestBiomedicalAccuracyValidation: Domain-specific accuracy and relevance validation

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
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass, field
import random
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import core components
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from lightrag_integration.pdf_processor import BiomedicalPDFProcessor
from lightrag_integration.config import LightRAGConfig

# Import test utilities from existing infrastructure
from test_primary_clinical_metabolomics_query import (
    ResponseQualityAssessor, PerformanceMonitor, MockClinicalMetabolomicsRAG,
    ResponseQualityMetrics, PerformanceMetrics, FactualAccuracyAssessment
)

# Import comprehensive test fixtures
from test_comprehensive_pdf_query_workflow import (
    ComprehensivePDFScenario, WorkflowValidationResult, 
    ComprehensiveScenarioBuilder, ComprehensiveWorkflowValidator
)


# =====================================================================
# QUERY PROCESSING TEST DATA MODELS
# =====================================================================

@dataclass
class QueryTestScenario:
    """Represents a comprehensive query test scenario."""
    scenario_id: str
    name: str
    description: str
    pdf_collection: List[str]  # PDF files to ingest
    query_sets: Dict[str, List[str]]  # Query type -> list of queries
    expected_performance: Dict[str, float]  # Performance benchmarks
    expected_quality: Dict[str, float]  # Quality thresholds
    query_modes_to_test: List[str] = field(default_factory=lambda: ['hybrid', 'local', 'global'])
    biomedical_validation_required: bool = True
    cross_document_synthesis_expected: bool = True


@dataclass
class QueryExecutionResult:
    """Results from query execution with comprehensive metrics."""
    query: str
    query_type: str
    query_mode: str
    response_content: str
    execution_time: float
    cost_usd: float
    quality_metrics: ResponseQualityMetrics
    context_retrieved: List[str] = field(default_factory=list)
    sources_referenced: List[str] = field(default_factory=list)
    biomedical_entities_found: List[str] = field(default_factory=list)
    synthesis_indicators: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EndToEndWorkflowResult:
    """Comprehensive results from end-to-end workflow execution."""
    scenario_id: str
    ingestion_success: bool
    documents_processed: int
    queries_executed: int
    query_results: List[QueryExecutionResult]
    total_workflow_time: float
    total_cost: float
    average_query_time: float
    average_quality_score: float
    cross_document_synthesis_success: bool
    performance_flags: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    biomedical_accuracy_flags: List[str] = field(default_factory=list)


# =====================================================================
# ENHANCED TEST SCENARIOS AND BUILDERS
# =====================================================================

class EndToEndQueryScenarioBuilder:
    """Builds comprehensive end-to-end query processing scenarios."""
    
    @classmethod
    def build_clinical_metabolomics_comprehensive_scenario(cls) -> QueryTestScenario:
        """Build comprehensive clinical metabolomics query scenario."""
        return QueryTestScenario(
            scenario_id="E2E_CLINICAL_METABOLOMICS_001",
            name="comprehensive_clinical_metabolomics_query_validation",
            description="Complete workflow validation for clinical metabolomics queries",
            pdf_collection=[
                "Clinical_Metabolomics_paper.pdf",
                "diabetes_metabolomics_study.pdf", 
                "cardiovascular_proteomics_research.pdf"
            ],
            query_sets={
                "simple_factual": [
                    "What is clinical metabolomics?",
                    "What is LC-MS?",
                    "Define metabolic biomarkers",
                    "What are metabolites?"
                ],
                "complex_analytical": [
                    "Compare LC-MS versus GC-MS analytical approaches for metabolomics",
                    "Explain the complete workflow for metabolomic biomarker discovery",
                    "What are the advantages of targeted versus untargeted metabolomics?",
                    "How do sample preparation methods affect metabolomic results?"
                ],
                "cross_document_synthesis": [
                    "Compare biomarkers identified across different disease studies",
                    "What methodological approaches are shared across research papers?",
                    "Synthesize quality control recommendations from multiple sources",
                    "Identify common analytical platforms used across studies"
                ],
                "domain_specific": [
                    "What metabolic pathways are altered in diabetes based on the literature?",
                    "How does sample collection method impact metabolomic analysis results?",
                    "What statistical methods are most appropriate for metabolomics data?",
                    "What are the clinical applications of metabolomics in personalized medicine?"
                ]
            },
            expected_performance={
                "simple_factual_max_time": 10.0,      # 10 seconds max
                "complex_analytical_max_time": 25.0,   # 25 seconds max
                "synthesis_max_time": 35.0,            # 35 seconds max
                "domain_specific_max_time": 20.0       # 20 seconds max
            },
            expected_quality={
                "simple_factual_min_relevance": 85.0,
                "complex_analytical_min_relevance": 80.0,
                "synthesis_min_relevance": 75.0,
                "domain_specific_min_relevance": 82.0,
                "overall_min_accuracy": 75.0
            }
        )
    
    @classmethod
    def build_multi_disease_biomarker_scenario(cls) -> QueryTestScenario:
        """Build multi-disease biomarker discovery scenario."""
        return QueryTestScenario(
            scenario_id="E2E_MULTI_DISEASE_002", 
            name="multi_disease_biomarker_synthesis",
            description="Cross-disease biomarker synthesis and comparison",
            pdf_collection=[
                "diabetes_metabolomics_study.pdf",
                "cardiovascular_proteomics_research.pdf",
                "cancer_genomics_analysis.pdf",
                "liver_disease_biomarkers.pdf",
                "kidney_disease_metabolites.pdf"
            ],
            query_sets={
                "cross_disease_comparison": [
                    "Compare biomarkers between diabetes and cardiovascular disease",
                    "What biomarkers are common across multiple diseases?",
                    "How do biomarker profiles differ between cancer and metabolic diseases?",
                    "Identify disease-specific versus shared metabolic alterations"
                ],
                "methodology_synthesis": [
                    "Compare analytical methods used across different disease studies",
                    "What sample preparation approaches are used in multi-disease research?",
                    "Synthesize statistical analysis approaches across studies",
                    "Compare sample sizes and study designs across disease areas"
                ],
                "clinical_translation": [
                    "How can these biomarkers be translated to clinical practice?",
                    "What are the diagnostic accuracies of biomarkers across diseases?",
                    "Which biomarkers show the most clinical promise?",
                    "How do biomarker validation approaches differ across diseases?"
                ]
            },
            expected_performance={
                "cross_disease_comparison_max_time": 30.0,
                "methodology_synthesis_max_time": 35.0,
                "clinical_translation_max_time": 25.0
            },
            expected_quality={
                "cross_disease_comparison_min_relevance": 78.0,
                "methodology_synthesis_min_relevance": 75.0,
                "clinical_translation_min_relevance": 80.0
            }
        )
    
    @classmethod
    def build_performance_stress_scenario(cls) -> QueryTestScenario:
        """Build performance stress testing scenario."""
        return QueryTestScenario(
            scenario_id="E2E_PERFORMANCE_003",
            name="performance_stress_testing",
            description="Performance stress testing with complex queries and large document sets",
            pdf_collection=[f"research_paper_{i:03d}.pdf" for i in range(1, 16)],  # 15 papers
            query_sets={
                "rapid_fire": [f"What is mentioned about biomarker {i}?" for i in range(1, 21)],  # 20 quick queries
                "complex_synthesis": [
                    "Provide a comprehensive overview of all research methodologies mentioned",
                    "Synthesize all biomarkers mentioned across all studies",
                    "Compare and contrast all analytical platforms discussed",
                    "Generate a complete summary of clinical applications described"
                ],
                "edge_case": [
                    "What information is available about quantum metabolomics?",  # Should handle gracefully
                    "Compare studies from 1990 to 2000",  # No matching timeframe
                    "What are the conclusions about artificial intelligence in metabolomics?",  # Modern topic
                    ""  # Empty query test
                ]
            },
            expected_performance={
                "rapid_fire_max_time": 8.0,      # Quick responses
                "complex_synthesis_max_time": 45.0,  # More complex processing
                "edge_case_max_time": 15.0        # Should handle gracefully
            },
            expected_quality={
                "rapid_fire_min_relevance": 70.0,
                "complex_synthesis_min_relevance": 75.0,
                "edge_case_min_relevance": 50.0  # Lower expectation for edge cases
            },
            cross_document_synthesis_expected=True
        )


class EnhancedWorkflowValidator:
    """Enhanced validator for end-to-end query processing workflows."""
    
    def __init__(self):
        self.quality_assessor = ResponseQualityAssessor()
        self.performance_monitor = PerformanceMonitor()
        self.validation_history = []
    
    async def validate_end_to_end_workflow(
        self,
        scenario: QueryTestScenario,
        rag_system,
        pdf_processor,
        temp_dir: Path
    ) -> EndToEndWorkflowResult:
        """Validate complete end-to-end query processing workflow."""
        
        logging.info(f"Starting end-to-end workflow validation: {scenario.name}")
        workflow_start_time = time.time()
        
        # Phase 1: PDF Ingestion and Knowledge Base Initialization
        papers_dir = temp_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        
        # Create realistic PDF files for the scenario
        pdf_files = await self._create_scenario_pdfs(scenario.pdf_collection, papers_dir)
        
        try:
            # Initialize knowledge base with PDF ingestion
            ingestion_start = time.time()
            ingestion_result = await rag_system.initialize_knowledge_base(
                papers_dir=papers_dir,
                enable_unified_progress_tracking=True
            )
            ingestion_time = time.time() - ingestion_start
            
            # Validate ingestion success
            ingestion_success = ingestion_result.get('success', False)
            documents_processed = ingestion_result.get('documents_processed', 0)
            
            if not ingestion_success:
                return self._create_failed_result(
                    scenario.scenario_id, 
                    "PDF ingestion failed", 
                    time.time() - workflow_start_time
                )
            
            # Phase 2: Execute Query Test Sets
            query_results = []
            total_cost = ingestion_result.get('cost_usd', 0.0)
            
            for query_type, queries in scenario.query_sets.items():
                for query in queries:
                    for query_mode in scenario.query_modes_to_test:
                        try:
                            query_result = await self._execute_and_validate_query(
                                rag_system, query, query_type, query_mode, scenario
                            )
                            query_results.append(query_result)
                            total_cost += query_result.cost_usd
                            
                        except Exception as e:
                            logging.error(f"Query execution failed: {query} (mode: {query_mode}) - {e}")
                            query_results.append(QueryExecutionResult(
                                query=query,
                                query_type=query_type,
                                query_mode=query_mode,
                                response_content="",
                                execution_time=0.0,
                                cost_usd=0.0,
                                quality_metrics=ResponseQualityMetrics(0, 0, 0, 0, 0, 0, 0),
                                success=False,
                                error_message=str(e)
                            ))
            
            # Phase 3: Analyze Workflow Results
            total_workflow_time = time.time() - workflow_start_time
            successful_queries = [r for r in query_results if r.success]
            
            # Calculate performance metrics
            avg_query_time = statistics.mean([r.execution_time for r in successful_queries]) if successful_queries else 0.0
            avg_quality_score = statistics.mean([r.quality_metrics.overall_quality_score for r in successful_queries]) if successful_queries else 0.0
            
            # Check cross-document synthesis capability
            synthesis_success = self._validate_cross_document_synthesis(query_results, scenario)
            
            # Generate performance and quality flags
            performance_flags = self._analyze_performance_flags(query_results, scenario)
            quality_flags = self._analyze_quality_flags(query_results, scenario)
            biomedical_flags = self._analyze_biomedical_accuracy_flags(query_results, scenario)
            
            # Create comprehensive result
            result = EndToEndWorkflowResult(
                scenario_id=scenario.scenario_id,
                ingestion_success=ingestion_success,
                documents_processed=documents_processed,
                queries_executed=len(query_results),
                query_results=query_results,
                total_workflow_time=total_workflow_time,
                total_cost=total_cost,
                average_query_time=avg_query_time,
                average_quality_score=avg_quality_score,
                cross_document_synthesis_success=synthesis_success,
                performance_flags=performance_flags,
                quality_flags=quality_flags,
                biomedical_accuracy_flags=biomedical_flags
            )
            
            self.validation_history.append(result)
            logging.info(f"End-to-end workflow validation completed: {scenario.name}")
            return result
            
        except Exception as e:
            logging.error(f"Critical error in end-to-end workflow: {e}")
            return self._create_failed_result(scenario.scenario_id, str(e), time.time() - workflow_start_time)
    
    async def _create_scenario_pdfs(self, pdf_collection: List[str], papers_dir: Path) -> List[Path]:
        """Create realistic PDF files for the scenario."""
        pdf_files = []
        
        # Enhanced biomedical content based on filename patterns
        content_templates = {
            'clinical_metabolomics': """
                Clinical metabolomics represents the application of metabolomics technologies to clinical
                medicine and healthcare. This field involves comprehensive analysis of small molecules
                (metabolites) in biological samples such as blood, urine, and tissue to understand disease
                processes, diagnose conditions, and develop personalized medicine approaches.
                
                Key applications include biomarker discovery for disease diagnosis, understanding metabolic
                pathways altered in disease states, supporting drug development through mechanism elucidation,
                and enabling personalized treatment strategies based on individual metabolic profiles.
                
                Analytical platforms commonly used include LC-MS/MS for targeted and untargeted profiling,
                GC-MS for volatile metabolites, and NMR spectroscopy for quantitative analysis.
            """,
            'diabetes_metabolomics': """
                Type 2 diabetes mellitus represents a complex metabolic disorder with significant metabolomic
                alterations. This study analyzed plasma samples from 200 diabetes patients and 100 controls
                using LC-MS/MS techniques. Key findings include elevated branched-chain amino acids (leucine,
                isoleucine, valine), altered glucose metabolism intermediates, and disrupted lipid profiles.
                
                Statistical analysis revealed 47 significantly altered metabolites (p < 0.05). Pathway analysis
                indicated enrichment in glycolysis, gluconeogenesis, and amino acid metabolism. These metabolic
                signatures provide insights into disease progression and potential therapeutic targets.
            """,
            'cardiovascular_proteomics': """
                Cardiovascular disease proteomics revealed distinct protein expression patterns in heart failure
                patients. iTRAQ-based analysis of 180 patients and 120 controls identified 89 differentially
                expressed proteins including troponin I, BNP, and inflammatory markers like CRP and TNF-alpha.
                
                Pathway analysis indicated involvement in cardiac remodeling, complement activation, and
                blood coagulation processes. These protein biomarkers offer potential for improved diagnosis
                and monitoring of cardiovascular disease progression.
            """,
            'cancer_genomics': """
                Cancer genomics analysis of tumor metabolism revealed fundamental reprogramming supporting
                rapid proliferation. RNA-seq of 200 tumor samples identified 1,234 differentially expressed
                genes involved in metabolic pathways. Key alterations include enhanced glycolysis (Warburg effect),
                glutamine addiction, and altered one-carbon metabolism.
                
                Integrated metabolomics identified 156 altered metabolites including elevated lactate,
                glutamine, and nucleotide precursors. These findings highlight metabolic vulnerabilities
                as therapeutic targets in precision oncology.
            """
        }
        
        for pdf_filename in pdf_collection:
            pdf_path = papers_dir / pdf_filename
            
            # Select appropriate content based on filename
            content = ""
            filename_lower = pdf_filename.lower()
            
            if 'clinical_metabolomics' in filename_lower or 'Clinical_Metabolomics' in pdf_filename:
                content = content_templates['clinical_metabolomics']
            elif 'diabetes' in filename_lower:
                content = content_templates['diabetes_metabolomics']
            elif 'cardiovascular' in filename_lower:
                content = content_templates['cardiovascular_proteomics']
            elif 'cancer' in filename_lower:
                content = content_templates['cancer_genomics']
            else:
                # Generic biomedical research content
                content = f"""
                    This biomedical research study investigates molecular mechanisms underlying disease
                    progression. The research utilized advanced analytical techniques including mass
                    spectrometry, proteomics, and genomics approaches. Statistical analysis was performed
                    using appropriate methods with multiple testing correction.
                    
                    Key findings include identification of potential biomarkers and therapeutic targets.
                    The results contribute to our understanding of disease pathophysiology and support
                    the development of precision medicine approaches.
                    
                    Study ID: {pdf_filename.replace('.pdf', '')}
                """
            
            # Create enhanced content with metadata
            enhanced_content = f"""
Title: {pdf_filename.replace('.pdf', '').replace('_', ' ').title()}
Authors: Dr. Researcher A, Dr. Scientist B, Dr. Expert C
Journal: Journal of Biomedical Research
Year: {random.randint(2020, 2024)}
DOI: 10.1000/test.{random.randint(1000, 9999)}
Keywords: biomarkers, clinical research, analytical methods, precision medicine

Abstract:
{content.strip()}

Methods: Comprehensive analytical approaches were employed including sample collection,
preparation, instrumental analysis, and statistical evaluation. Quality control measures
were implemented throughout the analytical workflow.

Results: Significant alterations were identified in molecular profiles with statistical
significance (p < 0.05). Pathway analysis revealed biological relevance of findings.

Conclusions: These results advance our understanding of disease mechanisms and provide
foundation for clinical translation of biomarker discoveries.
            """
            
            # Write enhanced content to file
            pdf_path.write_text(enhanced_content.strip())
            pdf_files.append(pdf_path)
        
        return pdf_files
    
    async def _execute_and_validate_query(
        self,
        rag_system,
        query: str,
        query_type: str,
        query_mode: str,
        scenario: QueryTestScenario
    ) -> QueryExecutionResult:
        """Execute and validate a single query with comprehensive metrics."""
        
        query_start = time.time()
        
        # Execute query with performance monitoring
        try:
            response, perf_metrics = await self.performance_monitor.monitor_query_performance(
                rag_system.query, query, mode=query_mode
            )
            
            execution_time = time.time() - query_start
            
            # Assess response quality
            quality_metrics = self.quality_assessor.assess_response_quality(response, query)
            
            # Extract biomedical entities and synthesis indicators
            biomedical_entities = self._extract_biomedical_entities(response)
            synthesis_indicators = self._extract_synthesis_indicators(response)
            context_retrieved = self._extract_context_references(response)
            sources_referenced = self._extract_source_references(response)
            
            return QueryExecutionResult(
                query=query,
                query_type=query_type,
                query_mode=query_mode,
                response_content=response,
                execution_time=execution_time,
                cost_usd=perf_metrics.cost_usd,
                quality_metrics=quality_metrics,
                context_retrieved=context_retrieved,
                sources_referenced=sources_referenced,
                biomedical_entities_found=biomedical_entities,
                synthesis_indicators=synthesis_indicators,
                success=True
            )
            
        except Exception as e:
            return QueryExecutionResult(
                query=query,
                query_type=query_type,
                query_mode=query_mode,
                response_content="",
                execution_time=time.time() - query_start,
                cost_usd=0.0,
                quality_metrics=ResponseQualityMetrics(0, 0, 0, 0, 0, 0, 0),
                success=False,
                error_message=str(e)
            )
    
    def _extract_biomedical_entities(self, response: str) -> List[str]:
        """Extract biomedical entities from response content."""
        response_lower = response.lower()
        entities = []
        
        # Common biomedical entities
        biomedical_terms = [
            'metabolomics', 'proteomics', 'genomics', 'biomarkers', 'metabolites',
            'proteins', 'genes', 'pathways', 'glucose', 'insulin', 'diabetes',
            'cardiovascular', 'cancer', 'lc-ms', 'gc-ms', 'nmr', 'mass spectrometry',
            'clinical', 'diagnosis', 'treatment', 'therapy', 'patient', 'disease'
        ]
        
        for term in biomedical_terms:
            if term in response_lower:
                entities.append(term)
        
        return entities
    
    def _extract_synthesis_indicators(self, response: str) -> List[str]:
        """Extract indicators of cross-document synthesis."""
        response_lower = response.lower()
        indicators = []
        
        synthesis_terms = [
            'across studies', 'multiple papers', 'different research', 'various investigations',
            'collectively', 'together', 'combined', 'integrated', 'comprehensive review',
            'meta-analysis', 'systematic', 'literature shows', 'studies indicate',
            'research suggests', 'evidence from', 'comparing', 'contrast'
        ]
        
        for term in synthesis_terms:
            if term in response_lower:
                indicators.append(term)
        
        return indicators
    
    def _extract_context_references(self, response: str) -> List[str]:
        """Extract context references from response."""
        # Simple implementation - could be enhanced with more sophisticated parsing
        context_refs = []
        response_lower = response.lower()
        
        if 'based on' in response_lower:
            context_refs.append('explicit_source_reference')
        if 'according to' in response_lower:
            context_refs.append('attribution_reference')
        if 'literature' in response_lower:
            context_refs.append('literature_reference')
        if 'studies' in response_lower or 'research' in response_lower:
            context_refs.append('research_reference')
        
        return context_refs
    
    def _extract_source_references(self, response: str) -> List[str]:
        """Extract source document references from response."""
        # Implementation would parse actual source references
        # For testing purposes, return mock references
        sources = []
        if len(response) > 100:  # Substantial response likely draws from sources
            sources.append('knowledge_base_reference')
        if 'study' in response.lower() or 'research' in response.lower():
            sources.append('research_document_reference')
        
        return sources
    
    def _validate_cross_document_synthesis(
        self, 
        query_results: List[QueryExecutionResult], 
        scenario: QueryTestScenario
    ) -> bool:
        """Validate that cross-document synthesis occurred successfully."""
        if not scenario.cross_document_synthesis_expected:
            return True  # Not required for this scenario
        
        synthesis_queries = [r for r in query_results if r.query_type == 'cross_document_synthesis']
        if not synthesis_queries:
            return False
        
        # Check if synthesis queries have synthesis indicators
        successful_synthesis = 0
        for query_result in synthesis_queries:
            if (len(query_result.synthesis_indicators) >= 2 and
                len(query_result.biomedical_entities_found) >= 3 and
                query_result.quality_metrics.overall_quality_score >= 70.0):
                successful_synthesis += 1
        
        # At least 75% of synthesis queries should show successful synthesis
        return successful_synthesis >= len(synthesis_queries) * 0.75
    
    def _analyze_performance_flags(
        self, 
        query_results: List[QueryExecutionResult], 
        scenario: QueryTestScenario
    ) -> List[str]:
        """Analyze performance and generate flags."""
        flags = []
        
        # Check individual query performance against expectations
        for query_result in query_results:
            query_type = query_result.query_type
            expected_max_time_key = f"{query_type}_max_time"
            expected_max_time = scenario.expected_performance.get(expected_max_time_key, 30.0)
            
            if query_result.execution_time > expected_max_time:
                flags.append(f"SLOW_QUERY_{query_type.upper()}")
        
        # Check average performance
        successful_queries = [r for r in query_results if r.success]
        if successful_queries:
            avg_time = statistics.mean([r.execution_time for r in successful_queries])
            if avg_time > 20.0:
                flags.append("OVERALL_SLOW_PERFORMANCE")
        
        # Check for excessive failures
        failure_rate = len([r for r in query_results if not r.success]) / len(query_results) if query_results else 0
        if failure_rate > 0.1:  # More than 10% failures
            flags.append("HIGH_FAILURE_RATE")
        
        return flags
    
    def _analyze_quality_flags(
        self, 
        query_results: List[QueryExecutionResult], 
        scenario: QueryTestScenario
    ) -> List[str]:
        """Analyze quality metrics and generate flags."""
        flags = []
        
        # Check individual query quality against expectations
        for query_result in query_results:
            query_type = query_result.query_type
            expected_min_relevance_key = f"{query_type}_min_relevance"
            expected_min_relevance = scenario.expected_quality.get(expected_min_relevance_key, 70.0)
            
            if query_result.quality_metrics.relevance_score < expected_min_relevance:
                flags.append(f"LOW_RELEVANCE_{query_type.upper()}")
            
            if query_result.quality_metrics.overall_quality_score < 65.0:
                flags.append(f"LOW_QUALITY_{query_type.upper()}")
        
        # Check average quality
        successful_queries = [r for r in query_results if r.success]
        if successful_queries:
            avg_quality = statistics.mean([r.quality_metrics.overall_quality_score for r in successful_queries])
            if avg_quality < 70.0:
                flags.append("OVERALL_LOW_QUALITY")
        
        return flags
    
    def _analyze_biomedical_accuracy_flags(
        self, 
        query_results: List[QueryExecutionResult], 
        scenario: QueryTestScenario
    ) -> List[str]:
        """Analyze biomedical accuracy and generate flags."""
        flags = []
        
        if not scenario.biomedical_validation_required:
            return flags
        
        # Check biomedical entity extraction
        for query_result in query_results:
            if len(query_result.biomedical_entities_found) < 2:
                flags.append(f"INSUFFICIENT_BIOMEDICAL_ENTITIES")
            
            if query_result.quality_metrics.biomedical_terminology_score < 60.0:
                flags.append(f"LOW_BIOMEDICAL_TERMINOLOGY")
        
        # Check factual accuracy
        low_accuracy_count = len([
            r for r in query_results 
            if r.success and r.quality_metrics.factual_accuracy_score < 70.0
        ])
        
        if low_accuracy_count > len(query_results) * 0.25:  # More than 25% low accuracy
            flags.append("HIGH_FACTUAL_INACCURACY")
        
        return flags
    
    def _create_failed_result(self, scenario_id: str, error_message: str, workflow_time: float) -> EndToEndWorkflowResult:
        """Create a failed workflow result."""
        return EndToEndWorkflowResult(
            scenario_id=scenario_id,
            ingestion_success=False,
            documents_processed=0,
            queries_executed=0,
            query_results=[],
            total_workflow_time=workflow_time,
            total_cost=0.0,
            average_query_time=0.0,
            average_quality_score=0.0,
            cross_document_synthesis_success=False,
            performance_flags=["WORKFLOW_FAILED"],
            quality_flags=["VALIDATION_FAILED"],
            biomedical_accuracy_flags=["ACCURACY_VALIDATION_FAILED"]
        )


# =====================================================================
# TEST FIXTURES
# =====================================================================

@pytest.fixture
def query_scenario_builder():
    """Provide query scenario builder."""
    return EndToEndQueryScenarioBuilder()


@pytest.fixture
def enhanced_workflow_validator():
    """Provide enhanced workflow validator."""
    return EnhancedWorkflowValidator()


@pytest.fixture
def mock_enhanced_rag_system(mock_config):
    """Create enhanced mock RAG system with realistic query responses."""
    rag = MockClinicalMetabolomicsRAG(mock_config)
    
    # Override query method with enhanced responses
    original_query = rag.query
    
    async def enhanced_query(question: str, mode: str = "hybrid", **kwargs) -> str:
        """Enhanced query with mode-specific and complexity-aware responses."""
        await asyncio.sleep(random.uniform(0.5, 2.0))  # Realistic processing time
        
        question_lower = question.lower()
        
        # Mode-specific response variations
        mode_prefix = {
            'local': 'Based on local document analysis, ',
            'global': 'From comprehensive knowledge synthesis, ',
            'hybrid': 'Integrating multiple information sources, ',
            'naive': 'According to the available information, '
        }
        
        prefix = mode_prefix.get(mode, '')
        
        # Enhanced response generation based on query complexity
        if any(term in question_lower for term in ['compare', 'contrast', 'synthesize', 'across']):
            # Cross-document synthesis queries
            response = f"""{prefix}the analysis across multiple studies reveals several key insights. 
            Research from different investigations shows complementary findings in biomarker identification 
            and analytical methodologies. The literature collectively indicates significant advances in 
            clinical applications, with studies demonstrating consistent patterns in metabolomic profiles. 
            
            Comparing methodological approaches, various investigations have employed LC-MS/MS, GC-MS, and 
            NMR techniques with different sample preparation protocols. The evidence from multiple research 
            papers suggests that integrated analytical platforms provide the most comprehensive insights 
            into disease mechanisms and biomarker discovery.
            
            These findings support the development of precision medicine approaches through systematic 
            integration of multi-omics data across different disease contexts."""
            
        elif any(term in question_lower for term in ['workflow', 'methods', 'analytical', 'technical']):
            # Complex analytical queries
            response = f"""{prefix}the comprehensive analytical workflow involves multiple critical steps. 
            Sample collection follows standardized protocols with appropriate storage conditions. Sample 
            preparation includes protein precipitation, extraction, and cleanup procedures optimized for 
            the analytical platform.
            
            Instrumental analysis utilizes state-of-the-art mass spectrometry systems including LC-MS/MS 
            for targeted analysis and high-resolution accurate mass instruments for untargeted profiling. 
            Data processing employs specialized software for peak detection, alignment, and statistical analysis.
            
            Quality control measures are implemented throughout including internal standards, blank samples, 
            and biological replicates. Statistical analysis incorporates multiple testing correction and 
            appropriate multivariate approaches for biomarker discovery and validation."""
            
        elif any(term in question_lower for term in ['what is', 'define', 'what are']):
            # Simple factual queries  
            base_response = await original_query(question, **kwargs)
            response = f"{prefix}{base_response}"
            
        else:
            # General queries
            response = f"""{prefix}clinical research in metabolomics and related fields has advanced 
            significantly in recent years. The integration of analytical chemistry, statistics, and 
            clinical medicine enables comprehensive investigation of disease mechanisms and biomarker 
            discovery. Advanced analytical platforms provide unprecedented insights into molecular 
            alterations associated with various disease states, supporting the development of precision 
            medicine approaches."""
        
        return response
    
    rag.query = enhanced_query
    return rag


@pytest.fixture
def real_clinical_paper_path():
    """Path to real clinical metabolomics paper if available."""
    paper_path = Path("/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf")
    return paper_path if paper_path.exists() else None


# =====================================================================
# CORE END-TO-END WORKFLOW TESTS
# =====================================================================

@pytest.mark.biomedical
@pytest.mark.integration
@pytest.mark.lightrag
class TestEndToEndQueryWorkflow:
    """Core end-to-end query processing workflow tests."""

    @pytest.mark.asyncio
    async def test_complete_clinical_metabolomics_workflow(
        self,
        temp_dir,
        mock_config,
        query_scenario_builder,
        enhanced_workflow_validator,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test complete end-to-end workflow for clinical metabolomics queries."""
        
        # Setup configuration
        mock_config.working_dir = temp_dir / "knowledge_base"
        
        # Build comprehensive scenario
        scenario = query_scenario_builder.build_clinical_metabolomics_comprehensive_scenario()
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            # Setup mocks
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True, 'cost': 0.05})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Execute complete workflow validation
            result = await enhanced_workflow_validator.validate_end_to_end_workflow(
                scenario,
                mock_enhanced_rag_system,
                mock_comprehensive_pdf_processor,
                temp_dir
            )
            
            # Comprehensive workflow assertions
            assert result.ingestion_success == True, "PDF ingestion should succeed"
            assert result.documents_processed >= len(scenario.pdf_collection), "All PDFs should be processed"
            assert result.queries_executed > 0, "Queries should be executed"
            assert result.total_workflow_time < 300.0, "Workflow should complete within 5 minutes"
            
            # Query execution validation
            successful_queries = [r for r in result.query_results if r.success]
            assert len(successful_queries) >= len(result.query_results) * 0.8, "At least 80% of queries should succeed"
            
            # Performance validation
            assert result.average_query_time < 30.0, f"Average query time {result.average_query_time}s should be reasonable"
            assert result.average_quality_score >= 70.0, f"Average quality {result.average_quality_score}% should meet threshold"
            
            # Cross-document synthesis validation
            if scenario.cross_document_synthesis_expected:
                assert result.cross_document_synthesis_success == True, "Cross-document synthesis should succeed"
            
            # Quality flags validation
            critical_quality_flags = [f for f in result.quality_flags if 'LOW_QUALITY' in f]
            assert len(critical_quality_flags) <= 2, f"Should have minimal critical quality issues: {critical_quality_flags}"
            
            # Performance flags validation
            critical_performance_flags = [f for f in result.performance_flags if 'SLOW' in f or 'FAILED' in f]
            assert len(critical_performance_flags) <= 1, f"Should have minimal critical performance issues: {critical_performance_flags}"
            
            # Log comprehensive results
            logging.info(f"✅ Complete E2E Workflow Results for {scenario.name}:")
            logging.info(f"  - Ingestion: {result.documents_processed} PDFs processed successfully")
            logging.info(f"  - Queries: {len(successful_queries)}/{result.queries_executed} successful")
            logging.info(f"  - Performance: {result.average_query_time:.2f}s avg query time")
            logging.info(f"  - Quality: {result.average_quality_score:.1f}% avg quality score")
            logging.info(f"  - Synthesis: {'✅' if result.cross_document_synthesis_success else '❌'}")
            logging.info(f"  - Total Cost: ${result.total_cost:.3f}")
            logging.info(f"  - Workflow Time: {result.total_workflow_time:.1f}s")

    @pytest.mark.asyncio
    async def test_multi_disease_biomarker_synthesis_workflow(
        self,
        temp_dir,
        mock_config,
        query_scenario_builder,
        enhanced_workflow_validator,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test end-to-end workflow for multi-disease biomarker synthesis."""
        
        # Setup
        mock_config.working_dir = temp_dir / "knowledge_base"
        scenario = query_scenario_builder.build_multi_disease_biomarker_scenario()
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True, 'cost': 0.08})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Execute workflow
            result = await enhanced_workflow_validator.validate_end_to_end_workflow(
                scenario,
                mock_enhanced_rag_system,
                mock_comprehensive_pdf_processor,
                temp_dir
            )
            
            # Multi-disease specific validations
            assert result.ingestion_success == True
            assert result.documents_processed == len(scenario.pdf_collection)
            assert result.cross_document_synthesis_success == True, "Multi-disease synthesis should succeed"
            
            # Validate cross-disease comparison capabilities
            cross_disease_queries = [
                r for r in result.query_results 
                if r.query_type == 'cross_disease_comparison' and r.success
            ]
            assert len(cross_disease_queries) > 0, "Should successfully handle cross-disease queries"
            
            # Check for synthesis indicators in responses
            synthesis_queries = [r for r in cross_disease_queries if len(r.synthesis_indicators) >= 2]
            assert len(synthesis_queries) >= len(cross_disease_queries) * 0.5, "Should show synthesis in responses"
            
            # Biomedical entity validation for multi-disease context
            all_entities = set()
            for query_result in result.query_results:
                all_entities.update(query_result.biomedical_entities_found)
            
            disease_entities = {'diabetes', 'cardiovascular', 'cancer', 'liver', 'kidney'}
            found_diseases = disease_entities.intersection(all_entities)
            assert len(found_diseases) >= 3, f"Should reference multiple diseases: {found_diseases}"
            
            logging.info(f"✅ Multi-Disease Workflow Results:")
            logging.info(f"  - Cross-disease synthesis: {'✅' if result.cross_document_synthesis_success else '❌'}")
            logging.info(f"  - Disease entities found: {found_diseases}")
            logging.info(f"  - Synthesis queries successful: {len(synthesis_queries)}/{len(cross_disease_queries)}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_performance_stress_workflow(
        self,
        temp_dir,
        mock_config,
        query_scenario_builder,
        enhanced_workflow_validator,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test end-to-end workflow under performance stress conditions."""
        
        # Setup
        mock_config.working_dir = temp_dir / "knowledge_base"
        scenario = query_scenario_builder.build_performance_stress_scenario()
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True, 'cost': 0.15})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Execute stress test workflow
            start_time = time.time()
            result = await enhanced_workflow_validator.validate_end_to_end_workflow(
                scenario,
                mock_enhanced_rag_system,
                mock_comprehensive_pdf_processor,
                temp_dir
            )
            total_time = time.time() - start_time
            
            # Performance stress validations
            assert result.ingestion_success == True
            assert result.documents_processed == len(scenario.pdf_collection)
            assert total_time < 600.0, f"Stress test should complete within 10 minutes: {total_time:.1f}s"
            
            # Query performance validation
            rapid_fire_queries = [r for r in result.query_results if r.query_type == 'rapid_fire']
            if rapid_fire_queries:
                avg_rapid_time = statistics.mean([r.execution_time for r in rapid_fire_queries])
                assert avg_rapid_time < 10.0, f"Rapid fire queries too slow: {avg_rapid_time:.1f}s"
            
            # Complex synthesis performance
            complex_queries = [r for r in result.query_results if r.query_type == 'complex_synthesis']
            if complex_queries:
                for query in complex_queries:
                    assert query.execution_time < 60.0, f"Complex query too slow: {query.execution_time:.1f}s"
            
            # Edge case handling validation
            edge_case_queries = [r for r in result.query_results if r.query_type == 'edge_case']
            if edge_case_queries:
                # Should handle edge cases gracefully (not crash)
                successful_edge_cases = [r for r in edge_case_queries if r.success]
                # At least some edge cases should be handled successfully
                assert len(successful_edge_cases) >= len(edge_case_queries) * 0.5, "Should handle edge cases gracefully"
            
            # Resource efficiency validation
            queries_per_second = result.queries_executed / total_time if total_time > 0 else 0
            assert queries_per_second > 0.5, f"Query throughput too low: {queries_per_second:.2f} q/s"
            
            logging.info(f"✅ Performance Stress Test Results:")
            logging.info(f"  - Total workflow time: {total_time:.1f}s")
            logging.info(f"  - Documents processed: {result.documents_processed}")
            logging.info(f"  - Queries executed: {result.queries_executed}")
            logging.info(f"  - Query throughput: {queries_per_second:.2f} queries/second")
            logging.info(f"  - Average query time: {result.average_query_time:.2f}s")
            logging.info(f"  - Performance flags: {result.performance_flags}")


# =====================================================================
# QUERY TYPE VALIDATION TESTS
# =====================================================================

@pytest.mark.biomedical
@pytest.mark.integration
class TestQueryTypeValidation:
    """Test validation of different query types and complexity levels."""

    @pytest.mark.asyncio
    async def test_simple_factual_query_validation(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test simple factual queries for accuracy and performance."""
        
        # Setup simple scenario
        simple_queries = [
            "What is clinical metabolomics?",
            "What is LC-MS?",
            "Define metabolic biomarkers",
            "What are metabolites?",
            "What is mass spectrometry?"
        ]
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize system
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            (papers_dir / "test.pdf").write_text("Clinical metabolomics content")
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test each simple query
            query_results = []
            for query in simple_queries:
                start_time = time.time()
                response = await mock_enhanced_rag_system.query(query)
                execution_time = time.time() - start_time
                
                # Quality assessment
                quality_metrics = ResponseQualityAssessor.assess_response_quality(response, query)
                
                query_results.append({
                    'query': query,
                    'response': response,
                    'execution_time': execution_time,
                    'quality_metrics': quality_metrics
                })
            
            # Validate simple query characteristics
            for result in query_results:
                # Performance expectations for simple queries
                assert result['execution_time'] < 10.0, f"Simple query too slow: {result['execution_time']:.1f}s"
                
                # Quality expectations for simple queries
                assert result['quality_metrics'].relevance_score >= 80.0, \
                    f"Simple query relevance too low: {result['quality_metrics'].relevance_score}%"
                
                # Content expectations
                assert len(result['response']) >= 100, "Simple queries should have substantial responses"
                assert any(term in result['response'].lower() for term in ['clinical', 'metabolomic', 'biomarker', 'mass']), \
                    "Simple queries should contain relevant biomedical terms"
            
            # Overall simple query performance
            avg_time = statistics.mean([r['execution_time'] for r in query_results])
            avg_quality = statistics.mean([r['quality_metrics'].overall_quality_score for r in query_results])
            
            assert avg_time < 8.0, f"Average simple query time too high: {avg_time:.1f}s"
            assert avg_quality >= 80.0, f"Average simple query quality too low: {avg_quality:.1f}%"
            
            logging.info(f"✅ Simple Factual Query Validation:")
            logging.info(f"  - Queries tested: {len(query_results)}")
            logging.info(f"  - Average execution time: {avg_time:.2f}s")
            logging.info(f"  - Average quality score: {avg_quality:.1f}%")

    @pytest.mark.asyncio
    async def test_complex_analytical_query_validation(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test complex analytical queries requiring deeper analysis."""
        
        # Complex analytical queries
        complex_queries = [
            "Compare LC-MS versus GC-MS analytical approaches for metabolomics",
            "Explain the complete workflow for metabolomic biomarker discovery",
            "What are the advantages of targeted versus untargeted metabolomics?",
            "How do sample preparation methods affect metabolomic results?",
            "What statistical methods are most appropriate for metabolomics data analysis?"
        ]
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize with comprehensive content
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            (papers_dir / "methods.pdf").write_text("""
                Comprehensive analytical methods in metabolomics include LC-MS/MS for targeted analysis
                and high-resolution mass spectrometry for untargeted profiling. Sample preparation 
                involves protein precipitation, extraction, and cleanup procedures. Statistical analysis
                employs multivariate methods including PCA, OPLS-DA, and univariate testing with FDR correction.
            """)
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test complex queries
            complex_results = []
            for query in complex_queries:
                start_time = time.time()
                response = await mock_enhanced_rag_system.query(query)
                execution_time = time.time() - start_time
                
                quality_metrics = ResponseQualityAssessor.assess_response_quality(response, query)
                
                # Check for analytical complexity indicators
                complexity_indicators = [
                    'compare', 'contrast', 'advantages', 'disadvantages', 'workflow',
                    'methods', 'approaches', 'procedures', 'considerations'
                ]
                
                response_lower = response.lower()
                complexity_score = sum(1 for indicator in complexity_indicators if indicator in response_lower)
                
                complex_results.append({
                    'query': query,
                    'response': response,
                    'execution_time': execution_time,
                    'quality_metrics': quality_metrics,
                    'complexity_score': complexity_score
                })
            
            # Validate complex query characteristics
            for result in complex_results:
                # Performance expectations (more lenient for complex queries)
                assert result['execution_time'] < 30.0, f"Complex query too slow: {result['execution_time']:.1f}s"
                
                # Quality expectations
                assert result['quality_metrics'].relevance_score >= 75.0, \
                    f"Complex query relevance too low: {result['quality_metrics'].relevance_score}%"
                
                # Complexity validation
                assert result['complexity_score'] >= 2, \
                    f"Complex query response lacks analytical depth: {result['complexity_score']}"
                
                # Content length expectation for complex queries
                assert len(result['response']) >= 200, \
                    f"Complex query response too brief: {len(result['response'])} chars"
            
            # Overall complex query performance
            avg_time = statistics.mean([r['execution_time'] for r in complex_results])
            avg_quality = statistics.mean([r['quality_metrics'].overall_quality_score for r in complex_results])
            avg_complexity = statistics.mean([r['complexity_score'] for r in complex_results])
            
            assert avg_time < 25.0, f"Average complex query time too high: {avg_time:.1f}s"
            assert avg_quality >= 75.0, f"Average complex query quality too low: {avg_quality:.1f}%"
            assert avg_complexity >= 2.5, f"Average complexity score too low: {avg_complexity:.1f}"
            
            logging.info(f"✅ Complex Analytical Query Validation:")
            logging.info(f"  - Queries tested: {len(complex_results)}")
            logging.info(f"  - Average execution time: {avg_time:.2f}s")
            logging.info(f"  - Average quality score: {avg_quality:.1f}%")
            logging.info(f"  - Average complexity score: {avg_complexity:.1f}")

    @pytest.mark.asyncio
    async def test_cross_document_synthesis_query_validation(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test cross-document synthesis queries for integration capabilities."""
        
        # Cross-document synthesis queries
        synthesis_queries = [
            "Compare biomarkers identified across different disease studies",
            "What methodological approaches are shared across research papers?",
            "Synthesize quality control recommendations from multiple sources",
            "Identify common analytical platforms used across studies",
            "What are the consistent findings across cardiovascular and diabetes research?"
        ]
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize with multiple diverse documents
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            
            # Create multiple papers with different focuses
            papers = {
                "diabetes_study.pdf": "Diabetes metabolomics study using LC-MS with glucose and insulin biomarkers",
                "cardiovascular_study.pdf": "Cardiovascular proteomics research using iTRAQ with troponin and BNP biomarkers",
                "cancer_study.pdf": "Cancer genomics analysis using RNA-seq with lactate and glutamine biomarkers"
            }
            
            for filename, content in papers.items():
                (papers_dir / filename).write_text(content)
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test synthesis queries
            synthesis_results = []
            for query in synthesis_queries:
                start_time = time.time()
                response = await mock_enhanced_rag_system.query(query)
                execution_time = time.time() - start_time
                
                quality_metrics = ResponseQualityAssessor.assess_response_quality(response, query)
                
                # Check for synthesis indicators
                response_lower = response.lower()
                synthesis_indicators = [
                    'across studies', 'multiple', 'different', 'various', 'compare',
                    'collectively', 'together', 'integrated', 'comprehensive'
                ]
                synthesis_score = sum(1 for indicator in synthesis_indicators if indicator in response_lower)
                
                # Check for multi-document references
                document_references = ['diabetes', 'cardiovascular', 'cancer']
                reference_score = sum(1 for ref in document_references if ref in response_lower)
                
                synthesis_results.append({
                    'query': query,
                    'response': response,
                    'execution_time': execution_time,
                    'quality_metrics': quality_metrics,
                    'synthesis_score': synthesis_score,
                    'reference_score': reference_score
                })
            
            # Validate synthesis characteristics
            for result in synthesis_results:
                # Performance (synthesis queries may take longer)
                assert result['execution_time'] < 40.0, f"Synthesis query too slow: {result['execution_time']:.1f}s"
                
                # Quality validation
                assert result['quality_metrics'].relevance_score >= 70.0, \
                    f"Synthesis query relevance too low: {result['quality_metrics'].relevance_score}%"
                
                # Synthesis validation
                assert result['synthesis_score'] >= 2, \
                    f"Insufficient synthesis indicators: {result['synthesis_score']}"
                
                # Cross-document reference validation
                assert result['reference_score'] >= 2, \
                    f"Insufficient cross-document references: {result['reference_score']}"
                
                # Content length for synthesis queries
                assert len(result['response']) >= 250, \
                    f"Synthesis response too brief: {len(result['response'])} chars"
            
            # Overall synthesis performance
            avg_time = statistics.mean([r['execution_time'] for r in synthesis_results])
            avg_quality = statistics.mean([r['quality_metrics'].overall_quality_score for r in synthesis_results])
            avg_synthesis = statistics.mean([r['synthesis_score'] for r in synthesis_results])
            avg_references = statistics.mean([r['reference_score'] for r in synthesis_results])
            
            assert avg_time < 35.0, f"Average synthesis query time too high: {avg_time:.1f}s"
            assert avg_quality >= 70.0, f"Average synthesis quality too low: {avg_quality:.1f}%"
            assert avg_synthesis >= 2.5, f"Average synthesis score too low: {avg_synthesis:.1f}"
            assert avg_references >= 2.0, f"Average reference score too low: {avg_references:.1f}"
            
            logging.info(f"✅ Cross-Document Synthesis Query Validation:")
            logging.info(f"  - Queries tested: {len(synthesis_results)}")
            logging.info(f"  - Average execution time: {avg_time:.2f}s")
            logging.info(f"  - Average quality score: {avg_quality:.1f}%")
            logging.info(f"  - Average synthesis score: {avg_synthesis:.1f}")
            logging.info(f"  - Average reference score: {avg_references:.1f}")


# =====================================================================
# QUERY MODE COMPARISON TESTS
# =====================================================================

@pytest.mark.biomedical
@pytest.mark.integration
class TestQueryModeComparison:
    """Test and compare different query modes (hybrid, local, global, naive)."""

    @pytest.mark.asyncio
    async def test_query_mode_performance_comparison(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Compare performance characteristics across different query modes."""
        
        test_query = "What are the key metabolomic biomarkers for diabetes?"
        query_modes = ['hybrid', 'local', 'global', 'naive']
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize knowledge base
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            (papers_dir / "diabetes.pdf").write_text("Diabetes metabolomics study with glucose, insulin, and lactate biomarkers")
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test each query mode
            mode_results = {}
            for mode in query_modes:
                start_time = time.time()
                response = await mock_enhanced_rag_system.query(test_query, mode=mode)
                execution_time = time.time() - start_time
                
                quality_metrics = ResponseQualityAssessor.assess_response_quality(response, test_query)
                
                mode_results[mode] = {
                    'response': response,
                    'execution_time': execution_time,
                    'quality_metrics': quality_metrics,
                    'response_length': len(response)
                }
            
            # Validate mode-specific characteristics
            for mode, result in mode_results.items():
                # All modes should respond within reasonable time
                assert result['execution_time'] < 20.0, f"{mode} mode too slow: {result['execution_time']:.1f}s"
                
                # All modes should provide relevant responses
                assert result['quality_metrics'].relevance_score >= 70.0, \
                    f"{mode} mode relevance too low: {result['quality_metrics'].relevance_score}%"
                
                # All modes should provide substantial responses
                assert result['response_length'] >= 100, \
                    f"{mode} mode response too brief: {result['response_length']} chars"
                
                # Mode-specific validation
                response_lower = result['response'].lower()
                if mode == 'local':
                    assert 'local' in response_lower or 'document' in response_lower, \
                        "Local mode should indicate local analysis"
                elif mode == 'global':
                    assert 'comprehensive' in response_lower or 'synthesis' in response_lower, \
                        "Global mode should indicate comprehensive analysis"
                elif mode == 'hybrid':
                    assert 'integrating' in response_lower or 'sources' in response_lower, \
                        "Hybrid mode should indicate integration"
            
            # Compare modes
            execution_times = [result['execution_time'] for result in mode_results.values()]
            quality_scores = [result['quality_metrics'].overall_quality_score for result in mode_results.values()]
            
            # Validate reasonable performance spread
            time_range = max(execution_times) - min(execution_times)
            assert time_range < 15.0, f"Excessive performance variation across modes: {time_range:.1f}s"
            
            # Validate consistent quality
            quality_range = max(quality_scores) - min(quality_scores)
            assert quality_range < 30.0, f"Excessive quality variation across modes: {quality_range:.1f}%"
            
            logging.info(f"✅ Query Mode Performance Comparison:")
            for mode, result in mode_results.items():
                logging.info(f"  - {mode}: {result['execution_time']:.2f}s, {result['quality_metrics'].overall_quality_score:.1f}%")

    @pytest.mark.asyncio
    async def test_query_mode_response_quality_differences(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test quality differences between query modes for different query types."""
        
        test_queries = {
            'simple': "What is clinical metabolomics?",
            'complex': "Compare analytical methods for biomarker discovery",
            'synthesis': "Synthesize findings across multiple disease studies"
        }
        
        query_modes = ['hybrid', 'local', 'global']
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize with diverse content
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            (papers_dir / "study1.pdf").write_text("Clinical metabolomics study with LC-MS analysis")
            (papers_dir / "study2.pdf").write_text("Biomarker discovery using GC-MS methods")
            (papers_dir / "study3.pdf").write_text("Multi-disease analysis with various analytical approaches")
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test each combination
            results = {}
            for query_type, query in test_queries.items():
                results[query_type] = {}
                
                for mode in query_modes:
                    response = await mock_enhanced_rag_system.query(query, mode=mode)
                    quality_metrics = ResponseQualityAssessor.assess_response_quality(response, query)
                    
                    results[query_type][mode] = {
                        'response': response,
                        'quality_metrics': quality_metrics
                    }
            
            # Validate mode effectiveness for different query types
            for query_type, mode_results in results.items():
                for mode, result in mode_results.items():
                    quality_score = result['quality_metrics'].overall_quality_score
                    
                    # Query type specific expectations
                    if query_type == 'simple':
                        # Simple queries should work well in all modes
                        assert quality_score >= 75.0, f"Simple query quality too low in {mode}: {quality_score}%"
                    elif query_type == 'synthesis':
                        # Synthesis queries should work better in global/hybrid modes
                        if mode in ['global', 'hybrid']:
                            assert quality_score >= 70.0, f"Synthesis query quality too low in {mode}: {quality_score}%"
                        # Local mode may have lower quality for synthesis
                        assert quality_score >= 60.0, f"Synthesis query failed in {mode}: {quality_score}%"
            
            # Compare mode effectiveness
            logging.info(f"✅ Query Mode Quality Comparison:")
            for query_type in test_queries.keys():
                logging.info(f"  {query_type} queries:")
                for mode in query_modes:
                    quality = results[query_type][mode]['quality_metrics'].overall_quality_score
                    logging.info(f"    - {mode}: {quality:.1f}%")


# =====================================================================
# CONTEXT RETRIEVAL VALIDATION TESTS
# =====================================================================

@pytest.mark.biomedical
@pytest.mark.integration
class TestContextRetrievalValidation:
    """Test context retrieval accuracy and relevance."""

    @pytest.mark.asyncio
    async def test_context_retrieval_accuracy(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test accuracy of context retrieval for queries."""
        
        # Create specific content to test retrieval
        test_content = {
            "metabolomics_methods.pdf": """
                LC-MS/MS is the gold standard for targeted metabolomics analysis.
                Sample preparation involves protein precipitation with methanol.
                Quality control requires internal standards and blank samples.
                Statistical analysis uses MetaboAnalyst software.
            """,
            "diabetes_biomarkers.pdf": """
                Glucose levels are elevated in diabetes patients.
                Branched-chain amino acids show significant alterations.
                HbA1c serves as a long-term glycemic control marker.
                Insulin resistance affects multiple metabolic pathways.
            """,
            "cardiovascular_study.pdf": """
                TMAO levels correlate with cardiovascular risk.
                Troponin I indicates cardiac muscle damage.
                BNP is elevated in heart failure patients.
                Cholesterol metabolism is disrupted in CVD.
            """
        }
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize knowledge base with specific content
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            
            for filename, content in test_content.items():
                (papers_dir / filename).write_text(content)
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test context retrieval with specific queries
            context_tests = [
                {
                    'query': "What sample preparation methods are used in metabolomics?",
                    'expected_context': ['protein precipitation', 'methanol', 'internal standards'],
                    'source_document': 'metabolomics_methods'
                },
                {
                    'query': "What biomarkers are associated with diabetes?",
                    'expected_context': ['glucose', 'hba1c', 'amino acids', 'insulin'],
                    'source_document': 'diabetes_biomarkers'
                },
                {
                    'query': "What are cardiovascular disease biomarkers?",
                    'expected_context': ['tmao', 'troponin', 'bnp', 'cholesterol'],
                    'source_document': 'cardiovascular_study'
                }
            ]
            
            retrieval_results = []
            for test in context_tests:
                response = await mock_enhanced_rag_system.query(test['query'])
                response_lower = response.lower()
                
                # Check context retrieval accuracy
                retrieved_context = []
                for expected_term in test['expected_context']:
                    if expected_term.lower() in response_lower:
                        retrieved_context.append(expected_term)
                
                retrieval_accuracy = len(retrieved_context) / len(test['expected_context'])
                
                retrieval_results.append({
                    'query': test['query'],
                    'response': response,
                    'expected_context': test['expected_context'],
                    'retrieved_context': retrieved_context,
                    'accuracy': retrieval_accuracy,
                    'source_document': test['source_document']
                })
            
            # Validate context retrieval
            for result in retrieval_results:
                # Should retrieve at least 50% of expected context
                assert result['accuracy'] >= 0.5, \
                    f"Low context retrieval accuracy: {result['accuracy']:.1%} for '{result['query']}'"
                
                # Should contain relevant biomedical terminology
                assert len(result['retrieved_context']) >= 2, \
                    f"Insufficient context retrieved: {result['retrieved_context']}"
            
            # Overall retrieval performance
            avg_accuracy = statistics.mean([r['accuracy'] for r in retrieval_results])
            assert avg_accuracy >= 0.6, f"Overall context retrieval accuracy too low: {avg_accuracy:.1%}"
            
            logging.info(f"✅ Context Retrieval Validation:")
            for result in retrieval_results:
                logging.info(f"  - Query: {result['query'][:50]}...")
                logging.info(f"    Accuracy: {result['accuracy']:.1%}")
                logging.info(f"    Retrieved: {result['retrieved_context']}")

    @pytest.mark.asyncio
    async def test_context_relevance_scoring(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test relevance scoring of retrieved context."""
        
        # Create content with varying relevance levels
        diverse_content = {
            "highly_relevant.pdf": """
                Clinical metabolomics is essential for personalized medicine.
                Biomarker discovery enables early disease detection.
                LC-MS analysis provides high sensitivity and specificity.
                Statistical validation ensures clinical applicability.
            """,
            "moderately_relevant.pdf": """
                Analytical chemistry supports biomedical research.
                Laboratory techniques require proper standardization.
                Data processing involves computational methods.
                Research findings need peer review validation.
            """,
            "less_relevant.pdf": """
                Scientific instruments require regular maintenance.
                Laboratory safety protocols are important.
                Equipment calibration ensures accuracy.
                Training is essential for proper operation.
            """
        }
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize knowledge base
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            
            for filename, content in diverse_content.items():
                (papers_dir / filename).write_text(content)
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test queries with different relevance expectations
            relevance_tests = [
                {
                    'query': "What is clinical metabolomics?",
                    'expected_high_relevance': ['clinical metabolomics', 'personalized medicine', 'biomarker discovery'],
                    'expected_low_relevance': ['maintenance', 'calibration', 'training']
                },
                {
                    'query': "What analytical methods are used in biomedical research?",
                    'expected_high_relevance': ['lc-ms', 'analytical chemistry', 'sensitivity'],
                    'expected_low_relevance': ['safety protocols', 'equipment', 'operation']
                }
            ]
            
            relevance_results = []
            for test in relevance_tests:
                response = await mock_enhanced_rag_system.query(test['query'])
                response_lower = response.lower()
                
                # Check for high relevance terms
                high_relevance_count = sum(
                    1 for term in test['expected_high_relevance'] 
                    if term.lower() in response_lower
                )
                
                # Check for low relevance terms (should be fewer)
                low_relevance_count = sum(
                    1 for term in test['expected_low_relevance'] 
                    if term.lower() in response_lower
                )
                
                relevance_score = high_relevance_count / len(test['expected_high_relevance'])
                irrelevance_score = low_relevance_count / len(test['expected_low_relevance'])
                
                relevance_results.append({
                    'query': test['query'],
                    'response': response,
                    'relevance_score': relevance_score,
                    'irrelevance_score': irrelevance_score,
                    'high_relevance_count': high_relevance_count,
                    'low_relevance_count': low_relevance_count
                })
            
            # Validate relevance scoring
            for result in relevance_results:
                # Should retrieve more high-relevance than low-relevance content
                assert result['high_relevance_count'] >= result['low_relevance_count'], \
                    f"Retrieved more irrelevant than relevant content for '{result['query']}'"
                
                # Should have reasonable relevance score
                assert result['relevance_score'] >= 0.4, \
                    f"Low relevance score: {result['relevance_score']:.1%}"
                
                # Should minimize irrelevant content
                assert result['irrelevance_score'] <= 0.5, \
                    f"High irrelevance score: {result['irrelevance_score']:.1%}"
            
            logging.info(f"✅ Context Relevance Scoring:")
            for result in relevance_results:
                logging.info(f"  - Query: {result['query'][:50]}...")
                logging.info(f"    Relevance: {result['relevance_score']:.1%}, Irrelevance: {result['irrelevance_score']:.1%}")


# =====================================================================
# ERROR SCENARIO HANDLING TESTS
# =====================================================================

@pytest.mark.biomedical
@pytest.mark.integration
class TestErrorScenarioHandling:
    """Test edge cases and error handling during query processing."""

    @pytest.mark.asyncio
    async def test_empty_knowledge_base_handling(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system
    ):
        """Test query handling with empty or minimal knowledge base."""
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_instance = MagicMock()
            mock_instance.aquery = AsyncMock(return_value="I don't have sufficient information to answer this query.")
            mock_lightrag.return_value = mock_instance
            
            # Test queries on empty knowledge base
            test_queries = [
                "What is clinical metabolomics?",
                "Compare different analytical methods",
                "Synthesize findings across studies"
            ]
            
            for query in test_queries:
                try:
                    response = await mock_enhanced_rag_system.query(query)
                    
                    # Should handle gracefully, not crash
                    assert isinstance(response, str), "Should return string response"
                    assert len(response) > 0, "Should not return empty response"
                    
                    # Should indicate limited information
                    response_lower = response.lower()
                    limitation_indicators = [
                        'insufficient', 'limited', 'no information', 'unable to', 'don\'t have'
                    ]
                    
                    has_limitation_indicator = any(
                        indicator in response_lower for indicator in limitation_indicators
                    )
                    
                    # For empty KB, should either provide general info or indicate limitations
                    assert len(response) >= 50 or has_limitation_indicator, \
                        "Should provide meaningful response or indicate limitations"
                    
                except Exception as e:
                    pytest.fail(f"Query crashed on empty KB: {query} - {e}")
            
            logging.info("✅ Empty knowledge base handling: All queries handled gracefully")

    @pytest.mark.asyncio
    async def test_malformed_query_handling(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test handling of malformed and edge case queries."""
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize with basic content
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            (papers_dir / "basic.pdf").write_text("Basic biomedical research content")
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test edge case queries
            edge_case_queries = [
                "",  # Empty query
                "   ",  # Whitespace only
                "a",  # Single character
                "?" * 100,  # Repetitive characters
                "What is " + "very " * 50 + "long query?",  # Extremely long query
                "Wh@t !s cl!n!c@l m3t@b0l0m!cs?",  # Special characters
                "What is quantum metabolomics artificial intelligence?",  # Non-existent concepts
                "Compare 1990 studies with 2025 studies",  # Out of range timeframe
            ]
            
            edge_case_results = []
            for query in edge_case_queries:
                try:
                    start_time = time.time()
                    response = await mock_enhanced_rag_system.query(query)
                    execution_time = time.time() - start_time
                    
                    edge_case_results.append({
                        'query': query[:50] + "..." if len(query) > 50 else query,
                        'response': response,
                        'execution_time': execution_time,
                        'success': True,
                        'error': None
                    })
                    
                except Exception as e:
                    edge_case_results.append({
                        'query': query[:50] + "..." if len(query) > 50 else query,
                        'response': "",
                        'execution_time': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            # Validate edge case handling
            successful_queries = [r for r in edge_case_results if r['success']]
            
            # Should handle most edge cases gracefully
            success_rate = len(successful_queries) / len(edge_case_results)
            assert success_rate >= 0.7, f"Edge case success rate too low: {success_rate:.1%}"
            
            # Successful queries should complete in reasonable time
            for result in successful_queries:
                assert result['execution_time'] < 30.0, \
                    f"Edge case query too slow: {result['execution_time']:.1f}s"
                
                # Should provide some response (even if indicating limitations)
                assert len(result['response']) >= 20, \
                    f"Edge case response too brief: '{result['response']}'"
            
            logging.info(f"✅ Malformed Query Handling:")
            logging.info(f"  - Success rate: {success_rate:.1%}")
            logging.info(f"  - Queries tested: {len(edge_case_queries)}")
            for result in edge_case_results:
                status = "✅" if result['success'] else "❌"
                logging.info(f"    {status} {result['query']}")

    @pytest.mark.asyncio
    async def test_system_overload_handling(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test system behavior under query overload conditions."""
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize knowledge base
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            (papers_dir / "content.pdf").write_text("Biomedical research content for testing")
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Simulate overload with concurrent queries
            overload_queries = [
                f"What is biomarker {i}?" for i in range(20)
            ]
            
            async def execute_query_with_timeout(query, timeout=15):
                """Execute query with timeout to prevent hanging."""
                try:
                    return await asyncio.wait_for(
                        mock_enhanced_rag_system.query(query), 
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    return "TIMEOUT_ERROR"
            
            # Execute concurrent queries
            start_time = time.time()
            concurrent_results = await asyncio.gather(
                *[execute_query_with_timeout(query) for query in overload_queries],
                return_exceptions=True
            )
            total_time = time.time() - start_time
            
            # Analyze overload handling
            successful_responses = []
            timeout_responses = []
            error_responses = []
            
            for i, result in enumerate(concurrent_results):
                if isinstance(result, Exception):
                    error_responses.append((overload_queries[i], str(result)))
                elif result == "TIMEOUT_ERROR":
                    timeout_responses.append(overload_queries[i])
                else:
                    successful_responses.append((overload_queries[i], result))
            
            # Validate overload handling
            success_rate = len(successful_responses) / len(overload_queries)
            timeout_rate = len(timeout_responses) / len(overload_queries)
            error_rate = len(error_responses) / len(overload_queries)
            
            # Should handle most queries successfully or timeout gracefully
            assert success_rate + timeout_rate >= 0.8, \
                f"Poor overload handling: {success_rate:.1%} success, {timeout_rate:.1%} timeout"
            
            # Should not have excessive errors (crashes)
            assert error_rate <= 0.2, f"Too many errors under overload: {error_rate:.1%}"
            
            # Total processing should complete within reasonable time
            assert total_time < 120.0, f"Overload processing too slow: {total_time:.1f}s"
            
            # Successful responses should be meaningful
            for query, response in successful_responses[:5]:  # Check first 5
                assert len(response) >= 30, f"Overload response too brief: {len(response)} chars"
            
            logging.info(f"✅ System Overload Handling:")
            logging.info(f"  - Success rate: {success_rate:.1%}")
            logging.info(f"  - Timeout rate: {timeout_rate:.1%}")
            logging.info(f"  - Error rate: {error_rate:.1%}")
            logging.info(f"  - Total processing time: {total_time:.1f}s")
            logging.info(f"  - Average time per query: {total_time/len(overload_queries):.2f}s")


# =====================================================================
# BIOMEDICAL ACCURACY VALIDATION TESTS
# =====================================================================

@pytest.mark.biomedical
@pytest.mark.integration
class TestBiomedicalAccuracyValidation:
    """Test domain-specific accuracy and relevance validation."""

    @pytest.mark.asyncio
    async def test_biomedical_terminology_accuracy(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test accuracy of biomedical terminology in responses."""
        
        # Create content with specific biomedical terminology
        biomedical_content = {
            "metabolomics_terminology.pdf": """
                LC-MS/MS (Liquid Chromatography-Tandem Mass Spectrometry) is used for metabolite analysis.
                HILIC (Hydrophilic Interaction Liquid Chromatography) separates polar metabolites.
                ESI (Electrospray Ionization) is the ionization method for LC-MS.
                MRM (Multiple Reaction Monitoring) enables targeted quantification.
                OPLS-DA (Orthogonal Partial Least Squares-Discriminant Analysis) is used for classification.
            """,
            "clinical_terminology.pdf": """
                HbA1c (Hemoglobin A1c) indicates long-term glucose control.
                eGFR (estimated Glomerular Filtration Rate) assesses kidney function.
                CRP (C-Reactive Protein) is an inflammatory biomarker.
                HDL (High-Density Lipoprotein) is protective cholesterol.
                BMI (Body Mass Index) measures obesity status.
            """
        }
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize knowledge base
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            
            for filename, content in biomedical_content.items():
                (papers_dir / filename).write_text(content)
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test terminology-specific queries
            terminology_tests = [
                {
                    'query': "What is LC-MS/MS used for?",
                    'expected_terms': ['liquid chromatography', 'tandem mass spectrometry', 'metabolite', 'analysis'],
                    'avoid_terms': ['incorrect', 'wrong', 'mistake']
                },
                {
                    'query': "What does HbA1c measure?",
                    'expected_terms': ['hemoglobin a1c', 'glucose control', 'long-term'],
                    'avoid_terms': ['blood pressure', 'cholesterol', 'protein']
                },
                {
                    'query': "What is OPLS-DA?",
                    'expected_terms': ['orthogonal partial least squares', 'discriminant analysis', 'classification'],
                    'avoid_terms': ['correlation', 'regression', 'clustering']
                }
            ]
            
            terminology_results = []
            for test in terminology_tests:
                response = await mock_enhanced_rag_system.query(test['query'])
                response_lower = response.lower()
                
                # Check for expected terminology
                expected_found = [term for term in test['expected_terms'] if term in response_lower]
                expected_accuracy = len(expected_found) / len(test['expected_terms'])
                
                # Check for terms to avoid (incorrect terminology)
                avoid_found = [term for term in test['avoid_terms'] if term in response_lower]
                avoid_rate = len(avoid_found) / len(test['avoid_terms'])
                
                terminology_results.append({
                    'query': test['query'],
                    'response': response,
                    'expected_found': expected_found,
                    'expected_accuracy': expected_accuracy,
                    'avoid_found': avoid_found,
                    'avoid_rate': avoid_rate
                })
            
            # Validate terminology accuracy
            for result in terminology_results:
                # Should include expected biomedical terms
                assert result['expected_accuracy'] >= 0.5, \
                    f"Low terminology accuracy: {result['expected_accuracy']:.1%} for '{result['query']}'"
                
                # Should avoid incorrect terminology
                assert result['avoid_rate'] <= 0.3, \
                    f"High incorrect terminology rate: {result['avoid_rate']:.1%} for '{result['query']}'"
                
                # Should use proper scientific language
                response_lower = result['response'].lower()
                scientific_indicators = ['analysis', 'measurement', 'technique', 'method', 'study']
                scientific_count = sum(1 for indicator in scientific_indicators if indicator in response_lower)
                assert scientific_count >= 2, "Should use appropriate scientific language"
            
            # Overall terminology accuracy
            avg_accuracy = statistics.mean([r['expected_accuracy'] for r in terminology_results])
            avg_avoid_rate = statistics.mean([r['avoid_rate'] for r in terminology_results])
            
            assert avg_accuracy >= 0.6, f"Overall terminology accuracy too low: {avg_accuracy:.1%}"
            assert avg_avoid_rate <= 0.2, f"Overall incorrect terminology rate too high: {avg_avoid_rate:.1%}"
            
            logging.info(f"✅ Biomedical Terminology Accuracy:")
            logging.info(f"  - Average accuracy: {avg_accuracy:.1%}")
            logging.info(f"  - Average incorrect rate: {avg_avoid_rate:.1%}")
            for result in terminology_results:
                logging.info(f"  - {result['query'][:40]}...: {result['expected_accuracy']:.1%}")

    @pytest.mark.asyncio
    async def test_clinical_context_appropriateness(
        self,
        temp_dir,
        mock_config,
        mock_enhanced_rag_system,
        mock_comprehensive_pdf_processor
    ):
        """Test appropriateness of responses in clinical context."""
        
        clinical_content = {
            "clinical_guidelines.pdf": """
                Clinical metabolomics requires standardized protocols for sample collection.
                Patient consent and ethical approval are mandatory for biomarker studies.
                Quality assurance includes internal standards and batch processing controls.
                Clinical validation requires independent cohorts and proper statistical analysis.
                Regulatory compliance follows FDA and EMA guidelines for biomarker qualification.
            """,
            "diagnostic_applications.pdf": """
                Diagnostic biomarkers must demonstrate clinical sensitivity and specificity.
                Reference ranges should be established in healthy populations.
                Analytical validation includes precision, accuracy, and stability testing.
                Clinical utility requires evidence of improved patient outcomes.
                Cost-effectiveness analysis supports clinical implementation decisions.
            """
        }
        
        mock_config.working_dir = temp_dir / "kb"
        
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
             patch('lightrag_integration.clinical_metabolomics_rag.BiomedicalPDFProcessor') as mock_pdf_class:
            
            mock_instance = MagicMock()
            mock_instance.ainsert = AsyncMock(return_value={'success': True})
            mock_instance.aquery = AsyncMock(side_effect=mock_enhanced_rag_system.query)
            mock_lightrag.return_value = mock_instance
            mock_pdf_class.return_value = mock_comprehensive_pdf_processor
            
            # Initialize knowledge base
            papers_dir = temp_dir / "papers"
            papers_dir.mkdir()
            
            for filename, content in clinical_content.items():
                (papers_dir / filename).write_text(content)
            
            await mock_enhanced_rag_system.initialize_knowledge_base(papers_dir=papers_dir)
            
            # Test clinical appropriateness
            clinical_tests = [
                {
                    'query': "What are the requirements for clinical biomarker validation?",
                    'appropriate_concepts': ['validation', 'clinical', 'sensitivity', 'specificity', 'cohorts'],
                    'inappropriate_concepts': ['speculation', 'unproven', 'experimental only']
                },
                {
                    'query': "How should metabolomics be implemented in clinical practice?",
                    'appropriate_concepts': ['standardized', 'protocols', 'quality', 'regulatory', 'guidelines'],
                    'inappropriate_concepts': ['research only', 'preliminary', 'hypothesis']
                }
            ]
            
            clinical_results = []
            for test in clinical_tests:
                response = await mock_enhanced_rag_system.query(test['query'])
                response_lower = response.lower()
                
                # Check for appropriate clinical concepts
                appropriate_found = [
                    concept for concept in test['appropriate_concepts'] 
                    if concept in response_lower
                ]
                appropriate_score = len(appropriate_found) / len(test['appropriate_concepts'])
                
                # Check for inappropriate concepts
                inappropriate_found = [
                    concept for concept in test['inappropriate_concepts'] 
                    if concept in response_lower
                ]
                inappropriate_score = len(inappropriate_found) / len(test['inappropriate_concepts'])
                
                # Check for clinical responsibility indicators
                responsibility_indicators = [
                    'validation', 'approved', 'standardized', 'guidelines', 'evidence'
                ]
                responsibility_count = sum(
                    1 for indicator in responsibility_indicators 
                    if indicator in response_lower
                )
                
                clinical_results.append({
                    'query': test['query'],
                    'response': response,
                    'appropriate_score': appropriate_score,
                    'inappropriate_score': inappropriate_score,
                    'responsibility_count': responsibility_count
                })
            
            # Validate clinical appropriateness
            for result in clinical_results:
                # Should include appropriate clinical concepts
                assert result['appropriate_score'] >= 0.4, \
                    f"Low clinical appropriateness: {result['appropriate_score']:.1%}"
                
                # Should minimize inappropriate concepts
                assert result['inappropriate_score'] <= 0.3, \
                    f"High inappropriate content: {result['inappropriate_score']:.1%}"
                
                # Should demonstrate clinical responsibility
                assert result['responsibility_count'] >= 2, \
                    f"Insufficient clinical responsibility indicators: {result['responsibility_count']}"
            
            logging.info(f"✅ Clinical Context Appropriateness:")
            for result in clinical_results:
                logging.info(f"  - Query: {result['query'][:40]}...")
                logging.info(f"    Appropriateness: {result['appropriate_score']:.1%}")
                logging.info(f"    Responsibility indicators: {result['responsibility_count']}")


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])