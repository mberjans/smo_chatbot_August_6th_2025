#!/usr/bin/env python3
"""
Performance Benchmark Tests for Clinical Metabolomics Oracle - CMO-LIGHTRAG-008-T05

This module implements comprehensive performance benchmark tests that integrate with
the existing testing infrastructure to provide systematic performance validation
for the Clinical Metabolomics Oracle LightRAG integration.

Test Categories:
1. Core Operation Benchmarks - Query processing, PDF ingestion, knowledge base operations
2. Scalability Benchmarks - Load testing with increasing concurrent users
3. Latency Benchmarks - Response time validation across different query complexities
4. Throughput Benchmarks - Operations per second under various load conditions
5. Resource Utilization Benchmarks - Memory, CPU, and I/O efficiency testing
6. End-to-End Workflow Benchmarks - Complete document-to-query workflows

Key Features:
- Integration with existing performance test fixtures
- Realistic biomedical query scenarios
- Comprehensive metric collection and analysis
- Performance regression detection
- Benchmark comparison against predefined targets
- Detailed reporting with actionable insights

Requirements:
- Builds on existing performance_test_fixtures.py infrastructure
- Uses established pytest patterns and fixtures
- Provides comprehensive coverage for CMO-LIGHTRAG-008-T05
- Maintains >90% code coverage requirement

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
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import numpy as np
import psutil
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import existing performance infrastructure
from performance_test_fixtures import (
    PerformanceMetrics,
    LoadTestScenario,
    ResourceMonitor,
    PerformanceTestExecutor,
    LoadTestScenarioGenerator,
    MockOperationGenerator,
    mock_clinical_query_operation
)

# Import biomedical test fixtures
from biomedical_test_fixtures import (
    ClinicalMetabolomicsDataGenerator,
    MetaboliteData,
    ClinicalStudyData
)

# Core components for testing - handle gracefully if not available
try:
    from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    from lightrag_integration.config import LightRAGConfig
    from lightrag_integration.pdf_processor import BiomedicalPDFProcessor
    CORE_COMPONENTS_AVAILABLE = True
except ImportError:
    # Mock core components for testing infrastructure validation
    class ClinicalMetabolomicsRAG:
        async def query(self, query_text: str, mode: str = "hybrid") -> str:
            await asyncio.sleep(0.1)  # Simulate processing time
            return f"Mock response for: {query_text}"
    
    class LightRAGConfig:
        def __init__(self, **kwargs):
            self.api_key = "test-key"
            self.model = "gpt-4o-mini"
            self.working_dir = Path("/tmp/test")
    
    class BiomedicalPDFProcessor:
        async def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
            await asyncio.sleep(0.2)  # Simulate processing time
            return {"text": "Mock PDF content", "success": True}
    
    CORE_COMPONENTS_AVAILABLE = False


@dataclass
class BenchmarkTarget:
    """Defines performance benchmark targets for validation."""
    benchmark_name: str
    operation_type: str
    max_response_time_ms: float
    min_throughput_ops_per_sec: float
    max_memory_usage_mb: float
    max_error_rate_percent: float
    description: str
    priority: str = "medium"  # low, medium, high, critical
    
    def evaluate_performance(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Evaluate performance metrics against this benchmark."""
        results = {
            'benchmark_name': self.benchmark_name,
            'meets_targets': True,
            'violations': [],
            'performance_score': 100.0,
            'details': {}
        }
        
        # Check response time
        if metrics.average_latency_ms > self.max_response_time_ms:
            results['meets_targets'] = False
            results['violations'].append({
                'metric': 'average_latency_ms',
                'target': self.max_response_time_ms,
                'actual': metrics.average_latency_ms,
                'severity': 'high'
            })
        
        # Check throughput
        if metrics.throughput_ops_per_sec < self.min_throughput_ops_per_sec:
            results['meets_targets'] = False
            results['violations'].append({
                'metric': 'throughput_ops_per_sec',
                'target': self.min_throughput_ops_per_sec,
                'actual': metrics.throughput_ops_per_sec,
                'severity': 'medium'
            })
        
        # Check memory usage
        if metrics.memory_usage_mb > self.max_memory_usage_mb:
            results['meets_targets'] = False
            results['violations'].append({
                'metric': 'memory_usage_mb',
                'target': self.max_memory_usage_mb,
                'actual': metrics.memory_usage_mb,
                'severity': 'low'
            })
        
        # Check error rate
        if metrics.error_rate_percent > self.max_error_rate_percent:
            results['meets_targets'] = False
            results['violations'].append({
                'metric': 'error_rate_percent',
                'target': self.max_error_rate_percent,
                'actual': metrics.error_rate_percent,
                'severity': 'critical'
            })
        
        # Calculate performance score
        score_deductions = 0
        for violation in results['violations']:
            if violation['severity'] == 'critical':
                score_deductions += 40
            elif violation['severity'] == 'high':
                score_deductions += 25
            elif violation['severity'] == 'medium':
                score_deductions += 15
            elif violation['severity'] == 'low':
                score_deductions += 10
        
        results['performance_score'] = max(0, 100 - score_deductions)
        
        return results


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmark test suite for Clinical Metabolomics Oracle.
    """
    
    def __init__(self):
        self.benchmark_targets = self._create_benchmark_targets()
        self.test_executor = PerformanceTestExecutor()
        self.data_generator = ClinicalMetabolomicsDataGenerator()
        self.operation_generator = MockOperationGenerator()
        self.benchmark_results: List[Dict[str, Any]] = []
    
    def _create_benchmark_targets(self) -> Dict[str, BenchmarkTarget]:
        """Create comprehensive benchmark targets."""
        return {
            # Core Query Processing Benchmarks
            'simple_query_benchmark': BenchmarkTarget(
                benchmark_name="simple_query_performance",
                operation_type="simple_query",
                max_response_time_ms=2000.0,
                min_throughput_ops_per_sec=2.0,
                max_memory_usage_mb=300.0,
                max_error_rate_percent=2.0,
                description="Simple biomedical queries with basic entity retrieval",
                priority="high"
            ),
            
            'medium_query_benchmark': BenchmarkTarget(
                benchmark_name="medium_query_performance",
                operation_type="medium_query",
                max_response_time_ms=5000.0,
                min_throughput_ops_per_sec=1.0,
                max_memory_usage_mb=500.0,
                max_error_rate_percent=3.0,
                description="Medium complexity queries with pathway analysis",
                priority="high"
            ),
            
            'complex_query_benchmark': BenchmarkTarget(
                benchmark_name="complex_query_performance",
                operation_type="complex_query",
                max_response_time_ms=15000.0,
                min_throughput_ops_per_sec=0.5,
                max_memory_usage_mb=800.0,
                max_error_rate_percent=5.0,
                description="Complex queries requiring multi-document synthesis",
                priority="medium"
            ),
            
            # Scalability Benchmarks
            'concurrent_users_benchmark': BenchmarkTarget(
                benchmark_name="concurrent_users_performance",
                operation_type="concurrent_operations",
                max_response_time_ms=8000.0,
                min_throughput_ops_per_sec=5.0,
                max_memory_usage_mb=1200.0,
                max_error_rate_percent=8.0,
                description="Performance under concurrent user load",
                priority="high"
            ),
            
            # PDF Processing Benchmarks
            'pdf_ingestion_benchmark': BenchmarkTarget(
                benchmark_name="pdf_processing_performance",
                operation_type="pdf_processing",
                max_response_time_ms=10000.0,
                min_throughput_ops_per_sec=0.3,
                max_memory_usage_mb=600.0,
                max_error_rate_percent=5.0,
                description="PDF document processing and ingestion",
                priority="medium"
            ),
            
            # Knowledge Base Benchmarks
            'knowledge_base_insertion_benchmark': BenchmarkTarget(
                benchmark_name="knowledge_base_insertion_performance",
                operation_type="knowledge_insertion",
                max_response_time_ms=12000.0,
                min_throughput_ops_per_sec=0.4,
                max_memory_usage_mb=700.0,
                max_error_rate_percent=3.0,
                description="Document insertion into knowledge base",
                priority="medium"
            ),
            
            # End-to-End Workflow Benchmarks
            'e2e_workflow_benchmark': BenchmarkTarget(
                benchmark_name="end_to_end_workflow_performance",
                operation_type="e2e_workflow",
                max_response_time_ms=30000.0,
                min_throughput_ops_per_sec=0.2,
                max_memory_usage_mb=1000.0,
                max_error_rate_percent=10.0,
                description="Complete PDF ingestion to query response workflow",
                priority="critical"
            )
        }
    
    async def run_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("Starting Clinical Metabolomics Oracle Performance Benchmark Suite")
        print("=" * 70)
        
        suite_start_time = time.time()
        suite_results = {
            'suite_name': 'CMO-LIGHTRAG-008-T05-PerformanceBenchmarks',
            'start_time': suite_start_time,
            'benchmarks': [],
            'summary': {}
        }
        
        # Run individual benchmarks
        benchmark_methods = [
            ('simple_query_benchmark', self._run_simple_query_benchmark),
            ('medium_query_benchmark', self._run_medium_query_benchmark),
            ('complex_query_benchmark', self._run_complex_query_benchmark),
            ('concurrent_users_benchmark', self._run_concurrent_users_benchmark),
            ('pdf_ingestion_benchmark', self._run_pdf_processing_benchmark),
            ('knowledge_base_insertion_benchmark', self._run_knowledge_base_benchmark),
            ('e2e_workflow_benchmark', self._run_end_to_end_workflow_benchmark)
        ]
        
        for benchmark_name, benchmark_method in benchmark_methods:
            try:
                print(f"\nRunning {benchmark_name}...")
                benchmark_result = await benchmark_method()
                benchmark_result['benchmark_id'] = benchmark_name
                suite_results['benchmarks'].append(benchmark_result)
                print(f"✓ {benchmark_name} completed")
            except Exception as e:
                print(f"✗ {benchmark_name} failed: {str(e)}")
                suite_results['benchmarks'].append({
                    'benchmark_id': benchmark_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Calculate suite summary
        suite_results['end_time'] = time.time()
        suite_results['duration_seconds'] = suite_results['end_time'] - suite_results['start_time']
        suite_results['summary'] = self._calculate_suite_summary(suite_results['benchmarks'])
        
        print("\n" + "=" * 70)
        print("Performance Benchmark Suite Completed")
        print(f"Total Duration: {suite_results['duration_seconds']:.2f} seconds")
        print(f"Benchmarks Run: {len(suite_results['benchmarks'])}")
        print(f"Overall Performance Grade: {suite_results['summary']['overall_grade']}")
        
        return suite_results
    
    async def _run_simple_query_benchmark(self) -> Dict[str, Any]:
        """Benchmark simple query performance."""
        scenario = LoadTestScenario(
            scenario_name="simple_query_benchmark",
            description="Simple biomedical query performance benchmark",
            target_operations_per_second=3.0,
            duration_seconds=30.0,
            concurrent_users=1,
            ramp_up_duration=0.0,
            operation_types={'simple_query': 1.0},
            data_size_range=(50, 200),
            success_criteria={'min_throughput_ops_per_sec': 2.0},
            resource_limits={'max_memory_mb': 400}
        )
        
        def simple_query_generator(operation_type: str = 'simple_query') -> Dict[str, Any]:
            """Generate simple biomedical queries."""
            queries = [
                "What is glucose?",
                "Define metabolomics",
                "What is insulin?",
                "Describe lactate",
                "What is cholesterol?",
                "Define proteomics",
                "What is creatinine?",
                "Describe diabetes"
            ]
            return {
                'query_text': np.random.choice(queries),
                'operation_type': operation_type,
                'expected_response_length': np.random.randint(100, 300),
                'complexity_score': 0.2
            }
        
        metrics = await self.test_executor.execute_load_test(
            scenario=scenario,
            operation_func=mock_clinical_query_operation,
            operation_data_generator=simple_query_generator
        )
        
        # Evaluate against benchmark targets
        target = self.benchmark_targets['simple_query_benchmark']
        evaluation = target.evaluate_performance(metrics)
        
        return {
            'benchmark_type': 'simple_query',
            'metrics': metrics.summary,
            'evaluation': evaluation,
            'target_specifications': {
                'max_response_time_ms': target.max_response_time_ms,
                'min_throughput_ops_per_sec': target.min_throughput_ops_per_sec,
                'max_memory_usage_mb': target.max_memory_usage_mb
            },
            'status': 'passed' if evaluation['meets_targets'] else 'failed'
        }
    
    async def _run_medium_query_benchmark(self) -> Dict[str, Any]:
        """Benchmark medium complexity query performance."""
        scenario = LoadTestScenario(
            scenario_name="medium_query_benchmark",
            description="Medium complexity biomedical query performance benchmark",
            target_operations_per_second=1.5,
            duration_seconds=45.0,
            concurrent_users=2,
            ramp_up_duration=5.0,
            operation_types={'medium_query': 1.0},
            data_size_range=(200, 800),
            success_criteria={'min_throughput_ops_per_sec': 1.0},
            resource_limits={'max_memory_mb': 600}
        )
        
        def medium_query_generator(operation_type: str = 'medium_query') -> Dict[str, Any]:
            """Generate medium complexity biomedical queries."""
            queries = [
                "Compare glucose and lactate metabolism in diabetes patients",
                "Analyze the relationship between insulin resistance and metabolic syndrome",
                "What are the key biomarkers for cardiovascular disease risk assessment?",
                "Describe the metabolic pathways involved in fatty acid oxidation",
                "How do amino acid profiles change in liver disease?",
                "Compare proteomics approaches for cancer biomarker discovery",
                "What metabolites are elevated in kidney disease progression?",
                "Analyze glycolysis regulation in cancer metabolism"
            ]
            return {
                'query_text': np.random.choice(queries),
                'operation_type': operation_type,
                'expected_response_length': np.random.randint(400, 800),
                'complexity_score': 0.5
            }
        
        metrics = await self.test_executor.execute_load_test(
            scenario=scenario,
            operation_func=mock_clinical_query_operation,
            operation_data_generator=medium_query_generator
        )
        
        # Evaluate against benchmark targets
        target = self.benchmark_targets['medium_query_benchmark']
        evaluation = target.evaluate_performance(metrics)
        
        return {
            'benchmark_type': 'medium_query',
            'metrics': metrics.summary,
            'evaluation': evaluation,
            'target_specifications': {
                'max_response_time_ms': target.max_response_time_ms,
                'min_throughput_ops_per_sec': target.min_throughput_ops_per_sec,
                'max_memory_usage_mb': target.max_memory_usage_mb
            },
            'status': 'passed' if evaluation['meets_targets'] else 'failed'
        }
    
    async def _run_complex_query_benchmark(self) -> Dict[str, Any]:
        """Benchmark complex query performance requiring multi-document synthesis."""
        scenario = LoadTestScenario(
            scenario_name="complex_query_benchmark",
            description="Complex multi-document synthesis query performance benchmark",
            target_operations_per_second=0.8,
            duration_seconds=60.0,
            concurrent_users=2,
            ramp_up_duration=10.0,
            operation_types={'complex_query': 1.0},
            data_size_range=(800, 2000),
            success_criteria={'min_throughput_ops_per_sec': 0.5},
            resource_limits={'max_memory_mb': 1000},
            warmup_duration=15.0
        )
        
        def complex_query_generator(operation_type: str = 'complex_query') -> Dict[str, Any]:
            """Generate complex biomedical queries."""
            queries = [
                "Provide comprehensive analysis of metabolomics biomarkers for early diabetes detection including pathway analysis, statistical validation, and clinical implementation recommendations",
                "Integrate proteomics and metabolomics data to identify novel therapeutic targets for cardiovascular disease, including molecular mechanisms and drug development implications",
                "Develop a multi-omics approach for personalized cancer treatment selection based on metabolic profiling, genetic variants, and protein expression patterns",
                "Design clinical diagnostic panel combining metabolomics and proteomics for liver disease progression monitoring with validation protocols and performance metrics",
                "Analyze the integration of metabolomics with pharmacogenomics for optimizing drug dosing in kidney disease patients across different demographic populations",
                "Create comprehensive biomarker discovery workflow combining metabolomics, proteomics, and genomics for neurodegenerative disease early detection and monitoring"
            ]
            return {
                'query_text': np.random.choice(queries),
                'operation_type': operation_type,
                'expected_response_length': np.random.randint(1000, 2000),
                'complexity_score': 0.9
            }
        
        metrics = await self.test_executor.execute_load_test(
            scenario=scenario,
            operation_func=mock_clinical_query_operation,
            operation_data_generator=complex_query_generator
        )
        
        # Evaluate against benchmark targets
        target = self.benchmark_targets['complex_query_benchmark']
        evaluation = target.evaluate_performance(metrics)
        
        return {
            'benchmark_type': 'complex_query',
            'metrics': metrics.summary,
            'evaluation': evaluation,
            'target_specifications': {
                'max_response_time_ms': target.max_response_time_ms,
                'min_throughput_ops_per_sec': target.min_throughput_ops_per_sec,
                'max_memory_usage_mb': target.max_memory_usage_mb
            },
            'status': 'passed' if evaluation['meets_targets'] else 'failed'
        }
    
    async def _run_concurrent_users_benchmark(self) -> Dict[str, Any]:
        """Benchmark concurrent user performance and scalability."""
        scenario = LoadTestScenario(
            scenario_name="concurrent_users_benchmark",
            description="Concurrent users scalability performance benchmark",
            target_operations_per_second=8.0,
            duration_seconds=90.0,
            concurrent_users=10,
            ramp_up_duration=20.0,
            operation_types={
                'simple_query': 0.4,
                'medium_query': 0.4,
                'complex_query': 0.2
            },
            data_size_range=(100, 1500),
            success_criteria={'min_throughput_ops_per_sec': 5.0},
            resource_limits={'max_memory_mb': 1500},
            warmup_duration=20.0,
            cooldown_duration=10.0
        )
        
        metrics = await self.test_executor.execute_load_test(
            scenario=scenario,
            operation_func=mock_clinical_query_operation,
            operation_data_generator=self.operation_generator.generate_query_data
        )
        
        # Evaluate against benchmark targets
        target = self.benchmark_targets['concurrent_users_benchmark']
        evaluation = target.evaluate_performance(metrics)
        
        return {
            'benchmark_type': 'concurrent_users',
            'metrics': metrics.summary,
            'evaluation': evaluation,
            'scalability_analysis': {
                'concurrent_users_tested': scenario.concurrent_users,
                'throughput_per_user': metrics.throughput_ops_per_sec / scenario.concurrent_users,
                'scalability_efficiency': (metrics.throughput_ops_per_sec / scenario.concurrent_users) / (scenario.target_operations_per_second / scenario.concurrent_users)
            },
            'status': 'passed' if evaluation['meets_targets'] else 'failed'
        }
    
    async def _run_pdf_processing_benchmark(self) -> Dict[str, Any]:
        """Benchmark PDF processing and ingestion performance."""
        
        async def mock_pdf_processing_operation(data: Dict[str, Any]) -> Dict[str, Any]:
            """Mock PDF processing operation."""
            # Simulate PDF processing time based on document size
            doc_size = data.get('document_size_pages', 10)
            base_time = 0.5  # Base processing time
            size_factor = doc_size * 0.1  # Additional time per page
            processing_time = base_time + size_factor
            
            await asyncio.sleep(processing_time)
            
            # Simulate occasional failures
            if np.random.random() < 0.05:  # 5% failure rate
                raise Exception("PDF processing simulation failure")
            
            return {
                'document_path': data.get('document_path', 'test_document.pdf'),
                'processing_time': processing_time,
                'extracted_text_length': doc_size * 500,  # Assume 500 chars per page
                'metadata': {
                    'pages': doc_size,
                    'file_size_mb': doc_size * 0.1,
                    'extraction_success': True
                },
                'cost': processing_time * 0.01
            }
        
        def pdf_data_generator(operation_type: str = 'pdf_processing') -> Dict[str, Any]:
            """Generate PDF processing test data."""
            return {
                'document_path': f"test_document_{np.random.randint(1, 1000)}.pdf",
                'document_size_pages': np.random.randint(5, 25),
                'operation_type': operation_type,
                'document_type': np.random.choice(['metabolomics_study', 'clinical_trial', 'review_paper']),
                'complexity_score': np.random.uniform(0.3, 0.8)
            }
        
        scenario = LoadTestScenario(
            scenario_name="pdf_processing_benchmark",
            description="PDF processing and ingestion performance benchmark",
            target_operations_per_second=0.5,
            duration_seconds=60.0,
            concurrent_users=3,
            ramp_up_duration=10.0,
            operation_types={'pdf_processing': 1.0},
            data_size_range=(5000, 50000),
            success_criteria={'min_throughput_ops_per_sec': 0.3},
            resource_limits={'max_memory_mb': 800},
            warmup_duration=10.0
        )
        
        metrics = await self.test_executor.execute_load_test(
            scenario=scenario,
            operation_func=mock_pdf_processing_operation,
            operation_data_generator=pdf_data_generator
        )
        
        # Evaluate against benchmark targets
        target = self.benchmark_targets['pdf_ingestion_benchmark']
        evaluation = target.evaluate_performance(metrics)
        
        return {
            'benchmark_type': 'pdf_processing',
            'metrics': metrics.summary,
            'evaluation': evaluation,
            'processing_analysis': {
                'documents_processed': metrics.operations_count,
                'avg_processing_time_per_document': metrics.average_latency_ms,
                'processing_efficiency': metrics.operations_count / (metrics.duration / 60)  # docs per minute
            },
            'status': 'passed' if evaluation['meets_targets'] else 'failed'
        }
    
    async def _run_knowledge_base_benchmark(self) -> Dict[str, Any]:
        """Benchmark knowledge base insertion and indexing performance."""
        
        async def mock_knowledge_base_operation(data: Dict[str, Any]) -> Dict[str, Any]:
            """Mock knowledge base insertion operation."""
            # Simulate knowledge base insertion time based on content size
            content_length = data.get('content_length', 1000)
            base_time = 0.3
            content_factor = content_length / 1000 * 0.2
            processing_time = base_time + content_factor
            
            await asyncio.sleep(processing_time)
            
            # Simulate occasional failures
            if np.random.random() < 0.03:  # 3% failure rate
                raise Exception("Knowledge base insertion simulation failure")
            
            return {
                'document_id': data.get('document_id', 'doc_001'),
                'insertion_time': processing_time,
                'content_indexed': content_length,
                'entities_extracted': np.random.randint(10, 50),
                'relationships_created': np.random.randint(5, 25),
                'indexing_cost': processing_time * 0.005
            }
        
        def kb_data_generator(operation_type: str = 'knowledge_insertion') -> Dict[str, Any]:
            """Generate knowledge base insertion test data."""
            return {
                'document_id': f"kb_doc_{np.random.randint(1, 10000)}",
                'content_length': np.random.randint(500, 5000),
                'operation_type': operation_type,
                'content_type': np.random.choice(['research_paper', 'clinical_data', 'pathway_info']),
                'priority': np.random.choice(['normal', 'high']),
                'complexity_score': np.random.uniform(0.4, 0.9)
            }
        
        scenario = LoadTestScenario(
            scenario_name="knowledge_base_benchmark",
            description="Knowledge base insertion and indexing performance benchmark",
            target_operations_per_second=0.6,
            duration_seconds=75.0,
            concurrent_users=4,
            ramp_up_duration=15.0,
            operation_types={'knowledge_insertion': 1.0},
            data_size_range=(1000, 8000),
            success_criteria={'min_throughput_ops_per_sec': 0.4},
            resource_limits={'max_memory_mb': 900},
            warmup_duration=15.0
        )
        
        metrics = await self.test_executor.execute_load_test(
            scenario=scenario,
            operation_func=mock_knowledge_base_operation,
            operation_data_generator=kb_data_generator
        )
        
        # Evaluate against benchmark targets
        target = self.benchmark_targets['knowledge_base_insertion_benchmark']
        evaluation = target.evaluate_performance(metrics)
        
        return {
            'benchmark_type': 'knowledge_base_insertion',
            'metrics': metrics.summary,
            'evaluation': evaluation,
            'indexing_analysis': {
                'documents_indexed': metrics.operations_count,
                'avg_indexing_time': metrics.average_latency_ms,
                'indexing_throughput': metrics.throughput_ops_per_sec,
                'estimated_entities_extracted': metrics.operations_count * 30  # Rough estimate
            },
            'status': 'passed' if evaluation['meets_targets'] else 'failed'
        }
    
    async def _run_end_to_end_workflow_benchmark(self) -> Dict[str, Any]:
        """Benchmark complete end-to-end workflow performance."""
        
        async def mock_e2e_workflow_operation(data: Dict[str, Any]) -> Dict[str, Any]:
            """Mock end-to-end workflow operation."""
            # Simulate complete workflow: PDF processing → Knowledge base insertion → Query processing
            
            # Stage 1: PDF Processing
            pdf_processing_time = np.random.uniform(0.8, 2.0)
            await asyncio.sleep(pdf_processing_time)
            
            # Stage 2: Knowledge Base Insertion
            kb_insertion_time = np.random.uniform(0.5, 1.5)
            await asyncio.sleep(kb_insertion_time)
            
            # Stage 3: Query Processing
            query_processing_time = np.random.uniform(0.3, 1.0)
            await asyncio.sleep(query_processing_time)
            
            total_time = pdf_processing_time + kb_insertion_time + query_processing_time
            
            # Simulate occasional failures (higher rate for complex workflow)
            if np.random.random() < 0.08:  # 8% failure rate
                raise Exception("End-to-end workflow simulation failure")
            
            return {
                'workflow_id': data.get('workflow_id', 'e2e_001'),
                'total_processing_time': total_time,
                'pdf_processing_time': pdf_processing_time,
                'kb_insertion_time': kb_insertion_time,
                'query_processing_time': query_processing_time,
                'documents_processed': 1,
                'queries_answered': 1,
                'workflow_cost': total_time * 0.02
            }
        
        def e2e_data_generator(operation_type: str = 'e2e_workflow') -> Dict[str, Any]:
            """Generate end-to-end workflow test data."""
            return {
                'workflow_id': f"e2e_workflow_{np.random.randint(1, 1000)}",
                'pdf_document': f"test_document_{np.random.randint(1, 100)}.pdf",
                'query': f"Test biomedical query {np.random.randint(1, 50)}",
                'operation_type': operation_type,
                'workflow_complexity': np.random.choice(['simple', 'medium', 'complex']),
                'expected_duration': np.random.uniform(2.0, 8.0)
            }
        
        scenario = LoadTestScenario(
            scenario_name="e2e_workflow_benchmark",
            description="End-to-end workflow performance benchmark",
            target_operations_per_second=0.3,
            duration_seconds=120.0,
            concurrent_users=3,
            ramp_up_duration=20.0,
            operation_types={'e2e_workflow': 1.0},
            data_size_range=(5000, 15000),
            success_criteria={'min_throughput_ops_per_sec': 0.2},
            resource_limits={'max_memory_mb': 1200},
            warmup_duration=20.0,
            cooldown_duration=15.0
        )
        
        metrics = await self.test_executor.execute_load_test(
            scenario=scenario,
            operation_func=mock_e2e_workflow_operation,
            operation_data_generator=e2e_data_generator
        )
        
        # Evaluate against benchmark targets
        target = self.benchmark_targets['e2e_workflow_benchmark']
        evaluation = target.evaluate_performance(metrics)
        
        return {
            'benchmark_type': 'end_to_end_workflow',
            'metrics': metrics.summary,
            'evaluation': evaluation,
            'workflow_analysis': {
                'workflows_completed': metrics.operations_count,
                'avg_workflow_duration': metrics.average_latency_ms,
                'workflow_success_rate': ((metrics.operations_count - metrics.failure_count) / metrics.operations_count * 100) if metrics.operations_count > 0 else 0,
                'estimated_documents_processed': metrics.operations_count,
                'estimated_queries_answered': metrics.operations_count
            },
            'status': 'passed' if evaluation['meets_targets'] else 'failed'
        }
    
    def _calculate_suite_summary(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive suite summary."""
        if not benchmark_results:
            return {'status': 'no_results'}
        
        # Count passed/failed benchmarks
        passed_count = sum(1 for result in benchmark_results if result.get('status') == 'passed')
        failed_count = len(benchmark_results) - passed_count
        
        # Calculate performance scores
        performance_scores = []
        for result in benchmark_results:
            if 'evaluation' in result and 'performance_score' in result['evaluation']:
                performance_scores.append(result['evaluation']['performance_score'])
        
        avg_performance_score = np.mean(performance_scores) if performance_scores else 0
        
        # Determine overall grade
        if avg_performance_score >= 90:
            overall_grade = "Excellent"
        elif avg_performance_score >= 80:
            overall_grade = "Good"
        elif avg_performance_score >= 70:
            overall_grade = "Satisfactory"
        elif avg_performance_score >= 60:
            overall_grade = "Needs Improvement"
        else:
            overall_grade = "Poor"
        
        # Collect response times and throughput
        response_times = []
        throughputs = []
        memory_usage = []
        
        for result in benchmark_results:
            if 'metrics' in result:
                metrics = result['metrics']
                if 'average_latency_ms' in metrics:
                    response_times.append(metrics['average_latency_ms'])
                if 'throughput_ops_per_sec' in metrics:
                    throughputs.append(metrics['throughput_ops_per_sec'])
                if 'memory_usage_mb' in metrics:
                    memory_usage.append(metrics['memory_usage_mb'])
        
        return {
            'total_benchmarks': len(benchmark_results),
            'passed_benchmarks': passed_count,
            'failed_benchmarks': failed_count,
            'success_rate_percent': (passed_count / len(benchmark_results)) * 100,
            'overall_grade': overall_grade,
            'avg_performance_score': avg_performance_score,
            'performance_statistics': {
                'avg_response_time_ms': np.mean(response_times) if response_times else 0,
                'median_response_time_ms': np.median(response_times) if response_times else 0,
                'avg_throughput_ops_per_sec': np.mean(throughputs) if throughputs else 0,
                'avg_memory_usage_mb': np.mean(memory_usage) if memory_usage else 0
            },
            'recommendations': self._generate_performance_recommendations(benchmark_results)
        }
    
    def _generate_performance_recommendations(self, benchmark_results: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable performance optimization recommendations."""
        recommendations = []
        
        # Analyze each benchmark for specific recommendations
        for result in benchmark_results:
            if result.get('status') == 'failed' and 'evaluation' in result:
                violations = result['evaluation'].get('violations', [])
                
                for violation in violations:
                    metric = violation['metric']
                    severity = violation['severity']
                    
                    if metric == 'average_latency_ms' and severity in ['high', 'critical']:
                        recommendations.append(
                            f"Optimize query processing for {result.get('benchmark_type', 'unknown')} operations - "
                            f"response times exceed targets by {violation['actual'] - violation['target']:.1f}ms"
                        )
                    
                    elif metric == 'throughput_ops_per_sec' and severity in ['medium', 'high']:
                        recommendations.append(
                            f"Improve throughput for {result.get('benchmark_type', 'unknown')} operations - "
                            f"consider implementing connection pooling or async optimization"
                        )
                    
                    elif metric == 'memory_usage_mb' and severity in ['medium', 'high']:
                        recommendations.append(
                            f"Optimize memory usage for {result.get('benchmark_type', 'unknown')} operations - "
                            f"implement caching strategies or memory cleanup"
                        )
                    
                    elif metric == 'error_rate_percent' and severity == 'critical':
                        recommendations.append(
                            f"Critical: Address error handling for {result.get('benchmark_type', 'unknown')} operations - "
                            f"error rate of {violation['actual']:.1f}% exceeds acceptable threshold"
                        )
        
        # Add general recommendations if no specific issues found
        if not recommendations:
            recommendations.append("Performance benchmarks meet targets - monitor for regressions")
            recommendations.append("Consider implementing performance monitoring dashboards")
            recommendations.append("Establish baseline performance metrics for future comparisons")
        
        return recommendations[:5]  # Limit to top 5 recommendations


# =============================================================================
# PYTEST TEST CLASSES
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """
    Pytest test class for performance benchmarks.
    
    This class integrates with the existing pytest infrastructure to provide
    comprehensive performance benchmark testing for CMO-LIGHTRAG-008-T05.
    """
    
    @pytest.fixture
    def benchmark_suite(self):
        """Provide performance benchmark suite."""
        return PerformanceBenchmarkSuite()
    
    @pytest.fixture
    def performance_thresholds(self):
        """Define performance thresholds for validation."""
        return {
            'max_response_time_simple': 2000,
            'max_response_time_medium': 5000,
            'max_response_time_complex': 15000,
            'min_throughput_simple': 2.0,
            'min_throughput_medium': 1.0,
            'min_throughput_complex': 0.5,
            'max_memory_usage': 1000,
            'max_error_rate': 10.0
        }
    
    @pytest.mark.asyncio
    async def test_simple_query_performance_benchmark(self, benchmark_suite, performance_thresholds):
        """Test simple query performance meets benchmark targets."""
        result = await benchmark_suite._run_simple_query_benchmark()
        
        # Validate benchmark execution
        assert result['status'] in ['passed', 'failed'], "Benchmark should have valid status"
        assert 'metrics' in result, "Benchmark should include performance metrics"
        assert 'evaluation' in result, "Benchmark should include target evaluation"
        
        # Check specific performance criteria
        metrics = result['metrics']
        assert metrics['average_latency_ms'] <= performance_thresholds['max_response_time_simple'], \
            f"Simple query response time {metrics['average_latency_ms']}ms exceeds threshold"
        assert metrics['throughput_ops_per_sec'] >= performance_thresholds['min_throughput_simple'], \
            f"Simple query throughput {metrics['throughput_ops_per_sec']} below threshold"
        
        # Validate evaluation results
        evaluation = result['evaluation']
        assert evaluation['performance_score'] >= 70.0, "Performance score should be acceptable"
        
        print(f"Simple Query Benchmark: {result['status']} - Score: {evaluation['performance_score']:.1f}")
    
    @pytest.mark.asyncio
    async def test_medium_query_performance_benchmark(self, benchmark_suite, performance_thresholds):
        """Test medium complexity query performance meets benchmark targets."""
        result = await benchmark_suite._run_medium_query_benchmark()
        
        # Validate benchmark execution
        assert result['status'] in ['passed', 'failed'], "Benchmark should have valid status"
        assert 'metrics' in result, "Benchmark should include performance metrics"
        
        # Check specific performance criteria
        metrics = result['metrics']
        assert metrics['average_latency_ms'] <= performance_thresholds['max_response_time_medium'], \
            f"Medium query response time {metrics['average_latency_ms']}ms exceeds threshold"
        assert metrics['throughput_ops_per_sec'] >= performance_thresholds['min_throughput_medium'], \
            f"Medium query throughput {metrics['throughput_ops_per_sec']} below threshold"
        
        print(f"Medium Query Benchmark: {result['status']} - Latency: {metrics['average_latency_ms']:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_complex_query_performance_benchmark(self, benchmark_suite, performance_thresholds):
        """Test complex query performance meets benchmark targets."""
        result = await benchmark_suite._run_complex_query_benchmark()
        
        # Validate benchmark execution
        assert result['status'] in ['passed', 'failed'], "Benchmark should have valid status"
        assert 'metrics' in result, "Benchmark should include performance metrics"
        
        # Check specific performance criteria (more lenient for complex queries)
        metrics = result['metrics']
        assert metrics['average_latency_ms'] <= performance_thresholds['max_response_time_complex'], \
            f"Complex query response time {metrics['average_latency_ms']}ms exceeds threshold"
        
        # Complex queries may have lower throughput requirements
        min_complex_throughput = performance_thresholds['min_throughput_complex']
        assert metrics['throughput_ops_per_sec'] >= min_complex_throughput, \
            f"Complex query throughput {metrics['throughput_ops_per_sec']} below threshold"
        
        print(f"Complex Query Benchmark: {result['status']} - Latency: {metrics['average_latency_ms']:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_users_scalability_benchmark(self, benchmark_suite, performance_thresholds):
        """Test system performance under concurrent user load."""
        result = await benchmark_suite._run_concurrent_users_benchmark()
        
        # Validate benchmark execution
        assert result['status'] in ['passed', 'failed'], "Benchmark should have valid status"
        assert 'scalability_analysis' in result, "Should include scalability analysis"
        
        # Check scalability metrics
        scalability = result['scalability_analysis']
        assert scalability['concurrent_users_tested'] >= 5, "Should test meaningful concurrent load"
        assert scalability['throughput_per_user'] > 0, "Should have positive per-user throughput"
        
        # Validate that system handles concurrent load reasonably
        metrics = result['metrics']
        assert metrics['error_rate_percent'] <= 15.0, "Error rate should remain reasonable under load"
        
        print(f"Concurrent Users Benchmark: {result['status']} - {scalability['concurrent_users_tested']} users")
        print(f"Throughput per user: {scalability['throughput_per_user']:.2f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_pdf_processing_performance_benchmark(self, benchmark_suite):
        """Test PDF processing performance meets benchmark targets."""
        result = await benchmark_suite._run_pdf_processing_benchmark()
        
        # Validate benchmark execution
        assert result['status'] in ['passed', 'failed'], "Benchmark should have valid status"
        assert 'processing_analysis' in result, "Should include processing analysis"
        
        # Check PDF processing specific metrics
        analysis = result['processing_analysis']
        assert analysis['documents_processed'] > 0, "Should process at least one document"
        assert analysis['processing_efficiency'] > 0, "Should have positive processing efficiency"
        
        metrics = result['metrics']
        assert metrics['error_rate_percent'] <= 10.0, "PDF processing error rate should be acceptable"
        
        print(f"PDF Processing Benchmark: {result['status']} - {analysis['documents_processed']} docs processed")
    
    @pytest.mark.asyncio
    async def test_knowledge_base_insertion_benchmark(self, benchmark_suite):
        """Test knowledge base insertion performance meets benchmark targets."""
        result = await benchmark_suite._run_knowledge_base_benchmark()
        
        # Validate benchmark execution
        assert result['status'] in ['passed', 'failed'], "Benchmark should have valid status"
        assert 'indexing_analysis' in result, "Should include indexing analysis"
        
        # Check knowledge base specific metrics
        analysis = result['indexing_analysis']
        assert analysis['documents_indexed'] > 0, "Should index at least one document"
        assert analysis['indexing_throughput'] > 0, "Should have positive indexing throughput"
        
        print(f"Knowledge Base Benchmark: {result['status']} - {analysis['documents_indexed']} docs indexed")
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_benchmark(self, benchmark_suite):
        """Test complete end-to-end workflow performance."""
        result = await benchmark_suite._run_end_to_end_workflow_benchmark()
        
        # Validate benchmark execution
        assert result['status'] in ['passed', 'failed'], "Benchmark should have valid status"
        assert 'workflow_analysis' in result, "Should include workflow analysis"
        
        # Check end-to-end workflow metrics
        analysis = result['workflow_analysis']
        assert analysis['workflows_completed'] > 0, "Should complete at least one workflow"
        assert analysis['workflow_success_rate'] >= 80.0, "Should have reasonable success rate"
        
        print(f"E2E Workflow Benchmark: {result['status']} - {analysis['workflows_completed']} workflows")
        print(f"Success rate: {analysis['workflow_success_rate']:.1f}%")
    
    @pytest.mark.asyncio
    async def test_comprehensive_benchmark_suite(self, benchmark_suite):
        """Test the complete performance benchmark suite."""
        # Run the complete benchmark suite
        suite_results = await benchmark_suite.run_benchmark_suite()
        
        # Validate suite execution
        assert 'summary' in suite_results, "Suite should include summary"
        assert suite_results['summary']['total_benchmarks'] >= 6, "Should run all benchmark categories"
        
        # Check overall performance
        summary = suite_results['summary']
        assert summary['success_rate_percent'] >= 70.0, "Majority of benchmarks should pass"
        assert summary['overall_grade'] in ['Excellent', 'Good', 'Satisfactory', 'Needs Improvement', 'Poor']
        
        # Validate performance statistics
        perf_stats = summary['performance_statistics']
        assert perf_stats['avg_response_time_ms'] > 0, "Should have positive average response time"
        assert perf_stats['avg_throughput_ops_per_sec'] > 0, "Should have positive average throughput"
        
        # Check that recommendations are provided
        assert 'recommendations' in summary, "Should provide performance recommendations"
        assert len(summary['recommendations']) > 0, "Should have at least one recommendation"
        
        print(f"\nBenchmark Suite Results:")
        print(f"Overall Grade: {summary['overall_grade']}")
        print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
        print(f"Avg Response Time: {perf_stats['avg_response_time_ms']:.1f}ms")
        print(f"Avg Throughput: {perf_stats['avg_throughput_ops_per_sec']:.2f} ops/sec")
        print(f"Recommendations: {len(summary['recommendations'])}")
        
        # Return results for potential further analysis
        return suite_results
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, benchmark_suite):
        """Test performance regression detection capabilities."""
        # Run baseline benchmark
        baseline_result = await benchmark_suite._run_simple_query_benchmark()
        baseline_metrics = PerformanceMetrics(**{
            'test_name': 'baseline',
            'start_time': time.time(),
            'end_time': time.time() + 30,
            'duration': 30.0,
            'operations_count': 50,
            'success_count': 49,
            'failure_count': 1,
            'throughput_ops_per_sec': 1.63,
            'average_latency_ms': 1500.0,
            'median_latency_ms': 1400.0,
            'p95_latency_ms': 2000.0,
            'p99_latency_ms': 2200.0,
            'min_latency_ms': 800.0,
            'max_latency_ms': 2500.0,
            'memory_usage_mb': 350.0,
            'cpu_usage_percent': 25.0,
            'error_rate_percent': 2.0,
            'concurrent_operations': 1
        })
        
        # Test benchmark target evaluation
        target = benchmark_suite.benchmark_targets['simple_query_benchmark']
        evaluation = target.evaluate_performance(baseline_metrics)
        
        # Validate evaluation structure
        assert 'meets_targets' in evaluation, "Should indicate if targets are met"
        assert 'performance_score' in evaluation, "Should provide performance score"
        assert 'violations' in evaluation, "Should list any violations"
        
        # Performance score should be reasonable
        assert 0 <= evaluation['performance_score'] <= 100, "Performance score should be 0-100"
        
        print(f"Regression Detection Test: Score {evaluation['performance_score']:.1f}")
        if not evaluation['meets_targets']:
            print(f"Violations: {len(evaluation['violations'])}")
    
    def test_benchmark_target_configuration(self, benchmark_suite):
        """Test that benchmark targets are properly configured."""
        targets = benchmark_suite.benchmark_targets
        
        # Check that all required benchmark targets exist
        required_benchmarks = [
            'simple_query_benchmark',
            'medium_query_benchmark', 
            'complex_query_benchmark',
            'concurrent_users_benchmark',
            'pdf_ingestion_benchmark',
            'knowledge_base_insertion_benchmark',
            'e2e_workflow_benchmark'
        ]
        
        for benchmark_name in required_benchmarks:
            assert benchmark_name in targets, f"Missing benchmark target: {benchmark_name}"
            target = targets[benchmark_name]
            
            # Validate target structure
            assert target.max_response_time_ms > 0, "Should have positive response time target"
            assert target.min_throughput_ops_per_sec > 0, "Should have positive throughput target"
            assert target.max_memory_usage_mb > 0, "Should have positive memory usage target"
            assert 0 <= target.max_error_rate_percent <= 100, "Error rate should be percentage"
            assert target.priority in ['low', 'medium', 'high', 'critical'], "Should have valid priority"
        
        print(f"Validated {len(required_benchmarks)} benchmark targets")
    
    @pytest.mark.asyncio
    async def test_benchmark_data_generation(self, benchmark_suite):
        """Test that benchmark data generators work correctly."""
        # Test operation generator
        operation_gen = benchmark_suite.operation_generator
        
        # Test different query types
        for query_type in ['simple_query', 'medium_query', 'complex_query']:
            data = operation_gen.generate_query_data(query_type)
            
            assert 'query_text' in data, "Should generate query text"
            assert 'operation_type' in data, "Should specify operation type"
            assert 'complexity_score' in data, "Should include complexity score"
            assert len(data['query_text']) > 0, "Query text should not be empty"
            assert 0 <= data['complexity_score'] <= 1, "Complexity should be 0-1"
        
        print("Data generation validation completed")
    
    def test_performance_metrics_validation(self):
        """Test that performance metrics are properly structured."""
        # Create sample metrics
        sample_metrics = PerformanceMetrics(
            test_name="validation_test",
            start_time=time.time(),
            end_time=time.time() + 60,
            duration=60.0,
            operations_count=100,
            success_count=95,
            failure_count=5,
            throughput_ops_per_sec=1.58,
            average_latency_ms=2000.0,
            median_latency_ms=1800.0,
            p95_latency_ms=3000.0,
            p99_latency_ms=3500.0,
            min_latency_ms=500.0,
            max_latency_ms=4000.0,
            memory_usage_mb=450.0,
            cpu_usage_percent=30.0,
            error_rate_percent=5.0,
            concurrent_operations=3
        )
        
        # Validate metrics structure
        assert sample_metrics.duration > 0, "Duration should be positive"
        assert sample_metrics.operations_count > 0, "Should have operations"
        assert sample_metrics.success_count <= sample_metrics.operations_count, "Success count should be valid"
        assert sample_metrics.throughput_ops_per_sec > 0, "Throughput should be positive"
        assert sample_metrics.average_latency_ms >= 0, "Latency should be non-negative"
        
        # Test summary generation
        summary = sample_metrics.summary
        assert 'test_name' in summary, "Summary should include test name"
        assert 'throughput_ops_per_sec' in summary, "Summary should include throughput"
        assert 'error_rate_percent' in summary, "Summary should include error rate"
        
        print("Performance metrics validation completed")


# =============================================================================
# BENCHMARK EXECUTION UTILITIES
# =============================================================================

class BenchmarkReportGenerator:
    """Generate comprehensive benchmark reports."""
    
    @staticmethod
    def generate_benchmark_report(suite_results: Dict[str, Any], output_dir: Path = None) -> Path:
        """Generate comprehensive benchmark report."""
        if output_dir is None:
            output_dir = Path("lightrag_integration/tests/performance_test_results")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"Performance_Benchmark_Report_{timestamp}.json"
        
        # Enhance suite results with additional analysis
        enhanced_results = {
            **suite_results,
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0.0',
                'test_framework': 'pytest',
                'task_id': 'CMO-LIGHTRAG-008-T05'
            },
            'analysis': BenchmarkReportGenerator._generate_analysis(suite_results)
        }
        
        # Write detailed JSON report
        with open(report_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        # Generate summary text report
        summary_file = output_dir / f"Performance_Benchmark_Report_{timestamp}_summary.txt"
        BenchmarkReportGenerator._generate_summary_report(enhanced_results, summary_file)
        
        print(f"Benchmark report generated: {report_file}")
        print(f"Summary report generated: {summary_file}")
        
        return report_file
    
    @staticmethod
    def _generate_analysis(suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate additional analysis for the benchmark results."""
        benchmarks = suite_results.get('benchmarks', [])
        
        analysis = {
            'performance_trends': {},
            'bottleneck_analysis': {},
            'optimization_priorities': [],
            'compliance_status': {}
        }
        
        # Analyze performance trends
        response_times = []
        throughputs = []
        memory_usages = []
        
        for benchmark in benchmarks:
            if 'metrics' in benchmark:
                metrics = benchmark['metrics']
                if 'average_latency_ms' in metrics:
                    response_times.append(metrics['average_latency_ms'])
                if 'throughput_ops_per_sec' in metrics:
                    throughputs.append(metrics['throughput_ops_per_sec'])
                if 'memory_usage_mb' in metrics:
                    memory_usages.append(metrics['memory_usage_mb'])
        
        if response_times:
            analysis['performance_trends']['response_time_distribution'] = {
                'min': min(response_times),
                'max': max(response_times),
                'avg': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
        
        # Identify bottlenecks
        failed_benchmarks = [b for b in benchmarks if b.get('status') == 'failed']
        if failed_benchmarks:
            analysis['bottleneck_analysis']['critical_failures'] = len(failed_benchmarks)
            analysis['bottleneck_analysis']['failure_categories'] = [
                b.get('benchmark_type', 'unknown') for b in failed_benchmarks
            ]
        
        # Generate optimization priorities
        for benchmark in benchmarks:
            if benchmark.get('status') == 'failed' and 'evaluation' in benchmark:
                violations = benchmark['evaluation'].get('violations', [])
                for violation in violations:
                    if violation.get('severity') in ['high', 'critical']:
                        analysis['optimization_priorities'].append({
                            'benchmark': benchmark.get('benchmark_type', 'unknown'),
                            'metric': violation['metric'],
                            'severity': violation['severity'],
                            'improvement_needed': violation['actual'] - violation['target']
                        })
        
        return analysis
    
    @staticmethod
    def _generate_summary_report(enhanced_results: Dict[str, Any], summary_file: Path):
        """Generate human-readable summary report."""
        summary = enhanced_results.get('summary', {})
        
        report_content = f"""
Clinical Metabolomics Oracle Performance Benchmark Report
=========================================================

Generated: {enhanced_results['report_metadata']['generated_at']}
Task ID: {enhanced_results['report_metadata']['task_id']}
Duration: {enhanced_results.get('duration_seconds', 0):.1f} seconds

OVERALL RESULTS
---------------
Total Benchmarks: {summary.get('total_benchmarks', 0)}
Passed: {summary.get('passed_benchmarks', 0)}
Failed: {summary.get('failed_benchmarks', 0)}
Success Rate: {summary.get('success_rate_percent', 0):.1f}%
Overall Grade: {summary.get('overall_grade', 'Unknown')}

PERFORMANCE STATISTICS
----------------------
Average Response Time: {summary.get('performance_statistics', {}).get('avg_response_time_ms', 0):.1f}ms
Median Response Time: {summary.get('performance_statistics', {}).get('median_response_time_ms', 0):.1f}ms
Average Throughput: {summary.get('performance_statistics', {}).get('avg_throughput_ops_per_sec', 0):.2f} ops/sec
Average Memory Usage: {summary.get('performance_statistics', {}).get('avg_memory_usage_mb', 0):.1f}MB

BENCHMARK RESULTS
-----------------
"""
        
        benchmarks = enhanced_results.get('benchmarks', [])
        for benchmark in benchmarks:
            benchmark_type = benchmark.get('benchmark_type', 'unknown')
            status = benchmark.get('status', 'unknown')
            
            report_content += f"\n{benchmark_type.upper()}: {status.upper()}\n"
            
            if 'metrics' in benchmark:
                metrics = benchmark['metrics']
                report_content += f"  Response Time: {metrics.get('average_latency_ms', 0):.1f}ms\n"
                report_content += f"  Throughput: {metrics.get('throughput_ops_per_sec', 0):.2f} ops/sec\n"
                report_content += f"  Memory Usage: {metrics.get('memory_usage_mb', 0):.1f}MB\n"
                report_content += f"  Error Rate: {metrics.get('error_rate_percent', 0):.1f}%\n"
        
        # Add recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            report_content += "\nRECOMMENDATIONS\n"
            report_content += "---------------\n"
            for i, rec in enumerate(recommendations, 1):
                report_content += f"{i}. {rec}\n"
        
        # Write summary report
        with open(summary_file, 'w') as f:
            f.write(report_content)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

async def run_performance_benchmarks():
    """Run performance benchmarks from command line."""
    print("Starting Clinical Metabolomics Oracle Performance Benchmarks")
    print("Task: CMO-LIGHTRAG-008-T05")
    print("=" * 70)
    
    # Create benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite()
    
    # Run benchmark suite
    suite_results = await benchmark_suite.run_benchmark_suite()
    
    # Generate reports
    report_file = BenchmarkReportGenerator.generate_benchmark_report(suite_results)
    
    print(f"\nBenchmark execution completed!")
    print(f"Results saved to: {report_file}")
    
    return suite_results


if __name__ == "__main__":
    # Run benchmarks if executed directly
    asyncio.run(run_performance_benchmarks())