#!/usr/bin/env python3
"""
Advanced Test Fixtures and Utilities for Integration Testing.

This module provides comprehensive test fixtures, mock objects, and utilities 
specifically designed for integration testing between PDF processor and LightRAG 
components. It extends the base fixtures in conftest.py with specialized utilities
for complex integration test scenarios.

Components:
- Advanced PDF test data generators with failure simulation
- Sophisticated mock systems with realistic behavior patterns
- Error injection frameworks for testing robustness
- Performance monitoring utilities
- Test scenario builders for integration workflows

Author: Claude Code (Anthropic)
Created: August 6, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import tempfile
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from unittest.mock import MagicMock, AsyncMock, Mock
from contextlib import asynccontextmanager
import threading
import queue


# =====================================================================
# SPECIALIZED PDF TEST DATA GENERATORS
# =====================================================================

@dataclass
class AdvancedPDFScenario:
    """Represents complex PDF processing scenarios for integration testing."""
    name: str
    description: str
    pdf_count: int
    failure_rate: float  # 0.0 to 1.0
    processing_complexity: str  # 'simple', 'medium', 'complex'
    expected_entities: int
    expected_relationships: int
    cost_budget: float
    timeout_seconds: float
    memory_limit_mb: float
    
    def __post_init__(self):
        """Validate scenario parameters."""
        if not 0.0 <= self.failure_rate <= 1.0:
            raise ValueError("failure_rate must be between 0.0 and 1.0")
        if self.processing_complexity not in ['simple', 'medium', 'complex']:
            raise ValueError("processing_complexity must be 'simple', 'medium', or 'complex'")


class AdvancedBiomedicalPDFGenerator:
    """Advanced PDF generator with failure simulation and realistic complexity."""
    
    # Disease-specific content patterns
    DISEASE_PATTERNS = {
        'diabetes': {
            'metabolites': ['glucose', 'insulin', 'HbA1c', 'fructose', 'lactate'],
            'proteins': ['insulin', 'glucagon', 'GLUT4', 'adiponectin'],
            'pathways': ['glucose metabolism', 'insulin signaling', 'glycolysis'],
            'symptoms': ['hyperglycemia', 'polyuria', 'polydipsia', 'fatigue'],
            'treatments': ['metformin', 'insulin therapy', 'lifestyle modification']
        },
        'cardiovascular': {
            'metabolites': ['TMAO', 'cholesterol', 'triglycerides', 'homocysteine'],
            'proteins': ['troponin', 'CRP', 'BNP', 'LDL', 'HDL'],
            'pathways': ['lipid metabolism', 'inflammation', 'coagulation'],
            'symptoms': ['chest pain', 'dyspnea', 'edema', 'arrhythmia'],
            'treatments': ['statins', 'ACE inhibitors', 'beta blockers']
        },
        'cancer': {
            'metabolites': ['lactate', 'glutamine', 'succinate', 'oncometabolites'],
            'proteins': ['p53', 'VEGF', 'HER2', 'EGFR', 'ki67'],
            'pathways': ['Warburg effect', 'angiogenesis', 'apoptosis'],
            'symptoms': ['weight loss', 'fatigue', 'pain', 'bleeding'],
            'treatments': ['chemotherapy', 'immunotherapy', 'targeted therapy']
        },
        'liver_disease': {
            'metabolites': ['bilirubin', 'albumin', 'ammonia', 'bile acids'],
            'proteins': ['ALT', 'AST', 'ALP', 'gamma-GT', 'prothrombin'],
            'pathways': ['detoxification', 'protein synthesis', 'bile metabolism'],
            'symptoms': ['jaundice', 'ascites', 'hepatomegaly', 'fatigue'],
            'treatments': ['antiviral therapy', 'corticosteroids', 'transplant']
        }
    }
    
    # Technical complexity patterns
    COMPLEXITY_PATTERNS = {
        'simple': {
            'techniques': ['ELISA', 'colorimetric assay'],
            'sample_size': (20, 50),
            'statistical_methods': ['t-test', 'Mann-Whitney U'],
            'page_range': (5, 10),
            'entity_density': 0.3  # entities per 100 words
        },
        'medium': {
            'techniques': ['LC-MS/MS', 'GC-MS', 'NMR'],
            'sample_size': (50, 200),
            'statistical_methods': ['ANOVA', 'multivariate analysis', 'PCA'],
            'page_range': (10, 20),
            'entity_density': 0.5
        },
        'complex': {
            'techniques': ['multi-omics', 'systems biology', 'machine learning'],
            'sample_size': (200, 1000),
            'statistical_methods': ['random forest', 'neural networks', 'pathway analysis'],
            'page_range': (20, 40),
            'entity_density': 0.8
        }
    }
    
    @classmethod
    def generate_failure_scenarios(cls) -> List[AdvancedPDFScenario]:
        """Generate predefined failure scenarios for testing."""
        scenarios = [
            AdvancedPDFScenario(
                name="high_failure_rate",
                description="Test resilience with 30% PDF processing failures",
                pdf_count=10,
                failure_rate=0.3,
                processing_complexity='medium',
                expected_entities=50,
                expected_relationships=25,
                cost_budget=5.0,
                timeout_seconds=30.0,
                memory_limit_mb=500
            ),
            AdvancedPDFScenario(
                name="memory_pressure",
                description="Test under memory constraints with large PDFs",
                pdf_count=5,
                failure_rate=0.1,
                processing_complexity='complex',
                expected_entities=100,
                expected_relationships=80,
                cost_budget=10.0,
                timeout_seconds=60.0,
                memory_limit_mb=200
            ),
            AdvancedPDFScenario(
                name="timeout_stress",
                description="Test timeout handling with slow processing",
                pdf_count=15,
                failure_rate=0.2,
                processing_complexity='simple',
                expected_entities=30,
                expected_relationships=15,
                cost_budget=3.0,
                timeout_seconds=10.0,
                memory_limit_mb=1000
            ),
            AdvancedPDFScenario(
                name="budget_exhaustion",
                description="Test behavior when cost budget is exceeded",
                pdf_count=20,
                failure_rate=0.0,
                processing_complexity='complex',
                expected_entities=200,
                expected_relationships=150,
                cost_budget=2.0,  # Very low budget
                timeout_seconds=120.0,
                memory_limit_mb=1000
            ),
            AdvancedPDFScenario(
                name="mixed_complexity",
                description="Test with mixed complexity documents",
                pdf_count=12,
                failure_rate=0.15,
                processing_complexity='medium',
                expected_entities=75,
                expected_relationships=50,
                cost_budget=8.0,
                timeout_seconds=45.0,
                memory_limit_mb=750
            )
        ]
        
        return scenarios
    
    @classmethod
    def create_disease_specific_content(cls, disease: str, complexity: str = 'medium') -> str:
        """Generate disease-specific biomedical content."""
        if disease not in cls.DISEASE_PATTERNS:
            disease = 'diabetes'  # Default fallback
        
        disease_data = cls.DISEASE_PATTERNS[disease]
        complexity_data = cls.COMPLEXITY_PATTERNS.get(complexity, cls.COMPLEXITY_PATTERNS['medium'])
        
        # Generate study details
        sample_size = random.randint(*complexity_data['sample_size'])
        technique = random.choice(complexity_data['techniques'])
        statistical_method = random.choice(complexity_data['statistical_methods'])
        
        # Select relevant biomarkers
        metabolites = random.sample(disease_data['metabolites'], min(3, len(disease_data['metabolites'])))
        proteins = random.sample(disease_data['proteins'], min(3, len(disease_data['proteins'])))
        pathways = random.sample(disease_data['pathways'], min(2, len(disease_data['pathways'])))
        
        content = f"""
        CLINICAL STUDY: {disease.upper()} BIOMARKER RESEARCH
        
        ABSTRACT
        Background: {disease.title()} remains a significant health challenge affecting millions globally.
        This study investigates metabolic biomarkers associated with {disease} progression using {technique}.
        
        Objective: To identify and validate biomarkers for {disease} diagnosis and monitoring.
        
        Methods: We analyzed samples from {sample_size} patients with {disease} and {sample_size//2} controls.
        Key metabolites measured include {', '.join(metabolites)}. Protein analysis focused on {', '.join(proteins)}.
        Statistical analysis employed {statistical_method} with significance set at p<0.05.
        
        RESULTS
        Primary findings:
        - Significant alterations in {metabolites[0]} levels (p=0.001)
        - Elevated {proteins[0]} expression in patient samples
        - Dysregulation of {pathways[0]} pathway
        - Strong correlation between {metabolites[0]} and {proteins[0]} (r=0.78, p<0.001)
        
        Secondary metabolites showing significance:
        """
        
        # Add complexity-appropriate additional content
        if complexity == 'complex':
            content += f"""
        
        PATHWAY ANALYSIS
        Metabolic pathway enrichment revealed significant alterations in:
        1. {pathways[0]} - affected genes: {', '.join(random.sample(disease_data.get('genes', ['GENE1', 'GENE2']), 2))}
        2. {pathways[1] if len(pathways) > 1 else 'secondary pathway'}
        
        MACHINE LEARNING ANALYSIS
        Random forest classification achieved 85% accuracy in distinguishing {disease} patients from controls.
        Key features: {', '.join(metabolites[:2])}, {proteins[0]}
        
        NETWORK ANALYSIS
        Protein-protein interaction networks identified hub proteins: {proteins[0]}, {proteins[1] if len(proteins) > 1 else 'PROT2'}
        """
        
        content += f"""
        
        CONCLUSIONS
        This study demonstrates the potential of {technique}-based approaches for {disease} biomarker discovery.
        Key metabolites {', '.join(metabolites[:2])} show promise as diagnostic markers.
        Integration with {pathways[0]} pathway analysis provides mechanistic insights.
        
        Future directions include validation in larger cohorts and development of clinical assays.
        """
        
        return content
    
    @classmethod
    def create_failure_prone_document(cls, failure_type: str) -> Dict[str, Any]:
        """Create documents designed to trigger specific failure modes."""
        
        failure_scenarios = {
            'corrupted_pdf': {
                'content': "CORRUPTED_PDF_HEADER\x00\x01\x02INVALID_CONTENT",
                'metadata': {'corrupted': True},
                'expected_error': 'PDFValidationError'
            },
            'oversized_content': {
                'content': "OVERSIZED DOCUMENT\n" + "X" * 10000000,  # 10MB of X's
                'metadata': {'size_mb': 10},
                'expected_error': 'PDFMemoryError'
            },
            'slow_processing': {
                'content': cls.create_disease_specific_content('diabetes', 'complex'),
                'metadata': {'processing_delay': 30.0},  # 30 second delay
                'expected_error': 'PDFProcessingTimeoutError'
            },
            'invalid_encoding': {
                'content': "Invalid UTF-8: \xff\xfe\xfd",
                'metadata': {'encoding_error': True},
                'expected_error': 'PDFContentError'
            },
            'missing_file': {
                'content': None,
                'metadata': {'file_exists': False},
                'expected_error': 'PDFFileAccessError'
            }
        }
        
        return failure_scenarios.get(failure_type, failure_scenarios['corrupted_pdf'])


# =====================================================================
# SOPHISTICATED MOCK SYSTEMS
# =====================================================================

class RealisticLightRAGMock:
    """Sophisticated LightRAG mock with realistic response patterns and timing."""
    
    def __init__(self, working_dir: Path, enable_costs: bool = True):
        self.working_dir = working_dir
        self.enable_costs = enable_costs
        
        # Realistic system state
        self.documents_indexed = []
        self.entity_graph = {}
        self.relationship_graph = {}
        self.query_cache = {}
        self.system_stats = {
            'queries_processed': 0,
            'documents_indexed': 0,
            'total_cost': 0.0,
            'avg_query_time': 0.5,
            'cache_hit_rate': 0.0
        }
        
        # Performance simulation
        self.base_processing_time = 0.1
        self.complexity_multiplier = 1.0
        self.failure_probability = 0.0
        
        # Cost simulation
        self.cost_per_token = {
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'text-embedding-3-small': {'input': 0.00002, 'output': 0.0}
        }
        
    async def ainsert(self, documents: Union[str, List[str]]) -> Dict[str, Any]:
        """Sophisticated document insertion simulation."""
        if isinstance(documents, str):
            documents = [documents]
        
        # Simulate processing time based on document complexity
        total_chars = sum(len(doc) for doc in documents)
        processing_time = self.base_processing_time + (total_chars / 1000) * 0.01
        
        # Add complexity-based delay
        processing_time *= self.complexity_multiplier
        
        await asyncio.sleep(processing_time)
        
        # Simulate random failures
        if random.random() < self.failure_probability:
            raise Exception("Simulated LightRAG processing failure")
        
        # Process each document
        results = []
        total_cost = 0.0
        
        for i, doc in enumerate(documents):
            # Extract realistic entities and relationships
            entities = self._extract_biomedical_entities(doc)
            relationships = self._extract_relationships(doc, entities)
            
            # Calculate costs
            doc_cost = self._calculate_processing_cost(doc)
            total_cost += doc_cost
            
            # Store in mock database
            doc_record = {
                'id': f"doc_{len(self.documents_indexed) + i}",
                'content_preview': doc[:200] + "..." if len(doc) > 200 else doc,
                'entities': entities,
                'relationships': relationships,
                'cost': doc_cost,
                'processing_time': processing_time / len(documents),
                'timestamp': time.time()
            }
            
            self.documents_indexed.append(doc_record)
            results.append(doc_record)
            
            # Update entity and relationship graphs
            self._update_knowledge_graph(entities, relationships)
        
        # Update system statistics
        self.system_stats['documents_indexed'] += len(documents)
        self.system_stats['total_cost'] += total_cost
        
        return {
            'status': 'success',
            'documents_processed': len(documents),
            'entities_extracted': sum(len(r['entities']) for r in results),
            'relationships_found': sum(len(r['relationships']) for r in results),
            'total_cost': total_cost,
            'processing_time': processing_time,
            'results': results
        }
    
    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """Sophisticated query processing with caching and realistic responses."""
        query_start = time.time()
        
        # Check cache first
        cache_key = f"{query}_{mode}"
        if cache_key in self.query_cache:
            self.system_stats['cache_hit_rate'] = (
                self.system_stats['cache_hit_rate'] * 0.9 + 0.1
            )  # Moving average
            cached_result = self.query_cache[cache_key]
            await asyncio.sleep(0.01)  # Minimal cache retrieval time
            return cached_result['response']
        
        # Simulate query processing time
        query_complexity = len(query) / 100  # Complexity based on query length
        processing_time = max(0.1, random.gauss(0.5, 0.2)) * (1 + query_complexity)
        
        await asyncio.sleep(processing_time)
        
        # Generate contextual response
        response = self._generate_contextual_response(query, mode)
        
        # Calculate query cost
        query_cost = self._calculate_query_cost(query, response)
        
        # Cache the result
        self.query_cache[cache_key] = {
            'response': response,
            'cost': query_cost,
            'timestamp': time.time(),
            'processing_time': processing_time
        }
        
        # Update statistics
        self.system_stats['queries_processed'] += 1
        self.system_stats['total_cost'] += query_cost
        self.system_stats['avg_query_time'] = (
            self.system_stats['avg_query_time'] * 0.9 + processing_time * 0.1
        )
        
        return response
    
    def _extract_biomedical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract realistic biomedical entities from text."""
        entities = []
        text_lower = text.lower()
        
        # Biomedical entity patterns
        entity_patterns = {
            'METABOLITE': ['glucose', 'insulin', 'lactate', 'cholesterol', 'creatinine', 'urea', 'bilirubin'],
            'PROTEIN': ['albumin', 'hemoglobin', 'transferrin', 'CRP', 'troponin', 'BNP'],
            'GENE': ['APOE', 'BRCA1', 'TP53', 'EGFR', 'KRAS', 'PIK3CA'],
            'DISEASE': ['diabetes', 'cancer', 'cardiovascular', 'alzheimer', 'parkinson'],
            'DRUG': ['metformin', 'aspirin', 'insulin', 'statin', 'warfarin'],
            'PATHWAY': ['glycolysis', 'tca cycle', 'fatty acid', 'protein synthesis'],
            'TECHNIQUE': ['lc-ms', 'gc-ms', 'nmr', 'elisa', 'western blot']
        }
        
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Add some context around the entity
                    start_idx = text_lower.find(pattern)
                    context_start = max(0, start_idx - 50)
                    context_end = min(len(text), start_idx + len(pattern) + 50)
                    context = text[context_start:context_end]
                    
                    entities.append({
                        'type': entity_type,
                        'name': pattern,
                        'context': context,
                        'confidence': random.uniform(0.7, 0.95),
                        'position': start_idx
                    })
        
        # Limit number of entities to avoid overwhelming
        return entities[:20]
    
    def _extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract realistic relationships between entities."""
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Relationship patterns
        relationship_types = [
            'regulates', 'inhibits', 'activates', 'correlates_with',
            'metabolizes', 'produces', 'degrades', 'transports',
            'associated_with', 'biomarker_for', 'causally_related'
        ]
        
        # Create relationships between nearby entities
        for i in range(min(5, len(entities) - 1)):
            entity1 = entities[i]
            entity2 = entities[i + 1]
            
            # Calculate relationship strength based on proximity and types
            distance = abs(entity1['position'] - entity2['position'])
            strength = max(0.1, 1.0 - (distance / 1000))
            
            relationship = {
                'source': entity1['name'],
                'target': entity2['name'],
                'type': random.choice(relationship_types),
                'strength': strength,
                'evidence': f"Co-occurrence in text within {distance} characters"
            }
            
            relationships.append(relationship)
        
        return relationships
    
    def _update_knowledge_graph(self, entities: List[Dict[str, Any]], 
                               relationships: List[Dict[str, Any]]):
        """Update the mock knowledge graph."""
        # Update entity graph
        for entity in entities:
            entity_key = f"{entity['type']}:{entity['name']}"
            if entity_key not in self.entity_graph:
                self.entity_graph[entity_key] = {
                    'type': entity['type'],
                    'name': entity['name'],
                    'mentions': 0,
                    'contexts': []
                }
            
            self.entity_graph[entity_key]['mentions'] += 1
            self.entity_graph[entity_key]['contexts'].append(entity['context'])
        
        # Update relationship graph
        for rel in relationships:
            rel_key = f"{rel['source']}_{rel['type']}_{rel['target']}"
            if rel_key not in self.relationship_graph:
                self.relationship_graph[rel_key] = rel
                self.relationship_graph[rel_key]['frequency'] = 0
            
            self.relationship_graph[rel_key]['frequency'] += 1
    
    def _calculate_processing_cost(self, document: str) -> float:
        """Calculate realistic processing cost for document."""
        if not self.enable_costs:
            return 0.0
        
        # Estimate tokens (rough approximation)
        estimated_tokens = len(document.split()) * 1.3
        
        # Cost for embedding generation
        embedding_cost = estimated_tokens * self.cost_per_token['text-embedding-3-small']['input']
        
        # Cost for LLM processing (entity extraction, etc.)
        llm_cost = estimated_tokens * 0.5 * self.cost_per_token['gpt-4o-mini']['input']
        
        return embedding_cost + llm_cost
    
    def _calculate_query_cost(self, query: str, response: str) -> float:
        """Calculate realistic query cost."""
        if not self.enable_costs:
            return 0.0
        
        query_tokens = len(query.split()) * 1.3
        response_tokens = len(response.split()) * 1.3
        
        input_cost = query_tokens * self.cost_per_token['gpt-4o-mini']['input']
        output_cost = response_tokens * self.cost_per_token['gpt-4o-mini']['output']
        
        return input_cost + output_cost
    
    def _generate_contextual_response(self, query: str, mode: str) -> str:
        """Generate contextually appropriate responses."""
        query_lower = query.lower()
        
        # Analyze query intent
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            return self._generate_definition_response(query_lower)
        elif any(word in query_lower for word in ['how', 'mechanism', 'pathway']):
            return self._generate_mechanism_response(query_lower)
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return self._generate_comparison_response(query_lower)
        elif any(word in query_lower for word in ['find', 'search', 'list']):
            return self._generate_search_response(query_lower)
        else:
            return self._generate_general_response(query_lower)
    
    def _generate_definition_response(self, query: str) -> str:
        """Generate definition-type responses."""
        if 'metabolite' in query:
            return """A metabolite is a small molecule that is an intermediate or end product of metabolism. 
            In clinical metabolomics, key metabolites include glucose, lactate, amino acids, and lipids. 
            These molecules serve as biomarkers for various diseases and provide insights into metabolic pathway activity."""
        elif 'protein' in query:
            return """Proteins are large biomolecules composed of amino acids that perform various functions in organisms. 
            In clinical settings, proteins like albumin, hemoglobin, and enzymes serve as important biomarkers 
            for disease diagnosis and monitoring."""
        else:
            return """Based on the available literature, this term refers to a biomedical concept 
            that plays an important role in human health and disease processes."""
    
    def _generate_mechanism_response(self, query: str) -> str:
        """Generate mechanism-focused responses."""
        return """The mechanism involves complex molecular interactions within cellular pathways. 
        Key steps include substrate binding, enzymatic catalysis, and product formation. 
        Regulatory mechanisms involve feedback inhibition and allosteric modulation. 
        Clinical implications include altered pathway activity in disease states."""
    
    def _generate_comparison_response(self, query: str) -> str:
        """Generate comparative responses."""
        return """Comparative analysis reveals significant differences between conditions. 
        Key distinguishing features include altered metabolite profiles, differential protein expression, 
        and distinct pathway activities. Statistical analysis shows p-values < 0.05 for major differences."""
    
    def _generate_search_response(self, query: str) -> str:
        """Generate search-type responses."""
        return """Based on the indexed literature, relevant findings include:
        1. Multiple studies demonstrating significant associations
        2. Biomarker validation across different populations  
        3. Pathway enrichment in relevant biological processes
        4. Clinical applications in diagnosis and monitoring"""
    
    def _generate_general_response(self, query: str) -> str:
        """Generate general responses."""
        return """The literature provides comprehensive evidence supporting the clinical relevance 
        of multi-omics approaches in understanding disease mechanisms. Integration of metabolomics, 
        proteomics, and genomics data offers valuable insights for biomarker discovery and 
        personalized medicine applications."""
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            **self.system_stats,
            'entity_count': len(self.entity_graph),
            'relationship_count': len(self.relationship_graph),
            'cache_size': len(self.query_cache),
            'knowledge_graph_density': len(self.relationship_graph) / max(1, len(self.entity_graph)),
            'working_directory': str(self.working_dir)
        }
    
    def set_failure_rate(self, probability: float):
        """Set failure probability for testing."""
        self.failure_probability = max(0.0, min(1.0, probability))
    
    def set_complexity_multiplier(self, multiplier: float):
        """Set processing complexity multiplier."""
        self.complexity_multiplier = max(0.1, multiplier)
    
    def clear_cache(self):
        """Clear query cache."""
        self.query_cache.clear()
    
    def reset_statistics(self):
        """Reset system statistics."""
        self.system_stats = {
            'queries_processed': 0,
            'documents_indexed': 0,
            'total_cost': 0.0,
            'avg_query_time': 0.5,
            'cache_hit_rate': 0.0
        }


# =====================================================================
# PERFORMANCE MONITORING UTILITIES
# =====================================================================

class IntegrationPerformanceMonitor:
    """Monitor performance metrics during integration testing."""
    
    def __init__(self):
        self.metrics = {
            'test_start_time': time.time(),
            'operations': [],
            'resource_usage': [],
            'errors': [],
            'warnings': []
        }
        self.active_operations = {}
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, **context):
        """Context manager for monitoring operation performance."""
        start_time = time.time()
        operation_id = f"{operation_name}_{start_time}"
        
        self.active_operations[operation_id] = {
            'name': operation_name,
            'start_time': start_time,
            'context': context
        }
        
        try:
            yield operation_id
            
            # Record successful operation
            end_time = time.time()
            self.metrics['operations'].append({
                'id': operation_id,
                'name': operation_name,
                'duration': end_time - start_time,
                'status': 'success',
                'context': context,
                'timestamp': start_time
            })
            
        except Exception as e:
            # Record failed operation
            end_time = time.time()
            self.metrics['operations'].append({
                'id': operation_id,
                'name': operation_name,
                'duration': end_time - start_time,
                'status': 'error',
                'error': str(e),
                'context': context,
                'timestamp': start_time
            })
            
            self.metrics['errors'].append({
                'operation': operation_name,
                'error': str(e),
                'timestamp': end_time,
                'context': context
            })
            
            raise
        
        finally:
            self.active_operations.pop(operation_id, None)
    
    def record_resource_usage(self):
        """Record current resource usage."""
        try:
            import psutil
            process = psutil.Process()
            
            self.metrics['resource_usage'].append({
                'timestamp': time.time(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'active_operations': len(self.active_operations)
            })
        except ImportError:
            # psutil not available
            pass
    
    def add_warning(self, message: str, **context):
        """Add warning to metrics."""
        self.metrics['warnings'].append({
            'message': message,
            'timestamp': time.time(),
            'context': context
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_time = time.time() - self.metrics['test_start_time']
        operations = self.metrics['operations']
        
        if not operations:
            return {
                'total_test_time': total_time,
                'operations_completed': 0,
                'error_count': len(self.metrics['errors']),
                'warning_count': len(self.metrics['warnings'])
            }
        
        successful_ops = [op for op in operations if op['status'] == 'success']
        failed_ops = [op for op in operations if op['status'] == 'error']
        
        return {
            'total_test_time': total_time,
            'operations_completed': len(operations),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'success_rate': len(successful_ops) / len(operations) * 100,
            'average_operation_time': sum(op['duration'] for op in successful_ops) / max(1, len(successful_ops)),
            'fastest_operation': min(op['duration'] for op in successful_ops) if successful_ops else 0,
            'slowest_operation': max(op['duration'] for op in successful_ops) if successful_ops else 0,
            'error_count': len(self.metrics['errors']),
            'warning_count': len(self.metrics['warnings']),
            'peak_memory_mb': max((ru['memory_mb'] for ru in self.metrics['resource_usage']), default=0),
            'resource_samples': len(self.metrics['resource_usage'])
        }


# =====================================================================
# TEST SCENARIO BUILDERS
# =====================================================================

class IntegrationTestScenarioBuilder:
    """Build complex integration test scenarios."""
    
    def __init__(self):
        self.scenarios = {}
    
    def create_end_to_end_workflow(self, name: str, **params) -> Dict[str, Any]:
        """Create end-to-end integration test workflow."""
        scenario = {
            'name': name,
            'type': 'end_to_end',
            'steps': [
                {
                    'step': 'setup_environment',
                    'description': 'Initialize test environment and dependencies'
                },
                {
                    'step': 'create_test_pdfs',
                    'description': 'Generate realistic biomedical PDF test documents',
                    'pdf_count': params.get('pdf_count', 5),
                    'complexity': params.get('complexity', 'medium')
                },
                {
                    'step': 'initialize_lightrag',
                    'description': 'Initialize LightRAG system with test configuration'
                },
                {
                    'step': 'process_pdfs',
                    'description': 'Process PDFs through biomedical processor',
                    'batch_processing': params.get('batch_processing', True),
                    'parallel_workers': params.get('parallel_workers', 3)
                },
                {
                    'step': 'index_documents',
                    'description': 'Index processed documents in LightRAG',
                    'chunk_strategy': params.get('chunk_strategy', 'adaptive')
                },
                {
                    'step': 'execute_queries',
                    'description': 'Execute test queries against knowledge base',
                    'query_types': params.get('query_types', ['entity', 'relationship', 'semantic'])
                },
                {
                    'step': 'validate_results',
                    'description': 'Validate query results and system behavior'
                },
                {
                    'step': 'monitor_costs',
                    'description': 'Verify cost tracking and budget compliance'
                },
                {
                    'step': 'cleanup',
                    'description': 'Clean up test environment and resources'
                }
            ],
            'expected_outcomes': {
                'documents_processed': params.get('pdf_count', 5),
                'entities_extracted': params.get('expected_entities', 50),
                'relationships_found': params.get('expected_relationships', 25),
                'max_cost_usd': params.get('max_cost', 10.0),
                'max_duration_seconds': params.get('max_duration', 120)
            },
            'success_criteria': {
                'processing_success_rate': 0.9,
                'query_response_time_max': 5.0,
                'cost_under_budget': True,
                'no_memory_leaks': True
            }
        }
        
        self.scenarios[name] = scenario
        return scenario
    
    def create_failure_recovery_scenario(self, name: str, **params) -> Dict[str, Any]:
        """Create failure recovery test scenario."""
        scenario = {
            'name': name,
            'type': 'failure_recovery',
            'failure_injection': {
                'pdf_processing_failures': params.get('pdf_failure_rate', 0.3),
                'lightrag_failures': params.get('lightrag_failure_rate', 0.1),
                'network_timeouts': params.get('timeout_rate', 0.2),
                'memory_pressure': params.get('memory_pressure', True)
            },
            'recovery_expectations': {
                'graceful_degradation': True,
                'error_logging': True,
                'partial_success_handling': True,
                'resource_cleanup': True
            },
            'validation_steps': [
                'verify_error_handling',
                'check_system_stability',
                'validate_partial_results',
                'confirm_resource_cleanup'
            ]
        }
        
        self.scenarios[name] = scenario
        return scenario
    
    def create_performance_benchmark_scenario(self, name: str, **params) -> Dict[str, Any]:
        """Create performance benchmarking scenario."""
        scenario = {
            'name': name,
            'type': 'performance_benchmark',
            'workload': {
                'document_count': params.get('document_count', 50),
                'concurrent_operations': params.get('concurrent_ops', 10),
                'query_load': params.get('query_count', 100),
                'duration_minutes': params.get('duration', 10)
            },
            'performance_targets': {
                'documents_per_second': params.get('target_doc_rate', 2.0),
                'queries_per_second': params.get('target_query_rate', 10.0),
                'average_response_time': params.get('target_response_time', 2.0),
                'memory_limit_mb': params.get('memory_limit', 1000),
                'cost_efficiency_target': params.get('cost_per_doc', 0.1)
            },
            'monitoring': {
                'resource_sampling_interval': 1.0,
                'performance_alerts': True,
                'detailed_profiling': params.get('profiling', False)
            }
        }
        
        self.scenarios[name] = scenario
        return scenario
    
    def get_scenario(self, name: str) -> Optional[Dict[str, Any]]:
        """Get scenario by name."""
        return self.scenarios.get(name)
    
    def list_scenarios(self) -> List[str]:
        """List all available scenarios."""
        return list(self.scenarios.keys())


# =====================================================================
# ADVANCED PYTEST FIXTURES
# =====================================================================

@pytest.fixture
def advanced_pdf_scenarios():
    """Provide advanced PDF test scenarios."""
    return AdvancedBiomedicalPDFGenerator.generate_failure_scenarios()


@pytest.fixture
def realistic_lightrag_mock(temp_dir):
    """Provide sophisticated LightRAG mock system."""
    return RealisticLightRAGMock(temp_dir, enable_costs=True)


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities."""
    return IntegrationPerformanceMonitor()


@pytest.fixture
def scenario_builder():
    """Provide integration test scenario builder."""
    return IntegrationTestScenarioBuilder()


@pytest.fixture
def disease_specific_content():
    """Provide disease-specific content generator."""
    def _generate(disease: str, complexity: str = 'medium') -> str:
        return AdvancedBiomedicalPDFGenerator.create_disease_specific_content(disease, complexity)
    
    return _generate


@pytest.fixture
def failure_simulation():
    """Provide failure simulation utilities."""
    def _create_failure_document(failure_type: str) -> Dict[str, Any]:
        return AdvancedBiomedicalPDFGenerator.create_failure_prone_document(failure_type)
    
    return _create_failure_document


# =====================================================================
# INTEGRATION TEST UTILITIES
# =====================================================================

def assert_integration_success(results: Dict[str, Any], expectations: Dict[str, Any]):
    """Assert integration test success criteria."""
    assert results['status'] == 'success', f"Integration failed: {results.get('error', 'Unknown error')}"
    
    if 'documents_processed' in expectations:
        assert results['documents_processed'] >= expectations['documents_processed']
    
    if 'entities_extracted' in expectations:
        assert results['entities_extracted'] >= expectations['entities_extracted']
    
    if 'max_cost' in expectations:
        assert results.get('total_cost', 0) <= expectations['max_cost']
    
    if 'max_duration' in expectations:
        assert results.get('processing_time', 0) <= expectations['max_duration']


async def run_integration_workflow(scenario: Dict[str, Any], 
                                 environment: Any,
                                 monitor: IntegrationPerformanceMonitor) -> Dict[str, Any]:
    """Execute complete integration test workflow."""
    results = {
        'scenario_name': scenario['name'],
        'status': 'running',
        'steps_completed': [],
        'step_results': {},
        'errors': [],
        'performance_metrics': {}
    }
    
    try:
        for step in scenario['steps']:
            step_name = step['step']
            
            async with monitor.monitor_operation(step_name, **step):
                # Execute step based on type
                if step_name == 'setup_environment':
                    await _execute_setup_step(environment)
                elif step_name == 'create_test_pdfs':
                    step_result = await _execute_pdf_creation_step(environment, step)
                    results['step_results'][step_name] = step_result
                elif step_name == 'process_pdfs':
                    step_result = await _execute_pdf_processing_step(environment, step)
                    results['step_results'][step_name] = step_result
                # Add more step handlers as needed
                
                results['steps_completed'].append(step_name)
        
        results['status'] = 'success'
        
    except Exception as e:
        results['status'] = 'failed'
        results['errors'].append(str(e))
    
    finally:
        results['performance_metrics'] = monitor.get_performance_summary()
    
    return results


async def _execute_setup_step(environment):
    """Execute environment setup step."""
    # Ensure directories exist
    environment.working_dir.mkdir(exist_ok=True)
    (environment.working_dir / "pdfs").mkdir(exist_ok=True)
    (environment.working_dir / "logs").mkdir(exist_ok=True)


async def _execute_pdf_creation_step(environment, step_config):
    """Execute PDF creation step."""
    pdf_count = step_config.get('pdf_count', 5)
    complexity = step_config.get('complexity', 'medium')
    
    pdfs_created = environment.create_test_pdf_collection(pdf_count)
    
    return {
        'pdfs_created': len(pdfs_created),
        'pdf_paths': [str(p) for p in pdfs_created]
    }


async def _execute_pdf_processing_step(environment, step_config):
    """Execute PDF processing step."""
    batch_processing = step_config.get('batch_processing', True)
    
    # Get PDF files from environment
    pdf_dir = environment.working_dir / "pdfs"
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.txt"))
    
    if batch_processing:
        results = await environment.pdf_processor.process_batch_pdfs(pdf_files)
    else:
        results = []
        for pdf_file in pdf_files:
            result = await environment.pdf_processor.process_pdf(pdf_file)
            results.append(result)
    
    return {
        'files_processed': len(pdf_files),
        'processing_results': results if not batch_processing else results
    }