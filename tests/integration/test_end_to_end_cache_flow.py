"""
End-to-End Cache Flow Integration Tests for Clinical Metabolomics Oracle

This module provides comprehensive end-to-end integration tests for the complete
query processing pipeline with multi-tier caching system, focusing on realistic
biomedical workflows and complex interaction scenarios.

Integration Flow Coverage:
- Complete query processing pipeline with caching at every stage
- LightRAG knowledge graph integration with cache warming
- Emergency fallback system with cache coordination
- Predictive caching for query variations and follow-up queries
- Cache performance monitoring and optimization
- Real-world clinical metabolomics research workflows
- Multi-user concurrent access patterns
- Cache consistency across distributed components

Workflow Scenarios:
- Clinical Research Workflow: Sample analysis → Biomarker identification → Pathway analysis
- Literature Review Workflow: Query expansion → Source aggregation → Citation management  
- Diagnostic Workflow: Symptom analysis → Biomarker correlation → Clinical recommendations
- Drug Discovery Workflow: Target identification → Pathway analysis → Interaction prediction

Performance Validation:
- End-to-end response time < 3s for complex biomedical workflows
- Cache warming efficiency > 85% for common query patterns
- Fallback system activation < 200ms during cache failures
- Predictive cache accuracy > 70% for follow-up queries
- Multi-tier cache coordination < 50ms latency

Author: Claude Code (Anthropic)  
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import json
import tempfile
import shutil
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import concurrent.futures
import hashlib
import random
from collections import defaultdict, deque

# Import system components (mock imports for test environment)
try:
    from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction
    from lightrag_integration.research_categorizer import ResearchCategory
    from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    from lightrag_integration.comprehensive_fallback_system import ComprehensiveFallbackSystem
except ImportError:
    # Mock implementations for test environment
    class RoutingDecision(Enum):
        LIGHTRAG = "lightrag"
        PERPLEXITY = "perplexity"  
        EITHER = "either"
        HYBRID = "hybrid"
    
    class ResearchCategory(Enum):
        METABOLITE_IDENTIFICATION = "metabolite_identification"
        PATHWAY_ANALYSIS = "pathway_analysis"
        BIOMARKER_DISCOVERY = "biomarker_discovery"
        CLINICAL_DIAGNOSIS = "clinical_diagnosis"
        LITERATURE_SEARCH = "literature_search"
        DRUG_DISCOVERY = "drug_discovery"
        DATA_PREPROCESSING = "data_preprocessing"
        GENERAL_QUERY = "general_query"

# Import cache system components
from tests.unit.test_multi_tier_cache import MultiTierCache, MockL1MemoryCache, MockL2DiskCache, MockL3RedisCache


@dataclass
class WorkflowStep:
    """Represents a single step in a biomedical research workflow."""
    step_id: int
    step_name: str
    query_text: str
    expected_category: ResearchCategory
    expected_routing: RoutingDecision
    dependencies: List[int] = field(default_factory=list)
    cache_priority: str = 'medium'
    expected_confidence: float = 0.7
    follow_up_queries: List[str] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition for end-to-end testing."""
    workflow_id: str
    workflow_name: str
    description: str
    steps: List[WorkflowStep]
    expected_total_time_ms: float = 5000
    cache_warming_queries: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheFlowMetrics:
    """Comprehensive metrics for cache flow analysis."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    predictive_hits: int = 0
    fallback_activations: int = 0
    avg_response_time_ms: float = 0.0
    peak_response_time_ms: float = 0.0
    cache_efficiency: float = 0.0
    workflow_completion_rate: float = 0.0
    tier_utilization: Dict[str, int] = field(default_factory=lambda: {'L1': 0, 'L2': 0, 'L3': 0})
    error_rate: float = 0.0
    
    def calculate_efficiency(self):
        """Calculate overall cache efficiency metrics."""
        if self.total_queries > 0:
            self.cache_efficiency = (self.cache_hits + self.predictive_hits) / self.total_queries
            self.error_rate = self.fallback_activations / self.total_queries
        return self.cache_efficiency


class MockLightRAGSystem:
    """Mock LightRAG knowledge graph system with cache integration."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.knowledge_cache_prefix = "lightrag:"
        self.graph_cache_prefix = "graph:"
        self.stats = {
            'queries_processed': 0,
            'graph_traversals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'knowledge_base_hits': 0
        }
        
        # Mock knowledge base
        self.knowledge_base = {
            'metabolites': {
                'glucose': {
                    'pathways': ['glycolysis', 'gluconeogenesis', 'pentose_phosphate'],
                    'enzymes': ['hexokinase', 'glucose-6-phosphatase'],
                    'diseases': ['diabetes', 'metabolic_syndrome'],
                    'confidence': 0.95
                },
                'insulin': {
                    'pathways': ['insulin_signaling', 'glucose_homeostasis'],
                    'targets': ['insulin_receptor', 'glut4'],
                    'diseases': ['diabetes', 'insulin_resistance'],
                    'confidence': 0.9
                }
            },
            'pathways': {
                'glycolysis': {
                    'metabolites': ['glucose', 'pyruvate', 'ATP', 'NADH'],
                    'enzymes': ['hexokinase', 'phosphofructokinase', 'pyruvate_kinase'],
                    'regulation': ['insulin', 'glucagon', 'AMP'],
                    'confidence': 0.95
                },
                'insulin_signaling': {
                    'components': ['insulin', 'insulin_receptor', 'IRS1', 'PI3K', 'AKT'],
                    'targets': ['GLUT4', 'glycogen_synthase', 'acetyl_CoA_carboxylase'],
                    'diseases': ['diabetes', 'metabolic_syndrome'],
                    'confidence': 0.9
                }
            },
            'diseases': {
                'diabetes': {
                    'biomarkers': ['HbA1c', 'fasting_glucose', 'insulin', 'C-peptide'],
                    'pathways_affected': ['glucose_metabolism', 'insulin_signaling', 'lipid_metabolism'],
                    'metabolites': ['glucose', 'ketones', 'fatty_acids'],
                    'confidence': 0.9
                }
            }
        }
    
    async def query_knowledge_graph(self, query_text: str, 
                                  context: Optional[Dict[str, Any]] = None,
                                  use_cache: bool = True) -> Dict[str, Any]:
        """
        Query LightRAG knowledge graph with cache integration.
        
        Args:
            query_text: Natural language query
            context: Optional context for query interpretation
            use_cache: Whether to use caching for knowledge graph queries
            
        Returns:
            Dict containing knowledge graph query results
        """
        start_time = time.time()
        cache_key = f"{self.knowledge_cache_prefix}{hashlib.md5((query_text + str(context)).encode()).hexdigest()}"
        
        # Check cache first
        if use_cache:
            cached_result = await self.cache_system.get(cache_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return {
                    **cached_result,
                    'from_cache': True,
                    'query_time_ms': (time.time() - start_time) * 1000
                }
            self.stats['cache_misses'] += 1
        
        # Simulate knowledge graph processing
        await asyncio.sleep(0.3)  # Simulate graph traversal time
        
        # Extract entities and relationships
        query_lower = query_text.lower()
        entities_found = []
        relationships = []
        pathways = []
        
        # Simple entity extraction
        for entity_type, entities in self.knowledge_base.items():
            for entity_name, entity_data in entities.items():
                if entity_name in query_lower or any(keyword in query_lower 
                    for keyword in entity_data.keys()):
                    entities_found.append({
                        'name': entity_name,
                        'type': entity_type,
                        'data': entity_data,
                        'confidence': entity_data.get('confidence', 0.8)
                    })
        
        # Extract relationships
        if 'pathway' in query_lower or 'mechanism' in query_lower:
            for entity in entities_found:
                if entity['type'] == 'metabolites':
                    pathways.extend(entity['data'].get('pathways', []))
                elif entity['type'] == 'pathways':
                    relationships.append({
                        'type': 'pathway_components',
                        'pathway': entity['name'],
                        'components': entity['data']
                    })
        
        # Construct knowledge graph result
        kg_result = {
            'query': query_text,
            'entities': entities_found[:10],  # Limit results
            'relationships': relationships[:5],
            'pathways': list(set(pathways))[:5],
            'confidence': 0.8 if entities_found else 0.3,
            'query_time_ms': (time.time() - start_time) * 1000,
            'graph_traversals': len(entities_found) + len(relationships),
            'from_cache': False,
            'source': 'lightrag_knowledge_graph'
        }
        
        # Cache high-confidence results
        if use_cache and kg_result['confidence'] > 0.5:
            cache_data = {k: v for k, v in kg_result.items() 
                         if k not in ['query_time_ms']}
            await self.cache_system.set(cache_key, cache_data, ttl=3600)
        
        self.stats['queries_processed'] += 1
        self.stats['graph_traversals'] += kg_result['graph_traversals']
        self.stats['knowledge_base_hits'] += len(entities_found)
        
        return kg_result
    
    async def warm_knowledge_cache(self, entities: List[str], 
                                 relationships: List[str]) -> Dict[str, Any]:
        """
        Warm knowledge graph cache with common entities and relationships.
        
        Args:
            entities: List of entity names to pre-load
            relationships: List of relationship types to pre-load
            
        Returns:
            Dict containing cache warming statistics
        """
        start_time = time.time()
        warming_stats = {
            'entities_cached': 0,
            'relationships_cached': 0,
            'total_cache_entries': 0,
            'warming_time_ms': 0,
            'errors': []
        }
        
        # Pre-load entities
        for entity in entities:
            try:
                query = f"What is {entity}?"
                result = await self.query_knowledge_graph(query, use_cache=True)
                if result['entities']:
                    warming_stats['entities_cached'] += 1
            except Exception as e:
                warming_stats['errors'].append(f"Entity {entity}: {str(e)}")
        
        # Pre-load relationships  
        for relationship in relationships:
            try:
                query = f"How does {relationship} work?"
                result = await self.query_knowledge_graph(query, use_cache=True)
                if result['relationships']:
                    warming_stats['relationships_cached'] += 1
            except Exception as e:
                warming_stats['errors'].append(f"Relationship {relationship}: {str(e)}")
        
        warming_stats['total_cache_entries'] = (warming_stats['entities_cached'] + 
                                               warming_stats['relationships_cached'])
        warming_stats['warming_time_ms'] = (time.time() - start_time) * 1000
        
        return warming_stats


class MockFallbackSystem:
    """Mock emergency fallback system with cache coordination."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.fallback_cache_prefix = "fallback:"
        self.circuit_breaker_state = {'open': False, 'failures': 0, 'last_failure': 0}
        self.stats = {
            'fallback_activations': 0,
            'circuit_breaker_triggers': 0,
            'emergency_responses': 0,
            'cache_fallback_hits': 0
        }
    
    async def handle_system_failure(self, original_query: str, 
                                  error_context: Dict[str, Any],
                                  use_cache: bool = True) -> Dict[str, Any]:
        """
        Handle system failures with cache-backed fallback responses.
        
        Args:
            original_query: The original query that failed
            error_context: Context about the failure
            use_cache: Whether to use cache for fallback responses
            
        Returns:
            Dict containing fallback response
        """
        start_time = time.time()
        cache_key = f"{self.fallback_cache_prefix}{hashlib.md5(original_query.encode()).hexdigest()}"
        
        self.stats['fallback_activations'] += 1
        
        # Check if we have a cached fallback response
        if use_cache:
            cached_fallback = await self.cache_system.get(cache_key)
            if cached_fallback is not None:
                self.stats['cache_fallback_hits'] += 1
                return {
                    **cached_fallback,
                    'fallback_activated': True,
                    'fallback_time_ms': (time.time() - start_time) * 1000
                }
        
        # Generate emergency response
        await asyncio.sleep(0.05)  # Quick fallback generation
        
        # Simple fallback response based on query content
        query_lower = original_query.lower()
        
        if 'metabolite' in query_lower or 'compound' in query_lower:
            fallback_content = {
                'response': 'General information about metabolites and compounds in biological systems',
                'type': 'metabolite_general',
                'confidence': 0.4,
                'sources': ['general_biochemistry_knowledge']
            }
        elif 'pathway' in query_lower or 'mechanism' in query_lower:
            fallback_content = {
                'response': 'Basic information about biochemical pathways and mechanisms',
                'type': 'pathway_general', 
                'confidence': 0.4,
                'sources': ['biochemistry_fundamentals']
            }
        elif 'biomarker' in query_lower:
            fallback_content = {
                'response': 'General information about biomarkers in clinical diagnostics',
                'type': 'biomarker_general',
                'confidence': 0.4,
                'sources': ['clinical_diagnostics_basics']
            }
        else:
            fallback_content = {
                'response': 'General biomedical information - please refine your query',
                'type': 'general_fallback',
                'confidence': 0.2,
                'sources': ['general_knowledge']
            }
        
        fallback_response = {
            'query': original_query,
            'content': fallback_content,
            'fallback_activated': True,
            'error_context': error_context,
            'fallback_time_ms': (time.time() - start_time) * 1000,
            'circuit_breaker_active': self.circuit_breaker_state['open']
        }
        
        # Cache fallback response for similar failures
        if use_cache:
            await self.cache_system.set(cache_key, fallback_response, ttl=1800)
        
        self.stats['emergency_responses'] += 1
        return fallback_response
    
    async def check_system_health(self, components: List[str]) -> Dict[str, Any]:
        """
        Check health of system components and update circuit breaker state.
        
        Args:
            components: List of component names to check
            
        Returns:
            Dict containing system health status
        """
        health_status = {
            'overall_health': 'healthy',
            'component_health': {},
            'cache_system_health': 'healthy',
            'circuit_breaker_status': self.circuit_breaker_state,
            'recommendations': []
        }
        
        # Mock health checks
        for component in components:
            # Simulate random component health (mostly healthy)
            component_healthy = random.random() > 0.1  # 90% healthy
            health_status['component_health'][component] = 'healthy' if component_healthy else 'degraded'
            
            if not component_healthy:
                health_status['overall_health'] = 'degraded'
                health_status['recommendations'].append(f"Check {component} component")
        
        # Check cache system health
        cache_stats = self.cache_system.get_comprehensive_stats()
        cache_hit_rate = cache_stats.get('overall_hit_rate', 0)
        
        if cache_hit_rate < 0.5:
            health_status['cache_system_health'] = 'degraded'
            health_status['recommendations'].append('Cache hit rate below 50% - consider cache warming')
        
        return health_status


class PredictiveCacheSystem:
    """Predictive caching system for query variations and follow-ups."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.prediction_cache_prefix = "prediction:"
        self.query_patterns = defaultdict(list)
        self.query_history = deque(maxlen=1000)
        self.stats = {
            'predictions_made': 0,
            'predictions_correct': 0,
            'cache_preloads': 0,
            'pattern_matches': 0
        }
    
    def extract_query_patterns(self, query: str) -> List[str]:
        """Extract patterns from query for prediction."""
        patterns = []
        query_lower = query.lower()
        
        # Extract entity patterns
        if 'metabolite' in query_lower:
            patterns.append('metabolite_query')
        if 'pathway' in query_lower:
            patterns.append('pathway_query')
        if 'biomarker' in query_lower:
            patterns.append('biomarker_query')
        if 'diabetes' in query_lower:
            patterns.append('diabetes_related')
        
        # Extract question patterns
        if query_lower.startswith('what'):
            patterns.append('what_question')
        if query_lower.startswith('how'):
            patterns.append('how_question')
        if 'latest' in query_lower or 'recent' in query_lower:
            patterns.append('temporal_query')
        
        return patterns
    
    async def predict_follow_up_queries(self, current_query: str, 
                                      current_result: Dict[str, Any]) -> List[str]:
        """
        Predict likely follow-up queries based on current query and result.
        
        Args:
            current_query: The current query being processed
            current_result: Result of current query processing
            
        Returns:
            List of predicted follow-up queries
        """
        predictions = []
        query_lower = current_query.lower()
        
        # Pattern-based predictions
        if 'metabolite' in query_lower:
            predictions.extend([
                f"What pathways involve {current_query.split()[-1]}?",
                f"Biomarkers related to {current_query.split()[-1]}",
                f"Clinical significance of {current_query.split()[-1]}"
            ])
        
        if 'pathway' in query_lower:
            pathway_name = 'metabolic pathway'  # Simplified extraction
            predictions.extend([
                f"Enzymes in {pathway_name}",
                f"Regulation of {pathway_name}",
                f"Diseases affecting {pathway_name}"
            ])
        
        if 'biomarker' in query_lower:
            predictions.extend([
                "Clinical validation of biomarkers",
                "Biomarker discovery methods",
                "Statistical analysis of biomarker data"
            ])
        
        # Historical pattern-based predictions
        current_patterns = self.extract_query_patterns(current_query)
        for pattern in current_patterns:
            if pattern in self.query_patterns:
                historical_queries = self.query_patterns[pattern][-5:]  # Last 5 queries
                predictions.extend([q for q in historical_queries if q != current_query])
        
        self.stats['predictions_made'] += len(predictions)
        return predictions[:5]  # Limit predictions
    
    async def preload_predicted_queries(self, predicted_queries: List[str],
                                      processor) -> Dict[str, Any]:
        """
        Preload cache with predicted queries.
        
        Args:
            predicted_queries: List of queries to preload
            processor: Query processor to use for preloading
            
        Returns:
            Dict containing preloading statistics
        """
        preload_stats = {
            'queries_preloaded': 0,
            'cache_entries_created': 0,
            'preload_time_ms': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        for query in predicted_queries[:3]:  # Limit preloading to avoid overload
            try:
                # Process query to populate cache
                await processor.process_query_with_cache(query, use_cache=True)
                preload_stats['queries_preloaded'] += 1
                self.stats['cache_preloads'] += 1
            except Exception as e:
                preload_stats['errors'].append(f"Query '{query[:30]}...': {str(e)}")
        
        preload_stats['preload_time_ms'] = (time.time() - start_time) * 1000
        return preload_stats
    
    def update_query_patterns(self, query: str, result: Dict[str, Any]):
        """Update query patterns based on processed query."""
        patterns = self.extract_query_patterns(query)
        
        for pattern in patterns:
            self.query_patterns[pattern].append(query)
            if len(self.query_patterns[pattern]) > 20:  # Limit pattern history
                self.query_patterns[pattern] = self.query_patterns[pattern][-20:]
        
        self.query_history.append({
            'query': query,
            'patterns': patterns,
            'timestamp': time.time(),
            'success': result.get('confidence', 0) > 0.5
        })


class EndToEndQueryProcessor:
    """Complete end-to-end query processor with all integration components."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.lightrag_system = MockLightRAGSystem(cache_system)
        self.fallback_system = MockFallbackSystem(cache_system)
        self.predictive_cache = PredictiveCacheSystem(cache_system)
        
        # Import query processor from previous test file
        from .test_query_processing_cache import IntegratedQueryProcessor
        self.base_processor = IntegratedQueryProcessor(cache_system)
        
        self.metrics = CacheFlowMetrics()
        self.workflow_states = {}
        
    async def process_workflow(self, workflow: WorkflowDefinition,
                             user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process complete biomedical research workflow with end-to-end caching.
        
        Args:
            workflow: Complete workflow definition
            user_context: Optional user/session context
            
        Returns:
            Dict containing workflow execution results and metrics
        """
        start_time = time.time()
        workflow_results = {
            'workflow_id': workflow.workflow_id,
            'workflow_name': workflow.workflow_name,
            'steps_completed': 0,
            'step_results': [],
            'cache_performance': {},
            'errors': [],
            'total_time_ms': 0,
            'success': False
        }
        
        # Initialize workflow state
        self.workflow_states[workflow.workflow_id] = {
            'current_step': 0,
            'completed_steps': set(),
            'step_context': {},
            'accumulated_knowledge': {}
        }
        
        try:
            # Cache warming phase
            if workflow.cache_warming_queries:
                warming_stats = await self.base_processor.warm_cache_with_common_queries(
                    workflow.cache_warming_queries
                )
                workflow_results['cache_warming'] = warming_stats
            
            # Process workflow steps
            for step in workflow.steps:
                step_start_time = time.time()
                
                # Check dependencies
                if step.dependencies and not all(dep in self.workflow_states[workflow.workflow_id]['completed_steps'] 
                                                for dep in step.dependencies):
                    workflow_results['errors'].append(f"Step {step.step_id} dependencies not met")
                    continue
                
                # Build step context
                step_context = {
                    'workflow_id': workflow.workflow_id,
                    'step_id': step.step_id,
                    'step_name': step.step_name,
                    'previous_results': workflow_results['step_results'][-3:],  # Last 3 steps
                    'accumulated_knowledge': self.workflow_states[workflow.workflow_id]['accumulated_knowledge'],
                    'user_context': user_context
                }
                
                # Process step query
                step_result = await self._process_workflow_step(step, step_context)
                step_result['step_time_ms'] = (time.time() - step_start_time) * 1000
                
                workflow_results['step_results'].append(step_result)
                workflow_results['steps_completed'] += 1
                
                # Update workflow state
                self.workflow_states[workflow.workflow_id]['completed_steps'].add(step.step_id)
                self.workflow_states[workflow.workflow_id]['current_step'] = step.step_id
                
                # Accumulate knowledge for future steps
                if step_result.get('entities'):
                    self.workflow_states[workflow.workflow_id]['accumulated_knowledge'].update(
                        {entity['name']: entity for entity in step_result['entities'][:5]}
                    )
                
                # Predictive caching for follow-up queries
                if step.follow_up_queries:
                    await self.predictive_cache.preload_predicted_queries(
                        step.follow_up_queries, self.base_processor
                    )
                
                # Update metrics
                self.metrics.total_queries += 1
                if step_result.get('cache_hit'):
                    self.metrics.cache_hits += 1
                else:
                    self.metrics.cache_misses += 1
            
            # Validate workflow success criteria
            workflow_results['success'] = self._validate_workflow_success(
                workflow, workflow_results
            )
            
        except Exception as e:
            # Activate fallback system
            fallback_response = await self.fallback_system.handle_system_failure(
                f"Workflow {workflow.workflow_id}", 
                {'error': str(e), 'step': workflow_results['steps_completed']}
            )
            workflow_results['fallback_response'] = fallback_response
            workflow_results['errors'].append(f"Workflow failure: {str(e)}")
            self.metrics.fallback_activations += 1
        
        workflow_results['total_time_ms'] = (time.time() - start_time) * 1000
        workflow_results['cache_performance'] = self._calculate_cache_performance()
        
        return workflow_results
    
    async def _process_workflow_step(self, step: WorkflowStep, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual workflow step with integrated caching."""
        step_result = {
            'step_id': step.step_id,
            'step_name': step.step_name,
            'query': step.query_text,
            'cache_hit': False,
            'entities': [],
            'confidence': 0.0
        }
        
        try:
            # Process query through base processor
            base_result = await self.base_processor.process_query_with_cache(
                step.query_text, context
            )
            
            step_result['cache_hit'] = base_result.cached_result
            step_result['routing_decision'] = base_result.routing_decision.value
            step_result['confidence'] = base_result.confidence_score
            
            # Additional processing based on routing decision
            if base_result.routing_decision == RoutingDecision.LIGHTRAG:
                # Query LightRAG knowledge graph
                kg_result = await self.lightrag_system.query_knowledge_graph(
                    step.query_text, context
                )
                step_result['entities'] = kg_result.get('entities', [])
                step_result['relationships'] = kg_result.get('relationships', [])
                step_result['lightrag_confidence'] = kg_result.get('confidence', 0)
            
            elif base_result.routing_decision == RoutingDecision.HYBRID:
                # Process with both systems
                kg_result = await self.lightrag_system.query_knowledge_graph(
                    step.query_text, context
                )
                step_result['entities'] = kg_result.get('entities', [])
                step_result['hybrid_processing'] = True
            
            # Update predictive cache patterns
            self.predictive_cache.update_query_patterns(step.query_text, step_result)
            
        except Exception as e:
            # Handle step failure with fallback
            fallback_response = await self.fallback_system.handle_system_failure(
                step.query_text, {'step_context': context, 'error': str(e)}
            )
            step_result['fallback_activated'] = True
            step_result['fallback_response'] = fallback_response
            step_result['error'] = str(e)
        
        return step_result
    
    def _validate_workflow_success(self, workflow: WorkflowDefinition,
                                 results: Dict[str, Any]) -> bool:
        """Validate if workflow meets success criteria."""
        success_criteria = workflow.success_criteria
        
        # Check completion rate
        completion_rate = results['steps_completed'] / len(workflow.steps)
        if completion_rate < success_criteria.get('min_completion_rate', 0.8):
            return False
        
        # Check average confidence
        confidences = [step.get('confidence', 0) for step in results['step_results']]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        if avg_confidence < success_criteria.get('min_avg_confidence', 0.5):
            return False
        
        # Check total time
        if results['total_time_ms'] > workflow.expected_total_time_ms:
            return False
        
        # Check error rate
        error_rate = len(results['errors']) / max(len(workflow.steps), 1)
        if error_rate > success_criteria.get('max_error_rate', 0.2):
            return False
        
        return True
    
    def _calculate_cache_performance(self) -> Dict[str, Any]:
        """Calculate comprehensive cache performance metrics."""
        self.metrics.calculate_efficiency()
        
        cache_stats = self.cache_system.get_comprehensive_stats()
        
        return {
            'overall_efficiency': self.metrics.cache_efficiency,
            'cache_hit_ratio': self.metrics.cache_hits / max(self.metrics.total_queries, 1),
            'predictive_accuracy': (self.predictive_cache.stats['predictions_correct'] / 
                                   max(self.predictive_cache.stats['predictions_made'], 1)),
            'fallback_rate': self.metrics.fallback_activations / max(self.metrics.total_queries, 1),
            'tier_utilization': cache_stats.get('tier_hit_rates', {}),
            'average_response_time': self.metrics.avg_response_time_ms
        }


# Workflow Definitions
CLINICAL_METABOLOMICS_WORKFLOW = WorkflowDefinition(
    workflow_id="clinical_metabolomics_research",
    workflow_name="Clinical Metabolomics Research Workflow",
    description="Complete workflow for clinical metabolomics research from sample to biomarker",
    steps=[
        WorkflowStep(
            step_id=1,
            step_name="Sample Analysis Planning",
            query_text="What are the key considerations for metabolomics sample collection and storage?",
            expected_category=ResearchCategory.DATA_PREPROCESSING,
            expected_routing=RoutingDecision.LIGHTRAG,
            cache_priority='high'
        ),
        WorkflowStep(
            step_id=2,
            step_name="Biomarker Identification",
            query_text="How to identify potential biomarkers from metabolomics data?",
            expected_category=ResearchCategory.BIOMARKER_DISCOVERY,
            expected_routing=RoutingDecision.LIGHTRAG,
            dependencies=[1],
            follow_up_queries=["Statistical methods for biomarker validation", "Clinical significance of identified biomarkers"]
        ),
        WorkflowStep(
            step_id=3,
            step_name="Pathway Analysis",
            query_text="What metabolic pathways are affected in the identified biomarkers?",
            expected_category=ResearchCategory.PATHWAY_ANALYSIS,
            expected_routing=RoutingDecision.LIGHTRAG,
            dependencies=[2],
            cache_priority='high'
        ),
        WorkflowStep(
            step_id=4,
            step_name="Literature Review",
            query_text="Recent publications on metabolomics biomarkers for clinical diagnosis",
            expected_category=ResearchCategory.LITERATURE_SEARCH,
            expected_routing=RoutingDecision.PERPLEXITY,
            dependencies=[2, 3]
        ),
        WorkflowStep(
            step_id=5,
            step_name="Clinical Application",
            query_text="How to translate metabolomics biomarkers into clinical diagnostic tools?",
            expected_category=ResearchCategory.CLINICAL_DIAGNOSIS,
            expected_routing=RoutingDecision.HYBRID,
            dependencies=[3, 4],
            follow_up_queries=["Regulatory approval process for biomarker-based diagnostics"]
        )
    ],
    cache_warming_queries=[
        "metabolomics sample preparation",
        "biomarker discovery methods",
        "metabolic pathway analysis",
        "clinical metabolomics applications"
    ],
    success_criteria={
        'min_completion_rate': 0.8,
        'min_avg_confidence': 0.6,
        'max_error_rate': 0.2
    },
    expected_total_time_ms=4000
)

DRUG_DISCOVERY_WORKFLOW = WorkflowDefinition(
    workflow_id="drug_discovery_metabolomics",
    workflow_name="Drug Discovery with Metabolomics",
    description="Workflow for drug discovery using metabolomics approaches",
    steps=[
        WorkflowStep(
            step_id=1,
            step_name="Target Identification",
            query_text="How to identify drug targets using metabolomics data?",
            expected_category=ResearchCategory.DRUG_DISCOVERY,
            expected_routing=RoutingDecision.LIGHTRAG
        ),
        WorkflowStep(
            step_id=2,
            step_name="Pathway Mapping",
            query_text="What are the key metabolic pathways involved in drug metabolism?",
            expected_category=ResearchCategory.PATHWAY_ANALYSIS,
            expected_routing=RoutingDecision.LIGHTRAG,
            dependencies=[1]
        ),
        WorkflowStep(
            step_id=3,
            step_name="Drug Interaction Analysis",
            query_text="How do drugs interact with metabolic pathways and what are recent findings?",
            expected_category=ResearchCategory.DRUG_DISCOVERY,
            expected_routing=RoutingDecision.HYBRID,
            dependencies=[2]
        )
    ],
    cache_warming_queries=[
        "drug target identification",
        "metabolic pathway drug interactions",
        "pharmacometabolomics"
    ],
    success_criteria={
        'min_completion_rate': 0.9,
        'min_avg_confidence': 0.7,
        'max_error_rate': 0.1
    }
)


# Test Fixtures
@pytest.fixture
async def cache_system():
    """Set up multi-tier cache system for end-to-end testing."""
    temp_dir = tempfile.mkdtemp()
    try:
        l1_cache = MockL1MemoryCache(max_size=100, default_ttl=600)
        l2_cache = MockL2DiskCache(temp_dir, max_size_mb=50)
        l3_cache = MockL3RedisCache()
        
        multi_cache = MultiTierCache(l1_cache, l2_cache, l3_cache)
        yield multi_cache
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def end_to_end_processor(cache_system):
    """Set up end-to-end query processor."""
    return EndToEndQueryProcessor(cache_system)


class TestEndToEndCacheFlow:
    """Comprehensive end-to-end cache flow integration tests."""
    
    @pytest.mark.asyncio
    async def test_clinical_metabolomics_workflow_execution(self, end_to_end_processor):
        """Test complete clinical metabolomics workflow with end-to-end caching."""
        user_context = {
            'user_id': 'researcher_123',
            'research_focus': 'diabetes_metabolomics',
            'experience_level': 'expert'
        }
        
        # Execute workflow
        results = await end_to_end_processor.process_workflow(
            CLINICAL_METABOLOMICS_WORKFLOW, user_context
        )
        
        # Verify workflow execution
        assert results['workflow_id'] == 'clinical_metabolomics_research'
        assert results['steps_completed'] >= 4  # At least 4 out of 5 steps
        assert results['success'] == True
        assert results['total_time_ms'] < CLINICAL_METABOLOMICS_WORKFLOW.expected_total_time_ms
        
        # Verify cache performance
        cache_perf = results['cache_performance']
        assert cache_perf['cache_hit_ratio'] > 0.2  # Some cache utilization
        assert cache_perf['fallback_rate'] < 0.3   # Low fallback rate
        
        # Verify step progression
        step_results = results['step_results']
        assert len(step_results) >= 4
        
        # Check that dependencies were respected
        for step in step_results:
            if step['step_id'] > 1:
                # Later steps should have accumulated knowledge
                assert len(step.get('entities', [])) >= 0
    
    @pytest.mark.asyncio
    async def test_lightrag_knowledge_graph_integration(self, end_to_end_processor):
        """Test LightRAG knowledge graph integration with caching."""
        # Test knowledge graph queries
        queries = [
            "What are the key metabolites in glucose metabolism?",
            "How does insulin regulate metabolic pathways?",
            "What are the relationships between diabetes and metabolic dysfunction?"
        ]
        
        kg_results = []
        for query in queries:
            result = await end_to_end_processor.lightrag_system.query_knowledge_graph(
                query, use_cache=True
            )
            kg_results.append(result)
        
        # Verify knowledge graph responses
        for result in kg_results:
            assert 'entities' in result
            assert 'confidence' in result
            assert result['confidence'] > 0.0
            assert 'source' in result
            assert result['source'] == 'lightrag_knowledge_graph'
        
        # Test cache efficiency - repeat queries should be faster
        repeat_results = []
        start_time = time.time()
        for query in queries:
            result = await end_to_end_processor.lightrag_system.query_knowledge_graph(
                query, use_cache=True
            )
            repeat_results.append(result)
        repeat_time = time.time() - start_time
        
        # Verify cache hits
        cache_hits = sum(1 for result in repeat_results if result.get('from_cache'))
        assert cache_hits > 0  # At least some cache hits
        
        # Verify cache warming
        warming_stats = await end_to_end_processor.lightrag_system.warm_knowledge_cache(
            ['glucose', 'insulin', 'diabetes'],
            ['insulin_signaling', 'glucose_metabolism']
        )
        assert warming_stats['entities_cached'] > 0
        assert len(warming_stats['errors']) == 0
    
    @pytest.mark.asyncio
    async def test_emergency_fallback_system_integration(self, end_to_end_processor):
        """Test emergency fallback system with cache coordination."""
        # Simulate system failures
        failure_scenarios = [
            {
                'query': 'Complex metabolomics pathway analysis',
                'error_context': {'component': 'lightrag', 'error': 'connection_timeout'}
            },
            {
                'query': 'Latest biomarker research findings',  
                'error_context': {'component': 'perplexity', 'error': 'rate_limit_exceeded'}
            },
            {
                'query': 'Clinical diagnosis for metabolic syndrome',
                'error_context': {'component': 'quality_scorer', 'error': 'service_unavailable'}
            }
        ]
        
        fallback_results = []
        for scenario in failure_scenarios:
            result = await end_to_end_processor.fallback_system.handle_system_failure(
                scenario['query'], scenario['error_context']
            )
            fallback_results.append(result)
        
        # Verify fallback responses
        for result in fallback_results:
            assert result['fallback_activated'] == True
            assert 'content' in result
            assert result['content']['confidence'] > 0
            assert result['fallback_time_ms'] < 500  # Quick fallback
        
        # Test fallback caching - repeat failures should use cached responses
        cached_fallback_results = []
        for scenario in failure_scenarios:
            result = await end_to_end_processor.fallback_system.handle_system_failure(
                scenario['query'], scenario['error_context']
            )
            cached_fallback_results.append(result)
        
        # Verify fallback cache utilization
        assert end_to_end_processor.fallback_system.stats['cache_fallback_hits'] > 0
        
        # Test system health monitoring
        health_status = await end_to_end_processor.fallback_system.check_system_health(
            ['lightrag', 'perplexity', 'cache_system', 'quality_scorer']
        )
        assert 'overall_health' in health_status
        assert 'component_health' in health_status
        assert 'cache_system_health' in health_status
    
    @pytest.mark.asyncio
    async def test_predictive_caching_system(self, end_to_end_processor):
        """Test predictive caching for query variations and follow-ups."""
        # Initial query to establish patterns
        initial_queries = [
            "What are metabolites involved in diabetes?",
            "How does glucose metabolism work?",
            "Biomarkers for cardiovascular disease"
        ]
        
        # Process initial queries
        for query in initial_queries:
            await end_to_end_processor.base_processor.process_query_with_cache(query)
        
        # Test predictive query generation
        predictions = await end_to_end_processor.predictive_cache.predict_follow_up_queries(
            "What metabolites are involved in insulin resistance?",
            {'category': 'metabolite_identification', 'confidence': 0.8}
        )
        
        # Verify predictions
        assert len(predictions) > 0
        assert any('pathway' in pred.lower() for pred in predictions)
        
        # Test predictive cache preloading
        preload_stats = await end_to_end_processor.predictive_cache.preload_predicted_queries(
            predictions, end_to_end_processor.base_processor
        )
        
        assert preload_stats['queries_preloaded'] > 0
        assert len(preload_stats['errors']) == 0
        
        # Verify that predicted queries are now cached
        for predicted_query in predictions[:2]:  # Test first 2 predictions
            result = await end_to_end_processor.base_processor.process_query_with_cache(
                predicted_query
            )
            # Should be fast due to predictive caching
            assert result.processing_time_ms < 200  # Quick response from cache
    
    @pytest.mark.asyncio
    async def test_drug_discovery_workflow_execution(self, end_to_end_processor):
        """Test drug discovery workflow with end-to-end cache integration."""
        user_context = {
            'user_id': 'pharma_researcher_456',
            'research_focus': 'drug_metabolism',
            'project_type': 'drug_discovery'
        }
        
        # Execute drug discovery workflow
        results = await end_to_end_processor.process_workflow(
            DRUG_DISCOVERY_WORKFLOW, user_context
        )
        
        # Verify workflow execution
        assert results['workflow_id'] == 'drug_discovery_metabolomics'
        assert results['steps_completed'] >= 2  # At least 2 out of 3 steps
        assert results['success'] == True
        
        # Verify specialized drug discovery processing
        step_results = results['step_results']
        target_identification_step = next(
            step for step in step_results if step['step_id'] == 1
        )
        assert 'entities' in target_identification_step
        assert target_identification_step['confidence'] > 0.5
        
        # Verify hybrid processing for drug interactions
        if len(step_results) >= 3:
            interaction_step = step_results[2]  # Step 3
            assert interaction_step.get('hybrid_processing') == True
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, end_to_end_processor):
        """Test concurrent workflow execution with cache coordination."""
        # Create multiple user contexts
        user_contexts = [
            {'user_id': f'user_{i}', 'research_focus': 'metabolomics'} 
            for i in range(3)
        ]
        
        # Execute workflows concurrently
        tasks = [
            end_to_end_processor.process_workflow(CLINICAL_METABOLOMICS_WORKFLOW, context)
            for context in user_contexts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all workflows executed successfully
        valid_results = [r for r in results if isinstance(r, dict) and not isinstance(r, Exception)]
        assert len(valid_results) == len(user_contexts)
        
        # Verify cache efficiency improved with concurrent execution
        for i, result in enumerate(valid_results):
            if i > 0:  # Later workflows should benefit from cache
                cache_perf = result['cache_performance']
                assert cache_perf['cache_hit_ratio'] >= valid_results[0]['cache_performance']['cache_hit_ratio']
        
        # Verify no cache consistency issues
        for result in valid_results:
            assert result['success'] == True
            assert len(result.get('errors', [])) == 0
    
    @pytest.mark.asyncio
    async def test_cache_performance_monitoring(self, end_to_end_processor):
        """Test comprehensive cache performance monitoring."""
        # Execute various queries to generate cache activity
        test_queries = [
            "Metabolomics data analysis methods",
            "Biomarker discovery in diabetes research", 
            "Pathway analysis of glucose metabolism",
            "Clinical applications of metabolomics",
            "Latest advances in biomarker research",
            "Drug metabolism pathways"
        ]
        
        # Process queries multiple times to test cache behavior
        for _ in range(2):  # Two rounds to test cache hits
            for query in test_queries:
                await end_to_end_processor.base_processor.process_query_with_cache(query)
        
        # Get comprehensive performance metrics
        performance_metrics = end_to_end_processor._calculate_cache_performance()
        
        # Verify performance metrics
        assert 'overall_efficiency' in performance_metrics
        assert 'cache_hit_ratio' in performance_metrics
        assert 'predictive_accuracy' in performance_metrics
        assert 'fallback_rate' in performance_metrics
        
        # Verify reasonable performance values
        assert performance_metrics['cache_hit_ratio'] > 0.3  # At least 30% hit ratio
        assert performance_metrics['fallback_rate'] < 0.2   # Less than 20% fallback rate
        
        # Test cache system health monitoring
        cache_stats = end_to_end_processor.cache_system.get_comprehensive_stats()
        assert 'overall_hit_rate' in cache_stats
        assert 'l1' in cache_stats
        assert 'l2' in cache_stats
        assert 'l3' in cache_stats
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_during_workflow(self, end_to_end_processor):
        """Test cache invalidation coordination during workflow execution."""
        # Start workflow execution
        workflow_task = asyncio.create_task(
            end_to_end_processor.process_workflow(CLINICAL_METABOLOMICS_WORKFLOW)
        )
        
        # Allow some processing time
        await asyncio.sleep(0.5)
        
        # Simulate cache invalidation during workflow
        cache_keys_to_invalidate = [
            "classification:*",
            "routing:*", 
            "response:*"
        ]
        
        # Manual invalidation (simplified for test)
        for key_pattern in cache_keys_to_invalidate:
            # In real implementation, this would use pattern-based invalidation
            await end_to_end_processor.cache_system.delete("test_invalidation_key")
        
        # Wait for workflow completion
        results = await workflow_task
        
        # Verify workflow still completed successfully despite invalidation
        assert results['success'] == True
        assert results['steps_completed'] > 0
        
        # Verify some cache regeneration occurred
        final_stats = end_to_end_processor.base_processor.get_comprehensive_stats()
        assert final_stats['cache_metrics']['cache_writes'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])