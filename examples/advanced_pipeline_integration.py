#!/usr/bin/env python3
"""
Advanced Pipeline Integration Example for CMO-LightRAG

This example demonstrates a hybrid approach that supports both Perplexity API
and LightRAG systems with configuration-driven switching, feature flags for 
gradual rollout, and integration with existing pipelines.py patterns.

Key Features:
- Hybrid system supporting both Perplexity and LightRAG
- Configuration-driven backend switching
- Feature flag support for A/B testing and gradual rollout
- Seamless fallback mechanisms
- Performance comparison and metrics collection
- Integration with existing pipeline patterns
- Cost optimization and budget management across systems

Usage:
    # Environment configuration
    export HYBRID_MODE="auto"  # auto, perplexity, lightrag, split
    export LIGHTRAG_ROLLOUT_PERCENTAGE="25"  # Percentage of traffic to LightRAG
    export ENABLE_PERFORMANCE_COMPARISON="true"
    export FALLBACK_TO_PERPLEXITY="true"
    
    # Run with Chainlit
    chainlit run examples/advanced_pipeline_integration.py
"""

import asyncio
import logging
import os
import random
import time
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
import json
from datetime import datetime, timedelta

import chainlit as cl
import requests
from lingua import LanguageDetector
from openai import OpenAI

# Import LightRAG integration components
from lightrag_integration import (
    create_clinical_rag_system,
    ClinicalMetabolomicsRAG,
    LightRAGConfig,
    QueryResponse,
    setup_lightrag_logging,
    get_integration_status,
    CostSummary
)

# Import existing CMO components
from src.translation import BaseTranslator, detect_language, get_language_detector, get_translator, translate
from src.lingua_iso_codes import IsoCode639_1

# Initialize logging
setup_lightrag_logging()
logger = logging.getLogger(__name__)


class QueryBackend(Enum):
    """Enumeration of available query backends."""
    PERPLEXITY = "perplexity"
    LIGHTRAG = "lightrag" 
    AUTO = "auto"
    SPLIT = "split"  # Use both for comparison


class HybridSystemConfig:
    """Configuration class for the hybrid system."""
    
    def __init__(self):
        self.mode = QueryBackend(os.getenv('HYBRID_MODE', 'auto'))
        self.lightrag_rollout_percentage = int(os.getenv('LIGHTRAG_ROLLOUT_PERCENTAGE', '25'))
        self.enable_performance_comparison = os.getenv('ENABLE_PERFORMANCE_COMPARISON', 'false').lower() == 'true'
        self.fallback_to_perplexity = os.getenv('FALLBACK_TO_PERPLEXITY', 'true').lower() == 'true'
        self.perplexity_api_key = os.getenv('PERPLEXITY_API')
        self.max_lightrag_cost_per_query = float(os.getenv('MAX_LIGHTRAG_COST_PER_QUERY', '0.10'))
        self.performance_log_file = os.getenv('PERFORMANCE_LOG_FILE', 'logs/hybrid_performance.jsonl')
        
        # Feature flags
        self.enable_cost_optimization = os.getenv('ENABLE_COST_OPTIMIZATION', 'true').lower() == 'true'
        self.enable_quality_scoring = os.getenv('ENABLE_QUALITY_SCORING', 'true').lower() == 'true'
        self.enable_automatic_switching = os.getenv('ENABLE_AUTOMATIC_SWITCHING', 'true').lower() == 'true'


class PerformanceMetrics:
    """Class to track and compare performance between backends."""
    
    def __init__(self):
        self.metrics = {
            'perplexity': {'queries': 0, 'total_time': 0, 'errors': 0, 'total_cost': 0},
            'lightrag': {'queries': 0, 'total_time': 0, 'errors': 0, 'total_cost': 0}
        }
    
    def record_query(self, backend: str, duration: float, cost: float = 0, error: bool = False):
        """Record query metrics for performance tracking."""
        if backend in self.metrics:
            self.metrics[backend]['queries'] += 1
            self.metrics[backend]['total_time'] += duration
            self.metrics[backend]['total_cost'] += cost
            if error:
                self.metrics[backend]['errors'] += 1
    
    def get_average_response_time(self, backend: str) -> float:
        """Get average response time for a backend."""
        metrics = self.metrics.get(backend, {})
        queries = metrics.get('queries', 0)
        return metrics.get('total_time', 0) / queries if queries > 0 else 0
    
    def get_error_rate(self, backend: str) -> float:
        """Get error rate for a backend."""
        metrics = self.metrics.get(backend, {})
        queries = metrics.get('queries', 0)
        errors = metrics.get('errors', 0)
        return errors / queries if queries > 0 else 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        for backend, metrics in self.metrics.items():
            if metrics['queries'] > 0:
                summary[backend] = {
                    'queries': metrics['queries'],
                    'avg_response_time': metrics['total_time'] / metrics['queries'],
                    'error_rate': metrics['errors'] / metrics['queries'],
                    'total_cost': metrics['total_cost'],
                    'avg_cost_per_query': metrics['total_cost'] / metrics['queries']
                }
        return summary


class HybridQueryProcessor:
    """
    Advanced hybrid query processor supporting multiple backends.
    
    This class intelligently routes queries between Perplexity and LightRAG
    based on configuration, performance metrics, cost considerations, and
    feature flags.
    """
    
    def __init__(self, config: HybridSystemConfig):
        """Initialize the hybrid processor with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.performance_metrics = PerformanceMetrics()
        
        # Initialize systems
        self.lightrag_system: Optional[ClinicalMetabolomicsRAG] = None
        self.perplexity_client: Optional[OpenAI] = None
        
        # Initialize Perplexity client if API key available
        if self.config.perplexity_api_key:
            self.perplexity_client = OpenAI(
                api_key=self.config.perplexity_api_key, 
                base_url="https://api.perplexity.ai"
            )
    
    async def initialize(self) -> bool:
        """Initialize both systems based on configuration."""
        try:
            # Initialize LightRAG system if needed
            if self.config.mode in [QueryBackend.LIGHTRAG, QueryBackend.AUTO, QueryBackend.SPLIT]:
                self.logger.info("Initializing LightRAG system...")
                self.lightrag_system = create_clinical_rag_system(
                    daily_budget_limit=float(os.getenv('LIGHTRAG_DAILY_BUDGET_LIMIT', '50.0')),
                    enable_quality_validation=self.config.enable_quality_scoring,
                    enable_cost_tracking=True,
                    relevance_confidence_threshold=0.75
                )
                
                await self.lightrag_system.initialize_rag()
                self.logger.info("LightRAG system initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid system: {e}")
            return False
    
    async def process_query(self, query: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query using the appropriate backend(s).
        
        Args:
            query: User query string
            session_data: Session context data
            
        Returns:
            Dict containing response, performance data, and metadata
        """
        start_time = time.time()
        
        try:
            # Determine which backend(s) to use
            backend_choice = await self._determine_backend(query, session_data)
            
            self.logger.info(f"Processing query with backend: {backend_choice}")
            
            # Process based on backend choice
            if backend_choice == QueryBackend.SPLIT:
                # Use both backends for comparison
                return await self._process_with_both_backends(query, session_data)
            elif backend_choice == QueryBackend.LIGHTRAG:
                return await self._process_with_lightrag(query, session_data)
            elif backend_choice == QueryBackend.PERPLEXITY:
                return await self._process_with_perplexity(query, session_data)
            else:
                # AUTO mode - intelligent selection
                return await self._process_with_auto_selection(query, session_data)
                
        except Exception as e:
            self.logger.error(f"Error in hybrid query processing: {e}")
            # Fallback to Perplexity if available and enabled
            if self.config.fallback_to_perplexity and self.perplexity_client:
                self.logger.info("Falling back to Perplexity API")
                return await self._process_with_perplexity(query, session_data)
            else:
                return {
                    "content": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                    "backend_used": "error",
                    "processing_time": time.time() - start_time,
                    "error": str(e)
                }
    
    async def _determine_backend(self, query: str, session_data: Dict[str, Any]) -> QueryBackend:
        """
        Determine which backend to use based on configuration and context.
        
        Args:
            query: User query
            session_data: Session context
            
        Returns:
            QueryBackend enum value
        """
        # If explicitly set, use that mode
        if self.config.mode != QueryBackend.AUTO:
            return self.config.mode
        
        # Automatic selection logic
        
        # Check rollout percentage for gradual deployment
        if random.randint(1, 100) <= self.config.lightrag_rollout_percentage:
            # User is in LightRAG rollout group
            
            # Check if LightRAG system is available and cost-effective
            if self.lightrag_system:
                # Estimate cost for this query
                estimated_cost = await self._estimate_lightrag_cost(query)
                if estimated_cost <= self.config.max_lightrag_cost_per_query:
                    return QueryBackend.LIGHTRAG
            
            # Fall back to Perplexity if LightRAG not suitable
            return QueryBackend.PERPLEXITY
        else:
            # Use Perplexity for users not in rollout
            return QueryBackend.PERPLEXITY
    
    async def _estimate_lightrag_cost(self, query: str) -> float:
        """Estimate the cost of processing a query with LightRAG."""
        # Rough estimation based on query length and system settings
        # This is a simplified estimation - real implementation would be more sophisticated
        query_length = len(query)
        base_cost = 0.001  # Base cost per query
        length_cost = query_length * 0.00001  # Cost per character
        return min(base_cost + length_cost, 0.05)  # Cap at 5 cents
    
    async def _process_with_lightrag(self, query: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using LightRAG system."""
        start_time = time.time()
        
        try:
            if not self.lightrag_system:
                raise RuntimeError("LightRAG system not initialized")
            
            response = await self.lightrag_system.query(
                query=query,
                mode="hybrid",
                include_metadata=True,
                enable_quality_scoring=self.config.enable_quality_scoring
            )
            
            # Get cost information
            cost_summary = await self.lightrag_system.get_cost_summary()
            processing_time = time.time() - start_time
            
            # Record performance metrics
            self.performance_metrics.record_query(
                'lightrag', 
                processing_time,
                cost_summary.daily_total if cost_summary else 0
            )
            
            # Format response
            formatted_response = await self._format_lightrag_response(response)
            formatted_response.update({
                "backend_used": "lightrag",
                "processing_time": processing_time,
                "cost_info": cost_summary.__dict__ if cost_summary else {},
                "performance_metrics": self.performance_metrics.get_summary()
            })
            
            # Log performance data if enabled
            if self.config.enable_performance_comparison:
                await self._log_performance_data(query, "lightrag", formatted_response)
            
            return formatted_response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_metrics.record_query('lightrag', processing_time, error=True)
            raise e
    
    async def _process_with_perplexity(self, query: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using Perplexity API."""
        start_time = time.time()
        
        try:
            if not self.perplexity_client:
                raise RuntimeError("Perplexity client not initialized")
            
            # Prepare Perplexity API request (from original main.py)
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in clinical metabolomics. You respond to "
                            "user queries in a helpful manner, with a focus on correct "
                            "scientific detail. Include peer-reviewed sources for all claims. "
                            "For each source/claim, provide a confidence score from 0.0-1.0, formatted as (confidence score: X.X) "
                            "Respond in a single paragraph, never use lists unless explicitly asked."
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                "temperature": 0.1,
                "search_domain_filter": ["-wikipedia.org"],
            }
            
            headers = {
                "Authorization": f"Bearer {self.config.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions", 
                json=payload, 
                headers=headers,
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                citations = response_data.get('citations', [])
                
                # Record performance metrics (estimated cost for Perplexity)
                estimated_cost = 0.01  # Rough estimate
                self.performance_metrics.record_query('perplexity', processing_time, estimated_cost)
                
                # Format response in consistent format
                formatted_response = await self._format_perplexity_response(content, citations)
                formatted_response.update({
                    "backend_used": "perplexity",
                    "processing_time": processing_time,
                    "cost_info": {"estimated_cost": estimated_cost},
                    "performance_metrics": self.performance_metrics.get_summary()
                })
                
                # Log performance data if enabled
                if self.config.enable_performance_comparison:
                    await self._log_performance_data(query, "perplexity", formatted_response)
                
                return formatted_response
                
            else:
                raise RuntimeError(f"Perplexity API error: {response.status_code}, {response.text}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_metrics.record_query('perplexity', processing_time, error=True)
            raise e
    
    async def _process_with_both_backends(self, query: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process query with both backends for comparison."""
        try:
            # Run both backends concurrently
            lightrag_task = asyncio.create_task(self._process_with_lightrag(query, session_data))
            perplexity_task = asyncio.create_task(self._process_with_perplexity(query, session_data))
            
            # Wait for both to complete
            lightrag_result, perplexity_result = await asyncio.gather(
                lightrag_task, perplexity_task, return_exceptions=True
            )
            
            # Determine which result to return as primary
            primary_result = lightrag_result if not isinstance(lightrag_result, Exception) else perplexity_result
            
            # Add comparison data
            comparison_data = {
                "comparison_mode": True,
                "lightrag_result": lightrag_result if not isinstance(lightrag_result, Exception) else {"error": str(lightrag_result)},
                "perplexity_result": perplexity_result if not isinstance(perplexity_result, Exception) else {"error": str(perplexity_result)},
                "performance_comparison": self._compare_results(lightrag_result, perplexity_result)
            }
            
            if isinstance(primary_result, Exception):
                raise primary_result
            
            primary_result.update(comparison_data)
            return primary_result
            
        except Exception as e:
            self.logger.error(f"Error in both-backend processing: {e}")
            raise e
    
    async def _process_with_auto_selection(self, query: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with automatic backend selection based on performance and cost."""
        # Analyze query characteristics
        query_complexity = self._analyze_query_complexity(query)
        
        # Check recent performance metrics
        lightrag_performance = self.performance_metrics.get_average_response_time('lightrag')
        perplexity_performance = self.performance_metrics.get_average_response_time('perplexity')
        
        # Make intelligent choice
        if query_complexity > 0.7 and self.lightrag_system:
            # Complex query - use LightRAG if available
            return await self._process_with_lightrag(query, session_data)
        elif perplexity_performance > 0 and perplexity_performance < lightrag_performance * 1.5:
            # Perplexity is significantly faster
            return await self._process_with_perplexity(query, session_data)
        elif self.lightrag_system:
            # Default to LightRAG if available
            return await self._process_with_lightrag(query, session_data)
        else:
            # Fall back to Perplexity
            return await self._process_with_perplexity(query, session_data)
    
    def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity to help with backend selection."""
        # Simple heuristics for query complexity
        complexity_score = 0.0
        
        # Length factor
        complexity_score += min(len(query) / 200, 0.3)
        
        # Scientific terms
        scientific_terms = ['metabolite', 'pathway', 'biomarker', 'metabolism', 'enzyme', 'protein']
        term_matches = sum(1 for term in scientific_terms if term.lower() in query.lower())
        complexity_score += min(term_matches / len(scientific_terms), 0.4)
        
        # Question complexity
        if any(word in query.lower() for word in ['compare', 'analyze', 'explain', 'mechanism']):
            complexity_score += 0.2
        
        if '?' in query:
            complexity_score += 0.1
        
        return min(complexity_score, 1.0)
    
    async def _format_lightrag_response(self, response: QueryResponse) -> Dict[str, Any]:
        """Format LightRAG response for consistent output."""
        content = response.response if hasattr(response, 'response') else str(response)
        
        citations = []
        if hasattr(response, 'metadata') and response.metadata:
            sources = response.metadata.get('sources', [])
            citations = [source.get('url', source.get('title', f'Source {i}')) 
                        for i, source in enumerate(sources, 1)]
        
        bibliography = self._format_bibliography_from_sources(citations)
        
        return {
            "content": content,
            "citations": citations,
            "bibliography": bibliography,
            "confidence_score": getattr(response, 'confidence_score', None),
            "source_count": len(citations)
        }
    
    async def _format_perplexity_response(self, content: str, citations: List[str]) -> Dict[str, Any]:
        """Format Perplexity response for consistent output."""
        # Extract confidence scores from content (from original main.py logic)
        import re
        pattern = r"confidence score:\s*([0-9.]+)(?:\s*\)\s*((?:\[\d+\]\s*)+)|\s+based on\s+(\[\d+\]))"
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        bibliography_dict = {}
        counter = 1
        for citation in citations:
            bibliography_dict[str(counter)] = [citation]
            counter += 1
        
        # Process confidence scores
        for score, refs1, refs2 in matches:
            confidence = score
            refs = refs1 if refs1 else refs2
            ref_nums = re.findall(r"\[(\d+)\]", refs)
            for num in ref_nums:
                if num in bibliography_dict:
                    bibliography_dict[num].append(confidence)
        
        # Clean content
        clean_pattern = r"\(\s*confidence score:\s*[0-9.]+\s*\)"
        content = re.sub(clean_pattern, "", content, flags=re.IGNORECASE)
        content = re.sub(r'\s+', ' ', content)
        
        bibliography = self._format_bibliography_from_dict(bibliography_dict)
        
        return {
            "content": content,
            "citations": citations,
            "bibliography": bibliography,
            "source_count": len(citations)
        }
    
    def _format_bibliography_from_sources(self, sources: List[str]) -> str:
        """Format bibliography from source list."""
        if not sources:
            return ""
        
        references = "\n\n\n**References:**\n"
        for i, source in enumerate(sources, 1):
            references += f"[{i}]: {source}\n"
        
        return references
    
    def _format_bibliography_from_dict(self, bibliography_dict: Dict[str, List]) -> str:
        """Format bibliography from dictionary (Perplexity format)."""
        if not bibliography_dict:
            return ""
        
        bibliography = ""
        references = "\n\n\n**References:**\n"
        further_reading = "\n**Further Reading:**\n"
        
        for key, value in bibliography_dict.items():
            if len(value) > 1:
                references += f"[{key}]: {value[0]} \n      (Confidence: {value[1]})\n"
            else:
                further_reading += f"[{key}]: {value[0]} \n"
        
        if references != "\n\n\n**References:**\n":
            bibliography += references
        if further_reading != "\n**Further Reading:**\n":
            bibliography += further_reading
        
        return bibliography
    
    def _compare_results(self, lightrag_result: Any, perplexity_result: Any) -> Dict[str, Any]:
        """Compare results from both backends."""
        comparison = {}
        
        if not isinstance(lightrag_result, Exception) and not isinstance(perplexity_result, Exception):
            comparison['response_times'] = {
                'lightrag': lightrag_result.get('processing_time', 0),
                'perplexity': perplexity_result.get('processing_time', 0)
            }
            
            comparison['source_counts'] = {
                'lightrag': lightrag_result.get('source_count', 0),
                'perplexity': perplexity_result.get('source_count', 0)
            }
            
            comparison['content_lengths'] = {
                'lightrag': len(lightrag_result.get('content', '')),
                'perplexity': len(perplexity_result.get('content', ''))
            }
            
            # Recommend better option
            if comparison['response_times']['lightrag'] < comparison['response_times']['perplexity']:
                comparison['recommendation'] = 'lightrag_faster'
            else:
                comparison['recommendation'] = 'perplexity_faster'
        
        return comparison
    
    async def _log_performance_data(self, query: str, backend: str, result: Dict[str, Any]):
        """Log performance data for analysis."""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'backend': backend,
                'query_length': len(query),
                'processing_time': result.get('processing_time', 0),
                'source_count': result.get('source_count', 0),
                'content_length': len(result.get('content', '')),
                'had_error': 'error' in result,
                'cost': result.get('cost_info', {}).get('estimated_cost', 0)
            }
            
            # Ensure log directory exists
            os.makedirs(os.path.dirname(self.config.performance_log_file), exist_ok=True)
            
            # Append to log file
            with open(self.config.performance_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.warning(f"Failed to log performance data: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'config': {
                'mode': self.config.mode.value,
                'rollout_percentage': self.config.lightrag_rollout_percentage,
                'performance_comparison': self.config.enable_performance_comparison,
                'fallback_enabled': self.config.fallback_to_perplexity
            },
            'systems': {
                'lightrag_available': self.lightrag_system is not None,
                'perplexity_available': self.perplexity_client is not None
            },
            'performance_metrics': self.performance_metrics.get_summary()
        }


# Integration with Chainlit - similar structure to basic integration but with hybrid processing

HYBRID_PROCESSOR: Optional[HybridQueryProcessor] = None


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authentication callback - unchanged."""
    if (username, password) == ("admin", "admin123") or (username, password) == ("testing", "ku9R_3"):
        return cl.User(
            identifier="admin",
            metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session with hybrid system."""
    global HYBRID_PROCESSOR
    
    try:
        # Initialize hybrid processor if not already done
        if HYBRID_PROCESSOR is None:
            config = HybridSystemConfig()
            HYBRID_PROCESSOR = HybridQueryProcessor(config)
            success = await HYBRID_PROCESSOR.initialize()
            
            if not success:
                logger.error("Failed to initialize hybrid system")
                await cl.Message(
                    content="⚠️ System initialization failed. Falling back to basic mode.",
                    author="CMO"
                ).send()
        
        # Store processor in session
        cl.user_session.set("hybrid_processor", HYBRID_PROCESSOR)
        
        # Display enhanced intro message with system info
        system_status = HYBRID_PROCESSOR.get_system_status()
        
        descr = 'Hello! Welcome to the Clinical Metabolomics Oracle (Advanced Hybrid System)'
        subhead = (f"I'm running in {system_status['config']['mode']} mode with both LightRAG and Perplexity API support. "
                  f"Current rollout: {system_status['config']['rollout_percentage']}% to LightRAG.\n\n"
                  f"System Status:\n"
                  f"• LightRAG: {'✅ Available' if system_status['systems']['lightrag_available'] else '❌ Unavailable'}\n"
                  f"• Perplexity: {'✅ Available' if system_status['systems']['perplexity_available'] else '❌ Unavailable'}\n"
                  f"• Performance Comparison: {'✅ Enabled' if system_status['config']['performance_comparison'] else '❌ Disabled'}\n\n"
                  f"To learn more, checkout the Readme page.")
        
        disclaimer = ('The Clinical Metabolomics Oracle is an automated question answering tool, and is not intended to replace the advice of a qualified healthcare professional.\n'
                     'Content generated by the Clinical Metabolomics Oracle is for informational purposes only, and is not advice for the treatment or diagnosis of any condition.')
        
        elements = [
            cl.Text(name=descr, content=subhead, display='inline'),
            cl.Text(name='Disclaimer', content=disclaimer, display='inline')
        ]
        
        await cl.Message(
            content='',
            elements=elements,
            author="CMO",
        ).send()

        # Continue with user agreement flow (same as basic integration)
        accepted = False
        while not accepted:
            res = await cl.AskActionMessage(
                content='Do you understand the purpose and limitations of the Clinical Metabolomics Oracle?',
                actions=[
                    cl.Action(name='I Understand', label='I Understand', description='Agree and continue', payload={"response": "agree"}),
                    cl.Action(name='Disagree', label='Disagree', description='Disagree to terms of service', payload={"response": "disagree"})
                ],
                timeout=300,
                author="CMO",
            ).send()

            accepted = res["label"] == "I Understand"
            if not accepted:
                await cl.Message(content="You must agree to the terms of service to continue.", author="CMO").send()

        welcome = "Welcome! Ask me anything about clinical metabolomics. I'll intelligently route your query to the best available system for optimal results."
        await cl.Message(content=welcome, author="CMO").send()

        # Set up translation components
        translator: BaseTranslator = get_translator()
        cl.user_session.set("translator", translator)
        await set_chat_settings(translator)

        iso_codes = [IsoCode639_1[code.upper()].value for code in translator.get_supported_languages(as_dict=True).values() if code.upper() in IsoCode639_1._member_names_]
        detector = get_language_detector(*iso_codes)
        cl.user_session.set("detector", detector)
        
        logger.info("Hybrid chat session initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during hybrid chat initialization: {e}")
        await cl.Message(content="⚠️ There was an error initializing the hybrid system. Please refresh and try again.", author="CMO").send()


async def set_chat_settings(translator):
    """Set up chat settings UI - enhanced with system controls."""
    initial_language_value = "Detect language"
    languages_to_iso_codes = translator.get_supported_languages(as_dict=True)
    language_values = [initial_language_value] + [language.title() for language in languages_to_iso_codes.keys()]
    
    await cl.ChatSettings([
        cl.input_widget.Select(
            id="translator",
            label="Translator",
            values=["Google", "OPUS-MT"],
            initial_value="Google",
        ),
        cl.input_widget.Select(
            id="language",
            label="Language",
            values=language_values,
            initial_value=initial_language_value,
        ),
        cl.input_widget.Select(
            id="backend_preference",
            label="Backend Preference",
            values=["Auto", "LightRAG", "Perplexity", "Compare Both"],
            initial_value="Auto",
        )
    ]).send()


@cl.author_rename
def rename(orig_author: str):
    """Author rename function."""
    rename_dict = {"Chatbot": "CMO"}
    return rename_dict.get(orig_author, orig_author)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle messages with hybrid processing."""
    start_time = time.time()
    
    try:
        # Get session components
        detector: LanguageDetector = cl.user_session.get("detector")
        translator: BaseTranslator = cl.user_session.get("translator")
        processor: HybridQueryProcessor = cl.user_session.get("hybrid_processor")
        
        if not processor:
            await cl.Message(content="⚠️ Hybrid system not properly initialized. Please refresh the page.", author="CMO").send()
            return
        
        content = message.content

        # Show thinking message with system info
        thinking_message = await cl.Message(content="🤔 Analyzing query and selecting optimal processing system...", author="CMO").send()

        # Handle language detection and translation
        language = cl.user_session.get("language")
        if not language or language == "auto":
            detection = await detect_language(detector, content)
            language = detection["language"]
        
        if language != "en" and language is not None:
            content = await translate(translator, content, source=language, target="en")

        # Process query using hybrid system
        session_data = {
            "language": language,
            "translator": translator,
            "detector": detector,
            "backend_preference": cl.user_session.get("backend_preference", "Auto")
        }
        
        response_data = await processor.process_query(content, session_data)
        
        # Update thinking message to show which system was used
        backend_used = response_data.get("backend_used", "unknown")
        processing_time = response_data.get("processing_time", 0)
        
        await thinking_message.update(content=f"✅ Processed using {backend_used.upper()} system in {processing_time:.2f}s")
        
        # Get response content and metadata
        response_content = response_data.get("content", "")
        bibliography = response_data.get("bibliography", "")
        
        # Handle translation back to user language
        if language != "en" and language is not None:
            response_content = await translate(translator, response_content, source="en", target=language)

        # Add performance and system information
        if bibliography:
            response_content += bibliography

        # Add system information footer
        end_time = time.time()
        system_info = f"\n\n*Processed by {backend_used.upper()} in {end_time - start_time:.2f}s*"
        
        # Add performance comparison if available
        if response_data.get("comparison_mode"):
            comparison = response_data.get("performance_comparison", {})
            if comparison:
                system_info += f"\n*Performance comparison: LightRAG {comparison.get('response_times', {}).get('lightrag', 0):.2f}s vs Perplexity {comparison.get('response_times', {}).get('perplexity', 0):.2f}s*"
        
        response_content += system_info

        # Send final response
        response_message = cl.Message(content=response_content)
        await response_message.send()
        
        logger.info(f"Hybrid message processed successfully in {end_time - start_time:.2f}s using {backend_used}")
        
    except Exception as e:
        logger.error(f"Error processing message with hybrid system: {e}")
        await cl.Message(content="I apologize, but I encountered an error processing your request. Please try again.", author="CMO").send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Handle settings updates including backend preference."""
    # Handle translator settings
    translator = settings["translator"]
    if translator == "Google":
        translator: BaseTranslator = get_translator("google")
    elif translator == "OPUS-MT":
        translator: BaseTranslator = get_translator("opusmt")
    
    await set_chat_settings(translator)
    cl.user_session.set("translator", translator)
    
    # Handle language settings
    language = settings["language"]
    if language == "Detect language":
        language = "auto"
    else:
        languages_to_iso_codes = translator.get_supported_languages(as_dict=True)
        language = languages_to_iso_codes.get(language.lower(), "auto")
    
    cl.user_session.set("language", language)
    
    # Handle backend preference
    backend_preference = settings.get("backend_preference", "Auto")
    cl.user_session.set("backend_preference", backend_preference)
    
    logger.info(f"Settings updated: backend_preference={backend_preference}")


# Development and testing utilities

async def test_hybrid_system():
    """Test function to verify hybrid system works correctly."""
    print("Testing hybrid system integration...")
    
    try:
        config = HybridSystemConfig()
        processor = HybridQueryProcessor(config)
        success = await processor.initialize()
        
        if not success:
            print("❌ Hybrid system initialization failed")
            return False
        
        # Test query processing with different backends
        test_query = "What are the main metabolites involved in glucose metabolism?"
        session_data = {"language": "en", "translator": None, "detector": None}
        
        print(f"Testing query: {test_query}")
        
        # Test auto mode
        result = await processor.process_query(test_query, session_data)
        
        if result.get("error"):
            print(f"❌ Query processing failed: {result['error']}")
            return False
        
        backend_used = result.get("backend_used")
        processing_time = result.get("processing_time", 0)
        
        print(f"✅ Hybrid system test successful!")
        print(f"   - Backend used: {backend_used}")
        print(f"   - Response length: {len(result.get('content', ''))}")
        print(f"   - Citations: {len(result.get('citations', []))}")
        print(f"   - Processing time: {processing_time:.2f}s")
        
        # Show system status
        status = processor.get_system_status()
        print(f"   - System status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid system test failed: {e}")
        return False


if __name__ == "__main__":
    """Main entry point for testing or running the integration."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_hybrid_system())
    else:
        print("🔬 Clinical Metabolomics Oracle - Advanced Hybrid Integration")
        print("=" * 65)
        config = HybridSystemConfig()
        print("Configuration:")
        print(f"  Mode: {config.mode.value}")
        print(f"  LightRAG Rollout: {config.lightrag_rollout_percentage}%")
        print(f"  Performance Comparison: {config.enable_performance_comparison}")
        print(f"  Fallback to Perplexity: {config.fallback_to_perplexity}")
        print(f"  Cost Optimization: {config.enable_cost_optimization}")
        print("\nTo run: chainlit run examples/advanced_pipeline_integration.py")
        print("To test: python examples/advanced_pipeline_integration.py test")