#!/usr/bin/env python3
"""
Main Integration: Integration patterns for main.py with LightRAG feature flags.

This module provides example integration patterns showing how to modify the existing
main.py file to support the LightRAG feature flag system. It demonstrates how to
maintain backward compatibility while adding the new routing capabilities.

Key Integration Points:
- Minimal changes to existing main.py structure
- Seamless fallback to existing Perplexity implementation
- Enhanced logging and monitoring integration
- Support for A/B testing and performance comparison
- Gradual rollout management integration
- Quality assessment and validation

Integration Approach:
- Factory pattern for service creation based on feature flags
- Adapter pattern to maintain existing API interfaces
- Strategy pattern for routing decisions
- Observer pattern for metrics collection

This file serves as documentation and example code showing how to integrate
the feature flag system with minimal disruption to the existing codebase.

Author: Claude Code (Anthropic)
Created: 2025-08-08
Version: 1.0.0
"""

import asyncio
import logging
import os
import sys
import time
from typing import Optional, Dict, Any, List
import chainlit as cl
from datetime import datetime

# Import existing modules from main.py
from lingua import LanguageDetector
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine.types import BaseChatEngine

# Import our feature flag system modules
from .config import LightRAGConfig
from .integration_wrapper import (
    IntegratedQueryService, QueryRequest, ServiceResponse, ResponseType
)
from .feature_flag_manager import FeatureFlagManager, RoutingContext
from .rollout_manager import RolloutManager


class EnhancedClinicalMetabolomicsOracle:
    """
    Enhanced Clinical Metabolomics Oracle with LightRAG integration.
    
    This class provides a drop-in replacement for the existing query processing
    logic in main.py, adding LightRAG routing capabilities while maintaining
    full backward compatibility.
    """
    
    def __init__(self, perplexity_api_key: str, lightrag_config: Optional[LightRAGConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the Enhanced Clinical Metabolomics Oracle.
        
        Args:
            perplexity_api_key: API key for Perplexity service
            lightrag_config: Optional LightRAG configuration (auto-created if None)
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize configuration
        self.lightrag_config = lightrag_config or self._create_default_config()
        
        # Initialize integrated query service
        self.query_service = IntegratedQueryService(
            config=self.lightrag_config,
            perplexity_api_key=perplexity_api_key,
            logger=self.logger
        )
        
        # Initialize rollout manager if enabled
        self.rollout_manager: Optional[RolloutManager] = None
        if (self.lightrag_config.lightrag_integration_enabled and 
            self.lightrag_config.lightrag_enable_performance_comparison):
            
            self.rollout_manager = RolloutManager(
                config=self.lightrag_config,
                feature_manager=self.query_service.feature_manager,
                logger=self.logger
            )
            
            # Load existing rollout state if available
            self.rollout_manager.load_state()
        
        # Quality assessment setup
        if self.lightrag_config.lightrag_enable_quality_metrics:
            self.query_service.set_quality_assessor(self._assess_response_quality)
        
        # Session tracking for consistent user experience
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Enhanced Clinical Metabolomics Oracle initialized")
    
    def _create_default_config(self) -> LightRAGConfig:
        """
        Create default LightRAG configuration from environment variables.
        
        Returns:
            LightRAGConfig instance with default settings
        """
        try:
            return LightRAGConfig.get_config()
        except Exception as e:
            self.logger.warning(f"Failed to create LightRAG config: {e}. Using fallback configuration.")
            
            # Create minimal fallback configuration
            config = LightRAGConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                lightrag_integration_enabled=False,  # Disable integration on config failure
                auto_create_dirs=False
            )
            return config
    
    def _assess_response_quality(self, response: ServiceResponse) -> Dict[str, float]:
        """
        Assess response quality for monitoring and routing decisions.
        
        This is a simplified quality assessment that can be enhanced with
        more sophisticated methods like the existing relevance scorer.
        
        Args:
            response: ServiceResponse to assess
        
        Returns:
            Dictionary of quality scores (0.0-1.0)
        """
        try:
            quality_scores = {}
            
            # Basic content quality metrics
            content = response.content.strip()
            
            # Length-based quality (reasonable length indicates substantive response)
            min_length, max_length = 100, 5000
            length_score = min(1.0, max(0.0, (len(content) - min_length) / (max_length - min_length)))
            quality_scores['length'] = length_score
            
            # Citation quality (presence of citations/references)
            citation_score = 1.0 if response.citations else 0.5
            quality_scores['citations'] = citation_score
            
            # Confidence score quality (from existing patterns)
            if response.confidence_scores:
                avg_confidence = sum(response.confidence_scores.values()) / len(response.confidence_scores)
                quality_scores['confidence'] = avg_confidence
            else:
                quality_scores['confidence'] = 0.7  # Default moderate confidence
            
            # Response time quality (faster responses score higher)
            if response.processing_time > 0:
                # Score 1.0 for responses under 5 seconds, decreasing linearly to 0.5 at 30 seconds
                time_score = max(0.5, min(1.0, 1.0 - (response.processing_time - 5) / 25))
                quality_scores['response_time'] = time_score
            else:
                quality_scores['response_time'] = 0.8  # Default for cached responses
            
            # Content structure quality (paragraphs, sentences)
            sentences = content.count('.') + content.count('!') + content.count('?')
            paragraphs = content.count('\n\n') + 1
            
            if sentences > 0:
                structure_score = min(1.0, (sentences / paragraphs) / 10)  # Expect ~5-10 sentences per paragraph
                quality_scores['structure'] = structure_score
            else:
                quality_scores['structure'] = 0.3
            
            # Error indicator quality (absence of error messages)
            error_indicators = ['error', 'failed', 'unable', 'sorry', 'apologize', 'technical difficulties']
            error_count = sum(1 for indicator in error_indicators if indicator.lower() in content.lower())
            error_score = max(0.0, 1.0 - (error_count * 0.2))
            quality_scores['error_free'] = error_score
            
            return quality_scores
            
        except Exception as e:
            self.logger.warning(f"Quality assessment error: {e}")
            # Return default quality scores on error
            return {
                'length': 0.5,
                'citations': 0.5,
                'confidence': 0.5,
                'response_time': 0.5,
                'structure': 0.5,
                'error_free': 0.5
            }
    
    async def process_query(self, content: str, user_session: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query with intelligent LightRAG/Perplexity routing.
        
        This method provides a drop-in replacement for the existing query processing
        logic in main.py, with enhanced routing capabilities.
        
        Args:
            content: User query text
            user_session: Optional session information from Chainlit
        
        Returns:
            Dictionary containing response content, citations, timing, and metadata
        """
        start_time = time.time()
        
        try:
            # Extract user/session identification
            user_id = None
            session_id = None
            
            if user_session:
                user_id = user_session.get('user_id') or user_session.get('identifier')
                session_id = user_session.get('session_id')
                
                # Store session metadata for consistency
                if user_id or session_id:
                    key = user_id or session_id
                    if key not in self._session_metadata:
                        self._session_metadata[key] = {
                            'first_query_time': datetime.now(),
                            'query_count': 0,
                            'routing_history': []
                        }
                    
                    self._session_metadata[key]['query_count'] += 1
            
            # Create query request
            query_request = QueryRequest(
                query_text=content,
                user_id=user_id,
                session_id=session_id,
                timeout_seconds=self.lightrag_config.lightrag_integration_timeout_seconds,
                context_metadata={
                    'timestamp': datetime.now().isoformat(),
                    'session_info': user_session or {}
                }
            )
            
            # Execute query with intelligent routing
            response = await self.query_service.query_async(query_request)
            
            # Record result for rollout monitoring
            if self.rollout_manager:
                self.rollout_manager.record_request_result(
                    success=response.is_success,
                    quality_score=response.average_quality_score,
                    error_details=response.error_details
                )
            
            # Update session metadata
            if user_id or session_id:
                key = user_id or session_id
                if key in self._session_metadata:
                    self._session_metadata[key]['routing_history'].append({
                        'timestamp': datetime.now().isoformat(),
                        'service': response.response_type.value,
                        'success': response.is_success,
                        'processing_time': response.processing_time,
                        'quality_score': response.average_quality_score
                    })
                    
                    # Keep only last 10 entries per session
                    if len(self._session_metadata[key]['routing_history']) > 10:
                        self._session_metadata[key]['routing_history'] = \
                            self._session_metadata[key]['routing_history'][-10:]
            
            # Format response for main.py compatibility
            total_time = time.time() - start_time
            
            # Process citations for bibliography format (matching existing pattern)
            bibliography = self._format_bibliography(response.citations, response.confidence_scores)
            
            # Prepare final content
            final_content = response.content
            if bibliography:
                final_content += bibliography
            
            # Add timing information (matching existing pattern)
            final_content += f"\n\n*{total_time:.2f} seconds*"
            
            return {
                'content': final_content,
                'citations': response.citations or [],
                'confidence_scores': response.confidence_scores or {},
                'processing_time': total_time,
                'service_used': response.response_type.value,
                'quality_scores': response.quality_scores,
                'routing_metadata': response.metadata,
                'success': response.is_success,
                'error_details': response.error_details
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Query processing error: {str(e)}"
            self.logger.error(error_msg)
            
            # Record failure for rollout monitoring
            if self.rollout_manager:
                self.rollout_manager.record_request_result(
                    success=False,
                    error_details=error_msg
                )
            
            # Return fallback response
            return {
                'content': "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                'citations': [],
                'confidence_scores': {},
                'processing_time': total_time,
                'service_used': 'fallback',
                'quality_scores': None,
                'routing_metadata': {'error': error_msg},
                'success': False,
                'error_details': error_msg
            }
    
    def _format_bibliography(self, citations: Optional[List[Dict[str, Any]]], 
                           confidence_scores: Optional[Dict[str, float]]) -> str:
        """
        Format citations into bibliography format matching existing main.py pattern.
        
        Args:
            citations: List of citation dictionaries
            confidence_scores: Confidence scores by citation
        
        Returns:
            Formatted bibliography string
        """
        if not citations:
            return ""
        
        bibliography_dict = {}
        
        # Build bibliography mapping (matching existing pattern)
        for i, citation in enumerate(citations, 1):
            citation_url = citation if isinstance(citation, str) else citation.get('url', str(citation))
            confidence = confidence_scores.get(citation_url) if confidence_scores else None
            
            if confidence is not None:
                bibliography_dict[str(i)] = [citation_url, confidence]
            else:
                bibliography_dict[str(i)] = [citation_url]
        
        # Format bibliography (matching existing pattern)
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
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for monitoring and debugging.
        
        Returns:
            Dictionary containing system status information
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'lightrag_integration_enabled': self.lightrag_config.lightrag_integration_enabled,
            'service_performance': self.query_service.get_performance_summary(),
            'session_count': len(self._session_metadata),
            'config_summary': {
                'rollout_percentage': self.lightrag_config.lightrag_rollout_percentage,
                'ab_testing_enabled': self.lightrag_config.lightrag_enable_ab_testing,
                'circuit_breaker_enabled': self.lightrag_config.lightrag_enable_circuit_breaker,
                'quality_metrics_enabled': self.lightrag_config.lightrag_enable_quality_metrics,
                'fallback_to_perplexity': self.lightrag_config.lightrag_fallback_to_perplexity
            }
        }
        
        # Add rollout status if available
        if self.rollout_manager:
            rollout_status = self.rollout_manager.get_rollout_status()
            status['rollout_status'] = rollout_status
        
        return status
    
    def start_rollout(self, strategy: str = "linear", **kwargs) -> Optional[str]:
        """
        Start a new rollout with the specified strategy.
        
        Args:
            strategy: Rollout strategy ('linear', 'exponential', 'canary')
            **kwargs: Additional parameters for rollout configuration
        
        Returns:
            Rollout ID if started, None if rollout manager not available
        """
        if not self.rollout_manager:
            self.logger.warning("Rollout manager not available")
            return None
        
        try:
            if strategy == "linear":
                config = self.rollout_manager.create_linear_rollout(
                    start_percentage=kwargs.get('start_percentage', 5.0),
                    increment=kwargs.get('increment', 10.0),
                    stage_duration_minutes=kwargs.get('stage_duration', 60),
                    final_percentage=kwargs.get('final_percentage', 100.0)
                )
            elif strategy == "exponential":
                config = self.rollout_manager.create_exponential_rollout(
                    start_percentage=kwargs.get('start_percentage', 1.0),
                    stage_duration_minutes=kwargs.get('stage_duration', 60),
                    final_percentage=kwargs.get('final_percentage', 100.0)
                )
            elif strategy == "canary":
                config = self.rollout_manager.create_canary_rollout(
                    canary_percentage=kwargs.get('canary_percentage', 1.0),
                    canary_duration_minutes=kwargs.get('canary_duration', 120),
                    full_percentage=kwargs.get('full_percentage', 100.0)
                )
            else:
                raise ValueError(f"Unknown rollout strategy: {strategy}")
            
            rollout_id = self.rollout_manager.start_rollout(config)
            self.logger.info(f"Started {strategy} rollout: {rollout_id}")
            return rollout_id
            
        except Exception as e:
            self.logger.error(f"Failed to start rollout: {e}")
            return None


# Integration examples for main.py modification
def create_enhanced_oracle(perplexity_api_key: str) -> EnhancedClinicalMetabolomicsOracle:
    """
    Factory function to create Enhanced Clinical Metabolomics Oracle.
    
    This can be used as a drop-in replacement for the existing query processing
    logic in main.py.
    
    Args:
        perplexity_api_key: API key for Perplexity service
    
    Returns:
        EnhancedClinicalMetabolomicsOracle instance
    """
    return EnhancedClinicalMetabolomicsOracle(perplexity_api_key)


# Example integration patterns for main.py
class MainIntegrationHelper:
    """Helper class demonstrating integration patterns for main.py."""
    
    @staticmethod
    def setup_enhanced_oracle() -> EnhancedClinicalMetabolomicsOracle:
        """
        Set up enhanced oracle with configuration from environment.
        
        Returns:
            Configured EnhancedClinicalMetabolomicsOracle
        """
        perplexity_api_key = os.environ.get("PERPLEXITY_API")
        if not perplexity_api_key:
            raise ValueError("PERPLEXITY_API environment variable is required")
        
        return create_enhanced_oracle(perplexity_api_key)
    
    @staticmethod
    async def enhanced_on_message_handler(message: 'cl.Message', 
                                        oracle: EnhancedClinicalMetabolomicsOracle) -> None:
        """
        Example enhanced message handler for Chainlit integration.
        
        This shows how to modify the existing on_message handler in main.py
        to use the enhanced oracle with LightRAG routing.
        
        Args:
            message: Chainlit message object
            oracle: EnhancedClinicalMetabolomicsOracle instance
        """
        # Get user session information
        user_session = {
            'user_id': cl.user_session.get("user_id"),
            'session_id': cl.user_session.get("session_id"),
            'user': cl.user_session.get("user")
        }
        
        # Show thinking message
        await cl.Message(content="Thinking...", author="CMO").send()
        
        # Process query with enhanced oracle
        result = await oracle.process_query(message.content, user_session)
        
        # Create response message
        response_message = cl.Message(content=result['content'])
        
        # Add metadata for debugging/monitoring (if needed)
        if result.get('routing_metadata'):
            routing_info = result['routing_metadata']
            service_used = result.get('service_used', 'unknown')
            
            # Optionally add service information to response
            if oracle.lightrag_config.lightrag_enable_performance_comparison:
                response_message.content += f"\n\n*Processed via {service_used}*"
        
        await response_message.send()
    
    @staticmethod
    def create_system_monitor_endpoint(oracle: EnhancedClinicalMetabolomicsOracle) -> Dict[str, Any]:
        """
        Create a system monitoring endpoint for health checks and debugging.
        
        Args:
            oracle: EnhancedClinicalMetabolomicsOracle instance
        
        Returns:
            System status dictionary
        """
        return oracle.get_system_status()


# Example environment variable configuration
EXAMPLE_ENV_VARS = """
# Basic LightRAG configuration
OPENAI_API_KEY=your_openai_api_key
PERPLEXITY_API=your_perplexity_api_key

# Feature flag configuration
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=10.0
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true

# Quality and performance settings
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.7
LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS=30.0

# Circuit breaker settings
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0

# Advanced routing settings
LIGHTRAG_ENABLE_CONDITIONAL_ROUTING=false
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true
LIGHTRAG_USER_HASH_SALT=your_unique_salt_value
"""


def print_integration_guide():
    """Print integration guide for main.py modification."""
    print("""
    LightRAG Integration Guide for main.py
    =====================================
    
    1. Import the integration module:
       from lightrag_integration.main_integration import create_enhanced_oracle
    
    2. Replace existing Perplexity logic with enhanced oracle:
       # Initialize at startup
       oracle = create_enhanced_oracle(PERPLEXITY_API)
    
    3. Modify the on_message handler:
       @cl.on_message
       async def on_message(message: cl.Message):
           user_session = {
               'user_id': cl.user_session.get("user_id"),
               'session_id': cl.user_session.get("session_id")
           }
           
           result = await oracle.process_query(message.content, user_session)
           
           response_message = cl.Message(content=result['content'])
           await response_message.send()
    
    4. Optional: Add system monitoring endpoint:
       def get_system_status():
           return oracle.get_system_status()
    
    5. Environment variables:
       Set LIGHTRAG_INTEGRATION_ENABLED=true to enable LightRAG routing
       Configure other feature flags as needed
    
    The integration maintains full backward compatibility - if LightRAG is
    disabled or fails, the system automatically falls back to Perplexity.
    """)


if __name__ == "__main__":
    print_integration_guide()