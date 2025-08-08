#!/usr/bin/env python3
"""
Main Integration Example: Feature Flag Integration with CMO Chatbot

This example demonstrates how to integrate the LightRAG feature flag system
with the main CMO chatbot application, showing practical patterns for:

- Feature flag routing in message handlers
- Graceful fallback to Perplexity when LightRAG is disabled
- Performance monitoring and quality scoring
- Circuit breaker integration for error handling
- Chainlit message patterns with feature flags

This example follows the patterns from src/main.py and shows how to
enhance the existing chatbot with intelligent routing capabilities.

Author: Claude Code (Anthropic)
Created: 2025-08-08
Version: 1.0.0
"""

import os
import sys
import time
import logging
import asyncio
import re
from typing import Optional, Dict, Any, Tuple
import chainlit as cl
import requests
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LightRAG integration imports
from lightrag_integration import (
    LightRAGConfig,
    FeatureFlagManager,
    RolloutManager,
    RoutingContext,
    UserCohort,
    RoutingDecision,
    EnhancedResponseQualityAssessor,
    RelevanceScorer,
    create_clinical_rag_system_with_features
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration and managers
config: Optional[LightRAGConfig] = None
feature_manager: Optional[FeatureFlagManager] = None
rollout_manager: Optional[RolloutManager] = None
quality_assessor: Optional[EnhancedResponseQualityAssessor] = None
relevance_scorer: Optional[RelevanceScorer] = None
lightrag_system: Optional[Any] = None

# Perplexity API setup (following main.py pattern)
PERPLEXITY_API = os.environ.get("PERPLEXITY_API")
if PERPLEXITY_API:
    perplexity_client = OpenAI(api_key=PERPLEXITY_API, base_url="https://api.perplexity.ai")


async def initialize_feature_flag_system() -> bool:
    """
    Initialize the feature flag system with configuration.
    
    Returns:
        True if initialization successful, False otherwise
    """
    global config, feature_manager, rollout_manager, quality_assessor, relevance_scorer, lightrag_system
    
    try:
        # Load configuration from environment or defaults
        config = LightRAGConfig()
        
        # Set example configuration values
        config.lightrag_integration_enabled = os.getenv('LIGHTRAG_INTEGRATION_ENABLED', 'true').lower() == 'true'
        config.lightrag_rollout_percentage = float(os.getenv('LIGHTRAG_ROLLOUT_PERCENTAGE', '10.0'))
        config.lightrag_enable_ab_testing = os.getenv('LIGHTRAG_ENABLE_AB_TESTING', 'true').lower() == 'true'
        config.lightrag_enable_circuit_breaker = os.getenv('LIGHTRAG_ENABLE_CIRCUIT_BREAKER', 'true').lower() == 'true'
        config.lightrag_enable_quality_metrics = os.getenv('LIGHTRAG_ENABLE_QUALITY_METRICS', 'true').lower() == 'true'
        config.lightrag_min_quality_threshold = float(os.getenv('LIGHTRAG_MIN_QUALITY_THRESHOLD', '0.7'))
        config.lightrag_circuit_breaker_failure_threshold = int(os.getenv('LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5'))
        config.lightrag_circuit_breaker_recovery_timeout = int(os.getenv('LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '300'))
        config.lightrag_user_hash_salt = os.getenv('LIGHTRAG_USER_HASH_SALT', 'cmo_chatbot_salt')
        
        # Initialize feature flag manager
        feature_manager = FeatureFlagManager(config, logger)
        
        # Initialize rollout manager
        rollout_manager = RolloutManager(config, feature_manager, logger)
        
        # Initialize quality assessment components if enabled
        if config.lightrag_enable_quality_metrics:
            quality_assessor = EnhancedResponseQualityAssessor()
            relevance_scorer = RelevanceScorer()
        
        # Initialize LightRAG system if integration is enabled
        if config.lightrag_integration_enabled:
            try:
                lightrag_system = create_clinical_rag_system_with_features()
                logger.info("LightRAG system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LightRAG system: {e}")
                lightrag_system = None
        
        logger.info("Feature flag system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize feature flag system: {e}")
        return False


def create_routing_context(message: cl.Message, user_id: Optional[str] = None) -> RoutingContext:
    """
    Create routing context from Chainlit message.
    
    Args:
        message: Chainlit message object
        user_id: Optional user identifier
    
    Returns:
        RoutingContext for feature flag decision
    """
    # Extract query characteristics for routing decisions
    query_text = message.content
    query_length = len(query_text)
    
    # Simple complexity scoring based on query characteristics
    complexity_indicators = [
        len(re.findall(r'\b\w+\b', query_text)),  # Word count
        len(re.findall(r'[?]', query_text)),       # Question marks
        len(re.findall(r'\b(and|or|but|because|therefore|however)\b', query_text.lower())),  # Connectors
        len(re.findall(r'\b(metabol|clinic|biomark|pathway|compound)\w*\b', query_text.lower()))  # Domain terms
    ]
    
    # Normalize complexity score (0-1)
    query_complexity = min(sum(complexity_indicators) / 20.0, 1.0)
    
    # Determine query type based on content
    query_type = "general"
    if re.search(r'\b(metabol\w*|compound\w*|pathway\w*)\b', query_text.lower()):
        query_type = "metabolomics"
    elif re.search(r'\b(clinic\w*|patient\w*|diagnos\w*)\b', query_text.lower()):
        query_type = "clinical"
    elif re.search(r'\b(research\w*|study\w*|paper\w*)\b', query_text.lower()):
        query_type = "research"
    
    return RoutingContext(
        user_id=user_id,
        session_id=cl.user_session.get("id"),
        query_text=query_text,
        query_type=query_type,
        query_complexity=query_complexity,
        metadata={
            'query_length': query_length,
            'message_id': getattr(message, 'id', None),
            'channel': 'chainlit'
        }
    )


async def query_lightrag_system(query: str, context: RoutingContext) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Query the LightRAG system with error handling and monitoring.
    
    Args:
        query: User query text
        context: Routing context for the query
    
    Returns:
        Tuple of (response_content, response_time_seconds, error_message)
    """
    if not lightrag_system:
        return None, 0.0, "LightRAG system not initialized"
    
    start_time = time.time()
    
    try:
        # Mock LightRAG query - replace with actual implementation
        await asyncio.sleep(0.5)  # Simulate processing time
        
        response_content = f"""Based on clinical metabolomics research, here's what I found regarding your query: "{query[:100]}..."

This is a LightRAG-powered response that would analyze our comprehensive knowledge base of clinical metabolomics literature to provide you with precise, evidence-based information.

Key findings:
• Relevant metabolic pathways and biomarkers identified
• Clinical significance and therapeutic implications
• Supporting evidence from peer-reviewed studies
• Quality-assessed information with confidence scoring

[This is a mock response for demonstration purposes - actual LightRAG integration would provide real analysis]"""
        
        response_time = time.time() - start_time
        return response_content, response_time, None
        
    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"LightRAG query failed: {str(e)}"
        logger.error(error_msg)
        return None, response_time, error_msg


async def query_perplexity_system(query: str, context: RoutingContext) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Query the Perplexity system (existing implementation from main.py).
    
    Args:
        query: User query text
        context: Routing context for the query
    
    Returns:
        Tuple of (response_content, response_time_seconds, error_message)
    """
    if not PERPLEXITY_API or not perplexity_client:
        return None, 0.0, "Perplexity API not configured"
    
    start_time = time.time()
    
    try:
        # Perplexity API call (following main.py pattern)
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
            "search_domain_filter": [
                "-wikipedia.org",
            ],
        }
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API}",
            "Content-Type": "application/json"
        }
        
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               json=payload, headers=headers)
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            citations = response_data.get('citations', [])
            
            # Process citations and confidence scores (from main.py)
            bibliography = ""
            if citations:
                bibliography_dict = {}
                counter = 1
                for citation in citations:
                    bibliography_dict[str(counter)] = [citation]
                    counter += 1
                
                # Extract confidence scores
                pattern = r"confidence score:\s*([0-9.]+)(?:\s*\)\s*((?:\[\d+\]\s*)+)|\s+based on\s+(\[\d+\]))"
                matches = re.findall(pattern, content, re.IGNORECASE)
                for score, refs1, refs2 in matches:
                    confidence = score
                    refs = refs1 if refs1 else refs2
                    ref_nums = re.findall(r"\[(\d+)\]", refs)
                    for num in ref_nums:
                        if num in bibliography_dict:
                            bibliography_dict[num].append(confidence)
                
                # Format bibliography
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
                
                # Clean confidence scores from main content
                clean_pattern = r"\(\s*confidence score:\s*[0-9.]+\s*\)"
                content = re.sub(clean_pattern, "", content, flags=re.IGNORECASE)
                content = re.sub(r'\s+', ' ', content)
                
                if bibliography:
                    content += bibliography
            
            return content, response_time, None
        else:
            error_msg = f"Perplexity API error: {response.status_code}, {response.text}"
            return None, response_time, error_msg
            
    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"Perplexity query failed: {str(e)}"
        logger.error(error_msg)
        return None, response_time, error_msg


async def assess_response_quality(response_content: str, query: str, 
                                context: RoutingContext, service: str) -> float:
    """
    Assess the quality of a response using the quality assessment system.
    
    Args:
        response_content: The response content to assess
        query: Original user query
        context: Routing context
        service: Service that generated the response ('lightrag' or 'perplexity')
    
    Returns:
        Quality score (0.0-1.0)
    """
    if not quality_assessor or not relevance_scorer:
        # Simple fallback quality scoring based on content characteristics
        if not response_content or len(response_content.strip()) < 50:
            return 0.1
        
        # Basic quality indicators
        quality_score = 0.5  # Base score
        
        # Length and completeness
        if len(response_content) > 200:
            quality_score += 0.1
        
        # Domain relevance
        if re.search(r'\b(metabol\w*|clinic\w*|biomark\w*)\b', response_content.lower()):
            quality_score += 0.2
        
        # References/citations
        if re.search(r'\[?\d+\]?', response_content) or 'reference' in response_content.lower():
            quality_score += 0.15
        
        # Confidence indicators
        if 'confidence' in response_content.lower():
            quality_score += 0.05
        
        return min(quality_score, 1.0)
    
    try:
        # Use the actual quality assessment system
        relevance_score = await relevance_scorer.score_response_relevance(
            query, response_content, context.query_type or "general"
        )
        
        quality_metrics = await quality_assessor.assess_response_quality(
            response_content, query, {
                'service': service,
                'query_type': context.query_type,
                'query_complexity': context.query_complexity
            }
        )
        
        # Combine relevance and quality scores
        combined_score = (relevance_score * 0.4 + quality_metrics.get('overall_score', 0.5) * 0.6)
        return min(max(combined_score, 0.0), 1.0)
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return 0.5  # Neutral score on error


async def handle_query_with_feature_flags(query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Handle a query using the feature flag system for routing decisions.
    
    Args:
        query: User query text
        user_id: Optional user identifier
    
    Returns:
        Dictionary containing response and metadata
    """
    if not feature_manager:
        # Fallback to Perplexity if feature flags not initialized
        logger.warning("Feature flags not initialized, falling back to Perplexity")
        mock_context = RoutingContext(query_text=query, user_id=user_id)
        response, response_time, error = await query_perplexity_system(query, mock_context)
        return {
            'content': response or "Sorry, I couldn't process your query.",
            'service_used': 'perplexity_fallback',
            'response_time': response_time,
            'error': error,
            'routing_decision': 'fallback'
        }
    
    # Create routing context
    mock_message = type('MockMessage', (), {'content': query})()
    context = create_routing_context(mock_message, user_id)
    
    # Get routing decision
    routing_result = feature_manager.should_use_lightrag(context)
    
    response_content = None
    response_time = 0.0
    error_msg = None
    service_used = None
    quality_score = None
    
    # Route to appropriate service
    if routing_result.decision == RoutingDecision.LIGHTRAG:
        logger.info(f"Routing to LightRAG (reason: {routing_result.reason.value})")
        service_used = 'lightrag'
        response_content, response_time, error_msg = await query_lightrag_system(query, context)
        
        if response_content is None:
            # LightRAG failed, fallback to Perplexity
            logger.warning("LightRAG failed, falling back to Perplexity")
            feature_manager.record_failure('lightrag', error_msg)
            service_used = 'perplexity_fallback'
            response_content, response_time, error_msg = await query_perplexity_system(query, context)
        else:
            # Assess quality
            quality_score = await assess_response_quality(response_content, query, context, 'lightrag')
            feature_manager.record_success('lightrag', response_time, quality_score)
    
    else:
        logger.info(f"Routing to Perplexity (reason: {routing_result.reason.value})")
        service_used = 'perplexity'
        response_content, response_time, error_msg = await query_perplexity_system(query, context)
        
        if response_content is not None:
            # Assess quality
            quality_score = await assess_response_quality(response_content, query, context, 'perplexity')
            feature_manager.record_success('perplexity', response_time, quality_score)
        else:
            feature_manager.record_failure('perplexity', error_msg)
    
    # Record rollout metrics if rollout is active
    if rollout_manager and rollout_manager.rollout_state:
        rollout_manager.record_request_result(
            success=response_content is not None,
            quality_score=quality_score,
            error_details=error_msg
        )
    
    return {
        'content': response_content or "Sorry, I couldn't process your query. Please try again.",
        'service_used': service_used,
        'response_time': response_time,
        'error': error_msg,
        'quality_score': quality_score,
        'routing_decision': routing_result.decision.value,
        'routing_reason': routing_result.reason.value,
        'routing_metadata': routing_result.metadata,
        'user_cohort': routing_result.user_cohort.value if routing_result.user_cohort else None,
        'circuit_breaker_state': routing_result.circuit_breaker_state
    }


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authentication callback (from main.py)."""
    if (username, password) == ("admin", "admin123") or (username, password) == ("testing", "ku9R_3"):
        return cl.User(
            identifier="admin",
            metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize chat session with feature flag system.
    """
    # Initialize feature flag system if not already done
    if not feature_manager:
        initialization_success = await initialize_feature_flag_system()
        if not initialization_success:
            await cl.Message(
                content="⚠️ Feature flag system initialization failed. Using fallback mode.",
                author="CMO"
            ).send()
    
    # Display welcome message with feature flag status
    welcome_content = """🔬 **Clinical Metabolomics Oracle with Intelligent Routing**

Welcome! This enhanced version uses advanced feature flags to intelligently route your queries between different AI systems for optimal responses.

**Current Configuration:**"""
    
    if feature_manager:
        performance_summary = feature_manager.get_performance_summary()
        config_info = performance_summary.get('configuration', {})
        
        welcome_content += f"""
• **Integration Status**: {'✅ Enabled' if config_info.get('integration_enabled') else '❌ Disabled'}
• **Rollout Percentage**: {config_info.get('rollout_percentage', 0)}%
• **A/B Testing**: {'✅ Active' if config_info.get('ab_testing_enabled') else '❌ Inactive'}
• **Circuit Breaker**: {'✅ Protected' if config_info.get('circuit_breaker_enabled') else '❌ Unprotected'}
• **Quality Monitoring**: {'✅ Active' if config_info.get('quality_metrics_enabled') else '❌ Inactive'}

**Performance Stats:**
• **LightRAG Success Rate**: {performance_summary.get('performance', {}).get('lightrag', {}).get('success_count', 0)} requests
• **Perplexity Success Rate**: {performance_summary.get('performance', {}).get('perplexity', {}).get('success_count', 0)} requests
• **Circuit Breaker Status**: {performance_summary.get('circuit_breaker', {}).get('is_open', False) and 'Open' or 'Closed'}"""
    else:
        welcome_content += "\n• **Status**: Running in fallback mode (Perplexity only)"
    
    welcome_content += """\n\n**Disclaimer**: This is an automated question answering tool for informational purposes only and is not intended to replace qualified healthcare professional advice."""
    
    await cl.Message(content=welcome_content, author="CMO").send()
    
    # Get user agreement (following main.py pattern)
    res = await cl.AskActionMessage(
        content='Do you understand the purpose and limitations of the Clinical Metabolomics Oracle?',
        actions=[
            cl.Action(
                name='I Understand',
                label='I Understand',
                description='Agree and continue',
                payload={"response": "agree"}
            ),
            cl.Action(
                name='Disagree',
                label='Disagree',
                description='Disagree to terms of service',
                payload={"response": "disagree"}
            )
        ],
        timeout=300,
        author="CMO",
    ).send()
    
    if res["label"] != "I Understand":
        await cl.Message(
            content="You must agree to the terms of service to continue.",
            author="CMO"
        ).send()
        return
    
    await cl.Message(
        content="Perfect! Ask me anything about clinical metabolomics, and I'll use intelligent routing to get you the best possible answer.",
        author="CMO"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages with feature flag routing.
    """
    start_time = time.time()
    
    # Show thinking message
    thinking_msg = cl.Message(content="🤔 Thinking and routing your query...", author="CMO")
    await thinking_msg.send()
    
    # Get user identifier
    user = cl.user_session.get("user")
    user_id = user.identifier if user else None
    
    # Process query with feature flags
    result = await handle_query_with_feature_flags(message.content, user_id)
    
    # Prepare response content
    response_content = result['content']
    total_time = time.time() - start_time
    
    # Add performance footer
    service_icon = {
        'lightrag': '🧠',
        'perplexity': '🔍',
        'perplexity_fallback': '🔄'
    }.get(result['service_used'], '❓')
    
    footer = f"\n\n---\n"
    footer += f"{service_icon} **Service**: {result['service_used'].replace('_', ' ').title()}"
    footer += f" | ⏱️ **Time**: {total_time:.2f}s"
    
    if result.get('quality_score'):
        footer += f" | 📊 **Quality**: {result['quality_score']:.2f}"
    
    if result.get('user_cohort'):
        footer += f" | 👥 **Cohort**: {result['user_cohort']}"
    
    # Add routing reason for transparency
    footer += f"\n📍 **Routing**: {result['routing_reason'].replace('_', ' ').title()}"
    
    if result.get('error'):
        footer += f"\n⚠️ **Note**: {result['error']}"
    
    response_content += footer
    
    # Update thinking message with final response
    thinking_msg.content = response_content
    await thinking_msg.update()


async def demo_feature_flag_integration():
    """
    Demonstrate feature flag integration capabilities.
    """
    print("\n🔬 Clinical Metabolomics Oracle - Feature Flag Integration Demo")
    print("=" * 70)
    
    # Initialize system
    success = await initialize_feature_flag_system()
    if not success:
        print("❌ Failed to initialize feature flag system")
        return False
    
    # Test queries with different characteristics
    test_queries = [
        "What is metabolomics?",
        "How do metabolic pathways relate to diabetes diagnosis in clinical settings?",
        "Can you explain the role of biomarkers in metabolomics research for cardiovascular disease?",
        "What are the latest developments in clinical metabolomics for cancer detection?"
    ]
    
    print(f"\n📊 Testing {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query[:50]}...")
        
        result = await handle_query_with_feature_flags(query, f"demo_user_{i}")
        
        print(f"✅ Service: {result['service_used']}")
        print(f"⏱️ Time: {result['response_time']:.2f}s")
        print(f"📊 Quality: {result.get('quality_score', 'N/A')}")
        print(f"🎯 Routing: {result['routing_reason']}")
        if result.get('user_cohort'):
            print(f"👥 Cohort: {result['user_cohort']}")
        
        # Show response preview
        content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        print(f"📝 Response: {content}")
    
    # Show performance summary
    if feature_manager:
        print(f"\n📈 Performance Summary:")
        summary = feature_manager.get_performance_summary()
        print(f"Circuit Breaker: {summary['circuit_breaker']['is_open'] and 'Open' or 'Closed'}")
        print(f"Total Requests: {summary['circuit_breaker']['total_requests']}")
        print(f"Success Rate: {summary['circuit_breaker']['success_rate']:.2%}")
    
    print("\n✅ Feature flag integration demo completed!")
    return True


if __name__ == "__main__":
    import asyncio
    
    # Run demonstration
    asyncio.run(demo_feature_flag_integration())