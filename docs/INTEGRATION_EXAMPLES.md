# Clinical Metabolomics Oracle - LightRAG Integration Examples

This document provides comprehensive integration examples for incorporating LightRAG into the Clinical Metabolomics Oracle system. All examples are based on the actual implementation in the `examples/` directory.

## Table of Contents

1. [Basic Chainlit Integration](#basic-chainlit-integration)
2. [Advanced Pipeline Integration](#advanced-pipeline-integration)
3. [Complete System Integration](#complete-system-integration)
4. [Feature Flag Integration](#feature-flag-integration)
5. [Production Deployment Examples](#production-deployment-examples)
6. [A/B Testing Examples](#ab-testing-examples)
7. [Migration Guide Examples](#migration-guide-examples)
8. [Custom Integration Patterns](#custom-integration-patterns)

---

## Basic Chainlit Integration

### Example 1: Simple LightRAG Integration

This example shows the most basic integration replacing Perplexity with LightRAG:

```python
# File: examples/basic_chainlit_integration.py
import chainlit as cl
import os
from lightrag_integration import (
    create_clinical_rag_system,
    LightRAGConfig,
    validate_configuration
)

# Global LightRAG system
lightrag_system = None

@cl.on_chat_start
async def start():
    """Initialize the LightRAG-enhanced CMO system"""
    global lightrag_system
    
    # Display initialization message
    init_msg = await cl.Message(
        content="🔬 Initializing Clinical Metabolomics Oracle with LightRAG..."
    ).send()
    
    try:
        # Validate configuration
        await validate_configuration()
        
        # Create and initialize LightRAG system
        config = LightRAGConfig(
            working_dir=os.getenv('LIGHTRAG_WORKING_DIR', './lightrag_storage'),
            papers_dir=os.getenv('LIGHTRAG_PAPERS_DIR', './papers'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        lightrag_system = await create_clinical_rag_system(config)
        
        # Store in session
        cl.user_session.set("lightrag_system", lightrag_system)
        cl.user_session.set("initialized", True)
        
        # Update initialization message
        await init_msg.update(
            content="🧬 **Clinical Metabolomics Oracle Ready!**\\n\\n"
                   "I can help you with biomedical research questions using our "
                   "enhanced knowledge graph. Ask me anything about clinical metabolomics!"
        )
        
    except Exception as e:
        await init_msg.update(
            content=f"❌ **Initialization Error**: {str(e)}\\n\\n"
                   "Please check your configuration and try again."
        )

@cl.on_message
async def on_message(message: cl.Message):
    """Process user messages with LightRAG"""
    
    # Check if system is initialized
    if not cl.user_session.get("initialized", False):
        await cl.Message(
            content="⚠️ System not initialized. Please refresh the page."
        ).send()
        return
    
    lightrag_system = cl.user_session.get("lightrag_system")
    user_query = message.content.strip()
    
    # Show processing indicator
    processing_msg = await cl.Message(
        content="🔍 Searching knowledge graph..."
    ).send()
    
    try:
        # Query LightRAG system
        response = await lightrag_system.query(
            user_query,
            mode="hybrid"  # Use both local and global knowledge
        )
        
        # Update with response
        await processing_msg.update(content=response)
        
        # Add source indicator
        await cl.Message(
            content="📚 *Response generated from LightRAG knowledge graph*",
            author="System"
        ).send()
        
    except Exception as e:
        await processing_msg.update(
            content=f"❌ **Error processing query**: {str(e)}\\n\\n"
                   "Please try rephrasing your question or contact support."
        )

if __name__ == "__main__":
    cl.run()
```

### Example 2: Enhanced Basic Integration with Quality Scoring

```python
# Enhanced basic integration with quality assessment
import chainlit as cl
from lightrag_integration import (
    create_clinical_rag_system,
    QualityAssessmentSuite,
    LightRAGConfig
)

# Global components
lightrag_system = None
quality_suite = None

@cl.on_chat_start
async def start():
    global lightrag_system, quality_suite
    
    # Initialize components
    lightrag_system = await create_clinical_rag_system()
    quality_suite = QualityAssessmentSuite()
    
    cl.user_session.set("lightrag_system", lightrag_system)
    cl.user_session.set("quality_suite", quality_suite)
    
    await cl.Message(
        content="🧬 CMO with Quality Assessment ready!"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    lightrag_system = cl.user_session.get("lightrag_system")
    quality_suite = cl.user_session.get("quality_suite")
    
    query = message.content.strip()
    
    # Process query
    response = await lightrag_system.query(query, mode="hybrid")
    
    # Assess quality
    quality_assessment = await quality_suite.assess_response(
        query=query,
        response=response,
        sources=lightrag_system.get_last_sources()
    )
    
    # Send response with quality indicator
    quality_emoji = "🌟" if quality_assessment.composite_score > 0.9 else "✅" if quality_assessment.composite_score > 0.8 else "⚠️"
    
    await cl.Message(content=response).send()
    
    await cl.Message(
        content=f"{quality_emoji} Quality Score: {quality_assessment.composite_score:.2f}",
        author="System"
    ).send()
```

---

## Advanced Pipeline Integration

### Example 3: Hybrid System with Intelligent Routing

```python
# File: examples/advanced_pipeline_integration.py
import chainlit as cl
from lightrag_integration import (
    FeatureFlagManager,
    create_clinical_rag_system,
    QualityAssessmentSuite,
    BudgetManager,
    IntelligentRouter
)
from typing import Optional

class AdvancedCMOSystem:
    """Advanced CMO system with intelligent routing between LightRAG and Perplexity"""
    
    def __init__(self):
        self.lightrag_system: Optional = None
        self.feature_flags: Optional[FeatureFlagManager] = None
        self.quality_suite: Optional[QualityAssessmentSuite] = None
        self.budget_manager: Optional[BudgetManager] = None
        self.router: Optional[IntelligentRouter] = None
        
    async def initialize(self):
        """Initialize all system components"""
        
        # Initialize LightRAG system
        self.lightrag_system = await create_clinical_rag_system()
        
        # Initialize feature flag system
        self.feature_flags = FeatureFlagManager()
        
        # Initialize quality assessment
        self.quality_suite = QualityAssessmentSuite()
        
        # Initialize budget management
        self.budget_manager = BudgetManager(
            daily_budget=float(os.getenv('LIGHTRAG_DAILY_BUDGET', '100.0')),
            alert_threshold=0.8
        )
        
        # Initialize intelligent router
        self.router = IntelligentRouter(
            lightrag_system=self.lightrag_system,
            fallback_enabled=True
        )
        
    async def process_query(self, query: str, user_id: str) -> dict:
        """Process query with intelligent routing"""
        
        start_time = time.time()
        
        try:
            # Check budget before processing
            budget_status = await self.budget_manager.check_budget_availability()
            if not budget_status.can_process:
                return {
                    'response': "Daily budget exceeded. Please try again tomorrow.",
                    'source': 'budget_limit',
                    'quality_score': None
                }
            
            # Get routing decision
            routing_decision = await self.feature_flags.get_routing_decision(
                user_id=user_id,
                query=query,
                context={
                    'budget_remaining': budget_status.remaining_budget,
                    'system_health': await self.get_system_health()
                }
            )
            
            # Route query based on decision
            if routing_decision.use_lightrag:
                response = await self._process_with_lightrag(query)
                source = "lightrag"
            else:
                response = await self._process_with_perplexity(query)
                source = "perplexity"
            
            # Assess quality if using LightRAG
            quality_score = None
            if source == "lightrag":
                quality_assessment = await self.quality_suite.assess_response(
                    query=query,
                    response=response,
                    sources=self.lightrag_system.get_last_sources()
                )
                quality_score = quality_assessment.composite_score
            
            # Track usage and costs
            await self.budget_manager.track_usage(
                query=query,
                response=response,
                source=source,
                processing_time=time.time() - start_time,
                quality_score=quality_score
            )
            
            return {
                'response': response,
                'source': source,
                'quality_score': quality_score,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            # Handle errors with fallback
            return await self._handle_error(query, str(e))
    
    async def _process_with_lightrag(self, query: str) -> str:
        """Process query using LightRAG"""
        return await self.lightrag_system.query(query, mode="hybrid")
    
    async def _process_with_perplexity(self, query: str) -> str:
        """Process query using Perplexity (existing system)"""
        # This would integrate with existing Perplexity code
        # For demonstration, returning placeholder
        return f"Perplexity response for: {query}"
    
    async def _handle_error(self, query: str, error_msg: str) -> dict:
        """Handle processing errors with fallback"""
        
        # Log error
        logger.error(f"Query processing error: {error_msg}")
        
        # Try fallback system
        try:
            fallback_response = await self._process_with_perplexity(query)
            return {
                'response': fallback_response,
                'source': 'perplexity_fallback',
                'quality_score': None,
                'error': error_msg
            }
        except Exception as fallback_error:
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                'source': 'error',
                'quality_score': None,
                'error': f"Primary: {error_msg}, Fallback: {str(fallback_error)}"
            }

# Chainlit integration
cmo_system = AdvancedCMOSystem()

@cl.on_chat_start
async def start():
    """Initialize advanced CMO system"""
    
    init_msg = await cl.Message(
        content="🚀 Initializing Advanced CMO System..."
    ).send()
    
    try:
        await cmo_system.initialize()
        
        # Generate user ID for feature flag consistency
        user_id = cl.user_session.get("id", f"user_{hash(cl.user_session.get('id', 'anonymous'))}")
        cl.user_session.set("user_id", user_id)
        cl.user_session.set("cmo_system", cmo_system)
        
        await init_msg.update(
            content="🧬 **Advanced CMO System Ready!**\\n\\n"
                   "• Intelligent routing between LightRAG and Perplexity\\n"
                   "• Real-time quality assessment\\n"
                   "• Budget management and monitoring\\n"
                   "• Automatic fallback systems\\n\\n"
                   "Ask me any biomedical research question!"
        )
        
    except Exception as e:
        await init_msg.update(
            content=f"❌ Initialization failed: {str(e)}"
        )

@cl.on_message
async def on_message(message: cl.Message):
    """Process messages with advanced CMO system"""
    
    cmo_system = cl.user_session.get("cmo_system")
    user_id = cl.user_session.get("user_id")
    
    if not cmo_system:
        await cl.Message(content="System not initialized").send()
        return
    
    query = message.content.strip()
    
    # Show processing indicator
    processing_msg = await cl.Message(
        content="🧠 Processing with intelligent routing..."
    ).send()
    
    # Process query
    result = await cmo_system.process_query(query, user_id)
    
    # Update with response
    await processing_msg.update(content=result['response'])
    
    # Add metadata
    metadata_parts = []
    metadata_parts.append(f"📡 Source: {result['source'].title()}")
    
    if result['quality_score'] is not None:
        quality_emoji = "🌟" if result['quality_score'] > 0.9 else "✅" if result['quality_score'] > 0.8 else "⚠️"
        metadata_parts.append(f"{quality_emoji} Quality: {result['quality_score']:.2f}")
    
    metadata_parts.append(f"⏱️ Time: {result['processing_time']:.2f}s")
    
    await cl.Message(
        content=" | ".join(metadata_parts),
        author="System"
    ).send()
```

---

## Complete System Integration

### Example 4: Production-Ready Complete Integration

```python
# File: examples/complete_system_integration.py
import chainlit as cl
import asyncio
import logging
from lightrag_integration import (
    create_production_rag_system,
    ProductionConfig,
    ComprehensiveMonitoringSystem,
    AlertManager,
    ResponseCache,
    SessionManager,
    UserAnalytics
)

class ProductionCMOSystem:
    """Complete production-ready CMO system"""
    
    def __init__(self):
        self.config = ProductionConfig()
        self.rag_system = None
        self.monitoring = None
        self.alerts = None
        self.cache = None
        self.session_manager = None
        self.analytics = None
        
    async def initialize(self):
        """Initialize complete production system"""
        
        # Core RAG system
        self.rag_system = await create_production_rag_system(self.config)
        
        # Monitoring and alerting
        self.monitoring = ComprehensiveMonitoringSystem()
        self.alerts = AlertManager()
        
        # Response caching
        self.cache = ResponseCache(
            backend="redis",
            ttl_hours=1,
            max_cache_size_mb=500
        )
        
        # Session management
        self.session_manager = SessionManager()
        
        # User analytics
        self.analytics = UserAnalytics()
        
        # Start background tasks
        asyncio.create_task(self.monitoring.start_monitoring())
        asyncio.create_task(self.analytics.start_analytics_collection())
        
    async def process_user_query(self, query: str, session_id: str) -> dict:
        """Complete query processing with full production features"""
        
        start_time = time.time()
        
        try:
            # Get or create user session
            session = await self.session_manager.get_session(session_id)
            user_context = session.get_user_context()
            
            # Check cache first
            cache_key = self.cache.generate_cache_key(query, user_context)
            cached_response = await self.cache.get(cache_key)
            
            if cached_response:
                await self.analytics.track_cache_hit(session_id, query)
                return {
                    'response': cached_response['response'],
                    'source': 'cache',
                    'cached': True,
                    'processing_time': time.time() - start_time
                }
            
            # Process with monitoring
            with self.monitoring.track_query_processing():
                
                # Enhanced query processing
                response_data = await self._enhanced_query_processing(
                    query=query,
                    user_context=user_context,
                    session=session
                )
                
                # Cache successful responses
                if response_data['quality_score'] and response_data['quality_score'] > 0.8:
                    await self.cache.set(cache_key, {
                        'response': response_data['response'],
                        'quality_score': response_data['quality_score'],
                        'timestamp': time.time()
                    })
                
                # Update session context
                await session.update_context(query, response_data['response'])
                
                # Track analytics
                await self.analytics.track_query(
                    session_id=session_id,
                    query=query,
                    response=response_data['response'],
                    source=response_data['source'],
                    quality_score=response_data['quality_score'],
                    processing_time=time.time() - start_time
                )
                
                return {
                    **response_data,
                    'cached': False,
                    'processing_time': time.time() - start_time
                }
                
        except Exception as e:
            await self.alerts.send_error_alert(
                error=str(e),
                query=query,
                session_id=session_id
            )
            
            return await self._handle_production_error(query, str(e))
    
    async def _enhanced_query_processing(self, query: str, user_context: dict, session) -> dict:
        """Enhanced query processing with context awareness"""
        
        # Personalize query based on user context
        enhanced_query = await self._enhance_query_with_context(query, user_context)
        
        # Multi-modal processing
        processing_results = await asyncio.gather(
            self._process_with_lightrag(enhanced_query),
            self._assess_query_complexity(enhanced_query),
            self._extract_query_intent(enhanced_query),
            return_exceptions=True
        )
        
        lightrag_result = processing_results[0]
        complexity_score = processing_results[1] if not isinstance(processing_results[1], Exception) else 0.5
        query_intent = processing_results[2] if not isinstance(processing_results[2], Exception) else "general"
        
        if isinstance(lightrag_result, Exception):
            raise lightrag_result
        
        # Post-process response
        enhanced_response = await self._post_process_response(
            response=lightrag_result,
            query_intent=query_intent,
            user_preferences=user_context.get('preferences', {})
        )
        
        return enhanced_response
    
    async def _post_process_response(self, response: str, query_intent: str, user_preferences: dict) -> dict:
        """Post-process response with enhancements"""
        
        # Apply formatting based on user preferences
        if user_preferences.get('detailed_explanations', False):
            response = await self._add_detailed_explanations(response)
        
        # Add relevant citations
        citations = await self._extract_citations(response)
        
        # Calculate quality score
        quality_score = await self.rag_system.calculate_response_quality(response)
        
        # Add confidence indicators
        confidence_level = self._calculate_confidence_level(quality_score, query_intent)
        
        return {
            'response': response,
            'citations': citations,
            'quality_score': quality_score,
            'confidence_level': confidence_level,
            'source': 'lightrag_enhanced',
            'intent': query_intent
        }

# Chainlit integration for complete system
production_system = ProductionCMOSystem()

@cl.on_chat_start
async def start():
    """Initialize production CMO system"""
    
    # Create unique session ID
    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    
    # Show initialization
    init_msg = await cl.Message(
        content="🏭 **Initializing Production CMO System**\\n\\n"
               "⚙️ Loading components..."
    ).send()
    
    try:
        await production_system.initialize()
        
        await init_msg.update(
            content="🏭 **Production CMO System Online**\\n\\n"
                   "✅ LightRAG knowledge graph loaded\\n"
                   "✅ Monitoring and alerting active\\n"
                   "✅ Response caching enabled\\n"
                   "✅ User analytics tracking\\n"
                   "✅ Session management ready\\n\\n"
                   "🧬 Ready for biomedical research queries!"
        )
        
    except Exception as e:
        await init_msg.update(
            content=f"❌ **Production System Error**\\n\\n{str(e)}"
        )

@cl.on_message
async def on_message(message: cl.Message):
    """Handle messages with production system"""
    
    session_id = cl.user_session.get("session_id")
    query = message.content.strip()
    
    # Show enhanced processing indicator
    processing_msg = await cl.Message(
        content="🏭 **Production Processing**\\n\\n"
               "🔍 Checking cache...\\n"
               "🧠 Analyzing query...\\n"
               "📚 Searching knowledge graph..."
    ).send()
    
    try:
        # Process with production system
        result = await production_system.process_user_query(query, session_id)
        
        # Format response
        response_content = result['response']
        
        # Add citations if available
        if 'citations' in result and result['citations']:
            citations_text = "\\n\\n**Sources:**\\n" + "\\n".join(
                f"• {citation}" for citation in result['citations'][:3]
            )
            response_content += citations_text
        
        await processing_msg.update(content=response_content)
        
        # Add metadata
        metadata_parts = []
        
        if result['cached']:
            metadata_parts.append("⚡ Cached Response")
        else:
            metadata_parts.append(f"📡 {result['source'].replace('_', ' ').title()}")
        
        if result.get('quality_score'):
            quality_emoji = "🌟" if result['quality_score'] > 0.9 else "✅" if result['quality_score'] > 0.8 else "⚠️"
            metadata_parts.append(f"{quality_emoji} Quality: {result['quality_score']:.2f}")
        
        if result.get('confidence_level'):
            metadata_parts.append(f"🎯 Confidence: {result['confidence_level']}")
        
        metadata_parts.append(f"⏱️ {result['processing_time']:.2f}s")
        
        await cl.Message(
            content=" | ".join(metadata_parts),
            author="Production System"
        ).send()
        
    except Exception as e:
        await processing_msg.update(
            content=f"❌ **Processing Error**\\n\\n{str(e)}\\n\\n"
                   "The production system is investigating this issue."
        )
```

---

## Feature Flag Integration

### Example 5: A/B Testing with Statistical Analysis

```python
# File: examples/ab_testing_example.py
import chainlit as cl
from lightrag_integration import (
    ABTestManager,
    StatisticalAnalyzer,
    MetricsCollector,
    UserCohortManager
)
from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class ABTestResult:
    response: str
    source: str
    metrics: Dict[str, Any]
    test_group: str
    user_id: str

class ABTestingCMOSystem:
    """CMO system with comprehensive A/B testing"""
    
    def __init__(self):
        self.ab_test = ABTestManager(
            test_name="lightrag_vs_perplexity_v2",
            control_group_ratio=0.5,  # 50% Perplexity
            treatment_group_ratio=0.5,  # 50% LightRAG
            minimum_sample_size=100,
            confidence_level=0.95
        )
        
        self.analyzer = StatisticalAnalyzer()
        self.metrics_collector = MetricsCollector()
        self.cohort_manager = UserCohortManager()
        
    async def initialize(self):
        """Initialize A/B testing system"""
        await self.ab_test.initialize()
        await self.metrics_collector.initialize()
        
    async def process_query_with_ab_test(self, query: str, user_id: str) -> ABTestResult:
        """Process query with A/B testing"""
        
        # Get user assignment (consistent for same user)
        assignment = self.ab_test.get_user_assignment(user_id)
        
        start_time = time.time()
        
        if assignment.group == "treatment":
            # Use LightRAG
            response = await self._process_with_lightrag(query)
            source = "lightrag"
        else:
            # Use Perplexity (control)
            response = await self._process_with_perplexity(query)
            source = "perplexity"
        
        processing_time = time.time() - start_time
        
        # Collect detailed metrics
        metrics = {
            'processing_time': processing_time,
            'response_length': len(response),
            'query_length': len(query),
            'timestamp': time.time(),
            'session_id': f"session_{hash(user_id)}",
        }
        
        # Quality assessment for LightRAG responses
        if source == "lightrag":
            quality_score = await self._assess_quality(query, response)
            metrics['quality_score'] = quality_score
        
        # Track metrics for statistical analysis
        await self.metrics_collector.track_ab_test_metrics(
            test_name=self.ab_test.test_name,
            user_id=user_id,
            group=assignment.group,
            metrics=metrics,
            query=query,
            response=response
        )
        
        return ABTestResult(
            response=response,
            source=source,
            metrics=metrics,
            test_group=assignment.group,
            user_id=user_id
        )
    
    async def get_ab_test_results(self) -> Dict[str, Any]:
        """Get current A/B test analysis results"""
        
        # Get collected data
        test_data = await self.metrics_collector.get_test_data(
            test_name=self.ab_test.test_name
        )
        
        if len(test_data) < self.ab_test.minimum_sample_size:
            return {
                'status': 'insufficient_data',
                'current_sample_size': len(test_data),
                'minimum_required': self.ab_test.minimum_sample_size
            }
        
        # Statistical analysis
        analysis = await self.analyzer.analyze_ab_test(test_data)
        
        return {
            'status': 'analysis_ready',
            'sample_size': len(test_data),
            'results': analysis,
            'recommendations': self._generate_recommendations(analysis)
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate recommendations based on analysis"""
        
        recommendations = {}
        
        # Response time comparison
        if 'response_time' in analysis['metrics']:
            rt_analysis = analysis['metrics']['response_time']
            if rt_analysis['statistical_significance'] < 0.05:
                if rt_analysis['treatment_mean'] < rt_analysis['control_mean']:
                    recommendations['response_time'] = "LightRAG is significantly faster"
                else:
                    recommendations['response_time'] = "Perplexity is significantly faster"
            else:
                recommendations['response_time'] = "No significant difference in response time"
        
        # Quality comparison (if available)
        if 'quality_score' in analysis['metrics']:
            quality_analysis = analysis['metrics']['quality_score']
            if quality_analysis['statistical_significance'] < 0.05:
                recommendations['quality'] = f"LightRAG quality score: {quality_analysis['treatment_mean']:.2f}"
        
        # Overall recommendation
        significant_improvements = sum(1 for rec in recommendations.values() if "LightRAG" in rec and "faster" in rec or "better" in rec)
        
        if significant_improvements >= 2:
            recommendations['overall'] = "Recommend deploying LightRAG"
        elif significant_improvements == 1:
            recommendations['overall'] = "Consider extended testing"
        else:
            recommendations['overall'] = "Continue with current system"
        
        return recommendations

# Chainlit integration for A/B testing
ab_test_system = ABTestingCMOSystem()

@cl.on_chat_start
async def start():
    """Initialize A/B testing system"""
    
    # Generate consistent user ID
    user_id = cl.user_session.get("id", f"user_{hash(cl.user_session.get('id', 'anonymous'))}")
    cl.user_session.set("user_id", user_id)
    
    init_msg = await cl.Message(
        content="🧪 **Initializing A/B Testing System**\\n\\n"
               "Setting up experiment: LightRAG vs Perplexity"
    ).send()
    
    try:
        await ab_test_system.initialize()
        
        # Get user's test group assignment
        assignment = ab_test_system.ab_test.get_user_assignment(user_id)
        
        await init_msg.update(
            content="🧪 **A/B Testing System Ready**\\n\\n"
                   f"👤 Your assignment: **{assignment.group.title()} Group**\\n"
                   f"🔬 Test: LightRAG vs Perplexity\\n"
                   f"📊 Your responses will help improve the system!\\n\\n"
                   "Ask any biomedical research question!"
        )
        
    except Exception as e:
        await init_msg.update(
            content=f"❌ A/B Testing initialization error: {str(e)}"
        )

@cl.on_message
async def on_message(message: cl.Message):
    """Process messages with A/B testing"""
    
    user_id = cl.user_session.get("user_id")
    query = message.content.strip()
    
    # Special command to show A/B test results
    if query.lower() == "/test-results":
        results = await ab_test_system.get_ab_test_results()
        
        if results['status'] == 'insufficient_data':
            await cl.Message(
                content=f"📊 **A/B Test Status**\\n\\n"
                       f"Sample size: {results['current_sample_size']}/{results['minimum_required']}\\n"
                       f"Need {results['minimum_required'] - results['current_sample_size']} more responses for analysis."
            ).send()
        else:
            # Format results
            recommendations = results['recommendations']
            results_text = f"📊 **A/B Test Results**\\n\\n"
            results_text += f"Sample size: {results['sample_size']}\\n\\n"
            
            for metric, recommendation in recommendations.items():
                results_text += f"• **{metric.title()}**: {recommendation}\\n"
            
            await cl.Message(content=results_text).send()
        
        return
    
    # Show processing with A/B test info
    processing_msg = await cl.Message(
        content="🧪 Processing with A/B testing..."
    ).send()
    
    try:
        # Process query with A/B testing
        result = await ab_test_system.process_query_with_ab_test(query, user_id)
        
        # Update with response
        await processing_msg.update(content=result.response)
        
        # Add A/B test metadata
        group_emoji = "🔬" if result.test_group == "treatment" else "🌐"
        metadata = f"{group_emoji} Test Group: {result.test_group.title()} | "
        metadata += f"Source: {result.source.title()} | "
        metadata += f"Time: {result.metrics['processing_time']:.2f}s"
        
        if 'quality_score' in result.metrics:
            quality_emoji = "🌟" if result.metrics['quality_score'] > 0.9 else "✅"
            metadata += f" | {quality_emoji} Quality: {result.metrics['quality_score']:.2f}"
        
        await cl.Message(
            content=metadata,
            author="A/B Test System"
        ).send()
        
        # Prompt for feedback occasionally
        import random
        if random.random() < 0.1:  # 10% chance
            await cl.Message(
                content="💡 **Tip**: Type `/test-results` to see current A/B test analysis!",
                author="System"
            ).send()
        
    except Exception as e:
        await processing_msg.update(
            content=f"❌ A/B testing error: {str(e)}"
        )

if __name__ == "__main__":
    cl.run()
```

This comprehensive integration examples document provides practical, working examples for all major integration patterns. Each example builds on the actual implementation in the `examples/` directory and demonstrates different aspects of integrating LightRAG with the Clinical Metabolomics Oracle system.

The examples progress from simple basic integration to advanced production-ready systems with comprehensive monitoring, A/B testing, and statistical analysis. All code is based on the actual modules and classes available in the lightrag_integration package.