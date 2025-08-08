#!/usr/bin/env python3
"""
Basic Chainlit Integration Example for LightRAG

This example demonstrates how to replace the existing Perplexity API calls
with LightRAG integration while maintaining all existing Chainlit functionality,
session management, and user interface patterns.

Key Features:
- Drop-in replacement for Perplexity API
- Maintains existing session management with cl.user_session
- Preserves citation format and confidence scoring
- Keeps structured logging and error handling
- Supports async/await patterns throughout
- Configurable through environment variables

Usage:
    # Set up environment variables
    export OPENAI_API_KEY="your-api-key"
    export LIGHTRAG_MODEL="gpt-4o-mini"
    export LIGHTRAG_ENABLE_COST_TRACKING="true"
    export LIGHTRAG_DAILY_BUDGET_LIMIT="25.0"
    
    # Run with Chainlit
    chainlit run examples/basic_chainlit_integration.py
"""

import asyncio
import logging
import os
import re
import sys
import time
from typing import Optional, Dict, Any, List, Tuple

import chainlit as cl
from lingua import LanguageDetector

# Import LightRAG integration components
from lightrag_integration import (
    create_clinical_rag_system,
    ClinicalMetabolomicsRAG,
    LightRAGConfig,
    QueryResponse,
    setup_lightrag_logging,
    get_integration_status
)

# Import existing CMO components
from src.translation import BaseTranslator, detect_language, get_language_detector, get_translator, translate
from src.lingua_iso_codes import IsoCode639_1

# Initialize logging
setup_lightrag_logging()
logger = logging.getLogger(__name__)

# Global RAG system instance
RAG_SYSTEM: Optional[ClinicalMetabolomicsRAG] = None


class LightRAGChainlitIntegration:
    """
    Integration class that wraps LightRAG functionality for Chainlit.
    
    This class provides a seamless interface between the existing Chainlit
    application and the new LightRAG system, maintaining compatibility
    while adding enhanced capabilities.
    """
    
    def __init__(self, rag_system: ClinicalMetabolomicsRAG):
        """Initialize the integration with a configured RAG system."""
        self.rag_system = rag_system
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> bool:
        """
        Initialize the RAG system asynchronously.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing LightRAG system...")
            await self.rag_system.initialize_rag()
            
            # Verify system is ready
            status = await self.rag_system.health_check()
            if status.get("status") == "healthy":
                self.logger.info("LightRAG system initialized successfully")
                return True
            else:
                self.logger.error(f"RAG system health check failed: {status}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    async def process_query(self, query: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user query using LightRAG and format for Chainlit.
        
        This method replaces the Perplexity API call while maintaining
        the same response format and citation structure.
        
        Args:
            query: User query string
            session_data: Chainlit session data (language, translator, etc.)
            
        Returns:
            Dict containing processed response, citations, and metadata
        """
        start_time = time.time()
        
        try:
            # Process query through LightRAG
            self.logger.info(f"Processing query: {query[:100]}...")
            
            response = await self.rag_system.query(
                query=query,
                mode="hybrid",  # Use hybrid mode for best results
                include_metadata=True,
                enable_quality_scoring=True
            )
            
            # Format response for Chainlit compatibility
            formatted_response = await self._format_response_for_chainlit(response)
            
            # Add timing information
            processing_time = time.time() - start_time
            formatted_response["processing_time"] = processing_time
            
            self.logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "content": "I apologize, but I encountered an error processing your request. Please try again.",
                "citations": [],
                "bibliography": "",
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def _format_response_for_chainlit(self, response: QueryResponse) -> Dict[str, Any]:
        """
        Format LightRAG response to match Chainlit/Perplexity format.
        
        Args:
            response: QueryResponse from LightRAG
            
        Returns:
            Dict with formatted content, citations, and bibliography
        """
        # Extract main content
        content = response.response if hasattr(response, 'response') else str(response)
        
        # Process citations and references
        citations = []
        bibliography_dict = {}
        
        if hasattr(response, 'metadata') and response.metadata:
            # Extract source documents and create citations
            sources = response.metadata.get('sources', [])
            for i, source in enumerate(sources, 1):
                citation_info = {
                    'id': str(i),
                    'url': source.get('url', ''),
                    'title': source.get('title', f'Source {i}'),
                    'content': source.get('content', ''),
                    'confidence': source.get('confidence_score', 0.8)
                }
                citations.append(citation_info['url'] if citation_info['url'] else citation_info['title'])
                bibliography_dict[str(i)] = [citation_info['url'] or citation_info['title'], citation_info['confidence']]
        
        # Add confidence scores to content if available
        if hasattr(response, 'confidence_score') and response.confidence_score:
            content = f"{content} (confidence score: {response.confidence_score:.2f})"
        
        # Format bibliography
        bibliography = self._format_bibliography(bibliography_dict)
        
        return {
            "content": content,
            "citations": citations,
            "bibliography": bibliography,
            "confidence_score": getattr(response, 'confidence_score', None),
            "source_count": len(citations)
        }
    
    def _format_bibliography(self, bibliography_dict: Dict[str, List]) -> str:
        """Format bibliography in the same style as the original system."""
        if not bibliography_dict:
            return ""
        
        references = "\n\n\n**References:**\n"
        further_reading = "\n**Further Reading:**\n"
        
        for key, value in bibliography_dict.items():
            if len(value) > 1:
                references += f"[{key}]: {value[0]} \n      (Confidence: {value[1]:.2f})\n"
            else:
                further_reading += f"[{key}]: {value[0]} \n"
        
        bibliography = ""
        if references != "\n\n\n**References:**\n":
            bibliography += references
        if further_reading != "\n**Further Reading:**\n":
            bibliography += further_reading
        
        return bibliography
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for debugging and monitoring."""
        try:
            health_status = await self.rag_system.health_check()
            cost_summary = await self.rag_system.get_cost_summary()
            
            return {
                "health": health_status,
                "cost_summary": cost_summary.__dict__ if cost_summary else {},
                "integration_status": get_integration_status()
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}


# Chainlit event handlers with LightRAG integration

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authentication callback - unchanged from original."""
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
    Initialize chat session with LightRAG system.
    
    This replaces the commented-out LlamaIndex initialization
    with LightRAG system setup while maintaining all existing
    UI elements and user flow.
    """
    global RAG_SYSTEM
    
    try:
        # Initialize RAG system if not already done
        if RAG_SYSTEM is None:
            # Create RAG system with clinical metabolomics configuration
            RAG_SYSTEM = create_clinical_rag_system(
                # Enhanced settings for clinical use
                daily_budget_limit=float(os.getenv('LIGHTRAG_DAILY_BUDGET_LIMIT', '25.0')),
                enable_quality_validation=True,
                enable_cost_tracking=True,
                relevance_confidence_threshold=0.75,
                model=os.getenv('LIGHTRAG_MODEL', 'gpt-4o-mini')
            )
        
        # Initialize the integration wrapper
        integration = LightRAGChainlitIntegration(RAG_SYSTEM)
        success = await integration.initialize()
        
        if not success:
            logger.error("Failed to initialize LightRAG system")
            await cl.Message(
                content="⚠️ System initialization failed. Some features may be limited.",
                author="CMO"
            ).send()
        
        # Store integration in session
        cl.user_session.set("lightrag_integration", integration)
        
        # Display intro message and disclaimer (unchanged)
        descr = 'Hello! Welcome to the Clinical Metabolomics Oracle'
        subhead = "I'm a chat tool designed to help you stay informed about clinical metabolomics. I can access and understand a large database of scientific publications.\n\nTo learn more, checkout the Readme page."
        disclaimer = 'The Clinical Metabolomics Oracle is an automated question answering tool, and is not intended to replace the advice of a qualified healthcare professional.\nContent generated by the Clinical Metabolomics Oracle is for informational purposes only, and is not advice for the treatment or diagnosis of any condition.'
        
        elements = [
            cl.Text(name=descr, content=subhead, display='inline'),
            cl.Text(name='Disclaimer', content=disclaimer, display='inline')
        ]
        
        await cl.Message(
            content='',
            elements=elements,
            author="CMO",
        ).send()

        # Continue with user agreement flow (unchanged)
        accepted = False
        while not accepted:
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
                timeout=300,  # five minutes
                author="CMO",
            ).send()

            accepted = res["label"] == "I Understand"

            if not accepted:
                await cl.Message(
                    content="You must agree to the terms of service to continue.",
                    author="CMO",
                ).send()

        welcome = "Welcome! Ask me anything about clinical metabolomics, and I'll do my best to find you the most relevant and up-to-date information."

        await cl.Message(
            content=welcome,
            author="CMO",
        ).send()

        # Set up translation components (unchanged)
        translator: BaseTranslator = get_translator()
        cl.user_session.set("translator", translator)
        await set_chat_settings(translator)

        iso_codes = [
            IsoCode639_1[code.upper()].value
            for code in translator.get_supported_languages(as_dict=True).values()
            if code.upper() in IsoCode639_1._member_names_
        ]
        detector = get_language_detector(*iso_codes)
        cl.user_session.set("detector", detector)
        
        logger.info("Chat session initialized successfully with LightRAG")
        
    except Exception as e:
        logger.error(f"Error during chat initialization: {e}")
        await cl.Message(
            content="⚠️ There was an error initializing the system. Please refresh and try again.",
            author="CMO"
        ).send()


@cl.author_rename
def rename(orig_author: str):
    """Author rename function - unchanged from original."""
    rename_dict = {"Chatbot": "CMO"}
    return rename_dict.get(orig_author, orig_author)


async def set_chat_settings(translator):
    """Set up chat settings UI - unchanged from original."""
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
        )
    ]).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages using LightRAG instead of Perplexity API.
    
    This replaces the entire Perplexity API section while maintaining
    the same user experience, translation support, and response formatting.
    """
    start_time = time.time()
    
    try:
        # Get session components
        detector: LanguageDetector = cl.user_session.get("detector")
        translator: BaseTranslator = cl.user_session.get("translator")
        integration: LightRAGChainlitIntegration = cl.user_session.get("lightrag_integration")
        
        if not integration:
            await cl.Message(
                content="⚠️ System not properly initialized. Please refresh the page.",
                author="CMO"
            ).send()
            return
        
        content = message.content

        # Show thinking message
        await cl.Message(
            content="Thinking...",
            author="CMO",
        ).send()

        # Handle language detection and translation (unchanged logic)
        language = cl.user_session.get("language")
        if not language or language == "auto":
            detection = await detect_language(detector, content)
            language = detection["language"]
        
        if language != "en" and language is not None:
            content = await translate(translator, content, source=language, target="en")

        # Process query using LightRAG (replaces Perplexity API section)
        session_data = {
            "language": language,
            "translator": translator,
            "detector": detector
        }
        
        response_data = await integration.process_query(content, session_data)
        
        # Get response content and metadata
        response_content = response_data.get("content", "")
        bibliography = response_data.get("bibliography", "")
        processing_time = response_data.get("processing_time", 0)
        
        # Handle translation back to user language
        if language != "en" and language is not None:
            response_content = await translate(translator, response_content, source="en", target=language)

        # Add bibliography and timing
        if bibliography:
            response_content += bibliography

        end_time = time.time()
        response_content += f"\n\n*{end_time - start_time:.2f} seconds*"

        # Send final response
        response_message = cl.Message(content=response_content)
        await response_message.send()
        
        logger.info(f"Message processed successfully in {end_time - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await cl.Message(
            content="I apologize, but I encountered an error processing your request. Please try again.",
            author="CMO"
        ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Handle settings updates - unchanged from original."""
    translator = settings["translator"]
    if translator == "Google":
        translator: BaseTranslator = get_translator("google")
    elif translator == "OPUS-MT":
        translator: BaseTranslator = get_translator("opusmt")
    
    await set_chat_settings(translator)
    cl.user_session.set("translator", translator)
    
    language = settings["language"]
    if language == "Detect language":
        language = "auto"
    else:
        languages_to_iso_codes = translator.get_supported_languages(as_dict=True)
        language = languages_to_iso_codes.get(language.lower(), "auto")
    
    cl.user_session.set("language", language)


# Development and testing utilities

async def test_integration():
    """Test function to verify LightRAG integration works correctly."""
    print("Testing LightRAG integration...")
    
    try:
        # Create and initialize system
        rag = create_clinical_rag_system(
            daily_budget_limit=5.0,
            enable_quality_validation=True
        )
        
        integration = LightRAGChainlitIntegration(rag)
        success = await integration.initialize()
        
        if not success:
            print("❌ Integration initialization failed")
            return False
        
        # Test query processing
        test_query = "What are the main metabolites involved in glucose metabolism?"
        session_data = {"language": "en", "translator": None, "detector": None}
        
        result = await integration.process_query(test_query, session_data)
        
        if result.get("error"):
            print(f"❌ Query processing failed: {result['error']}")
            return False
        
        print(f"✅ Integration test successful!")
        print(f"   - Response length: {len(result.get('content', ''))}")
        print(f"   - Citations: {len(result.get('citations', []))}")
        print(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    """
    Main entry point for testing or running the integration.
    
    Usage:
        # Test integration
        python examples/basic_chainlit_integration.py
        
        # Run with Chainlit
        chainlit run examples/basic_chainlit_integration.py
    """
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test mode
        asyncio.run(test_integration())
    else:
        # Normal Chainlit mode - configuration info
        print("🔬 Clinical Metabolomics Oracle - LightRAG Integration")
        print("=" * 60)
        print("Configuration:")
        print(f"  Model: {os.getenv('LIGHTRAG_MODEL', 'gpt-4o-mini')}")
        print(f"  Daily Budget: ${os.getenv('LIGHTRAG_DAILY_BUDGET_LIMIT', '25.0')}")
        print(f"  Cost Tracking: {os.getenv('LIGHTRAG_ENABLE_COST_TRACKING', 'true')}")
        print(f"  Quality Validation: {os.getenv('LIGHTRAG_ENABLE_QUALITY_VALIDATION', 'true')}")
        print("\nTo run: chainlit run examples/basic_chainlit_integration.py")
        print("To test: python examples/basic_chainlit_integration.py test")