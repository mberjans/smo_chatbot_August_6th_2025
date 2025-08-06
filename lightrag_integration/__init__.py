"""
LightRAG Integration Module for SMO Chatbot

This module provides integration with LightRAG for enhanced graph-based
retrieval-augmented generation capabilities in the SMO chatbot system.

The module includes:
- LightRAG component integration
- PDF processing utilities
- Configuration management
- Comprehensive testing suite

Author: SMO Chatbot Development Team
Created: August 6, 2025
"""

__version__ = "0.1.0"
__author__ = "SMO Chatbot Development Team"
__description__ = "LightRAG Integration Module for SMO Chatbot"

# Import main configuration classes and functions
from .config import (
    LightRAGConfig,
    LightRAGConfigError,
    setup_lightrag_logging
)

# Import PDF processing classes and functions
from .pdf_processor import (
    BiomedicalPDFProcessor,
    BiomedicalPDFProcessorError
)

# This module will be expanded with component imports as they are implemented
# Example future imports:
# from .lightrag_component import LightRAGComponent

__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "LightRAGConfig",
    "LightRAGConfigError", 
    "setup_lightrag_logging",
    "BiomedicalPDFProcessor",
    "BiomedicalPDFProcessorError"
]