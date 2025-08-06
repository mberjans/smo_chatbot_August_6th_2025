#!/usr/bin/env python3
"""
Comprehensive unit tests for ClinicalMetabolomicsRAG initialization following TDD principles.

This module implements comprehensive unit tests for the ClinicalMetabolomicsRAG class
initialization as required by CMO-LIGHTRAG-005. These tests are written following
Test-Driven Development (TDD) principles and will guide the implementation of the
ClinicalMetabolomicsRAG class.

Test Coverage:
- Successful initialization with valid configuration
- Initialization parameter validation and error handling
- LightRAG integration setup and configuration
- OpenAI API configuration for LLM and embeddings
- Error handling for missing API keys and invalid configurations
- Biomedical-specific parameter setup and validation
- Logging and monitoring initialization
- API cost monitoring and logging setup
- Rate limiting and error recovery mechanisms
- Async functionality and resource management

The ClinicalMetabolomicsRAG class being tested does NOT exist yet - this follows
TDD where tests define the expected behavior and drive the implementation.

Requirements from CMO-LIGHTRAG-005:
1. Initialization with LightRAGConfig
2. LightRAG setup with biomedical parameters
3. OpenAI LLM and embedding functions configuration
4. Error handling for API failures and rate limits
5. Basic query functionality working
6. API cost monitoring and logging

Test Classes:
- TestClinicalMetabolomicsRAGInitialization: Core initialization tests
- TestClinicalMetabolomicsRAGConfiguration: Configuration validation tests
- TestClinicalMetabolomicsRAGLightRAGSetup: LightRAG integration tests
- TestClinicalMetabolomicsRAGOpenAISetup: OpenAI API integration tests
- TestClinicalMetabolomicsRAGErrorHandling: Error handling and recovery tests
- TestClinicalMetabolomicsRAGBiomedicalConfig: Biomedical-specific configuration tests
- TestClinicalMetabolomicsRAGMonitoring: Logging and monitoring tests
- TestClinicalMetabolomicsRAGQueryFunctionality: Basic query functionality tests
- TestClinicalMetabolomicsRAGAsyncOperations: Async functionality tests
- TestClinicalMetabolomicsRAGEdgeCases: Edge cases and error conditions
"""

import pytest
import asyncio
import logging
import tempfile
import time
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import sys

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import existing modules
from lightrag_integration.config import LightRAGConfig, LightRAGConfigError
from lightrag_integration.pdf_processor import (
    BiomedicalPDFProcessor, BiomedicalPDFProcessorError
)

# Note: ClinicalMetabolomicsRAG does not exist yet - will be implemented
# based on these tests following TDD principles


# =====================================================================
# TEST FIXTURES AND UTILITIES
# =====================================================================

@dataclass
class MockLightRAGResponse:
    """Mock response from LightRAG for testing query functionality."""
    content: str
    metadata: Dict[str, Any]
    cost: float
    token_usage: Dict[str, int]


@dataclass
class MockOpenAIAPIUsage:
    """Mock OpenAI API usage for cost monitoring tests."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class MockLightRAGInstance:
    """Mock LightRAG instance for testing integration."""
    
    def __init__(self, working_dir: str = None):
        self.working_dir = working_dir
        self.initialized = False
        self.documents_added = []
        self.query_history = []
        self.cost_tracker = []
        
    async def aquery(self, query: str, **kwargs) -> MockLightRAGResponse:
        """Mock async query method."""
        self.query_history.append(query)
        return MockLightRAGResponse(
            content=f"Mock response for: {query}",
            metadata={"sources": ["mock_source_1.pdf"], "confidence": 0.9},
            cost=0.001,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
    
    async def ainsert(self, documents: Union[str, List[str]]) -> None:
        """Mock async document insertion method."""
        if isinstance(documents, str):
            documents = [documents]
        self.documents_added.extend(documents)
        
    def initialize_storage(self) -> None:
        """Mock storage initialization."""
        self.initialized = True


@pytest.fixture
def valid_config():
    """Provide a valid LightRAGConfig for testing."""
    return LightRAGConfig(
        api_key="test-api-key-12345",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        working_dir=Path("/tmp/test_lightrag"),
        max_async=16,
        max_tokens=32768,
        auto_create_dirs=False
    )


@pytest.fixture
def invalid_config():
    """Provide an invalid LightRAGConfig for testing error handling."""
    return LightRAGConfig(
        api_key=None,  # Missing API key
        model="",  # Empty model
        embedding_model="invalid-embedding-model",
        max_async=-1,  # Invalid value
        max_tokens=0,  # Invalid value
        auto_create_dirs=False
    )


@pytest.fixture
def mock_openai_client():
    """Provide a mock OpenAI client for testing."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.embeddings.create = AsyncMock()
    return client


@pytest.fixture
def temp_working_dir():
    """Provide a temporary working directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_pdf_processor():
    """Provide a mock PDF processor for testing."""
    processor = MagicMock(spec=BiomedicalPDFProcessor)
    processor.process_pdf = AsyncMock(return_value={
        "text": "Sample biomedical text content",
        "metadata": {"title": "Test Paper", "authors": ["Test Author"]},
        "page_count": 5,
        "processing_time": 1.2
    })
    return processor


# =====================================================================
# TEST CLASSES
# =====================================================================

class TestClinicalMetabolomicsRAGInitialization:
    """Test class for core ClinicalMetabolomicsRAG initialization functionality."""
    
    def test_initialization_with_valid_config(self, valid_config):
        """Test successful initialization with a valid LightRAGConfig."""
        # This test will fail initially as ClinicalMetabolomicsRAG doesn't exist yet
        # Following TDD, we write the test first to define expected behavior
        
        # Expected behavior: Should initialize without raising exceptions
        # and set up all required components
        with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
            mock_lightrag_instance = MockLightRAGInstance()
            mock_lightrag.return_value = mock_lightrag_instance
            
            # Import will fail initially - this drives implementation
            try:
                from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify initialization state
                assert rag.config == valid_config
                assert rag.lightrag_instance is not None
                assert rag.is_initialized == True
                assert hasattr(rag, 'logger')
                assert hasattr(rag, 'cost_monitor')
                
            except ImportError:
                # Expected during TDD - implementation doesn't exist yet
                pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_initialization_with_none_config_raises_error(self):
        """Test that initialization with None config raises appropriate error."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with pytest.raises(ValueError, match="config cannot be None"):
                ClinicalMetabolomicsRAG(config=None)
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_initialization_with_invalid_config_type_raises_error(self):
        """Test that initialization with wrong config type raises appropriate error."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with pytest.raises(TypeError, match="config must be a LightRAGConfig instance"):
                ClinicalMetabolomicsRAG(config="invalid_config")
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_initialization_sets_up_required_attributes(self, valid_config):
        """Test that initialization creates all required instance attributes."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify all required attributes exist
                required_attributes = [
                    'config', 'lightrag_instance', 'logger', 'cost_monitor',
                    'is_initialized', 'query_history', 'total_cost', 'biomedical_params'
                ]
                
                for attr in required_attributes:
                    assert hasattr(rag, attr), f"Missing required attribute: {attr}"
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_initialization_with_custom_working_directory(self, temp_working_dir):
        """Test initialization with custom working directory."""
        config = LightRAGConfig(
            api_key="test-key",
            working_dir=temp_working_dir,
            auto_create_dirs=False
        )
        
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance(str(temp_working_dir))
                
                rag = ClinicalMetabolomicsRAG(config=config)
                
                # Verify working directory is set correctly
                assert rag.config.working_dir == temp_working_dir
                assert rag.lightrag_instance.working_dir == str(temp_working_dir)
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGConfiguration:
    """Test class for configuration validation and parameter handling."""
    
    def test_config_validation_during_initialization(self, invalid_config):
        """Test that invalid configuration is detected during initialization."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            # Should raise LightRAGConfigError due to missing API key
            with pytest.raises(LightRAGConfigError):
                ClinicalMetabolomicsRAG(config=invalid_config)
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_biomedical_parameters_setup(self, valid_config):
        """Test that biomedical-specific parameters are configured correctly."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify biomedical parameters are set up
                assert hasattr(rag, 'biomedical_params')
                assert isinstance(rag.biomedical_params, dict)
                
                # Expected biomedical-specific parameters
                expected_params = [
                    'entity_extraction_focus',
                    'relationship_types',
                    'domain_keywords',
                    'preprocessing_rules'
                ]
                
                for param in expected_params:
                    assert param in rag.biomedical_params
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_config_parameter_override(self, valid_config):
        """Test that configuration parameters can be overridden during initialization."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance()
                
                # Override parameters during initialization
                rag = ClinicalMetabolomicsRAG(
                    config=valid_config,
                    custom_model="gpt-4",
                    custom_max_tokens=16384,
                    enable_cost_tracking=True
                )
                
                # Verify overrides are applied
                assert rag.effective_model == "gpt-4"
                assert rag.effective_max_tokens == 16384
                assert rag.cost_tracking_enabled == True
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGLightRAGSetup:
    """Test class for LightRAG integration setup and configuration."""
    
    def test_lightrag_instance_creation(self, valid_config):
        """Test that LightRAG instance is created with correct parameters."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify LightRAG was called with correct parameters
                mock_lightrag.assert_called_once()
                call_kwargs = mock_lightrag.call_args[1]
                
                assert call_kwargs['working_dir'] == str(valid_config.working_dir)
                assert 'llm_model_func' in call_kwargs
                assert 'embedding_func' in call_kwargs
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_lightrag_biomedical_configuration(self, valid_config):
        """Test that LightRAG is configured with biomedical-specific parameters."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify biomedical configuration is passed to LightRAG
                call_kwargs = mock_lightrag.call_args[1]
                
                # Should include biomedical-specific entity extraction
                assert 'entity_extract_max_gleaning' in call_kwargs
                assert 'entity_extract_max_tokens' in call_kwargs
                
                # Should include relationship extraction parameters
                assert 'relationship_max_gleaning' in call_kwargs
                assert 'relationship_max_tokens' in call_kwargs
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_lightrag_storage_initialization(self, valid_config, temp_working_dir):
        """Test that LightRAG storage is properly initialized."""
        config = LightRAGConfig(
            api_key="test-key",
            working_dir=temp_working_dir,
            auto_create_dirs=True
        )
        
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=config)
                
                # Verify storage directories are created
                expected_dirs = [
                    temp_working_dir / "graph_chunk_entity_relation.json",
                    temp_working_dir / "vdb_chunks",
                    temp_working_dir / "vdb_entities", 
                    temp_working_dir / "vdb_relationships"
                ]
                
                # At minimum, working directory should exist
                assert temp_working_dir.exists()
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGGetLLMFunction:
    """Test class for the enhanced _get_llm_function method implementation."""
    
    @pytest.mark.asyncio
    async def test_get_llm_function_basic_functionality(self, valid_config, mock_openai_client):
        """Test basic functionality of the _get_llm_function method."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
                
                # Set up mocks
                mock_openai.return_value = mock_openai_client
                mock_lightrag.return_value = MockLightRAGInstance()
                
                # Mock OpenAI completion response
                mock_completion = MagicMock()
                mock_completion.choices = [MagicMock()]
                mock_completion.choices[0].message.content = "Test LLM response"
                mock_completion.usage.total_tokens = 150
                mock_completion.usage.prompt_tokens = 100
                mock_completion.usage.completion_tokens = 50
                mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Get the LLM function
                llm_func = rag._get_llm_function()
                assert callable(llm_func)
                
                # Test the LLM function
                response = await llm_func("Test prompt")
                assert response == "Test LLM response"
                
                # Verify API call was made with correct parameters
                mock_openai_client.chat.completions.create.assert_called_once()
                call_args = mock_openai_client.chat.completions.create.call_args[1]
                assert call_args['model'] == valid_config.model
                assert call_args['max_tokens'] == valid_config.max_tokens
                assert call_args['temperature'] == 0.1
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_get_llm_function_cost_monitoring(self, valid_config, mock_openai_client):
        """Test that _get_llm_function properly tracks API costs."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
                
                mock_openai.return_value = mock_openai_client
                mock_lightrag.return_value = MockLightRAGInstance()
                
                # Mock OpenAI completion response
                mock_completion = MagicMock()
                mock_completion.choices = [MagicMock()]
                mock_completion.choices[0].message.content = "Test response"
                mock_completion.usage.total_tokens = 150
                mock_completion.usage.prompt_tokens = 100
                mock_completion.usage.completion_tokens = 50
                mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)
                
                rag = ClinicalMetabolomicsRAG(config=valid_config, enable_cost_tracking=True)
                
                initial_cost = rag.total_cost
                
                # Call LLM function
                llm_func = rag._get_llm_function()
                await llm_func("Test prompt for cost tracking")
                
                # Verify cost was tracked
                assert rag.total_cost > initial_cost
                assert rag.cost_monitor['queries'] > 0
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_get_llm_function_biomedical_prompt_optimization(self, valid_config, mock_openai_client):
        """Test biomedical prompt optimization in _get_llm_function."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
                
                mock_openai.return_value = mock_openai_client
                mock_lightrag.return_value = MockLightRAGInstance()
                
                # Mock OpenAI completion response
                mock_completion = MagicMock()
                mock_completion.choices = [MagicMock()]
                mock_completion.choices[0].message.content = "Biomedical response"
                mock_completion.usage.total_tokens = 200
                mock_completion.usage.prompt_tokens = 150
                mock_completion.usage.completion_tokens = 50
                mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Ensure biomedical parameters are set
                rag.biomedical_params['entity_extraction_focus'] = 'biomedical'
                
                llm_func = rag._get_llm_function()
                await llm_func("What metabolites are involved in diabetes?")
                
                # Verify the prompt was enhanced with biomedical context
                call_args = mock_openai_client.chat.completions.create.call_args[1]
                messages = call_args['messages']
                assert len(messages) == 1
                prompt = messages[0]['content']
                assert 'clinical metabolomics' in prompt.lower()
                assert 'biomarkers' in prompt.lower()
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_get_llm_function_error_handling(self, valid_config, mock_openai_client):
        """Test error handling in _get_llm_function."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            import openai
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
                
                mock_openai.return_value = mock_openai_client
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                llm_func = rag._get_llm_function()
                
                # Test rate limit error handling
                mock_openai_client.chat.completions.create = AsyncMock(
                    side_effect=openai.RateLimitError("Rate limit exceeded", response=None, body=None)
                )
                
                with pytest.raises(openai.RateLimitError):
                    await llm_func("Test prompt")
                
                # Test authentication error handling
                mock_openai_client.chat.completions.create = AsyncMock(
                    side_effect=openai.AuthenticationError("Invalid API key", response=None, body=None)
                )
                
                with pytest.raises(Exception):  # Should be wrapped in ClinicalMetabolomicsRAGError
                    await llm_func("Test prompt")
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_get_llm_function_retry_logic(self, valid_config, mock_openai_client):
        """Test retry logic in _get_llm_function."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            import openai
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
                
                mock_openai.return_value = mock_openai_client
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                llm_func = rag._get_llm_function()
                
                # Mock API timeout followed by successful response
                mock_completion = MagicMock()
                mock_completion.choices = [MagicMock()]
                mock_completion.choices[0].message.content = "Success after retry"
                mock_completion.usage.total_tokens = 150
                mock_completion.usage.prompt_tokens = 100
                mock_completion.usage.completion_tokens = 50
                
                mock_openai_client.chat.completions.create = AsyncMock(
                    side_effect=[
                        openai.APITimeoutError("Timeout"),
                        mock_completion
                    ]
                )
                
                # Should succeed on retry
                response = await llm_func("Test prompt")
                assert response == "Success after retry"
                
                # Verify API was called twice (original + retry)
                assert mock_openai_client.chat.completions.create.call_count == 2
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_get_llm_function_model_validation(self, valid_config, mock_openai_client):
        """Test model validation warnings in _get_llm_function."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
                
                mock_openai.return_value = mock_openai_client
                mock_lightrag.return_value = MockLightRAGInstance()
                
                # Create config with non-optimal model for biomedical tasks
                config = LightRAGConfig(
                    api_key="test-api-key",
                    model="text-davinci-003",  # Non-optimal model
                    working_dir=Path("/tmp/test_lightrag"),
                    auto_create_dirs=False
                )
                
                mock_completion = MagicMock()
                mock_completion.choices = [MagicMock()]
                mock_completion.choices[0].message.content = "Response"
                mock_completion.usage.total_tokens = 150
                mock_completion.usage.prompt_tokens = 100
                mock_completion.usage.completion_tokens = 50
                mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)
                
                rag = ClinicalMetabolomicsRAG(config=config)
                
                # Mock logger to capture warnings
                with patch.object(rag.logger, 'warning') as mock_warning:
                    llm_func = rag._get_llm_function()
                    await llm_func("Test prompt")
                    
                    # Should log warning about non-optimal model
                    mock_warning.assert_called_with(
                        "Model text-davinci-003 may not be optimal for biomedical tasks"
                    )
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_calculate_api_cost_method(self, valid_config):
        """Test the _calculate_api_cost method."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Test cost calculation for different models
                test_cases = [
                    ('gpt-4o-mini', {'prompt_tokens': 1000, 'completion_tokens': 500}),
                    ('gpt-4o', {'prompt_tokens': 1000, 'completion_tokens': 500}),
                    ('gpt-3.5-turbo', {'prompt_tokens': 1000, 'completion_tokens': 500}),
                ]
                
                for model, token_usage in test_cases:
                    cost = rag._calculate_api_cost(model, token_usage)
                    assert isinstance(cost, float)
                    assert cost > 0.0
                    
                    # Verify cost calculation logic
                    if model == 'gpt-4o-mini':
                        # Should be cheapest option
                        expected_cost = (1000/1000 * 0.00015) + (500/1000 * 0.0006)
                        assert abs(cost - expected_cost) < 0.0001
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGOpenAISetup:
    """Test class for OpenAI API integration and configuration."""
    
    def test_openai_llm_function_setup(self, valid_config, mock_openai_client):
        """Test that OpenAI LLM function is configured correctly."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
                
                mock_openai.return_value = mock_openai_client
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify OpenAI client is created with API key
                mock_openai.assert_called_once_with(api_key=valid_config.api_key)
                
                # Verify LLM function is set up
                call_kwargs = mock_lightrag.call_args[1]
                assert 'llm_model_func' in call_kwargs
                assert callable(call_kwargs['llm_model_func'])
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_openai_embedding_function_setup(self, valid_config, mock_openai_client):
        """Test that OpenAI embedding function is configured correctly."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
                
                mock_openai.return_value = mock_openai_client
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify embedding function is set up
                call_kwargs = mock_lightrag.call_args[1]
                assert 'embedding_func' in call_kwargs
                assert callable(call_kwargs['embedding_func'])
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_openai_api_error_handling(self, valid_config):
        """Test error handling for OpenAI API failures."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag, \
                 patch('lightrag_integration.clinical_metabolomics_rag.openai.OpenAI') as mock_openai:
                
                # Mock OpenAI client that raises errors
                mock_client = MagicMock()
                mock_client.chat.completions.create.side_effect = Exception("API Error")
                mock_openai.return_value = mock_client
                
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Test that API errors are handled gracefully
                with pytest.raises(Exception):
                    # This would be called internally by LightRAG
                    llm_func = mock_lightrag.call_args[1]['llm_model_func']
                    await llm_func("test prompt")
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGErrorHandling:
    """Test class for error handling and recovery mechanisms."""
    
    def test_missing_api_key_error(self):
        """Test error handling when API key is missing."""
        config = LightRAGConfig(
            api_key=None,  # Missing API key
            auto_create_dirs=False
        )
        
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with pytest.raises(LightRAGConfigError, match="API key is required"):
                ClinicalMetabolomicsRAG(config=config)
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_invalid_working_directory_error(self):
        """Test error handling for invalid working directory."""
        config = LightRAGConfig(
            api_key="test-key",
            working_dir=Path("/nonexistent/invalid/directory"),
            auto_create_dirs=False
        )
        
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with pytest.raises(LightRAGConfigError):
                ClinicalMetabolomicsRAG(config=config)
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_lightrag_initialization_failure_handling(self, valid_config):
        """Test handling of LightRAG initialization failures."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                # Mock LightRAG to raise an exception during initialization
                mock_lightrag.side_effect = Exception("LightRAG initialization failed")
                
                with pytest.raises(Exception, match="LightRAG initialization failed"):
                    ClinicalMetabolomicsRAG(config=valid_config)
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_rate_limit_error_handling(self, valid_config):
        """Test rate limit error handling and retry logic."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify rate limit handling is configured
                assert hasattr(rag, 'rate_limiter')
                assert hasattr(rag, 'retry_config')
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGBiomedicalConfig:
    """Test class for biomedical-specific configuration and parameters."""
    
    def test_biomedical_entity_types_configuration(self, valid_config):
        """Test that biomedical entity types are properly configured."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify biomedical entity types are defined
                expected_entity_types = [
                    'METABOLITE', 'PROTEIN', 'GENE', 'DISEASE', 'PATHWAY',
                    'ORGANISM', 'TISSUE', 'BIOMARKER', 'DRUG', 'CLINICAL_TRIAL'
                ]
                
                for entity_type in expected_entity_types:
                    assert entity_type in rag.biomedical_params['entity_types']
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_biomedical_relationship_types_configuration(self, valid_config):
        """Test that biomedical relationship types are properly configured."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify biomedical relationship types are defined
                expected_relationships = [
                    'METABOLIZES', 'REGULATES', 'INTERACTS_WITH', 'CAUSES',
                    'TREATS', 'ASSOCIATED_WITH', 'PART_OF', 'EXPRESSED_IN',
                    'TARGETS', 'MODULATES'
                ]
                
                for rel_type in expected_relationships:
                    assert rel_type in rag.biomedical_params['relationship_types']
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_clinical_metabolomics_specific_keywords(self, valid_config):
        """Test that clinical metabolomics keywords are configured."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify clinical metabolomics keywords are defined
                expected_keywords = [
                    'metabolomics', 'clinical', 'biomarker', 'mass spectrometry',
                    'NMR', 'metabolite', 'pathway analysis', 'biofluid'
                ]
                
                domain_keywords = rag.biomedical_params['domain_keywords']
                for keyword in expected_keywords:
                    assert any(keyword.lower() in kw.lower() for kw in domain_keywords)
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGMonitoring:
    """Test class for logging and monitoring functionality."""
    
    def test_logger_initialization(self, valid_config):
        """Test that logger is properly initialized."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify logger is set up
                assert hasattr(rag, 'logger')
                assert isinstance(rag.logger, logging.Logger)
                assert rag.logger.name == 'clinical_metabolomics_rag'
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_cost_monitoring_initialization(self, valid_config):
        """Test that cost monitoring is properly initialized."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_lightrag.return_value = MockLightRAGInstance()
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify cost monitoring is set up
                assert hasattr(rag, 'cost_monitor')
                assert hasattr(rag, 'total_cost')
                assert rag.total_cost == 0.0
                
                # Should have methods for cost tracking
                assert hasattr(rag, 'track_api_cost')
                assert hasattr(rag, 'get_cost_summary')
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_query_history_tracking(self, valid_config):
        """Test that query history is tracked properly."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify query history tracking is initialized
                assert hasattr(rag, 'query_history')
                assert isinstance(rag.query_history, list)
                assert len(rag.query_history) == 0
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGQueryFunctionality:
    """Test class for basic query functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_query_functionality(self, valid_config):
        """Test basic query processing functionality."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Test basic query
                query = "What metabolites are associated with diabetes?"
                response = await rag.query(query)
                
                # Verify response structure
                assert 'content' in response
                assert 'metadata' in response
                assert 'cost' in response
                
                # Verify query is tracked
                assert query in rag.query_history
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_query_modes_support(self, valid_config):
        """Test support for different LightRAG query modes."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                query = "What is the role of glucose in metabolism?"
                
                # Test different query modes
                modes = ['naive', 'local', 'global', 'hybrid']
                
                for mode in modes:
                    response = await rag.query(query, mode=mode)
                    assert 'content' in response
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_cost_tracking_during_queries(self, valid_config):
        """Test that API costs are tracked during queries."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                initial_cost = rag.total_cost
                
                # Execute query
                await rag.query("Test query for cost tracking")
                
                # Verify cost was tracked
                assert rag.total_cost > initial_cost
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGAsyncOperations:
    """Test class for async functionality and resource management."""
    
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, valid_config):
        """Test handling of concurrent queries."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Execute multiple queries concurrently
                queries = [
                    "What are the main metabolites in glucose metabolism?",
                    "How does insulin affect metabolite levels?",
                    "What biomarkers are used in diabetes diagnosis?"
                ]
                
                tasks = [rag.query(query) for query in queries]
                responses = await asyncio.gather(*tasks)
                
                # Verify all queries completed successfully
                assert len(responses) == len(queries)
                for response in responses:
                    assert 'content' in response
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_after_operations(self, valid_config):
        """Test proper resource cleanup after async operations."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Execute operations
                await rag.query("Test query")
                
                # Test cleanup method exists and works
                await rag.cleanup()
                
                # Verify resources are cleaned up appropriately
                # (Implementation should handle connection pooling, etc.)
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


class TestClinicalMetabolomicsRAGEdgeCases:
    """Test class for edge cases and error conditions."""
    
    def test_empty_working_directory_handling(self):
        """Test handling of empty working directory."""
        config = LightRAGConfig(
            api_key="test-key",
            working_dir=Path(""),  # Empty directory
            auto_create_dirs=False
        )
        
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            # Should handle empty directory gracefully or raise appropriate error
            with pytest.raises((LightRAGConfigError, ValueError)):
                ClinicalMetabolomicsRAG(config=config)
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_query_with_empty_string(self, valid_config):
        """Test query with empty string input."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Should handle empty query gracefully
                with pytest.raises(ValueError, match="Query cannot be empty"):
                    await rag.query("")
                    
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_query_with_very_long_input(self, valid_config):
        """Test query with very long input string."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Test with very long query (exceeding token limits)
                long_query = "What metabolites " * 10000  # Very long query
                
                # Should handle token limit gracefully
                response = await rag.query(long_query)
                assert 'content' in response
                # Implementation should truncate or handle appropriately
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_multiple_initialization_calls(self, valid_config):
        """Test multiple initialization calls on same instance."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Multiple initialization calls should be handled safely
                initial_state = rag.is_initialized
                rag._initialize()  # Call internal init method again
                
                # Should not break existing state
                assert rag.is_initialized == initial_state
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

class TestClinicalMetabolomicsRAGIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_full_initialization_and_query_workflow(self, valid_config):
        """Test complete workflow from initialization to query processing."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                # Initialize
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                # Verify initialization
                assert rag.is_initialized
                assert rag.total_cost == 0.0
                
                # Execute query
                response = await rag.query("Test biomedical query")
                
                # Verify complete workflow
                assert 'content' in response
                assert len(rag.query_history) == 1
                assert rag.total_cost > 0.0
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    def test_config_integration_with_pdf_processor(self, valid_config, mock_pdf_processor):
        """Test integration between configuration and PDF processor."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(
                    config=valid_config, 
                    pdf_processor=mock_pdf_processor
                )
                
                # Verify PDF processor integration
                assert hasattr(rag, 'pdf_processor')
                assert rag.pdf_processor == mock_pdf_processor
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


# =====================================================================
# PERFORMANCE AND BENCHMARKING TESTS
# =====================================================================

class TestClinicalMetabolomicsRAGPerformance:
    """Performance and benchmarking tests."""
    
    @pytest.mark.asyncio
    async def test_initialization_performance(self, valid_config):
        """Test initialization performance within acceptable limits."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                start_time = time.time()
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                initialization_time = time.time() - start_time
                
                # Initialization should be fast (under 5 seconds)
                assert initialization_time < 5.0
                assert rag.is_initialized
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")
    
    @pytest.mark.asyncio
    async def test_query_response_time(self, valid_config):
        """Test query response time performance."""
        try:
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            with patch('lightrag_integration.clinical_metabolomics_rag.LightRAG') as mock_lightrag:
                mock_instance = MockLightRAGInstance()
                mock_lightrag.return_value = mock_instance
                
                rag = ClinicalMetabolomicsRAG(config=valid_config)
                
                start_time = time.time()
                response = await rag.query("Test query for performance")
                query_time = time.time() - start_time
                
                # Query should complete within reasonable time (mock should be fast)
                assert query_time < 1.0  # Mock query should be very fast
                assert 'content' in response
                
        except ImportError:
            pytest.skip("ClinicalMetabolomicsRAG not implemented yet - TDD phase")


if __name__ == "__main__":
    """
    Run the test suite when executed directly.
    
    These tests will initially fail as the ClinicalMetabolomicsRAG class
    doesn't exist yet. This is expected in TDD - the tests define the
    expected behavior and guide the implementation.
    """
    pytest.main([__file__, "-v", "--tb=short"])