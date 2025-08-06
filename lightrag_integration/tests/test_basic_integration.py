#!/usr/bin/env python3
"""
Simple Integration Test for PDF Processor and LightRAG Connection.

This is a basic integration test to validate that the fixtures work
and the basic testing infrastructure is functional.
"""

import pytest
import asyncio


@pytest.mark.asyncio
async def test_basic_integration_setup(integration_test_environment, pdf_test_documents):
    """Test basic integration setup works."""
    env = integration_test_environment
    docs = pdf_test_documents
    
    # Basic validation
    assert env is not None
    assert env.config is not None
    assert env.lightrag_system is not None
    assert env.pdf_processor is not None
    
    assert docs is not None
    assert len(docs) > 0
    
    # Test first document
    first_doc = docs[0]
    assert first_doc.filename is not None
    assert first_doc.content is not None
    assert len(first_doc.content) > 10
    
    print(f"✓ Integration environment setup successful")
    print(f"✓ Found {len(docs)} test documents")
    print(f"✓ First document: {first_doc.filename}")


@pytest.mark.asyncio
async def test_mock_pdf_processing(integration_test_environment, pdf_test_documents):
    """Test mock PDF processing works."""
    env = integration_test_environment
    docs = pdf_test_documents
    first_doc = docs[0]
    
    # Test PDF processing through mock
    result = await env.pdf_processor.process_pdf(first_doc.filename)
    
    assert result is not None
    assert 'text' in result
    assert 'metadata' in result
    assert len(result['text']) > 0
    
    print(f"✓ PDF processing mock works: {first_doc.filename}")


@pytest.mark.asyncio 
async def test_mock_lightrag_operations(integration_test_environment):
    """Test mock LightRAG operations work."""
    env = integration_test_environment
    
    # Test document insertion
    test_text = "This is a test document about metabolomics and diabetes research."
    insert_result = await env.lightrag_system.ainsert([test_text])
    
    assert insert_result is not None
    
    # Test querying
    query_result = await env.lightrag_system.aquery("What is metabolomics?")
    
    assert query_result is not None
    assert len(query_result) > 0
    
    print(f"✓ LightRAG mock operations work")


def test_sync_fixtures(pdf_test_documents, disease_specific_content):
    """Test that sync fixtures work properly."""
    docs = pdf_test_documents
    content_gen = disease_specific_content
    
    assert docs is not None
    assert len(docs) > 0
    
    # Test disease content generation
    diabetes_content = content_gen('diabetes', 'simple')
    assert diabetes_content is not None
    assert 'diabetes' in diabetes_content.lower()
    
    print(f"✓ Sync fixtures work correctly")
    print(f"✓ Generated content length: {len(diabetes_content)}")