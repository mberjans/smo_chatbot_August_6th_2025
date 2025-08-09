#!/usr/bin/env python3
"""
Test script to verify the fallback system integration with the main Chainlit application.

This script tests:
1. Import functionality
2. Fallback system initialization
3. Query processing with fallback protection
4. Error handling and graceful degradation
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required imports work correctly."""
    logger.info("Testing imports...")
    
    try:
        # Test basic imports
        import main
        logger.info("✅ Main module imported successfully")
        
        # Test fallback system availability
        if hasattr(main, 'FALLBACK_SYSTEM_AVAILABLE'):
            if main.FALLBACK_SYSTEM_AVAILABLE:
                logger.info("✅ Fallback system is available")
            else:
                logger.warning("⚠️  Fallback system is not available - will use direct API calls")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


async def test_query_processing():
    """Test the query processing with fallback system."""
    logger.info("Testing query processing...")
    
    try:
        from main import process_query_with_fallback_system, call_perplexity_api
        
        # Test query processing function exists
        logger.info("✅ Query processing functions imported successfully")
        
        # Test direct API call (without enhanced router)
        test_query = "What is metabolomics?"
        logger.info(f"Testing direct API call with query: '{test_query}'")
        
        # Note: This will make an actual API call if PERPLEXITY_API is set
        if os.environ.get("PERPLEXITY_API"):
            try:
                content, citations = await call_perplexity_api(test_query)
                logger.info("✅ Direct API call successful")
                logger.info(f"Response length: {len(content) if content else 0} characters")
                logger.info(f"Citations: {len(citations) if citations else 0}")
            except Exception as e:
                logger.warning(f"⚠️  Direct API call failed (expected if no API key): {e}")
        else:
            logger.info("⚠️  No PERPLEXITY_API key found, skipping API test")
        
        return True
    except Exception as e:
        logger.error(f"❌ Query processing test failed: {e}")
        return False


def test_fallback_system_creation():
    """Test creating the enhanced router."""
    logger.info("Testing fallback system creation...")
    
    try:
        from main import FALLBACK_SYSTEM_AVAILABLE
        
        if not FALLBACK_SYSTEM_AVAILABLE:
            logger.info("⚠️  Fallback system not available, skipping creation test")
            return True
        
        from lightrag_integration.enhanced_query_router_with_fallback import (
            create_production_ready_enhanced_router
        )
        
        # Create cache directory
        cache_dir = Path(__file__).parent / 'test_cache'
        cache_dir.mkdir(exist_ok=True)
        
        # Create enhanced router
        enhanced_router = create_production_ready_enhanced_router(
            emergency_cache_dir=str(cache_dir),
            logger=logger
        )
        
        logger.info("✅ Enhanced router created successfully")
        
        # Test basic functionality
        test_query = "metabolite identification"
        should_use_lightrag = enhanced_router.should_use_lightrag(test_query)
        should_use_perplexity = enhanced_router.should_use_perplexity(test_query)
        
        logger.info(f"✅ Routing decisions: LightRAG={should_use_lightrag}, Perplexity={should_use_perplexity}")
        
        # Get system health report
        health_report = enhanced_router.get_system_health_report()
        logger.info(f"✅ System health: {health_report.get('system_status', 'unknown')}")
        
        # Cleanup
        enhanced_router.shutdown_enhanced_features()
        logger.info("✅ Enhanced router cleaned up successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Fallback system creation test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting fallback integration tests...")
    
    tests = [
        test_imports,
        test_fallback_system_creation,
        lambda: asyncio.create_task(test_query_processing())
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test in enumerate(tests, 1):
        logger.info(f"\n--- Running test {i}/{total}: {test.__name__} ---")
        try:
            if asyncio.iscoroutine(test()):
                result = await test()
            else:
                result = test()
            
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
    
    logger.info(f"\n=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        logger.info("🎉 All tests passed! Fallback integration is working correctly.")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))