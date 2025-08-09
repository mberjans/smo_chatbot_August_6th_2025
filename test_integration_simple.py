#!/usr/bin/env python3
"""
Simple test to verify the fallback system integration code structure.
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fallback_system_import():
    """Test that the fallback system can be imported."""
    logger.info("Testing fallback system import...")
    
    try:
        # Add lightrag_integration to path
        lightrag_path = Path(__file__).parent / 'lightrag_integration'
        sys.path.insert(0, str(lightrag_path))
        
        # Test import
        from enhanced_query_router_with_fallback import (
            create_production_ready_enhanced_router,
            FallbackIntegrationConfig
        )
        
        logger.info("✅ Fallback system imports successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Fallback system import failed: {e}")
        return False

def test_main_structure():
    """Test that the main.py file has the correct structure."""
    logger.info("Testing main.py structure...")
    
    try:
        main_path = Path(__file__).parent / 'src' / 'main.py'
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Check for key integration points
        checks = [
            ("Fallback system imports", "from lightrag_integration.enhanced_query_router_with_fallback import"),
            ("FALLBACK_SYSTEM_AVAILABLE flag", "FALLBACK_SYSTEM_AVAILABLE"),
            ("Enhanced router initialization", "create_production_ready_enhanced_router"),
            ("Fallback system in on_chat_start", "cl.user_session.set(\"enhanced_router\""),
            ("Query processing function", "process_query_with_fallback_system"),
            ("API call function", "call_perplexity_api"),
            ("Enhanced router usage", "enhanced_router = cl.user_session.get(\"enhanced_router\")"),
            ("Session cleanup", "on_chat_end"),
            ("Citation processing preserved", "bibliography_dict"),
            ("Translation preserved", "await translate(translator")
        ]
        
        passed = 0
        for check_name, check_text in checks:
            if check_text in content:
                logger.info(f"✅ {check_name}")
                passed += 1
            else:
                logger.error(f"❌ {check_name}")
        
        logger.info(f"Structure check: {passed}/{len(checks)} components found")
        return passed == len(checks)
        
    except Exception as e:
        logger.error(f"❌ Main structure test failed: {e}")
        return False

def main():
    """Run tests."""
    logger.info("Starting integration structure tests...")
    
    tests = [
        test_fallback_system_import,
        test_main_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test in enumerate(tests, 1):
        logger.info(f"\n--- Running test {i}/{total}: {test.__name__} ---")
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
    
    logger.info(f"\n=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        logger.info("🎉 All structure tests passed! Integration code is correctly implemented.")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())