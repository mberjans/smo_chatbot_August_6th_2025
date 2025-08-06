#!/usr/bin/env python3
"""
Test script for the enhanced PDF processor with error recovery capabilities.

This script demonstrates the new error recovery features including:
- Retry mechanisms with exponential backoff
- Error classification and recovery strategies
- Memory recovery and file lock handling
- Comprehensive error reporting and statistics
"""

import logging
import sys
import time
from pathlib import Path

# Add the current directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lightrag_integration.pdf_processor import (
    BiomedicalPDFProcessor,
    ErrorRecoveryConfig,
    PDFValidationError,
    PDFProcessingTimeoutError,
    PDFMemoryError,
    PDFFileAccessError
)

def setup_logging():
    """Setup comprehensive logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_error_recovery.log')
        ]
    )
    return logging.getLogger(__name__)

def test_error_recovery_config():
    """Test ErrorRecoveryConfig functionality."""
    print("\n=== Testing Error Recovery Configuration ===")
    
    # Test default configuration
    default_config = ErrorRecoveryConfig()
    print(f"Default config: max_retries={default_config.max_retries}, base_delay={default_config.base_delay}")
    
    # Test custom configuration
    custom_config = ErrorRecoveryConfig(
        max_retries=5,
        base_delay=0.5,
        max_delay=30.0,
        exponential_base=1.8
    )
    print(f"Custom config: max_retries={custom_config.max_retries}, base_delay={custom_config.base_delay}")
    
    # Test delay calculation
    print("\nDelay calculation (exponential backoff):")
    for attempt in range(5):
        delay = custom_config.calculate_delay(attempt)
        print(f"  Attempt {attempt + 1}: {delay:.2f}s delay")
    
    return custom_config

def test_processor_initialization():
    """Test BiomedicalPDFProcessor initialization with error recovery."""
    print("\n=== Testing Processor Initialization ===")
    
    logger = logging.getLogger("pdf_processor_test")
    
    # Test with default error recovery
    processor_default = BiomedicalPDFProcessor(logger=logger)
    print(f"Default processor: max_retries={processor_default.error_recovery.max_retries}")
    
    # Test with custom error recovery
    custom_recovery = ErrorRecoveryConfig(max_retries=2, base_delay=1.0)
    processor_custom = BiomedicalPDFProcessor(
        logger=logger,
        error_recovery_config=custom_recovery
    )
    print(f"Custom processor: max_retries={processor_custom.error_recovery.max_retries}")
    
    return processor_custom

def test_error_classification(processor):
    """Test error classification functionality."""
    print("\n=== Testing Error Classification ===")
    
    test_errors = [
        PDFMemoryError("Memory allocation failed"),
        PDFProcessingTimeoutError("Processing timed out after 300s"),
        PDFFileAccessError("File is locked by another process"),
        PDFValidationError("PDF is corrupted"),
        IOError("Disk I/O error"),
        Exception("Unknown error")
    ]
    
    for error in test_errors:
        is_recoverable, category, strategy = processor._classify_error(error)
        print(f"  {type(error).__name__}: recoverable={is_recoverable}, category={category}, strategy={strategy}")

def test_processing_stats(processor):
    """Test processing statistics functionality."""
    print("\n=== Testing Processing Statistics ===")
    
    # Get initial stats
    stats = processor.get_processing_stats()
    print("Processing stats:")
    for key, value in stats.items():
        if key == 'error_recovery':
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # Test error recovery stats (initially empty)
    recovery_stats = processor.get_error_recovery_stats()
    print(f"\nError recovery stats: {recovery_stats['files_with_retries']} files with retries")
    print(f"Recovery actions by type: {recovery_stats['recovery_actions_by_type']}")

def simulate_file_processing_with_errors():
    """Simulate file processing that might encounter errors."""
    print("\n=== Simulating Error Recovery Scenarios ===")
    
    logger = logging.getLogger("error_simulation")
    
    # Create processor with aggressive retry settings for testing
    recovery_config = ErrorRecoveryConfig(
        max_retries=2,
        base_delay=0.1,
        max_delay=2.0,
        jitter=False  # Disable jitter for predictable testing
    )
    
    processor = BiomedicalPDFProcessor(
        logger=logger,
        processing_timeout=10,  # Short timeout for testing
        error_recovery_config=recovery_config
    )
    
    # Test scenarios (these would normally fail, but we can test the logic)
    print("\nTesting error recovery logic:")
    
    # Test memory recovery
    print("1. Testing memory recovery...")
    recovery_result = processor._attempt_memory_recovery()
    print(f"   Memory recovery attempted: {recovery_result}")
    
    # Test simple recovery
    print("2. Testing simple recovery...")
    recovery_result = processor._attempt_simple_recovery(0)
    print(f"   Simple recovery attempted: {recovery_result}")
    
    # Test recovery statistics tracking
    processor._recovery_actions_attempted['memory_cleanup'] = 2
    processor._recovery_actions_attempted['simple_retry'] = 1
    
    stats = processor.get_error_recovery_stats()
    print(f"3. Recovery statistics: {stats['total_recovery_actions']} total actions")
    print(f"   Actions by type: {stats['recovery_actions_by_type']}")

def main():
    """Main test function."""
    print("=== PDF Processor Error Recovery System Test ===")
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting error recovery system tests")
    
    try:
        # Test configuration
        config = test_error_recovery_config()
        
        # Test processor initialization
        processor = test_processor_initialization()
        
        # Test error classification
        test_error_classification(processor)
        
        # Test processing statistics
        test_processing_stats(processor)
        
        # Simulate error scenarios
        simulate_file_processing_with_errors()
        
        print("\n=== All Tests Completed Successfully ===")
        logger.info("All error recovery tests completed successfully")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())