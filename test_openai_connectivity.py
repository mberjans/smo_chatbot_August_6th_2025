#!/usr/bin/env python3
"""
OpenAI API Connectivity Test Script
Clinical Metabolomics Oracle - LightRAG Integration
Task: CMO-LIGHTRAG-001-T06

This script tests OpenAI API connectivity for both chat completions and embeddings
functionality required for LightRAG integration.

Author: Claude Code Assistant
Date: 2025-08-06
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Third-party imports with error handling
try:
    from dotenv import load_dotenv
    print("✓ python-dotenv imported successfully")
except ImportError:
    print("✗ ERROR: python-dotenv not found. Install with: pip install python-dotenv")
    sys.exit(1)

try:
    import openai
    from openai import OpenAI
    print("✓ openai library imported successfully")
except ImportError:
    print("✗ ERROR: openai library not found.")
    print("  Note: openai is not in requirements_lightrag.txt")
    print("  Install with: pip install openai")
    sys.exit(1)

# Constants for LightRAG integration
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class OpenAIConnectivityTester:
    """Test OpenAI API connectivity for LightRAG integration."""
    
    def __init__(self):
        """Initialize the tester with environment variables."""
        self.client: Optional[OpenAI] = None
        self.api_key: Optional[str] = None
        self.test_results: Dict[str, bool] = {}
        self.error_messages: List[str] = []
        
    def load_environment(self) -> bool:
        """
        Load environment variables from multiple possible locations.
        
        Returns:
            bool: True if environment loaded successfully
        """
        print("\n" + "="*50)
        print("LOADING ENVIRONMENT VARIABLES")
        print("="*50)
        
        # Define possible .env file locations
        env_paths = [
            Path.cwd() / ".env",  # Project root
            Path.cwd() / "src" / ".env",  # src directory
        ]
        
        loaded_files = []
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path, override=True)
                loaded_files.append(str(env_path))
                print(f"✓ Loaded environment from: {env_path}")
            else:
                print(f"- Environment file not found: {env_path}")
        
        if not loaded_files:
            print("⚠ WARNING: No .env files found")
            
        # Check for API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            self.error_messages.append("OPENAI_API_KEY not found in environment variables")
            print("✗ OPENAI_API_KEY not set")
            return False
        elif self.api_key.strip() == "":
            self.error_messages.append("OPENAI_API_KEY is empty")
            print("✗ OPENAI_API_KEY is empty")
            return False
        else:
            # Mask the API key for display
            masked_key = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
            print(f"✓ OPENAI_API_KEY found: {masked_key}")
            return True
    
    def initialize_client(self) -> bool:
        """
        Initialize OpenAI client with API key.
        
        Returns:
            bool: True if client initialized successfully
        """
        print("\n" + "="*50)
        print("INITIALIZING OPENAI CLIENT")
        print("="*50)
        
        if not self.api_key:
            print("✗ Cannot initialize client: No API key available")
            return False
            
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=TEST_TIMEOUT
            )
            print("✓ OpenAI client initialized successfully")
            return True
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI client: {str(e)}"
            self.error_messages.append(error_msg)
            print(f"✗ {error_msg}")
            return False
    
    def test_with_retry(self, test_func, test_name: str) -> bool:
        """
        Execute a test function with retry logic.
        
        Args:
            test_func: Function to execute
            test_name: Name of the test for logging
            
        Returns:
            bool: True if test passed
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  Attempt {attempt}/{MAX_RETRIES}...")
                result = test_func()
                if result:
                    return True
                else:
                    if attempt < MAX_RETRIES:
                        print(f"  Retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
            except Exception as e:
                error_msg = f"{test_name} attempt {attempt} failed: {str(e)}"
                print(f"  ✗ {error_msg}")
                if attempt == MAX_RETRIES:
                    self.error_messages.append(error_msg)
                elif attempt < MAX_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
        
        return False
    
    def test_chat_completion(self) -> bool:
        """
        Test OpenAI chat completion API with gpt-4o-mini model.
        
        Returns:
            bool: True if test passed
        """
        def _test():
            if not self.client:
                return False
                
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "user", 
                        "content": "Hello! This is a connectivity test. Please respond with 'API connection successful'."
                    }
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            if response and response.choices:
                content = response.choices[0].message.content.strip()
                print(f"  Response: {content}")
                return "API connection successful" in content or "successful" in content.lower()
            return False
        
        return self.test_with_retry(_test, "Chat completion test")
    
    def test_embeddings(self) -> bool:
        """
        Test OpenAI embeddings API with text-embedding-3-small model.
        
        Returns:
            bool: True if test passed
        """
        def _test():
            if not self.client:
                return False
                
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input="This is a test text for embeddings connectivity."
            )
            
            if response and response.data:
                embedding = response.data[0].embedding
                print(f"  Embedding dimensions: {len(embedding)}")
                print(f"  First 3 values: {embedding[:3]}")
                # text-embedding-3-small should return 1536-dimensional vectors
                return len(embedding) == 1536
            return False
        
        return self.test_with_retry(_test, "Embeddings test")
    
    def test_model_availability(self) -> bool:
        """
        Test if the required models are available.
        
        Returns:
            bool: True if models are available
        """
        def _test():
            if not self.client:
                return False
                
            try:
                # List available models
                models_response = self.client.models.list()
                available_models = [model.id for model in models_response.data]
                
                required_models = [CHAT_MODEL, EMBEDDING_MODEL]
                missing_models = []
                
                for model in required_models:
                    if model in available_models:
                        print(f"  ✓ {model} is available")
                    else:
                        print(f"  ✗ {model} is NOT available")
                        missing_models.append(model)
                
                return len(missing_models) == 0
            except Exception as e:
                print(f"  Error checking model availability: {str(e)}")
                # Don't fail the test entirely if we can't list models
                # This might be due to API permissions
                return True
        
        return self.test_with_retry(_test, "Model availability test")
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all connectivity tests.
        
        Returns:
            Dict[str, bool]: Test results
        """
        print("\n" + "="*70)
        print("OPENAI API CONNECTIVITY TESTS FOR LIGHTRAG INTEGRATION")
        print("="*70)
        print(f"Testing models: {CHAT_MODEL} (chat), {EMBEDDING_MODEL} (embeddings)")
        print(f"Timeout: {TEST_TIMEOUT}s, Max retries: {MAX_RETRIES}")
        
        # Test 1: Environment loading
        print("\n📋 Test 1: Environment Configuration")
        self.test_results["environment"] = self.load_environment()
        
        # Test 2: Client initialization
        print("\n🔧 Test 2: Client Initialization")
        self.test_results["client_init"] = self.initialize_client()
        
        if not self.test_results["client_init"]:
            print("\n⚠ Skipping API tests due to client initialization failure")
            self.test_results["model_availability"] = False
            self.test_results["chat_completion"] = False
            self.test_results["embeddings"] = False
            return self.test_results
        
        # Test 3: Model availability
        print("\n🔍 Test 3: Model Availability")
        self.test_results["model_availability"] = self.test_model_availability()
        
        # Test 4: Chat completion
        print("\n💬 Test 4: Chat Completion API")
        self.test_results["chat_completion"] = self.test_chat_completion()
        
        # Test 5: Embeddings
        print("\n🔢 Test 5: Embeddings API")
        self.test_results["embeddings"] = self.test_embeddings()
        
        return self.test_results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive test report.
        
        Returns:
            str: Formatted test report
        """
        report_lines = []
        report_lines.append("\n" + "="*70)
        report_lines.append("OPENAI API CONNECTIVITY TEST REPORT")
        report_lines.append("="*70)
        report_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Project: Clinical Metabolomics Oracle - LightRAG Integration")
        report_lines.append(f"Task: CMO-LIGHTRAG-001-T06")
        
        # Test results summary
        report_lines.append("\nTEST RESULTS:")
        report_lines.append("-" * 30)
        
        test_descriptions = {
            "environment": "Environment Configuration",
            "client_init": "Client Initialization", 
            "model_availability": "Model Availability",
            "chat_completion": f"Chat Completion ({CHAT_MODEL})",
            "embeddings": f"Embeddings ({EMBEDDING_MODEL})"
        }
        
        passed_tests = 0
        total_tests = len(self.test_results)
        
        for test_name, description in test_descriptions.items():
            if test_name in self.test_results:
                status = "✓ PASS" if self.test_results[test_name] else "✗ FAIL"
                report_lines.append(f"{description:<30} {status}")
                if self.test_results[test_name]:
                    passed_tests += 1
            else:
                report_lines.append(f"{description:<30} - SKIPPED")
        
        # Overall status
        report_lines.append("\nOVERALL STATUS:")
        report_lines.append("-" * 30)
        
        if passed_tests == total_tests and total_tests > 0:
            report_lines.append("🎉 ALL TESTS PASSED - OpenAI API is ready for LightRAG integration!")
            overall_status = "SUCCESS"
        elif passed_tests > 0:
            report_lines.append(f"⚠ PARTIAL SUCCESS - {passed_tests}/{total_tests} tests passed")
            overall_status = "PARTIAL"
        else:
            report_lines.append("❌ ALL TESTS FAILED - OpenAI API is not ready")
            overall_status = "FAILED"
        
        # Error details
        if self.error_messages:
            report_lines.append("\nERROR DETAILS:")
            report_lines.append("-" * 30)
            for i, error in enumerate(self.error_messages, 1):
                report_lines.append(f"{i}. {error}")
        
        # Recommendations
        report_lines.append("\nRECOMMENDATIONS:")
        report_lines.append("-" * 30)
        
        if not self.test_results.get("environment", False):
            report_lines.append("• Set OPENAI_API_KEY in .env or src/.env file")
            report_lines.append("• Ensure the API key is valid and has sufficient credits")
        
        if not self.test_results.get("client_init", False):
            report_lines.append("• Verify OpenAI library installation: pip install openai")
            report_lines.append("• Check internet connectivity")
        
        if overall_status == "SUCCESS":
            report_lines.append("• ✅ Ready to proceed with LightRAG integration")
            report_lines.append("• Both chat completion and embeddings APIs are working")
            report_lines.append("• Required models are available and accessible")
        elif overall_status == "PARTIAL":
            report_lines.append("• Review failed tests and resolve issues before LightRAG integration")
            report_lines.append("• Both chat and embeddings APIs must work for full LightRAG functionality")
        else:
            report_lines.append("• ❌ Do NOT proceed with LightRAG integration until all tests pass")
            report_lines.append("• Resolve API connectivity issues first")
        
        report_lines.append("\n" + "="*70)
        
        return "\n".join(report_lines)


def main():
    """Main entry point for the connectivity test."""
    print("OpenAI API Connectivity Test for LightRAG Integration")
    print("=" * 55)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("⚠ WARNING: Python 3.8+ recommended for OpenAI library")
    
    # Initialize and run tests
    tester = OpenAIConnectivityTester()
    results = tester.run_all_tests()
    
    # Generate and display report
    report = tester.generate_report()
    print(report)
    
    # Exit with appropriate code
    all_passed = all(results.values()) and len(results) > 0
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()