#!/usr/bin/env python3
"""
Enhanced Query Error Handling Validation Test for CMO-LIGHTRAG-007-T04

This comprehensive test validates the enhanced QueryParam error handling implementation
including parameter validation, type checking, range validation, and error recovery.

Author: Claude Code (Anthropic)
Created: 2025-08-07
"""

import asyncio
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add project path
sys.path.append(str(Path(__file__).parent))

try:
    from lightrag_integration.clinical_metabolomics_rag import (
        ClinicalMetabolomicsRAG, ClinicalMetabolomicsRAGError, LightRAGConfig
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestEnhancedQueryErrorHandling:
    """Comprehensive test suite for enhanced QueryParam error handling."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def valid_config(self, temp_dir):
        """Create a valid configuration for testing."""
        return LightRAGConfig(
            working_dir=temp_dir / "test_kb",
            api_key="test-key-123",
            model="gpt-4o-mini",
            max_tokens=1000,
            embedding_model="text-embedding-3-large"
        )
    
    @pytest.fixture
    def rag_instance(self, valid_config):
        """Create a RAG instance for testing."""
        return ClinicalMetabolomicsRAG(config=valid_config)
    
    @pytest.mark.asyncio
    async def test_invalid_mode_validation(self, rag_instance):
        """Test validation of invalid mode parameters."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Test invalid mode
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", mode="invalid_mode")
        
        assert "Invalid mode 'invalid_mode'" in str(exc_info.value)
        assert "Must be one of:" in str(exc_info.value)
        
        # Test empty mode (should use default)
        try:
            await rag_instance.query("test query", mode="")
            assert False, "Empty mode should be invalid"
        except ClinicalMetabolomicsRAGError as e:
            assert "Invalid mode ''" in str(e)
    
    @pytest.mark.asyncio
    async def test_response_type_validation(self, rag_instance):
        """Test validation of response_type parameter."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Test non-string response_type
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", response_type=123)
        
        assert "response_type must be a string" in str(exc_info.value)
        
        # Test empty response_type
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", response_type="")
        
        assert "response_type cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_top_k_validation(self, rag_instance):
        """Test validation of top_k parameter."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Test non-integer top_k
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", top_k="invalid")
        
        assert "top_k must be an integer" in str(exc_info.value)
        
        # Test negative top_k
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", top_k=-5)
        
        assert "top_k must be positive" in str(exc_info.value)
        
        # Test zero top_k
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", top_k=0)
        
        assert "top_k must be positive" in str(exc_info.value)
        
        # Test very large top_k (should raise error)
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", top_k=2000)
        
        assert "top_k too large" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_max_total_tokens_validation(self, rag_instance):
        """Test validation of max_total_tokens parameter."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Test non-integer max_total_tokens
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", max_total_tokens="invalid")
        
        assert "max_total_tokens must be an integer" in str(exc_info.value)
        
        # Test negative max_total_tokens
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", max_total_tokens=-100)
        
        assert "max_total_tokens must be positive" in str(exc_info.value)
        
        # Test zero max_total_tokens
        with pytest.raises(ClinicalMetabolomicsRAGError) as exc_info:
            await rag_instance.query("test query", max_total_tokens=0)
        
        assert "max_total_tokens must be positive" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_automatic_parameter_adjustment(self, rag_instance):
        """Test automatic adjustment of excessive parameters."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Test automatic reduction of excessive max_total_tokens
        # This should not raise an error but should log a warning and adjust the value
        try:
            result = await rag_instance.query("test query", max_total_tokens=100000)
            # Should succeed with adjusted parameters
            assert result is not None
            assert 'content' in result
        except Exception as e:
            # Should not raise an error for automatic adjustment
            assert False, f"Automatic parameter adjustment should not raise error: {e}"
    
    @pytest.mark.asyncio
    async def test_parameter_type_conversion(self, rag_instance):
        """Test automatic type conversion for valid string parameters."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Test string-to-int conversion for valid numeric strings
        try:
            result = await rag_instance.query("test query", top_k="5")
            assert result is not None
            assert 'content' in result
        except Exception as e:
            assert False, f"Valid string-to-int conversion should succeed: {e}"
        
        try:
            result = await rag_instance.query("test query", max_total_tokens="1000")
            assert result is not None
            assert 'content' in result
        except Exception as e:
            assert False, f"Valid string-to-int conversion should succeed: {e}"
    
    @pytest.mark.asyncio
    async def test_valid_parameter_combinations(self, rag_instance):
        """Test valid parameter combinations work correctly."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Test all valid modes
        valid_modes = ['naive', 'local', 'global', 'hybrid']
        
        for mode in valid_modes:
            try:
                result = await rag_instance.query("test query", mode=mode)
                assert result is not None
                assert 'content' in result
                assert result['query_mode'] == mode
            except Exception as e:
                assert False, f"Valid mode '{mode}' should work: {e}"
    
    @pytest.mark.asyncio
    async def test_performance_warning_combinations(self, rag_instance):
        """Test that performance warning combinations are handled correctly."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Mock logger to capture warnings
        mock_logger = Mock()
        rag_instance.logger = mock_logger
        
        # Test high top_k with large max_total_tokens (should generate warning but succeed)
        try:
            result = await rag_instance.query(
                "test query", 
                top_k=80,  # High but valid
                max_total_tokens=900  # Within model limit but large
            )
            assert result is not None
            # Should have logged a warning about performance impact
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if "may cause long response times" in str(call)]
            assert len(warning_calls) > 0, "Should log performance warning"
        except Exception as e:
            assert False, f"Valid but high parameters should succeed with warning: {e}"
    
    def test_error_message_quality(self, rag_instance):
        """Test that error messages are helpful and informative."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Capture error messages and verify they contain helpful information
        error_scenarios = [
            ("mode", "invalid_mode", ["Invalid mode", "Must be one of"]),
            ("top_k", "invalid", ["top_k must be an integer"]),
            ("top_k", -5, ["top_k must be positive"]),
            ("max_total_tokens", -100, ["max_total_tokens must be positive"]),
        ]
        
        async def check_error_scenario(param_name, param_value, expected_phrases):
            try:
                await rag_instance.query("test query", **{param_name: param_value})
                assert False, f"Should have raised error for {param_name}={param_value}"
            except ClinicalMetabolomicsRAGError as e:
                error_message = str(e)
                for phrase in expected_phrases:
                    assert phrase in error_message, f"Error message should contain '{phrase}': {error_message}"
        
        # Run all scenarios
        for param_name, param_value, expected_phrases in error_scenarios:
            asyncio.run(check_error_scenario(param_name, param_value, expected_phrases))


async def run_comprehensive_validation():
    """Run comprehensive validation of enhanced error handling."""
    print("=" * 70)
    print("ENHANCED QUERY ERROR HANDLING VALIDATION")
    print("CMO-LIGHTRAG-007-T04 Final Verification")
    print("=" * 70)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available - cannot run validation")
        return False
    
    passed_tests = 0
    failed_tests = 0
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = LightRAGConfig(
            working_dir=Path(tmp_dir) / "test_kb",
            api_key="test-key-123",
            model="gpt-4o-mini",
            max_tokens=1000,
            embedding_model="text-embedding-3-large"
        )
        
        rag = ClinicalMetabolomicsRAG(config=config)
        
        print("🔧 Testing Enhanced QueryParam Error Handling...")
        
        # Test 1: Invalid mode validation
        print("\n1. Invalid Mode Validation")
        try:
            await rag.query("test", mode="invalid_mode")
            print("  ❌ Invalid mode should have been caught")
            failed_tests += 1
        except ClinicalMetabolomicsRAGError as e:
            if "Invalid mode 'invalid_mode'" in str(e):
                print("  ✅ Invalid mode properly validated")
                passed_tests += 1
            else:
                print(f"  ❌ Unexpected error message: {e}")
                failed_tests += 1
        
        # Test 2: Type validation
        print("\n2. Parameter Type Validation")
        try:
            await rag.query("test", top_k="invalid")
            print("  ❌ Invalid top_k type should have been caught")
            failed_tests += 1
        except ClinicalMetabolomicsRAGError as e:
            if "top_k must be an integer" in str(e):
                print("  ✅ Invalid top_k type properly validated")
                passed_tests += 1
            else:
                print(f"  ❌ Unexpected error message: {e}")
                failed_tests += 1
        
        # Test 3: Range validation
        print("\n3. Parameter Range Validation")
        try:
            await rag.query("test", top_k=-5)
            print("  ❌ Negative top_k should have been caught")
            failed_tests += 1
        except ClinicalMetabolomicsRAGError as e:
            if "top_k must be positive" in str(e):
                print("  ✅ Negative top_k properly validated")
                passed_tests += 1
            else:
                print(f"  ❌ Unexpected error message: {e}")
                failed_tests += 1
        
        # Test 4: Automatic adjustment
        print("\n4. Automatic Parameter Adjustment")
        try:
            result = await rag.query("test", max_total_tokens=100000)
            print("  ✅ Excessive max_total_tokens automatically adjusted")
            passed_tests += 1
        except Exception as e:
            print(f"  ❌ Automatic adjustment failed: {e}")
            failed_tests += 1
        
        # Test 5: Type conversion
        print("\n5. Automatic Type Conversion")
        try:
            result = await rag.query("test", top_k="5")
            print("  ✅ String-to-int conversion working")
            passed_tests += 1
        except Exception as e:
            print(f"  ❌ Type conversion failed: {e}")
            failed_tests += 1
        
        # Test 6: Valid operations
        print("\n6. Valid Parameter Operations")
        valid_modes = ['naive', 'local', 'global', 'hybrid']
        mode_tests = 0
        for mode in valid_modes:
            try:
                result = await rag.query("test", mode=mode)
                mode_tests += 1
            except Exception as e:
                print(f"  ❌ Valid mode '{mode}' failed: {e}")
        
        if mode_tests == len(valid_modes):
            print("  ✅ All valid modes working correctly")
            passed_tests += 1
        else:
            print(f"  ❌ Only {mode_tests}/{len(valid_modes)} valid modes working")
            failed_tests += 1
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"✅ Passed Tests: {passed_tests}")
        print(f"❌ Failed Tests: {failed_tests}")
        print(f"📊 Success Rate: {passed_tests/(passed_tests + failed_tests)*100:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 ALL ENHANCED ERROR HANDLING TESTS PASSED!")
            print("✅ CMO-LIGHTRAG-007-T04 IMPLEMENTATION COMPLETE")
            return True
        else:
            print(f"\n⚠️  {failed_tests} tests failed - needs attention")
            return False


if __name__ == "__main__":
    print("Starting Enhanced Query Error Handling Validation...")
    
    try:
        success = asyncio.run(run_comprehensive_validation())
        if success:
            print("\n🎉 Enhanced error handling validation completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Enhanced error handling validation failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)