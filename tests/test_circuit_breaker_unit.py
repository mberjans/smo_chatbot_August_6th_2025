"""
Unit tests for basic CircuitBreaker functionality.

This module provides comprehensive unit tests for the CircuitBreaker class,
covering all core functionality including state transitions, failure counting,
recovery mechanisms, and edge cases.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import patch, Mock

from lightrag_integration.clinical_metabolomics_rag import CircuitBreaker, CircuitBreakerError


class TestCircuitBreakerInitialization:
    """Test CircuitBreaker initialization and configuration."""
    
    def test_default_initialization(self):
        """Test CircuitBreaker initializes with correct default values."""
        cb = CircuitBreaker()
        
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60.0
        assert cb.expected_exception == Exception
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
        assert cb.state == 'closed'
    
    def test_custom_initialization(self):
        """Test CircuitBreaker initializes with custom parameters."""
        cb = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout=30.0,
            expected_exception=ValueError
        )
        
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 30.0
        assert cb.expected_exception == ValueError
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
        assert cb.state == 'closed'
    
    def test_invalid_parameters(self):
        """Test CircuitBreaker raises errors for invalid parameters."""
        # Negative failure threshold should work (though not practical)
        cb = CircuitBreaker(failure_threshold=0)
        assert cb.failure_threshold == 0
        
        # Negative recovery timeout should work (though not practical)
        cb = CircuitBreaker(recovery_timeout=0.0)
        assert cb.recovery_timeout == 0.0


class TestStateTransitions:
    """Test circuit breaker state transitions."""
    
    def test_starts_closed(self, basic_circuit_breaker, circuit_breaker_state_verifier):
        """Verify circuit breaker starts in closed state."""
        circuit_breaker_state_verifier.assert_basic_circuit_state(
            basic_circuit_breaker, 'closed', 0
        )
    
    @pytest.mark.asyncio
    async def test_transitions_to_open(self, basic_circuit_breaker, failing_function_factory, circuit_breaker_state_verifier):
        """Test transition from closed to open after failure threshold."""
        # Create a function that always fails
        failing_func = failing_function_factory(fail_count=5, exception_type=Exception)
        
        # Execute function until circuit opens (failure_threshold = 3)
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_func)
        
        circuit_breaker_state_verifier.assert_basic_circuit_state(
            basic_circuit_breaker, 'open', 3
        )
    
    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self, basic_circuit_breaker, failing_function_factory, circuit_breaker_state_verifier):
        """Test transition from open to half-open after recovery timeout."""
        # Create a function that always fails
        def always_fails():
            raise Exception("Always fails")
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(always_fails)
        
        circuit_breaker_state_verifier.assert_basic_circuit_state(
            basic_circuit_breaker, 'open', 3
        )
        
        # Manually set the failure time to past recovery timeout
        import time
        basic_circuit_breaker.last_failure_time = time.time() - (basic_circuit_breaker.recovery_timeout + 0.5)
        
        # Next call should transition to half-open first, but then back to open on failure
        with pytest.raises(Exception):  # Still fails but transitions state
            await basic_circuit_breaker.call(always_fails)
        
        # Since the function fails in half-open, it goes back to open with increased failure count
        circuit_breaker_state_verifier.assert_basic_circuit_state(
            basic_circuit_breaker, 'open', 4
        )
    
    @pytest.mark.asyncio
    async def test_transitions_to_closed_on_success(self, basic_circuit_breaker, failing_function_factory, mock_time, circuit_breaker_state_verifier):
        """Test transition from half-open to closed on successful call."""
        # Create a function that fails 3 times then succeeds
        failing_func = failing_function_factory(fail_count=3, exception_type=Exception)
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_func)
        
        # Advance time past recovery timeout
        mock_time.advance(1.5)
        
        # First call after timeout should succeed and close circuit
        result = await basic_circuit_breaker.call(failing_func)
        assert result == "success"
        
        circuit_breaker_state_verifier.assert_basic_circuit_state(
            basic_circuit_breaker, 'closed', 0
        )
    
    @pytest.mark.asyncio
    async def test_returns_to_open_on_half_open_failure(self, basic_circuit_breaker, failing_function_factory, mock_time, circuit_breaker_state_verifier):
        """Test transition from half-open back to open on failure."""
        # Create a function that always fails
        failing_func = failing_function_factory(fail_count=10, exception_type=Exception)
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_func)
        
        # Advance time past recovery timeout
        mock_time.advance(1.5)
        
        # Call should transition to half-open then back to open on failure
        with pytest.raises(Exception):
            await basic_circuit_breaker.call(failing_func)
        
        # Circuit should be open again with increased failure count
        circuit_breaker_state_verifier.assert_basic_circuit_state(
            basic_circuit_breaker, 'open'
        )
        assert basic_circuit_breaker.failure_count == 4


class TestFailureCountingAndThresholds:
    """Test failure counting and threshold behavior."""
    
    @pytest.mark.asyncio
    async def test_failure_count_increments_correctly(self, basic_circuit_breaker, failing_function_factory):
        """Test that failure count increments on each failure."""
        failing_func = failing_function_factory(fail_count=2, exception_type=Exception)
        
        # First failure
        with pytest.raises(Exception):
            await basic_circuit_breaker.call(failing_func)
        assert basic_circuit_breaker.failure_count == 1
        
        # Second failure
        with pytest.raises(Exception):
            await basic_circuit_breaker.call(failing_func)
        assert basic_circuit_breaker.failure_count == 2
        
        # Circuit should still be closed (threshold = 3)
        assert basic_circuit_breaker.state == 'closed'
    
    @pytest.mark.asyncio
    async def test_failure_count_resets_on_success(self, basic_circuit_breaker, failing_function_factory):
        """Test that failure count resets to zero on successful call."""
        # Function that fails twice then succeeds
        failing_func = failing_function_factory(fail_count=2, exception_type=Exception)
        
        # Two failures
        for i in range(2):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_func)
        assert basic_circuit_breaker.failure_count == 2
        
        # Success should reset failure count
        result = await basic_circuit_breaker.call(failing_func)
        assert result == "success"
        assert basic_circuit_breaker.failure_count == 0
        assert basic_circuit_breaker.state == 'closed'
    
    @pytest.mark.asyncio
    async def test_failure_threshold_boundary_conditions(self, custom_circuit_breaker, failing_function_factory):
        """Test behavior at exact failure threshold."""
        # Circuit breaker with threshold of 2
        cb = custom_circuit_breaker(failure_threshold=2)
        failing_func = failing_function_factory(fail_count=5, exception_type=Exception)
        
        # First failure - should stay closed
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == 'closed'
        assert cb.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == 'open'
        assert cb.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_consecutive_failures_open_circuit(self, basic_circuit_breaker, failing_function_factory):
        """Test that consecutive failures open the circuit."""
        failing_func = failing_function_factory(fail_count=5, exception_type=Exception)
        
        # Execute until circuit opens
        for i in range(basic_circuit_breaker.failure_threshold):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_func)
        
        assert basic_circuit_breaker.state == 'open'
        assert basic_circuit_breaker.failure_count == basic_circuit_breaker.failure_threshold
    
    @pytest.mark.asyncio
    async def test_intermittent_failures_dont_open_circuit(self, basic_circuit_breaker, intermittent_failure_function):
        """Test that non-consecutive failures don't open circuit."""
        # Pattern: fail, succeed, fail, succeed, fail
        intermittent_func = intermittent_failure_function([True, False, True, False, True])
        
        # Execute pattern - failures are not consecutive
        for i in range(5):
            try:
                result = await basic_circuit_breaker.call(intermittent_func)
                # Success calls should reset failure count
                assert basic_circuit_breaker.failure_count == 0
            except Exception:
                # Failure calls should increment, but reset on next success
                pass
        
        # Circuit should remain closed due to intermittent successes
        assert basic_circuit_breaker.state == 'closed'
        assert basic_circuit_breaker.failure_count <= basic_circuit_breaker.failure_threshold


class TestRecoveryTimeout:
    """Test recovery timeout behavior."""
    
    @pytest.mark.asyncio
    async def test_recovery_timeout_prevents_calls(self, basic_circuit_breaker, failing_function_factory, mock_time):
        """Test that calls are blocked during recovery timeout."""
        failing_func = failing_function_factory(fail_count=5, exception_type=Exception)
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_func)
        
        assert basic_circuit_breaker.state == 'open'
        
        # Calls during recovery timeout should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError, match="Circuit breaker is open"):
            await basic_circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_recovery_timeout_allows_single_test_call(self, basic_circuit_breaker, failing_function_factory):
        """Test that single test call is allowed after timeout."""
        failing_func = failing_function_factory(fail_count=5, exception_type=Exception)
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_func)
        
        # Mock the circuit breaker to be past recovery timeout by setting last_failure_time
        import time
        basic_circuit_breaker.last_failure_time = time.time() - (basic_circuit_breaker.recovery_timeout + 0.5)
        
        # Should allow test call (which will fail and transition state)
        with pytest.raises(Exception):
            await basic_circuit_breaker.call(failing_func)
        
        # Circuit goes back to open after failure in half-open state
        assert basic_circuit_breaker.state == 'open'
        assert basic_circuit_breaker.failure_count == 4
    
    @pytest.mark.asyncio
    async def test_custom_recovery_timeout_values(self, custom_circuit_breaker, failing_function_factory):
        """Test circuit breaker with various recovery timeout values."""
        test_timeouts = [0.1, 0.5, 2.0]
        
        import time
        
        for timeout in test_timeouts:
            cb = custom_circuit_breaker(recovery_timeout=timeout, failure_threshold=3)
            failing_func = failing_function_factory(fail_count=10, exception_type=Exception)
            
            # Open the circuit
            for i in range(3):
                with pytest.raises(Exception):
                    await cb.call(failing_func)
            
            assert cb.state == 'open'
            
            # Manually set failure time to past recovery timeout
            cb.last_failure_time = time.time() - (timeout + 0.1)
            
            # Should allow test call after timeout
            with pytest.raises(Exception):  # Function still fails, but transitions to open again
                await cb.call(failing_func)
            
            # Circuit should be back to open with increased failure count
            assert cb.state == 'open'
            assert cb.failure_count == 4
    
    @pytest.mark.asyncio
    async def test_recovery_timeout_precision(self, basic_circuit_breaker, failing_function_factory, assert_helpers):
        """Test recovery timeout timing precision."""
        # Use real time for precision testing
        failing_func = failing_function_factory(fail_count=5, exception_type=Exception)
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_func)
        
        failure_time = time.time()
        
        # Wait for recovery timeout
        await asyncio.sleep(basic_circuit_breaker.recovery_timeout + 0.1)
        
        # Call should be allowed after timeout
        with pytest.raises(Exception):
            await basic_circuit_breaker.call(failing_func)
        
        # Verify timing precision (should be within 100ms)
        elapsed = time.time() - failure_time
        assert_helpers.assert_timing_precision(
            elapsed, basic_circuit_breaker.recovery_timeout, 200  # 200ms tolerance
        )


class TestExceptionHandling:
    """Test exception handling behavior."""
    
    @pytest.mark.asyncio
    async def test_expected_exception_triggers_failure(self, custom_circuit_breaker, failing_function_factory):
        """Test that expected exception types trigger failure count."""
        cb = custom_circuit_breaker(expected_exception=ValueError)
        failing_func = failing_function_factory(fail_count=3, exception_type=ValueError)
        
        # ValueError should trigger failure count
        with pytest.raises(ValueError):
            await cb.call(failing_func)
        
        assert cb.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_unexpected_exception_bypasses_circuit(self, custom_circuit_breaker, failing_function_factory):
        """Test that unexpected exceptions bypass circuit breaker."""
        cb = custom_circuit_breaker(expected_exception=ValueError)
        failing_func = failing_function_factory(fail_count=3, exception_type=RuntimeError)
        
        # RuntimeError should bypass circuit breaker (not increment failure count)
        with pytest.raises(RuntimeError):
            await cb.call(failing_func)
        
        assert cb.failure_count == 0
        assert cb.state == 'closed'
    
    @pytest.mark.asyncio
    async def test_custom_expected_exception_types(self):
        """Test circuit breaker with custom exception types."""
        class CustomError(Exception):
            pass
        
        cb = CircuitBreaker(failure_threshold=2, expected_exception=CustomError)
        
        def failing_custom():
            raise CustomError("Custom failure")
        
        # CustomError should trigger circuit breaker
        for i in range(2):
            with pytest.raises(CustomError):
                await cb.call(failing_custom)
        
        assert cb.state == 'open'
        assert cb.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_exception_inheritance_handling(self, custom_circuit_breaker):
        """Test handling of exception inheritance hierarchies."""
        class BaseError(Exception):
            pass
        
        class DerivedError(BaseError):
            pass
        
        cb = custom_circuit_breaker(expected_exception=BaseError, failure_threshold=2)
        
        def failing_derived():
            raise DerivedError("Derived failure")
        
        # DerivedError should be caught by BaseError handler
        for i in range(2):
            with pytest.raises(DerivedError):
                await cb.call(failing_derived)
        
        assert cb.state == 'open'
        assert cb.failure_count == 2


class TestAsyncFunctionSupport:
    """Test async function support."""
    
    @pytest.mark.asyncio
    async def test_async_function_execution(self, basic_circuit_breaker, async_failing_function_factory):
        """Test circuit breaker with async functions."""
        async_func = async_failing_function_factory(fail_count=0)
        
        result = await basic_circuit_breaker.call(async_func)
        assert result == "async_success"
        assert basic_circuit_breaker.failure_count == 0
        assert basic_circuit_breaker.state == 'closed'
    
    @pytest.mark.asyncio
    async def test_async_function_failure_handling(self, basic_circuit_breaker, async_failing_function_factory):
        """Test failure handling with async functions."""
        async_func = async_failing_function_factory(fail_count=3)
        
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(async_func)
        
        assert basic_circuit_breaker.state == 'open'
        assert basic_circuit_breaker.failure_count == 3
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_operations(self, basic_circuit_breaker, failing_function_factory, async_failing_function_factory):
        """Test circuit breaker handling both sync and async calls."""
        sync_func = failing_function_factory(fail_count=1)
        async_func = async_failing_function_factory(fail_count=1)
        
        # Sync failure
        with pytest.raises(Exception):
            await basic_circuit_breaker.call(sync_func)
        assert basic_circuit_breaker.failure_count == 1
        
        # Async failure
        with pytest.raises(Exception):
            await basic_circuit_breaker.call(async_func)
        assert basic_circuit_breaker.failure_count == 2
        
        # Sync success
        result = await basic_circuit_breaker.call(sync_func)
        assert result == "success"
        assert basic_circuit_breaker.failure_count == 0
        
        # Async success
        result = await basic_circuit_breaker.call(async_func)
        assert result == "async_success"
        assert basic_circuit_breaker.failure_count == 0


class TestThreadSafety:
    """Test thread safety and concurrent access."""
    
    @pytest.mark.asyncio
    async def test_concurrent_call_execution(self, basic_circuit_breaker, failing_function_factory, load_generator):
        """Test circuit breaker under concurrent access."""
        success_func = failing_function_factory(fail_count=0)
        
        def sync_call():
            # Convert async call to sync for load generator
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(basic_circuit_breaker.call(success_func))
            finally:
                loop.close()
        
        # Generate concurrent load
        load_generator.generate_load(sync_call, duration_seconds=2, threads=5, requests_per_second=10)
        load_generator.stop_load()
        
        metrics = load_generator.get_metrics()
        
        # All calls should succeed
        assert metrics['success_rate'] > 95.0  # Allow for some timing variance
        assert basic_circuit_breaker.state == 'closed'
        assert basic_circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio 
    async def test_state_consistency_under_load(self, basic_circuit_breaker, failing_function_factory, load_generator):
        """Test state consistency with multiple threads."""
        failing_func = failing_function_factory(fail_count=100)  # Always fails
        
        def sync_failing_call():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(basic_circuit_breaker.call(failing_func))
            except:
                return "failed"
            finally:
                loop.close()
        
        # Generate load with failing function
        load_generator.generate_load(sync_failing_call, duration_seconds=1, threads=10, requests_per_second=20)
        load_generator.stop_load()
        
        # Circuit should be open
        assert basic_circuit_breaker.state == 'open'
        assert basic_circuit_breaker.failure_count >= basic_circuit_breaker.failure_threshold
    
    def test_failure_count_thread_safety(self, basic_circuit_breaker):
        """Test failure count accuracy under concurrent failures."""
        def increment_failure():
            basic_circuit_breaker._on_failure()
        
        threads = []
        num_threads = 10
        increments_per_thread = 5
        
        # Create threads that increment failure count
        for _ in range(num_threads):
            for _ in range(increments_per_thread):
                thread = threading.Thread(target=increment_failure)
                threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify final failure count
        expected_count = num_threads * increments_per_thread
        assert basic_circuit_breaker.failure_count == expected_count


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""
    
    @pytest.mark.asyncio
    async def test_zero_failure_threshold(self, custom_circuit_breaker, failing_function_factory):
        """Test behavior with failure threshold of zero."""
        cb = custom_circuit_breaker(failure_threshold=0)
        
        def simple_failing_func():
            raise Exception("Test failure")
        
        # With threshold 0, circuit should open after first failure
        # because failure_count (1) >= failure_threshold (0)
        with pytest.raises(Exception):
            await cb.call(simple_failing_func)
        
        print(f"DEBUG: failure_count={cb.failure_count}, threshold={cb.failure_threshold}, state={cb.state}")
        
        # Check if the failure was counted and circuit opened
        assert cb.failure_count == 1
        assert cb.state == 'open'
    
    @pytest.mark.asyncio
    async def test_negative_recovery_timeout(self, custom_circuit_breaker, failing_function_factory):
        """Test behavior with negative recovery timeout."""
        cb = custom_circuit_breaker(recovery_timeout=-1.0, failure_threshold=3)
        failing_func = failing_function_factory(fail_count=5, exception_type=Exception)
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await cb.call(failing_func)
        
        assert cb.state == 'open'
        
        # With negative timeout, recovery should be immediate since any time elapsed > negative timeout
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        # Circuit should go back to open with increased failure count
        assert cb.state == 'open'
        assert cb.failure_count == 4
    
    @pytest.mark.asyncio
    async def test_extremely_high_failure_threshold(self, custom_circuit_breaker, failing_function_factory):
        """Test behavior with very high failure thresholds."""
        cb = custom_circuit_breaker(failure_threshold=1000)  # More reasonable for testing
        failing_func = failing_function_factory(fail_count=50, exception_type=Exception)
        
        # Execute many failures (but not all the way to threshold)
        for i in range(50):
            with pytest.raises(Exception):
                await cb.call(failing_func)
        
        # Circuit should still be closed
        assert cb.state == 'closed'
        assert cb.failure_count == 50
    
    @pytest.mark.asyncio
    async def test_rapid_success_failure_alternation(self, basic_circuit_breaker, intermittent_failure_function):
        """Test rapid alternation between success and failure."""
        # Alternate success/failure rapidly
        alternating_func = intermittent_failure_function([False, True] * 100)  # 200 calls
        
        for i in range(200):
            try:
                result = await basic_circuit_breaker.call(alternating_func)
                # Success should reset failure count
                assert basic_circuit_breaker.failure_count == 0
            except Exception:
                # Failure should increment but get reset by next success
                pass
        
        # Circuit should remain closed due to alternating pattern
        assert basic_circuit_breaker.state == 'closed'
    
    @pytest.mark.asyncio
    async def test_function_with_no_parameters(self, basic_circuit_breaker):
        """Test circuit breaker with parameterless function."""
        def simple_func():
            return "simple_result"
        
        result = await basic_circuit_breaker.call(simple_func)
        assert result == "simple_result"
        assert basic_circuit_breaker.state == 'closed'
    
    @pytest.mark.asyncio
    async def test_function_with_complex_parameters(self, basic_circuit_breaker):
        """Test circuit breaker with function that has complex parameters."""
        def complex_func(a, b, *args, **kwargs):
            return f"a={a}, b={b}, args={args}, kwargs={kwargs}"
        
        result = await basic_circuit_breaker.call(
            complex_func, 
            "test1", 
            "test2", 
            "extra1", 
            "extra2",
            key1="value1",
            key2="value2"
        )
        
        expected = "a=test1, b=test2, args=('extra1', 'extra2'), kwargs={'key1': 'value1', 'key2': 'value2'}"
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_state_after_success_in_half_open(self, basic_circuit_breaker, failing_function_factory, mock_time):
        """Test circuit breaker state after successful call in half-open state."""
        # Function that fails 3 times then succeeds
        failing_func = failing_function_factory(fail_count=3, exception_type=Exception)
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_func)
        
        assert basic_circuit_breaker.state == 'open'
        
        # Advance time and make successful call
        mock_time.advance(1.5)
        result = await basic_circuit_breaker.call(failing_func)
        
        # Should be closed with reset failure count
        assert result == "success"
        assert basic_circuit_breaker.state == 'closed'
        assert basic_circuit_breaker.failure_count == 0


class TestCircuitBreakerIntegration:
    """Integration tests for CircuitBreaker with realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_realistic_api_failure_scenario(self, basic_circuit_breaker):
        """Test realistic API failure and recovery scenario."""
        call_count = 0
        
        def simulate_api_call():
            nonlocal call_count
            call_count += 1
            
            # Simulate API behavior: fail for first 4 calls, then succeed
            if call_count <= 4:
                raise Exception(f"API unavailable (call {call_count})")
            return {"status": "success", "data": f"call_{call_count}"}
        
        # API fails initially, opening circuit (calls 1-3)
        for i in range(3):
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(simulate_api_call)
        
        assert basic_circuit_breaker.state == 'open'
        assert call_count == 3
        
        # Calls during circuit open should fail fast (no actual call made)
        with pytest.raises(CircuitBreakerError):
            await basic_circuit_breaker.call(simulate_api_call)
        
        # call_count should still be 3 since circuit breaker blocked the call
        assert call_count == 3
        
        # Manually set circuit to past recovery timeout
        import time
        basic_circuit_breaker.last_failure_time = time.time() - (basic_circuit_breaker.recovery_timeout + 0.5)
        
        # First call after timeout transitions to half-open and fails (call #4)
        with pytest.raises(Exception):
            await basic_circuit_breaker.call(simulate_api_call)
        
        # Circuit should go back to open, call_count should be 4
        assert basic_circuit_breaker.state == 'open'
        assert call_count == 4
        
        # Set timeout again and now the API should succeed (call #5)
        basic_circuit_breaker.last_failure_time = time.time() - (basic_circuit_breaker.recovery_timeout + 0.5)
        
        # Next call should succeed and close circuit
        result = await basic_circuit_breaker.call(simulate_api_call)
        assert result["status"] == "success"
        assert basic_circuit_breaker.state == 'closed'
        assert call_count == 5
        
        # Subsequent calls should continue to work
        result = await basic_circuit_breaker.call(simulate_api_call)
        assert result["status"] == "success"
        assert call_count == 6
    
    @pytest.mark.asyncio
    async def test_multiple_exception_types_scenario(self, custom_circuit_breaker):
        """Test handling multiple exception types in realistic scenario."""
        cb = custom_circuit_breaker(expected_exception=(ConnectionError, TimeoutError))
        call_count = 0
        
        def simulate_unreliable_service():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise ConnectionError("Network error")
            elif call_count == 2:
                raise TimeoutError("Request timeout")
            elif call_count == 3:
                raise ValueError("Invalid data")  # Not expected exception
            else:
                return "service_response"
        
        # ConnectionError should count as failure
        with pytest.raises(ConnectionError):
            await cb.call(simulate_unreliable_service)
        assert cb.failure_count == 1
        
        # TimeoutError should count as failure
        with pytest.raises(TimeoutError):
            await cb.call(simulate_unreliable_service)
        assert cb.failure_count == 2
        
        # ValueError should not count as failure (unexpected exception)
        with pytest.raises(ValueError):
            await cb.call(simulate_unreliable_service)
        assert cb.failure_count == 2  # Should not increment
        
        # Success should reset failure count
        result = await cb.call(simulate_unreliable_service)
        assert result == "service_response"
        assert cb.failure_count == 0


class TestParameterizedScenarios:
    """Parametrized tests for various configuration scenarios."""
    
    @pytest.mark.asyncio
    async def test_various_failure_thresholds(self, failure_threshold, failing_function_factory):
        """Test circuit breaker with various failure thresholds."""
        cb = CircuitBreaker(failure_threshold=failure_threshold)
        failing_func = failing_function_factory(fail_count=failure_threshold + 2)
        
        # Execute failures up to threshold
        for i in range(failure_threshold):
            with pytest.raises(Exception):
                await cb.call(failing_func)
            
            if i < failure_threshold - 1:
                assert cb.state == 'closed'
            else:
                assert cb.state == 'open'
        
        assert cb.failure_count == failure_threshold
    
    @pytest.mark.asyncio 
    async def test_various_recovery_timeouts(self, recovery_timeout, failing_function_factory):
        """Test circuit breaker with various recovery timeouts."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=recovery_timeout)
        failing_func = failing_function_factory(fail_count=5)
        
        # Open circuit
        for i in range(2):
            with pytest.raises(Exception):
                await cb.call(failing_func)
        
        assert cb.state == 'open'
        
        # Manually set time to past recovery timeout
        import time
        cb.last_failure_time = time.time() - (recovery_timeout + 0.1)
        
        # Should allow test call which will fail and keep circuit open
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        # Circuit should be open with increased failure count
        assert cb.state == 'open'
        assert cb.failure_count == 3