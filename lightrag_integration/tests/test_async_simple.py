#!/usr/bin/env python3
"""
Simple async test to verify pytest-asyncio configuration.

This is a minimal test to check if async testing is working.
"""

import pytest
import asyncio


@pytest.mark.asyncio
async def test_simple_async():
    """Simple async test without fixtures."""
    
    async def async_operation():
        await asyncio.sleep(0.01)
        return "success"
    
    result = await async_operation()
    assert result == "success"


@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent async operations."""
    
    async def async_task(delay, result):
        await asyncio.sleep(delay)
        return result
    
    # Run multiple tasks concurrently
    tasks = [
        async_task(0.01, f"result_{i}")
        for i in range(3)
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result == f"result_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])