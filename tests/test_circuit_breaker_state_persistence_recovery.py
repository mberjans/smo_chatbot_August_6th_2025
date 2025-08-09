"""
State Persistence and Recovery Tests for Circuit Breaker System

This module provides comprehensive tests for validating circuit breaker state persistence
across system restarts and configuration changes. Tests cover both basic CircuitBreaker
and CostBasedCircuitBreaker persistence scenarios.

Priority: 5 (Critical)
Purpose: Validate circuit breaker reliability across system failures/restarts

Test Categories:
- Basic state persistence and serialization
- System restart simulation and recovery
- Configuration hot-reloading without state loss  
- Multi-instance state synchronization
- Error handling for corrupted/missing state files
- Backup and restore mechanisms
"""

import pytest
import asyncio
import time
import json
import pickle
import tempfile
import shutil
import threading
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from pathlib import Path
import copy

# Import circuit breaker classes
from lightrag_integration.clinical_metabolomics_rag import CircuitBreaker, CircuitBreakerError
from lightrag_integration.cost_based_circuit_breaker import (
    CostBasedCircuitBreaker,
    CostThresholdRule,
    CostThresholdType,
    CircuitBreakerState,
    OperationCostEstimator,
    CostCircuitBreakerManager
)


# =============================================================================
# STATE PERSISTENCE UTILITIES
# =============================================================================

class StateManager:
    """Utility class for managing circuit breaker state persistence."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize state manager with optional custom base path."""
        self.base_path = Path(base_path) if base_path else Path(tempfile.mkdtemp())
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def serialize_basic_circuit_breaker_state(self, cb: CircuitBreaker) -> Dict[str, Any]:
        """Serialize basic circuit breaker state to dictionary."""
        return {
            'failure_threshold': cb.failure_threshold,
            'recovery_timeout': cb.recovery_timeout,
            'expected_exception': cb.expected_exception.__name__,
            'failure_count': cb.failure_count,
            'last_failure_time': cb.last_failure_time,
            'state': cb.state,
            'timestamp': time.time(),
            'version': '1.0'
        }
    
    def serialize_cost_circuit_breaker_state(self, cb: CostBasedCircuitBreaker) -> Dict[str, Any]:
        """Serialize cost-based circuit breaker state to dictionary."""
        return {
            'name': cb.name,
            'failure_threshold': cb.failure_threshold,
            'recovery_timeout': cb.recovery_timeout,
            'state': cb.state.value if hasattr(cb.state, 'value') else str(cb.state),
            'failure_count': cb.failure_count,
            'last_failure_time': cb.last_failure_time,
            'last_success_time': cb.last_success_time,
            'half_open_start_time': cb.half_open_start_time,
            'throttle_rate': cb._throttle_rate,
            'rule_cooldowns': cb._rule_cooldowns.copy(),
            'operation_stats': cb._operation_stats.copy(),
            'threshold_rules': [self._serialize_cost_rule(rule) for rule in cb.threshold_rules],
            'timestamp': time.time(),
            'version': '2.0'
        }
    
    def _serialize_cost_rule(self, rule: CostThresholdRule) -> Dict[str, Any]:
        """Serialize cost threshold rule."""
        return {
            'rule_id': rule.rule_id,
            'threshold_type': rule.threshold_type.value,
            'threshold_value': rule.threshold_value,
            'action': rule.action,
            'priority': rule.priority,
            'applies_to_operations': rule.applies_to_operations,
            'applies_to_categories': rule.applies_to_categories,
            'time_window_minutes': rule.time_window_minutes,
            'throttle_factor': rule.throttle_factor,
            'allow_emergency_override': rule.allow_emergency_override,
            'cooldown_minutes': rule.cooldown_minutes,
            'recovery_threshold': rule.recovery_threshold,
            'recovery_window_minutes': rule.recovery_window_minutes
        }
    
    def deserialize_basic_circuit_breaker_state(self, state_data: Dict[str, Any]) -> CircuitBreaker:
        """Deserialize basic circuit breaker from state dictionary."""
        # Map exception name back to type
        exception_map = {
            'Exception': Exception,
            'ValueError': ValueError,
            'RuntimeError': RuntimeError,
            'ConnectionError': ConnectionError
        }
        exception_type = exception_map.get(state_data.get('expected_exception', 'Exception'), Exception)
        
        # Create circuit breaker with original configuration
        cb = CircuitBreaker(
            failure_threshold=state_data['failure_threshold'],
            recovery_timeout=state_data['recovery_timeout'],
            expected_exception=exception_type
        )
        
        # Restore runtime state
        cb.failure_count = state_data['failure_count']
        cb.last_failure_time = state_data['last_failure_time']
        cb.state = state_data['state']
        
        return cb
    
    def deserialize_cost_circuit_breaker_state(self, state_data: Dict[str, Any], 
                                              budget_manager: Mock, 
                                              cost_estimator: Mock) -> CostBasedCircuitBreaker:
        """Deserialize cost-based circuit breaker from state dictionary."""
        # Deserialize threshold rules
        threshold_rules = []
        for rule_data in state_data.get('threshold_rules', []):
            rule = CostThresholdRule(
                rule_id=rule_data['rule_id'],
                threshold_type=CostThresholdType(rule_data['threshold_type']),
                threshold_value=rule_data['threshold_value'],
                action=rule_data['action'],
                priority=rule_data['priority'],
                applies_to_operations=rule_data.get('applies_to_operations'),
                applies_to_categories=rule_data.get('applies_to_categories'),
                time_window_minutes=rule_data.get('time_window_minutes'),
                throttle_factor=rule_data['throttle_factor'],
                allow_emergency_override=rule_data['allow_emergency_override'],
                cooldown_minutes=rule_data['cooldown_minutes'],
                recovery_threshold=rule_data.get('recovery_threshold'),
                recovery_window_minutes=rule_data['recovery_window_minutes']
            )
            threshold_rules.append(rule)
        
        # Create circuit breaker with original configuration
        cb = CostBasedCircuitBreaker(
            name=state_data['name'],
            budget_manager=budget_manager,
            cost_estimator=cost_estimator,
            threshold_rules=threshold_rules,
            failure_threshold=state_data['failure_threshold'],
            recovery_timeout=state_data['recovery_timeout']
        )
        
        # Restore runtime state
        state_value = state_data['state']
        if isinstance(state_value, str):
            cb.state = CircuitBreakerState(state_value)
        else:
            cb.state = state_value
            
        cb.failure_count = state_data['failure_count']
        cb.last_failure_time = state_data['last_failure_time']
        cb.last_success_time = state_data['last_success_time']
        cb.half_open_start_time = state_data['half_open_start_time']
        cb._throttle_rate = state_data['throttle_rate']
        cb._rule_cooldowns = state_data.get('rule_cooldowns', {}).copy()
        cb._operation_stats = state_data.get('operation_stats', {}).copy()
        
        return cb
    
    def save_state_to_file(self, state_data: Dict[str, Any], filename: str, format_type: str = 'json') -> Path:
        """Save state data to file in specified format."""
        file_path = self.base_path / filename
        
        try:
            if format_type == 'json':
                with open(file_path, 'w') as f:
                    json.dump(state_data, f, indent=2, default=str)
            elif format_type == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(state_data, f)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
            
            return file_path
        except Exception as e:
            raise RuntimeError(f"Failed to save state to {file_path}: {e}")
    
    def load_state_from_file(self, filename: str, format_type: str = 'json') -> Dict[str, Any]:
        """Load state data from file in specified format."""
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"State file not found: {file_path}")
        
        try:
            if format_type == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif format_type == 'pickle':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to load state from {file_path}: {e}")
    
    def create_corrupted_state_file(self, filename: str, corruption_type: str = 'invalid_json') -> Path:
        """Create corrupted state file for error handling tests."""
        file_path = self.base_path / filename
        
        corrupted_data = {
            'invalid_json': '{"invalid": json content}',
            'empty_file': '',
            'binary_garbage': b'\x00\x01\x02\x03\xff\xfe\xfd',
            'partial_json': '{"name": "test", "state":'
        }
        
        content = corrupted_data.get(corruption_type, corrupted_data['invalid_json'])
        
        mode = 'wb' if isinstance(content, bytes) else 'w'
        with open(file_path, mode) as f:
            f.write(content)
        
        return file_path
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state_manager():
    """Provide StateManager instance for testing."""
    manager = StateManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def persistence_directory():
    """Provide temporary directory for persistence testing."""
    temp_dir = tempfile.mkdtemp(prefix="circuit_breaker_persistence_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def system_restart_simulator():
    """Provide system restart simulation utilities."""
    class SystemRestartSimulator:
        def __init__(self):
            self.pre_restart_state = {}
            self.post_restart_objects = {}
        
        def capture_pre_restart_state(self, objects: Dict[str, Any]) -> None:
            """Capture state of objects before simulated restart."""
            for name, obj in objects.items():
                if hasattr(obj, '__dict__'):
                    # Deep copy the object state
                    self.pre_restart_state[name] = copy.deepcopy(obj.__dict__)
        
        def simulate_restart(self, factory_functions: Dict[str, callable]) -> Dict[str, Any]:
            """Simulate system restart by creating new object instances."""
            self.post_restart_objects = {}
            for name, factory_func in factory_functions.items():
                self.post_restart_objects[name] = factory_func()
            return self.post_restart_objects
        
        def verify_state_restoration(self, name: str, exclude_fields: List[str] = None) -> bool:
            """Verify that state was properly restored after restart."""
            exclude_fields = exclude_fields or []
            
            if name not in self.pre_restart_state or name not in self.post_restart_objects:
                return False
            
            pre_state = self.pre_restart_state[name]
            post_state = self.post_restart_objects[name].__dict__
            
            for key, pre_value in pre_state.items():
                if key in exclude_fields:
                    continue
                    
                if key not in post_state:
                    return False
                
                post_value = post_state[key]
                if pre_value != post_value:
                    return False
            
            return True
    
    return SystemRestartSimulator()


# =============================================================================
# BASIC STATE PERSISTENCE TESTS
# =============================================================================

class TestBasicCircuitBreakerStatePersistence:
    """Test state persistence for basic CircuitBreaker."""
    
    def test_circuit_breaker_state_serialization(self, basic_circuit_breaker, state_manager):
        """Test basic circuit breaker state can be serialized and deserialized."""
        # Modify circuit breaker state
        basic_circuit_breaker.failure_count = 3
        basic_circuit_breaker.last_failure_time = time.time()
        basic_circuit_breaker.state = 'open'
        
        # Serialize state
        state_data = state_manager.serialize_basic_circuit_breaker_state(basic_circuit_breaker)
        
        # Verify serialized data contains expected fields
        assert state_data['failure_count'] == 3
        assert state_data['state'] == 'open'
        assert state_data['last_failure_time'] is not None
        assert state_data['failure_threshold'] == basic_circuit_breaker.failure_threshold
        assert state_data['recovery_timeout'] == basic_circuit_breaker.recovery_timeout
        assert 'timestamp' in state_data
        assert 'version' in state_data
    
    def test_circuit_breaker_state_deserialization(self, state_manager):
        """Test basic circuit breaker can be restored from serialized state."""
        # Create original state data
        original_state = {
            'failure_threshold': 5,
            'recovery_timeout': 30.0,
            'expected_exception': 'ValueError',
            'failure_count': 2,
            'last_failure_time': 1234567890.5,
            'state': 'half-open',
            'timestamp': time.time(),
            'version': '1.0'
        }
        
        # Deserialize to circuit breaker
        restored_cb = state_manager.deserialize_basic_circuit_breaker_state(original_state)
        
        # Verify restored state
        assert restored_cb.failure_threshold == 5
        assert restored_cb.recovery_timeout == 30.0
        assert restored_cb.expected_exception == ValueError
        assert restored_cb.failure_count == 2
        assert restored_cb.last_failure_time == 1234567890.5
        assert restored_cb.state == 'half-open'
    
    def test_circuit_breaker_json_persistence(self, basic_circuit_breaker, state_manager):
        """Test circuit breaker state persistence to/from JSON file."""
        # Set up circuit breaker state
        basic_circuit_breaker.failure_count = 4
        basic_circuit_breaker.state = 'open'
        basic_circuit_breaker.last_failure_time = time.time()
        
        # Serialize and save to JSON
        state_data = state_manager.serialize_basic_circuit_breaker_state(basic_circuit_breaker)
        file_path = state_manager.save_state_to_file(state_data, 'basic_cb_state.json', 'json')
        
        # Verify file exists
        assert file_path.exists()
        
        # Load state from file
        loaded_state = state_manager.load_state_from_file('basic_cb_state.json', 'json')
        
        # Verify loaded state matches original
        assert loaded_state['failure_count'] == 4
        assert loaded_state['state'] == 'open'
        assert loaded_state['last_failure_time'] is not None
    
    def test_circuit_breaker_pickle_persistence(self, basic_circuit_breaker, state_manager):
        """Test circuit breaker state persistence to/from pickle file."""
        # Set up circuit breaker state
        basic_circuit_breaker.failure_count = 2
        basic_circuit_breaker.state = 'closed'
        
        # Serialize and save to pickle
        state_data = state_manager.serialize_basic_circuit_breaker_state(basic_circuit_breaker)
        file_path = state_manager.save_state_to_file(state_data, 'basic_cb_state.pkl', 'pickle')
        
        # Verify file exists
        assert file_path.exists()
        
        # Load state from file
        loaded_state = state_manager.load_state_from_file('basic_cb_state.pkl', 'pickle')
        
        # Verify loaded state matches original
        assert loaded_state['failure_count'] == 2
        assert loaded_state['state'] == 'closed'


class TestCostBasedCircuitBreakerStatePersistence:
    """Test state persistence for CostBasedCircuitBreaker."""
    
    def test_cost_circuit_breaker_state_serialization(self, cost_based_circuit_breaker, state_manager):
        """Test cost-based circuit breaker state can be serialized."""
        # Modify circuit breaker state
        cost_based_circuit_breaker.state = CircuitBreakerState.BUDGET_LIMITED
        cost_based_circuit_breaker.failure_count = 2
        cost_based_circuit_breaker._throttle_rate = 0.5
        cost_based_circuit_breaker._rule_cooldowns = {'test_rule': time.time()}
        cost_based_circuit_breaker._operation_stats['total_calls'] = 100
        
        # Serialize state
        state_data = state_manager.serialize_cost_circuit_breaker_state(cost_based_circuit_breaker)
        
        # Verify serialized data
        assert state_data['name'] == 'test_breaker'
        assert state_data['state'] == 'budget_limited'
        assert state_data['failure_count'] == 2
        assert state_data['throttle_rate'] == 0.5
        assert 'test_rule' in state_data['rule_cooldowns']
        assert state_data['operation_stats']['total_calls'] == 100
        assert len(state_data['threshold_rules']) > 0
        assert 'timestamp' in state_data
        assert 'version' in state_data
    
    def test_cost_circuit_breaker_state_deserialization(self, mock_budget_manager, mock_cost_estimator, state_manager):
        """Test cost-based circuit breaker can be restored from serialized state."""
        # Create comprehensive state data
        original_state = {
            'name': 'restored_breaker',
            'failure_threshold': 5,
            'recovery_timeout': 60.0,
            'state': 'open',
            'failure_count': 3,
            'last_failure_time': 1234567890.5,
            'last_success_time': 1234567880.0,
            'half_open_start_time': None,
            'throttle_rate': 0.7,
            'rule_cooldowns': {'rule_1': 1234567895.0},
            'operation_stats': {
                'total_calls': 50,
                'allowed_calls': 40,
                'blocked_calls': 10,
                'cost_savings': 15.50
            },
            'threshold_rules': [
                {
                    'rule_id': 'test_rule',
                    'threshold_type': 'percentage_daily',
                    'threshold_value': 90.0,
                    'action': 'throttle',
                    'priority': 10,
                    'applies_to_operations': None,
                    'applies_to_categories': None,
                    'time_window_minutes': None,
                    'throttle_factor': 0.5,
                    'allow_emergency_override': False,
                    'cooldown_minutes': 5.0,
                    'recovery_threshold': None,
                    'recovery_window_minutes': 10.0
                }
            ],
            'timestamp': time.time(),
            'version': '2.0'
        }
        
        # Deserialize to circuit breaker
        restored_cb = state_manager.deserialize_cost_circuit_breaker_state(
            original_state, mock_budget_manager, mock_cost_estimator
        )
        
        # Verify restored state
        assert restored_cb.name == 'restored_breaker'
        assert restored_cb.failure_threshold == 5
        assert restored_cb.recovery_timeout == 60.0
        assert restored_cb.state == CircuitBreakerState.OPEN
        assert restored_cb.failure_count == 3
        assert restored_cb.last_failure_time == 1234567890.5
        assert restored_cb.last_success_time == 1234567880.0
        assert restored_cb._throttle_rate == 0.7
        assert 'rule_1' in restored_cb._rule_cooldowns
        assert restored_cb._operation_stats['total_calls'] == 50
        assert restored_cb._operation_stats['cost_savings'] == 15.50
        assert len(restored_cb.threshold_rules) == 1
        assert restored_cb.threshold_rules[0].rule_id == 'test_rule'
    
    def test_cost_circuit_breaker_complex_persistence(self, cost_based_circuit_breaker, state_manager):
        """Test persistence of complex cost circuit breaker state with multiple rules."""
        # Set up complex state
        cost_based_circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        cost_based_circuit_breaker.failure_count = 1
        cost_based_circuit_breaker.half_open_start_time = time.time()
        cost_based_circuit_breaker._rule_cooldowns = {
            'daily_rule': time.time() - 100,
            'monthly_rule': time.time() - 50,
            'operation_rule': time.time() - 25
        }
        cost_based_circuit_breaker._operation_stats.update({
            'total_calls': 500,
            'allowed_calls': 450,
            'blocked_calls': 50,
            'throttled_calls': 25,
            'cost_blocked_calls': 30,
            'total_estimated_cost': 125.75,
            'total_actual_cost': 120.50,
            'cost_savings': 45.25
        })
        
        # Serialize to JSON file
        state_data = state_manager.serialize_cost_circuit_breaker_state(cost_based_circuit_breaker)
        file_path = state_manager.save_state_to_file(state_data, 'complex_cb_state.json', 'json')
        
        # Load and verify
        loaded_state = state_manager.load_state_from_file('complex_cb_state.json', 'json')
        
        assert loaded_state['state'] == 'half_open'
        assert loaded_state['failure_count'] == 1
        assert 'daily_rule' in loaded_state['rule_cooldowns']
        assert loaded_state['operation_stats']['total_calls'] == 500
        assert loaded_state['operation_stats']['cost_savings'] == 45.25


# =============================================================================
# SYSTEM RESTART SIMULATION TESTS
# =============================================================================

class TestSystemRestartRecovery:
    """Test circuit breaker state recovery after simulated system restarts."""
    
    def test_basic_circuit_breaker_restart_recovery(self, state_manager, system_restart_simulator):
        """Test basic circuit breaker state survives system restart."""
        # Create and configure original circuit breaker
        original_cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        original_cb.failure_count = 2
        original_cb.last_failure_time = time.time()
        original_cb.state = 'open'
        
        # Capture pre-restart state
        system_restart_simulator.capture_pre_restart_state({'cb': original_cb})
        
        # Save state to persistent storage
        state_data = state_manager.serialize_basic_circuit_breaker_state(original_cb)
        state_manager.save_state_to_file(state_data, 'restart_test.json')
        
        # Simulate system restart - create new circuit breaker from persisted state
        def create_restored_cb():
            loaded_state = state_manager.load_state_from_file('restart_test.json')
            return state_manager.deserialize_basic_circuit_breaker_state(loaded_state)
        
        restored_objects = system_restart_simulator.simulate_restart({'cb': create_restored_cb})
        restored_cb = restored_objects['cb']
        
        # Verify state restoration
        assert restored_cb.failure_count == original_cb.failure_count
        assert restored_cb.state == original_cb.state
        assert restored_cb.last_failure_time == original_cb.last_failure_time
        assert restored_cb.failure_threshold == original_cb.failure_threshold
        assert restored_cb.recovery_timeout == original_cb.recovery_timeout
    
    def test_cost_circuit_breaker_restart_recovery(self, mock_budget_manager, mock_cost_estimator, 
                                                  state_manager, system_restart_simulator):
        """Test cost circuit breaker state survives system restart."""
        # Create original circuit breaker
        threshold_rules = [
            CostThresholdRule(
                rule_id='restart_test_rule',
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=85.0,
                action='throttle',
                throttle_factor=0.6
            )
        ]
        
        original_cb = CostBasedCircuitBreaker(
            name='restart_test_cb',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=threshold_rules
        )
        
        # Set complex state
        original_cb.state = CircuitBreakerState.BUDGET_LIMITED
        original_cb.failure_count = 1
        original_cb._throttle_rate = 0.4
        original_cb._rule_cooldowns = {'restart_test_rule': time.time() - 10}
        original_cb._operation_stats['total_calls'] = 200
        
        # Save state
        state_data = state_manager.serialize_cost_circuit_breaker_state(original_cb)
        state_manager.save_state_to_file(state_data, 'cost_restart_test.json')
        
        # Simulate restart
        def create_restored_cost_cb():
            loaded_state = state_manager.load_state_from_file('cost_restart_test.json')
            return state_manager.deserialize_cost_circuit_breaker_state(
                loaded_state, mock_budget_manager, mock_cost_estimator
            )
        
        restored_objects = system_restart_simulator.simulate_restart({'cb': create_restored_cost_cb})
        restored_cb = restored_objects['cb']
        
        # Verify comprehensive state restoration
        assert restored_cb.name == original_cb.name
        assert restored_cb.state == original_cb.state
        assert restored_cb.failure_count == original_cb.failure_count
        assert restored_cb._throttle_rate == original_cb._throttle_rate
        assert restored_cb._rule_cooldowns == original_cb._rule_cooldowns
        assert restored_cb._operation_stats['total_calls'] == original_cb._operation_stats['total_calls']
        assert len(restored_cb.threshold_rules) == len(original_cb.threshold_rules)
        assert restored_cb.threshold_rules[0].rule_id == 'restart_test_rule'
    
    def test_multiple_circuit_breakers_restart_recovery(self, mock_budget_manager, mock_cost_estimator, 
                                                       state_manager, system_restart_simulator):
        """Test multiple circuit breakers can be restored after restart."""
        # Create multiple circuit breakers
        cb1 = CircuitBreaker(failure_threshold=2)
        cb1.failure_count = 1
        cb1.state = 'closed'
        
        cb2 = CircuitBreaker(failure_threshold=5)
        cb2.failure_count = 3
        cb2.state = 'open'
        cb2.last_failure_time = time.time()
        
        cost_rules = [CostThresholdRule('test_rule', CostThresholdType.OPERATION_COST, 1.0, 'block')]
        cb3 = CostBasedCircuitBreaker(
            name='cost_cb',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=cost_rules
        )
        cb3.state = CircuitBreakerState.BUDGET_LIMITED
        cb3._throttle_rate = 0.3
        
        # Save all states
        state1 = state_manager.serialize_basic_circuit_breaker_state(cb1)
        state2 = state_manager.serialize_basic_circuit_breaker_state(cb2)
        state3 = state_manager.serialize_cost_circuit_breaker_state(cb3)
        
        state_manager.save_state_to_file(state1, 'cb1_state.json')
        state_manager.save_state_to_file(state2, 'cb2_state.json')
        state_manager.save_state_to_file(state3, 'cb3_state.json')
        
        # Simulate restart with factory functions
        def create_restored_cb1():
            loaded = state_manager.load_state_from_file('cb1_state.json')
            return state_manager.deserialize_basic_circuit_breaker_state(loaded)
        
        def create_restored_cb2():
            loaded = state_manager.load_state_from_file('cb2_state.json')
            return state_manager.deserialize_basic_circuit_breaker_state(loaded)
        
        def create_restored_cb3():
            loaded = state_manager.load_state_from_file('cb3_state.json')
            return state_manager.deserialize_cost_circuit_breaker_state(
                loaded, mock_budget_manager, mock_cost_estimator
            )
        
        restored = system_restart_simulator.simulate_restart({
            'cb1': create_restored_cb1,
            'cb2': create_restored_cb2,
            'cb3': create_restored_cb3
        })
        
        # Verify all circuit breakers were restored correctly
        assert restored['cb1'].failure_count == 1
        assert restored['cb1'].state == 'closed'
        
        assert restored['cb2'].failure_count == 3
        assert restored['cb2'].state == 'open'
        
        assert restored['cb3'].state == CircuitBreakerState.BUDGET_LIMITED
        assert restored['cb3']._throttle_rate == 0.3


# =============================================================================
# CONFIGURATION HOT-RELOADING TESTS
# =============================================================================

class TestConfigurationHotReloading:
    """Test configuration updates without losing circuit breaker state."""
    
    def test_basic_circuit_breaker_config_update(self, state_manager):
        """Test updating basic circuit breaker configuration while preserving runtime state."""
        # Create circuit breaker with initial config
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        cb.failure_count = 2
        cb.state = 'half-open'
        cb.last_failure_time = time.time() - 10
        
        # Save current state
        original_state = state_manager.serialize_basic_circuit_breaker_state(cb)
        
        # Create new circuit breaker with updated config but preserved state
        updated_cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)  # Updated config
        
        # Restore state (preserving runtime state, updating config)
        updated_cb.failure_count = original_state['failure_count']
        updated_cb.state = original_state['state']
        updated_cb.last_failure_time = original_state['last_failure_time']
        
        # Verify config updated and state preserved
        assert updated_cb.failure_threshold == 5  # Updated
        assert updated_cb.recovery_timeout == 60.0  # Updated
        assert updated_cb.failure_count == 2  # Preserved
        assert updated_cb.state == 'half-open'  # Preserved
    
    def test_cost_circuit_breaker_rule_addition(self, mock_budget_manager, mock_cost_estimator, state_manager):
        """Test adding new cost rules without losing existing state."""
        # Create circuit breaker with initial rules
        initial_rules = [
            CostThresholdRule('rule1', CostThresholdType.PERCENTAGE_DAILY, 80.0, 'throttle')
        ]
        
        cb = CostBasedCircuitBreaker(
            name='config_test',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=initial_rules
        )
        
        # Set runtime state
        cb.state = CircuitBreakerState.CLOSED
        cb.failure_count = 1
        cb._throttle_rate = 0.8
        cb._rule_cooldowns = {'rule1': time.time() - 5}
        cb._operation_stats['total_calls'] = 50
        
        # Save current state
        original_state = state_manager.serialize_cost_circuit_breaker_state(cb)
        
        # Add new rule while preserving state
        updated_rules = initial_rules + [
            CostThresholdRule('rule2', CostThresholdType.OPERATION_COST, 2.0, 'block')
        ]
        
        updated_cb = CostBasedCircuitBreaker(
            name='config_test',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=updated_rules
        )
        
        # Restore runtime state
        updated_cb.state = CircuitBreakerState(original_state['state'])
        updated_cb.failure_count = original_state['failure_count']
        updated_cb._throttle_rate = original_state['throttle_rate']
        updated_cb._rule_cooldowns = original_state['rule_cooldowns'].copy()
        updated_cb._operation_stats = original_state['operation_stats'].copy()
        
        # Verify rule added and state preserved
        assert len(updated_cb.threshold_rules) == 2
        assert any(rule.rule_id == 'rule2' for rule in updated_cb.threshold_rules)
        assert updated_cb.state == CircuitBreakerState.CLOSED  # Preserved
        assert updated_cb.failure_count == 1  # Preserved
        assert updated_cb._throttle_rate == 0.8  # Preserved
        assert updated_cb._operation_stats['total_calls'] == 50  # Preserved
    
    def test_cost_circuit_breaker_rule_modification(self, mock_budget_manager, mock_cost_estimator, state_manager):
        """Test modifying existing cost rules while preserving state."""
        # Create circuit breaker with rules
        original_rule = CostThresholdRule(
            rule_id='modifiable_rule',
            threshold_type=CostThresholdType.PERCENTAGE_DAILY,
            threshold_value=80.0,
            action='throttle',
            throttle_factor=0.5
        )
        
        cb = CostBasedCircuitBreaker(
            name='modify_test',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=[original_rule]
        )
        
        # Set state including rule-specific cooldowns
        cb._rule_cooldowns = {'modifiable_rule': time.time() - 30}
        cb._operation_stats['cost_blocked_calls'] = 10
        
        # Save state
        original_state = state_manager.serialize_cost_circuit_breaker_state(cb)
        
        # Modify rule parameters
        modified_rule = CostThresholdRule(
            rule_id='modifiable_rule',  # Same ID
            threshold_type=CostThresholdType.PERCENTAGE_DAILY,
            threshold_value=70.0,  # Changed threshold
            action='block',  # Changed action
            throttle_factor=0.3  # Changed throttle factor
        )
        
        updated_cb = CostBasedCircuitBreaker(
            name='modify_test',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=[modified_rule]
        )
        
        # Restore state (cooldowns should be preserved for same rule ID)
        updated_cb._rule_cooldowns = original_state['rule_cooldowns'].copy()
        updated_cb._operation_stats = original_state['operation_stats'].copy()
        
        # Verify rule modified and relevant state preserved
        assert updated_cb.threshold_rules[0].threshold_value == 70.0  # Updated
        assert updated_cb.threshold_rules[0].action == 'block'  # Updated
        assert 'modifiable_rule' in updated_cb._rule_cooldowns  # Preserved
        assert updated_cb._operation_stats['cost_blocked_calls'] == 10  # Preserved


# =============================================================================
# MULTI-INSTANCE STATE SYNCHRONIZATION TESTS
# =============================================================================

class TestMultiInstanceStateSynchronization:
    """Test state synchronization across multiple circuit breaker instances."""
    
    def test_shared_state_storage(self, persistence_directory, mock_budget_manager, mock_cost_estimator):
        """Test multiple circuit breaker instances can share state storage."""
        shared_state_file = persistence_directory / "shared_cb_state.json"
        
        # Create first instance and set state
        cb1 = CostBasedCircuitBreaker(
            name='shared_cb',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=[CostThresholdRule('shared_rule', CostThresholdType.OPERATION_COST, 1.0, 'block')]
        )
        cb1._operation_stats['total_calls'] = 100
        cb1._rule_cooldowns = {'shared_rule': time.time()}
        
        # Save state to shared location
        state_manager = StateManager(str(persistence_directory))
        state_data = state_manager.serialize_cost_circuit_breaker_state(cb1)
        state_manager.save_state_to_file(state_data, shared_state_file.name)
        
        # Create second instance from shared state
        loaded_state = state_manager.load_state_from_file(shared_state_file.name)
        cb2 = state_manager.deserialize_cost_circuit_breaker_state(
            loaded_state, mock_budget_manager, mock_cost_estimator
        )
        
        # Verify both instances have same state
        assert cb1._operation_stats['total_calls'] == cb2._operation_stats['total_calls']
        assert cb1._rule_cooldowns.keys() == cb2._rule_cooldowns.keys()
        assert cb1.threshold_rules[0].rule_id == cb2.threshold_rules[0].rule_id
    
    def test_concurrent_state_updates(self, persistence_directory, mock_budget_manager, mock_cost_estimator):
        """Test handling of concurrent state updates from multiple instances."""
        state_manager = StateManager(str(persistence_directory))
        
        # Create two instances of same circuit breaker
        rule = CostThresholdRule('concurrent_rule', CostThresholdType.PERCENTAGE_DAILY, 90.0, 'throttle')
        
        cb1 = CostBasedCircuitBreaker(
            name='concurrent_cb',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=[rule]
        )
        
        cb2 = CostBasedCircuitBreaker(
            name='concurrent_cb',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=[rule]
        )
        
        # Simulate concurrent updates
        def update_instance1():
            cb1._operation_stats['total_calls'] = 50
            cb1._operation_stats['allowed_calls'] = 45
            state1 = state_manager.serialize_cost_circuit_breaker_state(cb1)
            state_manager.save_state_to_file(state1, 'concurrent_cb_1.json')
        
        def update_instance2():
            cb2._operation_stats['total_calls'] = 60
            cb2._operation_stats['blocked_calls'] = 10
            state2 = state_manager.serialize_cost_circuit_breaker_state(cb2)
            state_manager.save_state_to_file(state2, 'concurrent_cb_2.json')
        
        # Execute updates concurrently
        import threading
        t1 = threading.Thread(target=update_instance1)
        t2 = threading.Thread(target=update_instance2)
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        # Load both states and verify they were saved independently
        state1 = state_manager.load_state_from_file('concurrent_cb_1.json')
        state2 = state_manager.load_state_from_file('concurrent_cb_2.json')
        
        assert state1['operation_stats']['total_calls'] == 50
        assert state1['operation_stats']['allowed_calls'] == 45
        assert state2['operation_stats']['total_calls'] == 60
        assert state2['operation_stats']['blocked_calls'] == 10
    
    def test_state_consistency_verification(self, persistence_directory, mock_budget_manager, mock_cost_estimator):
        """Test verification of state consistency across instances."""
        state_manager = StateManager(str(persistence_directory))
        
        # Create reference circuit breaker with known state
        reference_cb = CostBasedCircuitBreaker(
            name='consistency_test',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=[CostThresholdRule('test_rule', CostThresholdType.OPERATION_COST, 0.5, 'alert_only')]
        )
        
        reference_cb.failure_count = 2
        reference_cb.state = CircuitBreakerState.HALF_OPEN
        reference_cb._operation_stats['total_calls'] = 200
        reference_cb._rule_cooldowns = {'test_rule': time.time() - 100}
        
        # Serialize reference state
        reference_state = state_manager.serialize_cost_circuit_breaker_state(reference_cb)
        state_manager.save_state_to_file(reference_state, 'reference_state.json')
        
        # Create multiple instances from same state
        instances = []
        for i in range(3):
            loaded_state = state_manager.load_state_from_file('reference_state.json')
            instance = state_manager.deserialize_cost_circuit_breaker_state(
                loaded_state, mock_budget_manager, mock_cost_estimator
            )
            instances.append(instance)
        
        # Verify all instances have consistent state
        for instance in instances:
            assert instance.failure_count == reference_cb.failure_count
            assert instance.state == reference_cb.state
            assert instance._operation_stats['total_calls'] == reference_cb._operation_stats['total_calls']
            assert instance._rule_cooldowns.keys() == reference_cb._rule_cooldowns.keys()


# =============================================================================
# ERROR HANDLING AND CORRUPTION RECOVERY TESTS
# =============================================================================

class TestStateCorruptionRecovery:
    """Test recovery from corrupted or missing state files."""
    
    def test_missing_state_file_handling(self, state_manager):
        """Test graceful handling when state file is missing."""
        # Attempt to load non-existent state file
        with pytest.raises(FileNotFoundError):
            state_manager.load_state_from_file('nonexistent_state.json')
    
    def test_corrupted_json_state_handling(self, state_manager):
        """Test handling of corrupted JSON state files."""
        # Create corrupted JSON file
        corrupted_file = state_manager.create_corrupted_state_file('corrupted.json', 'invalid_json')
        
        # Attempt to load corrupted file
        with pytest.raises(RuntimeError):
            state_manager.load_state_from_file('corrupted.json')
    
    def test_empty_state_file_handling(self, state_manager):
        """Test handling of empty state files."""
        # Create empty file
        empty_file = state_manager.create_corrupted_state_file('empty.json', 'empty_file')
        
        # Attempt to load empty file
        with pytest.raises(RuntimeError):
            state_manager.load_state_from_file('empty.json')
    
    def test_partial_state_file_handling(self, state_manager):
        """Test handling of partially written state files."""
        # Create partial JSON file
        partial_file = state_manager.create_corrupted_state_file('partial.json', 'partial_json')
        
        # Attempt to load partial file
        with pytest.raises(RuntimeError):
            state_manager.load_state_from_file('partial.json')
    
    def test_state_validation_during_deserialization(self, mock_budget_manager, mock_cost_estimator, state_manager):
        """Test validation of state data during deserialization."""
        # Create state data with missing required fields
        invalid_state = {
            'name': 'test_cb',
            'state': 'invalid_state',  # Invalid state value
            'failure_count': -1,  # Invalid negative count
            # Missing required fields
        }
        
        # Attempt to deserialize invalid state should handle gracefully
        with pytest.raises((ValueError, KeyError)):
            state_manager.deserialize_cost_circuit_breaker_state(
                invalid_state, mock_budget_manager, mock_cost_estimator
            )
    
    def test_fallback_to_default_configuration(self, mock_budget_manager, mock_cost_estimator):
        """Test fallback to default configuration when state loading fails."""
        # Simulate state loading failure by creating circuit breaker with defaults
        default_rules = [
            CostThresholdRule(
                rule_id='fallback_rule',
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=100.0,
                action='block'
            )
        ]
        
        fallback_cb = CostBasedCircuitBreaker(
            name='fallback_cb',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=default_rules,
            failure_threshold=5,  # Default
            recovery_timeout=60.0  # Default
        )
        
        # Verify fallback circuit breaker has expected defaults
        assert fallback_cb.failure_count == 0
        assert fallback_cb.state == CircuitBreakerState.CLOSED
        assert len(fallback_cb.threshold_rules) == 1
        assert fallback_cb.failure_threshold == 5
        assert fallback_cb.recovery_timeout == 60.0


# =============================================================================
# BACKUP AND RESTORE MECHANISMS TESTS
# =============================================================================

class TestBackupRestoreMechanisms:
    """Test backup and restore mechanisms for circuit breaker state."""
    
    def test_incremental_state_backup(self, cost_based_circuit_breaker, state_manager):
        """Test incremental backup of circuit breaker state changes."""
        backup_states = []
        
        # Initial state backup
        initial_state = state_manager.serialize_cost_circuit_breaker_state(cost_based_circuit_breaker)
        backup_states.append(('initial', initial_state.copy()))
        
        # Simulate state changes and backup each change
        cost_based_circuit_breaker.failure_count = 1
        state_1 = state_manager.serialize_cost_circuit_breaker_state(cost_based_circuit_breaker)
        backup_states.append(('failure_1', state_1.copy()))
        
        cost_based_circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        cost_based_circuit_breaker._throttle_rate = 0.7
        state_2 = state_manager.serialize_cost_circuit_breaker_state(cost_based_circuit_breaker)
        backup_states.append(('throttled', state_2.copy()))
        
        cost_based_circuit_breaker._operation_stats['total_calls'] = 100
        state_3 = state_manager.serialize_cost_circuit_breaker_state(cost_based_circuit_breaker)
        backup_states.append(('active', state_3.copy()))
        
        # Verify each backup captures incremental changes
        assert backup_states[0][1]['failure_count'] == 0
        assert backup_states[1][1]['failure_count'] == 1
        assert backup_states[2][1]['state'] == 'half_open'
        assert backup_states[2][1]['throttle_rate'] == 0.7
        assert backup_states[3][1]['operation_stats']['total_calls'] == 100
        
        # Verify timestamps show progression
        timestamps = [state[1]['timestamp'] for state in backup_states]
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    
    def test_point_in_time_state_restore(self, mock_budget_manager, mock_cost_estimator, state_manager):
        """Test restoring circuit breaker to specific point in time."""
        # Create circuit breaker and simulate progression through states
        cb = CostBasedCircuitBreaker(
            name='pit_restore_test',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=[CostThresholdRule('test_rule', CostThresholdType.OPERATION_COST, 1.0, 'block')]
        )
        
        # Checkpoint 1: Initial state
        cb._operation_stats['total_calls'] = 10
        checkpoint_1 = state_manager.serialize_cost_circuit_breaker_state(cb)
        state_manager.save_state_to_file(checkpoint_1, 'checkpoint_1.json')
        time.sleep(0.1)  # Ensure timestamp difference
        
        # Checkpoint 2: After some activity
        cb._operation_stats['total_calls'] = 50
        cb.failure_count = 1
        checkpoint_2 = state_manager.serialize_cost_circuit_breaker_state(cb)
        state_manager.save_state_to_file(checkpoint_2, 'checkpoint_2.json')
        time.sleep(0.1)
        
        # Checkpoint 3: After problems
        cb.state = CircuitBreakerState.OPEN
        cb.failure_count = 3
        cb._operation_stats['blocked_calls'] = 10
        checkpoint_3 = state_manager.serialize_cost_circuit_breaker_state(cb)
        state_manager.save_state_to_file(checkpoint_3, 'checkpoint_3.json')
        
        # Restore to checkpoint 2 (before problems)
        restore_state = state_manager.load_state_from_file('checkpoint_2.json')
        restored_cb = state_manager.deserialize_cost_circuit_breaker_state(
            restore_state, mock_budget_manager, mock_cost_estimator
        )
        
        # Verify restoration to correct point in time
        assert restored_cb._operation_stats['total_calls'] == 50
        assert restored_cb.failure_count == 1
        assert restored_cb.state == CircuitBreakerState.CLOSED  # Before it went open
        assert restored_cb._operation_stats.get('blocked_calls', 0) == 0  # Before blocks occurred
    
    def test_backup_file_rotation(self, cost_based_circuit_breaker, state_manager):
        """Test backup file rotation to prevent disk space issues."""
        max_backups = 3
        
        # Create multiple backup files
        for i in range(5):  # More than max_backups
            cost_based_circuit_breaker._operation_stats['total_calls'] = i * 10
            state_data = state_manager.serialize_cost_circuit_breaker_state(cost_based_circuit_breaker)
            state_manager.save_state_to_file(state_data, f'backup_{i:03d}.json')
            time.sleep(0.01)  # Small delay for timestamp differentiation
        
        # Get all backup files
        backup_files = list(state_manager.base_path.glob('backup_*.json'))
        
        # Simulate rotation logic (keep only most recent max_backups files)
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        files_to_keep = backup_files[:max_backups]
        files_to_remove = backup_files[max_backups:]
        
        # Remove old backups
        for file_path in files_to_remove:
            file_path.unlink()
        
        # Verify correct number of files remain
        remaining_files = list(state_manager.base_path.glob('backup_*.json'))
        assert len(remaining_files) == max_backups
        
        # Verify the most recent files were kept
        for kept_file in files_to_keep:
            assert kept_file.exists()
    
    def test_compressed_backup_storage(self, cost_based_circuit_breaker, state_manager):
        """Test compressed backup storage for space efficiency."""
        import gzip
        
        # Create large state data
        cost_based_circuit_breaker._operation_stats.update({
            'detailed_metrics': {f'metric_{i}': i * 0.1 for i in range(1000)},
            'historical_data': [{'timestamp': time.time() + i, 'value': i} for i in range(100)]
        })
        
        state_data = state_manager.serialize_cost_circuit_breaker_state(cost_based_circuit_breaker)
        
        # Save uncompressed
        uncompressed_path = state_manager.save_state_to_file(state_data, 'large_state.json')
        uncompressed_size = uncompressed_path.stat().st_size
        
        # Save compressed
        compressed_path = state_manager.base_path / 'large_state.json.gz'
        with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        compressed_size = compressed_path.stat().st_size
        
        # Verify compression achieved
        compression_ratio = compressed_size / uncompressed_size
        assert compression_ratio < 0.5  # At least 50% compression
        
        # Verify compressed file can be restored
        with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
            restored_data = json.load(f)
        
        assert restored_data['name'] == state_data['name']
        assert restored_data['operation_stats']['detailed_metrics'] == state_data['operation_stats']['detailed_metrics']


# =============================================================================
# INTEGRATION AND STRESS TESTS
# =============================================================================

class TestStatePersistenceStress:
    """Stress tests for state persistence under high load conditions."""
    
    def test_high_frequency_state_updates(self, cost_based_circuit_breaker, state_manager):
        """Test state persistence under high frequency updates."""
        update_count = 100
        states_saved = []
        
        # Perform rapid state updates
        for i in range(update_count):
            cost_based_circuit_breaker._operation_stats['total_calls'] = i
            cost_based_circuit_breaker._operation_stats[f'custom_metric_{i}'] = i * 0.1
            
            state_data = state_manager.serialize_cost_circuit_breaker_state(cost_based_circuit_breaker)
            file_path = state_manager.save_state_to_file(state_data, f'rapid_update_{i:03d}.json')
            states_saved.append(file_path)
        
        # Verify all states were saved correctly
        assert len(states_saved) == update_count
        
        # Spot check a few saved states
        for i in [0, update_count//2, update_count-1]:
            loaded_state = state_manager.load_state_from_file(f'rapid_update_{i:03d}.json')
            assert loaded_state['operation_stats']['total_calls'] == i
            assert loaded_state['operation_stats'][f'custom_metric_{i}'] == i * 0.1
    
    def test_concurrent_persistence_operations(self, mock_budget_manager, mock_cost_estimator, state_manager):
        """Test concurrent persistence operations from multiple threads."""
        import threading
        import time
        
        results = {'successes': 0, 'errors': 0}
        results_lock = threading.Lock()
        
        def persistence_worker(worker_id):
            """Worker function for concurrent persistence testing."""
            try:
                # Create circuit breaker instance
                cb = CostBasedCircuitBreaker(
                    name=f'worker_{worker_id}',
                    budget_manager=mock_budget_manager,
                    cost_estimator=mock_cost_estimator,
                    threshold_rules=[CostThresholdRule(f'rule_{worker_id}', CostThresholdType.OPERATION_COST, 1.0, 'block')]
                )
                
                # Perform multiple persistence operations
                for i in range(10):
                    cb._operation_stats['total_calls'] = worker_id * 100 + i
                    state_data = state_manager.serialize_cost_circuit_breaker_state(cb)
                    state_manager.save_state_to_file(state_data, f'worker_{worker_id}_{i:02d}.json')
                    
                    # Brief pause to simulate realistic timing
                    time.sleep(0.001)
                
                with results_lock:
                    results['successes'] += 1
                    
            except Exception as e:
                with results_lock:
                    results['errors'] += 1
        
        # Launch multiple concurrent workers
        num_workers = 5
        threads = []
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=persistence_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify all operations succeeded
        assert results['successes'] == num_workers
        assert results['errors'] == 0
        
        # Verify all files were created correctly
        for worker_id in range(num_workers):
            for i in range(10):
                filename = f'worker_{worker_id}_{i:02d}.json'
                assert (state_manager.base_path / filename).exists()
                
                # Spot check content
                loaded_state = state_manager.load_state_from_file(filename)
                expected_calls = worker_id * 100 + i
                assert loaded_state['operation_stats']['total_calls'] == expected_calls
    
    def test_large_state_persistence(self, mock_budget_manager, mock_cost_estimator, state_manager):
        """Test persistence of large circuit breaker states."""
        # Create circuit breaker with large state
        large_rules = []
        for i in range(50):  # Many rules
            rule = CostThresholdRule(
                rule_id=f'large_rule_{i}',
                threshold_type=CostThresholdType.OPERATION_COST,
                threshold_value=float(i + 1),  # Ensure positive values (1-50)
                action='throttle' if i % 2 == 0 else 'block'
            )
            large_rules.append(rule)
        
        cb = CostBasedCircuitBreaker(
            name='large_state_test',
            budget_manager=mock_budget_manager,
            cost_estimator=mock_cost_estimator,
            threshold_rules=large_rules
        )
        
        # Add large amounts of operational data
        cb._operation_stats.update({
            'detailed_timing_data': {f'operation_{i}': [j * 0.001 for j in range(100)] for i in range(100)},
            'per_rule_statistics': {f'rule_{i}': {'triggers': i * 10, 'savings': i * 2.5} for i in range(50)},
            'historical_metrics': [{'timestamp': time.time() + i, 'metric': f'value_{i}'} for i in range(500)]
        })
        
        cb._rule_cooldowns = {f'large_rule_{i}': time.time() + i for i in range(50)}
        
        # Serialize and save large state
        start_time = time.time()
        state_data = state_manager.serialize_cost_circuit_breaker_state(cb)
        serialization_time = time.time() - start_time
        
        start_time = time.time()
        file_path = state_manager.save_state_to_file(state_data, 'large_state.json')
        save_time = time.time() - start_time
        
        # Load and deserialize large state
        start_time = time.time()
        loaded_state = state_manager.load_state_from_file('large_state.json')
        load_time = time.time() - start_time
        
        start_time = time.time()
        restored_cb = state_manager.deserialize_cost_circuit_breaker_state(
            loaded_state, mock_budget_manager, mock_cost_estimator
        )
        deserialization_time = time.time() - start_time
        
        # Verify large state was preserved correctly
        assert len(restored_cb.threshold_rules) == 50
        assert len(restored_cb._rule_cooldowns) == 50
        assert len(restored_cb._operation_stats['detailed_timing_data']) == 100
        assert len(restored_cb._operation_stats['per_rule_statistics']) == 50
        assert len(restored_cb._operation_stats['historical_metrics']) == 500
        
        # Verify performance is reasonable (should complete within reasonable time)
        total_time = serialization_time + save_time + load_time + deserialization_time
        assert total_time < 5.0  # Should complete within 5 seconds
        
        # Log performance metrics for analysis
        print(f"Large state persistence performance:")
        print(f"  Serialization: {serialization_time:.3f}s")
        print(f"  Save: {save_time:.3f}s")
        print(f"  Load: {load_time:.3f}s")
        print(f"  Deserialization: {deserialization_time:.3f}s")
        print(f"  Total: {total_time:.3f}s")
        print(f"  File size: {file_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])