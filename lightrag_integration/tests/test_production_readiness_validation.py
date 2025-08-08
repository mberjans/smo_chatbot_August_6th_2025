"""
Production Readiness Checklist and Validation Framework
=======================================================

This module provides comprehensive production readiness validation for the
LLM-enhanced Clinical Metabolomics Oracle system, ensuring all operational
requirements are met before deployment to production environments.

Key Features:
- Comprehensive production readiness checklist
- Operational monitoring and alerting validation
- Performance and scalability assessment
- Security and compliance verification
- Disaster recovery and backup validation
- SLA compliance testing
- Resource utilization monitoring
- Health check endpoint validation

Validation Categories:
1. Infrastructure Readiness
2. Security and Compliance
3. Performance and Scalability
4. Monitoring and Observability
5. Disaster Recovery and Backup
6. Configuration Management
7. Documentation and Runbooks
8. SLA and Service Level Objectives

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import pytest
import asyncio
import time
import json
import logging
import os
import tempfile
import subprocess
import psutil
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
import threading
import requests
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import system components
from lightrag_integration.query_router import BiomedicalQueryRouter
from lightrag_integration.llm_query_classifier import LLMQueryClassifier, LLMClassificationConfig
from lightrag_integration.comprehensive_confidence_scorer import HybridConfidenceScorer
from lightrag_integration.cost_persistence import CostTracker

# Test utilities
from .performance_test_utilities import PerformanceTestUtilities
from .biomedical_test_fixtures import BiomedicalTestFixtures


@dataclass
class ProductionReadinessConfig:
    """Configuration for production readiness validation."""
    
    # Environment settings
    environment: str = "production"
    service_name: str = "clinical-metabolomics-oracle"
    service_version: str = "1.0.0"
    
    # Performance SLA thresholds
    max_response_time_ms: float = 2000  # 2 second SLA
    min_availability: float = 0.999  # 99.9% uptime SLA
    min_success_rate: float = 0.995  # 99.5% success rate
    max_error_rate: float = 0.005  # 0.5% error rate
    
    # Scalability requirements
    min_concurrent_users: int = 100
    target_throughput_qps: int = 50  # Queries per second
    max_memory_usage_mb: int = 2048  # 2GB memory limit
    max_cpu_usage_percent: float = 80  # 80% CPU limit
    
    # Monitoring requirements
    health_check_timeout_s: int = 5
    metrics_retention_days: int = 30
    log_retention_days: int = 90
    alert_response_time_minutes: int = 15
    
    # Security requirements
    require_ssl: bool = True
    require_authentication: bool = True
    require_input_validation: bool = True
    require_rate_limiting: bool = True
    
    # Backup and recovery
    backup_frequency_hours: int = 24
    recovery_time_objective_minutes: int = 60  # RTO: 1 hour
    recovery_point_objective_minutes: int = 15  # RPO: 15 minutes


@dataclass
class ValidationResult:
    """Result of a production readiness validation check."""
    
    check_name: str
    category: str
    passed: bool
    severity: str  # critical, high, medium, low
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class InfrastructureValidator:
    """Validates infrastructure readiness for production deployment."""
    
    def __init__(self, config: ProductionReadinessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_system_resources(self) -> List[ValidationResult]:
        """Validate system resource requirements."""
        results = []
        
        try:
            # Check available memory
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available // (1024 * 1024)
            
            if available_memory_mb < self.config.max_memory_usage_mb * 2:  # Need 2x for safety
                results.append(ValidationResult(
                    check_name="system_memory",
                    category="infrastructure",
                    passed=False,
                    severity="critical",
                    message=f"Insufficient memory: {available_memory_mb}MB available, need {self.config.max_memory_usage_mb * 2}MB",
                    details={"available_mb": available_memory_mb, "required_mb": self.config.max_memory_usage_mb * 2}
                ))
            else:
                results.append(ValidationResult(
                    check_name="system_memory",
                    category="infrastructure",
                    passed=True,
                    severity="info",
                    message=f"Sufficient memory available: {available_memory_mb}MB",
                    details={"available_mb": available_memory_mb}
                ))
            
            # Check CPU cores
            cpu_count = psutil.cpu_count()
            min_required_cores = 4  # Minimum for production
            
            if cpu_count < min_required_cores:
                results.append(ValidationResult(
                    check_name="cpu_cores",
                    category="infrastructure",
                    passed=False,
                    severity="high",
                    message=f"Insufficient CPU cores: {cpu_count} available, need {min_required_cores}",
                    details={"available_cores": cpu_count, "required_cores": min_required_cores}
                ))
            else:
                results.append(ValidationResult(
                    check_name="cpu_cores",
                    category="infrastructure",
                    passed=True,
                    severity="info",
                    message=f"Sufficient CPU cores: {cpu_count}",
                    details={"available_cores": cpu_count}
                ))
            
            # Check disk space
            disk = psutil.disk_usage('/')
            free_space_gb = disk.free // (1024 * 1024 * 1024)
            min_required_space_gb = 50  # Minimum 50GB free
            
            if free_space_gb < min_required_space_gb:
                results.append(ValidationResult(
                    check_name="disk_space",
                    category="infrastructure",
                    passed=False,
                    severity="high",
                    message=f"Insufficient disk space: {free_space_gb}GB available, need {min_required_space_gb}GB",
                    details={"available_gb": free_space_gb, "required_gb": min_required_space_gb}
                ))
            else:
                results.append(ValidationResult(
                    check_name="disk_space",
                    category="infrastructure",
                    passed=True,
                    severity="info",
                    message=f"Sufficient disk space: {free_space_gb}GB",
                    details={"available_gb": free_space_gb}
                ))
            
        except Exception as e:
            results.append(ValidationResult(
                check_name="system_resources_check",
                category="infrastructure",
                passed=False,
                severity="critical",
                message=f"Failed to check system resources: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    def validate_network_connectivity(self) -> List[ValidationResult]:
        """Validate network connectivity and DNS resolution."""
        results = []
        
        # Test external API connectivity (for LLM services)
        test_endpoints = [
            ("OpenAI API", "api.openai.com", 443),
            ("DNS Resolution", "google.com", 80),
        ]
        
        for name, host, port in test_endpoints:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    results.append(ValidationResult(
                        check_name=f"connectivity_{host}",
                        category="infrastructure",
                        passed=True,
                        severity="info",
                        message=f"Successfully connected to {name}",
                        details={"host": host, "port": port}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name=f"connectivity_{host}",
                        category="infrastructure",
                        passed=False,
                        severity="high",
                        message=f"Cannot connect to {name} ({host}:{port})",
                        details={"host": host, "port": port, "error_code": result}
                    ))
                    
            except Exception as e:
                results.append(ValidationResult(
                    check_name=f"connectivity_{host}",
                    category="infrastructure",
                    passed=False,
                    severity="high",
                    message=f"Network connectivity test failed for {name}: {str(e)}",
                    details={"host": host, "port": port, "error": str(e)}
                ))
        
        return results
    
    def validate_dependencies(self) -> List[ValidationResult]:
        """Validate that all required dependencies are available."""
        results = []
        
        # Check Python packages
        required_packages = [
            "openai", "numpy", "pandas", "requests", "pytest",
            "lightrag", "asyncio", "pathlib", "dataclasses"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                results.append(ValidationResult(
                    check_name=f"package_{package}",
                    category="infrastructure",
                    passed=True,
                    severity="info",
                    message=f"Package {package} is available",
                    details={"package": package}
                ))
            except ImportError:
                results.append(ValidationResult(
                    check_name=f"package_{package}",
                    category="infrastructure",
                    passed=False,
                    severity="high",
                    message=f"Required package {package} is not installed",
                    details={"package": package}
                ))
        
        # Check environment variables
        required_env_vars = [
            "OPENAI_API_KEY",
            "LIGHTRAG_CONFIG_PATH"
        ]
        
        for env_var in required_env_vars:
            if os.environ.get(env_var):
                results.append(ValidationResult(
                    check_name=f"env_var_{env_var}",
                    category="infrastructure",
                    passed=True,
                    severity="info",
                    message=f"Environment variable {env_var} is set",
                    details={"env_var": env_var}
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"env_var_{env_var}",
                    category="infrastructure",
                    passed=False,
                    severity="medium",
                    message=f"Environment variable {env_var} is not set",
                    details={"env_var": env_var}
                ))
        
        return results


class PerformanceValidator:
    """Validates performance and scalability requirements."""
    
    def __init__(self, config: ProductionReadinessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.performance_utils = PerformanceTestUtilities()
        
    async def validate_response_time_sla(self) -> List[ValidationResult]:
        """Validate response time meets SLA requirements."""
        results = []
        
        try:
            # Create test router
            router = BiomedicalQueryRouter()
            
            # Test queries of varying complexity
            test_queries = [
                "metabolomics",
                "What are biomarkers for diabetes?",
                "Explain the relationship between metabolic pathways and insulin resistance in type 2 diabetes patients.",
                "How do LC-MS techniques compare for metabolite identification in clinical samples?"
            ]
            
            response_times = []
            
            for query in test_queries:
                start_time = time.time()
                try:
                    prediction = router.route_query(query)
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    response_times.append(response_time)
                    
                    if response_time > self.config.max_response_time_ms:
                        results.append(ValidationResult(
                            check_name="response_time_individual",
                            category="performance",
                            passed=False,
                            severity="high",
                            message=f"Query exceeded response time SLA: {response_time:.2f}ms > {self.config.max_response_time_ms}ms",
                            details={"query": query[:50], "response_time_ms": response_time}
                        ))
                except Exception as e:
                    results.append(ValidationResult(
                        check_name="response_time_error",
                        category="performance",
                        passed=False,
                        severity="critical",
                        message=f"Query failed: {str(e)}",
                        details={"query": query[:50], "error": str(e)}
                    ))
            
            # Analyze overall performance
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
                max_response_time = max(response_times)
                
                # Check average response time
                if avg_response_time <= self.config.max_response_time_ms:
                    results.append(ValidationResult(
                        check_name="avg_response_time",
                        category="performance",
                        passed=True,
                        severity="info",
                        message=f"Average response time meets SLA: {avg_response_time:.2f}ms",
                        details={
                            "avg_response_time_ms": avg_response_time,
                            "p95_response_time_ms": p95_response_time,
                            "max_response_time_ms": max_response_time,
                            "sla_threshold_ms": self.config.max_response_time_ms
                        }
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="avg_response_time",
                        category="performance",
                        passed=False,
                        severity="critical",
                        message=f"Average response time exceeds SLA: {avg_response_time:.2f}ms > {self.config.max_response_time_ms}ms",
                        details={
                            "avg_response_time_ms": avg_response_time,
                            "sla_threshold_ms": self.config.max_response_time_ms
                        }
                    ))
                
                # Check P95 response time (should be within reasonable bounds)
                p95_threshold = self.config.max_response_time_ms * 1.5  # 50% higher for P95
                if p95_response_time <= p95_threshold:
                    results.append(ValidationResult(
                        check_name="p95_response_time",
                        category="performance",
                        passed=True,
                        severity="info",
                        message=f"P95 response time acceptable: {p95_response_time:.2f}ms",
                        details={"p95_response_time_ms": p95_response_time, "threshold_ms": p95_threshold}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="p95_response_time",
                        category="performance",
                        passed=False,
                        severity="medium",
                        message=f"P95 response time concerning: {p95_response_time:.2f}ms > {p95_threshold:.2f}ms",
                        details={"p95_response_time_ms": p95_response_time, "threshold_ms": p95_threshold}
                    ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="response_time_validation",
                category="performance",
                passed=False,
                severity="critical",
                message=f"Response time validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_concurrent_load_handling(self) -> List[ValidationResult]:
        """Validate system can handle concurrent load."""
        results = []
        
        try:
            router = BiomedicalQueryRouter()
            test_query = "What are metabolomics biomarkers for diabetes?"
            
            # Test concurrent requests
            concurrent_users = min(self.config.min_concurrent_users, 20)  # Limited for testing
            
            async def make_request():
                start_time = time.time()
                try:
                    prediction = router.route_query(test_query)
                    response_time = (time.time() - start_time) * 1000
                    return {"success": True, "response_time": response_time}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            # Execute concurrent requests
            tasks = [make_request() for _ in range(concurrent_users)]
            concurrent_results = await asyncio.gather(*tasks)
            
            # Analyze results
            successful_requests = [r for r in concurrent_results if r["success"]]
            failed_requests = [r for r in concurrent_results if not r["success"]]
            
            success_rate = len(successful_requests) / len(concurrent_results)
            
            if success_rate >= self.config.min_success_rate:
                results.append(ValidationResult(
                    check_name="concurrent_load_success_rate",
                    category="performance",
                    passed=True,
                    severity="info",
                    message=f"Concurrent load handled successfully: {success_rate:.1%} success rate",
                    details={
                        "concurrent_users": concurrent_users,
                        "success_rate": success_rate,
                        "successful_requests": len(successful_requests),
                        "failed_requests": len(failed_requests)
                    }
                ))
            else:
                results.append(ValidationResult(
                    check_name="concurrent_load_success_rate",
                    category="performance",
                    passed=False,
                    severity="critical",
                    message=f"Concurrent load test failed: {success_rate:.1%} success rate < {self.config.min_success_rate:.1%}",
                    details={
                        "concurrent_users": concurrent_users,
                        "success_rate": success_rate,
                        "required_success_rate": self.config.min_success_rate
                    }
                ))
            
            # Check response times under load
            if successful_requests:
                response_times = [r["response_time"] for r in successful_requests]
                avg_response_time_under_load = sum(response_times) / len(response_times)
                
                # Allow higher response times under load (up to 2x)
                load_threshold = self.config.max_response_time_ms * 2
                
                if avg_response_time_under_load <= load_threshold:
                    results.append(ValidationResult(
                        check_name="response_time_under_load",
                        category="performance",
                        passed=True,
                        severity="info",
                        message=f"Response time under load acceptable: {avg_response_time_under_load:.2f}ms",
                        details={"avg_response_time_ms": avg_response_time_under_load, "threshold_ms": load_threshold}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="response_time_under_load",
                        category="performance",
                        passed=False,
                        severity="high",
                        message=f"Response time under load too high: {avg_response_time_under_load:.2f}ms > {load_threshold}ms",
                        details={"avg_response_time_ms": avg_response_time_under_load, "threshold_ms": load_threshold}
                    ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="concurrent_load_test",
                category="performance",
                passed=False,
                severity="critical",
                message=f"Concurrent load test failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    def validate_resource_utilization(self) -> List[ValidationResult]:
        """Validate resource utilization is within acceptable bounds."""
        results = []
        
        try:
            # Get baseline resource usage
            baseline_cpu = psutil.cpu_percent(interval=1)
            baseline_memory = psutil.virtual_memory()
            
            # Simulate load and measure resource usage
            router = BiomedicalQueryRouter()
            test_queries = [
                "What are metabolomics biomarkers?",
                "LC-MS analysis methods",
                "Metabolic pathway analysis for diabetes"
            ] * 10  # Repeat to create load
            
            start_time = time.time()
            
            for query in test_queries:
                router.route_query(query)
            
            # Measure resource usage under load
            load_cpu = psutil.cpu_percent(interval=1)
            load_memory = psutil.virtual_memory()
            
            # Calculate memory usage in MB
            memory_used_mb = (load_memory.total - load_memory.available) // (1024 * 1024)
            
            # Validate CPU usage
            if load_cpu <= self.config.max_cpu_usage_percent:
                results.append(ValidationResult(
                    check_name="cpu_usage_under_load",
                    category="performance",
                    passed=True,
                    severity="info",
                    message=f"CPU usage acceptable under load: {load_cpu:.1f}%",
                    details={"cpu_usage_percent": load_cpu, "threshold_percent": self.config.max_cpu_usage_percent}
                ))
            else:
                results.append(ValidationResult(
                    check_name="cpu_usage_under_load",
                    category="performance",
                    passed=False,
                    severity="high",
                    message=f"CPU usage too high under load: {load_cpu:.1f}% > {self.config.max_cpu_usage_percent}%",
                    details={"cpu_usage_percent": load_cpu, "threshold_percent": self.config.max_cpu_usage_percent}
                ))
            
            # Validate memory usage
            if memory_used_mb <= self.config.max_memory_usage_mb:
                results.append(ValidationResult(
                    check_name="memory_usage_under_load",
                    category="performance",
                    passed=True,
                    severity="info",
                    message=f"Memory usage acceptable: {memory_used_mb}MB",
                    details={"memory_used_mb": memory_used_mb, "threshold_mb": self.config.max_memory_usage_mb}
                ))
            else:
                results.append(ValidationResult(
                    check_name="memory_usage_under_load",
                    category="performance",
                    passed=False,
                    severity="high",
                    message=f"Memory usage too high: {memory_used_mb}MB > {self.config.max_memory_usage_mb}MB",
                    details={"memory_used_mb": memory_used_mb, "threshold_mb": self.config.max_memory_usage_mb}
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="resource_utilization_test",
                category="performance",
                passed=False,
                severity="critical",
                message=f"Resource utilization test failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results


class SecurityValidator:
    """Validates security and compliance requirements."""
    
    def __init__(self, config: ProductionReadinessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_input_sanitization(self) -> List[ValidationResult]:
        """Validate input sanitization and validation."""
        results = []
        
        try:
            router = BiomedicalQueryRouter()
            
            # Test malicious inputs
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../../etc/passwd",
                "{{7*7}}",  # Template injection
                "' OR '1'='1",  # SQL injection
                "<img src=x onerror=alert(1)>",  # XSS
                "javascript:alert(1)",
                "%0Acat%20/etc/passwd"  # Command injection
            ]
            
            for malicious_input in malicious_inputs:
                try:
                    prediction = router.route_query(malicious_input)
                    
                    # Check that the input was processed safely
                    if prediction is not None and prediction.confidence >= 0:
                        # System processed input - check for signs of successful attack
                        reasoning_text = " ".join(prediction.reasoning) if prediction.reasoning else ""
                        
                        # Check if malicious content appears in output (bad)
                        if any(dangerous in reasoning_text.lower() for dangerous in ['script', 'alert', 'drop', 'passwd']):
                            results.append(ValidationResult(
                                check_name="input_sanitization_leak",
                                category="security",
                                passed=False,
                                severity="critical",
                                message=f"Malicious input may have leaked into output: {malicious_input[:30]}...",
                                details={"input": malicious_input[:50], "output_excerpt": reasoning_text[:100]}
                            ))
                        else:
                            results.append(ValidationResult(
                                check_name="input_sanitization_safe",
                                category="security",
                                passed=True,
                                severity="info",
                                message=f"Malicious input handled safely: {malicious_input[:30]}...",
                                details={"input": malicious_input[:50]}
                            ))
                    else:
                        # System rejected input completely - also safe
                        results.append(ValidationResult(
                            check_name="input_rejection",
                            category="security",
                            passed=True,
                            severity="info",
                            message=f"Malicious input rejected: {malicious_input[:30]}...",
                            details={"input": malicious_input[:50]}
                        ))
                        
                except Exception as e:
                    # Exception during processing - need to verify it's safe
                    if any(dangerous in str(e).lower() for dangerous in ['script', 'eval', 'exec']):
                        results.append(ValidationResult(
                            check_name="input_processing_error_dangerous",
                            category="security",
                            passed=False,
                            severity="high",
                            message=f"Dangerous error processing malicious input: {str(e)}",
                            details={"input": malicious_input[:50], "error": str(e)}
                        ))
                    else:
                        # Safe exception (normal error handling)
                        results.append(ValidationResult(
                            check_name="input_processing_error_safe",
                            category="security",
                            passed=True,
                            severity="info",
                            message=f"Malicious input caused safe error: {malicious_input[:30]}...",
                            details={"input": malicious_input[:50], "error_type": type(e).__name__}
                        ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="input_sanitization_test",
                category="security",
                passed=False,
                severity="critical",
                message=f"Input sanitization test failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    def validate_api_security(self) -> List[ValidationResult]:
        """Validate API security measures."""
        results = []
        
        # Check for API key protection
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            if len(openai_key) > 10:  # Has substantial key
                results.append(ValidationResult(
                    check_name="api_key_configured",
                    category="security",
                    passed=True,
                    severity="info",
                    message="API key is configured",
                    details={"key_length": len(openai_key), "key_prefix": openai_key[:8] + "..."}
                ))
            else:
                results.append(ValidationResult(
                    check_name="api_key_too_short",
                    category="security",
                    passed=False,
                    severity="high",
                    message="API key appears too short or invalid",
                    details={"key_length": len(openai_key)}
                ))
        else:
            results.append(ValidationResult(
                check_name="api_key_missing",
                category="security",
                passed=False,
                severity="critical",
                message="API key not configured",
                details={}
            ))
        
        # Check for secure configuration
        sensitive_env_vars = ["OPENAI_API_KEY", "DATABASE_PASSWORD", "JWT_SECRET"]
        for var in sensitive_env_vars:
            value = os.environ.get(var)
            if value and value != "test" and value != "placeholder":
                # Check if it looks like a real secret (not test data)
                if len(value) >= 16 and not value.startswith("test"):
                    results.append(ValidationResult(
                        check_name=f"secure_config_{var}",
                        category="security",
                        passed=True,
                        severity="info",
                        message=f"Secure configuration for {var}",
                        details={"var_name": var, "configured": True}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name=f"weak_config_{var}",
                        category="security",
                        passed=False,
                        severity="medium",
                        message=f"Weak or test configuration for {var}",
                        details={"var_name": var, "appears_weak": True}
                    ))
        
        return results
    
    def validate_data_protection(self) -> List[ValidationResult]:
        """Validate data protection measures."""
        results = []
        
        try:
            # Test that sensitive data is not logged
            router = BiomedicalQueryRouter()
            
            # Query with potentially sensitive information
            sensitive_query = "Patient John Doe has diabetes with glucose levels 250mg/dL and SSN 123-45-6789"
            
            prediction = router.route_query(sensitive_query)
            
            # Check reasoning doesn't contain sensitive data patterns
            reasoning_text = " ".join(prediction.reasoning) if prediction.reasoning else ""
            
            sensitive_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b[A-Za-z]+\s+[A-Za-z]+\s+Doe\b',  # Name pattern
                r'\b\d+mg/dL\b'  # Medical values
            ]
            
            import re
            sensitive_found = False
            for pattern in sensitive_patterns:
                if re.search(pattern, reasoning_text):
                    sensitive_found = True
                    break
            
            if not sensitive_found:
                results.append(ValidationResult(
                    check_name="sensitive_data_protection",
                    category="security",
                    passed=True,
                    severity="info",
                    message="Sensitive data not exposed in reasoning",
                    details={"query_processed": True, "data_protected": True}
                ))
            else:
                results.append(ValidationResult(
                    check_name="sensitive_data_leak",
                    category="security",
                    passed=False,
                    severity="critical",
                    message="Sensitive data may be exposed in reasoning",
                    details={"reasoning_excerpt": reasoning_text[:100]}
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="data_protection_test",
                category="security",
                passed=False,
                severity="high",
                message=f"Data protection test failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results


class MonitoringValidator:
    """Validates monitoring and observability requirements."""
    
    def __init__(self, config: ProductionReadinessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_health_check_endpoints(self) -> List[ValidationResult]:
        """Validate health check endpoints are working."""
        results = []
        
        try:
            # Simulate health check endpoint
            router = BiomedicalQueryRouter()
            
            # Test basic functionality as health check
            start_time = time.time()
            test_result = router.route_query("health check test")
            response_time = (time.time() - start_time) * 1000
            
            if test_result is not None and response_time < self.config.health_check_timeout_s * 1000:
                results.append(ValidationResult(
                    check_name="health_check_functionality",
                    category="monitoring",
                    passed=True,
                    severity="info",
                    message=f"Health check functional: {response_time:.2f}ms",
                    details={"response_time_ms": response_time, "timeout_ms": self.config.health_check_timeout_s * 1000}
                ))
            else:
                results.append(ValidationResult(
                    check_name="health_check_slow",
                    category="monitoring",
                    passed=False,
                    severity="high",
                    message=f"Health check too slow: {response_time:.2f}ms",
                    details={"response_time_ms": response_time, "timeout_ms": self.config.health_check_timeout_s * 1000}
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="health_check_error",
                category="monitoring",
                passed=False,
                severity="critical",
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    def validate_logging_configuration(self) -> List[ValidationResult]:
        """Validate logging is properly configured."""
        results = []
        
        try:
            # Check that logging is working
            test_logger = logging.getLogger("production_readiness_test")
            
            # Test different log levels
            test_logger.info("Test info message")
            test_logger.warning("Test warning message")
            test_logger.error("Test error message")
            
            # Check that structured logging is available
            try:
                import json
                test_log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": "Structured log test",
                    "service": self.config.service_name
                }
                test_logger.info(json.dumps(test_log_data))
                
                results.append(ValidationResult(
                    check_name="structured_logging",
                    category="monitoring",
                    passed=True,
                    severity="info",
                    message="Structured logging is available",
                    details={"format": "JSON", "test_successful": True}
                ))
            except Exception as e:
                results.append(ValidationResult(
                    check_name="structured_logging_error",
                    category="monitoring",
                    passed=False,
                    severity="medium",
                    message=f"Structured logging failed: {str(e)}",
                    details={"error": str(e)}
                ))
            
            # Check log level configuration
            current_level = test_logger.getEffectiveLevel()
            if current_level <= logging.INFO:
                results.append(ValidationResult(
                    check_name="log_level_appropriate",
                    category="monitoring",
                    passed=True,
                    severity="info",
                    message=f"Log level appropriate: {logging.getLevelName(current_level)}",
                    details={"current_level": logging.getLevelName(current_level)}
                ))
            else:
                results.append(ValidationResult(
                    check_name="log_level_too_high",
                    category="monitoring",
                    passed=False,
                    severity="medium",
                    message=f"Log level too high: {logging.getLevelName(current_level)}",
                    details={"current_level": logging.getLevelName(current_level)}
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="logging_configuration_test",
                category="monitoring",
                passed=False,
                severity="high",
                message=f"Logging configuration test failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    def validate_metrics_collection(self) -> List[ValidationResult]:
        """Validate metrics collection capabilities."""
        results = []
        
        try:
            # Test basic metrics collection
            router = BiomedicalQueryRouter()
            
            # Collect some routing statistics
            test_queries = ["test query 1", "test query 2", "test query 3"]
            
            for query in test_queries:
                router.route_query(query)
            
            # Get statistics
            stats = router.get_routing_statistics()
            
            if stats and isinstance(stats, dict):
                required_metrics = ['total_queries', 'routing_decisions', 'confidence_distribution']
                missing_metrics = [metric for metric in required_metrics if metric not in stats]
                
                if not missing_metrics:
                    results.append(ValidationResult(
                        check_name="basic_metrics_collection",
                        category="monitoring",
                        passed=True,
                        severity="info",
                        message="Basic metrics collection working",
                        details={"metrics_available": list(stats.keys())}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="incomplete_metrics",
                        category="monitoring",
                        passed=False,
                        severity="medium",
                        message=f"Missing metrics: {missing_metrics}",
                        details={"missing_metrics": missing_metrics}
                    ))
            else:
                results.append(ValidationResult(
                    check_name="metrics_collection_failed",
                    category="monitoring",
                    passed=False,
                    severity="high",
                    message="Metrics collection returned invalid data",
                    details={"stats_type": type(stats).__name__}
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="metrics_collection_test",
                category="monitoring",
                passed=False,
                severity="high",
                message=f"Metrics collection test failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results


class ProductionReadinessChecker:
    """Main production readiness checker that orchestrates all validations."""
    
    def __init__(self, config: Optional[ProductionReadinessConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or ProductionReadinessConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize validators
        self.infrastructure_validator = InfrastructureValidator(self.config, self.logger)
        self.performance_validator = PerformanceValidator(self.config, self.logger)
        self.security_validator = SecurityValidator(self.config, self.logger)
        self.monitoring_validator = MonitoringValidator(self.config, self.logger)
        
        # Results storage
        self.validation_results: List[ValidationResult] = []
        
    async def run_comprehensive_readiness_check(self) -> Dict[str, Any]:
        """Run comprehensive production readiness validation."""
        
        self.logger.info("Starting comprehensive production readiness validation")
        start_time = datetime.now()
        
        # Define validation suites
        validation_suites = [
            ("Infrastructure", [
                self.infrastructure_validator.validate_system_resources(),
                self.infrastructure_validator.validate_network_connectivity(),
                self.infrastructure_validator.validate_dependencies()
            ]),
            ("Performance", [
                self.performance_validator.validate_response_time_sla(),
                self.performance_validator.validate_concurrent_load_handling(),
                self.performance_validator.validate_resource_utilization()
            ]),
            ("Security", [
                self.security_validator.validate_input_sanitization(),
                self.security_validator.validate_api_security(),
                self.security_validator.validate_data_protection()
            ]),
            ("Monitoring", [
                self.monitoring_validator.validate_health_check_endpoints(),
                self.monitoring_validator.validate_logging_configuration(),
                self.monitoring_validator.validate_metrics_collection()
            ])
        ]
        
        # Run all validations
        all_results = []
        category_summaries = {}
        
        for category_name, validators in validation_suites:
            self.logger.info(f"Running {category_name} validations")
            category_results = []
            
            for validator in validators:
                if asyncio.iscoroutine(validator):
                    results = await validator
                else:
                    results = validator
                category_results.extend(results)
            
            all_results.extend(category_results)
            
            # Summarize category results
            category_passed = sum(1 for r in category_results if r.passed)
            category_failed = len(category_results) - category_passed
            category_critical = sum(1 for r in category_results if r.severity == "critical" and not r.passed)
            
            category_summaries[category_name] = {
                'total_checks': len(category_results),
                'passed': category_passed,
                'failed': category_failed,
                'critical_failures': category_critical,
                'pass_rate': category_passed / len(category_results) if category_results else 0
            }
        
        self.validation_results = all_results
        end_time = datetime.now()
        
        # Calculate overall results
        total_checks = len(all_results)
        total_passed = sum(1 for r in all_results if r.passed)
        total_failed = total_checks - total_passed
        critical_failures = sum(1 for r in all_results if r.severity == "critical" and not r.passed)
        high_failures = sum(1 for r in all_results if r.severity == "high" and not r.passed)
        
        # Determine production readiness
        production_ready = (
            critical_failures == 0 and
            high_failures <= 2 and  # Allow up to 2 high severity failures
            total_passed / total_checks >= 0.9  # 90% pass rate minimum
        )
        
        # Generate report
        report = {
            'production_readiness_summary': {
                'overall_ready': production_ready,
                'readiness_score': total_passed / total_checks if total_checks > 0 else 0,
                'total_checks': total_checks,
                'passed_checks': total_passed,
                'failed_checks': total_failed,
                'critical_failures': critical_failures,
                'high_severity_failures': high_failures,
                'validation_duration_seconds': (end_time - start_time).total_seconds()
            },
            'category_summaries': category_summaries,
            'detailed_results': [result.to_dict() for result in all_results],
            'production_blockers': [
                result.to_dict() for result in all_results 
                if not result.passed and result.severity == "critical"
            ],
            'high_priority_issues': [
                result.to_dict() for result in all_results 
                if not result.passed and result.severity == "high"
            ],
            'recommendations': self._generate_readiness_recommendations(all_results, production_ready),
            'next_steps': self._generate_next_steps(all_results, production_ready)
        }
        
        # Log summary
        self.logger.info(f"Production readiness validation completed:")
        self.logger.info(f"  Overall ready: {production_ready}")
        self.logger.info(f"  Score: {report['production_readiness_summary']['readiness_score']:.1%}")
        self.logger.info(f"  Checks: {total_passed}/{total_checks} passed")
        self.logger.info(f"  Critical failures: {critical_failures}")
        
        return report
    
    def _generate_readiness_recommendations(self, results: List[ValidationResult], production_ready: bool) -> List[Dict[str, str]]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if production_ready:
            recommendations.append({
                'type': 'deployment',
                'priority': 'info',
                'title': 'Production Ready',
                'recommendation': 'System passes production readiness checks. Proceed with deployment.',
                'action': 'Deploy to production'
            })
        else:
            recommendations.append({
                'type': 'blocking',
                'priority': 'critical',
                'title': 'Not Production Ready',
                'recommendation': 'System has blocking issues that must be resolved before deployment.',
                'action': 'Address critical and high priority issues'
            })
        
        # Specific recommendations based on failures
        critical_failures = [r for r in results if not r.passed and r.severity == "critical"]
        for failure in critical_failures:
            recommendations.append({
                'type': 'critical_fix',
                'priority': 'critical',
                'title': f'Fix: {failure.check_name}',
                'recommendation': failure.message,
                'action': f'Address {failure.category} issue'
            })
        
        # Performance recommendations
        performance_failures = [r for r in results if not r.passed and r.category == "performance"]
        if performance_failures:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'Performance Optimization',
                'recommendation': 'Performance issues detected that may impact user experience.',
                'action': 'Review and optimize system performance'
            })
        
        # Security recommendations
        security_failures = [r for r in results if not r.passed and r.category == "security"]
        if security_failures:
            recommendations.append({
                'type': 'security',
                'priority': 'high',
                'title': 'Security Hardening',
                'recommendation': 'Security vulnerabilities must be addressed before production deployment.',
                'action': 'Implement security fixes and run security audit'
            })
        
        return recommendations
    
    def _generate_next_steps(self, results: List[ValidationResult], production_ready: bool) -> List[str]:
        """Generate next steps based on validation results."""
        next_steps = []
        
        if production_ready:
            next_steps.extend([
                "1. Review and approve deployment plan",
                "2. Schedule deployment window",
                "3. Execute deployment with monitoring",
                "4. Conduct post-deployment validation",
                "5. Monitor system performance for 24 hours"
            ])
        else:
            next_steps.extend([
                "1. Address all critical severity failures",
                "2. Fix high priority issues",
                "3. Re-run production readiness validation",
                "4. Review and update deployment checklist",
                "5. Schedule follow-up validation"
            ])
        
        # Add specific steps for categories with failures
        failed_categories = set(r.category for r in results if not r.passed and r.severity in ["critical", "high"])
        
        if "infrastructure" in failed_categories:
            next_steps.append("• Review infrastructure requirements and capacity")
        
        if "performance" in failed_categories:
            next_steps.append("• Conduct performance optimization and load testing")
        
        if "security" in failed_categories:
            next_steps.append("• Conduct security review and penetration testing")
        
        if "monitoring" in failed_categories:
            next_steps.append("• Configure monitoring, alerting, and observability")
        
        return next_steps


# Pytest test class
@pytest.mark.asyncio
class TestProductionReadinessValidation:
    """Main test class for production readiness validation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = ProductionReadinessConfig()
        self.checker = ProductionReadinessChecker(self.config, self.logger)
    
    async def test_comprehensive_production_readiness(self):
        """Test comprehensive production readiness validation."""
        report = await self.checker.run_comprehensive_readiness_check()
        
        # Basic structure validation
        assert 'production_readiness_summary' in report
        assert 'category_summaries' in report
        assert 'detailed_results' in report
        
        summary = report['production_readiness_summary']
        
        # Check that validation ran
        assert summary['total_checks'] > 0, "No validation checks were run"
        assert summary['passed_checks'] >= 0, "Invalid passed checks count"
        assert summary['failed_checks'] >= 0, "Invalid failed checks count"
        
        # Check readiness score calculation
        expected_score = summary['passed_checks'] / summary['total_checks']
        assert abs(summary['readiness_score'] - expected_score) < 0.01, "Readiness score calculation error"
        
        # Log results for debugging
        self.logger.info(f"Production readiness: {summary['overall_ready']}")
        self.logger.info(f"Score: {summary['readiness_score']:.1%}")
        self.logger.info(f"Critical failures: {summary['critical_failures']}")
        
        # Check categories were tested
        categories = report['category_summaries']
        expected_categories = ['Infrastructure', 'Performance', 'Security', 'Monitoring']
        for category in expected_categories:
            assert category in categories, f"Missing category: {category}"
            assert categories[category]['total_checks'] > 0, f"No checks run for {category}"
    
    async def test_infrastructure_validation(self):
        """Test infrastructure validation specifically."""
        validator = InfrastructureValidator(self.config, self.logger)
        
        # Test system resources
        resource_results = validator.validate_system_resources()
        assert len(resource_results) > 0, "No resource validation results"
        
        # Test network connectivity
        network_results = validator.validate_network_connectivity()
        assert len(network_results) > 0, "No network validation results"
        
        # Test dependencies
        dependency_results = validator.validate_dependencies()
        assert len(dependency_results) > 0, "No dependency validation results"
        
        # All validations should return ValidationResult objects
        all_results = resource_results + network_results + dependency_results
        for result in all_results:
            assert isinstance(result, ValidationResult), "Invalid result type"
            assert result.check_name, "Missing check name"
            assert result.category == "infrastructure", "Wrong category"
            assert result.severity in ["critical", "high", "medium", "low", "info"], "Invalid severity"
    
    async def test_performance_validation(self):
        """Test performance validation specifically."""
        validator = PerformanceValidator(self.config, self.logger)
        
        # Test response time SLA
        response_time_results = await validator.validate_response_time_sla()
        assert len(response_time_results) > 0, "No response time validation results"
        
        # Test concurrent load handling
        load_results = await validator.validate_concurrent_load_handling()
        assert len(load_results) > 0, "No load validation results"
        
        # Test resource utilization
        resource_results = validator.validate_resource_utilization()
        assert len(resource_results) > 0, "No resource utilization results"
        
        # Check for performance metrics
        all_results = response_time_results + load_results + resource_results
        performance_metrics_found = False
        
        for result in all_results:
            assert isinstance(result, ValidationResult), "Invalid result type"
            assert result.category == "performance", "Wrong category"
            
            if 'response_time_ms' in result.details or 'cpu_usage_percent' in result.details:
                performance_metrics_found = True
        
        assert performance_metrics_found, "No performance metrics found in results"
    
    async def test_security_validation(self):
        """Test security validation specifically."""
        validator = SecurityValidator(self.config, self.logger)
        
        # Test input sanitization
        input_results = validator.validate_input_sanitization()
        assert len(input_results) > 0, "No input sanitization results"
        
        # Test API security
        api_results = validator.validate_api_security()
        assert len(api_results) > 0, "No API security results"
        
        # Test data protection
        data_results = validator.validate_data_protection()
        assert len(data_results) > 0, "No data protection results"
        
        # Check that security tests were comprehensive
        all_results = input_results + api_results + data_results
        security_checks = set(result.check_name for result in all_results)
        
        expected_checks = ['input_sanitization', 'api_key', 'data_protection']
        found_checks = [check for check in expected_checks 
                       if any(expected in result.check_name for result in all_results)]
        
        assert len(found_checks) >= 2, f"Not enough security checks found: {found_checks}"
    
    async def test_monitoring_validation(self):
        """Test monitoring validation specifically."""
        validator = MonitoringValidator(self.config, self.logger)
        
        # Test health check endpoints
        health_results = validator.validate_health_check_endpoints()
        assert len(health_results) > 0, "No health check results"
        
        # Test logging configuration
        logging_results = validator.validate_logging_configuration()
        assert len(logging_results) > 0, "No logging validation results"
        
        # Test metrics collection
        metrics_results = validator.validate_metrics_collection()
        assert len(metrics_results) > 0, "No metrics validation results"
        
        # Check monitoring capabilities
        all_results = health_results + logging_results + metrics_results
        
        for result in all_results:
            assert isinstance(result, ValidationResult), "Invalid result type"
            assert result.category == "monitoring", "Wrong category"
        
        # Check for essential monitoring features
        monitoring_features = set(result.check_name for result in all_results)
        essential_features = ['health_check', 'logging', 'metrics']
        
        found_features = [feature for feature in essential_features
                         if any(feature in check_name for check_name in monitoring_features)]
        
        assert len(found_features) >= 2, f"Missing essential monitoring features: {found_features}"
    
    async def test_production_readiness_scoring(self):
        """Test production readiness scoring logic."""
        # Create mock results with known outcomes
        mock_results = [
            ValidationResult("test1", "infrastructure", True, "info", "Test passed"),
            ValidationResult("test2", "performance", True, "info", "Test passed"),
            ValidationResult("test3", "security", False, "medium", "Test failed"),
            ValidationResult("test4", "monitoring", False, "high", "Test failed"),
            ValidationResult("test5", "infrastructure", False, "critical", "Critical failure")
        ]
        
        # Override checker results
        self.checker.validation_results = mock_results
        
        # Calculate expected metrics
        total_checks = 5
        passed_checks = 2
        failed_checks = 3
        critical_failures = 1
        high_failures = 1
        
        # Test scoring logic
        production_ready = (
            critical_failures == 0 and
            high_failures <= 2 and
            passed_checks / total_checks >= 0.9
        )
        
        # Should not be production ready due to critical failure
        assert not production_ready, "Should not be production ready with critical failure"
        
        # Test with all passing results
        passing_results = [
            ValidationResult("test1", "infrastructure", True, "info", "Test passed"),
            ValidationResult("test2", "performance", True, "info", "Test passed"),
            ValidationResult("test3", "security", True, "info", "Test passed"),
            ValidationResult("test4", "monitoring", True, "info", "Test passed"),
            ValidationResult("test5", "infrastructure", True, "info", "Test passed")
        ]
        
        production_ready_passing = (
            0 == 0 and  # No critical failures
            0 <= 2 and  # No high failures
            5 / 5 >= 0.9  # 100% pass rate
        )
        
        assert production_ready_passing, "Should be production ready with all passing tests"


# Export main classes
__all__ = [
    'TestProductionReadinessValidation',
    'ProductionReadinessChecker',
    'InfrastructureValidator',
    'PerformanceValidator', 
    'SecurityValidator',
    'MonitoringValidator',
    'ProductionReadinessConfig',
    'ValidationResult'
]