#!/usr/bin/env python3
"""
IntegrationWrapper: Conditional integration patterns for LightRAG/Perplexity routing.

This module provides comprehensive integration wrapper patterns that enable seamless
switching between LightRAG and Perplexity APIs based on feature flags, performance
metrics, and quality assessments. It maintains backward compatibility while adding
advanced routing capabilities.

Key Features:
- Transparent fallback between LightRAG and Perplexity
- Performance comparison and quality assessment
- Circuit breaker protection for unstable integrations
- Response caching and optimization
- Comprehensive error handling and recovery
- Metrics collection for A/B testing analysis
- Thread-safe operations with async support

Integration Patterns:
- Factory pattern for creating appropriate service instances
- Strategy pattern for routing decisions
- Adapter pattern for uniform API interfaces
- Observer pattern for metrics collection
- Circuit breaker pattern for fault tolerance

Requirements:
- Compatible with existing main.py integration
- Maintains existing Perplexity API patterns
- Supports async/await patterns used in Chainlit
- Integrates with existing logging and monitoring

Author: Claude Code (Anthropic)
Created: 2025-08-08
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Coroutine
from contextlib import asynccontextmanager
import openai
import requests
import re
import hashlib

from .config import LightRAGConfig
from .feature_flag_manager import (
    FeatureFlagManager, RoutingContext, RoutingResult, 
    RoutingDecision, UserCohort
)


class ResponseType(Enum):
    """Types of responses from different services."""
    LIGHTRAG = "lightrag"
    PERPLEXITY = "perplexity"
    CACHED = "cached"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER_BLOCKED = "circuit_breaker_blocked"
    HEALTH_CHECK = "health_check"


class QualityMetric(Enum):
    """Quality metrics for response assessment."""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    CITATION_QUALITY = "citation_quality"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"


@dataclass
class ServiceResponse:
    """Unified response structure from any service."""
    content: str
    citations: Optional[List[Dict[str, Any]]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    response_type: ResponseType = ResponseType.PERPLEXITY
    processing_time: float = 0.0
    quality_scores: Optional[Dict[QualityMetric, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    service_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if response was successful."""
        return self.error_details is None and bool(self.content.strip())
    
    @property
    def average_quality_score(self) -> float:
        """Calculate average quality score across all metrics."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores.values()) / len(self.quality_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            'content': self.content,
            'citations': self.citations,
            'confidence_scores': self.confidence_scores,
            'response_type': self.response_type.value,
            'processing_time': self.processing_time,
            'quality_scores': {k.value: v for k, v in self.quality_scores.items()} if self.quality_scores else None,
            'metadata': self.metadata,
            'error_details': self.error_details,
            'service_info': self.service_info,
            'is_success': self.is_success,
            'average_quality_score': self.average_quality_score
        }


@dataclass
class QueryRequest:
    """Unified query request structure."""
    query_text: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query_type: Optional[str] = None
    context_metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    quality_requirements: Dict[QualityMetric, float] = field(default_factory=dict)
    
    def to_routing_context(self) -> RoutingContext:
        """Convert to RoutingContext for feature flag evaluation."""
        return RoutingContext(
            user_id=self.user_id,
            session_id=self.session_id,
            query_text=self.query_text,
            query_type=self.query_type,
            metadata=self.context_metadata
        )


class BaseQueryService(ABC):
    """Abstract base class for query services."""
    
    @abstractmethod
    async def query_async(self, request: QueryRequest) -> ServiceResponse:
        """Execute query asynchronously."""
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get service identifier."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if service is currently available."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform async health check on the service."""
        pass
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service-specific metrics."""
        return {
            'service_name': self.get_service_name(),
            'is_available': self.is_available()
        }


class PerplexityQueryService(BaseQueryService):
    """Perplexity API query service implementation."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.perplexity.ai", 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Perplexity service.
        
        Args:
            api_key: Perplexity API key
            base_url: Base URL for Perplexity API
            logger: Optional logger instance
        """
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        
        # Configure OpenAI client for Perplexity
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    async def query_async(self, request: QueryRequest) -> ServiceResponse:
        """
        Execute query against Perplexity API.
        
        Args:
            request: Unified query request
        
        Returns:
            ServiceResponse with Perplexity results
        """
        start_time = time.time()
        
        try:
            # Prepare the payload for Perplexity API
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in clinical metabolomics. You respond to "
                            "user queries in a helpful manner, with a focus on correct "
                            "scientific detail. Include peer-reviewed sources for all claims. "
                            "For each source/claim, provide a confidence score from 0.0-1.0, formatted as (confidence score: X.X) "
                            "Respond in a single paragraph, never use lists unless explicitly asked."
                        )
                    },
                    {
                        "role": "user",
                        "content": request.query_text
                    }
                ],
                "temperature": 0.1,
                "search_domain_filter": ["-wikipedia.org"],
                "timeout": request.timeout_seconds
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the API request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=request.timeout_seconds
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                citations = response_data.get('citations', [])
                
                # Process content and extract confidence scores
                processed_content, confidence_scores, citation_mapping = self._process_perplexity_response(content, citations)
                
                return ServiceResponse(
                    content=processed_content,
                    citations=citations,
                    confidence_scores=confidence_scores,
                    response_type=ResponseType.PERPLEXITY,
                    processing_time=processing_time,
                    metadata={
                        'model': 'sonar',
                        'status_code': response.status_code,
                        'citation_count': len(citations),
                        'confidence_score_count': len(confidence_scores)
                    },
                    service_info={
                        'service': 'perplexity',
                        'api_version': 'v1',
                        'request_id': response.headers.get('x-request-id'),
                        'model_used': 'sonar'
                    }
                )
            else:
                error_msg = f"Perplexity API error {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                
                return ServiceResponse(
                    content="",
                    response_type=ResponseType.PERPLEXITY,
                    processing_time=processing_time,
                    error_details=error_msg,
                    metadata={'status_code': response.status_code}
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Perplexity service error: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            return ServiceResponse(
                content="",
                response_type=ResponseType.PERPLEXITY,
                processing_time=processing_time,
                error_details=error_msg,
                metadata={'exception_type': type(e).__name__}
            )
    
    def _process_perplexity_response(self, content: str, citations: List[Dict[str, Any]]) -> Tuple[str, Dict[str, float], Dict[str, List[str]]]:
        """
        Process Perplexity response to extract confidence scores and format citations.
        
        Args:
            content: Raw content from Perplexity
            citations: Citation data from Perplexity
        
        Returns:
            Tuple of (processed_content, confidence_scores, citation_mapping)
        """
        # Extract confidence scores from content
        confidence_pattern = r"confidence score:\s*([0-9.]+)(?:\s*\)\s*((?:\[\d+\]\s*)+)|\s+based on\s+(\[\d+\]))"
        matches = re.findall(confidence_pattern, content, re.IGNORECASE)
        
        confidence_scores = {}
        citation_mapping = {}
        
        # Build bibliography mapping
        bibliography_dict = {}
        if citations:
            for i, citation in enumerate(citations, 1):
                bibliography_dict[str(i)] = citation
        
        # Process confidence scores and citations
        for score, refs1, refs2 in matches:
            confidence = float(score)
            refs = refs1 if refs1 else refs2
            ref_nums = re.findall(r"\[(\d+)\]", refs)
            
            for num in ref_nums:
                if num in bibliography_dict:
                    citation_url = bibliography_dict[num]
                    confidence_scores[citation_url] = confidence
                    if num not in citation_mapping:
                        citation_mapping[num] = []
                    citation_mapping[num].append(str(confidence))
        
        # Clean content by removing confidence score annotations
        clean_pattern = r"\(\s*confidence score:\s*[0-9.]+\s*\)"
        cleaned_content = re.sub(clean_pattern, "", content, flags=re.IGNORECASE)
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
        
        return cleaned_content, confidence_scores, citation_mapping
    
    def get_service_name(self) -> str:
        """Get service identifier."""
        return "perplexity"
    
    def is_available(self) -> bool:
        """Check if Perplexity service is available."""
        return bool(self.api_key)
    
    async def health_check(self) -> bool:
        """Perform health check on Perplexity service."""
        try:
            # Simple health check with minimal query
            test_request = QueryRequest(
                query_text="health check",
                timeout_seconds=5.0
            )
            response = await self.query_async(test_request)
            return response.is_success
        except Exception as e:
            self.logger.warning(f"Perplexity health check failed: {e}")
            return False


class LightRAGQueryService(BaseQueryService):
    """LightRAG query service implementation."""
    
    def __init__(self, config: LightRAGConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize LightRAG service.
        
        Args:
            config: LightRAG configuration instance
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.lightrag_instance = None
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
    
    async def _ensure_initialized(self) -> bool:
        """
        Ensure LightRAG is initialized before use.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized and self.lightrag_instance:
            return True
        
        async with self._initialization_lock:
            if self._initialized:
                return True
            
            try:
                # Import and initialize LightRAG
                from lightrag import LightRAG
                from lightrag.llm import openai_complete_if_cache, openai_embedding
                from lightrag.utils import EmbeddingFunc
                
                # Create LightRAG instance with biomedical configuration
                self.lightrag_instance = LightRAG(
                    working_dir=str(self.config.graph_storage_dir),
                    llm_model_func=openai_complete_if_cache,
                    llm_model_name=self.config.model,
                    llm_model_max_async=self.config.max_async,
                    llm_model_max_tokens=self.config.max_tokens,
                    embedding_func=EmbeddingFunc(
                        embedding_dim=1536,
                        max_token_size=8192,
                        func=lambda texts: openai_embedding(
                            texts,
                            model=self.config.embedding_model
                        )
                    )
                )
                
                self._initialized = True
                self.logger.info("LightRAG service initialized successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize LightRAG: {e}")
                return False
    
    async def query_async(self, request: QueryRequest) -> ServiceResponse:
        """
        Execute query against LightRAG.
        
        Args:
            request: Unified query request
        
        Returns:
            ServiceResponse with LightRAG results
        """
        start_time = time.time()
        
        # Ensure LightRAG is initialized
        if not await self._ensure_initialized():
            processing_time = time.time() - start_time
            return ServiceResponse(
                content="",
                response_type=ResponseType.LIGHTRAG,
                processing_time=processing_time,
                error_details="LightRAG initialization failed",
                metadata={'initialization_error': True}
            )
        
        try:
            # Import QueryParam for query configuration
            from lightrag import QueryParam
            
            # Configure query parameters based on request
            query_param = QueryParam(
                mode="hybrid",  # Use hybrid mode for best results
                response_type="Multiple Paragraphs",
                top_k=10,
                max_total_tokens=self.config.max_tokens,
                max_keywords=30
            )
            
            # Execute the query with timeout
            response = await asyncio.wait_for(
                self.lightrag_instance.aquery(request.query_text, param=query_param),
                timeout=request.timeout_seconds
            )
            
            processing_time = time.time() - start_time
            
            # Process the response
            if response and isinstance(response, str) and response.strip():
                return ServiceResponse(
                    content=response.strip(),
                    response_type=ResponseType.LIGHTRAG,
                    processing_time=processing_time,
                    metadata={
                        'query_mode': 'hybrid',
                        'top_k': 10,
                        'max_tokens': self.config.max_tokens,
                        'model': self.config.model,
                        'embedding_model': self.config.embedding_model
                    },
                    service_info={
                        'service': 'lightrag',
                        'version': '1.0',
                        'working_dir': str(self.config.graph_storage_dir)
                    }
                )
            else:
                return ServiceResponse(
                    content="",
                    response_type=ResponseType.LIGHTRAG,
                    processing_time=processing_time,
                    error_details="Empty or invalid response from LightRAG",
                    metadata={'empty_response': True}
                )
                
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            error_msg = f"LightRAG query timeout after {request.timeout_seconds}s"
            self.logger.warning(error_msg)
            
            return ServiceResponse(
                content="",
                response_type=ResponseType.LIGHTRAG,
                processing_time=processing_time,
                error_details=error_msg,
                metadata={'timeout': True, 'timeout_seconds': request.timeout_seconds}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"LightRAG service error: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            return ServiceResponse(
                content="",
                response_type=ResponseType.LIGHTRAG,
                processing_time=processing_time,
                error_details=error_msg,
                metadata={'exception_type': type(e).__name__}
            )
    
    def get_service_name(self) -> str:
        """Get service identifier."""
        return "lightrag"
    
    def is_available(self) -> bool:
        """Check if LightRAG service is available."""
        return bool(self.config.api_key) and self._initialized
    
    async def health_check(self) -> bool:
        """Perform health check on LightRAG service."""
        try:
            if not await self._ensure_initialized():
                return False
            
            # Simple health check with minimal query
            test_request = QueryRequest(
                query_text="test",
                timeout_seconds=5.0
            )
            response = await self.query_async(test_request)
            return response.is_success
        except Exception as e:
            self.logger.warning(f"LightRAG health check failed: {e}")
            return False


class AdvancedCircuitBreaker:
    """Advanced circuit breaker with health monitoring and recovery."""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 300.0, 
                 logger: Optional[logging.Logger] = None):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.logger = logger or logging.getLogger(__name__)
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.is_open = False
        self.recovery_attempts = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.is_open:
                if self._should_attempt_recovery():
                    self.logger.info(f"Circuit breaker attempting recovery (attempt {self.recovery_attempts + 1})")
                    try:
                        result = await func(*args, **kwargs)
                        await self._record_success()
                        return result
                    except Exception as e:
                        await self._record_failure()
                        raise
                else:
                    raise Exception("Circuit breaker is open - service unavailable")
            
            try:
                result = await func(*args, **kwargs)
                await self._record_success()
                return result
            except Exception as e:
                await self._record_failure()
                raise
    
    def _should_attempt_recovery(self) -> bool:
        """Check if recovery should be attempted."""
        if not self.last_failure_time:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
    
    async def _record_success(self) -> None:
        """Record successful operation."""
        if self.is_open:
            self.logger.info("Circuit breaker recovered successfully")
        
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None
        self.recovery_attempts = 0
    
    async def _record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold and not self.is_open:
            self.is_open = True
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        
        if self.is_open:
            self.recovery_attempts += 1
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'is_open': self.is_open,
            'failure_count': self.failure_count,
            'recovery_attempts': self.recovery_attempts,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class ServiceHealthMonitor:
    """Monitor service health and availability."""
    
    def __init__(self, check_interval: float = 60.0, logger: Optional[logging.Logger] = None):
        """Initialize health monitor."""
        self.check_interval = check_interval
        self.logger = logger or logging.getLogger(__name__)
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._services: List[BaseQueryService] = []
        self._running = False
    
    def register_service(self, service: BaseQueryService) -> None:
        """Register a service for health monitoring."""
        self._services.append(service)
        self.health_status[service.get_service_name()] = {
            'is_healthy': False,
            'last_check': None,
            'consecutive_failures': 0,
            'total_checks': 0,
            'successful_checks': 0
        }
    
    async def start_monitoring(self) -> None:
        """Start health monitoring background task."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Service health monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Service health monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_services(self) -> None:
        """Check health of all registered services."""
        for service in self._services:
            service_name = service.get_service_name()
            status = self.health_status[service_name]
            
            try:
                is_healthy = await service.health_check()
                status['is_healthy'] = is_healthy
                status['last_check'] = datetime.now().isoformat()
                status['total_checks'] += 1
                
                if is_healthy:
                    status['successful_checks'] += 1
                    status['consecutive_failures'] = 0
                else:
                    status['consecutive_failures'] += 1
                    
            except Exception as e:
                self.logger.warning(f"Health check failed for {service_name}: {e}")
                status['is_healthy'] = False
                status['consecutive_failures'] += 1
                status['total_checks'] += 1
    
    def get_service_health(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get health status for specific service."""
        return self.health_status.get(service_name)
    
    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all services."""
        return self.health_status.copy()


class IntegratedQueryService:
    """
    Advanced integrated query service that routes between LightRAG and Perplexity.
    
    This is the main service class that applications should use. It handles
    feature flag evaluation, routing decisions, fallback logic, performance
    monitoring, circuit breaker protection, and health monitoring transparently.
    """
    
    def __init__(self, config: LightRAGConfig, perplexity_api_key: str,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the integrated query service.
        
        Args:
            config: LightRAG configuration with feature flags
            perplexity_api_key: API key for Perplexity service
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize services
        self.perplexity_service = PerplexityQueryService(
            api_key=perplexity_api_key,
            logger=self.logger
        )
        
        self.lightrag_service = LightRAGQueryService(
            config=config,
            logger=self.logger
        ) if config.lightrag_integration_enabled else None
        
        # Initialize feature flag manager
        self.feature_manager = FeatureFlagManager(config=config, logger=self.logger)
        
        # Response cache for performance optimization
        self._response_cache: Dict[str, Tuple[ServiceResponse, datetime]] = {}
        self._cache_ttl_minutes = 10
        
        # Quality assessment function (can be overridden)
        self.quality_assessor: Optional[Callable[[ServiceResponse], Dict[QualityMetric, float]]] = None
        
        # Advanced circuit breakers for each service
        self.lightrag_circuit_breaker = AdvancedCircuitBreaker(
            failure_threshold=config.lightrag_circuit_breaker_failure_threshold,
            recovery_timeout=config.lightrag_circuit_breaker_recovery_timeout,
            logger=self.logger
        ) if config.lightrag_enable_circuit_breaker else None
        
        self.perplexity_circuit_breaker = AdvancedCircuitBreaker(
            failure_threshold=3,  # Default for Perplexity
            recovery_timeout=300.0,
            logger=self.logger
        )
        
        # Health monitoring
        self.health_monitor = ServiceHealthMonitor(logger=self.logger)
        if self.perplexity_service:
            self.health_monitor.register_service(self.perplexity_service)
        if self.lightrag_service:
            self.health_monitor.register_service(self.lightrag_service)
        
        # A/B testing metrics
        self._ab_test_metrics: Dict[str, List[Dict[str, Any]]] = {
            'lightrag': [],
            'perplexity': []
        }
        
        # Performance tracking
        self._performance_window = 100  # Keep last 100 requests
        self._request_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"IntegratedQueryService initialized (LightRAG: {'enabled' if self.lightrag_service else 'disabled'})")
        
        # Start health monitoring in background
        asyncio.create_task(self._start_background_tasks())
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        try:
            await self.health_monitor.start_monitoring()
        except Exception as e:
            self.logger.warning(f"Failed to start health monitoring: {e}")
    
    def set_quality_assessor(self, assessor: Callable[[ServiceResponse], Dict[QualityMetric, float]]) -> None:
        """
        Set custom quality assessment function.
        
        Args:
            assessor: Function that takes ServiceResponse and returns quality scores
        """
        self.quality_assessor = assessor
        self.logger.info("Custom quality assessor registered")
    
    async def query_async(self, request: QueryRequest) -> ServiceResponse:
        """
        Execute query with intelligent routing.
        
        This is the main entry point for queries. It handles:
        - Feature flag evaluation
        - Service routing decisions  
        - Fallback logic on failures
        - Performance and quality monitoring
        - Response caching
        
        Args:
            request: Unified query request
        
        Returns:
            ServiceResponse from appropriate service
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                cached_response.response_type = ResponseType.CACHED
                return cached_response
            
            # Get routing decision
            routing_context = request.to_routing_context()
            routing_result = self.feature_manager.should_use_lightrag(routing_context)
            
            # Log routing decision
            self.logger.info(f"Routing decision: {routing_result.decision.value} (reason: {routing_result.reason.value})")
            
            # Execute query based on routing decision
            primary_response = None
            fallback_response = None
            
            if routing_result.decision == RoutingDecision.LIGHTRAG and self.lightrag_service:
                # Try LightRAG first with circuit breaker protection
                if self.lightrag_circuit_breaker:
                    try:
                        primary_response = await self.lightrag_circuit_breaker.call(
                            self._query_with_timeout_protected,
                            self.lightrag_service, request, "LightRAG"
                        )
                    except Exception as e:
                        self.logger.warning(f"LightRAG circuit breaker blocked request: {e}")
                        primary_response = ServiceResponse(
                            content="",
                            response_type=ResponseType.CIRCUIT_BREAKER_BLOCKED,
                            processing_time=0.0,
                            error_details=str(e),
                            metadata={'circuit_breaker': 'open'}
                        )
                else:
                    primary_response = await self._query_with_timeout(
                        self.lightrag_service, request, "LightRAG"
                    )
                
                if primary_response.is_success:
                    # Record success and metrics
                    quality_score = await self._assess_quality(primary_response)
                    self.feature_manager.record_success(
                        "lightrag", 
                        primary_response.processing_time, 
                        quality_score
                    )
                    await self._record_ab_test_metrics("lightrag", primary_response, request)
                else:
                    # Record failure and try fallback
                    self.feature_manager.record_failure("lightrag", primary_response.error_details)
                    
                    if self.config.lightrag_fallback_to_perplexity:
                        self.logger.info("Falling back to Perplexity after LightRAG failure")
                        if self.perplexity_circuit_breaker:
                            try:
                                fallback_response = await self.perplexity_circuit_breaker.call(
                                    self._query_with_timeout_protected,
                                    self.perplexity_service, request, "Perplexity (fallback)"
                                )
                            except Exception as e:
                                fallback_response = ServiceResponse(
                                    content="Service temporarily unavailable",
                                    response_type=ResponseType.CIRCUIT_BREAKER_BLOCKED,
                                    processing_time=0.0,
                                    error_details=str(e)
                                )
                        else:
                            fallback_response = await self._query_with_timeout(
                                self.perplexity_service, request, "Perplexity (fallback)"
                            )
            else:
                # Use Perplexity directly with circuit breaker protection
                if self.perplexity_circuit_breaker:
                    try:
                        primary_response = await self.perplexity_circuit_breaker.call(
                            self._query_with_timeout_protected,
                            self.perplexity_service, request, "Perplexity"
                        )
                    except Exception as e:
                        self.logger.warning(f"Perplexity circuit breaker blocked request: {e}")
                        primary_response = ServiceResponse(
                            content="Service temporarily unavailable. Please try again later.",
                            response_type=ResponseType.CIRCUIT_BREAKER_BLOCKED,
                            processing_time=0.0,
                            error_details=str(e),
                            metadata={'circuit_breaker': 'open'}
                        )
                else:
                    primary_response = await self._query_with_timeout(
                        self.perplexity_service, request, "Perplexity"
                    )
                
                if primary_response.is_success:
                    # Record success and metrics
                    quality_score = await self._assess_quality(primary_response)
                    self.feature_manager.record_success(
                        "perplexity", 
                        primary_response.processing_time, 
                        quality_score
                    )
                    await self._record_ab_test_metrics("perplexity", primary_response, request)
                else:
                    # Record failure
                    self.feature_manager.record_failure("perplexity", primary_response.error_details)
            
            # Determine final response
            final_response = fallback_response if fallback_response and fallback_response.is_success else primary_response
            
            # Add routing metadata
            final_response.metadata.update({
                'routing_decision': routing_result.decision.value,
                'routing_reason': routing_result.reason.value,
                'user_cohort': routing_result.user_cohort.value if routing_result.user_cohort else None,
                'routing_confidence': routing_result.confidence,
                'fallback_used': bool(fallback_response),
                'total_processing_time': time.time() - start_time
            })
            
            # Cache successful responses
            if final_response.is_success:
                self._cache_response(cache_key, final_response)
            
            # Record request history for performance analysis
            await self._record_request_history(request, final_response, routing_result)
            
            return final_response
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"IntegratedQueryService error: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Return fallback error response
            return ServiceResponse(
                content="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                response_type=ResponseType.FALLBACK,
                processing_time=processing_time,
                error_details=error_msg,
                metadata={'exception_type': type(e).__name__}
            )
    
    async def _query_with_timeout(self, service: BaseQueryService, request: QueryRequest, service_name: str) -> ServiceResponse:
        """
        Execute query with timeout protection.
        
        Args:
            service: Service to query
            request: Query request
            service_name: Human-readable service name for logging
        
        Returns:
            ServiceResponse from the service
        """
        try:
            self.logger.debug(f"Querying {service_name}...")
            response = await asyncio.wait_for(
                service.query_async(request),
                timeout=request.timeout_seconds
            )
            
            if response.is_success:
                self.logger.info(f"{service_name} query successful ({response.processing_time:.2f}s)")
            else:
                self.logger.warning(f"{service_name} query failed: {response.error_details}")
            
            return response
            
        except asyncio.TimeoutError:
            timeout_msg = f"{service_name} query timeout after {request.timeout_seconds}s"
            self.logger.warning(timeout_msg)
            
            return ServiceResponse(
                content="",
                response_type=ResponseType.LIGHTRAG if "lightrag" in service_name.lower() else ResponseType.PERPLEXITY,
                processing_time=request.timeout_seconds,
                error_details=timeout_msg,
                metadata={'timeout': True}
            )
        
        except Exception as e:
            error_msg = f"{service_name} unexpected error: {str(e)}"
            self.logger.error(error_msg)
            
            return ServiceResponse(
                content="",
                response_type=ResponseType.LIGHTRAG if "lightrag" in service_name.lower() else ResponseType.PERPLEXITY,
                processing_time=0.0,
                error_details=error_msg,
                metadata={'exception_type': type(e).__name__}
            )
    
    async def _assess_quality(self, response: ServiceResponse) -> Optional[float]:
        """
        Assess response quality using configured assessor.
        
        Args:
            response: ServiceResponse to assess
        
        Returns:
            Average quality score or None if assessment failed
        """
        if not self.quality_assessor or not response.is_success:
            return None
        
        try:
            quality_scores = self.quality_assessor(response)
            response.quality_scores = quality_scores
            return response.average_quality_score
        
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
            return None
    
    def _generate_cache_key(self, request: QueryRequest) -> str:
        """
        Generate cache key for request.
        
        Args:
            request: Query request
        
        Returns:
            Cache key string
        """
        # Create a hash of the query and key parameters
        key_data = {
            'query': request.query_text,
            'type': request.query_type,
            'timeout': request.timeout_seconds
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _get_cached_response(self, cache_key: str) -> Optional[ServiceResponse]:
        """
        Retrieve cached response if still valid.
        
        Args:
            cache_key: Cache key to look up
        
        Returns:
            Cached ServiceResponse or None if expired/missing
        """
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            
            # Check if cache entry is still valid
            if datetime.now() - timestamp < timedelta(minutes=self._cache_ttl_minutes):
                self.logger.debug("Returning cached response")
                return response
            else:
                # Remove expired entry
                del self._response_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: ServiceResponse) -> None:
        """
        Cache response for future use.
        
        Args:
            cache_key: Cache key for storage
            response: ServiceResponse to cache
        """
        # Limit cache size
        if len(self._response_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self._response_cache.keys(),
                key=lambda k: self._response_cache[k][1]
            )[:20]
            
            for key in oldest_keys:
                del self._response_cache[key]
        
        self._response_cache[cache_key] = (response, datetime.now())
        self.logger.debug(f"Cached response for key {cache_key[:16]}...")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance metrics and service status
        """
        summary = self.feature_manager.get_performance_summary()
        
        # Add service availability info
        summary['services'] = {
            'perplexity': {
                'available': self.perplexity_service.is_available(),
                'service_name': self.perplexity_service.get_service_name(),
                'circuit_breaker': self.perplexity_circuit_breaker.get_state() if self.perplexity_circuit_breaker else None
            }
        }
        
        if self.lightrag_service:
            summary['services']['lightrag'] = {
                'available': self.lightrag_service.is_available(),
                'service_name': self.lightrag_service.get_service_name(),
                'circuit_breaker': self.lightrag_circuit_breaker.get_state() if self.lightrag_circuit_breaker else None
            }
        
        # Add cache info
        summary['cache_info'] = {
            'response_cache_size': len(self._response_cache),
            'cache_ttl_minutes': self._cache_ttl_minutes,
            'quality_assessor_enabled': self.quality_assessor is not None
        }
        
        # Add health monitoring info
        summary['health_monitoring'] = self.health_monitor.get_all_health_status()
        
        # Add A/B testing metrics
        summary['ab_testing'] = self.get_ab_test_metrics()
        
        # Add request history summary
        if self._request_history:
            recent_requests = self._request_history[-20:]  # Last 20 requests
            summary['recent_performance'] = {
                'total_requests': len(self._request_history),
                'success_rate': len([r for r in recent_requests if r['success']]) / len(recent_requests),
                'avg_response_time': sum(r['processing_time'] for r in recent_requests) / len(recent_requests),
                'routing_distribution': {
                    decision: len([r for r in recent_requests if r['routing_decision'] == decision])
                    for decision in set(r['routing_decision'] for r in recent_requests)
                }
            }
        else:
            summary['recent_performance'] = {'total_requests': 0}
        
        return summary
    
    async def _query_with_timeout_protected(self, service: BaseQueryService, request: QueryRequest, service_name: str) -> ServiceResponse:
        """Protected query method for circuit breaker usage."""
        response = await self._query_with_timeout(service, request, service_name)
        if not response.is_success:
            raise Exception(response.error_details or "Query failed")
        return response
    
    async def _record_ab_test_metrics(self, service: str, response: ServiceResponse, request: QueryRequest) -> None:
        """Record A/B testing metrics."""
        try:
            metric_entry = {
                'timestamp': datetime.now().isoformat(),
                'service': service,
                'processing_time': response.processing_time,
                'quality_score': response.average_quality_score,
                'success': response.is_success,
                'query_length': len(request.query_text),
                'query_type': request.query_type,
                'user_id': request.user_id,
                'session_id': request.session_id
            }
            
            self._ab_test_metrics[service].append(metric_entry)
            
            # Keep only recent metrics to prevent memory growth
            if len(self._ab_test_metrics[service]) > self._performance_window:
                self._ab_test_metrics[service] = self._ab_test_metrics[service][-self._performance_window:]
                
        except Exception as e:
            self.logger.warning(f"Failed to record A/B test metrics: {e}")
    
    async def _record_request_history(self, request: QueryRequest, response: ServiceResponse, routing_result: RoutingResult) -> None:
        """Record request history for performance analysis."""
        try:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'query_hash': hashlib.md5(request.query_text.encode()).hexdigest()[:16],
                'routing_decision': routing_result.decision.value,
                'routing_reason': routing_result.reason.value,
                'user_cohort': routing_result.user_cohort.value if routing_result.user_cohort else None,
                'response_type': response.response_type.value,
                'processing_time': response.processing_time,
                'success': response.is_success,
                'quality_score': response.average_quality_score,
                'error': response.error_details is not None
            }
            
            self._request_history.append(history_entry)
            
            # Keep only recent history
            if len(self._request_history) > self._performance_window:
                self._request_history = self._request_history[-self._performance_window:]
                
        except Exception as e:
            self.logger.warning(f"Failed to record request history: {e}")
    
    def get_ab_test_metrics(self) -> Dict[str, Any]:
        """Get A/B testing performance comparison metrics."""
        try:
            metrics = {}
            
            for service, data in self._ab_test_metrics.items():
                if not data:
                    metrics[service] = {'sample_size': 0}
                    continue
                
                successful = [d for d in data if d['success']]
                response_times = [d['processing_time'] for d in data if d['success']]
                quality_scores = [d['quality_score'] for d in data if d['success'] and d['quality_score'] > 0]
                
                metrics[service] = {
                    'sample_size': len(data),
                    'success_rate': len(successful) / len(data) if data else 0,
                    'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                    'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                    'error_count': len([d for d in data if not d['success']]),
                    'total_requests': len(data)
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate A/B test metrics: {e}")
            return {}
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        self._response_cache.clear()
        self.feature_manager.clear_caches()
        self.logger.info("All caches cleared")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        self.logger.info("Shutting down IntegratedQueryService...")
        await self.health_monitor.stop_monitoring()
        self.clear_cache()
        self.logger.info("IntegratedQueryService shutdown complete")


# Convenience factory functions for easy integration
def create_integrated_service(config: LightRAGConfig, perplexity_api_key: str,
                            logger: Optional[logging.Logger] = None) -> IntegratedQueryService:
    """
    Factory function to create IntegratedQueryService.
    
    Args:
        config: LightRAG configuration with feature flags
        perplexity_api_key: Perplexity API key
        logger: Optional logger instance
    
    Returns:
        Configured IntegratedQueryService instance
    """
    return IntegratedQueryService(
        config=config,
        perplexity_api_key=perplexity_api_key,
        logger=logger
    )


def create_perplexity_only_service(api_key: str, logger: Optional[logging.Logger] = None) -> PerplexityQueryService:
    """
    Factory function to create Perplexity-only service.
    
    Args:
        api_key: Perplexity API key
        logger: Optional logger instance
    
    Returns:
        Configured PerplexityQueryService instance
    """
    return PerplexityQueryService(api_key=api_key, logger=logger)


def create_lightrag_only_service(config: LightRAGConfig, logger: Optional[logging.Logger] = None) -> LightRAGQueryService:
    """
    Factory function to create LightRAG-only service.
    
    Args:
        config: LightRAG configuration
        logger: Optional logger instance
    
    Returns:
        Configured LightRAGQueryService instance
    """
    return LightRAGQueryService(config=config, logger=logger)


# Backward compatibility factory functions
def create_service_with_fallback(lightrag_config: LightRAGConfig, perplexity_api_key: str,
                                enable_ab_testing: bool = False, 
                                logger: Optional[logging.Logger] = None) -> IntegratedQueryService:
    """
    Backward compatibility factory for existing integrations.
    
    Args:
        lightrag_config: LightRAG configuration
        perplexity_api_key: Perplexity API key
        enable_ab_testing: Enable A/B testing features
        logger: Optional logger instance
    
    Returns:
        IntegratedQueryService with backward compatibility
    """
    if enable_ab_testing:
        lightrag_config.lightrag_enable_ab_testing = True
        lightrag_config.lightrag_enable_performance_comparison = True
    
    return IntegratedQueryService(
        config=lightrag_config,
        perplexity_api_key=perplexity_api_key,
        logger=logger
    )


def create_production_service(config: LightRAGConfig, perplexity_api_key: str,
                            quality_assessor: Optional[Callable[[ServiceResponse], Dict[QualityMetric, float]]] = None,
                            logger: Optional[logging.Logger] = None) -> IntegratedQueryService:
    """
    Factory for production-ready service with all features enabled.
    
    Args:
        config: LightRAG configuration with feature flags
        perplexity_api_key: Perplexity API key
        quality_assessor: Optional quality assessment function
        logger: Optional logger instance
    
    Returns:
        Production-ready IntegratedQueryService
    """
    service = IntegratedQueryService(
        config=config,
        perplexity_api_key=perplexity_api_key,
        logger=logger
    )
    
    if quality_assessor:
        service.set_quality_assessor(quality_assessor)
    
    return service


# Context manager for service lifecycle
@asynccontextmanager
async def managed_query_service(config: LightRAGConfig, perplexity_api_key: str,
                              logger: Optional[logging.Logger] = None):
    """
    Async context manager for service lifecycle management.
    
    Args:
        config: LightRAG configuration
        perplexity_api_key: Perplexity API key
        logger: Optional logger instance
    
    Yields:
        IntegratedQueryService with automatic lifecycle management
    """
    service = None
    try:
        service = IntegratedQueryService(
            config=config,
            perplexity_api_key=perplexity_api_key,
            logger=logger
        )
        yield service
    finally:
        if service:
            await service.shutdown()