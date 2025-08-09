#!/usr/bin/env python3
"""
Clinical Metabolomics Fallback System Configuration
=================================================

This module provides specialized configuration for the fallback system optimized
specifically for clinical metabolomics queries. It includes tuned parameters for
biomedical accuracy, scientific query timeouts, and metabolomics-specific caching.

Key Features:
- Higher confidence thresholds for biomedical accuracy
- Extended timeouts for complex scientific queries
- Metabolomics-specific cache patterns
- Clinical reliability triggers
- Scientific literature processing optimizations

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Configure integrated fallback system for clinical metabolomics
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

try:
    from .config import LightRAGConfig
    from .comprehensive_fallback_system import FallbackLevel, DegradationStrategy
    from .query_router import FallbackStrategy
    from .cost_based_circuit_breaker import CostBasedCircuitBreaker
except ImportError:
    # Allow standalone execution
    try:
        from config import LightRAGConfig
        from comprehensive_fallback_system import FallbackLevel, DegradationStrategy
        from query_router import FallbackStrategy
        from cost_based_circuit_breaker import CostBasedCircuitBreaker
    except ImportError:
        # Create mock classes for validation
        from enum import IntEnum
        
        class FallbackLevel(IntEnum):
            FULL_LLM_WITH_CONFIDENCE = 1
            SIMPLIFIED_LLM = 2
            KEYWORD_BASED_ONLY = 3
            EMERGENCY_CACHE = 4
            DEFAULT_ROUTING = 5
        
        class DegradationStrategy:
            pass
            
        class FallbackStrategy:
            def __init__(self, name, level, description, confidence_threshold, timeout_seconds, max_retries):
                self.name = name
                self.level = level
                self.description = description
                self.confidence_threshold = confidence_threshold
                self.timeout_seconds = timeout_seconds
                self.max_retries = max_retries
        
        # Mock LightRAGConfig for validation
        class LightRAGConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)


class ClinicalAccuracyLevel(Enum):
    """Clinical accuracy requirements for different query types"""
    DIAGNOSTIC = "diagnostic"          # Highest accuracy for diagnostic queries
    THERAPEUTIC = "therapeutic"        # High accuracy for treatment queries  
    RESEARCH = "research"              # Standard accuracy for research queries
    EDUCATIONAL = "educational"        # Lower accuracy for educational content
    GENERAL = "general"                # Basic accuracy for general questions


@dataclass
class ClinicalMetabolomicsThresholds:
    """
    Confidence thresholds optimized for clinical metabolomics accuracy requirements.
    
    These thresholds are higher than general-purpose settings to ensure
    biomedical accuracy and reliability for scientific applications.
    """
    
    # Primary confidence thresholds (higher for biomedical accuracy)
    diagnostic_confidence_threshold: float = 0.90      # Very high for diagnostic queries
    therapeutic_confidence_threshold: float = 0.85     # High for therapeutic queries
    research_confidence_threshold: float = 0.75        # Standard for research queries
    educational_confidence_threshold: float = 0.65     # Lower for educational content
    general_confidence_threshold: float = 0.60         # Basic for general questions
    
    # Fallback decision thresholds
    lightrag_fallback_threshold: float = 0.70          # When to fallback from LightRAG
    perplexity_fallback_threshold: float = 0.60        # When to fallback to emergency cache
    emergency_cache_threshold: float = 0.40            # When to use cached responses
    
    # Quality assessment thresholds
    minimum_scientific_accuracy: float = 0.80          # Minimum for scientific content
    citation_quality_threshold: float = 0.75           # Minimum for citation accuracy
    factual_accuracy_threshold: float = 0.85           # Minimum for factual claims
    
    # Uncertainty-aware thresholds
    high_uncertainty_threshold: float = 0.30           # High uncertainty indicator
    medium_uncertainty_threshold: float = 0.20         # Medium uncertainty indicator
    low_uncertainty_threshold: float = 0.10            # Low uncertainty indicator


@dataclass
class ClinicalMetabolomicsTimeouts:
    """
    Timeout configurations optimized for complex scientific queries.
    
    Clinical metabolomics queries often involve complex literature searches
    and detailed analysis, requiring longer processing times than general queries.
    """
    
    # Primary processing timeouts (extended for scientific complexity)
    lightrag_primary_timeout: float = 45.0             # Primary LightRAG processing
    lightrag_complex_query_timeout: float = 60.0       # Complex multi-step queries
    lightrag_literature_search_timeout: float = 90.0   # Literature search queries
    
    # Perplexity API timeouts
    perplexity_standard_timeout: float = 35.0          # Standard Perplexity queries
    perplexity_scientific_timeout: float = 50.0        # Scientific literature queries
    
    # Fallback system timeouts
    router_decision_timeout: float = 5.0               # Quick routing decisions
    confidence_analysis_timeout: float = 10.0          # Confidence scoring timeout
    fallback_activation_timeout: float = 2.0           # Emergency fallback timeout
    
    # Health check and monitoring timeouts
    health_check_timeout: float = 15.0                 # Backend health checks
    system_monitoring_timeout: float = 5.0             # System status checks
    
    # Circuit breaker timeouts
    circuit_breaker_timeout: float = 300.0             # 5 minutes recovery time
    cost_breaker_timeout: float = 600.0                # 10 minutes for cost issues


@dataclass
class MetabolomicsCachePatterns:
    """
    Cache warming patterns for common clinical metabolomics queries.
    
    Pre-populated cache entries help ensure fast responses for frequently
    asked metabolomics questions and improve system reliability.
    """
    
    # Common metabolomics query patterns
    common_queries: List[str] = field(default_factory=lambda: [
        "what is clinical metabolomics",
        "metabolomics biomarkers for diabetes",
        "LC-MS methods in metabolomics",
        "GC-MS metabolomics analysis",
        "metabolomics data preprocessing",
        "clinical metabolomics applications",
        "metabolomics pathway analysis",
        "biomarker discovery metabolomics",
        "metabolomics quality control",
        "untargeted metabolomics workflow",
        "targeted metabolomics analysis",
        "metabolomics statistical analysis",
        "clinical metabolomics validation",
        "metabolomics sample preparation",
        "metabolomics data interpretation",
        "plasma metabolomics profiling",
        "urine metabolomics analysis", 
        "tissue metabolomics methods",
        "metabolomics disease diagnosis",
        "pharmacometabolomics applications"
    ])
    
    # High-priority scientific terms for caching
    priority_terms: Set[str] = field(default_factory=lambda: {
        "metabolomics", "metabolome", "biomarker", "LC-MS", "GC-MS", "NMR",
        "pathway analysis", "KEGG", "HMDB", "MetaboAnalyst", "XCMS",
        "clinical validation", "biomarker discovery", "precision medicine",
        "pharmacometabolomics", "lipidomics", "glycomics", "proteomics",
        "systems biology", "omics integration", "biostatistics"
    })
    
    # Cache warming schedule
    warm_cache_on_startup: bool = True
    cache_refresh_interval_hours: int = 24
    cache_size_limit: int = 1000
    cache_ttl_hours: int = 72


@dataclass 
class ClinicalReliabilitySettings:
    """
    Reliability and safety settings for clinical metabolomics applications.
    
    These settings ensure high reliability and safety for medical/scientific
    applications where accuracy is critical.
    """
    
    # Failure tolerance (stricter for clinical applications)
    max_consecutive_failures: int = 2               # Stricter failure tolerance
    failure_rate_threshold: float = 0.05           # 5% max failure rate
    quality_degradation_threshold: float = 0.15    # 15% quality drop triggers alert
    
    # Recovery settings (faster recovery for clinical needs)
    auto_recovery_enabled: bool = True
    recovery_validation_samples: int = 5           # Samples to validate recovery
    recovery_success_threshold: float = 0.85       # Success rate for recovery
    
    # Safety mechanisms
    enable_citation_verification: bool = True       # Verify scientific citations
    require_source_attribution: bool = True        # Require source attribution
    enable_fact_checking: bool = True              # Enable factual accuracy checks
    
    # Monitoring and alerting (enhanced for clinical use)
    enable_real_time_monitoring: bool = True
    alert_on_accuracy_drop: bool = True
    alert_threshold_accuracy: float = 0.80
    enable_performance_tracking: bool = True


@dataclass
class LightRAGScientificOptimization:
    """
    LightRAG parameter optimizations for scientific literature processing.
    
    These parameters are tuned for processing scientific papers, technical
    documents, and biomedical literature effectively.
    """
    
    # Model selection (optimized for scientific content)
    primary_model: str = "gpt-4o"                   # Higher capability for complex analysis
    fallback_model: str = "gpt-4o-mini"           # Efficient fallback
    embedding_model: str = "text-embedding-3-large" # Better scientific embeddings
    
    # Token limits (increased for scientific documents)
    max_tokens_primary: int = 32768                 # Full context for complex analysis
    max_tokens_fallback: int = 16384               # Reasonable fallback limit
    max_tokens_summary: int = 8192                 # Summary generation limit
    
    # Processing parameters
    max_async_operations: int = 8                   # Balanced for scientific processing
    chunk_size_scientific: int = 2048              # Larger chunks for scientific text
    chunk_overlap_scientific: int = 256            # Overlap for context preservation
    
    # Quality and accuracy settings
    enable_scientific_validation: bool = True       # Validate scientific accuracy
    require_peer_review_sources: bool = False      # Prefer peer-reviewed sources (optional)
    enable_citation_extraction: bool = True        # Extract and verify citations
    
    # Search and retrieval optimization
    max_search_results: int = 20                   # More results for comprehensive analysis
    similarity_threshold: float = 0.75             # Higher threshold for scientific relevance
    rerank_results: bool = True                    # Re-rank results for scientific relevance


class ClinicalMetabolomicsFallbackConfig:
    """
    Comprehensive fallback system configuration optimized for clinical metabolomics.
    
    This class combines all the specialized configurations into a unified system
    that provides the optimal settings for biomedical query processing.
    """
    
    def __init__(self, 
                 environment: str = "production",
                 accuracy_level: ClinicalAccuracyLevel = ClinicalAccuracyLevel.RESEARCH):
        """
        Initialize clinical metabolomics fallback configuration.
        
        Args:
            environment: Deployment environment (development, staging, production)
            accuracy_level: Required accuracy level for clinical applications
        """
        self.environment = environment
        self.accuracy_level = accuracy_level
        self.created_at = datetime.now()
        
        # Initialize configuration components
        self.thresholds = ClinicalMetabolomicsThresholds()
        self.timeouts = ClinicalMetabolomicsTimeouts()
        self.cache_patterns = MetabolomicsCachePatterns()
        self.reliability = ClinicalReliabilitySettings()
        self.lightrag_optimization = LightRAGScientificOptimization()
        
        # Apply environment-specific adjustments
        self._apply_environment_adjustments()
        
        # Apply accuracy level adjustments
        self._apply_accuracy_level_adjustments()
    
    def _apply_environment_adjustments(self):
        """Apply environment-specific configuration adjustments."""
        if self.environment == "development":
            # More lenient settings for development
            self.thresholds.research_confidence_threshold = 0.65
            self.thresholds.lightrag_fallback_threshold = 0.60
            self.timeouts.lightrag_primary_timeout = 30.0
            self.reliability.max_consecutive_failures = 5
            self.lightrag_optimization.max_async_operations = 4
            
        elif self.environment == "staging":
            # Balanced settings for staging
            self.thresholds.research_confidence_threshold = 0.70
            self.timeouts.lightrag_primary_timeout = 40.0
            self.reliability.max_consecutive_failures = 3
            
        elif self.environment == "production":
            # Strictest settings for production
            self.thresholds.research_confidence_threshold = 0.75
            self.reliability.max_consecutive_failures = 2
            self.reliability.enable_real_time_monitoring = True
    
    def _apply_accuracy_level_adjustments(self):
        """Apply accuracy level specific adjustments."""
        if self.accuracy_level == ClinicalAccuracyLevel.DIAGNOSTIC:
            # Highest accuracy for diagnostic applications
            self.thresholds.lightrag_fallback_threshold = 0.85
            self.thresholds.minimum_scientific_accuracy = 0.90
            self.reliability.max_consecutive_failures = 1
            self.reliability.require_source_attribution = True
            
        elif self.accuracy_level == ClinicalAccuracyLevel.THERAPEUTIC:
            # High accuracy for therapeutic applications
            self.thresholds.lightrag_fallback_threshold = 0.80
            self.thresholds.minimum_scientific_accuracy = 0.85
            self.reliability.max_consecutive_failures = 2
            
        elif self.accuracy_level == ClinicalAccuracyLevel.EDUCATIONAL:
            # More lenient for educational content
            self.thresholds.lightrag_fallback_threshold = 0.65
            self.thresholds.minimum_scientific_accuracy = 0.70
            self.reliability.max_consecutive_failures = 3
    
    def get_lightrag_config(self) -> LightRAGConfig:
        """
        Generate optimized LightRAG configuration for clinical metabolomics.
        
        Returns:
            LightRAGConfig: Configured LightRAG instance
        """
        return LightRAGConfig(
            # Model configuration
            model=self.lightrag_optimization.primary_model,
            embedding_model=self.lightrag_optimization.embedding_model,
            max_tokens=self.lightrag_optimization.max_tokens_primary,
            max_async=self.lightrag_optimization.max_async_operations,
            
            # Timeout configuration
            lightrag_integration_timeout_seconds=self.timeouts.lightrag_primary_timeout,
            
            # Quality and reliability
            lightrag_min_quality_threshold=self.thresholds.minimum_scientific_accuracy,
            relevance_confidence_threshold=self.thresholds.research_confidence_threshold,
            relevance_minimum_threshold=self.thresholds.general_confidence_threshold,
            
            # Circuit breaker configuration
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=self.reliability.max_consecutive_failures,
            lightrag_circuit_breaker_recovery_timeout=self.timeouts.circuit_breaker_timeout,
            
            # Feature flags
            lightrag_integration_enabled=True,
            lightrag_fallback_to_perplexity=True,
            enable_relevance_scoring=True,
            enable_parallel_relevance_processing=True,
            
            # Cost management
            enable_cost_tracking=True,
            enable_budget_alerts=True,
            cost_alert_threshold_percentage=75.0,  # Conservative for clinical use
            
            # Monitoring and logging
            enable_audit_trail=True,
            log_level="INFO" if self.environment != "development" else "DEBUG"
        )
    
    def get_fallback_strategies(self) -> List[FallbackStrategy]:
        """
        Generate fallback strategies optimized for clinical metabolomics.
        
        Returns:
            List[FallbackStrategy]: Configured fallback strategies
        """
        return [
            # Primary strategy: Full LLM analysis with high confidence
            FallbackStrategy(
                name="clinical_primary_analysis",
                level=FallbackLevel.FULL_LLM_WITH_CONFIDENCE,
                description="Full LLM analysis with clinical-grade confidence scoring",
                confidence_threshold=self.thresholds.research_confidence_threshold,
                timeout_seconds=self.timeouts.lightrag_primary_timeout,
                max_retries=2
            ),
            
            # Secondary strategy: Simplified LLM for complex queries
            FallbackStrategy(
                name="clinical_simplified_analysis", 
                level=FallbackLevel.SIMPLIFIED_LLM,
                description="Simplified LLM analysis for complex scientific queries",
                confidence_threshold=self.thresholds.educational_confidence_threshold,
                timeout_seconds=self.timeouts.lightrag_complex_query_timeout,
                max_retries=2
            ),
            
            # Tertiary strategy: Keyword-based routing
            FallbackStrategy(
                name="metabolomics_keyword_routing",
                level=FallbackLevel.KEYWORD_BASED_ONLY,
                description="Metabolomics-specific keyword-based query classification",
                confidence_threshold=self.thresholds.general_confidence_threshold,
                timeout_seconds=self.timeouts.router_decision_timeout,
                max_retries=1
            ),
            
            # Emergency strategy: Cached responses
            FallbackStrategy(
                name="metabolomics_emergency_cache",
                level=FallbackLevel.EMERGENCY_CACHE,
                description="Pre-cached responses for common metabolomics queries",
                confidence_threshold=self.thresholds.emergency_cache_threshold,
                timeout_seconds=self.timeouts.fallback_activation_timeout,
                max_retries=1
            ),
            
            # Last resort: Default routing
            FallbackStrategy(
                name="clinical_default_routing",
                level=FallbackLevel.DEFAULT_ROUTING,
                description="Default routing with clinical safety warnings",
                confidence_threshold=0.1,  # Always available
                timeout_seconds=self.timeouts.fallback_activation_timeout,
                max_retries=1
            )
        ]
    
    def get_cache_warming_queries(self) -> List[str]:
        """
        Get cache warming queries for metabolomics applications.
        
        Returns:
            List[str]: Queries to pre-populate cache
        """
        return self.cache_patterns.common_queries
    
    def get_degradation_strategies(self) -> Dict[str, Any]:
        """
        Get graceful degradation strategies for clinical use.
        
        Returns:
            Dict: Degradation strategy configuration
        """
        return {
            "progressive_timeout_reduction": {
                "enabled": True,
                "reduction_factor": 0.8,  # Conservative reduction
                "minimum_timeout": 10.0   # Minimum timeout for clinical safety
            },
            "quality_threshold_adjustment": {
                "enabled": True,
                "adjustment_factor": 0.9,  # Conservative adjustment
                "minimum_threshold": self.thresholds.minimum_scientific_accuracy
            },
            "cache_warming": {
                "enabled": True,
                "queries": self.cache_patterns.common_queries,
                "refresh_interval": self.cache_patterns.cache_refresh_interval_hours
            },
            "load_shedding": {
                "enabled": True,
                "priority_queries": ["diagnostic", "therapeutic"],
                "shed_threshold": 0.8  # Start shedding at 80% capacity
            }
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        Get monitoring configuration for clinical applications.
        
        Returns:
            Dict: Monitoring configuration
        """
        return {
            "enable_real_time_monitoring": self.reliability.enable_real_time_monitoring,
            "health_check_interval": 30,  # Every 30 seconds for clinical apps
            "performance_tracking": self.reliability.enable_performance_tracking,
            "alert_thresholds": {
                "accuracy_drop": self.reliability.alert_threshold_accuracy,
                "failure_rate": self.reliability.failure_rate_threshold,
                "response_time": self.timeouts.lightrag_primary_timeout * 1.5
            },
            "quality_metrics": {
                "enable_citation_verification": self.reliability.enable_citation_verification,
                "enable_fact_checking": self.reliability.enable_fact_checking,
                "source_attribution_required": self.reliability.require_source_attribution
            }
        }
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the clinical metabolomics configuration.
        
        Returns:
            List[str]: Validation errors (empty if valid)
        """
        errors = []
        
        # Validate thresholds are in reasonable ranges
        if not (0.5 <= self.thresholds.research_confidence_threshold <= 1.0):
            errors.append("Research confidence threshold must be between 0.5 and 1.0")
        
        if not (0.8 <= self.thresholds.minimum_scientific_accuracy <= 1.0):
            errors.append("Minimum scientific accuracy must be between 0.8 and 1.0")
        
        # Validate timeouts are reasonable
        if self.timeouts.lightrag_primary_timeout < 10.0:
            errors.append("Primary timeout too low for scientific queries (minimum 10s)")
        
        if self.timeouts.lightrag_literature_search_timeout < 30.0:
            errors.append("Literature search timeout too low (minimum 30s)")
        
        # Validate reliability settings
        if self.reliability.max_consecutive_failures < 1:
            errors.append("Max consecutive failures must be at least 1")
        
        # Validate cache settings
        if len(self.cache_patterns.common_queries) == 0:
            errors.append("Cache patterns must include at least one common query")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dict: Configuration as dictionary
        """
        return {
            "environment": self.environment,
            "accuracy_level": self.accuracy_level.value,
            "created_at": self.created_at.isoformat(),
            "thresholds": asdict(self.thresholds),
            "timeouts": asdict(self.timeouts),
            "cache_patterns": asdict(self.cache_patterns),
            "reliability": asdict(self.reliability),
            "lightrag_optimization": asdict(self.lightrag_optimization)
        }
    
    def save_to_file(self, file_path: Path):
        """
        Save configuration to JSON file.
        
        Args:
            file_path: Path to save configuration
        """
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'ClinicalMetabolomicsFallbackConfig':
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            ClinicalMetabolomicsFallbackConfig: Loaded configuration
        """
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create instance with basic parameters
        instance = cls(
            environment=config_dict.get("environment", "production"),
            accuracy_level=ClinicalAccuracyLevel(config_dict.get("accuracy_level", "research"))
        )
        
        # Override with saved values
        if "thresholds" in config_dict:
            instance.thresholds = ClinicalMetabolomicsThresholds(**config_dict["thresholds"])
        if "timeouts" in config_dict:
            instance.timeouts = ClinicalMetabolomicsTimeouts(**config_dict["timeouts"])
        if "reliability" in config_dict:
            instance.reliability = ClinicalReliabilitySettings(**config_dict["reliability"])
        if "lightrag_optimization" in config_dict:
            instance.lightrag_optimization = LightRAGScientificOptimization(**config_dict["lightrag_optimization"])
        
        return instance


def create_clinical_metabolomics_config(
    environment: str = "production", 
    accuracy_level: str = "research"
) -> ClinicalMetabolomicsFallbackConfig:
    """
    Factory function to create clinical metabolomics configuration.
    
    Args:
        environment: Deployment environment
        accuracy_level: Required accuracy level
        
    Returns:
        ClinicalMetabolomicsFallbackConfig: Configured instance
    """
    accuracy_enum = ClinicalAccuracyLevel(accuracy_level)
    return ClinicalMetabolomicsFallbackConfig(environment, accuracy_enum)


def main():
    """Example usage of clinical metabolomics configuration."""
    
    print("Clinical Metabolomics Fallback Configuration")
    print("=" * 50)
    
    # Create configuration for production research use
    config = create_clinical_metabolomics_config("production", "research")
    
    # Validate configuration
    errors = config.validate_configuration()
    if errors:
        print(f"Configuration errors: {errors}")
        return
    
    print("✓ Configuration is valid")
    
    # Display key settings
    print(f"Environment: {config.environment}")
    print(f"Accuracy Level: {config.accuracy_level.value}")
    print(f"Research Confidence Threshold: {config.thresholds.research_confidence_threshold}")
    print(f"Primary Timeout: {config.timeouts.lightrag_primary_timeout}s")
    print(f"Max Failures: {config.reliability.max_consecutive_failures}")
    
    # Generate LightRAG configuration
    lightrag_config = config.get_lightrag_config()
    print(f"✓ LightRAG configuration generated")
    print(f"  Model: {lightrag_config.model}")
    print(f"  Max Tokens: {lightrag_config.max_tokens}")
    print(f"  Quality Threshold: {lightrag_config.lightrag_min_quality_threshold}")
    
    # Get fallback strategies
    strategies = config.get_fallback_strategies()
    print(f"✓ {len(strategies)} fallback strategies configured")
    
    # Get cache warming queries
    cache_queries = config.get_cache_warming_queries()
    print(f"✓ {len(cache_queries)} cache warming queries configured")
    
    print("\nClinical metabolomics configuration ready for deployment!")


if __name__ == "__main__":
    main()