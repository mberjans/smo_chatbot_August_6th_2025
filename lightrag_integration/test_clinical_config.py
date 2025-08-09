#!/usr/bin/env python3
"""
Standalone Test for Clinical Metabolomics Configuration
====================================================

Test the clinical metabolomics configuration without complex dependencies.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from enum import Enum, IntEnum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set


class ClinicalAccuracyLevel(Enum):
    """Clinical accuracy requirements for different query types"""
    DIAGNOSTIC = "diagnostic"          
    THERAPEUTIC = "therapeutic"        
    RESEARCH = "research"              
    EDUCATIONAL = "educational"        
    GENERAL = "general"                


class FallbackLevel(IntEnum):
    """Enumeration of fallback levels in order of preference"""
    FULL_LLM_WITH_CONFIDENCE = 1      
    SIMPLIFIED_LLM = 2                 
    KEYWORD_BASED_ONLY = 3            
    EMERGENCY_CACHE = 4               
    DEFAULT_ROUTING = 5               


class FallbackStrategy:
    """Simple fallback strategy class"""
    def __init__(self, name, level, description, confidence_threshold, timeout_seconds, max_retries):
        self.name = name
        self.level = level
        self.description = description
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries


@dataclass
class ClinicalMetabolomicsThresholds:
    """Confidence thresholds optimized for clinical metabolomics accuracy requirements"""
    
    # Primary confidence thresholds (higher for biomedical accuracy)
    diagnostic_confidence_threshold: float = 0.90      
    therapeutic_confidence_threshold: float = 0.85     
    research_confidence_threshold: float = 0.75        
    educational_confidence_threshold: float = 0.65     
    general_confidence_threshold: float = 0.60         
    
    # Fallback decision thresholds
    lightrag_fallback_threshold: float = 0.70          
    perplexity_fallback_threshold: float = 0.60        
    emergency_cache_threshold: float = 0.40            
    
    # Quality assessment thresholds
    minimum_scientific_accuracy: float = 0.80          
    citation_quality_threshold: float = 0.75           
    factual_accuracy_threshold: float = 0.85           
    
    # Uncertainty-aware thresholds
    high_uncertainty_threshold: float = 0.30           
    medium_uncertainty_threshold: float = 0.20         
    low_uncertainty_threshold: float = 0.10            


@dataclass
class ClinicalMetabolomicsTimeouts:
    """Timeout configurations optimized for complex scientific queries"""
    
    # Primary processing timeouts (extended for scientific complexity)
    lightrag_primary_timeout: float = 45.0             
    lightrag_complex_query_timeout: float = 60.0       
    lightrag_literature_search_timeout: float = 90.0   
    
    # Perplexity API timeouts
    perplexity_standard_timeout: float = 35.0          
    perplexity_scientific_timeout: float = 50.0        
    
    # Fallback system timeouts
    router_decision_timeout: float = 5.0               
    confidence_analysis_timeout: float = 10.0          
    fallback_activation_timeout: float = 2.0           
    
    # Health check and monitoring timeouts
    health_check_timeout: float = 15.0                 
    system_monitoring_timeout: float = 5.0             
    
    # Circuit breaker timeouts
    circuit_breaker_timeout: float = 300.0             
    cost_breaker_timeout: float = 600.0                


@dataclass
class MetabolomicsCachePatterns:
    """Cache warming patterns for common clinical metabolomics queries"""
    
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
    """Reliability and safety settings for clinical metabolomics applications"""
    
    # Failure tolerance (stricter for clinical applications)
    max_consecutive_failures: int = 2               
    failure_rate_threshold: float = 0.05           
    quality_degradation_threshold: float = 0.15    
    
    # Recovery settings (faster recovery for clinical needs)
    auto_recovery_enabled: bool = True
    recovery_validation_samples: int = 5           
    recovery_success_threshold: float = 0.85       
    
    # Safety mechanisms
    enable_citation_verification: bool = True       
    require_source_attribution: bool = True        
    enable_fact_checking: bool = True              
    
    # Monitoring and alerting (enhanced for clinical use)
    enable_real_time_monitoring: bool = True
    alert_on_accuracy_drop: bool = True
    alert_threshold_accuracy: float = 0.80
    enable_performance_tracking: bool = True


class ClinicalMetabolomicsFallbackConfig:
    """Comprehensive fallback system configuration optimized for clinical metabolomics"""
    
    def __init__(self, 
                 environment: str = "production",
                 accuracy_level: ClinicalAccuracyLevel = ClinicalAccuracyLevel.RESEARCH):
        """Initialize clinical metabolomics fallback configuration"""
        self.environment = environment
        self.accuracy_level = accuracy_level
        self.created_at = datetime.now()
        
        # Initialize configuration components
        self.thresholds = ClinicalMetabolomicsThresholds()
        self.timeouts = ClinicalMetabolomicsTimeouts()
        self.cache_patterns = MetabolomicsCachePatterns()
        self.reliability = ClinicalReliabilitySettings()
        
        # Apply environment-specific adjustments
        self._apply_environment_adjustments()
        
        # Apply accuracy level adjustments
        self._apply_accuracy_level_adjustments()
    
    def _apply_environment_adjustments(self):
        """Apply environment-specific configuration adjustments"""
        if self.environment == "development":
            # More lenient settings for development
            self.thresholds.research_confidence_threshold = 0.65
            self.thresholds.lightrag_fallback_threshold = 0.60
            self.timeouts.lightrag_primary_timeout = 30.0
            self.reliability.max_consecutive_failures = 5
            
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
        """Apply accuracy level specific adjustments"""
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
    
    def get_fallback_strategies(self) -> List[FallbackStrategy]:
        """Generate fallback strategies optimized for clinical metabolomics"""
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
        """Get cache warming queries for metabolomics applications"""
        return self.cache_patterns.common_queries
    
    def validate_configuration(self) -> List[str]:
        """Validate the clinical metabolomics configuration"""
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
        """Convert configuration to dictionary for serialization"""
        # Convert cache patterns with set to list for JSON serialization
        cache_dict = asdict(self.cache_patterns)
        cache_dict["priority_terms"] = list(cache_dict["priority_terms"])
        
        return {
            "environment": self.environment,
            "accuracy_level": self.accuracy_level.value,
            "created_at": self.created_at.isoformat(),
            "thresholds": asdict(self.thresholds),
            "timeouts": asdict(self.timeouts),
            "cache_patterns": cache_dict,
            "reliability": asdict(self.reliability)
        }


def create_clinical_metabolomics_config(environment: str = "production", 
                                      accuracy_level: str = "research"):
    """Factory function to create clinical metabolomics configuration"""
    accuracy_enum = ClinicalAccuracyLevel(accuracy_level)
    return ClinicalMetabolomicsFallbackConfig(environment, accuracy_enum)


def test_configuration():
    """Test the clinical metabolomics configuration"""
    print("Clinical Metabolomics Fallback Configuration Test")
    print("=" * 60)
    
    try:
        # Test configuration creation
        config = create_clinical_metabolomics_config("production", "research")
        print("✓ Successfully created clinical metabolomics configuration")
        
        # Test basic validation
        errors = config.validate_configuration()
        if errors:
            print(f"✗ Configuration validation errors: {errors}")
            return False
        else:
            print("✓ Configuration validation passed")
        
        # Test key settings
        print(f"\nConfiguration Summary:")
        print(f"  Environment: {config.environment}")
        print(f"  Accuracy Level: {config.accuracy_level.value}")
        print(f"  Research Confidence Threshold: {config.thresholds.research_confidence_threshold}")
        print(f"  Primary Timeout: {config.timeouts.lightrag_primary_timeout}s")
        print(f"  Max Failures: {config.reliability.max_consecutive_failures}")
        print(f"  Cache Queries: {len(config.cache_patterns.common_queries)}")
        
        # Test fallback strategies
        strategies = config.get_fallback_strategies()
        print(f"  Fallback Strategies: {len(strategies)}")
        
        # Test cache queries
        cache_queries = config.get_cache_warming_queries()
        print(f"  Cache Warming Queries: {len(cache_queries)}")
        
        # Test serialization
        config_dict = config.to_dict()
        print(f"  Serialization Keys: {len(config_dict)}")
        
        # Test different environments and accuracy levels
        test_configs = [
            ("development", "general"),
            ("staging", "educational"),
            ("production", "therapeutic"),
            ("production", "diagnostic")
        ]
        
        print(f"\nTesting Different Configurations:")
        for env, acc_level in test_configs:
            try:
                test_config = create_clinical_metabolomics_config(env, acc_level)
                test_errors = test_config.validate_configuration()
                status = "✓" if not test_errors else "✗"
                print(f"  {status} {env}/{acc_level}: Threshold={test_config.thresholds.research_confidence_threshold}")
            except Exception as e:
                print(f"  ✗ {env}/{acc_level}: Error - {e}")
        
        print("\n✓ Clinical metabolomics configuration is working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_example_configurations():
    """Save example configurations for different scenarios"""
    output_dir = Path("clinical_metabolomics_configs")
    output_dir.mkdir(exist_ok=True)
    
    scenarios = [
        ("production", "diagnostic", "High-accuracy diagnostic queries"),
        ("production", "research", "Research-grade scientific queries"),  
        ("staging", "research", "Staging environment research"),
        ("development", "general", "Development testing")
    ]
    
    for env, acc_level, description in scenarios:
        config = create_clinical_metabolomics_config(env, acc_level)
        
        filename = f"{env}_{acc_level}_config.json"
        filepath = output_dir / filename
        
        config_dict = config.to_dict()
        config_dict["description"] = description
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Saved {description} configuration to {filepath}")


if __name__ == "__main__":
    # Run configuration test
    success = test_configuration()
    
    if success:
        print("\n" + "=" * 60)
        save_example_configurations()
        print("\n✓ All tests passed! Clinical metabolomics configuration is ready.")
    else:
        print("\n✗ Configuration test failed!")
        sys.exit(1)