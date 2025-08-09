"""
Clinical Metabolomics Oracle - Comprehensive Test Configuration System
====================================================================

This module provides comprehensive test scenarios for different concurrent user patterns
in the CMO system, building upon the enhanced CMO load framework. It includes realistic
user behavior patterns, clinical workflows, research scenarios, and component-specific
testing configurations.

Key Features:
1. Predefined test scenarios for clinical, research, and emergency workflows
2. Realistic user behavior patterns with authentic query patterns
3. Scalability validation scenarios (50-250+ concurrent users)
4. Component-specific testing (RAG, caching, circuit breakers, fallbacks)
5. Configuration factory methods for easy test setup
6. Advanced load patterns (sustained, burst, ramp-up, spike, realistic)
7. Comprehensive validation and performance targets

Scenario Categories:
- Clinical Workflows: Hospital rush, emergency department, clinic consultations
- Research Scenarios: Academic sessions, pharmaceutical research, literature reviews
- Scalability Testing: Gradual adoption, peak usage, stress testing
- Component Testing: RAG-intensive, cache effectiveness, circuit breaker, fallback

Author: Claude Code (Anthropic)
Version: 3.0.0
Created: 2025-08-09
Updated: 2025-08-09 (Complete rewrite for comprehensive scenarios)
Production Ready: Yes
"""

import asyncio
import logging
import json
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import uuid

# Import enhanced CMO framework components
from .enhanced_cmo_load_framework import (
    CMOTestConfiguration, CMOLoadMetrics, CMOLoadTestController,
    LoadPattern, UserBehavior, ComponentType
)

# Import base framework components
from .concurrent_load_framework import ConcurrentLoadMetrics


# ============================================================================
# ADVANCED USER BEHAVIOR PATTERNS
# ============================================================================

class ClinicalUserType(Enum):
    """Detailed clinical user types with specific behavior patterns."""
    EMERGENCY_PHYSICIAN = "emergency_physician"
    ICU_SPECIALIST = "icu_specialist"
    GENERAL_PRACTITIONER = "general_practitioner"
    CLINICAL_RESEARCHER = "clinical_researcher"
    MEDICAL_RESIDENT = "medical_resident"
    PHARMACY_SPECIALIST = "pharmacy_specialist"
    LABORATORY_TECHNICIAN = "laboratory_technician"


class ResearchUserType(Enum):
    """Research user types with specific query patterns."""
    ACADEMIC_RESEARCHER = "academic_researcher"
    PHARMACEUTICAL_RESEARCHER = "pharmaceutical_researcher"
    BIOINFORMATICS_SPECIALIST = "bioinformatics_specialist"
    GRADUATE_STUDENT = "graduate_student"
    POST_DOC_RESEARCHER = "post_doc_researcher"
    LITERATURE_REVIEWER = "literature_reviewer"
    DATA_SCIENTIST = "data_scientist"


class QueryComplexity(Enum):
    """Query complexity levels for realistic testing."""
    SIMPLE = "simple"           # Basic metabolite lookups
    MODERATE = "moderate"       # Pathway analysis
    COMPLEX = "complex"         # Multi-metabolite interactions
    ADVANCED = "advanced"       # Complex research queries


@dataclass
class UserBehaviorProfile:
    """Comprehensive user behavior profile for realistic simulation."""
    user_type: Union[ClinicalUserType, ResearchUserType]
    session_duration_minutes: Tuple[int, int]  # min, max
    queries_per_session: Tuple[int, int]       # min, max
    query_complexity_distribution: Dict[QueryComplexity, float]
    think_time_seconds: Tuple[float, float]    # min, max between queries
    peak_hours: List[int]                      # 0-23 hours when most active
    concurrent_likelihood: float               # 0-1, likelihood of concurrent sessions
    error_tolerance: float                     # 0-1, tolerance for system errors
    cache_affinity: float                      # 0-1, likelihood to query similar things
    
    def get_session_duration(self) -> int:
        """Get randomized session duration within profile range."""
        return random.randint(*self.session_duration_minutes)
    
    def get_queries_per_session(self) -> int:
        """Get randomized query count within profile range."""
        return random.randint(*self.queries_per_session)
    
    def get_think_time(self) -> float:
        """Get randomized think time within profile range."""
        return random.uniform(*self.think_time_seconds)
    
    def get_query_complexity(self) -> QueryComplexity:
        """Get weighted random query complexity based on profile."""
        complexities = list(self.query_complexity_distribution.keys())
        weights = list(self.query_complexity_distribution.values())
        return random.choices(complexities, weights=weights)[0]


# ============================================================================
# PREDEFINED USER BEHAVIOR PROFILES
# ============================================================================

def get_clinical_user_profiles() -> Dict[ClinicalUserType, UserBehaviorProfile]:
    """Get predefined clinical user behavior profiles."""
    return {
        ClinicalUserType.EMERGENCY_PHYSICIAN: UserBehaviorProfile(
            user_type=ClinicalUserType.EMERGENCY_PHYSICIAN,
            session_duration_minutes=(5, 20),
            queries_per_session=(1, 8),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.4,
                QueryComplexity.MODERATE: 0.4,
                QueryComplexity.COMPLEX: 0.2,
                QueryComplexity.ADVANCED: 0.0
            },
            think_time_seconds=(10, 60),
            peak_hours=[6, 7, 8, 14, 15, 16, 20, 21, 22, 23],  # Emergency peaks
            concurrent_likelihood=0.9,  # High concurrency in emergency
            error_tolerance=0.2,         # Low tolerance for errors
            cache_affinity=0.3           # Less likely to repeat queries
        ),
        
        ClinicalUserType.ICU_SPECIALIST: UserBehaviorProfile(
            user_type=ClinicalUserType.ICU_SPECIALIST,
            session_duration_minutes=(15, 45),
            queries_per_session=(3, 12),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.2,
                QueryComplexity.MODERATE: 0.5,
                QueryComplexity.COMPLEX: 0.3,
                QueryComplexity.ADVANCED: 0.0
            },
            think_time_seconds=(30, 120),
            peak_hours=[6, 7, 8, 9, 17, 18, 19, 20],
            concurrent_likelihood=0.7,
            error_tolerance=0.3,
            cache_affinity=0.4
        ),
        
        ClinicalUserType.GENERAL_PRACTITIONER: UserBehaviorProfile(
            user_type=ClinicalUserType.GENERAL_PRACTITIONER,
            session_duration_minutes=(10, 30),
            queries_per_session=(2, 10),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.6,
                QueryComplexity.MODERATE: 0.3,
                QueryComplexity.COMPLEX: 0.1,
                QueryComplexity.ADVANCED: 0.0
            },
            think_time_seconds=(20, 90),
            peak_hours=[8, 9, 10, 11, 14, 15, 16, 17],  # Clinic hours
            concurrent_likelihood=0.6,
            error_tolerance=0.4,
            cache_affinity=0.6
        ),
        
        ClinicalUserType.CLINICAL_RESEARCHER: UserBehaviorProfile(
            user_type=ClinicalUserType.CLINICAL_RESEARCHER,
            session_duration_minutes=(30, 90),
            queries_per_session=(5, 20),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.2,
                QueryComplexity.MODERATE: 0.3,
                QueryComplexity.COMPLEX: 0.3,
                QueryComplexity.ADVANCED: 0.2
            },
            think_time_seconds=(60, 300),
            peak_hours=[9, 10, 11, 13, 14, 15, 16],
            concurrent_likelihood=0.4,
            error_tolerance=0.6,
            cache_affinity=0.7
        ),
        
        ClinicalUserType.MEDICAL_RESIDENT: UserBehaviorProfile(
            user_type=ClinicalUserType.MEDICAL_RESIDENT,
            session_duration_minutes=(20, 60),
            queries_per_session=(4, 15),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.4,
                QueryComplexity.MODERATE: 0.4,
                QueryComplexity.COMPLEX: 0.2,
                QueryComplexity.ADVANCED: 0.0
            },
            think_time_seconds=(45, 180),
            peak_hours=[7, 8, 9, 10, 16, 17, 18, 19, 20],
            concurrent_likelihood=0.5,
            error_tolerance=0.5,
            cache_affinity=0.5
        )
    }


def get_research_user_profiles() -> Dict[ResearchUserType, UserBehaviorProfile]:
    """Get predefined research user behavior profiles."""
    return {
        ResearchUserType.ACADEMIC_RESEARCHER: UserBehaviorProfile(
            user_type=ResearchUserType.ACADEMIC_RESEARCHER,
            session_duration_minutes=(45, 180),
            queries_per_session=(8, 30),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.1,
                QueryComplexity.MODERATE: 0.3,
                QueryComplexity.COMPLEX: 0.4,
                QueryComplexity.ADVANCED: 0.2
            },
            think_time_seconds=(90, 600),
            peak_hours=[9, 10, 11, 13, 14, 15, 16, 19, 20],
            concurrent_likelihood=0.3,
            error_tolerance=0.7,
            cache_affinity=0.8
        ),
        
        ResearchUserType.PHARMACEUTICAL_RESEARCHER: UserBehaviorProfile(
            user_type=ResearchUserType.PHARMACEUTICAL_RESEARCHER,
            session_duration_minutes=(60, 240),
            queries_per_session=(10, 40),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.05,
                QueryComplexity.MODERATE: 0.25,
                QueryComplexity.COMPLEX: 0.45,
                QueryComplexity.ADVANCED: 0.25
            },
            think_time_seconds=(120, 900),
            peak_hours=[9, 10, 11, 13, 14, 15, 16],
            concurrent_likelihood=0.4,
            error_tolerance=0.8,
            cache_affinity=0.9
        ),
        
        ResearchUserType.BIOINFORMATICS_SPECIALIST: UserBehaviorProfile(
            user_type=ResearchUserType.BIOINFORMATICS_SPECIALIST,
            session_duration_minutes=(90, 300),
            queries_per_session=(15, 50),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.1,
                QueryComplexity.MODERATE: 0.2,
                QueryComplexity.COMPLEX: 0.3,
                QueryComplexity.ADVANCED: 0.4
            },
            think_time_seconds=(60, 480),
            peak_hours=[10, 11, 12, 14, 15, 16, 17, 20, 21],
            concurrent_likelihood=0.2,
            error_tolerance=0.9,
            cache_affinity=0.7
        ),
        
        ResearchUserType.GRADUATE_STUDENT: UserBehaviorProfile(
            user_type=ResearchUserType.GRADUATE_STUDENT,
            session_duration_minutes=(30, 120),
            queries_per_session=(6, 25),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.3,
                QueryComplexity.MODERATE: 0.4,
                QueryComplexity.COMPLEX: 0.25,
                QueryComplexity.ADVANCED: 0.05
            },
            think_time_seconds=(60, 300),
            peak_hours=[10, 11, 12, 14, 15, 16, 17, 19, 20, 21],
            concurrent_likelihood=0.6,
            error_tolerance=0.6,
            cache_affinity=0.6
        ),
        
        ResearchUserType.LITERATURE_REVIEWER: UserBehaviorProfile(
            user_type=ResearchUserType.LITERATURE_REVIEWER,
            session_duration_minutes=(60, 240),
            queries_per_session=(20, 80),
            query_complexity_distribution={
                QueryComplexity.SIMPLE: 0.4,
                QueryComplexity.MODERATE: 0.4,
                QueryComplexity.COMPLEX: 0.15,
                QueryComplexity.ADVANCED: 0.05
            },
            think_time_seconds=(30, 180),
            peak_hours=[9, 10, 11, 13, 14, 15, 16, 17],
            concurrent_likelihood=0.3,
            error_tolerance=0.7,
            cache_affinity=0.9  # Very high cache affinity for literature reviews
        )
    }


# ============================================================================
# COMPREHENSIVE TEST SCENARIO DEFINITIONS
# ============================================================================

@dataclass
class CMOTestScenario:
    """Comprehensive test scenario definition with detailed configuration."""
    scenario_id: str
    name: str
    description: str
    category: str
    user_profiles: List[UserBehaviorProfile]
    user_distribution: List[float]  # Weights for user profile distribution
    load_pattern: LoadPattern
    concurrent_users: int
    total_operations: int
    test_duration_seconds: int
    performance_targets: Dict[str, Any]
    component_focus: List[ComponentType]
    special_conditions: Dict[str, Any] = field(default_factory=dict)
    
    def to_cmo_test_configuration(self) -> CMOTestConfiguration:
        """Convert scenario to CMOTestConfiguration."""
        # Map user profiles to UserBehavior enum values
        behavior_mapping = {
            ClinicalUserType.EMERGENCY_PHYSICIAN: UserBehavior.EMERGENCY,
            ClinicalUserType.ICU_SPECIALIST: UserBehavior.CLINICIAN,
            ClinicalUserType.GENERAL_PRACTITIONER: UserBehavior.CLINICIAN,
            ClinicalUserType.CLINICAL_RESEARCHER: UserBehavior.RESEARCHER,
            ClinicalUserType.MEDICAL_RESIDENT: UserBehavior.STUDENT,
            ResearchUserType.ACADEMIC_RESEARCHER: UserBehavior.RESEARCHER,
            ResearchUserType.PHARMACEUTICAL_RESEARCHER: UserBehavior.RESEARCHER,
            ResearchUserType.BIOINFORMATICS_SPECIALIST: UserBehavior.RESEARCHER,
            ResearchUserType.GRADUATE_STUDENT: UserBehavior.STUDENT,
            ResearchUserType.LITERATURE_REVIEWER: UserBehavior.RESEARCHER
        }
        
        user_behaviors = []
        behavior_weights = []
        for profile, weight in zip(self.user_profiles, self.user_distribution):
            if profile.user_type in behavior_mapping:
                behavior = behavior_mapping[profile.user_type]
                if behavior not in user_behaviors:
                    user_behaviors.append(behavior)
                    behavior_weights.append(weight)
                else:
                    # Add weight to existing behavior
                    idx = user_behaviors.index(behavior)
                    behavior_weights[idx] += weight
        
        # Normalize weights
        total_weight = sum(behavior_weights)
        if total_weight > 0:
            behavior_weights = [w / total_weight for w in behavior_weights]
        
        # Determine primary component type
        if len(self.component_focus) == 1:
            component_type = self.component_focus[0]
        else:
            component_type = ComponentType.FULL_SYSTEM
        
        return CMOTestConfiguration(
            test_name=self.scenario_id,
            component_type=component_type,
            load_pattern=self.load_pattern,
            concurrent_users=self.concurrent_users,
            total_operations=self.total_operations,
            test_duration_seconds=self.test_duration_seconds,
            user_behaviors=user_behaviors,
            behavior_weights=behavior_weights,
            
            # Performance targets
            target_success_rate=self.performance_targets.get('success_rate', 0.95),
            target_p95_response_ms=self.performance_targets.get('p95_response_ms', 2000),
            max_memory_growth_mb=self.performance_targets.get('max_memory_growth_mb', 150),
            
            # LightRAG specific settings
            lightrag_enabled=self.special_conditions.get('lightrag_enabled', True),
            lightrag_mode=self.special_conditions.get('lightrag_mode', 'hybrid'),
            lightrag_response_type=self.special_conditions.get('lightrag_response_type', 'Multiple Paragraphs'),
            lightrag_timeout_seconds=self.special_conditions.get('lightrag_timeout_seconds', 30),
            
            # Circuit breaker settings
            circuit_breaker_enabled=self.special_conditions.get('circuit_breaker_enabled', True),
            circuit_breaker_failure_threshold=self.special_conditions.get('circuit_breaker_failure_threshold', 10),
            
            # Cache settings
            enable_l1_cache=self.special_conditions.get('enable_l1_cache', True),
            enable_l2_cache=self.special_conditions.get('enable_l2_cache', True),
            enable_l3_cache=self.special_conditions.get('enable_l3_cache', True),
            
            # Fallback settings
            enable_perplexity_fallback=self.special_conditions.get('enable_perplexity_fallback', True),
            enable_cache_fallback=self.special_conditions.get('enable_cache_fallback', True),
            
            # Ramp-up settings
            ramp_up_duration=self.special_conditions.get('ramp_up_duration', 30)
        )
    
    def validate_configuration(self) -> List[str]:
        """Validate scenario configuration and return any issues."""
        issues = []
        
        if len(self.user_profiles) != len(self.user_distribution):
            issues.append("User profiles and distribution lengths don't match")
        
        if abs(sum(self.user_distribution) - 1.0) > 0.01:
            issues.append(f"User distribution doesn't sum to 1.0: {sum(self.user_distribution)}")
        
        if self.concurrent_users <= 0:
            issues.append("Concurrent users must be positive")
        
        if self.total_operations <= 0:
            issues.append("Total operations must be positive")
        
        if self.test_duration_seconds <= 0:
            issues.append("Test duration must be positive")
        
        return issues


# ============================================================================
# CLINICAL WORKFLOW SCENARIOS
# ============================================================================

def create_clinical_scenarios() -> List[CMOTestScenario]:
    """Create comprehensive clinical workflow test scenarios."""
    clinical_profiles = get_clinical_user_profiles()
    
    return [
        # Hospital Morning Rush - Mixed clinical users with burst pattern
        CMOTestScenario(
            scenario_id="clinical_morning_rush_60",
            name="Hospital Morning Rush Load",
            description="Simulates morning hospital rush with mixed clinical users accessing CMO for patient care decisions",
            category="clinical_workflows",
            user_profiles=[
                clinical_profiles[ClinicalUserType.GENERAL_PRACTITIONER],
                clinical_profiles[ClinicalUserType.MEDICAL_RESIDENT],
                clinical_profiles[ClinicalUserType.ICU_SPECIALIST],
                clinical_profiles[ClinicalUserType.EMERGENCY_PHYSICIAN]
            ],
            user_distribution=[0.4, 0.3, 0.2, 0.1],
            load_pattern=LoadPattern.BURST,
            concurrent_users=60,
            total_operations=300,
            test_duration_seconds=900,  # 15 minutes
            performance_targets={
                'success_rate': 0.96,
                'p95_response_ms': 1500,  # Fast response needed for clinical decisions
                'max_memory_growth_mb': 120,
                'cache_hit_rate': 0.75
            },
            component_focus=[ComponentType.FULL_SYSTEM],
            special_conditions={
                'lightrag_timeout_seconds': 20,  # Shorter timeout for clinical urgency
                'circuit_breaker_failure_threshold': 8,
                'ramp_up_duration': 180,  # 3-minute ramp-up for morning rush
                'emergency_user_percentage': 0.15
            }
        ),
        
        # Emergency Department Load - High urgency, spike pattern
        CMOTestScenario(
            scenario_id="emergency_department_spike_40",
            name="Emergency Department Spike Load",
            description="Emergency department mass casualty event simulation with urgent CMO queries",
            category="clinical_workflows",
            user_profiles=[
                clinical_profiles[ClinicalUserType.EMERGENCY_PHYSICIAN],
                clinical_profiles[ClinicalUserType.ICU_SPECIALIST],
                clinical_profiles[ClinicalUserType.MEDICAL_RESIDENT]
            ],
            user_distribution=[0.6, 0.25, 0.15],
            load_pattern=LoadPattern.SPIKE,
            concurrent_users=40,
            total_operations=200,
            test_duration_seconds=600,  # 10 minutes - emergency duration
            performance_targets={
                'success_rate': 0.98,  # Very high success rate needed
                'p95_response_ms': 1200,  # Very fast response for emergency
                'max_memory_growth_mb': 100,
                'lightrag_success_rate': 0.97
            },
            component_focus=[ComponentType.RAG_QUERY, ComponentType.CIRCUIT_BREAKER],
            special_conditions={
                'lightrag_timeout_seconds': 15,  # Very short timeout
                'circuit_breaker_failure_threshold': 5,
                'enable_burst_detection': True,
                'emergency_user_percentage': 0.8,
                'clinician_response_urgency': 'high'
            }
        ),
        
        # Clinic Consultation Hours - Sustained realistic pattern
        CMOTestScenario(
            scenario_id="clinic_consultations_80",
            name="Clinic Consultation Hours",
            description="Regular clinic consultation hours with steady CMO usage for patient consultations",
            category="clinical_workflows",
            user_profiles=[
                clinical_profiles[ClinicalUserType.GENERAL_PRACTITIONER],
                clinical_profiles[ClinicalUserType.CLINICAL_RESEARCHER],
                clinical_profiles[ClinicalUserType.MEDICAL_RESIDENT]
            ],
            user_distribution=[0.6, 0.2, 0.2],
            load_pattern=LoadPattern.REALISTIC,
            concurrent_users=80,
            total_operations=480,  # 6 queries per user average
            test_duration_seconds=1800,  # 30 minutes
            performance_targets={
                'success_rate': 0.95,
                'p95_response_ms': 2000,
                'max_memory_growth_mb': 140,
                'cache_hit_rate': 0.80  # High cache hit rate expected in clinic settings
            },
            component_focus=[ComponentType.FULL_SYSTEM, ComponentType.CACHING],
            special_conditions={
                'cache_prewarming_enabled': True,
                'enable_clinical_workflows': True,
                'clinician_response_urgency': 'medium'
            }
        )
    ]


# ============================================================================
# RESEARCH SCENARIO DEFINITIONS
# ============================================================================

def create_research_scenarios() -> List[CMOTestScenario]:
    """Create comprehensive research workflow test scenarios."""
    research_profiles = get_research_user_profiles()
    
    return [
        # Academic Research Sessions - Sustained deep research
        CMOTestScenario(
            scenario_id="academic_research_sustained_50",
            name="Academic Research Sessions",
            description="Academic researchers conducting sustained deep research sessions with complex queries",
            category="research_scenarios",
            user_profiles=[
                research_profiles[ResearchUserType.ACADEMIC_RESEARCHER],
                research_profiles[ResearchUserType.GRADUATE_STUDENT],
                research_profiles[ResearchUserType.POST_DOC_RESEARCHER]
            ],
            user_distribution=[0.5, 0.3, 0.2],
            load_pattern=LoadPattern.SUSTAINED,
            concurrent_users=50,
            total_operations=400,  # 8 queries per user average
            test_duration_seconds=2400,  # 40 minutes for deep research
            performance_targets={
                'success_rate': 0.93,
                'p95_response_ms': 3000,  # Longer acceptable for research
                'max_memory_growth_mb': 180,
                'lightrag_success_rate': 0.94,
                'cache_hit_rate': 0.85
            },
            component_focus=[ComponentType.RAG_QUERY, ComponentType.FULL_SYSTEM],
            special_conditions={
                'lightrag_mode': 'global',  # Use global mode for comprehensive research
                'lightrag_response_type': 'Multiple Paragraphs',
                'lightrag_timeout_seconds': 45,
                'researcher_session_complexity': 'high',
                'enable_real_time_monitoring': True
            }
        ),
        
        # Pharmaceutical Research - Complex high-value queries
        CMOTestScenario(
            scenario_id="pharmaceutical_research_complex_30",
            name="Pharmaceutical Research Complex Queries",
            description="Pharmaceutical researchers with highly complex, high-value metabolomics queries",
            category="research_scenarios",
            user_profiles=[
                research_profiles[ResearchUserType.PHARMACEUTICAL_RESEARCHER],
                research_profiles[ResearchUserType.BIOINFORMATICS_SPECIALIST]
            ],
            user_distribution=[0.7, 0.3],
            load_pattern=LoadPattern.SUSTAINED,
            concurrent_users=30,
            total_operations=300,  # 10 queries per user - complex queries
            test_duration_seconds=3600,  # 1 hour for complex research
            performance_targets={
                'success_rate': 0.90,  # Lower due to complexity
                'p95_response_ms': 4000,  # Longer for complex processing
                'max_memory_growth_mb': 220,
                'lightrag_success_rate': 0.92,
                'max_cost_per_query': 0.15
            },
            component_focus=[ComponentType.RAG_QUERY, ComponentType.CIRCUIT_BREAKER],
            special_conditions={
                'lightrag_mode': 'hybrid',
                'lightrag_max_tokens': 6000,  # Allow longer responses
                'lightrag_temperature': 0.3,  # More precise for research
                'lightrag_cost_threshold': 8.0,  # Higher cost threshold
                'circuit_breaker_cost_threshold': 10.0,
                'researcher_session_complexity': 'high'
            }
        ),
        
        # Literature Review Intensive - High query volume, burst pattern
        CMOTestScenario(
            scenario_id="literature_review_burst_40",
            name="Literature Review Intensive",
            description="Literature reviewers conducting intensive review sessions with high query volumes",
            category="research_scenarios",
            user_profiles=[
                research_profiles[ResearchUserType.LITERATURE_REVIEWER],
                research_profiles[ResearchUserType.GRADUATE_STUDENT],
                research_profiles[ResearchUserType.ACADEMIC_RESEARCHER]
            ],
            user_distribution=[0.6, 0.25, 0.15],
            load_pattern=LoadPattern.BURST,
            concurrent_users=40,
            total_operations=800,  # 20 queries per user - high volume
            test_duration_seconds=1800,  # 30 minutes
            performance_targets={
                'success_rate': 0.94,
                'p95_response_ms': 2500,
                'max_memory_growth_mb': 160,
                'cache_hit_rate': 0.90  # Very high cache hit rate expected
            },
            component_focus=[ComponentType.CACHING, ComponentType.FULL_SYSTEM],
            special_conditions={
                'cache_prewarming_enabled': True,
                'l1_cache_size': 15000,  # Larger cache for literature review
                'l2_cache_size': 75000,
                'enable_burst_detection': True,
                'cache_compression_enabled': True
            }
        )
    ]


# ============================================================================
# SCALABILITY VALIDATION SCENARIOS
# ============================================================================

def create_scalability_scenarios() -> List[CMOTestScenario]:
    """Create scalability validation test scenarios."""
    clinical_profiles = get_clinical_user_profiles()
    research_profiles = get_research_user_profiles()
    
    return [
        # Gradual Adoption Ramp - Realistic growth pattern
        CMOTestScenario(
            scenario_id="gradual_adoption_ramp_200",
            name="Gradual System Adoption Ramp",
            description="Gradual adoption ramp testing system scalability as user base grows",
            category="scalability_validation",
            user_profiles=[
                clinical_profiles[ClinicalUserType.GENERAL_PRACTITIONER],
                clinical_profiles[ClinicalUserType.CLINICAL_RESEARCHER],
                research_profiles[ResearchUserType.ACADEMIC_RESEARCHER],
                research_profiles[ResearchUserType.GRADUATE_STUDENT]
            ],
            user_distribution=[0.4, 0.2, 0.25, 0.15],
            load_pattern=LoadPattern.RAMP_UP,
            concurrent_users=200,
            total_operations=800,
            test_duration_seconds=1800,  # 30 minutes
            performance_targets={
                'success_rate': 0.90,  # Allow some degradation at scale
                'p95_response_ms': 2500,
                'max_memory_growth_mb': 250,
                'target_memory_efficiency': 0.80
            },
            component_focus=[ComponentType.FULL_SYSTEM],
            special_conditions={
                'ramp_up_duration': 600,  # 10-minute ramp-up
                'enable_degradation_testing': True,
                'enable_recovery_testing': True,
                'max_fallback_attempts': 4
            }
        ),
        
        # Peak Usage Simulation - Maximum expected load
        CMOTestScenario(
            scenario_id="peak_usage_spike_200",
            name="Peak Usage Simulation",
            description="Maximum expected concurrent load simulation during peak usage periods",
            category="scalability_validation",
            user_profiles=[
                clinical_profiles[ClinicalUserType.EMERGENCY_PHYSICIAN],
                clinical_profiles[ClinicalUserType.GENERAL_PRACTITIONER],
                clinical_profiles[ClinicalUserType.MEDICAL_RESIDENT],
                research_profiles[ResearchUserType.ACADEMIC_RESEARCHER]
            ],
            user_distribution=[0.15, 0.35, 0.25, 0.25],
            load_pattern=LoadPattern.SPIKE,
            concurrent_users=200,
            total_operations=600,
            test_duration_seconds=900,  # 15 minutes
            performance_targets={
                'success_rate': 0.88,  # Allow degradation during peak
                'p95_response_ms': 3000,
                'max_memory_growth_mb': 300,
                'circuit_breaker_activation_rate': 0.05
            },
            component_focus=[ComponentType.FULL_SYSTEM, ComponentType.CIRCUIT_BREAKER],
            special_conditions={
                'circuit_breaker_failure_threshold': 15,
                'circuit_breaker_recovery_timeout': 30,
                'enable_burst_detection': True,
                'fallback_success_threshold': 0.75
            }
        ),
        
        # Stress Testing - Beyond normal capacity
        CMOTestScenario(
            scenario_id="stress_test_sustained_250",
            name="System Stress Testing",
            description="Stress testing beyond normal capacity to identify system breaking points",
            category="scalability_validation",
            user_profiles=[
                clinical_profiles[ClinicalUserType.GENERAL_PRACTITIONER],
                clinical_profiles[ClinicalUserType.MEDICAL_RESIDENT],
                research_profiles[ResearchUserType.GRADUATE_STUDENT],
                research_profiles[ResearchUserType.LITERATURE_REVIEWER]
            ],
            user_distribution=[0.3, 0.2, 0.3, 0.2],
            load_pattern=LoadPattern.SUSTAINED,
            concurrent_users=250,
            total_operations=750,
            test_duration_seconds=2400,  # 40 minutes
            performance_targets={
                'success_rate': 0.85,  # Expect significant degradation
                'p95_response_ms': 4000,
                'max_memory_growth_mb': 400,
                'system_stability_threshold': 0.70
            },
            component_focus=[ComponentType.FULL_SYSTEM],
            special_conditions={
                'enable_chaos_engineering': False,  # Start with controlled stress
                'circuit_breaker_failure_threshold': 20,
                'max_fallback_attempts': 5,
                'enable_performance_regression_detection': True
            }
        )
    ]


# ============================================================================
# COMPONENT-SPECIFIC TESTING SCENARIOS
# ============================================================================

def create_component_specific_scenarios() -> List[CMOTestScenario]:
    """Create component-specific testing scenarios."""
    research_profiles = get_research_user_profiles()
    clinical_profiles = get_clinical_user_profiles()
    
    return [
        # RAG-Intensive Testing - Focus on LightRAG performance
        CMOTestScenario(
            scenario_id="rag_intensive_sustained_60",
            name="RAG-Intensive Query Testing",
            description="Intensive testing focused on LightRAG performance with complex queries",
            category="component_testing",
            user_profiles=[
                research_profiles[ResearchUserType.BIOINFORMATICS_SPECIALIST],
                research_profiles[ResearchUserType.PHARMACEUTICAL_RESEARCHER],
                clinical_profiles[ClinicalUserType.CLINICAL_RESEARCHER]
            ],
            user_distribution=[0.4, 0.35, 0.25],
            load_pattern=LoadPattern.SUSTAINED,
            concurrent_users=60,
            total_operations=480,  # 8 queries per user - RAG intensive
            test_duration_seconds=1800,  # 30 minutes
            performance_targets={
                'success_rate': 0.92,
                'lightrag_success_rate': 0.94,
                'p95_response_ms': 3500,
                'lightrag_p95_response_ms': 3000,
                'max_cost_per_query': 0.12
            },
            component_focus=[ComponentType.RAG_QUERY],
            special_conditions={
                'lightrag_mode': 'hybrid',
                'lightrag_max_tokens': 5000,
                'lightrag_temperature': 0.4,
                'enable_cost_tracking': True,
                'researcher_session_complexity': 'high'
            }
        ),
        
        # Cache Effectiveness Testing - Multi-tier cache performance
        CMOTestScenario(
            scenario_id="cache_effectiveness_realistic_80",
            name="Cache Effectiveness Testing",
            description="Testing multi-tier cache effectiveness with realistic query patterns",
            category="component_testing",
            user_profiles=[
                clinical_profiles[ClinicalUserType.GENERAL_PRACTITIONER],
                research_profiles[ResearchUserType.LITERATURE_REVIEWER],
                research_profiles[ResearchUserType.GRADUATE_STUDENT]
            ],
            user_distribution=[0.4, 0.4, 0.2],
            load_pattern=LoadPattern.REALISTIC,
            concurrent_users=80,
            total_operations=640,  # 8 queries per user
            test_duration_seconds=1200,  # 20 minutes
            performance_targets={
                'success_rate': 0.95,
                'cache_hit_rate_l1': 0.85,
                'cache_hit_rate_l2': 0.70,
                'cache_hit_rate_l3': 0.60,
                'p95_response_ms': 1800
            },
            component_focus=[ComponentType.CACHING],
            special_conditions={
                'cache_prewarming_enabled': True,
                'l1_cache_size': 12000,
                'l2_cache_size': 60000,
                'l3_cache_size': 120000,
                'cache_compression_enabled': True
            }
        ),
        
        # Circuit Breaker Validation - Failure handling
        CMOTestScenario(
            scenario_id="circuit_breaker_spike_100",
            name="Circuit Breaker Validation",
            description="Testing circuit breaker effectiveness during system stress and failures",
            category="component_testing",
            user_profiles=[
                clinical_profiles[ClinicalUserType.EMERGENCY_PHYSICIAN],
                clinical_profiles[ClinicalUserType.ICU_SPECIALIST],
                research_profiles[ResearchUserType.PHARMACEUTICAL_RESEARCHER]
            ],
            user_distribution=[0.4, 0.3, 0.3],
            load_pattern=LoadPattern.SPIKE,
            concurrent_users=100,
            total_operations=400,
            test_duration_seconds=600,  # 10 minutes
            performance_targets={
                'success_rate': 0.85,  # Lower due to induced failures
                'circuit_breaker_activation_rate': 0.10,
                'circuit_breaker_recovery_rate': 0.90,
                'fallback_success_rate': 0.88
            },
            component_focus=[ComponentType.CIRCUIT_BREAKER],
            special_conditions={
                'circuit_breaker_failure_threshold': 8,
                'circuit_breaker_recovery_timeout': 45,
                'circuit_breaker_cost_threshold': 6.0,
                'enable_chaos_engineering': True  # Induce failures for testing
            }
        ),
        
        # Fallback System Testing - Comprehensive fallback validation
        CMOTestScenario(
            scenario_id="fallback_system_burst_70",
            name="Fallback System Testing",
            description="Comprehensive fallback system testing with LightRAG → Perplexity → Cache chain",
            category="component_testing",
            user_profiles=[
                clinical_profiles[ClinicalUserType.MEDICAL_RESIDENT],
                research_profiles[ResearchUserType.GRADUATE_STUDENT],
                clinical_profiles[ClinicalUserType.GENERAL_PRACTITIONER]
            ],
            user_distribution=[0.35, 0.35, 0.3],
            load_pattern=LoadPattern.BURST,
            concurrent_users=70,
            total_operations=350,
            test_duration_seconds=900,  # 15 minutes
            performance_targets={
                'success_rate': 0.90,
                'fallback_success_rate': 0.92,
                'perplexity_fallback_success_rate': 0.88,
                'cache_fallback_success_rate': 0.95,
                'fallback_response_time': 8000  # 8 seconds max for fallback chain
            },
            component_focus=[ComponentType.FALLBACK_SYSTEM],
            special_conditions={
                'enable_perplexity_fallback': True,
                'enable_cache_fallback': True,
                'enable_static_fallback': True,
                'fallback_timeout_seconds': 12,
                'max_fallback_attempts': 4,
                'lightrag_failure_simulation_rate': 0.2  # Simulate 20% LightRAG failures
            }
        )
    ]


# ============================================================================
# CONFIGURATION FACTORY AND MANAGEMENT
# ============================================================================

class CMOTestConfigurationFactory:
    """Factory class for creating and managing CMO test configurations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._scenarios_cache = {}
        self._load_all_scenarios()
    
    def _load_all_scenarios(self):
        """Load all predefined scenarios into cache."""
        self._scenarios_cache.update({
            'clinical': create_clinical_scenarios(),
            'research': create_research_scenarios(),
            'scalability': create_scalability_scenarios(),
            'component_testing': create_component_specific_scenarios()
        })
        
        self.logger.info(f"Loaded {sum(len(scenarios) for scenarios in self._scenarios_cache.values())} test scenarios")
    
    def get_scenario_by_id(self, scenario_id: str) -> Optional[CMOTestScenario]:
        """Get specific scenario by ID."""
        for category_scenarios in self._scenarios_cache.values():
            for scenario in category_scenarios:
                if scenario.scenario_id == scenario_id:
                    return scenario
        return None
    
    def get_scenarios_by_category(self, category: str) -> List[CMOTestScenario]:
        """Get all scenarios in a specific category."""
        return self._scenarios_cache.get(category, [])
    
    def get_all_scenarios(self) -> List[CMOTestScenario]:
        """Get all available scenarios."""
        all_scenarios = []
        for scenarios in self._scenarios_cache.values():
            all_scenarios.extend(scenarios)
        return all_scenarios
    
    def create_custom_scenario(self, 
                             scenario_id: str,
                             base_scenario_id: str = None,
                             overrides: Dict[str, Any] = None) -> CMOTestScenario:
        """Create custom scenario based on existing one with overrides."""
        if base_scenario_id:
            base_scenario = self.get_scenario_by_id(base_scenario_id)
            if not base_scenario:
                raise ValueError(f"Base scenario '{base_scenario_id}' not found")
        else:
            # Use default clinical scenario as base
            base_scenario = self.get_scenarios_by_category('clinical')[0]
        
        # Create copy of base scenario
        import copy
        custom_scenario = copy.deepcopy(base_scenario)
        custom_scenario.scenario_id = scenario_id
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(custom_scenario, key):
                    setattr(custom_scenario, key, value)
                else:
                    self.logger.warning(f"Unknown attribute '{key}' in scenario override")
        
        return custom_scenario
    
    def validate_all_scenarios(self) -> Dict[str, List[str]]:
        """Validate all scenarios and return validation results."""
        results = {}
        for category, scenarios in self._scenarios_cache.items():
            category_results = {}
            for scenario in scenarios:
                issues = scenario.validate_configuration()
                if issues:
                    category_results[scenario.scenario_id] = issues
            if category_results:
                results[category] = category_results
        return results
    
    def get_configuration_recommendations(self, 
                                        target_users: int,
                                        test_duration_minutes: int,
                                        primary_use_case: str) -> List[str]:
        """Get recommended scenario IDs based on requirements."""
        recommendations = []
        
        # Filter scenarios based on user count (within 20% tolerance)
        user_tolerance = 0.2
        suitable_scenarios = []
        
        for scenario in self.get_all_scenarios():
            user_diff = abs(scenario.concurrent_users - target_users) / target_users
            duration_diff = abs(scenario.test_duration_seconds/60 - test_duration_minutes) / test_duration_minutes
            
            if user_diff <= user_tolerance and duration_diff <= 0.5:  # 50% duration tolerance
                suitable_scenarios.append(scenario)
        
        # Prioritize by use case
        use_case_priorities = {
            'clinical': ['clinical', 'component_testing', 'scalability'],
            'research': ['research', 'component_testing', 'scalability'],
            'testing': ['component_testing', 'scalability', 'clinical'],
            'scalability': ['scalability', 'component_testing', 'clinical']
        }
        
        priority_order = use_case_priorities.get(primary_use_case.lower(), 
                                               ['clinical', 'research', 'component_testing', 'scalability'])
        
        # Sort scenarios by priority and suitability
        for priority_category in priority_order:
            category_scenarios = [s for s in suitable_scenarios if s.category == priority_category]
            recommendations.extend([s.scenario_id for s in category_scenarios])
        
        return recommendations[:5]  # Return top 5 recommendations


# ============================================================================
# COMPREHENSIVE TEST SUITE ORCHESTRATOR
# ============================================================================

class CMOTestSuiteOrchestrator:
    """Orchestrates comprehensive CMO test suite execution."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.factory = CMOTestConfigurationFactory()
        self.test_results = {}
        
    async def run_comprehensive_test_suite(self, 
                                         categories: List[str] = None,
                                         max_concurrent_suites: int = 2) -> Dict[str, Any]:
        """Run comprehensive test suite across specified categories."""
        if categories is None:
            categories = ['clinical', 'research', 'scalability', 'component_testing']
        
        self.logger.info(f"Starting comprehensive CMO test suite for categories: {categories}")
        
        # Validate scenarios first
        validation_results = self.factory.validate_all_scenarios()
        if validation_results:
            self.logger.warning(f"Scenario validation issues found: {validation_results}")
        
        # Collect all scenarios to run
        scenarios_to_run = []
        for category in categories:
            scenarios_to_run.extend(self.factory.get_scenarios_by_category(category))
        
        self.logger.info(f"Executing {len(scenarios_to_run)} test scenarios")
        
        # Run scenarios with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent_suites)
        tasks = []
        
        for scenario in scenarios_to_run:
            task = asyncio.create_task(self._run_single_scenario(scenario, semaphore))
            tasks.append(task)
        
        # Execute all scenarios
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = {}
        failed_results = {}
        
        for scenario, result in zip(scenarios_to_run, results):
            if isinstance(result, Exception):
                failed_results[scenario.scenario_id] = str(result)
                self.logger.error(f"Scenario {scenario.scenario_id} failed: {result}")
            else:
                successful_results[scenario.scenario_id] = result
        
        # Generate comprehensive analysis
        analysis = await self._generate_comprehensive_analysis(successful_results, failed_results)
        
        return {
            'execution_summary': {
                'total_scenarios': len(scenarios_to_run),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'categories_tested': categories
            },
            'successful_results': successful_results,
            'failed_results': failed_results,
            'comprehensive_analysis': analysis,
            'recommendations': self._generate_system_recommendations(successful_results)
        }
    
    async def _run_single_scenario(self, 
                                 scenario: CMOTestScenario, 
                                 semaphore: asyncio.Semaphore) -> CMOLoadMetrics:
        """Run a single test scenario with concurrency control."""
        async with semaphore:
            self.logger.info(f"Starting scenario: {scenario.scenario_id}")
            
            # Convert scenario to configuration
            config = scenario.to_cmo_test_configuration()
            
            # Create and run test controller
            controller = CMOLoadTestController(config)
            
            try:
                await controller.initialize_cmo_components()
                metrics = await controller.run_cmo_load_scenario(scenario.scenario_id, config)
                
                self.logger.info(f"Completed scenario: {scenario.scenario_id}")
                return metrics
                
            finally:
                await controller.cleanup_cmo_components()
                # Brief pause between scenarios to prevent resource conflicts
                await asyncio.sleep(5)
    
    async def _generate_comprehensive_analysis(self, 
                                             successful_results: Dict[str, Any],
                                             failed_results: Dict[str, str]) -> Dict[str, Any]:
        """Generate comprehensive analysis across all test results."""
        if not successful_results:
            return {'status': 'no_successful_tests', 'analysis': 'No successful test results to analyze'}
        
        # Aggregate metrics across all scenarios
        aggregated_metrics = {
            'success_rates': [],
            'response_times': [],
            'lightrag_success_rates': [],
            'cache_hit_rates': [],
            'memory_usage': [],
            'concurrent_peaks': []
        }
        
        category_performance = {}
        
        for scenario_id, metrics in successful_results.items():
            if isinstance(metrics, (CMOLoadMetrics, ConcurrentLoadMetrics)):
                # Aggregate basic metrics
                aggregated_metrics['success_rates'].append(metrics.get_success_rate())
                aggregated_metrics['response_times'].extend(metrics.response_times)
                aggregated_metrics['concurrent_peaks'].append(metrics.concurrent_peak)
                
                if metrics.memory_samples:
                    aggregated_metrics['memory_usage'].extend(metrics.memory_samples)
                
                # CMO-specific metrics if available
                if isinstance(metrics, CMOLoadMetrics):
                    aggregated_metrics['lightrag_success_rates'].append(metrics.get_lightrag_success_rate())
                    
                    # Cache analysis
                    cache_analysis = metrics.get_multi_tier_cache_analysis()
                    if cache_analysis.get('overall_metrics', {}).get('overall_hit_rate'):
                        aggregated_metrics['cache_hit_rates'].append(cache_analysis['overall_metrics']['overall_hit_rate'])
                
                # Category-wise performance
                scenario = self.factory.get_scenario_by_id(scenario_id)
                if scenario:
                    category = scenario.category
                    if category not in category_performance:
                        category_performance[category] = []
                    category_performance[category].append({
                        'scenario_id': scenario_id,
                        'success_rate': metrics.get_success_rate(),
                        'avg_response_time': statistics.mean(metrics.response_times) if metrics.response_times else 0,
                        'concurrent_peak': metrics.concurrent_peak
                    })
        
        # Calculate aggregate statistics
        analysis = {
            'overall_performance': {
                'avg_success_rate': statistics.mean(aggregated_metrics['success_rates']) if aggregated_metrics['success_rates'] else 0,
                'median_response_time': statistics.median(aggregated_metrics['response_times']) if aggregated_metrics['response_times'] else 0,
                'p95_response_time': statistics.quantiles(aggregated_metrics['response_times'], n=20)[18] if len(aggregated_metrics['response_times']) >= 20 else 0,
                'max_concurrent_users': max(aggregated_metrics['concurrent_peaks']) if aggregated_metrics['concurrent_peaks'] else 0,
                'avg_memory_usage': statistics.mean(aggregated_metrics['memory_usage']) if aggregated_metrics['memory_usage'] else 0
            },
            'cmo_specific_performance': {
                'avg_lightrag_success_rate': statistics.mean(aggregated_metrics['lightrag_success_rates']) if aggregated_metrics['lightrag_success_rates'] else 0,
                'avg_cache_hit_rate': statistics.mean(aggregated_metrics['cache_hit_rates']) if aggregated_metrics['cache_hit_rates'] else 0
            },
            'category_performance': category_performance,
            'system_stability': self._assess_system_stability(successful_results),
            'scalability_assessment': self._assess_scalability(successful_results)
        }
        
        return analysis
    
    def _assess_system_stability(self, results: Dict[str, Any]) -> str:
        """Assess overall system stability based on test results."""
        success_rates = []
        for metrics in results.values():
            if hasattr(metrics, 'get_success_rate'):
                success_rates.append(metrics.get_success_rate())
        
        if not success_rates:
            return "Unknown"
        
        avg_success_rate = statistics.mean(success_rates)
        success_variance = statistics.variance(success_rates) if len(success_rates) > 1 else 0
        
        if avg_success_rate >= 0.95 and success_variance < 0.01:
            return "Excellent"
        elif avg_success_rate >= 0.90 and success_variance < 0.02:
            return "Good"
        elif avg_success_rate >= 0.85:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_scalability(self, results: Dict[str, Any]) -> str:
        """Assess system scalability based on user load vs performance."""
        user_performance_pairs = []
        
        for scenario_id, metrics in results.items():
            scenario = self.factory.get_scenario_by_id(scenario_id)
            if scenario and hasattr(metrics, 'get_success_rate'):
                user_performance_pairs.append((
                    scenario.concurrent_users,
                    metrics.get_success_rate()
                ))
        
        if len(user_performance_pairs) < 2:
            return "Insufficient Data"
        
        # Sort by user count and check if performance degrades significantly
        user_performance_pairs.sort(key=lambda x: x[0])
        
        high_load_performance = [perf for users, perf in user_performance_pairs if users >= 100]
        low_load_performance = [perf for users, perf in user_performance_pairs if users <= 50]
        
        if not high_load_performance or not low_load_performance:
            return "Limited Load Range"
        
        avg_high_load = statistics.mean(high_load_performance)
        avg_low_load = statistics.mean(low_load_performance)
        
        performance_drop = avg_low_load - avg_high_load
        
        if performance_drop <= 0.05:  # Less than 5% drop
            return "Excellent"
        elif performance_drop <= 0.10:  # Less than 10% drop
            return "Good"
        elif performance_drop <= 0.20:  # Less than 20% drop
            return "Fair"
        else:
            return "Poor"
    
    def _generate_system_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate system-wide recommendations based on comprehensive test results."""
        recommendations = []
        
        # Analyze success rates
        success_rates = [metrics.get_success_rate() for metrics in results.values() if hasattr(metrics, 'get_success_rate')]
        if success_rates:
            avg_success_rate = statistics.mean(success_rates)
            if avg_success_rate < 0.90:
                recommendations.append(
                    f"Overall system success rate ({avg_success_rate:.1%}) needs improvement. "
                    "Consider enhancing error handling, implementing better circuit breakers, and optimizing resource allocation."
                )
        
        # Analyze response times
        all_response_times = []
        for metrics in results.values():
            if hasattr(metrics, 'response_times'):
                all_response_times.extend(metrics.response_times)
        
        if all_response_times:
            p95_response_time = statistics.quantiles(all_response_times, n=20)[18] if len(all_response_times) >= 20 else 0
            if p95_response_time > 3000:  # 3 seconds
                recommendations.append(
                    f"P95 response time ({p95_response_time:.0f}ms) is high. "
                    "Consider optimizing query processing, implementing better caching strategies, and reviewing system architecture."
                )
        
        # Analyze CMO-specific metrics
        lightrag_success_rates = []
        cache_hit_rates = []
        
        for metrics in results.values():
            if isinstance(metrics, CMOLoadMetrics):
                lightrag_success_rates.append(metrics.get_lightrag_success_rate())
                
                cache_analysis = metrics.get_multi_tier_cache_analysis()
                if cache_analysis.get('overall_metrics', {}).get('overall_hit_rate'):
                    cache_hit_rates.append(cache_analysis['overall_metrics']['overall_hit_rate'])
        
        if lightrag_success_rates:
            avg_lightrag_success = statistics.mean(lightrag_success_rates)
            if avg_lightrag_success < 0.92:
                recommendations.append(
                    f"LightRAG success rate ({avg_lightrag_success:.1%}) below target. "
                    "Review RAG configuration, optimize embeddings, and consider adjusting timeout settings."
                )
        
        if cache_hit_rates:
            avg_cache_hit_rate = statistics.mean(cache_hit_rates)
            if avg_cache_hit_rate < 0.75:
                recommendations.append(
                    f"Cache hit rate ({avg_cache_hit_rate:.1%}) below target. "
                    "Optimize cache sizes, review TTL settings, and implement better cache warming strategies."
                )
        
        # Scalability recommendations
        scalability_assessment = self._assess_scalability(results)
        if scalability_assessment in ['Fair', 'Poor']:
            recommendations.append(
                f"Scalability assessment: {scalability_assessment}. "
                "Consider implementing horizontal scaling, optimizing resource usage, and reviewing system architecture for bottlenecks."
            )
        
        return recommendations


# ============================================================================
# UTILITY FUNCTIONS AND HELPERS
# ============================================================================

async def run_scenario_by_id(scenario_id: str) -> Dict[str, Any]:
    """Run a specific scenario by ID."""
    factory = CMOTestConfigurationFactory()
    scenario = factory.get_scenario_by_id(scenario_id)
    
    if not scenario:
        raise ValueError(f"Scenario '{scenario_id}' not found")
    
    # Validate scenario
    issues = scenario.validate_configuration()
    if issues:
        raise ValueError(f"Scenario validation failed: {issues}")
    
    # Convert to configuration and run
    config = scenario.to_cmo_test_configuration()
    controller = CMOLoadTestController(config)
    
    try:
        await controller.initialize_cmo_components()
        return await controller.run_cmo_load_scenario(scenario_id, config)
    finally:
        await controller.cleanup_cmo_components()


def get_scenario_documentation() -> Dict[str, Any]:
    """Get comprehensive documentation for all available scenarios."""
    factory = CMOTestConfigurationFactory()
    documentation = {}
    
    for category in ['clinical', 'research', 'scalability', 'component_testing']:
        scenarios = factory.get_scenarios_by_category(category)
        category_doc = []
        
        for scenario in scenarios:
            category_doc.append({
                'id': scenario.scenario_id,
                'name': scenario.name,
                'description': scenario.description,
                'concurrent_users': scenario.concurrent_users,
                'duration_minutes': scenario.test_duration_seconds // 60,
                'load_pattern': scenario.load_pattern.value,
                'user_types': [profile.user_type.value for profile in scenario.user_profiles],
                'performance_targets': scenario.performance_targets,
                'component_focus': [comp.value for comp in scenario.component_focus]
            })
        
        documentation[category] = category_doc
    
    return documentation


# ============================================================================
# QUICK ACCESS FACTORY METHODS
# ============================================================================

def create_clinical_workflow_scenarios() -> Dict[str, CMOTestConfiguration]:
    """Create all clinical workflow scenarios as CMO configurations."""
    scenarios = create_clinical_scenarios()
    return {scenario.scenario_id: scenario.to_cmo_test_configuration() for scenario in scenarios}


def create_research_workflow_scenarios() -> Dict[str, CMOTestConfiguration]:
    """Create all research workflow scenarios as CMO configurations."""
    scenarios = create_research_scenarios()
    return {scenario.scenario_id: scenario.to_cmo_test_configuration() for scenario in scenarios}


def create_scalability_test_scenarios() -> Dict[str, CMOTestConfiguration]:
    """Create all scalability test scenarios as CMO configurations."""
    scenarios = create_scalability_scenarios()
    return {scenario.scenario_id: scenario.to_cmo_test_configuration() for scenario in scenarios}


def create_component_test_scenarios() -> Dict[str, CMOTestConfiguration]:
    """Create all component-specific test scenarios as CMO configurations."""
    scenarios = create_component_specific_scenarios()
    return {scenario.scenario_id: scenario.to_cmo_test_configuration() for scenario in scenarios}


def create_all_test_scenarios() -> Dict[str, CMOTestConfiguration]:
    """Create all available test scenarios as CMO configurations."""
    all_configs = {}
    all_configs.update(create_clinical_workflow_scenarios())
    all_configs.update(create_research_workflow_scenarios())
    all_configs.update(create_scalability_test_scenarios())
    all_configs.update(create_component_test_scenarios())
    return all_configs


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        # Create factory and validate scenarios
        factory = CMOTestConfigurationFactory()
        validation_results = factory.validate_all_scenarios()
        
        if validation_results:
            print("Scenario validation issues:")
            print(json.dumps(validation_results, indent=2))
        else:
            print("All scenarios validated successfully")
        
        # Get scenario recommendations
        recommendations = factory.get_configuration_recommendations(
            target_users=75,
            test_duration_minutes=15,
            primary_use_case='clinical'
        )
        
        print(f"\nRecommended scenarios for 75 users, 15 minutes, clinical use case:")
        for rec in recommendations:
            print(f"  - {rec}")
        
        # Print documentation
        docs = get_scenario_documentation()
        print(f"\nAvailable scenarios: {sum(len(scenarios) for scenarios in docs.values())} total")
        for category, scenarios in docs.items():
            print(f"\n{category.upper()} ({len(scenarios)} scenarios):")
            for scenario in scenarios:
                print(f"  {scenario['id']}: {scenario['name']} - {scenario['concurrent_users']} users")
    
    asyncio.run(main())