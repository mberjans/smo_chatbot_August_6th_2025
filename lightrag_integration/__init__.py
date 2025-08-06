"""
Enhanced Cost Tracking System for Clinical Metabolomics Oracle LightRAG Integration

This package provides comprehensive cost tracking, budget management, research categorization,
and audit trail capabilities for the Clinical Metabolomics Oracle chatbot with LightRAG integration.

Key Components:
    - ClinicalMetabolomicsRAG: Main integration class with enhanced cost tracking
    - LightRAGConfig: Configuration management with cost tracking settings
    - CostPersistence: Database persistence for historical cost data
    - BudgetManager: Real-time budget monitoring with progressive alerts
    - ResearchCategorizer: Automatic categorization of metabolomics research queries
    - AuditTrail: Comprehensive audit logging and compliance monitoring

Enhanced Features:
    - Daily and monthly budget limits with configurable alerts
    - Research-specific cost categorization and analysis
    - Historical cost tracking with database persistence
    - Compliance monitoring and audit trails
    - Thread-safe operations for concurrent access
    - Configurable data retention policies
    - Real-time budget alerts and notifications

Usage:
    from lightrag_integration import ClinicalMetabolomicsRAG, LightRAGConfig
    
    # Basic usage with enhanced cost tracking
    config = LightRAGConfig.get_config()
    rag = ClinicalMetabolomicsRAG(config)
    await rag.initialize_rag()
    
    # Set budget limits
    rag.set_budget_limits(daily_limit=50.0, monthly_limit=1000.0)
    
    # Query with automatic cost tracking and categorization
    result = await rag.query("What metabolites are involved in glucose metabolism?")
    
    # Generate comprehensive cost report
    report = rag.generate_cost_report(days=30)

Environment Variables:
    # Enhanced Cost Tracking Configuration
    LIGHTRAG_ENABLE_COST_TRACKING=true
    LIGHTRAG_DAILY_BUDGET_LIMIT=50.0
    LIGHTRAG_MONTHLY_BUDGET_LIMIT=1000.0
    LIGHTRAG_COST_ALERT_THRESHOLD=80.0
    LIGHTRAG_ENABLE_BUDGET_ALERTS=true
    LIGHTRAG_COST_PERSISTENCE_ENABLED=true
    LIGHTRAG_COST_DB_PATH=cost_tracking.db
    LIGHTRAG_ENABLE_RESEARCH_CATEGORIZATION=true
    LIGHTRAG_ENABLE_AUDIT_TRAIL=true
    LIGHTRAG_COST_REPORT_FREQUENCY=daily
    LIGHTRAG_MAX_COST_RETENTION_DAYS=365

Author: Claude Code (Anthropic) & SMO Chatbot Development Team
Created: August 6, 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude Code (Anthropic) & SMO Chatbot Development Team"
__description__ = "Enhanced Cost Tracking System for Clinical Metabolomics Oracle LightRAG Integration"

# Core components
from .config import (
    LightRAGConfig,
    LightRAGConfigError,
    setup_lightrag_logging
)

from .clinical_metabolomics_rag import (
    ClinicalMetabolomicsRAG,
    ClinicalMetabolomicsRAGError,
    CostSummary,
    QueryResponse,
    CircuitBreaker,
    CircuitBreakerError,
    RateLimiter,
    RequestQueue,
    add_jitter
)

# Enhanced cost tracking components
from .cost_persistence import (
    CostPersistence, 
    CostRecord, 
    ResearchCategory,
    CostDatabase
)

from .budget_manager import (
    BudgetManager,
    BudgetThreshold,
    BudgetAlert,
    AlertLevel
)

from .research_categorizer import (
    ResearchCategorizer,
    CategoryPrediction,
    CategoryMetrics,
    QueryAnalyzer
)

from .audit_trail import (
    AuditTrail,
    AuditEvent,
    AuditEventType,
    ComplianceRule,
    ComplianceChecker
)

# Additional utility components
from .pdf_processor import (
    BiomedicalPDFProcessor,
    BiomedicalPDFProcessorError
)

# API usage metrics logging components
from .api_metrics_logger import (
    APIUsageMetricsLogger,
    APIMetric,
    MetricType,
    MetricsAggregator
)

# Public API
__all__ = [
    # Version and metadata
    "__version__",
    "__author__", 
    "__description__",
    
    # Core components
    "LightRAGConfig",
    "LightRAGConfigError", 
    "setup_lightrag_logging",
    "ClinicalMetabolomicsRAG",
    "ClinicalMetabolomicsRAGError",
    "CostSummary",
    "QueryResponse",
    "CircuitBreaker",
    "CircuitBreakerError",
    "RateLimiter",
    "RequestQueue",
    "add_jitter",
    
    # Cost persistence
    "CostPersistence",
    "CostRecord",
    "ResearchCategory",
    "CostDatabase",
    
    # Budget management
    "BudgetManager",
    "BudgetThreshold", 
    "BudgetAlert",
    "AlertLevel",
    
    # Research categorization
    "ResearchCategorizer",
    "CategoryPrediction",
    "CategoryMetrics",
    "QueryAnalyzer",
    
    # Audit and compliance
    "AuditTrail",
    "AuditEvent",
    "AuditEventType",
    "ComplianceRule",
    "ComplianceChecker",
    
    # Utilities
    "BiomedicalPDFProcessor",
    "BiomedicalPDFProcessorError",
    
    # API metrics logging
    "APIUsageMetricsLogger",
    "APIMetric",
    "MetricType", 
    "MetricsAggregator",
    
    # Factory functions
    "create_enhanced_rag_system",
    "get_default_research_categories"
]


def create_enhanced_rag_system(config_source=None, **config_overrides):
    """
    Factory function to create a fully configured ClinicalMetabolomicsRAG system
    with enhanced cost tracking enabled.
    
    Args:
        config_source: Configuration source (None for env vars, path for file, dict for direct config)
        **config_overrides: Additional configuration overrides
        
    Returns:
        ClinicalMetabolomicsRAG: Configured RAG system with enhanced features
        
    Example:
        # Create with default configuration from environment variables
        rag = create_enhanced_rag_system()
        
        # Create with custom budget limits
        rag = create_enhanced_rag_system(
            daily_budget_limit=25.0,
            monthly_budget_limit=500.0,
            cost_alert_threshold_percentage=75.0
        )
        
        # Create from configuration file
        rag = create_enhanced_rag_system("config.json")
    """
    
    # Ensure enhanced cost tracking is enabled
    config_overrides.setdefault('enable_cost_tracking', True)
    config_overrides.setdefault('cost_persistence_enabled', True)
    config_overrides.setdefault('enable_research_categorization', True)
    config_overrides.setdefault('enable_audit_trail', True)
    
    # Create configuration
    config = LightRAGConfig.get_config(
        source=config_source,
        validate_config=True,
        ensure_dirs=True,
        **config_overrides
    )
    
    # Create RAG system
    rag = ClinicalMetabolomicsRAG(config)
    
    return rag


def get_default_research_categories():
    """
    Get the default research categories available for metabolomics cost tracking.
    
    Returns:
        List of ResearchCategory enum values with descriptions
    """
    categories = []
    for category in ResearchCategory:
        categories.append({
            'name': category.name,
            'value': category.value,
            'description': _get_category_description(category)
        })
    
    return categories


def _get_category_description(category: ResearchCategory) -> str:
    """Get human-readable description for a research category."""
    descriptions = {
        ResearchCategory.METABOLITE_IDENTIFICATION: "Identification and characterization of metabolites using MS, NMR, and other analytical techniques",
        ResearchCategory.PATHWAY_ANALYSIS: "Analysis of metabolic pathways, networks, and biochemical processes",
        ResearchCategory.BIOMARKER_DISCOVERY: "Discovery and validation of metabolic biomarkers for disease diagnosis and monitoring",
        ResearchCategory.DRUG_DISCOVERY: "Drug development, mechanism of action studies, and pharmaceutical research",
        ResearchCategory.CLINICAL_DIAGNOSIS: "Clinical applications, patient samples, and diagnostic metabolomics",
        ResearchCategory.DATA_PREPROCESSING: "Data processing, quality control, normalization, and preprocessing workflows",
        ResearchCategory.STATISTICAL_ANALYSIS: "Statistical methods, multivariate analysis, and machine learning approaches",
        ResearchCategory.LITERATURE_SEARCH: "Literature review, research article analysis, and knowledge discovery",
        ResearchCategory.KNOWLEDGE_EXTRACTION: "Text mining, information extraction, and semantic analysis",
        ResearchCategory.DATABASE_INTEGRATION: "Database queries, cross-referencing, and data integration tasks",
        ResearchCategory.EXPERIMENTAL_VALIDATION: "Experimental design, validation studies, and laboratory protocols",
        ResearchCategory.GENERAL_QUERY: "General metabolomics questions and miscellaneous queries",
        ResearchCategory.SYSTEM_MAINTENANCE: "System operations, maintenance tasks, and administrative functions"
    }
    
    return descriptions.get(category, "No description available")


# Module initialization
import logging
_logger = logging.getLogger(__name__)
_logger.info(f"Enhanced Cost Tracking System v{__version__} initialized")