"""
Clinical Metabolomics Oracle LightRAG Integration Module

A comprehensive integration module that combines LightRAG (Light Retrieval-Augmented Generation)
with clinical metabolomics knowledge for enhanced research and diagnostic capabilities. This module
provides a complete suite of tools for cost tracking, quality validation, performance monitoring,
and intelligent query processing in metabolomics research contexts.

🔬 Core Features:
    • Advanced RAG system optimized for clinical metabolomics
    • Intelligent cost tracking and budget management
    • Comprehensive quality validation and accuracy assessment
    • Performance benchmarking and monitoring
    • PDF processing for biomedical literature
    • Research categorization and audit trails
    • Real-time progress tracking and reporting

📊 Key Components:
    • ClinicalMetabolomicsRAG: Main RAG system with enhanced capabilities
    • LightRAGConfig: Comprehensive configuration management
    • Quality Assessment Suite: Relevance scoring, factual validation, accuracy metrics
    • Performance Monitoring: Benchmarking utilities and correlation analysis
    • Cost Management: Budget tracking, persistence, and alerting
    • Document Processing: Specialized PDF handling for biomedical content

🚀 Quick Start:
    ```python
    from lightrag_integration import create_clinical_rag_system
    
    # Create a fully configured system
    rag = await create_clinical_rag_system(
        daily_budget_limit=50.0,
        enable_quality_validation=True
    )
    
    # Process a metabolomics query
    result = await rag.query(
        "What are the key metabolites in glucose metabolism?",
        mode="hybrid"
    )
    
    # Generate quality report
    report = await rag.generate_quality_report()
    ```

📈 Advanced Usage:
    ```python
    from lightrag_integration import (
        ClinicalMetabolomicsRAG, 
        LightRAGConfig,
        QualityReportGenerator,
        PerformanceBenchmarkSuite
    )
    
    # Custom configuration
    config = LightRAGConfig.from_file("config.json")
    rag = ClinicalMetabolomicsRAG(config)
    
    # Initialize with quality validation
    await rag.initialize_rag()
    
    # Run performance benchmarks
    benchmarks = PerformanceBenchmarkSuite(rag)
    results = await benchmarks.run_comprehensive_benchmarks()
    
    # Generate quality reports
    reporter = QualityReportGenerator(rag)
    await reporter.generate_comprehensive_report()
    ```

🔧 Environment Configuration:
    # Core Settings
    OPENAI_API_KEY=your_api_key_here
    LIGHTRAG_MODEL=gpt-4o-mini
    LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
    
    # Cost Management
    LIGHTRAG_ENABLE_COST_TRACKING=true
    LIGHTRAG_DAILY_BUDGET_LIMIT=50.0
    LIGHTRAG_MONTHLY_BUDGET_LIMIT=1000.0
    
    # Quality Validation
    LIGHTRAG_ENABLE_QUALITY_VALIDATION=true
    LIGHTRAG_RELEVANCE_THRESHOLD=0.75
    LIGHTRAG_ACCURACY_THRESHOLD=0.80
    
    # Performance Monitoring
    LIGHTRAG_ENABLE_PERFORMANCE_MONITORING=true
    LIGHTRAG_BENCHMARK_FREQUENCY=daily

📚 Module Organization:
    Core System: Main RAG integration and configuration
    Quality Suite: Validation, scoring, and accuracy assessment  
    Performance: Benchmarking, monitoring, and optimization
    Cost Management: Tracking, budgeting, and persistence
    Document Processing: PDF handling and content extraction
    Utilities: Helper functions and integration tools

Author: Claude Code (Anthropic) & SMO Chatbot Development Team
Created: August 6, 2025
Updated: August 8, 2025  
Version: 1.1.0
License: MIT
"""

# Version and metadata
__version__ = "1.1.0"
__author__ = "Claude Code (Anthropic) & SMO Chatbot Development Team"
__description__ = "Clinical Metabolomics Oracle LightRAG Integration Module"
__license__ = "MIT"
__status__ = "Production"

# =============================================================================
# CORE SYSTEM COMPONENTS
# =============================================================================

# Configuration Management
from .config import (
    LightRAGConfig,
    LightRAGConfigError,
    setup_lightrag_logging
)

# Main RAG System
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

# =============================================================================
# QUALITY VALIDATION SUITE  
# =============================================================================

# Relevance and Accuracy Assessment
try:
    from .relevance_scorer import (
        RelevanceScorer,
        RelevanceScore,
        RelevanceMetrics
    )
except ImportError:
    # Create stub classes for missing modules
    RelevanceScorer = RelevanceScore = RelevanceMetrics = None

try:
    from .accuracy_scorer import (
        AccuracyScorer,
        AccuracyScore,
        AccuracyMetrics
    )
except ImportError:
    AccuracyScorer = AccuracyScore = AccuracyMetrics = None

try:
    from .factual_accuracy_validator import (
        FactualAccuracyValidator,
        FactualValidationResult,
        ValidationMetrics
    )
except ImportError:
    FactualAccuracyValidator = FactualValidationResult = ValidationMetrics = None

# Claim Extraction and Validation
try:
    from .claim_extractor import (
        ClaimExtractor,
        ExtractedClaim,
        ClaimExtractionResult
    )
except ImportError:
    ClaimExtractor = ExtractedClaim = ClaimExtractionResult = None

# Quality Assessment and Reporting
try:
    from .enhanced_response_quality_assessor import (
        EnhancedResponseQualityAssessor,
        QualityAssessmentResult,
        QualityMetrics
    )
except ImportError:
    EnhancedResponseQualityAssessor = QualityAssessmentResult = QualityMetrics = None

try:
    from .quality_report_generator import (
        QualityReportGenerator,
        QualityReport,
        QualityTrend
    )
except ImportError:
    QualityReportGenerator = QualityReport = QualityTrend = None

# =============================================================================
# PERFORMANCE MONITORING & BENCHMARKING
# =============================================================================

# Performance Benchmarking
try:
    from .performance_benchmarking import (
        QualityValidationBenchmarkSuite,
        QualityValidationMetrics,
        QualityBenchmarkConfiguration,
        QualityPerformanceThreshold
    )
except ImportError:
    QualityValidationBenchmarkSuite = QualityValidationMetrics = None
    QualityBenchmarkConfiguration = QualityPerformanceThreshold = None

# Progress Tracking
try:
    from .unified_progress_tracker import (
        UnifiedProgressTracker,
        ProgressEvent,
        ProgressMetrics
    )
except ImportError:
    UnifiedProgressTracker = ProgressEvent = ProgressMetrics = None

try:
    from .progress_tracker import (
        ProgressTracker,
        ProgressReport
    )
except ImportError:
    ProgressTracker = ProgressReport = None

# =============================================================================
# COST MANAGEMENT & MONITORING
# =============================================================================

# Cost Persistence and Database
from .cost_persistence import (
    CostPersistence, 
    CostRecord, 
    ResearchCategory,
    CostDatabase
)

# Budget Management
from .budget_manager import (
    BudgetManager,
    BudgetThreshold,
    BudgetAlert,
    AlertLevel
)

# Real-time Monitoring
try:
    from .realtime_budget_monitor import (
        RealtimeBudgetMonitor,
        BudgetStatus,
        CostAlert
    )
except ImportError:
    RealtimeBudgetMonitor = BudgetStatus = CostAlert = None

# API Metrics and Usage Tracking
from .api_metrics_logger import (
    APIUsageMetricsLogger,
    APIMetric,
    MetricType,
    MetricsAggregator
)

# =============================================================================
# RESEARCH & CATEGORIZATION
# =============================================================================

from .research_categorizer import (
    ResearchCategorizer,
    CategoryPrediction,
    CategoryMetrics,
    QueryAnalyzer
)

# =============================================================================
# AUDIT & COMPLIANCE
# =============================================================================

from .audit_trail import (
    AuditTrail,
    AuditEvent,
    AuditEventType,
    ComplianceRule,
    ComplianceChecker
)

# =============================================================================
# DOCUMENT PROCESSING & INDEXING
# =============================================================================

from .pdf_processor import (
    BiomedicalPDFProcessor,
    BiomedicalPDFProcessorError
)

try:
    from .document_indexer import (
        DocumentIndexer,
        IndexedDocument,
        IndexingResult
    )
except ImportError:
    DocumentIndexer = IndexedDocument = IndexingResult = None

# =============================================================================
# RECOVERY & ERROR HANDLING
# =============================================================================

try:
    from .advanced_recovery_system import (
        AdvancedRecoverySystem,
        RecoveryStrategy,
        RecoveryResult
    )
except ImportError:
    AdvancedRecoverySystem = RecoveryStrategy = RecoveryResult = None

try:
    from .alert_system import (
        AlertSystem,
        Alert,
        AlertPriority
    )
except ImportError:
    AlertSystem = Alert = AlertPriority = None

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # =========================================================================
    # PACKAGE METADATA
    # =========================================================================
    "__version__",
    "__author__", 
    "__description__",
    "__license__",
    "__status__",
    
    # =========================================================================
    # CORE SYSTEM COMPONENTS
    # =========================================================================
    
    # Configuration Management
    "LightRAGConfig",
    "LightRAGConfigError", 
    "setup_lightrag_logging",
    
    # Main RAG System
    "ClinicalMetabolomicsRAG",
    "ClinicalMetabolomicsRAGError",
    "CostSummary",
    "QueryResponse",
    "CircuitBreaker",
    "CircuitBreakerError",
    "RateLimiter",
    "RequestQueue",
    "add_jitter",
    
    # =========================================================================
    # QUALITY VALIDATION SUITE
    # =========================================================================
    
    # Relevance and Accuracy Assessment
    "RelevanceScorer",
    "RelevanceScore",
    "RelevanceMetrics",
    "AccuracyScorer",
    "AccuracyScore",
    "AccuracyMetrics",
    "FactualAccuracyValidator",
    "FactualValidationResult",
    "ValidationMetrics",
    
    # Claim Extraction and Validation
    "ClaimExtractor",
    "ExtractedClaim", 
    "ClaimExtractionResult",
    
    # Quality Assessment and Reporting
    "EnhancedResponseQualityAssessor",
    "QualityAssessmentResult",
    "QualityMetrics",
    "QualityReportGenerator",
    "QualityReport",
    "QualityTrend",
    
    # =========================================================================
    # PERFORMANCE MONITORING & BENCHMARKING
    # =========================================================================
    
    # Performance Benchmarking
    "QualityValidationBenchmarkSuite",
    "QualityValidationMetrics",
    "QualityBenchmarkConfiguration",
    "QualityPerformanceThreshold",
    
    # Progress Tracking
    "UnifiedProgressTracker",
    "ProgressEvent",
    "ProgressMetrics",
    "ProgressTracker",
    "ProgressReport",
    
    # =========================================================================
    # COST MANAGEMENT & MONITORING
    # =========================================================================
    
    # Cost Persistence and Database
    "CostPersistence",
    "CostRecord",
    "ResearchCategory",
    "CostDatabase",
    
    # Budget Management
    "BudgetManager",
    "BudgetThreshold", 
    "BudgetAlert",
    "AlertLevel",
    "RealtimeBudgetMonitor",
    "BudgetStatus",
    "CostAlert",
    
    # API Metrics and Usage Tracking
    "APIUsageMetricsLogger",
    "APIMetric",
    "MetricType", 
    "MetricsAggregator",
    
    # =========================================================================
    # RESEARCH & CATEGORIZATION
    # =========================================================================
    
    "ResearchCategorizer",
    "CategoryPrediction",
    "CategoryMetrics",
    "QueryAnalyzer",
    
    # =========================================================================
    # AUDIT & COMPLIANCE
    # =========================================================================
    
    "AuditTrail",
    "AuditEvent",
    "AuditEventType",
    "ComplianceRule",
    "ComplianceChecker",
    
    # =========================================================================
    # DOCUMENT PROCESSING & INDEXING
    # =========================================================================
    
    "BiomedicalPDFProcessor",
    "BiomedicalPDFProcessorError",
    "DocumentIndexer",
    "IndexedDocument",
    "IndexingResult",
    
    # =========================================================================
    # RECOVERY & ERROR HANDLING
    # =========================================================================
    
    "AdvancedRecoverySystem",
    "RecoveryStrategy", 
    "RecoveryResult",
    "AlertSystem",
    "Alert",
    "AlertPriority",
    
    # =========================================================================
    # FACTORY FUNCTIONS & UTILITIES
    # =========================================================================
    
    "create_clinical_rag_system",
    "create_enhanced_rag_system",  # Backward compatibility
    "get_default_research_categories",
    "get_quality_validation_config",
    "create_performance_benchmark_suite",
    "get_integration_status",
    "validate_integration_setup",
]


# =============================================================================
# FACTORY FUNCTIONS & INTEGRATION UTILITIES
# =============================================================================

def create_clinical_rag_system(config_source=None, **config_overrides):
    """
    Primary factory function to create a fully configured Clinical Metabolomics RAG system.
    
    This function creates a complete RAG system optimized for clinical metabolomics research
    with all enhanced features enabled by default, including cost tracking, quality validation,
    performance monitoring, and comprehensive error handling.
    
    Args:
        config_source: Configuration source (None for env vars, path for file, dict for direct config)
        **config_overrides: Additional configuration overrides
        
    Returns:
        ClinicalMetabolomicsRAG: Fully configured RAG system with all enhanced features
        
    Key Features Enabled:
        • Cost tracking and budget management
        • Quality validation and accuracy assessment
        • Performance monitoring and benchmarking
        • Research categorization and audit trails
        • Advanced recovery and error handling
        • Progress tracking and reporting
        
    Examples:
        ```python
        # Basic usage with defaults
        rag = create_clinical_rag_system()
        await rag.initialize_rag()
        
        # Custom configuration with quality validation
        rag = create_clinical_rag_system(
            daily_budget_limit=50.0,
            monthly_budget_limit=1000.0,
            enable_quality_validation=True,
            relevance_threshold=0.75,
            accuracy_threshold=0.80
        )
        
        # From configuration file  
        rag = create_clinical_rag_system("clinical_config.json")
        
        # Research-specific configuration
        rag = create_clinical_rag_system(
            model="gpt-4o",
            enable_performance_monitoring=True,
            enable_factual_validation=True,
            cost_alert_threshold_percentage=85.0
        )
        ```
        
    Environment Variables:
        Core settings: OPENAI_API_KEY, LIGHTRAG_MODEL, LIGHTRAG_EMBEDDING_MODEL
        Cost management: LIGHTRAG_DAILY_BUDGET_LIMIT, LIGHTRAG_MONTHLY_BUDGET_LIMIT
        Quality validation: LIGHTRAG_RELEVANCE_THRESHOLD, LIGHTRAG_ACCURACY_THRESHOLD
        Performance: LIGHTRAG_ENABLE_PERFORMANCE_MONITORING, LIGHTRAG_BENCHMARK_FREQUENCY
    """
    
    # Set enhanced defaults for clinical metabolomics (only valid config parameters)
    defaults = {
        'enable_cost_tracking': True,
        'cost_persistence_enabled': True,
        'enable_research_categorization': True,
        'enable_audit_trail': True,
        'enable_relevance_scoring': True,
        'cost_alert_threshold_percentage': 80.0,
        'relevance_confidence_threshold': 0.75,
        'relevance_minimum_threshold': 0.60,
    }
    
    # Merge defaults with user overrides
    for key, value in defaults.items():
        config_overrides.setdefault(key, value)
    
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


def create_enhanced_rag_system(config_source=None, **config_overrides):
    """
    Legacy factory function for backward compatibility.
    
    This function is maintained for backward compatibility with existing code.
    For new implementations, prefer `create_clinical_rag_system()` which provides
    the same functionality with additional quality validation features.
    
    Args:
        config_source: Configuration source (None for env vars, path for file, dict for direct config)
        **config_overrides: Additional configuration overrides
        
    Returns:
        ClinicalMetabolomicsRAG: Configured RAG system with enhanced features
        
    Note:
        This function is deprecated. Use `create_clinical_rag_system()` instead.
    """
    
    import warnings
    warnings.warn(
        "create_enhanced_rag_system() is deprecated. Use create_clinical_rag_system() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return create_clinical_rag_system(config_source=config_source, **config_overrides)


def get_quality_validation_config(**overrides):
    """
    Create a configuration optimized for quality validation workflows.
    
    This function returns a configuration dictionary specifically tuned for
    quality validation tasks, including relevance scoring, accuracy assessment,
    and factual validation.
    
    Args:
        **overrides: Configuration parameter overrides
        
    Returns:
        dict: Configuration dictionary optimized for quality validation
        
    Example:
        ```python
        config = get_quality_validation_config(
            relevance_threshold=0.85,
            accuracy_threshold=0.90,
            enable_claim_extraction=True
        )
        rag = create_clinical_rag_system(**config)
        ```
    """
    
    quality_config = {
        'enable_relevance_scoring': True,
        'relevance_confidence_threshold': 0.80,
        'relevance_minimum_threshold': 0.70,
        'relevance_scoring_mode': 'comprehensive',  # Use valid mode
        'enable_parallel_relevance_processing': True,
        'model': 'gpt-4o',  # Use more capable model for quality tasks
        'enable_cost_tracking': True,
        'cost_persistence_enabled': True,
    }
    
    # Apply user overrides
    quality_config.update(overrides)
    
    return quality_config


def create_performance_benchmark_suite(rag_system=None, **config_overrides):
    """
    Create a performance benchmark suite for comprehensive testing.
    
    Args:
        rag_system: Optional existing RAG system to benchmark
        **config_overrides: Configuration overrides for the benchmark suite
        
    Returns:
        QualityValidationBenchmarkSuite: Configured benchmark suite
        
    Example:
        ```python
        # Create with existing RAG system
        rag = create_clinical_rag_system()
        benchmarks = create_performance_benchmark_suite(rag)
        
        # Create standalone benchmark suite
        benchmarks = create_performance_benchmark_suite(
            test_query_count=100,
            include_quality_metrics=True,
            benchmark_timeout=300
        )
        ```
    """
    
    # Default benchmark configuration
    benchmark_config = QualityBenchmarkConfiguration(
        test_query_count=50,
        include_quality_metrics=True,
        include_cost_metrics=True,
        include_performance_metrics=True,
        benchmark_timeout=180,
        parallel_execution=True,
        **config_overrides
    )
    
    # Create benchmark suite
    suite = QualityValidationBenchmarkSuite(
        rag_system=rag_system,
        config=benchmark_config
    )
    
    return suite


def get_default_research_categories():
    """
    Get the default research categories available for metabolomics research tracking.
    
    Returns a comprehensive list of research categories used for automatic categorization
    of metabolomics queries and cost tracking. Each category includes name, value,
    and detailed description.
    
    Returns:
        List[Dict[str, str]]: List of research category dictionaries
        
    Categories include:
        • Metabolite identification and characterization
        • Pathway analysis and network studies  
        • Biomarker discovery and validation
        • Drug discovery and pharmaceutical research
        • Clinical diagnosis and patient samples
        • Data processing and quality control
        • Statistical analysis and machine learning
        • Literature search and knowledge discovery
        • Database integration and cross-referencing
        • Experimental validation and protocols
        
    Example:
        ```python
        categories = get_default_research_categories()
        for category in categories:
            print(f"{category['name']}: {category['description']}")
        ```
    """
    categories = []
    for category in ResearchCategory:
        categories.append({
            'name': category.name,
            'value': category.value,
            'description': _get_category_description(category)
        })
    
    return categories


# =============================================================================
# INTEGRATION HELPERS & CONFIGURATION UTILITIES
# =============================================================================

def get_integration_status():
    """
    Get the current status of all integration components.
    
    Returns:
        Dict[str, Any]: Status information for all major components
    """
    import importlib
    import sys
    
    status = {
        'module_version': __version__,
        'python_version': sys.version,
        'components': {},
        'optional_features': {},
        'environment_config': {}
    }
    
    # Check core components
    core_components = [
        'config', 'clinical_metabolomics_rag', 'pdf_processor',
        'cost_persistence', 'budget_manager', 'research_categorizer'
    ]
    
    for component in core_components:
        try:
            module = importlib.import_module(f'.{component}', package='lightrag_integration')
            status['components'][component] = 'available'
        except ImportError as e:
            status['components'][component] = f'unavailable: {str(e)}'
    
    # Check optional features
    optional_features = [
        ('quality_report_generator', 'Quality Reporting'),
        ('relevance_scorer', 'Relevance Scoring'),
        ('factual_accuracy_validator', 'Factual Validation'),
        ('performance_benchmarking', 'Performance Benchmarking'),
        ('unified_progress_tracker', 'Progress Tracking')
    ]
    
    for module_name, feature_name in optional_features:
        try:
            importlib.import_module(f'.{module_name}', package='lightrag_integration')
            status['optional_features'][feature_name] = 'available'
        except ImportError:
            status['optional_features'][feature_name] = 'unavailable'
    
    # Check environment configuration
    import os
    env_vars = [
        'OPENAI_API_KEY', 'LIGHTRAG_MODEL', 'LIGHTRAG_WORKING_DIR',
        'LIGHTRAG_ENABLE_COST_TRACKING', 'LIGHTRAG_DAILY_BUDGET_LIMIT',
        'LIGHTRAG_ENABLE_QUALITY_VALIDATION'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask API keys for security
            if 'KEY' in var or 'TOKEN' in var:
                status['environment_config'][var] = f"{'*' * (len(value) - 4)}{value[-4:]}" if len(value) > 4 else "****"
            else:
                status['environment_config'][var] = value
        else:
            status['environment_config'][var] = None
    
    return status


def validate_integration_setup():
    """
    Validate that the integration is properly set up and configured.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
        
    Example:
        ```python
        is_valid, issues = validate_integration_setup()
        if not is_valid:
            for issue in issues:
                print(f"Setup issue: {issue}")
        ```
    """
    import importlib
    import os
    from pathlib import Path
    
    issues = []
    
    try:
        # Test configuration loading
        config = LightRAGConfig.get_config()
        if not config.api_key:
            issues.append("OPENAI_API_KEY environment variable is not set")
    except Exception as e:
        issues.append(f"Configuration validation failed: {str(e)}")
    
    # Check required directories
    required_dirs = ['working_dir', 'knowledge_base_dir', 'log_dir']
    
    try:
        config = LightRAGConfig.get_config()
        for dir_attr in required_dirs:
            if hasattr(config, dir_attr):
                dir_path = Path(getattr(config, dir_attr))
                if not dir_path.exists():
                    issues.append(f"Required directory does not exist: {dir_path}")
                elif not dir_path.is_dir():
                    issues.append(f"Path is not a directory: {dir_path}")
                elif not os.access(dir_path, os.W_OK):
                    issues.append(f"Directory is not writable: {dir_path}")
    except Exception as e:
        issues.append(f"Directory validation failed: {str(e)}")
    
    # Check optional dependencies
    optional_deps = [
        ('lightrag', 'LightRAG core functionality'),
        ('openai', 'OpenAI API integration'),
        ('aiohttp', 'Async HTTP operations'),
        ('tenacity', 'Retry mechanisms'),
    ]
    
    for dep_name, description in optional_deps:
        try:
            importlib.import_module(dep_name)
        except ImportError:
            issues.append(f"Optional dependency missing: {dep_name} ({description})")
    
    return len(issues) == 0, issues


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


# =============================================================================
# MODULE INITIALIZATION & LOGGING
# =============================================================================

# Import required modules for initialization
import importlib
import logging
import os

# Set up module-level logger
_logger = logging.getLogger(__name__)

try:
    # Initialize logging if not already configured
    if not _logger.handlers:
        # Try to use the setup_lightrag_logging function if available
        try:
            setup_lightrag_logging()
            _logger.info(f"Clinical Metabolomics Oracle LightRAG Integration v{__version__} initialized with enhanced logging")
        except Exception:
            # Fallback to basic logging configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            _logger.info(f"Clinical Metabolomics Oracle LightRAG Integration v{__version__} initialized with basic logging")
    
    # Log integration status
    _logger.debug("Checking integration component status...")
    status = get_integration_status()
    
    available_components = [name for name, status in status['components'].items() if status == 'available']
    _logger.info(f"Available components: {', '.join(available_components)}")
    
    available_features = [name for name, status in status['optional_features'].items() if status == 'available']
    if available_features:
        _logger.info(f"Available optional features: {', '.join(available_features)}")
    
    # Validate setup
    is_valid, issues = validate_integration_setup()
    if not is_valid:
        _logger.warning(f"Integration setup issues detected: {'; '.join(issues)}")
    else:
        _logger.info("Integration setup validation passed")
        
except Exception as e:
    # Ensure initialization doesn't fail completely if logging setup fails
    print(f"Warning: Failed to initialize integration module logging: {e}")
    print(f"Clinical Metabolomics Oracle LightRAG Integration v{__version__} initialized with minimal logging")

# Cleanup temporary variables
del importlib, logging, os