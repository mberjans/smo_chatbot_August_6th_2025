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
# FEATURE FLAG INITIALIZATION (Must be first)
# =============================================================================

# Environment-based feature detection - initialize before any conditional imports
_FEATURE_FLAGS = {}
_INTEGRATION_MODULES = {}
_FACTORY_FUNCTIONS = {}

def _load_feature_flags():
    """Load feature flags from environment variables."""
    import os
    
    flags = {
        # Core integration flags
        'lightrag_integration_enabled': os.getenv('LIGHTRAG_INTEGRATION_ENABLED', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'quality_validation_enabled': os.getenv('LIGHTRAG_ENABLE_QUALITY_VALIDATION', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        'performance_monitoring_enabled': os.getenv('LIGHTRAG_ENABLE_PERFORMANCE_MONITORING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'cost_tracking_enabled': os.getenv('LIGHTRAG_ENABLE_COST_TRACKING', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Quality validation sub-features
        'relevance_scoring_enabled': os.getenv('LIGHTRAG_ENABLE_RELEVANCE_SCORING', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        'accuracy_validation_enabled': os.getenv('LIGHTRAG_ENABLE_ACCURACY_VALIDATION', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'factual_validation_enabled': os.getenv('LIGHTRAG_ENABLE_FACTUAL_VALIDATION', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'claim_extraction_enabled': os.getenv('LIGHTRAG_ENABLE_CLAIM_EXTRACTION', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Performance and monitoring features
        'benchmarking_enabled': os.getenv('LIGHTRAG_ENABLE_BENCHMARKING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'progress_tracking_enabled': os.getenv('LIGHTRAG_ENABLE_PROGRESS_TRACKING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'unified_progress_tracking_enabled': os.getenv('LIGHTRAG_ENABLE_UNIFIED_PROGRESS_TRACKING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Document processing features
        'document_indexing_enabled': os.getenv('LIGHTRAG_ENABLE_DOCUMENT_INDEXING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'pdf_processing_enabled': os.getenv('LIGHTRAG_ENABLE_PDF_PROCESSING', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Advanced features
        'recovery_system_enabled': os.getenv('LIGHTRAG_ENABLE_RECOVERY_SYSTEM', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'alert_system_enabled': os.getenv('LIGHTRAG_ENABLE_ALERT_SYSTEM', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'budget_monitoring_enabled': os.getenv('LIGHTRAG_ENABLE_BUDGET_MONITORING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Integration control flags
        'circuit_breaker_enabled': os.getenv('LIGHTRAG_ENABLE_CIRCUIT_BREAKER', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        'ab_testing_enabled': os.getenv('LIGHTRAG_ENABLE_AB_TESTING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'conditional_routing_enabled': os.getenv('LIGHTRAG_ENABLE_CONDITIONAL_ROUTING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Debug and development flags
        'debug_mode_enabled': os.getenv('LIGHTRAG_DEBUG_MODE', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'development_features_enabled': os.getenv('LIGHTRAG_ENABLE_DEVELOPMENT_FEATURES', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
    }
    
    return flags

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific feature is enabled via feature flags."""
    return _FEATURE_FLAGS.get(feature_name, False)

def get_enabled_features() -> dict:
    """Get all enabled features and their status."""
    return {key: value for key, value in _FEATURE_FLAGS.items() if value}

def _register_integration_module(module_name: str, feature_flag: str, required: bool = False):
    """Register a module for conditional loading based on feature flags."""
    _INTEGRATION_MODULES[module_name] = {
        'feature_flag': feature_flag,
        'required': required,
        'loaded': False,
        'module': None,
        'exports': None
    }

def _check_integration_availability(module_name: str) -> bool:
    """Check if an integration module is available and enabled."""
    if module_name not in _INTEGRATION_MODULES:
        return False
    
    module_info = _INTEGRATION_MODULES[module_name]
    feature_flag = module_info['feature_flag']
    
    # Check feature flag
    if not _FEATURE_FLAGS.get(feature_flag, False):
        return False
    
    # Check if module can be imported
    if not module_info['loaded']:
        try:
            import importlib
            module = importlib.import_module(f'.{module_name}', package='lightrag_integration')
            module_info['module'] = module
            module_info['loaded'] = True
            return True
        except ImportError:
            return False
    
    return module_info['loaded']

def get_integration_status() -> dict:
    """Get comprehensive integration status including feature flags and module availability."""
    status = {
        'feature_flags': _FEATURE_FLAGS.copy(),
        'modules': {},
        'factory_functions': list(_FACTORY_FUNCTIONS.keys()),
        'integration_health': 'healthy'
    }
    
    # Check module status
    for module_name, module_info in _INTEGRATION_MODULES.items():
        status['modules'][module_name] = {
            'feature_flag': module_info['feature_flag'],
            'required': module_info['required'],
            'enabled': _FEATURE_FLAGS.get(module_info['feature_flag'], False),
            'available': _check_integration_availability(module_name),
            'loaded': module_info['loaded']
        }
    
    # Determine overall health
    required_modules_failed = [
        name for name, info in _INTEGRATION_MODULES.items() 
        if info['required'] and not _check_integration_availability(name)
    ]
    
    if required_modules_failed:
        status['integration_health'] = 'degraded'
        status['failed_required_modules'] = required_modules_failed
    
    return status

def validate_integration_setup() -> tuple[bool, list[str]]:
    """Validate integration setup and return status with any issues."""
    issues = []
    
    # Check core requirements
    if not _FEATURE_FLAGS.get('lightrag_integration_enabled', False):
        issues.append("LightRAG integration is disabled (LIGHTRAG_INTEGRATION_ENABLED=false)")
    
    # Check required modules
    required_modules = [name for name, info in _INTEGRATION_MODULES.items() if info['required']]
    for module_name in required_modules:
        if not _check_integration_availability(module_name):
            feature_flag = _INTEGRATION_MODULES[module_name]['feature_flag']
            if not _FEATURE_FLAGS.get(feature_flag, False):
                issues.append(f"Required module '{module_name}' is disabled by feature flag '{feature_flag}'")
            else:
                issues.append(f"Required module '{module_name}' cannot be imported")
    
    # Check environment configuration
    import os
    if not os.getenv('OPENAI_API_KEY'):
        issues.append("OPENAI_API_KEY environment variable is not set")
    
    # Check directory permissions
    try:
        config = LightRAGConfig.get_config()
        from pathlib import Path
        
        working_dir = Path(config.working_dir)
        if not working_dir.exists() or not os.access(working_dir, os.W_OK):
            issues.append(f"Working directory is not accessible or writable: {working_dir}")
            
    except Exception as e:
        issues.append(f"Configuration validation failed: {str(e)}")
    
    return len(issues) == 0, issues

# Initialize feature flags immediately
_FEATURE_FLAGS = _load_feature_flags()

# =============================================================================
# QUALITY VALIDATION SUITE  
# =============================================================================

# Conditional imports based on feature flags - Relevance and Accuracy Assessment
if is_feature_enabled('relevance_scoring_enabled'):
    try:
        from .relevance_scorer import (
            RelevanceScorer,
            RelevanceScore,
            RelevanceMetrics
        )
    except ImportError:
        # Create stub classes for missing modules
        RelevanceScorer = RelevanceScore = RelevanceMetrics = None
else:
    RelevanceScorer = RelevanceScore = RelevanceMetrics = None

if is_feature_enabled('accuracy_validation_enabled'):
    try:
        from .accuracy_scorer import (
            AccuracyScorer,
            AccuracyScore,
            AccuracyMetrics
        )
    except ImportError:
        AccuracyScorer = AccuracyScore = AccuracyMetrics = None
else:
    AccuracyScorer = AccuracyScore = AccuracyMetrics = None

if is_feature_enabled('factual_validation_enabled'):
    try:
        from .factual_accuracy_validator import (
            FactualAccuracyValidator,
            FactualValidationResult,
            ValidationMetrics
        )
    except ImportError:
        FactualAccuracyValidator = FactualValidationResult = ValidationMetrics = None
else:
    FactualAccuracyValidator = FactualValidationResult = ValidationMetrics = None

# Claim Extraction and Validation
if is_feature_enabled('claim_extraction_enabled'):
    try:
        from .claim_extractor import (
            ClaimExtractor,
            ExtractedClaim,
            ClaimExtractionResult
        )
    except ImportError:
        ClaimExtractor = ExtractedClaim = ClaimExtractionResult = None
else:
    ClaimExtractor = ExtractedClaim = ClaimExtractionResult = None

# Quality Assessment and Reporting
if is_feature_enabled('quality_validation_enabled'):
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
else:
    EnhancedResponseQualityAssessor = QualityAssessmentResult = QualityMetrics = None
    QualityReportGenerator = QualityReport = QualityTrend = None

# =============================================================================
# PERFORMANCE MONITORING & BENCHMARKING
# =============================================================================

# Performance Benchmarking
if is_feature_enabled('benchmarking_enabled'):
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
else:
    QualityValidationBenchmarkSuite = QualityValidationMetrics = None
    QualityBenchmarkConfiguration = QualityPerformanceThreshold = None

# Progress Tracking
if is_feature_enabled('unified_progress_tracking_enabled'):
    try:
        from .unified_progress_tracker import (
            UnifiedProgressTracker,
            ProgressEvent,
            ProgressMetrics
        )
    except ImportError:
        UnifiedProgressTracker = ProgressEvent = ProgressMetrics = None
else:
    UnifiedProgressTracker = ProgressEvent = ProgressMetrics = None

if is_feature_enabled('progress_tracking_enabled'):
    try:
        from .progress_tracker import (
            ProgressTracker,
            ProgressReport
        )
    except ImportError:
        ProgressTracker = ProgressReport = None
else:
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
if is_feature_enabled('budget_monitoring_enabled'):
    try:
        from .realtime_budget_monitor import (
            RealtimeBudgetMonitor,
            BudgetStatus,
            CostAlert
        )
    except ImportError:
        RealtimeBudgetMonitor = BudgetStatus = CostAlert = None
else:
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

if is_feature_enabled('document_indexing_enabled'):
    try:
        from .document_indexer import (
            DocumentIndexer,
            IndexedDocument,
            IndexingResult
        )
    except ImportError:
        DocumentIndexer = IndexedDocument = IndexingResult = None
else:
    DocumentIndexer = IndexedDocument = IndexingResult = None

# =============================================================================
# RECOVERY & ERROR HANDLING
# =============================================================================

if is_feature_enabled('recovery_system_enabled'):
    try:
        from .advanced_recovery_system import (
            AdvancedRecoverySystem,
            RecoveryStrategy,
            RecoveryResult
        )
    except ImportError:
        AdvancedRecoverySystem = RecoveryStrategy = RecoveryResult = None
else:
    AdvancedRecoverySystem = RecoveryStrategy = RecoveryResult = None

if is_feature_enabled('alert_system_enabled'):
    try:
        from .alert_system import (
            AlertSystem,
            Alert,
            AlertPriority
        )
    except ImportError:
        AlertSystem = Alert = AlertPriority = None
else:
    AlertSystem = Alert = AlertPriority = None

# =============================================================================
# DYNAMIC EXPORT MANAGEMENT
# =============================================================================

def _build_dynamic_exports():
    """Build dynamic __all__ list based on available features and modules."""
    exports = [
        # Package metadata (always available)
        "__version__", "__author__", "__description__", "__license__", "__status__",
        
        # Feature flag & integration management (always available)
        "is_feature_enabled", "get_enabled_features", "get_integration_status", "validate_integration_setup",
        
        # Core system components (always available)
        "LightRAGConfig", "LightRAGConfigError", "setup_lightrag_logging",
        "ClinicalMetabolomicsRAG", "ClinicalMetabolomicsRAGError", "CostSummary", "QueryResponse",
        "CircuitBreaker", "CircuitBreakerError", "RateLimiter", "RequestQueue", "add_jitter",
        
        # Cost management & monitoring (always available)
        "CostPersistence", "CostRecord", "ResearchCategory", "CostDatabase",
        "BudgetManager", "BudgetThreshold", "BudgetAlert", "AlertLevel",
        "APIUsageMetricsLogger", "APIMetric", "MetricType", "MetricsAggregator",
        
        # Research & categorization (always available)
        "ResearchCategorizer", "CategoryPrediction", "CategoryMetrics", "QueryAnalyzer",
        
        # Audit & compliance (always available)
        "AuditTrail", "AuditEvent", "AuditEventType", "ComplianceRule", "ComplianceChecker",
        
        # Document processing (PDF always available)
        "BiomedicalPDFProcessor", "BiomedicalPDFProcessorError",
        
        # Factory functions (always available)
        "create_clinical_rag_system", "create_clinical_rag_system_with_features",
        "create_enhanced_rag_system", "get_default_research_categories", "get_quality_validation_config",
    ]
    
    # Conditional exports based on feature availability
    conditional_exports = {
        # Quality validation suite
        'relevance_scoring_enabled': ["RelevanceScorer", "RelevanceScore", "RelevanceMetrics"],
        'accuracy_validation_enabled': ["AccuracyScorer", "AccuracyScore", "AccuracyMetrics"],
        'factual_validation_enabled': ["FactualAccuracyValidator", "FactualValidationResult", "ValidationMetrics"],
        'claim_extraction_enabled': ["ClaimExtractor", "ExtractedClaim", "ClaimExtractionResult"],
        'quality_validation_enabled': [
            "EnhancedResponseQualityAssessor", "QualityAssessmentResult", "QualityMetrics",
            "QualityReportGenerator", "QualityReport", "QualityTrend",
            "create_quality_validation_system"
        ],
        
        # Performance monitoring & benchmarking
        'benchmarking_enabled': [
            "QualityValidationBenchmarkSuite", "QualityValidationMetrics",
            "QualityBenchmarkConfiguration", "QualityPerformanceThreshold",
            "create_performance_benchmark_suite"
        ],
        'unified_progress_tracking_enabled': ["UnifiedProgressTracker", "ProgressEvent", "ProgressMetrics"],
        'progress_tracking_enabled': ["ProgressTracker", "ProgressReport"],
        'performance_monitoring_enabled': ["create_performance_monitoring_system"],
        
        # Cost management advanced features
        'budget_monitoring_enabled': ["RealtimeBudgetMonitor", "BudgetStatus", "CostAlert"],
        
        # Document processing
        'document_indexing_enabled': ["DocumentIndexer", "IndexedDocument", "IndexingResult"],
        
        # Recovery & error handling
        'recovery_system_enabled': ["AdvancedRecoverySystem", "RecoveryStrategy", "RecoveryResult"],
        'alert_system_enabled': ["AlertSystem", "Alert", "AlertPriority"],
    }
    
    # Add conditional exports based on enabled features
    for feature_flag, symbols in conditional_exports.items():
        if is_feature_enabled(feature_flag):
            # Only add symbols that actually exist (not None)
            for symbol in symbols:
                if symbol in globals() and globals()[symbol] is not None:
                    exports.append(symbol)
    
    return exports


# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

# Environment-based feature detection - initialize before building exports
_FEATURE_FLAGS = {}
_INTEGRATION_MODULES = {}
_FACTORY_FUNCTIONS = {}

def _load_feature_flags():
    """Load feature flags from environment variables."""
    import os
    
    flags = {
        # Core integration flags
        'lightrag_integration_enabled': os.getenv('LIGHTRAG_INTEGRATION_ENABLED', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'quality_validation_enabled': os.getenv('LIGHTRAG_ENABLE_QUALITY_VALIDATION', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        'performance_monitoring_enabled': os.getenv('LIGHTRAG_ENABLE_PERFORMANCE_MONITORING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'cost_tracking_enabled': os.getenv('LIGHTRAG_ENABLE_COST_TRACKING', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Quality validation sub-features
        'relevance_scoring_enabled': os.getenv('LIGHTRAG_ENABLE_RELEVANCE_SCORING', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        'accuracy_validation_enabled': os.getenv('LIGHTRAG_ENABLE_ACCURACY_VALIDATION', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'factual_validation_enabled': os.getenv('LIGHTRAG_ENABLE_FACTUAL_VALIDATION', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'claim_extraction_enabled': os.getenv('LIGHTRAG_ENABLE_CLAIM_EXTRACTION', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Performance and monitoring features
        'benchmarking_enabled': os.getenv('LIGHTRAG_ENABLE_BENCHMARKING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'progress_tracking_enabled': os.getenv('LIGHTRAG_ENABLE_PROGRESS_TRACKING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'unified_progress_tracking_enabled': os.getenv('LIGHTRAG_ENABLE_UNIFIED_PROGRESS_TRACKING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Document processing features
        'document_indexing_enabled': os.getenv('LIGHTRAG_ENABLE_DOCUMENT_INDEXING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'pdf_processing_enabled': os.getenv('LIGHTRAG_ENABLE_PDF_PROCESSING', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Advanced features
        'recovery_system_enabled': os.getenv('LIGHTRAG_ENABLE_RECOVERY_SYSTEM', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'alert_system_enabled': os.getenv('LIGHTRAG_ENABLE_ALERT_SYSTEM', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'budget_monitoring_enabled': os.getenv('LIGHTRAG_ENABLE_BUDGET_MONITORING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Integration control flags
        'circuit_breaker_enabled': os.getenv('LIGHTRAG_ENABLE_CIRCUIT_BREAKER', 'true').lower() in ('true', '1', 'yes', 't', 'on'),
        'ab_testing_enabled': os.getenv('LIGHTRAG_ENABLE_AB_TESTING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'conditional_routing_enabled': os.getenv('LIGHTRAG_ENABLE_CONDITIONAL_ROUTING', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        
        # Debug and development flags
        'debug_mode_enabled': os.getenv('LIGHTRAG_DEBUG_MODE', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
        'development_features_enabled': os.getenv('LIGHTRAG_ENABLE_DEVELOPMENT_FEATURES', 'false').lower() in ('true', '1', 'yes', 't', 'on'),
    }
    
    return flags


def _register_integration_module(module_name: str, feature_flag: str, required: bool = False):
    """Register a module for conditional loading based on feature flags."""
    _INTEGRATION_MODULES[module_name] = {
        'feature_flag': feature_flag,
        'required': required,
        'loaded': False,
        'module': None,
        'exports': None
    }


def _check_integration_availability(module_name: str) -> bool:
    """Check if an integration module is available and enabled."""
    if module_name not in _INTEGRATION_MODULES:
        return False
    
    module_info = _INTEGRATION_MODULES[module_name]
    feature_flag = module_info['feature_flag']
    
    # Check feature flag
    if not _FEATURE_FLAGS.get(feature_flag, False):
        return False
    
    # Check if module can be imported
    if not module_info['loaded']:
        try:
            import importlib
            module = importlib.import_module(f'.{module_name}', package='lightrag_integration')
            module_info['module'] = module
            module_info['loaded'] = True
            return True
        except ImportError:
            return False
    
    return module_info['loaded']


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific feature is enabled via feature flags."""
    return _FEATURE_FLAGS.get(feature_name, False)


def get_enabled_features() -> dict:
    """Get all enabled features and their status."""
    return {key: value for key, value in _FEATURE_FLAGS.items() if value}


def get_integration_status() -> dict:
    """Get comprehensive integration status including feature flags and module availability."""
    status = {
        'feature_flags': _FEATURE_FLAGS.copy(),
        'modules': {},
        'factory_functions': list(_FACTORY_FUNCTIONS.keys()),
        'integration_health': 'healthy'
    }
    
    # Check module status
    for module_name, module_info in _INTEGRATION_MODULES.items():
        status['modules'][module_name] = {
            'feature_flag': module_info['feature_flag'],
            'required': module_info['required'],
            'enabled': _FEATURE_FLAGS.get(module_info['feature_flag'], False),
            'available': _check_integration_availability(module_name),
            'loaded': module_info['loaded']
        }
    
    # Determine overall health
    required_modules_failed = [
        name for name, info in _INTEGRATION_MODULES.items() 
        if info['required'] and not _check_integration_availability(name)
    ]
    
    if required_modules_failed:
        status['integration_health'] = 'degraded'
        status['failed_required_modules'] = required_modules_failed
    
    return status


def validate_integration_setup() -> tuple[bool, list[str]]:
    """Validate integration setup and return status with any issues."""
    issues = []
    
    # Check core requirements
    if not _FEATURE_FLAGS.get('lightrag_integration_enabled', False):
        issues.append("LightRAG integration is disabled (LIGHTRAG_INTEGRATION_ENABLED=false)")
    
    # Check required modules
    required_modules = [name for name, info in _INTEGRATION_MODULES.items() if info['required']]
    for module_name in required_modules:
        if not _check_integration_availability(module_name):
            feature_flag = _INTEGRATION_MODULES[module_name]['feature_flag']
            if not _FEATURE_FLAGS.get(feature_flag, False):
                issues.append(f"Required module '{module_name}' is disabled by feature flag '{feature_flag}'")
            else:
                issues.append(f"Required module '{module_name}' cannot be imported")
    
    # Check environment configuration
    import os
    if not os.getenv('OPENAI_API_KEY'):
        issues.append("OPENAI_API_KEY environment variable is not set")
    
    # Check directory permissions
    try:
        config = LightRAGConfig.get_config()
        from pathlib import Path
        
        working_dir = Path(config.working_dir)
        if not working_dir.exists() or not os.access(working_dir, os.W_OK):
            issues.append(f"Working directory is not accessible or writable: {working_dir}")
            
    except Exception as e:
        issues.append(f"Configuration validation failed: {str(e)}")
    
    return len(issues) == 0, issues




# =============================================================================
# CONDITIONAL FACTORY FUNCTIONS
# =============================================================================

def create_clinical_rag_system_with_features(**config_overrides):
    """Create a Clinical RAG system with features enabled based on feature flags."""
    if not is_feature_enabled('lightrag_integration_enabled'):
        raise RuntimeError(
            "LightRAG integration is disabled. Set LIGHTRAG_INTEGRATION_ENABLED=true to enable."
        )
    
    # Apply feature-flag based defaults
    feature_defaults = {}
    
    if is_feature_enabled('cost_tracking_enabled'):
        feature_defaults.update({
            'enable_cost_tracking': True,
            'cost_persistence_enabled': True,
        })
    
    if is_feature_enabled('quality_validation_enabled'):
        feature_defaults.update({
            'enable_relevance_scoring': is_feature_enabled('relevance_scoring_enabled'),
        })
    
    if is_feature_enabled('performance_monitoring_enabled'):
        feature_defaults.update({
            'enable_performance_monitoring': True,
        })
    
    # Merge with user overrides
    feature_defaults.update(config_overrides)
    
    return create_clinical_rag_system(**feature_defaults)


def create_quality_validation_system(**config_overrides):
    """Create a system optimized for quality validation if the feature is enabled."""
    if not is_feature_enabled('quality_validation_enabled'):
        raise RuntimeError(
            "Quality validation is disabled. Set LIGHTRAG_ENABLE_QUALITY_VALIDATION=true to enable."
        )
    
    quality_config = get_quality_validation_config(**config_overrides)
    return create_clinical_rag_system(**quality_config)


def create_performance_monitoring_system(**config_overrides):
    """Create a system optimized for performance monitoring if the feature is enabled."""
    if not is_feature_enabled('performance_monitoring_enabled'):
        raise RuntimeError(
            "Performance monitoring is disabled. Set LIGHTRAG_ENABLE_PERFORMANCE_MONITORING=true to enable."
        )
    
    performance_config = {
        'enable_performance_monitoring': True,
        'enable_cost_tracking': True,
        'model': 'gpt-4o',  # Use more capable model for monitoring
        **config_overrides
    }
    return create_clinical_rag_system(**performance_config)


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
# MODULE INITIALIZATION & FEATURE FLAG SETUP
# =============================================================================

# Initialize feature flags
_FEATURE_FLAGS = _load_feature_flags()

# Register integration modules with their feature flags
_register_integration_module('relevance_scorer', 'relevance_scoring_enabled', required=False)
_register_integration_module('accuracy_scorer', 'accuracy_validation_enabled', required=False)
_register_integration_module('factual_accuracy_validator', 'factual_validation_enabled', required=False)
_register_integration_module('claim_extractor', 'claim_extraction_enabled', required=False)
_register_integration_module('enhanced_response_quality_assessor', 'quality_validation_enabled', required=False)
_register_integration_module('quality_report_generator', 'quality_validation_enabled', required=False)
_register_integration_module('performance_benchmarking', 'benchmarking_enabled', required=False)
_register_integration_module('unified_progress_tracker', 'unified_progress_tracking_enabled', required=False)
_register_integration_module('progress_tracker', 'progress_tracking_enabled', required=False)
_register_integration_module('realtime_budget_monitor', 'budget_monitoring_enabled', required=False)
_register_integration_module('document_indexer', 'document_indexing_enabled', required=False)
_register_integration_module('advanced_recovery_system', 'recovery_system_enabled', required=False)
_register_integration_module('alert_system', 'alert_system_enabled', required=False)

# Register factory functions
_FACTORY_FUNCTIONS.update({
    'create_clinical_rag_system_with_features': create_clinical_rag_system_with_features,
    'create_quality_validation_system': create_quality_validation_system,
    'create_performance_monitoring_system': create_performance_monitoring_system,
})

# Now rebuild exports with actual feature availability
__all__ = _build_dynamic_exports()

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
    
    # Log feature flag status
    enabled_features = get_enabled_features()
    if enabled_features:
        _logger.info(f"Enabled features: {', '.join(enabled_features.keys())}")
    else:
        _logger.info("No optional features enabled")
    
    # Log integration status
    _logger.debug("Checking integration component status...")
    status = get_integration_status()
    
    integration_health = status.get('integration_health', 'unknown')
    _logger.info(f"Integration health: {integration_health}")
    
    if integration_health == 'degraded':
        failed_modules = status.get('failed_required_modules', [])
        _logger.warning(f"Failed required modules: {', '.join(failed_modules)}")
    
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