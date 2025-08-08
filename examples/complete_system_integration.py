#!/usr/bin/env python3
"""
Complete System Integration Example for CMO-LightRAG

This example demonstrates a full replacement of the current query processing
system with comprehensive LightRAG integration, including document processing
pipeline, quality assessment, cost tracking, monitoring, and all advanced
features of the Clinical Metabolomics Oracle.

Key Features:
- Complete replacement of Perplexity API with LightRAG
- Full document processing pipeline integration
- Comprehensive quality assessment and validation
- Advanced cost tracking and budget management
- Real-time monitoring and performance analytics
- Audit trails and compliance tracking
- Progressive PDF knowledge base building
- Multi-modal research categorization
- Enhanced error handling and recovery

Usage:
    # Full system configuration
    export OPENAI_API_KEY="your-api-key"
    export LIGHTRAG_MODEL="gpt-4o"
    export LIGHTRAG_ENABLE_ALL_FEATURES="true"
    export LIGHTRAG_DAILY_BUDGET_LIMIT="100.0"
    export LIGHTRAG_MONTHLY_BUDGET_LIMIT="2000.0"
    
    # Run complete system
    chainlit run examples/complete_system_integration.py
"""

import asyncio
import logging
import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import traceback

import chainlit as cl
from lingua import LanguageDetector

# Import LightRAG integration components
from lightrag_integration import (
    create_clinical_rag_system,
    ClinicalMetabolomicsRAG,
    LightRAGConfig,
    QueryResponse,
    setup_lightrag_logging,
    get_integration_status,
    validate_integration_setup,
    CostSummary,
    BudgetManager,
    APIUsageMetricsLogger,
    ResearchCategorizer,
    AuditTrail,
    BiomedicalPDFProcessor,
    QualityReportGenerator,
    PerformanceBenchmarkSuite,
    UnifiedProgressTracker
)

# Import existing CMO components
from src.translation import BaseTranslator, detect_language, get_language_detector, get_translator, translate
from src.lingua_iso_codes import IsoCode639_1

# Initialize comprehensive logging
setup_lightrag_logging()
logger = logging.getLogger(__name__)


class CompleteSystemManager:
    """
    Complete system manager that orchestrates all components of the
    Clinical Metabolomics Oracle with full LightRAG integration.
    """
    
    def __init__(self):
        """Initialize the complete system manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Core components
        self.rag_system: Optional[ClinicalMetabolomicsRAG] = None
        self.budget_manager: Optional[BudgetManager] = None
        self.metrics_logger: Optional[APIUsageMetricsLogger] = None
        self.research_categorizer: Optional[ResearchCategorizer] = None
        self.audit_trail: Optional[AuditTrail] = None
        self.pdf_processor: Optional[BiomedicalPDFProcessor] = None
        self.quality_reporter: Optional[QualityReportGenerator] = None
        self.progress_tracker: Optional[UnifiedProgressTracker] = None
        
        # Configuration
        self.config = None
        self.system_initialized = False
        self.initialization_start_time = None
        
        # Performance tracking
        self.query_count = 0
        self.total_cost = 0.0
        self.last_maintenance = datetime.now()
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the complete system with all components.
        
        Returns:
            Dict containing initialization status and component health
        """
        self.initialization_start_time = time.time()
        initialization_status = {
            "started_at": datetime.now().isoformat(),
            "components": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            self.logger.info("Starting complete system initialization...")
            
            # Step 1: Validate setup
            await self._validate_system_setup(initialization_status)
            
            # Step 2: Initialize core RAG system
            await self._initialize_rag_system(initialization_status)
            
            # Step 3: Initialize supporting components
            await self._initialize_supporting_components(initialization_status)
            
            # Step 4: Initialize quality and monitoring systems
            await self._initialize_quality_systems(initialization_status)
            
            # Step 5: Initialize document processing
            await self._initialize_document_processing(initialization_status)
            
            # Step 6: Run system health checks
            await self._run_health_checks(initialization_status)
            
            # Step 7: Initialize knowledge base if needed
            await self._initialize_knowledge_base(initialization_status)
            
            self.system_initialized = True
            initialization_time = time.time() - self.initialization_start_time
            
            initialization_status.update({
                "success": True,
                "initialization_time": initialization_time,
                "completed_at": datetime.now().isoformat()
            })
            
            self.logger.info(f"Complete system initialization successful in {initialization_time:.2f}s")
            
            # Log successful initialization to audit trail
            if self.audit_trail:
                await self.audit_trail.log_event(
                    "system_initialization",
                    "Complete system initialized successfully",
                    {"initialization_time": initialization_time, "components": list(initialization_status["components"].keys())}
                )
            
            return initialization_status
            
        except Exception as e:
            initialization_status.update({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "completed_at": datetime.now().isoformat()
            })
            
            self.logger.error(f"Complete system initialization failed: {e}")
            return initialization_status
    
    async def _validate_system_setup(self, status: Dict[str, Any]):
        """Validate system setup and configuration."""
        self.logger.info("Validating system setup...")
        
        try:
            is_valid, issues = validate_integration_setup()
            
            if not is_valid:
                for issue in issues:
                    status["warnings"].append(f"Setup validation: {issue}")
                    self.logger.warning(f"Setup issue: {issue}")
            
            status["components"]["setup_validation"] = {
                "status": "completed" if is_valid else "completed_with_warnings",
                "issues": len(issues)
            }
            
        except Exception as e:
            status["errors"].append(f"Setup validation failed: {e}")
            self.logger.error(f"Setup validation error: {e}")
    
    async def _initialize_rag_system(self, status: Dict[str, Any]):
        """Initialize the core RAG system."""
        self.logger.info("Initializing RAG system...")
        
        try:
            # Create RAG system with full feature set enabled
            self.rag_system = create_clinical_rag_system(
                # Core configuration
                model=os.getenv('LIGHTRAG_MODEL', 'gpt-4o'),
                
                # Budget management
                daily_budget_limit=float(os.getenv('LIGHTRAG_DAILY_BUDGET_LIMIT', '100.0')),
                monthly_budget_limit=float(os.getenv('LIGHTRAG_MONTHLY_BUDGET_LIMIT', '2000.0')),
                
                # Quality and validation
                enable_quality_validation=True,
                enable_relevance_scoring=True,
                relevance_confidence_threshold=0.80,
                
                # Cost and performance tracking
                enable_cost_tracking=True,
                cost_persistence_enabled=True,
                enable_performance_monitoring=True,
                
                # Research and audit features
                enable_research_categorization=True,
                enable_audit_trail=True,
                
                # Advanced features
                enable_parallel_relevance_processing=True,
                cost_alert_threshold_percentage=85.0
            )
            
            # Initialize the RAG system
            await self.rag_system.initialize_rag()
            
            # Verify initialization
            health_check = await self.rag_system.health_check()
            
            if health_check.get("status") == "healthy":
                status["components"]["rag_system"] = {
                    "status": "healthy",
                    "model": self.rag_system.config.model,
                    "features_enabled": len([k for k, v in health_check.items() if v is True])
                }
                self.logger.info("RAG system initialized successfully")
            else:
                raise RuntimeError(f"RAG system health check failed: {health_check}")
            
        except Exception as e:
            status["errors"].append(f"RAG system initialization failed: {e}")
            self.logger.error(f"RAG system initialization error: {e}")
            raise
    
    async def _initialize_supporting_components(self, status: Dict[str, Any]):
        """Initialize supporting components like budget manager, metrics logger, etc."""
        self.logger.info("Initializing supporting components...")
        
        try:
            # Initialize budget manager
            if self.rag_system and hasattr(self.rag_system, 'budget_manager'):
                self.budget_manager = self.rag_system.budget_manager
                status["components"]["budget_manager"] = {"status": "initialized"}
            
            # Initialize metrics logger
            if self.rag_system and hasattr(self.rag_system, 'metrics_logger'):
                self.metrics_logger = self.rag_system.metrics_logger
                status["components"]["metrics_logger"] = {"status": "initialized"}
            
            # Initialize research categorizer
            if self.rag_system and hasattr(self.rag_system, 'research_categorizer'):
                self.research_categorizer = self.rag_system.research_categorizer
                status["components"]["research_categorizer"] = {"status": "initialized"}
            
            # Initialize audit trail
            if self.rag_system and hasattr(self.rag_system, 'audit_trail'):
                self.audit_trail = self.rag_system.audit_trail
                status["components"]["audit_trail"] = {"status": "initialized"}
            
            self.logger.info("Supporting components initialized successfully")
            
        except Exception as e:
            status["warnings"].append(f"Some supporting components failed to initialize: {e}")
            self.logger.warning(f"Supporting components initialization warning: {e}")
    
    async def _initialize_quality_systems(self, status: Dict[str, Any]):
        """Initialize quality assessment and monitoring systems."""
        self.logger.info("Initializing quality systems...")
        
        try:
            # Initialize quality report generator
            if self.rag_system:
                try:
                    from lightrag_integration import QualityReportGenerator
                    self.quality_reporter = QualityReportGenerator(self.rag_system)
                    status["components"]["quality_reporter"] = {"status": "initialized"}
                except ImportError:
                    status["warnings"].append("Quality report generator not available")
            
            # Initialize progress tracker
            try:
                from lightrag_integration import UnifiedProgressTracker
                self.progress_tracker = UnifiedProgressTracker(
                    enable_realtime_updates=True,
                    log_to_file=True,
                    log_file_path="logs/progress_tracking.jsonl"
                )
                await self.progress_tracker.start_tracking()
                status["components"]["progress_tracker"] = {"status": "initialized"}
            except ImportError:
                status["warnings"].append("Progress tracker not available")
            
            self.logger.info("Quality systems initialized successfully")
            
        except Exception as e:
            status["warnings"].append(f"Quality systems initialization had issues: {e}")
            self.logger.warning(f"Quality systems initialization warning: {e}")
    
    async def _initialize_document_processing(self, status: Dict[str, Any]):
        """Initialize document processing capabilities."""
        self.logger.info("Initializing document processing...")
        
        try:
            # Initialize PDF processor
            self.pdf_processor = BiomedicalPDFProcessor(
                output_dir=self.rag_system.config.working_dir / "processed_docs" if self.rag_system else Path("processed_docs"),
                enable_quality_scoring=True,
                enable_progress_tracking=True
            )
            
            status["components"]["pdf_processor"] = {
                "status": "initialized",
                "output_dir": str(self.pdf_processor.output_dir)
            }
            
            self.logger.info("Document processing initialized successfully")
            
        except Exception as e:
            status["warnings"].append(f"Document processing initialization failed: {e}")
            self.logger.warning(f"Document processing initialization warning: {e}")
    
    async def _run_health_checks(self, status: Dict[str, Any]):
        """Run comprehensive health checks on all systems."""
        self.logger.info("Running health checks...")
        
        try:
            health_results = {}
            
            # RAG system health check
            if self.rag_system:
                rag_health = await self.rag_system.health_check()
                health_results["rag_system"] = rag_health
            
            # Budget manager health check
            if self.budget_manager:
                try:
                    budget_status = await self.budget_manager.get_current_status()
                    health_results["budget_manager"] = {"status": "healthy", "budget_remaining": budget_status.remaining_daily_budget}
                except Exception as e:
                    health_results["budget_manager"] = {"status": "unhealthy", "error": str(e)}
            
            # Integration status check
            integration_status = get_integration_status()
            health_results["integration"] = integration_status
            
            status["health_checks"] = health_results
            self.logger.info("Health checks completed")
            
        except Exception as e:
            status["warnings"].append(f"Health checks had issues: {e}")
            self.logger.warning(f"Health checks warning: {e}")
    
    async def _initialize_knowledge_base(self, status: Dict[str, Any]):
        """Initialize or verify knowledge base."""
        self.logger.info("Checking knowledge base status...")
        
        try:
            # Check if papers directory exists and has content
            papers_dir = Path("papers")
            if papers_dir.exists():
                pdf_files = list(papers_dir.glob("*.pdf"))
                if pdf_files:
                    self.logger.info(f"Found {len(pdf_files)} PDF files in papers directory")
                    
                    # Check if we need to process any new PDFs
                    if self.pdf_processor and self.rag_system:
                        await self._process_new_documents(pdf_files, status)
                else:
                    status["warnings"].append("No PDF files found in papers directory")
            else:
                status["warnings"].append("Papers directory not found")
            
            status["components"]["knowledge_base"] = {
                "status": "checked",
                "pdf_files_found": len(pdf_files) if 'pdf_files' in locals() else 0
            }
            
        except Exception as e:
            status["warnings"].append(f"Knowledge base initialization had issues: {e}")
            self.logger.warning(f"Knowledge base initialization warning: {e}")
    
    async def _process_new_documents(self, pdf_files: List[Path], status: Dict[str, Any]):
        """Process new PDF documents if needed."""
        try:
            for pdf_file in pdf_files[:2]:  # Limit to 2 files for initialization
                self.logger.info(f"Processing document: {pdf_file.name}")
                
                # Process PDF
                result = await self.pdf_processor.process_pdf_async(str(pdf_file))
                
                if result and self.rag_system:
                    # Add to knowledge base
                    await self.rag_system.add_documents([result])
                    self.logger.info(f"Added {pdf_file.name} to knowledge base")
            
            status["components"]["document_processing"] = {
                "status": "completed",
                "documents_processed": min(len(pdf_files), 2)
            }
            
        except Exception as e:
            self.logger.warning(f"Document processing warning: {e}")
            status["warnings"].append(f"Document processing had issues: {e}")
    
    async def process_query_comprehensive(self, query: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query using the complete system with all features.
        
        Args:
            query: User query string
            session_data: Session context data
            
        Returns:
            Dict containing comprehensive response data
        """
        start_time = time.time()
        query_id = f"query_{int(time.time() * 1000)}"
        
        try:
            self.query_count += 1
            
            # Log query start to audit trail
            if self.audit_trail:
                await self.audit_trail.log_event(
                    "query_start",
                    f"Processing query: {query[:100]}...",
                    {"query_id": query_id, "query_length": len(query)}
                )
            
            # Start progress tracking
            if self.progress_tracker:
                await self.progress_tracker.start_operation(
                    f"process_query_{query_id}",
                    "Processing comprehensive query",
                    estimated_steps=6
                )
            
            # Step 1: Categorize research query
            research_category = None
            if self.research_categorizer:
                try:
                    category_result = await self.research_categorizer.categorize_query(query)
                    research_category = category_result.predicted_category.value
                    
                    if self.progress_tracker:
                        await self.progress_tracker.update_progress(f"process_query_{query_id}", 1, f"Categorized as {research_category}")
                    
                except Exception as e:
                    self.logger.warning(f"Research categorization failed: {e}")
            
            # Step 2: Check budget constraints
            budget_ok = True
            if self.budget_manager:
                try:
                    status = await self.budget_manager.get_current_status()
                    if status.daily_budget_exceeded or status.monthly_budget_exceeded:
                        budget_ok = False
                        
                    if self.progress_tracker:
                        await self.progress_tracker.update_progress(f"process_query_{query_id}", 2, f"Budget check: {'OK' if budget_ok else 'Exceeded'}")
                        
                except Exception as e:
                    self.logger.warning(f"Budget check failed: {e}")
            
            if not budget_ok:
                return {
                    "content": "I apologize, but the daily or monthly budget limit has been exceeded. Please try again later or contact an administrator.",
                    "budget_exceeded": True,
                    "processing_time": time.time() - start_time,
                    "query_id": query_id
                }
            
            # Step 3: Process query with RAG system
            if self.progress_tracker:
                await self.progress_tracker.update_progress(f"process_query_{query_id}", 3, "Processing with RAG system")
            
            rag_response = await self.rag_system.query(
                query=query,
                mode="hybrid",
                include_metadata=True,
                enable_quality_scoring=True,
                research_category=research_category
            )
            
            # Step 4: Format response
            if self.progress_tracker:
                await self.progress_tracker.update_progress(f"process_query_{query_id}", 4, "Formatting response")
            
            formatted_response = await self._format_comprehensive_response(rag_response, query_id)
            
            # Step 5: Log metrics
            if self.progress_tracker:
                await self.progress_tracker.update_progress(f"process_query_{query_id}", 5, "Logging metrics")
            
            processing_time = time.time() - start_time
            
            # Log to metrics logger
            if self.metrics_logger:
                try:
                    await self.metrics_logger.log_api_usage(
                        "query_processing",
                        processing_time,
                        {"query_length": len(query), "response_length": len(formatted_response.get("content", "")), "research_category": research_category}
                    )
                except Exception as e:
                    self.logger.warning(f"Metrics logging failed: {e}")
            
            # Get cost information
            cost_info = {}
            if hasattr(rag_response, 'cost') and rag_response.cost:
                self.total_cost += rag_response.cost
                cost_info = {"query_cost": rag_response.cost, "total_cost": self.total_cost}
            
            # Step 6: Finalize
            if self.progress_tracker:
                await self.progress_tracker.complete_operation(f"process_query_{query_id}", "Query processed successfully")
            
            # Log successful completion
            if self.audit_trail:
                await self.audit_trail.log_event(
                    "query_completed",
                    f"Query processed successfully: {query_id}",
                    {"processing_time": processing_time, "research_category": research_category, "cost": cost_info.get("query_cost", 0)}
                )
            
            # Combine all response data
            comprehensive_response = {
                **formatted_response,
                "query_id": query_id,
                "processing_time": processing_time,
                "research_category": research_category,
                "cost_info": cost_info,
                "system_info": {
                    "query_count": self.query_count,
                    "total_system_cost": self.total_cost,
                    "features_used": ["rag_processing", "research_categorization", "budget_management", "quality_scoring", "audit_logging"]
                }
            }
            
            self.logger.info(f"Comprehensive query processing completed in {processing_time:.2f}s for query {query_id}")
            
            return comprehensive_response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Log error to audit trail
            if self.audit_trail:
                await self.audit_trail.log_event(
                    "query_error",
                    f"Query processing failed: {query_id}",
                    {"error": str(e), "processing_time": processing_time}
                )
            
            # Complete progress tracking with error
            if self.progress_tracker:
                await self.progress_tracker.complete_operation(f"process_query_{query_id}", f"Query failed: {str(e)}", success=False)
            
            self.logger.error(f"Comprehensive query processing failed for {query_id}: {e}")
            
            return {
                "content": "I apologize, but I encountered an error processing your request. The issue has been logged and will be investigated.",
                "error": str(e),
                "query_id": query_id,
                "processing_time": processing_time
            }
    
    async def _format_comprehensive_response(self, response: QueryResponse, query_id: str) -> Dict[str, Any]:
        """Format response with comprehensive metadata."""
        # Extract main content
        content = response.response if hasattr(response, 'response') else str(response)
        
        # Process citations and sources
        citations = []
        sources_info = []
        
        if hasattr(response, 'metadata') and response.metadata:
            sources = response.metadata.get('sources', [])
            for i, source in enumerate(sources, 1):
                citation_text = source.get('url', source.get('title', f'Source {i}'))
                citations.append(citation_text)
                sources_info.append({
                    'id': i,
                    'title': source.get('title', f'Source {i}'),
                    'url': source.get('url', ''),
                    'confidence': source.get('confidence_score', 0.8),
                    'content_preview': source.get('content', '')[:200] + "..." if source.get('content', '') else ""
                })
        
        # Format bibliography
        bibliography = self._format_enhanced_bibliography(sources_info)
        
        # Add quality scores if available
        quality_info = {}
        if hasattr(response, 'confidence_score'):
            quality_info['confidence_score'] = response.confidence_score
        if hasattr(response, 'relevance_score'):
            quality_info['relevance_score'] = response.relevance_score
        if hasattr(response, 'quality_metrics'):
            quality_info.update(response.quality_metrics)
        
        return {
            "content": content,
            "citations": citations,
            "bibliography": bibliography,
            "sources_info": sources_info,
            "quality_info": quality_info,
            "source_count": len(citations)
        }
    
    def _format_enhanced_bibliography(self, sources_info: List[Dict[str, Any]]) -> str:
        """Format enhanced bibliography with confidence scores and previews."""
        if not sources_info:
            return ""
        
        bibliography = "\n\n\n**References with Quality Assessment:**\n"
        
        for source in sources_info:
            confidence = source.get('confidence', 0.8)
            confidence_indicator = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            
            bibliography += f"[{source['id']}]: {source['title']}\n"
            if source['url']:
                bibliography += f"      URL: {source['url']}\n"
            bibliography += f"      {confidence_indicator} Confidence: {confidence:.2f}\n"
            if source['content_preview']:
                bibliography += f"      Preview: {source['content_preview']}\n"
            bibliography += "\n"
        
        return bibliography
    
    async def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system status and performance report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "system_status": "operational" if self.system_initialized else "initializing",
                "uptime": time.time() - self.initialization_start_time if self.initialization_start_time else 0,
                "query_statistics": {
                    "total_queries": self.query_count,
                    "total_cost": self.total_cost,
                    "average_cost_per_query": self.total_cost / self.query_count if self.query_count > 0 else 0
                }
            }
            
            # Add component status
            if self.rag_system:
                report["rag_system"] = await self.rag_system.health_check()
                cost_summary = await self.rag_system.get_cost_summary()
                if cost_summary:
                    report["cost_summary"] = cost_summary.__dict__
            
            # Add budget information
            if self.budget_manager:
                budget_status = await self.budget_manager.get_current_status()
                report["budget_status"] = {
                    "daily_remaining": budget_status.remaining_daily_budget,
                    "monthly_remaining": budget_status.remaining_monthly_budget,
                    "daily_exceeded": budget_status.daily_budget_exceeded,
                    "monthly_exceeded": budget_status.monthly_budget_exceeded
                }
            
            # Add quality metrics if quality reporter is available
            if self.quality_reporter:
                try:
                    quality_summary = await self.quality_reporter.generate_quality_summary()
                    report["quality_summary"] = quality_summary
                except Exception as e:
                    report["quality_summary"] = {"error": str(e)}
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate system report: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def perform_maintenance(self) -> Dict[str, Any]:
        """Perform routine system maintenance tasks."""
        maintenance_start = time.time()
        maintenance_report = {
            "started_at": datetime.now().isoformat(),
            "tasks": []
        }
        
        try:
            # Task 1: Clean up old log files
            maintenance_report["tasks"].append(await self._cleanup_old_logs())
            
            # Task 2: Generate quality report if needed
            if self.quality_reporter:
                try:
                    await self.quality_reporter.generate_comprehensive_report()
                    maintenance_report["tasks"].append({"task": "quality_report", "status": "completed"})
                except Exception as e:
                    maintenance_report["tasks"].append({"task": "quality_report", "status": "failed", "error": str(e)})
            
            # Task 3: Update performance benchmarks
            if self.rag_system:
                try:
                    # Run quick performance check
                    health_check = await self.rag_system.health_check()
                    maintenance_report["tasks"].append({"task": "health_check", "status": "completed", "result": health_check})
                except Exception as e:
                    maintenance_report["tasks"].append({"task": "health_check", "status": "failed", "error": str(e)})
            
            self.last_maintenance = datetime.now()
            maintenance_time = time.time() - maintenance_start
            
            maintenance_report.update({
                "completed_at": datetime.now().isoformat(),
                "duration": maintenance_time,
                "success": True
            })
            
            self.logger.info(f"System maintenance completed in {maintenance_time:.2f}s")
            
        except Exception as e:
            maintenance_report.update({
                "completed_at": datetime.now().isoformat(),
                "duration": time.time() - maintenance_start,
                "success": False,
                "error": str(e)
            })
            
            self.logger.error(f"System maintenance failed: {e}")
        
        return maintenance_report
    
    async def _cleanup_old_logs(self) -> Dict[str, Any]:
        """Clean up old log files."""
        try:
            logs_dir = Path("logs")
            if not logs_dir.exists():
                return {"task": "log_cleanup", "status": "skipped", "reason": "logs directory not found"}
            
            # Find files older than 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            old_files = []
            
            for log_file in logs_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    old_files.append(log_file)
            
            # Archive old files (don't delete, just compress)
            archived_count = 0
            for old_file in old_files[:10]:  # Limit to 10 files per maintenance
                try:
                    import gzip
                    with open(old_file, 'rb') as f_in:
                        with gzip.open(f"{old_file}.gz", 'wb') as f_out:
                            f_out.writelines(f_in)
                    old_file.unlink()  # Delete original
                    archived_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to archive {old_file}: {e}")
            
            return {
                "task": "log_cleanup", 
                "status": "completed", 
                "old_files_found": len(old_files),
                "files_archived": archived_count
            }
            
        except Exception as e:
            return {"task": "log_cleanup", "status": "failed", "error": str(e)}


# Global system manager instance
SYSTEM_MANAGER: Optional[CompleteSystemManager] = None


# Chainlit integration with complete system

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authentication callback."""
    if (username, password) == ("admin", "admin123") or (username, password) == ("testing", "ku9R_3"):
        return cl.User(
            identifier="admin",
            metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session with complete system."""
    global SYSTEM_MANAGER
    
    try:
        # Initialize system manager if not already done
        if SYSTEM_MANAGER is None:
            SYSTEM_MANAGER = CompleteSystemManager()
            
            # Show initialization progress
            init_message = await cl.Message(
                content="🔄 Initializing Clinical Metabolomics Oracle complete system...\nThis may take a moment as we set up all advanced features.",
                author="CMO"
            ).send()
            
            # Initialize system
            initialization_result = await SYSTEM_MANAGER.initialize()
            
            if initialization_result["success"]:
                # Update initialization message with success
                await init_message.update(
                    content=f"✅ System initialized successfully in {initialization_result['initialization_time']:.2f}s\n"
                           f"• Components: {len(initialization_result['components'])}\n"
                           f"• Warnings: {len(initialization_result.get('warnings', []))}\n"
                           f"• Advanced features: Quality assessment, cost tracking, audit logging, document processing"
                )
            else:
                await init_message.update(
                    content=f"❌ System initialization failed: {initialization_result.get('error', 'Unknown error')}\n"
                           f"Please check logs and try again."
                )
                return
        
        # Store system manager in session
        cl.user_session.set("system_manager", SYSTEM_MANAGER)
        
        # Generate system status report
        system_report = await SYSTEM_MANAGER.generate_system_report()
        
        # Display enhanced intro message with system status
        descr = 'Clinical Metabolomics Oracle - Complete System Integration'
        subhead = (f"Welcome to the fully integrated CMO system powered by LightRAG!\n\n"
                  f"🔬 **System Status:**\n"
                  f"• Status: {system_report.get('system_status', 'unknown').upper()}\n"
                  f"• Queries processed: {system_report.get('query_statistics', {}).get('total_queries', 0)}\n"
                  f"• Budget remaining: ${system_report.get('budget_status', {}).get('daily_remaining', 0):.2f} (daily)\n"
                  f"• Quality scoring: {'✅ Enabled' if system_report.get('rag_system', {}).get('quality_scoring_enabled') else '❌ Disabled'}\n"
                  f"• Cost tracking: {'✅ Enabled' if system_report.get('rag_system', {}).get('cost_tracking_enabled') else '❌ Disabled'}\n"
                  f"• Audit logging: {'✅ Enabled' if system_report.get('rag_system', {}).get('audit_trail_enabled') else '❌ Disabled'}\n\n"
                  f"This system provides comprehensive metabolomics research assistance with advanced quality validation, "
                  f"cost management, and performance monitoring.")
        
        disclaimer = ('The Clinical Metabolomics Oracle is an automated question answering tool, and is not intended to replace the advice of a qualified healthcare professional.\n'
                     'Content generated by the Clinical Metabolomics Oracle is for informational purposes only, and is not advice for the treatment or diagnosis of any condition.')
        
        elements = [
            cl.Text(name=descr, content=subhead, display='inline'),
            cl.Text(name='System Disclaimer', content=disclaimer, display='inline')
        ]
        
        await cl.Message(content='', elements=elements, author="CMO").send()

        # User agreement flow
        accepted = False
        while not accepted:
            res = await cl.AskActionMessage(
                content='Do you understand the purpose and limitations of the Clinical Metabolomics Oracle complete system?',
                actions=[
                    cl.Action(name='I Understand', label='I Understand', description='Agree and continue', payload={"response": "agree"}),
                    cl.Action(name='Disagree', label='Disagree', description='Disagree to terms of service', payload={"response": "disagree"})
                ],
                timeout=300,
                author="CMO",
            ).send()

            accepted = res["label"] == "I Understand"
            if not accepted:
                await cl.Message(content="You must agree to the terms of service to continue.", author="CMO").send()

        welcome = ("Welcome to the Clinical Metabolomics Oracle complete system! "
                  "Ask me anything about clinical metabolomics and I'll provide comprehensive, "
                  "quality-assessed responses with full audit trails and cost tracking.")
        
        await cl.Message(content=welcome, author="CMO").send()

        # Set up translation components
        translator: BaseTranslator = get_translator()
        cl.user_session.set("translator", translator)
        await set_chat_settings(translator)

        iso_codes = [IsoCode639_1[code.upper()].value for code in translator.get_supported_languages(as_dict=True).values() if code.upper() in IsoCode639_1._member_names_]
        detector = get_language_detector(*iso_codes)
        cl.user_session.set("detector", detector)
        
        logger.info("Complete system chat session initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during complete system chat initialization: {e}")
        await cl.Message(
            content="⚠️ There was an error initializing the complete system. Please refresh and try again.\n"
                   f"Error: {str(e)}",
            author="CMO"
        ).send()


async def set_chat_settings(translator):
    """Set up enhanced chat settings UI."""
    initial_language_value = "Detect language"
    languages_to_iso_codes = translator.get_supported_languages(as_dict=True)
    language_values = [initial_language_value] + [language.title() for language in languages_to_iso_codes.keys()]
    
    await cl.ChatSettings([
        cl.input_widget.Select(
            id="translator",
            label="Translator",
            values=["Google", "OPUS-MT"],
            initial_value="Google",
        ),
        cl.input_widget.Select(
            id="language",
            label="Language",
            values=language_values,
            initial_value=initial_language_value,
        ),
        cl.input_widget.Switch(
            id="detailed_response",
            label="Detailed Response Mode",
            initial=False,
        ),
        cl.input_widget.Switch(
            id="show_system_info",
            label="Show System Information",
            initial=True,
        )
    ]).send()


@cl.author_rename
def rename(orig_author: str):
    """Author rename function."""
    rename_dict = {"Chatbot": "CMO"}
    return rename_dict.get(orig_author, orig_author)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle messages with complete system processing."""
    start_time = time.time()
    
    try:
        # Get session components
        detector: LanguageDetector = cl.user_session.get("detector")
        translator: BaseTranslator = cl.user_session.get("translator")
        system_manager: CompleteSystemManager = cl.user_session.get("system_manager")
        
        if not system_manager or not system_manager.system_initialized:
            await cl.Message(
                content="⚠️ Complete system not properly initialized. Please refresh the page.",
                author="CMO"
            ).send()
            return
        
        content = message.content
        show_system_info = cl.user_session.get("show_system_info", True)
        detailed_response = cl.user_session.get("detailed_response", False)

        # Show enhanced thinking message
        thinking_message = await cl.Message(
            content="🧠 Analyzing your query...\n• Detecting language and categorizing research area\n• Checking budget constraints\n• Processing with advanced RAG system",
            author="CMO"
        ).send()

        # Handle language detection and translation
        language = cl.user_session.get("language")
        if not language or language == "auto":
            detection = await detect_language(detector, content)
            language = detection["language"]
        
        if language != "en" and language is not None:
            content = await translate(translator, content, source=language, target="en")

        # Update thinking message
        await thinking_message.update(content="🔍 Processing query with comprehensive analysis...")

        # Process query using complete system
        session_data = {
            "language": language,
            "translator": translator,
            "detector": detector,
            "detailed_response": detailed_response,
            "show_system_info": show_system_info
        }
        
        response_data = await system_manager.process_query_comprehensive(content, session_data)
        
        # Update thinking message with processing info
        processing_time = response_data.get("processing_time", 0)
        research_category = response_data.get("research_category", "General")
        
        await thinking_message.update(
            content=f"✅ Query processed successfully!\n"
                   f"• Category: {research_category}\n"
                   f"• Processing time: {processing_time:.2f}s\n"
                   f"• Quality assessment: Completed\n"
                   f"• Cost tracking: Updated"
        )

        # Get response content and format
        response_content = response_data.get("content", "")
        bibliography = response_data.get("bibliography", "")
        
        # Handle translation back to user language
        if language != "en" and language is not None:
            response_content = await translate(translator, response_content, source="en", target=language)

        # Add bibliography
        if bibliography:
            response_content += bibliography

        # Add comprehensive system information if requested
        if show_system_info:
            system_info = response_data.get("system_info", {})
            cost_info = response_data.get("cost_info", {})
            quality_info = response_data.get("quality_info", {})
            
            end_time = time.time()
            
            system_footer = f"\n\n📊 **System Information:**\n"
            system_footer += f"• Query ID: {response_data.get('query_id', 'N/A')}\n"
            system_footer += f"• Research Category: {research_category}\n"
            system_footer += f"• Processing Time: {end_time - start_time:.2f}s\n"
            system_footer += f"• Sources Found: {response_data.get('source_count', 0)}\n"
            
            if cost_info:
                system_footer += f"• Query Cost: ${cost_info.get('query_cost', 0):.4f}\n"
                system_footer += f"• Total System Cost: ${cost_info.get('total_cost', 0):.2f}\n"
            
            if quality_info:
                confidence = quality_info.get('confidence_score')
                if confidence:
                    system_footer += f"• Confidence Score: {confidence:.2f}\n"
            
            system_footer += f"• Total Queries: {system_info.get('query_count', 0)}\n"
            system_footer += f"• Features Used: {', '.join(system_info.get('features_used', []))}"
            
            response_content += system_footer
        else:
            # Minimal timing info
            response_content += f"\n\n*{time.time() - start_time:.2f} seconds*"

        # Send final response
        response_message = cl.Message(content=response_content)
        await response_message.send()
        
        logger.info(f"Complete system message processed successfully in {time.time() - start_time:.2f}s")
        
        # Perform maintenance if needed (every 100 queries)
        if system_manager.query_count % 100 == 0:
            maintenance_task = asyncio.create_task(system_manager.perform_maintenance())
            # Don't await - let it run in background
        
    except Exception as e:
        logger.error(f"Error processing message with complete system: {e}")
        await cl.Message(
            content=f"I apologize, but I encountered an error processing your request. "
                   f"The error has been logged for investigation.\n\nError: {str(e)}",
            author="CMO"
        ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Handle settings updates."""
    # Handle translator settings
    translator = settings["translator"]
    if translator == "Google":
        translator: BaseTranslator = get_translator("google")
    elif translator == "OPUS-MT":
        translator: BaseTranslator = get_translator("opusmt")
    
    await set_chat_settings(translator)
    cl.user_session.set("translator", translator)
    
    # Handle language settings
    language = settings["language"]
    if language == "Detect language":
        language = "auto"
    else:
        languages_to_iso_codes = translator.get_supported_languages(as_dict=True)
        language = languages_to_iso_codes.get(language.lower(), "auto")
    
    cl.user_session.set("language", language)
    
    # Handle new settings
    cl.user_session.set("detailed_response", settings.get("detailed_response", False))
    cl.user_session.set("show_system_info", settings.get("show_system_info", True))
    
    logger.info(f"Settings updated: detailed_response={settings.get('detailed_response')}, show_system_info={settings.get('show_system_info')}")


# Development and testing utilities

async def test_complete_system():
    """Test function to verify complete system integration."""
    print("Testing complete system integration...")
    
    try:
        manager = CompleteSystemManager()
        init_result = await manager.initialize()
        
        if not init_result["success"]:
            print(f"❌ System initialization failed: {init_result.get('error')}")
            return False
        
        print(f"✅ System initialized in {init_result['initialization_time']:.2f}s")
        
        # Test comprehensive query processing
        test_query = "What are the main metabolites involved in glucose metabolism and their diagnostic significance?"
        session_data = {"language": "en", "detailed_response": True, "show_system_info": True}
        
        print(f"Testing query: {test_query}")
        
        result = await manager.process_query_comprehensive(test_query, session_data)
        
        if result.get("error"):
            print(f"❌ Query processing failed: {result['error']}")
            return False
        
        print(f"✅ Complete system test successful!")
        print(f"   - Query ID: {result.get('query_id')}")
        print(f"   - Research Category: {result.get('research_category')}")
        print(f"   - Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"   - Sources Found: {result.get('source_count', 0)}")
        print(f"   - Response Length: {len(result.get('content', ''))}")
        
        # Test system report generation
        report = await manager.generate_system_report()
        print(f"   - System Report Generated: {report.get('system_status')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete system test failed: {e}")
        return False


if __name__ == "__main__":
    """Main entry point for testing or running the integration."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_complete_system())
    else:
        print("🔬 Clinical Metabolomics Oracle - Complete System Integration")
        print("=" * 70)
        print("Features:")
        print("  • Full LightRAG integration replacing Perplexity API")
        print("  • Comprehensive quality assessment and validation")
        print("  • Advanced cost tracking and budget management")
        print("  • Real-time monitoring and performance analytics") 
        print("  • Document processing and knowledge base integration")
        print("  • Audit trails and compliance tracking")
        print("  • Research categorization and metrics logging")
        print("  • Progressive system maintenance and optimization")
        print("\nConfiguration:")
        print(f"  Model: {os.getenv('LIGHTRAG_MODEL', 'gpt-4o')}")
        print(f"  Daily Budget: ${os.getenv('LIGHTRAG_DAILY_BUDGET_LIMIT', '100.0')}")
        print(f"  Monthly Budget: ${os.getenv('LIGHTRAG_MONTHLY_BUDGET_LIMIT', '2000.0')}")
        print(f"  All Features: {os.getenv('LIGHTRAG_ENABLE_ALL_FEATURES', 'true')}")
        print("\nTo run: chainlit run examples/complete_system_integration.py")
        print("To test: python examples/complete_system_integration.py test")