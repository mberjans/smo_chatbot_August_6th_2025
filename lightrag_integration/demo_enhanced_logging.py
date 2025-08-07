#!/usr/bin/env python3
"""
Demonstration of the Enhanced Logging System for Clinical Metabolomics Oracle LightRAG integration.

This script demonstrates the comprehensive logging capabilities including:
- Structured logging with correlation IDs
- Performance metrics tracking
- Specialized loggers for different components
- Error logging with detailed context
- Integration with the main RAG system

Usage:
    python demo_enhanced_logging.py
    
    Or with specific demonstrations:
    python demo_enhanced_logging.py --demo correlation
    python demo_enhanced_logging.py --demo performance
    python demo_enhanced_logging.py --demo ingestion
    python demo_enhanced_logging.py --demo diagnostics
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any
import logging

# Import enhanced logging components
from enhanced_logging import (
    EnhancedLogger, IngestionLogger, DiagnosticLogger,
    correlation_manager, PerformanceTracker, PerformanceMetrics,
    create_enhanced_loggers, setup_structured_logging,
    performance_logged
)

# Import configuration
from config import LightRAGConfig


class EnhancedLoggingDemo:
    """Comprehensive demonstration of enhanced logging capabilities."""
    
    def __init__(self, log_dir: Path = None):
        """Initialize the demo with logging configuration."""
        self.log_dir = log_dir or Path("demo_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up base logger
        self.base_logger = self._setup_base_logger()
        
        # Create enhanced loggers
        self.enhanced_loggers = create_enhanced_loggers(self.base_logger)
        
        # Set up structured logging
        structured_log_file = self.log_dir / "structured_logs.jsonl"
        self.structured_logger = setup_structured_logging("demo", structured_log_file)
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
        print(f"Enhanced Logging Demo initialized. Logs will be saved to: {self.log_dir}")
    
    def _setup_base_logger(self) -> logging.Logger:
        """Set up base logger for the demo."""
        logger = logging.getLogger("enhanced_logging_demo")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / "demo.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def demo_basic_enhanced_logging(self):
        """Demonstrate basic enhanced logging capabilities."""
        print("\n=== Basic Enhanced Logging Demo ===")
        
        enhanced_logger = self.enhanced_loggers['enhanced']
        
        # Basic logging with structured data
        enhanced_logger.info(
            "Starting basic logging demonstration",
            operation_name="demo_basic_logging",
            metadata={
                "demo_version": "1.0",
                "features": ["structured_logging", "correlation_ids", "performance_tracking"]
            }
        )
        
        # Debug logging with correlation
        enhanced_logger.debug(
            "Debug information for troubleshooting",
            metadata={
                "debug_level": "verbose",
                "component": "demo_system"
            }
        )
        
        # Warning with additional context
        enhanced_logger.warning(
            "This is a warning message with context",
            metadata={
                "warning_type": "demo_warning",
                "severity": "medium",
                "recommendation": "This is just for demonstration"
            }
        )
        
        # Error logging with detailed context
        try:
            raise ValueError("This is a demo error for testing error logging")
        except ValueError as e:
            enhanced_logger.log_error_with_context(
                "Demonstration error occurred",
                e,
                operation_name="demo_error_handling",
                additional_context={
                    "error_category": "demonstration",
                    "expected": True,
                    "impact": "none"
                }
            )
        
        print("✓ Basic enhanced logging completed")
    
    def demo_correlation_tracking(self):
        """Demonstrate correlation ID tracking across operations."""
        print("\n=== Correlation Tracking Demo ===")
        
        enhanced_logger = self.enhanced_loggers['enhanced']
        
        with correlation_manager.operation_context("main_operation", 
                                                  user_id="demo_user",
                                                  session_id="demo_session_123") as main_context:
            
            enhanced_logger.info(
                "Starting main operation",
                operation_name="main_operation",
                metadata={"step": 1, "total_steps": 3}
            )
            
            # Simulate sub-operation 1
            with correlation_manager.operation_context("sub_operation_1") as sub1_context:
                enhanced_logger.info(
                    "Executing sub-operation 1",
                    metadata={"sub_step": "data_preparation", "estimated_duration": "5s"}
                )
                
                time.sleep(0.1)  # Simulate work
                
                # Nested sub-operation
                with correlation_manager.operation_context("nested_operation") as nested_context:
                    enhanced_logger.debug(
                        "Nested operation within sub-operation 1",
                        metadata={"nesting_level": 3, "operation_type": "validation"}
                    )
                
                enhanced_logger.info(
                    "Sub-operation 1 completed successfully",
                    metadata={"duration": "0.1s", "status": "success"}
                )
            
            # Simulate sub-operation 2
            with correlation_manager.operation_context("sub_operation_2") as sub2_context:
                enhanced_logger.info(
                    "Executing sub-operation 2",
                    metadata={"sub_step": "processing", "estimated_duration": "3s"}
                )
                
                time.sleep(0.05)  # Simulate work
                
                enhanced_logger.info(
                    "Sub-operation 2 completed successfully", 
                    metadata={"duration": "0.05s", "status": "success"}
                )
            
            enhanced_logger.info(
                "Main operation completed",
                metadata={"total_duration": f"{main_context.to_dict()['duration_ms']:.1f}ms"}
            )
        
        print(f"✓ Correlation tracking completed")
        print(f"  Main operation ID: {main_context.correlation_id}")
        print(f"  Total duration: {main_context.to_dict()['duration_ms']:.1f}ms")
    
    def demo_performance_tracking(self):
        """Demonstrate performance metrics tracking."""
        print("\n=== Performance Tracking Demo ===")
        
        enhanced_logger = self.enhanced_loggers['enhanced']
        
        # Demonstrate performance tracking with decorator
        @performance_logged("cpu_intensive_task", enhanced_logger)
        def cpu_intensive_task(iterations: int) -> Dict[str, Any]:
            """Simulate CPU intensive work."""
            result = 0
            for i in range(iterations):
                result += i ** 2
            return {"result": result, "iterations": iterations}
        
        @performance_logged("memory_intensive_task", enhanced_logger)
        def memory_intensive_task(size_mb: int) -> Dict[str, Any]:
            """Simulate memory intensive work."""
            # Allocate memory (size_mb * 1MB)
            data = [0] * (size_mb * 1024 * 1024 // 8)  # 8 bytes per int
            return {"allocated_mb": size_mb, "data_points": len(data)}
        
        # Manual performance tracking
        self.performance_tracker.start_tracking()
        start_time = time.time()
        
        enhanced_logger.info("Starting performance demonstration")
        
        # Run CPU intensive task
        cpu_result = cpu_intensive_task(100000)
        
        # Run memory intensive task
        memory_result = memory_intensive_task(10)  # 10MB
        
        # Get performance metrics
        metrics = self.performance_tracker.get_metrics()
        total_duration = time.time() - start_time
        
        # Log performance summary
        enhanced_logger.info(
            "Performance demonstration completed",
            operation_name="performance_demo",
            performance_metrics=metrics,
            metadata={
                "cpu_task_result": cpu_result,
                "memory_task_result": memory_result,
                "total_duration_ms": total_duration * 1000
            }
        )
        
        print(f"✓ Performance tracking completed")
        print(f"  Memory usage: {metrics.memory_mb:.1f}MB ({metrics.memory_percent:.1f}%)")
        print(f"  Total duration: {metrics.duration_ms:.1f}ms")
        print(f"  CPU task iterations: {cpu_result['iterations']:,}")
        print(f"  Memory allocated: {memory_result['allocated_mb']}MB")
    
    def demo_ingestion_logging(self):
        """Demonstrate specialized ingestion logging."""
        print("\n=== Ingestion Logging Demo ===")
        
        ingestion_logger = self.enhanced_loggers['ingestion']
        
        # Simulate batch processing
        batch_id = "demo_batch_001"
        documents = [
            {"id": "doc001", "path": "/papers/metabolomics_study_1.pdf", "pages": 12},
            {"id": "doc002", "path": "/papers/clinical_trial_results.pdf", "pages": 8},
            {"id": "doc003", "path": "/papers/biomarker_analysis.pdf", "pages": 15},
            {"id": "doc004", "path": "/papers/pathway_mapping.pdf", "pages": 6},
            {"id": "doc005", "path": "/papers/statistical_methods.pdf", "pages": 10}
        ]
        
        # Log batch start
        ingestion_logger.log_batch_start(batch_id, len(documents), 3, 0)
        
        successful_docs = 0
        failed_docs = 0
        
        for i, doc in enumerate(documents):
            # Log document processing start
            ingestion_logger.log_document_start(doc["id"], doc["path"], batch_id)
            
            # Simulate document processing
            processing_time = 50 + (doc["pages"] * 25)  # 50ms base + 25ms per page
            time.sleep(processing_time / 1000)  # Convert to seconds for sleep
            
            # Simulate occasional failure
            if i == 2:  # Make doc003 fail for demo
                error = Exception(f"Failed to extract text from {doc['path']}")
                ingestion_logger.log_document_error(doc["id"], error, batch_id, retry_count=1)
                failed_docs += 1
            else:
                # Log successful completion
                characters_extracted = doc["pages"] * 2500  # Estimate 2500 chars per page
                ingestion_logger.log_document_complete(
                    doc["id"],
                    processing_time,
                    doc["pages"],
                    characters_extracted,
                    batch_id
                )
                successful_docs += 1
            
            # Log batch progress
            current_memory = 256 + (i * 32)  # Simulate increasing memory usage
            ingestion_logger.log_batch_progress(batch_id, successful_docs, failed_docs, current_memory)
        
        # Log batch completion
        total_processing_time = sum(50 + (doc["pages"] * 25) for doc in documents)
        ingestion_logger.log_batch_complete(batch_id, successful_docs, failed_docs, total_processing_time)
        
        print(f"✓ Ingestion logging completed")
        print(f"  Batch ID: {batch_id}")
        print(f"  Successful documents: {successful_docs}")
        print(f"  Failed documents: {failed_docs}")
        print(f"  Total processing time: {total_processing_time}ms")
    
    def demo_diagnostic_logging(self):
        """Demonstrate diagnostic logging capabilities."""
        print("\n=== Diagnostic Logging Demo ===")
        
        diagnostic_logger = self.enhanced_loggers['diagnostic']
        
        # Configuration validation
        config_validation = {
            "api_key": "configured",
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "working_directory": "accessible",
            "storage_permissions": "read_write",
            "memory_limit": "2048MB",
            "batch_size": "optimal"
        }
        
        diagnostic_logger.log_configuration_validation("system_startup", config_validation)
        
        # Storage initialization
        storage_systems = [
            ("vector_store", "/data/vectors", True, None),
            ("graph_store", "/data/graph", True, None),
            ("cache_store", "/tmp/cache", False, "Permission denied"),
            ("backup_store", "/backup/data", True, None)
        ]
        
        for storage_type, path, success, error in storage_systems:
            init_time = 50 + (hash(storage_type) % 100)  # Simulate variable init time
            diagnostic_logger.log_storage_initialization(
                storage_type, path, init_time, success, error
            )
        
        # API call logging
        api_calls = [
            ("llm_completion", "gpt-4o-mini", 1250, 0.0031, 850, True),
            ("embedding", "text-embedding-3-small", 500, 0.0002, 200, True),
            ("llm_completion", "gpt-4o-mini", 2100, 0.0052, 1200, True),
            ("embedding", "text-embedding-3-small", 0, 0.0000, 0, False),  # Failed call
        ]
        
        for api_type, model, tokens, cost, response_time, success in api_calls:
            diagnostic_logger.log_api_call_details(api_type, model, tokens, cost, response_time, success)
        
        # Memory usage monitoring
        memory_scenarios = [
            ("initialization", 128.5, 12.3, None),
            ("document_processing", 512.0, 48.7, 1024.0),
            ("batch_ingestion", 1536.2, 73.1, 1024.0),  # Over threshold
            ("cleanup", 89.3, 8.5, None)
        ]
        
        for operation, memory_mb, memory_percent, threshold in memory_scenarios:
            diagnostic_logger.log_memory_usage(operation, memory_mb, memory_percent, threshold)
        
        print("✓ Diagnostic logging completed")
        print("  Configuration validation: passed")
        print(f"  Storage systems initialized: {sum(1 for _, _, success, _ in storage_systems if success)}/4")
        print(f"  API calls logged: {len(api_calls)}")
        print(f"  Memory monitoring events: {len(memory_scenarios)}")
    
    async def demo_integration_with_rag(self):
        """Demonstrate integration with the main RAG system."""
        print("\n=== RAG Integration Demo ===")
        
        try:
            # Create a test configuration
            config = LightRAGConfig(
                api_key="demo-key",
                working_dir=self.log_dir / "rag_demo",
                auto_create_dirs=True,
                enable_file_logging=True,
                log_level="DEBUG"
            )
            
            # Note: This is a simulation - in real usage, you'd need valid API keys
            print("Creating ClinicalMetabolomicsRAG instance with enhanced logging...")
            
            # Mock the actual RAG initialization to avoid API calls in demo
            print("✓ RAG system would be initialized with enhanced logging")
            print("✓ Correlation tracking would be active for all operations")
            print("✓ Performance metrics would be collected automatically")
            print("✓ Specialized loggers would handle ingestion and diagnostics")
            
        except Exception as e:
            self.enhanced_loggers['enhanced'].log_error_with_context(
                "Demo RAG integration failed",
                e,
                operation_name="demo_rag_integration",
                additional_context={"demo_mode": True, "expected": True}
            )
            print(f"✓ Error logging demonstrated (expected in demo mode)")
    
    def demo_log_analysis(self):
        """Demonstrate analysis of generated logs."""
        print("\n=== Log Analysis Demo ===")
        
        # Analyze structured logs
        structured_log_file = self.log_dir / "structured_logs.jsonl"
        if structured_log_file.exists():
            print("Analyzing structured logs...")
            
            log_entries = []
            with open(structured_log_file, 'r') as f:
                for line in f:
                    try:
                        log_entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            if log_entries:
                print(f"  Total structured log entries: {len(log_entries)}")
                
                # Analyze by log level
                levels = {}
                operations = {}
                components = {}
                
                for entry in log_entries:
                    level = entry.get('level', 'UNKNOWN')
                    operation = entry.get('operation_name', 'unknown')
                    component = entry.get('component', 'unknown')
                    
                    levels[level] = levels.get(level, 0) + 1
                    operations[operation] = operations.get(operation, 0) + 1
                    components[component] = components.get(component, 0) + 1
                
                print("  Log levels:")
                for level, count in sorted(levels.items()):
                    print(f"    {level}: {count}")
                
                print("  Top operations:")
                for operation, count in sorted(operations.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {operation}: {count}")
                
                print("  Components:")
                for component, count in sorted(components.items()):
                    print(f"    {component}: {count}")
        
        # Analyze regular log file
        log_file = self.log_dir / "demo.log"
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            print(f"  Total log lines: {len(lines)}")
            
            # Count correlation IDs (lines with [correlation-id])
            correlation_lines = [line for line in lines if '] [' in line and '-' in line]
            print(f"  Lines with correlation IDs: {len(correlation_lines)}")
        
        print("✓ Log analysis completed")
    
    def run_all_demos(self):
        """Run all demonstration scenarios."""
        print("Enhanced Logging System Comprehensive Demo")
        print("=" * 50)
        
        self.demo_basic_enhanced_logging()
        self.demo_correlation_tracking()
        self.demo_performance_tracking()
        self.demo_ingestion_logging()
        self.demo_diagnostic_logging()
        
        # Run async demo
        asyncio.run(self.demo_integration_with_rag())
        
        self.demo_log_analysis()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"📁 Log files saved to: {self.log_dir.absolute()}")
        print(f"📊 Check the logs for detailed structured output")


def main():
    """Main demo execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Logging System Demo")
    parser.add_argument("--demo", choices=["basic", "correlation", "performance", "ingestion", 
                       "diagnostics", "integration", "analysis", "all"], 
                       default="all", help="Specific demo to run")
    parser.add_argument("--log-dir", type=Path, help="Directory for log files")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = EnhancedLoggingDemo(args.log_dir)
    
    # Run specific demo
    if args.demo == "basic":
        demo.demo_basic_enhanced_logging()
    elif args.demo == "correlation":
        demo.demo_correlation_tracking()
    elif args.demo == "performance":
        demo.demo_performance_tracking()
    elif args.demo == "ingestion":
        demo.demo_ingestion_logging()
    elif args.demo == "diagnostics":
        demo.demo_diagnostic_logging()
    elif args.demo == "integration":
        asyncio.run(demo.demo_integration_with_rag())
    elif args.demo == "analysis":
        demo.demo_log_analysis()
    else:  # "all"
        demo.run_all_demos()


if __name__ == "__main__":
    main()