#!/usr/bin/env python3
"""
Interactive Routing Decision Logging and Analytics Demonstration

This script provides a comprehensive, interactive demonstration of the routing decision
logging and analytics capabilities. It showcases all key features including:

- Different logging levels and storage strategies
- Real-time analytics and metrics collection
- Anomaly detection capabilities
- Performance monitoring and optimization
- Integration with enhanced production router
- Export and reporting functionality

Usage:
    python routing_logging_demo.py

Interactive Features:
- Menu-driven interface for exploring different aspects
- Real-time logging demonstration with simulated queries
- Live analytics dashboard updates
- Configurable logging parameters
- Export functionality demonstration

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Routing Decision Logging Demo Implementation
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import statistics
from pathlib import Path

# Add the lightrag_integration directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'lightrag_integration'))

# Import the routing decision analytics system
from lightrag_integration.routing_decision_analytics import (
    RoutingDecisionLogger,
    RoutingAnalytics,
    LoggingConfig,
    RoutingDecisionLogEntry,
    ProcessingMetrics,
    SystemState,
    LogLevel,
    StorageStrategy,
    create_routing_logger,
    create_routing_analytics,
    create_logging_config_from_env
)

# Import router components
from lightrag_integration.query_router import (
    RoutingDecision, 
    RoutingPrediction, 
    ConfidenceMetrics
)

# Try to import enhanced router if available
try:
    from lightrag_integration.enhanced_production_router import (
        EnhancedProductionIntelligentQueryRouter,
        EnhancedFeatureFlags,
        create_enhanced_production_router
    )
    ENHANCED_ROUTER_AVAILABLE = True
except ImportError:
    ENHANCED_ROUTER_AVAILABLE = False
    print("Note: Enhanced production router not available, using standalone logging demo")


class InteractiveDemo:
    """
    Interactive demonstration of routing decision logging and analytics.
    
    This class provides a comprehensive demo interface that allows users to:
    - Configure logging parameters
    - Generate realistic routing decisions
    - View real-time analytics
    - Test different scenarios
    - Export results
    """
    
    def __init__(self):
        self.logger: Optional[RoutingDecisionLogger] = None
        self.analytics: Optional[RoutingAnalytics] = None
        self.enhanced_router = None
        self.demo_session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logged_decisions = 0
        self.demo_start_time = datetime.now()
        
        # Demo query sets for realistic testing
        self.sample_queries = {
            'biomedical_research': [
                "What are the metabolic pathways involved in diabetes type 2?",
                "Latest research on CRISPR gene editing applications in metabolomics",
                "How do mitochondrial dysfunctions affect cellular metabolism?",
                "Biomarkers for early detection of Alzheimer's disease progression",
                "Role of gut microbiome in metabolic syndrome development"
            ],
            'clinical_questions': [
                "What are the side effects of metformin in diabetic patients?",
                "Normal ranges for glucose levels in adults",
                "Symptoms of metabolic acidosis and treatment options",
                "Drug interactions between statins and diabetes medications",
                "Dietary recommendations for patients with fatty liver disease"
            ],
            'general_health': [
                "How to maintain healthy blood sugar levels naturally?",
                "Benefits of intermittent fasting for metabolic health",
                "Best exercises for improving insulin sensitivity",
                "Foods that boost metabolism and energy levels",
                "Signs of vitamin B12 deficiency and sources"
            ],
            'complex_research': [
                "Multi-omics integration approaches for personalized medicine in metabolic disorders",
                "Computational models for predicting drug-metabolite interactions using machine learning",
                "Systems biology analysis of metabolic network perturbations in cancer metabolism",
                "Epigenetic regulation of metabolic gene expression in response to environmental factors",
                "Comparative metabolomics of different diabetes subtypes using mass spectrometry"
            ]
        }
        
        print("🔬 Routing Decision Logging & Analytics Demo")
        print("=" * 50)
        print(f"Session ID: {self.demo_session_id}")
        print(f"Started at: {self.demo_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    async def run(self):
        """Run the interactive demonstration"""
        try:
            await self._show_welcome_message()
            
            while True:
                choice = await self._show_main_menu()
                
                if choice == '1':
                    await self._configure_logging_demo()
                elif choice == '2':
                    await self._basic_logging_demo()
                elif choice == '3':
                    await self._analytics_dashboard_demo()
                elif choice == '4':
                    await self._anomaly_detection_demo()
                elif choice == '5':
                    await self._performance_monitoring_demo()
                elif choice == '6':
                    await self._enhanced_router_demo()
                elif choice == '7':
                    await self._export_and_reporting_demo()
                elif choice == '8':
                    await self._configuration_examples_demo()
                elif choice == '9':
                    await self._view_current_status()
                elif choice == '0':
                    await self._cleanup_and_exit()
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                
                input("\\nPress Enter to continue...")
                
        except KeyboardInterrupt:
            print("\\n\\n⚠️  Demo interrupted by user")
            await self._cleanup_and_exit()
        except Exception as e:
            print(f"\\n❌ Demo error: {e}")
            await self._cleanup_and_exit()
    
    async def _show_welcome_message(self):
        """Show welcome message and system status"""
        print("Welcome to the Routing Decision Logging & Analytics Demo!")
        print()
        print("This interactive demo showcases:")
        print("• Asynchronous logging with batching")
        print("• Multiple storage strategies (memory, file, hybrid)")
        print("• Real-time analytics and metrics collection")
        print("• Anomaly detection capabilities")
        print("• Performance monitoring and optimization")
        print("• Integration with enhanced production router")
        print()
        
        # Check system capabilities
        if ENHANCED_ROUTER_AVAILABLE:
            print("✅ Enhanced production router: Available")
        else:
            print("⚠️  Enhanced production router: Not available (standalone mode)")
        
        print(f"✅ Log directory: {Path('logs/routing_decisions').absolute()}")
        print(f"✅ Session ID: {self.demo_session_id}")
        print()
    
    async def _show_main_menu(self) -> str:
        """Display main menu and get user choice"""
        print("\\n" + "=" * 60)
        print("📋 MAIN MENU")
        print("=" * 60)
        print("1. 🔧 Configure Logging System")
        print("2. 📝 Basic Logging Demonstration")
        print("3. 📊 Real-time Analytics Dashboard")
        print("4. 🚨 Anomaly Detection Demo")
        print("5. ⚡ Performance Monitoring")
        print("6. 🚀 Enhanced Router Integration")
        print("7. 📤 Export & Reporting")
        print("8. ⚙️  Configuration Examples")
        print("9. 📋 View Current Status")
        print("0. ❌ Exit Demo")
        print("=" * 60)
        
        return input("Select an option (0-9): ").strip()
    
    async def _configure_logging_demo(self):
        """Interactive logging configuration"""
        print("\\n🔧 LOGGING CONFIGURATION")
        print("-" * 40)
        
        # Get configuration preferences
        print("\\n1. Logging Level:")
        print("   1) Minimal - Basic decision info only")
        print("   2) Standard - Decision + basic metrics (recommended)")
        print("   3) Detailed - Full context and system state")
        print("   4) Debug - Everything including raw data")
        
        level_choice = input("Select level (1-4) [2]: ").strip() or "2"
        log_levels = {
            '1': LogLevel.MINIMAL,
            '2': LogLevel.STANDARD,
            '3': LogLevel.DETAILED,
            '4': LogLevel.DEBUG
        }
        log_level = log_levels.get(level_choice, LogLevel.STANDARD)
        
        print("\\n2. Storage Strategy:")
        print("   1) Memory - Fast, volatile (good for development)")
        print("   2) File - Persistent storage (good for production)")
        print("   3) Hybrid - Memory + File (recommended)")
        print("   4) Streaming - Real-time streaming (advanced)")
        
        storage_choice = input("Select storage (1-4) [3]: ").strip() or "3"
        storage_strategies = {
            '1': StorageStrategy.MEMORY,
            '2': StorageStrategy.FILE,
            '3': StorageStrategy.HYBRID,
            '4': StorageStrategy.STREAMING
        }
        storage_strategy = storage_strategies.get(storage_choice, StorageStrategy.HYBRID)
        
        # Additional options
        batch_size = int(input("\\nBatch size for file operations [50]: ").strip() or "50")
        enable_analytics = input("Enable real-time analytics? (y/n) [y]: ").strip().lower() != 'n'
        anonymize = input("Anonymize query content? (y/n) [n]: ").strip().lower() == 'y'
        
        # Create configuration
        config = LoggingConfig(
            enabled=True,
            log_level=log_level,
            storage_strategy=storage_strategy,
            log_directory=f"logs/routing_decisions_{self.demo_session_id}",
            batch_size=batch_size,
            anonymize_queries=anonymize,
            enable_real_time_analytics=enable_analytics,
            max_memory_entries=1000,  # Smaller for demo
            analytics_aggregation_interval_minutes=1  # Faster for demo
        )
        
        # Initialize logging system
        if self.logger:
            await self.logger.stop_async_logging()
        
        self.logger = create_routing_logger(config)
        self.analytics = create_routing_analytics(self.logger)
        
        await self.logger.start_async_logging()
        
        print(f"\\n✅ Logging system configured!")
        print(f"   Level: {log_level.value}")
        print(f"   Storage: {storage_strategy.value}")
        print(f"   Batch size: {batch_size}")
        print(f"   Analytics: {'Enabled' if enable_analytics else 'Disabled'}")
        print(f"   Anonymization: {'Enabled' if anonymize else 'Disabled'}")
        print(f"   Log directory: {config.log_directory}")
    
    async def _basic_logging_demo(self):
        """Demonstrate basic logging functionality"""
        print("\\n📝 BASIC LOGGING DEMONSTRATION")
        print("-" * 40)
        
        if not self.logger:
            print("❌ Logging system not configured. Please run configuration first.")
            return
        
        print("Generating sample routing decisions...")
        
        # Select query category
        print("\\nQuery categories available:")
        for i, category in enumerate(self.sample_queries.keys(), 1):
            print(f"   {i}) {category.replace('_', ' ').title()}")
        
        category_choice = input("\\nSelect category (1-4) [1]: ").strip() or "1"
        categories = list(self.sample_queries.keys())
        selected_category = categories[int(category_choice) - 1] if category_choice.isdigit() and 1 <= int(category_choice) <= 4 else categories[0]
        
        queries = self.sample_queries[selected_category]
        num_queries = int(input(f"Number of queries to generate [5]: ").strip() or "5")
        num_queries = min(num_queries, len(queries))
        
        print(f"\\n🚀 Logging {num_queries} routing decisions...")
        print()
        
        # Generate and log decisions
        for i in range(num_queries):
            query_text = queries[i % len(queries)]
            
            # Simulate routing decision
            prediction = self._create_mock_routing_prediction(query_text, selected_category)
            
            # Create processing metrics
            processing_metrics = ProcessingMetrics(
                decision_time_ms=15.0 + (i * 2.5),
                total_time_ms=45.0 + (i * 8.2),
                backend_selection_time_ms=3.1 + (i * 0.5),
                query_complexity=self._calculate_query_complexity(query_text),
                memory_usage_mb=128.5 + (i * 2.1),
                cpu_usage_percent=25.0 + (i * 1.5)
            )
            
            # Create system state
            system_state = SystemState(
                backend_health={'lightrag': True, 'perplexity': True},
                backend_load={'lightrag': 0.3 + (i * 0.1), 'perplexity': 0.2 + (i * 0.05)},
                resource_usage={'cpu_percent': 25.0 + (i * 1.5), 'memory_percent': 45.0 + (i * 0.8)},
                selection_algorithm='weighted_round_robin',
                load_balancer_metrics={'active_backends': 2, 'total_requests': i + 1},
                backend_weights={'lightrag': 0.7, 'perplexity': 0.3},
                request_counter=i + 1,
                session_id=self.demo_session_id,
                deployment_mode='demo'
            )
            
            # Log the decision
            await self.logger.log_routing_decision(
                prediction, query_text, processing_metrics, system_state, self.demo_session_id
            )
            
            # Record in analytics
            if self.analytics:
                log_entry = RoutingDecisionLogEntry.from_routing_prediction(
                    prediction, query_text, processing_metrics, system_state, 
                    self.logger.config, self.demo_session_id
                )
                self.analytics.record_decision_metrics(log_entry)
            
            # Display progress
            backend = getattr(prediction, 'backend_selected', prediction.routing_decision.value)
            print(f"   {i+1:2d}. {query_text[:60]}... → {backend} ({prediction.confidence_metrics.overall_confidence:.3f})")
            
            self.logged_decisions += 1
            
            # Small delay for realism
            await asyncio.sleep(0.1)
        
        # Wait for batching
        print("\\n⏳ Waiting for batch processing...")
        await asyncio.sleep(2)
        
        # Show statistics
        stats = self.logger.get_statistics()
        print(f"\\n📊 Logging Statistics:")
        print(f"   Total logged: {stats['total_logged']}")
        print(f"   Total batches: {stats['total_batches']}")
        print(f"   Memory entries: {stats['memory_entries']}")
        print(f"   Average logging time: {stats['avg_logging_time_ms']:.2f}ms")
        print(f"   Queue full errors: {stats['queue_full_errors']}")
        
        print(f"\\n✅ Successfully logged {num_queries} routing decisions!")
    
    async def _analytics_dashboard_demo(self):
        """Show real-time analytics dashboard"""
        print("\\n📊 REAL-TIME ANALYTICS DASHBOARD")
        print("-" * 40)
        
        if not self.analytics:
            print("❌ Analytics system not available. Please configure logging first.")
            return
        
        # Get current metrics
        metrics = self.analytics.get_real_time_metrics()
        
        print(f"\\n🔢 Current Metrics (Session: {self.demo_session_id})")
        print("=" * 50)
        
        # Basic metrics
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Average Confidence: {metrics['avg_confidence']:.3f}")
        print(f"Average Decision Time: {metrics['avg_decision_time_ms']:.1f}ms")
        print(f"Max Decision Time: {metrics['max_decision_time_ms']:.1f}ms")
        print(f"P95 Decision Time: {metrics['p95_decision_time_ms']:.1f}ms")
        
        print(f"\\nError Rate: {metrics['error_rate_percent']:.1f}%")
        print(f"Fallback Rate: {metrics['fallback_rate_percent']:.1f}%")
        print(f"Anomaly Score: {metrics['anomaly_score']:.3f}")
        print(f"Detected Anomalies: {metrics['detected_anomalies']}")
        
        # Backend distribution
        print("\\n🎯 Backend Distribution:")
        for backend, count in metrics['requests_per_backend'].items():
            percentage = (count / max(metrics['total_requests'], 1)) * 100
            print(f"   {backend}: {count} requests ({percentage:.1f}%)")
        
        # Confidence distribution
        print("\\n📈 Confidence Distribution:")
        for bucket, count in metrics['confidence_distribution'].items():
            percentage = (count / max(metrics['total_requests'], 1)) * 100
            bar_length = int(percentage / 5)  # Scale for display
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {bucket}: {bar} {percentage:5.1f}% ({count})")
        
        # Backend availability
        if metrics['backend_availability']:
            print("\\n💚 Backend Availability:")
            for backend, availability in metrics['backend_availability'].items():
                status = "🟢" if availability > 0.9 else "🟡" if availability > 0.7 else "🔴"
                print(f"   {backend}: {status} {availability:.1%}")
        
        # Trend analysis
        print("\\n📉 Trends:")
        if len(metrics['hourly_request_trend']) > 1:
            recent_trend = metrics['hourly_request_trend'][-2:]
            if recent_trend[1][1] > recent_trend[0][1]:
                print("   Request Volume: 📈 Increasing")
            elif recent_trend[1][1] < recent_trend[0][1]:
                print("   Request Volume: 📉 Decreasing")
            else:
                print("   Request Volume: ➡️  Stable")
        else:
            print("   Request Volume: ➡️  Insufficient data")
        
        if len(metrics['confidence_trend']) > 1:
            if metrics['confidence_trend'][-1][1] > metrics['confidence_trend'][0][1]:
                print("   Confidence: 📈 Improving")
            elif metrics['confidence_trend'][-1][1] < metrics['confidence_trend'][0][1]:
                print("   Confidence: 📉 Degrading")
            else:
                print("   Confidence: ➡️  Stable")
        else:
            print("   Confidence: ➡️  Insufficient data")
        
        # Real-time update option
        update_choice = input("\\nShow live updates? (y/n) [n]: ").strip().lower()
        if update_choice == 'y':
            await self._live_analytics_updates()
    
    async def _live_analytics_updates(self):
        """Show live analytics updates"""
        print("\\n🔴 LIVE ANALYTICS (Press Ctrl+C to stop)")
        print("-" * 40)
        
        try:
            update_count = 0
            while update_count < 10:  # Limit for demo
                # Generate a new routing decision
                query = "Live demo query for real-time analytics"
                prediction = self._create_mock_routing_prediction(query, 'biomedical_research')
                
                processing_metrics = ProcessingMetrics(
                    decision_time_ms=20.0 + (update_count * 3.0),
                    total_time_ms=50.0 + (update_count * 5.0),
                    query_complexity=2.0 + (update_count * 0.2)
                )
                
                system_state = SystemState(
                    backend_health={'lightrag': True, 'perplexity': True},
                    request_counter=self.logged_decisions + update_count + 1,
                    session_id=self.demo_session_id
                )
                
                # Log and record
                await self.logger.log_routing_decision(
                    prediction, query, processing_metrics, system_state, self.demo_session_id
                )
                
                log_entry = RoutingDecisionLogEntry.from_routing_prediction(
                    prediction, query, processing_metrics, system_state, 
                    self.logger.config, self.demo_session_id
                )
                self.analytics.record_decision_metrics(log_entry)
                
                # Show updated metrics
                metrics = self.analytics.get_real_time_metrics()
                print(f"\\rRequests: {metrics['total_requests']:3d} | "
                      f"Avg Confidence: {metrics['avg_confidence']:.3f} | "
                      f"Avg Time: {metrics['avg_decision_time_ms']:5.1f}ms | "
                      f"Anomalies: {metrics['detected_anomalies']:2d}", end="", flush=True)
                
                update_count += 1
                await asyncio.sleep(1)
            
            print("\\n\\n✅ Live updates completed!")
            
        except KeyboardInterrupt:
            print("\\n\\n⏹️  Live updates stopped by user")
    
    async def _anomaly_detection_demo(self):
        """Demonstrate anomaly detection capabilities"""
        print("\\n🚨 ANOMALY DETECTION DEMONSTRATION")
        print("-" * 40)
        
        if not self.analytics:
            print("❌ Analytics system not available. Please configure logging first.")
            return
        
        print("This demo will generate various scenarios to trigger anomaly detection.")
        print()
        
        scenarios = [
            ("Normal Operation", self._generate_normal_decisions),
            ("Low Confidence Spike", self._generate_low_confidence_decisions),
            ("High Latency Spike", self._generate_slow_decisions),
            ("Error Rate Spike", self._generate_error_decisions),
            ("Backend Monopolization", self._generate_monopolized_decisions)
        ]
        
        print("Available scenarios:")
        for i, (name, _) in enumerate(scenarios, 1):
            print(f"   {i}) {name}")
        
        choice = input("\\nSelect scenario (1-5) [2]: ").strip() or "2"
        scenario_index = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= 5 else 1
        
        scenario_name, generator_func = scenarios[scenario_index]
        print(f"\\n🎭 Running scenario: {scenario_name}")
        
        # Generate baseline data first
        print("   Generating baseline data...")
        await self._generate_normal_decisions(10)
        
        # Wait and check baseline
        await asyncio.sleep(1)
        initial_anomalies = self.analytics.detect_anomalies(lookback_hours=1)
        print(f"   Baseline anomalies: {len(initial_anomalies)}")
        
        # Generate scenario-specific data
        print(f"   Generating {scenario_name.lower()} data...")
        await generator_func(15)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Detect anomalies
        print("\\n🔍 Analyzing for anomalies...")
        anomalies = self.analytics.detect_anomalies(lookback_hours=1, sensitivity=1.0)
        
        if anomalies:
            print(f"\\n⚠️  Detected {len(anomalies)} anomalies:")
            for i, anomaly in enumerate(anomalies, 1):
                severity_icon = {"warning": "🟡", "critical": "🔴", "error": "⚫"}.get(anomaly.get('severity', 'warning'), "🟡")
                print(f"\\n   {i}. {severity_icon} {anomaly['type'].upper()}")
                print(f"      {anomaly['description']}")
                if 'details' in anomaly:
                    for key, value in anomaly['details'].items():
                        print(f"      {key}: {value}")
                print(f"      Affected requests: {anomaly.get('affected_requests', 'N/A')}")
        else:
            print("\\n✅ No anomalies detected (detection may need more data or higher sensitivity)")
        
        # Show anomaly report
        anomaly_report = self.analytics.detect_anomalies()
        print(f"\\n📋 Anomaly Detection Summary:")
        print(f"   Total anomalies found: {len(anomaly_report)}")
        
        if anomaly_report:
            anomaly_types = {}
            for anomaly in anomaly_report:
                anomaly_type = anomaly.get('type', 'unknown')
                anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            print("   Breakdown by type:")
            for atype, count in anomaly_types.items():
                print(f"     - {atype}: {count}")
    
    async def _generate_normal_decisions(self, count: int):
        """Generate normal routing decisions for baseline"""
        for i in range(count):
            query = f"Normal query {i+1}: metabolic pathway analysis"
            
            # Normal confidence range
            confidence = 0.7 + (i % 3) * 0.1  # 0.7-0.9 range
            prediction = self._create_mock_routing_prediction(query, 'biomedical_research', confidence)
            
            # Normal processing times
            processing_metrics = ProcessingMetrics(
                decision_time_ms=15.0 + (i % 5) * 2.0,  # 15-23ms
                total_time_ms=45.0 + (i % 5) * 5.0,     # 45-65ms
                query_complexity=2.0 + (i % 3) * 0.5
            )
            
            system_state = SystemState(
                backend_health={'lightrag': True, 'perplexity': True},
                request_counter=self.logged_decisions + i + 1,
                session_id=self.demo_session_id
            )
            
            await self._log_and_record(prediction, query, processing_metrics, system_state)
            await asyncio.sleep(0.05)  # Small delay
    
    async def _generate_low_confidence_decisions(self, count: int):
        """Generate decisions with unusually low confidence"""
        for i in range(count):
            query = f"Low confidence query {i+1}: ambiguous medical terminology"
            
            # Abnormally low confidence
            confidence = 0.2 + (i % 3) * 0.1  # 0.2-0.4 range
            prediction = self._create_mock_routing_prediction(query, 'general_health', confidence)
            
            processing_metrics = ProcessingMetrics(
                decision_time_ms=18.0 + (i % 3) * 2.0,
                total_time_ms=48.0 + (i % 3) * 4.0,
                query_complexity=3.0 + (i % 2) * 0.5
            )
            
            system_state = SystemState(
                backend_health={'lightrag': True, 'perplexity': True},
                request_counter=self.logged_decisions + i + 1,
                session_id=self.demo_session_id
            )
            
            await self._log_and_record(prediction, query, processing_metrics, system_state)
            await asyncio.sleep(0.05)
    
    async def _generate_slow_decisions(self, count: int):
        """Generate decisions with high latency"""
        for i in range(count):
            query = f"Slow query {i+1}: complex computational analysis"
            
            confidence = 0.75 + (i % 2) * 0.1
            prediction = self._create_mock_routing_prediction(query, 'complex_research', confidence)
            
            # Abnormally high processing times
            processing_metrics = ProcessingMetrics(
                decision_time_ms=150.0 + i * 50.0,  # 150ms+ (very slow)
                total_time_ms=300.0 + i * 100.0,    # 300ms+ (very slow)
                query_complexity=4.0 + (i % 2) * 0.5
            )
            
            system_state = SystemState(
                backend_health={'lightrag': True, 'perplexity': True},
                request_counter=self.logged_decisions + i + 1,
                session_id=self.demo_session_id
            )
            
            await self._log_and_record(prediction, query, processing_metrics, system_state)
            await asyncio.sleep(0.05)
    
    async def _generate_error_decisions(self, count: int):
        """Generate decisions with system errors"""
        for i in range(count):
            query = f"Error-prone query {i+1}: system failure scenario"
            
            confidence = 0.6 + (i % 3) * 0.1
            prediction = self._create_mock_routing_prediction(query, 'biomedical_research', confidence)
            
            processing_metrics = ProcessingMetrics(
                decision_time_ms=25.0 + i * 3.0,
                total_time_ms=60.0 + i * 8.0,
                query_complexity=2.5
            )
            
            # Simulate system errors
            system_state = SystemState(
                backend_health={'lightrag': i % 3 != 0, 'perplexity': i % 4 != 0},  # Intermittent failures
                errors=[f"Backend connection timeout", f"Rate limit exceeded"] if i % 2 == 0 else [],
                fallback_used=i % 3 == 0,
                fallback_reason="Primary backend unavailable" if i % 3 == 0 else None,
                request_counter=self.logged_decisions + i + 1,
                session_id=self.demo_session_id
            )
            
            await self._log_and_record(prediction, query, processing_metrics, system_state)
            await asyncio.sleep(0.05)
    
    async def _generate_monopolized_decisions(self, count: int):
        """Generate decisions that heavily favor one backend"""
        for i in range(count):
            query = f"Monopolized query {i+1}: heavily biased routing"
            
            confidence = 0.8 + (i % 2) * 0.05
            # Force all decisions to go to one backend (simulate load balancer issue)
            prediction = RoutingPrediction(
                routing_decision=RoutingDecision.LIGHTRAG,  # Always LIGHTRAG
                confidence_metrics=self._create_mock_confidence_metrics(confidence),
                reasoning=[f"Forced routing to LIGHTRAG (monopolization test)"],
                research_category="forced_category"
            )
            
            # Add backend_selected attribute
            prediction.backend_selected = "lightrag"
            
            processing_metrics = ProcessingMetrics(
                decision_time_ms=18.0 + (i % 3) * 2.0,
                total_time_ms=50.0 + (i % 3) * 5.0,
                query_complexity=2.2
            )
            
            system_state = SystemState(
                backend_health={'lightrag': True, 'perplexity': True},
                backend_weights={'lightrag': 0.95, 'perplexity': 0.05},  # Heavily skewed
                selection_algorithm='biased_selection',
                request_counter=self.logged_decisions + i + 1,
                session_id=self.demo_session_id
            )
            
            await self._log_and_record(prediction, query, processing_metrics, system_state)
            await asyncio.sleep(0.05)
    
    async def _log_and_record(self, prediction, query, processing_metrics, system_state):
        """Helper method to log and record a decision"""
        await self.logger.log_routing_decision(
            prediction, query, processing_metrics, system_state, self.demo_session_id
        )
        
        log_entry = RoutingDecisionLogEntry.from_routing_prediction(
            prediction, query, processing_metrics, system_state, 
            self.logger.config, self.demo_session_id
        )
        self.analytics.record_decision_metrics(log_entry)
        
        self.logged_decisions += 1
    
    async def _performance_monitoring_demo(self):
        """Demonstrate performance monitoring capabilities"""
        print("\\n⚡ PERFORMANCE MONITORING DEMONSTRATION")
        print("-" * 40)
        
        if not self.logger:
            print("❌ Logging system not available. Please configure logging first.")
            return
        
        print("Performance monitoring tracks logging overhead and system impact.")
        print("\\nGenerating load to demonstrate performance monitoring...")
        
        # Generate different load patterns
        load_patterns = [
            ("Low Load", 5, 0.2),
            ("Medium Load", 20, 0.1),
            ("High Load", 50, 0.05),
            ("Burst Load", 100, 0.01)
        ]
        
        print("\\nLoad patterns:")
        for i, (name, count, delay) in enumerate(load_patterns, 1):
            print(f"   {i}) {name}: {count} requests, {delay}s intervals")
        
        choice = input("\\nSelect load pattern (1-4) [2]: ").strip() or "2"
        pattern_index = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= 4 else 1
        
        pattern_name, request_count, delay = load_patterns[pattern_index]
        
        print(f"\\n🚀 Running {pattern_name} test...")
        
        # Record initial stats
        initial_stats = self.logger.get_statistics()
        start_time = time.time()
        
        # Generate load
        for i in range(request_count):
            query = f"{pattern_name} test query {i+1}"
            prediction = self._create_mock_routing_prediction(query, 'biomedical_research')
            
            processing_metrics = ProcessingMetrics(
                decision_time_ms=15.0 + (i % 5) * 1.0,
                total_time_ms=45.0 + (i % 5) * 2.0,
                query_complexity=2.0
            )
            
            system_state = SystemState(
                request_counter=self.logged_decisions + i + 1,
                session_id=self.demo_session_id
            )
            
            # Measure logging time
            log_start = time.time()
            await self.logger.log_routing_decision(
                prediction, query, processing_metrics, system_state, self.demo_session_id
            )
            log_time = (time.time() - log_start) * 1000
            
            # Show progress every 10 requests
            if (i + 1) % max(1, request_count // 10) == 0:
                progress = (i + 1) / request_count * 100
                print(f"   Progress: {progress:5.1f}% ({i+1:3d}/{request_count}) - Last log time: {log_time:.2f}ms")
            
            self.logged_decisions += 1
            await asyncio.sleep(delay)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Show performance results
        end_time = time.time()
        final_stats = self.logger.get_statistics()
        total_duration = end_time - start_time
        
        print(f"\\n📊 Performance Results for {pattern_name}:")
        print("=" * 50)
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Requests Generated: {request_count}")
        print(f"Throughput: {request_count / total_duration:.1f} requests/second")
        print()
        
        print("Logging Performance:")
        print(f"   Average logging time: {final_stats['avg_logging_time_ms']:.2f}ms")
        print(f"   Maximum logging time: {final_stats['max_logging_time_ms']:.2f}ms")
        print(f"   Total batches processed: {final_stats['total_batches'] - initial_stats['total_batches']}")
        print(f"   Queue full errors: {final_stats['queue_full_errors'] - initial_stats['queue_full_errors']}")
        print(f"   Current memory entries: {final_stats['memory_entries']}")
        print(f"   Current queue size: {final_stats['queue_size']}")
        
        # Performance assessment
        avg_overhead = final_stats['avg_logging_time_ms']
        if avg_overhead < 5.0:
            print(f"\\n✅ Performance: Excellent (< 5ms average overhead)")
        elif avg_overhead < 10.0:
            print(f"\\n🟡 Performance: Good (< 10ms average overhead)")
        elif avg_overhead < 20.0:
            print(f"\\n🟠 Performance: Fair (< 20ms average overhead)")
        else:
            print(f"\\n🔴 Performance: Poor (> 20ms average overhead)")
        
        # Resource usage simulation
        import psutil
        print(f"\\nSystem Resource Usage:")
        print(f"   CPU: {psutil.cpu_percent(interval=0.1):.1f}%")
        print(f"   Memory: {psutil.virtual_memory().percent:.1f}%")
        print(f"   Available Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    async def _enhanced_router_demo(self):
        """Demonstrate enhanced router integration"""
        print("\\n🚀 ENHANCED ROUTER INTEGRATION")
        print("-" * 40)
        
        if not ENHANCED_ROUTER_AVAILABLE:
            print("❌ Enhanced router not available in this environment.")
            print("   This demo would show:")
            print("   • Seamless integration with production router")
            print("   • Automatic logging of all routing decisions")
            print("   • Enhanced analytics and monitoring")
            print("   • Production-ready deployment options")
            return
        
        print("Creating enhanced router with integrated logging...")
        
        # Create enhanced router
        feature_flags = EnhancedFeatureFlags(
            enable_production_load_balancer=False,  # Simplified for demo
            enable_routing_logging=True,
            routing_log_level=LogLevel.DETAILED,
            enable_real_time_analytics=True,
            enable_anomaly_detection=True
        )
        
        try:
            self.enhanced_router = create_enhanced_production_router(
                enable_production=False,
                enable_logging=True,
                log_level="detailed"
            )
            
            print("✅ Enhanced router created successfully!")
            
            # Start monitoring
            await self.enhanced_router.start_monitoring()
            print("✅ Monitoring started")
            
            # Test routing with integrated logging
            print("\\n📝 Testing integrated routing and logging...")
            
            test_queries = [
                "What are the latest developments in metabolomics for cancer research?",
                "How do genetic variations affect drug metabolism in personalized medicine?",
                "Clinical applications of biomarkers in diabetes management"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\\n   Query {i}: {query[:50]}...")
                
                try:
                    # Route query (this automatically logs the decision)
                    prediction = await self.enhanced_router.route_query(query)
                    
                    print(f"      → Routed to: {prediction.routing_decision.value}")
                    print(f"      → Confidence: {prediction.confidence_metrics.overall_confidence:.3f}")
                    print(f"      → Backend: {getattr(prediction, 'backend_selected', 'N/A')}")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                
                await asyncio.sleep(0.5)
            
            # Show enhanced analytics
            print("\\n📊 Enhanced Router Analytics:")
            analytics_summary = self.enhanced_router.get_routing_analytics_summary()
            
            if 'routing_analytics' in analytics_summary:
                ra = analytics_summary['routing_analytics']
                print(f"   Status: {ra.get('status', 'N/A')}")
                print(f"   Total logged decisions: {ra.get('total_logged_decisions', 0)}")
                
                if 'logging_performance' in ra:
                    lp = ra['logging_performance']
                    print(f"   Average logging overhead: {lp.get('avg_overhead_ms', 0):.2f}ms")
                    print(f"   Logging health: {'✅' if lp.get('overhead_healthy', True) else '❌'}")
                
                print(f"   Detected anomalies: {ra.get('detected_anomalies', 0)}")
            
            # Health status
            print("\\n🏥 System Health:")
            health_status = self.enhanced_router.get_health_status()
            if 'routing_analytics' in health_status:
                rh = health_status['routing_analytics']
                print(f"   Logging enabled: {'✅' if rh.get('logging_enabled') else '❌'}")
                print(f"   Analytics enabled: {'✅' if rh.get('analytics_enabled') else '❌'}")
                print(f"   Total decisions: {rh.get('total_logged_decisions', 0)}")
            
            # Stop monitoring
            await self.enhanced_router.stop_monitoring()
            print("\\n✅ Enhanced router demo completed!")
            
        except Exception as e:
            print(f"❌ Enhanced router demo failed: {e}")
    
    async def _export_and_reporting_demo(self):
        """Demonstrate export and reporting functionality"""
        print("\\n📤 EXPORT & REPORTING DEMONSTRATION")
        print("-" * 40)
        
        if not self.analytics:
            print("❌ Analytics system not available. Please configure logging first.")
            return
        
        print("This demo shows various export and reporting options.")
        
        # Generate some data if needed
        if self.logged_decisions < 10:
            print("\\nGenerating sample data for reporting...")
            await self._generate_normal_decisions(10)
            await asyncio.sleep(1)
        
        # Analytics report
        print("\\n1. 📋 Analytics Report Generation")
        print("-" * 30)
        
        try:
            report = self.analytics.generate_analytics_report(lookback_hours=24)
            
            print(f"Generated comprehensive analytics report:")
            print(f"   Time period: {report.time_period_hours} hours")
            print(f"   Total requests: {report.total_requests}")
            print(f"   Unique sessions: {report.unique_sessions}")
            print(f"   Average confidence: {report.avg_confidence:.3f}")
            print(f"   Average response time: {report.avg_response_time_ms:.1f}ms")
            print(f"   Success rate: {report.success_rate_percent:.1f}%")
            print(f"   Detected anomalies: {len(report.detected_anomalies)}")
            print(f"   Request trend: {report.request_volume_trend}")
            print(f"   Performance trend: {report.performance_trend}")
            print(f"   Confidence trend: {report.confidence_trend}")
            
            print(f"\\n   Backend Distribution:")
            for backend, percentage in report.backend_distribution.items():
                print(f"     {backend}: {percentage:.1f}%")
            
            print(f"\\n   Recommendations ({len(report.recommendations)}):")
            for i, rec in enumerate(report.recommendations[:3], 1):  # Show first 3
                print(f"     {i}. {rec}")
            
        except Exception as e:
            print(f"   ❌ Report generation failed: {e}")
        
        # Data export
        print("\\n2. 📁 Data Export Options")
        print("-" * 30)
        
        export_options = [
            ("JSON Analytics Export", "analytics"),
            ("CSV Summary Export", "csv"),
            ("Log File Export", "logs")
        ]
        
        print("Export formats:")
        for i, (name, format_type) in enumerate(export_options, 1):
            print(f"   {i}) {name}")
        
        choice = input("\\nSelect export format (1-3) [1]: ").strip() or "1"
        
        try:
            if choice == "1":
                # JSON Analytics Export
                export_file = self.analytics.export_analytics()
                print(f"   ✅ Analytics exported to: {export_file}")
                
                # Show file info
                if os.path.exists(export_file):
                    file_size = os.path.getsize(export_file) / 1024
                    print(f"   📁 File size: {file_size:.1f} KB")
                    
                    # Show sample content
                    with open(export_file, 'r') as f:
                        sample = json.load(f)
                    
                    print(f"   📄 Export contains:")
                    print(f"     • Export timestamp: {sample.get('export_timestamp', 'N/A')}")
                    print(f"     • Total entries: {sample.get('total_entries', 0)}")
                    print(f"     • Raw entries: {len(sample.get('raw_entries', []))}")
                    print(f"     • Analytics report: {'✅' if 'analytics_report' in sample else '❌'}")
                
            elif choice == "2":
                # CSV Export (simulated)
                print("   📊 CSV export would include:")
                print("     • Timestamp, Query Hash, Routing Decision, Confidence")
                print("     • Processing Metrics, Backend Selected, Errors")
                print("     • System State Summary, Session ID")
                print("   📁 Format: routing_decisions_export_YYYYMMDD_HHMMSS.csv")
                
            elif choice == "3":
                # Log File Export
                stats = self.logger.get_statistics()
                current_log_file = stats.get('current_log_file')
                
                if current_log_file and os.path.exists(current_log_file):
                    file_size = os.path.getsize(current_log_file) / 1024
                    print(f"   ✅ Current log file: {current_log_file}")
                    print(f"   📁 File size: {file_size:.1f} KB")
                    
                    # Show sample log entries
                    print("   📄 Sample log entries:")
                    with open(current_log_file, 'r') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines[:2]):  # Show first 2 lines
                            try:
                                entry = json.loads(line)
                                print(f"     Entry {i+1}: {len(entry.get('entries', []))} routing decisions")
                            except:
                                print(f"     Line {i+1}: {line[:60]}...")
                else:
                    print("   ❌ No log files available yet")
            
        except Exception as e:
            print(f"   ❌ Export failed: {e}")
        
        # Comprehensive export (if enhanced router available)
        if hasattr(self, 'enhanced_router') and self.enhanced_router:
            print("\\n3. 🚀 Enhanced Router Comprehensive Export")
            print("-" * 30)
            
            try:
                comprehensive_file = self.enhanced_router.export_comprehensive_analytics()
                print(f"   ✅ Comprehensive export to: {comprehensive_file}")
                
                if os.path.exists(comprehensive_file):
                    file_size = os.path.getsize(comprehensive_file) / 1024
                    print(f"   📁 File size: {file_size:.1f} KB")
                    print(f"   📄 Includes: routing analytics, performance data, anomaly reports")
                
            except Exception as e:
                print(f"   ❌ Comprehensive export failed: {e}")
    
    async def _configuration_examples_demo(self):
        """Show configuration examples for different scenarios"""
        print("\\n⚙️  CONFIGURATION EXAMPLES")
        print("-" * 40)
        
        examples = [
            ("Development Environment", {
                "enabled": True,
                "log_level": "debug",
                "storage_strategy": "memory",
                "batch_size": 10,
                "anonymize_queries": False,
                "enable_real_time_analytics": True,
                "retention_days": 7
            }),
            ("Staging Environment", {
                "enabled": True,
                "log_level": "detailed",
                "storage_strategy": "hybrid",
                "batch_size": 25,
                "anonymize_queries": False,
                "enable_compression": True,
                "retention_days": 30
            }),
            ("Production Environment", {
                "enabled": True,
                "log_level": "standard",
                "storage_strategy": "file",
                "batch_size": 100,
                "anonymize_queries": True,
                "hash_sensitive_data": True,
                "enable_compression": True,
                "retention_days": 90,
                "max_file_size_mb": 500
            }),
            ("High-Performance Production", {
                "enabled": True,
                "log_level": "minimal",
                "storage_strategy": "streaming",
                "batch_size": 200,
                "anonymize_queries": True,
                "max_queue_size": 50000,
                "batch_timeout_seconds": 1.0,
                "enable_performance_tracking": True
            })
        ]
        
        print("Configuration examples for different environments:")
        print()
        
        for i, (env_name, config) in enumerate(examples, 1):
            print(f"{i}. 🏷️  {env_name}")
            print(f"   Environment: {env_name.lower().replace(' ', '_')}")
            print("   Configuration:")
            
            for key, value in config.items():
                value_str = json.dumps(value) if isinstance(value, (bool, str)) else str(value)
                print(f"     {key}: {value_str}")
            
            print(f"   Use case: {self._get_use_case_description(env_name)}")
            print()
        
        # Environment variables example
        print("🌐 Environment Variables Example (Production):")
        print("-" * 40)
        env_vars = [
            "ROUTING_LOGGING_ENABLED=true",
            "ROUTING_LOG_LEVEL=standard",
            "ROUTING_STORAGE_STRATEGY=file",
            "ROUTING_LOG_DIRECTORY=/var/log/cmo/routing_decisions",
            "ROUTING_BATCH_SIZE=100",
            "ROUTING_ANONYMIZE_QUERIES=true",
            "ROUTING_HASH_SENSITIVE_DATA=true",
            "ROUTING_RETENTION_DAYS=90",
            "ROUTING_ENABLE_COMPRESSION=true",
            "ROUTING_REAL_TIME_ANALYTICS=true",
            "ROUTING_MAX_LOGGING_OVERHEAD_MS=5.0"
        ]
        
        for var in env_vars:
            print(f"export {var}")
        
        print()
        
        # Docker compose example
        print("🐳 Docker Compose Configuration:")
        print("-" * 40)
        docker_config = '''
version: '3.8'
services:
  cmo-router:
    image: cmo/enhanced-router:latest
    environment:
      - ROUTING_LOGGING_ENABLED=true
      - ROUTING_LOG_LEVEL=standard
      - ROUTING_STORAGE_STRATEGY=hybrid
      - ROUTING_LOG_DIRECTORY=/app/logs/routing
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    ports:
      - "8080:8080"
'''
        print(docker_config)
        
        # Kubernetes example
        print("☸️  Kubernetes ConfigMap:")
        print("-" * 40)
        k8s_config = '''
apiVersion: v1
kind: ConfigMap
metadata:
  name: routing-logging-config
data:
  ROUTING_LOGGING_ENABLED: "true"
  ROUTING_LOG_LEVEL: "standard"
  ROUTING_STORAGE_STRATEGY: "file"
  ROUTING_BATCH_SIZE: "100"
  ROUTING_ANONYMIZE_QUERIES: "true"
  ROUTING_RETENTION_DAYS: "90"
'''
        print(k8s_config)
    
    def _get_use_case_description(self, env_name: str) -> str:
        """Get use case description for environment"""
        descriptions = {
            "Development Environment": "Local development, debugging, full visibility",
            "Staging Environment": "Pre-production testing, performance validation",
            "Production Environment": "Live production, privacy-compliant, optimized",
            "High-Performance Production": "High-volume production, minimal overhead"
        }
        return descriptions.get(env_name, "General purpose configuration")
    
    async def _view_current_status(self):
        """Show current system status"""
        print("\\n📋 CURRENT SYSTEM STATUS")
        print("-" * 40)
        
        # Session information
        uptime = datetime.now() - self.demo_start_time
        print(f"📅 Session Information:")
        print(f"   Session ID: {self.demo_session_id}")
        print(f"   Started: {self.demo_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Uptime: {uptime}")
        print(f"   Logged decisions: {self.logged_decisions}")
        
        # Logging system status
        print(f"\\n📝 Logging System:")
        if self.logger:
            config = self.logger.config
            stats = self.logger.get_statistics()
            
            print(f"   Status: ✅ Configured and running")
            print(f"   Level: {config.log_level.value}")
            print(f"   Storage: {config.storage_strategy.value}")
            print(f"   Directory: {config.log_directory}")
            print(f"   Batch size: {config.batch_size}")
            print(f"   Memory entries: {stats['memory_entries']}")
            print(f"   Total logged: {stats['total_logged']}")
            print(f"   Total batches: {stats['total_batches']}")
            print(f"   Avg logging time: {stats['avg_logging_time_ms']:.2f}ms")
            print(f"   Queue errors: {stats['queue_full_errors']}")
            print(f"   Running: {'✅' if stats['is_running'] else '❌'}")
        else:
            print(f"   Status: ❌ Not configured")
        
        # Analytics system status
        print(f"\\n📊 Analytics System:")
        if self.analytics:
            metrics = self.analytics.get_real_time_metrics()
            
            print(f"   Status: ✅ Active")
            print(f"   Total requests: {metrics['total_requests']}")
            print(f"   Average confidence: {metrics['avg_confidence']:.3f}")
            print(f"   Average decision time: {metrics['avg_decision_time_ms']:.1f}ms")
            print(f"   Error rate: {metrics['error_rate_percent']:.1f}%")
            print(f"   Anomaly score: {metrics['anomaly_score']:.3f}")
            print(f"   Detected anomalies: {metrics['detected_anomalies']}")
            
            # Backend distribution
            if metrics['requests_per_backend']:
                print(f"   Backend distribution:")
                for backend, count in metrics['requests_per_backend'].items():
                    percentage = (count / max(metrics['total_requests'], 1)) * 100
                    print(f"     {backend}: {count} ({percentage:.1f}%)")
        else:
            print(f"   Status: ❌ Not available")
        
        # Enhanced router status
        print(f"\\n🚀 Enhanced Router:")
        if hasattr(self, 'enhanced_router') and self.enhanced_router:
            print(f"   Status: ✅ Available and configured")
            try:
                health = self.enhanced_router.get_health_status()
                print(f"   Health: {'✅' if health else '⚠️ '}")
            except:
                print(f"   Health: ⚠️  Unable to check")
        else:
            print(f"   Status: {'❌ Not available' if not ENHANCED_ROUTER_AVAILABLE else '⚠️  Not configured'}")
        
        # File system status
        print(f"\\n📁 File System:")
        if self.logger and self.logger.config.storage_strategy in [StorageStrategy.FILE, StorageStrategy.HYBRID]:
            log_dir = Path(self.logger.config.log_directory)
            if log_dir.exists():
                files = list(log_dir.glob("routing_decisions_*.jsonl*"))
                total_size = sum(f.stat().st_size for f in files) / 1024 / 1024  # MB
                
                print(f"   Log directory: {log_dir}")
                print(f"   Log files: {len(files)}")
                print(f"   Total size: {total_size:.1f} MB")
                
                if files:
                    latest_file = max(files, key=lambda f: f.stat().st_mtime)
                    latest_size = latest_file.stat().st_size / 1024  # KB
                    print(f"   Latest file: {latest_file.name} ({latest_size:.1f} KB)")
            else:
                print(f"   Log directory: ❌ Does not exist")
        else:
            print(f"   File logging: ❌ Not configured")
        
        # System resources
        try:
            import psutil
            print(f"\\n💻 System Resources:")
            print(f"   CPU usage: {psutil.cpu_percent(interval=0.1):.1f}%")
            print(f"   Memory usage: {psutil.virtual_memory().percent:.1f}%")
            print(f"   Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
            print(f"   Disk usage: {psutil.disk_usage('/').percent:.1f}%")
        except ImportError:
            print(f"\\n💻 System Resources: ⚠️  psutil not available")
    
    async def _cleanup_and_exit(self):
        """Clean up resources and exit"""
        print("\\n🧹 CLEANUP AND EXIT")
        print("-" * 40)
        
        cleanup_tasks = []
        
        # Stop logging system
        if self.logger:
            print("Stopping logging system...")
            cleanup_tasks.append(self.logger.stop_async_logging())
        
        # Stop enhanced router monitoring
        if hasattr(self, 'enhanced_router') and self.enhanced_router:
            print("Stopping enhanced router monitoring...")
            cleanup_tasks.append(self.enhanced_router.stop_monitoring())
        
        # Execute cleanup
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks)
                print("✅ All systems stopped cleanly")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
        
        # Show session summary
        session_duration = datetime.now() - self.demo_start_time
        print(f"\\n📊 Session Summary:")
        print(f"   Duration: {session_duration}")
        print(f"   Total decisions logged: {self.logged_decisions}")
        print(f"   Session ID: {self.demo_session_id}")
        
        # Show log files created
        if self.logger:
            log_dir = Path(self.logger.config.log_directory)
            if log_dir.exists():
                files = list(log_dir.glob("routing_decisions_*.jsonl*"))
                if files:
                    print(f"   Log files created: {len(files)}")
                    for f in files:
                        size_kb = f.stat().st_size / 1024
                        print(f"     {f.name} ({size_kb:.1f} KB)")
        
        print(f"\\n🎉 Thank you for using the Routing Decision Logging & Analytics Demo!")
        print(f"   All log files and exports are preserved for your review.")
    
    def _create_mock_routing_prediction(self, query_text: str, category: str, confidence: Optional[float] = None) -> RoutingPrediction:
        """Create a realistic mock routing prediction"""
        
        # Determine routing decision based on query category and content
        if category in ['biomedical_research', 'complex_research']:
            routing_decision = RoutingDecision.LIGHTRAG
            backend_selected = "lightrag"
            base_confidence = 0.8
        else:
            routing_decision = RoutingDecision.PERPLEXITY
            backend_selected = "perplexity"
            base_confidence = 0.7
        
        # Use provided confidence or calculate based on query characteristics
        if confidence is None:
            # Adjust confidence based on query complexity
            complexity_bonus = min(0.15, len(query_text.split()) / 100)
            confidence = base_confidence + complexity_bonus
            
            # Add some randomness
            import random
            confidence += random.uniform(-0.1, 0.1)
            confidence = max(0.1, min(0.95, confidence))  # Clamp to reasonable range
        
        # Create confidence metrics
        confidence_metrics = self._create_mock_confidence_metrics(confidence)
        
        # Create reasoning
        reasoning = [
            f"Query categorized as {category.replace('_', ' ')}",
            f"Confidence score: {confidence:.3f}",
            f"Selected backend: {backend_selected}"
        ]
        
        # Add complexity-based reasoning
        if len(query_text.split()) > 15:
            reasoning.append("Complex query detected - detailed analysis required")
        
        if any(term in query_text.lower() for term in ['latest', 'recent', 'new', 'current']):
            reasoning.append("Temporal query - recent information prioritized")
        
        # Create prediction
        prediction = RoutingPrediction(
            routing_decision=routing_decision,
            confidence_metrics=confidence_metrics,
            reasoning=reasoning,
            research_category=category
        )
        
        # Add backend selection info
        prediction.backend_selected = backend_selected
        
        return prediction
    
    def _create_mock_confidence_metrics(self, overall_confidence: float) -> ConfidenceMetrics:
        """Create realistic confidence metrics"""
        import random
        
        # Generate related confidence scores with some variance
        variance = 0.05
        
        return ConfidenceMetrics(
            overall_confidence=overall_confidence,
            research_category_confidence=max(0.0, min(1.0, overall_confidence + random.uniform(-variance, variance))),
            temporal_analysis_confidence=max(0.0, min(1.0, overall_confidence + random.uniform(-variance, variance))),
            signal_strength_confidence=max(0.0, min(1.0, overall_confidence - 0.05 + random.uniform(-variance, variance))),
            context_coherence_confidence=max(0.0, min(1.0, overall_confidence + 0.02 + random.uniform(-variance, variance))),
            keyword_density=random.uniform(0.6, 0.9),
            pattern_match_strength=overall_confidence * random.uniform(0.8, 1.1),
            biomedical_entity_count=random.randint(2, 8),
            ambiguity_score=1.0 - overall_confidence + random.uniform(-0.1, 0.1),
            conflict_score=random.uniform(0.0, 0.2),
            alternative_interpretations=[],
            calculation_time_ms=random.uniform(10.0, 25.0)
        )
    
    def _calculate_query_complexity(self, query_text: str) -> float:
        """Calculate query complexity score"""
        # Simple complexity calculation based on various factors
        word_count = len(query_text.split())
        char_count = len(query_text)
        
        # Base complexity from length
        complexity = min(5.0, word_count / 10.0)
        
        # Adjust for technical terms
        technical_terms = ['metabolomics', 'biomarker', 'pathway', 'metabolism', 'clinical', 'genomic', 'proteomic']
        tech_count = sum(1 for term in technical_terms if term.lower() in query_text.lower())
        complexity += tech_count * 0.5
        
        # Adjust for question complexity
        if '?' in query_text:
            complexity += 0.2
        if any(word in query_text.lower() for word in ['how', 'why', 'explain', 'analyze']):
            complexity += 0.3
        
        return min(5.0, complexity)


async def main():
    """Main entry point for the demo"""
    try:
        demo = InteractiveDemo()
        await demo.run()
    except KeyboardInterrupt:
        print("\\n\\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Routing Decision Logging & Analytics Demo...")
    asyncio.run(main())