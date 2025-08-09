# Routing Decision Logging and Analytics System Design

## Overview

This document outlines the comprehensive design for the Routing Decision Logging and Analytics system integrated into the Clinical Metabolomics Oracle (CMO) platform. The system provides detailed insights into routing patterns, performance metrics, anomaly detection, and system behavior analysis.

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Router Layer                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐  ┌─────────────────────────────┐  │
│  │ Enhanced Production  │  │    Routing Decision         │  │
│  │ Query Router         │──│    Logger                   │  │
│  └──────────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 Analytics and Storage Layer                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐  ┌─────────────────────────────┐  │
│  │ Routing Analytics    │  │    Storage Strategies       │  │
│  │ Engine              │  │    • Memory                 │  │
│  └──────────────────────┘  │    • File System           │  │
│                             │    • Hybrid                 │  │
│                             │    • Streaming              │  │
│                             └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Monitoring and Alerting                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐  ┌─────────────────────────────┐  │
│  │ Anomaly Detection    │  │    Performance Monitoring   │  │
│  │ System              │  │    Dashboard                │  │
│  └──────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Asynchronous Logging with Batching

**Design Principles:**
- Non-blocking logging operations
- Configurable batch sizes for optimal performance
- Queue-based architecture with overflow handling
- Automatic batch timeout for consistent data persistence

**Implementation:**
```python
class RoutingDecisionLogger:
    def __init__(self, config: LoggingConfig):
        self.log_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.batch_size = config.batch_size
        self.batch_timeout_seconds = config.batch_timeout_seconds
    
    async def log_routing_decision(self, prediction, query_text, metrics, state):
        # Non-blocking enqueue operation
        await self._queue_for_batch_write(log_entry)
```

### 2. Multiple Storage Strategies

#### Memory Storage
- **Use Case:** High-performance, volatile storage for real-time analytics
- **Capacity:** Configurable ring buffer (default: 10,000 entries)
- **Cleanup:** Automatic LRU eviction and periodic maintenance

#### File Storage
- **Use Case:** Persistent storage for historical analysis
- **Features:** 
  - Log rotation based on size limits
  - Compression support (gzip)
  - Automatic cleanup based on retention policies
- **Format:** JSONL (JSON Lines) for streaming compatibility

#### Hybrid Storage
- **Use Case:** Best of both worlds - immediate access + persistence
- **Behavior:** Writes to both memory and file simultaneously
- **Benefits:** Instant analytics with permanent record keeping

#### Streaming Storage
- **Use Case:** Real-time integration with external systems
- **Targets:** Kafka, ElasticSearch, CloudWatch, etc.
- **Features:** Configurable endpoints and retry logic

### 3. Configurable Verbosity Levels

#### Minimal Level
```json
{
  "log_id": "routing_abc123",
  "timestamp": "2025-08-09T10:30:00Z",
  "routing_decision": "LIGHTRAG",
  "confidence_score": 0.85
}
```

#### Standard Level (Default)
```json
{
  "log_id": "routing_abc123",
  "timestamp": "2025-08-09T10:30:00Z",
  "routing_decision": "LIGHTRAG",
  "confidence_score": 0.85,
  "query_hash": "sha256_hash_of_query",
  "query_length": 42,
  "backend_selected": "lightrag_primary",
  "processing_metrics": {
    "decision_time_ms": 15.5,
    "total_time_ms": 45.2
  }
}
```

#### Detailed Level
```json
{
  "log_id": "routing_abc123",
  "timestamp": "2025-08-09T10:30:00Z",
  "routing_decision": "LIGHTRAG",
  "confidence_score": 0.85,
  "query_hash": "sha256_hash_of_query",
  "query_length": 42,
  "query_complexity": 2.5,
  "backend_selected": "lightrag_primary",
  "reasoning": ["High confidence research query", "Biomedical entities detected"],
  "processing_metrics": {
    "decision_time_ms": 15.5,
    "total_time_ms": 45.2,
    "backend_selection_time_ms": 3.1,
    "memory_usage_mb": 128.5
  },
  "system_state": {
    "backend_health": {"lightrag": true, "perplexity": true},
    "resource_usage": {"cpu_percent": 25.0, "memory_percent": 45.0},
    "selection_algorithm": "weighted_round_robin",
    "load_balancer_metrics": {...}
  }
}
```

#### Debug Level
- Includes all detailed information plus:
- Raw query text (if privacy settings allow)
- Complete confidence breakdown
- Full system context and environment variables
- Detailed error traces and warnings

### 4. Real-Time Analytics and Metrics

#### Core Metrics Tracked

**Routing Metrics:**
- Total requests per time period
- Backend distribution percentages
- Average confidence scores
- Confidence distribution histograms

**Performance Metrics:**
- Average decision time
- P95 and P99 response times
- Maximum observed latencies
- Throughput rates

**Reliability Metrics:**
- Error rates and types
- Fallback usage statistics
- Backend availability metrics
- Circuit breaker activations

**Trend Analysis:**
- Hourly request volume trends
- Performance degradation detection
- Confidence score trends over time
- Seasonal pattern identification

#### Analytics Engine Architecture

```python
class RoutingAnalytics:
    def __init__(self, logger: RoutingDecisionLogger):
        self.metrics = AnalyticsMetrics()
        self.aggregation_interval = timedelta(minutes=5)
        self.anomaly_detector = AnomalyDetectionEngine()
    
    def record_decision_metrics(self, log_entry: RoutingDecisionLogEntry):
        # Update real-time metrics
        self._update_counters(log_entry)
        self._update_performance_metrics(log_entry)
        self._update_reliability_metrics(log_entry)
        
        # Check for anomalies
        if self._should_check_anomalies():
            self._detect_real_time_anomalies(log_entry)
```

### 5. Anomaly Detection System

#### Detection Categories

**Performance Anomalies:**
- Sudden increases in decision latency (>2x baseline)
- Unusual response time patterns
- Memory or CPU usage spikes correlated with routing

**Confidence Anomalies:**
- Significant drops in average confidence (>15%)
- Unusual confidence distribution patterns
- High variance in confidence scores

**Reliability Anomalies:**
- Error rate spikes (>5% sustained)
- Increased fallback usage (>20%)
- Backend health degradation patterns

**Distribution Anomalies:**
- Unexpected backend monopolization (>90% to one backend)
- Sudden shifts in routing patterns
- Load balancing algorithm effectiveness issues

#### Anomaly Detection Algorithms

**Statistical Methods:**
- Moving averages with configurable sensitivity
- Standard deviation-based outlier detection
- Percentile-based threshold monitoring

**Time Series Analysis:**
- Trend detection using linear regression
- Seasonal pattern recognition
- Change point detection algorithms

**Machine Learning (Future Enhancement):**
- Isolation Forest for multivariate anomalies
- LSTM-based sequence anomaly detection
- Clustering for pattern recognition

### 6. Privacy and Compliance Features

#### Query Anonymization
- **Hash-only mode:** Store only SHA-256 hashes of queries
- **Anonymized mode:** Remove all query content, keep metadata
- **Configurable sensitivity:** Different levels for different environments

#### Data Retention Policies
- **Time-based retention:** Automatic cleanup after configurable days (default: 90)
- **Size-based limits:** Maximum file sizes and entry counts
- **Compliance modes:** GDPR, HIPAA-compliant data handling

#### Security Features
- **Encryption at rest:** Optional file encryption for sensitive environments
- **Access controls:** Integration with existing authentication systems
- **Audit trails:** Complete logging of access to routing decision data

## Configuration System

### Environment-Based Configuration

```bash
# Basic logging settings
ROUTING_LOGGING_ENABLED=true
ROUTING_LOG_LEVEL=standard  # minimal, standard, detailed, debug
ROUTING_STORAGE_STRATEGY=hybrid  # memory, file, hybrid, streaming

# File storage settings
ROUTING_LOG_DIRECTORY=/var/log/cmo/routing_decisions
ROUTING_MAX_FILE_SIZE_MB=100
ROUTING_MAX_FILES_TO_KEEP=30
ROUTING_ENABLE_COMPRESSION=true

# Performance settings
ROUTING_BATCH_SIZE=50
ROUTING_BATCH_TIMEOUT_SECONDS=5.0
ROUTING_MAX_QUEUE_SIZE=10000

# Privacy settings
ROUTING_ANONYMIZE_QUERIES=false
ROUTING_HASH_SENSITIVE_DATA=true
ROUTING_RETENTION_DAYS=90

# Analytics settings
ROUTING_REAL_TIME_ANALYTICS=true
ROUTING_ANALYTICS_INTERVAL_MINUTES=5
ROUTING_ANOMALY_DETECTION=true

# Performance monitoring
ROUTING_PERF_MONITORING=true
ROUTING_MAX_LOGGING_OVERHEAD_MS=5.0
```

### Programmatic Configuration

```python
config = LoggingConfig(
    enabled=True,
    log_level=LogLevel.DETAILED,
    storage_strategy=StorageStrategy.HYBRID,
    log_directory="logs/routing_decisions",
    batch_size=50,
    max_memory_entries=10000,
    anonymize_queries=False,
    enable_real_time_analytics=True,
    analytics_aggregation_interval_minutes=5,
    retention_days=90
)
```

## Integration Patterns

### 1. Drop-in Enhancement Pattern

The system is designed to enhance existing routers without breaking changes:

```python
# Before: Standard router
router = ProductionIntelligentQueryRouter(config)

# After: Enhanced with logging
enhanced_router = EnhancedProductionIntelligentQueryRouter(
    config,
    logging_config=LoggingConfig(log_level=LogLevel.DETAILED)
)

# Same interface, enhanced functionality
prediction = await enhanced_router.route_query(query_text)
```

### 2. Mixin Pattern for Existing Classes

```python
class MyCustomRouter(RoutingLoggingMixin, BaseRouter):
    def __init__(self):
        super().__init__()
        # Automatically gains logging capabilities
        self.enable_routing_logging(LoggingConfig())
    
    async def route_query(self, query_text):
        # Custom routing logic
        prediction = self._make_routing_decision(query_text)
        
        # Automatic logging
        await self._log_routing_decision_if_enabled(
            prediction, query_text, metrics, system_state
        )
        
        return prediction
```

### 3. Standalone Analytics Pattern

```python
# Use logging system independently
logger = create_routing_logger(config)
analytics = create_routing_analytics(logger)

# Manual logging and analysis
await logger.log_routing_decision(prediction, query, metrics, state)
report = analytics.generate_analytics_report()
```

## Performance Considerations

### 1. Logging Overhead

**Target Performance:**
- <5ms logging overhead per request (configurable threshold)
- Non-blocking operations for critical path
- Automatic performance monitoring and alerting

**Optimization Techniques:**
- Asynchronous batching to minimize I/O operations
- Memory pooling for log entry objects
- Configurable queue sizes to balance memory and performance
- Background compression and cleanup processes

### 2. Memory Management

**Memory Usage Patterns:**
- Bounded memory usage with configurable limits
- LRU eviction for memory storage
- Periodic cleanup of aged entries
- Memory pressure monitoring and adaptive behavior

**Monitoring:**
```python
stats = logger.get_statistics()
print(f"Memory entries: {stats['memory_entries']}")
print(f"Queue size: {stats['queue_size']}")
print(f"Average logging time: {stats['avg_logging_time_ms']}ms")
```

### 3. Scalability Design

**Horizontal Scaling:**
- Separate logging instances per router instance
- Shared analytics aggregation across instances
- Distributed anomaly detection coordination

**Vertical Scaling:**
- Configurable resource limits
- Adaptive batch sizes based on load
- Intelligent queue management

## Deployment Strategies

### 1. Gradual Rollout

**Phase 1: Shadow Mode**
- Enable logging without affecting production routing
- Collect baseline metrics and validate performance
- Monitor resource usage and adjust configurations

**Phase 2: Partial Deployment**
- Enable on a subset of traffic (A/B testing)
- Compare performance with baseline
- Validate analytics accuracy and anomaly detection

**Phase 3: Full Deployment**
- Roll out to all routing instances
- Enable real-time analytics and alerting
- Begin using insights for optimization

### 2. Environment-Specific Configurations

**Development Environment:**
```yaml
routing_logging:
  enabled: true
  log_level: debug
  storage_strategy: memory
  anonymize_queries: false
  enable_real_time_analytics: true
```

**Staging Environment:**
```yaml
routing_logging:
  enabled: true
  log_level: detailed
  storage_strategy: hybrid
  anonymize_queries: false
  retention_days: 30
```

**Production Environment:**
```yaml
routing_logging:
  enabled: true
  log_level: standard
  storage_strategy: file
  anonymize_queries: true
  hash_sensitive_data: true
  enable_compression: true
  retention_days: 90
```

## Monitoring and Alerting

### 1. System Health Metrics

**Logging System Health:**
- Queue fill levels and overflow events
- Batch processing success rates
- File write errors and retry attempts
- Memory usage and cleanup effectiveness

**Performance Metrics:**
- Logging latency percentiles
- Throughput rates (entries/second)
- Storage utilization and growth rates
- Background task performance

### 2. Business Intelligence Metrics

**Routing Effectiveness:**
- Confidence score trends and distributions
- Backend performance comparisons
- Error rate analysis by category
- Load balancing effectiveness metrics

**System Optimization Insights:**
- Query complexity patterns
- Peak usage time identification
- Resource utilization correlations
- Anomaly frequency and resolution times

### 3. Alerting Rules

**Critical Alerts:**
- Sustained error rates >5%
- Logging system failures
- Storage space exhaustion
- Performance degradation >2x baseline

**Warning Alerts:**
- Confidence degradation >15%
- Backend imbalance >90% to single backend
- Anomaly detection increases
- Queue utilization >80%

## Future Enhancements

### 1. Machine Learning Integration

**Predictive Analytics:**
- Route success prediction based on query characteristics
- Optimal backend selection using reinforcement learning
- Proactive anomaly detection using time series forecasting

**Automated Optimization:**
- Self-tuning confidence thresholds
- Dynamic load balancing weight adjustment
- Intelligent fallback strategy optimization

### 2. Advanced Analytics Features

**Multi-dimensional Analysis:**
- Cross-correlation analysis between routing decisions and outcomes
- Cohort analysis for user behavior patterns
- A/B testing framework integration

**Real-time Dashboards:**
- Interactive performance visualization
- Drill-down capabilities for root cause analysis
- Real-time anomaly highlighting and investigation tools

### 3. Integration Enhancements

**External System Integration:**
- Kafka streaming for real-time data pipelines
- ElasticSearch integration for advanced search and analysis
- Prometheus metrics export for monitoring stack integration

**API Enhancements:**
- REST API for external analytics tools
- GraphQL interface for flexible data querying
- Webhook support for event-driven integrations

## Conclusion

The Routing Decision Logging and Analytics system provides comprehensive visibility into the CMO platform's routing behavior while maintaining high performance and flexibility. The modular design allows for gradual adoption and customization based on specific deployment requirements.

Key benefits:
- **Operational Visibility:** Complete insight into routing decisions and patterns
- **Performance Optimization:** Data-driven optimization opportunities
- **Reliability Monitoring:** Proactive detection of system issues
- **Compliance Support:** Privacy-aware logging with retention management
- **Scalable Design:** Handles growth from development to enterprise deployment

The system is designed to evolve with the platform, providing a solid foundation for advanced analytics and machine learning capabilities while maintaining backwards compatibility and operational simplicity.