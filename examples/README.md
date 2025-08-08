# CMO-LightRAG Integration Examples

This directory contains comprehensive integration examples showing how to seamlessly integrate the LightRAG module with the existing Clinical Metabolomics Oracle (CMO) system. These examples demonstrate various integration patterns from basic replacement to complete system integration, including advanced feature flag management and intelligent routing.

## 🔥 New Feature Flag Integration Examples

### 🧠 Main Integration Example (`main_integration_example.py`)

**Purpose**: Complete CMO chatbot integration with advanced feature flag routing

**Key Features**:
- Intelligent routing between LightRAG and Perplexity based on feature flags
- Real-time performance monitoring and quality assessment
- Circuit breaker protection for system stability
- User cohort assignment for A/B testing
- Graceful fallback mechanisms
- Integration with Chainlit message patterns from main.py

### 📊 Rollout Scenarios (`rollout_scenarios.py`)

**Purpose**: Comprehensive rollout strategy demonstrations and monitoring

**Key Features**:
- Linear rollout with steady percentage increases
- Exponential rollout for rapid scaling
- Canary deployments with strict validation
- Emergency rollback capabilities
- Real-time monitoring and alerting
- Quality gates and thresholds

### 🧪 A/B Testing Framework (`ab_testing_example.py`)

**Purpose**: Statistical analysis and comparison between LightRAG and Perplexity

**Key Features**:
- Statistical significance testing (t-tests, confidence intervals)
- Performance comparison metrics
- Quality score analysis
- Business impact assessment
- User satisfaction tracking
- ROI calculations

### 🚀 Production Deployment Guide (`production_deployment_guide.py`)

**Purpose**: Production-ready configuration for different environments

**Key Features**:
- Environment-specific configurations (dev, staging, prod, enterprise)
- Docker Compose and Kubernetes manifests
- Security best practices
- Monitoring and alerting setup
- Compliance and audit features

## 📚 Integration Examples Overview

### 1. Basic Chainlit Integration (`basic_chainlit_integration.py`)

**Purpose**: Drop-in replacement for Perplexity API with minimal code changes

**Key Features**:
- Direct replacement of Perplexity API calls with LightRAG
- Maintains existing session management with `cl.user_session`
- Preserves citation format and confidence scoring
- Keeps structured logging and error handling
- Supports async/await patterns throughout

**Use Cases**:
- Quick migration with minimal disruption
- Testing LightRAG functionality
- Proof of concept deployment

**Configuration**:
```bash
export OPENAI_API_KEY="your-api-key"
export LIGHTRAG_MODEL="gpt-4o-mini"
export LIGHTRAG_ENABLE_COST_TRACKING="true"
export LIGHTRAG_DAILY_BUDGET_LIMIT="25.0"
```

**Usage**:
```bash
# Test integration
python examples/basic_chainlit_integration.py test

# Run with Chainlit
chainlit run examples/basic_chainlit_integration.py
```

---

### 2. Advanced Pipeline Integration (`advanced_pipeline_integration.py`)

**Purpose**: Hybrid approach supporting both Perplexity and LightRAG with intelligent switching

**Key Features**:
- Hybrid system supporting both backends
- Configuration-driven switching between systems
- Feature flag support for gradual rollout (A/B testing)
- Seamless fallback mechanisms
- Performance comparison and metrics collection
- Cost optimization across systems

**Use Cases**:
- Gradual migration with risk mitigation
- A/B testing between systems
- Performance comparison studies
- Production deployment with fallback

**Configuration**:
```bash
export HYBRID_MODE="auto"                    # auto, perplexity, lightrag, split
export LIGHTRAG_ROLLOUT_PERCENTAGE="25"     # Percentage of traffic to LightRAG
export ENABLE_PERFORMANCE_COMPARISON="true"
export FALLBACK_TO_PERPLEXITY="true"
export ENABLE_COST_OPTIMIZATION="true"
```

**Usage**:
```bash
# Test hybrid system
python examples/advanced_pipeline_integration.py test

# Run hybrid system
chainlit run examples/advanced_pipeline_integration.py
```

---

### 3. Complete System Integration (`complete_system_integration.py`)

**Purpose**: Full replacement with comprehensive LightRAG integration and all advanced features

**Key Features**:
- Complete replacement of Perplexity API
- Full document processing pipeline integration
- Comprehensive quality assessment and validation
- Advanced cost tracking and budget management
- Real-time monitoring and performance analytics
- Audit trails and compliance tracking
- Research categorization and metrics logging
- Progressive system maintenance

**Use Cases**:
- Production deployment with full features
- Research environments requiring quality validation
- Systems requiring audit trails and compliance
- High-volume deployments with cost optimization

**Configuration**:
```bash
export OPENAI_API_KEY="your-api-key"
export LIGHTRAG_MODEL="gpt-4o"
export LIGHTRAG_ENABLE_ALL_FEATURES="true"
export LIGHTRAG_DAILY_BUDGET_LIMIT="100.0"
export LIGHTRAG_MONTHLY_BUDGET_LIMIT="2000.0"
export LIGHTRAG_ENABLE_QUALITY_VALIDATION="true"
export LIGHTRAG_ENABLE_PERFORMANCE_MONITORING="true"
```

**Usage**:
```bash
# Test complete system
python examples/complete_system_integration.py test

# Run complete system
chainlit run examples/complete_system_integration.py
```

---

### 4. Migration Guide (`migration_guide.py`)

**Purpose**: Comprehensive step-by-step migration tool with validation and rollback capabilities

**Key Features**:
- Step-by-step migration process (7 stages)
- Backward compatibility preservation
- Comprehensive testing and validation patterns
- Risk mitigation and rollback strategies
- Performance comparison utilities
- Data migration and validation tools

**Migration Steps**:
1. **Environment Setup** - Configure LightRAG environment
2. **Parallel System Setup** - Initialize both systems
3. **Comparison Testing** - Run side-by-side tests
4. **Gradual Traffic Routing** - Route percentage of traffic
5. **Performance Monitoring** - Monitor and optimize
6. **Full Migration** - Complete switch to LightRAG
7. **Legacy Cleanup** - Archive legacy components

**Usage**:
```bash
# Assess migration readiness
python examples/migration_guide.py assess

# Execute specific migration step
python examples/migration_guide.py migrate --step 1 --validate

# Test systems
python examples/migration_guide.py test --system lightrag
python examples/migration_guide.py test --system current

# Compare systems side-by-side
python examples/migration_guide.py compare --output comparison.json

# Run full migration
python examples/migration_guide.py full-migrate --validate

# Check migration status
python examples/migration_guide.py status

# Rollback specific step
python examples/migration_guide.py rollback --step 3
```

## 🚀 Quick Start Guide

### 1. Choose Your Integration Pattern

- **Production systems with feature flags**: Start with **Main Integration Example**
- **Gradual rollout**: Use **Rollout Scenarios** for safe deployment
- **A/B testing**: Use **A/B Testing Framework** for statistical analysis
- **Production deployment**: Use **Production Deployment Guide** for config
- **New deployments**: Start with **Complete System Integration**
- **Existing production systems**: Use **Migration Guide** for safe transition
- **Testing/evaluation**: Use **Basic Chainlit Integration**
- **Risk-averse deployments**: Use **Advanced Pipeline Integration**

### 2. Set Up Environment

```bash
# Core requirements
export OPENAI_API_KEY="your-openai-api-key"
export PERPLEXITY_API="your-perplexity-api-key"  # For hybrid/migration

# LightRAG configuration
export LIGHTRAG_MODEL="gpt-4o-mini"
export LIGHTRAG_EMBEDDING_MODEL="text-embedding-3-small"
export LIGHTRAG_WORKING_DIR="./lightrag_data"

# Budget management
export LIGHTRAG_DAILY_BUDGET_LIMIT="50.0"
export LIGHTRAG_MONTHLY_BUDGET_LIMIT="1000.0"
export LIGHTRAG_ENABLE_COST_TRACKING="true"

# Quality validation
export LIGHTRAG_ENABLE_QUALITY_VALIDATION="true"
export LIGHTRAG_RELEVANCE_THRESHOLD="0.75"
export LIGHTRAG_ACCURACY_THRESHOLD="0.80"

# Performance monitoring
export LIGHTRAG_ENABLE_PERFORMANCE_MONITORING="true"
export LIGHTRAG_BENCHMARK_FREQUENCY="daily"

# Feature flag configuration
export LIGHTRAG_ROLLOUT_PERCENTAGE="25.0"
export LIGHTRAG_ENABLE_AB_TESTING="true"
export LIGHTRAG_ENABLE_CIRCUIT_BREAKER="true"
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD="5"
export LIGHTRAG_MIN_QUALITY_THRESHOLD="0.7"
export LIGHTRAG_USER_HASH_SALT="your_secure_salt"
```

### 3. Install Dependencies

```bash
# Install LightRAG integration module
pip install -r requirements_lightrag.txt

# Verify installation
python -c "from lightrag_integration import get_integration_status; print(get_integration_status())"
```

### 4. Test Integration

```bash
# Test feature flag integration
python examples/main_integration_example.py

# Test rollout scenarios
python examples/rollout_scenarios.py

# Test A/B testing framework
python examples/ab_testing_example.py

# Generate production configs
python examples/production_deployment_guide.py

# Test basic functionality
python examples/basic_chainlit_integration.py test

# Run migration assessment
python examples/migration_guide.py assess

# Compare systems (if migrating)
python examples/migration_guide.py compare
```

## 🔧 Configuration Reference

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LightRAG | Required | `sk-...` |
| `PERPLEXITY_API` | Perplexity API key (for hybrid) | Optional | `pplx-...` |
| `LIGHTRAG_MODEL` | LLM model for LightRAG | `gpt-4o-mini` | `gpt-4o` |
| `LIGHTRAG_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` | `text-embedding-3-large` |
| `LIGHTRAG_DAILY_BUDGET_LIMIT` | Daily budget limit (USD) | `25.0` | `100.0` |
| `LIGHTRAG_MONTHLY_BUDGET_LIMIT` | Monthly budget limit (USD) | `500.0` | `2000.0` |
| `LIGHTRAG_ENABLE_COST_TRACKING` | Enable cost tracking | `true` | `false` |
| `LIGHTRAG_ENABLE_QUALITY_VALIDATION` | Enable quality validation | `true` | `false` |
| `LIGHTRAG_RELEVANCE_THRESHOLD` | Minimum relevance score | `0.75` | `0.80` |
| `HYBRID_MODE` | Hybrid system mode | `auto` | `lightrag`, `perplexity`, `split` |
| `LIGHTRAG_ROLLOUT_PERCENTAGE` | Traffic to LightRAG (%) | `25` | `50` |
| `LIGHTRAG_ENABLE_AB_TESTING` | Enable A/B testing | `true` | `false` |
| `LIGHTRAG_ENABLE_CIRCUIT_BREAKER` | Enable circuit breaker | `true` | `false` |
| `LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | Failure threshold | `5` | `3` |
| `LIGHTRAG_MIN_QUALITY_THRESHOLD` | Minimum quality score | `0.7` | `0.8` |
| `LIGHTRAG_USER_HASH_SALT` | User assignment salt | `default_salt` | `secure_salt` |
| `LIGHTRAG_FORCE_USER_COHORT` | Force cohort assignment | `null` | `lightrag`, `perplexity` |

### Advanced Configuration

For detailed configuration options, see:
- [`lightrag_integration/config.py`](../lightrag_integration/config.py) - Core configuration
- [`docs/LIGHTRAG_CONFIG_REFERENCE.md`](../docs/LIGHTRAG_CONFIG_REFERENCE.md) - Complete reference

## 📊 Performance Comparison

Each integration example includes performance monitoring capabilities:

### Metrics Tracked
- **Response Time**: Average query processing time
- **Success Rate**: Percentage of successful queries
- **Cost per Query**: Average cost per query (USD)
- **Quality Score**: Relevance and accuracy metrics
- **Citation Count**: Number of sources per response

### Comparison Tools
- **Built-in Benchmarking**: Each example includes test functions
- **Migration Guide Comparisons**: Side-by-side system comparison
- **Performance Logs**: Detailed performance tracking

## 🛡️ Safety and Rollback

### Backup Strategy
- **Configuration Backup**: Automatic backup of original settings
- **Database Backup**: Cost and usage data preservation
- **Code Preservation**: Original Perplexity code maintained

### Rollback Options
- **Step-by-Step Rollback**: Migration guide supports individual step rollback
- **Full System Rollback**: Complete reversion to original system
- **Hybrid Fallback**: Automatic fallback in hybrid mode

### Risk Mitigation
- **Gradual Rollout**: Percentage-based traffic routing
- **Health Monitoring**: Continuous system health checks
- **Budget Limits**: Automatic cost controls
- **Quality Gates**: Minimum quality thresholds

## 🔍 Troubleshooting

### Common Issues

1. **Initialization Failures**
   ```bash
   # Check integration status
   python -c "from lightrag_integration import validate_integration_setup; print(validate_integration_setup())"
   
   # Verify API keys
   python examples/basic_chainlit_integration.py test
   ```

2. **Budget Exceeded**
   ```bash
   # Check current budget status
   python -c "from lightrag_integration import create_clinical_rag_system; import asyncio; rag = create_clinical_rag_system(); print(asyncio.run(rag.get_cost_summary()).__dict__)"
   ```

3. **Performance Issues**
   ```bash
   # Run performance comparison
   python examples/migration_guide.py compare --output performance_analysis.json
   ```

4. **Quality Issues**
   ```bash
   # Check quality validation settings
   python examples/complete_system_integration.py test
   ```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
export LIGHTRAG_LOG_LEVEL="DEBUG"
export LIGHTRAG_ENABLE_DETAILED_LOGGING="true"
```

### Support Resources

- **Integration Status**: `get_integration_status()` function
- **Health Checks**: Built-in health monitoring
- **Log Files**: Comprehensive logging in `logs/` directory
- **Configuration Validation**: `validate_integration_setup()` function

## 📈 Production Deployment

### Recommended Deployment Process

1. **Assessment Phase**
   ```bash
   python examples/migration_guide.py assess
   ```

2. **Testing Phase**
   ```bash
   python examples/basic_chainlit_integration.py test
   python examples/migration_guide.py compare
   ```

3. **Gradual Migration**
   ```bash
   # Start with 10% traffic
   export LIGHTRAG_ROLLOUT_PERCENTAGE="10"
   chainlit run examples/advanced_pipeline_integration.py
   
   # Monitor and gradually increase
   export LIGHTRAG_ROLLOUT_PERCENTAGE="25"
   export LIGHTRAG_ROLLOUT_PERCENTAGE="50"
   export LIGHTRAG_ROLLOUT_PERCENTAGE="100"
   ```

4. **Full Production**
   ```bash
   chainlit run examples/complete_system_integration.py
   ```

### Monitoring and Maintenance

- **Daily**: Check budget usage and performance metrics
- **Weekly**: Review quality reports and user feedback
- **Monthly**: Run comprehensive performance analysis
- **Quarterly**: Update configurations and optimize costs

### Scaling Considerations

- **Budget Planning**: Monitor costs and adjust limits
- **Performance Optimization**: Tune model parameters
- **Quality Assurance**: Regular validation and testing
- **Capacity Planning**: Monitor usage patterns

## 🤝 Contributing

To add new integration examples or improve existing ones:

1. Follow the established pattern structure
2. Include comprehensive documentation
3. Add test functions for validation
4. Provide configuration examples
5. Include error handling and logging

## 📄 License

These integration examples are part of the Clinical Metabolomics Oracle project and are subject to the same licensing terms.