# API Cost Monitoring System - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Initial Setup](#initial-setup)
3. [Understanding the Dashboard](#understanding-the-dashboard)
4. [Managing Budgets](#managing-budgets)
5. [Monitoring and Alerts](#monitoring-and-alerts)
6. [Research Analytics](#research-analytics)
7. [Cost Optimization](#cost-optimization)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Advanced Usage](#advanced-usage)
10. [FAQ](#faq)

---

## Getting Started

The API Cost Monitoring System helps you track, manage, and optimize your AI API usage costs in the Clinical Metabolomics Oracle environment. This guide will walk you through setup, daily usage, and optimization strategies.

### Who Should Use This Guide

- **Researchers**: Scientists using AI tools for metabolomics analysis
- **Lab Administrators**: Personnel managing research budgets and resources  
- **IT Administrators**: Staff responsible for system configuration and monitoring
- **Principal Investigators**: Research leaders overseeing project costs

### What You'll Learn

- How to set up budget limits and monitoring
- How to interpret dashboard metrics and alerts
- How to optimize costs while maintaining research productivity
- How to generate reports for budget planning and compliance

---

## Initial Setup

### Prerequisites

Before starting, ensure you have:
- Access to the Clinical Metabolomics Oracle system
- An OpenAI API key (contact your administrator if needed)
- Basic understanding of your research budget requirements

### Step 1: Environment Configuration

Set up your environment variables for basic configuration:

```bash
# Required: OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Budget limits (adjust based on your research budget)
export LIGHTRAG_DAILY_BUDGET_LIMIT="50.0"      # $50 per day
export LIGHTRAG_MONTHLY_BUDGET_LIMIT="1000.0"  # $1000 per month

# Alert settings
export LIGHTRAG_ENABLE_BUDGET_ALERTS="true"
export LIGHTRAG_COST_ALERT_THRESHOLD="80.0"    # Alert at 80% of budget
```

### Step 2: First-Time Initialization

Run the system initialization script:

```python
# Initialize the budget monitoring system
from lightrag_integration import BudgetManagementFactory
from lightrag_integration.config import LightRAGConfig

# Create configuration
config = LightRAGConfig.get_config()

# Initialize budget system
budget_system = BudgetManagementFactory.create_complete_system(
    lightrag_config=config,
    daily_budget_limit=50.0,    # $50 per day
    monthly_budget_limit=1000.0 # $1000 per month
)

# Start monitoring
budget_system.start()
print("✅ Budget monitoring system started successfully")
```

### Step 3: Verify Setup

Check that everything is working correctly:

```python
# Get current status
status = budget_system.get_budget_status()
print(f"Daily budget: ${status['daily_budget']['budget_limit']:.2f}")
print(f"Monthly budget: ${status['monthly_budget']['budget_limit']:.2f}")
print(f"System health: {status['system_health']['overall_status']}")
```

---

## Understanding the Dashboard

### Accessing the Dashboard

The dashboard provides a comprehensive view of your API usage and costs. Access it through:

```python
# Get dashboard overview
overview = budget_system.dashboard.get_dashboard_overview()
metrics = overview['data']['metrics']

print(f"Budget Health: {metrics['budget_health_status']}")
print(f"Daily Usage: {metrics['daily_percentage']:.1f}%")
print(f"Monthly Usage: {metrics['monthly_percentage']:.1f}%")
```

### Key Metrics Explained

#### 1. Budget Health Score
- **Green (Healthy)**: Usage below 75% of budget
- **Yellow (Warning)**: Usage between 75-90% of budget  
- **Red (Critical)**: Usage above 90% of budget

#### 2. Cost Breakdown
- **Daily Cost**: Current day's API usage cost
- **Monthly Cost**: Current month's total cost
- **Projected Cost**: Estimated costs based on usage trends
- **Cost Efficiency**: How effectively you're using your budget

#### 3. Usage Analytics
- **Total API Calls**: Number of requests made
- **Average Response Time**: Performance of API calls
- **Error Rate**: Percentage of failed requests
- **Tokens Consumed**: Total tokens used across all models

### Dashboard Sections

#### Budget Overview
```
┌─────────────────────────────────┐
│ Daily Budget Status             │
│ Used: $23.45 / $50.00 (46.9%)  │
│ Projected: $48.20 (96.4%)      │
│ Health: ⚠️  Warning              │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ Monthly Budget Status           │
│ Used: $346.78 / $1000.00 (34.7%)│
│ Projected: $1,012.50 (101.3%)  │
│ Health: 🔴 Critical             │
└─────────────────────────────────┘
```

#### Research Categories
```
Top Research Areas by Cost:
1. Metabolite Identification    $156.34 (45.1%)
2. Pathway Analysis            $89.23 (25.7%)  
3. Literature Search           $67.89 (19.6%)
4. Data Validation            $33.32 (9.6%)
```

#### Recent Activity
```
Recent API Calls:
• 14:32 - GPT-4o Mini - Metabolite analysis - $0.023
• 14:28 - Text Embedding - Literature search - $0.001
• 14:25 - GPT-4o - Pathway prediction - $0.156
• 14:20 - GPT-4o Mini - Data validation - $0.012
```

---

## Managing Budgets

### Setting Budget Limits

#### Daily Budget Management

```python
# Set daily budget limit
budget_system.budget_manager.update_budget_limits(
    daily_budget=75.0,  # Increase daily limit to $75
    monthly_budget=1500.0  # Keep monthly at $1500
)

# Check updated limits
status = budget_system.get_budget_status()
print(f"New daily limit: ${status['daily_budget']['budget_limit']:.2f}")
```

#### Monthly Budget Planning

Consider your research timeline and activities:

```python
# Example budget allocation for different research phases
research_phases = {
    "literature_review": {
        "daily_budget": 25.0,
        "duration_days": 14,
        "total_budget": 350.0
    },
    "data_analysis": {
        "daily_budget": 75.0,
        "duration_days": 21,
        "total_budget": 1575.0
    },
    "manuscript_writing": {
        "daily_budget": 15.0,
        "duration_days": 10,
        "total_budget": 150.0
    }
}

# Calculate total project budget
total_project_cost = sum(phase["total_budget"] for phase in research_phases.values())
print(f"Total project budget needed: ${total_project_cost:.2f}")
```

### Budget Allocation Strategies

#### By Research Category

```python
# Allocate budget by research focus
category_budgets = {
    "metabolite_identification": 0.40,  # 40% of budget
    "pathway_analysis": 0.25,           # 25% of budget
    "literature_search": 0.20,          # 20% of budget
    "data_validation": 0.15             # 15% of budget
}

daily_budget = 100.0
for category, percentage in category_budgets.items():
    allocated_amount = daily_budget * percentage
    print(f"{category}: ${allocated_amount:.2f} per day")
```

#### By Time Period

```python
# Weekly budget planning
weekly_schedule = {
    "monday": 20.0,     # Heavy analysis day
    "tuesday": 15.0,    # Literature review
    "wednesday": 25.0,  # Major analysis work
    "thursday": 10.0,   # Review and validation
    "friday": 15.0,     # Documentation and reporting
    "weekend": 5.0      # Light work if needed
}

weekly_total = sum(weekly_schedule.values())
print(f"Weekly budget plan: ${weekly_total:.2f}")
```

---

## Monitoring and Alerts

### Understanding Alert Levels

#### 1. Information Alerts (ℹ️)
- Routine budget status updates
- Daily/weekly usage summaries
- System health notifications

#### 2. Warning Alerts (⚠️)
- 75% of daily budget reached
- 80% of monthly budget reached  
- Unusual spending patterns detected

#### 3. Critical Alerts (🔴)
- 90% of daily budget reached
- 95% of monthly budget reached
- Budget projections indicate overage

#### 4. Budget Exceeded (🚨)
- Daily budget limit reached
- Monthly budget limit reached
- Operations may be automatically blocked

### Setting Up Email Alerts

```bash
# Configure email alerts
export ALERT_EMAIL_SMTP_SERVER="smtp.gmail.com"
export ALERT_EMAIL_SMTP_PORT="587"
export ALERT_EMAIL_USERNAME="your-research-email@university.edu"
export ALERT_EMAIL_PASSWORD="your-app-password"
export ALERT_EMAIL_RECIPIENTS="pi@university.edu,admin@university.edu"
```

Test your email configuration:

```python
# Test email alerts
test_result = budget_system.alert_system.test_channels()
if test_result['email']['success']:
    print("✅ Email alerts configured successfully")
else:
    print("❌ Email alert configuration failed:", test_result['email']['error'])
```

### Setting Up Slack Alerts

```bash
# Configure Slack webhook
export ALERT_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
export ALERT_SLACK_CHANNEL="#budget-alerts"
export ALERT_SLACK_USERNAME="Budget Monitor"
```

### Custom Alert Thresholds

```python
# Customize alert thresholds
from lightrag_integration.budget_manager import BudgetThreshold

custom_thresholds = BudgetThreshold(
    warning_percentage=70.0,    # Alert at 70% instead of 75%
    critical_percentage=85.0,   # Alert at 85% instead of 90%
    exceeded_percentage=100.0   # Block at 100%
)

budget_system.budget_manager.update_thresholds(custom_thresholds)
```

### Responding to Alerts

#### When You Receive a Warning Alert:

1. **Check current usage**:
   ```python
   status = budget_system.get_budget_status()
   print(f"Current usage: {status['daily_budget']['percentage_used']:.1f}%")
   ```

2. **Review recent activity**:
   ```python
   recent_costs = budget_system.cost_persistence.get_recent_costs(hours=2)
   for cost in recent_costs[-5:]:  # Last 5 operations
       print(f"{cost['timestamp']}: {cost['operation_type']} - ${cost['cost_usd']:.3f}")
   ```

3. **Adjust usage if needed**:
   - Reduce API call frequency
   - Switch to more cost-effective models
   - Optimize prompts to use fewer tokens

#### When You Receive a Critical Alert:

1. **Review budget projections**:
   ```python
   if budget_system.real_time_monitor:
       projections = budget_system.real_time_monitor.get_projections()
       print(f"Daily projection: ${projections['daily']['projected_cost']:.2f}")
   ```

2. **Consider temporary budget increase**:
   ```python
   # Temporarily increase daily budget if critical work needed
   budget_system.budget_manager.update_budget_limits(daily_budget=75.0)
   ```

3. **Implement cost-saving measures immediately**

---

## Research Analytics

### Understanding Your Research Costs

#### Cost per Research Category

```python
# Get category analysis for last 30 days
from lightrag_integration.budget_dashboard import DashboardTimeRange

analytics = budget_system.dashboard.get_cost_analytics(
    time_range="last_30_days",
    include_categories=True
)

category_data = analytics['data']['category_analysis']
print("Research Category Costs (Last 30 Days):")
for category in category_data['top_categories'][:5]:
    print(f"• {category['category']}: ${category['total_cost']:.2f} "
          f"({category['total_calls']} calls, "
          f"${category['total_cost']/category['total_calls']:.4f} per call)")
```

#### Identifying Cost Patterns

```python
# Analyze cost trends
trends = analytics['data']['trends']
if trends['trend_analysis']['trend_direction'] == 'increasing':
    print(f"⚠️ Costs are trending upward by {trends['trend_analysis']['trend_percentage']:.1f}%")
    print("Consider reviewing usage patterns and optimizing operations")
elif trends['trend_analysis']['trend_direction'] == 'decreasing':
    print(f"✅ Costs are trending downward by {trends['trend_analysis']['trend_percentage']:.1f}%")
    print("Current cost optimization strategies are working")
```

### Model Usage Analysis

```python
# Compare model costs and efficiency
performance = budget_system.api_metrics_logger.get_performance_summary()

print("Model Performance Summary:")
for model, stats in performance.get('model_breakdown', {}).items():
    avg_cost = stats['total_cost'] / max(stats['total_calls'], 1)
    print(f"• {model}:")
    print(f"  - Total calls: {stats['total_calls']}")
    print(f"  - Total cost: ${stats['total_cost']:.2f}")
    print(f"  - Average cost per call: ${avg_cost:.4f}")
    print(f"  - Success rate: {stats['success_rate']:.1f}%")
```

### Research Productivity Metrics

```python
def calculate_research_roi():
    """Calculate research return on investment metrics."""
    
    # Get last 7 days of data
    weekly_report = budget_system.cost_persistence.generate_cost_report(
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now()
    )
    
    total_cost = weekly_report['summary']['total_cost']
    total_calls = weekly_report['summary']['total_calls']
    success_rate = weekly_report['summary']['success_rate']
    
    # Calculate productivity metrics
    cost_per_successful_operation = total_cost / (total_calls * success_rate / 100)
    
    print("Research Productivity Metrics (Last 7 Days):")
    print(f"• Total investment: ${total_cost:.2f}")
    print(f"• Successful operations: {int(total_calls * success_rate / 100)}")
    print(f"• Cost per successful result: ${cost_per_successful_operation:.4f}")
    print(f"• Research efficiency: {success_rate:.1f}%")
    
    return {
        'total_cost': total_cost,
        'cost_per_result': cost_per_successful_operation,
        'efficiency': success_rate
    }

roi_metrics = calculate_research_roi()
```

---

## Cost Optimization

### Model Selection Strategies

#### Choose the Right Model for the Task

```python
# Model selection guide for different research tasks
model_recommendations = {
    "literature_search": {
        "primary": "gpt-4o-mini",
        "reason": "Cost-effective for information extraction",
        "estimated_cost": "$0.001-0.010 per query"
    },
    "metabolite_identification": {
        "primary": "gpt-4o",
        "fallback": "gpt-4o-mini",
        "reason": "Higher accuracy needed for critical analysis",
        "estimated_cost": "$0.050-0.200 per analysis"
    },
    "pathway_analysis": {
        "primary": "gpt-4o",
        "reason": "Complex reasoning required",
        "estimated_cost": "$0.100-0.500 per analysis"
    },
    "data_validation": {
        "primary": "gpt-4o-mini",
        "reason": "Simple validation tasks",
        "estimated_cost": "$0.001-0.005 per validation"
    }
}

# Print recommendations
for task, info in model_recommendations.items():
    print(f"\n{task.title().replace('_', ' ')}:")
    print(f"  Recommended model: {info['primary']}")
    print(f"  Reason: {info['reason']}")
    print(f"  Cost estimate: {info['estimated_cost']}")
```

#### Implement Model Cascading

```python
def cost_aware_analysis(query, complexity="medium"):
    """Use different models based on complexity and cost constraints."""
    
    current_usage = budget_system.get_budget_status()
    daily_percentage = current_usage['daily_budget']['percentage_used']
    
    if daily_percentage < 50:
        # Budget is healthy, use best model
        model = "gpt-4o" if complexity == "high" else "gpt-4o-mini"
    elif daily_percentage < 80:
        # Budget approaching limit, use cost-effective model
        model = "gpt-4o-mini"
    else:
        # Budget critical, use most economical approach
        model = "gpt-4o-mini"
        query = optimize_query_for_cost(query)  # Shorten query
    
    print(f"Using {model} for analysis (budget usage: {daily_percentage:.1f}%)")
    return model

# Example usage
model_choice = cost_aware_analysis("Analyze this metabolite structure", "high")
```

### Prompt Optimization

#### Token-Efficient Prompting

```python
def optimize_prompt_for_cost(original_prompt, max_tokens=None):
    """Optimize prompts to reduce token usage while maintaining quality."""
    
    # Remove unnecessary words and phrases
    optimizations = {
        "Please analyze the following": "Analyze:",
        "I would like you to": "",
        "Could you please": "",
        "Can you help me": "",
        "It would be great if": "",
    }
    
    optimized = original_prompt
    for phrase, replacement in optimizations.items():
        optimized = optimized.replace(phrase, replacement)
    
    # Truncate if max_tokens specified
    if max_tokens:
        words = optimized.split()
        if len(words) > max_tokens * 0.75:  # Rough token estimation
            optimized = " ".join(words[:int(max_tokens * 0.75)])
            optimized += "... [truncated for cost optimization]"
    
    print(f"Original prompt: {len(original_prompt)} chars")
    print(f"Optimized prompt: {len(optimized)} chars")
    print(f"Estimated savings: {(1 - len(optimized)/len(original_prompt))*100:.1f}%")
    
    return optimized

# Example
original = "Please analyze the following metabolite data and provide detailed insights about its potential biological pathways and interactions"
optimized = optimize_prompt_for_cost(original, max_tokens=100)
```

### Batch Processing Strategies

#### Combine Related Queries

```python
def batch_metabolite_analysis(metabolites, batch_size=5):
    """Process multiple metabolites in batches to reduce API calls."""
    
    results = []
    
    for i in range(0, len(metabolites), batch_size):
        batch = metabolites[i:i + batch_size]
        
        # Create combined prompt
        batch_prompt = "Analyze these metabolites:\n"
        for j, metabolite in enumerate(batch, 1):
            batch_prompt += f"{j}. {metabolite}\n"
        batch_prompt += "\nProvide analysis for each metabolite separately."
        
        # Track batch operation
        with budget_system.track_operation("batch_metabolite_analysis") as tracker:
            # Make API call for batch
            response = make_api_call(batch_prompt)
            
            # Update tracking
            tracker.set_tokens(
                prompt=estimate_tokens(batch_prompt),
                completion=estimate_tokens(response)
            )
            tracker.set_cost(calculate_batch_cost(response))
            
            results.extend(parse_batch_response(response, batch))
    
    return results

# Example usage
metabolite_list = ["glucose", "fructose", "sucrose", "lactose", "galactose"]
batch_results = batch_metabolite_analysis(metabolite_list, batch_size=3)
print(f"Processed {len(metabolite_list)} metabolites in {len(metabolite_list)//3 + 1} API calls")
```

### Caching and Reuse

#### Implement Response Caching

```python
import hashlib
import json

class ResponseCache:
    """Cache API responses to avoid repeated costs."""
    
    def __init__(self, cache_duration_hours=24):
        self.cache = {}
        self.cache_duration = cache_duration_hours * 3600
    
    def get_cache_key(self, prompt, model):
        """Generate cache key from prompt and model."""
        content = f"{prompt}:{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_response(self, prompt, model):
        """Get cached response if available and not expired."""
        cache_key = self.get_cache_key(prompt, model)
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                print("Using cached response - $0.00 cost")
                return cached_data['response']
        
        return None
    
    def cache_response(self, prompt, model, response, cost):
        """Cache response with metadata."""
        cache_key = self.get_cache_key(prompt, model)
        self.cache[cache_key] = {
            'response': response,
            'cost': cost,
            'timestamp': time.time()
        }
    
    def get_cache_stats(self):
        """Get cache usage statistics."""
        total_entries = len(self.cache)
        total_savings = sum(entry['cost'] for entry in self.cache.values())
        
        return {
            'cached_responses': total_entries,
            'estimated_savings': total_savings
        }

# Initialize cache
response_cache = ResponseCache(cache_duration_hours=24)

# Use cache in analysis
def cached_analysis(prompt, model="gpt-4o-mini"):
    # Check cache first
    cached_result = response_cache.get_cached_response(prompt, model)
    if cached_result:
        return cached_result
    
    # Make API call if not cached
    with budget_system.track_operation("cached_analysis") as tracker:
        response = make_api_call(prompt, model)
        cost = calculate_cost(response)
        
        tracker.set_cost(cost)
        response_cache.cache_response(prompt, model, response, cost)
        
        return response

# Check cache savings
stats = response_cache.get_cache_stats()
print(f"Cache has saved approximately ${stats['estimated_savings']:.2f}")
```

---

## Troubleshooting Common Issues

### Issue 1: Budget Exceeded Unexpectedly

**Symptoms:**
- Operations are being blocked
- Critical alert emails received
- Dashboard shows budget exceeded

**Diagnosis:**
```python
# Check recent high-cost operations
recent_costs = budget_system.cost_persistence.get_recent_costs(hours=4)
expensive_ops = [op for op in recent_costs if op['cost_usd'] > 0.10]

print("Recent expensive operations:")
for op in expensive_ops:
    print(f"• {op['timestamp']}: {op['operation_type']} - ${op['cost_usd']:.3f}")
```

**Solutions:**
1. **Increase budget temporarily**:
   ```python
   budget_system.budget_manager.update_budget_limits(daily_budget=100.0)
   ```

2. **Switch to cost-effective models**:
   ```python
   # Use GPT-4o-mini instead of GPT-4o for less critical tasks
   ```

3. **Review and optimize prompts**

### Issue 2: Alerts Not Being Received

**Symptoms:**
- No email or Slack notifications
- Missing budget threshold alerts

**Diagnosis:**
```python
# Test alert channels
test_results = budget_system.alert_system.test_channels()
print("Alert channel test results:")
for channel, result in test_results.items():
    status = "✅ Working" if result['success'] else "❌ Failed"
    print(f"• {channel}: {status}")
    if not result['success']:
        print(f"  Error: {result.get('error', 'Unknown error')}")
```

**Solutions:**
1. **Check email configuration**:
   ```bash
   echo $ALERT_EMAIL_SMTP_SERVER
   echo $ALERT_EMAIL_USERNAME
   # Verify environment variables are set correctly
   ```

2. **Verify Slack webhook**:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
   --data '{"text":"Test message"}' \
   $ALERT_SLACK_WEBHOOK_URL
   ```

3. **Check network connectivity and firewall settings**

### Issue 3: High API Response Times

**Symptoms:**
- Slow research workflows
- High average response times in dashboard

**Diagnosis:**
```python
performance = budget_system.api_metrics_logger.get_performance_summary()
current_hour = performance['current_hour']

print(f"Average response time: {current_hour['avg_response_time_ms']:.0f}ms")
print(f"Error rate: {current_hour['error_rate_percent']:.1f}%")

if current_hour['avg_response_time_ms'] > 5000:
    print("⚠️ Response times are higher than optimal")
```

**Solutions:**
1. **Optimize prompt length**
2. **Use appropriate model for task complexity**
3. **Check network connectivity**
4. **Consider using batch processing**

### Issue 4: Database Errors

**Symptoms:**
- Cost tracking not working
- Database connection errors

**Diagnosis:**
```python
# Check database connection
try:
    status = budget_system.cost_persistence.get_database_stats()
    print(f"Database records: {status['total_records']}")
    print(f"Database size: {status['database_size_mb']:.1f} MB")
except Exception as e:
    print(f"Database error: {e}")
```

**Solutions:**
1. **Check disk space**:
   ```python
   import shutil
   db_path = budget_system.config.cost_db_path
   free_space = shutil.disk_usage(db_path.parent).free / 1024**3
   print(f"Free disk space: {free_space:.1f} GB")
   ```

2. **Verify database permissions**
3. **Run database maintenance**:
   ```python
   budget_system.cost_persistence.optimize_database()
   ```

---

## Advanced Usage

### Custom Research Categories

```python
# Define custom categories for your research
from lightrag_integration.cost_persistence import ResearchCategory

# Add custom categories
custom_categories = {
    "drug_discovery": "Drug Discovery and Development",
    "biomarker_validation": "Biomarker Validation Studies",
    "clinical_correlation": "Clinical Data Correlation"
}

# Use custom categories in tracking
with budget_system.track_operation(
    operation_type="llm_call",
    research_category="drug_discovery"
) as tracker:
    # Your research-specific API call
    pass
```

### Multi-Project Budget Tracking

```python
# Track costs across different research projects
project_budgets = {
    "alzheimers_metabolomics": {"daily": 30.0, "monthly": 600.0},
    "diabetes_biomarkers": {"daily": 40.0, "monthly": 800.0},
    "aging_pathways": {"daily": 20.0, "monthly": 400.0}
}

# Create separate tracking for each project
for project_name, limits in project_budgets.items():
    project_system = BudgetManagementFactory.create_complete_system(
        lightrag_config=config,
        daily_budget_limit=limits["daily"],
        monthly_budget_limit=limits["monthly"],
        system_name=project_name
    )
    
    # Use project-specific tracking
    with project_system.track_operation(
        operation_type="pathway_analysis",
        project=project_name
    ) as tracker:
        # Project-specific analysis
        pass
```

### Integration with External Tools

#### Jupyter Notebook Integration

```python
# Magic command for Jupyter notebooks
%load_ext budget_monitoring

# Use magic command to track cells
%%track_budget --operation-type metabolite_analysis --category pathway_analysis
# Your analysis code here
result = analyze_metabolic_pathway(data)
display(result)
```

#### R Integration

```python
# Python-R bridge for budget tracking
import rpy2.robjects as robjects

def r_analysis_with_tracking(r_script, operation_type="r_analysis"):
    """Execute R script with budget tracking."""
    with budget_system.track_operation(operation_type) as tracker:
        # Execute R script
        result = robjects.r(r_script)
        
        # Estimate cost based on computation time and complexity
        estimated_cost = estimate_r_computation_cost(r_script)
        tracker.set_cost(estimated_cost)
        
        return result
```

---

## FAQ

### General Questions

**Q: How accurate are the cost estimates?**
A: Cost estimates are based on actual OpenAI pricing and token usage. Accuracy is typically within 5% of actual costs. The system learns from actual usage to improve estimates over time.

**Q: What happens if I exceed my budget?**
A: When budget limits are reached, the system can either:
- Send alerts and continue operations (warning mode)
- Automatically block expensive operations (protection mode)
- Allow manual override for critical research needs

**Q: Can I change budget limits during active research?**
A: Yes, budget limits can be adjusted at any time through the configuration system or dashboard interface.

### Technical Questions

**Q: How is data stored and secured?**
A: Cost data is stored in a local SQLite database with optional encryption. No API responses or research data are stored, only metadata and cost information.

**Q: Can I export my cost data?**
A: Yes, you can export cost data in various formats:
```python
# Export to CSV
report = budget_system.dashboard.get_cost_report(
    start_date="2025-01-01T00:00:00Z",
    end_date="2025-08-06T23:59:59Z",
    format="csv"
)
```

**Q: How do I backup my cost data?**
A: The SQLite database can be backed up by copying the database file:
```python
import shutil
db_path = budget_system.config.cost_db_path
backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d')}"
shutil.copy2(db_path, backup_path)
```

### Research-Specific Questions

**Q: How should I budget for different research phases?**
A: Consider these typical patterns:
- **Literature Review**: $10-25/day (mostly GPT-4o-mini)
- **Data Analysis**: $50-100/day (mixed models)
- **Manuscript Writing**: $15-30/day (moderate usage)

**Q: Which model should I use for metabolomics analysis?**
A: 
- **Simple queries**: GPT-4o-mini ($0.15/$0.60 per 1M tokens)
- **Complex analysis**: GPT-4o ($5.00/$15.00 per 1M tokens)
- **Embeddings**: text-embedding-3-small ($0.02 per 1M tokens)

**Q: How can I optimize costs without sacrificing research quality?**
A:
1. Use model cascading (start with cheaper models)
2. Implement response caching
3. Batch related queries
4. Optimize prompt efficiency
5. Use appropriate models for each task

**Q: Can I share budget limits across a research team?**
A: Yes, the system supports shared budgets. Configure team-wide limits and individual user tracking for detailed analysis.

---

**Need More Help?**

- Check the [Configuration Reference](./API_COST_MONITORING_CONFIGURATION_REFERENCE.md) for detailed settings
- See the [Troubleshooting Guide](./API_COST_MONITORING_TROUBLESHOOTING_GUIDE.md) for common issues
- Review the [API Reference](./API_COST_MONITORING_API_REFERENCE.md) for programmatic access
- Consult the [Deployment Guide](./API_COST_MONITORING_DEPLOYMENT_GUIDE.md) for advanced setup

---

*This user guide is part of the Clinical Metabolomics Oracle API Cost Monitoring System documentation suite. For technical implementation details, see the [Developer Guide](./API_COST_MONITORING_DEVELOPER_GUIDE.md).*