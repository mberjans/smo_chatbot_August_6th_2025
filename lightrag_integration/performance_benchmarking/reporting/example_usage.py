#!/usr/bin/env python3
"""
Example Usage of Performance Benchmarking Reporting System.

This script demonstrates how to use the comprehensive performance reporting
system for quality validation benchmarking. It shows integration between
all major components: QualityPerformanceReporter, PerformanceDashboard,
RecommendationEngine, and StatisticalAnalyzer.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import random
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import reporting components
try:
    from quality_performance_reporter import (
        QualityPerformanceReporter, PerformanceReportConfiguration,
        ReportFormat, generate_comprehensive_performance_report
    )
    from performance_dashboard import (
        PerformanceDashboard, DashboardConfiguration, 
        generate_performance_dashboard
    )
    from recommendation_engine import (
        RecommendationEngine, RecommendationType, 
        generate_performance_recommendations
    )
    from statistical_analyzer import (
        StatisticalAnalyzer, analyze_performance_statistics
    )
    
    # Import parent modules for data structures
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from quality_performance_benchmarks import (
        QualityValidationMetrics, QualityValidationBenchmarkSuite
    )
    from quality_aware_metrics_logger import QualityAPIMetric
    
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Please ensure all reporting modules are properly installed.")
    sys.exit(1)


def generate_sample_benchmark_data(num_samples: int = 50) -> list:
    """Generate sample benchmark data for demonstration."""
    benchmark_data = []
    
    base_time = time.time() - (24 * 3600)  # 24 hours ago
    
    for i in range(num_samples):
        # Create realistic sample data with some trends and anomalies
        timestamp = base_time + (i * 1800)  # Every 30 minutes
        
        # Add some trend and noise
        trend_factor = 1 + (i * 0.01)  # Gradual increase
        noise_factor = random.uniform(0.8, 1.2)
        
        # Occasional anomalies
        if random.random() < 0.05:  # 5% chance of anomaly
            anomaly_factor = random.uniform(2.0, 3.0)
        else:
            anomaly_factor = 1.0
        
        metric = QualityValidationMetrics(
            scenario_name=f"scenario_{i % 10}",
            start_time=timestamp,
            end_time=timestamp + 300,  # 5 minutes
            operations_count=random.randint(10, 50),
            success_count=random.randint(8, 50),
            average_latency_ms=max(100, 1500 * trend_factor * noise_factor * anomaly_factor),
            p95_latency_ms=max(150, 2500 * trend_factor * noise_factor * anomaly_factor),
            throughput_ops_per_sec=max(0.5, 8.0 / (trend_factor * noise_factor)),
            memory_usage_mb=random.uniform(200, 800) * trend_factor,
            cpu_usage_percent=random.uniform(20, 80),
            error_rate_percent=max(0, random.uniform(0, 10) * anomaly_factor),
            validation_accuracy_rate=max(70, random.uniform(85, 98) / anomaly_factor),
            claims_extracted_count=random.randint(5, 20),
            claims_validated_count=random.randint(3, 18),
            avg_validation_confidence=random.uniform(75, 95),
            claim_extraction_time_ms=random.uniform(100, 500) * trend_factor,
            factual_validation_time_ms=random.uniform(200, 800) * trend_factor,
            relevance_scoring_time_ms=random.uniform(50, 300) * trend_factor,
            integrated_workflow_time_ms=random.uniform(500, 1500) * trend_factor,
            peak_validation_memory_mb=random.uniform(400, 1200) * trend_factor,
            avg_validation_cpu_percent=random.uniform(30, 90),
            quality_flags_raised=random.randint(0, 3)
        )
        
        benchmark_data.append(metric)
    
    return benchmark_data


def generate_sample_api_metrics(num_samples: int = 50) -> list:
    """Generate sample API metrics data for demonstration."""
    api_metrics = []
    
    base_time = time.time() - (24 * 3600)  # 24 hours ago
    
    for i in range(num_samples):
        timestamp = base_time + (i * 1800)  # Every 30 minutes
        
        # Generate realistic API metrics
        cost_base = 0.005
        cost_trend = 1 + (i * 0.005)  # Slight cost increase over time
        
        metric = QualityAPIMetric(
            timestamp=timestamp,
            operation_name=f"quality_validation_{i % 5}",
            total_tokens=random.randint(100, 1000),
            cost_usd=cost_base * cost_trend * random.uniform(0.8, 1.5),
            response_time_ms=random.uniform(200, 2000),
            success=random.random() > 0.05,  # 95% success rate
            quality_validation_type=random.choice(["relevance", "factual_accuracy", "integrated"]),
            quality_score=random.uniform(75, 95),
            quality_validation_cost_usd=cost_base * 0.6 * cost_trend * random.uniform(0.8, 1.3),
            claims_extracted=random.randint(1, 10),
            claims_validated=random.randint(1, 8),
            validation_passed=random.random() > 0.1,  # 90% validation pass rate
            validation_confidence=random.uniform(70, 95)
        )
        
        api_metrics.append(metric)
    
    return api_metrics


async def demonstrate_comprehensive_reporting():
    """Demonstrate the complete performance reporting workflow."""
    print("🚀 Starting Comprehensive Performance Reporting Demonstration")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("demo_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    print("📊 Generating sample performance data...")
    benchmark_data = generate_sample_benchmark_data(100)
    api_metrics = generate_sample_api_metrics(80)
    
    print(f"   Generated {len(benchmark_data)} benchmark data points")
    print(f"   Generated {len(api_metrics)} API metrics data points")
    
    # 1. QUALITY PERFORMANCE REPORTER
    print("\n📋 1. GENERATING PERFORMANCE REPORTS")
    print("-" * 50)
    
    # Configure comprehensive reporting
    report_config = PerformanceReportConfiguration(
        report_name="Clinical Metabolomics Oracle - Performance Analysis",
        report_description="Comprehensive performance analysis of quality validation system",
        include_executive_summary=True,
        include_detailed_analysis=True,
        include_recommendations=True,
        analysis_period_hours=24,
        generate_charts=True,
        output_formats=[ReportFormat.JSON, ReportFormat.HTML, ReportFormat.CSV],
        output_directory=output_dir,
        performance_thresholds={
            'response_time_ms_threshold': 2000,
            'throughput_ops_per_sec_threshold': 5.0,
            'accuracy_threshold': 85.0,
            'cost_per_operation_threshold': 0.008,
            'memory_usage_mb_threshold': 1000,
            'error_rate_threshold': 5.0
        }
    )
    
    # Create and use reporter
    reporter = QualityPerformanceReporter(config=report_config, output_directory=output_dir)
    
    # Load data
    await reporter.load_benchmark_data(data=benchmark_data)
    await reporter.load_api_metrics_data(metrics_data=api_metrics)
    
    print("   Loaded data into performance reporter")
    
    # Generate comprehensive report
    report_data = await reporter.generate_comprehensive_report()
    print("   Generated comprehensive performance report")
    
    # Export reports
    exported_files = await reporter.export_report(report_data)
    print(f"   Exported reports: {list(exported_files.keys())}")
    
    # 2. PERFORMANCE DASHBOARD
    print("\n📊 2. GENERATING PERFORMANCE DASHBOARD")
    print("-" * 50)
    
    # Create dashboard
    dashboard_config = DashboardConfiguration(
        dashboard_title="Quality Validation Performance Dashboard",
        dashboard_subtitle="Real-time monitoring and analysis",
        theme="professional",
        generate_charts=True,
        auto_refresh=False  # Static for demo
    )
    
    dashboard = PerformanceDashboard(config=dashboard_config, output_directory=output_dir)
    
    # Load data into dashboard
    await dashboard.load_data(
        benchmark_data=benchmark_data,
        api_metrics_data=api_metrics
    )
    
    print("   Loaded data into dashboard generator")
    
    # Generate static dashboard
    dashboard_path = await dashboard.generate_static_dashboard("performance_dashboard.html")
    print(f"   Generated dashboard: {dashboard_path}")
    
    # 3. RECOMMENDATION ENGINE
    print("\n💡 3. GENERATING PERFORMANCE RECOMMENDATIONS")
    print("-" * 50)
    
    # Create recommendation engine
    rec_engine = RecommendationEngine(
        performance_thresholds=report_config.performance_thresholds,
        cost_targets={'cost_per_operation_usd': 0.008, 'monthly_budget_usd': 500},
        quality_targets={'min_quality_score': 85.0, 'target_accuracy_rate': 90.0}
    )
    
    # Load data
    await rec_engine.load_performance_data(
        benchmark_data=benchmark_data,
        api_metrics_data=api_metrics
    )
    
    print("   Loaded data into recommendation engine")
    
    # Generate recommendations
    recommendations = await rec_engine.generate_recommendations(
        focus_areas=[
            RecommendationType.PERFORMANCE_OPTIMIZATION,
            RecommendationType.COST_REDUCTION,
            RecommendationType.QUALITY_IMPROVEMENT
        ],
        max_recommendations=15
    )
    
    print(f"   Generated {len(recommendations)} performance recommendations")
    
    # Export recommendations
    rec_file = await rec_engine.export_recommendations(str(output_dir / "recommendations.json"))
    print(f"   Exported recommendations: {rec_file}")
    
    # Display top recommendations
    print("\n   📌 TOP 5 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"      {i}. {rec.title}")
        print(f"         Priority: {rec.priority.value.upper()}")
        print(f"         Impact: {rec.expected_impact_level.value}")
        print(f"         ROI: {rec.roi_estimate_percentage or 'N/A'}%")
        print()
    
    # 4. STATISTICAL ANALYZER
    print("\n📈 4. STATISTICAL ANALYSIS")
    print("-" * 50)
    
    # Create statistical analyzer
    analyzer = StatisticalAnalyzer(
        confidence_level=0.95,
        anomaly_sensitivity=0.05,
        trend_window_size=20
    )
    
    # Load data
    await analyzer.load_performance_data(
        benchmark_data=benchmark_data,
        api_metrics_data=api_metrics
    )
    
    print("   Loaded data into statistical analyzer")
    
    # Perform analyses
    key_metrics = ['average_latency_ms', 'quality_efficiency_score', 'error_rate_percent', 'cost_usd']
    
    # Trend analysis
    trend_results = await analyzer.analyze_trends(metrics=key_metrics)
    print(f"   Performed trend analysis on {len(trend_results)} metrics")
    
    # Anomaly detection
    anomaly_results = await analyzer.detect_anomalies(metrics=key_metrics)
    total_anomalies = sum(result.anomaly_count for result in anomaly_results.values())
    print(f"   Detected {total_anomalies} anomalies across {len(anomaly_results)} metrics")
    
    # Correlation analysis
    correlation_matrix = await analyzer.calculate_correlations(metrics=key_metrics)
    print(f"   Calculated correlations between {len(key_metrics)} metrics")
    
    # Predictive modeling
    try:
        model = await analyzer.build_predictive_model(
            target_metric='average_latency_ms',
            feature_metrics=['quality_efficiency_score', 'error_rate_percent']
        )
        print(f"   Built predictive model (R² = {model.r_squared_score:.3f})")
    except Exception as e:
        print(f"   Predictive modeling: {e}")
    
    # Export statistical analysis
    stats_file = await analyzer.export_analysis_results(str(output_dir / "statistical_analysis.json"))
    print(f"   Exported statistical analysis: {stats_file}")
    
    # Display key findings
    print("\n   📊 KEY STATISTICAL FINDINGS:")
    
    for metric, trend in trend_results.items():
        if trend.statistical_significance.value != "not_significant":
            print(f"      • {metric}: {trend.get_trend_description()}")
    
    for metric, anomalies in anomaly_results.items():
        if anomalies.anomaly_count > 0:
            print(f"      • {metric}: {anomalies.get_anomaly_summary()}")
    
    if correlation_matrix.strong_correlations:
        print(f"      • Found {len(correlation_matrix.strong_correlations)} strong correlations")
    
    # 5. INTEGRATION DEMONSTRATION
    print("\n🔗 5. INTEGRATED REPORTING WORKFLOW")
    print("-" * 50)
    
    # Use convenience function for end-to-end reporting
    print("   Running integrated reporting workflow...")
    
    integrated_reports = await generate_comprehensive_performance_report(
        benchmark_data=benchmark_data,
        api_metrics_data=api_metrics,
        config=report_config,
        output_directory=output_dir / "integrated"
    )
    
    print(f"   Generated integrated reports: {list(integrated_reports.keys())}")
    
    # Generate integrated dashboard
    integrated_dashboard = await generate_performance_dashboard(
        benchmark_data=benchmark_data,
        api_metrics_data=api_metrics,
        output_filename="integrated_dashboard.html",
        dashboard_title="Integrated Performance Dashboard"
    )
    
    print(f"   Generated integrated dashboard: {integrated_dashboard}")
    
    # 6. SUMMARY AND NEXT STEPS
    print("\n✅ DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print("📁 Generated Files:")
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            print(f"   • {file_path.relative_to(output_dir)}")
    
    print("\n🎯 Key Capabilities Demonstrated:")
    print("   ✓ Comprehensive performance report generation")
    print("   ✓ Interactive dashboard creation")
    print("   ✓ AI-powered performance recommendations")
    print("   ✓ Advanced statistical analysis and forecasting")
    print("   ✓ Multi-format report export (JSON, HTML, CSV)")
    print("   ✓ Real-time performance monitoring capabilities")
    print("   ✓ Trend analysis and anomaly detection")
    print("   ✓ Correlation analysis between metrics")
    print("   ✓ Predictive modeling for performance forecasting")
    
    print("\n📖 Next Steps:")
    print("   1. Open the generated HTML files in your browser")
    print("   2. Review the JSON reports for detailed data")
    print("   3. Implement the top recommendations")
    print("   4. Set up real-time dashboard monitoring")
    print("   5. Integrate with your existing monitoring systems")
    
    # Performance metrics for the demo itself
    print(f"\n⏱️  Demo completed in {time.time() - demo_start_time:.2f} seconds")
    

def demonstrate_real_time_dashboard():
    """Demonstrate real-time dashboard capabilities."""
    print("\n🚀 REAL-TIME DASHBOARD DEMONSTRATION")
    print("=" * 50)
    
    try:
        # This would start a real-time dashboard server
        # Commented out to avoid blocking in example
        """
        dashboard = PerformanceDashboard()
        await dashboard.load_data(benchmark_data=generate_sample_benchmark_data(20))
        
        print("Starting real-time dashboard server...")
        print("Dashboard will be available at: http://localhost:8050")
        print("Press Ctrl+C to stop the server")
        
        await dashboard.start_realtime_dashboard_server(
            host="127.0.0.1",
            port=8050,
            debug=False
        )
        """
        
        print("   📊 Real-time dashboard server capabilities:")
        print("      • Live data updates every 30 seconds")
        print("      • Interactive charts with zoom and pan")
        print("      • Real-time anomaly detection alerts")
        print("      • Performance threshold monitoring")
        print("      • Multi-metric correlation displays")
        print("      • Automatic chart refresh and scaling")
        
        print("\n   🔧 To start real-time dashboard:")
        print("      1. Uncomment the server code in this function")
        print("      2. Run: python example_usage.py --real-time")
        print("      3. Open browser to http://localhost:8050")
        
    except Exception as e:
        print(f"   ⚠️  Real-time dashboard demo: {e}")


async def demonstrate_advanced_analytics():
    """Demonstrate advanced analytics capabilities."""
    print("\n📊 ADVANCED ANALYTICS DEMONSTRATION")
    print("=" * 50)
    
    # Generate time series data with clear patterns
    print("   Generating synthetic data with known patterns...")
    
    time_series_data = {}
    timestamps = []
    
    # Generate 100 data points over 7 days
    base_time = time.time() - (7 * 24 * 3600)
    
    for i in range(100):
        timestamp = base_time + (i * 3600)  # Hourly data
        timestamps.append(timestamp)
        
        # Response time with daily pattern + trend + noise + anomalies
        daily_pattern = 200 + 100 * math.sin(2 * math.pi * i / 24)  # Daily cycle
        trend = i * 5  # Gradual increase
        noise = random.uniform(-50, 50)
        
        # Add occasional spikes (anomalies)
        if i in [25, 45, 78]:  # Specific anomalies
            spike = 800
        else:
            spike = 0
            
        response_time = max(100, daily_pattern + trend + noise + spike)
        
        # Quality score inversely related to response time with noise
        quality_score = max(60, min(100, 100 - (response_time - 200) / 20 + random.uniform(-10, 10)))
        
        # Error rate correlated with response time
        error_rate = max(0, min(15, (response_time - 200) / 200 * 5 + random.uniform(-1, 1)))
        
        # Cost related to processing time and quality
        cost = (response_time / 1000) * 0.01 + (quality_score / 100) * 0.005 + random.uniform(-0.002, 0.002)
        
        time_series_data.setdefault('response_time_ms', []).append((timestamp, response_time))
        time_series_data.setdefault('quality_score', []).append((timestamp, quality_score))
        time_series_data.setdefault('error_rate_percent', []).append((timestamp, error_rate))
        time_series_data.setdefault('cost_usd', []).append((timestamp, max(0.001, cost)))
    
    # Create analyzer and load time series data
    analyzer = StatisticalAnalyzer()
    await analyzer.load_performance_data(time_series_data=time_series_data)
    
    print(f"   Loaded {len(time_series_data)} metrics with {len(timestamps)} data points each")
    
    # Comprehensive analysis
    print("\n   🔍 Performing comprehensive statistical analysis...")
    
    # Trend analysis
    trends = await analyzer.analyze_trends()
    print(f"      Trend Analysis: {len(trends)} metrics analyzed")
    
    for metric, trend in trends.items():
        print(f"         {metric}: {trend.get_trend_description()}")
    
    # Anomaly detection with multiple methods
    anomaly_methods = ["isolation_forest", "statistical", "zscore"]
    
    for method in anomaly_methods:
        try:
            anomalies = await analyzer.detect_anomalies(method=method)
            total_anomalies = sum(r.anomaly_count for r in anomalies.values())
            print(f"      Anomaly Detection ({method}): {total_anomalies} anomalies found")
            
            # Show specific anomalies for response_time
            if 'response_time_ms' in anomalies:
                rt_anomalies = anomalies['response_time_ms']
                if rt_anomalies.anomalies_detected:
                    print(f"         Response time anomalies: {[a['index'] for a in rt_anomalies.anomalies_detected]}")
            
        except Exception as e:
            print(f"      Anomaly Detection ({method}): Failed - {e}")
    
    # Correlation analysis
    correlations = await analyzer.calculate_correlations()
    print(f"      Correlation Analysis: {len(correlations.strong_correlations)} strong correlations")
    
    for metric1, metric2, coeff in correlations.strong_correlations:
        print(f"         {metric1} ↔ {metric2}: {coeff:.3f} ({correlations.get_correlation_strength(metric1, metric2)})")
    
    # Predictive modeling
    print("\n   🔮 Building predictive models...")
    
    target_metrics = ['response_time_ms', 'quality_score']
    
    for target in target_metrics:
        try:
            feature_metrics = [m for m in time_series_data.keys() if m != target][:2]  # Max 2 features
            model = await analyzer.build_predictive_model(target, feature_metrics)
            
            print(f"      {target} Model:")
            print(f"         R² Score: {model.r_squared_score:.3f}")
            print(f"         Features: {', '.join(model.feature_metrics)}")
            print(f"         Top Feature: {max(model.feature_importance.items(), key=lambda x: x[1])[0] if model.feature_importance else 'N/A'}")
            
        except Exception as e:
            print(f"      {target} Model: Failed - {e}")
    
    # Generate forecast
    print("\n   📈 Performance forecasting...")
    
    # Simple forecast based on trend analysis
    for metric, trend in trends.items():
        if trend.short_term_forecast:
            current_value = time_series_data[metric][-1][1]
            forecast_change = (trend.short_term_forecast[-1] - current_value) if trend.short_term_forecast else 0
            
            print(f"      {metric}:")
            print(f"         Current: {current_value:.2f}")
            print(f"         5-step forecast change: {forecast_change:+.2f}")
            print(f"         Trend confidence: {trend.trend_confidence:.2f}")


if __name__ == "__main__":
    import sys
    
    # Track demo execution time
    demo_start_time = time.time()
    
    if "--advanced" in sys.argv:
        # Run advanced analytics demonstration
        asyncio.run(demonstrate_advanced_analytics())
    
    elif "--real-time" in sys.argv:
        # Run real-time dashboard demonstration
        demonstrate_real_time_dashboard()
    
    else:
        # Run comprehensive demonstration
        asyncio.run(demonstrate_comprehensive_reporting())