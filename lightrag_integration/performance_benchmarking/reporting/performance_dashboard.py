#!/usr/bin/env python3
"""
Performance Dashboard Generator for Clinical Metabolomics Oracle.

This module creates interactive dashboards for visualizing quality validation
performance metrics, trends, and insights. It provides real-time monitoring
capabilities and comprehensive visualization of system performance.

Classes:
    - DashboardConfiguration: Configuration for dashboard generation
    - DashboardComponent: Individual dashboard component/widget
    - PerformanceDashboard: Main dashboard generator
    - DashboardServer: Optional server for real-time dashboard hosting

Key Features:
    - Interactive charts and visualizations
    - Real-time performance monitoring
    - Customizable dashboard layouts
    - Multi-metric correlation displays
    - Performance alerting and notifications
    - Export capabilities for dashboard views
    - Mobile-responsive design

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
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import queue
from collections import defaultdict, deque

# Visualization and dashboard imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    import dash
    from dash import dcc, html, Input, Output, callback
    import dash_bootstrap_components as dbc
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Dashboard libraries not available: {e}. Dashboard functionality will be limited.")
    DASHBOARD_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class DashboardTheme(Enum):
    """Available dashboard themes."""
    PROFESSIONAL = "professional"
    DARK = "dark"
    LIGHT = "light"
    CLINICAL = "clinical"
    MINIMAL = "minimal"


class ChartType(Enum):
    """Types of charts available for dashboard."""
    LINE_CHART = "line"
    BAR_CHART = "bar"
    SCATTER_PLOT = "scatter"
    PIE_CHART = "pie"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"


class UpdateFrequency(Enum):
    """Dashboard update frequencies."""
    REAL_TIME = "real_time"  # Every 5 seconds
    HIGH = "high"           # Every 30 seconds
    MEDIUM = "medium"       # Every 2 minutes
    LOW = "low"            # Every 10 minutes
    MANUAL = "manual"       # Manual refresh only


@dataclass
class DashboardConfiguration:
    """Configuration settings for dashboard generation."""
    
    # Dashboard identification
    dashboard_title: str = "Quality Performance Dashboard"
    dashboard_subtitle: str = "Real-time Quality Validation Performance Monitoring"
    
    # Layout and appearance
    theme: DashboardTheme = DashboardTheme.PROFESSIONAL
    layout_columns: int = 12  # Bootstrap-style grid
    responsive_design: bool = True
    show_navigation: bool = True
    show_filters: bool = True
    
    # Update and refresh settings
    update_frequency: UpdateFrequency = UpdateFrequency.HIGH
    auto_refresh: bool = True
    cache_duration_minutes: int = 5
    
    # Chart and visualization settings
    default_chart_height: int = 400
    default_chart_width: Optional[int] = None  # Auto-width
    show_chart_controls: bool = True
    enable_zoom: bool = True
    enable_pan: bool = True
    
    # Data filtering
    default_time_range_hours: int = 24
    max_data_points_per_chart: int = 1000
    enable_outlier_filtering: bool = True
    outlier_threshold_std: float = 2.0
    
    # Performance monitoring
    show_health_indicators: bool = True
    show_alert_notifications: bool = True
    enable_performance_alerts: bool = True
    
    # Export and sharing
    enable_export_features: bool = True
    enable_screenshot: bool = True
    enable_data_download: bool = True
    
    # Customization
    custom_css_file: Optional[Path] = None
    custom_logo_file: Optional[Path] = None
    brand_colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#17a2b8',
        'background': '#ffffff'
    })


@dataclass
class DashboardComponent:
    """Individual dashboard component or widget."""
    
    component_id: str
    title: str
    description: Optional[str] = None
    chart_type: ChartType = ChartType.LINE_CHART
    
    # Layout positioning (Bootstrap grid)
    column_span: int = 6  # Out of 12
    row_span: int = 1
    order: int = 0
    
    # Data configuration
    data_source: str = "benchmark_data"  # "benchmark_data", "api_metrics", "correlation_data"
    metrics: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Display configuration
    show_title: bool = True
    show_legend: bool = True
    show_grid: bool = True
    color_scheme: Optional[List[str]] = None
    
    # Interactivity
    enable_hover: bool = True
    enable_selection: bool = False
    enable_crossfilter: bool = False
    
    # Refresh settings
    auto_update: bool = True
    update_interval_seconds: int = 30
    
    # Thresholds and alerts
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    show_threshold_lines: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert component to dictionary."""
        result = asdict(self)
        result['chart_type'] = self.chart_type.value
        return result


class PerformanceDashboard:
    """
    Interactive performance dashboard generator for quality validation systems.
    
    Creates comprehensive, real-time dashboards for monitoring quality validation
    performance, system health, and operational metrics.
    """
    
    def __init__(self,
                 config: Optional[DashboardConfiguration] = None,
                 output_directory: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the performance dashboard generator.
        
        Args:
            config: Dashboard configuration settings
            output_directory: Directory for saving dashboard files
            logger: Logger instance for dashboard operations
        """
        self.config = config or DashboardConfiguration()
        self.output_directory = output_directory or Path("dashboard_output")
        self.logger = logger or logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.benchmark_data: List[Any] = []
        self.api_metrics_data: List[Any] = []
        self.correlation_data: List[Any] = []
        
        # Dashboard components
        self.components: List[DashboardComponent] = []
        self.layout_rows: List[List[DashboardComponent]] = []
        
        # Real-time data management
        self.data_queue = queue.Queue()
        self.update_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Performance tracking
        self.performance_metrics = {
            'dashboard_load_time': 0.0,
            'chart_render_time': defaultdict(float),
            'data_update_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        # Initialize default components
        self._initialize_default_components()
        
        self.logger.info("PerformanceDashboard initialized")
    
    def _initialize_default_components(self) -> None:
        """Initialize default dashboard components."""
        
        # Response Time Timeline
        self.components.append(DashboardComponent(
            component_id="response_time_timeline",
            title="Response Time Trends",
            description="Response time performance over time",
            chart_type=ChartType.LINE_CHART,
            column_span=8,
            order=1,
            data_source="benchmark_data",
            metrics=["average_latency_ms", "p95_latency_ms"],
            alert_thresholds={"average_latency_ms": 2000},
            show_threshold_lines=True
        ))
        
        # Quality Score Distribution
        self.components.append(DashboardComponent(
            component_id="quality_score_distribution",
            title="Quality Score Distribution",
            description="Distribution of quality efficiency scores",
            chart_type=ChartType.HISTOGRAM,
            column_span=4,
            order=2,
            data_source="benchmark_data",
            metrics=["quality_efficiency_score"],
            color_scheme=["#2ca02c"]
        ))
        
        # System Health Gauge
        self.components.append(DashboardComponent(
            component_id="system_health_gauge",
            title="System Health",
            description="Overall system health indicator",
            chart_type=ChartType.GAUGE,
            column_span=4,
            order=3,
            data_source="benchmark_data",
            metrics=["overall_health_score"],
            alert_thresholds={"overall_health_score": 70}
        ))
        
        # Cost Analysis
        self.components.append(DashboardComponent(
            component_id="cost_analysis",
            title="Cost Analysis",
            description="API cost breakdown and trends",
            chart_type=ChartType.BAR_CHART,
            column_span=4,
            order=4,
            data_source="api_metrics",
            metrics=["cost_usd", "quality_validation_cost_usd"]
        ))
        
        # Processing Stage Performance
        self.components.append(DashboardComponent(
            component_id="stage_performance",
            title="Processing Stage Performance",
            description="Performance by validation stage",
            chart_type=ChartType.BAR_CHART,
            column_span=4,
            order=5,
            data_source="benchmark_data",
            metrics=["claim_extraction_time_ms", "factual_validation_time_ms", 
                    "relevance_scoring_time_ms", "integrated_workflow_time_ms"]
        ))
        
        # Error Rate Timeline
        self.components.append(DashboardComponent(
            component_id="error_rate_timeline",
            title="Error Rate Trends",
            description="Error rates over time",
            chart_type=ChartType.LINE_CHART,
            column_span=6,
            order=6,
            data_source="benchmark_data",
            metrics=["error_rate_percent"],
            alert_thresholds={"error_rate_percent": 5.0},
            show_threshold_lines=True,
            color_scheme=["#d62728"]
        ))
        
        # Quality vs Performance Correlation
        self.components.append(DashboardComponent(
            component_id="quality_performance_correlation",
            title="Quality vs Performance",
            description="Correlation between quality scores and response times",
            chart_type=ChartType.SCATTER_PLOT,
            column_span=6,
            order=7,
            data_source="benchmark_data",
            metrics=["quality_efficiency_score", "average_latency_ms"],
            enable_selection=True
        ))
        
        # Resource Utilization
        self.components.append(DashboardComponent(
            component_id="resource_utilization",
            title="Resource Utilization",
            description="Memory and CPU usage patterns",
            chart_type=ChartType.LINE_CHART,
            column_span=6,
            order=8,
            data_source="benchmark_data",
            metrics=["peak_validation_memory_mb", "avg_validation_cpu_percent"],
            alert_thresholds={"peak_validation_memory_mb": 1500, "avg_validation_cpu_percent": 80}
        ))
        
        # Operations Summary Table
        self.components.append(DashboardComponent(
            component_id="operations_summary",
            title="Operations Summary",
            description="Summary of recent operations",
            chart_type=ChartType.TABLE,
            column_span=6,
            order=9,
            data_source="benchmark_data",
            metrics=["scenario_name", "operations_count", "success_count", 
                    "average_latency_ms", "quality_efficiency_score"],
            show_legend=False
        ))
    
    async def load_data(self,
                       benchmark_data: Optional[List[Any]] = None,
                       api_metrics_data: Optional[List[Any]] = None,
                       correlation_data: Optional[List[Any]] = None) -> int:
        """
        Load data for dashboard visualization.
        
        Args:
            benchmark_data: Quality validation benchmark data
            api_metrics_data: API metrics data
            correlation_data: Correlation analysis data
            
        Returns:
            Total number of data points loaded
        """
        total_loaded = 0
        
        try:
            if benchmark_data:
                self.benchmark_data = benchmark_data
                total_loaded += len(benchmark_data)
                self.logger.info(f"Loaded {len(benchmark_data)} benchmark data points")
            
            if api_metrics_data:
                self.api_metrics_data = api_metrics_data
                total_loaded += len(api_metrics_data)
                self.logger.info(f"Loaded {len(api_metrics_data)} API metrics data points")
            
            if correlation_data:
                self.correlation_data = correlation_data
                total_loaded += len(correlation_data)
                self.logger.info(f"Loaded {len(correlation_data)} correlation data points")
            
            self.logger.info(f"Total data points loaded: {total_loaded}")
            
        except Exception as e:
            self.logger.error(f"Error loading dashboard data: {e}")
            raise
        
        return total_loaded
    
    def add_component(self, component: DashboardComponent) -> None:
        """Add a custom component to the dashboard."""
        self.components.append(component)
        self.logger.info(f"Added dashboard component: {component.component_id}")
    
    def remove_component(self, component_id: str) -> bool:
        """Remove a component from the dashboard."""
        original_count = len(self.components)
        self.components = [c for c in self.components if c.component_id != component_id]
        removed = len(self.components) < original_count
        
        if removed:
            self.logger.info(f"Removed dashboard component: {component_id}")
        else:
            self.logger.warning(f"Component not found: {component_id}")
        
        return removed
    
    def update_component_config(self, component_id: str, **updates) -> bool:
        """Update configuration of an existing component."""
        for component in self.components:
            if component.component_id == component_id:
                for key, value in updates.items():
                    if hasattr(component, key):
                        setattr(component, key, value)
                        
                self.logger.info(f"Updated component {component_id}: {updates}")
                return True
        
        self.logger.warning(f"Component not found for update: {component_id}")
        return False
    
    async def generate_static_dashboard(self, output_filename: str = "dashboard.html") -> Path:
        """
        Generate static HTML dashboard file.
        
        Args:
            output_filename: Name of the output HTML file
            
        Returns:
            Path to the generated dashboard file
        """
        if not DASHBOARD_AVAILABLE:
            raise RuntimeError("Dashboard libraries not available. Install plotly and dash to use dashboard features.")
        
        start_time = time.time()
        
        try:
            self.logger.info("Generating static dashboard...")
            
            # Create dashboard layout
            dashboard_html = await self._create_static_dashboard_html()
            
            # Save to file
            output_path = self.output_directory / output_filename
            with open(output_path, 'w') as f:
                f.write(dashboard_html)
            
            # Track performance
            self.performance_metrics['dashboard_load_time'] = time.time() - start_time
            
            self.logger.info(f"Static dashboard generated: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating static dashboard: {e}")
            raise
    
    async def _create_static_dashboard_html(self) -> str:
        """Create static dashboard HTML content."""
        
        # Generate charts for each component
        charts_html = []
        
        # Sort components by order
        sorted_components = sorted(self.components, key=lambda c: c.order)
        
        # Create layout rows
        current_row = []
        current_row_width = 0
        
        for component in sorted_components:
            if current_row_width + component.column_span > self.config.layout_columns:
                # Start new row
                if current_row:
                    self.layout_rows.append(current_row)
                current_row = [component]
                current_row_width = component.column_span
            else:
                current_row.append(component)
                current_row_width += component.column_span
        
        # Add final row
        if current_row:
            self.layout_rows.append(current_row)
        
        # Generate charts for each component
        for component in sorted_components:
            try:
                chart_html = await self._generate_component_chart(component)
                charts_html.append(chart_html)
            except Exception as e:
                self.logger.error(f"Error generating chart for component {component.component_id}: {e}")
                charts_html.append(f"<div>Error loading chart: {component.title}</div>")
        
        # Create complete HTML
        dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.dashboard_title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        {self._get_dashboard_css()}
    </style>
</head>
<body>
    <div class="container-fluid">
        <header class="dashboard-header">
            <h1>{self.config.dashboard_title}</h1>
            <p class="text-muted">{self.config.dashboard_subtitle}</p>
            <div class="update-info">
                <small>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
        </header>
        
        <div class="dashboard-content">
            {self._create_dashboard_rows_html(charts_html)}
        </div>
        
        <footer class="dashboard-footer">
            <p class="text-center text-muted">
                Generated by Clinical Metabolomics Oracle Performance Dashboard
            </p>
        </footer>
    </div>
    
    <script>
        {self._get_dashboard_javascript()}
    </script>
</body>
</html>"""
        
        return dashboard_html
    
    def _create_dashboard_rows_html(self, charts_html: List[str]) -> str:
        """Create HTML for dashboard rows and columns."""
        rows_html = []
        chart_index = 0
        
        for row in self.layout_rows:
            row_html = '<div class="row mb-4">'
            
            for component in row:
                col_class = f"col-md-{component.column_span}"
                chart_html = charts_html[chart_index] if chart_index < len(charts_html) else ""
                
                row_html += f'''
                <div class="{col_class}">
                    <div class="dashboard-component">
                        {f'<h5 class="component-title">{component.title}</h5>' if component.show_title else ''}
                        {f'<p class="component-description text-muted">{component.description}</p>' if component.description else ''}
                        <div class="component-content">
                            {chart_html}
                        </div>
                    </div>
                </div>
                '''
                chart_index += 1
            
            row_html += '</div>'
            rows_html.append(row_html)
        
        return '\n'.join(rows_html)
    
    async def _generate_component_chart(self, component: DashboardComponent) -> str:
        """Generate chart HTML for a specific component."""
        try:
            # Get data for the component
            chart_data = self._get_component_data(component)
            
            if not chart_data:
                return f"<div class='alert alert-info'>No data available for {component.title}</div>"
            
            # Create chart based on type
            if component.chart_type == ChartType.LINE_CHART:
                fig = self._create_line_chart(component, chart_data)
            elif component.chart_type == ChartType.BAR_CHART:
                fig = self._create_bar_chart(component, chart_data)
            elif component.chart_type == ChartType.SCATTER_PLOT:
                fig = self._create_scatter_plot(component, chart_data)
            elif component.chart_type == ChartType.PIE_CHART:
                fig = self._create_pie_chart(component, chart_data)
            elif component.chart_type == ChartType.HISTOGRAM:
                fig = self._create_histogram(component, chart_data)
            elif component.chart_type == ChartType.GAUGE:
                fig = self._create_gauge_chart(component, chart_data)
            elif component.chart_type == ChartType.TABLE:
                return self._create_table(component, chart_data)
            else:
                fig = self._create_line_chart(component, chart_data)  # Default fallback
            
            # Configure chart appearance
            self._configure_chart_appearance(fig, component)
            
            # Convert to HTML
            chart_html = fig.to_html(
                include_plotlyjs=False,
                div_id=f"chart_{component.component_id}",
                config={'displayModeBar': self.config.show_chart_controls}
            )
            
            return chart_html
            
        except Exception as e:
            self.logger.error(f"Error generating chart for component {component.component_id}: {e}")
            return f"<div class='alert alert-danger'>Error loading chart: {e}</div>"
    
    def _get_component_data(self, component: DashboardComponent) -> Dict[str, List[Any]]:
        """Extract data for a specific component."""
        data_dict = {}
        
        # Select data source
        if component.data_source == "benchmark_data":
            source_data = self.benchmark_data
        elif component.data_source == "api_metrics":
            source_data = self.api_metrics_data
        elif component.data_source == "correlation_data":
            source_data = self.correlation_data
        else:
            return {}
        
        if not source_data:
            return {}
        
        # Extract specified metrics
        for metric in component.metrics:
            metric_values = []
            timestamps = []
            
            for data_point in source_data:
                # Handle different data structures
                if hasattr(data_point, metric):
                    value = getattr(data_point, metric)
                    if value is not None and (not isinstance(value, (int, float)) or value > 0):
                        metric_values.append(value)
                        
                        # Try to get timestamp
                        timestamp = getattr(data_point, 'timestamp', None) or getattr(data_point, 'start_time', None)
                        if timestamp:
                            timestamps.append(datetime.fromtimestamp(timestamp))
                        else:
                            timestamps.append(datetime.now())
                
                elif isinstance(data_point, dict) and metric in data_point:
                    value = data_point[metric]
                    if value is not None and (not isinstance(value, (int, float)) or value > 0):
                        metric_values.append(value)
                        timestamps.append(data_point.get('timestamp', datetime.now()))
            
            if metric_values:
                data_dict[metric] = metric_values
                if not data_dict.get('timestamps'):
                    data_dict['timestamps'] = timestamps
        
        return data_dict
    
    def _create_line_chart(self, component: DashboardComponent, data: Dict[str, List[Any]]) -> go.Figure:
        """Create line chart."""
        fig = go.Figure()
        
        timestamps = data.get('timestamps', list(range(len(list(data.values())[0]))))
        
        for i, metric in enumerate(component.metrics):
            if metric in data:
                color = None
                if component.color_scheme and i < len(component.color_scheme):
                    color = component.color_scheme[i]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=data[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color) if color else None
                ))
        
        # Add threshold lines if configured
        if component.show_threshold_lines and component.alert_thresholds:
            for metric, threshold in component.alert_thresholds.items():
                if metric in data:
                    fig.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"{metric} threshold"
                    )
        
        return fig
    
    def _create_bar_chart(self, component: DashboardComponent, data: Dict[str, List[Any]]) -> go.Figure:
        """Create bar chart."""
        fig = go.Figure()
        
        # Use metric names as categories
        categories = [metric.replace('_', ' ').title() for metric in component.metrics if metric in data]
        values = [statistics.mean(data[metric]) if data[metric] else 0 for metric in component.metrics if metric in data]
        
        colors = component.color_scheme if component.color_scheme else None
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors
        ))
        
        return fig
    
    def _create_scatter_plot(self, component: DashboardComponent, data: Dict[str, List[Any]]) -> go.Figure:
        """Create scatter plot."""
        fig = go.Figure()
        
        if len(component.metrics) >= 2:
            x_metric = component.metrics[0]
            y_metric = component.metrics[1]
            
            if x_metric in data and y_metric in data:
                # Ensure same length
                min_len = min(len(data[x_metric]), len(data[y_metric]))
                
                fig.add_trace(go.Scatter(
                    x=data[x_metric][:min_len],
                    y=data[y_metric][:min_len],
                    mode='markers',
                    name=f"{x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()}",
                    marker=dict(
                        size=8,
                        color=data[x_metric][:min_len] if len(data[x_metric]) > 0 else 'blue',
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
        
        return fig
    
    def _create_histogram(self, component: DashboardComponent, data: Dict[str, List[Any]]) -> go.Figure:
        """Create histogram."""
        fig = go.Figure()
        
        for i, metric in enumerate(component.metrics):
            if metric in data and data[metric]:
                color = None
                if component.color_scheme and i < len(component.color_scheme):
                    color = component.color_scheme[i]
                
                fig.add_trace(go.Histogram(
                    x=data[metric],
                    name=metric.replace('_', ' ').title(),
                    marker_color=color,
                    opacity=0.7
                ))
        
        return fig
    
    def _create_pie_chart(self, component: DashboardComponent, data: Dict[str, List[Any]]) -> go.Figure:
        """Create pie chart."""
        fig = go.Figure()
        
        # Use aggregated values for pie chart
        labels = [metric.replace('_', ' ').title() for metric in component.metrics if metric in data]
        values = [sum(data[metric]) if data[metric] else 0 for metric in component.metrics if metric in data]
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=component.color_scheme) if component.color_scheme else None
        ))
        
        return fig
    
    def _create_gauge_chart(self, component: DashboardComponent, data: Dict[str, List[Any]]) -> go.Figure:
        """Create gauge chart."""
        fig = go.Figure()
        
        if component.metrics and component.metrics[0] in data:
            metric = component.metrics[0]
            current_value = statistics.mean(data[metric]) if data[metric] else 0
            
            # Determine gauge range and color
            gauge_max = 100  # Default
            if component.alert_thresholds and metric in component.alert_thresholds:
                gauge_max = max(gauge_max, component.alert_thresholds[metric] * 1.2)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=current_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': metric.replace('_', ' ').title()},
                gauge={
                    'axis': {'range': [None, gauge_max]},
                    'bar': {'color': self.config.brand_colors.get('primary', '#1f77b4')},
                    'steps': [
                        {'range': [0, gauge_max * 0.5], 'color': "lightgray"},
                        {'range': [gauge_max * 0.5, gauge_max * 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': component.alert_thresholds.get(metric, gauge_max * 0.9)
                    }
                }
            ))
        
        return fig
    
    def _create_table(self, component: DashboardComponent, data: Dict[str, List[Any]]) -> str:
        """Create HTML table."""
        if not data:
            return "<div class='alert alert-info'>No data available</div>"
        
        # Create table headers
        headers = [metric.replace('_', ' ').title() for metric in component.metrics if metric in data]
        
        # Get table data
        table_data = []
        max_rows = min(20, max(len(values) for values in data.values() if values))  # Limit to 20 rows
        
        for i in range(max_rows):
            row = []
            for metric in component.metrics:
                if metric in data and i < len(data[metric]):
                    value = data[metric][i]
                    if isinstance(value, float):
                        row.append(f"{value:.2f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("-")
            table_data.append(row)
        
        # Generate HTML table
        table_html = '<table class="table table-striped table-sm">'
        table_html += '<thead><tr>'
        for header in headers:
            table_html += f'<th>{header}</th>'
        table_html += '</tr></thead><tbody>'
        
        for row in table_data:
            table_html += '<tr>'
            for cell in row:
                table_html += f'<td>{cell}</td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table>'
        
        return table_html
    
    def _configure_chart_appearance(self, fig: go.Figure, component: DashboardComponent) -> None:
        """Configure chart appearance and styling."""
        fig.update_layout(
            height=self.config.default_chart_height,
            width=self.config.default_chart_width,
            showlegend=component.show_legend,
            plot_bgcolor=self.config.brand_colors.get('background', 'white'),
            paper_bgcolor=self.config.brand_colors.get('background', 'white'),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#333333"
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Configure grid
        if component.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Configure hover
        if component.enable_hover:
            fig.update_traces(hovertemplate='%{y}<extra></extra>')
        
        # Configure zoom and pan
        fig.update_layout(
            xaxis=dict(fixedrange=not self.config.enable_zoom),
            yaxis=dict(fixedrange=not self.config.enable_zoom)
        )
    
    def _get_dashboard_css(self) -> str:
        """Get CSS styles for the dashboard."""
        theme_colors = self.config.brand_colors
        
        return f"""
        body {{
            background-color: {theme_colors.get('background', '#ffffff')};
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        .dashboard-header {{
            background: linear-gradient(135deg, {theme_colors.get('primary', '#1f77b4')} 0%, {theme_colors.get('secondary', '#ff7f0e')} 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .dashboard-header h1 {{
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .dashboard-component {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: 100%;
        }}
        
        .component-title {{
            color: {theme_colors.get('primary', '#1f77b4')};
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .component-description {{
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        
        .component-content {{
            height: calc(100% - 60px);
        }}
        
        .dashboard-footer {{
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid #e0e0e0;
        }}
        
        .update-info {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255,255,255,0.2);
            padding: 5px 10px;
            border-radius: 5px;
        }}
        
        .alert {{
            margin: 10px 0;
        }}
        
        .table {{
            font-size: 0.85em;
        }}
        
        .table th {{
            background-color: {theme_colors.get('primary', '#1f77b4')};
            color: white;
            border: none;
        }}
        
        @media (max-width: 768px) {{
            .dashboard-header {{
                padding: 20px;
            }}
            
            .update-info {{
                position: static;
                margin-top: 10px;
            }}
            
            .dashboard-component {{
                margin-bottom: 20px;
            }}
        }}
        """
    
    def _get_dashboard_javascript(self) -> str:
        """Get JavaScript for dashboard interactivity."""
        return """
        // Auto-refresh functionality
        function refreshDashboard() {
            if (confirm('Refresh dashboard with latest data?')) {
                location.reload();
            }
        }
        
        // Add refresh button functionality
        document.addEventListener('DOMContentLoaded', function() {
            // Add refresh button to header if auto-refresh is enabled
            const header = document.querySelector('.dashboard-header');
            if (header) {
                const refreshBtn = document.createElement('button');
                refreshBtn.innerHTML = '🔄 Refresh';
                refreshBtn.className = 'btn btn-light btn-sm';
                refreshBtn.onclick = refreshDashboard;
                refreshBtn.style.position = 'absolute';
                refreshBtn.style.top = '20px';
                refreshBtn.style.left = '20px';
                header.appendChild(refreshBtn);
            }
            
            // Add chart responsiveness
            window.addEventListener('resize', function() {
                const charts = document.querySelectorAll('.js-plotly-plot');
                charts.forEach(chart => {
                    Plotly.Plots.resize(chart);
                });
            });
        });
        
        // Performance monitoring
        console.log('Dashboard loaded at:', new Date().toLocaleString());
        """
    
    async def start_realtime_dashboard_server(self, 
                                            host: str = "127.0.0.1",
                                            port: int = 8050,
                                            debug: bool = False) -> None:
        """
        Start real-time dashboard server using Dash.
        
        Args:
            host: Server host address
            port: Server port
            debug: Enable debug mode
        """
        if not DASHBOARD_AVAILABLE:
            raise RuntimeError("Dashboard libraries not available. Install dash to use real-time dashboard server.")
        
        self.logger.info(f"Starting real-time dashboard server at http://{host}:{port}")
        
        # Create Dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.title = self.config.dashboard_title
        
        # Create layout
        app.layout = self._create_dash_layout()
        
        # Add callbacks for interactivity
        self._register_dash_callbacks(app)
        
        # Start server
        app.run_server(host=host, port=port, debug=debug)
    
    def _create_dash_layout(self):
        """Create Dash layout for real-time dashboard."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1(self.config.dashboard_title, className="text-center mb-1"),
                    html.P(self.config.dashboard_subtitle, className="text-center text-muted mb-4"),
                    html.P(f"Last updated: {datetime.now().strftime('%H:%M:%S')}", 
                          id="last-updated", className="text-center text-muted")
                ])
            ]),
            
            html.Div(id="dashboard-content"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000 if self.config.update_frequency == UpdateFrequency.HIGH else 120*1000,
                n_intervals=0
            )
        ], fluid=True)
    
    def _register_dash_callbacks(self, app):
        """Register callbacks for dashboard interactivity."""
        
        @app.callback(
            [Output('dashboard-content', 'children'),
             Output('last-updated', 'children')],
            Input('interval-component', 'n_intervals')
        )
        def update_dashboard_content(n):
            # Regenerate charts with latest data
            charts = []
            
            for component in sorted(self.components, key=lambda c: c.order):
                try:
                    chart_data = self._get_component_data(component)
                    
                    if chart_data:
                        # Create Dash/Plotly figure
                        if component.chart_type == ChartType.LINE_CHART:
                            fig = self._create_line_chart(component, chart_data)
                        elif component.chart_type == ChartType.BAR_CHART:
                            fig = self._create_bar_chart(component, chart_data)
                        elif component.chart_type == ChartType.SCATTER_PLOT:
                            fig = self._create_scatter_plot(component, chart_data)
                        elif component.chart_type == ChartType.HISTOGRAM:
                            fig = self._create_histogram(component, chart_data)
                        elif component.chart_type == ChartType.GAUGE:
                            fig = self._create_gauge_chart(component, chart_data)
                        else:
                            fig = self._create_line_chart(component, chart_data)
                        
                        self._configure_chart_appearance(fig, component)
                        
                        charts.append(
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5(component.title)),
                                    dbc.CardBody([
                                        html.P(component.description, className="text-muted") if component.description else html.Div(),
                                        dcc.Graph(figure=fig)
                                    ])
                                ])
                            ], width=component.column_span)
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error updating component {component.component_id}: {e}")
            
            # Arrange in rows
            rows = []
            current_row = []
            current_width = 0
            
            for chart in charts:
                chart_width = chart.width if hasattr(chart, 'width') else 6
                if current_width + chart_width > 12:
                    if current_row:
                        rows.append(dbc.Row(current_row, className="mb-4"))
                    current_row = [chart]
                    current_width = chart_width
                else:
                    current_row.append(chart)
                    current_width += chart_width
            
            if current_row:
                rows.append(dbc.Row(current_row, className="mb-4"))
            
            return rows, f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
    
    def get_dashboard_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the dashboard itself."""
        return {
            'dashboard_load_time_seconds': self.performance_metrics['dashboard_load_time'],
            'chart_render_times': dict(self.performance_metrics['chart_render_time']),
            'data_update_time_seconds': self.performance_metrics['data_update_time'],
            'total_components': len(self.components),
            'active_data_sources': len([d for d in [self.benchmark_data, self.api_metrics_data, self.correlation_data] if d]),
            'memory_usage_mb': self.performance_metrics['memory_usage_mb']
        }


# Convenience functions
def create_default_dashboard(dashboard_title: str = "Quality Performance Dashboard") -> PerformanceDashboard:
    """Create dashboard with default configuration."""
    config = DashboardConfiguration(dashboard_title=dashboard_title)
    return PerformanceDashboard(config=config)


async def generate_performance_dashboard(
    benchmark_data: Optional[List[Any]] = None,
    api_metrics_data: Optional[List[Any]] = None,
    correlation_data: Optional[List[Any]] = None,
    output_filename: str = "performance_dashboard.html",
    dashboard_title: str = "Quality Performance Dashboard"
) -> Path:
    """
    Convenience function to generate performance dashboard.
    
    Args:
        benchmark_data: Quality validation benchmark data
        api_metrics_data: API metrics data
        correlation_data: Correlation analysis data
        output_filename: Output HTML filename
        dashboard_title: Dashboard title
        
    Returns:
        Path to generated dashboard file
    """
    dashboard = create_default_dashboard(dashboard_title)
    
    await dashboard.load_data(
        benchmark_data=benchmark_data,
        api_metrics_data=api_metrics_data,
        correlation_data=correlation_data
    )
    
    return await dashboard.generate_static_dashboard(output_filename)


# Make main classes available at module level
__all__ = [
    'PerformanceDashboard',
    'DashboardConfiguration',
    'DashboardComponent',
    'DashboardTheme',
    'ChartType',
    'UpdateFrequency',
    'create_default_dashboard',
    'generate_performance_dashboard'
]