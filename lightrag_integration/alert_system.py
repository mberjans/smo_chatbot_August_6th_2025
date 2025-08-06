"""
Alert Notification System for Clinical Metabolomics Oracle LightRAG Integration

This module provides comprehensive alert notification delivery mechanisms that integrate
with the existing BudgetManager and cost tracking infrastructure.

Classes:
    - AlertChannel: Enum for different alert delivery channels
    - AlertConfig: Configuration for alert notifications
    - EmailAlertConfig: Configuration for email notifications
    - WebhookAlertConfig: Configuration for webhook notifications
    - SlackAlertConfig: Configuration for Slack notifications
    - AlertNotificationSystem: Main alert delivery system
    - AlertEscalationManager: Progressive alert escalation system

The alert notification system supports:
    - Multiple delivery channels (email, webhooks, Slack, logging)
    - Progressive escalation based on alert severity and frequency
    - Template-based alert messages with customizable formatting
    - Retry mechanisms for failed deliveries
    - Rate limiting and deduplication
    - Integration with existing budget and metrics systems
    - Audit trail for notification compliance
"""

import asyncio
import json
import smtplib
import ssl
import time
import threading
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import jinja2

from .budget_manager import BudgetAlert, AlertLevel, BudgetManager
from .cost_persistence import CostPersistence


class AlertChannel(Enum):
    """Supported alert delivery channels."""
    
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    LOGGING = "logging"
    CONSOLE = "console"
    SMS = "sms"  # Future implementation
    DISCORD = "discord"  # Future implementation


@dataclass
class EmailAlertConfig:
    """Configuration for email alert notifications."""
    
    smtp_server: str
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    sender_email: str = ""
    recipient_emails: List[str] = field(default_factory=list)
    use_tls: bool = True
    use_ssl: bool = False
    timeout: float = 30.0
    
    # Email formatting
    subject_template: str = "Budget Alert: {alert_level} - {period_type}"
    html_template_path: Optional[str] = None
    text_template_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate email configuration."""
        if not self.smtp_server:
            raise ValueError("SMTP server is required for email alerts")
        if not self.recipient_emails:
            raise ValueError("At least one recipient email is required")
        if self.use_ssl and self.use_tls:
            raise ValueError("Cannot use both SSL and TLS simultaneously")


@dataclass 
class WebhookAlertConfig:
    """Configuration for webhook alert notifications."""
    
    url: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    verify_ssl: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Authentication
    auth_type: Optional[str] = None  # "basic", "bearer", "api_key"
    auth_credentials: Dict[str, str] = field(default_factory=dict)
    
    # Payload formatting
    payload_template: Optional[str] = None
    content_type: str = "application/json"
    
    def __post_init__(self):
        """Validate webhook configuration."""
        if not self.url:
            raise ValueError("URL is required for webhook alerts")
        if self.method not in ["POST", "PUT", "PATCH"]:
            raise ValueError("Webhook method must be POST, PUT, or PATCH")


@dataclass
class SlackAlertConfig:
    """Configuration for Slack alert notifications."""
    
    webhook_url: str
    channel: Optional[str] = None
    username: str = "Budget Alert Bot"
    icon_emoji: str = ":warning:"
    timeout: float = 30.0
    
    # Message formatting
    message_template: Optional[str] = None
    use_rich_formatting: bool = True
    mention_users: List[str] = field(default_factory=list)
    mention_channels: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate Slack configuration."""
        if not self.webhook_url:
            raise ValueError("Webhook URL is required for Slack alerts")


@dataclass
class AlertConfig:
    """Main configuration for alert notification system."""
    
    # Channel configurations
    email_config: Optional[EmailAlertConfig] = None
    webhook_config: Optional[WebhookAlertConfig] = None
    slack_config: Optional[SlackAlertConfig] = None
    
    # Global settings
    enabled_channels: Set[AlertChannel] = field(default_factory=lambda: {AlertChannel.LOGGING})
    rate_limit_window: float = 300.0  # 5 minutes
    max_alerts_per_window: int = 10
    dedupe_window: float = 60.0  # 1 minute deduplication
    
    # Escalation settings
    enable_escalation: bool = True
    escalation_levels: Dict[AlertLevel, List[AlertChannel]] = field(default_factory=lambda: {
        AlertLevel.WARNING: [AlertChannel.LOGGING],
        AlertLevel.CRITICAL: [AlertChannel.LOGGING, AlertChannel.EMAIL],
        AlertLevel.EXCEEDED: [AlertChannel.LOGGING, AlertChannel.EMAIL, AlertChannel.SLACK]
    })
    
    # Retry settings
    max_retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    retry_initial_delay: float = 1.0
    
    # Template settings
    template_dir: Optional[str] = None
    custom_templates: Dict[str, str] = field(default_factory=dict)
    
    def get_channels_for_alert(self, alert_level: AlertLevel) -> List[AlertChannel]:
        """Get the appropriate channels for an alert level."""
        if self.enable_escalation and alert_level in self.escalation_levels:
            channels = self.escalation_levels[alert_level]
            return [ch for ch in channels if ch in self.enabled_channels]
        else:
            return list(self.enabled_channels)


class AlertNotificationSystem:
    """
    Main alert notification delivery system.
    
    Handles delivery of budget alerts through multiple channels with retry logic,
    rate limiting, and deduplication capabilities.
    """
    
    def __init__(self,
                 config: AlertConfig,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize alert notification system.
        
        Args:
            config: Alert configuration
            logger: Logger instance for operations
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Rate limiting and deduplication
        self._alert_history: List[Dict[str, Any]] = []
        self._delivery_stats: Dict[str, Dict[str, Any]] = {}
        
        # Template engine
        self._template_env = self._setup_template_engine()
        
        # Async event loop for non-blocking operations
        self._loop = None
        self._loop_thread = None
        self._start_async_loop()
        
        self.logger.info("Alert notification system initialized")
    
    def _setup_template_engine(self) -> jinja2.Environment:
        """Set up Jinja2 template engine for alert formatting."""
        if self.config.template_dir:
            template_dir = Path(self.config.template_dir)
            if template_dir.exists():
                loader = jinja2.FileSystemLoader(str(template_dir))
            else:
                self.logger.warning(f"Template directory not found: {template_dir}")
                loader = jinja2.DictLoader(self.config.custom_templates)
        else:
            loader = jinja2.DictLoader(self.config.custom_templates)
        
        env = jinja2.Environment(loader=loader, autoescape=True)
        
        # Add custom filters
        env.filters['currency'] = lambda x: f"${x:.2f}" if x else "$0.00"
        env.filters['percentage'] = lambda x: f"{x:.1f}%" if x else "0.0%"
        env.filters['timestamp'] = lambda x: datetime.fromtimestamp(x, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        return env
    
    def _start_async_loop(self) -> None:
        """Start async event loop in separate thread for non-blocking operations."""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        
        # Wait for loop to be ready
        while self._loop is None:
            time.sleep(0.001)
    
    def send_alert(self, alert: BudgetAlert, force: bool = False) -> Dict[str, Any]:
        """
        Send an alert through configured channels.
        
        Args:
            alert: Budget alert to send
            force: Skip rate limiting and deduplication if True
            
        Returns:
            Dict containing delivery results for each channel
        """
        with self._lock:
            # Check rate limiting and deduplication
            if not force and not self._should_send_alert(alert):
                return {'skipped': True, 'reason': 'rate_limited_or_duplicate'}
            
            # Get appropriate channels for this alert level
            channels = self.config.get_channels_for_alert(alert.alert_level)
            
            if not channels:
                self.logger.warning(f"No channels configured for alert level {alert.alert_level}")
                return {'skipped': True, 'reason': 'no_channels_configured'}
            
            # Record alert attempt
            self._record_alert_attempt(alert)
            
            # Send through each channel
            results = {}
            for channel in channels:
                try:
                    result = self._send_to_channel(alert, channel)
                    results[channel.value] = result
                except Exception as e:
                    self.logger.error(f"Error sending alert to {channel.value}: {e}")
                    results[channel.value] = {
                        'success': False,
                        'error': str(e),
                        'timestamp': time.time()
                    }
            
            # Update delivery stats
            self._update_delivery_stats(results)
            
            return {
                'alert_id': alert.id if hasattr(alert, 'id') else None,
                'channels': results,
                'timestamp': time.time()
            }
    
    def _should_send_alert(self, alert: BudgetAlert) -> bool:
        """Check if alert should be sent based on rate limiting and deduplication."""
        now = time.time()
        
        # Clean old history entries
        self._alert_history = [
            entry for entry in self._alert_history
            if now - entry['timestamp'] < self.config.rate_limit_window
        ]
        
        # Check rate limiting
        if len(self._alert_history) >= self.config.max_alerts_per_window:
            self.logger.warning("Alert rate limit exceeded, skipping alert")
            return False
        
        # Check deduplication
        for entry in self._alert_history:
            if (now - entry['timestamp'] < self.config.dedupe_window and
                entry['alert_level'] == alert.alert_level.value and
                entry['period_type'] == alert.period_type and
                entry['period_key'] == alert.period_key):
                self.logger.debug("Duplicate alert detected, skipping")
                return False
        
        return True
    
    def _record_alert_attempt(self, alert: BudgetAlert) -> None:
        """Record alert attempt for rate limiting and deduplication."""
        self._alert_history.append({
            'timestamp': time.time(),
            'alert_level': alert.alert_level.value,
            'period_type': alert.period_type,
            'period_key': alert.period_key,
            'percentage_used': alert.percentage_used
        })
    
    def _send_to_channel(self, alert: BudgetAlert, channel: AlertChannel) -> Dict[str, Any]:
        """Send alert to specific channel."""
        start_time = time.time()
        
        try:
            if channel == AlertChannel.EMAIL and self.config.email_config:
                result = self._send_email_alert(alert)
            elif channel == AlertChannel.WEBHOOK and self.config.webhook_config:
                result = self._send_webhook_alert(alert)
            elif channel == AlertChannel.SLACK and self.config.slack_config:
                result = self._send_slack_alert(alert)
            elif channel == AlertChannel.LOGGING:
                result = self._send_logging_alert(alert)
            elif channel == AlertChannel.CONSOLE:
                result = self._send_console_alert(alert)
            else:
                raise ValueError(f"Unsupported or unconfigured channel: {channel}")
            
            result['delivery_time_ms'] = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'delivery_time_ms': (time.time() - start_time) * 1000,
                'timestamp': time.time()
            }
    
    def _send_email_alert(self, alert: BudgetAlert) -> Dict[str, Any]:
        """Send alert via email."""
        config = self.config.email_config
        if not config:
            raise ValueError("Email configuration not provided")
        
        # Format subject
        subject = config.subject_template.format(
            alert_level=alert.alert_level.value.upper(),
            period_type=alert.period_type.title(),
            percentage_used=alert.percentage_used,
            current_cost=alert.current_cost,
            budget_limit=alert.budget_limit
        )
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config.sender_email
        msg['To'] = ', '.join(config.recipient_emails)
        
        # Generate email content
        text_content = self._generate_text_alert(alert)
        html_content = self._generate_html_alert(alert)
        
        msg.attach(MIMEText(text_content, 'plain'))
        if html_content:
            msg.attach(MIMEText(html_content, 'html'))
        
        # Send email
        try:
            if config.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(config.smtp_server, config.smtp_port, 
                                      context=context, timeout=config.timeout) as server:
                    if config.username and config.password:
                        server.login(config.username, config.password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(config.smtp_server, config.smtp_port, 
                                  timeout=config.timeout) as server:
                    if config.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                    if config.username and config.password:
                        server.login(config.username, config.password)
                    server.send_message(msg)
            
            return {
                'success': True,
                'recipients': config.recipient_emails,
                'subject': subject,
                'timestamp': time.time()
            }
            
        except Exception as e:
            raise Exception(f"Failed to send email: {e}")
    
    def _send_webhook_alert(self, alert: BudgetAlert) -> Dict[str, Any]:
        """Send alert via webhook."""
        config = self.config.webhook_config
        if not config:
            raise ValueError("Webhook configuration not provided")
        
        # Prepare headers
        headers = config.headers.copy()
        if 'Content-Type' not in headers:
            headers['Content-Type'] = config.content_type
        
        # Add authentication headers
        if config.auth_type == "bearer" and "token" in config.auth_credentials:
            headers['Authorization'] = f"Bearer {config.auth_credentials['token']}"
        elif config.auth_type == "api_key" and "key" in config.auth_credentials:
            headers['X-API-Key'] = config.auth_credentials['key']
        
        # Prepare payload
        if config.payload_template:
            try:
                template = self._template_env.from_string(config.payload_template)
                payload_str = template.render(alert=alert)
                payload = json.loads(payload_str) if config.content_type == "application/json" else payload_str
            except Exception as e:
                raise Exception(f"Failed to render webhook payload template: {e}")
        else:
            payload = alert.to_dict()
        
        # Send webhook with retries
        last_exception = None
        for attempt in range(config.retry_attempts):
            try:
                if config.content_type == "application/json":
                    response = requests.request(
                        config.method,
                        config.url,
                        json=payload,
                        headers=headers,
                        timeout=config.timeout,
                        verify=config.verify_ssl,
                        auth=self._get_requests_auth(config)
                    )
                else:
                    response = requests.request(
                        config.method,
                        config.url,
                        data=payload,
                        headers=headers,
                        timeout=config.timeout,
                        verify=config.verify_ssl,
                        auth=self._get_requests_auth(config)
                    )
                
                response.raise_for_status()
                
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'attempt': attempt + 1,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                last_exception = e
                if attempt < config.retry_attempts - 1:
                    time.sleep(config.retry_delay * (2 ** attempt))
                    continue
                break
        
        raise Exception(f"Webhook failed after {config.retry_attempts} attempts: {last_exception}")
    
    def _get_requests_auth(self, config: WebhookAlertConfig) -> Optional[tuple]:
        """Get requests auth tuple for webhook."""
        if config.auth_type == "basic" and "username" in config.auth_credentials and "password" in config.auth_credentials:
            return (config.auth_credentials["username"], config.auth_credentials["password"])
        return None
    
    def _send_slack_alert(self, alert: BudgetAlert) -> Dict[str, Any]:
        """Send alert via Slack webhook."""
        config = self.config.slack_config
        if not config:
            raise ValueError("Slack configuration not provided")
        
        # Generate Slack message
        if config.message_template:
            try:
                template = self._template_env.from_string(config.message_template)
                text = template.render(alert=alert)
            except Exception as e:
                text = alert.message
                self.logger.warning(f"Failed to render Slack template: {e}")
        else:
            text = self._generate_slack_message(alert)
        
        # Prepare Slack payload
        payload = {
            'text': text,
            'username': config.username,
            'icon_emoji': config.icon_emoji
        }
        
        if config.channel:
            payload['channel'] = config.channel
        
        if config.use_rich_formatting:
            payload['attachments'] = [{
                'color': self._get_alert_color(alert.alert_level),
                'fields': [
                    {'title': 'Period', 'value': f"{alert.period_type.title()} ({alert.period_key})", 'short': True},
                    {'title': 'Current Cost', 'value': f"${alert.current_cost:.2f}", 'short': True},
                    {'title': 'Budget Limit', 'value': f"${alert.budget_limit:.2f}", 'short': True},
                    {'title': 'Usage', 'value': f"{alert.percentage_used:.1f}%", 'short': True}
                ],
                'timestamp': int(alert.timestamp)
            }]
        
        # Add mentions
        mentions = []
        for user in config.mention_users:
            mentions.append(f"<@{user}>")
        for channel in config.mention_channels:
            mentions.append(f"<!{channel}>")
        
        if mentions:
            payload['text'] = f"{' '.join(mentions)} {payload['text']}"
        
        # Send to Slack
        response = requests.post(
            config.webhook_url,
            json=payload,
            timeout=config.timeout
        )
        response.raise_for_status()
        
        return {
            'success': True,
            'status_code': response.status_code,
            'timestamp': time.time()
        }
    
    def _send_logging_alert(self, alert: BudgetAlert) -> Dict[str, Any]:
        """Send alert via logging system."""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
            AlertLevel.EXCEEDED: logging.CRITICAL
        }.get(alert.alert_level, logging.INFO)
        
        self.logger.log(log_level, f"BUDGET ALERT: {alert.message}")
        
        return {
            'success': True,
            'log_level': log_level,
            'timestamp': time.time()
        }
    
    def _send_console_alert(self, alert: BudgetAlert) -> Dict[str, Any]:
        """Send alert to console output."""
        alert_symbol = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EXCEEDED: "❌"
        }.get(alert.alert_level, "ℹ️")
        
        message = f"{alert_symbol} BUDGET ALERT: {alert.message}"
        print(message)
        
        return {
            'success': True,
            'message': message,
            'timestamp': time.time()
        }
    
    def _generate_text_alert(self, alert: BudgetAlert) -> str:
        """Generate plain text alert content."""
        template = """
Budget Alert Notification
=========================

Alert Level: {{ alert.alert_level.value.upper() }}
Period: {{ alert.period_type.title() }} ({{ alert.period_key }})
Current Cost: {{ alert.current_cost | currency }}
Budget Limit: {{ alert.budget_limit | currency }}
Usage: {{ alert.percentage_used | percentage }}
Threshold: {{ alert.threshold_percentage | percentage }}

Message: {{ alert.message }}

Timestamp: {{ alert.timestamp | timestamp }}

{% if alert.metadata %}
Additional Details:
{% for key, value in alert.metadata.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

This is an automated notification from the Clinical Metabolomics Oracle budget monitoring system.
        """.strip()
        
        jinja_template = self._template_env.from_string(template)
        return jinja_template.render(alert=alert)
    
    def _generate_html_alert(self, alert: BudgetAlert) -> str:
        """Generate HTML alert content."""
        template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .alert-header { background-color: {{ alert_color }}; color: white; padding: 15px; border-radius: 5px; }
        .alert-content { border: 1px solid #ddd; padding: 15px; margin-top: 10px; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
        .footer { margin-top: 20px; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="alert-header">
        <h2>🚨 Budget Alert: {{ alert.alert_level.value.upper() }}</h2>
    </div>
    <div class="alert-content">
        <p><strong>{{ alert.message }}</strong></p>
        
        <div class="metrics">
            <div class="metric">
                <strong>Period:</strong><br>
                {{ alert.period_type.title() }} ({{ alert.period_key }})
            </div>
            <div class="metric">
                <strong>Current Cost:</strong><br>
                {{ alert.current_cost | currency }}
            </div>
            <div class="metric">
                <strong>Budget Limit:</strong><br>
                {{ alert.budget_limit | currency }}
            </div>
            <div class="metric">
                <strong>Usage:</strong><br>
                {{ alert.percentage_used | percentage }}
            </div>
        </div>
        
        {% if alert.metadata %}
        <h3>Additional Details:</h3>
        <ul>
        {% for key, value in alert.metadata.items() %}
            <li><strong>{{ key }}:</strong> {{ value }}</li>
        {% endfor %}
        </ul>
        {% endif %}
    </div>
    <div class="footer">
        <p>This is an automated notification from the Clinical Metabolomics Oracle budget monitoring system.</p>
        <p>Generated at: {{ alert.timestamp | timestamp }}</p>
    </div>
</body>
</html>
        """.strip()
        
        alert_color = self._get_alert_color(alert.alert_level)
        jinja_template = self._template_env.from_string(template)
        return jinja_template.render(alert=alert, alert_color=alert_color)
    
    def _generate_slack_message(self, alert: BudgetAlert) -> str:
        """Generate Slack-formatted message."""
        emoji = {
            AlertLevel.INFO: ":information_source:",
            AlertLevel.WARNING: ":warning:",
            AlertLevel.CRITICAL: ":rotating_light:",
            AlertLevel.EXCEEDED: ":x:"
        }.get(alert.alert_level, ":information_source:")
        
        return f"{emoji} *Budget Alert*: {alert.message}"
    
    def _get_alert_color(self, alert_level: AlertLevel) -> str:
        """Get color code for alert level."""
        colors = {
            AlertLevel.INFO: "#36a64f",      # Green
            AlertLevel.WARNING: "#ff9500",   # Orange
            AlertLevel.CRITICAL: "#ff4444",  # Red
            AlertLevel.EXCEEDED: "#8b0000"   # Dark Red
        }
        return colors.get(alert_level, "#36a64f")
    
    def _update_delivery_stats(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Update delivery statistics."""
        for channel, result in results.items():
            if channel not in self._delivery_stats:
                self._delivery_stats[channel] = {
                    'total_attempts': 0,
                    'successful_deliveries': 0,
                    'failed_deliveries': 0,
                    'average_delivery_time_ms': 0.0,
                    'last_delivery': None
                }
            
            stats = self._delivery_stats[channel]
            stats['total_attempts'] += 1
            stats['last_delivery'] = result.get('timestamp', time.time())
            
            if result.get('success', False):
                stats['successful_deliveries'] += 1
                
                # Update average delivery time
                delivery_time = result.get('delivery_time_ms', 0)
                current_avg = stats['average_delivery_time_ms']
                total_successful = stats['successful_deliveries']
                stats['average_delivery_time_ms'] = (
                    (current_avg * (total_successful - 1) + delivery_time) / total_successful
                )
            else:
                stats['failed_deliveries'] += 1
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get delivery statistics for monitoring."""
        with self._lock:
            return {
                'alert_history_size': len(self._alert_history),
                'channels': dict(self._delivery_stats),
                'config_summary': {
                    'enabled_channels': [ch.value for ch in self.config.enabled_channels],
                    'rate_limit_window': self.config.rate_limit_window,
                    'max_alerts_per_window': self.config.max_alerts_per_window,
                    'escalation_enabled': self.config.enable_escalation
                },
                'timestamp': time.time()
            }
    
    def test_channels(self) -> Dict[str, Any]:
        """Test all configured channels with a test alert."""
        test_alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.INFO,
            period_type="test",
            period_key="test-period",
            current_cost=50.0,
            budget_limit=100.0,
            percentage_used=50.0,
            threshold_percentage=50.0,
            message="This is a test alert from the Budget Alert System",
            metadata={'test': True, 'source': 'alert_system_test'}
        )
        
        return self.send_alert(test_alert, force=True)
    
    def close(self) -> None:
        """Clean shutdown of alert notification system."""
        try:
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
            
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5.0)
            
            self.logger.info("Alert notification system shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during alert notification system shutdown: {e}")


class AlertEscalationManager:
    """
    Progressive alert escalation system.
    
    Manages escalation of alerts based on frequency, severity, and time-based rules.
    """
    
    def __init__(self,
                 notification_system: AlertNotificationSystem,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize alert escalation manager.
        
        Args:
            notification_system: Alert notification system for delivery
            logger: Logger instance for operations
        """
        self.notification_system = notification_system
        self.logger = logger or logging.getLogger(__name__)
        
        # Escalation tracking
        self._escalation_history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()
        
        # Escalation rules
        self.escalation_rules = {
            'frequency_threshold': 5,  # Escalate after 5 alerts in window
            'frequency_window': 3600,  # 1 hour window
            'time_based_escalation': {
                AlertLevel.WARNING: 1800,     # 30 minutes
                AlertLevel.CRITICAL: 900,     # 15 minutes
                AlertLevel.EXCEEDED: 300      # 5 minutes
            }
        }
        
        self.logger.info("Alert escalation manager initialized")
    
    def process_alert(self, alert: BudgetAlert) -> Dict[str, Any]:
        """
        Process alert and determine if escalation is needed.
        
        Args:
            alert: Budget alert to process
            
        Returns:
            Dict containing processing results and escalation decisions
        """
        with self._lock:
            alert_key = f"{alert.period_type}_{alert.period_key}"
            
            # Record alert in escalation history
            self._record_escalation_event(alert_key, alert)
            
            # Check escalation conditions
            escalation_needed = self._check_escalation_conditions(alert_key, alert)
            
            # Send alert (with potential escalation)
            delivery_result = self.notification_system.send_alert(alert)
            
            result = {
                'alert_processed': True,
                'escalation_needed': escalation_needed,
                'delivery_result': delivery_result,
                'timestamp': time.time()
            }
            
            if escalation_needed:
                escalation_result = self._perform_escalation(alert_key, alert)
                result['escalation_result'] = escalation_result
            
            return result
    
    def _record_escalation_event(self, alert_key: str, alert: BudgetAlert) -> None:
        """Record alert event in escalation history."""
        now = time.time()
        
        if alert_key not in self._escalation_history:
            self._escalation_history[alert_key] = []
        
        self._escalation_history[alert_key].append({
            'timestamp': now,
            'alert_level': alert.alert_level,
            'percentage_used': alert.percentage_used,
            'current_cost': alert.current_cost
        })
        
        # Clean old history entries
        window = self.escalation_rules['frequency_window']
        self._escalation_history[alert_key] = [
            event for event in self._escalation_history[alert_key]
            if now - event['timestamp'] < window
        ]
    
    def _check_escalation_conditions(self, alert_key: str, alert: BudgetAlert) -> bool:
        """Check if alert should be escalated."""
        history = self._escalation_history.get(alert_key, [])
        
        # Frequency-based escalation
        if len(history) >= self.escalation_rules['frequency_threshold']:
            self.logger.warning(f"Frequency-based escalation triggered for {alert_key}")
            return True
        
        # Time-based escalation (persistent high usage)
        time_threshold = self.escalation_rules['time_based_escalation'].get(alert.alert_level)
        if time_threshold:
            persistent_alerts = [
                event for event in history
                if event['alert_level'] == alert.alert_level
            ]
            
            if len(persistent_alerts) >= 2:  # At least 2 alerts of same level
                first_alert_time = persistent_alerts[0]['timestamp']
                if time.time() - first_alert_time >= time_threshold:
                    self.logger.warning(f"Time-based escalation triggered for {alert_key}")
                    return True
        
        return False
    
    def _perform_escalation(self, alert_key: str, alert: BudgetAlert) -> Dict[str, Any]:
        """Perform alert escalation."""
        # Create escalated alert
        escalated_alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.EXCEEDED,  # Escalate to highest level
            period_type=alert.period_type,
            period_key=alert.period_key,
            current_cost=alert.current_cost,
            budget_limit=alert.budget_limit,
            percentage_used=alert.percentage_used,
            threshold_percentage=100.0,  # Mark as escalated
            message=f"ESCALATED ALERT: {alert.message}",
            metadata={
                **(alert.metadata or {}),
                'escalated': True,
                'original_alert_level': alert.alert_level.value,
                'escalation_reason': 'repeated_alerts_or_persistent_condition'
            }
        )
        
        # Send escalated alert through all available channels
        return self.notification_system.send_alert(escalated_alert, force=True)
    
    def get_escalation_status(self) -> Dict[str, Any]:
        """Get current escalation status and statistics."""
        with self._lock:
            now = time.time()
            window = self.escalation_rules['frequency_window']
            
            status = {
                'active_escalation_keys': len(self._escalation_history),
                'escalation_rules': self.escalation_rules,
                'recent_activity': {},
                'timestamp': now
            }
            
            for key, history in self._escalation_history.items():
                recent_events = [
                    event for event in history
                    if now - event['timestamp'] < window
                ]
                
                if recent_events:
                    status['recent_activity'][key] = {
                        'event_count': len(recent_events),
                        'latest_alert_level': recent_events[-1]['alert_level'].value,
                        'latest_percentage': recent_events[-1]['percentage_used'],
                        'escalation_risk': len(recent_events) / self.escalation_rules['frequency_threshold']
                    }
            
            return status
    
    def clear_escalation_history(self, alert_key: Optional[str] = None) -> None:
        """Clear escalation history for specific key or all keys."""
        with self._lock:
            if alert_key:
                self._escalation_history.pop(alert_key, None)
                self.logger.info(f"Escalation history cleared for {alert_key}")
            else:
                self._escalation_history.clear()
                self.logger.info("All escalation history cleared")