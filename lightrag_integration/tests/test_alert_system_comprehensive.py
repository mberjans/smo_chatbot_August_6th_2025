#!/usr/bin/env python3
"""
Comprehensive test suite for Alert System and Notification Infrastructure.

This test suite provides complete coverage of the alert system components including:
- AlertChannel enumeration and configuration validation
- EmailAlertConfig, WebhookAlertConfig, and SlackAlertConfig setup
- AlertConfig main configuration and channel management
- AlertNotificationSystem core notification delivery
- AlertEscalationManager progressive escalation logic
- Template rendering and message formatting
- Retry mechanisms and error handling
- Rate limiting and deduplication
- Integration with budget management system
- Performance under concurrent alert scenarios

Author: Claude Code (Anthropic)
Created: August 6, 2025
"""

import pytest
import time
import json
import smtplib
import requests
import threading
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Test imports
from lightrag_integration.alert_system import (
    AlertNotificationSystem,
    AlertEscalationManager,
    AlertConfig,
    EmailAlertConfig,
    WebhookAlertConfig,
    SlackAlertConfig,
    AlertChannel
)
from lightrag_integration.budget_manager import (
    BudgetAlert,
    AlertLevel,
    BudgetManager
)


class TestAlertChannel:
    """Tests for AlertChannel enumeration."""
    
    def test_alert_channel_values(self):
        """Test all alert channel values are defined."""
        expected_channels = [
            'email',
            'webhook',
            'slack',
            'logging',
            'console',
            'sms',
            'discord'
        ]
        
        actual_channels = [channel.value for channel in AlertChannel]
        
        for expected_channel in expected_channels:
            assert expected_channel in actual_channels
    
    def test_alert_channel_categorization(self):
        """Test alert channel categorization by implementation status."""
        # Currently implemented channels
        implemented_channels = [
            AlertChannel.EMAIL,
            AlertChannel.WEBHOOK,
            AlertChannel.SLACK,
            AlertChannel.LOGGING,
            AlertChannel.CONSOLE
        ]
        
        # Future implementation channels
        future_channels = [
            AlertChannel.SMS,
            AlertChannel.DISCORD
        ]
        
        for channel in implemented_channels + future_channels:
            assert channel in AlertChannel


class TestEmailAlertConfig:
    """Comprehensive tests for EmailAlertConfig."""
    
    def test_email_config_basic_creation(self):
        """Test basic EmailAlertConfig creation."""
        config = EmailAlertConfig(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            recipient_emails=["admin@example.com", "alerts@company.com"]
        )
        
        assert config.smtp_server == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert len(config.recipient_emails) == 2
        assert "admin@example.com" in config.recipient_emails
        assert config.use_tls is True
        assert config.use_ssl is False
        assert config.timeout == 30.0
    
    def test_email_config_comprehensive_creation(self):
        """Test comprehensive EmailAlertConfig creation."""
        config = EmailAlertConfig(
            smtp_server="mail.company.com",
            smtp_port=465,
            username="alert_system@company.com",
            password="secure_password",
            sender_email="noreply@company.com",
            recipient_emails=["ops@company.com"],
            use_tls=False,
            use_ssl=True,
            timeout=60.0,
            subject_template="URGENT: {alert_level} Budget Alert - {period_type}",
            html_template_path="/templates/budget_alert.html",
            text_template_path="/templates/budget_alert.txt"
        )
        
        assert config.smtp_server == "mail.company.com"
        assert config.smtp_port == 465
        assert config.username == "alert_system@company.com"
        assert config.sender_email == "noreply@company.com"
        assert config.use_tls is False
        assert config.use_ssl is True
        assert config.timeout == 60.0
        assert config.subject_template == "URGENT: {alert_level} Budget Alert - {period_type}"
        assert config.html_template_path == "/templates/budget_alert.html"
    
    def test_email_config_validation_success(self):
        """Test successful EmailAlertConfig validation."""
        # Valid configuration
        config = EmailAlertConfig(
            smtp_server="smtp.example.com",
            smtp_port=587,
            recipient_emails=["test@example.com"]
        )
        
        # Should not raise any exceptions
        assert config.smtp_server is not None
        assert len(config.recipient_emails) > 0
    
    def test_email_config_validation_failures(self):
        """Test EmailAlertConfig validation failures."""
        # Missing SMTP server
        with pytest.raises(ValueError, match="SMTP server is required"):
            EmailAlertConfig(
                smtp_server="",
                smtp_port=587,
                recipient_emails=["test@example.com"]
            )
        
        # No recipient emails
        with pytest.raises(ValueError, match="At least one recipient email is required"):
            EmailAlertConfig(
                smtp_server="smtp.example.com",
                smtp_port=587,
                recipient_emails=[]
            )
        
        # Invalid SSL/TLS combination
        with pytest.raises(ValueError, match="Cannot use both SSL and TLS simultaneously"):
            EmailAlertConfig(
                smtp_server="smtp.example.com",
                smtp_port=587,
                recipient_emails=["test@example.com"],
                use_ssl=True,
                use_tls=True
            )


class TestWebhookAlertConfig:
    """Comprehensive tests for WebhookAlertConfig."""
    
    def test_webhook_config_basic_creation(self):
        """Test basic WebhookAlertConfig creation."""
        config = WebhookAlertConfig(
            url="https://hooks.example.com/budget-alerts"
        )
        
        assert config.url == "https://hooks.example.com/budget-alerts"
        assert config.method == "POST"
        assert config.timeout == 30.0
        assert config.verify_ssl is True
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.content_type == "application/json"
    
    def test_webhook_config_comprehensive_creation(self):
        """Test comprehensive WebhookAlertConfig creation."""
        headers = {
            "User-Agent": "Budget-Alert-System/1.0",
            "X-Custom-Header": "budget-monitoring"
        }
        
        auth_credentials = {
            "username": "alert_user",
            "password": "webhook_password"
        }
        
        config = WebhookAlertConfig(
            url="https://internal-api.company.com/alerts/budget",
            method="PUT",
            headers=headers,
            timeout=45.0,
            verify_ssl=False,
            retry_attempts=5,
            retry_delay=2.0,
            auth_type="basic",
            auth_credentials=auth_credentials,
            payload_template='{"alert": "{{alert.message}}", "level": "{{alert.alert_level.value}}"}',
            content_type="application/json"
        )
        
        assert config.url == "https://internal-api.company.com/alerts/budget"
        assert config.method == "PUT"
        assert config.headers == headers
        assert config.timeout == 45.0
        assert config.verify_ssl is False
        assert config.retry_attempts == 5
        assert config.retry_delay == 2.0
        assert config.auth_type == "basic"
        assert config.auth_credentials == auth_credentials
        assert config.payload_template is not None
    
    def test_webhook_config_validation_success(self):
        """Test successful WebhookAlertConfig validation."""
        # Valid configuration
        config = WebhookAlertConfig(
            url="https://api.example.com/webhook"
        )
        assert config.url is not None
        
        # Valid with different methods
        for method in ["POST", "PUT", "PATCH"]:
            config = WebhookAlertConfig(
                url="https://api.example.com/webhook",
                method=method
            )
            assert config.method == method
    
    def test_webhook_config_validation_failures(self):
        """Test WebhookAlertConfig validation failures."""
        # Missing URL
        with pytest.raises(ValueError, match="URL is required for webhook alerts"):
            WebhookAlertConfig(url="")
        
        # Invalid HTTP method
        with pytest.raises(ValueError, match="Webhook method must be POST, PUT, or PATCH"):
            WebhookAlertConfig(
                url="https://example.com/webhook",
                method="GET"
            )
        
        with pytest.raises(ValueError, match="Webhook method must be POST, PUT, or PATCH"):
            WebhookAlertConfig(
                url="https://example.com/webhook",
                method="DELETE"
            )


class TestSlackAlertConfig:
    """Comprehensive tests for SlackAlertConfig."""
    
    def test_slack_config_basic_creation(self):
        """Test basic SlackAlertConfig creation."""
        config = SlackAlertConfig(
            webhook_url="https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        )
        
        assert config.webhook_url.startswith("https://hooks.slack.com")
        assert config.username == "Budget Alert Bot"
        assert config.icon_emoji == ":warning:"
        assert config.timeout == 30.0
        assert config.use_rich_formatting is True
        assert len(config.mention_users) == 0
        assert len(config.mention_channels) == 0
    
    def test_slack_config_comprehensive_creation(self):
        """Test comprehensive SlackAlertConfig creation."""
        config = SlackAlertConfig(
            webhook_url="https://hooks.slack.com/services/TEAM/CHANNEL/TOKEN",
            channel="#budget-alerts",
            username="Financial Monitor",
            icon_emoji=":money_with_wings:",
            timeout=60.0,
            message_template="Alert: {{alert.message}} ({{alert.percentage_used}}%)",
            use_rich_formatting=True,
            mention_users=["admin", "finance_manager"],
            mention_channels=["here", "channel"]
        )
        
        assert config.webhook_url == "https://hooks.slack.com/services/TEAM/CHANNEL/TOKEN"
        assert config.channel == "#budget-alerts"
        assert config.username == "Financial Monitor"
        assert config.icon_emoji == ":money_with_wings:"
        assert config.timeout == 60.0
        assert config.message_template is not None
        assert config.mention_users == ["admin", "finance_manager"]
        assert config.mention_channels == ["here", "channel"]
    
    def test_slack_config_validation_success(self):
        """Test successful SlackAlertConfig validation."""
        config = SlackAlertConfig(
            webhook_url="https://hooks.slack.com/services/valid/webhook/url"
        )
        assert config.webhook_url is not None
    
    def test_slack_config_validation_failures(self):
        """Test SlackAlertConfig validation failures."""
        # Missing webhook URL
        with pytest.raises(ValueError, match="Webhook URL is required for Slack alerts"):
            SlackAlertConfig(webhook_url="")


class TestAlertConfig:
    """Comprehensive tests for AlertConfig."""
    
    def test_alert_config_default_creation(self):
        """Test AlertConfig creation with defaults."""
        config = AlertConfig()
        
        assert AlertChannel.LOGGING in config.enabled_channels
        assert len(config.enabled_channels) == 1
        assert config.rate_limit_window == 300.0  # 5 minutes
        assert config.max_alerts_per_window == 10
        assert config.dedupe_window == 60.0  # 1 minute
        assert config.enable_escalation is True
        assert config.max_retry_attempts == 3
        assert config.retry_backoff_factor == 2.0
    
    def test_alert_config_comprehensive_creation(self):
        """Test comprehensive AlertConfig creation."""
        email_config = EmailAlertConfig(
            smtp_server="smtp.company.com",
            smtp_port=587,
            recipient_emails=["ops@company.com"]
        )
        
        webhook_config = WebhookAlertConfig(
            url="https://api.company.com/alerts"
        )
        
        slack_config = SlackAlertConfig(
            webhook_url="https://hooks.slack.com/services/T/B/X"
        )
        
        escalation_levels = {
            AlertLevel.WARNING: [AlertChannel.LOGGING, AlertChannel.EMAIL],
            AlertLevel.CRITICAL: [AlertChannel.LOGGING, AlertChannel.EMAIL, AlertChannel.SLACK],
            AlertLevel.EXCEEDED: [AlertChannel.LOGGING, AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK]
        }
        
        config = AlertConfig(
            email_config=email_config,
            webhook_config=webhook_config,
            slack_config=slack_config,
            enabled_channels={AlertChannel.EMAIL, AlertChannel.WEBHOOK, AlertChannel.SLACK, AlertChannel.LOGGING},
            rate_limit_window=600.0,  # 10 minutes
            max_alerts_per_window=5,
            dedupe_window=120.0,  # 2 minutes
            enable_escalation=True,
            escalation_levels=escalation_levels,
            max_retry_attempts=5,
            retry_backoff_factor=1.5,
            retry_initial_delay=0.5,
            template_dir="/custom/templates",
            custom_templates={"test": "custom template"}
        )
        
        assert config.email_config == email_config
        assert config.webhook_config == webhook_config
        assert config.slack_config == slack_config
        assert len(config.enabled_channels) == 4
        assert config.rate_limit_window == 600.0
        assert config.max_alerts_per_window == 5
        assert config.dedupe_window == 120.0
        assert config.escalation_levels[AlertLevel.EXCEEDED] == [AlertChannel.LOGGING, AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK]
        assert config.template_dir == "/custom/templates"
        assert config.custom_templates["test"] == "custom template"
    
    def test_get_channels_for_alert_level(self):
        """Test getting appropriate channels for alert levels."""
        # Setup escalation configuration
        escalation_levels = {
            AlertLevel.WARNING: [AlertChannel.LOGGING],
            AlertLevel.CRITICAL: [AlertChannel.LOGGING, AlertChannel.EMAIL],
            AlertLevel.EXCEEDED: [AlertChannel.LOGGING, AlertChannel.EMAIL, AlertChannel.SLACK]
        }
        
        config = AlertConfig(
            enabled_channels={AlertChannel.LOGGING, AlertChannel.EMAIL, AlertChannel.SLACK},
            enable_escalation=True,
            escalation_levels=escalation_levels
        )
        
        # Test escalation levels
        warning_channels = config.get_channels_for_alert(AlertLevel.WARNING)
        assert warning_channels == [AlertChannel.LOGGING]
        
        critical_channels = config.get_channels_for_alert(AlertLevel.CRITICAL)
        assert set(critical_channels) == {AlertChannel.LOGGING, AlertChannel.EMAIL}
        
        exceeded_channels = config.get_channels_for_alert(AlertLevel.EXCEEDED)
        assert set(exceeded_channels) == {AlertChannel.LOGGING, AlertChannel.EMAIL, AlertChannel.SLACK}
        
        # Test with disabled channel
        config.enabled_channels = {AlertChannel.LOGGING, AlertChannel.EMAIL}  # Remove Slack
        exceeded_channels_filtered = config.get_channels_for_alert(AlertLevel.EXCEEDED)
        assert AlertChannel.SLACK not in exceeded_channels_filtered
        assert set(exceeded_channels_filtered) == {AlertChannel.LOGGING, AlertChannel.EMAIL}
    
    def test_get_channels_without_escalation(self):
        """Test getting channels when escalation is disabled."""
        config = AlertConfig(
            enabled_channels={AlertChannel.LOGGING, AlertChannel.EMAIL, AlertChannel.CONSOLE},
            enable_escalation=False
        )
        
        # Should return all enabled channels regardless of alert level
        for alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EXCEEDED]:
            channels = config.get_channels_for_alert(alert_level)
            assert set(channels) == {AlertChannel.LOGGING, AlertChannel.EMAIL, AlertChannel.CONSOLE}


class TestAlertNotificationSystem:
    """Comprehensive tests for AlertNotificationSystem."""
    
    @pytest.fixture
    def basic_alert_config(self):
        """Create basic alert configuration for testing."""
        return AlertConfig(
            enabled_channels={AlertChannel.LOGGING, AlertChannel.CONSOLE},
            rate_limit_window=300.0,
            max_alerts_per_window=5,
            dedupe_window=60.0
        )
    
    @pytest.fixture
    def comprehensive_alert_config(self):
        """Create comprehensive alert configuration for testing."""
        email_config = EmailAlertConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            recipient_emails=["test@example.com"]
        )
        
        webhook_config = WebhookAlertConfig(
            url="https://test-webhook.com/alerts"
        )
        
        return AlertConfig(
            email_config=email_config,
            webhook_config=webhook_config,
            enabled_channels={AlertChannel.LOGGING, AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.WEBHOOK},
            rate_limit_window=60.0,
            max_alerts_per_window=3
        )
    
    @pytest.fixture
    def test_budget_alert(self):
        """Create test budget alert."""
        return BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=75.0,
            budget_limit=100.0,
            percentage_used=75.0,
            threshold_percentage=75.0,
            message="Daily budget warning: 75% of budget used"
        )
    
    def test_alert_system_initialization(self, basic_alert_config):
        """Test AlertNotificationSystem initialization."""
        logger = Mock(spec=logging.Logger)
        alert_system = AlertNotificationSystem(basic_alert_config, logger)
        
        assert alert_system.config == basic_alert_config
        assert alert_system.logger == logger
        assert len(alert_system._alert_history) == 0
        assert len(alert_system._delivery_stats) == 0
        assert alert_system._template_env is not None
    
    def test_send_alert_basic_logging(self, basic_alert_config, test_budget_alert):
        """Test basic alert sending to logging channel."""
        alert_system = AlertNotificationSystem(basic_alert_config)
        
        result = alert_system.send_alert(test_budget_alert, force=True)
        
        assert 'channels' in result
        assert 'logging' in result['channels']
        assert result['channels']['logging']['success'] is True
        assert 'timestamp' in result
        assert 'alert_id' in result
    
    def test_send_alert_multiple_channels(self, comprehensive_alert_config, test_budget_alert):
        """Test alert sending to multiple channels."""
        # Mock external services
        with patch('smtplib.SMTP') as mock_smtp, \
             patch('requests.request') as mock_requests:
            
            # Configure mocks
            mock_smtp_instance = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests.return_value = mock_response
            
            alert_system = AlertNotificationSystem(comprehensive_alert_config)
            result = alert_system.send_alert(test_budget_alert, force=True)
            
            # Verify multiple channels were attempted
            assert 'channels' in result
            expected_channels = ['logging', 'console', 'email', 'webhook']
            
            for channel in expected_channels:
                assert channel in result['channels']
                # Some channels might fail due to mocking, but should attempt
    
    def test_rate_limiting_functionality(self, basic_alert_config, test_budget_alert):
        """Test alert rate limiting functionality."""
        alert_system = AlertNotificationSystem(basic_alert_config)
        
        # Send first alert (should succeed)
        result1 = alert_system.send_alert(test_budget_alert, force=True)
        assert result1['channels']['logging']['success'] is True
        
        # Send same alert immediately (should be rate limited)
        result2 = alert_system.send_alert(test_budget_alert)
        assert result2.get('skipped') is True
        assert result2.get('reason') == 'rate_limited_or_duplicate'
        
        # Force sending should bypass rate limiting
        result3 = alert_system.send_alert(test_budget_alert, force=True)
        assert result3['channels']['logging']['success'] is True
    
    def test_deduplication_functionality(self, basic_alert_config):
        """Test alert deduplication functionality."""
        alert_system = AlertNotificationSystem(basic_alert_config)
        
        # Create identical alerts
        alert1 = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=80.0,
            budget_limit=100.0,
            percentage_used=80.0,
            threshold_percentage=75.0,
            message="Budget warning"
        )
        
        alert2 = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=82.0,  # Slightly different cost
            budget_limit=100.0,
            percentage_used=82.0,
            threshold_percentage=75.0,
            message="Budget warning updated"
        )
        
        # Send first alert
        result1 = alert_system.send_alert(alert1, force=True)
        assert result1['channels']['logging']['success'] is True
        
        # Send similar alert (should be deduplicated)
        result2 = alert_system.send_alert(alert2)
        assert result2.get('skipped') is True
        
        # Create alert with different level (should not be deduplicated)
        alert3 = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.CRITICAL,  # Different level
            period_type="daily",
            period_key="2025-08-06",
            current_cost=95.0,
            budget_limit=100.0,
            percentage_used=95.0,
            threshold_percentage=90.0,
            message="Budget critical"
        )
        
        result3 = alert_system.send_alert(alert3, force=True)
        assert result3['channels']['logging']['success'] is True
    
    @patch('smtplib.SMTP')
    def test_email_alert_delivery(self, mock_smtp, comprehensive_alert_config, test_budget_alert):
        """Test email alert delivery functionality."""
        mock_smtp_instance = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
        
        alert_system = AlertNotificationSystem(comprehensive_alert_config)
        result = alert_system.send_alert(test_budget_alert, force=True)
        
        # Verify email channel result
        email_result = result['channels']['email']
        assert email_result['success'] is True
        assert 'recipients' in email_result
        assert 'subject' in email_result
        assert 'delivery_time_ms' in email_result
        
        # Verify SMTP calls
        mock_smtp.assert_called_once()
        mock_smtp_instance.send_message.assert_called_once()
    
    @patch('requests.request')
    def test_webhook_alert_delivery(self, mock_requests, comprehensive_alert_config, test_budget_alert):
        """Test webhook alert delivery functionality."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response
        
        alert_system = AlertNotificationSystem(comprehensive_alert_config)
        result = alert_system.send_alert(test_budget_alert, force=True)
        
        # Verify webhook channel result
        webhook_result = result['channels']['webhook']
        assert webhook_result['success'] is True
        assert webhook_result['status_code'] == 200
        assert 'attempt' in webhook_result
        assert 'delivery_time_ms' in webhook_result
        
        # Verify webhook request
        mock_requests.assert_called_once()
        call_args = mock_requests.call_args
        assert call_args[0][0] == 'POST'  # Method
        assert call_args[0][1] == 'https://test-webhook.com/alerts'  # URL
    
    @patch('requests.request')
    def test_webhook_retry_mechanism(self, mock_requests, comprehensive_alert_config, test_budget_alert):
        """Test webhook retry mechanism on failures."""
        # Configure mock to fail first attempts, succeed on last
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        
        mock_requests.side_effect = [
            mock_response_fail,  # First attempt fails
            mock_response_fail,  # Second attempt fails
            mock_response_success  # Third attempt succeeds
        ]
        
        alert_system = AlertNotificationSystem(comprehensive_alert_config)
        result = alert_system.send_alert(test_budget_alert, force=True)
        
        # Should succeed on third attempt
        webhook_result = result['channels']['webhook']
        assert webhook_result['success'] is True
        assert webhook_result['attempt'] == 3
        
        # Verify all attempts were made
        assert mock_requests.call_count == 3
    
    def test_slack_message_formatting(self, basic_alert_config, test_budget_alert):
        """Test Slack message formatting."""
        alert_system = AlertNotificationSystem(basic_alert_config)
        
        # Test Slack message generation
        slack_message = alert_system._generate_slack_message(test_budget_alert)
        
        assert isinstance(slack_message, str)
        assert len(slack_message) > 0
        assert test_budget_alert.message in slack_message
        
        # Should contain appropriate emoji
        emojis = [":information_source:", ":warning:", ":rotating_light:", ":x:"]
        assert any(emoji in slack_message for emoji in emojis)
    
    def test_template_rendering(self, basic_alert_config, test_budget_alert):
        """Test alert template rendering functionality."""
        alert_system = AlertNotificationSystem(basic_alert_config)
        
        # Test text alert generation
        text_alert = alert_system._generate_text_alert(test_budget_alert)
        
        assert isinstance(text_alert, str)
        assert test_budget_alert.message in text_alert
        assert str(test_budget_alert.percentage_used) in text_alert
        assert str(test_budget_alert.current_cost) in text_alert
        assert str(test_budget_alert.budget_limit) in text_alert
        assert test_budget_alert.period_type in text_alert
        assert test_budget_alert.period_key in text_alert
        
        # Test HTML alert generation
        html_alert = alert_system._generate_html_alert(test_budget_alert)
        
        assert isinstance(html_alert, str)
        assert "<html>" in html_alert
        assert "</html>" in html_alert
        assert test_budget_alert.message in html_alert
        assert "Budget Alert" in html_alert
    
    def test_delivery_statistics_tracking(self, basic_alert_config, test_budget_alert):
        """Test delivery statistics tracking."""
        alert_system = AlertNotificationSystem(basic_alert_config)
        
        # Send multiple alerts
        for i in range(3):
            alert_system.send_alert(test_budget_alert, force=True)
        
        stats = alert_system.get_delivery_stats()
        
        assert 'alert_history_size' in stats
        assert 'channels' in stats
        assert 'config_summary' in stats
        assert 'timestamp' in stats
        
        # Verify logging channel stats
        if 'logging' in stats['channels']:
            logging_stats = stats['channels']['logging']
            assert logging_stats['total_attempts'] >= 3
            assert logging_stats['successful_deliveries'] >= 3
            assert logging_stats['failed_deliveries'] >= 0
    
    def test_concurrent_alert_sending(self, basic_alert_config):
        """Test concurrent alert sending functionality."""
        alert_system = AlertNotificationSystem(basic_alert_config)
        
        def send_alert_worker(worker_id):
            alert = BudgetAlert(
                timestamp=time.time(),
                alert_level=AlertLevel.WARNING,
                period_type="daily",
                period_key=f"2025-08-06-{worker_id}",  # Unique to avoid deduplication
                current_cost=70.0 + worker_id,
                budget_limit=100.0,
                percentage_used=70.0 + worker_id,
                threshold_percentage=75.0,
                message=f"Concurrent test alert {worker_id}"
            )
            return alert_system.send_alert(alert, force=True)
        
        # Run concurrent alert sending
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_alert_worker, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # Verify all alerts were processed
        assert len(results) == 10
        
        for result in results:
            assert 'channels' in result
            assert result['channels']['logging']['success'] is True
    
    def test_error_handling_in_channels(self, comprehensive_alert_config, test_budget_alert):
        """Test error handling in different alert channels."""
        with patch('smtplib.SMTP') as mock_smtp, \
             patch('requests.request') as mock_requests:
            
            # Configure mocks to raise exceptions
            mock_smtp.side_effect = Exception("SMTP server unavailable")
            mock_requests.side_effect = Exception("Network timeout")
            
            alert_system = AlertNotificationSystem(comprehensive_alert_config)
            result = alert_system.send_alert(test_budget_alert, force=True)
            
            # Logging and console should succeed
            assert result['channels']['logging']['success'] is True
            assert result['channels']['console']['success'] is True
            
            # Email and webhook should fail gracefully
            assert result['channels']['email']['success'] is False
            assert result['channels']['webhook']['success'] is False
            assert 'error' in result['channels']['email']
            assert 'error' in result['channels']['webhook']


class TestAlertEscalationManager:
    """Comprehensive tests for AlertEscalationManager."""
    
    @pytest.fixture
    def alert_system(self):
        """Create alert system for escalation testing."""
        config = AlertConfig(
            enabled_channels={AlertChannel.LOGGING, AlertChannel.CONSOLE},
            rate_limit_window=60.0,
            max_alerts_per_window=10
        )
        return AlertNotificationSystem(config)
    
    @pytest.fixture
    def escalation_manager(self, alert_system):
        """Create AlertEscalationManager for testing."""
        logger = Mock(spec=logging.Logger)
        return AlertEscalationManager(alert_system, logger)
    
    def test_escalation_manager_initialization(self, escalation_manager):
        """Test AlertEscalationManager initialization."""
        assert escalation_manager.notification_system is not None
        assert hasattr(escalation_manager, 'escalation_rules')
        assert len(escalation_manager._escalation_history) == 0
        
        # Verify default escalation rules
        rules = escalation_manager.escalation_rules
        assert 'frequency_threshold' in rules
        assert 'frequency_window' in rules
        assert 'time_based_escalation' in rules
        assert rules['frequency_threshold'] == 5
        assert rules['frequency_window'] == 3600  # 1 hour
    
    def test_process_alert_basic(self, escalation_manager):
        """Test basic alert processing without escalation."""
        alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=60.0,
            budget_limit=100.0,
            percentage_used=60.0,
            threshold_percentage=50.0,
            message="Basic escalation test"
        )
        
        result = escalation_manager.process_alert(alert)
        
        assert result['alert_processed'] is True
        assert result['escalation_needed'] is False
        assert 'delivery_result' in result
        assert 'timestamp' in result
    
    def test_frequency_based_escalation(self, escalation_manager):
        """Test frequency-based escalation logic."""
        alert_key = "daily_2025-08-06"
        
        # Send multiple alerts to trigger frequency escalation
        for i in range(escalation_manager.escalation_rules['frequency_threshold']):
            alert = BudgetAlert(
                timestamp=time.time() + i * 60,  # 1 minute apart
                alert_level=AlertLevel.WARNING,
                period_type="daily",
                period_key="2025-08-06",
                current_cost=70.0 + i * 2,
                budget_limit=100.0,
                percentage_used=70.0 + i * 2,
                threshold_percentage=75.0,
                message=f"Frequency escalation test {i}"
            )
            
            result = escalation_manager.process_alert(alert)
            
            if i < escalation_manager.escalation_rules['frequency_threshold'] - 1:
                assert result['escalation_needed'] is False
            else:
                # Last alert should trigger escalation
                assert result['escalation_needed'] is True
                assert 'escalation_result' in result
    
    def test_time_based_escalation(self, escalation_manager):
        """Test time-based escalation logic."""
        base_time = time.time()
        alert_key = "monthly_2025-08"
        
        # Send first alert
        alert1 = BudgetAlert(
            timestamp=base_time,
            alert_level=AlertLevel.CRITICAL,
            period_type="monthly",
            period_key="2025-08",
            current_cost=2700.0,
            budget_limit=3000.0,
            percentage_used=90.0,
            threshold_percentage=90.0,
            message="Time escalation test 1"
        )
        
        result1 = escalation_manager.process_alert(alert1)
        assert result1['escalation_needed'] is False
        
        # Send second alert after time threshold (15 minutes for CRITICAL)
        alert2 = BudgetAlert(
            timestamp=base_time + 1000,  # 16+ minutes later
            alert_level=AlertLevel.CRITICAL,
            period_type="monthly",
            period_key="2025-08",
            current_cost=2750.0,
            budget_limit=3000.0,
            percentage_used=91.7,
            threshold_percentage=90.0,
            message="Time escalation test 2"
        )
        
        result2 = escalation_manager.process_alert(alert2)
        # Should trigger time-based escalation
        assert result2['escalation_needed'] is True
    
    def test_escalated_alert_properties(self, escalation_manager):
        """Test properties of escalated alerts."""
        # Trigger escalation
        alert_key = "daily_2025-08-06"
        base_alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=80.0,
            budget_limit=100.0,
            percentage_used=80.0,
            threshold_percentage=75.0,
            message="Original warning alert"
        )
        
        # Send enough alerts to trigger escalation
        for i in range(escalation_manager.escalation_rules['frequency_threshold']):
            escalation_manager.process_alert(base_alert)
        
        # Last result should contain escalation
        escalation_history = escalation_manager._escalation_history[alert_key]
        assert len(escalation_history) == escalation_manager.escalation_rules['frequency_threshold']
    
    def test_escalation_status_reporting(self, escalation_manager):
        """Test escalation status and statistics reporting."""
        # Add some alert history
        for i in range(3):
            alert = BudgetAlert(
                timestamp=time.time() + i * 300,
                alert_level=AlertLevel.WARNING,
                period_type="daily",
                period_key=f"2025-08-0{6+i}",
                current_cost=70.0,
                budget_limit=100.0,
                percentage_used=70.0,
                threshold_percentage=75.0,
                message=f"Status test {i}"
            )
            escalation_manager.process_alert(alert)
        
        status = escalation_manager.get_escalation_status()
        
        assert 'active_escalation_keys' in status
        assert 'escalation_rules' in status
        assert 'recent_activity' in status
        assert 'timestamp' in status
        
        assert status['active_escalation_keys'] >= 3
        assert status['escalation_rules'] == escalation_manager.escalation_rules
    
    def test_escalation_history_cleanup(self, escalation_manager):
        """Test escalation history cleanup functionality."""
        # Add test data
        alert_key = "test_cleanup"
        escalation_manager._escalation_history[alert_key] = [
            {'timestamp': time.time(), 'test': 'data1'},
            {'timestamp': time.time(), 'test': 'data2'}
        ]
        
        # Clear specific key
        escalation_manager.clear_escalation_history(alert_key)
        assert alert_key not in escalation_manager._escalation_history
        
        # Add more test data
        escalation_manager._escalation_history["key1"] = [{'test': 'data'}]
        escalation_manager._escalation_history["key2"] = [{'test': 'data'}]
        
        # Clear all history
        escalation_manager.clear_escalation_history()
        assert len(escalation_manager._escalation_history) == 0
    
    def test_concurrent_escalation_processing(self, escalation_manager):
        """Test concurrent escalation processing."""
        def process_alert_worker(worker_id):
            alert = BudgetAlert(
                timestamp=time.time(),
                alert_level=AlertLevel.WARNING,
                period_type="daily",
                period_key=f"2025-08-{6+worker_id}",
                current_cost=75.0,
                budget_limit=100.0,
                percentage_used=75.0,
                threshold_percentage=75.0,
                message=f"Concurrent escalation test {worker_id}"
            )
            return escalation_manager.process_alert(alert)
        
        # Run concurrent escalation processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_alert_worker, i) for i in range(6)]
            results = [future.result() for future in futures]
        
        # Verify all alerts were processed
        assert len(results) == 6
        
        for result in results:
            assert result['alert_processed'] is True
            assert 'delivery_result' in result


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])