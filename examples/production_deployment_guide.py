#!/usr/bin/env python3
"""
Production Deployment Guide: Real-world configuration examples for feature flags

This module provides comprehensive production deployment examples showing how to
configure and deploy the LightRAG feature flag system in various environments:

1. **Development Environment**
   - Local testing configuration
   - Debug settings and monitoring
   - Rapid iteration support

2. **Staging Environment**
   - Pre-production validation
   - Integration testing setup
   - Performance benchmarking

3. **Production Environment**
   - High-availability configuration
   - Security best practices
   - Monitoring and alerting
   - Disaster recovery

4. **Enterprise Deployment**
   - Multi-region setup
   - Load balancing considerations
   - Compliance and audit trails
   - Advanced security

5. **Configuration Management**
   - Environment variable management
   - Secret management
   - Configuration validation
   - Hot reloading capabilities

Author: Claude Code (Anthropic)
Created: 2025-08-08
Version: 1.0.0
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import secrets

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag_integration import (
    LightRAGConfig,
    FeatureFlagManager,
    RolloutManager,
    RolloutStrategy,
    RolloutConfiguration
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Environment(NamedTuple):
    """Represents a deployment environment."""
    name: str
    description: str
    domain: str
    replicas: int
    monitoring_level: str
    security_level: str


@dataclass
class DeploymentConfig:
    """Complete deployment configuration for an environment."""
    environment: Environment
    feature_flags: Dict[str, Any]
    rollout_strategy: Dict[str, Any]
    monitoring: Dict[str, Any]
    security: Dict[str, Any]
    performance: Dict[str, Any]
    alerting: Dict[str, Any]
    
    def to_env_vars(self) -> Dict[str, str]:
        """Convert configuration to environment variables."""
        env_vars = {}
        
        # Feature flag settings
        for key, value in self.feature_flags.items():
            env_key = f"LIGHTRAG_{key.upper()}"
            if isinstance(value, bool):
                env_vars[env_key] = str(value).lower()
            else:
                env_vars[env_key] = str(value)
        
        # Monitoring settings
        for key, value in self.monitoring.items():
            env_key = f"LIGHTRAG_MONITORING_{key.upper()}"
            env_vars[env_key] = str(value)
        
        # Security settings
        for key, value in self.security.items():
            env_key = f"LIGHTRAG_SECURITY_{key.upper()}"
            env_vars[env_key] = str(value)
        
        return env_vars
    
    def save_to_file(self, filepath: Path) -> None:
        """Save configuration to file."""
        config_data = {
            'environment': self.environment._asdict(),
            'feature_flags': self.feature_flags,
            'rollout_strategy': self.rollout_strategy,
            'monitoring': self.monitoring,
            'security': self.security,
            'performance': self.performance,
            'alerting': self.alerting,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)


class ProductionConfigGenerator:
    """Generates production-ready configurations for different environments."""
    
    def __init__(self):
        self.environments = {
            'development': Environment(
                name='development',
                description='Local development environment',
                domain='localhost:8000',
                replicas=1,
                monitoring_level='debug',
                security_level='basic'
            ),
            'staging': Environment(
                name='staging',
                description='Staging environment for pre-production testing',
                domain='staging.clinicalmetabolomics.org',
                replicas=2,
                monitoring_level='detailed',
                security_level='enhanced'
            ),
            'production': Environment(
                name='production',
                description='Production environment serving live traffic',
                domain='clinicalmetabolomics.org',
                replicas=5,
                monitoring_level='comprehensive',
                security_level='maximum'
            ),
            'enterprise': Environment(
                name='enterprise',
                description='Enterprise deployment with advanced features',
                domain='enterprise.clinicalmetabolomics.org',
                replicas=10,
                monitoring_level='enterprise',
                security_level='enterprise'
            )
        }
    
    def generate_development_config(self) -> DeploymentConfig:
        """Generate development environment configuration."""
        env = self.environments['development']
        
        return DeploymentConfig(
            environment=env,
            feature_flags={
                'integration_enabled': True,
                'rollout_percentage': 100.0,  # Full rollout for dev
                'enable_ab_testing': False,   # Disabled for dev
                'enable_circuit_breaker': True,
                'circuit_breaker_failure_threshold': 10,  # Lenient for dev
                'circuit_breaker_recovery_timeout': 60,
                'enable_quality_metrics': True,
                'min_quality_threshold': 0.5,  # Lower threshold for dev
                'enable_conditional_routing': True,
                'enable_performance_monitoring': True,
                'user_hash_salt': 'dev_salt_not_secure',
                'force_user_cohort': None  # Allow normal routing
            },
            rollout_strategy={
                'strategy': 'manual',
                'auto_rollback': True,
                'quality_gates': False
            },
            monitoring={
                'log_level': 'DEBUG',
                'metrics_interval': 30,  # 30 seconds
                'enable_detailed_logs': True,
                'log_requests': True,
                'log_responses': True,
                'performance_profiling': True
            },
            security={
                'require_https': False,  # HTTP OK for dev
                'api_rate_limiting': False,
                'request_signing': False,
                'audit_logging': 'basic',
                'secret_rotation_days': 0  # No rotation in dev
            },
            performance={
                'max_concurrent_requests': 10,
                'request_timeout': 30,
                'cache_ttl': 300,  # 5 minutes
                'batch_size': 1
            },
            alerting={
                'enable_alerts': False,
                'email_notifications': False,
                'slack_webhook': None,
                'pagerduty_key': None
            }
        )
    
    def generate_staging_config(self) -> DeploymentConfig:
        """Generate staging environment configuration."""
        env = self.environments['staging']
        
        return DeploymentConfig(
            environment=env,
            feature_flags={
                'integration_enabled': True,
                'rollout_percentage': 50.0,  # 50% rollout for staging
                'enable_ab_testing': True,   # Test A/B functionality
                'enable_circuit_breaker': True,
                'circuit_breaker_failure_threshold': 5,
                'circuit_breaker_recovery_timeout': 120,
                'enable_quality_metrics': True,
                'min_quality_threshold': 0.7,
                'enable_conditional_routing': True,
                'enable_performance_monitoring': True,
                'user_hash_salt': self._generate_secure_salt(),
                'force_user_cohort': None
            },
            rollout_strategy={
                'strategy': 'canary',
                'canary_percentage': 10.0,
                'auto_rollback': True,
                'quality_gates': True,
                'approval_required': True
            },
            monitoring={
                'log_level': 'INFO',
                'metrics_interval': 60,
                'enable_detailed_logs': True,
                'log_requests': True,
                'log_responses': False,  # Reduced logging
                'performance_profiling': True
            },
            security={
                'require_https': True,
                'api_rate_limiting': True,
                'rate_limit_per_minute': 1000,
                'request_signing': True,
                'audit_logging': 'detailed',
                'secret_rotation_days': 30
            },
            performance={
                'max_concurrent_requests': 50,
                'request_timeout': 20,
                'cache_ttl': 600,  # 10 minutes
                'batch_size': 5
            },
            alerting={
                'enable_alerts': True,
                'email_notifications': True,
                'alert_email': 'staging-alerts@clinicalmetabolomics.org',
                'slack_webhook': '${SLACK_WEBHOOK_STAGING}',
                'alert_thresholds': {
                    'error_rate': 0.05,
                    'response_time': 5.0,
                    'quality_score': 0.6
                }
            }
        )
    
    def generate_production_config(self) -> DeploymentConfig:
        """Generate production environment configuration."""
        env = self.environments['production']
        
        return DeploymentConfig(
            environment=env,
            feature_flags={
                'integration_enabled': True,
                'rollout_percentage': 25.0,  # Conservative rollout
                'enable_ab_testing': True,
                'enable_circuit_breaker': True,
                'circuit_breaker_failure_threshold': 3,  # Strict threshold
                'circuit_breaker_recovery_timeout': 300,  # 5 minutes
                'enable_quality_metrics': True,
                'min_quality_threshold': 0.8,  # High quality bar
                'enable_conditional_routing': True,
                'enable_performance_monitoring': True,
                'user_hash_salt': self._generate_secure_salt(),
                'force_user_cohort': None
            },
            rollout_strategy={
                'strategy': 'linear',
                'start_percentage': 1.0,
                'increment': 5.0,
                'stage_duration_minutes': 60,
                'auto_rollback': True,
                'quality_gates': True,
                'approval_required': True
            },
            monitoring={
                'log_level': 'INFO',
                'metrics_interval': 60,
                'enable_detailed_logs': False,  # Performance optimization
                'log_requests': False,
                'log_responses': False,
                'performance_profiling': False,  # Disable in prod
                'structured_logging': True
            },
            security={
                'require_https': True,
                'api_rate_limiting': True,
                'rate_limit_per_minute': 10000,
                'request_signing': True,
                'audit_logging': 'comprehensive',
                'secret_rotation_days': 7,  # Weekly rotation
                'ip_whitelisting': True,
                'request_encryption': True
            },
            performance={
                'max_concurrent_requests': 1000,
                'request_timeout': 15,
                'cache_ttl': 1800,  # 30 minutes
                'batch_size': 10,
                'connection_pooling': True,
                'compression_enabled': True
            },
            alerting={
                'enable_alerts': True,
                'email_notifications': True,
                'alert_email': 'prod-alerts@clinicalmetabolomics.org',
                'slack_webhook': '${SLACK_WEBHOOK_PROD}',
                'pagerduty_key': '${PAGERDUTY_API_KEY}',
                'alert_thresholds': {
                    'error_rate': 0.01,  # 1%
                    'response_time': 3.0,
                    'quality_score': 0.75,
                    'circuit_breaker_trips': 1
                },
                'escalation_rules': {
                    'immediate': ['circuit_breaker_open', 'high_error_rate'],
                    'urgent': ['quality_degradation', 'slow_response_time'],
                    'normal': ['rollout_completed', 'threshold_breach']
                }
            }
        )
    
    def generate_enterprise_config(self) -> DeploymentConfig:
        """Generate enterprise deployment configuration."""
        env = self.environments['enterprise']
        
        return DeploymentConfig(
            environment=env,
            feature_flags={
                'integration_enabled': True,
                'rollout_percentage': 10.0,  # Very conservative
                'enable_ab_testing': True,
                'enable_circuit_breaker': True,
                'circuit_breaker_failure_threshold': 2,  # Very strict
                'circuit_breaker_recovery_timeout': 600,  # 10 minutes
                'enable_quality_metrics': True,
                'min_quality_threshold': 0.85,  # Very high quality
                'enable_conditional_routing': True,
                'enable_performance_monitoring': True,
                'user_hash_salt': self._generate_secure_salt(),
                'force_user_cohort': None,
                'enable_geographic_routing': True,
                'enable_customer_segmentation': True
            },
            rollout_strategy={
                'strategy': 'custom',
                'stages': [
                    {'name': 'Internal Users', 'percentage': 1.0, 'duration': 120},
                    {'name': 'Beta Customers', 'percentage': 5.0, 'duration': 240},
                    {'name': 'Premium Tier', 'percentage': 15.0, 'duration': 480},
                    {'name': 'General Release', 'percentage': 100.0, 'duration': 1440}
                ],
                'auto_rollback': True,
                'quality_gates': True,
                'approval_required': True,
                'compliance_checks': True
            },
            monitoring={
                'log_level': 'INFO',
                'metrics_interval': 30,
                'enable_detailed_logs': False,
                'log_requests': False,
                'log_responses': False,
                'performance_profiling': False,
                'structured_logging': True,
                'distributed_tracing': True,
                'custom_metrics': True,
                'business_metrics': True
            },
            security={
                'require_https': True,
                'api_rate_limiting': True,
                'rate_limit_per_minute': 50000,
                'request_signing': True,
                'audit_logging': 'enterprise',
                'secret_rotation_days': 1,  # Daily rotation
                'ip_whitelisting': True,
                'request_encryption': True,
                'multi_factor_auth': True,
                'sso_integration': True,
                'compliance_mode': 'HIPAA',
                'data_residency': True
            },
            performance={
                'max_concurrent_requests': 10000,
                'request_timeout': 10,
                'cache_ttl': 3600,  # 1 hour
                'batch_size': 50,
                'connection_pooling': True,
                'compression_enabled': True,
                'cdn_enabled': True,
                'auto_scaling': True,
                'load_balancing': 'geographic'
            },
            alerting={
                'enable_alerts': True,
                'email_notifications': True,
                'alert_email': 'enterprise-alerts@clinicalmetabolomics.org',
                'slack_webhook': '${SLACK_WEBHOOK_ENTERPRISE}',
                'pagerduty_key': '${PAGERDUTY_API_KEY}',
                'custom_webhooks': ['${CUSTOM_ALERT_WEBHOOK}'],
                'alert_thresholds': {
                    'error_rate': 0.005,  # 0.5%
                    'response_time': 2.0,
                    'quality_score': 0.8,
                    'circuit_breaker_trips': 1,
                    'compliance_violations': 0
                },
                'escalation_rules': {
                    'critical': ['compliance_violation', 'data_breach'],
                    'immediate': ['circuit_breaker_open', 'high_error_rate'],
                    'urgent': ['quality_degradation', 'slow_response_time'],
                    'normal': ['rollout_completed', 'threshold_breach']
                },
                'sla_monitoring': {
                    'availability': 99.99,
                    'response_time': 2.0,
                    'error_rate': 0.01
                }
            }
        )
    
    def _generate_secure_salt(self) -> str:
        """Generate a cryptographically secure salt."""
        return secrets.token_hex(32)


def create_docker_compose_config(config: DeploymentConfig) -> Dict[str, Any]:
    """Create Docker Compose configuration for deployment."""
    env_vars = config.to_env_vars()
    
    compose_config = {
        'version': '3.8',
        'services': {
            'cmo-chatbot': {
                'image': 'clinical-metabolomics-oracle:latest',
                'ports': ['8000:8000'],
                'environment': env_vars,
                'restart': 'unless-stopped',
                'healthcheck': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '40s'
                },
                'logging': {
                    'driver': 'json-file',
                    'options': {
                        'max-size': '10m',
                        'max-file': '3'
                    }
                }
            }
        }
    }
    
    # Add additional services based on environment
    if config.environment.name != 'development':
        # Add monitoring services
        compose_config['services']['prometheus'] = {
            'image': 'prom/prometheus:latest',
            'ports': ['9090:9090'],
            'volumes': [
                './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml'
            ],
            'command': [
                '--config.file=/etc/prometheus/prometheus.yml',
                '--storage.tsdb.path=/prometheus',
                '--web.console.libraries=/etc/prometheus/console_libraries',
                '--web.console.templates=/etc/prometheus/consoles',
                '--storage.tsdb.retention.time=200h',
                '--web.enable-lifecycle'
            ]
        }
        
        compose_config['services']['grafana'] = {
            'image': 'grafana/grafana:latest',
            'ports': ['3000:3000'],
            'environment': {
                'GF_SECURITY_ADMIN_PASSWORD': '${GRAFANA_PASSWORD}'
            },
            'volumes': [
                'grafana-storage:/var/lib/grafana'
            ]
        }
        
        # Add volumes
        compose_config['volumes'] = {
            'grafana-storage': {}
        }
    
    if config.environment.name == 'production':
        # Add load balancer
        compose_config['services']['nginx'] = {
            'image': 'nginx:alpine',
            'ports': ['80:80', '443:443'],
            'volumes': [
                './nginx/nginx.conf:/etc/nginx/nginx.conf',
                './nginx/ssl:/etc/nginx/ssl'
            ],
            'depends_on': ['cmo-chatbot']
        }
        
        # Scale the main service
        compose_config['services']['cmo-chatbot']['deploy'] = {
            'replicas': config.environment.replicas
        }
    
    return compose_config


def create_kubernetes_config(config: DeploymentConfig) -> Dict[str, Any]:
    """Create Kubernetes configuration for deployment."""
    env_vars = config.to_env_vars()
    
    # Convert environment variables to Kubernetes format
    k8s_env_vars = [
        {'name': key, 'value': value} for key, value in env_vars.items()
    ]
    
    k8s_config = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': f"cmo-chatbot-{config.environment.name}",
            'labels': {
                'app': 'cmo-chatbot',
                'environment': config.environment.name
            }
        },
        'spec': {
            'replicas': config.environment.replicas,
            'selector': {
                'matchLabels': {
                    'app': 'cmo-chatbot',
                    'environment': config.environment.name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': 'cmo-chatbot',
                        'environment': config.environment.name
                    }
                },
                'spec': {
                    'containers': [{
                        'name': 'cmo-chatbot',
                        'image': 'clinical-metabolomics-oracle:latest',
                        'ports': [{'containerPort': 8000}],
                        'env': k8s_env_vars,
                        'resources': {
                            'requests': {
                                'memory': '512Mi',
                                'cpu': '250m'
                            },
                            'limits': {
                                'memory': '2Gi',
                                'cpu': '1000m'
                            }
                        },
                        'livenessProbe': {
                            'httpGet': {
                                'path': '/health',
                                'port': 8000
                            },
                            'initialDelaySeconds': 30,
                            'periodSeconds': 10
                        },
                        'readinessProbe': {
                            'httpGet': {
                                'path': '/ready',
                                'port': 8000
                            },
                            'initialDelaySeconds': 5,
                            'periodSeconds': 5
                        }
                    }]
                }
            }
        }
    }
    
    return k8s_config


def create_monitoring_config(config: DeploymentConfig) -> Dict[str, Any]:
    """Create monitoring configuration (Prometheus/Grafana)."""
    return {
        'prometheus': {
            'global': {
                'scrape_interval': f"{config.monitoring['metrics_interval']}s",
                'evaluation_interval': f"{config.monitoring['metrics_interval']}s"
            },
            'scrape_configs': [
                {
                    'job_name': f"cmo-chatbot-{config.environment.name}",
                    'static_configs': [
                        {
                            'targets': ['cmo-chatbot:8000']
                        }
                    ],
                    'scrape_interval': f"{config.monitoring['metrics_interval']}s",
                    'metrics_path': '/metrics'
                }
            ]
        },
        'alerting_rules': {
            'groups': [
                {
                    'name': 'cmo-chatbot-alerts',
                    'rules': [
                        {
                            'alert': 'HighErrorRate',
                            'expr': f"rate(http_requests_total{{status=~'5..'}}[5m]) > {config.alerting['alert_thresholds']['error_rate']}",
                            'for': '5m',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'High error rate detected',
                                'description': f"Error rate is above {config.alerting['alert_thresholds']['error_rate']}"
                            }
                        },
                        {
                            'alert': 'SlowResponseTime',
                            'expr': f"histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > {config.alerting['alert_thresholds']['response_time']}",
                            'for': '5m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'Slow response time detected',
                                'description': f"95th percentile response time is above {config.alerting['alert_thresholds']['response_time']}s"
                            }
                        }
                    ]
                }
            ]
        }
    }


def validate_configuration(config: DeploymentConfig) -> List[str]:
    """Validate deployment configuration and return any issues."""
    issues = []
    
    # Validate feature flag settings
    if config.feature_flags.get('rollout_percentage', 0) > 100:
        issues.append("Rollout percentage cannot exceed 100%")
    
    if config.feature_flags.get('rollout_percentage', 0) < 0:
        issues.append("Rollout percentage cannot be negative")
    
    # Validate security settings
    if config.environment.name == 'production' and not config.security.get('require_https'):
        issues.append("HTTPS must be required in production")
    
    if config.environment.name == 'production' and not config.security.get('request_signing'):
        issues.append("Request signing should be enabled in production")
    
    # Validate monitoring settings
    if config.environment.name != 'development' and not config.alerting.get('enable_alerts'):
        issues.append("Alerts should be enabled in non-development environments")
    
    # Validate performance settings
    if config.performance.get('request_timeout', 0) <= 0:
        issues.append("Request timeout must be positive")
    
    return issues


def main():
    """Generate all production deployment configurations."""
    print("🚀 Production Deployment Configuration Generator")
    print("=" * 60)
    
    generator = ProductionConfigGenerator()
    output_dir = Path("deployment_configs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate configurations for all environments
    environments = ['development', 'staging', 'production', 'enterprise']
    
    for env_name in environments:
        print(f"\n🔧 Generating {env_name} configuration...")
        
        # Generate configuration
        if env_name == 'development':
            config = generator.generate_development_config()
        elif env_name == 'staging':
            config = generator.generate_staging_config()
        elif env_name == 'production':
            config = generator.generate_production_config()
        elif env_name == 'enterprise':
            config = generator.generate_enterprise_config()
        
        # Validate configuration
        issues = validate_configuration(config)
        if issues:
            print(f"⚠️ Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"✅ Configuration validation passed")
        
        # Create environment directory
        env_dir = output_dir / env_name
        env_dir.mkdir(exist_ok=True)
        
        # Save main configuration
        config.save_to_file(env_dir / 'config.json')
        
        # Generate Docker Compose configuration
        docker_config = create_docker_compose_config(config)
        with open(env_dir / 'docker-compose.yml', 'w') as f:
            import yaml
            yaml.dump(docker_config, f, default_flow_style=False)
        
        # Generate Kubernetes configuration
        k8s_config = create_kubernetes_config(config)
        with open(env_dir / 'deployment.yaml', 'w') as f:
            import yaml
            yaml.dump(k8s_config, f, default_flow_style=False)
        
        # Generate monitoring configuration
        monitoring_config = create_monitoring_config(config)
        with open(env_dir / 'monitoring.json', 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        # Generate environment variables file
        env_vars = config.to_env_vars()
        with open(env_dir / '.env', 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\\n")
        
        print(f"📁 Configuration saved to: {env_dir}")
    
    # Generate deployment documentation
    create_deployment_documentation(output_dir)
    
    print(f"\n✅ All deployment configurations generated successfully!")
    print(f"📂 Output directory: {output_dir.absolute()}")
    
    return output_dir


def create_deployment_documentation(output_dir: Path):
    """Create comprehensive deployment documentation."""
    doc_content = """# Clinical Metabolomics Oracle - Deployment Guide

This directory contains production-ready deployment configurations for the LightRAG feature flag system.

## Environments

### Development
- **Purpose**: Local development and testing
- **Configuration**: Full rollout, debug logging, minimal security
- **Usage**: `docker-compose -f development/docker-compose.yml up`

### Staging
- **Purpose**: Pre-production validation and A/B testing
- **Configuration**: 50% rollout, enhanced monitoring, moderate security
- **Usage**: Apply Kubernetes manifests or use Docker Compose

### Production
- **Purpose**: Live production environment
- **Configuration**: Conservative rollout, comprehensive monitoring, maximum security
- **Usage**: Use Kubernetes deployment with proper secrets management

### Enterprise
- **Purpose**: Enterprise deployment with advanced features
- **Configuration**: Very conservative rollout, enterprise security, compliance features
- **Usage**: Multi-region Kubernetes deployment with custom configurations

## Quick Start

1. **Choose Environment**: Select the appropriate configuration directory
2. **Set Secrets**: Configure environment variables and secrets
3. **Deploy**: Use Docker Compose or Kubernetes manifests
4. **Monitor**: Set up monitoring and alerting

## Configuration Files

- `config.json`: Main feature flag configuration
- `docker-compose.yml`: Docker Compose deployment
- `deployment.yaml`: Kubernetes deployment manifest
- `monitoring.json`: Prometheus/Grafana configuration
- `.env`: Environment variables

## Security Considerations

- Always use HTTPS in production
- Rotate secrets regularly
- Enable audit logging
- Use strong authentication
- Implement rate limiting

## Monitoring and Alerting

Each environment includes:
- Prometheus metrics collection
- Grafana dashboards
- Alert rules and thresholds
- Escalation procedures

## Rollout Strategies

- **Development**: Immediate full rollout
- **Staging**: Canary deployment with approval gates
- **Production**: Linear rollout with quality gates
- **Enterprise**: Multi-stage rollout with compliance checks

## Support

For deployment support or questions, contact the development team.
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(doc_content)


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("⚠️ PyYAML not installed. Installing...")
        os.system("pip install PyYAML")
        import yaml
    
    output_dir = main()
    
    print("\n📚 Deployment Guide Contents:")
    for env_dir in output_dir.iterdir():
        if env_dir.is_dir():
            print(f"\n📁 {env_dir.name}/")
            for file in sorted(env_dir.iterdir()):
                if file.is_file():
                    print(f"  📄 {file.name}")