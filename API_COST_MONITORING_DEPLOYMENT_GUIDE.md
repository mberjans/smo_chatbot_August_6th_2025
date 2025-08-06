# API Cost Monitoring System - Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
4. [Configuration Management](#configuration-management)
5. [Container Deployment](#container-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Cloud Deployment](#cloud-deployment)
8. [Production Hardening](#production-hardening)
9. [Monitoring and Observability](#monitoring-and-observability)
10. [Backup and Recovery](#backup-and-recovery)
11. [Scaling and Performance](#scaling-and-performance)
12. [Maintenance and Updates](#maintenance-and-updates)

---

## Overview

This guide provides comprehensive deployment instructions for the API Cost Monitoring System in production environments. It covers everything from basic installation to enterprise-grade deployments with high availability, monitoring, and security.

### Deployment Architecture Options

1. **Standalone Deployment**: Single-server installation for small teams
2. **Container Deployment**: Docker-based deployment for consistency and portability  
3. **Kubernetes Deployment**: Orchestrated deployment for scalability and reliability
4. **Cloud Deployment**: Cloud-native deployment with managed services
5. **Hybrid Deployment**: Combination of on-premises and cloud components

---

## Prerequisites

### System Requirements

#### Minimum Requirements (Development/Small Teams)
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+), macOS 11+, Windows 10+
- **Python**: 3.8+ (recommended: 3.11+)
- **Memory**: 2GB RAM available
- **Storage**: 5GB available disk space
- **Network**: Internet access for API calls and updates

#### Production Requirements
- **Operating System**: Linux (Ubuntu 22.04 LTS recommended)
- **Python**: 3.11+ with virtual environment
- **Memory**: 8GB+ RAM (16GB+ for high-throughput environments)
- **Storage**: 50GB+ SSD storage with backup capabilities
- **CPU**: 4+ cores (8+ for high-throughput environments)
- **Network**: Reliable internet with 100+ Mbps bandwidth

### Software Dependencies

```bash
# System packages (Ubuntu/Debian)
sudo apt update
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    sqlite3 \
    supervisor \
    nginx \
    certbot

# System packages (CentOS/RHEL)
sudo dnf update
sudo dnf install -y \
    python3.11 \
    python3.11-devel \
    python3-pip \
    git \
    curl \
    wget \
    sqlite \
    supervisor \
    nginx \
    certbot
```

### Network Requirements

- **Outbound HTTPS (443)**: OpenAI API access
- **Outbound SMTP (587/465)**: Email alerts
- **Inbound HTTP/HTTPS (80/443)**: Dashboard access (if web interface enabled)
- **Internal connectivity**: Database and logging systems

### Security Prerequisites

```bash
# Create dedicated system user
sudo useradd -r -s /bin/bash -d /opt/lightrag lightrag
sudo mkdir -p /opt/lightrag
sudo chown lightrag:lightrag /opt/lightrag

# Set up basic firewall
sudo ufw enable
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

---

## Installation Methods

### Method 1: Direct Installation

#### Step 1: System Setup

```bash
# Switch to lightrag user
sudo -u lightrag -i

# Create directory structure
mkdir -p /opt/lightrag/{app,data,logs,config,backups}
cd /opt/lightrag

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools
```

#### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install \
    sqlite3 \
    requests \
    jinja2 \
    psutil \
    cryptography \
    prometheus-client

# Install optional dependencies for enhanced features
pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    plotly

# Save requirements
pip freeze > requirements.txt
```

#### Step 3: Install System Components

```bash
# Clone or copy the lightrag_integration module
# (Assuming the module is available as a package)
pip install lightrag-integration

# Or if installing from source:
git clone https://github.com/your-org/lightrag-integration.git
cd lightrag-integration
pip install -e .
```

#### Step 4: Configuration

```bash
# Create environment configuration
cat > /opt/lightrag/config/.env << 'EOF'
# Core Configuration
OPENAI_API_KEY=your-openai-api-key-here
LIGHTRAG_WORKING_DIR=/opt/lightrag/data
LIGHTRAG_LOG_DIR=/opt/lightrag/logs

# Budget Configuration
LIGHTRAG_DAILY_BUDGET_LIMIT=100.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=2000.0
LIGHTRAG_COST_ALERT_THRESHOLD=75.0

# Database Configuration
LIGHTRAG_COST_DB_PATH=/opt/lightrag/data/cost_tracking.db

# Logging Configuration
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_MAX_BYTES=52428800
LIGHTRAG_LOG_BACKUP_COUNT=10

# Alert Configuration
ALERT_EMAIL_SMTP_SERVER=smtp.yourdomain.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_USERNAME=alerts@yourdomain.com
ALERT_EMAIL_PASSWORD=your-email-password
ALERT_EMAIL_RECIPIENTS=admin@yourdomain.com,research@yourdomain.com

# Slack Configuration (optional)
ALERT_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
ALERT_SLACK_CHANNEL=#budget-alerts
EOF

# Set secure permissions
chmod 600 /opt/lightrag/config/.env
```

#### Step 5: Service Configuration

```bash
# Create systemd service file
sudo tee /etc/systemd/system/lightrag-monitor.service << 'EOF'
[Unit]
Description=LightRAG Budget Monitoring System
After=network.target

[Service]
Type=simple
User=lightrag
Group=lightrag
WorkingDirectory=/opt/lightrag
Environment=PATH=/opt/lightrag/venv/bin
EnvironmentFile=/opt/lightrag/config/.env
ExecStart=/opt/lightrag/venv/bin/python -m lightrag_integration.services.monitor_service
Restart=always
RestartSec=10
StandardOutput=append:/opt/lightrag/logs/service.log
StandardError=append:/opt/lightrag/logs/service.log

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/lightrag/data /opt/lightrag/logs

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable lightrag-monitor
sudo systemctl start lightrag-monitor
```

### Method 2: Package Installation

```bash
# Create package repository (if available)
curl -fsSL https://pkg.yourdomain.com/gpg.key | sudo apt-key add -
echo "deb https://pkg.yourdomain.com/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/lightrag.list

# Install package
sudo apt update
sudo apt install lightrag-budget-monitor

# Configure
sudo lightrag-config setup
```

---

## Configuration Management

### Environment-Based Configuration

#### Development Environment

```bash
# development.env
LIGHTRAG_ENVIRONMENT=development
LIGHTRAG_DAILY_BUDGET_LIMIT=10.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=200.0
LIGHTRAG_LOG_LEVEL=DEBUG
LIGHTRAG_ENABLE_BUDGET_ALERTS=false
LIGHTRAG_COST_DB_PATH=/tmp/dev_costs.db
```

#### Staging Environment

```bash
# staging.env
LIGHTRAG_ENVIRONMENT=staging
LIGHTRAG_DAILY_BUDGET_LIMIT=50.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=1000.0
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_BUDGET_ALERTS=true
LIGHTRAG_COST_DB_PATH=/opt/lightrag/staging/cost_tracking.db
```

#### Production Environment

```bash
# production.env
LIGHTRAG_ENVIRONMENT=production
LIGHTRAG_DAILY_BUDGET_LIMIT=200.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=4000.0
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_BUDGET_ALERTS=true
LIGHTRAG_ENABLE_AUDIT_TRAIL=true
LIGHTRAG_COST_DB_PATH=/opt/lightrag/data/cost_tracking.db
```

### Configuration Management Tools

#### Using Ansible

```yaml
# playbook.yml
---
- name: Deploy LightRAG Budget Monitor
  hosts: lightrag_servers
  become: yes
  vars:
    lightrag_version: "1.0.0"
    lightrag_user: "lightrag"
    lightrag_home: "/opt/lightrag"
    
  tasks:
    - name: Create lightrag user
      user:
        name: "{{ lightrag_user }}"
        system: yes
        shell: /bin/bash
        home: "{{ lightrag_home }}"
        
    - name: Create directory structure
      file:
        path: "{{ lightrag_home }}/{{ item }}"
        state: directory
        owner: "{{ lightrag_user }}"
        group: "{{ lightrag_user }}"
        mode: '0755'
      loop:
        - app
        - data
        - logs
        - config
        - backups
        
    - name: Install system packages
      apt:
        name:
          - python3.11
          - python3.11-venv
          - python3.11-dev
          - sqlite3
          - supervisor
        state: present
        update_cache: yes
        
    - name: Create virtual environment
      become_user: "{{ lightrag_user }}"
      pip:
        name: 
          - lightrag-integration
        virtualenv: "{{ lightrag_home }}/venv"
        virtualenv_python: python3.11
        
    - name: Deploy configuration
      template:
        src: lightrag.env.j2
        dest: "{{ lightrag_home }}/config/.env"
        owner: "{{ lightrag_user }}"
        group: "{{ lightrag_user }}"
        mode: '0600'
      notify: restart lightrag
        
    - name: Deploy systemd service
      template:
        src: lightrag-monitor.service.j2
        dest: /etc/systemd/system/lightrag-monitor.service
      notify:
        - reload systemd
        - restart lightrag
        
  handlers:
    - name: reload systemd
      systemd:
        daemon_reload: yes
        
    - name: restart lightrag
      systemd:
        name: lightrag-monitor
        state: restarted
        enabled: yes
```

#### Using Terraform (Cloud Deployment)

```hcl
# main.tf
resource "aws_instance" "lightrag_monitor" {
  ami           = "ami-0c02fb55956c7d316" # Ubuntu 22.04 LTS
  instance_type = "t3.medium"
  
  vpc_security_group_ids = [aws_security_group.lightrag.id]
  subnet_id              = aws_subnet.public.id
  
  user_data = base64encode(templatefile("${path.module}/user-data.sh", {
    openai_api_key = var.openai_api_key
    daily_budget   = var.daily_budget_limit
    monthly_budget = var.monthly_budget_limit
  }))
  
  root_block_device {
    volume_type = "gp3"
    volume_size = 50
    encrypted   = true
  }
  
  tags = {
    Name        = "lightrag-budget-monitor"
    Environment = var.environment
    Project     = "clinical-metabolomics"
  }
}

resource "aws_security_group" "lightrag" {
  name_prefix = "lightrag-monitor"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

---

## Container Deployment

### Docker Setup

#### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -r -d /app -s /bin/bash lightrag
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=lightrag:lightrag . .

# Create data directories
RUN mkdir -p /app/{data,logs} && chown -R lightrag:lightrag /app

# Switch to application user
USER lightrag

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "lightrag_integration.services.monitor_service"]
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  lightrag-monitor:
    build: .
    container_name: lightrag-budget-monitor
    restart: unless-stopped
    
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LIGHTRAG_WORKING_DIR=/app/data
      - LIGHTRAG_LOG_DIR=/app/logs
      - LIGHTRAG_DAILY_BUDGET_LIMIT=${DAILY_BUDGET_LIMIT:-100.0}
      - LIGHTRAG_MONTHLY_BUDGET_LIMIT=${MONTHLY_BUDGET_LIMIT:-2000.0}
      - LIGHTRAG_LOG_LEVEL=${LOG_LEVEL:-INFO}
    
    volumes:
      - lightrag_data:/app/data
      - lightrag_logs:/app/logs
      - ./config:/app/config:ro
    
    ports:
      - "8000:8000"
    
    networks:
      - lightrag_network
    
    # Resource limits
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Database backup service
  backup:
    image: alpine:latest
    volumes:
      - lightrag_data:/data
      - ./backups:/backups
    command: |
      sh -c "
        while true; do
          echo 'Creating backup...'
          tar -czf /backups/lightrag-backup-$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
          find /backups -name '*.tar.gz' -mtime +30 -delete
          sleep 86400
        done
      "
    restart: unless-stopped
    networks:
      - lightrag_network

volumes:
  lightrag_data:
  lightrag_logs:

networks:
  lightrag_network:
    driver: bridge
```

#### Multi-Stage Production Dockerfile

```dockerfile
# Dockerfile.prod
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Create non-root user
RUN useradd -r -d /app -s /bin/bash -u 1001 lightrag

# Copy Python packages from builder
COPY --from=builder /root/.local /home/lightrag/.local

# Set up application directory
WORKDIR /app
COPY --chown=lightrag:lightrag . .

# Create data directories
RUN mkdir -p /app/{data,logs} && chown -R lightrag:lightrag /app

# Security: remove unnecessary files
RUN find /app -name "*.pyc" -delete \
    && find /app -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Switch to non-root user
USER lightrag

# Add local packages to PATH
ENV PATH=/home/lightrag/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use exec form for better signal handling
CMD ["python", "-m", "lightrag_integration.services.monitor_service"]
```

### Container Registry

#### Push to Registry

```bash
# Build and tag image
docker build -f Dockerfile.prod -t lightrag-budget-monitor:1.0.0 .
docker tag lightrag-budget-monitor:1.0.0 your-registry.com/lightrag-budget-monitor:1.0.0

# Push to registry
docker push your-registry.com/lightrag-budget-monitor:1.0.0
```

#### Private Registry Setup

```bash
# Set up private registry with authentication
docker run -d \
  --restart=always \
  --name registry \
  -v registry-data:/var/lib/registry \
  -p 5000:5000 \
  registry:2

# Configure registry authentication
mkdir -p /etc/docker/registry
htpasswd -Bbn admin your-password > /etc/docker/registry/htpasswd
```

---

## Kubernetes Deployment

### Basic Kubernetes Manifests

#### Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: lightrag-system
  labels:
    name: lightrag-system
---
```

#### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lightrag-config
  namespace: lightrag-system
data:
  LIGHTRAG_WORKING_DIR: "/app/data"
  LIGHTRAG_LOG_DIR: "/app/logs"
  LIGHTRAG_LOG_LEVEL: "INFO"
  LIGHTRAG_ENABLE_FILE_LOGGING: "true"
  LIGHTRAG_ENABLE_COST_TRACKING: "true"
  LIGHTRAG_ENABLE_BUDGET_ALERTS: "true"
  LIGHTRAG_DAILY_BUDGET_LIMIT: "200.0"
  LIGHTRAG_MONTHLY_BUDGET_LIMIT: "4000.0"
  LIGHTRAG_COST_ALERT_THRESHOLD: "75.0"
  ALERT_EMAIL_SMTP_SERVER: "smtp.yourdomain.com"
  ALERT_EMAIL_SMTP_PORT: "587"
  ALERT_SLACK_CHANNEL: "#budget-alerts"
---
```

#### Secret

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: lightrag-secrets
  namespace: lightrag-system
type: Opaque
stringData:
  OPENAI_API_KEY: "your-openai-api-key-here"
  ALERT_EMAIL_USERNAME: "alerts@yourdomain.com"
  ALERT_EMAIL_PASSWORD: "your-email-password"
  ALERT_SLACK_WEBHOOK_URL: "https://hooks.slack.com/services/..."
---
```

#### Persistent Volume Claim

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: lightrag-data-pvc
  namespace: lightrag-system
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp2
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: lightrag-logs-pvc
  namespace: lightrag-system
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp2
  resources:
    requests:
      storage: 20Gi
---
```

#### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightrag-budget-monitor
  namespace: lightrag-system
  labels:
    app: lightrag-budget-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: lightrag-budget-monitor
  template:
    metadata:
      labels:
        app: lightrag-budget-monitor
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: lightrag-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
      
      containers:
      - name: lightrag-monitor
        image: your-registry.com/lightrag-budget-monitor:1.0.0
        imagePullPolicy: Always
        
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        
        envFrom:
        - configMapRef:
            name: lightrag-config
        - secretRef:
            name: lightrag-secrets
        
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
      
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: lightrag-data-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: lightrag-logs-pvc
      - name: config-volume
        configMap:
          name: lightrag-config
      
      imagePullSecrets:
      - name: registry-secret
      
      restartPolicy: Always
      terminationGracePeriodSeconds: 30
---
```

#### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: lightrag-budget-monitor-service
  namespace: lightrag-system
  labels:
    app: lightrag-budget-monitor
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: lightrag-budget-monitor
---
```

#### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: lightrag-budget-monitor-ingress
  namespace: lightrag-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    # Rate limiting
    nginx.ingress.kubernetes.io/rate-limit-requests-per-second: "10"
    nginx.ingress.kubernetes.io/rate-limit-burst-multiplier: "2"
    # Authentication
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: lightrag-auth
    nginx.ingress.kubernetes.io/auth-realm: "Authentication Required"
spec:
  tls:
  - hosts:
    - budget-monitor.yourdomain.com
    secretName: lightrag-tls-secret
  rules:
  - host: budget-monitor.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: lightrag-budget-monitor-service
            port:
              number: 80
---
```

### Helm Chart

#### Chart.yaml

```yaml
# Chart.yaml
apiVersion: v2
name: lightrag-budget-monitor
description: A Helm chart for LightRAG Budget Monitoring System
version: 1.0.0
appVersion: "1.0.0"
keywords:
  - monitoring
  - budget
  - cost-tracking
  - ai
home: https://github.com/your-org/lightrag-integration
sources:
  - https://github.com/your-org/lightrag-integration
maintainers:
  - name: Your Team
    email: team@yourdomain.com
```

#### values.yaml

```yaml
# values.yaml
replicaCount: 1

image:
  repository: your-registry.com/lightrag-budget-monitor
  tag: "1.0.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: budget-monitor.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: lightrag-tls-secret
      hosts:
        - budget-monitor.yourdomain.com

persistence:
  data:
    enabled: true
    size: 50Gi
    storageClass: gp2
  logs:
    enabled: true
    size: 20Gi
    storageClass: gp2

config:
  daily_budget_limit: "200.0"
  monthly_budget_limit: "4000.0"
  log_level: "INFO"
  enable_alerts: true

secrets:
  openai_api_key: ""
  email_username: ""
  email_password: ""
  slack_webhook_url: ""

resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 3
  targetCPUUtilizationPercentage: 80

nodeSelector: {}
tolerations: []
affinity: {}
```

### Deploy with Helm

```bash
# Add custom Helm repository (if available)
helm repo add lightrag-charts https://charts.yourdomain.com/
helm repo update

# Install with Helm
helm install lightrag-monitor lightrag-charts/lightrag-budget-monitor \
  --namespace lightrag-system \
  --create-namespace \
  --values production-values.yaml

# Upgrade
helm upgrade lightrag-monitor lightrag-charts/lightrag-budget-monitor \
  --namespace lightrag-system \
  --values production-values.yaml

# Check status
helm status lightrag-monitor -n lightrag-system
```

---

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS

```json
// ecs-task-definition.json
{
  "family": "lightrag-budget-monitor",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/lightragTaskRole",
  "containerDefinitions": [
    {
      "name": "lightrag-monitor",
      "image": "your-account.dkr.ecr.region.amazonaws.com/lightrag-budget-monitor:1.0.0",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LIGHTRAG_WORKING_DIR",
          "value": "/app/data"
        },
        {
          "name": "LIGHTRAG_LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:lightrag/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/lightrag-budget-monitor",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "mountPoints": [
        {
          "sourceVolume": "lightrag-data",
          "containerPath": "/app/data"
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "lightrag-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "transitEncryption": "ENABLED",
        "authorizationConfig": {
          "accessPointId": "fsap-12345678"
        }
      }
    }
  ]
}
```

#### CloudFormation Template

```yaml
# cloudformation-template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'LightRAG Budget Monitoring System'

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]
  
  OpenAIAPIKey:
    Type: String
    NoEcho: true
    Description: OpenAI API Key

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-lightrag-vpc

  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true

  # Security Groups
  SecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for LightRAG Budget Monitor
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          CidrIp: 0.0.0.0/0

  # Secrets Manager
  OpenAISecret:
    Type: AWS::SecretsManager::Secret
    Properties:
      Name: !Sub ${Environment}/lightrag/openai-key
      Description: OpenAI API Key for LightRAG
      SecretString: !Ref OpenAIAPIKey

  # IAM Roles
  InstanceRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
      Policies:
        - PolicyName: SecretsManagerAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - secretsmanager:GetSecretValue
                Resource: !Ref OpenAISecret

  # EC2 Instance
  LightRAGInstance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c02fb55956c7d316  # Ubuntu 22.04 LTS
      InstanceType: t3.medium
      KeyName: !Ref KeyPairName
      SecurityGroupIds:
        - !Ref SecurityGroup
      SubnetId: !Ref PublicSubnet
      IamInstanceProfile: !Ref InstanceProfile
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          apt-get update
          apt-get install -y python3.11 python3.11-venv awscli
          
          # Create lightrag user and setup
          useradd -r -s /bin/bash -d /opt/lightrag lightrag
          mkdir -p /opt/lightrag
          chown lightrag:lightrag /opt/lightrag
          
          # Install application
          sudo -u lightrag bash -c "
            cd /opt/lightrag
            python3.11 -m venv venv
            source venv/bin/activate
            pip install lightrag-integration
          "
          
          # Get secrets from AWS Secrets Manager
          aws secretsmanager get-secret-value --secret-id ${OpenAISecret} --region ${AWS::Region} --query SecretString --output text > /tmp/openai-key
          
          # Create environment file
          cat > /opt/lightrag/.env << EOF
          OPENAI_API_KEY=$(cat /tmp/openai-key)
          LIGHTRAG_WORKING_DIR=/opt/lightrag/data
          LIGHTRAG_DAILY_BUDGET_LIMIT=200.0
          LIGHTRAG_MONTHLY_BUDGET_LIMIT=4000.0
          EOF
          
          # Clean up temporary file
          rm /tmp/openai-key
          chown lightrag:lightrag /opt/lightrag/.env
          chmod 600 /opt/lightrag/.env
          
          # Start service
          systemctl enable lightrag-monitor
          systemctl start lightrag-monitor

Outputs:
  InstanceId:
    Description: Instance ID of the LightRAG Budget Monitor
    Value: !Ref LightRAGInstance
  
  PublicIP:
    Description: Public IP address of the instance
    Value: !GetAtt LightRAGInstance.PublicIp
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to Google Container Registry
gcloud auth configure-docker
docker build -t gcr.io/your-project/lightrag-budget-monitor:1.0.0 .
docker push gcr.io/your-project/lightrag-budget-monitor:1.0.0

# Deploy to Cloud Run
gcloud run deploy lightrag-budget-monitor \
  --image gcr.io/your-project/lightrag-budget-monitor:1.0.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --concurrency 10 \
  --max-instances 3 \
  --set-env-vars LIGHTRAG_LOG_LEVEL=INFO \
  --set-secrets OPENAI_API_KEY=openai-key:latest
```

### Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name lightrag-rg --location eastus

# Create storage account for persistent data
az storage account create \
  --name lightragstorageacct \
  --resource-group lightrag-rg \
  --location eastus \
  --sku Standard_LRS

# Create file share
az storage share create \
  --name lightrag-data \
  --account-name lightragstorageacct

# Deploy container instance
az container create \
  --resource-group lightrag-rg \
  --name lightrag-budget-monitor \
  --image your-registry.azurecr.io/lightrag-budget-monitor:1.0.0 \
  --cpu 1 \
  --memory 2 \
  --ports 8000 \
  --environment-variables \
    LIGHTRAG_LOG_LEVEL=INFO \
    LIGHTRAG_WORKING_DIR=/app/data \
  --secure-environment-variables \
    OPENAI_API_KEY=your-openai-api-key \
  --azure-file-volume-account-name lightragstorageacct \
  --azure-file-volume-account-key $(az storage account keys list --resource-group lightrag-rg --account-name lightragstorageacct --query "[0].value" --output tsv) \
  --azure-file-volume-share-name lightrag-data \
  --azure-file-volume-mount-path /app/data
```

---

## Production Hardening

### Security Hardening

#### System Security

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Configure automatic security updates
sudo apt install -y unattended-upgrades
echo 'Unattended-Upgrade::Automatic-Reboot "false";' >> /etc/apt/apt.conf.d/50unattended-upgrades

# Configure firewall
sudo ufw --force enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow from 10.0.0.0/8 to any port 8000  # Internal monitoring

# Secure SSH
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# Install and configure fail2ban
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

#### Application Security

```bash
# Set secure file permissions
sudo chown -R lightrag:lightrag /opt/lightrag
sudo chmod 750 /opt/lightrag
sudo chmod 600 /opt/lightrag/config/.env
sudo chmod 640 /opt/lightrag/config/*.conf

# Enable log rotation
sudo tee /etc/logrotate.d/lightrag << 'EOF'
/opt/lightrag/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 lightrag lightrag
    postrotate
        systemctl reload lightrag-monitor
    endscript
}
EOF

# Set up AppArmor profile (Ubuntu/Debian)
sudo tee /etc/apparmor.d/lightrag-monitor << 'EOF'
#include <tunables/global>

/opt/lightrag/venv/bin/python {
  #include <abstractions/base>
  #include <abstractions/python>
  
  capability setuid,
  capability setgid,
  
  /opt/lightrag/** rwk,
  /tmp/ r,
  /tmp/** rw,
  /proc/sys/kernel/random/uuid r,
  
  # Network access
  network inet stream,
  network inet6 stream,
  
  # Deny dangerous operations
  deny /etc/shadow r,
  deny /etc/passwd w,
  deny /root/** rwx,
  deny /home/** rwx,
  deny mount,
  deny umount,
}
EOF

sudo apparmor_parser -r /etc/apparmor.d/lightrag-monitor
```

### SSL/TLS Configuration

#### Let's Encrypt Setup

```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d budget-monitor.yourdomain.com

# Test automatic renewal
sudo certbot renew --dry-run

# Set up automatic renewal cron job
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

#### Custom SSL Certificate

```bash
# Generate private key and CSR
sudo openssl genrsa -out /etc/ssl/private/lightrag.key 2048
sudo openssl req -new -key /etc/ssl/private/lightrag.key -out /tmp/lightrag.csr

# After receiving certificate from CA
sudo cp your-certificate.crt /etc/ssl/certs/lightrag.crt
sudo cp ca-bundle.crt /etc/ssl/certs/lightrag-ca.crt

# Set secure permissions
sudo chmod 600 /etc/ssl/private/lightrag.key
sudo chmod 644 /etc/ssl/certs/lightrag.crt
```

### Network Security

#### Nginx Configuration

```nginx
# /etc/nginx/sites-available/lightrag-monitor
server {
    listen 80;
    server_name budget-monitor.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name budget-monitor.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/lightrag.crt;
    ssl_certificate_key /etc/ssl/private/lightrag.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'";

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=lightrag:10m rate=10r/s;
    limit_req zone=lightrag burst=20 nodelay;

    # Authentication (optional)
    auth_basic "Restricted Area";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security
        proxy_hide_header X-Powered-By;
        proxy_hide_header Server;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Health check endpoint (no auth required)
    location /health {
        auth_basic off;
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }

    # Static files (if any)
    location /static/ {
        alias /opt/lightrag/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

---

## Monitoring and Observability

### Prometheus Integration

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "lightrag_rules.yml"

scrape_configs:
  - job_name: 'lightrag-budget-monitor'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 30s
    metrics_path: /metrics
    scrape_timeout: 10s
```

#### Alert Rules

```yaml
# lightrag_rules.yml
groups:
- name: lightrag.rules
  rules:
  - alert: LightRAGBudgetExceeded
    expr: lightrag_budget_usage_percent > 100
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "LightRAG budget exceeded"
      description: "Budget usage is {{ $value }}% which exceeds the limit"

  - alert: LightRAGHighBudgetUsage
    expr: lightrag_budget_usage_percent > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "LightRAG high budget usage"
      description: "Budget usage is {{ $value }}% of the limit"

  - alert: LightRAGHighErrorRate
    expr: rate(lightrag_api_errors_total[5m]) > 0.05
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "LightRAG high error rate"
      description: "Error rate is {{ $value }} errors per second"

  - alert: LightRAGServiceDown
    expr: up{job="lightrag-budget-monitor"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "LightRAG service is down"
      description: "LightRAG budget monitoring service is not responding"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "LightRAG Budget Monitoring",
    "tags": ["lightrag", "budget", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Budget Usage",
        "type": "stat",
        "targets": [
          {
            "expr": "lightrag_budget_usage_percent",
            "legendFormat": "Budget Usage %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 75},
                {"color": "red", "value": 90}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "API Calls Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(lightrag_api_calls_total[5m])",
            "legendFormat": "API Calls/sec"
          }
        ]
      },
      {
        "id": 3,
        "title": "Cost Over Time",
        "type": "graph",
        "targets": [
          {
            "expr": "increase(lightrag_total_cost[1h])",
            "legendFormat": "Hourly Cost"
          }
        ]
      }
    ],
    "time": {
      "from": "now-24h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### Log Management

#### ELK Stack Integration

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /opt/lightrag/logs/*.log
  fields:
    service: lightrag-budget-monitor
    environment: production
  multiline.pattern: '^\d{4}-\d{2}-\d{2}'
  multiline.negate: true
  multiline.match: after

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "lightrag-logs-%{+yyyy.MM.dd}"

setup.template.settings:
  index.number_of_shards: 1
  index.number_of_replicas: 1

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

#### Logstash Configuration

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "lightrag-budget-monitor" {
    grok {
      match => {
        "message" => "%{TIMESTAMP_ISO8601:timestamp} - %{DATA:logger_name} - %{LOGLEVEL:level} - %{GREEDYDATA:message}"
      }
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if "cost" in [message] {
      grok {
        match => {
          "message" => ".*cost.*%{NUMBER:cost:float}.*"
        }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "lightrag-%{+YYYY.MM.dd}"
  }
}
```

---

This deployment guide provides comprehensive instructions for deploying the API Cost Monitoring System across various environments and platforms. For ongoing maintenance and troubleshooting, refer to the [Troubleshooting Guide](./API_COST_MONITORING_TROUBLESHOOTING_GUIDE.md).

---

*This deployment guide is part of the Clinical Metabolomics Oracle API Cost Monitoring System documentation suite.*