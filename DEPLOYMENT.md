# 🚀 Deployment Guide - Advanced Trading Platform

This guide covers all deployment scenarios from local development to production-scale cloud deployment.

## 📋 **Prerequisites**

### **System Requirements**

- **CPU**: 8+ cores (16+ recommended for production)
- **RAM**: 16GB minimum (32GB+ for production)
- **Storage**: 100GB+ SSD (1TB+ for production)
- **Network**: 1Gbps+ (10Gbps+ for high-frequency trading)

### **Software Dependencies**

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Kubernetes**: 1.24+ (for production)
- **Python**: 3.11+
- **PostgreSQL**: 15+
- **Redis**: 7+
- **Node.js**: 18+ (for frontend)

## 🏠 **Local Development**

### **Quick Start**

```bash
# Clone repository
git clone https://github.com/yourusername/advanced-trading-platform.git
cd advanced-trading-platform

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### **Environment Configuration**

```bash
# .env file configuration
DATABASE_URL=postgresql://trading_user:trading_password@localhost:5432/trading_platform
REDIS_URL=redis://localhost:6379
RABBITMQ_URL=amqp://trading_user:trading_password@localhost:5672

# Exchange API Keys (for testing)
BINANCE_API_KEY=your_binance_testnet_key
BINANCE_SECRET_KEY=your_binance_testnet_secret
COINBASE_API_KEY=your_coinbase_sandbox_key
COINBASE_SECRET_KEY=your_coinbase_sandbox_secret

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256

# AI/ML Configuration
OPENAI_API_KEY=your_openai_key_for_sentiment_analysis
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_for_market_data
```

### **Service URLs (Development)**

- **API Gateway**: http://localhost:8000
- **Strategy Marketplace**: http://localhost:8007
- **Web Dashboard**: http://localhost:8080
- **Grafana Monitoring**: http://localhost:3000
- **RabbitMQ Management**: http://localhost:15672

## 🐳 **Docker Deployment**

### **Production Docker Compose**

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - '80:80'
      - '443:443'
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api-gateway
    restart: unless-stopped

  # API Gateway (Multiple instances)
  api-gateway:
    build: ./services/api-gateway
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    deploy:
      replicas: 3
    restart: unless-stopped

  # Strategy Marketplace (Multiple instances)
  strategy-marketplace:
    build: ./services/strategy-marketplace
    environment:
      - STRATEGY_MARKETPLACE_DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    deploy:
      replicas: 2
    restart: unless-stopped

  # Database with persistence
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped

  # Redis Cluster
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --cluster-enabled yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### **Build and Deploy**

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale api-gateway=5 --scale strategy-marketplace=3
```

## ☁️ **Cloud Deployment**

### **AWS EKS Deployment**

#### **1. Create EKS Cluster**

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster \
  --name trading-platform \
  --region us-west-2 \
  --nodegroup-name workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed
```

#### **2. Deploy Application**

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/api-gateway.yaml
kubectl apply -f k8s/strategy-marketplace.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -n trading-platform
kubectl get services -n trading-platform
```

#### **3. Configure Load Balancer**

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trading-platform-ingress
  namespace: trading-platform
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/ssl-redirect: '443'
spec:
  rules:
    - host: api.tradingplatform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-gateway
                port:
                  number: 8000
    - host: marketplace.tradingplatform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: strategy-marketplace
                port:
                  number: 8007
```

### **Google GKE Deployment**

#### **1. Create GKE Cluster**

```bash
# Set project and zone
gcloud config set project your-project-id
gcloud config set compute/zone us-central1-a

# Create cluster
gcloud container clusters create trading-platform \
  --num-nodes=3 \
  --machine-type=n1-standard-4 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade
```

#### **2. Deploy with Helm**

```bash
# Install Helm
curl https://get.helm.sh/helm-v3.10.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/

# Deploy application
helm install trading-platform ./helm-chart \
  --namespace trading-platform \
  --create-namespace \
  --set image.tag=latest \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=tradingplatform.com
```

### **Azure AKS Deployment**

#### **1. Create AKS Cluster**

```bash
# Create resource group
az group create --name trading-platform-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group trading-platform-rg \
  --name trading-platform \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys
```

#### **2. Configure and Deploy**

```bash
# Get credentials
az aks get-credentials --resource-group trading-platform-rg --name trading-platform

# Deploy application
kubectl apply -f k8s/
```

## 🔧 **Production Configuration**

### **Environment Variables**

```bash
# Production .env
NODE_ENV=production
ENVIRONMENT=production

# Database (Use managed services in production)
DATABASE_URL=postgresql://user:pass@prod-db-cluster:5432/trading_platform
REDIS_URL=redis://prod-redis-cluster:6379

# Security
JWT_SECRET_KEY=super-secure-production-key-256-bits
ENCRYPTION_KEY=another-super-secure-key-for-data-encryption

# Exchange APIs (Production)
BINANCE_API_KEY=prod_binance_key
BINANCE_SECRET_KEY=prod_binance_secret
COINBASE_API_KEY=prod_coinbase_key
COINBASE_SECRET_KEY=prod_coinbase_secret

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
DATADOG_API_KEY=your_datadog_key
NEW_RELIC_LICENSE_KEY=your_newrelic_key

# Scaling
MAX_WORKERS=10
REDIS_POOL_SIZE=20
DB_POOL_SIZE=20
```

### **Database Configuration**

```sql
-- Production PostgreSQL settings
-- postgresql.conf

# Connection Settings
max_connections = 200
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on

# Replication (for HA)
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
```

### **Redis Configuration**

```conf
# Production Redis settings
# redis.conf

# Memory
maxmemory 8gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# Security
requirepass your-super-secure-redis-password
rename-command FLUSHDB ""
rename-command FLUSHALL ""

# Networking
bind 0.0.0.0
port 6379
tcp-keepalive 300
```

## 📊 **Monitoring Setup**

### **Prometheus Configuration**

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - 'rules/*.yml'

scrape_configs:
  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'strategy-marketplace'
    static_configs:
      - targets: ['strategy-marketplace:8007']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### **Grafana Dashboards**

```json
{
  "dashboard": {
    "title": "Trading Platform Overview",
    "panels": [
      {
        "title": "Active Users",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(active_users_total)"
          }
        ]
      },
      {
        "title": "Trading Volume",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(trading_volume_total[5m]))"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))"
          }
        ]
      }
    ]
  }
}
```

## 🔒 **Security Hardening**

### **SSL/TLS Configuration**

```nginx
# nginx/nginx.conf
server {
    listen 443 ssl http2;
    server_name tradingplatform.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    location / {
        proxy_pass http://api-gateway:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### **Firewall Rules**

```bash
# UFW firewall configuration
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow from 10.0.0.0/8 to any port 5432  # Database access
ufw allow from 10.0.0.0/8 to any port 6379  # Redis access
ufw enable
```

## 📈 **Scaling Strategies**

### **Horizontal Scaling**

```bash
# Scale API Gateway
kubectl scale deployment api-gateway --replicas=10

# Scale Strategy Marketplace
kubectl scale deployment strategy-marketplace --replicas=5

# Auto-scaling based on CPU
kubectl autoscale deployment api-gateway --cpu-percent=70 --min=3 --max=20
```

### **Database Scaling**

```bash
# PostgreSQL Read Replicas
# Master-slave replication setup
pg_basebackup -h master-db -D /var/lib/postgresql/replica -U replication -v -P -W

# Redis Cluster
redis-cli --cluster create \
  redis-1:6379 redis-2:6379 redis-3:6379 \
  redis-4:6379 redis-5:6379 redis-6:6379 \
  --cluster-replicas 1
```

## 🔄 **CI/CD Pipeline**

### **GitHub Actions**

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          docker-compose -f docker-compose.test.yml up --abort-on-container-exit
          docker-compose -f docker-compose.test.yml down

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push images
        run: |
          docker build -t trading-platform/api-gateway:${{ github.sha }} services/api-gateway
          docker push trading-platform/api-gateway:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/api-gateway api-gateway=trading-platform/api-gateway:${{ github.sha }}
          kubectl rollout status deployment/api-gateway
```

## 🚨 **Disaster Recovery**

### **Backup Strategy**

```bash
# Database backups
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | gzip > /backups/db_backup_$DATE.sql.gz

# Upload to S3
aws s3 cp /backups/db_backup_$DATE.sql.gz s3://trading-platform-backups/

# Cleanup old backups (keep 30 days)
find /backups -name "db_backup_*.sql.gz" -mtime +30 -delete
```

### **Recovery Procedures**

```bash
# Database recovery
gunzip -c /backups/db_backup_20241201_120000.sql.gz | psql -h $DB_HOST -U $DB_USER $DB_NAME

# Application rollback
kubectl rollout undo deployment/api-gateway
kubectl rollout undo deployment/strategy-marketplace
```

## 📞 **Support & Troubleshooting**

### **Common Issues**

#### **Database Connection Issues**

```bash
# Check database connectivity
kubectl exec -it postgres-pod -- psql -U trading_user -d trading_platform -c "SELECT 1;"

# Check connection pool
kubectl logs api-gateway-pod | grep "database"
```

#### **High Memory Usage**

```bash
# Check memory usage
kubectl top pods
kubectl describe pod high-memory-pod

# Scale up if needed
kubectl scale deployment api-gateway --replicas=5
```

#### **SSL Certificate Issues**

```bash
# Check certificate expiry
openssl x509 -in /etc/nginx/ssl/cert.pem -text -noout | grep "Not After"

# Renew Let's Encrypt certificate
certbot renew --nginx
```

### **Performance Tuning**

```bash
# Database performance
EXPLAIN ANALYZE SELECT * FROM strategies WHERE category = 'momentum';

# Redis performance
redis-cli --latency-history -i 1

# Application performance
kubectl exec -it api-gateway-pod -- python -m cProfile -s cumulative app.py
```

---

## 🎯 **Production Checklist**

### **Pre-Deployment**

- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance testing done
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] SSL certificates installed
- [ ] Environment variables set
- [ ] Database migrations applied

### **Post-Deployment**

- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Log aggregation working
- [ ] Backup verification
- [ ] Performance baseline established
- [ ] Documentation updated
- [ ] Team training completed

---

**🚀 Ready for production deployment!**

For additional support, contact: devops@tradingplatform.com
