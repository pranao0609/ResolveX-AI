# ResolveX — Deployment & Operations Guide

## 1. Deployment Architecture

ResolveX uses a clean, production-oriented two-tier architecture:

```text
Browser Client
     │
     ▼
Nginx Reverse Proxy / Web Server (Port 80/443)
 ├── Serves React Static Bundle (Frontend/dist)
 └── Proxies /api Requests ──► FastAPI Uvicorn Server (Backend, Port 8000)
                                 │
                                 ├── Systemd Service (resolvex-backend)
                                 ├── LangGraph State Machine
                                 ├── FAISS & BM25 Knowledge Store
                                 └── External Groq LLM API
```

---

## 2. System Prerequisites

- **OS**: Linux (Ubuntu 22.04 LTS recommended)
- **Python**: `Python 3.11`
- **Node.js**: `Node 18+` / `npm` (for frontend build)
- **Web Server**: `Nginx`
- **Database**: SQLite (default single-node) or AWS RDS PostgreSQL
- **System Service**: `systemd`

---

## 3. Backend Deployment (Systemd + Uvicorn)

### 3.1 Environment Setup

```bash
cd ~/ResolveX-AI/Backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Production Environment File (`Backend/.env`)

Copy `Backend/.env.example` to `Backend/.env` and populate production configuration:

```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
CORS_ORIGINS=http://your-domain.com,https://your-domain.com

# Groq Cloud LLM Credentials
GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=openai/gpt-oss-120b

# PostgreSQL (Optional; defaults to SQLite if unconfigured)
RDS_HOST=your-rds-host.amazonaws.com
RDS_PORT=5432
RDS_DB=resolvex_db
RDS_USER=resolvex_user
RDS_PASSWORD=your_secure_password
RDS_SSL=True
```

> [!WARNING]
> Never commit `Backend/.env` to Git. Keep `Backend/.env.example` committed with template placeholders only.

### 3.3 Systemd Service Configuration (`/etc/systemd/system/resolvex-backend.service`)

```ini
[Unit]
Description=ResolveX FastAPI Backend Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ResolveX-AI/Backend
ExecStart=/home/ubuntu/ResolveX-AI/Backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
EnvironmentFile=/home/ubuntu/ResolveX-AI/Backend/.env
StandardOutput=append:/home/ubuntu/ResolveX-AI/Backend/logs/backend.log
StandardError=append:/home/ubuntu/ResolveX-AI/Backend/logs/backend_error.log

[Install]
WantedBy=multi-user.target
```

### 3.4 Enable & Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable resolvex-backend
sudo systemctl start resolvex-backend
sudo systemctl status resolvex-backend
```

---

## 4. Frontend Deployment (Nginx)

### 4.1 Build Production Static Bundle

```bash
cd ~/ResolveX-AI/Frontend
cp .env.example .env
# Set VITE_API_URL="http://your-domain.com/api" in .env
npm install
npm run build
```

### 4.2 Deploy Dist Bundle to Web Server

```bash
sudo mkdir -p /var/www/resolvex
sudo rm -rf /var/www/resolvex/*
sudo cp -r dist/* /var/www/resolvex/
sudo chown -R www-data:www-data /var/www/resolvex
```

### 4.3 Nginx Configuration (`/etc/nginx/sites-available/resolvex`)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # React Frontend Static Assets
    location / {
        root /var/www/resolvex;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # FastAPI Backend Proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Request-ID $http_x_request_id;
        proxy_cache_bypass $http_upgrade;
    }

    # Top-Level Health Check Endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }
}
```

```bash
sudo ln -sf /etc/nginx/sites-available/resolvex /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 5. Model & Index Artifacts

- **FAISS & DocStore**: Cached in `Backend/data/faiss/index.faiss` and `Backend/data/faiss/docstore.json`. Loaded on demand by vector store modules.
- **Embedding & Reranker Models**:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Downloaded once during initial run and cached locally in `~/.cache/huggingface/`.

---

## 6. Health & Readiness Endpoints

- **Liveness Probe**: `GET /health` or `GET /api/v1/health`
  - Response: `{"status": "ok", "app": "ResolveX-AI", "version": "1.0.0"}`
- **Readiness Probe**: `GET /api/v1/health/ready`
  - Response: `{"status": "ready", "database": "ok", "vector_store": "ok"}`

---

## 7. Deployment Smoke Testing

Execute after deployment:

```bash
# 1. Health Liveness Check
curl -i http://localhost:8000/health

# 2. Readiness Check
curl -i http://localhost:8000/api/v1/health/ready

# 3. Create Ticket Verification (API + Graph Flow)
curl -i -X POST http://localhost:8000/api/v1/tickets \
  -F "title=Email Sync Issue" \
  -F "description=Cannot sync corporate email inbox on mobile device." \
  -F "category=software" \
  -F "submitted_by=user@company.com"
```

---

## 8. Safety & Security Considerations

- **CORS Policy**: Configured strictly to match deployed frontend origins in `CORS_ORIGINS`. Wildcards (`*`) are prohibited in production mode.
- **Fail-Closed Safety**: In the event of external LLM API failure or rate limit exhaustion, the system defaults safely to `ask_clarification` or `human_review`.
- **Secret Redaction**: Production logs automatically redact API keys (`gsk_`, `sk-`, `Bearer`, `password=`).

---

## 9. Rollback & Service Shutdown

```bash
# Restart Backend
sudo systemctl restart resolvex-backend

# Stop Backend
sudo systemctl stop resolvex-backend

# Revert Frontend Build
cd ~/ResolveX-AI
git checkout main
cd Frontend && npm run build
sudo cp -r dist/* /var/www/resolvex/
sudo systemctl reload nginx
```
