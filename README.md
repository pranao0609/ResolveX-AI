<div align="center">
  
# 🚀 ResolveX-AI
**Intelligent Auto-Handling of Support Tickets with Confidence-Based Human-in-the-Loop**

**Finalist Demo Video Link:** [https://youtu.be/QMl_igWXLVM](https://youtu.be/QMl_igWXLVM) <br/>
> *Experience the real-time AI resolution engine live in production.*

</div>

<br/>

## 📖 Overview
**ResolveX-AI** is a cutting-edge, automated customer support ticketing platform. It leverages state-of-the-art vector embeddings (FAISS) and Large Language Models (Groq's Llama-3) to instantly read, classify, and solve incoming user tickets. 

If the AI's confidence score drops below a specific threshold, it triggers a strict **Human-In-The-Loop (HITL)** protocol, dynamically escalating the ticket to the most appropriate human expert while providing the expert with the AI's intermediate context. 

## ✨ Key Features
- **Instant AI Auto-Resolution:** Fully autonomous ticket solving using Lightning-fast Groq APIs and semantic context retrieval.
- **True Real-Time Execution Trace:** Watch the AI "think". The frontend utilizes a native **WebSocket integration** to stream the background AI pipeline step-by-step (`Extracted` → `Classified` → `Resolution` → `Decision`), visualizing the progress instantly.
- **Always-Up Resilience:** Engineered with React Error Boundaries. If one chart or component fails, the rest of the application stays perfectly functional without a "White Screen of Death".
- **Zero-Bloat Code Splitting:** `React.lazy()` ensures users only download the Javascript modules they actively navigate to, resulting in blazing fast frontend loads.

---

## 🛠️ Technology Stack

### Frontend Architecture
- **Framework:** React 19 + TypeScript + Vite
- **Styling:** Tailwind CSS v4 + Lucide React Icons
- **Data Vis:** Recharts (Analytics Dashboard)
- **Performance:** Route-Level Lazy Loading, `React.memo` structurals

### Backend Architecture
- **Framework:** FastAPI (Python 3.10+) 
- **Database:** PostgreSQL via SQLAlchemy ORM
- **Async Engine:** FastAPI `BackgroundTasks` + WebSockets ConnectionManager
- **AI Integration:** Sentence Transformers, FAISS (Vector DB), Groq API 

---

## 📁 Repository Structure

```text
ResolveX-AI/
├── Backend/                 # Python FastAPI Server
│   ├── ai/                  # RAG Pipeline, Semantic Search Logic
│   ├── app/
│   │   ├── core/            # Websockets, Exceptions, Logger
│   │   ├── models/          # SQLAlchemy DB Models
│   │   ├── routes/          # RESTful + WebSocket Endpoints
│   │   ├── schemas/         # Pydantic validation shapes
│   │   └── services/        # Business Logic & Background Handlers
│   ├── data/                # Vector store binaries
│   └── requirements.txt     # Python Dependencies
│
└── Frontend/                # React Vite Application
    ├── src/
    │   ├── components/      # Global UI (ErrorBoundary, Navbar, Sidebar)
    │   ├── pages/           # Code-split views (Analytics, AuditLogs, etc)
    │   └── services/        # Fetch API wrappers
    ├── index.html
    └── package.json
```

---

## ⚙️ Local Development Setup

### 1. Backend Setup
Navigate to the `Backend` directory, install requirements, and configure your database.

```bash
cd Backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

#### Set up Backend `.env`
Create a `.env` file in the `Backend` root and paste the following template. **IMPORTANT: Never commit your actual `.env` file!**

```env
# ─────────────────────────────────────────────────────────────────────────────
# Backend / .env
# ─────────────────────────────────────────────────────────────────────────────

# Application
APP_NAME=ResolveX-AI
APP_VERSION=1.0.0
DEBUG=true

# PostgreSQL (RDS / Local DB)
RDS_HOST=your-db-host.amazonaws.com
RDS_PORT=5432
RDS_DB=your_database_name
RDS_USER=your_db_user
RDS_PASSWORD=your_secure_password
RDS_SSL=false

# Groq LLM Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# AI Confidence Thresholds
AUTO_RESOLVE_THRESHOLD=0.75     # Above this → auto resolve
HITL_THRESHOLD=0.50             # Below this → send to human

# File / Vector Storage
UPLOAD_DIR=uploads
MAX_FILE_SIZE_MB=10
FAISS_INDEX_PATH=storage/faiss_index.bin
```

Start the FastAPI server:
```bash
uvicorn app.main:app --reload --port 8000
```
*API docs available at `http://localhost:8000/docs`*

### 2. Frontend Setup
Open a new terminal, navigate to the `Frontend` directory, and install the NPM packages.

```bash
cd Frontend
npm install
```

#### Set up Frontend `.env`
Create a `.env` file in the `Frontend` root.

```env
# ─────────────────────────────────────────────────────────────────────────────
# Frontend / .env
# ─────────────────────────────────────────────────────────────────────────────

VITE_API_URL=http://localhost:8000/api
```

Start the Vite development server:
```bash
npm run dev
```

---

## 🚀 Live Testing the WebSocket Flow
1. Submit a new ticket through the UI or `POST /tickets`.
2. Instantly click on the **Audit Logs** tab on the sidebar.
3. Watch the UI animate through the steps (`Extracted`, `Classified`, etc.) as the backend server calculates the Groq RAG pipelines in the background and broadcasts the status via WebSockets!

---

> Built with passion to redefine what Automated SaaS Ticketing means.
