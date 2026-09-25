# ResolveX-AI — Baseline Reproduction & Setup Guide

This guide provides step-by-step instructions to reproduce and run the exact **ResolveX-AI baseline system** on a local environment.

---

## 1. Environment Prerequisites

* **Operating System**: Windows, Linux, or macOS.
* **Python**: Version 3.10, 3.11, or 3.12.
* **Node.js**: Version 18.0+ and `npm`.
* **Database**: PostgreSQL server (local instance or AWS RDS instance).
* **External API Key**: Valid Groq API Key ([console.groq.com](https://console.groq.com)).

---

## 2. Backend Setup

1. **Navigate to the Backend directory**:
   ```bash
   cd Backend
   ```

2. **Create and activate a Python virtual environment**:
   * **Linux/macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file inside `Backend/` with the following variables:
   ```env
   APP_NAME=ResolveX-AI
   APP_VERSION=1.0.0
   DEBUG=true

   RDS_HOST=localhost
   RDS_PORT=5432
   RDS_DB=resolvex
   RDS_USER=postgres
   RDS_PASSWORD=your_postgres_password
   RDS_SSL=False

   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile

   AUTO_RESOLVE_THRESHOLD=0.75
   HITL_THRESHOLD=0.50

   UPLOAD_DIR=uploads
   MAX_FILE_SIZE_MB=10
   FAISS_INDEX_PATH=storage/faiss_index.bin
   ```

5. **Seed Database and Build FAISS Index**:
   Run the seeding script to initialize database tables, seed 195 KB entries, and populate the FAISS vector index:
   ```bash
   python scripts/seed_data.py
   ```

6. **Start the FastAPI Backend Server**:
   ```bash
   python run.py
   ```
   The backend will start at `http://localhost:8000`. API documentation is available at `http://localhost:8000/docs`.

---

## 3. Frontend Setup

1. **Navigate to the Frontend directory**:
   ```bash
   cd Frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Configure Frontend Environment Variables**:
   Create a `.env` file inside `Frontend/`:
   ```env
   VITE_API_URL="http://localhost:8000/api"
   ```

4. **Start the Vite Development Server**:
   ```bash
   npm run dev
   ```
   The frontend application will start at `http://localhost:5173`.

---

## 4. Verification & Testing Steps

1. **Verify Backend Liveness**:
   Open a terminal or browser and query:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```
   Expected response: `{"status": "ok", "app": "ResolveX-AI", "version": "1.0.0"}`.

2. **Test End-to-End Resolution**:
   * Open `http://localhost:5173` in your browser and click on **Simulation** in the sidebar.
   * Submit a new test ticket (e.g. Title: "Login 500 error", Category: "software", Description: "I get a 500 internal server error when trying to log in.").
   * Observe the pipeline progress steps and verify that an AI solution and confidence score are generated and returned.
