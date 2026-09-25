# ResolveX-AI — Dependency & Component Inventory

## 1. Backend Python Dependencies ([requirements.txt](file:///d:/ResolveX-AI/Backend/requirements.txt))

| Package Name | Specified Version | Purpose | Actual Status | Primary File Location |
| ------------ | ----------------- | ------- | ------------- | --------------------- |
| `fastapi` | Unpinned | Web application framework | Active | [app/main.py](file:///d:/ResolveX-AI/Backend/app/main.py) |
| `uvicorn[standard]` | Unpinned | ASGI web server | Active | [run.py](file:///d:/ResolveX-AI/Backend/run.py) |
| `pydantic` | Unpinned | Data schema validation | Active | [app/schemas/ticket_schema.py](file:///d:/ResolveX-AI/Backend/app/schemas/ticket_schema.py) |
| `pydantic-settings` | Unpinned | Environment configuration management | Active | [app/config.py](file:///d:/ResolveX-AI/Backend/app/config.py) |
| `sqlalchemy` | Unpinned | Object-Relational Mapping (ORM) | Active | [app/database.py](file:///d:/ResolveX-AI/Backend/app/database.py) |
| `psycopg2-binary` | Unpinned | PostgreSQL database driver | Active | [app/database.py](file:///d:/ResolveX-AI/Backend/app/database.py) |
| `sentence-transformers` | Unpinned | Dense vector embedding generation | Active | [ai/embedding/embedding_model.py](file:///d:/ResolveX-AI/Backend/ai/embedding/embedding_model.py) |
| `faiss-cpu` | Unpinned | Efficient vector similarity search | Active | [ai/rag/vector_store.py](file:///d:/ResolveX-AI/Backend/ai/rag/vector_store.py) |
| `groq` | Unpinned | Groq API LLM client | Active | [ai/llm/solution_generator.py](file:///d:/ResolveX-AI/Backend/ai/llm/solution_generator.py) |
| `python-multipart` | Unpinned | Form data & file upload handling | Active | [app/routes/ticket_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/ticket_routes.py) |
| `pillow` | Unpinned | PIL image processing library | Active | [ai/preprocessing/ocr_processor.py](file:///d:/ResolveX-AI/Backend/ai/preprocessing/ocr_processor.py) |
| `python-dotenv` | Unpinned | `.env` environment loading | Active | [app/config.py](file:///d:/ResolveX-AI/Backend/app/config.py) |
| `pytesseract` | Unpinned | OCR text extraction wrapper | Partial | [ai/preprocessing/ocr_processor.py](file:///d:/ResolveX-AI/Backend/ai/preprocessing/ocr_processor.py) |
| `PyPDF2` | Unpinned | PDF file text extraction | Active | [ai/preprocessing/file_parser.py](file:///d:/ResolveX-AI/Backend/ai/preprocessing/file_parser.py) |
| `alembic` | Unpinned | Database migration tool | **Unused** | Declared in requirements, no scripts exist |
| `httpx` | Unpinned | Async HTTP client | Active | Used implicitly by FastAPI tests/clients |

---

## 2. Frontend Node.js Dependencies ([package.json](file:///d:/ResolveX-AI/Frontend/package.json))

### Production Dependencies

| Package | Version | Purpose | Status |
| ------- | ------- | ------- | ------ |
| `react` | `^19.2.4` | Component framework | Active |
| `react-dom` | `^19.2.4` | DOM rendering engine | Active |
| `react-router-dom` | `^7.13.1` | Client-side routing | Active |
| `tailwindcss` | `^4.2.2` | Utility CSS styling | Active |
| `@tailwindcss/vite` | `^4.2.2` | Tailwind Vite integration | Active |
| `recharts` | `^3.8.1` | Analytics charts & visualizations | Active |
| `lucide-react` | `^1.6.0` | UI icon set | Active |
| `react-markdown` | `^10.1.0` | Markdown rendering | Active |
| `rehype-highlight` | `^7.0.2` | Syntax highlighting | Active |
| `rehype-katex` | `^7.0.1` | Math formula rendering | Active |
| `remark-gfm` | `^4.0.1` | GitHub Flavored Markdown plugin | Active |

### Development Dependencies

| Package | Version | Purpose |
| ------- | ------- | ------- |
| `typescript` | `~5.9.3` | Type checking and compilation |
| `vite` | `^8.0.1` | Fast dev server and bundler |
| `@vitejs/plugin-react` | `^6.0.1` | React plugin for Vite |
| `eslint` | `^9.39.4` | Linter |

---

## 3. AI & ML Model Inventory

| Model Name / Identifier | Source / Provider | Inference Location | Dimension / Size | Task Purpose |
| ----------------------- | ----------------- | ------------------ | ---------------- | ------------ |
| `facebook/bart-large-mnli` | HuggingFace Hub | Local CPU (PyTorch) | 407M params | Zero-shot ticket classification |
| `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace Hub | Local CPU (PyTorch) | 384 dimensions | Text vector embedding generation |
| `llama-3.3-70b-versatile` | Groq Cloud API | External Groq Cloud | 70B params | Ticket solution generation & LLM fallback |

---

## 4. Infrastructure Specifications

* **Database Server**: PostgreSQL (AWS RDS instance `resolvex-ai.cbysk4062bt7.ap-south-1.rds.amazonaws.com`).
* **Vector Store**: Local FAISS `IndexFlatIP` persisted on disk at `Backend/data/faiss/index.faiss`.
* **File Storage**: Local filesystem at `Backend/uploads/`.
* **Deployment Target**: AWS EC2 instance running Ubuntu Linux, Nginx, and systemd service `resolvex-backend`.
