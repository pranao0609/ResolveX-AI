# ResolveX-AI — Current AI Pipeline Specification

## Pipeline Overview

The ResolveX-AI resolution engine processes support tickets through a 7-step sequential pipeline defined in [Backend/ai/pipeline/ticket_pipeline.py](file:///d:/ResolveX-AI/Backend/ai/pipeline/ticket_pipeline.py). The entry point function is `async def run_pipeline(ticket)`.

```text
Ticket Input
  ↓
[Step 1] Preprocessing (Text cleaner & Attachment parser)
  ↓
[Step 2] Classification (Regex -> Zero-shot BART -> Groq LLM fallback)
  ↓
[Step 3] Embedding (SentenceTransformer all-MiniLM-L6-v2)
  ↓
[Step 4] RAG Retrieval (FAISS IndexFlatIP & JSON docstore search)
  ↓
[Step 5] LLM Solution Generation (Groq Llama-3.3-70b)
  ↓
[Step 6] Confidence Calculation (Weighted linear formula)
  ↓
[Step 7] Explainability & Routing (Markdown reasoning & Expert Pool allocation)
```

---

## Detailed Step Specification

### Step 1: Preprocessing

* **File**: [Backend/ai/preprocessing/text_cleaner.py](file:///d:/ResolveX-AI/Backend/ai/preprocessing/text_cleaner.py) & [file_parser.py](file:///d:/ResolveX-AI/Backend/ai/preprocessing/file_parser.py)
* **Function**: `clean_text(ticket.description)` and `parse_attachments(attachment_paths)`
* **Input**: Raw ticket description string and comma-separated attachment file paths.
* **Processing**:
  1. Strips HTML tags via `re.sub(r"<[^>]+>", " ", text)`.
  2. Collapses whitespace via `re.sub(r"\s+", " ", text)`.
  3. Converts text to lowercase.
  4. Parses attached files (PyPDF2 for PDFs, `extract_text_from_image` with pytesseract for images, UTF-8 text reader for `.txt`/`.log`).
* **Output**: Single preprocessed text string.
* **Failure Behavior**: Returns raw string or empty string on error.

---

### Step 2: Classification

* **File**: [Backend/ai/classification/classifier.py](file:///d:/ResolveX-AI/Backend/ai/classification/classifier.py)
* **Function**: `classify_ticket(cleaned_text)`
* **Input**: Preprocessed ticket string.
* **Processing Order**:
  1. **Regex Pattern Matching**: Searches 15 pre-defined regex rules (e.g., `500 internal server`, `vpn`, `mouse`, `password reset`). If matched, maps internal error category to standard taxonomy (`software`, `hardware`, `network`, `access_permission`, `security`, `other`) and returns hardcoded confidence score ($0.92$ to $0.98$).
  2. **Zero-Shot Machine Learning**: If no regex matches, runs HuggingFace `pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)` on `cleaned_text[:512]`. If top score $> 0.35$, calibrates score via $\min(0.95, \text{score} \cdot 1.2 + 0.05)$ and maps to taxonomy.
  3. **Groq LLM Fallback**: If zero-shot score $\le 0.35$ or fails, calls Groq API asking to classify into `IT_CATEGORIES`. Returns parsed category and confidence.
* **Output**: Tuple `(category: str, classification_score: float)`.
* **Failure Behavior**: Defaults to `("software", 0.75)`.

---

### Step 3: Embedding Generation

* **File**: [Backend/ai/embedding/embedding_model.py](file:///d:/ResolveX-AI/Backend/ai/embedding/embedding_model.py)
* **Function**: `generate_embedding(cleaned_text)`
* **Input**: Preprocessed ticket text string.
* **Processing**: Lazy-loads `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`. Encodes input text with `normalize_embeddings=True`.
* **Output**: Dense NumPy float32 vector of shape `(384,)`.
* **Failure Behavior**: Returns a zero vector of shape `(384,)` if model fails to load or execute.

---

### Step 4: RAG Retrieval

* **File**: [Backend/ai/rag/retriever.py](file:///d:/ResolveX-AI/Backend/ai/rag/retriever.py) & [vector_store.py](file:///d:/ResolveX-AI/Backend/ai/rag/vector_store.py)
* **Function**: `retrieve_context(query_embedding)`
* **Input**: 384-dimensional query embedding vector.
* **Processing**:
  1. Validates alignment between FAISS index vector count (`index.faiss`) and JSON document store count (`docstore.json`).
  2. Executes inner-product similarity search: `vector_store.search(query_embedding, top_k=5)`.
  3. Filters out documents with similarity score below threshold (`FAISS_SCORE_THRESHOLD = 0.35`).
* **Output**: List of document metadata dictionaries containing `title`, `category`, `content`, `source`, and `score`.
* **Failure Behavior**: Returns an empty list `[]` if index is empty, unaligned, or fails.

---

### Step 5: LLM Solution Generation

* **File**: [Backend/ai/llm/solution_generator.py](file:///d:/ResolveX-AI/Backend/ai/llm/solution_generator.py)
* **Function**: `generate_solution(cleaned_text, context_text)`
* **Input**: Cleaned ticket text and concatenated RAG retrieved context string.
* **Processing**:
  1. Constructs system prompt: *"You are ResolveX-AI, an expert enterprise support resolution assistant..."*
  2. Invokes Groq API `client.chat.completions.create(model="llama-3.3-70b-versatile", max_tokens=1024, temperature=0.3)`.
  3. Computes heuristic LLM quality score: $\text{llm\_score} = \min\left(\frac{\text{len(solution)}}{1000}, 0.95\right)$.
* **Output**: Tuple `(solution_text: str, llm_score: float)`.
* **Failure Behavior**: Returns hardcoded static template response and score $0.3$ on exception.

---

### Step 6: Confidence Calculation

* **File**: [Backend/ai/confidence/confidence_engine.py](file:///d:/ResolveX-AI/Backend/ai/confidence/confidence_engine.py)
* **Function**: `compute_confidence(similarity_score, llm_score, classification_score)`
* **Input**: `similarity_score` (max RAG similarity), `llm_score` (heuristic length proxy), `classification_score` (classifier confidence).
* **Formula**:
  $$\text{Confidence} = 0.4 \cdot \text{clamp}(S_{\text{sim}}) + 0.3 \cdot \text{clamp}(S_{\text{llm}}) + 0.3 \cdot \text{clamp}(S_{\text{cls}})$$
* **Output**: Composite float score in $[0.0, 1.0]$, rounded to 4 decimal places.

---

### Step 7: Explainability & Expert Routing

* **File**: [Backend/ai/explainability/explainer.py](file:///d:/ResolveX-AI/Backend/ai/explainability/explainer.py) & [Backend/app/core/expert_resolvers.py](file:///d:/ResolveX-AI/Backend/app/core/expert_resolvers.py)
* **Function**: `explain()` & `get_best_expert_resolver(category, ticket_text)`
* **Decision Logic** (in `resolution_service.py`):
  * If $\text{Confidence} \ge 0.75$: Status set to `auto_resolved`.
  * If $\text{Confidence} < 0.75$: Status set to `escalated`.
* **Resolver Allocation**:
  * For escalated tickets, evaluates 38 expert resolvers across categories.
  * Selects resolver by matching keyword occurrences in `resolver["specialization"]`.
  * If no keywords match, falls back to category-level `itertools.cycle` round-robin iterator.
* **Explanation Output**: Markdown text formatting category, confidence %, decision status, supporting KB count, top 5 keyword frequencies, and solution preview.
