# ResolveX — Continuous Integration (CI/CD) Architecture

## 1. Overview & Purpose

The ResolveX CI pipeline automatically validates every code push and pull request to ensure high code quality, security posture, formatting consistency, and regression safety without sacrificing fail-closed runtime safety.

---

## 2. Workflow Trigger

The workflow is defined in [`.github/workflows/ci.yml`](file:///d:/ResolveX-AI/.github/workflows/ci.yml) and triggers on:
- `push` to `main` or `develop` branches.
- `pull_request` targeting `main` or `develop` branches.

---

## 3. Environment & Python Version

- **Runner Environment**: `ubuntu-latest`
- **Target Python Version**: `Python 3.11`
- **Isolation Mode**: Non-privileged runner execution (`pull_request` event scope).

---

## 4. Pipeline Validation Stages

| Stage | Command / Action | Description | Fail Policy |
| :--- | :--- | :--- | :--- |
| **Checkout** | `actions/checkout@v4` | Checks out full workspace history | Mandatory |
| **Python Setup** | `actions/setup-python@v5` | Provisions Python 3.11 with `pip` caching | Mandatory |
| **Dependency Install** | `pip install -r Backend/requirements.txt black ruff` | Installs runtime, dev, and test tools | Mandatory |
| **Format Check** | `black --check Backend` | Enforces Black code formatting in check mode | Fail on format mismatch |
| **Code Linting** | `ruff check Backend` | Checks Python code against lint rules in `pyproject.toml` | Fail on lint error |
| **Security Tests** | `pytest Backend/tests/security -v` | Executes dedicated security test suite (23+ tests) | Fail on test failure |
| **Full Test Suite** | `pytest Backend/tests` | Executes complete test suite (753+ tests) | Fail on test failure |
| **Benchmark Smoke Test**| `python Backend/scripts/benchmark_smoke_test.py` | Validates benchmark engine, dataset schema & retrieval evaluation | Fail on schema/engine error |

---

## 5. Benchmark Smoke Test vs. Live Evaluation

- **CI Smoke Test**: Runs `Backend/scripts/benchmark_smoke_test.py` deterministically in an offline, credential-free environment. It verifies dataset loading, schema validity, retrieval calculations (`Hybrid Recall@5`, `Reranked Recall@5`), and metric aggregation without invoking external LLM APIs.
- **Live Benchmark**: `Backend/scripts/run_final_benchmark.py` executes full end-to-end LLM graph evaluation during local/staging validation using provider credentials (`GROQ_API_KEY`).

---

## 6. Secrets & Credentials Policy

- **No Secrets in CI**: CI pipelines run credential-free to prevent secret exposure to untrusted pull-request branches.
- **Fail-Closed Default**: Environment variables in CI default to `ENVIRONMENT=test` and `DEBUG=false`.
- **Placeholder Example**: Environment defaults are derived from [`Backend/.env.example`](file:///d:/ResolveX-AI/Backend/.env.example).

---

## 7. Local Reproduction Commands

Developers can reproduce all CI checks locally prior to pushing code:

```powershell
# 1. Format check
Backend\venv\Scripts\python.exe -m black --check Backend

# 2. Lint check
Backend\venv\Scripts\python.exe -m ruff check Backend

# 3. Security test suite
Backend\venv\Scripts\python.exe -m pytest Backend/tests/security -v

# 4. Full test suite
Backend\venv\Scripts\python.exe -m pytest Backend/tests

# 5. Offline benchmark engine smoke test
Backend\venv\Scripts\python.exe Backend/scripts/benchmark_smoke_test.py
```

---

## 8. CI Boundary & Limitations

> [!IMPORTANT]
> **CI Boundary Statement**
> CI validates code syntax, formatting, linting rules, security unit boundaries, and deterministic regression behavior.
> CI does **NOT** prove:
> - Real-world live LLM provider uptime or latency.
> - Live production database network connectivity.
> - Third-party API rate limits under heavy production load.
