# ResolveX 2.0 — Production Security Specification & Architecture

This document describes the security architecture, threat model, input validation, and fail-closed guardrails of **ResolveX 2.0**.

---

## 1. Threat Model & Trust Boundaries

### Untrusted Entities
1. **User Ticket Content**: External ticket text, titles, attachment content, and user messages are untrusted inputs.
2. **LLM Generated Output**: Raw model outputs from external inference endpoints are treated as untrusted until verified.

### Security Invariants
- **Data Isolation**: User ticket content must never alter or override agent system instructions.
- **Deterministic Precedence**: Deterministic safety rules (`SafetyPolicyGuard`, `UnsafeActionGuard`, `SecretLeakageGuard`) take absolute precedence over machine-learning policy recommendations.
- **Verification Gate**: No ticket may be `auto_resolve`d without passing evidence-grounded verification.
- **Fail-Closed Default**: System errors, missing information, verification failures, or rate limits default to `ask_clarification`, `human_review`, or `escalate`.

---

## 2. Secret Management & Credential Policy

- Secrets (`GROQ_API_KEY`, `LANGSMITH_API_KEY`, `RDS_PASSWORD`, `DATABASE_URL`) are managed strictly through environment variables and `.env`.
- `.env` files are ignored by source control (`.gitignore`).
- Credentials and API key strings (`gsk_`, `sk-`, `lsv2_`, `Bearer `) are automatically redacted in server logs via `SecretRedactingFormatter`.

---

## 3. Production Configuration & CORS Hardening

- Production configuration validation (`Settings.validate_critical_settings()`) runs during application startup.
- In `ENVIRONMENT=production`:
  - `DEBUG` mode must be `False`.
  - Wildcard CORS origins (`*`) are strictly forbidden.
  - Valid API credentials must be configured.
- `allow_credentials=True` is strictly incompatible with wildcard `*` origins across all environments.

---

## 4. API Input Validation & Security Headers

- FastAPI endpoints enforce Pydantic request models with field constraints (`min_length`, `max_length`, category enums).
- HTTP response headers set defensive security directives:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: no-referrer`
  - `Content-Security-Policy: default-src 'self'; frame-ancestors 'none';`

---

## 5. Prompt-Injection Defense

- System prompts in Prompt Version 2 (`diagnosis`, `resolution`, `verification`) enforce strict prompt injection isolation boundaries.
- Prompts instruct models to treat ticket text as raw data and reject injection attempts such as *"Ignore previous instructions"* or *"Disable verification"*.

---

## 6. Output Guardrails & Safety Guards

- **Unsafe Action Guard**: Automatically detects destructive operations (`DROP DATABASE`, `grant root`, `rm -rf`, `disable firewall`) and overrides `auto_resolve` to `human_review`.
- **Secret Leakage Guard**: Automatically detects accidental credential/API key patterns in generated outputs and overrides `auto_resolve` to `human_review`.

---

## 7. Known Security Limitations

- **LLM Non-Determinism**: Prompts provide robust defense against prompt injection, but defense-in-depth relies on downstream deterministic safety rules.
- **External Provider Dependency**: Availability depends on external LLM provider quotas and rate limits; rate limits trigger safe fail-closed fallback.
