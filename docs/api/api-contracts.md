# ResolveX-AI — API Contracts & Endpoints Baseline

## 1. Endpoint Summary Table

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Purpose | Request Type | Response Schema | Auth | File |
| ------ | -------- | ------- | ------------ | --------------- | ---- | ---- |
| `POST` | `/tickets` | Submit new ticket & run AI pipeline | Multipart Form | `TicketResponse` (201) | None | [ticket_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/ticket_routes.py#L26) |
| `GET` | `/tickets` | List tickets with pagination | Query Params | `TicketListResponse` (200) | None | [ticket_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/ticket_routes.py#L92) |
| `GET` | `/tickets/{ticket_id}` | Fetch single ticket details | Path Parameter | `TicketResponse` (200) | None | [ticket_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/ticket_routes.py#L105) |
| `GET` | `/tickets/{ticket_id}/pipeline` | Fetch visual progression steps | Path Parameter | `TicketPipelineTracker` (200) | None | [ticket_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/ticket_routes.py#L112) |
| `PATCH` | `/tickets/{ticket_id}` | Update ticket fields | JSON | `TicketResponse` (200) | None | [ticket_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/ticket_routes.py#L192) |
| `DELETE` | `/tickets/{ticket_id}` | Delete ticket record | Path Parameter | 204 No Content | None | [ticket_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/ticket_routes.py#L199) |
| `POST` | `/resolve/{ticket_id}` | Manually trigger AI pipeline | Query (`force`) | `ResolutionResult` (200) | None | [resolution_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/resolution_routes.py#L22) |
| `GET` | `/resolve/{ticket_id}` | Fetch stored AI resolution | Path Parameter | `ResolutionResult` (200) | None | [resolution_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/resolution_routes.py#L33) |
| `POST` | `/hitl/review` | Submit human agent review | JSON | `MessageResponse` (200) | None | [resolution_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/resolution_routes.py#L40) |
| `GET` | `/analytics` | Aggregated analytics snapshot | None | `AnalyticsSummary` (200) | None | [analytics_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/analytics_routes.py#L20) |
| `GET` | `/analytics/tickets` | Ticket status statistics | None | `TicketStats` (200) | None | [analytics_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/analytics_routes.py#L27) |
| `GET` | `/analytics/confidence` | Confidence statistics | None | `ConfidenceStats` (200) | None | [analytics_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/analytics_routes.py#L34) |
| `GET` | `/health` | Application liveness probe | None | JSON `{"status": "ok"}` (200) | None | [health_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/health_routes.py#L11) |
| `GET` | `/health/ready` | Dependency readiness probe | None | JSON `{"status": "ready"}` (200) | None | [health_routes.py](file:///d:/ResolveX-AI/Backend/app/routes/health_routes.py#L21) |

---

## 2. Detailed Endpoint Contracts

### 2.1 `POST /tickets`

* **Purpose**: Creates a ticket record in the database and automatically triggers the full AI resolution pipeline.
* **Content-Type**: `multipart/form-data`
* **Form Fields**:
  * `title` (str, required): Ticket title ($3 \le \text{len} \le 255$).
  * `description` (str, required): Ticket problem description ($\text{len} \ge 10$).
  * `category` (str, required): Initial category tag (e.g. `software`, `hardware`, `network`).
  * `submitted_by` (str, optional): User email address.
  * `image` (file, optional): Uploaded attachment file.
* **Success Response** (HTTP 201 Created):
```json
{
  "id": 1,
  "title": "Login fails with 500 error",
  "description": "I get a 500 Internal Server Error when trying to log in.",
  "category": "software",
  "status": "auto_resolved",
  "solution": "Clear sessions, restart auth service, check DB connection.",
  "confidence": 88,
  "explanation": "## ResolveX-AI Decision Explanation...",
  "decision": "auto-resolved",
  "intent": "software",
  "submitted_by": "user@example.com",
  "assigned_to": null,
  "assigned_resolver_id": null,
  "assigned_resolver_name": null,
  "assigned_resolver_category": null,
  "created_at": "2026-09-23T18:00:00Z",
  "updated_at": "2026-09-23T18:00:01Z"
}
```

---

### 2.2 `GET /tickets`

* **Purpose**: Retrieves a paginated list of support tickets.
* **Query Parameters**:
  * `page` (int, default=1, $\ge 1$): Page number.
  * `page_size` (int, default=20, $1 \le N \le 100$): Items per page.
  * `status` (str, optional): Filter by status (`open`, `in_progress`, `auto_resolved`, `escalated`, `closed`).
  * `category` (str, optional): Filter by category.
* **Success Response** (HTTP 200 OK):
```json
{
  "total": 5,
  "page": 1,
  "page_size": 20,
  "tickets": [
    {
      "id": 1,
      "title": "Login fails with 500 error",
      "description": "...",
      "category": "software",
      "status": "auto_resolved",
      "solution": "...",
      "confidence": 0.88,
      "explanation": "...",
      "submitted_by": "user@example.com",
      "assigned_to": null,
      "created_at": "2026-09-23T18:00:00Z",
      "updated_at": "2026-09-23T18:00:01Z"
    }
  ]
}
```

---

### 2.3 `GET /tickets/{ticket_id}/pipeline`

* **Purpose**: Generates the visual progression tracker steps for a specific ticket.
* **Path Parameter**: `ticket_id` (int)
* **Success Response** (HTTP 200 OK):
```json
{
  "ticket_id": 1,
  "current_step": "closed",
  "steps": [
    {
      "key": "created",
      "label": "Ticket Created",
      "state": "completed",
      "timestamp": "2026-09-23T18:00:00Z",
      "details": null
    },
    {
      "key": "extracted",
      "label": "Context Extracted",
      "state": "completed",
      "timestamp": "2026-09-23T18:00:01Z",
      "details": null
    },
    {
      "key": "classified",
      "label": "AI Classification",
      "state": "completed",
      "timestamp": null,
      "details": { "predicted_category": "software" }
    },
    {
      "key": "solution",
      "label": "AI Resolution Ready",
      "state": "completed",
      "timestamp": null,
      "details": { "confidence_score": 0.88 }
    },
    {
      "key": "decision",
      "label": "Decision / Review",
      "state": "completed",
      "timestamp": null,
      "details": { "decision": "Auto-Resolved by AI" }
    },
    {
      "key": "closed",
      "label": "Closed",
      "state": "pending",
      "timestamp": null,
      "details": null
    }
  ]
}
```

---

### 2.4 `POST /resolve/{ticket_id}`

* **Purpose**: Forces execution of the AI resolution pipeline for an existing ticket.
* **Query Parameter**: `force` (bool, default=false): Re-run even if already resolved.
* **Success Response** (HTTP 200 OK):
```json
{
  "ticket_id": 1,
  "category": "software",
  "solution": "Clear sessions, restart auth service, check DB connection.",
  "confidence": 0.88,
  "auto_resolved": true,
  "escalated_to_human": false,
  "explanation": "## ResolveX-AI Decision Explanation...",
  "assigned_resolver_id": null,
  "assigned_resolver_name": null,
  "assigned_resolver_category": null
}
```

---

### 2.5 `POST /hitl/review`

* **Purpose**: Receives manual resolution input from a human support agent for an escalated ticket.
* **Content-Type**: `application/json`
* **Payload**:
```json
{
  "ticket_id": 2,
  "agent_id": 1,
  "agent_solution": "Manually re-sent March invoice PDF to customer email.",
  "close_ticket": true
}
```
* **Success Response** (HTTP 200 OK):
```json
{
  "message": "Ticket 2 reviewed and closed.",
  "success": true
}
```

---

### 2.6 `GET /analytics`

* **Purpose**: Combined snapshot of ticket volumes, average confidence, and category breakdowns.
* **Success Response** (HTTP 200 OK):
```json
{
  "ticket_stats": {
    "total_tickets": 5,
    "open_tickets": 0,
    "auto_resolved": 3,
    "escalated": 2,
    "closed": 0
  },
  "confidence_stats": {
    "avg_confidence": 0.792,
    "high_confidence_count": 3,
    "low_confidence_count": 1
  },
  "category_breakdown": {
    "category_counts": {
      "software": 3,
      "access_permission": 1,
      "other": 1
    }
  }
}
```
