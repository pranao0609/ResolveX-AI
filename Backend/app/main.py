"""
main.py — FastAPI application factory for ResolveX-AI.

Registers all routers, sets up CORS, lifespan events, and global exception handlers.
"""

import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import settings
from app.database import init_db
from app.core.logger import logger
from app.core.exceptions import ResolveXException
from app.routes import ticket_routes, resolution_routes, analytics_routes, health_routes


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup logic, validate configuration, then yield, then run shutdown logic."""
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION} (env: {settings.ENVIRONMENT})")
    
    # Configuration validation
    try:
        settings.validate_critical_settings()
        logger.info(f"✅ Configuration validated (Auto-Resolve: {settings.AUTO_RESOLVE_THRESHOLD}, HITL: {settings.HITL_THRESHOLD})")
        logger.info(f"🔒 CORS origins configured: {settings.parsed_cors_origins}")
        has_groq_key = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your-groq-api-key-here")
        logger.info(f"🔑 Groq API Key configured: {has_groq_key}")
    except Exception as exc:
        logger.critical(f"❌ Configuration validation failed: {exc}")
        raise

    init_db()       # create tables if they don't exist
    logger.info("✅ Database tables initialised")
    yield
    logger.info("👋 Shutting down ResolveX-AI")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Intelligent Auto-Handling of Support Tickets "
        "with Confidence-Based Human-in-the-Loop"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware: Request Correlation ID ───────────────────────────────────────

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Assign a unique X-Request-ID header to every incoming HTTP request."""
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins,   # configured allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(ResolveXException)
async def resolvex_exception_handler(request: Request, exc: ResolveXException):
    logger.error(f"ResolveXException [req_id={getattr(request.state, 'request_id', 'n/a')}]: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "n/a")
    logger.exception(f"Unhandled exception [req_id={req_id}]")
    
    # Hide details in production environment
    if settings.ENVIRONMENT.lower() == "production":
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": req_id})
    
    return JSONResponse(status_code=500, content={"detail": str(exc), "request_id": req_id})


# ── Routers ───────────────────────────────────────────────────────────────────

API_PREFIX = "/api/v1"

app.include_router(health_routes.router,     prefix=API_PREFIX, tags=["Health"])
app.include_router(ticket_routes.router,     prefix=API_PREFIX, tags=["Tickets"])
app.include_router(resolution_routes.router, prefix=API_PREFIX, tags=["Resolution"])
app.include_router(analytics_routes.router,  prefix=API_PREFIX, tags=["Analytics"])
