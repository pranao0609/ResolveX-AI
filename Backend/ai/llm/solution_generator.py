"""
solution_generator.py — ResolveX structured LLM resolution generation.

Prompt loading and rendering remain here because they are part of the
resolution-generation domain.

LLM reliability responsibilities are delegated to LLMGateway.
"""

from __future__ import annotations

from typing import Tuple

from app.config import settings
from app.core.logger import logger

from ai.config.ai_config import (
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
)

from ai.llm.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)

from ai.llm.gateway import LLMGateway

from ai.llm.providers.groq_provider import (
    GroqProvider,
)

from ai.llm.prompt_loader import (
    load_prompt,
    render_prompt,
)

from ai.llm.schemas import (
    ResolutionResult,
)

try:
    from groq import Groq  # noqa: F401

    GROQ_AVAILABLE = True

except ImportError:
    GROQ_AVAILABLE = False
    
DEFAULT_PROMPT_VERSION = 1

def _parse_resolution(
    response_text: str,
) -> ResolutionResult:
    """
    Parse and validate an LLM response into ResolutionResult.

    Kept for backward compatibility with existing tests and
    callers. The production generation path uses LLMGateway
    for structured-output validation.
    """

    cleaned_response = response_text.strip()

    if cleaned_response.startswith("```"):
        cleaned_response = (
            cleaned_response
            .removeprefix("```json")
            .strip()
        )

        cleaned_response = (
            cleaned_response
            .removeprefix("```")
            .strip()
        )

        if cleaned_response.endswith("```"):
            cleaned_response = (
                cleaned_response[:-3]
                .strip()
            )

    return ResolutionResult.model_validate_json(
        cleaned_response
    )

def _get_gateway() -> LLMGateway:
    """
    Create the configured LLM gateway.

    Provider construction remains isolated here so the rest of the
    resolution-generation code does not depend directly on Groq.
    """

    provider = GroqProvider(
        api_key=settings.GROQ_API_KEY,
    )

    return LLMGateway(
        provider=provider,
    )

def _fallback_reason(exc: Exception) -> str:
    """
    Convert an LLM failure into a stable fallback reason.
    """

    if isinstance(exc, LLMAuthenticationError):
        return "authentication_error"

    if isinstance(exc, LLMTimeoutError):
        return "timeout"

    if isinstance(exc, LLMRateLimitError):
        return "rate_limit"

    if isinstance(exc, LLMInvalidResponseError):
        return "invalid_response"

    if isinstance(exc, LLMError):
        return "provider_error"

    return "unknown_error"

def generate_solution(
    ticket_text: str,
    context: str,
    diagnosis: str = "",
    root_cause: str = "",
    classification: str = "",
    conversation_history: str = "",
    prompt_version: int = DEFAULT_PROMPT_VERSION,
) -> Tuple[ResolutionResult, bool]:
    """
    Generate a structured ticket resolution through the LLM Gateway.

    Args:
        ticket_text:
            Pre-processed ticket description.

        context:
            Concatenated RAG retrieval results.

        prompt_version:
            Version of the resolution prompt.

    Returns:
        Tuple containing:

        - ResolutionResult
        - fallback_used
    """

    # ------------------------------------------------------------------
    # Load versioned prompt
    # ------------------------------------------------------------------

    prompt = load_prompt(
        "resolution",
        prompt_version,
    )

    clean_ticket = (
        ticket_text[:4000]
        if ticket_text
        else ""
    )

    clean_context = (
        context.strip()
        if context
        else "No context available."
    )

    system_prompt, user_prompt = render_prompt(
        prompt,
        ticket_text=clean_ticket,
        context=clean_context,
    )
    diagnosis_section = (
        "### Diagnosis\n"
        f"Problem: {diagnosis.strip() or 'Not available'}\n"
        f"Possible Root Cause: "
        f"{root_cause.strip() or 'Not available'}\n"
    )

    classification_section = (
        "### Classification\n"
        f"{classification.strip() or 'Not available'}\n"
    )

    history_section = (
        "### Conversation History\n"
        f"{conversation_history.strip() or 'No previous conversation available.'}\n"
    )

    user_prompt = (
        f"{user_prompt}\n\n"
        f"{classification_section}\n"
        f"{diagnosis_section}\n"
        f"{history_section}"
    )

    # ------------------------------------------------------------------
    # Resolve model/runtime configuration
    # ------------------------------------------------------------------

    prompt_model = prompt.get(
        "model",
        GROQ_MODEL,
    )

    prompt_temperature = prompt.get(
        "temperature",
        GROQ_TEMPERATURE,
    )

    model = (
        settings.GROQ_MODEL
        or prompt_model
    )

    temperature = prompt_temperature

    # ------------------------------------------------------------------
    # Gateway availability
    # ------------------------------------------------------------------

    try:
        gateway = _get_gateway()

    except ValueError as exc:
        logger.warning(
            "LLM provider unavailable: "
            f"{exc}"
        )

        return _placeholder_solution(), True

    # ------------------------------------------------------------------
    # LLM Gateway execution
    # ------------------------------------------------------------------

    try:
        response = gateway.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=GROQ_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )

        resolution = gateway.validate_resolution(
            response,
        )

        logger.info(
            "Structured resolution generated "
            f"prompt_version={prompt_version} "
            f"model={model} "
            f"confidence={resolution.confidence:.2f} "
            f"prompt_tokens="
            f"{response.usage.prompt_tokens} "
            f"completion_tokens="
            f"{response.usage.completion_tokens} "
            f"total_tokens="
            f"{response.usage.total_tokens}"
        )

        return resolution, False

    except LLMError as exc:
        reason = _fallback_reason(exc)

        logger.error(
            "LLM resolution generation failed; "
            f"using safe fallback "
            f"reason={reason} "
            f"prompt_version={prompt_version} "
            f"model={model} "
            f"error={type(exc).__name__}: {exc}"
        )

    return _placeholder_solution(), True


def _placeholder_solution() -> ResolutionResult:
    """
    Return a safe structured fallback when the LLM
    cannot generate a validated resolution.

    The fallback deliberately avoids claiming a specific
    root cause or resolution when the LLM failed.
    """

    return ResolutionResult(
        diagnosis=(
            "The reported support issue requires "
            "further investigation."
        ),
        root_cause=(
            "No validated root cause could be established "
            "by the automated resolution system."
        ),
        resolution_steps=[
            (
                "Review the ticket details and reproduce "
                "the reported issue."
            ),
            (
                "Collect relevant application, system, "
                "or service logs."
            ),
            (
                "Escalate the ticket to a support agent "
                "for further investigation."
            ),
        ],
        evidence=[],
        confidence=0.0,
        requires_human=True,
    )