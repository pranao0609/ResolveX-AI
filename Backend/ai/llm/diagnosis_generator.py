"""
ResolveX diagnosis generation.

Generates a structured, evidence-grounded diagnosis and
root-cause assessment through the centralized LLM Gateway.
"""

from __future__ import annotations

from app.config import settings
from app.core.logger import logger

from ai.config.ai_config import (
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
)

from ai.llm.errors import LLMError
from ai.llm.gateway import LLMGateway
from ai.llm.providers.groq_provider import GroqProvider
from ai.llm.prompt_loader import load_prompt, render_prompt
from ai.llm.schemas import DiagnosisResult

DEFAULT_DIAGNOSIS_PROMPT_VERSION = 1


def _get_gateway() -> LLMGateway:
    """
    Create the configured LLM Gateway.
    """

    provider = GroqProvider(
        api_key=settings.GROQ_API_KEY,
    )

    return LLMGateway(
        provider=provider,
    )


def _placeholder_diagnosis() -> DiagnosisResult:
    """
    Safe fallback when diagnosis generation fails.
    """

    return DiagnosisResult(
        problem=("The reported support issue requires " "further investigation."),
        possible_root_cause=(
            "No validated root cause could be established "
            "from the available evidence."
        ),
        evidence=[],
        missing_information=["Additional diagnostic information is required."],
        confidence=0.0,
    )


def generate_diagnosis(
    ticket_text: str,
    context: str,
    classification: str = "",
    prompt_version: int = DEFAULT_DIAGNOSIS_PROMPT_VERSION,
) -> tuple[DiagnosisResult, bool]:
    """
    Generate a structured diagnosis through the LLM Gateway.

    The diagnosis agent determines:

        1. What problem is actually occurring.
        2. What root cause is supported by the evidence.
        3. Which evidence supports that conclusion.
        4. What information is still missing.
        5. How confident the system is in the diagnosis.

    It must not generate user-facing resolution instructions.

    Returns:
        (DiagnosisResult, fallback_used)
    """

    # --------------------------------------------------------------
    # Load versioned prompt
    # --------------------------------------------------------------

    prompt = load_prompt(
        "diagnosis",
        prompt_version,
    )

    clean_ticket = ticket_text[:4000] if ticket_text else ""

    clean_context = (
        context.strip() if context else "No retrieved evidence is available."
    )

    clean_classification = (
        classification.strip() if classification else "Unknown / unavailable"
    )

    system_prompt, user_prompt = render_prompt(
        prompt,
        ticket_text=clean_ticket,
        context=clean_context,
    )

    # --------------------------------------------------------------
    # Add classification context
    # --------------------------------------------------------------
    #
    # Classification is deliberately injected separately instead
    # of changing the shared prompt_loader contract.
    #
    # This keeps existing resolution/classification prompts
    # backward compatible.
    # --------------------------------------------------------------

    user_prompt = (
        f"{user_prompt}\n\n" "### Ticket Classification\n" f"{clean_classification}\n"
    )

    # --------------------------------------------------------------
    # Resolve model configuration
    # --------------------------------------------------------------

    prompt_model = prompt.get(
        "model",
        GROQ_MODEL,
    )

    prompt_temperature = prompt.get(
        "temperature",
        GROQ_TEMPERATURE,
    )

    model = settings.GROQ_MODEL or prompt_model

    temperature = prompt_temperature

    # --------------------------------------------------------------
    # Create Gateway
    # --------------------------------------------------------------

    try:
        gateway = _get_gateway()

    except ValueError as exc:
        logger.warning("Diagnosis LLM provider unavailable: " f"{exc}")

        return _placeholder_diagnosis(), True

    # --------------------------------------------------------------
    # Generate diagnosis
    # --------------------------------------------------------------

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

        diagnosis = DiagnosisResult.model_validate_json(response.content)

        logger.info(
            "Structured diagnosis generated "
            f"prompt_version={prompt_version} "
            f"model={model} "
            f"confidence={diagnosis.confidence:.2f} "
            f"evidence_count={len(diagnosis.evidence)} "
            f"missing_information_count="
            f"{len(diagnosis.missing_information)} "
            f"prompt_tokens={response.usage.prompt_tokens} "
            f"completion_tokens={response.usage.completion_tokens} "
            f"total_tokens={response.usage.total_tokens}"
        )

        return diagnosis, False

    except LLMError as exc:
        logger.error(
            "Diagnosis generation failed; "
            f"using safe fallback "
            f"prompt_version={prompt_version} "
            f"model={model} "
            f"error={type(exc).__name__}: {exc}"
        )

        return _placeholder_diagnosis(), True

    except Exception as exc:
        logger.exception(
            "Unexpected diagnosis generation failure: " f"{type(exc).__name__}: {exc}"
        )

        return _placeholder_diagnosis(), True
