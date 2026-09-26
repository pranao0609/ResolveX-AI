"""
ResolveX verification generation.

Evaluates whether a proposed resolution is sufficiently supported
by the ticket, diagnosis, root cause, and retrieved evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings
from app.core.logger import logger

from ai.config.ai_config import (
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
)

from ai.llm.errors import LLMError
from ai.llm.gateway import LLMGateway
from ai.llm.prompt_loader import load_prompt
from ai.llm.providers.groq_provider import GroqProvider

DEFAULT_VERIFICATION_PROMPT_VERSION = 1

# Minimum confidence required for verification to pass.
VERIFICATION_CONFIDENCE_THRESHOLD = 0.70


class VerificationResult(BaseModel):
    """
    Structured verification result.

    The detailed verification dimensions provide explicit
    verification signals, while the legacy fields remain available
    for backward compatibility with the existing graph and executor.
    """

    supported_by_evidence: bool = Field(
        ...,
        description=(
            "Whether the proposed resolution is directly supported "
            "by the retrieved evidence."
        ),
    )

    hallucination_detected: bool = Field(
        ...,
        description=(
            "Whether the proposed resolution contains unsupported "
            "or fabricated information."
        ),
    )

    complete: bool = Field(
        ...,
        description=(
            "Whether the proposed resolution adequately addresses "
            "the diagnosed problem."
        ),
    )

    policy_compliant: bool = Field(
        ...,
        description=(
            "Whether the proposed resolution follows applicable "
            "support and safety policies."
        ),
    )

    resolution_correct: bool = Field(
        ...,
        description=(
            "Whether the proposed resolution is technically "
            "consistent with the diagnosis, root cause, and evidence."
        ),
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=("Confidence in the verification assessment."),
    )

    # ------------------------------------------------------------------
    # Backward-compatible fields
    # ------------------------------------------------------------------

    verification_passed: bool = Field(
        default=False,
        description=(
            "Overall verification decision. This is derived "
            "deterministically from the detailed verification checks "
            "and confidence threshold."
        ),
    )

    verification_reason: str = Field(
        default="",
        description=("Explanation supporting the overall verification decision."),
    )

    def model_post_init(self, __context) -> None:
        """
        Derive the overall verification decision deterministically.

        A resolution can pass only when every required verification
        dimension passes and confidence is sufficiently high.
        """

        self.verification_passed = bool(
            self.supported_by_evidence
            and not self.hallucination_detected
            and self.complete
            and self.policy_compliant
            and self.resolution_correct
            and self.confidence >= VERIFICATION_CONFIDENCE_THRESHOLD
        )

        reasons: list[str] = []

        if not self.supported_by_evidence:
            reasons.append("Resolution is not sufficiently supported by evidence.")

        if self.hallucination_detected:
            reasons.append(
                "Potential hallucinated or unsupported information detected."
            )

        if not self.complete:
            reasons.append("Resolution is incomplete.")

        if not self.policy_compliant:
            reasons.append("Resolution does not satisfy policy requirements.")

        if not self.resolution_correct:
            reasons.append(
                "Resolution is not sufficiently consistent with "
                "the diagnosis, root cause, or evidence."
            )

        if self.confidence < VERIFICATION_CONFIDENCE_THRESHOLD:
            reasons.append("Verification confidence is below the required threshold.")

        if not reasons:
            reasons.append("Resolution passed all verification checks.")

        self.verification_reason = " ".join(reasons)


def _render_verification_prompt(
    prompt: dict,
    *,
    ticket_text: str,
    diagnosis: str,
    root_cause: str,
    resolution_steps: list[str],
    context: str,
) -> tuple[str, str]:
    """
    Render the verification prompt.

    We intentionally perform explicit replacement instead of
    str.format() so JSON braces inside the YAML remain untouched.
    """

    system_prompt = prompt["system_prompt"]

    user_prompt = prompt["user_prompt"]

    user_prompt = user_prompt.replace(
        "{ticket_text}",
        ticket_text,
    )

    user_prompt = user_prompt.replace(
        "{diagnosis}",
        diagnosis,
    )

    user_prompt = user_prompt.replace(
        "{root_cause}",
        root_cause,
    )

    user_prompt = user_prompt.replace(
        "{resolution_steps}",
        "\n".join(
            f"{index + 1}. {step}" for index, step in enumerate(resolution_steps)
        ),
    )

    user_prompt = user_prompt.replace(
        "{context}",
        context,
    )

    return system_prompt, user_prompt


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


def _placeholder_verification() -> VerificationResult:
    """
    Safe verification fallback.

    If verification cannot be performed, the proposed resolution
    must not automatically pass the verification gate.
    """

    return VerificationResult(
        supported_by_evidence=False,
        hallucination_detected=True,
        complete=False,
        policy_compliant=False,
        resolution_correct=False,
        confidence=0.0,
    )


def verify_resolution(
    *,
    ticket_text: str,
    diagnosis: str,
    root_cause: str,
    resolution_steps: list[str],
    context: str,
    prompt_version: int = DEFAULT_VERIFICATION_PROMPT_VERSION,
) -> tuple[VerificationResult, bool]:
    """
    Verify a proposed resolution through the LLM Gateway.

    Returns:
        (VerificationResult, fallback_used)
    """

    prompt = load_prompt(
        "verification",
        prompt_version,
    )

    clean_ticket = ticket_text[:4000] if ticket_text else ""

    clean_diagnosis = diagnosis.strip() if diagnosis else "No diagnosis available."

    clean_root_cause = root_cause.strip() if root_cause else "No root cause available."

    clean_context = context.strip() if context else "No context available."

    clean_steps = [str(step).strip() for step in resolution_steps if str(step).strip()]

    system_prompt, user_prompt = _render_verification_prompt(
        prompt,
        ticket_text=clean_ticket,
        diagnosis=clean_diagnosis,
        root_cause=clean_root_cause,
        resolution_steps=clean_steps,
        context=clean_context,
    )

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

    try:
        gateway = _get_gateway()

    except ValueError as exc:
        logger.warning("Verification LLM provider unavailable: " f"{exc}")

        return _placeholder_verification(), True

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

        verification = VerificationResult.model_validate_json(response.content)

        logger.info(
            "Structured resolution verification generated "
            f"prompt_version={prompt_version} "
            f"model={model} "
            f"passed={verification.verification_passed} "
            f"evidence={verification.supported_by_evidence} "
            f"hallucination={verification.hallucination_detected} "
            f"complete={verification.complete} "
            f"policy={verification.policy_compliant} "
            f"correct={verification.resolution_correct} "
            f"confidence={verification.confidence:.2f} "
            f"prompt_tokens={response.usage.prompt_tokens} "
            f"completion_tokens={response.usage.completion_tokens} "
            f"total_tokens={response.usage.total_tokens}"
        )

        return verification, False

    except LLMError as exc:
        logger.error(
            "Resolution verification failed; "
            "using safe verification fallback "
            f"prompt_version={prompt_version} "
            f"model={model} "
            f"error={type(exc).__name__}: {exc}"
        )

        return _placeholder_verification(), True

    except Exception as exc:
        logger.exception(
            "Unexpected verification failure: " f"{type(exc).__name__}: {exc}"
        )

        return _placeholder_verification(), True
