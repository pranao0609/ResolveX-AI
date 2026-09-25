from __future__ import annotations

from typing import Tuple

import re
from dotenv import load_dotenv
from groq import Groq
from transformers import pipeline

from app.config import settings
from app.core.logger import logger


load_dotenv()


# ---------------------------------------------------------------------------
# Lazy zero-shot classifier
# ---------------------------------------------------------------------------

_zero_shot_classifier = None


def _get_zero_shot_classifier():
    """
    Lazy-load the BART zero-shot classification pipeline.

    Loading is deferred until classification is actually required so that
    importing this module does not immediately load the large model.
    """
    global _zero_shot_classifier

    if _zero_shot_classifier is None:
        try:
            _zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,
            )
        except Exception as exc:
            logger.error(
                "Failed to load zero-shot classifier pipeline: %s",
                exc,
            )
            _zero_shot_classifier = None

    return _zero_shot_classifier


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------


def _get_groq_client():
    """
    Create the Groq client when a valid API key is configured.
    """

    if (
        not settings.GROQ_API_KEY
        or settings.GROQ_API_KEY == "your-groq-api-key-here"
    ):
        return None

    try:
        return Groq(
            api_key=settings.GROQ_API_KEY,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.error(
            "Failed to initialize Groq client in classifier: %s",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Fine-grained internal categories
# ---------------------------------------------------------------------------

IT_CATEGORIES = [
    "server_error",
    "database_error",
    "network_error",
    "vpn_error",
    "hardware_error",
    "windows_error",
    "login_error",
    "email_error",
    "billing_error",
    "access_error",
    "performance_error",
    "security_error",
    "mouse_error",
    "printer_error",
    "keyboard_error",
]


# ---------------------------------------------------------------------------
# Final ResolveX category mapping
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    "server_error": "software",
    "database_error": "software",
    "windows_error": "software",
    "performance_error": "software",
    "email_error": "software",
    "network_error": "network",
    "vpn_error": "network",
    "hardware_error": "hardware",
    "mouse_error": "hardware",
    "printer_error": "hardware",
    "keyboard_error": "hardware",
    "login_error": "access_permission",
    "access_error": "access_permission",
    "security_error": "security",
    "billing_error": "other",
}


# ---------------------------------------------------------------------------
# Groq fallback classifier
# ---------------------------------------------------------------------------


def llm_classify_fallback(text: str) -> Tuple[str, float]:
    """
    Use the configured LLM as the final classification fallback.
    """

    client = _get_groq_client()

    if not client:
        return "software", 0.75

    try:
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""
Classify the IT support ticket into exactly ONE of these categories:

{', '.join(IT_CATEGORIES)}

Return ONLY in this exact format:

category,confidence

Where:
- category must be one of the listed categories
- confidence must be a float between 0.0 and 1.0

Example:
login_error,0.92
""".strip(),
                },
                {
                    "role": "user",
                    "content": text[:1000],
                },
            ],
            max_tokens=30,
            temperature=0.1,
        )

        response = completion.choices[0].message.content.strip()

        if "," in response:
            category, conf_str = response.split(",", 1)

            category = category.strip()

            try:
                confidence = float(conf_str.strip())
            except ValueError:
                logger.warning(
                    "Groq classifier returned invalid confidence: %s",
                    conf_str,
                )
                return "software", 0.80

            confidence = max(0.0, min(1.0, confidence))

            final_category = CATEGORY_MAP.get(
                category,
                "other",
            )

            logger.info(
                "Groq LLM fallback: %s -> %s (%.3f)",
                category,
                final_category,
                confidence,
            )

            return final_category, confidence

        return "software", 0.80

    except Exception as exc:
        logger.error(
            "Groq fallback failed: %s",
            exc,
        )

        return "software", 0.75


# ---------------------------------------------------------------------------
# Primary classification
# ---------------------------------------------------------------------------


def classify_ticket(text: str) -> Tuple[str, float]:
    """
    Primary ResolveX classification pipeline.

    Order:

    1. Input validation
    2. Regex/pattern classification
    3. BART zero-shot classification
    4. Groq LLM fallback
    """

    text = text.strip()

    if len(text) < 5:
        return "software", 0.50

    text_lower = text.lower()

    patterns = {
        r"\b(500|internal server|server error)\b": (
            "server_error",
            0.98,
        ),
        r"\b(database|postgres|mysql|sql|connection refused|db error|query failed)\b": (
            "database_error",
            0.97,
        ),
        r"\b(blue screen|bsod|windows.*error|win\d+.*error)\b": (
            "windows_error",
            0.97,
        ),
        r"\b(slow|performance|lag|hang|hanging|freezing|freeze|memory leak|high cpu|high ram)\b": (
            "performance_error",
            0.92,
        ),
        r"\b(email|outlook|mailbox|smtp|imap|email delivery|mail not working)\b": (
            "email_error",
            0.93,
        ),
        r"\b(vpn)\b": (
            "vpn_error",
            0.97,
        ),
        r"\b(network|wifi|wi-fi|internet|connection timeout|ping fail|latency|packet loss|dns)\b": (
            "network_error",
            0.96,
        ),
        r"\b(mouse|trackpad|click|cursor|pointer)\b.*?(broken|not working|dead|stopped|issue|problem)": (
            "mouse_error",
            0.97,
        ),
        r"\b(keyboard|keys?|typing)\b.*?(not working|broken|dead|stuck|issue|problem)": (
            "keyboard_error",
            0.96,
        ),
        r"\b(printer|print)\b.*?(not working|jam|offline|error|issue|problem)": (
            "printer_error",
            0.95,
        ),
        r"\b(monitor|screen|display|laptop|desktop|cpu|battery|charger|fan|motherboard|usb port)\b.*?(blank|black|flicker|dead|broken|damaged|not working|issue|problem)": (
            "hardware_error",
            0.95,
        ),
        r"\b(login|password|auth|authentication|access denied|permission denied|unauthorized|forbidden|reset password|password reset)\b": (
            "login_error",
            0.95,
        ),
        r"\b(role access|permission|privilege|grant access|revoke access|cannot access|no access)\b": (
            "access_error",
            0.94,
        ),
        r"\b(security|malware|virus|phishing|hacked|breach|suspicious login|ransomware|threat)\b": (
            "security_error",
            0.97,
        ),
        r"\b(billing|invoice|payment|refund|charge|subscription)\b": (
            "billing_error",
            0.94,
        ),
    }

    for pattern, (category, confidence) in patterns.items():
        if re.search(
            pattern,
            text_lower,
            re.IGNORECASE,
        ):
            final_category = CATEGORY_MAP.get(
                category,
                "other",
            )

            logger.info(
                "Pattern match: %s -> %s (%.3f)",
                category,
                final_category,
                confidence,
            )

            return final_category, confidence

    # -----------------------------------------------------------------------
    # BART zero-shot primary classification
    # -----------------------------------------------------------------------

    try:
        classifier = _get_zero_shot_classifier()

        if classifier is not None:
            result = classifier(
                text[:512],
                IT_CATEGORIES,
            )

            raw_category = result["labels"][0]
            raw_confidence = float(result["scores"][0])

            if raw_confidence > 0.35:
                final_category = CATEGORY_MAP.get(
                    raw_category,
                    "other",
                )

                calibrated = min(
                    0.95,
                    raw_confidence * 1.2 + 0.05,
                )

                logger.info(
                    "Zero-shot: %s -> %s (%.3f)",
                    raw_category,
                    final_category,
                    calibrated,
                )

                return final_category, float(calibrated)

    except Exception as exc:
        logger.warning(
            "Zero-shot classification failed: %s",
            exc,
        )

    # -----------------------------------------------------------------------
    # Groq fallback
    # -----------------------------------------------------------------------

    logger.info(
        "Using Groq LLM fallback for classification"
    )

    return llm_classify_fallback(text)


# ---------------------------------------------------------------------------
# Phase 15.4 — Controlled reclassification
# ---------------------------------------------------------------------------


def reclassify_with_zero_shot(
    text: str,
) -> Tuple[str, float]:
    """
    Perform an independent zero-shot reclassification.

    This function intentionally bypasses the complete primary
    classify_ticket() pipeline.

    Purpose:
        Provide a second classification strategy when the primary
        classification confidence is below the reclassification threshold.

    Returns:
        Tuple[str, float]:
            final ResolveX category
            calibrated confidence

    Raises:
        ValueError:
            If the input text is too short.

        RuntimeError:
            If the zero-shot model is unavailable or returns
            an invalid result.
    """

    if not isinstance(text, str):
        raise TypeError(
            "Reclassification input must be a string"
        )

    cleaned_text = text.strip()

    if len(cleaned_text) < 5:
        raise ValueError(
            "Reclassification input must contain at least 5 characters"
        )

    classifier = _get_zero_shot_classifier()

    if classifier is None:
        raise RuntimeError(
            "Zero-shot classifier is unavailable"
        )

    try:
        result = classifier(
            cleaned_text[:512],
            IT_CATEGORIES,
        )

        labels = result.get("labels")
        scores = result.get("scores")

        if not labels or not scores:
            raise RuntimeError(
                "Zero-shot classifier returned an invalid result"
            )

        raw_category = labels[0]
        raw_confidence = float(scores[0])

        if raw_category not in IT_CATEGORIES:
            raise RuntimeError(
                f"Invalid zero-shot category: {raw_category}"
            )

        if not 0.0 <= raw_confidence <= 1.0:
            raise RuntimeError(
                "Zero-shot confidence outside [0, 1]"
            )

        final_category = CATEGORY_MAP.get(
            raw_category,
            "other",
        )

        calibrated_confidence = min(
            0.95,
            raw_confidence * 1.2 + 0.05,
        )

        logger.info(
            "Zero-shot reclassification: "
            "%s -> %s (raw=%.3f calibrated=%.3f)",
            raw_category,
            final_category,
            raw_confidence,
            calibrated_confidence,
        )

        return (
            final_category,
            float(calibrated_confidence),
        )

    except RuntimeError:
        raise

    except Exception as exc:
        raise RuntimeError(
            f"Zero-shot reclassification failed: {exc}"
        ) from exc