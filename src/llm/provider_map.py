"""
Per-role model assignments for the wuxing reasoning engine.

Cost-conscious stack: skips Opus 4.7 and GPT-5.5 entirely.
All slugs are OpenRouter-format (single key, all providers).

Role → model resolution happens in LLMClient._resolve_model() based on the
(domain, concept) tuple of each call. Engine code does NOT need to know
which model is used — the client picks based on these maps.

To experiment with different assignments, edit this file only.
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# OpenRouter base URL (one key, all models) — still used as the fallback
# path when a direct provider key isn't configured.
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# xAI's OpenAI-compatible API endpoint (Grok models).
XAI_BASE_URL = "https://api.x.ai/v1"


# ---------------------------------------------------------------------------
# Provider dispatch — which native SDK should handle a given model slug.
# Slugs are OpenRouter-format ("anthropic/claude-...", "google/gemini-...").
# When a direct provider key is configured at runtime, calls route to that
# provider's native SDK and skip OpenRouter's margin. Slugs without a
# direct key (currently "deepseek/*") fall back to OpenRouter.
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    ANTHROPIC  = "anthropic"
    OPENAI     = "openai"
    GOOGLE     = "google"
    XAI        = "xai"
    DEEPSEEK   = "deepseek"
    OPENROUTER = "openrouter"


def provider_of(model_slug: str) -> Provider:
    """
    Map a model slug to the provider whose native SDK can serve it.

    Examples:
        "anthropic/claude-sonnet-4-6"   -> Provider.ANTHROPIC
        "openai/gpt-5.4-nano"           -> Provider.OPENAI
        "google/gemini-2.5-flash"       -> Provider.GOOGLE
        "xai/grok-4"                    -> Provider.XAI
        "deepseek/deepseek-v4-pro"      -> Provider.DEEPSEEK   (no direct key — falls back to OpenRouter)

    Unknown prefixes fall back to OpenRouter so nothing breaks silently.
    """
    if "/" not in model_slug:
        return Provider.OPENROUTER
    prefix = model_slug.split("/", 1)[0].lower()
    try:
        return Provider(prefix)
    except ValueError:
        return Provider.OPENROUTER


def strip_provider_prefix(model_slug: str) -> str:
    """
    Strip the "<provider>/" prefix from a slug so it can be passed to a
    native SDK. Native SDKs expect just the model name
    ("claude-sonnet-4-6", "gemini-2.5-flash") without the org prefix.
    """
    if "/" in model_slug:
        return model_slug.split("/", 1)[1]
    return model_slug


# ---------------------------------------------------------------------------
# Default fallback (used when no per-role assignment matches)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Per-domain models (Sheng cycle — each domain runs in its own lane)
#
# 2026-05-28 reassignment: Sonnet 4.6 leaves the angle layer entirely. The
# angles are "laborious workers" — they dig one perspective deep, output
# structured findings, and never need cross-domain synthesis. Smart-model
# spend concentrates on the synthesizer (speech.py) where everything
# converges. Provider diversity preserved across three families (DeepSeek
# / Anthropic / Google) so the synthesizer sees genuinely different
# perspectives, not three takes from one training corpus.
# ---------------------------------------------------------------------------
DOMAIN_MODELS: dict[str, str] = {
    "physics":     "deepseek/deepseek-v4-pro",       # heavy multi-step reasoning over forces/conservation/trajectory
    "mathematics": "deepseek/deepseek-v4-pro",       # formal logic, Bayesian inference, long structured output
    "psychology":  "anthropic/claude-haiku-4-5",     # nuanced prose, identity-safe RLHF, dual-process clarity
    "philosophy":  "google/gemini-2.5-flash",        # long-context ontology + different training bias for diversity
    "chemistry":   "anthropic/claude-haiku-4-5",     # FIRST agent — needs fast + reliable structured JSON
}


# ---------------------------------------------------------------------------
# Per Ke-critic-pair models (cross-domain checking cycle)
# Each pair: (challenger, target) → critic model
#
# 2026-05-28 reassignment: every critic is Haiku 4.5. Ke critics produce
# short structured cross-checks ("did the target domain miss X?"), not
# long-form reasoning — Haiku's strength. Provider diversity is already
# carried by the Sheng (DOMAIN_MODELS) layer above.
# ---------------------------------------------------------------------------
KE_CRITIC_MODELS: dict[tuple[str, str], str] = {
    # Earth dams Water — does psychology survive material reality?
    ("physics", "psychology"):    "anthropic/claude-haiku-4-5",

    # Water extinguishes Fire — does the chemistry hold up under emotional reality?
    ("psychology", "chemistry"):  "anthropic/claude-haiku-4-5",

    # Fire melts Metal — does math survive the actual catalyst structure?
    ("chemistry", "mathematics"): "anthropic/claude-haiku-4-5",

    # Metal chops Wood — does philosophy survive formal scrutiny?
    ("mathematics", "philosophy"): "anthropic/claude-haiku-4-5",

    # Wood penetrates Earth — has physics questioned its own assumptions?
    ("philosophy", "physics"):    "anthropic/claude-haiku-4-5",
}


# ---------------------------------------------------------------------------
# Specialty roles
# ---------------------------------------------------------------------------
SYNTHESIZER_MODEL = "anthropic/claude-sonnet-4-6"   # final voice; citation grounding matters
GATING_MODEL      = "google/gemini-2.5-flash-lite"  # cheap, sub-second, 1M context for reading 5 outputs
ROUTER_MODEL      = "google/gemini-2.5-flash"       # chemistry self-assembly — fast structured JSON


# ---------------------------------------------------------------------------
# Per-model pricing (OpenRouter, May 2026, $ per 1M tokens)
# Format: model_slug -> (input_price, output_price)
# Used by LLMClient.get_total_cost_estimate() for accurate per-model cost tracking.
# ---------------------------------------------------------------------------
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic (cost-conscious tier — Opus 4.7 intentionally excluded)
    "anthropic/claude-sonnet-4-6":     (3.00, 15.00),
    "anthropic/claude-haiku-4-5":      (1.00,  5.00),

    # OpenAI (GPT-5.5 intentionally excluded — too expensive)
    "openai/gpt-5.4-nano":             (0.10,  0.40),

    # Google
    "google/gemini-2.5-pro":           (1.25, 10.00),
    "google/gemini-2.5-flash":         (0.30,  2.50),
    "google/gemini-2.5-flash-lite":    (0.10,  0.40),

    # DeepSeek
    "deepseek/deepseek-v4-pro":        (1.74,  3.48),   # post-promo (May 31, 2026)
    "deepseek/deepseek-v4-flash":      (0.14,  0.28),

    # Embedding models (output_cost = 0 — embeddings return vectors, not tokens)
    "openai/text-embedding-3-small":   (0.02,  0.00),
    "openai/text-embedding-3-large":   (0.13,  0.00),
    "google/text-embedding-004":       (0.025, 0.00),
}


# Default embedding model. Cheap, 1536-dim, good baseline. Override per call
# via EmbeddingScorer(model="openai/text-embedding-3-large") if you want
# higher dimensionality.
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


# Fallback pricing when a model slug isn't in PRICING (conservative Sonnet-tier estimate)
FALLBACK_PRICING: tuple[float, float] = (3.00, 15.00)


# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------

def resolve_model(domain: str, concept: str) -> str:
    """
    Resolve which model to use based on the call's (domain, concept) tuple.

    Resolution order:
    1. Ke critic pair: domain == "critic" + concept == "{challenger}_checks_{target}"
    2. Domain role: domain in DOMAIN_MODELS
    3. Specialty: domain == "synthesizer" / "gating" / "router"
    4. Default
    """
    # Critic role — concept encodes the pair (e.g. "physics_checks_psychology")
    if domain == "critic" and "_checks_" in concept:
        challenger, _, target = concept.partition("_checks_")
        return KE_CRITIC_MODELS.get((challenger, target), DEFAULT_MODEL)

    # Specialty roles
    if domain == "synthesizer":
        return SYNTHESIZER_MODEL
    if domain == "gating":
        return GATING_MODEL
    if domain == "router":
        return ROUTER_MODEL

    # Domain role (Sheng cycle)
    if domain in DOMAIN_MODELS:
        return DOMAIN_MODELS[domain]

    # Chemistry router — current code passes domain="chemistry", concept="self_assembly"
    # which already matches DOMAIN_MODELS["chemistry"]. No special handling needed.

    return DEFAULT_MODEL


def get_pricing(model_slug: str) -> tuple[float, float]:
    """Return (input_price, output_price) per 1M tokens for a model slug."""
    return PRICING.get(model_slug, FALLBACK_PRICING)
