"""
LLM Client — The River.

Single async connection to OpenRouter (one key → all providers).
All domain agents call through this wrapper.

Per-call model resolution: each call's (domain, concept) tuple is mapped to a
model slug via src/llm/provider_map.py. Engine code does NOT pass model names —
it passes domain + concept, and the client picks. To change models, edit
provider_map.py.

Two modes:
- LIVE: Real OpenRouter API calls (requires OPENROUTER_API_KEY env var)
- MOCK: Deterministic responses for architecture testing (no API credits spent)

Every call logs: domain, concept, model, token count, latency, success/failure.

ISOLATION: This module does NOT import from any domain module.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Observability logger — every LLM call emits one INFO line tagged
# CALL {...}, and every per-request summary emits one INFO multi-line
# block. Filter with `grep -E "CALL |LLM SUMMARY"` for raw analysis.
_obs_log = logging.getLogger("constellax.obs")

from src.llm.provider_map import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    OPENROUTER_BASE_URL,
    XAI_BASE_URL,
    Provider,
    get_pricing,
    provider_of,
    resolve_model,
    strip_provider_prefix,
)


class ClientMode(Enum):
    """Operating mode of the LLM client."""
    LIVE = "live"       # real API calls (via OpenRouter)
    MOCK = "mock"       # deterministic responses for testing


@dataclass
class LLMCallLog:
    """Log entry for a single LLM call."""
    domain: str
    concept: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error: str | None = None
    timestamp: float = 0.0


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    model: str = ""           # which model actually answered this call
    error: str | None = None
    raw: Any = None           # raw API response object (live mode only)


class LLMClient:
    """
    The River — single OpenRouter connection for all domain agents.

    Usage:
        client = LLMClient(mode=ClientMode.MOCK)   # for testing
        response = await client.call(
            system_prompt="You are LoRa's Physics agent...",
            user_message="Analyze this problem...",
            domain="physics",
            concept="first_principles",
        )
        # Model is resolved automatically from (domain, concept) via provider_map.

    Per-call override (optional):
        response = await client.call(..., model="anthropic/claude-haiku-4-5")
    """

    # Default fallback model. Actual per-call model comes from provider_map.resolve_model().
    DEFAULT_MODEL = DEFAULT_MODEL

    # Safety limits.
    #
    # TIMEOUT_SECONDS was 60. Observability logs proved Sonnet on big
    # domain prompts routinely hits 60-90s wall time — the old value
    # triggered a retry on every slow call, so we were paying 2x for
    # those calls AND blowing past the 12-min request cap. 150s gives
    # one clean attempt without retry-doubling. MAX_RETRIES dropped to 1
    # for the same reason: a single 60s false-fail used to multiply.
    MAX_RETRIES = 1
    TIMEOUT_SECONDS = 150
    MAX_OUTPUT_TOKENS = 4096

    # OpenRouter optional headers (helps with analytics + rate limit prioritization)
    APP_REFERER = "https://github.com/nikhiljangra/reasoningEngine"
    APP_TITLE = "Constellax Reasoning Engine"

    def __init__(
        self,
        mode: ClientMode = ClientMode.MOCK,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.mode = mode
        self.model = model or self.DEFAULT_MODEL    # fallback when resolve_model returns nothing useful
        self.call_log: list[LLMCallLog] = []
        # Per-provider native client cache. Keys = Provider enum members.
        # When a model slug's native provider isn't in this dict, the call
        # falls back to the OpenRouter client (which sits at Provider.OPENROUTER
        # AND Provider.DEEPSEEK because DeepSeek has no direct key).
        self._provider_clients: dict[Provider, Any] = {}
        # Backward-compat alias — some older code paths read self._client
        # directly. Points at the OpenRouter client when configured.
        self._client = None

        if mode != ClientMode.LIVE:
            return

        # Lazy import — only needed in live mode.
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required for live mode. "
                "Install with: pip install 'openai>=1.50,<2.0'"
            )

        # --------------------------------------------------------------
        # Per-provider native clients — initialized only if their key is
        # set. Missing keys are NOT an error; calls to those providers
        # transparently fall back to OpenRouter (if it's configured).
        # This is the cost-savings path: each direct key bypasses the
        # OpenRouter margin (~5%) on every call to that provider.
        # --------------------------------------------------------------

        # Anthropic direct (Claude Sonnet 4-6, Haiku 4-5)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        # Heuristic: an OpenRouter key starts with "sk-or-". If
        # ANTHROPIC_API_KEY is set to an OpenRouter key by mistake
        # (a leftover from the old single-key setup), skip the Anthropic
        # direct path and let it fall through to OpenRouter.
        if anthropic_key and not anthropic_key.startswith("sk-or-"):
            try:
                from anthropic import AsyncAnthropic
                self._provider_clients[Provider.ANTHROPIC] = AsyncAnthropic(
                    api_key=anthropic_key,
                )
            except ImportError:
                pass  # anthropic SDK not installed — silently fall back

        # OpenAI direct (also used for openai/text-embedding-3-* models)
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if openai_key:
            self._provider_clients[Provider.OPENAI] = AsyncOpenAI(api_key=openai_key)

        # Google Gemini direct — try GEMINI_API_KEY first, then GOOGLE_API_KEY
        google_key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        if google_key:
            try:
                from google import genai as google_genai
                self._provider_clients[Provider.GOOGLE] = google_genai.Client(
                    api_key=google_key,
                )
            except ImportError:
                pass  # google-genai SDK not installed

        # xAI (Grok) — OpenAI-compatible API at a different base URL
        xai_key = os.environ.get("XAI_API_KEY", "").strip()
        if xai_key:
            self._provider_clients[Provider.XAI] = AsyncOpenAI(
                api_key=xai_key,
                base_url=XAI_BASE_URL,
            )

        # --------------------------------------------------------------
        # OpenRouter fallback — required for DeepSeek (no direct key)
        # and any unknown-provider slug. The legacy single-key setup
        # used OPENROUTER_API_KEY; the explicit api_key arg still wins.
        # --------------------------------------------------------------
        openrouter_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
        )
        if openrouter_key:
            openrouter_client = AsyncOpenAI(
                api_key=openrouter_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": self.APP_REFERER,
                    "X-Title": self.APP_TITLE,
                },
            )
            self._provider_clients[Provider.OPENROUTER] = openrouter_client
            self._provider_clients[Provider.DEEPSEEK]   = openrouter_client
            self._client = openrouter_client  # back-compat alias

        # --------------------------------------------------------------
        # Sanity check — refuse to init with zero providers configured.
        # If OpenRouter is missing, warn about DeepSeek calls failing.
        # --------------------------------------------------------------
        if not self._provider_clients:
            raise ValueError(
                "No provider API keys configured. Set at least one of: "
                "OPENROUTER_API_KEY (recommended fallback), "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "GEMINI_API_KEY/GOOGLE_API_KEY, XAI_API_KEY."
            )

        if Provider.OPENROUTER not in self._provider_clients:
            _obs_log.warning(
                "OPENROUTER_API_KEY not set — DeepSeek calls (mathematics "
                "lane + psychology→chemistry critic) will fail. Set the key "
                "OR rebind those roles to a direct-provider model in provider_map.py."
            )

        routes = sorted(p.value for p in self._provider_clients
                        if p != Provider.DEEPSEEK)  # dedupe (DEEPSEEK shares openrouter)
        _obs_log.info(f"LLMClient routing initialized: {', '.join(routes)}")

    async def call(
        self,
        system_prompt: str,
        user_message: str,
        domain: str,
        concept: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        model: str | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        """
        Make a single LLM call.

        Model resolution order:
        1. Explicit `model` param (per-call override)
        2. provider_map.resolve_model(domain, concept)
        3. self.model (instance default)

        Every domain agent calls this. The (domain, concept) tuple drives both
        the system prompt selection (handled by the caller) and the model
        selection (handled here).

        `effort` controls the thinking budget for adaptive-thinking models
        (Fable 5+ require this; older models ignore it). Valid values:
        "low" | "medium" | "high". When None, the model uses its default
        (which for Fable 5 is "adaptive" — may exhaust max_tokens on
        thinking and emit no visible TextBlock for complex structured-
        output prompts). For sort / classification work, prefer "low"
        or "medium" to keep budget for visible output.
        """
        max_tokens = max_tokens or self.MAX_OUTPUT_TOKENS
        chosen_model = model or resolve_model(domain, concept) or self.model

        start_time = time.monotonic()

        for attempt in range(self.MAX_RETRIES):
            try:
                if self.mode == ClientMode.MOCK:
                    response = await self._mock_call(
                        system_prompt, user_message, domain, concept
                    )
                else:
                    response = await self._live_call(
                        system_prompt, user_message, max_tokens, temperature, chosen_model, effort,
                    )

                elapsed_ms = (time.monotonic() - start_time) * 1000
                response.latency_ms = elapsed_ms
                response.model = chosen_model

                self._log_call(
                    domain=domain,
                    concept=concept,
                    model=chosen_model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=elapsed_ms,
                    success=True,
                )
                return response

            except Exception as e:
                elapsed_ms = (time.monotonic() - start_time) * 1000

                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(1.0)
                    continue

                # All retries exhausted
                self._log_call(
                    domain=domain,
                    concept=concept,
                    model=chosen_model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=elapsed_ms,
                    success=False,
                    error=str(e),
                )
                return LLMResponse(
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=elapsed_ms,
                    success=False,
                    model=chosen_model,
                    error=str(e),
                )

    async def call_batch(
        self,
        calls: list[dict],
    ) -> list[LLMResponse]:
        """
        Fan-out: launch multiple LLM calls in parallel.

        Each call dict has: system_prompt, user_message, domain, concept.
        Optional: max_tokens, temperature, model (per-call override).

        All calls run simultaneously via asyncio.gather().
        Returns responses in the same order as the input calls.

        Different calls in the same batch can target different models — the
        per-call (domain, concept) tuple resolves to the right one automatically.
        """
        tasks = [
            self.call(
                system_prompt=c["system_prompt"],
                user_message=c["user_message"],
                domain=c["domain"],
                concept=c["concept"],
                max_tokens=c.get("max_tokens"),
                temperature=c.get("temperature", 0.7),
                model=c.get("model"),
            )
            for c in calls
        ]
        return await asyncio.gather(*tasks)

    # -----------------------------------------------------------------------
    # Live mode — dispatch to whichever provider can actually serve the call
    # -----------------------------------------------------------------------

    def _resolve_actual_provider(self, model: str) -> Provider:
        """
        Pick which provider's client will actually serve a call for `model`.

        Rules:
        - DEEPSEEK always routes through OpenRouter (no direct key exists).
        - If a direct client is configured for the slug's provider, use it.
        - Otherwise fall back to OPENROUTER.
        """
        slug_provider = provider_of(model)
        if slug_provider == Provider.DEEPSEEK:
            return Provider.OPENROUTER
        if slug_provider in self._provider_clients:
            return slug_provider
        return Provider.OPENROUTER

    async def _live_call(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        model: str,
        effort: str | None = None,
    ) -> LLMResponse:
        """
        Make a real API call, routed to the provider's native SDK when
        a direct key is set. Falls back to OpenRouter for DeepSeek and any
        provider whose direct client isn't configured.

        `effort` only flows to providers whose adaptive-thinking API uses
        it (Anthropic Fable 5+). Other providers ignore it.
        """
        actual = self._resolve_actual_provider(model)
        client = self._provider_clients.get(actual)
        if client is None:
            raise RuntimeError(
                f"No client available for {model}. Configure OPENROUTER_API_KEY "
                f"as a fallback, or set the direct provider key for "
                f"{provider_of(model).value}."
            )

        if actual == Provider.ANTHROPIC:
            return await self._call_via_anthropic(
                client, system_prompt, user_message, max_tokens, temperature, model,
                effort=effort,
            )
        if actual == Provider.GOOGLE:
            return await self._call_via_gemini(
                client, system_prompt, user_message, max_tokens, temperature, model,
            )
        if actual == Provider.OPENAI:
            # OpenAI's newer models (gpt-5.x, o1/o3) require
            # max_completion_tokens; the old max_tokens param is rejected.
            return await self._call_via_openai_compat(
                client, system_prompt, user_message, max_tokens, temperature,
                model, strip_prefix=True, use_max_completion_tokens=True,
            )
        if actual == Provider.XAI:
            # xAI still uses the classic max_tokens param.
            return await self._call_via_openai_compat(
                client, system_prompt, user_message, max_tokens, temperature,
                model, strip_prefix=True, use_max_completion_tokens=False,
            )
        # OPENROUTER (and DEEPSEEK, which shares the OpenRouter client).
        # OpenRouter wants the FULL slug including provider prefix and
        # accepts the classic max_tokens.
        return await self._call_via_openai_compat(
            client, system_prompt, user_message, max_tokens, temperature,
            model, strip_prefix=False, use_max_completion_tokens=False,
        )

    # ─── per-provider call helpers ──────────────────────────────────────────

    async def _call_via_openai_compat(
        self,
        client,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        model: str,
        strip_prefix: bool,
        use_max_completion_tokens: bool = False,
    ) -> LLMResponse:
        """
        Call any OpenAI-protocol endpoint — OpenAI direct, xAI, or OpenRouter.

        `strip_prefix=True` for direct providers (they want bare model names),
        `False` for OpenRouter (it wants the slug including provider org).

        `use_max_completion_tokens=True` for OpenAI direct's newer models —
        they reject the legacy `max_tokens` param. xAI + OpenRouter keep the
        classic `max_tokens`.
        """
        api_model = strip_provider_prefix(model) if strip_prefix else model
        create_kwargs: dict = {
            "model":       api_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        }
        if use_max_completion_tokens:
            create_kwargs["max_completion_tokens"] = max_tokens
        else:
            create_kwargs["max_tokens"] = max_tokens

        response = await asyncio.wait_for(
            client.chat.completions.create(**create_kwargs),
            timeout=self.TIMEOUT_SECONDS,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=content,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=0,
            success=True,
            model=model,    # keep the original slug for accounting
            raw=response,
        )

    #: Anthropic models that have deprecated the `temperature` request
    #: parameter. Calls to these slugs MUST omit the temperature kwarg or
    #: the API returns HTTP 400 invalid_request_error. Discovered 2026-06-12
    #: via Fable 5 ping. Add new slugs here as Anthropic ships more models
    #: that drop the param.
    _ANTHROPIC_NO_TEMPERATURE: tuple[str, ...] = (
        "claude-fable-5",
        # Opus 4.8 also dropped temperature — Anthropic is standardizing
        # on adaptive defaults for new models. Verified 2026-06-12 via
        # 400 invalid_request_error probe.
        "claude-opus-4-8",
    )

    #: Anthropic models that use the NEW adaptive-thinking API where
    #: `output_config.effort` controls thinking depth. For these models
    #: thinking defaults to "adaptive" — which on complex structured-
    #: output prompts can exhaust max_tokens on hidden thinking and emit
    #: zero visible TextBlock content. Discovered 2026-06-12 when the
    #: first live sort returned 3296 output_tokens with empty content.
    #: Setting effort="low" or "medium" caps thinking budget so visible
    #: JSON gets emitted. Older Anthropic models (Sonnet/Opus 4.6) do
    #: not accept output_config and the param is dropped for them.
    _ANTHROPIC_USES_OUTPUT_CONFIG: tuple[str, ...] = (
        "claude-fable-5",
        # Opus 4.8 accepts output_config.effort cleanly. Its default
        # behavior emits TextBlock only (no ThinkingBlock by default,
        # unlike Fable 5), but the effort kwarg is still useful for
        # explicit budget control on long structured outputs.
        "claude-opus-4-8",
    )

    async def _call_via_anthropic(
        self,
        client,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        model: str,
        effort: str | None = None,
    ) -> LLMResponse:
        """Call Anthropic's native /v1/messages API.

        `effort` is forwarded as output_config.effort for adaptive-thinking
        models (Fable 5+). For older models it is ignored.
        """
        native_model = strip_provider_prefix(model)
        kwargs = {
            "model":      native_model,
            "max_tokens": max_tokens,
            "system":     system_prompt,
            "messages":   [{"role": "user", "content": user_message}],
        }
        # Newer Anthropic models (Fable 5+) reject temperature. Older
        # models still accept it; the default at the API server is fine
        # for our use, so dropping it for incompatible models doesn't
        # change behavior for them.
        if native_model not in self._ANTHROPIC_NO_TEMPERATURE:
            kwargs["temperature"] = temperature
        # Forward effort as output_config.effort for adaptive-thinking
        # models. Without this, Fable 5 defaults to "adaptive" and
        # complex prompts emit ThinkingBlock-only responses with no
        # visible TextBlock.
        if effort and native_model in self._ANTHROPIC_USES_OUTPUT_CONFIG:
            kwargs["output_config"] = {"effort": effort}
        response = await asyncio.wait_for(
            client.messages.create(**kwargs),
            timeout=self.TIMEOUT_SECONDS,
        )
        # Anthropic returns content as a list of TextBlock objects.
        # ThinkingBlock objects also appear for adaptive-thinking models
        # but they have `.thinking` instead of `.text` and are correctly
        # skipped here.
        content = "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )
        usage = response.usage
        return LLMResponse(
            content=content,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            latency_ms=0,
            success=True,
            model=model,
            raw=response,
        )

    async def _call_via_gemini(
        self,
        client,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        model: str,
    ) -> LLMResponse:
        """Call Google Gemini via the google-genai SDK (async)."""
        native_model = strip_provider_prefix(model)
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=native_model,
                contents=user_message,
                config={
                    "system_instruction": system_prompt,
                    "temperature":        temperature,
                    "max_output_tokens":  max_tokens,
                },
            ),
            timeout=self.TIMEOUT_SECONDS,
        )
        content = (response.text or "")
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            content=content,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            latency_ms=0,
            success=True,
            model=model,
            raw=response,
        )

    # -----------------------------------------------------------------------
    # Mock mode (unchanged — deterministic responses for architecture testing)
    # -----------------------------------------------------------------------

    async def _mock_call(
        self,
        system_prompt: str,
        user_message: str,
        domain: str,
        concept: str,
    ) -> LLMResponse:
        """
        Generate a deterministic mock response for architecture testing.

        Simulates realistic response structure without API calls. Each
        domain/concept gets a structured response the parser can process,
        allowing the full fan-out/fan-in/convergence architecture to be
        tested without spending credits.
        """
        # Simulate realistic latency (50-200ms for mock)
        await asyncio.sleep(0.05 + hash(domain + concept) % 150 / 1000)

        mock_content = self._generate_mock_response(domain, concept, user_message)

        # Rough token estimates
        input_tokens = len(system_prompt.split()) + len(user_message.split())
        output_tokens = len(mock_content.split())

        return LLMResponse(
            content=mock_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0,
            success=True,
        )

    def _generate_mock_response(
        self, domain: str, concept: str, user_message: str
    ) -> str:
        """Generate a structured mock response for a given domain/concept."""

        words = user_message.lower().split()
        key_topics = [w for w in words if len(w) > 5][:5]
        topic_str = ", ".join(key_topics) if key_topics else "the situation"

        mock_responses = {
            "physics": {
                "default": json.dumps({
                    "findings": [
                        {
                            "type": "ROOT_CAUSE",
                            "name": f"force_analysis_{concept}",
                            "description": f"Physics {concept} analysis of {topic_str}: causal forces identified.",
                            "magnitude": 0.75,
                            "direction": "negative",
                            "confidence": 0.8,
                            "evidence": [f"Decomposed from {concept} framework", f"Key factors: {topic_str}"],
                            "label": "CONTRIBUTING_FACTOR",
                        }
                    ],
                    "assumptions": [f"Assuming {concept} applies to this problem domain"],
                    "anomalies": [],
                }),
            },
            "mathematics": {
                "default": json.dumps({
                    "findings": [
                        {
                            "type": "PATTERN",
                            "name": f"structural_pattern_{concept}",
                            "description": f"Mathematical {concept} pattern in {topic_str}.",
                            "magnitude": 0.7,
                            "direction": "neutral",
                            "confidence": 0.75,
                            "evidence": [f"Detected via {concept}", f"Variables: {topic_str}"],
                            "label": "VERIFIED",
                        }
                    ],
                    "convergence_status": "not_converged",
                    "dimensional_reduction": {"original": 10, "reduced": 4},
                }),
            },
            "psychology": {
                "default": json.dumps({
                    "findings": [
                        {
                            "type": "BIAS_DETECTION",
                            "name": f"psychological_{concept}",
                            "description": f"Psychology {concept} detected in {topic_str}.",
                            "magnitude": 0.65,
                            "direction": "negative",
                            "confidence": 0.7,
                            "evidence": [f"Detected via {concept}", "Language pattern analysis"],
                            "label": "INFERRED",
                            "system_classification": "S1",
                        }
                    ],
                    "metacognition_score": 0.45,
                    "delivery_mode": "building",
                }),
            },
            "philosophy": {
                "default": json.dumps({
                    "findings": [
                        {
                            "type": "EPISTEMIC",
                            "name": f"philosophical_{concept}",
                            "description": f"Philosophy {concept} analysis of {topic_str}.",
                            "magnitude": 0.7,
                            "direction": "neutral",
                            "confidence": 0.65,
                            "evidence": [f"Examined via {concept}"],
                            "label": "BELIEF",
                            "classification": "assumption",
                        }
                    ],
                    "ontological_core": f"The essential nature of this problem is about {topic_str}",
                    "hidden_utility": "identity_preservation",
                }),
            },
            "chemistry": {
                "default": json.dumps({
                    "findings": [
                        {
                            "type": "GOVERNANCE",
                            "name": f"chemical_{concept}",
                            "description": f"Chemistry {concept} analysis of {topic_str}.",
                            "magnitude": 0.6,
                            "direction": "neutral",
                            "confidence": 0.7,
                            "evidence": [f"Assessed via {concept}"],
                        }
                    ],
                    "formation_plan": {
                        "active_domains": ["physics", "psychology", "philosophy"],
                        "estimated_agents": 10,
                        "complexity": "medium",
                    },
                }),
            },
            "critic": {
                "default": json.dumps({
                    "scrutiny_score": 0.35,
                    "contradictions": [],
                    "unsupported_claims": [f"Claim about {topic_str} lacks sufficient evidence chain"],
                    "flags": [f"High confidence without multi-framework agreement on {topic_str}"],
                    "confidence_adjustments": {},
                }),
            },
        }

        domain_responses = mock_responses.get(domain, mock_responses["physics"])
        return domain_responses.get(concept, domain_responses["default"])

    # -----------------------------------------------------------------------
    # Embeddings (semantic similarity backbone)
    # -----------------------------------------------------------------------

    async def embed(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """
        Get an embedding vector for text via OpenRouter (live) or a
        deterministic mock (test mode).

        Live mode uses the standard OpenAI /v1/embeddings endpoint (which
        OpenRouter mirrors). Default model is openai/text-embedding-3-small
        ($0.02 per 1M input tokens, 1536 dimensions).

        Mock mode returns a hashing-trick pseudo-embedding: identical text
        always yields identical vectors; texts sharing tokens have vectors
        that align in cosine space. NOT semantically meaningful but
        sufficient for testing the scorer plumbing without API calls.

        Every call is logged into self.call_log under domain="embedding"
        with output_tokens=0 (embeddings have no output cost). The budget
        tracker picks them up via get_pricing() on the embedding model.
        """
        chosen_model = model or DEFAULT_EMBEDDING_MODEL
        start_time = time.monotonic()

        for attempt in range(self.MAX_RETRIES):
            try:
                if self.mode == ClientMode.MOCK:
                    vec = _mock_embedding(text)
                    # Rough token estimate for mock cost accounting
                    input_tokens = max(1, len(text.split()))
                else:
                    vec, input_tokens = await self._live_embed(text, chosen_model)

                elapsed_ms = (time.monotonic() - start_time) * 1000
                self._log_call(
                    domain="embedding",
                    concept="text",
                    model=chosen_model,
                    input_tokens=input_tokens,
                    output_tokens=0,    # embeddings produce vectors, not tokens
                    latency_ms=elapsed_ms,
                    success=True,
                )
                return vec

            except Exception as e:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(1.0)
                    continue
                self._log_call(
                    domain="embedding",
                    concept="text",
                    model=chosen_model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=elapsed_ms,
                    success=False,
                    error=str(e),
                )
                # Embedding failures are propagated — callers (the
                # EmbeddingScorer) decide how to degrade. The default
                # MemoryAdapter behavior is to fall back to keyword
                # similarity only if a fallback scorer is configured.
                raise

    async def _live_embed(self, text: str, model: str) -> tuple[list[float], int]:
        """
        Make a real embedding call, routed to a direct provider when the
        key is set. Returns (vector, input_tokens).

        Default embedding model is openai/text-embedding-3-small — so when
        OPENAI_API_KEY is configured, this hits OpenAI directly and skips
        the OpenRouter margin entirely.
        """
        slug_provider = provider_of(model)

        # Direct OpenAI embeddings — uses the standard /v1/embeddings endpoint
        if slug_provider == Provider.OPENAI and Provider.OPENAI in self._provider_clients:
            native_model = strip_provider_prefix(model)
            response = await asyncio.wait_for(
                self._provider_clients[Provider.OPENAI].embeddings.create(
                    model=native_model,
                    input=text,
                ),
                timeout=self.TIMEOUT_SECONDS,
            )
            vec = list(response.data[0].embedding)
            usage = getattr(response, "usage", None)
            return vec, (getattr(usage, "prompt_tokens", 0) if usage else 0)

        # Direct Google Gemini embeddings (text-embedding-004 and family)
        if slug_provider == Provider.GOOGLE and Provider.GOOGLE in self._provider_clients:
            native_model = strip_provider_prefix(model)
            response = await asyncio.wait_for(
                self._provider_clients[Provider.GOOGLE].aio.models.embed_content(
                    model=native_model,
                    contents=text,
                ),
                timeout=self.TIMEOUT_SECONDS,
            )
            vec = list(response.embeddings[0].values)
            # Gemini doesn't return token usage on embeddings; approximate.
            return vec, max(1, len(text.split()))

        # Fall back to OpenRouter (full slug, OpenAI-compatible endpoint)
        openrouter = self._provider_clients.get(Provider.OPENROUTER)
        if openrouter is None:
            raise RuntimeError(
                f"No client available for embedding model {model}. "
                f"Configure OPENROUTER_API_KEY (fallback) or the direct key "
                f"for provider {slug_provider.value}."
            )
        response = await asyncio.wait_for(
            openrouter.embeddings.create(model=model, input=text),
            timeout=self.TIMEOUT_SECONDS,
        )
        vec = list(response.data[0].embedding)
        usage = getattr(response, "usage", None)
        return vec, (getattr(usage, "prompt_tokens", 0) if usage else 0)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log_call(
        self,
        domain: str,
        concept: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Log a call for monitoring and credit calculation."""
        log = LLMCallLog(
            domain=domain,
            concept=concept,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error,
            timestamp=time.time(),
        )
        self.call_log.append(log)

        # Structured per-call line — easy to grep / parse from the
        # server log. One JSON object per line. The `provider` field
        # surfaces which key was actually billed: "anthropic" / "openai" /
        # "google" / "xai" mean a direct key, "openrouter" means OpenRouter
        # was used (either by intent, e.g. deepseek, or as fallback when
        # the direct key was missing).
        try:
            in_price, out_price = get_pricing(model)
            cost_usd = (
                (input_tokens  * in_price  / 1_000_000)
                + (output_tokens * out_price / 1_000_000)
            )
        except Exception:
            cost_usd = 0.0

        # Resolve which provider key actually paid for this call.
        if self.mode == ClientMode.MOCK:
            actual_provider = "mock"
        else:
            try:
                actual_provider = self._resolve_actual_provider(model).value
            except Exception:
                actual_provider = "unknown"

        _obs_log.info(
            "CALL %s",
            json.dumps({
                "role":     f"{domain}/{concept}",
                "model":    model,
                "provider": actual_provider,
                "in_tok":   input_tokens,
                "out_tok":  output_tokens,
                "ms":       round(latency_ms),
                "cost_usd": round(cost_usd, 6),
                "ok":       success,
                **({"err": error} if error else {}),
            }),
        )

    # -----------------------------------------------------------------------
    # Observability — per-request summary
    # -----------------------------------------------------------------------

    def summarize_calls(self, request_id: str | None = None) -> dict:
        """
        Build a per-request breakdown of every call in self.call_log.

        Returns:
          {
            "total_calls":  int,
            "total_ms":     int (wall-time sum of all calls — actual wall
                            time is smaller because of parallel fan-out),
            "total_cost":   float,
            "by_model": [
              { "model": "...", "calls": N, "total_ms": int,
                "avg_ms": int, "cost_usd": float },
              ...                              # sorted by total_ms desc
            ],
            "slowest": [
              { "role": "...", "model": "...", "ms": int },
              ...                              # top 5 longest single calls
            ],
          }

        Also emits a multi-line SUMMARY log block so the breakdown is
        visible without parsing JSON.
        """
        if not self.call_log:
            return {"total_calls": 0, "total_ms": 0, "total_cost": 0.0,
                    "by_model": [], "slowest": []}

        # Aggregate by model AND by provider (the actual billed key).
        # Provider rollup is the cost-routing audit: if a slug should be
        # going direct but the row shows "openrouter", a key is missing.
        by_model:    dict[str, dict] = {}
        by_provider: dict[str, dict] = {}
        for entry in self.call_log:
            row = by_model.setdefault(entry.model or "(unknown)", {
                "calls": 0, "total_ms": 0.0, "in_tok": 0, "out_tok": 0,
            })
            row["calls"]    += 1
            row["total_ms"] += entry.latency_ms
            row["in_tok"]   += entry.input_tokens
            row["out_tok"]  += entry.output_tokens

            # Provider rollup
            try:
                prov = self._resolve_actual_provider(entry.model).value
            except Exception:
                prov = "unknown"
            prow = by_provider.setdefault(prov, {
                "calls": 0, "in_tok": 0, "out_tok": 0, "cost_usd": 0.0,
            })
            prow["calls"]   += 1
            prow["in_tok"]  += entry.input_tokens
            prow["out_tok"] += entry.output_tokens
            in_price, out_price = get_pricing(entry.model)
            prow["cost_usd"] += (
                (entry.input_tokens  * in_price  / 1_000_000)
                + (entry.output_tokens * out_price / 1_000_000)
            )

        # Compute per-model cost
        by_model_list = []
        total_cost = 0.0
        for model, row in by_model.items():
            in_price, out_price = get_pricing(model)
            cost = (
                (row["in_tok"]  * in_price  / 1_000_000)
                + (row["out_tok"] * out_price / 1_000_000)
            )
            total_cost += cost
            by_model_list.append({
                "model":    model,
                "calls":    row["calls"],
                "total_ms": int(row["total_ms"]),
                "avg_ms":   int(row["total_ms"] / max(1, row["calls"])),
                "in_tok":   row["in_tok"],
                "out_tok":  row["out_tok"],
                "cost_usd": round(cost, 6),
            })
        by_model_list.sort(key=lambda r: r["total_ms"], reverse=True)

        # Top 5 slowest single calls
        slowest = sorted(self.call_log, key=lambda e: e.latency_ms, reverse=True)[:5]
        slowest_list = [
            {
                "role":  f"{e.domain}/{e.concept}",
                "model": e.model,
                "ms":    int(e.latency_ms),
            }
            for e in slowest
        ]

        total_calls = len(self.call_log)
        total_ms = sum(e.latency_ms for e in self.call_log)

        # Sort provider rollup by cost desc
        by_provider_list = [
            {
                "provider": prov,
                "calls":    row["calls"],
                "in_tok":   row["in_tok"],
                "out_tok":  row["out_tok"],
                "cost_usd": round(row["cost_usd"], 6),
            }
            for prov, row in by_provider.items()
        ]
        by_provider_list.sort(key=lambda r: r["cost_usd"], reverse=True)

        summary = {
            "total_calls":  total_calls,
            "total_ms":     int(total_ms),
            "total_cost":   round(total_cost, 4),
            "by_model":     by_model_list,
            "by_provider":  by_provider_list,
            "slowest":      slowest_list,
        }

        # Human-friendly multi-line summary block.
        prefix = f"[{request_id}] " if request_id else ""
        lines = [
            "",
            f"  ╔══ {prefix}LLM SUMMARY  total_calls={total_calls}  "
            f"cumulative_ms={int(total_ms)}  cost=${total_cost:.4f}",
            "  ╟── by provider (which key was billed) ──────────────────────",
            f"  ║    {'provider':<14} {'calls':>5} {'in_tok':>9} {'out_tok':>9} {'cost':>9}",
        ]
        for p in by_provider_list:
            lines.append(
                f"  ║    {p['provider']:<14} {p['calls']:>5} "
                f"{p['in_tok']:>9} {p['out_tok']:>9} ${p['cost_usd']:>7.4f}"
            )
        lines.append("  ╟── by model ────────────────────────────────────────────────")
        lines.append(f"  ║    {'model':<42} {'calls':>5} {'total':>8} {'avg':>6} {'cost':>9}")
        for r in by_model_list:
            lines.append(
                f"  ║    {r['model']:<42} {r['calls']:>5} "
                f"{r['total_ms']:>7}ms {r['avg_ms']:>5}ms ${r['cost_usd']:>7.4f}"
            )
        lines.append("  ╟── slowest single calls ────────────────────────────────────")
        for s in slowest_list:
            lines.append(f"  ║    {s['ms']:>6}ms  {s['model']:<42} {s['role']}")
        lines.append("  ╚════════════════════════════════════════════════════════════")
        _obs_log.info("\n".join(lines))

        return summary

    # -----------------------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------------------

    def get_total_tokens(self) -> dict[str, int]:
        """Get total token usage across all calls."""
        total_input = sum(log.input_tokens for log in self.call_log)
        total_output = sum(log.output_tokens for log in self.call_log)
        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

    def get_total_cost_estimate(self) -> float:
        """
        Estimate total cost in USD by summing per-call costs at each model's
        actual pricing (from provider_map.PRICING). Unknown models fall back
        to Sonnet-tier pricing as a conservative estimate.
        """
        total = 0.0
        for log in self.call_log:
            in_price, out_price = get_pricing(log.model)
            total += log.input_tokens * in_price / 1_000_000
            total += log.output_tokens * out_price / 1_000_000
        return total

    def get_call_summary(self) -> dict:
        """Get a summary of all calls made — per-domain and per-model breakdown."""
        total_calls = len(self.call_log)
        successful = sum(1 for log in self.call_log if log.success)
        failed = total_calls - successful
        avg_latency = (
            sum(log.latency_ms for log in self.call_log) / total_calls
            if total_calls > 0 else 0
        )

        # Per-domain breakdown
        domain_breakdown: dict[str, dict] = {}
        for log in self.call_log:
            if log.domain not in domain_breakdown:
                domain_breakdown[log.domain] = {
                    "calls": 0, "tokens": 0, "avg_latency_ms": 0, "failures": 0
                }
            db = domain_breakdown[log.domain]
            db["calls"] += 1
            db["tokens"] += log.input_tokens + log.output_tokens
            db["avg_latency_ms"] += log.latency_ms
            if not log.success:
                db["failures"] += 1

        for db in domain_breakdown.values():
            if db["calls"] > 0:
                db["avg_latency_ms"] /= db["calls"]

        # Per-model breakdown (new — useful when multiple providers are in play)
        model_breakdown: dict[str, dict] = {}
        for log in self.call_log:
            if log.model not in model_breakdown:
                model_breakdown[log.model] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "failures": 0,
                }
            mb = model_breakdown[log.model]
            mb["calls"] += 1
            mb["input_tokens"] += log.input_tokens
            mb["output_tokens"] += log.output_tokens
            in_price, out_price = get_pricing(log.model)
            mb["cost_usd"] += log.input_tokens * in_price / 1_000_000
            mb["cost_usd"] += log.output_tokens * out_price / 1_000_000
            if not log.success:
                mb["failures"] += 1

        return {
            "total_calls": total_calls,
            "successful": successful,
            "failed": failed,
            "avg_latency_ms": avg_latency,
            "total_tokens": self.get_total_tokens(),
            "estimated_cost_usd": self.get_total_cost_estimate(),
            "domain_breakdown": domain_breakdown,
            "model_breakdown": model_breakdown,
        }


# ---------------------------------------------------------------------------
# Mock embedding — hashing-trick pseudo-embedding for tests
# ---------------------------------------------------------------------------

_EMBED_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MOCK_EMBED_DIM = 1536  # matches openai/text-embedding-3-small


def _mock_embedding(text: str, dim: int = _MOCK_EMBED_DIM) -> list[float]:
    """
    Deterministic pseudo-embedding for the test path.

    The hashing trick: each token in the text is hashed to a deterministic
    bucket in a dim-sized vector; the bucket gets incremented. The final
    vector is L2-normalized. Two texts sharing tokens produce vectors
    that align in cosine space; texts with no shared tokens are
    near-orthogonal. NOT a substitute for real semantic embeddings —
    just enough plumbing to test that EmbeddingScorer works without
    making API calls.
    """
    if not text:
        return [0.0] * dim
    tokens = {t for t in _EMBED_TOKEN_RE.findall(text.lower()) if len(t) > 2}
    if not tokens:
        return [0.0] * dim
    vec = [0.0] * dim
    for token in tokens:
        bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]
