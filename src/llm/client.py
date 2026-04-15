"""
LLM Client — The River.

Single async Sonnet connection. All domain agents call through this wrapper.
The model is the same — the system prompt is what differentiates agents.

Two modes:
- LIVE: Real Anthropic API calls (requires ANTHROPIC_API_KEY env var)
- MOCK: Deterministic responses for architecture testing (no API credits spent)

Every call logs: domain, concept, token count, latency, success/failure.

ISOLATION: This module does NOT import from any domain module.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClientMode(Enum):
    """Operating mode of the LLM client."""
    LIVE = "live"       # real API calls
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
    error: str | None = None
    raw: Any = None         # raw API response object (live mode only)


class LLMClient:
    """
    The River — single LLM connection for all domain agents.

    Usage:
        client = LLMClient(mode=ClientMode.MOCK)  # for testing
        response = await client.call(
            system_prompt="You are LoRa's Physics agent...",
            user_message="Analyze this problem...",
            domain="physics",
            concept="first_principles",
        )
    """

    # Default model — Sonnet is the muscle, the architecture is the brain
    DEFAULT_MODEL = "claude-sonnet-4-6"

    # Safety limits
    MAX_RETRIES = 2
    TIMEOUT_SECONDS = 30
    MAX_OUTPUT_TOKENS = 4096

    def __init__(
        self,
        mode: ClientMode = ClientMode.MOCK,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.mode = mode
        self.model = model or self.DEFAULT_MODEL
        self.call_log: list[LLMCallLog] = []
        self._client = None

        if mode == ClientMode.LIVE:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is required for live mode. "
                    "Set it with: export ANTHROPIC_API_KEY='your-key'"
                )
            # Lazy import — only needed in live mode
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=key)
            except ImportError:
                raise ImportError(
                    "anthropic package required for live mode. "
                    "Install with: pip install anthropic"
                )

    async def call(
        self,
        system_prompt: str,
        user_message: str,
        domain: str,
        concept: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Make a single LLM call.

        This is one tributary flowing from the river.
        Every domain agent calls this with its own system prompt.
        """
        max_tokens = max_tokens or self.MAX_OUTPUT_TOKENS
        start_time = time.monotonic()

        for attempt in range(self.MAX_RETRIES):
            try:
                if self.mode == ClientMode.MOCK:
                    response = await self._mock_call(
                        system_prompt, user_message, domain, concept
                    )
                else:
                    response = await self._live_call(
                        system_prompt, user_message, max_tokens, temperature
                    )

                elapsed_ms = (time.monotonic() - start_time) * 1000

                # Update response timing
                response.latency_ms = elapsed_ms

                # Log the call
                self._log_call(
                    domain=domain,
                    concept=concept,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=elapsed_ms,
                    success=True,
                )

                return response

            except Exception as e:
                elapsed_ms = (time.monotonic() - start_time) * 1000

                if attempt < self.MAX_RETRIES - 1:
                    # Retry after brief pause
                    await asyncio.sleep(1.0)
                    continue

                # All retries exhausted
                self._log_call(
                    domain=domain,
                    concept=concept,
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
                    error=str(e),
                )

    async def call_batch(
        self,
        calls: list[dict],
    ) -> list[LLMResponse]:
        """
        Fan-out: launch multiple LLM calls in parallel.

        Each call dict has: system_prompt, user_message, domain, concept.
        All calls run simultaneously via asyncio.gather().
        Returns responses in the same order as the input calls.
        """
        tasks = [
            self.call(
                system_prompt=c["system_prompt"],
                user_message=c["user_message"],
                domain=c["domain"],
                concept=c["concept"],
                max_tokens=c.get("max_tokens"),
                temperature=c.get("temperature", 0.7),
            )
            for c in calls
        ]

        return await asyncio.gather(*tasks)

    # -----------------------------------------------------------------------
    # Live mode
    # -----------------------------------------------------------------------

    async def _live_call(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Make a real API call to Anthropic."""
        response = await asyncio.wait_for(
            self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ),
            timeout=self.TIMEOUT_SECONDS,
        )

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=0,  # will be updated by caller
            success=True,
            raw=response,
        )

    # -----------------------------------------------------------------------
    # Mock mode
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

        Simulates realistic response structure without API calls.
        Each domain/concept gets a structured response that the parser
        can process, allowing the full fan-out/fan-in/convergence
        architecture to be tested without spending credits.
        """
        # Simulate realistic latency (50-200ms for mock)
        await asyncio.sleep(0.05 + hash(domain + concept) % 150 / 1000)

        # Generate domain-specific mock response
        mock_content = self._generate_mock_response(domain, concept, user_message)

        # Simulate token counts (rough estimates)
        input_tokens = len(system_prompt.split()) + len(user_message.split())
        output_tokens = len(mock_content.split())

        return LLMResponse(
            content=mock_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0,  # will be updated by caller
            success=True,
        )

    def _generate_mock_response(
        self, domain: str, concept: str, user_message: str
    ) -> str:
        """Generate a structured mock response for a given domain/concept."""

        # Extract some keywords from user message for realistic mock
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
    # Logging
    # -----------------------------------------------------------------------

    def _log_call(
        self,
        domain: str,
        concept: str,
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
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error,
            timestamp=time.time(),
        )
        self.call_log.append(log)

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
        Estimate total cost in USD based on Sonnet pricing.
        Sonnet: $3/M input, $15/M output (as of 2025).
        """
        tokens = self.get_total_tokens()
        input_cost = tokens["input_tokens"] * 3.0 / 1_000_000
        output_cost = tokens["output_tokens"] * 15.0 / 1_000_000
        return input_cost + output_cost

    def get_call_summary(self) -> dict:
        """Get a summary of all calls made."""
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

        return {
            "total_calls": total_calls,
            "successful": successful,
            "failed": failed,
            "avg_latency_ms": avg_latency,
            "total_tokens": self.get_total_tokens(),
            "estimated_cost_usd": self.get_total_cost_estimate(),
            "domain_breakdown": domain_breakdown,
        }
