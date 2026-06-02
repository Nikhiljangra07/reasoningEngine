"""
CallTracker — per-LLM-call audit log for a Wandering Room session.

WHY THIS EXISTS
---------------
Run #1 of the live wander (May 31, 2026) configured 6 agents with a
2×DeepSeek + 4×Haiku model_mix, but the AgentState.model_slug field was
assigned and never consumed: every client.call(...) at the call sites
(_run_dig_iteration, score_mechanism, score_non_map, match_content)
hardcoded `domain="synthesizer"` or `domain="psychology"`, which
provider_map.resolve_model() routes to a fixed model irrespective of the
agent. Effect: the entire cohort silently collapsed to a Sonnet+Haiku
monoculture. The "cohort diversity" claim was structurally false. (This
is the same class of silent regression as the LoRa Sonnet-4 drift from
March → April 2026 — assigned-but-unread variables are a known
production failure mode.)

This module fixes the audit half of the problem. AgentScopedLLMClient
wraps a base LLMClient so every call routed through it is:

  1. Tagged with the agent_id of the agent that issued it.
  2. Forced through that agent's `default_model` (= AgentState.model_slug)
     when the caller did not pass an explicit `model=` override. Explicit
     model= still wins, so legacy code paths that depend on a specific
     model can keep working unchanged.
  3. Recorded into a CallTracker with: agent_id, purpose (= concept),
     domain, model_requested, model_actually_used (from response.model),
     input/output tokens, latency, success/error, timestamp.

The tracker can stream every record to a jsonl file as calls happen
(append-mode, flushed per record — crash-safe). Callers that prefer
in-memory only pass `jsonl_path=None`.

DOCTRINE LINK
-------------
Per Law 4 ("the Wandering Room reads and reasons; it does not act"),
this tracker is observation-only: it never alters routing decisions,
retries, or response content. It is a passive recorder beside the
existing client._log_call observability — separate from
provider_map.resolve_model() so cohort accounting and provider routing
remain decoupled.

ISOLATION
---------
Imports only LLMClient + LLMResponse from src.llm.client and stdlib.
No imports back into src.wandering.* so the runtime can import this
without circulars.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.llm.client import LLMClient, LLMResponse


log = logging.getLogger("constellax.wandering.call_tracker")


# ---------------------------------------------------------------------------
# CallRecord — one row of the audit log
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    """One LLM call's audit record.

    `model_requested` is what AgentScopedLLMClient sent into LLMClient.call;
    `model_actually_used` is the model that LLMClient.call reports back on
    LLMResponse.model. Divergence between the two means provider_map
    rerouted (e.g., fallback) or an explicit `model=` override won — both
    are important to surface, which is why we record both.
    """

    session_id:          str
    agent_id:            str           # "" for non-agent calls (e.g. cushion compose)
    call_index:          int           # monotonic per session
    timestamp:           float
    purpose:             str           # = concept passed to LLMClient.call
    domain:              str           # = domain passed to LLMClient.call
    model_requested:     str           # what AgentScopedLLMClient asked for
    model_actually_used: str           # what LLMResponse.model reports
    input_tokens:        int
    output_tokens:       int
    latency_ms:          float
    success:             bool
    error:               str = ""


# ---------------------------------------------------------------------------
# CallTracker — append-only collector + optional jsonl stream
# ---------------------------------------------------------------------------


class CallTracker:
    """Append-only audit log for every LLM call made during one Wandering
    Room session.

    Construction:
      tracker = CallTracker(session_id="live_wander_..", jsonl_path="/tmp/x.jsonl")

    When `jsonl_path` is set, every record() call appends one JSON object
    line to that file (newline-delimited JSON, RFC 7464-ish) and flushes
    the file handle, so a crash mid-wander still preserves the trail.
    When jsonl_path is None, records live in memory only.

    Thread/asyncio safety: record() is called from inside agent loops that
    run concurrently in asyncio.gather. Python's list.append + int
    increment are atomic under the GIL; file writes are serialized through
    an asyncio.Lock so two concurrent agents can't interleave bytes in the
    jsonl file.
    """

    def __init__(
        self,
        session_id: str,
        jsonl_path: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.jsonl_path = jsonl_path
        self.records: list[CallRecord] = []
        self._counter = 0
        self._write_lock = asyncio.Lock()
        self._fh: Any = None

        if jsonl_path:
            # Open in append + line-buffered mode so each write is flushed
            # to disk immediately on newline. This is crash-safe at the
            # cost of one fsync-equivalent per record — fine for the
            # ~100-1000 records a wander produces.
            try:
                os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
                self._fh = open(jsonl_path, "a", buffering=1, encoding="utf-8")
                log.info(
                    "CallTracker[%s]: streaming to %s",
                    session_id, jsonl_path,
                )
            except OSError as e:
                log.warning(
                    "CallTracker[%s]: failed to open %s for append (%s); "
                    "falling back to in-memory only",
                    session_id, jsonl_path, e,
                )
                self._fh = None
                self.jsonl_path = None

    async def record(
        self,
        *,
        agent_id: str,
        purpose: str,
        domain: str,
        model_requested: str,
        response: LLMResponse,
    ) -> CallRecord:
        """Record one call. Pure observation — never raises, never alters
        the response. Returns the recorded CallRecord for callers that
        want it.
        """
        rec = CallRecord(
            session_id=self.session_id,
            agent_id=agent_id,
            call_index=self._counter,
            timestamp=time.time(),
            purpose=purpose,
            domain=domain,
            model_requested=model_requested,
            model_actually_used=response.model or "",
            input_tokens=int(response.input_tokens or 0),
            output_tokens=int(response.output_tokens or 0),
            latency_ms=float(response.latency_ms or 0.0),
            success=bool(response.success),
            error=(response.error or "") if not response.success else "",
        )
        self._counter += 1
        self.records.append(rec)

        if self._fh is not None:
            try:
                async with self._write_lock:
                    self._fh.write(json.dumps(asdict(rec), ensure_ascii=False))
                    self._fh.write("\n")
            except Exception as e:
                # Stay non-fatal — telemetry must never block the wander.
                log.warning("CallTracker[%s]: jsonl write failed (%s)", self.session_id, e)

        return rec

    def close(self) -> None:
        """Idempotent. Closes the jsonl file handle if one is open."""
        fh = self._fh
        self._fh = None
        if fh is not None:
            try:
                fh.flush()
                fh.close()
            except Exception:  # pragma: no cover — defensive
                pass

    def __enter__(self) -> "CallTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---- Aggregations / summary helpers — read-only ----

    def total_calls(self) -> int:
        return len(self.records)

    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def calls_by_agent(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.agent_id] = out.get(r.agent_id, 0) + 1
        return out

    def models_actually_used(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            key = r.model_actually_used or "(unknown)"
            out[key] = out.get(key, 0) + 1
        return out

    def models_per_agent(self) -> dict[str, dict[str, int]]:
        """Per-agent breakdown of model_actually_used. The diagnostic
        readers of run #2 will check this first: every record under
        agent P01/P02 (DeepSeek slots) should report
        `model_actually_used = "deepseek/deepseek-v4-pro"` for the dig
        calls; if it doesn't, the per-call routing is still broken."""
        out: dict[str, dict[str, int]] = {}
        for r in self.records:
            slot = out.setdefault(r.agent_id, {})
            key = r.model_actually_used or "(unknown)"
            slot[key] = slot.get(key, 0) + 1
        return out

    def purpose_breakdown(self) -> dict[str, int]:
        """Count calls by purpose. Useful for spotting whether a category
        of calls (e.g. interpreter_mechanism_judge) was actually exercised
        at all in this run."""
        out: dict[str, int] = {}
        for r in self.records:
            out[r.purpose] = out.get(r.purpose, 0) + 1
        return out

    def failures(self) -> list[CallRecord]:
        return [r for r in self.records if not r.success]

    def summary(self) -> dict[str, Any]:
        """Compact dict for the result JSON. Includes the top-line
        counts a human reading the rerun output needs."""
        latencies = [r.latency_ms for r in self.records if r.success]
        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p95 = latencies[p95_idx] if latencies else 0.0
        return {
            "session_id": self.session_id,
            "total_calls": self.total_calls(),
            "total_tokens": self.total_tokens(),
            "failures": len(self.failures()),
            "calls_by_agent": self.calls_by_agent(),
            "models_actually_used": self.models_actually_used(),
            "models_per_agent": self.models_per_agent(),
            "purpose_breakdown": self.purpose_breakdown(),
            "latency_p50_ms": round(p50, 1),
            "latency_p95_ms": round(p95, 1),
            "jsonl_path": self.jsonl_path or "",
        }


# ---------------------------------------------------------------------------
# AgentScopedLLMClient — drop-in wrapper that tags + records every call
# ---------------------------------------------------------------------------


class AgentScopedLLMClient:
    """A thin wrapper around LLMClient that:

      1. Tags every call with the wrapping agent's `agent_id` (or "" for
         pre-agent contexts like cushion composition).
      2. Substitutes the agent's `default_model` for any call that did not
         explicitly pass a `model=` keyword (so AgentState.model_slug
         actually governs the API the agent talks to).
      3. Records the call into a CallTracker.

    Everything else is delegated to the underlying client — same return
    shape (LLMResponse), same retries inside the base client, same
    observability via the base client's _log_call. This wrapper is purely
    additive.

    DESIGN NOTE on `default_model` semantics
    ----------------------------------------
    When an explicit `model=` kwarg is passed at the call site, we honor
    it verbatim. This preserves the legacy contract for code that
    intentionally wants a specific model (e.g. an extraction helper that
    needs Haiku for cheap structured judgment regardless of the wrapping
    agent). The wrapper only fills in `default_model` when the caller did
    not specify one.

    Implementation detail: we do NOT subclass LLMClient — we duck-type
    its `.call(...)` signature. That keeps `isinstance` checks elsewhere
    in the codebase unchanged and avoids inheriting LLMClient's init-time
    side effects (provider registration, key validation).
    """

    def __init__(
        self,
        base: LLMClient,
        tracker: CallTracker,
        agent_id: str,
        default_model: str,
    ) -> None:
        self._base = base
        self._tracker = tracker
        self._agent_id = agent_id
        self._default_model = default_model

    @property
    def base(self) -> LLMClient:
        """Escape hatch for callers that need the raw base client. Use
        sparingly — anything that goes through this property bypasses
        the audit log."""
        return self._base

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def default_model(self) -> str:
        return self._default_model

    # Pass through the base client's mode attribute so callers can
    # introspect without unwrapping.
    @property
    def mode(self) -> object:
        return self._base.mode

    async def call(
        self,
        system_prompt: str,
        user_message: str,
        domain: str,
        concept: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        model: str | None = None,
    ) -> LLMResponse:
        """See LLMClient.call. Adds tagging + tracking."""
        chosen_model = model if model is not None else self._default_model
        response = await self._base.call(
            system_prompt=system_prompt,
            user_message=user_message,
            domain=domain,
            concept=concept,
            max_tokens=max_tokens,
            temperature=temperature,
            model=chosen_model,
        )
        try:
            await self._tracker.record(
                agent_id=self._agent_id,
                purpose=concept,
                domain=domain,
                model_requested=chosen_model or "",
                response=response,
            )
        except Exception as e:  # pragma: no cover — defensive
            log.warning(
                "CallTracker.record raised for agent=%s purpose=%s: %s",
                self._agent_id, concept, e,
            )
        return response

    # NOTE: call_batch is intentionally NOT overridden here. The
    # wandering pipeline doesn't call client.call_batch anywhere; adding
    # a passthrough would only invent new lint surface (the identity
    # source-proof test sees `system_prompt=c['system_prompt']` and
    # can't exempt subscript expressions). If a future caller needs
    # batched audit, wire a thin call_batch that pre-extracts each
    # dict's `system_prompt` into a local Name before forwarding.


__all__ = [
    "CallRecord",
    "CallTracker",
    "AgentScopedLLMClient",
]
