# Payment System — STANDBY

**Status:** Designed, not implemented. Pick this up when subscription + tier gating becomes the next priority.
**Last touched:** 2026-05-28
**Owner:** Nikhil

---

## TL;DR

Constellax has NO subscription infrastructure today. Auth exists (Supabase JWT verification), guest discriminator exists (`is_guest_user_id`), but no Stripe wiring, no Pro/Free distinction, no usage counter, no effort-tier gate.

Design below is locked. Numbers are locked. Implementation is deferred until traction signals are clear.

---

## Product decisions (locked 2026-05-28)

### Pricing
- **Pro tier:** USD $19.99/mo (NOT CAD — Constellax is USD; LoRa is CAD)
- **Free tier:** $0

### Effort tier gating
- **Free users:** LOW (3 iter) + AUTO (4 iter) only
- **Pro users:** LOW + AUTO + MEDIUM (5 iter) + HIGH (8 iter)

### Monthly Thinking Mode answer caps
- **Free:** 25 answers / month
- **Pro:** 80 answers / month
- Both reset on UTC first-of-month boundary
- Founder bypass (Nikhil's user_id) for testing

### Cost per Thinking Mode answer (under current API model assignments)

| Effort | Iter | Cost | Wall time | Tier |
|---|---|---|---|---|
| LOW | 3 | $0.24 | 20s | Free + Pro |
| AUTO | 4 | $0.31 | 26s | Free + Pro |
| MEDIUM | 5 | $0.38 | 32s | Pro only |
| HIGH | 8 | $0.60 | 50s | Pro only |

---

## Unit economics (at locked caps)

**Free user worst case:** 25 answers × $0.27 avg = **$6.75/user/month burn** (zero revenue)
**Pro user at cap:** 80 × $0.39 avg = $31.20 cost vs $19.99 revenue = **-$11.21/user loss at cap**

### Pareto distribution model (Pro users)

| Segment | % of base | Answers/mo | Cost | Margin |
|---|---|---|---|---|
| Power | 10% | 80 (cap) | $31.20 | -$11.21 |
| Active | 30% | 50 | $19.50 | +$0.49 |
| Casual | 40% | 15 | $5.85 | +$14.14 |
| Inactive | 20% | 2 | $0.78 | +$19.21 |

**Net at 100 Pro users: ~$850/mo profit** (~43% margin)
**Net at 100 Pro + 100 Free: ~$175/mo profit** (margin protected by Free→Pro conversion)

Break-even conversion rate: ~3-5% of Free users convert to Pro.

---

## Long-term cost play (Nemotron 3 Nano)

API costs are the **interim** problem, not the permanent one. Strategic plan once traction + capital allow:

| Layer | API era | Nemotron era |
|---|---|---|
| 5 Sheng angles | DeepSeek/Haiku/Gemini ($0.07/iter) | Self-hosted Nemotron 3 Nano (~free at margin) |
| 5 Ke critics | Haiku ($0.04/iter) | Self-hosted Nemotron |
| Synthesizer | Sonnet 4.6 ($0.04/answer) | Stay Sonnet until 70B+ self-host viable |

- Self-host cost: ~$1,500–3,000/mo (one H100 or 2-4× A100s)
- Breakeven: ~10M tokens/month (~1K active users at current volume)
- Target margin at 1K Pro users with Nemotron live: **~80%** (vs ~40% on APIs)
- Fine-tune Nemotron on the 5 angle-specific prompts; each angle is a narrow task where small models shine

---

## Implementation plan (when picked up)

### Phase 1A — Counter + gate, SOFT-MONITOR mode (~1 hour)
- New file: `src/server/think_cap.py` (~80 lines)
  - Redis monthly counter: `constellax:think_count:{userId}:{YYYY-MM}`
  - `check_and_increment(user_id, effort) -> ThinkCapResult`
  - Founder bypass via `UNLIMITED_USERS` env-var allowlist
  - Mode flag: `LORA_THINKING_GATE_MODE=monitor|enforce` (default `monitor`)
- `src/dispatcher.py` integration:
  - Call `check_and_increment` at top of `_dispatch_deep`
  - In `monitor` mode: log "would have blocked" but pass through
  - In `enforce` mode: return 402 with friendly message
- Operational counters: add `thinking_mode_attempts_*`, `would_block_free_*`, `would_block_pro_*`

### Phase 1B — Stripe subscription wiring (~3-5 days)
- Mirror the LoRa pattern (`src/server/subscription/SubscriptionService.ts`):
  - Redis hash `constellax:sub:{userId}` with `customerId`, `subscriptionId`, `status`, `currentPeriodEnd`
  - `isActiveSubscriber(user_id)` checker
  - Fail-open on Redis read errors (don't block paying users on infra hiccups)
- New routes:
  - `POST /api/subscription/checkout` (creates Stripe Checkout Session)
  - `POST /api/subscription/portal` (creates Customer Portal session)
  - `POST /api/stripe/webhook` (signature-verified, mirrors to Redis)
- Webhook registered **BEFORE** FastAPI's JSON middleware (signature verification needs raw body)
- New env vars: `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_SUB_PRICE_ID`

### Phase 1C — Flip the gate to enforce (~5 min)
- Set `LORA_THINKING_GATE_MODE=enforce` in Railway/prod env
- Update gate to read `isActiveSubscriber(user_id)` instead of stub
- Done — gate is live

---

## What's already in place (foundation we don't have to rebuild)

- **Supabase Auth** with HS256 JWT verification (`src/auth/supabase_auth.py`)
- **Guest vs signed-in discriminator** (`is_guest_user_id`)
- **`get_effective_user_id(request, body_user_id)`** — JWT-first, body-fallback resolution
- **Thread ownership gate** strict-when-authed (`_assert_thread_ownership`)
- **Memory V2 retrieval gating** for guests (skips graph for non-signed-in)
- **`require_auth` FastAPI dependency** built but unwired — ready to enforce per-route when gate flips

## What's missing (must build at Phase 1A/1B)

- Redis client for the counter (Constellax doesn't currently use Redis for app state — only Neo4j)
  - Option: reuse the existing FalkorDB/Redis instance if env-var set; otherwise stand up a Redis service
- Stripe SDK + webhook handler
- Customer portal route
- Free-tier UX (showing "X / 25 used this month" in UI, "Upgrade to Pro" CTA)
- Email receipts (Stripe handles transactional, but signup confirmation flow TBD)

---

## Risks to flag when picking up

1. **Free tier burn at scale.** Even with 25-answer cap, 100 active free users = ~$675/mo burn with zero revenue. Need Free→Pro conversion ≥3% to break even at scale. Watch this carefully in first 60 days post-launch.
2. **Power-user tail.** 10% of Pro users hitting the 80-answer cap = $11/user loss. Consider tighter cap (60?) or soft-cap with overage tier ($19.99 + $19.99 = 200 answers) if tail grows.
3. **Stripe review wait.** New Stripe accounts can sit in "review in progress" for 1-2 weeks. Don't launch Pro until Stripe is fully live. LoRa hit this exact issue April 4-6, 2026.
4. **Webhook signature order.** Stripe webhook MUST register before FastAPI JSON middleware. LoRa's `adapter.ts:84` pattern is the reference. Easy to get wrong; pre-launch smoke test with unsigned POST should return 400.
5. **Founder bypass should never reach production accidentally.** When `UNLIMITED_USERS` env-var is set, those users skip ALL gates. Document this as a security boundary; review on every deploy.

---

## State of working tree at standby (2026-05-28)

Three files modified, NOT committed:

| File | Change | Status |
|---|---|---|
| `src/llm/provider_map.py` | Sonnet → DeepSeek/Haiku/Gemini Flash at angle layer | Tests green, committed-pending |
| `src/llm/effort.py` | EFFORT_ITERATIONS: 2/3/5/8 → 3/5/8/4 | Tests green, committed-pending |
| `tests/test_dispatch_preview.py` | Fixtures updated for new iter counts | All 23 tests pass |

Rollback any: `git checkout -- <path>`
Full suite state: 537 passed, 0 failed across 20 test modules.

---

---

# Deploy-time guardrails (set via Railway env vars, NOT in code)

Same spirit as the payment system: documented here, switched on at deploy. Nothing wired in code yet because these are environment-level concerns that should live alongside the Pro launch decision, not slip in beforehand.

## 1. User-level rate limiting

**Current state:** no rate limiter wired in Constellax. Single abusive user can drain credits.

**Approach when picked up:** port LoRa's `slidingWindowRateLimit` pattern. Per-user sliding window in Redis. Two-key strategy (per-userId + per-IP for guests) to defeat localStorage-uuid abuse.

**Default caps (proposed, tune at launch):**

| Cap | Window | Free tier | Pro tier |
|---|---|---|---|
| Total requests | 1 min | 10 | 30 |
| Total requests | 1 hour | 60 | 300 |
| Total requests | 24 hr | 200 | 2000 |
| Daily token spend per user | 24 hr UTC | 50,000 tokens | 500,000 tokens |
| Daily Thinking Mode answers | 24 hr UTC | (covered by 25/mo cap) | (covered by 80/mo cap) |

**Env vars to add:**
```
LORA_RATE_LIMIT_ENABLED=1
LORA_RATE_LIMIT_FREE_PER_MIN=10
LORA_RATE_LIMIT_FREE_PER_HOUR=60
LORA_RATE_LIMIT_PRO_PER_MIN=30
LORA_RATE_LIMIT_PRO_PER_HOUR=300
LORA_DAILY_TOKEN_LIMIT_FREE=50000
LORA_DAILY_TOKEN_LIMIT_PRO=500000
```

**Implementation:** new `src/server/rate_limit.py` (~120 lines), Redis-backed sliding-window counter, fail-open on Redis errors (don't block legit users on infra hiccups). Wire as FastAPI middleware AFTER auth middleware (so it knows who you are).

**Token spend cap:** in-memory daily counter per user, resets UTC midnight. LoRa pattern in `src/server/usage/DailyTokenUsage.ts`. Resets on redeploy — acceptable since deploys are rare.

## 2. CORS production allowlist

**Current state:** `CORS_ORIGINS=*` in local `.env` (permissive). Fine for dev, NOT for prod.

**Set in Railway env at deploy:**
```
CORS_ORIGINS=https://constellax.app,https://www.constellax.app
```
(Add the localhost dev origin only if you need browser-based testing against prod, which you usually don't.)

## 3. Security headers (LoRa parity)

LoRa's adapter.ts ships HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Permissions-Policy, X-Permitted-Cross-Domain-Policies. Constellax currently does NOT. When picking this up, port the security headers middleware. ~30 lines, single FastAPI middleware.

## 4. Input sanitization — prompt injection filter

LoRa has 20 prompt injection patterns in `src/server/auth/inputSanitizer.ts` (strip + log approach). Constellax has only `MAX_QUESTION_CHARS=8000` length check. Same prompt injection vectors apply to a thinking partner that fires LLM calls — port the filter when picking up the Pro launch.

## 5. Health dashboard (LoRa parity)

LoRa has `GET /api/health` — mobile-friendly HTML dashboard showing service statuses + 17 operational counters (LLM calls, fallbacks, rate limits, identity guard rewrites, perspective engine, memory V2, errors). Constellax has `/health` (returns dict status only).

**To port:** new `src/server/health_dashboard.py` + operational counters module. Token-gate the full dashboard behind `HEALTH_DASHBOARD_TOKEN` env var (LoRa learned this lesson the hard way — initial version was unauthenticated and leaked user-id fragments + system info).

## 6. Cleanup items (cosmetic, do at any pickup)

- `.env` has `CONSTELLAX_DB_BACKEND` listed TWICE — last-wins behavior, no actual bug, just de-dupe
- `.env` has `USERNAME=…` not documented in `.env.example` — either remove or document (likely shell-env bleed-through, harmless)
- Local dev: if you see `neo4j driver not installed — falling back to in-memory` on boot, run `pip install -r requirements.txt` to refresh the local venv. Production via Docker has it baked in.

## 7. `require_auth` flip day

The `require_auth` FastAPI dependency exists in `src/auth/supabase_auth.py:278` but is NOT wired to any endpoint. Today every endpoint accepts both signed-in and guest traffic (gated only on ownership). When Pro launches, decide per-endpoint:
- Endpoints that should remain guest-accessible (chat, dispatch preview, public threads): leave alone
- Endpoints that should require sign-in (subscription routes, customer portal, Stripe checkout): add `Depends(require_auth)`

Most likely: only the subscription routes need require_auth. The chat surface stays open with the 25-answer/month cap for guests.

---

## Pick-up checklist (future-me)

When you come back to this:

- [ ] Re-read this entire doc, especially the unit economics
- [ ] Re-confirm caps still match product strategy (numbers may need updating)
- [ ] Check whether Nemotron 3 Nano is now viable (GPU pricing, model availability)
- [ ] Decide on Phase 1A vs 1A+1B vs 1A+1B+1C sequencing
- [ ] If shipping Phase 1A only: add big "PRE-RELEASE" warning to UI so users know they're in soft-monitor mode
- [ ] Test webhook signature enforcement LIVE before launch (`curl -X POST` with no signature → must 400)
- [ ] Verify founder bypass works (you should never be blocked)
- [ ] Stripe live mode activated (not test mode) before announcing
