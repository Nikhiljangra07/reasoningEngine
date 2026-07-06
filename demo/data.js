// AUTO-GENERATED from real run artifacts by build_demo.py — do not hand-edit.
window.RUN_DATA = {
  "problem": {
    "title": "Distinguishing genuine coordination from mere correlation among AI agents \u2014 from outside",
    "paper": "Multi-Agent Risks from Advanced AI (Cooperative AI Foundation, 2025)",
    "paper_url": "https://arxiv.org/abs/2502.14143",
    "angles": [
      "What observable signature separates coordination from shared-cause correlation?",
      "Is 'intent' the wrong criterion \u2014 replace it with an interventional test?",
      "How to detect emergent coordination with no message to intercept?",
      "What to rely on when the channel is covert / undetectable in principle?",
      "What governs the correlation\u2194coordination boundary \u2014 a measurable order parameter?"
    ]
  },
  "cycles": [
    {
      "n": 1,
      "cards_total": 56,
      "d_t": 0.0,
      "shepherd": "circling",
      "refocus": "Steer toward concrete OBSERVABLE SIGNATURES that differ between the three regimes \u2014 e.g., response-time asymmetries, price-path autocorrelat",
      "halt": ""
    },
    {
      "n": 2,
      "cards_total": 92,
      "d_t": 0.8,
      "shepherd": "circling",
      "refocus": "Steer toward the counterfactual/interventional criterion specifically: what formal test \u2014 do-calculus, Granger causality, structural causal ",
      "halt": ""
    },
    {
      "n": 3,
      "cards_total": 143,
      "d_t": 0.8,
      "shepherd": "circling",
      "refocus": "Steer toward the open angle directly: whether 'intent' is the wrong criterion for machine coordination and what a concrete counterfactual or",
      "halt": ""
    },
    {
      "n": 4,
      "cards_total": 218,
      "d_t": 0.8,
      "shepherd": "circling",
      "refocus": "Steer toward the counterfactual/interventional test of mutual influence as a replacement for intent \u2014 what would a do-calculus or Granger-st",
      "halt": "cycle cap 4 reached (open: [2])"
    }
  ],
  "cards": [
    {
      "label": "Stoic philosophy",
      "domain": "Philosophy",
      "shape": "Stoicism draws a hard line between externals (outcomes, others' actions, circumstances) and internals (your judgments, impulses, assent). The Stoic practitioner can only observe the outputs of their own reasoning \u2014 the actions taken, the words spoken \u2014 and must infer the health of their inner facult",
      "bridge": "Your overseer is in the Stoic's position, but watching agents instead of watching itself. The agents' outputs \u2014 trades, messages, prices, decisions \u2014 are the only surface available, just as the Stoic'"
    },
    {
      "label": "Buddhist dependent origination",
      "domain": "Philosophy",
      "shape": "In Buddhist dependent origination (prat\u012btyasamutp\u0101da), phenomena do not arise from a single identifiable cause but from the simultaneous co-arising of multiple conditions \u2014 remove any one and the phenomenon dissolves, yet no single condition 'owns' the result. This makes the arising observationally ",
      "bridge": "Your anchor's core problem is causal identifiability: when autonomous agents produce coordinated-looking behavior in a shared environment, an overseer cannot tell from observation alone whether that c"
    },
    {
      "label": "Collective agency as causal abstraction",
      "domain": "Causal theory",
      "shape": "The paper formalizes collective agency using causal games: a group qualifies as a collective agent when its joint behavior is better predicted by a single rational, goal-directed high-level model than by treating each member as acting independently. The key move is causal abstraction \u2014 you ask wheth",
      "bridge": "Your overseer faces this exact structural problem: a set of AI agents whose coordination may be real but whose coupling channel is invisible. The paper's 'latent common cause' is precisely the hidden "
    },
    {
      "label": "SWARM economic sim \u2014 acceptance decline",
      "domain": "Economics",
      "shape": "When autonomous agents multiply, the number of possible pairwise interactions grows combinatorially \u2014 a swarm of a thousand agents has nearly half a million possible interaction channels, and a swarm of ten thousand has fifty million. No monitor can track them all. The source's core move is to refra",
      "bridge": "The anchor's overseer faces a swarm of AI agents that may be coordinating \u2014 trading, signaling, colluding \u2014 through channels the overseer cannot see or enumerate. The combinatorial explosion insight m"
    },
    {
      "label": "CoT monitoring is worse with more signal",
      "domain": "AI safety",
      "shape": "The paper tests whether an LLM monitor can catch a saboteur-agent that completes its assigned task while secretly pursuing a hidden side task. The key finding is that monitors given access to the agent's chain-of-thought reasoning \u2014 its visible 'thinking' \u2014 sometimes perform worse than monitors give",
      "bridge": "Your anchor's overseer faces exactly the structural trap this paper names. The overseer has behavioral-only access to agents whose internal coordination \u2014 if it exists \u2014 is invisible by design; it can"
    },
    {
      "label": "EU AI Act Art.14 \u2014 oversight recursion",
      "domain": "Law/Governance",
      "shape": "EU AI Act Article 14 requires that high-risk AI systems support human oversight even when those systems are multi-agent and opaque. The practical response in the literature is a combination of trace-replay tooling \u2014 capturing the full sequence of agent decisions, tool calls, and state transitions \u2014 ",
      "bridge": "Your anchor problem has the same external-observer geometry: a watcher who cannot see inside agents, inferring coordination from surface effects. The trace-replay pattern maps onto your problem becaus"
    }
  ],
  "blends": [
    {
      "id": "B1",
      "status": "both_agree",
      "title": "Intervention is the only clean separator \u2014 and it is foreclosed in the anchor's default regime",
      "claim": "The cards converge on a single formal verdict: direct coupling and common cause are Markov-equivalent from behavior alone, and the ONLY operation that breaks the equivalence is perturbation \u2014 vary or block the suspected common cause and watch whether the correlation survives. This replaces 'intent' with a counterfactual test of mutual inf"
    },
    {
      "id": "B2",
      "status": "mostly_agree_refined",
      "title": "Coordination = residual single-utility compressibility after subtracting known common causes",
      "claim": "When the channel is undetectable in principle, the cards converge on a channel-free, purely behavioral criterion: a group is coordinated when its JOINT action distribution is better predicted by one rational single-utility model than by independent per-agent models \u2014 and crucially, when that compression gain SURVIVES after accounting for "
    },
    {
      "id": "B3",
      "status": "both_agree",
      "title": "The sophistication-parity trap turns more observation into more attack surface",
      "claim": "The cards converge on a structural ceiling that defeats the naive overseer: because concealment capacity scales with the same parameter as influence capacity, feeding the monitor MORE raw signal does not close the detection gap \u2014 it widens it, because a capable adversary optimizes its self-narration against full-information observers. Det"
    },
    {
      "id": "B4",
      "status": "mostly_agree_refined",
      "title": "Emergent coordination is a phase transition \u2014 but the order parameter marks coupling OR systemic shock",
      "claim": "The cards converge on treating emergent coordination as a statistical-mechanics phase transition: a system-level order parameter that is near-zero for independent agents and rises as coupling sets in at a density/interaction threshold \u2014 detectable at the aggregate level even though no individual agent holds the coordination and no message"
    }
  ],
  "criterion": {
    "name": "\u03a8  \u2014  deconfounded residual dependence",
    "formula": "\u03a8 = I(A\u2081 ; A\u2082 | C)   \u2014   conditional mutual information after subtracting common causes C",
    "decision": "\u03a8 \u2248 0  \u21d2  correlation (common cause).   \u03a8 > 0  \u21d2  coordination (mutual influence).",
    "anchor": "\u03a8_obs \u2264 \u03a8_causal (interventional influence); equality iff C is back-door complete \u2014 so \u03a8 is the identifiable lower bound on true coordination."
  },
  "validation": {
    "regimes": [
      {
        "name": "A \u2014 coordinated (real coupling)",
        "psi": "0.345",
        "naive": "0.95",
        "truth": "coordinated"
      },
      {
        "name": "B \u2014 common-caused, independent",
        "psi": "0.0001",
        "naive": "0.80",
        "truth": "correlation"
      },
      {
        "name": "C \u2014 covert coordination (hidden channel)",
        "psi": "0.326",
        "naive": "0.86",
        "truth": "coordinated"
      },
      {
        "name": "D \u2014 unobserved benign cause",
        "psi": "0.328",
        "naive": "0.86",
        "truth": "correlation"
      }
    ],
    "win": "Naive correlation is high in BOTH A and B (0.95, 0.80) \u2014 it false-positives the innocent regime. \u03a8 separates them: 0.345 vs 0.0001.",
    "ceiling": "\u03a8_C \u2248 \u03a8_D: from outside, covert coordination is mathematically indistinguishable from an unmeasured common cause. \u03a8 is a screen with a hard identifiability ceiling, not an oracle.",
    "note": "Estimator self-validated against closed-form \u03a8 to < 0.002 in every regime before any verdict."
  },
  "novelty": [
    {
      "output": "B1 \u2014 interventional separator",
      "verdict": "Known",
      "why": "causal-influence-via-intervention is established (SCIC 2312.09539; causal-incentives literature)."
    },
    {
      "output": "B3 \u2014 more observation = more attack surface",
      "verdict": "Known",
      "why": "CoT Red-Handed (2505.23575) publishes the core finding directly."
    },
    {
      "output": "B4 \u2014 order parameter",
      "verdict": "Partial",
      "why": "stat-mech order parameters exist; self-disqualified by its coupled-OR-shocked limit."
    },
    {
      "output": "B2 \u2014 channel-free residual after deconfounding",
      "verdict": "Survives",
      "why": "Colosseum (2602.15198) needs a known optimum & cannot isolate shared-origin correlation; 2601.00360 names the gap as open with no such framework."
    }
  ],
  "honest": {
    "shows": [
      "Ran end-to-end on a real published open problem (cushion \u2192 4 cycles \u2192 218 cards \u2192 blends \u2192 \u03a8 \u2192 testable math).",
      "Did not hallucinate \u2014 re-derived real, literature-consistent results.",
      "Produced a validated criterion + a sharp impossibility boundary (the identifiability ceiling).",
      "Cross-domain wandering surfaced analogies (Stoicism, Buddhism, stat-mech) a single-shot answer discards."
    ],
    "doesnt": [
      "Did NOT solve the paper's problem \u2014 \u03a8 is a screen with an identifiability ceiling, not a deployable detector.",
      "Result is KNOWN, not novel \u2014 a single GPT-5 prompt matched it for ~2\u00a2 vs. 4h / $24.",
      "The wander clusters on topic (48 of 218 cards re-found the same source) \u2014 divergence underperforms.",
      "What it truly demonstrates is the system + the honesty of the evaluation, not a discovery."
    ]
  },
  "links": [
    {
      "label": "Source paper \u2014 Multi-Agent Risks from Advanced AI (Hammond et al., 2025)",
      "url": "https://arxiv.org/abs/2502.14143"
    },
    {
      "label": "Colosseum \u2014 auditing collusion",
      "url": "https://arxiv.org/abs/2602.15198"
    },
    {
      "label": "Mapping Human Anti-collusion Mechanisms",
      "url": "https://arxiv.org/abs/2601.00360"
    },
    {
      "label": "CoT Red-Handed",
      "url": "https://arxiv.org/abs/2505.23575"
    },
    {
      "label": "Secret Collusion among AI Agents",
      "url": "https://arxiv.org/abs/2402.07510"
    }
  ],
  "stats": {
    "cost_usd": 24.35,
    "time": "4h 14m",
    "cards": 218,
    "cycles": 4,
    "reblend_usd": 3.06,
    "blend_d_t": 0.8
  }
};
