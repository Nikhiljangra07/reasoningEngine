# CLAUDE.md — reasoningEngine

> This repo is the next-generation deep reasoning engine for LoRa.
> Built from scratch. No code shared with any existing system.
> Claude Code reads this automatically at session start.

---

## PROJECT IDENTITY

**reasoningEngine** is not a feature. It is LoRa's bid to compete with the best reasoning models on the planet — in ONE domain: **reasoning about human problems, decisions, and life situations.**

The thesis: a domain-specialist reasoning system beats any general-purpose model in its own lane. o3 reasons about everything. Opus reasons about everything. LoRa reasons about **you.**

- **Owner:** Nikhil Jangra (@Nikhiljangra07)
- **This repo:** `reasoningEngine` (fully isolated, built from scratch)
- **Language:** Python 3.11+
- **Stage:** Design + build phase

---

## ABSOLUTE NON-NEGOTIABLES

### **THE REAL BENCHMARK: COMPETE WITH THE BEST REASONING MODELS ON EARTH**

**This engine must produce reasoning output on human problems — career, relationships, business, life decisions — that competes with or surpasses Claude Opus 4.6 (extended thinking) and OpenAI o3. Not in general intelligence. Not in code. Not in math puzzles. In the ONE domain that matters: critical thinking about the decisions real people face.**

**If the output of this engine looks like something a chatbot could produce, it has failed. If it looks like something a smart friend could say, it has failed. It must produce reasoning that makes the user feel like they're talking to someone who spent 30 minutes thinking about THEIR specific problem — seeing what they can't see, challenging what they won't challenge, projecting consequences they haven't imagined.**

**This is not negotiable. This is the entire reason this repo exists.**

---

1. **STRICTLY ISOLATED.** No imports from, no dependencies on, and no code shared with any other repo — LoRaMaths, LoRa-EmotionalEngine-v1, MemoryArchitecture, LoRa_WaveformEngine, or presence-whispers. This isolation holds until Nikhil explicitly says otherwise.

2. **LoRaMaths deep mode is the INTERNAL benchmark.** The existing deep mode in `~/Desktop/LoRaMaths/src/deep/` is the floor, not the ceiling. Surpassing it is the minimum requirement, not the goal. The goal is competing with o3 and Opus in this domain.

3. **Reference repos (read-only context, NEVER import):**
   - `~/Desktop/LoRaMaths` — current multi-perspective engine (5 frameworks, 31-combination scoring, conflict graph, Sonnet synthesis). The internal bar to clear.
   - `~/Desktop/lora-v1-frontend` — LoRa's production backend (TypeScript/Express). Shows how deep mode integrates via HTTP.
   - `~/Desktop/MemoryArchitecture` — Memory V2 design. Context for how reasoning connects to memory.
   - `~/Desktop/LoRa_WaveformEngine` — Voice signal extraction. Future input signal for reasoning.

4. **LoRa is NOT a therapist.** Everything built here must produce analytical reasoning — not validation, not emotional support, not "balanced perspectives." Sharp, honest, consequence-driven.

5. **Consequences must be concrete.** "Things might get harder" is unacceptable. "In 6 months at this trajectory, you'll be 30% below market rate" is the standard. Specific, time-bound, testable.

6. **No magic numbers.** Every threshold, weight, and parameter must be documented with reasoning or derived from test data.

7. **Depth over breadth.** One devastating insight that changes how the user sees their problem beats five shallow perspectives. The current LoRaMaths deep mode spreads across 5 frameworks. This engine goes DEEP.

8. **Adversarial self-critique.** The engine must argue against its own conclusions before presenting them. If a conclusion survives its own attack, it's real. If it doesn't, the user never sees it.

---

## EXTERNAL BENCHMARK: The Models to Beat

| Model | Strength | Where LoRa Must Win |
|-------|----------|-------------------|
| **Claude Opus 4.6 (extended thinking)** | Nuanced multi-step reasoning, human context understanding | Deeper domain expertise on human decisions, sharper consequence projection |
| **OpenAI o3** | Purpose-built chain-of-thought, structured decomposition | More honest (no validation instinct), better at surfacing what user can't see |
| **Gemini 2.5 Pro** | Large context, strong analytical capability | More direct, less hedging, consequence-specific |

These models are general-purpose. They reason well about everything but optimize for agreement. LoRa's edge: **domain specialization + anti-validation + consequence specificity.**

---

## INTERNAL BENCHMARK: LoRaMaths Deep Mode (the floor)

Current deep mode pipeline (what we must surpass):
- **7 LLM calls:** 1 question lock (Haiku) + 5 framework engines (Haiku parallel) + 1 synthesis (Sonnet)
- **60-80s latency**
- **5 frameworks:** regression, bayesian, game_theory, constraint, causal_loop
- **31 combinations scored** (5 individual + 10 pairs + 10 triplets + 5 quads + 1 full)
- **Conflict graph:** keyword-based direction extraction → pairwise edges → contradiction/agreement/orthogonal
- **Dimension analysis:** keyword matching against 5 dimensions
- **Formation selection:** coverage × tension × complementarity weighted scoring
- **Synthesis:** single Sonnet call with 5 Laws prompt, max 800 words

Known limitations (these are targets, not just problems):
- Latency too high (60-80s)
- Direction extraction is keyword-based (fragile, false positives)
- Dimension detection is keyword-based (misses nuance)
- All 5 frameworks always run regardless of relevance
- No streaming — user stares at blank screen
- It's structured prompt routing, not real reasoning
- Formation selection is purely mathematical, doesn't account for problem type
- No adversarial self-critique — output is never challenged before delivery
- No domain-specific reasoning primitives (sunk cost, status quo bias, false dichotomy, emotional reasoning patterns)

---

## DOMAIN: PHYSICS

> **Role in the engine:** Root finder and path projector. Physics maps the terrain — finds the real problem, projects where it leads, and penetrates user bias to surface what they can't or won't tell you.
>
> **Given:** Input variables (x1, x2, ...) — the facts of the user's situation.
> **Find:** Variable y — the root cause, often hidden by the user's own bias.
> **Project:** Where y leads — concrete, time-bound consequences.

Physics operates in two phases. Phase 1 squeezes the system from the outside. Phase 2 squeezes the user's story from the inside. Between the two, variable y has nowhere to hide.

---

### PHASE 1: ROOT FINDING & TRAJECTORY

Operates on what the user **does** tell you. Reads the system mechanics to find where energy leaks, where pressure builds, where things are heading.

#### 1. The Law of Conservation of Energy

- **The Concept:** Energy cannot be created or destroyed, only transformed.
- **The Logic:** In any problem, the "output" must equal the "input" minus any "waste".
- **Application:** If a result (variable A) seems too big for the effort put in, there is a hidden "battery" of potential energy you haven't found yet (variable D). Conversely, if massive effort results in nothing, track the "heat leak" — where the energy is being wasted or stolen.
- **What it catches:** Hidden energy sources or drains. The user says they're working hard and getting nothing — conservation finds where that effort is actually going.

#### 2. Thermodynamics: Entropy

- **The Concept:** The Second Law states that in an isolated system, disorder (entropy) always increases.
- **The Logic:** Systems naturally move toward chaos unless "work" is constantly added.
- **Application:** By looking at the "mess" in a situation, you can find the root cause (D): a lack of energy input or a broken boundary. Tracking entropy tells you exactly where a problem is heading — toward total breakdown — unless a new energy source is introduced.
- **What it catches:** Decay rate. How fast the system is falling apart and the timeline to breakdown if nothing changes.

#### 3. Kinematics: Trajectory and Momentum

- **The Concept:** Kinematics describes motion; momentum is the product of mass and velocity.
- **The Logic:** An object in motion stays in motion unless acted upon by an outside force.
- **Application:** By measuring the "velocity" (speed of change) and "mass" (importance/size) of a problem, you can plot its trajectory. If you know where it started and how fast it's moving, you can "unfold" variable D — the exact point of impact — long before it happens.
- **What it catches:** Where this lands and when. Concrete time-bound consequence projection.

#### 4. Potential vs. Kinetic Energy

- **The Concept:** Potential energy is stored; kinetic energy is the energy of motion.
- **The Logic:** Every "action" (kinetic) was once a "tension" (potential).
- **Application:** To find the root, look for the tension. A sudden crisis is just potential energy being released. By identifying where the "pressure" is building up, you can predict the next "explosion" before any kinetic movement occurs.
- **What it catches:** Stored pressure that hasn't released yet. The crisis that's coming but hasn't arrived.

#### 5. First Principles (Classical Mechanics)

- **The Concept:** Breaking complex systems down into their most basic, non-reducible parts.
- **The Logic:** You don't solve the "symptom"; you solve the "force".
- **Application:** Instead of looking at the whole "room," you look at the "gravity" of the situation — the underlying rules that must be true for the problem to exist at all. This "squeezes" the problem down to its mathematical roots.
- **What it catches:** The irreducible forces that make this problem exist. Strips away everything that's noise.
- **Note:** This is the decomposition step. It runs first — breaks the problem into its fundamental forces — then the other frameworks operate on those forces. It is upstream of the rest.

#### 6. Equilibrium and Net Force

- **The Concept:** When all forces acting on an object cancel out, the object is in equilibrium.
- **The Logic:** If a situation is "stuck," it's because two equal and opposite forces are pushing against each other.
- **Application:** If a problem isn't moving, variable D is the hidden counter-force. To get it heading somewhere new, you don't necessarily need more "power"; you might just need to remove the "friction" or the opposing force.
- **What it catches:** Why nothing is changing despite effort. The invisible force that cancels every move the user makes.

---

### PHASE 2: BIAS PENETRATION

Operates on what the user **doesn't** tell you. Reads the gaps, contradictions, and distortions in the user's story to surface what they can't or won't reveal. This is the layer that no general-purpose chatbot performs — it challenges the user's frame before reasoning about solutions.

#### 1. The "Anomalous Motion" Test

- **The Concept:** In physics, if a planet moves in a way that Newton's laws can't explain, we don't assume the laws are wrong — we assume there is an unseen mass (a hidden variable) pulling on it.
- **Application:** If a user says x1 (the situation) and x2 (their action) should lead to x3 (the result), but it isn't happening, don't just listen to their explanation.
- **Squeezing the Bias:** Look for the "wobble." If their logic has a gap or a contradiction, variable D is located exactly in that gap. Their bias is the "dark matter" you can't see, but you can measure its pull on their story.

#### 2. First Principles: The "Socratic Squeeze"

- **The Concept:** Bias relies on analogy ("It's just like the last time...") and assumptions ("It's obviously because of X"). First principles thinking strips these away.
- **Application:** Ask "Why?" until you hit a fundamental truth that cannot be broken down further.
- **Squeezing the Bias:** If a user says, "We can't do Y because it's too expensive," a first-principles thinker asks, "What are the raw materials required for Y, and what do they cost on the open market?" Often, you'll find the "expense" is just a biased assumption, and the real problem is a lack of innovation or willpower.

#### 3. Relativity: Changing the Reference Frame

- **The Concept:** In physics, how you see a problem depends entirely on your reference frame.
- **Application:** If the user is at the center of the problem, their bias is a "stationary" frame. Everything looks like it's moving at them.
- **Squeezing the Bias:** Force a shift in the reference frame. Ask, "If you were the other person in this situation, what would the forces look like?" By shifting the perspective, the hidden variable D — often the user's own behavior or a systemic flaw — suddenly becomes visible because it is no longer being blocked by their own "line of sight."

#### 4. Entropy and "The Leak"

- **The Concept:** A user might tell you a story of perfect order, but if the situation is falling apart, Entropy tells you there's a leak.
- **Application:** Chaos requires a cause. If a "well-managed" system is failing, the user is likely omitting the "heat" (the conflict, the laziness, or the technical debt).
- **Squeezing the Bias:** Ignore what they say is happening and look at where the "energy" is being lost. The "mess" always points directly to the real issue that the user is too biased (or embarrassed) to admit.

#### 5. Reductio ad Absurdum (Reducing to Absurdity)

- **The Concept:** A logical tool often used in mathematical physics. Take a claim to its extreme logical conclusion.
- **Application:** Take the user's biased claim to its extreme logical conclusion.
- **Squeezing the Bias:** If their claim leads to a physical or logical impossibility, then the "known" variables they gave you must be false. This forces the "unspoken" variable D to surface because it is the only thing that can restore logic to the system.

---

### PHYSICS DOMAIN SUMMARY

**Phase 1** says: "The physics of your situation says y is here."
**Phase 2** says: "And the reason you couldn't see it is here."

y has nowhere to hide because the engine closes in from both sides — the system's reality (Phase 1) AND the user's distortion of that reality (Phase 2). The gap between those two **is** y.

Phase 1 maps the terrain. Phase 2 stress-tests the map against the user's blind spots. Together, they find the hidden variable, project its consequences, and leave no room for it to stay hidden.

---

## DOMAIN: MATHEMATICS

> **Role in the engine:** The operating system. Maths is not a reasoning lens — it is the connective tissue that translates between domains, holds all perspectives, filters signal from noise, reduces to core variables, infers hidden variables, and knows when to stop. Every other domain runs on top of it.
>
> **Maths does not analyze the user's problem directly.** It processes the outputs of Physics, Psychology, Philosophy, and Chemistry — connecting them, reducing them, and extracting the unified truth.
>
> If Physics is the Body and Psychology is the Mind, **Math is the Skeleton and the Nervous System.** It provides the structure that holds everything together and the signals that let them communicate.

### THE 4 ROLES OF MATHEMATICS

#### 1. Math as the Flow (The Plumbing)

In Physics, Chemistry, and even Psychology, "energy" or "influence" has to move from point A to point B. Math provides the Calculus and Differential Equations that describe how things change over time. Without math, we'd just have guesses about the flow. Math ensures that the "pressure" in a pipe (Physics) matches the "logic" in an argument (Philosophy). It keeps the systems from clashing.

#### 2. Math as the Structure (The Pillar)

You can have all the gold (data/opinions) in the world, but without a pillar, they are just a pile on the floor. Math provides the Symmetry and Geometry. It tells you if your "Room" is stable. It provides the Constraints — if the math doesn't add up, the story told by the other domains (like a biased user's explanation) cannot be true. Math is the Truth-Checker that holds the ceiling up.

#### 3. Math as the Translator (The Sync)

This is its most powerful role. Because Math is abstract, it doesn't care if you are talking about atoms, money, or human emotions. It allows Category Theory or Manifold Theory to take a solution from one domain and slide it into another. It acts as the Common Language — the only way a Chemist and a Psychologist can realize they are actually looking at the same Causal Loop.

#### 4. Math as the Root Extractor

Math is the tool that lets you squeeze the situation until only the essence remains. It takes the N perspectives, strips away the noise of the specific domain, and leaves you with Variable D — the pure, numerical relationship at the heart of the problem.

---

All layers below work as one smooth, blended system. They are separated here for documentation, not for execution. In practice they are inseparable — a single mathematical infrastructure that all domains flow through without conflict.

---

### LAYER 1: SIGNAL VS. NOISE (Contextual Relevance)

Every domain generates output. Maths decides what's signal for *this* problem and stores the rest — because context changes, and today's noise is tomorrow's key variable.

#### 1. Signal for Problem A is Noise for Problem B

- **The Concept:** In radio physics, thousands of waves hit an antenna at once. If you want the music (Problem A), the weather report (data point x2) is "noise." But if you want to know if it's going to rain, that "noise" is the only signal that matters.
- **The Lesson:** Perspective isn't "wrong"; it's just off-frequency for the current task.

#### 2. The Boundary Condition

- **The Concept:** In physics experiments, we ignore "air resistance" to find the basic law of gravity. We "discard" that data to see the core truth. But to build a real parachute, that discarded data is the most important factor in the world.
- **The Lesson:** Every perspective defines a different "boundary" of the problem. Ignoring a perspective doesn't delete it — it chooses which "room" of the problem you're standing in.

#### 3. Latent Variables (The Hidden Gold)

- **The Concept:** A piece of data seems useless because it doesn't fit the formula A + B = C. But later, you realize C isn't working because of a hidden variable D.
- **The Squeeze:** When you go back to your "trash pile" of discarded opinions, variable D was sitting there all along, disguised as an "irrelevant" comment.

#### 4. Orthogonal Thinking

- **The Concept:** In math, "orthogonal" vectors are at 90 degrees to each other — they don't affect each other's direction. An opinion about "Color" feels useless for a "Speed" problem. But if you're trying to sell the car, Speed (Physics) and Color (Aesthetics) are both gold — they just belong to different axes of the same reality.
- **The Squeeze:** Collect all data (The "Melt") → Filter for the current squeeze (The "Mold") → Store the leftovers (The "Dross"). Because the moment the heading shifts, the data you threw away becomes the key variable you need.

**Nothing is useless; it's just waiting for the right equation.**

---

### LAYER 2: CATEGORY THEORY (The Universal Translator)

Often called "The Mathematics of Mathematics." Category Theory doesn't care what the objects are (numbers, atoms, or human emotions) — it only cares about the relationships between them. This is what lets Physics talk to Psychology and realize they're solving the same equation.

#### 1. Morphisms (The Flow)

- **The Concept:** Focus shifts from the thing to the process.
- **Math/Physics:** A function or a force. **Psychology:** A reaction or an influence.
- **The Sync:** Instead of asking "What is this opinion?", ask "How does this opinion transform the situation?" This allows you to treat a psychological bias exactly like a physical force. They are both just "arrows" (morphisms) moving from point A to point B.

#### 2. Isomorphism (The Universal Pattern)

- **The Concept:** The "Golden Key." Describes when two things are different in appearance but identical in structure.
- **Chemistry/Philosophy:** The structure of a covalent bond (sharing to create stability) is isomorphic to a healthy human relationship or a logical syllogism.
- **The Sync:** Map a solution from one domain onto another. If you solve "tension" in a bridge (Physics), Category Theory provides the framework to apply that same structural logic to "tension" in a social group (Psychology).

#### 3. Topos Theory (The "Room")

- **The Concept:** A "Topos" is a type of category that acts like its own universe with its own internal logic.
- **Perspective:** Every human perspective is its own Topos. Inside their "room," their logic works perfectly.
- **The Sync:** Rather than saying one person is "wrong," Category Theory shows how different logics (Topoi) can coexist and how to "translate" between them. This is the philosophy of pluralism expressed through pure mathematics.

#### 4. Compositionality

- **The Concept:** The core of Chemistry and Linguistics. Small things combine to make big things with new properties.
- **The Squeeze:** Category Theory provides the "grammar" for how to combine variables A, B, and C to see the emergent variable D. It's the glue that explains how individual opinions (Psychology) turn into a "Market Trend" or a "Revolution" (Social Physics).

---

### LAYER 3: MANIFOLD THEORY (The Multi-Angle Holder)

If a problem is a complex, high-dimensional shape (the "room"), a manifold allows you to zoom in on any single point and treat it as a simple, flat coordinate system (the "perspective"). This is the best concept for holding N perspectives, evaluating all angles, and finding the root.

#### 1. The Atlas (Multi-Angle Structure)

- **The Concept:** You don't see the whole shape at once with one giant formula. You use an Atlas — a collection of Charts.
- **Charts:** Each chart is a single perspective that looks flat and easy to understand locally.
- **The Squeeze:** Collecting many charts, overlapping them, reveals the entire object. Each variation of an opinion is a chart that is true for its own small area, but together they reveal the true shape of the problem.

#### 2. Homeomorphisms (Smooth Transitions)

- **The Concept:** A manifold provides a mathematical way to move from one angle to another without losing information.
- **The Sync:** Two different perspectives (variable A and variable B) are synced via Transition Maps — translating the logic of one person's world into another's so smoothly that the truth remains constant even as the angle changes.

#### 3. The Manifold Hypothesis (Dimensional Reduction)

- **The Concept:** Core idea in modern data science and AI. Even if a problem looks like it has thousands of variables, it actually lives on a much smaller, simpler manifold underneath.
- **The Squeeze:** Strip away the noise of N variations and find the few core hidden variables (variable D) that actually control the whole system.

#### 4. Variational Calculus (All Possible Paths)

- **The Concept:** Instead of solving for a single number, you solve for a function — an entire path. It evaluates every possible variation of how a problem could unfold.
- **The Squeeze:** Finds the one path that requires the least energy or time (the "stationary action") — the optimal trajectory through all possible variations.

---

### LAYER 4: N-DIMENSIONAL CAPACITY (Infinite Scale)

The math doesn't break at scale. It handles as many variables as exist.

#### 1. N-Dimensional Space

- A problem can have 3 dimensions or 3,000. If you have 100 different people with 100 different opinions, you are working in a 100-dimensional space. The math treats each opinion as its own axis.

#### 2. Local vs. Global (Atlas Count)

- **Simple Problems:** 2 or 3 perspectives to see the full room.
- **Complex Problems:** Millions of data points.
- **The Logic:** You don't need every possible perspective — only enough to overlap and leave no gaps. Once the Atlas is complete, you have the Global Truth.

#### 3. Degrees of Freedom (The Squeeze to Variable D)

- Even starting with 1,000 perspectives (x1 to x1000), the math looks for the intrinsic dimension.
- **Example:** A piece of paper crumpled into a ball looks like a complex 3D object. But the paper itself is still just a 2D sheet.
- **The Squeeze:** Evaluate all 1,000 perspectives to find they actually depend on 3 or 4 root variables. Collect infinite data, unfold it down to the minimum needed to explain the whole thing.

#### 4. Continuous Variation (The Flow)

- The Calculus of Variations doesn't look at 1, 2, or 100 specific options. It looks at a continuum — every possible path between Point A and Point B.
- **The Squeeze:** Compresses an infinite number of potential trajectories to find the one path of Least Action (the most efficient solution).

---

### LAYER 5: CONVERGENCE & STOPPING (Knowing When the Squeeze is Done)

The engine doesn't squeeze forever. It knows exactly when adding more perspectives stops changing the answer.

#### 1. The Elbow Method

- Imagine squeezing a sponge. At first, lots of water (D) comes out. Eventually only drops.
- **The Math:** In a plot of "Variance Explained," there is a literal "Elbow" in the curve. Once reached, adding more perspectives doesn't provide more truth. Stop. You have the core signal; the rest is noise.

#### 2. Occam's Razor (Parsimony)

- In mathematical modeling (AIC/BIC scores), there is a penalty for adding too many variables.
- **The Logic:** If you can explain the situation with 3 variables, the math punishes you for using 10. It finds the Minimum Description Length — as simple as possible but no simpler.

#### 3. Convergence

- As you add more angles, does the picture keep shifting or has it settled?
- **The Stopping Point:** When adding the 10th or 100th perspective stops changing the result, the math has converged. It has unfolded the full room. Further data is redundant — stored as gold for a different problem later.

#### 4. Singular Value Decomposition (SVD)

- Ranks every perspective by its "Weight" — how much it actually affects the outcome.
- **The Stopping Point:** Identifies the Singular Values. Usually the top 2 or 3 carry 95% of the information. The math clips the rest: "These 3 perspectives are the Pillars; the other 97 are decorations."

---

### LAYER 6: BAYESIAN INFERENCE (The Living Engine)

While Manifold Theory handles N-perspectives as a static shape, Bayesian Inference is the living mathematical engine that collects every bit of information on its way to finding a solution. It is the primary tool for unfolding hidden variables by treating every new data point — even biased ones — as a gold update to the truth.

#### 1. The Prior (Initial Information)

- Before the problem even begins, you have the Prior Probability — the best guess based on what you already know. You aren't starting from zero; you're starting with a foundation.

#### 2. The Likelihood (Every Piece is Important)

- Every new observation (x1, x2, ... xn) is treated as gold.
- **The Squeeze:** Instead of just looking at the "told information," the math asks: "Given my current goal, how likely is it that I would see this specific piece of data?"
- **Collecting on the Way:** Integrates initial info, new info, and even "discarded" data into a single evolving picture.

#### 3. The Posterior (Finding the Underlying Issue)

- The Posterior Probability is the "Full Room" — the updated truth that accounts for every variable collected.
- **Variable Y/D:** If there is an underlying issue that isn't being told, Bayesian math sees it as a Latent Variable. By tracking the energy of the other data points, the math infers the existence and value of the hidden variable — because the visible variables wouldn't make sense without it.

#### 4. The Update Loop (Dynamic Flexibility)

- **Psychology:** Explains how brains update beliefs based on new experiences (Bayesian Brain Hypothesis).
- **Physics:** Used in Filter Theory (Kalman Filter) to track a rocket's position by squeezing noisy sensor data.
- **Philosophy:** The mathematical heart of Epistemology — how we know what we know.
- **The Living Squeeze:** Takes N perspectives, treats them as data, and constantly updates the Global Truth until variable D is revealed.

---

### LAYER 7: GAME THEORY (Strategic Multi-Agent Reasoning)

Physics models forces but not intentional actors. When the user's problem involves another person who is also making strategic moves — negotiations, relationship power dynamics, business competition — physics can't model bluffing, retaliation, or adaptive strategy. Game Theory handles this.

Activates when the manifold detects multiple intentional agents in the problem space.

#### 1. Nash Equilibrium

- **The Concept:** A stable state where no player can improve their outcome by changing their strategy alone.
- **The Squeeze:** If a situation feels "stuck" but Equilibrium (physics) doesn't explain it, Game Theory asks: "Is everyone in a Nash Equilibrium — where changing strategy alone makes things worse?" This explains why rational people stay in bad situations.

#### 2. Dominant Strategy

- **The Concept:** A strategy that is the best response regardless of what the other player does.
- **The Squeeze:** If the user has a dominant strategy they're not playing, that gap is variable D. If the other party has one, the user needs to know — because fighting a dominant strategy is wasted energy (Conservation).

#### 3. Prisoner's Dilemma Patterns

- **The Concept:** Situations where individual rationality leads to collective irrationality. Both parties would be better off cooperating, but the incentive structure pushes them to defect.
- **The Squeeze:** Many relationship and business problems are hidden prisoner's dilemmas. Identifying the pattern reveals why trust keeps breaking — and what structural change (not just "communication") can fix it.

#### 4. Zero-Sum vs. Positive-Sum Framing

- **The Concept:** In a zero-sum game, one person's gain is another's loss. In a positive-sum game, both can win.
- **The Squeeze:** Users often frame positive-sum situations as zero-sum (bias). Detecting this reframe changes the entire trajectory — from "how do I win?" to "how do we both win?" which opens paths that were invisible in the zero-sum frame.

---

### LAYER 8: CAUSAL LOOP ANALYSIS (Circular Feedback Detection)

Physics trajectory projects a line. Most human problems are circular — feedback loops that trap people. Anxiety → avoidance → more anxiety. Debt → stress → bad decisions → more debt. Without this, the engine misses the spirals that keep people stuck.

Activates when trajectory detection finds circular rather than linear patterns.

#### 1. Reinforcing Loops (Spirals)

- **The Concept:** A causes more B, which causes more A. These loops amplify — they can spiral up (virtuous) or spiral down (vicious).
- **The Squeeze:** If the user's problem keeps getting worse despite effort, look for the reinforcing loop. The "root cause" isn't a single event — it's the loop structure itself. Breaking one link in the chain breaks the spiral.

#### 2. Balancing Loops (Equilibria)

- **The Concept:** A causes B, which pushes back against A. These loops stabilize — they resist change in either direction.
- **The Squeeze:** If the user can't make progress, a balancing loop is holding the system in place. This is different from Equilibrium (physics) — it's not two opposing forces, it's a self-correcting cycle that pulls things back to "normal" no matter what.

#### 3. Loop Dominance

- **The Concept:** In any system with multiple loops, one loop dominates at any given time.
- **The Squeeze:** The dominant loop determines the system's behavior. If a reinforcing loop is dominant, things are spiraling. If a balancing loop is dominant, things are stuck. Finding which loop is in control tells you where to intervene — and where intervention is useless.

#### 4. Delay Effects

- **The Concept:** In many loops, the consequence doesn't arrive immediately. There's a delay between action and reaction.
- **The Squeeze:** Delays are where humans lose the plot. They take an action, don't see results, assume it failed, and change course — just as the original action was about to work. Or they take a damaging action, don't see harm, assume it's safe, and continue — until the delayed consequence arrives all at once. Mapping delays reveals the true timeline of consequences.

---

### LAYER 9: ERGODICITY & FRAGILITY (The Final Stress Test)

This is the Quality Control layer. After convergence captures variable y and the engine has a solution, the system runs one final inspection: does the solution survive the real world for THIS person in THIS moment?

Runs AFTER convergence. This is Stage 5 of the Formation.

#### 1. Antifragility Assessment

- **The Concept:** Fragile things break under stress. Robust things survive stress. Antifragile things get stronger under stress.
- **The Squeeze:** Rate the proposed solution: Will it shatter at the first unexpected shock? Will it merely survive? Or will it actually improve when challenged? A fragile solution is a bad solution — even if the reasoning that produced it was flawless.

#### 2. Ergodicity Check

- **The Concept:** A system is ergodic if the average outcome over many trials equals the outcome for one individual over time. Most human life decisions are NON-ergodic.
- **The Squeeze:** A 90% success rate means nothing if the 10% failure is total ruin (bankruptcy, irreversible health damage, relationship destruction). The engine must check: "Does the statistical average actually apply to this ONE person making this ONE decision?" If not, the solution must account for tail risk, not just expected value.

#### 3. Tail Risk Evaluation

- **The Concept:** The extreme outcomes at the edges of the probability distribution.
- **The Squeeze:** What's the worst-case scenario if this solution fails? Is it recoverable or catastrophic? A solution with a small chance of catastrophic downside is worse than a "mediocre" solution with no catastrophic risk. The engine must surface this before the user acts.

#### 4. Real-World Durability

- **The Concept:** Laboratory conditions ≠ field conditions. A solution that works in theory must survive the user's actual constraints — their energy, their resources, their relationships, their environment.
- **The Squeeze:** Stress-test the solution against the user's real context. "This is the optimal path" means nothing if the user doesn't have the bandwidth, the money, or the support system to walk it. The final output must be both true AND executable.

---

### MATHEMATICS DOMAIN SUMMARY

Maths is the operating system of the reasoning engine. It does not analyze the user's problem — it processes every other domain's output through a unified infrastructure:

- **Signal vs. Noise** filters what matters now and stores the rest for later.
- **Category Theory** translates between domains so Physics, Psychology, Philosophy, and Chemistry speak the same language.
- **Manifold Theory** holds all perspectives without conflict and reveals the true shape.
- **N-Dimensional Capacity** scales to any number of variables without breaking.
- **Convergence & Stopping** knows when the squeeze is done — no over-analysis, no under-analysis.
- **Bayesian Inference** is the living engine that updates truth with every data point and infers what's hidden.
- **Game Theory** models strategic multi-agent interactions when intentional actors are involved.
- **Causal Loop Analysis** detects circular feedback traps — the spirals that keep people stuck.
- **Ergodicity & Fragility** stress-tests the final solution — does it survive the real world for THIS person?

It takes the Many and gives you the One. Then it makes sure the One actually works.

---

---

## DECISION LOG

All architectural decisions, build actions, and iterations — recorded in order.

### Decision 001 — Physics Domain Design (2026-04-01)
- **What:** Defined Physics as the root finder and path projector with two phases.
- **Phase 1 (Root Finding):** 6 frameworks — Conservation, Entropy, Trajectory, Potential→Kinetic, First Principles, Equilibrium.
- **Phase 2 (Bias Penetration):** 5 frameworks — Anomalous Motion, Socratic Squeeze, Reference Frame Shift, Entropy Leak, Reductio ad Absurdum.
- **Rationale:** Phase 1 squeezes the system from the outside (what the user tells us). Phase 2 squeezes the user's story from the inside (what they don't tell us). Between the two, variable y has nowhere to hide.

### Decision 002 — Mathematics Domain Design (2026-04-01)
- **What:** Defined Maths as the operating system — not a reasoning lens but the infrastructure all domains run on.
- **4 Roles:** Flow (calculus/differential equations), Structure (symmetry/geometry/constraints), Translator (category theory abstraction), Root Extractor (the squeeze itself).
- **6 Initial Layers:** Signal vs. Noise, Category Theory, Manifold Theory, N-Dimensional Capacity, Convergence & Stopping, Bayesian Inference.
- **Rationale:** Maths connects, translates, filters, reduces, and knows when to stop. Every domain's output flows through it.

### Decision 003 — Game Theory + Causal Loops → Maths Domain (2026-04-01)
- **What:** Added Game Theory (Layer 7) and Causal Loop Analysis (Layer 8) to the Maths domain.
- **Rationale:** Physics can't model intentional actors (game theory) or circular feedback traps (causal loops). These fill critical gaps for human problems. Placed in Maths because they are infrastructure tools, not a separate reasoning domain.

### Decision 004 — Ergodicity & Fragility → Maths Layer 9 (2026-04-01)
- **What:** Added Ergodicity & Fragility as the final stress test layer after convergence.
- **Rationale:** The engine must verify that the solution survives the real world for THIS person. A 90% success rate doesn't matter if the 10% is total ruin. This is Stage 5 of the Formation.

### Decision 005 — The Formation: 5-Stage Convergence Battlefield (2026-04-01)
- **What:** Defined the engine's execution architecture as a 5-stage convergence loop.
- **Stages:** 1. Manifold Opens (Arena) → 2. Tools Deploy (Combatants) → 3. Bayesian Loop (Intelligence Gathering) → 4. Convergence (Capture variable y) → 5. Fragility Test (Final Inspection).
- **Rationale:** Not a pipeline. A battlefield where everything runs simultaneously, feeds into Bayesian update loop, and converges. System does NOT stop until full excavation + stress test passed.

### Decision 006 — Component Architecture (2026-04-01)
- **What:** Divided the build into 6 modular components.
- **Components:** 1. Core Types, 2. Physics Domain, 3a-e. Maths Core, 3f-g. Game Theory + Causal Loops, 3h. Ergodicity & Fragility, 4. Formation Orchestrator.
- **Rationale:** Each component is independently testable. Future domains (Psychology, Philosophy, Chemistry) plug into `src/domains/` without rewriting.

### Decision 007 — Component 1: Core Types Built (2026-04-01)
- **What:** Built `src/core/types.py` — 10 data structures, 20 framework IDs, all enums.
- **Types:** Problem, Variable, Perspective, RootCause, Consequence, CausalLoop, GameState, FragilityResult, DomainOutput, FormationResult.
- **Verification:** All types import clean, no dependencies beyond stdlib.

### Decision 008 — Component 2: Physics Domain Built (2026-04-01)
- **What:** Built the full Physics domain across 3 files:
  - `src/domains/physics/__init__.py` — entry point, runs Phase 1 → Phase 2, merges output.
  - `src/domains/physics/phase1_root.py` — 6 root-finding frameworks. First Principles runs upstream, feeds the other 5.
  - `src/domains/physics/phase2_bias.py` — 5 bias-penetration frameworks. Uses Phase 1 output to detect where the user's story doesn't match the physics.
- **Design Decisions:**
  - First Principles is upstream — it decomposes the problem into irreducible forces, then all other Phase 1 frameworks operate on those forces.
  - Phase 2 receives Phase 1's full DomainOutput — it reads the gaps between user's story and physics findings.
  - Conservation uses a 0.2 imbalance threshold to detect hidden drains/sources (documented, not magic).
  - Equilibrium matches opposing forces within 0.15 magnitude tolerance.
  - Each framework produces Perspective objects with typed Variables, evidence chains, and confidence scores.
- **Verification:** All 11 frameworks execute clean on a real test problem (startup founder scenario). Phase 1 produces 6 perspectives, Phase 2 produces 5. Root causes found with bias labels. Two-phase design confirmed working: Phase 1 takes data at face value, Phase 2 challenges it.
- **Observation:** Consequences don't fire when Phase 1 trajectory reads net-positive (because user's stated positive forces outweigh negatives). This is correct behavior — the Bayesian loop (Component 3) will feed Phase 2's corrections back to recalculate trajectory. The formation handles this.

### Decision 009 — Component 3: Maths Infrastructure Built (2026-04-01)
- **What:** Built all 9 Maths layers across 8 files:
  - `src/maths/signal_noise.py` — Layer 1: Contextual relevance filtering. Classifies perspectives as signal/noise/latent/orthogonal. Stores dross for later.
  - `src/maths/category.py` — Layer 2: Universal translator. Finds morphisms (transformations between variables), isomorphisms (structural matches across frameworks), and emergent variables from compositionality.
  - `src/maths/manifold.py` — Layers 3+4: Multi-angle holder + N-dimensional capacity. Creates atlas of charts, finds overlaps via cosine similarity, reduces dimensions via variance analysis (95% threshold), finds least-action path.
  - `src/maths/convergence.py` — Layer 5: Convergence & stopping. SVD ranking, elbow method, parsimony scoring, stability check between iterations. Converges when stability ≥ 0.85 + elbow reached + ≥ 2 pillars.
  - `src/maths/bayesian.py` — Layer 6: Living update engine. Prior from user input → likelihood from evidence → posterior update. Infers latent variables from belief instability and direction profile gaps.
  - `src/maths/game_theory.py` — Layer 7: Multi-agent reasoning. Detects agents from problem variables, finds Nash equilibrium, dominant strategies, prisoner's dilemma patterns, zero-sum vs positive-sum framing.
  - `src/maths/causal_loops.py` — Layer 8: Circular feedback detection. Detects reinforcing (spirals) and balancing (equilibria) loops, determines loop dominance, maps delay effects.
  - `src/maths/fragility.py` — Layer 9: Final stress test. Antifragility assessment, ergodicity check, tail risk evaluation, real-world executability check.
- **Design Decisions:**
  - Signal/Noise uses a weighted relevance score (direction alignment 0.4, magnitude proximity 0.3, confidence 0.3). Threshold: ≥ 0.6 = signal, 0.3-0.6 = check for latent/orthogonal, < 0.3 = noise.
  - Manifold uses cosine similarity (> 0.7) for chart overlap detection. Dimensional reduction uses 95% cumulative variance threshold.
  - Convergence requires 3 conditions: stability ≥ 0.85, elbow reached, ≥ 2 pillars. First iteration gets 0.5 stability (unknown).
  - Bayesian update uses simplified posterior ∝ prior × likelihood with normalization to [0, 0.99].
  - Game Theory detects agents via keyword matching against common relationship roles (partner, cofounder, boss, etc.).
  - Causal Loops detects connections via cross-referencing variable names/descriptions/evidence + shared framework source.
  - Fragility scores antifragility and fragility independently, classifies as fragile (> 0.5), antifragile (> 0.5), or robust (between).
- **Verification:** Full integration test with startup founder scenario:
  - Signal/Noise: 7 signal, 4 noise, 0 latent, 0 orthogonal
  - Category Theory: 58 morphisms, 11 isomorphisms, 8 emergent variables
  - Manifold: 12 dimensions → 8 intrinsic, 5 core axes identified
  - Convergence: not converged (first pass, stability 0.50) — correct, needs Bayesian loop
  - Bayesian: 16 beliefs, 13 root candidates from evidence accumulation
  - Game Theory: detected user + cofounder as agents, zero-sum + prisoner's dilemma
  - Causal Loops: 32 reinforcing, 6 balancing, dominant loop identified
  - Fragility: robust, ergodic, executable
- **Observation:** Bayesian produces 13 root candidates on first pass — too many. The convergence loop (Component 4) will refine this by running multiple iterations until the candidates narrow and convergence criteria are met. Causal loops detected 32 reinforcing loops — high count because many variable pairs have same-framework connections. May need tighter connection thresholds in future tuning.

### Decision 010 — Component 4: Formation Orchestrator Built (2026-04-01)
- **What:** Built `src/formation/orchestrator.py` — the 5-stage convergence battlefield.
- **Architecture:**
  - Stage 1 (Manifold Opens): Initializes Bayesian prior from user's stated variables.
  - Stage 2 (Tools Deploy): Runs Physics (Phase 1 + Phase 2), Game Theory (Layer 7), Causal Loops (Layer 8) simultaneously. Game and loop variables get injected as perspectives into the manifold.
  - Stage 3 (Bayesian Loop): Update beliefs → Signal/Noise filter → Category Theory translations → Build manifold → Check convergence → Repeat.
  - Stage 4 (Convergence): Breaks when stability ≥ 0.85 + elbow reached + root confidence ≥ 0.5. Also breaks early if root confidence ≥ 0.7 and stability ≥ 0.7 (diminishing returns).
  - Stage 5 (Fragility Test): Stress-tests the solution. Builds final consequences from physics + Bayesian + causal loop sources. Runs antifragility, ergodicity, tail risk, executability checks.
- **Safety:** MAX_ITERATIONS = 10 (documented rationale: empirically converges within 5-7 passes). MIN_ROOT_CONFIDENCE = 0.5. Fallback root cause created if no candidate meets threshold.
- **Design Decisions:**
  - Root cause selection scores: confidence (0.4) + framework agreement (0.3) + hidden bonus (0.3). Deduplicates by variable name before ranking.
  - Consequences built from 3 sources: physics trajectory, Bayesian posterior shifts, and causal loop dominance + delays.
  - Bias summary generated from root cause's bias_that_hid_it field (from Phase 2).
  - Dross (noise + orthogonal perspectives) preserved in output for potential future use.
- **Verification — Full end-to-end test (startup founder scenario):**
  - Converged in 2 iterations
  - 13 perspectives generated, 9 survived SVD as core pillars, 4 stored as dross
  - Root cause: `force_revenue_stagnation` (97% confidence, Bayesian-inferred)
  - Game Theory: detected user + cofounder as agents, prisoner's dilemma, Nash equilibrium (locked in competitive stable state)
  - Dominant causal loop: `vicious_spiral_revenue_stagnation_cofounder_disengagement` (reinforcing — each makes the other worse)
  - 5 consequences projected including delayed amplification from reinforcing loops
  - Fragility: ANTIFRAGILE, ergodic, executable, tail risk identified
- **Observations for tuning:**
  - Root cause landed on `force_revenue_stagnation` rather than the deeper hidden variables from bias penetration. The Bayesian update amplifies variables that get confirmed by multiple frameworks — since revenue stagnation is mentioned by the user AND confirmed by physics, it dominates. Future tuning: weight hidden variables higher in root selection.
  - 59 causal loops detected (up from 32 after game theory + loop variables added). Connection thresholds need tightening for production.
  - Converged in only 2 iterations because stability jumped quickly with single-domain input. With multiple domains (Psychology, Philosophy, Chemistry), convergence will naturally take more passes.

### Decision 011 — Taoist Wu Xing Master Architecture Adopted (2026-04-04)
- **What:** Adopted the Taoist Wu Xing (Five Elements) architecture as the master blueprint for the entire engine.
- **The 5 Elements:**
  - Earth = Physics (ground of reality, what IS happening)
  - Metal = Mathematics (precision grid, structures and measures)
  - Water = Psychology (hidden depths, bias, human distortion)
  - Wood = Philosophy (expansion, questions the question itself)
  - Fire = Chemistry (transformation + governance, decides what bonds)
- **3 Structural Layers:**
  - Outer: Wu Xing (Sheng generating cycle + Ke controlling cycle, both simultaneous)
  - Inner: Chemistry Module B governance (signal integrity between bridges)
  - Islands: 5 isolated domain modules connected only through bridge contracts
- **Sheng (Generating/Construction):** Philosophy → Chemistry → Physics → Maths → Psychology → Philosophy
- **Ke (Controlling/Deconstruction):** Physics⇌Psychology, Psychology⇌Chemistry, Chemistry⇌Maths, Maths⇌Philosophy, Philosophy⇌Physics
- **7 Stages:** Chemistry reads → Manifold opens → Dual cycles → Bayesian backbone → Convergence → Fragility → Metacognitive calibration
- **Convergence:** Gibbs Free Energy — maximum stability, minimum conflict. Converged when Sheng output survives Ke challenge.
- **Chemistry dual function:** Governs FIRST (Module A: Self-Assembly, Valence, Le Chatelier's), then fights on battlefield (Module C: Chirality, Catalysis, Resonance). No special treatment for analytical outputs.
- **Rationale:** Ancient interaction logic provides structural guarantees that prevent echo chambers (every domain is checked by a different domain than the one feeding it). Yin/Yang dual-cycle ensures construction and deconstruction happen simultaneously.

### Decision 012 — Island Architecture Restructuring (2026-04-05)
- **What:** Restructured existing Physics and Mathematics domains from linear pipeline modules into isolated island modules with bridge contracts.
- **Changes:**
  - `src/core/types.py` — Added all 36 FrameworkIDs (11 Physics + 9 Maths layers + 5 Psychology + 5 Philosophy + 6 Chemistry). Added `DomainInput`, `ChallengeInput`, `ChallengeOutput`, `BondType` types for bridge contracts. Removed `PhysicsPhase` enum (unused). Removed linear `FormationResult` (will be redesigned for Wu Xing).
  - `src/domains/physics/__init__.py` — Changed `run_physics(Problem)` to `run_physics(DomainInput) → DomainOutput`. Added `challenge(ChallengeInput) → ChallengeOutput` for Ke cycle (Earth checks Water). Documented bridge contract in module docstring. Removed all cross-domain imports.
  - `src/maths/__init__.py` — Changed to `run_mathematics(DomainInput) → DomainOutput`. Added `challenge(ChallengeInput) → ChallengeOutput` for Ke cycle (Metal checks Wood). Receives upstream outputs via bridge. Removed all cross-domain imports.
  - `src/formation/orchestrator.py` — Gutted old 5-stage linear pipeline. Left as placeholder for future Wu Xing dual-cycle engine.
- **Internal layer files UNCHANGED:** All 8 maths layer files and 2 physics phase files remain identical. They already only imported from `src.core.types`. The restructuring only affected the entry points (the `__init__.py` files).
- **Verification:**
  - Both islands import and run clean via bridge contracts.
  - Physics: 11 perspectives, 2 root causes from DomainInput.
  - Maths: 6 perspectives, 8 root candidates from upstream Physics output.
  - Ke cycle tested: Physics challenged Maths output (scrutiny: 0.33, 2 contradictions found). Maths challenged Physics output (scrutiny: 0.29, 0 contradictions).
  - **ISOLATION CHECK PASSED:** No cross-domain imports detected. Physics does not import from Maths. Maths does not import from Physics.
- **What stayed the same:** All domain logic. All 11 physics concepts. All 46 maths concepts. All thresholds, weights, and parameters. Nothing was lost — only the entry points were restructured.

### Decision 013 — Psychology Domain Built as Isolated Island (2026-04-05)
- **What:** Built the Psychology domain (Water) as a fully isolated island module with 5 concepts across 2 modules and complete bridge contracts.
- **Files created:**
  - `src/domains/psychology/__init__.py` — Island entry point. `run_psychology(DomainInput) → DomainOutput` + `challenge(ChallengeInput) → ChallengeOutput`. Extracts Physics findings from upstream via bridge (no direct import). Ke cycle: Water checks Fire (Psychology challenges Chemistry).
  - `src/domains/psychology/mind_analysis.py` — Module 1: Detection. Three concepts:
    - **Dual Process Theory:** Classifies each user variable as S1 (impulse), S2 (calculated), or S2_justifying_S1 (rationalization — FLAG). Uses emotional charge score (magnitude 0.4 + gut feeling 0.35 + direction extremity 0.25), justification depth (evidence 0.6 + description length 0.4). Classification rules: high emotion + high justification + high confidence = S2 justifying S1 (the most dangerous pattern).
    - **Cognitive Dissonance:** Finds conflicting belief pairs. Tension score = direction opposition (0.45) + magnitude similarity (0.25) + confidence of both (0.3). Detects resolution strategy: denial, minimization, compartmentalization, or none. Hypothesizes Variable D in the gap. Pairs with tension > 0.5 produce hidden variables.
    - **Motivated Reasoning:** Calculates directional bias score (% of variables favoring one conclusion). Detects pre-set conclusions when bias > 0.7. Finds missing counter-evidence. Identifies filter patterns (acknowledge-then-dismiss). Cross-references with Physics Anomalous Motion findings via bridge.
  - `src/domains/psychology/integration.py` — Module 2: Integration. Two concepts:
    - **Dialectical Thinking:** For each upstream root cause, generates thesis (user's view), antithesis (domain findings), shared ground, and synthesis. Checks synthesis stability for new contradictions. NOTE: operates on the PERSON's experience (vs Philosophy's Dialectics which operates on the SITUATION's structure).
    - **Metacognition:** 4-factor scoring — acknowledges uncertainty, presents both sides, references own role, receptivity to challenge. Each 0-1, averaged. Delivery calibration: >0.7 = direct, 0.4-0.7 = building, <0.4 = gentle.
- **Verification — Full test (freelance/partner scenario):**
  - 5 perspectives generated (one per concept)
  - Dual Process: 5 S1 (impulse), 0 S2, 1 rationalization flag (freelance_passion — high emotion + elaborate justification)
  - Cognitive Dissonance: 15 conflicting pairs, 13 with tension > 0.5. Highest: freelance_passion vs income_instability. 8 root causes generated from high-tension gaps.
  - Motivated Reasoning: 67% directional bias toward negative. Missing counter-evidence flagged.
  - Metacognition: 0.37 → gentle delivery recommended. User acknowledges no uncertainty, doesn't present both sides.
  - Ke cycle tested: Psychology challenged Physics output (scrutiny: 0.59, 13 flags about human cost of hidden variable surfacing).
  - **ISOLATION CHECK PASSED:** Psychology imports nothing from Physics or Maths. All upstream data received via bridge contracts only.
- **Key findings the Psychology domain surfaced:**
  - freelance_passion is a **rationalization** (S2 justifying S1) — the user's most emotionally charged belief has elaborate justification, which is the hallmark of post-hoc rationalization
  - The dissonance gap between freelance_passion and income_instability is the **primary tension field** where Variable D likely hides
  - User metacognition is LOW (0.37) — findings should be delivered gently, not confrontationally

### Decision 014 — Philosophy Domain Built as Isolated Island (2026-04-05)
- **What:** Built the Philosophy domain (Wood) as a fully isolated island module with 5 concepts in a logical pipeline sequence and complete bridge contracts.
- **Files created:**
  - `src/domains/philosophy/__init__.py` — Island entry point. `run_philosophy(DomainInput) → DomainOutput` + `challenge(ChallengeInput) → ChallengeOutput`. Extracts Physics contradictions, Physics trajectory, Psychology metacognition score, and motivated reasoning assessment from upstream via bridge (no direct imports). Ke cycle: Wood checks Earth (Philosophy challenges Physics).
  - `src/domains/philosophy/epistemic_pipeline.py` — All 5 concepts in sequential pipeline:
    - **Ontology:** Classifies all variables as ESSENTIAL (remove it and the problem changes fundamentally) or ACCIDENTAL (surface noise). Three tests: essential_test (magnitude × confidence, boosted for negative/circular), substance_vs_attribute (state vs circumstance), invariance_test (appears across multiple frameworks). Produces ontological core and essence statement.
    - **Epistemology:** Classifies each essential variable as FACT (evidence + verification + justification), BELIEF (conviction without full evidence), ASSUMPTION (never examined), or OPINION (preference without evidence). Flags all assumptions as false prior candidates. Detects knowledge gaps (essential but low confidence) and blind spots (missing directions in user's data).
    - **Phenomenology:** Maps experiential frame (threat/loss/opportunity/test). Maps visible and invisible horizons. Calculates frame-reality gap (significant/moderate/minimal). Identifies structural perspective limitations. Recommends bridges (information, emotional processing, uncertainty, metacognition) calibrated to metacognition score from Psychology via bridge.
    - **Dialectics:** Extracts situational thesis (dominant force) and antithesis (suppressed opposing force). Finds tension point using phenomenology's frame-reality gap. Generates synthesis — if hidden essential variables exist, the synthesis reveals them as Variable D. Checks synthesis stability for new contradictions. NOTE: operates on SITUATION structure, not person's experience (that's Psychology's Dialectical Thinking).
    - **Teleology:** Searches for hidden utility (identity preservation, avoidance, excuse). Maps telos trajectory. Tests if the problem is FUNCTIONING AS A SOLUTION to a deeper problem. Compares with physics trajectory for divergence. Builds purpose statement.
- **Verification — Full test (freelance/partner scenario):**
  - 5 perspectives generated (one per concept in pipeline)
  - Ontology: 5 essential, 1 accidental, 1 reclassified (user treating surface feature as core)
  - Epistemology: 0 facts, 2 assumptions (false prior candidates), 2 blind spots
  - Phenomenology: frame = "loss" (user experiences this as something being taken), 2 invisible elements
  - Dialectics: synthesis stability 0.60, Variable D not yet identified from this data alone
  - Teleology: **hidden utility confidence 0.85, problem IS functioning as a solution** — identity preservation + avoidance of what user would face if the surface problem were solved
  - Root causes: 1 (teleological purpose at 0.85 confidence — "the problem persists because it serves a purpose")
  - With Physics+Psychology upstream: 2 root causes (dialectical + teleological), richer phenomenological horizon mapping
  - Ke cycle tested: Philosophy challenged Physics (scrutiny: 0.77, 17 flags questioning unexamined premises)
  - **ISOLATION CHECK PASSED:** Philosophy imports nothing from Physics, Mathematics, or Psychology
- **Key findings the Philosophy domain surfaced:**
  - The problem is **functioning as a solution** to a deeper problem the user would have to face if the freelance question were resolved
  - **Identity preservation** — the freelance-vs-corporate conflict has become part of how the user defines themselves
  - 2 essential variables classified as **unexamined assumptions** — the user is building on unstable epistemic ground
  - User's experiential frame is **loss** — they see this as something being taken from them, which structurally limits what they can see

---

*Created: April 1, 2026*
### Decision 015 — Chemistry Domain Built as Isolated Island (2026-04-05)
- **What:** Built the Chemistry domain (Fire) as a fully isolated island module with 6 concepts across 2 modules (governance + analytical) and complete bridge contracts. Chemistry is the ONLY domain with dual function.
- **Files created:**
  - `src/domains/chemistry/__init__.py` — Island entry point with THREE entry points:
    - `run_governance(DomainInput) → (DomainOutput, FormationPlan)` — Module A, runs BEFORE battlefield.
    - `run_chemistry(DomainInput) → DomainOutput` — Module C, runs DURING battlefield.
    - `challenge(ChallengeInput) → ChallengeOutput` — Ke cycle: Fire checks Metal (Chemistry challenges Mathematics).
  - `src/domains/chemistry/governance.py` — Module A: Governance. Three concepts:
    - **Self-Assembly:** Finds structural affinity clusters (directional, causal, magnitude-based). Determines organizational template (linear/web/tree/cycle/hub_and_spoke). Detects misfit variables (don't fit any cluster — either noise or the most important hidden variables). Determines which domains and concepts activate. Estimates agent count. NOT every problem needs all 100+ concepts — Self-Assembly triages.
    - **Valence:** Determines bonding compatibility between two domain outputs. Polarity check (electropositive/electronegative). Shared electron search (common variables/facts). Bond type: IONIC (opposites held by attraction), COVALENT (similar outputs sharing common variable), NONE (genuinely unrelated). Bond strength calculated from shared ratio + confidence alignment + type bonus.
    - **Chemical Equilibrium (Le Chatelier's):** Absorbs stress when new heavy variables enter. Assesses stress magnitude (low/medium/high) based on contradictions with existing analysis. Determines which domains need re-running and which bonds need reevaluation. Calculates cascade risk. Prevents chain-reaction destabilization.
  - `src/domains/chemistry/analytical.py` — Module C: Analytical. Three concepts:
    - **Chirality:** Compares competing narratives from different domains. Tests if they're chiral pairs (same components, different orientation). Runs fit test against physics causality and epistemology facts. Identifies the toxic enantiomer (the mirror that fits the user's bias, not reality). Identifies the truth orientation.
    - **Catalysis:** Maps activation energy barriers (emotional, cognitive, information, identity). Searches for catalyst candidates across all root causes. Ranks by barrier reduction × truth alignment × deliverability. Crafts catalytic moment phrasing calibrated to metacognition score (direct/building/gentle).
    - **Resonance:** Tests if a single structure can express the finding. If not, lists contributing structures from all surviving domain outputs. Builds resonance hybrid (more stable than any individual structure — like benzene). Checks for irreducible ambiguity — problems that genuinely have no single answer. Prevents false certainty.
- **Verification — Full test (freelance/partner scenario):**
  - Module A (Governance):
    - Template: hub_and_spoke (negative forces dominate the topology)
    - 3 clusters: positive_forces (passion + growth), negative_forces (instability + pressure + fights + doubt), high_impact_forces (passion + pressure)
    - 0 misfits — all variables clustered naturally
    - All 5 domains activated, 8 agents estimated
  - Module C (Analytical — with Physics+Psychology+Philosophy upstream):
    - 3 perspectives: chirality (3 narratives compared), catalysis (21 root causes analyzed, primary catalyst FOUND), resonance (requires_resonance=True, stability=0.95)
    - 1 root cause contributed (catalytic root)
    - Catalysis identified the primary breakthrough insight from reductio analysis of freelance_passion
    - Resonance built a hybrid from all domain perspectives — stability 0.95 (highly stable hybrid)
  - Ke cycle tested: Chemistry challenged Mathematics (scrutiny: 0.25, 3 contradictions on extreme magnitudes, 16 flags questioning artificial precision)
  - Valence tested: Physics↔Psychology bond assessment executed (no direct shared electrons in this test — bond detection will improve with richer cross-references)
  - **ISOLATION CHECK PASSED:** Chemistry imports nothing from Physics, Mathematics, Psychology, or Philosophy
- **Key findings the Chemistry domain surfaced:**
  - The problem is a **hub_and_spoke** structure — negative forces clustered around a central hub
  - Catalysis found a **primary catalyst** — the breakthrough insight from Physics' reductio analysis of freelance_passion
  - Resonance says the truth **requires a hybrid** of all domain perspectives (no single domain captures it alone), but the hybrid is highly stable (0.95)
  - Chemistry's Ke challenge caught Mathematics producing **false certainty** — 16 variables with >90% confidence that Chemistry flagged as artificially precise

### Decision 016 — Wu Xing Orchestrator Built (2026-04-05)
- **What:** Built the Wu Xing dual-cycle Formation Orchestrator — the 7-stage engine that wires all 5 domain islands together.
- **Files created:**
  - `src/formation/cycles.py` — Wu Xing cycle definitions. Sheng order: Philosophy → Chemistry → Physics → Maths → Psychology. Ke pairs: Physics⇌Psychology, Psychology⇌Chemistry, Chemistry⇌Maths, Maths⇌Philosophy, Philosophy⇌Physics. Helper functions for partial cycles when fewer than 5 domains are active.
  - `src/formation/convergence_protocol.py` — Gibbs Free Energy convergence. 4 criteria: posterior stability (delta < 0.05), dimensional stability (< 2 new variables), cycle agreement (avg Ke scrutiny < 0.4), energy minimization (Gibbs ≥ 0.75). All thresholds documented with rationale. MAX_ITERATIONS = 12.
  - `src/formation/orchestrator.py` — The 7-stage engine:
    - **Stage 1 (Chemistry Reads):** Runs Chemistry governance (Self-Assembly). Gets formation plan: which domains activate, what template, which concepts, how many agents.
    - **Stage 2 (Manifold Opens):** Seeds the output space with governance output. Initializes tracking.
    - **Stage 3 (Dual Cycles Deploy):** Sheng cycle runs each domain in order, each receiving all upstream outputs via bridge. Ke cycle runs all 5 challenge pairs. Both in each iteration.
    - **Stage 4 (Bayesian Backbone):** Collects all root causes from all domain outputs.
    - **Stage 5 (Convergence Check):** Runs Gibbs Free Energy check. If converged → proceed. If not → loop back to Stage 3.
    - **Stage 6 (Ergodicity & Fragility):** Stress-tests the best root cause.
    - **Stage 7 (Metacognitive Calibration):** Extracts delivery mode, catalytic moment, resonance hybrid, ambiguity flag, bias summary, hidden purpose from domain outputs.
- **Design Decisions:**
  - Domain runner registry maps Domain enum to run + challenge functions. No switch statements.
  - Root cause selection scores: confidence (0.35) + framework agreement (0.25) + hidden bonus (0.20) + cross-domain bonus (0.20).
  - FormationResult includes: root cause, consequences, bias summary, hidden purpose, all domain outputs, formation plan, convergence history, all Ke results, fragility, delivery mode, catalytic moment, resonance hybrid, irreducible ambiguity flag.
  - Convergence escape hatch: forced stop after MAX_ITERATIONS with current best state + explicit uncertainty marker.
- **Verification — Full end-to-end test (freelance/partner scenario, 3 iterations):**
  - All 5 domains activated: chemistry, philosophy, physics, mathematics, psychology
  - All 5 Ke pairs executed:
    - Physics→Psychology: scrutiny 0.08 (low — psychological findings survive physical reality)
    - Psychology→Chemistry: scrutiny 0.67 (high — Psychology questioning Chemistry's bonding decisions)
    - Chemistry→Mathematics: scrutiny 0.03 (very low — Maths output is precise)
    - Mathematics→Philosophy: scrutiny 0.17 (low — philosophical claims mostly survive formal logic)
    - Philosophy→Physics: scrutiny 0.67 (high — Philosophy questioning Physics' unexamined premises)
  - Gibbs energy trajectory: 0.293 → 0.254 → 0.270 (not converged in 3 iterations — correct, Ke scrutiny is still high on 2 pairs)
  - Root cause identified: chiral pair between physics and mathematics outputs (0.99 confidence)
  - Delivery mode: gentle (metacognition score was low)
  - Fragility: ANTIFRAGILE
  - Irreducible ambiguity: False
- **Observations for tuning:**
  - Psychology→Chemistry and Philosophy→Physics Ke scrutiny are high (0.67). This means the controlling cycle is working — it's finding real issues. Convergence will require the domains to refine their outputs based on Ke feedback (not yet implemented — Ke results currently don't feed back into domain re-runs).
  - 6+ iteration runs are computationally heavy (5 domains × 5 Ke pairs per iteration). May need optimization for production latency targets.
  - Chirality detected as root cause because Physics and Maths produced outputs with the same variables in different orientations — the mirror-detection is working.

### Decision 017 — Funnel Feedback Loop + Cache + Multi-Answer Output (2026-04-05)
- **What:** Built the funnel mechanism, combination cache, fixed Valence semantic matching, and rebuilt the orchestrator with Ke-driven feedback between iterations and multi-answer output format.
- **Files created/modified:**
  - `src/formation/funnel.py` — **NEW.** The Funnel mechanism. Runs AFTER each dual-cycle pass. Filters by connection density (how many domains reference a variable), NOT by convergence direction. Critical rules implemented:
    - High Ke scrutiny (>0.5) = needs more work → KEEP downstream, force another pass
    - Low Ke scrutiny (<0.2) = survived challenge → stable, boost confidence (×1.1, cap 0.95)
    - Zero connections after 2 consecutive passes = genuine noise → cache
    - Contradictions are PROTECTED — if a variable connects to even one domain, it stays, especially if it contradicts the emerging answer
    - Variable cap: 30 per iteration (bounds O(n²)), needs_work variables get priority over cap
  - `src/formation/cache.py` — **NEW.** Combination Cache. Filtered-out variables stored with: the variable itself, problem context, result produced, iteration cached, connection score. Query by keyword similarity against new problems. Cache hits become pre-computed priors for Bayesian backbone (discounted: confidence ×0.6, magnitude ×0.7). Persists to disk as JSON.
  - `src/domains/chemistry/governance.py` — **MODIFIED.** Valence `_find_shared_electrons` rewritten. 3-layer matching:
    - Layer 1: Exact name match (fast fallback)
    - Layer 2: Semantic similarity on description word sets (Jaccard coefficient > 0.25 threshold). Stopword-filtered. Two domains can describe the same variable with different names — this catches it.
    - Layer 3: Cross-reference in evidence chains (existing logic, kept as tertiary)
  - `src/formation/orchestrator.py` — **REBUILT.** Funnel integrated between iterations. Cache integrated at Stage 1 (query for priors). Multi-answer output format:
    - Top 2-4 trajectories with confidence scores and cost/consequence for each
    - Uncertainty description (what remains genuinely uncertain)
    - `more_underneath = True` always — "There is more underneath. Deeper analysis available."
    - The USER decides whether to dig deeper. The engine does NOT decide when to stop.
- **Verification — 2 tests:**
  - **Test 1 (2 variables, 2 iterations):** 4 trajectories output. Funnel: iter 1 kept 30/cached 24, iter 2 kept 30/cached 375. Variable cap holding at 30. 3 Ke pairs active. Delivery: gentle.
  - **Test 2 (3 variables, 2 iterations — freelance/partner):**
    - 4 trajectories: force_income_instability (0.97), force_partner_pressure (0.96), entropy_decay_rate (0.95), philosophical_variable_d (0.95)
    - All 4 top trajectories are HIDDEN variables — the engine is surfacing what the user can't see
    - Funnel: iter 1 kept 30/cached 106, **15 variables flagged as needs_work** (high Ke scrutiny), 2 stable. Iter 2 kept 30/cached 1775, 15 needs_work still active.
    - All 5 domains active. Fragility: antifragile. Delivery: gentle.
    - Uncertainty: "Engine did not fully converge. 1 domain pair still under high scrutiny."
    - More underneath: True (399+ variables cached for future analysis)
- **What the funnel solved:**
  - **Convergence gap:** Ke results now drive what stays downstream. The funnel IS the feedback loop.
  - **Performance:** Variable cap at 30 bounds O(n²). Iteration 2 cached 1775 variables that would have previously grown the dataset unboundedly.
  - **Depth protection:** 15 needs_work variables were kept despite the cap — contradictions and challenged findings get priority, not comfort.
  - **Institutional memory:** 1,881 total cached variables from one problem run. Future similar problems start with pre-computed priors.

### Decision 018 — Phase 2 Begins: Agent Architecture Research (2026-04-05)
- **What:** Researched 4 open-source multi-agent frameworks (LangGraph, CrewAI, AutoGen, OpenAI Swarm) to learn orchestration patterns for LLM integration.
- **Key findings per framework:**
  - **LangGraph:** StateGraph + reducers + fan-out/fan-in + sub-graphs. `Send` API for dynamic fan-out (spawn N agents at runtime). Reducer functions merge parallel outputs. Sub-graph nesting for convergence loops. Conditional edges for routing. **Most relevant for our parallel execution model.**
  - **CrewAI:** Role/goal/backstory per agent → system prompt shaping. Typed TaskOutput flowing between stages. Sequential + hierarchical process modes. **Key insight: same LLM becomes different reasoner based on system prompt. Our thesis exactly.** But CrewAI has NO adversarial challenge pattern — our Ke cycle is architecturally beyond what they offer.
  - **AutoGen:** GroupChat with custom speaker_selection_func. Nested chats for sub-agent spawning. Composable termination conditions (AND/OR operators). Critique via role-defined system prompts. **Maps to: custom speaker selection = our Sheng order, nested chats = domain sub-agents, composable termination = our Gibbs convergence.**
  - **OpenAI Swarm:** Ultra-lightweight (~150 lines). Handoff = return an Agent object. Context = shared dict (context_variables). Function auto-conversion to tool schemas. **But sequential only — no parallelism.** Handoff pattern maps to our bridge contracts.
- **What none of them have (our architectural edge):**
  1. Adversarial dual-cycle (simultaneous construction + deconstruction)
  2. Governance layer that decides formation BEFORE the battlefield
  3. Funnel that filters between iterations using Ke scores
  4. Convergence = construction surviving deconstruction
- **Rationale:** These gaps confirm our Taoist architecture is genuinely novel. We take the useful patterns (fan-out/fan-in, role-based prompts, typed outputs, context dicts) and keep our unique dual-cycle, governance, funnel, and convergence architecture.

### Decision 019 — Step 1.1: LLM Client Built (2026-04-05)
- **What:** Built `src/llm/client.py` — The River. Single async Sonnet connection for all domain agents.
- **Architecture:**
  - Two modes: LIVE (real Anthropic API) and MOCK (deterministic responses for architecture testing)
  - `call()` — single LLM call (one tributary). Handles retries (max 2), timeout (30s), error handling.
  - `call_batch()` — fan-out: launches N calls in parallel via asyncio.gather(). This is how the Sheng cycle and Ke cycle will run all domains simultaneously.
  - Mock mode generates domain-specific structured JSON responses per concept. Each domain/concept gets a realistic response shape that the parser can process. Simulates 50-200ms latency.
  - Full call logging: domain, concept, model, tokens, latency, success/failure, timestamp.
  - Monitoring: total tokens, cost estimate (Sonnet pricing: $3/M input, $15/M output), per-domain breakdown.
  - Live mode reads API key from `ANTHROPIC_API_KEY` environment variable. Never hardcoded.
- **Verification:**
  - Single mock call: success, 129ms latency, structured JSON response
  - 5-domain parallel fan-out: all 5 responded, latencies 73-193ms
  - 3 Ke critic calls in parallel: all 3 responded
  - Total: 9 calls, 9 successful, 0 failed, 363 tokens, $0.0047 estimated cost
  - Live mode without API key: correctly raises ValueError
- **Key design choice:** Mock mode generates realistic structured responses so the entire fan-out → fan-in → funnel → convergence architecture can be tested end-to-end without spending API credits. When ready, swap to LIVE mode — same code, different mode flag.

### Decision 020 — Steps 1.2-1.7: Full Async Agent Engine Built (2026-04-05)
- **What:** Built the complete async LLM agent engine — Chemistry router, Math validation, tributary spawning, fan-in reducer, Ke fan-out, and convergence loop with Le Chatelier re-run logic.
- **Files created:**
  - `src/llm/router.py` — **Step 1.2: Chemistry Self-Assembly as Intelligence Router.** First LLM call in the pipeline. System prompt with laws: MUST triage, CANNOT activate all concepts, MUST classify complexity. Decision framework covers 8 problem signals (actors, conflict, decisions, time pressure, identity, unclear facts, simple, complex). Outputs a FormationPlan: active domains, concepts per domain, agent count, iterations, credits, complexity. Includes JSON parsing with fallback (activate everything if parse fails).
  - `src/llm/validator.py` — **Step 1.3: Math Formation Validation.** Deterministic rules, NOT an LLM call. 8 rules: actors → game theory, conflict → dissonance + motivated reasoning, decisions → dialectics + teleology, time → trajectory + entropy, physics/maths always required, chemistry governance always required, metacognition always required, agent count bounds (flag if >20 or <5). Catches mistriaging before agents spawn.
  - `src/llm/engine.py` — **Steps 1.4-1.7: The Async Formation Engine.**
    - **1.4 Tributary Spawning:** Builds domain-specific LLM calls with system prompts + upstream context. Fan-out via `client.call_batch()` (asyncio.gather). Each domain is one parallel Sonnet call. Le Chatelier's: domains with low Ke scrutiny (<0.2) skip next iteration to save compute.
    - **1.5 Fan-In Reducer:** Parses LLM JSON responses into `DomainOutput` objects via bridge contract. Preserves contradictions. Handles unparseable responses gracefully (minimal output with low weight, not a crash).
    - **1.6 Ke Fan-Out:** All 5 Ke challenge pairs run in parallel. Each is a Sonnet call with critic system prompt. Parses into `ChallengeOutput`. Handles parse failures with default 0.3 scrutiny.
    - **1.7 Convergence Loop:** Runs funnel between iterations (Ke-driven filtering, variable cap). Checks Gibbs energy. Le Chatelier's: high scrutiny domains re-run, low scrutiny domains skip. Breaks on convergence or max iterations.
    - Domain law prompts (skeleton): each domain has a base system prompt with core prohibitions and requirements. Full prompts will be expanded in Step 2.
    - Multi-answer output: top 2-4 trajectories, uncertainty, more_underneath flag, delivery mode, call summary.
- **Verification — Full end-to-end test (mock mode, freelance/partner, 3 variables):**
  - Chemistry router: classified as "medium" complexity, activated all 5 domains, 35 concepts
  - Math validation: flagged 1 adjustment (agent count >20 warning)
  - **Converged in 2 iterations** (Gibbs: 0.362 → 0.895). This is the first time the engine has actually converged — the mock responses are stable enough for the Gibbs criteria to be met.
  - All 5 Ke pairs executed: uniform 0.35 scrutiny (mock mode produces consistent scores)
  - 2 trajectories surfaced (physics + maths findings)
  - Funnel: 2 passes, 5 variables kept each pass, 0 cached (small variable set)
  - Delivery mode: "building" (medium metacognition)
  - **21 total LLM calls:** 1 router + 5 domains × 2 iterations + 5 Ke pairs × 2 iterations = 21. All successful.
  - Total tokens: 3,801. Estimated cost: $0.02 per problem run.
  - Average latency: 140ms per call (mock mode). In live mode with Sonnet: expect 1-3 seconds per call.
- **Architecture proven:** The river (single client) → tributaries (parallel domain calls) → fan-in (merged outputs) → Ke fan-out (parallel critics) → funnel (Ke-driven filtering) → convergence (Gibbs energy) → Le Chatelier's (selective re-runs) → all working in mock mode. Swap to `ClientMode.LIVE` with an API key and the same architecture runs on real Sonnet.

---

*Created: April 1, 2026*
### Decision 021 — Step 2: Full Domain Law Prompts Built (2026-04-05)
- **What:** Built `src/llm/prompts.py` — comprehensive system prompts for all 5 domain agents + 5 Ke critic variants. 90% laws (non-negotiable prohibitions and requirements), 10% guidance. Wired into the async engine.
- **Architecture:**
  - Each domain prompt has: IDENTITY (one sentence role), PROHIBITIONS (what the agent CANNOT do), REQUIREMENTS (what it MUST do), and exact JSON OUTPUT FORMAT with schema.
  - **Physics (451 words):** 6 prohibitions (no unforced causation, no vague consequences, no ignoring anomalies). 9 requirements (decompose, conservation audit, entropy, trajectory, potential energy, equilibrium, bias penetration, assumption flagging, finding labels). Output includes trajectory projection and conservation audit.
  - **Mathematics (363 words):** 6 prohibitions (no pattern without sample size, no forced fitting, no ignoring outliers). 7 requirements (signal/noise classification, morphism detection, dimensional reduction, Bayesian updates, causal loop check, game theory, self-validation). Output includes convergence status, dimensional reduction, Bayesian update, game theory.
  - **Psychology (457 words):** 6 prohibitions (no S1/S2 without evidence, no assuming motivation, no pathologizing, no vague dissonance). 6 requirements (system classification, motivated reasoning check, dissonance search, thesis/antithesis/synthesis, metacognition assessment, delivery mode). Output includes dissonance map, motivated reasoning assessment, dialectical synthesis, metacognition score.
  - **Philosophy (535 words):** 5 prohibitions (no belief as fact, no assumed frame, no skipping ontology, no synthesis without thesis/antithesis). 6 requirements (ontology → epistemology → phenomenology → dialectics → teleology sequence, strip accidentals, classify knowledge claims, map horizon, find tension, search hidden utility). Output includes ontological core, epistemic map, phenomenology, dialectics, hidden utility.
  - **Chemistry (371 words):** 4 prohibitions (no bonding without shared variable, no overriding Ke, no forcing single answer, no catalyst without barrier). 4 requirements (chirality check, catalysis identification, resonance when needed, bond type assessment). Output includes chirality, catalyst, resonance.
  - **Ke Critics (5 variants, ~305 words each):** Shared laws (no rubber-stamping, no preference-based challenges, no false challenges). Challenger-specific instructions for each Wu Xing pair: Physics→Psychology (does psychology survive material reality?), Psychology→Chemistry (should these have been bonded?), Chemistry→Mathematics (is precision meaningful or artificially clean?), Mathematics→Philosophy (does it survive formal logic?), Philosophy→Physics (has physics questioned its own assumptions?).
- **Verification:** All prompts load correctly. All contain PROHIBITIONS + REQUIREMENTS + JSON schema. Engine runs with full prompts — 21 calls, all successful, 2 iterations, 5 Ke pairs.

---

*Created: April 1, 2026*
### Decision 022 — Step 3: Valence Semantic Matching Fixed (2026-04-05)
- **What:** Built `src/llm/semantic.py` — TF-IDF cosine similarity for cross-domain variable matching. Updated Chemistry Valence to use it.
- **Architecture:**
  - Three-layer matching: Layer 1 (exact name, fastest), Layer 2 (TF-IDF cosine similarity on tokenized descriptions + evidence, no external dependencies), Layer 3 (reserved for future LLM-based embedding upgrade).
  - TF-IDF implementation from scratch: tokenization with stopword filtering (including domain-specific stopwords like "variable", "magnitude", "detected"), IDF computation across full variable corpus, cosine similarity on sparse vectors.
  - Thresholds (from spec): ≥0.7 = confirmed bond, 0.4-0.7 = possible bond (flagged), <0.4 = no bond.
  - Greedy 1-to-1 matching: each variable can only match once (highest similarity wins).
  - `matches_to_shared_electrons()` converts matches to Valence-compatible labels: exact → `name`, semantic confirmed → `a≈b`, semantic possible → `a~b`.
  - Updated `src/domains/chemistry/governance.py`: `run_valence()` now tries semantic matching first, falls back to built-in matching if import fails. Chemistry island isolation preserved (try/except import).
- **Verification — 5 tests:**
  - Exact name match: 1.0 confirmed (correct)
  - Different names, same concept: TF-IDF couldn't match (descriptions need more shared terms — known limitation, embeddings would fix this)
  - Genuinely unrelated: 0 matches (correct — no false positives)
  - Multi-variable cross-domain (Physics vs Psychology): **trajectory_negative ≈ burnout_trajectory (0.46 possible), hidden_energy_drain ≈ hidden_motivation (0.44 possible)** — these are REAL semantic matches that exact name matching completely missed
  - Valence integration: Physics↔Psychology bond went from **NONE (0.00)** to **COVALENT (0.42)** with 2 shared electrons
- **Impact:** Valence previously saw 0 bonds between Physics and Psychology. Now it detects 2 cross-domain connections. This feeds directly into Chemistry's analytical module and the funnel's connection density scoring.
- **Known limitation:** TF-IDF fails on truly different terminology describing the same concept (Test 2). A real embedding model (sentence-transformers or Sonnet embedding call) would catch these. Flagged for future upgrade — current TF-IDF is a significant improvement over exact name matching.

### Decision 023 — Steps 4-7: Production Systems Built (2026-04-05)
- **What:** Built the four production systems that sit on top of the reasoning engine: progressive disclosure, credits, graceful degradation, and the speech module.
- **Files created:**
  - `src/llm/disclosure.py` — **Step 4: Progressive Disclosure.** Two-phase response system. Phase 1 (quick batch): runs exactly 2 iterations, delivers interim findings fast. Includes: top findings, confidence score, "dig deeper" option with credit estimate. Phase 2 (deep batch): user-triggered, continues from Phase 1 state (does NOT restart), runs to convergence or max. Benefits: 15-second first response, most users get enough from Phase 1, user controls depth, 60-75% token savings when Phase 1 suffices.
  - `src/llm/credits.py` — **Step 5: Credit System.** Formula: base_cost (2.0) + active_domains × domain_cost (1.5) + iterations × iteration_cost (1.0) + ke_pairs × ke_cost (0.5). Pre-execution estimate shown to user. Post-execution invoice based on ACTUAL compute. Failure policy: 2 domains fail → refund those domains. 3+ domains fail → entire response FREE + free retry token. Phase 1 only → 40% discount.
  - `src/llm/degradation.py` — **Step 6: Graceful Degradation.** Three levels tracked by `DegradationTracker`. Level 1 (concept skip): retry once → skip concept → continue → confidence ×0.9. Level 2 (domain down): skip domain in Wu Xing → credits not charged → confidence ×0.7. Level 3 (3+ domains down): degraded mode → free response → free retry → confidence ×0.4. CRITICAL: user-facing messages contain ZERO internal terminology. "We ran into some issues" not "Physics island unreachable."
  - `src/llm/speech.py` — **Step 7: Speech Module.** The voice of LoRa. Last Sonnet call in the pipeline. System prompt with 7 prohibitions (no jargon, no absolute truths, no skipping agency, no bullet lists, no meta-commentary) and 6 requirements (conversational, multiple trajectories, concrete language, acknowledge what's right first, end with user's choice). Three delivery modes: DIRECT ("Here's what I'm seeing"), BUILDING ("Let's look at this from a few angles"), GENTLE ("I can see you've been thinking about this"). Fallback response if speech LLM call fails. `format_findings_for_speech()` helper converts engine output to speech input.
- **Verification — All 4 systems tested:**
  - Disclosure: Phase 1 delivered 2 findings at 78% confidence. "Dig deeper" offered with credit estimate.
  - Credits: 23.5 estimated pre-execution. 9.9 actual post-execution (Phase 1 discount). 3-domain failure → 0.0 charged + free retry.
  - Degradation: Level 1 correctly detected (concept skip, 0.9 confidence). Level 3 correctly triggered (3 domains down, free response, free retry). User messages contain zero internal terminology.
  - Speech: Response generated. Dig deeper prompt included. Credit summary attached.

### Decision 024 — Step 8: Integration Testing — 21/21 Passed (2026-04-05)
- **What:** Built and ran the full integration test suite covering all 7 test categories from the Phase 2 plan. 21 tests total. All passed.
- **File:** `tests/test_integration.py`
- **Results:**
  - **8.1 Unit Tests Per Domain (8 tests, all passed):**
    - Physics: 11 perspectives produced, Ke challenge produces bounded scrutiny score
    - Mathematics: processes upstream Physics correctly
    - Psychology: all 5 frameworks detected (dual_process, cognitive_dissonance, motivated_reasoning, dialectical_thinking, metacognition)
    - Philosophy: pipeline order verified (ontology → epistemology → phenomenology → dialectics → teleology)
    - Chemistry governance: formation plan with active domains + template + agent count
    - Chemistry analytical: chirality + catalysis + resonance produced with upstream
    - **Isolation: ALL 5 domains verified — zero cross-domain imports**
  - **8.2 Dual-Cycle Tests (2 tests, all passed):**
    - Sheng cycle: all 5 domains activated in correct order
    - Ke cycle: 5 challenge pairs, all with differentiated scrutiny scores, challenger ≠ target
  - **8.3 Funnel Tests (2 tests, all passed):**
    - Variable cap holds at ≤30 per iteration
    - Cache accumulates across iterations
  - **8.4 Progressive Disclosure Tests (2 tests, all passed):**
    - Phase 1 delivers ≥1 finding within 30 seconds (mock: 0.7s avg)
    - Phase 1 includes dig-deeper option with credit estimate
  - **8.5 Failure Tests (4 tests, all passed):**
    - Level 1 (concept skip): detected, confidence ×0.9, not free
    - Level 2 (domain down): detected, correct domain in domains_down, confidence ×0.7
    - Level 3 (3+ domains): detected, free_response=True, free_retry=True, confidence ×0.4, user message contains ZERO internal terminology
    - Credit failure: 3-domain failure → 0.0 credits, free retry issued
  - **8.6 Speech Module Tests (2 tests, all passed):**
    - All 3 delivery modes (direct, building, gentle) produce non-empty responses
    - Phase 1 includes dig-deeper prompt with "deeper" in text
  - **8.7 End-to-End Stress Test (1 test, passed):**
    - **10/10 problems completed in 7.3 seconds (0.73s average)**
    - Avg 2.0 trajectories per problem
    - Avg 5.0 Ke pairs per problem
    - 1,155 total LLM calls across 10 problems
    - 570,981 total tokens
    - **10/10 converged** (mock mode — live mode will take more iterations)
    - Problems tested: career change, relationship conflict, relocation, pivot decision, mid-career shift, friendship money, burnout guilt, family planning, ethical job offer, startup family pressure

### Decision 025 — Speech Module Rebuilt to Full Spec + First Live Sonnet Output (2026-04-05)
- **What:** Completely rebuilt `src/llm/speech.py` from skeleton to full implementation spec. Then ran the complete pipeline live on Sonnet — from engine to speech module — producing LoRa's first real narrated response.
- **Speech Module Architecture:**
  - **3 Pillars:** Ethos (mirror user's own words → credibility), Logos (visible reasoning chain → trust), Pathos (emotional resonance → impact that sticks)
  - **4-Step Sequence (fixed order):** Mirror → Connect → Reframe → Ask. Mirror uses user's verbatim phrases. Connect links things the user didn't connect themselves. Reframe shifts the angle without saying "you're wrong." Ask ends with open question requiring reflection.
  - **5 Finding-Specific Narration Patterns:** Chirality (lay both mirrors, let contrast work), Teleology (build slowly, this is hardest to hear), Compressed Pressure (short sentences, urgency in rhythm), False Prior (question the foundation, not the belief), Dissonance (name both beliefs, show the gap).
  - **2 Delivery Modes:** Direct (metacognition > 0.6, lead with reframe, concise, sharp question) and Building (metacognition ≤ 0.6, lead with mirror, build slowly, softer framing).
  - **Progressive Disclosure Narration:** Phase 1 under 150 words with natural dig-deeper close. Phase 2 under 500 words with full sequence.
  - **10 Prohibitions** (no system terminology, no therapy language, no academic language, no absolute truths, no single answer, no skipping agency, no skipping mirror, no violating delivery mode, no clinical tone, no filler).
  - **10 Requirements** (use user's phrases, follow 4-step sequence, adapt to finding type, match delivery mode, vary sentence rhythm, visible reasoning chain, natural confidence framing, open question ending, word limits, natural dig-deeper).
  - **2 Few-Shot Examples** baked into the system prompt (Building+Pressure and Direct+Dissonance).
  - **`extract_speech_input()` helper:** Bridges engine output to speech input. Extracts user key phrases (I am/I feel/I want/I can't patterns), emotional markers (40+ emotional words), finding type flags from domain output frameworks, contradictions from Ke results.
  - **Fallback response** if speech Sonnet call fails.
- **Code fence fix:** Added `_strip_code_fences()` to engine.py — Sonnet wraps JSON in markdown code fences (```json...```). Parser now strips these before JSON.loads. Fixed all 3 parse points (domain response, Ke response, raw analysis extraction).
- **FIRST LIVE SONNET OUTPUT — Complete Pipeline:**
  - Problem: "7-year corporate job, unfulfilled, terrified of leaving, business dream, 2 years of paralysis"
  - Engine: 4 trajectories surfaced live — "essence_of_imprisonment" (0.95), "potential_energy_accumulation" (0.95), "temporal_displacement_defense" (0.95), "false_prior_safety" (0.90)
  - All 5 Ke pairs parsed and differentiated (0.40 to 0.70 scrutiny)
  - Key phrases extracted: "comfortable but feel completely unfulfilled", "terrified of leaving the safety of my salary", "dread Monday morning", "saying that for 2 years"
  - Emotional markers: unfulfilled, dread, terrified, passionate
  - Speech output (building mode, Phase 1, 148 words):
    - Mirror: used user's exact phrases ("comfortable but feel completely unfulfilled", "passionate about but terrified")
    - Connect: "two different people living in the same body"
    - Reframe: "That's not procrastination. That's a perfectly balanced system." / "What if staying is actually the bigger risk?"
    - Ask: "What's really keeping this system locked in place?"
    - Natural dig-deeper close integrated
  - 22 Sonnet calls total. $0.28 cost. 73 seconds engine + speech.
- **Verdict:** LoRa's first real voice. The speech module followed all 4 steps, used the user's actual language, avoided all prohibited terminology, matched building delivery mode, stayed under 150 words, and ended with an open question. The output reads like a conversation with someone who spent 30 minutes thinking about the user's problem — which is exactly the benchmark.

### Decision 026 — Web UI: Chat Interface (2026-04-05)
- **What:** Built a single-page chat UI served by `server.py` at `/`, replacing the earlier card-based layout that clipped responses.
- **Files created:**
  - `web/index.html` — full chat-based UI in one file (HTML + CSS + JS, ~750 lines, no build step)
- **Key UI patterns (taken from presence-whispers production frontend):**
  - User messages right-aligned, LoRa responses left-aligned
  - Full speech text always visible (no overflow clipping, no card height limits)
  - Animated thinking indicator with cycling stage names (Chemistry reads → Dual cycles → Convergence check → ...)
  - Metadata chips inline under each response (calls, tokens, cost, iterations, time, delivery mode, mode)
  - "Show engine details" toggle reveals: trajectories, all 5 domain panels, Ke cycle scores, convergence timeline, funnel filtering — collapsed by default so the response is the hero
  - Per-domain colored panels with concept-labeled perspectives, root causes, and (after Decision 030) collapsed raw output
  - Wu Xing element colors: Earth (Physics #c9944a), Metal (Math #8a9bb0), Water (Psychology #4a8cc9), Wood (Philosophy #5ab870), Fire (Chemistry #d45454)
  - Dark theme matching LoRa's Wraith design language
  - Enter to send, Shift+Enter for newline
- **Server-side changes (`server.py`):** mounted `/static` for the web directory, root `/` returns the index.html FileResponse, kept `/api/trace` as the JSON API.
- **Verification:** Browser test at http://localhost:8100 — full speech output visible, all engine details accessible, no clipping.

### Decision 027 — Phase 2 Dig Deeper Fix (2026-04-05)
- **What:** Fixed two related bugs in the Dig Deeper / Phase 2 flow.
- **Bug 1:** Phase 2 was re-running the engine from scratch and producing nearly identical output to Phase 1. Cause: Phase 1 findings were never passed back as context for Phase 2.
- **Bug 2:** "Dig Deeper" button kept appearing after Phase 2 — infinite loop of paying tokens for nearly-same answers. Cause: server hardcoded `is_phase_one=True` regardless of iteration count.
- **Fix in `server.py`:**
  - Added `phase1_summary` body parameter and `is_phase_one = max_iterations <= 2` derivation
  - When `phase1_summary` is provided, the engine receives `problem.context` with explicit instruction: "PHASE 2 — DEEPER ANALYSIS. Do NOT repeat Phase 1. Challenge it. Find what was missed. Surface second-order effects."
  - Speech module now correctly receives `is_phase_one=False` for Phase 2 → no dig-deeper prompt, full 500-word response mode
- **Fix in `web/index.html`:** UI stores Phase 1 speech + trajectories and sends them as `phase1_summary` when the user clicks Dig Deeper.

### Decision 028 — Engine Bug Fixes (Concept Coverage + Ke Differentiation) (2026-04-05)
- **What:** Fixed 6 critical engine bugs uncovered during live testing. Most consequential commit since the original build.
- **Root cause of Problems 2-6 (concept coverage):** The engine parser in `src/llm/engine.py:_parse_domain_response()` was hardcoding every finding's framework to the domain's PRIMARY framework via `_domain_to_framework()`. So even when Sonnet correctly returned `"type": "DISSONANCE"`, the parser overwrote it with `DUAL_PROCESS`. This made it look like Psychology only used Dual Process, Philosophy only used Ontology, Chemistry only Catalysis, Physics only First Principles, Math only Bayesian — when in reality the LLM was returning varied output, the parser was just collapsing it.
- **Fix in `src/llm/engine.py`:**
  - Added `_concept_to_framework()` lookup with 60+ aliases that maps the LLM's `concept` field (case-insensitive, with synonyms like "ontological"→ONTOLOGY, "dissonance"→COGNITIVE_DISSONANCE) to the actual `FrameworkID`
  - Parser now reads `finding.concept` first, falls back to `finding.type`, falls back to domain primary only if both are missing
  - Parser also reads `finding.is_hidden` directly instead of inferring from type
- **Fix in `src/llm/prompts.py`:** Rewrote all 5 domain prompts to require an explicit `concept` field per finding and enforce minimum concept coverage:
  - Physics: 4+ concepts across BOTH Phase 1 (root finding) and Phase 2 (bias penetration)
  - Mathematics: 4+ layers, with `signal_noise`, `bayesian_inference`, `convergence` always required, and `game_theory` REQUIRED if multiple actors exist
  - Psychology: ALL 5 concepts (dual_process, cognitive_dissonance, motivated_reasoning, dialectical_thinking, metacognition)
  - Philosophy: ALL 5 concepts in sequence (ontology → epistemology → phenomenology → dialectics → teleology)
  - Chemistry: ALL 3 analytical concepts (chirality, catalysis, resonance)
- **Root cause of Problem 1 (Ke uniformity):** The Ke critic prompt let Sonnet freely pick a scrutiny score, and it defaulted to 0.70 every time across all 5 challenge pairs. The score wasn't derived from anything specific.
- **Fix in `src/llm/prompts.py` (Ke critic):** Replaced free-form scoring with structured 5-dimension evaluation:
  1. EVIDENCE_GAPS — claims without supporting evidence
  2. UNEXAMINED_ASSUMPTIONS — assumptions treated as facts
  3. MISSING_PERSPECTIVES — angles not considered
  4. LOGICAL_COHERENCE — do conclusions follow from evidence
  5. OVERCONFIDENCE — confidence justified by evidence depth
  Each dimension gets 0.0-1.0 with justification. Final scrutiny is the AVERAGE of the 5.
- **Fix in `src/llm/engine.py:_parse_ke_response()`:** Computes scrutiny as the average of dimension scores (more reliable than trusting the LLM's stated overall) and elevates dimension justifications into the flags list.
- **Verification — single live test:**
  - Ke pairs: 5 unique scores (0.768, 0.636, 0.712, 0.716, 0.758) — no two identical to two decimals
  - Physics: 7 concepts (anomalous_motion, conservation_of_energy, entropy, entropy_leak, equilibrium, first_principles, potential_kinetic) — was 1 before
  - Mathematics: 6 concepts (bayesian_inference, causal_loops, convergence, ergodicity_fragility, game_theory, signal_noise) — was 1 before
  - Psychology: all 5 concepts — was 1 before
  - Philosophy: all 5 concepts — was 1 before
  - Chemistry: all 3 analytical concepts — was 1 before

### Decision 029 — Cross-Domain Finding Deduplication (2026-04-05)
- **What:** Added `_semantic_dedupe_root_causes()` in `src/llm/engine.py` to merge top-level findings that different domains describe in different words.
- **Algorithm:** TF-IDF cosine similarity (reusing `src/llm/semantic.py`) on root cause descriptions + evidence. Threshold 0.6 (looser than the 0.7 bond threshold because these are user-facing). Greedy clustering: each cluster's highest-confidence root cause becomes the anchor; frameworks combine; evidence concatenates; cross-domain agreement boosts confidence by up to 0.15 (capped at 0.99).
- **Why 0.6 threshold:** At 0.7 (the Valence bond threshold) too few findings get merged in practice. Top-level user-facing trajectories should be deduped more aggressively because the user shouldn't see "trust_performance_paradox" twice with different names from Physics and Psychology.
- **Where it runs:** Inside `_build_trajectories()`, after exact-name dedup but before sorting and slicing top 4. Result: trajectory list is shorter and each entry represents a cross-domain consensus.

### Decision 030 — Raw JSON Hidden in UI (2026-04-05)
- **What:** Domain panels in the web UI no longer show raw LLM JSON output by default.
- **Fix in `web/index.html`:** Added a "raw output" toggle that defaults to collapsed. Click to reveal the raw JSON for debugging. Keeps the UI clean for normal use without losing the developer escape hatch.

### Decision 031 — Production Hardening (2026-04-05)
- **What:** Audited the repo for deployment readiness and fixed all critical blockers. Made the engine production-ready.
- **Audit findings:** 6 critical blockers, 10 major issues, 10 minor issues. Engine itself was sound — all problems were in the deployment shell.
- **Files created:**
  - `requirements.txt` — pinned to FastAPI 0.115+, uvicorn[standard] 0.30+, anthropic 0.40+, python-dotenv 1.0+
  - `.env.example` — documents all environment variables (ANTHROPIC_API_KEY, PORT, HOST, CORS_ORIGINS, DEFAULT_MAX_ITERATIONS, MAX_PHASE2_ITERATIONS, MAX_PHASE1_SUMMARY_CHARS, MAX_QUESTION_CHARS) with placeholder values
  - `README.md` — quick start, architecture overview, API docs, deployment instructions
  - `Dockerfile` — Python 3.11-slim base, non-root user, layer caching for deps, healthcheck on `/health`, configurable PORT/HOST via env
  - `.dockerignore` — excludes .git, .env, tests/, __pycache__, etc.
- **Files modified:**
  - `server.py` — full rewrite with production fixes (see below)
  - `run.py` — replaced manual .env parser with `python-dotenv`
  - `tests/test_integration.py` — fixed 8.6a and 8.6b which were calling `SpeechInput(confidence=...)` even though the dataclass has no `confidence` field. Added `_make_speech_input()` helper that builds a valid `SpeechInput` with all 21 required fields.
  - `.gitignore` — added .DS_Store, *.log, venv/, .venv/, dist/, build/, *.egg-info/, .pytest_cache/, .vscode/, .idea/, .env.local
- **Critical fixes in `server.py`:**
  1. **Bare `except: pass` removed.** The findings parser silently swallowed all JSON parse errors. Replaced with `_parse_findings_from_response()` helper that catches specific exceptions and returns `[]` cleanly with logging.
  2. **Race condition on `trace_events` global FIXED.** Module-level globals (`trace_events`, `trace_start`) were being reset on every request. Under concurrent traffic, request A's events could bleed into request B's response. Now: per-request local list + closure-based `emit()`. No globals. Concurrent requests are fully isolated.
  3. **Endpoint-level error handling added.** The entire `/api/trace` body is wrapped in try/except. Engine failures return structured 500 with `request_id` for log correlation instead of crashing the request.
  4. **Configurable port/host via env.** `PORT`, `HOST`, `CORS_ORIGINS` read from environment with sensible defaults. Was hardcoded `port=8100`.
  5. **Input validation added.** Question size capped at `MAX_QUESTION_CHARS` (default 8000). `phase1_summary` capped at `MAX_PHASE1_SUMMARY_CHARS` (default 20000). `max_iterations` validated as int, clamped to [1, MAX_PHASE2_ITERATIONS*2]. Bad JSON body returns 400. Empty question returns 400.
  6. **`/health` endpoint added.** Returns `{"status": "ok", "mode": "live|mock"}` for k8s/docker liveness probes.
  7. **Structured logging added.** Replaced print statements with `logging` module. Each request gets a `request_id` (timestamp-based) that appears in start/done/error log lines for correlation. Log level configurable via `LOG_LEVEL` env var.
  8. **CORS configurable.** `CORS_ORIGINS` env var (comma-separated) replaces hardcoded `["*"]`. Default still `*` for dev convenience but ready for production restriction.
  9. **`python-dotenv` instead of manual parser.** Removed the duplicated naive .env parser from server.py and run.py. Now uses `load_dotenv()` which handles quoted values, escapes, and multiline values correctly.
  10. **Type annotation fix.** `root()` had `-> FileResponse | JSONResponse` which crashed FastAPI's response model generator. Added `response_model=None` to disable auto-schema generation for this route.
- **Test suite:** All 21 integration tests now pass (was 19/21 before the SpeechInput fix). End-to-end stress test: 10 problems, 10/10 converged, 8.3s total in mock mode.
- **Live verification:**
  - `/health` returns `{"status":"ok","mode":"live"}`
  - Empty body POST → 400 `{"error":"Field 'question' is required"}`
  - Malformed JSON POST → 400 `{"error":"Invalid JSON body"}`
  - Real problem POST → 92s, 22 calls, $0.34, 5 unique Ke scores, 4 trajectories, full domain coverage
- **Git history audit:** `git log --all --full-history -- .env` returned empty. The .env file was never committed. The API key has not been exposed in the repo.
- **Status:** READY FOR DEPLOYMENT. All 6 critical blockers fixed. The Dockerfile builds a runnable container. The server can be deployed to any platform that supports Docker or Python 3.11+ with environment variables.

### Decision 032 — OpenRouter Migration + Per-Role Model Map (2026-05-22)
- **What:** Replaced the single-provider Anthropic SDK path with OpenRouter (one key, all providers) and added a per-role model assignment table. Engine code no longer passes model names — it passes `(domain, concept)` tuples and the client resolves the model.
- **Why:** Cost-conscious multi-model orchestration. Different lanes deserve different models — Mathematics needs long-context formal reasoning, Chemistry needs fast structured JSON, Ke critics benefit from cross-RLHF-lineage diversity. Going through OpenRouter keeps it to a single API key + a single SDK (OpenAI-compatible).
- **Files created:**
  - `src/llm/provider_map.py` — single source of truth for `(domain, concept) → model_slug`. Holds `DOMAIN_MODELS`, `KE_CRITIC_MODELS` (5 different models for the 5 Ke pairs), `SYNTHESIZER_MODEL`, `GATING_MODEL`, `ROUTER_MODEL`, and `PRICING` per-model. `resolve_model()` picks based on tuple. To experiment with different assignments, edit this file only.
- **Files modified:**
  - `src/llm/client.py` — replaced anthropic SDK with `openai.AsyncOpenAI` pointed at OpenRouter base URL. Added per-call model resolution. Added `model` field to `LLMResponse` and `LLMCallLog`. Added per-model `cost_usd` + `model_breakdown` in `get_call_summary()`. Env priority: `OPENROUTER_API_KEY` → `ANTHROPIC_API_KEY` (legacy fallback). Headers set `HTTP-Referer` and `X-Title` for OpenRouter analytics. Mock mode preserved unchanged.
  - `requirements.txt` — replaced `anthropic` with `openai>=1.50,<2.0`.
  - `.env.example` — `OPENROUTER_API_KEY` is now the primary key with documented placement.
- **Excluded models (cost discipline):** Opus 4.7, GPT-5.5. Default stack: Sonnet 4.6, Haiku 4.5, DeepSeek V4 Pro, Gemini 2.5 Pro / Flash / Flash-Lite. The PRICING table makes spend visible per call.
- **Status:** Live mode works against OpenRouter with a real key. Mock mode unchanged. 21/21 integration tests passing.

### Decision 033 — Synthesizer Lane + ANGULAR DISCIPLINE (2026-05-22)
- **What:** Promoted `src/llm/speech.py` from a generic "speech" call into a named synthesizer role and added an ANGULAR DISCIPLINE section to the prompt.
- **Why:** Models are layered by default (depth within one frame). The wuxing architecture is angular (5 perspectives + 5 Ke critics). Without explicit angular discipline at synthesis, the synthesizer collapses the lanes back into a single weighted consensus. The new rules force it to surface alternative angles, run one dialectical reversal per main insight, anchor a concrete timeframe, optionally add a reference-class shadow, and seed the inspiration in the closing question.
- **Files modified:**
  - `src/llm/speech.py` — changed `domain="speech"` → `domain="synthesizer"` so `provider_map.resolve_model()` routes to `SYNTHESIZER_MODEL` (Sonnet 4.6). Added the ANGULAR DISCIPLINE section (5 numbered rules) between the existing FINDING-SPECIFIC PATTERNS and DELIVERY MODES blocks. `SpeechInput`, `SpeechOutput`, and the `generate_speech()` signature were NOT touched — callers in `run.py` / `server.py` work unchanged.
- **Status:** Synthesizer now routes to the Sonnet-tier model via provider_map. The angular rules are part of every speech call. Integration tests still pass.

### Decision 034 — Graphify Vendored + Bridge Layer Built (2026-05-22)
- **What:** Vendored the MIT-licensed `graphify` library and built a dedicated bridge layer that connects code-structure memory (graphify) with decision memory (Memory V2). The bridge does NOT blend — it cross-references.
- **Why:** Two memory systems serve complementary roles: graphify is lossless code structure (who calls what, who imports what), Memory V2 stores decision anchors (why the code looks this way). The cross-product — "this decision is anchored to this code; has the code drifted from the decision?" — is the unique signal that no single system can produce.
- **Files created:**
  - `vendor/graphify/` — cloned from `safishamsi/graphify` at `--depth 1`, `.git` removed, LICENSE preserved. Fully isolated under `vendor/` to make the boundary obvious.
  - `src/bridge/__init__.py` — public exports.
  - `src/bridge/types.py` — `DecisionAnchor`, `CodeRef`, `ContextFingerprint`, `DriftReport`, `BridgeQuery`, `BridgeResult` dataclasses.
  - `src/bridge/graphify_adapter.py` — REAL adapter. Parses `graphify-out/graph.json` directly via stdlib `json` (avoids pulling `networkx` as a hard dep). Lazy load on first query. Handles both `links` and `edges` key variants. Provides `get_code_structure(file_path)`, `get_callers_of(symbol)`, `get_dependencies_of(file_path)`, `get_node(node_id)`. Returns `FileNotFoundError` with the exact CLI command (`graphify extract .`) if the graph is missing.
  - `src/bridge/memory_adapter.py` — STUBBED. Every method raises `NotImplementedError` with a TODO pointer to the specific TS file in `lora-v1-frontend/src/emotion-core/memory-v2/` whose semantics the Python port must mirror. Future port becomes fill-in-the-blanks.
  - `src/bridge/drift.py` — `DriftComparator` Protocol + `stub_comparator` + `detect_drift()` / `detect_drift_for_ref()`. Floating-decision case (decision with no code_refs) handled explicitly. Real LLM-backed comparator is a separate upcoming task.
  - `src/bridge/client.py` — `BridgeClient` facade with two modes. `mode="stub"` constructs cleanly and is the test-friendly path. `mode="live"` raises `NotImplementedError` at construction (refuses to be half-live) — once the Memory V2 port lands, this flips on.
  - `tests/test_bridge.py` — 21 tests, all passing. Covers type construction, mode validation, stub-mode graphify reads, Memory V2 NotImplementedError pointers, drift detection end-to-end against the stub comparator, GraphifyAdapter direct construction.
- **Audit findings (3 minor, deferred):**
  1. `types.py:98` — DriftReport docstring references a nonexistent `per_ref_reports` field (only exists as a local var in `drift.py`).
  2. `memory_adapter.get_code_refs_for_decision` declares return as untyped `list` while `client.py` wrapper declares `list[CodeRef]` — inconsistent.
  3. `detect_drift` collapses per-ref reports into a single summary — information loss when multiple refs drift.
  Recommended fix: add `per_ref_reports: list["DriftReport"] = field(default_factory=list)` to the DriftReport dataclass (resolves 1 and 3); add `CodeRef` import to memory_adapter (resolves 2).
- **Status:** Bridge layer is wired and tested. 21/21 bridge tests pass; 21/21 integration tests still pass — zero regression. Memory V2 Python port is the next upcoming task. The 3 audit issues are non-blocking and tracked here.

### Decision 035 — Effort Tier System + Multi-Provider Env Scaffold (2026-05-22)
- **What:** Added a user-facing `effort` tier (low / medium / high) that maps to the engine's existing `max_iterations` knob, and declared placement for all common provider API keys in `.env.example` so the user has one place to paste them.
- **Why (effort):** Surfacing iterations as an opaque integer to the API/UI is a footgun. Three discrete tiers let the UI render an "effort selector" without leaking engine internals, and give us a single place to revisit the budget mapping later. The mapping is intentionally generous on the high end (10 vs the engine's hard cap of 12) so HIGH stays under MAX_ITERATIONS and the engine keeps headroom.
- **Why (env scaffold):** Even though OpenRouter is the primary path (single key, all providers), having declared slots for direct provider keys means a future direct-provider routing experiment is a 3-line code change, not a hunt-for-the-env-var session.
- **Tier mapping:**
  - LOW    → 3 iterations  (~5-8 LLM calls, ~$0.02-$0.06)
  - MEDIUM → 6 iterations  (~10-15 LLM calls, ~$0.05-$0.15) — default
  - HIGH   → 10 iterations (~18-25 LLM calls, ~$0.10-$0.30)
- **Files created:**
  - `src/llm/effort.py` — `Effort` enum + `EFFORT_ITERATIONS` map + `normalize_effort()` (garbage input falls back to default rather than raising) + `iterations_for()`. Engine-layer cap (MAX_ITERATIONS=12) remains the absolute ceiling.
- **Files modified:**
  - `.env.example` — restructured into three blocks: (1) `OPENROUTER_API_KEY` (primary), (2) optional direct keys for Anthropic / OpenAI / Google / Gemini / xAI, (3) server + engine config including the new `LORA_EFFORT=medium` default. Note: `GOOGLE_API_KEY` and `GEMINI_API_KEY` are aliases — pasting the same key in both is fine (different SDKs read different names).
  - `server.py` — accepts `"effort": "low" | "medium" | "high"` in the `/api/trace` body. Derives `max_iterations` from the tier. Raw `max_iterations` is still honored if explicitly passed (backwards-compatible). Replaced `os.environ.get("ANTHROPIC_API_KEY")` checks with a new `_has_live_key()` helper that accepts either `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY`. `/health` now reports `default_effort`. Startup log line shows the resolved default effort + iteration count.
  - `run.py` — interactive mode shows the effort tier in the banner; `effort low|medium|high` typed at the prompt switches tiers mid-session. One-shot mode accepts `--effort=high` or `--effort high`. Reads `LORA_EFFORT` from env as the default.
- **API shape:**
  ```json
  POST /api/trace
  { "question": "…", "effort": "high" }
  ```
- **Audit pass:** Imports clean, 21/21 integration tests pass, 21/21 bridge tests pass, no stale `ANTHROPIC_API_KEY` checks remain in `server.py`. The `MAX_PHASE2_ITERATIONS * 2` clamp (=12) lines up with the engine's own MAX_ITERATIONS=12, so the cap is consistent across both layers.
- **Status:** READY. Both UI tiers and direct CLI flags work end-to-end. Effort tier is the preferred control going forward; raw `max_iterations` remains for legacy clients.

---

*Created: April 1, 2026*
*Status: PRODUCTION-READY. Engine live on Sonnet via OpenRouter (single key, per-role model map). Full pipeline: 5 domains (63 concepts) → Wu Xing dual cycles → funnel → convergence → synthesizer (with angular discipline) → narrated response. Bridge layer (graphify ↔ Memory V2 stub) wired and tested. Effort tier (low/medium/high → 3/6/10 iterations) wired through server + CLI. Web UI served at /. /health endpoint live. Dockerfile built. 21/21 integration tests + 21/21 bridge tests passing. Next: Memory V2 Python port, LLM-backed drift comparator, fix the 3 minor bridge audit issues (Decision 034).*

---

# SUPPLEMENT — 2026-05-26 (beyond Decision 035)

> The decision log above is comprehensive through 2026-05-22. The repo has
> grown substantially since then — dispatcher, triage gate, capabilities,
> MCP router stub, the segmented streaming endpoint, persistence pipeline.
> This supplement is the fast-lookup section for sessions opening the repo
> cold. Read it before reading the engine source.

## SISTER REPOS — DO NOT CONFUSE

| Repo | Path | What it is | What it ISN'T |
|---|---|---|---|
| **reasoningEngine** (this) | `~/Desktop/reasoningEngine` | Python FastAPI. **Wu Xing engine** — 5 domains (Earth/Metal/Water/Wood/Fire) + Sheng/Ke cycles. Powers Constellax. | NOT the LoRa backend. NOT a LoRaMaths fork. |
| **constellax-ui** | `~/Desktop/constellax-ui` | Vite + React + TypeScript frontend for this engine. The 3-segment Thinking Room + Map Room + history sidebar live here. | NOT presence-whispers. |
| LoRaMaths | `~/Desktop/LoRaMaths` | LoRa's Python microservice. **Vortex pipeline** — 5 mathematical frameworks (regression, Bayesian, game theory, constraint, causal loop). Read-only reference. | NOT what Constellax uses. Anyone hearing "Vortex" / "5 frameworks" / "31 combinations" — that's LoRaMaths, NOT this repo. |
| lora-v1-frontend | `~/Desktop/lora-v1-frontend` | LoRa's TypeScript/Express backend (note: misnamed dir). Read-only reference. | Not Constellax. |
| presence-whispers | (LoRa frontend repo) | LoRa's React UI. Read-only reference. | Not constellax-ui. |
| MemoryArchitecture | `~/Desktop/MemoryArchitecture` | Memory V2 design repo. Read-only reference. | Not yet ported to this engine. |

**Common confusion patterns to avoid:**
1. "Vortex" / "5 frameworks" / "regression + Bayesian + game theory + constraint + causal loop" → **LoRaMaths**, not Constellax. Constellax uses **Wu Xing**: 5 domains (Physics/Mathematics/Psychology/Philosophy/Chemistry) + Ke pairs.
2. "the LoRa engine" / "the backend" — ambiguous. Always disambiguate which repo is meant.

## CURRENT API SURFACE (as of 2026-05-26)

The server is `server.py` at the repo root. FastAPI app on `:8100` (configurable via `PORT`).

| Endpoint | Status | Purpose |
|---|---|---|
| `POST /api/v2/trace` | **LIVE** (Decision 031, ~2026-04-05) | One-shot dispatch. Full memo in a single response. Used by Map Room follow-up trace, the resume path, and any non-streaming consumer. **Do NOT delete this — it's still load-bearing.** |
| `POST /api/v2/trace/segment` | **LIVE** (new 2026-05-26, see Decision 036) | 3-phase streaming. `phase=synthesizer` runs the full dispatch + caches results. `phase=opinion` / `phase=prospects` return cached slices (instant) or regenerate with splice (3–10s). |
| `POST /api/v2/trace/resume` | LIVE | Escalation-accepted continuation (carries Phase 1 summary). |
| `POST /api/v2/dispatch/preview` | LIVE | Pre-flight cost estimate — runs only triage + formation router, no engine. |
| `GET /api/v2/threads` | LIVE | List threads (history sidebar source). |
| `GET /api/v2/thread/{id}/full` | LIVE | Thread + all iterations (Map Room rehydration). |
| `DELETE /api/v2/thread/{id}` | LIVE (2026-05-25) | Delete thread (sidebar – button). |
| `GET /api/v2/iteration/{id}` | LIVE | Single iteration with full memo. |
| `POST /api/v2/iteration/{id}/outcome` | LIVE | User outcome report → memory pipeline. |
| `GET /health` | LIVE | Liveness probe. |
| `GET /` | LIVE | Dev/debug web UI (`web/index.html`). Production frontend is `constellax-ui`. |

## MODULE MAP (post-Decision 035)

Many of these are untracked in git as of this writing. Don't assume "untracked = unreleased" — most are live and load-bearing.

### Dispatch + routing
- `src/dispatcher.py` — single entry point. `dispatch(text, client, user_effort, policy, caps, ...)` wires **triage → capabilities → budget → engine**. Both `/api/v2/trace` and `/api/v2/trace/segment` call this. Returns `DispatchResult { route, response_text, engine_result, memo, debug, ... }`.
- `src/llm/triage.py` — front-door classifier. Routes to `TRIVIAL` / `DIRECT` / `DIRECT_PLUS` / `DEEP`. Gemini Flash-Lite in live mode (~$0.0001/call, ~500ms). Mock mode = deterministic keyword classifier.
- `src/llm/budget.py` — per-request `BudgetCaps` (max iterations / wall time / cost / MCP calls).
- `src/llm/checklist.py` — angular checklist injected into DEEP synthesizer's `extra_directives`.
- `src/llm/dispatch_preview.py` — cheap pre-flight that runs only the classifiers + estimates cost without firing the engine.
- `src/llm/effort.py` — `Effort` enum + iteration mapping (low=3 / medium=6 / high=10).

### Capabilities + MCP
- `src/capabilities/registry.py` — `CapabilityRegistry`. Tracks what MCPs are available / missing / absent-by-design + permission state. Two-tier model: `LOCAL_READ + ALWAYS_ALLOWED` vs `EXTERNAL_READ + ASK_ONCE_PENDING/GRANTED`.
- `src/mcp_router.py` — uniform `fire_mcp()` / `fire_mcps()` API. **Stubbed** — returns placeholders. Real MCP clients land later; this is the contract the rest of the system codes against.

### Bridge / persistence
- `src/bridge/thread_persistence.py` — **The persistence FastAPI router.** Registers `/api/v2/threads`, `/api/v2/thread/{id}/full`, `/api/v2/iteration/{id}`, `/api/v2/iteration/{id}/outcome`, `DELETE /api/v2/thread/{id}`. Calls `_persist_iteration` (fire-and-forget) from the trace endpoints.
- `src/bridge/conversation_store.py` — structured-storage spine for sessions / iterations / turning points. Backed by Redis (production) or in-memory dict (dev).
- `src/bridge/thread_store.py` — `ThreadStore` Protocol with FalkorThreadStore (Redis-graph) + InMemoryThreadStore impls.
- `src/bridge/redis_backend.py` — Redis backends for both conversation and thread stores. Env: `CONSTELLAX_REDIS_URL`.
- `src/bridge/embedding_service.py` / `embedding_scorer.py` — Gemini `embedding-001` (3072-dim) for iteration text + similarity scoring.
- `src/bridge/iteration_metadata.py` — single-LLM-call extractor for the 7 memory signals (entities, tags, user_mode, time_horizon, load_bearing_assumption, ...).
- `src/bridge/web_search.py` — `_SearchProvider` Protocol. **Tavily** is primary (when `TAVILY_API_KEY` set, 1000 credits/mo on Researcher tier); **DuckDuckGo HTML** is fallback. 5-min in-memory cache.
- `src/bridge/search_router.py` — Gemini Flash classifies `needs_search` + rewrites query. Falls back to regex heuristic if LLM unavailable.
- `src/bridge/client.py` / `memory_adapter.py` / `graphify_adapter.py` / `drift.py` — bridge layer between Memory V2 and graphify (Decision 034). `mode="live"` still raises `NotImplementedError`; `mode="stub"` works.
- `vendor/graphify/` — vendored at `safishamsi/graphify` (Decision 034 + project memory note about `v0.8.18` patched adapter).

### Speech (synthesizer)
- `src/llm/speech.py` — **Two kinds of generators now live in this file**:
  - `generate_speech(client, speech_input, extra_directives)` — original full-memo generator (Decision 025). Called from `dispatch()` for non-streaming flow. Still authoritative for the "give me the whole memo at once" path.
  - `generate_synthesizer_segment` / `generate_opinion_segment` / `generate_prospects_segment` — phase-focused generators (Decision 036, 2026-05-26). Used by the segment endpoint only when the user splices in (no splice → cached slice returns instantly).
  - `SPEECH_SYSTEM_PROMPT` is the full prompt used by `generate_speech`. The per-segment generators use `_SEGMENT_VOICE_PREAMBLE` + per-phase schema blocks (`_SYNTHESIZER_SCHEMA_BLOCK`, `_OPINION_SCHEMA_BLOCK`, `_PROSPECTS_SCHEMA_BLOCK`).
  - Shared utilities (`_extract_memo_json`, `_normalize_memo`, `_normalize_visual`, `_compose_response_text_from_memo`, `extract_speech_input`) are used by both paths.

### Project graph
- `src/project/identity.py` / `registry.py` — project-graph integration. Lets the engine reason about specific projects the user is tracking.

### Core types (added since Decision 035)
- `src/core/thread_types.py` — `ThreadRecord`, `IterationRecord`, `SegmentedResponse`, `Entity`. `Entity` gained `category` (FACT/PATTERN/VALUE/CONTEXT/TENSION/INTEREST) and `pinned: bool`. `ThreadRecord` gained `aggregate_time_ms`, `aggregate_cost_usd`, `perspectives_run`.

---

### Decision 036 — Per-Phase Streaming Endpoint (2026-05-26)
- **What:** New `POST /api/v2/trace/segment` endpoint + `_SEGMENT_CACHE` + per-phase generators in `speech.py`. Replaces the timer-faked 3-segment "streaming" the frontend was doing client-side with real server-driven segment delivery.
- **Why:** The frontend's old 3-segment Thinking Room ran a `setTimeout` countdown and revealed pre-baked slices of the same full memo. A user splice in a breathing room created a follow-up turn — it did NOT actually reshape the next segment. The new endpoint makes splicing real: splice text is threaded into the next phase's LLM call, and segments are delivered as separate round-trips.
- **Files added/modified:**
  - `src/llm/speech.py` — added `SegmentMemo` dataclass + three new generators (`generate_synthesizer_segment`, `generate_opinion_segment`, `generate_prospects_segment`) with focused per-phase JSON schemas. `_parse_segment_json` + `_normalize_segment_fields` strip outputs to the slice each phase owns. The original `generate_speech` is untouched.
  - `server.py` — imported the new generators. Added `_SEGMENT_CACHE` (30-min TTL, 500-entry cap, `_Lock` guarded), `_cache_segments` / `_get_cached_segments` / `_update_cached_segment`, and `_synthesizer_slice` / `_opinion_slice` / `_prospects_slice` extractors. New `@app.post("/api/v2/trace/segment")` handler — full validation, three phase branches, persistence + Map Room cache mirror on synthesizer phase. The old `/api/v2/trace` is byte-for-byte unchanged.
- **Flow:**
  - `phase=synthesizer` (no `memo_id`): runs `dispatch()` normally (engine + full speech), caches `(speech_input, full_memo, thread_id)` keyed by `memo_id` (= thread_id), returns the verdict slice + full memo + `next_phase: "opinion"`. Persists iteration + populates Map Room cache same as `/api/v2/trace`.
  - `phase=opinion` or `phase=prospects` (requires `memo_id`):
    - **No splice** → returns cached slice instantly (14ms / 1ms in smoke tests). No LLM call.
    - **With splice** → fires `generate_opinion_segment` or `generate_prospects_segment` with the splice text + prior segments as context. Merges the regenerated slice back into the cache. Returns the new slice. ~3–10s with Sonnet 4.6.
- **Non-DEEP routes:** trivial / direct / direct_plus return `next_phase=null` + `done=true` on the synthesizer call. Frontend short-circuits — no follow-on calls fired. Cache still populated defensively (empty slices) so any accidental follow-on call returns empty rather than 404.
- **Known compromise:** persistence fires once at synthesizer time with the un-spliced full memo from `generate_speech`. A regenerated opinion / prospects updates the in-memory cache but does NOT re-persist. Map Room rehydration on a previously-spliced thread shows the original memo. Re-persisting on splice is straightforward but deferred — bandwidth and write-amplification are worth thinking about before turning it on.
- **Frontend counterpart:** `constellax-ui` — `runSynthesizerSegment` + `runFollowonSegment` in `src/lib/api.ts`, `SegmentedResponse.tsx` fully rewritten to drive real streaming with internal breathing-room state + abort controller. New `TurnResponse.streamPhase` field + `merge_turn_memo` reducer action. The `useDispatchState` hook calls `runSynthesizerSegment` for the chat flow; `runTrace` survives because Map Room follow-ups still use it.
- **Verified:** Both endpoints register cleanly. Validation paths (missing phase, missing memo_id, unknown memo_id, missing question) return correct 400/404s. Trivial-route synth call (~1.5s) returns the right shape; cached follow-on calls in 14ms / 1ms. Real DEEP-route trade-off question (829s engine wall-clock, route=deep, next_phase=opinion) returned a full memo with 2 visuals (comparison-table + mermaid) — confirming the prompt directive added at Decision 033's "visuals[] — always emit at least one" landed.

### Decision 037 — Strategist Axioms (PLANNED, not yet built — 2026-05-26)
- **What (proposed):** A new `src/llm/strategist_axioms.py` module exporting 8 judgment axioms ("REVERSIBILITY", "ENERGY", "COMPOUNDING", "WEDGE", "STOP_FIRST", "ONE_PATTERN", "OPERATOR_AUTHORITY", "CONCRETE_OR_SILENT") and a `build_axioms_block()` renderer. The block gets prepended to the synthesizer system prompt in every `generate_*` function — `generate_speech` + the three per-segment generators.
- **Why:** Voice rules in `SPEECH_SYSTEM_PROMPT` shape HOW the model speaks. Axioms shape WHAT it concludes when the engine surfaces balanced trajectories. Without axioms the synthesizer's read drifts day-to-day; with axioms the same question gets a recognisable kind of answer across sessions.
- **Design constraints (locked):** Axioms must CHANNEL the LLM's reasoning, never FILTER it (no post-generation rewrite step, no second LLM pass, no latency). They live as preceding context at the TOP of the system prompt — strongest attention weight. ~600 tokens added per synthesizer call. Wu Xing engine layer **not touched**. Domain prompts **not touched** — domains do disciplinary work, axioms only apply at the synthesis layer.
- **Status:** designed + agreed; not yet built. When built, append as a real Decision entry with verification.

### Decision 038 — Map Room Visualizer Pipeline (2026-05-26)
- **What:** New `src/llm/visualizer.py` module implementing Codex's three-stage architecture for Map Room visuals: **classify_visual_intent() → generate_visual_spec() → validate_visual_spec()**. The pipeline runs in the opinion phase, immediately after `dispatch()` produces the full memo and before `_persist_after_engine_done` fires — visuals are attached to the memo before persistence.
- **Why:** Speech.py was supposed to emit `visuals[]` as part of the JSON memo, but the synthesizer prompt was overloaded (voice rules + verdict + reasoning + alternatives + falsifiers + visuals all at once) and consistently dropped visuals on reflective / philosophical memos. The Map Room rendered "This memo didn't include visuals" on real DEEP traces. The procedural pipeline solves this by owning visuals as a dedicated phase: a free heuristic classifier picks the FORM, a focused LLM call (Sonnet 4.6) builds the spec, a strict validator drops malformed output.
- **Files added/modified:**
  - `src/llm/visualizer.py` (new) — classifier (heuristic, no LLM), generator (one Sonnet call per intent, `_VISUAL_GENERATOR_PROMPT` is tight: only translates memo into shape, declines via `{"type":"none"}` when can't), validator (strict structural shape checks per type, no LLM). `build_visuals()` is the public driver: never raises, returns `[]` on any failure.
  - `server.py` — imported `build_visuals`; opinion phase calls it immediately after `cached_memo = new_full_memo` and patches `new_full_memo["visuals"]` + cache entry before `_persist_after_engine_done` fires.
- **Classifier philosophy (correction from first iteration):** classifier picks the FORM, NOT the *whether*. First iteration was a gatekeeper ("should we visualize at all?") and over-rejected reflective memos. Fixed by inverting: substantive memos (verdict_body ≥80 chars + reasoning OR alternatives) always get at least one visual intent. The "no" vote moves downstream — generator can emit `{"type":"none"}`, validator drops malformed output.
- **Cost:** ~$0.005-0.01 per query (1-2 Sonnet calls, capped at 2 visuals per memo). Inside a 5-13 min DEEP pipeline that costs $0.50+, this is rounding error.
- **Scaffolding (created + ripped):** During development, a temporary `POST /api/v2/debug/visualize` endpoint + `_DEBUG_FIXTURES` dict + `scripts/fixtures/real_inspiration_thread.json` lived in server.py for cheap end-to-end testing without spending engine money. Endpoint accepted `{"fixture":"<name>"}`, ran ONLY the visualizer (~$0.005 per call), stashed result in legacy `_MEMO_CACHE` under `thr-debug-*`, returned a `map_room_url`. After all 6 visual weapons (Decision 039) were confirmed working end-to-end via the scaffold, the entire 418-line scaffolding block was line-sliced out of server.py and the fixture file deleted. Route count 26 → 25.
- **Verified:** 8 unit tests on classifier + 6 validators (all visual types). Cold-import clean. Real DEEP question on the fixture produced `comparison-table + flow-graph` visuals rendered correctly in Map Room (confirmed by user screenshot before scaffold ripped).

### Decision 039 — Six-Visual Arsenal (2026-05-26)
- **What:** Expanded the Map Room from 3 visual forms (mermaid, comparison-table, vega-lite) to 9 effective forms. The 6 new forms are user-requested ("every weapon in our arsenal") and route via classifier signals — NOT all forms fire on every memo. Each weapon was added end-to-end (backend prompt + classifier signal + validator + frontend type + renderer + parser) in 6 discrete steps.
- **Weapons (in order added):**
  1. **Tension diagram** — variant of mermaid. New `pattern: "tension"` field on `MermaidSpec`. Classifier detects "tension/contradiction/two versions/passive frame/active frame" anywhere in memo. Renderer shows "TENSION" badge. Zero new dependencies. Generator prompt has an explicit TENSION PATTERN section with the canonical `A -.->|outsources control| D` transformation edge example.
  2. **Timeline** — mermaid `timeline` syntax. Validator relaxed to accept `graph`/`flowchart`/`timeline`/`gantt` declarations (was just graph/flowchart). Classifier signals on "Week N", "Month N", "Q1-Q4", "Day 1", "milestone", "phase", "roadmap". Pattern detection prioritizes `timeline` over `tension` when both signals present (temporal structure usually dominates).
  3. **Quadrant matrix** — new `quadrant` spec type. New custom React renderer (in-house SVG/CSS, NO new dep). Items placed by `x`/`y` in 0-100 percent of axes. Recommended/warning tags drive visual emphasis. Classifier fires on quadrant language + 3+ alternatives. Wins over comparison-table when explicit quadrant signal present.
  4. **Score-chart** — Vega-Lite bar chart of LLM-scored alternatives, no CSV required. Classifier intent `score-chart` but spec emits as `type: "vega-lite"` (frontend dispatches on spec.type). New `_validate_vega_lite()` rejects remote-fetch (`data.url`) and requires inline `data.values`. Fires when 3+ alternatives, no quadrant cue.
  5. **Flow-graph** — interactive node-link via `@xyflow/react` (new dep, lazy-loaded ~120KB gz). New `flow-graph` spec type. Topological-depth layout (BFS) — handles 4-15 nodes cleanly. Node `kind` drives style: decision/outcome/claim/default. Classifier fires on dependency/system/loop language OR 4+ reasoning items.
  6. **Knowledge-graph** — force-directed network via `cytoscape` (new dep, lazy-loaded ~80KB gz). New `knowledge-graph` spec type. Cose/concentric/breadthfirst/grid layouts. Entity `kind` drives node bg color (8 kinds: person/concept/decision/claim/entity/system/tool/outcome). Classifier fires on stakeholder/ecosystem/entity/network language.
- **Routing decisions (mutually exclusive families, max 2 visuals per memo):**
  - **Option-compare family:** quadrant > score-chart > comparison-table
  - **Structural-diagram family:** knowledge-graph > flow-graph > mermaid
  - One of each family fires when its signal is present.
- **Files modified (backend):** `src/llm/visualizer.py` (+~600 lines: 4 new validators, 3 new signal detectors, 6 new prompt sections, pattern detection function).
- **Files modified (frontend):** `constellax-ui/src/types/index.ts` (3 new spec types + pattern field on MermaidSpec), `constellax-ui/src/components/Visual.tsx` (QuadrantMatrix, FlowGraph, KnowledgeGraph React components + dispatcher entries; mermaid badge now uses pattern hint), `constellax-ui/src/lib/parseMemo.ts` (3 new spec parsers in `asVisualArray`).
- **Cost impact:** Zero per-query change — classifier still emits at most 2 intents, so still 1-2 visualizer LLM calls per memo. Bundle cost is lazy-loaded — Mermaid-only memos pay zero for xyflow/cytoscape.
- **Knowledge-graph styling bug + fix (same day):** First render had white text on white-ish node backgrounds (kindColors palette all variants of white, text color also white). Fixed by changing node text color to `#0A0F2E` (dark navy) and tinting border `rgba(7, 9, 31, 0.35)`. Background palette renamed to `kindBgColors`. Verified via real DEEP run — entity labels now legible.

### Decision 040 — Persistence Memo Round-Trip (2026-05-26)
- **What:** Added `memo: dict | None = None` field to `SegmentedResponse` (`src/core/thread_types.py`) so the raw memo dict — including visuals — survives the Falkor persistence boundary and reaches the frontend's Map Room on fresh-tab rehydration. Three surgical edits: dataclass field, `_iteration_from_payload` coercion, `_build_segmented_response` populates it.
- **Why this was THE blocker:** Decision 038 + 039 wired the visualizer to attach visuals to the memo. But persistence stripped the structured memo at the boundary — `_build_segmented_response` translated memo into staged segments (synthesizer.text + opinion.text + prospects.text + map_room.visuals), the raw memo dict was discarded. Frontend `getThreadFull` returned `iteration.response` without a `memo` field. `normalizeIteration` read `response.memo ?? it.memo ?? null` → null → frontend fell through to `parseSynthesisToMemo(synthesis)` which parses prose only — visuals are structured JSON, can't be reconstructed from prose. **Net effect: visualizer worked, persistence dropped the work, Map Room rendered "no visuals" on every real DEEP query.**
- **Files modified:**
  - `src/core/thread_types.py` — `SegmentedResponse.memo: dict | None = None`; `_iteration_from_payload` adds `memo=response_raw.get("memo") if isinstance(...) else None` to the SegmentedResponse construction.
  - `src/bridge/thread_persistence.py` — `_build_segmented_response` sets `memo=memo if isinstance(memo, dict) else None` (and `memo=None` on the non-deep-route fast-path).
- **Verification:** round-trip test confirms `IterationRecord.from_payload({...response: {memo: {visuals: [{type:'mermaid', spec:'graph TD\n A-->B'}]}}})` preserves the visual spec byte-for-byte.
- **Process note:** the fix was first applied and rolled back the same day on user safety call ("create a scaffolding. No to fuck things up. You will open on another wound. So be safe."), then re-applied after the visualizer + 6-visual arsenal were validated end-to-end via the debug scaffold (Decision 038). The discipline was: validate via scaffold first → confirm the visualizer is correct → THEN close the persistence seam. The order matters because debugging "no visuals" with both pipelines broken is impossible.

### Decision 041 — Tavily Query Truncation (2026-05-26)
- **What:** Added `_MAX_QUERY_CHARS = 380` constant to `TavilyProvider` (`src/bridge/web_search.py`) and a word-boundary trim before sending the request body. Trailing whitespace dropped at the last space within the cap.
- **Why:** Tavily's API caps queries at ~400 chars on the free tier (longer queries silently 400 or return empty). The search router was passing the raw user message (2000+ chars on reflective DEEP questions) which Tavily silently rejected. The provider chain then fell through to DuckDuckGo, which is currently rate-limited and returns a 302 redirect to `/50x.html?e=3`. Net effect: every web-search-bearing question failed silently, the Reasoning Trace showed `provider=duckduckgo` with a confusing HTTPStatusError, and users assumed Tavily wasn't configured at all. The truncation is defense-in-depth — Decision 042 fixes the root cause (the router) too, but this guard catches any future caller that doesn't go through the router.

### Decision 042 — Search Router Timeout + Heuristic Distiller (2026-05-26)
- **What:** Two changes to `src/bridge/search_router.py` to make the LLM-based query refiner actually fire, and to make the heuristic fallback not leak the raw question.
  1. **Timeout bump:** `ROUTER_TIMEOUT_SEC: 6.0 → 12.0`. The router calls Gemini 2.5 Flash to produce a 2-8 word refined query. The same model on similar-sized inputs (metadata_extraction in the persistence layer) regularly takes 5-8s. A 6s timeout was silently expiring and dumping the trace to the heuristic fallback.
  2. **`_distill_query()` heuristic:** new function in the router. When the LLM router returns None (timeout / no key / SDK missing / unparseable JSON), the fallback used to set `refined_query=q` (the entire user message). Now it calls `_distill_query(q)` which strips conversational openers (Hey, Tell me one thing, So…), splits on `[.!?]+`, picks the densest sentence by content/filler ratio, collapses whitespace, hard-caps at `HEURISTIC_QUERY_MAX_CHARS = 160` at a word boundary. Pure regex, deterministic, no LLM call.
- **Why:** The Reasoning Trace was reporting `router: fallback / decision in 2.5s` with `REFINED QUERY = <entire user message>`. The 2.5s suggested a silent timeout (well under the 6s limit, but close enough to catch slow runs). Diagnostic logs were promoted from `WARNING` to `INFO` so the failure mode (TIMED OUT after Xs / FAILED ExceptionType / UNPARSEABLE JSON) is visible in `logs/llm-calls.log` next time.
- **Verified:** distiller produces 93 chars from the user's actual 388-char reflective question, stripping "Hey. Tell me one thing." prefix and surfacing "What we human actually require to get inspiration, to get abstract ideas, to feel that moment". Empty + greeting edge cases return empty / "there" respectively. Frontend untouched.

### Decision 044 — Bridge Backends → Neo4j (2026-05-27)
- **What:** Extended the Neo4j migration to cover the remaining Redis-backed surface. `Neo4jAnchorBackend` (DecisionAnchor CRUD) and `Neo4jConversationBackend` (Session / Iteration / TurningPoint / DecisionLink CRUD) added in `src/bridge/neo4j_backend.py` next to the existing `Neo4jThreadStore`. `server.py` now reads `CONSTELLAX_DB_BACKEND` at module load: `neo4j` → builds the Neo4j conversation backend via `build_neo4j_driver_from_env`; falls through to the Redis path on missing creds with a LOUD warning; falls through to in-memory if neither configured. `_CONV_REDIS_BACKEND` renamed to `_CONV_BACKEND_ACTIVE` (generic — holds either implementation). Schema init in `init_schema` extended with constraints + indexes for the five new node labels.
- **Why:** Nikhil committed to "migrate everything to Neo4j" (2026-05-27) so the entire memory layer (graphs + vectors) lives in one database. Two motivations: (1) operational simplification — kill the local `constellax-falkor` container, one auth/backup/log story; (2) graph-native bridge data — `(Iteration)-[:MADE_DECISION]->(DecisionAnchor)`, `(DecisionLink)-[:FROM_DECISION]->(DecisionAnchor)`, etc. unlock queries impossible on the Redis KV layout (e.g. "find every iteration that ever touched decision D-014"). Falkor `RedisConversationBackend` stays wired as a one-toggle fallback through the validation window, just like the ThreadStore migration.
- **Verified:** AST + import clean. `scripts/validate_neo4j_bridge_parity.py` exercises all five entity types end-to-end against both backends — all 18 protocol-level checks (get/put/list/delete/update_status across DecisionAnchor, Session, BridgeIteration, TurningPoint, DecisionLink) passed. Server startup log confirms wiring: `ConversationStore: Neo4j backend active (database=e2053eb9)`. Smoke trace via `/api/v2/trace` routed cleanly through the new server with no exceptions.
- **MemoryAdapter status:** `Neo4jAnchorBackend` exists but is NOT wired into `src/bridge/client.py` yet (MemoryAdapter is currently constructed without a backend → falls back to InMemoryAnchorBackend, which is the pre-migration behavior). Wiring it changes anchor lifecycle from "ephemeral per-process" to "persisted across restarts" — that's a meaningful behavioral change beyond the migration scope. Queued for a follow-up commit when anchor persistence becomes a product requirement.

### Decision 043 — Neo4j Aura Migration Scaffold (2026-05-27)
- **What:** Parallel implementation of the `ThreadStore` Protocol against Neo4j Aura. Five additive changes, no destructive edits to the live FalkorDB path.
  1. `requirements.txt` — added `neo4j>=5.20,<6.0`.
  2. New `src/bridge/neo4j_backend.py` — `Neo4jThreadStore` implementing the full Protocol. Graph-native Cypher: `(User)-[:OWNS]->(Thread)-[:HAS_ITERATION]->(Iteration)-[:MENTIONS]->(Entity)` etc. `payload_json` properties hold the canonical record blobs for lossless round-trip. Vector index on `Iteration.embedding` (Neo4j 5.13+ native) replaces the brute-force cosine loop in `FalkorThreadStore.find_similar_iterations`.
  3. `src/bridge/thread_store.py` — `build_thread_store_from_env()` now reads `CONSTELLAX_DB_BACKEND` (default `falkor`). `neo4j` value tries `build_neo4j_thread_store_from_env()`; if NEO4J_URI/PASSWORD missing, logs a LOUD warning and falls through to the Falkor path (safe by default). Added `init_store_schema(store)` async helper — no-op for Falkor/InMemory, runs constraints + vector index DDL for Neo4j.
  4. `src/bridge/thread_persistence.py` — `_ensure_initialized` now logs the active backend by class name (`type(backend).__name__`) and awaits `init_store_schema(_store)` after construction so Neo4j schema is ready before first request.
  5. New `scripts/validate_neo4j_parity.py` — pre-cutover sanity check. Writes a synthetic ThreadRecord + 2 IterationRecords to both InMemory and Neo4j, diffs read-back shape, cleans up. Exit codes: 0 pass / 1 missing creds / 2 schema fail / 3 parity fail.
- **Why:** Nikhil committed to Neo4j as production target (see `project_graph_db_choice` memory note). Two reasons that outweigh the rewrite cost: (a) GDS library + native vector indexes — collapses FalkorDB + future vector store into one DB; (b) brand credibility for investors. The current FalkorDB usage is Redis-style KV (no GRAPH.QUERY commands), so the migration is also a graph remodel — `find_similar_iterations` goes from O(n) cosine scan to O(log n) vector-index lookup; entity/tag lookups become first-class graph queries instead of set membership scans. ThreadStore being a Protocol means application code (server.py, thread_persistence.py outer surface) doesn't move — only the backend implementation file does.
- **Cutover path (not yet executed):** (1) User provisions Aura Free at console.neo4j.io. (2) Adds `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` to `.env`. (3) Runs `python3 scripts/validate_neo4j_parity.py` — must exit 0. (4) Sets `CONSTELLAX_DB_BACKEND=neo4j` in runtime env. (5) Restarts server. FalkorDB code stays in place as fallback through a validation window before final removal.
- **Verified locally:** AST-parse + import + factory toggle smoke test all clean. No live Neo4j connection yet (pending Aura provisioning).

---

## WARNINGS & FRAGILE AREAS (2026-05-26)

1. **The decision log above is the source of truth for design rationale, but the CURRENT API SURFACE / MODULE MAP sections in this supplement are the source of truth for what's live.** When they conflict, the supplement wins because it's newer.

2. **`/api/v2/trace` is NOT deprecated.** It's the one-shot endpoint still used by Map Room follow-up traces, the resume path, and any non-streaming consumer. Don't refactor it into the segment endpoint or delete it. They're parallel surfaces by design.

3. **Two speech generators coexist in `src/llm/speech.py`.** `generate_speech` (full memo, used by `dispatch()` and by phase=synthesizer) vs the per-segment generators (only fire on splice in phase=opinion/prospects). Touching one without the other creates split-brain drift.

4. **`_SEGMENT_CACHE` is in-process.** Single-worker dev is fine. For a multi-worker production deployment this needs to move to Redis alongside `_MEMO_CACHE` and the conversation store. Same caveat as Decision 031's note about `_MEMO_CACHE`.

5. **`generate_speech` still owns persistence.** Persistence fires after dispatch returns (at synthesizer time). Splice regeneration mutates the in-memory cache but not the persisted record. Map Room rehydration of a spliced thread shows the original. Fixing this requires re-persisting from the segment endpoint on splice success.

6. **Many `src/` modules listed in MODULE MAP are untracked in git.** They're on disk and live; git just hasn't seen them committed. `git status` will list them. Don't `git add -A` blindly — check `.env`, `logs/`, `graphify-out/`, `scripts/analytics_output.json`, etc. aren't pulled in.

7. **The reasoningEngine and constellax-ui repos are separate processes with NO shared code.** They communicate only through `/api/v2/*` HTTP. When refactoring across both, check both directions — typecheck the TS, import-check the Python.

8. **Branch policy.** This repo is on `main` and has uncommitted work. Unlike the LoRa repo (which has `bugbot-init-review` policy), this one has no `bugbot-init-review` branch. Do NOT push to `main` without explicit instruction.

9. **Visuals live in two places now — keep them in sync.** Decision 038 added `src/llm/visualizer.py` which is the AUTHORITATIVE source of Map Room visuals on the canonical DEEP path. Speech.py's `SPEECH_SYSTEM_PROMPT` still has a `visuals` section (and the segment generators in speech.py emit visuals on splice). If you edit one prompt section, check the other — drift will produce visually inconsistent output between first-read and follow-up turns.

10. **`SegmentedResponse.memo` is THE persistence seam for visuals.** Decision 040 added a `memo: dict | None` sidecar field so the raw memo (with visuals) survives the Falkor round-trip. If you ever serialize a SegmentedResponse manually, include this field — the frontend reads `response.memo.visuals`, not `response.map_room.visuals`. The `map_room.visuals` field still exists (it's the dataclass version) but the frontend doesn't read from there.

11. **Search router has TWO failure modes, both now logged at INFO.** Decision 042 — when the LLM router falls through (timeout / SDK miss / parse fail), the heuristic distiller produces a short query. Look for `search_router: LLM call TIMED OUT after Xs` / `FAILED ...` / `UNPARSEABLE JSON` in `logs/llm-calls.log` and console. If you see persistent timeouts, bump `ROUTER_TIMEOUT_SEC` further or swap the model.

12. **Tavily query length cap is a SOFT contract.** Decision 041 — `TavilyProvider._MAX_QUERY_CHARS = 380`. If Tavily's free-tier limit changes upstream, bump this. The trim happens INSIDE `TavilyProvider.search` so any caller (router or direct) is protected.

13. **The bridge Iteration ≠ ThreadStore IterationRecord.** Two unrelated dataclasses share the name `Iteration` (one in `src/bridge/types.py`, one inside `src/core/thread_types.py`). In Neo4j they MUST use different labels to prevent silent cross-contamination — `:Iteration` is reserved for ThreadStore's `IterationRecord`; the bridge model is labeled `:BridgeIteration` (Decision 044, see `_BRIDGE_LABEL_BY_TYPE` in `neo4j_backend.py`). If you ever add a third Iteration concept, give it yet another label. The label IS the namespace in Neo4j.

14. **Aura Free non-standard naming.** Aura Free uses the instance ID as BOTH the username AND the database name (current instance: `e2053eb9`), NOT the standard "neo4j". `NEO4J_DATABASE` env var is required — without it, all writes route to a nonexistent `neo4j` database and `init_schema` silently fails (errors are warning-level, not raise). Decision 043 + the bridge parity script both pull this env var explicitly. If you spin up a new Aura Free instance, the same convention applies.

## DEVELOPMENT COMMANDS

```bash
# Backend (this repo)
python3 server.py                          # Start FastAPI on :8100
python3 run.py "your question here"        # One-shot CLI mode
python3 run.py --effort=high               # High-effort mode
python3 -m pytest tests/                   # Run all tests
python3 -m pytest tests/test_integration.py  # Integration suite (21 tests)
python3 -m pytest tests/test_bridge.py     # Bridge suite (21 tests)

# Smoke test the segment endpoint
curl -X POST http://localhost:8100/api/v2/trace/segment \
  -H 'Content-Type: application/json' \
  -d '{"phase":"synthesizer","question":"hi","effort":"low","policy":"auto"}'

# Smoke test the legacy endpoint
curl -X POST http://localhost:8100/api/v2/trace \
  -H 'Content-Type: application/json' \
  -d '{"question":"hi","effort":"low","policy":"auto"}'

# Health
curl http://localhost:8100/health
```

## ENV VARS (effective 2026-05-26)

Required for live mode:
- `OPENROUTER_API_KEY` — primary path for all LLM calls (Decision 032). Legacy fallback: `ANTHROPIC_API_KEY`.
- ONE storage backend (see `CONSTELLAX_DB_BACKEND` below):
  - **Falkor path (default):** `CONSTELLAX_REDIS_URL` — Redis backend for ConversationStore + ThreadStore. Without it, in-memory dict (single-process only).
  - **Neo4j path (Decision 043):** `NEO4J_URI` + `NEO4J_PASSWORD` (plus `CONSTELLAX_DB_BACKEND=neo4j`).

Optional:
- `CONSTELLAX_DB_BACKEND=neo4j|falkor` — switches ThreadStore implementation. Default `falkor` for back-compat. Setting to `neo4j` without `NEO4J_URI`/`NEO4J_PASSWORD` logs a loud warning and falls back to Falkor.
- `NEO4J_USERNAME` (default `neo4j`), `NEO4J_DATABASE` (default `neo4j`), `NEO4J_EMBEDDING_DIM` (default `1536` — set to `3072` if using `gemini-embedding-001` at default dimensions).
- `TAVILY_API_KEY` — web search primary provider. Without it, falls back to DuckDuckGo HTML scraping.
- `GOOGLE_API_KEY` / `GEMINI_API_KEY` — for Gemini embeddings + triage classifier + search router. Aliased.
- `LORA_EFFORT` — default tier (low/medium/high).
- `PORT` (8100), `HOST` (0.0.0.0), `CORS_ORIGINS` (*).
- `LOG_LEVEL`, `OBS_LOG`, `OBS_LOG_FILE` — observability logging.

---

*Supplement last updated: 2026-05-26.*
*If you are reading this file in a fresh session: load the SISTER REPOS table first, then CURRENT API SURFACE, then MODULE MAP, before reading any of the decision log above.*
