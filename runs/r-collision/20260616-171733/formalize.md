# R1 Formalizer — results

- **model:** `deepseek/deepseek-r1`
- **total cost:** $0.0812

> Open VS Code preview (⇧⌘V) to render the math.


---

## blend-01 — formalizable: **partial** · confidence 0.85 · $0.0118

**Reference lookups R1 requested:**
- Shannon entropy formula discrete probability distribution
- circuit breaker design pattern failure threshold mechanisms
- ART network vigilance parameter complement coding

### Formal core

Let $\mathcal{G}$ be the set of active gaps. For each gap $g \in \mathcal{G}$, define:
- $F_g \in \mathbb{Z}_{\geq 0}$: failure count for $g$, incremented per failed attempt.
- $\mathcal{A}_g$: multiset of approach vectors used in failed attempts on $g$.

**Approach vector entropy** (Shannon entropy of approach distribution):  
Discretize approach vectors into $k$ classes $c_1, \dots, c_k$ via hashing (e.g., SHA-256 truncation). For $g$, compute class probabilities:  


$$
p_i(g) = \frac{|\{a \in \mathcal{A}_g \mid \text{class}(a) = c_i\}|}{F_g} \quad \forall i \in \{1,\dots,k\}
$$

  
Shannon entropy $H_g$ is:  


$$
H_g = -\sum_{i=1}^k p_i(g) \log_2 p_i(g) \quad \text{(with } 0 \log_2 0 \equiv 0\text{)}
$$

  
*Derivation:* Direct application of discrete entropy to approach-class distribution. Measures diversity of attempts; maximal when classes are equiprobable, minimal when one class dominates.  

**Governor decision** (using thresholds $F_{\text{thresh}}$, $H_{\text{thresh}}$):  


$$
\text{action}(g) = 
\begin{cases} 
\text{terminate} & \text{if } F_g \geq F_{\text{thresh}} \text{ and } H_g \geq H_{\text{thresh}} \\
\text{re-route} & \text{if } F_g \geq F_{\text{thresh}} \text{ and } H_g < H_{\text{thresh}} \\
\text{open} & \text{otherwise}
\end{cases}
$$

  
*Derivation:* Maps blend's quadrant logic to piecewise conditions. Termination requires high failures *and* high diversity; re-route triggers on high failures *with* low diversity.  

**Re-route mechanism** (forced novelty via cost adjustment):  
For $g$ in re-route state, define cost function $\text{cost}_g: \{\text{approach classes}\} \to \mathbb{R}^+$. Update:  


$$
\text{cost}_g(c_i) \leftarrow \text{cost}_g(c_i) + \Delta \cdot \mathbf{1}_{p_i(g) > 0} \quad \forall i
$$

  
where $\Delta > 0$ is a fixed cost increment. Subsequent routing minimizes $\sum \text{cost}_g(c_i)$.  
*Derivation:* ART-inspired cost increment penalizes tried classes. High cost crowds them out, forcing novel approaches per blend.  

**Termination**: Removes $g$ from $\mathcal{G}$.

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $\mathcal{G}$ | set of active gaps | from gap-centric decision logic (Source Methods) |
| $F_g$ | failure count per gap | circuit breaker failure counting (wander-P02-005) |
| $\mathcal{A}_g$ | multiset of approach vectors | attempt fingerprinting (wander-P02-003) |
| $\text{class}(a)$ | discretization of approach vector | hashing to finite classes (OmniCoreAgent fingerprinting) |
| $p_i(g)$ | probability of approach class $c_i$ | required for entropy computation |
| $H_g$ | Shannon entropy of approach distribution | diversity measure (blend thesis) |
| $F_{\text{thresh}}$, $H_{\text{thresh}}$ | failure/entropy thresholds | circuit breaker tripping (wander-P02-005) |
| $\text{cost}_g$ | routing cost per approach class | ART-style capacity control (wander-S4f31ec-002) |
| $\Delta$ | cost increment | magnitude of penalty for exhausted approaches (ART complement coding) |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| Shannon entropy formula (https://tdhopper.com/blog/entropy-of-a-discrete-probability-distribution) | Measures uncertainty in discrete distributions. | Quantifies approach diversity as entropy; justifies $H_g$ as a formal axis. |
| Circuit breaker threshold logic (https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker) | Trips based on local failure count. | Provides $F_g \geq F_{\text{thresh}}$ condition; separates per-gap decisions. |
| ART complement coding (https://open.bu.edu/bitstreams/6f3a7ac6-f2b6-4ba4-a613-2836786eb908/download) | Full categories raise routing costs against themselves. | Models cost increment $\Delta$ to force novelty during re-route. |

### Left qualitative (prose, not faked)

- **Approach vector definition**: Left qualitative; blend does not specify structure (e.g., function signatures, embeddings). Discretization via hashing is assumed but not formalized beyond class existence.
- **Routing implementation**: Cost-minimization mechanism abstracted; no formalization of bidding/selection (inspired by ART but not mathematically mapped).
- **Threshold calibration**: $F_{\text{thresh}}$, $H_{\text{thresh}}$, $\Delta$ left as parameters; blend provides no derivation method.

### Test (predict / confirm / falsify)

predicts:  
For a gap with $F_g \geq F_{\text{thresh}}$:  
- If $H_g \geq H_{\text{thresh}}$, future attempts cease (termination).  
- If $H_g < H_{\text{thresh}}$, next attempt uses a class $c_j$ with $\text{cost}_g(c_j)$ minimized (i.e., not in $\mathcal{A}_g$).  
confirms:  
Observe termination when high-$F_g$ gaps show near-uniform class distribution ($H_g \approx \log_2 k$).  
falsifies:  
Termination occurring for high-$F_g$ gap with low $H_g$ (e.g., 10 identical failures), or re-route not forcing a novel class.

### Where it breaks

1. **Non-discrete approaches**: Entropy computation fails if approach vectors are continuous (assumes discretization).
2. **Cost minimization impractical**: If no novel classes exist, re-route stalls (blend assumes unsaturated diversity).
3. **State dependence**: Ignores temporal ordering of attempts; entropy/cost updates may erase history.
4. **Threshold sensitivity**: Fixed $H_{\text{thresh}}$ may misclassify if $k$ (class count) is unknown/dynamic.

### Honest caveat

Entropy thresholds ($H_{\text{thresh}}$) lack calibration; real systems need $k$-dependent normalization.


---

## blend-02 — formalizable: **partial** · confidence 0.75 · $0.0169

**Reference lookups R1 requested:**
- ART complement coding formal definition
- state machine autonomy level formal model
- reversible system snapshots equations

### Formal core

The blend formalizes as a dynamical system combining ART routing, autonomy state machines, and reversible mutations. Agents $a_i \in A$ pursue gaps $g_j \in G$. Key equations:

1. **Autonomy radius** (organism-core):  
   Let $s_a(t) = \frac{1}{N_a(t)}\sum_{k=1}^{N_a(t)} o_k$ be agent $a$'s success rate over $N_a(t)$ trials, with $o_k \in \{0,1\}$ outcomes. Radius $r_a(t)$ is bounded by:  
   

$$
r_a(t) = r_{\text{min}} + (r_{\text{max}} - r_{\text{min}}) \cdot s_a(t)
$$

  
   *Derivation*: Maps "earned autonomy" from organism-core. Radius expands with proven success, clamped to $[r_{\text{min}}, r_{\text{max}}]$ to prevent degeneracy. Breaks if success isn't measurable.

2. **Bias budget** (governor checkpoints):  
   At dispatch time $t$, agent $a$ receives bias:  
   

$$
b_a(t) = \kappa \cdot r_a(t) \quad \text{(where } \kappa > 0 \text{ is a scaling constant)}
$$

  
   *Derivation*: Encodes "bias budget = f(agent_track_record)". Budget proportional to autonomy radius. Breaks if radius doesn't correlate with capability.

3. **Routing threshold** (ART complement coding):  
   Let $U_j(t)$ be utilization (fraction of wave capacity consumed) by gap $g_j$. Threshold for routing to $g_j$:  
   

$$
\theta_j(t) = \theta_0 + \gamma U_j(t) \quad \text{(with } \theta_0, \gamma > 0\text{)}
$$

  
   *Derivation*: From ART's "effective routing threshold rises" via complement coding. $U_j(t)$ proxies "accumulated routed agents". Linearity simplifies vigilance increase. Breaks if capacity isn't quantifiable.

4. **Dispatch condition**:  
   Agent $a$ can route to gap $g_j$ iff:  
   

$$
b_a(t) \geq \theta_j(t)
$$

  
   *Derivation*: Implements "each dispatch must clear a higher bar". Fails if thresholds grow faster than bias budgets.

5. **Utilization update**:  
   If $a$ routes to $g_j$:  
   

$$
U_j(t+1) = U_j(t) + \frac{r_a(t)}{\sum_{k \in A} r_k(t)} \cdot \Delta U
$$

  
   where $\Delta U$ is capacity consumed per agent.  
   *Derivation*: "Gap consumes wave capacity" scaled by agent's radius (larger agents consume more). $\Delta U$ normalizes total capacity. Breaks if radius doesn't affect resource use.

6. **Rollback and demotion** (forge-os-core):  
   After excursion outcome $o \in \{0,1\}$:  
   - If $o = 0$ (failure):  
     - Restore pre-dispatch system snapshot $S(t)$.  
     - Demote radius: $r_a(t+1) = \max(r_{\text{min}}, \beta r_a(t))$ ($\beta < 1$).  
   - If $o = 1$ (success):  
     - Update success rate: $s_a(t+1) = \frac{s_a(t) N_a(t) + 1}{N_a(t) + 1}$.  
   *Derivation*: Reversible snapshots enable rollback; demotion penalizes failure. Assumes state restoration is possible.

7. **Circuit breaker** (forge-os-core):  
   Suspend agent after $K$ consecutive failures.  
   *Derivation*: Prevents "runaway distortion" from thrashing. $K$ is a robustness parameter.

**Diversity equilibrium**:  
As $U_j \uparrow$, $\theta_j \uparrow$, diverting agents to other gaps. The utilization vector $\mathbf{U}(t)$ converges to a fixed point where:  


$$
\min_j \theta_j \approx \max_j \theta_j
$$

  
*Derivation*: From "diversity is the equilibrium of competing pressures". High-utilization gaps self-limit via threshold growth. Stability requires $\gamma > 0$.

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| A | Set of agents | From "agents" in mechanism |
| G | Set of gaps | From "gaps" in thesis |
| r_a(t) | Autonomy radius of agent a at time t | Maps "earned autonomy" (organism-core); bounded by track record |
| s_a(t) | Success rate of agent a | Quantifies "track record" for radius calculation |
| b_a(t) | Bias budget for agent a | Encodes "bias budget" granted at checkpoints |
| U_j(t) | Utilization of gap g_j | Measures "wave capacity consumed" (ART crowding) |
| θ_j(t) | Routing threshold for gap g_j | Formalizes "effective routing threshold rises" (complement coding) |
| κ | Bias scaling constant | Scales radius to bias units |
| θ_0 | Base routing threshold | Minimum threshold when U_j=0 |
| γ | Utilization sensitivity | Controls threshold growth rate with utilization |
| β | Demotion factor | Penalty for failure during rollback |
| K | Circuit breaker threshold | From "repeated failures" in forge-os-core |
| o | Excursion outcome (0/1) | Success/failure of gap-chase |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| ART vigilance model | Implements complement coding's "full workers raise vigilance threshold" | Formalizes gap self-limiting (threshold θ_j ∝ U_j) |
| Finite state machines (Wikipedia) | Models autonomy levels in organism-core | Justifies radius r_a as a state variable updated by outcomes |
| Reversible snapshots (Greenberg-Hastings systems) | Enables rollback in forge-os-core | Provides mathematical basis for state restoration |
| Success rate as trust metric (RL literature) | Quantifies track record for autonomy | Warrants s_a(t) for radius calculation |

### Left qualitative (prose, not faked)

- "Governor checkpoints framing state": Not formalized; abstract governance layer without mathematical specification.
- "Integration track record": Reduced to success rate; qualitative aspects like task complexity omitted.
- "Clean integration" of excursions: Outcome o is binary; nuanced failure modes left qualitative.
- "Three competing pressures": Described but not quantified; equilibrium is inferred, not derived from first principles.

### Test (predict / confirm / falsify)

predicts: Utilization variance across gaps decreases over time; high-success agents dominate initially but suppress their own gap's future access.
confirms: Simulation shows negative correlation between gap utilization and subsequent routing rate (Pearson r < -0.7).
falsifies: If a gap's utilization exceeds 90% without routing suppression (θ_j remains low), or if failures don't reduce agent radius.

### Where it breaks

- If wave capacity is unbounded (violating ART crowding), U_j(t) doesn't increase θ_j(t).
- If success outcomes are non-Bernoulli, s_a(t) doesn't reflect reliability.
- If snapshot restoration isn't perfect (e.g., side effects), rollback fails.
- If γ = 0, thresholds don't rise, leading to monoculture.

### Honest caveat

The utilization update assumes radius directly consumes capacity; real systems may have nonlinear resource coupling.


---

## blend-03 — formalizable: **partial** · confidence 0.85 · $0.0355

**Reference lookups R1 requested:**
- CUSUM control chart algorithm
- binomial hypothesis test for proportions
- fixed priority scheduling real-time systems

### Formal core

The system operates in discrete cycles $t = 1, 2, \dots$. For a **single conflict class** (extensible to multiple independent classes):  

1. **Detector Inputs** (goal-stripped baseline):  
   - $A_t \in \{0,1\}$: Goal-aware (GA) detector report at $t$ (1 = reports gap)  
   - $B_t \in \{0,1\}$: Goal-blind (GB) detector report at $t$ (1 = reports gap).  
   By the digital-stoic split, $B_t$ is independent of goal context.  

2. **Orchestration Mode**:  
   $M_t \in \{\text{CONV}, \text{DIV}\}$, where:  
   - CONV: Convergence mode (GA should dominate)  
   - DIV: Divergence mode (GB should dominate)  

3. **Inversion State**:  
   $I_t \in \{0,1\}$, triggered by meta-monitor. Duration = $K$ cycles.  

4. **Priority Resolution**:  
   Winner $W_t$ determined as:  
   

$$
W_t = 
   \begin{cases} 
   \text{GB} & \text{if } I_t = 1 \\
   \text{GA} & \text{if } I_t = 0 \text{ and } M_t = \text{CONV} \\
   \text{GB} & \text{if } I_t = 0 \text{ and } M_t = \text{DIV} \\
   \varnothing & \text{if } A_t = B_t = 0
   \end{cases}
$$

  
   - **Win recorded** only if $A_t = 1$ or $B_t = 1$ (non-empty conflict).  

5. **Meta-Monitor (Drift Auditor)**:  
   At each $t$, collect eligible cycles:  
   

$$
\mathcal{D}_t = \{\tau \in [t-N+1, t] : M_\tau = \text{DIV}, I_\tau = 0, W_\tau \neq \varnothing\}
$$

  
   Let $n_t = |\mathcal{D}_t|$, $x_t = \sum_{\tau \in \mathcal{D}_t} \mathbb{I}_{W_\tau = \text{GA}}$.  
   Test: $H_0: p = p_0^{\text{DIV}}$ vs. $H_1: p > p_0^{\text{DIV}}$, where:  
   - $p_0^{\text{DIV}}$: Expected GA win rate in DIV mode (historical baseline).  
   - Test statistic: $p$-value from **binomial test**:  
     

$$
p\text{-val} = \mathbb{P}(X \geq x_t \mid X \sim \text{Binomial}(n_t, p_0^{\text{DIV}}))
$$

  
     

$$
= \sum_{k=x_t}^{n_t} \binom{n_t}{k} (p_0^{\text{DIV}})^k (1 - p_0^{\text{DIV}})^{n_t - k}
$$

  
   - Trigger inversion if $p\text{-val} < \alpha$ (significance level).  

6. **Inversion Trigger**:  
   If triggered at $t$, set $I_{t+1:t+K} = 1$ (GB wins all modes for $K$ cycles).  

**Derivation**:  
- **Input independence**: GB’s goal-stripped $B_t$ ensures uncontaminated baseline (digital-stoic split).  
- **Win recording**: GA wins in DIV mode **iff** $A_t = 1 \land B_t = 0$ (GB abstains), enabling drift detection.  
- **Binomial test**: Detects excess GA wins (motivated convergence) during DIV. Matches "anomalous streak" as deviation from expected rate $p_0^{\text{DIV}}$.

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $t$ | Cycle number | Discrete time index |
| $A_t$ | GA detector report (gap ranking) | Goal-aware input; 1 if reports conflict |
| $B_t$ | GB detector report (gap ranking) | Goal-blind input; 1 if reports conflict; goal-stripped per digital-stoic split |
| $M_t$ | Orchestration mode (CONV/DIV) | Mode signal determining expected winner |
| $I_t$ | Inversion state (0/1) | Meta-layer override; 1 forces GB win |
| $W_t$ | Winner (GA/GB/∅) | Lattice output per cycle; logged in win-history |
| $\mathcal{D}_t$ | Eligible cycles for meta-monitor | DIV-mode, no-inversion, non-empty cycles in window |
| $n_t$ | Number of eligible cycles | Sample size for hypothesis test |
| $x_t$ | GA wins in eligible cycles | Test statistic; counts anomalous wins |
| $p_0^{\text{DIV}}$ | Expected GA win rate in DIV | Historical baseline; drift benchmark |
| $N$ | Meta-monitor window size | Design parameter; memory span |
| $\alpha$ | Significance level | False-positive rate for inversion |
| $K$ | Inversion duration | Design parameter; temporary override length |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| Binomial test | Tests proportion of GA wins against expected rate $p_0^{\text{DIV}}$ | Detects motivated convergence; applies as wins are Bernoulli trials. (https://en.wikipedia.org/wiki/Binomial_test) |

### Left qualitative (prose, not faked)

- **Multi-class extension**: Assumed independent per conflict class (no interaction modeled).  
- **Mode setting**: Orchestration mode $M_t$ is exogenous (not modeled).  
- **Parameter calibration**: $p_0^{\text{DIV}}, N, \alpha, K$ set from historical data/design (not derived).  
- **Detector internals**: GA/GB gap-ranking methods left abstract.

### Test (predict / confirm / falsify)

predicts: In DIV mode (no inversion), if GA win rate exceeds $p_0^{\text{DIV}}$ (binomial test, $p < \alpha$), inversion triggers for $K$ cycles.  
confirms: Observe inversion trigger when GA wins in ≥ $x^*$ of last $N$ DIV cycles, where $\mathbb{P}(\text{Binomial}(N, p_0^{\text{DIV}}) \geq x^*) < \alpha$.  
falsifies: If GA wins in DIV at rate $\gg p_0^{\text{DIV}}$ but no inversion triggers, or triggers when rate $\leq p_0^{\text{DIV}}$.

### Where it breaks

1. **Dependence**: Breaks if conflict classes interact or detectors correlate.  
2. **Non-stationarity**: Fails if $p_0^{\text{DIV}}$ drifts (e.g., due to system evolution).  
3. **Mode coupling**: Invalid if $M_t$ depends on prior $W_t$ (feedback not modeled).  
4. **Window selection**: Poor $N$ causes delayed/missed detection (e.g., small $N$ misses slow drift).

### Honest caveat

Assumes GA wins in DIV mode **only** when GB abstains ($B_t = 0$); if other mechanisms exist, formalization fails.


---

## blend-04 — formalizable: **partial** · confidence 0.85 · $0.0171

**Reference lookups R1 requested:**
- stopping rules multi-criteria optimization
- model capacity saturation information theory
- convergence detection iterative algorithms

### Formal core

The blend formalizes as an iterative system with discrete cycles $t = 0, 1, 2, \dots$. At each cycle $t$, compute three scalars:
1. **Novelty Rate** $N_t \in \mathbb{Z}_{\geq 0}$:  
   Count of new gap types observed at $t$, derived from the set of gap types $G_t$ (types represented by the system up to $t$).  
   $N_t = |G_t \setminus G_{t-1}|$ with $G_{-1} = \emptyset$.  
   *Justification*: Measures capacity saturation (goal-blind). From [wander-S589f28-003], capacity is fixed; saturation occurs when no new types are representable.  

2. **DoD Satisfaction** $D_t \in [0, 1]$:  
   Score against a re-derived Definition of Done (DoD) at $t$.  
   $D_t = \text{score}(O_t, \text{DoD}_t)$, where $O_t$ is the output at $t$, and $\text{DoD}_t$ is the goal-aware target researched at $t$.  
   *Justification*: Measures coverage completeness [wander-P02-001]. DoD is re-derived each cycle, making the target dynamic.  

3. **Change Rate** $C_t \in \mathbb{R}_{\geq 0}$:  
   Output-change magnitude, computed via projections. Let $P_t$ be a projection (e.g., vector summary) of the ledger $L_t$ (append-only event history).  
   $C_t = \|P_t - P_{t-1}\|$ (Euclidean norm).  
   *Justification*: Measures diminishing returns [wander-P06-003]; $P_t$ is a fold over $L_t$.  

**Halting Condition**:  
Halt at $t$ if either:  
- **Triangulated Halt**: $N_t \leq \theta_N$ $\land$ $D_t \geq \theta_D$ $\land$ $C_t \leq \theta_C$ (all thresholds met),  
- **Hard Cap**: $t \geq T_{\text{max}}$ (unilateral override).  
Otherwise, continue.  
*Justification*: Orthogonal signals act as mutual vetoes [Blend THESIS]. $\theta_N, \theta_D, \theta_C, T_{\text{max}}$ are preconfigured thresholds.  

**Derivation**:  
- Independence: Signals computed in distinct "reference frames" (goal-blind novelty, goal-aware DoD, temporal output-change) ensure uncorrelated failures [Blend EMERGENT STRUCTURE].  
- Veto Logic: Formalizes "any single signal can VETO" via AND condition; hard cap is OR condition [Blend MECHANISM].

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $t$ | cycle index | Discrete time for iterative computation [wander-P06-003] |
| $G_t$ | set of represented gap types | Capacity ceiling for novelty; fixed architectural limit [wander-S589f28-003] |
| $N_t$ | novelty rate (new gap-types per cycle) | Measures saturation; $N_t = 0$ implies no new distinctions [wander-S589f28-003] |
| $O_t$ | system output | Artifact scored against DoD [wander-P02-001] |
| $\text{DoD}_t$ | Definition of Done at $t$ | Dynamically re-derived target [wander-P02-001] |
| $D_t$ | DoD satisfaction score | Completeness relative to $\text{DoD}_t$ [wander-P02-001] |
| $L_t$ | event ledger | Append-only record; basis for projections [wander-P06-003] |
| $P_t$ | projection of $L_t$ | State summary via fold operation [wander-P06-003] |
| $C_t$ | output-change rate | Diminishing returns signal; $\|P_t - P_{t-1}\|$ quantifies "meaningful change" [wander-P06-003] |
| $\theta_N$ | novelty threshold | $\theta_N \approx 0$ (saturation) [wander-S589f28-003] |
| $\theta_D$ | DoD threshold | $\theta_D \in [0,1]$ (e.g., 0.95) [wander-P02-001] |
| $\theta_C$ | change threshold | $\theta_C > 0$ (e.g., 0.01) [wander-P06-003] |
| $T_{\text{max}}$ | hard cap | Maximum cycles or compute [Blend MECHANISM] |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| Manopt.jl | StopWhenAll criterion | Matches AND logic for multi-criteria halt; justifies mutual veto [https://manoptjl.org/stable/plans/stopping_criteria] |
| Springer MOO article | Stopping criteria affect performance/solution | Warns against single-criterion stops; supports orthogonality [https://link.springer.com/article/10.1007/s00521-021-05805-1] |
| Law of Information Capacity | Bounded representable distinctions | Justifies $N_t$ saturation as architectural limit [https://www.emergentmind.com/topics/law-of-information-capacity] |
| ScienceDirect Information Theory | Quantifies representable symbols | Underpins capacity ceiling for $G_t$ [https://www.sciencedirect.com/topics/mathematics/information-theory] |
| MIT CSAIL convergence analysis | Iterative methods with non-convex objectives | Validates $C_t$ for output stabilization [https://www.csail.mit.edu/research/understanding-convergence-iterative-algorithms] |

### Left qualitative (prose, not faked)

- **Gap type definition**: What constitutes a "gap type" is system-specific (e.g., semantic, syntactic). Not mathematically defined.  
- **DoD derivation**: How $\text{DoD}_t$ is researched from "external sources" is qualitative; no formal input-output mapping given.  
- **Projection function**: Exact form of $P_t = \text{fold}(L_t)$ (e.g., embeddings, aggregates) is implementation-dependent.  
- **Agent internals**: Stateless agent logic and conductor scheduling (subscriptions/ticks) are abstracted.

### Test (predict / confirm / falsify)

predicts: System halts iff (all $N_t \leq \theta_N$, $D_t \geq \theta_D$, $C_t \leq \theta_C$) OR ($t \geq T_{\text{max}}$).  
confirms: In a run, halt occurs at $t$ where either (a) all three thresholds are met simultaneously, or (b) $t = T_{\text{max}}$.  
falsifies: If halt occurs at $t < T_{\text{max}}$ without all thresholds met, or continues past $T_{\text{max}}$.

### Where it breaks

- **Capacity unbounded**: If new gap types never saturate ($N_t > \theta_N$ indefinitely), halt relies on hard cap.  
- **Moving target instability**: If $\text{DoD}_t$ changes such that $D_t$ never converges to $\theta_D$, halt fails.  
- **Oscillations**: If $C_t > \theta_C$ persistently (e.g., cyclic outputs), change-rate veto prevents halt.  
- **Threshold tuning**: Poor $\theta_*$ choices cause early/late termination; thresholds are static.

### Honest caveat

DoD satisfaction $D_t$ assumes a normalized score; real systems may lack a well-defined metric.
