# R1 Formalizer — results

- **model:** `deepseek/deepseek-r1`
- **total cost:** $0.09

> Open VS Code preview (⇧⌘V) to render the math.


---

## blend-01 — formalizable: **partial** · confidence 0.85 · $0.0259

**Reference lookups R1 requested:**
- discrete event rate estimation sliding window
- second derivative of count process time series
- structural admissibility condition graph assembly

### Formal core

The fertility signal is formalized as a discrete-time stochastic process evaluated through derivatives of the emergence rate. Let:  
- $\mathcal{G}_t$ be the assembly state (graph/simplicial complex) at event time $t$  
- $f_t$ be the fragment arriving at $t$  
- $\mathcal{A}: \mathcal{G}_{t-1} \times f_t \to \{0,1\}$ be the admissibility function (structural compatibility check)  

**Admissibility Gate (Zeroth Derivative):**  


$$
A_t = \begin{cases} 
1 & \text{if } \exists \text{ isomorphic embedding } \iota: f_t \hookrightarrow \mathcal{G}_{t-1} \\
0 & \text{otherwise}
\end{cases}
$$

  
If $A_t = 0$, $f_t$ is discarded; if $A_t = 1$, proceed.  

**Link Classification:**  
For admitted fragments ($A_t = 1$):  
- Emergence links $E_t = |\{ e \in \text{edges}(f_t) : e \not\subset \mathcal{G}_{t-1} \}|$ (new cross-connections)  
- Reinforcement links $R_t = |\{ e \in \text{edges}(f_t) : e \subset \mathcal{G}_{t-1} \}|$ (existing node confirmation)  

**Emergence Rate (First Derivative):**  
Over sliding window $w$, the emergence rate is:  


$$
r_t = \frac{1}{w} \sum_{k=t-w+1}^{t} A_k E_k \quad \text{(count per event)}
$$

  
with $r_t := 0$ for $t < w$.  

**Rate Acceleration (Second Derivative):**  
Using central finite differences:  


$$
\dot{r}_t = r_t - r_{t-1} \quad \text{(first difference)}
$$

  


$$
\ddot{r}_t = \dot{r}_t - \dot{r}_{t-1} = r_t - 2r_{t-1} + r_{t-2} \quad \text{(second difference)}
$$

  
valid for $t \geq w+2$.  

**Sterility Condition:**  


$$
\text{Sterile}_t = \begin{cases} 
1 & \text{if } A_t = 1 \,\land\, r_t > 0 \,\land\, \dot{r}_t < 0 \,\land\, \ddot{r}_t \leq 0 \\
0 & \text{otherwise}
\end{cases}
$$

  

**Fertility Signal:**  
The triplet $\Phi_t = \left( A_t, \, \dot{r}_t, \, \ddot{r}_t \right)$ defines fertility at $t$, with:  
- $A_t$: Admissibility (zeroth derivative)  
- $\dot{r}_t$: Emergence rate trend (first derivative)  
- $\ddot{r}_t$: Acceleration/deceleration (second derivative)

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $t$ | fragment arrival event index | discrete time variable for event sequence |
| $\mathcal{G}_t$ | assembly state (graph/simplicial complex) | represents topological structure from fragments |
| $f_t$ | arriving fragment | input data unit |
| $\mathcal{A}$ | admissibility function | structural compatibility (Chern-Weil/De Bruijn inspiration) |
| $A_t$ | admissibility gate output | binary check from mechanism step (1) |
| $E_t$ | emergence link count | new cross-connections per mechanism step (2) |
| $w$ | sliding window size | parameter for rate estimation (fixed event count) |
| $r_t$ | emergence rate | first derivative of connectivity (formation rate) |
| $\dot{r}_t$ | rate trend | first time-derivative of $r_t$ |
| $\ddot{r}_t$ | rate acceleration | second time-derivative of $r_t$ |
| $\text{Sterile}_t$ | sterility flag | output of decay detection (mechanism step 3) |
| $\Phi_t$ | fertility signal | triplet of derivatives (thesis core) |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| De Bruijn graph formulation (Preprints 2026) | Global states from local overlaps | justifies $\mathcal{A}$ as structural admissibility |
| Sliding-window estimation (Emergent Mind) | Real-time rate approximation | defines $r_t$ over fixed event window |
| Finite differences (Wikipedia) | Discrete derivatives for time series | provides $\dot{r}_t, \ddot{r}_t$ formulas |
| Gaussian process derivatives (Nature Comms) | Inferring dynamics | validates second-derivative significance for trend stability |

### Left qualitative (prose, not faked)

- **Structural admissibility**: Exact embedding condition $\mathcal{A}$ depends on domain-specific topology (e.g., graph vs. simplicial complex).  
- **Emergence/reinforcement distinction**: Requires predefined notion of "cross-connection" (e.g., new edges between previously disconnected components).  
- **Pure reinforcement state**: Qualitative interpretation of $r_t \to 0$ with $E_t \equiv 0$.

### Test (predict / confirm / falsify)

predicts: Sterility flag when $\dot{r}_t < 0$ and $\ddot{r}_t \leq 0$ under $A_t=1$, $r_t>0$  
confirms: Emergence rate recovery after $\ddot{r}_t > 0$ (false sterility avoidance)  
falsifies: Sterility flag followed by spontaneous emergence recovery without new fragment types

### Where it breaks

1. **Topological rigidity**: If admissibility $\mathcal{A}$ isn't binary or depends on global state (violating local checks).  
2. **Event spacing**: Non-uniform arrival times make event-window derivatives inaccurate.  
3. **Link classification**: If $E_t/R_t$ can't be counted unambiguously (e.g., overlapping connections).

### Honest caveat

Second-derivative threshold $\ddot{r}_t \leq 0$ may be too strict for noisy systems; could require $\ddot{r}_t < -\epsilon$.


---

## blend-02 — formalizable: **partial** · confidence 0.75 · $0.0258

**Reference lookups R1 requested:**
- hierarchical control systems theory separation of layers
- gradient-based resource allocation in network science
- boundary crossing detection in dynamical systems

### Formal core

We formalize the blend's core claim: two evaluator functions (redirector and terminal gate) are register-stratified. The redirector operates in the Houtian register (dynamic fragment flux) using local connection-density gradients, while the terminal gate operates at the Houtian→exit boundary using cumulative structure crossings.  

### 1. **Dynamical System with Two Registers**  
- **Houtian register (dynamic)**: Represents the evolving fragment network as a time-varying graph $\mathcal{H}(t) = (V(t), E(t))$.  
  - $V(t)$: Set of fragments at time $t$ (fragments are added/removed).  
  - $E(t)$: Set of relationships (edges) between fragments at $t$.  
- **Exit register (static)**: Cumulative structure $\mathcal{S}(t)$ formed by fragments that crossed the boundary $\mathcal{B}$.  
  - $\mathcal{S}(t) = (V_{\text{exit}}(t), E_{\text{exit}}(t))$, where:  
    - $V_{\text{exit}}(t)$: Fragments that crossed $\mathcal{B}$ by time $t$.  
    - $E_{\text{exit}}(t)$: All edges $(u,v)$ where $u,v \in V_{\text{exit}}(t)$ and $(u,v)$ existed in $\mathcal{H}(\max(t_u, t_v))$ (recorded at later exit time).  

### 2. **Redirector (Embedded in Houtian)**  
Monitors local connection-density stagnation to trigger reallocation.  
- **Local connection density** for fragment $v \in V(t)$:  
  

$$
\rho(v, t) = \frac{|\{ (u,w) \in E(t) : u, w \in \mathcal{N}(v, r, t) \}|}{\binom{|\mathcal{N}(v, r, t)|}{2}}
$$

  
  where $\mathcal{N}(v, r, t)$ is the set of fragments within graph distance $r$ of $v$ at $t$.  
- **Temporal gradient**:  
  

$$
\Delta\rho(v, t) = \rho(v, t) - \rho(v, t - \Delta t)
$$

  
- **Redirector trigger condition**: For a region $\mathcal{R}_i$ (subgraph of $\mathcal{H}(t)$), reallocation fires if:  
  

$$
\frac{1}{|\mathcal{R}_i|} \sum_{v \in \mathcal{R}_i} \Delta\rho(v, t) < \delta \quad (\delta > 0 \text{ threshold})
$$

  
  Resources are reallocated away from $\mathcal{R}_i$ to regions with $\Delta\rho(v, t) \geq \delta$.  

### 3. **Terminal Gate (At Boundary $\mathcal{B}$)**  
Fires when cumulative exited structure $\mathcal{S}(t)$ satisfies a completeness condition.  
- **Boundary-crossing event**: At time $t$, a fragment $v$ exits $\mathcal{H}(t)$ and enters $\mathcal{S}(t)$.  
- **Completeness predicate**:  
  

$$
P(\mathcal{S}(t)) = \begin{cases} 
  1 & \text{if } \exists \text{ connected component in } \mathcal{S}(t) \text{ with size } \geq K \\
  0 & \text{otherwise}
  \end{cases}
$$

  
- **Gate activation**:  
  

$$
\tau = \min \{ t : P(\mathcal{S}(t)) = 1 \}
$$

  
  The gate fires at time $\tau$.  

### 4. **Separation of Registers (Key Reconciliation)**  
- Redirector uses only $\mathcal{H}(t)$ (Houtian register).  
- Terminal gate uses only $\mathcal{S}(t)$ (exit register).  
- No shared state: $\mathcal{H}(t)$ and $\mathcal{S}(t)$ evolve independently.  

### Derivation Justification  
- **Fragment network as dynamic graph**: From P02-002 (self-organization via internal instability) and blend's "fragment flux".  
- **Connection density**: Standard graph metric (GradNet) for local information accumulation; blend's "cross-connection density".  
- **Gradient stagnation trigger**: From P02-002 ("tension accumulates where fragments crystallize, equilibrium cancels tension"); formalized as $\Delta\rho < \delta$.  
- **Cumulative boundary crossing**: From S76946a ("governance ring activates at exit"); modeled as $\mathcal{S}(t)$ with edges preserved at exit time.  
- **Completeness as connectivity**: From blend's "clears completeness threshold"; quantified via component size (network science convention).

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $\mathcal{H}(t)$ | fragment network in Houtian register | blend's "fragment flux"; P02-002 internal dynamics |
| $\rho(v, t)$ | local connection density | blend's "cross-connection density"; GradNet variational optimization |
| $\Delta\rho(v, t)$ | temporal density gradient | P02-002 "tension accumulation/cancellation" |
| $\mathcal{R}_i$ | region for reallocation | blend's "local" scope; hierarchical control layers |
| $\delta$ | stagnation threshold | blend's "stops producing new links"; empirical tuning |
| $\mathcal{S}(t)$ | cumulative exited structure | blend's "crossing structure"; S76946a governance ring |
| $P(\mathcal{S}(t))$ | completeness condition | blend's "completeness threshold"; graph connectivity heuristic |
| $\tau$ | gate firing time | blend's "terminal gate fires" |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| GradNet (arXiv:2603.09197v1) | gradient-based network optimization | formalizes connection density $\rho$ as differentiable objective for resource allocation |
| Hierarchical Control (IIASA) | separation of control layers | justifies redirector (optimization) and gate (boundary) as distinct strata |
| Boundary Crossing (ScienceDirect) | detecting transitions | models $\mathcal{S}(t)$ as record of crossings at $\mathcal{B}$ |
| P02-002 | internal evaluator | warrants embedded redirector via $\Delta\rho$ in $\mathcal{H}(t)$ |
| S76946a-001 | structural separation | warrants terminal gate using only $\mathcal{S}(t)$ |

### Left qualitative (prose, not faked)

We leave qualitative:  
- **Resource reallocation mechanics**: Blend does not specify how resources move (e.g., fragment migration vs. attention weighting).  
- **Region partitioning**: Blend implies "local" regions but does not define topology (geometric vs. graph-theoretic).  
- **Edge formation rules**: Blend's "fragment relationship network" lacks dynamics for edge creation/deletion.  
- **Exit triggering**: Blend states fragments cross $\mathcal{B}$ but does not specify what prompts exit (e.g., age, stability).

### Test (predict / confirm / falsify)

predicts: Redirector reallocates resources when regional $\Delta\rho < \delta$; gate fires when $\mathcal{S}(t)$ has a component of size $K$.  
confirms: Simulation shows resource shift from stagnant regions; gate fires only after exited fragments form a large connected component.  
falsifies: Gate fires when $\mathcal{S}(t)$ is disconnected (premature) or redirector fails to reallocate despite stagnation (resource deadlock).

### Where it breaks

- **Graph distance metric**: If fragments lack spatial embedding, $\mathcal{N}(v, r, t)$ is undefined.  
- **Stagnation threshold $\delta$**: Arbitrary; system fails if $\delta$ is miscalibrated to noise.  
- **Component size $K$**: If $\mathcal{S}(t)$ completeness requires semantic coherence (not just connectivity), formalization breaks.  
- **Exit recording**: Assumes edges are preserved at exit time; fails if edges decay post-exit.

### Honest caveat

Completeness predicate $P$ is purely topological; blend may imply semantic coherence unmodeled here.


---

## blend-03 — formalizable: **partial** · confidence 0.85 · $0.0165

**Reference lookups R1 requested:**
- dynamic time warping similarity measure
- survival analysis time-to-event models
- case-based reasoning exemplar retrieval

### Formal core

**Definitions and Structures:**  
1. **Trajectory Representation**:  
   Let $\mathcal{T}$ denote a trajectory, represented as a sequence of state vectors:  
   $$
   \mathcal{T}_{1:k} = (\mathbf{s}_1, \mathbf{s}_2, \dots, \mathbf{s}_k), \quad \mathbf{s}_t \in \mathbb{R}^d
   $$  
   where $t$ is the timestep (stage), $k$ is the current stage index, and $\mathbf{s}_t$ encodes the assembly state (e.g., component linkages, emergence metrics).  

2. **Exemplar Bank**:  
   For stage $k$, define two disjoint sets:  
   - $\mathcal{F}_k = \{ \mathcal{T}^{(i)}_{1:k} \mid i \text{ indexes fertile trajectories}\}$  
     (trajectory prefixes ending in high emergence).  
   - $\mathcal{S}_k = \{ \mathcal{T}^{(j)}_{1:k} \mid j \text{ indexes sterile trajectories}\}$  
     (trajectory prefixes ending in stagnation).  

3. **Similarity Measure**:  
   For two equal-length trajectories $\mathcal{T}_A, \mathcal{T}_B$ of stage $k$, compute the **Dynamic Time Warping (DTW)** distance:  
   $$
   \text{DTW}(\mathcal{T}_A, \mathcal{T}_B) = \min_{\pi \in \mathcal{P}} \sum_{(p,q) \in \pi} \| \mathbf{s}_p^A - \mathbf{s}_q^B \|_2
   $$  
   where $\mathcal{P}$ is the set of all warping paths aligning sequences under monotonicity and continuity constraints.  

4. **Fertility Score**:  
   At stage $k$, for a live trajectory $\mathcal{T}_{\text{live}_{1:k}}$, compute:  
   $$
   \text{Score}(k) = \frac{1}{|\mathcal{S}_k|} \sum_{\mathcal{T}_s \in \mathcal{S}_k} \text{DTW}(\mathcal{T}_{\text{live}}, \mathcal{T}_s) - \frac{1}{|\mathcal{F}_k|} \sum_{\mathcal{T}_f \in \mathcal{F}_k} \text{DTW}(\mathcal{T}_{\text{live}}, \mathcal{T}_f)
   $$  
   A high $\text{Score}(k)$ indicates $\mathcal{T}_{\text{live}}$ is closer to fertile exemplars than sterile ones.  

**Derivation Justification**:  
- **Trajectory indexing by stage** derives from the blend's "stage-indexed trajectory exemplars". This allows amortized retrieval without content specificity.  
- **Fertility/sterility partitioning** follows the blend's "compiled fertile-trajectory exemplars" and "shapes that turned out sterile".  
- **DTW as the distance metric** is chosen because the blend requires "similarity" for "trajectory shapes" (e.g., rising emergence rates, pivots), and DTW handles variable-speed patterns in temporal sequences (see CITATIONS).  
- **Score formulation** implements "scoring live trajectories by similarity to fertile vs. sterile exemplars". The difference in average distances quantifies relative alignment to fertile priors.

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $\mathcal{T}_{1:k}$ | trajectory prefix up to stage $k$ | core object being compared; represents assembly state evolution. |
| $\mathcal{F}_k$ | fertile exemplars at stage $k$ | compiled from past high-emergence trajectories. |
| $\mathcal{S}_k$ | sterile exemplars at stage $k$ | compiled from past stagnant trajectories. |
| DTW | dynamic time warping distance | measures shape similarity for state sequences under temporal variation. |
| $\text{Score}(k)$ | fertility score at stage $k$ | quantifies trajectory's match to fertile priors vs. sterile traps. |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| Wikipedia/arXiv: Dynamic Time Warping | Measures similarity between temporal sequences with speed variations; applies because assembly trajectories may have non-uniform stage progression. Enables shape-based comparison (e.g., "rising emergence-link rate"). |  |
| Survival analysis literature | Models time-to-event (e.g., stagnation); applies because sterile trajectories represent "failure events". Helps formalize $\mathcal{S}_k$ as censored data for risk prediction. |  |
| Case-Based Reasoning (CBR) papers | Retrieves exemplars for new problems; applies because the bank uses past trajectories as cases. Justifies $\mathcal{F}_k$/$mathcal{S}_k$ indexing and similarity-based scoring. |  |

### Left qualitative (prose, not faked)

- **State vector $\mathbf{s}_t$ contents**: Left qualitative because the blend does not specify assembly-state features (e.g., "emergence-link rate" is domain-dependent).  
- **Fertility/sterility labeling**: Left qualitative as "design-time decisions" (e.g., human judgment or emergence thresholds), per the blend's "slow design pass extracts... trajectories that turned out fertile/sterile".  
- **Exemplar compilation**: Left procedural (e.g., how trajectories are stored/retrieved) as it relies on implementation-specific engineering.

### Test (predict / confirm / falsify)

predicts: Trajectories with $\text{Score}(k) > \theta$ (threshold) at stage $k$ will achieve higher emergence.  
confirms: Positive correlation between $\text{Score}(k)$ and emergence metrics (e.g., novelty rate) in new runs.  
falsifies: $\text{Score}(k)$ shows no significant correlation ($p>0.05$) with outcomes, or sterile trajectories consistently score higher than fertile ones.

### Where it breaks

1. If state vectors $\mathbf{s}_t$ lack shape-discriminative features, DTW fails to capture fertility signals.  
2. If exemplar banks ($\mathcal{F}_k$, $\mathcal{S}_k$) are non-representative, scoring inherits bias.  
3. If assembly dynamics are non-ergodic, past trajectories poorly predict open-domain futures.

### Honest caveat

The DTW distance assumes trajectories are comparable as sequences; it may break if assembly states have heterogeneous structures.


---

## blend-04 — formalizable: **partial** · confidence 0.75 · $0.0219

**Reference lookups R1 requested:**
- Lyapunov function for consensus algorithms
- structural emergence percolation theory
- bridge detection graph theory

### Formal core

Consider a multi-agent system with $n$ agents. Let $x_i(t) \in \mathbb{R}^d$ be agent $i$'s state at time $t$, and $\mathbf{X}(t) = [x_1(t); \dots; x_n(t)] \in \mathbb{R}^{nd}$ the stacked state vector. The communication graph $G(t) = (V, E(t))$ has vertices $V = \{1, \dots, n\}$ and edges $E(t)$. Define:

1. **Divergence measure** (Monitor A):  
   Disagreement at $t$ is quantified by the quadratic Lyapunov function for consensus:  
   

$$
d(t) = \frac{1}{2} \sum_{(i,j) \in E(t)} w_{ij}(t) \|x_i(t) - x_j(t)\|^2
$$

  
   where $w_{ij}(t) \geq 0$ are edge weights. The residual change is:  
   

$$
\Delta d(t) = d(t) - d(t-1).
$$

  
   Divergence behavior is categorized with threshold $\epsilon > 0$:  
   - Contracting: $\Delta d(t) < -\epsilon$  
   - Stable: $|\Delta d(t)| \leq \epsilon$  
   - Expanding: $\Delta d(t) > \epsilon$.

2. **Shape indicator** (Monitor B):  
   A "structurally integrated shape" exists iff $\exists$ a connected component $C \subseteq V$ in $G(t)$ such that:  
   - Size: $|C| \geq f n$ (fraction $f \in (0,1]$),  
   - Bridge: $\exists$ edge $e \in E(t) \cap (C \times C)$ that is a bridge in the subgraph induced by $C$.  
   The binary shape signal is:  
   

$$
s(t) = \begin{cases} 
      1 & \text{if conditions satisfied} \\
      0 & \text{otherwise}
   \end{cases}
$$



3. **Decision rule** (Truth table):  
   Actions at $t$ are:  
   

$$
\begin{cases}
      \text{close} & \text{if } \Delta d(t) < -\epsilon \quad \land \quad s(t) = 1 \\
      \text{alarm} & \text{if } \Delta d(t) < -\epsilon \quad \land \quad s(t) = 0 \\
      \text{reallocate} & \text{if } |\Delta d(t)| \leq \epsilon \quad \land \quad s(t) = 1 \\
      \text{hold} & \text{otherwise}
   \end{cases}
$$

  
   - **close**: Terminate; output consensus state.  
   - **alarm**: Protect tension; re-inject divergence (e.g., add noise to $\mathbf{X}(t)$).  
   - **reallocate**: Shift resources to agents in $C$ (e.g., exploration budget).  
   - **hold**: Persist with current dynamics.

**Derivation**:  
- $d(t)$ formalizes "inter-angle divergence" from Gossip, using the Lyapunov function for consensus systems. The quadratic form is standard for residual disagreement and decreases under averaging dynamics.  
- $s(t)$ captures "load-bearing shape" via percolation (giant component) and bridge detection from graph theory, aligning with Nomad's shape-gate.  
- The truth table implements the blend's core claim: contracting divergence alone is ambiguous; pairing it with $s(t)$ disambiguates convergence from premature collapse.

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| n | number of agents | fixed system size from consensus setup |
| d | state dimension | agent state space |
| x_i(t) | state of agent i at time t | fundamental agent variable |
| \mathbf{X}(t) | stacked state vector | global state representation |
| G(t) | communication graph at t | encodes agent interactions |
| d(t) | divergence measure | Lyapunov residual for disagreement (Gossip) |
| \Delta d(t) | change in divergence | detects contraction/expansion dynamics |
| \epsilon | divergence threshold | numerical tolerance for trend detection |
| w_{ij}(t) | edge weights | model communication strength (Gossip) |
| C | connected component | candidate emergent structure (Nomad) |
| f | size threshold fraction | percolation parameter for "giant" component |
| s(t) | shape indicator | 1 if load-bearing shape exists (Nomad/graph theory) |
| close, alarm, reallocate, hold | actions | blend's truth-table outputs |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| Zeng et al. (2017) | Lyapunov functions stabilize consensus under quantization | justifies d(t) as residual metric for Monitor A (divergence) |
| MIT Lyapunov note (JNT) | quadratic Lyapunov functions for stochastic matrices | supports d(t) for averaging dynamics (Gossip basis) |
| Wikipedia "Bridge (graph theory)" | bridge = edge whose removal disconnects component | defines load-bearing link for s(t) |
| Percolation theory references | giant component emergence | grounds |C| ≥ f n for "integrated shape" in Monitor B |

### Left qualitative (prose, not faked)

- **Re-injection mechanism**: How "alarm" adds noise/divergence (e.g., distribution or amplitude) is unspecified.  
- **Resource reallocation**: How "reallocate" shifts computational budgets between agents/angles is qualitative.  
- **Parameters**: Thresholds $\epsilon$, $f$, and edge weights $w_{ij}(t)$ require empirical calibration.  
- **Graph dynamics**: Evolution of $G(t)$ (e.g., via agent interactions) is context-dependent.  
Reason: Blend describes high-level actions but not implementation details.

### Test (predict / confirm / falsify)

predicts:  
  - Closure occurs iff both contracting divergence ($\Delta d < -\epsilon$) AND shape present ($s(t)=1$).  
  - Alarm triggers iff divergence contracts without shape ($\Delta d < -\epsilon \land s(t)=0$).  
confirms:  
  In simulation, if agents converge (d(t) → 0) when a giant component with a bridge exists, system closes; if convergence starts without such shape, alarm fires and disrupts convergence (e.g., via added noise).  
falsifies:  
  Closure without both conditions (e.g., closes when $s(t)=0$ or $\Delta d \geq -\epsilon$), or failure to close when both hold. Alarm not triggering during contraction without shape.

### Where it breaks

1. **Non-quadratic dynamics**: If consensus lacks a quadratic Lyapunov function (per MIT note), d(t) may not contract monotonically.  
2. **No giant component**: If $|C| < f n$ always, s(t) never activates, blocking closure.  
3. **Bridge-free shapes**: If emergent structures are 2-edge-connected (no bridges), s(t)=0 even if "shape exists".  
4. **Static graphs**: If $G(t)$ is static and disconnected, s(t) may not reflect emergent integration.

### Honest caveat

Shape criterion $s(t)$: "load-bearing link" as a bridge in a giant component is an interpretation; blend may intend a different structural property.
