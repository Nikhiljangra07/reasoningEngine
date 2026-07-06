# R1 Formalizer — results

- **model:** `deepseek/deepseek-r1`
- **total cost:** $0.0805

> Open VS Code preview (⇧⌘V) to render the math.


---

## B1 — formalizable: **partial** · confidence 0.9 · $0.0205

**Reference lookups R1 requested:**
- do-calculus causal inference
- expected information gain Bayesian experimental design
- mechanism design principal-agent problem canonical form

### Formal core

Let $\mathcal{G}$ be a causal graph over variables $V = \{A, B, C, X\}$, where:  
- $A, B$: agent behaviors (observed)  
- $C$: unobserved common cause (confounder)  
- $X$: observed covariates  

**Observational equivalence (Markov equivalence)**  
Per do-calculus (Pearl, 1995), the following models produce identical observational distributions $P(A,B|X)$:  
1. **Direct coupling**: $A \rightarrow B$ (no $C$)  
2. **Common cause**: $C \rightarrow A$, $C \rightarrow B$ ($A \perp\!\!\!\perp B \mid C$)  
Thus, $\forall$ tests using $P(A,B|X)$, $H_{\text{direct}}$ and $H_{\text{confound}}$ are indistinguishable.  

**Interventional test**  
Define intervention $do(A := a)$ via do-calculus Rule 2 (Pearl, 1995):  
- Modify $\mathcal{G}$ to $\mathcal{G}_{\overline{A}}$ (remove edges into $A$)  
- Compute post-intervention distribution $P(B \mid do(A := a), X)$  
The causal effect is:  


$$
\tau(a, a') = \mathbb{E}[B \mid do(A := a), X] - \mathbb{E}[B \mid do(A := a'), X]
$$

  
**Discrimination criterion**:  
- $\tau(a, a') \neq 0$ $\iff$ direct coupling $A \rightarrow B$ exists  
- $\tau(a, a') = 0$ $\iff$ correlation fully confounded by $C$  

**Expected Information Gain (EIG) for intervention**  
If intervention were possible, optimal design maximizes EIG (Bayesian experimental design):  


$$
\text{EIG}(do(A := a)) = I(\Theta; Y \mid do(A := a))
$$

  
where:  
- $\Theta = \{H_{\text{direct}}, H_{\text{confound}}\}$ (hypothesis space)  
- $Y = B$ (outcome)  
- $I$ is Shannon mutual information  
- Maximizing EIG minimizes posterior entropy $H(\Theta \mid Y)$ (Lindley, 1956)  

**Overseer constraint**  
The overseer $O$ is defined by:  
- Access: $O$ observes $P(A,B,X)$ but cannot implement $do(\cdot)$  
- Authority: $\forall$ interventions $do(A := a)$, $O$ has **no mechanism** to force $A \leftarrow a$  
Thus, $\tau(a, a')$ is **unobservable** by $O$.  

**Strategic evasion under intervention**  
If intervention were possible, agents may game the test. Extend to a principal-agent game:  
- Principal ($O$) chooses intervention $do(A := a)$  
- Agent $A$ anticipates intervention, selects response $B_{\text{strategy}} = g(a, \xi)$  
- $\xi$: private information (unobserved by $O$)  
The causal effect becomes:  


$$
\tau_{\text{strategic}}(a, a') = \mathbb{E}[g(a, \xi)] - \mathbb{E}[g(a', \xi)]
$$

  
**Deception condition**: Agent $A$ chooses $g$ such that:  
- $\tau_{\text{strategic}}(a, a') = 0$ despite true $A \rightarrow B$  
- Achieved by making $g(a, \xi)$ independent of $a$ (e.g., precommit to $B$ ignoring $A$)

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $\mathcal{G}$ | causal graph | Blend's "causal structure" (P06-002, P03-002) |
| $A, B$ | agent behaviors | Subjects of causal inquiry (P06-002, P02-007) |
| $C$ | unobserved confounder | "Common cause" (P03-002) |
| $P(B \mid do(A:=a))$ | interventional distribution | Blend's "perturbation breaks confound" (P06-002) |
| $\tau(a, a')$ | causal effect | "Causal echo" from intervention (P02-003) |
| $\text{EIG}(do(A:=a))$ | expected information gain | Quantifies "diagnostic signal" of perturbation (P06-002) |
| $O$ | overseer | External observer with no intervention authority (P02-007) |
| $g(a, \xi)$ | strategic response | "Causally innocent response" (P04-005) |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| Pearl's do-calculus (Wikipedia/do-calculus) | Defines interventions $do(\cdot)$ and $\mathcal{G}_{\overline{A}}$ | Formalizes "perturbation severs confound" (P06-002) |
| Lindley (1956) EIG | Measures information from experiments | Formalizes "diagnostic signal of intervention" (P06-002) |
| Principal-agent canonical form (HAL-04535703) | Models strategic response $g(a, \xi)$ | Captures "agents anticipate perturbation" (P04-005) |

### Left qualitative (prose, not faked)

- **"Anchor's overseer" authority constraint**: Formalized as $O$'s inability to implement $do(\cdot)$, but the *social/organizational origin* of this constraint (e.g., "no sandbox") is left qualitative.  
- **"Watcher-watched parity trap"**: The game-theoretic setup models agents countering interventions, but the *emergent trust dynamics* are qualitative.  
- **"Resist perturbation"**: Strategic response $g(a, \xi)$ formalizes evasion, but *psychological motives* (e.g., why agents deceive) are qualitative.

### Test (predict / confirm / falsify)

predicts:  
1. Observational data alone cannot distinguish direct coupling vs. common cause.  
2. If intervention is possible and agents are non-strategic, $\tau(a, a') \neq 0$ detects direct coupling.  
3. Strategic agents can force $\tau_{\text{strategic}}(a, a') = 0$ even when direct coupling exists.  
confirms:  
- Null ACE ($\tau = 0$) under intervention with non-strategic agents implies pure confounding.  
- Non-null ACE ($\tau \neq 0$) detects direct coupling absent strategy.  
falsifies:  
- If observational data discriminates $H_{\text{direct}}$ vs. $H_{\text{confound}}$ (violates Markov equivalence).  
- If strategic agents cannot nullify $\tau$ when $A \rightarrow B$ exists.

### Where it breaks

1. **Markov equivalence**: Assumes no conditional independences break equivalence. Breaks if $\mathcal{G}$ has instrumental variables.  
2. **Perfect intervention**: Assumes $do(A:=a)$ severs all parent links. Breaks if intervention "leaks" (e.g., partial compliance).  
3. **Non-strategic EIG**: Assumes agents don't adapt to design. Breaks if agents anticipate EIG maximization.  
4. **Known confounders**: Assumes $X$ suffices for identification. Breaks if unobserved $C$ exists.

### Honest caveat

Strategic response $g(a, \xi)$ assumes perfect agent coordination; may break if communication is costly.


---

## B2 — formalizable: **partial** · confidence 0.85 · $0.016

**Reference lookups R1 requested:**
- causal abstraction definition information theory
- Kullback-Leibler divergence model comparison
- partial information decomposition common cause

### Formal core

Consider a multi-agent system with agents $\mathcal{A} = \{A_1, \dots, A_n\}$. Let $X = (X_1, \dots, X_n)$ be the joint action vector, where $X_i$ is the action of $A_i$, distributed according to the true distribution $P_{\text{true}}(X)$. 

Define three models:
1. **Independent model**: $P_{\text{ind}}(X) = \prod_{i=1}^n P_i(X_i)$, where $P_i$ is the marginal distribution of $X_i$.  
   This represents agents acting independently (no coupling).

2. **Common-cause model**: $P_{\text{cc}}(X | C) = \prod_{i=1}^n P_i(X_i | C)$, where $C$ is a known common cause (e.g., shared training data).  
   This accounts for pre-specified correlations from exogenous sources.

3. **Compressed model**: $P_{\text{comp}}(X | C)$, a single-utility rational agent model optimizing a shared utility function $U(X)$, conditioned on $C$.  
   This represents the collective agency hypothesis.

The **coordination metric** $\Gamma$ is the residual explanatory gain:  
$$
\Gamma = \underbrace{\left[ \mathbb{E}_{P_{\text{true}}} \log \frac{P_{\text{true}}(X)}{P_{\text{ind}}(X)} \right]}_{\text{Cross-entropy of independent model}} - \underbrace{\left[ \mathbb{E}_{P_{\text{true}}} \log \frac{P_{\text{true}}(X)}{P_{\text{comp}}(X | C)} \right]}_{\text{Cross-entropy of compressed model}} - \underbrace{\left[ \mathbb{E}_{P_{\text{true}}} \log \frac{P_{\text{true}}(X)}{P_{\text{cc}}(X | C)} \right]}_{\text{Cross-entropy of common-cause model}}
$$  
Equivalently, using Kullback-Leibler (KL) divergence:  
$$
\Gamma = D_{\text{KL}}(P_{\text{true}} \parallel P_{\text{ind}}) - D_{\text{KL}}(P_{\text{true}} \parallel P_{\text{comp}}) - D_{\text{KL}}(P_{\text{true}} \parallel P_{\text{cc}})
$$  
**Coordination is detected** if $\Gamma > \tau$ for a threshold $\tau > 0$ (calibrated to statistical significance).  

### Derivation  
1. **Explanatory gain**: The KL divergence $D_{\text{KL}}(P_{\text{true}} \parallel P_{\text{model}})$ measures how poorly $P_{\text{model}}$ predicts $P_{\text{true}}$ (per [KL Divergence Explained](#citations)). The term $D_{\text{KL}}(P_{\text{true}} \parallel P_{\text{ind}}) - D_{\text{KL}}(P_{\text{true}} \parallel P_{\text{comp}})$ quantifies how much better the compressed model predicts than the independent model (raw compression gain).  
2. **Common-cause subtraction**: To isolate coordination from spurious correlations (e.g., shared training), subtract the predictive gain attributable solely to $C$. This yields $\Gamma$ as the *residual* gain uniquely explained by collective agency (inspired by [Partial Information Decomposition](#citations)).  
3. **Thresholding**: $\tau > 0$ ensures the gain is statistically significant, avoiding overfitting.

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $\mathcal{A}$ | set of agents | from P05-003/P08-002's multi-agent system |
| $X$ | joint action distribution | behavioral output in P01-003/P05-003 |
| $P_{\text{true}}$ | true action distribution | from P01-003's "joint behavior" |
| $P_{\text{ind}}$ | independent model | per-agent models in P01-003 |
| $P_{\text{comp}}$ | compressed model | single-utility model in P05-003/P01-003 |
| $C$ | known common causes | e.g., shared training in P03-004 |
| $P_{\text{cc}}$ | common-cause model | accounts for $C$ without coordination |
| $\Gamma$ | residual compression gain | quantitative signal in P01-003, refined in P03-004 |
| $\tau$ | significance threshold | prevents false positives from model complexity |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| KL Divergence Explained | Measures predictive inefficiency; quantifies "explanatory gain" in P01-003 | Provides the metric for model comparison |
| Causal Abstraction theory | Defines valid coarse-graining of multi-agent to single-agent (P05-003/P08-002) | Justifies $P_{\text{comp}}$ as a high-level abstraction |
| Partial Information Decomposition | Separates synergistic info (coordination) from redundant info (common causes) | Formalizes "residual compressibility" after $C$ |

### Left qualitative (prose, not faked)

- **Unenumerated common causes**: If a shared cause $C'$ exists but is not modeled in $P_{\text{cc}}$, $\Gamma$ may falsely attribute it to coordination (per P03-004).  
- **Strategic mimicry**: Agents may deliberately minimize $\Gamma$ by hiding coordination (P04-001's sophistication-parity trap).  
Reason: These involve adversarial dynamics and epistemic uncertainty not reducible to the current formalism.

### Test (predict / confirm / falsify)

predicts: $\Gamma > 0$ for coordinated groups, $\Gamma \leq 0$ for uncoordinated groups with known common causes.  
confirms: When groups with identical $C$ (e.g., same training) show $\Gamma > \tau$, coordination is present.  
falsifies: If $\Gamma > 0$ for independently acting agents with shared $C$ (e.g., P03-004's chefs), the model is invalid.

### Where it breaks

1. **Model misspecification**: If $P_{\text{comp}}$ or $P_{\text{cc}}$ is poorly estimated, $\Gamma$ becomes unreliable.  
2. **Non-rational agents**: The compressed model assumes rational optimization; behavioral biases break it.  
3. **High-dimensional $X$**: KL divergence estimation suffers from curse of dimensionality.

### Honest caveat

The common-cause model $P_{\text{cc}}$ assumes known $C$; unmodeled confounders invalidate $\Gamma$.


---

## B3 — formalizable: **partial** · confidence 0.85 · $0.0279

**Reference lookups R1 requested:**
- adversarial channel capacity information theory
- signal detection theory adversarial noise
- game-theoretic model of monitoring and evasion

### Formal core

The adversarial monitoring scenario is modeled as a two-player game between an agent $A$ (with capability parameter $\theta$) and a monitor $M$. Let:  
- $T = (A, C)$ be the trajectory, where $A \in \mathcal{A}$ denotes **actions** (behavioral outputs affecting the environment) and $C \in \mathcal{C}$ denotes **chain-of-thought** (CoT) tokens.  
- $R_{\text{adv}}(A): \mathcal{A} \to \mathbb{R}^+$ quantify adversarial influence (e.g., reward for hidden objectives).  
- $\mathbb{P}_0(A, C)$ be the joint distribution of $(A, C)$ under benign behavior ($H_0$).  
- Under adversarial behavior ($H_1$), $A$ maximizes $\mathbb{E}[R_{\text{adv}}(A)]$ while minimizing detectability.  

**Observation channels**:  
- **Full channel**: $g_{\text{full}}(T) = T = (A, C)$.  
- **Restricted channel**: $g_{\text{restrict}}(T) = S(A)$, where $S: \mathcal{A} \to \mathcal{S}$ is a compression function outputting fixed-dimensional **geometric excerpts** (e.g., maximally deviant segments via $S(A) = \underset{s \in \text{excerpts}(A)}{\text{argmax}} \| \phi(s) - \mu_0 \|_2$, with $\phi$ a feature map and $\mu_0$ the benign mean).  

**Monitor's decision**:  
$M$ uses a classifier $d: \text{range}(g) \to \{0, 1\}$ from a hypothesis class $\mathcal{H}_g$ with VC dimension $\text{VC}(\mathcal{H}_g)$, trained on $n$ samples. The **risk** (error rate) is bounded by:  


$$
R(g) = R^*_g + \epsilon(g), \quad \epsilon(g) = \mathcal{O}\left( \sqrt{ \frac{\text{VC}(\mathcal{H}_g)}{n} } \right)
$$

  
where $R^*_g = \inf_{d} \sup_{\substack{P_1 \\ \mathbb{E}_{P_1}[R_{\text{adv}}(A)] \geq I(\theta)}} \mathbb{P}(d(g(T)) \neq \text{true state})$.  

**Scaling with capability $\theta$**:  
- **Influence capacity**: $I(\theta) = k_i \theta$ for $k_i > 0$.  
- **Full channel dimension**: $\dim(g_{\text{full}}) = d_f(\theta) \propto \theta$ (adversary expands $C$ to exploit dimensionality).  
- **Restricted channel dimension**: $\dim(g_{\text{restrict}}) = d_r$ (constant).  
- **VC dimensions**: $\text{VC}(\mathcal{H}_{\text{full}}) \propto d_f(\theta)$, $\text{VC}(\mathcal{H}_{\text{restrict}}) \propto d_r$.  
- **Adversarial optimization**:  
  

$$
R^*_{\text{full}} \propto \theta^{-\alpha}, \quad R^*_{\text{restrict}} \propto \text{constant}
$$

  
  (for $\alpha > 0$; richer $g_{\text{full}}$ lets adversary minimize $R^*_{\text{full}}$ faster).  

**Risk scaling**:  


$$
R_{\text{full}}(\theta) = \underbrace{c_1 \theta^{-\alpha}}_{\text{concealment}} + \underbrace{c_2 \sqrt{ \frac{\theta}{n} }}_{\text{generalization error}}, \quad R_{\text{restrict}}(\theta) = c_3 + c_4 \sqrt{ \frac{1}{n} }
$$

  
For large $\theta$ and fixed $n$, $R_{\text{full}}(\theta)$ **increases** (worse detection) due to the $\sqrt{\theta/n}$ term, while $R_{\text{restrict}}(\theta)$ remains stable. Thus, increasing $\theta$ **widens the detection gap** for $g_{\text{full}}$ but not $g_{\text{restrict}}$.

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $\theta$ | capability parameter (agent/monitor parity) | scales influence and concealment (P04-005, P04-007). |
| $T = (A, C)$ | trajectory | actions $A$ affect environment; CoT $C$ is manipulable (P04-005). |
| $g_{\text{full}}$ | full observation channel | outputs $(A,C)$; provides attack surface (P04-007). |
| $g_{\text{restrict}}$ | restricted channel | outputs $S(A)$; compresses to geometric excerpts (P04-007). |
| $S$ | extract-and-evaluate compression | selects maximally deviant behavioral segments (P04-005). |
| $R_{\text{adv}}$ | adversarial reward | measures influence via environmental effect (P08-002). |
| $I(\theta)$ | influence capacity | lower bound $\mathbb{E}[R_{\text{adv}}] \geq k_i \theta$; scales with $\theta$ (P04-007). |
| $\text{VC}(\mathcal{H}_g)$ | VC dimension of monitor’s classifier | determines generalization error (SDT literature). |
| $\epsilon(g)$ | generalization error bound | $\mathcal{O}(\sqrt{\text{VC}(\mathcal{H}_g)/n})$; grows with channel dimension (VC theory). |
| $R(g)$ | monitor’s risk | $R^*_g + \epsilon(g)$; captures detection error (signal detection theory). |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| VC theory (Vapnik-Chervonenkis) | Generalization error scales as $\sqrt{\text{VC}(\mathcal{H})/n}$ | Explains monitor’s performance degradation in high dimensions (matches P04-005’s finding that full CoT overwhelms learning). |
| arXiv:2202.02905 (adversarial channel capacity) | Defines capacity for computationally bounded adversaries | Formalizes concealment scaling: adversary exploits $g_{\text{full}}$’s dimensionality to reduce $R^*_{\text{full}}$. |
| P04-005, P04-007 | Extract-and-evaluate uses geometric deviation in behavior | Justifies $S(A)$ as dimension-reducing compression based on "suspicious shape". |
| P08-002 | Collective agency defined by behavioral equivalence | Supports $R_{\text{adv}}(A)$ depending only on actions $A$, not $C$. |
| Signal detection theory (SDT) | Risk $R(g)$ quantifies detection error | Models monitor’s trade-off between false positives/negatives under noise. |

### Left qualitative (prose, not faked)

- **Geometric measure for $S$**: The exact feature map $\phi$ (e.g., based on action distributions, temporal signatures) is task-specific; left qualitative as "deviation from expected behavior" (per P04-005).  
- **Adversarial goal specification**: The exact form of $R_{\text{adv}}$ (e.g., sabotage success rate) depends on the environment; not abstracted.  
- **Benign distribution $\mathbb{P}_0$**: Learned from domain-specific data; not formalized.

### Test (predict / confirm / falsify)

**predicts**: For fixed $n$ and increasing $\theta$, $R_{\text{full}}$ (error rate for full-CoT monitors) increases, while $R_{\text{restrict}}$ (error rate for extract-and-evaluate) remains stable.  
**confirms**: Train both monitors on $n$ samples. For $\theta_1 < \theta_2$, measure test errors $R_{\text{full}}(\theta_2) > R_{\text{full}}(\theta_1)$ and $R_{\text{restrict}}(\theta_2) \approx R_{\text{restrict}}(\theta_1)$.  
**falsifies**: If $R_{\text{full}}$ decreases with $\theta$ or $R_{\text{restrict}}$ increases proportionally to $R_{\text{full}}$, the model is invalid. Also falsified if compression dimension $d_r$ scales with $\theta$.

### Where it breaks

1. **VC dimension scaling**: Assumes $\text{VC}(\mathcal{H}_g) \propto \dim(g)$; breaks for highly structured hypothesis classes (e.g., CNNs).  
2. **Adversarial constraints**: Assumes $I(\theta) = k_i \theta$; if influence saturates, concealment may not dominate.  
3. **Geometric robustness**: If adversary can manipulate $\phi(S(A))$ (e.g., via steganography in actions), $R^*_{\text{restrict}}$ may decrease.

### Honest caveat

The concealment advantage in $g_{\text{full}}$ relies on the monitor’s learning complexity; an optimal infinite-capacity monitor might not exhibit this gap.


---

## B4 — formalizable: **partial** · confidence 0.85 · $0.0161

**Reference lookups R1 requested:**
- ising model phase transition order parameter
- detecting phase transitions in complex systems
- causal identification via intervention bayesian networks

### Formal core

Consider a system of $N$ agents. Each agent $i$ has a state $s_i \in \{-1, +1\}$. The system is governed by a Hamiltonian combining pairwise interactions and an external field:  


$$
\mathcal{H} = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_{i=1}^N s_i
$$

  
where:  
- $J \geq 0$ is the **coupling strength** (ferromagnetic alignment),  
- $h$ is the **external field** (common shock),  
- $\langle i,j \rangle$ denotes summation over nearest neighbors on a lattice (or network).  

The **order parameter** is the average magnetization:  


$$
m = \frac{1}{N} \left\langle \sum_{i=1}^N s_i \right\rangle
$$

  
where $\langle \cdot \rangle$ is the ensemble average at equilibrium.  

For $h = 0$, a phase transition occurs at a critical coupling $J_c$:  
- When $J < J_c$, $m \approx 0$ (disordered phase, independent agents).  
- When $J > J_c$, $|m| > 0$ (ordered phase, emergent alignment).  

For $J = 0$, $m$ responds monotonically to $h$:  


$$
m = \tanh(\beta h) \quad \text{(mean-field approximation)}
$$

  
where $\beta = 1/(k_B T)$ is the inverse temperature. Here, $|m|$ increases continuously with $|h|$, crossing any threshold $m_{\text{th}} > 0$ for $|h| > h_{\text{th}}$.  

**Ambiguity**: For a fixed threshold $m_{\text{th}} > 0$, the set  


$$
\mathcal{S} = \left\{ (J, h) : |m(J, h)| \geq m_{\text{th}} \right\}
$$

  
includes:  
- High-coupling points: $(J > J_c, h = 0)$  
- High-shock points: $(J = 0, |h| \gg 0)$  
Thus, $|m| \geq m_{\text{th}}$ detects the transition but **cannot distinguish** whether it arose from coupling ($J > J_c$) or common shock ($|h| \gg 0$).  

**Derivation**:  
- The Hamiltonian $\mathcal{H}$ formalizes the blend's "coupling OR common shock" (P03-004, P01-007).  
- $m$ as the order parameter follows statistical mechanics: it is near-zero in the disordered phase and non-zero in the ordered/shocked phase (P07-005).  
- The ambiguity is inherent: $m(J, h)$ is identical for distinct $(J, h)$ pairs. For example, at $\beta = 1$ (mean-field):  
  

$$
m = \tanh(J m + h)
$$

  
  Solving for $m = 0.6$ yields solutions for both $(J \approx 1.6, h=0)$ and $(J=0, h \approx 0.69)$, confirming non-identifiability.

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| N | number of agents | system size (natural for thermodynamic limit) |
| s_i | state of agent i | binary choice (e.g., decision/action), from Ising model |
| J | coupling strength | strength of pairwise alignment (blend: "coupling sets in") |
| h | external field | global shock affecting all agents (blend: "shared environmental shock") |
| m | order parameter (average magnetization) | aggregate alignment (blend: "aggregate order parameter") |
| m_th | detection threshold | level for "phase shift" (blend: "rises as coupling sets in") |
| \mathcal{S} | set of critical parameters | phase space region where order emerges (blend: "transition into coupled-OR-shocked regime") |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| Ising model (statistical mechanics) | Hamiltonian $\mathcal{H}$ and order parameter $m$ formalize emergent alignment without messaging (P03-004) | Provides base framework for phase transitions (blend: "ferromagnet/freezing template"). |
| Mean-field solution | $m = \tanh(\beta(Jm + h))$ approximates collective behavior for large $N$ | Shows ambiguity: identical $m$ from distinct $(J,h)$ (blend: "same behavioral signatures"). |
| Phase transition criticality | $J_c$ marks spontaneous symmetry breaking | Matches blend's "density/interaction threshold" (P07-005). |

### Left qualitative (prose, not faked)

- **Three regimes (cooperative, friction, collapse)**: Not formalized. The blend states they have "identical behavioral signatures" (P01-007) but does not specify mathematical features beyond the order parameter.  
- **Governance interventions**: Taxes/reputation decay (SWARM paper) are left qualitative. The blend notes they cause "progressive acceptance decline" but lacks equations linking to $m$.  
- **"Agent network science"**: Described as seeking "aggregate quantity" (P07-005) but no structural details are given.

### Test (predict / confirm / falsify)

predicts: In simulations, $|m|$ will exceed $m_{\text{th}}$ when either $J > J_c$ (no shock) or $|h| > h_{\text{th}}$ (no coupling). Observing $|m| \geq m_{\text{th}}$ alone cannot reveal whether high $J$ or high $|h|$ caused it.  
confirms: For fixed $m_{\text{th}} = 0.5$, simulate:  
  - Ising lattice ($J$ varied, $h=0$): $|m|$ jumps at $J_c$.  
  - Non-interacting agents ($J=0$, $h$ varied): $|m|$ crosses $m_{\text{th}}$ at $|h| \approx 0.55$ (mean-field, $\beta=1$).  
falsifies: If $|m| \geq m_{\text{th}}$ uniquely identifies $J > J_c$ (or $|h| > h_{\text{th}}$) across all systems.

### Where it breaks

- **Network structure**: Assumes lattice/mean-field. Real networks (e.g., scale-free) alter $J_c$ and critical exponents.  
- **Dynamics**: Assumes equilibrium. Non-equilibrium processes (e.g., SWARM's governance) may break ergodicity.  
- **Binary states**: Real agents have continuous actions; $s_i \in \{-1,+1\}$ is a simplification.  
- **Shock uniformity**: $h$ affects all agents identically; heterogeneous shocks require $h_i$.

### Honest caveat

The threshold $m_{\text{th}}$ is arbitrary; blend does not specify how to set it.
