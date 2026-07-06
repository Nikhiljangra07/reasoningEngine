# R1 Formalizer — results

- **model:** `deepseek/deepseek-r1`
- **total cost:** $0.0181

> Open VS Code preview (⇧⌘V) to render the math.


---

## blend-02 — formalizable: **partial** · confidence 0.85 · $0.0181

**Reference lookups R1 requested:**
- expected information gain formula Bayesian experimental design
- knowledge space theory formal definition
- active learning acquisition functions

### Formal core

**Formalization of the Blend's Core Mechanism**  

Let $\mathcal{Q}$ be a finite set of skills/concepts (the domain), structured as a prerequisite hierarchy via a partial order $\preceq$ (where $q_i \preceq q_j$ denotes "$q_i$ must be mastered before $q_j$"). A **knowledge state** $K \subseteq \mathcal{Q}$ is a down-set of $\preceq$ (if $q_j \in K$ and $q_i \preceq q_j$, then $q_i \in K$). The **anchor** (user's partial state) is a function $s: \mathcal{Q} \to \{0, 1, \bot\}$:  
- $s(q) = 1$: mastered (known present),  
- $s(q) = 0$: not mastered (known absent),  
- $s(q) = \bot$: gap (unknown).  

Let $\mathcal{G} = \{ q \in \mathcal{Q} \mid s(q) = \bot \}$ be the set of gaps. For a prior distribution $\mathbb{P}$ over possible knowledge states $K$ (consistent with $s$), the **inferential leverage** of probing gap $q \in \mathcal{G}$ is quantified by its *expected information gain (EIG)* about the entire state $K$.  

Given event $E = \{ K : K \text{ agrees with } s \text{ on } \{q : s(q) \neq \bot\} \}$, EIG is the mutual information between observation $y_q$ (outcome of probing $q$) and $K$:  


$$
\mathrm{EIG}(q) = I(K; y_q \mid E) = H(y_q \mid E) - \mathbb{E}_{K \mid E} \left[ H(y_q \mid K) \right].
$$

  
Since $y_q$ is deterministic given $K$ (i.e., $y_q = \mathbf{1}_{\{q \in K\}}$), $H(y_q \mid K) = 0$. Thus:  


$$
\mathrm{EIG}(q) = H(y_q \mid E) = -\left[ p_q \log_2 p_q + (1 - p_q) \log_2 (1 - p_q) \right],
$$

  
where $p_q = \mathbb{P}(q \in K \mid E)$ is the probability $q$ is mastered given the anchor.  

The **probe budget allocation** over gaps $\mathcal{G}$ uses EIG as leverage weights. For a wanderer with budget $B$ probes, the selection probability for $q \in \mathcal{G}$ at each step is:  


$$
P_{\text{select}}(q) = \frac{\mathrm{EIG}(q)}{\sum_{q' \in \mathcal{G}} \mathrm{EIG}(q')}.
$$

  
After probing $q$, update $s$ with $y_q$ and repeat for $B$ steps.  

**Derivation Justification**:  
1. **Prerequisite hierarchy**: From knowledge space theory (Doignon & Falmagne), $\preceq$ induces down-sets as valid states. The blend treats gaps as coordinates in this structure.  
2. **Binary gaps**: The anchor $s$ extends KST's binary mastery to include explicit unknowns ($\bot$), aligning with the blend's "partial binary state."  
3. **EIG as leverage**: The blend defines leverage as "reduction in uncertainty about the whole shape." Mutual information $I(K; y_q \mid E)$ rigorously measures this reduction in Shannon entropy. The simplification $\mathrm{EIG}(q) = H(y_q \mid E)$ holds because $y_q$ is $K$-deterministic.  
4. **Weighted allocation**: The blend mandates weighting by leverage; $P_{\text{select}} \propto \mathrm{EIG}(q)$ operationalizes "spend most budget on highest-yield gaps."

### Objects (each justified)

| Symbol | Maps to (blend element) | Why it belongs |
| --- | --- | --- |
| $\mathcal{Q}$ | skills/concepts in the hierarchy | Source: KST (Doignon & Falmagne); blend requires a structural domain. |
| $\preceq$ | prerequisite order | Source: KST; blend specifies "structural dimensions" with dependencies. |
| $s$ | anchor state (partial) | Blend's "partial binary state"; maps gaps to $\bot$. |
| $\mathcal{G}$ | set of gaps | Blend's "unfilled coordinates" targeted for probes. |
| $\mathbb{P}$ | prior over knowledge states | Needed for Bayesian EIG; blend assumes uncertainty about "whole shape." |
| $\mathrm{EIG}(q)$ | inferential leverage of gap $q$ | Blend's "expected inferential leverage"; formalized via mutual information. |
| $P_{\text{select}}(q)$ | probe allocation weight | Blend's "weight the wandering agents' domain selection by leverage score." |
| $B$ | probe budget | Blend's "finite traversal"; from MaD Physics's measurement constraint. |

### Citations

| Source | Why it applies | How it helps |
| --- | --- | --- |
| Doignon & Falmagne (Knowledge Space Theory) | Defines $\mathcal{Q}$, $\preceq$, and knowledge states. *Applies*: Blend anchors to KST's hierarchy. *Helps*: Provides the "diagnostic address" for gaps. |  |
| Bayesian EIG (Emergent Mind, Go et al. 2022) | Defines $I(K; y_q \mid E)$. *Applies*: Blend requires quantifying "inferential leverage." *Helps*: Formalizes "measurement-leverage weighting" from MaD Physics. |  |
| Shannon entropy | Defines $H(y_q \mid E)$. *Applies*: EIG simplifies to entropy under deterministic $y_q$. *Helps*: Computes leverage scores efficiently. |  |
| Active learning acquisition functions (e.g., BALD) | Use EIG for probe selection. *Applies*: Blend's "wandering agents" align with active learning. *Helps*: Implements "high-leverage moves under a budget" from MaD Physics. |  |

### Left qualitative (prose, not faked)

- "Resonant find": Left qualitative. Refers to a probe result that meaningfully reduces uncertainty, but "resonance" implies a subjective fit to the user's intent not captured by entropy.  
- "User's emerging vision": Left qualitative. Refers to the latent goal behind the partial concept; formalized as $K$ but the "vision" aspect transcends the state vector.  
- "Generative blind spots": Left qualitative. Describes why high-leverage gaps are prioritized, but "generative" is a cognitive claim not reduced to EIG.

### Test (predict / confirm / falsify)

predicts: Probes prioritize gaps with highest $H(y_q \mid E)$ (i.e., $p_q \approx 0.5$, where uncertainty is maximal).  
confirms: Simulate prerequisite hierarchies; measure correlation between $P_{\text{select}}(q)$ and $\mathrm{EIG}(q)$. Efficiency gain: faster reduction in total state entropy $H(K \mid E)$ vs. uniform probing.  
falsifies: If targeting maximal-entropy gaps yields *less* information about $K$ than probing low-entropy gaps or random gaps, the leverage-weighting fails.

### Where it breaks

1. **Prior misspecification**: If $\mathbb{P}$ is inaccurate, $p_q$ and EIG are misleading.  
2. **Non-down-set states**: If knowledge isn't well-modeled by down-sets (e.g., skills are interdependent), KST assumptions break.  
3. **Tractable computation**: $\mathbb{P}(q \in K \mid E)$ requires summing over $\mathcal{K}$; intractable for large $\mathcal{Q}$ without factorization.  
4. **Deterministic probes**: Assumes noiseless $y_q$; noisy observations require full $I(K; y_q \mid E)$ (does not simplify to $H(y_q \mid E)$).

### Honest caveat

EIG reduces to $H(y_q \mid E)$ only if $y_q$ is deterministic given $K$; if probes are noisy (e.g., user errors), this breaks.
