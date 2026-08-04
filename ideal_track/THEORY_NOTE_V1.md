## T2Sem-X Theory Note v1

Scope: hierarchical semantic offloading CMDP with feasibility projection and
two-timescale control, targeting analyzable guarantees for ideal-track method
innovation.

### 1) Setup and notation

- Slots are indexed by `k`, long frames by `t`.
- Short action chooses offload target, semantic ratio, and resource requests.
- Long action controls UAV displacement and optional tier weighting.
- Tier set is `K={critical, interactive, best_effort}`.
- Each tier `kappa` has SLA requirement `(T_kappa_max, S_kappa_min)`.

Define expected per-tier violation cost:

- `g_kappa(pi) = E_pi[max(0, T - T_kappa_max) + max(0, S_kappa_min - S)]`.

Define constrained objective:

- minimize `J_E(pi)` subject to `g_kappa(pi) <= epsilon_kappa` for all tiers.

### 2) Assumptions

A1. **Lipschitz environment response**  
Expected delay and similarity are locally Lipschitz in the continuous action.

A2. **Bounded stochasticity**  
Task-size jitter and channel randomness are sub-Gaussian with bounded moments.

A3. **Non-empty robust feasible set**  
For each state, at least one action remains feasible after conservative margin.

A4. **Two-timescale separation**  
Long updates are slower than short updates (`beta_t / alpha_t -> 0`), and both
step sizes satisfy Robbins-Monro conditions.

A5. **Bounded dual variables**  
Projected dual updates keep `lambda_kappa in [lambda_min, lambda_max]`.

### 3) Core propositions (draft)

Proposition P1 (Robust feasibility retention).  
Under A1-A3, feasibility projection with delay margin `alpha<1` and similarity
headroom `delta_s>0` yields non-zero probability of SLA satisfaction for each
tier and strictly lower worst-case one-step violation than unprojected actions.

Proof sketch:  
Projection is a contraction onto a closed feasible set under A1. A2 bounds
perturbation around model estimates; margin terms absorb bounded disturbances.
Hence projected action has tighter violation upper envelope.

Proposition P2 (Tier-wise primal-dual boundedness).  
Under A4-A5, projected per-tier dual ascent generates bounded dual iterates and
bounded empirical violation averages.

Proof sketch:  
Projected stochastic approximation on compact interval plus bounded gradients
implies bounded dual sequence. Two-timescale separation allows treating short
policy updates as quasi-stationary for dual dynamics.

Proposition P3 (Suboptimality decomposition).  
Expected performance gap between optimal constrained policy and learned policy
can be decomposed into:

- model bias term (semantic/channel approximation),
- optimization term (policy class + finite updates),
- projection conservatism term (alpha, delta_s).

This explains why stronger projection robustness can reduce action volume while
improving feasibility reliability.

Proof sketch:  
Start from constrained Bellman error decomposition and add projection-induced
distortion term bounded by Lipschitz constants and margin parameters.

### 4) Counterexamples and boundary failures

C1. **Over-conservative mask collapse**  
If `alpha` too small and `delta_s` too large, feasible action volume collapses;
throughput and reward degrade despite low violations.

C2. **Tier conflict under fixed scalar dual**  
Single global lambda can satisfy easy tiers while starving critical tier; this
matches observed critical-vs-interactive imbalance.

C3. **Timescale mismatch instability**  
If long controller updates too aggressively, short policy cannot track induced
distribution shift, causing oscillatory constraints and energy spikes.

C4. **Model misspecification under severe NLoS**  
When expected-channel approximation is badly biased, feasibility projection may
pass actions that fail under realized channel, weakening P1 constants.

### 5) What must be validated experimentally

E1. P1 -> compare violation tails and SLA floors with/without robust projection.

E2. P2 -> track per-tier lambda trajectories and time-averaged violation bounds.

E3. P3 -> sweep margin parameters `(alpha, delta_s)` to estimate conservatism
vs performance tradeoff and fit decomposition trends.

E4. C1-C4 -> dedicated stress tests that intentionally violate assumptions.

### 6) Immediate implications for T2Sem-X design

1. Structured action parameterization should expose feasibility degrees of
freedom explicitly (not only post-hoc clipping).
2. Dual updates should include uncertainty-aware correction to avoid overreacting
to high-variance estimates.
3. Reporting must include theorem-linked diagnostics, not only aggregate SLA.
