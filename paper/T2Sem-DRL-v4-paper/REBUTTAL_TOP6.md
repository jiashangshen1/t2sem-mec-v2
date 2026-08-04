# Rebuttal Top-6 Points (Ready to Paste)

1. **Paper positioning clarified (interface, not novel RL backbone).**  
   We now frame the contribution as a model-informed feasibility interface plus empirical decomposition. The manuscript explicitly states RL is a secondary refinement layer inside the shared feasible region.

2. **Contribution hierarchy is quantified, not claimed qualitatively.**  
   Ablation evidence shows semantic modeling (+46.0 pp) and feasibility mask (+31.8 pp) dominate; RL adds +6.2 pp over masked greedy under the canonical protocol.

3. **Baseline strength improved with a shared-interface semantic comparator.**  
   We added Semantic MAPPO + F under the same mask/reward interface. On canonical seeds {42,123,456}: 91.43% vs T2Sem-DRL 93.99% (paired +2.56 pp, 95% CI [1.14, 4.41]); on expanded seeds {42,123,456,789,1011}: paired gap -0.25 pp (95% CI [-3.35, 2.84]), indicating no conclusive superiority at that scope.

4. **Energy claims are now conservative and protocol-aligned.**  
   We removed universal energy-dominance language and report operating points (Table I-b), separating SLA/delay head-to-head from deployment energy trade-offs.

5. **Tier trade-off is openly reported.**  
   We explicitly state that mixed-tier optimization can reduce critical-tier SLA while protecting interactive-tier SLA; this is presented as a deployment choice, not hidden as a universal win.

6. **Robustness checks expanded (scale, alpha, deployment transfer).**  
   Supplementary checks now include scale-up topology (deployment triangle closed: greedy 71.03%, semantic 73.72%, T2Sem 80.31%; paired T2Sem–Semantic +6.59 pp, 95% CI [2.71, 11.61]), alpha validation, and a critical-priority deployment triangle. Validation-selected alpha does not guarantee universal test improvement (0.6 vs 0.8 tie on held-out seed), and deployment transfer under critical-priority config remains competitive (greedy 87.77%, semantic 91.41%, T2Sem 93.99%, critical-priority trained 90.47%).

