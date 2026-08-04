# T2Sem-MEC v2

Multi-UAV, multi-user, multi-terminal hierarchical semantic MEC with two-timescale DRL.

## Architecture

| Layer | Entities | Default scale |
|-------|----------|---------------|
| Users | Ground users with mobility | 8 |
| Terminals | Heterogeneous devices per user (phone, sensor, …) | 2 per user → 16 terminals |
| UAVs | Mobile edge relays | 3 |
| HAPs | High-altitude edge servers | 2 |

**Short timescale (PPO, shared per-terminal actor):** per-terminal offload target (local / UAV_m / HAP_k), semantic ratio μ, bandwidth & CPU requests with **resource contention** normalization.

**Long timescale (DDPG):** joint control of **M UAV trajectories** + SLA tier sampling weights.

## Quick start

```bash
cd t2sem-mec-v2
pip install -r requirements.txt
python tests/test_sanity.py
python experiments/run_smoke.py
```

## Full training

```bash
python experiments/train.py --method t2sem_drl
python experiments/evaluate.py --method t2sem_drl
python experiments/train.py --method greedy
python experiments/train.py --method tjcct
```

## Ideal track (T2Sem-X prototype)

```bash
python tests/test_ideal_t2semx.py
python scripts/ideal_t2semx_run.py --seeds 42 123 456
python scripts/ideal_stress_suite.py --method ideal_t2sem_x
python scripts/build_ideal_pack.py
```

## Configuration (`configs/default.yaml`)

Key knobs:

- `num_users`, `terminals_per_user`, `terminal_assignment`, `terminal_types`
- `num_uavs`, `num_haps`, `uav.init_positions`, `hap.positions`
- `resource_contention: true` — normalize bw/cpu shares among terminals targeting the same edge node
- `drl.short.algo: shared_ppo` — parameter-shared terminal actor (CTDE-style)

## Project layout

```
t2sem-mec-v2/
  configs/          # default + smoke
  src/env/          # MultiMECEnv, channel, semantic, SLA
  src/agents/       # Shared PPO + multi-UAV DDPG + T2SemDRL
  src/baselines/    # greedy, local, random, TJCCT (K-means multi-UAV)
  experiments/      # train, evaluate, run_smoke
  tests/            # sanity checks
```

## vs v1 (`uav-semantic-mec`)

| Feature | v1 | v2 |
|---------|----|----|
| UAVs | 1 | M (configurable) |
| HAPs | 1 | K (configurable) |
| Terminals | UE only | heterogeneous multi-terminal per user |
| Resource model | independent fractions | contention-aware normalization |
| Short agent | flat MLP | shared terminal actor + global critic |
| Long action | 3 + tiers | 3M + tiers |
| Tier weights | not wired to sampling | wired via `sample_tier(..., tier_weights)` |
