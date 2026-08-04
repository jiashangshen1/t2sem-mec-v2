# T2Sem-MEC v2

Multi-UAV, multi-user, multi-terminal hierarchical semantic MEC with **two-timescale DRL** (short-scale PPO + long-scale DDPG).

**Repository:** https://github.com/jiashangshen1/t2sem-mec-v2

This is the v2 / v4.3 simulator line. Paper drafts, figures, and advisor presentation materials live under `paper/` and `presentations/`.

## Architecture

| Layer | Entities | Default scale |
|-------|----------|---------------|
| Users | Ground users with mobility | 8 |
| Terminals | Heterogeneous devices per user | 2 per user → 16 terminals |
| UAVs | Mobile edge relays | 3 |
| HAPs | High-altitude edge servers | 2 |

- **Short timescale (shared PPO):** per-terminal offload target (local / UAV / HAP), semantic ratio \(\mu\), bandwidth & CPU requests with contention-aware normalization.
- **Long timescale (DDPG):** joint control of multi-UAV trajectories and SLA tier sampling weights.

## Quick start

```bash
pip install -r requirements.txt
python tests/test_sanity.py
python experiments/run_smoke.py
```

## Training and evaluation

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

## Configuration

Primary config: `configs/default.yaml`.

Important knobs:

- `num_users`, `terminals_per_user`, `terminal_types`
- `num_uavs`, `num_haps`, UAV / HAP positions
- `resource_contention: true`
- `drl.short.algo: shared_ppo`

## Layout

```text
configs/          default + smoke (+ paper overlays)
src/env/          MultiMECEnv, channel, semantic, SLA
src/agents/       shared PPO + multi-UAV DDPG + T2SemDRL
src/baselines/    greedy, local, random, TJCCT
experiments/      train / evaluate / smoke
ideal_track/      T2Sem-X prototypes
paper/            T2Sem-DRL paper packages
presentations/    advisor presentation pack
results/          experiment outputs (tracked)
tests/
```

## Compared with v1 (`uav-semantic-mec`)

| Feature | v1 | v2 |
|---------|----|----|
| UAVs | 1 | \(M\) (configurable) |
| HAPs | 1 | \(K\) (configurable) |
| Terminals | UE only | heterogeneous multi-terminal per user |
| Resource model | independent fractions | contention-aware normalization |
| Short agent | flat MLP | shared terminal actor + global critic |
| Long action | 3 + tiers | \(3M\) + tiers |

## Related repositories

| Repo | Role |
|------|------|
| [uav-semantic-mec](https://github.com/jiashangshen1/uav-semantic-mec) | v1 single-UAV prototype |
| [safe-tier-mec](https://github.com/jiashangshen1/safe-tier-mec) | Safety-layer greenfield on this environment |
| [sem-contract-mec](https://github.com/jiashangshen1/sem-contract-mec) | Non-learning contract algorithm line |
