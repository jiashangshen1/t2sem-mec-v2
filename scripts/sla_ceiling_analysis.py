"""Analytical SLA-ceiling check: for each tier, find the best achievable delay
(over offload target / mu / full resources, ignoring contention) and whether the
similarity + delay constraints can be jointly met. Also estimates the aggregate
compute-capacity vs offered-load ratio to gauge the contention-limited ceiling.
"""

from __future__ import annotations

import math

# --- params from configs/default.yaml + improved configs ---
DATA = 2.0e6
CYCLES = 1.0e9
C_SEM = 0.5
S_MAX = 1.0
LAMBDA = 5.0
NOISE = 1.0e-13
FREQ = 2.0e9
TX_POWER = 0.5  # phone max_power
TERM_CPU = {"phone": 1.0e9, "sensor": 0.6e9}

UAV_CPU, UAV_BW = 5.0e9, 20.0e6
HAP_CPU, HAP_BW = 50.0e9, 100.0e6

TIERS = {  # name: (t_max, s_min, weight)
    "critical": (0.1, 0.92, 0.2),
    "interactive": (0.3, 0.85, 0.5),
    "best_effort": (0.8, 0.75, 0.3),
}


def sim(mu):
    return S_MAX * (1.0 - math.exp(-LAMBDA * mu))


def min_mu_for(s_min):
    # s_max*(1-exp(-lam*mu)) >= s_min
    return -math.log(1.0 - s_min / S_MAX) / LAMBDA


def path_loss_db(dist, los=True):
    pl = 20 * math.log10(max(dist, 1.0)) + 20 * math.log10(FREQ) - 147.55
    return pl if los else pl + 20.0


def gain(dist):
    return 10 ** (-path_loss_db(dist) / 10.0)


def rate(bw, g):
    snr = TX_POWER * g / NOISE
    return bw * math.log2(1.0 + snr)


def delay_for(target, mu, term_cpu, cpu_frac=1.0, bw_frac=1.0, dist=None):
    t_sem = C_SEM * (1.0 / max(mu, 1e-3) - 1.0) * DATA / term_cpu
    if target == "local":
        t_comp = CYCLES / (term_cpu * cpu_frac)
        return t_sem + t_comp
    if target == "uav":
        bw = UAV_BW * bw_frac + 1e5
        cpu = UAV_CPU * cpu_frac
        d = dist if dist is not None else 100.0
    else:  # hap
        bw = HAP_BW * bw_frac + 1e5
        cpu = HAP_CPU * cpu_frac
        d = dist if dist is not None else 20000.0
    t_tx = (mu * DATA) / rate(bw, gain(d))
    t_comp = (mu * CYCLES) / cpu
    return t_sem + t_tx + t_comp


print("=" * 74)
print("PER-TIER FEASIBILITY (best case: full resources, good channel, phone)")
print("=" * 74)
ceiling_weight = 0.0
for name, (t_max, s_min, w) in TIERS.items():
    mu = max(min_mu_for(s_min) + 0.005, 0.2)
    print(f"\n[{name}] t_max={t_max}s s_min={s_min} weight={w}  -> need mu>={mu:.3f} (sim={sim(mu):.3f})")
    feasible = False
    for tgt in ("local", "uav", "hap"):
        d = delay_for(tgt, mu, TERM_CPU["phone"], cpu_frac=1.0, bw_frac=1.0)
        ok = d <= t_max
        feasible = feasible or ok
        print(f"    {tgt:6s}: delay={d*1000:7.1f} ms   {'OK' if ok else 'FAIL'}")
    if feasible:
        ceiling_weight += w
    print(f"    => tier {'FEASIBLE' if feasible else 'INFEASIBLE'} (single task, no contention)")

print("\n" + "=" * 74)
print(f"Ideal SLA ceiling (sum of feasible tier weights): {ceiling_weight:.2f}")

# --- contention / capacity estimate ---
print("\n" + "=" * 74)
print("CAPACITY vs LOAD (contention-limited ceiling)")
print("=" * 74)
num_terminals = 16
arrival = 0.235
slots_per_frame = 10
# offered compute load per second (cycles): tasks/slot * cycles, slot=1s? slot_duration=1
tasks_per_slot = num_terminals * arrival
# Each task needs ~mu*CYCLES remote cycles; assume mu~0.5 avg
offered_cycles = tasks_per_slot * 0.5 * CYCLES
total_remote_cpu = 3 * UAV_CPU + 2 * HAP_CPU  # 3 UAV + 2 HAP
print(f"tasks/slot ~ {tasks_per_slot:.2f}, offered remote cycles/slot ~ {offered_cycles:.2e}")
print(f"aggregate remote CPU (3 UAV + 2 HAP) = {total_remote_cpu:.2e} cycles/s")
print(f"compute utilization ~ {offered_cycles/total_remote_cpu*100:.1f}% (>100% => queueing/SLA loss)")
# bandwidth
total_bw = 3 * UAV_BW + 2 * HAP_BW
print(f"aggregate bandwidth = {total_bw/1e6:.0f} MHz across {num_terminals} terminals")
