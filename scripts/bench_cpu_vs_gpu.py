"""Compare one DRL training episode: CPU vs CUDA (same seed, skip BC)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env, train_t2sem
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config


def bench_episodes(device: str, n: int = 3, seed: int = 42) -> dict:
    cfg = load_config("configs/default.yaml", "configs/improved_v4_pertier.yaml")
    cfg["drl"]["total_long_frames"] = 1
    cfg["drl"]["bc_demo_episodes"] = 0
    cfg["drl"]["bc_train_epochs"] = 0
    env = build_env(cfg, "full")
    agent = T2SemDRL(env, device=device, use_long_agent=True)

    import torch

    times = []
    for i in range(n):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = agent.train_episode()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {
        "device": device,
        "seconds_mean": sum(times) / len(times),
        "seconds_list": times,
        "sla_last": m.sla_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    import torch

    print(f"platform_torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    devices = []
    if args.only in ("cpu", "both"):
        devices.append("cpu")
    if args.only in ("cuda", "both") and torch.cuda.is_available():
        devices.append("cuda")

    results = []
    for dev in devices:
        if dev == "cuda":
            torch.cuda.synchronize()
        r = bench_episodes(dev, n=args.episodes)
        results.append(r)
        print(
            f"device={dev} mean={r['seconds_mean']:.3f}s "
            f"per_ep={args.episodes} times={[round(t,2) for t in r['seconds_list']]}"
        )

    if len(results) == 2:
        cpu_t, gpu_t = results[0]["seconds_mean"], results[1]["seconds_mean"]
        ratio = cpu_t / gpu_t
        winner = "cuda" if gpu_t < cpu_t else "cpu"
        print(f"\nWINNER={winner}  cpu/gpu={ratio:.2f}x  (ep time cpu={cpu_t:.2f}s gpu={gpu_t:.2f}s)")
        est_cpu_h = cpu_t * 300 * 6 * 3 / 3600
        est_gpu_h = gpu_t * 300 * 6 * 3 / 3600
        print(f"revision_suite_estimate: cpu={est_cpu_h:.1f}h  gpu={est_gpu_h:.1f}h")


if __name__ == "__main__":
    main()
