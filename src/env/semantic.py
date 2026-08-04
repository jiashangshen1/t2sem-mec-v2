"""Semantic communication model."""

from __future__ import annotations


def clip_mu(mu: float, mu_min: float, mu_max: float) -> float:
    return max(mu_min, min(mu_max, mu))


def transmission_bits(data_bits: float, mu: float) -> float:
    return mu * data_bits


def semantic_extraction_delay(
    data_bits: float, mu: float, cpu_hz: float, c_sem: float
) -> float:
    return c_sem * (1.0 / max(mu, 1e-3) - 1.0) * data_bits / max(cpu_hz, 1e6)


def similarity(mu: float, s_max: float, lambda_acc: float) -> float:
    import math

    return s_max * (1.0 - math.exp(-lambda_acc * mu))


def remote_cpu_cycles(base_cycles: float, mu: float) -> float:
    return mu * base_cycles
