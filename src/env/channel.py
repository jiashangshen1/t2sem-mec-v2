"""Air-ground channel and delay models."""

from __future__ import annotations

import numpy as np


def distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def los_probability(elevation_deg: float, a: float, b: float) -> float:
    return 1.0 / (1.0 + a * np.exp(-b * (elevation_deg - a)))


def path_loss_db(dist_m: float, freq_hz: float, los: bool) -> float:
    c = 3e8
    pl_fs = 20 * np.log10(max(dist_m, 1.0)) + 20 * np.log10(freq_hz) - 147.55
    return pl_fs if los else pl_fs + 20.0


def channel_gain(
    ue_pos: np.ndarray,
    aerial_pos: np.ndarray,
    carrier_freq: float,
    los_a: float,
    los_b: float,
    rng: np.random.Generator,
) -> float:
    dist = distance_3d(ue_pos, aerial_pos)
    horiz = max(np.linalg.norm(ue_pos[:2] - aerial_pos[:2]), 1.0)
    elev = np.degrees(np.arctan2(aerial_pos[2] - ue_pos[2], horiz))
    p_los = los_probability(elev, los_a, los_b)
    los = rng.random() < p_los
    pl = path_loss_db(dist, carrier_freq, los)
    return 10 ** (-pl / 10.0)


def shannon_rate(bandwidth: float, tx_power: float, gain: float, noise: float) -> float:
    snr = tx_power * gain / max(noise, 1e-20)
    return bandwidth * np.log2(1.0 + snr)


def transmission_delay(data_bits: float, rate_bps: float) -> float:
    return data_bits / max(rate_bps, 1e3)


def computation_delay(cycles: float, cpu_hz: float) -> float:
    return cycles / max(cpu_hz, 1e6)
