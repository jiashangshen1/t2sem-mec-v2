"""Entity state containers for multi-UAV multi-terminal MEC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Task:
    terminal_id: int
    user_id: int
    device_type: str
    data_bits: float
    cpu_cycles: float
    tier: str
    arrival_slot: int
    deadline: float
    s_min: float


@dataclass
class TerminalState:
    terminal_id: int
    user_id: int
    device_type: str
    position: np.ndarray
    cpu_freq: float
    tx_power: float
    energy: float = 100.0
    queue: List[Task] = field(default_factory=list)


@dataclass
class UserState:
    user_id: int
    position: np.ndarray
    mobility_speed: float


@dataclass
class UAVState:
    uav_id: int
    position: np.ndarray
    velocity: np.ndarray
    cpu_freq: float
    bandwidth: float
    energy: float = 5000.0
    queue: List[Task] = field(default_factory=list)
    active_bw_load: float = 0.0
    active_cpu_load: float = 0.0


@dataclass
class HAPState:
    hap_id: int
    position: np.ndarray
    cpu_freq: float
    bandwidth: float
    queue: List[Task] = field(default_factory=list)
    active_bw_load: float = 0.0
    active_cpu_load: float = 0.0


@dataclass
class StepResult:
    terminal_id: int
    user_id: int
    device_type: str
    tier: str
    offload_target: str
    target_id: int
    mu: float
    delay: float
    similarity: float
    energy: float
    sla_met: bool
    t_max: float
    s_min: float


# Offload target encoding
TARGET_LOCAL = "local"
TARGET_UAV = "uav"
TARGET_HAP = "hap"
