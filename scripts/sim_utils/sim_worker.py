#!/usr/bin/env python3
"""
Simulation entry point; runs in its own process, once per evaluation.

Usage: `python3 sim_worker.py --sensor-cfg sensor_cfg.yaml --out outputs.pkl

Keep heavy imports (Isaac Lab etc.) inside run_simulation, so the optimizer process
can import SimOutputs from this module without loading them.
"""
import argparse
import numpy as np
import pickle
import yaml

from dataclasses import asdict, dataclass


@dataclass
class SimOutputs:
    poses: np.ndarray
    t_final: float
    bin_counts: dict[str, np.ndarray]
    instance_map: dict[str, dict[str, int]]


def run_simulation(sensor_cfg: dict) -> SimOutputs:
    """Run the IsaacLab simulation with 'cfg' and return the outputs below"""
    raise NotImplementedError


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--sensor-cfg", required=True)
#     parser.add_argument("--out", required=True)

#     args = parser.parse_args()

