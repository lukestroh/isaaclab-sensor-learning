#!/usr/bin/env python3
"""
Live progress of one NSGA-II generation: a tqdm bar over the generation's rigs that also advances *within* each
running rig, using the ``progress.json`` that evaluate_rig.py writes to its --out-dir (capture pose i / N, then NBV
time t / budget). Each rig is one unit: capture fills its first CAPTURE_SHARE, the NBV loop the rest.
"""
import json
import os
import sys
import threading

from tqdm import tqdm

CAPTURE_SHARE = 0.4  # share of a rig's wall time spent capturing (measured ~36% with 20 envs, 8 sensors)


def read_rig_progress(eval_dir: str) -> dict | None:
    try:
        with open(os.path.join(eval_dir, "progress.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None  # not started yet, or mid-replace


def rig_fraction(progress: dict | None) -> float:
    """How far one rig is, in [0, 1)."""
    if not progress:
        return 0.0
    if progress["phase"] == "capture":
        return CAPTURE_SHARE * progress["done"] / max(progress["total"], 1)
    if progress["phase"] == "nbv":
        return CAPTURE_SHARE + (1.0 - CAPTURE_SHARE) * min(progress["t"] / max(progress["budget"], 1e-9), 0.999)
    return 0.0


def describe(progress: dict | None) -> str:
    if not progress:
        return "starting"
    if progress["phase"] == "capture":
        return f"capture {progress['done']}/{progress['total']}"
    if progress["phase"] == "nbv":
        return f"NBV {progress['t']:.0f}/{progress['budget']:.0f} s ({progress['active_envs']} envs running)"
    return progress["phase"]


class GenerationProgress:
    def __init__(self, poll_s: float = 2.0, file=sys.stderr):
        self.poll_s = poll_s
        self.file = file
        self._lock = threading.Lock()
        self._bar = None
        self._stop = threading.Event()
        self._thread = None

    def start(self, generation: int, n_rigs: int) -> None:
        self.close()
        self._done = 0
        self._counts = {"ok": 0, "failed": 0, "infeasible": 0, "cached": 0}
        self._running: dict[str, str] = {}  # rig_id -> eval_dir
        self._best: tuple[float, str] | None = None
        self._bar = tqdm(
            total=n_rigs, desc=f"gen {generation}", unit="rig", file=self.file, dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total} rigs [{elapsed}<{remaining}]{postfix}",
        )  # fmt: skip
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def rig_running(self, rig_id: str, eval_dir: str) -> None:
        with self._lock:
            self._running[rig_id] = eval_dir
        self._refresh()

    def rig_done(self, rig_id: str, status: str, coverage: float | None = None, cached: bool = False) -> None:
        with self._lock:
            self._running.pop(rig_id, None)
            self._done += 1
            self._counts["cached" if cached else status] = self._counts.get("cached" if cached else status, 0) + 1
            if coverage is not None and (self._best is None or coverage > self._best[0]):
                self._best = (coverage, rig_id)
        self._refresh()

    def close(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        if self._bar is not None:
            self._refresh()
            self._bar.close()
            self._bar = None

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_s):
            self._refresh()

    def _refresh(self) -> None:
        with self._lock:
            if self._bar is None:
                return
            running = {rid: read_rig_progress(d) for rid, d in self._running.items()}
            counts = " ".join(f"{k} {v}" for k, v in self._counts.items() if v)
            parts = [counts] if counts else []
            parts += [f"{rid}: {describe(p)}" for rid, p in running.items()]
            if self._best is not None:
                parts.append(f"best {self._best[0]:.3f} ({self._best[1]})")
            self._bar.n = self._done + sum(rig_fraction(p) for p in running.values())
            self._bar.set_postfix_str(" | ".join(parts), refresh=False)
            self._bar.refresh()
