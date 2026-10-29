#!/usr/bin/env python3
"""
Genome <-> rig encoding for the NSGA-II sensor-rig search. Runs without Isaac Sim.

Genome (integer vector):  [radius_idx, colat_idx, n_sensors, model_0, ..., model_{max_total-1}]

- radius / colatitude index the search space's discrete lists and define the spherical-cap mount platform.
- The Lloyd generator alone turns (radius, colatitude, n_sensors) into sites (site 0 = pole). It is the same code the
  simulation runs (loaded by file path, since the ``isaaclab_sensor_learning`` package needs Kit to import).
- model_i is the sensor model on Lloyd site i; only the first n_sensors model genes are used.
"""
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import yaml

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG_DIR = os.path.join(PROJECT_DIR, "source", "isaaclab_sensor_learning", "isaaclab_sensor_learning")
SENSOR_CFG_DIR = os.path.join(PKG_DIR, "config", "sensors")
SCHEMA_VERSION = 1
N_PLATFORM_GENES = 3  # radius_idx, colat_idx, n_sensors


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lloyd = _load_by_path("spherical_sensor_layout_generator", os.path.join(PKG_DIR, "sensor", "spherical_sensor_layout_generator.py"))


# ----------------------------------------------------------------------------------------------------------------------
# Search space and sensor catalog
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchSpace:
    name: str
    radii_m: tuple[float, ...]
    colatitudes_deg: tuple[float, ...]
    allowed_models: tuple[str, ...]
    min_total: int
    max_total: int
    constraints: dict = field(hash=False, compare=False)
    digest: str = ""  # hash of the yaml contents; part of every rig_id

    @classmethod
    def from_yaml(cls, path: str) -> "SearchSpace":
        with open(path) as f:
            text = f.read()
        cfg = yaml.safe_load(text)
        if cfg["attachment"]["type"] != "spherical":
            raise ValueError(f"Only spherical search spaces are supported, got {cfg['attachment']['type']!r}")
        space = cls(
            name=cfg["name"],
            radii_m=tuple(float(r) for r in cfg["attachment"]["radii_m"]),
            colatitudes_deg=tuple(float(c) for c in cfg["attachment"]["colatitudes_deg"]),
            allowed_models=tuple(cfg["sensors"]["allowed"]),
            min_total=int(cfg["sensors"]["min_total"]),
            max_total=int(cfg["sensors"]["max_total"]),
            constraints=dict(cfg.get("constraints", {})),
            digest=hashlib.sha1(text.encode()).hexdigest()[:8],
        )
        if not 1 <= space.min_total <= space.max_total:
            raise ValueError(f"Need 1 <= min_total <= max_total, got {space.min_total}, {space.max_total}")
        return space

    @property
    def n_var(self) -> int:
        return N_PLATFORM_GENES + self.max_total

    @property
    def xl(self) -> np.ndarray:
        return np.array([0, 0, self.min_total] + [0] * self.max_total, dtype=int)

    @property
    def xu(self) -> np.ndarray:
        n_models = len(self.allowed_models)
        return np.array(
            [len(self.radii_m) - 1, len(self.colatitudes_deg) - 1, self.max_total] + [n_models - 1] * self.max_total,
            dtype=int,
        )


class SensorCatalog:
    """Sensor yamls under ``config/sensors/<sensor_type>/<model>.yaml``, by model name."""

    def __init__(self, sensor_cfg_dir: str = SENSOR_CFG_DIR):
        self.sensor_cfg_dir = sensor_cfg_dir
        self._entries = {}
        for sensor_type in sorted(os.listdir(sensor_cfg_dir)):
            type_dir = os.path.join(sensor_cfg_dir, sensor_type)
            if not os.path.isdir(type_dir):
                continue
            for fname in sorted(os.listdir(type_dir)):
                if fname.endswith(".yaml"):
                    self._entries[os.path.splitext(fname)[0]] = (sensor_type, os.path.join(type_dir, fname))

    def _load(self, model: str) -> tuple[str, dict]:
        if model not in self._entries:
            raise KeyError(f"No sensor yaml for model {model!r} in {self.sensor_cfg_dir}")
        sensor_type, path = self._entries[model]
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            raise ValueError(f"Sensor yaml {path} is empty")
        return sensor_type, data

    def sensor_type(self, model: str) -> str:
        return self._load(model)[0]

    def cost(self, model: str) -> float:
        _, data = self._load(model)
        if "cost" not in data:
            raise ValueError(f"Sensor {model!r} has no 'cost'")
        return float(data["cost"])

    def footprint(self, model: str) -> float:
        """Diagonal (m) of the sensor's mounting face (dimensions x, y); its optical axis is the box's z."""
        _, data = self._load(model)
        dims = data.get("dimensions")
        if not dims or "x" not in dims or "y" not in dims:
            raise ValueError(f"Sensor {model!r} has no footprint: 'dimensions' with 'x' and 'y' is required")
        return float(np.hypot(dims["x"], dims["y"]))


# ----------------------------------------------------------------------------------------------------------------------
# Genome
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Genome:
    radius_idx: int
    colat_idx: int
    n_sensors: int
    models: tuple[int, ...]  # all max_total model genes; only the first n_sensors are used

    @classmethod
    def from_x(cls, x, space: SearchSpace) -> "Genome":
        x = np.rint(np.asarray(x, dtype=float)).astype(int)
        if x.shape != (space.n_var,):
            raise ValueError(f"Genome must have {space.n_var} genes, got shape {x.shape}")
        x = np.clip(x, space.xl, space.xu)
        return cls(int(x[0]), int(x[1]), int(x[2]), tuple(int(m) for m in x[N_PLATFORM_GENES:]))

    def to_x(self) -> np.ndarray:
        return np.array([self.radius_idx, self.colat_idx, self.n_sensors, *self.models], dtype=int)

    def canonical(self) -> tuple:
        """What defines the rig: unused model genes are ignored."""
        return (self.radius_idx, self.colat_idx, self.n_sensors, self.models[: self.n_sensors])

    def to_dict(self) -> dict:
        return {
            "radius_idx": self.radius_idx,
            "colat_idx": self.colat_idx,
            "n_sensors": self.n_sensors,
            "models": list(self.models[: self.n_sensors]),
        }


def rig_id(genome: Genome, space: SearchSpace) -> str:
    """Stable id: hash of the canonical genome and the search space it indexes into."""
    payload = json.dumps({"space": space.digest, "genome": genome.canonical()}, sort_keys=True)
    return "rig_" + hashlib.sha1(payload.encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------------------------------------------------
# Decode
# ----------------------------------------------------------------------------------------------------------------------


def layout_dict(radius: float, colatitude: float, space: SearchSpace) -> dict:
    """The Lloyd layout dict exactly as SensorLearningEnv builds it: rig ``layout`` merged with search-space
    ``constraints``."""
    return {"type": "spherical", "radius": radius, "colatitude": colatitude, **space.constraints}


@lru_cache(maxsize=4096)
def lloyd_sites(radius: float, colatitude: float, n_sensors: int, constraints_items: tuple) -> np.ndarray:
    """(n_sensors, 7) site poses [x, y, z, qx, qy, qz, qw] from the same Lloyd generator and arguments as RigGenerator."""
    layout = {"radius": radius, "colatitude": colatitude, **dict(constraints_items)}
    generator = _lloyd.LloydSphereSensorLayout(
        sensors=[{} for _ in range(n_sensors)],
        n_sensors=n_sensors,
        radius=layout["radius"],
        colatitude=layout["colatitude"],
        max_iter=layout.get("max_iter", 200),
        tol=layout.get("tol", 1e-6),
        mesh_points=layout.get("mesh_points", 8000),
    )
    return np.asarray(generator.generate_sensor_layout()["poses"], dtype=float)


@dataclass
class RigDesign:
    rig_id: str
    genome: Genome
    radius: float
    colatitude: float
    models: list[str]  # model on site i
    site_poses: np.ndarray  # (n_sensors, 7)
    min_spacing: float  # smallest center-to-center distance between sites (m); inf for one sensor
    spacing_violation: float  # max over site pairs of (required - actual) distance; <= 0 is feasible
    cost: float

    @property
    def feasible(self) -> bool:
        return self.spacing_violation <= 0.0

    @property
    def sensor_names(self) -> list[str]:
        return [f"{model}_s{site:02d}" for site, model in enumerate(self.models)]


def decode(genome: Genome, space: SearchSpace, catalog: SensorCatalog) -> RigDesign:
    """
    Genome -> rig design, including the spacing feasibility check.

    Two sensors fit side by side when their centers are at least the mean of their footprint diagonals apart
    (each sensor treated as a disk of its footprint diagonal). Raises if a used model has no footprint.
    """
    radius = space.radii_m[genome.radius_idx]
    colatitude = space.colatitudes_deg[genome.colat_idx]
    models = [space.allowed_models[m] for m in genome.models[: genome.n_sensors]]
    poses = lloyd_sites(radius, colatitude, genome.n_sensors, tuple(sorted(space.constraints.items())))
    footprints = np.array([catalog.footprint(m) for m in models])

    if genome.n_sensors > 1:
        pos = poses[:, :3]
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        required = 0.5 * (footprints[:, None] + footprints[None, :])
        iu = np.triu_indices(genome.n_sensors, k=1)
        min_spacing = float(dist[iu].min())
        violation = float((required - dist)[iu].max())
    else:
        min_spacing, violation = float("inf"), -float("inf")

    return RigDesign(
        rig_id=rig_id(genome, space),
        genome=genome,
        radius=radius,
        colatitude=colatitude,
        models=models,
        site_poses=poses,
        min_spacing=min_spacing,
        spacing_violation=violation,
        cost=float(sum(catalog.cost(m) for m in models)),
    )


def rig_yaml_dict(
    design: RigDesign,
    space: SearchSpace,
    catalog: SensorCatalog,
    run_id: str | None = None,
    generation: int | None = None,
    parents: list[str] | None = None,
) -> dict:
    """Rig yaml in the ``test_rig0.yaml`` format plus NSGA metadata. ``sensors`` are listed in Lloyd site order."""
    return {
        "schema_version": SCHEMA_VERSION,
        "name": design.rig_id,
        "run_id": run_id,
        "generation": generation,
        "parents": list(parents or []),
        "search_space": f"{space.name}@{space.digest}",
        "genome": design.genome.to_dict(),
        "layout": {"type": "spherical", "radius": design.radius, "colatitude": design.colatitude},
        "sensors": [
            {"name": name, "model": model, "sensor_type": catalog.sensor_type(model), "site": site}
            for site, (name, model) in enumerate(zip(design.sensor_names, design.models))
        ],
        "derived": {
            "n_sensors": len(design.models),
            "cost_usd": design.cost,
            "min_site_spacing_m": design.min_spacing if np.isfinite(design.min_spacing) else None,
            "spacing_violation_m": design.spacing_violation if np.isfinite(design.spacing_violation) else None,
            "site_poses": np.round(design.site_poses, 6).tolist(),
        },
    }


def write_rig_yaml(path: str, rig: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(rig, f, sort_keys=False)
    return path


def genome_from_rig_yaml(path: str, space: SearchSpace) -> Genome:
    """
    Genome for a hand-written spherical rig yaml (e.g. data/rigs/test_rig0.yaml), to seed the population.

    Radius and colatitude snap to the nearest search-space values (ties go to the smaller one); every sensor's model
    must be allowed by the search space. Sensors keep their yaml order, i.e. sensor i -> Lloyd site i.
    """
    with open(path) as f:
        rig = yaml.safe_load(f)
    layout = rig["layout"]
    if layout["type"] != "spherical":
        raise ValueError(f"{path}: only spherical rigs can seed this search space")
    models = [s["model"] for s in rig["sensors"]]
    unknown = sorted(set(models) - set(space.allowed_models))
    if unknown:
        raise ValueError(f"{path}: models {unknown} are not allowed by search space {space.name}")
    if not space.min_total <= len(models) <= space.max_total:
        raise ValueError(f"{path}: {len(models)} sensors is outside [{space.min_total}, {space.max_total}]")
    model_genes = [space.allowed_models.index(m) for m in models] + [0] * (space.max_total - len(models))
    return Genome(
        radius_idx=int(np.argmin(np.abs(np.asarray(space.radii_m) - float(layout["radius"])))),
        colat_idx=int(np.argmin(np.abs(np.asarray(space.colatitudes_deg) - float(layout["colatitude"])))),
        n_sensors=len(models),
        models=tuple(model_genes),
    )
