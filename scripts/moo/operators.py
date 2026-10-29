#!/usr/bin/env python3
"""
Genetic operators for the sensor-rig genome  [radius_idx, colat_idx, n_sensors, model_0, ..., model_{max_total-1}].

- Platform genes (radius, colatitude, sensor count) are ordinal: mutation steps them by +-1 (sometimes +-2 for the
  count). Crossover swaps them between parents (uniform): with only 5-20 values per gene, swapping keeps every child
  a recombination of its parents.
- Model genes are categorical (model on Lloyd site i): uniform crossover and random-reset mutation.
All randomness comes from pymoo's ``random_state`` so runs are reproducible from the seed.
"""
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling

from moo.rig_codec import N_PLATFORM_GENES, Genome, SearchSpace

RADIUS, COLAT, N_SENSORS = 0, 1, 2


def _clip(X: np.ndarray, space: SearchSpace) -> np.ndarray:
    return np.clip(np.rint(X), space.xl, space.xu).astype(int)


class RigSampling(Sampling):
    """Seed genomes first (e.g. test_rig0), then uniform random rigs."""

    def __init__(self, space: SearchSpace, seeds: list[np.ndarray] | None = None):
        super().__init__()
        self.space = space
        self.seeds = [np.asarray(s, dtype=int) for s in (seeds or [])]

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        rng = random_state
        X = rng.integers(self.space.xl, self.space.xu + 1, size=(n_samples, self.space.n_var))
        for i, seed in enumerate(self.seeds[:n_samples]):
            X[i] = seed
        return _clip(X, self.space)


class RigCrossover(Crossover):
    """Two parents -> two children by uniform crossover of every gene (each gene swapped with probability 0.5)."""

    def __init__(self, space: SearchSpace, prob: float = 0.9):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.space = space

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        rng = random_state
        _, n_matings, n_var = X.shape
        swap = rng.random((n_matings, n_var)) < 0.5
        children = X.copy()
        children[0][swap], children[1][swap] = X[1][swap], X[0][swap]
        return children


class RigMutation(Mutation):
    """
    Per child: each platform gene steps by +-1 with probability ``prob_platform`` (the sensor count steps +-2 with
    probability ``prob_big_count_step`` instead); each used model gene resets to a random model with probability
    1 / n_sensors. At least one gene always changes.
    """

    def __init__(self, space: SearchSpace, prob_platform: float = 0.3, prob_big_count_step: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.space = space
        self.prob_platform = prob_platform
        self.prob_big_count_step = prob_big_count_step

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        rng = random_state
        X = X.astype(int).copy()
        n_models = len(self.space.allowed_models)
        for x in X:
            before = x.copy()
            for gene in (RADIUS, COLAT, N_SENSORS):
                if rng.random() < self.prob_platform:
                    size = 2 if gene == N_SENSORS and rng.random() < self.prob_big_count_step else 1
                    x[gene] += size * rng.choice((-1, 1))
            if n_models > 1:
                n_used = int(np.clip(x[N_SENSORS], self.space.min_total, self.space.max_total))
                for site in range(n_used):
                    if rng.random() < 1.0 / n_used:
                        x[N_PLATFORM_GENES + site] = rng.integers(0, n_models)
            x[:] = _clip(x, self.space)
            if Genome.from_x(x, self.space).canonical() == Genome.from_x(before, self.space).canonical():
                self._force_change(x, rng)
        return X

    def _force_change(self, x: np.ndarray, rng) -> None:
        """Step one platform gene that can move (keeps mutation from returning the parent unchanged)."""
        for gene in rng.permutation([RADIUS, COLAT, N_SENSORS]):
            options = [x[gene] + d for d in (-1, 1) if self.space.xl[gene] <= x[gene] + d <= self.space.xu[gene]]
            if options:
                x[gene] = rng.choice(options)
                return


class RigRepair(Repair):
    """Round and clip every gene to its bounds (also enforces the hard sensor-count bounds)."""

    def __init__(self, space: SearchSpace):
        super().__init__()
        self.space = space

    def _do(self, problem, X, **kwargs):
        return _clip(X, self.space)


class RigDuplicateElimination(ElementwiseDuplicateElimination):
    """Two genomes are the same rig when their canonical forms match (unused model genes are ignored)."""

    def __init__(self, space: SearchSpace):
        super().__init__()
        self.space = space

    def is_equal(self, a, b) -> bool:
        return Genome.from_x(a.X, self.space).canonical() == Genome.from_x(b.X, self.space).canonical()
