"""NSGA-II problem, objectives and operators with a fake simulator (no Isaac Sim)."""

import os
import sys
import threading

import dill
import numpy as np
import pytest
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual
from pymoo.parallelization import StarmapParallelization
from pymoo.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from moo import rig_codec as rc  # noqa: E402
from moo.objectives import objectives_from_metrics, time_objective  # noqa: E402
from moo.operators import (  # noqa: E402
    RigCrossover,
    RigDuplicateElimination,
    RigMutation,
    RigRepair,
    RigSampling,
)
from moo.run_dir import RunDir  # noqa: E402
from moo.sensor_cfg_problem import SensorRigProblem  # noqa: E402

BUDGET, C_REF = 180.0, 0.4
SPACE_YAML = """\
name: test_space
attachment: {type: spherical, radii_m: [0.10, 0.20, 0.30], colatitudes_deg: [10, 30, 50]}
sensors: {allowed: ['small', 'big'], min_total: 1, max_total: 5}
constraints: {mesh_points: 1500}
"""


@pytest.fixture
def space(tmp_path):
    (tmp_path / "space.yaml").write_text(SPACE_YAML)
    return rc.SearchSpace.from_yaml(str(tmp_path / "space.yaml"))


@pytest.fixture
def catalog(tmp_path):
    tof = tmp_path / "sensors" / "tof"
    tof.mkdir(parents=True)
    (tof / "small.yaml").write_text(yaml.safe_dump({"dimensions": {"x": 0.01, "y": 0.01}, "cost": 10.0}))
    (tof / "big.yaml").write_text(yaml.safe_dump({"dimensions": {"x": 0.05, "y": 0.05}, "cost": 50.0}))
    return rc.SensorCatalog(str(tmp_path / "sensors"))


def fake_metrics(n_sensors: int, radius: float, n_envs: int = 2) -> dict:
    """metrics.json in evaluate_rig.py's format: more sensors / larger radius -> more coverage, sooner."""
    coverage = min(0.9, 0.05 * n_sensors + radius)
    reached = coverage >= C_REF
    env = {"reason_name": "TIME_BUDGET", "coverage_total": coverage, "reached_reference": reached,
           "time_to_reference_s": 100.0 / n_sensors if reached else None, "path_length_m": 30.0,
           "path_to_reference_m": 10.0 if reached else None}  # fmt: skip
    return {"status": "ok", "envs": [dict(env) for _ in range(n_envs)],
            "config": {"reference_coverage": C_REF, "termination": {"max_sim_time_s": BUDGET}}}  # fmt: skip


class FakeEvaluator:
    def __init__(self, fail_rigs=()):
        self.calls = []
        self.fail_rigs = set(fail_rigs)
        self._lock = threading.Lock()

    def __call__(self, rig_yaml_path, out_dir):
        with open(rig_yaml_path) as f:
            rig = yaml.safe_load(f)
        with self._lock:
            self.calls.append(rig["name"])
        if rig["name"] in self.fail_rigs:
            return {"status": "failed", "error": "CUDA out of memory", "out_of_memory": True}
        return fake_metrics(len(rig["sensors"]), rig["layout"]["radius"])


def make_problem(space, catalog, tmp_path, evaluator=None, runner=None):
    run_dir = RunDir("nsga_test", runs_dir=str(tmp_path / "runs"))
    return SensorRigProblem(space, catalog, BUDGET, run_dir, evaluator or FakeEvaluator(), runner)


# ----------------------------------------------------------------------------------------------------------------------
# Objectives
# ----------------------------------------------------------------------------------------------------------------------


def test_time_objective_penalty():
    reached = {"reached_reference": True, "time_to_reference_s": 42.0, "coverage_total": 0.6}
    assert time_objective(reached, BUDGET, C_REF) == 42.0
    short = lambda c: {"reached_reference": False, "time_to_reference_s": None, "coverage_total": c}  # noqa: E731
    assert time_objective(short(0.2), BUDGET, C_REF) == pytest.approx(270.0)  # half short -> 1.5 x budget
    assert time_objective(short(0.0), BUDGET, C_REF) == pytest.approx(360.0)  # worst case: 2 x budget
    assert time_objective(short(0.39), BUDGET, C_REF) > BUDGET  # any miss is worse than any hit (<= budget)


def test_objectives_from_metrics():
    m = fake_metrics(n_sensors=4, radius=0.3)  # coverage 0.5, reached at 25 s
    out = objectives_from_metrics(m, cost=40.0)
    assert out["F"] == pytest.approx([-0.5, 25.0, 40.0])
    assert out["n_envs_reached_reference"] == 2 and out["path_length_mean"] == 30.0
    with pytest.raises(ValueError):
        objectives_from_metrics({"status": "failed", "error": "boom"}, cost=1.0)


# ----------------------------------------------------------------------------------------------------------------------
# Problem: caching, infeasible rigs, failures, persistence
# ----------------------------------------------------------------------------------------------------------------------


def test_feasible_rig_is_simulated_once_and_cached(space, catalog, tmp_path):
    problem = make_problem(space, catalog, tmp_path)
    x = rc.Genome(2, 2, 3, (0, 0, 0, 1, 1)).to_x()
    twin = rc.Genome(2, 2, 3, (0, 0, 0, 0, 0)).to_x()  # differs only in unused model genes: same rig
    out1, out2 = problem.evaluate(np.array([x, twin]), return_as_dictionary=True)["F"]
    assert problem.evaluator.calls == [rc.rig_id(rc.Genome.from_x(x, space), space)]
    np.testing.assert_allclose(out1, out2)
    np.testing.assert_allclose(out1, [-0.45, 100.0 / 3, 30.0])
    assert os.path.exists(problem.run_dir.rig_yaml(problem.evaluator.calls[0]))


def test_infeasible_rig_is_not_simulated(space, catalog, tmp_path):
    problem = make_problem(space, catalog, tmp_path)
    cramped = rc.Genome(0, 0, 5, (1, 1, 1, 1, 1)).to_x()  # 5 big sensors on the smallest, narrowest cap
    res = problem.evaluate(np.array([cramped]), return_as_dictionary=True)
    assert problem.evaluator.calls == []
    assert res["G"][0][0] > 0 and res["G"][0][1] < 0
    row = problem.run_dir.cached(rc.rig_id(rc.Genome.from_x(cramped, space), space))
    assert row["status"] == "infeasible"


def test_failed_simulation_is_infeasible_and_recorded(space, catalog, tmp_path):
    x = rc.Genome(2, 2, 2, (0,) * 5).to_x()
    rid = rc.rig_id(rc.Genome.from_x(x, space), space)
    problem = make_problem(space, catalog, tmp_path, evaluator=FakeEvaluator(fail_rigs={rid}))
    res = problem.evaluate(np.array([x]), return_as_dictionary=True)
    assert res["G"][0][1] > 0
    row = problem.run_dir.cached(rid)
    assert row["status"] == "failed" and "out of memory" in row["error"]


def test_index_survives_restart(space, catalog, tmp_path):
    x = rc.Genome(1, 1, 2, (0,) * 5).to_x()
    make_problem(space, catalog, tmp_path).evaluate(np.array([x]))
    restarted = make_problem(space, catalog, tmp_path)  # new process: cache loaded from index.csv
    restarted.evaluate(np.array([x]))
    assert restarted.evaluator.calls == []


def test_problem_pickles_without_runtime_parts(space, catalog, tmp_path):
    problem = make_problem(space, catalog, tmp_path)
    loaded = dill.loads(dill.dumps(problem))
    assert loaded.evaluator is None and loaded.run_dir is None
    loaded.attach(RunDir("nsga_test", runs_dir=str(tmp_path / "runs")), FakeEvaluator())
    loaded.evaluate(np.array([rc.Genome(1, 1, 2, (0,) * 5).to_x()]))


# ----------------------------------------------------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------------------------------------------------


def test_sampling_puts_seeds_first_and_stays_in_bounds(space, catalog, tmp_path):
    problem = make_problem(space, catalog, tmp_path)
    seed = rc.Genome(1, 1, 3, (1, 0, 1, 0, 0)).to_x()
    X = RigSampling(space, seeds=[seed])._do(problem, 30, random_state=np.random.default_rng(0))
    assert X.shape == (30, space.n_var) and (X[0] == seed).all()
    assert (X >= space.xl).all() and (X <= space.xu).all()


def test_crossover_children_recombine_parent_genes(space, catalog, tmp_path):
    problem = make_problem(space, catalog, tmp_path)
    p0 = np.zeros(space.n_var, dtype=int) + space.xl
    p1 = space.xu.copy()
    X = np.stack([np.stack([p0] * 10), np.stack([p1] * 10)])  # (n_parents, n_matings, n_var)
    children = RigCrossover(space)._do(problem, X, random_state=np.random.default_rng(0))
    assert children.shape == X.shape
    for c0, c1 in zip(children[0], children[1]):
        assert all(v in (a, b) for v, a, b in zip(c0, p0, p1))
        np.testing.assert_array_equal(c0 + c1, p0 + p1)  # genes were swapped, not invented


def test_mutation_always_changes_the_rig_and_stays_in_bounds(space, catalog, tmp_path):
    problem = make_problem(space, catalog, tmp_path)
    X = RigSampling(space)._do(problem, 200, random_state=np.random.default_rng(1))
    M = RigMutation(space, prob_platform=0.0)._do(problem, X, random_state=np.random.default_rng(2))
    assert (M >= space.xl).all() and (M <= space.xu).all()
    for a, b in zip(X, M):
        assert rc.Genome.from_x(a, space).canonical() != rc.Genome.from_x(b, space).canonical()


def test_repair_and_duplicates(space, catalog, tmp_path):
    problem = make_problem(space, catalog, tmp_path)
    repaired = RigRepair(space)._do(problem, np.array([[9.4, -2, 99, 0, 1, 7, 0, 0]]))
    assert repaired.tolist() == [[2, 0, 5, 0, 1, 1, 0, 0]]
    dup = RigDuplicateElimination(space)
    a = Individual(X=rc.Genome(1, 1, 2, (0, 1, 0, 0, 0)).to_x())
    b = Individual(X=rc.Genome(1, 1, 2, (0, 1, 1, 1, 1)).to_x())  # same rig, different unused genes
    c = Individual(X=rc.Genome(1, 1, 2, (1, 0, 0, 0, 0)).to_x())  # models on other sites
    assert dup.is_equal(a, b) and not dup.is_equal(a, c)


# ----------------------------------------------------------------------------------------------------------------------
# End to end with pymoo's NSGA-II
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("parallel", [False, True])
def test_nsga2_runs_with_fake_simulator(space, catalog, tmp_path, parallel):
    from multiprocessing.pool import ThreadPool

    pool = ThreadPool(2) if parallel else None
    try:
        evaluator = FakeEvaluator()
        runner = StarmapParallelization(pool.starmap) if pool else None
        problem = make_problem(space, catalog, tmp_path, evaluator=evaluator, runner=runner)
        algorithm = NSGA2(
            pop_size=8,
            sampling=RigSampling(space, seeds=[rc.Genome(1, 1, 3, (0,) * 5).to_x()]),
            crossover=RigCrossover(space),
            mutation=RigMutation(space),
            repair=RigRepair(space),
            eliminate_duplicates=RigDuplicateElimination(space),
        )
        res = minimize(problem, algorithm, ("n_gen", 4), seed=3, verbose=False)
    finally:
        if pool:
            pool.close()
    assert res.F is not None and res.F.shape[1] == 3
    assert len(evaluator.calls) == len(set(evaluator.calls))  # no rig simulated twice
    assert len(problem.run_dir) >= len(evaluator.calls)


class AlwaysFailing(FakeEvaluator):
    def __call__(self, rig_yaml_path, out_dir):
        super().__call__(rig_yaml_path, out_dir)
        return {"status": "failed", "error": "KeyError: 'env_2'"}


def test_run_stops_after_consecutive_failures(space, catalog, tmp_path):
    run_dir = RunDir("nsga_test", runs_dir=str(tmp_path / "runs"))
    problem = SensorRigProblem(space, catalog, BUDGET, run_dir, AlwaysFailing(), max_consecutive_failures=3)
    genomes = [rc.Genome(2, 2, n, (0,) * 5).to_x() for n in (1, 2, 3, 4)]
    with pytest.raises(RuntimeError, match="3 simulations failed in a row.*env_2.*--resume nsga_test"):
        problem.evaluate(np.array(genomes))
    assert len(problem.evaluator.calls) == 3


def test_failed_rigs_are_retried_after_restart_but_finished_ones_are_not(space, catalog, tmp_path):
    x_fail, x_ok = rc.Genome(2, 2, 2, (0,) * 5).to_x(), rc.Genome(2, 2, 3, (0,) * 5).to_x()
    rid_fail = rc.rig_id(rc.Genome.from_x(x_fail, space), space)
    make_problem(space, catalog, tmp_path, evaluator=FakeEvaluator(fail_rigs={rid_fail})).evaluate(
        np.array([x_fail, x_ok])
    )
    restarted = make_problem(space, catalog, tmp_path)  # the cause of the failure has been fixed
    restarted.evaluate(np.array([x_fail, x_ok]))
    assert restarted.evaluator.calls == [rid_fail]  # only the failed rig is simulated again
    assert restarted.run_dir.cached(rid_fail)["status"] == "ok"
