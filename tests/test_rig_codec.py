"""Genome <-> rig encoding for the NSGA-II search (scripts/moo/rig_codec.py)."""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from moo import rig_codec as rc  # noqa: E402

SPACE_YAML = """\
name: test_space
attachment:
  type: spherical
  radii_m: [0.10, 0.20]
  colatitudes_deg: [10, 40]
sensors:
  allowed: ['small', 'big']
  min_total: 1
  max_total: 4
constraints:
  mesh_points: 2000
"""


@pytest.fixture
def catalog(tmp_path):
    tof = tmp_path / "sensors" / "tof"
    tof.mkdir(parents=True)
    (tof / "small.yaml").write_text(yaml.safe_dump({"model": "small", "dimensions": {"x": 0.01, "y": 0.01}, "cost": 10.0}))
    (tof / "big.yaml").write_text(yaml.safe_dump({"model": "big", "dimensions": {"x": 0.06, "y": 0.08}, "cost": 100.0}))
    (tof / "nodims.yaml").write_text(yaml.safe_dump({"model": "nodims", "cost": 1.0}))
    (tof / "empty.yaml").write_text("")
    return rc.SensorCatalog(str(tmp_path / "sensors"))


@pytest.fixture
def space(tmp_path):
    path = tmp_path / "space.yaml"
    path.write_text(SPACE_YAML)
    return rc.SearchSpace.from_yaml(str(path))


def test_bounds(space):
    assert space.n_var == 3 + 4
    assert space.xl.tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert space.xu.tolist() == [1, 1, 4, 1, 1, 1, 1]


def test_genome_round_trip_and_clipping(space):
    g = rc.Genome.from_x([1, 0, 3, 1, 0, 1, 0], space)
    assert g.to_x().tolist() == [1, 0, 3, 1, 0, 1, 0]
    clipped = rc.Genome.from_x([9, -1, 0, 5, 0, 0, 0], space)  # out of bounds -> clipped to [xl, xu]
    assert clipped.to_x().tolist() == [1, 0, 1, 1, 0, 0, 0]


def test_unused_model_genes_do_not_change_the_rig(space):
    a = rc.Genome(0, 1, 2, (1, 0, 0, 0))
    b = rc.Genome(0, 1, 2, (1, 0, 1, 1))  # differs only after n_sensors
    c = rc.Genome(0, 1, 2, (0, 1, 0, 0))  # same models, swapped sites -> a different rig
    assert a.canonical() == b.canonical() and rc.rig_id(a, space) == rc.rig_id(b, space)
    assert rc.rig_id(a, space) != rc.rig_id(c, space)


def test_rig_id_depends_on_search_space(space, tmp_path):
    other = tmp_path / "other.yaml"
    other.write_text(SPACE_YAML.replace("0.20]", "0.25]"))  # same genome indexes a different radius
    g = rc.Genome(1, 1, 2, (0, 0, 0, 0))
    assert rc.rig_id(g, space) != rc.rig_id(g, rc.SearchSpace.from_yaml(str(other)))


def test_decode_matches_lloyd_and_lists_sensors_in_site_order(space, catalog):
    g = rc.Genome(1, 1, 3, (0, 1, 0, 1))
    d = rc.decode(g, space, catalog)
    assert (d.radius, d.colatitude) == (0.20, 40.0)
    assert d.models == ["small", "big", "small"]
    assert d.sensor_names == ["small_s00", "big_s01", "small_s02"]
    assert d.site_poses.shape == (3, 7)
    np.testing.assert_allclose(d.site_poses[0, :3], [0, 0, 0.20], atol=1e-9)  # site 0 is the Lloyd pole
    np.testing.assert_allclose(np.linalg.norm(d.site_poses[:, :3], axis=1), 0.20, atol=1e-6)  # all on the sphere
    assert d.cost == 120.0


def test_spacing_feasibility(space, catalog):
    roomy = rc.decode(rc.Genome(1, 1, 2, (0, 0, 0, 0)), space, catalog)  # 2 small sensors on a wide platform
    cramped = rc.decode(rc.Genome(0, 0, 4, (1, 1, 1, 1)), space, catalog)  # 4 big sensors on a tiny cap
    single = rc.decode(rc.Genome(0, 0, 1, (1, 0, 0, 0)), space, catalog)
    assert roomy.feasible and roomy.spacing_violation < 0
    assert not cramped.feasible and cramped.spacing_violation > 0
    assert single.feasible and np.isinf(single.min_spacing)
    # required distance is the mean of the two footprint diagonals
    small_big = rc.decode(rc.Genome(1, 1, 2, (0, 1, 0, 0)), space, catalog)
    required = 0.5 * (np.hypot(0.01, 0.01) + np.hypot(0.06, 0.08))
    assert small_big.spacing_violation == pytest.approx(required - small_big.min_spacing)


def test_missing_footprint_raises(catalog):
    with pytest.raises(ValueError, match="no footprint"):
        catalog.footprint("nodims")
    with pytest.raises(ValueError, match="empty"):
        catalog.footprint("empty")
    with pytest.raises(KeyError):
        catalog.footprint("unknown")


def test_rig_yaml(space, catalog, tmp_path):
    d = rc.decode(rc.Genome(1, 1, 2, (1, 0, 0, 0)), space, catalog)
    rig = rc.rig_yaml_dict(d, space, catalog, run_id="nsga_x", generation=3, parents=["rig_a", "rig_b"])
    path = rc.write_rig_yaml(str(tmp_path / "rigs" / f"{d.rig_id}.yaml"), rig)
    with open(path) as f:
        loaded = yaml.safe_load(f)
    assert loaded["name"] == d.rig_id and loaded["generation"] == 3 and loaded["parents"] == ["rig_a", "rig_b"]
    assert loaded["layout"] == {"type": "spherical", "radius": 0.2, "colatitude": 40.0}
    assert [s["name"] for s in loaded["sensors"]] == ["big_s00", "small_s01"]
    assert [s["site"] for s in loaded["sensors"]] == [0, 1]
    assert loaded["sensors"][0]["sensor_type"] == "tof"
    assert loaded["genome"] == {"radius_idx": 1, "colat_idx": 1, "n_sensors": 2, "models": [1, 0]}
    assert loaded["derived"]["cost_usd"] == 110.0 and len(loaded["derived"]["site_poses"]) == 2


def test_project_search_space_and_catalog_load():
    space = rc.SearchSpace.from_yaml(os.path.join(rc.PROJECT_DIR, "scripts", "moo", "config", "spherical_search_space.yaml"))
    catalog = rc.SensorCatalog()
    assert space.allowed_models == ("vl53l8cx",)
    for model in space.allowed_models:
        assert catalog.footprint(model) > 0 and catalog.cost(model) > 0
