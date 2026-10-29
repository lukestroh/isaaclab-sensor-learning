from pathlib import Path

# env
decimation = 1
episode_length_s = 60.0
# - spaces definition
action_space = 7
observation_space = 4
state_space = 0
num_envs = 2
physics_dt = 1 / 120

# rig. Stand-in default; real rigs are passed in by the NSGA-II worker. Kept in <repo>/data/rigs, not in the package.
default_rig_yaml_path = str(Path(__file__).resolve().parents[6] / "data" / "rigs" / "test_rig0.yaml")
