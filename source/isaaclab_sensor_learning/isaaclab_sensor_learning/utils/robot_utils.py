#!/usr/bin/env python3

from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import spawn_articulation_from_usd


def load_robot(usd_path: str, prim_path: str, env: int = 0) -> Articulation:

    robot = spawn_articulation_from_usd(
        usd_path=usd_path,
        prim_path=prim_path,
        env=env,
    )
    return robot
