#!/usr/bin/env python3
import re


_ENV_RE = re.compile(r"/envs/(env_\d+)(?:/|$)")


def parse_instance(path: str, semantic_labels: list[str]) -> str:
    """Project-specific parser: prim path -> (env_name, semantic_label).

    Example paths:
        /World/envs/env_0/tree/trunkId_0                          -> ("env_0", "trunk")
        /World/envs/env_0/tree/trunkId_0/branchL_0_Id_007         -> ("env_0", "branch")
        /World/envs/env_0/tree/trunkId_0/spurId_0806              -> ("env_0", "spur")

    The label is the entry of `semantic_labels` that the last path component starts with
    (longest match wins, so "branch" and "branchlet" can coexist).
    """
    leaf = path.rstrip("/").rsplit("/", 1)[-1]
    matches = [label for label in semantic_labels if leaf.startswith(label)]
    if not matches:
        raise ValueError(f"'{leaf}' does not start with any of {semantic_labels} (path: {path})")
    return max(matches, key=len)

