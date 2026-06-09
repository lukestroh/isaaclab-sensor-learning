# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

USD_DIR = os.path.join(PKG_DIR, "usd")

URDF_DIR = os.path.join(PKG_DIR, "urdf")

TREES_DIR = os.path.join(PKG_DIR, "trees")
