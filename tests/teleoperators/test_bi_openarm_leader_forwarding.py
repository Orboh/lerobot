# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BiOpenArmLeader rebuilds each arm's config field by field. A field added to
OpenArmLeaderConfigBase but not forwarded here is silently dropped — that is how
frozen_pose_path was lost on 2026-09-23 and the "frozen" arm kept teleoperating.
This test fails the moment a new field is not forwarded for both sides."""

import dataclasses
import inspect
import re

from lerobot.teleoperators.bi_openarm_leader import bi_openarm_leader
from lerobot.teleoperators.openarm_leader.config_openarm_leader import OpenArmLeaderConfigBase

# Fields the wrapper sets itself rather than copying (port/id/side come from the
# bi config or are fixed per arm).
OWN = {"port", "id", "calibration_dir", "gravity_side", "can_interface", "use_can_fd"}


def test_every_base_field_is_forwarded_for_both_arms():
    src = inspect.getsource(bi_openarm_leader.BiOpenArmLeader.__init__)
    missing = []
    for f in dataclasses.fields(OpenArmLeaderConfigBase):
        if f.name in OWN:
            continue
        for side in ("left", "right"):
            pat = rf"\b{f.name}\s*=\s*config\.{side}_arm_config\.{f.name}\b"
            if not re.search(pat, src):
                missing.append(f"{side}: {f.name}")
    assert not missing, "BiOpenArmLeader does not forward: " + ", ".join(missing)
