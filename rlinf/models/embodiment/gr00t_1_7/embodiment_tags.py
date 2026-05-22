# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf override of ``gr00t.data.embodiment_tags`` for N1.7.

This module is monkey-patched onto the upstream ``gr00t.data.embodiment_tags``
by ``rlinf.models.embodiment.gr00t_1_7.__init__.get_model`` via Patcher, so
any code importing ``EmbodimentTag`` after model construction sees this
extended enum + RLinf-specific id mapping.

Tag values and projector ids MUST match what GR00T N1.7 baked into shipped
checkpoints (see ``<ckpt>/embodiment_id.json`` for the source of truth). When
adding a new IsaacLab task that finetunes with ``new_embodiment``, the
projector id must equal the id used during SFT, otherwise the action expert
selects the wrong CategorySpecificMLP slot.
"""

from enum import Enum


class EmbodimentTag(Enum):
    # ===== Pretrain tags shipped in GR00T-N1.7-3B base =====
    OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT = "oxe_droid_relative_eef_relative_joint"
    XDOF = "xdof"
    XDOF_SUBTASK = "xdof_subtask"
    REAL_G1 = "real_g1"
    REAL_R1_PRO_SHARPA = "real_r1_pro_sharpa"
    REAL_R1_PRO_SHARPA_HUMAN = "real_r1_pro_sharpa_human"
    REAL_R1_PRO_SHARPA_MAXINSIGHTS = "real_r1_pro_sharpa_maxinsights"
    REAL_R1_PRO_SHARPA_MECKA = "real_r1_pro_sharpa_mecka"

    # ===== Posttrain tags (require a finetuned ckpt) =====
    UNITREE_G1 = "unitree_g1"
    UNITREE_G1_SONIC = "unitree_g1_sonic"
    SIMPLER_ENV_GOOGLE = "simpler_env_google"
    SIMPLER_ENV_WIDOWX = "simpler_env_widowx"
    LIBERO_PANDA = "libero_panda"

    # ===== Finetuning-only =====
    NEW_EMBODIMENT = "new_embodiment"

    # ===== Legacy RLinf tags kept so old yamls still resolve =====
    # These exist for back-compat with N1.6 RLinf configs. They are NOT
    # registered tags in GR00T N1.7 — using them with a real N1.7 ckpt will
    # fail at modality lookup unless you route them to NEW_EMBODIMENT via
    # `_resolve_embodiment_tag` in this package's __init__.
    LIBERO_FRANKA = "libero_franka"
    MANISKILL_WIDOWX = "maniskill_widowx"
    ISAACLAB_FRANKA = "isaaclab_franka"
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    GR1 = "gr1"
    AGIBOT_GENIE1 = "agibot_genie1"
    OXE_GOOGLE = "oxe_google"
    OXE_WIDOWX = "oxe_widowx"
    OXE_DROID = "oxe_droid"
    BEHAVIOR_R1_PRO = "behavior_r1_pro"


# Projector indices used by Gr00tN1d7 CategorySpecificMLP. Source: the
# embodiment_id.json shipped in the upstream GR00T N1.7 base checkpoint.
# Update entries here when a new IsaacLab task finetunes with a different id
# (the slot has to match the SFT-time mapping; see your ckpt's
# `embodiment_id.json` to confirm).
EMBODIMENT_TAG_MAPPING = {
    # N1.7 base-shipped ids
    EmbodimentTag.SIMPLER_ENV_GOOGLE.value: 0,
    EmbodimentTag.SIMPLER_ENV_WIDOWX.value: 1,
    EmbodimentTag.LIBERO_PANDA.value: 2,
    EmbodimentTag.NEW_EMBODIMENT.value: 10,
    EmbodimentTag.ROBOCASA_PANDA_OMRON.value: 13,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT.value: 24,
    EmbodimentTag.XDOF.value: 24,
    EmbodimentTag.REAL_G1.value: 25,
    EmbodimentTag.UNITREE_G1.value: 25,
    EmbodimentTag.REAL_R1_PRO_SHARPA.value: 27,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,

    # Legacy RLinf tags routed to ids RLinf used historically; harmless if
    # unused.
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.MANISKILL_WIDOWX.value: 30,
    EmbodimentTag.LIBERO_FRANKA.value: 31,
    EmbodimentTag.ISAACLAB_FRANKA.value: 31,
}
