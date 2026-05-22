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

"""RLinf adapter for GR00T N1.7.

Upstream API differences vs N1.6 that motivated this module (kept here so the
intent is visible at the entry point):

- N1.7 uses ``gr00t.model.gr00t_n1d7.gr00t_n1d7.Gr00tN1d7`` with a
  Cosmos-Reason2-2B / Qwen3-VL backbone (``backbone_embedding_dim=2048``)
  instead of N1.6's Eagle backbone (3584).
- ``Gr00tN1d7.__init__`` only accepts ``(config, transformers_loading_kwargs)``;
  everything else (``tune_visual``/``tune_llm``/``max_action_dim``/...) now
  lives in ``Gr00tN1d7Config``.
- ``gr00t.experiment.data_config`` and ``gr00t.model.transforms`` were removed
  in N1.7, so we no longer call ``load_data_config(...)`` here — modality
  config is expected to be supplied by the caller (e.g. an IsaacLab task that
  registers its own DataConfig via ``DATA_CONFIG_MAP``).
"""

import sys
import types
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Inject a stub for rlinf.envs.libero.asset_paths so env workers that don't
# ship the LIBERO assets can still import (N1.6 needed this; we keep it for
# parity with the LIBERO e2e tests that may share rollout workers).
_OLD_ASSET_PATHS = sys.modules.get("rlinf.envs.libero.asset_paths")
if _OLD_ASSET_PATHS is None:
    _stub = types.ModuleType("rlinf.envs.libero.asset_paths")

    def _noop(*args, **kwargs):
        pass

    _stub.apply_standard_libero_env_vars = _noop
    sys.modules["rlinf.envs.libero.asset_paths"] = _stub


def _load_ckpt_modality_config(ckpt_dir: Path):
    """Load the modality_configs dict that the ckpt was *actually* trained with.

    The Hugging Face ``AutoProcessor.from_pretrained`` route is unreliable for
    N1.7 finetuned ckpts: ``Gr00tN1d7Processor`` isn't registered with
    ``AutoProcessor`` (raises ``Unrecognized processing class``), so the
    fallback in ``GR00T_N1_7_ForRLActionPrediction.__init__`` silently sets
    ``self._modality_config = None`` and ``_load_metadata`` then falls back to
    ``valid_action_dim = max_action_dim = 132`` and ``image_nums = 1`` — both
    wrong for any real custom embodiment (e.g. our 28-dim G1+Dex3 setup with 3
    cameras).

    The SFT training script saves the true modality_configs dict (keyed by
    embodiment tag string) under ``ckpt/experiment_cfg/config.yaml`` as a
    YAML-pickled ``gr00t.configs.base_config.Config`` object. Load it with
    ``yaml.UnsafeLoader`` so we get back the real ``ModalityConfig`` objects
    (with ``.modality_keys`` / ``.delta_indices`` / ``.action_configs``
    attributes) that ``_load_metadata`` knows how to introspect.

    Returns the per-embodiment modality_configs dict, or ``None`` if the file
    isn't present (older ckpt layout — caller falls back to the old behaviour).
    """
    import yaml

    cfg_path = ckpt_dir / "experiment_cfg" / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        exp_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    return getattr(getattr(exp_cfg, "data", None), "modality_configs", None)


def _resolve_embodiment_tag(embodiment_tag_str: str):
    """Map RLinf yaml ``embodiment_tag`` strings to N1.7 ``EmbodimentTag``.

    N1.7 dropped ROBOCASA_PANDA_OMRON / BEHAVIOR_R1_PRO / GR1 and added
    UNITREE_G1, UNITREE_G1_SONIC, REAL_R1_PRO_SHARPA_*. For custom robots
    (IsaacLab tasks like ``assemble_trocar``) we route through
    ``NEW_EMBODIMENT``, which is the finetuning-only slot N1.7 reserves for
    user-defined embodiments.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    direct = {
        "libero_panda": EmbodimentTag.LIBERO_PANDA,
        "unitree_g1": EmbodimentTag.UNITREE_G1,
        "unitree_g1_sonic": EmbodimentTag.UNITREE_G1_SONIC,
        "simpler_env_google": EmbodimentTag.SIMPLER_ENV_GOOGLE,
        "simpler_env_widowx": EmbodimentTag.SIMPLER_ENV_WIDOWX,
        "new_embodiment": EmbodimentTag.NEW_EMBODIMENT,
        # IsaacLab custom embodiments → NEW_EMBODIMENT projector slot
        "isaaclab_franka": EmbodimentTag.NEW_EMBODIMENT,
        "isaaclab_g1_dex3": EmbodimentTag.NEW_EMBODIMENT,
    }
    if embodiment_tag_str in direct:
        return direct[embodiment_tag_str]
    raise ValueError(
        f"Invalid or unsupported embodiment tag for GR00T N1.7: {embodiment_tag_str!r}. "
        f"Supported: {sorted(direct.keys())}"
    )


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Instantiate a GR00T N1.7 model wrapped for RLinf PPO/SFT."""
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers import AutoConfig, AutoModel

    # Idempotent registration so repeated calls (e.g. rollout + actor) don't fail.
    try:
        AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
        AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
        print("[gr00t_1_7] Registered Gr00tN1d7 / Gr00tN1d7Config with transformers")
    except ValueError:
        # Already registered — fine.
        pass

    # Customize FSDP wrap policy to recognize N1.7-specific transformer blocks
    # (Qwen3 + DiT). Keep the heuristic from N1.6, just refresh the keyword
    # list with N1.7 layer names.
    import rlinf.hybrid_engines.fsdp.strategy.fsdp as fsdp_strategy

    if not hasattr(fsdp_strategy, "_is_gr00t_n17_patched"):
        orig_policy = fsdp_strategy.get_fsdp_wrap_policy

        def custom_fsdp_wrap_policy(
            module, config=None, is_lora=False, model_type=None
        ):
            import functools

            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            # N1.7 transformer blocks we want FSDP to slice on:
            #   - Qwen3DecoderLayer       (backbone)
            #   - BasicTransformerBlock   (DiT, in modules/dit.py)
            #   - SelfAttentionTransformer(VL self-attn)
            #   - Gr00tN1d7ActionHead     (top-level wrapper, do NOT wrap whole thing
            #     in FSDP — let the inner blocks be wrapped instead)
            #   - ValueHead               (our RL addition)
            target_keywords = [
                "DecoderLayer",
                "Qwen3",
                "BasicTransformerBlock",
                "SelfAttentionTransformer",
                "ValueHead",
                "Timestep",
            ]
            found_classes = set()
            for name, mod in module.named_modules():
                cname = mod.__class__.__name__
                if any(key in cname for key in target_keywords):
                    found_classes.add(mod.__class__)

            if found_classes:
                print(f"\n  FSDP Slicer (N1.7): {[c.__name__ for c in found_classes]}\n")
                return functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=found_classes
                )

            return orig_policy(module, config, is_lora, model_type)

        fsdp_strategy.get_fsdp_wrap_policy = custom_fsdp_wrap_policy
        fsdp_strategy._is_gr00t_n17_patched = True

    # Patch the EmbodimentTag enum so downstream gr00t imports resolve to our
    # extended tag set (kept for parity with N1.6; harmless if no extras).
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t_1_7.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t_1_7.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    Patcher.apply()

    from rlinf.models.embodiment.gr00t_1_7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t_1_7.utils import replace_dropout_with_identity

    emb_tag = _resolve_embodiment_tag(cfg.embodiment_tag)

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if cfg.get("model_type") == "gr00t_1_7_sft":
        from .gr00t_17_sft_model import GR00T_1_7_SFT_Model

        model_cls = GR00T_1_7_SFT_Model
    else:
        model_cls = GR00T_N1_7_ForRLActionPrediction

    # N1.7 does not accept tune_visual/tune_llm/local_model_path on __init__;
    # those values are in the saved config.json. We pass them through our RL
    # wrapper instead, which forwards what it needs and stashes the rest.
    obs_converter_type = OmegaConf.select(cfg, "obs_converter_type", default=None)
    processor_path = OmegaConf.select(cfg, "processor_path", default=None)

    # Load the real modality_configs the ckpt was trained with (see helper
    # docstring for why AutoProcessor's path is unreliable for SFT ckpts).
    modality_config = _load_ckpt_modality_config(model_path)
    if modality_config is not None:
        tags_in_cfg = list(modality_config.keys())
        print(
            f"[gr00t_1_7] Loaded ckpt modality_configs for tags={tags_in_cfg} "
            f"from {model_path / 'experiment_cfg' / 'config.yaml'}"
        )
    else:
        print(
            "[gr00t_1_7] WARNING: no experiment_cfg/config.yaml in ckpt; "
            "will fall back to inferring valid_action_dim from config.json "
            "(this is almost certainly wrong for custom embodiments)."
        )

    model = model_cls.from_pretrained(
        pretrained_model_name_or_path=str(model_path),
        local_model_path=str(model_path),
        torch_dtype=torch_dtype,
        embodiment_tag=emb_tag,
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=obs_converter_type,
        rl_head_config=cfg.rl_head_config,
        processor_path=processor_path,
        modality_config=modality_config,
        # Pass-through kwargs that PreTrainedModel.from_pretrained
        # forwards into Gr00tN1d7Config / Gr00tN1d7.__init__.
        trust_remote_code=True,
    )

    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head and hasattr(model.action_head, "value_head"):
        # Re-init value head after loading; HF init may leave it with NaNs.
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model
