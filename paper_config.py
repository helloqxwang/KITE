"""Map the public configuration names used in this repository to the runtime names used by the code.

Checkpoints, dataset payloads and asset identifiers use the runtime names. Only
configuration dictionaries are translated here.
"""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


# Unambiguous field names, including nested embodiment overrides.
PUBLIC_TO_RUNTIME = {
    "decoder_bank": "co_training",
    "synthetic_data": "probe",
    "surface_template": "contact_candidates",
    "per_embodiment": "per_manipulator",
    "embodiment_specific_projectors": "per_manipulator_action_io",
    "interaction_policy_model": "high_level_model",
    "interaction_policy_model_config": "high_level_model_config",
    "interaction_policy_config": "high_level_policy_config",
    "interaction_policy_config_path": "high_level_config_path",
    "kinematic_decoder_config": "action_policy_config",
    "kinematic_decoder_config_path": "low_level_config_path",
    "kinematic_decoder_checkpoint_path": "low_level_checkpoint_path",
    "intent_encoder_checkpoint": "latent_contact_encoder_checkpoint",
    "intent_encoder_device": "latent_encoder_device",
    "intent_encoder_batch_size": "latent_encoder_batch_size",
    "intent_encoder_prefer_ema": "latent_prefer_ema",
    "latent_intent_use_xyz_translation": "latent_use_xyz_translation",
    "skip_latent_intent": "skip_latent",
    "random_q_init": "random_q1",
    "latent_intent_interface": "contact_bottleneck",
    "embodiment_opacity": "hand_opacity",
    "initial_view": "initial_vis_mode",
}
TASK_NAMES = {
    "objects_placing": "object_suite_pick_place",
    "keyboard_pressing": "keyboard_press",
    "bottles_pumping": "pumpbottle_use",
    "nut_rotating": "screw_rotating",
}
CAPABILITY_NAMES = {
    "interaction_policy_evaluation": "high_level_policy_eval",
    "kinematic_decoder_rollout": "low_level_online_rollout",
    "interaction_intent_preprocessing": "high_level_preprocess",
    "objects_placing_synthesis": "object_suite_pick_place_synthesis",
    "objects_placing_retargeting": "retarget_object_suite_pick_place",
    "keyboard_pressing_synthesis": "keyboard_press_synthesis",
    "bottles_pumping_synthesis": "pumpbottle_use_synthesis",
}


def runtime_key(key: str, parent: tuple[str, ...] = ()) -> str:
    if parent == ("capabilities",):
        return CAPABILITY_NAMES.get(key, key)
    if key == "embodiments" and parent == ("co_training",):
        return "manipulators"
    if key == "embodiment" and parent in {("probe",), ("visualization",)}:
        return "robot_name"
    if key == "source_embodiment" and parent == ("preprocess",):
        return "manipulator_name"
    if key == "target_embodiment" and parent in {("online_rollout",), ("policy_eval",)}:
        return "low_level_manipulator"
    if key.startswith("q_init_"):
        return "q1_" + key[len("q_init_"):]
    if key.startswith("q_contact_"):
        return "q2_" + key[len("q_contact_"):]
    return PUBLIC_TO_RUNTIME.get(key, key)


def _convert(value: Any, parent: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            target = runtime_key(str(key), parent)
            converted = _convert(item, (*parent, target))
            if target in result and result[target] != converted:
                raise ValueError(f"Conflicting public and runtime configuration fields at {'.'.join((*parent, target))}")
            result[target] = converted
        return result
    if isinstance(value, list):
        return [_convert(item, parent) for item in value]
    if isinstance(value, str):
        if parent and parent[-1] == "supported_by":
            return CAPABILITY_NAMES.get(value, value)
        if parent == ("training", "supervision_mode") and value == "contact_relevant_joints":
            return "full_mix"
        if parent == ("visualization", "initial_vis_mode"):
            views = ["all_configurations", "goal_prediction", "contact_configuration_prediction",
                     "initial_goal_prediction", "contact_links", "initial_to_prediction"]
            if value in views:
                return views.index(value)
        if parent == ("task", "name") or (parent and parent[-1] == "task"):
            return TASK_NAMES.get(value, value)
        if parent and parent[-1] == "action_space":
            if value == "latent_intent":
                return "latent_contact" if parent[0] == "preprocess" else "latent"
            if value == "interaction_intent":
                return "contact_points"
    return value


def runtime_config(cfg: DictConfig | dict) -> DictConfig:
    """Resolve interpolations, then translate public field names to runtime names."""
    data = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    return OmegaConf.create(_convert(data))


def public_key(key: str, parent: tuple[str, ...] = ()) -> str:
    if key == "manipulators" and parent == ("co_training",):
        return "embodiments"
    if key in {"robot_name", "manipulator", "output_manipulator"} and parent in {("probe",), ("visualization",)}:
        return "embodiment"
    if key == "manipulator_name" and parent == ("preprocess",):
        return "source_embodiment"
    if key == "low_level_manipulator" and parent in {("online_rollout",), ("policy_eval",)}:
        return "target_embodiment"
    if key.startswith("q1_"):
        return "q_init_" + key[3:]
    if key.startswith("q2_"):
        return "q_contact_" + key[3:]
    return {v: k for k, v in PUBLIC_TO_RUNTIME.items()}.get(key, key)


def canonicalize_cli(argv=None):
    """Rewrite command-line overrides that use runtime field names to the public names."""
    import sys
    args = sys.argv if argv is None else argv
    for i, arg in enumerate(args):
        if "=" not in arg or arg.startswith("--"):
            continue
        key, value = arg.split("=", 1)
        prefix = key[:len(key)-len(key.lstrip("+~"))]
        keys = key.lstrip("+~").split(".")
        renamed, parent = [], ()
        for item in keys:
            renamed.append(public_key(item, parent))
            parent = (*parent, runtime_key(item, parent))
        if parent == ("task", "name") or parent[-1] == "task":
            value = {v: k for k, v in TASK_NAMES.items()}.get(value, value)
        if parent[-1] == "action_space" and value in {"latent", "latent_contact"}:
            value = "latent_intent"
        if parent == ("training", "supervision_mode") and value == "full_mix":
            value = "contact_relevant_joints"
        if parent == ("visualization", "initial_vis_mode") and value.isdigit() and int(value) < 6:
            value = ["all_configurations", "goal_prediction", "contact_configuration_prediction",
                     "initial_goal_prediction", "contact_links", "initial_to_prediction"][int(value)]
        args[i] = prefix + ".".join(renamed) + "=" + value
    return args


