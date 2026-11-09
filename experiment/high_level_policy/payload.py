from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


SCHEMA = "high_level_training_v2"
REQUIRED_DATA_KEYS = ("obs_point_cloud", "proprio", "action")
OBJECT_ONLY_WORLD_POINTCLOUD_CONTRACT = "object_mask_eroded1_depth_neighbor3cm_aabb5mm_world_v1"


def with_object_only_world_observation(meta: dict[str, Any]) -> dict[str, Any]:
    """Attach the object-suite point-cloud contract without changing XYZ values."""

    out = copy.deepcopy(dict(meta))
    observation = out.get("observation")
    if not isinstance(observation, dict):
        raise ValueError("meta.observation must be a dictionary")
    observation.update(
        {
            "frame": "world",
            "centering": "none",
            "point_selection": "object_mask_only",
            "mask_erosion_pixels": 1,
            "depth_neighbor_threshold_m": 0.03,
            "bounds_filter": "object_local_aabb",
            "bounds_filter_tolerance_m": 0.005,
            "object_only": True,
            "includes_context": False,
            "contract": OBJECT_ONLY_WORLD_POINTCLOUD_CONTRACT,
        }
    )
    return json_safe(out)


def validate_object_only_world_observation(
    meta: dict[str, Any],
    *,
    label: str = "payload",
) -> None:
    """Reject object-suite observations that do not declare the fixed-frame contract."""

    observation = dict(meta.get("observation", {}))
    expected = {
        "frame": "world",
        "centering": "none",
        "point_selection": "object_mask_only",
        "mask_erosion_pixels": 1,
        "depth_neighbor_threshold_m": 0.03,
        "bounds_filter": "object_local_aabb",
        "bounds_filter_tolerance_m": 0.005,
        "object_only": True,
        "includes_context": False,
        "contract": OBJECT_ONLY_WORLD_POINTCLOUD_CONTRACT,
    }
    mismatches = [
        f"{key}={observation.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if observation.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"{label} does not satisfy the object-only world-frame point-cloud contract: "
            + "; ".join(mismatches)
            + ". Regenerate the object-suite observation point clouds before preprocessing or training."
        )


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def as_float_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).detach().cpu()


def as_long_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long).detach().cpu()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_episode_ends(lengths: Sequence[int]) -> torch.Tensor:
    total = 0
    ends: list[int] = []
    for length in lengths:
        length_int = int(length)
        if length_int <= 0:
            raise ValueError(f"Episode lengths must be positive, got {length_int}")
        total += length_int
        ends.append(total)
    return torch.as_tensor(ends, dtype=torch.long)


def previous_action_proprio(action: Any, episode_ends: Any) -> torch.Tensor:
    action_t = as_float_tensor(action)
    ends = as_long_tensor(episode_ends).view(-1)
    proprio = torch.empty_like(action_t)
    start = 0
    for end_t in ends:
        end = int(end_t.item())
        if end <= start:
            raise ValueError(f"episode_ends must be strictly increasing, got end={end} after start={start}")
        proprio[start] = action_t[start]
        if end - start > 1:
            proprio[start + 1 : end] = action_t[start : end - 1]
        start = end
    if start != int(action_t.shape[0]):
        raise ValueError(f"episode_ends[-1]={start} does not match action frames={action_t.shape[0]}")
    return proprio


def concat_single_manipulator_action(base_pos: Any, base_quat_wxyz: Any, joint_target: Any) -> torch.Tensor:
    pos = as_float_tensor(base_pos).reshape(-1, 3)
    quat = as_float_tensor(base_quat_wxyz).reshape(-1, 4)
    joint = as_float_tensor(joint_target)
    if joint.ndim == 1:
        joint = joint[:, None]
    joint = joint.reshape(joint.shape[0], -1)
    frame_counts = {int(pos.shape[0]), int(quat.shape[0]), int(joint.shape[0])}
    if len(frame_counts) != 1:
        raise ValueError(f"Single-manipulator components have different frame counts: {sorted(frame_counts)}")
    return torch.cat([pos, quat, joint], dim=-1)


def single_manipulator_action_space(*, joint_key: str, joint_dim: int) -> dict[str, Any]:
    joint_dim = int(joint_dim)
    return {
        "type": "single_manipulator",
        "dim": 7 + joint_dim,
        "component_slices": {
            "base_pos": [0, 3],
            "base_quat_wxyz": [3, 7],
            "joint_target": [7, 7 + joint_dim],
        },
        "joint_key": str(joint_key),
        "representation": f"concat(base_pos(3), base_quat_wxyz(4), {joint_key}({joint_dim}))",
    }


def latent_contact_action_space(
    *,
    dim: int,
    latent_encoder_dim: int | None = None,
    latent_use_xyz_translation: bool | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "latent_contact",
        "dim": int(dim),
        "component_slices": {"latent_action": [0, int(dim)]},
        "representation": "latent_action",
    }
    if latent_encoder_dim is not None:
        out["latent_encoder_dim"] = int(latent_encoder_dim)
    if latent_use_xyz_translation is not None:
        out["latent_use_xyz_translation"] = bool(latent_use_xyz_translation)
    return out


def contact_points_action_space(*, slot_count: int, point_dim: int = 6) -> dict[str, Any]:
    slot_count = int(slot_count)
    point_dim = int(point_dim)
    dim = slot_count * point_dim
    return {
        "type": "contact_points",
        "dim": dim,
        "component_slices": {"contact_points": [0, dim]},
        "slot_count": slot_count,
        "point_dim": point_dim,
        "representation": f"flatten(contact_points({slot_count}, {point_dim}))",
        "frame": "low_level_shifted",
    }


def build_common_meta(
    *,
    source: dict[str, Any],
    task: str,
    embodiment: str,
    dt: float | None,
    fps: float | None,
    observation_point_count: int,
    observation_frame: str,
    observation_dtype: str,
    action_space: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(extra or {})
    meta.update(
        {
            "schema": SCHEMA,
            "source": copy.deepcopy(source),
            "task": str(task),
            "embodiment": str(embodiment),
            "time": {
                "dt": None if dt is None else float(dt),
                "fps": None if fps is None else float(fps),
            },
            "observation": {
                "point_count": int(observation_point_count),
                "frame": str(observation_frame),
                "dtype": str(observation_dtype),
            },
            "action_space": copy.deepcopy(action_space),
        }
    )
    return json_safe(meta)


def build_training_payload(
    *,
    obs_point_cloud: Any,
    action: Any,
    episode_ends: Any,
    episode_meta: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    proprio: Any | None = None,
    extra_data: dict[str, Any] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    obs = as_float_tensor(obs_point_cloud)
    action_t = as_float_tensor(action)
    ends = as_long_tensor(episode_ends).view(-1)
    proprio_t = previous_action_proprio(action_t, ends) if proprio is None else as_float_tensor(proprio)
    data = {str(key): torch.as_tensor(value).detach().cpu() for key, value in dict(extra_data or {}).items()}
    data["obs_point_cloud"] = obs
    data["proprio"] = proprio_t
    data["action"] = action_t
    payload = {
        "meta": json_safe(dict(meta)),
        "data": data,
        "episode_ends": ends,
        "episode_meta": [json_safe(dict(item)) for item in episode_meta],
    }
    if validate:
        validate_high_level_payload(payload)
    return payload


def validate_high_level_payload(payload: dict[str, Any], *, check_proprio: bool = True) -> None:
    if not isinstance(payload, dict):
        raise TypeError("High-level payload must be a dict")
    required = {"meta", "data", "episode_ends", "episode_meta"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"High-level payload is missing top-level keys: {sorted(missing)}")
    meta = dict(payload["meta"])
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"Expected meta.schema={SCHEMA!r}, got {meta.get('schema')!r}")
    data = dict(payload["data"])
    missing_data = set(REQUIRED_DATA_KEYS).difference(data)
    if missing_data:
        raise KeyError(f"High-level payload data is missing required keys: {sorted(missing_data)}")

    obs = as_float_tensor(data["obs_point_cloud"])
    proprio = as_float_tensor(data["proprio"])
    action = as_float_tensor(data["action"])
    episode_ends = as_long_tensor(payload["episode_ends"]).view(-1)
    if obs.ndim != 3 or int(obs.shape[-1]) != 3:
        raise ValueError(f"obs_point_cloud must have shape (T, N, 3), got {tuple(obs.shape)}")
    frame_count = int(obs.shape[0])
    if int(action.shape[0]) != frame_count or int(proprio.shape[0]) != frame_count:
        raise ValueError(
            "obs_point_cloud, proprio, and action must share the same leading frame count; "
            f"got obs={obs.shape[0]} proprio={proprio.shape[0]} action={action.shape[0]}"
        )
    if tuple(proprio.shape) != tuple(action.shape):
        raise ValueError(f"proprio shape {tuple(proprio.shape)} must match action shape {tuple(action.shape)}")
    if episode_ends.ndim != 1 or int(episode_ends.numel()) == 0:
        raise ValueError("episode_ends must be a non-empty 1D tensor/array")
    if int(episode_ends[-1].item()) != frame_count:
        raise ValueError(f"episode_ends[-1]={int(episode_ends[-1].item())} does not match frames={frame_count}")
    diffs = episode_ends.diff()
    if int(episode_ends[0].item()) <= 0 or (diffs.numel() and bool(torch.any(diffs <= 0))):
        raise ValueError("episode_ends must be strictly increasing positive frame boundaries")
    if len(list(payload["episode_meta"])) != int(episode_ends.numel()):
        raise ValueError("episode_meta length must match episode_ends length")
    for key, tensor in {"obs_point_cloud": obs, "proprio": proprio, "action": action}.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{key} contains non-finite values")
    action_space = meta.get("action_space", {})
    if isinstance(action_space, dict) and "dim" in action_space:
        flat_action_dim = int(np.prod(tuple(action.shape[1:])))
        if int(action_space["dim"]) != flat_action_dim:
            raise ValueError(
                f"meta.action_space.dim={action_space['dim']} does not match flattened action dim={flat_action_dim}"
            )
    if check_proprio:
        expected = previous_action_proprio(action, episode_ends)
        if not bool(torch.allclose(proprio, expected, atol=1.0e-5, rtol=1.0e-5)):
            raise ValueError("proprio must equal previous-step action within each episode")


def save_high_level_payload(payload: dict[str, Any], path: str | Path, *, validate: bool = True) -> Path:
    if validate:
        validate_high_level_payload(payload)
    out = Path(path).expanduser().resolve()
    if out.suffix == "":
        out = out.with_suffix(".pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out))
    return out


def load_high_level_payload(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix != ".pt":
        raise ValueError(f"High-level training payloads are .pt only; got {resolved}")
    payload = torch.load(str(resolved), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload at {path}")
    required = {"meta", "data", "episode_ends", "episode_meta"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Missing keys in payload {path}: {sorted(missing)}")
    if validate and dict(payload.get("meta", {})).get("schema") == SCHEMA:
        validate_high_level_payload(payload)
    return payload


def split_by_episode(value: Any, episode_ends: Any) -> list[np.ndarray]:
    arr = tensor_to_numpy(value)
    ends = tensor_to_numpy(episode_ends).astype(np.int64).reshape(-1)
    out: list[np.ndarray] = []
    start = 0
    for end in ends:
        out.append(np.asarray(arr[start : int(end)]))
        start = int(end)
    return out


def stack_or_object(episodes: Iterable[Any], *, dtype: Any | None = None) -> np.ndarray:
    arrays = [np.asarray(ep, dtype=dtype) if dtype is not None else np.asarray(ep) for ep in episodes]
    try:
        return np.stack(arrays, axis=0)
    except ValueError:
        return np.asarray(arrays, dtype=object)


def single_manipulator_payload_to_replay_source(payload: dict[str, Any], *, source_path: str | Path | None = None) -> dict[str, np.ndarray]:
    meta = dict(payload.get("meta", {}))
    action_space = dict(meta.get("action_space", {}))
    action_space_type = str(action_space.get("type", "")).strip()
    if action_space_type == "robot_model_q":
        return _robot_model_q_payload_to_replay_source(payload, source_path=source_path)
    if action_space_type != "single_manipulator":
        raise ValueError(
            "Replay of latent-contact .pt payloads is not supported by tabletop replay scripts; "
            "route latent-contact payloads through the high-low latent pipeline instead."
        )
    validate_high_level_payload(payload)
    data = dict(payload["data"])
    episode_ends = payload["episode_ends"]
    action = tensor_to_numpy(data["action"]).reshape(tensor_to_numpy(data["action"]).shape[0], -1)
    slices = dict(action_space.get("component_slices", {}))

    def component(name: str) -> np.ndarray:
        if name not in slices:
            raise KeyError(f"single_manipulator action_space is missing component slice {name!r}")
        start, end = [int(x) for x in slices[name]]
        return action[:, start:end].astype(np.float32, copy=False)

    joint_key = str(action_space.get("joint_key") or _default_joint_key(str(meta.get("embodiment", ""))))
    replay: dict[str, np.ndarray] = {
        "schema_version": np.asarray(2, dtype=np.int32),
        "embodiment": np.asarray(str(meta.get("embodiment", ""))),
        "mode": np.asarray(str(meta.get("mode", "single_right"))),
        "task": np.asarray(str(meta.get("task", ""))),
        "action_space_type": np.asarray(action_space_type),
        "action_ee_pos": stack_or_object(split_by_episode(component("base_pos"), episode_ends), dtype=np.float32),
        "action_ee_quat_wxyz": stack_or_object(split_by_episode(component("base_quat_wxyz"), episode_ends), dtype=np.float32),
        joint_key: stack_or_object(split_by_episode(component("joint_target"), episode_ends), dtype=np.float32),
        "action_effector_ctrl": stack_or_object(split_by_episode(component("joint_target"), episode_ends), dtype=np.float32),
    }
    if source_path is not None:
        replay["source_training_pt_path"] = np.asarray(str(Path(source_path).expanduser().resolve()))

    for scalar_key in (
        "backend",
        "frame_skip",
        "dt",
        "task_objective",
        "gravity_compensation",
        "target_pointcloud_num",
        "exclude_robot_pointcloud",
        "exclude_table_pointcloud",
    ):
        if scalar_key in meta:
            replay[scalar_key] = np.asarray(meta[scalar_key])
    time_meta = meta.get("time")
    if isinstance(time_meta, dict) and "dt" not in replay and time_meta.get("dt") is not None:
        replay["dt"] = np.asarray(float(time_meta["dt"]), dtype=np.float32)

    episode_count = len(split_by_episode(data["action"], episode_ends))
    frame_count = int(tensor_to_numpy(data["action"]).shape[0])
    for key in (
        "cube_pose",
        "object_pose",
        "object_index",
        "bin_pose",
        "placement_margin",
        "task_success",
        "phase",
        "obs_point_cloud",
        "pointcloud_occluded",
        "cube_side_length",
        "cube_height",
        "cube_yaw",
        "executed_qpos",
        "executed_qvel",
        "executed_ctrl",
        "contact_flags",
        "source_contact_flags",
        "trajectory_seed",
        "attempt_index",
        "lift_success",
        "max_cube_z",
        "source_trajectory_index",
        "source_trajectory_seed",
        "source_attempt_index",
        "cube_x_range",
        "cube_y_range",
        "cube_side_length_range",
        "cube_height_ratio",
        "cube_height_range",
        "cube_yaw_range_deg",
        "pointcloud_workspace_bounds",
        "object_manifest",
        "bin_center_xy",
        "bin_inner_size",
        "group_names",
        "qpos_names",
        "ctrl_names",
    ):
        if key in data:
            replay[key] = _episode_or_raw_array(data[key], episode_ends, episode_count, frame_count)
        elif key in meta:
            replay[key] = np.asarray(meta[key])
    if "cube_pose" not in replay and "object_pose" in replay:
        replay["cube_pose"] = np.asarray(replay["object_pose"])
    episode_meta = list(payload.get("episode_meta", []))
    if len(episode_meta) == episode_count and all(isinstance(item, dict) for item in episode_meta):
        if all(item.get("object_id") is not None for item in episode_meta):
            replay["object_id"] = np.asarray([str(item["object_id"]) for item in episode_meta])
    return replay


def _robot_model_q_payload_to_replay_source(
    payload: dict[str, Any],
    *,
    source_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    meta = dict(payload.get("meta", {}))
    action_space = dict(meta.get("action_space", {}))
    validate_high_level_payload(payload)
    data = dict(payload["data"])
    episode_ends = payload["episode_ends"]
    action = tensor_to_numpy(data["action"]).reshape(tensor_to_numpy(data["action"]).shape[0], -1)
    slices = dict(action_space.get("component_slices", {}))

    def component(name: str) -> np.ndarray:
        if name not in slices:
            raise KeyError(f"robot_model_q action_space is missing component slice {name!r}")
        start, end = [int(x) for x in slices[name]]
        return action[:, start:end].astype(np.float32, copy=False)

    base_xyz = component("base_xyz")
    base_euler = component("base_euler_xyz")
    hand_q = component("hand_q")
    base_quat = _euler_xyz_to_quat_wxyz(base_euler)
    replay: dict[str, np.ndarray] = {
        "schema_version": np.asarray(2, dtype=np.int32),
        "embodiment": np.asarray(str(meta.get("embodiment", ""))),
        "mode": np.asarray(str(meta.get("mode", "floating_right"))),
        "task": np.asarray(str(meta.get("task", ""))),
        "action_space_type": np.asarray("robot_model_q"),
        "action_ee_pos": stack_or_object(split_by_episode(base_xyz, episode_ends), dtype=np.float32),
        "action_ee_euler_xyz": stack_or_object(split_by_episode(base_euler, episode_ends), dtype=np.float32),
        "action_ee_quat_wxyz": stack_or_object(split_by_episode(base_quat, episode_ends), dtype=np.float32),
        "action_arm_q": stack_or_object(split_by_episode(np.concatenate([base_xyz, base_euler], axis=-1), episode_ends), dtype=np.float32),
        "action_hand_q": stack_or_object(split_by_episode(hand_q, episode_ends), dtype=np.float32),
        "action_effector_ctrl": stack_or_object(split_by_episode(hand_q, episode_ends), dtype=np.float32),
        "manipulator_action": stack_or_object(split_by_episode(action, episode_ends), dtype=np.float32),
    }
    if source_path is not None:
        replay["source_training_pt_path"] = np.asarray(str(Path(source_path).expanduser().resolve()))

    for scalar_key in (
        "backend",
        "frame_skip",
        "dt",
        "gravity_compensation",
        "target_pointcloud_num",
        "exclude_robot_pointcloud",
        "exclude_table_pointcloud",
        "manipulator_name",
        "robot_name",
        "low_level_manipulator",
    ):
        if scalar_key in meta:
            replay[scalar_key] = np.asarray(meta[scalar_key])
    time_meta = meta.get("time")
    if isinstance(time_meta, dict) and "dt" not in replay and time_meta.get("dt") is not None:
        replay["dt"] = np.asarray(float(time_meta["dt"]), dtype=np.float32)

    episode_count = len(split_by_episode(data["action"], episode_ends))
    frame_count = int(tensor_to_numpy(data["action"]).shape[0])
    for key in (
        "cube_pose",
        "object_pose",
        "object_index",
        "bin_pose",
        "placement_margin",
        "task_success",
        "phase",
        "obs_point_cloud",
        "cube_side_length",
        "cube_height",
        "cube_yaw",
        "executed_qpos",
        "executed_qvel",
        "executed_ctrl",
        "target_qpos",
        "target_ctrl",
        "contact_flags",
        "source_contact_flags",
        "trajectory_seed",
        "attempt_index",
        "lift_success",
        "max_cube_z",
        "source_trajectory_index",
        "source_trajectory_seed",
        "source_attempt_index",
        "cube_x_range",
        "cube_y_range",
        "cube_side_length_range",
        "cube_height_ratio",
        "cube_height_range",
        "cube_yaw_range_deg",
        "pointcloud_workspace_bounds",
        "object_manifest",
        "bin_center_xy",
        "bin_inner_size",
        "group_names",
        "joint_orders",
        "qpos_names",
        "ctrl_names",
        "joint_lower",
        "joint_upper",
    ):
        if key in data:
            replay[key] = _episode_or_raw_array(data[key], episode_ends, episode_count, frame_count)
        elif key in meta:
            replay[key] = np.asarray(meta[key])
        elif key == "joint_orders" and "joint_orders" in action_space:
            replay[key] = np.asarray(action_space["joint_orders"])
    if "cube_pose" not in replay and "object_pose" in replay:
        replay["cube_pose"] = np.asarray(replay["object_pose"])
    episode_meta = list(payload.get("episode_meta", []))
    if len(episode_meta) == episode_count and all(isinstance(item, dict) for item in episode_meta):
        if all(item.get("object_id") is not None for item in episode_meta):
            replay["object_id"] = np.asarray([str(item["object_id"]) for item in episode_meta])
    return replay


def _euler_xyz_to_quat_wxyz(euler_xyz: Any) -> np.ndarray:
    euler = np.asarray(euler_xyz, dtype=np.float32).reshape(-1, 3)
    quat_xyzw = R.from_euler("XYZ", euler).as_quat().astype(np.float32)
    return np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)


def _episode_or_raw_array(value: Any, episode_ends: Any, episode_count: int, frame_count: int) -> np.ndarray:
    arr = tensor_to_numpy(value)
    if arr.ndim >= 1 and int(arr.shape[0]) == int(frame_count):
        return stack_or_object(split_by_episode(arr, episode_ends), dtype=arr.dtype)
    if arr.ndim >= 1 and int(arr.shape[0]) == int(episode_count):
        return np.asarray(arr)
    return np.asarray(arr)


def _default_joint_key(embodiment: str) -> str:
    if "wuji" in str(embodiment).lower():
        return "action_hand_q"
    return "action_gripper"
